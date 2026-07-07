# arXiv:2406.05338v6[cs.CV]22Oct2024

## MOTIONCLONE: TRAINING-FREE MOTION CLONING FOR CONTROLLABLE VIDEO GENERATION

Pengyang Ling1,4∗ Jiazi Bu2,4∗ Pan Zhang4† Xiaoyi Dong4 Yuhang Zang4 Tong Wu3 Huaian Chen1 Jiaqi Wang4 Yi Jin1† 1University of Science and Technology of China 2Shanghai Jiao Tong University 3The Chinese University of Hong Kong 4Shanghai AI Laboratory https://github.com/LPengYang/MotionClone

ABSTRACT

Motion-based controllable video generation offers the potential for creating captivating visual content. Existing methods typically necessitate model training to encode particular motion cues or incorporate fine-tuning to inject certain motion patterns, resulting in limited flexibility and generalization. In this work, we propose MotionClone, a training-free framework that enables motion cloning from reference videos to versatile motion-controlled video generation, including textto-video and image-to-video. Based on the observation that the dominant components in temporal-attention maps drive motion synthesis, while the rest mainly capture noisy or very subtle motions, MotionClone utilizes sparse temporal attention weights as motion representations for motion guidance, facilitating diverse motion transfer across varying scenarios. Meanwhile, MotionClone allows for the direct extraction of motion representation through a single denoising step, bypassing the cumbersome inversion processes and thus promoting both efficiency and flexibility. Extensive experiments demonstrate that MotionClone exhibits proficiency in both global camera motion and local object motion, with notable superiority in terms of motion fidelity, textual alignment, and temporal consistency.

1 INTRODUCTION

Video generations that align with human intentions and produce high-quality outputs has recently attracted significant attention, particularly with the rise of mainstream text-to-video (Guo et al., 2023b; Blattmann et al., 2023b; Chen et al., 2024) and image-to-video (Guo et al., 2023a; Blattmann et al., 2023a; Dai et al., 2023) diffusion models. Despite the substantial progress witnessed in conditional image generation, the domain of video generation presents unique challenges, primarily due to the complexities introduced by motion synthesis. Incorporating additional motion control not only mitigates the ambiguity inherent in video synthesis for superior motion modeling but also enhances the manipulability of the synthesized content for customized creations.

In the realm of video generation that is steered by motion cues, pioneering methodologies can be generally classified into two principal strategies: one that leverages the dense depth or sketch of reference videos (Wang et al., 2024; Jeong & Ye, 2023; Guo et al., 2023a), and another that relies on motion trajectories (Wang et al., 2023b; Yin et al., 2023; Niu et al., 2024). The former methodology typically involves the integration of a pre-trained model to extract motion cues at the pixel level. Despite achieving highly aligned motion, these dense motion cues can be intricately entangled with the structural elements of the reference videos, impeding their transferability in novel scenarios. The latter trajectory-based methodology, by contrast, provides a more user-friendly approach for capturing broader object movements but struggles to delineate finer, localized motions such as head turns or hand raises. Additionally, both methodologies typically entail model training to encode particular motion cues, implying suboptimal generation when applied outside the trained domain. Such limitation is also observed in approaches relying on fine-tuning (Jeong et al., 2023; Zhao et al., 2023), which aim to fit the motion patterns of certain videos.

*Equal contribution. † Corresponding author.

Object motion cloning Camera motion cloning

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

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Duck, swims in the river Robot, runs in the street

Dog, sitting on the grass

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

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Lion, tuns its head on grass

Pig, drinks water on beach Building, around with flowers

- Figure 1: Motion cloning in varying scenarios. Given a reference video, MotionClone can clone the contained motion into novel scenarios with excellent prompt-following ability, without motionspecific fine-tuning. The red arrows indicate the motion direction.

In this work, we introduce MotionClone, a novel training-free framework designed to clone motions from reference videos for controllable video generation. Diverging from traditional approaches involving tailored training or fine-tuning, MotionClone employs the commonly used temporalattention mechanism within video generation models to capture the motion in reference videos. This strategy effectively renders detailed motion while concurrently preserving minimal interdependencies with the structural components of the reference video, offering flexible motion cloning in varying scenarios, as shown in Fig. 1. To be specific, it is observed that the dominant components in temporal-attention weights significantly drive motion synthesis, while the rest mainly refer to noisy or very subtle motions. When the whole temporal-attention is applied uniformly across the model, the majority of temporal-attention weights can overshadow the motion guidance, consequently resulting in the suppression of the primary motion. Therefore, we propose to leverage the principal components of the temporal-attention weights as motion representation, which serves as motion guidance that overlooks noisy or less significant motions and concentrates on the primary motion, thus substantially enhancing the fidelity of motion cloning. Moreover, it has been demonstrated that the motion representation extracted from a certain denoising step holds effective guidance across all time steps, offering high efficiency without the burden of cumbersome video inversion. Furthermore, MotionClone is compatible with a range of video generation tasks, including text-to-video (T2V) and image-to-video (I2V), highlighting its versatility and broad applicability.

In summary, (1) we propose MotionClone, a novel motion-guided video generation framework that enables training-free motion cloning from given reference videos; (2) we design a primary motion control strategy to perform substantial motion guidance over sparse temporal attention map, allowing for efficient motion transfer across scenarios; (3) we validate the effectiveness and versatility of MotionClone in various video generation tasks, in which extensive experiments demonstrate its proficiency in both global camera motion and local object action, with notable superiority in terms of motion fidelity, text alignment, and temporal consistency.

- 2 RELATED WORK

- 2.1 TEXT-TO-VIDEO DIFFUSION MODELS

Equipped with sophisticated text encoders (Radford et al., 2021; Zhang et al., 2024), a great breakthrough has been achieved in the realm of text-to-image (T2I) generation (Gu et al., 2022; Nichol et al., 2021; Rombach et al., 2022; Podell et al., 2023), which sparks the enthusiasm for advanced text-to-video (T2V) models (Blattmann et al., 2023b; Wang et al., 2023a; Chen et al., 2023a; 2024; Guo et al., 2023b). Notably, VideoLDM (Blattmann et al., 2023b) introduces a motion module that

utilizes 3D convolutions and temporal attention to capture frame-to-frame correlations. In a novel approach, AnimateDiff (Guo et al., 2023b) enhances a pre-trained T2I diffusion model with motion modeling capabilities. This is achieved by fine-tuning a series of specialized temporal attention layers on extensive video datasets, allowing for a harmonious fusion with the original T2I generation process. To address the challenge of data scarcity, VideoCraft2 (Chen et al., 2024) suggests an innovative strategy of learning motion from low-quality videos (Bain et al., 2021) while simultaneously learning appearance from high-quality images (Sun et al., 2024). Despite these advancements, there remains a significant disparity in the quality of generated content between the available T2V models and their sophisticated T2I counterparts, primarily due to the intricate nature of diverse motions and the limited availability of high-quality video data. In this work, a motion guidance strategy is developed, which ingeniously incorporates motion cues from given videos to ease the challenges of motion modeling, yielding more realistic and coherent video sequences, without model fine-tuning.

- 2.2 CONTROLLABLE VIDEO GENERATION

Building on the success of controllable image generation through the integration of additional conditions (Zhang et al., 2023; Kim et al., 2023; Li et al., 2023; Qin et al., 2023; Huang et al., 2023), a multitude of studies (Chen et al., 2023a; Yin et al., 2023; Dai et al., 2023; Ma et al., 2024; Blattmann

- et al., 2023a) have endeavored to introduce diverse control signals for versatile video generation. These include control over the first video frame (Chen et al., 2023a), motion trajectory (Yin et al., 2023), motion region (Dai et al., 2023), and motion object (Ma et al., 2024). Furthermore, in pursuit of high-quality video customization, several studies delve into reference-based video generation, leveraging the motion from an existing real video to direct the creation of new video content. A straightforward solution developed in Wang et al. (2024); Esser et al. (2023); Xing et al. (2024), involves the direct integration of frame-wise depth maps or canny maps to regularize motion. However, this approach inadvertently introduces motion-independent features, such as structures in static areas, which can disrupt the alignment of the resulting video appearance with new text. To address this issue, motion-specific fine-tuning frameworks, as explored in (Zhao et al., 2023; Jeong et al., 2023), have been developed to extract a distinct motion pattern from a single video or a collection of videos with identical motion. While holding promise, these methods are subject to complex training processes and potential model degradation. To address this, we present a novel motion cloning scheme, which extracts temporal correlations from existing videos as explicit motion clues to guide the generation of new video content, providing a plug-and-play motion customization solution.

- 2.3 ATTENTION FEATURE CONTROL

Attention mechanisms have been confirmed as vital for high-quality content generation. Prompt2Prompt (Hertz et al., 2022) illustrates that cross-attention maps are instrumental in dictating the spatial layout of synthesized images. This observation subsequently motivates serious work in semantic preservation (Chefer et al., 2023), multi-object generation (Ma et al., 2023; Xiao et al., 2023), and video editing (Liu et al., 2023). AnyV2V (Ku et al., 2024) reveals dense injection of both CNN and attention features facilitates improved alignment with source videos in video editing. FreeControl (Mo et al., 2023) highlights that the feature space within self-attention layers encodes structural image information, facilitating reference-based image generation. While previous methods mainly concentrate on spatial attention layers, our work uncovers the untapped potential of temporal attention layers for effective motion guidance, enabling flexible motion cloning.

- 3 MOTIONCLONE

In this section, we first introduce video diffusion models and temporal attention mechanisms. Then we explore the potential of primary control over sparse temporal attention maps for substantial motion guidance. Subsequently, we elaborate on the proposed MotionClone framework, which performs motion cloning by deliberately manipulating temporal attention weights.

- 3.1 PRELIMINARIES

Diffusion sampling. Following pioneering work (Rombach et al., 2022), video diffusion models encode a input video x into latent representation z0 = E(x) by using a pre-trained encoder E(·). To

Reference

Prompt: A cat plays in the forest

Reference

Prompt: A tank runs in the desert

video

Generated videos from same initial noise

video

Generated videos from same initial noise

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

w/o control Plain control Primary control w/o control Plain control Primary control

- Figure 2: Comparision of plain control and primary control over temporal attention map. Leveraging temporal attention maps derived from reference videos to guide video generation. Plain control refers to a rudimentary approach whereby all weights are uniformly applied. Primary control only applies constraint to the sparse temporal attention map.

enable video distribution learning, diffusion model ϵθ is encouraged to estimate noise component ϵ from noised latent zt that follows time-dependent scheduler (Ho et al., 2020), i.e.,

L(θ) = EE(x),ϵ∈N(0,1),t∼U(1,T) ∥ϵ − ϵθ(zt,c,t)∥22 , (1) where t is the time step, and c is the condition signal such as text or image. In the inference phase, the sampling process commences with a standard Gaussian noise. The sampling trajectory, however, can be adjusted by incorporating guidance for extra controllability. This is typically achieved by customized energy function g(zt,y,t) with label y indicating guidance direction, i.e.,

#### ϵˆθ = ϵθ(zt,c,t) + s(ϵθ(zt,c,t) − ϵθ(zt,ϕ,t)) − λ√1 − α¯t∇zt

g(zt,y,t), (2) where ϵθ(zt,ϕ,t) is the classifier-free guidance (Ho & Salimans, 2022), ϕ denotes the unconditional √class1 −identifierα¯t is used to convert the gradient of energy function(such as null text for textual condition), s andg(·)λinto noise prediction, in whichare guidance weights, and the√αterm¯t is the hyperparameter of noise schedule, i.e., zt = √α¯tz0 + √1 − α¯tϵ. During sampling, the gradient generated by energy function g(·) indicates the direction toward generation target.

Temporal attention. In video motion synthesis, temporal attention mechanism is broadly applied to establish correlation across frames. Given a f-frame video feature fin ∈ Rb×f×c×h×w where b denotes batch size, c denotes channel number, h and w are spatial resolution, temporal attention first reshapes it into 3D tensor f

′

in ∈ R(b×h×w)×f×c by merging the spatial dimensions into the batch size. Subsequently, it executes self-attention along the frame axis, which can be expressed as:

′

′

′

in)), (3) where Q(·), K(·), and V (·) are projection layers. Correspondingly, the attention map is labeled as A ∈ R(b×h×w)×f×f, which captures the temporal relation for each pixel feature.

fout = Attention(Q(f

in),K(f

in),V (f

- 3.2 OBSERVATION

Since temporal attention mechanism governs the motion in the generated video, videos with similar temporal attention maps are expected to share similar motion characteristics. To investigate this hypothesis, we manipulate the sampling trajectory by aligning the temporal attention maps of the generated video with those from a reference video. As depicted in Fig. 2, simply enforcing alignment on the entire temporal attention map (plain control) can only partly restore coarse motion patterns of reference videos, such as the gait of a cat and the directional movement of a tank, demonstrating limited motion alignment. We postulate that this is because not all temporal attention weights are essential for motion synthesis, with some reflecting scene-specific noise or extremely small motions. Indiscriminate alignment with the entire temporal attention maps dilutes critical motion guidance, resulting in suboptimal motion cloning in novel scenarios. As evidence, primary control over the sparse temporal attention map significantly boosts motion alignment, which can be attributed to the emphasis on motion-related cues and the disregard of motion-irrelevant factors.

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

frame-1 frame-6 frame-11 frame-16 =800 =600 =400 =200

### Figure 3: Visualization of motion representation. The mean intensity of Lt

α in frame axis from “up blocks.1” (resized to the represented resolution) indicates the area and magnitude of motion. This performance encounters decline in complex “head turning” scenario when tα = 800.

- 3.3 MOTION REPRESENTATION

Given a reference video, the corresponding temporal attention map in t denoising step is denoted as Atref ∈ R(1×h×w)×f×f, which satisfies fj=1[Atref]p,i,j = 1. The value of [Atref]p,i,j reflects the relation between i frame and j frame in position p, and a larger value of [Atref]p,i,j implies a stronger correlation. The motion guidance over temporal attention maps, depicted by energy function g(·), is modeled as:

g = Mt · (Atref − Atgen) 22 , (4) where Mt is the temporal mask for sparse constraint, and Atgen is the temporal attention weights of generated videos in time step t. Essentially, Eq. 4 promotes motion cloning by forcing Atgen close

to Atref, while Mt determines the sparsity of constraint, time-dependence Atref,Mt constitute the motion guidance. Particularly, Mt ≡ 1 refers to the “plain control” that exhibits limited motion transfer capability as illustrated in Fig. 2. Since the value of Atref is indicative of the strength of inter-frame correlation, we propose to obtain the sparse temporal mask according to the rank of Atref value in the temporal axis, i.e.,

1, if [Atref]p,i,j ∈ Ωtp,i 0, otherwise,

Mtp,i,j :=

(5)

where Ωtp,i = {τ1,τ2,...,τk} is the subset of index that comprising the top k values in attention map Atref along the temporal axis j, and k is a hyper-parameter. Particularly, in the case where k = 1, motion guidance focuses solely on the highest activation for each spatial location. Supervised by Eq. 5, motion guidance in Eq. 4 encourages the sparse alignment with the primary component in Atref while ensures spatially even constraint, facilitating a stable and reliable motion transfer.

Despite enabling effective motion cloning, the above scheme has obvious flaws: i) for real reference videos, laborious and time-consuming inversion operation is required for preparing Atref; and ii) the considerable size of the time-dependent Atref,Mt poses significant challenges for largescale preparation and efficient deployment. Fortunately, it is noted that the representation from certain denoising step can provide substantial and consistent motion guidance in generation process. Mathematically, motion guidance in Eq. 4 can be converted into

2 2

2 2 , (6)

· (At

g = Mt

ref − Atgen)

= Lt

− Mt

· Atgen

α

α

α

α

· At

where tα denotes certain time step, and Lt

ref. For given reference videos, the corresponding motion representation is denoted as Ht

= Mt

α

α

α

}, comprising two elements that are both highly temporally sparse. For real reference videos, their Ht

= {Lt

,Mt

α

α

α

α can be easily derived by directly

adding noise to shift them into the noised latent of tα time step, followed by a single denoising step. This straightforward strategy, impressively, proves to be remarkably effective. As shown in Fig. 3,

over a larger range of time steps (tα from 200 to 600), the mean intensity of Ht

α effectively highlights the region and magnitude of motion. However, it is also observed that Ht

α in early denoising stage (tα = 800) shows some discrepancies with the “head-turning” motion. This can be attributed

Reference video Motion representation

Prompt: null-text

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

Noise adding

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

Ⅰ

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

Primary Control

[Figure 584]

[Figure 585]

[Figure 586]

Time step decrease

Motion guidance stage

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

Motion guidance

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

Prompt: Tiger, raises its head

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

Random noise

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

Ⅰ

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

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

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

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

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

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

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

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

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

[Figure 1024]

[Figure 1025]

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

[Figure 1039]

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

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

[Figure 1085]

[Figure 1086]

[Figure 1087]

[Figure 1088]

[Figure 1089]

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

[Figure 1094]

Iterative denoising

[Figure 1095]

[Figure 1096]

- Figure 4: The pipeline of MotionClone, in which the motion representation Ht

α extracted from reference videos serves as motion guidance in novel video synthesis.

to the fact that motion synthesis has not yet been fully determined at this early stage. Therefore, we suggest to employ the motion-aligned Ht

α from latter denoising stage to guide motion synthesis in the entire sampling process, facilitating substantial and consistent motion guidance for superior motion alignment.

- 3.4 MOTION GUIDANCE

The pipeline of MotionClone is depicted in Fig. 4. Given a real reference video, the corresponding motion representation Ht

α is obtained by performing a single noise-adding and denoising step. During the video generation process, an initial latent is initialized from a standard Gaussian distribution and subsequently undergoes an iterative denoising procedure via a pre-trained video diffusion model, advised by both classifier-free guidance and the proposed motion guidance. Given that image structure is determined in the early steps of the denoising process (Hertz et al., 2022), whereas motion fidelity primarily depends on the structure of each frame, motion guidance only involves the early denoising steps, allowing for sufficient flexibility for semantic adjustment and thus empowering premium video generation with compelling motion fidelity and precise textual alignment.

- 4 EXPERIMENTS

- 4.1 IMPLEMENTATION DETAILS

In this work, we employ AnimateDiff(Guo et al., 2023b) as the base text-to-video generation model and leverage SparseCtrl (Guo et al., 2023a) for image-to-video and sketch-to-video generator. For given real videos, we apply single denoising in tα = 400 for motion representation extraction. k = 1 is adopted for mask in Eq. 5 to facilitate sparse constraint. “null-text” is uniformly used as textual prompt for preparing motion representations, promoting a more convenient video customization. The motion guidance is conducted on temporal attention layers in “up block.1”. The detailed ablations of above setting are represented in 4.6. Guidance weight s and λ in Eq. 2 are empirically set as 7.5, and 2000, respectively. For camera motion cloning, the denoising step is configured to 100, in which the motion guidance steps set as 50. For object motion cloning, the denoising step is raised to 300, while applying motion guidance in the early 180 steps.

Prompt: Teddy bear, on the grass.

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

[Figure 1102]

[Figure 1103]

[Figure 1104]

[Figure 1105]

[Figure 1106]

[Figure 1107]

[Figure 1108]

[Figure 1109]

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

[Figure 1114]

[Figure 1115]

[Figure 1116]

[Figure 1117]

Prompt: Island, on the ocean.

[Figure 1118]

[Figure 1119]

[Figure 1120]

[Figure 1121]

[Figure 1122]

[Figure 1123]

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

[Figure 1135]

[Figure 1136]

[Figure 1137]

[Figure 1138]

Reference VMC VideoComposer Gen-1 Tune-A-Video Control-A-Video MotionClone

- Figure 5: Visual comparison in camera motion cloning, in which MotionClone achieves superior textual alignment by better suppressing the original structure.

- 4.2 EXPERIMENTAL SETUP

Dataset. For experimental evaluation, 40 real videos sourced from DAVIS (Pont-Tuset et al., 2017) and website are utilized for a thorough analysis, comprising 15 videos with camera motion and 25 videos for object motion. These videos encompass a rich tapestry of motion types and scenarios, ranging from the dynamic motions of animals and humans to the global camera motion.

Evaluation metrics For objective evaluation, two commonly used metrics are adopted: i) Textual alignment, which quantifies the congruence with the provided textual prompt. Following previous work (Wang et al., 2024), it is measured by the average CLIP (Radford et al., 2021) cosine similarity between all video frames and text (Jeong et al., 2023); ii) Temporal consistency, the indicator of video smoothness, is quantified by calculating the average CLIP similarity among consecutive video frames. Beyond the scope of objective metrics, a user study is employed for a more nuanced assessment of human preferences in video quality, incorporating two additional criteria: i) motion preservation which evaluates the motion’s adherence to the reference video, and ii) appearance diversity which assesses the visual range and diversity in contrast to the reference video. The user study scores are derived from the average ratings provided by 20 volunteers, ranging from 1 to 5.

Baselines. For a thorough comparative analysis, various alternative methods have been examined in the comparison, including VideoComposer (Wang et al., 2024), Tune-A-video (Wu et al., 2023), Control-A-Video (Chen et al., 2023b), VMC (Jeong et al., 2023), and Gen-1 (Esser et al., 2023). A detailed description of each method is depicted in the Appendix.

- 4.3 QUALITATIVE COMPARISON

Camera motion cloning. As shown in Fig. 5, the ”clockwise rotation” and ”view switching” motion present a significant challenge. VMC and Tune-A-Video generate scenes with acceptable textual

Cat, moves its head, in the bedroom.

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

Iron man, walks, in the New York City.

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

[Figure 1173]

[Figure 1174]

[Figure 1175]

[Figure 1176]

[Figure 1177]

[Figure 1178]

[Figure 1179]

[Figure 1180]

Reference VMC VideoComposer Gen-1 Tune-A-Video Control-A-Video MotionClone

- Figure 6: Visual comparison in object motion cloning, in which MotionClone performs preferable motion fidelity with improved prompt-following ability.

Table 1: Quantitative comparison by using automotive metrics and user study.

Method VMC VideoComposer Gen-1 Tune-A-Video Control-A-Video MotionClone Textual Alignment 0.3134 0.2854 0.2462 0.3002 0.2859 0.3187

Temporal Consistency 0.9614 0.9577 0.9563 0.9351 0.9513 0.9621

Motion Preservation 2.59 3.28 3.50 2.44 3.33 3.69 Appearance Diversity 3.51 3.23 3.25 3.09 3.27 4.31

Textual Alignment 3.79 2.71 2.80 3.04 2.82 4.15 Temporal Consistency 2.85 2.79 3.34 2.28 2.81 4.28

alignment but exhibit deficiencies in motion transfer. The outputs from VideoComposer, Gen-1, and Control-A-Video are notably unrealistic, which can be attributed to the dense integration of the structural elements from the original videos. Conversely, MotionClone demonstrates superior text alignment and motion consistency, thereby suggesting its superior video motion transfer capabilities within global camera motion scenarios.

Object motion cloning. Beyond the scope of camera motion, the proficiency in handling local object motions has been rigorously validated. As evidenced by Fig. 6, VMC falls short in matching motion with the source videos. Videocomposer appears to generate grayish colors with limited prompt-following ability. Gen-1 is inhibited by the original videos’ structure. Tune-A-Video struggles with capturing detailed body motions, while Control-A-Video cannot maintain a faithful appearance. In contrast, MotionClone stands out in scenarios with localized object motions, enhancing motion accuracy and improved text alignment.

[Figure 1181]

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

[Figure 1186]

Prompt: Blue car, runs in the road.

Prompt: Girl, similing in house.

[Figure 1187]

[Figure 1188]

[Figure 1189]

[Figure 1190]

[Figure 1191]

[Figure 1192]

[Figure 1193]

[Figure 1194]

[Figure 1195]

[Figure 1196]

[Figure 1197]

[Figure 1198]

[Figure 1199]

[Figure 1200]

Prompt: Airplane at airport, sunny day.

Prompt: Lion, walks in the forest.

[Figure 1201]

[Figure 1202]

[Figure 1203]

[Figure 1204]

[Figure 1205]

[Figure 1206]

[Figure 1207]

[Figure 1208]

- Figure 7: MotionClone also supports I2V and sketch-to-video, facilitating versatile applications. The red arrows indicate the motion direction.

- 4.4 QUANTITATIVE COMPARISON

The quantitative comparison on 40 real videos with various motion pattern are outlined in Tab. 1. It is observed that MotionClone gains competitive scores in both textual alignment and temporal consistency. Moreover, MotionClone has outperformed its rivals in motion preservation, appearance diversity, temporal consistency, and textual alignment in human preference tests, underscoring its ability to produce visually compelling outcomes.

- 4.5 VERSATILE APPLICATION

Beyond T2V, MotionClone is also compatible with I2V and sketch-to-video. As shown in Fig. 7, by incorporating the first frame or a sketch image as an additional condition, MotionClone achieves impressive motion transfer while aligning with the specified condition, underscoring its significant potential for a wide range of applications.

- 4.6 ABLATION AND ANALYSIS

Choice of k. k determines the mask in Eq. 5 and thus impacts the sparsity of motion constraint. As illustrated in Fig. 8, a lower k value helps better primary motion alignment, attributed by the enhanced elimination of scene-specific noise and subtle motion.

Choice of tα. The value of tα determines diffusion feature distribution used for preparing motion representations. As shown in Fig. 8, an excessively large tα = 800 causes substantial loss of motion information due to excessive noise injection, while tα ∈ {200,400,600} can all achieve a certain degree of motion alignment, implying the robustness of tα. In this work, we chose tα = 400 as default value as it typically yields appealing motion cloning in our experiments.

Choice of temporal attention block. Fig. 9 illustrates the results with motion guidance applied in different blocks. It is observed that “up block.1” stands out for its superior motion manipulation capabilities while safeguarding visual quality, underscoring its dominant role in motion synthesis.

Does precise prompt help ? During motion representation preparation procedure, few differences arise when using tailored prompts regrading video content, as represented Fig. 9. We speculate that motion-related information is effective preserved in diffusion features at tα = 400, thereby diminishing the significance of the precise prompt.

Does video inversion help ? Video inversion can provide time-dependence Atref,Mt for Eq. 4 and certain time step {Lt

} for Eq. 6, but entails considerable time costs. As shown in Fig. 9 (Inversion 1 vs. Inversion 2), {Lt

#### ,Mt

α

α

} outperforms Atref,Mt due to the consistent motion guidance from the same representation. Meanwhile, there is not obvious quality difference regarding

#### ,Mt

α

α

Promt: Woman,walks in the mall. Promt: Shark, swims on the ocean.

[Figure 1209]

[Figure 1210]

[Figure 1211]

[Figure 1212]

[Figure 1213]

[Figure 1214]

[Figure 1215]

[Figure 1216]

[Figure 1217]

[Figure 1218]

[Figure 1219]

[Figure 1220]

[Figure 1221]

[Figure 1222]

[Figure 1223]

[Figure 1224]

[Figure 1225]

[Figure 1226]

[Figure 1227]

[Figure 1228]

[Figure 1229]

[Figure 1230]

[Figure 1231]

[Figure 1232]

[Figure 1233]

[Figure 1234]

[Figure 1235]

[Figure 1236]

[Figure 1237]

[Figure 1238]

[Figure 1239]

[Figure 1240]

[Figure 1241]

[Figure 1242]

[Figure 1243]

[Figure 1244]

[Figure 1245]

[Figure 1246]

[Figure 1247]

[Figure 1248]

[Figure 1249]

[Figure 1250]

Reference Reference

### Figure 8: Influence of different k value and different time step tα.

Prompt: Dog, walks in the street. Prompt: Spider Man, turns his head.

[Figure 1251]

[Figure 1252]

[Figure 1253]

[Figure 1254]

[Figure 1255]

[Figure 1256]

[Figure 1257]

[Figure 1258]

[Figure 1259]

[Figure 1260]

[Figure 1261]

[Figure 1262]

[Figure 1263]

[Figure 1264]

[Figure 1265]

[Figure 1266]

[Figure 1267]

[Figure 1268]

[Figure 1269]

[Figure 1270]

[Figure 1271]

[Figure 1272]

[Figure 1273]

[Figure 1274]

[Figure 1275]

[Figure 1276]

[Figure 1277]

[Figure 1278]

[Figure 1279]

[Figure 1280]

[Figure 1281]

[Figure 1282]

[Figure 1283]

[Figure 1284]

[Figure 1285]

[Figure 1286]

[Figure 1287]

[Figure 1288]

[Figure 1289]

[Figure 1290]

[Figure 1291]

[Figure 1292]

Reference MotionClone Up_block.2 Up_block.3 Prompt Inversion_1 Inversion_2 Reference MotionClone Up_block.2 Up_block.3 Prompt Inversion_1 Inversion_2

Figure 9: Influence of different attention block, precise prompt, and DDIM inversion. “Prompt” denotes motion representation involves precise prompt (“Leopard, walks in the forest” for the left case and “Man, turns his head.” for the right case); “Inversion 1” represents the time-dependence

Atref,Mt from DDIM inversion; “Inversion 2” indicates {Lt

} from DDIM inversion.

,Mt

α

α

whether DDIM inversion is applied (MotionClone vs. Inversion 2). We leave how to perform better diffusion inversion for enhanced motion cloning to further work.

- 4.7 LIMITATION

Given that MotioClone is conducted in latent space, the spatial resolution of diffusion features in temporary attention is significantly lower than that of input videos, thus MotionClone struggles in local subtle motion, such as winking, as shown in Fig. 10. Additionally, when multiple moving objects overlap, MotionClone risks quality dropping, attributing that coupled motion raises the difficulty of motion cloning.

[Figure 1293]

[Figure 1294]

[Figure 1295]

[Figure 1296]

[Figure 1297]

[Figure 1298]

[Figure 1299]

[Figure 1300]

[Figure 1301]

[Figure 1302]

[Figure 1303]

[Figure 1304]

Prompt: A dog is winking. Prompt: Pigs, play in the mud.

Figure 10: MotionClone struggles to handle local subtle motion and overlapping motion.

- 5 CONCLUSION

In this work, we observe that the temporal attention layers embedded within video generation models harbor substantial representational capacities pertinent to video motion transfer. Motivated by these findings, we introduce a training-free method dubbed MotionClone for motion cloning. Leveraging sparse temporal attention weights as motion representations, MotionClone facilitates motion guidance by promoting primary motion alignment, enabling diverse motion transfer across different scenarios. Employing a real reference video, MotionClone demonstrates its capability to preserve motion fidelity robustly while concurrently assimilating novel textual semantics. Furthermore, MotionClone demonstrates efficiency by avoiding cumbersome inversion processes and offers versatility across various video generation tasks, establishing itself as a highly adaptable and efficient tool for motion customization.

REFERENCES

Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1728–1738, 2021.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023a.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22563–22575, 2023b.

Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–10, 2023.

Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for highquality video generation. arXiv preprint arXiv:2310.19512, 2023a.

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. arXiv preprint arXiv:2401.09047, 2024.

Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models. arXiv preprint arXiv:2305.13840, 2023b.

Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Animateanything: Fine-grained open domain image animation with motion guidance. arXiv eprints, pp. arXiv–2311, 2023.

Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7346–7356, 2023.

Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10696–10706, 2022.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. arXiv preprint arXiv:2311.16933, 2023a.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023b.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023.

Hyeonho Jeong and Jong Chul Ye. Ground-a-video: Zero-shot grounded video editing using textto-image diffusion models. arXiv preprint arXiv:2310.01107, 2023.

Hyeonho Jeong, Geon Yeong Park, and Jong Chul Ye. Vmc: Video motion customization using temporal attention adaption for text-to-video diffusion models. arXiv preprint arXiv:2312.00845, 2023.

Yunji Kim, Jiyoung Lee, Jin-Hwa Kim, Jung-Woo Ha, and Jun-Yan Zhu. Dense text-to-image generation with attention modulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7701–7711, 2023.

Max Ku, Cong Wei, Weiming Ren, Huan Yang, and Wenhu Chen. Anyv2v: A plug-and-play framework for any video-to-video editing tasks. arXiv preprint arXiv:2403.14468, 2024.

Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22511–22521, 2023.

Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. arXiv preprint arXiv:2303.04761, 2023.

Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning. arXiv preprint arXiv:2307.11410, 2023.

Yue Ma, Yingqing He, Hongfa Wang, Andong Wang, Chenyang Qi, Chengfei Cai, Xiu Li, Zhifeng Li, Heung-Yeung Shum, Wei Liu, et al. Follow-your-click: Open-domain regional image animation via short prompts. arXiv preprint arXiv:2403.08268, 2024.

Sicheng Mo, Fangzhou Mu, Kuan Heng Lin, Yanli Liu, Bochen Guan, Yin Li, and Bolei Zhou. Freecontrol: Training-free spatial control of any text-to-image diffusion model with any condition. arXiv preprint arXiv:2312.07536, 2023.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

Muyao Niu, Xiaodong Cun, Xintao Wang, Yong Zhang, Ying Shan, and Yinqiang Zheng. Mofavideo: Controllable image animation via generative motion field adaptions in frozen image-tovideo diffusion model. arXiv preprint arXiv:2405.20222, 2024.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017.

Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, et al. Unicontrol: A unified diffusion model for controllable visual generation in the wild. arXiv preprint arXiv:2305.11147, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. Advances in Neural Information Processing Systems, 36, 2024.

Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36, 2024.

Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023a.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. arXiv preprint arXiv:2312.03641, 2023b.

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7623–7633, 2023.

Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multi-subject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023.

Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Y He, H Liu, H Chen, X Cun, X Wang, Y Shan, et al. Make-your-video: Customized video generation using textual and structural guidance. IEEE Transactions on Visualization and Computer Graphics, 2024.

Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023.

Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. Long-clip: Unlocking the long-text capability of clip. arXiv preprint arXiv:2403.15378, 2024.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3836–3847, 2023.

Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. Motiondirector: Motion customization of text-to-video diffusion models. arXiv preprint arXiv:2310.08465, 2023.

A APPENDIX

- A.1 BASELINE DESCRIPTION

Among the compared methods, VideoComposer (Wang et al., 2024) creates videos by extracting specific features such as frame-wise depth or canny maps from existing videos, achieving a compositional approach to controllable video generation. Gen-1 (Esser et al., 2023) leverages the original structure of reference videos to generate new video content, akin to video-to-video translation. Tune-A-Video expands the spatial self-attention of pre-trained text-to-image models into spatiotemporal attention, and then fine-tuning it for motion-specific generation. Control-A-Video (Chen et al., 2023b) incorporates the first video frame as an additional motion cue for customized video generation. VMC (Jeong et al., 2023) aims to distill motion patterns by fine-tuning the temporal attention layers in a pre-trained text-to-video diffusion model.

- A.2 MORE GENERATED RESULTS

A broader array of generated content is displayed to validate the versatile generation capability. As shown in Figs. 11- 14, MotionClone is able to adeptly extract motion cues from a diverse range of existing videos and thus enables the creation of content that is both prompt-aligned and motionpreserved, showcasing its robust motion cloning capabilities. For a better demonstration of MotionClone, we highly recommend viewing the video file at https://github.com/LPengYang/ MotionClone.

- A.3 BROADER IMPACT

The development of MotionClone, a novel training-free framework for motion-based controllable video generation, carry distinct societal implications, both beneficial and challenging.

On the positive side, MotionClone’s capability to efficiently clone motions from reference videos while ensuring high fidelity and textual alignment opens new avenues in numerous fields. In the realm of digital content creation, film and media professionals can utilize this technology to streamline the production process, enhance narrative expressions, and create more engaging visual experiences without extensive resource commitments. Furthermore, in the educational sector, instructors and content creators can leverage this innovation to produce customized instructional videos that incorporate precise motions aligned with textual descriptions, potentially increasing engagement and comprehension among students. This could be particularly transformative for subjects where demonstration of physical actions or processes plays a crucial role, such as in sports training or scientific experiments.

On the negative side, the power of MotionClone to generate realistic videos based on text and existing motion cues raises concerns about its potential misuse, including the creation of deepfakes or misleading media content. Such applications can undermine trust in media, affect public opinion through the dissemination of false information, and infringe on personal rights and privacy. Moreover, the ease of generating convincing videos might enable the proliferation of propaganda or harmful content that can have widespread negative implications on society.

In conclusion, while MotionClone presents significant advancements in the field of AI-driven video generation, it is imperative that these technologies are developed and utilized with a conscious commitment to ethical standards and regulatory oversight. Promoting transparency in AI-generated content, establishing clear usage guidelines, and fostering an open dialogue about the capabilities and ethics of such technologies are crucial steps in ensuring that the benefits of MotionClone are realized while its risks are effectively mitigated. This involves collaborative efforts among technologists, policymakers, industry stakeholders, and the broader public to steer the responsible development and application of AI-driven media tools.

[Figure 1305]

[Figure 1306]

[Figure 1307]

[Figure 1308]

[Figure 1309]

[Figure 1310]

[Figure 1311]

[Figure 1312]

Reference video (camera zoom in)

[Figure 1313]

[Figure 1314]

[Figure 1315]

[Figure 1316]

[Figure 1317]

[Figure 1318]

[Figure 1319]

[Figure 1320]

Prompt: A road, in the mountain.

[Figure 1321]

[Figure 1322]

[Figure 1323]

[Figure 1324]

[Figure 1325]

[Figure 1326]

[Figure 1327]

[Figure 1328]

Prompt: Relices, on the seabed.

[Figure 1329]

[Figure 1330]

[Figure 1331]

[Figure 1332]

[Figure 1333]

[Figure 1334]

[Figure 1335]

[Figure 1336]

Prompt: Caves, a path for exploration.

[Figure 1337]

[Figure 1338]

[Figure 1339]

[Figure 1340]

[Figure 1341]

[Figure 1342]

[Figure 1343]

[Figure 1344]

Prompt: Rail way for train.

[Figure 1345]

[Figure 1346]

[Figure 1347]

[Figure 1348]

[Figure 1349]

[Figure 1350]

[Figure 1351]

[Figure 1352]

Reference video (camera pan up)

[Figure 1353]

[Figure 1354]

[Figure 1355]

[Figure 1356]

[Figure 1357]

[Figure 1358]

[Figure 1359]

[Figure 1360]

Prompt: Camel, in the desert.

[Figure 1361]

[Figure 1362]

[Figure 1363]

[Figure 1364]

[Figure 1365]

[Figure 1366]

[Figure 1367]

[Figure 1368]

Prompt: Forest, in winter.

[Figure 1369]

[Figure 1370]

[Figure 1371]

[Figure 1372]

[Figure 1373]

[Figure 1374]

[Figure 1375]

[Figure 1376]

Prompt: Lake surface, in the crater.

[Figure 1377]

[Figure 1378]

[Figure 1379]

[Figure 1380]

[Figure 1381]

[Figure 1382]

[Figure 1383]

[Figure 1384]

Prompt: Ship on the ocean.

[Figure 1385]

[Figure 1386]

[Figure 1387]

[Figure 1388]

[Figure 1389]

[Figure 1390]

[Figure 1391]

[Figure 1392]

Reference video (camera pan down)

[Figure 1393]

[Figure 1394]

[Figure 1395]

[Figure 1396]

[Figure 1397]

[Figure 1398]

[Figure 1399]

[Figure 1400]

Prompt: Car, in the street.

[Figure 1401]

[Figure 1402]

[Figure 1403]

[Figure 1404]

[Figure 1405]

[Figure 1406]

[Figure 1407]

[Figure 1408]

Prompt: Cliffs by the sea.

[Figure 1409]

[Figure 1410]

[Figure 1411]

[Figure 1412]

[Figure 1413]

[Figure 1414]

[Figure 1415]

[Figure 1416]

Prompt: Eagle, standing on a tree.

[Figure 1417]

[Figure 1418]

[Figure 1419]

[Figure 1420]

[Figure 1421]

[Figure 1422]

[Figure 1423]

[Figure 1424]

Prompt: Forest, with various flowers.

[Figure 1425]

[Figure 1426]

[Figure 1427]

[Figure 1428]

[Figure 1429]

[Figure 1430]

[Figure 1431]

[Figure 1432]

Reference video (camera zoom out)

[Figure 1433]

[Figure 1434]

[Figure 1435]

[Figure 1436]

[Figure 1437]

[Figure 1438]

[Figure 1439]

[Figure 1440]

Prompt: Man, standing in his garden.

[Figure 1441]

[Figure 1442]

[Figure 1443]

[Figure 1444]

[Figure 1445]

[Figure 1446]

[Figure 1447]

[Figure 1448]

Prompt: Penguin, on the beach.

[Figure 1449]

[Figure 1450]

[Figure 1451]

[Figure 1452]

[Figure 1453]

[Figure 1454]

[Figure 1455]

[Figure 1456]

Prompt: Red car, on the track.

[Figure 1457]

[Figure 1458]

[Figure 1459]

[Figure 1460]

[Figure 1461]

[Figure 1462]

[Figure 1463]

[Figure 1464]

Prompt: Tree, in the mountain.

[Figure 1465]

[Figure 1466]

[Figure 1467]

[Figure 1468]

[Figure 1469]

[Figure 1470]

[Figure 1471]

[Figure 1472]

[Figure 1473]

[Figure 1474]

[Figure 1475]

[Figure 1476]

[Figure 1477]

[Figure 1478]

[Figure 1479]

[Figure 1480]

Prompt: Batman, turns his head.

[Figure 1481]

[Figure 1482]

[Figure 1483]

[Figure 1484]

[Figure 1485]

[Figure 1486]

[Figure 1487]

[Figure 1488]

[Figure 1489]

[Figure 1490]

[Figure 1491]

[Figure 1492]

[Figure 1493]

[Figure 1494]

[Figure 1495]

[Figure 1496]

Prompt: Red car, runs on the beach.

[Figure 1497]

[Figure 1498]

[Figure 1499]

[Figure 1500]

[Figure 1501]

[Figure 1502]

[Figure 1503]

[Figure 1504]

[Figure 1505]

[Figure 1506]

[Figure 1507]

[Figure 1508]

[Figure 1509]

[Figure 1510]

[Figure 1511]

[Figure 1512]

Prompt: Petals falling in the wind.

[Figure 1513]

[Figure 1514]

[Figure 1515]

[Figure 1516]

[Figure 1517]

[Figure 1518]

[Figure 1519]

[Figure 1520]

[Figure 1521]

[Figure 1522]

[Figure 1523]

[Figure 1524]

[Figure 1525]

[Figure 1526]

[Figure 1527]

[Figure 1528]

Prompt: Cat, runs in house.

### Figure 13: More results of MotionClone in object motion cloning.

[Figure 1529]

[Figure 1530]

[Figure 1531]

[Figure 1532]

[Figure 1533]

[Figure 1534]

[Figure 1535]

[Figure 1536]

[Figure 1537]

[Figure 1538]

[Figure 1539]

[Figure 1540]

[Figure 1541]

[Figure 1542]

[Figure 1543]

[Figure 1544]

Prompt: Blue car, runs on the beach.

[Figure 1545]

[Figure 1546]

[Figure 1547]

[Figure 1548]

[Figure 1549]

[Figure 1550]

[Figure 1551]

[Figure 1552]

[Figure 1553]

[Figure 1554]

[Figure 1555]

[Figure 1556]

[Figure 1557]

[Figure 1558]

[Figure 1559]

[Figure 1560]

Prompt: Greek sculpture, walks in the forest.

[Figure 1561]

[Figure 1562]

[Figure 1563]

[Figure 1564]

[Figure 1565]

[Figure 1566]

[Figure 1567]

[Figure 1568]

[Figure 1569]

[Figure 1570]

[Figure 1571]

[Figure 1572]

[Figure 1573]

[Figure 1574]

[Figure 1575]

[Figure 1576]

Prompt: Cat, turns its head in house.

[Figure 1577]

[Figure 1578]

[Figure 1579]

[Figure 1580]

[Figure 1581]

[Figure 1582]

[Figure 1583]

[Figure 1584]

[Figure 1585]

[Figure 1586]

[Figure 1587]

[Figure 1588]

[Figure 1589]

[Figure 1590]

[Figure 1591]

[Figure 1592]

Prompt: Pig, walks in the forest.

### Figure 14: More results of MotionClone in object motion cloning.

