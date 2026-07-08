# arXiv:2308.03610v1[cs.CV]7Aug2023

## AvatarVerse: High-quality & Stable 3D Avatar Creation from Text and Pose

### Huichao Zhang1*, Bowen Chen1*, Hao Yang1, Liao Qu1,2, Xu Wang1 Li Chen1, Chao Long1, Feida Zhu1, Kang Du1, Min Zheng1

1ByteDance, Beijing, China. 2Department of Electrical and Computer Engineering, Carnegie Mellon University, PA, USA. {zhanghuichao.hc, chenbowen.cbw, wangxu.ailab, chenli.phd, longchao, zhufeida, dukang.daniel, zhengmin.666}@bytedance.com, liaoq@andrew.cmu.edu, yanghao.alexis@foxmail.com

###### Abstract

Creating expressive, diverse and high-quality 3D avatars from highly customized text descriptions and pose guidance is a challenging task, due to the intricacy of modeling and texturing in 3D that ensure details and various styles (realistic, fictional, etc). We present AvatarVerse, a stable pipeline for generating expressive high-quality 3D avatars from nothing but text descriptions and pose guidance. In specific, we introduce a 2D diffusion model conditioned on DensePose signal to establish 3D pose control of avatars through 2D images, which enhances view consistency from partially observed scenarios. It addresses the infamous Janus Problem and significantly stablizes the generation process. Moreover, we propose a progressive high-resolution 3D synthesis strategy, which obtains substantial improvement over the quality of the created 3D avatars. To this end, the proposed AvatarVerse pipeline achieves zero-shot 3D modeling of 3D avatars that are not only more expressive, but also in higher quality and fidelity than previous works. Rigorous qualitative evaluations and user studies showcase AvatarVerse’s superiority in synthesizing high-fidelity 3D avatars, leading to a new standard in high-quality and stable 3D avatar creation. Our project page is: https://avatarverse3d.github.io/ .

### 1. Introduction

The creation of high-quality 3D avatars has garnered significant interest due to their widespread applications in domains such as game production, social media and communication, augmented and virtual reality (AR/VR), and human-computer interaction. Traditional manual construction of these intricate 3D models is a labor-intensive and time-consuming process, requiring thousands of hours from skilled artists possessing extensive aesthetic and 3D modeling expertise. Consequently, automating the generation of high-quality 3D avatars using only natural language descriptions holds great research prospects with the potential to save resources, which is also the goal of our work.

In recent years, significant efforts have been made in reconstructing high-fidelity 3D avatars from multi-view videos (Isik et al. 2023; Jiang et al. 2022; Li et al. 2023b; Wang et al. 2023a; Zheng et al. 2023) or reference images (Wang et al. 2021; Xiu et al. 2022). These methods primarily rely on limited visual priors sourced from videos or

*These authors contributed equally.

reference images, leading to constrained ability to generate creative avatars with complex text prompts. In 2D image generation, diffusion models (Rombach et al. 2021; Saharia et al. 2022; Zhang and Agrawala 2023) illustrate considerable creativity, primarily due to the availability of large-scale text-image pairs. Nevertheless, the scarcity and limited diversity of 3D models present challenges to effectively training a 3D diffusion model. Recent studies (Cao et al. 2023; Huang et al. 2023; Kolotouros et al. 2023; Poole et al. 2022) have investigated the use of pre-trained text-image generative models to optimize Neural Radiance Fields (NeRF) (Mildenhall et al. 2020) for generating high-fidelity 3D models. Yet, stable creation of high-quality 3D avatars exhibiting various poses, appearances, and shapes remains a difficult task. For example, employing common score distillation sampling (SDS) (Poole et al. 2022) to guide NeRF optimization without additional control tends to bring in the Janus (multi-face) problem. Also, the avatars produced by current approaches tend to exhibit noticeable blurriness and coarseness, leading to the absence of high-resolution local texture details, accessories, and other relevant features.

To cope with these weaknesses, we propose AvatarVerse, a novel framework designed for generating high-quality and stable 3D avatars from textual descriptions and pose guidances. We first train a new ControlNet with human DensePose condition (G¨uler, Neverova, and Kokkinos 2018) over 800K images. SDS loss conditinal on the 2D DensePose signal is then implemented on top of the ControlNet. Through this way, we obtain precise view correspondence between different 2D views as well as between every 2D view and the 3D space. Our approach not only enables pose control of the generated avatars, but also eliminates the Janus Problem suffered by most existing methods. It thus ensures a more stable and view-consistent avatar creation process. Additionally, benefiting from the accurate and flexible supervision signals provided by DensePose, the generated avatars can be highly aligned with the joints of the SMPL model, enabling simple and effective skeletal binding and control.

While relying solely on DensePose-conditioned ControlNet may result in local artifacts, we introduce a progressive high-resolution generation strategy to enhance the fidelity and detail of local geometry. To alleviate the coarseness of the generated avatar, we incorporate a smoothness loss, which regularizes the synthesis procedure by encourag-

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Elsa in Frozen Disney Woody in Toy Story Captain America Buzz Lightyear

Super Saiyan Goku

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

a man wearing a white tanktop and shorts

Nick Wilde from film Zootopia Simba from The Lion King a Viking wearingabodyabuildertanktop

a person dresed at the Venice Carnival

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Ronald Weasley

Captain Jack Sparrow

Mobile suit Gundam

Link in Zelda Spiderman

Hulk

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

a karate master wearing a black belt

Jake Sully in Avatar series

The Flash Deadpool Albus Dumbledore

a security guard

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Batman Stormtrooper Monkey D. Luffy

A young man with curly hair wearing glasses

Master Chief in Halo Series

Yoda in Star Wars Series

Figure 1: High-quality 3D avatars generated by AvatarVerse based on a simple text description.

ing a smoother gradient of the density voxel grid within our computationally efficient explicit Neural Radiance Fields (NeRF).

The overall contributions are as follows:

- • We present AvatarVerse, a method that can automatically create a high-quality 3D avatar accoding to nothing but a text description and a reference human pose.
- • We present the DensePose-Conditioned Score Distillation Sampling Loss, an approach that facilitates poseaware 3D avatar synthesis and effectively mitigates the Janus problem, thereby enhancing system stability.
- • We bolster the quality of the produced 3D avatars via a progressive high-resolution generation strategy. This method, through a meticulous coarse-to-fine refining process, synthesizes 3D avatars with superior detail, encompassing elements like hands, accessories, and beyond.
- • AvatarVerse delivers exceptional performance, excelling in both quality and stability. Rigorous qualitative evaluations, complemented by comprehensive user studies, underscore AvatarVerse’s supremacy in crafting highfidelity 3D avatars, thereby setting a new benchmark in stable, zero-shot 3D avatar creation of the highest quality.

### 2. Related work

#### 2.1. Text-guided 3D content generation

The success in text-guided 2D image generation has paved the way for the development of text-guided 3D content generation methods. CLIP-forge (Sanghi et al. 2021), DreamFields (Jain et al. 2021), and CLIP-Mesh (Khalid et al. 2022) utilize the CLIP model (Radford et al. 2021) to optimize underlying 3D representations such as meshes and NeRF. DreamFusion (Poole et al. 2022) first proposes score distillation sampling (SDS) loss to get supervision from a pre-trained diffusion model (Saharia et al. 2022) during the 3D generation. Latent-NeRF (Metzer et al. 2022) improves upon DreamFusion by optimizing a NeRF that operates the diffusion process in a latent space. TEXTure (Richardson et al. 2023) generates texture maps using a depth diffusion model for a given 3D mesh. ProlificDreamer (Wang et al. 2023b) proposes variational score distillation and produces high-resolution and high-fidelity results. Despite their promising performance in 3D general content generation, these methods often produce suboptimal results when generating avatars, exhibiting issues like low quality, Janus (multiface) problem, and incorrect body parts. In contrast, our AvatarVerse enables an accurate and high-quality generation of 3D avatars from text prompts.

#### 2.2. Text-guided 3D Avatar generation

Avatar-CLIP (Hong et al. 2022) first initializes 3D human geometry with a shape VAE network and utilizes CLIP (Radford et al. 2021) to facilitate geometry sculpting and texture generation. DreamAvatar (Cao et al. 2023) and AvatarCraft (Jiang et al. 2023) employ the SMPL model as a shape prior and utilize pretrained text-to-image diffusion models to generate 3D avatars. DreamFace (Zhang et al.

2023) introduces a coarse-to-fine scheme to create personalized 3D facial structures. HeadSculpt (Han et al. 2023) generates 3D head avatars by leveraging landmark-based control and a learned textual embedding representing the back view appearance of heads. Concurrent with our work, DreamWaltz (Huang et al. 2023) presents 3D-consistent occlusion-aware score distillation sampling, which incorporates 3D-aware skeleton conditioning for view-aligned supervision. Constrained by the original training data, the skeleton-conditioned diffusion model may still exhibit view inconsistencies such as failing to generate the backside of desired avatars or struggling to generate specific body parts when provided with partial skeleton information. Furthermore, the sparse nature of the skeleton makes it challenging for the model to determine avatar contours and edges, leading to low-quality results. On the contrary, our proposed DensePose-conditioned ControlNet ensures highquality, view-consistent image generation of various viewpoints and body parts, including full body, legs, head, and more, guaranteeing superior avatar quality.

#### 2.3. High-quality 3D Avatar Generation

Recently, there has been a growing focus on achieving highquality or high-fidelity 3D generation and reconstruction. Some methods attempt to generate high-fidelity 3D human avatars from multi-view RGB videos (Isik et al. 2023; Jiang et al. 2022; Li et al. 2023b; Wang et al. 2023a; Zheng et al. 2023). There has also been work (Lin et al. 2022) explored a coarse-to-fine methodology, specifically by optimizing a high-resolution latent diffusion model to refine a textured 3D mesh model. In parallel to our work, DreamHuman (Kolotouros et al. 2023) zooms in and renders a 64 × 64 image for 6 important body regions during optimization. However, limited by the computation needs of Mip-NeRF-360, it can only produce low-resolution avatars without highresolution details. Also, DreamHuman use SMPL shape for direct geometric supervision, which tends to provide skintight avatars. Our method, on the other hand, is more controllable and flexible, allowing for the creation of a wider range of accessories, clothing, and other features. Our AvatarVerse introduces a progressive high-resolution generation strategy. This involves gradually decreasing the camera’s radius and focusing on distinct body parts, which facilitates the creation of a diverse range of accessories, clothing, and other elements. Our use of progressive grid also ensures a finegrained generation.

### 3. Methodology

In this section, we present AvatarVerse, a fully automatic pipeline that can make a realistic 3D avatar from nothing but a text description and a body pose. After introducing some preliminaries, we first explain the DensePose-conditioned SDS loss, which facilitates pose-aware 3D avatar synthesis and effectively mitigates the Janus problem. We then introduce novel strategies that enhance the synthesis quality: the progressive high-resolution generation strategy and the avatar surface smoothing strategy.

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

A DLSR photo of Caption America

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

|[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]|
|---|

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

|[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>|
|---|

[Figure 195]

|[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]<br><br>[Figure 206]<br><br>[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]<br><br>[Figure 211]|
|---|

|[Figure 212]<br><br>[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>[Figure 218]<br><br>[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]<br><br>[Figure 222]<br><br>[Figure 223]<br><br>[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]<br><br>[Figure 228]<br><br>[Figure 229]<br><br>[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]<br><br>[Figure 238]<br><br>[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]<br><br>[Figure 242]<br><br>[Figure 243]<br><br>[Figure 244]<br><br>[Figure 245]<br><br>[Figure 246]<br><br>[Figure 247]<br><br>[Figure 248]<br><br>[Figure 249]<br><br>[Figure 250]<br><br>[Figure 251]<br><br>[Figure 252]<br><br>[Figure 253]<br><br>[Figure 254]<br><br>[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]<br><br>[Figure 258]<br><br>[Figure 259]<br><br>[Figure 260]<br><br>[Figure 261]<br><br>[Figure 262]<br><br>[Figure 263]<br><br>[Figure 264]<br><br>[Figure 265]<br><br>[Figure 266]<br><br>[Figure 267]<br><br>[Figure 268]<br><br>[Figure 269]<br><br>[Figure 270]<br><br>[Figure 271]<br><br>[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]<br><br>[Figure 276]<br><br>[Figure 277]<br><br>[Figure 278]<br><br>[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]<br><br>[Figure 284]<br><br>[Figure 285]<br><br>[Figure 286]<br><br>[Figure 287]<br><br>[Figure 288]<br><br>[Figure 289]<br><br>[Figure 290]<br><br>[Figure 291]<br><br>[Figure 292]<br><br>[Figure 293]<br><br>[Figure 294]<br><br>[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>[Figure 298]<br><br>[Figure 299]<br><br>[Figure 300]<br><br>[Figure 301]<br><br>[Figure 302]<br><br>[Figure 303]<br><br>[Figure 304]<br><br>[Figure 305]<br><br>[Figure 306]<br><br>[Figure 307]<br><br>[Figure 308]<br><br>[Figure 309]<br><br>[Figure 310]<br><br>[Figure 311]<br><br>[Figure 312]<br><br>[Figure 313]<br><br>[Figure 314]<br><br>[Figure 315]<br><br>[Figure 316]<br><br>[Figure 317]<br><br>[Figure 318]<br><br>[Figure 319]<br><br>[Figure 320]<br><br>[Figure 321]<br><br>[Figure 322]<br><br>[Figure 323]<br><br>[Figure 324]<br><br>[Figure 325]<br><br>[Figure 326]<br><br>[Figure 327]<br><br>[Figure 328]<br><br>[Figure 329]<br><br>[Figure 330]<br><br>[Figure 331]<br><br>[Figure 332]<br><br>[Figure 333]<br><br>[Figure 334]<br><br>[Figure 335]<br><br>[Figure 336]<br><br>[Figure 337]<br><br>[Figure 338]<br><br>[Figure 339]<br><br>[Figure 340]<br><br>[Figure 341]<br><br>[Figure 342]<br><br>[Figure 343]<br><br>[Figure 344]<br><br>[Figure 345]<br><br>[Figure 346]<br><br>[Figure 347]<br><br>[Figure 348]<br><br>[Figure 349]<br><br>[Figure 350]<br><br>[Figure 351]<br><br>[Figure 352]<br><br>[Figure 353]<br><br>[Figure 354]<br><br>[Figure 355]<br><br>[Figure 356]<br><br>[Figure 357]<br><br>[Figure 358]<br><br>[Figure 359]<br><br>[Figure 360]<br><br>[Figure 361]|
|---|

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

##### ...

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

densepose condition

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

[Figure 413]

[Figure 414]

[Figure 415]

densepose render

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

(1) progressive grid (2) bbox tightening

ControlNet ℒ

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

|[Figure 439]<br><br>[Figure 440]|
|---|

[Figure 441]

|[Figure 442]<br><br>[Figure 443]|
|---|

shared viewpoint

volume render

|[Figure 444]|
|---|

|[Figure 445]|
|---|

|[Figure 446]|
|---|

[Figure 447]

[Figure 448]

|[Figure 449]<br><br>[Figure 450]|
|---|

| |
|---|

| |
|---|

[Figure 451]

[Figure 452]

|[Figure 453]<br><br>[Figure 454]|
|---|

|[Figure 455]<br><br>[Figure 456]|
|---|

|[Figure 457]<br><br>[Figure 458]|
|---|

explicit NeRF

|[Figure 459]<br><br>[Figure 460]|
|---|

[Figure 461]

[Figure 462]

[Figure 463]

| |
|---|

|MLP|
|---|

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

|[Figure 511]<br><br>[Figure 512]<br><br>[Figure 513]<br><br>[Figure 514]<br><br>[Figure 515]<br><br>[Figure 516]<br><br>[Figure 517]<br><br>[Figure 518]<br><br>[Figure 519]<br><br>[Figure 520]<br><br>[Figure 521]<br><br>[Figure 522]<br><br>[Figure 523]<br><br>[Figure 524]<br><br>[Figure 525]<br><br>[Figure 526]<br><br>[Figure 527]<br><br>[Figure 528]<br><br>[Figure 529]<br><br>[Figure 530]<br><br>[Figure 531]<br><br>[Figure 532]<br><br>[Figure 533]<br><br>[Figure 534]<br><br>[Figure 535]<br><br>[Figure 536]<br><br>[Figure 537]<br><br>[Figure 538]<br><br>[Figure 539]<br><br>[Figure 540]<br><br>[Figure 541]<br><br>[Figure 542]<br><br>[Figure 543]<br><br>[Figure 544]<br><br>[Figure 545]<br><br>[Figure 546]<br><br>[Figure 547]<br><br>𝑽(𝒅𝒆𝒏𝒔𝒊𝒕𝒚)|
|---|

|[Figure 548]<br><br>[Figure 549]<br><br>[Figure 550]<br><br>[Figure 551]<br><br>[Figure 552]<br><br>[Figure 553]<br><br>[Figure 554]<br><br>[Figure 555]<br><br>[Figure 556]<br><br>[Figure 557]<br><br>[Figure 558]<br><br>[Figure 559]<br><br>[Figure 560]<br><br>[Figure 561]<br><br>[Figure 562]<br><br>[Figure 563]<br><br>[Figure 564]<br><br>[Figure 565]<br><br>[Figure 566]<br><br>[Figure 567]<br><br>[Figure 568]<br><br>[Figure 569]<br><br>[Figure 570]<br><br>[Figure 571]<br><br>[Figure 572]<br><br>[Figure 573]<br><br>[Figure 574]<br><br>[Figure 575]<br><br>[Figure 576]<br><br>[Figure 577]<br><br>[Figure 578]<br><br>[Figure 579]<br><br>[Figure 580]<br><br>[Figure 581]<br><br>[Figure 582]<br><br>[Figure 583]<br><br>[Figure 584]<br><br>[Figure 585]<br><br>[Figure 586]<br><br>[Figure 587]<br><br>[Figure 588]<br><br>[Figure 589]<br><br>[Figure 590]<br><br>[Figure 591]<br><br>[Figure 592]<br><br>[Figure 593]<br><br>[Figure 594]<br><br>[Figure 595]<br><br>[Figure 596]<br><br>[Figure 597]<br><br>[Figure 598]<br><br>[Figure 599]<br><br>[Figure 600]<br><br>[Figure 601]<br><br>[Figure 602]<br><br>[Figure 603]<br><br>[Figure 604]<br><br>[Figure 605]<br><br>[Figure 606]<br><br>[Figure 607]<br><br>[Figure 608]<br><br>[Figure 609]<br><br>[Figure 610]<br><br>[Figure 611]<br><br>[Figure 612]<br><br>[Figure 613]<br><br>[Figure 614]<br><br>[Figure 615]<br><br>[Figure 616]<br><br>[Figure 617]<br><br>[Figure 618]<br><br>[Figure 619]<br><br>[Figure 620]<br><br>[Figure 621]<br><br>[Figure 622]<br><br>[Figure 623]<br><br>[Figure 624]<br><br>[Figure 625]<br><br>[Figure 626]<br><br>[Figure 627]<br><br>[Figure 628]<br><br>[Figure 629]<br><br>[Figure 630]<br><br>[Figure 631]<br><br>[Figure 632]<br><br>[Figure 633]<br><br>[Figure 634]<br><br>[Figure 635]<br><br>[Figure 636]<br><br>[Figure 637]<br><br>[Figure 638]<br><br>[Figure 639]<br><br>[Figure 640]<br><br>[Figure 641]<br><br>[Figure 642]<br><br>[Figure 643]<br><br>[Figure 644]<br><br>[Figure 645]<br><br>[Figure 646]<br><br>[Figure 647]<br><br>[Figure 648]<br><br>[Figure 649]<br><br>[Figure 650]<br><br>[Figure 651]<br><br>[Figure 652]<br><br>[Figure 653]<br><br>[Figure 654]<br><br>[Figure 655]<br><br>[Figure 656]<br><br>[Figure 657]<br><br>[Figure 658]<br><br>[Figure 659]<br><br>[Figure 660]<br><br>[Figure 661]<br><br>[Figure 662]<br><br>[Figure 663]<br><br>[Figure 664]<br><br>[Figure 665]<br><br>[Figure 666]<br><br>[Figure 667]<br><br>[Figure 668]<br><br>[Figure 669]<br><br>[Figure 670]<br><br>[Figure 671]<br><br>[Figure 672]<br><br>[Figure 673]<br><br>[Figure 674]<br><br>[Figure 675]<br><br>[Figure 676]<br><br>[Figure 677]<br><br>[Figure 678]<br><br>[Figure 679]<br><br>[Figure 680]<br><br>[Figure 681]<br><br>[Figure 682]<br><br>[Figure 683]<br><br>[Figure 684]<br><br>[Figure 685]<br><br>[Figure 686]<br><br>[Figure 687]<br><br>[Figure 688]<br><br>[Figure 689]<br><br>[Figure 690]<br><br>[Figure 691]<br><br>[Figure 692]<br><br>[Figure 693]<br><br>[Figure 694]<br><br>[Figure 695]<br><br>[Figure 696]<br><br>[Figure 697]<br><br>[Figure 698]<br><br>[Figure 699]<br><br>𝑽(𝒄𝒐𝒍𝒐𝒓)|
|---|

[Figure 700]

[Figure 701]

Shallow M

[Figure 702]

[Figure 703]

| |
|---|

| |
|---|

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

(3) progressive radius (4) focus mode

(a) Avatar Generation (b) Progressive High-Resolution Generation

- Figure 2: The overview of AvatarVerse. Our network takes a text prompt and DensePose signal as input to optimize an explicit NeRF via a DensePose-COCO pre-trained ControlNet. We use strategies including progressive grid, progressive radius, and focus mode to generate high-resolution and high-quality 3D avatars.

#### 3.1. Preliminaries

blend weights W ∈ RN×K. The body vertex vo under the observation pose is

(1) Score Distillation Sampling, first proposed by DreamFusion (Poole et al. 2022), distills the prior knowledge from a pretrained diffusion model ϵϕ into a differentiable 3D representation θ. Given a rendered image x = g(θ) from the differentiable NeRF model g, we add random noise ϵ to obtain a noisy image. SDS then calculates the gradients of parameter θ by minimizing the difference between the predicted noise ϵϕ (xt;y,t) and the added noise ϵ:

K

wkGk (ξ,jk), (4)

vo =

k=1

where wk is the skinning weight, Gk (ξ,jk) is the affine deformation transforms the k-th joint jk from canonical space to the observation space.

(3) DensePose (G¨uler, Neverova, and Kokkinos 2018) is a pioneering technique that facilitates the establishment of dense correspondences between a 2D image and a 3D, surface-based model of the human body. Leveraging the SMPL model (Loper et al. 2015), DensePose can assign each triangular face within the SMPL mesh to one of the 24 pre-defined body parts. This correspondence allows for the generation of part-labeled 2D body images from any given viewpoint by rendering the associated regions from the SMPL mesh.

∂x ∂θ

, (1)

∇θLSDS (ϕ,xθ) = Et,ϵ w(t)(ϵϕ (zt;y,t) − ϵ)

where zt denotes the noisy image at noise level t, w(t) is a weighting function that depends on the noise level t and the text prompt y.

(2) SMPL (Bogo et al. 2016; Loper et al. 2015) is a 3D parametric human body model. It contains 6,890 body vertices and 24 keypoints. By assembling pose parameters ξ ∈ RK×3 and body shape parameter β ∈ R10, the 3D SMPL model can be represented by:

#### 3.2. DensePose SDS Loss

Prior research (Lin et al. 2022; Poole et al. 2022) predominantly employs supplementary text prompts, such as “front view” or “overhead view”, to enhance view consistency. However, reliance solely on text prompts proves inadequate for accurately conditioning a 2D diffusion model on arbitrary views. This inadequacy engenders instability in 3D model synthesis, giving rise to issues like the Janus problem. As a solution, we propose the utilization of DensePose (G¨uler, Neverova, and Kokkinos 2018) as a more robust control signal, as depicted in Figure 2.

T(β,ξ) = T¯ + BS(β) + BP(ξ) (2)

M(β,ξ) = LBS(T(β,ξ),J(β),ξ,W), (3) where T(β,ξ) denotes the non-rigid deformation combining the mean template shape T¯ from the canonical space, the shape-dependent deformations BS(β) ∈ RN×3 and the pose-dependent deformations BP(ξ) ∈ RN×3. LBS(·) represents the linear blend skinning function corresponding to articulated deformation. It maps T(β,ξ) based on the corresponding keypoint positions J(β) ∈ RN×3, pose ξ and

[Figure 741]

- Figure 3: Qualitative results of our DensePose-conditioned ControlNet. (a) 10 generated images controlled by DensePose with varying viewpoints and body parts. (b) 10 corresponding images with the same viewpoints controlled by human pose (Openpose) signals. It often fails to generate the backside of the avatar (4-th (b)) and struggles with part generation (the last two columns). (c) non-skin-tight generation results in both realistic and fictional avatars.

We choose DensePose as the condition because it delivers precise localization of 3D body parts in 2D images, affording intricate details and boundary conditions that may be overlooked by skeletal or other types of conditions. Notably, it exhibits resilience in challenging scenarios, facilitating accurate control even when body parts are partially concealed.

We first train a ControlNet (Zhang and Agrawala 2023) conditioned by DensePose part-labeled annotations using the DeepFashion (Liu et al. 2016) dataset. Figure 3 illustrates the capabilities of our ControlNet in generating highquality view-consistent images, including various viewpoints and body parts such as full body, legs, head, and more. Given a specific camera viewpoint and pose P, we generate the DensePose condition image c by rendering the partlabeled SMPL model with the corresponding pose P. The conditioned SDS loss is shown in the following equation:

∂x ∂θ

∇θLP-SDS (ϕ, x = g(θ, P)) = Et,ϵ w(t) (ϵˆ− ϵ)

(5)

ϵˆ= ϵϕ (zt;y,t,c = h(SMPL,P)) (6)

Here, g and h represent the NeRF render function and SMPL render function, respectively. The NeRF model and the SMPL pose model share identical camera viewpoints. This alignment of viewpoints enables coherent and consistent representations between the scene captured by NeRF and the corresponding human pose modeled by SMPL, allowing for better avatar generation. Our DensePoseconditioned ControlNet can generate various non-skin-tight realistic and fictional avatars as shown in Figure 3 (c).

#### 3.3. Progressive High-Resolution Generation

Previous studies commonly apply SDS loss over the entire body, such global guidance often fails to produce highquality details, especially for areas like hands, face, etc. These approaches lack effective guidance mechanisms to ensure the generation of high-quality, detailed geometry and

textures. To address this limitation, we propose a variety of guidance strategies aimed at promoting the generation of accurate and detailed representations, including progressive grid, focus mode, and progressive radius.

Progressive grid Progressive training strategy is commonly used in 2d generation and 3d reconstruction method (Karras et al. 2019; Liu et al. 2020; Sun, Sun, and Chen 2021), while we find it critical in our method for neat and efficient 3d avatar generation. We set a predetermined number of voxels Nv as the final model resolution and double the voxel number after certain steps of optimization. The voxel size sv is updated accordingly. During the early stage of training, we only need to generate a rough avatar shape. By allocating fewer grids, we can reduce the learning space and minimize floating artifacts. This strategy enables a gradual refinement of the avatar throughout the optimization process, allowing the model to adaptively allocate computational resources.

Also, the early stage of NeRF optimization is dominated by free space (i.e., space with low density). Motivated by this fact, we aim to find the areas of coarse avatar and allocate computational and memory resources to these important regions. To delineate the targeted area, we employ a density threshold to filter the scene and use a bounding box (bbox) to tightly enclose this area.

Let dx, dy, dz represent the lengths of the tightened bbox, he voxel size can be computed as sv = 3 dx×Ndy×dz

. By shrinking the lengths of the bbox, the voxel size decreases, enabling high-resolution and more voxel around the avatar. This would enhance the model’s ability to capture and model intricate details, such as fine-grained body contours, facial features, and clothing folds.

v

Progressive Radius Let pg ckpt be the set of checkpoint steps. When reaching the training step in pg ckpt, we decrease the radius of the camera by 20%. This allows for gradual rendering of finer details stage by stage. By ap-

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

(a)

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

- (b)

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

- (c)

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

DreamFusion DreamAvatar DreamWaltz Ours DreamHuman Ours

- Figure 4: Qualitative comparisons with four SOTA methods. We show several non-cherry-picked results generated by AvatarVerse. Our method generates higher-resolution details and maintains a fine-grained geometry compared with other methods. (a): ”Spiderman”; ” a man wearing a white tanktop and shorts”, (b): ”Joker”; ”a karate master wearing a Black belt”, (c): ”Stormtrooper”; ”a Roman soldier wearing his armor”.

plying the conditioned SDS loss to smaller regions of the avatar, the model can capture and emphasize intricate features, ultimately producing more realistic and visually appealing outputs.

Focus Mode Similarly, to generate better intricacy in specific body parts, we introduce a focus mode (as illustrated in Fig. 2 (b)) during both the coarse stage and fine stage. Thanks to the SMPL prior, we can easily compute the raw body parts positions for any given pose. By placing the camera close to important body parts, loss calculation can be performed in a very small avatar region with 512 × 512 resolution. Owing to the stable performance of our DensePose ControlNet, as shown in Fig. 2, partial body can be generated without additional computational resources. Focus mode can thus facilitate the creation of high-quality avatar details.

Mesh Refinement To render fine-grained high-resolution avatars within reasonable memory constraints and computation budgets, we further incorporate deformable tetrahedral grids (Lin et al. 2022; Shen et al. 2021) to learn textured 3D meshes of the generated avatars. Similar to (Lin et al. 2022), we use the trained explicit NeRF as the initialization for the mesh geometry, and optimize the mesh via backpropagation using the DensePose conditioned SDS gradient (Eq. 5).

#### 3.4. Avatar Surface Smoothing

Maintaining a globally coherent avatar shape for explicit grids during optimization can be challenging due to the high

degree of freedom and lack of spatial coherence. Individual optimization of each voxel point limits information sharing across the grid, resulting in a less smooth surface for the generated avatar and some local minima.

To address this problem, we follow the definition of the Gaussian convolution G in (Wu et al. 2022) and include a modified smoothness regularization formulated as:

Lsmooth(V ) = ∥G (V,kg,σg) − V ∥22 (7)

Here, kg represents the kernel size, and σg represents the standard deviation. We apply this smoothness term to the gradient of the density voxel grid, resulting in a gradient smoothness loss Lsmooth(∇V (density)). This encourages a smoother surface and mitigates the presence of noisy points in the free space. The overall loss of our approach is defined as follows, with λ representing the smoothness coefficient:

L = LP-SDS + λ ∗ Lsmooth(V ) (8)

### 4. Experiments

In this section, we illustrate the effectiveness of our proposed method. We demonstrate the efficacy of each proposed strategy and provide a detailed comparison against recent state-of-the-art methods.

#### 4.1. Implementation Details

We follow (Sun, Sun, and Chen 2021) to implement the explicit NeRF in our method. For each text prompt, we train

AvatarVerse for 5000 and 4000 iterations in the coarse stage and mesh refinement stage, respectively. The whole generation process takes around 2 hours on one single NVIDIA A100 GPU. We include initialization, densepose training and progressive high-resolution generation details in this section. For more comprehensive experiment details, we refer the reader to our Supplementary Material.

Initialization To aid in the early stages of optimization, we adopt a technique inspired by (Poole et al. 2022) and introduce a small ellipsoidal density ”blob” around the origin. The dimensions of the ”blob” in the XYZ axes are determined based on the range of coordinates in the SMPL pose model. Furthermore, we incorporate additional SMPLderived density bias (Cao et al. 2023) to facilitate avatar generation.

DensePose Training We annotate the DeepFashion dataset (Liu et al. 2016) using a pretrained DensePose (G¨uler, Neverova, and Kokkinos 2018) model, resulting in over 800K image pairs. The ControlNet is trained using these image pairs with BLIP2-generated text prompt (Li et al. 2023a). The diffusion model employed in our approach is SD1.5.

Progressive High-Resolution Generation For the progressive grid, we double the number of voxels at 500, 1500, and 2000 iterations at the coarse stage. After 3000 steps in the coarse stage, we shrink the bounding box to the region where the density exceeds 0.1. Our progressive radius consists of three stages, where the camera radius ranges from 1.4 to 2.1, 1 to 1.5, and 0.8 to 1.2 respectively. We reduce the radius at 1000 and 2000 iterations across both stages. Our focus mode starts from the 1000th step in the coarse stage and is consistently employed throughout the mesh refinement phase.

#### 4.2. Qualitative Results

Comparison with SOTA methods We present qualitative comparisons with DreamFusion (Poole et al. 2022), DreamAvatar (Cao et al. 2023), DreamWaltz (Huang et al. 2023), and DreamHuman (Kolotouros et al. 2023) in Fig. 4. Our method consistently outperforms these approaches in terms of both geometry and texture quality. The surface of the avatars generated by our method is exceptionally clear, owing to our progressive high-resolution generation strategy. In comparison to DreamHuman, the avatars produced by our method exhibit a richer array of details across all cases, encompassing skin, facial features, clothing, and more.

Flexible Avatar Generation In Fig. 5, we demonstrate the capability of our method in generating 3D partial avatars, which is not achievable by other existing methods due to the absence of the DensePose control. Our method enables the partial generation by directly modifying the input DensePose signal, eliminating the need for additional descriptive information such as ”The head of...” or ”The upper body of...”. This allows us to generate partial avatars of various types thanks to the attached semantics, including fullbody, half-body, head-only, hand-only, and more. Additionally, our AvatarVerse is capable of generating avatars in var-

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

- (a)
- (b)

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

Figure 5: Flexible Avatar Generation. (a) Partial Generation. All results are generated with the same text prompt ”Stormtrooper” and ”Batman”. (b) Arbitrary Pose Generation.

ious poses, showcasing our stable control over view consistency.

#### 4.3. User Study

Preference between different methods

DreamFusion

0.5%

DreamAvatar

1.5%

DreamWaltz

13.0%

Ours 85.0%

DreamHuman 19.0%

Ours 81.0%

0% 20% 40% 60% 80% 100%

Figure 6: Quantitative results of user study.

To further assess the quality of our generated 3D avatars, we conduct user studies comparing the performance of our results with four SOTA methods under the same text prompts. We randomly select 30 generated outcomes (presented as rendered rotating videos) and ask 16 volunteers to vote for their favorite results based on geometry and texture quality. In Fig. 6, we compare AvatarVerse with DreamFusion (Poole et al. 2022), DreamAvatar (Cao et al. 2023), and DreamWaltz (Huang et al. 2023), demonstrating a significant preference for our method over the other three approaches.

We also compare our method with DreamHuman (Kolotouros et al. 2023) in terms of realistic human. A remarkable 81% of volunteers voted in favor of our AvatarVerse.

#### 4.4. Ablation Study

Effectiveness of Progressive Strategies To evaluate the design choices of AvatarVerse, we conduct an ablation study

on the effectiveness of b) the progressive grid, c) the progressive radius, d) the focus mode, and e) the mesh refinement. We sequentially add these components and report the results in Fig. 7. The initial result lacks detail (e.g., no sword in the back, no armguards) and exhibits numerous floating artifacts. The overall quality is blurry and unclear. Upon incorporating the progressive grid, more voxels are gathered around the avatar region, this introduces more details into the avatar. By progressively narrowing the camera distance, the model can leverage the detail inherent in the latent diffusion, thereby eliminating a large number of floating artifacts and enhancing local details, such as the sword in the back. The focus mode further zooms in and utilizes a resolution of 512 × 512 to target and optimize certain body parts, generating high-definition and intricate local details. The mesh refinement further optimize 3D mesh of the coarse avatar, resulting in finer avatar texture.

[Figure 790]

[Figure 791]

+ prog. grid + prog. rad. + focus mode + mesh refinement

(a) (b) (c) (d) (e)

Figure 7: Impact of progressive strategies. (a) none progressive strategy; (b) add progressive grid; (c) add progressive radius upon (b); (d) add focus mode upon (c); (e) add mesh refinement, our full method.

Effectiveness of DensePose Control Figure 8 illustrates the influence of various control signals. When conditioned by the skeleton, the model can generate avatars that more closely resemble human figures. However, the avatar’s edges appear blurry and still face severe Janus problem. By incorporating DensePose control into our framework, we achieve more precise avatar boundaries, intricate details, and stable avatar control, resulting in a substantial improvement in the overall quality and appearance of the generated avatars.

Effectiveness of Surface Smoothing Avatar surface smoothing plays a critical role in the AvatarVerse framework, as it guarantees the generated avatars exhibit compact geometry and smooth surfaces. As shown in Figure 9, by finding a balance between the smooth loss and the conditioned SDS loss, the visual quality and realism of the avatars

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

(a)w/ocontrol(b)skeleton(c)DensePose

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

- Figure 8: Impact of control signal. (a) without additional control; (b) with skeleton control; (c) with our DensePose control. For each type, we show the RGB, normal, depth, and the corresponding control signal. are greatly improved.

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

- (a) w/o surface smoothing
- (b) w/ surface smoothing

- Figure 9: Impact of surface smoothing strategy. (a) without surface smoothing; (b) with surface smoothing. Results are generated with the same text prompt.

### Conclusion

In this paper, we introduce AvatarVerse, a novel framework designed to generate high-quality and stable 3D avatars from textual prompts and poses. By employing our trained DensePose-conditioned ControlNet, we facilitate stable partial or full-body control during explicit NeRF optimization. Our 3D avatar outcomes exhibit superior texture and geometry quality, thanks to our progressive high-resolution generation strategy. Furthermore, the generated avatars are easily animatable through skeletal binding, as they exhibit high alignment with the joints of the SMPL model. Through comprehensive experiments and user studies, we demonstrate that our AvatarVerse significantly outperforms previous and contemporary approaches. We believe that our approach renews the generation of high-quality 3D avatars in the neural and prompt-interaction era.

### References

Bogo, F.; Kanazawa, A.; Lassner, C.; Gehler, P.; Romero, J.; and Black, M. J. 2016. Keep It SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image. ArXiv, abs/1607.08128.

Cao, Y.; Cao, Y.-P.; Han, K.; Shan, Y.; and Wong, K.-

- Y. K. 2023. DreamAvatar: Text-and-Shape Guided 3D Human Avatar Generation via Diffusion Models. ArXiv,

- abs/2304.00916.

G¨uler, R. A.; Neverova, N.; and Kokkinos, I. 2018. DensePose: Dense Human Pose Estimation in the Wild. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7297–7306.

Han, X.; Cao, Y.; Han, K.; Zhu, X.; Deng, J.; Song, Y.-Z.; Xiang, T.; and Wong, K.-Y. K. 2023. HeadSculpt: Crafting 3D Head Avatars with Text. ArXiv, abs/2306.03038.

Hong, F.; Zhang, M.; Pan, L.; Cai, Z.; Yang, L.; and Liu, Z. 2022. AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars. ACM Trans. Graph., 41: 161:1– 161:19.

Huang, Y.; Wang, J.; Zeng, A.; Cao, H.; Qi, X.; Shi, Y.; Zha,

- Z.; and Zhang, L. 2023. DreamWaltz: Make a Scene with Complex 3D Animatable Avatars. ArXiv, abs/2305.12529.

Isik, M.; R¨unz, M.; Georgopoulos, M.; Khakhulin, T.; Starck, J.; de Agapito, L.; and Nießner, M. 2023. HumanRF: High-Fidelity Neural Radiance Fields for Humans in Motion. ArXiv, abs/2305.06356.

Jain, A.; Mildenhall, B.; Barron, J. T.; Abbeel, P.; and Poole, B. 2021. Zero-Shot Text-Guided Object Generation with Dream Fields. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 857–866.

Jiang, R.; Wang, C.; Zhang, J.; Chai, M.; He, M.; Chen, D.; and Liao, J. 2023. AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control. ArXiv, abs/2303.17606.

Jiang, T.; Chen, X.; Song, J.; and Hilliges, O. 2022. InstantAvatar: Learning Avatars from Monocular Video in 60 Seconds. ArXiv, abs/2212.10550.

Karras, T.; Laine, S.; Aittala, M.; Hellsten, J.; Lehtinen, J.; and Aila, T. 2019. Analyzing and Improving the Image Quality of StyleGAN. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 8107–8116.

Khalid, N. M.; Xie, T.; Belilovsky, E.; and Popa, T. 2022. CLIP-Mesh: Generating textured meshes from text using pretrained image-text models. SIGGRAPH Asia 2022 Conference Papers.

Kolotouros, N.; Alldieck, T.; Zanfir, A.; Bazavan, E. G.; Fieraru, M.; and Sminchisescu, C. 2023. DreamHuman: Animatable 3D Avatars from Text. ArXiv, abs/2306.09329.

Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023a. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. In ICML.

Li, Z.; Zheng, Z.; Liu, Y.; Zhou, B.; and Liu, Y. 2023b. PoseVocab: Learning Joint-structured Pose Embeddings for Human Avatar Modeling. ArXiv, abs/2304.13006.

Lin, C.-H.; Gao, J.; Tang, L.; Takikawa, T.; Zeng, X.; Huang,

- X.; Kreis, K.; Fidler, S.; Liu, M.-Y.; and Lin, T.-Y. 2022. Magic3D: High-Resolution Text-to-3D Content Creation. ArXiv, abs/2211.10440. Liu, L.; Gu, J.; Lin, K. Z.; Chua, T.-S.; and Theobalt, C.

- 2020. Neural Sparse Voxel Fields. ArXiv, abs/2007.11571. Liu, Z.; Luo, P.; Qiu, S.; Wang, X.; and Tang, X. 2016. DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1096– 1104. Loper, M.; Mahmood, N.; Romero, J.; Pons-Moll, G.; and Black, M. J. 2015. SMPL: a skinned multi-person linear model. ACM Trans. Graph., 34: 248:1–248:16. Metzer, G.; Richardson, E.; Patashnik, O.; Giryes, R.; and Cohen-Or, D. 2022. Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures. arXiv preprint arXiv:2211.07600. Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. ArXiv, abs/2003.08934. Poole, B.; Jain, A.; Barron, J. T.; and Mildenhall, B. 2022. DreamFusion: Text-to-3D using 2D Diffusion. ArXiv, abs/2209.14988. Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models From Natural Language Supervision. In International Conference on Machine Learning. Richardson, E.; Metzer, G.; Alaluf, Y.; Giryes, R.; and Cohen-Or, D. 2023. TEXTure: Text-Guided Texturing of 3D Shapes. ArXiv, abs/2302.01721. Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2021. High-Resolution Image Synthesis with Latent Diffusion Models. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10674– 10685. Saharia, C.; Chan, W.; Saxena, S.; Li, L.; Whang, J.; Denton, E. L.; Ghasemipour, S. K. S.; Ayan, B. K.; Mahdavi, S. S.; Lopes, R. G.; Salimans, T.; Ho, J.; Fleet, D. J.; and Norouzi, M. 2022. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. ArXiv, abs/2205.11487. Sanghi, A.; Chu, H.; Lambourne, J.; Wang, Y.; Cheng, C.-

Y.; and Fumero, M. 2021. CLIP-Forge: Towards Zero-Shot Text-to-Shape Generation. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 18582– 18592.

Shen, T.; Gao, J.; Yin, K.; Liu, M.-Y.; and Fidler, S.

- 2021. Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis. ArXiv, abs/2111.04276. Sun, C.; Sun, M.; and Chen, H.-T. 2021. Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5449–5459.

Wang, C.; Chai, M.; He, M.; Chen, D.; and Liao, J. 2021. Cross-Domain and Disentangled Face Manipulation With 3D Guidance. IEEE Transactions on Visualization and Computer Graphics, 29: 2053–2066.

Wang, L.; Zhao, X.; Sun, J.; Zhang, Y.; Zhang, H.; Yu, T.; and Liu, Y. 2023a. StyleAvatar: Real-time Photo-realistic Portrait Avatar from a Single Video. ArXiv, abs/2305.00942. Wang, Z.; Lu, C.; Wang, Y.; Bao, F.; Li, C.; Su, H.; and Zhu, J. 2023b. ProlificDreamer: High-Fidelity and Diverse Textto-3D Generation with Variational Score Distillation. ArXiv,

- abs/2305.16213.

Wu, T.; Wang, J.; Pan, X.; Xu, X.; Theobalt, C.; Liu, Z.; and Lin, D. 2022. Voxurf: Voxel-based Efficient and Accurate Neural Surface Reconstruction. ArXiv, abs/2208.12697.

Xiu, Y.; Yang, J.; Cao, X.; Tzionas, D.; and Black, M. J. 2022. ECON: Explicit Clothed humans Obtained from Normals. ArXiv, abs/2212.07422.

Zhang, L.; and Agrawala, M. 2023. Adding Conditional Control to Text-to-Image Diffusion Models. ArXiv, abs/2302.05543.

Zhang, L.; Qiu, Q.; Lin, H.; Zhang, Q.; Shi, C.; Yang, W.; Shi, Y.; Yang, S.; Xu, L.; and Yu, J. 2023. DreamFace: Progressive Generation of Animatable 3D Faces under Text Guidance. ArXiv, abs/2304.03117.

Zheng, Z.; Zhao, X.; Zhang, H.; Liu, B.; and Liu, Y. 2023. AvatarReX: Real-time Expressive Full-body Avatars. ArXiv, abs/2305.04789.

