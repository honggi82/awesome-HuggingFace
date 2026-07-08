# arXiv:2402.12712v3[cs.CV]30Apr2024

## MVDiffusion++: A Dense High-resolution Multi-view Diffusion Model for Single or Sparse-view 3D Object Reconstruction

Shitao Tang1∗, Jiacheng Chen1∗, Dilin Wang2∗, Chengzhou Tang2, Fuyang Zhang1, Yuchen Fan2, Vikas Chandra2, Yasutaka Furukawa1†, and Rakesh Ranjan2†

1 Simon Fraser University {shitaot, jca348, fuyangz, furukawa}@sfu.ca 2 Meta Reality Labs {wdilin, chengzhout, ycfan, vchandra, rakeshr}@meta.com

Abstract. This paper presents a neural architecture MVDiffusion++ for 3D object reconstruction that synthesizes dense and high-resolution views of an object given one or a few images without camera poses. MVDiffusion++ achieves superior flexibility and scalability with two surprisingly simple ideas: 1) A “pose-free architecture” where standard self-attention among 2D latent features learns 3D consistency across an arbitrary number of conditional and generation views without explicitly using camera pose information; and 2) A “view dropout strategy” that discards a substantial number of output views during training, which reduces the training-time memory footprint and enables dense and high-resolution view synthesis at test time. We use the Objaverse for training and the Google Scanned Objects for evaluation with standard novel view synthesis and 3D reconstruction metrics, where MVDiffusion++ significantly outperforms the current state of the arts. We also demonstrate a text-to-3D application example by combining MVDiffusion++ with a text-to-image generative model. The project page is at https://mvdiffusion-plusplus.github.io.

### 1 Introduction

Human vision demonstrates remarkable flexibility. Look at the images of objects at the left in Figure 1. While unable to create millimeter-accurate 3D models, our visual system can combine information from a few images to form a coherent 3D representation in our minds, including intricate facial features of a tiger or the arrangement of blocks forming a toy train, even parts that are fully obscured.

3D reconstruction technology [1, 7, 32, 45] has evolved over the last fifteen years in a fundamentally different way. Unlike the human ability to infer 3D shapes from a few images, the technology takes hundreds of images of an object,

*Equal contribution. †Joint last author. Work done during an internship with Meta.

[Figure 1]

[Figure 2]

[Figure 3]

###### 2 Tang et al.

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

###### Input images (pose-free) Dense, high-resolution generation 3D model

Fig. 1: MVDiffusion++ generates dense(32) and high-resolution(512×512) images of an object from a single or multiple unposed images. The input images of the three examples are from a latent diffusion model, OmniObject3D[41], and Google Scanned Objects[5], respectively.

estimates their precise camera parameters, and reconstructs high-fidelity 3D geometry at a sub-millimeter accuracy.

This paper explores a new paradigm of 3D reconstruction that combines the high-fidelity of computational methods and the flexibility of human visual systems. Our inspiration comes from exciting recent developments in multiview image generative models [15, 18, 19, 30, 31, 33, 37]. MVDiffusion [33] is an early attempt to extend pre-trained image diffusion models [27] to a multiview generative system, when pixel correspondences across views are available (e.g., generating perspective images to form a panorama). MVDream [31] and Wonder3D [19] further extend to more general settings where generated images yield 3D reconstruction via techniques such as NeRF [21] or NeuS [36].

This paper pushes the frontier of multi-view diffusion models towards flexible and high-fidelity 3D reconstruction systems. Concretely, the paper presents MVDiffusion++, a novel approach to generate dense (32) and high-resolution (512×512) images of an object, conditioned with single or sparse input views without camera poses, whose reliable estimation is difficult due to minimal or no visual overlaps. Standard 3D reconstruction techniques turn generated images

into a 3D model. Two simple ideas are at the heart of our method. First, we leverage a latent diffusion inpainting model with conditional and generation branches, where self-attention among 2D features learns 3D consistency without using camera poses or image projection formula. Second, we introduce “view dropout” training strategy, which randomly excludes generation views in each batch, enabling the use of high-resolution images during training. During testing, this simple approach surprisingly generates high-quality, dense views for all the images simultaneously.

MVDiffusion++ achieves state-of-the-art performance on the task of novel view synthesis, single-view reconstruction, and sparse-view reconstruction. For single-view reconstruction, our method achieves 0.6973 IoU and 0.0165 Chamfer distance on the Google Scanned Objects dataset, higher than SyncDreamer [18] by 0.1552 in terms of Vol. IOU. For novel view synthesis in sparse view setting, MVDiffusion++ improves the PSNR by 8.19 compared with a recent pose-free view synthesis method, LEAP [12]. Lastly, we demonstrate applications in textto-3D by combining MVDiffusion++ with a text-to-image generative model.

### 2 Related work

This paper presents a multi-view image generative model for object reconstruction, given one or a few condition images. The section reviews related work on multiview image generation and single to sparse-view 3D reconstruction techniques.

Multi-view image generation. The evolution of text-to-image diffusion models has paved the way for multi-view image generation. MVDiffusion [33] introduces an innovative multi-branch Unet architecture for denoising multi-view images simultaneously. This approach, however, is constrained to cases with one-to-one image correspondences. Syncdreamer [18] uses 3D volumes and depth-wise attention for maintaining multi-view consistency. MVDream [31] takes a different path, incorporating 3D self-attention to extend the work to more general cases. Similarly, Wonder3D [19] and Zero123++ [30] apply 3D self-attention to singleimage conditioned multi-view image generation. These methods, while innovative, tend to produce sparse, low-resolution images due to the computational intensity of the attention mechanism. In contrast, our framework represents a more versatile solution capable of generating dense, high-resolution multi-view images conditioned on an arbitrary number of images.

Single view reconstruction. Single View Image Reconstruction is an active research area [18, 19, 22, 39, 42, 43], driven by the advancements of generative models [18, 19, 22, 39]. Large reconstruction model [11] and DMV3D [42] predict triplanes from a single image, but the 3D volume limits its resolutions. The other method, Syncdreamer [18] generates multi-view images with a latent diffusion model by constructing a cost volume. These images are then used to recover

###### 3D structures using conventional reconstruction methods like Neus. However, this process requires substantial GPU memory, limiting it to low resolutions. Similarly, Wonder3D faces challenges due to the computational demands of self-

attention, leading to similar restrictions. In contrast, our approach introduces a ”view dropout” technique, which randomly samples a limited number of views for training in each iteration. This enables our model to generate a variable number of high-resolution images while employing full 3D self-attention, effectively addressing the limitations faced by existing methods.

Sparse view reconstruction. Sparse View Image Reconstruction (SVIR) [12, 44] is a challenging task where only a limited number of images, typically two to ten, are given. Traditional 3D reconstruction methods estimate camera poses first, then perform dense reconstruction using techniques such as multi-view stereo [32, 45] or NeRF [36]. However, camera pose estimation is difficult for SVIR, where visual overlaps are none to minimal. To address this, FvOR [44] optimizes camera poses and shapes jointly. LEAP [12] along with PF-LRM [38] highlight the issues of noisy camera poses and suggest a pose-free approach. However, they are not based on generative models, lacking generative priors, and suffer from low-resolution outputs due to the use of volume rendering. In contrast, our method employs a diffusion model to generate high-resolution multi-view images directly, then a reconstruction system Neus [36] to recover a mesh model.

### 3 Preliminary: Multi-view latent diffusion models

MVDiffusion [33] is a multi-view latent diffusion model [18, 30, 31, 33], generating multiple images given a text or an image, when pixel-wise correspondences are available across views. MVDiffusion is the foundation of the proposed approach, where the section reviews its architecture and introduces notations (See Figure 3).

For generating eight perspective views forming a panorama, eight latent diffusion models (LDM) denoise eight noisy latent images {Z1(t),Z2(t),···Z8(t)} simultaneously. A UNet is the core of a LDM model, consisting of a sequence of blocks through the four levels of the feature pyramid.

Let Ubi denote the feature image of i-th image at b-th block. A CNN initializes an input Ui0 from Zi(t) at the first block. Each UNet block has four network modules. The first is a novel correspondence-aware attention (CAA), enforcing consistency across views with visual overlaps: The left/right neighboring images (Uib−1,Uib+1) for panorama. The remaining three modules are from the original: 1) Self-attention (SA) layers; 2) Cross-attention (CA) layers from the condition with the CLIP embedding; and 3) CNN layers with the pixel-wise concatenation of a positional encoding of time τ(t). At test time, a standard DDPM sampler [10] updates all noisy latents with the predicted noise from the last CNN layer. The training objective is defined as follows by omitting the conditions for notation simplicity, where ϵi is a Gaussian and ϵθ denotes the UNet output.

LMV LDM := E{Z

i(0)}Ni=1,{ϵi∼N(0,I)}Ni=1,t

N

∥ϵi − ϵiθ({Zi(t)},τ(t))∥22 . (1)

i=1

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

###### Input views Target views

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

Azimuth

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

###### 0° 90° 180° 270°

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

Ref. view

[Figure 121]

[Figure 122]

[Figure 123]

Azimuth=0°

- Fig.2: The input and output specification (§4.1) of MVDiffusion++. The 32 target images are defined in eight azimuths and four elevation levels. During training, our view dropout strategy (§4.3) randomly drops a substantial number of views (dashed blue) and trains the model to denoise the remaining views (red).

### 4 MVDiffusion++

MVDiffusion++ pushes the frontier of multi-view diffusion models for 3D modeling in their flexibility and scalability by generating denser and higher-resolution images given an arbitrary number of un-posed condition views. With the prevalence of Transformer models [34], high-fidelity 3D modeling would require largescale attention over dense and high-resolution image features, potentially with volumetric ones. Furthermore, 3D consistency learning is at the heart of the task, which would usually require precise image projection models and/or camera parameters. Our surprising discovery is that self-attention among 2D latent image features is all we need for 3D learning without projection models or camera parameters, and a simple training strategy would further achieve dense and high-resolution multi-view image generation. The section defines the task (i.e., input condition and output target images), then explains the two key ideas: 1) pose-free multi-view conditional diffusion model for flexibility and 2) view dropout training strategy for scalability. §5 provides the remaining system details.

###### 4.1 Task: Input condition images and output target images

The generation target is a set of dense (32) and high-resolution (512×512) images, positioned at uniform 2D grid points on a sphere along the azimuth and elevation angles for 3D object reconstruction (See Figure 2). Specifically, there are eight azimuth angles (every 45◦) and four elevation angles (every 30◦ in the range [−30◦,60◦]). Camera up-vectors are aligned with gravity, and their optical axes pass through the sphere center. Our input condition is one or a few images without camera poses, where visual overlaps are too minimal or possibly none for Structure from Motion algorithms to work reliably. The number of condition images is up to a pre-determined number, which is 10 in our experiments but can easily change. The input image resolution is 512×512. The horizontal and vertical field-of-view of both the input and output views is 60◦.

We use synthetic rendered images from 3D object databases for training and evaluations, then real-world images as the input condition for further qualitative evaluations. The task settings vary slightly between datasets, with details provided

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

MVDiffusion Block MVDiffHD Block [At first block]

[At first block]

[Figure 135]

[Figure 136]

[For each block]

[Figure 137]

[For each block]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[At last block] [At last block]

[Figure 144]

[Figure 145]

- Fig.3: The denoising architectures for MVDiffusion and MVDiffusion++for sampling multi-view images. The order of the MVDiffusion network modules is rearranged to highlight the differences (in orange) with MVDiffusion++.

in §5. Here, we explain one preprocessing step that removes ambiguity in the training task. 3D object databases such as Objaverse [4] and Google Scanned Object [5] align the Z-axis with the object up-vectors. However, the azimuth of the ground-truth object pose is ambiguous without camera poses of the condition images. Therefore, we rotate the output views to align the azimuth of the first condition and the first output image as shown by the right of Figure 2.

###### 4.2 Pose-free multi-view conditional diffusion model

MVDiffusion++ is a multi-view latent diffusion model as defined in §3, comprising of a condition branch for single or sparse-view input images and a generation branch for output images (See Figure 3 and Figure 4). Note that the condition branch shares the same architecture and is tasked to generate the condition images that are also given as guidance (i.e., a trivial task).

Diffusion process. The forward diffusion process is the same as MVDiffusion, except for the image resolution and the pre-trained VAE. Concretely, it 1) converts all 512 × 512 × 3 input/output image (Ii) with foreground masks to 64 × 64 × 4 latent images (Zi) by a fine-tuned latent diffusion VAE (denoted as MVAE, see §5 for the fine-tuning process); and 2) adds a Gaussian noise with a linear schedule, as suggested by zero-123++ [30] to each feature of Zi.

Denoising process. The denoising process is highlighted in Figure 3, where a latent diffusion UNet with a few modifications processes a noisy latent Zi(t) at each denoising step t. The UNet consists of 9 blocks of network modules over the four levels of feature pyramids on either side of the encoder/decoder. The details are explained as follows.

1

Mpos Mneg

- ⌧(t) + V1 (1)
- ⌧(t) + V2 (2)

1

Mpos Mneg

1

Mpos Mneg

- ⌧(t) + V1 (1)
- ⌧(t) + V2 (2)

- ⌧(t) + V11 (3)
- ⌧(t) + V12 (4)
- ⌧(t) + V13 (5)

1

- N1C(t)
- N2C(t)

- ⌧(t) + V1 (1)
- ⌧(t) + V2 (2)

1

1

Mpos Mneg

- N1G(t)
- N2G(t)
- N3G(t) MVAE(Ic)

- ⌧(t) + V11 (3)
- ⌧(t) + V12 (4)

Mpos Mneg

###### MVDiffusion++ 7

1

1

1

1

- N1C(t)
- N2C(t)
- 3

- N1C(t)
- N2C(t)

⌧(t) + V1 (1)

Mpos Mneg

Mpos Mneg

- ⌧(t) + V11 (3)
- ⌧(t) + V12 (4)
- ⌧(t) + V13 (5)

- Z1(t)

- Z1(t   1)

1

Mpos Mneg

- ⌧(t) + V1 (1)
- ⌧(t) + V2 (2)

- ⌧(t) + V11 (3)
- ⌧(t) + V12 (4)
- ⌧(t) + V13 (5)

- Z2(t)

- Z2(t   1)

CLIP

- ⌧(t) + sV1 (1)
- ⌧(t) + sV2 (2)

1

|[Figure 146]<br><br>[Figure 147]<br><br>1<br><br>Z2(t) Z11(t) ZZ121(t(t))|
|---|

|[Figure 148]<br><br>N1G(t)<br>N2G(t)<br>N3G(t)<br><br><br>N1C(t)<br>N2C(t)<br>|
|---|

|[Figure 149]|
|---|

|t)1+|VN211G(|1<br><br>1<br><br>t)|
|---|---|---|
|tt))1++|NMVA1G<br><br>VV131N21C(|E(1I<br><br>2<br><br>t()t)|
|t)1+V|NMVAN1GC(<br><br>12|t(E()t1)I|

1

⌧( (2)

c)

ConditionBranch

Mpos Mneg

- ⌧(t) + sV1 (1)
- ⌧(t) + sV2 (2)

###### Conv

=

=

⌧( (1)

⌧( (5)

, ,

1

white) Mpos

CA

Mpos Mneg

- 1

ZG2 (t) ZC1 (t   1) ZC2 (t   1)

- ZG1 (t   1)
- ZG2 (t   1)

- 2

⌧(t) + V11 (3)

- Z1(t)

- Z1(t   1)

1

Mpos Mneg

- ⌧(t) + V1 (1)
- ⌧(t) + V2 (2)

⌧(t) + V11 (3)

- ⌧( (4)
- ⌧( (5)

- Z2(t)

- Z2(t   1)

1

MVAE(I1)

N1G(t)

- ⌧(t) + sV1 (1)
- ⌧(t) + sV2 (2)

- ⌧(t) + sV11 (3)
- ⌧(t) + sV12 (4)
- ⌧(t) + sV13 (5)

⌧(t) + V2 (2)

NG(t)

MVAE(Ic) c)

|t)1+V|M1neg<br><br>1<br><br>NG<br><br>13|1<br><br>1<br><br>(t)|
|---|---|---|
|)1+V|⌧(1t) MVA2 12N1C|+1⌧(<br><br>E(I2 (t)|
|t)1+V|⌧⌧((1tt))<br><br>MVAN3G N2C<br><br>11|++1⌧⌧(( (E(t) I (t)|

|[Figure 150]<br><br>Z2(t) Z11(t) ZZ112(t(t))|
|---|

|[Figure 151]<br><br>N1C(t)<br>N2C(t)<br><br><br>N1G(t)<br><br>MVAE(I2) MVAE(Iwhite Mpos<br><br>N2G(t)<br>N3G(t) MVAE(I1)<br>|
|---|

|[Figure 152]<br><br>[Figure 153]<br><br>Z2(t   1)1<br><br>Z11(t   1)<br>Z12(t<br><br><br>Z11(t)<br>Z12(t) Z1(t   1)<br><br><br>Z1(t)<br>Z2(t)<br>|
|---|

- ⌧(t) + sV1 (1)
- ⌧(t) + sV2 (2)

)

Conv

=

=

V1C) V2C) V1G)

⌧(t (4)

- ⌧(t) + sV11 (3)
- ⌧(t) + sV12 (4)
- ⌧(t) + sV13 (5)

, ,

⌧( (3)

white)

CA

⌧(t) + ⌧(V2G) ⌧(t) + ⌧(V3G)

MVAE(I1c) MVAE(I2c)

1

Mpos M

Mneg

MVAE(I2) MVAE(Iwhite)

N2G(t)

- Z11(t)

- Z11(t   1)

1

Mpos Mneg

- ⌧(t) + V1 (1)
- ⌧(t) + V2 (2)

- ⌧(t) + V11 (3)
- ⌧(t) + V12 (4)
- ⌧(t) + V13 (5)

Z (t)

- Z12(t)

- Z12(t   1)

N1G(t)

⌧(t) + V13 (5)

……

- ⌧(t) + sV11 (3)
- ⌧(t) + sV12 (4)
- ⌧(t) + sV13 (5)

#### SA

…

⌧(t) + ⌧(V1C)

NC(t)

⌧(t) + V (4)

|[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>Z11(t   1)<br>Z12(t   1)<br><br><br>Z12(t)<br><br>Z1(t   1)<br>Z2(t   1)<br><br><br>1 Z2(t) Z11(t)|
|---|

|[Figure 157]<br><br>Z2(t   1)<br><br>Z11(t   1)<br>Z12(t   1)<br><br><br>Z11(t)<br>Z12(t) Z1(t   1)<br><br><br>Z1(t)<br>Z2(t)<br>|
|---|

|0|Z0C1(t<br><br>neg N2G(<br><br>12|)t)0|
|---|---|---|
|0|Z0C2(t<br><br>⌧(t) + ⌧(t) +<br><br>MVA N3G(|)0<br><br>⌧(V ⌧(V E(I<br><br>t)|
|)0+V|Z0G1(t<br><br>⌧(t) + ⌧(t) +<br><br>Mpo MVA<br><br>13|)0<br><br>⌧(V ⌧(V<br><br>s<br><br>E(I|

|N3G(t)<br><br>MVAE(I1c) MVAE(I2c)<br><br>1<br><br>N2C(t) N1G(t) G<br><br>[Figure 158]<br><br>❓<br><br>⌧(t) + ⌧(V2C) ⌧(t) + ⌧(V1G) ⌧(t) + ⌧(V2G) ⌧(t) + ⌧(V3G)<br><br>ZC1 (t)<br><br>Mpos Mneg ⌧(t) + ⌧(V C)|
|---|

GenerationBranch

C

white)

- 1 ) C
- 2 ) G

- ⌧(t) + sV11 (3)
- ⌧(t) + sV12 (4)
- ⌧(t) + sV13 (5)

Conv

- Z1(t)
- Z2(t)

=

=

, ,

CA

- 1 ) G
- 2 )

⌧(t (5)

c 1)

N2 (t)

1

⌧(t) + ⌧(V3G)

- Z1(t)
- Z2(t)

Mneg

###### Z11(t)

MVAE(Iwhite)

- Z11(t)
- Z12(t)

⌧(t) + ⌧(V2C)

###### ZG(t)

###### ZC(t)

MVAE(I2c) white)

G

|[Figure 159]<br><br>[Figure 160]<br><br>Z12(t   1)<br><br>Z1(t   1)<br>Z2(t   1) Z11(t   1)<br>|
|---|

|[Figure 161]<br><br>Z11(t   1)<br>Z12(t   1)<br><br><br>Z12(t)<br><br>Z1(t   1)<br>Z2(t   1)<br><br><br>Z1(t)<br>Z2(t) Z11(t)<br>|
|---|

|Bpos Bneg ⌧(t) + ⌧(Cr)<br><br>N3G(t)<br><br>MVAE(I1c) MVAE(I2c)<br><br>[Figure 162]<br><br>❓<br><br>2<br><br>ZG1 (t)<br>ZG2 (t) ZC(t 1)<br><br><br>⌧(t) + ⌧(V1 ) ⌧(t) + ⌧(V2G) ⌧(t) + ⌧(V3G)<br><br>ZC1 (t)<br><br>C|
|---|

|0|0<br><br>2 ZC(t<br><br>Z⌧C1(t()t|0<br>1)<br><br><br>)+ ⌧|
|---|---|---|
|0|0<br>1<br><br><br>ZC2 (t ⌧(t) ⌧(t) MVA|) 0<br><br>+ ⌧ + ⌧<br><br>E(I|
|0|Z0<br><br>C 2 (t<br><br>ZG(t ⌧(t) ⌧(t) Mpo|)++ 0⌧⌧1)<br><br>s|

(V1C) (V2C) (V1G) (V2G) (V3G)

Conv

- Z11(t)
- Z12(t)

- Z1(t   1)
- Z2(t   1)

=

=

- Z1(t)
- Z2(t)

, ,

CA

1  

Z2 (t)

⌧(t) + ⌧(Cd)

- ZG1 (t   1)
- ZG2 (t   1)

- Z1(t   1)
- Z2(t   1)

- Z11(t   1)
- Z12(t   1)

Mneg

MVAE(Iwhite)

###### Z12(t)

- Z1(t)
- Z2(t)

- Z11(t   1)
- Z12(t   1)

× 𝑁 UNet blocks

- ⌧(t) + ⌧(G1)
- ⌧(t) + ⌧(G2)
- ⌧(t) + ⌧(G3)

- Z11(t)
- Z12(t)

ZC1 (t) ZC2 (t)

ZC2 (t   1)

…

- ZG1 (t)
- ZG2 (t) ZC1 (t   1) ZC2 (t   1)

⌧(t) + ⌧(V1C) ⌧(t) + ⌧(V2C) ⌧(t) + ⌧(V1G) ⌧(t) + ⌧(V2G) ⌧(t) + ⌧(V3G)

- Z1(t   1)
- Z2(t   1)

Bpos Bneg

- ZG1 (t   1)
- ZG2 (t

- Z11(t   1)
- Z12(t   1)

- Z11(t)
- Z12(t)

- Fig.4: Illustration of the pose-free multi-view conditional diffusion model of MVDiffusion++. The model takes any number of input images and generates images at fixed viewpoints. The condition branch and generation branch have different input configurations but share the same structure and weights.

- Z1(t   1)
- Z2(t   1)

- ZG1 (t)
- ZG2 (t) ZC1 (t   1) ZC2 (t   1)

- Z11(t   1)
- Z12(t   1)

⌧(t) + ⌧(Cr) ⌧(t) + ⌧(Cd)

- Z1(t   1)
- Z2(t   1)

- Z11(t   1)
- Z12(t   1)

- ⌧(t) + ⌧(G1)
- ⌧(t) + ⌧(G2)
- ⌧(t) + ⌧(G3)

ZC1 (t) ZC2 (t)

- ZG1 (t   1)
- ZG2 (t   1)

- Z11(t   1)
- Z12(t   1)

- ZG1 (t)
- ZG2 (t) ZC1 (t   1) ZC2 (t   1)

[At first block] The UNet feature Ui0 at the first block is initialized with the concatenation of 1) the noisy latents Zi(t); 2) a constant binary mask of either 1 or 0, denoted by Mpos or Mneg to indicate the branch type (condition or generation); and 3) the condition latents (MVAE(Ii,Mi)) where we use the conditonal VAE from latent diffusion to encode the condition image (Ii) with its segmentation mask (Mi). Note that this concatenation has 9 = (4 + 4 + 1) channels, and a 1 × 1 final convolution layer reduces the channel dimension to

- ZG1 (t   1)
- ZG2 (t   1)

- ZG1 (t   1)
- ZG2 (t   1)

- 4. For a generation branch, we pass a white image as Ii and a binary image of 1 (i.e., Mpos) as Mi. For Objaverse and Google Scaned Object datasets, we use the masks provided by the datasets. Otherwise, we run segmentation to generate the masks.

[For each block] Three network modules are at the heart of the processing: 1) Global self-attention mechanism among the UNet features across all the images, learning 3D consistency; 2) Cross-attention mechanism, injecting the CLIP embedding of the condition images to all the other images through the CLIP embedding; and 3) CNN layers, process per-image features while injecting the timestep frequency encoding τ(t) and the learnable embedding of an image index Vi. For the self-attention module, we copy the network architecture and model weights and apply it across all the views. This module is inspired by MVDream [31], while the key differences in our work are 1) Scalability deployment via the view-drop training strategy in §4.3; and 2) Handling of multiple condition

images without camera poses via the network design. 42 = (32 + 10) learnable embedding vectors {Vi} are trained for 32 generation and 10 condition images, each of which is multiplied with a zero-initialized trainable scale s to avoid model disruption at the start of training.

[At last block] The output of the last UNet block yields the noise estimation, and a standard DDPM sampler [10] takes it to produce the noisy latent of the next timestep Zi(t − 1) for each sampling step. The loss function is the same as MVDiffusion. Note that the model is first trained with ϵ-prediction and then with v-prediction (See §5), where Equation 1 is the loss function for the ϵ-prediction model. The velocity [29], vi(t) = αtϵi − γtZi(0), becomes the prediction target for the v-prediction model, while αt and γt are predefined angular parameters.

###### 4.3 View dropout training strategy

MVDiffusion++ training would face a scalability challenge. 42(= 32 + 10) copies of UNet features yield more than 130k tokens, where the global self-attention mechanism becomes infeasible even with the latest memory efficient transformers for large language models [2, 3]. We propose a simple yet surprisingly effective view dropout training strategy, which completely discards a set of views across all layers during training. Specifically, we randomly drop 24 out of 32 views for each object at each training iteration, significantly reducing memory consumption at training. At test time, we run the entire architecture and generate 32 views.

### 5 Remaining system details

This section explains the remaining system details on the data preparations, the mesh extraction process, the MVAE pre-fine-tuning, and the three-stage training strategy.

###### 5.1 Training data preparation

Out of 800k 3D object models from Objaverse [4], we use 180k models whose aesthetic scores [23] are at least 5 for training. For each object 3D model, we translate the bounding box center to the origin and apply uniform scaling so that the longest dimension matches [−1,1]. The output camera centers are placed at a distance of 1.5 from the origin. Input condition views are chosen in a similar way as Zero-123 [17]. Concretely, an azimuth angle is randomly chosen from one of the eight discrete angles of the output cameras (also see §4.1). The elevation angle is set randomly from [-10◦, 45◦]. The distance of the camera center from the origin is set randomly from [1.5, 2.2]. We use Blender to render images.

###### 5.2 Testing data preparation

Single-view cases. Google Scanned Object (GSO) [5] is our testing dataset, where we borrow the rendered images and the evaluation pipeline from SyncDreamer [18]. Concretely, the test set consists of 30 objects. Each object has 16

images with a fixed elevation of 30◦ and every 22.5◦ for azimuth. SyncDreamer selected condition images by “visual plausibility”, which we copy. Since the azimuth angles in our training setting are every 45◦, eight images (starting from and including the condition image) are used for evaluation. The resolution of the rendered images is 256x256, while the image resolution of our architecture is 512x512. We upscale the condition images to 512x512 for our system inputs. The ground-truth images are 256x256 and we downscale our generated images to 256x256 for evaluation, while 512x512 images are used for the mesh reconstruction. The Chamfer Distances (CD) and volume IoU between the ground-truth and reconstructed shapes are reported for single-view 3D reconstruction. The PSNR, SSIM [40], and LPIPS [47] are reported for novel view synthesis (NVS) by averaging over the eight images.

Sparse-view cases. Sparse-view un-posed condition is a new setup (except the work of LEAP [12] and PF-LRM [38] to our knowledge). We use a process similar to the single-view setting to render images. Concretely, we first render 10 condition images for each of the 30 GSO objects. The azimuth and the elevation angles are chosen randomly from [0, 360) and [-10, 45] respectively. We render 32 ground-truth target images while aligning the azimuth of the first target view and the first input view (See §4.1 and Fig. 2). The same evaluation metrics are used, while we vary the number of condition images to be 1, 2, 4, 8, and 10.

###### 5.3 Mesh extraction from generated images

After generating 32 images (all target views in Fig. 2), a neural implicit reconstruction method recovers a mesh model, similar to SyncDreamer [18] and Wonder3D [19]. Specifically, we use grid-based NeuS [8, 14], where the foreground masks are decoded from the latent images {Zi(0)} by MVAE. Since our generated images have high resolution and quality, we directly run the monocular normal estimator released by Omnidata [6] to obtain additional normal supervisions for NeuS without a normal generation module like Wonder3D. We borrow the NeuS implementation from Wonder3D’s official codebase but do not use their ranking-based loss. With a single Nvidia 2080 Ti, it takes around 3 minutes to reconstruct a textured mesh model. The mesh could directly use the exported vertex color or be re-textured with the generated images.

###### 5.4 Mask-aware VAE pre-fine-tuning

We copy the network architecture and model weights of the default VAE [28] and add additional input and output channels to handle the mask. We found that fine-tuning Mask-aware VAE (M-VAE) only with object images improves performance. Concretely, we use approximately 3 million RGBA images rendered from Objaverse to fine-tune M-VAE as a pre-processing. We follow the original VAE hyperparameters with a base learning rate of 4.5e-6 and a batch size of 64. The training runs for 60,000 iterations. The binary cross entropy loss is used for the mask channel. The process improves PSNR from 36.6 to 41.2.

- Table 1: Single-view object modeling results, evaluating reconstructed meshes (left) and generated images (right). The ground-truth meshes and images are prepared by SyncDreamer [18] based on the Google Scanned Object [5] dataset. ICP is necessary to align reconstructed meshes for methods marked with ∗.

Task → 3D reconstruction Novel view synthesis Method Chamfer Dist.↓ Vol. IoU↑ PSNR↑ SSIM↑ LPIPS↓ Realfusion [20] 0.0819 0.2741 15.26 0.722 0.283 Magic123 [26] 0.0516 0.4528 - - One-2-3-45 [16] 0.0629 0.4086 - - Point-E [24] 0.0426 0.2875 - - Shap-E [13] 0.0436 0.3584 - - Zero123 [17] 0.0339 0.5035 18.93 0.779 0.166 SyncDreamer [18] 0.0261 0.5421 20.05 0.798 0.146 Wonder3D [19]∗ 0.0329 0.5768 - - Open-LRM [9]∗ 0.0285 0.5945 - - Ours 0.0165 0.6973 21.45 0.844 0.129

###### 5.5 Three-stage training strategy

After initializing the UNet model weights by a pre-trained latent diffusion inpainting model, we train the proposed system in three stages. First, we train as an ϵ-prediction model only with single-view conditioning cases, because our pretrained model was trained as ϵ-prediction. Second, we fine-tune as a v-prediction model [29] still with single-view conditioning cases. Third, we fine-tune as a v-prediction model with both single and sparse-view conditioning cases. Half the samples are single-view conditioning, and the other half are sparse-view conditioning, where the number of condition images is uniformly sampled between 2 and 10.

### 6 Experiments

We train the model with a batch size of 1024 using 128 Nvidia H100 GPUs for about a week. At test time, we use DDPM [10] sampler with 75 steps to sample the multi-view images, and it takes our model 30s, 77s, 123s, and 181s to generate 8, 16, 24, and 32 images, respectively. The section presents the single view experiments in §6.1, the sparse view experiments in §6.2, and text-to-3D application experiments in §6.3.

###### 6.1 Single-view object modeling

Three state-of-the-art single-view object modeling methods are our main baselines: SyncDreamer [18], Wonder3D [19], and Open-LRM [9]. Since the evaluation pipeline is the same as SyncDreamer, we copy numbers of other baselines in their paper for comparison, which includes Zero123 [17], RealFusion [20], Magic123 [26],

[Figure 163]

[Figure 164]

[Figure 165]

###### Input Wonder3d SyncDreamer Ours

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

Fig. 5: Single-view object modeling results of generated images. The input image and the generated images by Wonder3D and SyncDreamer are in 256×256. Our rendered images are in 512×512, showing higher fidelity and richer details.

One-2-3-45 [16], Point-E [24], and Shap-E [13]. The following introduces the three main baselines and how we reproduce their systems:

- • SyncDreamer generates 16 images from fixed viewpoints given a single input

image. The image resolution is 256x256. Their denoising network ϵθ initializes from Zero123 and leverages 3D feature volumes and depth-wise attention to learn multi-view consistency. It requires users to provide the elevation of the input image.

- • Wonder3D takes a single input image as the canonical view and generates 6 images as well as the normal maps. The image resolution is 256×256. Multi-view self-attention and an extra cross-domain attention ensure the consistency of generation results, while the views are sparser than ours. We run the official codebase on the GSO input images to get the results. However, the released model assumes orthographic cameras and we cannot use the same test set to evaluate

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

###### Input Wonder3D SyncDreamer OpenLRM Ours

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

Fig.6: Single-view object modeling results of reconstructed mesh models. Our meshes are exported from dense (32) and high-resolution (512×512) generated images, demonstrating finer details.

the NVS performance. ICP aligns the reconstructed mesh with the ground truth before computing the metrics.

• Open-LRM is an open-source implementation of Large Reconstruction Model (LRM) [11], a generalized reconstruction model that predicts a triplane NeRF from a single input image using a feed-forward transformer-based network. ICP aligns the reconstructed mesh with the ground truth before computing the CD and volume IoU.

- Results. Table 1 presents the quantitative evaluations of the reconstructed 3D meshes and the generated images. MVDiffusion++ consistently outperforms all the competing methods with clear margins. Note that the evaluation is not completely fair for Wonder3D that assumes orthographic camera projections, where perspective images are used in the experiments. However, we believe the clear performance gaps suffice to demonstrate the strength of our method.

Figure 5 and Figure 6 show generated images and reconstructed mesh models. In Figure 5, our method clearly shows the number on the clock (row 3), while others exhibit blurry numbers. Another example (row 5) showcases two perfectly

Input views Generated novel views

Input views Mesh

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

|[Figure 252]|
|---|

[Figure 253]

[Figure 254]

[Figure 255]

LEAP

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

Ours

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

|[Figure 264]<br><br>[Figure 265]|
|---|

[Figure 266]

[Figure 267]

[Figure 268]

LEAP

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

Ours

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

|[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]<br><br>[Figure 282]|
|---|

[Figure 283]

[Figure 284]

[Figure 285]

LEAP

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

Ours

NeuS (w/ G.T. pose) Ours

Fig. 7: Novel view synthesis and 3D reconstruction with sparse-view input images. Left: a qualitative example of novel view synthesis, comparing LEAP [12] and MVDiffusion++ with different numbers of unposed input images. Right: qualitative comparison of reconstructed meshes between NeuS [36] with ground-truth relative poses and our pose-free MVDiffusion++.

symmetrical windows generated by our method, contrasting with Wonder3D’s failure to maintain symmetry or clarity. In Figure 6, our method can recover a plausible and detailed shape of the turtle example (row 1), while Wonder3D and OpenLRM fail to recognize it as a turtle and exhibit significant artifacts.

###### 6.2 Sparse-view object modeling

- Table 2: Sparse-view object modeling results, evaluating reconstructed meshes (left) and generated images (right), based on the GSO [5] dataset.

Method Views Chamfer Dist.↓ Vol. IoU↑

Method Views PSNR↑ SSIM↑ LPIPS↓

SyncDreamer

SyncDreamer

1 0.0318 0.5610

1 19.46 0.847 0.188

- 1 0.0536 0.4400
- 2 0.0307 0.5884 4 0.0158 0.7323

- 1 14.66 0.47 0.43
- 2 16.22 0.59 0.36 4 16.54 0.61 0.35

NeuS[36] (G.T. pose)

LEAP[12]

10 16.84 0.64 0.34

10 0.0096 0.8092

- 1 20.25 0.862 0.157
- 2 21.73 0.872 0.137 4 23.44 0.886 0.117

- 1 0.0208 0.6689
- 2 0.0158 0.7260 4 0.0122 0.7737

Ours

Ours

10 25.03 0.899 0.102

10 0.0101 0.8046

Sparse-view un-posed input images is a challenging setting, where we are aware of only a few existing approaches such as LEAP [12] and PF-LRM [38], a

Text-to-3D examples

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

Failure examples

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

Fig.8: Text-to-3D application examples. (Top) A text-to-image model generates an image given a text-prompt. (Bottom) MVDiffusion++ turns the generated image into a 3D model. We also show some failure examples in the bottom.

sparse-view pose-free extension of LRM [11]. There is no public implementation of PF-LRM, and we pick LEAP as the first baseline. The literature on multi-view 3D reconstruction is extensive. It would be valuable to contrast our approach, even though they require camera poses as input. As a compromise, we have selected NeuS [36] as our second benchmark by providing the ground-truth camera poses as their input.

- • LEAP leverages a transformer to predict neural volumes of radiance fields from a sparse number of views and is also pose-free. LEAP employs DINOv2 [25] as the feature extractor and has reasonable generalization capacity.
- • NeuS is a 3D reconstruction method, where we provide the ground-truth camera poses of the condition images as well as surface normals estimated by Omnidata’s monocular normal estimator [6]. We use the public grid-based NeuS implementation [8]. This baseline is similar to MonoSDF [46] or NeurIS [35]

equipped with ground-truth foreground masks and camera poses, thus sets a performance upper bound for methods without generative priors.

- Results. Table 2 and Figure 7 present the quantitative and qualitative comparison results, respectively. Compared to LEAP, MVDiffusion++ generates images with much better quality. LEAP and our method both exploit multi-view selfattention to establish global 3D consistency. Therefore, we attribute our better performance to the strong image priors inherited from the pre-trained latent diffusion models. Our reconstructed meshes outperform NeuS in most settings, a notable achievement considering that NeuS uses ground-truth camera poses. This comparison highlights the practicality of our method, enabling users to achieve high-quality 3D models from just a few object snapshots.

###### 6.3 Text-to-3D application

MVDiffusion++ shows consistent performance with minimal errors on the GSO dataset. Note that our training data solely comes from Objaverse dataset [4], and MVDiffusion++ already achieves remarkable generalization capabilities. To further challenge the system, we demonstrate a text-to-3D application, where a text-to-image model prepares an input condition image. MVDiffusion++ turns the condition image into a 3D model. Figure 8 has four examples demonstrating the power of our approach.

### 7 Limitations and future challenges

This paper presents a pose-free technique for reconstructing objects using an arbitrary number of images. Central to this approach is a sophisticated multi-branch, multi-view diffusion model. This model processes any number of conditional images to produce dense, consistent views from fixed perspectives. This capability significantly enhances the performance of existing reconstruction algorithms, enabling them to generate high-quality 3D models. Our results show that MVDiffusion++ sets a new standard in performance for both single-view and sparse-view object reconstruction.

Figure 8 presents typical failure modes and the limitations of our approach. Our method struggles with thin structures as in the leftmost example, which fails to reconstruct a cable. Our method occasionally generates implausible images for views occluded in the input, a notable instance being the depiction of a cat with two tails. These shortcomings are predominantly attributed to the lack of training data, where one future work will expand the framework to incorporate videos, which offer richer contextual and spatial information, potentially enabling dynamic video generation.

Acknowledgements. This research is partially supported by NSERC Discovery Grants with Accelerator Supplements and DND/NSERC Discovery Grant Supplement, NSERC Alliance Grants, and John R. Evans Leaders Fund (JELF). We thank the Digital Research Alliance of Canada and BC DRI Group for providing computational resources.

## Bibliography

- [1] Agarwal, S., Furukawa, Y., Snavely, N., Simon, I., Curless, B., Seitz, S.M., Szeliski, R.: Building rome in a day. Communications of the ACM 54(10), 105–112 (2011)
- [2] Dao, T.: Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691 (2023)
- [3] Dao, T., Fu, D., Ermon, S., Rudra, A., R´e, C.: Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems 35, 16344–16359 (2022)
- [4] Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A.: Objaverse: A universe of annotated 3d objects. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13142–13153 (2023)
- [5] Downs, L., Francis, A., Koenig, N., Kinman, B., Hickman, R., Reymann, K., McHugh, T.B., Vanhoucke, V.: Google scanned objects: A high-quality dataset of 3d scanned household items. In: ICRA (2022)
- [6] Eftekhar, A., Sax, A., Malik, J., Zamir, A.: Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 10786– 10796 (2021)
- [7] Furukawa, Y., Curless, B., Seitz, S.M., Szeliski, R.: Towards internet-scale multi-view stereo. In: 2010 IEEE computer society conference on computer vision and pattern recognition. pp. 1434–1441. IEEE (2010)
- [8] Guo, Y.C.: Instant neural surface reconstruction (2022), https://github.com/bennyguo/instant-nsr-pl
- [9] He, Z., Wang, T.: Openlrm: Open-source large reconstruction models. https: //github.com/3DTopia/OpenLRM (2023)
- [10] Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)
- [11] Hong, Y., Zhang, K., Gu, J., Bi, S., Zhou, Y., Liu, D., Liu, F., Sunkavalli, K., Bui, T., Tan, H.: Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023)
- [12] Jiang, H., Jiang, Z., Zhao, Y., Huang, Q.: Leap: Liberate sparse-view 3d modeling from camera poses. arXiv preprint arXiv:2310.01410 (2023)
- [13] Jun, H., Nichol, A.: Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463 (2023)
- [14] Li, Z., M¨uller, T., Evans, A., Taylor, R.H., Unberath, M., Liu, M.Y., Lin, C.H.: Neuralangelo: High-fidelity neural surface reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8456–8465 (2023)
- [15] Liu, M., Shi, R., Chen, L., Zhang, Z., Xu, C., Wei, X., Chen, H., Zeng, C., Gu, J., Su, H.: One-2-3-45++: Fast single image to 3d objects with consistent

multi-view generation and 3d diffusion. arXiv preprint arXiv:2311.07885

(2023)

- [16] Liu, M., Xu, C., Jin, H., Chen, L., Xu, Z., Su, H.: One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. arXiv preprint arXiv:2306.16928 (2023)
- [17] Liu, R., Wu, R., Van Hoorick, B., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero-1-to-3: Zero-shot one image to 3d object. In: ICCV (2023)
- [18] Liu, Y., Lin, C., Zeng, Z., Long, X., Liu, L., Komura, T., Wang, W.: Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453 (2023)
- [19] Long, X., Guo, Y.C., Lin, C., Liu, Y., Dou, Z., Liu, L., Ma, Y., Zhang, S.H., Habermann, M., Theobalt, C., et al.: Wonder3d: Single image to 3d using cross-domain diffusion. arXiv preprint arXiv:2310.15008 (2023)
- [20] Melas-Kyriazi, L., Laina, I., Rupprecht, C., Vedaldi, A.: Realfusion: 360deg reconstruction of any object from a single image. In: CVPR (2023)
- [21] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: ECCV (2020)
- [22] Mittal, P., Cheng, Y.C., Singh, M., Tulsiani, S.: Autosdf: Shape priors for 3d completion, reconstruction and generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 306–315 (2022)
- [23] Murray, N., Marchesotti, L., Perronnin, F.: Ava: A large-scale database for aesthetic visual analysis. In: 2012 IEEE conference on computer vision and pattern recognition. pp. 2408–2415. IEEE (2012)
- [24] Nichol, A., Jun, H., Dhariwal, P., Mishkin, P., Chen, M.: Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751 (2022)
- [25] Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193

(2023)

- [26] Qian, G., Mai, J., Hamdi, A., Ren, J., Siarohin, A., Li, B., Lee, H.Y., Skorokhodov, I., Wonka, P., Tulyakov, S., et al.: Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843 (2023)
- [27] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: Highresolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10684–10695 (2022)
- [28] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: Highresolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- [29] Salimans, T., Ho, J.: Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512 (2022)

18 Tang et al.

- [30] Shi, R., Chen, H., Zhang, Z., Liu, M., Xu, C., Wei, X., Chen, L., Zeng, C., Su, H.: Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110 (2023)
- [31] Shi, Y., Wang, P., Ye, J., Long, M., Li, K., Yang, X.: Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512 (2023)
- [32] Stereopsis, R.M.: Accurate, dense, and robust multiview stereopsis. IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 32(8) (2010)
- [33] Tang, S., Zhang, F., Chen, J., Wang, P., Furukawa, Y.: Mvdiffusion: Enabling holistic multi-view image generation with correspondence-aware diffusion. arXiv preprint arXiv:2307.01097 (2023)
- [34] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser,  L., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)
- [35] Wang, J., Wang, P., Long, X., Theobalt, C., Komura, T., Liu, L., Wang, W.: Neuris: Neural reconstruction of indoor scenes using normal priors. In: European Conference on Computer Vision. pp. 139–155. Springer (2022)
- [36] Wang, P., Liu, L., Liu, Y., Theobalt, C., Komura, T., Wang, W.: Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. In: NeurIPS (2021)
- [37] Wang, P., Shi, Y.: Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201 (2023)
- [38] Wang, P., Tan, H., Bi, S., Xu, Y., Luan, F., Sunkavalli, K., Wang, W., Xu, Z., Zhang, K.: Pf-lrm: Pose-free large reconstruction model for joint pose and shape prediction. arXiv preprint arXiv:2311.12024 (2023)
- [39] Wang, Y., Lira, W., Wang, W., Mahdavi-Amiri, A., Zhang, H.: Slice3d: Multi-slice, occlusion-revealing, single view 3d reconstruction. arXiv preprint arXiv:2312.02221 (2023)
- [40] Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. TIP (2004)
- [41] Wu, T., Zhang, J., Fu, X., Wang, Y., Ren, J., Pan, L., Wu, W., Yang, L., Wang, J., Qian, C., et al.: Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 803–814 (2023)
- [42] Xu, Y., Tan, H., Luan, F., Bi, S., Wang, P., Li, J., Shi, Z., Sunkavalli, K., Wetzstein, G., Xu, Z., et al.: Dmv3d: Denoising multi-view diffusion using 3d large reconstruction model. arXiv preprint arXiv:2311.09217 (2023)
- [43] Yan, X., Yang, J., Yumer, E., Guo, Y., Lee, H.: Perspective transformer nets: Learning single-view 3d object reconstruction without 3d supervision. Advances in neural information processing systems 29 (2016)
- [44] Yang, Z., Ren, Z., Bautista, M.A., Zhang, Z., Shan, Q., Huang, Q.: Fvor: Robust joint shape and pose optimization for few-view object reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2497–2507 (2022)

- [45] Yao, Y., Luo, Z., Li, S., Fang, T., Quan, L.: Mvsnet: Depth inference for unstructured multi-view stereo. In: Proceedings of the European conference on computer vision (ECCV). pp. 767–783 (2018)
- [46] Yu, Z., Peng, S., Niemeyer, M., Sattler, T., Geiger, A.: Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction. Advances in neural information processing systems 35, 25018–25032 (2022)
- [47] Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: CVPR (2018)

