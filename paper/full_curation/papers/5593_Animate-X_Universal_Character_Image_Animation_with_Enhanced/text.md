# arXiv:2410.10306v2[cs.CV]11Dec2024

## ANIMATE-X: UNIVERSAL CHARACTER IMAGE ANIMATION WITH ENHANCED MOTION REPRESENTATION

##### Shuai Tan1∗, Biao Gong1†, Xiang Wang2, Shiwei Zhang2, Dandan Zheng1, Ruobing Zheng1, Kecheng Zheng1, Jingdong Chen1, Ming Yang1 1Ant Group 2Alibaba Group

{tanshuai2001,a.biao.gong}@gmail.com, {xiaolao.wx,zhangjin.zsw}@alibaba-inc.com,{yuandan.zdd, zhengruobing.zrb,zhengkecheng.zkc,jingdongchen.cjd,m.yang}@antgroup.com Project Page: https://lucaria-academy.github.io/Animate-X/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

|[Figure 5]|
|---|

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

|[Figure 15]|
|---|

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

|[Figure 23]|
|---|

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

|[Figure 32]|
|---|

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

|[Figure 40]|
|---|

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

|[Figure 50]|
|---|

[Figure 51]

[Figure 52]

Figure 1: Animations produced by Animate-X which extends beyond human to anthropomorphic characters with various body structures, e.g., without limbs, from games, animations, and posters.

ABSTRACT

Character image animation, which generates high-quality videos from a reference image and target pose sequence, has seen significant progress in recent years. However, most existing methods only apply to human figures, which usually do not generalize well on anthropomorphic characters commonly used in industries like gaming and entertainment. Our in-depth analysis suggests to attribute this limitation to their insufficient modeling of motion, which is unable to comprehend the movement pattern of the driving video, thus imposing a pose sequence rigidly onto the target character. To this end, this paper proposes Animate-X, a universal animation framework based on LDM for various character types (collectively named X), including anthropomorphic characters. To enhance motion representation, we introduce the Pose Indicator, which captures comprehensive motion pattern from the driving video through both implicit and explicit manner. The former leverages CLIP visual features of a driving video to extract its gist of motion, like the overall movement pattern and temporal relations among motions, while the latter strengthens the generalization of LDM by simulating possible inputs in advance that may arise during inference. Moreover, we introduce a new Animated Anthropomorphic Benchmark (A2Bench) to evaluate the performance of Animate-X on universal and widely applicable animation images. Extensive experiments demonstrate the superiority and effectiveness of Animate-X compared to state-of-the-art methods.

∗ Work done during internship at Ant Group. † Project lead and corresponding author.

- 1 INTRODUCTION

Character image animation Yang et al. (2018); Zablotskaia et al. (2019b) is a compelling and challenging task that aims to generate lifelike, high-quality videos from a reference image and a target pose sequence. A modern image animation method shall ideally balance the identity preservation and motion consistency, which contributes to the promise of broad utilization Hu et al. (2023); Xu et al. (2023a); Chang et al. (2023a); Jiang et al. (2022). The phenomenal successes of GAN Goodfellow et al. (2014); Yu et al. (2023); Zhang et al. (2022b) and generative diffusion models Ho et al. (2022; 2020); Guo et al. (2023) have reshaped the performance of character animation generation. Nevertheless, most existing methods only apply to the human-specific character domain. In practice, the concept of “character” encompasses a much broader concept than human, including anthropomorphic figures in cartoons and games, collectively referred to as X, which are often more desirable in gaming, film, short videos, etc. The difficulty in extending current models to these domains can be attributed to two main factors: (1) the predominantly human-centered nature of available datasets, and (2) the limited generalization capabilities of current motion representations.

The limitations are clearly evidenced for non-human characters in Fig. 5. To replicate the given poses, the diffusion models trained on human dance video datasets tend to introduce unrelated human characteristics which may not make sense to reference figures, resulting in abnormal distortions. In other words, these models treat identity preservation and motion consistency as conflicting goals and struggle to balance them, while motion control often prevails. This issue is particularly pronounced for non-human anthropomorphic characters, whose body structures often differ from human anatomy—such as disproportionately large heads or the absence of arms, as shown in Fig. 1. The primary cause is that the motion representations extracted merely from pose conditions are hard to generalize to a broad range of common cartoon characters with unique physical characteristics, leading to their excessive sacrifices in identity preservation in favor of strict pose consistency, which is an unsensible trade-off between these conflicting goals.

To address this issue, the natural approach is to enhance the flexibility of motion representations without discarding current pose condition, which can prevent the model from making unsensible trade-offs between overly precise poses and low fidelity to reference images. To this end, we identify two key limitations of existing methods. First, the simple 2D pose skeletons, constructed by connecting sparse keypoints, lack of image-level details and therefore cannot capture the essence of the reference video, such as motion-induced deformations (e.g., body part overlap and occlusion) and overall motion patterns. Second, the self-driven reconstruction strategy aligns reference and pose skeletons by body shape, simplifying animation but ignoring shape differences during inference. These inspire us to design the new Pose Indicator from both implicit and explicit perspectives.

In this paper, we propose Animate-X for animating any character X. Sparked by generative diffusion models Rombach et al. (2022), we employ a 3D-UNet Blattmann et al. (2023) as the denoising network and provide it with motion feature and figure identity as condition. To fully capture the gist of motion from the driving video, we introduce the Pose Indicator, which consists of the Implicit Pose Indicator (IPI) and the Explicit Pose Indicator (EPI). Specifically, IPI extracts implicit motionrelated features with the assistance of CLIP image feature, isolating essential motion patterns and relations that cannot be directly represented by the pose skeletons from the driving video. Meanwhile, EPI enhances the representation and understanding of the pose encoder by simulating real-world misalignments between the reference image and driven poses during training, strengthening the ability to generate explicit pose features. With the combined power of implicit and explicit features, Animate-X demonstrates strong character generalization and pose robustness, enabling general X character animation even though it is trained solely on human datasets. Moreover, we introduce a new Animated Anthropomorphic Benchmark (A2Bench), which includes 500 anthropomorphic characters along with corresponding dance videos, to evaluate the performance of Animate-X on other types of characters. Extensive experiments on both public human animation datasets and A2Bench demonstrate that Animate-X outperforms state-of-the-art methods in preserving identity and maintaining motion consistency in animating X. Main contributions summarized as follows:

- • We present Animate-X, which facilitates image-conditioned pose-guided video generation with high generalizability, particularly for attractive anthropomorphic characters. To the best of our knowledge, this is the first work to animate generic cartoon images without the need for strict pose alignment.

- • The rethinking about the motion inspire us to propose Pose Indicator, which extracts motion representation suitable for anthropomorphic characters in both implicit and explicit manner, enhancing the robustness of Animate-X.
- • Since the popular datasets only contain human video with limited character diversity, we present a new A2Bench, specifically for evaluating performance on anthropomorphic characters. Extensive experiments demonstrate that our Animate-X outperforms the competing methods quantitatively and qualitatively on both A2Bench and current human animation benchmark.

- 2 RELATED WORK

- 2.1 DIFFUSION MODELS FOR IMAGE/VIDEO GENERATION

In recent years, diffusion models Song et al. (2021); Ho et al. (2020) have demonstrated strong generative capabilities, pushing image generation technique towards a daily productivity tool Nichol et al. (2022); Ramesh et al. (2022); Mou et al. (2023); Huang et al. (2023); Zhang et al. (2023a); Liu et al. (2023). Pioneering works such as DALL-E 2 Ramesh et al. (2022) and Imagen Saharia

- et al. (2022) have showcased the extraordinary potential of diffusion models for high-quality image synthesis. Notable contributions, including Stable Diffusion Rombach et al. (2022), have well balanced scalability and efficiency, making diffusion-based image generation accessible and versatile across various applications. On the video generation front, diffusion models are making amazing progress Singer et al. (2023); Wang et al. (2023a; 2024c); Wu et al. (2023); Chai et al. (2023); Ceylan et al. (2023); Guo et al. (2023); Zhou et al. (2022); An et al. (2023); Xing et al. (2023); Qing et al. (2023); Yuan et al. (2023); Tan et al. (2024d); Gong et al. (2024). These methods joint spatio-temporal modeling to generate realistic motion dynamics and ensure temporal consistency, marking a substantial step forward in generative models for video content. In this work, we aim to tackle the character-centered image animation task, a dedicated of conditional video generation. Our approach enables the transformation of static images into dynamic animations by conditioning on desired motion. This innovation bridges the gap between image and video generation, highlights the versatility and adaptability of diffusion models in creating engaging visual narratives.

2.2 POSE-GUIDED CHARACTER MOTION TRANSFER

Character image animation aims to transfer motion from the source character to the target identity Zhang et al. (2024); Chang et al. (2023b), which has experienced an impressive journey to improve animation quality and versatility. Early works Li et al. (2019); Siarohin et al. (2019b; 2021b); Zhao & Zhang (2022b); Tan et al. (2024a); Wang et al. (2022); Tan et al. (2024c;b; 2023) predominantly utilize Generative Adversarial Networks (GANs) to generate animated human images. However, these GAN-based models are often confronted by the emergence of various artifacts in the generated outputs. With the advent of diffusion models, researchers Shen et al. (2024); Zhu et al. (2024) explored how to go beyond GANs. One effort is Disco Wang et al. (2023b), which leverages ControlNet Zhang et al. (2023b) to facilitate human dance generation, demonstrating the potential of diffusion models in generating dynamic human poses. Following this, MagicAnimate Xu

- et al. (2023b) and Animate Anyone Hu et al. (2023) introduce transformer-based temporal attention modules Vaswani (2017), enhancing the temporal consistency of animations and resulting in more smooth movement transitions. Sparked by the linear time efficiency of Mamba Gu & Dao (2023); Gu et al. (2021) conceptually merges the merits of parallelism and non-locality, Unianimate Wang
- et al. (2024b) resorts to it resorts to Mamba for efficient temporal modeling.

While these approaches have improved the realism of the animations, a notable limitation remains: most current methods require strict alignment between a reference image and driving video. This restricts their applicability in the scenarios where poses cannot be easily extracted, such as anthropomorphic characters, often resulting in bizarre and unsatisfactory outputs. In contrast, our approach adopts a robust and flexible motion representation to mitigate the dependence on pose alignment. This enables the generation of high-quality animations even in cases where previous methods struggle with non-alignable poses. In this manner, our method enhances the versatility and applicability of character image animation across a broad range of contexts (X character).

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Implicit Pose Indicator (IPI)

[Figure 57]

…

DWPOSE CLIP

…

…

DWPOSE Raw Data (video)

CLIP Feature (video)

Ref. Image Driven Video

[Figure 58]

[Figure 59]

Transformer

Cross-Attention

Encoder

CLIP VAE DWPOSE CLIP

Block

FFN

…

| | |
|---|---|
|C<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>Noise<br><br>[Figure 63]| |
| | |
| | |

[Figure 64]

[Figure 65]

Pose Query

[Figure 66]

P

###### EPI

IPI

[Figure 67]

###### +

[Figure 68]

Learnable Query

[Figure 69]

Merge Query

…

ImageCondition

Explicit Pose Indicator (EPI)

[Figure 70]

Sample

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

…

Each DWPOSE Step

[Figure 75]

###### …

Pose

[Figure 76]

Transformed Pose

Transformation

DWPOSE Raw Data (video)

Detail

…

[Figure 77]

[Figure 78]

- Transformed Pose
- - location
- - …

[Figure 79]

Pose Pool

Pose

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Encoder

Pose #1

###### …

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

VAE

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Animated Video

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

###### C

#2

…

Rescale Pool

Noise Ref.

[Figure 109]

[Figure 110]

- - body len
- - arm len
- - drop parts

- - face
- - leg len
- - neck len

Pose

Spatial Attn. Motion Attn. Temporal Attn.

#3

Figure 2: (a) The overview of our Animate-X. Given a reference image Ir, we first extract CLIP image feature fφr and latent feature fer via CLIP image encoder Φ and VAE encoder E. The proposed Implicit Pose Indicator (IPI) and Explicit Pose Indicator (EPI) produce motion feature fi and pose feature fe, respectively. fe is concatenated with the noised input ϵ along the channel dimension, then further concatenated with fer along the temporal dimension. This serves as the input to the diffusion model ϵθ for progressive denoising. During the denoising process, fφr and fi provide appearance condition from Ir and motion condition from I1:dF. At last, a VAE decoder D is adopted to map the generated latent representation z0 to the animation video. (b) The detailed structure of Implicit Pose Indicator. (c) The pipeline of pose transformation by Explicit Pose Indicator.

- 3 METHOD

In this work, we aim to generate an animated video that maintains consistency in identity with a reference image Ir and body movement with a driving video I1:dF. Different from previous works, our primary objective is to animate a general characters beyond human, particularly like anthropomorphic ones, which has broader applications in entertainment industry.

- 3.1 PRELIMINARIES OF LATENT DIFFUSION MODEL

A diffusion model (DM) operates by learning a probabilistic process that models data generation through noise. To mitigate the heavy computational load of traditional pixel-based diffusion models in high-dimensional RGB spaces, latent diffusion models (LDMs) Rombach et al. (2022) propose to shift the process into a lower-dimensional latent space using a pre-trained variational autoencoder (VAE) Kingma (2013). It encodes the input data into a compressed latent representation z0. Gaussian noise is then incrementally added to this latent representation over several steps, reducing computational requirements while maintaining the generative capabilities of the model. The process can be formalized as:

q(zt|zt−1) = N(zt; 1 − βtzt−1,βtI), (1)

where βt ∈ (0,1) represents the noise schedule. As t ∈ 1,2,...,T increases, the cumulative noise applied to the original z0 intensifies, causing zt to progressively resemble random Gaussian noise.

Compared to the forward diffusion process, the reverse denoising process pθ aims to reconstruct the clean sample z0 from the noisy input zt. We represent the denoising step p(zt − 1|zt) as follows:

###### pθ(zt−1|zt) = N(zt−1;µθ(zt,t),Σθ(zt,t)), (2)

in which µθ(zt,t) refers to the estimated target of the reverse diffusion process and the process typically is achieved by a diffusion model ϵθ with the parameters θ. To model the temporal dimension, the denoising model ϵθ is commonly built on a 3D-UNet architecture Blattmann et al. (2023) in video generation methods Hu et al. (2023); Wang et al. (2023c). Given the input conditional guidance c, they usually use an L2 loss to reduce the difference between the predicted noise and the ground-truth noise during the optimization process:

###### L = Eθ ∥ϵ − ϵθ(zt,t,c)∥2 (3)

once the reversed denoising stage is complete, the predicted clean latent is passed through the VAE decoder to reconstruct the predicted video in pixel space.

- 3.2 POSE INDICATOR

To extract motion representations, previous works typically detect the pose keypoints via DWPose Yang et al. (2023) from the driven video I1:dF and further visualize them as pose image Ip, which are trained using self-driven reconstruction strategy. However, it brings several limitations as mentioned in Sec. 1: (1) The sole pose skeletons lack image-level details and are therefore unable to capture the essence of the reference video, such as motion-induced deformations and overall motion patterns. (2) The self-driven reconstruction training strategy naturally aligns the reference and pose images in terms of body shape, which simplifies the animation task by overlooking likely body shape differences between the reference image and the pose image during inference. Both limitations weaken the model to develop a deep, holistic motion understanding, leading to inadequate motion representation. To address these issues, we propose Pose Indicator, which consists of Implicit Pose Indicator (IPI) and Explicit Pose Indicator (EPI).

Implicit Pose Indicator (IPI). To extract unified motion representations from the driving video in the first limitation, we resort to the CLIP image feature fφd = Φ(I1:dF) extracted by a CLIP Image Encoder. CLIP utilizes contrastive learning to align the embeddings of related images and texts, which may include descriptions of appearance, movement, spatial relationships and etc. Therefore, the CLIP image feature is actually a highly entangled representation, containing motion patterns and relations helpful to animation generation. As presented in Fig. 2 (a), we introduce a lightweight extractor P which is composed of N stacked layers of cross-attention and feed-forward networks (FFN). In cross attention layer, we employ fφd as the keys (K) and values (V ). Consequently, the challenge becomes designing an appropriate query (Q), which should act as a guidance for motion extraction. Considering that the keypoints pd extracted by DWPose provide a direct description of the motion, we design a transformer-based encoder to obtain the embedding qp, which is regarded as an ideal candidate for Q. Nevertheless, motion modeling using sole sparse keypoints is overly simplistic, resulting in the loss of underlying motion patterns. To this end, we draw inspiration from query transformer architecture Awadalla et al. (2023); Jaegle et al. (2021) and initialize a learnable query vector ql to complement sparse keypoints. Subsequently, we feed the merged query qm = qp + ql and fφd into P and get the implicit pose indicator fi, which contains the essential representation of motion that cannot be represented by the simple 2D pose skeletons.

Explicit Pose Indicator (EPI). To deal with the second limitation in the training strategy, we propose EPI, designed to train the model to handle misaligned input pairs during inference. The key insight lies in simulating misalignments between reference image and pose images during training while ensuring the motion remains consistent with the given driving video I1:dF. Therefore, we explore two pose transformation schemes: Pose Realignment and Pose Rescale. As shown in Fig. 2 (b), in the pose realignment scheme, we first establish a pose pool containing pose images from the training set. In each training step, we first sample the reference image Ir and the driving pose Ip following previous works. Additionally, we randomly select an align anchor pose Ianchorp from the pose pool. This anchor serves as a reference for aligning the driving pose, producing the aligned pose Irealignp . However, since the characters we aim to animate are often anthropomorphic characters, whose shapes can significantly differ from human, such as varying head-to-shoulder ratios, extremely short legs, or even the absence of arms (as shown in Fig. 1 and Fig. 5), relying solely

on pose realignment is insufficient to capture these variations for simulation. Therefore, we further introduce Pose Rescale. Specifically, we define a set of keypoint rescaling operations, including modifying the length of the body, legs, arms, neck, and shoulders, altering face size, even adding or removing specific body parts and etc. These transformations are stored in a rescale pool. After obtaining the realigned poses Irealignp , we apply a random selection of transformations from this pool with a certain probability on them, generating the final transformed poses Inp (additional examples of transformations are provided in the Appendix A). Note that we set the probability of λ ∈ [0,1] to apply the pose transformation, and with a probability of 1 − λ, the pose image remains unchanged. Subsequently, Inp is encoded to the explicit feature fe via a Pose Encoder.

- 3.3 FRAMEWORK AND IMPLEMENT DETAILS

In light of the success of previous works Hu et al. (2023); Zhang et al. (2024), Animate-X follows the main framework, which consists of several encoders for feature extraction and a 3D-UNet Wang et al. (2023a;c); Blattmann et al. (2023) for video generation. As shown in Fig. 2, given a reference image Ir, we employ the pretrained CLIP Image Encoder Φ Radford et al. (2021) to extract appearance feature fφr from Ir. To reduce the parameters of the framework and facilitate appearance alignment, we exclude the Reference Net presented in most of the previous works Hu et al. (2023); Zhang et al. (2024); Zhu et al. (2024). Instead, a VAE encoder E is utilized to extract the latent representation fer from Ir, which is then directly used as part of the input for the denoising network ϵθ following Wang et al. (2024b). For the driven video I1:dF, we detect the pose keypoints pd and CLIP feature Id via a DWPose Yang et al. (2023) and CLIP Image Encoder Φ. Subsequently, IPI and EPI introduced in Sec. 3.2 extract the implicit latent fi and explicit latent fe, respectively. The explicit fe is first concatenated with the noised latent ϵ to obtain the fused features along the channel dimension, which is further stacked with fer along the temporal dimension, resulting in combined features fmerge. Then, the combined features are fed into the video diffusion model ϵθ for jointly appearance alignment and motion modeling. The diffusion model ϵθ comprises multiple stacked layers of Spatial Attention, Motion Attention and Temporal Attention. The Spatial Attention receives inputs from fmerge and fir and fuses the identity condition from Ir with the motion condition from Id through cross-attention (CA), producing an intermediate representation x. To further enhance motion consistency, the implicit representation fi is fed into the Motion Attention module, along with x in the form of a residual connection, resulting in the representation x′ = x + CA(x,fi). Inpsired by the linear time efficiency of Mamba Gu & Dao (2023) in long sequence processing, we employ it as Temporal Attention module to maintain the temporal consistency.

Training and Inference. To improve the model’s robustness against pose and reference image misalignments, we adopt two key training schemes. First, we set a high transformation probability λ (over 98%) in the EPI, enabling the model to handle a wide range of misalignment scenarios. Second, we apply random dropout to the input conditions at a predefined rate Wang et al. (2024b). After that, while the reference image and driven video are from the same human dancing video during training, in the inference phase (Fig. 9 (b)), Animate-X can handle an arbitrary reference image and driven video, which may differ in appearance.

- 3.4 A2BENCH

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

The main task of our Animate-X is to animate an anthropomorphic character with vivid and smooth motions. However, current publicly available datasets Jafarian & Park (2021); Zablotskaia et al. (2019a) primarily focus on human animation and fall short in capturing a broad range of anthropomorphic characters and corresponding dancing videos. This gap makes these datasets and benchmarks unsuitable for quantitatively evaluating different methods in anthropomorphic character animation.

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

Figure 3: Examples from our A2Bench.

|Method<br><br>|PSNR* ↑ SSIM ↑ L1 ↑ LPIPS ↓ FID ↓<br><br>|FID-VID ↓ FVD ↓|
|---|---|---|
|Moore-AnimateAnyone Corporation (2024) MimicMotion Zhang et al. (2024) (ArXiv24) ControlNeXt Peng et al. (2024) (ArXiv24) MusePose Tong et al. (2024) (ArXiv24) Unianimate Wang et al. (2024b) (ArXiv24) Animate-X|9.86 0.299 1.58E-04 0.626 50.97<br><br>10.18 0.318 1.51E-04 0.622 122.92<br><br>10.88 0.379 1.38E-04 0.572 68.15<br><br>11.05 0.397 1.27E-04 0.549 100.91<br><br><br>11.82 0.398 1.24E-04 0.532 48.47 13.60 0.452 1.02E-04 0.430 26.11<br><br><br>|75.11 1367.84 129.40 2250.13<br><br>81.05 1652.09<br><br>114.15 1760.46 61.03 1156.36 32.23 703.87<br><br>|

- Table 1: Quantitative comparisons with SOTAs on A2Bench with the rescaled pose setting. “PSNR*” means using the modified metric Wang et al. (2024a) to avoid numerical overflow.

|Method|PSNR* ↑ SSIM ↑ L1 ↑ LPIPS ↓ FID ↓<br><br>|FID-VID ↓ FVD ↓|
|---|---|---|
|FOMM Siarohin et al. (2019a) (NeurIPS19) MRAA Siarohin et al. (2021a) (CVPR21) LIA Wang et al. (2022) (ICLR22)<br><br>|10.49 0.363 1.47E-04 0.613 183.18<br><br>12.62 0.420 1.09E-04 0.556 161.57<br>13.78 0.445 9.70E-05 0.497 105.13<br><br><br>|147.82 2535.12 196.87 3094.68<br><br>78.51 1813.28|

DreamPose Karras et al. (2023) (ICCV23) 7.76 0.305 2.28E-04 0.534 277.64 315.58 4324.42 MagicAnimate Xu et al. (2023a) (CVPR24) 11.90 0.396 1.17E-04 0.523 117.09 117.54 2021.93 Moore-AnimateAnyone Corporation (2024) (CVPR24) 11.56 0.360 1.27E-04 0.532 37.82 59.80 1117.29 MimicMotion Zhang et al. (2024) (ArXiv24) 12.66 0.407 1.07E-04 0.497 96.46 61.77 1368.83 ControlNeXt Peng et al. (2024) (ArXiv24) 12.82 0.421 1.02E-04 0.472 46.66 59.41 1152.96 MusePose Tong et al. (2024) (ArXiv24) 12.92 0.438 9.90E-05 0.470 80.22 87.97 1401.96 Animate-X 14.10 0.463 8.92E-05 0.425 31.58 33.15 849.19

- Table 2: Quantitative comparisons with existing methods on A2Bench in the self-driven setting. Underline means the second best result.

To bridge this gap, we propose the Animated Anthropomorphic character Benchmark (A2Bench) to comprehensively evaluate the performance of different methods. Specifically, we first provide a prompt template to GPT-4 OpenAI (2024) and leverage it to generate 500 prompts, each of which contains a textual description of an anthropomorphic character. Please refer to Appendix B.2 for details. Inspired by the powerful image generation capability of KLing AI Technology (2024), we feed the produced prompts into its Text-To-Image module, which synthesizes the corresponding anthropomorphic character images according to the given text prompts. Subsequently, the ImageTo-Video module is employed to further make the characters in the images dance vividly. For each prompt, we repeat the process for 4 times and filter the most satisfactory image-video pairs as the output corresponding to this prompt. In this manner, we collect 500 anthropomorphic characters and the corresponding dance videos, as shown in Fig. 3. Please refer to Appendix B for details.

- 4 EXPERIMENTS

- 4.1 EXPERIMENTAL SETTINGS

Dataset. We collect approximately 9,000 human videos from the internet and supplement this with TikTok dataset Jafarian & Park (2021) and Fashion dataset Zablotskaia et al. (2019a) for training. Following previous works Hu et al. (2023); Zablotskaia et al. (2019a); Jafarian & Park (2021), we use 10 and 100 videos for both qualitative and quantitative comparisons from TikTok and Fashion dataset, respectively. We additionally experimented on 100 image-video pairs selected from the newly proposed A2Bench introduced in Sec 3.4. Please note that, to ensure a fair comparison, the data in the A2Bench are not included in the training set to train our model. The data are only used to evaluate the quantitative results and provide interesting reference image cases.

Evaluation Metrics. We assess the results using evaluation metrics in Appendix B.1, including PSNR Hore & Ziou (2010), SSIM Wang et al. (2004), L1, LPIPS Zhang et al. (2018), which are widely-used image metrics for measuring the visual quality of the generated results. In addition, we introduce FID Heusel et al. (2017), FID-VID Balaji et al. (2019) and FVD Unterthiner et al. (2018) to quantify the discrepancy between the generated video distribution and the real video distribution.

- 4.2 EXPERIMENTAL RESULTS

[Figure 147]

Figure 4: The illustration of comparison settings.

Quantitative Results. Since our Animate-X primarily focuses on animating the anthropomorphic characters, very few of which, if not none, can be extracted the pose skeleton accurately by

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

23

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

31

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

28

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

11

LIA [ICLR22]

Moore-AA [CVPR24]

MusePose [ArXiv24]

Unianimate [ArXiv24]

MimicMotion [ArXiv24]

ControlNeXt [ArXiv24]

Ref. Image Ref. Pose Ours

Figure 5: Qualitative comparisons with state-of-the-art methods.

DWPose Yang et al. (2023). It naturally leads to a misalignment of the input reference image with the driving pose images. To compute quantitative results in this case, we set up a new comparison setting. For each case in A2Bench (i.e., a reference image Ia and a pose Pa, as shown in Fig. 4), we randomly select one human’s pose image Pb and align the anthropomorphic character’s pose Pa to it, such that the aligned pose pab retains the movements of Pa but has the same body shape (fat/thin, tall/short, etc.) as pb. Ultimately, we take the anthropomorphic character Ia and the aligned driving pose image pab as inputs to the model, generating results that allow it to calculate quantitative metrics with the original anthropomorphic character dancing video in A2Bench. In this setting, we compare our method with Animate Anyone Hu et al. (2023), Unianimate Wang et al. (2024b), MimicMotion Zhang et al. (2024), ControlNeXt Peng et al. (2024) and MusePose Tong et al. (2024), which also use pose images (e.g., Pb in Fig. 4) as input. The results of Animate Anyone Hu et al. (2023) are obtained by leveraging the publicly available reproduced code Corporation (2024). Tab. 1 presents the quantitative results, where Animate-X markedly surpasses all comparative methods in terms of all metrics. It is worth noting that, we do not use A2Bench as training data to avoid overfitting and ensure fair comparisons, in line with other comparative methods.

Following previous works which evaluate quantitative results in self-driven and reconstruction manner, we additionally compare our method with (a) GAN-based image animate works: FOMM Siarohin et al. (2019a), MRAA Siarohin et al. (2021a), LIA Wang et al. (2022). (b) Diffusion model-based image animate works: DreamPose Karras et al. (2023), MagicAnimate Xu et al. (2023a) and present the results in Tab. 2, which indicates that our method achieves the best performance across all the metrics. Moreover, we provide the quantitative results on the human dataset (TikTok and Fashion) in Tab. 7 and Tab. 8, respectively. Please refer to Appendix D.2 for details. Animate-X reaches the comparable score to Unianimate and exceeds other SOTA methods, which demonstrates the superiority of Animate-X on both anthropomorphic and human benchmarks.

Qualitative Results. Qualitative comparisons of anthropomorphic animation are shown in Fig. 5. We observe that GAN-based LIA Wang et al. (2022) does not generalize well, which can only work on a specific dataset like Siarohin et al. (2019b). Benefiting from the powerful generative capabilities of the diffusion model, Animate Anyone Hu et al. (2023) renders a higher resolution image, but the identity of the image changes and do not generate an accurate reference pose motion. Although MusePose Tong et al. (2024), Unianimate Wang et al. (2024b) and MimicMotion Zhang et al. (2024) improve the accuracy of the motion transfer, these methods generate a unseen person, which is not

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

Unianimate

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

Ours

Ref. Image

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

Unianimate

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

Ours

Ref. Image

Figure 6: Qualitative comparisons with Unianimate in terms of long video generation.

|Method<br><br>|Moore-AA MimicMotion ControlNeXt MusePose Unianimate Animate-X|
|---|---|
|Identity preservation ↑ Temporal consistency ↑ Visual quality ↑|60.4% 14.8% 52.0% 31.3% 43.0% 98.5% 19.8% 24.9% 36.9% 43.9% 81.1% 93.4% 27.0% 17.2% 40.4% 40.3% 79.3% 95.8%<br><br>|

Table 3: User study results.

the desired result. ControlNeXt combines the advantages of the above two types of methods, so maintains the consistency of identity and motion transfer to some extent, yet the results are somewhat unnatural and unsatisfactory, e.g., the ears of the rabbit and the legs of the banana in Fig. 5. In contrast, Animate-X ensures both identity and consistency with the reference image while generating expressive and exaggerated figure motion, rather than simply adopting quasi-static motion of the target character. Further, we present some long video comparisons in Fig. 6. Unianimate generates a woman out of thin air who dances according to the given pose images. Animate-X animates the reference image in a cute way while preserving appearance and temporal continuity, and it does not generate parts that do not originally exist. In summary, Animate-X excels in maintaining appearance and producing precise, vivid animations with a high temporal consistency. Please refer to Appendix D.1 for details.

User Study. To estimate the quality of our method and SOTAs from human perspectives, we conduct a blind user study with 10 participants. Specifically, we randomly select 10 characters from A2Bench and collect 10 driving video from the website. For each of 6 methods tested, 10 animation clips are generated, resulting in a total of 60 clips. Each participant is presented two results generated by different methods for the same set of inputs and asked to choose which one is better in terms of visual quality, identity preservation, and temporal consistency. This process is repeated C26 times. The results are summarized in Tab. 3, where our method noticeably outperforms other methods in all aspects, demonstrating its superiority and effectiveness. Details in Appendix C.

- 4.3 ABLATION STUDY

Ablation on Implicit Pose Indicator. To analyze the contributions of Implicit Pose Indicator, we remove it from Animate-X as w/o IPI and compare it with Baseline and Animate-X. From the first row of Fig. 7, we observe that Baseline generates a person whose appearance is appreciably distinct from the reference image. With the help of EPI, this problem is mildly mitigated. However, due to the absence of IPI, compared to Ours, there are still strange things and human-like hands

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

Ref. Image Ours

Baseline w/o IPI

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

Ref. Image

Baseline Ours

w/o EPI

Figure 7: Visualization of ablation study on IPI and EPI.

appearing, as indicated by the blue circle. For more detailed analysis about the structure of IPI, we set up several variants: (1) remove IPI: w/o IPI. (2) remove learnable query: w/o LQ. (3) remove DWPose query: w/o DQ. The quantitative results are shown in Tab. 4. It can be seen that removing the entire IPI presents the worst performance. By modifying the IPI module, although it improves on the w/o IPI, it still falls short of the final result of Animate-X, which suggests that our current IPI structure is the most reasonable and achieves the best performance.

Ablation on Explicit Pose Indicator. We demonstrate the visual results of ablating EPI setting in the second row of Fig. 7 by removing EPI. Without EPI, although the appearance of the panda is preserved thanks to IPI, the model incorrectly treats the panda’s ears as arms and forcibly stretches the legs to match the length of the legs in the pose image indicated by red circles. In contrast, these issues are completely resolved by the assistance of EPI. We further conduct more detailed ablation experiments for different pairs of pose transformations by (1) removing the entire EPI: w/o EPI. (3) remove Pose Realignment: w/o Realignment. (2) removing Pose Rescale: w/o Rescale; From the results displayed in Tab. 4, we found that Pose Realignment contributes the most. It suggests that simulating misalignment case in inference is the the key factor.

|Method|PSNR* ↑ SSIM ↑ L1 ↑ LPIPS ↓ FID ↓<br><br>|FID-VID ↓ FVD ↓|
|---|---|---|
|w/o IPI w/o LQ w/o DQ<br><br>|13.30 0.433 1.35E-04 0.454 32.56 13.48 0.445 1.76E-04 0.454 28.24 13.39 0.445 1.01E-04 0.456 30.33<br><br>|64.31 893.31 42.74 754.37 62.34 913.33<br><br>|

w/o EPI 12.63 0.403 1.80E-04 0.509 42.17 58.17 948.25 w/o Realign 12.27 0.433 1.17E-04 0.434 34.60 49.33 860.25 w/o Rescale 13.23 0.438 1.21E-04 0.464 27.64 35.95 721.11

Animate-X 13.60 0.452 1.02E-04 0.430 26.11 32.23 703.87

Table 4: Quantitative results of ablation study.

In summary, we can draw conclusions: (1) IPI facilitates the preservation of appearance and prevents the generation of content that does not exist in the reference image like human arms. (2) EPI prevents the forced alignment of a pose image that is not naturally aligned with the reference image during animation, thus avoiding the unintended animation of parts that should remain static like the panda’s ears shown in Fig. 7. Please refer to Appendix D.4 for details.

- 5 CONCLUSIONS

In this study, we present Animate-X, a novel approach to character animation capable of generalizing across different types of characters named X. To address the imbalance between identity preservation and movement consistency caused by the insufficient motion representation, we introduce the Pose Indicator, which leverages both implicit and explicit features to enhance the motion understanding of the model. In this way, Animate-X demonstrates strong generalization and robustness, achieving general X character animation. The proposed framework showcases significant improvements over state-of-the-art methods in terms of identity preservation and motion consistency, as evidenced by experiments on both public datasets and the newly introduced A2Bench, which features anthropomorphic characters. Limitation and ethical considerations see Appendix E.

REFERENCES

Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, Jia-Bin Huang, Jiebo Luo, and Xi Yin. Latentshift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv preprint arXiv:2304.08477, 2023.

Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023.

Yogesh Balaji, Martin Renqiang Min, Bing Bai, Rama Chellappa, and Hans Peter Graf. Conditional GAN with discriminative filter generation for text-to-video synthesis. In IJCAI, volume 1, pp. 2, 2019.

Ankan Kumar Bhunia, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer, Jorma Laaksonen, Mubarak Shah, and Fahad Shahbaz Khan. Person image synthesis via denoising diffusion model. In CVPR, pp. 5968–5976, 2023.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, pp. 22563–22575, 2023.

Zinelabidine Boulkenafet, Jukka Komulainen, and Abdenour Hadid. Face anti-spoofing based on color texture analysis. In 2015 IEEE international conference on image processing (ICIP), pp. 2636–2640. IEEE, 2015.

Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In ICCV, pp. 23206–23217, 2023.

Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In ICCV, pp. 23040–23050, 2023.

Di Chang, Yichun Shi, Quankai Gao, Jessica Fu, Hongyi Xu, Guoxian Song, Qing Yan, Xiao Yang, and Mohammad Soleymani. Magicdance: Realistic human dance video generation with motions & facial expressions transfer. arXiv preprint arXiv:2311.12052, 2023a.

Di Chang, Yichun Shi, Quankai Gao, Hongyi Xu, Jessica Fu, Guoxian Song, Qing Yan, Yizhe Zhu, Xiao Yang, and Mohammad Soleymani. Magicpose: Realistic human poses and facial expressions retargeting with identity-aware diffusion. In Forty-first International Conference on Machine Learning, 2023b.

Moore Threads Corporation. Moore-AnimateAnyone. 2024. URL https://github.com/ MooreThreads/Moore-AnimateAnyone.

Biao Gong, Siteng Huang, Yutong Feng, Shiwei Zhang, Yuyuan Li, and Yu Liu. Check locate rectify: A training-free layout calibration system for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6624–6634, 2024.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. NeurIPS, 27, 2014.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Albert Gu, Karan Goel, and Christopher R´e. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33: 6840–6851, 2020.

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.

Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition, pp. 2366–2369. IEEE, 2010.

Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. arXiv preprint arXiv:2311.17117, 2023.

Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. ICML, 2023.

Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pp. 4651–4664. PMLR, 2021.

Yasamin Jafarian and Hyun Soo Park. Learning high fidelity depths of dressed humans by watching social media dance videos. In CVPR, pp. 12753–12762, 2021.

Yuming Jiang, Shuai Yang, Haonan Qiu, Wayne Wu, Chen Change Loy, and Ziwei Liu. Text2human: Text-driven controllable human image generation. ACM Transactions on Graphics, 41(4):1–11, 2022.

Johanna Karras, Aleksander Holynski, Ting-Chun Wang, and Ira Kemelmacher-Shlizerman. Dream-

pose: Fashion video synthesis with stable diffusion. In ICCV, pp. 22680–22690, 2023. Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. Yining Li, Chen Huang, and Chen Change Loy. Dense intrinsic appearance flow for human

pose transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3693–3702, 2019.

Ming Liu, Yuxiang Wei, Xiaohe Wu, Wangmeng Zuo, and Lei Zhang. Survey on leveraging pre-trained generative adversarial networks for image editing and restoration. Science China Information Sciences, 66(5):151101, 2023.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023.

Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In ICML, pp. 16784–16804, 2022.

OpenAI. Chatgpt-4o. 2024. URL https://chat.openai.com/chat. Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, Ming-Chang Yang, and Jiaya Jia. Controlnext:

Powerful and efficient control for image and video generation. arXiv preprint arXiv:2408.06070, 2024.

Zhiwu Qing, Shiwei Zhang, Jiayu Wang, Xiang Wang, Yujie Wei, Yingya Zhang, Changxin Gao, and Nong Sang. Hierarchical spatio-temporal decoupling for text-to-video generation. arXiv preprint arXiv:2312.04483, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pp. 8748–8763, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Yurui Ren, Xiaoqing Fan, Ge Li, Shan Liu, and Thomas H Li. Neural texture extraction and distribution for controllable person image synthesis. In CVPR, pp. 13535–13544, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, pp. 10684–10695, 2022.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 35:36479–36494, 2022.

Fei Shen, Hu Ye, Jun Zhang, Cong Wang, Xiao Han, and Yang Wei. Advancing pose-guided image synthesis with progressive conditional diffusion models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=rHzapPnCgT.

Aliaksandr Siarohin, St´ephane Lathuili`ere, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. First order motion model for image animation. NeurIPS, 32, 2019a.

Aliaksandr Siarohin, St´ephane Lathuili`ere, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. First order motion model for image animation. Advances in neural information processing systems, 32, 2019b.

Aliaksandr Siarohin, Oliver J Woodford, Jian Ren, Menglei Chai, and Sergey Tulyakov. Motion representations for articulated animation. In CVPR, pp. 13653–13662, 2021a.

Aliaksandr Siarohin, Oliver J Woodford, Jian Ren, Menglei Chai, and Sergey Tulyakov. Motion representations for articulated animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13653–13662, 2021b.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. ICLR, 2023.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021.

Shuai Tan, Bin Ji, and Ye Pan. Emmn: Emotional motion memory network for audio-driven emotional talking face generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22146–22156, 2023.

Shuai Tan, Bin Ji, Mengxiao Bi, and Ye Pan. Edtalk: Efficient disentanglement for emotional talking head synthesis. arXiv preprint arXiv:2404.01647, 2024a.

Shuai Tan, Bin Ji, Yu Ding, and Ye Pan. Say anything with any style. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 5088–5096, 2024b.

Shuai Tan, Bin Ji, and Ye Pan. Flowvqtalker: High-quality emotional talking face generation through normalizing flow and quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26317–26327, 2024c.

Shuai Tan, Bin Ji, and Ye Pan. Style2talker: High-resolution talking head generation with emotion style and art style. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 5079–5087, 2024d.

Kuaishou Technology. Kling ai. 2024. URL https://klingai.kuaishou.com. Zhengyan Tong, Chao Li, Zhaokang Chen, Bin Wu, and Wenjiang Zhou. Musepose: a pose-driven

image-to-video framework for virtual human generation. arxiv, 2024.

Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.

A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Mod-

elscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023a.

Tan Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, and Lijuan Wang. Disco: Disentangled control for referring human dance generation in real world. arXiv e-prints, pp. arXiv–2307, 2023b.

Tan Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, and Lijuan Wang. Disco: Disentangled control for referring human dance generation in real world. In ICLR, 2024a.

Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. NeurIPS, 2023c.

Xiang Wang, Shiwei Zhang, Changxin Gao, Jiayu Wang, Xiaoqiang Zhou, Yingya Zhang, Luxin Yan, and Nong Sang. Unianimate: Taming unified video diffusion models for consistent human image animation. arXiv preprint arXiv:2406.01188, 2024b.

Xiang Wang, Shiwei Zhang, Hangjie Yuan, Zhiwu Qing, Biao Gong, Yingya Zhang, Yujun Shen, Changxin Gao, and Nong Sang. A recipe for scaling up text-to-video generation with text-free videos. In CVPR, 2024c.

Yaohui Wang, Di Yang, Francois Bremond, and Antitza Dantcheva. Latent image animator: Learning to animate images via latent space navigation. arXiv preprint arXiv:2203.09043, 2022.

Zezheng Wang, Zitong Yu, Chenxu Zhao, Xiangyu Zhu, Yunxiao Qin, Qiusheng Zhou, Feng Zhou, and Zhen Lei. Deep spatial gradient and temporal depth learning for face anti-spoofing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5042– 5051, 2020.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600– 612, 2004.

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In ICCV, pp. 7623–7633, 2023.

Zhen Xing, Qi Dai, Han Hu, Zuxuan Wu, and Yu-Gang Jiang. Simda: Simple diffusion adapter for efficient video generation. arXiv preprint arXiv:2308.09710, 2023.

Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. arXiv preprint arXiv:2311.16498, 2023a.

Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In arXiv, 2023b.

Ceyuan Yang, Zhe Wang, Xinge Zhu, Chen Huang, Jianping Shi, and Dahua Lin. Pose guided human video generation. In ECCV, pp. 201–216, 2018.

Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. Effective whole-body pose estimation with two-stages distillation. In ICCV, pp. 4210–4220, 2023.

Wing-Yin Yu, Lai-Man Po, Ray CC Cheung, Yuzhi Zhao, Yu Xue, and Kun Li. Bidirectionally deformable motion modulation for video-based human pose transfer. In ICCV, pp. 7502–7512, 2023.

Zitong Yu, Chenxu Zhao, Zezheng Wang, Yunxiao Qin, Zhuo Su, Xiaobai Li, Feng Zhou, and Guoying Zhao. Searching central difference convolutional networks for face anti-spoofing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5295– 5305, 2020.

Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. Instructvideo: Instructing video diffusion models with human feedback. arXiv preprint arXiv:2312.12490, 2023.

Polina Zablotskaia, Aliaksandr Siarohin, Bo Zhao, and Leonid Sigal. Dwnet: Dense warp-based

- network for pose-guided human video generation. arXiv preprint arXiv:1910.09139, 2019a.

Polina Zablotskaia, Aliaksandr Siarohin, Bo Zhao, and Leonid Sigal. Dwnet: Dense warp-based

- network for pose-guided human video generation. arXiv preprint arXiv:1910.09139, 2019b.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, pp. 3836–3847, 2023a.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023b.

Pengze Zhang, Lingxiao Yang, Jian-Huang Lai, and Xiaohua Xie. Exploring dual-task correlation

- for pose guided person image generation. In CVPR, pp. 7713–7722, 2022a.

Pengze Zhang, Lingxiao Yang, Jian-Huang Lai, and Xiaohua Xie. Exploring dual-task correlation

- for pose guided person image generation. In CVPR, pp. 7713–7722, 2022b.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pp. 586–595, 2018.

Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, Junqi Cheng, Yuefeng Zhu, and Fangyuan Zou. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance. arXiv preprint arXiv:2406.19680, 2024.

Jian Zhao and Hui Zhang. Thin-plate spline motion model for image animation. In CVPR, pp. 3657–3666, 2022a.

Jian Zhao and Hui Zhang. Thin-plate spline motion model for image animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3657–3666, 2022b.

Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022.

Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. In European Conference on Computer Vision (ECCV), 2024.

APPENDICES

- A NETWORK DETAILS

Due to space constraints in the main paper, we only present a brief overview of the EPI process. Here, in Fig. 8, we provide a more detailed explanation of the pose transformation in EPI, along with additional case examples. First, we sample a driving pose Ip and then randomly select an anchor pose Ianchorp from the pose pool (two examples are shown in Fig. 8). The driving pose Ip is aligned to the anchor pose Ianchorp , resulting in the aligned pose Irealignp . Next, we apply several rescaling operations randomly chosen from the rescale pool to further modify the aligned pose Irealignp . By combining different rescaling options, we can obtain multiple transformed poses Inp. However, it is important to note that in each training step, only one anchor pose Ianchorp and one rescaling combination are selected, so only one transformed pose Inp is used for training. As shown in the Fig. 8, the transformed pose Inp retains the same motion as the sampled pose Ip but has a body shape similar to the anchor pose Ianchorp . This simulates scenarios during inference where there are body shape differences between the reference image and the driving pose, enabling the model to generalize to such cases.

In the experiments, we use the visual encoder of the multi-modal CLIP-Huge model Radford et al. (2021) in Stable Diffusion v2.1 Rombach et al. (2022) to encode the CLIP embedding of the reference image and driving videos. The pose encoder, composed of several convolutional layers, follows a similar structure to the STC-encoder in VideoComposer Wang et al. (2023c). For model initialization, we employ a pre-trained video generation model Wang et al. (2024c), as done in previous approaches Xu et al. (2023a); Hu et al. (2023); Zhu et al. (2024); Wang et al. (2024b). The experiments are carried out using 8 NVIDIA A100 GPUs. During training, videos are resized to a spatial resolution of 768×512 pixels, and we feed the model with uniformly sampled video segments of 32 frames to ensure temporal consistency. We use the AdamW optimizer Loshchilov & Hutter (2017) with learning rates of 5e-7 for the implicit pose indicator and 5e-5 for other modules. For noise sampling, DDPM Ho et al. (2020) with 1000 steps is applied during training. In the inference phase, we adjust the length of the driving pose to align roughly with the reference pose and used the DDIM sampler Song et al. (2021) with 50 steps for faster sampling.

Same Motion

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

Same Body Shape

[Figure 298]

Pose Rescale

[Figure 299]

[Figure 300]

[Figure 301]

|[Figure 302]<br><br>Anchor pose<br><br>| |
|---|---|
|[Figure 303]<br><br>Anchor pose<br><br>| |

...

Aligned pose

[Figure 304]

Transformed pose

Select several pose rescale process from

rescale pool

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

Sampled pose

[Figure 309]

Pose Rescale

[Figure 310]

[Figure 311]

[Figure 312]

Select one anchor from pose pool

...

Aligned pose

Same Body Shape

Same Motion

Transformed pose

Figure 8: More example for EPI.

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

...

...

[Figure 326]

Ref. Image Driven Video

Ref. Image Driven Video

### Animate-X

### Animate-X

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

...

###### ...

Animated Video

Animated Video

(a) Train

(b) Inference

- Figure 9: The difference of training and inference pipeline. During training, the reference image and the driven video come from the same video, while in the inference pipeline, the reference image and the driven video can be from any sources and appreciably different.

- B BENCHMARK DETAILS

- B.1 EVALUATION METRIC

We employ several evaluation metrics to quantitatively assess our results, including PSNR, SSIM, L1, LPIPS, FID, FID-VID and FVD. The detailed metrics are introduced as follows:

- • PSNR is a measure used to evaluate the quality of reconstructed images compared to the original ones. It is expressed in decibels (dB) and higher values indicate better quality. PSNR is commonly used in image compression and restoration fields.
- • SSIM assesses the similarity between two images based on their luminance, contrast, and structural information. It considers perceptual phenomena affecting human vision and thus provides a better correlation with perceived image quality than PSNR.
- • The L1 metric refers to the mean absolute difference between the corresponding pixel values of two images. It quantifies the average magnitude of errors in predictions without considering their direction, making it useful for measuring the extent of differences.
- • LPIPS is a perceptual distance metric based on deep learning. It evaluates the similarity between images by analyzing the feature representations of image patches and tends to align well with human visual perception, making it suitable for tasks like image generation.
- • FID is used to assess the quality of images generated by generative models (like GANs) by comparing the distribution of generated images to that of real images in feature space (extracted by a pretrained CNN). Lower FID values suggest that the generated images are more similar to real images.
- • FID-VID extends the FID metric to video data. It measures the quality of generated videos by comparing the distribution of generated video features to real video features, providing insights into the temporal aspects of video generation.
- • FVD is another metric for evaluating video generation, similar to FID. It measures the distance between the feature distributions of real and generated videos, taking both spatial and temporal dimensions into account. Lower FVD indicates that generated videos are closer to real ones regarding visual quality and dynamics.

[Figure 333]

Below is a prompt template: a cute anthropomorphic [object], cute, anthropomorphic, with arms and legs, standing, dancing, [season], [province], [specific location], high quality.

Please replace the words in the [] brackets. Here are a few examples:

A cute anthropomorphic mobile phone, cute, anthropomorphic, with arms and legs, standing, dancing, in spring, in Liaoning, on the plains, high quality.

A cute anthropomorphic southern potato, cute, anthropomorphic, with arms and legs, standing, dancing, during the Chinese New Year, in Harbin, at the Ice and Snow World, high quality.

A cute anthropomorphic water bottle, cute, anthropomorphic, with arms and legs, standing, dancing, in the hot summer, in Dalian, by the seaside, high quality.

Please use your imagination and generate 500 similar sentences, with each sentence on a new line. The object can vary widely, such as everyday items, furniture, fruits, natural creatures, etc.

[Figure 334]

A cute anthropomorphic tissue box, cute, anthropomorphic, hands and feet, standing, dancing, spring, Guangdong, restaurant, high quality

...... A cute anthropomorphic TV set, cute, anthropomorphic, with hands and feet, standing, dancing, summer, Zhejiang, in the living room, high quality

...... A lovely anthropomorphic coconut, cute, anthropomorphic, with hands and feet, standing, dancing, summer, Hainan, in the coconut orchard, high quality

...... A cute anthropomorphic panda, cute, anthropomorphic, hands and feet, standing, dancing, spring, Sichuan, zoo, high quality

......

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

KLing AI Image2Video

KLing AI Text2Image

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

KLing AI Text2Image

KLing AI Image2Video

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

KLing AI Image2Video

KLing AI Text2Image

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

KLing AI Image2Video

KLing AI Text2Image

- Figure 10: Detailed pipeline for building A2Bench based on large-scale pretrained models, including Open-ChatGPT 4o and KLing AI.

- B.2 DATA DETAILS

The detailed process for constructing A2Bench is outlined in Fig. 10. We initially provide GPT-

- 4o with a template that clearly specifies the demand to generate ‘anthropomorphized’ images. The images were required to be cute, with arms and legs, standing, dancing, and of high quality. To allow for a variety of image outputs, we left the fields for ‘object’, ‘season’, ‘province’, and ‘specific

location’ empty. For the key factor influencing diversity and relevance, i.e., ‘object’, we provide a selectable range, such as everyday items, furniture, fruits, and natural creatures. To help GPT-4o better understand our intent, we additionally provide two examples, where the prompts had already been proven to generate satisfactory images by text-to-image module of KLing AI. Thanks to the text understanding and generation capabilities of GPT-4o, we collect 500 prompts for image generation. We then fed these 500 prompts into the text-to-image module of Keling AI, obtaining corresponding anthropomorphic characters images. Based on these images, we further generate videos of them dancing using the image-to-video module of Keling AI. In this way, we collect 500 pairs of images and videos of anthropomorphic characters, forming our A2Bench.

Since most current animation methods Wang et al. (2024b); Hu et al. (2023); Zhang et al. (2024) take a pose image sequence as motion source, we also provide our A2Bench with additional pose images. To achieve this, we employ DWPose Yang et al. (2023) to extract pose sequences from the videos. However, since DWPose is trained on human data, it does not accurately extract every pose in the dancing video of the anthropomorphic character, so after extraction, we manually screen 100 videos with accurate poses, and view them as test videos for calculating quantitative metrics. Fig. 3 displays several examples, which include anthropomorphic characters of plants, animals, food, furniture, etc. For images and videos where pose extraction is not feasible, we take them as key sources of reference images in our qualitative demonstrations. This will inspire the community to animate a wider range of interesting cases. We also anticipate that these data could serve as an important resource for future pose extraction algorithms tailored to anthropomorphic datasets, making them accessible for broader use.

- C USER STUDY

In Fig. 11, we present examples shown to participants for evaluation in our user study. To obtain genuine feedback reflective of practical applications, the ten participants in our user study experiment come from diverse academic backgrounds. Since many of them do not major in computer vision, we provide detailed explanations for each question to assist their judgments.

- • Identity Preservation: By comparing the reference image with the two generated videos by different methods, determine which video’s character more closely resembles the character in the image.
- • Temporal Consistency: Evaluate the motion changes of the character within the video and compare which video exhibits more coherent movement.
- • Visual Quality: Compared to the previous two questions, this one involves more subjective judgment. Participants should assess the videos comprehensively based on visual content (e.g., flashes, distortions, afterimages), motion effects (e.g., smoothness, physical logic), and overall plausibility.

- D ADDITIONAL EXPERIMENTAL RESULTS

- D.1 MORE QUALITATIVE RESULTS

In the main paper, we present qualitative comparison results between our method and the state-ofthe-art (SOTA) methods under a cross-driven setting on a human-like character, where our approach demonstrates outstanding performance. Considering that the other methods are primarily self-driven and trained on human characters, making them more suitable for inference in such settings, we additionally provide comparison results under a self-reconstruction setting on Tiktok and Abench. As shown in Fig. 14, when there is a appreciably difference between the reference pose and the reference image, the GAN-based LIA Wang et al. (2022) produces noticeable artifacts. Thanks to the powerful generative capabilities of diffusion models, diffusion-based models generate higher-quality results. However, MusePose Tong et al. (2024) and MimicMotion Zhang et al. (2024) generate awkward arms and blurry hands, respectively, while ControlNeXt Peng et al. (2024) synthesizes incorrect movements. Only Unianimate Wang et al. (2024b) can obtain results comparable to ours. Yet, when the reference image is a non-human character, even in a self-driven setting with the same training strategy as Unianimate, their results still show distorted heads. Fig. 15 provides results of

[Figure 363]

[Figure 364]

Ref. Image

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

Ref. Pose

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

ControlNeXt

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

MimicMotion

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

Moore-AA

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

MusePose

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

Unianimate

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

Ours

Figure 11: Visualization of cases in the user study

more comparison results, including MRAA Siarohin et al. (2021a), MagicAnimate Xu et al. (2023a) and Moore-AnimateAnyone Corporation (2024). In contrast, our method consistently generates satisfactory results for both human and anthropomorphic characters, demonstrating its ability to drive X character and highlighting its strong generalization and robustness.

- D.2 MORE QUANTITATIVE RESULTS

Tab. 7 and Tab. 8 presents the quantitative results on TikTok Jafarian & Park (2021) and Fashion Zablotskaia et al. (2019a) dataset, which suggests the superiority of methods over the comparison SOTA methods. Only Unianimate achieves comparable performance; however, our method is applicable to a wider range of characters and various unaligned pose inputs, as demonstrated in

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

Pose

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

Results

Ref.

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

Pose

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

Results

Ref.

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

Pose

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

Results

Ref.

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

Pose

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

Results

Ref.

Figure 12: Visualization of the robustness of Animate-X.

Tab. 1. This addresses the main issue that this paper aims to solve: developing a universal character image animation model.

- D.3 ROBUSTNESS

Our method demonstrates robustness to both input X character and pose variations. On the one hand, as shown in Fig. 1, our approach successfully handles inputs from diverse subjects, including characters vastly different from humans, such as those without limbs, as well as game characters or those generated by other models. Despite these variations, our method consistently produces satisfactory results without crashing, showcasing its robustness to the input reference images. On

|Method|PSNR* ↑ SSIM ↑ L1 ↑ LPIPS ↓ FID ↓<br><br>|FID-VID ↓ FVD ↓|
|---|---|---|
|w/o IPI w/o LQ w/o DQ PA KV Q<br><br>|13.30 0.433 1.35E-04 0.454 32.56 13.48 0.445 1.76E-04 0.454 28.24 13.39 0.445 1.01E-04 0.456 30.33 13.25 0.436 1.11E-04 0.464 27.63 13.34 0.443 1.17E-04 0.459 26.75<br><br>|64.31 893.31 42.74 754.37 62.34 913.33 46.54 785.36 42.14 785.69<br><br>|

w/o EPI 12.63 0.403 1.80E-04 0.509 42.17 58.17 948.25 w/o Add 13.28 0.442 1.56E-04 0.459 34.24 52.94 804.37 w/o Drop 13.36 0.441 1.94E-04 0.458 26.65 44.55 764.52 w/o BS 13.27 0.443 1.08E-04 0.461 29.60 56.56 850.17 w/o NF 13.41 0.446 1.82E-04 0.455 29.21 56.48 878.11 w/o AL 13.04 0.429 1.04E-04 0.474 27.17 33.97 765.69 w/o Rescalings 13.23 0.438 1.21E-04 0.464 27.64 35.95 721.11 w/o Realign 12.27 0.433 1.17E-04 0.434 34.60 49.33 860.25

Animate-X 13.60 0.452 1.02E-04 0.430 26.11 32.23 703.87

Table 5: Quantitative results of ablation study.

the other hand, as illustrated in Fig. 12, even when the pose images exhibit body part omissions (highlighted by the red circles), our method correctly interprets the intended motion and generates coherent results for the reference images. This highlights the robustness of our approach to different pose images.

- D.4 MORE ABLATION STUDY

In the main paper, we present the results of the primary ablation experiments for IPI and EPI. In this section, we supplement those results with additional ablation experiments to further demonstrate the contribution of each individual module.

Ablation on Implicit Pose Indicator. For more detailed analysis about the structure of IPI, we set up several variants: (1) remove IPI: w/o IPI. (2) remove learnable query: w/o LQ. (3) remove DWPose query: w/o DQ. (4) set IPI and spatial Attention to Parallel: PA. (5) set CLIP features as Q and DWPose as K,V in IPI: KV Q. The quantitative results are shown in Tab. 5. It can be seen that removing the entire IPI presents the worst performance. By modifying the IPI module, although it improves on the w/o IPI, it still falls short of the final result of Animate-X, which suggests that our current IPI structure is the most reasonable and achieves the best performance.

Since IPI is embedded in Animate-X in the form of residual connection, i.e., x = x + αIPI(x), we also explore the impact of the weight α of IPI on performance as illustrated in Fig. 13, as α increases from 0 to 1, all metrics show a stable improvement despite some fluctuations. The best performance is achieved when α is set to 1, so we empirically set α to 1 in the final configuration.

Ablation on Explicit Pose Indicator. We conduct more detailed ablation experiments for different pairs of pose transformations by (1) removing the entire EPI: w/o EPI; (2)&(3) removing adding and dropping parts; canceling the change of the length of (4) body and should: w/o BS; (5) neck and face: w/o NF; (6) arm and leg: w/o AL; (7) removing all rescaling process: w/o Rescalings; (8) remove another person pose alignment: w/o Realign. From the results displayed in Tab. 5, we found that each pose transformation contributes compared to w/o EPI, with aligned transformations with another person’s pose contributing the most. It suggests that maintaining the overall integrity of the pose while allowing for some variations is the most important factor, and EPI also learns the overall integrity of the pose. The final result indicates that all the transformations together achieve the best performance.

To explore the effect of different probabilities λ of using pose transformation for EPI on the model performance, we set λ as 100%, 98%, 95%, 90% and 80% for the ablation experiments on two datasets. The results presented in Tab. 6 suggest that a high λ performs better on A2Bench, i.e., it performs better when the reference image and pose image are not aligned, but harms performance on the TikTok dataset, i.e., when the reference image and pose image are strictly aligned. In contrast,

1.00

- 0.00

0.05

0.19

0.33

0.22

0.54

0.67

0.82

0.87

0.77

- 1.00 1.00

1.00

1.0

0.91

0.90

0.90

0.84

0.81

0.8

0.78

0.73

0.70

0.67

0.64

NormalizedValues

0.59

0.6

0.53

0.52

0.47

0.46

0.44

0.43

0.4

0.37 0.38

0.36

0.34

0.32

0.28

0.27

0.26

0.21

0.21

0.19

0.2

0.17

FVD-3DInception

0.14

LPIPS PSNR SSIM

0.11

0.09

0.00

0.00

0.0

L1

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 Wights for IPI

Figure 13: Ablation study on the weight α of Implicit Pose Indicator. To better visualize the impact of α on performance, we normalize all the values to the range of 0 to 1.

A2Bench TikTok Jafarian & Park (2021)

Method

SSIM↑ FID↓ FID-VID↓ FVD↓ SSIM↑ FID↓ FID-VID↓ FVD↓

100% 0.452 26.11 32.23 703.87 0.802 55.26 17.47 138.36 98% 0.448 26.93 37.67 775.24 0.797 55.81 16.28 129.48 95% 0.447 27.46 39.21 785.55 0.804 52.72 14.61 124.92 90% 0.444 27.15 38.03 775.38 0.806 52.81 14.82 139.01 80% 0.442 29.13 47.93 803.97 0.802 54.51 14.42 133.78

Table 6: Quantitative results for different probabilities of using pose transformation.

|Method<br><br>|L1 ↓ PSNR ↑ PSNR* ↑ SSIM ↑ LPIPS ↓<br><br>|FVD ↓|
|---|---|---|
|FOMM Siarohin et al. (2019a) (NeurIPS19) MRAA Siarohin et al. (2021a) (CVPR21) TPS Zhao & Zhang (2022a) (CVPR22)|3.61E-04 - 17.26 0.648 0.335 3.21E-04 - 18.14 0.672 0.296 3.23E-04 - 18.32 0.673 0.299<br><br>|405.22 284.82 306.17|

DreamPose Karras et al. (2023) (ICCV23) 6.88E-04 28.11 12.82 0.511 0.442 551.02 DisCo Wang et al. (2024a) (CVPR24) 3.78E-04 29.03 16.55 0.668 0.292 292.80 MagicAnimate Xu et al. (2023a) (CVPR24) 3.13E-04 29.16 - 0.714 0.239 179.07 Animate Anyone Hu et al. (2023) (CVPR24) - 29.56 - 0.718 0.285 171.90 Champ Zhu et al. (2024) (ECCV24) 2.94E-04 29.91 - 0.802 0.234 160.82 Unianimate Wang et al. (2024b) (ArXiv24) 2.66E-04 30.77 20.58 0.811 0.231 148.06 MusePose Tong et al. (2024) (ArXiv24) 3.86E-04 - 17.67 0.744 0.297 215.72 MimicMotion Zhang et al. (2024) (ArXiv24) 5.85E-04 - 14.44 0.601 0.414 232.95 ControlNeXt Peng et al. (2024) (ArXiv24) 6.20E-04 - 13.83 0.615 0.416 326.57 Animate-X 2.70E-04 30.78 20.77 0.806 0.232 139.01

Table 7: Quantitative comparisons with existing methods on TikTok dataset.

a relatively low λ, e.g., 90%, would be in this case perform better. It is reasonable that in the case of strict alignment, we expect the pose to provide a strictly accurate motion source, and thus need to reduce the percentage λ of pose transformation. However, in the non-strictly aligned case, we expect the pose image to provide an approximate motion trend, so we need to increase λ.

|Method|PSNR ↑ PSNR* ↑ SSIM ↑ LPIPS ↓<br><br>|FVD ↓|
|---|---|---|
|MRAA Siarohin et al. (2021a) (CVPR21) TPS Zhao & Zhang (2022a) (CVPR22) DPTN Zhang et al. (2022a) (CVPR22) NTED Ren et al. (2022) (CVPR22) PIDM Bhunia et al. (2023) (CVPR23) DBMM Yu et al. (2023) (ICCV23)<br><br>|- - 0.749 0.212<br>- - 0.746 0.213<br>- 24.00 0.907 0.060<br>- 22.03 0.890 0.073<br>- - 0.713 0.288<br>- 24.07 0.918 0.048<br>|253.6 247.5 215.1 278.9 1197.4 168.3<br><br>|

DreamPose Karras et al. (2023) (ICCV23) - - 0.885 0.068 238.7 DreamPose w/o Finetune Karras et al. (2023) (ICCV23) 34.75 - 0.879 0.111 279.6 Animate Anyone Hu et al. (2023) (CVPR24) 38.49 - 0.931 0.044 81.6 Unianimate Wang et al. (2024b) (ArXiv24) 37.92 27.56 0.940 0.031 68.1 MimicMotion Zhang et al. (2024) (ArXiv24) - 27.06 0.928 0.036 118.48 Animate-X 36.73 27.78 0.940 0.030 79.4

Table 8: Quantitative comparisons with existing methods on the Fashion dataset. “w/o Finetune” represents the method without additional finetuning on the fashion dataset.

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

#### 2TikTokABench

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

Ref. Image Ref. Pose LIA MusePose Unianimate MimicMotion ControlNeXt Ours

Figure 14: Visualization comparison on TikTok dataset and A2Bench.

- E DISCUSSION

- E.1 LIMITATION AND FUTURE WORK

Although our method has made remarkable progress, it still has certain limitations. Firstly, its ability to model hands and faces remains insufficient, a limitation commonly faced by most current generative models. While our IPI leverages CLIP features to extract implicit information such as motion patterns from the driving video, mitigating the reliance on potentially inaccurate hand and face detection by DWPose, there is still a gap between our results and the desired realism. Secondly, due to the multiple denoising steps in the diffusion process, even though we replace the transformer with a more efficient Mamba model for temporal modeling, Animate-X still cannot achieve real-

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

MRAA (CVPR 22)

MagicAnimate (CVPR 24)

Moore-AnimateAnyone (CVPR 24)

Ours

Figure 15: Comparison with more SOTAs on A2Bench.

time animation. In future work, we aim to address these two limitations. Additionally, we will focus on studying interactions between the character and the surrounding environment, such as the background, as a key task to resolve.

- E.2 ETHICAL CONSIDERATIONS

Our approach focuses on generating high-quality character animation videos, which can be applied in diverse fields such as gaming, virtual reality, and cinematic production. By providing body movement, our method enables animators to create more lifelike and dynamic characters. However, the potential misuse of this technology, particularly in creating misleading or harmful content on digital platforms, is a concern. While greatly progress has been made in detecting manipulated animations Boulkenafet et al. (2015); Wang et al. (2020); Yu et al. (2020), challenges remain in accurately identifying increasingly sophisticated forgeries. We believe that our animation results can contribute to the development of better detection techniques, ensuring the responsible use of animation technology across different domains.

