## InstructVideo: Instructing Video Diffusion Models with Human Feedback

Hangjie Yuan1* Shiwei Zhang2 Xiang Wang2 Yujie Wei2 Tao Feng3 Yining Pan4 Yingya Zhang2 Ziwei Liu5 Samuel Albanie6 Dong Ni1†

1Zhejiang University 2Alibaba Group 3Tsinghua University 4Singapore University of Technology and Design 5S-Lab, Nanyang Technological University

Version 1 Teaser figure用的是: 我的模型可以做两件事情，一件事情是text-tovideo，一件事情是editing

6CAML Lab, University of Cambridge Project page: https://instructvideo.github.io/

# arXiv:2312.12490v1[cs.CV]19Dec2023

还可以加一个简要的reward模型来训练的简图。

###### Abstract

###### InstructVideo

###### InstructVideo

###### Original Generation

###### Original Generation

###### InstructVideo

###### Original Generation

###### Generation

###### Generation

###### Fine-tuning

###### Fine-tuning

❆

❆

###### Generation

###### Fine-tuning

Frozen

Frozen

Noised Video

Noised Video

Diffusion models have emerged as the de facto paradigm for video generation. However, their reliance on web-scale data of varied quality often yields results that are visually unappealing and misaligned with the textual prompts. To tackle this problem, we propose InstructVideo to instruct text-to-video diffusion models with human feedback by reward fine-tuning. InstructVideo has two key ingredients: 1) To ameliorate the cost of reward fine-tuning induced by generating through the full DDIM sampling chain, we recast reward fine-tuning as editing. By leveraging the diffusion process to corrupt a sampled video, InstructVideo requires only partial inference of the DDIM sampling chain, reducing fine-tuning cost while improving fine-tuning efficiency. 2) To mitigate the absence of a dedicated video reward model for human preferences, we repurpose established image reward models, e.g., HPSv2. To this end, we propose Segmental Video Reward, a mechanism to provide reward signals based on segmental sparse sampling, and Temporally Attenuated Reward, a method that mitigates temporal modeling degradation during finetuning. Extensive experiments, both qualitative and quantitative, validate the practicality and efficacy of using image reward models in InstructVideo, significantly enhancing the visual quality of generated videos without compromising generalization capabilities. Code and models will be made publicly available.

Caption

Caption

Caption

Caption

Caption

Caption

Noised Video

Caption

Caption

Caption

❆

[Figure 1]

Trainable

Frozen

❆

❆

❆

❆

❆

❆

❆

❆

❆

Pre-trained UNet

|[Figure 2]|
|---|

|❆|
|---|

|❆|
|---|

|❆|
|---|

Pre-trained UNet

Pre-trained UNet

|[Figure 3]|
|---|

|❆|
|---|

|❆|
|---|

❆

❆

❆

❆

|❆|
|---|

❆

❆

❆

❆

❆

LoRA

[Figure 4]

[Figure 6]

[Figure 8]

[Figure 10]

[Figure 12]

[Figure 14]

[Figure 16]

[Figure 18]

LoRA

[Figure 24]

LoRA

[Figure 26]

[Figure 30]

[Figure 32]

[Figure 36]

[Figure 38]

[Figure 40]

[Figure 42]

[Figure 44]

Image Reward Model

Image Reward Model

Image Reward Model

Gradient

Gradient

Gradient

Figure 1. Overview of the InstructVideo framework. Our method performs efficient fine-tuning on sampled video-text pairs, instructed by human preferences in image reward models.

[Figure 55]

Reward Model

[Figure 56]

[Figure 57]

despite the challenges of working with high-dimensional data. While diffusion models are one factor driving progress, the scaling of training datasets has also played a key role [67, 83]. However, despite recent progress, the visual quality of generated videos still leaves room for improvement [78, 87]. A significant contributing factor to this issue is the varying quality of web-scale data employed during pre-training [3, 66], which can yield models capable of generating content that is visually unappealing, toxic and misaligned with the prompt.

While aligning model outputs with human preferences has proven highly effective for control [13], text generation [2, 44, 56, 57, 74] and image generation [43, 89, 93], it remains a notion unexplored in video diffusion models. The most widely-adopted methods for aligning models with human preferences include off-line reinforcement learning (RL) [4, 45, 57] and direct reward back-propagation [16, 59]. Typically, this entails training a reward model on manually annotated datasets that is then subsequently used to fine-tune the pre-trained generative model.

###### 1. Introduction

The emergence of diffusion models [33, 69, 71, 99] has significantly boosted generation quality across a wide range of media content [35, 41, 63, 78]. This generation paradigm has shown promise for video generation [5, 34, 35, 78, 81],

*Work conducted during his research internships at DAMO Academy. †Corresponding author.

Two major challenges arise when seeking to align video

generation models with human preferences: 1) The optimization process for optimizing human preferences is computationally demanding, often requiring video generation from textual inputs. While video generation pre-training using DDPM [33] requires only a single-step inference for every iteration, reward optimization requires a 50-step DDIM inference [70]. 2) The curation of a large annotated dataset to capture human preferences of videos is labor-intensive, while the computation- and memory-intensive demands of utilizing ViT-H [88] or ViT-L [93]-based computational alternatives to evaluate the entire video are high.

To surmount these mentioned challenges, we propose InstructVideo, a model that efficiently instructs textto-video diffusion models to follow human feedback, as illustrated in Fig. 1. Regarding the first challenge of the demanding reward fine-tuning process caused by generating through the full DDIM sampling chain, we recast the problem of reward fine-tuning as an editing procedure. This reformulation requires only partial inference of the DDIM sampling chain, thereby reducing computational demands while improving fine-tuning efficiency. Drawing inspiration from established editing workflows in diffusion models [6, 7, 17, 27, 50, 53, 55, 60, 102], where primary visual content is initially corrupted with noise and then reshaped by a target prompt, our method focuses on refining coarse and structural videos into more detailed and nuanced outputs. This contrasts with previous methods [4, 16, 43] that generate results directly from text. Such blurry and structural videos, serving as the starting point for reward fine-tuning, are procured by a simple diffusion process with negligible cost. During generation, the optimized model retains the capability to produce videos directly from textual inputs. In conjunction with back-propagation truncation of the sampling chain, we make reward fine-tuning on text-tovideo diffusion models computationally attainable and effective.

Regarding the second challenge (the lack of a reward model tailored for video generation), we postulate that the visual excellence of a video is tied to both the quality of its individual frames and the fluidity of motion across consecutive frames. To this end, we resort to off-the-shelf image reward models, e.g., HPSv2 [88], to ascertain frame quality. Drawing inspiration from temporal segment networks [79], we propose Segmental Video Reward (SegVR), which strategically evaluates video quality based on a subset of sparely sampled frames. By providing sparse reward signals, SegVR offers dual benefits: it not only ameliorates computational burden but also mitigates temporal modeling collapse. On the other hand, although LoRA [37] is adopted by default to retain the capability to generate temporally smooth videos, SegVR still leads to videos with visual artifacts, such as structure twitching and color jittering. To mitigate this, we propose Temporally Attenuated

Reward (TAR), which operates under the hypothesis that central frames should be assigned paramount importance, with emphasis tapering off towards peripheral frames. This strategic allocation of importance across frames ensures a more stable and visually coherent video generation process.

As part of our pioneering effort to align video diffusion models with human preferences, we conduct extensive experiments to assess the practicality and efficacy of integrating image reward models within InstructVideo. Our findings reveal that InstructVideo markedly enhances the visual quality of generated videos without sacrificing the model’s generalization capabilities, setting a new precedent for future research in video generation.

###### 2. Related Work

Video generation via diffusion models. Early efforts at video generation focused on GANs [25, 36, 48, 58, 68, 76, 86, 96] and VAEs [48, 54, 94]. However, due to the complexity of jointly modeling spatio-temporal dynamics, generating videos from texts remains an unresolved challenge. Recent methods for video generation aim to mitigate this by utilizing the de facto generation method, i.e., diffusion models [33, 69, 70, 99], for generating videos with diversity and fidelity [1, 5, 8–10, 12, 20, 26, 29– 31, 34, 35, 38, 49, 51, 52, 61, 65, 73, 80, 82–84, 90– 92, 95, 97, 98, 100, 103–105] and scaling up the pretraining data or model architecture [34, 36, 67, 83, 85]. VDM [35] represents a pioneering work that extended image diffusion models to video generation. Owing to the computation-intensive nature of diffusion models, followup research sought to reduce overhead by leveraging the latent space [63], e.g., ModelScopeT2V [78], Video LDM [5], MagicVideo [104] and SimDA [92], etc.. To enable more controllable generation, further efforts introduce spatiotemporal conditions [11, 20, 42, 81, 90, 95], e.g., VideoComposer [81], Gen-1 [20], DragNUWA [95], etc.. However, generating videos that adhere to human preferences remains a challenge.

Human preference model. Understanding human preference in visual content generation remains challenging [40, 43, 46, 88, 89, 93]. Some pioneering works target solving this problem by annotating a dataset with human preferences, e.g., Pick-a-pic [40], ImageReward [93] and HPD [88, 89]. Language-image models, e.g. CLIP [62] and BLIP [47], are then fine-tuned on the resulting annotated data. As such, the fine-tuned models represent a datadriven approach to modelling human preferences. However, the annotation process for capturing human preferences is highly labor- and cost-intensive. Thus, in this paper, we adopt off-the-shelf image preference models to improve the quality of generated videos.

Learning from human feedback. Learning from human feedback was first studied in the context of reinforce-

##### Version 1

###### Fine-tuning phase

𝑧 ( ) Back-propagation −𝑅 Teaser figure用的是: 我的模型可以做两件事情，一件事情 是text-to-video，一件事情是editing

𝑧 𝑧 (   ) 𝑧

𝑥

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

DDIM Sampling ×(𝝉 𝑫)

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Diffusion

[Figure 87]

Reward model

[Figure 88]

[Figure 89]

还可以加一个简要的reward模型来训 练的简图。

TAR

𝑅

[Figure 90]

[Figure 91]

v

v

Segmental sampling and decoding

Dogwood blossoms are blowing in the wind.

𝑐

Figure 2. The reward fine-tuning framework of InstructVideo. During fine-tuning, we sample video-text pairs and apply a diffusion process to corrupt the videos to a noise level τ. Subsequently, we perform partial inference of the DDIM sampling chain to obtain the human preference edited videos. By utilizing SegVR and TAR, we can leverage image reward models to perform reward fine-tuning for video generation. The VQGAN encoder and decoder are omitted for clarity. In this example, the blurry video z is edited guided by human preferences, producing a result that highlights the vibrancy and structure of the dogwood blossoms.

𝑧

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

These models typically operate in a latent space to handle complex video data [63, 78, 81]. During pre-training, a sampled video x is processed by a fixed encoder [19] to derive its latent representation z ∈ RF×h×w×3, where the video’s spatial dimensions are compressed by a factor of 8. Next, random noise is injected into the sampled video by the forward diffusion process according to a predetermined schedule {βt}Tt=1. This process can be described as zt = √α¯tz + √1 − α¯tϵ, where ϵ ∈ N(0,1) is random noise with identical dimensions to z, α¯t = ts=1 αs and αt = 1 − βt. A UNet [15, 64] ϵθ is adopted to perform denoising, enabling the generation of videos through a reverse diffusion process, conditioned on the video caption c. Optimisation employs the following reweighted variational bound [33]:

ment learning and agent alignment [14, 44] and later in large language models [57, 72], enabling them to generate helpful, honest and harmless textual outputs. This goal of learning from human feedback is also desirable in visual content generation. In image generation and editing, a key objective is to align generated images with the prompt [4, 16, 22, 59, 77, 101], preventing surprising and toxic results [43, 89, 93]. Lee et al. utilizes rewardweighted regression on a manually collected dataset, aiming to mitigate misalignment with respect to factors such as count, color and background. DDPO [4] and DPOK [22] propose to use policy gradients on a multi-step diffusion model [21], demonstrating improved rewards of aesthetic quality, image-text alignment, compressibility, etc.. DRaFT [16] and AlignProp [59] achieve feedback optimization by back-propagating the gradients of a differentiable reward function through the sampling procedure via gradient checkpointing [28]. However, learning from human feedback for video diffusion models remains underexplored owing to its prohibitive cost. InstructVideo aims to fill this gap, providing a solution for more efficient reward fine-tuning.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Video-text pair

L(θ) = Ez,ϵ,c,t ∥ϵ − ϵθ(√α¯tz + √1 − α¯tϵ,c,t)∥22 (1)

During inference, we adopt the DDIM sampling [70] method for realistic video generation.

Reward fine-tuning. Reward fine-tuning aims to optimize a pre-trained model to enhance the expected rewards of a reward function r(·,·). In our case, we target optimizing the parameters θ of a text-to-video diffusion model to enhance the expected reward of the generated videos given the text distribution:

###### 3. Methodology

In this section, we commence with preliminaries. Next, we delve into the details of InstructVideo, which encompasses: 1) A reformulation of reward fine-tuning as editing that ensures computational efficiency and efficacy. 2) Segmental Video Reward (SegVR) and Temporally Attenuated Reward (TAR) that enable efficient reward fine-tuning with image reward models.

θ(x0|c)[−r(x0,c)] (2)

Lr(θ) = EP(c)EP

where x0 is the video generated from the sampled text c via the diffusion model through the DDIM sampling chain. The reward function r(·,·) is typically a pre-trained model to assess the quality of the model output.

###### 3.1. Preliminaries

###### 3.2. InstructVideo

Text-to-video diffusion models. Text-to-video diffusion models aim to map textual input into a distribution representing video data via a reverse diffusion process [33, 69].

In Fig. 2, we illustrate InstructVideo’s fine-tuning pipeline and elaborate on the technical contributions below.

###### 3.2.1 Reward Fine-tuning as Editing

Reward fine-tuning with diffusion models is costly due to the iterative refinement process during generation using DDIM [70]. During generation, initial steps are crucial for shaping coarse, structural aspects of videos, with subsequent steps refining the coarse videos. Understanding that the essence of reward fine-tuning is not to drastically alter the model’s output but to subtly adjust it in line with human preferences, we propose to reinterpret reward fine-tuning as a form of editing [17, 53, 87]. This perspective shift allows us to perform partial inference of the DDIM sampling chain, reducing computational demands and easing optimization.

To implement this idea, we first curate a small amount of fine-tuning data from pre-training data for reward finetuning. For each video-text pair (x,c), we acquire the video’s latent embedding z as stated in Sec. 3.1. We aim to smooth out the video to eliminate undesirable artifacts and distortions [53]. To achieve this, we leverage the diffusion process rather than DDIM inversion [17, 53] to enable efficient editing. If we denote the number of DDIM steps as D and the number of pre-training DDPM steps as T, we define a mapping d : {1,...,D} → {1,...,T} that maps DDIM step index to the DDPM step index1, formulated as:

T D · (i − 1) + 1 (3)

d(i) =

Given the noise level τ, the targeted diffusion step tnoi for injecting noise is formulated as:

tnoi = d(τ · D) (4)

This allows us to obtain the starting point for reward finetuning via the diffusion process:

ϵ (5) Based on zt

z + 1 − α¯t

###### zt

= α¯t

noi

noi

noi

, we can perform τ × D steps of DDIM sampling [70] along the DDIM sub-sequence to obtain the edited result z0, which consumes τ of the computation of the full sampling chain. Utilizing the decoder [19], we decode z0 in the latent space to x0 in the RGB space.

noi

###### 3.2.2 Reward Fine-tuning with Image Reward Models

Since curating large datasets to capture human preferences for training video reward models is prohibitively expensive, we resort to off-the-shelf image reward models r(·,·), e.g., HPSv2 [88]. HPSv2 is trained on 430k pairs of images, which are annotated by humans for text-image alignment and image quality. Given that videos are natural extensions of images, we posit that these human preferences are also applicable to videos. However, initial experiments with applying dense reward fine-tuning produced degraded

1For example, if D = 20 and T = 1000, then the DDIM step subsequence is {1, 51, . . . , 901, 951}, i.e., d(2) = 51.

motion continuity. Taking inspiration from temporal segment networks [79], given a video x0 ∈ RF×H×W×3 generated from its caption c, we evenly divide it into S segments. Within each segment, we perform random frame sampling, obtaining a sparse set capturing the essence of the video xg0 = {xg0(1),...,xg0(S)}. Here, g(i) = Uniform (i − 1) · FS ,i · FS − 1 denotes a uniform sampling of index within ith segment. Utilizing r(·,·), we compute the reward score R with respect to x0 as follows:

R = Aggi[r(x0g(i),c)], i = 1,...,S (6)

where Aggi denotes the aggregation function along index i. To consider the impact of all frames in xg0, an intuitive implementation of Aggi is the mean function.

However, the simple aggregation function leads to noticeable visual artifacts in the generated videos, such as structural twitching and color jittering. This issue arises because the mean function places equal weight on all frames, disregarding the inherent dynamic nature of videos where the reward scores of frames can vary throughout the sequence. To address this, we introduce TAR that strategically emphasizes central frames, with the emphasis tapering off towards the peripheral frames, thereby avoiding uniformly optimizing all frames’ reward scores to be equally high. We define the temporally attenuated coefficient as:

tar|g(i)−F2 | (7)

fi = e−λ

where λtar controls the degree of the attenuating rate. We set λtar = 1 by default. Incorporating this coefficient, we rewrite the reward score R:

1 S

S i=1

fi · r(xg0(i),c) (8) The optimization objective in Eq. (2) can be rewritten as:

R =

θ(x0|c)[−R] (9)

Lr(θ) = EP(c)EP

###### 3.3. Reward Fine-tuning and Inference

Data preparation and evaluation metric. We follow DDPO [4] to experiment on prompts describing 45 animal species. In contrast to DDPO, since InstructVideo relies on video-text data, we select video-text pairs as the finetuning data from the base model’s pre-training dataset, i.e., WebVid10M, ensuring that no extra data is introduced. It is worth noting that we do not apply any quality filtering method to ensure that the selected videos are of high quality. Specifically, we select about 20 video-text pairs for each animal species. To evaluate the model’s ability to optimize the reward scores, we also collect evaluation data comprising about 6 prompts for each animal. We use HPSv2 score to measure the optimization performance of reward finetuning on the first frames of all segments.

ModelScopeT2V (D=20) ModelScopeT2V (D=50) InstructVideo (D=20)

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Cat walking to the camera.

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Mountain goat grazing on a cliff by the sea.

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Bee collecting sunflower pollen closeup footage.

Parrot fish in mediterranean sea underwater eating stone coral.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Parrot fish in sea underwater eating stone coral.

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Small bird during rain in india.

Figure 3. Comparion of InstructVideo with the base model ModelScopeT2V. ModelScopeT2V utilizes 20 and 50 DDIM steps.

Reward fine-tuning. We adopt the publicly available textto-video diffusion model ModelScopeT2V [78] as our base model. ModelScopeT2V is trained on WebVid10M [3] with T = 1000 and is able to generate videos of 16 × 256 × 256 resolution, which we divide into S = 4 segments. By default, we adopt the differentiable HPSv2 [88] as the reward model and perform 20-step DDIM inference, i.e., D = 20. Classifier-free guidance [32] is adopted by default. Directly back-propagating the reward loss to the diffusion models can be computationally intensive and risks catastrophic forgetting [23, 24, 39]. To circumvent these issues, we incorporate LoRA [37] by default. To further accelerate finetuning, we truncate the gradient to only back-propagate the last DDIM sampling step following [16]. Experiments are conducted on 4 NVIDIA A100s, with the batch size set to 8 and the learning rate set to 1 × 10−5. To strike a costperformance balance, we fine-tune InstructVideo with default parameters for 20k steps if not otherwise stated.

steps in Fig. 3. Examining the examples, we observe that the quality of videos generated by InstructVideo consistently outperforms the base model by a margin. Specifically, notable enhancements include 1) clearer and more coherent structures and scenes even if the animal is moving, exemplified by the walking cat and the swimming fish; 2) more appealing coloration, exemplified by the sunflower, the bee and the mountain goat; 3) an enhanced delineation of scene details, exemplified by the rock and grass on the cliff, and the texture of all the animals; and 4) improved video-text alignment, exemplified by the distinct portrayal of sunflowers and the bird’s reflections on the water. Remarkably, these advancements are achieved without compromising motion fluidity and the resultant videos can often surpass the video quality of the WebVid10M dataset. Notably, InstructVideo even attenuates watermarks present in WebVid10M. These qualitative leaps, consistently favored by human annotators, are attributed to the reward fine-tuning process, which effectively refines the video diffusion model.

Inference. After reward fine-tuning, we merge the LoRA weights into the ModelScopeT2V parameters to ensure that InstructVideo’s inference cost is identical to ModelScopeT2V [37]. For text-to-video generation, InstructVideo uses 20-step DDIM inference.

Comparison with other reward fine-tuning methods. This aims to validate the efficacy of reward fine-tuning conceptualized as an editing process. We compare with other representative reward fine-tuning methods, including policy gradient algorithm, DDPO [4], reward-weighted regression, RWR [43] and direct reward back-propagation method, DRaFT [16]. For DRaFT, we adopt DRaFT-1 for efficient fine-tuning. SegVR and TAR are employed for all compared methods to standardize reward signals. The comparative analysis on the evaluation set, presented in Fig. 5(a),

###### 4. Experiments

###### 4.1. Effectiveness of InstructVideo

Comparison with the base model ModelScopeT2V. To verify the efficacy of InstructVideo, we compare it with ModelScopeT2V [78] utilizing 20 and even 50 DDIM

White butterfly on violet flower closeup macro view.

Mule deer in Nebraska running across landscape.

Duck walks by the lake.

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

RWR

这里主要还是动物为主， 非动物也放一个？

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

DDPO

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

DRaFT

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

InstructVideo

Figure 4. Comparison of InstructVideo with other reward fine-tuning methods. We set D = 20 for all methods.

Comparison with Reward Fine-tuning Methods InstructVideo

The Effect of Varying Noise Level

The Effect of Varying tar

0.1 0.3 0.5 0.6 0.7 0.9

0.5 1.0 2.0

0.270

0.270

0.28

DRaFT

RWR

0.27

0.265

0.265

DDPO

RewardScore

RewardScore

RewardScore

0.26

0.260

0.260

0.25

0.255

0.24

0.255

0.23

0.250

0.250

0.22

0 10 20 30 40 50

0 10 20 30 40 50

0 10 20 30 40

Time (h)

Time (h)

Time (h)

(a) (b) (c)

Figure 5. (a) Comparison with other reward fine-tuning methods. (b) The effect of varying noise level τ. The vertical dashed line indicates 20k steps of optimization for τ = 0.6. (c) The effect of varying λtar.

leads to two findings: 1) Both RWR and DDPO exhibit a performance plateau after about 11 hours of fine-tuning, with further optimization failing to enhance or even deteriorating performance. 2) Direct reward back-propagation methods, including InstructVideo and DRaFT, initially lag during the first 11 hours but subsequently demonstrate fine-tuning efficiency, especially InstructVideo. To further validate the efficacy of our method, we provide visual comparisons in Fig. 4, where we adopt the optimal fine-tuned checkpoint for each method in Fig. 5(a). The examples reflect InstructVideo ’s superiority, evident in: 1) the clarity and coherence of structural and scenic elements, 2) the vibrancy of colors, 3) the precision in depicting intricacies, and 4) enhanced video-text alignment.

46 prompts for non-animals. Our comparative analysis of InstructVideo, the base model ModelScopeT2V and other reward fine-tuning methods is presented in Tab. 1. It is worth noting that during fine-tuning, SegVR and TAR are adopted by default to provide reward signals. Our observations are threefold: 1) Increasing the number of DDIM steps enhances the video quality. 2) Compared to the base model ModelScopeT2V in the second row, all methods improve reward scores for these unseen prompts. 3) Among all methods, InstructVideo outperforms other alternatives, affirming its superior generalization capabilities. To further demonstrate this, we provide visual comparisons in Fig. 6, featuring a set of unseen animal species, various sceneries, and human figures. The presented examples display an enhanced quality and exhibit an improvement in video-text alignment.

Generalization to unseen text prompts. We assessed the model’s generalization capabilities using two distinct sets of prompts: 1) those describing new animals and 2) those related to non-animals, none of which are present in the fine-tuning data. For this purpose, we curate about 4 prompts each for 6 new animal species following [59] and

User study. To further qualitatively compare the videos generated by InstructVideo and other methods, we conduct a user study comparing our methods with the base model ModelScopeT2V and the reward fine-tuned model

Land snail in southern of Thailand moving forward.

Fly over teen on rocks revealing frosted mountain.

Side view of elderly woman petting cat in window.

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

RWR

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

DDPO

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

DRaFT

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

InstructVideo

###### Figure 6. Comparison of InstructVideo’s generalization capabilities with other methods. We set D = 20 for all methods.

###### 4.2. Ablation Study

|Method|In-domain New Animals Non-animals<br><br>|
|---|---|
|ModelScopeT2V† ModelScopeT2V DDPO [4] RWR [43] DRaFT [16]|0.2542 ±0.0122 0.2541 ±0.0109 0.2610 ±0.0158 0.2506 ±0.0155 0.2502 ±0.0138 0.2557 ±0.0177 0.2511 ±0.0114 0.2524 ±0.0112 0.2564 ±0.0171 0.2550 ±0.0166 0.2517 ±0.0101 0.2625 ±0.0146 0.2584 ±0.0123 0.2561 ±0.0098 0.2644 ±0.0174<br><br>|
|InstructVideo|0.2717 ±0.0137 0.2645 ±0.0125 0.2682 ±0.0202|

The effect of varying noise level τ. To determine the optimal choice for noise level τ, we vary its value and evaluate its impact on reward scores using the evaluation data. We illustrate the results in Fig. 5(b). An increase in τ from 0.1 to 0.5 correlates with a progressive enhancement in the highest reward scores achieved by InstructVideo. However, excessively prolonged fine-tuning precipitates a sharp decline in generative performance. This phenomenon can be attributed to the limited edited space available to the model at lower noise levels, which constricts its ability to find the optimal output space as directed by the reward scores. When we further increase τ from 0.6 to 0.9, we observe that the reward score enhancement per hour becomes minor, suggesting challenges associated with generating from an extended sampling chain. Optimally, a noise level of τ = 0.6 strikes a balance, providing a feasible starting point for editing that still allows for a substantial exploration of the edited space. After 20k steps, more optimization leads to over-optimization [59], meaning that further steps can degrade the visual quality of the output despite potential increases in the reward score. Thus, we finalize on τ = 0.6 with 20k steps of fine-tuning.

- Table 1. Generalization to unseen text prompts. † denotes the model utilizes D = 50 while others adopt D = 20. ‘In-domain’ denotes in-domain animal prompts from the evaluation data.

|InstructVideo vs<br><br>|Quality Ours Tie Other|Alignment Ours Tie Other<br><br>|
|---|---|---|
|ModelScopeT2V DRaFT<br><br>|75.5% 15.5% 9.0%<br>76.0% 10.5% 13.5%<br>|28.5% 57.0% 14.5% 30.0% 51.5% 18.5%<br><br>|

- Table 2. User study. ‘Tie’ indicates instances where annotators think two videos are of comparable quality. ‘Quality’ and ‘Alignment’ represent video quality and video-text alignment.

DRaFT in Tab. 2. We recruited two participants who have related research experience in generative models to assess the quality of videos in terms of video quality and video-text alignment. To simplify the annotation process, participants were presented with pairs of videos and asked to identify which video was superior or if both were of equal quality. To ensure a comprehensive comparison, we chose 60 prompts from the 45 fine-tuning animal species, 20 prompts from the 6 new animal species and 20 prompts describing non-animals. More details are presented in the Appendix. We observe that our method consistently outperforms other methods. Specifically, improvements in video quality, a noted shortcoming of the base model, are more pronounced than improvements in video-text alignment.

The effect of varying λtar. To determine the optimal choice for λtar, we vary its value and evaluate its impact. We illustrate the results in Fig. 5(c). A relatively high value λtar, such as 2.0, results in fi decaying exponentially faster towards those border frames, thus providing diminished reward signals. A relatively low value λtar, such as 0.5, leads to fi decaying more gently towards those border frames, thus strengthening the reward signals. This equalized weighting across frames can destabilize fine-tuning, leading to a precipitous decline in reward scores. Subse-

Ablation on SegVR and TAR

How does the video evolve？ ModelScopeT2V

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

w/o SegVR

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

InstructVideo (5k steps)

w/o TAR

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

InstructVideo (10k steps)

InstructVideo

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

Figure 7. Ablation study on SegVR and TAR. This video shows a white dog walking in the park in slow motion.

InstructVideo (15k steps)

quent increases in scores do not necessarily indicate improved video quality but indicate quality degradation, as revealed by the rising variance. Thus, an appropriate coefficient to ensure stable fine-tuning is imperative and we finalize on λtar = 1.0.

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

InstructVideo (20k steps)

- Figure 8. The evolution of generated videos during fine-tuning. The video shows a bobtail dog walking.

0 10 20 30 40

Time (h)

0.24

0.25

0.26

0.27

0.28

RewardScore

The Effect of Utilizing Different Fine-tuning Data WebVid data

Collected data

- Figure 9. The effect of utilizing different fine-tuning data. The colored horizontal dashed lines denote the reward scores of different fine-tuning data, matched by the color of the respective curve.

Ablation on SegVR and TAR. To qualitatively verify the efficacy of SegVR and TAR, we present illustrative results in Fig. 7. Removing either SegVR or TAR results in a noticeable reduction in temporal modeling capabilities. This suggests that overly dense or excessively strong reward signals can lead to generation collapse. The degradation of modeling temporal dynamics often leads to the degraded quality of the individual frames due to the intertwined nature of spatial and temporal parameters. These observations underscore the critical roles of SegVR and TAR in maintaining fine-tuning stability.

###### 4.3. Further Analysis

The evolution of the generated videos during reward fine-tuning. To elucidate how the reward fine-tuning works, we present a visual progression in Fig. 8. The top row depicts a video generated without fine-tuning. All the frames in this video exhibit a lack of the dog’s fur texture. Moreover, a notable blurriness characterizes the third frame due to sudden and unanticipated motion, while the fourth frame suffers from a loss of facial clarity. As reward fine-tuning proceeds, we observe a noticeable enhancement in terms of all aspects mentioned above. Surprisingly, watermarks, which are consistently present across the dataset, also gradually fade. The resultant video is full of clear details and aesthetically pleasing coloration.

are comparable for two kinds of fine-tuning data, the employment of higher-quality data, i.e., WebVid10M, yields superior average reward scores compared to that obtained using the lower-quality counterpart. This suggests that superior fine-tuning data can facilitate reward fine-tuning.

Constraints of fine-tuning data on resultant video quality. Fig. 9 showcases that InstructVideo is capable of generating videos that achieve reward scores significantly exceeding those of the fine-tuning data itself, as denoted by the horizontal dashed lines. This observation leads us to conclude that the quality of the fine-tuning data does not impose a ceiling on the potential quality of the fine-tuned results. Our fine-tuning pipeline has the propensity to surpass the initial data quality, thus facilitating the generation of videos with substantially enhanced reward scores.

Impact of fine-tuning data quality on the fine-tuning results. To investigate this, we self-collect a dataset comprising an equivalent number of video-caption pairs for 45 animal species, which are employed for fine-tuning. The results are illustrated in Fig. 9. We employ horizontal dashed lines to indicate the quality of different data, inferred from reward scores. While the variance of the generated videos

###### 5. Conslusion

In this paper, we introduce InstructVideo, a method that pioneers instructing video diffusion models with human feedback by reward fine-tuning. We recast reward fine-tuning as an editing process that mitigates computational burden and enhances fine-tuning efficiency. We resort to image reward models to provide human feedback on generated videos and propose SegVR and TAR to ensure effective fine-tuning. Extensive experiments validate that InstructVideo not only elevates visual quality but also maintains robust generalization capabilities.

###### References

- [1] Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, JiaBin Huang, Jiebo Luo, and Xi Yin. Latent-shift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv preprint arXiv:2304.08477, 2023. 2
- [2] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, et al. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861, 2021. 1
- [3] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021. 1, 5
- [4] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 1, 2, 3, 4, 5, 7
- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 1, 2
- [6] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23206–23217, 2023. 2
- [7] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23040–23050,

2023. 2

- [8] Hong Chen, Xin Wang, Guanning Zeng, Yipeng Zhang, Yuwei Zhou, Feilin Han, and Wenwu Zhu. Videodreamer: Customized multi-subject text-to-video generation with disen-mix finetuning. arXiv preprint arXiv:2311.00990,

2023. 2

- [9] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation, 2023.

- [10] Tsai-Shien Chen, Chieh Hubert Lin, Hung-Yu Tseng, Tsung-Yi Lin, and Ming-Hsuan Yang. Motion-conditioned diffusion model for controllable video synthesis. arXiv preprint arXiv:2304.14404, 2023. 2
- [11] Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models, 2023. 2
- [12] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. arXiv preprint arXiv:2310.20700, 2023. 2
- [13] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017. 1
- [14] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017. 3
- [15] Ozg¨¨ un ¸Ci¸cek, Ahmed Abdulkadir, Soeren S Lienkamp, Thomas Brox, and Olaf Ronneberger. 3d u-net: learning dense volumetric segmentation from sparse annotation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19, pages 424–432. Springer, 2016. 3
- [16] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023. 1, 2, 3, 5, 7
- [17] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427, 2022. 2, 4
- [18] Hanze Dong, Wei Xiong, Deepanshu Goyal, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767,

2023. 1

- [19] Patrick Esser, Robin Rombach, and Bj¨orn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 3, 4
- [20] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. arXiv preprint arXiv:2302.03011, 2023. 2
- [21] Ying Fan and Kangwook Lee. Optimizing ddpm sampling with shortcut fine-tuning. arXiv preprint arXiv:2301.13362, 2023. 3
- [22] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. arXiv preprint arXiv:2305.16381, 2023. 3, 1

- [23] Tao Feng, Mang Wang, and Hangjie Yuan. Overcoming catastrophic forgetting in incremental object detection via elastic response distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9427–9436, 2022. 5
- [24] Tao Feng, Hangjie Yuan, Mang Wang, Ziyuan Huang, Ang Bian, and Jianzhou Zhang. Progressive learning without forgetting. arXiv preprint arXiv:2211.15215, 2022. 5
- [25] Tsu-Jui Fu, Licheng Yu, Ning Zhang, Cheng-Yang Fu, Jong-Chyi Su, William Yang Wang, and Sean Bell. Tell me what happened: Unifying text-guided video completion via multimodal masked video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10681–10692, 2023. 2
- [26] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, Ming-Yu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22930–22941, 2023. 2
- [27] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023. 2
- [28] Audrunas Gruslys, R´emi Munos, Ivo Danihelka, Marc Lanctot, and Alex Graves. Memory-efficient backpropagation through time. Advances in neural information processing systems, 29, 2016. 3
- [29] Jiaxi Gu, Shicong Wang, Haoyu Zhao, Tianyi Lu, Xing Zhang, Zuxuan Wu, Songcen Xu, Wei Zhang, Yu-Gang Jiang, and Hang Xu. Reuse and diffuse: Iterative denoising for text-to-video generation. arXiv preprint arXiv:2309.03549, 2023. 2
- [30] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for highfidelity long video generation. 2022.
- [31] Yingqing He, Menghan Xia, Haoxin Chen, Xiaodong Cun, Yuan Gong, Jinbo Xing, Yong Zhang, Xintao Wang, Chao Weng, Ying Shan, et al. Animate-a-story: Storytelling with retrieval-augmented video generation. arXiv preprint arXiv:2307.06940, 2023. 2
- [32] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5
- [33] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 1, 2, 3
- [34] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 1, 2
- [35] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. arXiv preprint arXiv:2204.03458, 2022. 1, 2
- [36] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 2

- [37] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2, 5, 1
- [38] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023. 2
- [39] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka GrabskaBarwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017. 5
- [40] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. arXiv preprint arXiv:2305.01569, 2023. 2
- [41] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In International Conference on Learning Representations, 2021. 1
- [42] Ariel Lapid, Idan Achituve, Lior Bracha, and Ethan Fetaya. Gd-vdm: Generated depth for better diffusion-based video generation. arXiv preprint arXiv:2306.11173, 2023. 2
- [43] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning textto-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023. 1, 2, 3, 5, 7
- [44] Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, and Shane Legg. Scalable agent alignment via reward modeling: a research direction. arXiv preprint arXiv:1811.07871, 2018. 1, 3
- [45] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020. 1
- [46] Chunyi Li, Zicheng Zhang, Haoning Wu, Wei Sun, Xiongkuo Min, Xiaohong Liu, Guangtao Zhai, and Weisi Lin. Agiqa-3k: An open database for ai-generated image quality assessment. arXiv preprint arXiv:2306.04717,

2023. 2

- [47] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, pages 12888–12900. PMLR, 2022. 2
- [48] Yitong Li, Martin Min, Dinghan Shen, David Carlson, and Lawrence Carin. Video generation from text. In Proceedings of the AAAI conference on artificial intelligence, 2018. 2
- [49] Binhui Liu, Xin Liu, Anbo Dai, Zhiyong Zeng, Zhen Cui, and Jian Yang. Dual-stream diffusion net for text-to-video generation. arXiv preprint arXiv:2308.08316, 2023. 2
- [50] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. arXiv preprint arXiv:2303.04761, 2023. 2

- [51] Shijie Ma, Huayi Xu, Mengjian Li, Weidong Geng, Meng Wang, and Yaxiong Wang. Optimal noise pursuit for augmenting text-to-video generation. arXiv preprint arXiv:2311.00949, 2023. 2
- [52] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Ying Shan, Xiu Li, and Qifeng Chen. Follow your pose: Poseguided text-to-video generation using pose-free videos. arXiv preprint arXiv:2304.01186, 2023. 2
- [53] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 2, 4
- [54] Gaurav Mittal, Tanya Marwah, and Vineeth N Balasubramanian. Sync-draw: Automatic video generation using deep recurrent attentive architectures. In Proceedings of the 25th ACM international conference on Multimedia, pages 1096–1104, 2017. 2
- [55] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid Hoshen. Dreamix: Video diffusion models are general video editors. arXiv preprint arXiv:2302.01329, 2023. 2
- [56] OpenAI. GPT-4 technical report, 2023. 1
- [57] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022. 1, 3
- [58] Yingwei Pan, Zhaofan Qiu, Ting Yao, Houqiang Li, and Tao Mei. To create what you tell: Generating videos from captions. In Proceedings of the 25th ACM international conference on Multimedia, pages 1789–1798, 2017. 2
- [59] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023. 1, 3, 6, 7
- [60] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. arXiv preprint arXiv:2303.09535, 2023. 2
- [61] Zhiwu Qing, Shiwei Zhang, Jiayu Wang, Xiang Wang, Yujie Wei, Yingya Zhang, Changxin Gao, and Nong Sang. Hierarchical spatio-temporal decoupling for text-to-video generation. arXiv preprint arXiv:2312.04483, 2023. 2
- [62] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2
- [63] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 1, 2, 3

- [64] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical Image Computing and ComputerAssisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015. 3
- [65] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. Mm-diffusion: Learning multi-modal diffusion models for joint audio and video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10219–10228, 2023. 2
- [66] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 1
- [67] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 1, 2

- [68] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3626–3636, 2022. 2
- [69] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Computer Vision, pages 2256–2265. PMLR, 2015. 1, 2, 3
- [70] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2, 3, 4
- [71] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1
- [72] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020. 3
- [73] Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable diffusion. arXiv preprint arXiv:2305.11846, 2023. 2
- [74] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. 1
- [75] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, pages 5998–6008, 2017. 1

- [76] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399,

2022. 2

- [77] Dimitri von R¨utte, Elisabetta Fedele, Jonathan Thomm, and Lukas Wolf. Fabric: Personalizing diffusion models with iterative feedback. arXiv preprint arXiv:2307.10159, 2023. 3
- [78] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 1, 2, 3, 5
- [79] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. Temporal segment networks: Towards good practices for deep action recognition. In ECCV, pages 20–36. Springer, 2016. 2, 4
- [80] Wenjing Wang, Huan Yang, Zixi Tuo, Huiguo He, Junchen Zhu, Jianlong Fu, and Jiaying Liu. Videofactory: Swap attention in spatiotemporal diffusions for text-to-video generation. arXiv preprint arXiv:2305.10874, 2023. 2
- [81] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018, 2023. 1, 2, 3
- [82] Xiang Wang, Shiwei Zhang, Han Zhang, Yu Liu, Yingya Zhang, Changxin Gao, and Nong Sang. Videolcm: Video latent consistency model. arXiv preprint arXiv:2312.09109, 2023. 2
- [83] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 1, 2
- [84] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. arXiv preprint arXiv:2312.04433, 2023. 2
- [85] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021. 2
- [86] Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, and Nan Duan. N¨uwa: Visual synthesis pretraining for neural visual world creation. In European Conference on Computer Vision, pages 720–736. Springer,

2022. 2

- [87] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 1, 4
- [88] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human pref-

- erence score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023. 2, 4, 5
- [89] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-toimage models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2096–2105, 2023. 1, 2, 3
- [90] Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Yong Zhang, Yingqing He, Hanyuan Liu, Haoxin Chen, Xiaodong Cun, Xintao Wang, et al. Make-your-video: Customized video generation using textual and structural guidance. arXiv preprint arXiv:2306.00943, 2023. 2
- [91] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. 2023.
- [92] Zhen Xing, Qi Dai, Han Hu, Zuxuan Wu, and Yu-Gang Jiang. Simda: Simple diffusion adapter for efficient video generation. arXiv preprint arXiv:2308.09710, 2023. 2
- [93] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-toimage generation. arXiv preprint arXiv:2304.05977, 2023. 1, 2, 3
- [94] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 2
- [95] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023. 2
- [96] Sihyun Yu, Jihoon Tack, Sangwoo Mo, Hyunsu Kim, Junho Kim, Jung-Woo Ha, and Jinwoo Shin. Generating videos with dynamics-aware implicit generative adversarial networks. arXiv preprint arXiv:2202.10571, 2022. 2
- [97] Sihyun Yu, Kihyuk Sohn, Subin Kim, and Jinwoo Shin. Video probabilistic diffusion models in projected latent space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18456– 18466, 2023. 2
- [98] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation, 2023. 2
- [99] Qinsheng Zhang, Molei Tao, and Yongxin Chen. gddim: Generalized denoising diffusion implicit models. arXiv preprint arXiv:2206.05564, 2022. 1, 2
- [100] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023. 2
- [101] Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, ChiaChih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, et al. Hive: Harnessing human feedback for instructional visual editing. arXiv preprint arXiv:2303.09618, 2023. 3

- [102] Min Zhao, Rongzhen Wang, Fan Bao, Chongxuan Li, and Jun Zhu. Controlvideo: Adding conditional control for one shot text-to-video editing. arXiv preprint arXiv:2305.17098, 2023. 2
- [103] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. Motiondirector: Motion customization of text-to-video diffusion models. arXiv preprint arXiv:2310.08465, 2023. 2
- [104] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 2
- [105] Junchen Zhu, Huan Yang, Huiguo He, Wenjing Wang, Zixi Tuo, Wen-Huang Cheng, Lianli Gao, Jingkuan Song, and Jianlong Fu. Moviefactory: Automatic movie creation from text using large generative models for language and images. arXiv preprint arXiv:2306.07257, 2023. 2

## InstructVideo: Instructing Video Diffusion Models with Human Feedback Supplementary Material

In this Appendix, we first detail on the potential societal impact (Appendix A), and limitations and potential future work (Appendix B). Subsequently, we provide additional details about implementing LoRA (Appendix C) and user study (Appendix D). Moreover, we provide more visualization results to demonstrate the efficacy of InstructVideo (Appendix E). Finally, we provide experiments to validate the efficacy of generation with 50-step DDIM inference (Appendix F), reward finetuning with 50-step DDIM inference (Appendix G) and InstructVideo’s adaptation to other reward functions (Appendix H).

###### A. Potential Societal Impact

InstructVideo, as the pioneering effort in instructing video diffusion models with human feedback, prioritizes users’ preferences for AI-generated content. We conducted this research, motivated by the varied quality of generated videos induced by the varied quality of the curated webscale datasets. Pre-training models on such unfiltered data can lead to outputs that deviate from human preferences. In the context of the broader research community, we advocate that video generation systems, akin to other generative models like language models [56, 57], should prioritize ethical considerations and human values.

Moreover, conventional video generation systems might not always resonate with all users in terms of aesthetic style and often struggle with accurately reflecting textual prompts. InstructVideo steps in as a human-centered technology, efficiently addressing these issues in a data- and computation-efficient way and opening up possibilities for commercial applications, particularly in sectors like education and entertainment.

However, as InstructVideo primarily targets research, aiming at investigating the practicality of aligning video diffusion models with human preferences, its deployment to any circumstance beyond research should be approached with thorough oversight and evaluation to ensure responsible and ethical use.

###### B. Limitation and Future work

We recognize that InstructVideo, as an initial endeavor in this area, comes with its limitations. Although we validate the efficacy of image reward models, we anticipate that specialized video reward models capturing human preferences might be even more superior since they evaluate one generated video as a whole. Additionally, as a common issue mentioned in previous works [4, 18, 22, 59], reward

fine-tuning carries a risk of over-optimization, meaning that excessive optimization steps will result in the degradation of the video quality despite potential increases in the reward score. Addressing these aspects presents avenues for future research, including the development of a more advanced video reward model and the design of strategic mechanisms to identify and ameliorate over-optimization.

###### C. More Details about Implementing LoRA

To instantiate LoRA [37] for efficient tuning, we adopt the implementation used in Diffusers2. Specifically, we configure the intrinsic rank within LoRA to 4 to ensure fast processing. LoRA modifications are applied to every Transformer [75] layer within our model, targeting the linear layers responsible for query, key, value, and output projections. ModelScopeT2V [78] contains 1,347.44M parameters, whereas the additional parameters introduced by adding LoRA amount to only 1.58M – approximately 0.1% of the total ModelScopeT2V parameters.

###### D. More Details about User Study

In the main paper, we present a user study to demonstrate the effectiveness of InstructVideo. This study involves a comparative analysis of videos generated by InstructVideo and other methods, focusing on two key aspects: video quality and video-text alignment. For video quality, we asked annotators to evaluate: 1) The overall visual quality of the videos, 2) Alignment with general human aesthetic preferences, such as pleasing visuals, texture and details, and 3) The smoothness and consistency in terms of structural and color transitions within the video. Regarding video-text alignment, annotators are tasked with determining the extent to which the generated videos accurately and clearly represent the content of the provided text prompts. This assessment included evaluating the depiction of entities, attributes, relationships, and motions as described in the prompts. To simplify the evaluation process, annotators are asked to perform pairwise comparisons between videos, thereby streamlining their task to direct contrasts rather than isolated assessments.

###### E. More Visualization Results

We provide more visualization results to exemplify the conclusions we draw in the main paper, including: 1) More results demonstrating how the generated videos evolve as

2https://github.com/huggingface/diffusers/blob/ main/src/diffusers/models/lora.py

|Method<br><br>|In-domain New Animals Non-animals|
|---|---|
|ModelScopeT2V ModelScopeT2V†<br><br>|0.2506 ±0.0155 0.2502 ±0.0138 0.2557 ±0.0177 0.2542 ±0.0122 0.2541 ±0.0109 0.2610 ±0.0158|
|InstructVideo InstructVideo†<br><br>|0.2717 ±0.0137 0.2645 ±0.0125 0.2682 ±0.0202 0.2736 ±0.0125 0.2664 ±0.0131 0.2739 ±0.0210|

Table A.1. Generation with 50-step DDIM inference after finetuning with 20-step DDIM inference. † denotes the model utilizes D = 50 while others adopt D = 20. ‘In-domain’ denotes indomain animal prompts from the evaluation data.

the fine-tuning process proceeds as shown in Fig. A.2; 2) More results showcasing the comparison between InstructVideo and the base model ModelScopeT2V as illustrated in Fig. A.3; 3) More results exemplifying the comparison between InstructVideo and other reward fine-tuning methods as shown in Fig. A.4; 4) More results showing the InstructVideo’s generalization capabilities to unseen text prompts as shown in Fig. A.5.

###### F. 50-Step Generation with InstructVideo

To showcase the adaptability and effectiveness of InstructVideo, we conduct experiments using a 50-step DDIM inference for generation after initial finetuning with 20-step DDIM inference. The results are shown in Tab. A.1. We observe that InstructVideo, despite being fine-tuned with a 20-step protocol, remains effective under a longer 50-step DDIM inference protocol, as demonstrated by the boosted reward scores. We present several cases to further illustrate InstructVideo’s efficacy as shown in Fig. A.6. We observe that both inference schemes can significantly improve over the base model and adopting more inference steps can occasionally lead to better results.

###### G. 50-Step Reward Fine-tuning

To assess the adaptation of InstructVideo to different DDIM steps, we experiment on reward fine-tuning with the commonly-used 50-step DDIM inference and evaluate its 20-step generation quality for a fair comparison. We present the results in Fig. A.1. The results demonstrate that InstructVideo could be optimized towards higher reward scores in both settings. However, utilizing 50 steps degrades the fine-tuning efficiency, likely due to the increased computation brought by longer sampling chains.

###### H. Adaptation to Other Reward Functions

In the main paper, we focus on the utilization of HPSv2 [88]. To further validate the generalization of our method to other reward functions, we explore the application of ImageReward [93] as our reward model. ImageReward is a general-purpose text-to-image human preference reward model, fine-tuned on BLIP [47]. We perform reward

Reward Fine-tuning with 50 DDIM Steps

20 steps 50 steps

0.28

0.27

RewardScore

0.26

0.25

0.24

0 10 20 30 40 50

Time (h)

Figure A.1. Reward finetuning with 50-step DDIM inference.

fine-tuning as HPSv2 and present results in Fig. A.7. We observe that the quality of the videos are generally boosted in terms of structures, color vibrancy and details, despite that the stylistic aspects of the videos differ from those finetuned with HPSv2.

How does the video evolve？

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

ModelScopeT2V

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

InstructVideo (5k steps)

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

InstructVideo (10k steps)

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

InstructVideo (15k steps)

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

InstructVideo (20k steps)

Figure A.2. More examples showing the evolution of generated videos during fine-tuning.

Comparison with The Base Model

ModelScopeT2V (D=20) ModelScopeT2V (D=50) InstructVideo (D=20)

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

Bee collects honey in flower at morning.

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

Bighorn sheep ovis canadensisram between snowcovered sage bushes.

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

Bantam or chicken on the garden.

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

Great tit in bird feeder.

Figure A.3. More examples showing the comparion of InstructVideo with the base model ModelScopeT2V. ModelScopeT2V utilizes 20 and 50 DDIM steps.

Comparison with Other reward fine-tuning

Horse grazing on pasture and eating green grass.

Close up grey rabbit eating corn.

Brush turkey head comes toward bower.

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

这里主要还是动物为主， 非动物也放一个？

RWR

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

DDPO

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

DRaFT

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

InstructVideo

###### Figure A.4. More examples showing the comparison of InstructVideo with other reward fine-tuning methods. We set D = 20 for all methods.

#### Generalization to unseen prompts

Portrait of cheetah acinonyx jubatus.

Drone fly up near old historical white Christianity church.

Lobster moth caterpillar is eating leaf of host plant.

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

RWR

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

DDPO

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

DRaFT

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

InstructVideo

Figure A.5. More examples showing the comparison of InstructVideo’s generalization capabilities with other methods. We set D = 20 for all methods.

50-step DDIM inference

ModelScopeT2V (D=50) InstructVideo (D=20) InstructVideo (D=50)

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

Close up of llama in autumn.

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

Water bird on the lake in spring podiceps cristatus.

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

Goat in a green summer.

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

Fish swimming by in kelp.

Figure A.6. Generation with 50-step DDIM inference after fine-tuning with 20-step DDIM inference.

### ImageReward

ModelScopeT2V InstructVideo

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

A zebra eating grass on nature.

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

Close up of blacknose sheep head.

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

An beetle on the branch in close up.

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

Russian blue cat playfully biting and scratching owners hand.

Figure A.7. Comparison of InstructVideo fine-tuned using ImageReward with the base model ModelScopeT2V. We set D = 20 for two methods.

