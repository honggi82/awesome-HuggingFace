## VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models

# arXiv:2312.00845v1[cs.CV]1Dec2023

Hyeonho Jeong1* Geon Yeong Park2* Jong Chul Ye1,2 1Kim Jaechul Graduate School of AI, 2Bio and Brain Engineering Korea Advanced Institute of Science and Technology (KAIST) * indicates co-first authors {hyeonho.jeong, pky3436, jong.ye}@kaist.ac.kr

[Figure 1]

Figure 1. Using only a single video portraying any type of motion, our Video Motion Customization framework allows for generating a wide variety of videos characterized by the same motion but in entirely distinct contexts and better spatial/temporal resolution. 8-frame input videos are translated to 29-frame videos in different contexts while closely following the target motion. The visualized frames for the first video are at indexes 1, 9, and 17. A comprehensive view of these motions in the form of videos can be explored at our project page.

### Abstract

Text-to-video diffusion models have advanced video generation significantly. However, customizing these models to generate videos with tailored motions presents a substantial challenge. In specific, they encounter hurdles in (a) accurately reproducing motion from a target video, and (b) creating diverse visual variations. For example, straightforward extensions of static image customization methods

to video often lead to intricate entanglements of appearance and motion data. To tackle this, here we present the Video Motion Customization (VMC) framework, a novel one-shot tuning approach crafted to adapt temporal attention layers within video diffusion models. Our approach introduces a novel motion distillation objective using residual vectors between consecutive frames as a motion reference. The diffusion process then preserves low-frequency motion trajectories while mitigating high-frequency motion-

unrelated noise in image space. We validate our method against state-of-the-art video generative models across diverse real-world motions and contexts. Our codes, data and the project demo can be found at https://video-motioncustomization.github.io.

### 1. Introduction

The evolution of diffusion models [13, 28, 31] has significantly advanced Text-to-Image (T2I) generation, notably when paired with extensive text-image datasets [3, 25]. While cascaded diffusion pipelines [2, 9, 14, 27, 33, 37, 39] have extended this success to Text-to-Video (T2V) generation, current models lack the ability to replicate specific motions or generate diverse variations of the same motion with distinct visual attributes and backgrounds. Addressing this, we tackle the challenge of Motion Customization [38]—adapting pre-trained Video Diffusion Models (VDM) to produce motion-specific videos in different contexts, while maintaining the same motion patterns of target subjects.

Given a few subject images for reference, appearance customization [8, 19, 23, 24, 26, 35] in generative models aims to fine-tune models to generate subject images in diverse contexts. However, these approaches, despite varying optimization objectives, commonly strive for faithful image (frame) reconstruction by minimizing the ℓ2-distance between predicted and ground-truth noise. This may lead to the entangled learning of appearance and motion.

To tackle this, we present VMC, a new framework aimed at adapting pre-trained VDM’s temporal attention layers via our proposed Motion Distillation objective. This approach utilizes residual vectors between consecutive (latent) frames to obtain the motion vectors that trace motion trajectories in the target video. Consequently, we fine-tune VDM’s temporal attention layers to align the ground-truth image-space residuals with their denoised estimates, which equivalently aligns predicted and ground-truth source noise differences within VDM. This enables lightweight and fast one-shot training. To further facilitate the appearanceinvariant motion distillation, we transform faithful text prompts into appearance-invariant prompts, e.g. "A bird is flying above a lake in the forest" Ñ "A bird is flying" in Fig. 1. This encourages the modules to focus on the motion information and ignore others, such as appearance, distortions, background, etc. During inference, our procedure initiates by sampling key-frames using the adapted key-frame generation UNet, followed by temporal interpolation and spatial superresolution. To summarize, VMC makes the following key contributions:

• We introduce a novel fine-tuning strategy which focuses solely on temporal attention layers in the key-frame gen-

eration module. This enables lightweight training (15GB vRAM) and fast training (ă 5 minutes).

- • To our knowledge, we mark a pioneering case of finetuning only the temporal attention layers in video diffusion models, without optimizing spatial self or crossattention layers, while achieving successful motion customization.
- • We introduce a novel motion distillation objective that leverages the residual vectors between consecutive (latent) frames as motion vectors.
- • We present the concept of appearance-invariant prompts, which further facilitates the process of motion learning when combined with our motion distillation loss.

### 2. Preliminaries

Diffusion Models. Diffusion models aim to generate samples from the Gaussian noise through iterative denoising processes. Given a clean sample x0 „ pdatapxq, the forward process is defined as a Markov chain with forward conditional densities

ppxt | xt´1q “ Npxt | βtxt´1,p1 ´ βtqIq ptpxt | x0q “ Npxt |

?

α¯x0,p1 ´ α¯qIq,

(1)

where xt P Rd is a noisy latent variable at a timestep t that has the same dimension as x0, and βt denotes an increasing sequence of noise schedule where αt :“ 1 ´ βt and α¯t :“ Πti“1αi. Then, the goal of diffusion model training is to obtain a residual denoiser ϵθ:

t„ptpxt | x0q,x0„pdatapx0q,ϵ„Np0,Iq“∥ϵθpxt,tq ´ ϵ∥‰

Ex

.

min

θ

(2) It can be shown that this epsilon matching in (2) is equivalent to the Denoising Score Matching (DSM [16, 30]) with different parameterization:

“

log ptpxt | x0q ‰

stθpxtq ´ ∇xt

, (3)

Ex

min

t,x0,ϵ

θ

?α¯tx0

t´

where sθ˚pxt,tq » ´x

ϵθ˚pxt,tq.

1´α¯ “ ´?1´1 α¯

t

The reverse sampling from qpxt´1|xt,ϵθ˚pxt,tqq is then achieved by

?αt ´xt ´

ϵθ˚pxt,tq¯ ` β˜tϵ, (4)

1 ´ αt ?1 ´ α¯t

1

xt´1 “

where ϵ „ Np0,Iq and β˜t :“ 1´α¯

1´α¯t βt. To accelerate sampling, DDIM [29] further proposes another sampling method as follows:

t´1

?α¯t´1xˆ0ptq`b1 ´ α¯t´1 ´ η2β˜t2ϵθ˚pxt,tq`ηβ˜tϵ,

xt´1 “

(5)

[Figure 2]

- Figure 2. Overview. The proposed Video Motion Customization (VMC) framework distills the motion trajectories from the residual

between consecutive (latent) frames, namely motion vector δvnt for t ě 0. We fine-tune only the temporal attention layers of the keyframe generation model by aligning the ground-truth and predicted motion vectors. After training, the customized key-frame generator is leveraged for target motion-driven video generation with new appearances context, e.g. "A chicken is walking in a city".

where η P r0,1s is a stochasticity parameter, and xˆ0ptq is the denoised estimate which can be equivalently derived using Tweedie’s formula [6]:

1

?α¯tpxt ´

xˆ0ptq :“

?

1 ´ α¯tϵθ˚pxt,tqq. (6)

For a text-guided Diffusion Model, the training objective is often given by:

“∥ϵθpxt,t,cq ´ ϵ∥‰

, (7)

Ex

min

t,x0,ϵ,c

θ

where c represents the textual embedding. Throughout this paper, we will often omit c from ϵθpxt,t,cq if it does not lead to notational ambiguity.

Video Diffusion Models. Video diffusion models [12, 14, 37] further attempt to model the video data distribution. Specifically, Let pvnqnPt1,...,Nu represents the N-frame input video sequence. Then, for a given n-th frame vn P Rd, let v1:N P RNˆd represents a whole video vector. Let vnt “

?1 ´ α¯tϵnt represents the n-th noisy frame latent sampled from ptpvnt |vnq, where ϵnt „ Np0,Iq. We similarly define pvnt qnP1,...,N, v1:t N, and ϵ1:t N. The goal of video diffusion model training is then to obtain a residual denoiser ϵθ with textual condition c and video input that satisfies:

?α¯tvn `

“

‰

ϵθpv1:t N,t,cq ´ ϵ1:t N

, (8)

Ev1:N

min

t ,v1:N,ϵ1:t N,c

θ

where ϵθpv1:t N,t,cq,ϵ1:t N P RNˆd. In this work, we denote the predicted noise of n-th frame as ϵnθpv1:t N,t,cq P Rd.

In practice, contemporary video diffusion models often employ cascaded inference pipelines for high-resolution

outputs. For instance, [37] initially generates a lowresolution video with strong text-video correlation, further enhancing its resolution via temporal interpolation and spatial super-resolution modules.

In exploring video generative tasks through diffusion models, two primary approaches have emerged: foundational Video Diffusion Models (VDMs) or leveraging pretrained Text-to-Image (T2I) models. To extend image diffusion models to videos, several architectural modifications are made. Typically, U-Net generative modules integrate temporal attention blocks after spatial attentions [12]. Moreover, 2D convolution layers are inflated to 3D convolution layers by altering kernels [12].

### 3. Video Motion Customization

Given an input video, our main goal is to (a) distill the motion patterns M˚ of target subjects, and (b) customize the input video in different contexts while maintaining the same motion patterns M˚, e.g. Sharks w/ motion M˚ Ñ Airplanes w/ motion M˚, with minimal computational costs.

To this end, we propose a novel video motion customization framework, namely VMC, which leverages cascaded video diffusion models with robust temporal priors. One notable aspect of the proposed framework is that we perform fine-tuning only on the key-frame generation module, also referred to as the T2V base model, within the cascaded VDMs, which guarantees computational and memory efficiency. Specifically, within the key-frame generation model, our fine-tuning process only targets the temporal attention layers. This facilitates adaptation while preserving the model’s inherent capacity for generic synthesis. Notably, we freeze the subsequent frame interpolation and spatial super-resolution modules as-is (Fig. 2).

[Figure 3]

- Figure 3. Training. The proposed framework aims to learn motion

by δϵnt -alignment using (16) or (17). Note that we only fine-tune the temporal attention layers in the key-frame generation U-Net. The blue circle represents the diffusion forward process.

#### 3.1. Temporal Attention Adaptation

In order to distill the motion M˚, we first propose a new objective function for temporal attention adaptation using residual cosine similarity. Our intuition is that residual vectors between consecutive frames may include information about the motion trajectories.

Let pvnqnPt1,...,Nu represents the N-frame input video sequence. As defined in Section 2, for a given noisy video latent vector v1:t N with ϵ1:t N, let vnt represents the n-th noisy frame latent sampled from ptpvnt |vnq with ϵnt . We will interchangeably use vn and vn0 for notational simplicity. Likewise, vnt `c is defined as vnt , with c ą 0 representing the fixed frame stride. Then, we define the frame residual vector at time t ě 0 as

δvnt :“ vnt `c ´ vnt , (9)

where we similarly define the epsilon residual vector δϵnt . In the rest of the paper, we interchangeably use frame residual vector and motion vector.

We expect that these motion vectors may encode information about motion patterns, where such information may vary depending on the time t and its corresponding noise level. The difference vector δvnt can be delineated as:

?

?α¯tpv0n`c ´ vn0q `

1 ´ α¯tpϵnt `c ´ ϵnt q “

δvnt “

(10)

?

?α¯tδvn0 `

1 ´ α¯tδϵnt ,

where δϵnt is normally distributed with zero mean and 2I variance. In essence, δvnt can be acquired through the following diffusion kernel:

?α¯tδvn0,2p1 ´ α¯tqIq. (11)

ppδvnt | δvn0q “ Npδvnt |

In light of this, our goal is to transfer motion information to the temporal attention layers by leveraging the motion vectors. For this, we first simulate the motion vectors using video diffusion models. Specifically, as similarly done in (6), the denoised video vector estimates vˆ1:0 Nptq can be derived by applying Tweedie’s formula:

?

?α¯t`

1 ´ α¯tϵθpv1:t N,tq˘

1

vˆ1:0 Nptq :“

v1:t N ´

, (12)

where vˆ1:0 Nptq is an empirical Bayes optimal posterior expectation Erv1:0 N | v1:t Ns. Then, the denoised motion vector estimate δvˆn0 can be defined in terms of δvnt and δϵnθpv1:t N,tq by using (12):

?α¯t´δvnt ´

1 ´ α¯tδϵnθ,t¯, (13)

?

1

δvˆn0ptq :“

where δϵnθpv1:t N,tq :“ ϵnθ`cpv1:t N,tq ´ ϵnθpv1:t N,tq is abbreviated as δϵnθ,t for notational simplicity. Similarly, one can obtain ground-truth motion vector δvn0 in terms of δvnt and δϵnt by using (10):

1 ´ α¯tδϵnt ¯. (14)

?α¯t´δvnt ´

?

1

δvn0 “

Then, our objective is to finetune θ by aligning the motion vector δvn0 and its denoised estimate δvˆn0ptq:

δvn0,δvˆn0ptq˘ı, (15)

”ℓalign

`

Et,n,ϵn

min

t ,ϵnt `c

θ

with a loss function ℓalign : Rd ˆ Rd Ñ R. By using ℓ2distance for ℓalign, this is equivalent to matching δϵnθ,t and δϵnt :

`

δvn0,δvˆn0ptq˘ “

1 ´ α¯t α¯t

δϵnt ´ δϵnθ,t 2 . (16)

ℓalign

Notably, aligning the ground-truth and predicted motion vectors translates into aligning epsilon residuals.

While this objective demonstrates effective empirical performance, our additional observations indicate that using ℓcospδϵnt ,δϵnθ,tq may further improve the distillation, where ℓcospx,yq “ 1 ´ }xxx}},yyy} for x,y P Rd (more analysis in section 4.3). Accordingly, our optimization framework is finally defined as follows:

t ,ϵnt`crℓcospδϵnt ,δϵnθ,tqs. (17) Thus, the proposed optimization framework aims to maximize the residual cosine similarity between δϵnt and δϵnθ,t. In our observation, aligning the image-space residuals (δvn0 and δvˆn0ptq) corresponds to aligning the latent-space epsilon residuals (δϵnt and δϵnθ,t) across varying time steps. This relationship stems from expressing the motion vector δvn0 and its estimation δvˆn0ptq in terms of δvnt , δϵnt , and δϵnθ,t. Consequently, the proposed optimization framework fine-tunes temporal attention layers by leveraging diverse diffusion latent spaces at time t which potentially contains multi-scale rich descriptions of video frames. Hence, this optimization approach can be seamlessly applied to video diffusion models trained using epsilon-matching, thanks to the equivalence between δϵnt -matching and δvn0-matching. Practically, we exclusively fine-tune the temporal attention layers θTA Ă θ, originally designed for dynamic temporal data assimilation [35]. The frame stride remains fixed at c “ 1 across all experiments.

Et,n,ϵn

min

θ

[Figure 4]

- Figure 4. Appearance-invariant Prompt. Comparison of input reconstruction with and without appearance-invariant prompt: (a) and (b) depict sampled low-resolution (64x40) keyframes. For

- (a), the training prompt used was “A cat is roaring,” while for
- (b), the training prompt was “A cat is roaring on the grass under the tree.” Our appearance-invariant prompt enables the removal of background information that can disturb motion distillation.

#### 3.2. Appearance-invariant Prompts

In motion distillation, it is crucial to filter out disruptive variations that are unrelated to motion. These variations may include changes in appearance and background, distortions, consecutive frame inconsistencies, etc. To achieve this, we further utilize appearance-invariant prompts. Diverging from traditional generative customization frameworks [23, 24, 35, 38] that rely on text prompts that “faithfully” describe the input image or video during model finetuning, our framework purposedly employs “unfaithful” text prompts during the training phase. Specifically, our approach involves the removal of background information. For instance, the text prompt ‘a cat is roaring on the grass under the tree’ is simplified to ‘a cat is roaring’ as presented in Fig. 4. This reduces background complexity as in Fig. 4a comapred to Fig. 4b, facilitating the application of new appearance in motion distillation.

#### 3.3. Inference Pipeline

Once trained, in the inference phase, our process begins by computing inverted latents from the input video through DDIM inversion. Subsequently, the inverted latents are fed into the temporally fine-tuned keyframe generation model, yielding short and low-resolution keyframes. These keyframes then undergo temporal extension using the unaltered frame interpolation model. Lastly, the interpolated frames are subjected to spatial enlargement through the spatial super-resolution model. Overview of the process is depicted in Fig. 2.

### 4. Experiments 4.1. Implementation Details

In our experiments, we choose Show-1 [37] as our VDM backbone and its publicly available pre-trained weights 1. All experiments were conducted using a single NVIDIA RTX 6000 GPU. VMC with Show-1 demonstrates efficient resource usage, requiring only 15GB of vRAM during

1https://huggingface.co/showlab/show-1-base

mixed-precision training [20], which is completed within 5 minutes. During inference, generating a single video comprising 29 frames at a resolution of 576 x 320 consumes 18GB of vRAM and takes approximately 12 minutes.

#### 4.2. Baseline Comparisons

Dataset Selection. In our experiments, we draw upon a dataset that comprises 24 videos. These videos encompass a broad spectrum of motion types occurring in various contexts, encompassing vehicles, humans, birds, plants, diffusion processes, mammals, sea creatures, and more. This diversity provides a comprehensive range of motion scenarios for our assessment. Out of these 24 videos, 13 are sourced from the DAVIS dataset [21], 10 from the WebVid dataset [1], and 1 video is obtained from LAMP [36].

Baselines. Our method is compared against four contemporary baselines that integrate depth map signals into the diffusion denoising process to assimilate motion information. Notably, our approach operates without the necessity of depth maps during both training and inference, in contrast to these baseline methods.

Specifically, VideoComposer (VC) [32] is an opensource latent-based video diffusion model tailored for compositional video generation tasks. Gen-1 [7] introduces a video diffusion architecture incorporating additional structure and content guidance for video-to-video translation. In contrast to our targeted fine-tuning of temporal attention, Tune-A-Video (TAV) [35] fine-tunes self, cross, and temporal attention layers within a pre-trained, but inflated T2I model on input videos. Control-A-Video (CAV) [5] introduces a controllable T2V diffusion model utilizing control signals and a first-frame conditioning strategy. Notably, while closely aligned with our framework, Motion Director [38] lacks available code at the time of our research.

Qualitative Results. We offer visual comparisons of our method against four baselines in Fig. 5. The compared baselines face challenges in adapting the motion of the input video to new contexts. They exhibit difficulties in applying the overall motion, neglecting the specific background indicated in the target text (e.g., “underwater” or “on the sand”). Additionally, they face difficulties in deviating from the original shape of the subject in the input video, leading to issues like a shark-shaped airplane, an owl-shaped seagull, or preservation of the shape of the ground where a seagull is taking off. In contrast, the proposed framework succeeds in motion-driven customization, even for difficult compositional customization, e.g. Two sharks are moving. Ñ Two airplanes are moving in the sky.

Quantitative Results. We further quantitatively demonstrate the effectiveness of our method against the baselines

[Figure 5]

##### Figure 5. Qualitative comparison against state-of-the-art baselines. In contrast to other baselines, the proposed framework succeeds in motion-driven customization, even for difficult compositional customization.

[Figure 6]

- Figure 6. Comparative analysis of the proposed frameworks with fine-tuning (a) temporal attention and (b) self- and cross-attention layers.

[Figure 7]

Figure 7. Comparative analysis of the proposed frameworks with (a) ℓcos and (b) ℓ2 loss functions.

through automatic metrics and user study.

Automatic Metrics. We use CLIP [22] encoders for automatic metrics. For textual alignment, we compute the average cosine similarity between the target prompt and the generated frames. In terms of frame consistency, we obtain CLIP image features within the output video and then calculate the average cosine similarity among all pairs of video frames. For methods that generate temporally interpolated frames, we utilized the keyframe indexes to calculate the metric for a fair evaluation. To illustrate, in the case of VMC, which takes an 8-frame input and produces a 29-frame output, we considered the frames at the following indexes: 1, 5, 9, 13, 17, 21, 25, 29. As shown in Table 1, VMC outperforms baselines in both text alignment and temporal consistency.

User Study. We conducted a survey involving a total of 27 participants to assess four key aspects: the preservation of motion between the input video and the generated output video, appearance diversity in the output video compared to the input video, the text alignment with the target prompt, and the overall consistency of the generated frames. The survey utilized a rating scale ranging from 1 to 5. For assessing motion preservation, we employed the question: “To what extent is the motion of the input video retained in

the output video?” To evaluate appearance diversity, participants were asked: “To what extent does the appearance of the output video avoid being restricted on the input video’s appearance?” Tab. 1 shows that our method surpasses the baselines in all four aspects.

| |Text Temporal Alignment Consistency<br><br>|Motion Appearance Text Temporal Preservation Diversity Alignment Consistency|
|---|---|---|
|VC Gen-1 TAV CAV Ours|0.798 0.958 0.780 0.957 0.758 0.947 0.764 0.952 0.801 0.959<br><br>|3.45 3.43 2.96 3.03<br>3.46 3.17 2.87 2.73<br><br><br>3.50 2.88 2.67 2.80 2.75 2.45 2.07 2.00<br>4.42 4.54 4.56 4.57<br>|

Table 1. Quantitative evaluation using CLIP and user study. Our method significantly outperforms the other baselines.

#### 4.3. Ablation Studies

Comparisons on attention layers. We conducted a comparative study evaluating the performance of fine-tuning: (a) temporal attention layers and (b) self- and crossattention layers. Illustrated in Fig. 6, both frameworks exhibit proficient motion learning capabilities. Notably, the utilization of customized temporal attention layers (a) yields smoother frame transitions, indicating the effectiveness of the optimization framework (17) in encouraging mo-

[Figure 8]

Figure 8. Left: Style transfer on two videos. Right: Motion customization results on the video of “A seagull is walking backward.”

tion distillation, with a slight preference observed for customized temporal attention layers.

This observation stems from the premise that integrating the proposed motion distillation objective (17) may autonomously and accurately embed motion information within temporal attention layers [12, 14]. This suggests a potential application of the motion distillation objective for training large-scale video diffusion models, warranting further exploration in future research endeavors.

Choice of loss functions. In addition, we conducted a comparative analysis on distinct training loss functions in (17): the ℓ2-distance and ℓcos as delineated in (17). As depicted in Fig. 7, the δϵ-matching process in (15) and (17) demonstrates compatibility with generic loss functions. While both ℓ2pδϵnt ,δϵnθ,tq and ℓcospδϵnt ,δϵnθ,tq are promising objectives, the marginal superiority of ℓcospδϵnt ,δϵnθ,tq led to its adoption for visualizations in this study.

Importance of adaptation. To assess the importance of temporal attention adaptation, we conducted a visualization of customized generations without temporal attention adaptation, as detailed in Section 3.1. Specifically, from our original architecture in Fig. 2, we omitted attention adaptation and performed inference by maintaining the U-Net modules in a frozen state. The outcomes depicted in Fig. 9 indicate that while DDIM inversion guides the generations to mimic the motion of the input video, it alone does not ensure successful motion distillation. The observed changes in appearance and motion exhibit an entangled relationship. Consequently, this underlines the necessity of an explicit motion distillation objective to achieve consistent motion transfer, independent of any alterations in appearance.

#### 4.4. Additional results

Video Style Transfer. We illustrate video style transfer applications in Fig. 8-Left. We incorporate style prompts at the end of the text after applying appearance-invariant prompt-

[Figure 9]

Figure 9. Ablation study on temporal attention adaptation. Without temporal attention adaptation, motion distillation fails.

ing (see Section 3.2). Target styles are fluidly injected while preserving the distilled motion of an input video.

Learning Backward Motion. To further verify our video motion customization capabilities, we present a challenging scenario: extracting backward motion from a reversed video sequence where frames are arranged in reverse order. This scenario, an exceedingly rare event in real-world videos, is highly improbable within standard training video datasets [1]. Illustrated in Fig. 8, our VMC framework showcases proficiency in learning “a bird walking backward” motion and generating diverse videos with distinct subjects and backgrounds. This capability not only enables leveraging the distilled motion but also offers prospects for further contextual editing.

- 5. Conclusion This paper introduces Video Motion Customization

(VMC), addressing challenges in adapting Text-to-Video (T2V) models to generate motion-driven diverse visual customizations. Existing models struggle with accurately replicating motion from a target video and creating varied visual outputs, leading to entanglements of appearance and motion data. To overcome this, our VMC framework presents a novel one-shot tuning approach, focusing on adapting temporal attention layers within video diffusion models. This framework stands out for its efficiency in time and memory, ease of implementation, and minimal hyperparameters. We demonstrated the efficacy of our customization methods across diverse motion types, appearances, and contexts.

Ethics Statement. Our work is based on a generative model with potential for misuse, including the creation of deceptive content, which may have negative societal impacts. Additionally, inappropriate content and biases could be included in the datasets used for the foundational training.

### References

- [1] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738,

2021. 5, 8

- [2] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2
- [3] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 2
- [4] Hong Chen, Xin Wang, Guanning Zeng, Yipeng Zhang, Yuwei Zhou, Feilin Han, and Wenwu Zhu. Videodreamer: Customized multi-subject text-to-video generation with disen-mix finetuning. arXiv preprint arXiv:2311.00990,

2023. 11

- [5] Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models. arXiv preprint arXiv:2305.13840, 2023. 5
- [6] Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106(496):1602– 1614, 2011. 3
- [7] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023. 5
- [8] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-

- Or. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2, 11
- [9] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, MingYu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22930–22941, 2023. 2
- [10] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. arXiv preprint arXiv:2305.18292, 2023. 11
- [11] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. arXiv preprint arXiv:2303.11305, 2023. 11
- [12] J Ho, T Salimans, A Gritsenko, W Chan, M Norouzi, and DJ Fleet. Video diffusion models. arxiv 2022. arXiv preprint arXiv:2204.03458. 3, 8
- [13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [14] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2, 3, 8
- [15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 11
- [16] Aapo Hyv¨arinen and Peter Dayan. Estimation of nonnormalized statistical models by score matching. Journal of Machine Learning Research, 6(4), 2005. 2
- [17] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 11
- [18] Haoming Lu, Hazarapet Tunanyan, Kai Wang, Shant Navasardyan, Zhangyang Wang, and Humphrey Shi. Specialist diffusion: Plug-and-play sample-efficient fine-tuning of text-to-image diffusion models to learn any unseen style. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14267–14276, 2023. 11
- [19] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning. arXiv preprint arXiv:2307.11410, 2023. 2
- [20] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al. Mixed precision training. arXiv preprint arXiv:1710.03740, 2017. 5
- [21] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 5

- [22] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 7
- [23] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 2, 5, 11
- [24] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. arXiv preprint arXiv:2307.06949, 2023. 2, 5, 11
- [25] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 2
- [26] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. arXiv preprint arXiv:2304.03411, 2023. 2, 11
- [27] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 2

- [28] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 2
- [29] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2, 11
- [30] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. 2
- [31] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2
- [32] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018, 2023. 5
- [33] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 2

- [34] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848, 2023. 11
- [35] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 2, 4, 5, 11
- [36] Ruiqi Wu, Liangyu Chen, Tong Yang, Chunle Guo, Chongyi Li, and Xiangyu Zhang. Lamp: Learn a motion pattern for few-shot-based video generation. arXiv preprint arXiv:2310.10769, 2023. 5, 11
- [37] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023. 2, 3, 5, 11
- [38] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. Motiondirector: Motion customization of text-tovideo diffusion models. arXiv preprint arXiv:2310.08465,

2023. 2, 5, 11

- [39] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 2

### A. Appendix

The supplementary sections in Appendix are organized as follows. Section B introduces the pseudo training algorithm behind our Video Motion Customization (VMC) framework. In Section C, we provide a discussion on related works in the field of generative model customization. Following this, we delve into the details on our training and inference configurations in Section D. Concluding the document, Section E features a showcase of additional results obtained from our VMC framework.

### B. Pseudo Training Algorithm

Algorithm 1 Temporal Attention Adaption

- 1: Input: N-frame input video sequence pvn0qnPt1,...,Nu, appearance-invariant training prompt Pinv, textual encoder ψ, Training iterations M, key-frame generator parameterized by θ and its temporal attention parameters θTA.
- 2: Output: Fine-tuned temporal attention layers θTA˚ .
- 3:
- 4: for step “ 1 to M do
- 5: Sample timestep t P r0,Ts and Gaussian noise ϵ1:t N, where ϵnt P Rd „ Np0,Iq
- 6: Prepare text embeddings cinv “ ψpPinvq
- 7: vnt “

?α¯tvn0 `

?1 ´ α¯tϵnt , @n.

- 8: δϵnθ,t “ ϵnθ`1pv1:t N,t,cinvq ´ ϵnθpv1:t N,t,cinvq, @n ď N ´ 1.
- 9: δϵnt “ ϵnt `1 ´ ϵnt , @n ď N ´ 1
- 10: Update θTA with N1´1

ř

n ℓcospδϵnt ,δϵnθ,tq

- 11: end for

We express Gaussian noises as ϵ1:t N to avoid confusion. In our observation, aligning the image-space residuals (δvn0 and δvˆn0ptq) corresponds to aligning the latent-space epsilon residuals (δϵnt and δϵnθ,t) across varying time steps t P r0,Ts. This relationship stems from expressing the motion vector δvn0 and its estimation δvˆn0ptq in terms of δvnt , δϵnt , and δϵnθ,t. Consequently, the proposed optimization framework fine-tunes temporal attention layers by leveraging diverse diffusion latent spaces at time t which potentially contains multi-scale rich descriptions of video frames. Therefore, this optimization approach seamlessly applies to video diffusion models trained using epsilon-matching, thanks to the equivalence between δϵnt -matching and δvn0matching.

### C. Related Works

Image Customization. Prior methodologies in text-toimage customization, termed personalization [8, 10, 11, 18, 23, 24, 26, 34], aimed at capturing specific subject ap-

pearances while maintaining the model’s ability to generate varied contents. However, this pursuit of personalization poses challenges in time and memory demands [23]. Fine-tuning each personalized model requires substantial time costs while storing multiple personalized models may strain storage capacity. To address these hurdles, some approaches prioritize efficient parameter customization, leveraging techniques like LoRA [10, 15] or HyperNetwork [24] rather than training the entire model.

Video Customization. Building on the success of textto-image customization, recent efforts have adopted textto-image or text-to-video diffusion models for customizing videos in terms of appearance or motion. These endeavors, such as frameworks proposed by [4, 35], focus on creating videos faithful to given subjects or motions. Moreover, works by [38] or [36] delve into motion-centric video customization, employing various fine-tuning approaches ranging from temporal-spatial motion learning layers to newly introduced LoRAs. In this paper, the proposed VMC framework emphasizes efficient motion customization with explicit motion distillation objectives, specifically targeting temporal attention layers. This approach, facilitated by cascaded video diffusion models, efficiently distills motion from a single video clip while minimizing computational burdens in terms of both time and memory.

### D. Training & Inference Details

For our work, we utilize the cascaded video diffusion models from Show-1 [37], employing its publicly accessible pre-trained weights 2. Our approach maintains the temporal interpolation and spatial super-resolution modules in their original state while focusing our temporal optimization solely on the keyframe generator. In specific, we finetune Query, Key, Value projection matrices WQ,WK,WV of temporal attention layers of the keyframe UNet. We use AdamW [17] optimizer, with weight decay of 0.01 and learning rate 0.0001. By default, we employ 400 training steps. During the inference phase, we perform DDIM inversion [29] for 75 steps. For the temporal interpolation and spatial super resolution stages, we follow the default settings of Show-1.

### E. Additional Results

This section is dedicated to presenting further results in motion customization. We display keyframes (7 out of the total 8 frames) from input videos in Figures 10, 11, 12, and 13, accompanied by various visual variations that maintain the essential motion patterns. Specifically, Figure 10 showcases input videos featuring car movements. In Figure 11, we exhibit input videos capturing the dynamics of airplanes in flight and the blooming of a flower. Figure 12

2https://huggingface.co/showlab

focuses on bird movements, including walking, taking off, floating, and flying. Lastly, Figure 13-top highlights input videos of mammals, while 13-bottom illustrates the motion of pills falling. Moreover, for a comprehensive comparison between the motion in the input and generated videos, complete frames from these videos are presented in Figures 14, 15, 16, 17, and 18. In each of these figures, the left columns show the 8-frame input video, while the adjacent three columns on the right exhibit 29 frames from the generated videos, replicating the same motion pattern.

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

##### Figure 14. Full-frame results of Video Motion Customization: Text prompt “Sharks are moving” is used for training the keyframe generation UNet.

[Figure 15]

##### Figure 15. Full-frame results of Video Motion Customization: Text prompt “A seagull is walking” is used for training the keyframe

[Figure 16]

##### Figure 16. Full-frame results of Video Motion Customization: Text prompt “Ink is spreading” is used for training the keyframe generation

[Figure 17]

##### Figure 17. Full-frame results of Video Motion Customization: Text prompt “A man is snowboarding” is used for training the keyframe

[Figure 18]

##### Figure 18. Full-frame results of Video Motion Customization: Text prompt “A tiger is walking” is used for training the keyframe generation

