# arXiv:2410.23287v2[cs.CV]6Aug2025

## ReferEverything: Towards Segmenting Everything We Can Speak of in Videos

Anurag Bagchi1 Zhipeng Bao1 Yu-Xiong Wang2 Pavel Tokmakov3† Martial Hebert1†

1Carnegie Mellon University 2University of Illinois Urbana-Champaign 3Toyota Research Institute https://refereverything.github.io/

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

“the raindrop running down the center of the windowpane” “the smoke blown out” “the building in the middle falling down”

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

“the spinning column of water” “the cracks in the glass” “the light column”

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

“the green bottle of paint shattering from the impact of the ball”

“the wave crashing in the ocean” “the bubble being popped”

Figure 1. Results of our method, REM, which leverages visual-language representations learned by video diffusion models to segment a wide range of concepts based on natural language descriptions (shown in grey boxes). REM exhibits zero-shot generalization to challenging, dynamic concepts, such as raindrops or shattering glass. Video visualizations are available on the project page.

#### Abstract

We present REM, a framework for segmenting a wide range of concepts in video that can be described through natural language. Our method leverages the universal visuallanguage mapping learned by video diffusion models on Internet-scale data by fine-tuning them on small-scale Referring Object Segmentation datasets. Our key insight is to preserve the entirety of the generative model’s architecture by shifting its objective from predicting noise to predicting mask latents. The resulting model can accurately segment rare and unseen objects, despite only being trained on a limited set of categories. Additionally, it can effortlessly generalize to non-object dynamic concepts, such as smoke or raindrops, as demonstrated in our new benchmark for Referring Video Process Segmentation (Ref-VPS). REM performs on par with the state-of-the-art on in-domain datasets, like Ref-DAVIS, while outperforming them by up to

†Equal advising.

12 IoU points out-of-domain, leveraging the power of generative pre-training. We also show that advancements in video generation directly improve segmentation.

#### 1. Introduction

One of the most remarkable properties of natural language lies in its ability to convey the richness and complexity of human visual experience. From fleeting moments, like raindrops rolling down the window or smoke dissipating from a cigarette (see row 1 in Figure 1), to dynamic processes, such as a glass shattering or a whirlpool forming in the water (row 2 in Figure 1), language enables us to descrive and reference events with precision. Crucially, if an event can be expressed verbally, it can often be accurately localized in both space and time. This universal mapping between the discrete, symbolic realm of language and the continuous, ever-changing visual world is developed through a lifetime of visual-linguistic interaction [4, 42].

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

“Explode colorful smoke coming out” “Time-lapse of a blooming flower on a stem ”

- Figure 2. Through Internet-scale pre-training, video diffusion models learn to generate videos capturing the diversity of the dynamic visual world (samples shown above). We leverage their powerful visual-language representation for open-world referring video segmentation.

The corresponding task in computer vision - Referring Video Segmentation (RVS) [18, 22], is defined as the task of segmenting a specific region in a video based on a natural language expression. However, virtually all existing literature focuses on a narrow subset of RVS - Referring Video Object Segmentation (RVOS) [50, 58]. This emphasis stems from data constraints: RVOS datasets were built by annotating object tracking benchmarks [41, 64], which are inherently object-centric and limited in scale. Recent advances in large visual-language datasets [2, 49] and generative models [46, 55, 56] present opportunities to overcome these limitations. Visual-language representations learned by these models from Internet data demonstrated strong generalization in image-based segmentation [37, 73], but their potential in videos remains underexplored [76].

In this work, we introduce REM – a novel approach to RVS that enables spatio-temporal localization of a wide range of concepts [19] in video that can be described through natural language, beyond conventional object tracking (shown in Figure 1). A key factor behind REM’s success is preserving the universal visual-language mapping learned by generative models during Web-scale pre-training (see Figure 2). To this end, we retain the original model architecture and fine-tune it on small-scale RVOS datasets, adjusting the output to generate target mask latents instead of Gaussian noise. As shown in Table 1, this model, originally designed and trained for video generation, demonstrates competitive performance against specialized RVOS methods on popular benchmarks [41, 64]. More significantly, it exhibits much stronger generalization.

To quantify this effect, we report zero-shot evaluation results on the open-world object tracking benchmark BURST [1], and the non-object ‘Stuff’ segmentation dataset - VSPW [34], as well as on a newly collected benchmark that expands the focus of RVS to include dynamic process in Section 4. We define the latter as temporally evolving events, where subjects undergo continuous changes in state, shape, or appearance (see examples in Figure 1). Our new benchmark, which we call Ref-VPS for Referring Video Process Segmentation, consists of 145 videos that are labeled with referring expressions and masks at 6 fps and span 39 unique concepts. Experiments in Section 5.2 demonstrate that existing approaches, including the very recent method of Zhu et al. [76], fail to generalize outside of the

narrow training distribution, whereas our method effortlessly segments a wide spectrum of targets (see Figures 1 and 5).

The primary contribution of this paper is to demonstrate that Web-scale video diffusion models have learned universal visual-language mapping that can be repurposed for open-world referring video segmentation. We further introduce a new benchmark for Referring Video Process Segmentation in Section 4, expanding the focus of RVS beyond conventional object tracking. Finally, we provide a detailed analysis of our approach in Section 5.3, demonstrating that retaining the full architecture of the generative model, rather than isolating the de-noising network as a feature extractor, is key to unlocking the strongest generalization in RVS.

#### 2. Related Work

Referring Video Segmentation (RVS) involves segmenting specific regions in a video based on a natural language description [18, 27, 50]. Most benchmarks for this task were developed by adding referring expression annotations to existing Video Object Segmentation (VOS) datasets, such as DAVIS’17 [41] or YouTube-VOS [64]. Consequently, the role of language in these benchmarks is limited to providing an interface for user-initialized object tracking [39, 60]. While segmenting objects is valuable, it addresses only a narrow subset of the possible interactions between language and the space-time continuum of videos. Equally important is the ability of RVS methods to segment video concepts beyond common object categories. To address this gap, we introduce a new benchmark focused on segmenting dynamic processes, which we term Referring Video Process Segmentation (Ref-VPS).

Earlier RVOS approaches [5, 23, 35] generally employed a bottom-up strategy: first, image-level methods [9, 40, 48, 68] were applied to obtain frame masks, followed by spatiotemporal reasoning, such as mask propagation [50], to refine the segmentation across frames. More recently, with the success of cross-attention-based methods [33, 54, 71] in object segmentation and tracking, query-based architectures have been introduced to RVOS, leading to significant improvements [57, 58, 67]. The limited scale of paired video-language data with segmentation annotations has always been a major limitation, causing most methods to train

jointly on videos and images [24, 25]. The latest approaches go further and unify all object localization tasks in a single framework [12, 59, 65]. However, while these models excel in object tracking, they struggle to generalize to more dynamic concepts. In contrast, we demonstrate that generative pre-training on Internet-scale data [2, 49] results in a universal (i.e., not limited to one domain) mapping between the space of language and the ever-changing visual world.

Diffusion models have emerged as the de facto standard for generative learning in computer vision [21, 51] and beyond [13]. Among them, the Denoising Diffusion Probabilistic Model (DDPM) [21] leverages neural network components to model the denoising process. Stable Diffusion (SD) [46] shifts the denoising process into the latent space of a pre-trained autoencoder [28], allowing for model scaling. Expanding from images to videos, diffusion models have seen success in text-to-video (T2V) generation [7, 10, 11, 56, 74]. In addition to the capacity to generate high-fidelity images based on text prompts, the T2V diffusion models implicitly learn the mapping from linguistic descriptions to video regions, providing an opportunity to repurpose them for RVOS. Among current T2V methods, ModelScope [56] and Wan [55] stand out for their opensource implementations and top performance.

Visual-language pre-training for perception: in addition to being highly effective in image and video generation, diffusion models have been shown to learn a strong representation of the natural image manifold. Several works demonstrated that they can be re-purposed for computer vision problems, including semantic segmentation [62, 72, 73] and pixel-level correspondence [52], achieving an impressive degree of generalization. Others have shown that image diffusion models learn powerful object representations, enabling open-world novel view synthesis [31] and amodal segmentation [37]. Most recently, Zhu et al. [76] also leverages pre-trained T2V models for RVOS, however, our analysis shows that their approach fails to fully capitalize on the universal visual-language mapping learned in generative pre-training. In this work, we explore the application of video diffusion models to RVS, demonstrating how to maintain a high-level generalizability during fine-tuning.

In a separate line of work, visual-language representations learned with contrastive objectives [3, 43] have been adapted for referring image [29, 44, 63, 69] and video segmentation [75]. However, their performance remains limited compared to both generative models and classical referring segmentation approaches.

#### 3. Method

##### 3.1. Visual-language mapping via video denoising

Text-to-Video (T2V) diffusion models [11, 56, 74] generate videos that align with a given language description, starting

from Gaussian noise. The process can be formalized as:

###### xˆ = fvdm(xT,c,T), (1)

where xˆ is the generated video, T denotes the maximum timestep specified by the video diffusion model fvdm, xT is a sample drawn from a Gaussian distribution N(µT,σT2 ) predefined by the video diffusion model, and c is the conditioning prompt. To reduce computational complexity, these models often perform denoising in the latent space [46]. Specifically, a pre-trained Variational Autoencoder (VAE) [28] is employed to map the video x from pixel space into latent space, denoted as E(x) = z, while a decoder reconstructs it from the latent space, D(z) ≈ x. Thus, the generation process becomes:

xˆ = D (fvdm(zT,c,T)). (2)

During training, rather than denoising from pure Gaussian latents, T2V models denoise from partially noisy video latents and optimize the following latent diffusion objective:

Ez∼E(x),t,ϵ∼N(0,1) ∥ϵ − ϵθ(zt,ec,t)∥22 , (3)

min

θ

where ϵ is the Gaussian noise added to the clean video latent, zt represents the noisy video latent at timestep t derived by the diffusion forward pass [21, 46], and ec is the conditional embedding generated from c using a text encoder [43]. The denoising network ϵθ(zt,ec,t), typically a U-Net [47] or a Diffusion Transformer (DiT) [38], is tasked with predicting the noise ϵ. In this network, the conditional embedding ec interacts with the latent representations through attention mechanisms, guiding the model to generate diverse, semantically accurate videos.

##### 3.2. From language-conditioned denoising to RVS

Referring Video Segmentation (RVS) involves segmenting an entity in a video across spatial and temporal dimensions, guided by a natural language description. Formally, the task is defined as:

mˆ = fRVS(x,c), (4)

where fRVS is the RVS model, x represents a video sequence, c denotes a referring text prompt, and mˆ is the binary masks produced as output. This task aligns naturally with T2V models, which establish a robust mapping between the entities described in the text and the corresponding spatial-temporal regions in the video by optimizing the denoising objective in Equation 3.

Several prior works have explored the alignment of diffusion models with referring segmentation [62, 73, 76], typically employing these models as feature extractors. Specifically, they adjust the input format of a referring segmentation model to match that of the denoising network ϵθ, and

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

…

…

Input video GT Masks

Repeat 3x

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

VAE Encoder

VAE Encoder

###### ~ L2 ~

Latent Diffusion Model

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

VAE Decoder

[Figure 60]

### ~

[Figure 61]

Text-Encoder

[Figure 62]

[Figure 63]

Mean ~3

Predicted Masks

[Figure 64]

[Figure 65]

[Figure 66]

"a brown and white cow walking towards the camera"

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

…

Gradient flow Inference

- Figure 3. The model architecture of Refer Everything with Diffusion Models (REM). Like a video diffusion model it is based on, our approach takes video frames with added noise and a language expression as input. Our key insight is preserving the entirety of the generative model’s architecture by shifting its objective from predicting noise to predicting mask latents.

model. This subtle yet powerful modification allows the model to better preserve its universal visual-language mapping learned on Internet-scale data during generative pretraining while adapting to the task of RVS.

pass the resulting features to a task-specific decoder fdec (e.g., a convolutional network) to predict the target masks:

mˆ = fdec(ϵ(θn)(zt,ec,t)), (5)

Training and optimization. During training, to encode the ground-truth segmentation masks with the VAE, we broadcast the single-channel mask into three channels by simply duplicating it (shown in the top right of Figure 3). For simplicity, we still denote this three-channel mask representation as m. The pre-trained VAE can then map the mask sequence into the latent space via E(m) = zm and decode the masks back from predicted latents via D(zm) ≈ m. For the noisy latent zt and timestep t, we prioritize using latents that remain as clean as possible. Therefore, we always set the timestep to its minimum value, t = 0. To train the model, we supervise the predicted mask latents using an L2 loss (shown in the center-right of Figure 3) by minimizing:

where zt is the noisy latent representation of the input images at timestep t, ec is a feature embedding of the referring expression c, and ϵ(θn) denotes the intermediate feature at the nth layer. In practice, t is usually set to a small value (e.g., 50), and n is set to the later layer indexes to obtain the optimal performance. The entire model is then trained in a conventional discriminative learning setup. However, replacing parts of the generative model with newly initialized layers can disrupt alignment between the model’s representation from pre-training and the new features learned on narrow-domain datasets, leading to a substantial loss of generalization capabilities.

In our approach, shown in Figure 3, we propose to instead preserve the architecture of the video diffusion model in its entirety. Specifically, rather than using intermediate

Ezm∼E(m),t=0 ∥zm − ϵθ(zt,ec,t)∥22 . (7)

min

θ

features ϵ(θn), REM repurposes the whole denoising network ϵθ (together with the VAE) by shifting its objective from predicting noise to predicting mask latents (shown on the right in Figure 3):

Model inference. During inference, we follow Equation 6, with t = 0, to decode the predicted mask latent and generate three-channel mask predictions. We then compute the single-channel masks by averaging the pixel values of the three channels and applying a constant threshold of 0.5 to binarize the result (shown in the bottom right of Figure 3). Notably, the inference is non-iterative (the mask is predicted in a single forward pass), making the computational cost of REM on par with other approaches in the literature.

mˆ = D(ϵθ(zt,ec,t)), (6)

where D denotes the (frozen) VAE decoder used to produce the actual binary segmentation masks from the predicted latents. That is, instead of learning the decoder network fdec from scratch, we reuse the VAE from the video diffusion

#### 4. Benchmark Design and Collection

SAM2 mask generation with click prompts

- Stage I
- Stage II

[Figure 73]

Input Video

[Figure 74]

Refine prompts

Existing datasets only allow for quantifying the generalization of RVS models to rare object categories [1] or static ‘Stuff’ [34]. However, covering the entire spectrum of concepts that can be spoken of in videos in a single benchmark would be extremely costly. Instead, we narrow our focus to the most salient subset that necessitates joint modeling of language and temporal dynamics. Specifically, we target dynamic processes, defined as temporally evolving events, where the subjects undergo continuous changes in state, shape, or appearance. Importantly, these subjects are not limited to objects but include any spatio-temporally localizable phenomena, such as light or fire. Our new benchmark, Referring Video Process Segmentation (Ref-VPS), is built by selecting representative videos and annotating them with referring expressions and segmentation masks.

[Figure 75]

[Figure 76]

###### …

SAM2 masks

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

… …

[Figure 81]

Initial prompt

Manual mask refinement + Ignore region labelling

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

###### …

[Figure 86]

###### “the dandelion being blown away by a woman”

##### 4.1. Video selection

SAM2 Video mask

To source the videos, we require a large, public, and diverse database that supports natural language queries and permits content redistribution for research. Based on these criteria, we selected TikTok — a platform with tens of millions of daily uploads that capture a wide range of dynamic scenarios. Moreover, TikTok’s policies generally allow free redistribution of content, with individual users retaining the option to opt out.

Figure 4. Our mask annotation pipeline. We first use SAM2 to interactively segment the region of interest in the video (shown above). We then manually refine the masks where SAM2 fails and label ambiguous regions as Ignore (shown in yellow below).

##### 4.2. Annotation collection and evaluation

To label the selected videos, we first adjust each clip’s temporal boundaries to focus on the event of interest and avoid shot changes, ensuring that the event, along with some contextual frames before and after, is captured in its entirety. The clips are then exported as frames at 24 FPS.

To identify a representative pool of videos, we established a taxonomy of five broad, possibly overlapping concepts (e.g., ‘object transformations’, or ‘pattern evolution’; see Section A in the supplementary for definitions). Although these do not cover every possible dynamic process, we selected them to create a clear, focused framework for sourcing representative samples. For each concept, we use ChatGPT [36] to generate concrete examples and search queries for TikTok (e.g., ‘a wax candle melting’ for ‘object transformations’), resulting in 120 fine-grained concepts. The queries retrieved over 1,000 samples, but many were unsuitable - either because the events (e.g., ‘glaciers melting over time’) occur over extended periods and are rarely captured on TikTok, or because of ambiguous search terms. After this step, we obtained a set of 342 samples.

To collect referring expressions, we first manually identify the target entity in each clip and then instruct two independent annotators to provide descriptions. Each annotator contributes two expressions per target, yielding a total of four distinct expressions per clip. Following standard protocol [27, 50], evaluation is conducted on all queries, and results are reported as the average performance across them.

Next, we label the identified targets with segmentation masks at 6 FPS using a semi-automatic pipeline. To this end, we leverage the recently introduced SAM2 [45] foundational model for interactive video segmentation. In particular, as shown in the top part of Figure 4, we first provide positive and negative clicks to a frame of the video. SAM2 then automatically segments the entity of interest in the frame, as well as propagates the mask across the entire clip. We interactively improve segmentation quality by providing additional clicks as needed.

We then manually filtered the videos using these criteria: (1) exclude videos lacking significant dynamic changes (e.g., mostly stationary clouds); (2) exclude events that occur too rapidly to label a sufficient number of non-empty frames (e.g., lightning flashes); and (3) exclude videos with frequent shot changes that prevent extraction of a continuous clip capturing the event. Additionally, for videos that are compilations of similar events, we split them into individual clips and treat each independently. The final dataset comprises 145 clips covering 39 concepts. As it is intended for zero-shot evaluation, no additional splits are defined. Representative samples with results are shown in Figure 1.

As SAM2 cannot always accurately segment the challenging and often ambiguous entities featured in Ref-VPS even with a large number of clicks, we manually refine the masks in frames where it fails (Figure 4, bottom). We additionally label ambiguous regions, such as the stem of the dandelion, as Ignore (shown in yellow). Visualizations

|Method|Pre-training Data Mask/Box Supervision<br><br>|Ref-DAVIS|Ref-YTB|
|---|---|---|---|
| | |J &F J F|J &F J F|
|Referformer [58] MUTR [67] VLMO-L [75] UNINEXT [65] VD-IT [76]|ImageNet + Kinetics + SSv2 Ref-COCO/+/g + Ref-YTB<br><br>ImageNet + Kinetics + SSv2 Ref-YTB + AVS<br><br>Unknown Ref-COCO/+/g + Ref-YTB<br><br>Object365 10+ Image/Video datasets<br><br>LAION5B + WebVid Ref-COCO/+/g + Ref-YTB|61.1 58.1 64.1<br><br>68.0 64.8 71.3<br><br>70.2 66.3 74.1 72.5 68.2 76.8<br><br>69.4 66.2 72.6<br><br><br>|62.9 61.3 64.6 68.4 66.4 70.4 67.6 65.3 69.8 70.1 67.6 72.7 66.5 64.4 68.5|
|REM (MS-1.4B) REM (Wan-14B)<br><br>|LAION5B + WebVid Ref-COCO/+/g + Ref-YTB Internal + Public Images/Videos Ref-COCO/+/g + Ref-YTB|72.6 69.9 75.3 75.0 71.3 78.7<br><br>|68.4 67.1 69.7 71.7 69.2 74.3|

- Table 1. Comparison to the state of the art on the validation set of the Ref-DAVIS and the test set of Ref-YTB benchmarks using the standard metrics. Even the base version of our method performs on par with the strong UNINEXT approach, despite not being specifically designed for object localization and having access to only a fraction of the localization labels used by that method.

of Ref-VPS annotations, together with additional statistics, are included in the supplementary. For evaluation, we follow Tokmakov et al. [53] and only report region similarity J [16] as contour accuracy F [39] is often not well defined for dynamic entities like smoke or light. Pixels inside the Ignore regions are not included in the metric calculation.

#### 5. Experiments

Datasets and evaluation. We evaluate our method on six benchmarks in total. Ref-YTB [50], Ref-DAVIS [27], and MeViS [15] are standard RVOS benchmarks, with MeViS focusing on challenging, motion-guided referring expressions. For evaluating generalization to rare objects and ‘Stuff’ categories, we use BURST [1] and VSPW [34] datasets, respectively. Finally, we evaluate REM and the strongest baselines on our newly introduced Ref-VPS (detailed in Section 4). All the datasets, except Ref-YTB and MeViS, are only used for evaluation in a zero-shot manner.

For Ref-YTB [50], Ref-DAVIS [27], and MeViS [15], we use the standard evaluation metrics - Region Similarity (J ) [16], Contour accuracy (F) [39] and their mean (J &F) [41]. For all other evaluations, we use the J metric. The evaluations on Ref-YTB and MeViS are done on the official servers, and we use the official metric implementation of Ref-DAVIS for all the other benchmarks.

Implementation details. Our approach builds upon two state-of-the-art text-to-video diffusion architectures: ModelScope [56] and Wan [55]. Additional video diffusion backbones are evaluated in the supplementary. ModelScope comprises 1.4 billion parameters and extends Stable Diffusion [7] with temporal modules. We adopt a two-stage training protocol following Zhu et al. [76]: in Stage I, we fine-tune only the spatial weights on Ref-COCO image-text pairs[70] for one epoch; in Stage II, we fine-tune all network weights for 40 epochs using Ref-YTB video–text examples [50] supplemented with 12K Ref-COCO images converted into pseudo-videos following Wu et al. [58]. In contrast, Wan employs a 14 billion parameter diffusion transformer that jointly models spatial and temporal information, without dedicated temporal modules [38]. Accordingly, we

train this variant in a single stage on the combined RefCOCO and Ref-YTB datasets for 80k iterations. Throughout training, the text encoder and VAE remain frozen. Unless otherwise stated, all models are trained and evaluated at a resolution of 512 × 512.

More details about the datasets, evaluation setup on BURST and VSPW, implementation, as well as ablations of our training strategy and runtime analysis are included in the supplementary.

##### 5.1. Referring video object segmentation results

In this section, we compare REM to the state of the art on the standard RVOS benchmarks. We report results on the validation set of Ref-DAVIS [27] and the test set of RefYTB [50] in Table 1. The basic variant of our method, which is based on the ModelScope video diffusion model (denoted as ‘REM (MS-1.4B)’ in the table), outperforms the state of the art in terms of J on Ref-DAVIS and is only second to UNINEXT [65] on Ref-YTB. Note that this approach is specifically designed for object segmentation and trained on more than 10 datasets with localization annotations. In contrast, REM adopts an architecture of a video generation model and is only fine-tuned on one image- and one video-segmentation dataset. Despite this, our method is competitive with UNINEXT on standard RVOS benchmarks, and, as we will show next, outperforms it by up to 21 points out-of-domain in terms of J .

Another notable observation is that this variant of REM outperforms VD-IT [76], which is built on top of the same MS-1.4B backbone, on both datasets. This result demonstrates the effectiveness of our approach to preserving the generative model’s architecture, which will become even more evident in out-of-domain evaluation. Using the recent, large-scale Wan T2V diffusion model [55] (denoted as ‘REM (Wan-14B)’) further improves performance, achieving state-of-the-art results on both benchmarks.

We also benchmark our REM against the top entrants on the MeViS leaderboard [15], a dataset specifically designed for segmenting objects based on referring expressions describing their motion. As shown in Table 2, our base vari-

|Method<br><br>|MeViS|
|---|---|
| |J &F J F|
|Referformer [58] VISA-13B [66] DsHmp [20] GLUS [30]<br><br>|31.0 29.8 32.2 44.5 41.8 47.1 46.4 43.0 49.8 51.3 48.5 54.2|
|REM (MS-1.4B) REM (Wan-14B)<br><br>|53.5 50.8 56.3 60.3 57.2 63.4|

- Table 2. Comparison to the state of the art on the MeViS benchmark. REM outperforms approaches specifically designed for this dataset without any modifications.

ant already outperforms the strongest baseline, GLUS [30], which combines a pre-trained multi-modal LLM with the SAM2 foundational video segmentation model [45], by 2.2 points in J &F. Our Wan-14B variant further improves the performance to 60.3, setting the new state-of-the-art for this benchmark. These results validate our method’s ability to reuse the motion–language priors encoded in video diffusion models by preserving their full generative architecture.

##### 5.2. Out-of-domain generalization

Rare objects and ‘Stuff’. We begin by performing a generalization study on the existing open-world tracking BURST dataset [1] as well as on the ‘Stuff’ categories [8] from VSPW [34] in Table 3. BURST is a large-scale, open-world video object segmentation benchmark with diverse scenes and rare objects, whereas VSPW tests the ability to generalize to non-object categories. We report zero-shot evaluation results on the validation set of VSPW and combined validation and test sets of BURST and compare to the topperforming methods in Table 1.

We observe that on both datasets our method outperforms the baselines by significant margins. The improvements are especially noticeable on BURST, demonstrating that REM successfully preserves the object representation learned in generative pre-training. In contrast, VD-IT loses this generalization capacity. We analyze per-category performance in Section B.8 in supp., showing that REM is especially effective for the most challenging objects. On the ‘Stuff’ categories, all the methods do relatively poorly, reflecting the challenge of generalizing to more amorphous ‘Stuff’. Here, VD-IT maintains a lead over entirely objectcentric UNINEXT, but REM still outperforms both.

Dynamic processes. We compare REM to the topperforming RVS baselines on our Ref-VPS benchmark in Table 4. As before, all the evaluations are zero-shot. Our approach outperforms all baselines by up to 12.1 points in Region Similarity (31.9% relative improvement), and notably surpasses the top RVOS method, UNINEXT, by 21.3 points (74.2% relative improvement). While generative pretraining enhances VD-IT’s generalization ability compared

|Benchmark<br><br>|MUTR UNINEXT VD-IT<br><br>|REM (Ours)|
|---|---|---|
| | |MS-1.4B Wan-14B|
|VSPW BURST<br><br>|10.5 10.1 12.7 27.2 30.2 29.0|15.2 18.5 37.5 40.9<br><br>|

- Table 3. Comparison against the state of the art on the ‘Stuff’ categories in VSPW and on the open-world object-tracking BURST benchmark shows that REM achieves substantially stronger generalization. In particular, it outperforms VD-IT, which shares the same underlying diffusion backbone.

|MUTR UNINEXT VD-IT GLUS<br><br>|REM (Ours)|
|---|---|
| |MS-1.4B Wan-14B<br><br>|
|25.4 28.7 37.9 34.6|49.0 50.0|

- Table 4. Comparison to the state of the art on the new Ref-VPS benchmark. REM shows much stronger zero-shot generalization to challenging, dynamic concepts in this dataset compared to the baselines by effectively capitalizing on Internet-scale pre-training.

to UNINEXT, it struggles to preserve its representations as effectively as REM. The recent GLUS approach, which was designed for MeViS, also fails to generalize to Ref-VPS, highlighting the complementarity of the two benchmarks.

A qualitative comparison of REM (MS-1.4B) with VDIT and UNINEXT is provided in Figure 5. In the first row, our approach is able to track the sponge (which was never seen in training) in a challenging sequence from BURST, whereas other methods focus on foreground objects. In the second sequence from VSPW REM successfully generalizes to the non-object ‘wall’ category, whereas UNINEXT focuses on a nearby object and VD-IT fails entirely. The following examples from Ref-VPS illustrate that both baselines exhibit object-centric bias, as in the examples with the lizard skin in row 3 and blue smoke in row 6. While VD-IT shows better generalization, it often latches on the dominant region (rows 4 and 7 in Figure 5). In contrast, REM demonstrates both good coverage of rare concepts and high precision with respect to the language prompt. See more examples of highly dynamic sequences in the supplementary.

- 5.3. Ablation analysis

We now assess how effectively our REM transfers generative representations to RVS. We report results on the indistribution Ref-YTB dataset and our Ref-VPS benchmark, using the MS-1.4B variant of our model. We further ablate the effect of the generative pre-training strategy in the supplementary. For efficiency, we train our model on a reduced subset of ∼12K image and video samples for these experiments, which accounts for the slightly lower performance compared to the full-data results reported above.

Is supervising in the latent space important? We evaluate the effectiveness of our key design decision to supervise mask prediction in the latent space of the frozen VAE decoder in Table 5. Instead of the latent space, we supervised the masks in the raw pixels space (RGB) by propagating

OURS VD-IT UNINEXT OURS VD-IT UNINEXT

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

“the sponge” “the wall”

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

OURS VD-IT UNINEXT OURS VD-IT UNINEXT

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

“the skin being pulled” “the rainbow appearing in the mist”

OURS VD-IT UNINEXT OURS VD-IT UNINEXT

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

“the wave” “the blue smoke like patterns in the bowl”

OURS VD-IT UNINEXT

“the smoke from the car”

Figure 5. Qualitative results of REM (MS-1.4B) and state-of-the-art baselines on BURST (top row left), VSPW (top row right), and Ref-VPS (bottom three rows) benchmarks. Our method demonstrates both superior coverage of rare, dynamic concepts, such as smoke or waves, and higher segmentation precision (e.g., only capturing the skin on the lizard). Video comparisons are available here.

|Supervision<br><br>|Resolution|Decoder<br><br>|Ref-YTB (J &F)|Ref-VPS J<br><br>|
|---|---|---|---|---|
|Latent RGB RGB<br><br>|256×256 256×256 256×256|VAE (Frozen) VAE (Frozen) VAE (Fine-tuned)<br><br>|61.4 58.4 60.4|35.8 31.6 32.4<br><br>|
|Latent RGB RGB|512×512 512×512 512×512<br><br>|VAE (Frozen) CNN MLP|63.5 59.6 59.3<br><br>|40.0 29.4 33.1|

the gradients through the VAE decoder (rows two and three in the table). This increases memory requirements during training, forcing a lower 256×256 input resolution to fit on an A100 GPU. The results show that latent-space supervision is crucial for maintaining generalization, as evidenced by the performance drop of the RGB-based variants on RefVPS, even when the VAE decoder is additionally finetuned.

Table 5. Analysis of the effect of the fine-tuning strategy on RefYTB and Ref-VPS. The key to the success of REM is in preserving the entirety of the generative model’s architecture.

Is it better to train a mask decoder from scratch? Several prior works have proposed to use a de-noising network as a feature extractor and learn a mask decoder head from scratch [73, 76]. Our approach uses a pre-trained VAE instead, which is not only a highly effective, general-purpose image encoder, but is also used in the de-noising network pre-training. We now ablate this design choice by replacing the VAE with a dedicated mask prediction module. We ablate both a CNN mask decoder from [73] and an MLP decoder from [61] and train them jointly with the rest of the model at the full resolution (last 2 rows in Table 5). Removing the pre-trained VAE has a moderately negative effect on performance on Ref-YTB, and, notably, destroys the model’s ability to generalize to the out-of-distribution RefVPS (10.6 and 6.9 points drop for CNN and MLP, respectively). These results underscore the main takeaway of our analysis - preserving the entirety of the generative model’s architecture is key for maximizing generalization in RVS.

#### 6. Discussion

We introduced REM, a framework that leverages Internetscale video-language representations learned by diffusion models to enable open-world referring video segmentation. By preserving the full generative model architecture, REM preserves the universal visual-language mapping, allowing it to generalize beyond object-centric segmentation to dynamic, non-object concepts. Our experiments show strong generalization on standard datasets and our new Ref-VPS benchmark for dynamic video processes. Despite limited training on object masks, REM outperforms prior methods by up to 12 points in region similarity out-of-domain. Notably, we also show that advances in video diffusion models directly improve video segmentation performance.

#### Acknowledgement

This project is supported in part by Toyota Research Institute, NSF Grant 2106825, and NIFA Award 2020-6702132799.

#### References

- [1] Ali Athar, Jonathon Luiten, Paul Voigtlaender, Tarasha Khurana, Achal Dave, Bastian Leibe, and Deva Ramanan. BURST: A benchmark for unifying object recognition, segmentation and tracking in video. In WACV, 2023. 2, 5, 6, 7, 16, 19
- [2] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV, 2021. 2, 3
- [3] Zhipeng Bao, Martial Hebert, and Yu-Xiong Wang. Generative modeling for multi-task visual learning. In ICML, 2022. 3
- [4] Lawrence W Barsalou. Perceptual symbol systems. Behavioral and brain sciences, 22(4):577–660, 1999. 1
- [5] Miriam Bellver, Carles Ventura, Carina Silberer, Ioannis Kazakos, Jordi Torres, and Xavier Giro-i Nieto. A closer look at referring expressions for video object segmentation. Multimedia Tools and Applications, 82(3):4419–4438, 2023. 2
- [6] Rodrigo Benenson, Stefan Popov, and Vittorio Ferrari. Large-scale interactive object segmentation with human annotators. In CVPR, 2019. 12
- [7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3, 6, 15, 17
- [8] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. COCOStuff: Thing and stuff classes in context. In CVPR, 2018. 7, 12
- [9] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In ECCV, 2020. 2
- [10] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. VideoCrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 3, 15
- [11] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. VideoCrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, 2024. 3, 15
- [12] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, and Joon-Young Lee. Tracking anything with decoupled video segmentation. In CVPR, 2023. 3
- [13] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In RSS, 2023. 3

- [14] Achal Dave, Tarasha Khurana, Pavel Tokmakov, Cordelia Schmid, and Deva Ramanan. Tao: A large-scale benchmark for tracking any object. In ECCV, 2020. 14
- [15] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. Mevis: A large-scale benchmark for video segmentation with motion expressions. In ICCV, 2023. 6, 19
- [16] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The PASCAL visual object classes (VOC) challenge. International journal of computer vision, 88:303–338, 2010. 6
- [17] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. DataComp: In search of the next generation of multimodal datasets. In NeurIPS, 2023. 15
- [18] Kirill Gavrilyuk, Amir Ghodrati, Zhenyang Li, and Cees GM Snoek. Actor and action video segmentation from a sentence. In CVPR, 2018. 2
- [19] Amirata Ghorbani, James Wexler, James Y Zou, and Been Kim. Towards automatic concept-based explanations. NeurIPS, 2019. 2
- [20] Shuting He and Henghui Ding. Decoupling static and hierarchical motion perception for referring video segmentation. In CVPR, 2024. 7, 16
- [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 3
- [22] Ronghang Hu, Marcus Rohrbach, and Trevor Darrell. Segmentation from natural language expressions. In ECCV,

2016. 2

- [23] Tianrui Hui, Shaofei Huang, Si Liu, Zihan Ding, Guanbin Li, Wenguan Wang, Jizhong Han, and Fei Wang. Collaborative spatial-temporal modeling for language-queried video actor segmentation. In CVPR, 2021. 2
- [24] Hueihan Jhuang, Juergen Gall, Silvia Zuffi, Cordelia Schmid, and Michael J Black. Towards understanding action recognition. In ICCV, 2013. 3
- [25] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to objects in photographs of natural scenes. In EMNLP, 2014. 3
- [26] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In CVPR, 2024. 14
- [27] Anna Khoreva, Anna Rohrbach, and Bernt Schiele. Video object segmentation with language referring expressions. In ACCV, 2019. 2, 5, 6, 19
- [28] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014. 3
- [29] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. LISA: Reasoning segmentation via large language model. In CVPR, 2024. 3
- [30] Lang Lin, Xueyang Yu, Ziqi Pang, and Yu-Xiong Wang. GLUS: Global-local reasoning unified into a single large language model for video segmentation. In CVPR, 2025. 7, 16
- [31] Ruoshi Liu, Rundi We, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3D object. In ICCV, 2023. 3

- [32] Ilya Loshchilov, Frank Hutter, et al. Decoupled weight decay regularization. In ICLR, 2019. 19
- [33] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. TrackFormer: Multi-object tracking with transformers. In CVPR, 2022. 2
- [34] Jiaxu Miao, Yunchao Wei, Yu Wu, Chen Liang, Guangrui Li, and Yi Yang. VSPW: A large-scale dataset for video scene parsing in the wild. In CVPR, 2021. 2, 5, 6, 7, 19
- [35] Ke Ning, Lingxi Xie, Fei Wu, and Qi Tian. Polar relative positional encoding for video-language segmentation. In IJCAI, 2020. 2
- [36] OpenAI. Chatgpt, 2023. 5
- [37] Ege Ozguroglu, Ruoshi Liu, D´ıdac Sur´ıs, Dian Chen, Achal Dave, Pavel Tokmakov, and Carl Vondrick. pix2gestalt: Amodal segmentation by synthesizing wholes. In CVPR,

2024. 2, 3

- [38] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 3, 6, 19
- [39] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In CVPR, 2016. 2, 6
- [40] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30K entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In ICCV,

2015. 2

- [41] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alexander Sorkine-Hornung, and Luc Van Gool. The 2017 DAVIS challenge on video object segmentation. arXiv:1704.00675, 2017. 2, 6, 16
- [42] Sara F Popham, Alexander G Huth, Natalia Y Bilenko, Fatma Deniz, James S Gao, Anwar O Nunez-Elizalde, and Jack L Gallant. Visual and linguistic semantic representations are aligned at the border of human visual cortex. Nature neuroscience, 24(11):1628–1636, 2021. 1
- [43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 3, 15
- [44] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. GLaMM: Pixel grounding large multimodal model. In CVPR, 2024. 3
- [45] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. SAM 2: Segment anything in images and videos. In ICLR, 2025. 5, 7
- [46] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3
- [47] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. UNet: Convolutional networks for biomedical image segmentation. In MICCAI, 2015. 3

- [48] Carsten Rother, Vladimir Kolmogorov, and Andrew Blake. GrabCut interactive foreground extraction using iterated graph cuts. ACM TOG, 2004. 2
- [49] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. LAION-5B: An open large-scale dataset for training next generation image-text models. NeurIPS, 2022. 2, 3
- [50] Seonguk Seo, Joon-Young Lee, and Bohyung Han. URVOS: Unified referring video object segmentation network with a large-scale benchmark. In ECCV, 2020. 2, 5, 6, 19
- [51] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 3
- [52] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. NeurIPS, 2023. 3
- [53] Pavel Tokmakov, Jie Li, and Adrien Gaidon. Breaking the ”object” in video object segmentation. In CVPR, 2023. 6
- [54] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017. 2
- [55] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 3, 6, 17
- [56] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. ModelScope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 2, 3, 6, 15, 17
- [57] Dongming Wu, Tiancai Wang, Yuang Zhang, Xiangyu Zhang, and Jianbing Shen. OnlineRefer: A simple online baseline for referring video object segmentation. In CVPR,

2023. 2

- [58] Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, and Ping Luo. Language as queries for referring video object segmentation. In CVPR, 2022. 2, 6, 7, 16, 19
- [59] Junfeng Wu, Yi Jiang, Qihao Liu, Zehuan Yuan, Xiang Bai, and Song Bai. General object foundation model for images and videos at scale. In CVPR, 2024. 3
- [60] Yi Wu, Jongwoo Lim, and Ming-Hsuan Yang. Online object tracking: A benchmark. In CVPR, 2013. 2
- [61] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. SegFormer: Simple and efficient design for semantic segmentation with transformers. In NeurIPS, 2021. 8
- [62] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. ODISE: Open-vocabulary panoptic segmentation with text-to-image diffusion models. In CVPR, 2023. 3
- [63] Jiarui Xu, Xingyi Zhou, Shen Yan, Xiuye Gu, Anurag Arnab, Chen Sun, Xiaolong Wang, and Cordelia Schmid. Pixelaligned language model. In CVPR, 2024. 3
- [64] Ning Xu, Linjie Yang, Yuchen Fan, Jianchao Yang, Dingcheng Yue, Yuchen Liang, Brian Price, Scott Cohen,

- and Thomas Huang. YouTube-VOS: Sequence-to-sequence video object segmentation. In ECCV, 2018. 2
- [65] Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Ping Luo, Zehuan Yuan, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In CVPR, 2023. 3, 6, 15, 16
- [66] Cilin Yan, Haochen Wang, Shilin Yan, Xiaolong Jiang, Yao Hu, Guoliang Kang, Weidi Xie, and Efstratios Gavves. VISA: Reasoning video object segmentation via large language models. In ECCV, 2024. 7, 16
- [67] Shilin Yan, Renrui Zhang, Ziyu Guo, Wenchao Chen, Wei Zhang, Hongyang Li, Yu Qiao, Hao Dong, Zhongjiang He, and Peng Gao. Referred by multi-modality: A unified temporal transformer for video object segmentation. In AAAI,

2024. 2, 6, 16

- [68] Linwei Ye, Mrigank Rochan, Zhi Liu, and Yang Wang. Cross-modal self-attention network for referring image segmentation. In CVPR, 2019. 2
- [69] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. In ICLR, 2024. 3
- [70] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg. Modeling context in referring expressions, 2016. 6, 17, 19
- [71] Fangao Zeng, Bin Dong, Yuang Zhang, Tiancai Wang, Xiangyu Zhang, and Yichen Wei. MOTR: End-to-end multipleobject tracking with transformer. In ECCV, 2022. 2
- [72] Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, and Ming-Hsuan Yang. A tale of two features: Stable diffusion complements DINO for zero-shot semantic correspondence. NeurIPS,

2023. 3

- [73] Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu. Unleashing text-to-image diffusion models for visual perception. In ICCV, 2023. 2, 3, 8
- [74] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-Sora: Democratizing efficient video production for all, 2024. 3
- [75] Zikun Zhou, Wentao Xiong, Li Zhou, Xin Li, Zhenyu He, and Yaowei Wang. Driving referring video object segmentation with vision-language pre-trained models. arXiv preprint arXiv:2405.10610, 2024. 3, 6
- [76] Zixin Zhu, Xuelu Feng, Dongdong Chen, Junsong Yuan, Chunming Qiao, and Gang Hua. Exploring pre-trained textto-video diffusion models for referring video object segmentation. In ECCV, 2024. 2, 3, 6, 8, 16, 17

[Figure 129]

- Figure A. Distribution of sample lengths in Ref-VPS. Most of our samples are between 2.5 to 5 seconds in length, but can go up to more than 20 seconds.

In this supplementary material, we first include additional details for our Ref-VPS dataset in Section A. Next, we offer a deeper quantitative analysis, covering VAEbased mask reconstruction, failure modes, and computational costs, etc., in Section B. Section C presents additional qualitative evaluations, including visualizations on typical failure cases, challenging fight scenes, and ambiguous or overlapping scenarios. Finally, in Section D, we report all the implementation details.

#### A. Ref-VPS Dataset Details

##### A.1. Dataset collection pipeline and statistics

During our dataset collection, we first established a nonexhaustive taxonomy of five broad and possibly overlapping concepts. This taxonomy was designed to encompass key modes of dynamic change while offering a structured framework for the task. The concepts and their definitions are as follows:

- • Temporal Object Changes: Phenomena where an object’s state or shape evolves over time (e.g., object deformation, melting)
- • Motion Patterns: Motion in amorphous or non-rigid regions (e.g., water ripples, flickering flames)
- • Dynamic Environmental Changes: Environmental transformations affecting spatial regions over time (e.g., clouds moving across the sky, waves rising )
- • Interaction Sequences: Events characterized by interactions between objects (e.g., bullet hitting glass, object collisions)
- • Pattern Evolution: Progressive changes in patterns or textures (e.g., changing patterns of smoke dispersion, fluctuating light levels)

Our final dataset comprises 145 video clips representing

Clips 145 FPS 24 Frames 23,442 Concepts 39 Avg length (s) 6.74 Annotation FPS 6 Min-resolution 712 × 576 Max-resolution 1024 × 576

Table A. Statistics of our Ref-VPS benchmark. Our dataset contains 145 video clips covering 39 concepts for dynamic processes.

39 distinct dynamic process concepts. We report a comprehensive list of key statistics in Table A. Most of our samples are between 2.5 and 5 seconds in length, but can go up to more than 20 seconds. The distribution of our sample lengths is reported in Figure A.

##### A.2. Annotation visualizations

Figure B showcases examples of our Ref-VPS segmentation mask annotations. Our annotations capture the full extent of target objects, as seen with the icicle (second row) and the glass (fourth row). For more ambiguous cases, such as glowing water (first row) or a dandelion being blown (third row), only the confident regions are labeled, while uncertain areas are marked as Ignore (yellow). These Ignore Regions are excluded from metric computation, ensuring that evaluations focus on reliable mask regions and are not penalized for inherently ambiguous boundaries.

##### A.3. Mask annotation accuracy evaluation

To assess the quality of our annotations, we compute interannotator agreement on the Ref-VPS dataset. Specifically, an independent annotator relabeled a subset of the dataset, covering all 39 dynamic concepts, using the same annotation protocol. Following the evaluation approach of Benenson et al. [6], we report an inter-annotator mean IoU (mIoU) of 87.1%, significantly higher than the ∼80% agreement number reported for COCO [8]. This high agreement demonstrates the effectiveness of our annotation protocol, particularly the use of Ignore labels to handle ambiguity in subjective scenarios.

#### B. Additional Quantitative Evaluations

In this section, we provide additional quantitative evaluations for our proposed REM. Same as our ablation study in the main paper, we conduct these experiments using the MS-1.4B version, unless staged otherwise.

##### B.1. Mask reconstruction accuracy analysis

In designing our REM model, we repurpose a pre-trained VAE as the mask decoder, based on the intuition that

###### Original GroundTruth

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

“the bright blue lights moving”

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

“the ice forming”

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

“the dandelion being blown away by a woman”

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

“the glass jar breaking into pieces”

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

“the light reflected in the water”

- Figure B. Samples from our Ref-VPS dataset. Ground-truth masks are shown in pink, and the Ignore regions are shown in yellow. Pixels inside the Ignore regions are not included in the metric calculation.

[Figure 150]

- Figure C. Quantitative evaluation of our REM (MS-1.4B) failure modes on Ref-VPS: Significant shape change is the primary failure mode, with motion and prompt complexity having secondary impacts.

large-scale pre-training enables the VAE to effectively reconstruct masks as images. To validate this assumption, we quantitatively evaluate the VAE’s reconstruction performance on binary mask images, following the methodology of Marigold [26]. Specifically, we assess reconstruction accuracy on 3,471 binary masks from the Ref-YTB training set (one per video). The VAE achieves a mean absolute error (MAE) of 0.0144 for mask reconstruction. In comparison, reconstructing the corresponding RGB frames yields a higher MAE of 0.1236, reflecting the greater difficulty of the RGB task. Furthermore, the VAE attains a mask reconstruction mIoU of 99.33% between the input and output masks. These results support our approach of converting masks into 3-channel inputs for compatibility with pretrained auto-encoders, effectively mitigating concerns about domain mismatch.

##### B.2. Failure mode analysis

We conducted a quantitative evaluation of our REM failure modes on Ref-VPS in Figure C. We measure performance degradation across motion and shape changes (following Dave et al. [14]), and prompt complexity (sentence length). Our analysis reveals that significant shape change is the primary failure mode, with motion and prompt complexity having secondary impacts. These results further illustrate the challenge of segmenting dynamic concepts in videos.

##### B.3. Computational cost

We report the inference speed and memory consumption of REM (MS-1.4B) alongside key baselines in Table B, using their official public implementations. All measurements are conducted on 32-frame clips from Ref-DAVIS using a single NVIDIA A100 GPU, with averages computed over

|Method|Memory (GB) Speed (FPS)<br><br>|
|---|---|
|MUTR UNINEXT VD-IT|34.1 13.6 9.7 3.3<br><br>72.8 7.1<br><br>|
|REM (MS-1.4B)|41.8 7.1|

- Table B. Inference costs of REM and top RVS methods on RefDAVIS. Both the memory requirements and the runtime of REM are on par with other models in the literature.

|Method|Memory (GB) Total Runtime (hr)|
|---|---|
|MUTR UNINEXT VD-IT<br><br>|30.4 134 30.2 1906 68.5 260|
|REM (MS-1.4B)<br><br>|61.8 174|

- Table C. Training costs of REM and top RVS methods. Our costs are on par with prior work and are notably significantly lower compared to UNINEXT, the state-of-the-art RVOS approach.

80 runs. As shown, the inference costs of REM align with those of other state-of-the-art approaches.

For training, REM requires 174 hours on four A100 GPUs. Since most baselines do not disclose training costs, we estimate them under equivalent hardware and conditions (excluding I/O time), and report the results in Table C, including per-GPU memory consumption. Our training efficiency is on par with prior works. Notably, UNINEXT, the current state-of-the-art in RVOS, requires approximately 6.3 times longer to train than REM due to its reliance on over ten supervised datasets to achieve strong object segmentation performance. In contrast, REM leverages Internet-scale pre-training to attain comparable performance on in-domain benchmarks and significantly outperforms UNINEXT in out-of-distribution scenarios, all while incurring a fraction of the training cost.

##### B.4. Ablation of the training strategy

Our model adopts a two-stage training strategy (detailed in Section D), where we first pre-train on image-only data to learn spatial representations, followed by joint fine-tuning on mixed image and video data. In this section, we compare the two-stage approach to a single-stage alternative and analyze the impact of varying the amount of image data used in the second stage.

Benefits of two-stage training. As reported in Table D, the two-stage training strategy yields superior performance on both the standard RVS benchmark and the out-ofdistribution Ref-VPS dataset. This improvement stems from allowing the model to first acquire strong spatial priors from image-only data before incorporating the more complex temporal dynamics of videos. Additionally, initializing with well-trained spatial weights enhances training stability

|Training stages|Training strategy<br><br>|Ref-YTB (J &F)|Ref-VPS J<br><br>|
|---|---|---|---|
|Two-stages (Ours) Single-stage<br><br>|Images → Images & Videos Images & Videos|68.4 66.3<br><br>|49.0 46.33|

- Table D. Comparison between our default two-stage training and single-stage training strategy. The two-stage training strategy allows the model to first learn strong spatial representations from image-only data before incorporating the more complex temporal dynamics present in videos. Therefore, it achieves better results for both the standard RVS benchmark and our out-of-distribution Ref-VPS dataset.

|Images:Videos<br><br>|Ref-YTB (J &F)<br><br>|Ref-VPS J|
|---|---|---|
|2:1<br><br>1:1 (Ours) No images<br><br>|67.0 68.4 68.7<br><br>|48.0 49.0 40.7|

- Table E. Impact of the image data volume used in the second stage. Incorporating image data in the second stage significantly enhances the model’s generalization on the Ref-VPS dataset compared to the version trained without images, while using more image data yields suboptimal results. Overall, a balanced integration of image and video data is key to the success of our approach.

and convergence.

Impact of image data used in the second stage. In our default two-stage setup, we use an equal amount of image and video data during the second stage. To investigate the effect of image data volume, we consider two variants: one with twice as many images and one with no image data. As shown in Table E, using no image data significantly degrades generalization on Ref-VPS, underscoring the importance of image supervision. Conversely, doubling the image data leads to performance degradation on both benchmarks, suggesting that excessive reliance on static visual information can hinder the learning of spatiotemporal dynamics. These results highlight the importance of a balanced integration of image and video data for effective training.

##### B.5. Effect of generative pre-training on RVS

We focus on how generative pre-training affects the RVS performance in this section. We focus on comparing our MS-1.4B model variants with other pre-trained diffusion models of a similar parameter size.

We begin by evaluating the effect of Image generation pre-training in Table F. As a baseline, we first fine-tune Stable Diffusion 2.1 [7] (an image generation model, denoted as SD2.1) on individual frames (column 1 in the table). This variant has no temporal modeling capacity, but neither does UNINEXT [65] - the state-of-the-art approach for RVOS. However, it strongly underperforms compared to our best video-based variants, not only on Ref-VPS but also on the object-centric Ref-YTB. This shows that while generative

|Benchmark|SD2.1 VC-1 VC-2 MS-1.4B<br><br>|
|---|---|
|Ref-YTB (J &F) Ref-VPS (J )<br><br>|60.2 57.5 64.9 63.5 29.8 28.0 36.8 40.0|

- Table F. Analysis of the effects of generative pre-training on RefYTB and Ref-VPS. Both large-scale image pre-training as well as learning to model video-language interactions are important for robust RVS performance.

|Noise Level|Ref-YTB (J &F) Ref-YTB (J )<br><br>|
|---|---|
|200 50|59.2 36.2 62.9 35.0<br><br>|
|0 (Ours)<br><br>|62.9 40.5|

- Table G. Ablation study on the choice of the noisy timestep. The best performance is achieved with minimal noise (t = 0), validating our design.

|Method<br><br>|Ref-VPS<br><br>|Ref-DAVIS|
|---|---|---|
| |J Temp. Con.<br><br>|J Temp. Con.|
|MUTR UNINEXT VD-IT<br><br>|24.1 2.9 26.3 5.2 35.3 4.7<br><br>|64.8 3.4 68.2 5.2 66.2 3.1|
|REM (MS-1.4B)|49.0 2.8<br><br>|69.9 2.1|

- Table H. Temporal Consistency comparison to the state of the art on Ref-VPS and Ref-DAVIS. Our approach demonstrates the best temporal consistency on both object-centric and non-objectcentric datasets.

pre-training relies heavily on images, video data is crucial for learning effective representations for tracking.

Next, we evaluate two variants of the VideoCrafter model [10, 11] (denoted as VC-1 and VC-2 in Table F), both initialized from Stable Diffusion 2.1 [7] and trained on 600M images and 10-20M videos. VC-2 focuses on highquality data curation, which has been shown to be important for representation learning in the past [17, 43], and leads to substantial performance gains across both benchmarks. Finally, the ModelScope [56] approach is also initialized from Stable Diffusion, but trained on the larger LAION 2B and a similar amount of high-quality video data (last column in Table F). It performs comparably to VC2 on RefYTB, while demonstrating the best zero-shot generalization to Ref-VPS among all the variants, making it our default representation. These results highlight that large-scale image pre-training, combined with generative video-language modeling, is important for generalization in RVS.

##### B.6. Ablation study on the noisy timestep

In the main paper, we default to a noise timestep of t = 0, based on the observation that our task formulation focuses on direct mask latent prediction rather than denoising. This design choice eliminates the need for injecting noise into the latent space. To empirically validate this decision, we conduct a single-stage training experiment on the Ref-YTB

|Method|Mask annotation<br><br>|MeViS|
|---|---|---|
| | |J &F J F|
|Referformer [58] VISA-13B [66] DsHmp [20] GLUS [30]|MeViS<br><br>Ref-COCO/+/g, Ref-YTB, MeViS, Ref-DAVIS, ReVOS, LVVIS, Refclef, ADE20k<br><br>MeViS<br><br>Ref-YTB, MeViS, Ref-DAVIS, ReVOS, LVVIS<br><br>|31.0 29.8 32.2 44.5 41.8 47.1 46.4 43.0 49.8 51.3 48.5 54.2|
|REM (Wan-14B) REM (Wan-14B)<br><br>|MeViS Ref-COCO/+/g, Ref-YTB, MeViS|57.6 54.3 60.9 60.3 57.2 63.4<br><br>|

- Table I. Comparison to the state of the art on the MeViS benchmark with a comprehensive list of mask annotations used in training. Our REM (Wan-14B) reaches a new state of the art on MeViS while relying on orders of magnitude fewer pixel-level annotations than prior work.

dataset and report the results in Table G. The findings confirm our hypothesis: minimal noise (i.e., t = 0) consistently leads to the best performance, reinforcing the suitability of this choice for our predictive framework.

[Figure 151]

[Figure 152]

MUTR 27.2

VDIT 29.0

##### B.7. Temporal consistency evaluation

Evaluating temporal consistency in video segmentation remains a challenging task, as it is difficult to disentangle variations caused by model inconsistency from those arising due to genuine object deformations. For example, the temporal consistency metric initially introduced in the DAVIS dataset [41] was applied only to videos with minimal object deformation and occlusion, and was eventually deprecated by the dataset authors due to its limited applicability.

[Figure 153]

[Figure 154]

UNINEXT 30.2 REM MS-1.4B 37.5

To address these challenges, we adopt a simple yet effective temporal consistency metric that quantifies frameto-frame stability. Specifically, we compute the average difference in Intersection-over-Union (IoU) between predicted masks and ground truth masks across consecutive frames. Formally, the metric is defined as:

Figure D. Class-wise J scores (mIoU) across 454 object classes demonstrating concept coverage on BURST. As indicated by the arrows, REM is more robust on the most challenging categories compared to other methods.

traditional RVOS benchmarks, shows the poorest temporal stability across both datasets.

Tn

N

1 N

1 Tn

Temp. Con. =

(IoUdiff) , (A)

##### B.8. Concept coverage plot on BURST dataset

n=1

t=1

where N is the number of samples and Tn is the number of frames in the nth sample, and

Figure D presents the concept coverage plots on the BURST [1] dataset for VD-IT [76], MUTR [67], UNINEXT [65], and our method. As shown, REM consistently outperforms the baselines on the most challenging categories, further highlighting its strong generalization ability across a diverse range of visual concepts.

IoUdiff = IoU(Predt+1,GTt+1) − IoU(Predt,GTt).

(B) Lower values indicate better temporal consistency. However, it is important to interpret this metric in conjunction with prediction accuracy, as trivially empty predictions would yield a perfect consistency score of zero without meaningful segmentation.

##### B.9. Additional evaluation on MeViS

Data efficiency. We first provide the data source for all the baseline models we compared in Table I. Our REM (Wan-14B) reaches a new state of the art on MeViS while relying on orders of magnitude fewer pixel-level annotations than prior work. In particular, the strongest baseline, GLUS, couples a large-scale multimodal LLM with SAM2 masks and aggregates supervision from at least six videoand image-level datasets. In contrast, the full version of REM is fine-tuned on just three datasets, yet it improves

We report both region similarity and temporal consistency on Ref-VPS and Ref-DAVIS (both sampled at 24 fps) in Table H. REM achieves the best temporal consistency on both object-centric and non-object-centric datasets. Interestingly, while MUTR also attains a strong consistency score on Ref-VPS, this is primarily due to its frequent output of empty masks, as reflected in its low region similarity. Conversely, UNINEXT, despite being the state-of-the-art on

the previous best J &F from 51.3% to 60.3%. These results underline that preserving the generative architecture of a diffusion model transfers rich visual–language priors so effectively that only a modest amount of task-specific data is needed to surpass far heavier-supervised baselines.

Single-dataset training. To isolate the contribution of our training protocol, we additionally built a variant of our Wan model by finetuning on the MeViS training set alone. Performance decreases by only 2.7 in terms of J &F, yet it still outperforms the strongest published baseline by 6.3 points. This resilience demonstrates that our mixed training strategy endows the model with robust spatial–temporal representations that generalize even when the downstream supervision is extremely limited. We include the training details for our Wan variant in Section D.

#### C. Additional Qualitative Evaluations

##### C.1. Failure cases visualizations

A few representative failure cases of REM (MS-1.4B) on Ref-VPS are shown in Figure E. Our method suffers from object-centric bias in the most challenging scenarios (e.g., light reflection and veins) and struggles with extremely fast motion (e.g., the lightning strike).

##### C.2. Evaluation on challenging fight scenes

Fight sequences in movies and animated shows present a particularly challenging setting for referring video segmentation. These scenes are often characterized by severe and frequent occlusions, objects or characters exiting the frame, and rapid camera pose changes. Such factors cause drastic variations in appearance, demanding high temporal and semantic consistency to accurately track, re-identify, and segment the referred entities.

Our REM excels in this domain of extremely challenging samples as illustrated in Figure F. In contrast, both UNINEXT and VD-IT exhibit clear failure cases when the referred entity undergoes large occlusions or momentarily disappears from view. Notably, despite utilizing a video diffusion backbone, VD-IT fails to fully exploit the temporal consistency learned during video diffusion pre-training, whereas REM maintains robust performance under these challenging conditions.

##### C.3. Comparisons on ambiguous or overlapping scenarios

To assess how well our method handles visually ambiguous or overlapping scenarios, we present a qualitative comparison between REM and VD-IT, the strongest baseline on this benchmark, in Figure G. While many of these examples lack a single ground-truth segmentation, REM consistently produces more accurate and coherent masks in the confidently visible regions. For instance, in the first row,

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

“the light reflecting off the bald head” “the light reflected in the water”

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

“the lightning strike” “the veins on the arm”

Figure E. Failure cases of REM (MS-1.4B) on Ref-VPS. Our model still exhibits some object-centric bias and struggles with extremely dynamic entities such as lightning.

our method accurately segments only the clearly visible portions of lava that become apparent after being struck by a wave, whereas VD-IT incorrectly includes the entire wave. In the second row, REM reliably segments all regions of glowing water, while VD-IT detects only a few scattered patches. These results demonstrate our model’s robustness in ambiguous settings and its capacity to avoid over-segmentation.

#### D. Implementation Details

Benchmark details and baseline models We report the details about the benchmarks we used in Table J. We quote the results of all the baseline models on Ref-YTB, Ref-DAVIS, and MeViS from their original papers. For the zero-shot evaluation of BURST, VSPW, and Ref-VPS, we report the numbers by running the official checkpoints of MUTR1, UNINEXT2, VD-IT3, and GLUS4.

Training details. Our approach builds upon two stateof-the-art text-to-video diffusion architectures: ModelScope [56] and Wan [55]. Additional video diffusion backbones are evaluated in Section B. ModelScope comprises 1.4 billion parameters and extends Stable Diffusion [7] with temporal modules. We adopt a two-stage training protocol following Zhu et al. [76]: in Stage I, we fine-tune only the spatial weights on Ref-COCO imagetext pairs[70] for one epoch; in Stage II, we fine-tune all

- 1https://github.com/OpenGVLab/MUTR
- 2https://github.com/MasterBin-IIAU/UNINEXT
- 3https://github.com/buxiangzhiren/VD-IT
- 4https://github.com/GLUS-video/GLUS

“the man with red cape”

“the man with white hair”

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Ours

|[Figure 169]|[Figure 170]|[Figure 171]|
|---|---|---|

|[Figure 172]|[Figure 173]|[Figure 174]|
|---|---|---|

UNINEXT

|[Figure 175]|[Figure 176]|[Figure 177]|
|---|---|---|

|[Figure 178]|[Figure 179]|
|---|---|

[Figure 180]

VD-IT

“the boy with red collar and pink hair”

“the man in red and gold suit”

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Ours

|[Figure 187]|[Figure 188]|
|---|---|

|[Figure 189]|[Figure 190]|[Figure 191]|
|---|---|---|
|[Figure 192]|[Figure 193]|[Figure 194]|

[Figure 195]

UNINEXT

|[Figure 196]|[Figure 197]|[Figure 198]|
|---|---|---|

VD-IT

“the boy with dark hair”

“the boy wearing blue shirt”

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Ours

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|[Figure 208]|[Figure 209]|
|---|---|---|
|[Figure 210]|[Figure 211]|[Figure 212]|

[Figure 213]

UNINEXT

|[Figure 214]|
|---|

|[Figure 215]|
|---|

[Figure 216]

VD-IT

“the man with white hair”

“the man without a shirt”

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

Ours

|[Figure 223]|
|---|

|[Figure 224]|[Figure 225]|
|---|---|

[Figure 226]

[Figure 227]

[Figure 228]

UNINEXT

|[Figure 229]|
|---|

|[Figure 230]|[Figure 231]|
|---|---|

[Figure 232]

[Figure 233]

[Figure 234]

VD-IT

- Figure F. Qualitative comparison of REM (MS-1.4B) with state-of-the-art baselines on dynamic and challenging fight scenes. The incorrectly segmented frames are outlined in red. REM outperforms the other methods in handling frequent occlusions and POV changes. For a better illustration of the differences, please watch the full videos here.

“the lava flowing” VDIT REM (Ours)

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

“the luminescence in the water”

VDIT

REM (Ours)

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

VDIT

“the smoke blown out”

REM (Ours)

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

- Figure G. Comparison on ambiguous or overlapping scenarios in Ref-VPS between VD-IT and REM (MS-1.4B). While no single perfect prediction exists for these samples, our method is both more precise and more consistent.

|Benchmark|Type Training Samples Testing Samples<br><br>|
|---|---|
|Ref-COCO [70]<br><br>|Image 320K -|
|Ref-YTB [50] MeViS [15] Ref-DAVIS [27] BURST [1] VSPW [34]<br><br>|Video 12,913 2,096 Video 1,712 140 Video - 90 Video - 2,049 Video - 343|

- Table J. Details about the benchmarks we used for training and evaluation.

Ref-YTB, for an additional 8 epochs on MeViS.

Evaluation details. We follow the standard evaluation protocol for Ref-YTB, Ref-DAVIS, and MeViS. For BURST [1] and VSPW [34], neither of them contains referring text for the segmented entities. We automatically generate referring expressions using only the category information of the mask entity as “the <class>” (e.g., the hat). For VSPW, we conduct our evaluation on the validation set, which has 66 different stuff categories. In the case of BURST, we evaluate the combined validation and test set, which contains 454 classes and a total of 2,049 sequences. For inference and evaluation, we follow the standard VSPW protocol for our RVS evaluation. For BURST, we predict masks for all the original frames, and compute the metrics for the annotated ones provided by the dataset. For Ref-VPS, to ensure high-quality performances, we perform inference at the original 24 FPS and compute evaluation metrics on the annotated frames at 6 FPS.

network weights for 40 epochs using Ref-YTB video–text examples [50] supplemented with 12K Ref-COCO images converted into pseudo-videos following Wu et al. [58]. By contrast, Wan employs a unified diffusion transformer that jointly models spatial and temporal information, without dedicated temporal modules [38]. Accordingly, we train this variant in a single stage on the combined Ref-COCO and Ref-YTB datasets for 80k steps, with half of the steps trained with images and half trained with videos. Throughout training, the text encoder and VAE remain frozen.

Unless otherwise stated, all models are trained and evaluated at a resolution of 512 × 512. We use AdamW [32] for optimization with a constant learning rate of 1e-6. The training batch size is 4 for ModelScope and 8 for Wan, and for each sample, we randomly load an 8-frame video clip for ModelScope and a 17-frame video clip for Wan. We train our model using eight NVIDIA 80GB A100 GPUs, and it takes about 1 week to finish the whole training process.

For MeViS, we train our MS-1.4B variant by finetuning our Stage I checkpoint jointly on MeViS and Ref-YTB for 37 epochs. We achieve our best results on MeViS by finetuning the Wan-14B checkpoint trained on Ref-COCO and

