## arXiv:2501.03059v1[cs.CV]6Jan2025

###### Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation

Guy Yariv1,3 Yuval Kirstain1 Amit Zohar1 Shelly Sheynin1 Yaniv Taigman1 Yossi Adi2,3 Sagie Benaim3 Adam Polyak1

1GenAI, Meta 2FAIR, Meta 3The Hebrew University of Jerusalem

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

‘A monster and squirrel shake hands gently, and continue moving while shaking hands’

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

‘The metallic robot with blue eyes performs a series of fast push-ups.’

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

(i) Input (ii) Output

Figure 1. THROUGH-THE-MASK is an Image-to-Video method that animates an input image based on a provided text caption. The generated video (rows 2 and 4) leverages mask-based motion trajectories (rows 1 and 3), enabling accurate animation of multiple objects.

###### Abstract

We consider the task of Image-to-Video (I2V) generation, which involves transforming static images into realistic video sequences based on a textual description. While recent advancements produce photorealistic outputs, they frequently struggle to create videos with accurate and consistent object motion, especially in multi-object scenarios. To address these limitations, we propose a two-stage compo-

sitional framework that decomposes I2V generation into: (i) An explicit intermediate representation generation stage, followed by (ii) A video generation stage that is conditioned on this representation. Our key innovation is the introduction of a mask-based motion trajectory as an intermediate representation, that captures both semantic object information and motion, enabling an expressive but compact representation of motion and semantics. To incorporate the learned representation in the second stage, we utilize

object-level attention objectives. Specifically, we consider a spatial, per-object, masked-cross attention objective, integrating object-specific prompts into corresponding latent space regions and a masked spatio-temporal self-attention objective, ensuring frame-to-frame consistency for each object. We evaluate our method on challenging benchmarks with multi-object and high-motion scenarios and empirically demonstrate that the proposed method achieves state-of-the-art results in temporal coherence, motion realism, and text-prompt faithfulness. Additionally, we introduce SA-V-128, a new challenging benchmark for singleobject and multi-object I2V generation, and demonstrate our method’s superiority on this benchmark. Project page is available at https://guyyariv.github.io/TTM/.

###### 1. Introduction

Image-to-Video (I2V) generation transforms static images into realistic video sequences guided by textual descriptions. Recently, significant progress has been made in this task, with models such as [6, 12, 30, 41, 43, 55], which enable the generation of photorealistic and consistent output. However, current works still struggle to generate videos with consistent and faithful object motion. These limitations are especially evident in scenarios with multiple objects, as shown in our experiments, where capturing the correct motion and interactions is challenging.

Several works [6, 12, 30, 41] directly map an input image (and possibly text) to an output video in a single, end-to-end pipeline. By doing so, the underlying model must implicitly reason about object semantics and motion while simultaneously generating a plausible appearance of all objects. As the range of possible motions and interactions scales significantly with the number of objects, this makes it difficult for current models to generate plausible outputs. An alternative approach is to decompose the training process into a two-stage compositional process: (i) Given the input image, generate an explicit intermediate representation; (ii) Utilize the generated representation and the input image to generate the full video. Recent work [43], proposed using Optical Flow (OF) for this representation. However, this has several drawbacks. First, only motion, without semantics, is represented in OF. Second, such motion representation is redundant in the context of I2V generation. Predicting per-point pixel motion may not be required to depict plausible object motion. Doing so may result in unnecessary errors (e.g., wrong prediction in pixels from non-moving objects). Such errors can then significantly influence the second stage, as the model tries to predict the correct appearance while adhering to incorrect motion.

In this work, we argue that the choice of representation is critical and should capture several properties: (i) it should express both motion and semantics; (ii) it should represent

the motion and interaction of individual objects; and (iii) it should be robust to signal variations and operate at the object level rather than at the pixel level. We argue that a suitable choice satisfying these properties is a mask-based motion trajectory, a time-consistent per-frame semantic mask capturing semantic objects and their motion (see Fig. 1).

Our method follows a two-stage process: first, a network is trained to generate a mask-based motion trajectory conditioned on the input image, segmentation mask, and text prompt. In the second stage, the motion-to-video network generates the final video conditioned on the input image, text prompt, and the generated motion trajectory from the first stage.

For the second stage of our framework (i.e., motion-tovideo), we propose to integrate the generated mask-based motion trajectory’s structure softly, using learned attention layers, ensuring the network adheres to the generated semantics and motion. Specifically, we propose using (i) a masked cross-attention objective, which integrates object-specific prompts directly into corresponding regions of the latent space, using masked cross-attention, and (ii) a masked self-attention objective, which ensures that each object maintains consistency across frames, by using the generated mask in the self-attention mechanism to restrict attention to positions corresponding to the same object.

We compare our approach to a diverse set of recent image-to-video generation approaches on challenging benchmarks that include several objects and significant motion. We demonstrate state-of-the-art performance across diverse metrics, including temporal coherence, motion realism, visual consistency across frames, and text faithfulness to the input prompt. To further advance research on I2V generation, we introduce a new benchmark that includes distinct sets for single-object and multi-object videos, demonstrating superior performance. Finally, we ablate our method, demonstrating the contribution of each component.

###### 2. Related work

Text-to-Video Generation. Recent advances in diffusion models [20, 45, 46] and flow matching techniques [1, 27, 28] have enhanced the capability to generate high-quality images conditioned on textual descriptions [4, 11, 14]. In the context of text-conditioned video generation, several approaches perform diffusion in a low-dimensional latent space, adopting the Latent Diffusion Models (LDM) architecture [37, 42]. Many text-to-video (T2V) models adapt T2I architectures to generate temporally coherent videos, extending beyond the spatial knowledge of T2I training [2, 7, 10, 15, 17, 18, 21, 22, 44, 49–51, 56]. A common approach extends pre-trained T2I models with temporal modules, such as convolutions or attention layers, followed by additional training for video generation [15, 21, 22, 44].

EmuVideo [16] and VideoGen [24], for instance, factorized the text-to-video to two stages: text-to-image and image-tovideo. Recent studies have adopted the transformer-based Diffusion Transformer (DiT) [36] due to its performance over U-Net [31, 32, 34, 38]. Notably, our approach is architecture-agnostic and works with both U-Net and DiT.

Image-To-Video Generation. In I2V, the video generation model is conditioned on the input text, as well as an additional visual input that represents the initial frame of the output video [5, 35]. Several works leverage this additional visual input by fine-tuning of a pre-trained T2V model [6, 9]. Despite encouraging progress in generated video aesthetic [16], current I2V models struggle to generate complex actions or interactions between objects [43].

Recent work has attempted to tackle this challenge. VideoCrafter [9] incorporate an additional image input to preserve the content and style of the reference image. DynamiCrafter [54] use a query transformer to project the image into a text-aligned context, leveraging motion priors from T2V models to animate images. I2VGen-XL [55] employ a two-stage cascade, with the second stage refining resolution and temporal coherence. ConsistI2V [41] use first-frame conditioning by combining the initial latent frame with input noise and enhancing self-attention with intermediate states. AnimateAnything [12] include an additional mask to constrain motion areas. Cinemo [30] introduce three enhancements: (i) prediction of motion residuals, (ii) fine-grained control over motion intensity, and (iii) noise refinement at inference to reduce motion shifts. Our work considers a different two-step compositional approach and is orthogonal to these advances.

Perhaps most relevant to our work is Motion-I2V [43], which follows a two-step generation process: (i) prediction of optical flow displacement, and (ii) generation of the video based on generated optical flow. Our method similarly follows a two-stage approach. However, we offer two key differences. First, we use a different intermediate representation of mask-based motion trajectories. We argue that this choice is significant in multiple aspects: (i) we represent not only motion, but also semantics, enhancing expressivity. (ii) Simultaneously, our representation captures only object-level motion as opposed to pixel-level motion. Doing so makes our generation less susceptible to errors in the first stage, as also observed in [3]. Second, our representation also enables additional flexibility in the second generation stage, which generates a video conditioned on this representation. Specifically, instead of wrapping the generated video using predicted flow, we softly condition the video generation model on the intermediate representation using object-level and temporal attention. We note that recent T2I models have introduced conditioning on specific areas with targeted information to improve fine-grained controllability [25, 33]. To address the I2V setting, we extend the

masked cross-attention introduced in [33] to the video setting and introduce a novel masked self-attention objective.

###### 3. Method

Our method THROUGH-THE-MASK, illustrated in Fig. 2, factorizes I2V into two compositional stages:

- 1. Image-to-Motion Generation: In the first stage, outlined in Sec. 3.2, we generate motion trajectory conditioned on the reference image and motion prompt. This motion trajectory encapsulates the dynamic behavior of individual objects.
- 2. Motion-to-Video Generation: In the second stage, outlined in Sec. 3.3, we use the generated motion trajectory, along with the object-specific prompts and the reference image, to produce a photorealistic video.

Our two-stage process is based on the choice of an explicit intermediate representation of objects’ motion. Ideally, such representation should (i) express both motion and semantics, (ii) represent the interaction of objects, and (iii) be robust to signal variations. We claim that a motion trajectory, i.e., a consistent per-frame video segmentation, satisfies these properties by definition. First, it captures motion, interactions, and type (i.e., semantics), hence satisfying both (i) and (ii). Second, as image segmentation operates at a coarse level (i.e., object level rather than pixel level), we achieve the following desired separation of tasks: The first stage handles coarse object-level semantics, motion, and inter-object interactions. The second stage then handles the intra-object level semantics and appearance. On average, this results in fewer overall errors produced than the pixel-level counterpart in the first stage, satisfying (iii). It also enables greater flexibility in the second stage.

To allow the modeling of our framework, we pre-process our training data, as outlined in Sec. 3.1. The Image-toMotion and Motion-to-Video stages are trained independently but are combined during inference to produce the final video (see supplementary Sec. 9). Additional implementation details are provided in supplementary Sec. 10.

- 3.1. Data Pre-processing

We assume a training dataset of text-video pairs, where the input contains a reference image x(0) and a text prompt c. Our pre-processing pipeline comprises the following components applied on each text-video pair: (i) extraction of prompts for motion-capable objects from the input text, (ii) video segmentation, and (iii) extraction of motion-specific and object-specific prompts from the input text.

Motion-capable Object Prompt Extraction. Using a pretrained Large Language Model (LLM), we extract L objects’ prompts, {o(1),...,o(L)}, relevant to generating specific motion pathways (motion-capable objects) from the input text, c. Notice, L is variable and video-specific. We refer the readers to supplementary Sec. 8 for more details.

[Figure 23]

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

|[Figure 52]|
|---|

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

|[Figure 66]|
|---|

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

- Figure 2. Overview of our I2V framework, transforming a reference image x(0) and text prompt c into a coherent video sequence xˆ. A

pre-trained LLM is used to derive the motion-specific prompt cmotion and object-specific prompts clocal = {c(1)local, . . . , c(localL) }, capturing each object’s intended motion. We generate an initial segmentation mask s(0) from x(0) using SAM2. In Stage 1, the Image-to-Motion

utilizes x(0), s(0), and cmotion to generate mask-based motion trajectories sˆthat represent object-specific movement paths. In Stage 2, the Motion-to-Video takes as input x(0), the generated trajectories sˆ, the text prompt c as a global condition, and object-specific prompts clocal through a masked attention blocks (Section 3.3), producing the final video xˆ.

Video Segmentation. We assume an N-frame video, x = {x(0),...,x(N)}, where x(i) ∈ R3×H×W are frames of resolution H ×W. To obtain a trajectory, we first use Grounding DINO [29] to obtain bounding boxes for each motioncapable object caption o(i) within the first frame, x(0). Using these bounding boxes as input, we use SAM2 [40] to create video segmentation s = {s(0),...,s(N)}, where mask s(i) matches video frame x(i).

Motion and Object-Specific Prompts. Using an LLM, we extract two variants of text prompts to guide motion generation: (i) a motion-specific prompt cmotion that consolidates all motion information without spatial details, and (ii) a set of object-specific prompts clocal = {c(1)local,...,c(localL) }, where each prompt provides details specific to each object’s motion (see Fig. 2 for an example). By assigning each object a constant color in the mask trajectory, we can reliably match each object-specific prompt to its spatial location at any time. Additional prompt generation details are provided in supplementary Sec. 7. Following pre-processing, each data sample consists of the tuple (x,s,c,cmotion,clocal), where x is the ground truth video, s the segmentation, c the initial text prompt, cmotion the motion-specific prompt, and clocal the set of object-specific prompts.

###### 3.2. Image-to-Motion

In the first stage of our framework, we train a model to generate a sequence of fine-grained, mask-based motion trajectories, sˆ, conditioned on an input frame x(0), a segmentation of the input frame s(0), and a motion-specific prompt cmotion. The Image-to-Motion model is denoted as sˆθ(st,t,E(x(0)),E(s(0)),cmotion), where st is a noisy masked-based motion trajectory at the denoising timestep

t, θ are the learned parameters of the network, and E is a VAE [23] encoder. For brevity, we omit the activation of the encoder, E, in the rest of the section. We apply a denoising process in the latent space of a VAE as in LDM [42]. We initialize sˆθ with a pre-trained text-to-video model by concatenating the encodings of the first frame x(0) and its mask s(0) along the input channel dimension. Text conditioning is applied as in LDM, using cross-attention layers. See the supplementary for full details.

###### 3.3. Motion-to-Video

In the second stage, we train a model to generate a sequence of video frames xˆ, conditioned on the reference image x(0), the generated motion trajectory sˆ, the text prompt c, and the object-specific prompts clocal. Using the same denoising approach as in the first stage, we train the Motion-to-Video model, xˆψ(xt,t,x(0),s,c,cˆ local) with parameters ψ, to iteratively refine a noisy latent representation towards a clean video output, following LDM [42]’s formulation. As in the first stage, we finetune a pre-trained text-to-video model, concatenating the encodings of the first frame x(0) and the predicted mask-based motion trajectory sˆ to the noisy latent representation xt along the channel dimension. Text is integrated using cross-attention as in the first stage.

###### 3.3.1. Masked Attention Blocks

We introduce two masked attention-based objectives to condition I2V models in specific areas with targeted information, as shown in Fig. 3. We apply these objectives in the first K blocks to extend the model’s attention capabilities. In the following section, we build upon the notation of [33]. Masked Cross-Attention. We employ masked crossattention to integrate object-specific prompts directly into

| | | | | | |
|---|---|---|---|---|---|
| |[Figure 82]| | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | |[Figure 83]| | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | |[Figure 84]| | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| |[Figure 85]| | | | |
| | | | | | |

[Figure 86]

[Figure 87]

[Figure 88]

| |
|---|
| |
|[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]|
| |
| |

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

- Figure 3. Illustration of the masked attention block. Squares represent video latent patches, color-coded to indicate objects (e.g., cat or dog). Triangles denote prompt tokens: gray for global prompts and object-specific colors for local prompts. The pipeline features self-attention for all patches, masked self-attention restricted to each object, cross-attention integrating global prompts, and masked cross-attention aligning object-specific prompts.

the corresponding regions of the latent space, ensuring each object’s latent representation attends only to its own prompt. Our approach extends that of [33], which considered an object-level cross-attention for text-to-image generation.

′×H′×W′×d be the latent features serving as queries, where N′,H′,W′ are the temporal and spatial dimension of the latent features and d is the model

Formally, let z ∈ RN

dimension size. For L object-specific prompts {c(locali) }Li=1, we encode the prompts to obtain a sequence of embed-

txt×d, where Ntxt is the sequence length of the encoded prompts. We denote the query, key, and value of the masked-cross attention layers as follows q = zWq ∈ RN

dings {e(i)}Li=1 ∈ RN

txt×d, and v(i) = e(i)Wv ∈ RN

tokens×d, k(i) = e(i)Wk ∈ RN

txt×d, where Ntokens = N′ · H′ · W′. All object-specific keys and values are concatenated along the sequence dimension, k = [k(1);...;k(L)] and v = [v(1);...;v(L)]. The masked cross-attention then becomes,

Mcross = [M(1);...;M(l)] (1) hcross = σ

qkT √

+ log Mcross v, (2)

d

where [·;·] is a concatenation along the sequence dimension, σ(·) is the softmax function, and h is the intermediary hidden features passed to the next layer. We construct binary masks M(l) ∈ {0,1}N

tokens indicating the spatial locations associated with each object l along the frames, derived from bounding boxes obtained during training (from ground truth segmentation s) or inference (from generated segmentation sˆ). The masked cross-attention is computed by restricting each query position to attend only to the keys corresponding to objects present at that location.

Masked Self-Attention. Unlike cross-attention, where the queries come from one sequence and the keys and values come from another, self-attention derives the query q, key k, and value v from the same input sequence, which is the latent features z. We introduce a novel objective that ensures that each position attends only to positions of the same object, enhancing temporal consistency and preventing interference between different objects. To this end, we introduce a mask into the self-attention mechanism that restricts attention to positions corresponding to the same object. We construct an attention mask Mself ∈ {0,1}N

tokens×Ntokens, where Mself(i,j) = 1 if positions i and j belong to the same object (based on the segmentation masks sˆ), and Mself(i,j) = 0 otherwise. The masked self-attention then becomes,

hself = σ

qkT √

+ log Mself v. (3)

d

Applying this attention mask we get masked selfattention.

###### 4. Experiments

To evaluate our method, we assess temporal coherence, motion realism, visual consistency across frames, and text faithfulness. First, we compare our approach with current state-of-the-art image-to-video methods on the ImageAnimation-Bench, featuring 2,500 high-quality videos. We use two different neural network architectures for the denoising network: a U-Net, adapted from AnimateDiff [17], and a DiT, adapted from Movie Gen [38]. Following this, we ablate our method’s design. Image-to-video examples are presented in Fig. 4 with additional samples and qualitative comparisons in the supplementary. We additionally introduce a new benchmark, SA-V-128, for image-to-video generation, which includes distinct sets for single-object and multi-object videos. This separation enables focused testing on both scenarios.

###### 4.1. Experimental Setup

Evaluation Benchmarks. To evaluate the effectiveness of our method, we introduce the SA-V-128 benchmark, designed to test performance across both singleand multi-object animations in diverse scenarios. Current

Single-Object Multi-Object Method FVD ↓ CF ↑ ViCLIP-T ↑ ViCLIP-V ↑ AD Text Motion Quality FVD ↓ CF ↑ ViCLIP-T ↑ ViCLIP-V ↑ AD Text Motion Quality

align. consist. align. consist.

VideoCrafter [10] 1484.18 0.966 0.209 0.796 2.93 84.3 84.3 81.2 1413.83 0.966 0.208 0.802 3.75 84.3 87.5 92.1 DynamiCrafter [54] 1442.48 0.942 0.214 0.817 8.94 75.0 81.2 82.8 1300.07 0.947 0.211 0.834 7.56 75.0 73.4 76.5 Motion-I2V [43] 1195.08 0.937 0.220 0.822 6.28 75.0 89.0 93.7 1162.06 0.935 0.219 0.821 6.97 81.0 89.0 95.3 ConsistI2V [41] 1206.61 0.951 0.218 0.839 5.21 65.6 78.1 81.2 1186.10 0.935 0.217 0.850 7.25 81.2 82.8 84.3 TI2V (UNet) 1285.99 0.942 0.219 0.877 5.90 53.1 59.3 70.3 1410.68 0.942 0.218 0.883 7.93 62.5 64.0 60.0 Ours (UNet) 925.39 0.969 0.220 0.888 4.70 - - - 1089.86 0.966 0.220 0.896 5.59 - - -

TI2V (DiT) 1232.89 0.924 0.223 0.797 10.87 65.6 73.4 68.7 1156.82 0.917 0.221 0.805 10.52 64.0 59.3 64.0 Ours (DiT) 1216.83 0.945 0.226 0.860 7.22 - - - 1134.71 0.948 0.225 0.863 7.48 - - -

- Table 1. Results for single-object and multi-object settings on the SA-V-128 Benchmark. We report FVD, CLIPFrame (CF), ViCLIP-T, ViCLIP-V, and Average Displacement (AD), along with human ratings. Human evaluation shows the percentage of raters that prefer the results of THROUGH-THE-MASK.

image-to-video benchmarks lack explicit distinctions between single- and multi-object cases, particularly when assessing “motion-capable objects” such as humans and animals. This limitation hinders accurate evaluation of models in complex multi-object animations. To address this, we constructed a balanced test set of 128 videos from the SAV dataset [40], with equal representation of single-object and multi-object cases (64 videos each), averaging 14 seconds per video. The filtering process consisted of generating text captions and categorizing each video from a set of predefined categories using Llama v3.2-11B [13], based on selected frames. Each video was then assigned aesthetic and motion scores, with motion quantified by optical flow magnitude via RAFT [47]. From this, the 500 videos with the highest combined scores were automatically selected, and 64 single-object and 64 multi-object videos were randomly chosen from this set. We provide further details in the supplementary Sec. 11. Additionally, we use the ImageAnimation-Bench, a curated collection of 2,500 videos, all meeting strict resolution requirements and filtered based on aesthetic standards. Further details are provided in the supplementary Sec. 12. We then evaluate our method’s effectiveness across diverse scenarios using both the ImageAnimation-Bench and the SA-V-128 benchmark.

Evaluation Metrics. The objective of image-to-video generation is to produce videos that are high-quality, temporally consistent, faithful to the input text, and maintain key elements of the initial input image across frames. We assess these aspects using both objective and subjective metrics. For video realism, we employ Fréchet Video Distance (FVD) [48], which measures the visual disparity between feature embeddings of generated and reference videos. To evaluate temporal consistency, we use CLIPFrame (Frame Consistency) [53], which computes the average cosine similarity between CLIP [39] embeddings of individual frames to measure frame-to-frame coherence. To verify text faithfulness, we use ViCLIP-T, a metric based on ViCLIP [52], a video CLIP model that incorporates temporal information when processing videos. ViCLIP-T calculates the cosine similarity between text and video em-

beddings, measuring how well the generated video aligns with the input text prompt. For image faithfulness, we use ViCLIP-V, which, similar to ViCLIP-T, measures cosine similarity between the ViCLIP embeddings of the generated video and a reference video derived from the input image to ensure that the generated video maintains essential visual elements of the initial input image across frames. Given the generation setting of a maximum of 128 frames, this metric ensures relative alignment with the reference video, supporting fidelity to the original input. Furthermore, we report the Average Displacement by taking the average magnitude of the OF vector between consecutive frames to estimate the degree of dynamics. In this metric, we ensure that the videos exhibit realistic motion by maintaining displacement levels that are neither excessively high nor unnaturally low. For subjective evaluation, we rely on human raters to compare our approach against the baselines. We present the raters with the input frame, a caption describing the motion, and two generated videos: one from our method and one from the baseline. The raters are tasked with answering three questions: (i) Text faithfulness: Which video better matches the caption? (ii) Motion: Which video has the best overall motion consistency? and (iii) Quality: Aesthetically, which video is better? To ensure a fair comparison, each pair of videos, along with the input image and text prompt, was rated by 5 different raters. We rate 128 randomly sampled samples from the Image-Animation-Bench and all 128 samples from SA-V-128 benchmark. We then used the majority vote to determine which video was preferred.

###### 4.2. Baseline Comparisons

For comparison with U-Net-based models, we evaluate our method against several open-sourced state-of-the-art imageto-video models: VideoCrafter [9], DynamiCrafter [54], Motion-I2V [43], and ConsistI2V [41].

For both U-Net and DiT models, we also report results for a single-step image-to-video baseline, denoted as TI2V, which is a variant of the proposed architecture with two main differences: (i) TI2V accepts a concatenation of the first frame and input noise as input, without any motion

[Figure 100]

[Figure 101]

‘A dark brown goat wearing a blue shirt walks through a field.’

‘A small white boat cruises through calm waters at sunset.’

Input

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Ours (U-Net)

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

TI2V (U-Net)

[Figure 118]

[Figure 119]

‘Two white puppies are engrossed in eating food off the ground.’

‘A butterfly perches delicately on a vibrant yellow flower.’

Input

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Ours (DiT)

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

TI2V (DiT)

- Figure 4. Qualitative comparison: Visual examples of generated videos for THROUGH-THE-MASK compared to the TI2V baseline on examples from the SA-V-128 benchmark.

Method FVD ↓ CF ↑ ViCLIP-T ↑ ViCLIP-V ↑ AD Text Motion Quality

align. consist.

VideoCrafter [10] 266.83 0.961 0.195 0.810 4.87 78.9 78.1 80.4 DynamiCrafter [54] 217.40 0.946 0.200 0.840 8.28 55.4 53.9 53.9 Motion-I2V [43] 286.42 0.928 0.209 0.746 7.46 81.2 82.8 83.5 ConsistI2V [41] 283.59 0.938 0.202 0.838 6.38 57.0 67.1 65.6 TI2V (UNet) 242.18 0.954 0.203 0.858 5.99 49.2 57.8 66.4 Ours (UNet) 196.23 0.962 0.210 0.865 5.69 - - -

TI2V (DiT) 212.23 0.937 0.206 0.789 9.00 61.7 63.2 67.1 Ours (DiT) 192.45 0.948 0.215 0.847 7.42 - - -

- Table 2. Image-Animation-Bench results. We report FVD, CLIPFrame (CF), ViCLIP-T, ViCLIP-V, and Average Displacement (AD), along with human ratings. Human evaluation shows the percentage of raters that prefer the results of THROUGH-THEMASK.

trajectory representation, and (ii) instead of our proposed masked attention blocks, TI2V includes additional standard attention layers to match the parameter count. Specifically, TI2V includes additional self-attention and cross-attention layers that attend to the full video patches without masking and with text prompt, respectively.

Tab. 1 depicts our comparison on the SA-V-128 benchmark for both single-object and multi-object. As can be seen, the proposed method outperforms all of the evalu-

ated baselines considering both single- and multiple object setups, while holding comparable average displacement. Furthermore, we demonstrate significant enhancements in ViCLIP-V and FVD, underscoring its ability to generate videos that are both image-faithful and highly realistic. Additionally, human raters show a preference for our method over the TI2V baseline. Similarly, in the DiTbased models, THROUGH-THE-MASK exhibits clear superiority over the TI2V baseline across all evaluated metrics in both single- and multi-object settings. Notably, we achieve a significant improvement in ViCLIP-V, demonstrating substantially higher image fidelity.

Next, we report results on Image-Animation-Bench in Tab. 2. Consistent with the findings on the SA-V-128 benchmark, THROUGH-THE-MASK surpasses all baselines on Image-Animation-Bench. The U-Net and DiT variants of THROUGH-THE-MASK achieve lower FVD and higher ViCLIP-T and ViCLIP-V scores. For human evaluation, we randomly sample 128 videos from the benchmark. Human evaluations further support these findings, with THROUGHTHE-MASK receiving the highest preference rates across almost all comparison metrics—winning against all baselines

Config. FVD↓ CF↑ ViCLIP-T↑ ViCLIP-V↑ AD TI2V (UNet) 974.07 0.942 0.218 0.880 6.91 no mask attn (UNet) 972.25 0.962 0.214 0.880 4.99 w. cross-attn (UNet) 670.92 0.965 0.220 0.890 5.25 w. self-attn (UNet) 658.92 0.968 0.218 0.892 5.00 Ours (UNet) 648.59 0.968 0.220 0.892 5.15 TI2V (DiT) 1199.86 0.921 0.222 0.802 10.70 no mask attn (DiT) 1182.49 0.943 0.223 0.851 6.78 w. cross-attn (DiT) 1105.91 0.945 0.226 0.859 7.23 w. self-attn (DiT) 1152.38 0.946 0.223 0.855 7.01 Ours (DiT) 1082.23 0.947 0.226 0.861 7.35

- Table 3. Ablation study results on the SA-V-128 benchmark comparing the performance of different attention configurations, in both U-Net and DiT-based models.

except in text alignment with TI2V, where TI2V slightly outperforms THROUGH-THE-MASK (50.8 vs. 49.2). However, despite this slight outperformance by TI2V in human evaluations, THROUGH-THE-MASK still demonstrates superior text-alignment according to automatic metrics. To further illustrate the results, Fig. 4 presents four visual examples comparing our model against TI2V, showcasing generated video examples across the SA-V-128 benchmark. These examples highlight superiority in motion (top left), aesthetics (top right), text alignment (bottom left), and visual consistency (bottom right). Additional samples and qualitative comparisons are available, comparisons with DiT-based models are shown in Fig.8, while comparisons with U-Net-based models are presented in Fig. 9.

- 4.3. Ablation Study

Effect of The Masked Attention Mechanism. We evaluate the impact of spatial and temporal masked attention layers in THROUGH-THE-MASK. We consider the following configurations: (i) without masked attention (no mask attn), (ii) with only masked cross-attention (w. cross-attn), (iii) with only masked self-attention (w. self-attn), and (iv) with both masked cross and self-attention (THROUGH-THE-MASK). We also compare against the TI2V baseline model that includes additional attention layers positioned as our masked layers, but performs self-attention and cross-attention without masking, ensuring it has the same number of parameters as THROUGH-THE-MASK. Tab. 3 reports the results of the ablation study on the SA-V-128 benchmark using both UNet based and DiT-based models. Results indicate that the addition of masked attention significantly improves performance across all metrics, particularly when compared to the baseline with the same architecture but without masking. A qualitative comparison between the variants evaluated in this ablation study, is available in Fig. 6.

Motion Representation Ablation. Lastly, we compare a mask-based trajectory versus optical flow based motion trajectory. We train separate models for Stage 1 and Stage 2 of THROUGH-THE-MASK for each of the trajectories, keeping all other aspects the same. Tab. 4 reports the results

‘The cartoon superhero raised his right leg to the knee on his left leg and put his hands

[Figure 136]

Input

together as if in prayer.’

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Mask

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Flow

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Mask-based

output

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

Flow-based

output

Figure 5. Qualitative comparison of generated videos using segmentation masks vs optical flow as an intermediate motion representation. The first row shows the input image and text, the second row displays the generated masks, and the third row presents the generated optical flow. The fourth and fifth rows show the generated videos, with the fourth row using our mask-based model and the fifth using our flow-based model.

Configuration FVD↓ CF↑ ViCLIP-T↑ ViCLIP-V↑ AD w. OF 1014.72 0.934 0.219 0.879 7.04 w. Seg. 648.59 0.968 0.220 0.892 5.15

Table 4. Ablation study comparing segmentation masks and optical flow as motion trajectory representation. w. Seg refers to models with segmentation-based motion trajectories, while w. OF denotes models with optical flow-based motion trajectories.

using the U-Net variant of our method. As can be seen, using segmentation masks significantly outperforms the optical flow. We hypothesize that optical flow is too restrictive for this task, as it enforces precise pixel-wise correspondences and tends to collapse, as shown in the bottom right frame of Fig. 5, which provides a qualitative illustration of this ablation. Another qualitative example is shown in Fig.7. This rigidity leads Stage 2 to overly rely on the provided motion, limiting its ability to generate realistic and detailed video content. In contrast, segmentation masks offer a higher-level semantic representation of object motion, providing guidance without constraining the model to exact pixel movements as demonstrated by SpaText [3] (see Sec .4.3.). The latter allows Stage 2 to follow the motion cues while autonomously refining fine details, resulting in more natural and coherent video sequences.

###### 5. Conclusion

We presented THROUGH-THE-MASK, a novel two-stage framework for image-to-video generation leveraging maskbased motion trajectories as an intermediate representation, enabling coherent and realistic multi-object motion in generated videos. Motion trajectories are injected via two attention-based objectives that effectively use this representation to enforce the predicted motion and semantics in the generated video. Our approach empirically achieves SOTA performance in challenging single- and multi-object settings.

###### References

- [1] Michael S Albergo, Nicholas M Boffi, and Eric VandenEijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797,

2023. 2

- [2] Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, Jia-Bin Huang, Jiebo Luo, and Xi Yin. Latent-shift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv preprint arXiv:2304.08477, 2023. 2
- [3] Omri Avrahami, Thomas Hayes, Oran Gafni, Sonal Gupta, Yaniv Taigman, Devi Parikh, Dani Lischinski, Ohad Fried, and Xi Yin. Spatext: Spatio-textual representation for controllable image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18370–18380, 2023. 3, 8
- [4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 2
- [5] Andreas Blattmann, Timo Milbich, Michael Dorkenwald, and Bjorn Ommer. Understanding object dynamics for interactive image-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5171–5181, 2021. 3
- [6] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets, 2023. 2, 3
- [7] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2
- [8] Tim Brooks et al. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 1
- [9] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation, 2023. 3, 6

- [10] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7310– 7320, 2024. 2, 6, 7
- [11] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023. 2
- [12] Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Animateanything: Finegrained open domain image animation with motion guidance, 2023. 2, 3
- [13] Abhimanyu Dubey et al. The llama 3 herd of models, 2024. 6, 1, 2
- [14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 2
- [15] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, MingYu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22930–22941, 2023. 2
- [16] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023. 3
- [17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning, 2024. 2, 5, 1
- [18] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221,

2022. 2

- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 1
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [21] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2
- [22] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 2

- [23] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 4
- [24] Xin Li, Wenqing Chu, Ye Wu, Weihang Yuan, Fanglong Liu, Qi Zhang, Fu Li, Haocheng Feng, Errui Ding, and Jingdong Wang. Videogen: A reference-guided latent diffusion approach for high definition text-to-video generation. arXiv preprint arXiv:2309.00398, 2023. 3
- [25] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation, 2023. 3
- [26] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed, 2024. 1
- [27] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. 2
- [28] Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022. 2
- [29] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection, 2024. 4
- [30] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, YuanFang Li, Cunjian Chen, and Yu Qiao. Cinemo: Consistent and controllable image animation with motion diffusion models, 2024. 2, 3
- [31] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 3
- [32] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, Tsai-Shien Chen, Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. Snap video: Scaled spatiotemporal transformers for text-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7038– 7048, 2024. 3
- [33] Weili Nie, Sifei Liu, Morteza Mardani, Chao Liu, Benjamin Eckart, and Arash Vahdat. Compositional text-to-image generation with dense blob representations, 2024. 3, 4, 5
- [34] OpenAI. Video generation models as world simulators. https : / / openai . com / index / video generation-models-as-world-simulators/,

2024. 3

- [35] Junting Pan, Chengyu Wang, Xu Jia, Jing Shao, Lu Sheng, Junjie Yan, and Xiaogang Wang. Video generation from single semantic label map. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3733–3742, 2019. 3
- [36] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 3

- [37] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and

- Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2
- [38] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dimitry Vengertsev, Edgar Schonfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie gen: A cast of media foundation models, 2024. 3, 5, 1
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 6
- [40] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos,

2024. 4, 6, 1, 2

- [41] Weiming Ren, Huan Yang, Ge Zhang, Cong Wei, Xinrun Du, Wenhao Huang, and Wenhu Chen. Consisti2v: Enhancing visual consistency for image-to-video generation, 2024. 2, 3, 6, 7
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 4
- [43] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling, 2024. 2, 3, 6, 7
- [44] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data, 2022. 2

- [45] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2
- [46] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2
- [47] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow, 2020. 6, 2
- [48] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges, 2019. 6
- [49] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 2
- [50] Wenjing Wang, Huan Yang, Zixi Tuo, Huiguo He, Junchen Zhu, Jianlong Fu, and Jiaying Liu. Videofactory: Swap attention in spatiotemporal diffusions for text-to-video generation. arXiv preprint arXiv:2305.10874, 2023.
- [51] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 2
- [52] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, Conghui He, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Yu Qiao. Internvid: A large-scale video-text dataset for multimodal understanding and generation, 2024. 6
- [53] Jay Zhangjie Wu, Xiuyu Li, Difei Gao, Zhen Dong, Jinbin Bai, Aishani Singh, Xiaoyu Xiang, Youzeng Li, Zuwei Huang, Yuanxi Sun, Rui He, Feng Hu, Junhua Hu, Hai Huang, Hanyu Zhu, Xu Cheng, Jie Tang, Mike Zheng Shou, Kurt Keutzer, and Forrest Iandola. Cvpr 2023 text guided video editing competition, 2023. 6
- [54] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors, 2023. 3, 6, 7
- [55] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models, 2023. 2, 3
- [56] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 2

###### Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation

###### Supplementary Material

###### 6. Additional Results

- 6.1. Qualitative Comparison of Masked Attention Mechanism

- Fig. 6 shows qualitative comparison of generated videos for each configuration of THROUGH-THE-MASK, demonstrating the differences when applying masked cross-attention, self-attention, both, or no masked attention layers.

6.2. Qualitative Comparison of Motion Representation Ablation

- Fig. 7 shows a qualitative comparison of the generated videos for different intermediate representation configurations of THROUGH-THE-MASK. Specifically, it compares our chosen representation, which is mask-based motion trajectories, to optical flow.

- 6.3. Additional Qualitative Comparisons of DiT Architecture

Building upon the comparisons presented in Sec. 4.2, we provide further qualitative results comparing our approach to existing baselines, based on DiT architecture, in Fig. 8

- 6.4. Additional Qualitative Comparisons of U-Net Architecture

Building upon the comparisons presented in Sec. 4.2, we provide further qualitative results comparing our approach to existing baselines, based on U-Net architecture, in Fig. 9

###### 7. Motion and Object-Specific Prompts Details

As described in Sec. 3.1, our pre-processing pipeline extracts a motion-specific prompt, cmotion, from the input text c, using a pre-trained LLM. This prompt provides a consolidated description of all motion in the scene, excluding any spatial, color, or object-specific details, and serves as a high-level guide for motion generation.

To generate the motion-specific prompt, we use Llama v3.1-8B [13] in a frozen configuration. The input prompt instructs the LLM to focus solely on motion, as shown in Fig. 10, ensuring that descriptions remain centered on movement dynamics, ignoring background information and visual characteristics of objects.

###### 8. Motion-capable Objects’ Prompt Extraction Details

As described in Sec. 3.1, the pre-processing process begins with extracting motion-capable object prompts from the global prompt c. We utilize Llama v3.1-8B [13] as a frozen LLM and provide the prompt shown in Fig. 11, which outlines the process for generating local prompts for motion-capable objects.

###### 9. Inference

Given the reference image x(0) and text prompt c, inference is carried out in two stages. First, the initial segmentation s(0) is extracted from x(0) using SAM2 [40]. Concurrently, the text prompt c is processed by a pre-trained LLM to obtain the motion-specific prompt cmotion and objectspecific prompts clocal = {c(1)local,...,c(localL) } as detailed in Section 3.1. At stage 1, the image-to-motion generates motion trajectories sˆ conditioned on (s(0),x(0),cmotion) . Next, in stage 2, the motion-to-video produces the final video xˆ by conditioning on (x(0),s,c,cˆ local) and incorporating masked attention mechanisms to ensure consistency and controllability, as described in Section 3.3. For both stages, we adapt the Classifier-Free Guidance [19] approach suggested by Brooks et al. [8], where, to align exactly with their method, we treat the concatenated visual conditions as a single visual condition, and do the same for the text.

###### 10. Implementation Details

As detailed above, we demonstrate the applicability of our approach to two architectures.

The first is the U-Net architecture. We follow the AnimateDiff V3 [17] design, consisting of approximately 1.4B parameters. In the second stage of motion-to-video, detailed in Sec. 3.3, we set K = 6, where K represents the number of attention blocks expanded into masked attention blocks—specifically, by adding masked self-attention and masked cross-attention into the spatial attention blocks within the U-Net’s encoder layers. The U-Net-based model was optimized using the solver suggested by [26], incorporating the DDIM diffusion solver with v-prediction and zero signal-to-noise ratio (SNR). The latter was found to be critically important to enable image-to-mask-based motion trajectory generation.

The second architecture is DiT-based. We train a DiT model following the MovieGen [38] design, containing four

billion parameters. For the DiT-based model in stage two, we used K = 10, corresponding to the first 10 attention blocks out of a total of 40. The DiT-based model was optimized as described in the MovieGen paper, with Flow Matching [27], using a first-order Euler ODE solver. During inference, we adopted MovieGen’s efficient inference method by combining a linear-quadratic t-schedule, as detailed in the MovieGen paper.

For both architectures, text-to-video pre-training followed the methodology outlined in MovieGen. Across both training stages (Sec. 3.2 and Sec. 3.3), we utilized the finegrained mask-based motion trajectories dataset described in Sec. 3.1. The U-Net model was trained at a resolution of 512 × 512, predicting 16 frames, while the DiT model was trained at a resolution of 256 × 256, predicting 128 frames. Both models were trained with a batch size of 32, using a constant learning rate of 2 × 10−5, a warm-up period of 2000 steps, and a total of 50,000 steps.

###### 11. SA-V-128 Benchmark

We introduce a balanced test set of 128 videos from the SAV dataset [40], comprising 64 single-object and 64 multiobject cases, with an average duration of 14 seconds per video. The filtering of 128 videos, out of the full SA-V dataset, involved several steps. First, for each video, we generated a text caption using Llama v3.2-11B [13] by providing the first, middle, and last frames and asking the model to generate a caption describing the video. Next, from a closed set of categories (Animal, Architecture, Digital Art, Food, Landscape, Lifestyle, Plant, Vehicles, Visual Art, and Other), we used Llama v3.2-11B [13] to categorize each video based on these frames. We then iterated over the categories, selecting a unique category at each step and adding a related video to ensure a balanced test set. We assigned an aesthetic score and a motion score by calculating the magnitude of the optical flow extracted with RAFT [47]. After assigning captions and scores, we filtered 500 videos by iterating through each category and selecting those with the highest combined aesthetic and motion scores. From these 500 automatically filtered videos, we randomly selected 64 single-object and 64 multi-object videos. To ensure a fair comparison for shorter video settings, we also provided short captions, generated using the same methodology, extracted from frames 0 to 127 of each video.

###### 12. Image-Animation-Bench

The Image-Animation-Bench comprises 2,500 videos, meticulously curated to meet high-resolution requirements and aesthetic quality thresholds. To ensure comprehensive coverage of diverse visual scenarios, the dataset is divided into 16 categories: Portraits, Scenery-Nature, Pets, Food, Animation, Science, Sports, Scenery-City, Animation-

Static, Music, Game, Animals, Industry, Painting, Vehicles, and others.

[Figure 153]

#### ‘The cartoon superhero raised his left leg to the knee on his right leg and put his hands together as if in prayer.’

Input

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Mask

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

No mask attn

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

With selfattn

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

With crossattn

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

###### Ours

- Figure 6. Qualitative comparison of generated videos for each configuration of THROUGH-THE-MASK. The results highlight differences when applying masked cross-attention (With cross-attn), self-attention (With self-attn), both (Ours), or no masked attention layers (No mask attn). Without masked attention, the cartoon superhero fails to perform a prayer. With masked self-attention, the superhero also fails, but the movement appears smoother and more consistent. With masked cross-attention, the superhero successfully performs the prayer, though his fingers turn blue. When integrating the full masked attention mechanism, the superhero performs the action correctly.

[Figure 179]

# ‘The jeep rumbles forward on the winding mountain road.’

Input

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Mask

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Flow

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Mask-based

output

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Flow-based

output

- Figure 7. Qualitative comparison of generated videos using segmentation masks vs optical flow as an intermediate motion representation. The first row shows the input image and text, the second row displays the generated masks, and the third row presents the generated optical flow. The fourth and fifth rows show the generated videos, with the fourth row using our mask-based model and the fifth using our flowbased model.

[Figure 196]

### Input ‘The tank rolls forward slowly across a dusty field.’

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Ours

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

TI2V

[Figure 207]

### Input ‘Two cockatoos, one spinning in mid-air while the other does a wing flap.’

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

Ours

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

##### TI2V

- Figure 8. Qualitative comparison of video generations produced by THROUGH-THE-MASK (DiT-based) and the TI2V baseline (DiTbased).

[Figure 218]

### ‘A happy penguin rolling along on bright pink roller skates, wings flapping to keep steady as it moves.’

Input

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Ours

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

TI2V

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

ConsistI2V

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Motion-

I2V

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

Dynami Crafter

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

Video Crafter

- Figure 9. Qualitative comparison of video generations produced by THROUGH-THE-MASK (U-Net-based) and TI2V (U-Net-based), ConsistI2V, Motion-I2V, DynamiCrafter, and VideoCrafter.

Task: Extract a single motion-specific prompt from the caption that describes the overall motion without including any spatial, color, size, or background details.

Format your answer like this: Motion-specific prompt: "description of overall motion"

Examples: Caption: "A large, red ball rolls to the right on a grassy field while a small, blue kite flies upward in the clear, blue sky." Motion-specific prompt: "The ball rolls to the right, and the kite flies upward." Caption: "A sleek, black car drives down a busy city street with tall buildings in the background as several pedestrians wearing bright clothing cross." Motion-specific prompt: "The car drives down the street as pedestrians cross." Caption: "A fluffy, white cat jumps onto a wooden table set against a plain, beige wall and knocks over a glass of water, spilling it onto the floor." Motion-specific prompt: "The cat jumps onto the table and knocks over the glass." Now, please provide the answer. Caption: "{global_prompt}" Motion-specific prompt:

- Figure 10. The input prompt used for extracting a motion-specific description from the global prompt c, designed for use with a pretrained LLM. The prompt focuses solely on describing the overall motion, explicitly excluding details such as sizes, colors, or background elements. Here, c refers to the global prompt, which is inserted in place of {global_prompt}.

Task: For each object mentioned in the caption, write a local prompt that describes everything about that object as mentioned in the caption.

Format your answer like this: Answer: [[Object 1: description of object 1] [Object 2: description of object 2] ...]

Examples: Caption: "An alien rides a horse through a field." Answer: [[alien: A alien rides a horse through a field.]

[horse: A horse is being ridden through a field.]]

Caption: "A dog chases a ball while a robot runs after it." Answer: [[dog: A dog chases a ball.]

[ball: A ball is being chased by a dog.] [child: A robot runs after it.]]

Caption: "An eagle flies above the mountains." Answer: [[eagle: The eagle flies above the mountains.]]

Caption: "Two playful dogs run along the beach, with one dog on the left and the other in the middle of the frame, as waves crash onto the shore." Answer: [[left dog: The dog runs playfully along the beach, staying closer to the dry sand.] [middle dog: The dog runs beside its companion, edging nearer to the waves.]]

Caption: "Three cats sit in a row on a sunny windowsill, all basking in the warm sunlight, when the cat on the right starts to move his paw."

Answer: [[left cat: The cat sits on the windowsill, soaking in the sunlight.] [middle cat: The cat sits on the windowsill, soaking in the sunlight.] [right cat: The cat sits on the windowsill, then starts to move his paw.]]

Caption: "A bustling farmers’ market filled with a variety of colorful fruit stands, where a monkey is carefully picking ripe, red tomatoes while a street musician plays lively tunes on an acoustic guitar, adding a vibrant atmosphere to the scene." Answer: [[monkey: A monkey carefully picks ripe, red tomatoes from one of the stands.]

[musician: A street musician plays lively tunes on an acoustic guitar]] Now, please provide the answer. Caption: "{global_prompt}" Answer:

- Figure 11. The input prompt used for extracting motion-capable object descriptions from the global prompt c, designed for use with a pre-trained LLM. Here, c refers to the global prompt, which is inserted in place of {global_prompt}.

