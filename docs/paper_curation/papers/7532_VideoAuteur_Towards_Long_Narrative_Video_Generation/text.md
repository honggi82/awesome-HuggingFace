# VideoAuteur: Towards Long Narrative Video Generation

Junfei Xiao1, Feng Cheng2, Lu Qi2, Liangke Gui2, Jiepeng Cen2, Zhibei Ma2, Alan Yuille1, Lu Jiang2 1Johns Hopkins University 2ByteDance Project Page: https://videoauteur.github.io

|[Figure 1]| |
|---|---|
| | |

|[Figure 2]| |
|---|---|
| | |

|[Figure 3]|
|---|

|[Figure 4]| |
|---|---|
| | |

|[Figure 5]|
|---|

|[Figure 6]| |
|---|---|
| | |

|[Figure 7]|
|---|

|[Figure 8]|
|---|

How to cook buffalo chicken dip?

arXiv:2501.06173v2[cs.CV]7Jun2025

Action: Explain something related to food in the bow …

Action: Pour creamy

Action: Use a spoon to spread the cheese evenly across …

Action: Pour sour cream over the baked dish …

Action: Hold a

Action: Use a spatula to serve a baked dish …

Action: Hold a spatula and prepare food on a plate …

Action: Arrange the chicken on top of the vegetable…

white sauce, use a spoon to spread …

spatula and is seen stirring or mixing …

[Figure 9]

Narrative Director

Caption: A man in a black t-shirt stand in a modern kitchen…

Caption: A person is pouring a creamy white sauce from …

Caption: A person is preparing a dish in a black pan …

Caption: A person is pouring a generous amount of sour cream.

Caption: A man in a black t-shirt stand in a kitchen …

Caption: A man in a black t-shirt stand in a kitchen …

Caption: A man in a black t-shirt stand in a kitchen …

Caption: A man in a black t-shirt is preparing a dish in …

|[Figure 10]| |
|---|---|
| | |

|[Figure 11]| |
|---|---|
| | |

|[Figure 12]|
|---|

|[Figure 13]| |
|---|---|
| | |

|[Figure 14]|
|---|

|[Figure 15]| |
|---|---|
| | |

|[Figure 16]|
|---|

|[Figure 17]|
|---|

###### How to cook blueberry muffins?

Action: Mix ingredients in a large glass bowl …

Action: Transfer batter into muffin cups with a tray …

Action: Check muffin doneness with a toothpick …

Action: Display muffins on a cooling rack

Action: Pour liquid ingredient and mix …

Action: Mix batter with honey with …

Action: Mix batter with blueberries …

Action: Place muffins in oven …

Caption: A woman… mixes a batter with honey …

Caption: A woman… mixes a batter with blueberries…

Caption: A woman uses a toothpick to check the doneness…

Caption: A batch of freshly baked blueberry muffins …

Caption: A woman wearing a pink and white striped shirt …

Caption: A woman … filling cups with batter and evenly pressing…

Caption: A woman ... places a tray of blueberry muffins …

Caption: A woman wearing a pink and white striped apron …

- Figure 1. Long Narrative Video Generation. We curate a large-scale cooking video dataset to develop an interleaved auto-regressive model – VideoAuteur, which acts as a narrative director, sequentially generating actions, captions, and keyframes (two generated examples here). These elements condition a video generation model to create long narrative videos.

#### Abstract

#### 1. Introduction

Video generation [5, 6, 19, 20, 41, 51, 62] has witnessed remarkable advancements with diffusion [2, 21, 34, 57] and auto-regressive models [25, 43, 44, 53]. A primary objective is to generate video clips from text prompts and supports various downstream applications, such as image animation [8, 54], video editing [4, 11], video stylization [23].

Recent video generation models have shown promising results in producing high-quality video clips lasting several seconds. However, these models face challenges in generating long sequences that convey clear and informative events, limiting their ability to support coherent narrations. In this paper, we present a large-scale cooking video dataset designed to advance long-form narrative generation in the cooking domain. We validate the quality of our proposed dataset in terms of visual fidelity and textual caption accuracy using state-of-the-art Vision-Language Models (VLMs) and video generation models, respectively. We further introduce a Long Narrative Video Director to enhance both visual and semantic coherence in generated videos and emphasize the role of aligning visual embeddings to achieve improved overall video quality. Our method demonstrates substantial improvements in generating visually detailed and semantically aligned keyframes, supported by finetuning techniques that integrate text and image embeddings within the video generation process.

With the maturity of generating high-fidelity short video clips, researchers begin setting their sights on the next north-star: creating videos capable of conveying a complete narrative which captures an account of events unfolding over time. The importance of narratives has been highlighted in the literature. For example, Bruner argues that narratives are essential tools for organizing experiences and memories [3]. The book Sapiens: A Brief History of Humankind emphasizes that the ability to share narratives (stories) has been pivotal in human development, setting humans apart from other animals [15].

Long Narrative Video Generation (NVG) introduces several challenges. One particularly challenge is the scarcity of video data suitable for learning coherent narratives in video. While our community has developed many

video datasets, most are unsuitable for NVG. First, most videos are tagged with descriptions that are partially to NVG. Second, even for the relevant descriptions, these descriptions may be either too coarse or lack detailed actions needed for NVG. Finally, not all videos contain meaningful narratives suitable for learning and can be well evaluated.

Consequently, video data with clear, complete, and meaningful narratives is crucial not only for training but also for evaluating and comparing NVG methods. However, compared to story generation through a sequence of images [14, 24, 32, 56], progress in narrative video generation has been relatively slow, partly due to the absence of standardized training and evaluation benchmarks.

This paper contributes to advancing research in narrative video generation in two ways. First, we curate and annotate a large-scale video dataset on the cooking domain. The samples in our dataset are structured with clear narrative flows, each composed of sequential actions and visual states. Our dataset consists of approximately 200,000 video clips, with an average duration of 9.5 seconds per clip. We select cooking videos for their well-defined and less ambiguous narratives, making them more objective to evaluate consistently. To address video copyright concerns, we source videos from existing video datasets, YouCook2 [63] and HowTo100M [33]. We design various mechanisms to ensure high-quality videos and captions, organized in a structured storyboard format, as illustrated in Figure 1.

Additionally, we propose a new auto-regressive pipeline for long narrative video generation, comprising three main components: a long narrative director, a rolling-context conditioned keyframe renderer, and a visual-conditioned video generation model. The long narrative director produces a coherent narrative flow by generating a sequence of visual embeddings or keyframes that represent the story’s logical progression. Building upon this, the rolling-context conditioned keyframe renderer utilizes a rolling history of reference images as contextual conditioning to generate high-quality keyframes with consistency. Finally, the visual-conditioned video generation model produces video clips based on these visual conditions to do narrative.

Extensive experiments on the large-scale collected dataset demonstrate the effectiveness of the proposed pipeline for long narrative video generation. To sum up, our contributions are as follows:

- • We construct CookGen, a large-scale, structured dataset accompanied by an effective data pipeline to benchmark long-form narrative video generation. The dataset along with the necessary functionalities will be opensourced to advance future research in the area.
- • We propose VideoAuteur, a novel approach for automated long video generation. It effectively bridges interleaved auto-regressive multimodal LLMs with pretrained DiTs, employing a rolling context strategy for enhanced

generation quality and visual consistency.

• Extensive experimental results and ablation studies show that VideoAuteur achieves the state-of-the-art performance in long narrative video generation.

#### 2. Related Works

Text-to-Image/Video Generation Text-to-image [7, 26, 35, 36, 38, 50, 58] and video generation [5, 6, 19, 20, 41, 51, 62] have made remarkable progress to generate highfidelity video clip of 5-10 seconds. For example, latent design [38] has become mainstream, balancing effectiveness with efficiency. Building upon this design, diffusion-based models like DiT [34], Sora [2], and CogVideo [21, 57] leveraged larger datasets and explored refined architectures and loss functions to enhance performance. In contrast, auto-regressive models such as VideoPoet [25] and Emu series [43, 44, 53] sequentially predict image or video tokens. Instead, our work focuses on the model’s ability to generate long narrative videos beyond a few seconds.

Interleaved Image-Text Modeling Interleaved image-text generation [1, 9, 13, 13, 45, 55] has garnered attention as a compelling research area that merges visual and textual modalities to produce rich outputs. Earlier approaches [29, 37, 37, 42] primarily relied on large-scale image-text paired datasets [12, 39] but were often confined to single-modality tasks, such as captioning or text-to-image generation. With the emergence of large language models [47], various vision-language models [28, 31, 52] have stepped in a new era of unified representations, leveraging well-curated datasets for interleaved generation. However, most existing works focus on the one-time generation and do not address the coherence of generated content, which is our focus.

Narrative Visual Generation Existing narrative visual generation primarily focuses on addressing challenges related to semantic and visual consistency. Recent approaches such as Narrative Visual Generation, VideoDirectorGPT [30], Vlogger [65], Animate-a-story [16], VideoTeris [46], IC Lora [22], Vlogger [65], and Animate-astory [16] employ various methods to enhance semantic coherence and visual continuity. Unlike most prior methods that mainly focus on consistent image generation [22, 56, 64], our target is generating coherent narrative videos. While some works make efforts to be language-centric using text as conditions for video generation [54, 65] or appending with keyframes [61], different from these work, we propose an integrated approach that leverages multimodal large language models (LLMs) in conjunction with in-context diffusion transformer models to ensure global narrative coherence, subsequently conditioning the video generation model.

Action: Take rolling pin and wrap fondant ..

Action: Cut fondant circles with cutter ..

[Figure 18]

[Figure 19]

###### Caption:

Caption:

A woman … is rolling out red fondant on a wooden board... uses a wooden rolling pin to flatten … dusted flour

... preparing dough on a wooden surface dusted with flour … uses a metal cutter to create circular .. dips it into .. sugar …

01:27 03:14

### Act.

[Figure 20]

[Figure 21]

… …

00:25 02:01

[Figure 22]

Action: Apply buttercream frosting on edge

Action: Use hands to smooth fondant …

[Figure 23]

[Figure 24]

Caption: A person wearing a festive sweater with 'HAPPY HOLIDAYS’ …cake on a metal stand…hands are steady…

Caption: … She uses a pair of scissors to cut a piece of red fabric…The cake is red with white icing… adjusts the fabric …

- Figure 2. CookGen contains long narrative videos annotated with actions and captions. Each source video is cut into clips and matched with the labeled “actions”. We use refined pseudo labels from ASR for Howto100M videos and use manual annotations for Youcook2 videos. We use state-of-the-art VLMs (i.e. GPT-4o and an expert captioner) to provide high-quality captions for all video clips.
- 3. CookGen: a Long Narrative Video Dataset

Data Source # Vid. (train/val) # Clips Clip Len. # Clips / Vid.

YouCook2 1333 / 457 ∼10K 19.6s 7.7 HowTo100M (subset) 30039 / 933 ∼183K 9.5s 5.9

To the best of our knowledge, datasets for long narrative video generation research is extremely limited. To enable in-depth exploration and establish an experimental setting, we establish CookGen, a large video dataset with detailed annotations on captions, actions, and annotations. As the data example provided in Figure 2, our dataset focuses on cooking videos. We prioritize cooking over other video categories because each dish follows a pre-defined, strict sequence of action steps. These structured and unambiguous objectives in cooking videos are essential for learning and evaluating long video narrative generation.

- Table 1. Long narrative dataset sources. Our dataset is built upon Youcook2 and a cooking subset of Howto100M.

Datasets Modality Type # Images Text Length Flintstones Image Comic 122k 86

Pororo Image Comic 74k 74

StorySalon Image Comic 160k 106 StoryStream Image Comic 258k 146

VIST Image Real world 210K ∼70

CookGen Video Real world 39M 763.8

- Table 2. Comparison with multi-modal narrative datasets. Most existing datasets focus on image-based comic story generation. In contrast, our dataset consists of long narrative videos, containing 150× the number of frames and 5× the dense text annotations compared to the previous largest dataset, StoryStream.
- 3.2. Annotation and Processing

###### 3.1. Overview

We source over 30,000 raw videos about from two existing video datasets: YouCook2 [63] and HowTo100M [33]. Each video is filtered and cropped with processing to remove corruptions. Table 1 provides detailed information about the dataset statistics, video and clip details, and the train/val partitioning. Appendix B provides more details.

To ensure scalability and quality, we design an efficient annotation pipeline to support the annotation as below.

Captions. For open-source and scalability, we train a video captioner based on open-sourced VLM. Inspired by LLaVA-Hound [59], we begin by collecting a caption dataset using GPT-4o, with a focus on object attributes, subject-object interactions, and temporal dynamics. Subsequently, we fine-tune a captioning model based on LLaVANeXT [60] to optimize captioning performance.

Table 2 compares our dataset with existing datasets most relevant to multimodal narrative generation. Unlike existing datasets that primarily focus on image-based comic story generation, our real-world narrative dataset offers several advantages. First, the videos in our dataset depict procedural activities (i.e., cooking), providing unambiguous narratives that are easier to annotate and evaluate. Second, our dataset contains 150× the number of frames compared to the previous largest dataset, StoryStream. Third, we offer 5× denser textual descriptions, with an average of 763.8 words per video. These advantages make our dataset a better resource for narrative video generation.

Actions. We use HowTo100M ASR-based pseudo labels for ‘actions’ in each video, further refined by LLMs to provide enhanced annotations of the actions throughout the video [40]. This refinement improves the action quality to capture events and narrative context. However, the annotations are still noisy and sometimes not informative due to

the inherent errors in ASR scripts.

Caption-Action Matching and Filtering. To ensure alignment between captions and actions, we implement a matching process based on time intervals. Using Intersectionover-Union (IoU) as a metric, we evaluate whether the overlap between the captioned clip time and action time meets a threshold. An action is considered a match if the following conditions are met: the difference between the clip start time and the action start time (start diff) is less than 5 seconds; the clip end time is later than the action end time; and the IoU between the clip and action time intervals is greater than 0.25, or if IoU>0.5. Here, clip time and action time represent the time intervals for the clip and action, respectively. Using this rule, we filter and match captions to actions, ensuring that each caption aligns with the relevant action. We found this step is important for creating narrative consistency throughout the video.

Annotation Quality Reverification. High-quality captions are essential for narrative visual generation. To verify the quality of our annotations, we build an evaluation pipline of inverse generation and visual understanding through VLM experts, which are detailed in Appendix §C.1 and §C.2.

#### 4. Method

Given the text input, the task of long narrative video generation aims at generating a coherent long video Y ∈ RH×W×F that aligns with the progression of the text input sequentially. The H, W, and F are generated videos’ height, width, and frame numbers. To achieve this, we propose VideoAuteur, which involves three main components: an interleaved long narrative video director, a rolling-context conditioned keyframe renderer, and a visualconditioned video generation model. The long narrative video director creates a sequence of language states and visual embeddings to represent the narrative flow (§4.1). A pretrained DiT model then renders keyframes using a rolling history of reference images as contextual conditioning (§4.2). Finally, the video generation model produces video clips based on these visual conditions (§4.3).

###### 4.1. Long Narrative Interleaved Director

As shown in Figure 3a, the long narrative video director generates a sequence of visual embeddings (or keyframes) that capture the narrative flow. The interleaved image-text director creates a sequence where text tokens and visual embeddings are interleaved, integrating narrative and visual content tightly. Using an auto-regressive model, it predicts the next token based on the accumulated context of both text and images. This helps maintain narrative coherence and align visuals with the text semantics.

Interleaved auto-regressive model. Our model performs next-token prediction for cross-modal generation, learning

from sequences of interleaved image-text pairs with a context window size T. Each text token is supervised with cross-entropy loss, and the final visual embedding zT is regressed using learnable query tokens, as illustrated in Figure 3b. The auto-regressive conditioning is given by:

p(yt | y1:t−1) = p(ct | c1:t−1) · p(zt | c1:t,z1:t−1), (1)

where ct represents texts and zt denotes visual embeddings. Regression latent space. We utilize a CLIP-Diffusion visual autoencoder with a CLIP encoder Eclip and a diffusion decoder Ddiff to encode raw images x to visual embeddings for auto-regressive generation:

###### z = Eclip(x), xˆ = Ddiff(z) (2)

This setup generates language-aligned visual embeddings and reconstructs images from them. Regression loss. To align the generated visual latents zpred with the target latents ztarget, we use a combined loss:

N

zpred · ztarget ∥zpred∥∥ztarget∥

1 N

(ˆzi − zi)2

Lreg = α 1 −

+ β

i=1

(3) where α and β are hyper-parameters.

Narrative from “actions” to “visual states”. The interleaved model generates a coherent narrative sequence by progressively conditioning each step on the cumulative context from previous steps, Figure 3b. At each time step t, the model generates an action at, a caption ct, and a visual state zt, conditioned on the cumulative history Ht−1:

Ht−1 = {a1:t−1,c1:t−1,z1:t−1} at | Ht−1 → ct | {Ht−1,at} → zt | {Ht−1,at,ct}

(4)

This layered conditioning improves coherence across the sequence, aligning actions, language, and visuals.

###### 4.2. Rolling Context Conditioned Render

While the interleaved auto-regressive director model can learn visual consistency, the CLIP representation space struggles to preserve fine visual details (e.g., character features, clothing patterns), as demonstrated in Appendix Figure 16. To address this limitation and improve generation quality, we employ a pretrained Text-to-Image diffusion transformer model to render high-quality keyframes, conditioning on a rolling history of reference images. The context length can vary dynamically from 1 to 3, balancing flexibility and efficiency when generating keyframes.

As illustrated in Figure 3b, we use a rolling history of two reference images, It and It−1. This setup is further conditioned by the tiled global caption

ctiled = tiled(ct−3, ct−2, ct−1, ct), (5)

(a) Interleaved Auto-regressive Director (b) Rolling Context Conditioned Render

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Flow Matching Loss

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Cross Entropy Loss No Loss Cross Entropy Loss No Loss Cross Entropy Loss Reg. Loss

V. Emb. …

MM-DiT Blocks

###### Act. Cap. Act. Cap.

V. Emb.

Act. Cap. V. Emb.

…

[Figure 37]

[Figure 38]

[Figure 39]

Interleaved Auto-regressive Director

MM-DiT Blocks

|[Figure 40]|
|---|

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

…

###### …

Act. Cap. V. Emb. Act. Cap. V. Emb. Act. Cap. Queries

V. Embed.

Text Embed.

Start with fresh tuna.

Cutting tuna into chunks.

The burger is cooking nicely. A blue frying pan sits on a stovetop, containing onions and a golden-brown patty … CLIP Visual Embeddings

Tiled Global Caption

Patchified VAE Latents & Noisy Latents

A raw piece of red tuna steak is placed on a wooden board. To the left of the tuna …

A man with a ring on his left hand is slicing raw tuna on a wooden cutting board…

| |[Figure 46]|[Figure 47]| |[Figure 48]|[Figure 49]|
|---|---|---|---|---|---|
| | | | | | |

To cook ``Banana Nutella Chocolate Milkshake” …

…

- [SCENE-1] A woman places a blender on counter …

- [SCENE-2] …uses a piping bag to add whipped cream

- [SCENE-3] … decorating three chocolate milkshakes

- [SCENE-4] The woman is presenting milkshakes …

CLIP Visual Embeddings

CLIP Visual Embeddings

[Figure 50]

[Figure 51]

[Figure 52]

Sample

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

Step 1: Step 2: Step N:

Not Masked Optional Mask Masked

Figure 3. Long Narrative Visual Condition Generation. (a) Interleaved Auto-regressive Director: an auto-regressive vison-language model, takes a user query (e.g., “How to cook a tuna sandwich?”) and an initial image-text pair as input. It then generates actions, captions, and visual states (i.e., visual embeddings) step-by-step. (b) Rolling Context Conditioned Render: Apart from the semantics consistency through interleaved generation, we use a rolling of reference images as direct context conditions to further improve visual consistency with a diffusion transformer model. With them, a long narrative video can be created using these generated visual conditions (i.e., visual embeddings and/or keyframes derived from the interleaved director and the keyframe render with rolling context conditioning.)

the predicted visual embeddings zt and zt−1, as well as the reference images It−3 and It−2.

Visual-Conditioned Video Generation

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Self-Attention ×N

Cross-Attention

+ Regularization

D(ctiled, zt−1, zt, It−3, It−2) → It−1,It, (6)

|Caption Keyframe VAE<br><br>|
|---|

Caption Visual Embed.

where D(·) denotes the diffusion model for synthesizing a new keyframe It by integrating the rolling context of images, captions, and visual embeddings. This layered conditioning improves coherence across frames.

[Figure 62]

[Figure 63]

Keyframe Condition

Visual Embed. Condition

Visual Embedding Regularization

Visual Embeds. Noise Mask Shuffle

[Figure 64]

Flow Matching Loss. We employ a flow matching loss that aligns the learned drift function fθ with the ground-truth path from xT to xT+1. We define:

|[Figure 65]|
|---|

|1|
|---|

|2|
|---|

|3|
|---|

|4|
|---|

|[Figure 66]|
|---|

|M|
|---|

|4|
|---|

|2|
|---|

|3|
|---|

|1|
|---|

Figure 4. Visual-conditioned video generation. Our interleaved auto-regressive director and rolling context renderer generates both text and visual conditions, enabling the video generation process to be conditioned on keyframes (VAE embeddings) and CLIP latents. We apply Gaussian noise, random masking and random shuffling as regularization during the training process to improve robustness with the imperfect visual embeddings.

T,xT+1, T fθ(xT,T) − v(T) 2 , (7)

Lflow(θ) = Ex

where v(T) denotes the ideal drift path that transitions xT towards xT+1. This objective enforces consistency across frames without relying on a separate diffusion loss.

###### 4.3. Visual-Conditioned Video Generation

Using the sequence of actions at, captions ct,visual states zt and keyframe It generated by the interleaved director and rolling context conditioned render, we condition a video generation model to produce coherent long narrative videos. Unlike the classic Image-to-Video (I2V) pipeline that only uses an image as the starting frame, our approach leverages the regressed visual latents zt as continuous conditions throughout the sequence (see §4.3.1). Furthermore, we improve the robustness and quality of the generated videos by adapting the model to handle noisy visual embeddings, since the regressed visual latents may not be perfect due to regression errors and keyframe uncertainty (see §4.3.2).

###### 4.3.1 Visual Conditions Beyond Keyframes

Conventional visual-conditioned video generation typically uses initial keyframes to guide the model, where each frame xt is generated as xt = Dvisual(It). Our interleaved autoregressive director supports generating visual states zt in a semantically aligned latent space, allowing direct conditioning from a pretrained visual decoder, as shown in Figure 4. By using these regressed visual latents zt directly, each frame is generated as xt = Dvisual(zt). This follows the narrative and enhancing consistency by relying on narrative-aligned embeddings.

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Figure 5. Rolling Context Conditioned Render. We integrate tiled global captions, predicted visual embeddings, and a rolling context of previous keyframes to render new keyframes throughout the narrative. By combining semantic conditioning from textual captions and CLIP embeddings with detailed information from VAE embeddings, the diffusion transformer maintains consistency in visual details such as clothing, food details, and character identities. Generated frames are highlighted with red edges.

- 4.3.2 Learning from Noisy Visual Conditions

To enhance the robustness over imperfect visual embeddings zt from the auto-regressive director, we fine-tune the model using noisy embeddings z′t defined by:

z′t = S(M(zt + ϵ)) (8)

where ϵ ∼ N(0,σ2zt) represents Gaussian noise, M is a masking operator that sets a fraction of elements to zero, and S is a shuffling operator that permutes the order.

- 5. Experiments

###### 5.1. Experimental Setup

Models. We initialize the auto-regressive model with [13], a pretrained 7B multi-modal LLM. We initialize the context conditioned render model with FLUX.1 Fill model [27]. For video generation, we employ a video generation model which has been pre-trained on large-scale video-text pairs and could accept both text and visual conditions.

Data. We use a total of ∼32K narrative videos for model training and another ∼1K videos for validation. All the videos are resized to 448 (short-side) and then centercropped with 448×448 resolution.

Training & Evaluation. We train the interleaved autoregressive director model for 5,000 steps by default. Training loss is a combination of cosine similarity loss and MSE loss for visual tokens and CrossEntropy loss for language tokens. For rolling context conditioned render, we use the flow matching loss following FLUX [27]. For visualconditioned video generation, we use the diffusion loss following DiT [34] and Stable Diffusion 3 [10]. Narrative generation is mostly evaluated on the Youcook2 validation set because of the high-quality of action annotations and the Howto100M validation set is mostly used for data quality evaluation and I2V generation. Please refer to the appendix for implementation details.

Evaluation Metrics. The common metrics CLIP score [17] and FVD [49] are used to assess overall video quality, while the FID [18] score evaluates the quality of the generated keyframes. Additionally, when comparing to state-of-theart baselines, human evaluation is used to assess generation aesthetics, realism, visual consistency across video clips, and the narrative score which reflects the coherence of the generated cooking steps, and if the cooking process has been successfully completed.

###### 5.2. Rolling Context Conditioning

- As detailed in Section 4.2, we leverage the in-context conditioning capabilities of the transformer architecture and adopt a rolling context conditioning strategy to enable DiT to render keyframes with superior visual consistency, while adhering to the extended narrative semantics produced by the interleaved auto-regressive director model. As shown in Figure 5, our keyframe renderer preserves fine visual details and exhibits high visual quality and aesthetics with the help of large-scale pretraining [27]. The reason is that the in-context conditioned VAE features could preserve visual details and the semantics are preserved through the autoregressive model. Notably, the rolling context conditioning approach allows the renderer to strike a flexible balance between generation efficiency and visual consistency by dynamically adjusting the number of frames generated in each forward pass (i.e., a dynamic number of frames).

5.3. Visual-Conditioned Video Generation

- As detailed in Section 4.3, we fine-tune the model to be directly conditioned on the visual latents and generated by our interleaved director and keyframes generated by rolling-context renderer. Table 3 compares the keyframe-conditioned approach with our visual embeddingconditioned strategy. Our method improves CLIP-T [17] scores on both validation sets—from 25.9 to 26.4 on YouCook2 and from 26.6 to 27.3 on HowTo100M. Additionally, FVD scores decrease, indicating better video quality (557.7 vs. 512.6 on YouCook2, 541.1 vs. 520.7 on HowTo100M). Videos conditioned on visual embeddings

[Figure 71]

[Figure 72]

OursDiffusion Seed-

Vlogger Story-

[Figure 73]

[Figure 74]

[Figure 75]

Story

[Figure 76]

Figure 6. Quality comparison on long narrative generation. Here is a case with a narrative topic of “Step-by-step guide to cooking blueberry muffins”. Our interleaved director sequentially generates ”actions,” ”captions,” and image embeddings to construct a narrative on how to cook the dish step by step and then render keyframes. Our method shows state-of-the-art visual quality with superior consistency.

|Visual Condition<br><br>|YouCook2 HowTo100M<br><br>CLIP-T ↑ FVD ↓ CLIP-T ↑ FVD ↓|
|---|---|
|Keyframe Embedding (w/o Reg.) Embedding (w/ Reg.)<br><br>|25.9 557.7 26.6 541.1<br><br>25.5 590.3 26.4 554.3<br>26.4 512.6 27.3 520.7<br>|

Table 3. Visual-conditioned Video Generation with Regularization. Evaluate CLIP-T and FVD scores for video generation conditioned on keyframes versus visual embeddings generated by our interleaved director with and without regularization.

demonstrate higher semantic alignment and improved generation quality. We also provide qualitative samples on the demo page and in the appendix.

###### 5.4. Comparisons of Long Narrative Generation

As most existing narrative generation methods [55, 64] only support image generation, we compare our model with state-of-the-art methods on the task of long narrative keyframe generation. We provide both quantitative comparisons in (§5.4.1) and qualitative comparisons (§5.4.2).

###### 5.4.1 Long Narrative Keyframe Generation

We compare our method with leading narrative keyframe generation approaches, including IC Lora [22], StoryDiffusion [64], Vlogger [65], and Seed-Story [55], as well as a language-centric strategy that relies solely on captions (using models such as SD-XL [35] and FLUX.1-schnell [27]).

Prompting Gen. Metric Human Evaluation

Method

Prompt Src. Cond. CLIP-T FID Aes. Real. Consist. Narr. SD-XL [35] External Text 27.1 - 4.0 2.9 3.3 N/A

FLUX.1-s [27] External Text 27.9 - 4.8 3.1 3.4 N/A

IC Lora [22] External Text 27.9 34.1 4.7 4.1 4.7 N/A StoryDiffusion [64] External Text 25.9 36.4 3.9 2.9 3.7 N/A

Vlogger [65] LLM Text 25.5 45.5 4.0 2.4 3.1 3.7 Seed-Story [55] Interleaved V. Emb. 24.1 32.1 1.9 4.1 4.2 4.1 Ours (w.o RCC) Interleaved V. Emb. 26.1 25.3 2.1 4.3 4.5 4.4 Ours (w. RCC) Interleaved T+V. Emb. 28.0 29.4 4.8 4.5 4.8 4.6

Table 4. Quantitative comparisons with metrics and human evaluation. Each method is evaluated by both image generation metrics (CLIP-T and FID) and human ratings. Higher values indicate better performance for all human-evaluation metrics (5 tiers, from 1 to 5, higher is better). SD-XL and FLUX.1-s use narrative captions generated by our model and IC-Lora uses a tiled version. RCC: Rolling Context Conditioning. We use our generated narrative captions for the text-conditioned methods (row 1-5).

Except for IC Lora and Seed-Story, which are fine-tuned on our CookGen dataset for two epochs, all other methods follow their official inference guidelines with the official checkpoints. As shown in Table 4, our approach achieves the highest generation scores, with a CLIP-Text score of 28.0 and an FID score of 25.3. We also conduct a human evaluation (Table 4) using a five-tier rating scale, where higher is better. Our method attains top performance in aesthetics (4.8 vs. 4.7, IC Lora), realism (4.5 vs. 4.1, SeedStory), and visual consistency (4.8 vs. 4.7, IC Lora), as well

|Loss Type MSE Cosine|Training Validation<br><br>L2 Dist ↓ Cosine Sim. ↑ CLIP-T ↑ FID ↓<br><br>|
|---|---|
|✓ ✗ ✗ ✓ ✓ ✓|0.41 0.82 23.6 31.9<br><br>1.1 0.82 24.1 32.1<br><br><br>0.41 0.83 25.1 30.1|

Table 5. Both scale and direction matter. We track the training convergence and evaluate models with the CLIP-T and FID metrics on the validation set. The combination of both MSE loss and Cosine Similarity loss performs best on the validation metrics.

as the highest narrative score of 4.6. These results demonstrate that our method achieves state-of-the-art performance for long narrative generation.

###### 5.4.2 Qualitative Comparisons

In Figure 6, we compare our method with state-of-theart long narrative keyframe generation approaches, including StoryDiffusion, Vlogger, and Seed-Story, and observe that our results maintain superior visual quality and consistency. In particular, our keyframes balance realism with appealing aesthetics while preserving character identities and smooth transitions. In contrast, competing methods often exhibit color inconsistencies or lose track of concepts—Vlogger occasionally produces uneven color schemes between frames, StoryDiffusion can introduce visual confusion, and Seed-Story sometimes generates mismatched clothing across different scenes. This comparison aligns with the human evaluation results in Table 4, demonstrating our method achieves state-of-the-art performance for long narrative visual generation. The generated keyframes can be extended into full video clips with consistent visuals and coherent storytelling.

###### 5.5. Ablation Studies

In this section, we ablate important designs in VideoAuteur, which improve the interleaved auto-regressive model and the visual-conditioned video generation model for interleaved narrative visual generation.

Latent scale and direction matter. To determine an effective supervision strategy for visual embeddings, we firstly test the robustness of the latents to pseudo regression errors by rescaling (multiplying by a factor) and adding random Gaussian noise. Figure 17 indicates that both scale and direction are critical in latent regression. Notably, rescaling primarily affects object shape while preserving key semantic information (i.e. object type and location), whereas adding noise drastically impacts reconstruction quality. As shown in Section 5.5, combining MSE loss (for scale ) and cosine similarity (for direction) leads to the best generation quality, improving CLIP-T by 1.5 points and reducing FID by 1.8 points compared to using MSE alone.

From “Actions” to “Visual States”. We also explore how different regression tasks influence the director’s capability in narrative visual generation. Specifically, we compare var-

|Regression Task|Training Validation<br><br>L2 Dist ↓ Cosine Sim. ↑ CLIP-T ↑ FID ↓<br><br>|
|---|---|
|Action → Vis. Embed. Caption → Vis. Embed. Action → Caption → Vis. Embed.|0.43 0.82 22.7 27.9<br><br>0.41 0.82 25.7 26.1<br><br>0.41 0.83 26.1 25.3<br>|

- Table 6. From “Actions” to “Visual States”. We report the L2 distance and cosine similarity scores for tracking the training convergence and evaluate the generation images with CLIP score and FID score. Models are trained and evaluated on the collected Howto100M subset. SEED-X latent is used for visual regression.

Regularization Setting CLIP-T ↑ FVD ↓

Naive Baseline 26.4 554.3 +Random Masking 26.9 539.7 +Random Gaussian. Noise 27.2 522.1 +Random Shuffling 27.3 520.7

- Table 7. Learn from Noisy Visual Conditions. Our training regularization strategy enhances the robustness of the visualconditioned video generation model. Specifically, we apply random masking and shuffling at a rate of 25%, and introduce Gaussian noise with 0.5 std of the embeddings of two thousand samples.

ious reasoning settings for the interleaved director, examining transitions from sequential actions to language states, and ultimately to visual embeddings. As shown in Table 6, a chain of reasoning that progresses from actions to language states and then to visual states proves effective for long narrative visual generation. This approach enhances both training convergence, achieving a lower L2 distance (0.41 vs. 0.43), and generation quality, reflected in a superior FID score of 25.3 (an improvement of +0.8).

Learn from noisy visual conditions. Table 7 presents an ablation study examining the effect of robustness regularization on the visual-conditioned video generation model. We evaluate the generated videos using CLIP-T and FVD. The progressively improved results from 26.4 to 27.3 on CLIP-T and 554.3 to 520.7 on FVD demonstrate the effectiveness of our regularization strategy, which combines random masking, Gaussian noise, and shuffling.

#### 6. Conclusion

In this paper, we tackle the challenges of generating longform narrative videos and empirically evaluate its efficacy in the cooking domain. We curate and annotate a large-scale cooking video dataset, capturing clear and high-quality narratives essential for training and evaluation. Our proposed two-stage auto-regressive pipeline, which includes a long narrative director, a rolling context conditioned keyframe renderer and a visual-conditioned video generation model, demonstrates promising improvements in semantic and visual consistency in generated long narrative videos with an unified pipeline. Through experiments on our dataset, we observe enhancements in spatial and temporal coherence across video sequences. We hope our work can facilitate further research in long narrative video generation.

#### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurlPS,

2022. 2

- [2] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators, 2024. 1, 2
- [3] Jerome Bruner. The narrative construction of reality. Critical inquiry, 18(1):1–21, 1991. 1
- [4] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In CVPR,

2023. 1

- [5] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. In arXiv,

2023. 1, 2

- [6] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, 2024. 1, 2
- [7] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In CVPR, 2024. 2
- [8] Xi Chen, Zhiheng Liu, Mengting Chen, Yutong Feng, Yu Liu, Yujun Shen, and Hengshuang Zhao. Livephoto: Real image animation with text-guided motion control. In ECCV,

2025. 1

- [9] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. In arXiv, 2023. 2
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning, pages 12606–12633. PMLR, 2024. 6
- [11] Ruoyu Feng, Wenming Weng, Yanhui Wang, Yuhui Yuan, Jianmin Bao, Chong Luo, Zhibo Chen, and Baining Guo. Ccedit: Creative and controllable video editing via diffusion models. In CVPR, 2024. 1
- [12] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. In NeurlPS, 2024. 2
- [13] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. In arXiv, 2024. 2, 6

- [14] Tanmay Gupta, Dustin Schwenk, Ali Farhadi, Derek Hoiem, and Aniruddha Kembhavi. Imagine this! scripts to compositions to videos. In Proceedings of the European conference on computer vision (ECCV), pages 598–613, 2018. 2
- [15] Yuval Noah Harari. Sapiens: A brief history of humankind. Random House, 2014. 1
- [16] Yingqing He, Menghan Xia, Haoxin Chen, Xiaodong Cun, Yuan Gong, Jinbo Xing, Yong Zhang, Xintao Wang, Chao Weng, Ying Shan, et al. Animate-a-story: Storytelling with retrieval-augmented video generation. arXiv preprint arXiv:2307.06940, 2023. 2
- [17] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718,

2021. 6

- [18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6
- [19] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. In arXiv, 2022. 1, 2
- [20] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NeurlPS, 2022. 1, 2
- [21] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In arXiv, 2022. 1, 2
- [22] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024. 2, 7
- [23] Nisha Huang, Yuxin Zhang, and Weiming Dong. Style-avideo: Agile diffusion for arbitrary text-based video style transfer. IEEE Signal Processing Letters, 2024. 1
- [24] Ting-Hao Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Aishwarya Agrawal, Jacob Devlin, Ross Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, et al. Visual storytelling. In Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: Human language technologies, pages 1233–1239, 2016. 2
- [25] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. In ICML, 2024. 1, 2
- [26] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, 2023. 2
- [27] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 6, 7
- [28] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with

frozen image encoders and large language models. In ICML,

2023. 2

- [29] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking. In CVPR, 2023. 2
- [30] Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. In COLM, 2024. 2
- [31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurlPS, 2024. 2
- [32] Adyasha Maharana and Mohit Bansal. Integrating visuospatial, linguistic and commonsense structure into story visualization. arXiv preprint arXiv:2110.10834, 2021. 2
- [33] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2630–2640, 2019. 2, 3
- [34] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 1, 2, 6
- [35] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 7
- [36] Lu Qi, Lehan Yang, Weidong Guo, Yu Xu, Bo Du, Varun Jampani, and Ming-Hsuan Yang. Unigs: Unified representation for image generation and segmentation. In CVPR, 2024. 2
- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2
- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2
- [39] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurlPS, 2022. 2
- [40] Nina Shvetsova, Anna Kukleva, Xudong Hong, Christian Rupprecht, Bernt Schiele, and Hilde Kuehne. Howtocaption: Prompting llms to transform video annotations at scale. In European Conference on Computer Vision, pages 1–18. Springer, 2025. 3
- [41] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. In arXiv, 2022. 1, 2
- [42] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. In arXiv, 2023. 2
- [43] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun

- Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. In ICLR, 2023. 1, 2
- [44] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In CVPR, 2024. 1, 2, 23
- [45] Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, et al. Mm-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronizer. In arXiv, 2024. 2
- [46] Ye Tian, Ling Yang, Haotian Yang, Yuan Gao, Yufan Deng, Jingmin Chen, Xintao Wang, Zhaochen Yu, Xin Tao, Pengfei Wan, et al. Videotetris: Towards compositional text-to-video generation. arXiv preprint arXiv:2406.04277, 2024. 2
- [47] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aur´elien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2
- [48] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 17
- [49] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv:1812.01717, 2018. 6, 18
- [50] Chaoyang Wang, Xiangtai Li, Lu Qi, Henghui Ding, Yunhai Tong, and Ming-Hsuan Yang. Semflow: Binding semantic segmentation and image synthesis via rectified flow. In NeurlPS, 2024. 2
- [51] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. In arXiv, 2023. 1, 2
- [52] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. Visionllm: Large language model is also an openended decoder for vision-centric tasks. In Advances in Neural Information Processing Systems, 2024. 2
- [53] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. In arXiv, 2024. 1, 2
- [54] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In CVPR, 2024. 1, 2
- [55] Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen. Seed-story: Multimodal long story generation with large language model. In arXiv,

2024. 2, 7

- [56] Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen. Seed-story: Multimodal long story generation with large language model. arXiv preprint arXiv:2407.08683, 2024. 2

- [57] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In arXiv, 2024. 1, 2
- [58] Xuanyu Yi, Zike Wu, Qingshan Xu, Pan Zhou, Joo-Hwee Lim, and Hanwang Zhang. Diffusion time-step curriculum for one image to 3d generation. In CVPR, 2024. 2
- [59] Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, et al. Direct preference optimization of video large multimodal models from language model reward. arXiv preprint arXiv:2404.01258, 2024. 3, 18, 20
- [60] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. 3
- [61] Canyu Zhao, Mingyu Liu, Wen Wang, Jianlong Yuan, Hao Chen, Bo Zhang, and Chunhua Shen. Moviedreamer: Hierarchical generation for coherent long visual sequence. arXiv preprint arXiv:2407.16655, 2024. 2
- [62] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. In arXiv, 2022. 1, 2
- [63] Luowei Zhou, Chenliang Xu, and Jason J Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018. 2, 3
- [64] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent selfattention for long-range image and video generation. arXiv preprint arXiv:2405.01434, 2024. 2, 7
- [65] Shaobin Zhuang, Kunchang Li, Xinyuan Chen, Yaohui Wang, Ziwei Liu, Yu Qiao, and Yali Wang. Vlogger: Make your dream a vlog. In CVPR, 2024. 2, 7

## Appendix

This appendix provides comprehensive supplementary materials to support our study. Below are brief descriptions of all the sections covered in the appendix. Please visit our project page for more visualization.

###### • Appendix A: Data Examples with Annotations

- – Presents data examples from our CookGen dataset.
- – Showcases annotated “actions” and “captions” that provide detailed multimodal information of cooking processes.

###### • Appendix B: Additional Data Statistics

- – Offers distributions of video lengths, clip lengths, and textual annotations.
- – Demonstrates the dataset’s richness and suitability for long narrative video generation.

###### • Appendix C: Data Evaluation Details

- – Details our data evaluation process.
- – Includes inverse video generation results, the prompts used for video captioning, GPT-4o evaluations, and human evaluation results.

###### • Appendix D: Implementation Details

- – Outlines the implementation details of our models.
- – Provides key hyperparameters and training & inference configurations.

###### • Appendix E: Action-Caption Matching Pseudo Code

- – Includes the pseudo code for our action-caption matching algorithm.
- – Essential for aligning video clips with their corresponding annotations.

###### • Appendix F: CLIP beats VAE for interleaved generation

- – Introduces three autoencoders (EMU-2, SEED-X, SDXL-VAE) and compares reconstruction vs. generation performance.
- – Demonstrates that CLIP-diffusion embeddings (EMU-2, SEED-X) outperform SDXL-VAE in language-driven visual generation due to better vision-language alignment.

###### • Appendix G: Generated Video Examples

- – Showcases generated video examples.
- – Illustrates the effectiveness of our pipeline in producing long narrative videos for cooking recipes like “Fried Chicken” and “Shish Kabob.”

###### • Appendix H: Limitations

- – Discusses the limitations of our approach.
- – Includes issues with noisy “actions” from automatic speech recognition and potential failure cases in video generation.

#### A. Data Examples with Annotations

Figures 7 and 8 shows two data examples from our CookGen dataset, annotated with high-quality descriptions that provide detailed multi-modal information of cooking processes. The examples clearly show structured annotations of key actions and corresponding visual descriptions, making the dataset ideal for generating long narrative videos.

[Figure 77]

(a) Action: Elise works with chicken thighs, advises to trim excess skin and fat

Caption: A person is preparing chicken on a wooden cutting board. He uses a pair of black-handled scissors to cut through the chicken pieces, which are spread out on a clear cutting mat.

[Figure 78]

(c) Action: Elise heats up a large skillet with two teaspoons of olive oil and a teaspoon of butter Caption: A person is seen in a kitchen setting, holding a wooden spoon. He places a small piece of butter into a black frying pan on a gas stove.

[Figure 79]

(b) Action: She offers alternatives with chicken breast bone-in skin-on or chicken drumsticks

Caption: A person with light skin is preparing raw chicken pieces on a wooden surface. He places several pieces of chicken on a white cutting board.

[Figure 80]

(d) Action: Turn over the chicken pieces and cook for another 4 minutes Remove the chicken from the pan but keep the browned pieces in the pan Caption: Golden-brown chicken pieces are sizzling in a black frying pan on a gas stove.

[Figure 81]

(e) Action: Use the remaining oil in the pan to brown the orzo Cook the orzo like a traditional rice pilaf, using the same method as before

Caption: A person is cooking rice in a black frying pan on a gas stove. He pours the rice from a glass bowl into the pan, then uses a wooden spatula to spread and stir the rice.

[Figure 82]

(g) Action: Combine the mixture with the orzo and cook for a few minutes until the sauce thickens Caption: A woman is cooking on a stovetop, adding pieces of breaded chicken to a pan filled with chopped onions and rice.

[Figure 83]

(f) Action: Add 2 cups of gordo’s to a hot pan

Caption: A person wearing a blue shirt is cooking rice in a black frying pan on a stovetop. Using a wooden spatula, he stirs the rice, ensuring it is evenly cooked.

[Figure 84]

(h) Action: Stock is cooked until orzo has fully absorbed liquid and chicken is cooked through, about 10-12 minutes Dish is removed from heat and left to sit for five minutes Dish is sprinkled with unspecified seasoning Caption: A delicious dish of roasted chicken pieces is presented in a black skillet, surrounded by a colorful mix of diced vegetables and grains.

Figure 7. Data examples with annotated “actions” and “captions”. A video of cooking recipe of “One Pot Chicken and Orzo”.

[Figure 85]

(a) Action: Hi everyone, this one’s called rainbow broken glass jello

Caption: A colorful, multi-layered dessert is displayed on a black surface. The dessert features vibrant red, green, blue, and purple segments, arranged in a geometric pattern.

[Figure 86]

(c) Action: I find the easiest way to do this is to put the small container into a larger container of hot water Caption: A person with light skin is holding a clear plastic container filled with a yellow liquid, inspecting its contents.

[Figure 87]

(b) Action: Now normally when you make jello you use two cups of boiling water, but in this case we’re only using one cup because we want the jello to be extra firm

Caption: The video shows the interior of a refrigerator, focusing on the door shelf. The containers are filled with dark, blue, orange, and red liquids.

[Figure 88]

(d) Action: Loosen the edges of the Jello piece Slide the Jello piece out and cut it into cubes Cut the Jello cubes into half-inch pieces Caption: A person is slicing a block of yellow gelatin on a wooden cutting board, cutting it into uniform strips.

[Figure 89]

(e) Action: Spread out the different colored Jello pieces in a 9 by 13 inch baking dish Caption: A person is arranging colorful gelatin cubes in a glass baking dish, adjusting the placement of green, orange, purple, and black cubes.

[Figure 90]

(f) Action: Make a separate gelatin mixture by boiling two cups of water and adding two envelopes of gelatin Caption: A clear glass measuring cup is placed on a countertop, containing water. A person pours a white powder into it.

[Figure 91]

(g) Action: Stir the sweetened condensed milk into the gelatin and water mixture Caption: A person is vigorously whisking a creamy mixture in a clear glass measuring cup.

[Figure 92]

(h) Action: Let it set for several hours, then cut it into squares and serve Caption: A glass baking dish is filled with a creamy white liquid, topped with colorful, triangular-shaped glass pieces.

Figure 8. Data examples with annotated “actions” and “captions”. A video of preparing “Rainbow Broken Glass Jello”.

#### B. Additional Data Statistics

##### Video Length Distribution

##### Clip Length Distribution

##### Clip  umber Distribution

100

12

6

Frequency

Frequency

Frequency

8

4

60

4

2

20

| | |
|---|---|
| | |

| | |
|---|---|
| | |

(0,2](2,4](4,6](6,8]

(8,10]

(0,5]s(5,10]s(10,15]s(15,20]s(20,25]s(25,30]s(30,35]s(35,40]s

(10,12](12,14](14,16](16,18]

(0,30]s(30,60]s(60,90]s(90,120]s(120,150]s(150,180]s(180,210]s(210,240]s(240,270]s(270,300]s

- Figure 9. Statistics on the video data. We do statistics on the video lengths of the collected whole videos, the clip lengths of the scene-cut video clips, and the number of clips selected for each video.

(0,5](5,10](10,15](15,20](20,25](25,30](30,35](35,40]>40

20

50

80

Frequency

Word  umber Distribution of Actions

(0,40](40,50](50,60](60,70](70,80](80,90](90,100]>100

20

50

80

Frequency

Word  umber Distribution of Captions

(0,10](10,20](20,30](30,40](40,50](50,60]>60

20

50

80

Frequency

Token  umber Distribution of Actions

(0,60](60,70](70,80](80,90](90,100](100,110](110,120](120,130]

20

50

80

Frequency

Token  umber Distribution of Captions

- Figure 10. Statistics on the text annotations. We do statistics on the number of words and tokens (Llama [48] tokenized) of annotated “actions” and “captions,” respectively.

The statistics in Figure 9 and Figure 10 demonstrate the high quality and suitability of our dataset for long narrative video generation. Figure 9 reveals that the video lengths range broadly, with most videos falling between 30 and 150 seconds. Clip lengths are primarily distributed between 5 and 30 seconds, ensuring manageable segments for modeling. Additionally, the majority of videos contain 4 to 12 clips, providing a balanced structure for narrative flow. Figure 10 shows that the word counts for ”actions” predominantly range from 10 to 25, while ”captions” range from 40 to 70. Token distributions further highlight their richness, with ”actions” having 20 to 60 tokens and ”captions” extending up to 120 tokens. These detailed annotations ensure well-aligned and contextually rich representations of the video content.

Overall, the dataset’s design ensures coherent sequences of actions and captions with reasonable clip and video lengths, making it well-suited for generating high-quality, long-form narrative videos.

#### C. Annotation Quality Reverification Details

High-quality captions are essential for narrative visual generation. To verify the quality of our annotations, we build an evaluation pipline of inverse generation (§C.1) and visual understanding through VLM experts (§C.2).

###### C.1. Inverse Video Generation

This evaluation is motivated by the understanding that high quality captions, when combined with ground truth keyframes, more effectively reconstruct the original videos. We evaluate the dataset’s ability to reconstruct original videos using the annotated captions, with and without conditioning with ground truth keyframes. For this evaluation, we assess the validation set (∼5,000 video clips). We measure reconstruction quality using FVD [49]. The results, shown in Table 12, indicate that our captions capture sufficient semantic information, enabling effective representation of the original videos. When generating with ground-truth keyframes, the video quality is very high and closely aligned with the original videos, as shown by the low FVD score (116.3). Without keyframes, the captions alone still provide reasonable alignment. Examples of reconstructed videos are included in the supplementary materials.

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Our Captioner

SoTA VLM (Qwen2-VL 72B)

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Text-to-Video Model

Please score the annotations, …

Inverse Video Generation GPT-4o & Human Evaluation

Figure 11. Annotation quality evaluation pipeline. We verify our annotation quality through a pipeline of two major aspects: 1) inverse video generation 2) GPT-4o and human evaluation.

|Validation Set<br><br>|w/. GT keyframe|W/o. GT keyframe|
|---|---|---|
|# Clips 5504<br><br>|FVD 116.3<br><br>|FVD 561.1|

Figure 12. High-quality captions enable inverse video generation. We utilize the annotated captions and actions to inversely generate video clips using a pretrained Text-to-Video diffusion model. Higher reconstruction fidelity (i.e., similarity to the original videos) indicates superior captions and actions. Our inversely generated videos achieve very low FVD scores compared to the original videos, highlighting the high quality of our annotations.

| |GPT-4o Evaluation<br><br>|Human Evaluation|
|---|---|---|
|Score (0-100)|Qwen2-VL-72B Ours<br><br>98.0 95.2<br><br>|Qwen2-VL-72B Ours 79.3 82.0|

Table 8. Caption Quality Evaluation. We compare the caption quality between our captioner and the Qwen2-VL-72B model by both GPT-4o and human annotators. Our model achieves competitive results despite a much smaller model size.

###### C.2. Semantic Consistency across VLM Experts

GPT-4o & human evaluation. We evaluate the quality of our captions using both GPT-4o and six human annotators, in which we ask humans and GPT-4o to rate our dataset provided captions according to two criteria: the coverage of video elements and the absence of hallucinations in the caption. Following [59], hallucination refers to the model generating content absent or unsupported by the video, such as incorrect details about objects, actions, or counts.

To demonstrate the quality, we compare our captions with those generated by a state-of-the-art open-source VLM (Qwen2VL-72B). As shown in Table 8, our dataset’s captions receive a decent score of 95.2 out of 100, showing slightly better alignment with rigorous human evaluation than the Qwen2-VL-72B model. Results from both human evaluators and GPT-4 assessments indicate that the dataset contains high-quality captions.

###### C.3. Inverse Video Generation Results

As discussed in Appendix C.1, high-quality captions, especially with ground truth keyframes, enable effective video reconstruction. We compare ground truth video frames with inversely generated frames using the GT first keyframe and annotated captions, as shown in Figures 13 to 15. The reconstruction aligns well with the narrative, accurately capturing actions, though patterns and interactions differ slightly from the original video. This shows that while the captions convey crucial information for reconstruction, they lack finer visual details, a limitation for current vision-language models and human annotators.

For example, in Figure 13, the ground truth shows a hand pouring creamy liquid into a slow cooker and stirring, while the generated frames replicate the actions with slight differences in texture and liquid mixing. Similarly, in Figure 15, the ground truth shows a face drawn with cream on orange liquid, but the generated frames vary in precision and interaction details.

These examples highlight the captions’ strength in preserving narrative flow while exposing gaps in capturing fine-grained visual detail.

[Figure 102]

[Figure 103]

- Figure 13. Left: Ground truth, Right: Inverse generation with GT keyframe. Caption: Chunks of meat are simmering in a dark-colored slow cooker. A hand pours a creamy liquid into the pot, causing the liquid to mix with the meat and broth. The mixture bubbles and thickens as the liquid is added. The person stirs the contents with a black spoon, ensuring the ingredients are well combined. The slow cooker continues to cook the meat, which appears tender and well-cooked.

[Figure 104]

[Figure 105]

- Figure 14. Left: Ground truth, Right: Inverse generation with GT keyframe. Caption: A person wearing a black sleeve is whisking a creamy mixture in a clear glass bowl. The mixture appears to be a batter or dough, gradually becoming smoother and more uniform. The person’s left hand holds the bowl steady on a light-colored countertop. The whisking motion is consistent and thorough, ensuring the mixture is well-blended. The background is plain, focusing attention on the mixing process.

[Figure 106]

[Figure 107]

- Figure 15. Left: Ground truth, Right: Inverse generation with GT keyframe. Caption: A red bowl filled with a thick, orange liquid is placed on a stovetop. A woman’s hand, holding a white spoon, appears and begins to draw on the surface of the liquid. She creates a face with white cream, adding details to the eyes and mouth. The background shows a granite countertop with a bunch of red tomatoes and a white pot. The woman continues to add finishing touches to the face.

###### C.4. Prompt for Video Captioning

Below is the prompt we designed to effectively caption video clips and also for benchmarking VLMs, ensuring detailed and accurate descriptions while avoiding redundancy:

You are an expert in describing videos and catching the sequential motions from video frames. For the given ten video frames, you need to generate a detailed good description within five sentences / 80 words. Please do not include the word ’frame’ or ’frames’ in your answer. If the gender of a person is clear, use ’he’ or ’she’ instead of they. Do not describe a single motion/action twice like ’xxx continues doing yyy’. Don’t assume actions like discussion or

having a conversation unless it is very clear in the frames. Describe the video given the frame sequence. Describe both the appearance of people (gender, clothes, etc), objects, background in the video, and the actions they take.

Listing 1. Video Captioning Prompt

###### C.5. GPT-4o Evaluation on Captions

Below is the evaluation prompt designed to objectively assess the quality of video captions generated by a Large Multimodal Model (LMM), focusing on coverage and hallucination.

Your role is to serve as an impartial and objective evaluator of a video caption provided by a Large Multimodal Model (LMM). Based on the input frames of a video, assess primarily on two criteria: the coverage of video elements in the caption and the absence of hallucinations in the response. In this context, ’hallucination’ refers to the model generating content not present or implied in the video, mainly focused on incorrect details about objects, actions, counts, temporal order, or other aspects not evidenced in the video frames.

To evaluate the LMM’s response: Start with a brief explanation of your evaluation process. Then, assign a rating from the following scale:

Rating 6: Very informative with good coverage, no hallucination Rating 5: Very informative, no hallucination Rating 4: Somewhat informative with some missing details, no hallucination Rating 3: Not informative, no hallucination Rating 2: Very informative, with hallucination Rating 1: Somewhat informative, with hallucination Rating 0: Not informative, with hallucination

Do not provide any other output symbols, text, or explanation for the score. Listing 2. GPT-4o Evaluation Prompt

###### C.6. Human Evaluation on Captions

|Matching Tier|Action (Important Info.) Object (Important Info.) Score<br><br>|
|---|---|
|Very Match Good Match Somehow Match Not Match<br><br>|Good Coverage, No Hallucination Good Coverage, No Hallucination 100 Good Coverage, Limited Hallucination Good Coverage, Limited Hallucination 85<br><br>Fair Coverage, Some Hallucination Fair Coverage, Some Hallucination 70 Little Coverage or High Hallucination Little Coverage or High Hallucination 0|

Table 9. Human Evaluation Matching Rules. Captions are rated based on coverage and hallucination levels, using four matching tiers.

We assess the quality of our captions through evaluations by six human annotators, who rate the captions based on two key criteria: the coverage of video elements (such as objects and actions) and the absence of hallucinations, defined as generating content unsupported or absent in the video [59]. As shown in Table 8, our captions achieve a high human evaluation score of 82.0, surpassing the state-of-the-art open-source VLM (Qwen2-VL-72B) score of 79.3. These results demonstrate the superior quality of our captions, which are more aligned with human preferences and exhibit better narrative accuracy.

For evaluation, annotators rate the captions across four tiers—Very Match, Good Match, Somehow Match, and Not Match—based on consistency with video content. The scoring rubric, detailed in Table 9, considers both coverage and hallucination levels. Our captioner consistently achieves high scores in the top tiers, validating its reliability and quality for narrative video generation.

#### D. Implementation Details

We provide the training and inference hyperparameters for the interleaved auto-regressive model and the visual-conditioned video generation model in Table 10 and Table 12, respectively. The interleaved auto-regressive model is trained on images with a resolution of 448 × 448, using a batch size of 512 and bfloat16 precision. It employs AdamW as the optimizer, with a peak learning rate of 2 × 10−4 and a cosine decay schedule, training for 2,500 steps. Training context pairs vary between 2 and 8, while inference always uses 8 pairs for consistency. The visual-conditioned video generation model processes video data at a resolution of 448 × 448 × T, with a batch size of 64 and bfloat16 precision. It uses AdamW with a peak learning rate of 1 × 10−5 and a constant decay schedule, training for 20,000 steps to handle temporal conditioning effectively.

Configuration Setting Image resolution 448 × 448

Optimizer AdamW Optimizer hyperparameters β1 = 0.9, β2 = 0.98, ϵ = 10−6 Peak learning rate 2 × 10−4 Learning rate schedule Linear warm-up, cosine decay Gradient clip 1.0

Total training steps 2, 500 Warm-up steps 200

Batch size 512

Numerical precision bfloat16 Training context pairs [2, 8] Inference context pairs 8

###### Table 10. Implementation details of the interleaved auto-regressive model.

Configuration Setting

Image Resolutions 448 × 448 × 4 Optimizer AdamW

Optimizer hyperparameters β1 = 0.9, β2 = 0.999, ϵ = 10−8 Peak learning rate 5 × 10−4 Learning rate schedule Linear warm-up, constant Gradient clip 1.0

Total training steps 5, 000 Warm-up steps 500

Batch size 16 Optional Masking Probability 0.25 Numerical precision bfloat16

###### Table 11. Implementation details of the rolling context conditioned render.

Configuration Setting Image/Video resolution 448 × 448 × T

Optimizer AdamW Optimizer hyperparameters β1 = 0.9, β2 = 0.95, ϵ = 10−8 Peak learning rate 1 × 10−5 Learning rate schedule Linear warm-up, constant

Gradient clip 1.0 Total training steps 20, 000

Warm-up steps 1, 000 Batch size 64

Numerical precision bfloat16 Table 12. Implementation details of the visual-conditioned video generation model.

#### E. Action-Caption Matching Pseudo Code

The action-caption matching algorithm detailed in Algorithm 1 aligns video clips with actions based on temporal overlap and specific rules. It uses the Intersection over Union (IoU) to measure the overlap between the time intervals of video clips and actions. A match is identified if either the IoU exceeds 0.5 or all of the following conditions are met: the start time difference (start diff) is less than 5 seconds, the clip’s end time exceeds the action’s end time, and the IoU is greater than 0.2.

The algorithm processes each video iteratively. For each video, it retrieves all associated actions and their time intervals. Then, for each clip in the video, it calculates the IoU with every action and evaluates the matching conditions. Valid matches, along with their metadata (clip info and descriptions), are stored in a list M. This systematic approach ensures that the matched actions and captions are temporally consistent, providing high-quality annotations for keyframe visual states.

###### Algorithm 1 Pseudo code for action-caption matching.

- 1: function IOU([s1,e1],[s2,e2])
- 2: intersection ← max(0,min(e1,e2) − max(s1,s2))
- 3: union ← max(e1,e2) − min(s1,s2)
- 4: if union > 0 then
- 5: return intersectionunion

- 6: else
- 7: return 0
- 8: end if
- 9: end function
- 10: Initialize an empty list M ← []
- 11: for all v ∈ V do
- 12: vid ← v.id
- 13: if vid ∈ A then
- 14: Av ← A[vid]
- 15: action times ← Av.times

- 16: action descriptions ← Av.descriptions

- 17: for all c ∈ v.clips do
- 18: [sc,ec] ← c.start end

- 19: for all a ∈ action times do

- 20: [sa,ea] ← a
- 21: start diff ← |sc − sa|

- 22: iou ← IOU([sc,ec],[sa,ea])
- 23: if (start diff < 5 ∧ ec > ea ∧ iou > 0.2) ∨ iou > 0.5 then

- 24: Create match: M ← M ∪ m
- 25: end if
- 26: end for
- 27: end for
- 28: end if
- 29: end for
- 30: return M

#### F. CLIP beats VAE for interleaved generation.

We experiment with three different auto-encoded visual latent spaces for regression: the EMU-2 [44] CLIP-Diffusion autoencoder, the SEED-X CLIP-Diffusion autoencoder, and the KL Variational autoencoder (VAE) used by SDXL. Both SEED-X and EMU-2 use a CLIP vision encoder and a finetuned SDXL diffusion model as the decoder for encoding visual latent. From appendix Figure 16, we observe that SDXL-VAE achieves the best reconstruction quality. However, in terms of visual generation quality, as shown in Table 13, the CLIP-Diffusion based autoencoders significantly outperform VAE (i.e., +12.2 CLIP-T score and 256.6 better FID). This suggests that CLIP embeddings are more suitable for interleaved visual generation compared to VAE’s latent space. This is reasonable, as SDXL-VAE is not aligned with language and lacks semantics.

|Method Autoencoder Style VL Aligned. Recon. Ability|CLIP-T FID<br><br>|
|---|---|
|SDXL-VAE Variational U-Net ✗ High<br><br>EMU-2 CLIP-Diffusion ✓ Medium SEED-X CLIP-Diffusion ✓ Low|13.2 286.6<br><br>25.4 76.7 25.1 30.1<br><br>|

Table 13. Visual latent spaces for visual regression. The VAE latent space is challenging for auto-regressive models to regress in a single step due to its limited correlation with language. In contrast, the language-aligned latent spaces (EMU-2 and SEED-X) allow for easier and effective regression in an interleaved manner.

Original Image SEED-X CLIP-Diffusion EMU-2 CLIP-Diffusion SDXL-VAE

|[Figure 108]|[Figure 109]|[Figure 110]|[Figure 111]|
|---|---|---|---|

|[Figure 112]|[Figure 113]|[Figure 114]|[Figure 115]|
|---|---|---|---|

- Figure 16. Auto-encoded results with different latent spaces. While SEED-X and EMU-2 both use a CLIP vision encoder and a diffusion model (i.e. finetuned SDXL) as decoder for autoencoding visual latents, SEED-X is semantic-biased and EMU-2 keeps much more visual details. SDXL-VAE shows the best image reconstruction ability, however, the latent space is not aligned with language (i.e. without pretraining on image-text pairs like CLIP).

|[Figure 116]|[Figure 117]|[Figure 118]|[Figure 119]|[Figure 120]|[Figure 121]|
|---|---|---|---|---|---|
|[Figure 122]|[Figure 123]|[Figure 124]|[Figure 125]|[Figure 126]|[Figure 127]|

Original Image Reconstruction + Noise (0.5 Std) +Noise (1.0 Std) Rescale (× 0.5) Rescale (× 2.0)

SEED-XEMU-2

- Figure 17. Both Scale and Direction Matters. We experiment with pseudo regression errors by altering latent direction and scale using Gaussian noise and scaling factors. Reconstruction results confirm that preserving both scale and direction is important for latent regression.

#### G. Generated Video Examples

Figures 18 and 19 present two examples of long narrative video generation for cooking “Fried Chicken” and “Shish Kabob,” illustrated step-by-step. The generation process begins with our interleaved auto-regressive director, which generates keyframe visual embeddings and their corresponding captions. These embeddings and captions are then used as conditions for the video generation model, which produces high-quality video clips that effectively narrate the cooking process and emphasize the crucial “action” information. The resulting video clips demonstrate excellent performance in capturing the step-by-step cooking instructions. All video clips are also included in the supplementary materials for further review.

[Figure 128]

[Figure 129]

(a) Action: Add raw chicken pieces and seasoning to a bowl of flour. (b) Action: Mix yogurt or buttermilk with seasoning in a bowl.

[Figure 130]

[Figure 131]

(c) Action: Dip chicken pieces into the batter to coat evenly. (d) Action: Coat the battered chicken in the flour mixture.

[Figure 132]

[Figure 133]

(e) Action: Fry the coated chicken in hot oil until crispy and golden. (f) Action: Sprinkle seasoning on the fried chicken and serve.

- Figure 18. Video generation example. Our pipeline effectively accomplishes long narrative video generation by producing six essential steps (i.e., video clips) for cooking ”Fried Chicken.” It delivers a clear, structured, and instructional step-by-step narrative, showcasing the model’s capability to generate coherent and comprehensive videos.

[Figure 134]

[Figure 135]

(a) Action: Mix chopped vegetables in a glass bowl. (b) Action: Add seasoning to the mixture of chopped vegetables.

[Figure 136]

[Figure 137]

(c) Action: Thoroughly mix the seasoned vegetable mixture. (d) Action: Add chicken pieces to vegetable and chicken mixture.

[Figure 138]

[Figure 139]

(e) Action: Brush oil onto the skewered chicken and vegetable kebabs. (f) Action: Place the prepared chicken and vegetable kebabs onto a grill.

[Figure 140]

[Figure 141]

(g) Action: Drizzle olive oil over the chicken and vegetable kebabs. (h) Action: Check on the grilling skewered chicken and vegetable kebabs.

- Figure 19. Video generation example. Our pipeline successfully generates eight crucial steps (i.e., video clips) to prepare the dish ”Shish Kabob.” This showcases a clear, structured, and instructional step-by-step narrative, demonstrating the model’s capability to produce coherent and comprehensive video content.

#### H. Limitations

###### H.1. Noisy “Actions" from ASR

While our CookGen dataset provides high-quality visual and contextual annotations, the action annotations derived from automatic speech recognition (ASR) have notable limitations. ASR-generated text often contains noise, resulting in action descriptions that are incomplete, ambiguous, or not directly informative for capturing the crucial steps in cooking processes. For instance, in Figure 8(a), the action annotation “Hi everyone, this one’s called rainbow broken glass jello” offers little value for understanding the cooking process, while another annotation in Figure 8(b) “Now normally when you make jello you use two cups of boiling water” provides vague guidance without specific details about the method. Such noisy annotations fail to align with the detailed and instructive nature of cooking instructions, which require precision and clarity to guide long narrative video generation effectively. This limitation underscores the importance of refining action annotations to improve their informativeness and utility for modeling cooking tasks.

###### H.2. Failure Cases While our method generates high-quality long narrative videos, there are instances where the model fails to produce meaningful cooking steps, and the rendered video clips contain unrealistic or irrelevant content due to hallucination.

Auto-regressive Director: Repeated “Steps”. Figure 20 illustrates a failure case where the auto-regressive director repeatedly generates similar visual embeddings, resulting in redundant and uninformative cooking video clips. For example, in the provided frames, the generated steps involve repeatedly cutting the salmon fillet, which adds little value to the narrative and fails to progress meaningfully. This issue is a known limitation of auto-regressive models, often caused by a lack of diversity in the embedding generation process. A potential solution is to introduce penalties for repeated embeddings or add constraints to encourage greater variability in visual outputs.

[Figure 142]

[Figure 143]

(a) Action: Cutting away the salmon fillet from the backbone (b) Action: Slicing the salmon fillet into even pieces

Figure 20. Failure Case. Auto-regressive model could generate repeated “Steps”, which is not informative to viewer.

Video Generation Model: Unrealistic Hallucination. Unrealistic hallucination occurs when a video generation model produces content inconsistent with the intended narrative. In Figure 21(a), the action ”placing the fried chicken into an oven set to preheat” is misrepresented as frying chicken in a pan, with an unrealistic increase in the quantity of chicken, showing a lack of object continuity. In Figure 21(b), the action ”adding a drizzle of sauce to a plate of grilled skewers” introduces an illogical appearance of new grilled food items, deviating from the intended action and disrupting narrative coherence.

[Figure 144]

[Figure 145]

(a) Action: Placing the fried chicken into a oven set to preheat (b) Action: Adding a drizzle of sauce to a plate of grilled skewers

Figure 21. Failure Case. Video generation model could make unrealisc hallucination to generate things from “air”.

