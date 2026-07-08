## Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks

Miran Heo1,2,†, Min-Hung Chen1, De-An Huang1, Sifei Liu1, Subhashree Radhakrishnan1 Seon Joo Kim2, Yu-Chiang Frank Wang1,3, Ryo Hachiuma1 1NVIDIA, 2Yonsei University, 3National Taiwan University https://miranheo.github.io/omni-rgpt

# arXiv:2501.08326v2[cs.CV]22Mar2025

###### Image: Region-level Captioning

|[Figure 1]<br><br>[0]<br><br>[1]<br><br>[2]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

USER: Briefly describe the content of [1].

Omni-RGPT: A hand reaching out to touch a deer.

Image: Region-level QA

USER: Why [1] is reaching out to [0]?

Omni-RGPT: The person in the image is reaching out to touch the deer's head, likely to interact with or pet the animal. This interaction could be a part of a wildlife encounter, where people get the opportunity to observe and engage with animals in their natural habitat. The person might be curious about the deer or simply enjoying the experience of being close to the animal. It is important to note that such interactions should be done with caution and respect for the animal's well-being … (omitted)

Video: Region-level Captioning Video: Region-level QA

USER: What is the region [2] doing? Omni-RGPT: The deer is lying down on the ground,

USER: Provide a comprehensive description of the area [1].

Omni-RGPT: The person is positioned on the right side of appearing relaxed and stationary.

the frames, interacting with a group of deer. They are wearing a light blue shirt and have their hair styled in a braid. The person is holding a branch with green leaves, which they use to feed the deer. The background is a lush, green forest with tall trees and dense foliage, indicating a natural, outdoor setting. The person's movements are gentle and deliberate, suggesting a calm and respectful interaction with the animals.

USER: Why are [0] and [1] facing each other?

Omni-RGPT: They are facing each other because the girl is interacting with the deer, possibly feeding or petting it.

USER: What will happen to [0] next?

Omni-RGPT: The deer will likely lower its head to nibble on the leaves.

Figure 1. Representative demo examples of Omni-RGPT. We introduce a unified multimodal large language model that integrates region-level understanding for both images and videos. Given user-defined localized region inputs (boxes or masks) accompanied by a corresponding text prompt, Omni-RGPT generates responses tailored to the visual context of each region for both images and videos.1

#### Abstract

#### 1. Introduction

Multimodal large language models (MLLMs) have evolved to interpret visual elements, progressing from text prompts for holistic image understanding to sophisticated approaches for region-level understanding. To achieve interactive region-specific comprehension in images, recent methods employ various strategies to represent target regions: encoding textual box coordinates within the text tokens [6, 7, 49, 50, 60, 68, 91, 94], utilizing visual RoI features [19, 20, 42, 69, 84, 87], or applying visual markers [5, 77]. Extending these capabilities to the video domain, some approaches [64, 79] incorporate initial frame bounding box coordinates as a textual form for the regionlevel video understanding tasks. Nonetheless, a general approach that effectively addresses region-specific tasks across both image and video remains an open challenge.

We present Omni-RGPT, a multimodal large language model designed to facilitate region-level comprehension for both images and videos. To achieve consistent region representation across spatio-temporal dimensions, we introduce Token Mark, a set of tokens highlighting the target regions within the visual feature space. These tokens are directly embedded into spatial regions using region prompts (e.g., boxes or masks) and simultaneously incorporated into the text prompt to specify the target, establishing a direct connection between visual and text tokens. To further support robust video understanding without requiring tracklets, we introduce an auxiliary task that guides Token Mark by leveraging the consistency of the tokens, enabling stable region interpretation across the video. Additionally, we introduce a large-scale region-level video instruction dataset (RegVID300k). Omni-RGPT achieves state-of-the-art results on image and video-based commonsense reasoning benchmarks while showing strong performance in captioning and referring expression comprehension tasks.

One key challenge in developing a solution is achieving scalability for video sequences. Since videos can contain a large number of frames, approaches that rely on bounding box coordinates as textual input face scaling limitations, as input region tokens increase linearly with the number of

1Our model refers to the girl in the video as “they” as a singular term to alleviate ethical issues instead of “she”, similar to ChatGPT [48].

† Miran Heo was an intern at NVIDIA during the project.

frames. RoI-based methods [19, 87] also encounter this issue, as they require repeated extraction of visual features from spatial regions (see Fig. 2 (a)). Relying on a single frame (e.g., the initial frame) as an alternative [64, 79] is also suboptimal, as it lacks a robust reference for the target across subsequent frames.

###### LLM LLM

Visual tokens Text tokens Visual tokens Text tokens

Vis. Enc. Pool Vis. Enc.

scattered

Sample

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Another challenge is addressing the temporal drift issue. There is no standardized method for unifying the multiple vectors representing the same object across different frames (e.g., bounding boxes in each frame) into a single, consistent vector. Unlike in static images, this issue becomes particularly problematic in videos, as target objects often change in appearance across frames due to motion, scale shifts, and perspective changes. Consequently, merging RoI features into a single representation can introduce inconsistencies, resulting in a loss of essential visual details. We identify that a key limitation of previous methods [19, 87] is the reliance on representations that may not consistently capture regions across frames, particularly when aiming for a unified solution for both images and videos.

(a) RoI Feature [19] (b) Token Mark (Ours)

Figure 2. Method comparison. (a) RoI-based methods generate visual region prompts using RoI-aligned visual features, potentially leading to temporal drift in the visual features of the target object in the video domain. (b) In contrast, our Token Mark is assigned to the corresponding region, preserving a consistent spatio-temporal target reference.

dent on large-scale data [38, 43], the research community lacks video instruction data for region-level understanding. Therefore, we introduce a large-scale, diverse, and finegrained Region-level Video Instruction Dataset (RegVID300k). The dataset includes 98k unique videos, with 214k regions curated from 10 public video datasets and 294k region-level instruction samples. We present an automated pipeline for curating the large-scale region-level video instruction samples based on powerful GPT4o [48].

In this paper, we present Omni-RGPT, a region-level MLLM designed for both images and videos. At the core of our framework is Token Mark, a novel region representation that enables seamless region-level understanding across both inputs (Fig. 2 (b)). Our key insight is to invert the traditional approach: rather than generating region embeddings from visual features, we predefine a set of tokens, using them as markers to identify regions within the latent space. Given visual-text inputs paired with target region prompts (e.g., boxes or masks), we sample a token mark and embed it within the spatial location defined by the region prompt. This embedding is further injected into the corresponding text prompt, allowing the LLM to directly reason the alignment between visual regions and text prompts. This approach effectively addresses two key challenges: 1) Scalability—since each target has a unique representation shared across frames, the number of input text tokens remains independent of the number of frames, and 2) Temporal drift—representing each target as a token ensures consistent reference across frames.

We demonstrate the capabilities of Omni-RGPT across both image and video inputs on diverse region-specific comprehension tasks, including visual commonsense reasoning, captioning, and referring expression comprehension. Notably, we achieve state-of-the-art performance on challenging commonsense reasoning benchmarks, excelling in image-based (VCR [85]) and video-based (CausalVidQA [30]) tasks. Supported by our newly introduced RegVID-300k dataset, Omni-RGPT also exhibits enhanced capabilities for video captioning. Additionally, it performs strongly in region-level understanding tasks in images, such as captioning and referring expression comprehension.

#### 2. Related Work

##### 2.1. Vision-Language MLLMs

Recent advancements in vision-language MLLMs such as GPT-4V [46] and Gemini [59], along with opensource counterparts like LLaVA [38], VILA [37], and QwenVL [1], have demonstrated robust capabilities in interpreting visual inputs. These models leverage multimodal capabilities to bridge visual and linguistic information, enhancing performance on tasks requiring visual comprehension. Building on these advancements, recent efforts have aimed to extend these models to video understanding [31, 35, 43, 71, 75, 86, 89]. However, while current approaches address global visual comprehension, achieving fine-grained, region-specific understanding in both images and videos remains an open challenge. This limitation is particularly evident in interactive settings, where accurate interpretation of localized visual details is essential.

Building on Token Mark, we introduce Temporal Region Guide Head, an auxiliary head specifically designed for video input to address the limitations of tracking-dependent region prompts (i.e., tracklets), which are often impractical in real-world applications. Using the region prompt from the initial frame, this auxiliary head operates on the LLM’s output visual tokens, classifying each visual token according to its assigned token marks. The representation of Token Mark supports effective region guidance during training, enabling robust and consistent region understanding across frames during inference without the need for full tracklets and additional cost (see Fig. 1).

Although the capabilities of MLLMs are heavily depen-

[Figure 9]

[Figure 10]

Output: Response

- (0, 0) (0, 1)
- (1, 0) (1, 1)

The person is moving towards the animal

to engage in a friendly interaction.

Temporal Region Guide Head (Training only)

LLM

|𝑟1|
|---|

|𝑟2|
|---|

Ƹ Ƹ

|0.0|
|---|
|0.1|
|0.0|
|0.0<br><br>|
|0.9|

|0.0|
|---|
|0.0|
|0.0|
|0.8<br><br>|
|0.2|

|0.0|
|---|
|0.5|
|0.0|
|0.0<br><br>|
|0.5|

|0.0|
|---|
|0.0|
|0.0|
|0.7|
|0.3|

|𝑉෠1|
|---|

|moving towards to| | |
|---|---|---|
| |𝑅෠<br><br>|Input:|

Why is <region1> <region2>?

|𝑆መ|
|---|

𝑉1

𝑉2 𝑉3 𝑉4

t: Text prompt

shared Projection Layer

Projector

Projector

Down-sampler

Visual Encoder

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Temporal Region Guide Head

|𝑚𝟏|
|---|

|𝑚𝟐|
|---|

𝑟2 𝑟1

Token Mark

LLM

| |
|---|

| |
|---|

𝐹 (0,0) (0,1) (1,0) (1,1) 𝑟1

𝑅 𝑟2

|𝑉>1|
|---|

Input: Video 𝑋 + first frame region prompt 𝑚

𝑆

Random Sampling

(a) Overview of Omni-RGPT (b) Auxiliary head

- Figure 3. (a) Overview: Omni-RGPT enables region-level understanding across image and video inputs. Given region prompts (e.g. boxes or masks) in a single image or the initial frame of a video, we assign Token Mark — a set of vectors serving as spatio-temporal region indicators — to the region. These vectors are embedded into the spatial region localized by the region prompt and directly injected into both visual and text prompts to indicate the target. (b) Auxiliary Head: We further introduce Temporal Region Guide Head to achieve robust region understanding in videos without relying on tracklet prompts. Building on Token Mark’s consistent representation of target objects across frames, this auxiliary task classifies the target Token Mark for visual tokens in subsequent frames.

##### 2.2. Region-level MLLMs

Image-based. Early methods employ numerical bounding box coordinates as textual prompts to specify visual regions within the LLM’s input [6, 7, 26, 49, 50, 60, 68, 91, 94]. By encoding region-level cues as text, these approaches facilitate seamless integration with global vision-language MLLMs. Instead of directly using bounding box coordinates as text, KOSMOS-2 [49] introduces location tokens, which embed these coordinates as discretized representations.

Another line of work leverages RoI-aligned visual features derived from region prompts (e.g., bounding boxes or masks) [19, 42, 70, 87]. ASM [70] introduces a locationaware image tokenizer, a transformer decoder module that uses RoI features as query inputs to generate region-level tokens for LLMs. Similarly, VisionLLM [67] and RegionBLIP [93] apply transformer decoders to derive regionspecific tokens. In contrast, GPT4RoI [87] and RegionGPT [19] directly embed extracted RoI features into region-assigned tokens within language prompts.

An alternative approach employs visual markers [5, 77]. SoM [77] offers a zero-shot solution using GPT-4V by overlaying masks with numbered markers on the image, which GPT-4V’s OCR capabilities interpret. While effective, this method may alter the image’s original appearance, potentially affecting details like color, and relies on rule-based algorithms to determine marker size and placement. In contrast, ViP-LLaVA [5] enables more interactive, user-defined prompts (e.g., free-form scribbles), allowing for a flexible and intuitive approach to region understanding.

Video-based. In contrast to the image domain, regionaware comprehension in the video domain remains relatively underexplored. Elysium [64] and Merlin [79] input

bounding box coordinates from the initial frame as text, enabling object tracking by predicting bounding box coordinates across the sequence. Additionally, Elysium introduces a new dataset; however, it provides only brief descriptions of dominant objects with limited temporal detail. MotionEpic [15] takes a different approach by using RoI information as nodes within a dynamic scene graph representation for videos. Initialized with frame-level proposals, they build a spatio-temporal scene graph through recurrent encoding. Unlike the above approaches, we present a novel approach that flexibly supports both box and mask representations for videos without relying on off-the-shelf tracking methods. Additionally, we introduce the instruction dataset to enhance region-level understanding capability.

#### 3. Omni-RGPT

We visualize our architecture in Fig. 3. We build on the core design of LLaVA [38], where the input image or video X ∈ RT×3×H

0×W0 (with T = 1 for images) is processed by a vision encoder f(·), producing visual features. Through a projection layer, these visual features are then projected into visual tokens V ∈ RT×D×H×W, where D is the input dimension of LLM. The visual tokens are then processed by the LLM FLLM(·) with a text prompt, which enables joint reasoning across textual and visual modalities.

Our objective is to enable the model to understand specific visual elements in response to an input text prompt by incorporating N input region prompts {mi}Ni=1, where each mi ∈ {0,1}H

0×W0 defines a target region (e.g., bounding box or mask). These region prompts, corresponding to a special token <region> as a placeholder in the text prompt, serve to identify and infer designated areas across the spatio-temporal dimension.

At a high-level, we have a set of tokens (Token Mark), which can be thought of as different paint colors on a palette. We randomly select a color to represent each target specified by the region prompt. In Fig. 3 (a), two tokens are chosen to represent the “person” and “animal”, respectively. This color is then applied to both visual and text token prompts. For visual tokens, we create a blank canvas and apply the selected color to the specified regions, overlaying this colored canvas onto the visual tokens. For text tokens, we replace the target placeholder (e.g., <region>) with its assigned token. Through this process, the model learns “where to look” during training by internalizing the predefined palette. In Sec. 3.1, we provide the concrete formulation of Token Mark, and introduce an auxiliary head (Fig. 3 (b)) specifically tailored for video input in Sec. 3.2.

##### 3.1. Token Mark

We define Token Mark as a set of tokens F ∈ RN

F×C, where NF is the total number of tokens and C denotes the feature dimension. To represent a region using Token Mark, we uniformly sample N indices from [NF] without replacement, obtaining the set of tokens R = {ri}Ni=1. Each sampled token ri is then matched one-to-one with corresponding region prompt mi so that the i-th Token Mark aligns with the i-th region prompt. These tokens serve as spatio-temporal region indicators and are injected into the language-side input for the associated visual content. Specifically, we project the Token Mark directly into the word embedding space using a linear layer: Rˆ = Fproj(R) ∈ RN×D.

To associate the sampled Token Mark ri with its corresponding region mi, we embed the tokens into the relevant pixels defined by the region prompts. Specifically the Spatial Token Mark S ∈ RC×H

0×W0 at each pixel location (h,w) is computed as:

N i=1 mi,h,w · ri

, (1)

S:,h,w =

ϵ + Ni=1 mi,h,w

where ϵ is a small positive constant added to prevent division by zero when no masks are active at position (h,w).

Next, we then downscale S to match the shape of the visual tokens V by applying adaptive average pooling, resulting in the updated Spatial Token Mark S˜. We then project it into the same feature space as Rˆ using the shared projection layer, yielding Sˆ = Fproj(S˜) ∈ RD×H×W. Finally, we integrate the spatial region-specific information into the visual tokens: Vˆ = V + Sˆ.

The proposed approach offers several key advantages. i) Preventing temporal drift: By encoding the target region as unique representation shared across frames, our method ensures consistent region assignments throughout video sequences. This consistency distinguishes our approach from

RoI-based methods, where representations of target objects often vary across frames. ii) Direct region-language connection: Projecting Token Mark directly within the word embedding space enables efficient modeling of regionlanguage relationships. Unlike methods that rely on textual descriptions for each region, our approach facilitates seamless user interaction without additional textual input for the region. iii) Preserving vision-language global alignment: By incorporating region information as residual features, our architecture retains alignment with the base image-text pair multimodal framework (e.g., LLaVA). In cases without region prompts, the model functions identically to the base architecture.

##### 3.2. Temporal Region Guide Head

For video input, an auxiliary head (Fig. 3 (b)) is introduced during training to enhance region consistency across frames, ensuring an accurate representation of regions even when a region prompt is provided for only the first frame. This auxiliary head classifies the corresponding Token Mark for each visual token, implicitly guiding the model to understand the target region without relying on explicit video object correspondence from tracklets.

Let Vt represent the visual tokens at the t-th frame, forming a sequence of visual tokens for the entire video, denoted as Vvid = (Vˆ1,V2,...,VT), where Vˆ1 contains the target region information. The sequence Vvid is then processed by the language model, which aims to generate region-aware predictions for the entire video sequence.

The auxiliary classification head Faux performs as:

Faux(FLLM(Vvid)) ∈ RT×H×W×(N

F+1), (2)

where NF + 1 is classification categories (the NF Token Mark and the background).

Since the visual tokens are downscaled from the original input resolution, multiple Token Marks may exist within a single visual token. To handle this, we apply soft-label classification, assigning each token a soft-label distribution over the NF +1 categories to reflect the proportion of each token belonging to multiple regions or the background.

Loss Function. The final loss is defined as L = LLLM + αLaux, where α balances the contribution of the auxiliary classification loss. The language model loss, LLLM, is computed as the cross-entropy loss between the predicted tokens and the ground truth tokens. Meanwhile, the auxiliary classification loss, Laux, is defined as the cross-entropy loss between the predicted soft-label distributions and the ground truth soft-label distributions for each visual token. This region guide head is used only during training and does not introduce additional latency during inference.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

- [0] Cat

- [1] Cat

- [2] Hand

2

0

2

0

2

0

1 1

0

1

1

###### Stage 1: Region-level Caption Stage 2: Hallucination Mitigation Stage 3: Instruction Generation

▶ Type 1: region-level brief description USER: Provide a short account of the region [2] . ASSISTANT: A person’s hand pets a cat on a bench, creating a calm park atmosphere.

###### ▶ Closed-Ended QAs for [0] :

[0] : The cat is partially visible on a wooden bench, sitting upright and looking towards the camera. It is positioned near a white post that divides the sitting area. In the background, there are lush green trees and a paved path, suggesting a park setting. The cat seems relaxed, occasionally glancing around its environment ... (omitted)

- Q1: Is the cat partially visible on a wooden bench? Select from (A): Deﬁnitely Yes, (B): Possibly Yes, (C): Possibly No, (D): Deﬁnitely No, (E): Not sure.
- Q2: Is the cat sitting upright? Select from ...
- Q3: Is the cat looking towards the camera? Select from ...

- ▶ Type 2: region-level detailed description USER: Provide a detailed account of the object located at [1] in the video. ASSISTANT: The cat is lying on a wooden bench, fully stretched out with its head resting on its front paws. The cat seems content, ...(omitted)

- ▶ Type 3: region-level conversation USER: What is the cat next to in [0] ? ASSISTANT: The cat is positioned near a white post that divides the sitting area.

- [1] : The cat is lying on a wooden bench, fully stretched out with its head resting on its front paws. The cat seems content, ... (omitted)

- [2] : The person’s hand is reaching out to pet the cat on the bench. Only the arm is visible, extending towards the cat and oﬀering gentle strokes along the cat’s back and head ... (omitted)

###### ▶ Reﬁned Caption for [0] :

The cat is partially visible on a wooden bench, sitting upright and looking towards the camera. It is positioned near a white post that divides the sitting area. In the background, there are lush green trees and a paved path, suggesting a park setting ... (omitted)

- Figure 4. Overview of our instruction sample generation pipeline. From a video with region masklets and nouns, the region-level captions, which contain contextual and temporal information about regions, are generated from GPT4o (left). Then, the hallucinations in the captions are mitigated (middle). Lastly, the instruction samples that cover diverse aspects of the regions are generated (right).

dataset are i) large-scale: our dataset consists of 98k unique videos, 214k tracklets or masklets, and 294k instructions, such as region-level detailed captioning, conversations, ii) diverse: the videos are collected from 10 public datasets used in different tasks, iii) fine-grained QAs: each region is described within about 60 words, including contextual and temporal information of the regions, resulting in diverse instruction samples, and iv) high-fidelity: the visual hallucinations in detailed captions are mitigated.

Caption Length

Video Sources

Instruction Samples

Datasets Videos Regions

Vid-STG [90] 7.8k 44k 10.8 1 HC-STVG [58] 12k 12k 17.3 1 LV-VIS [63] 4.8k 26k 1 1 BenSMOT [34] 3.3k 7.7k 39.9 1 Elysium [64] 1.2M 1.2M 2.2 1 -

V-ChatGPT [43] 13k - - 1 100k Valley [41] 65k - - 1 65k VCG [44] 6.2k - - 1 112k Ours 98k 214k 59.7 10 294k

Data Collection. The videos are collected from 10 public datasets [12, 23, 27, 55, 58, 63, 64, 80, 90, 92] that contain annotated regions (e.g., masklets, tracklets, or a single frame bounding box) along with nouns.

- Table 1. Comparison of our RegVID-300k with existing video datasets. Our dataset is the first region-level video instruction dataset that contains diverse video sources with detailed captions.

##### 4.1. GPT4o-Assisted Region-level Captions

#### 4. Region-level Video Instruction Dataset

From paired videos and masklets of regions2, we adapt the visual prompting technique of SoM [77] to overlay object masks with region indices at the center of each mask for every frame in the video. We then input the SoMprocessed videos into GPT4o [48], requesting enriched captions by including contextual and temporal information of each masklet from nouns in text prompts, such as “Generate the detailed description of [0]: cat, [1]: cat, [2]: hand”.

Overview. Since there is no existing region-level video instruction dataset, we propose RegVID-300k to enhance MLLMs’ dialog capability and obtain accurate responses about the regions in the videos (Fig. 4). Inspired by GPT4Vassisted detailed captioning [8, 9, 42] and caption-guided instruction dataset [38, 42, 43], our approach consists of three-steps, i) GPT4o-assisted region-level detailed captioning, ii) visual hallucination mitigation, and iii) captionguided region-level instruction sample generation. Detailed information (e.g., full video sources and GPT4o text prompts) are shown in the supplementary material.

##### 4.2. Visual Hallucination Mitigation

We mitigate the visual hallucination in the generated captions to improve the fidelity. Although the region-level captions generated by GPT4o contain fine-grained information, the synthetically generated detailed captions contain visual hallucinations [21, 22], and it is crucial to mitigate these to generate high-fidelity instruction samples.

Key Characteristics. Our dataset statistics and comparison with existing region-level video caption [34, 58, 63, 64, 90] and common video instruction datasets [41, 43, 44] are summarized in Tab. 1. For example, although Elysium [64] contains 1.2M videos, the region captions are only a single or few words (e.g., nouns, phrases), and the videos are collected only from WebVid [2]. The key characteristics of our

2For datasets with only tracklets or a single bounding box, we apply SAM [24] or SAM2 [54] to generate masklets.

Category

Acc@P Acc@C

Models LLM size

Acc@All input A R AR A R AR

Acc@D Acc@E

VGT† [74] - ✓ 70.8 70.3 55.2 56.9 38.4 61.0 59.3 42.0 55.4 TranSTR [33] - ✓ 73.6 75.8 65.1 65.0 48.9 68.6 65.3 50.3 62.2

Video-LLaMA‡ [86] 7B ✓ 69.2 71.0 63.6 62.4 44.4 65.4 60.1 45.0 57.4 VideoChat‡ [31] 7B ✓ 72.9 73.9 65.2 63.1 45.9 66.0 62.7 45.8 59.6 Video-ChatGPT‡ [43] 7B ✓ 73.1 75.1 66.0 63.9 46.0 67.8 63.6 50.0 61.1 Video-LLaVA‡ [36] 7B ✓ 73.7 74.4 67.6 65.4 47.7 68.0 64.9 51.5 61.8 MotionEpic [15] 7B ✓ 81.2 83.0 74.3 73.7 54.7 74.5 73.8 58.6 69.4

Omni-RGPT 7B 84.0 84.6 84.2 85.4 76.9 74.7 74.0 64.3 77.5 Omni-RGPT 13B 84.5 85.1 85.5 85.6 78.0 76.7 75.3 67.5 78.8

- Table 2. Accuracy (%) on Causal-VidQA validation set, it consists of four sub-tasks: Description (D), Explanation (E), Prediction (P), and Counterfactual Reasoning (C), each presented as multiple-choice questions with an answer (A) component. Prediction and Counterfactual tasks additionally include a reason (R) choice. †: Reproduced by TranSTR, ‡: Reproduced by MotionEpic.

Inspired by VFC [17], we apply multi-stage visual hallucination mitigation using LLMs and MLLMs. First, we decompose detailed region-level captions into multiple closedended questions that ask about the contents in the captions using LLMs. Then, we input these questions into MLLMs along with videos to validate whether the content is correct. In the third stage, we gather the questions not verified in the previous step and ask LLMs to remove the unverified contents in the original captions and re-generate them.

- 4.3. GPT-Assited Region-level Instruction Data

In the final step, building on the concept of caption-guided instruct-tuning data construction [38, 42, 43, 94], we further process the captions to generate region-level video instructions. We utilize text-only GPT4 [47] to create regionspecific question-answer pairs from the detailed captions, addressing various aspects of the captions. The samples include detailed descriptions, summaries, and general QAs for the specific regions. We provide a few in-context examples to enhance the quality of sample generation. The generated instructions cover both contextual (e.g., color, spatial positions) and temporal aspects (e.g., motions, actions).

- 5. Experiments

- 5.1. Implementation Details

We leverage Llama-2 [61] as the language model, CLIPViT-L [52] as the vision encoder, and a two-layer MLP for the projection layer. For Token Mark generation, we set the number of Token Mark (NF) to 100 with C = 256 and employ a single linear layer as the projection layer. The input image is resized to 336 × 336. To embed the spatial Token Mark at the same resolution as the image tokens, we apply adaptive average pooling with a window size of 24. Additionally, a single linear layer is applied for the auxiliary classifier, with Laux coefficient α set to 0.05. For video data, we uniformly sample four frames.

Image Pre-training. We adopt the pretraining recipe of RegionGPT [19], which utilizes a global image-text paired dataset in conjunction with an image-region dataset.

Joint Fine-tuning. During fine-tuning, we employ a joint image-video training strategy. We include image-text paired datasets [10, 38] and image region-level dataset [19]. Additionally, we incorporate Visual Commonsense Reasoning (VCR) [85], which combines recognition and cognitivelevel understanding of complex scenes through a multiplechoice question-answering (QA) format.

For the region-level video dataset, we reformulate annotations from publicly available datasets, including VidSTG [90], Causal-VidQA [30], and ReVOS [76]. We use captions paired with regions from Vid-STG and ReVOS to create region-level video captioning and Referring Expression Comprehension (REC)-style prompts. Our RegVID300k is also included to enhance the model’s capacity to generate rich, detailed object captions. In this stage, we train the model for one epoch, freezing only the vision encoder while allowing all other parameters to be updated.

##### 5.2. Quantitative Evaluation

We report the performance of our method on both image and video domains. All results are evaluated using the 7B language model with joint image-video trained weights unless stated otherwise.

Region-level Video QA. In Tab. 2, we present our performance on the Causal-VidQA [30] benchmark, designed to evaluate video question answering with a focus on causal reasoning across temporal and spatial dimensions. CausalVidQA comprises video clips paired with questions that assess the model’s ability to interpret events, interactions, and object relationships, requiring not only frame-level recognition but also temporal reasoning across sequences.

Our approach achieves state-of-the-art performance across all sub-tasks with a substantial margin, demonstrating robust region-level video reasoning. In particular, our

Models Vid-STG Extended-Elysium Elysium BensMOT

ViP-LLaVA [5] 9.9 3.1 10.8 7.0 LLaVA-OV [28] 4.9 12.1 9.6 8.9 Groma [42] 10.5 2.1 13.1 6.0 RegionGPT [19] 10.4 14.4 13.0 14.0 Elysium [64] 7.2 0.7 19.1 1.1

Omni-RGPT 14.2 19.3 9.3 14.6

###### Table 3. Performance on the region-level video captioning task. METEOR metric is reported.

Model Q → A QA → R Q → AR

ViLBERT [40] 72.4 74.5 54.0 Unicoder-VL [29] 72.6 74.5 54.5 VLBERT-L [57] 75.5 77.9 58.9 ERNIE-ViL-L [81] 78.5 83.4 65.8 VILLA-L [16] 78.5 82.6 65.2 GPT4RoI [87] 87.4 89.6 78.6 ViP-LLaVA [5] 87.7 89.8 78.9

Omni-RGPT 88.5 90.1 79.9

###### Table 4. Accuracy (%) on VCR validation set. VCR includes three accuracy measures: (Q → A) selects the correct answer, (QA

→ R) selects the correct rationale given the correct answer, and (Q

→ AR) chooses both the correct answer and rationale.

method achieves a significant performance margin in the Prediction (P) task, emphasizing advanced temporal reasoning capabilities. Additionally, we observe substantial improvements in the “AR” accuracy, which requires accurate reasoning for both answer and reason selections, underscoring the capability to justify its choices coherently. Compared to MotionEpic [15], which relies on frame-level proposals from every frame and an iterative scene graph encoding process, our method achieves strong performance without requiring complete tracklet inputs or complex relational encoding modules. Notably, we do not use the region’s category as input, further emphasizing the effectiveness of direct region-language connection.

Region-level Video Captioning. We evaluate the regionlevel video captioning capability in Tab. 3. For comparison, we include the zero-shot performance of image-based MLLMs [5, 19, 28, 42], using only the first frame as input. Since public benchmarks (Vid-STG [90] and Elysium [64]) offer brief descriptions with limited temporal context (see Tab. 1), we additionally generate detailed region-level captions on Elysium (Extended-Elysium) using GPT-4o [48]. Moreover, to evaluate the zero-shot captioning performance, we employ the BenSMOT dataset [34], which includes relatively detailed region-level video captions (40 words). Our method demonstrates strong performance across datasets, except Elysium. We attribute this to the nature of Elysium’s captions, which are typically brief nouns without motion-related information, favoring finetuned [64] or image-specialized methods [5, 19, 28, 42].

RefCOCOg Visual Genome

Models

METEOR CIDEr METEOR CIDEr

GRiT [72]‡ 15.2 71.6 17.1 142.0 ControlMLLM [73] 14.0 59.8 - SLR [83] 15.9 66.2 - Kosmos-2 [49] 14.1 62.3 - GLaMM [53] 15.7 104.0 17.0 127.0 OMG-LLaVA [88] 15.3 - - ViP-LLaVA [5] 16.6 105.9 - Groma [42] 16.8 107.3 16.8† 137.4† RegionGPT [19] 16.9 109.9 17.0 145.6

Omni-RGPT 17.0 109.7 17.0 139.3

- Table 5. Performance on region-level image captioning on RefCOCOg and Visual Genome, † denotes re-experimented results on the full evaluation set, ‡ denotes the region caption specialists.

GPT4o [48] RegionGPT [19] Groma [42] Omni-RGPT

val 74.08 86.44 88.19 95.99 test 72.23 86.96 88.38 95.88

- Table 6. Accuracy (%) of REC on RefCOCOg val and test set.

Region-level Image QA. We report our model’s performance on Visual Commonsense Reasoning (VCR) [85] validation set (Tab. 4). VCR is designed to assess a model’s capability in commonsense reasoning within visual contexts, with each sample consisting of questions (Q) and multiple-choice options for both the answer (A) and rationale (R). Each question requires the model to select the correct answer and provide a rationale for supporting that answer. Following the protocols of GPT4RoI [87] and ViPLLaVA [5], the model is finetuned on VCR.

Our approach achieves state-of-the-art performance across all metrics, outperforming methods specifically designed for image-based tasks. This highlights our method’s effectiveness in both image and video contexts using a unified architecture. Furthermore, the proposed Token Mark facilitates a robust region-level connection between text prompts and visual context, enabling reasoning over complex, multi-region relationships.

Region-level Image Captioning. We evaluate our regionlevel image captioning performance on RefCOCOg [82] and Visual Genome [25] in Tab. 5. Following prior works, we report METEOR [3] and CIDEr [62] scores to measure captioning quality. Note that Groma [42] reported results on a subset of the Visual Genome validation set; accordingly, we re-evaluated results on the complete set to ensure consistency with prior studies. Across both datasets, our approach achieves comparable scores to those of methods specifically designed for the image domain, demonstrating its effectiveness at region-level captioning.

Image REC. To assess the region-level natural language referring capability, we adopt the REC evaluation protocol of RegionGPT [19]. For a concrete comparison, we evaluate GPT4o [48], Groma [42], and ours using the same region

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

1

RegionPrompt

InputVideo+

0

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

|0|
|---|

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

|1|
|---|

###### Figure 5. Heatmap of Temporal Region Guide Head outputs.

Temporal Region Guide Head Extended-Elysium BenSMOT

✓ 19.3 14.6 16.2 12.2

Table 7. Ablation study of temporal region guide head on video region-level captioning task.

proposals [19]. As shown in Tab. 6, our approach achieves strong performance with a significant margin. The results demonstrate that our method establishes a robust and precise region-language connection by projecting token marks directly into visual features to indicate regions.

##### 5.3. Visualization Analysis

In Fig. 5, we analyze our Temporal Region Guide Head on the example shown in Fig. 1 sourced from [45]. The first row displays input video frames, with region prompts and corresponding numbers overlaid on the initial frame for reference3. We then assign randomly sampled Token Mark to each region prompt (deer and hand) in the first frame. In the second and third rows, we visualize the output probabilities of the assigned Token Mark from the Region Guide Head as a heatmap. In the first frame, where the regions are inputted as masks, the outputs accurately predict the region prompts. In subsequent frames, even without the region prompt, the model successfully predicts the corresponding region by leveraging visual correlations across frames.

##### 5.4. Ablation Study

Temporal Region Guide Head. We report the performance of our model trained without the auxiliary task in Tab. 7. The results demonstrate that incorporating the auxiliary task further enhances the detailed object captions in videos.

Number of Input Frames. We study the impact of the number of input frames on Causal-VidQA (Tab. 8). The average accuracy across sub-tasks gradually decreases as the number of frames is reduced. When comparing four and single-frame inputs, scene description (D) and explanation (E) show a modest accuracy. In contrast, prediction and counterfactual tasks (P and C) exhibit a substantial decline, highlighting the importance of temporal information.

3Note that these masks and annotations are added for illustration only, and original image frames are used as input for the vision encoder.

Input Frames

D (A) E (A) P (AR) C (AR) All 4 84.0 84.6 76.9 64.3 77.5

3 84.1+0.1 84.5−0.1 76.1−0.8 63.9−0.4 77.2−0.3 2 83.3−0.7 84.5−0.1 75.3−1.6 62.2−2.1 76.3−1.2 1 82.7−1.3 82.8−1.8 71.7 −5.2 59.9−4.4 74.3−3.2

- Table 8. Number of input frames ablation study on CausalVidQA dataset. We also show the performance gap from the four frames input.

Vision encoder Downscale Token size mAP Acc CLIP-ViT-L-336 ✗ 24×24 68.2 78.6 SigLip-SO400M-384 Pixel unshuffle 14×14 59.7 70.0

- Table 9. Vision encoding methods ablation study on COCO classification.

Vision Encoding Method. In Tab. 9, we investigate the impact of vision encoding methods inspired by recent conventions. Our region classification results on the COCO dataset indicate that reducing the resolution of image tokens significantly diminishes region-level understanding. This result aligns with recent studies in image-based MLLMs, which benefit from larger token sizes [39, 56].

#### 6. Limitation

As shown in our ablation studies, robust region-level video understanding requires dense spatial and temporal visual information. However, similar to existing region-level video understanding models [64, 79], achieving high fidelity in both dimensions is challenging within our current framework. Using a four-frame video input may not fully capture the complexity of diverse, real-world scenarios. Extending region-level understanding to accommodate long-form videos is a promising direction for future research.

#### 7. Conclusion

We present Omni-RGPT, a multimodal LLM with regionspecific comprehension capabilities for both images and videos. The main idea is to use Token Mark, a set of tokens that bridges the language and spatio-temporal visual tokens at region-level. For robust region understanding in videos, we introduce an auxiliary task tailored for video inputs. To further enhance detailed regiondescription capabilities in videos, we introduce RegVID300k, a new dataset re-annotated with GPT-4o on publicly sourced videos. With its intuitive design, Omni-RGPT surpasses existing methods on challenging visual commonsense reasoning benchmarks in both image-based (VCR) and video-based (Causal-VidQA) tasks. Additionally, it demonstrates strong performance on diverse region-level understanding tasks, including captioning and referring expression comprehension.

Acknowledgements. We would like to thank Qiushan Guo for assistance in setting up the project. We also appreciate the NVIDIA VILA team for the efforts in developing a robust framework that greatly facilitated our research.

#### References

- [1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 2
- [2] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In CVPR, 2021. 5
- [3] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In ACL Workshop, 2005. 7
- [4] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset, 2022. 1
- [5] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Vipllava: Making large multimodal models understand arbitrary visual prompts. In CVPR, 2024. 1, 3, 7
- [6] Gongwei Chen, Leyang Shen, Rui Shao, Xiang Deng, and Liqiang Nie. Lion: Empowering multimodal large language model with dual-level visual knowledge. In CVPR, 2024. 1, 3
- [7] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 1, 3
- [8] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In ECCV, 2024. 5
- [9] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. In NeurIPS, 2024. 5
- [10] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instructionfinetuned language models. In JMLR, 2024. 6
- [11] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In NeurIPS, 2023. 2, 3
- [12] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. MeViS: A large-scale benchmark for video segmentation with motion expressions. In ICCV, 2023. 5, 3, 4
- [13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is

- worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 4
- [14] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 4

- [15] Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, Meishan Zhang, Mong-Li Lee, and Wynne Hsu. Video-of-thought: Step-by-step video reasoning from perception to cognition. In ICML, 2024. 3, 6, 7
- [16] Zhe Gan, Yen-Chun Chen, Linjie Li, Chen Zhu, Yu Cheng, and Jingjing Liu. Large-scale adversarial training for visionand-language representation learning. In NeurIPS, 2020. 7
- [17] Yunhao Ge, Xiaohui Zeng, Jacob Samuel Huffman, TsungYi Lin, Ming-Yu Liu, and Yin Cui. Visual fact checker: Enabling high-fidelity detailed caption generation. In CVPR,

2024. 6

- [18] Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790, 2023. 3
- [19] Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei Liu. Regiongpt: Towards region understanding vision language model. In CVPR, 2024. 1, 2, 3, 6, 7, 8
- [20] Junwen He, Yifan Wang, Lijun Wang, Huchuan Lu, Jun-Yan He, Jin-Peng Lan, Bin Luo, and Xuansong Xie. Multi-modal instruction tuned llms with fine-grained visual perception. In CVPR, 2024. 1
- [21] Yusuke Hirota, Ryo Hachiuma, Chao-Han Huck Yang, and Yuta Nakashima. From descriptive richness to bias: Unveiling the dark side of generative image caption enrichment. In EMNLP, 2024. 5
- [22] Wen Huang, Hongbin Liu, Minxin Guo, and Neil Zhenqiang Gong. Visual hallucinations of multi-modal large language models. In ACL, 2024. 5
- [23] Jingwei Ji, Ranjay Krishna, Li Fei-Fei, and Juan Carlos Niebles. Action genome: Actions as compositions of spatiotemporal scene graphs. In CVPR, 2020. 5, 4, 14
- [24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything. In ICCV, 2023. 5, 4
- [25] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. In IJCV, 2017. 7, 1
- [26] Byung-Kwan Lee, Beomchan Park, Chae Won Kim, and Yong Man Ro. CoLLaVO: Crayon large language and vision mOdel. In ACL, 2024. 3
- [27] Ang Li, Meghana Thotakuri, David A. Ross, Jo˜ao Carreira, Alexander Vostrikov, and Andrew Zisserman. The ava-kinetics localized human actions video dataset. arXiv preprint arXiv:2005.00214, 2020. 5, 3, 4

- [28] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 7
- [29] Gen Li, Nan Duan, Yuejian Fang, Ming Gong, Daxin Jiang, and Ming Zhou. Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training. In AAAI, 2020. 7
- [30] Jiangtong Li, Li Niu, and Liqing Zhang. From representation to reasoning: Towards both evidence and commonsense reasoning for video question-answering. In CVPR, 2022. 2, 6
- [31] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 2, 6
- [32] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. EMNLP, 2023. 2
- [33] Yicong Li, Junbin Xiao, Chun Feng, Xiang Wang, and TatSeng Chua. Discovering spatio-temporal rationales for video question answering. In ICCV, 2023. 6
- [34] Yunhao Li, Hao Wang, Xue Ma, Jiali Yao, Shaohua Dong, Heng Fan, and Libo Zhang. Beyond mot: Semantic multiobject tracking. In ECCV, 2024. 5, 7
- [35] Zhaowei Li, Qi Xu, Dong Zhang, Hang Song, YiQing Cai, Qi Qi, Ran Zhou, Junting Pan, Zefeng Li, Vu Tu, Zhida Huang, and Tao Wang. GroundingGPT: Language enhanced multi-modal grounding model. In ACL, 2024. 2
- [36] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In EMNLP, 2024. 6
- [37] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In CVPR, 2024. 2, 1
- [38] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2, 3, 5, 6, 4
- [39] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 8
- [40] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In NeurIPS, 2019. 7
- [41] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. arXiv preprint arXiv:2306.07207, 2023. 5
- [42] Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, and Xiaojuan Qi. Groma: Localized visual tokenization for grounding multimodal large language models. In ECCV, 2024. 1, 3, 5, 6, 7, 2
- [43] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In ACL,

2024. 2, 5, 6

- [44] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Videogpt+: Integrating image and video

- encoders for enhanced video understanding. arXiv preprint arXiv:2406.09418, 2024. 5
- [45] Jiaxu Miao, Xiaohan Wang, Yu Wu, Wei Li, Xu Zhang, Yunchao Wei, and Yi Yang. Large-scale video panoptic segmentation in the wild: A benchmark. In CVPR, 2022. 8, 2
- [46] OpenAI. GPT-4V System Card, 2023. 2
- [47] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2024. 6, 3, 4
- [48] OpenAI. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 1, 2, 5, 7
- [49] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Qixiang Ye, and Furu Wei. Grounding multimodal large language models to the world. In ICLR,

2024. 1, 3, 7

- [50] Shraman Pramanick, Guangxing Han, Rui Hou, Sayan Nag, Ser-Nam Lim, Nicolas Ballas, Qifan Wang, Rama Chellappa, and Amjad Almahairi. Jack of all tasks master of many: Designing general-purpose coarse-to-fine visionlanguage model. In CVPR, 2024. 1, 3
- [51] Jiyang Qi, Yan Gao, Yao Hu, Xinggang Wang, Xiaoyu Liu, Xiang Bai, Serge Belongie, Alan Yuille, Philip HS Torr, and Song Bai. Occluded video instance segmentation: A benchmark. In IJCV, 2022. 3
- [52] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 6
- [53] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M. Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S. Khan. Glamm: Pixel grounding large multimodal model. In CVPR,

2024. 7

- [54] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 5, 4
- [55] Seonguk Seo, Joon-Young Lee, and Bohyung Han. Urvos: Unified referring video object segmentation network with a large-scale benchmark. In ECCV, 2020. 5, 4
- [56] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, Bryan Catanzaro, Andrew Tao, Jan Kautz, Zhiding Yu, and Guilin Liu. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv:2408.15998, 2024. 8
- [57] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visuallinguistic representations. In ICLR, 2020. 7
- [58] Zongheng Tang, Yue Liao, Si Liu, Guanbin Li, Xiaojie Jin, Hongxu Jiang, Qian Yu, and Dong Xu. Human-centric spatio-temporal video grounding with visual transformers. IEEE Trans. on Circuits and Systems for Video Technology, 32(12):8238–8249, 2021. 5, 4

- [59] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2
- [60] Yunjie Tian, Tianren Ma, Lingxi Xie, Jihao Qiu, Xi Tang, Yuan Zhang, Jianbin Jiao, Qi Tian, and Qixiang Ye. Chatterbox: Multi-round multimodal referring and grounding. arXiv preprint arXiv:2401.13307, 2024. 1, 3
- [61] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 6, 4
- [62] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In CVPR, 2015. 7
- [63] Haochen Wang, Cilin Yan, Shuai Wang, Xiaolong Jiang, XU Tang, Yao Hu, Weidi Xie, and Efstratios Gavves. Towards open-vocabulary video instance segmentation. In ICCV,

2023. 5, 4

- [64] Han Wang, Yongjie Ye, Yanjie Wang, Yuxiang Nie, and Can Huang. Elysium: Exploring object-level perception in videos via mllm. In ECCV, 2024. 1, 2, 3, 5, 7, 8, 4
- [65] Jiaqi Wang, Pan Zhang, Tao Chu, Yuhang Cao, Yujie Zhou, Tong Wu, Bin Wang, Conghui He, and Dahua Lin. V3det: Vast vocabulary visual detection dataset. In ICCV, 2023. 1
- [66] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 4
- [67] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. Visionllm: Large language model is also an openended decoder for vision-centric tasks. In NeurIPS, 2023. 3
- [68] Weiyun Wang, Yiming Ren, Haowen Luo, Tiantong Li, Chenxiang Yan, Zhe Chen, Wenhai Wang, Qingyun Li, Lewei Lu, Xizhou Zhu, et al. The all-seeing project v2: Towards general relation comprehension of the open world. In ECCV, 2024. 1, 3
- [69] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, Yushi Chen, Tong Lu, Jifeng Dai, and Yu Qiao. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. In ICLR, 2024. 1
- [70] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. In ICLR, 2024. 3
- [71] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 2

- [72] Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang. Grit: A generative region-to-text transformer for object understanding. In ECCV, 2024. 7
- [73] Mingrui Wu, Xinyue Cai, Jiayi Ji, Jiale Li, Oucheng Huang, Gen Luo, Hao Fei, Xiaoshuai Sun, and Rongrong Ji. Controlmllm: Training-free visual prompt learning for multimodal large language models. In NeurIPS, 2024. 7
- [74] Junbin Xiao, Pan Zhou, Tat-Seng Chua, and Shuicheng Yan. Video graph transformer for video question answering. In ECCV, 2022. 6
- [75] Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. Slowfast-llava: A strong training-free baseline for video large language models. arXiv preprint arXiv:2407.15841, 2024. 2
- [76] Cilin Yan, Haochen Wang, Shilin Yan, Xiaolong Jiang, Yao Hu, Guoliang Kang, Weidi Xie, and Efstratios Gavves. Visa: Reasoning video object segmentation via large language models. In ECCV, 2024. 6
- [77] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023. 1, 3, 5, 4
- [78] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023. 3
- [79] En Yu, Liang Zhao, Yana Wei, Jinrong Yang, Dongming Wu, Lingyu Kong, Haoran Wei, Tiancai Wang, Zheng Ge, Xiangyu Zhang, et al. Merlin: Empowering multimodal llms with foresight minds. In ECCV, 2024. 1, 2, 3, 8
- [80] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. Bdd100k: A diverse driving dataset for heterogeneous multitask learning. In CVPR, 2020. 5, 4, 13
- [81] Fei Yu, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. Ernie-vil: Knowledge enhanced visionlanguage representations through scene graph. In AAAI,

2021. 7

- [82] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In ECCV, 2016. 7, 1, 2
- [83] Licheng Yu, Hao Tan, Mohit Bansal, and Tamara L. Berg. A joint speaker-listener-reinforcer model for referring expressions. In CVPR, 2017. 7
- [84] Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. Osprey: Pixel understanding with visual instruction tuning. In CVPR, 2024. 1
- [85] Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In CVPR, 2019. 2, 6, 7, 1
- [86] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In EMNLP, 2023. 2, 6

- [87] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Yu Liu, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on regionof-interest. arXiv preprint arXiv:2307.03601, 2023. 1, 2, 3, 7
- [88] Tao Zhang, Xiangtai Li, Hao Fei, Haobo Yuan, Shengqiong Wu, Shunping Ji, Chen Change Loy, and Shuicheng Yan. Omg-llava: Bridging image-level, object-level, pixel-level reasoning and understanding. In NeurIPS, 2024. 7
- [89] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. 2
- [90] Zhu Zhang, Zhou Zhao, Yang Zhao, Qi Wang, Huasheng Liu, and Lianli Gao. Where does it exist: Spatio-temporal video grounding for multi-form sentences. In CVPR, 2020. 5, 6, 7, 4
- [91] Liang Zhao, En Yu, Zheng Ge, Jinrong Yang, Haoran Wei, Hongyu Zhou, Jianjian Sun, Yuang Peng, Runpei Dong, Chunrui Han, and Xiangyu Zhang. Chatspot: Bootstrapping multimodal llms via precise referring instruction tuning. In IJCAI, 2024. 1, 3
- [92] Luowei Zhou, Yannis Kalantidis, Xinlei Chen, Jason J Corso, and Marcus Rohrbach. Grounded video description. In CVPR, 2019. 5, 3, 4
- [93] Qiang Zhou, Chaohui Yu, Shaofeng Zhang, Sitong Wu, Zhibing Wang, and Fan Wang. Regionblip: A unified multimodal pre-training framework for holistic and regional comprehension. arXiv preprint arXiv:2308.02299, 2023. 3
- [94] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In ICLR, 2024. 1, 3, 6
- [95] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billion-scale corpus of images interleaved with text. In NeurIPS, 2023. 1

## Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks

### Supplementary Material

In this supplementary material, we provide additional insights and details to support the main paper. In Sec. 8, we outline the implementation details, including dataset setups and training strategies. Sec. 9 presents extended quantitative results, offering further validation of our method’s performance. To provide a more comprehensive understanding, Sec. 10 includes a set of qualitative results, showcasing visual examples that demonstrate our model’s effectiveness in diverse scenarios. Sec. 11 provides a visualization analysis, including an exploration of the limitations of our model. In Sec. 12, we offer additional details about our newly constructed dataset, RegVID-300k. Finally, in Sec. 13, we discuss important ethical considerations.

#### 8. Implementation Details

We further provide detailed setups for the image dataset. Our training consists of two stages: image pre-training, followed by image-video joint fine-tuning, which incorporates our newly proposed region-level video instruction dataset, RegVID-300k. For pretraining, we utilize a imagetext paired dataset alongside an image-region pre-training dataset as proposed in RegionGPT [19]. Specifically, following VILA [37], we use interleaved image-text data [95] and conventional image-text pairs [4]. The image-region pre-training dataset includes Visual Genome [25], RefCOCOg [82], and V3Det [65]. For the image region-level dataset in the joint fine-tuning stage, we utilize the RegionGPT’s fine-tuning dataset, which includes the ReCapD, a high-quality, GPT-assisted region-aware image dataset.

During pre-training, all learnable parameters, except those in the visual encoder and the language model, are trained for one epoch. In the fine-tuning stage, the model is trained for one epoch, with the vision encoder frozen while allowing all other parameters to be updated. We use a batch size of 16 and a learning rate of 5 × 10−5. The entire training process, including both pre-training and finetuning stages, is completed within 24 hours using 8 nodes of 8×A100 GPUs.

#### 9. More Quantitative Results

##### 9.1. VCR

- In Tab. 10, we present evaluation results on VCR [85] validation set under various settings, including category name usage, different region prompt representations (e.g., bounding boxes and masks), and dataset-specific fine-tuning. The first row in the table corresponds to the results presented in

Fine-tuning Category Region Q → A QA → R Q → AR

✓ ✓ mask 88.5 90.1 79.9 ✓ mask 88.2 90.1 79.8 ✓ bbox 87.7 90.0 79.2

bbox 85.8 87.5 75.3

Table 10. Accuracy (%) on VCR with different settings.

the main paper, following the same evaluation protocol as existing methods [5, 87].

Category Name Usage. Manually specifying category names for each target can be inefficient in realworld, user-interactive applications. For instance, consider the question “Why is <region1> moving towards <region2>?” paired with region prompts that include category names, such as <region1>:person and <region2>:animal. Incorporating these categories transforms the question into, “Why is person <region1> moving towards animal <region2>?”

In the second row, we evaluate our model’s performance without utilizing such category names for the provided region prompts. While the results show a slight decrease in performance, they highlight that Omni-RGPT internalizes the semantic knowledge of regions within the question. This is achieved by leveraging visual context through a direct region-language connection using Token Mark.

Notably, unlike RoI-based methods [19, 87], which inject explicit visual features into the text prompt to embed semantic knowledge, Omni-RGPT accomplishes this by utilizing randomly sampled Token Mark within the text prompt. This highlights its ability to generalize and infer semantics directly from the visual context itself.

Region Prompt Representation. VCR provides two types of localization information for targets: bounding box coordinates and polygon-style segmentations. In the third row, we report the result using bounding box coordinates as the region prompt representation, which offers a less finegrained target representation compared to masks. The overall performance shows a marginal decrease compared to the results obtained using masks, highlighting the flexibility of our framework in accommodating dual region prompt types. Task-specific Fine-tuning. In the last row, we evaluate the performance of Omni-RGPT without VCR-specific finetuning. We use joint image-video trained weights, maintaining consistency with the settings used for other quantitative results presented in the main paper. The results here demonstrate that Omni-RGPT retains competitive perfor-

RegVID -300k

Causal-VidQA Extended D (A) E (A) P (AR) C (AR) -Elysium

###### ✓ 84.0 84.6 76.9 64.3 19.3

82.7−1.3 84.4−0.2 76.9 63.5−0.8 13.5−5.8

Table 11. Effectiveness of RegVID-300k. Accuracy for CausalVidQA and METOR scores for Extended-Elysium are reported.

mance even without task-specific fine-tuning, showcasing its robustness and broader applicability.

##### 9.2. Effectiveness of RegVID-300k

- In Tab. 11, we report the effectiveness of RegVID-300k, our new region-level video instruction dataset, on CausalVidQA [30] and Extended-Elysium. The first row is the results reported in the main paper, obtained using the finetuned weight that incorporates the new dataset. In contrast, the second row shows results where the model is trained under the same settings but without including the new dataset, relying solely on annotations from public video datasets. The results underscore the effectiveness of RegVID-300k, as evidenced by the performance degradation, including the Description (D) task within Causal-VidQA (−1.3) and Extended-Elysium (−5.8). In particular, the large performance drop in Extended-Elysium highlights the contribution of our new dataset’s diverse and detailed descriptions, which enhance the model’s region-level captioning capability in the video domain.

##### 9.3. Hallucination

Following the POPE [32] evaluation protocol, we report object hallucination evaluation results in Tab. 12. Performance is evaluated on the MSCOCO dataset under three settings: Random, Popular, and Adversarial. In the Random setting, our method achieves the best Accuracy and a competitive F1 Score, trailing the best-performing model (InstructBLIP [11]) by only −0.47. In the Popular setting, our Accuracy is marginally below the best-performing model (RegionGPT) by −0.57 while achieving the highest F1 Score among all methods. Finally, in the Adversarial setting, both Accuracy and F1 scores are lower than RegionGPT, with differences of −4.0 and −1.97, respectively. Overall, our method demonstrates competitive performance compared to the image-specialized counterparts.

#### 10. More Qualitative Results

We showcase visualizations of Omni-RGPT’s region-level understanding capabilities across both image and video scenarios. For all examples, we use a jointly fine-tuned model using the 7B language model. The inputs, including mask proposals, are sourced from the VIPSeg [45], which is not included in the training dataset (see Tab. 13).

##### 10.1. Video Region-level Understanding

Brief Region-level Captioning. We present Omni-RGPT’s brief region-level video captioning capability in Tab. 14. The results show that Omni-RGPT effectively generates concise captions for the given prompts, incorporating visual context across the entire sequence. For example, in the second case, the caption for the red backpack (<region0>) notes that it is unzipped, a detail observable from the third frame.

Detailed Region-level Captioning. Tab. 15 visualizes examples of detailed region-level video captioning. Key information is highlighted in light orange color. The outputs effectively capture transitions in the target’s location and motion across the sequence. For instance, in the first case, Omni-RGPT describes the pillow (<region0>) as initially lying flat, later being lifted and adjusted by the person.

Region-level QA. In Tab. 16, we present various examples of region-level question-answering in videos. OmniRGPT generates answers to diverse questions, including those about multi-object relationships and the location of a target within a sequence (e.g., see the second example).

##### 10.2. Image Region-level Understanding

Region-level Captioning. In Tab. 17, we demonstrate Omni-RGPT’s captioning capabilities, including both brief and detailed descriptions. The results highlight the model’s ability to describe specific details about the target object (e.g., the names of instruments), including its visual features such as color, material, and size.

Region-level QA. In Tab. 18, we present a question-andanswering example that reflects a visual commonsense reasoning scenario. In this example, Omni-RGPT successfully generates answers by considering the commonsense context within the scene. Specifically, the output recognizes that wearing a helmet in an office is unusual, explains why, and suggests multiple possible reasons for such a case.

Referring Expression Comprehension (REC). In Tab. 19 and Tab. 20, we provide a visualization comparison with existing image-specialized methods [19, 42] on REC task using the RefCOCOg [82]. Among multiple proposals in the image, Omni-RGPT accurately identifies the most suitable one for the given caption. As shown in the examples, the challenge in this task lies in the presence of multiple objects with similar appearances in the scene, requiring a precise understanding of the spatial location of each visual element.

Our understanding of Omni-RGPT’s robust REC capacity is that Token Mark establishes a strong ability to refer to visual tokens at the pixel level. This enables better differentiation between visually similar objects compared to RoI-based methods, which primarily rely on similarity using averaged appearances to locate the target region prompt.

POPE Metrics Omni-RGPT RegionGPT [19] Shikra [7] InstructBLIP [11] MiniGPT4 [94] LLaVA [38] MM-GPT [18] mPLUG-Owl [78]

Accuracy (↑) 88.76 87.80 86.90 88.73 77.83 86.00 50.03 53.30 Precision (↑) 91.15 97.75 94.40 85.08 75.38 87.50 50.02 51.71 Recall (↑) 86.60 78.13 79.26 93.33 82.67 84.00 100.00 99.53 F1 Score (↑) 88.82 86.85 86.19 89.29 78.86 85.71 66.68 68.06 Yes 48.96 41.20 43.26 55.20 54.83 48.00 99.97 96.23

Random

Accuracy (↑) 86.63 87.20 83.97 81.37 68.30 76.67 50.00 50.63 Precision (↑) 86.20 95.44 87.55 75.07 64.27 72.22 50.00 50.32 Recall (↑) 86.60 78.13 79.20 93.33 82.40 86.67 100.00 99.27 F1 Score (↑) 86.40 85.92 83.16 84.35 72.21 78.79 66.67 66.79 Yes 50.23 40.93 45.23 62.57 64.10 60.00 100.00 98.63

Popular

Accuracy (↑) 81.67 85.67 83.10 74.37 66.60 73.33 50.00 50.67 Precision (↑) 78.82 91.99 85.60 67.67 62.45 69.02 50.00 50.34 Recall (↑) 86.60 78.13 79.60 93.33 83.27 84.67 100.00 99.33 F1 Score (↑) 82.53 84.50 82.49 78.45 71.37 66.32 66.67 66.82 Yes 54.93 42.47 46.50 68.97 66.67 61.33 100.00 98.67

Adversarial

Table 12. Results on the object hallucination benchmark using the POPE evaluation pipeline on MSCOCO.

#### 11. Visualization Analysis

Elysium dataset. As discussed in the main paper, we observe that the video region-level captioning dataset from Elysium [64] is limited in capturing the rich information within video sequences. In Tab. 21, we illustrate Elysium’s original inputs with ground-truth captions alongside responses generated by our model. From the examples, we note that: 1) the videos typically exhibit monotonous motion with limited dynamics, 2) the dataset primarily contains single dominant objects, which restricts its ability to evaluate multi-object scenarios, 3) the original ground truth captions are closer to noun-based descriptions rather than full captions. To facilitate better evaluation, we refined its annotations (referred to as Extended-Elysium in our main paper) using the same pipeline as our dataset curation.

Failure case. In Tab. 22, the first example shows that the current design occasionally exhibits unstable performance to understand small objects. Additionally, since we employ frame-independent visual encoding without explicit temporal embeddings, our model has limited capability in interpreting the direction of objects.

Occluded objects. In Tab. 23, we analyze a heatmap of the Temporal Region Guide Head in diverse scenarios of occluded objects from [51].

#### 12. RegVID-300k 12.1. Data Preparation

We select public datasets containing videos and manually annotated regions (e.g., masklets, tracklets, single-frame bounding boxes) along with their associated region nouns. The full list of data sources, number of videos, and tasks used in the original datasets is summarized in Tab. 13. Since each dataset inherently contains biases due to the dataset construction process—–such as video selection, annotation rules, or targeted scenes–—we diversify the source

30~60

1

0~30

>120

>7

- 5
- 6

90~120

2

4

60~90

3

(a) Number of regions

(b) Caption length

Figure 6. RegVID-300k statistics. (a) The dataset contains multiple regions in videos. (b) The captions primarily range from 30 to 60 words, resulting in diverse instruction-following samples.

datasets based on the targeted tasks. We use only a subset of the ElysiumTrack-1M [64] and AVA-Kinetics [27] datasets to diversify the video sources instead of increasing the videos within a single dataset. Because the original MeViS dataset [12] does not provide objects’ nouns, we generate them by using GPT-4 [47] to extract target nouns from annotated referring sentences for each object. Also, since the ActivityNet-Entities dataset [92] consists of relatively long frames and includes scene changes, we divide each video into shorter clips based on the ground-truth event segments. Detailed statistics are visualized in Fig. 6. Our dataset includes multiple regions within each video, with detailed descriptions for each region.

##### 12.2. Data Visualization

The QA instruction-following samples and their corresponding videos are presented in Tabs. 24 and 25. These visualizations demonstrate the automatic generation of diverse and high-quality question-and-answer pairs related to the regions within the videos. The automated, multi-stage approach used to generate these region-level instructionfollowing samples is detailed in the following sections.

Dataset Original Task Region Annotation Number of Used Videos MeViS [12]

1.6k Ref-YTVOS [55] 3.4k Vid-STG [90]

Referring video object segmentation masklet

5.3k HC-STVG [58] 10.1k BDD100k [80]

Spatio-Temporal Grounding tracklet

4.1k ElysiumTrack-1M [64] 10k LV-VIS [63] Open-vocabulary video instance segmentation masklet 3k ActivityNet-Entities [92] Dense Video Grounding bounding box 38.8k AVA-Kinetics [27] Action Localization bounding box 13.4k Action Genome [23] Action Graph bounding box 8.4k

Object Tracking tracklet

Table 13. Source video dataset summary table. We collect the videos and corresponding region annotations from diverse datasets.

##### 12.3. GPT4o-Assisted Region-level Captioning

Since not all datasets have annotated object masks per frame (i.e., masklets), we apply SAM [24] with ViT-H backbone [13] to the datasets that only contain the tracklets, such as Vid-STG [90], HC-STVG [58], BDD100k [80], and ElysiumTrack-1M [64]. Moreover, for the datasets that only contain the single frame bounding box, such as ActivityNetEntities [92], AVA-Kinetics [27], and Action Genome [23], we apply SAM2 [54] to propagate the annotated bounding box to before and after the annotated frames to obtain the masklets. We followed the official code of SoM [77] to overlay the masklets onto the image frame-by-frame. Due to the context length and the budget limitation, we randomly subsample 16 frames from the video and resize to 334×334 to input to GPT4o with the text prompt Tab. 26. We input SoM-processed videos along with the object nouns to GPT4o to obtain the region-level captions. We employ gpt-4o-2024-08-06 model in the paper.

##### 12.4. Visual Hallucination Mitigation

We employ LLaMA-3.1-8B [14] for the LLM in the first and third stages (decomposing the caption into questions and re-generating the caption) and Qwen2-VL-7B [66] for MLLMs for the second step (multi-modal fact-checking). In the first step, the multiple questions, which ask about the contents in the captions, are generated from the regionlevel caption using the text prompt Tab. 27. As mentioned in the text prompt, we query the LLM only to include the questions that ask the contents in the input captions. In the second step, we input the generated questions into the Qwen2-VL with the original videos to verify the contents. We ask Qwen2-VL to answer the multi-choice closed-ended questions (Tab. 28). We input 8 video frames due to the context length. We collect the questions whose answers are “(D) Definitely No” and input these questions with the original captions to remove the non-validated contents from the original captions and re-generate them (Tab. 29).

##### 12.5. Region-level Instruction Data Generation

From the refined region-level captions, we generate diverse instruction-following data. Three types of instructionfollowing data are generated: detailed descriptions, brief descriptions, and general conversations.

Region-level Detailed Description. We directly use the refined captions for each region as an answer. For the instructions, we augmented the input instruction prompt using GPT4 [47] from the original text prompt (Describe the region in detail), resulting in diverse input text prompts to enhance the LLM’s adaptability to users’ various text prompts (Tab. 33). For each region-level caption, we randomly select one of the instruction samples from the list to construct the question-and-answer pairs.

Region-level Brief Description. We employ GPT4 [47] to summarize the region-level detailed caption using the text prompt Tab. 30 and generate the region-level short descriptions. Similar to the region-level detailed description, we also diversify the input instruction from (Briefly describe the region), allowing users’ diverse input questions (Tab. 32), and the question is randomly selected from the list to construct the question-and-answer pairs for the region-level brief description task.

Region-level Conversation. Inspired by LLaVA [38], we generate diverse region-specific question-and-answer pairs from the region-level captions using GPT4. The prompt to GPT4 is shown in Tab. 31. Since the captions contain diverse aspects of the regions in the video, the generated questions also cover the diverse aspects of them.

#### 13. Ethics Concerns

We utilize the pre-trained weights of the LLM (Llama2 [61]) and fine-tune them using publicly available image and video datasets, along with our re-annotated datasets. Consequently, societal biases present in the original LLM model weights, which are derived from large-scale web text corpora, may be inherited in the output response.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

User: Give a brief interpretation of what stands out in <region0>. Omni-RGPT: The monitor is on a desk with a printer, indicating a workspace setup.

User: Provide a short summary of what’s visible in <region1>. Omni-RGPT: The printer is a sleek, modern device on a desk in a home office setting.

User: Describe the notable objects or subjects you can see in <region2>. Omni-RGPT: A person in a blue shirt operates a printer in a home office setting.

User: How would you quickly explain what is featured in <region3>? Omni-RGPT: The keyboard is on a desk near a printer, used for computer tasks.

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

User: Provide a concise rundown of what is visible in <region0>. Omni-RGPT: A red backpack is being unzipped and organized near a tent in a misty outdoor setting.

User: Describe the notable objects or subjects you can see in <region1>. Omni-RGPT: A person in a green jacket and brown pants is organizing a red backpack near a tent in a foggy outdoor setting.

User: What are the key points of focus in this section <region2>? Omni-RGPT: A red and gray tent is set up on a grassy area in a misty forest.

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

User: Create a short label for the visible content in <region0>. Omni-RGPT: A bicycle is being lifted onto a rack on a van in a parking lot.

User: Draft a short title for the video content enclosed by <region1>. Omni-RGPT: A person is securing a bicycle to a trailer attached to a van in a rural setting.

User: Provide a concise label for the highlighted <region2>. Omni-RGPT: A silver van is parked on a gravel surface with a bicycle attached to its rear.

Table 14. Qualitative visualization of brief region-level video captioning capability.

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

User: Give an extensive description of the area labeled <region0>, paying attention to textures, colors, movements, and notable details.

Omni-RGPT: The pillow is positioned on the top bunk of a bunk bed. It is initially lying flat, but as the sequence progresses, it is lifted and adjusted by the person. The pillow is white and appears soft, contrasting with the gray bedding. It is situated in a minimalist room with a wooden floor and white walls, providing a simple and clean backdrop.

User: Explain the visible details of the subject in <region1>, including its appearance, movements, and any distinguishing traits.

Omni-RGPT: The person is positioned on a bunk bed, initially lying on the top mattress. They are wearing a light-colored shirt and shorts, and appear to be adjusting their position. The room is minimalistic, with a plain wall and a few pieces of furniture, including a ladder leading to the top bunk. The person is seen moving from a reclined position to sitting up, then standing, indicating a transition from rest to activity. The setting suggests a casual, relaxed environment.

User: Explain the key attributes and features of the subject found in <region2> with in-depth detail. Omni-RGPT: The ladder is positioned on the right side of the room, attached to the bunk bed. It is metallic and extends upwards, providing access to the upper bunk. The ladder is stationary, serving as a functional piece of furniture in the room.

|[Figure 48]|
|---|

|[Figure 49]|
|---|

[Figure 50]

[Figure 51]

User: Create a thorough, descriptive caption for the portion of the video located in <region0>, touching on its materials, origin, and symbolic elements.

Omni-RGPT: The motorbike is a small dirt bike, designed for off-road racing. It is positioned on a dirt track, moving steadily across the uneven terrain. The bike is equipped with knobby tires for better traction on the loose surface. The rider is wearing protective gear, including a helmet and a red racing suit, indicating participation in a competitive event. The background shows a dirt field with several parked vehicles and spectators, suggesting a motocross event. The bike maintains a consistent speed, navigating the bumps and turns of the course.

User: Delve deeply into the description of the contents at <region1>, focusing on the interplay of textures, colors, movements, and standout aspects.

Omni-RGPT: The person is riding a small dirt bike, navigating through a rugged, dirt-covered terrain. They are wearing a red outfit and a helmet, indicating they are likely participating in a motocross event. The person is positioned centrally on the bike, maintaining balance as they move forward. The background shows a dirt track with several vehicles parked, suggesting a motorsport event or competition. The person appears focused, maneuvering the bike over uneven ground, showcasing their riding skills in this challenging environment.

Table 15. Qualitative visualization of detailed region-level video captioning capability. Specifically, the sentences highlighted in orange indicate that our model effectively captures the temporal motion (or dynamics) of the targeted objects.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

User: What is the setting of the scene? Omni-RGPT: The scene is set in a laboratory environment with a clean and organized workspace.

User: What objects are near <region0>? Omni-RGPT: The person is near a laboratory bench with various laboratory equipment, including a box of tissues, a beaker, and a bottle.

User: Could you explain what <region0> is doing in the scene? Omni-RGPT: The person is engaged in a laboratory experiment, handling a test tube filled with a pink liquid.

User: What is <region0> wearing, and what is it used for? Omni-RGPT: The person is wearing a white lab coat and purple gloves, which are used for protection in a laboratory setting.

User: What is the object <region0> is holding in their left hand and what is used for? Omni-RGPT: The person is holding a dropper in their left hand, which is used for precise liquid measurements.

User: What is the purpose of the object in region <region1>? Omni-RGPT: The beaker is used for mixing or measuring substances.

|[Figure 56]|
|---|

|[Figure 57]|
|---|

[Figure 58]

[Figure 59]

User: Describe the setting in the video. Omni-RGPT: The setting is a ballroom with a polished wooden floor, surrounded by an audience seated at tables, creating a formal and elegant atmosphere.

User: Is there any interaction visible between the regions <region0> and <region1>? Omni-RGPT: Yes, the couple is engaged in a synchronized dance routine, with the woman in a flowing dress and the man in a formal suit.

User: What can you say about the movement of the region <region0>? Omni-RGPT: The person is engaged in a graceful dance, moving fluidly across the dance floor.

User: Where is the region <region1> located in the frame at the beginning of the video? Omni-RGPT: The person is positioned towards the center of the dance floor.

User: Where is the region <region1> positioned in the frame at the end of the video? Omni-RGPT: The person is positioned towards the center-right of the frame.

Table 16. Qualitative result of region-level video QA.

[Figure 60]

###### • Brief captioning

User: Can you briefly explain what you see in the region <region0>? Omni-RGPT: A black and white accordion. User: What are the key features or subjects captured in this region <region1>? Omni-RGPT: A man in a white tank top and blue jean shorts playing a banjo.

###### • Detailed captioning

User: Zoom in on the area indicated by <re gion2> and describe every discernible detail, from texture and color to form and function.

Omni-RGPT: The banjo in the close-up is a large, round, and wooden instrument. It has a distinctive shape, with a curved body and a long neck. The banjo is held by a person in the group, who is likely playing it as part of the musical performance. The banjo’s color is brown, which is a common color for wooden instruments. The close-up view of the banjo allows for a detailed examination of its design and craftsmanship, showcasing its unique features and the skill of the musician playing it.

User: Craft a thorough narrative about the piece of the image highlighted by <region3>, from its aesthetic qualities to its possible historical context.

Omni-RGPT: The double bass in the close-up is a large, wooden instrument with a brown color. It has a distinctive shape, with a curved body and a long neck. The double bass is held by a person in the group, who is likely playing it as part of the musical performance. The instrument’s size and style suggest that it is a traditional, acoustic double bass, which is commonly used in jazz and classical music.

Table 17. Qualitative result of region-level image captioning.

[Figure 61]

User: Why <region0> is looking at <re gion1>?

Omni-RGPT: The man in the suit is looking at the man in the helmet, possibly because he is wearing a helmet in an office setting, which is an unusual sight. This could be a conversation starter or a point of interest for the man in the suit.

User: Please explain why <region1>’s outfit is unusual in office.

Omni-RGPT: The man’s outfit is unusual in the office because he is wearing a suit and a helmet. Typically, suits are associated with formal or professional attire, while helmets are more commonly worn for safety purposes, such as in construction or sports. Wearing a helmet in an office setting is not a common sight and can be considered unconventional or even humorous. This unusual combination of attire might draw attention and spark conversations among the people in the office.

User: What might be the purpose of <re gion1> wearing a helmet to the office?

Omni-RGPT: The man wearing a helmet to the office might be doing so for a variety of reasons. One possibility is that he is a cyclist or a motorcyclist who has just arrived at work and is wearing the helmet as a safety measure. Another possibility is that he is participating in a themed event or a costume party at the office, where wearing a helmet is part of the dress code. Alternatively, he could be wearing the helmet as a prop for a presentation or a demonstration, where the helmet serves a specific purpose related to his job or the topic of his presentation. It is essential to consider the context and the man’s role in the office to better understand the purpose behind his helmet.

Table 18. Qualitative result of region-level image QA.

###### Input:

[Figure 62]

Caption: the bottom right tray with a salad in it Omni-RGPT:

[Figure 63]

RegionGPT:

[Figure 64]

Groma:

[Figure 65]

Table 19. Qualitative comparison of image REC.

###### Input:

[Figure 66]

Caption: a mother giraffe nursing her young Omni-RGPT:

[Figure 67]

RegionGPT:

[Figure 68]

Groma:

[Figure 69]

Table 20. Qualitative comparison of image REC.

|[Figure 70]|
|---|

|[Figure 71]|
|---|

[Figure 72]

[Figure 73]

###### • Original Elysium

User: Can you briefly explain what you see in the <region>? Ground Truth: cattle Omni-RGPT: A black and white cow grazes peacefully in a serene, rural field.

###### • Extended-Elysium

User: Explain the key attributes and features of the subject found in <region> with in-depth detail. Ground truth: The cattle depicted in the series of frames is grazing steadily in a lush, green pasture. Positioned calmly in the foreground, it remains focused on eating the grass beneath. The vast field extends into the distance, bordered by a line of trees providing a natural barrier and creating a serene rural landscape. The bright, cloudy sky casts a gentle light across the meadow, highlighting the cattle’s black and white patches. Throughout the frames, the cattle maintains its position, engrossed in the act of feeding.

Omni-RGPT: The cow is positioned in the middle of a lush green field, surrounded by a backdrop of dense trees under a clear sky. The cow is grazing, with its head lowered towards the ground, indicating it is feeding on the grass. The cow’s body is mostly white with distinct black patches, and it appears to be calmly moving its head as it feeds. The field is expansive, with a few scattered trees and a fence line visible in the distance, suggesting a rural setting. The cow’s motion is gentle and unhurried, typical of a peaceful pastoral scene.

|[Figure 74]|
|---|

|[Figure 75]|
|---|

[Figure 76]

[Figure 77]

###### • Original Elysium

User: What short caption would best capture the details of <region>. Ground truth: flower Omni-RGPT: A white water lily with a yellow center blooms serenely on a pond surrounded by lily pads.

###### • Extended-Elysium

User: Offer a full description of the object found in <region>. Ground truth: The flower, likely a water lily, is prominently positioned atop broad, flat green leaves on the water’s surface. Its pristine white petals radiate outward from a vibrant yellow center, creating a striking contrast against the leafy backdrop. In the serene water setting, the flower remains still, exhibiting a delicate and serene presence. The surrounding green leaves occasionally display small imperfections, adding a natural, untouched feel to the scene. The overall environment is tranquil, exemplifying a typical pond ecosystem where such flowers thrive.

Omni-RGPT: The water lily is prominently positioned in the center of the frame, displaying its vibrant white petals that are slightly ruffled, suggesting a gentle breeze. The lily is surrounded by large, broad green leaves that float serenely on the water’s surface, providing a lush, natural backdrop. The water lily remains stationary, its petals radiating outward, creating a symmetrical and elegant appearance. The scene is set in a tranquil pond environment, with the water reflecting the surrounding greenery, enhancing the serene and picturesque atmosphere.

- Table 21. Visualized analysis of Elysium dataset. Top: The original Elysium annotations are short and predominantly noun-centric. Bottom: We present Extended-Elysium, refined annotations created using our dataset curation pipeline.

###### • Small object understanding

|[Figure 78]|
|---|

|[Figure 79]|
|---|

[Figure 80]

[Figure 81]

User: Please provide a short depiction of the content shown in <region0> in the video. (✓) Omni-RGPT: There is a white cup above a table in the office.

|[Figure 82]|
|---|

|[Figure 83]|
|---|

[Figure 84]

[Figure 85]

User: Provide a description of what is presented in the <region0>. (✗) Omni-RGPT: A stationary backpack is on the floor near a desk in a classroom.

###### • Direction understanding in video

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

User: Give an extensive description of the area labeled <region0>, paying attention to textures, colors, movements, and notable details.

(✗) Omni-RGPT: The cat is positioned on a staircase, appearing to be in motion as it climbs upwards. Its body is oriented towards the top of the stairs, suggesting it is either ascending or exploring the area. The setting is indoors, with a light-colored wall and ceiling visible in the background. The cat’s movement is fluid, indicating a sense of curiosity or playfulness. The scene is well-lit, with a window nearby, adding to the cozy and domestic atmosphere.

- Table 22. Failure Cases. Top: Omni-RGPT occasionally exhibits unstable performance on small objects. The model mistakenly identifes the black chair as a backpack in the second example. Bottom: Omni-RGPT shows limited capacity for understanding the direction of objects. The model answers that the cat is climbing upward, even though it is actually moving downward.

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

User: A short description of <region0> from the video: Omni-RGPT: There is a dog in white and black towards a child in gray.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

User: What stands out to you the most in this <region0>? Omni-RGPT: There is a cat hitting another cat on the floor.

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

User: What stands out to you the most in this <region0>? Omni-RGPT: A black horse gallops energetically through a snowy landscape.

Table 23. Heatmap visualization of Temporal Region Guide Head on occluded objects.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

User: Generate a long, detailed caption for the segment of the video at <region1>, covering aspects such as its origin, material, and any symbolic meaning.

Assistant: The car is a yellow taxi positioned on the right side of the road. It stays largely consistent in position, seemingly moving at a steady pace parallel to the dashcam car. The taxi’s presence remains constant, indicating it’s traveling in the same general direction as the surrounding traffic. The background suggests an urban environment, with multiple lanes and buildings visible, typical of a city street.

User: Expand upon the visual description of what is contained within the marked area <region2>, covering its appearance, size, and stylistic features.

Assistant: The bus is located ahead of the taxi and further up the road. Initially, it appears small due to its distance, but gradually becomes larger and more detailed as the dashcam car advances down the road. The bus is traveling in the same direction as the rest of the traffic and navigating through the urban setting. Close proximity to other vehicles and buildings in the scene are evident as they approach a more congested area.

User: Explain the key attributes and features of the subject found in <region3> with in-depth detail. Assistant: The car is a black sedan situated to the left of the bus, becoming visible later in the sequence of frames. Initially, it is further behind but gradually moves up to parallel with the dashcam car. The black sedan is traveling in the same direction as the taxi and bus within the lanes. It doesn’t exhibit any erratic movements, maintaining a steady course. The scene’s background remains consistently urban, with the overhead signage and multiple vehicles indicating a busy city road.

User: How would you briefly summarize the key points in <region1>? Assistant: A yellow taxi is traveling steadily alongside the dashcam car on a city street.

User: Describe what’s highlighted within <region2>. Assistant: A bus is ahead of the taxi and grows larger as the dashcam car approaches in urban traffic.

User: Write a short description for the region <region3>. Assistant: A black sedan becomes parallel with the dashcam car, moving steadily in city traffic.

User: What color is the taxi in region <region1>? Assistant: The taxi is yellow.

User: Where is the taxi positioned relative to the road? Assistant: The taxi is positioned on the right side of the road.

User: Is the yellow taxi moving erratically? Assistant: No, the taxi is moving at a steady pace parallel to the dashcam car.

User: Where is the bus located in relation to the taxi? Assistant: The bus is located ahead of the taxi and further up the road.

User: When does the black sedan in region <region3> become visible? Assistant: The black sedan becomes visible later in the sequence of frames.

- Table 24. Visualized instruction-following sample and the video in RegVID-300k. The object masks are overlayed onto the video frames. The video is from the BDD100k dataset [80].

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

User: Elaborate on the artwork shown in the region indicated by <region0>, mentioning its color, appearance, size, style, and any standout features.

Assistant: The book is held in the person’s left hand. It appears to be a soft-cover book, with the pages slightly visible as they move and adjust their grip on it. The person keeps the book close to their torso while standing near a table in a room with an orange wall and a wooden cabinet in the background. As they maneuver, the book remains prominently in hand, indicating it could be light and easy to handle.

User: Provide a detailed account of the object located at <region1> in the video. Assistant: The object, looking like a cup or bottle, is initially near the center of the person’s torso and gradually moves toward their mouth in the latter frames. This action suggests it might contain a beverage as they appear to be preparing to drink from it. The cup has a distinct rim, which contrasts against the orange wall, flanked by the cabinet on the left side of the frame.

User: Describe comprehensively the object at <region2>, including its physical traits, style, actions, and visible characteristics.

Assistant: The person is standing, wearing a green T-shirt and a beanie. They are constantly adjusting their position while holding a book in one hand and a cup or bottle in the other, showing a casual and relaxed demeanor. The scene is set in a room with a bold orange paint and a wooden cabinet with reflective glass, indicating a cozy indoor environment.

User: Write a short description for the region <region1>. Assistant: A cup or bottle moves towards the person’s mouth, suggesting a drink.

User: How would you briefly summarize the key points in <region2>? Assistant: A person in a green T-shirt holds a book and cup in a cozy room.

User: Describe the action being performed with the object in region <region1>. Assistant: The object, which looks like a cup or bottle, is gradually moving toward the person’s mouth, suggesting they are preparing to drink from it.

User: What is the clothing color of the person in region <region2>? Assistant: The person is wearing a green T-shirt.

User: Is there any motion involving the book in region <region0>? Assistant: Yes, the book slightly moves as the person adjusts their grip on it.

User: In what hand is the object in region <region1>? Assistant: The cup or bottle is in one hand, likely not the same where the book is held.

User: What is the backdrop environment in the entire video? Assistant: The environment is a room with an orange wall and a wooden cabinet with reflective glass, which gives a cozy indoor feel.

- Table 25. Visualized instruction-following sample and the video in RegVID-300k. The object masks are overlayed onto the video frames. The video is from the Action Genome dataset [23].

You are an intelligent video captioner designed to generate the description of objects. Your task is to generate detailed descriptions of the objects labeled with a bright numeric ID at the center of video frames.

##INSTRUCTIONS:

- - The description should start with the object noun. DO NOT add any irrelevant content to the answer. Each object should be described in around 90 words, but DO NOT include information that you are unaware of.
- - INCLUDE the following information in the object descriptions: (1) temporal (motion, action), (2) contextual (spatial position) information of objects, and (3) global (background, scene) information. If you cannot see the object clearly, output the following <object id>: None.
- - MUST NOT include the outline color information in the description. Use bracket [] to refer to the object IDs in the descriptions if needed.

These are video frames. Generate descriptions of each labeled object, [<id1>]: <noun1>, [<id2>]: <noun2>..., .

- Table 26. Input text prompts to GPT4o for the detailed region-level captioning. We input the above prompts and SoM-processed videos to obtain region-level captions. The placeholders <idX> and <nounX> are replaced with X-th object indices marked in the video and the corresponding object’s ground-truth noun, respectively.

Your task is to generate multiple questions comprehensively from given object descriptions in JSON format. The question should ask about the object from the given descriptions. The questions comprehensively ask all the contents in the object descriptions. DO NOT generate a question whose answer will be no. Respond only with valid JSON. Do not write an introduction or summary.

- Description: <ic sample1 caption>

- Question: {“questions”: [ <ic sample1 q1>, <ic sample1 q2>, ..., ] }

Description: <ic sample2 caption>

- Question: {“questions”: [ <ic sample2 q1>, <ic sample2 q2>, ..., ] }

Description: <caption> Question:

- Table 27. The text prompts to decompose the object-level to multiple fact-checking questions. To enhance the output quality, we manually create in-context pairs (input object description and output questions). The placeholder <ic sampleX caption> is replaced with the object descriptions for the X-th in-context samples, and <ic sampleX qY> is replaced with the manually annotated Y th question of Xth in-context sample. The target object caption is injected to the placeholder <caption>.

<Q> Answer from the following options: (A): Definitely Yes, (B): Possibly Yes, (C): Possibly No, (D): Definitely No, (E): Not sure. Do not write an introduction or summary. Only respond the letter.

- Table 28. The text prompt to input to the MLLM to verify the contents in the captions. The placeholder <Q> is replaced with the input questions generated during the previous stage.

Your task is to generate a refined description of the objects based on the given questions about the objects in a list format. The answer is always no, so remove that information from the object description and re-generate the refined description. Do not change the irrelevant content as much as possible. Do not write an introduction or summary, and only write the description in JSON format.

======= Description: <ic sample original caption> Questions: [ <ic sample q1>, <ic sample q2>, ..., ] Refined Description: <ic sample output caption>

====== Description: <original caption> Questions: [ <q1>, <q2>, ..., ] Refined Description:

- Table 29. The text prompt to refine the region-level captions. We collect the questions whose answers are “D” in Tab. 28 and prompt the LLM to remove the contents in the questions from the original caption. One in-context sample is provided to improve the output quality. The placeholders <ic sample original caption>, <ic sample qX>, <ic sample output caption> are replaced with the object-level caption, the X-th question which includes the non-verified content, and the manually annotated refined caption, respectively. The target caption and non-verified questions are injected to <original caption> and <qX> placeholders, respectively.

You are a helpful assistant in summarizing the long description of the object into a single sentence or a short description. Output the JSON format, for example, [<object ID>]: <description>, ... ##Instructions

- - DO NOT include any contents that have not appeared in the long descriptions.
- - The short description should briefly cover the abstracted contents of the long description.
- - The short description should be a single sentence, and the number of words had better be less than 20.

The long descriptions of objects (<id1>, <id2>, ...) are as follows: [<id1>]: <description1>, [<id2>]: <description2>, ...

- Table 30. The text prompt to create a summary of the objects’ descriptions. The placeholders <idX> and <descriptionX> are replaced with X-th object region indices and the corresponding detailed caption.

You are a helpful assistant in creating diverse multi-round conversation question-and-answer pairs from the given object descriptions. Your task is to design a conversation between you and a person asking about objects. The answers should be in a tone that a visual AI assistant is watching the video and answering the question. Output the JSON format with key ’Q’ and value ’A’, for example, {’QA_pairs’: [{’Q’: ’What is the object next to region [0]...’, ’A’: ’It is...’},{’Q’: ’Why this object is...’} ...]}.

Include diverse questions asking about the visual content of the video, including the object types, temporal information, object motions, actions, textures, spatial positions, etc. Only include questions that have definite answers:

- (1) one can see the content in the video that the question asks about and can answer confidently;
- (2) One can confidently determine from the video that it is not in the video. Do not ask any questions that cannot be answered confidently. Please follow the below instructions. ## Instructions

- - Do not add additional information to the output answer from the input object descriptions. DO NOT ask about uncertain details.
- - If you refer to a certain object from the list, USE region indices (<id1>, <id2>, ...) in the questions. DO NOT include the region indices in the answer. MUST generate several different QAs.

The object descriptions for <id1>, <id2>, ... are as below: [<id1>]: <description1>, [<id2>]: <description2>, ...

- Table 31. The text prompt to generate region-level conversation samples from object-level detailed captions. The placeholders <idX> and <descriptionX> are replaced with X-th object region indices and the corresponding detailed caption.

- • Provide a brief caption for the area indicated by <region>.
- • Describe in a short phrase the content within the bounds of <region>.
- • How would you succinctly caption the region highlighted by <region>?
- • Summarize the scene or object present in the section marked by <region>.
- • Can you give a concise description of what’s depicted in <region>?
- • Draft a short title for the video content enclosed by <region>.
- • What brief caption would best describe the visual within <region>?
- • Offer a succinct interpretation of the area pointed out by <region>.
- • If you were to provide a short tagline for the content at <region>, what would it be?
- • Give a one-liner description of the region demarcated by <region>.
- • How would you encapsulate the essence of the segment labeled <region> in a few words?
- • Characterize the content of the video portion specified by <region> briefly.
- • Craft a mini headline for the visual element spotlighted by <region>.
- • In a few words, how would you narrate the content found within <region>?
- • Pen down a concise caption for the video section delineated by <region>.
- • A short caption of region <region>:
- • A short description of region <region>:
- • A photo containing the region <region>:
- • A region <region> that shows
- • Write a short description for the region <region>
- • Write a description for the region <region>
- • Provide a description of what is presented in the region <region>.
- • Briefly describe the content of the region <region>.
- • Can you briefly explain what you see in the region <region>?
- • Could you use a few words to describe what you perceive in the region <region>?
- • Please provide a short depiction of the region <region>.
- • Using language, provide a short account of the region <region>.
- • Use a few words to illustrate what is happening in the region <region>.
- • Provide an overview of what you see in the region <region>.
- • Can you break down the main elements present in this region <region>?
- • What are the key features or subjects captured in this region <region>?
- • Summarize the primary components of this region <region>.
- • Walk me through the different aspects of this region <region>.
- • Highlight the main points of interest in this region <region>.
- • What stands out to you the most in this region <region>?
- • If you were to give a brief overview of this region <region>, what would you mention?
- • List the primary objects or subjects you identify in this region <region>.
- • Describe the first few things that catch your attention in this region <region>.
- • How would you introduce this region <region> to someone who hasn’t seen it?
- • What are the defining characteristics of this region <region>?
- • Give a concise description of the main content in this region <region>.
- • If you were to caption this region <region>, what might you say?
- • Describe the scene or setting depicted in this region <region>.
- • Provide a concise label for the highlighted <region>.
- • Offer a short depiction of what’s enclosed within <region>.
- • How would you briefly label the segment outlined by <region>?
- • Summarize the contents of the space marked by <region>.
- • Can you give a brief explanation of what’s featured in <region>?
- • Suggest a quick headline for the area depicted within <region>.
- • What short caption would best capture the details of <region>?
- • Provide a concise overview of the highlighted portion of <region>.
- • If you were to summarize the content in <region>, what would it be?
- • Offer a one-line explanation of what is shown in <region>.
- • How would you briefly summarize the key points in <region>?
- • Characterize the details of the video enclosed within <region> in a few words.
- • Draft a small title for the area spotlighted in <region>.
- • How would you narrate the content observed in <region> concisely?
- • Write a brief caption for the section of the video enclosed by <region>.
- • Create a short label for the visible content in <region>.
- • Offer a quick description of the content shown in <region>.
- • A snapshot of what <region> includes.
- • Describe what’s highlighted within <region>.
- • Provide a quick summary of the contents present in <region>.
- • What’s the best way to describe the content within <region> briefly?
- • What’s shown in <region>?Please describe.
- • How would you quickly explain what is featured in <region>?
- • Provide a short summary of what’s visible in <region>.
- • Could you break down what you see in <region>?
- • Can you briefly outline the main elements in <region>?
- • Give a brief interpretation of what stands out in <region>.
- • What are the key points of focus in this section <region>?
- • Provide a concise rundown of what is visible in <region>.
- • Describe the notable objects or subjects you can see in <region>.
- • How would you introduce the details shown in <region> to someone new?
- • List the main features or subjects captured in <region>.
- • Summarize the primary components present in <region>.
- • What catches your attention first in <region>? Table 32. The list of instructions for brief region description.

- • Describe in detail the object located at <region> in the video, including its appearance, style, actions, and any visible details.
- • Provide a comprehensive description of the area marked by <region>, focusing on textures, colors, motions, and any notable features.
- • Elaborate on the artwork shown in the region indicated by <region>, mentioning its color, appearance, size, style, and any standout features.
- • Give a detailed analysis of the scene within the boundary of <region>, touching upon its components, ambiance, and any thematic expressions.
- • Craft a thorough narrative about the piece of the video highlighted by <region>, from its aesthetic qualities to its possible historical context.
- • Explain in depth the characteristics and attributes of the subject found in the segment tagged with <region>.
- • Generate a long, detailed caption for the segment of the video at <region>, covering aspects such as its origin, material, and any symbolic meaning.
- • Paint a vivid picture with words about the region at <region>, diving into the intricacies and nuances present in the area.
- • Zoom in on the area indicated by <region> and describe every discernible detail, from texture and color to form and function.
- • Offer an expanded description of the contents within the area marked by <region>, encompassing its color, appearance, size, style, and any remarkable features.
- • Provide a detailed account of the object located at <region> in the video.
- • Offer a full description of the object found in <region>.
- • Describe comprehensively the object at <region>, including its physical traits, style, actions, and visible characteristics.
- • Give an extensive description of the area labeled <region>, paying attention to textures, colors, movements, and notable details.
- • Elaborate thoroughly on the artwork situated in the area indicated by <region>, noting its color, size, style, and key features.
- • Analyze the scene within the confines of <region> in detail, considering its components, atmosphere, and any thematic elements.
- • Develop a rich narrative for the part of the video highlighted in <region>, exploring its aesthetic aspects and potential historical significance.
- • Explain the key attributes and features of the subject found in <region> with in-depth detail.
- • Compose an extended caption for the video segment located at <region>, focusing on its origin, materials, and possible symbolic meaning.
- • Describe vividly the region marked <region>, exploring the intricacies and nuances present in the area.
- • Focus on the area highlighted by <region> and provide a meticulous description, covering its texture, color, shape, and function.
- • Expand on the description of what’s present in the area of <region>, addressing its colors, appearance, dimensions, style, and distinctive qualities.
- • Give a precise and thorough breakdown of the object located at <region>, considering its visual and functional features.
- • Delve deeply into the description of the contents at <region>, focusing on the interplay of textures, colors, movements, and standout aspects.
- • Explore the artwork or object featured in <region> with a focus on its color, style, and size.
- • Offer a detailed breakdown of the scene in <region>, discussing its components, mood, and any visible themes.
- • Provide a rich account of the aesthetic qualities and potential historical or cultural relevance of the area highlighted by <region>.
- • Explain the visible details of the subject in <region>, including its appearance, movements, and any distinguishing traits.
- • Create a thorough, descriptive caption for the portion of the video located in <region>, touching on its materials, origin, and symbolic elements.
- • Use vivid language to describe the region marked by <region>, diving into the subtle details and unique aspects of the area.
- • Take a close look at <region> and describe in detail the features you observe, including texture, color, form, and function.
- • Expand upon the visual description of what is contained within the marked area <region>, covering its appearance, size, and stylistic features.

###### Table 33. The list of instructions for detailed region description.

