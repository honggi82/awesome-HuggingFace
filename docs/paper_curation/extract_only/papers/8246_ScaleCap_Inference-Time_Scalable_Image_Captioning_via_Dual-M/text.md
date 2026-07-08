arXiv:2506.19848v1[cs.CV]24Jun2025

# ScaleCap: Inference-Time Scalable Image Captioning via Dual-Modality Debiasing

### Long Xing1,2∗, Qidong Huang1∗, Xiaoyi Dong2,3 , Pan Zhang2, Yuhang Zang2, Yuhang Cao2, Jinsong Li2,3, Shuangrui Ding2,3, Weiming Zhang1, Nenghai Yu1, Jiaqi Wang2 , Feng Wu1, Dahua Lin2,3 1University of Science and Technology of China 2Shanghai Artificial Intelligence Laboratory 3The Chinese University of Hong Kong

{xing_long@, hqd0037@}mail.ustc.edu.cn

## Abstract

This paper presents ScaleCap, an inference-time scalable image captioning strategy that generates comprehensive and detailed image captions. The key challenges of high-quality image captioning lie in the inherent biases of LVLMs: multimodal bias resulting in imbalanced descriptive granularity, offering detailed accounts of some elements while merely skimming over others; linguistic bias leading to hallucinated descriptions of non-existent objects. To address these issues, we propose a scalable debiased captioning strategy, which continuously enriches and calibrates the caption with increased inference budget. Specifically, we propose two novel components: heuristic question answering and contrastive sentence rating. The former generates content-specific questions based on the image and answers them to progressively inject relevant information into the caption. The latter employs sentence-level offline contrastive decoding to effectively identify and eliminate hallucinations caused by linguistic biases. With increased inference cost, more heuristic questions are raised by ScaleCap to progressively capture additional visual details, generating captions that are more accurate, balanced, and informative. Extensive modality alignment experiments demonstrate the effectiveness of ScaleCap. Annotating 450K images with ScaleCap and using them for LVLM pretraining leads to consistent performance gains across 11 widely used benchmarks. Furthermore, ScaleCap showcases superb richness and fidelity of generated captions with two additional tasks: replacing images with captions in VQA task, and reconstructing images from captions to assess semantic coverage. Code is available at https://github.com/Cooperx521/ScaleCap.

## 1 Introduction

In the realm of large vision language models (LVLMs)[2, 1, 43, 44, 35, 11, 12, 66, 25], the quality and richness of image-text pairs play a pivotal role in determining the effectiveness of pre-training[5, 35, 29]. Particularly, longer and more descriptive captions have shown increasing importance in supporting fine-grained vision-language alignment, moving from early-stage captions with only a few generic words[9, 29] to recent efforts that generate paragraph-level, context-rich descriptions[5, 49]. As the field pushes toward building ever more capable foundation models[41, 19, 1, 52], the need for vast quantities of high-quality multimodal data becomes increasingly urgent. However, relying on human annotation[59, 53] or proprietary APIs[13, 5] to produce such detailed captions proves prohibitively expensive and fundamentally non-scalable. This challenge has spurred growing interest in developing scalable captioning strategies based on open-source LVLMs, offering a more costeffective and flexible path for constructing large-scale, high-quality captions.

Preprint.

[Figure 1]

ScaleCap:

1 2 3 4 5 6 7 8 9 10

7

The image portrays a meticulously designed, modern home office setup that exudes a minimalist aesthetic, characterized by clean lines and functional elegance. The flooring is comprised of light-colored wood planks, arranged neatly in a parallel fashion, lending a warm and natural feel to the space. Over the wooden floor, a geometric-patterned rug is artfully placed, adding a touch of sophistication and visual interest. At the heart of this office is a double-desk setup, a centerpiece that defines the room's functionality. The desks are white, sleek, and minimalist in design, offering ample workspace for two individuals.

…

5

2

…

1

4

10

Two black chairs, with a unique woven design, are positioned in front of the desks, providing ergonomic seating for the users. These chairs feature a sturdy metal frame and a comfortable woven seat, designed to support long hours of work. On the left desk, a desktop computer is set up, complete with a monitor, keyboard, and mouse. The monitor displays a captivating scenic image of a mountainous landscape with a tranquil blue lake, offering a picturesque view that can inspire creativity and relaxation. The keyboard and mouse are placed conveniently in front of the monitor, ready for use. A small potted plant, likely a succulent or a small fern, adds a touch of greenery to the left side of the desk, enhancing the room's air quality and bringing a bit of nature indoors. On the right desk, a white laptop is positioned, its screen showing a vibrant and colorful image, possibly a photograph or a graphic design project.

…

- 8

- 9

3

…

6

A black printer, a Canon model, is placed on the central shelf, slightly to the right of the center of the desk setup, ensuring easy access for printing needs. The central shelf itself is white, multi-tiered, and offers several compartments and shelves for organizing and storing office supplies. The poster above the left desk features a black and white image of a woman holding a surfboard, evoking a sense of adventure and sportsmanship. The poster above the right desk depicts a dandelion in black and white, symbolizing resilience and the beauty of nature. A small white ceramic cup is placed on the right side of the right desk, perhaps holding a beverage or serving as a pen holder, adding a practical yet subtle detail to the workspace.

…

[Figure 2]

1 ~ 10 1 ~ 10

Detailed Description

… …

[Figure 3]

Indetailed Description

Qwen2-VL-72B: GPT4o:

1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10

The image depicts a modern home office setup with a clean and organized aesthetic. Two white desks are arranged in an L-shape, connected by a shelving unit in the corner. Two black, minimalist chairs with a wireframe design are placed in front of each desk.

The image depicts a modern, minimalist home office setup. The room features a light green wall and light wooden flooring, creating a bright and airy atmosphere.

…

There are two white desks positioned side by side, forming an L-shape. The desk on the left has a desktop computer setup, including a monitor displaying a scenic landscape, a keyboard, and a mouse. The desk on the right has a laptop open, displaying a different image, and a cup of coffee placed next to it. Two black chairs with a unique, modern design are placed in front of the desks. The chairs have a woven seat and backrest, providing a contemporary look. Between the two desks, there is a white shelving unit with three shelves. The top shelf holds a black printer. The middle shelf has a few books

On the left desk, there's an all-in-one desktop computer displaying a scenic landscape. On the right desk, there's a laptop showing a close-up image of flowers. A Canon printer is placed on the corner shelving unit. A small potted plant is on the left desk. A white cup is on the right desk.

… …

…

Two framed pictures are hung on the light green wall. The left picture shows a person holding a surfboard above their head at the beach. The right picture features a black and white dandelion design. The floor is covered with lightcolored wood tiles. A patterned rug is partially visible in the foreground.

The picture on the left shows a person from behind, holding a surfboard above their head. A small potted plant is placed on the left desk, adding a touch of greenery to the workspace. The floor has a light-colored rug with a subtle pattern, adding texture to the room.

…

…

…

- Figure 1: Comparison between the captions generated by our ScaleCap and those produced by other advanced VLMs. The parts of the caption that are bolded refer to the detailed descriptions of the object, while the parts that do not mention the target object are included in the ellipsis.

Despite growing interest, open-source LVLMs still generate suboptimal captions due to two intrinsic biases: multimodal and linguistic. First, multimodal datasets often contain imbalanced annotations, causing models to over-describe salient objects while glossing over others, leading to inconsistent granularity and reduced caption completeness. Second, inheriting language habits from LLMs[28, 37], LVLMs tend to favor generic phrasing and frequent co-occurrence patterns, resulting in visual hallucinations: descriptions of non-existent objects or attributes that misrepresent the image. Together, these biases hinder the generation of high-quality, faithful, and fully detailed captions.

To mitigate these limitations, recent efforts[30, 49] have explored the use of auxiliary tools or expert modules, such as object detectors [16] or image taggers [65] to enrich captions or reduce hallucinations [60]. While these designs can offer targeted improvements, the overall caption quality is ultimately bound by the precision and coverage of the supporting tools. Given the combinatorial diversity of real-world objects and their attributes, it is unrealistic to rely on handcrafted or category-specific modules as a general solution. These tool-dependent approaches thus fall short in achieving the breadth and adaptability required for generating truly comprehensive and scalable image descriptions.

In contrast to tool-based approaches, we argue that general-purpose LVLMs already possess sufficient perceptual capacity for rich captioning—if guided properly. In particular, we observe that the lack of detail is not necessarily due to insufficient visual understanding but rather stems from suboptimal information extraction during generation. As shown in Figure. 2, when we explicitly ask for more details about an object that is only roughly described in the original caption, the model can provide precise descriptions. Notably, this perceptual capacity is not confined to large models. We find that even compact LVLMs with only 7B parameters can match the descriptive quality of much larger models when equipped with the right prompting. This observation highlights a promising and cost-effective path toward scalable caption generation: leveraging smaller models with proper guidance, rather than relying on brute-force scaling.

Motivated by this insight, we propose ScaleCap, a scalable debiasing strategy that stimulates the model to revisit and refine the caption through a structured, recurrent process. ScaleCap contains two complementary components: heuristic question answering and contrastive sentence rating. The first component prompts a general-purpose LLM to generate content-specific follow-up questions based on an initially generated caption. These questions target under-described or ambiguous elements, such as object attributes or spatial relations, and are then answered by the LVLM to progressively

[Figure 4]

7B vs 72B Perception

Stastistical Results

[Figure 5]

The image depicts a scene from an animated show ..

Miss the details!

[Figure 6]

The food on the table includes several pieces of what appears to be a type of bread or pastry, some of which appear to be topped with jam or butter.

[Figure 7]

Describe this image in detail.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

###### …

The background includes a television set on a stand, a curtain...

[Figure 14]

there are several plates of food

[Figure 15]

100 imgs

7B

[Figure 16]

[Figure 17]

Generate simple question about food

[Figure 18]

Find out all objects missing detailed descriptions.

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Initial Caption

[Figure 23]

[Figure 24]

[Figure 25]

Describe more details about the food。

[Figure 26]

There are a few plates of bread scattered on the table, with some butter left on the surface, making the food look very delicious.

Organize Instructions

[Figure 27]

[Figure 28]

The food on the table includes several pieces of what appears to be a type of bread or pastry, some of which appear to be topped with jam or butter.

Describe more details about xxx

72B

[Figure 29]

[Figure 30]

[Figure 31]

The 7B and 72B LVLMs exhibit similar perception capabilities

93% of the new responses provide the objects’ details.

[Figure 32]

[Figure 33]

- Figure 2: The reason for certain object detail omissions in LVLM captions is mainly due to the absence of guiding heuristic questions rather than insufficient perceptual capability. We also observe that 7B and 72B LVLMs exhibit similar perceptual capabilities.

inject additional visual details into the caption. This iterative question-answering loop enables a scalable enrichment process, allowing the model to uncover increasingly fine-grained details with each refinement round. To ensure the factuality and fluency of the evolving caption, both the initial caption and its subsequent refinements are evaluated by the contrastive sentence rating module. This second component addresses hallucinations through an offline sentence-level contrastive decoding strategy, avoiding the coherence issues commonly seen in online decoding. Candidate sentences are generated independently and scored to identify high-quality, visually grounded variants, ensuring that the final caption is not only rich in detail but also coherent and faithful to the image.

Case Study

[Figure 34]

7B 和72B的LVLM表现出比较接近的 perception能力

[Figure 35]

[Figure 36]

[Figure 37]

Qwen2-VL-72B: The picture on the left shows a person from behind, holding a surfboard above their head.

[Figure 38]

GPT4o: The left picture shows a person holding a surfboard above their head at the beach.

[Figure 39]

7

ScaleCap: The poster above the left desk features a black and white image of a woman holding a surfboard

We comprehensively evaluate the effectiveness of ScaleCap across three complementary settings. First and most critically, we use ScaleCap to annotate a large-scale dataset of 450K images and apply it to pretrain multiple LVLM architectures. Across all settings, models trained with ScaleCap consistently achieve best performance on 11 widely used multimodal benchmarks, demonstrating its broad applicability and pretraining benefits. Second, under the Prism framework, we assess caption informativeness via downstream performance and find that ScaleCap-based Qwen2-VL-7B outperforms even larger models like Qwen2-VL-72B—highlighting the efficiency of our strategy. Finally, we evaluate semantic coverage through image reconstruction, showing that ScaleCap captions better preserve visual content than existing open-source models and GPT-4o. These results collectively demonstrate that the captions generated by our approach are highly informative and accurate.

## 2 ScaleCap

In this section, we first introduce the details of the proposed scalable captioning pipeline ScaleCap, then present ScaleCap-450K, a large-scale, high-quality dataset constructed using ScaleCap.

The overall pipeline of ScaleCap is illustrated in Figure 3, which integrates heuristic question answering and contrastive sentence rating in a scalable generation-refinement framework. Specifically, given an input image, the model is prompted to generate an initial caption, and the contrastive sentence rating module is applied to extract high-quality sentences from it. We denote it as the “golden sentences”, which is the starting point of the ScaleCap. Building upon these golden sentences, the heuristic question answering module generates a series of content-relevant questions to explore additional visual details. Each answer is then evaluated by the contrastive sentence rating module to filter out hallucinated or low-quality content. As more questions are raised, the caption is progressively enriched with finer-grained and more balanced descriptions. At last, we use a capable LLM to integrate the complex and abundant visual information into a complete and structural image caption. To manage inference overhead, the process is governed by a pre-defined scale budget, which limits the maximum number of questions that can be asked. ScaleCap’s scalable refinement strategy flexibly balances caption quality and computational cost, producing informative and faithful outputs.

### 2.1 Heuristic Question Answering Module

Heuristic question raising. To extract richer object- and position-level details, we generate simple, structured instructions (e.g., “Describe more details about the airplane”) based on golden sentences. We use a powerful LLM ML in an in-context manner for instruction generation. Let SG = {S1,S2,...,Sq} denote the golden sentence set. For each Sk, we construct an object prompt

###### Scale Caption

[Figure 40]

[Figure 41]

###### Contrastive Sentence Rating

[Figure 42]

Initial Caption

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Offline Probability Contrast

[Figure 47]

forward

[Figure 48]

The image depicts a white race car... The background shows a typical desert scene...

[Figure 49]

[Figure 50]

desert

atypical

scene

token probability

[Figure 51]

###### desert

atypical

scene

Rating and Filter

[Figure 52]

[Figure 53]

[Figure 54]

LLM

[Figure 55]

The image depicts a white race car...

forward

[Figure 56]

[Figure 57]

- 1. The image depicts a white race car, a limited...
- 2. The car features a large rear wing, which is typical for race cars to...

[Figure 58]

Golden Sentences

[Figure 59]

[Figure 60]

LVLM

The background shows a typical desert scene...

[Figure 61]

Keep sentences without hallucinations.

desert

atypical

scene

[Figure 62]

token probability

[Figure 63]

[Figure 64]

[Figure 65]

Heuristic Question

[Figure 66]

[Figure 67]

###### Caption Scaling

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Golden Sentences

Final Caption

Object Instructions

Rating and Filter

DetailCoverage

Object Detail Summarization

[Figure 73]

- 1. Describe more details about the race car.
- 2. Describe more details about the headlights.

[Figure 74]

The race car is...

The image portrays a pristine white BMW M3 GTS race car, a limited...

[Figure 75]

N

[Figure 76]

The headlights in...

...

· · ·

[Figure 77]

Golden Sentences

[Figure 78]

N

The car is adorned with an array of sponsor logos and decals, showcasing its affiliation ...

Number of Instr

[Figure 79]

Position Instructions

[Figure 80]

[Figure 81]

[Figure 82]

Golden Sentences

- 1. Describe more details about the position of the car.
- 2. Describe more details about the position of the headlights.

Rating and Filter

Adjust N to trade off the richness of image detail and cost

[Figure 83]

Position Detail

[Figure 84]

The position of...

Summarization ...

N

...

The headlights are...

· · ·

[Figure 85]

- Figure 3: Overview of ScaleCap. ScaleCap is composed of two synergistic parts: heuristic question answering and contrastive sentence rating. The first module utilizes a general-purpose LLM to create guiding questions, and the second module addresses hallucinations by offline contrastive strategy.

Tict with in-context examples to guide the LLM in generating a set of object-related instructions Iok = Io,k1,Io,k2,...,Io,kk

, where each Io,ik queries additional details about an object in Sk. Together, Iok covers all objects mentioned in Sk. This process can be mathematically represented as

v

Iok = ML(Tict,Sk), k = 1,2,...,q (1)

Likewise, we generate positional instructions to capture the spatial relationships between objects and the overall image layout. Building on the object instruction set Io, we construct the position instruction set Ip by simply adding a position-specific prefix to each object instruction, yielding prompts like “Describe more details about the position of the airplane.”

For the full set of golden sentences SG = {S1,S2,...,Sq}, we generate both object and position instructions per sentence, forming Io = qk=1 Iok and Ip = qk=1 Ipk. Together, these instruction sets support a comprehensive understanding of object appearances and their position. To manage inference overhead, the process is governed by a pre-defined scale budget N, which limits the maximum number of object and position instructions. In the following experiments, unless otherwise specified, N is generally set to a large value to include all instructions.

Efficient visual answering. As discussed in Prism [46], LVLMs exhibit comparable perceptual capabilities across scales, with differences mainly in reasoning. Since the constructed Io and Ip are straightforward and require minimal reasoning, a small-scale LVLM can effectively handle these instructions. Thus, we directly apply Io and Ip to a lightweight LVLM to extract fine-grained image details at low cost. Formally, for any instruction Io,ik , the object-specific details are obtained as Do,ik = MV (I,Io,ik ), where both the image I and instruction Io,ik are input to the model. By processing Io and Ip, we collect object details Do and position details Dp.

### 2.2 Contrastive Sentence Rating Module

Basic Formulation. Let MV denote the LVLM parameterized by θ. Given an image I and a captioning instruction T, the model generates a caption C as: C = MV (I,T). where C = {c1,c2,...,cn} is a sequence of n tokens generated autoregressively. At each decoding step t, the token ct is drawn from the conditional distribution: yt ∼ pθ (yt | I,[T,c<t]), with yt denoting the predicted token and c<t the preceding context. The probability distribution is conditioned on both the image I and the textual input [T,c<t], guiding the model to iteratively generate each token.

###### Data Processing and Analysis

[Figure 86]

[Figure 87]

[Figure 88]

Average Character Length of Captions

###### LAION 5B ShareGPT4V

###### ScaleCap-450k

[Figure 89]

2542

###### +1289

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

2500

[Figure 97]

[Figure 98]

###### Our caption has the longest length!

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

2000

[Figure 108]

[Figure 109]

Filtering Criteria

- (a) Resolution
- (b) Complexity

100k Images

+311

1500

1253

[Figure 110]

[Figure 111]

+333

942

###### 450k Images

1000

[Figure 112]

ScaleCap

[Figure 113]

[Figure 114]

High resolution Moderate complexity

+555

609

[Figure 115]

- (a) Diverse data sources
- (b) Appropriate richness of image content

- (a) Golden Sentence Selection
- (b) Cascading Caption Amplification

Too simple Too complex Low resolution

500

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

350k Images

54

0

BLIP-LCS LLaVA-23K ShareGPT4V DenseFusion ScaleCap

- Figure 4: Data processing and analysis. During the image collecting and processing stage, we primarily focus on the diversity and richness of image content. In the resulting ScaleCap-450k, the captions are significantly longer than those in other datasets.

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Offline Contrastive Probability Analysis. To detect hallucinated content without disrupting the natural language distribution, we adopt an offline contrastive probability analysis, in contrast to prior online decoding methods [28, 56, 23]. Given the initial caption tokens C = {c1,c2,...,cn}, we compute two sequences of token probabilities: caption token with and without the image input. The probability sequence P conditioned on the image I is:

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

It includes the precise description of each instance.

[Figure 137]

P = {p1,p2,...,pn}, pt = pθ(yt = ct | I,[T,c<t]), t ∈ [1,n]. (2)

Here, pt denotes the likelihood of generating token ct given the image I, the instruction prompt T, and previous tokens c<t. We then compute the counterpart P′ by conditioning only on textual input:

P′ = {p′1,p′2,...,p′n}, p′t = pθ(yt = ct | [T,c<t]), t ∈ [1,n]. (3)

This sequence reflects the model’s inherent linguistic prior—how likely it is to generate ct without visual evidence. Since large vision-language models (LVLMs) inherit strong language modeling capabilities from LLMs, they may over-rely on textual co-occurrence patterns, leading to hallucinated outputs [28, 37]. To quantify this effect, we define the contrastive probability sequence ∆P as:

∆P = P − P′ = {∆p1,∆p2,...,∆pn}, ∆pk = pk − p′k. (4)

A high ∆pk indicates that token ck benefits significantly from the visual context and is thus more likely grounded in the image. In contrast, a low ∆pk suggests that the token is generated primarily based on language priors, signaling potential hallucination.

Sentence-Level Rating. To mitigate hallucinations without disrupting fluency, we filter at the sentence level rather than at the token level. The initial caption C is segmented into sentences as C = {C1,C2,...,Cm} using punctuation (e.g., periods), where m is the number of sentences. Sentence Ck = {ck1,ck2,...,ckk

} consists of kl tokens, and each token associates with a contrastive probability difference ∆pki from Equation 4, forming a sentence-wise sequence ∆Pk = {∆pk1,...,∆pkk

l

}.

l

To assess whether a sentence is language-biased, we compute the maximum ∆pki over critical tokens1. Sentences with strong visual grounding are retained as the Golden Sentences SG:

SG = {Ck | max ∆pk1,∆pk2,...,∆pkk

> τ} (5) where τ is a tunable threshold, with higher values leading to stricter filtering.

l

### 2.3 Caption Integration

Our ultimate goal is to form a complete and structural image caption. Leveraging the strong summarization and logical reasoning capabilities of the LLM, we instruct it to organize and consolidate the relatively fragmented content from the detail sets using two prompts, To and Tp, for object-level details and position-level details respectively. To ensure coherence and structure during summarization, we also provide the LLM with the golden sentences as a caption backbone. This helps the LLM maintain awareness of the overall caption structure, enhancing the quality of the summarization. As a result, we generate the following summaries:

Co = ML(SG,To,Do), Cp = ML(SG,Tp,Dp), (6)

1Identified via part-of-speech tagging to exclude function words such as adpositions.

- Table 1: Comparison with different datasets on 11 benchmarks. ScaleCap-450k significantly improves pretraining efficiency, achieving the best results on nearly all benchmarks with the same amount of data, which demonstrates the superior quality of the captions generated by ScaleCap.

Pretraining Data

Info VQA

Doc VQA

Chart QA

MM Star

Math Vista

LLaVA Bench

Model

MMVet MMB MMMU SEED AI2D Average

Vanilla 46.2 81.5 75.5 47.0 47.0 72.8 46.7 74.6 45.2 69.9 71.6 61.6 ShareGPT4V-450k 47.5 82.9 76.0 48.8 46.2 72.9 48.9 75.2 43.8 71.3 72.7 62.4 DenseFusion-450k 49.4 84.8 77.1 49.2 47.5 70.3 52.4 73.1 44.3 70.6 73.9 63.0 ScaleCap-450k 51.8 85.7 77.8 48.8 49.7 74.7 55.9 75.6 46.1 71.6 74.0 64.7

Qwen2.5-7B + Qwen2.5-ViT

Vanilla 39.1 76.3 72.1 44.8 41.6 66.4 39.9 69.1 37.4 67.1 69.7 56.7 ShareGPT4V-450k 42.4 78.3 73.0 44.8 43.7 66.8 43.3 69.5 38.5 67.9 70.2 58.0 DenseFusion-450k 44.5 81.1 73.8 43.5 45.9 69.4 39.9 68.7 42.1 68.2 70.8 58.9 ScaleCap-450k 47.2 81.7 74.7 44.9 46.1 68.3 45.6 70.0 41.3 69.1 71.8 60.1

Qwen2.5-3B + Qwen2.5-ViT

Vanilla 36.2 72.3 68.4 47.8 43.5 68.7 41.3 72.2 41.2 72.7 73.9 58.0 ShareGPT4V-450k 36.9 72.3 69.1 48.6 42.3 66.6 45.8 72.5 41.5 73.2 72.6 58.3 DenseFusion-450k 39.1 75.1 70.0 48.7 44.1 66.9 47.2 72.2 40.1 73.6 73.0 59.1 ScaleCap-450k 39.6 75.5 71.3 48.7 44.5 70.3 48.0 73.4 42.1 74.0 74.7 60.2

InternLM2.5-7B + CLIP-ViT-L

InfoVQA

ChartQA

###### SEED

48.5

75.0

69.5

ScaleCap

ScaleCap

ScaleCap

47.0

69.0

DenseFusion

DenseFusion

DenseFusion

Accuracy(%)

74.5

45.5

68.5

74.0

44.0

68.0

73.5

42.5

67.5

41.0

73.0

67.0

100 150 200 250 300 350 400 450

100 150 200 250 300 350 400 450

100 150 200 250 300 350 400 450

Number of Captions (k)

Number of Captions (k)

Number of Captions (k)

Figure 5: The benchmark performance under different number of pretraining data.

where Co summarizes the object-level details, and Cp summarizes the positional details of the objects in the image. Finally, we utilize the LLM once more to integrate Co and Cp into a comprehensive final caption. The final caption is generated as follows:

Fc = ML(SG,Tfinal,Co,Cp), (7) where Tfinal is the integration prompt. All prompts used in ScaleCap can be found in Appendix.A.

### 2.4 ScaleCap-450k Dataset

Based on ScaleCap, we create a hyper-detailed image caption dataset as Figure. 4 presents. We first collect 450,000 images and then annotate them using ScaleCap to generate high-quality image-text pairs. The annotation and data processing details are as follows.

Data Source and Processing. In collecting images for our dataset, we primarily focus on two aspects: diversity and richness of image content. Given that the ShareGPT4V-100k already includes a wide range of categories, such as artworks, landmarks, etc., it inherently offers a certain level of diversity. Therefore, we opt to directly incorporate these images into our dataset. To further enhance the dataset’s diversity and to obtain more content-rich images, we additionally select 350k images from the LAION-5B[27] dataset. During filtering, we retain only images with high resolution and moderate complexity. The specific tools and details used in filtering can be found in the Appendix.B.

Caption Model Selection. As previously mentioned, ScaleCap leverages the collaboration between a Vision Language Model and a Large Language Model to generate high-quality captions. For LVLM, as we discussed above, a small model is capable of capturing visual content, so we use Qwen2-VL-7B by default. When it comes to the LLM, the question-raising task is relatively simple, while the integration of complex and abundant visual information within thousands of tokens requires advanced reasoning capability, so we resort to Qwen2-72B based on an empirical study.

## 3 Pretraining Experiments

To comprehensively evaluate the effectiveness of the ScaleCap-450k dataset, we conduct extensive pretraining experiments. The experimental details are as follows.

- Table 2: Caption informativeness comparison results in Prism Framework. ScaleCap significantly outperforms the quality of captions generated by Qwen2-VL-7B and Qwen2-VL-72B.

Caption Strategy

MM Vet

MM Star

Info VQA

Chart QA

Text VQA

LVLM

Average

Prism Qwen2-VL-7B 53.3 47.7 49.3 68.5 51.7 54.1 Prism Qwen2-VL-72B 57.3 48.7 50.0 69.5 54.4 56.0 ScalCap Qwen2-VL-7B 58.8 50.3 53.8 72.9 55.3 58.2

Table 3: Ablation study of Object Instructions and Position Instructions on benchmarks subset.

Text VQA

MM Vet

Chart QA

Method

Avg

Only Object Instr 52.9 54.5 69.1 58.8 Only Position Instr 52.3 54.3 65.7 57.4 ScaleCap 53.2 58.8 72.5 61.5

Table 4: ScaleCap, equipped with GPT-4o in the Prism framework and supplemented with image-visible responses, outperforms other advanced proprietary models.

ScaleCap

Method Sonnet3.5 GPT4V GPT4o Gemini-2.0-Pro Qwen2-VL-72B

(GPT4o+GPT4o)

MMVet 66.0 67.5 69.1 70.4 74.0 76.1

### 3.1 Implementation Details

Table 5: Ablation study on the summarization model scale in ScaleCap on benchmarks subset.

Summarization Model MMVet MMStar

Qwen2-7B 43.6 40.3 Qwen2-72B 58.8 49.5

Our model structure follows LLaVA-NeXT[34], comprising a vision encoder, a MLP projector, and a LLM. To thoroughly evaluate our approach, we experiment with three configurations: (1) Qwen2.5-7B + Qwen2.5-ViT, (2) Qwen2.5-3B + Qwen2.5-ViT, and (3) InternLM2.5-7B + CLIPViT-L/14-336. Training involves three stages: initial pretraining on BLIP-558K, further pretraining using high-quality image caption, and final instruction-tuning with Open-LLaVA-NeXT-Instruct1M[8]. Baselines for comparison include a two-stage Vanilla method (without further pretraining), ShareGPT4V-450k[5], and DenseFusion-450k[30]. To ensure fair evaluation, differences are limited to the further pretraining datasets only. Details of the settings can be found in the Appendix.C.

### 3.1.1 Main results

Comparison with different pretraining datasets. As shown in Table 1, comprehensive experiments under three settings demonstrate that pretraining with ScaleCap-450k consistently yields the best performance across most benchmarks. For instance, in Qwen2.5-7B setting, ScaleCap-450k improved InfoVQA scores by 4.3% over ShareGPT4V-450k and 2.4% over DenseFusion-450k. On natural image QA benchmarks like MMVet, ScaleCap-450k achieved a 7% gain over ShareGPT4V-450k and

- 3.5% over DenseFusion. Similar gains were observed with other LLMs.

Compared to DenseFusion, which uses captions from multiple expert models that often miss object details due to limited attribute coverage, ScaleCap leverages general-purpose models for higherquality captions. These detailed descriptions of objects significantly enhance modality alignment during pretraining. Consequently, the better-aligned visual features are more readily understood by the LLM, enabling finer-grained image comprehension and improved benchmark performance.

The pre-training data scaling performance. We then study the data efficiency of captions generated by ScaleCap. We conduct pretraining using varying data volume from 100K to 450K, with the Qwen2.5-3B setting. The results show that, given the same pretraining samples, modality alignment using the ScaleCap dataset significantly outperforms the DenseFusion dataset. Moreover, as the pretraining data volume increases, the advantage of ScaleCap becomes even more pronounced. These findings indicate that captions generated by ScaleCap enable more efficient modality alignment. Additionally, the steep upward trend of the ScaleCap curve suggests that further expanding the data volume will continue to yield substantial performance gains.

4 Dive into ScaleCap

- 4.1 Validating informativeness of ScaleCap via VQA

Setup. Prism [46] is a framework designed to disentangle the perception and reasoning processes of LVLMs. It separates the problem-solving pipeline into two stages: in the perception stage, LVLMs extract information from images and convert it into textual form without access to the question, thus producing general-purpose captions; in the reasoning stage, LLMs generate answers based on the extracted text. With a fixed LLM, the benchmark performance directly reflects the informativeness

contained within the caption. We use Qwen2-72B as the LLM in the Prism framework to answer visual questions based on captions. For comparison, we provide the carefully designed Generic Instruction used in [46] as a prompt to Qwen2-VL-7B and Qwen2-VL-72B to generate as detailed a caption as possible, establishing strong baselines.

Main Results. As illustrated in Table 2, we can observe that ScaleCap outperforms both Qwen2VL-7B and Qwen2-VL-72B across all benchmarks, demonstrating outstanding performance. These results strongly validate the informativeness of caption generated by ScaleCap, demonstrating that its heuristic question answering effectively extracts fine-grained image details. This significantly enhances the caption quality initially generated by Qwen2-VL-7B, even surpassing Qwen2-VL-72B by a substantial margin. These findings indicate that ScaleCap can generate captions of significantly higher quality than those produced by a large LVLM trained on extensive image-text data.

The Potentials of ScaleCap. ScaleCap is a flexible captioning pipeline that allows for the replacement of both the LVLM and LLM with any open-source or proprietary models. This flexibility suggests significant potential, especially when equipped with powerful models like GPT-4o, which could enhance caption quality. To test this, we compare ScaleCap with GPT-4o against other advanced models, including Sonnet 3.5 and Gemini-2.0-Pro. In experiments, ScaleCap generates the questionirrelevant captions first, with the LVLM’s direct image-visible response also attached as supplemental information to the LLM. As shown in Table 4, ScaleCap not only outperforms its direct answering baseline but also all proprietary LVLMs, indicating its high potential.

### 4.2 Validating Informativeness of ScaleCap via Reconstruction

##### Image Reconstruction Comparison

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Ranking Score

Caption

[Figure 143]

[Figure 144]

2.07 2.15

The image dipicts an exquisitely arranged table setting ...

[Figure 145]

FLUX

[Figure 146]

[Figure 147]

1.78

[Figure 148]

The focal point of this setting is a pristine white plate with a slender ...

Generate Caption

User Study

Generate Image

Qwen2VL72B

ScaleCap

GPT4o

[Figure 149]

...

Most Similar!

Original Image ScaleCap GPT4o Qwen2-VL-72B

[Figure 150]

- Figure 6: Human evaluation of image similarity with original image over 50 samples and 25 volunteers. Volunteers rank the images based on their similarity to the original image.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Leveraging the capabilities of powerful modern text-to-image generation models, the similarity between a reconstructed image and its original can effectively reflect the extent to which a caption covers the image’s content. To gain an intuitive understanding of caption quality, we utilize one of the best image generation models, FLUX. FLUX can adhere to very detailed instructions, such as "wearing a silver cross-shaped necklace around his neck." As a result, it is an ideal tool for validating caption quality. When the caption is highly detailed and covers every object in the image, the FLUX-generated image shows a high degree of similarity to the original image. Conversely, if the caption contains errors or lacks detail, the similarity will be much lower. We randomly sampled 50 images and used ScaleCap, GPT4o, and Qwen2-VL-72B to generate captions. FLUX then generate corresponding images based on these captions. Finally, we invite 25 volunteers to rank the similarity of the generated images to the originals. A model that ranks first in the three categories receives three points. As shown in Figure. 6, captions generated by ScaleCap result in images that more accurately reflect the original images than those generated by GPT4o, significantly surpassing Qwen2-VL-72B. This further demonstrates the superior quality of our approach.

model

[Figure 158]

[Figure 159]

[Figure 160]

### 4.3 Analysis on ScaleCap components

Caption quality increases with scaling of inference budgets N. In ScaleCap, we adjust the number of heuristic questions to trade off between the richness of image detail and computational cost. To intuitively present this effect through the Prism framework, we conduct experiments on MMVet and MMStar. The results in Figure 7 show that initially increasing the number of heuristic questions leads to a sharp rise in benchmark performance, indicating that these questions significantly enrich image detail. However, when the number exceeds 20, the performance curve begins to plateau, suggesting that most objects mentioned in VQA tasks are already covered. This trend aligns well with the scaling laws of inference observed in previous work.

MMVet

- 46
- 47
- 48
- 49
- 50
- 51 MMStar

60

60

| | | | | |56.|7<br><br>58|.6 58.8| |
|---|---|---|---|---|---|---|---|---|
| | | |5.4|56.|7| | | |
| | | | | | | | | |
| |5|4.|2| | | | | |
| | | | | | | | | |
| |51|.8| | | | | | |
| | | | | | | | | |

| | | | | |49|.8 49|.9<br><br>50.|3|
|---|---|---|---|---|---|---|---|---|
| | | |49.|1<br><br>49|.3| | | |
| |4|8.|3| | | | | |
| | | | | | | | | |
| |46|.9| | | | | | |
| | | | | | | | | |

Qwen2-VL-2B Qwen2-VL-7B Qwen2-VL-72B

51.7

51.5

58

Accuracy(%)

50

42.9

56

Scores

41.0

39.0

40

54

30.5

30

52

50

20

0 2 6 10 15 20 all

0 2 6 10 15 20 all

Recognition Konwledge MMVet Category

Number of Instructions

Number of Instructions

(a)

(b)

- Figure 7: (a) Caption scaling. We adjust the number of instructions N used in ScaleCap to explore changes in benchmark performance within the Prism framework. (b) As the LVLM in ScaleCap scales up, perception capability saturates at 7B, while world knowledge continues to increase.

Table 6: Ablation on LVLM and LLM scale in benchmarks subset. As the LVLM scales up, caption quality saturates from 7B; as the LLM scales up, caption quality continues to improve markedly.

Table 7: Hallucination evaluation results. Golden Sentence Selection strategy performs the best.

MM Vet

MM Star

Chart QA

Text VQA

Method CHAIRS↓ CHAIRI↓

LVLM LLM

Average

LLaVA-v1.5 7B 48.8 13.9 +VCD 46.8 13.2 +OPERA 44.6 12.8 +Golden Sentence 33.6 11.3

Qwen2-VL-2B Qwen2-72B 54.0 43.6 58.8 44.4 50.2 Qwen2-VL-7B Qwen2-72B 58.8 49.5 72.5 53.2 58.5 Qwen2-VL-72B Qwen2-72B 59.0 52.3 69.4 54.1 58.7

Qwen2-VL-7B Qwen2-7B 45.5 40.0 53.0 47.5 46.5 Qwen2-VL-7B Qwen2-72B 58.8 49.5 72.5 53.2 58.5

Qwen2-VL 7B 44.2 7.5 +Golden Sentence 25.8 6.8

LVLM with 7B size is sufficient and efficient. In this section, we study the scale of the LVLM used for visual information extraction. Here we use the Qwen2-VL series from 2B to 72B and evaluate the benchmark performance under the Prism setting above. Due to the high inference cost of large LVLMs, we use a randomly sampled 300-question subset from MMStar, ChartQA, and TextVQA. As shown in the first part of Table 6, we find that LVLM from 2B to 7B shows a promising improvement, but the further improvement for 72B is minor. This is consistent with the conclusion in Prism and proves our analysis that a 7B size model is capable enough for extracting most visual information.

Large LVLM introduces more world knowledge in the caption. Despite the minor gap between 7B and 72B models, we further analyze the difference in Figure 7b by different categories of questions in MMVet. We find the models perform similarly at the recognition question, but larger models get better performance on questions that are related to world knowledge, which is reasonable because a model with larger size contains more knowledge and could introduce it in the caption.

Small LLM struggle with complex visual information integration. Then we study the influence of the LLM used for heuristic question raising and information integration. As shown in the second part of Table 6, the 7B LLM shows significantly worse performance than the 72B model. To position the performance bottleneck, we first compare the question-raising quality and find the gap is minor. So we use the same context before the Caption Integration Phase and evaluate the integration quality between models in Table 5, and observe a consistent performance gap with Table 6. This indicates a substantial difference in summarization capabilities among model sizes. During summarization, the combined object details can result in a context length of up to 20k tokens, which causes Qwen2-7B to miss important information due to its limited ability to handle long contexts.

Object and Position Instructions are equally important In ScaleCap, questions are raised for both the object and its position. Here we study their effectiveness in Table3 within the Prism framework. Removing either Object Instructions or Position Instructions results in a notable drop in performance.

Ablating Contrastive Sentence Rating Strategy Here we study the effectiveness of the Contrastive Sentence Rating Module on CHAIR, a benchmark specialized for hallucination evaluation. As shown in Table 7, Contrastive Sentence Rating strategy significantly mitigates hallucination in the LLaVA1.5, outperforming baselines such as OPERA[21] and VCD[28]. It also achieves notable hallucination reduction on capable models like Qwen2-VL-7B, proving the effectiveness and generalization of the Contrastive Sentence Rating Module in hallucination elimination.

## 5 Related Work

Hallucination in LVLMs. Although the field of artificial intelligence is developing rapidly [7, 64, 22, 14, 32, 45, 57, 38–40, 50, 33, 58], hallucination remains a significant challenge in LVLMs [3, 10, 36], despite the rapid development of multimodal. Object hallucination occurs when large vision-language models produce textual descriptions that mention objects or attributes that are not actually present in the corresponding image. This issue is commonly seen in tasks like image captioning and visual question answering, where it is essential to ensure a precise correlation between the visual and textual elements [20, 31]. Currently, a range of methods have been proposed to address hallucination. These methods can be broadly categorized into two types: one requires training, with numerous approaches utilizing diversified training data to enhance the instruction tuning phase [61], while others leverage preference data through DPO or other reinforcement learning strategies to mitigate hallucinations [51, 63, 67]. Another approach is training-free [23, 56, 55, 24], with notable works such as OPERA [21], which employs a novel MLLM decoding method based on an over-trust penalty and a retrospection-allocation strategy that addresses the internal causes of hallucinations. VCD [28] contrasts the output distributions derived from original and distorted visual inputs to mitigate overreliance on statistical biases and unimodal priors. These methods work during the decoding phase, which is online, and detect hallucinations at the token level. We propose a sentence-level hallucination detection method, which is performed after the entire sentence is generated. This approach can enhance the coherence of the sentence and improve the stability of detection.

Image Caption. To enhance LVLMs, early works focused on large-scale image-text datasets. CC3M [48] and CC12M [4] leveraged web-crawled alt-text but lacked fine-grained details, while manually annotated datasets like SBU [54] and COCO-Captions [9] offered higher quality but struggled with contextual richness. Later efforts [15, 26, 62, 47, 18, 42] improved captions using LLMs. LLaVA [35] introduced human-annotated captions and bounding boxes to guide GPT4 but remained annotation-heavy. ShareGPT4V [6] constructs a large-scale dataset with 1.2M highly descriptive captions, demonstrating significant improvements in LVLMs’ performance across multiple benchmarks. DCE [49] enhances captions with fine-grained attributes and 3D spatial relationships using open-source visual specialists, optimizing cost-effective annotation for complex scenes. Perceptual Fusion [30] leverages high-resolution image processing and multi-expert signal fusion (detection, OCR, tagging) to train a scalable caption engine for open-domain visual perception.

## 6 Limitation

While ScaleCap effectively mitigates linguistic bias through sentence-level contrastive decoding, our current approach focuses solely on identifying and eliminating hallucinations from a probabilistic perspective by comparing sentence likelihoods across with or without image setting. However, this strategy lacks explicit supervision over the semantic content of the captions. Consequently, if the generated captions inherently contain biased, harmful, or offensive content—arising from either the image itself or the underlying tendencies of the pretrained language model—our method is unable to detect or suppress such content. This poses a limitation of ScaleCap in addressing deeper ethical or societal biases embedded within the model or triggered by ambiguous visual stimuli. Future work could explore the integration of external knowledge or content-level bias supervision to enhance the robustness and safety of generated captions.

## 7 Conclusion

In this work, we present ScaleCap, an inference-time scalable image captioning framework. By integrating heuristic question answering and contrastive sentence rating, ScaleCap progressively enriches and calibrates captions with increased inference budget, resulting in more detailed and balanced descriptions. Extensive experiments demonstrate that captions generated by ScaleCap not only excel in downstream tasks such as VQA and image reconstruction, but also significantly boost LVLM pretraining when used at scale, paving the way for high-quality captioning systems.

## References

- [1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint

- arXiv:2308.12966, 2023.
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [3] Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024.
- [4] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3558–3568, 2021.
- [5] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023.
- [6] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370–387. Springer, 2024.
- [7] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Zhenyu Tang, Li Yuan, et al. Sharegpt4video: Improving video understanding and generation with better captions. Advances in Neural Information Processing Systems, 37:19472–19495, 2024.
- [8] Lin Chen and Long Xing. Open-llava-next: An open-source implementation of llava-next series for facilitating the large multi-modal model community. https://github.com/xiaoachen98/Open-LLaVA-NeXT, 2024.
- [9] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015.
- [10] Chenhang Cui, Yiyang Zhou, Xinyu Yang, Shirley Wu, Linjun Zhang, James Zou, and Huaxiu Yao. Holistic analysis of hallucination in gpt-4v (ision): Bias and interference challenges. arXiv preprint arXiv:2311.03287, 2023.
- [11] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Albert Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. ArXiv, abs/2305.06500, 2023.
- [12] Shengyuan Ding, Shenxi Wu, Xiangyu Zhao, Yuhang Zang, Haodong Duan, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Mm-ifengine: Towards multimodal instruction following. arXiv preprint arXiv:2504.07957, 2025.
- [13] Hongyuan Dong, Jiawen Li, Bohong Wu, Jiacong Wang, Yuan Zhang, and Haoyuan Guo. Benchmarking and improving detail image caption. arXiv preprint arXiv:2405.19092, 2024.
- [14] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198–11201, 2024.
- [15] Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, and Yonglong Tian. Improving clip training with language rewrites. Advances in Neural Information Processing Systems, 36:35544–35575, 2023.
- [16] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19358–19369, 2023.
- [17] Tinglei Feng, Yingjie Zhai, Jufeng Yang, Jie Liang, Deng-Ping Fan, Jing Zhang, Ling Shao, and Dacheng Tao. Ic9600: A benchmark dataset for automatic image complexity assessment. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(01):1–17, 2023.
- [18] Roopal Garg, Andrea Burns, Burcu Karagol Ayan, Yonatan Bitton, Ceslee Montgomery, Yasumasa Onoe, Andrew Bunner, Ranjay Krishna, Jason Baldridge, and Radu Soricut. Imageinwords: Unlocking hyper-detailed image descriptions, 2024.
- [19] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [20] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385, 2024.
- [21] Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13418–13427, 2024.
- [22] Qidong Huang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. Deciphering cross-modal alignment in large vision-language models with modality integration rate. arXiv preprint arXiv:2410.07167, 2024.
- [23] Fushuo Huo, Wenchao Xu, Zhong Zhang, Haozhao Wang, Zhicheng Chen, and Peilin Zhao. Selfintrospective decoding: Alleviating hallucinations for large vision-language models. arXiv preprint arXiv:2408.02032, 2024.

- [24] Zhehan Kan, Ce Zhang, Zihan Liao, Yapeng Tian, Wenming Yang, Junyuan Xiao, Xu Li, Dongmei Jiang, Yaowei Wang, and Qingmin Liao. Catch: Complementary adaptive token-level contrastive decoding to mitigate hallucinations in lvlms. arXiv preprint arXiv:2411.12713, 2024.
- [25] Yushan Lai, Guowen Li, Haoyuan Liang, Juepeng Zheng, and Zhiyu Ye. Adu: Adaptive detection of unknown categories in black-box domain adaptation. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 30588–30598, June 2025.
- [26] Zhengfeng Lai, Haotian Zhang, Bowen Zhang, Wentao Wu, Haoping Bai, Aleksei Timofeev, Xianzhi Du, Zhe Gan, Jiulong Shan, Chen-Nee Chuah, et al. Veclip: Improving clip training via visual-enriched captions. In European Conference on Computer Vision, pages 111–127. Springer, 2024.
- [27] LAION. LAION-5B: A New Era of Open Large-scale Multimodal Datasets, 2022. Accessed: 2025-04-22.
- [28] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13872– 13882, 2024.
- [29] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR, 2022.
- [30] Xiaotong Li, Fan Zhang, Haiwen Diao, Yueze Wang, Xinlong Wang, and Ling-Yu Duan. Densefusion-1m: Merging vision experts for comprehensive multimodal perception. arXiv preprint arXiv:2407.08303, 2024.
- [31] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.
- [32] Chen Lin and Xing Long. Open-llava-next: An open-source implementation of llava-next series for facilitating the large multi-modal model community. GitHub-xiaoachen98/Open-LLaVA-NeXT: AnopensourceimplementationfortrainingLLaVA-NeXT, 2024.
- [33] Cuiyu Liu, Wei Zhai, Yuhang Yang, Hongchen Luo, Sen Liang, Yang Cao, and Zheng-Jun Zha. Grounding 3d scene affordance from egocentric interactions. arXiv preprint arXiv:2409.19650, 2024.
- [34] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024.
- [35] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.
- [36] Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253, 2024.
- [37] Shi Liu, Kecheng Zheng, and Wei Chen. Paying more attention to image: A training-free method for alleviating hallucination in lvlms. In European Conference on Computer Vision, pages 125–140. Springer, 2024.
- [38] Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, et al. Mmdu: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for lvlms. arXiv preprint arXiv:2406.11833, 2024.
- [39] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.
- [40] Ziyu Liu, Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Haodong Duan, Conghui He, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Mia-dpo: Multi-image augmented direct preference optimization for large vision-language models. arXiv preprint arXiv:2410.17637, 2024.
- [41] Meta AI. LLaMA 4 Models, 2025. Accessed.
- [42] Yasumasa Onoe, Sunayana Rane, Zachary Berger, Yonatan Bitton, Jaemin Cho, Roopal Garg, Alexander Ku, Zarana Parekh, Jordi Pont-Tuset, Garrett Tanzer, et al. Docci: Descriptions of connected and contrasting images. In European Conference on Computer Vision, pages 291–309. Springer, 2024.
- [43] OpenAI. Chatgpt. https://chat.openai.com/, 2023.
- [44] OpenAI. Gpt-4v(ision) system card. https://cdn.openai.com/papers/GPTV_System_Card.pdf, 2023.
- [45] Zhangyang Qi, Yunhan Yang, Mengchen Zhang, Long Xing, Xiaoyang Wu, Tong Wu, Dahua Lin, Xihui Liu, Jiaqi Wang, and Hengshuang Zhao. Tailor3d: Customized 3d assets editing and generation with dual-side images. arXiv preprint arXiv:2407.06191, 2024.
- [46] Yuxuan Qiao, Haodong Duan, Xinyu Fang, Junming Yang, Lin Chen, Songyang Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. Prism: A framework for decoupling and assessing the capabilities of vlms. arXiv preprint arXiv:2406.14544, 2024.
- [47] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13009–13018, 2024.
- [48] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018.

- [49] Yanpeng Sun, Jing Hao, Ke Zhu, Jiang-Jiang Liu, Yuxiang Zhao, Xiaofan Li, Gang Zhang, Zechao Li, and Jingdong Wang. Descriptive caption enhancement with visual specialists for multimodal perception. arXiv preprint arXiv:2412.14233, 2024.
- [50] Zeyi Sun, Ziyang Chu, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. X-prompt: Towards universal in-context image generation in auto-regressive vision language foundation models. arXiv preprint arXiv:2412.01824, 2024.
- [51] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, LiangYan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. arXiv preprint arXiv:2309.14525, 2023.
- [52] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [53] Jack Urbanek, Florian Bordes, Pietro Astolfi, Mary Williamson, Vasu Sharma, and Adriana RomeroSoriano. A picture is worth more than 77 text tokens: Evaluating clip-style models on dense captions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26700– 26709, 2024.
- [54] Yoshitaka Ushiku, Masataka Yamaguchi, Yusuke Mukuta, and Tatsuya Harada. Common subspace for model and similarity: Phrase learning for caption generation from images. In Proceedings of the IEEE international conference on computer vision, pages 2668–2676, 2015.
- [55] David Wan, Jaemin Cho, Elias Stengel-Eskin, and Mohit Bansal. Contrastive region guidance: Improving grounding in vision-language models without training. In European Conference on Computer Vision, pages 198–215. Springer, 2024.
- [56] Xintong Wang, Jingheng Pan, Liang Ding, and Chris Biemann. Mitigating hallucinations in large visionlanguage models with instruction contrastive decoding. arXiv preprint arXiv:2403.18715, 2024.
- [57] Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Jian Tong, Haodong Duan, Qipeng Guo, Jiaqi Wang, et al. Videorope: What makes for good video rotary position embedding? arXiv preprint arXiv:2502.05173, 2025.
- [58] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247, 2024.
- [59] Hu Xu, Po-Yao Huang, Xiaoqing Ellen Tan, Ching-Feng Yeh, Jacob Kahn, Christine Jou, Gargi Ghosh, Omer Levy, Luke Zettlemoyer, Wen-tau Yih, et al. Altogether: Image captioning via re-aligning alt-text. arXiv preprint arXiv:2410.17251, 2024.
- [60] Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, and Enhong Chen. Woodpecker: Hallucination correction for multimodal large language models. Science China Information Sciences, 67(12):220105, 2024.
- [61] Qifan Yu, Juncheng Li, Longhui Wei, Liang Pang, Wentao Ye, Bosheng Qin, Siliang Tang, Qi Tian, and Yueting Zhuang. Hallucidoctor: Mitigating hallucinatory toxicity in visual instruction data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12944–12953, 2024.
- [62] Qiying Yu, Quan Sun, Xiaosong Zhang, Yufeng Cui, Fan Zhang, Yue Cao, Xinlong Wang, and Jingjing Liu. Capsfusion: Rethinking image-text data at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14022–14032, 2024.
- [63] Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13807–13816, 2024.
- [64] Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, et al. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. arXiv preprint arXiv:2407.03320, 2024.
- [65] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1724–1732, 2024.
- [66] Xiangyu Zhao, Shengyuan Ding, Zicheng Zhang, Haian Huang, Maosong Cao, Weiyun Wang, Jiaqi Wang, Xinyu Fang, Wenhai Wang, Guangtao Zhai, Haodong Duan, Hua Yang, and Kai Chen. Omnialign-v: Towards enhanced alignment of mllms with human preference. arXiv preprint arXiv:2502.18411, 2024.
- [67] Ke Zhu, Liang Zhao, Zheng Ge, and Xiangyu Zhang. Self-supervised visual preference alignment. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 291–300, 2024.

## A Prompts Used in ScaleCap

Prompts used in ScaleCap and Prism are presented in Figure 8 and Figure 9.

Instructions for summarize object details:

[Figure 161]

Descriptions: {}

Collect all details about each object from the descriptions, including detailed appearance, structure, material, and special marks or logos. Do not include any analysis or your opinions.

Instructions for summarize position details:

[Figure 162]

Descriptions: {}

Extract and abstract only the position information about each object from the decriptions. Do not include any analysis or your opinions.

Instructions used for other LVLMs in Prism:

Describe the fine-grained content of the image, including scenes, objects, relationships, instance location, and any text present.

[Figure 163]

Instructions used for LLM in Prism:

You are an excellent text-based reasoning expert. You are required to answer the question based on the detailed description of the image. Description: {} Question: {}

Instructions used for summarizing object and position detsils:

[Figure 164]

Basic Context: {}

Object Information: {}

Position Information: {}

Following the logic of the above Basic Context, organize all details provided in Object Information and Position Information to give a very comprehensive description about the image. Do not include any analysis or your opinions.

Figure 8: Prompts used in ScaleCap and Prism.

## B Dataset processing

Data Source and Processing. In collecting images for our dataset, we primarily focus on two aspects: diversity and richness of image content. Given that the ShareGPT4V-100k already includes a wide range of categories, such as artworks, landmarks, etc., it inherently offers a certain level of diversity. Therefore, we opt to directly incorporate these images into our dataset. To further enhance the dataset’s diversity and to obtain more content-rich images, we additionally select 350k images from the LAION-5B[27] dataset. The LAION-5B dataset is sourced from publicly available content on the internet, encompassing a wide range of subjects, styles, and domains due to its web-based origin. This ensures a high level of diversity in the collected data. During filtering, we retain only images with high resolution and moderate complexity. During the image selection process, we filter out images with a short-edge resolution of less than 600 pixels to preserve the richness of visual content. To further ensure the complexity and informativeness of the images, we employed [17] to score image complexity. Images were filtered based on a complexity range of [0.4, 0.8], which helped exclude both overly simplistic and excessively complex images, thereby maintaining the overall quality of the dataset. During the image selection process, we also conduct manual sampling to filter out potentially harmful images.

Caption Model Selection. As previously mentioned, ScaleCap leverages the collaboration between a Vision Language Model and a Large Language Model to generate high-quality captions. For LVLM, as we discussed above, a small model is capable of capturing visual content, so we use Qwen2-VL-7B by default. When it comes to the LLM, the question-raising task is relatively simple, while the integration of complex and abundant visual information within thousands of tokens requires advanced reasoning capability, so we resort to Qwen2-72B based on an empirical study.

Your task is to convert each Object mentioned in a given sentence into a corresponding instruction, and all the resulting instructions are output as "Describe more details about the [Object]". Ensure your instructions do not cover the raw question, options, or thought process of answering the instructions. You should ignore the Objects that appear in some inferences, such as the sentences that begins with 'it might be' or 'there are probably'. Sentence: The image depicts a man in a suit and tie jumping in the air above a bed in a bedroom Instructions: Describe more details about the man. Describe more details about the suit. Describe more details about the tie. Describe more details about the bed. Describe more details about the bedroom.

Sentence: The train appears to be the main subject of the image, showcasing its sleek design and modern appearance Instructions: Describe more details about the train.

Sentence: The table has a few other items on it, including a camera, a jar of jam, and a spoon, suggesting that there might be some people ready to eat Instructions: Describe more details about the table. Describe more details about the camera. Describe more details about the jam. Describe more details about the spoon.

The object prompt with in-context examples to guide the LLM in generating a set of object-related instructions

[Figure 165]

Sentence: The text "You see the world as you are!" is a playful and thought-provoking statement, encouraging viewers to appreciate their unique qualities and perspectives Instructions: Describe more details about the text.

Sentence:

1. **Preheat the Oven**: Preheat your oven to 350\u00b0F (175\u00b0C). Instructions: Describe more details about the oven. Describe more details about the preheat temperature.

Sentence: {} Instructions:

Figure 9: Prompt Tict used in ScaleCap to generate object instructions.

## C Pretraining Details

Training Setting. We follow the training strategy consistent with ShareGPT4V, dividing the overall training process into three stages. (1) Initial Pretraining Stage. In this stage, we train the projector from scratch using the BLIP-558K dataset for pre-alignment. The objective is to establish an initial mapping between visual and textual modalities. We adopt a learning rate of 1e-3 and a batch size of 256. (2) Further Pretraining Stage. We initialize the projector with the weights obtained from the initial pretraining stage. Both the projector and the LLM are jointly optimized using high-quality image caption in this stage. This stage leverages high-quality image-text pairs to achieve effective alignment of the visual features extracted by the vision encoder and the corresponding text features in the LLM. We set the learning rate to 4e-5 and the batch size to 256. (3) Instruction-tuning Stage. During this stage, we use open-source datasets Open-LLaVA-NeXT-Instruct-1M[8] to finetune the projector and the LLM. Previous research has demonstrated the effectiveness of using open-source datasets such as Open-LLaVA-NeXT-Instruct-1M, which includes diverse data sources such as DocVQA and SynDog-EN. We adopt this dataset directly for instruction tuning. During this stage, both the projector and the LLM are jointly fine-tuned. A learning rate of 2e-5 and a batch size of 128 are used.

Baselines. (1) Vanilla. We select the model trained through only two stages, namely the Initial Pretraining Stage and the Instruction-tuning Stage, as our basic baseline. This model does not undergo any additional further pretraining and corresponds to the default configuration of the original LLaVA-NeXT. (2) ShareGPT4V-450k. For fair comparison, the amount of data is strictly aligned with that of ScaleCap-450k. Specifically, ShareGPT4V-450k consists of ShareGPT4V-100k and 350k samples randomly selected from ShareGPT4V-PT. (3) DenseFusion-450k. It is a subset randomly sampled from DenseFusion-1M, which is a high-quality dataset constructed using diverse perception experts as image priors to provide explicit information on visual elements. During the training, we use different datasets only in the Further Pretraining Stage, while keeping other settings exactly the same to ensure a fair comparison.

Our experiments are conducted on 8 A100 GPUs. The initial pretraining stage using BLIP-558K took approximately 2.5 hours, the further pretraining using ScaleCap-450K required 13.5 hours, and the final instruction-tuning stage took 17.2 hours.

## D Potencial Societal Impacts

The strategy for generating detailed captions has the potential to positively impact society in several ways. By providing richer and more descriptive information about visual content, it can significantly improve accessibility for individuals with visual impairments, allowing them to better understand and engage with visual media. This can also enhance educational experiences by supporting learning tools that rely on clear visual explanations, making complex materials more understandable. Furthermore, detailed captions can improve the organization and retrieval of digital content, aiding content moderation, digital archiving, and searchability. In cross-cultural contexts, such captions may help bridge language and cultural barriers by offering clearer context for translation and interpretation.

However, this technology also carries potential negative societal impacts. Detailed captions may unintentionally disclose private or sensitive information captured in images, raising serious privacy concerns. If the system is trained on biased or unrepresentative data, it may reinforce harmful stereotypes or propagate discriminatory narratives. Additionally, if captions are overly interpretive or inaccurate, they could mislead users and contribute to the spread of misinformation. There is also a risk that over-reliance on automated captioning might diminish critical human oversight, especially in domains where precise and context-aware interpretation is crucial. Therefore, careful consideration must be given to the ethical and societal implications of deploying such systems.

## E User Study Instructions

To directly compare the richness and accuracy of captions generated by different LVLMs, we use the captions produced by each model as prompts for FLUX to generate corresponding images. In the following questionnaire, please rank the images based on their similarity to the original image.

## NeurIPS Paper Checklist

### 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: In both the abstract and introduction, we emphasize that the key contribution of our work is the development of a pipeline for generating high-quality captions. Our method is specifically designed to be compatible with open-source models.

Guidelines:

- • The answer NA means that the abstract and introduction do not include the claims made in the paper.
- • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

#### 2. Limitations Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discuss the limitations in Sec. 6. Guidelines:

- • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- • The authors are encouraged to create a separate "Limitations" section in their paper.
- • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

### 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [NA] Justification: Our formulations are intended solely for methodological illustration and do not include theoretical results. Guidelines:

- • The answer NA means that the paper does not include theoretical results.
- • All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- • All assumptions should be clearly stated or referenced in the statement of any theorems.
- • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- • Theorems and Lemmas that the proof relies upon should be properly referenced.

### 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: What we propose is a caption generation pipeline. We have provided a detailed explanation of each component in Section 2, and included all the prompts used in the Appendix, along with all training details. These materials are sufficient to reproduce our experimental results.

Guidelines:

- • The answer NA means that the paper does not include experiments.
- • If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- • While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

### 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes] Justification: We provide all the code used for experiments in supplemental material. Guidelines:

- • The answer NA means that paper does not include experiments requiring code.
- • Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.
- • While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
- • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

### 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes] Justification: All the training and test details are included in Appendix and Sec. 4. Guidelines:

- • The answer NA means that the paper does not include experiments.
- • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- • The full details can be provided either with the code, in appendix, or as supplemental material.

### 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No] Justification: Due to the high cost of model training and benchmark evaluation, we did not report error bars. Guidelines:

- • The answer NA means that the paper does not include experiments.
- • The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- • The assumptions made should be given (e.g., Normally distributed errors).
- • It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- • If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

### 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes] Justification: We provide sufficient information in Appendix.C on the computer resources for reproducing the experiments. Guidelines:

- • The answer NA means that the paper does not include experiments.
- • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

### 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes] Justification: The research conducted in the paper conform with the NeurIPS Code of Ethics. Guidelines:

- • The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- • If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- • The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes] Justification: Potential positive societal impacts and negative societal impacts are discussed in Appendix. D Guidelines:

- • The answer NA means that there is no societal impact of the work performed.
- • If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [Yes] Justification: We will release the dataset under an appropriate license to prevent improper use and we have sampled images to filter out harmful images in Appendix. B. Guidelines:

- • The answer NA means that the paper poses no such risks.
- • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes] Justification: The license and terms of use explicitly are mentioned and properly respected. We cited the creators of DenseFusion dataset and ShareGPT4V dataset. Guidelines:

- • The answer NA means that the paper does not use existing assets.
- • The authors should cite the original paper that produced the code package or dataset.
- • The authors should state which version of the asset is used and, if possible, include a URL.
- • The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- • If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes] Justification: Our caption dataset are well introduced in Appendix. B. Dataset will be publicly available at huggingface. Guidelines:

- • The answer NA means that the paper does not release new assets.
- • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- • The paper should discuss whether and how consent was obtained from people whose asset is used.
- • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [Yes] Justification: We provide the instructions used for user study in Appendix. E. Guidelines:

- • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [No] Justification: Our user study is limited to comparing image similarity. Guidelines:

- • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

### 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [Yes] Justification: We incorporated an LLM into our pipeline and provided a detailed explanation of how it was utilized. Guidelines:

- • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.

