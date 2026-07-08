## VistaDPO : Video Hierarchical Spatial-Temporal Direct Preference Optimization for Large Video Models

# arXiv:2504.13122v1[cs.CV]17Apr2025

Haojian Huang*1 Haodong Chen*2 Shengqiong Wu3 Meng Luo3 Jinlan Fu3 Xinya Du4 Hanwang Zhang5 Hao Fei3

### Abstract

Large Video Models (LVMs) built upon Large Language Models (LLMs) have shown promise in video understanding but often suffer from misalignment with human intuition and video hallucination issues. To address these challenges, we introduce VistaDPO, a novel framework for Video Hierarchical SpatialTemporal Direct Preference Optimization. VistaDPO enhances text-video preference alignment across three hierarchical levels: i) Instance Level, aligning overall video content with responses; ii) Temporal Level, aligning video temporal semantics with event descriptions; and iii) Perceptive Level, aligning spatial objects with language tokens. Given the lack of datasets for fine-grained video-language preference alignment, we construct VistaDPO-7k, a dataset of 7.2K QA pairs annotated with chosen and rejected responses, along with spatial-temporal grounding information such as timestamps, keyframes, and bounding boxes. Extensive experiments on benchmarks such as Video Hallucination, Video QA, and Captioning performance tasks demonstrate that VistaDPO significantly improves the performance of existing LVMs, effectively mitigating video-language misalignment and hallucination. The code and data are available at VistaDPO Repository.

### 1. Introduction

Achieving human-like reasoning capabilities for videos is a critical research topic in the field of AI. In recent years, Large Video Models (LVMs) (Li et al., 2023; Zhang et al., 2023a; Lin et al., 2023; Li et al., 2024d; Wu et al., 2024a; Cheng et al., 2024b; Fei et al., 2024b; Jin et al., 2024; Qian et al., 2024; Li et al., 2025) have garnered significant research attention. Built upon Large Language Models

*Equal contribution 1The University of Hong Kong 2The Hong Kong University of Science and Technology 3National University of Singapore 4University of Texas at Dallas 5Nanyang Technological University. Correspondence to: Hao Fei <haofei37@nus.edu.sg>.

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

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

(a) Textual DPO (b) Previous Multimodal DPO

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

(c) VistaDPO

[Figure 49]

: "What object did the person put down in the video?" : "The person put down the shoe in the video." : "... placed some clothes on the ground." : "... a musical note instead."

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

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

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

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Perceptive Level Temporal Level Instance Level

(d) VistaDPO Overview

Figure 1. (a) Traditional textual DPO overlooks multimodal information, limiting video-language tasks. (b) Existing multimodal DPO methods rely on coarse alignment, missing rich temporal and perceptual details. (c&d) VistaDPO overcomes these limitations with a hierarchical spatiotemporal preference optimization framework, enabling fine-grained video-language alignment and precise reasoning over video dynamics. Here, yw is the preferred response over yl, and vw the visual input more likely to produce it than vl.

(LLMs) (Touvron et al., 2023; Bai et al., 2023; Peng et al., 2023; Dubey et al., 2024), LVMs leverage the powerful intelligence of LLMs in language, achieving unprecedented understanding of video content. However, increasing studies reveal that LVMs encounter critical issues, such as video understanding that deviates from human intuition (Zhou et al., 2024a; Fei et al., 2024a; Cheng et al., 2024a; Hu et al., 2024) or the phenomenon of video hallucination (Wang et al., 2024; Yuan et al., 2024; Ma et al., 2024), where the model outputs content that does not align with the input, e.g., user instructions, video content. The root of these issues lies in the inherent nature of current LVM architectures (Yan et al., 2021; Cheng et al., 2024b; Lin et al., 2023), where most LVMs integrate a video encoder (e.g., ViT) into textoriented LLMs through a connector to achieve video signal

interpretation. Since backbone LLMs undergo extensive pre-training on large-scale language data while video encoders lack peer capability, this gap leads LLMs to produce overly confident outputs based on biased or even incorrect perceptions of video content from the encoder. While the supervised fine-tuning (SFT) with video-language pairs (Wang et al., 2024; Leng et al., 2024; Yuan et al., 2024) can partially improve the alignment between the two modalities in LVMs, fundamentally addressing the issue requires reliance on extremely large-scale data.

Recently, Direct Preference Optimization (DPO) (Rafailov et al., 2024) has been proposed as a promising alternative to SFT. It trains LLMs to prefer responses chosen by evaluators over rejected ones when presented with a user query. By identifying which response better aligns with human preferences rather than requiring precise target outputs, DPO significantly alleviates dependence on annotated data while enhancing alignment with human values and effectively addressing hallucination issues. Some follow-up studies (Xie et al., 2024; Liu et al., 2024f; Zhou et al., 2024b; Fu et al., 2025b) have extended DPO from textual to multimodal LLMs, facilitating cross-modal alignment and improving the generalization capabilities of the models. Most recently, Hound-DPO (Zhang et al., 2024b) pioneers a video DPO, demonstrating that tailored rewards through DPO can significantly enhance the performance of LVMs. Unfortunately, we find that this work straightforwardly applies the DPO strategy designed for image-text LLMs to video-language preference alignment (as shown in Figure 1), which introduces two critical limitations. First, Zhang et al. (2024b) fails to adequately consider the temporal characteristics of videos. Unlike static images, videos always require both spatial semantic understanding and dynamic temporal reasoning (Fei et al., 2024c), necessitating a comprehensive modeling of the spatial-temporal attributes of videos. Second, their work focuses solely on coarse-grained alignment between video and language (response text) at the instance level, which may lead to suboptimal preference alignment (Zeng et al., 2024; Gunjal et al., 2024). We emphasize that achieving proper alignment between two modalities requires a fine-grained preference alignment. Intuitively, dynamic videos correspond to paired text at multiple hierarchical levels.

To address these challenges, we propose a novel framework, Video Hierarchical Spatial-Temporal Direct Preference Optimization (namely VistaDPO), aiming to strengthen LVMs. VistaDPO improves text-video preference alignment across hierarchical granularities. Specifically, we design three levels of alignment (as shown in Figure 1):

▶ Instance Level: Matching the overall video content with

the most appropriate response for semantic alignment.

▶ Temporal Level: Aligning video temporal semantics with event descriptions, enabling temporal reasoning.

▶ Perceptive Level: Aligning video spatial objects (i.e., regions of interest) with objective tokens or phrases in the language at a fine-grained semantic level.

To implement such fine-grained preference optimization, we construct a large-scale spatial-temporally grounded video dataset called VistaDPO-7k. We manually annotate 3,878 videos with spatial-temporal groundings in a video QA format, providing high-quality labels for hallucinated and non-hallucinated answers, along with timestamps, keyframes, and bounding boxes of relevant semantics.

We conduct extensive evaluation on benchmarks including Video Hallucination, Video QA, Captioning Tasks, by post-training existing popular LVMs with the proposed VistaDPO. The results show that VistaDPO consistently improves baseline LVMs, achieving significant average improvements of 26.42% over PLLaVA and 53.92% over Video-LLaVA respectively. Through in-depth analysis, we show that VistaDPO effectively and comprehensively captures the dynamic interactions between video content and texts, thanks to its hierarchical spatial-temporal alignment strategy. To summarize, this work contributes in threefold:

- • Propose a novel Video Hierarchical Spatial-Temporal DPO (VistaDPO) mechanism, a more fine-grained DPO strategy to optimize the alignment between video and language in LVMs.
- • Construct and release a large-scale (7.2K) high-quality annotated QA pairs dataset, which can serve as a valuable resource for follow-up video DPO research.
- • Empirically, VistaDPO significantly improves the generalization capabilities of existing LVMs, effectively mitigating video-language misalignment and hallucination.

### 2. Related Work

By building on powerful LLMs and integrating various multimodal encoders, researchers have developed MLLMs (Liu et al., 2024a; Fu et al., 2025a; Yin et al., 2024; Wu et al., 2024b) and LVMs (Li et al., 2023; Zhang et al., 2023a; Lin

- et al., 2023; Li et al., 2024d; Cheng et al., 2024b; Jin et al., 2024; Li et al., 2025). Through necessary SFT on visual instruction-tuning data, MLLMs and LVMs have not only developed robust multimodal understanding capabilities but have also significantly enhanced human-computer interaction, making cross-modal interactions more intuitive and seamless. Unfortunately, inheriting the intrinsic hallucination issues of LLMs, LVMs also frequently suffer from hallucinations (Liu et al., 2024b; Zhang et al., 2024b; Li
- et al., 2024a; Liu et al., 2024e) or fail to align their understanding of visual content with human values. Increasing the volume of multimodal SFT data has been shown to alleviate these issues to some extent (Ahn et al., 2024; Tan et al., 2024; Jiang et al., 2024; Chen et al., 2024a). However, this approach is often accompanied by higher annotation costs and computational expenses. This challenge is partic-

ularly pronounced in video scenarios, where LVMs demand significantly larger datasets and higher training costs.

Subsequently, the community has introduced the DPO technique (Rafailov et al., 2024), where preference alignment aligns LLMs with human values, reducing hallucinations by guiding the model’s adjustments using pairs of preferred and rejected data. Multimodal preference alignment, as an extension of preference alignment techniques to visual and textual inputs, has been widely applied to MLLMs to improve crossmodal alignment (Liu et al., 2024f; Xie et al., 2024; Zhou et al., 2024b) as shown in Table 5. Recently, Hound-DPO, pioneered by Zhang et al. (2024b), successfully applies multimodal DPO to LVMs, improving video understanding and addressing hallucination issues. However, it overlooks the preference alignment of visual inputs. In this paper, we aim to further enhance the effectiveness of DPO in video scenarios by modeling fine-grained alignments between video and language. To achieve this, we propose a hierarchical preference optimization framework that efficiently captures dynamic spatial-temporal dependencies in video tasks.

### 3. Preliminaries

Direct Preference Optimization (DPO) (Rafailov et al., 2024) aligns language models with human preferences, removing the need for explicit reward modeling or reinforcement learning (RL). Given a model πθ (the target model) and a reference policy πref (from supervised fine-tuning), the RL objective in reinforcement learning with human feedback (RLHF), initialized with πθ = πref, is expressed as:

Ex∼D,y∼π

θ(y|x) r(x,y) − βDKL πθ(y | x) ∥ πref(y | x) ,

max

πθ

(1)

where r(x,y) denotes the reward function with x as the input instruction and y as the response. DPO establishes a mapping between the reward model and the optimal policy under the reverse KL divergence, obtaining a representation of the reward function concerning the policy:

πθ(y|x) πref(y|x)

+ β log Z(x), (2)

r(x,y) = β log

where β is a coefficient for the reverse KL divergence penalty, and Z(x) is the partition function.

Given the chosen response yw, preferred over the rejected response yl, DPO aligns with human preference using the Bradley-Terry model for pairwise comparisons:

exp(r(x,yw)) exp(r(x,yw)) + exp(r(x,yl))

. (3)

PBT(yw ≻ yl|x) =

By substituting Eq. 2 into Eq. 3 and leveraging the negative log-likelihood loss, DPO derives the objective function:

πθ(yw|x) πref(yw|x) − β log

πθ(yl|x) πref(yl|x)

u(x,yw,yl) = β log

, LDPO = −E(x,y

(4)

w,yl) [log σ (u(x,yw,yl))],

where the action score with yi denotes the i-th token of the response y can be formulated as:

log p(yi|x,y<i). (5)

log π(y|x) =

yi∈y

### 4. VistaDPO-7k: A Spatial-temporal Grounded Video DPO Dataset

Existing LVMs often suffer from limited spatial-temporal perception, leading to video-language misalignment and hallucination issues (Lan et al., 2024). We propose VistaDPO with spatial-temporal DPO to achieve fine-grained alignment between video and language modalities. To support this, we construct a spatial-temporal grounded dataset, VistaDPO-7k, by integrating data from 14 prevalent video datasets and systematically designing QA pairs to evaluate and mitigate hallucinations. These hallucinations are categorized into two major dimensions: Perception (e.g., Object, Static/Dynamic Attribute, Static Relation, OCR) and Temporal (e.g., Action, Dynamic Relation, Sequence), covering both static and dynamic aspects of video understanding. The dataset provides chosen and rejected responses, along with fine-grained temporal dependencies that include key timestamps, frames, and bounding boxes, enabling models to better capture spatial-temporal interactions, as can be shown in Figure 2(a). VistaDPO-7k supports multilevel preference optimization across Temporal, Perceptive, and Instance levels, offering a robust benchmark to reduce hallucinations and enhance the spatial-temporal reasoning capabilities of LVMs. Please refer to Appendix §B for more details on dataset construction and specifications.

### 5. Methodology

To tackle the spatiotemporal complexities in video-language tasks, we propose VistaDPO, which implements hierarchical preference optimization across three aspects: (i) Instancewise Semantic Preference Optimization, aligning preferences at response and video levels; (ii) Temporal ActionEvent Preference Optimization, capturing overlooked temporal dynamics; and (iii) Perceptive Spatial-Object Preference Optimization, enabling fine-grained alignment between tokens and objects. Figure 2(b) illustrates the overall architecture of VistaDPO.

#### 5.1. Instance-wise Semantic Preference Optimization

Effective video-language alignment hinges on distinguishing preferred (chosen) from non-preferred (rejected) responses while capturing global video content. To address hallucinations and misalignments caused by spatiotemporal complexities and over-reliance on text, we propose response-level alignment to refine preference differentiation and video-level alignment to enhance instance-wise semantic understanding.

[Figure 127]

Action

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

[Figure 147]

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

Object

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

16%

17%

Dynamic Attribute

Te mporal (44 %)

Number

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

4%

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

7%

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

Perception

[Figure 221]

[Figure 222]

[Figure 223]

Which object did the person open in the video?

[Figure 224]

[Figure 225]

[Figure 226]

Location

7%

[Figure 227]

(56 %)

Dynamic Relation

21%

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

11%

Static Relation

Visual Encoder

[Figure 240]

7% 7% 3%

Projection Embedding

[Figure 241]

[Figure 242]

Color

Sequence

OCR

V V V V V V V V V V T T T T T

V

[Figure 243]

:“Which object did the person sit on during the scene?”

[Figure 244]

[Figure 245]

[Figure 246]

Large Language Model

LoRA

###### Key Frame

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

...

...

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

###### Shopping cart Table Appear

Chair Disappear

Tokens Tokens

False True

34.13s

“IS IN VIDEO?”

26.69s

34.13s

22.61s

{...}

{x: 277.0, y: 207.0, w: 70.0, h: 44.0}

[Figure 261]

[Figure 262]

{...}

{...}

[Figure 263]

The person in the video seemed to be handling clothes, folding them neatly on a table.

In the video, the person opened the window, revealing a bright and sunny day outside.

The person in the video seemed to be handling clothes, folding them neatly on a table.

[Figure 264]

[Figure 265]

[Figure 266]

: “... sat on a chair...” For Temporal-level For Perceptive-level

(a) Metadata of VistaDPO-7k (b) Illustration of VistaDPO

Figure 2. (a) The metadata of VistaDPO-7k highlights its focus on fine-grained video-language tasks, emphasizing temporal (44%) and perceptual (56%) reasoning. ylir and ylre denote the irrelevant and relevant non-preferred responses respectively. (b) VistaDPO introduces a hierarchical spatiotemporal preference optimization framework. Instance (vv) and perceptive (vf) levels align global-to-local semantics with spatial visual features, leveraging both text-relevant and irrelevant rejected responses for robust cross-modal interaction. Temporal (vc) level aligns clip-level semantics with temporal dynamics, enabling precise reasoning across spatial and temporal dimensions.

Response-Level Alignment. LVMs often face challenges in maintaining global consistency when generating responses. While these models effectively capture the general context of video input v and prompt x, they frequently struggle to distinguish user-preferred responses yw from non-preferred responses yl at the response level, leading to suboptimal alignment with user intent. To promote overall consistency by encouraging the model to align its response-level preferences with human expectations, the objective function can be formulated as:

w,yl) [log σ (ur(v,x,yw,yl))], (6) where

= −E(v,x,y

LDPOr

πθ(yl|v,x) πref(yl|v,x)

πθ(yw|v,x) πref(yw|v,x) − β log

. (7) Here, log π(y|v,x) is defined as:

ur = β log

log p(yi|v,x,y<i). (8)

log π(y|v,x) =

yi∈y

The existing method of Hound-DPO (Zhang et al., 2024b) directly adopts the above approach, focusing solely on aligning the chosen response with the prompt. Nevertheless, the complex spatial-temporal dependencies in rejected responses are completely neglected. Intuitively, intrinsic hallucinations in generative models typically arise from: 1) erroneously inferring content that does not exist in the video; 2) failing to capture the fine-grained spatial-temporal dependencies of the correct content in the video. To mitigate this, we further introduce two types of non-preferred responses

into the optimization process:

πθ(yli|v,x) πref(yli|v,x)

πθ(yl|v,x) πref(yl|v,x) ←

βi log

log

, (9)

i∈{re,ir}

where ylre denotes the relevant non-preferred for these are semantically relevant to the video content but contain spatial

or temporal inconsistencies, e.g., incorrect temporal ordering, wrong actions, or misinterpreted spatial locations. In contrast, ylir denotes the irrelevant non-preferred responses, which are entirely unrelated to the video content, introducing noise by hallucinating events or objects with no connection to the actual video.

Video-Level Alignment. Unlike most prior DPO works, which focus exclusively on textual optimization, we introduce video-level preference optimization for the first time to reduce LVMs’ overreliance on language. At the video level, the model needs to understand the preference relationships of the entire video as a coherent semantic unit. However, since LVMs are prone to hallucinations involving irrelevant video content, we optimize the model to recognize global discrepancies among videos. To this end, we construct video-level preferred and non-preferred sample pairs, denoted as vwv and vlv. Thus uv(vwv ,vlv,x,yw) within LDPOv

can be formulated as: uv = β log

πθ(yw|vwv ,x) πref(yw|vwv ,x) − β log

πθ(yl|vlv,x) πref(yl|vlv,x)

, (10)

where vlv is sampled from the mini-batch that is unrelated to the query x in this work.

#### 5.2. Temporal Semantic Preference Optimization

Clip-Level Alignment. While previous multimodal DPO methods have mainly focused on the spatial aspects of visual samples (as shown in Table 5), unlike static images, videos require both spatial semantic understanding and dynamic temporal reasoning. This necessitates a comprehensive modeling of the spatial-temporal attributes of videos.

At the temporal level, the model must distinguish between time segments in the video that are relevant to the prompt and those that are irrelevant. To align video temporal semantics with event descriptions provided in the prompt, we treat time segments related to the prompt as preferred clips vwc and time segments unrelated to the prompt as non-preferred clips vlc, as shown in Figure 2. Following Eq. (10), the clip-level objective function can be defined as:

LDPOc ∼ log σ(uc(vwc ,vlc,x,yw)). (11)

##### 5.3. Perceptive Spatial-Object Preference Optimization While instance-wise alignment captures global semantics, fine-grained perceptual alignment is crucial for precise video-language interaction. Videos inherently involve complex spatial relationships, where objects, actions, and regions dynamically interact over time. Language, in turn, encodes these interactions through specific tokens, making it essential to establish detailed alignment between spatial objects and their corresponding linguistic references.

Object-Level Spatial Alignment. At the spatial level, the model needs to capture the key locations and states of objects within the video. However, LVMs are often prone to hallucinations in spatial layouts, leading to incorrect object placements or misinterpretations of scene context. To address this, we strengthen the model’s understanding of spatial information through object-level preferred and non-preferred sample design. Specifically, we select the keyframe relevant to the prompt x as the preferred instance vwf as shown in Figure. 2. For the non-preferred sample vlf, we further apply a masking operation to the key regions within the selected frame, thereby focusing the model’s attention on the relevant spatial content while reducing the influence of irrelevant regions. Accordingly, the object-level loss LDPOo

can be defined in a manner similar to Eq. (11).

Token-Level Alignment. While response-level optimization enhances global consistency, it lacks the granularity required to address token-specific errors, such as misattributed objects or incorrect temporal markers (e.g., “after” vs. “before”). Token-level optimization ensures that the model aligns its preferences at a finer granularity, thereby reducing hallucinations in object-action relationships. Inspired by TDPO (Zeng et al., 2024), we implement token-level optimization to evaluate preferences for individual tokens and align them coherently to form a consistent response. The

sequential KL divergence can be defined as:

= sg βDSeqKL(x,vwf ,yw;πref||πθ) − βDSeqKL(x,vwf ,yl;πref∥πθ),

LDPOt

(12)

where sg represents the stop-gradient operator, ensuring that gradients are not propagated through the reference policy πref, and DSeqKL is the sequence-level KL divergence:

T

DKL(πref(y|x,y<t)∥πθ(y|x,y<t)). (13)

DSeqKL =

t=1

Overall, after incorporating instance-wise, temporal, and perceptive-level preference optimization, the overall loss function for VistaDPO is formulated as follows:

LV istaDPO =LDPOv

+ LDPOr Instance

(14)

+ λLDPOc Temporal

+µLDPOo

+ ρLDPOt Perceptive

,

where λ, µ, and ρ represent the loss weights.

### 6. Experiments

In this section, we empirically investigate the effectiveness of VistaDPO in reducing hallucinations.

#### 6.1. Experimental Settings

Baselines. We apply VistaDPO to two different 7B-size LVMs: Video-LLaVA (Lin et al., 2023) and PLLaVA (Xu et al., 2024). For Video-LLaVA, it employs LanguageBind (Zhu et al., 2023) encoder for visual inputs, and Vicuna-7B v1.5 (Chiang et al., 2023) as the LLM backbone. For PLLaVA, the visual input is processed through ViT-L (Radford et al., 2021) and MM projector, with Vicuna as the LLM backbone. While other LVMs cannot be directly compared due to differences in base models, preference data, and alignment strategies, we provide these results for reference: VideoChatGPT (Maaz et al., 2023), VideoChat2 (Li et al., 2024d), LLaMA-VID (Li et al., 2025), LLaMAAdapter (Zhang et al., 2023b), and Video-LLaMA (Zhang et al., 2023a).

Evaluations. To evaluate the effectiveness of VistaDPO, we adopt benchmarks for three aspects: (1) Video Hallucination: VideoHallucer (Wang et al., 2024) and EventHallusion (Zhang et al., 2024a); (2) General Video QA: MSVDQA (Xu et al., 2017), MSR-VTT-QA (Xu et al., 2017), TGIF-QA (Jang et al., 2017), and ActivityNet-QA (Yu et al., 2019); and (3) Captioning Performance: VideoChatGPTBench (Maaz et al., 2023). For ablation studies and analysis, we mainly employ our VistaDPO on Video-LLaVA.

Implementation Details. We train the Video-LLaVA 7B (Lin et al., 2023) and PLLaVA 7B (Xu et al., 2024) with VistaDPO for 3 epochs, with a learning rate of 5e − 7 and a batch size of 8 on H100 GPUs. For training, we followed

Table 1. Main results on video hallucination benchmarks. Bold values indicate the best performance and ∆ denotes the corresponding improvement percentages over the baselines (i.e. PLLaVA and Video-LLaVA). “↑” denotes higher is better.

VideoHallucer EventHallusion Basic↑ Hallucinated↑ Overall↑ Entire Mix Misleading Overall Binary↑ Desc.↑ Binary↑ Desc.↑ Binary↑ Binary↑ Desc.↑

Models

VideoChatGPT (Maaz et al., 2023) 92.8 10.4 6.4 14.9 5.5 57.0 3.6 21.6 36.4 4.3 VideoChat2 (Li et al., 2024d) 29.7 25.8 7.8 16.7 4.6 12.4 1.6 22.6 16.1 2.6 LLaMA-VID (Li et al., 2025) 89.9 26.6 21.0 30.7 16.5 73.6 7.8 43.1 54.0 10.9

- PLLaVA (Xu et al., 2024) 75.1 55.5 38.1 45.6 16.5 58.5 3.1 81.4 60.6 6.1

+ Hound-DPO (Zhang et al., 2024b) 69.3 58.1 36.2 47.4 19.3 24.9 4.1 83.3 45.7 9.8

+ VistaDPO (Ours) 82.5 72.1 57.8 55.3 23.6 62.2 6.2 97.1 68.9 12.7 ∆% 9.9 29.9 51.7 21.3 42.7 6.3 100.0 19.3 13.7 108.2

Video-LLaVA (Lin et al., 2023) 95.1 20.3 17.8 30.7 8.3 57.5 7.3 41.2 45.9 7.6 + Hound-DPO (Zhang et al., 2024b) 83.4 43.0 29.5 35.9 9.8 15.5 9.3 63.7 33.3 9.5

+ VistaDPO (Ours) 98.2 64.4 54.3 50.9 14.9 62.2 10.4 95.1 67.2 12.1 ∆% 3.3 217.2 205.1 65.8 79.5 8.2 42.5 130.8 46.4 59.2

Table 2. Main results on video QA and captioning benchmarks. Symbols follow the definitions in Table 1.

Models Question-Answer Captioning MSVD↑ MSR-VTT↑ TGIF↑ Act.Net↑ Correct↑ Detail↑ Context↑ Temporal↑ Consist↑

VideoChatGPT (Maaz et al., 2023) 64.9 49.3 51.4 35.2 2.4 2.5 2.6 2.0 2.4 LLaMA-Adapter (Zhang et al., 2023b) 54.9 43.8 - 34.2 2.0 2.3 2.3 2.0 2.2 Video-LLaMA (Zhang et al., 2023a) 51.6 29.6 - 12.4 2.0 2.2 2.2 1.8 1.8

- PLLaVA (Xu et al., 2024) 76.6 62.0 77.5 56.3 3.2 2.9 3.6 2.3 2.9

+ Hound-DPO (Zhang et al., 2024b) 82.3 73.1 79.9 54.7 3.2 2.8 3.4 2.4 2.7 + VistaDPO (Ours) 86.4 80.2 84.3 59.1 3.5 3.0 3.9 2.8 2.9 ∆% 12.8 29.4 8.8 5.0 9.4 3.5 8.3 21.7 0.0

Video-LLaVA (Lin et al., 2023) 71.8 59.0 48.4 45.3 2.8 2.9 3.4 2.5 2.6 + Hound-DPO (Zhang et al., 2024b) 80.7 70.2 61.4 40.9 3.0 2.7 3.3 2.0 2.6 + VistaDPO (Ours) 85.3 76.9 74.1 55.0 3.4 2.9 3.6 2.6 2.9 ∆% 18.8 30.3 53.1 21.5 21.4 0.0 5.9 4.0 11.5

Table 3. Ablation study of level losses on VideoHallucer. HoundDPO (Zhang et al., 2024b) employs the same strategy as DPO (Rafailov et al., 2024), but based on its own constructed dataset.

Methods Basic↑ Hallu.↑ Over.↑

VistaDPO 98.2 64.4 54.3 w/o LDPOc 97.8 62.3 53.0 w/o LDPOo 98.1 62.0 52.8 w/o LDPOo, LDPOt 97.6 61.5 49.4 w/o LDPOo, LDPOt, LDPOc 97.2 60.1 46.6 only w/ LDPOr 95.8 52.3 39.8 Vanilla DPO w/ VistaDPO-7K 95.4 50.8 38.1 Hound-DPO 83.4 43.0 29.5

Zhang et al. (2024b) to set the hyperparameter β = 0.1 and followed Zeng et al. (2024) to set ρ = 0.1 for LDPOt

. As for hyperparameters of LDPOc

, we set λ = 0.4 and µ = 0.2 respectively. Moreover, we set βre = 0.7 and βir = 0.3 for the relevant and irrelevant non-preferred responses respectively for LDPOr

and LDPOo

.

#### 6.2. Main Results

We compare VistaDPO with Hound-DPO (Zhang et al., 2024b) on video hallucination, video QA, and captioning benchmarks to verify the effectiveness of our approach.

Video Hallucination. To benchmark VistaDPO, we focused on the model hallucination problem that DPO posttraining aims to mitigate and compared its performance against the previous video DPO strategy, specifically Hound-

DPO, based on LVMs PLLaVA (Xu et al., 2024) and VideoLLaVA (Lin et al., 2023). As shown in Table 1, we adopted two video hallucination benchmarks, VideoHallucer (Wang et al., 2024) and EventHallusion (Zhang et al., 2024a). The results indicate that VistaDPO significantly alleviates hallucination issues compared to Hound-DPO. Notably, while Hound-DPO improved hallucination-related performance, they introduced undesirable trade-offs, such as reduced accuracy in addressing fundamental categories like the “Basic” class in VideoHallucer. Furthermore, Hound-DPO led to a decline in the model’s descriptive capabilities and accuracy, as observed in the “Desc. (Descriptive)” category of EventHallusion. These limitations highlight the shortcomings of prior methods and underscore the superiority of our VistaDPO framework and the accompanying VistaDPO-7K dataset. To provide a comprehensive assessment of LVMs’ performance post-training, we evaluate both their general and captioning capabilities in the following sections.

Video Question-Answering. In addition to assessing the effectiveness of our VistaDPO in addressing hallucination issues, evaluating the model’s general performance is equally critical. To this end, we conducted evaluations on four commonly used open-ended general question-answering benchmarks in a zero-shot setting, as illustrated on the left side of Table 2. VistaDPO consistently outperforms HoundDPO and demonstrates significant performance improvements on

[Figure 267]

###### Entire Mix Misleading Overall

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8 0.0 0.1 0.3 0.5 0.7 0.1/0.9 0.3/0.7 0.5/0.5 0.7/0.3 0.9/0.1

###### Figure 3. Ablation study of hyperparameters on EventHallusion.

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

100

100 50 0

100

75

50

50

25

PC3

PC3

PC3

0

0

- -50
- -100
- -100

- -75
- -50

- -25

- -50

- -100

- -100

- -50 0

- -50
- -100
- -100
- -50

PC2

PC2

PC2

0

0 50

50

50 100

100

100

150 100 50 0 -50 -100 -150 100 50 0 -50 -100 PC1

150 100 50 0 -50 -100 -150 PC1 PC1

(a) Video-LLaVA (b) +Hound-DPO (c) +VistaDPO

Figure 4. T-SNE visualization of representation. (a) Video-LLaVA shows substantial overlap between hallucinated (orange) and nonhallucinated (green) representations. (b) With Hound-DPO, there is no distinct improvement in the separation of the two clusters. (c) With VistaDPO, the representations achieve clear clustering, highlighting its superior discriminative capability.

both base models. These results indicate that VistaDPO not only mitigates hallucination issues to a large extent but also enhances its ability to comprehend video content and generate accurate responses to questions.

Captioning Capability. We further evaluate the captioning capabilities of the model using the video-based text generation benchmark proposed by Maaz et al. (2023), which assesses five critical dimensions: Correctness, Detail Orientation, Contextual Understanding, Temporal Understanding, and Consistency. As shown on the right of Table 2, VistaDPO consistently outperforms Hound-DPO across all dimensions on two base models. These results highlight VistaDPO’s ability to generate contextually relevant, detailed, and temporally accurate text from video inputs. Moreover, the findings demonstrate that the post-training process with VistaDPO-7K preserves the model’s captioning capabilities, avoiding the degradation observed in Hound-DPO.

#### 6.3. Ablation Studies

To evaluate the contributions of each level and their combinations, we conduct ablation studies on VistaDPO using VideoLLaVA (Table 3). The key findings are as follows: ❶ Effectiveness of Hierarchical Preference Optimization. The hierarchical optimization strategy significantly improves performance, demonstrating its effectiveness in capturing multilevel preferences for better learning and task alignment. ❷ Importance of Spatial-Temporal Dependencies. Spatialtemporal preference optimization, both explicit and implicit, plays a critical role in enhancing DPO performance: (i) VistaDPO explicitly captures spatial-temporal dependencies

through object-level (LDPOo

) optimization, enabling the model to better understand localized temporal and spatial relationships. (ii) Implicitly, it encodes spatial-temporal information via response-level (LDPOr

) and clip-level (LDPOc

) preference alignment, which incorporates both relevant (ylre) and irrelevant (ylir) non-preferred responses. These results highlight the importance of fine-grained spatial-temporal dependencies in video understanding, enabling more robust and effective video-language alignment. ❸ Impact of a Comprehensive High-quality Dataset. Under the vanilla DPO strategy, post-training with VistaDPO-7K outperforms Hound-DPO, which uses a less comprehensive dataset. This demonstrates that a richer and higher-quality dataset improves generalization, enhances performance, and effectively mitigates hallucinations. ❹ Impact of Hyperparameters. Additionally, we conduct hyperparameter ablation study (i.e. Figure 3). Specifically, we analyzed the impact of two hyperparameter sets on VistaDPO performance: ① Loss Weights: The optimal weights for all three levels balance the model’s ability to capture temporal (clip-level λ), spatial (object-level µ), and fine-grained token dependencies (tokenlevel ρ). Too low a weight for any level weakens the model’s ability to capture relevant dependencies, while excessively high weights disrupt the balance, leading to overfitting to specific details and loss of broader context. ② Weights for Relevant/Irrelevant Responses: The combined weight for both non-preferred samples (ylre, ylir) helps the model capture spatial-temporal relationships at the textual level, which also highlights the need for careful hyperparameter tuning to effectively capture spatial-temporal relationships.

### 7. Analyses and Discussions

We now take one step further, providing comprehensive analyses to demonstrate VistaDPO’s superiority.

#### 7.1. Enhanced Video-Language Representation

To empirically demonstrate the effectiveness of VistaDPO, we conduct an analysis from a representational perspective, as illustrated in Figure 4. Specifically using 95 samples (video, non-hallucinated captions, and hallucinated captions) from the “misleading” subset of EventHallusion (Zhang et al., 2024a), we evaluated the alignment of visual and textual embeddings. Video-LLaVA exhibits overlapping features and weak modality alignment, struggling to distinguish hallucinated from non-hallucinated captions.

70

70

70

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

67.2

67.2

67.2

66.0

[Figure 288]

[Figure 289]

[Figure 290]

64.8

64.3

64.3 63.6

65

65

65

[Figure 291]

62.6

62.3

61.1

60.9

Accuracy(%)

60

60

60

58.7

54.3 53.7

54.3

54.3

55

55

55

52.9

53.1

52.9

52.0

52.1

52.2

51.4

50.2

50

50

50

48.8

45

45

45

VideoHallucer EventHallusion VideoHallucer EventHallusion VideoHallucer EventHallusion

(a) Video-level (b) Clip-level (c) Object-level

Figure 5. Ablation study of visual non-preferred samples on two video hallucination benchmarks.

[Figure 292]

[Figure 293]

[Figure 294]

96.0 98.0

94.6 97.3

100

93.4 93.3

80.0

88.0

84.0

80

73.3

72.0

Accuracy(%)

52.3

60

40

20

0

Temporal Absolute Temporal Relative Length Relative

Figure 6. Adversarial temporal testing on VideoHallucer. The gray regions indicate the performance drop under adversarial scenarios for each method.

With Hound-DPO, this issue is partially mitigated through vanilla DPO, but a significant gap between textual and video embeddings remains. In contrast, with VistaDPO, which incorporates hierarchical fine-grained preference modeling, the alignment is significantly improved by narrowing the distance between visual and textual modalities and distinctly separating hallucinated from non-hallucinated captions. These results underscore VistaDPO’s superior capability to unify modalities and effectively reduce hallucination. 7.2. Analysis of Visual Non-preferred Samples

The quality of preference samples depends on the rejection visual samples and the gap between rejection and chosen samples. We explore strategies for constructing rejection samples at the video, clip, and object levels, while keeping the chosen samples (original video, event segment, and keyframe) unchanged for each level as shown in Figure 5.

- • Video-level: (i) Randomness: Select a random sample from the minibatch. (ii) Blackness: Set all RGB values of the chosen sample to 0. (iii) Reverse: Reverse the order of all frames in the chosen sample. (iv) Random Mask: Mask half the frames in the chosen sample.
- • Clip-level: (i) Randomness. (ii) Blackness. (iii) Reverse. (iv) Random Mask. (v) Relevant Segments: Use segments where the event does not occur.
- • Object-level: (i) Randomness. (ii) Blackness. (iii) ROI Mask: Mask the key object in the chosen sample. (iv) ROI Move: Move the key object to disrupt its original spatial relationships.

As demonstrated in Figure 5, we observe the following performance trends: Figure 5 demonstrates the impact of different negative sample construction strategies across video, clip, and object levels on model performance. At the video

[Figure 295]

4.0 3.5 3.0 2.5 2.0 1.5 1.0 0.5 0.0

[Figure 296]

[Figure 297]

4.0 3.5 3.0 2.5 2.0 1.5 1.0 0.5 0.0

- Shift = 1.86 Shift = 1.26

- Shift = 2.42 Shift = 3.85

- -40 -20 0 20 40 -40 -20 0 20 40

- -40 -20 0 20 40 -40 -20 0 20 40

(a) Video-LLaVA (b) w/ Hound-DPO

[Figure 298]

[Figure 299]

4.0 3.5 3.0 2.5 2.0 1.5 1.0 0.5 0.0

4.0 3.5 3.0 2.5 2.0 1.5 1.0 0.5 0.0

[Figure 300]

(c) only w/o (d) w/ VistaDPO

Figure 7. Kernel Density Estimation (KDE) of log-likelihood differences in adversarial masking experiments. The log-likelihood difference measures the separation between original and adversarial distributions, with the shift representing the mean difference. Larger shifts indicate greater model robustness.

level, the Reverse method achieves the highest overall accuracy (67.2%), significantly outperforming Randomness (54.3%), Blackness (50.2%), and Random Mask (52.1%). This suggests that disrupting temporal order provides more informative negative samples compared to random sampling or masking strategies (Chen et al., 2025), which fail to introduce sufficient semantic contrast. At the clip level, Relevant Segments yields the best performance (64.8%), surpassing Randomness (53.1%), Blackness (52.9%), Reverse (61.1%), and Random Mask (62.6%). This highlights that using eventirrelevant segments as negatives more effectively challenges the model to focus on event-specific semantics, whereas random or blackened clips lack meaningful contrast. At the object level, ROI Move achieves the highest accuracy (66.0%), outperforming ROI Mask (64.3%), Randomness (54.3%), and Blackness (53.7%). This indicates that spatially disrupting key objects introduces more challenging and informative negative samples compared to masking or random sampling. Overall, these results emphasize that welldesigned, semantically targeted negative samples—such as those disrupting temporal order, leveraging event irrelevance, or altering spatial relationships—are crucial for enhancing the model’s ability to distinguish fine-grained video-language alignments.

#### 7.3. Adversarial Temporal Testing

To evaluate the robustness of VistaDPO, we conducted adversarial temporal testing using the “Temporal” subset of

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

###### Does the girl play with her pet in the video?

Does 'a man tries to stop a man from shooting a basket ball' happen earlier than 'the ball is passed across the room and men chase it across the court'?

Does a dog run right a person in the video?

[Figure 313]

[Figure 314]

w/ Hound-DPO Yes, ... w/ VistaDPO Yes, ...

w/ Hound-DPO Yes, ... w/ VistaDPO Yes, ...

Yes, ...

Yes, ...

[Figure 315]

Does a dog stand still by a person in the video?

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

w/ Hound-DPO Yes, ... w/ VistaDPO Yes, ...

Yes, ...

[Figure 320]

Yes, ... w/ Hound-DPO No, ... w/ VistaDPO No, ...

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

Does a dog walk right a person in the video?

Does the girl play with her pet in the video?

[Figure 328]

[Figure 329]

Yes, the girl is seen playing with her pet in the video. She is seen teasing the dog and playing with it on the couch.

Yes, a dog walks right past a person in the video.

[Figure 330]

Does '...' happen earlier than '...'?

w/ Hound-DPO Yes, a dog is seen walking alongside the person in the video, indicating a training exercise or a demonstration of control and obedience.

w/ Hound-DPO Yes, the girl appears to play fully interact with the pet, encouraging it to play and move around.

[Figure 331]

Yes, '...' happens earlier than '...'.

w/ Hound-DPO Yes, ... happen earlier than ... only w/o Yes, a dog walks right past a person in the video.The dog seems to be following the person's footsteps.

[Figure 332]

[Figure 333]

only w/o Yes, the girl appears to play and interact

with the pet that is sprawled out on the only w/o Yes, '...' happens earlier than '...'. beige couch.

[Figure 334]

w/ VistaDPO No, 'a man tries to stop a man from shooting a basket ball' happens after than 'the ball is passed across the room and men chase it across the court'

w/ VistaDPO No, it is specifically a dog running towards a person in the video rather walking towards this person.

w/ VistaDPO No, there is no interaction or play between the girl and her pet in the video. There is no direct contact between the girl and her pet throughout.

(a) Temporal adversarial testing demonstration

(c) Token adversarial testing demonstration

(b) Spatial adversarial testing demonstration

Figure 8. Case Studies of Adversarial Testing for VistaDPO: We conduct case studies from three perspectives: (a) Temporal adversarial testing, which examines whether the model can infer the correct sequence of events by introducing reversed temporal order through video playback. (b) Spatial adversarial testing, which evaluates the model’s ability to understand subject-object interactions by masking frames or pixels related to the target object. (c) Token adversarial testing, which tests the model’s sensitivity to subtle linguistic differences by introducing similar action descriptions (e.g., contrasting “run” with “stand” and “walk”). Each test compares VistaDPO with baselines (i.e., Video-LLaVA and Hound-DPO) and corresponding ablated versions to assess the impact of key components.

VideoHallucer (Wang et al., 2024), which includes three categories of video-based QA tasks: (i) Temporal Absolute, focusing on when an event occurs; (ii) Temporal Relative, addressing the order of two events; and (iii) Length Relative, comparing the duration of two events. For adversarial testing, we reversed all videos and adjusted answers to align with the reversed timeline (as shown in Figure 8(a). As shown in Figure 6, the base model (Video-LLaVA) and prior work (Hound-DPO) suffer significant performance drops across all three adversarial scenarios, revealing their inability to effectively model temporal hallucinations and vulnerability to timeline modifications. In contrast, VistaDPO shows minor degradation, demonstrating better temporal awareness and robustness against adversarial challenges.

#### 7.4. Adversarial Spatial Testing

To evaluate spatial adversarial robustness, we test with a video and the question, “Does the girl play with her pet in the video?” As shown in Figure 8(b), all models correctly respond to the original video (upper side). However, in the adversarial version (lower side), where frames are masked to ensure the girl and pet never appear together, only VistaDPO correctly identifies the absence of interaction. To further assess adversarial discriminative capability, we use Kernel Density Estimation (KDE) on the VideoHallucer dataset to visualize how model representations shift when reasoning over noisy (adversarial) samples (see Figure 7). Video-LLaVA achieves a shift value of 1.86, showing limited ability to distinguish between original and adversarial samples. Adding Hound-DPO slightly reduces the shift to 1.26, indicating no improvement. VistaDPO achieves the highest shift value of 3.85, significantly outperforming

other models. Removing LDPOo

reduces the shift to 2.42, highlighting the importance of the proposed spatial-object preference optimization. These show VistaDPO’s superior ability to capture subtle semantic differences and enhance adversarial robustness.

#### 7.5. Adversarial Token Testing

As shown in Figure 8(c), we conduct adversarial token testing to evaluate model robustness. For the original question, “Does a dog run right a person in the video?”, all models answered correctly. When “run” was replaced with “stand” (a significant semantic shift), most models maintained accurate responses. However, with an adversarial sample replacing “run” with “walk” (a subtle semantic change), only VistaDPO correctly captured the nuanced difference. This underscores VistaDPO’s robust token-level understanding, capturing fine-grained semantic shifts and ensuring precise video-language alignment under adversarial conditions.

### 8. Conclusion

In this paper, we propose VistaDPO, a novel framework for Video Hierarchical Spatial-Temporal Direct Preference Optimization, which enhances the alignment between text and video preferences across three hierarchical levels: instance, temporal, and perceptive. To support fine-grained preference alignment, we introduce VistaDPO-7k, a dataset of 7.2K QA pairs with annotations for chosen/rejected responses and spatial-temporal groundings. Extensive evaluations on tasks, i.e., Video Hallucination, Video QA, and Captioning benchmarks demonstrate that VistaDPO significantly improves existing LVMs, addressing video-language misalignment and hallucination issues.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning, particularly in the domain of videolanguage alignment and large video models (LVMs). By introducingVistaDPO, a framework for hierarchical spatialtemporal direct preference optimization, and constructing the VistaDPO-7k dataset, we aim to improve the alignment between video content and human preferences, mitigating issues such as hallucination and misalignment in video-language tasks.

The potential societal impact of this work includes enhancing the robustness and reliability of AI systems in applications such as video analysis, autonomous systems, and multimedia content understanding. While these advancements could contribute positively to fields like education, accessibility, and entertainment, they also raise ethical considerations, including potential misuse in surveillance or biased decision-making if the models are not carefully evaluated for fairness and accountability.

We have taken steps to ensure that the dataset and methodology are designed to reduce biases and hallucinations, and we encourage future researchers to apply these methods responsibly. Beyond these considerations, there are no immediate societal consequences of this work that require specific attention.

### References

Ahn, D., Choi, Y., Yu, Y., Kang, D., and Choi, J. Tuning large multimodal models for videos using reinforcement learning from ai feedback. arXiv preprint arXiv:2402.03746, 2024.

Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M., Valko, M., and Calandriello, D. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pp. 4447–4455. PMLR, 2024.

Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Chen, D. and Dolan, W. B. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, pp. 190–200, 2011.

Chen, H., Huang, H., Dong, J., Zheng, M., and Shao, D. Finecliper: Multi-modal fine-grained clip for dynamic facial expression recognition with adapters. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 2301–2310, 2024a.

Chen, H., Huang, Y., Huang, H., Ge, X., and Shao, D. Gaussianvton: 3d human virtual try-on via multi-stage gaussian splatting editing with image prompting. arXiv preprint arXiv:2405.07472, 2024b.

Chen, H., Wang, L., Yang, H., and Lim, S.-N. Omnicreator: Self-supervised unified generation with universal editing. arXiv preprint arXiv:2412.02114, 2024c.

Chen, H. H., Huang, H., Wu, X., Liu, Y., Bai, Y., Shu, W.J., Yang, H., and Lim, S.-N. Temporal regularization makes your video generator stronger. arXiv preprint arXiv:2503.15417, 2025.

Chen, J., Ma, K., Huang, H., Shen, J., Fang, H., Zang, X., Ban, C., He, Z., Sun, H., and Kang, Y. Bovila: Bootstrapping video-language alignment via llmbased self-questioning and answering. arXiv preprint arXiv:2410.02768, 2024d.

Chen, M., Huang, H., and Li, Q. Towards robust uncertainty-aware incomplete multi-view classification. arXiv preprint arXiv:2409.06270, 2024e.

Cheng, S., Fang, K., Yu, Y., Zhou, S., Li, B., Tian, Y., Li, T., Han, L., and Liu, Y. Videgothink: Assessing egocentric video understanding capabilities for embodied ai. arXiv preprint arXiv:2410.11623, 2024a.

Cheng, Z., Leng, S., Zhang, H., Xin, Y., Li, X., Chen, G., Zhu, Y., Zhang, W., Luo, Z., Zhao, D., et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024b.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023.

Du, J., Liu, Y., Guo, H., Wang, J., Huang, H., Ni, Y., and Li, Z. Dependeval: Benchmarking llms for repository dependency understanding. arXiv preprint arXiv:2503.06689, 2025.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., and Kiela, D. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.

Fei, H., Wu, S., Ji, W., Zhang, H., Zhang, M., Lee, M.L., and Hsu, W. Video-of-thought: Step-by-step video reasoning from perception to cognition. In Forty-first International Conference on Machine Learning, 2024a.

Fei, H., Wu, S., Zhang, H., Chua, T.-S., and Yan, S. Vitron: A unified pixel-level vision llm for understanding, generating, segmenting, editing. 2024b.

Fei, H., Wu, S., Zhang, M., Zhang, M., Chua, T.-S., and Yan, S. Enhancing video-language representations with structural spatio-temporal alignment. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024c.

Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.

Fu, C., Lin, H., Wang, X., Zhang, Y.-F., Shen, Y., Liu, X., Li, Y., Long, Z., Gao, H., Li, K., et al. Vita-1.5: Towards gpt4o level real-time vision and speech interaction. arXiv preprint arXiv:2501.01957, 2025a.

Fu, J., Huangfu, S., Fei, H., Shen, X., Hooi, B., Qiu, X., and Ng, S.-K. Chip: Cross-modal hierarchical direct preference optimization for multimodal llms. arXiv preprint arXiv:2501.16629, 2025b.

Gunjal, A., Yin, J., and Bas, E. Detecting and preventing hallucinations in large vision language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 18135–18143, 2024.

Hu, G., Xin, Y., Lyu, W., Huang, H., Sun, C., Zhu, Z., Gui, L., Cai, R., Cambria, E., and Seifi, H. Recent trends of multimodal affective computing: A survey from nlp perspective. arXiv preprint arXiv:2409.07388, 2024.

Huang, H., Liu, Z., Letchmunan, S., Deveci, M., Lin, M., and Wang, W. Evidential deep partial multi-view classification with discount fusion. arXiv preprint arXiv:2408.13123, 2024a.

Huang, H., Qiao, X., Chen, Z., Chen, H., Li, B., Sun, Z., Chen, M., and Li, X. Crest: Cross-modal resonance through evidential deep learning for enhanced zero-shot learning. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 5181–5190, 2024b.

Huang, H., Qin, C., Liu, Z., Ma, K., Chen, J., Fang, H., Ban, C., Sun, H., and He, Z. Trusted unified featureneighborhood dynamics for multi-view classification. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 17413–17421, 2025a.

Huang, Y., Chen, H., Xu, Z., Jia, Z., Sun, H., and Shao, D. Sefar: Semi-supervised fine-grained action recognition with temporal perturbation and learning stabilization. arXiv preprint arXiv:2501.01245, 2025b.

Jang, Y., Song, Y., Yu, Y., Kim, Y., and Kim, G. Tgifqa: Toward spatio-temporal reasoning in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2758–2766, 2017.

Jiang, X., Ge, Y., Ge, Y., Shi, D., Yuan, C., and Shan, Y. Supervised fine-tuning in turn improves visual foundation models. arXiv preprint arXiv:2401.10222, 2024.

Jin, P., Takanobu, R., Zhang, W., Cao, X., and Yuan, L. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13700–13710, 2024.

Lan, W., Chen, W., Chen, Q., Pan, S., Zhou, H., and Pan, Y. A survey of hallucination in large visual language models. arXiv preprint arXiv:2410.15359, 2024.

Leng, S., Xing, Y., Cheng, Z., Zhou, Y., Zhang, H., Li, X., Zhao, D., Lu, S., Miao, C., and Bing, L. The curse of multi-modalities: Evaluating hallucinations of large multimodal models across language, visual, and audio. arXiv preprint arXiv:2410.12787, 2024.

Li, C., Im, E. W., and Fazli, P. Vidhalluc: Evaluating temporal hallucinations in multimodal large language models for video understanding. arXiv preprint arXiv:2412.03735, 2024a.

Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W., Ma, Z., and Li, C. Llava-next-interleave: Tackling multiimage, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024b.

- Li, J., Lu, W., Fei, H., Luo, M., Dai, M., Xia, M., Jin, Y., Gan, Z., Qi, D., Fu, C., et al. A survey on benchmarks of multimodal large language models. arXiv preprint arXiv:2408.08632, 2024c.
- Li, K., He, Y., Wang, Y., Li, Y., Wang, W., Luo, P., Wang, Y., Wang, L., and Qiao, Y. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.

Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., Wang, Z., Xu, J., Chen, G., Luo, P., et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22195–22206, 2024d.

Li, Y., Wang, C., and Jia, J. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pp. 323–340. Springer, 2025.

Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., and Yuan, L. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36, 2024a.

Liu, H., Xue, W., Chen, Y., Chen, D., Zhao, X., Wang, K., Hou, L., Li, R., and Peng, W. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253, 2024b.

Liu, R., Wu, H., Ziqiang, Z., Wei, C., He, Y., Pi, R., and Chen, Q. Videodpo: Omni-preference alignment for video diffusion generation. arXiv preprint arXiv:2412.14167, 2024c.

- Liu, Y., Li, S., Liu, Y., Wang, Y., Ren, S., Li, L., Chen, S., Sun, X., and Hou, L. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024d.

- Liu, Y., Liang, Z., Wang, Y., He, M., Li, J., and Zhao, B. Seeing clearly, answering incorrectly: A multimodal robustness benchmark for evaluating mllms on leading questions. arXiv preprint arXiv:2406.10638, 2024e.
- Liu, Z., Zang, Y., Dong, X., Zhang, P., Cao, Y., Duan, H., He, C., Xiong, Y., Lin, D., and Wang, J. Miadpo: Multi-image augmented direct preference optimization for large vision-language models. arXiv preprint arXiv:2410.17637, 2024f.

- Liu, Z., Sun, Z., Zang, Y., Dong, X., Cao, Y., Duan, H., Lin, D., and Wang, J. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.

Lu, J., Li, J., An, S., Zhao, M., He, Y., Yin, D., and Sun, X. Eliminating biased length reliance of direct preference optimization via down-sampled kl divergence. arXiv preprint arXiv:2406.10957, 2024.

Luo, M., Fei, H., Li, B., Wu, S., Liu, Q., Poria, S., Cambria, E., Lee, M.-L., and Hsu, W. Panosent: A panoptic sextuple extraction benchmark for multimodal conversational aspect-based sentiment analysis. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 7667–7676, 2024a.

Luo, M., Zhang, H., Wu, S., Li, B., Han, H., and Fei, H. Nus-emo at semeval-2024 task 3: Instruction-tuning llm for multimodal emotion-cause analysis in conversations. arXiv preprint arXiv:2501.17261, 2024b.

Ma, K., Huang, H., Chen, J., Chen, H., Ji, P., Zang, X., Fang, H., Ban, C., Sun, H., Chen, M., et al. Beyond uncertainty: Evidential deep learning for robust video temporal grounding. arXiv preprint arXiv:2408.16272, 2024.

Maaz, M., Rasheed, H., Khan, S., and Khan, F. S. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023.

Maaz, M., Rasheed, H., Khan, S., and Khan, F. Videogpt+: Integrating image and video encoders for enhanced video understanding. arXiv preprint arXiv:2406.09418, 2024.

Mosig, J. E., Mehri, S., and Kober, T. Star: A schemaguided dialog dataset for transfer learning. arXiv preprint arXiv:2010.11853, 2020.

Park, R., Rafailov, R., Ermon, S., and Finn, C. Disentangling length from quality in direct preference optimization. arXiv preprint arXiv:2403.19159, 2024.

Peng, B., Li, C., He, P., Galley, M., and Gao, J. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277, 2023.

Pi, R., Han, T., Xiong, W., Zhang, J., Liu, R., Pan, R., and Zhang, T. Strengthening multimodal large language model with bootstrapped preference optimization. In European Conference on Computer Vision, pp. 382–398. Springer, 2025.

Qian, L., Li, J., Wu, Y., Ye, Y., Fei, H., Chua, T.-S., Zhuang, Y., and Tang, S. Momentor: Advancing video large language model with fine-grained temporal reasoning. arXiv preprint arXiv:2402.11435, 2024.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

Sarkar, P., Ebrahimi, S., Etemad, A., Beirami, A., Arık, S. O.,¨ and Pfister, T. Mitigating object hallucination via data augmented contrastive tuning. arXiv preprint arXiv:2405.18654, 2024.

Shangguan, Z., Li, C., Ding, Y., Zheng, Y., Zhao, Y., Fitzgerald, T., and Cohan, A. Tomato: Assessing visual temporal reasoning capabilities in multimodal foundation models. arXiv preprint arXiv:2410.23266, 2024.

Tan, H., Ji, Y., Hao, X., Lin, M., Wang, P., Wang, Z., and Zhang, S. Reason-rft: Reinforcement fine-tuning for visual reasoning. arXiv preprint arXiv:2503.20752, 2025.

Tan, Z., Yang, X., Qin, L., Yang, M., Zhang, C., and Li, H. Evalalign: Supervised fine-tuning multimodal llms with human-aligned data for evaluating text-to-image models. arXiv preprint arXiv:2406.16562, 2024.

Tom, G., Mathew, M., Garcia-Bordils, S., Karatzas, D., and Jawahar, C. Reading between the lanes: Text videoqa on the road. In International Conference on Document Analysis and Recognition, pp. 137–154. Springer, 2023.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- Wang, X., Wu, J., Chen, J., Li, L., Wang, Y.-F., and Wang, W. Y. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4581–4591, 2019.
- Wang, Y., Wang, Y., Zhao, D., Xie, C., and Zheng, Z. Videohallucer: Evaluating intrinsic and extrinsic hallucinations in large video-language models. arXiv preprint arXiv:2406.16338, 2024.

Wu, S., Fei, H., Qu, L., Ji, W., and Chua, T.-S. NExTGPT: Any-to-any multimodal LLM. In Proceedings of the International Conference on Machine Learning, pp. 53366–53397, 2024a.

Wu, X., Bai, Y., Zheng, H., Chen, H. H., Liu, Y., Wang, Z., Ma, X., Shu, W.-J., Wu, X., Yang, H., et al. Lightgen: Efficient image generation through knowledge distillation and direct preference optimization. arXiv preprint arXiv:2503.08619, 2025.

Wu, Z., Chen, X., Pan, Z., Liu, X., Liu, W., Dai, D., Gao, H., Ma, Y., Wu, C., Wang, B., et al. Deepseek-vl2: Mixtureof-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024b.

Xiao, J., Shang, X., Yao, A., and Chua, T.-S. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9777–9786, 2021.

Xie, Y., Li, G., Xu, X., and Kan, M.-Y. V-dpo: Mitigating hallucination in large vision language models via visionguided direct preference optimization. arXiv preprint arXiv:2411.02712, 2024.

Xu, D., Zhao, Z., Xiao, J., Wu, F., Zhang, H., He, X., and Zhuang, Y. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international conference on Multimedia, pp. 1645–1653, 2017.

Xu, J., Mei, T., Yao, T., and Rui, Y. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5288–5296, 2016.

Xu, L., Zhao, Y., Zhou, D., Lin, Z., Ng, S. K., and Feng, J. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024.

Yan, W., Zhang, Y., Abbeel, P., and Srinivas, A. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021.

Yan, Y., Wen, H., Zhong, S., Chen, W., Chen, H., Wen, Q., Zimmermann, R., and Liang, Y. Urbanclip: Learning text-enhanced urban region profiling with contrastive language-image pretraining from the web. In Proceedings of the ACM Web Conference 2024, pp. 4006–4017, 2024.

Yang, K., Liu, Z., Xie, Q., Huang, J., Min, E., and Ananiadou, S. Selective preference optimization via token-level reward function estimation. arXiv preprint arXiv:2408.13518, 2024.

Yi, K., Gan, C., Li, Y., Kohli, P., Wu, J., Torralba, A., and Tenenbaum, J. B. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019.

Yin, S., Fu, C., Zhao, S., Xu, T., Wang, H., Sui, D., Shen, Y., Li, K., Sun, X., and Chen, E. Woodpecker: Hallucination correction for multimodal large language models. Science China Information Sciences, 67(12):220105, 2024.

Yu, Z., Xu, D., Yu, J., Yu, T., Zhao, Z., Zhuang, Y., and Tao, D. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 9127–9134, 2019.

Yuan, Y., Zhang, H., Li, W., Cheng, Z., Zhang, B., Li, L., Li, X., Zhao, D., Zhang, W., Zhuang, Y., et al. Videorefer suite: Advancing spatial-temporal object understanding with video llm. arXiv preprint arXiv:2501.00599, 2024.

Zeng, Y., Liu, G., Ma, W., Yang, N., Zhang, H., and Wang, J. Token-level direct preference optimization. arXiv preprint arXiv:2404.11999, 2024.

Zhang, H., Li, X., and Bing, L. Video-llama: An instructiontuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023a.

Zhang, J., Jiao, Y., Chen, S., Chen, J., and Jiang, Y.-G. Eventhallusion: Diagnosing event hallucinations in video llms. arXiv preprint arXiv:2409.16597, 2024a.

Zhang, R., Han, J., Liu, C., Gao, P., Zhou, A., Hu, X., Yan, S., Lu, P., Li, H., and Qiao, Y. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023b.

Zhang, R., Gui, L., Sun, Z., Feng, Y., Xu, K., Zhang, Y., Fu, D., Li, C., Hauptmann, A., Bisk, Y., et al. Direct preference optimization of video large multimodal models from language model reward. arXiv preprint arXiv:2404.01258, 2024b.

Zhao, M., Li, B., Wang, J., Li, W., Zhou, W., Zhang, L., Xuyang, S., Yu, Z., Yu, X., Li, G., et al. Towards video text visual question answering: Benchmark and baseline. Advances in Neural Information Processing Systems, 35: 35549–35562, 2022.

Zhao, Z., Wang, B., Ouyang, L., Dong, X., Wang, J., and He, C. Beyond hallucinations: Enhancing lvlms through hallucination-aware direct preference optimization. arXiv preprint arXiv:2311.16839, 2023.

Zhou, L., Xu, C., and Corso, J. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

Zhou, T., Chen, D., Jiao, Q., Ding, B., Li, Y., and Shen, Y. Humanvbench: Exploring human-centric video understanding capabilities of mllms with synthetic benchmark data. arXiv preprint arXiv:2412.17574, 2024a.

Zhou, Y., Cui, C., Rafailov, R., Finn, C., and Yao, H. Aligning modalities in vision large language models via preference fine-tuning. arXiv preprint arXiv:2402.11411, 2024b.

Zhu, B., Lin, B., Ning, M., Yan, Y., Cui, J., Wang, H., Pang, Y., Jiang, W., Zhang, J., Li, Z., et al. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. arXiv preprint arXiv:2310.01852, 2023.

### A. Limitation and Future Work

While VistaDPO excels at aligning video and language with fine-grained precision, its performance on long-duration videos with complex temporal dependencies leaves room for improvement. Such scenarios pose unique challenges for any alignment framework. Building on our strong spatial-temporal modeling foundation, future work could explore hierarchical architectures or memory-augmented mechanisms to further enhance the ability to capture long-term interactions, extending the reach of our method to even more complex video-language tasks.

### B. More Details of Data Annotation

Table 4. Summary of Hallucination Types, Sample Counts, and Data Sources. Hallucination Type Sample Count Data Source

Object 1,200 MSR-VTT, STAR, VATEX Number 500 ActivityNet-QA, MSR-VTT, NExT-QA, VATEX Location 500 MSR-VTT, NExT-QA, VATEX Color 500 ActivityNet-QA, CLEVRER, MSR-VTT, VATEX Static Relation 800 ActivityNet-QA, MSR-VTT, VATEX OCR 500 RoadTextVQA, ViteVQA Action 1,200 MSR-VTT, MSVD, STAR, VATEX Dynamic Attribute 300 TempCompass, Tomato Dynamic Relation 1,500 MSR-VTT, NExT-QA, STAR, VATEX, VCGBench-Diverse Sequence 200 Video-MME, YouCook2

Datasets Sources. We constructed a dataset by sampling from the validation sets of 14 existing datasets in Table 4, specifically MSR-VTT (Xu et al., 2016), STAR (Mosig et al., 2020), VATEX (Wang et al., 2019), ActivityNet-QA (Yu et al., 2019), NExT-QA (Xiao et al., 2021), CLEVRER (Yi et al., 2019), RoadTextVQA (Tom et al., 2023), ViteVQA (Zhao et al., 2022), MSVD (Chen & Dolan, 2011), TempCompass (Liu et al., 2024d), Tomato (Shangguan et al., 2024), VCGBench-Diverse (Maaz et al., 2024), Video-MME (Fu et al., 2024), and YouCook2 (Zhou et al., 2018), encompassing tasks such as binary QA, multiple-choice QA, and captioning-QA. To define hallucination within the context of video-based QA, we categorized it into two dimensions: Perception and Temporal, and generated corresponding chosen and rejected responses.

Specifically, the Perception dimension evaluates the model’s ability to recognize static information in videos. This includes object recognition, identifying static attributes (e.g., number, color, position), understanding spatial relationships between objects, and extracting other elements such as OCR. In contrast, the Temporal dimension assesses the model’s ability

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

(Irrelevant)

(Relevant but Wrong)

|[Figure 341]|
|---|

...

- (b) Augmented Chosen/Rejected QA-pairs

Question: Which object did the person sit on during the scene?

Chosen: The person sat on a chair during the scene.

Rejected2: During the scene, the person sat at the table instead.

Rejected1: In the scene, the person chose to sit in a shopping cart.

- (c) Manually Annotated Spatio-Temporal Information

||[Figure 349]|
|---|
<br><br>[Figure 343]<br><br>[Figure 344]<br><br>[Figure 345]<br><br>[Figure 346]<br><br>...<br><br>| |
|---|
| |[Figure 347]<br><br>| |
|---|
|[Figure 348]<br><br>| |
|---|
|
|---|---|---|---|
|“IS IN VIDEO?” shopping cart<br><br>(False)<br><br>table<br><br>(True)<br><br>|Appear (22.61s) {...}<br><br>Disappear (34.13s) {...}<br><br>Chair (26.69s) {(x,y,w,h)}| | |

Question: Which object did the person sit on during the scene?

|[Figure 349]|
|---|

Answer: The person sat on a chair during the scene.

(a) Original QA Pairs from Exsiting Prevalent Datasets

- Figure 9. Illustration of dataset pipeline for constructing augmented video-language QA pairs. (a) Original QA pairs are extracted from existing prevalent datasets, providing basic QA pairs. (b) These pairs are augmented by introducing chosen and rejected answers, where rejected answers include both irrelevant responses (e.g., ”shopping cart”) and relevant but incorrect ones (e.g., ”table”). (c) To enhance spatiotemporal understanding, manual annotations are added, specifying object appearances, spatial coordinates (e.g., bounding boxes), and temporal dynamics (e.g., appearance and disappearance timestamps). This pipeline ensures richer, more nuanced data for hierarchical preference optimization in video-language tasks.

to comprehend dynamic temporal information, such as recognizing actions, identifying subtle dynamic attributes (e.g., movement direction, speed, shape), understanding event relationships, and perceiving action sequences within the video. By leveraging the prompt structure illustrated in Figure 9, we expanded the original QA data into a dataset suitable for DPO training with chosen and rejected responses. During the construction of rejected response, we carefully considered whether the core semantics of the question were present in the video, generating both relevant and irrelevant rejected responses. This approach aims to enhance the model’s global understanding and robustness at the response level.

To explicitly strengthen the model’s spatiotemporal perception capabilities, we first identified all objects involved in the video. Subsequently, we manually annotated keyframes in which at least 30% of the object’s contours appeared or disappeared in the frame, as well as any keyframes directly relevant to answering the question. For each annotated keyframe, we labeled the bounding box coordinates (i.e., (x,y,w,h)) of the objects.

Quality Control. To ensure annotation quality, all annotators were PhD students from universities who underwent standardized training and utilized a unified annotation tool. Each video was annotated independently by two annotators, and cross-validation was performed. Samples with annotation discrepancies were discarded to maintain high data quality.

### C. More Discussions on Related Work

Table 5. Comparison among different DPO strategies.

Base Model DPO Textual Visual

Method LLM

(7B, if not specified) Text Image Video Granularity Dimension DPO (Rafailov et al., 2024) Text Pythia-2.8B ✓ ✗ ✗ Sentence IPO (Azar et al., 2024) Text Pythia-2.8B ✓ ✗ ✗ Sentence KTO (Ethayarajh et al., 2024) Text Llama-3-8B & Qwen-3B-Instruct ✓ ✗ ✗ Sentence R-DPO (Park et al., 2024) Text Pythia-2.8B ✓ ✗ ✗ Sentence SamPO (Lu et al., 2024) Text Tulu2-13B-SFT & Llama3-8B-Instruct ✓ ✗ ✗ Sentence SePO (Yang et al., 2024) Text LLaMA2-Chat & Pythia-SFT-6.9B ✓ ✗ ✗ Sentence & Token TDPO (Zeng et al., 2024) Text GPT-2-Large ✓ ✗ ✗ Sentence & Token HA-DPO (Zhao et al., 2023) Image LLaVA-v1.5 & MiniGPT-4 ✓ ✗ ✗ Sentence BPO (Pi et al., 2025) Image LLaVA-v1.5 ✓ ✗ ✗ Sentence FDPO (Gunjal et al., 2024) Image InstructBLIP-13B ✓ ✗ ✗ Sentence HALVA (Sarkar et al., 2024) Image LLaVA-v1.5 ✓ ✗ ✗ Sentence & Token POVID (Zhou et al., 2024b) Image LLaVA-v1.5 ✓ ✓ ✗ Sentence Spatial MIA-DPO (Liu et al., 2024f) Image LLaVA-v1.5 & InternLM-XC2.5 ✓ ✓ ✗ Sentence Spatial V-DPO (Xie et al., 2024) Image LLaVA-v1.5 ✓ ✓ ✗ Sentence Spatial Next-DPO (Li et al., 2024b) Video LLaVA-Next ✓ ✗ ✗ Sentence Hound-DPO (Zhang et al., 2024b) Video Video-LLaVA ✓ ✗ ✗ Sentence VistaDPO (Ours) Video Video-LLaVA & PLLaVA ✓ ✓ ✓ Sentence & Token Spatial & Temporal

To highlight our contributions, we detail in Table 5 how our proposed VistaDPO differs from previous DPO strategies. Two critical distinctions are summarized as follows:

- • Spatial-Temporal Video Preference Optimization: Previous DPO methods predominantly focused on language-level alignment. However, with the ongoing development of multi-task (Du et al., 2025; Wu et al., 2024a; Maaz et al., 2024) and multi-modal alignment (Huang et al., 2024b; 2025a; Wu et al., 2024a; Chen et al., 2024e; Huang et al., 2024a; Yan et al., 2024; Luo et al., 2024b; Li et al., 2024c; Luo et al., 2024a), reinforcement fine-tuning (Tan et al., 2025; Liu et al., 2025), self-correction techniques (Chen et al., 2024d), and visual generation (Wu et al., 2025; Chen et al., 2024b; Liu et al., 2024c), research efforts have gradually transitioned from pure language models towards vision-language models. While some works incorporated image-level visual alignment, these approaches remained limited to static images. Recent works like LLaVA-Next-DPO (Li et al., 2024b) and LLaVA-Hound-DPO (Zhang et al., 2024b) extended DPO strategies to video-language models. However, these methods only applied vanilla DPO strategies, optimizing alignment exclusively at the language level, with no explicit focus on visual modeling. In contrast, VistaDPO uniquely emphasizes optimizing spatial-temporal preferences in videos. By explicitly modeling both spatial and temporal preferences, VistaDPO bridges the gap between video content and textual understanding. This dual-layer spatial-temporal optimization enables our framework to address the complexities of video-language tasks comprehensively.
- • Hierarchical Finer Granularity: Most existing DPO approaches operate at a coarse granularity, typically limited to sentence-level alignment for text and holistic-level alignment for visuals. With the development of fine-grained understanding and generation (Huang et al., 2025b; Chen et al., 2024c), advanced methods explore token-level textual

alignment but still overlook hierarchical visual structures, which are crucial for video understanding. VistaDPO introduces a hierarchical granularity approach, incorporating both sentence- and token-level granularity for textual alignment and spatial- (object-) and temporal- (clip-) level granularity for visual alignment. By structuring alignment hierarchically across multiple layers—spanning from fine-grained token and object representations to coarse-grained sentence and video-level relationships—VistaDPO achieves a robust and precise preference optimization. This hierarchical approach empowers our framework to capture intricate cross-modal dependencies, ensuring superior performance in challenging scenarios such as adversarial testing and hallucination reduction.

### D. Extended Details of Methodology: Formulas and Prompts

This section details the core methodology used in VistaDPO, including the mathematical formulations and prompts employed during training. Key formulas for DPO are provided, along with the specific prompt templates used for generating and refining QA pairs. These details aim to provide a comprehensive understanding of the technical implementation.

#### D.1. Formulations of Token-Level Preference Optimization.

Token-Level Preference Optimization (TLPO) is a fine-grained optimization framework designed to align model outputs with human preferences by leveraging token-wise feedback. Unlike response-level optimization, TLPO avoids the cancellation of policies that may occur at the sentence level by focusing on sequential KL divergence at the token level.

Human Preference Modeling. We employ the Bradley-Terry model to represent the probability of human preferences for a winning response yw over a losing response yl, given the input x and auxiliary video context vwf . The preference probability is defined as:

PBT(yw ≻ yl|x,vwf ) = σ λ(x,vwf ,yw,yl) − δ(x,vwf ,yw,yl) ,

where σ(·) is the sigmoid function, λ(x,vwf ,yw,yl) represents the difference in rewards, and δ(x,vwf ,yw,yl) is the difference in sequential KL divergence between the preference pairs. These terms are defined as follows:

πθ(yl|x,vwf ) πref(yl|x,vwf )

πθ(yw|x,vwf ) πref(yw|x,vwf )

λ(x,vwf ,yw,yl) = β log

− β log

,

δ(x,vwf ,yw,yl) = βDSeqKL(x,vwf ,yw;πref||πθ) − βDSeqKL(x,vwf ,yl;πref||πθ).

Sequential KL Divergence. The sequential KL divergence DSeqKL is defined as the sum of token-level KL divergences across the sequence:

T

DKL(πref(yt|x,vwf ,y<t)||πθ(yt|x,vwf ,y<t)), where T is the length of the sequence y, and y<t denotes the tokens generated up to step t − 1. Loss Function for TLPO. Combining the Bradley-Terry model and the sequential KL divergence, the loss function for TLPO is expressed as:

DSeqKL(x,vwf ,y;πref||πθ) =

t=1

w,yw,yl) log σ λ(x,vwf ,yw,yl) − δ(x,vwf ,yw,yl) . Substituting λ(x,vwf ,yw,yl) and δ(x,vwf ,yw,yl), the loss function can be rewritten as:

LTLPO = −E(x,vf

πθ(yw|x,vwf ) πref(yw|x,vwf )

πθ(yl|x,vwf ) πref(yl|x,vwf )

LTLPO = − E(x,vf

− β log

w,yw,yl) log σ β log

(15)

− α DSeqKL(x,vwf ,yw;πref||πθ) − sg(DSeqKL(x,vwf ,yl;πref||πθ)) .

where α is a hyperparameter controlling the weight of the sequential KL divergence difference, and sg(·) represents the stop-gradient operator.

Final Formulation. The optimization term for TLPO, denoted as LDPOt

, focuses solely on the sequential KL divergence difference:

= sg βDSeqKL(x,vwf ,yw;πref||πθ) − βDSeqKL(x,vwf ,yl;πref||πθ).

LDPOt

This term ensures that the learned policy πθ aligns closely with the winning sequence yw while diverging from the losing sequence yl, effectively capturing human preferences at the token level.

#### D.2. Prompt templates for Generating QA pairs.

To adapt the existing dataset for fine-grained DPO training, we employed a template-based approach, as illustrated in Figure 10, and processed it using GPT-4. Specifically, we demonstrate the details of the prompt design using a multiplechoice dataset as an example.

System Prompt You are an expert in generating hallucinated sentences multiple-choice questions.

###### User Prompt

[Definition]

{Hallucination} in multimodal models refers to generating responses that are inconsistent with the actual visual content. A specific subtype of hallucination is {[HALLUCINATION_TYPE]}, where the model generates responses based on incorrect or imagined information.

[Task] Based on the input {video QA}, take the original correct answer and modify it into plausible but wrong

sentences (rejected responses) that fit the context but do not match the video. Then generate:

A {chosen response} (correct answer, rephrased naturally). Two {rejected responses}:

- -One modified from the original correct answer to be relevant but incorrect (full sentence).
- -One completely unrelated (full sentence).

[Example]

Input: (Original Video QA): {Prompt}: What type of car is featured in the video?

{Correct Answer}: A Porsche 911 GT3.

Output:

{Prompt}: What type of car is featured in the video? {Chosen}: The car shown is a Porsche 911 GT3 with carbon fiber detailing.

- {Rejected-1}: The video features a Ferrari 488 Spider, which looks similar to a Porsche.
- {Rejected-2}: The video shows a train passing through a rural area..

- Figure 10. A prompt template designed for generating hallucinated responses in multimodal models is presented. The template transforms original video QA pairs into a ”chosen response” (a rephrased correct answer) and two ”rejected responses” (one contextually relevant but incorrect, and one entirely unrelated). This framework supports preference optimization by providing plausible yet inaccurate alternatives for training and evaluation. An example illustrates the process, highlighting the generation of both coherent and unrelated hallucinated responses.

### E. More Comparison on MVBench

To more comprehensively evaluate VistaDPO, we conduct tests on MVBench (Li et al., 2024d), which contains 4,000 QA pairs across 11 video datasets covering a wide range of scenes, ranging from first-person to third-person and from indoor to outdoor environments. These tasks are categorized into 20 fine-grained temporal understanding tasks. The results in Table 6 shown an overall improvement of 2.7% and 3.3% compared to base model PLLaVA and Video-LLaVA, respectively. Notably, VistaDPO excels in Object Existence (8.5% and 7.5%), Object Interaction (5.0% and 6.5%), Moving Direction (2.5% and 7.0%), Action Localization (9.5% and 6.0%), and Fine-grained Pose (6.5% and 6.0%), demonstrating the effectiveness of our spatial-temporal and fine-grained modeling approach.

### F. Exhibition Board

Qualitative Demonstration. We show some unselected video QA cases in Figure 11, which are sourced from VideoHallucer (Wang et al., 2024) and EventHallusion (Zhang et al., 2024a).

VistaDPO-7K Sample Demonstration. We show examples of constructed VistaDPO-7K from temporal samples in

Table 6. Comparisons on MVBench. Bold values indicate the best performance achieved on the corresponding base model, while underlined values represent the second-best performance. The results of VideoChat, VideoChatGPT, Video-LLaMA, and VideoChat2 are included as references, but they are not directly related to the contributions of this paper.

Models Avg. AS AP AA FA UA OE OI OS MD AL ST AC MC MA SC FP CO EN ER CI

VideoChat (Li et al., 2023) 35.5 33.5 26.5 56.0 33.5 40.5 53.0 40.5 30.0 25.5 27.0 48.5 35.0 20.5 42.5 46.0 26.5 41.0 23.5 23.5 36.0 VideoChatGPT (Maaz et al., 2023) 32.7 23.5 26.0 62.0 22.5 26.5 54.0 28.0 40.0 23.0 20.0 31.0 30.5 25.5 39.5 48.5 29.0 33.0 29.5 26.0 35.5 Video-LLaMA (Zhang et al., 2023a) 34.1 27.5 25.5 51.0 29.0 39.0 48.0 40.5 38.0 22.5 22.5 43.0 34.0 22.5 32.5 45.5 32.5 40.0 30.0 21.0 37.0 VideoChat2 (Li et al., 2024d) 51.1 66.0 47.5 83.5 49.5 60.0 58.0 71.5 42.5 23.0 23.0 88.5 39.0 42.0 58.5 44.0 49.0 36.5 35.0 40.5 65.5

PLLaVA (Xu et al., 2024) 46.6 58.0 49.0 55.5 41.0 61.0 56.0 61.0 36.0 23.5 26.0 82.0 39.5 42.0 52.0 45.0 42.0 53.5 30.5 48.0 31.0 + Hound-DPO (Zhang et al., 2024b) 45.3 54.0 46.0 57.0 37.5 59.5 54.5 62.0 31.5 23.5 26.5 83.5 38.0 41.5 50.0 41.0 39.5 50.5 32.0 46.0 32.5 + VistaDPO (Ours) 49.3 59.5 51.0 60.0 41.5 59.0 64.5 66.0 35.0 27.0 35.5 82.5 40.0 45.5 51.5 48.0 48.5 54.0 31.0 50.0 35.0

Video-LLaVA (Lin et al., 2023) 43.0 46.0 42.5 56.5 39.0 53.5 53.0 48.0 41.0 29.0 31.5 82.5 45.0 26.0 53.0 41.5 33.5 41.5 27.5 38.5 31.5 + Hound-DPO (Zhang et al., 2024b) 43.3 44.5 40.0 59.0 39.0 52.5 53.5 49.5 36.5 32.0 33.5 79.0 43.0 28.0 55.5 42.0 30.0 43.0 31.0 39.0 35.0 + VistaDPO (Ours) 46.3 47.5 45.0 58.5 42.0 51.5 60.5 54.5 39.5 36.0 37.5 82.5 49.0 28.5 51.0 49.0 39.5 44.0 29.0 42.0 38.5

Note: Action: Action Sequence (AS), Action Prediction (AP), Action Antonym (AA), Fine-grained Action (FA), Unexpected Action (UA); Object: Object Existence (OE), Object Interaction (OI), Object Shuffle (OS); Position: Moving Direction (MD), Action Localization (AL); Scene: Scene Transition (ST); Count: Action Count (AC), Moving Count (MC); Attribute: Moving Attribute (MA), State Change (SC); Pose: Fine-grained Pose (FP); Character: Character Order (CO); Cognition: Egocentric Navigation (EN), Episodic Reasoning (ER), Counterfactual Inference (CI).

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

Does the person in the video touch his hair?

Does the car in the video move forward?

[Figure 360]

[Figure 361]

Yes, the car in the video moves forward.

Yes, the person in the video touches his hair while talking to the camera.

+Hound-DPO Yes, the person in the video touches his hair in some of frames.

+Hound-DPO Yes, the car does move forward in the video.

+VistaDPO No, the car in the video remains stationary, but the video shows

+VistaDPO No, the person in the video does not touch his hair, but focuses on

kids with school bags moving forward.

what looks like a drawing board object in front of him.

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

Is the person with a yellow helmet in the video sitting on a bicycle and riding it?

Did any accident or anything unexpected happen in the video?

[Figure 372]

No, the athlete is performing gymnastics with a balance beam without accident.

[Figure 373]

Yes, the person in the yellow helmet is sitting on the bike and riding it.

+Hound-DPO Yes, the person with a yellow helmet sits on a bicycle and rides it.

+Hound-DPO No, no accidents or any unexpected happend in the video.

+VistaDPO No, the person in the yellow helmet is not riding but repairing the

+VistaDPO Yes, the athlete fell in an accident while doing gymnastics on the

bicycle.

balance beam.

Figure 11. Cases of VistaDPO in video understanding.

Figure 12 and perception samples in Figure 13.

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

Question: What does the baby do after picking up a red toy at the start? Chosen: After picking up the red toy, the baby walks towards a woman dressed in jeans.

"is_in_video": true

- Reject1: The baby smiles before turning its head to look directly into the camera lens.

- Reject2: The baby stands up and begins to rotate a small ball in its hand. "is_in_video": false

"is_in_video": true

Spatial-Temporal Grounding Information： {Obejct-(t, x, y, w, h)}

red toy: (10.58, 145.2908, 309.6, 104.4653, 94.8)，(15.48, 314.5966, 136.8, 105.6660, 76.8)，(16.18, 255.7598, 265.2, 76.8480, 74.4)，(19.02, 280.9756, 120.0, 70.8443, 91.2)，(19.69, 283.3771, 91.2, 52.8330, 84.0)，(21.59, 347.0169, 129.6, 66.0413, 64.8)，(24.26, 315.7974, 158.4, 82.8518, 76.8)，(44.64, 154.8968, 156.0, 48.0300, 54.0)，(46.61, 208.9306, 268.8, 103.2645, 111.6),

baby: (12.88, 181.3133, 164.4, 297.7861, 234.0)，(16.35, 255.7598, 62.4, 220.9381, 375.6)，(19.85, 195.7223, 63.6, 189.7186, 294.0)，(21.76, 217.3358, 75.6, 133.2833, 216.0)，(25.99, 224.5403, 109.2, 132.0826, 182.4)，(50.78, 51.6323, 134.4, 139.2871, 192.0)

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

Question: What does the lady do after opening the bottle？ Chosen: After opening the bottle, the lady takes a sip directly from it.

"is_in_video": true

- Reject1: She carefully places the bottle back on the table without taking a single drink.

- Reject2: Instead of drinking it, she pours the contents into a glass. "is_in_video": false

"is_in_video": true

Spatial-Temporal Grounding Information： {Obejct-(t, x, y, w, h)}

bottle: (12.34, 420.2627, 244.8, 105.6660, 87.6)，(14.56, 419.0619, 152.4, 75.6473, 88.8), (15.78, 318.1989, 271.2, 68.4428, 110.4)。

Figure 12. Temporal data samples of VistaDPO-7K.

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

Question: What did the person pour into the container during the video? Chosen: The person poured juice into the container during the video.

"is_in_video": true

- Reject1: The person poured the contents of the bag into the container during the video.

- Reject2: The person used an umbrella to cover the container during the video. "is_in_video": false

"is_in_video": true

Spatial-Temporal Grounding Information： {Obejct-(t, x, y, w, h)} person: (3.04, 3.0, 5.0, 315.0, 355.0)，(13.11, 4.0, 123.0, 283.0, 237.0)，(53.65, 329.0, 22.0, 151.0, 339.0) juice: (0.57, 304.0, 65.0, 47.0, 74.0)，(13.11, 40.0, 171.0, 126.0, 78.0)，(20.05, 296.0, 292.0, 70.0, 67.0)。

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

Question: What is on the left of the river? Chosen: On the left side of the river, there is a tall tree providing shade over the bank.

"is_in_video": true

- Reject1: To the left of the river, you can see a striking fountain that adds a splash of color to the scene.
- Reject2: On the opposite side of the river, there's a simple wooden bench for people to rest. "is_in_video": false

"is_in_video": flase

Spatial-Temporal Grounding Information： {Obejct-(t, x, y, w, h)}

tree: (0.08, 16.0, 12.7811, 1539.2, 408.9941)，(170.68, 6.4, 12.7811, 1913.6, 731.7160)，(171.12, 9.6, 22.3669, 1932.8, 702.9586)

river: (0.12, 6.4, 415.3846, 1900.8, 664.6154)，(113.68, 35.2, 619.8817, 1849.6, 447.3373)，(170.0, 32.0, 610.2959, 1891.2, 456.9231)

Figure 13. Perception data samples of VistaDPO-7K.

