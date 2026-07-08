## Grounded 3D-LLM with Referent Tokens

Yilun Chen∗, Member, IEEE, Shuai Yang∗, Student Member, IEEE, Haifeng Huang∗, Student Member, IEEE, Tai Wang, Runsen Xu, Student Member, IEEE, Ruiyuan Lyu, Student Member, IEEE, Dahua Lin, Jiangmiao Pang†, Member, IEEE

### arXiv:2405.10370v2[cs.CV]18Nov2024

Abstract—Prior studies on 3D scene understanding have primarily developed specialized models for specific tasks or required task-specific fine-tuning. In this study, we propose Grounded 3D-LLM, which explores the potential of 3D large multi-modal models (3D LMMs) to consolidate various 3D vision tasks within a unified generative framework. The model uses scene referent tokens as special noun phrases to reference 3D scenes, enabling it to handle sequences that interleave 3D and textual data. Per-task instruction-following templates are employed to ensure natural and diversity in translating 3D vision tasks into language formats. To facilitate the use of referent tokens in subsequent language modeling, we provide a large-scale, automatically curated grounded scene-text dataset with over 1 million phrase-to-region correspondences and introduce Contrastive Language-Scene Pre-training (CLASP) to perform phrase-level scene-text alignment using this data. Our comprehensive evaluation covers open-ended tasks like dense captioning and 3D question answering, alongside close-ended tasks such as object detection and language grounding. Experiments across multiple 3D benchmarks reveal the leading performance and the broad applicability of Grounded 3D-LLM. Code and datasets are available at the project page.

Index Terms—Large Multi-modal Model, 3D Scene Understanding

✦

1 INTRODUCTION

# T

HE pursuit of a unified model for 3D scene understanding, capable of performing various vision tasks,

is a longstanding and significant challenge. Existing 3D scene-level models are typically customized for specific downstream tasks, including 3D object detection [1], [2], [3], [4], language grounding [5], [6], [7], [8], [9], question answering [10], [11], [12], [13], and dense captioning [8], [13], [14], [15], [16], [17], [18]. Recent 3D large multi-modal models (LMM), such as Chat-3D [19] and LL3DA [20], facilitate visual interactions based on detected objects for text reasoning, yet their capabilities remain confined to textlevel tasks. 3D-LLM [21], which extends 2D VLMs [22] to 3D, grapples with challenges in 3D spatial reasoning due to the intrinsic knowledge derived from perspective views. This results in suboptimal performance in crucial localization tasks like object localization and language grounding, thereby limiting its practical applicability. The capacity of large multi-modal models (LMMs) to pinpoint objects or regions in complex environments is vital, especially for applications such as VR/AR, robotics, interactive embodied agents, object navigation, and manipulation.

However, the exploration of a unified model for 3D perception and reasoning remains scarce, leaving it as an open problem. In our study, by leveraging the generative modeling capability of large language models (LLM), we introduce Grounded 3D-LLM for integration of multiple 3D vision tasks in linguistic formats (Fig. 1).

- • Y. Chen, S. Yang, H Huang, T Wang, R. Lyu, and J. Pang are with Shanghai AI Laboratory, Shanghai, China.
- • R. Xu and D. Lin are the Department of Information Engineering, The Chinese University of Hong Kong, Hong Kong, China. ∗ Equal Contribution. † Corresponding Author.

Its model design is inspired by the natural text, ”Two brown chairs are near the white nightstand,” which contains two noun phrases: ”two brown chairs” (region) and ”the white nightstand” (singular object). We refer to their corresponding 3D objects in the scene as ”phraseable objects,” indicating that each can be described in a text phrase and that together they are interrelated within the contextual description. Without loss of generality, we aim to represent these phraseable objects using a special token, termed the ”referent token” <ref>. Each referent token supports decoding to identify the corresponding objects (singular or plural) in a 3D scene. In the above example, <ref> can refer to items like “the white nightstand” or “two brown chairs”, substituting the corresponding text tokens in the language. The use of referent tokens as manageable noun phrases in linguistic descriptions offers two advantages: (i) a referent token can serve as either an alternative to or a companion for text phrases without altering the contextual description, as depicted in Fig. 1. and (ii) it enables the localization of nouns or text phrases within the language. Thus, this design positions referents as a unified interface, facilitating 3D grounding capabilities across various 3D vision tasks.

To accomplish this, the language modeling of the referent involves two high-level steps: (i) A more granular, phrase-level contrastive pre-training of the vision-language encoder in a large-scale scene-text dataset rich in phraselevel correspondence, and (ii) referent-aware instructionfollowing fine-tuning, which associates referent tokens with their respective object embeddings.

Specifically, unlike CLIP, which aligns text with the whole image, we first perform contrastive alignment between text primitives – noun phrases in the contextual description – with their corresponding visual embeddings from 3D scene encoder. This scene-level pre-training process is referred to as Contrastive LAnguage-Scene Pre-training

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

###### Single object Grounding

##### Grounded 3D-LLM

###### Object Detection

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Your task involves locating something. Here's the detail: this is a reddish brown leather armchair;

Catalog any shelfs found within this indoor space.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Quick check done. Yes, there is a <p> shelf </p> <ref> here.

[Figure 17]

[Figure 18]

[Figure 19]

I've pinpointed the <p> object </p> <ref> you're looking for.

[Figure 20]

[Figure 21]

Embodied Dialog

[Figure 22]

[Figure 23]

Multi-object Grounding

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Can you please fetch the printer from the nightstand <ref> and place it on <ref>?

[Figure 28]

[Figure 29]

[Figure 30]

Let‘s locate these specific items, the chairs near a bed.

[Figure 31]

[Figure 32]

printer nightstand

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

###### Certainly. I will move the <p> the printer </p> <ref> to <p> the table </p> <ref> right away.

<p> Objects </p> <ref> found and grounded.

[Figure 39]

[Figure 40]

bed

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Dense Captioning

Also, what is the design purpose of the cylindrical lamp with a white lampshade on <ref> ? It looks intriguing.

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Begin by outlining the visual characteristics of the object <ref>, followed by an account of its position relative to the objects around it.

[Figure 54]

[Figure 55]

[Figure 56]

<p> The cylindrical lamp </p> <ref> is designed to provide focused lighting on <p> the desk </p> <ref> for tasks like reading or working at the computer. . Its cylindrical shape allows it to distribute light evenly across the work area.

[Figure 57]

[Figure 58]

chair

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

This is a brown chair. it is against the brown table.

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

armchair table

[Figure 70]

[Figure 71]

lamp

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Question Answering

[Figure 77]

[Figure 78]

Embodied Planning

chair

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

What is on a small table? Please answer with a single word

I desire to enhance room lighting. Can you assist me with this?

or phrase.

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Lamp.

Got it. Let's begin the planning process now.

- step 1. Turn off <p> lamp </p> <ref>.
- step 2. Dust <p> lamp </p> <ref> shade.
- step 3. Replace bulb in <p> lamp </p> <ref>.
- step 4. Turn on <p> lamp </p> <ref> to test light. Planning stage complete. What's our next objective?

[Figure 92]

[Figure 93]

[Figure 94]

Where is the pillow on the bed? Please answer with a single word or phrase.

[Figure 95]

[Figure 96]

shelf

[Figure 97]

[Figure 98]

[Figure 99]

On top of bed.

- Fig. 1: Unifying scene-level 3D vision tasks via task-specific prompts in Grounded 3D-LLM framework. The interleaved referent token <ref> in language modeling allows the identification of external scene referents through masked decoding in raw point clouds. Words within <p> and </p> denote the accompanying text phrases. Matching colors are shown in the figure to connect phrases with their respective object masks.

(CLASP). However, existing paired scene-text data at the phrase level (referred to as grounded scene-text data in this paper) remain scarce. To address this, we develop an automated pipeline to synthesize grounded scene-text data using a densely annotated 3D detection dataset. The resulting dataset, called Grounded Scene Caption (G-SceneCap), comprises 107K scene captions with over 1+ million phrase-level correspondence. Compared to existing scene-text datasets, it provides richer phrase-level correspondence, averaging one object every nine tokens, as shown in Table 2. Besides, existing human-annotated 3D vision-language datasets [5], [23] are also into grounded language forms.

After the phrase-level scene-text alignment, we conduct multi-task instruction fine-tuning for Grounded 3D-LLM to associate the referents with their respective phraseable objects. We convert various existing grounded scene-text datasets into instruction-following format, where referent tokens are interleaved as alternatives or companions to the text phrases (examples shown in Fig. 1). During training, these special referent tokens are supervised to decode into specified object embeddings, identifying their external scene referents. These task-specific instruction-following conversions encompass a range of existing 3D vision-language tasks – from single to multi-object grounding, instance segmentation (object detection), to 3D QA and dense captioning. We compare this model’s prompt support and task versatility with previous large multi-modal models in Table 1. We believe this represents a significant step toward unifying various 3D vision tasks within generative language modeling.

To summarize, our contributions are as follows:

• We propose Grounded 3D-LLM, which establishes correspondence between 3D scenes and natural language using

referent tokens. This method facilitates scene referencing and effectively models various 3D vision-language problems within a unified language modeling framework.

- • We first provide an automated, curated grounded scene caption dataset with over 1 million phrase-level correspondences. Experiments show that CLASP, using this data in both supervised and zero-shot text settings, demonstrates broad generalization in phrase-level grounding.
- • Without requiring specialized models or task-specific finetuning, the single model Grounded 3D-LLM achieves top-tier performance in most downstream tasks among generative models, particularly in grounding problems.

###### 2 RELATED WORKS

2D large multi-modal models. Vision-language models [22], [27] are proposed to bridge visual and textual models via image-level contrastive learning. Their direct application enables open-world understanding [28], [29]. GLIP and its subsequent works [30], [31], [32] have innovatively formulated object detection as grounding problems, leveraging detection or pseudo-labels to align semantics. With stronger language models, various powerful large multimodal models are capable of handling tasks including image captioning [33], [34], VQA [35], [36], [37], [38], and object detection/segmentation [32], [39]. Besides, incorporating multi-modal tokens into language modeling has been validated, including visual understanding [34], [40], [41], [42], robot control [43], and interleaved text-image content generation [44], [45]. On the other hand, Chen et al. [46] demonstrates that existing 2D VLMs lack 3D spatial reasoning, primarily due to insufficient 3D spatial data.

TABLE 1: Comparison of 3D multi-modal models. We refer to instance segmentation as Inst.Seg., object box detection as Obj.Det., single-object grounding as Grd., point-level grounding as Point-Grd., multi-object grounding as Multi-Obj Grd., question answering as QA., and dense captioning as Cap..

Prompts Tasks Text Vision Inst.Seg. Obj.Det. Grd. Point-Grd. Multi-Obj Grd. QA. Cap.

Method LLM

PointGroup [1] ✗ – – ✓ ✓ ✗ ✗ ✗ ✗ ✗ Mask3D [4] ✗ – – ✓ ✓ ✗ ✗ ✗ ✗ ✗ Multi3DRef [23] ✗ – – ✗ ✗ ✓ ✗ ✓ ✗ ✗ BUTD-DETR [24] ✗ – – ✗ ✓ ✓ ✗ ✓ ✗ ✗ 3D-VisTA [25] ✗ – – ✗ ✗ ✓ ✗ ✗ ✓ ✓

Chat-3D [19] ✓ ✓ ✗ ✗ ✗ ✗ ✗ ✗ ✓ ✓ Chat-3D v2 [26] ✓ ✓ ✗ ✗ ✗ ✓ ✗ ✓ ✓ ✓ 3D-LLM [21] ✓ ✓ ✗ ✗ ✗ ✓ ✗ ✓ ✓ ✓ LL3DA [20] ✓ ✓ ✓ ✗ ✗ ✗ ✗ ✗ ✓ ✓

Grounded 3D-LLM ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

3D scene understanding. In 3D scene understanding, prior works [47], [48], [49] have advanced large-scale unsupervised learning for foundational models. Language is increasingly used to encapsulate user intentions due to its broad applicability, ranging from closed-set object detection [1], [2], [4], [24], [50] to open-set 3D detection [51], [52], [53], [54], [55], [56]. Typically, these 3D detectors lay the groundwork for subsequent scene-language tasks. Such tasks encompass 3D Visual Grounding [5], [6], [7], [9], [23], [57], [58], [59], where models pinpoint objects within a 3D scene based on language queries. They also include 3D Question Answering [10], [60], [61], which demands robust spatial perception and reasoning. Another task, 3D Dense Captioning [14], [62], [63], involves localizing and describing all scene objects, necessitating a comprehensive understanding of object locations and attributes. While initial efforts were often task-specific and lacked generalizability, some methodologies [8], [17] have integrated 3D visual grounding and dense captioning tasks, leveraging their complementary nature. Recent endeavors such as 3D-VisTA [25] and 3D-VLP [13] aim to establish a universal 3D visual-language framework by pre-aligning 3D scenes with language. In contrast, most approaches develop specialized models or require per-task training. They also rely on off-the-shelf 3D detectors.

3D large multi-modal models. 3D object-level LMMs [64], [65], [66], [67] have effectively bridged the gap between 3D visuals and texts by leveraging extensive 3D object datasets [68]. This has advanced the development of 3D object-level LMM [21], [67], [69], [70], [71], [72], [73]. However, these models struggle with interpreting complex spatial interrelationships in 3D scenes. For scene-level LLMs, models such as Chat-3D [19], Scene-LLM [74], LL3DA [20], and LEO [75] enable scene-level dialogue, achieving notable effectiveness in tasks such as question answering and captioning. Chat-3D v2 [26] enables language grounding by using object identifiers for effective referencing. 3DLLM [21] improves scene understanding by incorporating positional embeddings and location tokens. Our approach utilizes special referent tokens for flexible scene modeling and supports diverse tasks including object detection and grounding. Additionally, our model functions as a generalist model without requiring task-specific fine-tuning.

###### 3 METHOD

This section consists of three parts. Sec. 3.1 examines vision-language alignment at three levels. Sec. 3.2 details the Grounded 3D-LLM, from vision-language pre-training to multi-task instruction fine-tuning with referent tokens. Finally, Sec.3.3 describes the construction of the large-scale grounded scene-text dataset for pre-training, while Sec.3.4 illustrates the conversion of existing scene-text data to instruction-following formats for LLM fine-tuning.

###### 3.1 Preliminary: Vision-Language Correspondence

The significance of vision-language alignment in 2D foundation models, exemplified by the Contrastive LanguageImage Pre-training (CLIP) [27], is well-established. We explore three levels of vision-language alignment:

- • Sentence-level Correspondence: CLIP models [22], [27] introduced contrastive learning between image and sentencelevel text embeddings. However, the limited size (∼100Kscale) of paired scene-text datasets [5], [23], [57] hinders broader application, and the alignment lacks object-level grounding.
- • Word-level Correspondence: Initially introduced by GLIPs [30], it reinterprets object detection labels as phrase grounding tasks. However, this reliance on existing detection labels does not adequately capture appearance and spatial relationships.
- • Phrase-level Correspondence: As referred to as phrase-toregion correspondence in prior work [76], this constraint requires linking text phrases to specific objects (singular or plural) within a contextual description including category, attributes and etc.. However, such 3D datasets remain scarce, especially in 3D scenes.

Our motivation. To enable the natural referencing of scene objects in language modeling, we aim to model phraselevel correspondences, allowing phraseable objects to be represented as referent tokens. In other words, referent tokens can substitute for or accompany noun phrases without altering the language structure. Inspired by Llava [35], we first adopt contrastive pre-training for the vision-language encoder at the phrase level using grounded scene-text data. During instruction-following fine-tuning, the LLM

###### Step 1: Contrastive Language-Scene Pre-training

###### Step 2: Multi-task Instruction fine-tuning

Assistant: <p> A black chair </p> <ref> with four legs...

###### Contrastive Loss

Contrastive Loss

Contrastive Loss

[Figure 100]

𝑆𝑆𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡

de-tokenize

𝑅𝑅𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡

[Figure 101]

[Figure 102]

𝑆𝑆𝑟𝑟𝑡𝑡𝑟𝑟

[Figure 103]

[Figure 104]

𝑃𝑃𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡 𝑄𝑄𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡 𝑄𝑄𝑚𝑚𝑚𝑚𝑚𝑚𝑚𝑚

𝑆𝑆𝑚𝑚𝑚𝑚𝑚𝑚𝑚𝑚

...

... ...

[Figure 105]

[Figure 106]

Language head

𝑄𝑄𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡

Object Embeddings

Text embeddings

𝑃𝑃𝑚𝑚𝑚𝑚𝑚𝑚𝑚𝑚

LoRA

𝑘𝑘,𝑣𝑣

Large Language Model

Cross-Modal Interactor (CMI)

Point embeddings

𝑜𝑜𝑜𝑜𝑜𝑜 𝑞𝑞

𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡 𝑞𝑞

[Figure 107]

3D Encoder

...

...

...

[Figure 108]

tokenize

Raw points

tokenize

Chair; Table; Door;... A white microwave sits adjacent to the large kitchen counter, with a brown cabinet installed above it...

[Figure 109]

[Figure 110]

random queries

[Figure 111]

CMI

[Figure 112]

[Figure 113]

User: Describe the object<ref>?

[Figure 114]

[Figure 115]

Grounded point clouds

Grounded text

[Figure 116]

[Figure 117]

Object embeddings

[Figure 118]

Raw Points

Prompt

Grounded scene-text data

- Fig. 2: Main pipeline for Grounded 3D-LLM. The training process for Grounded 3D-LLM is divided into two key steps. Firstly, CLASP utilizes an extensive grounded scene-text dataset (at the phrase level) to pre-train a 3D point cloud encoder and a cross-modal interactor. The subsequent step involves multi-task instruction fine-tuning, which interlaces referent tokens within the instructions and responses, thereby facilitating flexible 3D scene understanding tasks.

###### object queries text queries

object queries text queries

𝑘𝑘,𝑣𝑣

associates the newly proposed referent token with these phraseable objects. For example, in the phrase ”two brown chairs near the table,” the average word embeddings of ”two brown chairs” align with the 3D visual embeddings of the corresponding chairs. By inputting these text-aligned object embeddings, the LLM can reference these phraseable objects using the special referent. This can be formatted as <p>two brown chairs</p> <ref> near the table or simply as <p>object</p> <ref>. The token pairs <p> and </p> indicate phrase boundaries [39], [42].

K ×

K ×

Text-to-object Cross-Attention

Cross-Attention

𝑘𝑘,𝑣𝑣 𝑞𝑞

𝑞𝑞

Self-Attention Self-Attention

Self-Attention Self-Attention

𝑘𝑘,𝑣𝑣

K ×

K ×

Text-to-object Cross-Attention

Cross-Attention

𝒌𝒌-level Point

𝒌𝒌-level Point

𝑞𝑞

𝑘𝑘,𝑣𝑣 𝑞𝑞

𝑘𝑘,𝑣𝑣

Object-to-text Cross-Attention

Cross-Attention

Self-Attention Self-Attention

Self-Attention Self-Attention

Emd

𝑞𝑞 𝑘𝑘,𝑣𝑣

Emd

𝑞𝑞

𝒌𝒌-level Point Emd

𝒌𝒌-level Point Emd

Self-Attention Self-Attention

Self-Attention Self-Attention

𝑘𝑘,𝑣𝑣

Cross-Attention

Object-to-text Cross-Attention

𝑞𝑞

𝑘𝑘, 𝑣𝑣

𝑞𝑞 𝑘𝑘,𝑣𝑣

𝑘𝑘, 𝑣𝑣

Cross-Attention

Cross-Attention

Self-Attention Self-Attention

Self-Attention Self-Attention

𝑞𝑞

𝑞𝑞

Self-Attention

Self-Attention

𝑘𝑘, 𝑣𝑣

𝑘𝑘, 𝑣𝑣

Cross-Attention

Cross-Attention

𝑞𝑞

𝑞𝑞

Self-Attention

Self-Attention

object queries text queries

object queries text queries

(a) Text-to-object CMI (b) Bi-directional CMI

###### 3.2 Grounded 3D-LLM

object queries text queries

object queries text queries

Based on the aforementioned motivation, the main pipeline of Grounded 3D-LLM is illustrated in Fig. 2. To reference phraseable objects in language modeling, a vision-language model is first pre-trained on large-scale grounded scenetext data to align text phrases with their corresponding 3D objects. Subsequently, a large language model (LLM) is fine-tuned using multi-modal instruction-following data, where referent tokens serve as interleaved “soft” text tokens representing the phraseable objects. Per-task instructionfollowing templates are employed to address the diverse range of 3D vision tasks within the unified language modeling framework.

(a) One-Way CMI

(b) Bi-directional CMI

Fig. 3: Designs of Cross-Modal Interactor (CMI). Subfigure (b) shows the bi-directional cross-attention design inspired by GroundingDINO [32], which is the default choice.

queries T. For each pyramid level, a cross-modal interactor (CMI) is used to extract object embeddings conditioned on both point cloud embeddings and their respective text descriptions. Specifically, CMI interacts with N learnable object queries Q as proxies to connect an M-word text (extracted from BERT) with point clouds. Initially, these object queries interact with k-level point-wise embeddings via cross-attention. Then, as shown in Fig. 3, to incorporate the contextual text description, they interact with text embeddings through plain one-way cross-attention or bidirectional cross-attention [32].

Step 1: Contrastive language-scene pre-training (CLASP). In the same spirit as CLIP [27], CLASP is pre-trained to align 3D scene embeddings with textual descriptions using grounded scene-text data. Each paired grounded scene-text annotation contains phrase-to-region correspondences, linking noun phrases inside the text and their respective 3D object IDs (singular or plural). CLASP co-trains a 3D encoder, a text encoder, a set of object queries, and a cross-modal interactor (CMI), as shown on the left side of Fig. 2. Specifically, a sparse convolutional network [77] serves as the 3D multilevel feature encoder, extracting point-wise embeddings P as a K-level feature pyramid, while BERT [78] extracts text

As shown in Fig. 2 (left), CMI yields both object and text embeddings, followed by a small head that transforms them into the textual embedding space: Ptext for positive text embeddings and Qtext for object embeddings, as well as the mask embedding space: Pmask and Qmask for mask

[Figure 119]

[Figure 120]

Region objects

Step 2

Step 3

Select region objects

[Figure 121]

Sample anchors

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Grounded Caption with Simple Relations

[Figure 126]

Complete Grounded Caption Extra Spatial Relations

[Figure 127]

postprocessing

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

GT labels correction

Object Captions

VLM +

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Step 1

multi-view image for each object

- Fig. 4: Grounded Scene Caption dataset generation pipeline. This pipeline leverages privileged knowledge in 3D, including dense object labels, multi-view images, and global spatial relationships, along with the capabilities of ChatGPT tools and spatial relationship programming tools, to establish accurate scene-level captions with grounding information.

embeddings of both, respectively:

we compute the similarity matrix between the referent and all object query embeddings Qtext:

[Pmask,Qmask],[Qtext,Ptext] = CMI(P,T,Q). (1) As depicted in Fig. 2, we compute the alignment scores Smask for per-point mask classification and Stext for the textual embedding similarities between noun phrases and object queries:

Sref = RtextQ⊤text. (3) The contrastive loss for Sref is supervised with the referent correspondence and applied with the same temperature η as CLASP. During inference, the output referents are decoded with a decoder (two MLP layers) to extract hidden embeddings Rtext, retrieve the respective object queries, and obtain their associated masks through prior CLASP.

Smask = PmaskQ⊤mask, Stext = PtextQ⊤text, (2) where we ignore the temperature η = 0.1. The per-point mask classification for Smask is computed to classify each point against the object queries using bipartite matching [79], [80]. Another contrastive loss for noun phrases and object queries, Stext, is supervised by the phrase-to-region correspondence. During inference, the mapping from text phrases to point-level masks can be computed using both similarity matrices, e.g., by employing the top-scored indices as the mapping.

Besides, we ensure linguistic naturalness by formatting referents with accompanying text phrases, such as “<p> three nearby chairs </p> <ref>.”. If no category is provided in scene-text annotations, we use the generic term “object” instead. For tasks requiring grounding or localization, we append “(with grounding)” to instructions to enable referencing in the language output.

Discussion: Referent token as localization token. Prior work, 3D-LLM [21], uses discrete location tokens, limiting localization accuracy and hindering grounded visual chat tasks. Chat-3D v2 [26] pairs object identifiers with features as text inputs but incurs high token costs and relies on off-the-shelf detectors and per-object feature extractors [67]. In contrast, our model jointly extracts per-object features through large-scale, fine-grained scene-level pre-training of CLASP. The design of the Referent token allows decoding into multiple instances, reducing token costs and increasing localization efficiency. Additionally, it is the first to support point-level mask decoding.

Step 2: Instruction fine-tuning with referent tokens. During instruction-following fine-tuning, Grounded 3D-LLM models input and output in a unified multimodal language format, fine-tuning a LLM based on the pre-trained CLASP. Specifically, the LLM inputs scene encoding (all object query embeddings) from the frozen CLASP with a plain detection prompt [30]. It aims to learn to use the referent token <ref> to reference the scene, given a large amount of grounded instruction-following data.

Given a single grounded instruction-following data point, the text is interleaved with referent and each referent has its mapping to some phrasesable objects (referent correspondence). We first convert the instruction and response into the USER-ASSISTANT format before feeding them into the LLM. An example is shown below:

3.3 Generating Grounded Scene-Text Data for Pretraining

In this subsection, we describe two approaches to generate grounded scene-text data for pre-training: (i) automatic generation of grounded scene captions, and (ii) transformation of existing language data into grounded formats.

[All object embeddings] USER: Describe the object <ref> in the scene. ASSISTANT: <p> A black chair </p> <ref> with four legs.

Grounded Scene Caption dataset generation. The dataset generation for the Grounded Scene Caption dataset (GSceneCap) leverages ChatGPT and 2D vision-language models, utilizing dense object annotations from existing 3D scan datasets [81], as illustrated in Fig. 4:

To enable closed-loop dialogue with the LLM using scene referents, we model referents as their corresponding objects in both language input and output. Specifically, for language input, the referents (visible to the model) are directly replaced with their respective object query embeddings, followed by a linear projection to match the LLM channel size. For language output, the referents serve as placeholders, with their hidden embeddings Rtext aligned with the corresponding object query embeddings. During fine-tuning,

Step 1: Bootstrapping object captions with GT label correction. Using 3D real-scan datasets, we annotate each object with the vision-language model CogVLM [38], using the images of the largest visible areas. Inconsistent annotations are rectified using raw instance labels.

[Figure 140]

[Figure 141]

[Figure 142]

[bed 1]: A full-sized bed with a dark-colored frame.

- [window 38]: A vertical sliding window with multiple uniformly spaced horizontal slats and a dark color.

[Figure 143]

[nightstand 36]: A wooden nightstand with a dark finish, rectangular shape.

[Figure 144]

- [window 39]: A sliding glass window with partially closed horizontal white blinds set within a darkcolored frame.

[Figure 145]

[chair 32]: A modern chair with a curved backrest and brown leather upholstery.

[Figure 146]

[chair 31]: A modern chair with a dark wooden frame and a textured brownish backrest.

[Figure 147]

[shelf 34]: A tall, dark brown or black shelf, made of wood or similar material with a smooth surface.

[Figure 148]

[door 40]: A sliding glass door with horizontal white blinds.

[Figure 149]

###### Grounded Caption with Simple Relations:

[A full-sized bed 1] sits prominently in the local scene. [A modern chair 31] with a dark wooden frame is separated from [another chair 32] with brown leather upholstery. [A tall shelf 34] stands against the wall, while [a wooden nightstand 36] is positioned adjacent to [the bed 1]. Flanking the space are [two windows 38, 39], with one featuring horizontal slats and the other white blinds. Nearby, there is [a sliding glass door 40] with matching white blinds.

[Figure 150]

Extra Spatial Relations:

(1) Facing the front of [the bed 1], [a tall shelf 34] is on the right side of it. (2) Facing the front of [the bed 1], [a wooden nightstand 36] is on the left side of it.

(3) [The window 38] is closer to [the nightstand 36]. (4) [A full-sized bed 1] is between [the window 39] and [the shelf 34].

[Figure 151]

Complete Grounded Caption:

[A full-sized bed 1] sits prominently in the local scene, flanked by [a tall shelf 34] on the right and [a wooden nightstand 36] on the left when facing the front of [the bed 1]. Close to the [nightstand 36] is [a window 38] with horizontal slats, while another [window 39] with white blinds is positioned on the side of [the bed 1]. Separated by design, [a modern chair 31] with a dark wooden frame and [another chair 32] with brown leather upholstery occupy the space. Additionally, [a sliding glass door 40] with matching white blinds is situated nearby.

[Figure 152]

Phrase-to-Region Correspondence: 𝐶𝐶𝐶𝐶𝐶𝐶𝐶0:16 → 𝑜𝑜𝑜𝑜𝑗𝑗1 , 𝐶𝐶𝐶𝐶𝐶𝐶𝐶64:77 → 𝑜𝑜𝑜𝑜𝑗𝑗34 , 𝐶𝐶𝐶𝐶𝐶𝐶𝐶94:114 → 𝑜𝑜𝑜𝑜𝑗𝑗36 ,𝐶𝐶𝐶𝐶𝐶𝐶𝐶152:159 → 𝑜𝑜𝑜𝑜𝑗𝑗1 ,……, 𝐶𝐶𝐶𝐶𝐶𝐶𝐶438:458 → 𝑜𝑜𝑜𝑜𝑗𝑗40

- Fig. 5: Grounded Scene Caption visualization. In each grounded text, a phrase-to-region correspondence is formatted as “[object phrase object IDs]”, where “object phrase” refers to a noun phrase and “object IDs” denotes the corresponding object IDs in the groud-truth annotation. The bottom line demonstrates explicit phrase-to-region correspondence example between the noun phrases (character positions) and the corresponding objects.

- TABLE 2: Grounded scene-text dataset comparison. “Sent.”, “Obj.”, “Multi.Obj.”, and “Corr.” refer to “Sentence”, “Object”, “Multi-objects”, and “Correspondence”. Tokens are calculated using BERT [78]. For sentences with single-object correspondence, each text is paired with one instance.

Correspondence #Corr. Type Corr/Token

Dataset #3D Scan #Tokens #Text #Tokens/Text

ScanRefer [5] 0.7K 1.17M 52K 22.6 52K Sent.-Obj. 4.4% Sr3D [57] 1.5K 1.10M 84K 13.1 84K Sent.-Obj. 7.7% Sr3D+ [57] 1.5K 1.48M 115K 12.9 115K Sent.-Obj. 7.7% Nr3D [57] 0.7K 0.617M 42K 14.7 42K Sent.-Obj. 6.8% Multi3DRefer [23] 0.7K 1.20M 62K 19.4 87K Sent.-Mult.Obj. 7.2% ScanScribe [25] 3.0K 18.5M 278K 66.5 278K Sent.-Obj. 1.5%

Grounded Scene Caption 1.5K 9.19M 107K 85.9 1.02M Phrase-Multi.Obj. 11.1%

- Step 2: Condensing objects in local scenes into a caption. For each enumerated anchor object, we form an initial object set by randomly selecting a group of nearby objects. Their captions and coordinates (x,y,z) are input into GPT-4 for captioning, which requires referencing object phrases with their object IDs in the format “[object phrase object ID]” in the caption.

- Step 3: Adding rule-based relations. To enrich scene captions, we integrate program-generated spatial relationships from Sr3D [57]. By selecting an anchor object from the set in step 2, we apply the spatial relation rules (e.g., between, supporting, nearest, back) to include related objects. GPT-
- 4 then combines these relationships into the prior caption from step 2.

A specific coarse-to-fine scene caption generation example with phrase-to-region correspondence, including step-bystep intermediate outputs, is illustrated in Fig. 10. The data generation prompts are detailed in the supplementary files. Grounded conversion of existing scene-text data. In this

work, we employ ChatGPT [82] to extract positive phrases for ScanRefer and Multi3DRef datasets for training and evaluation. We refer to the grounded datasets as G-ScanRefer and G-Multi3DRefer. The transformation of all detection labels into a fixed text prompt is called G-Detection as GLIP [30].

Comparsion with scene-text datasets. Llava [35] uses GPT-4 to generate multi-modal data based on 2D boxes. Similarly, 3D-LLM [21] relies only on 3D boxes to create instructionfollowing data. Other 3D works [53], [54] add cross-view information to capture view-level differences and interactions, improving open-vocabulary detection. ScanScribe [25] creates scene annotations from human scene graphs. Sceneverse [83] proposes generating scene graphs automatically, then rephrasing them into captions. Multi3DRef [23] generates multi-object correspondence using ScanRefer annotations.

Beyond these efforts, we focus on capturing noun relationships within natural language rather than generating

Task: single-object grounding Template:

Single object grounding Object detection 3D QA Object captioning

Instruction-following Convertion

Task Select

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 7

[Figure 153]

[Figure 154]

[Figure 155]

Instruction-following Templates

Grounded Scene-Text Data Task: Multi-object grounding Text: The chairs nearby the bed. Source: Multi3DRef Phrase-to-Region Correspondence: 𝐶𝐶𝐶𝐶𝐶𝐶𝐶5:10 → 𝒐𝒐𝒐𝒐𝒋𝒋𝟑𝟑,𝒐𝒐𝒐𝒐𝒋𝒋𝟓𝟓

[Figure 156]

Grounded Instruction-following Data Instruction: Can you locate these objects for me: the chairs nearby the bed? Answer: I’ve successfully identified the <p> chairs</p> <ref>. Referent Correspondence: {<ref>→ 𝒐𝒐𝒐𝒐𝒋𝒋𝟑𝟑,𝒐𝒐𝒐𝒐𝒋𝒋𝟓𝟓 }

[Figure 157]

Q: Can you locate these objects for me: {grounding text}? A: I’ve successfully identified the {category}

- Fig. 6: Converting grounded scene-text data to instruction-following format. This figure shows an instruction-following conversion example for the multi-object grounding task. Note that the referent correspondence is converted from the phraseto-region correspondence of grounded scene-text annotation.

...

[Figure 158]

Grounded Scene-Text Data Data Type: grounded scene caption Lang: The [chairs 3,4] nearby the bed. Source: Multi3DRef

object-level or region-level captions. Additionally, globalview spatial relationships are considered through GPT (Step 2) and rule-based programming (Step 3). As a result, our generated grounded scene-text data achieves the highest object density, averaging one correspondence per nine tokens, leading to over 1.02 million correspondences, as shown in Table 2. This provides rich phrase-level correspondence in natural text and allows for seamless conversion to referent correspondence in the following subsection.

grounding), ScanNet-200 Detection [85] (object detection/instance segmentation), Scan2Cap [14] (dense object captioning), and ScanQA [10] (3D QA) benchmarks. These benchmarks are built on the ScanNet dataset [81], a largescale indoor 3D scene dataset with 1,513 multi-modal scenes containing 3D point clouds, image sequences, and pointlevel instance segmentation annotations. The dataset is split into 1,201 scenes for training and 312 for validation, and all benchmarks follow these splits.

Embodied Dialogue

Metrics. We follow prior works and adopt the commonly used metrics for these benchmarks. For ScanRefer and Multi3DRef, we compute the Intersection over Union (IoU) at thresholds of 0.25 and 0.5. For ScanNet-200, we report 3D instance segmentation using average precision at 25%, 50% overlap, and mean average precision (mAP) with IoU thresholds in the [0.5:0.95:0.05] range across 200 classes. For Scan2Cap and ScanQA, we evaluate language generation with BLEU-4 and CIDEr, weighted by IoU above 0.25 or 0.5.

###### 3.4 Converting to Grounded Instruction-Following Format for LLM fine-tuning

In this subsection, we describe the transformation of grounded scene-text data from Sec. 3.3 to the instructionfollowing format.

Converting 3D vision-language tasks to instructionfollowing formats. For existing 3D vision-language tasks, we can transform them into instruction-following formats. An example of conversion for multi-object grounding tasks including referent correspondence generation is shown in Fig. 6. These convertible 3D vision-language tasks include single and multi-object grounding, object detection, dense captioning, 3D QA, etc.. For each task, we utilize approximately 10-20 structured task-specific instruction-following templates. Please refer to the supplementary file for details on the task-specific instruction-following templates.

To maintain consistency with prior evaluations across the ScanRefer, Multi3DRef, and Scan2Cap benchmarks, we extract 3D bounding boxes by taking the minimum and maximum values from the predicted masks, despite our model producing point-level predictions. For Multi3DRef, we apply a score threshold of 0.3 to filter the predicted objects.

Implementation Details. Unless otherwise specified, CLASP pre-training is conducted on Grounded Scene Caption, ScanNet-200 detection data, ScanRefer, and Multi3DRef data. 150 object queries via farthest point sampling (FPS) is initialized for CLASP and initialized with pretrained 3D detector Mask3D [4] to accelerate the training. We incorporate the same per-point mask classification loss Lmask with both sigmoid focal loss and Dice loss for segmentation. The overall loss function for CLASP is defined as:

Extension to embodied planning and dialogue. To illustrate the potential applications of referent tokens in daily life and enable multi-round conversations, we prompt GPT4 to generate embodied dialogue and planning based on the Grounded Scene Caption dataset, rather than using static templates. Unlike 3D-LLM [21], our approach produces not only plain dialogue text but also grounded instructionfollowing text with interleaved phrase-to-region correspondence (referent tokens). These dialogues adhere to several rules: they occur between a human and an embodied agent and involve tasks such as completing daily activities, discussing layouts, and conducting object-oriented QA based on the given caption. The embodied planning text must generate daily instructions and a step-by-step plan for interacting with objects within the scene caption. Refer to the supplementary files for the prompts used in specific dialogue data generation and the visualized results from our jointly trained model.

LCLASP = Lmask(Smask,Tmask) + λclsLcls(Stext,Ttext),

(4) where Tmask is computed using Hungarian matching [4] and Ttext are the ground-truth mappings computed according to ground-truth phrase-to-region correspondence. Lcls represents the sigmoid focal loss.

During instruction fine-tuning, Grounded 3D-LLM adopts Tiny-Vicuna-1B as the LLM by default, keeping it frozen except for the intermediate projection and LoRA [86] layers. Its vision-language model is initialized with a pretrained CLASP model. To minimize token costs, only 100 object queries are used. The fine-tuning leverages grounded instruction-following datasets, including ScanRefer, Multi3DRef, ScanNet-200 detection, Scan2Cap, ScanQA,

- 4 EXPERIMENTS

###### 4.1 Experimental Setup

Datasets. We evaluate our model on the ScanRefer [5] (single-object grounding), Multi3DRef [23] (multi-object

- TABLE 3: Evaluation of 3D scene-level LMMs as a generalist. Entries in gray denote models specialized for specific datasets. “Zero-shot” denotes models used directly without fine-tuning. “Specialist” and “Generalist” categorize models fine-tuned for particular tasks versus those trained jointly. Entries in grey denote the incorporation of ground-truth objects into the model inputs.

Method

Generative Model

ScanRefer Grd. Multi3DRef Grd. ScanQA Scan2Cap ScanNet-200 Det. Acc@0.25 Acc@0.5 F1@0.25 F1@0.5 B-4 C B-4@0.5 C@0.5 AP AP0.25 AP0.5

ScanRefer [5] ✗ 37.3 24.3 MVT [7] ✗ 40.8 33.3 3DVG-Trans [6] ✗ 45.9 34.5 ViL3DRel [9] ✗ 47.9 37.7 M3DRef-CLIP [23] ✗ 51.9 44.7 42.8 38.4 Scan2Cap [14] ✗ 22.4 35.2 ScanQA [10] ✗ 10.1 64.9 3D-VisTA [25] ✗ 50.6 45.8 13.1 72.9 34.0 66.9 Mask3D [5] ✗ 27.4 37.0 42.3

Zero-shot LMMs LLM-Grounder [84] ✓ 17.1 5.3 – – – – – – – – – Specialist LMMs

3D-LLM(Flamingo) [21] ✓ 21.2 – – – 7.2 59.2 – – – – – 3D-LLM(BLIP2-flant5) [21] ✓ 30.3 – – – 12.0 69.4 – – – – – Chat-3D [19] ✓ – – – – 6.4 53.2 – – – – – Chat-3D v2 [26] ✓ 35.9 30.4 – – 7.3 77.1 – – – LL3DA [20] ✓ – – – – 13.5 76.8 36.8 65.2 – – –

Generalist LMMs

LEO [75] ✓ – – – – 13.2 101.4 38.2 72.4 – – – LL3DA [20] ✓ – – – – 13.3 75.7 36.0 62.9 – – – Grounded 3D-LLM ✓ 48.6 44.0 44.7 40.8 13.2 75.9 35.0 70.2 12.1 16.8 18.9

- TABLE 4: Effects of diverse instruction-following templates on language tasks.

• Generative models include recent 3D LMMs [19], [20], [21], [26] without task heads, where specialist models require task-specific fine-tuning, and generalist 3D LMMs [20], [75] are trained in unified language format.

Diverse Templates

ScanQA Scan2Cap B-4 C B-4@0.25 C@0.25 B-4@0.5 C@0.5

Result analysis. As shown in Table 3, Grounded 3D-LLM, without task-specific fine-tuning, outperforms previous LLMbased models in most metrics, positioning it as a promising candidate for a unified framework in 3D scene understanding. Compared to prior 3D LMMs, it achieves the best grounding results, improving by over 10 points in both single and multiple object grounding. Additionally, it attains comparable outcomes in language-based tasks, even against LL3DA, which is limited to text output. Notably, it achieves leading performance with a CIDEr score of 75.9 and a BLEU4 score of 13.2 in ScanQA. Existing generalist 3D LMMs, such as LEO [75] and LL3DA [20], support language tasks but are limited in localization. Furthermore, leveraging the one-to-many correspondence ability of the referent token, we report that Grounded 3D-LLM performs 3D instance segmentation, achieving 12.1 mAP. Despite its inferior performance compared to prior expert models like Mask3D, it demonstrates the potential to unify pure 3D vision tasks, with further improvements remaining as future work.

✗ 11.2 70.8 31.5 65.8 29.2 62.1 ✓ 13.2 75.9 36.9 74.1 35.0 70.2

scene captioning, embodied dialogue and embodied planning. An object query is considered a positive match only if its mask IoU with the ground-truth object exceeds 0.3 otherwise the object will be ignored. The overall language loss is used as follows:

###### LLLM = Llang + Lref(Sref,Tref), (5)

where Llang is cross-entropy loss as LLAMA [87] and Lref computes the sigmoid focal loss between Sref and the ground-truth referent correspondence Tref from the grounded annotation.

###### 4.2 Main Results

###### 4.3 Ablation Studies

Baselines. Existing baseline methods for downstream tasks are categorized into discriminative and generative approaches in Table 3.

Diversity of task-specific instruction-following templates. Typical computer vision tasks such as language grounding, object detection, and dense captioning require instructionfollowing templates to better adapt to pre-trained weights of large language models. We assess the impact of diverse instruction-following templates on language tasks in Table 4. Results show that diverse instruction-following templates consistently yield significant improvements in tasks

• Discriminative models focus on single-task outputs with task-specific heads, covering single-object [6], [7], [9] and multi-object grounding [23]. 3D-VisTA [25] pre-trains a unified 3D backbone followed by task-specific finetuning, while Mask3D [4], a SOTA transformer-based 3D detector, is trained with instance segmentation heads.

- TABLE 5: Grounding ability of Grounded 3D-LLM. The term “all data” encompasses the G-ScanRefer, G-Multi3DRef, and G-SceneCap datasets used for pre-training.

# CLASP

Pre-training data

<ref> type

ScanRefer Grd. Multi3DRef Grd. Acc@0.25 Acc@0.5 F1@0.25 F1@0.5

- (a) ✗ – One-to-Many 6.5 5.8 6.4 5.3
- (b) ✓ G-ScanRefer One-to-Many 43.2 36.6 40.2 36.9
- (c) ✓ G-SceneCap One-to-Many 47.6 43.4 44.3 40.1
- (d) ✓ All data One-to-One 46.5 43.3 44.2 40.7
- (e) ✓ All data One-to-Many 47.9 44.1 45.2 40.6

- TABLE 6: Comparison of CLASP in phrase grounding tasks. “Point-supervision” involves using point-level instance segmentation supervision. “End-to-end models” are those that can be trained either fully end-to-end or using separate detectors.

End-to -end

Pointsupervision

ScanRefer Grd. Multi3DRef Grd. ScanNet200 Det.

Method

Acc@0.25 Acc@0.5 F1@0.25 F1@0.5 AP AP@0.25 AP@0.5 BUTD-DETR [24] ✗ ✗ 52.2 39.8 – – – – – ViL3DRel [9] ✗ ✓ 47.9 37.7 – – – – – 3D-VisTA [25] ✗ ✓ 50.6 45.8 – – – – – M3DRef-CLIP [23] ✗ ✗ 51.9 44.7 42.8 38.4 – – – Mask3D [4] ✓ ✓ – – – – 27.4 37.0 42.3 CLASP ✓ ✓ 53.2 46.7 51.5 47.3 27.2 36.4 42.1

TABLE 7: Effects of pre-training datasets for CLASP. † represents using ground-truth boxes.

Pre-training Method

Correspondence Level

ScanRefer Grd. Multi3DRef Grd. Acc@0.25 Acc@0.5 F1@0.25 F1@0.5

#

Training Data

- (a) M3DRef-CLIP† Sentence Multi3DRef 55.0 – –

- (b) 3D-VisTA Sentence

ScanScribe 41.1 36.4 – –

- (c) (b) w. ScanRefer 50.6 45.8 – –

- (d) CLASP Phrase

G-SceneCap 46.2 35.9 39.3 36.1

- (e) (d) w. G-ScanRefer 49.7 44.4 39.2 35.9
- (f) (e) w. G-Detection 51.1 44.8 40.2 37.0
- (g) (f) w. G-Multi3DRef 53.2 46.7 51.5 47.3
- (h) (g) w/o. G-SceneCap 51.1 45.4 48.5 43.1

like dense captioning and 3D QA. Notably, in templatebased captioning tasks (Scan2Cap), both CIDEr@0.25 and CIDEr@0.5 see improvements exceeding 8 points. Employing a single instruction-following template also leads to unstable results in repeated experiments if diverse templates are not used. Therefore, the diversity of instructionfollowing templates is crucial for sustaining language capabilities in visual tasks.

Ablation studies of the referent token for grounding. As shown in Table 5, comparing line (a) with others indicates that directly adapting existing closed-set detectors is ineffective for integrating referent tokens in Grounded 3D-LLM. Incorporating a grounded scene-text dataset for pre-training boosts performance to 43.2 Acc@0.25 and 40.2 F1@0.25. With only G-SceneCap for CLASP pre-training, Grounded 3D-LLM already achieves comparable results to all pretraining data.

Additionally, we analyze two types of referent tokens: one-to-one (d), where each object is matched with a unique token via Hungarian matching, and one-to-many (e), where a single token represents multiple objects. The latter gener-

ally yields better results.

Direct comparison of CLASP in phrase grounding tasks. CLASP performs phrase-level scene-text alignment for pretraining, allowing for direct comparison with previous grounding and detection methods. As shown in Table 6, CLASP excels in 3D grounding and detection benchmarks, particularly in ScanRefer grounding, Multi3DRef multiobject grounding, and ScanNet-200 detection, showcasing its enhanced phrase grounding capabilities compared to earlier models. Notably, in the Multi3DRef grounding task, CLASP outperforms specialized models by over 8 points, demonstrating its strong multi-object handling.

Dataset ablations for CLASP. Table 7 further ablates the effects of datasets for the scene-text alignment:

(i) Zero-shot text evaluation. Without training on ScanRefer or Multi3DRef language data, our CLASP model (d) proves effective compared to previous models (a) and (b). Notably, despite 3D-VisTA’s data generation adhering to sentencelevel grounding similar to ScanRefer, our model exceeds it by 4.9 points in Acc@0.25. Furthermore, in experiment (d), our direct evaluation on the Multi3DRef dataset demon-

###### TABLE 8: Comparison of cross-modal interactors (CMI).

- [8] D. Z. Chen, Q. Wu, M. Nießner, and A. X. Chang, “D3net: A speaker-listener architecture for semi-supervised dense captioning and visual grounding in rgb-d scans,” 2021.
- [9] S. Chen, P.-L. Guhur, M. Tapaswi, C. Schmid, and I. Laptev, “Language conditioned spatial relation reasoning for 3d object grounding,” NeurIPS, vol. 35, pp. 20522–20535, 2022.
- [10] D. Azuma, T. Miyanishi, S. Kurita, and M. Kawanabe, “Scanqa: 3d question answering for spatial scene understanding,” in CVPR, 2022, pp. 19129–19139.
- [11] A. Delitzas, M. Parelli, N. Hars, G. Vlassis, S. Anagnostidis, G. Bachmann, and T. Hofmann, “Multi-clip: Contrastive visionlanguage pre-training for question answering tasks in 3d scenes,” arXiv preprint arXiv:2306.02329, 2023.
- [12] M. Parelli, A. Delitzas, N. Hars, G. Vlassis, S. Anagnostidis, G. Bachmann, and T. Hofmann, “Clip-guided vision-language pretraining for question answering in 3d scenes,” in CVPR, 2023, pp. 5606–5611.
- [13] Z. Jin, M. Hayat, Y. Yang, Y. Guo, and Y. Lei, “Context-aware alignment and mutual masking for 3d-language pre-training,” in CVPR, 2023, pp. 10984–10994.
- [14] Z. Chen, A. Gholami, M. Nießner, and A. X. Chang, “Scan2cap: Context-aware dense captioning in rgb-d scans,” in CVPR, 2021, pp. 3193–3203.
- [15] Z. Chen, R. Hu, X. Chen, M. Nießner, and A. X. Chang, “Unit3d: A unified transformer for 3d dense captioning and visual grounding,” in ICCV, 2023, pp. 18109–18119.
- [16] H. Wang, C. Zhang, J. Yu, and W. Cai, “Spatiality-guided transformer for 3D dense captioning on point clouds,” in IJCAI, 2022.
- [17] D. Cai, L. Zhao, J. Zhang, L. Sheng, and D. Xu, “3djcg: A unified framework for joint dense captioning and visual grounding on 3d point clouds,” in CVPR, 2022, pp. 16464–16473.
- [18] S. Chen, H. Zhu, X. Chen, Y. Lei, G. Yu, and T. Chen, “End-toend 3d dense captioning with vote2cap-detr,” in CVPR, 2023, pp. 11124–11133.
- [19] Z. Wang, H. Huang, Y. Zhao, Z. Zhang, and Z. Zhao, “Chat3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes,” arXiv preprint arXiv:2308.08769, 2023.
- [20] S. Chen, X. Chen, C. Zhang, M. Li, G. Yu, H. Fei, H. Zhu, J. Fan, and T. Chen, “Ll3da: Visual interactive instruction tuning for omni-3d understanding, reasoning, and planning,” arXiv preprint arXiv:2311.18651, 2023.
- [21] Y. Hong, H. Zhen, P. Chen, S. Zheng, Y. Du, Z. Chen, and C. Gan, “3d-llm: Injecting the 3d world into large language models,” arXiv preprint arXiv:2307.12981, 2023.
- [22] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” arXiv preprint arXiv:2301.12597, 2023.
- [23] Y. Zhang, Z. Gong, and A. X. Chang, “Multi3drefer: Grounding text description to multiple 3d objects,” in ICCV, 2023, pp. 15225– 15236.
- [24] A. Jain, N. Gkanatsios, I. Mediratta, and K. Fragkiadaki, “Bottom up top down detection transformers for language grounding in images and point clouds,” in ECCV. Springer, 2022, pp. 417–433.
- [25] Z. Zhu, X. Ma, Y. Chen, Z. Deng, S. Huang, and Q. Li, “3dvista: Pre-trained transformer for 3d vision and text alignment,” in ICCV, 2023, pp. 2911–2921.
- [26] H. Huang, Z. Wang, R. Huang, L. Liu, X. Cheng, Y. Zhao, T. Jin, and Z. Zhao, “Chat-3d v2: Bridging 3d scene and large language models with object identifiers,” arXiv preprint arXiv:2312.08168, 2023.
- [27] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML. PMLR, 2021, pp. 8748–8763.
- [28] X. Gu, T.-Y. Lin, W. Kuo, and Y. Cui, “Zero-shot detection via vision and language knowledge distillation,” arXiv preprint arXiv:2104.13921, vol. 2, no. 3, p. 4, 2021.
- [29] A. Zareian, K. D. Rosa, D. H. Hu, and S.-F. Chang, “Openvocabulary object detection using captions,” in CVPR, 2021, pp. 14393–14402.
- [30] L. H. Li, P. Zhang, H. Zhang, J. Yang, C. Li, Y. Zhong, L. Wang, L. Yuan, L. Zhang, J.-N. Hwang et al., “Grounded language-image pre-training,” in CVPR, 2022, pp. 10965–10975.
- [31] L. Yao, J. Han, Y. Wen, X. Liang, D. Xu, W. Zhang, Z. Li, C. Xu, and H. Xu, “Detclip: Dictionary-enriched visual-concept paralleled pre-training for open-world detection,” NeurIPS, vol. 35, pp. 9125– 9138, 2022.

Crossinteractor types

ScanRefer Scan2Cap

Acc@0.25 Acc@0.5 B-4@0.5 C@0.5 One-way 47.2 44.0 34.9 70.4

Bi-directional. 47.9 44.1 35.5 70.6

strates robust multi-object grounding ability.

(ii) Multi-dataset evaluation. We further investigate the effects of joint training with grounded scene-text datasets, progressively integrating G-ScanRefer (e), G-Detection (f), and G-Multi3DRef (g). This integration boosts performance, achieving 53.2 in Acc@0.25 on ScanRefer and 51.5 in F1@0.25 on Multi3DRef. Notably, Comparison between (g) and (h) shows that the G-SceneCap dataset consistently raises performance by +2.1 points in ScanRefer Acc@0.25 and +3.0 points in Multi3DRef F1@0.25, highlighting its effectiveness. Comparison of cross-modal interactors. Table 8 ablates two key multi-modal interaction components—cross-modal interactors of Fig. 3: text-to-object one-way CMI and bidirectional CMI. The results indicate that incorporating bidirectional cross-attention modules in cross-modal interactors enhances grounding performance (with a 0.7 improvement in Acc@0.25 on ScanRefer and a 0.6 improvement in BLEU4@0.5 on Scan2Cap) and dense captioning metrics. This improvement suggests that deeper interactions between modalities provide greater benefits for CLASP.

###### 5 CONCLUSION

We explore the potential of large multi-modal 3D models to unify various downstream 3D vision tasks in unified language modeling. By interpreting scene referents as special language tokens, Grounded 3D-LLM connects scene objects or regions with language, offering a natural way to localize noun phrases in the 3D environment. We demonstrate its general effectiveness across various 3D vision tasks; however, it shows limitations in performance compared to previous expert models, highlighting the disparity between specialist and generalist approaches. Scaling scene-text data to bridge this gap is an objective for future work.

###### REFERENCES

- [1] L. Jiang, H. Zhao, S. Shi, S. Liu, C.-W. Fu, and J. Jia, “Pointgroup: Dual-set point grouping for 3d instance segmentation,” in CVPR, 2020, pp. 4867–4876.
- [2] Z. Liu, Z. Zhang, Y. Cao, H. Hu, and X. Tong, “Group-free 3d object detection via transformers,” in ICCV, 2021, pp. 2949–2958.
- [3] T. Vu, K. Kim, T. M. Luu, T. Nguyen, and C. D. Yoo, “Softgroup for 3d instance segmentation on point clouds,” in CVPR, 2022, pp. 2708–2717.
- [4] J. Schult, F. Engelmann, A. Hermans, O. Litany, S. Tang, and B. Leibe, “Mask3d: Mask transformer for 3d semantic instance segmentation,” in ICRA. IEEE, 2023, pp. 8216–8223.
- [5] D. Z. Chen, A. X. Chang, and M. Nießner, “Scanrefer: 3d object localization in rgb-d scans using natural language,” in ECCV. Springer, 2020, pp. 202–221.
- [6] L. Zhao, D. Cai, L. Sheng, and D. Xu, “3dvg-transformer: Relation modeling for visual grounding on point clouds,” in ICCV, 2021, pp. 2928–2937.
- [7] S. Huang, Y. Chen, J. Jia, and L. Wang, “Multi-view transformer for 3d visual grounding,” in CVPR, 2022, pp. 15524–15533.

- [32] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” arXiv preprint arXiv:2303.05499, 2023.
- [33] Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, and F. Wei, “Kosmos-2: Grounding multimodal large language models to the world,” arXiv preprint arXiv:2306.14824, 2023.
- [34] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” NeurIPS, vol. 35, pp. 23716–23736, 2022.
- [35] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” NeurIPS, vol. 36, 2024.
- [36] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” arXiv preprint arXiv:2304.10592, 2023.
- [37] W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. Fung, and S. Hoi, “Instructblip: Towards general-purpose vision-language models with instruction tuning,” 2023.
- [38] W. Wang, Q. Lv, W. Yu, W. Hong, J. Qi, Y. Wang, J. Ji, Z. Yang, L. Zhao, X. Song et al., “Cogvlm: Visual expert for pretrained language models,” arXiv preprint arXiv:2311.03079, 2023.
- [39] H. Rasheed, M. Maaz, S. Shaji, A. Shaker, S. Khan, H. Cholakkal, R. M. Anwer, E. Xing, M.-H. Yang, and F. S. Khan, “Glamm: Pixel grounding large multimodal model,” arXiv preprint arXiv:2311.03356, 2023.
- [40] W. Wang, Z. Chen, X. Chen, J. Wu, X. Zhu, G. Zeng, P. Luo, T. Lu, J. Zhou, Y. Qiao et al., “Visionllm: Large language model is also an open-ended decoder for vision-centric tasks,” NeurIPS, vol. 36, 2024.
- [41] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia, “Lisa: Reasoning segmentation via large language model,” arXiv preprint arXiv:2308.00692, 2023.
- [42] H. Zhang, H. Li, F. Li, T. Ren, X. Zou, S. Liu, S. Huang, J. Gao, L. Zhang, C. Li et al., “Llava-grounding: Grounded visual chat with large multimodal models,” arXiv preprint arXiv:2312.02949, 2023.
- [43] S. Reed, K. Zolna, E. Parisotto, S. G. Colmenarejo, A. Novikov, G. Barth-Maron, M. Gimenez, Y. Sulsky, J. Kay, J. T. Springenberg et al., “A generalist agent,” arXiv preprint arXiv:2205.06175, 2022.
- [44] Q. Sun, Q. Yu, Y. Cui, F. Zhang, X. Zhang, Y. Wang, H. Gao, J. Liu, T. Huang, and X. Wang, “Emu: Generative pretraining in multimodality,” in ICLR, 2023.
- [45] X. Dong, P. Zhang, Y. Zang, Y. Cao, B. Wang, L. Ouyang, X. Wei, S. Zhang, H. Duan, M. Cao, W. Zhang, Y. Li, H. Yan, Y. Gao, X. Zhang, W. Li, J. Li, K. Chen, C. He, X. Zhang, Y. Qiao, D. Lin, and J. Wang, “Internlm-xcomposer2: Mastering free-form textimage composition and comprehension in vision-language large model,” arXiv preprint arXiv:2401.16420, 2024.
- [46] B. Chen, Z. Xu, S. Kirmani, B. Ichter, D. Sadigh, L. Guibas, and F. Xia, “Spatialvlm: Endowing vision-language models with spatial reasoning capabilities,” in CVPR, 2024, pp. 14455–14465.
- [47] D. Huang, S. Peng, T. He, H. Yang, X. Zhou, and W. Ouyang, “Ponder: Point cloud pre-training via neural rendering,” in ICCV, 2023, pp. 16089–16098.
- [48] H. Zhu, H. Yang, X. Wu, D. Huang, S. Zhang, X. He, T. He, H. Zhao, C. Shen, Y. Qiao et al., “Ponderv2: Pave the way for 3d foundataion model with a universal pre-training paradigm,” arXiv preprint arXiv:2310.08586, 2023.
- [49] M. El Banani, A. Raj, K.-K. Maninis, A. Kar, Y. Li, M. Rubinstein, D. Sun, L. Guibas, J. Johnson, and V. Jampani, “Probing the 3d awareness of visual foundation models,” in CVPR, 2024, pp. 21795–21806.
- [50] T. Wang, X. Mao, C. Zhu, R. Xu, R. Lyu, P. Li, X. Chen, W. Zhang, K. Chen, T. Xue, X. Liu, C. Lu, D. Lin, and J. Pang, “Embodiedscan: A holistic multi-modal 3d perception suite towards embodied ai,” in CVPR, 2024.
- [51] R. Chen, Y. Liu, L. Kong, X. Zhu, Y. Ma, Y. Li, Y. Hou, Y. Qiao, and W. Wang, “Clip2scene: Towards label-efficient 3d scene understanding by clip,” in CVPR, 2023, pp. 7020–7030.
- [52] R. Ding, J. Yang, C. Xue, W. Zhang, S. Bai, and X. Qi, “Lowis3d: Language-driven open-world instance-level 3d scene understanding,” arXiv preprint arXiv:2308.00353, 2023.
- [53] ——, “Pla: Language-driven open-vocabulary 3d scene understanding,” in CVPR, 2023, pp. 7010–7019.

- [54] J. Yang, R. Ding, W. Deng, Z. Wang, and X. Qi, “Regionplc: Regional point-language contrastive learning for open-world 3d scene understanding,” in CVPR, 2024, pp. 19823–19832.
- [55] C. Zhu, W. Zhang, T. Wang, X. Liu, and K. Chen, “Object2scene: Putting objects in context for open-vocabulary 3d detection,” arXiv preprint arXiv:2309.09456, 2023.
- [56] S. Peng, K. Genova, C. M. Jiang, A. Tagliasacchi, M. Pollefeys, and T. Funkhouser, “Openscene: 3d scene understanding with open vocabularies,” in CVPR, 2023.
- [57] P. Achlioptas, A. Abdelreheem, F. Xia, M. Elhoseiny, and L. Guibas, “Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes,” in ECCV. Springer, 2020, pp. 422–440.
- [58] Z. Wang, H. Huang, Y. Zhao, L. Li, X. Cheng, Y. Zhu, A. Yin, and Z. Zhao, “3drp-net: 3d relative position-aware network for 3d visual grounding,” arXiv preprint arXiv:2307.13363, 2023.
- [59] Y. Man, L.-Y. Gui, and Y.-X. Wang, “Situational awareness matters in 3d vision language reasoning,” in CVPR, 2024.
- [60] M. Parelli, A. Delitzas, N. Hars, G. Vlassis, S. Anagnostidis, G. Bachmann, and T. Hofmann, “Clip-guided vision-language pretraining for question answering in 3d scenes,” in CVPR, 2023, pp. 5606–5611.
- [61] X. Ma, S. Yong, Z. Zheng, Q. Li, Y. Liang, S.-C. Zhu, and S. Huang, “Sqa3d: Situated question answering in 3d scenes,” arXiv preprint arXiv:2210.07474, 2022.
- [62] Z. Yuan, X. Yan, Y. Liao, Y. Guo, G. Li, S. Cui, and Z. Li, “Xtrans2cap: Cross-modal knowledge transfer using transformer for 3d dense captioning,” in CVPR, 2022, pp. 8563–8573.
- [63] Y. Jiao, S. Chen, Z. Jie, J. Chen, L. Ma, and Y.-G. Jiang, “More: Multi-order relation mining for dense captioning in 3d scenes,” in ECCV. Springer, 2022, pp. 528–545.
- [64] X. Yu, L. Tang, Y. Rao, T. Huang, J. Zhou, and J. Lu, “Pointbert: Pre-training 3d point cloud transformers with masked point modeling,” in CVPR, 2022, pp. 19313–19322.
- [65] L. Xue, M. Gao, C. Xing, R. Mart´ın-Mart´ın, J. Wu, C. Xiong, R. Xu, J. C. Niebles, and S. Savarese, “Ulip: Learning unified representation of language, image and point cloud for 3d understanding,” arXiv preprint arXiv:2212.05171, 2022.
- [66] M. Liu, R. Shi, K. Kuang, Y. Zhu, X. Li, S. Han, H. Cai, F. Porikli, and H. Su, “Openshape: Scaling up 3d shape representation towards open-world understanding,” NeurIPS, vol. 36, 2024.
- [67] J. Zhou, J. Wang, B. Ma, Y.-S. Liu, T. Huang, and X. Wang, “Uni3d: Exploring unified 3d representation at scale,” in ICLR, 2024.
- [68] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi, “Objaverse: A universe of annotated 3d objects,” in CVPR, 2023, pp. 13142–13153.
- [69] Z. Guo, R. Zhang, X. Zhu, Y. Tang, X. Ma, J. Han, K. Chen, P. Gao, X. Li, H. Li et al., “Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following,” arXiv preprint arXiv:2309.00615, 2023.
- [70] R. Xu, X. Wang, T. Wang, Y. Chen, J. Pang, and D. Lin, “Pointllm: Empowering large language models to understand point clouds,” arXiv preprint arXiv:2308.16911, 2023.
- [71] Z. Qi, Y. Fang, Z. Sun, X. Wu, T. Wu, J. Wang, D. Lin, and H. Zhao, “Gpt4point: A unified framework for point-language understanding and generation,” arXiv preprint arXiv:2312.02980, 2023.
- [72] Z. Qi, R. Dong, S. Zhang, H. Geng, C. Han, Z. Ge, L. Yi, and K. Ma, “Shapellm: Universal 3d object understanding for embodied interaction,” arXiv preprint arXiv:2402.17766, 2024.
- [73] D. Liu, X. Huang, Y. Hou, Z. Wang, Z. Yin, Y. Gong, P. Gao, and W. Ouyang, “Uni3d-llm: Unifying point cloud perception, generation and editing with large language models,” arXiv preprint arXiv:2402.03327, 2024.
- [74] R. Fu, J. Liu, X. Chen, Y. Nie, and W. Xiong, “Scene-llm: Extending language model for 3d visual understanding and reasoning,” arXiv preprint arXiv:2403.11401, 2024.
- [75] J. Huang, S. Yong, X. Ma, X. Linghu, P. Li, Y. Wang, Q. Li, S.-C. Zhu, B. Jia, and S. Huang, “An embodied generalist agent in 3d world,” arXiv preprint arXiv:2311.12871, 2023.
- [76] A. Kamath, M. Singh, Y. LeCun, G. Synnaeve, I. Misra, and N. Carion, “Mdetr-modulated detection for end-to-end multimodal understanding,” in ICCV, 2021, pp. 1780–1790.
- [77] B. Graham, M. Engelcke, and L. Van Der Maaten, “3d semantic segmentation with submanifold sparse convolutional networks,” in CVPR, 2018, pp. 9224–9232.

- [78] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pretraining of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.
- [79] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in ECCV. Springer, 2020, pp. 213–229.
- [80] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, “Masked-attention mask transformer for universal image segmentation,” in CVPR, 2022, pp. 1290–1299.
- [81] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor scenes,” in CVPR, 2017, pp. 5828–5839.
- [82] “Openai chatgpt,” https://openai.com/gpt-4.
- [83] B. Jia, Y. Chen, H. Yu, Y. Wang, X. Niu, T. Liu, Q. Li, and S. Huang, “Sceneverse: Scaling 3d vision-language learning for grounded scene understanding,” arXiv preprint arXiv:2401.09340, 2024.
- [84] J. Yang, X. Chen, S. Qian, N. Madaan, M. Iyengar, D. F. Fouhey, and J. Chai, “Llm-grounder: Open-vocabulary 3d visual grounding with large language model as an agent,” arXiv preprint arXiv:2309.12311, 2023.
- [85] D. Rozenberszki, O. Litany, and A. Dai, “Language-grounded indoor 3d semantic segmentation in the wild,” in ECCV, 2022.
- [86] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” arXiv preprint arXiv:2106.09685, 2021.
- [87] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar et al., “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.

#### Supplementary Files for Grounded 3D-LLM

The supplementary materials are organized as follows:

- 1) Additional ablation studies are included in Sec. 6.
- 2) Hyper-parameters of training are included in Sec. 7.
- 3) Visualizations of results and failure cases for downstream tasks are presented in Section 8.
- 4) Details on grounded scene-text data generation, visualizations of dataset statistics, and data generation prompts, are illustrated in Sec. 9.
- 5) Multi-task instruction-following templates are described in Sec. 10.
- 6) Extension to embodied dialogue and planning data is shown in Sec. 11

###### 6 ADDITIONAL ABLATION STUDIES

Effects of LLM model sizes. As shown in Table A-9, we compare the performance of GPT-2, Tiny-Vicuna-1B, Vicuna-7B, and Vicuna-13B. Smaller models like GPT-2 struggle with soft referent token learning. In contrast, Grounded 3D-LLM models, ranging from 1B to 13B parameters, significantly outperform GPT-2 and show marginal improvements on Scan2Cap, though they experience a slight decline on ScanQA. Notably, even with only the projection layer trainable (without LoRA), Grounded 3D-LLM (Vicuna-7B) achieves comparable performance, suggesting that the input scene embeddings are semantically rich and can be extracted by the LLM without intermediate trainable layers. Furthermore, the lack of improvement in language tasks with larger models indicates the current task scope may not fully challenge larger-scale LLMs.

- Table A - 9. Comparison of different LLMs. For GPT-2, we train the overall backbones with a smaller language head. In models without LoRA, only the projection layers are trained.

LLM LoRA

ScanQA Scan2Cap Trainable

Params B-4 C B-4@0.25 C@0.25 B-4@0.5 C@0.5

GPT-2 ✓ 8.42 44.6 28.4 45.2 26.7 41.9 169 MB Tiny-Vicuna-1B ✓ 13.2 75.9 36.9 74.1 35.0 70.2 157 MB Vicuna-7B ✓ 13.3 74.1 37.8 74.3 35.4 70.8 364 MB Vicuna-7B ✗ 12.4 70.9 37.4 74.2 34.7 69.9 330 MB Vicuna-13B ✓ 12.9 73.2 38.5 74.6 35.9 70.4 487 MB

7 IMPLEMENTATION DETAILS

The training hyper-parameters are detailed in Table A-10. All models are trained on 8 NVIDIA A100 (80G) GPUs.

- Table A - 10. Training hyper-parameters. The instruction fine-tuning batch size comprises 5 scenes, each containing up to 200 instructions.

Configuration CLASP pre-training Instruction fine-tuning

Queries 150 100 Epochs 600 50 Batch size 5 5×200 Learning rate 1e-4 8e-4

LR schedule Cosine Cosine Optimizer AdamW AdamW

###### 8 RESULT VISUALIZATIONS

Result visualization. We visualize the results in Fig. A-7 for a single model Grounded 3D-LLM that solves all downstream tasks, as opposed to per-task fine-tuning on downstream tasks [20], [21], [25].

Failure cases. As depicted in Fig. A-8, the primary reasons for our model’s failures in the grounding task include missing grounding objects (a, c), excessive grounding of semantically incorrect objects (b), and errors in detection/segmentation (c).

[Figure 159]

###### Object Detection

[Figure 160]

Can you assist me, by identifying any chairs in this vicinity? (with grounding)

Several <p> chairs </p> <ref> have been identified in this indoor setting.

[Figure 161]

###### ScanRefer (Single Object Grounding)

[Figure 162]

[Figure 163]

Your task involves locating something. Here's the detail: the trash can is blue and rectangular; the trash can is near the door with the knob on the left; the trashcan is on the left of the gray trash can. (with grounding)

[Figure 164]

The <p> object </p> <ref> has been detected.

[Figure 165]

###### Multi3DRefer (Multiple Object Grounding)

[Figure 166]

I have multiple items that need grounding: the black file cabinet is located beneath the desk (with grounding).

[Figure 167]

Objects found and grounded: <p> objects </p> <ref> .

[Figure 168]

###### ScanQA (Visual Question Answering)

[Figure 169]

What is at the foot of the bed? Please answer with a single word or phrase.

[Figure 170]

White rectangular nightstand.

###### Scan2Cap (Dense Captioning)

[Figure 171]

[Figure 172]

Can you describe the object <ref>'s form and then talk about how it's positioned with its environment?. (with grounding)

[Figure 173]

This is a brown cabinet. It is in the corner of the room. Ground Truth: This is a tan cabinet. It is below a picture.

###### Object Description

[Figure 174]

[Figure 175]

What does the object represented by <ref> look like?. An oval-shaped sink with a white, glossy finish. Ground Truth: A white oval-shaped sink with chrome faucet and speckled countertop.

[Figure 176]

Figure A - 7. One model to solve all tasks. Result visualization in multiple downstream benchmarks for Grounded 3DLLM. Please zoom in for better visualization.

###### 9 GROUNDED SCENE CAPTION DATASET

Data generation details. For the creation of Grounded Scene Captions (G-SceneCap), we selected the ScanNet-200 dataset due to its extensive range of semantic object categories. The data generation process employs heuristics to ensure that each local scene caption is concise, with fewer than 256 words. Initially, we identify each anchor object within a 2-meter radius to tally the objects. If this count exceeds 15, we reduce the radius to maintain this limit. Subsequent object selection is conducted through random sampling, with selection probabilities ranging from 0.6 to 0.9. This method prevents the selection of repetitive object types while preserving the anchor. Finally, for each caption, we generate program-derived relations that use existing objects as the anchor following SR3D [57]. These relations are then integrated into the caption. The overall process always keeps both the anchor and target object IDs.

Post processing. As ChatGPT/GPT-4 may generate errors such as incorrect output formats or erroneous object IDs, we filter out these wrong captions from the dataset. Dataset language components visualization. For the grounded scene caption dataset, we visualize in Fig. A-9 the

m3dref Scene0196_00

###### Scene0222_00

scene0490_00

Failure case

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 15

[Figure 177]

Missing Grounding

[Figure 178]

Can we identify the positions of these objects: discover a table surrounded by a few chairs?(with grounding)

[Figure 179]

The locations of the objects are marked as <p> objects </p> <ref>.

Missing Grounding

[Figure 180]

[Figure 181]

Could you pinpoint these objects for me: the bed, adorned in a soothing blend of grey sheets and a tranquil blue cover, rests snugly against the wall that boasts windows? (with grounding)

- (b)

(a)

- (c)

I've located the object you're looking for: <p> object </p> <ge>.

[Figure 182]

Wrong Grounding

Wrong Detection

[Figure 183]

[Figure 184]

Let's locate these specific items, a gleaming whiteboard adorns the wall, complete with a tray of markers and erasers at its bottom edge (with grounding).

[Figure 185]

Missing Grounding

I've located the object you're looking for: <p> object </p> <ref>.

Figure A - 8. Failure cases. Please zoom in for better visualization.

relationships between noun-adjective pairs and spatial preposition-noun pairs across all languages, along with a word cloud of all nouns, to demonstrate the diversity of our curated dataset.

As shown in {}, our model's failure in the grounding task is mainly due to missing grounding objects (a, c), excessively grounding semantically incorrect objects (b), and detection/segmentation faults (c).

wooden

black

wall side

rectangular round

[Figure 186]

several

wooden

light

right

top

cozy

table

on

modern

chair

table

spacious

room

modern

curtain

long

behind

shelf

from

colored

beside

white

door

window

tall

into

wooden

backpack

across

white

chair

within

table

door

pillow

around

large

standard

wooden

room

rectangular

book

box

below

room

toilet

red

black

standard

above

between

colored

bookshelf

cabinet

rectangular

desk

rectangular

bed

reach

towel

near

doorframe

space

couch

room

against

chair

sink

white

room

area

several

rectangular

table

white

brown

standard

picture

cabinet

book

cabinet

window

sized

white

various

wooden

rectangular

counter

cabinet

standard

double

chair

wooden

wooden white tall

sized

desk

window desk

bathtub

sink

wooden

white

table

wall

black

rectangular

couch

door

brown

bed

colored

white

dark

deep

stainless

light

tall

modern

light

- Figure A - 9. Data statistics of grounded scene caption. Left: Diagram of spatial preposition-noun statistics, Middle: Nounadjective statistics diagram, Right: Noun word cloud. Dataset visualization. We provide additional visualizations of grounded scene caption examples in Fig. A-10.

Grounded Scene Caption

Raw Point Cloud Segmented Point Cloud

[Figure 187]

[Figure 188]

Perched near [the rectangular kitchen counter 0], [a white rectangular microwave 13] finds its place, with [a wooden wall-mounted rectangular cabinet 12] securely mounted on the wall above it. Adjacent to [the kitchen counter 0] is [a rectangular gray trash can 1], suggesting an efficient layout for food preparation and waste management. In close proximity to [the trash can 1] stands [another wooden kitchen cabinet 17], which contributes additional storage space. Opposite [the kitchen counter 0], [another smooth, light-colored kitchen counter 5] serves as a complementary work surface, enhancing the room's functionality. Suspended just in front of [counter 5], [a colorful box 21] is cleverly positioned near the ceiling, hinting at a blend of practical storage and aesthetic display. To the left, from the vantage point facing [counter 5], [a stainless steel coffee maker 25] claims its territory, likely signifying a dedicated beverage area within this well-appointed kitchen.

- (a)
- (b)
- (c)

[The silver door handle 10] is conveniently located near [the orange towel 18], indicating its accessibility for someone immediately after bathing. [The teal mat 56], placed on the floor, offers a comfortable and dry spot to step onto after exiting [the round bathtub 73]. [The white shower curtain 72] is pulled closed around [the bathtub 73], providing privacy and delineating the wet area from the rest of the bathroom. Above [the bathtub 73], [a smooth horizontal bar 70] is mounted, likely serving as an

anchor for [the shower curtain 72], while [another horizontal metal bar 71], positioned closer to [the bathtub 73], suggests a functional role for hanging items such as [the towel 18] within arm's reach of the bath.

[Figure 189]

[Figure 190]

[Figure 191]

- (d)

[Figure 192]

[A multi-colored sectional couch 2] anchors the space, with [a chair with a black backrest and an orange seat 6] positioned nearby, emphasizing a cohesive yet dynamic seating area. [A glossy black piano 3], suggesting a music-focused corner, is accompanied by [a simple black music stand 11], reinforcing the area's intended use for practice or performance. Close to [the couch 2], there’s [a picture with blue scribbles and doodles 14], adding a visual element to the lounge area, while [a picture with vertical stripes of varying widths and colors 13] is placed further away, possibly

[Figure 193]

creating a visual point of interest across the room. Meanwhile, [a modern chair with a brown cushioned seat 9] stands at a distance from [the piano 3], whereas [a sleek chair with a black backrest and an orange cushioned

seat 7] is placed near it to cater to an audience or a resting musician.

[Figure 194]

[A lofty cardboard box 62] is positioned near [a light brown wooden nightstand 63]. Perched atop [the nightstand 63] lies [a lustrous metallic bag 64]. In proximity, [another cardboard box 65], adorned with a cheerful smiling face graphic, appears slightly raised with glimpses of red fabric emerging from its interior. Abutting these items is [a rectangular cardboard box 66] with a striking red and black motif. Nestled between [the nightstand 63] and [the box 66] is [a rectangular cardboard-based trash can 67], indicative of a space-conscious arrangement in the room.

- Figure A - 10. Example visualization of grounded scene caption dataset. The “[ ]” symbol marks each grounded phrase. The color of the phrase corresponds to the segmented point cloud.

Data generation prompts Fig. A-11 displays the object caption prompts, including the VLM captioning prompt [38] and the subsequent condensation of the caption into a descriptive phrase using ChatGPT. These object captions are then compiled to create local scene captions, as shown in Fig. A-12, while retaining their object IDs. Finally, spatial relationships derived from program-generated code are incorporated using the prompt depicted in Fig. A-13.

###### 10 TASK-SPECIFIC INSTRUCTION-FOLLOWING TEMPLATES

Grounded 3D-LLM simultaneously undertakes vision tasks—such as 3D visual grounding, 3D object detection—and language tasks, like question answering. Each task demands specific, diverse templates to enable the LLM to naturally provide scene-aware answers tailored to particular dataset specifications. We show several instruction-following templates in the following figures.

Object detection. The detection tasks employ various question templates for each category, along with three answer templates to accommodate various detection outcomes – single, multiple, or no objects detected – as illustrated in Fig. A-14.

User: CogVLM

Please provide a detailed description of the visual characteristics and physical attributes of the {object name} present in the image.

GPT

System:

You are an AI writing assistant that can analyze a caption and abbreviate it into a descriptive phrase. NOTE:

- 1. Only contain attributes (color, shape, and material) from the given caption. Do not add other attributes.
- 2. Remove all spatial relationships with other objects.
- 3. Aim for descriptions within 10 words.
- 4. Your response should solely consist of the generated phrase. No additional information or content is permitted.
- 5. If the key object in the raw caption is not consistent with the given class name, the output should be replaced by "A <class name>.“ Examples of the descriptive phrase:

- 1. A rectangular, hard-shell dark-colored suitcase made of sturdy material.
- 2. A dark brown wooden table with a rectangular shape and cylindrical legs.

User:

Given caption for the suitcase: "The suitcase in the image appears to be a rectangular, hard-shell suitcase. It has a dark color, possibly black or dark gray. The suitcase seems to be made of a sturdy material, likely ABS or polycarbonate, which are common materials for hard-shell suitcases. There are no visible brand logos or identifiers on the suitcase."

ChatGPT:

A rectangular, hard-shell dark-colored suitcase made of sturdy material.

- Figure A - 11. Object caption prompt for VLM and ChatGPT (used in Step-1 of data generation). The above is the prompt for CogVLM that utilizes the object labels from ScanNet-200. Below is the prompt for ChatGPT input, where “System” denotes the system prompt and “User” denotes user input.

You are an AI visual assistant that generates captions for 3D indoor scenes. Generate a caption for the given 3D indoor scene based on a list of key objects with unique IDs, 3D coordinates, and appearances. NOTE:

- 1. Input: List of key object information in format "[<object name> <object id>] (X, Y, Z): <object appearance>". The units of (X, Y, Z) are in meters.
- 2. Output: Natural language caption including all key objects and their attributes following the OBJ-ID format: "[<the/a/number> <object name (or descriptive phrase of the object)> <object id(s)>]". Merge similar objects.
- 3. Focus on spatial relationships between key objects. Infer simple relations (e.g., "nearby", "on top of", "below"). Ensure accuracy and avoid assuming complex relationships. The relation should consider the object's normal size. Be careful when dealing with large objects.
- 4. Avoid using coordinates in output.
- 5. The caption should be concise and clear. All the objects mentioned in the caption should follow the OBJ-ID format.
- 6. Word limit: Fewer than 100 words. Focus on the description of spatial relations instead of objects' functions. An example caption: "There are [two full-color magazines 10, 12] resting on [the rectangular table 9]. Nearby, [a brown wooden chair 5] is positioned to the left of [the table 9], while [another blue chair 6] is to the right of [the table 9]."\\

System:

User:

List of key objects: [couch 2] (1.0, 1.4, 0.3): A multi-colored sectional sofa upholstered in fabric. [piano 3] (-1.5, 0.7, 0.6): A glossy black piano with horizontal strings. [chair 6] (1.4, -1.1, 0.5): A chair with a black backrest and an orange seat cushion. [music stand 11] (-1.6, 1.7, 0.5): A simple, black rectangular music stand.

. . .

Please generate the scene caption given above inputs.

- Figure A - 12. GPT-4 prompt for scene caption (used in Step 2 of data generation). “System” denotes the system prompt and “User” denotes user input for GPT-4.

System:

You are an AI visual assistant that generates captions for 3D indoor scenes. You will be given an original caption and some additional spatial relationships. Rewrite the original caption to include these relationships. NOTE:

- 1. Each key object has its own unique object ID. You MUST refer to the key objects using the format: "[<the/a/number> <object name (or descriptive phrase of the object)> <object id(s)>]". You can group same category objects of similar appearance with plural form like: "[Two brown chairs 9, 10] rest close to [the rectangular table 9]".
- 2. Do not alter the provided spatial relationships.
- 3. Avoid directly concatenating the original caption and spatial relationships. Find the key objects that exist both in original caption and the given spatial relationships, combine them and rewrite it naturally.
- 4. The additional relations could contain something like "closer to". You should know which object it's comparing with, otherwise, do not directly use these relations.
- 5. Your response should solely consist of the rewritten caption. No additional information or content is permitted.
- 6. The generated caption should be natural and unambiguous.
- 7. Word limit: Fewer than 250 words.\\

User:

Original caption: "[A multi-colored sectional couch 2] is situated with [a chair 6] nearby, the latter featuring a black backrest and an orange seat cushion. Across the room, [a glossy black piano 3] is positioned near [a simple black music stand 11]."

Additional spatial relationships: [A rectangular artwork with vertical stripes of varying widths and colors 13] is far from [the couch 2]. [A whiteboard with blue scribbles and doodles 14] is close to [the couch 2]. [A sleek chair with a black backrest and an orange cushioned seat, made of metal 7] is near [the piano 3].

- Figure A - 13. GPT-4 prompt of including rule-based relation (used in Step 3 of data generation). “System” denotes the system prompt and “User” denotes user input for GPT-4.

Detection Prompt

Question templates:

“Detect all the {category}s present in this indoor scene.” “Try to give me all {category}s you can find.” “Can you find all {category}s in the indoor environment?” “What {category}s are available in this room?” “Identify all {category}s within this enclosed space.”

Answer templates:

Single object: “Detected {category} in this indoor scene.” “Yes, there is a {category} within the interior.” “A {category} has been identified in this indoor setting.” Multi-object: “Detected {category}s in this indoor scene.” “Yes, there are {category}s within the interior.” “Several {category}s have been identified in this indoor setting.” No object: “No {category} detected in this indoor scene.” “There are no {category} within the interior.” “No {category} found in the indoor environment.”

###### Figure A - 14. Instruction-following templates for object detection.

Single and multi-object grounding. The language grounding’s question template, as depicted in Fig. A-15, directs the model to identify referred objects. For the ScanRefer grounding dataset, the answer templates indicate single-object grounding. The answer templates for the Multi3DRef dataset accommodate various grounding scenarios—single, multiple, or no grounded objects.

3D Question answering. For ScanQA, we append suffixes to brief outputs (Fig. A-16) to indicate the outputs with only phrases or word output as the dataset annotation guideline.

Dense captioning. Question templates for Scan2Cap require the model to describe object appearance followed by spatial relations, as illustrated in Fig. A-17.

Grounding Prompt

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 19

Question templates:

“Can you locate these items for me: {grounding text}?” “I'm looking to find multiple objects: {grounding text}” “We need to identify several items: {grounding text}” “Could you help me by grounding these objects: {grounding text}?” “I require assistance in locating these objects: {grounding text}”

Answer templates:

Single object: “I've successfully identified the {category}.” “Here is the {category} based on your request.” “The grounding process identified this {category}.” Multi-object: “{category}s found and grounded.” “I've successfully identified the {category}s.” “Here are the {category}s based on your request.” None object: “The grounding process did not detect any objects.” “I couldn't ground any objects .” “No objects match the provided descriptions .”

scanqa Prompt

###### Figure A - 15. Instruction-following templates for single- or multi- object grounding.

Question templates:

{raw question} Please answer with a single word or phrase.

Figure A - 16. Instruction-following templates for ScanQA.

- 11 EXTENSION TO EMBODIED DIALOGUE AND EMBODIED PLANNING

Fig A-18 and A-19 list the prompts for grounded scene caption-based embodied dialogue and planning. The training dataset includes 11.9K dialogue examples and 4.4K planning examples.

Result visualization. As there is no well-defined benchmark for evaluating both phrase grounding and language understanding, we directly visualize the results of the jointly trained Grounded 3D-LLM for embodied dialogue (Fig. 20) and embodied planning (Fig. 21). These examples demonstrate the potential of Grounded 3D-LLM’s grounding capability to enhance robot navigation and manipulation.

Scan2Cap Prompt

Question templates:

"First, detail the look of the object <vp>, then explain how it is positioned in relation to its immediate surroundings.", "What does the object <vp> look like, and how is it situated within its nearby environment?", "Provide a description of the object <vp>'s appearance and then its spatial relationship with the surrounding area.", "How is the object <vp> visually characterized, and what is its placement in respect to its surroundings?", "Illustrate the physical features of the object <vp> and describe its location relative to nearby objects.", "Can you describe the object <vp>'s form and then talk about how it's positioned with its environment?", "Detail the appearance of the object <vp> first, followed by a description of its local spatial relations.", "Explain the visual attributes of the object <vp> and its spatial arrangement with the surrounding.", "What does the object <vp> resemble, and where does it stand in relation to its immediate surroundings?", "Describe the aesthetics of the object <vp> and its orientation within its immediate vicinity."

###### Figure A - 17. Instruction-following templates for Scan2Cap.

System:

System:

Given a local scene caption with object IDs, you are tasked with creating a dialogue with 3-5 rounds between a user and a robot assistant. The dialogue can include the following contents:

Given a local scene caption with object IDs, you are tasked with creating a JSON dictionary containing 2-4 high-level tasks for the robot assistant. Each task should include a clear, concise step-by-step plan using the objects identified in the scene. Rules:

- - User asks about the appearance(spatial relation, color, shape, material) or design purpose of objects.
- - User engages in discussions regarding the layout of the local scene.
- - User wants assistant to handle simple tasks related to scene objects and associated human activities. Rules:

- 1. Format your response as a JSON dictionary, where each key is a high-level task name and its value is a list of steps in the format of ```json {"name of task1 ": ["step1", "step2", ...], ...} ```
- 2. Each step should be simple and under 10 words.
- 3. Each given object has its own unique object ID. You MUST refer to the given objects using the OBJ-ID format: "[<the/a/number> <object name (or descriptive phrase of the object)> <object ID(s)>]".
- 4. Do not add any objects or details not mentioned in the scene caption.

- 1. You MUST refer to the given objects using the OBJ-ID format: "[<the/a/number> <object name (or descriptive phrase of the object)> <object id(s)>]".
- 2. You MUST start each part of the dialogue with either "USER:" or "ASSISTANT:". Ensure there is a line break after each role exchange.
- 3. Do not add any objects or details not mentioned in the scene caption.
- 4. The question and answer should be concise and natural. You can omit appearance details in the descriptive phrase and use only the object name in the OBJ-ID format.

User:

{Grounded scene captions}

User:

{Grounded scene captions}

Figure A - 18. GPT-4 Prompt of embodied dialogue. “System” denotes the system prompt and “User” denotes user input for GPT-4. “{Grounded scene caption}” is filled with our grounded scene caption data.

System:

System:

Given a local scene caption with object IDs, you are tasked with creating a dialogue with 3-5 rounds between a user and a robot assistant. The dialogue can include the following contents:

Given a local scene caption with object IDs, you are tasked with creating a JSON dictionary containing 2-4 high-level tasks for the robot assistant. Each task should include a clear, concise step-by-step plan using the objects identified in the scene. Rules:

- - User asks about the appearance(spatial relation, color, shape, material) or design purpose of objects.
- - User engages in discussions regarding the layout of the local scene.
- - User wants assistant to handle simple tasks related to scene objects and associated human activities. Rules:

- 1. Format your response as a JSON dictionary, where each key is a high-level task name and its value is a list of steps in the format of ```json {"name of task1 ": ["step1", "step2", ...], ...} ```
- 2. Each step should be simple and under 10 words.
- 3. Each given object has its own unique object ID. You MUST refer to the given objects using the OBJ-ID format: "[<the/a/number> <object name (or descriptive phrase of the object)> <object ID(s)>]".
- 4. Do not add any objects or details not mentioned in the scene caption.

- 1. You MUST refer to the given objects using the OBJ-ID format: "[<the/a/number> <object name (or descriptive phrase of the object)> <object id(s)>]".
- 2. You MUST start each part of the dialogue with either "USER:" or "ASSISTANT:". Ensure there is a line break after each role exchange.
- 3. Do not add any objects or details not mentioned in the scene caption.
- 4. The question and answer should be concise and natural. You can omit appearance details in the descriptive phrase and use only the object name in the OBJ-ID format.

User:

{Grounded scene captions}

User:

{Grounded scene captions}

Figure A - 19. GPT-4 Prompt of embodied planning. “System” denotes the system prompt and “User” denotes user input for GPT-4. “{Grounded scene caption}” is filled with our grounded scene caption data.

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Can you bring the sturdy black backpack and place it on the dark desk <ref> with white and green keys?

Can you please move the black backpack <ref> from next to the piano and place it on the brown leather ottoman?

[Figure 201]

[Figure 202]

###### Certainly. I will move <p> the backpack </p> <ref> to <p> the desk </p> <ref> right away.

Certainly, I will relocate <p> the backpack a</p> <ref> next to <p> the piano </p> <ref> and place it on <p> brown leather ottoman </p> <ref> for you.

[Figure 203]

Thanks. Also, can you describe the lamp <ref> to me? I‘m thinking about redecorating soon.

[Figure 204]

I noticed the piano <ref> looks quite elegant. Can you tell me, was it designed for professional use or just for a home setting?

[Figure 205]

<p> The tall, slender lamp </p> <ref> is a classic design with a white or light-colored shade. Its base is cylindrical and made of wood or similar material.

[Figure 206]

<p> The polished wooden piano </p> <ref> is designed for professional use. Since we're reorganizing, could you put the vibrant red toolbox and another toolbox into the storage area behind the wooden door with horizontal louvered panels and a brass knob? Of course. I'll place <p> the box </p> <ref> and <p> the box </p> <ref> neatly into the storage area behind <p> the door </p> <ref>.

[Figure 207]

[Figure 208]

It sounds elegant. I'm curious, why are the three monitors set up alongside the bed? Isn't that uncommon?

[Figure 209]

The placement of <p> monitors </p> <ref> next to <p> the bed </p> <ref> is likely for entertainment and work.

[Figure 210]

[Figure 211]

Makes sense. What's in the brown cardboard box <ref> ? The <p> box </p> <ref> is used for storage or could contain items for shipping or organizing. Would you like me check the contents for you? No, that's fine. Could you arrange the pillows at the head of the bed neatly? Certainly, I will place <p> pillow </p> <ref> at the head of <p> bed </p> <ref> to ensure a tidy appearance.

[Figure 212]

[Figure 213]

While you do that, can you tell me why there are two different styles of wooden doors? What's their purpose?

[Figure 214]

[Figure 215]

The two different styles of <p> door </p> <ref> are designed for different purposes. <p> The light-colored door </p> <ref> is intended for aesthetic appeal, while <p> the darker door </p> <ref> is designed for security or privacy.

[Figure 216]

- Figure A - 20. Result visualization for embodied dialogue. Please zoom in for better visualization.

[Figure 217]

[Figure 218]

I aim to accomplish dispose of trash.

[Figure 219]

I plan to prepare the area for piano practice.

[Figure 220]

[Figure 221]

[Figure 222]

I desire to polish the mirror and rail. Can you assist me with this?

Understood. Let's kick off the planning phase.

- step 1. Straighten <p> pillow </p> <ref>.
- step 2. Smooth out <p> bedding </p> <ref> on <p> bed </p> <ref>.

- step 2. Fluff <p> pillow </p> <ref>.
- step 3. Place <p> pillow </p> <ref> on <p> bed </p> <ref>. Planning is done. What's the next

[Figure 223]

[Figure 224]

Understood. Let's kick off the planning phase.

- step 1. Gather trash from <p> table </p> <ref>.
- step 2. Place trash into <p> trash can </p> <ref>.
- step 3. Return <p> trash can </p> <ref> to original spot. Finished planning. What's our next course of action?

Noted. We'll start the planning process promptly.

- step 1. Spray cleaner on <p> mirror </p> <ref>.
- step 2. Wipe <p> mirror </p> <ref> with cloth.
- step 3. Clean <p> rail </p> <ref> with cloth.
- step 4. Dry <p> rail </p> <ref> with dry cloth. Planning is done. What's the next step in our process?

Noted. We'll start the planning process promptly.

- step 1. Position <p> piano bench </p> <ref> against wall.
- step 2. Clear area around <p> piano </p> <ref>.
- step 3. Adjust <p> bench </p> <ref> parallel to <p> piano </p> <ref>.
- step 4. Ensure <p> bench </p> <ref> does not obstruct <p> piano </p> <ref>. I've finished the planning. What's our next move?

My objective is to achieve prepare the bed area. Could you provide some assistance?

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

- Figure A - 21. Result visualization for embodied planning. Please zoom in for better visualization.

