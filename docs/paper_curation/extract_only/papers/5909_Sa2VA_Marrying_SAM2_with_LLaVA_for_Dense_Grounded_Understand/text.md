# arXiv:2501.04001v3[cs.CV]3Nov2025

[Figure 1]

## Sa2VA: Marrying SAM2 with MLLM for Dense Grounded Understanding of Images and Videos

### Haobo Yuan1∗, Xiangtai Li2∗†, Tao Zhang2,3∗, Yueyi Sun4, Zilong Huang2, Shilin Xu4, Shunping Ji3, Yunhai Tong4, Lu Qi3, Jiashi Feng2, Ming-Hsuan Yang1

University of California, Merced1 Bytedance Seed2 Wuhan University3 Peking University4 †Project lead ∗Equal technical contribution Code: https://github.com/Bytedance/Sa2VA Model: https://hf.co/collections/ByteDance/sa2va-model-zoo

### Abstract

This work presents Sa2VA, the first comprehensive, unified model for dense grounded understanding of both images and videos. Unlike existing multi-modal large language models, which are often limited to specific modalities and tasks, Sa2VA supports a wide range of image and video tasks, including referring segmentation and conversation, with minimal one-shot instruction tuning. Sa2VA combines SAM-2, a foundation video segmentation model, with MLLM, the advanced vision-language model, and unifies text, image, and video into a shared LLM token space. Using the LLM, Sa2VA generates instruction tokens that guide SAM-2 in producing precise masks, enabling a grounded, multi-modal understanding of both static and dynamic visual content. Additionally, we introduce Ref-SAV, an auto-labeled dataset containing over 72k object expressions in complex video scenes, designed to boost model performance. We also manually validate 2k video objects in the Ref-SAV datasets to benchmark referring video object segmentation in complex environments. Experiments show that Sa2VA achieves strong performance across multiple tasks, particularly in referring video object segmentation, highlighting its potential for complex real-world applications. In addition, Sa2VA can be easily extended into various VLMs, including Qwen-VL and Intern-VL, which can be updated with rapid process in current open-sourced VLMs. Code and models have been provided to the community.

Date: November 4, 2025 Correspondence: Xiangtai Li at xiangtai.li@bytedance.com

### 1 Introduction

Multi-modal Large Language Models (MLLMs) have made significant progress, fueled by the rapid development of Large Language Models (LLMs) [39, 83, 101]. Numerous MLLMs have been applied to image- and video-level tasks such as visual question answering (VQA) [1, 80], narrative story generation [33, 102, 122], and interactive editing [36, 45, 85]. One impor-

tant direction is to understand video content in a fine-grained manner, including segmenting and tracking pixels with language descriptions, and performing VQA on visual prompts in the video. In particular, we aim to achieve promptable fine-grained analysis of video, enabling the user to be in the loop when playing the video in an interactive mode, as shown in Fig. 1 (a). This capability will enable various applications, such as short video editing [8, 64, 84], robot

|The video shows a woman walking down a busy city street at night. She is wearing a red dress and carrying a black handbag. The street is illuminated by bright lights, and there are many people walking around. The woman is walking down the street, passing by many people, and looking around. She is wearing sunglasses and has a con fident and stylish demeanor. The video captures the vibrant and lively atmosphere of the city at night, with the bright lights and bustling crowds…<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>Can you describe the video?<br><br>[Figure 16]<br><br>What is the person holding? The person is holding a black handbag. Can you segment the person wearing sunglasses on the street? Sure, it is [SEG].<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]|
|---|

(a)

|Image Caption The image features a snow boarder in mid-air, performing a jump on a snowy hill. The sn owboarder is skillfully riding a red snowboard, which is visibl e in the center of the scene. Th e snowboarder is wearing a wh ite jacket, which contrasts with the snowy surroundings.<br><br>[Figure 20]<br><br>[Figure 21]|Video Caption<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>The video shows a young man skateboarding in a city plaza. The man is seen riding the skateboard in different locations, in cluding a sidewalk and a street. The video captures the man's movements, showcasing his skills and the urban environment.<br><br>[Figure 29]|
|---|---|
|Image Conversation<br><br>[Figure 30]<br><br>[Figure 31]<br><br>What kind of animal is it?<br><br>It is a giraffe. What is the giraffe doing?<br><br>[Figure 32]<br><br>[Figure 33]<br><br>The giraffe is eating leaves from a tree.<br><br>[Figure 34]|Video Conversation<br><br>What is the weather now? The weather is foggy. What is the color of the front car? It is black.<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]|
|Referring Image Segmentation<br><br>[Figure 46]<br><br>[Figure 47]<br><br>Please segment the bl onde-haired girl dancing in a blue dress.<br><br>Sure, it is [SEG].<br><br>[Figure 48]|Referring Video Segmentation<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>Please segment the person weari ng a light-colored hoodie. Sure, it is [SEG].<br><br>[Figure 56]|

| | |
|---|---|
| | |

(b) (c)

- Figure 1 Illustration of capabilities of our proposed Sa2VA. (a). Given a video, Sa2VA can segment the referred object and understand the whole scene. (b). Sa2VA supports image chat, video chat, image referring segmentation, video referring segmentation, and grounded caption generation with single-shot instruction-tuning. (c). Sa2VA achieves strong results on multiple images, video referring segmentation, and chat benchmarks compared to existing MLLMs [76, 100, 114].

navigation [27, 29, 30], and surveillance analysis [72].

cific video task [4, 41, 70, 100]. To the best of our knowledge, no work has successfully integrated the strengths of both sides to build a unified model that supports both image and video inputs across a diverse range of tasks (Tab. 1).

However, neither state-of-the-art video perception models [16, 55, 77, 113] nor video MLLMs [32, 49, 56, 63, 110] can achieve this. The former are limited by constrained semantic concepts and lack openended capabilities, such as video question answering (Video-QA) [52, 56]. While the state-of-the-art SAM2 model [77] can perform promptable segmentation and tracking, it cannot handle text-aware tasks, such as understanding language expressions or video conversation. On the other hand, video MLLMs can understand videos and perform Video-QA. For example, the latest LLaVA [47] can achieve strong results on Video-QA. However, it cannot carry out perception tasks nor understand visual prompts. Several works [24, 75, 94, 100, 114] have explored the combination of perception models and MLLMs. However, existing works mainly focus on exploring image tasks [24, 107, 114] or trying to solve one spe-

Based on the above analysis, we set out to unify two visual foundation models, SAM-2 [77] and MLLMs [61] into one framework. As such, our model not only inherits the spatial-temporal perception capabilities of SAM-2 and the open-ended abilities of MLLMs but also benefits from learning additional knowledge derived from the additional training data. To achieve this goal, three key challenges must be addressed: (1) Task formulation: How to effectively formulate a range of tasks into a unified training setting, particularly for multi-modal inputs. (2) Performance balance: How to resolve conflicts between tasks, such as ensuring strong referring visual understanding abilities without sacrificing the language proficiency of

MLLMs. Note that existing methods [95, 114] find that conversation tasks are drastically degraded when grounding tasks are performed. The models are thus degraded to specialists. (3) Knowledge Inheritance: How to leverage the pre-trained knowledge from SAM2 and MLLMs to build a unified model that is both robust and flexible with different foundation models. We note that SAM-2 is trained on more than 1B masks, and most MLLMs are trained with massive instruction-following data pairs. Furthermore, both model families are evolving rapidly, and a flexible framework is crucial to more easily leverage their future progress.

In this work, we present Sa2VA, the first model that integrates SAM-2 with MLLMs, creating a unified, grounded understanding of both images and videos. We first formulate various tasks into a one-shot visual instruction tuning process, including referring segmentation, visual question answering (VQA), and grounded conversation generation (GCG) for both image and video. We leverage the token-length flexibility of LLMs, treating all input images, videos, and visual prompts as tokens without additional specific design. Through joint training, we demonstrate that conflicts between grounding and conversation tasks can be effectively resolved. Unlike existing approaches that use MLLMs as agents or connectors to call on visual experts, our model is trained end-to-end, showcasing model and dataset scalability. Moreover, we adopt a decoupled design in which SAM-2’s encoder and memory module are frozen, allowing us to retain its tracking capabilities. This design also makes our method a flexible framework, enabling our model to evolve with increasingly powerful MLLMs. To be more specific, our framework is compatible with advanced VLMs, such as InternVL [124] and Qwen3-VL [97].

In addition, we empirically observe that existing video segmentation datasets are limited to small-scale collections of short clips with a limited number of occlusions. Thus, to bridge the gaps between current datasets and the real-world requirements of interactive applications, we introduce Ref-SAV: a new, challenging reference video segmentation dataset curated through an automated pipeline based on the recent SA-V datasets [77]. Finally, we benchmark several existing models on this dataset and find significant room for further exploration.

Sa2VA is co-trained with multiple datasets, including image and video modalities and various tasks. Without any bells and whistles, Sa2VA achieves strong performance on six referring image and video segmen-

tation datasets while maintaining strong image and video QA capabilities compared to previous MLLMs, as shown in Fig. 1(c). On the proposed Ref-SAV dataset, our method achieves over 15% performance gain over existing approaches under the zero-shot test setting, and obtains even stronger results using a large-scale Ref-SAV training set (about 37k videos). As shown in Fig. 1(c), our work builds a new strong baseline for a unified, dense, grounded understanding of images and video. Our main contributions are:

- • We develop Sa2VA, the first simple framework that combines SAM-2 and MLLM models into one model. Sa2VA achieves strong spatialtemporal grounded understanding performance across various benchmarks. Sa2VA can be supported with various modern VLMs, including InternVL [124] and Qwen-VL [3].
- • We introduce a challenging video benchmark, Ref-SAV, with a manual check for each example. The benchmark introduces heavier occlusions, long text inputs, and motion blurs.
- • We develop a simple data annotation pipeline to build the Ref-SAV training dataset, where we find that the training dataset improves model performance on Ref-SAV.
- • Extensive experiments show the effectiveness of Sa2VA on various open-sourced benchmarks (over 15 benchmarks, including image, video VQA, visual prompt, referring image/video segmentation). To the best of our knowledge, few existing MLLMs have shown comparable performance across such a diverse set of the abovementioned domains.

### 2 Related Work

Multi-modal Large Language Models. Earlier works [37, 50, 51] explore better multi-modal fusion methods and feature extractors and design various fusion architectures, particularly for vision language tasks. With the advances of LLMs [5, 39, 83], multimodal instruction tuning on LLMs [2, 11, 61, 81] have recently attracted much attention. Notably, various benchmarks [25, 38, 57, 66, 82] have been established to enhance model performance, reinforcing the notion that data is crucial in developing state-of-theart MLLMs. One representative work, LLaVA [61], considers the visual features as visual tokens. Furthermore, LLaVA provides a unified data format to consolidate extensive VQA tasks. Several works [107] subsequently explore stronger visual cues to improve the visual inputs of LLaVA. Meanwhile, numerous

Table 1 Capabilities of different representative models. We compare Sa2VA against other representative works. As shown, while there are several groups of models that share similar capability sets (e.g., focusing on video conversation or dense grounding), no single existing method covers the full spectrum of tasks. In contrast, Sa2VA is designed to comprehensively support all listed modalities and tasks (also see Fig. 1 (a) and (b)).

Support Inputs Dense Grounding Conversation End to End

Method

Image Video Visual Prompts RES Ref-VOS Inter-VOS GCG Image-Chat Video-Chat Video Caption Interactive Caption Training

Image-Chat Only Models

LLAVA [61] Image & Video Chat Models LLaVA-OneVision [47] InternVL [14] LLaMA-VID [56] P-LLaVA [98]

Visual Prompt & Dense Grounding Models

Osprey [107] VIP-LLaVA [6] GLAMM [76] LISA [46] OMG-LLaVA [114]

Dense Grounding Models

F-LLM [95] VISA [100] HyperSeg [88] InstructSeg [87]

Sa2VA (Ours)

methods [20, 21, 35, 46, 58, 74, 78, 109–112, 115] incorporate additional components to adapt LLaVA for visual grounding, detection, segmentation, and video VQA. Recently, a new trend has emerged to unify image, video, and multi-image analysis within a single framework [47, 63]. LLaVA-OneVision [47] designs a single model to handle four different input sources. In visual perception, several works [55, 77, 99, 116] also explore multi-dataset and multi-task co-training. SAM-2 [77] proposes a unified model for joint image and video interactive segmentation. Our model, Sa2VA, integrates SAM-2 into existing VLM models to create a unified end-to-end model. It aims to combine image and video for dense, grounded understanding, encompassing tasks such as segmentation, chat, and captioning.

Referring Segmentation. This task aims to output specific masks (image) or tracked masks (video) driven by language description. Earlier studies [18, 44, 59, 93, 106, 119] examine various fusion modules to enhance performance. Then, several stronger models [99] adopt DETR-like methods [7, 12, 71, 121, 125] to achieve unified segmentation and tracking in video. Equipped with LLMs, several recent works [9, 46, 74, 89, 96, 114] have been developed to accomplish more complex referring tasks, including reasoning for referring or joint mask and caption generation. For example, LISA [46] involves reasoning-based segmentation. Then, GLaMM [76] annotates a new dataset and introduces region-level captioning and segmentation tasks. Meanwhile, several recent models exploit

joint instruction tuning for referring segmentation and conversation [11, 100, 117]. Our method expands on these studies in the video domain by utilizing SAM-2 [77], while maintaining superior performance in both image/video referring tasks and conversation tasks.

Video Segmentation and Grounding. Existing video segmentation methods [40, 54, 123] mainly focus on segmenting and tracking pixels in closed sets, while a few approaches [31, 120] explore open-vocabulary settings. However, concepts are still limited compared to the knowledge space of LLMs. For video grounding, a recent method [34] explores LLM in the joint understanding of video and audio. On the other hand, VISA [100] and VideoLISA [4] also explore referring video segmentation. However, they lack comprehensive training, which restricts their capability in other tasks and scaling up. In contrast, Sa2VA enables fine-grained spatial-temporal modeling of both static (image) and dynamic (video) visual content, achieving strong performance across various tasks.

### 3 Method

#### 3.1 Unifying Multi-task Representations

Developing a unified model to tackle various image and video understanding tasks is challenging due to their differences in spatial and temporal information. To address this, we begin by re-examining the task formulations and proposing a unified representation,

|[Figure 57]<br><br>Image Caption<br><br>[Figure 58]<br><br>Video Caption<br><br>[Figure 59]<br><br>Image QA<br><br>[Figure 60]<br><br>Video QA|
|---|

[Figure 61]

Language Embd

Text Decoder

Tokenizer

Language Features

Text

Prompt Embd

Prompt Encoder

LoRA

###### [SEG] Token

LLM

Language Outputs

Visual Prompt

|[Figure 62]<br><br>[Figure 63]<br><br>Referring Image Segmentation|
|---|

[Figure 64]

[Figure 65]

Image Encoder

Image Embd

Image

[Figure 66]

Image Outputs

SAM2 Decoder

Image

Image Encoder

|[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>Referring Video Segmentation|
|---|

SAM2 Encoder

[Figure 71]

[Figure 72]

Video Embd

[Figure 73]

Video

Image Encoder

Visual Features

Video

Video Outputs

- Figure 2 Proposed Sa2VA model. The model first encodes the input texts, visual prompts, images, and videos into token embeddings. These tokens are then processed through a large language model (LLM). The output text tokens are used to generate the “[SEG]” token and associated language outputs. The SAM-2 decoder receives the image and video features from the SAM-2 encoder, along with the “[SEG]” token, to generate corresponding image and video masks. Modules with a fire icon are trained during the one-shot instruction-tuning.

which lays the groundwork for the development of Sa2VA.

tially unified models (different tasks utilize different model weights). In this work, we argue that all the aforementioned tasks can be unified into a one-shot instruction-tuning process, as we can leverage LLMs to manage various visual tokens. The overall process can be formulated as:

Referring Image/Video Object Segmentation. For image-referring segmentation, given input text tokens Ti ∈ RN×D, the model processes input images Ii ∈ RH×W×3 and produces corresponding binary masks Mo ∈ RH×W that align with the text description, where N and D are the number and dimension of text tokens. For video object referring segmentation, the model takes input videos Vi ∈ RT×H×W×3 and outputs binary spatio-temporal masks (masklets) Mo ∈ RT×H×W where T, H, and W are video frame number, height, and width.

To,Mo = LLM({Ii,Vi,V Pi},Ti). (1)

The input tokens vary for different tasks. For chatonly tasks, the model only outputs text tokens To. For referring segmentation tasks, the model outputs masks (image) or masklets (video) Mo. For grounded caption generation tasks, the model outputs both To and Mo. Since the model primarily processes images/videos, text, and masks as inputs and outputs, it is reasonable to unify these tasks within a single framework and train the model end-to-end. This approach enables a single model to support all the aforementioned tasks effectively.

Image/Video Chat and Grounded Caption Generation. For image and video chat tasks, given input text tokens Ti and the corresponding images Ii or videos Vi, the model produces the answer text To. For grounded caption generation tasks, the model simultaneously outputs the masks Mo and the aligned text To.

#### 3.2 Sa2VA Framework

Visual Prompt Understanding Tasks. For visual prompt understanding tasks, in addition to text tokens Ti and image Ii, the model also takes additional visual prompt tokens V Pi (boxes or points on the image) as inputs, outputting corresponding masks Mo and aligned text answers To.

The overall framework of Sa2VA is illustrated in Fig. 2. It consists of two components: the MLLM model and SAM-2.

Pre-trained MLLMs. We adopt pre-trained MLLMs, which contain one visual encoder, one visual projection layer, and one LLM. The visual encoder takes input images, videos, and sub-images and outputs visual features. The visual projection layer maps the

Unified Task Representation. Existing works address the above tasks through specific design models or par-

Algorithm 1: Ref-VOS Inference Pipeline

###### Object-level expression annotation

If {Inconsistent} or {Unrecognizable}

|[Figure 74]|
|---|

|Caption 1|
|---|

- 1 Input: Video length N; Number of key frames M;

Video frames SN (X1, X2, X3,..., XN); Language description T;

- 2 Output: Sequence of masks M1, M2, M3,..., MN;
- 3 Run: Sa2VA Model for Ref-VOS;
- 4 Extract key frames: SM ← SN;
- 5 Visual embeddings: Ev ← Encoder(SM);
- 6 Language embeddings: El ← Encoder(T);
- 7 Answers: A ← LLM({Ev,El});
- 8 Prompt embedding: Pl ← Linear(Find(A, ’[SEG]’));
- 9 for i = 1,2,...,M do

- 10 SAM-2 feature: Fi ← Encoder(X0);
- 11 Mask: Mi ← Decoder({Pl,Fi});
- 12 Update Memory: Mem ← Cross-Attention({Mem,Mi});

- 13 for i = M + 1,M + 2,...,N do

- 14 SAM-2 feature: Fi ← Encoder(X0);
- 15 Mask: Mi ← Decoder({Mem,Fi});
- 16 Update Memory: Mem ← Cross-Attention({Mem,Mi});

- 17 emit M1, M2, M3,..., MN;

[Figure 75]

Drop

[Figure 76]

Intern-VL2 76B

Masked Object

Qwen2 72B

Object-level Else description

|Caption 2| |
|---|---|
| | |

Cropped Object

###### Video-level expression annotation

###### Scene-level expression annotation

|Object-level description|
|---|

|Scene-level description|
|---|

|Video-level description|
|---|

|Scene-level description|
|---|

Intern-VL2 76B

Intern-VL2 76B

[Figure 77]

[Figure 78]

[Figure 79]

8 frames

Final expression: The object is a black goat wearing a blue … (Object-level) It is moving around in a natural outdoor environment … (Scene-level) Throughout the video, the goat can be seen walking across … (Video-level).

Figure 3 Data annotation pipeline. Our proposed automatic data annotation pipeline consists of three stages: object-level, scene-level, and video-level text expression annotation.

which are fed into SAM-2’s Decoder, where they are decoded into segmentation masks. The hidden states of “[SEG]” can be viewed as a novel spatialtemporal prompt for SAM-2. SAM-2 segments the relevant object mask in images and videos based on this spatial-temporal prompt. During training, the SAM-2 decoder can be adjusted to comprehend the spatial-temporal prompt, and gradients can be backpropagated through the “[SEG]” token to the MLLM, enabling it to gain knowledge from the training datasets.

features into tokens. These tokens, along with the input text tokens, serve as the input for LLMs, which generate text token predictions based on them. Note that we utilize pre-trained MLLMs, as seen in prior research [46, 76, 100, 114], to harness their powerful capabilities. For both image and video conversation tasks, we use the same pipeline as the original pretrained MLLMs (e.g., InternVL2 [14], Qwen-VL [2]).

Utilizing SAM-2 knowledge for mask tracking. For Ref-VOS tasks, we develop a straightforward framework to attain strong results on public benchmarks. Specifically, we initially extract key frames (the first several frames) from the video. These key frames are then processed through MLLM. This MLLM then generates a “[SEG]” token as the SAM-2 prompt to create the masks for the key frames. Next, we utilize the memory encoded by the key frame features to generate the mask for the remaining frames. We present the default inference algorithm in Algorithm 1.

Decoupled Design. We append SAM-2 [77] alongside the pre-trained MLLMs. Note that we do not take the SAM-2’s output tokens (visual features or decoder outputs) into LLM. There are three reasons behind this design choice. First, we aim to simplify the combination without incurring additional computation costs. Second, adding extra tokens will impede knowledge inheritance from MLLMs because MLLMs and SAM-2 are misaligned. Third, with this design, we can fully convert our work into a flexible framework that utilizes pre-trained, evolving MLLMs, as the MLLM community progresses rapidly.

#### 3.3 Ref-SAV Dataset and Benchmark

Data Annotation Pipeline. We create an automatic annotation pipeline to generate referring object text expressions for the SA-V dataset [77]. As illustrated in Fig. 3, the pipeline consists of three stages:

Tuning SAM-2 Decoder via ‘‘[SEG]’’ Tokens. Similar to previous works [46, 100] which utilize SAM, we employ the special token “[SEG]” to connect the decoder of SAM-2 with the MLLM. The hidden states of the “[SEG]” token serve as a new kind of prompt,

(1) Object-level annotation. We first select the frame with the largest object area from the video and mask out the non-object pixels. The cropped (without masking) and full (with masking) images are then fed separately into InternVL2-76B [14] to generate

detailed descriptions. The descriptions are processed by Qwen2-72B [101] for consistency checking, and conflicting descriptions are discarded. This method allows us to filter out error-prone cases. (2) Scenelevel annotation. Both the image and the object-level description from the previous stage are sent into InternVL2-76B [14] to generate a detailed object description that includes relationships to the scene and surrounding objects. (3) Video-level annotation. We sample 8 frames uniformly from the video, applying yellow borders to emphasize the object in each frame. These frames, paired with the scene-level description, are processed by InternVL2-76B [14] to generate a video-level description that captures the object’s movement and actions.

Ref-SAV Dataset. Using the above pipeline, we have automatically annotated detailed object expressions for the SA-V dataset, resulting in the creation of RefSAV. The training set of Ref-SAV contains 37,311 videos and 72,509 object expressions, which do not have any human labeling. For the Ref-SAV validation set, we select a subset of videos from the training set of the SA-V dataset to create the Ref-SAV evaluation benchmark, as the validation and test sets contain only a limited number of videos. These videos are completely separate from the Ref-SAV training dataset. The evaluation benchmark has two parts: 1) The long-expression set, generated using the automatic annotation pipeline mentioned above and carefully filtered by human annotators. 2) The shortexpression set, which is manually annotated. This evaluation benchmark includes 1,147 videos and 1,945 object expressions, comprising 1,694 long expressions and 251 short expressions.

#### 3.4 Sa2VA Training and Testing.

One For All Co-Training. Sa2VA is co-trained on multiple datasets. For VQA tasks, we utilize text regression loss Ltext, similar to common MLLMs. For segmentation tasks Lmask, we employ pixel-wise cross-entropy loss LCE and dice loss LDICE. It is important to note that there is no pre-training stage as in the base MLLMs [14, 61]; instead, we conduct supervised fine-tuning in one-shot training with the following loss:

Linstruction = Ltext+Lmask, Lmask = LCE+LDICE.

(2)

One For All Testing. All tasks can be encompassed within the Eq. 1 paradigm. During the inference stage, we encode the necessary task requirements, such as text prompts, visual prompts, image features,

Table 2 Sa2VA training datasets.

Type Datasets Image QA LLaVA 1.5 (665K) Image Seg RefCOCO (17K), RefCOCO+ (17K), RefCOCOg (22K), Grand-f (214K) Video QA Video-ChatGPT (100K) Video Seg Ref-YTVOS (3.5K), MeVIS (0.6K), ReVOS (1.7K), Ref-SAV (37K)

Table 3 Comparison with previous Ref-VOS benchmarks.

|Property|DAVIS17-RVOS ReVOS Ref-YT-VOS MeVIS Ours|
|---|---|
|Short Text<br><br>Long Text Large Object Motion<br><br>Large Camera Motion Heavy Occlusion<br><br>|✓ ✓ ✓ ✓ ✓<br><br>✗ ✓ ✗ ✗ ✓<br>✗ ✗ ✓ ✗ ✓<br><br>✓ ✓ ✗ ✗ ✓<br><br>✗ ✗ ✗ ✗ ✓<br>|

and video features, into tokens to input into the LLM. The output tokens from the LLM are then decoded into text responses (LLM prediction head), segmentation masks (SAM-2 decoder), and responses from the SAM-2 mask tracking module according to the task definition.

### 4 Experiments

Baseline. We construct the baseline by combining the SOTA public MLLM model InternVL2 [14] with SAM-2 [77]. Similar to previous works [46, 76, 114], the segmentation mask is obtained by decoding the hidden state of the “[SEG]” token through SAM2’s decoder. Inspired by Mask2Former-VIS [15], an object shares the same “[SEG]” token throughout the video, enabling our model to handle both image and video referring segmentation tasks in a unified manner. In addition, we further use more advanced MLLM models, InternVL2.5 [13], to demonstrate the model scaling effect.

Datasets and Metrics. We use four types of datasets to train Sa2VA, which include image QA, video QA, image segmentation, and video segmentation datasets. As shown in Tab. 2, Sa2VA’s training data includes approximately 1.1 million image-text or video-text pairs. Since InternVL2 has been trained with a large amount of image QA and video QA data, we only used 665K LLaVA 1.5 [65] and 100K Video-ChatGPT [68] data to prevent the MLLM from forgetting its image and video QA capabilities. We use 56K referring expression data [42, 105] and 214K grounding conversation generation data [76] for image-level textdriven segmentation. For video-level referring expression segmentation, we used 5.8K existing referring VOS data from Ref-YouTubeVOS [79], MeVIS [18], and ReVOS [100]. Additionally, we used 37K long text referring VOS data (Ref-SAV), generated by our proposed automatic annotation pipeline, to enhance

###### Table 4 Statistics comparison with other Ref-VOS datasets.

Datasets Video Object Expression Mask Avg. Len.

Ref-YTVOS 3,978 7,451 15,009 131k 9.68 MeViS 2,006 8,171 28,570 443k 7.07 ReVOS 1,042 9,084 35,074 273k 10.5

Ref-SAV 37,311 72,509 72,509 6.0m 83.6

###### Table 5 Training strategies for model components.

MLLM SAM-2 Encoder SAM-2 Decoder SAM-2 Memory Training Strategy LoRA Frozen Finetune Frozen

Sa2VA’s understanding of long referring text and its object grounding capabilities for complex videos. For image referring segmentation, we adopt cIoU since it balances the large and small objects. For referring video object segmentation, we adopt J&F. For image and video chat tasks, we follow previous works [56, 61] and report performance accordingly.

Implementation Details. We adopt the xtuner [17] framework for training. Our model inherits from MLLM models and does not require pre-training stages, which allows it to effectively leverage the information from the different base models. Unless specified otherwise, the following details describe the training of our default InternVL2 models. During the instruction tuning phase, we use the AdamW optimizer with an initial learning rate of 4e-5 and a weight decay of 0.05. We employ a 5% warmup ratio and apply gradient clipping with max gradient norm as 1.0. For LoRA parameters, we set the rank (r) to 256. The MLLM input image size is 448×448, and the SAM-2 input image size is 1024×1024. The maximum sequence length for the LLM is set to 8,192. We train for one epoch with a total batch size of 256 (16x2x8), achieved using a per-device batch size of 2 and 8 gradient accumulation steps. The training is conducted using bfloat16 (bf16) mixed precision on 16 NVIDIA H100 80GB GPUs. The instruction tuning stage lasts approximately 24 hours. The training strategy is listed in Tab. 5 We adopt VLMEvalKit [22] for the evaluation of chat tasks, and we have provided our scripts for segmentation tasks. For Ref-SAV testing, we adopt the original opensourced codebase [92, 99, 100] and model weights to infer the video results. By default, we use oneshot instruction tuning and do not fine-tune on any specific task to show the effectiveness of one-shot instruction tuning.

Ref-SAV Dataset. In Tab. 3, we further compare the existing Ref-VOS benchmarks from five different aspects: short text, long text, large object motion,

large camera motion, and heavy occlusion. The previous benchmarks only contain partial aspects of these five challenging cases. On the other hand, our benchmark is built from SAM-2 [77] and labeled with long text, which is more challenging than previous benchmarks. This is why the previous Ref-VOS models cannot achieve good results on our benchmark. Furthermore, as shown in Tab. 4, our dataset has more detailed captions (Avg. Len. refers to the average length of the referring description) and many more masks (about 10x). The proposed data can lead to better instruction-following capability for complex descriptions.

#### 4.1 Main Results

Comparison With SOTA MLLMs. As shown in Tab. 6, Sa2V-8B achieves cIoU scores of 81.9, 76.5, and 78.9 on RefCOCO, RefCOCO+, and RefCOCOg, respectively, surpassing GLaMM-7B by 2.4, 3.9, and 4.7 cIoU. Sa2VA performs favorably against the state-ofthe-art methods on RefCOCO+ and RefCOCOg, significantly outperforming existing grounding MLLMs, including LISA, GLaMM, PixelLLM, PSALM, and OMG-LLaVA. Additionally, Sa2VA demonstrates strong conversational capabilities, achieving scores of 2229 (1651/578), 82.4, and 75.5 on MME, MMbench, and SEED-Bench, while existing grounding MLLMs perform poorly in conversation due to catastrophic forgetting. Sa2VA achieves performance comparable to that of InternVL2 in image QA benchmarks, indicating that Sa2VA largely maintains the chat performance of the base MLLM.

Sa2VA also performs well on video benchmarks. It achieves scores of 46.9, 75.2, and 57.6 J&F on MeVIS, Ref-DAVIS17, and ReVOS, respectively, surpassing the previous SOTA VISA-13B by 2.4, 4.8, and 6.7 J&F. Additionally, Sa2VA-8B earns a score of 1.34 on the video QA benchmark MMBench-Video, outperforming InternVL2-8B’s 1.28. Sa2VA does not achieve dominant results on video chat tasks, as it has both video segmentation and understanding capabilities, which current methods are not equipped to handle simultaneously. The main experimental results indicate that our Sa2VA is a versatile and powerful MLLM.

Detailed Results on Image Benchmarks. As indicated by [46, 95, 114], performing instruction tuning negatively impacts performance drastically on chat tasks. We utilize co-training to address this issue. In Tab. 7, we also present additional results on image chat datasets. Our model maintains good results across multiple image chat benchmarks while

- Table 6 Experiment results on various image/video benchmarks. We report cIoU for the image segmentation sets, J&F for the video segmentation sets, and AP50 for the GCG benchmark. For MME dataset [25], A / B denotes the perception (A) and cognition (B) scores, respectively, while C(+) represents the total score (C = A + B). Although it is more common to report perception and cognition separately, we report the sum as in their original paper if the individual scores are missing.

Method

Image Segmentation Video Segmentation Image Chat Video Chat GCG

RefCOCO [42] RefCOCO+ [42] RefCOCOg [105] MeViS [18] Ref-DAVIS17 [44] Ref-YTVOS [79] ReVOS [100] MME [25] MMBench [66] SEED-Bench [48] Video-MME [26] MMBench-Video [23] GCG [76]

LLAVA-1.5-13B [62] - - - - - - - 1531(+) 68.8 70.1 - - Video-LLaVA-7B [58] - - - - - - - - 60.9 - 39.9 1.03 LLaMA-VID-7B [56] - - - - - - - 1521(+) 65.1 59.9 - 1.08 -

mPLUG-Owl3-8B [104] - - - - - - - - 77.6 - 53.5 1.35 InternVL2-8B [14] - - - - - - - - 81.7 76.2 54.0 1.28 -

PixelLM-7B [78] 73.0 66.3 69.3 - - - - 309/135 17.4 - - - LaSagnA [86] 76.8 66.4 70.6 - - - - 0/0 0.0 - - - LISA-7B [46] 74.1 62.4 66.4 - - - - 1/1 0.4 - - - GLaMM-7B [76] 79.5 72.6 74.2 - - - - 14/9 36.8 - - - 28.9 LLaVA-G-7B [111] 77.1 68.8 71.5 - - - - - - - - - -

GSVA-13B [96] 79.2 70.3 75.7 - - - - - - - - - -

OMG-LLaVA-7B [114] 78.0 69.1 72.9 - - - - 1177/235 47.9 56.5 - - 29.9 VideoLISA-3.8B [4] 73.8 63.4 68.3 44.4 68.8 63.7 - - - - - - -

VISA-13B [100] 72.4 59.8 65.5 44.5 70.4 63.0 50.9 - - - - - -

Sa2VA-1B (Ours) 77.4 69.9 72.3 41.7 72.3 65.3 47.6 1381/405 68.3 64.8 39.9 1.07 23.8 Sa2VA-4B (Ours) 80.4 74.3 76.7 46.2 73.8 70.0 53.2 1553/540 76.8 72.6 50.4 1.23 28.2 Sa2VA-8B (Ours) 81.9 76.5 78.9 46.9 75.2 70.7 57.6 1651/578 82.4 75.5 52.1 1.34 31.0

Sa2VA-26B (Ours) 82.5 78.8 79.7 46.2 77.0 70.1 58.4 1691/538 83.7 76.8 52.6 1.45 33.5

- Table 7 Performance on image-level benchmarks with MLLMs that have segmentation capability. We report cIoU for the image segmentation sets. A / B denotes the perception (A) and cognition (B) scores, respectively, while C(+) represents the total score (C = A + B).

Method MME [25] MMBench [66] SEED-Bench [48] AI2D [43] MMStar [10] MMMU [108] SQAtest [67] RefCOCO RefCOCO+ RefCOCOg LLAVA-1.5-13B [62] 1531(+) 68.8 70.1 - - - - 0.0 0.0 0.0

LISA-7B [46] 1/1 0.4 - 0.0 - - - 74.1 62.4 66.4 PixelLM-7B [78] 309/135 17.4 - 0.0 - - - 73.0 66.3 69.3 LaSagnA-7B [86] 0/0 0.0 - 0.0 - - - 76.8 66.4 70.6 GLaMM-7B [76] 14/9 36.8 - 28.2 - - - 79.5 72.6 74.2

OMG-LLaVA-7B [114] 1177/235 47.9 56.5 42.9 - - - 78.0 69.1 72.9

Sa2VA-4B (ours) 1553/540 76.8 72.6 79.9 53.7 46.2 95.8 80.4 74.3 76.7 Sa2VA-8B (ours) 1651/578 82.4 75.5 82.1 60.3 44.7 96.8 81.9 76.5 78.9

###### Table 8 Experiment results using stronger InternVL2.5 in our Sa2VA.

Base Image Segmentation Video Segmentation Image Chat MLLM RefCOCO RefCOCO+ RefCOCOg MeViS (val_u) Ref-YTVOS Ref-DAVIS17 MME MMBench SEED-Bench AI2D MMStar SQAtest

|InternVL2.0-4B InternVL2.0-8B<br><br>|80.4 74.3 76.7<br><br>81.9 76.5 78.9<br><br><br>|52.1 70.0 73.8 57.0 70.7 75.2<br><br>|1553/540 76.8 72.6 79.9 53.7 95.8 1651/578 82.4 75.5 82.1 60.3 96.8|
|---|---|---|---|
|InternVL2.5-1B InternVL2.5-4B InternVL2.5-8B InternVL2.5-26B|79.6 73.6 77.7 82.4 77.6 79.7 82.6 78.0 80.3 82.9 79.3 81.2<br><br>|53.4 68.0 69.5 55.9 71.3 73.7 58.9 72.3 75.9 61.8 75.1 78.6<br><br>|1504/434 71.9 71.0 69.2 48.6 89.9 1691/610 81.8 74.9 81.4 57.9 96.8 1690/610 84.4 76.5 82.7 62.4 97.4 1698/653 85.8 78.3 85.7 67.0 98.4|

achieving strong referring segmentation performance, demonstrating its versatility.

Results on Ref-SAV validation set. In Tab. 10, we benchmark several state-of-the-art Ref-VOS models using our proposed Ref-SAV benchmark. We find that even the foundation model [99] and the recent video MLLM model [100] fail to achieve strong results on our benchmarks. This is due to our benchmark featuring more occlusions, longer text descriptions, and diverse annotations from SAM-2 [77]. Conversely, our method, Sa2VA, whether with or without the Ref-SAV training set, can yield strong results. Our method can be further enhanced with our training set, indicating that our Ref-SAV training set serves

as a valuable supplement to the video understanding community.

Comparison with specific fine-tuned methods. The original Sa2VA setting is more challenging because it requires the use of the same model weights to complete all tasks. For fair comparison, similar to prior works [76, 114], we evaluate with MLLMs on specific fine-tuned datasets. As shown in Tab. 11, we find 1.5% improvement across three datasets compared to the co-training model, with Sa2VA performing best among recent methods. From another perspective, fine-tuning on a single dataset does not yield much benefit (0.4 on RefCOCO). However, this requires fine-tuning each dataset and producing a new model.

###### Table 9 Ablation study on co-training effect on multiple datasets. We use Sa2VA-1B to test the performance.

Image Segmentation Video Segmentation Image Chat Video Chat

Data

RefCOCO RefCOCO+ RefCOCOg MeViS (val_u) Ref-DAVIS17 MME MMBench SEED-Bench Video-MME MMBench-Video

All Data 77.4 69.9 72.3 50.8 72.3 1381/405 68.3 64.8 39.9 1.07 w/o Image QA 78.0 70.1 72.2 48.3 73.0 1298/359 63.4 63.8 39.7 0.39

w/o Image Segmentation 20.2 20.6 23.2 38.0 48.8 1393/408 70.1 65.7 41.2 1.08

w/o Video QA 78.0 70.4 72.6 50.7 74.3 1370/402 69.1 65.0 41.3 0.71 w/o Video Segmentation 77.4 69.1 72.4 44.4 69.0 1403/398 67.8 64.9 40.4 1.04

###### Table 10 Ref-SAV validation sets. zs: zero-shot testing.

Long Short Overall J F J&F J F J&F J F J&F

Method

UniRef++ [92] (zs) 14.1 10.8 12.5 9.0 8.2 8.6 11.6 9.5 10.5 UNINEXT [99] (zs) 11.7 8.3 10.0 5.8 4.4 5.1 8.8 6.4 7.6 MeVIS [18] (zs) 12.1 7.1 11.3 6.2 5.3 5.5 12.2 9.8 10.3 VISA [100] (zs) 16.1 12.2 14.1 12.3 9.6 9.2 13.2 11.3 11.8

Sa2VA-8b (zs) 47.7 50.9 49.3 31.5 35.0 33.3 39.6 43.0 41.3 Sa2VA-8b (Ours) 57.0 60.4 58.7 39.5 42.9 41.2 48.3 51.7 50.0

###### Table 11 Comparison with Fine-tuned Models.

Model Type RefCOCO RefCOCO+ RefCOCOg LAVT [103] 72.7 62.1 61.2

GlaMM-7B [76] 79.5 72.6 74.2 OMG-LLaVA-7B [114] 78.0 69.1 72.9

F-LLM-7B [95] 76.1 65.2 68.5 Sa2VA-8B (ours) 81.9 76.5 78.9

Sa2VA-8B (ft) 82.3 77.3 79.3

Therefore, we advocate for more general models like Sa2VA for their versatility and convenience.

InternVL-2.5 Results. We further conduct ablation experiments to evaluate the scalability of Sa2VA with stronger MLLMs and additional datasets. As shown in Tab. 8, upgrading the MLLM from InternVL2 [14] to InternVL2.5 [13] consistently improves performance across image, video, and multimodal benchmarks.

#### 4.2 Ablation Studies

Effectiveness of Joint Co-training. The performance of Sa2VA on image and video QA and segmentation benchmarks can be attributed to unified instruction tuning and joint co-training. We perform ablation studies on the training datasets, with the results presented in Tab. 9. When trained without image QA datasets, Sa2VA’s scores on MME and MMBench drop by 129 and 4.9, respectively. Without image segmentation datasets, Sa2VA achieves only 20.2, 20.6, and 23.2 cIoU on RefCOCO, RefCOCO+, and RefCOCOg, while performance on the video segmentation benchmark drops significantly. Sa2VA on MMBench-Video decreases by 34% when trained without video QA datasets. When training without video segmentation datasets, Sa2VA’s performance on

Table 12 Ablation study on “[SEG]” token design.

Type RefCOCO RefCOCO+ RefCOCOg DAVIS MeVIS(val_u) Single 77.4 69.9 72.3 72.3 50.8

Repetitive 77.3 70.2 72.5 71.1 49.6 Unique 77.6 70.3 72.4 68.6 46.3

###### Table 13 Ablation study on using more datasets.

Dataset Size RefCOCO RefCOCOg MMBench MME MeVIS(val_u) baseline 1.2M 77.4 72.3 68.3 1381/405 50.8

Inifinity-MM [28] 1.2M+3M 77.1(-0.3) 72.6(+0.3) 70.4(+2.1) 1396/346(-44) 51.2(+0.4) Ref-SAV 1.2M+37K 77.2(-0.2) 72.6(+0.3) 68.2(-0.1) 1384/418(+16) 52.5(+1.7)

MeVIS and Ref-DAVIS17 drops by 4.4 and 3.3 J&F, respectively. The ablation results indicate that joint co-training with these four types of datasets is critical to Sa2VA’s performance across the various tasks. In Tab. 13, adding 3M image-QA samples from InfinityMM [28] yields a notable +2.1 gain on MMBench, with negligible changes in segmentation accuracy. When incorporating our Ref-SAV dataset, Sa2VA-1B achieves a +1.7 J&F improvement on MeViS while maintaining image-level performance. These results demonstrate that Sa2VA benefits synergistically from both model scaling and data expansion, indicating substantial headroom for further improvement.

Ablation study on the Segmentation Token Design.

Tab. 12 shows the results of our ablation on the segmentation token design. We analyze two primary alternative strategies based on whether to use the same special token or different special tokens for different frames. First, we test the repetitive strategy, which requires the LLM to output the same “[SEG]” token N times for N frames. This approach slightly reduces video segmentation performance. Second, we analyze using unique, frame-specific tokens (e.g., “[SEG 1]” for frame 1, “[SEG 2]” for frame 2, and so on). This design caused a larger performance drop. The reason is that it prevents knowledge sharing with the image segmentation task. The model is typically pre-trained to associate a single, generic “[SEG]” token with the concept of segmentation. By introducing new tokens (“[SEG 1]”, ..., “[SEG n]”), there is a mismatch between images and videos that blocks this knowledge transfer, forcing the model to

###### Table 14 Experiment results of Sa2VA with recent base MLLMs.

Base Image Segmentation Video Segmentation Image Chat

MLLM RefCOCO refCOCO+ refCOCOg MeViS (val_u) ReVOS Ref-DAVIS17 MME MMBench SEED-Bench AI2D MMStar SQAtest Sa2VA-8B 81.9 76.5 78.9 57.0 57.6 75.2 1651/578 82.4 75.5 82.1 60.3 96.8

|Sa2VA-InternVL3-2B Sa2VA-InternVL3-8B Sa2VA-InternVL3-14B<br><br>|81.4 75.7 80.3 83.3 78.9 81.8 83.6 79.9 83.6|53.9 56.2 74.5 56.4 60.8 76.3 59.2 60.7 76.6<br><br>|1631/559 79.8 73.9 77.1 59.2 93.7 1743/633 83.0 76.2 84.3 65.9 97.5 1746/724 84.3 76.6 85.2 67.4 98.7|
|---|---|---|---|
|Sa2VA-Qwen2.5VL-3B Sa2VA-Qwen2.5VL-7B Sa2VA-Qwen3VL-4B<br><br>|79.6 74.0 77.1 82.4 77.5 81.5 81.7 77.4 80.0|51.6 52.0 73.4<br><br>56.4 58.3 79.4<br><br>57.1 56.7 75.9<br><br><br>|1533/572 78.4 73.9 81.1 57.7 80.3 1552/676 84.5 75.0 84.5 62.3 87.9 1660/655 86.3 77.3 85.4 66.3 91.6|

###### Table 15 Comparison with Recent Video MLLMs.

Model Type MeViS ReVOS Ref-DAVIS17

PG-Video-LLaVA [69] 18.9 - GLaMM + SAM2 [70] 38.7 - -

VideoGLaMM (3.8B) [70] 45.2 - -

VISA-13B [100] 44.5 50.9 70.4 VideoLISA-3.8B [4] 44.4 - 68.8

HyperSeg-3B [88] - 55.7 71.2 InstructSeg [87] - 54.5 71.1

Sa2VA-4B (ours) 46.2 53.2 73.8 Sa2VA-8B (ours) 46.9 57.6 75.2

###### Table 16 Comparison with Vision Expert Models.

Model Type RefCOCO RefCOCO+ RefCOCOg MeVIS Ref-DAVIS17 LAVT [103] 72.7 62.1 61.2 - ReferFormer [91] - - - 31.0 61.1 UniRef++-L [92] 81.4 74.0 76.0 - 67.2

EVF-SAM [118] 82.4 76.5 78.2 - LMPM [18] - - - 37.2 UniVS [53] - - - - 59.4

Sa2VA-26B (ours) 82.5 78.8 79.7 46.2 77.0

learn the meaning of each new segmentation token independently.

#### 4.3 More Comparison Results

Comparison with recent video MLLM models on referral video segmentation Recently, several works have also combined MLLMs into video referring segmentation tasks. As shown in Tab. 15, we list and compare our models against several recent methods on three datasets. These competitors include VideoGLaMM [70], VideoLISA [4], HyperSeg [88], InstructSeg [87], and etc. We also include the combined strong baseline (SAM2 [77] + GlaMM [76]) reported in [70]. As shown in Tab. 15, our Sa2VA-8B model achieves strong performance across all three datasets. Our Sa2VA-4B model also demonstrates strong competitive performance, achieving the highest score on MeViS, while maintaining the ability for image/video chat, GCG, and visual prompt understanding. At the same time, other referring video segmentation methods only focus on this task, and due to the lack of training data for other tasks, the obtained model

- Table 17 Vision Expert on Ref-SAV validation sets.

Method

Long Short Overall J F J&F J F J&F J F J&F UniRef++ [92] (zero-shot) 14.1 10.8 12.5 9.0 8.2 8.6 11.6 9.5 10.5 UniRef++ (fine-tuning) 19.2 15.1 17.2 12.3 11.7 12.0 15.8 13.4 14.6

- Table 18 Region caption performance on RefCOCOg dataset.

|Method<br><br>|Sa2VA-4B<br><br>|OMG-LLaVA [114] Osprey [107] GLaMM [76] GRIT [90] Kosmos-2 [73]|
|---|---|---|
|METEOR<br><br>|17.3<br><br>|15.3 16.6 16.2 15.2 14.1|

will perform poorly on other tasks such as video or image chat.

Comparison with vision expert models. As shown

- in Tab. 16, we compare our model with recent vision expert models designed specifically for referring segmentation. Although these specialist models are typically more lightweight, our general-purpose 26B model (Sa2VA-26B) still achieves the strongest results, outperforming them across all five datasets. This is significant, as these expert models lack conversational capabilities and are mostly limited to a single modality (video or image), whereas our model is a generalist.

Effectiveness of training dataset on more methods. We further demonstrate the effectiveness of our proposed Ref-SAV training dataset by evaluating a representative method, UniRef++ [92]. As shown

- in Tab. 17, we compare two distinct settings to isolate the impact of our data. The zero-shot setting involves directly testing the pre-trained UniRef++ model on the Ref-SAV validation dataset without any new training. In contrast, the fine-tuning setting first trains the model on our proposed Ref-SAV training dataset before evaluation on the same validation set. The results clearly show that the model fine-tuned on our dataset achieves a significant improvement across all metrics. Most notably, the Overall J&F score increases substantially from 10.5 to 14.6. This considerable gain, which is consistent across both

###### Table 19 Computational cost comparision.

Model Params Base MLLM Base Language Model GPU Inference Time

Sa2VA-1B 1B InternVL2.5-1B Qwen/Qwen2.5-0.5B-Instruct NVIDIA-H100 0.123s Sa2VA-4B 4B InternVL2.5-4B Qwen/Qwen2.5-3B-Instruct NVIDIA-H100 0.282s Sa2VA-8B 8B InternVL2.5-8B internlm/internlm2_5-7b-chat NVIDIA-H100 0.201s Sa2VA-26B 26B InternVL2.5-26B internlm/internlm2_5-20b-chat NVIDIA-H100 0.463s

“Long” (12.5 to 17.2 J&F) and “Short” (8.6 to 12.0 J&F) validation splits, indicates that our automatic data engine has great potential for boosting the performance of Ref-VOS models.

Results on visual prompts understanding tasks. We also report on visual prompt understanding tasks, following previous works [107, 114]. Specifically, we evaluate region caption performance on the RefCOCOg dataset. As shown in Tab. 18, our method, Sa2VA4B, also achieves the best results among recent visual prompt understanding models. It scores 17.3 on the METEOR metric, significantly outperforming the next-best method, Osprey [37], which scored 16.6. This indicates that Sa2VA can also generate strong region-aware captions.

#### 4.4 Discussions

Sa2VA with recent MLLM models. Table 14 presents a comprehensive evaluation of our Sa2VA framework integrated with different recent base Multimodal Large Language Models (MLLMs), including Qwen2.5-VL [3], Qwen3-VL [97], and InternVL3 [124]. The results clearly demonstrate that the choice of the base MLLM significantly influences performance, and this effect varies across different tasks. For instance, while the Sa2VA-InternVL3-14B model achieves the strongest overall performance, particularly on image and video segmentation benchmarks like RefCOCO (83.6) and MeViS (59.2), other models exhibit competitive strengths in specific domains. Notably, the Sa2VA-Qwen3VL-4B model records the highest score on MMBench (86.3) and shows highly competitive results on other image chat datasets. This variation highlights that different base MLLM architectures possess distinct capabilities, making model selection a critical factor for task-specific optimization. To facilitate further research and allow the community to build upon these findings, we have publicly released all trained model checkpoints.

Discussion on the inference speed. The computational overhead in Sa2VA is dominated by the MLLM component due to its auto-regressive nature, where each inference step generates tokens sequentially. In contrast, the primary additional module, SAM-2, is relatively lightweight at 220M parameters (compared to the MLLMs, which range from 1B to 26B) and

Table 20 Ablation study on keyframe sampling strategies using Sa2VA-8B on the MeVIS (val_u) dataset.

Sampling Strategy MeVIS J&F (val_u) Inference Time (per sample) # Image Tokens

First 1 Frame 55.1 0.151s 256

- First 3 Frames 58.7 0.167s 768
- First 4 Frames 59.5 0.181s 1024
- First 5 Frames 58.9 0.207s 1280 Uniform 5 Frames 62.9 0.207s 1280

efficient, achieving 39.5 FPS on vision tasks with only a single forward pass per image. While the variable output length of the MLLM makes precise latency estimation challenging, to provide a more concrete comparison, we also construct a controlled benchmark with fixed frame and language lengths (“Please segment the man.” as the text prompt with 5 frames and totaling 1280 tokens) to estimate computational cost more systematically. The results of this benchmark are detailed in Table 19. The results show inference times of 0.123s for Sa2VA-1B, 0.282s for Sa2VA-4B, 0.201s for Sa2VA-8B, and 0.463s for Sa2VA-26B. We note that inference speed is largely dependent on optimization, and the discrepancy between the 4B model (0.282s) and the 8B model (0.201s) may be due to the different base LLMs (Qwen vs. InternLM) employed. Nonetheless, these results confirm that the majority of the model’s computational cost stems from the auto-regressive MLLM component rather than the efficient, single-pass SAM-2 module.

Inference Strategy. We also analyzed the impact of our keyframe sampling strategy. Our default approach of using the first five frames was chosen for simplicity and consistency, but we investigated whether this method limits the model’s ability to capture long-range temporal dependencies or objects that appear later in the video. To test this, we performed an ablation study on the MeVIS dataset using the Sa2VA-8B model. We compared the performance of selecting the first 1, 3, 4, and 5 consecutive frames against a uniform sampling strategy that selects 5 frames spread across the entire video. The results of this comparison are detailed in Table 20. As the table shows, performance generally improves when increasing the number of consecutive frames, moving from 55.1 J&F for a single frame to 59.5 J&F for four frames. However, the “First 5 Frames” strategy (58.9 J&F) is significantly outperformed by the “Uniform 5 Frames” strategy, which achieves 62.9 J&F. Notably, both 5-frame methods have an identical inference time (0.207s) and process the same number of image tokens (1280). This finding suggests that more sophisticated, interval-based sampling strategies improve performance. However, we would like to keep Sa2VA as simple as possible and leave designing advanced

techniques on sampling as future work.

#### 4.5 Visualizations

Results on image referring segmentation. As shown in Fig. 5, we visualize the image referring segmentation task. With different language descriptions, our Sa2VA can segment different masks with the clues from the description.

Results on video referring segmentation. As shown in Fig. 6, our method performs well in handling diverse and challenging conditions, showcasing a remarkable ability to adjust and perform effectively even in occlusion scenes or highly dynamic environments.

Results on visual prompt understanding. As shown in Fig. 7, our method can generate descriptions based on the visual prompts. The descriptions generated by our method demonstrate a high degree of contextual awareness, effectively capturing details within the visual cues.

Results on GCG task. In Fig. 4, we further show several visual examples in grounded caption generation tasks. Compared with the previous SOTA model, OMG-LLaVA, our model achieves better results in both mask quality and text-mask alignments. The former indicates better-aligned segmentation outputs, while the latter shows the latter indicates good textto-region alignments.

Video demo. Since dynamic visual content such as videos cannot be effectively presented within the static format of the manuscript, we have provided additional video demos in the supplementary material. These videos are intended to offer a clearer and more intuitive understanding of our method’s performance. Please refer to the ZIP file for the demos.

### 5 Failure Cases and Future Work

Failure cases of Sa2VA and Future work. Despite Sa2VA achieving stronger results on various ref-segmentation and video ref-segmentation datasets. However, there are still several cases of failure that need to be explored. We identify two shortcomings of Sa2VA in our future work. One is a long video with hard, distinguished referring examples, as shown in Fig 8. This is because our Sa2VA works in online mode without knowing the entire video content to align the long, complex text. The other is how to improve the VQA tasks (in image and video) without hurting the referring segmentation tasks. As shown in the ablation table of the main paper, increasing the scale of VAQ data leads to a performance drop

in referring segmentation tasks. Thus, how to balance both types and keep the grounded prediction knowledge is still unsolved.

Future directions on Ref-SAV benchmarks. As discussed in Tab. 3, long text, occlusion inputs, and longer videos with camera motion are critical problems in our benchmarks. Thus, the new models need to design a more robust Ref-VOS system to handle these challenges with several specific designs, including more robust memory designs. In addition, understanding long text and improving long text grounding ability is also important to explore.

### 6 Conclusion

In this work, we present Sa2VA, a versatile framework that integrates SAM-2 with LLaVA-like MLLMs to achieve a dense, grounded understanding of both images and video. Our method can handle various image and video understanding tasks, including referring image/video segmentation and image/video conversation, with just one-shot instruction tuning. By leveraging the knowledge from both LLaVA and SAM-2, our model has strong capabilities in both mask and language generation. To demonstrate the effectiveness of our proposed method, we propose a challenging referring video object segmentation benchmark, Ref-SAV. Extensive experimental results show that Sa2VA achieves strong performance on various tasks across benchmarks. In addition, Sa2VA can also be extended with various modern VLMs, with potential for building stronger baselines for pixel multi-modal systems.

Boarder Impact. Our works provide the first opensource leading pixel understanding systems. Since it was put on arixv (2025-01) with opensourced code, Sa2VA has been widely used in ICCV and CVPR workshop competations [19, 60], which serves as a strong baseline for nearly all winning teams solutions.

### References

- [1] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In ICCV, 2015.

- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

A person is seen playing with a skateboard. The skateboard is positioned over the grass. The sky is visible over the person, house, and tree.

[Figure 80]

[Figure 81]

[Figure 82]

A person is riding a motorcycle on a road. The

A man, referred to as person, is seen riding a

motorcycle is beside a truck. The truck is also driving on the road.

motorcycle. The motorcycle is on a road. The man is wearing a helmet.

Sa2VAOMG-LLaVA

[Figure 83]

[Figure 84]

[Figure 85]

A giraffe is seen walking on dirt. The sky is visible over the giraffe, trees, and grass. The grass is attached to the dirt.

A man in a white shirt is holding a tennis racket.

A woman, referred to as person, is standing on the sand at the beach. She is holding a kite. The sea is beside her, with the sky overhead.

[Figure 86]

[Figure 87]

[Figure 88]

A man riding a motorcycle down a road .

A boy is jumping off a roof .

A man in a black helmet is riding a motorcycle

[Figure 89]

[Figure 90]

[Figure 91]

A man, referred to as person is holding a tennis racket on a tennis court.

A giraffe is standing on the grass in a field. The sky is visible over the giraffe and the grass.

A woman is standing on the beach holding a kite .

Figure 4 Visualization results on GCG tasks. Top: our method. Bottom: OMG-LLaVA [114]. Note that, our method has stronger and fined-grained grounding ability and text alignment than OMG-LLaVA [114], previous strong baseline.

- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [4] Zechen Bai, Tong He, Haiyang Mei, Pichao Wang, Ziteng Gao, Joya Chen, Lei Liu, Zheng Zhang, and Mike Zheng Shou. One token to seg them all: Language instructed reasoning segmentation in videos. NeurIPS, 2024.

- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020.

- [6] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Making large multimodal models understand arbitrary visual prompts. In CVPR, 2024.

- [7] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV, 2020.

- [8] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In ICCV, 2023.

- [9] Jierun Chen, Fangyun Wei, Jinjing Zhao, Sizhe Song, Bohuai Wu, Zhuoxuan Peng, S-H Gary

- Chan, and Hongyang Zhang. Revisiting referring expression comprehension evaluation in the era of large multimodal models. arXiv preprint arXiv:2406.16866, 2024.
- [10] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024.

- [11] Wei-Ge Chen, Irina Spiridonova, Jianwei Yang, Jianfeng Gao, and Chunyuan Li. Llava-interactive: An all-in-one demo for image chat, segmentation, generation and editing. arXiv preprint arXiv:2311.00571, 2023.

- [12] Xinghao Chen, Siwei Li, Yijing Yang, and Yunhe Wang. Deco: Unleashing the potential of convnets for query-based detection and segmentation. In ICLR, 2025.

- [13] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

- [14] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal

[Figure 92]

[Figure 93]

Figure 5 Visualization results on image referring segmentation task.

models with open-source suites. arXiv preprint arXiv:2404.16821, 2024.

- [20] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2: Mastering freeform text-image composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024.

- [21] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. In NeurIPS, 2024.

- [22] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating

- [15] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Maskedattention mask transformer for universal image segmentation. In CVPR, 2022.

- [16] Ho Kei Cheng, Seoung Wug Oh, Brian Price, JoonYoung Lee, and Alexander Schwing. Putting the object back into video object segmentation. In CVPR, 2024.

- [17] XTuner Contributors. Xtuner: A toolkit for efficiently fine-tuning llm. https://github.com/ InternLM/xtuner, 2023.
- [18] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. Mevis: A large-scale benchmark for video segmentation with motion expressions. In ICCV, 2023.

- [19] Henghui Ding, Chang Liu, Nikhila Ravi, Shuting He, Yunchao Wei, Song Bai, and Philip Torr. Pvuw 2025 challenge report: Advances in pixel-level understanding of complex videos in the wild. In CVPR workshop, 2025.

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

Please segment the object is a dark-colored backpack with light-colored accents, featuring multiple compartments and pockets, securely fastened to an individual's back. The person is dressed in dark clothing and ascending an escalator in a public setting, likely a mall or transportation hub. The backpack has adjustable straps and a top handle, appearing functional for carrying various items. The individual moves steadily up the escalator, indicating a purposeful journey.

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Please segment a white dog with black patches.

Figure 6 Visualization results on video referring segmentation.

large multi-modality models. In ACMMM, 2024.

- [23] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. arXiv preprint arXiv:2406.14515, 2024.

- [24] Hao Fei, Shengqiong Wu, Hanwang Zhang, TatSeng Chua, and Shuicheng Yan. Vitron: A unified pixel-level vision llm for understanding, generating, segmenting, editing. In NeurIPS, 2024.

- [25] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

- [26] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-

- mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.
- [27] Brent Griffin, Victoria Florence, and Jason Corso. Video object segmentation-based visual servo control and object depth estimation on a mobile robot. In WACV, 2020.

- [28] Shuhao Gu, Jialing Zhang, Siyuan Zhou, Kevin Yu, Zhaohu Xing, Liangdong Wang, Zhou Cao, Jintao Jia, Zhuoyi Zhang, Yixuan Wang, et al. Infinity-mm: Scaling multimodal performance with large-scale and high-quality instruction data. arXiv preprint arXiv:2410.18558, 2024.

- [29] Tianrui Guan, Divya Kothandaraman, Rohan Chandra, Adarsh Jagan Sathyamoorthy, Kasun Weerakoon, and Dinesh Manocha. Ga-nav: Efficient terrain segmentation for robot navigation in unstructured outdoor environments. RA-L, 2022.

- [30] Tianrui Guan, Ruitao Song, Zhixian Ye, and

[Figure 116]

- The region marked by region1 is a person wearing a green jacket. They are the main focus of the image, sitting in a canoe and enjoying the serene surroundings. The person is wearing a life jacket, which adds to the safety precautions taken during the canoeing activity.

The region marked as region3 is the bottom of the boat. It is the lower part of the boat where the person is sitting. The boat is a canoe, and the person is wearing a life jacket while sitting in it. The region captures the perspective of the person from the bottom of the boat, providing a view of the surroundings.

The region marked by region4 is the body of water in the image. It is a calm and serene lake, with the water appearing blue in color. The lake is surrounded by a beautiful landscape, including a forest of trees and a mountain in the background. The water is clear and reflects the sky, creating a peaceful and picturesque scene.

The region marked by region5 is a dense forest of trees. It is a lush green area with tall trees, creating a picturesque and serene environment. The trees are surrounded by a calm lake, adding to the tranquility of the scene. The forest provides a natural backdrop to the person paddling the canoe, creating a harmonious connection between nature and human activity.

The region marked by region6 is a mountain in the background of the image. It is a large mountain with a snowy peak, adding a majestic and picturesque element to the scene. The mountain stands tall and imposing, creating a sense of grandeur and natural beauty.

[Figure 117]

- The region marked by region2 is a dirt road. It is a narrow pathway that runs through the grassy area. The road is surrounded by tall grass and trees, creating a natural and serene environment. The dirt road adds a sense of adventure and exploration to the scene.

- The region marked by region4 is the sky in the image. It is the background element that spans the entire width of the image. The sky is depicted as a clear blue color, providing a contrasting backdrop to the other elements in the scene.

The region marked by region3 is a tree located in the middle of the image. It is a large tree with a green canopy and a brown trunk. The tree stands out against the backdrop of the grassy field. It is positioned towards the left side of the image.

The region marked by region1 is a giraffe. It is a tall animal with a long neck and legs. The giraffe is walking down a dirt road, and its presence adds a sense of majesty to the scene. The giraffe is the main focus of this region, and its unique appearance stands out against the backdrop of the surrounding landscape.

[Figure 118]

The region marked by region1 is a person riding a motorcycle. The person is wearing a red shirt and is positioned in the middle of the image. They are riding the motorcycle on the road, which is a two-lane road. The person is the main focus of this region, and their presence adds to the overall scene of the image.

The region marked by region2 is a motorcycle. It is a black motorcycle with a person riding it. The motorcycle is positioned on the right side of the image, and the rider is wearing a helmet. The motorcycle is moving along the road, and the rider is actively engaged in riding it.

The region marked by region3 is the road itself. It is a paved road that the person on the motorcycle is riding on. The road is surrounded by trees and vegetation, creating a scenic environment. The person on the motorcycle is riding down this road, enjoying the ride.

- The region marked by region5 is a blue truck. It is a large truck with a blue color and a yellow box on the back. The truck is parked on the side of the road, and it is the only truck visible in the image. It is positioned towards the left side of the image.

Figure 7 Visualization results on visual prompt understanding task. We use the masks predicted by our model under the GCG task as visual prompts, and generated region-level descriptions for these masks. The object masks and their captions for the corresponding region are highlighted in the same color.

Liangjun Zhang. Vinet: Visual and inertial-based terrain classification and adaptive navigation over unknown terrain. In ICRA, 2023.

- Style-a-video: Agile diffusion for arbitrary textbased video style transfer. SPL, 2024.
- [37] Zhicheng Huang, Zhaoyang Zeng, Bei Liu, Dongmei Fu, and Jianlong Fu. Pixel-bert: Aligning image pixels with text by deep multi-modal transformers. arXiv preprint arXiv:2004.00849, 2020.

- [38] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.

- [39] Louis Martin Hugo Touvron, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and et al. Llama 2: Open foundation and fine-tuned chat models. arXiv:2307.09288, 2023.

- [40] Sukjun Hwang, Miran Heo, Seoung Wug Oh, and Seon Joo Kim. Video instance segmentation using inter-frame communication transformers. In NeurIPS, 2021.

- [41] Shibo Jie, Yehui Tang, Ning Ding, Zhi-Hong Deng, Kai Han, and Yunhe Wang. Memory-space visual prompting for efficient vision-language fine-tuning. arXiv preprint arXiv:2405.05615, 2024.

- [42] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring

- [31] Pinxue Guo, Tony Huang, Peiyang He, Xuefeng Liu, Tianjun Xiao, Zhaoyu Chen, and Wenqiang Zhang. Openvis: Open-vocabulary video instance segmentation. arXiv preprint arXiv:2305.16835, 2023.

- [32] Kai Han, Jianyuan Guo, Yehui Tang, Wei He, Enhua Wu, and Yunhe Wang. Free video-llm: Promptguided visual perception for efficient training-free video llms. arXiv preprint arXiv:2410.10441, 2024.

- [33] Yingqing He, Menghan Xia, Haoxin Chen, Xiaodong Cun, Yuan Gong, Jinbo Xing, Yong Zhang, Xintao Wang, Chao Weng, Ying Shan, et al. Animate-a-story: Storytelling with retrievalaugmented video generation. arXiv preprint arXiv:2307.06940, 2023.

- [34] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. In CVPR, 2024.

- [35] Kuan-Chih Huang, Xiangtai Li, Lu Qi, Shuicheng Yan, and Ming-Hsuan Yang. Reason3d: Searching and reasoning 3d segmentation via large language model. In 3DV, 2025.

- [36] Nisha Huang, Yuxin Zhang, and Weiming Dong.

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

[Figure 129]

[Figure 130]

The object is a silver car, captured from the rear, driving on a busy road at night. Its brake lights are illuminated, suggesting it is either slowing down or stopped. The car's make is visible on the back, and the license plate is clearly displayed. It is among various vehicles, including a white SUV directly in front and a truck to the left. Throughout the scene, the car remains in motion, navigating through traffic alongside other vehicles.

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

The object is a laptop. It starts inside a black bag on a desk and is taken by a person wearing a red outfit, who is sitting on a black rolling office chair. The person places the laptop on the desk, opens it, and interacts with it by typing on the keyboard and using the trackpad. The laptop stays on the desk in front of the person with the screen facing them. The setting appears to be an office or classroom, as indicated by a whiteboard in the background.

Figure 8 Visualization failure cases.

to objects in photographs of natural scenes. In EMNLP, 2014.

- [43] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, 2016.

- [44] Anna Khoreva, Anna Rohrbach, and Bernt Schiele. Video object segmentation with language referring expressions. In ACCV, 2018.

- [45] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zeroshot video generation. In ICML, 2024.

- [46] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li,

- Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In CVPR, 2024.
- [47] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

- [48] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seedbench: Benchmarking multimodal large language models. In CVPR, 2024.

- [49] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, and Junnan Li. Aria: An open mul-

- timodal native mixture-of-experts model. arXiv preprint arXiv:2410.05993, 2024.
- [50] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In ICML, 2022.

- [51] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In ICML, 2023.

- [52] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.

- [53] Minghan Li, Shuai Li, Xindong Zhang, and Lei Zhang. Univs: Unified and universal video segmentation with prompts as queries. In CVPR, 2024.

- [54] Xiangtai Li, Haobo Yuan, Wenwei Zhang, Guangliang Cheng, Jiangmiao Pang, and Chen Change Loy. Tube-link: A flexible cross tube baseline for universal video segmentation. In ICCV, 2023.

- [55] Xiangtai Li, Haobo Yuan, Wei Li, Henghui Ding, Size Wu, Wenwei Zhang, Yining Li, Kai Chen, and Chen Change Loy. Omg-seg: Is one model good enough for all segmentation? In CVPR, 2024.

- [56] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llamavid: An image is worth 2 tokens in large language models. In ECCV, 2024.

- [57] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In EMNLP, 2023.

- [58] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In EMNLP, 2024.

- [59] Chang Liu, Henghui Ding, and Xudong Jiang. GRES: Generalized referring expression segmentation. In CVPR, 2023.

- [60] Chang Liu, Henghui Ding, Kaining Ying, Lingyi Hong, Ning Xu, Linjie Yang, Yuchen Fan, Mingqi Gao, Jingkun Chen, Yunqi Miao, et al. Lsvos 2025 challenge report: Recent advances in complex video object segmentation. ICCV workshop, 2025.

- [61] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.

- [62] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024.

- [63] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024. URL https://llava-vl. github.io/blog/2024-01-30-llava-next/.
- [64] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with crossattention control. In CVPR, 2024.

- [65] Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, et al. Llava-plus: Learning to use tools for creating multimodal agents. In ECCV, 2024.

- [66] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In ECCV, 2024.

- [67] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS, 2022.

- [68] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In ACL, 2024.

- [69] Shehan Munasinghe, Rusiru Thushara, Muhammad Maaz, Hanoona Abdul Rasheed, Salman Khan, Mubarak Shah, and Fahad Khan. Pg-video-llava: Pixel grounding large video-language models. arXiv preprint arXiv:2311.13435, 2023.

- [70] Shehan Munasinghe, Hanan Gani, Wenqi Zhu, Jiale Cao, Eric Xing, Fahad Shahbaz Khan, and Salman Khan. Videoglamm: A large multimodal model for pixel-level visual grounding in videos. arXiv preprint arXiv:2411.04923, 2024.

- [71] Zhenliang Ni, Xinghao Chen, Yingjie Zhai, Yehui Tang, and Yunhe Wang. Context-guided spatial feature reconstruction for efficient semantic segmentation. In ECCV, 2024.

- [72] Prashant W Patil, Akshay Dudhane, Ashutosh Kulkarni, Subrahmanyam Murala, Anil Balaji Gonde, and Sunil Gupta. An unified recurrent video object segmentation framework for various surveillance environments. IEEE TIP, 2021.

- [73] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023.

- [74] Lu Qi, Yi-Wen Chen, Lehan Yang, Tiancheng Shen,

- Xiangtai Li, Weidong Guo, Yu Xu, and MingHsuan Yang. Generalizable entity grounding via assistance of large language model. arXiv preprint arXiv:2402.02555, 2024.
- [75] Miao Rang, Zhenni Bi, Chuanjian Liu, Yehui Tang, Kai Han, and Yunhe Wang. Eve: Efficient multimodal vision language models with elastic visual experts. arXiv preprint arXiv:2501.04322, 2025.

- [76] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M. Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S. Khan. Glamm: Pixel grounding large multimodal model. In CVPR, 2024.

- [77] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

- [78] Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel reasoning with large multimodal model. In CVPR, 2024.

- [79] Seonguk Seo, Joon-Young Lee, and Bohyung Han. URVOS: Unified referring video object segmentation network with a large-scale benchmark. In ECCV, 2020.

- [80] Zhenwei Shao, Zhou Yu, Meng Wang, and Jun Yu. Prompting large language models with answer heuristics for knowledge-based visual question answering. In CVPR, 2023.

- [81] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- [82] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In NeurIPS, 2024.

- [83] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv:2302.13971, 2023.

- [84] Wen Wang, Yan Jiang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua

- Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint, 2023.
- [85] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. In arXiv, 2024.

- [86] Cong Wei, Haoxian Tan, Yujie Zhong, Yujiu Yang, and Lin Ma. LaSagnA: Language-based segmentation assistant for complex queries. arXiv preprint arXiv:2404.08506, 2024.

- [87] Cong Wei, Yujie Zhong, Haoxian Tan, Yingsen Zeng, Yong Liu, Zheng Zhao, and Yujiu Yang. Instructseg: Unifying instructed visual segmentation with multi-modal large language models. arXiv preprint, 2024.

- [88] Cong Wei, Yujie Zhong, Haoxian Tan, Yong Liu, Zheng Zhao, Jie Hu, and Yujiu Yang. Hyperseg: Towards universal visual segmentation with large language model. In CVPR, 2025.

- [89] Fangyun Wei, Jinjing Zhao, Kun Yan, Hongyang Zhang, and Chang Xu. A large-scale human-centric benchmark for referring expression comprehension in the LMM era. In NeurIPS, 2024.

- [90] Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang. Grit: A generative region-to-text transformer for object understanding. In ECCV, 2024.

- [91] Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, and Ping Luo. Language as queries for referring video object segmentation. In CVPR, 2022.

- [92] Jiannan Wu, Yi Jiang, Bin Yan, Huchuan Lu, Zehuan Yuan, and Ping Luo. Uniref++: Segment every reference object in spatial and temporal spaces. In ICCV, 2023.

- [93] Jianzong Wu, Xiangtai Li, Chenyang Si, Shangchen Zhou, Jingkang Yang, Jiangning Zhang, Yining Li, Kai Chen, Yunhai Tong, Ziwei Liu, et al. Towards language-driven video inpainting via multimodal large language models. In CVPR, 2024.

- [94] Qiong Wu, Xiangcong Yang, Yiyi Zhou, Chenxin Fang, Baiyang Song, Xiaoshuai Sun, and Rongrong Ji. Grounded chain-of-thought for multimodal large language models. arXiv preprint arXiv:2503.12799, 2025.

- [95] Size Wu, Sheng Jin, Wenwei Zhang, Lumin Xu, Wentao Liu, Wei Li, and Chen Change Loy. F-lmm: Grounding frozen large multimodal models. arXiv preprint arXiv:2406.05821, 2024.

- [96] Zhuofan Xia, Dongchen Han, Yizeng Han, Xuran Pan, Shiji Song, and Gao Huang. Gsva: Generalized

- segmentation via multimodal large language models. In CVPR, 2024.
- [97] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.

- [98] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameterfree llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024.

- [99] Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Zehuan Yuan, Ping Luo, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In CVPR, 2023.

- [100] Cilin Yan, Haochen Wang, Shilin Yan, Xiaolong Jiang, Yao Hu, Guoliang Kang, Weidi Xie, and Efstratios Gavves. Visa: Reasoning video object segmentation via large language models. In ECCV, 2024.

- [101] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

- [102] Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen. Seedstory: Multimodal long story generation with large language model. arXiv preprint arXiv:2407.08683, 2024.

- [103] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Lavt: Language-aware vision transformer for referring image segmentation. In CVPR, 2022.

- [104] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mPLUG-Owl3: Towards long imagesequence understanding in multi-modal large language models. arXiv preprint, 2024.

- [105] Licheng Yu, Patrick Poirson, Shan Yang, Alexan-

- der C Berg, and Tamara L Berg. Modeling context in referring expressions. In ECCV, 2016.
- [106] Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, Mohit Bansal, and Tamara L Berg. Mattnet: Modular attention network for referring expression comprehension. In CVPR, 2018.

- [107] Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. Osprey: Pixel understanding with visual instruction tuning. In CVPR, 2024.

- [108] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024.

- [109] Yuhang Zang, Wei Li, Jun Han, Kaiyang Zhou, and Chen Change Loy. Contextual object detection with multimodal large language models. IJCV, 2024.

- [110] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In EMNLP, 2023.

- [111] Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Chunyuan Li, Jainwei Yang, et al. Llavagrounding: Grounded visual chat with large multimodal models. In ECCV, 2024.

- [112] Pan Zhang, Xiaoyi Dong, Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Wenwei Zhang, Hang Yan, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlmxcomposer: A vision-language large model for advanced text-image comprehension and composition. arXiv preprint arXiv:2309.15112, 2023.

- [113] Tao Zhang, Xingye Tian, Yu Wu, Shunping Ji, Xuebo Wang, Yuan Zhang, and Pengfei Wan. DVIS: Decoupled video instance segmentation framework. In ICCV, 2023.

- [114] Tao Zhang, Xiangtai Li, Hao Fei, Haobo Yuan, Shengqiong Wu, Shunping Ji, Change Loy Chen, and Shuicheng Yan. Omg-llava: Bridging imagelevel, object-level, pixel-level reasoning and understanding. In NeurIPS, 2024.

- [115] Tao Zhang, Xiangtai Li, Zilong Huang, Yanwei Li, Weixian Lei, Xueqing Deng, Shihao Chen, Shunping Ji, and Jiashi Feng. Pixel-sail: Single transformer for pixel-grounded understanding. arXiv preprint arXiv:2504.10465, 2025.

- [116] Tao Zhang, Xingye Tian, Yikang Zhou, Shunping Ji, Xuebo Wang, Xin Tao, Yuan Zhang, Pengfei Wan, Zhongyuan Wang, and Yu Wu. Dvis++: Improved decoupled framework for universal video segmentation. IEEE TPAMI, 2025.

- [117] Yichi Zhang, Ziqiao Ma, Xiaofeng Gao, Suhaila Shakiah, Qiaozi Gao, and Joyce Chai. Groundhog: Grounding large language models to holistic segmentation. In CVPR, 2024.

- [118] Yuxuan Zhang, Tianheng Cheng, Rui Hu, Lei Liu, Heng Liu, Longjin Ran, Xiaoxin Chen, Wenyu Liu, and Xinggang Wang. Evf-sam: Early visionlanguage fusion for text-prompted segment anything model. arXiv preprint arXiv:2406.20076, 2024.

- [119] Zheng Zhang, Yeyao Ma, Enming Zhang, and Xiang Bai. Psalm: Pixelwise segmentation with large multi-modal model. In ECCV, 2024.

- [120] Hao Zhou, Tiancheng Shen, Xu Yang, Hai Huang, Xiangtai Li, Lu Qi, and Ming-Hsuan Yang. Rethinking evaluation metrics of open-vocabulary segmentaion. arXiv preprint arXiv:2311.03352, 2023.

- [121] Yikang Zhou, Tao Zhang, Shunping Ji, Shuicheng Yan, and Xiangtai Li. Improving video segmentation via dynamic anchor queries. In ECCV, 2024.

- [122] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. arXiv preprint arXiv:2405.01434, 2024.

- [123] Feng Zhu, Zongxin Yang, Xin Yu, Yi Yang, and Yunchao Wei. Instance as identity: A generic online paradigm for video instance segmentation. In ECCV, 2022.

- [124] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

- [125] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020.

