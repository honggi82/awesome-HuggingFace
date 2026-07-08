## LLM-AD: Large Language Model based Audio Description System

Peng Chu Jiang Wang Andre Abrantes Microsoft

{pengchu, jiangwang, abrantes}@microsoft.com

# arXiv:2405.00983v1[cs.CV]2May2024

### Abstract

The development of Audio Description (AD) has been a pivotal step forward in making video content more accessible and inclusive. Traditionally, AD production has demanded a considerable amount of skilled labor, while existing automated approaches still necessitate extensive training to integrate multimodal inputs and tailor the output from a captioning style to an AD style. In this paper, we introduce an automated AD generation pipeline that harnesses the potent multimodal and instruction-following capacities of GPT-4V(ision). Notably, our methodology employs readily available components, eliminating the need for additional training. It produces ADs that not only comply with established natural language AD production standards but also maintain contextually consistent character information across frames, courtesy of a tracking-based character recognition module. A thorough analysis on the MAD dataset reveals that our approach achieves a performance on par with learning-based methods in automated AD production, as substantiated by a CIDEr score of 20.5.

### 1. Introduction

The advent of Audio Description (AD) represents a significant leap forward in making video content more accessible and inclusive. AD offers a spoken narrative of crucial visual elements within a video that are not captured by the original audio track. This innovation holds particular importance for individuals with visual impairments, as it enables them to fully engage with visual media. However, creating accurate AD is resource-intensive, requiring specialized expertise, equipment, and significant time investment. Automating the production of AD enhances the accessibility of videos for individuals with visual impairments and also broadens the scope of ”eyes-free” viewing for the general audience in the realm of popular online videos.

AD is a service that adheres to diverse and wellestablished production protocols, differing significantly from conventional image captioning tasks by emphasizing storytelling over mere description. To generate AD sen-

tences that meet these standards, traditional methods train language models on existing AD datasets to encapsulate the characteristic AD style. Another notable challenge in automating AD production is generating sentences of appropriate length that seamlessly fit into the varying temporal gaps within actor dialogue. Language models lacking specific instruction-following capabilities often struggle to produce sentences that comply with length constraints. However, the advent of Large Language Models (LLMs), particularly GPT-4V(ision), showcases a robust capacity for adhering to instructions. This capability enables the generation of more precise AD content by directly feeding the model with AD production guidelines and desired sentence lengths as natural language instruction prompts. GPT4V(ision) stands out as the latest LLM iteration, uniquely supporting both image and text inputs to produce textual outputs. Its advanced multimodal capabilities hold promising potential for establishing a practical automated AD generation framework.

In this paper, we introduce an automated pipeline that utilizes GPT-4V(ision) for the generation of accurate AD for videos. Our approach processes a movie clip and its title information to produce AD content. We provide an overview of this pipeline, depicted in Fig. 1. Our methodology harnesses the multimodal capabilities of GPT-4V, integrating visual cues from video frames with textual context, such as previous subtitles, to generate AD content. By allowing the input of AD production guidelines and preferred output sentence lengths as natural language prompts, our system adeptly generates AD of suitable length tailored to speech gaps and can swiftly adapt to various video categories. Furthermore, our pipeline incorporates a trackingbased character recognition module, which employs temporal data to deliver consistent character information across frames without necessitating additional training, thus ensuring uniform performance on new video content.

In summary, our contributions are twofold:

- • We propose an automated pipeline that leverages natural language AD production guidelines for the generation of precise AD content.
- • We introduce a tracking-based character recognition

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Title Info:

Search

Prompt: Generate AD using

“Harry Potter

[Figure 5]

and the goblet of fire”

[Figure 6]

Movie Database

Context Subtitles

|[Figure 7]<br><br>[Figure 8]<br><br>GPT-4V|
|---|

Actor: Harry Potter, Harry Potter, no, no, Harry Potter, come on, Harry, Harry, for goodness

[Figure 9]

Movie Clip

[Figure 10]

[Figure 11]

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

###### Actor: He's a

|[Figure 15]|
|---|

|[Figure 16]|
|---|

cheat! Actor: You're not even 17 yet! ……

[Figure 17]

[Figure 18]

[Figure 19]

AD: Snape confronts Harry in the Great Hall.

GPT-4V

|[Figure 20]<br><br>[Figure 21]<br><br>Tracklet Face<br><br>Match|
|---|

Visual Prompt

Image Frames

[Figure 22]

[Figure 23]

|[Figure 24]<br><br>[Figure 25]<br><br>Person Tracking|
|---|

[Figure 26]

|[Figure 27]<br><br>[Figure 28]<br><br>Shot Boundary<br><br>Detection|
|---|

[Figure 29]

[Figure 30]

Downsample

Input GPT-4V AD Generation

Figure 1. Overview of the GPT-4V based automated AD generation pipeline.

module that supplies GPT-4V with temporally consistent character information.

To validate our approach, we conducted extensive experiments on the MAD [11] dataset, with the results setting a new benchmark for state-of-the-art performance.

### 2. Related Work

#### 2.1. Large Multimodal Model

Large multimodal models (LMMs) are a recent trend in artificial intelligence that aims to combine different types of data, such as text, images, audio, and video, to achieve more general and robust intelligence. One of the most prominent examples of LMMs is GPT-4V, a model that extends the large language model GPT-4 with vision capabilities. GPT-4V can process arbitrarily interleaved multimodal inputs and generate coherent and diverse outputs, such as descriptions, captions, stories, questions, and answers. Illustrating its adaptability, GPT-4V has demonstrated outstanding performance across a spectrum of applications, such as visual question answering [7], video interpretation [9], and clinical diagnostics [7, 14].

#### 2.2. Audio Description Generation

Han et al.[3, 4] introduced an innovative approach to AD generation by adapting existing language models, such as

GPT-2, to handle multimodal information. Their methodology involves training adaptation layers using existing AD datasets to imbibe the distinctive linguistic nuances of AD narratives. Moreover, the efficacy of their learningbased character recognition module is contingent upon the availability of specialized training datasets. In contrast, our methodology harnesses natural language instructions to guide the GPT-4V model in producing ADs that adhere to established production guidelines.d The proposed method employs a tracking-based character recognition module that operates without the need for training, thus guaranteeing consistent performance across new video content.

MM-VID[9] marked a pioneering effort in utilizing the GPT-4V model for AD generation through a unique twostage pipeline. In this process, GPT-4V first synthesizes condensed frame captions, which GPT-4 then refines into the final AD output. A critical observation is that their approach lacks an explicit process for character recognition. Our methodology sets itself apart with a streamlined one-stage process complemented by a dedicated character recognition module. This module provides intricate details about characters in the film, enabling the generation of richer and more contextually aligned AD narratives.

### 3. Methodology

Our method takes the video frames of a movie clip and its title information as input to output AD. We leverage the visual clues in the video frames, character information and the story context to prompt GPT-4V for the generation as shown in Fig. 1.

#### 3.1. Character Recognition

In our approach, we harness the temporal dynamics within videos to execute tracklet-level face recognition, aiming to accurately identify characters from a predefined cast list. Initially, upon processing a movie clip’s frame sequence, our algorithm employs shot boundary detection techniques to partition the frames into subsequences. This partitioning is based on the continuity of camera movement, ensuring that abrupt changes in viewpoint are segmented into distinct subsequences. This step is crucial for facilitating the reliable tracking of characters, thereby yielding tracklets that maintain consistent identities throughout.

For each video subsequence, we implement multipleperson tracking to generate person tracklets. Tracklets characterized by low confidence scores or brief durations are systematically excluded to eliminate background characters or those not central to the narrative.

To gather comprehensive information about the movie’s cast, we query movie databases such as IMDb. This search retrieves not only the cast list but also cast profile images and, when available, subtitles. It is noteworthy that cast profile images may not always correspond temporally to the period of the movie’s production. To counteract discrepancies arising from age differences or variations in facial makeup, our method, akin to the strategy outlined in [4], involves the collection of exemplar faces of cast members directly from the movie footage. Specifically, a face recognition model, denoted as F(·), processes the cast profile images to produce cast face embeddings. This model is subsequently applied to all faces detected within the person tracklets to generate query face embeddings. For each cast member, we select the embeddings of the top K matched faces from the query set as augmented embeddings for that cast member’s face.

To identify the characters, we compare the face embeddings from each person tracklet against both the original and augmented face embeddings for each cast member. We compute an average distance between the tracklet embeddings and those of the cast members. The name of the cast member whose embeddings exhibit the shortest average distance to the tracklet and falls below a predefined threshold τ is assigned as the identity of the character in the tracklet. The specifics of this tracklet face matching process are detailed in Alg. 1. Additionally, we explore and evaluate alternative character recognition techniques in Sec. 4.2, contributing to a comprehensive understanding of the efficacy

Algorithm 1 Tracklet Face Match

- 1: Input: Movie frames I, face images of all casts Ic, and person tracklets without name T.
- 2: Output: Person tracklets with names Tˆ.
- 3: Extract original cast face embedding Eorg ← F(Ic).
- 4: Extract query face embedding Eq ← F(crop(I, T)).
- 5: for Eiorg ← E1org, . . . , EMorg do
- 6: d ← dist(Eiorg, Eq)
- 7: Eaugi ← Eq[topk index(−d, K)]

- 8: end for
- 9: Cast face embeddings Ec ← Eorg + {E1aug, . . . , EMaug}
- 10: for Tj ← T1, . . . , TN do

▷ F1 and FL are the first and last frame # of Tj

- 11: EqT

j

← {ETq

j,f=F1, . . . , ETq

j,f=FL}

- 12: for Eci ← Ec1, . . . , EcN do
- 13: D ← dist(Eci, EqT

j

)

- 14: di ← avg(flatten(D))
- 15: end for
- 16: d ← {d1, . . . , dM}

▷ argmin(·) returns the index of the smallest element

- 17: imin ← argmin(d)
- 18: if dimin < τ then
- 19: Tˆj ← {Tj, imin}
- 20: end if
- 21: end for
- 22: Tˆ ← {Tˆ1, . . . , TˆN}

of our proposed method.

#### 3.2. Audio Description Generation

The generation of Audio Descriptions (AD) is a complex process that involves the integration of visual cues, character information, and story context, as illustrated in Fig.1. Utilizing GPT-4V, a state-of-the-art language generation model with a vision backbone, enables our system to process both textual and visual inputs, making it ideally suited for our AD generation task.

For visual input, we select 10 image frames in temporal order from each movie clip. These frames are processed by GPT-4V, which is instructed to analyze the details within individual frames and to understand actions occurring over multiple frames, effectively interpreting the sequence as a video narrative. To incorporate character information, we employ visual prompts by overlaying each image frame with the names of the characters and their corresponding bounding boxes. This approach provides GPT-4V with spatial information about the characters, enhancing its ability to generate contextually rich AD. The effectiveness of various visual prompts for this task is further examined in Sec.4.3.

Building on the insights from [3], we recognize the significance of integrating story context into the AD production process. To this end, for each movie clip, we compile any available subtitles within the preceding T AD into the text prompts. This ensures the continuity and relevance of

the generated AD within the broader narrative context. The impact of different choices of T on the quality of the final AD is discussed in Sec.4.4.

Combining the prepared image frames with character annotations and the textual context, along with specific task instructions, we feed this comprehensive input into GPT-4V for AD generation. It is imperative to note that the linguistic style required for AD differs significantly from that of typical image captioning tasks. AD emphasizes narrative storytelling, whereas image captioning focuses more on describing visuals with even emphasis. Our experiments reveal that GPT-4V possesses inherent knowledge about AD and its characteristics, allowing our prompts to simply specify the desired output as AD rather than detailed captioning, thereby conserving token usage.

A crucial aspect of effective AD is ensuring that the length of generated AD is appropriately tailored to fit within the gaps between dialogue or subtitles. GPT-4V’s advanced instruction-following capabilities allow us to directly control the length of every output AD by specifying the desired word count in the task prompts. The influence of different task prompts on AD generation, particularly with respect to controlling AD length, is explored and evaluated in Sec.4.5.

### 4. Experiments

Our methodology is evaluated using the MAD dataset, as introduced in [11]. The MAD dataset is a rich collection comprising over 264,000 audio descriptions sourced from 488 movies. For our analysis, we leverage the evaluation subset of this dataset, which includes 10 carefully selected movies, to benchmark our proposed method and conduct comparisons with existing approaches. Given the considerable inference time associated with the LLM, we strategically sample approximately 400 movie clips from this subset for all ablation assessments. The performance on the complete evaluation subset is reported when compared with existing approaches.

#### 4.1. Implementation Details

In our implementation, we utilize a simplified version of the multiple-person tracker presented by [2] to generate person tracklets, capturing all characters appearing in the input movie clip. Initial processing involves employing TransNetV2, as described by [12], for detecting and segmenting clips that contain multiple shots. After tracklet generation, we extract square patches around each person from the frames, ensuring these patches include adequate visual context. Face detection within these person patches is accomplished using the YOLOv7 model [1], enabling us to crop, and align face patches to a standard size of 112 × 112 pixels. For the purpose of face recognition, we harness the capabilities of AdaFace [6] with an R100 backbone, trained

on the expansive WebFace12M [15] dataset, to extract 512dimensional face feature vectors. For visual inputs concerning each clip, we uniformly sample 10 frames. The AD generation phase is powered by the Azure OpenAI GPT-4V deployment.

Evaluation Metrics We follow the protocol of other work to adopt classic text generation metrics to report AD generation performance, e.g. the classic captioning metrics to compare the generated AD to the ground-truth AD, namely, ROUGE-L [8] and CIDEr [13].

#### 4.2. Tracking-based Character Recognition

Our investigation begins with an assessment of the character recognition capabilities across various implemented approaches. In this analysis, we evaluate clip-wise character recognition outcomes with their respective ground truths and compute the overall recall and precision across all clips, with results detailed in Tab. 1. It’s pertinent to note that the MAD dataset lacks specific annotations for character recognition. To address this, we employ GPT-4 for named entity recognition against a predefined cast list on the ground-truth AD to generate character annotations for each clip. Given that not every character is mentioned in the AD, our generated annotations are inherently imperfect and primarily serve to compare the efficacy of different configurations.

Given GPT-4V’s robust vision capabilities, we establish a GPT-4V based character recognition methodology as our baseline. As depicted in Fig. 2, this process involves prompting GPT-4V with nine image frames showing the characters for recognition, supplemented by an additional gallery image showcasing all cast members’ faces. To mitigate the influence of GPT-4V’s pre-existing movie knowledge, we reference the cast using their IMDb IDs rather than their real or character names. GPT-4V is then instructed to identify all observable persons’ IDs. It’s important to highlight that the Azure OpenAI GPT-4V deployment, by default, applies face blurring to user-submitted images depicting humans. This precaution significantly impairs recognition accuracy, with GPT-4V’s recall and precision approximating random guesses when face blurring is enabled. Disabling face blurring markedly enhances performance; however, GPT-4V still falls behind in precision compared with a dedicated face recognition model marked as ”face recognition only” in Tab. 1. This discrepancy is partly due to the cast gallery images not always matching the age period of the characters’ portrayal in the films. Consequently, GPT4V does not outperform a face recognition model trained with corresponding data in this task.

In our tracking-based character recognition approach, we contrast its performance against scenarios solely utilizing face recognition without tracking. Additionally, we explore the impact of incorporating exemplar faces. By integrating tracking with face recognition, our method achieves a sig-

Prompt: List the person in the [Gallery] who is also shown in the [Image frames]

[Image Frame] [Image Frame] [Gallery]

[Figure 31]

[Figure 32]

[Figure 33]

GPT-4V: [“nm0705356”, “nm0342488”]

Figure 2. Use GPT-4V for character recognition.

|Configuration<br><br>|Recall Precision|
|---|---|
|GPT-4V w face blur GPT-4V w/o face blur|0.342 0.276 0.468 0.518<br><br>|
|face recognition only w/o exemplar ours<br><br>|0.471 0.788 0.672 0.763 0.709 0.759|

Table 1. Ablations on different character recognition methods and configurations.

nificant improvement in recall, from 0.471 to 0.672, with a minimal decrease in precision of only 0.025. Incorporating exemplar faces further enhances recall by 0.037, albeit at a slight cost to precision of 0.004. When compared with the existing methodology, where [4] developed a transformerbased recognition module trained on the MovieNet [5] dataset, achieving average recall and precision of 0.83 and 0.75 respectively on four selected MAD-eval movies, our proposed tracking-based approach demonstrates a performance disparity in terms of recall and precision. However, our method benefits from not requiring additional training, ensuring its performance remains unaffected by the availability of training data and thus offering enhanced generalizability to newly encountered videos.

#### 4.3. Visual Prompting

In this section, we explore the impact of various visual prompts that incorporate character information into image frames for the generation of AD by GPT-4V. As depicted in Fig. 3, character information is visually represented by overlaying green bounding boxes and accompanying name tags on the image frames. We assess the quality of the generated AD under different settings of visual prompts, with the findings summarized in Tab. 2.

When visual prompts consist solely of character bounding boxes, the resulting CIDEr scores decreases 1.8 points compared to those observed in scenarios devoid of any visual prompting. This outcome indicates that the inclusion of bounding boxes may obscure pertinent visual cues or detracts from GPT-4V’s ability to generate accurate AD. Incorporating only the names of characters directly onto the

[Figure 34]

[Figure 35]

original frame bounding box only

[Figure 36]

[Figure 37]

name only bounding box & name

Figure 3. Illustration of different visual prompting to add character information.

|Configuration<br><br>|ROUGE-L CIDEr|
|---|---|
|w/o visual prompting bounding box only name only bounding box & name<br><br>|9.6 18.8 9.9 17.0<br><br>13.6 23.9 12.4 23.0|
|face recognition only<br><br>|10.8 18.8|

Table 2. Ablations on different visual prompting for adding character names.

image frames significantly enhances the quality of the generated AD, evidenced by a notable increase in the CIDEr score (from 18.8 to 23.9) and ROUGE-L (from 9.6 to 13.6). Further adding bounding boxes to the character names decreases the CIDEr score by 0.9. Based on these results, we opted to utilize character name text only in subsequent experiments.

The last entry in Tab. 2 demonstrates the comprehensive effectiveness of our tracking-based character recognition module by contrasting the AD quality against that produced using only a standard face recognition approach. Employing a conventional face recognition technique, GPT-4V generates AD with a notably lower CIDEr score (e.g., 18.8 vs. 23.9). An illustrative comparison presented in Fig. 4 reveals that, without the aid of tracking, the system struggles to deduce character names when faces are either occluded or not facing the camera, resulting in the omission of two characters’ names in the generated AD.

#### 4.4. Textual Context

The incorporation of textual context significantly enhances the quality of the generated AD, as demonstrated in [3]. We investigate the impact of varying the number of textual context, e.g. subtitles and previous AD, included in the prompt

###### [Face Recognition]

[Figure 38]

[Figure 39]

[Figure 40]

[Generated AD]: Merrill urgently carries the girl through dense maize.

###### [Tracking-based Character Recognition]

[Figure 41]

[Figure 42]

[Figure 43]

[Generated AD]: Merrill carries Bo, while Graham navigates the maize field.

###### [Ground-truth AD]: MERRILL picks up BO and GRAHAM dashes on through the

maize.

Figure 4. Generated AD with different character recognition methods.

on the generated AD quality, with our findings presented in Tab. 3. To quantify the context length, we follow the definition in [3] to treat the subtitles between the T-th AD and the current AD timestamp as context length T. Notably, the inclusion of 100 context subtitles markedly boosts the CIDEr score from 22.2 to 23.9 and the ROUGE-L score from 12.8 to 13.5, in comparison to scenarios without any context subtitles. This substantial improvement underscores the value of leveraging subtitles to generate context-coherent AD. However, augmenting the inputs with more than 100 context subtitles yields diminishing returns in performance enhancement, as illustrated by the results obtained for 200 contexts.

The role of previous AD in enhancing AD performance is also evident. Referring to the last row of Tab. 3, the inclusion of previous AD leads to a decrease in the CIDEr score by 1.6 points, from 23.9 to 22.3. This decrease in performance could be partially attributed to our asynchronous inference design. In order to process extensive videos such as movies, we have structured the pipeline to infer input clips in parallel. To incorporate previous AD as textual context, the pipeline must be executed twice. During the initial run, the AD is generated solely based on context subtitles. In the subsequent run, the AD generated from the previous step is incorporated as context AD to produce the final AD. This design deviates from the recursive configuration in[3], and as a result, may not fully exploit the information encapsulated in the context AD.

#### 4.5. Prompts for Audio Description Generation

In this section, we delve into the effects of varying task prompts on the generation of AD by comparing the performance impact of explicitly specifying an AD-style output against not doing so. The initial portion of Tab. 4 presents a

|Number of Context T<br><br>|ROUGE-L CIDEr|
|---|---|
|0 6 20 100 200<br><br>|12.8 22.2<br>13.1 23.2<br><br><br>13.0 23.5 13.6 23.9<br>13.1 20.0<br>|
|100 w context AD<br><br>|12.4 22.3|

Table 3. Ablations on the number of context AD and subtitles to include for AD generation.

comparison between using a task prompt that directs GPT4V to generate AD-style content and a prompt that does not. To isolate the influence from textual context, no contextual AD or subtitles were included, preventing GPT-4V from context learning the AD style from prior examples. The findings indicate a significant enhancement in the quality of AD generation, with a CIDEr score increase of 6.4 when GPT-4V is instructed to produce AD-style outputs as opposed to standard caption-style sentences. Fig. 5 showcases two examples illustrating that, under restricted word count conditions, a captioning-style output tends to enumerate observations in the image sequences, whereas an ADstyle output weaves these observations into a narrative more akin to the ground-truth AD.

[Image Frames]

[Figure 44]

[Figure 45]

[Figure 46]

[w/o style instruction]: Morgan Hess and Rev. Graham Hess are depicted in

various emotional states during a meal in a domestic setting.

[w AD style instruction]: Graham eats in distress as Morgan watches, both struggling with their emotions at the dining table.

[Ground-truth AD]: Observing him, MORGAN gets up from his chair and goes to his dad who is now throwing down his fork and is sobbing uncontrollably.

##### [Image Frames]

[Figure 47]

[Figure 48]

[Figure 49]

###### [w/o style instruction]: Dynamic scenes featuring Ron Weasley in various

states of action and rest.

[w AD style instruction]: Ron and companions endure a tumultuous fall and recover together.

[Ground-truth AD]: GINNY HERMIONE RON HARRY and the twins let go, and land heavily.

Figure 5. Generated AD with different linguistic styles instructions.

Another critical aspect of high-quality AD production is ensuring the generated AD appropriately fits within the time gaps of subtitles. Without explicit word count constraints,

GPT-4V’s output tends to be verbose, significantly diverging from the concise style of ground-truth AD. The second section of Tab. 4 highlights this issue, showing a mere 4.7 CIDEr score for AD generated without word count limitations. In contrast to other methods that infer a concise AD style through learning from training data, GPT-4V’s robust instruction-following capability allows us to directly specify the desired word count for AD outputs. We explore the impact of instructing GPT-4V to generate all AD in fixed word counts, such as 6, 10 and 20 words, with the performance outcomes detailed in Tab. 4. The choice of 6 words aligns with the average word count across the AudioVault [3] dataset, where 80% of the AD contain 10 words or fewer, and 99% of the AD do not exceed 20 words. Our results demonstrate that, amongst the fixed word counts of 6, 10, and 20, the 10-word prompts exhibit the highest ROUGE-L and CIDEr scores.

Given the variability in the duration of subtitle gaps, optimally, ADs should vary in length to match these intervals. In this study, we employ a setting that references the word count of each ground-truth AD to guide GPT-4V in generating AD of corresponding lengths, aiming to illustrate the importance of AD length. The potential for automatically estimating AD length based on the temporal gaps between subtitles represents an exciting avenue for future research. The results in Tab. 4 demonstrate the superior performance of the variant length AD setting, outperforming the 10-word setting by 3.3 points in the CIDEr metric.

The MM-VID [9] framework employs a two-stage approach for generating AD, where GPT-4V is utilized for generating clip-level descriptions, and GPT-4 is tasked with producing task-specific responses based on the output from GPT-4V. Inspired by this methodology, we too investigated a similar two-stage architecture. In our exploration, we directed GPT-4V to create detailed textual descriptions for each of the 10 image frames. Subsequently, GPT-4 was instructed to summarize GPT-4V’s detailed outputs into coherent AD, applying similar prompts discussed earlier (e.g., context and output word count constraints). The outcomes, as reported in the final section of Tab. 4, reveal a inferior performance in comparison to the results from the preceding setup (CIDEr from 23.9 to 21.5, ROUGE-L from 13.6 to 12.3).

This decline in performance could be attributed to the design intent behind MM-VID, which is crafted as a general tool for video understanding, necessitating GPT-4’s superior instruction-following capabilities to manage multiple tasks concurrently. In contrast, our framework is specifically tailored for the automatic generation of AD, for which the linguistic capabilities of GPT-4V alone appear to be adequately sufficient. Moreover, the streamlined one-stage design may offer the added advantage of preserving subtle visual cues that could potentially be overlooked in the

transition from frame-level textual descriptions to final AD output in a two-stage process.

|Prompts|ROUGE-L CIDEr<br><br>|
|---|---|
|w/o output style instruction w AD style instruction<br><br>|11.1 17.5<br>12.6 23.9<br>|
|w/o output word count limit output 6 words output 10 words output 20 words output ground-truth AD length<br><br>|12.1 4.7 10.4 19.0<br>13.6 23.9 13.1 19.8 12.9 27.2<br>|
|two-stage, GPT-4 summary|12.3 21.5|

Table 4. Ablations on different prompts for AD generation.

#### 4.6. Comparison with the SOTA

For an equitable comparison with existing methodologies, we assess our approach using the complete evaluation subset of the MAD dataset. To ensure a fair comparison with alternative approaches, we configured our method to generate AD consisting of a fixed length of 10 words. Our approach demonstrates superior performance over AutoAD-II, establishing a new state-of-the-art performance with CIDEr and ROUGE-L scores of 20.5 (vs 19.5) and 13.5 (vs 13.4), respectively.

|Methods<br><br>|context|ROUGE-L CIDEr<br><br>|
|---|---|---|
|ClipCap [10]<br><br>AutoAD-I [3]<br><br>AutoAD-II [4]<br><br><br>|no no no|8.5 4.4 10.3 12.1 13.1 19.2<br><br>|
|AutoAD-I [3]<br><br>AutoAD-II [4] Ours<br><br><br>|AD & subt. AD & subt. subt.|11.9 14.3<br><br>13.4 19.5<br><br>13.5 20.5<br>|

Table 5. Comparison with other methods.

### 5. Discussion and Future Work

The GPT-4V models, trained on extensive collections of visual and linguistic data, represent a significant stride forward in the automatic generation of Audio Descriptions (AD) for cinematic content. However, the generation of AD for films within the MAD dataset by GPT-4V may inherently incorporate biases due to the model’s exposure to potentially similar content during training. To mitigate such biases, it is advisable to evaluate the model’s performance using a more contemporary selection of movies, which are less likely to have been part of the model’s training corpus.

Another notable limitation of our current approach is the absence of a mechanism for determining appropriate mo-

ments within a film to insert AD and estimating the corresponding word count for that AD. This deficiency becomes particularly evident when considering the substantial decline in performance (a CIDEr reduction from 27.2 to 23.9) observed when imposing a fixed word count for AD output, as opposed to tailoring the length based on the ground truth AD. The performance gap shows the importance of AD length and exhibits a promising direction for future works to explore for improving the generated AD quality. As an example, one can customize a relatively lightweight language-rewritten model from existing AD data to tailor the over-verbose output from the LLM.

### References

- [1] yolo7-face. https://github.com/derronqi/ yolov7-face. 4
- [2] Peng Chu, Jiang Wang, Quanzeng You, Haibin Ling, and Zicheng Liu. Transmot: Spatial-temporal graph transformer for multiple object tracking. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4870–4880, 2023. 4
- [3] Tengda Han, Max Bain, Arsha Nagrani, G¨ul Varol, Weidi Xie, and Andrew Zisserman. AutoAD: Movie description in context. In CVPR, 2023. 2, 3, 5, 6, 7
- [4] Tengda Han, Max Bain, Arsha Nagrani, G¨ul Varol, Weidi Xie, and Andrew Zisserman. AutoAD II: The Sequel - who, when, and what in movie audio description. In ICCV, 2023. 2, 3, 5, 7
- [5] Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and Dahua Lin. Movienet: A holistic dataset for movie understanding. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16, pages 709–727. Springer, 2020. 5
- [6] Minchul Kim, Anil K Jain, and Xiaoming Liu. Adaface: Quality adaptive margin for face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 4
- [7] Yingshu Li, Yunyi Liu, Zhanyu Wang, Xinyu Liang, Lingqiao Liu, Lei Wang, Leyang Cui, Zhaopeng Tu, Longyue Wang, and Luping Zhou. A comprehensive study of gpt-4v’s multimodal capabilities in medical imaging. medRxiv, pages 2023–11, 2023. 2
- [8] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004. 4
- [9] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al. Mm-vid: Advancing video understanding with gpt-4v (ision). arXiv preprint arXiv:2310.19773, 2023. 2, 7
- [10] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. arXiv preprint arXiv:2111.09734, 2021. 7
- [11] Mattia Soldan, Alejandro Pardo, Juan Le´on Alc´azar, Fabian Caba, Chen Zhao, Silvio Giancola, and Bernard Ghanem. Mad: A scalable dataset for language grounding in videos

- from movie audio descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5026–5035, 2022. 2, 4
- [12] Tom´aˇs Souˇcek and Jakub Lokoˇc. Transnet v2: An effective deep network architecture for fast shot transition detection. arXiv preprint arXiv:2008.04838, 2020. 4
- [13] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4566–4575, 2015. 4
- [14] Chaoyi Wu, Jiayu Lei, Qiaoyu Zheng, Weike Zhao, Weixiong Lin, Xiaoman Zhang, Xiao Zhou, Ziheng Zhao, Ya Zhang, Yanfeng Wang, et al. Can gpt-4v (ision) serve medical applications? case studies on gpt-4v for multimodal medical diagnosis. arXiv preprint arXiv:2310.09909, 2023. 2
- [15] Zheng Zhu, Guan Huang, Jiankang Deng, Yun Ye, Junjie Huang, Xinze Chen, Jiagang Zhu, Tian Yang, Jiwen Lu, Dalong Du, et al. Webface260m: A benchmark unveiling the power of million-scale deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10492–10502, 2021. 4

### 6. Appendix

The GPT-4V prompt we used to generate the audio description is listed in Fig.6.

You are an expert movie audio description writer. Your task is to correctly describe and

write audio description for the scene in the provided movie clip. Procedure: Context Review: Review the context for the previous scenes provided in the [Context] section

to understand the story. The context is provided by lines of speech from the actors.

Video Review: Carefully examine the 10 images that are provided in temporal order in the [Video Clip] section. Some actors’ names are marked by the green text above their heads.

Describe: Describe what is happening in every frame in detail. You are not only required to

describe what is happening in every particular frame but also need to pay attention to the

scenes or events that are happening continuously across multiple frames. Summarize: Use the frame description in the previous step and the visual clues in the 10 images to summarize the movie clip and generate a one-sentence audio description. The number of words in the sentence can not exceed {n_word} words. Only include the first name of the characters if necessary. Output: Output your response in JSON format as {“Describe”: “frame description”, “Summarize”: “audio description”}

Figure 6. GPT-4V prompt to generate the audio description.

