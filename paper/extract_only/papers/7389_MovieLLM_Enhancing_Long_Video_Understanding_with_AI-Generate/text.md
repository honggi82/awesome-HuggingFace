# arXiv:2403.01422v5[cs.CV]11Aug2025

## DreamFrame: Enhancing Video Understanding via Automatically Generated QA and Style-Consistent Keyframes

Zhende Song

zdsong23@m.fudan.edu.cn Fudan University Shanghai, China

Chenchen Wang

24210720042@m.fudan.edu.cn Fudan University Shanghai, China

Jiamu Sheng

jmsheng22@m.fudan.edu.cn Fudan University Shanghai, China

Chi Zhang∗

chizhang@westlake.edu.cn Westlake University, Tencent PCG Hangzhou, China

Shengji Tang

21210720037@m.fudan.edu.cn Fudan University Shanghai, China

Jiayuan Fan†

jyfan@fudan.edu.cn Fudan University Shanghai, China

Tao Chen

eetchen@fudan.edu.cn Fudan University Shanghai, China

### ABSTRACT

Recent large vision-language models (LVLMs) for video understanding are primarily fine-tuned with various videos scraped from online platforms. Existing datasets, such as ActivityNet, require considerable human labor for structuring and annotation before effectively utilized for tuning LVLMs. While current LVLMs are primarily trained on existing datasets in broad, general-purpose settings, adapting them to specific downstream scenarios remains challenging, as collecting and annotating task-specific videos is highly labor-intensive and time-consuming. To address this issue, we propose a three-stage framework named DreamFrame for automatically generating style-consistent keyframes and corresponding question-answer (QA) pairs to support LVLM instruction tuning. DreamFrame generates datasets in a movie-like manner. First, we utilize an LLM to generate structured movie plots including movie prior information (like overview and style), frame descriptions and plot-related QA pairs, with a story expansion strategy to mitigate context length limitations.Then, to ensure visual consistency across generated frames, we design a Style Immobilization Process which maintains consistent style through an embedding learning strategy. Finally, frame descriptions and style embeddings are integrated to produce coherent keyframes. Using DreamFrame, we construct a dataset comprising approximately 1k stylized keyframe-like videos and 100k diverse QA pairs. Extensive fine-tuned experiments on various LVLM architectures demonstrate the effectiveness of the proposed dataset. Furthermore, based on the proposed dataset, we fine-tune a new LVLM named DreamFrame-7B, which significantly surpasses the previous similarsized LVLMs (+2.2 compared with VideoLLaVA-7B on MvBench) across different benchmarks. Code, data and supplementary will be at https://deaddawn.github.io/DreamFrame.

∗Work done while the author was a researcher at Tencent PCG †Corresponding Author.

### CCS CONCEPTS

• Computing methodologies → Computer vision; 2D Generation; Video Understanding;

### KEYWORDS

Text-to-2D, Vision-Language Models, Video Understanding Dataset

### 1 INTRODUCTION

Recent advancements in multi-modal learning, driven by large language models (LLMs) [2, 11, 27, 29, 30], have led to the development of numerous excellent vision-language models [3, 9, 10, 17, 22]. These models predominantly target tasks involving static images and text, such as image-text dialogue and text-to-image generation. Inspired by these achievements, LVLMs for video understanding [16, 18, 20, 21, 25, 33, 35] have emerged, extending the scope of multi-modal tasks to dynamic scenes. To optimize these models effectively, QA pairs based on video clips are required. Consequently, large volumes of videos from all sources have been utilized to construct training datasets.

Despite the abundance and diversity of video data available on various online platforms, most of them are unstructured and noisy, making unified collection a significant challenge.

Furthermore, substantial human efforts and time are also required for the latter high-quality annotation and filtering process. Additionally, current LVLMs for video understanding are typically trained on large-scale, pre-constructed datasets [4, 5, 7, 32], targeting general-purpose scenarios.

However, in the practical deployment, LVLMs follow a “pretraining&fine-tuning” manner to apply for specific downstream domains, such as movie understanding, security surveillance, or behavior recognition, where collecting and constructing tailored datasets remains a formidable challenge.

To address these challenges, we consider generating customized video datasets on demand. Inspired by the advances of generation models such as diffusion models, we explore the automatic generation of comprehensive datasets for video instruction tuning.

Specifically, we propose DreamFrame, a novel framework that automatically synthesizes style-consistent video keyframes and corresponding QA pairs. Our framework comprises three primary stages:

- 1) Movie Plot Generation. An LLM is first utilized to generate structured movie plots including movie prior information, frame descriptions and plot-related QA pairs. To mitigate context length limitations of LLMs, a story expansion strategy is proposed.
- 2) Style Immobilization Process. Then, to ensure visual consistency across generated frames, a style embedding is learned based on a diffusion model. This approach later guides the diffusion model to generate keyframes in a consistent style.
- 3) Video instruction data generation. By integrating the previously generated frame descriptions with the learned style embeddings, we generate style-consistent key frames, resulting in a comprehensive instruction tuning corpus.

Our methodology provides an innovative solution for the limitations of current datasets: 1) DreamFrame allows for the generation of datasets without constraints on data volume, ensuring a high degree of diversity within the generated content. 2) DreamFrame facilitates automatic annotation, significantly reducing the need for manual labor and associated costs. These advantages enhance the scalability, richness, and efficiency of dataset creation for video understanding. We hope that our preliminary exploration can offer promising insights for future research on automatic construction of video understanding datasets. Examples from our generated data are shown in Figure 1.

Our contributions are summarized as follows:

- • We propose a novel framework, i.e., DreamFrame, for generating video instruction tuning datasets. A story expanding strategy and a style immobilized strategy are designed to ensure the quality of the generated data.
- • Leveraging our generative approach, we have developed and will publicly release a comprehensive dataset for video understanding, alongside a superb LVLM, i.e., DreamFrame7B, trained for enhanced video understanding.
- • Extensive experiments are conducted to demonstrate that DreamFrame effectively and consistently enhances the capability of diverse LVLMs in video understanding. Meanwhile, our DreamFrame-7B remarkably outperforms previous LVLMs across different benchmarks, providing a new high-performance model for video understanding.

### 2 RELATED WORK

Vision Language Models. With the achievements of large language models (LLMs) such as GPT-4 [27] along with their opensource alternatives like LLaMA [30], researchers focus on leveraging the advanced language abilities of LLMs and developing the vision language models (VLMs) that integrate vision models with LLMs for cross-modality understanding. Representative VLMs like LLaVA [22], Flamingo [1] and InstructBLIP [10] have shown great capabilities in visual chat by constructing high-quality imageinstruction pairs to align the image and text dimensions. Further,

[Figure 1]

Film Name: The Duchess and the Commoner

||Frame 478 Frame 479<br><br>Line: Stay safe, My love.<br><br>[Figure 2]<br><br>[Figure 3]|
|---|
<br><br>|Frame 480 Frame 481<br><br>Line: The Duchess and the Commoner … This surely cannot remain hidden.<br><br>[Figure 4]<br><br>[Figure 5]|
|---|
<br><br>Question: Why are Rosamund and Thomas trying to keep their relationship a secret? Answer: They are attempting to protect their relationship from societal mores and prejudices due to their social ranks.|
|---|

||[Figure 6]<br><br>[Figure 7]<br><br>Frame 390 Frame 391<br><br>Line: Words penned with such passion… Could it be Thomas?|
|---|
<br><br>|Frame 392 Frame 393<br><br>Line:<br><br>How did he know of my feelings for him?<br><br>[Figure 8]<br><br>[Figure 9]|
|---|
<br><br>Question: Why is Rosamund so surprised and emotional after reading the letter? Answer: Because she believes that the love reflects the same feelings she has for Thomas.|
|---|

Figure 1: Examples of generated data. DreamFrame generates consistent key frames with reasonable lines and corresponding question-answer pairs. These data are used to fine-tune multi-modal large language models for video understanding.

VLMs are developed for video understanding [15, 18, 31, 34]. VideoLLaMA [35] utilizes BLIP-2 [17] fuse video embedding using QFormer. Video-ChatGPT [25] extracts video embedding by averaging frame-level features across temporal and spatial dimensions respectively. LLaMA-VID [20] is proposed for video understanding by encoding each frame with only two tokens. Recent LVLMs like QwenVL series comes with superior video comprehension ability. However, most of these LVLMs are trained on general-purpose tasks, with little or no fine-tuning for specific domains. A potential cause of this limitation is the lack of domain-specific datasets.

Video Instruction Tuning Datasets. Current training datasets for video understanding are predominantly constructed by annotating samples from existing datasets [4–7, 32, 38], with annotations typically generated through manual labeling or semi-automated methods. In contrast, our work is the first to explore fully automated dataset generation for video understanding, a direction that remains largely unexplored in the current literature.

### 3 METHOD

The overallframeworkof theproposed DreamFrame is shown inFigure 2, which consists of three stages: Movie Plot Generation, Style Immobilization Process and Video Instruction Data Generation.

### 3.1 Movie Plot Generation

The primary objective of this step is to generate diverse and compelling movie plots. Each movie plot comprises elements including overview, movie style, characters and frame descriptions, which also serves for the latter process. Here, to generate suitable textual plots, we enable GPT to adapt to our task through few-shot in-context learning with detailed prompts. Our ultimate goal is to generate consistent key frames. To achieve this, a detailed description is needed for each frame. Instinctively, all aforementioned generated textual information can be input into LLM 𝐺 with an instruction 𝐼 to gain the results at once. However, the consistency

###### Stage 1: Movie Plot Generation

Stage 3: Video Instruction Data Generation

[Figure 10]

“A love story about a duchess and a commoner ” Detailed Prompt

Frame Descriptions

A picture in Gothic style

- frame-1: In the ballroom, Rosamund is wearing a …
- frame-2:
- frame-3: …

###### Style

###### Overview

###### Characters

{Gothic}, ((Victorian elegance)), ((intricate detailing)), soft sepia, voluminous gowns, stiff collars, dimly lit rooms, lavish ballrooms, elaborative architecture, ......

The Duchess and the Commoner is a beautifully poignant chronicle of a scandalous romance that shakes Victorian England to its core. The story unfolds around Rosamund ......

Rosamund: a woman looks like Keira Knightley in lavender Thomas: a man looks like Hugh Jackman in grey

Text

- frame1
- frame2
- frame3 …

###### Story Expanding

expanding

expanding

###### Level-3

###### Level-2

###### Level-1

- Chapter 1: The Unraveling Corset Rosamund and Thomas's paths intertwine in an enchanting ballroom
- Chapter 2: …
- Chapter 3: … …

- Chapter 1: The Unraveling Corset

- Subchapter 1: The Grand Entrance ..
- Subchapter 2: The Unexpected Dance …

- Chapter 2: …
- Chapter 3: … …

- Chapter 1: The Unraveling Corset

- Subchapter 1: The Grand Entrance Frame-1: In the ballroom, Rosamund … Dialogue-1:"Let the show begin."
- Subchapter 2:…

- Chapter 2: … …

A picture in Gothic style: In the ballroom, Rosamund is wearing a …

QA Pairs QA Pairs QA Pairs

[Figure 11]

###### Embedding Lookup

###### Style-related Keywords

###### Stage 2: Style Immobilization Process

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

|A picture in style … Gothic||[Figure 17]|[Figure 18]|[Figure 19]| |
|---|---|---|---|
|[Figure 20]| |[Figure 21]| |
|[Figure 22]| |[Figure 23]| |
| | |[Figure 24]| |
<br><br>…|
|---|---|
| | |

[Figure 25]

[Figure 26]

###### Noise

pic

[Figure 27]

i st

… Go

Style {Gothic},….

gt noise

###### add noise

original embedding

[Figure 28]

[Figure 29]

replace

A picture in style

|[Figure 30]|[Figure 31]|[Figure 32]| |
|---|---|---|---|
|[Figure 33]| |[Figure 34]| |
|[Figure 35]| |[Figure 36]| |
| | |[Figure 37]| |

[Figure 38]

predict noise

| | |[Figure 39]| |
|---|---|---|---|

learned embedding

| |Learned| |
|---|---|---|
| |Style| |
| | | |

A picture in Gothic style

LLDM

[Figure 40]

Embedding

… … Gothic

[Figure 41]

[Figure 42]

Embedding Lookup

| | |[Figure 43]| |
|---|---|---|---|

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

###### …

[Figure 50]

Trainable Parameters

Forward Pass

1 2 3 4 5

Frames in Order

[Figure 51]

Frozen Parameters

Backward Pass

- Figure 2: Our proposed pipeline for generating video instruction tuning datasets. With merely a simple thematic description, our pipeline is capable of generating key frames of an entire film. The pipeline can be roughly divided into three stages: (a) movie plot generation, where we generate the whole movie plot based on a theme phrase. (b) style immobilization process, where we learn a style embedding to immobilize the style-related keywords generated from the plot into the latent space of the diffusion model, guiding it to generate frames with fixed style. (c) video instruction data generation, where we integrate all the previously obtained information to ultimately generate consistent key frames.

between each frame description can not be assured because of the long memory length limitation problem of LLMs. Generating all key frame descriptions simultaneously often results in insufficient context, leading to inconsistent and poor-quality text. To address this issue, we introduce a strategy using LLM to progressively expand the overview of the story into detailed frame descriptions. This expansion process is divided into three levels. Given the overview 𝑂 and instruction 𝐼, LLM 𝐺 is used to predict level-1 descriptions:

where 𝐹𝑖,𝑗,𝑘 denotes the current frame description, 𝑄𝐿3 denotes the corresponding QA pairs for level-3, 𝐷 denotes the corresponding dialogues and 𝑛 denotes the required length of the frame descriptions.

We then get totally 𝑙 ×𝑚 × 𝑛 frame descriptions and 𝑙𝑒𝑛(𝑄𝐿1) + 𝑙𝑒𝑛(𝑄𝐿2) + 𝑙𝑒𝑛(𝑄𝐿3) QA pairs.

### 3.2 Style Immobilization Process

𝐶1,𝐶2, ...,𝐶𝑖, ...,𝐶𝑙,𝑄𝐿1 = 𝐺 (𝑂,𝐼) (1) where 𝐶𝑖 denotes the current level-1 description, 𝑄𝐿1 denotes the

Now the issue lays in how to convert text information from the first stage into corresponding visuals. Simply employing a T2I model like stable diffusion for image generation does not guarantee consistency between visual frames. The key to maintaining consistency lies in ensuring stylistic uniformity throughout the movie. To achieve this, techniques like DreamBooth [28] are generally introduced. However, these methods usually need fine-tuning the entire model, which is clearly impractical when generating movies in bulk. Therefore, in light of these issues, we propose a method based on textual inversion to satisfy the requirements of both consistency and efficiency.

- corresponding QA pairs for level-1 and𝑙 denotes the required length of the descriptions.

Based on 𝐶𝑖, level-2 descriptions are then further predicted:

𝑆𝑖,1,𝑆𝑖,2, ...,𝑆𝑖,𝑗, ...,𝑆𝑖,𝑚,𝑄𝐿2 = 𝐺 (𝐶𝑖,𝑂,𝐼) (2) where 𝑆𝑖,𝑗 denotes the current level-2 description, 𝑄𝐿2 denotes the

- corresponding QA pairs for level-2 and 𝑚 denotes the required length of the descriptions.

Finally, based on 𝐶𝑖 and 𝑆𝑖,𝑗, desired frame descriptions are predicted:

Textual inversion is typically employed in conjunction with Latent Diffusion Models (LDMs). As is known, the goal of LDMs is to remove the noise added to a latent representation of an image.

𝐹𝑖,𝑗,1, 𝐹𝑖,𝑗,2, ..., 𝐹𝑖,𝑗,𝑘, ..., 𝐹𝑖,𝑗,𝑛,𝑄𝐿3,𝐷 = 𝐺 𝑆𝑖,𝑗,𝐶𝑖,𝑂,𝐼 (3)

The LDM loss is given by:

𝐿𝐿𝐷𝑀 = E𝑧∼E(𝑥),𝑦,𝜖∼N(0,1),𝑡 ∥𝜖 − 𝜖𝜃 (𝑧𝑡,𝑡,𝑐𝜃 (𝑦))∥22 (4)

where 𝑡 is the time step, 𝑧𝑡 is the latent noise, 𝜖 is the unscaled noise sample, 𝜖𝜃 is the denoising network, 𝑐𝜃 is the encoder for conditioning, 𝑦 is the conditioning input.

Assume 𝑐𝜃 is a text encoder, text input 𝑦 is first converted to tokens, which are indexes in some pre-defined dictionary. Each token is then linked to a unique embedding vector that can be obtained through an index-based lookup. These embedding vectors are typically learned as part of the text encoder 𝑐𝜃. The embedding space is the optimized objective of textual inversion. Now, in the first stage of DreamFrame, we generate a word that describes the style of the movie like “Gothic”. We use this word to represent the style concept we wish to learn. The learned embedding vector 𝑉∗ will be used to replace the associated vector of the word like “Gothic”. Then we can include the word that describes the style in a sentence like “Generate an image in Gothic style: ...” which follows by the actual content of the frame such as “In the ballroom, Rosamund is talking...”. In this way, the specific style word “Gothic” will “trigger” the learned style embedding as conditioning input to ensure that the content of the frames adhere to a consistent style while still allowing for content variation. Then by assigning a style-related keyword to each film and learning a style embedding associated with that keyword, the style consistency of the movie frames can be largely ensured.

To obtain the learned embedding as mentioned, a set of images (we use 10) which depicts the target style are needed. We generate those images using the movie style from stage one. The embedding related to the style is then learned through direct optimization, by minimizing the LDM loss of Equation (4) over images from the set. The optimization goal can be defined as:

𝑣∗ = argmin𝑣 E𝑧∼E(𝑥),𝑦,𝜖∼N(0,1),𝑡 ∥𝜖 − 𝜖𝜃 (𝑧𝑡,𝑡,𝑐𝜃 (𝑦))∥22 (5)

while 𝑐𝜃 and 𝜖𝜃 is fixed. This is a reconstruction task that motivate the learned embedding to capture fine visual details unique to the style concept.

Now, by our method, every movie can have their own stylerelated embedding through just minutes of training.

### 3.3 Video Instruction Data Generation

Based on previous two stages, we utilize the style embeddings to guide stable diffusion in generating key frames according to the key frame descriptions. More specifically, to generate key frames that are consistent in both characters and scenes, we initially replace character names in the frame description with corresponding celebrities (chosen by GPT-4). Following that, a style-immobilized embedding linked to a special token is utilized. This style embedding can serve as a condition to guide the stable diffusion model in generating scenes in a fixed style. This process is triggered by a special token, which denotes a specific style, such as “Gothic”. Hence, by incorporating a sentence like “generate an image in Gothic style:” at the beginning of our frame description, combined with character names, we can generate style-consistent key frames. We construct around 1K videos. Detailed distribution of our dataset, please refer to our supplementary material.

#### Table 1: The gain from tuning with our dataset is universal among model architectures. The best results are bold.

Model MvBench VideoBench TempCompass Avg. VideoChatGPT-7B [21] 32.7 38.5 42.4 37.9 VideoChatGPT-7B+Ours 35.8 41.6 45.7 41.0 VideoLLaVA-7B [21] 42.9 34.5 49.9 42.4 VideoLLaVA-7B+Ours 46.1 37.9 52.8 45.6 LLaMA-VID-7B [20] 41.5 36.5 44.2 40.7 LLaMA-VID-7B+Ours 45.1 40.1 48.1 44.4

4 EXPERIMENT

- 4.1 Setup

Datasets and Benchmarks. To thoroughly investigate how our high-quality video-caption dataset enhances the capabilities of LVLMs, we conduct comprehensive evaluations of the model across three multi-modal video benchmarks. MVBench [19] is designed to evaluate LVLMs on video tasks that go beyond single-frame understanding, comprising 4,000 QA pairs sourced from 11 public video datasets, such as TVQA and FunQA. VideoBench [26] aggregates a collection of approximately 15,000 QA pairs covering 10 evaluation aspects from 13 video question-answering datasets. TempCompass [23] evaluates the fine-grained temporal reasoning abilities of LVLMs, focusing on aspects like speed, direction, and attribute changes. It features 410 videos and 7,540 carefully curated instructions to highlight temporal understanding and interactive capabilities.

Implementation Details. We obtain our VLM model based on LLaMA-VID-7B [20] which provides basic abilities for video understanding (hereafter refered to as “DreamFrame”). LLaMA-VID basically consists of a visual encoder, a text decoder, a projector and a LLM. For visual encoder, LLaMA-VID use EVA-G [12] as its ViT-based backbone and the patch size is set to 14. For text decoder, Qformer-7b [10] is used in our experiment. For projector, onelayer MLP is used to transform the embedding into context token. For LLM, we use vicuna-7b. During the model training phase, we employed the original LLaMA-VID configuration as the foundation for our training process. We utilized 2 NVIDIA A100 GPUs. To conserve GPU memory, we employed deepspeed with zero3 during model training, disabling tf32 and opting for fp16. The model under all settings is trained with a standard AdamW optimizer for 1 epoch. The initial learning rate is set as 2e-5 and updated by a cosine decay scheduler with a batch of 4 per GPU. Visual encoder and text decoder are both freezed during the training. Specifically, we conduct tuning after the second stage of LLaMA-VID. All video frames are input at 224 resolution. Please refer to our supplementary material for more details.

- 4.2 Results on Video Understanding

Enhancing Current LVLMs with DreamFrame. We investigate the effectiveness of the video-QA data generated by DreamFrame to improve the performance of current LVLMs. VideoChatGPT [25], VideoLLaVA [21] and LLaMA-VID [20] are utilized as our base model. We further fine-tune their pre-trained models. For all three models, we only fine-tune the LLM component while keeping all

#### Table 2: Comparison with previous leading methods on MVBench. * denotes our evaluation results with the public checkpoints. The best results are bold and the second-best results are underlined.

Model AS AP AA FA UA OE OI OS MD AL ST AC MC MA SC FP CO EN ER CI Avg Otter-V-7B [16] 23.0 23.0 27.5 27.0 29.5 53.0 28.0 33.0 24.5 23.5 27.5 26.0 28.5 18.0 38.5 22.0 22.0 23.5 19.0 19.5 26.8 mPLUG-Owl-V-7B [33] 22.0 28.0 34.0 29.0 29.0 40.5 27.0 31.5 27.0 23.0 29.0 31.5 27.0 40.0 44.0 24.0 31.0 26.0 20.5 29.5 29.7 LLaMA-Adapter [36] 23.0 28.0 51.0 30.0 33.0 53.5 32.5 33.5 25.5 21.5 30.5 29.0 22.5 41.5 39.5 25.0 31.5 22.5 28.0 32.0 31.7 VideoChatGPT-7B [25] 23.5 26.0 62.0 22.5 26.5 54.0 28.0 40.0 23.0 20.0 31.0 30.5 25.5 39.5 48.5 29.0 33.0 29.5 26.0 35.5 32.7 VideoLLaMA-7B [35] 27.5 25.5 51.0 29.0 39.0 48.0 40.5 38.0 22.5 22.5 43.0 34.0 22.5 32.5 45.5 32.5 40.0 30.0 21.0 37.0 34.1 VideoChat-7B [18] 33.5 26.5 56.0 33.5 40.5 53.0 40.5 30.0 25.5 27.0 48.5 35.0 20.5 42.5 46.0 26.5 41.0 23.5 23.5 36.0 35.5 LLaMA-VID-7B* [20] 42.0 43.0 63.5 35.5 56.5 56.0 37.5 34.0 19.0 26.5 84.5 42.0 28.5 44.5 40.5 22.0 39.0 36.5 44.0 34.5 41.5 VideoLLaVA-7B* [21] 44.5 42.5 58.0 38.5 52.5 54.0 47.5 41.0 29.0 31.5 82.0 45.0 26.5 53.0 41.5 33.0 41.5 27.5 37.5 31.5 42.9

- DreamFrame-7B 48.0 46.0 67.0 39.5 59.0 58.5 41.5 36.5 25.5 30.5 87.5 45.5 30.5 48.5 44.5 26.0 42.5 40.0 46.5 37.5 45.1

Table 3: Comparison with previous leading methods on VideoBench. * denotes our evaluation results with the public checkpoints. The best results are bold and the second-best results are underlined.

Model

VEU PKQA CDM

Avg ANet MSVD MSRVTT TGIF YC2 UCF MOT TV MV NBA LE DM SQA3D

Video-LLaMA-7B [35] 39.9 41.2 34.1 31.3 28.9 27.6 16.7 24.8 32.4 26.2 60.6 49.1 31.2 32.8 mPLUG-Owl-7B [33] 41.5 42.5 36.3 31.7 27.1 22.8 27.8 24.0 30.2 25.1 33.3 51.0 32.0 33.2 Valley-7B [24] 38.1 32.0 28.0 31.4 29.1 20.3 11.1 23.7 32.6 31.3 41.7 56.5 33.3 34.0 VideoLLaVA-7B* [21] 44.3 34.2 30.3 39.4 30.7 19.5 22.4 27.1 33.4 25.5 33.5 50.6 39.1 34.5 ChatUniVi-7B [14] 49.0 48.6 41.7 41.3 29.0 28.3 16.7 23.1 33.6 25.7 38.9 53.1 29.1 35.3 VideoChat-7B [18] 44.6 42.2 37.4 33.7 27.7 22.4 27.8 26.2 34.1 28.6 39.9 55.4 31.4 35.4 LLaMA-VID-7B* [20] 45.2 44.5 39.1 29.1 29.3 27.9 11.1 34.1 32.5 28.9 36.1 47.8 36.8 36.5 Otter-7B [16] 44.3 55.0 47.0 34.3 32.7 22.4 16.7 27.7 37.1 34.3 52.8 48.7 29.7 37.5 PandaGPT-7B [8] 45.0 50.4 44.6 29.7 33.0 33.0 16.7 27.9 37.1 31.1 41.7 56.0 30.8 37.5 VideoChatGPT-7B [25] 46.6 57.5 46.3 35.6 34.8 24.1 27.8 28.8 36.5 22.5 41.7 58.2 37.2 38.5

- DreamFrame-7B 49.3 47.9 44.7 34.3 34.6 33.8 14.5 38.3 35.4 34.8 43.4 51.5 40.6 40.1

#### Table 4: Comparison with previous leading methods on TempCompass. The best results are bold and the second-best results are underlined.

Multi-Choice QA Yes/No QA Caption Matching Caption Generation

Avg.

Model

AC DI SP EV AT AC DI SP EV AT AC DI SP EV AT AC DI SP EV AT Valley-7B [24] 47.0 29.3 32.5 18.9 29.9 58.1 52.0 52.5 50.3 52.9 15.5 21.4 22.0 28.3 22.9 24.7 20.4 21.9 35.8 29.4 33.4 PandaGPT-13B [8] 35.5 27.8 29.3 31.8 30.9 53.0 49.6 50.8 53.7 52.2 56.6 51.4 44.3 55.0 49.0 23.7 25.7 26.0 29.8 32.6 40.4 VideoLLaMA-13B [35] 54.1 24.5 28.1 32.8 28.5 68.1 46.0 48.8 51.8 50.9 73.1 47.4 47.1 52.0 48.3 54.3 21.3 13.9 38.5 33.9 43.3 VideoChatGPT-7B [25] 47.0 31.6 28.4 37.1 30.9 52.5 50.0 49.5 51.0 50.0 64.6 48.6 47.8 49.3 48.6 40.9 28.4 24.5 31.8 33.9 42.4 mPLUG-Owl-7B [33] 66.6 29.3 32.2 34.8 35.4 64.4 50.6 51.2 51.3 52.0 56.9 45.3 46.4 49.3 49.0 46.5 28.2 30.4 31.2 36.5 44.5 VideoLLaVA-7B [21] 70.4 32.2 38.2 41.4 39.9 74.3 51.8 50.3 49.2 51.1 88.2 53.8 61.9 57.0 58.3 50.8 28.7 23.2 38.2 33.6 49.9 LLaMA-VID-7B [20] 58.6 29.9 29.3 30.5 26.0 63.0 48.8 49.2 48.4 52.7 72.7 45.6 52.2 49.0 49.0 53.0 28.0 21.9 35.5 35.9 44.2 DreamFrame-7B 61.3 32.7 33.5 34.1 30.2 66.3 52.9 55.3 53.4 55.1 75.6 49.1 56.5 52.8 53.9 57.2 32.8 25.4 40.1 39.7 48.1

#### Table 5: Ablation study. We study the effects of proposed strategies in our method.

Style Immobilization Story Expanding CLIP-Score ↑ LPIPS ↓

× ✓ 0.7175 0.7265 ✓ × 0.6902 0.5574 ✓ ✓ 0.8304 0.5316

other parts frozen. As shown in Table 1, our dataset consistently improves the alignment between video and language modalities in different LVLM architectures. Specifically, VideoLLaVA-7B achieves an average improvement of 3.1 across three comprehensive multimodal video benchmarks while VideoLLaVA-7B and LLaMA-VID7B achieve an average improvement of 3.2 and 3.7.

Comparison Results on Video Understanding. We compare our LVLM model DreamFrame-7B with other leading methods on

generate an image in Dramatic style: At the Town Square, a man looks like Gerard Butler in brown continues to stand holding the shield, looking around at the busy activities, with a serious face.

generate an image in Dramatic style: At the Town Square, a man looks like Gerard Butler in brown walking through the crowd, with the shield on his shoulder, his expression is firm.

generate an image in Dramatic style: In the blacksmith's shop, a man looks like Gerard Butler in brown is cleaning up, ready to end the day's work, his tired face lit by the last sparks jumping from the cooling forge.

generate an image in Dramatic style: At the Town Square, a man looks like Gerard Butler in brown is standing near a stone fountain, with a resolved expression.

generate an image in Dramatic style: In the blacksmith's shop, a man looks like Gerard Butler in brown is standing in front of the anvil with a tired yet satisfied smile.

generate an image in Dramatic style: In the blacksmith's shop, a man looks like Gerard Butler in brown is sitting on the stool near the anvil, his hands covered with soot. He looks exhausted but with a sense of satisfaction.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

With Style Immobilization

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

W/O Style Immobilization

- Figure 3: Qualitative results of ablation study on style immobilization. The results indicate that the ablation of the strategy will cause obvious style inconsistency.

three benchmarks as we mentioned. As shown in Table 2, DreamFrame outperforms the second-best method VideoLLaVA, by an average accuracy margin of 2.2% on MVBench. Similarly, Table 3 reports the results on VideoBench, where DreamFrame achieves a 1.6% improvement over the second-best method VideoChatGPT. Results on TempCompass are presented in Table 4. DreamFrame consistently performs favorably across most evaluation aspects, notably achieving the highest accuracy in all five sub-tasks under Caption Generation. Although DreamFrame ranks second in terms of overall average accuracy on TempCompass, it trails VideoLLaVA by only 1.8%, indicating that the gap is minimal. Overall, these results demonstrate that DreamFrame obtains competitive performance across all three multi-modal video benchmarks, indicating its strong capacity for comprehensive video understanding.

### 4.3 Ablation Study

To verify the effectiveness of our proposed strategies in DreamFrame, we conduct an ablation study on two strategies: Story Expanding strategy, which is designed to ensure semantic consistency across frame descriptions, thereby preserving semantic consistency between generated frames and Style Immobilization strategy, which is aimed at maintaining visual style consistency between adjacent frames. To evaluate these aspects, we use CLIP-Score [13] to assess semantic consistency between image frames and LPIPS [37] to measure visual style consistency. As is shown in the second row of Table 5, the ablation of Story Expanding results in a significant drop of approximately 0.14 in CLIP-Score, while only causing a

minor decrease of about 0.02 in LPIPS. This indicates that Story Expanding is a crucial component for ensuring semantic consistency across frames. The first row shows that removing Style Immobilization leads to a substantial performance degradation in LPIPS by around 0.17, highlighting its importance in maintaining visual style consistency. The qualitative results from Figure 3 also shows that the ablation of Style Immobilization leads to severe visual style inconsistency across frames.

### 5 CONCLUSIONS

In this paper, we propose an effective framework, DreamFrame, for generating style-consistent keyframes along with corresponding question–answer pairs, which can be used as tuning data for video understanding tasks. Extensive experiments validate the effectiveness of our approach. Beyond generating tuning dataset for video understanding, DreamFrame also holds potential for broader applications such as comic strip generation, artistic asset creation, and other domains that benefit from coherent multi-frame content generation. As an initial exploration of fully automatic training data generation, we hope that DreamFrame can inspire future research in this emerging field.

### REFERENCES

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikoł aj Bińkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. 2022. Flamingo: a Visual Language Model for Few-Shot Learning. In Advances in Neural Information Processing Systems, Vol. 35. 23716–23736.
- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenhang Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, K. Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Yu Bowen, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xing Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen Technical Report. ArXiv abs/2309.16609 (2023).
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond.
- [4] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles.

2015. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 961–970.

- [5] Santiago Castro, Mahmoud Azab, Jonathan Stroud, Cristina Noujaim, Ruoyao Wang, Jia Deng, and Rada Mihalcea. 2020. LifeQA: A Real-life Dataset for Video Question Answering. In Proceedings of the Twelfth Language Resources and Evaluation Conference. 4352–4358.
- [6] Santiago Castro, Naihao Deng, Pingxuan Huang, Mihai Burzo, and Rada Mihalcea. 2022. In-the-Wild Video Question Answering. In Proceedings of the 29th International Conference on Computational Linguistics. 5613–5635.
- [7] David Chen and William Dolan. 2011. Collecting Highly Parallel Data for Paraphrase Evaluation. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. 190–200.
- [8] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, MingHsuan Yang, et al. 2024. Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers. arXiv preprint arXiv:2402.19479 (2024).
- [9] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. 2023. Intern VL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023), 24185–24198.
- [10] Wenliang Dai, Junnan Li, DONGXU LI, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. 2023. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. In Advances in Neural Information Processing Systems, Vol. 36. 49250–49267.
- [11] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Jun-Mei Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiaoling Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bing-Li Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dong-Li Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Jiong Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, M. Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, Ruiqi Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shao-Kang Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wen-Xia Yu, Wentao Zhang, Wangding Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyu Jin, Xi-Cheng Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yi Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yu-Jing Zou, Yujia He, Yunfan Xiong, Yu-Wei Luo, Yu mei You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yao Li, Yi Zheng, Yuchen Zhu, Yunxiang Ma, Ying Tang, Yukun Zha, Yuting Yan, Zehui Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhen guo Zhang,

- Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zi-An Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. ArXiv abs/2501.12948 (2025).
- [12] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. 2023. EVA: Exploring the Limits of Masked Visual Representation Learning at Scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19358–19369.
- [13] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi.

2021. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wentau Yih (Eds.). Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 7514–7528.

- [14] Peng Jin, Ryuichi Takanobu, Caiwan Zhang, Xiaochun Cao, and Li Yuan. 2023. Chat-univi: Unified visual representation empowers large language models with image and video understanding. arXiv preprint arXiv:2311.08046 (2023).
- [15] Yang Jin, Zhicheng Sun, Kun Xu, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, Kun Gai, and Yadong Mu.

2024. Video-LaVIT: Unified Video-Language Pre-training with Decoupled VisualMotional Tokenization. In International Conference on Machine Learning. PMLR, 22185–22209.

- [16] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. 2023. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726 (2023).
- [17] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. In International Conference on Machine Learning. PMLR, 19730– 19742.
- [18] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. 2023. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355 (2023).
- [19] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. 2023. MVBench: A Comprehensive Multi-modal Video Understanding Benchmark. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023), 22195– 22206.
- [20] Yanwei Li, Chengyao Wang, and Jiaya Jia. 2024. LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models. In ECCV (46). 323–340.
- [21] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2024. Video-LLaVA: Learning United Visual Representation by Alignment Before Projection. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 5971–5984.
- [22] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual Instruction Tuning. In Advances in Neural Information Processing Systems, Vol. 36. 34892–34916.
- [23] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. 2024. TempCompass: Do Video LLMs Really Understand Videos?. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, 8731–8772.
- [24] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. 2023. Valley: Video assistant with large language model enhanced ability. arXiv preprint arXiv:2306.07207 (2023).
- [25] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. 2024. Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 12585–12602.
- [26] Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. 2023. Video-Bench: A Comprehensive Benchmark and Toolkit for Evaluating Video-based Large Language Models. ArXiv abs/2311.16103 (2023).
- [27] OpenAI. 2023. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774 (2023).
- [28] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22500–22510.
- [29] Xingwu Sun, Yanfeng Chen, Yiqing Huang, Ruobing Xie, Jiaqi Zhu, Kai Zhang, Shuaipeng Li, Zhen Yang, Jonny Han, Xiaobo Shu, Jiahao Bu, Zhongzhi Chen, Xuemeng Huang, Feng Lian, Saiyong Yang, Jianfeng Yan, Yuyuan Zeng, Xiaoqing Ren, Chao Yu, Lulu Wu, Yue Mao, Jun Xia, Tao Yang, Suncong Zheng, Kan Wu, Dian Jiao, Jinbao Xue, Xipeng Zhang, Decheng Wu, Kai Liu, Dengpeng Wu, Guanghui Xu, Shaohua Chen, Shuang Chen, Xiaowei Feng, Yigeng Hong, Junqiang Zheng, Chengcheng Xu, Zong-Rui Li, Xi Kuang, Jian hua Hu, Yiqi Chen, Yuchi Deng, Guiyang Li, Ao Liu, Chenchen Zhang, Shi-He Hu, Zilong Zhao, Zi-Hao Wu, Yao Ding, Weichao Wang, Han Liu, Roberts Wang, Haoyang Fei, Peijie Yu, Ze Zhao, Xun Cao, Hai Wang, Fusheng Xiang, Meng-Sheng Huang,

- Zhiyu Xiong, Bin Hu, Xue yan Hou, Lei Jiang, Jia bing Ma, Jiajia Wu, Yaping Deng, Yi Shen, Qian Wang, Weijie Liu, Jie Liu, Meng Chen, Liang Dong, Wei Jia, Hu Chen, Feifei Liu, Ruixin Yuan, Huilin Xu, Zhenxiang Yan, Tengfei Cao, Zhichao Hu, Xinhua Feng, Dong Du, Ting-Ting Yu, Yang-Dan Tao, Feng Zhang, Jianchen Zhu, Chengzhong Xu, Xirui Li, Chong Zha, Ouyang Wen, Yi Xia, Xiang Li, Ze He, Rongpeng Chen, Jiawei Song, Ruibin Chen, Fan Jiang, Chongqing Zhao, and Bo Wang. 2024. Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent. ArXiv abs/2411.02265 (2024).
- [30] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).
- [31] Junbin Xiao, Pan Zhou, Angela Yao, Yicong Li, Richang Hong, Shuicheng Yan, and Tat-Seng Chua. 2023. Contrastive Video Question Answering via Video Graph Transformer. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 11 (2023), 13265–13280. https://doi.org/10.1109/TPAMI.2023.3292266
- [32] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. 2016. MSR-VTT: A Large Video Description Dataset for Bridging Video and Language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5288–5296.
- [33] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. 2023. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint

- arXiv:2304.14178 (2023).
- [34] Yan Zeng, Xinsong Zhang, Hang Li, Jiawei Wang, Jipeng Zhang, and Wangchunshu Zhou. 2024. X22-VLM: All-in-One Pre-Trained Model for Vision-Language Tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence 46, 5 (2024), 3156–3168. https://doi.org/10.1109/TPAMI.2023.3339661
- [35] Hang Zhang, Xin Li, and Lidong Bing. 2023. Video-LLaMA: An Instructiontuned Audio-Visual Language Model for Video Understanding. In Proceedings of the Conference on Empirical Methods in Natural Language Processing: System Demonstrations. 543–553.
- [36] Renrui Zhang, Jiaming Han, Chris Liu, Aojun Zhou, Pan Lu, Yu Qiao, Hongsheng Li, and Peng Gao. 2024. LLaMA-Adapter: Efficient Fine-tuning of Large Language Models with Zero-initialized Attention. In International Conference on Learning Representations.
- [37] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang.

2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (2018), 586–595.

- [38] Luowei Zhou, Chenliang Xu, and Jason J. Corso. 2017. Towards Automatic Learning of Procedures From Web Instructional Videos. In AAAI Conference on Artificial Intelligence.

