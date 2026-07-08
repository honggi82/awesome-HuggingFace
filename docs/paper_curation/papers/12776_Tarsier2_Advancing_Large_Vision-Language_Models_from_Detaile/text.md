## Tarsier2: Advancing Large Vision-Language Models from Detailed Video Description to Comprehensive Video Understanding

# arXiv:2501.07888v3[cs.CV]24Jan2025

Liping Yuan∗ Jiawei Wang∗ Haomiao Sun∗ Yuchen Zhang∗ Yuan Lin† ByteDance Research

{yuanliping.0o0,wangjiawei.424,sunhaomiao,zhangyuchen.zyc,linyuan.0}@bytedance.com

Project Site: https://github.com/bytedance/tarsier

###### Abstract

We introduce Tarsier2, a state-of-the-art large vision-language model (LVLM) designed for generating detailed and accurate video descriptions, while also exhibiting superior general video understanding capabilities. Tarsier2 achieves significant advancements through three key upgrades: (1) Scaling pre-training data from 11M to 40M video-text pairs, enriching both volume and diversity; (2) Performing fine-grained temporal alignment during supervised fine-tuning; (3) Using model-based sampling to automatically construct preference data and applying DPO training for optimization. Extensive experiments show that Tarsier2-7B consistently outperforms leading proprietary models, including GPT-4o and Gemini 1.5 Pro, in detailed video description tasks. On the DREAM-1K benchmark, Tarsier2-7B improves F1 by 2.8% over GPT-4o and 5.8% over Gemini-1.5-Pro. In human side-by-side evaluations, Tarsier2-7B shows a +8.6% performance advantage over GPT-4o and +24.9% over Gemini-1.5-Pro. Tarsier2-7B also sets new stateof-the-art results across 15 public benchmarks, spanning tasks such as video question-answering, video grounding, hallucination test, and embodied question-answering, demonstrating its versatility as a robust generalist vision-language model.

[Figure 1]

Benchmark Previous SOTA

DREAM-1K[105] Tarsier-7B[105] MVBench[57] InternVL2.5-8B[20] TVBench[25] IXC-2.5 7B[124] TOMATO[94] Qwen2-VL-7B[106] Vinoground[123] LLaVA-OV-7B[53] TempCompass[69] Qwen2-VL-7B[106] Video-MME[31] NVILA-7B[70] LongVideoBench[110] Apollo-7B[130] TemporalBench[12] LLaVA-Video-7B[127] MLVU[128] InternVL2.5-8B[20] MMBench-Video[30] MiniCPM-V-2.6 [119] VideoHallucer[109] Qwen2-VL-7B[106] EventHallusion[122] Tarsier-7B[105] E.T. Bench[67] E.T. Chat[67]

Figure 1: Performance comparison of Tarsier2 with previous SOTA models at 7B-scale and GPT4o. We report the overall average scores for benchmarks with multiple subtasks/metrics.

∗Equally contributed. †Corresponding author.

### Contents

- 1 Introduction 3
- 2 Related Work 5
- 3 Approach 6

- 3.1 Pre-training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.2 Supervised fine-tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.3 Direct Preference Optimization . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 4 Experiments 11

- 4.1 Quantitative Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 4.1.1 Video Captioning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 4.1.2 Short-Video Question Answering . . . . . . . . . . . . . . . . . . . . 13
- 4.1.3 Long-Video Question Answering . . . . . . . . . . . . . . . . . . . . 15
- 4.1.4 Hallucination . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 4.1.5 Video Grounding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 4.1.6 Embodied Question Answering . . . . . . . . . . . . . . . . . . . . . 16

- 4.2 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- 4.2.1 Pre-training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 4.2.2 SFT . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 4.2.3 DPO . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- 4.3 Video Recaptioning using Tarsier2 . . . . . . . . . . . . . . . . . . . . . . . 20

- 5 Conclusion 21

- A Training hyper-parameters 36
- B Public datasets of pre-training stage 36
- C Annotation process for SFT data 36
- D Detail setting of DPO training 37
- E Detailed results of individual datasets at different stages 38
- F Tarsier2-Recap-585K Data Composition 38
- G Qualitative Comparison of the SFT Process 40
- H DREAM-1K cases 41

### 1 Introduction

With the rapid advancements in large vision-language models (LVLM) [21, 56, 61, 62, 105, 106], significant progress has also been made in video understanding. Leading proprietary models, such as GPT-4o [41] and Gemini-1.5-Pro [102], have achieved state-ofthe-art (SOTA) performance across a variety of video understanding tasks. Additionally, several open-source models [61, 114, 23, 52, 20, 53, 23] also demonstrate strong performance on several video understanding benchmarks [25, 57, 67, 109, 128], although they still lag behind proprietary models, particularly in complex, open-ended generation tasks. Despite these advancements, current models remain behind human-level video understanding [78, 86, 19], mainly due to persistent challenges such as accurately perceiving temporal dynamics, spatial-temporal reasoning, and model hallucinations.

In this paper, we introduce Tarsier2, a 7B-parameter LVLM model that can outperform both GPT-4o and Gemini-1.5-Pro in generating detailed video descriptions, a fundamental challenge in video understanding. Beyond video description generation, Tarsier2 also achieves SOTA performance across various video question-answering (VQA) benchmarks at the same model size, surpassing or closely matching the performance of proprietary models on these VQA benchmarks. Figure 1 provides a comprehensive comparison between Tarsier2, GPT-4o and previous SOTA results for open-source LVLMs with the same scale. Figure 2 presents examples illustrating Tarsier2’s video understanding capability across different tasks.

Tarsier2 employs a simple model architecture consisting of a vision encoder, a vision adaptor, and a large language model (LLM). We meticulously design a three-stage training procedure: pre-training, supervised fine-tuning (SFT), and reinforcement learning (RL). In comparison with Tarsier [105], Tarsier2 features several key improvements that significantly enhance its performance:

- • We scale up the pre-training dataset from 11 million to 40 million video-text pairs, addressing the challenge posed by the scarcity of high-quality video-text data. To achieve this, we implement meticulous filtering and sourcing. Specifically, we collect 11 million commentary videos, featuring explanations and analyses of movies and TV shows, providing rich contextual information to greatly enhance video understanding. Our experiments confirm that increasing the volume of pre-training data consistently improves model performance.
- • We construct a video description dataset containing 150K instances, each including a detailed video description along with the specific frames corresponding to each event described. During the SFT stage, we involve this dataset to provide the model with supervision on temporal fine-grained alignment. Experimental results show that, compared with traditional video-caption alignment training, this approach significantly improves accuracy in video description and reduces the hallucinations.

General Video Question Answering

Video Grounding

[Figure 2]

||[Figure 3]<br><br>[Figure 4]|
|---|
<br><br>[Figure 5]<br><br>|[Figure 6]<br><br>[Figure 7]|
|---|
<br><br>[Figure 8]<br><br>[Figure 9]<br><br>video<br><br>… …<br><br>0.0s 49.2s 53.5s 116.7s 122.7s 130.0s<br><br>Instruction<br><br>Detect and localize all the video segments containing the given action ‘baseball pitch’, and provide the outputs using start and end timestamps.|
|---|

[Figure 10]

|video<br><br>Instruction<br><br>Now pretend to be a broadcaster and provide a live commentary on this diving event.<br><br>[Figure 11]<br><br>[Figure 12]|
|---|

[Figure 13]

|The action happens in 49.2 - 53.5, and 116.7 - 122.5 seconds.<br><br>Response|
|---|

Multi-video Question Answering

Detailed Video Description

[Figure 14]

[Figure 15]

|video 1<br><br>Instruction<br><br>[Question] Which video is matching with the following caption? [Caption] A person first uses a needle to poke the yarn through the fabric before he ties a knot with the yarn. [Options]<br><br>- A. The first video.<br>- B. The second video.<br><br><br>video 2<br><br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]|
|---|

|video<br><br>Instruction<br><br>Please describe the video in detail.<br><br>[Figure 20]<br><br>[Figure 21]|
|---|

[Figure 22]

[Figure 23]

|Good evening, ladies and gentlemen, and welcome to the ‘2024 National Youth Diving Championship’! I am your commentator. Look, the diver is seen jumping off the platform, but his movements are not stable. He appears to be struggling to maintain his balance as he ascends. As he reaches the peak of his jump, his body starts to rotate. The diver's movements become more erratic as he descends towards the water. Unfortunately, he fails to execute a proper dive, and his entry into the water is quite rough, creating a huge splash.<br><br>Response|
|---|

|In a room filled with various plants, scientific equipment, and an easel, a woman in a blue dress and apron stands at a workbench. She turns around and throws several stones onto the floor. A woman in a blue coat stands with her arms crossed, observing the stones. The woman in the blue dress steps on one of the stones, causing it to emit a bright blue light. The light spreads from the stone to other stones. The camera follows the light and reveals a smiling man in a purple suit looking at the stones.<br><br>Response|
|---|

Embodied Question Answering

[Figure 24]

|video<br><br>Instruction<br><br>What can I use to find where a particular book is located?<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>1 2 3 4 5 6 7 8|
|---|

[Figure 33]

|[Answer] A. The first video. [Reason] In the second video, the person ties a knot with the yarn firstly and then uses a needle to poke the yarn through the fabric.<br><br>Response|
|---|

[Figure 34]

|Use the catalog on the computer.<br><br>Response|
|---|

Video Description (frame-by-frame)

Video Description (Chinese)

[Figure 35]

[Figure 36]

|video<br><br>Instruction 用中文写一段100字左右的视频描述。<br><br>[Figure 37]<br><br>[Figure 38]|
|---|

|video<br><br>Instruction<br><br>Create a video description by outlining each frame thoroughly. Use simple words and concise sentences. Also, include observations about the video's artistic style.<br><br>[Figure 39]<br><br>[Figure 40]|
|---|

[Figure 41]

|一群角色站在一个巨大的猫科动物前面。一个穿着皮毛的角色在左边，两个穿着条纹和黄色衣 服的角色在中间，一个穿着蓝色衣服的角色在右边。接着，镜头切换到一个头发蓬松的角色， 他手持一根棍子站在高处，表情从惊讶变为愤怒，然后开始挥舞棍子。<br><br>Response|
|---|

Video Description (concise)

[Figure 42]

|video<br><br>Instruction<br><br>用中文写一段100字左右的视频描述。<br><br>[Figure 43]<br><br>Please provide a concise description for this video.<br><br>[Figure 44]|
|---|

[Figure 45]

|Frame 1: A white dog with goggles and a green helmet sits on a red object against a blue sky. Frame 2: The same dog, still sitting, with a slightly different background. Frame 3: A red biplane flies over a bridge with a train below. Frame 4: The biplane continues to fly over the bridge, moving to the right. Frame 5: The biplane is further to the right, still over the bridge. Frame 6: The biplane is almost off the bridge, moving to the right. Frame 7: The white dog is back, sitting on the red object with a mountain background. Frame 8: The dog is now leaning forward, looking determined. Frame 9: A pink dog with a purple scarf is in a red biplane, with the white dog in the distance. Frame 10: The pink dog is closer to the white dog, who looks surprised. Frame 11: The pink dog reaches out towards the white dog. Frame 12: The white dog reaches back towards the pink dog. Frame 13: Close-up of the pink dog looking worried. Frame 14: The pink dog is hanging off the back of the biplane. Frame 15: A purple scarf is falling through the air. Frame 16: The screen is completely black. The artistic style is colorful and cartoonish, with smooth animations and vibrant backgrounds.<br><br>Response|
|---|

[Figure 46]

|The camera focuses on a dark, industrial interior with chains. It then shifts to a snowy exterior, revealing penguins outside the window.<br><br>Response|
|---|

###### Figure 2: Overview of Tarsier2 capabilities. Based on its strong ability for detailed video description, Tarsier2 excels in a variety of video-centric tasks. Click the play buttons to view the videos.

• To further enhance model performance, we use the model to generate samples that automatically construct preference data for DPO training [89]. To ensure high-quality preference data, we propose two methods: a negative sampling technique that uses corrupted videos to generate negative samples for preference pairs, and a preference data filtering method that employs AutoDQ [105] to automatically filter out pairs with minimal differences. Our experiments show that DPO training on these automatically generated preference data leads to continued performance improvements over the SFT stage.

We conduct extensive experiments to evaluate Tarsier2 against both proprietary and open-source LVLMs. For video description, Tarsier2 outperforms all other models, surpassing both proprietary and open-source LVLMs in evaluations on DREAM-1K [105] and E.T. Bench-Captioning [67]. In human side-by-side evaluations, Tarsier2-7B shows a +7.8% improvement over GPT-4o and a +12.3% advantage over Gemini-1.5-Pro. It also significantly outperforms the leading open-source model, Tarsier-34B, with a +51.4% advantage. Furthermore, Tarsier2-7B proves to be a versatile generalist model, setting new SOTA results on public benchmarks for video question-answering [25, 94, 123], hallucination test [122], video grounding [67] and embodied QA [93]. Finally, we present extensive ablation studies to identify the key factors contributing to the model’s strong performance. We also release a recaptioning dataset, Tarsier2-Recap-585K, and demonstrate its effectiveness in enhancing the capabilities of existing LVLMs for video description and general video understanding.

### 2 Related Work

Video-LLMs Recently, research on Video LLMs has surged [56, 76, 75, 121, 61, 6, 104, 114, 52, 62, 127, 106, 54, 27, 2, 72, 20, 130], with efforts focusing on model architectures and video-text data collection. On the architecture side, current studies emphasize visual representation [114, 106, 130], visual token resampling [114, 20, 115, 58], and the integration of Vision Transformers (ViT) with LLMs [106, 55, 65, 8]. Tarsier2 adopts a simple architecture composed of a visual encoder, a visual adaptor, and an LLM. Despite its simplicity, we demonstrate that a meticulously designed training strategy enables Tarsier2 to achieve strong video understanding capabilities.

In terms of video-text data, while many efforts aim to collect datasets for training Video LLMs, their quantity and quality remain limited. For example, LLaVA-Video [127] is trained on just 1.3 million video-text pairs, and several open-source models, such as InternVL2.5 [20], Aria [54], and VILA-1.5 [62], are trained on fewer than 5 million pairs. Although larger datasets like HowTo100M [81], HD-VILA [116], Panda-70M [18], and InternVid-10M [108] exist, they either cover limited domains or contain overly simplistic or low-quality text. Furthermore, some studies do not disclose the volume of video data used [106, 130, 27, 54].

To address these challenges, our work focuses on improving the quantity and quality of video-text data. We newly collected 20 million video-text pairs, spanning a wide range of video genres. In total, 40 million pairs are used in the final pre-training stage. Additionally, we annotated 150K fine-grained video descriptions for the SFT stage.

Video Description Video description, a foundational task in video understanding, has long been a central focus of research. Early work [112, 117, 17] typically involved pretraining video-language models and fine-tuning them on datasets such as MSVD[14], MSRVTT[113], and VATEX[107], which provide single-sentence video summaries.

Recent advancements in LVLMs have improved video description, enabling more detailed outputs beyond simple summarization. However, generating comprehensive video descriptions presents challenges beyond model architecture. While multi-frame processing and temporal modeling are crucial, large-scale and rich annotated ¡video, description¿ datasets are equally important. Existing alignment datasets, such as HD-VILA [116] and HoTo100M [81], provide concise descriptions, limiting detailed video understanding. To address this, datasets such as ShareGPT4Video[16] uses a pipeline where LVLMs (e.g., GPT-V[5]) annotate frames, and LLMs (e.g., GPT-4[1]) aggregate them. This improves detail but often leads to verbosity and hallucinations. Recent works [127, 99] uses proprietary Video-LLMs, such as GPT-4o[41] and Gemini-1.5[102], for annotation, but their high cost limits application to smaller datasets.

For Tarsier2, we collect a large dataset of video-text pairs. In particular, we automatically build meaningful video-text pairs from online commentary videos. These commentaries include both low-level (atomic actions) and high-level (plot) visual elements, enhancing the model’s understanding across various granularity. In addition to data collection, Tarsier2 also uses a meticulously designed three-stage training process, where DPO training after SFT further refines description accuracy and detail.

### 3 Approach

We initialized Tarsier with Qwen2-VL[106] weights and employed a three-stage training strategy. First, we pre-trained Tarsier2 on 40 million large-scale video-text pairs. Next, we fine-tuned the model on moderate-sized, curated, human-annotated datasets in two phases: one targeting video descriptions with fine-grained grounding and the other focusing on natural, instruction-following video descriptions. Finally, we applied Direct Preference Optimization[89] using automatically generated preference data to further enhance the quality of the video descriptions. The training process is detailed below; for a comprehensive list of hyper-parameters, please refer to Appendix A.

[Figure 47]

Figure 3: Summary of datasets used in the pre-training stage of Tarsier2.

#### 3.1 Pre-training

The pre-training stage encompasses a variety of tasks, including video captioning, video question answering, action recognition, action grounding, (multi-)image understanding, and text generation. The training data consists of 20 million public datasets and 20 million newly collected in-house datasets. Figure 3 illustrates the composition of the pre-training data, with a detailed breakdown presented in Appendix B. Our findings indicate that the in-house data significantly enhances model’s performance, complementing the public datasets. In the following, we describe the pipeline used for in-house data collection.

We collected a large group of videos from the Internet, spanning diverse genres such as animation, movies, TV series, short videos, stock footage, games and so on. The videos are categorized into three types:

- • Short videos with captions. This category consists of 2.4 million videos directly sourced from the Internet, preserving their original video-caption pairs.
- • Commentary videos for movies or TV shows. The videos were segmented into single-shot clips using PySceneDetect1. A filtering model removed static or

1https://www.scenedetect.com/

low-quality clips. Adjacent clips were then merged to create continuous segments, ensuring final video durations ranged from 2 to 30 seconds. We utilized an internal OCR tool to extract the commentary text from the video and use it as the caption. The areas containing the commentary text in the video were obscured. To ensure relevance, we trained a lightweight BERT-style[45] model to filter out clips where the commentary lacked direct visual correspondence (e.g., character dialogues). This process produced 11.0 million video clips.

• Other videos. These videos were processed similarly to the commentary videos, undergoing segmentation into shorter clips, filtering out low-quality clips, and merging adjacent clips. After this, we employed a multi-modal LLM to automatically generate video captions and question-answer pairs, resulting in a total of 2.7 million clips.

Commentary videos represent a significant portion of the pre-training data. Unlike traditional video-text datasets, such as HowTo100M [81], which rely on ASR transcripts, commentary data demonstrates stronger alignment between video and text. This commentary not only describes low-level visual elements, such as atomic actions, but also highlights high-level information like plot details. This type of data can substantially enhance the model’s visual understanding at varying levels of granularity.

In addition to video caption data, we incorporate large-scale synthetic datasets for tasks such as object tracking, frame order prediction, image retrieval, video question-answering, and image captioning during pre-training.

Overall, our pre-training dataset consists of 40 million samples. We trained Tarsier2 on this dataset using 128 H100 GPUs, with all components of Tarsier2 set to be trainable. For each video, we sampled between 16 and 128 frames, depending on its duration. In total, the pre-training stage of Tarsier2 processed approximately 200 billion tokens.

#### 3.2 Supervised fine-tuning

During the SFT phase, our primary objectives are to further improve the model’s accuracy and comprehensiveness in video descriptions and ensure the outputs are human-like: well-structured, appropriately detailed, and capable of generating accurate long-form descriptions. To achieve this, we collected 150K video clips and conducted SFT in two stages.

In the first stage, each video clip in the SFT dataset is annotated with a detailed description with fine-grained temporal grounding. As shown in Figure 4, the annotations specify the frames corresponding to each event in the description. The annotation process is detailed in Appendix C. This fine-grained frame-event alignment enhances the model’s ability to accurately identify and describe events by focusing on temporal and visual cues, complementing traditional video-caption alignment. Our experiments demonstrate that this approach mitigates the omission of key events in generated video descriptions.

[Figure 48]

Figure 4: An example of a video description with fine-grained temporal grounding. “<frame: i-j >” indicates that the following event is inferred from frames i to j. Events are distinguished by color, with corresponding frames and descriptions marked in the same color to indicate their association.

In the second stage of SFT, we refined the model’s output to achieve a more humanlike style. We observed that the data used in the initial stage of SFT often fragmented complete events into multiple steps due to event-grounding requirements. For instance, the action of pouring wine might be divided into steps like opening the bottle, lifting it, and pouring. To address this, we incorporated more natural and human-like video description data. Specifically, in this stage, we designed diverse description instructions to reflect realworld variations in language, granularity, and style requirements. We then annotated each video’s description to align with its corresponding instruction, as detailed in Appendix C. This data allowed the model to better interpret varying instructions and generate more accurate and diverse video descriptions.

The training data for SFT-1 contains 150k video description pairs, while SFT-2 comprises 50k diverse instructions and 150k refined video-description pairs. Each pair includes a video description aligned with one of the instructions. We trained Tarsier2 on this dataset using 32 H100 GPUs and set all components of Tarsier2 to trainable. For each video, we sampled 16 frames for training. The global training batch size was set to 64, and Tarsier2 was trained for 5000 iterations in each of the two phases. In addition, we used 2e-5 and 2e-6 as the learning rate of the model during the two-stage SFT respectively to obtain further performance improvement.

#### 3.3 Direct Preference Optimization

In this subsection, we introduce a novel automated method for collecting preference data for video description. By performing DPO [89] training on this data, we can further improve the model’s ability to generate high-quality, detailed video descriptions.

Negative sampling Existing works often conduct multiple times sampling on the same input (video and text prompt) to acquire preference pair candidates[111, 126, 100]. In

|A woman is walking a dog towards a back car with open door, the dog jumps into the car, and the scene switches to a hand petting the dog under its chin a few times before pulling back.<br><br>Ground Truth<br><br>DQR: 2/4 DQP: 2/4<br><br>DQR: 3/4 DQP: 3/3<br><br>AutoDQ Scorer<br><br>positive response<br><br>negative response<br><br>[Figure 49]<br><br>><br><br>Preference Pair Filtering<br><br>Chosen Rejected<br><br>[Figure 50]<br><br>[Figure 51]|
|---|

|[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>A woman, wearing a gray jacket and black pants, walks a dog on a leash towards a black car with an open door. The dog, wearing a green harness, jumps into the car. Inside the car, a person is seen petting the dog's chin. The dog looks up at the person while being petted.<br><br>User<br><br>Raw video<br><br>1<br><br>Describe the video in detail.<br><br>Positive Sampling<br><br>Positive response<br><br>2 3 4 5|
|---|

|A woman, dressed in a light-colored jacket and black pants, walks a dog on a leash towards a black car with an open door. The woman squats down to pet the dog under its chin. After petting, the woman stands up and the dog jumps into the car. The woman follows, holding the leash.<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>Describe the video in detail. Corrupted video<br><br>Negative Sampling<br><br>Negative response<br><br>[Figure 68]<br><br>User<br><br>[Figure 69]<br><br>1 4 5 2 3|
|---|

Figure 5: Preference data construction pipeline for DPO training.

practice, however, we found that 1) Low-temperature sampling produces minimal variation in responses; 2) High-temperature sampling often leads to uncontrollable or abnormal generations. To address these issues, we propose a new automated preference data collection approach that enhances controllability and consistently yields high-quality preference data.

In reinforcement learning (RL) terms, the VLM serves as a policy model πθ, typically initialized from the SFT model. Given an input prompt x, consisting of N frames sampled from a video, πθ generates an video description y. Then, the video frames are modified to produce a corrupted prompt x˜ through one of the following perturbations:

- • Clip-switching: Evenly divide the video into 4 clips, then randomly choose 2 clips and swap their order.
- • Clip-reversing: A random clip with N2 ∼ N frames is reversed.

- • Clip-cropping: N frames are resampled from a random clip with half of the video’s original duration.
- • Down-sampling: Half of the N frames are randomly dropped.

The corrupted prompt x˜ is input into πθ, generating a new description y˜. The resulting preference data is represented as {x,yw = y,yl = y˜}. The first two perturbations are designed to induce negative descriptions with temporal errors, while the latter two are

designed to induce incomplete descriptions. Consequently, through DPO training, the model can be enhanced to produce descriptions with improved accuracy and completeness.

Figure 5 provides an example to illustrate the preference data construction pipeline. From a raw video, we first generate a positive response using the current model. Next, a corrupted video, created through clip-switching, is fed into the model to obtain a negative sample, which contains two hallucinations (highlighted in red).

Preference data filtering Given a prompt x, response y˜ is generally more negative compared to y. However, an effective filter mechanism for valid preference data remains essential, as y˜ is not always strictly worse than y2. As shown on the right side of Figure 5, we utilize AutoDQ [105], an automatic method for evaluating the quality of video description, using two metrics, DQR and DQP3. A preference pair {x,yw = y,yl = y˜} is considered valid if the following conditions are met:

∆DQR ≥ 0 and ∆DQP ≥ 0 and ∆DQR + ∆DQP ≥ δ, (1)

where ∆DQR and ∆DQP denotes the difference of AutoDQ recall and precision scores between the y0 and y1. δ serves as an adjustable threshold to fine-tune the filtering criteria.

During the DPO training phase, we utilize videos from the same training dataset, D, as in the SFT phase, to construct preference data. The policy model is then optimized by minimizing the DPO loss, expressed as:

πθ(yw|x) πref(yw|x) − β log

πθ(yl|x) πref(yl|x)

LDPO = −E(x,yw,yl)∼D log σ β log

, (2)

where πref denotes the model obtained during the SFT phase.

We conducted DPO training on a dataset with 20k preference pairs produced by the above data collection approach, with all parameters set to be trainable. For each video, we sample 16 frames as same as the SFT phase. We trained Tarsier2 for 1,000 steps in total with 64 H100 GPUs and each GPU loaded one pair at each training step, resulting in a global batch size of 64. See Appendix D for more details of DPO training.

### 4 Experiments

In this section, we first evaluate the model’s performance on various video understanding benchmarks, comparing it to several baselines. We highlight Tarsier2’s advantages not only in video description but also across other video understanding tasks. We then present an ablation study to examine key components of our approach.

- 2An obvious counter example is that a low-dynamic video will not be significantly affected by the downsampling perturbation.
- 3Given a reference description (dref) and a description to be assessed (dpred), AutoDQ scorer outputs the recall score (DQR: the ratio of events in dref that are entailed by dpred) and the precision score (DQP: the ratio of events in dref that are entailed by dpred).

#### 4.1 Quantitative Results

##### 4.1.1 Video Captioning

We evaluate Tarsier2 on two video captioning benchmarks: DREAM-1K[105] and E.T. Bench-Captioning[67]. DREAM-1K is a detailed video description benchmark featuring dynamic and diverse videos, assessing the model’s ability to describe fine-grained actions and events. E.T Bench-Captioning is composed of four dense video captioning tasks, requiring key event localization and summary generation for segments in long-form videos.

Video Categories

Model

Overall

Live-action Animation Stock YouTube Shorts Proprietary models

GPT-4V [5] 34.8/39.2/31.3 27.4/31.9/24.0 40.7/46.7/36.1 33.8/40.1/29.2 34.8/46.1/28.0 34.4/40.8/29.7 GPT-4o [41] 39.8/42.1/37.8 35.8/39.1/33.1 44.0/46.6/41.7 35.9/41.5/31.7 39.9/47.9/34.2 39.2/43.4/35.7 Gemini-1.5-Flash [102] 34.8/36.4/33.3 29.2/32.5/26.5 39.4/39.7/39.1 34.3/38.6/30.9 35.6/42.4/30.7 34.8/37.9/32.1 Gemini-1.5-Pro [102] 36.4/36.4/36.4 30.7/31.8/29.7 42.2/40.7/43.8 34.0/36.7/31.6 37.0/42.4/32.7 36.2/37.6/34.8

Open-source models (>10B)

PLLaVA-34B [114] 29.3/34.9/25.2 20.9/32.0/15.6 35.1/42.5/29.9 28.9/40.8/22.3 25.6/41.9/18.4 28.2/38.4/22.3 VideoLLaMA2-72B [23] 27.3/29.3/25.6 19.7/21.7/18.1 33.9/37.0/31.3 27.7/33.0/23.8 26.5/33.1/22.1 27.1/30.8/24.2 LLaVA-OV-72B [53] 31.7/32.8/30.7 27.7/30.6/25.2 38.0/39.6/36.6 34.1/34.7/33.5 33.8/41.8/28.4 33.2/35.9/30.9 LLaVA-Video-72B [127] 33.5/36.3/31.1 28.6/31.7/26.1 39.3/41.1/37.6 32.8/34.7/31.1 35.7/42.8/30.6 34.0/37.3/31.3 Qwen2-VL-72B [106] 32.1/33.7/30.6 27.6/32.6/23.9 41.1/41.2/41.1 32.0/38.1/27.7 32.1/41.0/26.4 33.2/37.3/29.9 InternVL2.5-78B [20] 25.3/31.5/21.1 21.8/28.8/17.6 33.5/38.1/29.9 31.0/38.5/25.9 31.1/41.7/24.8 28.6/35.7/23.9 Tarsier-34B [105] 38.5/39.6/37.5 32.2/35.8/29.2 41.7/46.4/37.8 34.5/41.1/29.7 34.0/44.1/27.7 36.3/41.4/32.4

Open-source models (<10B)

Video-LLaVA-7B [61] 19.4/24.3/16.2 15.3/21.2/11.9 27.0/33.5/22.7 21.2/31.9/15.8 18.5/29.4/13.5 20.4/28.1/16.0 VideoLLaMA2-7B [23] 25.1/28.7/22.2 20.4/25.5/17.0 32.6/35.5/30.2 27.5/33.5/23.4 24.5/34.1/19.2 26.2/31.5/22.4 LLaVA-OV-7B [53] 31.2/33.2/29.3 26.8/29.0/25.0 38.1/39.1/37.1 30.6/32.1/29.2 31.4/38.3/26.6 31.7/34.3/29.4 LLaVA-Video-7B [127] 31.4/35.2/28.4 27.6/32.9/23.8 36.7/39.7/34.1 33.0/39.5/28.3 33.4/42.5/27.5 32.5/37.9/28.4 Qwen2-VL-7B [106] 27.7/32.5/24.2 22.2/28.0/18.4 37.0/36.1/38.0 30.7/35.5/27.0 29.1/37.6/23.8 29.6/33.9/26.3 InternVL2.5-8B [20] 26.6/32.0/22.8 21.3/28.9/16.9 32.7/37.2/29.1 27.9/35.4/23.0 28.9/39.9/22.7 27.6/34.7/22.9 Tarsier-7B [105] 36.6/38.5/34.8 29.3/34.6/25.5 39.6/44.7/35.5 33.0/39.2/28.4 33.6/44.6/26.9 34.6/40.3/30.2

Tarsier2-7B 44.4/41.9/47.3 39.3/39.5/39.1 45.7/45.4/46.0 36.0/38.4/33.9 43.7/48.9/39.4 42.0/42.8/41.1

- Table 1: Evaluation results on DREAM-1K. We report F1/Precision/Recall scores for each category and for the overall dataset. For open-source models, all results are tested with their official checkpoint and inference code under recommended setting. SOTA results of comparable scale (<10B) are bolded and overall best results are underlined.

As shown in Table 1, Tarsier2-7B outperforms all open-source models in both precision and recall across all categories in DREAM-1K, demonstrating its ability to generate more comprehensive and less hallucinatory video descriptions. Notably, Tarsier2-7B achieved an overall F1 score of 42.0%, surpassing the strongest proprietary model, GPT-4o (39.2%). It is also the first model to exceed a 40% overall recall score, highlighting its sensitivity to dynamic actions and events.

Figure 6 further presents the human side-by-side evaluation results of Tarsier2 versus the previous SOTA Tarsier-34B and two strong proprietary models, GPT-4o and Gemini 1.5 Pro. We randomly sampled 250 videos (50 videos for each category) from DREAM-1K, and asked experienced annotators to compare the descriptions generated by two different

[Figure 70]

Figure 6: Human side-by-side evaluation results of Tarsier2 versus other models.

models, collecting their preferences. Each pair of descriptions was randomly shuffled to ensure that the annotators were blind to the description sources. Compared to Tarsier34B, Tarsier2 has a slightly negative advantage rate (15.8%), but wins in a significant percentage of cases (42.8%). Compared to Gemini, Tarsier2 still maintains a significant advantage (45.6% vs 20.7%). Despite being tied with the strongest proprietary model, GPT-4o, on 40% cases, Tarsier2 still gains a slight advantage (8.6%), demonstrating the outstanding performance of Tarsier2 in detailed video description. For a comparison of generated descriptions from different models on DREAM-1K, see Appendix H.

Table 2 shows the evaluation results of dense video captioning on E.T. Bench-Captioning.

Tarsier2-7B outperforms all open-source models with comparable settings (similar model scale, fine-tuned on E.T. Instruct 164K [67]) across all metrics, except for the SLCF1 score, which is slightly lower than Qwen2-VL-7B (24.6% vs 25.7%). These results highlight Tarsier2’s strengths in generating fine-grained descriptions for short videos and providing coarse-grained summaries for long videos.

##### 4.1.2 Short-Video Question Answering

We evaluate Tarsier2-7B on several short-video question answering benchmarks to assess its ability to comprehend and reason about visual content. As shown in Table 3, Tarsier2-7B outperforms both proprietary and open-source models across various benchmarks, achieving state-of-the-art results. Tarsier2-7B exhibits exceptional performance in MVBench [57] and PerceptionTest [86], with scores of 71.5% and 71.6%, respectively.

Furthermore, Tarsier2-7B demonstrates significant performance improvements on bench-

marks featuring temporal reasoning, such as TVBench [25], TOMATO [94], and Vinoground [123]. Tarsier2-7B achieves strong results with 54.7% on TVBench, 42.0% on TOMATO,

E.T. Bench-Captioning [67] DVCF1 DVCSim SLCF1 SLCSim AvgF1 AvgSim Proprietary models

Model

GPT-4V [5] 16.1 19.4 21.9 13.5 19.0 16.4 GPT-4o [41] 46.9 22.3 23.1 14.9 35.0 18.6 Gemini-1.5-Flash [102] 31.6 14.9 16.5 13.3 24.1 14.1 Gemini-1.5-Pro [102] 24.0 17.5 5.8 9.8 14.9 13.7

Open-source models (>10B)

PLLaVA-34B [114] 13.3 10.6 9.7 11.8 11.5 11.2 LLaVA-OV-72B [53] 41.9 16.3 25.6 13.9 33.8 15.1 LLaVA-Video-72B [127] 37.0 15.7 20.4 13.5 28.7 14.6 Qwen2-VL-72B [106] 15.3 13.9 11.0 12.8 13.2 13.4

Open-source models (≤10B)

VideoLLaMA2-7B [23] 0.6 14.5 0.0 15.2 0.3 14.8 Video-LLaVA-7B [61] 28.0 15.0 0.9 8.3 14.4 11.7 LLaVA-OV-7B [53] 22.0 15.1 9.5 10.6 15.8 12.8 LLaVA-Video-7B [127] 20.6 14.7 6.5 13.4 13.6 14.1 E.T. Chat [67] † 38.4 19.7 24.4 14.6 31.4 17.1 Qwen2-VL-7B [106] † 44.3 25.3 25.7 15.6 35.0 20.4 Tarsier-7B [105] † 42.8 19.1 23.7 15.2 33.2 17.1

Tarsier2-7B † 46.5 28.8 24.6 16.4 35.5 22.6

- Table 2: Evaluation results on E.T. Bench-Captioning. Results marked in gray are tested on a subset. † denotes the model is fine-tuned on E.T. Instruct 164K. All results are transcribed from the official benchmark, except for LLaVA-OV, LLaVA-Video and Qwen2VL, which are our evaluation using the official checkpoint and inference code.

Model

MVBench[57] PerceptionTest[86] TVBench[25] TOMATO[94] Vinoground[123] TempCompass[69]

test val test test Text/Video/Group mc/yn/cm/cg Proprietary models

GPT-4o [41] 57.5 - 39.6 37.7 54.0/38.2/24.6 71.0/73.7/80.8/70.8 Gemini-1.5-Pro [102] - - 46.5 36.1 35.8/22.6/10.2 63.9/70.3/77.5/57.9

Open-source models (>10B)

LLaVA-OV-72B [53] 59.4 66.9 45.9 28.6 48.4/35.2/21.8 67.6/72.6/78.2/52.6 LLaVA-Video-72B [127] 64.1 74.3* 50.0 28.2 52.0/35.6/20.8 69.9/73.0/80.9/54.4 Qwen2-VL-72B [106] 73.6 66.5 52.7 37.9 50.4/32.6/17.4 76.0/75.9/84.6/58.6 Tarsier-34B [105] 67.6 60.4 53.8 34.3 37.8/32.0/15.0 69.8/74.0/73.0/60.9

Open-source models (≤10B)

LLaVA-OV-7B [53] 56.7 57.1 45.6 25.5 41.6/29.4/14.6 64.8/69.7/73.8/49.9 LLaVA-Video-7B [127] 58.6 67.9* 45.6 24.9 36.8/29.0/12.8 56.3/68.7/76.8/53.0 Qwen2-VL-7B [106] 67.0 - 43.8 31.5 40.0/23.4/12.4 68.5/72.8/77.3/54.2 Tarsier-7B [105] 62.6 53.9 45.8 28.6 29.8/22.2/8.6 58.7/58.0/54.2/55.3 Previous SOTA 72.0 [20] 70.0* [72] 51.6 [124] 31.5 [106] 41.6/29.4/14.6 [52] 68.5/72.8/77.3/54.2 [106]

Tarsier2-7B 71.5 71.6* 54.7 42.0 65.8/38.0/28.8 75.3/75.1/80.6/66.6

- Table 3: Evaluation results on short video question answering benchmarks. * indicates that the training set has been observed in the training data mixture.

and 65.8%/38.0%/28.8% on Vinoground’s Text/Video/Group tasks, respectively. These results surpass both open-source and proprietary models, including GPT-4o and Gemini1.5-Pro.

At last, Tarsier2-7B also excels on the TempCompass benchmark [69], which evaluates temporal perception in ten aspects and four task formats. Tarsier2-7B achieves impressive scores of 75.3%/75.1%/80.6%/66.6% on TempCompass’ mc/yn/cm/cg tasks, respectively, outperforming both open-source models and larger proprietary models in most cases. This performance further underscores Tarsier2’s advanced ability to process and interpret temporal information in video content.

##### 4.1.3 Long-Video Question Answering

Video-MME[31] LongVideoBench[110] TemporalBench[12] MLVU[128] MMBench-Video[30]

Model

w/o subs val Binary Accuracy M-Avg val Proprietary models

GPT-4o [41] 71.9 66.7 73.2 64.6 1.87 Gemini-1.5-Pro [102] 75.0 64.0 66.4 - 1.30

Open-source models (>10B)

VILA-1.5-40B [62] 60.1 - - 56.7 1.61 LLaVA-Video-72B [127] 70.5 61.9 72.4 74.4 1.71 Qwen2-VL-72B [106] 71.2 - 70.2 - 1.70 InternVL2.5-78B [20] 72.1 63.6 - 75.7 1.97 Tarsier-34B [105] 52.3 54.2 66.7 58.2 1.46

Open-source models (≤10B) LLaVA-Video-7B [127] 63.3 58.2 63.6 70.8 1.60 Qwen2-VL-7B [106] 63.3 55.6 62.0 - 1.44 InternVL2.5-8B [20] 64.2 60.0 - 68.9 1.68 Tarsier-7B [105] 42.2 39.8 56.9 49.3 Previous SOTA 64.2 [70] 60.0 [20] 63.6 [127] 70.9 [130] 1.70 [119]

Tarsier2-7B 64.5 (128f) 58.6 (128f) 65.3 (128f) 67.9 (256f) 1.82 (128f)

- Table 4: Evaluation results on long-video question answering benchmarks. We list the number of frames used for each benchmark during evaluating Tarsier2.

We evaluate Tarsier2 on long-video question answering benchmarks by uniformly sampling 128 or 256 frames, depending on the video length. Comparison results with other proprietary and open-source models are presented in Table 4. Despite our training set not including many long video data, Tarsier2, compared with others under 10 billion parameters, still achieves SOTA on three benchmarks and competitive performance on several other benchmarks.

##### 4.1.4 Hallucination

We evaluate Tarsier2 on two video hallucination benchmarks: VideoHallucer [109] and EventHallusion [122]. The results are summarized in Table 5. For VideoHallucer, Tarsier27B achieves an overall score of 67.0%, outperforming all comparable baselines of similar

|VideoHallucer [109]<br><br>|EventHallusion [122]|
|---|---|
|Yes/No QA Basic/Hallucinated/Overall<br><br>|Yes/No QA Desc GPT Entire/Interleave/Misleading/Overall Entire/Interleave/Misleading/Overall|

###### Model

Proprietary models

GPT-4o [41] 75.1/74.2/53.3 65.8/90.7/92.2/84.1 34.9/54.9/83.2/56.2 Gemini-1.5-Pro [102] 83.6/42.3/37.8 70.2/77.7/96.1/80.2 38.5/40.9/80.0/49.6

Open-Source models (>10B)

Qwen2-VL-72B [106] 87.1/79.4/70.2 33.3/77.7/56.4/60.0 16.5/25.4/70.2/33.6 LLaVA-OV-72B [53] 88.3/62.6/55.2 47.4/26.9/90.1/48.3 24.8/34.7/71.3/40.7 LLaVA-Video-72B [127] 88.2/73.5/64.6 57.9/11.9/96.0/45.6 32.1/35.8/75.5/44.2 InternVL2.5-78B [20] 82.5/82.5/67.8 57.9/67.9/88.2/70.2 45.0/43.0/76.8/51.6 Tarsier-34B [105] 84.8/80.0/67.7 49.1/92.7/69.6/74.8 38.5/40.4/83.2/50.1

Open-Source models (≤10B)

LLaVA-OV-7B [53] 81.1/69.6/53.8 46.5/67.4/86.1/66.2 22.0/26.4/73.4/36.4 LLaVA-Video-7B [127] 82.4/70.6/56.0 61.4/48.7/96.0/64.0 27.5/32.6/75.5/41.4 Qwen2-VL-7B [106] 85.0/70.8/59.3 35.1/94.3/57.4/68.6 14.7/16.1/67.0/27.8 InternVL2.5-8B [20] 72.7/78.3/53.6 46.5/69.2/90.2/68.2 23.9/20.7/60.0/31.0 Tarsier-7B [105] 76.4/60.8/41.4 43.9/82.4/79.4/70.9 35.8/29.5/72.6/41.6

Tarsier2-7B 86.5/78.3/67.0 60.5/93.3/95.1/84.6 54.6/53.1/93.7/63.3

Table 5: Evaluation results on hallucination benchmarks.

model scale and even proprietary models like GPT-4o and Gemini-1.5-pro. In EventHallusion, for video question-answering task, Tarsier2-7B achieves 84.6%, surpassing GPT-4o’s score of 84.1%, while outperforming all other baselines. For the detailed description matching task, which directly assesses video description hallucinations by prompting GPT-4 to answer questions based on each model’s generated video description, Tarsier2-7B demonstrates superior performance, even surpassing GPT-4o by 7.1% in terms of Overall score.

##### 4.1.5 Video Grounding

We evaluate the video grounding capability of models on E.T. Bench-Grounding, which combines various grounding tasks from multiple datasets, including QVHighlights [51], Charades-STA [32], THUMOS’14 [42], and Ego4D-NLQ [35], among others. The results, shown in Table 6, indicate that Tarsier2-7B achieves the highest mean F1 score of 35.5%, outperforming all baselines and highlighting its superior temporal perception capabilities.

##### 4.1.6 Embodied Question Answering

We evaluate Tarsier2 on embodied question answering to assess its performance in realworld robotic scenarios, using three benchmarks: EgoTaskQA [44], RoboVQA [93], and OpenEQA [77]. To align with the baselines, Tarsier2 is fine-tuned on the training sets for EgoTaskQA and RoboVQA, while for OpenEQA, it is evaluated in a zero-shot setting. The results, presented in Table 7, include exact match accuracy for EgoTaskQA, BLEU score for RoboVQA, and the correctness score evaluated by GPT-4-1106-preview [1] for OpenEQA. Tarsier2 achieves top-tier performance across all three benchmarks, outperforming both generalist and specialist models. Notably, on EgoTaskQA, its performance

E.T. Bench-Grounding [67] TVGF1 EPMF1 TALF1 EVSF1 VHDF1 MeanF1 Proprietary models

Model

GPT-4V [5] 27.0 1.8 18.0 28.6 55.1 26.1 GPT-4o [41] 40.4 4.5 20.0 17.6 56.9 27.9 Gemini-1.5-Flash [102] 43.9 5.4 27.0 5.4 60.8 28.5 Gemini-1.5-Pro [102] 43.1 6.2 33.8 7.9 47.0 27.6

Open-source models (<10B)

LITA [39] 22.2 4.6 18.0 29.7 23.9 19.7 VTG-LLM [37] 15.9 3.7 14.4 26.8 48.2 21.8 TimeChat [91] † - - - - - 24.3 E.T. Chat [67] † 38.6 10.2 30.8 25.4 62.5 33.5 Tarsier-7B [105] † 39.6 9.0 25.0 25.4 47.6 30.9 Qwen2-VL-7B [106] † 39.7 7.0 26.9 17.1 66.9 33.5

Tarsier2-7B † 38.4 11.0 31.8 19.4 66.8 35.5

- Table 6: Evaluation results on E.T. Bench-Grounding. Results marked in gray are tested on a subset. † denotes the model is fine-tuned on E.T. Instruct 164K.

Model

EgoTaskQA Exact Match

Human 80.0 HCRN [50] 42.2 GF [9] 44.3 EgoVLPv2 [88] 46.3

Tarsier2 77.5

Model

RoboVQA BLEU-1/2/3/4

LLaMA-AdapterV2 [33] 27.8/16.0/10.9/8.1 LLaVA-OV-7B [53] 38.1/33.6/31.8/31.0 RoboMamba [66] 54.9/44.2/39.5/36.3 MLCD [3] 73.2/66.4/60.6/56.6

Tarsier2 77.1/67.4/61.5/56.8

Model

OpenEQA GPT-4

Human 86.8 GPT-4V [5] 55.3 Gemini-1.5-Pro [102] 44.9 MLCD [3] 48.8

Tarsier2 58.7

- Table 7: Evaluation results on embodied question-answering tasks, including EgoTaskQA, RoboVQA and OpenEQA.

approaches human-level accuracy, highlighting the model’s significant potential in embodied intelligence.

#### 4.2 Ablation Study

We conduct a comprehensive ablation study to evaluate key components at different stages of the training process. The study is based on three tasks: 1) Caption: This includes the DREAM-1K dataset, the caption generation task from TempCompass (TempCompass-cg), and the caption matching task from Vinoground (Vinoground-Text) to assess captioning performance. 2) Video QA: This encompasses short-video QA, measured by the average accuracy on MVBench, TVBench, and TOMATO, and long-video QA, measured by the average accuracy on Video-MME, LongVideoBench, and TemporalBench. It evaluates the model’s video understanding capabilities. 3) Hallucination: We use the average score

of two sub-tasks from EventHallusion to assess hallucination in the model. The following subsections present the results for each task, with detailed results for individual datasets provided in the Appendix E.

##### 4.2.1 Pre-training

Caption Video QA

Model

Hallucination

DREAM-1K TempCompass-cg Vinoground-Text Short Long

- Tarsier1-7B 34.6 55.3 29.8 45.6 46.3 56.3

- Tarsier1-7B-Qwen upgrading model

38.4 (↑3.8) 59.3 (↑4.0) 48.6 (↑18.8) 52.4 (↑6.8) 57.6 (↑11.3) 62.1 (↑5.8)

- Tarsier2-7B

40.8 (↑6.2) 60.1 (↑4.8) 60.2 (↑30.4) 55.3 (↑9.7) 64.1 (↑17.8) 63.5 (↑7.2)

upgrading model+data

- Table 8: Results of the ablation study for pre-training. Tarsier1-7b-Qwen stands for the model where the base model is upgraded to Qwen2-VL, while the pre-training dataset remains the same as Tarsier1. Tarsier2 is trained from Qwen2-VL with an expanded pretraining dataset, growing from 13 million in Tarsier1 to 40 million samples.

In this section, we evaluate the impact of several factors during pre-training, including the base model, pre-training data and training steps. For the caption task, we report results after the SFT stage, which aligns the model’s responses with the desired style. For other tasks, we report results after pre-training stage.

Compared to Tarsier1, two key improvements are made in the pre-training phase: upgrading the base model to Qwen2-VL and expanding the training dataset from 13 million to 40 million samples. Table 8 illustrates the additive contributions for each improvement, showing that both enhancements consistently and significantly boost the model’s performance in caption generation, video QA, and hallucination reduction. Specifically, these enhancements lead to accuracy improvements of 9.7%, 17.8%, and 7.2% for short-video QA, long-video QA, and hallucination tests, respectively. For video description, the F1 score on the DREAM-1K dataset improves by 6.2%.

To better understand the effect of the number of training tokens on pre-training performance, we plot the model’s performance as a function of token count during the pre-training stage, as shown in Figure 7. The results show that model performance improves with an increase in the number of training tokens, reaching convergence after 160 billion tokens. This suggests that a large volume of data is essential for optimal video understanding performance.

4For consistency across all checkpoints, we evaluate the Qwen2-VL-7B model using the same frame sampling strategy applied to other checkpoints. This may differ from the official sampling strategy in some benchmarks. For instance, the official setting of Video-MME uses 768 frames, while we sample 128 frames.

65

60

55

Performance

DREAM-1K

50

Short Video QA Long Video QA Hallucination

45

40

0 20 40 60 80 100 120 140 160 180 200

#Training Tokens (B)

Figure 7: Model performance against training tokens. The results at the initial step reflect the performance of Qwen2-VL-7B.4

Caption Video QA

Model

Hallucination

DREAM-1K TempCompass-cg Vinoground-Text Short Long

- Tarsier2-7B-SFT 40.8 60.1 60.2 56.2 63.2 71.9

w/o SFT 35.2 (↓5.6) 50.5 (↓9.6) 57.2 (↓3.0) 55.3 (↓0.9) 64.1 (↑0.9) 63.5 (↓8.4) w/o grounding 37.4 (↓3.4) 50.2 (↓9.9) 60.6 (↑0.4) 55.9 (↓0.3) 61.9 (↓1.3) 68.6 (↓3.3)

- Table 9: Ablation study of temporal grounding dataset during the SFT phase. Tarsier27B-SFT refers to the model after the SFT phase. w/o SFT refers to the model after pretraining; w/o grounding refers to the model fine-tinued without grounding information.

##### 4.2.2 SFT

The key factor in the SFT phase is fine-grained alignment. To investigate its impact, we conduct an ablation study, with the results presented in Table 9. When the video description data, which includes fine-grained temporal grounding information, is excluded (i.e., without grounding), model performance significantly deteriorates. Specifically, the F1 score on DREAM-1K decreases by 3.4%, accuracy on TempCompass-cg drops by 9.9%, accuracy on long-video QA falls by 1.3%, and accuracy on the hallucination test declines by 3.3%.

Furthermore, the SFT phase leads to substantial improvements, highlighting the importance of high-quality manually labeled data. It boosts the F1 score on DREAM-1K by 5.6%, accuracy on TempCompass-cg by 9.6%, accuracy on Vinoground-Text by 3.0%,

Caption Video QA

Model

Hallucination

DREAM-1K TempCompass-cg Vinoground-Text Short Long Tarsier2-7B 42.0 66.6 65.8 56.1 62.8 74.0

w/o DPO 40.8 (↓1.2) 62.1 (↓6.5) 60.6 (↓5.6) 56.2 (↑0.1) 63.2 (↑0.4) 71.9 (↓2.1) w/o NS 41.5 (↓0.5) 61.1 (↓5.5) 59.8 (↓6.0) 56.1 (↓0.0) 62.8 (↓0.0) 72.9 (↓1.1) w/o PF 40.5 (↓1.5) 65.1 (↓1.5) 67.6 (↑1.8) 56.0 (↓0.1) 62.3 (↓0.5) 74.2 (↑0.2)

- Table 10: Ablation study for DPO training phase, negative sampling (NS) and preference data filtering (PF) strategies.

and accuracy on the hallucination test by 8.4%, demonstrating the SFT phase’s role in enhancing the model’s fine-grained video understanding and mitigating hallucinations.

##### 4.2.3 DPO

We conduct ablation experiments to evaluate the DPO phase, negative sampling (NS) and preference data filtering (PF) strategies. Specifically, we test the following settings: 1) w/o DPO: SFT model without DPO training. 2) w/o NS: Preference pairs generated by sampling the same video twice, without negative sampling. 3) w/o PF: Responses from negative sampling are treated as rejections, without utilizing AutoDQ Scorer to perform preference data filtering. For a fair comparison, the training data size and hyper-parameters for the latter two settings are kept consistent with the default setting, as detailed in Appendix D.

As shown in Table 10, Tarsier2 benefits a lot from the DPO training phase with significant improvement on caption tasks, especially TempCompass-cg (6.5%) and VinogroundText (5.6%). The hallucination capability also drops by 2.1% without DPO, while the performance on video QA is not obviously affected. When further ablating dataset construction strategy of DPO, negative sampling plays an important role, without which the model results on most of the tasks are degraded to be almost the same as the SFT model (“w/o DPO”), and the hallucination capability drops by 1.1%. Additionally, preference data filtering with AutoDQ scorer has a significant impact on maintaining the quality of DPO datasets. As shown in Table 10, “w/o PF” leads to degradation on more than a half of the tasks, and especially the DREAM-1K F1 score is even worse than the SFT model.

#### 4.3 Video Recaptioning using Tarsier2

In this section, we utilize Tarsier2 as a captioner to generate detailed descriptions for a diverse set of 1M videos sourced from public datasets, resulting in the recaptioning dataset Tarsier2-Recap-585K5. Details of the dataset composition are provided in Appendix F.

5Tarsier2-Recap-585K is available on HuggingFace.

Caption Video QA

Model

Hallucination

DREAM-1K TempCompass-cg Vinoground-Text Short Long Qwen2-VL-7B [106] 31.2 54.2 40.0 49.4 60.3 51.9

+ Original FT 35.2 (↑4.0) 49.9 (↓4.3) 39.0 (↓1.0) 46.9 (↓2.5) 55.4 (↓4.9) 43.0 (↓8.9)

+ Recaption FT 39.5 (↑8.3) 67.7 (↑13.5) 55.0 (↑15.0) 52.5 (↑3.1) 56.8 (↓3.5) 68.5 (↑16.6)

- Table 11: The experimental results of recaptioning. “Recaption FT” represents fine-tune the model on the Tarsier2-Recap-585K dataset. “Original FT” represents fine-tune the model with the same videos as Tarsier2-Recap-585K but taking their original labels as target output.

We fine-tune Qwen2-VL-7B [106] on Tarsier2-Recap-585K and present the evaluation results in Table 11. Fine-tuning on Tarsier2-Recap-585K significantly enhances the model’s performance on detailed video description, achieving improvements in DREAM1K (+8.3%), TempCompass-cg (+13.4%), and Vinoground-Text (+15.0%). Moreover, it achieves an improvement of 16.6% in hallucination test and an improvement of 3.1% in short video-QA.

In comparison, fine-tuning on the same 585K videos with original captions improves only the DREAM-1K F1 score (+4.0%), while other metrics show significant declines. It indicates that the performance gains from Tarsier2-Recap-585K are primarily due to its high-quality and detailed captions rather than the additional training data volume.

Table 17 in Appendix E provides detailed benchmark results corresponding to Table 11. These findings demonstrate that Tarsier2 can generate high-quality, detailed descriptions that offer fine-grained alignment information to help LVLMs to achieve significant improvements across various tasks.

### 5 Conclusion

In this paper, we introduce Tarsier2, a state-of-the-art large vision-language model that outperforms existing proprietary and open-source models in generating detailed and accurate video descriptions. Furthermore, Tarsier2 sets new benchmarks across a wide range of video understanding tasks. Our ablation studies demonstrate that Tarsier2 ’s advancements are driven by scaling the volume and diversity of the training dataset, fine-grained temporal alignment, and DPO training.

Looking ahead, we outline several promising directions for future research. First, extending Tarsier2 to handle longer video durations by developing more efficient model architectures and expanding the training dataset. Second, enhancing real-time video processing to improve the model’s ability to analyze and describe videos as they stream. Third, exploring richer interactions between video, audio, and text to create more comprehensive and context-aware video understanding systems.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 6, 16
- [2] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, et al. Pixtral 12b. arXiv preprint arXiv:2410.07073, 2024. 5
- [3] Xiang An, Kaicheng Yang, Xiangzi Dai, Ziyong Feng, and Jiankang Deng. Multi-label cluster discrimination for visual representation learning. In European Conference on Computer Vision, pages 428–444. Springer, 2025. 17
- [4] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, pages 5803–5812, 2017. 37
- [5] Sally Applin, Gerardo Adesso, Rubaid Ashfaq, Max Bai, Matthew Brammer, Ethan Fecht, Andrew Goodman, Shelby Grossman, Matthew Groh, Hannah Rose Kirk, et al. Gpt-4v (ision) system card. 2023. 6, 12, 14, 17
- [6] Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Essam Sleiman, Deyao Zhu, Jian Ding, and Mohamed Elhoseiny. Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens. arXiv preprint arXiv:2404.03413, 2024. 5
- [7] George Awad, Keith Curtis, Asad Butt, Jonathan Fiscus, Afzal Godil, Yooyoung Lee, Andrew Delgado, Eliot Godard, Lukas Diduch, Jeffrey Liu, et al. An overview on the evaluated video retrieval tasks at trecvid 2022. arXiv preprint arXiv:2306.13118,

2023. 37, 41

- [8] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 1(2):3, 2023. 5
- [9] Ziyi Bai, Ruiping Wang, and Xilin Chen. Glance and focus: Memory prompting for multi-event video question answering. Advances in Neural Information Processing Systems, 36, 2024. 17
- [10] Max Bain, Arsha Nagrani, Gu¨l Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738, 2021. 37, 41

- [11] Chandra Bhagavatula, Ronan Le Bras, Chaitanya Malaviya, Keisuke Sakaguchi, Ari Holtzman, Hannah Rashkin, Doug Downey, Scott Wen-tau Yih, and Yejin Choi. Abductive commonsense reasoning. arXiv preprint arXiv:1908.05739, 2019. 37
- [12] Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, et al. Temporalbench: Benchmarking fine-grained temporal understanding for multimodal video models. arXiv preprint arXiv:2410.10818, 2024. 1, 15
- [13] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017. 37, 39, 41
- [14] David Chen and William Dolan. Collecting highly parallel data for paraphrase evaluation. In Dekang Lin, Yuji Matsumoto, and Rada Mihalcea, editors, Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 190–200, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. 6
- [15] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 37
- [16] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, and Jiaqi Wang. Sharegpt4video: Improving video understanding and generation with better captions, 2024. 6
- [17] Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing Liu. Vast: A vision-audio-subtitle-text omni-modality foundation model and dataset. Advances in Neural Information Processing Systems, 36:72842–72866, 2023. 6
- [18] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiangwei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024. 5
- [19] Xiuyuan Chen, Yuan Lin, Yuchen Zhang, and Weiran Huang. Autoeval-video: An automatic benchmark for assessing large vision language models in open-ended video question answering. arXiv preprint arXiv:2311.14906, 2023. 3

- [20] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 1, 3, 5, 12, 14, 15, 16
- [21] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 3
- [22] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 36
- [23] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, and Lidong Bing. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms, 2024. 3, 12, 14
- [24] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. 37
- [25] Daniel Cores, Michael Dorkenwald, Manuel Mucientes, Cees GM Snoek, and Yuki M Asano. Tvbench: Redesigning video-language evaluation. arXiv preprint arXiv:2410.07752, 2024. 1, 3, 5, 13, 14
- [26] Erfei Cui, Yinan He, Zheng Ma, Zhe Chen, Hao Tian, Weiyun Wang, Kunchang Li, Yi Wang, Wenhai Wang, Xizhou Zhu, Lewei Lu, Tong Lu, Yali Wang, Limin Wang, Yu Qiao, and Jifeng Dai. Sharegpt-4o: Comprehensive multimodal annotations with gpt-4o. https://sharegpt4o.github.io/, 2024. 37
- [27] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 5
- [28] Dave Epstein, Boyuan Chen, and Carl Vondrick. Oops! predicting unintentional action in video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 919–929, 2020. 37, 39, 41

- [29] Chenyou Fan. Egovqa-an egocentric video question answering benchmark dataset. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pages 0–0, 2019. 37
- [30] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. arXiv preprint arXiv:2406.14515, 2024. 1, 15
- [31] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 1, 15
- [32] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, pages 5267–5275, 2017. 16
- [33] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameterefficient visual instruction model. arXiv preprint arXiv:2304.15010, 2023. 17
- [34] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The” something something” video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pages 5842–5850, 2017. 37, 39, 41
- [35] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995– 19012, 2022. 16, 37, 41
- [36] Chunhui Gu, Chen Sun, David A Ross, Carl Vondrick, Caroline Pantofaru, Yeqing Li, Sudheendra Vijayanarasimhan, George Toderici, Susanna Ricco, Rahul Sukthankar, et al. Ava: A video dataset of spatio-temporally localized atomic visual actions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6047–6056, 2018. 37
- [37] Yongxin Guo, Jingyu Liu, Mingda Li, Xiaoying Tang, Xi Chen, and Bo Zhao. Vtgllm: Integrating timestamp knowledge into video llms for enhanced video temporal grounding. arXiv preprint arXiv:2405.13382, 2024. 17

- [38] Yichen He, Yuan Lin, Jianchao Wu, Hanchong Zhang, Yuchen Zhang, and Ruicheng Le. Storyteller: Improving long video description through global audio-visual character identification. arXiv preprint arXiv:2411.07076, 2024. 36, 37
- [39] De-An Huang, Shijia Liao, Subhashree Radhakrishnan, Hongxu Yin, Pavlo Molchanov, Zhiding Yu, and Jan Kautz. Lita: Language instructed temporallocalization assistant. In ECCV, 2024. 17
- [40] Ting-Hao Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Aishwarya Agrawal, Jacob Devlin, Ross Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, et al. Visual storytelling. In Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: Human language technologies, pages 1233–1239, 2016. 37
- [41] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 3, 6, 12, 14, 15, 16, 17
- [42] Haroon Idrees, Amir R Zamir, Yu-Gang Jiang, Alex Gorban, Ivan Laptev, Rahul Sukthankar, and Mubarak Shah. The thumos challenge on action recognition for videos “in the wild”. Computer Vision and Image Understanding, 155:1–23, 2017. 16
- [43] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward spatio-temporal reasoning in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2758–2766,

2017. 37

- [44] Baoxiong Jia, Ting Lei, Song-Chun Zhu, and Siyuan Huang. Egotaskqa: Understanding human tasks in egocentric videos. In The 36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks, 2022. 16
- [45] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pretraining of deep bidirectional transformers for language understanding. In Proceedings of naacL-HLT, volume 1, page 2. Minneapolis, Minnesota, 2019. 8
- [46] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In European Conference on Computer Vision (ECCV), 2022. 37
- [47] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In International Conference on Computer Vision (ICCV), 2017. 37, 41

- [48] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017. 37
- [49] Hildegard Kuehne, Hueihan Jhuang, Estı´baliz Garrote, Tomaso Poggio, and Thomas Serre. Hmdb: a large video database for human motion recognition. In 2011 International conference on computer vision, pages 2556–2563. IEEE, 2011. 37
- [50] Thao Minh Le, Vuong Le, Svetha Venkatesh, and Truyen Tran. Hierarchical conditional relation networks for video question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9972–9981,

2020. 17

- [51] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. Advances in Neural Information Processing Systems, 34:11846–11858, 2021. 16
- [52] Bo Li, Hao Zhang, Kaichen Zhang, Dong Guo, Yuanhan Zhang, Renrui Zhang, Feng Li, Ziwei Liu, and Chunyuan Li. Llava-next: What else influences visual instruction tuning beyond data, 2024. 3, 5, 14, 37
- [53] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 1, 3, 12, 14, 16, 17
- [54] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, and Junnan Li. Aria: An open multimodal native mixture-of-experts model. arXiv preprint arXiv:2410.05993, 2024. 5
- [55] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR,

2023. 5

- [56] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 3, 5
- [57] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024. 1, 3, 13, 14

- [58] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pages 323–340. Springer, 2025. 5
- [59] Yuncheng Li, Yale Song, Liangliang Cao, Joel Tetreault, Larry Goldberg, Alejandro Jaimes, and Jiebo Luo. Tgif: A new dataset and benchmark on animated gif description. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4641–4650, 2016. 37, 41
- [60] W Lian, B Goodson, E Pentland, et al. Openorca: An open dataset of gpt augmented flan reasoning traces, 2023. 37
- [61] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 3, 5, 12, 14
- [62] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26689–26699, 2024. 3, 5, 15
- [63] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´r, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer,

2014. 37

- [64] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 37
- [65] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. arXiv preprint arXiv:2408.15542, 2024. 5
- [66] Jiaming Liu, Mengzhen Liu, Zhenyu Wang, Lily Lee, Kaichen Zhou, Pengju An, Senqiao Yang, Renrui Zhang, Yandong Guo, and Shanghang Zhang. Robomamba: Multimodal state space model for efficient robot reasoning and manipulation. arXiv preprint arXiv:2406.04339, 2024. 17
- [67] Ye Liu, Zongyang Ma, Zhongang Qi, Yang Wu, Chang Wen Chen, and Ying Shan. E.t. bench: Towards open-ended event-level video-language understanding. In Neural Information Processing Systems (NeurIPS), 2024. 1, 3, 5, 12, 13, 14, 17, 37

- [68] Yi Liu, Limin Wang, Yali Wang, Xiao Ma, and Yu Qiao. Fineaction: A fine-grained video dataset for temporal action localization. IEEE transactions on image processing, 31:6937–6950, 2022. 37
- [69] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024. 1, 14, 15
- [70] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, Xiuyu Li, Yunhao Fang, Yukang Chen, Cheng-Yu Hsieh, De-An Huang, An-Chieh Cheng, Vishwesh Nath, Jinyi Hu, Sifei Liu, Ranjay Krishna, Daguang Xu, Xiaolong Wang, Pavlo Molchanov, Jan Kautz, Hongxu Yin, Song Han, and Yao Lu. Nvila: Efficient frontier visual language models, 2024. 1, 15
- [71] Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, et al. Mmdu: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for lvlms. arXiv preprint arXiv:2406.11833, 2024. 37
- [72] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024. 5, 14
- [73] Shangbang Long, Siyang Qin, Dmitry Panteleev, Alessandro Bissacco, Yasuhisa Fujii, and Michalis Raptis. Towards end-to-end unified scene text detection and layout analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 37
- [74] Simon M Lucas, Alex Panaretos, Luis Sosa, Anthony Tang, Shirley Wong, Robert Young, Kazuki Ashida, Hiroki Nagai, Masayuki Okamoto, Hiroaki Yamamoto, et al. Icdar 2003 robust reading competitions: entries, results, and future directions. International Journal of Document Analysis and Recognition (IJDAR), 7:105–122, 2005. 37
- [75] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. arXiv preprint arXiv:2306.07207, 2023. 5
- [76] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 5, 37

- [77] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. Openeqa: Embodied question answering in the era of foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16488–16498, 2024. 16
- [78] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023. 3
- [79] Joanna Materzynska, Guillaume Berger, Ingo Bax, and Roland Memisevic. The jester dataset: A large-scale video dataset of human gestures. In Proceedings of the IEEE/CVF international conference on computer vision workshops, pages 0–0, 2019. 37
- [80] Antoine Miech, Jean-Baptiste Alayrac, Ivan Laptev, Josef Sivic, and Andrew Zisserman. Rareact: A video dataset of unusual interactions. arXiv preprint arXiv:2008.01018, 2020. 37
- [81] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2630–2640, 2019. 5, 6, 8
- [82] Mathew Monfort, SouYoung Jin, Alexander Liu, David Harwath, Rogerio Feris, James Glass, and Aude Oliva. Spoken moments: Learning joint audio-visual representations from video descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14871–14881, 2021. 37
- [83] Mathew Monfort, Bowen Pan, Kandan Ramakrishnan, Alex Andonian, Barry A McNamara, Alex Lascelles, Quanfu Fan, Dan Gutfreund, Roge´rio Schmidt Feris, and Aude Oliva. Multi-moments in time: Learning and interpreting models for multiaction video understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(12):9434–9445, 2021. 37
- [84] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. Advances in neural information processing systems, 24, 2011. 37
- [85] Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. Cord: a consolidated receipt dataset for post-ocr parsing. In Workshop on Document Intelligence at NeurIPS 2019, 2019. 37

- [86] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36, 2024. 3, 13, 14
- [87] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pages 2641–2649, 2015. 37
- [88] Shraman Pramanick, Yale Song, Sayan Nag, Kevin Qinghong Lin, Hardik Shah, Mike Zheng Shou, Rama Chellappa, and Pengchuan Zhang. Egovlpv2: Egocentric video-language pre-training with fusion in the backbone. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5285–5297, 2023. 17
- [89] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36,

2024. 5, 6, 9

- [90] Michaela Regneri, Marcus Rohrbach, Dominikus Wetzel, Stefan Thater, Bernt Schiele, and Manfred Pinkal. Grounding action descriptions in videos. Transactions of the Association for Computational Linguistics, 1:25–36, 2013. 37
- [91] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A timesensitive multimodal large language model for long video understanding. ArXiv, abs/2312.02051, 2023. 17
- [92] Anna Rohrbach, Atousa Torabi, Marcus Rohrbach, Niket Tandon, Christopher Pal, Hugo Larochelle, Aaron Courville, and Bernt Schiele. Movie description. International Journal of Computer Vision, 123:94–120, 2017. 37, 41
- [93] Pierre Sermanet, Tianli Ding, Jeffrey Zhao, Fei Xia, Debidatta Dwibedi, Keerthana Gopalakrishnan, Christine Chan, Gabriel Dulac-Arnold, Sharath Maddineni, Nikhil J Joshi, Pete Florence, Wei Han, Robert Baruch, Yao Lu, Suvir Mirchandani, Peng Xu, Pannag Sanketi, Karol Hausman, Izhak Shafran, Brian Ichter, and Yuan Cao. Robovqa: Multimodal long-horizon reasoning for robotics. In arXiv preprint arXiv:2311.00899, 2023. 5, 16
- [94] Ziyao Shangguan, Chuhan Li, Yuxuan Ding, Yanan Zheng, Yilun Zhao, Tesca Fitzgerald, and Arman Cohan. Tomato: Assessing visual temporal reasoning capabilities in multimodal foundation models. arXiv preprint arXiv:2410.23266, 2024. 1, 5, 13, 14

- [95] Baoguang Shi, Cong Yao, Minghui Liao, Mingkun Yang, Pei Xu, Linyan Cui, Serge Belongie, Shijian Lu, and Xiang Bai. Icdar2017 competition on reading chinese text in the wild (rctw-17). In 2017 14th iapr international conference on document analysis and recognition (ICDAR), volume 1, pages 1429–1434. IEEE, 2017. 37
- [96] Gunnar A Sigurdsson, Abhinav Gupta, Cordelia Schmid, Ali Farhadi, and Karteek Alahari. Charades-ego: A large-scale dataset of paired third and first person videos. arXiv preprint arXiv:1804.09626, 2018. 37, 41
- [97] Gunnar A Sigurdsson, G¨ul Varol, Xiaolong Wang, Ali Farhadi, Ivan Laptev, and Abhinav Gupta. Hollywood in homes: Crowdsourcing data collection for activity understanding. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14, pages 510–526. Springer, 2016. 37, 41
- [98] Yipeng Sun, Zihan Ni, Chee-Kheng Chng, Yuliang Liu, Canjie Luo, Chun Chet Ng, Junyu Han, Errui Ding, Jingtuo Liu, Dimosthenis Karatzas, et al. Icdar 2019 competition on large-scale street view text with partial labeling-rrc-lsvt. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1557–1562. IEEE, 2019. 37
- [99] Changli Tang, Yixuan Li, Yudong Yang, Jimin Zhuang, Guangzhi Sun, Wei Li, Zujun Ma, and Chao Zhang. Enhancing multimodal llm for detailed and accurate video captioning using multi-round preference optimization, 2024. 6
- [100] Changli Tang, Yixuan Li, Yudong Yang, Jimin Zhuang, Guangzhi Sun, Wei Li, Zujun Ma, and Chao Zhang. Enhancing multimodal llm for detailed and accurate video captioning using multi-round preference optimization, 2024. 9
- [101] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1207–1216, 2019. 37
- [102] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, Soroosh Mariooryad, Yifan Ding, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. 3, 6, 12, 14, 15, 16, 17
- [103] Andreas Veit, Tomas Matera, Lukas Neumann, Jiri Matas, and Serge Belongie. Cocotext: Dataset and benchmark for text detection and recognition in natural images. arXiv preprint arXiv:1601.07140, 2016. 37

- [104] Han Wang, Yanjie Wang, Yongjie Ye, Yuxiang Nie, and Can Huang. Elysium: Exploring object-level perception in videos via mllm. arXiv preprint arXiv:2403.16558,

2024. 5

- [105] Jiawei Wang, Liping Yuan, Yuchen Zhang, and Haomiao Sun. Tarsier: Recipes for training and evaluating large video description models, 2024. 1, 3, 5, 11, 12, 14, 15, 16, 17
- [106] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 3, 5, 6, 12, 14, 15, 16, 17, 21, 40
- [107] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4581–4591, 2019. 6, 41
- [108] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinyuan Chen, Yaohui Wang, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Yu Qiao. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023. 5
- [109] Yuxuan Wang, Yueqian Wang, Dongyan Zhao, Cihang Xie, and Zilong Zheng. Videohallucer: Evaluating intrinsic and extrinsic hallucinations in large video-language models. arXiv preprint arXiv:2406.16338, 2024. 1, 3, 15, 16
- [110] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. arXiv preprint arXiv:2407.15754, 2024. 1, 15
- [111] Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. Llava-critic: Learning to evaluate multimodal models,

2024. 9

- [112] Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang, et al. mplug-2: A modularized multi-modal foundation model across text, image and video. In International Conference on Machine Learning, pages 38728–38748. PMLR, 2023. 6
- [113] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016. 6

- [114] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024. 3, 5, 12, 14
- [115] Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. Slowfast-llava: A strong training-free baseline for video large language models. arXiv preprint arXiv:2407.15841, 2024. 5
- [116] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5036–5045, 2022. 5, 6
- [117] Shen Yan, Tao Zhu, Zirui Wang, Yuan Cao, Mi Zhang, Soham Ghosh, Yonghui Wu, and Jiahui Yu. Videococa: Video-text modeling with zero-shot transfer from contrastive captioners. arXiv preprint arXiv:2212.04979, 2022. 6
- [118] Cong Yao, Xiang Bai, Wenyu Liu, Yi Ma, and Zhuowen Tu. Detecting texts of arbitrary orientations in natural images. In 2012 IEEE conference on computer vision and pattern recognition, pages 1083–1090. IEEE, 2012. 37
- [119] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 1, 15
- [120] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019. 37
- [121] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audiovisual language model for video understanding. arXiv preprint arXiv:2306.02858,

2023. 5

- [122] Jiacheng Zhang, Yang Jiao, Shaoxiang Chen, Jingjing Chen, and Yu-Gang Jiang. Eventhallusion: Diagnosing event hallucinations in video llms. arXiv preprint arXiv:2409.16597, 2024. 1, 5, 15, 16
- [123] Jianrui Zhang, Mu Cai, and Yong Jae Lee. Vinoground: Scrutinizing lmms over dense temporal reasoning with short videos. arXiv preprint arXiv:2410.02763, 2024. 1, 5, 13, 14
- [124] Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, Songyang Zhang, Wenwei Zhang, Yining Li, Yang Gao, Peng Sun, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang,

- Hang Yan, Conghui He, Xingcheng Zhang, Kai Chen, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output, 2024. 1, 14
- [125] Rui Zhang, Yongsheng Zhou, Qianyi Jiang, Qi Song, Nan Li, Kai Zhou, Lei Wang, Dong Wang, Minghui Liao, Mingkun Yang, et al. Icdar 2019 robust reading challenge on reading chinese text on signboard. In 2019 international conference on document analysis and recognition (ICDAR), pages 1577–1581. IEEE, 2019. 37
- [126] Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, and Yiming Yang. Direct preference optimization of video large multimodal models from language model reward, 2024. 9
- [127] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713,

2024. 1, 5, 6, 12, 14, 15, 16, 36, 37

- [128] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024. 1, 3, 15
- [129] Yipin Zhou, Zhaowen Wang, Chen Fang, Trung Bui, and Tamara L Berg. Visual to sound: Generating natural sound for videos in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3550–3558, 2018. 37
- [130] Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe HansenEstruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, Serena YeungLevy, and Xide Xia. Apollo: An exploration of video understanding in large multimodal models, 2024. 1, 5, 15

### A Training hyper-parameters

- Table 12 shows the training hyper-parameters in pre-training, SFT-1&2 and DPO stage. We apply a layer-wise learning rate decay of 0.9 for visual encoder training [22].

Configuration Pre-training SFT-1 SFT-2 DPO

VLM init. Qwen2-VL-7B Tarsier2-Pre-trian Tarsier2-SFT-1 Tarsier2-SFT-2 Optimizer name AdamW

- Optimizer β1 0.9

- Optimizer β2 0.999 Optimizer eps 1e−6 Learning rate 2e−5 2e−5 2e−6 1e−6 Learning rate schedule cosine Training steps 200,000 5,000 5,000 1,000 Warm-up steps 1,000 250 250 100 Weight decay 0.01 Gradient clip 1.0 Dropout rate 0.0 Global batch size 384 64 64 64 Max pixels 460,800 Frames per video [8,128] 16 16 16 Numerical precision bfloat16

Table 12: Training hyper-parameters of Tarsier2

B Public datasets of pre-training stage

- Table 13 presents the pre-training datasets, which collectively include approximately 20 million public data and 20 million in-house data. Most of the public datasets are the same as Tarsier1, except we additionally gathered some newly released open-source data and OCR-releated data. For WebVid-10M, we used 2.9 million video-text pairs, selecting samples that are more likely to feature dynamic events. We have also incorporated some latest long video understanding datasets, such as MovieStory101[38] and LLaVA-Video178K [127]. This greatly enhances the model’s ability to understand long videos.

### C Annotation process for SFT data

In the first stage of SFT, we annotated each video clip with detailed descriptions that included fine-grained temporal grounding. Each clip first underwent manual annotation, where annotators described dynamic information such as character actions, events, scene

Video Captioning WebVid [10] (2.9M) LSMDC [92] (109K) TGIF [59] (105K) ActivityNet [47] (38K) Charades [97] (16K) Charades-Ego [96] (6K) YouCook2 [129] (9K) TACoS [90] (18K) Ego4D [35] (1.1M) Spoken Moments [82] (493K) Multi-Moments [83] (997K) TREC-VTT [7] (64K) ShareGPT-4o-video [26] (2K) MovieStory101[38] (11K) GPT4o-labeled Caption† (2.5M) Human-labeled Caption† (145K) Film&TV Commentary† (11.5M)

Action Recognition HMDB [49] (5.8K) COIN [101] (10K) SSV2 [34] (169K) Kinetics-700 [13] (537K) FineAction [68] (82K) RareAct [80] (2K) 20BN-jester [79] (46K)

Video QA CLEVRER [120] (83K) TGIF-QA [43] (72K) EgoQA [29] (5K) VideoInstruct [76] (89K) LLaVA-Video-178K [127] (165K) M4-Instruct-video [52] (255K) GPT4o-labeled QA† (16.2K)

Grounding DiDeMo [4] (82K) AVA [36] (28K) E.T. Instruct 164K [67] (147K) Object Tracking† (745K)

Video Self-Supervised Training Frame Order Prediction† (825K)

Intent Recognition Oops! [28] (15K)

Multi-Image Understanding VIST [40] (38K) MMDU [71] (45K) M4-Instruct-image [52] (616K) Image Retrival† (533K)

Single-Image Understanding ShareGPT4V [15] (95K) LLaVA-1.5 [64] (643K) ShareGPT-4o-image[26] (57K) MS COCO [63] (566K) Flicker [87] (145K) LLaVA-ReCap-CC3M [52] (2.9M) Visual Genome [48] (759K) SBU Captions [84] (860K) GPT4o-labeled Caption† (1.13M)

Image OCR RCTW-17 [95] (8K) LSVT [98] (430K) ReCTS [125] (20K) Art [11] (5.6K) COCOTextV2 [103] (16K) CORD-v2 [85] (1K) HierText [73] (10K) MSRA-TD500 [118] (465) IC03 [74] (499) SynthDoG-en [46] (100K) SynthDoG-zh [46] (100K)

Text Generation OpenOrca [60] (995K) ShareGPT [24] (80K)

- Table 13: Datasets and their sizes used in Tarsier2 pre-training. † indicates in-house datasets.

transitions, and camera movements, while avoiding unnecessary static elements. Annotators are also required to map the dynamic information in their descriptions to the corresponding frame numbers. We performed quality inspections on the annotated data and returned any data not meeting quality standards for re-annotation. We discarded any data that might involve copyright risks.

In the second stage of SFT, we utilized GPT-4o to generate a variety of instruction tuning samples based on manual annotations. We provided GPT-4o with 16 uniformly sampled frames from the video and the original manual annotations. Figure 8 shows the prompt for re-annotation in this stage.

### D Detail setting of DPO training

As a default setting, we leveraged the negative sampling and preference pair filtering strategy as introduced in Section 3.3 to construct the DPO training set. We set top p as 0.7 and temperature as 0.7 when running both positive sampling and negative sampling on our

###### The re-annotation prompt for diverse instruction data (SFT-2).

###### Character

You are an excellent video analyst. Utilizing your incredible attention to detail, you provide clear, sequential descriptions for video. You excel in identifying and conveying changes in actions, behaviors, environment, states and attributes of objects, and camera movements between video frames.

###### Prompt

Here are 16 frames from a video and a short video caption in Chinese. You need to process a two step tasks: First, establish a set of guiding principles to control the style of the video description. These principles should include one or more of the following aspects:

- 1. Specify the length constraints of the description, including the number of paragraphs and total word count.
- 2. Define the level of detail for human or creature appearance, non-creature appearance, and background.
- 3. Determine the granularity of the event information.
- 4. Decide on the output format, such as plain text, JSON, lists, narrative, poetry, etc.
- 5. Choose the output language, such as Chinese, English, Japanese, French, and so on.
- 6. Decide on the text style, such as fluent, concise, professional, or just using simple words and phrases.

Next, generate the corresponding video description based on these guiding principles and the input video clip, and rephrase the guiding principles into natural language as part of the output question. Input Origin Short Video Caption in Chinese: {Manual Labeled Chinese Caption} Requirement Return in JSON format: {“qustion”: xxx,“answer”: xxx}

Figure 8: The re-annotation prompt in SFT-2.

150K SFT dataset. The threshold δ of preference pair filtering was set as 0.3. We finally randomly sampled 20K preference pairs for DPO training. For the “w/o NS” setting, we kept other parameters and process unchanged but replaced the negative sampling with an additional positive sampling. For the “w/o PF” setting, we omitted the process of preference pair filtering and directly sample 20K pairs from all preference pair candidates. We utilized the vanilla DPO training objective (Equation 2), and set β as 0.1. See the “DPO” column of Table 12 for all the other hyper-parameters.

### E Detailed results of individual datasets at different stages In this section, we provide detailed results for individual datasets in our ablation study.

- Table 14, 15 and 16 list the results for pre-training, SFT and DPO respectively. Table 17 lists the results for the recaptioning experiment. We report F1/Precision/Recall for DREAM-1K and accuracy for other benchmarks.

### F Tarsier2-Recap-585K Data Composition

Table 18 lists the data composition details of Tarsier2-Recap-585K. We mainly took video caption datasets into account when picking the target datasets, together with two action

###### Capability Benchmark Tarsier1-7B Tarsier1-7B-Qwen Tarsier2-7B

DREAM-1K 34.6/30.2/40.3 38.4/40.6/36.4 40.8/42.5/39.3 TempCompass-cg 55.3 59.3 60.1 Vinoground-Text 29.8 48.6 60.2

Caption

MVBench 62.6 69.8 72.8 TVBench 45.8 51.0 53.5 TOMATO 28.6 36.5 39.5

Video QA Short

Video-MME 42.2 58.9 65.3 LongVideoBench 39.8 52.1 58.3 TemporalBench 56.9 61.9 68.7

Video QA Long

EventHallusion-Y/N 70.9 75.6 77.8 EventHallusion-Desc 41.6 48.6 49.1

Hallucination

- Table 14: Detailed results of the ablation study for pre-training. For the captioning task, results are reported after the SFT stage. For other tasks, results are reported after the pre-training stage.

###### Tarsier2-7B

Capability Benchmark

pre-train

SFT w/o grounding SFT

DREAM-1K 35.2/36.8/33.7 37.4/38.6/36.3 40.8/42.5/39.3 TempCompass-cg 50.5 50.2 60.1 Vinoground-Text 57.2 60.6 60.2

Caption

MVBench 72.8 71.9 72.5 TVBench 53.5 54.5 54.2 TOMATO 39.5 41.3 41.9

Video QA Short

Video-MME 65.3 64.0 64.7 LongVideoBench 58.3 54.7 58.2 TemporalBench 68.7 66.9 66.6

Video QA Long

EventHallusion-Y/N 77.8 80.1 84.4 EventHallusion-Desc 49.1 56.2 59.4

Hallucination

Table 15: Detailed results of the ablation study for SFT.

recognition datasets (Kinetics-700 [13] and SSV2 [34]), which contain video clips of durations of 5 ∼ 10 seconds about human actions, and a special intent recognition dataset (Oops [28]) to help models learn rare actions and unexpected events. For most of the datasets, we utilized all the original video clips of the selected splits (usually train and val set), except for:

- • WebVid-10M: We sampled around 30% of the total size of Tarsier2-Recap-585K from a pre-filtered subset of WebVid-10M, which are more likely to feature dynamic events.
- • Ego4D: We randomly merged multiple clips into a new one that contains multiple

###### Capability Benchmark Tarsier2-7B w/o DPO w/o NS w/o PF

DREAM-1K 42.0/42.8/41.1 40.8/42.5/39.3 41.5/44.5/39.0 40.5/39.9/41.1 TempCompass-cg 66.6 60.1 62.1 65.1 Vinoground-Text 65.8 60.2 60.6 67.6

Caption

MVBench 71.5 72.5 72.2 71.7 TVBench 54.7 54.2 54.9 54.6 TOMATO 42.0 41.9 41.3 41.8

Video QA Short

Video-MME 64.5 64.7 64.3 64.4 LongVideoBench 58.6 58.2 58.6 57.4 TemporalBench 65.3 66.6 65.4 65.2

Video QA Long

EventHallusion-Y/N 84.6 84.4 85.1 84.8 EventHallusion-Desc 63.3 59.4 60.7 63.5

Hallucination

Table 16: Detailed results of the ablation study for DPO.

Capability Benchmark Qwen2-VL-7B [106] + Original FT + Recaption FT

DREAM-1K 29.6/33.9/26.3 35.2/44.8/29.0 39.5/41.7/37.6 TempCompass-cg 54.2 49.9 67.7 Vinoground-Text 40.0 39.0 55.0

Caption

MVBench 67.0 59.8 66.8 TVBench 43.8 47.2 51.1 TOMATO 31.5 33.6 39.5

Video QA Short

Video-MME 63.3 56.1 57.0 LongVideoBench 55.6 51.4 51.9 TemporalBench 62.0 58.7 61.4

Video QA Long

EventHallusion-Y/N 68.6 39.6 80.7 EventHallusion-Desc 27.8 46.3 56.2

Hallucination

Table 17: Detailed results of the recaptioning experiment.

actions and result in around 1M merged clips in total. We sampled 50K clips from this dataset for recaptioning.

- • Kinetics-700 and SSV2: We randomly sampled 50K and 10K clips from the training set of Kinetics-700 and SSV2, respectively.

### G Qualitative Comparison of the SFT Process

- Figure 9 illustrates a qualitative comparison of our model at different stages, where we mark the differences in the prediction results of different models. From these differences, it can be seen that introducing temporal localization information in the first SFT stage significantly reduces the problem of hallucination in the model. However, the introduction of temporal localization information may also result in certain events being subdivided

###### Dataset Original Label Type Split Avg Duration (s) # Sampled Clips Proportion (%) WebVid-10M [10]

- 15.2 177,909 30.38 LSMDC [92] train/val/test 4.1 108,271 18.49 TGIF [59] train/test 12.3 94,775 16.18 Ego4D [35] - 4.1 50,000 8.54 ActivityNet [47] train/val/test 35.7 35,960 6.14 VATEX [107] train/val/test 10.0 22,435 3.83 TREC-VTT [7] train/val 6.3 14,199 2.42 Charades [97] train/test 29.8 7,985 1.36 Charades-Ego [96] train/test 30.2 6,161 1.05

Video Caption

Kinetics-700 [13]

train/val/test 8.9 50000 8.50 SSV2 [34] train/val/test 3.7 10000 1.71 Oops [28] Intent Recognition train/val 9.8 7,948 1.36 Sum - - 1,972 hours 585,643 100.00

Action Recognition

Table 18: Data composition of Tarsier2-Recap-585K. The “Split” column lists the original dataset partitioning, and we use bold to mark the parts which we sampled the video clips from to conduct recaptioning.

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

###### 1 2 3 4 5 6 7 8

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

###### 9 10 11 12 13 14 15 16

Reference: A person stands in the swimming pool, with both hands braced against the edge of the pool. With a forceful leap, they step onto the ground and finally stand up and walk forward out of the frame.

Pretrain: A woman wearing a black swim cap and goggles is seen in a swimming pool. She is holding onto the edge of the pool with both hands. The pool has clear blue water with lane dividers visible in the background. The poolside is lined with a beige-colored deck. The woman then releases her grip on the pool edge, submerges herself underwater, and swims forward. After swimming, she emerges from the water, places her hands on the pool edge, and begins to climb out of the pool. She steps onto the deck, leaving the pool area.

- SFT-1: <frame: 1> A woman is seen in a swimming pool, holding onto the edge with both hands. <frame: 2-5> She begins to submerge herself into the water, still holding the edge. As she goes underwater, her hands remain on the edge for support. <frame: 6> She then starts to push herself up from the water, preparing to stand. <frame: 7-8> The woman uses her hands to push herself up onto the pool edge, with water splashing around her. <frame: 9-10> She continues to lift her body out of the water, placing her feet on the edge. <frame: 11-12> The woman stands up on the pool edge, with water dripping from her body. <frame: 13-16> She then walks away from the pool, leaving the edge behind.

- SFT-2: A woman in a black swimsuit and swim cap is in a swimming pool, holding onto the edge with both hands. She submerges her head underwater while still holding the edge. She then pushes herself up from the water, using her hands for support. The woman stands up in the shallow water, water dripping from her body. She steps out of the pool, walking along the edge. The pool area is empty after she leaves.

Figure 9: Qualitative comparison of our model at different stages.

into finer actions. To address this issue, the second stage of training further improved the accuracy of the model description and optimized the output style.

### H DREAM-1K cases

Figure 10∼14 display the detailed video descriptions generated by Tarsier2-7B and other models (GPT-4o, Gemini-1.5-Pro and LLaVA-Video-7B-Qwen2) for different video categories in DREAM-1K. Click the play button on the first frames to view the raw video. The

###### correct descriptions of key objects/actions/events are marked in green, and the incorrect descriptions are marked in red.

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

1 2 3 4

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

5 6 7 8

GPT-4o: A person is seen in a dimly lit room filled with glass bottles and various objects. The person lights a match and ignites a substance, causing a brief flame. They then use a syringe to mix a green glowing liquid in a bowl. The person retrieves tools from a mirrored tray. The video focuses on the person's precise actions of lighting, mixing, and handling objects in the room.

Gemini-1.5-Pro: A young woman wearing a black dress stands at a bar, listening to someone out of frame. She then uses a long metal tool to scoop burning rose petals from one small bowl to another, creating a shower of sparks. As the sparks fall into the second bowl, they transform into a bubbling, bright green liquid. The woman uses a syringe to draw some of the liquid out of the bowl before placing the syringe onto a tray beside a pair of metal tongs.

LLaVA-Video-7B-Qwen2: The video opens with a shot of a person wearing a black sleeveless top and multiple necklaces, standing in front of a bar counter. The background is dimly lit, with shelves filled with various bottles and glasses. The scene transitions to a close-up of a hand holding a lighter, igniting a flame that is used to light a bowl containing red rose petals. The next frame shows the same person, now holding a small object that emits a spark, possibly a lighter or a matchstick. Following this, there is a close-up of a bowl with a green liquid that has a glowing effect, being stirred with a metal tool. The final frames show a hand placing a metallic object, which appears to be a syringe, on a reflective surface, possibly a mirror.

Tarsier2-7B: A woman with braided hair and wearing a black outfit with multiple necklaces and bracelets looks at a person in front of her in a room filled with various bottles and glassware. She lights a match and uses it to ignite rose petals in a small bowl, creating a burst of sparks. She then uses a syringe to draw a green liquid from a bowl and places the syringe on a reflective tray along with some metal tools.

- Figure 10: Qualitative comparative analysis of various Video-MLLMs on Dream-1K dataset (Live-action Subset).

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

1 2 3 4

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

5 6 7 8

GPT-4o: The video clip features two children on a rooftop. They are initially seen pulling on a string that is attached to a colorful kite. The kite is then launched into the sky, and the children look up as it ascends. The background shows a cityscape with buildings. The children continue to hold the kite string and watch the kite as it flies higher into the sky.

Gemini-1.5-Pro: A boy and a girl are standing on a rooftop preparing to launch a kite. With a powerful throw, the kite soars high into the air, resembling a team of three dogs pulling a sled. The children watch with delight as their kite dances in the wind. However, their joy turns to concern as they notice the kite string has snapped, leaving them staring at the broken connection with worry.

LLaVA-Video-7B-Qwen2: The video opens with two children standing on a rooftop, one holding a kite string and the other holding a kite. The child with the string is wearing a striped shirt and shorts, while the other child is in a green dress with star patterns. They are both looking up at the sky, which is a clear blue with a few clouds. The kite is colorful, predominantly pink with yellow accents and has a long tail. As the video progresses, the children appear to be struggling to control the kite, which is flying erratically. The background shows a cityscape with buildings and trees below them. The video ends with the children looking at each other, seemingly discussing their next move.

Tarsier2-7B: Two children run to the edge of a rooftop and throw a colorful kite into the sky. The kite starts to fly, but it soon begins to descend. The children, holding the kite string, look surprised and lower their hands. They then look at the kite string and start talking. Suddenly, the kite returns, and the children react to its sudden return.

- Figure 11: Qualitative comparative analysis of various Video-MLLMs on Dream-1K dataset (Animation Subset).

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

1 2 3 4

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

5 6 7 8

GPT-4o: A person in white clothing approaches a bar counter and places two cocktails on it. The cocktails are in clear, patterned glasses, one with a slice of orange and the other with a slice of lime. Two other individuals, one with a visible wristwatch and the other with painted nails, reach for the cocktails. The person with painted nails picks up the cocktail with the lime, gestures with it, and then places it back down on the counter. Both individuals hold the cocktails in their hands momentarily before setting them back on the bar counter. The background consists of a variety of bottles and dim lighting, typical of a bar setting.

Gemini-1.5-Pro: A blurry shot of a brightly lit bar with numerous bottles and glasses provides a lively backdrop to the scene. A bartender, dressed in white, leans into the frame and places two colorful cocktails on the counter. The cocktails, a pale yellow drink garnished with an orange slice and a vibrant blue drink with a lime wedge, are each adorned with two black straws. A man and woman, seated at the bar, reach out, the woman taking the blue drink and the man selecting the yellow. They bring their glasses together in a toast, the clinking sound barely audible above the implied music and chatter of the bar. The hands retract, leaving the cocktails standing on the bar as the focus shifts entirely to the vibrant ambiance of the setting.

LLaVA-Video-7B-Qwen2: The video opens with a close-up of a bar counter, where the focus is on the gleaming surface and the blurred background filled with colorful bottles. The lighting is dim, creating a moody atmosphere typical of a bar setting. A bartender, dressed in a white shirt, appears holding two glasses filled with drinks—one with ice and an orange slice garnish, the other with a dark liquid and a lime wedge. The bartender places the drinks on the counter, and a customer's hand reaches out to take one of the drinks. The scene transitions to a similar close-up of the bar counter, where the same two drinks are now being held by the customer's hands. The customer takes a sip from the drink with the orange slice, and the bartender is seen in the background, partially obscured by the drinks. The final scene shows the bar counter again, this time with the two drinks placed side by side. The customer's hand is no longer visible, but the drinks remain untouched, emphasizing their presentation.

Tarsier2-7B: The bar counter is shown with a blurred background of various bottles and colorful lights. A bartender approaches the counter, holding two drinks. The bartender places the two drinks on the counter, one with an orange slice and the other with a lime slice, both with black straws. Two hands, one from the left and one from the right, reach for the drinks. The person on the left picks up the drink with the orange slice, and the person on the right picks up the drink with the lime slice. The counter is left empty momentarily. The person on the right places the drink with the lime slice back on the counter, followed by the person on the left placing the drink with the orange slice back on the counter. Both drinks are now back on the counter, ready to be enjoyed.

- Figure 12: Qualitative comparative analysis of various Video-MLLMs on Dream-1K dataset (Stock Subset).

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

1 2 3 4

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

5 6 7 8

GPT-4o: A person is seen holding a small item over a filled trash bin. The bin is shown with various items including bottles. Text on screen indicates the bin is filled, and the person appears to be contemplating options. The person then presses a button on the bin. The trash bin begins to compact the trash inside and seals the trash bag automatically. The sealed trash bag is then lifted out from the bin. The video demonstrates the functionality of an automatic trash bin that can seal and lift the trash bag with the press of a button.

Gemini-1.5-Pro: The video starts with a hand putting trash into a bin that's already full. The text on the screen highlights the dilemma of a full trash bin. Then, it introduces a solution: a smart trash bin called "Townew". By pressing a single button, the Townew bin automatically seals the full trash bag and replaces it with a new one. A person's hand is shown pulling out the sealed bag from the bin.

LLaVA-Video-7B-Qwen2: The video begins with a close-up of a white trash bin filled to the brim with various types of waste. A hand is seen holding a small white object, presumably a button or sensor, above the bin's lid. The text on the screen reads "Your bin is filled to the brim." The scene transitions to the same bin with the text changing to "But you are lazy." The next frame shows the bin without any text, followed by the text "What options do you have?" The video then cuts to a plain white background with the text "Check out this" in the center. The final frame returns to the trash bin, now with a new feature: a bluelit compartment at the top. The text on the screen reads "By pressing one button, Townew deals with it all." The video continues with a close-up of the same white trash bin, now with the blue-lit compartment visible. The bin is shown from a side angle, and the compartment appears to be in operation, as indicated by the blue light. The background remains plain white, emphasizing the bin and its features.

Tarsier2-7B: A hand holding a small object approaches an overflowing trash bin, with the text 'Your bin is filled to the brim' appearing on the screen. The hand drops the object into the bin, and the text changes to 'But you are too lazy.' The hand retracts, and the text 'What options do you have?' appears. The screen briefly shows 'Check out this' before transitioning to 'By pressing one button, Tounew deals with it all.' The hand presses a button on the bin, causing the lid to close and the trash to be compressed. The bin's mechanism retracts a black garbage bag, and the text disappears. A hand reaches in and pulls out the full garbage bag from the bin.

- Figure 13: Qualitative comparative analysis of various Video-MLLMs on Dream-1K dataset (Youtube Subset).

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

###### 1 2 3 4 5 6 7 8

GPT-4o: A person is involved in processing and crafting shells. Initially, shells are collected in a basket and washed thoroughly in water. The person then uses a foot-powered saw to cut the shells into strips. The saw is operated by moving a foot pedal up and down, which in turn moves the blade. The cut shell strips are then sorted and further refined by using a smaller tool to make precise cuts. The person carefully holds the shells and ensures they are properly cut and shaped.

Gemini-1.5-Pro: A man is cleaning an iridescent abalone shell. He then uses a foot-powered saw to cut another abalone shell into thin strips. He secures the shell to a wooden frame with twine and carefully saws through it, creating a pile of shimmering strips.

LLaVA-Video-7B-Qwen2: The video begins with a close-up of a basket filled with various seashells, some of which are open to reveal their interiors. The shells have a glossy appearance and come in different colors, including shades of green, brown, and white. The next scene shows hands holding the basket, which is placed on a surface that appears to be wet, possibly indicating it was recently submerged in water. Following this, there is a shot of water being poured over a surface, creating ripples and bubbles, suggesting a natural setting like a stream or river. The subsequent frames show hands handling a large, rough-textured shell, possibly preparing it for further processing. The hands are seen using a tool, likely a knife, to cut or scrape the shell. The final scenes depict hands arranging thin, elongated pieces of shell on a flat surface, possibly for drying or further processing.

Tarsier2-7B: A basket filled with several shells is placed in water, and a pair of hands lifts the basket out of the water. The basket, now containing various shells, is placed on a surface. A person, wearing dark shoes, uses a foot to hold the shell in place while sawing it with a handsaw. The person continues to saw the shell into strips, which are then collected and tied together with a string. The person then uses the saw to further refine the shell strips.

- Figure 14: Qualitative comparison of different Video-MLLMs on Dream-1K dataset (Shorts Subset).

