# arXiv:2412.02611v1[cs.CV]3Dec2024

###### AV-Odyssey Bench: Can Your Multimodal LLMs Really Understand Audio-Visual Information?

Kaixiong Gong1*, Kaituo Feng1*, Bohao Li2*†, Yibing Wang, Mofan Cheng, Shijia Yang3, Jiaming Han1, Benyou Wang2‡, Yutong Bai4, Zhuoran Yang5, Xiangyu Yue1‡

1CUHK MMLab, 2CUHK (SZ), 3Stanford University, 4UC Berkeley, 5Yale University

https://av-odyssey.github.io/

###### Abstract

###### Loudness Comparison

###### Pitch Comparison

Question: Which audio is louder? The first audio [audio1] or the second audio [audio2]?

Question: Which audio has a higher pitch? The first audio [audio1] or the second audio [audio2]?

Recently, multimodal large language models (MLLMs), such as GPT-4o, Gemini 1.5 Pro, and Reka Core, have expanded their capabilities to include vision and audio modalities. While these models demonstrate impressive performance across a wide range of audio-visual applications, our proposed DeafTest reveals that MLLMs often struggle with simple tasks humans find trivial: 1) determining which of two sounds is louder, and 2) determining which of two sounds has a higher pitch. Motivated by these observations, we introduce AV-Odyssey Bench, a comprehensive audiovisual benchmark designed to assess whether those MLLMs can truly understand the audio-visual information. This benchmark encompasses 4,555 carefully crafted problems, each incorporating text, visual, and audio components. To successfully infer answers, models must effectively leverage clues from both visual and audio inputs. To ensure precise and objective evaluation of MLLM responses, we have structured the questions as multiple-choice, eliminating the need for human evaluation or LLM-assisted assessment. We benchmark a series of closed-source and open-source models and summarize the observations. By revealing the limitations of current models, we aim to provide useful insight for future dataset collection and model development.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

- [audio1]
- [audio2] the second audio

- [audio1]
- [audio2]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

the first audio

Gemini 1.5 Pro

Gemini 1.5 Pro

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

the first audio

the second audio

GPT-4o

GPT-4o

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

the first audio

the second audio

Reka Core

Reka Core

Figure 1. Illustration of two out of four DeafTest tasks. Loudness comparison is used to determine the louder sound of two given sounds. Pitch comparison is to determine which sound has the higher pitch.

abling them to tackle a wide array of vision-language tasks [80]. These models demonstrate remarkable capabilities, including counting objects in images [78], performing numerical calculations on tabular data [82], and solving geometric problems with provided figures [93]. Building upon this foundation, Multimodal Large Language Models (MLLMs)1 have further expanded their capabilities by incorporating audio modality, e.g., GPT-4o [27] and Gemini 1.5 [70]. These advancements push the boundaries of multimodal reasoning, particularly in areas including automatic speech recognition (ASR) [27], automatic speech translation (AST) [70], audio-visual captioning [23, 89], and general audio-visual processing [23].

###### 1. Introduction

Multimodal Large Language Models have evolved progressively, beginning with vision language models. Vision Language Models (VLMs), exemplified by GPT-4V(ision) [55], have endowed language models with visual perception, en-

Most previous multimodal benchmarks focus on visual problem-solving [30, 42, 86], e.g., general vision comprehension [21, 32, 42] and multimodal mathematical reasoning [8–10, 48, 49, 93]. To evaluate audio-visual comprehension, dedicated benchmarks such as AVQA [81] and Mu-

*Equal Contribution †Project Leader ‡Corresponding Authors: wangbenyou@cuhk.edu.cn and

1In this paper, MLLMs only refer to the audio-vision LLMs. We use VLM to refer to the vision-language LLMs.

xyyue@ie.cuhk.edu.hk

###### Comprehensive Audio Attributes

###### Extensive Domains

###### Interleaved Text, Audio, Images

Question: Considering the vocal timbre, which person from [img1], [img2], [img3], or [img4] is most likely to sing [audio1]?

[Figure 21]

Timbre Tone

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

audio content: an AI covered song with Taylor Swift’s timbre

[Figure 32]

[audio1]

[Figure 33]

[Figure 34]

[Figure 35]

Space Melody

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[img1]

[img2]

Hallucination Time Intricacy Music, Weapons, Animals, Materials,

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Transportation, Scenes, Actions, Speech, Memes, Daily life

[img3] [img4]

- Figure 2. Overview of AV-Odyssey Benchmark. AV-Odyssey Bench demonstrates three major features: 1. Comprehensive Audio Attributes; 2. Extensive Domains; 3. Interleaved Text, Audio, and Images.

Table 1. Results on four basic auditory tasks (DeafTest). The questions are designed as two-choice questions. The random baseline performance is 50%.

sicAVQA [33] have been introduced, assessing models in several audio-visual tasks [33, 37, 68, 81, 88].

However, we primarily identify two issues of previous audio-visual benchmarks: 1) They overlook assessing the basic listening ability (no reasoning) of MLLMs. 2) They are limited in evaluation dimensions (e.g., restricted audio attributes and narrow application domains), as illustrated in Table 2. This leads to an insufficient evaluation of the audio-visual information understanding capability of MLLMs. These findings raise the central question we aim to investigate in this paper: ⋆ Can Multimodal LLMs Truly Understand Audio-Visual Information? To investigate this, we propose DeafTest to assess the basic listening ability of MLLMs, along with the comprehensive AVOdyssey Benchmark to further assess their audio-visual understanding capabilities.

Sound Counting

Loudness Comparison

Pitch Comparison

Duration Comparison

Method

Random 50.0 50.0 50.0 50.0 Gemini 1.5 Flash [70] 55.0 62.0 54.0 89.0 Gemini 1.5 Flash-8B [70] 49.0 55.0 51.0 51.0 Gemini 1.5 Pro [70] 81.0 60.0 52.0 84.0 Reka Core [71] 54.0 43.0 42.0 40.0 Reka Flash [71] 48.0 58.0 51.0 44.0 Reka Edge [71] 47.0 56.0 50.0 44.0 GPT-4o audio-preview [27] 50.0 58.0 58.0 57.0

- 3. MLLMs also struggle to discriminate the higher pitch between two given sounds. None of the models can score above 60%.
- 4. The duration comparison task presents models with two sounds and asks them to determine which has the longer duration. Two Gemini models perform significantly better than the others that are merely on par with the random baseline.

DeafTest, a suite of four low-level auditory tasks (examples are illustrated in Figure 1), is introduced to examine the fundamental ability of MLLMs to listen, inspired by the Schwabach test in audiology [26] and BlindTest [61] in vision domain. We evaluate Gemini 1.5 [70], Reka [71], and GPT-4o [27] on our four simple tasks that only involve basic elements of sound (e.g., loudness, pitch, and duration) as shown in Table 1. The key findings are:

The results suggest that while MLLMs excel at some high-level multimodal reasoning tasks, they still have limitations in basic listening abilities, which may further hinder their integration of audio-visual information (hypothesis ▲). DeafTest allows us to critically assess whether MLLMs truly understand audio-visual information or if their apparent capabilities are limited to surface-level pattern recognition. By doing so, we aim to contribute to a deeper understanding of multimodal comprehension in current MLLMs and identify areas for future improvement—issues that have been overlooked by previous audiovisual benchmarks [33, 37, 81].

- 1. Despite their ability to recognize complex speech content, MLLMs do not perform as well as expected on sound counting tasks. The best-performing model, Gemini 1.5 Pro, achieves only 81%, while humans can easily score 100%. The sounds in these tasks are monotonous and are clearly separated by silent intervals within the audio clip.
- 2. MLLMs appear to be insensitive to sound volume. When two sounds, one louder than the other, are fed into models, they often fail to distinguish the louder one. All models perform below 65%, significantly lower than the expected 100%.

To further investigate question ⋆, we introduce AVOdyssey, a comprehensive audio-visual benchmark designed to challenge MLLMs by requiring them to lever-

Table 2. Comparisons between MLLM benchmarks / datasets.

Customized Question

Audio Attributes Multiple Domains

Interleaved Timbre Tone Melody Space Time Hallucination Intricacy

Benchmark / Dataset Modality Questions Answer Type

MME Bench [21] Image 2194 Y/N ✓ - - - - - - - ✓ ✗

MMBench [42] Image(s) 2974 A/B/C/D ✓ - - - - - - - ✓ ✗ SEED-Bench-2 [32] Image(s) & Video 24371 A/B/C/D ✓ - - - - - - - ✓ ✓ AVQA Dataset [81] Video & Audio 57335 A/B/C/D ✓ ✓ ✗ ✗ ✗ ✓ ✗ ✓ ✓ ✗

Pano-AVQA Dataset [88] Video & Audio 51700 defined words & bbox ✓ ✓ ✓ ✗ ✓ ✗ ✗ ✓ ✓ ✗ Music-AVQA Dataset [33] Video & Audio 45867 defined words ✓ ✓ ✗ ✓ ✓ ✓ ✓ ✓ ✗ ✗

SAVE Bench [68] Image & Video & Audio 4350 free-form ✗ ✓ ✗ ✗ ✓ ✗ ✗ ✓ ✓ ✗ OmniBench [37] Image & Audio 1142 A/B/C/D ✓ ✓ ✗ ✗ ✗ ✓ ✗ ✗ ✓ ✗

AV-Odyssey Bench (ours) Image(s) & Video & Audio(s) 4555 A/B/C/D ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

age information from all input modalities to derive accurate answers. This meticulously crafted dataset encompasses 4,555 carefully selected questions spanning 26 distinct tasks, with each question strategically constructed to include three critical components: text, image/video, and audio clip.

To ensure the benchmark’s rigor, we employ vision and audio language models to filter out questions that could be easily resolved by single-modal approaches. This guarantees that only complex, multi-dimensional questions remain, truly testing the models’ ability to integrate information across modalities. AV-Odyssey is designed with a broad scope, covering a wide range of sound attributes, including timbre, tone, spatial characteristics, and temporal dynamics, while simultaneously exploring diverse application domains such as music, daily life, and transportation, as illustrated in Figure 2. By structuring the evaluation as multiple-choice selections, we’ve streamlined the assessment process, eliminating the need for manual verification or LLM-assisted evaluation.

We conduct extensive experiments on closed-source model [27, 70, 71] and open-source models [15, 22, 23, 47, 67, 77, 89] on the proposed AV-Odyssey Bench. Considering the results of DeafTest and AV-Odyssey, we have the following findings:

- • Overall, current MLLMs still fall short in processing complex audio-visual information integration tasks.
- • The audio-caption-vision training paradigm fails to effectively combine audio and visual modalities, limiting the integration of audio-visual information (see Sec. 4.2).
- • Through error analysis (see Sec. 4.3), we find out that the major cause of error in audio-visual inference is the perceptual error of audio input, which is in line with the hypothesis ▲.

In a nutshell, this paper explores whether MLLMs really understand audio-visual information from two aspects. To begin with, we propose DeafTest to evaluate MLLMs and find out that these MLLMs still have obvious limitations in basic listening ability. This could lead to a bottleneck in subsequent audio-visual information integration for complex audio-visual reasoning, which is then

validated by the results and analysis of our proposed AVOdyssey benchmark.

###### 2. Related Work

Multimodal Large Language Models. Large language models (LLMs) have demonstrated remarkable performance across diverse textual domains [6, 7, 55, 60, 73]. The success of these models has catalyzed significant advancements in vision language models and multimodal large language models. Inspired by the textual prowess of LLMs, vision language models have emerged to extend computational capabilities into visual comprehension. These models enable LLMs to perform sophisticated visual tasks, including visual question answering [5, 18, 34, 39, 40, 85, 91, 96], visual grounding [11, 12, 58, 76], document understanding [25, 50, 85, 94], long video understanding [36, 41, 62, 64, 92]. Building upon vision-language achievements, researchers have further expanded multimodal horizons by integrating the audio modality [15, 22, 23, 47, 67, 77, 89]. These advanced models now accommodate audio inputs, further expanding the landscape of multimodal artificial intelligence.

Benchmarking Multimodal Large Language Models. The rapid development of vision language models has been accompanied by the emergence of specialized benchmarks to assess their performance across various domains [9, 21, 32, 49, 87]. A significant subset of these benchmarks focuses on vision comprehension [21, 31, 32, 87] and mathematical reasoning capabilities [8, 9, 49, 63, 87, 93]. However, current audio-visual benchmarks [33, 37, 68, 81, 88] face significant limitations in comprehensively assessing multimodal large language models (MLLMs). Firstly, they predominantly focus on high-level visual tasks and neglect to explore the basic auditory perception limitations. Secondly, they are limited in application domains. For example, Music-AVQA [33] limits evaluation to the music domain, and AVQA [81] primarily tests daily life applications. Thirdly, they do not comprehensively evaluate all attributes of the audio. In contrast, this paper introduces DeafTest tasks to evaluate fundamental capabilities and the AV-Odyssey benchmark, which spans a wide spectrum of

audio attributes and application domains, enabling a comprehensive assessment of the audio-visual comprehension performance of MLLMs.

###### 3. Method

###### 3.1. DeafTest Tasks

Drawing inspiration from the Schwabach test [26], we introduce DeafTest, a suite of four simple auditory tasks that critically examine the fundamental audio perception capabilities of Multimodal Large Language Models (MLLMs). DeafTest includes the determination of the number of sounds, identification of the louder sound, recognition of the sound with a higher pitch, and detection of the sound with a longer duration. We hypothesize that MLLMs may not perform as well as expected on these basic tasks. This potential shortcoming arises from the training objectives of these models, which primarily focus on achieving high-level semantic alignment between different modalities. Consequently, this approach tends to overlook the effective utilization of low-level auditory information, which is crucial for accurately processing and understanding basic sound characteristics.

###### 1. Count the Number of Sounds. Given that Mul-

timodal LLMs achieve impressive performance on ASR (GPT-4o’s 3% word error rate on ASR Western Europe) [54], we expect that counting the number of sounds is not difficult for MLLMs. In this task, we give an audio clip that contains several sounds ranging from 3 to 8 and ask MLLMs for the number of sounds. In an audio clip, the sounds are monotonous and clearly separated by a silent clip. We formulate these queries as two-choice questions. The MLLMs only need to predict the correct option. The question number for this task, as well as for all remaining tasks in DeafTest, is set to 100.

###### 2. Discriminate the Louder Sound. In this task, we

test the basic ability of MLLMs to distinguish between the loudness of sounds. The goal of MLLMs is to discriminate which sound is louder out of two given audio clips. Specifically, the decibel for quieter audio ranges from 30 dB to 60 dB, while the decibel for louder audio ranges from 70 dB to 100 dB. We randomly sample decibels from these two ranges to create two audio clips. In addition, we randomly switch the input order of the two audio clips; that is, for some questions, the quiet audio comes first, and for the rest, the loud audio comes first. Similarly, the question format is also a two-choice question.

###### 3. Discriminate the Higher Pitch. This task is simi-

lar to task 2 in 3.1. We also create two audio clips. The key difference between the two audios is pitch. Pitch is the basic element of sound, which is helpful in discriminating tone, emotion, environment, etc. For the lower pitch audio, we randomly sample its pitch from 100Hz to 500Hz, while

Table 3. Detailed statistics of AV-Odyssey Benchmark.

Statistics Number Total Questions 4555 Total Tasks 26

Domains 10

Questions with Multiple Images, Singe Audio 2610 Questions with Single Image, Multiple Audios 891 Questions with Singe Image, Singe Audio 434 Questions with Singe Video, Singe Audio 220

Questions with Single Video, Multiple Audios 400 Correct Option Distribution (A:B:C:D) 1167:1153:1119:1116

Average Audio Time 16.32 seconds Average Image Resolution 1267.72 × 891.40 Average Video Resolution 1678.69 × 948.56 Average Video Time 15.58 seconds

we randomly sample pitch from 1000Hz to 2000Hz for the higher pitch audio. We manually check these sounds to ensure that humans can easily discriminate between different pitches. In task 2, we randomly switch the input order of the two audio clips.

4. Recognize the Duration of Sound. We also test MLLMs with the duration of sound. In this task, we simplify the question by giving two audio clips of different durations. We sample the duration from 1s to 3s for the short audio, while we sample from 4s to 6s for the long audio. Similar to task 2, we provide the MLLMs with two audio clips, asking them to identify the longer one.

The results on DeafTest are shown in Table 1. Among the four tasks, sound counting and duration separation seem to be simpler than the other two for MLLMs, since Gemini 1.5 Pro achieves more than 80% accuracy on the two tasks. Nonetheless, all the results are far behind the expected 100%. Especially on loudness comparison and pitch comparison tasks, none of these MLLMs achieve a score over 65%. All these results suggest that MLLMs fall short in basic listening instinct, which might hinder further audio-visual information integration for solving sophisticated audio-visual comprehension tasks. To verify this, we further introduce a holistic AV-Odyssey benchmark to comprehensively evaluate the audio-visual performance of MLLMs as depicted in the following.

###### 3.2. Overview of AV-Odyssey Bench

Our AV-Odyssey Bench is a meticulously curated benchmark designed to comprehensively assess the audio-visual capabilities of MLLMs. To ensure a robust and unbiased assessment, all questions in AV-Odyssey are structured as multiple-choice, with four options per question, and options can be presented in various formats, including text, images, or audio clips. To mitigate format-specific biases, we have curated five distinct multi-choice question types. Additionally, all inputs, including text, image/video, and audio clips, are fed into MLLMs in an interleaved manner.

Audio 3D Angle Estimation

Material Recognition

- Figure 3. Overview of 26 evaluation tasks of AV-Odyssey Benchmark. We mainly categorize these tasks with the sound attributed into 7 classes.

We compare our AV-Odyssey benchmark with previ-

- ous MLLM benchmarks and datasets in Table 2. It can be found that previous works suffer from certain limitations, such as restricted audio attributes, which fail to capture the full spectrum of auditory complexity; narrow domain focus, limiting the generalizability of findings; and the absence of interleaved settings, crucial for assessing realworld multimodal integration capabilities. For instance, Music-AVQA [33] limits audio and visual data to the music domain, while OmniBench [37] lacks multiple audio attributes, making it difficult to comprehensively assess the capabilities of MLLMs in audio-visual tasks. In contrast, our AV-Odyssey encompasses 26 tasks across 10 diverse domains and includes 7 audio attributes, with interleaved and customized questions. The detailed statistics are shown in Table 3. This design enables an exhaustive evaluation of MLLMs, providing a nuanced and thorough assessment of their performance in complex, real-world audio-visual scenarios.

Next, we will briefly introduce the task categories that span a broad spectrum of audio attributes, including Timbre, Tone, Melody, Spatial characteristics, Temporal dynamics, and Hallucination detection. The detailed task distribution and task examples are shown in Figure 3 and Figure 4, respectively.

Timbre Tasks. In order to test the concept of matching across vision and audio modalities, MLLMs are required to match audio-visual pairs (e.g., lion’s roar sound with lion images) in timbre tasks. In addition, we have designed ad-

vanced tasks that demand internal expert-level knowledge learned from the large-scale pretraining data to solve, such as singer recognition and bird species identification.

Tone Tasks. These tasks target evaluating MLLMs with speech sentiment analysis and meme understanding. For example, meme understanding requires MLLMs to infer humorous reasons simultaneously from the voice tone and visual context.

Melody Tasks. For evaluating melody understanding abilities, we propose melody tasks. For example, the dance and music matching task requires the MLLM to understand the melody of the music and identify the one that aligns with the dance in a video.

Space Tasks. To test the spatial inference ability with audio and visual information, space tasks require MLLMs to infer the distance of a certain object producing a sound or to determine the 3D angle.

Time Tasks. These tasks test the cross-modal matching and temporal correlation abilities at the same time. For example, audio time estimation requires MLLMs to determine the start and end time of an action.

Hallucination Tasks. Inspired by POPE [35] that indicates severe object hallucination existing in vision language models, we designed this task to assess the hallucination issue in audio-visual reasoning.

Intricacy Tasks. These tasks challenge MLLMs to perform integrated analysis or reasoning through both visual and audio inputs, leveraging multiple attributes. For example, action prediction requires models to infer actions based on visual elements alongside various audio attributes, such as timbre and timing.

These diverse tasks provide a rigorous and multifaceted assessment of MLLMs’ audio-visual information integration capabilities, systematically probing the depth, nuance, and complexity of cross-modal perception and reasoning.

###### 3.3. Data Curation Process

Data Collection. AV-Odyssey Bench is an audio-visual benchmark to evaluate whether MLLMs truly have audiovisual reasoning capability. Since the audio is the newly added modality by these omni-modal models and there is already an array of visual benchmarks, we put our attention on the attributes of sound in the benchmark construction. We first go through all the attributes of sound, such as timbre, tone, time, space, etc. Then, we start from a specific attribute of sound and span the domains to cover a wide range of application domains, such as music, daily life, and transportation. We primarily use two strategies to construct questions: 1) For most conceptmatching questions (e.g., bird recognition, material recognition), we gather audio clips from public datasets and crawl corresponding visual data from the internet to automatically generate questions and options. Human experts con-

Animal Recognition

Hazard Recognition

###### Meme Understanding

Question: Out of the four audio pieces [audio1] [audio2] [audio3] [audio4], which one is the animal audio that corresponds with the image [img1]? A: the third audio B: the second audio C: the fourth audio D: the first audio

Question: Which potentially hazardous scene shown in [img1], [img2], [img3], or [img4] corresponds best to the sound in [audio1]? A: the fourth image B: the second image C: the third image D: the first image

Question: Based on [audio1] and [video1], what makes this meme funny?

- A: The meme is funny … 'Oh My God' is delivered in a calm, monotone voice, … facial expression remains neutral …
- B: … the exaggerated tone and the person's wide-eyed … and the way the phrase is said adds a musical quality that …

- C: The humor arises … is yelled repeatedly in an intense voice, but the facial expression stays completely blank, while …
- D: The meme is funny because … in a cheerful, upbeat tone, … smiling face contrasts humorously with the intense…

[Figure 46]

[Figure 47]

[audio1] audio content: burning sound

[img1]

[Figure 48]

[Figure 49]

[Figure 50]

[audio1] audio content: “oh my god”

[Figure 51]

[audio1] audio content: cow sound

[img1] [img2]

- [audio1]

[Figure 52]

audio content: a sequence of cleaning sounds for a cup

category: time; domain: daily life

Question: Which of the instruments depicted in [img1], [img2], [img3], or [img4] is not part of the sound of [audio1]? A: the third image B: the first image C: the fourth image D: the second image

Hallucination Evaluation

[audio1] audio content: music clip with

[Figure 53]

guitar, piano and drums [img1] [img2]

[img3] [img4]

category: hallucination; domain: music

Question: After [img1], what action is this person taking as heard in [audio1]?

- A: fill plate
- B: get meat mix from pan

- C: put meat mix on dough
- D: move plate

Action Prediction

[audio1]

[Figure 54]

audio content: the sound of objects scraping in the kitchen

[img1]

category: intricacy; domain: daily life

- [audio2] audio content: lion sound

[Figure 55]

- [audio3] audio content: frog sound

[Figure 56]

- [audio4] audio content: donkey sound

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[video1]

[img3] [img4]

[Figure 63]

category: timbre; domain: animals

category: timbre; domain: scenes

category: tone; domain: memes

Music Sentiment Analysis

Dance and Music Matching

Audio Distance Estimation

Question: Which image [img1] [img2] [img3] [img4] do you think best matches the emotion in the music [audio1]? A: the third image B: the second image C: the first image D: the fourth image

Question: Please select the audio [audio1], [audio2], [audio3], or [audio4] that you think best corresponds to the dance in [video1]. A: the second audio B: the first audio C: the third audio D: the fourth audio

Question: Based on [audio1], could you provide the distance of the sound next to the woman with white clothing in [img1]? The distance should be measured egocentrically in centimeters. A: 190 B: 153 C: 182 D: 159

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[audio1] audio content: happy music

[video1]

[Figure 69]

[audio1] spatial audio with 4 channels

[Figure 70]

[Figure 71]

- [audio1] krump dance music

[Figure 72]

- [audio2] breaking dance music

[Figure 73]

- [audio3] waacking dance music

[Figure 74]

- [audio4] LA-style hiphop dance music

[Figure 75]

[img1] [img2]

[img1]

[Figure 76]

[Figure 77]

[img3] [img4]

[Figure 78]

category: melody; domain: music

category: melody; domain: music

category: space; domain: daily life

Audio Time Estimation

Question: Based on [audio1], what are the times for the action outlined in [video1]?

- A: start time: 0.00 s, end time: 3.58 s

- B: start time: 3.45 s, end time: 4.52 s
- C: start time: 4.42 s, end time: 16.79 s
- D: start time: 17.21 s, end time: 18.55 s

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[video1]

[Figure 86]

[Figure 87]

Figure 4. Sampled examples from our AV-Odyssey Benchmark.

duct post-evaluation and filter out low-quality questions. 2) For other questions (e.g., meme understanding, audio 3D angle estimation), we manually collect audio and visual data from the internet or public datasets, relying on human experts to craft the questions and options. The datasets we used are listed as follows: [1–4, 19, 20, 24, 28, 43– 46, 51, 53, 56, 57, 59, 65, 66, 72, 74, 84, 95].

the answer from the internet-scale training dataset [13]. Inspired by this, we first ensure that our text questions’ context is as simple as possible. Then we filter out those questions that have redundant images or audio clips by leveraging VLMs and audio LLMs. Specifically, we test all the curated questions with VLM: InternVL2 [14], Qwen2-VL [75], MiniCPM-V 2.5 [83], BLIP3 [79], and VILA1.5 [38] and audio LLM Qwen-Audio [16], Qwen2Audio [17], SALMONN [69], and Typhoon-Audio [52], and filter out those questions that can be solved by either of

Quality Control. Duplicated text information that describes visual inputs will induce MLLMs to bypass the visual input to directly derive the answer by memorizing

Table 4. Evaluation results of various MLLMs in different parts of AV-Odyssey Bench. The highest performance is highlighted in bold, while the second highest is underlined. T¯ is the averaged accuracy across corresponding dimensions, and RT¯ is the rank based on the the averaged accuracy. “All Avg.” represents the averaged accuracy over all questions in our AV-Odyssey Bench.

Timbre Tone Melody Space Time Hallucination Intricacy All Avg. T¯ RT¯ T¯ RT¯ T¯ RT¯ T¯ RT¯ T¯ RT¯ T¯ RT¯ T¯ RT¯ T¯ RT¯ Random - 25.0 - 25.0 - 25.0 - 25.0 - 25.0 - 25.0 - 25.0 - 25.0 -

LLM Size

Model

Unified-IO-2 L [47] 1B 23.8 16 24.1 11 28.8 6 15.0 18 26.8 9 30.0 5 30.4 11 26.0 16

Unified-IO-2 XL [47] 3B 24.3 12 23.2 13 27.8 7 22.5 14 25.3 16 31.5 2 34.8 4 26.3 12 Unified-IO-2 XXL [47] 7B 26.3 6 22.7 15 26.4 12 32.5 4 26.8 9 24.5 14 33.8 7 27.2 6

OpenSource

OneLLM [23] 7B 25.0 10 25.5 6 21.5 18 37.5 2 29.3 1 25.5 11 38.4 1 27.4 5 PandaGPT [67] 7B 23.5 17 23.2 13 27.6 10 45.0 1 23.8 18 28.0 10 23.9 17 26.7 10 Video-llama [90] 7B 25.5 7 22.3 16 24.4 17 30.0 6 26.2 13 25.0 12 30.7 10 26.1 14

VideoLLaMA2 [15] 7B 24.1 13 25.5 6 26.4 14 30.0 6 27.2 8 33.0 1 34.5 5 26.8 9

AnyGPT [89] 7B 24.6 11 25.0 8 26.4 15 27.5 11 29.2 2 29.0 6 25.7 15 26.1 15 NExT-GPT [77] 7B 23.2 18 20.9 17 27.8 9 30.0 6 28.8 3 28.5 8 23.6 18 25.5 17 VITA [22] 8 × 7B 24.1 14 26.4 5 27.8 7 22.5 14 26.3 12 31.0 4 36.8 2 26.4 11

Gemini 1.5 Flash [70] - 27.2 4 25.0 8 28.8 5 30.0 6 25.3 16 28.5 8 31.2 9 27.8 4 Gemini 1.5 Flash-8B [70] - 25.1 9 24.5 10 28.9 4 27.5 11 27.5 5 29.0 6 30.2 12 26.8 8

ClosedSource

Gemini 1.5 Pro [70] - 30.8 3 31.4 2 31.3 3 37.5 2 27.7 4 20.5 18 33.0 8 30.8 3 Reka Core [71] 67B 26.7 5 27.7 4 26.4 13 22.5 14 26.5 11 24.0 15 34.3 6 26.9 7 Reka Flash [71] 21B 25.5 8 24.1 11 27.2 11 30.0 6 27.5 5 31.5 2 24.1 16 26.3 13 Reka Edge [71] 7B 23.8 15 20.5 18 26.3 16 22.5 14 25.5 14 22.5 17 36.8 3 25.0 18

GPT-4o visual caption [27] - 37.4 2 28.6 3 32.3 2 27.5 11 25.5 14 23.0 16 28.9 13 32.3 2 GPT-4o audio caption [27] - 38.6 1 31.8 1 33.6 1 32.5 4 27.5 5 25.0 12 26.1 14 34.5 1

these models. In experiment, 2.54% questions are filtered

- out because they are solved by all audio LLMs or VLMs

###### 4. Experiment

We test various closed-source and open-source MLLMs that accommodate the inputs of text, image/video, and audio. Experiments are conducted in the zero-shot setting to evaluate the performance of MLLMs without finetuning and fewshot prompting. The text prompts are designed as concise as possible to remove redundant information.

###### 4.1. Models

We evaluate 18 models in total, 8 closed-source models, including Gemini 1.5 Flash, Gemini 1.5 Flash-8B, Gemini 1.5 Pro [70], Reka Core, Reka Flash, Reka Edge [71], GPT-4o [27] and 10 open-source models including Unifed-IO-2 L [47], Unified-IO-2 XL, Unified-IO2 XXL, OneLLM [23], PandaGPT [67], Video-llama [90], VideoLLaMA2 [15], AnyGPT [89], NExT-GPT [77], VITA [22]. We test those open-source models based on their source code and the latest checkpoint and test the closed-source models with available APIs.

Since we currently cannot access the GPT-4o API that supports simultaneous image, video, and audio inputs, we have adopted an alternative approach to evaluating GPT4o models. The GPT-4o series includes two types of APIs: GPT-4o, which processes image and text inputs, and GPT4o-audio, which processes audio and text inputs. Based on these two APIs, we develop two methods to evaluate GPT4o: (1) We use GPT-4o-audio to generate captions for audio

clips, then feed the text, image/video, and audio captions into GPT-4o. We refer to this approach as the GPT-4o audio caption method. (2) Similarly, we use GPT-4o to generate captions for images or videos, then input the text, audio, and visual captions into GPT-4o-audio. We refer to this approach as the GPT-4o visual caption method.

We set a random baseline which is 25% for AV-Odyssey Bench with four-choice questions. When task performance is below the random baseline, it indicates that the model is unable to handle the task effectively. Consequently, if two models both perform worse than this random baseline, the performance gap between them becomes meaningless.

###### 4.2. Main Result Analysis

In this section, we analyze the performance of MLLMs in our AV-Odyssey benchmark, as presented in Table 4. Due to the space limit, detailed results and data distribution are provided in the Appendix. Our key findings are as follows:

Challenging Nature of AV-Odyssey. As presented in Table 4, the average performance of most existing MLLMs is only marginally higher than 25%—comparable to the expected accuracy of random guessing on four-choice questions. Notably, even the top-performing model in our AVOdyssey, GPT-4o audio caption, only achieves 34.5% accuracy. This result underscores the high level of challenge posed by our benchmark, which significantly goes beyond the distribution of training data of current models. By setting rigorous standards, our benchmark serves as a crucial tool for evaluating the robustness and versatility of MLLMs in audio-visual tasks. It highlights the limitations of existing models and provides directions for future improvements.

[Figure 88]

Reasoning

Text Understanding

Reject to Answer

13% 8% 6%

Vision Understanding

10%

63%

Audio Understanding

Figure 5. Distribution of 104 human-annotated errors in the Gemini 1.5 Pro.

Discrepancies Between Closed-Source and OpenSource Models. The gap between open-source and closedsource models is relatively small. For example, the leading open-source models PandaGPT and OneLLM, with accuracies of 27.4% and 27.2% respectively, are only marginally behind GPT-4o audio caption and Gemini 1.5, which achieve accuracies of 34.5% and 30.8%. Besides, it can be found that PandaGPT and OneLLM deliver performance comparable to certain closed-source models, such as Gemini 1.5 Flash 8B and Reka Core. This indicates that our AV-Odyssey is challenging for both open-source and closed-source models.

Comparison Between Audio Captions and Visual Captions. It can be observed that GPT-4o audio caption achieves higher performance than GPT-4o visual caption as shown in Table 4. This demonstrates that audio captions enable GPT-4o to process audio-visual information more effectively than visual captions. This advantage may be due to the greater information loss in visual captions compared to audio captions when handling audio-visual content.

Limitations of Open-Source MLLMs in Connecting Audio and Visual Information. The Audiocaps dataset [29] introduces audio captions, i.e., audio-text pairs (including animal sound audios), which are used to train MLLMs for learning audio recognition abilities by OneLLM [23], Unified-IO-2 [47], VideoLLaMA2 [15], and NExT-GPT [77]. In addition, they adopt the image-text paired data for training vision comprehension ability. The results on our AV-Odyssey Bench showcase that the audiotext-vision training pipeline is insufficient to bridge audio and vision modalities and truly learn the audio-visual information integration capability.

###### 4.3. Error Analysis

In this section, we focus on the errors of Gemini 1.5 Pro to analyze the underlying causes. For each task, we randomly

[Figure 89]

Audio Understanding Error

Question: Can you determine which of the four pictures

[img1] [img2] [img3] [img4] is the animal image that matches the audio [audio1]? A: the fourth image B: the second image

C: the first image D: the third image

[Figure 90]

[audio1] audio content: lion roar sound

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The audio clip is the sound of an elephant trumpeting.

Therefore, the correct answer is **A: the fourth image**.

Error Reason: Gemini 1.5 Pro correctly identified the image content, but

incorrectly identified the audio content. It misidentified

the roar of a lion as elephant trumpeting due to the similar sounds characteristics.

Figure 6. An example of audio understanding error. More examples are provided in the Appendix.

select 4 error instances for human experts to annotate the error reasons, resulting in a total of 104 instances with humanannotated error reasons. The distribution of these errors is illustrated in Figure 5. Detailed analyses for each case are provided in the Appendix.

Perception Understanding Errors (81%). Perception understanding errors, which include audio understanding errors (63%), vision understanding errors (10%), and text understanding errors (8%), form the majority of the errors. Audio Understanding Errors stand out among all the error types with an error rate of 63%. An example is illustrated in Figure 6, where the content of the audio clip is misidentified, leading to an incorrect answer. This result indicates that the major bottleneck of audio-visual information integration is still in the perception ability in audio. This is in line with the hypothesis induced by the DeafTest results that if an MLLM has a shortage in fundamental listening ability, it will struggle in audio-visual information integration.

Reasoning Errors (13%). In these cases, Gemini 1.5 Pro successfully extracts information from both audio and visual inputs but still produced an incorrect answer due to incorrect reasoning.

Other Errors (6%). Other errors, primarily due to rejected answers, arise from various factors. For instance, content may be mistakenly flagged for security reasons, preventing the model from providing an answer.

###### 5. Conclusion

In this work, we introduce AV-Odyssey Bench, a comprehensive audio-visual benchmark designed to evaluate the

capabilities of MLLMs in understanding audio-visual information. Our AV-Odyssey Bench includes 4,555 meticulously crafted multiple-choice problems, each designed to challenge models in integrating information from both visual and audio cues. Through benchmarking a range of closed-source and open-source models, we uncover the current limitations of MLLMs in effectively understanding audio-visual inputs. We hope that AV-Odyssey Bench will serve as a valuable resource for the community, facilitating the advancement of MLLMs and ultimately leading to more powerful and human-like audio-visual understanding.

###### References

- [1] Abdulvahap. Music instrument sounds for classification. https://www.kaggle.com/datasets/ abdulvahap / music - instrunment - sounds for-classification. 6
- [2] Erhan AKBAL, Turker TUNCER, and Sengul. Vehicle interior sound dataset, 2021.
- [3] Andrada. Gtzan dataset - music genre classification. https://www.kaggle.com/datasets/ andradaolteanu / gtzan - dataset - music genre-classification.
- [4] Emrah AYDEM˙IR. Gunshot audio dataset. https: //www.kaggle.com/datasets/emrahaydemr/ gunshot-audio-dataset. 6
- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 3
- [6] Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling opensource language models with longtermism. arXiv preprint arXiv:2401.02954, 2024. 3
- [7] Tom B Brown. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020. 3
- [8] Jie Cao and Jing Xiao. An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In Proceedings of the 29th International Conference on Computational Linguistics, pages 1511–1520, 2022. 1, 3
- [9] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. arXiv preprint arXiv:2105.14517, 2021. 3
- [10] Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. arXiv preprint arXiv:2212.02746, 2022. 1
- [11] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023. 3

- [12] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 3
- [13] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330,

2024. 6

- [14] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 6
- [15] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 3, 7, 8, 2
- [16] Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, and Jingren Zhou. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. arXiv preprint arXiv:2311.07919, 2023. 6
- [17] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024. 6
- [18] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023. 3
- [19] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Scaling egocentric vision: The epic-kitchens dataset. In Proceedings of the European conference on computer vision (ECCV), pages 720–736, 2018. 6
- [20] Micha¨el Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson. Fma: A dataset for music analysis. arXiv preprint arXiv:1612.01840, 2016. 6
- [21] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 1, 3
- [22] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, et al. Vita: Towards open-source interactive omni multimodal llm. arXiv preprint arXiv:2408.05211, 2024. 3, 7, 2
- [23] Jiaming Han, Kaixiong Gong, Yiyuan Zhang, Jiaqi Wang, Kaipeng Zhang, Dahua Lin, Yu Qiao, Peng Gao, and Xiangyu Yue. Onellm: One framework to align all modalities with language. In Proceedings of the IEEE/CVF Conference

- on Computer Vision and Pattern Recognition, pages 26584– 26595, 2024. 1, 3, 7, 8, 2
- [24] Toni Heittola, Annamaria Mesaros, and Tuomas Virtanen. TAU Urban Acoustic Scenes 2019, Development dataset,

2019. 6

- [25] Anwen Hu, Yaya Shi, Haiyang Xu, Jiabo Ye, Qinghao Ye, Ming Yan, Chenliang Li, Qi Qian, Ji Zhang, and Fei Huang. mplug-paperowl: Scientific diagram analysis with the multimodal large language model. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 6929– 6938, 2024. 3
- [26] EH Huizing. The early descriptions of the so-called tuningfork tests of weber, rinne, schwabach, and bing: Iii. the development of the schwabach and bing tests. ORL, 37(2):92– 96, 1975. 2, 4
- [27] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 1, 2, 3, 7
- [28] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950,

2017. 6

- [29] Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. Audiocaps: Generating captions for audios in the wild. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 119–132, 2019. 8
- [30] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 1
- [31] Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790, 2024. 3
- [32] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024. 1, 3
- [33] Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, JiRong Wen, and Di Hu. Learning to answer questions in dynamic audio-visual scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19108–19118, 2022. 2, 3, 5
- [34] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. ICML,

2023. 3

- [35] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023. 5

- [36] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, 2024. 3
- [37] Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Jian Yang, Siwei Wu, et al. Omnibench: Towards the future of universal omnilanguage models. arXiv preprint arXiv:2409.15272, 2024. 2, 3, 5
- [38] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26689–26699, 2024. 6
- [39] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 3
- [40] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

- 2023. 3

[41] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. arXiv preprint arXiv:2408.15542,

- 2024. 3

- [42] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 1, 3
- [43] Steven R. Livingstone. Ravdess emotional speech audio. https://www.kaggle.com/datasets/ uwrfkaggler / ravdess - emotional - speech audio. 6
- [44] Eu Jin Lok. Crowd sourced emotional multimodal actors dataset. https://www.kaggle.com/datasets/ ejlok1/cremad, .
- [45] Eu Jin Lok. Surrey audio-visual expressed emotion (savee). https://www.kaggle.com/datasets/ejlok1/ surrey - audiovisual - expressed - emotion savee, .
- [46] Eu Jin Lok. Toronto emotional speech set (tess). https:// www.kaggle.com/datasets/ejlok1/torontoemotional-speech-set-tess, . 6
- [47] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. In The Eleventh International Conference on Learning Representations, 2022. 3, 7, 8, 2
- [48] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021. 1
- [49] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 1, 3

- [50] Tengchao Lv, Yupan Huang, Jingye Chen, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, et al. Kosmos-2.5: A multimodal literate model. arXiv preprint arXiv:2309.11419, 2023. 3
- [51] Jeannette Shijie Ma. Eating sound collection. https:// www.kaggle.com/datasets/mashijie/eatingsound-collection. 6
- [52] Potsawee Manakul, Guangzhi Sun, Warit Sirichotedumrong, Kasima Tharnpipitchai, and Kunat Pipatanakul. Enhancing low-resource language and instruction following capabilities of audio language models. arXiv preprint arXiv:2409.10999,

2024. 6

- [53] Microsoft. Birdclef 2020. https://www.imageclef. org/BirdCLEF2020. 6
- [54] OpenAI. Hello gpt-4o. https://openai.com/ index/hello-gpt-4o/. 4
- [55] R OpenAI. Gpt-4 technical report. arxiv 2303.08774. View in Article, 2(5), 2023. 1, 3
- [56] Fabian Ostermann, Igor Vatolkin, and Martin Ebeling. Aam: a dataset of artificial audio multitracks for diverse music information retrieval tasks. EURASIP Journal on Audio, Speech, and Music Processing, 2023(1):13, 2023. 6
- [57] Andrew Owens, Phillip Isola, Josh McDermott, Antonio Torralba, Edward H Adelson, and William T Freeman. Visually indicated sounds. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2405–2413,

2016. 6

- [58] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 3
- [59] Rushi Balaji Putthewad. Sound classification of animal voice. https : / / www . kaggle . com / datasets / rushibalajiputthewad / sound classification-of-animal-voice. 6
- [60] Alec Radford. Improving language understanding by generative pre-training. 2018. 3
- [61] Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind. arXiv preprint arXiv:2407.06581, 2024. 2
- [62] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14313–14323, 2024. 3
- [63] Minjoon Seo, Hannaneh Hajishirzi, Ali Farhadi, Oren Etzioni, and Clint Malcolm. Solving geometry problems: Combining text and diagram interpretation. In Proceedings of the 2015 conference on empirical methods in natural language processing, pages 1466–1476, 2015. 3
- [64] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024. 3

- [65] Kazuki Shimada, Archontis Politis, Parthasaarathy Sudarsanam, Daniel A Krause, Kengo Uchida, Sharath Adavanne, Aapo Hakala, Yuichiro Koyama, Naoya Takahashi, Shusuke Takahashi, et al. Starss23: An audio-visual dataset of spatial recordings of real scenes with spatiotemporal annotations of sound events. Advances in Neural Information Processing Systems, 36, 2024. 6
- [66] Auston Sterling, Justin Wilson, Sam Lowe, and Ming C Lin. Isnn: Impact sound neural network for audio-visual object classification. In Proceedings of the European Conference on Computer Vision (ECCV), pages 555–572, 2018. 6
- [67] Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355, 2023. 3, 7, 2
- [68] Guangzhi Sun, Wenyi Yu, Changli Tang, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Yuxuan Wang, and Chao Zhang. video-salmonn: Speech-enhanced audio-visual large language models. arXiv preprint arXiv:2406.15704, 2024. 2, 3
- [69] Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. Salmonn: Towards generic hearing abilities for large language models. arXiv preprint arXiv:2310.13289, 2023. 6
- [70] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1, 2, 3, 7
- [71] Reka Team, Aitor Ormazabal, Che Zheng, Cyprien de Masson d’Autume, Dani Yogatama, Deyu Fu, Donovan Ong, Eric Chen, Eugenie Lamprecht, Hai Pham, et al. Reka core, flash, and edge: A series of powerful multimodal language models. arXiv preprint arXiv:2404.12387, 2024. 2, 3, 7
- [72] Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chenliang Xu. Audio-visual event localization in unconstrained videos. In Proceedings of the European conference on computer vision (ECCV), pages 247–263, 2018. 6
- [73] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3
- [74] Shuhei Tsuchida, Satoru Fukayama, Masahiro Hamasaki, and Masataka Goto. Aist dance video database: Multi-genre, multi-dancer, and multi-camera database for dance information processing. In Proceedings of the 20th International Society for Music Information Retrieval Conference, ISMIR 2019, Delft, Netherlands, 2019. 6
- [75] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 6
- [76] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 3

- [77] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023. 3, 7, 8, 2
- [78] Jingyi Xu, Hieu Le, Vu Nguyen, Viresh Ranjan, and Dimitris Samaras. Zero-shot object counting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15548–15557, 2023. 1
- [79] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, et al. xgen-mm (blip-3): A family of open large multimodal models. arXiv preprint arXiv:2408.08872, 2024. 6
- [80] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023. 1
- [81] Pinci Yang, Xin Wang, Xuguang Duan, Hong Chen, Runze Hou, Cong Jin, and Wenwu Zhu. Avqa: A dataset for audiovisual question answering on videos. In Proceedings of the 30th ACM international conference on multimedia, pages 3480–3491, 2022. 1, 2, 3
- [82] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421, 9(1):1, 2023. 1
- [83] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 6
- [84] yash bhaskar. Emotify - emotion classificaiton in songs. https : / / www . kaggle . com / datasets / yash9439 / emotify - emotion classificaiton-in-songs. 6
- [85] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023. 3
- [86] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 1
- [87] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 3
- [88] Heeseung Yun, Youngjae Yu, Wonsuk Yang, Kangil Lee, and Gunhee Kim. Pano-avqa: Grounded audio-visual question answering on 360deg videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2031–2041, 2021. 2, 3
- [89] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, et al. Anygpt: Unified multimodal llm with dis-

crete sequence modeling. arXiv preprint arXiv:2402.12226,

2024. 1, 3, 7, 2

- [90] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 7, 2
- [91] Pan Zhang, Xiaoyi Dong Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Hang Yan, et al. Internlmxcomposer: A vision-language large model for advanced text-image comprehension and composition. arXiv preprint arXiv:2309.15112, 2023. 3
- [92] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 3
- [93] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2025. 1, 3
- [94] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding,

2023. 3

- [95] Yu Zhang, Changhao Pan, Wenxiang Guo, Ruiqi Li, Zhiyuan Zhu, Jialei Wang, Wenhao Xu, Jingyu Lu, Zhiqing Hong, Chuxin Wang, et al. Gtsinger: A global multi-technique singing corpus with realistic music scores for all singing tasks. arXiv preprint arXiv:2409.13832, 2024. 6
- [96] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 3

###### AV-Odyssey Bench: Can Your Multimodal LLMs Really Understand Audio-Visual Information?

###### Supplementary Material Contents in Appendix

- A. Data Distribution 1
- B. Breakdown Results 2
- C. Case Study 3

###### A. Data Distribution

In this section, we present the detailed data distribution of our AV-Odyssey Bench in Table 5. Our AV-Odyssey bench consists of 26 tasks covering a wide range of task categories. We will make all the data and evaluation codes public.

Table 5. Detailed task statistics in AV-Odyssey Bench.

Task ID Task Name Task Category Class Number

- 1 Instrument Recognition Timbre 28 200
- 2 Singer Recognition Timbre 20 200
- 3 Gunshot Recognition Timbre 13 200
- 4 Bird Recognition Timbre 39 200
- 5 Animal Recognition Timbre 13 200
- 6 Transportation Recognition Timbre 8 200
- 7 Material Recognition Timbre 10 200
- 8 Scene Recognition Timbre 8 200
- 9 Hazard Recognition Timbre 8 108
- 10 Action Recognition Timbre 20 196
- 11 Eating Sound Recognition Timbre 20 200
- 12 Speech Sentiment Analysis Tone 7 200
- 13 Meme Understanding Tone N/A 20
- 14 Music Sentiment Analysis Melody 7 197
- 15 Music Genre Classification Melody 8 200
- 16 Dance and Music Matching Melody 10 200
- 17 Film and Music Matching Melody 5 200
- 18 Music Score Matching Melody N/A 200
- 19 Audio 3D Angle Estimation Space N/A 20
- 20 Audio Distance Estimation Space N/A 20
- 21 Audio Time Estimation Time N/A 200
- 22 Audio-Visual Synchronization Time N/A 200
- 23 Action Sequencing Time N/A 200
- 24 Hallucination Evaluation Hallucination 19 200
- 25 Action Prediction Intricacy N/A 199
- 26 Action Tracing Intricacy N/A 195

###### B. Breakdown Results

In this section, we provide detailed results of evaluated methods on our proposed AV-Odyssey Bench, as demonstrated in Table 6 and Table 7.

- Table 6. Evaluation results of various MLLMs in ‘Timbre’ part of AV-Odyssey Bench. The best (second best) is in bold (underline). The corresponding brackets for each task indicate the number of associated questions.

Model

LLM Size

Instrument Recognition

Singer Recognition

Gunshot Recognition

Bird Recognition

Animal Recognition

Transportation Recognition

Material Recognition

Scene Recognition

Hazard Recognition

Action Recognition

Eating Sound Recognition

(200) (200) (200) (200) (200) (200) (200) (200) (108) (196) (200)

OpenSource

Unified-IO-2 L [47] 1B 20.5 22.5 25.5 18.5 27.0 26.5 23.0 28.0 21.3 20.9 26.5 Unified-IO-2 XL [47] 3B 20.0 23.5 24.0 20.5 27.5 26.0 27.5 30.0 19.4 19.9 26.5 Unified-IO-2 XXL [47] 7B 29.5 24.0 23.5 29.0 23.5 25.5 30.5 26.5 23.1 27.0 25.5

OneLLM [23] 7B 26.0 21.5 27.0 26.0 22.0 20.0 29.5 24.5 26.9 23.0 29.5 PandaGPT [67] 7B 20.0 21.5 23.0 17.5 26.0 26.5 28.0 27.0 23.1 21.4 24.5

Video-llama [90] 7B 22.5 24.5 27.0 26.5 27.0 23.5 28.0 25.0 25.0 26.0 25.5 VideoLLaMA2 [15] 7B 22.5 24.0 27.0 17.0 23.5 27.5 26.5 26.5 19.4 23.0 25.5

AnyGPT [89] 7B 22.5 28.5 28.0 17.5 24.0 25.5 23.0 28.0 25.9 20.4 27.5 NExT-GPT [77] 7B 21.0 23.5 25.5 21.5 25.5 25.5 21.0 24.0 19.4 23.0 24.0

VITA [22] 8 × 7B 22.0 20.5 24.5 21.5 27.5 25.0 23.5 28.5 21.3 19.4 29.5

ClosedSource

Gemini 1.5 Flash [70] - 24.5 24.0 23.5 17.0 32.5 26.0 22.5 29.5 34.3 48.0 21.5 Gemini 1.5 Flash-8B [70] - 16.5 22.5 24.0 19.0 28.0 26.5 27.0 29.0 26.9 32.7 24.5

Gemini 1.5 Pro [70] - 33.0 26.0 29.0 25.0 25.5 26.0 29.5 30.0 38.0 57.7 22.5 Reka Core [71] 67B 32.5 20.0 26.5 25.0 24.0 27.0 30.0 27.0 25.0 34.2 21.5 Reka Flash [71] 21B 20.0 22.5 26.5 26.0 28.5 26.5 26.5 29.0 28.7 22.4 25.0 Reka Edge [71] 7B 21.5 24.0 30.5 20.0 19.5 22.5 20.5 25.5 25.9 23.5 29.0

GPT-4o visual caption [27] - 33.0 30.5 24.0 26.5 43.0 42.0 32.5 39.0 49.1 67.3 30.5 GPT-4o audio caption [27] - 40.0 38.0 27.5 26.5 45.0 42.0 27.0 41.0 42.6 62.2 35.5

- Table 7. Evaluation results of various MLLMs in ‘Time’, ‘Melody’, ‘Space’. ‘Time’, ‘Hallucination’, and ‘Intricacy’ parts of AV-Odyssey Bench. The best (second best) is in bold (underline). The corresponding brackets for each task indicate the number of associated questions.

Tone Melody Space Time Hallucination Intricacy Speech Sentiment Analysis

LLM Size

Meme Understanding

Music Sentiment Analysis

Music Genre Classification

Dance and Music Matching

Film and Music Matching

Music Score Matching

Audio 3D Angle Estimation

Audio Distance Estimation

Audio Time Estimation

Audio-Visual Synchronization

Action Sequencing

Hallucination Evaluation

Action Prediction

Action Tracing

Model

(200) (20) (97) (200) (200) (200) (200) (20) (20) (200) (200) (200) (200) (199) (195)

Unified-IO-2 L [47] 1B 24.5 20.0 27.9 31.0 27.5 32.5 24.5 15.0 15.0 28.0 25.5 27.0 30.0 27.1 33.8

Unified-IO-2 XL [47] 3B 23.0 25.0 26.9 30.5 27.0 31.5 22.5 30.0 15.0 26.5 25.5 24.0 31.5 35.7 33.8 Unified-IO-2 XXL [47] 7B 23.0 20.0 23.9 31.5 27.5 24.5 23.5 50.0 15.0 28.0 25.0 27.5 24.5 33.2 34.4

OpenSource

OneLLM [23] 7B 26.0 20.0 20.8 23.5 26.5 18.5 18.0 45.0 30.0 31.5 29.5 27.0 25.5 41.7 34.9

PandaGPT [67] 7B 23.5 20.0 21.6 28.0 27.0 32.5 26.0 45.0 45.0 18.5 26.0 27.0 28.0 19.6 28.2 Video-llama [90] 7B 23.0 15.0 25.8 24.0 20.0 25.0 28.0 45.0 15.0 28.5 23.5 26.5 25.0 28.6 32.8

VideoLLaMA2 [15] 7B 26.0 20.0 26.8 29.0 25.5 30.5 20.5 45.0 15.0 28.5 26.5 26.5 33.0 28.6 40.5

AnyGPT [89] 7B 25.5 20.0 23.4 29.5 25.5 26.0 26.0 40.0 15.0 30.5 28.0 29.0 29.0 21.1 30.3 NExT-GPT [77] 7B 21.5 15.0 23.7 26.0 28.0 31.0 28.0 45.0 15.0 31.5 24.0 31.0 28.5 20.6 26.7

VITA [22] 8 × 7B 24.5 45.0 26.8 26.0 27.5 33.5 24.5 25.0 20.0 26.5 25.5 27.0 31.0 34.2 39.5

Gemini 1.5 Flash [70] - 23.5 40.0 21.3 31.0 27.5 32.5 28.0 30.0 30.0 27.5 23.5 25.0 28.5 27.6 34.9 Gemini 1.5 Flash-8B [70] - 24.5 25.0 25.9 33.0 27.5 32.0 24.5 40.0 15.0 31.0 25.5 26.0 29.0 25.6 34.9

ClosedSource

Gemini 1.5 Pro [70] - 29.5 50.0 25.4 42.5 28.0 28.5 29.0 35.0 40.0 30.0 24.5 28.5 20.5 32.2 33.8 Reka Core [71] 67B 28.5 20.0 22.8 24.5 27.5 30.0 25.5 25.0 20.0 30.0 25.5 24.0 24.0 33.7 34.9 Reka Flash [71] 21B 24.5 20.0 30.5 29.5 27.5 25.5 24.5 45.0 15.0 30.0 25.5 27.0 31.5 19.1 29.2 Reka Edge [71] 7B 20.5 20.0 24.9 24.5 27.5 30.0 24.0 30.0 15.0 30.0 25.5 21.0 22.5 38.2 35.4

GPT-4o visual caption [27] - 26.0 55.0 24.4 48.0 27.0 34.5 23.5 25.0 30.0 21.5 22.5 32.5 23.0 32.2 25.6 GPT-4o audio caption [27] - 28.0 70.0 24.4 56.5 27.5 32.5 22.5 30.0 35.0 23.5 25.5 33.5 25.0 30.2 22.0

- C. Case Study

List of Case Study Figures

- 7 Timbre, Instrument Recognition: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 8 Timbre, Singer Recognition: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 9 Timbre, Gunshot Recognition: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 10 Timbre, Bird Recognition: Audio Understanding Error, Vision Understanding Error . . . . . . . . . . . . . . . . . 7
- 11 Timbre, Animal Recognition: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 12 Timbre, Transportation Recognition: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 13 Timbre, Material Recognition: Text Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 14 Timbre, Scene Recognition: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 15 Timbre, Hazard Recognition: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 16 Timbre, Action Recognition: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 17 Timbre, Eating Sound Recognition: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 18 Tone, Speech Recognition: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 19 Tone, Meme Recognition: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 20 Melody, Music Sentiment Recognition: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 21 Melody, Music Genre Classification: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 22 Melody, Dance and Music Matching: Reject to Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 23 Melody, Film and Music Matching: Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 24 Melody, Music Score Matching: Audio Understanding Error, Reasoning Error . . . . . . . . . . . . . . . . . . . . 21
- 25 Space, Audio 3D Angle Estimation: Vision Understanding Error, Audio Understanding Error . . . . . . . . . . . . 22
- 26 Space, Audio Distance Estimation: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- 27 Time, Audio Time Estimation: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- 28 Time, Audio-Visual Synchronization: Audio Understanding Error, Vision Understanding Error . . . . . . . . . . . 25
- 29 Time, Action Sequencing: Audio Understanding Error, Reasoning Error . . . . . . . . . . . . . . . . . . . . . . . 26
- 30 Hallucination, Hallucination Evaluation: Audio Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . 27
- 31 Intricacy, Action Prediction: Audio Understanding Error, Reasoning Error . . . . . . . . . . . . . . . . . . . . . . 28
- 32 Intricacy, Action Tracing: Text Understanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

[Figure 95]

###### Timbre: Instrument Recognition

Question: Out of these instruments highlighted in images, which do you think matches audio [audio1] best? [img1] [img2] [img3] [img4].

- A: the fourth image
- B: the first image
- C: the third image
- D: the second image

[Figure 96]

[audio1] audio content: music with the keyboard instrument

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The instrument in the audio is a vibraphone, which corresponds to image D, the second image. The distinct metallic tone and resonance are characteristic of the vibraphone's metal bars and resonators. So the answer is D.

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro accurately identified the image content but made an error in recognizing the audio. It mistakenly identified the keyboard’s audio as the vibraphone’s audio.

###### Correct Answer: C: the third image

Figure 7. A sampled error case in the instrument recognition task.

[Figure 101]

###### Timbre: Singer Recognition

Question: From the timbre in [audio1], which individual in [img1], [img2], [img3], or [img4] is most likely the singer? A: the second image B: the fourth image C: the third image D: the first image

[Figure 102]

[audio1] audio content: an AI covered song with Rihanna’s vocal timbre

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The singer in the audio clip is most likely **B: the fourth image**, Billie Eilish. This is based on the distinctive vocal timbre and phrasing, which are characteristic of Eilish's style.

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro accurately identified the visual content in the image but made an error in audio recognition, incorrectly attributing Rihanna's vocal timbre to Billie Eilish.

###### Correct Answer: D: the first image

Figure 8. A sampled error case in the singer recognition task.

[Figure 107]

###### Timbre: Gunshot Recognition

Question: Which image [img1], [img2], [img3], [img4] do you think best aligns with the gunfire audio [audio1]? A: the fourth image B: the third image C: the second image D: the first image

[Figure 108]

[audio1] audio content: The sound of the AK-12 firing precise single bursts instead of continuous automatic fire

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The audio clip sounds like sustained automatic gunfire, which best aligns with the first image, a machine gun. Therefore, the answer is D. The other images depict weapons that are not typically capable of sustained automatic fire in the same way.

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro accurately identified the visual content in the images but made an error in audio recognition, incorrectly interpreting the sound of the AK-12's controlled burst fire as sustained automatic gunfire, leading to the wrong conclusion that the answer is D, a machine gun.

Correct Answer: A: the fourth image

Figure 9. A sampled error case in the gunshot recognition task.

[Figure 113]

###### Timbre: Bird Recognition

Question: Which of the four audio segments ([audio1] [audio2] [audio3] [audio4]) shows the bird sound related to the image [img1]? A: the second audio B: the third audio C: the fourth audio D: the first audio

[Figure 114]

[audio1] audio content: sound of other bird categories

- [audio2] audio content: sound of other bird categories

[Figure 115]

- [audio3] audio content: sound of other bird categories

[Figure 116]

- [audio4] audio content: sound of common grackle birds

[Figure 117]

[Figure 118]

[img1]

Gemini 1.5 Pro: The bird in the image is a Brewer's Blackbird. The second audio clip contains the Brewer's Blackbird's call. So the answer is A.

Error Category: Audio Understanding Error, Vision Understanding Error Error Reason: Gemini 1.5 Pro made errors in both visual and audio recognition: it incorrectly identified the common grackle bird in the image as a Brewer's Blackbird and subsequently misattributed the sound of the common grackle to that of a Brewer's Blackbird.

Correct Answer: C: the fourth audio

Figure 10. A sampled error case in the bird recognition task.

[Figure 119]

## Timbre: Animal Recognition

Question: From the four audio options [audio1] [audio2] [audio3] [audio4], can you find the animal audio that corresponds to the image [img1]? A: the third audio B: the fourth audio C: the first audio D: the second audio

[Figure 120]

[audio1] audio content: cat meow

- [audio2] audio content: chicken cluck

[Figure 121]

- [audio3] audio content: elephant trumpet

[Figure 122]

- [audio4] audio content: frog croak

[Figure 123]

[Figure 124]

[img1]

Gemini 1.5 Pro: The audio of a cat meowing is the fourth audio clip. So the answer is B.

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro correctly identified the cat in the image but made an error in audio recognition, mistakenly identifying the sound of a frog as a cat's meow.

Correct Answer: C: the first audio

Figure 11. A sampled error case in the animal recognition task.

[Figure 125]

#### Timbre: Transportation Recognition

Question: Select the image ([img1], [img2], [img3], [img4]) that best represents the sound of a vehicle in [audio1]. A: the fourth image B: the second image C: the first image D: the third image

[Figure 126]

[audio1] audio content: the sound of an airplane

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The sound clip is of a motorcycle engine. Therefore, the correct answer is the fourth image, which depicts a Harley-Davidson motorcycle. Final Answer: The final answer is $\\boxed{A}$

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro correctly identified the motorcycle in the image but made an error in audio recognition, mistakenly identifying the sound of an airplane as the sound of a motorcycle.

Correct Answer: B: the second image

Figure 12. A sampled error case in the transportation recognition task.

[Figure 131]

###### Timbre: Material Recognition

Question: Based on [audio1], which of the materials shown in [img1] [img2] [img3] [img4] is most likely to be the source? A: the second image B: the first image C: the third image D: the fourth image

[Figure 132]

[audio1] audio content: the sound of hitting or stepping on fallen leaves

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: None of these images appear to be the source of the first image, which depicts a wooded path in autumn. The other images show water, tile, and a glass. There's no direct connection between any of them and the forest path.

Error Category: Text Understanding Error Error Reason: Gemini 1.5 Pro misunderstood the question text, incorrectly interpreting it as asking which image the first image originates from, rather than identifying which material depicted in the images most likely matches the sound in the audio clip.

###### Correct Answer: B: the first image

Figure 13. A sampled error case in the material recognition task.

[Figure 137]

##### Timbre: Scene Recognition

Question: Choose the picture that you think best fits the audio [audio1] from [img1] [img2] [img3] [img4]. A: the third image B: the fourth image C: the first image D: the second image

[Figure 138]

[audio1] audio content: the sound of traffic street with cars

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The audio sounds like the whooshing of a train passing quickly. Therefore, the image that best fits the audio is the inside of a subway train (C).

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro correctly identified the content of the image but made an error in audio recognition, mistakenly identifying the sound of the traffic street with cars as the sound inside of a subway train.

- Correct Answer: B: the fourth image

Figure 14. A sampled error case in the scene recognition task.

[Figure 143]

###### Timbre: Hazard Recognition

Question: Among [img1], [img2], [img3], and [img4], which potentially dangerous event fits the sound of [audio1] best? A: the third image B: the second image C: the fourth image D: the first image

[Figure 144]

[audio1] audio content: the sound of fire burning

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The sound is a high-pitched whistling or hissing, like steam escaping under pressure. This best fits the boiling water in the fourth image. So the answer is C.

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro correctly identified the content of the image but made an error in audio recognition, mistakenly identifying the sound of fire burning as the sound of water boiling.

###### Correct Answer: A: the third image

Figure 15. A sampled error case in the hazard recognition task.

[Figure 149]

###### Timbre: Action Recognition

Question: Which action in the images [img1], [img2], [img3], or [img4] do you think best corresponds to the sound [audio1]? A: the fourth image B: the second image C: the first image D: the third image

[Figure 150]

[audio1] audio content: the sound of running on a treadmill

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The answer is C. The sound of a bouncing basketball best corresponds to the first image, which depicts two people playing basketball.

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro correctly identified the content of the image but made an error in audio recognition, mistakenly identifying the sound of running on a treadmill as the sound of playing basketball.

- Correct Answer: A: the fourth image

Figure 16. A sampled error case in the action recognition task.

[Figure 155]

###### Timbre: Eating Sound Recognition

Question: From the foods in the images [img1], [img2], [img3], or [img4], which one is most likely to create the sound [audio1] when consumed? A: the fourth image B: the first image C: the second image D: the third image

[Figure 156]

[audio1] audio content: the sound of eating juicy grapes

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The answer is D: the third image (French fries). The sound "crunch" is associated with crispy or brittle foods. French fries, when properly cooked, have a crisp exterior that makes a crunching sound when bitten into. The other foods are less likely to produce this specific sound. Grapes are juicy and soft, chocolate is typically smooth or melts in the mouth, and noodles, though they can have some texture, don't typically produce a distinct crunch.

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro correctly identified the content of the image but made an error in audio recognition, mistakenly identifying the sound of eating juicy grapes as the sound of eating crispy chips.

###### Correct Answer: B: the first image

Figure 17. A sampled error case in the eating sound recognition task.

[Figure 161]

###### Tone: Speech Sentiment Analysis

Question: Which of the following images [img1] [img2] [img3] [img4] best represents the emotion conveyed in this audio clip [audio1]? A: the fourth image B: the second image C: the third image D: the first image

[Figure 162]

[audio1] audio content: an angry voice

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The audio clip expresses disgust. The woman in the first image best depicts this emotion. So the answer is D.

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro correctly identified the content of the image but made an error in audio recognition, mistakenly identifying an angry voice as a disgusted voice.

###### Correct Answer: C: the third image

Figure 18. A sampled error case in the speech sentiment analysis task.

[Figure 167]

###### Tone: Meme Understanding

Question: According to [audio1] and [video1], what is funny about this meme?

- A: The humor comes from the cat's overly excited face, which contradicts the serious tone of the sound, creating a funny visual and audio mismatch.
- B: The meme is funny because the cat's expression is overly dramatic, and the sound is yelled, adding intensity to simple or mundane events.
- C: The humor comes from the perfect match between the cat's blank, emotionless stare and the disinterested sound, both of which represent a complete lack of energy or surprise, making even absurd situations seem trivial.
- D: It's funny because the cat's face shows anger, but the sound is very calm, making the reaction seem less serious than expected in a high-stress scenario.

[Figure 168]

[audio1] audio content: “bruh”

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[video1]

Gemini 1.5 Pro: Reject to Answer. Error Category: Reject to Answer Error Reason: Content may be mistakenly flagged for security reasons by Gemini 1.5 Pro, preventing the model from providing an answer.

###### Correct Answer: C

Figure 19. A sampled error case in the meme understanding task.

[Figure 175]

### Melody: Music Sentiment Analysis

Question: Which image [img1] [img2] [img3] [img4] best represents the emotion conveyed by the music [audio1]? A: the fourth image B: the second image C: the first image D: the third image

[Figure 176]

[audio1] audio content: upbeat and cheerful music

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The music seems to evoke sadness and melancholy. Therefore, the best choice is A, the fourth image, which depicts crying emoji faces.

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro correctly identified the content of the image but made an error in audio recognition, mistakenly identifying joyful music as sad music.

###### Correct Answer: C: the first image

Figure 20. A sampled error case in the music sentiment analysis task.

[Figure 181]

###### Melody: Music Genre Classification

Question: Considering the audio [audio1] which image [img1] [img2] [img3] [img4] reflects the depicted music style? A: the fourth image B: the second image C: the first image D: the third image

[Figure 182]

[audio1] audio content: classical music

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The music has a distinct country twang, featuring pedal steel guitar and a walking bass line characteristic of country music. The first image, showing country music artists, best reflects this style. So the answer is C.

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro correctly identified the content of the image but made an error in audio recognition, mistakenly identifying classical music as country music.

Correct Answer: B: the second image

Figure 21. A sampled error case in the music genre classification task.

[Figure 187]

###### Melody: Dance and Music Matching

Question: Which of the audio files [audio1], [audio2], [audio3], or [audio4] do you think best fits the dance performed in [video1]? A: the second audio B: the third audio C: the first audio D: the fourth audio

- [audio1] audio content: dance music of other genres

[Figure 188]

- [audio2] audio content: the dance music corresponding to [video1]

[Figure 189]

[Figure 190]

[audio3] audio content: dance music of other genres

[Figure 191]

[audio4] audio content: dance music of other genres

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[video1]

Gemini 1.5 Pro: Reject to Answer. Error Category: Reject to Answer Error Reason: Content may be mistakenly flagged for security reasons by Gemini 1.5 Pro, preventing the model from providing an answer.

- Correct Answer: A: the second audio

Figure 22. A sampled error case in the dance and music matching task.

[Figure 196]

###### Melody: Film and Music Matching

Question: In your view, which movie scene in the images [img1] [img2] [img3] [img4] do you feel is most suitable for this music [audio1]? A: the third image B: the second image C: the first image D: the fourth image

[Figure 197]

[audio1] audio content: fast-paced, cheerful music accompanied by comical screams

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: I'd say A, the third image, is the most suitable. The fast-paced, somewhat chaotic energy of the music fits an action scene like the one depicted, with the characters on a motorcycle. The other images suggest romance, thriller, or comedy, none of which are as good a match for this particular piece of music.

Error Category: Reasoning Error Error Reason: Gemini 1.5 Pro successfully understood both the audio and visual inputs but made a reasoning error when interpreting the data. It incorrectly associated the fast-paced nature of the music with the action genre, assuming that the combination of high energy and disorder aligns with action films. However, the audio, characterized by fast-paced music accompanied by comical screams, is more indicative of a comedy film rather than an action movie. This mistake highlights a gap in the model's contextual reasoning, as it overlooked the playful and humorous tone of the audio, which is a key attribute of comedy rather than action.

Correct Answer: D: the fourth image

Figure 23. A sampled error case in the film and music matching task.

[Figure 202]

###### Melody: Music Score Matching

Question: Focusing on the rhythmic phrasing and articulation, which musical score in these images [img1] [img2] [img3] [img4] most closely matches this audio file [audio1]? A: the third image B: the second image C: the fourth image D: the first image

[Figure 203]

[audio1] audio content: slow-paced music with a long, sustained vocal at the end

[img1] [img2] [img3] [img4]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

Gemini 1.5 Pro: The answer is C: the fourth image. The audio clip has a moderately paced, syncopated rhythm with a swing feel. The fourth image features eighth and sixteenth notes grouped in a way that creates this rhythmic character, along with staccato markings that suggest a short, detached articulation similar to what's heard in the audio. The other images either have a slower tempo, longer sustained notes, or different rhythmic groupings that don't match the audio.

Error Category: Audio Understanding Error, Reasoning Error Error Reason: First, an audio understanding error occurred, as it failed to accurately interpret the audio input, which features slow-paced music with a long, sustained vocal at the end. Instead, the model incorrectly identified the audio as having a moderately paced, syncopated rhythm with a swing feel, entirely mischaracterizing its tempo and articulation. Second, the model incorrectly concluded that the fourth image, with its eighth and sixteenth notes and staccato markings, matches the rhythmic character. However, staccato markings, indicating short, detached articulation, conflict with the described swing feel, which requires smoother, connected phrasing. This reliance on rhythm alone is limiting, as pitch is equally important in musical notation. The model needs to capture the pitch in the audio and align it with the notes in the images to make accurate judgments.

- Correct Answer: B: the second image

Figure 24. A sampled error case in the music score matching task.

[Figure 208]

###### Space: Audio 3D Angle Estimation

Question: Based on [audio1], could you provide the azimuth and elevation angles of the sound located in front of the man dressed in white and black, as illustrated in [img1]? The angles are measured egocentrically, with the azimuth ranging from -180° to 180° (left to right) and the elevation ranging from -90° to 90° (bottom to top).

- A: azimuth: 0°, elevation: -1°
- B: azimuth: 38°, elevation: 9°
- C: azimuth: 51°, elevation: -6°
- D: azimuth: 54°, elevation: 10°

[Figure 209]

[audio1] audio content: 4-channel spatial audio featuring men's voice and the sound of water flowing from a tap

[Figure 210]

[img1]

Gemini 1.5 Pro: The man in the maroon shirt is the one dressed in white and black. The microphone pointed at him appears to be at approximately 0° azimuth (directly in front) and about -1° elevation (slightly below the horizontal). So the answer is A.

Error Category: Vision Understanding Error, Audio Understanding Error Error Reason: First, the statement, "The man in the maroon shirt is the one dressed in white and black," is incorrect. The individual dressed in white and black is not the same as the man in the maroon shirt but is located elsewhere in the scene. This indicates a failure to accurately identify and associate visual details, leading to a misunderstanding of the relationship between these individuals. Second, It failed to identify the sound of water flowing from the tap, which is visually obscured behind the individual dressed in white and black. This highlights a critical limitation in linking audio cues to visually occluded sources. Additionally, the model was unable to accurately determine the 3D spatial orientation of the tap's water flow.

- Correct Answer: C: azimuth: 51°, elevation: -6°

Figure 25. A sampled error case in the audio 3D angle estimation task.

[Figure 211]

###### Space: Audio Distance Estimation

Question: Using [audio1] as a reference, could you determine the distance of the sound coming from below the man in [img1]? The distance should be measured egocentrically in centimeters.

- A: 143
- B: 137
- C: 115
- D: 109

[Figure 212]

[audio1] audio content: 4-channel spatial audio featuring people’s talking, as well as the sound of a vacuum cleaner

[Figure 213]

- [img1]

Gemini 1.5 Pro: The sound source is the vacuum cleaner being operated by the person in the yellow hat. Considering the camera's position (likely near the head of the person sitting at the table), the vacuum is roughly 115 centimeters away. So the answer is **C: 115**.

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro correctly recognized the visual content in the scene but failed to effectively utilize the spatial audio data to determine the precise distance of the vacuum cleaner. While the audio content featured 4-channel spatial information that could have been used to calculate the vacuum cleaner's exact position relative to the camera, the model instead relied exclusively on visual estimations of the distance. This reliance on visual cues alone ignored the valuable depth and directional data provided by the audio, resulting in an inaccurate calculation of 115 centimeters. The error underscores a critical gap in multi-modal reasoning, where the model did not effectively integrate spatial audio with visual context. This highlights the need for enhanced alignment between auditory and visual processing, especially in scenarios where audio provides essential spatial information that visual inputs alone cannot resolve accurately.

###### Correct Answer: B: 137

Figure 26. A sampled error case in the audio distance estimation task.

[Figure 214]

###### Time: Audio Time Estimation

Question: From [audio1], what timestamps correspond to the action portrayed in [video1]?

- A: start time: 0.00 s, end time: 1.35 s
- B: start time: 1.31 s, end time: 2.66 s
- C: start time: 2.45 s, end time: 3.15 s
- D: start time: 4.30 s, end time: 7.16 s

[Figure 215]

[audio1] audio content: the sound of a series of actions

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[video1]

Gemini 1.5 Pro: The action portrayed, putting utensils into a drawer, primarily occurs from [00:00:04.30] to [00:00:07.16]. Therefore, the answer is **D**.

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro failed to accurately identify the timing of the action where a knife is placed into a drawer. Specifically, it misinterpreted the action as occurring in the time range start time: 4.30 s to end time: 7.16 s, instead of the correct range start time: 0.00 s to end time: 1.35 s.

- Correct Answer: A: start time: 0.00 s, end time: 1.35 s

Figure 27. A sampled error case in the audio time estimation task.

[Figure 220]

###### Time: Audio-Visual Synchronization

Question: Can you identify which of [audio1], [audio2], [audio3], or [audio4] best synchronizes with [video1]? A: the first audio B: the third audio C: the fourth audio D: the second audio

[Figure 221]

[audio1] audio content: audio with random offsets, speed-ups, slow-downs [audio2] audio content: audio with random offsets, speed-ups, slow-downs

[Figure 222]

[Figure 223]

[audio3] audio content: the original audio corresponding to [video1]

[Figure 224]

[audio4] audio content: audio with random offsets, speed-ups, slow-downs

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[video1]

Gemini 1.5 Pro: The first audio best synchronizes with the video. The correct answer is A. Error Category: Audio Understanding Error, Vision Understanding Error Error Reason: Gemini 1.5 Pro failed to understand the temporal synchronization between visual and audio elements, leading to errors in aligning events across modalities.

- Correct Answer: B: the third audio

Figure 28. A sampled error case in the audio-visual synchronization task.

[Figure 229]

#### Time: Action Sequencing

Question: Based on [audio1], which order most likely represents the actions in [img1],

- [img2], [img3], and [img4]? A: the first image --> the second image --> the third image --> the fourth image B: the third image --> the fourth image --> the second image --> the first image C: the first image --> the fourth image --> the third image --> the second image D: the third image --> the fourth image --> the first image --> the second image

[Figure 230]

[audio1] audio content: the sound of a series of actions

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The correct answer is A. Here's why: **Image 1:** Shows a potato being washed under the tap. This is likely the first step, as raw food preparation usually precedes dishwashing. **Image 2:** Shows a sponge and an empty sink with running water, implying the potato has been washed and now the dishes are being cleaned.* **Image 3:** Shows a plate being washed. This follows the cleaning of the potato and the preparation of the sink with soapy water. **Image 4:** Shows the plate being placed in a drying rack with other clean dishes, indicating the dishwashing process is nearing completion. The other options don't make logical sense in terms of the typical sequence of washing a potato and then doing dishes.

Error Category: Audio Understanding Error, Reasoning Error Error Reason: Gemini 1.5 Pro failed to accurately identify the sequence of actions based on the audio cues. The model misinterpreted the order in which sounds corresponding to specific actions occurred, leading to an incorrect understanding of the temporal flow of events. Furthermore, it compounded this error with a reasoning mistake by incorrectly inferring the logical order of the actions.

Correct Answer: D: the third image --> the fourth image --> the first image --> the second image

Figure 29. A sampled error case in the action sequencing task.

[Figure 235]

###### Hallucination: Hallucination Evaluation

Question: In the sound of [audio1], which instrument among [img1], [img2], [img3], and [img4] is included? A: the second image B: the third image C: the fourth image D: the first image

[Figure 236]

[audio1] audio content: a music clip including drums but excluding sitar, ukulele, and Fujara

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[img1] [img2] [img3] [img4]

Gemini 1.5 Pro: The instrument being played in the audio clip is a sitar, which corresponds to the first image. Therefore, the answer is D.

Error Category: Audio Understanding Error Error Reason: Gemini 1.5 Pro incorrectly identified the presence of a sitar in the music, despite the audio containing no such instrument. This hallucination demonstrates a failure in accurately analyzing the audio content, leading the model to perceive elements that are not present in the actual input.

- Correct Answer: A: the second image

Figure 30. A sampled error case in the hallucination evaluation task.

[Figure 241]

###### Intricacy: Action Prediction

Question: With [img1] preceding [audio1], what does the audio suggest about this person's actions? A: turn box B: wrap up coffee C: smell box D: put coffee in cupboard

[Figure 242]

[audio1] audio content: the sound of actions

[Figure 243]

[img1]

Gemini 1.5 Pro: The audio suggests the person is opening a box or bag, most likely of coffee. Therefore the answer is **B: wrap up coffee**.

Error Category: Audio Understanding Error, Reasoning Error Error Reason: Gemini 1.5 Pro failed to correctly understand the audio, leading to a misinterpretation of the events. Additionally, the model was unable to understand the order of actions, resulting in an incorrect ordering of the events. This highlights deficiencies in both audio recognition and the ability to reason about temporal relationships between actions.

Correct Answer: D: put coffee in cupboard

Figure 31. A sampled error case in the action prediction task.

[Figure 244]

###### Intricacy: Action Tracing

Question: With [img1] and the preceding [audio1] in mind, what actions were being performed by this person during the audio? A: bend machine B: rinse chopping board C: carry machine D: wipe chopping board with rag

[Figure 245]

[audio1] audio content: the sound of actions

[Figure 246]

[img1]

Gemini 1.5 Pro: The person is wiping the chopping board with a rag (D).

Error Category: Text Understanding Error Error Reason: Gemini 1.5 Pro misinterpreted the question, failing to recognize the temporal focus of the query. The model incorrectly assumed the question was asking about the person's current actions, rather than the actions at the time the audio occurred. This misunderstanding of both the question's intent and its temporal context led to an incorrect response.

- Correct Answer: B: rinse chopping board

Figure 32. A sampled error case in the action tracing task.

