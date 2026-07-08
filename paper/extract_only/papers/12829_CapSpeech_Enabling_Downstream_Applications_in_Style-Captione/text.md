## CapSpeech: Enabling Downstream Applications in Style-Captioned Text-to-Speech

Helin Wang†, Jiarui Hai†, Dading Chong, Karan Thakkar, Tiantian Feng, Dongchao Yang, Junhyeok Lee, Thomas Thebaud, Laureano Moro-Velázquez, Jesús Villalba, Zengyi Qin, Shrikanth Narayanan, Fellow, IEEE,

Mounya Elhiali, Senior Member, IEEE and Najim Dehak, Senior Member, IEEE

# arXiv:2506.02863v2[eess.AS]26Sep2025

Abstract—Recent advancements in generative artificial intelligence have significantly transformed the field of style-captioned text-to-speech synthesis (CapTTS). However, adapting CapTTS to real-world applications remains challenging due to the lack of standardized, comprehensive datasets and limited research on downstream tasks built upon CapTTS. To address these gaps, we introduce CapSpeech, a new benchmark designed for a series of CapTTS-related tasks, including style-captioned text-tospeech synthesis with sound events (CapTTS-SE), accent-captioned TTS (AccCapTTS), emotion-captioned TTS (EmoCapTTS), and text-to-speech synthesis for chat agent (AgentTTS). CapSpeech comprises over 10 million machine-annotated audio-caption pairs and nearly 0.36 million human-annotated audio-caption pairs. In addition, we introduce two new datasets collected and recorded by a professional voice actor and experienced audio engineers, specifically for the AgentTTS and CapTTS-SE tasks. Alongside the datasets, we conduct comprehensive experiments using both autoregressive and non-autoregressive models on CapSpeech. Our results demonstrate high-fidelity and highly intelligible speech synthesis across a diverse range of speaking styles. To the best of our knowledge, CapSpeech is the largest available dataset offering comprehensive annotations for CapTTS-related tasks. The experiments and findings further provide valuable insights into the challenges of developing CapTTS systems.

Index Terms—speaker style, text-to-speech synthesis, audio caption, emotional speech, chat agent.

I. INTRODUCTION

In recent years, large-scale text-to-speech (TTS) synthesis has witnessed remarkable progress, exemplified by models such as VALL-E 2 [1], NaturalSpeech 3 [2], LLasa [3], SimpleSpeech 2 [4], CosyVoice 2 [5], and MaskGCT [6]. Most of these efforts have centered on modeling audio-based speaker characteristics, such as zero-shot TTS with audio prompts [7]–[9], category-based traits [10]–[13], or embeddingbased speaker representations [14], [15]. However, the detailed understanding of speaking style from audio, particularly its subtle nuances, has received limited attention.

The style of speech encompasses both intrinsic traits tied to a speaker’s identity (e.g., age, gender, timbre) and expressive style traits specific to individual utterances (e.g., emotion, speaking rate). Recent studies have introduced the use of

Helin Wang and Jiarui Hai are co-first authors. Helin Wang, Jiarui Hai, Karan Thakkar, Junhyeok Lee, Thomas Thebaud, Laureano Moro-Velázquez, Jesús Villalba, Mounya Elhiali and Najim Dehak are with the Johns Hopkins University (email: {hwang258, jhai2}@jhu.edu).

Dading Chong is with the Peking University. Dongchao Yang is with the Chinese University of Hong Kong. Tiantian Feng and Shrikanth Narayanan are with the University of Southern California. Zengyi Qin is with the Massachusetts Institute of Technology.

natural language captions to describe these stylistic elements—a paradigm referred to as prompt TTS, expressive TTS, or stylecaptioned TTS [16]–[22]. In this work, we adopt the term style-captioned TTS and abbreviate it as CapTTS.

Developing a CapTTS system necessitates a large corpus of audio-caption pairs accompanied by transcriptions, the annotation of which is labor-intensive and costly. Parler-TTS [23], [24] addressed this challenge by automatically annotating basic speech style attributes such as pitch and speed using signal processing tools. ParaSpeechCaps [25] collected speaker-level intrinsic tags and utterance-level situational tags, LibriTTS-P [26] offered speaker identity-based captions, and EmoVoiceDB [27] provided emotion-based captions. However, these existing datasets lack a unified and comprehensive framework for style captioning, making cross-domain comparisons difficult. Moreover, there has been limited exploration of downstream applications, such as transferring models to new caption styles or incorporating sound events into the synthesized speech.

To address the challenges outlined above, we present CapSpeech, a novel benchmark featuring standardized and comprehensive datasets for CapTTS and its related downstream tasks. CapSpeech comprises a pretraining stage with largescale captioned speech, as well as five downstream tasks: CapTTS, CapTTS-SE, accent-captioned TTS (AccCapTTS), emotion-captioned TTS (EmoCapTTS), and TTS for chat agent (AgentTTS). The pretraining datasets consist of over 10 million machine-annotated audio-caption pairs, while the downstream datasets contain 358,783 human-annotated audio-caption pairs. These datasets encompass a broad range of intrinsic speaker traits and expressive style traits, curated from a wide array of audio sources, including Emilia [28], GigaSpeech [29], CommonVoice [30], MLS [31], LibriTTS-R [32], VoxCeleb [33], VoxCeleb2 [34], EARS [35], Expresso [20], VCTK [36], VGGSound [37], FSDKaggle2018 [38], and ESC-50 [39]. In addition, we introduce two new datasets: one for AgentTTS, built using professionally recorded voice actor speech, and another for CapTTS-SE, processed by five experienced audio engineers.

In addition, we develop two style-captioned TTS models based on state-of-the-art generative TTS backbones: one autoregressive (AR) and one non-autoregressive (NAR). We evaluate these models on the CapSpeech benchmark across 5 downstream tasks. For a comprehensive evaluation, both objective and subjective metrics are employed to assess speech style consistency, audio quality, and intelligibility. In summary, our contributions are as follows:

- TABLE I: A comparison of English speech style-captioned datasets. I1–I5 denote intrinsic speaker traits: age (I1), gender (I2), timbre (I3), mean pitch (I4), and accent (I5). E1–E4 represent expressive style traits: speaking rate (E1), emotion (E2), expressiveness of tone (E3), and volume (E4).

Intrinsic Traits Expressive Traits Coverage & Access

Dataset

I1 I2 I3 I4 I5 E1 E2 E3 E4 Duration (h) Sound Event Open Source

PromptSpeech [16] ✗ ✓ ✗ ✓ ✗ ✓ ✓ ✗ ✓ 0.3k ✗ ✓ Expresso [20] ✗ ✗ ✗ ✗ ✗ ✗ ✓ ✓ ✗ 47 ✗ ✓ EARS [35] ✓ ✓ ✗ ✓ ✓ ✓ ✓ ✓ ✓ 60 ✗ ✓ TextrolSpeech [40] ✗ ✓ ✗ ✓ ✗ ✓ ✓ ✗ ✓ 0.3k ✗ ✓ SpeechCraft [17] ✓ ✓ ✗ ✓ ✗ ✓ ✓ ✗ ✓ 2.4k ✗ ✓ VccmDataset [41] ✗ ✓ ✗ ✓ ✗ ✓ ✓ ✗ ✓ 0.3k ✗ ✓ LibriTTS-P [26] ✓ ✓ ✓ ✓ ✗ ✓ ✗ ✓ ✓ 0.6k ✗ ✓ DreamVoiceDB [42] ✓ ✓ ✓ ✗ ✗ ✗ ✗ ✓ ✗ 0.3k ✗ ✓ ParlerTTS [23] ✗ ✓ ✗ ✓ ✓ ✓ ✗ ✓ ✗ 45k ✗ ✓ ParaSpeechCaps [25] ✗ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ 2.9k ✗ ✓ NonVerbalSpeech-38K [43] ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ 0.1k ✓ ✓ PromptTTS2 [18] ✗ ✓ ✗ ✓ ✗ ✓ ✗ ✗ ✓ 44k ✗ ✗ VoxInstruct [44] ✓ ✓ ✗ ✓ ✗ ✓ ✓ ✗ ✓ 1.5k ✗ ✗ FleSpeech [45] ✗ ✓ ✗ ✓ ✗ ✓ ✓ ✓ ✓ 0.6k ✗ ✗ Audiobox [46] ✓ ✓ ✗ ✓ ✓ ✓ ✓ ✗ ✗ >0.5k ✓ ✗

CapSpeech ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ 33.6k ✓ ✓

- • We introduce CapSpeech, a large-scale style-captioned benchmark for CapTTS that encompasses diverse speech styles.
- • We propose 5 CapTTS-related downstream tasks, 4 of which are new, aimed at practical CapTTS applications, introduce 9,353,823 new captions for existing speech corpora, and curate 3 new speech datasets specifically designed for the CapTTS-SE and AgentTTS tasks, thereby enhancing the benchmark’s coverage of real-world scenarios.
- • We train and evaluate two CapTTS models, CapSpeechAR and CapSpeech-NAR, which address the lack of benchmarks for CapTTS downstream tasks, demonstrate the effectiveness of the proposed dataset, and highlight key challenges and future research directions.
- • We publicly release our datasets, listening samples, source code, pretrained checkpoints, and evaluation toolkit to support future research1. All resources are released under the CC BY-NC 4.0 license (Creative Commons Attribution-NonCommercial), which permits use for non-commercial research purposes with appropriate attribution.

II. RELATED WORKS A. Speech Style-Captioned Datasets

Speech datasets with natural language style captions offer greater flexibility and controllability compared to those that rely on audio prompts. Early efforts such as NLSpeech [21] recruited human annotators to describe speech emotions, laying the groundwork for later datasets like PromptStyle [47] and MM-TTS [48]. TextrolSpeech [40] aggregates several existing emotion datasets, primarily focusing on emotional states and a few basic style tags. In contrast, Expresso [20] and EARS [35] broaden the scope by covering a wider range of spontaneous expressive styles. Meanwhile, datasets such as LibriTTS-P [26] and DreamVoiceDB [42] focus on intrinsic speaker traits,

1https://github.com/WangHelin1997/CapSpeech

providing captioned annotations for LibriTTS-R [32], while Coco-Nut [49], [50] captures subjective descriptions of voice characteristics beyond traditional acoustic parameters. More recently, ParaSpeechCaps [25] introduces a human-annotated dataset that covers a richer and more diverse set of speaking style attributes. Given the high cost and limited scalability of human annotation, several works have explored automatic or semi-automatic approaches to expand style-captioned speech datasets. PromptSpeech [16] and PromptTTS2 [18] synthesize speech with diverse speaker identities and emotions using commercial TTS systems. Parler-TTS [23], [24] proposes large-scale annotation of basic style tags (e.g., pitch, speed) by leveraging signal processing techniques and rule-based binning. Other efforts, such as FleSpeech [45] and VccmDataset [41], re-caption existing speech datasets like TextrolSpeech using large language models to improve label quality and expressiveness. VoxInstruct [44] and SpeechCraft [17] further extend this direction by introducing more fine-grained emotion tags. In addition, AudioBox [46] integrates multiple strategies, combining large-scale automatically generated basic tags with high-quality human-annotated rich style labels to build a scalable and diverse dataset. It also supports sound events such as laughter and clapping, which are important for expressive speech generation.

Table I summarizes existing English style-captioned datasets. To the best of our knowledge, the proposed CapSpeech is the first large-scale open-source dataset that covers a wide range of both intrinsic speaker traits and expressive speaking styles. Similar to AudioBox [46] and NonVerbalSpeech-38K [43], CapSpeech also supports speech generation with sound events.

B. Speech Style-captioned TTS Models

Most existing methods for speech style-captioned TTS build upon prior audio-prompted TTS frameworks. For example, PromptTTS [16] is developed on top of FastSpeech 2 [51], while PromptTTS2 [18] is based on NaturalSpeech 2 [52]. Similarly, Salle [40] builds on VALL-E [53], and ParlerTTS [23] further introduces cross-attention mechanisms to

Transcription Style Caption Recording

Intrinsic Speaker Traits

Expressive Style Traits

Sound Event & Environment 🎵Original Task & New Data

🏷New Label on Existed Data

🏷CapTTS

###### 🏷AccCapTTS

- Transcription1: Now you can see that this is an odd setting right, odd setup which means that left subarray has one element extra compared to the right subarray.
- Transcription2: From these genres and from these spaces, you know, and the feelings of what these games can bring.

Transcription1: I don't know why you say goodbye.

- Style Caption1: An American teenager with a soft, cute, and gender-neutral voice, speaking slowly.

- Style Caption2: A dark male voice with an Indian accent, well-suited for fast-paced narration.

- Style Caption1: A male speaker with an American accent talks at a measured pace in a clean environment. His voice is medium-pitched and flows smoothly.

- Style Caption2: An elderly woman, with a low-pitched voice, delivers her speech in a slow, yet expressive and animated manner. Her words flow like a captivating story, each sentence filled with emotion and wisdom, resonating deeply with her audience.

Transcription2: A few days after making the video, I went to the inquiry.

EmoCapTTS

- Transcription1: The wind howled through broken windows, its wails sharp as a banshee's cry.
- Transcription2: You're saying this is the first time you've ever seen a penguin here?

Style Caption1: A middle-aged woman speaks in a slow, monotone voice, carrying a chilling sense of dread.

###### 🎵CapTTS-SE

- Transcription1: <telephone> <I> </I> Hello, this is john speaking, could you please help me with the documents?

- Transcription2: She hurried to him immediately <knock> <B> and led him off to look at the picture. </B>

Style Caption2: A middle-aged man, with a very highpitched voice, speaks slowly, his tone rising in shock.

- Style Caption1: A middle-aged man's voice maintains a steady, unvaried tone, reminiscent of a metronome. His slightly low-pitched speech carries a sense of gravitas, as he delivers his words at a moderate speed, demonstrating a methodical and measured approach.

- Style Caption2: A young woman, with a slight air of urgency, speaks in a high-pitched, somewhat monotone tone. Her pace is slightly fast, lending an efficient and focused quality to her delivery.

🎵AgentTTS

Transcription1: As we sit by this fire under the vast, starlit sky, remember, we're here for each other, always.

- Style Caption1: A soft, gentle voice with comforting pauses, offering quiet solace and heartfelt support.

- Style Caption2: Cold tone, delivered at a slow pace, each word dripping with contempt.

Transcription2: Is this really what you consider a treasure, or are you simply trying to swindle me?

Fig. 1: Overview of the proposed CapSpeech benchmark and its applications. Here, blue words denote intrinsic speaker traits, purple words indicate expressive style traits, and red words represent environment description and special sound event tokens used in CapTTS-SE transcriptions.

better integrate caption features. In parallel, diffusion-based approaches such as InstructTTS [21] and AudioBox [46] have also emerged, offering alternative generative paradigms for style-controllable speech synthesis. In this work, we adopt both autoregressive and non-autoregressive models to evaluate performance on our proposed benchmark.

III. CAPSPEECH BENCHMARK

CapSpeech comprises a pretraining (PT) stage that leverages large-scale machine-annotated speech–caption pairs, followed by a Supervised Fine-Tuning (SFT) stage using humanannotated captioned speech. In this section, we introduce the tasks and datasets included in the CapSpeech.

- A. Tasks and Applications

As shown in Fig. 1, the CapSpeech benchmark includes the following five tasks:

CapTTS synthesizes speech from text input conditioned on a natural language style caption that describes desired attributes

of the output—such as speaker traits (e.g., age, gender, accent), expressive styles (e.g., emotion, speaking rate), or situational context (e.g., conversational tone, whispering). This subtask reflects general use cases of caption-based TTS. While it is less focused on specific applications, it provides a versatile benchmark for generating speech under diverse, caption-guided conditions.

CapTTS-SE extends CapTTS by allowing the synthesis of speech that includes non-verbal sound events. The speaking style is specified through a natural language caption, while sound events (e.g., door knocking, applause, dog barking) are explicitly indicated in the transcription. Sound events can either play in the background of the speech or be inserted at specific points within the utterance. This subtask supports applications like audiobooks and livestreaming—any scenario where sound events enhance the experience beyond vanilla speech-only TTS.

AccCapTTS focuses specifically on accent control, in contrast to the broader functionality addressed by CapTTS. Unlike traditional accent-conditioned TTS systems that rely on predefined categories, AccCapTTS offers more user-friendly

and flexible control through free-form natural language prompts. This subtask enables applications such as cross-cultural voice design, personalized speech synthesis, and localized content creation—scenarios where nuanced accent control greatly enhances realism and user engagement.

EmoCapTTS generates speech from text input conditioned on a natural language caption that describes both the speaker’s emotional state and identity. Unlike traditional emotion TTS systems that rely on discrete categories (e.g., happy, sad, angry) [15], [54], EmoCapTTS enables more flexible and expressive emotion control through free-form textual descriptions. This subtask balances multi-speaker control with basic emotion expression, targeting applications like story narration and gaming NPCs where multiple AI speakers are required.

AgentTTS focuses on generating speech for a single, expressive virtual agent. While it also uses captions like EmoCapTTS, it refines broad emotional categories into more fine-grained states and captures nuanced differences between emotions (e.g., fearful vs. panicked), models interactions between emotional states and low-level speaking styles such as pitch and speed (e.g., happy and slow vs. happy and fast), and incorporates expressive non-speech vocalizations (e.g., sighs, laughter, sobs). This task closely reflects real-world scenarios in building customized dialogue agents, customer service bots, AI therapists, and other conversational AI applications.

- B. Datasets

In this section, we introduce the data collection process in CapSpeech2 and summarize the data sources used across different tasks in Table II. The total amount of annotated data is 30.8k hours for age, 33.5k hours for gender, 0.4k hours for timbre, 33.5 k hours for pitch, 2.5 k hours for accent, 33.5 k hours for speaking rate, 2.4 k hours for emotion, 33.5 k hours for expressiveness of tone, and 2.7 k hours for volume.

- TABLE II: CapSpeech data sources used across different tasks. Italics indicate machine-labeled data. Regular text indicates human-labeled data. Blue denotes newly annotated data provided by us, and ˇ “( indicates new audio samples.

Emilia-en, GigaSpeech, MLS-en, CommonVoice, ˇ “( CapSpeech-PT-SEDB

Task Audio Source Pretraining

CapTTS, EmoCapTTS, LibriTTS-R, EARS, Expresso, AccCapTTS VCTK, VoxCeleb, VoxCeleb2 CapTTS-SE ˇ “( CapSpeech-SEDB

### AgentTTS ˇ “( CapSpeech-AgentDB

1) Pretraining Sets: CapSpeech includes two pretraining tasks: CapTTS-PT and CapTTS-SE-PT, which are trained jointly during the pretraining phase.

CapTTS-PT comprises four English speech corpora: the English portion of Emilia [28], the English portion of Multilingual LibriSpeech (MLS) [31], GigaSpeech [29], and CommonVoice [30]. Since the overall audio quality of CommonVoice is poor,

2https://huggingface.co/datasets/OpenSound/CapSpeech

we only include the age groups under 20 and over 65, which are not well covered in other datasets. These datasets cover a wide range of speech sources: LibriVox, YouTube, Podcasts, Audiobooks, and etc. For Emilia, we directly use the style annotations provided in ParaSpeechCaps [25], which include 59 diverse style tags. In total, there are 9,053,734 speech–caption pairs.

We annotate age, gender, pitch, expressiveness of tone, and speaking rate for MLS, GigaSpeech, and CommonVoice, and use an LLM3 to generate natural language captions based on these traits. Althouth these datasets provide transcriptions, there are some audio files with noisy labels or multiple speakers. Thus, we apply three data cleaning steps: (i) samples with durations between 3 and 18 seconds are selected. (ii) we filter out samples with word error rates (WER) greater than 25%, estimated using the OpenAI Whisper toolkit4 [55]; (iii) we remove noisy utterances with estimated signal-to-noise ratios (SNR) below 20 dB, computed using the Squim Objective metric [56]. We use a pre-trained age and gender estimator5 [57] to annotate speaker demographics, if they are not provided by the dataset creator. The age groups are divided into five categories: "child" (1–12 years), "teenager" (13–19 years), "young adult" (20–39 years), "middle-aged adult" (40–64 years), and "elderly" (65 years and older). Following Parler-TTS [23], pitch and expressiveness are measured using speaker-level mean and utterance-level standard deviation of pitch, computed with the PENN library6. The speaker-level mean is used to generate a label for speaker pitch relative to gender, and the standard deviation is used as a proxy for how monotone or animated an individual utterance is. Speaking rate is calculated by dividing the number of phonemes in the transcription by the total duration of the silence-removed utterance.

Next, we convert all the variables described above into natural language sentences. To do so, we first generate keyword representations for each variable. Discrete labels provided by the dataset creators, such as gender, are used directly as keywords. For continuous variables like pitch and speaking rate, we apply binning to map them into discrete categories. After binning all continuous variables, we obtain a complete set of keywords for gender, accent, pitch, expressiveness of tone, and speaking rate.

More specifically, the age category includes "child" (0.15%), "teenager" (0.20%), "young adult" (32.72%), "middle-aged adult" (63.44%), and "elderly" (3.49%); the gender category includes "male" (57.48%) and "female" (42.52%); the pitch category includes "very low-pitch" (3.22%), "low-pitch" (7.19%), "slightly low-pitch" (26.26%), "moderate pitch" (28.05%), "slightly high-pitch" (21.34%), "high-pitch" (12.45%) and "very high-pitch" (1.49%); the expressiveness of tone category includes "very monotone" (10.59%), "monotone" (48.56%), "slightly expressive and animated" (24.58%), "expressive and animated" (8.32%) and "very expressive and animated" (7.96%); the speaking rate category includes "very slowly" (0.01%), "slowly" (1.60%), "slightly slowly" (15.03%), "mod-

- 3https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- 4https://huggingface.co/openai/whisper-large-v3-turbo
- 5https://github.com/audeering/w2v2-age-gender-how-to
- 6https://github.com/interactiveaudiolab/penn

erate speed" (59.81%), "slightly fast"(18.98%), "fast" (4.10%) and "very fast" (0.45%).

We then apply the LLM to generate natural language captions based on these keywords. To guide the model, we construct a prompt and curate a diverse set of manually verified examples. During caption generation, random examples are paired with different audio files to enhance the diversity of expressions. Finally, we apply post-processing to remove excessively long captions. Below is the LLM prompts used:

#### Prompt Design for Pretraining Sets

System: You are a voice style caption generator. User: Please generate a single text description of a speech sample using the provided keywords. Definition of Keywords:

- • Gender (e.g., male, female)
- • Age (e.g., teenager, young adult, etc.)
- • Tone (e.g., very monotone, quite expressive, etc.)
- • Pace (e.g., very slowly, quite fast, etc.)
- • Pitch (e.g., very low pitch, quite high pitch, etc.) Instructions:
- • Use the keywords to create a grammatically correct and easy-to-understand description of the speech sample, varying the sentence structure and phrasing as much as possible across examples.
- • Rearrange the keyword order, split ideas across multiple sentences, or introduce descriptive transitions to make the description fluid and natural.
- • Substitute synonymous terms where appropriate, and rephrase parts of the description to add variety and keep it engaging.
- • Explore different words to convey the tone, pace, and other characteristics.
- • You can drop some of the keywords for diversity.
- • Return only the generated description. Please review the following examples: [randomly selected human-written style captions.] Keywords: Gender: [e.g., female]; Age: [e.g., teenager]; Tone: [e.g., bright, expressive]; Pace: [e.g., quick]; Pitch: [e.g., high]. LLM Assistant: [e.g., A teenage girl speaks with a high-pitched, bright voice that’s lively and expressive, delivering her words at a quick pace.]

To assess potential biases introduced by machine annotation, we conducted a systematic human evaluation on the CapTTS-PT dataset. We randomly sampled 500 examples, each annotated independently by two experienced audio engineers. For each example, individual tags (e.g., age, gender, speaking rate) were evaluated with a binary correctness label (0 or 1) indicating whether the machine-predicted tag accurately reflected the speech. Also, a caption-level quality score was assigned on a 1–5 Likert scale to assess the overall coherence, coverage, and naturalness of the generated captions. Table III shows the detailed results, reporting the mean values along with their 95% confidence intervals. We can see that the average pertag correctness is 95.5%, demonstrating reliable labeling. The

average caption quality score is 4.2 out of 5, indicating strong overall fidelity.

TABLE III: Human evaluation on the CapTTS-PT dataset.

Attribute Score

Age 0.979 ± 0.019 Gender 1.000 ± 0.000

Pitch 0.944 ± 0.030 Speaking rate 0.991 ± 0.012

Expressive of tone 0.863 ± 0.004 Caption 4.220 ± 0.071

For CapTTS-SE-PT, we simulated data using the LibriTTS-R speech corpus and three sound event corpora: VGGSound [37], FSDKaggle2018 [38], and ESC-50 [39]. We curate a collection of 394 distinct sound events drawn from the categories of these datasets. We refer to this dataset as CapSpeech-PTSEDB. Sound events are introduced into speech using two modes: (i) insertion, where the event is inserted at a specific location; and (ii) background, where the event is layered beneath the speech. For the background mode, we use noisy audio from VGGSound, while the insertion mode relies on cleaner sources from FSDKaggle2018 and ESC-50. For each LibriTTS-R sample, a sound event and mode are randomly selected. Forced alignment is performed using the Montreal Forced Aligner7 to determine insertion points—either at the beginning, middle, or end of the utterance. To ensure the simulated audio sounds natural and smooth, we select insertion points between words with a minimum interval of 0.3 seconds, ensuring a pause that does not disrupt word continuity. The speech and sound event are then mixed at a random SNR ranging from –3 dB to 6 dB. To scale up the data, each speech sample is simulated five times with different configurations.

TABLE IV: Sound event examples in CapSpeech.

Applause Bark Burping or eructation Bus Chime Writing Drawer open or close Cough Fart Finger snapping Fireworks Crow Keys jangling Knock Laughter Meow Microwave oven Clock tick Shatter Hand saw Squeak Tearing Telephone Computer keyboard Dog Rooster Rain Crying baby Door knock Scissors Helicopter Sea waves Sneezing Mouse click Chainsaw Pig Crackling fire Clapping Keyboard typing Siren Cow Crickets Breathing Door wood creaks Car horn Frog Chirping birds Coughing Can opening Engine Cat Water drops Footsteps Washing machine Hand saw Train Hen Wind Laughing Vacuum cleaner Church bells Insects (flying) Pouring water Brushing teeth Clock alarm Airplane Sheep Toilet flush Snoring Gunshot or gunfire Thunderstorm Glass breaking Drinking, sipping

In total, the dataset contains 10,054,530 speech–caption pairs, with 7,894 pairs reserved for validation and 7,959 pairs held out for testing.

2) SFT Sets for CapTTS, EmoCapTTS, and AccCapTTS:

We group CapTTS, EmoCapTTS, and AccCapTTS together because they share both underlying speech corpora and common stylistic attributes (e.g., timbre, speaking rate). To support these

7https://huggingface.co/datasets/cdminix/libritts-r-aligned

tasks, we aggregate human-annotated data from six publicly available corpora: LibriTTS-R [32], VCTK [36], VoxCeleb [33], VoxCeleb2 [34], EARS [35], and Expresso [20]. The aggregation process involves directly using existing captions, enhancing captions with additional speaker traits, and generating captions using a large language model based on structured labels. For LibriTTS-R, we incorporate annotations from LibriTTS-P [26] and DreamVoiceDB [42], which particularly focus on intrinsic speaker traits, and use the LLM to generate captions. The prompt used for generating LibriTTS-P annotations is shown below.

#### Prompt Design for LibriTTS-P

System: You are a voice style caption generator. User:

- • Given a list of keywords related to a speaker’s vocal traits, generate 10 diverse and natural-sounding style captions.
- • You can freely drop or combine keywords. Vary the sentence structures and expressions as much as possible.
- • Respond with only the caption. Keywords: [e.g., teenager, female, bright, smooth, nasal, cute, and quick pace.] LLM Assistant: [e.g., A teenage girl’s voice is bright and smooth, with a slight nasal quality, and she speaks at a lively, quick pace.]

From DreamVoiceDB, we also use the VCTK portion, which includes accent annotations, and use the LLM to generate captions. The prompt used for this caption generation is shown below.

#### Prompt Design for VCTK

System: You are a voice style caption generator. User:

- • Please naturally incorporate the accent information into the following description, without changing its original meaning. Accent: [e.g., Indian Accent.] Original description: [e.g., A senior woman’s voice carries with warmth, depth, and an authoritative tone.]
- • You can freely vary the sentence structures and expressions as much as possible.
- • Respond with only the modified caption. LLM Assistant: [e.g., A senior woman’s voice resonates with warmth and depth, layered with an authoritative tone and gently marked by an Indian accent.]

For VoxCeleb, VoxCeleb2, EARS, and Expresso, we apply annotations from ParaSpeechCaps [25], which cover both intrinsic and expressive style traits.

Based on the collected and annotated captions, we construct SFT datasets to support three tasks: (1) CapTTS serves as a general-purpose task for caption-based speech synthesis. It leverages all above sources, resulting in 347,783 speech–caption pairs, with 18,348 used for validation and 20,756 held out for testing. (2) EmoCapTTS focuses on emotional expression,

using data from the EARS and Expresso corpora, which offer high-quality emotion annotations. This subset contains 26,428 samples, with 1,800 used for validation and 1,937 reserved for testing. (3) AccCapTTS targets accent control, utilizing data from VCTK, VoxCeleb, and VoxCeleb2, which offer reliable accent annotations. It comprises 113,197 samples, with 10,599 used for validation and 13,051 held out for testing.

- 3) SFT Set for CapTTS-SE: To the best of our knowledge, there are no existing open-source TTS datasets that include sound events. To address this gap, we propose a new humanprocessed high-quality dataset for the CapTTS-SE task, named CapSpeech-SEDB. CapSpeech-SEDB comprises 500 audio mixtures incorporating 10 common sound events (i.e. the sound of coughing, laughing, clapping, can opening, footsteps, keyboard typing, alarm clock, door knocking, dog barking, and cat meow), meticulously crafted by five audio engineers with expertise in music production or film sound design. To ensure the simulated audio sounds natural and smooth, we select insertion and background points between words with a minimum interval of 0.3 seconds, ensuring a pause that does not disrupt word continuity. For each speech recording and its corresponding set of sound event candidates, the engineers select the most appropriate event and follow detailed instructions—such as inserting the event or using it as background—to produce a new mixture. They fine-tune the result by balancing volume levels, applying fade-ins and fade-outs, and adjusting the equalizer and compressor to achieve a natural and seamless integration of speech and sound events. Moreover, we create another 500 transcription-caption pairs for testing.
- 4) SFT Set for AgentTTS: To the best of our knowledge, there are no existing open-source datasets that feature single-speaker recordings paired with diverse and fine-grained emotion and style prompts, suitable for applications such as customer service and AI therapy, which require nuanced expressiveness and broad stylistic control. To address this gap, we present CapSpeech-AgentDB, a new dataset comprising 10,000 caption-speech pairs, totaling approximately 25.2 hours of high-quality recordings by a single female speaker, in which 500 pairs are held out for testing. The dataset captures subtle gradations between emotional states (e.g., disappointed vs. sadness vs. grief, annoyed vs. angry vs. irate), and includes less commonly represented emotions (e.g., curious, jealous, resentful, focused, distracted), which are rarely present in existing speech emotion corpora. In addition, it features a wide variety of speaking styles (e.g., laughing, crying, panting, whispering), as well as nuanced combinations of emotions and styles (e.g., angry growl vs. angry scream, happy and slow vs. happy and fast), enabling fine-grained control over both expressive and stylistic variation.

To collect the recordings, we first generate prompt-content pairs, paying careful attention to balancing subcategories of emotions and styles. The content spans a wide range of conversational scenarios, including everyday dialogue, cafés, hospitals, classrooms, sci-fi films, horror movies, and more. During recording, the speaker follows the prompts to perform the specified emotion and style, and is allowed to revise the prompt or content if any part is unclear or unnatural.

The following prompt template guides an LLM in generating

expressive prosody descriptions and corresponding dialogue samples. As this task is more complex, we use GPT-4o-mini8 for better quality and naturalness. Below is the prompts used in this process.

Prompt Design for CapSpeech-AgentDB System: You are a speech and voice expert contributing to an expressive speech dataset.

User: Please generate a natural-language prosody prompt that vividly describes how the speaker’s voice would sound, given a high-level expression (e.g., emotion, mental state) and a low-level expression (e.g., pitch, pace, volume).

#### Guidelines:

- • Ensure that the prosody description is vivid, coherent, and captures the expression naturally.
- • It is acceptable to merge or omit specific low-level features for fluency.
- • Please follow these examples: [random sample 3 human-written style-captions] LLM Assistant: [e.g., The voice is quick and unsteady, occasionally faltering, revealing anxiousness.] User: Using the above prosody prompt and the provided conversation scene, generate a single line of dialogue or a short dialogue excerpt from only one speaker. Guidelines:
- • Ensure the speech sounds natural and is emotionally aligned with the described vocal tone and context.
- • Use varied sentence structures and lexical choices across examples to enhance dataset diversity.
- • Do not include or imply any lines from other characters—generate content exclusively for one speaker.
- • Optional: When appropriate, incorporate non-speech vocalizations from the provided list ([Gasp], [Laughter],

..., [Yawn]) to enhance the expressiveness of the speech. Conversation scene: [e.g., in a horror movie] LLM Assistant: [e.g., I—I think I saw something move behind the curtain. No, no, I’m not imagining it. It was real. I swear it was real. We can’t stay here. We have to go—now!]

These outputs form the basis of the CapSpeech-AgentDB, providing reference material for voice actors to produce the final audio recordings. The set of high-level emotional expressions, low-level prosody expressions, and non-speech vocalizations used to guide the prompts is presented in Table V.

IV. EXPERIMENTS A. Baselines

We adopt two strong baselines for the CapSpeech benchmark. CapSpeech-AR: The first is Parler-TTS9 [23], a state-of-

the-art AR method based on a codec language model. We use the 44.1kHz version of Descript Audio Codec (DAC) [58] to provide discrete audio representations. A delay pattern

- 8https://platform.openai.com/docs/models/gpt-4o-mini
- 9https://github.com/huggingface/parler-tts

is applied to handle multiple codebooks [59], and crossattention is used to incorporate caption-based style conditioning. FLAN-T510 [60] is used to extract the text features from the transcription and style caption. We retain the original architecture, with the exception of adding special tokens to support the CapTTS-SE task. The FLAN-T5 model used in the transcription is fine-tuned. As illustrated in Fig. 1, transcriptions in CapTTS-SE support two modes for integrating sound events: (i) Background mode and (ii) Insertion mode. sound event tokens (e.g., <telephone>, <knock>) are placed at the beginning of the sequence. The tags <B> and </B> mark the start and end of a background sound segment, while <I> </I> denote the insertion point. This design allows flexible control over the placement and type of sound events within the synthesized speech.

CapSpeech-NAR: In addition, we adopt F5-TTS11 [8], a state-of-the-art NAR method based on flow matching with a Diffusion Transformer. In our adaptation, we remove the audioprompt masking component and instead use cross-attention to incorporate caption-based style conditioning. BigVGAN12 [61] is used as the vocoder, and we apply QK-Norm [62] to stabilize training. Transcriptions are processed using graphemeto-phoneme conversion (g2p)13, and we insert special tokens <B>, </B>, <I>, and </I> to indicate background and insertion-based sound events. To enhance generalization, we do not include sound event tags directly in the input sequence. Instead, we extract LAION-CLAP [63] embeddings for the specified sound event and use them as the additional input. This design allows the model to generalize to unseen sound events during inference. Similar to the AR model, a Flan-T5 model is used to extract the text features from the style caption. Since the NAR model cannot predict audio duration directly, inspired by [4], we fine-tune a BERT model14 [64] that takes both the transcription and the caption as input to estimate the total duration of the entire audio.

B. Implementation Details

Table VI summarizes the computation cost and model size of AR and NAR models. In the Transformer column, the numbers denote the model dimension, the number of layers, the number of heads, and the multiples of hidden size. Training time is calculated on 8 NVIDIA-H100 GPUs and Real-time-factor (RTF) is calculated on a single NVIDIA-A100 GPU.

TABLE VI: Model complexity and computational cost.

Model Transformer Params(M) Training Time RTF CapSpeech-AR 1024,24,16,4 880 24 Days 1.24

CapSpeech-NAR 1024,24,16,4 724 10 Days 1.01

During pretraining of the AR model, we use a batch size of 32, a learning rate of 1e-3, a weight decay of 0.01, and gradient accumulation over 6 steps. Linear warm-up is applied

- 10https://huggingface.co/google/flan-t5-large
- 11https://github.com/SWivid/F5-TTS
- 12https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x
- 13https://github.com/Kyubyong/g2p
- 14https://huggingface.co/google-bert/bert-base-uncased

TABLE V: List of Expressive Voice Labels

Neutral Aggressive Angry Annoyed Contemptuous Hateful Raging

Disgusted Sarcastic Resentful Alarmed Apprehensive Fearful Nervous Panicked Terrified Accepting Admiring Amazed Excited Hopeful Interested Joyful Loving Peaceful Surprised Disappointed Dismayed

High-level Emotional Expressions

Distressed Grieving Hopeless Sad Authoritative Respectful Sympathetic

Trusting Anticipating Curious Eager Focused Tired Awkward Confused Distracted Skeptical Bored Envious Submissive Shameful

Default Monotonous Deep Sharp Gentle Loud Mumble

Low-level Prosody Expressions

Stutter Whispering Crying Laughing

Aww Throat-clearing Cheering Contemplation Gasp Groan Laughter

Non-speech Voices

Panting Scream Sigh Sneering laughter Sob Yawn

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

(a): CapSpeech-AR model (b): CapSpeech-NAR model

Fig. 2: Illustration of CapSpeech model architectures.

for the first 5000 steps. The model is trained for 10 epochs, taking approximately 24 days on 8 NVIDIA H100 GPUs. For fine-tuning the AR model, the learning rate is set to 1e-4. Warm-up is applied for 1000 steps for the CapTTS, EmoCapTTS, and AccCapTTS tasks, and for 100 steps for the AgentTTS and CapTTS-SE tasks. Fine-tuning on CapTTS, EmoCapTTS, and AccCapTTS takes 47 hours, 24 hours, and 24 hours, respectively (over 5 epochs) on a single NVIDIA A100 GPU. Fine-tuning on AgentTTS and CapTTS-SE takes 14 hours and 1 hour, respectively (over 50 epochs) on a single NVIDIA A100 GPU.

For the NAR model, pretraining is conducted with a batch size of 512, a learning rate of 2e-4, and a weight decay of 0.01. A linear warm-up of 5000 steps is followed by linear decay over the remaining training steps. The model is trained for 10 epochs, requiring approximately 10 days on 8 NVIDIA H100 GPUs. During fine-tuning of the NAR model, the learning rate is set to 2e-5. Warm-up is applied for 1000 steps for CapTTS, EmoCapTTS, and AccCapTTS, and for 100 steps for AgentTTS and CapTTS-SE. Fine-tuning on CapTTS, EmoCapTTS, and AccCapTTS takes 69 hours, 33 hours, and 33 hours, respectively (over 15 epochs) on a single NVIDIA A100 GPU. Fine-tuning on AgentTTS and CapTTS-SE takes 6.8 hours and 0.6 hours, respectively (over 50 epochs) on a single NVIDIA A100 GPU. For the duration predictor, we finetune the bert-base-uncased model with a learning rate of 1e-4 and a batch size of 256. The model is trained on the CapSpeech pretraining set for 2 epochs using a single NVIDIA A100 GPU, which takes approximately 70 hours. This predictor is used

for the CapTTS, CapTTS-SE, EmoCapTTS, and AccCapTTS tasks. For the AgentTTS task, we further fine-tune the model for the single female speaker using a learning rate of 1e-5 for 10 epochs, which takes about half an hour.

All models are trained using the AdamW optimizer [65]. Following [25], we initialize the AR model with the Parler-TTS Mini v115 model. During inference, we perform AR generation with a temperature of 1.0 and a repetition penalty of 1.0, and introduce inference-only classifier-free guidance (CFG) [66]. For the NAR model, following [8], we use the default CFG strength of 2 and a Sway Sampling coefficient of −1. To avoid any potential misuse, we introduce a novel watermarking technology [66] to each model, and open-source our code and model weights under CC BY-NC 4.0 license.

C. Metrics

Objective Metrics: We evaluate three key aspects of model performance: style consistency, audio quality, and intelligibility. For style consistency, we compute classification accuracy across multiple categories: age, gender, pitch, expressiveness of tone, speed, accent, and emotion. The average accuracy across these attributes is reported as Style-ACC. Following Section III-B1, we utilized existing toolkits to predict various speaker attributes, including age, gender, pitch, vocal expressiveness, and speaking rate. For emotion and accent classification, we introduced the state-of-the-art classifiers in [67]. The emotion classifier includes nine emotion labels: Anger, Contempt, Disgust, Fear,

15https://huggingface.co/parler-tts/parler-tts-mini-v1

TABLE VII: Results on the pretraining task.

Objective Evaluations Subjective Evaluations

Model Data

Style-ACC↑ UTMOSv2↑ WER↓ SMOS↑ NMOS↑ IMOS↑ AR

ParaSpeechCaps 44.0% 2.93 10.8% 3.48±0.12 3.21±0.08 3.88±0.12 Ours 52.2% 3.18 9.1% 3.62±0.11 3.65±0.10 4.23±0.12

ParaSpeechCaps 51.8% 3.16 9.5% 3.55±0.11 3.38±0.09 3.93±0.10 Ours 62.1% 3.43 8.8% 3.80±0.13 3.79±0.10 4.40±0.09

NAR

Happiness, Neutral, Sadness, Surprise, and Other. The accent classifier covers 16 accent categories: East Asia, English, Germanic (e.g., German), Irish, North America, Northern Irish, Oceania (e.g., Australia), Romance (e.g., Spanish, French), Scottish, Semitic, Slavic (e.g., Russian, Polish), South African, Southeast Asia, South Asia, Welsh, and Other accents not covered above.

Audio quality is assessed using UTMOSv216 [68]. For intelligibility, we compute a text-normalized WER between the ASR transcript of the generated speech and the input transcript. This is done using openai/whisper-large-v3-turbo along with the Whisper text normalizer.

Subjective Metrics: We experimented with state-of-theart speech understanding models [69], [70] to evaluate style consistency, but they failed to generate high-quality speech captions. Following [25], we recruit 15 native speakers via Prolific17 to evaluate three subjective aspects: Style Consistency MOS (SMOS), Naturalness MOS (NMOS), and Intelligibility MOS (IMOS). We recruit high-quality participants from Prolific with a minimum approval rate of 95%, at least 100 prior approved submissions. Before annotations, we perform a qualification task using 10 manually-selected samples, and select 15 annotators that succeeded on at least 9. For each task, we randomly select 100 test samples from the test sets and ensure that each sample is rated by three annotators. For the CapTTS task, annotators are instructed to evaluate overall style consistency. For the EmoCapTTS and AccCapTTS tasks, they are asked to place greater emphasis on emotion and accent characteristics. In the CapTTS-SE task, annotators are instructed to factor in the accuracy of sound events when rating intelligibility. For the AgentTTS task, speaker similarity to the reference recording is also considered as part of the intelligibility score. Each example is rated by three annotators, and we report the mean scores along with 95% confidence intervals.

D. Results and Analysis

Pretraining Stage: We compare the CapTTS task trained on our proposed CapTTS pretraining set against training on the previous large-scale dataset ParaSpeechCaps (PSC-Scaled partition). Table VII presents the results on the CapTTS task, comparing models using both objective and subjective metrics. Models trained on the CapTTS Pretraining set achieve significantly better style consistency, naturalness, and intelligibility than those trained on ParaSpeechCaps, demonstrating

- 16https://github.com/sarulab-speech/UTMOSv2
- 17https://www.prolific.com/

the effectiveness of our dataset. Compared to AR models, NAR models consistently achieve better performance across all metrics, highlighting their advancement on the CapTTS task.

Fine-tuning Stage: We pre-train the AR and NAR models on our proposed CapTTS and CapTTS-SE pretraining sets, and then fine-tune them on the downstream tasks. The results, shown in Table VIII, demonstrate that pretraining provides substantial benefits across all downstream tasks–particularly for CapTTS-SE and AgentTTS, where training data is limited. Notably, our benchmark indicates that strong style consistency, naturalness, and intelligibility can be achieved on the CapTTS, EmoCapTTS, and AccCapTTS tasks, with the NAR model reaching SMOS, NMOS, and IMOS scores of at least 3.77, 3.88, and 4.34, respectively. Additionally, the AR model surpasses the NAR model on certain metrics in the CapTTS-SE and AgentTTS tasks. We observe that maintaining style consistency in the AgentTTS task and achieving high intelligibility in the CapTTS-SE task remain particularly challenging. In particular, models achieve good WER results but perform poorly on IMOS in the CapTTS-SE task, indicating that sound events are generated with lower quality than speech.

V. CONCLUSION AND DISCUSSIONS

We propose CapSpeech, a comprehensive benchmark for a series of CapTTS-related tasks. The CapSpeech dataset consists of a large collection of audio-caption pairs exhibiting diverse speaking styles. In addition, we introduce two new real-world datasets for the AgentTTS and CapTTS-SE tasks. We evaluate two strong models on the benchmark, both demonstrating highfidelity and highly intelligible speech generation. However, we find that maintaining style consistency in the AgentTTS task and achieving high intelligibility in the CapTTS-SE task remain particularly challenging.

Some limitations of this work are language coverage and evaluation metrics. While the design of our benchmark can be readily extended to other languages, the current datasets are limited to English. Moreover, the style-captioned TTS tasks rely on costly and subjective human evaluations due to the absence of reliable automatic metrics. Currently, no existing understanding model can generate high-quality speech captions. However, our dataset provides a promising foundation for training such models, analogous to image-text models like CLIP [71] and BLIP [72]. In addition, the scale of CapSpeechSEDB and CapSpeech-AgentDB is still limited, we are actively collecting more data in the future works.

TABLE VIII: Results on the fine-tuning tasks. ⋆ denotes models without pretraining. CapTTS-SE results without pretraining are not reported as there are only 500 samples in CapTTS-SEDB. Style-ACC is not applicable to AgentTTS and CapTTS-SE, and UTMOSv2 is not applicable to CapTTS-SE.

Objective Evaluations Subjective Evaluations

Task Model

Style-ACC↑ UTMOSv2↑ WER↓ SMOS↑ NMOS↑ IMOS↑

##### AR⋆ 42.1% 2.45 21.5% 3.21±0.17 3.40±0.14 3.86±0.12 AR 56.0% 3.02 11.2% 3.72±0.12 3.62±0.11 4.15±0.10 NAR⋆ 46.4% 2.61 19.5% 3.40±0.12 3.60±0.13 3.95±0.10 NAR 66.0% 3.37 9.2% 3.85±0.13 3.95±0.12 4.34±0.11

CapTTS

AR / / 7.7% 3.69±0.12 3.52±0.14 3.45±0.14

CapTTS-SE

#### NAR / / 3.0% 3.75±0.13 3.60±0.11 3.33±0.15

##### AR⋆ 40.5% 1.98 54.8% 2.52±0.13 2.83±0.13 3.05±0.14 AR 58.6% 3.04 11.5% 3.72±0.14 3.58±0.12 4.12±0.13 NAR⋆ 44.5% 2.10 45.5% 2.72±0.14 3.00±0.12 3.28±0.12 NAR 67.2% 3.34 10.1% 3.77±0.12 3.88±0.13 4.35±0.10

EmoCapTTS

##### AR⋆ 37.2% 2.06 48.0% 2.98±0.10 3.30±0.12 3.75±0.14 AR 54.9% 3.10 10.9% 3.77±0.12 3.67±0.11 4.20±0.11 NAR⋆ 39.2% 2.41 30.2% 3.12±0.12 3.36±0.11 3.88±0.12 NAR 66.4% 3.45 8.8% 3.91±0.12 3.90±0.12 4.45±0.09

AccCapTTS

##### AR⋆ / 1.85 56.9% 1.50±0.22 2.35±0.20 2.52±0.13 AR / 3.26 10.2% 3.50±0.14 3.70±0.11 4.22±0.12 NAR⋆ / 1.92 54.8% 1.82±0.18 2.10±0.15 2.77±0.15 NAR / 3.07 9.5% 3.42±0.14 3.80±0.11 4.41±0.10

AgentTTS

REFERENCES

- [1] S. Chen, S. Liu, L. Zhou, Y. Liu, X. Tan, J. Li, S. Zhao, Y. Qian, and F. Wei, “VALL-E 2: Neural codec language models are human parity zero-shot text to speech synthesizers,” CoRR, vol. abs/2406.05370, 2024.
- [2] Z. Ju, Y. Wang, K. Shen, X. Tan, D. Xin, D. Yang, E. Liu, Y. Leng,

- K. Song, S. Tang, Z. Wu, T. Qin, X. Li, W. Ye, S. Zhang, J. Bian,
- L. He, J. Li, and S. Zhao, “Naturalspeech 3: Zero-shot speech synthesis with factorized codec and diffusion models,” in Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024.

- [3] Z. Ye, X. Zhu, C. Chan, X. Wang, X. Tan, J. Lei, Y. Peng, H. Liu, Y. Jin, Z. Dai, H. Lin, J. Chen, X. Du, L. Xue, Y. Chen, Z. Li, L. Xie, Q. Kong, Y. Guo, and W. Xue, “Llasa: Scaling train-time and inference-time compute for llama-based speech synthesis,” CoRR, vol. abs/2502.04128, 2025.
- [4] D. Yang, R. Huang, Y. Wang, H. Guo, D. Chong, S. Liu, X. Wu, and H. Meng, “Simplespeech 2: Towards simple and efficient text-to-speech with flow-based scalar latent transformer diffusion models,” CoRR, vol. abs/2408.13893, 2024.
- [5] Z. Du, Y. Wang, Q. Chen, X. Shi, X. Lv, T. Zhao, Z. Gao, Y. Yang, C. Gao, H. Wang, F. Yu, H. Liu, Z. Sheng, Y. Gu, C. Deng, W. Wang, S. Zhang, Z. Yan, and J. Zhou, “Cosyvoice 2: Scalable streaming speech synthesis with large language models,” CoRR, vol. abs/2412.10117, 2024.
- [6] Y. Wang, H. Zhan, L. Liu, R. Zeng, H. Guo, J. Zheng, Q. Zhang, S. Zhang, and Z. Wu, “Maskgct: Zero-shot text-to-speech with masked generative codec transformer,” CoRR, vol. abs/2409.00750, 2024.
- [7] P. Anastassiou, J. Chen, J. Chen, Y. Chen, Z. Chen, Z. Chen, J. Cong, L. Deng, C. Ding, L. Gao, M. Gong, P. Huang, Q. Huang, Z. Huang,

- Y. Huo, D. Jia, C. Li, F. Li, H. Li, J. Li, X. Li, X. Li, L. Liu, S. Liu, S. Liu, X. Liu, Y. Liu, Z. Liu, L. Lu, J. Pan, X. Wang, Y. Wang, Y. Wang,
- Z. Wei, J. Wu, C. Yao, Y. Yang, Y. Yi, J. Zhang, Q. Zhang, S. Zhang, W. Zhang, Y. Zhang, Z. Zhao, D. Zhong, and X. Zhuang, “Seed-tts: A family of high-quality versatile speech generation models,” CoRR, vol. abs/2406.02430, 2024.

- [8] Y. Chen, Z. Niu, Z. Ma, K. Deng, C. Wang, J. Zhao, K. Yu, and X. Chen, “F5-TTS: A fairytaler that fakes fluent and faithful speech with flow matching,” CoRR, vol. abs/2410.06885, 2024.
- [9] S. E. Eskimez, X. Wang, M. Thakker, C. Li, C. Tsai, Z. Xiao, H. Yang, Z. Zhu, M. Tang, X. Tan, Y. Liu, S. Zhao, and N. Kanda, “E2 TTS:

- embarrassingly easy fully non-autoregressive zero-shot TTS,” in IEEE Spoken Language Technology Workshop, SLT 2024, Macao, December 2-5, 2024. IEEE, 2024, pp. 682–689.
- [10] H. Wu, X. Wang, S. E. Eskimez, M. Thakker, D. Tompkins, C. Tsai, C. Li, Z. Xiao, S. Zhao, J. Li, and N. Kanda, “Laugh now cry later: Controlling time-varying emotional states of flow-matching-based zeroshot text-to-speech,” in IEEE Spoken Language Technology Workshop, SLT 2024, Macao, December 2-5, 2024. IEEE, 2024, pp. 690–697.
- [11] Y. Guo, C. Du, X. Chen, and K. Yu, “Emodiff: Intensity controllable emotional text-to-speech with soft-label guidance,” in IEEE International Conference on Acoustics, Speech and Signal Processing ICASSP 2023, Rhodes Island, Greece, June 4-10, 2023. IEEE, 2023, pp. 1–5.
- [12] Y. Lei, S. Yang, X. Wang, and L. Xie, “Msemotts: Multi-scale emotion transfer, prediction, and control for emotional speech synthesis,” IEEE ACM Trans. Audio Speech Lang. Process., vol. 30, pp. 853–864, 2022.
- [13] X. Zhou, M. Zhang, Y. Zhou, Z. Wu, and H. Li, “Accented text-tospeech synthesis with limited data,” IEEE ACM Trans. Audio Speech Lang. Process., vol. 32, pp. 1699–1711, 2024.
- [14] D. Cho, H. Oh, S. Kim, S. Lee, and S. Lee, “Emosphere-tts: Emotional style and intensity modeling via spherical emotion vector for controllable emotional text-to-speech,” CoRR, vol. abs/2406.07803, 2024.
- [15] H. Tang, X. Zhang, N. Cheng, J. Xiao, and J. Wang, “ED-TTS: multi-scale emotion modeling using cross-domain emotion diarization for emotional speech synthesis,” in IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2024, Seoul, Republic of Korea, April 14-19, 2024. IEEE, 2024, pp. 12146–12150.
- [16] Z. Guo, Y. Leng, Y. Wu, S. Zhao, and X. Tan, “Prompttts: Controllable text-to-speech with text descriptions,” in IEEE International Conference on Acoustics, Speech and Signal Processing ICASSP 2023, Rhodes Island, Greece, June 4-10, 2023. IEEE, 2023, pp. 1–5.
- [17] Z. Jin, J. Jia, Q. Wang, K. Li, S. Zhou, S. Zhou, X. Qin, and Z. Wu, “Speechcraft: A fine-grained expressive speech dataset with natural language description,” in Proceedings of the 32nd ACM International Conference on Multimedia, MM 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November 2024, J. Cai, M. S. Kankanhalli, B. Prabhakaran, S. Boll, R. Subramanian, L. Zheng, V. K. Singh, P. César, L. Xie, and D. Xu, Eds. ACM, 2024, pp. 1255–1264.
- [18] Y. Leng, Z. Guo, K. Shen, Z. Ju, X. Tan, E. Liu, Y. Liu, D. Yang, L. Zhang, K. Song, L. He, X. Li, S. Zhao, T. Qin, and J. Bian,

- “Prompttts 2: Describing and generating voices with text prompt,” in The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.
- [19] R. Shimizu, R. Yamamoto, M. Kawamura, Y. Shirahata, H. Doi, T. Komatsu, and K. Tachibana, “Prompttts++: Controlling speaker identity in prompt-based text-to-speech using natural language descriptions,” in IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2024, Seoul, Republic of Korea, April 14-19, 2024. IEEE, 2024, pp. 12672–12676.
- [20] T. A. Nguyen, W. Hsu, A. D’Avirro, B. Shi, I. Gat, M. Fazel-Zarandi, T. Remez, J. Copet, G. Synnaeve, M. Hassid, F. Kreuk, Y. Adi, and E. Dupoux, “Expresso: A benchmark and analysis of discrete expressive speech resynthesis,” in 24th Annual Conference of the International Speech Communication Association, Interspeech 2023, Dublin, Ireland, August 20-24, 2023, N. Harte, J. Carson-Berndsen, and G. Jones, Eds. ISCA, 2023, pp. 4823–4827.
- [21] D. Yang, S. Liu, R. Huang, C. Weng, and H. Meng, “Instructtts: Modelling expressive TTS in discrete latent space with natural language style prompt,” IEEE ACM Trans. Audio Speech Lang. Process., vol. 32, pp. 2913–2925, 2024.
- [22] X. Zhu, W. Tian, X. Wang, L. He, Y. Xiao, X. Wang, X. Tan, S. Zhao, and L. Xie, “Unistyle: Unified style modeling for speaking style captioning and stylistic speech synthesis,” in Proceedings of the 32nd ACM International Conference on Multimedia, MM 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November 2024, J. Cai, M. S. Kankanhalli, B. Prabhakaran, S. Boll, R. Subramanian, L. Zheng, V. K. Singh, P. César, L. Xie, and D. Xu, Eds. ACM, 2024, pp. 7513–7522.
- [23] D. Lyth and S. King, “Natural language guidance of high-fidelity text-tospeech with synthetic annotations,” CoRR, vol. abs/2402.01912, 2024.
- [24] Y. Lacombe, V. Srivastav, and S. Gandhi, “Parler-tts,” https://github.com/ huggingface/parler-tts, 2024.
- [25] A. Diwan, Z. Zheng, D. Harwath, and E. Choi, “Scaling rich styleprompted text-to-speech datasets,” CoRR, vol. abs/2503.04713, 2025.
- [26] M. Kawamura, R. Yamamoto, Y. Shirahata, T. Hasumi, and K. Tachibana, “Libritts-p: A corpus with speaking style and speaker identity prompts for text-to-speech and style captioning,” CoRR, vol. abs/2406.07969, 2024.
- [27] G. Yang, C. Yang, Q. Chen, Z. Ma, W. Chen, W. Wang, T. Wang, Y. Yang, Z. Niu, W. Liu et al., “Emovoice: Llm-based emotional text-to-speech model with freestyle text prompting,” arXiv preprint arXiv:2504.12867, 2025.
- [28] H. He, Z. Shang, C. Wang, X. Li, Y. Gu, H. Hua, L. Liu, C. Yang, J. Li, P. Shi, Y. Wang, K. Chen, P. Zhang, and Z. Wu, “Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation,” in IEEE Spoken Language Technology Workshop, SLT 2024, Macao, December 2-5, 2024. IEEE, 2024, pp. 885–890.
- [29] G. Chen, S. Chai, G. Wang, J. Du, W. Zhang, C. Weng, D. Su, D. Povey, J. Trmal, J. Zhang, M. Jin, S. Khudanpur, S. Watanabe, S. Zhao, W. Zou, X. Li, X. Yao, Y. Wang, Z. You, and Z. Yan, “Gigaspeech: An evolving, multi-domain ASR corpus with 10, 000 hours of transcribed audio,” in 22nd Annual Conference of the International Speech Communication Association, Interspeech 2021, Brno, Czechia, August 30 - September 3, 2021, H. Hermansky, H. Cernocký, L. Burget, L. Lamel, O. Scharenborg, and P. Motlícek, Eds. ISCA, 2021, pp. 3670–3674.
- [30] R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. M. Tyers, and G. Weber, “Common voice: A massively-multilingual speech corpus,” in Proceedings of The 12th Language Resources and Evaluation Conference, LREC 2020, Marseille, France, May 11-16, 2020, N. Calzolari, F. Béchet, P. Blache, K. Choukri, C. Cieri, T. Declerck, S. Goggi, H. Isahara, B. Maegaard, J. Mariani, H. Mazo, A. Moreno, J. Odijk, and S. Piperidis, Eds. European Language Resources Association, 2020, pp. 4218–4222.
- [31] V. Pratap, Q. Xu, A. Sriram, G. Synnaeve, and R. Collobert, “MLS: A large-scale multilingual dataset for speech research,” in 21st Annual Conference of the International Speech Communication Association, Interspeech 2020, Virtual Event, Shanghai, China, October 25-29, 2020, H. Meng, B. Xu, and T. F. Zheng, Eds. ISCA, 2020, pp. 2757–2761.
- [32] Y. Koizumi, H. Zen, S. Karita, Y. Ding, K. Yatabe, N. Morioka, M. Bacchiani, Y. Zhang, W. Han, and A. Bapna, “Libritts-r: A restored multi-speaker text-to-speech corpus,” in 24th Annual Conference of the International Speech Communication Association, Interspeech 2023, Dublin, Ireland, August 20-24, 2023, N. Harte, J. Carson-Berndsen, and G. Jones, Eds. ISCA, 2023, pp. 5496–5500.
- [33] A. Nagrani, J. S. Chung, and A. Zisserman, “Voxceleb: A largescale speaker identification dataset,” in 18th Annual Conference of the International Speech Communication Association, Interspeech 2017, Stockholm, Sweden, August 20-24, 2017, F. Lacerda, Ed. ISCA, 2017, pp. 2616–2620.

- [34] J. S. Chung, A. Nagrani, and A. Zisserman, “Voxceleb2: Deep speaker recognition,” in 19th Annual Conference of the International Speech Communication Association, Interspeech 2018, Hyderabad, India, September 2-6, 2018, B. Yegnanarayana, Ed. ISCA, 2018, pp. 1086–1090.
- [35] J. Richter, Y. Wu, S. Krenn, S. Welker, B. Lay, S. Watanabe, A. Richard, and T. Gerkmann, “EARS: an anechoic fullband speech dataset benchmarked for speech enhancement and dereverberation,” CoRR, vol. abs/2406.06185, 2024.
- [36] J. Yamagishi, C. Veaux, and K. MacDonald, “Cstr vctk corpus: English multi-speaker corpus for cstr voice cloning toolkit (version 0.92),” 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:213060286
- [37] H. Chen, W. Xie, A. Vedaldi, and A. Zisserman, “Vggsound: A largescale audio-visual dataset,” in 2020 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2020, Barcelona, Spain, May 4-8, 2020. IEEE, 2020, pp. 721–725.
- [38] E. Fonseca, M. Plakal, F. Font, D. P. W. Ellis, X. Favory, J. Pons, and X. Serra, “General-purpose tagging of freesound audio with audioset labels: task description, dataset, and baseline,” in Proceedings of the Workshop on Detection and Classification of Acoustic Scenes and Events, DCASE 2018, Surrey, UK, November 19-20, 2018, M. D. Plumbley, C. Kroos, J. P. Bello, G. Richard, D. P. W. Ellis, and A. Mesaros, Eds., 2018, pp. 69–73.
- [39] K. J. Piczak, “ESC: Dataset for Environmental Sound Classification,” in Proceedings of the 23rd Annual ACM Conference on Multimedia. ACM Press, pp. 1015–1018. [Online]. Available: http://dl.acm.org/citation.cfm? doid=2733373.2806390
- [40] S. Ji, J. Zuo, M. Fang, Z. Jiang, F. Chen, X. Duan, B. Huai, and Z. Zhao, “Textrolspeech: A text style control speech corpus with codec language text-to-speech models,” in IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2024, Seoul, Republic of Korea, April 14-19, 2024. IEEE, 2024, pp. 10301–10305.
- [41] S. Ji, J. Zuo, M. Fang, S. Zheng, Q. Chen, W. Wang, Z. Jiang, H. Huang, X. Cheng, R. Huang, and Z. Zhao, “Controlspeech: Towards simultaneous zero-shot speaker cloning and zero-shot language style control with decoupled codec,” CoRR, vol. abs/2406.01205, 2024.
- [42] J. Hai, K. Thakkar, H. Wang, Z. Qin, and M. Elhilali, “Dreamvoice: Text-guided voice conversion,” arXiv preprint arXiv:2406.16314, 2024.
- [43] R. Ye, Y. Zhou, R. Yu, Z. Lin, K. Li, X. Li, X. Liu, G. Zeng, and Z. Wu, “A scalable pipeline for enabling non-verbal speech generation and understanding,” arXiv preprint arXiv:2508.05385, 2025.
- [44] Y. Zhou, X. Qin, Z. Jin, S. Zhou, S. Lei, S. Zhou, Z. Wu, and J. Jia, “Voxinstruct: Expressive human instruction-to-speech generation with unified multilingual codec language modelling,” in Proceedings of the 32nd ACM International Conference on Multimedia, MM 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November 2024, J. Cai, M. S. Kankanhalli, B. Prabhakaran, S. Boll, R. Subramanian, L. Zheng, V. K. Singh, P. César, L. Xie, and D. Xu, Eds. ACM, 2024, pp. 554–563.
- [45] H. Li, Y. Li, X. Wang, J. Hu, Q. Xie, S. Yang, and L. Xie, “Flespeech: Flexibly controllable speech generation with various prompts,” CoRR, vol. abs/2501.04644, 2025.
- [46] A. Vyas, B. Shi, M. Le, A. Tjandra, Y. Wu, B. Guo, J. Zhang, X. Zhang, R. Adkins, W. Ngan, J. Wang, I. Cruz, B. Akula, A. Akinyemi, B. Ellis, R. Moritz, Y. Yungster, A. Rakotoarison, L. Tan, C. Summers, C. Wood, J. Lane, M. Williamson, and W. Hsu, “Audiobox: Unified audio generation with natural language prompts,” CoRR, vol. abs/2312.15821, 2023.
- [47] G. Liu, Y. Zhang, Y. Lei, Y. Chen, R. Wang, L. Xie, and Z. Li, “Promptstyle: Controllable style transfer for text-to-speech with natural language descriptions,” in 24th Annual Conference of the International Speech Communication Association, Interspeech 2023, Dublin, Ireland, August 20-24, 2023, N. Harte, J. Carson-Berndsen, and G. Jones, Eds. ISCA, 2023, pp. 4888–4892.
- [48] W. Guan, Y. Li, T. Li, H. Huang, F. Wang, J. Lin, L. Huang, L. Li, and Q. Hong, “MM-TTS: multi-modal prompt based style transfer for expressive text-to-speech synthesis,” in Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, M. J. Wooldridge, J. G. Dy, and S. Natarajan, Eds. AAAI Press, 2024, pp. 18117–18125.
- [49] A. Watanabe, S. Takamichi, Y. Saito, W. Nakata, D. Xin, and H. Saruwatari, “COCO-NUT: corpus of japanese utterance and voice characteristics description for prompt-based control,” in IEEE Automatic Speech Recognition and Understanding Workshop, ASRU 2023, Taipei, Taiwan, December 16-20, 2023. IEEE, 2023, pp. 1–8.
- [50] ——, “Building speech corpus with diverse voice characteristics for its prompt-based representation,” CoRR, vol. abs/2403.13353, 2024.

- [51] Y. Ren, C. Hu, X. Tan, T. Qin, S. Zhao, Z. Zhao, and T. Liu, “Fastspeech 2: Fast and high-quality end-to-end text to speech,” in 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.
- [52] K. Shen, Z. Ju, X. Tan, E. Liu, Y. Leng, L. He, T. Qin, S. Zhao, and J. Bian, “Naturalspeech 2: Latent diffusion models are natural and zero-shot speech and singing synthesizers,” in The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.
- [53] S. Chen, C. Wang, Y. Wu, Z. Zhang, L. Zhou, S. Liu, Z. Chen, Y. Liu, H. Wang, J. Li, L. He, S. Zhao, and F. Wei, “Neural codec language models are zero-shot text to speech synthesizers,” IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 705–718, 2025.
- [54] H. Tang, X. Zhang, J. Wang, N. Cheng, and J. Xiao, “Emomix: Emotion mixing via diffusion models for emotional speech synthesis,” in 24th Annual Conference of the International Speech Communication Association, Interspeech 2023, Dublin, Ireland, August 20-24, 2023, N. Harte, J. Carson-Berndsen, and G. Jones, Eds. ISCA, 2023, pp. 12–16.
- [55] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and

I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 2023, pp. 28492–28518.

- [56] A. Kumar, K. Tan, Z. Ni, P. Manocha, X. Zhang, E. Henderson, and B. Xu, “Torchaudio-squim: Reference-less speech quality and intelligibility measures in torchaudio,” in IEEE International Conference on Acoustics, Speech and Signal Processing ICASSP 2023, Rhodes Island, Greece, June 4-10, 2023. IEEE, 2023, pp. 1–5.
- [57] F. Burkhardt, J. Wagner, H. Wierstorf, F. Eyben, and B. Schuller, “Speechbased age and gender prediction with transformers,” P. Jax and S. Mölller, Eds., 2023.
- [58] R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar, “Highfidelity audio compression with improved RVQGAN,” in Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., 2023.
- [59] J. Copet, F. Kreuk, I. Gat, T. Remez, D. Kant, G. Synnaeve, Y. Adi, and A. Défossez, “Simple and controllable music generation,” in Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., 2023.
- [60] H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus,

- Y. Li, X. Wang, M. Dehghani, S. Brahma, A. Webson, S. S. Gu,
- Z. Dai, M. Suzgun, X. Chen, A. Chowdhery, A. Castro-Ros, M. Pellat, K. Robinson, D. Valter, S. Narang, G. Mishra, A. Yu, V. Y. Zhao, Y. Huang, A. M. Dai, H. Yu, S. Petrov, E. H. Chi, J. Dean, J. Devlin, A. Roberts, D. Zhou, Q. V. Le, and J. Wei, “Scaling instruction-finetuned language models,” J. Mach. Learn. Res., vol. 25, pp. 70:1–70:53, 2024.

- [61] S. Lee, W. Ping, B. Ginsburg, B. Catanzaro, and S. Yoon, “Bigvgan: A universal neural vocoder with large-scale training,” in The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.
- [62] A. Henry, P. R. Dachapally, S. S. Pawar, and Y. Chen, “Query-key normalization for transformers,” in Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020, ser. Findings of ACL, T. Cohn, Y. He, and Y. Liu, Eds., vol. EMNLP

2020. Association for Computational Linguistics, 2020, pp. 4246–4253.

- [63] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” in IEEE International Conference on Acoustics, Speech and Signal Processing ICASSP 2023, Rhodes Island, Greece, June 4-10, 2023. IEEE, 2023, pp. 1–5.
- [64] J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: pre-training of deep bidirectional transformers for language understanding,” CoRR, vol. abs/1810.04805, 2018.
- [65] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.
- [66] H. Wang, M. Yu, J. Hai, C. Chen, Y. Hu, R. Chen, N. Dehak, and D. Yu, “Ssr-speech: Towards stable, safe and robust zero-shot text-based speech editing and synthesis,” CoRR, vol. abs/2409.07556, 2024.
- [67] T. Feng, J. Lee, A. Xu, Y. Lee, T. Lertpetchpun, X. Shi, H. Wang, T. Thebaud, L. Moro-Velazquez, D. Byrd et al., “Vox-profile: A speech

- foundation model benchmark for characterizing diverse speaker and speech traits,” arXiv preprint arXiv:2505.14648, 2025.
- [68] K. Baba, W. Nakata, Y. Saito, and H. Saruwatari, “The T05 system for the voicemos challenge 2024: Transfer learning from deep image classifier to naturalness MOS prediction of high-quality synthetic speech,” in IEEE Spoken Language Technology Workshop, SLT 2024, Macao, December 2-5, 2024. IEEE, 2024, pp. 818–824.
- [69] Y. Chu, J. Xu, Q. Yang, H. Wei, X. Wei, Z. Guo, Y. Leng, Y. Lv, J. He, J. Lin, C. Zhou, and J. Zhou, “Qwen2-audio technical report,” CoRR, vol. abs/2407.10759, 2024.
- [70] KimiTeam, D. Ding, Z. Ju, Y. Leng, S. Liu, T. Liu, Z. Shang, K. Shen, W. Song, X. Tan, H. Tang, Z. Wang, C. Wei, Y. Xin, X. Xu, J. Yu, Y. Zhang, X. Zhou, Y. Charles, J. Chen, Y. Chen, Y. Du, W. He, Z. Hu, G. Lai, Q. Li, Y. Liu, W. Sun, J. Wang, Y. Wang, Y. Wu, Y. Wu, D. Yang, H. Yang, Y. Yang, Z. Yang, A. Yin, R. Yuan, Y. Zhang, and Z. Zhou, “Kimi-audio technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2504.18425
- [71] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR,

- 2021, pp. 8748–8763.

[72] J. Li, D. Li, C. Xiong, and S. C. H. Hoi, “BLIP: bootstrapping language-image pre-training for unified vision-language understanding and generation,” in International Conference on Machine Learning, ICML

- 2022, 17-23 July 2022, Baltimore, Maryland, USA, ser. Proceedings of Machine Learning Research, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, Eds., vol. 162. PMLR, 2022, pp. 12888–12900.

