arXiv:2505.14648v1[cs.SD]20May2025

# Vox-Profile: A Speech Foundation Model Benchmark for Characterizing Diverse Speaker and Speech Traits

Tiantian Feng1 Jihwan Lee1 Anfeng Xu1 Yoonjeong Lee1 Thanathai Lertpetchpun1 Xuan Shi1 Helin Wang2 Thomas Thebaud2 Laureano Moro-Velazquez2 Dani Byrd1 Najim Dehak2 Shrikanth Narayanan1 1 University of Southern California 2Johns Hopkins University tiantiaf@usc.edu

## Abstract

We introduce Vox-Profile, a comprehensive benchmark to characterize rich speaker and speech traits using speech foundation models. Unlike existing works that focus on a single dimension of speaker traits, Vox-Profile provides holistic and multi-dimensional profiles that reflect both static speaker traits (e.g., age, sex, accent) and dynamic speech traits (e.g., emotion, speech flow). This benchmark is grounded in speech science and linguistics, developed with domain experts to accurately index speaker and speech characteristics. We report benchmark experiments using over 15 publicly available speech datasets and several widely used speech foundation models that target various static and dynamic speaker and speech properties. In addition to benchmark experiments, we showcase several downstream applications supported by Vox-Profile. First, we show that Vox-Profile can augment existing speech recognition datasets to analyze ASR performance variability. Vox-Profile is also used as a tool to evaluate the performance of speech generation systems. Finally, we assess the quality of our automated profiles through comparison with human evaluation and show convergent validity. Vox-Profile is publicly available at: https://github.com/tiantiaf0627/vox-profile-release.

## 1 Introduction

Even brief intervals of voice carry a wealth of information about a speaker, including their age, sex, accent, emotion, and their physical environment and social context. The ability to accurately predict these attributes from speech has significant value in advancing a wide range of speech technologies. For example, conversational digital assistants like Alexa or Siri can adapt responses based on a user’s perceived emotional state. Automatic speech recognition systems (ASR) [1] can improve transcription accuracy by integrating different speaking accents. Despite these promising applications, much of the progress in speech modeling has been centered on conventional tasks such as ASR for transcribing spoken language, speaker diarization [2] for tracking who is speaking when, and speech enhancement [3] for improving audio speech quality. In contrast, there is limited work in systematically modeling and predicting diverse speaker and speech traits.

Much of the existing research characterizing speakers has focused on learning speaker embeddings, such as x-vectors [4], which capture the unique vocal traits of individual speakers. These embeddings emerged as foundational components in many applications, most notably in speaker recognition and verification systems. While significant progress has been made in using speaker embeddings to capture identity-specific vocal features, relatively little work has focused on comprehensive modeling of other human-interpretable speaker traits, such as accent and voice quality (or timbre). Recent work has shown that robust inclusion of such traits holds great promise for downstream applications, like style-prompted speech generation models. For example, the ParaSpeechCaps [5] dataset provides

Preprint.

- Table 1: Comparison of Vox-Profile with existing literature in modeling speaker and speech traits.

Categorized Arousal Voice Speech Speech Emotion Valence Quality Flow Expressiveness

Age Sex Accent

VoxCeleb Enrichment [12] ✓ ✓ ✗ ✗ ✗ ✗ ✗ ✗ CommonAccent [6] ✗ ✗ ✓ ✗ ✗ ✗ ✗ ✗ GLOBE [13] ✓ ✓ ✓ ✗ ✗ ✗ ✗ ✗ ParaSpeechCaps [5] ✗ ✓ ✓ ✓ ✗ ✓ ✗ ✓

Vox-Profile (Ours) ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

large-scale human annotations of speaker traits to facilitate the development of high-performing text-to-speech (TTS) models capable of generating speech with different speaking styles.

Although several studies have explored the modeling of speaker traits such as accents [6] or age [7], each study designs its unique taxonomy in the classification, leading to inconsistencies across the literature. As a result, a unified framework for defining, categorizing, and modeling different speaker or speech traits is lacking. In this work, we propose Vox-Profile, one of the first benchmarking efforts that systematically evaluate rich multi-dimensional speaker and speech traits from Englishspeaking voices. Specifically, Vox-Profile experiments with over 15 publicly available datasets to predict static traits (e.g., age, sex, accent, and voice quality) and dynamic traits (e.g., speech emotion, speech flow) in different recording conditions and elicitation settings (e.g., read, spontaneous, and conversational speech). Our benchmark covers popular speech models, including HuBERT [8], WavLM [9], ECAPA-TDNN [10], and Whisper family [11].

Our proposed Vox-Profile benchmark exhibits multifold merits compared to prior works: (1) Unlike previous work on modeling limited speaker or speech traits, Vox-Profile provides a holistic characterization on an extensive list of traits from speech, including, age, sex, accents, emotions, voice quality, speech flow, and expressiveness. (2) Instead of solely focusing on computational modeling, Vox-Profile integrates linguistically informed taxonomy to better address the inherently subjective nature of these traits. (3) The broad utility of Vox-Profile is established through investigating several potential applications including analyzing speech model performance, evaluating speech generation systems, and automatically tagging speaking styles. Both machine learning and human evaluations indicate that Vox-Profile offers reliable estimations of speaker and speech traits.

## 2 Related Works

Static Traits Modeling. We summarize several related representative works in Table 1. One of the most studied tasks in speaker trait prediction is estimating age and sex from speech. Burkhardt et al.[7] introduced a benchmark framework that uses several publicly available datasets for age and sex prediction based on the transformer. In a related effort, Yang et al.[14] investigated the effectiveness of WavLM embeddings for age and sex prediction, showing that self-supervised speech representation can achieve higher performances compared to i-vector features [15]. Apart from modeling age and sex attributes from speech, there has been growing interest in predicting speaker accents. For example, CommonAccent [6] introduced a benchmark for accent classification using samples from Mozilla Common Voice [16]. It reported strong performance in recognizing regional varieties of English, such as American, Canadian and British English, using the ECAPA-TDNN speaker embedding model. GLOBE [13] is a similar effort that develops classifiers using HuBERT pre-trained models on Common Voice to predict accent as well as sex and age.

Dynamic Traits Modeling Several works have also focused on modeling and characterizing dynamic speaker traits, such as speech emotion and fluency. One of the early efforts in supporting speech emotion recognition (SER) is the release of the IEMOCAP dataset [17], which provides annotated recordings of acted dialogues with fine-grained emotional labels to develop emotion recognition systems. Several datasets have since been designed to support more naturalistic emotion modeling, including the MSP-Podcast [18], MELD [19], CREMA-D [20], and CMU-MOSEI [21] datasets. Recent advances in speech foundation models have also led to the development of highperforming SER systems. Leveraging pre-trained representations from models such as WavLM and Whisper, researchers have demonstrated improvements over traditional SER approaches [22, 23]. Likewise, a growing body of research focused on predicting speech fluency to develop robust language assessment systems. For example, SEP-28K [24] presents a notable effort to release large-scale speech data annotated for disfluencies, including conditions such as sound blocks.

- Table 2: The taxonomy of speech traits in Vox-Profile benchmark. The green, blue, and violet indicate speaker accents from North America, the British Isles, and other regions or language backgrounds, respectively. For voice quality, the colors green, blue, violet, olive, and orange represent dimensions of speaker pitch, rhythm, clarity, voice texture, and volume defined by ParaSpeechCaps. Finally, the blue color denotes labels related to speech disfluencies in the speech flow category.

Category Traits Labeling Scheme Datasets

Static Traits

Speaker Sex Male; Female TIMIT[26]; VoxCeleb[27]; CommonVoice[16] Speaker Age

Young adults (<30 Years); Adults (30-60 years);

TIMIT[26]; VoxCeleb[27]; CommonVoice[16] Senior adults (>60 years)

Speaker Accent

North America, English, Welsh, Scottish CommonVoice[16]; EdAcc [28]; CSLU-FAE Northern Irish, Irish, Germanic, Romance British Isles [29]; L2-ARCTIC [30]; TIMIT [26]

Slavic, Semitic, Oceania, South Africa VoxPopuli [31]; Fair-Speech [32]; ESLTTS [33] East Asia, Southeast Asia, South Asia, Other Hispanic-English [34]; Nigerian-English [35]

Voice Quality

Shrill, Nasal, Deep; Singsong, Pitchy, Flowing,

ParaSpeechCaps [5]

Monotone, Staccato, Punctuated, Enunciated, Hesitant; Crisp, Slurred, Lisp, Stammering; Silky, Husky, Raspy, Guttural, Vocal-fry; Booming, Authoritative, Loud, Hushed, Soft

Dynamic Traits

Categorical Neutral, Happy, Sad, Angry, Contempt

MSP-Podcast [18] Emotion Fear, Disgust, Surprise, Other

Arousal/Valence 0-1(Calm->Active) / 0-1(Negative->Positive) MSP-Podcast [18] Speech Flow

Fluent, Disfluent (Prolongation, Word Repetition

Sep-28K [24], FluencyBank [36] Sound Repetition, Block, Interjection)

Speech Animated, Laughing, Passive

ParaSpeechCaps [5] Expressiveness Whispered, Enunciated

- 3 Taxonomy of Speaker and Speech Traits

Unlike prior benchmarks focused on general speech modeling tasks [25], Vox-Profile offers a structured benchmarking framework for systematically predicting a broad range of speaker and speech traits. In developing this automated evaluation pipeline, we observed considerable inconsistency across existing studies in how speech traits are categorized, defined, and modeled. One of the closest prior efforts is ParaSpeechCaps [5], which proposes a taxonomy that models “intrinsic” (e.g., accent) and “situational” (e.g., emotion) traits. However, its categorization remains underspecified and difficult to reconcile with conventions based on empirical foundations in speech science literature. For example, while ParaSpeechCaps includes a broad set of emotional descriptors, these labels deviate from standard categorical and dimensional emotion frameworks commonly used in speech and language research, creating challenges when aligning datasets or comparing across studies.

Many existing approaches in speech trait modeling lack grounding in speech science and linguistics, which often leads to ambiguous or ill-posed problem formulations. A prominent example is speaker age modeling, which is frequently treated as a regression task targeting a speaker’s exact chronological age [37, 38]. However, this framing overlooks numerous confounding variables such as biological sex, vocal anatomy, and speaking habits that affect vocal aging. Crucially, it misaligns with how age is typically perceived (and experienced) from speech: human listeners estimate age in broad intervals rather than in precise years [39, 40]. Similarly, accent classification is often based on a speaker’s nationality, as in ParaSpeechCaps, even though nationality is not reliably inferable from speech alone, particularly in multilingual or diasporic contexts [41, 42]. To address these challenges, we introduce a linguistically principled taxonomy in Vox-Profile, designed to be generalizable across diverse speech datasets and conditions.

We define two broad categories of speech traits to structure the benchmark: (1) Static traits capture relatively stable characteristics of the speaker: age, sex, accent, and voice quality. (2) Dynamic traits reflect context-dependent aspects of speech that can vary across situations, including emotion, speech expressiveness, and speech flow. The full taxonomy and the labeling scheme are summarized in Table 2, with the description of all datasets used in this benchmark and data splits for the benchmark experiment detailed in Appendix A.

### 3.1 Static Traits

Speaker Sex Biological sex as reflected in voice is relatively well-defined in the literature. We follow standard practice by modeling it as a binary classification (male vs. female), while recognizing that,

in practice, acoustic parameters of voice quality do not always map cleanly onto binary categories, particularly in the case of young children or speakers with atypical vocal profiles.

Speaker Age As discussed earlier, many existing studies treat age estimation as a regression task, aiming to predict a speaker’s exact chronological age from speech [37, 38]. We argue that this formulation is ill-posed: perceived vocal age is shaped by multiple interacting factors and is more naturally categorized in intervals than estimated precisely. Instead, we propose classifying speakers into three broad age groups: young adults (18-30 years), adults (30-60 years), and senior adults (>60 years). Speech science research [43] indicates that the vocal characteristics typically stabilize in early adulthood, with relatively consistent voice characteristics across the adult decades. In contrast, age-related vocal changes, linked to hormonal, physiological, and health-related factors, become increasingly salient from around age 60 years onward.

Speaker Accent Accent refers to a distinctive way of pronouncing a language, typically shaped by a speaker’s regional origin and linguistic background. However, the classification of accent has varied widely across the literature with inconsistent labeling schemes and taxonomic assumptions [6, 13, 44]. In this work, we introduce a unified and scalable taxonomy for accent classification, designed to integrate multiple mainstream English-accent datasets under a coherent framework. Focusing on English-speaking populations, we first organize accents into three broad regional groups: North American, British Isles, and other regions or other language backgrounds (e.g., Oceania, South Asia, and Africa). Within the British Isles, we further distinguish specific regional varieties: English (England), Scottish, Northern Irish, Welsh, and Irish. To better capture systematic cross-linguistic influences on English pronunciation, we also group certain English accents by the language family of the speakers’ first language (L1). This includes Germanic (e.g., German, Dutch), Slavic (e.g., Russian, Polish), Romance (e.g., Spanish, French, Italian), and Semitic (e.g., Arabic and Hebrew) language backgrounds. This typological grouping reflects the fact that related L1s often give rise to similar segmental and prosodic transfer patterns in second-language (L2) English—patterns that may not align cleanly with national or geographic borders. Asian accents are organized by region, East Asia, South Asia, and Southeast Asia, reflecting common typological and areal features. Additional categories include Oceania and South Africa, representing other major English speaking regions. Due to underrepresenation in existing datasets, some language backgrounds cannot be reliably classified into these categories. For example, accents associated with Uralic languages as L1s (e.g., Finnish) or with regions of Africa outside South Africa are grouped under the general “Other” label. This taxonomy enables the integration of 11 publicly available datasets, each with distinct accent labeling conventions, into a unified benchmark for training and evaluating accent classification models.

Voice Quality Only few publicly available datasets offer detailed annotations suitable for modeling voice quality. For this benchmark, we adopt ParaSpeechCaps [5], the dataset with the most extensive human-annotated labels currently available in this category. It defines voice quality across five perceptual dimensions: pitch, voice texture, volume, clarity, and rhythm, each reflecting a distinct aspect of how a speaker’s voice is perceived beyond segmental content. The specific labels associated with each dimension are provided in Table 2.

### 3.2 Dynamic Traits

Categorical Emotion There is an extensive list of SER datasets, including IEMOCAP [17], MELD [19], CMU-MOSEI [21], and CREMA-D [20], each employing distinct emotion labeling schemes. Among them, we find the MSP-Podcast dataset [18] provides the largest number of speech emotion samples recorded in naturalistic settings. Therefore, we adopt its emotion labeling scheme, including categories neutral, happy, sad, angry, fear, disgust, contempt, surprise, and others. Arousal/Valence Arousal and valence are also widely used to describe emotional states in speech [45]. We define both arousal and valence as continuous variables scaled from 0 to 1. Arousal represents the intensity of the emotion, with higher scores indicating greater activation or energy. Valence reflects emotional polarity, with lower scores corresponding to more negative affective states (e.g., sadness) and higher values to more positive (e.g., happiness).

Speech Flow Speech flow has gained growing attention in recent years as a dimension of fluency in spoken language [36, 36]. One of the most notable resources is the Sep-28K dataset [24], which annotates disfluencies at scale. Following this foundation, we categorize speech flow in two: fluent and disfluent. Disfluency is further defined to include prolongations, word repetitions, sound repetitions, blocks, and interjections (including filler words or hesitation sounds).

Speech Expressiveness The last dynamic trait we aim to model is speech expressiveness. Similar to voice quality, the datasets available are scarce in this field. We adopt the definitions provided

###### 1 Speech Model Performance Analysis

###### Vox-Profile

- Group1
- Group2 ......

ASR Model

Speech Transcription

Speaker Accent

Speaker Sex

|Vox-Profile|
|---|

###### Group1/2/...

Speaker Age

2 Speech Generation Evaluation

Speech Generation Target Output

|Vox-Profile|
|---|

Speech Data

|Male British|
|---|

|Male Likely British|
|---|

Speech Emotion

3 Speaking-style Prompt Generation

Vox-Profile (Automated)

Can I uh uh ask you a question?

An older woman with likely an Oceania (such as Australian) accent speaks fluently and at a normal speed, with an angry, flowing tone and a normal pitch.

|Vox-Profile|
|---|

[Figure 1]

Speech Flow

Attributes Value Confidence Level Speaker Accent

Oceania (e.g. Australian)

Likely Speaker Sex Female Confident Speaker Age Senior Adult Confident Speech Emotion Angry Likely Speech Flow Fluent Confident Voice Quality Flowing Maybe Expressiveness None NA

Paraspeechcaps (Human)

Voice Quality

A woman with an Australian accent speaks at a normal speed, with a flowing tone and a normal pitch.

Speech Expressiveness

- Figure 1: Overview of the proposed Vox-Profile Benchmark and its applications. We highlight three primary use cases: (1) speech model (such as ASR) performance analysis, (2) automated speech generation evaluation, and (3) automated speaking style tagging.

in the ParaSpeechCaps [5] dataset to guide our modeling. Specifically, we categorize speech expressiveness into five types: animated, laughing, passive, whispered, and enunciated.

## 4 Vox-Profile Benchmark

As shown in Figure 1, we propose a speaker and speech trait modeling benchmark, Vox-Profile, to model a wide range of traits defined above. Our Vox-Profile benchmark can be applied to augment metadata for analyzing speech model performances, evaluating speech generation systems, and curating synthetic speaking style descriptions.

Static Traits We examine the aforementioned speech foundation models to predict different static traits from speech. (1) For age and sex classification, we experiment with the multitask learning approach to jointly predict both traits. This is motivated by the existing studies that show changes in voice characteristics across age can differ significantly between males and females. Modeling them jointly is expected to allow the system to better capture these interactions and improve classification performance. We apply the concordance correlation coefficient (CCC) [46] loss to train age prediction systems. Instead of reporting regression results from absolute age values, we map ages into the three predefined age groups and report the corresponding classification performance. (2) We develop accent classification systems in two levels of categorization: broad and narrow. The former targets broader accent categories, while the latter predicts 16 narrow accent labels based on different regional and language backgrounds. (3) We formulate voice quality prediction as a multi-label classification task since each speech sample can be associated with multiple voice qualities.

Dynamic Traits (1) Our categorical emotion and dimensional emotion (arousal/valence) modeling systems are built upon our recent top-ranked solutions in the Speech Emotion Recognition in Naturalistic Conditions Challenge at Interspeech 2025 (IS25-SER challenge) 1. Specifically, the categorical emotion system applies the KL divergence loss to predict the distribution of categorical emotion annotations. For arousal and valence modeling, we use the sigmoid function to map arousal and valence values onto a normalized scale between 0 and 1. (2) For speech flow modeling, we propose a multitask learning framework of 2 classification heads with one to predict whether the

1https://lab-msp.com/MSP-Podcast_Competition/IS2025

Table 3: Comparison of different speech foundation models in predicting static traits. Overall, the results show that Whisper Large achieves the overall best performance, while the WavLM-based model shows moderate strength in accent prediction.

Speaker Age Speaker Sex Broad Accent Narrow Accent Voice Quality Acc F1 Acc F1 Acc F1 Acc F1 F1

Speaker Model

↰

ECAPA-TDNN 59.3 0.469 96.9 0.962 74.6 0.638 42.3 0.363 0.706 Self-Supervised

↰

HuBERT Large 57.1 0.501 96.7 0.960 85.4 0.841 49.1 0.443 0.692 ↰

WavLM Large 67.6 0.624 97.7 0.971 92.9 0.884 65.7 0.617 0.703 Whisper Family

↰

Whisper Tiny 58.1 0.520 94.3 0.934 89.1 0.831 54.1 0.484 0.587 ↰

Whisper Small 65.8 0.558 96.4 0.958 90.9 0.863 70.4 0.663 0.737 ↰

Whisper Large 69.5 0.643 98.0 0.975 92.8 0.889 75.8 0.724 0.704

speech is fluent or disfluent, the other to classify the specific types of disfluencies present. We highlight that classifying types of disfluencies is a multilabel task, as multiple breaks can co-occur in a single speech utterance. (3) We formulate speech expressiveness as a multilabel classification task. Modeling Approach In Vox-Profile, we evaluate several widely studied speech foundation models, including the HuBERT [8], WavLM [47], ECAPA-TDNN [10], and Whisper [11] family with details reported in Appendix B. Since speaker embedding models are primarily related to static speaker traits, we exclude the ECAPA-TDNN model for predicting dynamic speech traits. For all experiments, we use the downstream model architecture developed in [48], where many existing works have shown that this simple architecture achieves strong performance in a wide range of downstream speech modeling tasks. Specifically, we first apply weighted averaging to combine hidden representations of both the convolutional and Transformer [49] encoder layers. The aggregated output is then processed through 1D-pointwise convolutional layers. Finally, we average the convolutional outputs to obtain the final embeddings, which are passed through fully connected layers for classification or regression. In addition to this approach, we experiment with finetuning using LoRa [50], an effective method for adapting speech foundation models. Details of the training hyperparameters, such as learning rates and training epochs, and training resource, are provided in Appendix C and D.

Evaluation Metrics We compare different models in predicting speaker and speech traits including sex, age, accent, categorical emotions, and speech fluency/disfluency, using both accuracy and macro F1 scores. Additionally, we report the macro F1 score when comparing models predicting speech voice quality, specific speech disfluencies, and speech expressiveness. Finally, we assess the predictions of arousal and valence using the concordance correlation coefficient (CCC) [46] score.

## 5 Benchmark Performance

In this section, we present a detailed comparison of benchmark performance. We report the topperforming scores for each model in Table 3 and Table 4. We then analyze the impact of model ensembling, followed by a comparison with existing works from the literature.

Static Traits We compare the performance of different speech foundation models in predicting static speaker traits shown in Table 3. The results indicate a positive correlation between model parameter size and performance, with larger models generally achieving better performance in speaker trait classification. Moreover, although the ECAPA-TDNN model is optimized for speaker recognition tasks, finetuning ECAPA-TDNN still yields lower performance than self-supervised or supervised speech foundation models, particularly in accent and age classification. We observe that sex classification is relatively straightforward, with most models achieving around 95% accuracy. Moreover, classifying between broad accent groups (North American, British Isles, and other regional or language backgrounds) yields strong performance, with macro F1 scores exceeding 0.80 in most models. However, narrow accent prediction remains challenging with only Whisper Large achieving a 0.724 macro F1 score in this task.

Dynamic Traits We present results of dynamic traits prediction in Table 4. Consistent with findings for static trait prediction, we find that larger models perform better, with Whisper Large and WavLM Large achieving the highest scores across most tasks. Moreover, detecting disfluency in speech flow is relatively straightforward, with multiple models achieving over 80% accuracy. In contrast, the benchmark results indicate that SER remains challenging given the relatively low performance

Table 4: Comparison of different models in predicting dynamic speech traits.

Speech Emotion Arousal Valence Fluency Disfluency Type Speech Expressivenss

F1 CCC CCC Acc F1 F1 F1 Self-Supervised

↰

HuBERT Large 0.395 0.585 0.650 75.5 0.755 0.627 0.565

↰

WavLM Large 0.406 0.585 0.665 80.7 0.806 0.691 0.689 Whisper Family

↰

Whisper Tiny 0.337 0.545 0.522 76.4 0.762 0.639 0.484

↰

Whisper Small 0.387 0.583 0.623 80.4 0.804 0.653 0.751

↰

Whisper Large 0.416 0.588 0.645 80.6 0.805 0.670 0.848

- Table 5: Experiments to compare Vox-Profile with existing work and impact of model ensembles.

[Figure 2]

Model Acc F1 Model Acc F1 Accent (VCTK) Accent (British Isles)

↰

↰

CommonAccent 66.3 0.379

CommonAccent 69.4 0.581

↰

↰

Vox-Profile 83.9 0.770 Speaker Age (VoxCeleb) Speaker Sex (TIMIT)

Vox-Profile 72.6 0.423

↰

↰

Wav2Vec2-based[7] 74.9 0.717

Wav2Vec2-based[7] 99.4 0.993

↰

↰

Vox-Profile 99.5 0.995 Model F1 Model CCC Categorical Emotion Arousal / Valence

Vox-Profile 75.6 0.716

↰

↰

IS25-SAIL SER 0.428

IS25-SAIL SER 0.642 / 0.683

↰

↰

Vox-Profile 0.392

Vox-Profile 0.623 / 0.649

(b) Comparing Vox-Profile benchmark performance with the existing literature.

(a) Comparing model ensembles with single model in Vox-Profile.

across most experiments. We highlight that most models can only reach macro F1 scores around 0.4 (including our models ranked as the top-performing solutions in the IS25-SER challenge).

Model Ensemble We investigate whether ensembling top-performing models further improves the performance. The results in Table 5a show that naïve ensembles of the two top-performing models consistently yield moderate improvements compared to individual models. In particular, we observe substantial gains in modeling static traits such as speech voice quality, age, and sex.

Comparing Vox-Profile Benchmark with Existing Literature We compare the benchmark performance of Vox-Profile to several prior works. Regarding the accent classification, we compare our model against a previous approach CommonAccent [6] on the VCTK [51] and British Isles [52] speaker datasets. To ensure a fair comparison, we mapped the label schema used in CommonAccent to align with our accent labeling. As shown in Table 5b, our benchmark model consistently outperforms CommonAccent in both datasets. Moreover, we compare our speaker age and sex classification model with the approach presented in [7], using the test sets of the VoxCeleb and TIMIT datasets, respectively. The results indicate that Vox-Profile achieves strong performance in both age and sex prediction tasks. Finally, we compare our SER models with our top-performing system, SAIL-SER, which achieves top-1 and top-2 scores in the categorical and dimensional SER in the IS25-SER challenge. Note that SAIL-SER utilizes both text and speech modalities, while Vox-Profile relies solely on the speech input. The absence of text modality in Vox-Profile explains the performance gap; however, it still achieves comparable performance despite this limitation.

## 6 Enabling Versatile Speech Applications with Vox-Profile

### 6.1 Speech Model Performance Analysis

We show how Vox-Profile facilitates the analysis of speech model performance. Specifically, we generate speaker and speech traits for existing datasets and investigate whether these generated labels can lead to the same insights as using the ground truth trait information in analyzing the speech model performances. For this experiment, we focus on ASR as a representative task in speech modeling.

ASR Inference We augment accent and emotion labels for the VCTK and MSP-Podcast datasets. For MSP-Podcast, we provide emotion labels only for the dev set, following the 4 most frequently represented emotions: neutral, sadness, happiness, and anger. Only samples with confident predictions are used for computing ASR performance in the Vox-Profile experiment. To mitigate the potential artifacts from using the same model architecture for ASR and speech trait prediction, we use the Wav2Vec 2.0 Robust [53] to infer the transcript. In our analysis, we exclude the ASR models, which the VCTK dataset is trained on, such as Parakeet [54] or Canary [54], for fair comparison.

[Figure 3]

[Figure 4]

- Figure 2: ASR performance trends, grouped by ground truth and predicted labels by Vox-Profile. We measure WER, stratified by accent and emotion labels. We observe similar performance trends between the predicted and ground truth trait labels.

- Table 6: Utilizing Vox-Profile to evaluate the accent conversion performance of FreeVC and VALLE-X, by measuring cosine similarity (left table) and prediction scores (right table).

[Figure 5]

Similarity Between Source ⇒ Reference (Source, Output) ↓ (Reference, Output) ↑ British Isles ⇒ North America

↰

FreeVC 0.874 ± 0.046 0.319 ± 0.071

↰

VALLE-X 0.512 ± 0.103 0.543 ± 0.080 British Isles ⇒ Other

↰

FreeVC 0.818 ± 0.064 0.552 ± 0.089

↰

VALLE-X 0.497 ± 0.102 0.908 ± 0.055 North America ⇒ British Isles

↰

FreeVC 0.846 ± 0.046 0.503 ± 0.086

↰

VALLE-X 0.559 ± 0.082 0.595 ± 0.090 Other ⇒ British Isles

↰

FreeVC 0.937 ± 0.036 0.658 ± 0.090

↰

VALLE-X 0.960 ± 0.030 0.552 ± 0.097

Findings Figure 2 shows that ASR performance trends based on the ground truth trait align closely with those using the predicted labels from Vox-Profile. In the VCTK dataset, North American speakers consistently have lower WER scores than speakers from the British Isles and those with “Other” accents, regardless of whether ground truth or synthetic labels are used (p < 0.01 in both conditions). Moreover, whether using ground truth or predicted emotion labels, in this particular dataset, speech expressing sadness is associated with relatively lower WER scores, whereas happy speech tends to result in higher WER scores. Overall, these findings suggest that Vox-Profile can generate reliable synthetic metadata to facilitate in-depth analysis of speech model performance.

### 6.2 Automated Evaluation Tool for Speech Generation Tasks

We demonstrate the utility of Vox-Profile as an evaluation tool for speech generation tasks by comparing two representative models: FreeVC [55] and VALLE-X [56]. FreeVC is selected as a representative of textless voice conversion models that operate in the latent space, aiming to transform source speech to match the voice identity of a reference speaker. VALLE-X is chosen as a representative of voice cloning approaches based on neural codecs, utilizing a concatenated pipeline of ASR and TTS conditioned on the reference speech. It has been reported that textless models like FreeVC often struggle to accurately reflect the accent of the reference speaker [57]. On the other hand, VALLE-X, by incorporating an intermediate text representation through ASR, is better equipped to preserve accentual features in the generated speech.

To assess the ability of these models to reflect the accent of the reference speech, we randomly selected source-reference (target) pairs from the VCTK dataset [51]. For each pair, we synthesized speech samples and evaluated whether they more closely resembled the source or reference speaker in terms of accent, by measuring cosine similarities and accent prediction scores. Detailed configurations for FreeVC, VALLE-X, and the selected test speakers are provided in Appendix E.

As shown in Table 6, the accent prediction scores and the cosine similarity for the synthesized samples from FreeVC suggest greater similarity to the source speaker’s accent than to the reference speaker. In contrast, the scores for VALLE-X indicate closer alignment with the reference speaker’s accent in most conditions. These findings are consistent with previous studies, which report that FreeVC has limited capability in replicating the accentual features of the reference speaker, whereas VALLE-X, due to its intermediate text-based representation, more effectively preserves these features [56, 57].

### 6.3 Generating Synthetic Speaking Style Prompt

We further apply the Vox-Profile as a tool for generating synthetic speaker and speech traits to create speaking style prompts. Previous efforts to create synthetic speaking style prompts, such as

ParlerTTS [58], rely on heuristic-based categorizations of acoustic features, including fundamental frequency (F0) and signal-to-noise ratio (SNR). More recently, ParaSpeechCaps introduced a largescale, human-annotated dataset containing multiple speech traits, such as voice quality and speaker accent. In contrast to these prior works, Vox-Profile provides a more extensive and varied set of traits, including speech flow, arousal, valence, and speaker age. Moreover, computational models of Vox-Profile output probabilistic predictions for each trait, enabling more nuanced and confidencesensitive descriptions. For example, a Scottish accent prediction with a probability of 0.9 can be described as having a distinct Scottish accent, while a probability of 0.5 might be described as a likely Scottish accent. This probabilistic framing supports the generation of speech descriptions that better reflect the para-linguistic uncertainties with which humans naturally perceive and interpret speech for information extraneous to linguistic messaging. To evaluate the robustness and versatility of the style prompts generated by Vox-Profile, we propose a comparative analysis between human-annotated and machine-generated style prompts using the ParaSpeechCaps dataset. The detailed human evaluation procedure is described in Appendix G.

Prompt Generation Since our voice quality and expressiveness models are trained on the training set of the ParaSpeechCaps, we use speech samples from the test set for this experiment. Specifically, we select 30 speech samples stratified by speaker sex and accent (broader accent classes). For each selected sample, we use Vox-Profile to infer both static and dynamic traits. To ensure reliability, we keep the predictions with high and moderate confidence levels, labeling them as "distinct/very likely" and "likely," respectively. Traits with low confidence are excluded from prompt generation. Subsequently, we use GPT-4.1 [59] to generate the speaking style prompt using the synthetic tags. To ensure a fair comparison, we also apply the GPT-4.1 to generate the speaking style prompt using the ParaSpeechCaps tags. The prompt generation follows the template used in ParaSpeechCaps and the detailed description of confidence levels and prompt designs is in Appendix F.

Human-Evaluation Results Human-evaluation results comparing synthetic speaking style prompts from Vox-Profile and human-annotated speaking style prompts from ParaSpeechCaps are presented in Figure 3. Overall, the results suggest that this group of human raters shows similar preference levels for both synthetic and human-annotated speaking style prompts. Specifically, they favor the emotion, age, and speech flow descriptions generated by Vox-Profile over those from ParaSpeechCaps. For accent descriptions, the human raters reach similar levels of agreement between ParaSpeechCaps and Vox-Profile in more than 50% unique evaluations. However, ground truth accent labels from ParaSpeechCaps still outperform Vox-Profile in accurately describing speaker accents, highlighting that accent prediction remains challenging in speaker trait modeling. In summary, the human evaluation results provide evidence establishing that Vox-Profile is effective in creating synthetic speaking style prompts that closely match human-labeled data.

###### Human Evaluation of Speech Style-Prompt between Vox-Profile and ParaSpeechCaps

Both are Similar Vox-Profile is Better ParaSpeechCaps is Better

100%

50%

0%

Style-Prompt Speaker Sex Speaker Age Speaker Accent Voice Quality Speech Flow (Fluency)

Speech Emotion

- Figure 3: Comparing human evaluation results between synthetic speaking style prompts created by Vox-Profile and human-annotated prompts from ParaSpeechCaps. Human raters provided their preferences across overall prompt quality, sex, age, accent, voice quality, fluency, and emotion.

## 7 Conclusion and Limitations

We propose Vox-Profile, a comprehensive benchmark for modeling speaker and speech characteristics. This benchmark includes large-scale experiments across more than 15 public speech datasets, evaluated using widely adopted speech foundation models, and delivers a suite of high-performing trait prediction models. Our results show that Vox-Profile supports a broad range of applications, including speech model performance analysis, generated speech evaluation, and high-quality speaker and style prompts generation. Our next steps include testing our benchmark in multilingual conditions, and adding the prediction of multilingual speech attributes like languages identification and code-switching to improve accessibility for more diverse language communities. Moreover, while Vox-Profile currently uses speech models pre-trained mostly for ASR, we plan to expand to alternative architectures such as Emotion2Vec [60] and Speech-LLMs [47, 61], which we expect will broaden the benchmark’s applicability to affective and conversational speech technologies.

## References

- [1] Rohit Prabhavalkar, Takaaki Hori, Tara N Sainath, Ralf Schlüter, and Shinji Watanabe. End-toend speech recognition: A survey. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32:325–351, 2023.
- [2] Tae Jin Park, Naoyuki Kanda, Dimitrios Dimitriadis, Kyu J Han, Shinji Watanabe, and Shrikanth Narayanan. A review of speaker diarization: Recent advances with deep learning. Computer Speech & Language, 72:101317, 2022.
- [3] Daniel Michelsanti, Zheng-Hua Tan, Shi-Xiong Zhang, Yong Xu, Meng Yu, Dong Yu, and Jesper Jensen. An overview of deep-learning-based audio-visual speech enhancement and separation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:1368–1396, 2021.
- [4] David Snyder, Daniel Garcia-Romero, Gregory Sell, Daniel Povey, and Sanjeev Khudanpur. Xvectors: Robust dnn embeddings for speaker recognition. In 2018 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 5329–5333. IEEE, 2018.
- [5] Anuj Diwan, Zhisheng Zheng, David Harwath, and Eunsol Choi. Scaling rich style-prompted text-to-speech datasets. arXiv preprint arXiv:2503.04713, 2025.
- [6] Juan Zuluaga-Gomez, Sara Ahmed, Danielius Visockas, and Cem Subakan. Commonaccent: Exploring large acoustic pretrained models for accent classification based on common voice. In Proc. Interspeech 2023, pages 5291–5295, 2023.
- [7] Felix Burkhardt, Johannes Wagner, Hagen Wierstorf, Florian Eyben, and Björn Schuller. Speechbased age and gender prediction with transformers. In Speech Communication; 15th ITG Conference, pages 46–50. VDE, 2023.
- [8] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM transactions on audio, speech, and language processing, 29:3451–3460, 2021.
- [9] Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al. Wavlm: Large-scale self-supervised pretraining for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing, 16(6):1505–1518, 2022.
- [10] Brecht Desplanques, Jenthe Thienpondt, and Kris Demuynck. Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification. In 21st Annual conference of the International Speech Communication Association (INTERSPEECH 2020), pages 3830–3834. International Speech Communication Association (ISCA), 2020.
- [11] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning, pages 28492–28518. PMLR, 2023.
- [12] Khaled Hechmi, Trung Ngo Trong, Ville Hautamäki, and Tomi Kinnunen. Voxceleb enrichment for age and gender recognition. In 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 687–693. IEEE, 2021.
- [13] Wenbin Wang, Yang Song, and Sanjay Jha. Globe: A high-quality english corpus with global accents for zero-shot speaker adaptive text-to-speech. In Proc. Interspeech 2024, pages 1365– 1369, 2024.
- [14] Yuchen Yang, Thomas Thebaud, and Najim Dehak. Demographic attributes prediction from speech using wavlm embeddings. In 2025 59th Annual Conference on Information Sciences and Systems (CISS), pages 1–6. IEEE, 2025.
- [15] Najim Dehak, Patrick J Kenny, Réda Dehak, Pierre Dumouchel, and Pierre Ouellet. Front-end factor analysis for speaker verification. IEEE Transactions on Audio, Speech, and Language Processing, 19(4):788–798, 2010.

- [16] Rosana Ardila, Megan Branson, Kelly Davis, Michael Kohler, Josh Meyer, Michael Henretty, Reuben Morais, Lindsay Saunders, Francis Tyers, and Gregor Weber. Common voice: A massively-multilingual speech corpus. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 4218–4222, 2020.
- [17] Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N Chang, Sungbok Lee, and Shrikanth S Narayanan. Iemocap: Interactive emotional dyadic motion capture database. Language resources and evaluation, 42:335–359, 2008.
- [18] Reza Lotfian and Carlos Busso. Building naturalistic emotionally balanced speech corpus by retrieving emotional speech from existing podcast recordings. IEEE Transactions on Affective Computing, 10(4):471–483, 2017.
- [19] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. Meld: A multimodal multi-party dataset for emotion recognition in conversations. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 527–536, 2019.
- [20] Houwei Cao, David G Cooper, Michael K Keutmann, Ruben C Gur, Ani Nenkova, and Ragini Verma. Crema-d: Crowd-sourced emotional multimodal actors dataset. IEEE transactions on affective computing, 5(4):377–390, 2014.
- [21] AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. Multimodal language analysis in the wild: Cmu-mosei dataset and interpretable dynamic fusion graph. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2236–2246, 2018.
- [22] Johannes Wagner, Andreas Triantafyllopoulos, Hagen Wierstorf, Maximilian Schmitt, Felix Burkhardt, Florian Eyben, and Björn W Schuller. Dawn of the transformer era in speech emotion recognition: closing the valence gap. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(9):10745–10759, 2023.
- [23] Tiantian Feng and Shrikanth Narayanan. Peft-ser: On the use of parameter efficient transfer learning approaches for speech emotion recognition using pre-trained speech models. In 2023 11th International Conference on Affective Computing and Intelligent Interaction (ACII), pages 1–8. IEEE, 2023.
- [24] Colin Lea, Vikramjit Mitra, Aparna Joshi, Sachin Kajarekar, and Jeffrey P Bigham. Sep-28k: A dataset for stuttering event detection from podcasts with people who stutter. In ICASSP 20212021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6798–6802. IEEE, 2021.
- [25] Shu-wen Yang, Po-Han Chi, Yung-Sung Chuang, Cheng-I Jeff Lai, Kushal Lakhotia, Yist Y Lin, Andy T Liu, Jiatong Shi, Xuankai Chang, Guan-Ting Lin, et al. Superb: Speech processing universal performance benchmark. In Proc. Interspeech 2021, pages 1194–1198, 2021.
- [26] John S Garofolo, Lori F Lamel, William M Fisher, Jonathan G Fiscus, and David S Pallett. Darpa timit acoustic-phonetic continous speech corpus cd-rom. nist speech disc 1-1.1. NASA STI/Recon technical report n, 93:27403, 1993.
- [27] Arsha Nagrani, Joon Son Chung, and Andrew Zisserman. Voxceleb: a large-scale speaker identification dataset. arXiv preprint arXiv:1706.08612, 2017.
- [28] Ramon Sanabria, Nikolay Bogoychev, Nina Markl, Andrea Carmantini, Ondrej Klejch, and Peter Bell. The edinburgh international accents of english corpus: Towards the democratization of english asr. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.
- [29] Isin Demirsahin, Oddur Kjartansson, Alexander Gutkin, and Clara Rivera. Open-source multispeaker corpora of the english accents in the british isles. In Proceedings of the twelfth language resources and evaluation conference, pages 6532–6541, 2020.

- [30] Guanlong Zhao, Sinem Sonsaat, Alif Silpachai, Ivana Lucic, Evgeny Chukharev-Hudilainen, John Levis, and Ricardo Gutierrez-Osuna. L2-arctic: A non-native english speech corpus. In Proc. Interspeech 2018, pages 2783–2787, 2018.
- [31] Changhan Wang, Morgane Riviere, Ann Lee, Anne Wu, Chaitanya Talnikar, Daniel Haziza, Mary Williamson, Juan Pino, and Emmanuel Dupoux. Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 993–1003, 2021.
- [32] Irina-Elena Veliche, Zhuangqun Huang, Vineeth Ayyat Kochaniyan, Fuchun Peng, Ozlem Kalinli, and Michael L Seltzer. Towards measuring fairness in speech recognition: Fair-speech dataset. In Proc. Interspeech 2024, pages 1385–1389, 2024.
- [33] Wenbin Wang, Yang Song, and Sanjay Jha. Usat: A universal speaker-adaptive text-to-speech approach. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.
- [34] William Byrne, Eva Knodt, Jared Bernstein, and Farzhad Emami. Hispanic-english database (ldc2014s05). Linguistic Data Consortium, 2014.
- [35] Crowdsourced high-quality nigerian english speech data set. Open Speech and Language Resources, 2019.
- [36] Nan Bernstein Ratner and Brian MacWhinney. Fluency bank: A new resource for fluency research and practice. Journal of fluency disorders, 56:69–80, 2018.
- [37] Shijing Si, Jianzong Wang, Junqing Peng, and Jing Xiao. Towards speaker age estimation with label distribution learning. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4618–4622. IEEE, 2022.
- [38] Dareen Alharthi, Mahsa Zamani, Bhiksha Raj, and Rita Singh. Tessellated linear model for age prediction from voice. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–4, 2025.
- [39] Susanne Schötz. Perception, analysis and synthesis of speaker age, volume 47. Lund University, 2006.
- [40] Sara Skoog Waller, Mårten Eriksson, and Patrik Sörqvist. Can you hear my age? influences of speech rate and speech spontaneity on estimation of speaker age. Frontiers in psychology, 6:978, 2015.
- [41] Donald L Rubin. Nonlanguage factors affecting undergraduates’ judgments of nonnative english-speaking teaching assistants. Research in Higher education, 33:511–531, 1992.
- [42] Mohammad Ali Humayun, Junaid Shuja, and Pg Emeroylariffion Abas. A review of social background profiling of speakers from speech accents. PeerJ Computer Science, 10:e1984, 2024.
- [43] Ulrich Reubold, Jonathan Harrington, and Felicitas Kleber. Vocal aging effects on f0 and the first formant: A longitudinal analysis in adult speakers. Speech communication, 52(7-8):638–651, 2010.
- [44] Jinzuomu Zhong, Korin Richmond, Zhiba Su, and Siqi Sun. Accentbox: Towards high-fidelity zero-shot accent generation. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025.
- [45] Peter Kuppens, Francis Tuerlinckx, James A Russell, and Lisa Feldman Barrett. The relation between valence and arousal in subjective experience. Psychological bulletin, 139(4):917, 2013.
- [46] I Lawrence and Kuei Lin. A concordance correlation coefficient to evaluate reproducibility. Biometrics, pages 255–268, 1989.

- [47] Shujie Hu, Long Zhou, Shujie Liu, Sanyuan Chen, Lingwei Meng, Hongkun Hao, Jing Pan, Xunying Liu, Jinyu Li, Sunit Sivasankaran, et al. Wavllm: Towards robust and adaptive speech large language model. arXiv preprint arXiv:2404.00656, 2024.
- [48] Leonardo Pepino, Pablo Riera, and Luciana Ferrer. Emotion recognition from speech using wav2vec 2.0 embeddings. In Proc. Interspeech 2021, pages 3400–3404, 2021.
- [49] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [50] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.
- [51] Christophe Veaux, Junichi Yamagishi, and Kirsten MacDonald. Cstr vctk corpus: English multi-speaker corpus for cstr voice cloning toolkit. University of Edinburgh. The Centre for Speech Technology Research (CSTR), 2012.
- [52] Isin Demirsahin, Oddur Kjartansson, Alexander Gutkin, and Clara Rivera. Open-source Multispeaker Corpora of the English Accents in the British Isles. In Proceedings of The 12th Language Resources and Evaluation Conference (LREC), pages 6532–6541, Marseille, France, May 2020. European Language Resources Association (ELRA).
- [53] Wei-Ning Hsu, Anuroop Sriram, Alexei Baevski, Tatiana Likhomanenko, Qiantong Xu, Vineel Pratap, Jacob Kahn, Ann Lee, Ronan Collobert, Gabriel Synnaeve, et al. Robust wav2vec 2.0: Analyzing domain shift in self-supervised pre-training. In Proc. Interspeech 2021, pages 721–725, 2021.
- [54] Dima Rekesh, Nithin Rao Koluguri, Samuel Kriman, Somshubra Majumdar, Vahid Noroozi, He Huang, Oleksii Hrinchuk, Krishna Puvvada, Ankur Kumar, Jagadeesh Balam, et al. Fast conformer with linearly scalable attention for efficient speech recognition. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 1–8. IEEE, 2023.
- [55] Jingyi Li, Weiping Tu, and Li Xiao. Freevc: Towards high-quality text-free one-shot voice conversion. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.
- [56] Ziqiang Zhang, Long Zhou, Chengyi Wang, Sanyuan Chen, Yu Wu, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, et al. Speak foreign languages with your own voice: Cross-lingual neural codec language modeling. arXiv preprint arXiv:2303.03926, 2023.
- [57] Alan Baade, Puyuan Peng, and David Harwath. Neural codec language models for disentangled and textless voice conversion. In Interspeech 2024, pages 182–186, 2024.
- [58] Dan Lyth and Simon King. Natural language guidance of high-fidelity text-to-speech with synthetic annotations. arXiv preprint arXiv:2402.01912, 2024.
- [59] OpenAI. Gpt-4 technical report. https://openai.com/research/gpt-4, 2024. Accessed: 2025-05-14.
- [60] Ziyang Ma, Zhisheng Zheng, Jiaxin Ye, Jinchao Li, Zhifu Gao, ShiLiang Zhang, and Xie Chen. emotion2vec: Self-supervised pre-training for speech emotion representation. In Findings of the Association for Computational Linguistics ACL 2024, pages 15747–15760, 2024.
- [61] Yassir Fathullah, Chunyang Wu, Egor Lakomkin, Ke Li, Junteng Jia, Yuan Shangguan, Jay Mahadeokar, Ozlem Kalinli, Christian Fuegen, and Mike Seltzer. Audiochatllama: Towards general-purpose speech abilities for llms. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5522–5532, 2024.
- [62] Julius Richter, Yi-Chiao Wu, Steven Krenn, Simon Welker, Bunlong Lay, Shinji Watanabe, Alexander Richard, and Timo Gerkmann. Ears: An anechoic fullband speech dataset benchmarked for speech enhancement and dereverberation. In Proc. Interspeech 2024, pages 4873– 4877, 2024.

- [63] Tu Anh Nguyen, Wei-Ning Hsu, Antony D’Avirro, Bowen Shi, Itai Gat, Maryam Fazel-Zarani, Tal Remez, Jade Copet, Gabriel Synnaeve, Michael Hassid, et al. Expresso: A benchmark and analysis of discrete expressive speech resynthesis. In Proc. Interspeech 2023, pages 4823–4827, 2023.
- [64] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460, 2020.

## A Details on the Datasets

### A.1 Data Processing

For all experimental datasets used in speaker and speech trait classification, we resample the audio to 16 kHz. We exclude audio samples shorter than 3 seconds, as reliably estimating speaker and speech attributes is not feasible with very short speech utterances. Audios with corrupted formats are removed from the experiments. For accent classification, we discard samples with labels like British as it does not specify the regional varieties.

### A.2 Datasets Descriptions

TIMIT [26] The TIMIT Acoustic-Phonetic Continuous Speech Corpus was collected by the Massachusetts Institute of Technology (MIT), SRI International (SRI), and Texas Instruments, Inc. (TI). The dataset includes audio recordings from 630 speakers representing eight major American English dialects. Moreover, the dataset contains speaker traits information including speaker sex and speaker age. More than half of the speakers in this dataset are male.

VoxCeleb [27] The VoxCeleb dataset is a large-scale audio-visual dataset designed for speaker recognition. The dataset contains more than 1,000 speech recordings from YouTube. The speakers have various nationalities, ages, and professions. The dataset includes metadata about speaker identity and speaker sex. Although age information is not provided in the original dataset, researchers in [12] have enriched the age information by matching the speaker profiles with online sources.

CommonVoice [16] The Common Voice dataset is an open-source collection of voice recordings to support the development of speech recognition technologies across many languages and demographics. Each data sample is a short audio clip of a person reading a provided sentence, along with selfreported metadata such as the speaker’s age (every 10 years), gender, accent, and language. We follow data processing procedures in CommonAccent [6] to prepare the data for training the accent classifier. Moreover, we use the same data to train the age and sex classifier.

EdAcc [28] The Edinburgh International Accents of English Corpus (EdAcc) includes dyadic conversational recordings to study ASR performance with different language backgrounds. The datasets include speakers with different L1 speakers, such as L1-Indian languages and L1-Spanish. We repurposed this dataset for our accent classification task.

British Isles Speaker [29] The British Isles Speaker dataset includes high-quality audio recordings of English-speaking utterances from subjects from different language backgrounds within the British Isles. Specifically, the dataset consists of recordings of 120 participants who are reported native speakers of Irish, Scottish, Welsh, and different regions in England.

L2-Arctic [30] L2-Arctic corpus is a collection of English speech recordings produced by nonnative speakers. This dataset was designed to facilitate research like accented speech recognition. It includes recordings from speakers with diverse linguistic backgrounds, such as Korean, Mandarin, and Spanish.

VoxPopuli [31] The VoxPopuli dataset is a large-scale multilingual speech dataset based on the European Parliament event recordings. In this experiment, we focus on the accented English speech subset that includes 15 different L1 language backgrounds like French and German.

Fair-Speech [32] The Fair-Speech dataset is a collection of speed data with diverse speaker meta information including age, sex, education levels, and accents. While there is no detailed information regarding L1 English speakers, we chose to only model accents with the L2 speakers who provided the exact language background information.

ESLTTS [33] The ESLTTS is a dataset collected to advance research in text-to-speech (TTS) applications, with a particular focus on non-native English speakers. The dataset contains speech samples from 134 non-native English speakers.

Hispanic-English [34] The Hispanic-English Database (LDC2014S05) consists of 30 hours of English and Spanish conversations and read speech by native speakers of Spanish. Specifically, the English sentence prompts were designed based on the TIMIT dataset. We solely use this dataset to train the accent classifier.

Nigerian-English [35] The Crowdsourced Nigerian English Speech Dataset includes high-quality recordings of Nigerian English sentences, contributed by volunteers from Lagos Nigeria and London. The speech samples in this dataset are labeled under the "Other" category in training the accent classifier.

ParaSpeechCaps [5] The ParaSpeechCaps dataset is a newly released, large-scale, humanannotated dataset that includes diverse speaker and speech information. It contains detailed annotations for accent, voice quality, and speech expressiveness from three datasets: VoxCeleb[27], EARS[62], and Expresso [63]. Since only approximately 1.6% of the holdout data is from EARS, we rely primarily on VoxCeleb and Expresso for training models related to voice quality and speech expressiveness.

MSP-Podcast [18] The MSP-Podcast dataset contains podcast data from the Internet, featuring spontaneous speech with natural human emotion expressions. In this experiment, we use the MSPPodcast v1.12 dataset which is also used for the IS2025-SER challenge. The dataset includes both arousal/valence and categorical emotion annotations.

SEP-28K [24] The SEP-28k (Stuttering Events in Podcasts) dataset is one of the largest publicly available datasets for modeling disfluencies in speech. The dataset comprises 3-second audio clips from public podcast recordings. The disfluent speech includes blocks, prolongations, sound repetitions, word repetitions, and interjections. During inference, audio is processed using a sliding window approach, with a window length of 3 seconds and a step size of 1 second.

FluencyBank [36] The FluencyBank dataset is an effort within the larger TalkBank project. Particularly, this dataset is designed to support the study of speech disfluency. It includes recordings of both spontaneous conversations and reading tasks. In our work, we use the disfluency annotations provided by [24].

### A.3 Data Split

Speaker Age and Speaker Sex For age and sex prediction, we use the standard split provided in VoxCeleb Enriched [12] and TIMIT for training, validation, and testing. Moreover, we use the train, validation, and test split created from the CommonAccent [6] pipeline to model the Common Voice dataset.

Speaker Accent We use the standard data splits provided for the EdAcc, TIMIT, and Fair-Speech datasets. For the Common Voice dataset, we follow the data processing pipeline from the CommonAccent [6] to create train, validation, and test sets. For the remaining datasets, we partition the data by selecting 60% of the speakers for training, 20% for validation, and 20% for testing. We used a fixed seed when creating the data splits to ensure reproducibility.

Voice Quality We use the standard data splits from the ParaSpeechCaps [5] dataset.

Speech Expressiveness We use the standard data splits from the ParaSpeechCaps [5] dataset.

Speech Emotion (Categorical Emotion and Arousal/Valence) We use the standard data splits from the MSP-Podcast v1.12 dataset. We report the development set results in Table 3 and Table 4, given the limited chance in challenge submissions. We report the test set performance in Table 5b.

Speech Flow We use the standard data splits from the SEP-28K dataset [24]. We also use the splits provided from the SEP-28K dataset [24] for FluencyBank [36].

##### Speaker Sex

Female 23.5%

22438

73197

Male 76.5%

(a) Speaker sex label distribution.

###### Speaker Age Distribution

Sixties 4.1%

Thirties 20.8%

3898

19903

Twenties 32.4%

30959

23117

Fourties 24.2%

Seventies and above 1.2% Teens 8.8%

8455

Fifties 8.5%

8165

(b) Speaker age label distribution.

### A.4 Training Data Distribution

Given that we use the standard data splits from the ParaSpeechCaps dataset, the detailed label distribution for modeling voice quality and speech expressiveness can be found in [5]. Similarly, the detailed distribution of speech flow and emotion labels are reported in [24] and [18], respectively. Specifically, we plot the training label distribution of speaker sex, speaker age, and speaker accent in Figure 4a, Figure 4b, and Figure 5, respectively. Specifically, the sex distribution is male-dominant given that a large portion of the data is from Common Voice. Regarding the speaker’s age, more than half of the speech samples are from speakers between 30-60 years, and approximately 30% of the data is from speakers below 30 years. Lastly, around 25% of the data in Vox-Profile originates from North America, and about 12% from speakers based in England.

## B Details on the Models

ECAPA-TDNN [10] improves the traditional Time Delay Neural Network (TDNN) and xvector architectures by integrating Squeeze-and-Excitation (SE) blocks, which improve channelwise feature modeling. It achieves state-of-the-art performance across a wide range of speaker recognition and verification tasks. In this work, we utilize the ECAPA-TDNN model (speechbrain/spkrec-ecapa-voxceleb) available on Hugging Face.

HuBERT [8] is a self-supervised learning model for speech representation learning. Building on top of Wav2Vec 2.0 [64], HuBERT introduces the masked prediction of discrete units, leading to a robust representation of speech units. Specifically, HuBERT uses k-means clustering on MFCC or self-supervised features to create pseudo-labels as discrete units. In this paper, we use the HuBERT large (facebook/hubert-large-ll60k) from the hugging face.

###### Accent Label Distribution

Semitic

- 1.9% Slavic

- 1.3% Romance

- 6.2% Germanic

- 7.2%

Northern Irish 2.9%

Other 3.5%

Scottish 4.0%

South African 2.0%

East Asia 5.9%

Irish 3.2%

Oceania

- 8.8%

2978

9674

North America 25.4%

39701

11202

4614

5449

6336

Welsh 0.9%

3152

15521

South Asia 9.9%

9206

5006

7237

South-east Asia 4.6%

13843

19047

English 12.2%

Figure 5: Accent label distribution

WavLM [9] expands on the Wav2vec 2.0 [64] pre-training objectives by incorporating masked speech and frame prediction. This model achieves competitive performance on popular speech-based downstream tasks, such as speaker recognition, speaker diarization, and speech recognition. In this paper, we use the WavLM large (microsoft/wavlm-large) from the hugging face.

Whisper [11] is a speech foundation model trained for ASR, language identification, and voice activity detection. The model was trained with more than half a million hours of audio data and achieved state-of-the-art ASR performance across multiple benchmark datasets. In this paper, we use the Whisper Tiny, Whisper Small, and Whisper Large-V3 from the hugging face.

## C Details on the Modeling Experiments

All experiments were conducted using a fixed random seed to ensure reproducibility. During training, we applied several data augmentations to the input waveforms: Gaussian noise was added with a probability of 1.0, using an SNR range of 3–30 dB; time masking was applied with a probability of 1.0, using a masking ratio between 10% and 15%; time stretching was used with a probability of 1.0, with stretch rates ranging from 0.9 to 1.1; and polarity inversion was applied with a probability of 0.5.

In speaker sex, speaker age, voice quality, speech expressiveness, and speech flow classification experiments, we use a learning rate of [0.0001, 0.0005] and a training epoch of 10. Our experiments indicate that most models perform better with a learning rate of 0.0005. Moreover, we freeze the pre-trained model weights in all experiments. On the other hand, we use a learning rate of 0.0005 and a training epoch of 15 for training the speech emotion and speaker accent prediction systems. Specifically, for speech emotion recognition training, we also experiment with unfreezing the pretrained weights in the WavLM Large model, following the IS25-SER challenge baseline. Empirically, we observe that unfreezing pre-trained weights for WavLM leads to improved performance in emotion prediction tasks. However, this is not consistently observed in training other classification systems. In all experiments, we use a batch size of 32 for training without LoRa and a batch size of 16 for training with LoRa. For all experiments, we report the utterance-level prediction performance except for the voice quality given its high subjectivity.

When comparing with the existing literature, we use the hugging face model checkpoints of CommonAccent (Jzuluaga/accent-id-commonaccent_ecapa) and Robust Wav2Vec 2.0 age and sex prediction model (audeering/wav2vec2-large-robust-24-ft-age-gender).

## D Details on the Resources

All experiments are performed on one high-performance computing server with computing nodes on request. The computing node we request includes using NVIDIA A40 or NVIDIA V100 GPUs. All experiments require only a single GPU instead of multiple GPUs for training and inference. The longest training job is accent classification model training, taking approximately 60-80 GPU hours given the variability of computing nodes allocated. The training time for the remaining classification models is all within 48 GPU hours.

## E Details on Speech Generation Task Evaluation

For FreeVC and VALLE-X, we used the code and the checkpoint from here, respectively:https://github.com/coqui-ai/TTS and https://github.com/Plachtaa/VALL-E-X. The selected test speakers and their self-reported dialectal regions are as follows: p225 (England), p294 (USA), p326 (Australia), p234 (Scottish), p302 (Canada), p261 (Northern Ireland), p245 (Ireland), and p335 (New Zealand). These speakers are the held-out test speakers of the pre-trained FreeVC model we used for inference. 3 utterances from each speaker were selected, forming each source-reference speaker pair with 9 (3 by 3) combinations. In total, 56 source-reference speaker pairs were chosen, adding up to 504 utterances.

## F Details on Confidence Levels in Speaking Style Prompt Generation

A probability of 0.8 in sex classification is considered high confidence. For accent classification, we first apply the broad accent classifier given its strong performance in classifying speakers from North America, the British Isles, and Other regions or language backgrounds. A probability of 0.8 and 0.5 in broad accent classification is considered with high (distinct) and moderate (likely) confidence, respectively. When the accent is other, we further apply the narrow accent classifier to identify the precise accent labels. In narrow accent classification, a probability of 0.5 and 0.3 is considered with high (distinct) and moderate (likely) confidence given the number of unique labels. If the maximum probability is below 0.3, we describe the accent as very hard to tell. In categorical emotion prediction, a probability of 0.4 and 0.3 is considered with high (very) and moderate (likely) confidence and otherwise was described as maybe. In arousal prediction, arousal scores below 0.2 and above 0.8 are described as calm and active. Similarly, valence scores below 0.2 and above 0.8 are described as negative and positive. For voice quality and speech expressiveness, only a probability above 0.8 is being described given the subjectivity in how human listeners perceive these traits. Similarly, a maximum probability above 0.8 in disfluency prediction will lead to the description of such a disfluency label. In addition to these predictive labels, we generate descriptions of pitch, speaking rate, and background noises. Specifically, an example of our prompt design and the corresponding response from GPT 4.1 is shown in Figure 6.

## G Details on Text Description Human Evaluation

Human evaluators were first presented with a task overview and key instructions. Upon clicking the “Next” button, they received more detailed guidance on how to perform the evaluation, followed by a consent form. To help them become familiar with the task, evaluators completed a brief trial session with a few sample items. After the trial, participants proceeded to the main evaluation phase. Each evaluator was randomly assigned 10 samples from a total of 30, presented in random order. Each sample appeared on a separate page. Evaluators were allowed to listen to each sample as many times as desired, but once they advanced to the next page, they could not return to previous samples. For each sample, participants were asked to choose the better of two text descriptions—one generated by our method and the other by ParaSpeechCaps [5]. They then rated which description better captured specific traits, and these traits are presented in random order on every page. Finally, evaluators reported their confidence level for each judgment. Figure 7 presents example screenshots illustrating the human evaluation protocol. In total, we received 20 sets of evaluations with 200 utterance-level evaluations from approximately 10 pilot participants. Most of these participants are male and non-native English speakers.

|Speaking Style Prompt Generation<br><br>An audio sample of a person's speech can be described in several ways using descriptive keywords. These keywords may include demographic data about the person (e.g. sex, age, accent) and voice characteristics (e.g. related to pitch, gender, texture, rhythm, volume, clarity, speaking rate, fluency, emotion, expressiveness, and noise).<br><br>You will be provided with several keywords that describe the speech sample. Your task is to create a simple text description using the provided keywords that accurately describe the speech sample. Ensure that the description remains grammatically correct, easy to understand, and concise. You can rearrange the keyword order as necessary, and substitute synonymous terms where appropriate. After you are provided the keywords, generate only the description and do not output anything else.<br><br>An example is provided below. female, confused, hesitant, slightly noisy environment Description: A woman's speech sounds confused and hesitant, recorded in a slightly noisy environment. Now, generate a description for the following example:<br><br>["Distinct Scottish Accent", "Definitely Female", "Likely Sad", "Normal Speed", "Low Pitch", "Slow Speed", "Very Fluent", "Very Noisy"]|
|---|

|Answer: A woman's speech with a distinct Scottish accent is very fluent and low-pitched, spoken at a slow speed in a very noisy environment, with a possible hint of sadness.|
|---|

Figure 6: Prompt example for generating speaking style.

[Figure 6]

[Figure 7]

(a) Task overview page (b) Instruction page

[Figure 8]

(c) Main evaluation page

Figure 7: Example screenshots of the human evaluation protocol.

