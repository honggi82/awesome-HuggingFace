## Voxlect: A Speech Foundation Model Benchmark for Modeling Dialects and Regional Languages Around the Globe

Tiantian Feng1, Kevin Huang1, Anfeng Xu1, Xuan Shi1, Thanathai Lertpetchpun1, Jihwan Lee1, Yoonjeong Lee1, Dani Byrd1, Shrikanth Narayanan1

# arXiv:2508.01691v1[cs.SD]3Aug2025

Abstract. We present Voxlect, a novel benchmark for modeling dialects and regional languages worldwide using speech foundation models. Specifically, we report comprehensive benchmark evaluations on dialects and regional language varieties in English, Arabic, Mandarin and Cantonese, Tibetan, Indic languages, Thai, Spanish, French, German, Brazilian Portuguese, and Italian. Our study used over 2 million training utterances from 30 publicly available speech corpora that are provided with dialectal information. We evaluate the performance of several widely used speech foundation models in classifying speech dialects. We assess the robustness of the dialectal models under noisy conditions and present an error analysis that highlights modeling results aligned with geographic continuity. In addition to benchmarking dialect classification, we demonstrate several downstream applications enabled by Voxlect. Specifically, we show that Voxlect can be applied to augment existing speech recognition datasets with dialect information, enabling a more detailed analysis of ASR performance across dialectal variations. Voxlect is also used as a tool to evaluate the performance of speech generation systems. Voxlect is publicly available with the license of the RAIL family at: https://github.com/tiantiaf0627/voxlect.

Keywords: Speech Learning, Automatic Speech Recognition, Deep Learning, Dialect Classification, Speech Foundation Model

### 1. Introduction

A “dialect” is defined as a variety of a language spoken by a particular regional and/or social group. Specifically, dialects often differ from their standard language in terms of pronunciation, grammar, and vocabulary, whereas the more commonly used term “accent” typically refers to differences in pronunciation and prosody (intonation, rhythm, phrasing). In this paper, we primarily focus on dialect classification. For example, the English language has multiple dialects worldwide, including American, British, Singaporean, and Indian English, each with its distinct linguistic features. A common example is the word for a mechanical lift, where “elevator” is used in American English, while “lift” is preferred in British English. Similarly, Mandarin Chinese has many regional dialects across China, such as the Beijing and the Sichuan dialects. For example, while Beijing Mandarin articulates the retroflex consonants such as “zh [

> úù]” in the retroflex place of articulation, the Sichuan dialects

> ts]”. Apart from dialectal variation within a single language, some countries, such as India, have a wide range of regional languages (as well as their dialectal varieties), with neighboring states speaking distinct but culturally connected languages, such as Tamil, Telugu, and Malayalam.

typically merge them with their alveolar counterparts, pronouncing them as “z [

Classifying dialects (as well as regional languages) is important for building robust speech technologies that accommodate diverse linguistic contexts. However, automatic speech recognition (ASR) systems often exhibit substantial disparities across dialectal varieties of the same language. For example, while many ASR systems perform well on speech samples of widely explored varieties, their accuracy drops significantly for under-represented dialects such as African American Vernacular English and Chicano English, reflecting biases in their training corpora [1, 2]. Similarly, KeSpeech [3] demonstrates that the state-of-the-art ASR systems show significantly lower performance on eight Mandarin dialectal varieties compared to Standard Mandarin. Such performance differences can reduce the reliability and usability of the systems and their applications, such

1 University of Southern California, Los Angeles, CA, USA (corresponding emails: tiantiaf@usc.edu.

1

Table 1. Comparison of Voxlect with existing literature in modeling speaker dialects or regional languages.

Dataset/Study Language Speaker Dialects Covered GLOBE [10]

Global English varieties ParaSpeechCaps [11] Country-level English varieties Vox-Profile [12] Global English varieties

English

Mandarin varieties KeSpeech [3] Mandarin varieties ADI [14] Arabic Arabic varieties

AIShell-3 [13]

Mandarin

Global English varieties, Germany, Italian, Spanish

CommonAccent [9] Multilingual

Global English varieties and 10 non-English dialects (Arabic, Mandarin and Cantonese, Tibetan, Spanish, German, French, Italian Thai, Indic, Brazilian Portuguese

Voxlect (Ours) Multilingual

as virtual assistants for speakers of under-resourced dialects. By explicitly modeling and recognizing varying dialects, we can not only have a better understanding of the limitations of current speech technologies but also advance the development of more reliable and robust language technologies.

The current literature on modeling speaker dialects has largely focused on English varieties. For example, the Edinburgh International Accents of English Corpus (EdAcc) [4] includes English speech from participants with a variety of first language (L1) backgrounds, such as L1-Indian languages. Moreover, the British Isles Speaker Corpus[5] provides high-quality audio recordings of English utterances from speakers across the British Isles, such as Scotland, Wales, Northern Ireland, and Ireland. In contrast, research on modeling speaker dialects in non-English languages remains relatively limited. Much of the existing work in this area focuses instead on broader language identification (LID) tasks [6, 7]. One notable effort in this field is CommonVoice [8], a large-scale multilingual speech dataset that includes self-reported speaker dialect labels. Building on this, CommonAccent [9] presents one of the few benchmarking efforts for speaker dialect classification in three nonEnglish languages (German, Spanish, & Italian). Nonetheless, its language coverage is still limited, excluding other widely spoken language families such as Chinese and Arabic.

In this paper, we present Voxlect, one of the first benchmarks for classifying dialects and regional languages from multilingual speech data. Our proposed Voxlect benchmark show unique contributions compared to prior works: (1) Unlike previous studies that focus on a limited set of dialects, Voxlect enables dialect and regional language classification across an extensive list of languages, including English, Mandarin, Indic Languages, Spanish, German, Italian, French, Brazilian Portuguese, Tibetan, and Arabic. (2) To address inconsistencies in dialect labeling in different datasets, Voxlect maps dialect labels into a unified taxonomy for each language, enabling more consistent cross-corpus analysis. (3) We show the broad utility of Voxlect through two applications in ASR performance evaluation and speech generation system assessment, showing that dialect-level distinctions matter in real-world systems. Our extensive experiments confirm that Voxlect yields reliable estimates of speaker dialects across languages spoken worldwide, which presents unique opportunities for data mining, modeling, and knowledge discovery in dialectal speech data.

### 2. Related Work

Table 1 summarizes key prior works on speaker dialect modeling. We categorize the existing literature based on whether it focuses on English or non-English languages.

###### 2.1 Modeling English Dialects

We summarize several representative works in modeling English Dialects in Table 1. In particular, CommonAccent [9] introduced a dialect classification benchmark using samples from CommonVoice-en [8]. It reported strong performance in recognizing English accents, such as American, Canadian, and British English, using the wav2vec-xlsr model. GLOBE [10] is a similar effort that develops classifiers using HuBERT pre-trained models on CommonVoice-en to predict similar dialect labels. In addition to directly modeling English-speaking dialects, ParaSpeechCaps [11] uses language models to process Wikipedia pages to augment the country-level dialect information associated with each celebrity speaker. Finally, our prior benchmark, Vox-Profile [12],

explores speaker dialect classification by unifying English dialect labels from more than ten datasets. This benchmark provides high-performing English-speaking dialect classification models using Whisper Families [7] and WavLM [15], enabling robust English dialect recognition across diverse speaker samples.

###### 2.2 Modeling Non-English Dialects

- Table 1 presents related works on modeling speaker dialects in non-English languages. There has been

growing interest in dialect modeling for Mandarin. For example, AIShell-3 [13] provides over 80 hours of multispeaker Mandarin speech, annotated with regional accents. More recently, KeSpeech [3] introduced a large-scale dataset covering eight major Mandarin subdialects, comprising speech recordings from 27,237 speakers, with a total duration of 1,542 hours. For Arabic, Sullivan et al. [14] conducted experiments for Arabic dialect classification, exploring model performance across both five major dialect groups and a more fine-grained set of 17 specific dialect labels. In addition to modeling speaker dialects within a single language, CommonAccent [9] reports experiments modeling speaker dialects in several languages, including German, Spanish, and Italian. Compared to these prior efforts, our proposed Voxlect benchmark supports dialect classification across a more extensive list of spoken languages and dialects, creating opportunities to develop robust and reliable speech technologies that accommodate different linguistic backgrounds.

### 3. Voxlect Benchmark

In this section, we introduce the design of the Voxlect Benchmark. We begin by describing the dialect labels used for each classification. We then outline the speech foundation models used in our experiments. Overall, Figure 11 presents the overview of the Voxlect Benchmark that uses speech foundation models to classify speaker dialects in languages such as Mandarin, German, and Arabic.

[Figure 1]

|[Figure 2]|
|---|

| |
|---|

|[Figure 3]|
|---|

|Speech Data Speech Foundation Models<br><br>|
|---|

[Figure 4]

| |
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

......

Figure 1. Overview of dialect classification in the Voxlect benchmark. The figure plots the geographic distribution of dialects across several language families, including major Indic languages in Southern India and dialects of Mandarin, Arabic, Thai, Spanish, Brazilian Portuguese, and German. Detailed dialect labels and corresponding datasets are provided in Table 2. (The contours shown in the drawing are only schematic, and precise boundaries of dialects are specified in Table 2.)

1Disclaimer: The map visualizations in this work are only for illustration purposes for presenting the broad geographic distribution of dialects. The highlighted parts only correspond to those dialects that may be present in experiments. We acknowledge that a single highlighted area may include other dialects or languages. The maps are based on administrative boundary data from GADM-4.1 [16]. The authors make no claims regarding the accuracy, completeness, or any interpretations of the geographic data provided, and are not responsible for any implications arising from its use.

- Table 2. The labels of speaker dialect in Voxlect benchmark. The green, blue, and violet colors indicate speaker dialects from North America, the British Isles, and other regions or language backgrounds, respectively. For Indic languages, the color green indicates languages of Hindi, Urdu, and English, which are spoken in many regions across India. Moreover, the green and blue indicate Spanish dialects from Europe and Latin America. Similarly, green indicates French and German dialects spoken within France and Germany, while blue represents dialects spoken outside these two countries.

Language Dialect Labels Datasets #Train Utterances

English

North America, English, Welsh CommonVoice-en [8]; CSLU-FAE [17];

247,098

Scottish, Northern Irish, Germanic EdAcc [4]; British Isles [5]; Irish, Germanic, Romance, L2-ARCTIC [18]; TIMIT [19];

Slavic, Semitic, Oceania VoxPopuli [20]; ESLTTS [21]; South Africa, Southeast Asia Fair-Speech [22]; ParaSpeechCaps [11] East Asia, South Asia, Other Nigerian-English [23]; Hispanic-English [24];

Arabic

Egyptian, Levantine, Maghrebi, Peninsular MASC [25]; SADA [26];

341,154 Modern Standard Arabic (MSA) Dvoice [27]

Mandarin and Cantonese

Standard/Beijing/Northeastern Mandarin

544,867

Ji-Lu Mandarin, Southwestern Mandarin, KeSpeech [3];

Jiang-Huai Mandarin, CommonVoice-yue [8]; Lan-Yin Mandarin, Zhongyuan Mandarin, CommonVoice-hk [8]

Jiao-Liao Mandarin, Cantonese

Chinese Tibetan U-Tsang, Kham, Amdo TIBMD [28] 29,347

Hindi, Urdu, English, Punjabi, Dogri

247,302

Indic Kashmiri, Sanskrit, Assamese, Manipuri IndicVoices [29];

Languages & Bengali, Odia, Maithili, Santali, Gujarati CommonVoice-en [8] Indian English Bodo, Marathi, Nepali, Konkani, Sindhi

Tamil, Telugu, Kannada, Malayalam

Thai Khummuang, Korat, Pattani, Thai-central Thai-Dialect-Corpus [30] 302,087 Spanish

Penisular, Mexican, Chileno, Andino-Pacífico CommonVoice-sp [8];

123,102 Central America and the Caribbean, Rioplatense Latin American Spanish [31]

French

France, Switzerland/Belgium/Germany CommonVoice-fr [8];

56,280 Africa, Canada African Accented French

German

German-Non-NRW Area, German-NRW Area,

CommonVoice-de [8] 71,158 Switzerland, Austria, Other

Italian Central, Northern, Southern

CommonVoice-it [8];

14,883 ITALIC [32]

Brazilian

Minas Gerais, Recife, São Paulo CORAA [33] 26,865 Portuguese

- 3.1 Labeling Dialects and Regional Languages

We note that some dialect labels (e.g., English) have varied usage across existing datasets, highlighting the need for a standardized dialect labeling scheme within each language. This standardization process allows us to combine different available datasets for training reliable dialect classification models. For most non-English cases, dialect labels are defined based on the available dialectal information in the relevant datasets. Specifically, we propose a knowledge-driven dialect categorization for English, Spanish, and French, while adopting existing conventions to group available dialectal data from established datasets for the remaining dialects and regional languages. We summarize the dialect labels and associated experimental datasets in Table 2.

English Existing work on modeling English dialects often comes with inconsistent labeling schemes [10, 9, 34]. To address this issue, we adopt the dialect taxonomy proposed in our previous Vox-Profile benchmark [12]. In particular, we first categorize two broad regional categories: North America and the British Isles. Within the British Isles, we further distinguish varieties as English (England), Scottish, Northern Irish, Welsh, and Irish. For regions and language backgrounds outside these two, we first define dialects from Oceania and South Africa as two representative English-speaking regions. Next, following common geographic conventions, we categorize English dialects spoken in Asia into three major regions: East Asia, South Asia, and Southeast Asia. These Asian regions contain a significant population of native speakers and dialects of English, such as Indian English and Singaporean English. On top of that, we categorize English accent groups based on their first language (L1) influence, such as Germanic (e.g., German), Slavic (e.g., Russian), Romance (e.g., Spanish, French), and Semitic (e.g., Arabic, Hebrew) language backgrounds. L1 backgrounds with limited data samples in existing datasets are grouped into a miscellaneous general category labeled “Other.” For instance, this includes dialects spoken by individuals with Uralic language backgrounds (e.g., Finnish) or from African

regions outside South Africa. Compared to Vox-Profile [12], Voxlect expands the benchmark by integrating the ParaSpeechCaps [11] dataset, adding approximately 90,627 additional speech utterances in conversational contexts to the English dialect classification tasks.

Arabic Arabic dialect classification has been well-established in the literature [25, 26, 14]. However, we were unable to obtain the dataset used for the fine-grained 17-dialect classification in [14]. Therefore, we follow conventions in datasets MASC [25] and SADA [26] and categorize Arabic into five major groups: Egyptian, Levantine (e.g., Lebanon), Peninsular (e.g., Saudi Arabia), Maghrebi (e.g., Morocco), and Modern Standard Arabic (MSA).

Mandarin and Cantonese The classification of Mandarin dialects has been deeply explored within Chinese linguistics. We follow the Mandarin dialect labels listed by KeSpeech [3]. Given that there are limited speech samples for Beijing Mandarin and Northeastern Mandarin in KeSpeech and that these two dialects share mutual intelligibility with Standard Mandarin, we group these three varieties into a single category labeled simply Mandarin. However, we highlight that Beijing Mandarin and Northeastern Mandarin do differ from Standard Mandarin in certain lexical items and pronunciations. In addition to classifying Mandarin dialects, we add Cantonese samples from CommonVoice-yue [8] and CommonVoice-hk [8] to enrich the coverage of Chinese languages.

Chinese Tibetan Tibetan is a language spoken across parts of China, India, Nepal, and Bhutan. Since there are limited datasets of Tibetan dialects, we classify three major dialects of Greater Tibet in China as presented in TIBMD [28]: Ü-Tsang, Kham, and Amdo.

Indic Languages and Indian English Many languages, from a number of language families, are spoken in India, and each is typically considered a separate language rather than a dialect. In this paper, we consider 22 major Indic languages (shown in Table 2) as listed in IndicVoices [29]. Moreover, given the widespread use of English across India and its status as an official language of the country, we specifically include Indian English (labels associated with India in CommonVoice-en) as a separate language category to reflect its presence in the multilingual context.

Thai We use the dialect labels described in the Thai-Dialect-Corpus [30], one of the most notable efforts to document dialectal variation among Thai speakers. Here, we consider four major dialects: Thai-central, the standard Thai mainly spoken by Central Thai but also across Thailand (e.g., Bangkok); Khammuang, spoken in northern Thailand (e.g., Chiang Mai); Korat, a dialect influenced by both Central Thai and Isan (Northeastern Thai); and Pattani, a Malay-influenced dialect (Southern Thai).

Spanish We categorize Spanish dialects into two broad groups based on geographic landscapes guided by [35]: Peninsular Spanish (spoken in Europe) and Latin American Spanish. Moreover, we break down Latin American Spanish into five major dialectal groups: Mexican, Central America and the Caribbean (e.g., Costa Rica), Chileno (e.g., Chile), Andino-Pacífico (e.g., Peru), and Rioplatense (e.g., Argentina).

French French is spoken by a geographically diverse population worldwide. In addition to its use in France, there is a notable French-speaking population in its neighboring European countries such as parts of Switzerland, Belgium, and Germany. And apart from Europe, French is widely spoken in regions of Africa and Canada (e.g., Quebec). Therefore, we define four French dialect categorizations: France (standard French), the neighboring countries of France (French varieties spoken in Switzerland, Belgium, and Germany), Africa, and Canada. That said, we choose not to refine dialect labels for speakers from France’s neighboring countries due to the limited availability of speech samples.

German Akin to our approach to categorizing French dialects, we group German dialects into varieties spoken within Germany and those in other countries with German-speaking populations, adopting the label categories from CommonVoice-de [8]. Having German-North Rhine-Westphalia (NRW) as a separate group owing to its large number of utterances in the dataset, the German dialects are grouped into the following five categories: German-NRW (western Germany), German (non-NRW area), Swiss, Austrian, and the other German-speaking populations.

Italian Given the limited speech datasets presented for Italian dialects and limited speech samples available for individual cities, we create a common regional taxonomy and group Italian dialects into three major categories: Northern Italian (e.g., Venetian), Central Italian (e.g., Tuscan), and Southern Italian (e.g., Sicilian).

Brazilian Portuguese There are limited European Portuguese speech datasets with academically friendly licenses, and thus we are unable to include them. In contrast, we rely on CORAA [33] for studying dialects in Brazilian Portuguese. Based on this dataset, we categorize Brazilian Portuguese into three representative regional groups: São Paulo, Minas Gerais, and Recife.

[Figure 8]

###### Speech

|Speech Foundation Model<br><br>[Figure 9]|
|---|

|Weighted Average|
|---|

###### LORA

|1D point-wise Convolution| |
|---|---|
| | |

|Average Pooling| |
|---|---|
| | |

|Dialect Classifier|
|---|

Figure 2. Overview of speaker dialect classification architecture in the Voxlect benchmark.

- 3.2 Speech Foundation Models

In Voxlect, we evaluate several widely studied speech foundation models, including the Massively Multilingual Speech (MMS) [6], WavLM [36], and Whisper [7] family. Among these, MMS and Whisper are multilingual models trained on large-scale cross-lingual datasets, while WavLM is trained only on English. Given the generalizability of these speech foundation models, existing works [37, 12] have shown that direct fine-tuning of hidden outputs from the encoder layers can achieve strong performance in a wide range of downstream speech modeling tasks. We present our modeling architecture in Figure 2. In our implementation, we first compute a weighted average of the hidden states across all encoder layers, including both convolutional and transformer layers. The aggregated output is then processed through 1D-pointwise convolutional layers. Finally, we average the convolutional outputs to obtain the final embeddings, which are passed through fully connected layers for classification. To further improve the classification performance, we integrate LoRa [38] into all fine-tuning experiments as an effective approach for adapting speech foundation models.

- 4. Experiments

- 4.1 Datasets

We sampled 30 publicly available data sources to conduct benchmark experiments with a total of over 2 million speech samples from the 11 language groups detailed above. For all datasets used in dialect classification experiments, we resample audio to 16 kHz to match the sampling rate of speech foundation models. Audio clips shorter than 3 seconds are excluded, as such short utterances are insufficient for robust dialect classification. To manage computational constraints during training, all samples are truncated to a maximum duration of 15 seconds. For English dialect classification, we discard samples labeled as British, as this lacks specificity on regional varieties such as Scottish. In Spanish dialect experiments, we exclude Colombian speakers from the Latin American Spanish dataset [31], given the dialectal overlap between the Caribbean and Andino-Pacífico varieties. Due to the large size of the IndicVoice [29] dataset, we subsample a maximum of 10 utterances per speaker. The detailed distribution of dialects or regional languages in each language group is in the Appendix.

###### 4.2 Experimental Details

All experiments were conducted using a fixed random seed to ensure reproducibility. During training, we applied several data augmentations to the input waveforms: the Gaussian noise was added with a probability of 1.0, using an SNR range of 3–30 dB; time masking was applied with a probability of 1.0, using a masking ratio between 10% and 15%; time stretching was used with a probability of 1.0, with stretch rates ranging from 0.9 to 1.1; and polarity inversion was applied with a probability of 0.5. We use a learning rate of [0.0001, 0.0005] and a training epoch of 15. We perform the training for 5 epochs on Thai and Arabic dialect classification

Table 3. Comparison of different speech foundation models in classifying speaker dialects. Overall, the results show that multilingual models including Whisper-Large and MMS-LID-256 achieve the overall best performance, while the speech foundation model pre-trained only with English data (WavLM+) shows relatively lower performance in dialect prediction. Bold and underlines indicate the best and the second best performance, respectively.

Mandarin Indic Lang and

English Arabic

Tibetan Thai and Cantonese Indian English

Acc F1 Acc F1 Acc F1 Acc F1 Acc F1 Acc F1 Self-Supervised

↰

WavLM+ 79.4 0.705 79.2 0.681 79.5 0.655 63.8 0.634 80.6 0.627 91.0 0.853 ↰

MMS-300M 65.2 0.508 84.9 0.784 78.5 0.663 67.6 0.688 79.4 0.721 90.4 0.851 ↰

MMS-LID 256 80.4 0.714 91.8 0.860 82.9 0.708 82.6 0.795 86.4 0.783 95.9 0.935 Whisper Family

↰

Whisper Tiny 67.2 0.545 88.1 0.813 77.4 0.643 64.6 0.642 79.6 0.656 93.2 0.896 ↰

Whisper Small 78.3 0.688 93.0 0.892 81.7 0.712 75.0 0.748 78.5 0.620 95.4 0.931 ↰

Whisper Large 83.0 0.755 94.2 0.923 82.5 0.702 77.1 0.767 82.0 0.719 96.3 0.943

Brazilian Portuguese Acc F1 Acc F1 Acc F1 Acc F1 Acc F1

Spanish French Germany Italian

Self-Supervised

↰

WavLM+ 66.2 0.662 59.8 0.544 87.6 0.769 64.0 0.671 97.0 0.966 ↰

MMS-300M 58.1 0.568 83.3 0.665 89.8 0.737 56.3 0.604 95.1 0.942 ↰

MMS-LID 256 77.4 0.780 86.4 0.706 96.8 0.906 76.9 0.782 99.1 0.990 Whisper Family

↰

Whisper Tiny 62.5 0.630 72.7 0.520 78.9 0.691 60.2 0.614 86.8 0.819 ↰

Whisper Small 64.3 0.650 83.1 0.667 82.5 0.753 61.9 0.622 94.6 0.920 ↰

Whisper Large 77.8 0.789 87.0 0.712 93.6 0.875 73.9 0.745 98.6 0.980

due to a faster convergence rate. Our experiments indicate that models perform better with a learning rate of 0.0005. The pre-trained model weights are downloaded from Huggingface.

Moreover, we freeze the pre-trained weights in all experiments, while we apply a LoRa with a rank size of 64 to the feedforward layer as suggested by our previous works [39, 12]. Specifically, we use a batch size of 16 for training the Whisper and WavLM models, and reduce the batch size to 6 for MMS-LID-256 due to its larger parameter size. All pre-trained model checkpoints are downloaded from Huggingface. For evaluation, we report utterance-level Macro-F1 and accuracy on the test set for each language group. The default test set from each data source is used as the evaluation set; otherwise, we randomly select 20% of speakers as the test set.

### 5. Benchmark Results

###### 5.1 Dialect Classification Results

- Table 3 presents a comparative analysis of speech foundation models in classifying speaker dialects of 11

language groups. Overall, the models that are pre-trained with multilingual data, particularly Whisper-Large and MMS-LID 256, consistently achieve the best performance in most experiments. Specifically, WhisperLarge achieves the highest accuracy and Macro-F1 in 5 out of 11 language groups, including Arabic (MacroF1 = 0.923), Mandarin and Cantonese (Macro-F1 = 0.889), and Thai (Macro-F1 = 0.943). On the other hand, MMS-LID-256, a multilingual model fine-tuned for LID, outperforms others in languages like Tibetan (Macro-F1 = 0.783), German (Macro-F1 = 0.906), and Brazilian Portuguese (Macro-F1 = 0.990). In contrast, WavLM+, which is pre-trained only on English data, performs relatively poorly compared to multilingual models, especially across typologically distant languages such as Indic (Macro-F1 = 0.634) and Arabic (MacroF1 = 0.681). These results highlight the advantages of multilingual models in classifying dialectal variations across different linguistic contexts.

###### 5.2 Geographical Proximity and Classification

To better understand the performance of these dialect classification models, we visualize both the confusion matrix and the most significant misclassification patterns for two language groups, Spanish and Mandarin, in Figure 3. Given that Whisper-Large yields consistently better classification performance than most other models, we create the visualization based on Whisper-Large classification. Each map highlights the regions of major dialect groups, with arrows and percentages showing the most frequent misclassifications on the evaluation set. Overall, we identify a consistent pattern observed in the strong influence of geographic proximity on classification errors. Specifically, dialects spoken in neighboring regions or states are more likely to be confused by the dialect classification model. For example, in the Mandarin group, the highest confusion occurs between Zhongyuan and Ji-Lu Mandarin (21.3%), while in the Spanish group, Caribe and Central dialects

###### Confusion Matrix of Spanish Dialect Classification Confusion Matrix of Mandarin and Cantonese Dialect Classification

[Figure 10]

[Figure 11]

|[Figure 12]|
|---|

|[Figure 13]|
|---|

- Figure 3. Confusion matrices and geographic visualizations of dialect classification errors for Spanish and Mandarin. The maps highlight the most significant misclassification patterns, with arrows indicating frequently confused dialect pairs.

[Figure 14]

- Figure 4. Comparison of relative differences in dialect prediction performance between Whisper-Large and MMS-LID-256 under different noise levels. Each dot represents the performance change for a single dialect classifier of a language, and * indicates statistically significant differences (p<0.05).

are often misclassified as Andino-Pacífico (16.2%). In contrast, dialects that are more geographically distant, such as Peninsular (European) Spanish versus Latin American varieties, or Cantonese versus Mandarin, are less frequently confused. The confusion matrix for each dialect classifier is presented in the Appendix. These findings suggest that geographical proximity among speakers of varieties of a language often correlates with greater linguistic similarity. While this may present a challenge for a discrete dialect classification, it is consistent with the linguistic study [40, 41] of the evolution languages, which recognizes that dialects emerge through contact (and isolation) patterns between peoples over time.

- 5.3 Robustness of Dialect Classification

We further investigate the robustness of dialect classification under varied noise levels and utterance lengths. The goal here is to assess how robust the fine-tuned models are to real-world scenarios, where acoustic conditions may be degraded and speaking durations vary.

Robustness to Noise: We introduce the Gaussian noise at three signal-to-noise ratio (SNR) levels: 25dB, 15dB, and 5dB. We compare the relative performance changes of the two best-performing fine-tuned models (Whisper-Large and MMS-LID-256) against their performance on clean speech. The comparisons in Figure 4 show that Whisper-Large and MMS-LID-256 demonstrate similar robustness and relatively smaller performance degradation at moderate noise levels (SNR = 15 or 25 dB). However, under the high noise level (SNR = 5 dB), MMS-LID-256 shows a significantly larger drop in dialect prediction performance compared to Whisper-Large.

- Table 4. Comparison of dialect classification (Macro-F1) between short (≤6 sec) and long (>6 sec) utterances.

MMS-LID-256 Whisper-Large v3 Short Long Short Long

Arabic 0.856 0.842 0.920 0.934 Mandarin 0.707 0.710 0.702 0.701 Indic Language 0.690 0.977 0.659 0.971 Tibetan 0.774 0.831 0.703 0.827 Thai 0.921 0.949 0.921 0.966 Spanish 0.780 0.765 0.781 0.778 German 0.896 0.917 0.858 0.892 French 0.707 0.690 0.702 0.726

Robustness to Utterance Length: We compare differences in dialect classification between short and long utterances in Table 4. Specifically, we use a threshold of 6 seconds to define short utterances. The results indicate that, in more than half the cases, the classification performance of both models improves with longer utterances. Particularly, we observe a performance improvement of 0.3 F1 in Indic language classification with longer utterances.

### 6. Data-centric Applications of Voxlect

Here, we demonstrate how Voxlect facilitates two main speech technology applications: analysis of ASR models and automated evaluation of speech generation systems.

###### 6.1 Automatic Speech Recognition

Here, we investigate whether we can leverage the predicted dialect labels to analyze ASR performance across different dialects. We first predict dialect labels for existing datasets and examine whether these predicted labels lead to the same insights as using the ground truth dialect labels. Particularly, we predict dialect labels for the test sets of Mandarin and German. For ease of visualization, we include German, Austrian, Swiss, and Other for German, and standard Mandarin, Ji-Lu Mandarin, Southwestern Mandarin, and Zhongyuan Mandarin for Mandarin. We use fine-tuned Whisper-Large models to predict dialects and MMS for ASR.

[Figure 15]

[Figure 16]

Figure 5. ASR performance trends, grouped by ground truth and labels predicted by Voxlect.

- Table 5. Comparison of human evaluation and automated evaluation by Voxlect in assessing quality of dialect characteristics in generated speech of Chinese dialects.

Human(1-5) Voxlect(0-100%)

Shanghai(Jiang-Huai) 2.85 36.6% Sichuan(Southwestern) 3.35 40.4% Tianjin(Ju-Lu) 1.90 20.5% Zhengzhou(Zhongyuan) 3.15 32.3% Cantonese 3.50 53.4%

When computing ASR performance with generated labels, we include only test samples with a predicted dialect probability above 0.7. As shown in Figure 5, ASR performance trends using predicted labels from Voxlect closely align with those based on the ground truth labels. In the Mandarin ASR evaluation, speakers of standard Mandarin consistently have lower WER than those using regional sub-dialects, regardless of whether ground truth or predicted labels are used. Among the sub-dialects, Southwestern Mandarin shows the highest WER in both labeling methods. Similarly, we observe that, in the German ASR experiment, speakers labeled as “German (Non-NRW area)” demonstrate lower WER compared to those labeled as “Austria”, “Swiss”, or “Other.” Particularly, the "Other" category shows the highest WER across both ground truth and predicted labels. These findings show that Voxlect provides a reliable tool for identifying limitations in ASR models.

- 6.2 Evaluation of TTS Systems

With the rise of media personalization, the technology of generating dialectal speech is popularized in public services and entertainment. Here, we generate speech samples in specific dialects and investigate whether human evaluations align with evaluations generated by Voxlect regarding the dialect characteristics of the generated speech. Given the availability of publicly accessible TTS models, we focus our experiments on Mandarin. We note that related evaluations for English speech generation are presented in Vox-Profile [12]. Specifically, we use 10 text prompts designed by a phonetician and reference speakers drawn from the KeSpeech test set. To generate speech in five distinct Chinese dialects, we utilize the CosyVoice-2 [42]. We use the prompt “用{方 言}说这句话” (which translates to "Use dialect to say this sentence"), specifying each of the five dialects: Cantonese, Sichuan (Southwestern), Tianjin (Ji-Lu), Zhengzhou (Zhongyuan), and Shanghai (Jiang-Huai). These were selected to represent a range of major dialectal regions in China. The details of the prompts are in the Appendix.

We invite colleagues with a native language background in these dialects to assess the dialect characteristics of the generated speech samples. Participants were asked to rate the quality of the dialect characteristics on a 5-point scale. We compare the average predicted probability of the corresponding dialects using Voxlect with the averaged human ratings in Table 5. The results indicate that the human ratings closely match the model predictions, with the target dialect of Tianjin (Ju-Lu) receiving the lowest ratings and Cantonese receiving the highest scores in both human and machine evaluation.

- 7. Limitations and Responsible Use

While Voxlect introduces a comprehensive benchmark for dialect and regional language classifications, several limitations remain. First, the dialect labels often originate from self-reports, which may contain labeling noise. Second, the benchmark is constrained by the availability of public datasets with existing labels. Thus, refined labels in specific areas (e.g., Paris of France) are not precisely captured, and Mandarin dialects such as those used in Hainan are not studied. Moreover, many globally spoken dialects and languages, such as regional varieties of Korean, Eastern European, or African languages, are currently not included. Third, the robustness of our proposed benchmark has not yet been evaluated beyond different noise levels and utterance lengths, such as cross-domain generalization (e.g., training on read speech and evaluating on natural speech). Finally, dialect classification may potentially expose private data about speakers and raise privacy concerns. While all datasets used in this work are publicly available and Voxlect adheres to the scope of the data usage and license, we take additional measures to mitigate the risk of misuse. Specifically, we release the code and model checkpoints under the Responsible AI License (RAIL) to require responsible use of Voxlect. Users should respect the privacy and consent of the data subjects and adhere to the relevant laws and regulations in their jurisdictions when using Voxlect.

### 8. Conclusion and Future Work

In this work, we propose Voxlect, a benchmark for predicting dialects and regional languages using speech foundation models. This benchmark includes large-scale machine learning experiments using over 2 million speech utterances from 30 public data sources. We experiment with widely adopted speech foundation models and release a suite of high-performing dialects and regional language classification models. Benchmarks of this kind in multilingual contexts are rarely represented in the literature. Our dialect classification results demonstrate that geographic proximity is reflected in dialect similarity. While this creates challenges for strict classification, it could bring insight into the cultural and historical factors impacting language evolution. Moreover, Voxlect enables a wide range of downstream applications, including the analysis of speech recognition performance and the evaluation of speech generation systems. Our next steps include expanding the scope of the benchmark by integrating dialects of additional languages. For example, we plan to include dialectal variations in Korean, which show rich regional diversity such as the Seoul, Gyeongsang, and Jeolla dialects. We also plan to apply our benchmark models to enrich existing speech datasets with dialect data, which supports the development of downstream applications such as speech generation.

### Appendix

#### A. Data Distribution Attached is the training distribution for different dialects.

Khummuang 4.7% Pattani 6.3% Korat

19101

20089

- 6.7%

Thai Central 82.3%

Training Data Distribution - Thai

- Figure 6. Training distribution for Thai dialect classification.

4055 36027

5241

10957

Africa 19.5%

Swiss/Belgium/German 9.3%

Canada 7.2%

France 64.0%

Training Data Distribution - French

- Figure 7. Training distribution for French dialect classification.

28142

27609

5399

8659

Other 1.9% Austria

12.2% Swiss 7.6%

German-NRW Area 38.8%

German-Non-NRW Area 39.5%

Training Data Distribution - German

- Figure 8. Training distribution for German dialect classification.

248706

###### Training Data Distribution - Arabic

Maghrebi 1.7% Levantine 19.8%

Peninsular 20.7%

70520

67533

Egyptian 3.8%

13054

MSA 54.0%

184280

###### Figure 9. Training distribution for Arabic dialect classification.

###### Training Data Distribution - Mandarin and Cantonese

400000

300000

200000

100000

0

Standard Jiang-HuaiSouthwestern Jiao-Liao Zhongyuan Ji-Lu Lan-Yin Cantonese

###### Figure 10. Training distribution for Mandarin dialect and Cantonese classification.

###### Training Data Distribution - Tibetan

Amdo 25.5%

7485

Kham 4.9%

1432

Ü-Tsang 20430 69.6%

Figure 11. Training distribution for Tibetan (in China) dialect classification.

##### Training Data Distribution - English

125000

100000

75000

50000

25000

0

NorthAmericaWelshSouthAsiaSouth-eastAsiaEnglishOceania IrishEastAsiaSouthAfricanScottishOtherNortherIrishGermanicRomanceSlavicSemitic

###### Figure 12. Training distribution for English dialect classification.

###### Training Data Distribution - Brazilian Portuguese

São Paulo 14.1% Minas Gerais 2.6%

3796

22358

Recife 83.2%

###### Figure 13. Training distribution for Brazilian Portuguese dialect classification.

###### Training Data Distribution - Spanish

80000

60000

40000

20000

0

Penisular Mexican Andino-Pacífico Caribeand Rioplatense Chileno

Figure 14. Training distribution for Spanish dialect classification.

### B. Confusion Matrix

Confusion matrices for Thai, French, German, Italian, Arabic, and Indic languages are shown in Figure 15, 16, 17, 18, 19, and 20, respectively.

[Figure 17]

- Figure 15. Confusion Matrix of Thai dialect prediction.

[Figure 18]

- Figure 16. Confusion Matrix of French dialect prediction.

[Figure 19]

###### Figure 17. Confusion Matrix of German dialect prediction.

[Figure 20]

###### Figure 18. Confusion Matrix of Italian dialect prediction.

[Figure 21]

###### Figure 19. Confusion Matrix of Arabic dialect prediction.

[Figure 22]

###### Figure 20. Confusion Matrix of Indic languages prediction.

- C. Prompt The text prompts for the speech generation of Mandarin speech are listed below:

- 1. "早上，小兔子玉玉背着胡萝卜书包去上学。"
- 2. “今天是个特别的日子,年度南方森林运动会。”
- 3. “班主任熊猫翁大大说：“玉玉，你代表咱们班参加吧！””
- 4. “有一次，北风和太阳正在争论谁比较有本事。”
- 5. “他们正好看到有个人走过，那个人穿著一件斗篷。”
- 6. “他们就说了，谁可以让那个人脱掉那件斗篷，就算谁比较有本事。”
- 7. “于是，北风就拼命地吹。怎料，他吹得越厉害，那个人就越是用斗篷包紧自己。最后，北风没办法，只

好放弃。”

- 8. “接著，太阳出来晒了一下，那个人就立刻把斗篷脱掉了。于是，北风只好认输了。”
- 9. “中国疆域广阔，人口众多，首都在北京。官方语言是普通话，使用汉字。同时，中华各民族儿女也使用

民族语文。”

- 10. “这里风光旖旎，众多山河湖海，凭借南北各种美景，吸引了众多游客。”

### References

- [1] Camille Harris, Chijioke Mgbahurike, Neha Kumar, and Diyi Yang, “Modeling gender and dialect bias in automatic speech recognition,” in Findings of the Association for Computational Linguistics: EMNLP 2024, 2024, pp. 15166–15184.
- [2] Kalvin Chang, Yi-Hui Chou, Jiatong Shi, Hsuan-Ming Chen, Nicole Holliday, Odette Scharenborg, and David R Mortensen, “Self-supervised speech representations still struggle with african american vernacular english,” in Proc. Interspeech 2024, 2024, pp. 4643–4647.
- [3] Zhiyuan Tang, Dong Wang, Yanguang Xu, Jianwei Sun, Xiaoning Lei, Shuaijiang Zhao, Cheng Wen, Xingjun Tan, Chuandong Xie, Shuran Zhou, et al., “Kespeech: An open source speech dataset of mandarin and its eight subdialects,” in Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.
- [4] Ramon Sanabria, Nikolay Bogoychev, Nina Markl, Andrea Carmantini, Ondrej Klejch, and Peter Bell, “The edinburgh international accents of english corpus: Towards the democratization of english asr,” in ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.
- [5] Isin Demirsahin, Oddur Kjartansson, Alexander Gutkin, and Clara Rivera, “Open-source multi-speaker corpora of the english accents in the british isles,” in Proceedings of the twelfth language resources and evaluation conference, 2020, pp. 6532–6541.
- [6] Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam Fazel-Zarandi, et al., “Scaling speech technology to 1,000+ languages,” Journal of Machine Learning Research, vol. 25, no. 97, pp. 1–52, 2024.
- [7] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever, “Robust speech recognition via large-scale weak supervision,” in International Conference on Machine Learning. PMLR, 2023, pp. 28492–28518.
- [8] Rosana Ardila, Megan Branson, Kelly Davis, Michael Kohler, Josh Meyer, Michael Henretty, Reuben Morais, Lindsay Saunders, Francis Tyers, and Gregor Weber, “Common voice: A massively-multilingual speech corpus,” in Proceedings of the Twelfth Language Resources and Evaluation Conference, 2020, pp. 4218–4222.
- [9] Juan Zuluaga-Gomez, Sara Ahmed, Danielius Visockas, and Cem Subakan, “Commonaccent: Exploring large acoustic pretrained models for accent classification based on common voice,” in Proc. Interspeech 2023, 2023, pp. 5291–5295.
- [10] Wenbin Wang, Yang Song, and Sanjay Jha, “Globe: A high-quality english corpus with global accents for zero-shot speaker adaptive text-to-speech,” in Proc. Interspeech 2024, 2024, pp. 1365–1369.
- [11] Anuj Diwan, Zhisheng Zheng, David Harwath, and Eunsol Choi, “Scaling rich style-prompted text-to-speech datasets,” arXiv preprint arXiv:2503.04713, 2025.
- [12] Tiantian Feng, Jihwan Lee, Anfeng Xu, Yoonjeong Lee, Thanathai Lertpetchpun, Xuan Shi, Helin Wang, Thomas Thebaud, Laureano Moro-Velazquez, Dani Byrd, et al., “Vox-profile: A speech foundation model benchmark for characterizing diverse speaker and speech traits,” arXiv preprint arXiv:2505.14648, 2025.
- [13] Yao Shi, Hui Bu, Xin Xu, Shaoji Zhang, and Ming Li, “Aishell-3: A multi-speaker mandarin tts corpus,” in Proc. Interspeech 2021, 2021, pp. 2756–2760.
- [14] Peter Sullivan, AbdelRahim Elmadany, and Muhammad Abdul-Mageed, “On the robustness of arabic speech dialect identification,” in Proc. Interspeech 2023, 2023, pp. 5326–5330.
- [15] Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al., “Wavlm: Large-scale self-supervised pre-training for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.
- [16] GADM, “Database of global administrative areas, version 4.1,” https://gadm.org/, 2025, Accessed on 28 July 2025.
- [17] T Lander, “Cslu: 22 languages corpus (ldc2005s26),” Linguistic Data Consortium, 2005.
- [18] Guanlong Zhao, Sinem Sonsaat, Alif Silpachai, Ivana Lucic, Evgeny Chukharev-Hudilainen, John Levis, and Ricardo GutierrezOsuna, “L2-arctic: A non-native english speech corpus,” in Proc. Interspeech 2018, 2018, pp. 2783–2787.
- [19] John S Garofolo, Lori F Lamel, William M Fisher, Jonathan G Fiscus, and David S Pallett, “Darpa timit acoustic-phonetic continous speech corpus cd-rom. nist speech disc 1-1.1,” NASA STI/Recon technical report n, vol. 93, pp. 27403, 1993.
- [20] Changhan Wang, Morgane Riviere, Ann Lee, Anne Wu, Chaitanya Talnikar, Daniel Haziza, Mary Williamson, Juan Pino, and Emmanuel Dupoux, “Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2021, pp. 993–1003.
- [21] Wenbin Wang, Yang Song, and Sanjay Jha, “Usat: A universal speaker-adaptive text-to-speech approach,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.
- [22] Irina-Elena Veliche, Zhuangqun Huang, Vineeth Ayyat Kochaniyan, Fuchun Peng, Ozlem Kalinli, and Michael L Seltzer, “Towards measuring fairness in speech recognition: Fair-speech dataset,” in Proc. Interspeech 2024, 2024, pp. 1385–1389.
- [23] “Crowdsourced high-quality nigerian english speech data set.,” Open Speech and Language Resources, 2019.
- [24] William Byrne, Eva Knodt, Jared Bernstein, and Farzhad Emami, “Hispanic-english database (ldc2014s05),” Linguistic Data Consortium, 2014.
- [25] Mohammad Al-Fetyani, Muhammad Al-Barham, Gheith Abandah, Adham Alsharkawi, and Maha Dawas, “Masc: Massive arabic speech corpus,” in 2022 IEEE Spoken Language Technology Workshop (SLT). IEEE, 2023, pp. 1006–1013.
- [26] Sadeen Alharbi, Areeb Alowisheq, Zoltán Tüske, Kareem Darwish, Abdullah Alrajeh, Abdulmajeed Alrowithi, Aljawharah Bin Tamran, Asma Ibrahim, Raghad Aloraini, Raneem Alnajim, et al., “Sada: Saudi audio dataset for arabic,” in ICASSP 20242024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 10286–10290.
- [27] I Benelallam, AM Naira, and A Allak, “Dvoice: an open source dataset for automatic speech recognition on moroccan dialectal arabic (2021),” .
- [28] Yue Zhao, Xiaona Xu, Jianjian Yue, Wei Song, Xiali Li, Licheng Wu, and Qiang Ji, “An open speech resource for tibetan multi-dialect and multitask recognition,” International Journal of Computational Science and Engineering, vol. 22, no. 2-3, pp. 297–304, 2020.
- [29] Tahir Javed, Janki Nawale, Eldho George, Sakshi Joshi, Kaushal Bhogale, Deovrat Mehendale, Ishvinder Sethi, Aparna Ananthanarayanan, Hafsah Faquih, Pratiti Palit, et al., “Indicvoices: Towards building an inclusive multilingual speech dataset for indian languages,” in Findings of the Association for Computational Linguistics ACL 2024, 2024, pp. 10740–10782.

- [30] Artit Suwanbandit, Burin Naowarat, Orathai Sangpetch, and Ekapol Chuangsuwanich, “Thai dialect corpus and transfer-based curriculum learning investigation for dialect automatic speech recognition,” in Proc. Interspeech, 2023, vol. 2.
- [31] Adriana Guevara-Rukoz, Isin Demirsahin, Fei He, Shan-Hui Cathy Chu, Supheakmungkol Sarin, Knot Pipatsrisawat, Alexander Gutkin, Alena Butryna, and Oddur Kjartansson, “Crowdsourcing latin american spanish for low-resource text-to-speech,” in Proceedings of the Twelfth Language Resources and Evaluation Conference, 2020, pp. 6504–6513.
- [32] Alkis Koudounas, Moreno La Quatra, Lorenzo Vaiani, Luca Colomba, Giuseppe Attanasio, Eliana Pastor, Luca Cagliero, and Elena Baralis, “Italic: An italian intent classification dataset,” in Proc. Interspeech 2023, 2023, pp. 2153–2157.
- [33] Arnaldo Candido Junior, Edresson Casanova, Anderson Soares, Frederico Santos de Oliveira, Lucas Oliveira, Ricardo Corso Fernandes Junior, Daniel Peixoto Pinto da Silva, Fernando Gorgulho Fayet, Bruno Baldissera Carlotto, Lucas Rafael Stefanel Gris, et al., “Coraa asr: a large corpus of spontaneous and prepared speech manually validated for speech recognition in brazilian portuguese,” Language Resources and Evaluation, vol. 57, no. 3, pp. 1139–1171, 2023.
- [34] Jinzuomu Zhong, Korin Richmond, Zhiba Su, and Siqi Sun, “Accentbox: Towards high-fidelity zero-shot accent generation,” in ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.
- [35] Melvyn C Resnick, Phonological variants and dialect identification in Latin American Spanish, vol. 201, Walter de Gruyter, 2012.
- [36] Shujie Hu, Long Zhou, Shujie Liu, Sanyuan Chen, Lingwei Meng, Hongkun Hao, Jing Pan, Xunying Liu, Jinyu Li, Sunit Sivasankaran, et al., “Wavllm: Towards robust and adaptive speech large language model,” arXiv preprint arXiv:2404.00656, 2024.
- [37] Leonardo Pepino, Pablo Riera, and Luciana Ferrer, “Emotion recognition from speech using wav2vec 2.0 embeddings,” in Proc. Interspeech 2021, 2021, pp. 3400–3404.
- [38] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al., “Lora: Low-rank adaptation of large language models.,” ICLR, vol. 1, no. 2, pp. 3, 2022.
- [39] Tiantian Feng and Shrikanth Narayanan, “Peft-ser: On the use of parameter efficient transfer learning approaches for speech emotion recognition using pre-trained speech models,” in 2023 11th International Conference on Affective Computing and Intelligent Interaction (ACII). IEEE, 2023, pp. 1–8.
- [40] Jack K Chambers and Peter Trudgill, Dialectology, Cambridge University Press, 1998.
- [41] John Nerbonne, “Measuring the diffusion of linguistic change,” Philosophical Transactions of the Royal Society B: Biological Sciences, vol. 365, no. 1559, pp. 3821–3828, 2010.
- [42] Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, et al., “Cosyvoice 2: Scalable streaming speech synthesis with large language models,” arXiv preprint arXiv:2412.10117, 2024.

