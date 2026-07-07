## Integrating Large Language Models into a Tri-Modal Architecture for Automated Depression Classification

Santosh V. Patapati

### arXiv:2407.19340v5[cs.CV]11Oct2024

Abstract—Major Depressive Disorder (MDD) is a pervasive mental health condition that affects 300 million people worldwide. This work presents a novel, BiLSTM-based tri-modal model-level fusion architecture for the binary classification of depression from clinical interview recordings. The proposed architecture incorporates Mel Frequency Cepstral Coefficients, Facial Action Units, and uses a two-shot learning based GPT-4 model to process text data. This is the first work to incorporate large language models into a multi-modal architecture for this task. It achieves impressive results on the DAIC-WOZ AVEC 2016 Challenge train/validation/test split and Leave-One-Subject-Out cross-validation split, surpassing all baseline models and multiple state-of-the-art models. In Leave-One-Subject-Out testing, it achieves an accuracy of 91.01%, an F1-Score of 85.95%, a precision of 80%, and a recall of 92.86%.

Index Terms—Multi-Modal Neural Networks, Deep Learning, Large Language Models, Depression Diagnosis, Biomedical Informatics

I. INTRODUCTION

# M

AJOR Depressive Disorder (MDD) is a pervasive mental health condition that affects 300 million people

worldwide [1]. The COVID-19 pandemic has further exacerbated this issue, leading to a staggering 27% increase in the global prevalence of MDD [2].

Unlike biological disorders, MDD is not diagnosed through blood tests or imaging. It lacks these gold standards. Instead, the most common approach for MDD diagnosis relies on clinical interviews [3] and self-reported inventories (SRIs) [4], [5]. These systems have been subject to criticism for their subjectivity [6]–[8], which gives way for a number of issues. Bias, for example, can come from both parties. In clinical interviews and SRIs, patients may exaggerate or under-report symptoms in social desirability bias [9], [10]. Confirmation bias affects how clinicians weigh a patient’s symptoms against the assigned criteria [11]. As a result, the misdiagnosis rate of MDD is estimated to be as high as 54% [12].

In recent years, there has been a growing interest in the use of machine learning (ML) systems to automatically assess the presence and severity of MDD. This offers a low-cost and objective alternative to current methods. A heavily researched use case of ML involves making diagnoses from clinical interview recordings.

Numerous studies investigate a multi-modal approach, which combines both verbal and non-verbal cues to reach a diagnosis. These models achieve extremely high levels of success. Currently, the majority of state-of-the-art models take in three modes of input: Audio, video, and text-based data.

The text-based modality is generally seen as the weakest link in diagnosing MDD from clinical interviews [13]. This can be attributed to the scarcity of task-specific text-based training data. This lack of data hinders the effective training of NLP models, which are data-intensive, resulting in suboptimal performance compared to the audio and video modalities.

To date, no research has attempted to integrate Large Language Models (LLMs) into a multi-modal architecture for this task. Due to their training on large corpora, we hypothesized incorporating LLMs into such an architecture would improve the accuracy of depression diagnosis from clinical interview recordings and alleviate the issue of scarce training data.

In this work, we make the following contributions:

- 1) We propose a novel, tri-modal architecture that utilizes LLMs. This is the first work to incorporate LLMs into a multi-modal framework for this task.
- 2) We demonstrate that LLMs are effective for mental health diagnosis when incorporated into multi-modal architectures. The proposed architecture outperforms baseline and state-of-the-art models tested on the AVEC 2016 Challenge dataset and Leave-One-Subject-Out CrossValidation. The results of this work serve as an indicator for the potential of such an architecture in classifying depressed patients.
- 3) The proposed architecture is integrated into a locally hosted web application to emulate the potential use of such a model in the real-world.

II. RELATED WORKS

A. Mel Frequency Cepstral Coefficients in Audio-Based Models

Speech data has been found to be highly effective, more so than video and text-based data, for diagnosing depression from clinical interview recordings. In fact, multiple monomodal audio-based models have achieved results comparable to SRIs for diagnosing a variety of psychiatric disorders, such as Generalized Anxiety Disorder (GAD) [14]–[16].

Current audio-based diagnosis methods rely on low and high level features extracted from raw audio data. The extracted features are passed as input into deep learning models.

Many representations of audio data have been utilized for this task. (Wang et al.) [17] evaluated and compared the effectiveness of such representations, finding Mel Frequency Cepstral Coefficients (MFCCs) to be the best performing

[Figure 1]

Fig. 1: Example of Facial Action Unit descriptors in famous photos [18]. They can code nearly any anatomically possible facial expression.

TABLE I: State-of-the-art multi-modal fusion systems for binary depression classification. (*Classification done with LeaveOne-Subject-Out Cross-Validation

|Paper<br><br>|Architecture|Fusion Level|Reported Performance<br><br>|Modalities|Features|
|---|---|---|---|---|---|

|(Gimeno-G´omez et al.) [19]<br><br>|Transformer<br><br>|Late<br><br>|F1-Score: 72 Precision: 72 Recall: 72|Audio Video<br><br>|COVAREP [20] Vocal Tract Resonance Frequencies [21] 3D Facial Landmarks FAUs Gaze Tracking Head Pose Landmarks|
|---|---|---|---|---|---|

|(Muzammel et al.) [13]|LSTM|Model|Accuracy: 77.16 Precision: 53 Recall: 44<br><br>|Audio Video|MFCCs FAUs<br><br>|
|---|---|---|---|---|---|

|(Yang et al.) [22]|BiLSTM<br><br>|Model<br><br>|Accuracy: 81.10 F1-Score: 80.60 Precision: 80.20 Recall: 81|Video Audio Text<br><br>|FAUs COVAREP BERT [23]<br><br>|
|---|---|---|---|---|---|

|(Othmani et al.) [24]<br><br>|CNN|Model<br><br>|Accuracy*: 87.40 F1-Score*: 82.30|Video Audio<br><br>|FAUs VGGish [25]<br><br>|
|---|---|---|---|---|---|

|(Ceccerali et al.) [26]|BiLSTM|Late<br><br>|F1-Score: 87 F1-Score: 70 Precision: 89<br><br>|Audio Video Text<br><br>|Fisher Vector [27] Fisher Vector Bag of Words|
|---|---|---|---|---|---|

|(Wei et al.) [28]<br><br>|ConvBiLSTM<br><br>|Model|F1-Score: 0.70 Precision: 0.89<br><br>|Video Audio Text<br><br>|3D Facial Landmarks Gaze Tracking Log-Mel Spectrogram Sentence Embeddings|
|---|---|---|---|---|---|

audial feature for depression diagnosis of the 30 features which were tested in the study.

Derived using the Mel Scale [29], MFCCs provide a compact representation of the human perception of sound. This allows for the low-level analysis of timbre, pitch, and rhythm. Studies indicate such values are different between depressed and non-depressed patients [30]. (Wang et al.) [31] and (Taguchi et al.) [32] found certain MFCCs follow a consistent pattern that is only present in depressed patients.

Currently, MFCCs are the most used feature across stateof-the-art models to represent the audio mode for this task.

B. Facial Action Units in Video-Based Models

Research has shown facial expressions and movements are significantly different in depressed patients as opposed to nondepressed ones [33] [34]. Visual-based depression diagnosis

models rely on features extracted from patient videos and images.

Facial Action Units (FAUs), a low-level representation of the movements of specific facial muscles, were introduced as part of the Facial Action Coding System [35]. FAUs allow us to objectively study facial expressions and decipher non-verbal cues (Figure 3). This makes them highly effective and popular for affective computing tasks [36]. In this study, a subset of 20 out of 44 total FAUs are used, as shown in Table IV.

C. Multi-Modal Models

Multi-modal models combine both verbal and non-verbal cues to reach a diagnosis. To our knowledge, the majority of state-of-the-art models for depression diagnosis from clinical interview recordings consider at least audio and video data in making a diagnosis.

[Figure 2]

Fig. 2: Visualization of Early Fusion (left), Late Fusion (middle), and Model-Level Fusion (right) systems [37].

1) Data Fusion Strategies: The way in which the modalities are fused is referred to as the fusion strategy (Figure 2. Two basic fusion strategies which are highly prominent in machine learning models are Early Fusion and Late Fusion [38].

- • Early Fusion: Representations of different modalities are concatenated before being fed into a neural network. For example, one work combines unprocessed audio and text features before passing them into a Support Vector Machine. However, this method can produce a highdimensional feature representation, resulting in the ’curse of dimensionality’, where the volume of space increases exponentially and training data becomes sparse [39].
- • Late Fusion: Individual mono-modal models are trained for their respective modalities and their outputs are concatenated right before the final decision. (Samareh et al.) [40] employs Late Fusion for MDD diagnosis by evaluating outputs from mono-modal video, audio, and text-based models using Random Forests. When Late Fusion techniques are used, however, deeper complexities and relationships between the different modalities are lost.

Model-Level Fusion is a combination of these two strategies. In Model-Level Fusion, individual modalities are processed before concatenation. After concatenation, the data is processed even further to reach an output. This mitigates the curse of dimensionality and issue of uncaptured complexities between modalities. It allows patterns within a single modality and relationships across modalities to be learned effectively. For this reason, it is the most commonly used and best performing fusion method for multi-modal depression diagnosis from clinical interviews.

III. DATA COLLECTION AND PREPROCESSING A. DAIC-WOZ

The Distress Analysis Interview Corpus - Wizard of Oz (DAIC-WOZ) contains data collected from 189 clinical interviews ranging between 7-33 minutes [41]. Each interview contains a 16kHz .wav recording, a .CSV text transcript, an ID numbered 300 – 492, and features collected from video data. To keep data de-identified, raw video data is not made available in the DAIC-WOZ. Instead, the dataset provides six visual features extracted in 30 frames per second using the OpenFace pose estimation library [42]. The extracted visual features are as follows: 68 2D Facial Landmarks, 68 3D Facial Landmarks, Histogram of Oriented Gradients, Facial Action

Units (FAUs), 4 Gaze Vectors, and Head Pose. FAUs are the only visual feature considered in this study.

In the DAIC-WOZ, depression was assessed using an SRI: Patient Health Questionnare-8 (PHQ-8). The PHQ-8 measures 8 items on a scale of 0 to 3 based on how often problems have been occuring over a two-week period. The final output is a binary label and a score measuring the severity of MDD on a scale between 0-24. A cut-off score of 10 (inclusive) was used for the binary label, where participants scoring higher were considered depressed. These classifications were used to develop, train, and evaluate the proposed architecture.

Access to DAIC-WOZ was granted after signing an End User License Agreement (EULA)1.

- B. Dataset Errors

The DAIC-WOZ dataset contains many errors that needed to be resolved before the regular preprocessing steps could be applied. The following errors were resolved:

- 1) All audio files contained interactions between research assistants prior to the interview starting. These pieces of audio were identified and removed.
- 2) The interviews with IDs of 373 and 444 contained long interruptions which had to be removed. One such interruption, for example, was the phone of a research assistant ringing.
- 3) The interviews with IDs of 451, 458, and 480 were missing the utterances of the therapist. This issue was resolved by manually transcribing the therapist speech.
- 4) The interviews with IDs of 318, 321, 341, and 362 contained transcription files which were out of sync with the audio. This was resolved by manually adjusting the timestamps of the affected utterances.
- 5) The interview with ID 409 contained a labeling error, where the PHQ-8 score was 10 but the binary label was

0. This was manually resolved.

- C. Text Preprocessing

The DAIC-WOZ contains transcripts as tab-separated .CSV files with the following columns: speaker, start time, stop time, and value. The provided utterances contained numerous errors that needed to be fixed.2 Preprocessing was applied using RegEx and NLP techniques to make text suitable

- 1Zipped DAIC-WOZ files can be found after signing the necessary EULA at https://dcapswoz.ict.usc.edu/.
- 2Documentation detailing how the DAIC-WOZ can be found online.

[Figure 3]

Fig. 3: Visualization of the pre-processing steps applied to DAIC-WOZ data.

TABLE II: Transcript Sample From DAIC-WOZ

|ID<br><br>|Speaker<br><br>|Utterance|
|---|---|---|
|0<br><br>|Ellie<br><br>|how doingV (so how are you doing today)<br><br>|
|1<br><br>|Participant|good|
|2<br><br>|Participant<br><br>|it’s been a nice day|
|3|Ellie<br><br>|thats good (that’s good)<br><br>|
|4<br><br>|Ellie<br><br>|where originally (where are you from originally)<br><br>|

TABLE III: Adjusted Transcript Sample

|ID<br><br>|Speaker|Utterance|
|---|---|---|
|0|Therapist<br><br>|So how are you doing today?|
|1<br><br>|Patient|Good. It’s been a nice day.|
|2|Therapist|That’s good. Where are you from originally?|

to pass into the LLM. Refer to Table II for a sample of a transcript before pre-processing is applied.

- A major issue in the unprocessed transcripts is the presence

of unique identifiers within utterances, as seen in Table II. In the original DAIC-WOZ dataset, all transcriptions of the clinical interviewer were automatically generated for participants whose IDs were greater than 363. In such cases, a unique identifier is followed by the true text in parenthesis. These identifiers are heavily present in Table. RegEx was applied to identify and replace unique identifiers with their true meanings.

In the DAIC-WOZ, acronyms which are pronounced letter by letter are connected by underscores, e.g. ’l a’ represents ’Los Angeles’. A Python dictionary was generated, where the keys were such acronyms and the values were their extended form. All acronyms were replaced with what they represented.

If speech is cut-off, the full word is followed by what was actually pronounced, enclosed in inequality signs. Non-speech, such as coughing, is enclosed in inequality signs as well. These issues were resolved by simply removing text that followed this pattern using RegEx.

The use of proper grammar in LLM prompting can increase classification accuracy, but the transcripts provided in DAIC-

WOZ contain major grammatical errors. To resolve this issue, the ’t5-base-grammar-correction’ model [43] was applied to all text through the Hugging Face platform. The model ensured the proper use of punctuation and capitalization in every utterance. However, the model is limited in its ability to identify interrogative sentences and apply punctuation accordingly. A Support Vector Machine was trained to identify if utterances are interrogative and to apply the appropriate punctuation with an accuracy of 97%.

The names of diarized speakers were adjusted. ’Ellie’ was replaced with ’Therapist’ and ’Participant’ was replaced with ’Patient’. Using such naming conventions gave the LLM a better understanding of a conversation’s context and the speakers’ respective roles.

Following all corrections, subsequent utterances by the same speaker were merged together. Refer to Table III for a sample of a fully adjusted transcript.

The resulting .CSV files were converted into raw text data which could be fed into the LLM. In this raw text format, subsequent utterances were separated by a new-line. Speakers were diarized by preceding utterances with the speaker name, followed by a colon (e.g., ’Therapist: So how are you doing today?’).

D. Audio Preprocessing

Audio preprocessing followed four stages. (1) Audio was segmented to only include patient speech. (2) Audio data was augmented to increase the number of total samples by a factor of seven. (3) MFCCs were extracted using Librosa [44]. (4) The MFCCs were normalized using Cepstral Mean and Variance Normalization.

1) Audio Segmentation: In the proposed model, as well as past research, the audio modality is incorporated by analyzing patient responses to questions. Patient speech was separated from therapist speech by slicing audio according to the timestamps provided in the transcript, which are detailed down to

Speech Waveform

Windowing

Shift it into FFT order

Find the magnitude of FFT

Convert the FFT data into filter bank outputs

Find the log base 10

Find the cosine transform to reduce dimensionality

MFCC Vector

Fig. 4: Visualization of process used to derive MFCCs. The way in which MFCCs were derived in this work is based off the implementation provided by the Slaney auditory toolbox [45].

the centisecond. The cropped therapist audio was discarded and not used as input into the final architecture.

The remaining audio data was separated into smaller chunks to make it suitable for input into Bidirectional Long Short Term Memory (BiLSTM) layers. It was sliced into 8-second segments, where any remaining segment under 8 seconds was discarded.

As first done in (Muzammel et al.) [13], audio data was augmented through pitch shifting and noise injection strategies to prevent over-fitting and data scarcity. In this work, we combined and randomly applied the following augmentation strategies to increase sample size seven-fold:

- • Pitch Shifting: Pitch was adjusted by 0.5, 2.0, and 2.5 in semitones.
- • Noise Injection: A NumPy array with randomly assigned values was overlaid onto audio to create noise.

- 2) Audio Feature Extraction: MFCCs were extracted from

raw audio data using the Librosa Python library. 60 MFCCs were extracted from 124ms windows with a 92ms overlap. There were 15,420 total coefficients for every 8-second audio segment. Figure 4 displays the pipeline used to derive the MFCCs.

- 3) MFCC Normalization: Rather than normalizing raw

audio data, the extracted MFCCs were normalized using Cepstral Mean and Variance Normalization (CMVN). MFCCs are highly sensitive to additive noise and differences in recording conditions. CMVN solves the issue of signal variations due to different recording conditions in MFCCs by normalizing the spectrum to have zero mean and unit variance. Past research

TABLE IV: Facial Action Units Available in DAIC-WOZ.

|No.| |Feature| |Facial Feature Name|
|---|---|---|---|---|
|1| |AU01 r<br><br>| |Inner Brow Raiser|
|2| |AU02 r<br><br>| |Outer Brow Raiser|
|3| |AU04 r<br><br>| |Brow Lowerer|
|4| |AU05 r<br><br>| |Upper Lid Raiser|
|5| |AU06 r<br><br>| |Cheek Raiser|
|6| |AU09 r<br><br>| |Nose Wrinkler|
|7| |AU10 r<br><br>| |Upper Lip Raiser|
|8| |AU12 r<br><br>| |Lip Corner Puller|
|9| |AU14 r<br><br>| |Dimpler|
|10| |AU15 r<br><br>| |Lip Corner Depressor|
|11| |AU17 r<br><br>| |Chin Raiser|
|12| |AU20 r<br><br>| |Lip Stretcher|
|13| |AU25 r<br><br>| |Lips Part|
|14| |AU26 r<br><br>| |Jaw Drop|
|15| |AU04 c<br><br>| |Brow Lowerer|
|16| |AU12 c<br><br>| |Lip Corner Puller|
|17| |AU15 c<br><br>| |Lip Corner Depressor|
|18| |AU23 c<br><br>| |Lip Tightener|
|19| |AU28 c<br><br>| |Lip Suck|
|20| |AU45 c<br><br>| |Blink|

has shown applying CMVN to MFCCs greatly improves the performance of speech recognition algorithms [46].

E. Visual Preprocessing

Visual preprocessing followed two stages. (1) Video data was segmented to only include data from when the patient is speaking. (2) Individual columns were normalized for zero mean and unit variance.

- 1) Video Segmentation: In the proposed architecture, we

analyze patient visual data only while they are speaking, not in all parts of the interview. Using the timestamps provided in the transcript, FAU data was cropped to exclude and discard segments where the therapist is speaking, or where nobody is speaking at all. The data was then sliced into 8-second segments in such a way that it aligned with the segmented audio data. Any remaining segment under 8 seconds was discarded. After video segmentation was applied, there were an equal number of video and audio samples for each interview.

- 2) FAU Normalization: Normalization was applied to con-

tinuous, non-discrete FAU values. Namely, numbers 1-14 in Table IV. The continuous FAU values were scaled to zero mean and unit variance respectively.

IV. MODEL DEVELOPMENT A. Text-Based Model Development

Two-shot learning was employed with GPT-4 through the OpenAI API to reach a classification given full text transcripts from the DAIC-WOZ.

The model was given the prompt ’Take on the role of an expert in psychiatric diagnosis using the DSM 5. Read the following transcript and determine if the patient has depression.’ This prompt is a modified version of that presented in (Galatzer-Levly et al.) [47]. Given this prompt, GPT-4 was expected to output a binary classification of ’depressed’ or ’not depressed’ in JSON format. The function calling feature provided in the OpenAI API was manipulated to force an output in the desired JSON format in almost all cases.

[Figure 4]

Fig. 5: Proposed model architecture following Hyperband tuning.

- TABLE V: Hyperparameters optimized through Hyperband

| | |Options|Result|
|---|---|---|---|
|BiLSTM 1 Units| |512, 256<br><br>|256|
|Dropout 1 Rate| |0.1, 0.3, 0.5<br><br>|0.5|
|BiLSTM 2 Units| |128, 256|128|
|Dropout 2 Rate| |0.1, 0.3, 0.5|0.3|
|BiLSTM 3 Units| |64, 128<br><br>|64|
|Dropout 3 Rate| |0.1, 0.3, 0.5|0.5|
|Number of Dense Layers| |4, 5, 6|4|

[Figure 5]

Fig. 6: Visualization of LOSOCV

The examples used as part of few-shot learning consisted of four full-length text transcripts from the DAIC-WOZ dataset. Two of the provided examples had a ground truth of depressed, the other two had a ground truth of not depressed.

- B. Tri-Modal Model Development

Model development followed two stages. (1) The model architecture was optimized with the Hyperband Tuning Algorithm [48] in Keras-Tuner [49]. (2) The model parameters were reset and the optimized architecture was evaluated through Leave-One-Subject-Out Cross-Validation (LOSOCV).

1) Hyperparameter Tuning: The model was trained with the Adam optimizer [50] and binary cross entropy as the loss function. To account for the class imbalance of the DAIC-

WOZ, the loss function was weighted to give higher value to the depressed class. To achieve this functionality, part of the Keras-Tuner Python library source code was rebuilt.

The tuner was set to focus on validation loss as the only metric to gauge configuration performance. Through each bracket, the number of total epochs run and models tested was reduced by a factor of three. A callback was created to end training and move on to the next configuration if validation loss did not improve over three consecutive epochs. Two iterations of the full Hyperband algorithm were run.

2) Evaluation: LOSOCV (Figure 6) was applied by first resetting the optimized architecture’s trainable parameters which had been adjusted during hyperparameter tuning. A unique model was trained for every clinical interview in the dataset (n=180). For each model, all samples originating from one clinical interview were excluded from the training dataset. Each model was tested on the excluded clinical interivew, and all the model results were pooled for the final performance calculation. Augmented samples were not included in testing data.

C. Proposed Architecture

The final model architecture (Figure 5), which has been optimized by the Hyperband Tuning Algorithm, follows 7 steps to reach a classification.

- 1) The model extracts higher level features from the MFCCs in three consecutive blocks consisting of BiLSTM, Dropout, and Batch Normalization layers.
- 2) The input FAU data follows three processing steps to make it suitable for concatenation with the MFCC tensor: (a) The FAU data is first flattened into a 1D tensor. (b) The Keras [53] Repeat Vector layer repeats the flattened data for each timestep present in the processed

D

Predicted

ND

Actual D ND

|16 (34.04)|
|---|

|1 (2.13)|
|---|

|3 (6.38)|
|---|

|27 (57.45)|
|---|

Actual D ND total

|52 (27.51)|
|---|

|13 (6.88)|
|---|

D

79.99

Predicted

|4 (2.12)|
|---|

|120 (63.49)|
|---|

ND

96.77

total 92.85 90.22

Fig. 7: Confusion matrices of proposed architecture’s performance on AVEC 2016 train/validation/test split (left) and LOSOCV (right).

- TABLE VI: Comparison of proposed architecture with baseline binary depression classification models tested on the AVEC 2016 train/validation/test split.

|Paper<br><br>|Precision|Recall<br><br>|F1-Score|Accuracy|
|---|---|---|---|---|

|(Yang et al.) - Logistic Regression (Yang et al.) - Naive Bayes (Yang et al.) - Random Forest (Valstar et al.) (Qureshi et al.) Proposed Model<br><br>|53.20 56.70 54.80 57.90 58.00 92.86|55.80 58.40 55.20 59.60 61.00 68.42<br><br>|54.47 57.54 55.00 58.74 59.46 78.79<br><br>|54.12 57.02 58.41 58.41 61.11 85.11|
|---|---|---|---|---|

MFCC tensor, resulting in a 3D tensor. (c) Lastly, the Time Distributed operation applies a Dense layer to each temporal slice, where the number of neurons is the number of feature dimensions in the MFCC tensor. These operations result in a higher-level representation of the FAUs with the same dimensions as the processed MFCC data, allowing concatenation in the following step.

- 3) The processed FAU and MFCC data are concatenated along the features axis.
- 4) The concatenated data is processed by consecutive BiLSTM, Dropout, and Batch Normalization layers.
- 5) The same sequence of operations applied in step two is applied to the LLM binary output to prepare it for concatenation with the MFCC-FAU tensor.
- 6) The MFCC-FAU and LLM tensors are concatenated.
- 7) The combined tensor is processed by four consecutive Leaky ReLU-activated Dense layers and one sigmoidactivated layer. A binary classification is output.

V. RESULTS The proposed architecture was evaluated in two ways:

- 1) The model was evaluated using the DAIC-WOZ AVEC 2016 Challenge train/validation/test split to allow comparability with models from other studies.
- 2) The proposed architecture was evaluated through LOSOCV.

The proposed model outperforms all baseline models and multiple state-of-the-art models which were evaluated using no cross-validation split as well as LOSOCV. It performs

well on both the depressed and non-depressed classes, with accuracies of 92.9% and 90.2% on LOSOCV for the depressed and non-depressed classes, respectively. The large difference in the accuracy between the depressed and non-depressed classes (92.9% and 81.81%, respectively) on the AVEC 2016 train/validation/test split can be attributed to the extremely small sample size of just 14 for the depressed class. We believe the accuracy of 92.9% for the depressed class on the AVEC 2016 Challenge cross-validation split is inflated as a result of this sample size.

A. Computational Complexity

The speed of the proposed architecture was evaluated on the 12-minute 50-second DAIC-WOZ clinical interview with an ID of 301. The test was conducted using an Intel i7-12650H @ 2.30GHZ processor, NVIDIA RTX 4060 graphics card, and 64GB of RAM. The model took 2.67 seconds to process the interview, including GPT-4 computation time through the OpenAI API.

VI. DEPLOYMENT

The proposed architecture was integrated into a locally hosted web application, DepScope, for easy and accessible use of the model by clinicians. This application mimics how such a model may be implemented into real-world scenarios.

Clinicians first connect their Zoom account to the web application and agree to the necessary terms, consenting that their clinical interview recordings may be processed by the DepScope applicatoin. This is done through the Zoom Application Programming Interface (API). From there, any time a

- TABLE VII: Comparison of proposed architecture with state-of-the-art binary depression classification models tested on the AVEC 2016 train/validation/test split. A dash signifies that the metric was not reported in the paper. (*Evaluation done with LOSOCV)

Paper Precision Recall F1-Score Accuracy D ND D ND

|(Gimeno-G´omez et al.) (Muzammel et al.) (Yang et al.) (Huang et al.) Proposed Model|72 53 80.20 94 92.86<br><br>|– 83 – – 96.43<br><br>|72 44 81 95 68.42|– 88 – – 81.82<br><br>|72 – 80.60 94 78.79|– 77.16 81.10 96 85.11<br><br>|
|---|---|---|---|---|---|---|

|(Muzammel et al.)* (Othmani et al.)* (Salekin et al.)* Proposed Model*<br><br>|96 – – 80|95 – – 90.23<br><br>|89 – – 92.86<br><br>|98 – – 96.77<br><br>|95.48 82.30 85.44 85.95<br><br>|95.38 87.40 96.7 91.01|
|---|---|---|---|---|---|---|

[Figure 6]

Fig. 8: Interview collection and processing pipeline

clinical interview concludes over Zoom, a webhook [56] is sent to the DepScope backend. The recording is then retrieved from the Zoom servers for preprocessing, model inference, and clinical report generation via GPT-4. Clinical reports contain a summary of the interview, detailing all key points and a justification for why the diagnosis was drawn (Figure 10). Reports additionally contain the confidence of the model in the classification, which is output by the sigmoid activation function. Reports populate the clinician dashboard (Figure 9).

VII. CONCLUSION

This paper presents a novel machine learning architecture for diagnosing MDD from clinical interview recordings. We explored the integration of large language models and audiovisual features into a tri-modal architecture for this task. The

[Figure 7]

Fig. 9: Clinician’s dashboard

proposed architecture was optimized through automated hyperparameter tuning techniques and evaluated using the AVEC 2016 Challenge train/validation/test split and LOSOCV. We drew comparisons between the performance of the proposed architecture and previously developed models, finding that it outperforms all baseline models and multiple state-of-the-art models. The model was integrated into a locally hosted web application to emulate how it may be used in the real-world.

Though the proposed model achieved impressive results, the current architecture faces multiple limitations. The speed of the model suggests that it is not appropriate for realtime applications and may only be effective when integrated with batch-based data processing techniques. Due to the use of large language models in this architecture, it is highly unlikely this can be resolved. The dataset used in this study, the DAIC-WOZ, is small and homogenous. We believe this has negatively impacted the generalizability of the proposed model and may have resulted in metrics which do not reflect the effectiveness of the model in real-world settings. The lack of high-quality data available for this task is an issue which plagues the field as the whole.

In future work, we aim to further optimize the proposed

[Figure 8]

Fig. 10: Sample clinical report

architecture for increased accuracy and make the model easily accessible to use. As new large language models are developed, they will be integrated into the model architecture and evaluated. We plan to integrate existing language models into the architecture as well, such as Med-PaLM 2 [57]. The prompt given to the large language model will be improved through iterative prompting techniques.

ACKNOWLEDGMENTS

The author would like to extend thanks to the SMU Lyle School of Engineering and the American Psychological Association for their cash award which funded this research.

REFERENCES

- [1] P. Chodavadia, I. Teo, D. Poremski, D. S. S. Fung, and E. A. Finkelstein, “Prevalence and economic burden of depression and anxiety symptoms among singaporean adults: results from a 2022 web panel,” BMC Psychiatry, vol. 23, p. 104, 2 2023.
- [2] D. F. Santomauro, A. M. M. Herrera, J. Shadid, P. Zheng, C. Ashbaugh,

- D. M. Pigott, C. Abbafati, C. Adolph, J. O. Amlag, A. Y. Aravkin, B. L. Bang-Jensen, G. J. Bertolacci, S. S. Bloom, R. Castellano,
- E. Castro, S. Chakrabarti, J. Chattopadhyay, R. M. Cogen, J. K. Collins, X. Dai, W. J. Dangel, C. Dapper, A. Deen, M. Erickson, S. B. Ewald,

- A. D. Flaxman, J. J. Frostad, N. Fullman, J. R. Giles, A. Z. Giref, G. Guo, J. He, M. Helak, E. N. Hulland, B. Idrisov, A. Lindstrom, E. Linebarger, P. A. Lotufo, R. Lozano, B. Magistro, D. C. Malta, J. C. M˚ansson, F. Marinho, A. H. Mokdad, L. Monasta, P. Naik, S. Nomura, J. K. O’Halloran, S. M. Ostroff, M. Pasovic, L. Penberthy, R. C. R. Jr, G. Reinke, A. L. P. Ribeiro, A. Sholokhov, R. J. D. Sorensen, E. Varavikova, A. T. Vo, R. Walcott, S. Watson, C. S. Wiysonge,
- B. Zigler, S. I. Hay, T. Vos, C. J. L. Murray, H. A. Whiteford, and A. J. Ferrari, “Global prevalence and burden of depressive and anxiety disorders in 204 countries and territories in 2020 due to the covid-19 pandemic,” The Lancet, vol. 398, pp. 1700–1712, 11 2021.

- [3] J. Guidi, G. A. Fava, P. Bech, and E. Paykel, “The clinical interview for depression: A comprehensive review of studies and clinimetric properties,” Psychotherapy and Psychosomatics, vol. 80, pp. 10–27, 2011.

- [4] L. S. Radloff, “The ces-d scale,” Applied Psychological Measurement, vol. 1, pp. 385–401, 6 1977.
- [5] K. Kroenke, R. L. Spitzer, and J. B. W. Williams, “The phq-9,” Journal of General Internal Medicine, vol. 16, pp. 606–613, 9 2001.
- [6] Allsopp, K., Read, J., Corcoran, R. & Kinderman, P. Heterogeneity in psychiatric diagnostic classification. Psychiatry Res.. 279 pp. 15–22, 9 2019.
- [7] Silva, P. How to improve psychiatric services: a perspective from critical psychiatry. Br. J. Hosp. Med. (Lond.). 78, 503–507, 9 2017.
- [8] Fuchs, T. Subjectivity and intersubjectivity in psychiatric diagnosis. Psychopathology. 43, 268–274, 5 2010.
- [9] A. Althubaiti, “Information bias in health research: definition, pitfalls, and adjustment methods,” Journal of Multidisciplinary Healthcare, p. 211, 5 2016.
- [10] C. A. Latkin, C. Edwards, M. A. Davey-Rothwell, and K. E. Tobin, “The relationship between social desirability bias and self-reports of health, substance use, and social network factors among urban substance users in baltimore, maryland,” Addictive Behaviors, vol. 73, pp. 133–136, 10 2017.
- [11] R. Mendel, E. Traut-Mattausch, E. Jonas, S. Leucht, J. M. Kane, K. Maino, W. Kissling, and J. Hamann, “Confirmation bias: why psychiatrists stick to wrong preliminary diagnoses,” Psychological Medicine, vol. 41, pp. 2651–2659, 12 2011.
- [12] G. Ayano, S. Demelash, Z. yohannes, K. Haile, M. Tulu, D. Assefa, A. Tesfaye, K. Haile, M. Solomon, A. Chaka, and L. Tsegay, “Misdiagnosis, detection rate, and associated factors of severe psychiatric disorders in specialized psychiatry centers in ethiopia,” Annals of General Psychiatry, vol. 20, p. 10, 2 2021.
- [13] M. Muzammel, H. Salam, and A. Othmani, “End-to-end multimodal clinical depression recognition using deep neural networks: A comparative analysis,” Computer Methods and Programs in Biomedicine, vol. 211, p. 106433, 11 2021.
- [14] X. Ma, H. Yang, Q. Chen, D. Huang, and Y. Wang, “Depaudionet.” ACM, 10 2016, pp. 35–42.
- [15] R. Brueckner, N. Kwon, V. Subramanian, N. Blaylock, and H. O’Connell, Audio-Based Detection of Anxiety and Depression via Vocal Biomarkers, 2024, pp. 124–141.
- [16] P. Zhang, M. Wu, H. Dinkel, and K. Yu, “Depa.” ACM, 10 2021, pp. 135–143.
- [17] J. Wang, L. Zhang, T. Liu, W. Pan, B. Hu, and T. Zhu, “Acoustic differences between healthy and depressed people: a cross-situation study,” BMC Psychiatry, vol. 19, p. 300, 12 2019.
- [18] C.-H. Tu, C.-Y. Yang, and J. Y. jen Hsu, “Idennet: Identity-aware facial action unit detection.” IEEE, 5 2019, pp. 1–8.

- [19] D. Gimeno-G´omez, A.-M. Bucur, A. Cosma, C.-D. Mart´ınez-Hinarejos, and P. Rosso, Reading Between the Frames: Multi-modal Depression Detection in Videos from Non-verbal Cues, 2024, pp. 191–209.
- [20] Degottex, G., Kane, J., Drugman, T., Raitio, T., and Scherer, S. “COVAREP — A collaborative voice analysis repository for speech technologies.“ 2014 IEEE International Conference On Acoustics, Speech And Signal Processing (ICASSP), 5 2014.
- [21] Sataloff, R. Vocal Health and pedagogy: Science, assessment, and treatment, Third Edition. (Plural Publishing), 9 2017.
- [22] S. Yang, L. Cui, L. Wang, T. Wang, and J. You, “Enhancing multimodal depression diagnosis through representation learning and knowledge transfer,” Heliyon, vol. 10, p. e25959, 2 2024.
- [23] Devlin, J., Chang, M., Lee, K., and Toutanova, K. BERT: Pretraining of deep bidirectional Transformers for language understanding. (arXiv,2018)
- [24] A. Othmani, A.-O. Zeghina, and M. Muzammel, “A model of normality inspired deep learning framework for depression relapse prediction using audiovisual data,” Computer Methods and Programs in Biomedicine, vol. 226, p. 107132, 11 2022.
- [25] Simonyan, K., and Zisserman, A. “Very deep convolutional networks for large-scale image recognition.“ arXiv, 2014.
- [26] F. Ceccarelli, M. Mahmoud, “Multimodal temporal machine learning for Bipolar Disorder and Depression Recognition,“ Pattern Anal. Appl.. 25, 493-504, 8 2022.
- [27] Perronnin, F., S´anchez, J. & Mensink, T. Improving the fisher kernel for large-scale image classification. Computer Vision – ECCV 2010. pp. 143-156 (2010)
- [28] P. Wei, K. Peng, A. Roitberg, K. Yang, J. Zhang, and R. Stiefelhagen, “Multi-modal Depression Estimation based on Sub-attentional Fusion.,“ arXiv, 2022
- [29] Pedersen, P. The Mel scale. J. Music Theory. 9, 295 (1965)
- [30] S. A. Almaghrabi, S. R. Clark, and M. Baumert, “Bio-acoustic features of depression: A review,” Biomedical Signal Processing and Control, vol. 85, p. 105020, 8 2023.
- [31] Wang, Y., Liang, L., Zhang, Z., Xu, X., Liu, R., Fang, H., Zhang, R., Wei, Y., Liu, Z., Zhu, R., Zhang, X. & Wang, F. Fast and accurate assessment of depression based on voice acoustic features: a crosssectional and longitudinal study. Front. Psychiatry. 14 pp. 1195276 (2023,6)
- [32] T. Taguchi, H. Tachikawa, K. Nemoto, M. Suzuki, T. Nagano, R. Tachibana, M. Nishimura, and T. Arai, “Major depressive disorder discrimination using vocal acoustic features,” Journal of Affective Disorders, vol. 225, pp. 214–220, 1 2018.
- [33] S. Alghowinem, R. Goecke, M. Wagner, G. Parkerx, and M. Breakspear, “Head pose and movement analysis as an indicator of depression.” IEEE, 9 2013, pp. 283–288.
- [34] F. Schneider, H. Heimann, W. Himer, D. Huss, R. Mattes, and B. Adam, “Computer-based analysis of facial action in schizophrenic and depressed patients,” European Archives of Psychiatry and Clinical Neuroscience, vol. 240, pp. 67–76, 11 1990.
- [35] Ekman, P. & Friesen, W. Facial Action Coding System. (American Psychological Association (APA),2019,1), Title of the publication associated with this dataset: PsycTESTS Dataset
- [36] Jiang, Z., Luskus, M., Seyedi, S., Griner, E., Rad, A., Clifford, G., Boazak, M. & Cotes, R. Utilizing computer vision for facial behavior analysis in schizophrenia studies: A systematic review. PLoS One. 17, e0266828 (2022,4)
- [37] Borges, P., Peynot, T., Liang, S., Arain, B., Wildie, M., Minareci, M., Lichman, S., Samvedi, G., Sa, I., Hudson, N., Milford, M., Moghadam, P. & Corke, P. A survey on terrain traversability analysis for autonomous ground vehicles: Methods, sensors, and challenges. Field Robotics. 2, 1567-1627 (2022,3)
- [38] M. Pawłowski, A. Wr´oblewska, and S. Sysko-Roma´nczuk, “Effective techniques for multimodal data fusion: A comparative analysis,” Sensors, vol. 23, p. 2381, 2 2023.
- [39] M. Asgari, I. Shafran, and L. B. Sheeber, “Inferring clinical depression from speech and spoken utterances.” IEEE, 9 2014, pp. 1–5.
- [40] A. Samareh, Y. Jin, Z. Wang, X. Chang, and S. Huang, “Predicting depression severity by multi-modal feature engineering and fusion,” 11 2017.
- [41] J. Gratch, R. Artstein, G. Lucas, G. Stratou, S. Scherer, A. Nazarian,

- R. Wood, J. Boberg, D. DeVault, S. Marsella, D. Traum, S. Rizzo, and L.-P. Morency, “The distress analysis interview corpus of human and computer interviews,” N. Calzolari, K. Choukri, T. Declerck, H. Loftsson, B. Maegaard, J. Mariani, A. Moreno, J. Odijk, and
- S. Piperidis, Eds. European Language Resources Association (ELRA),

5 2014, pp. 3123–3128. [Online]. Available: http://www.lrec-conf.org/ proceedings/lrec2014/pdf/508 Paper.pdf

- [42] Baltrusaitis, T., Robinson, P. & Morency, L. OpenFace: An open source facial behavior analysis toolkit. 2016 IEEE Winter Conference On Applications Of Computer Vision (WACV). (2016,3)
- [43] Napoles, C., Sakaguchi, K. & Tetreault, J. JFLEG: A fluency corpus and benchmark for grammatical error correction. (arXiv,2017)
- [44] B. McFee, C. Raffel, D. Liang, D. Ellis, M. McVicar, E. Battenberg, and O. Nieto, “librosa: Audio and music signal analysis in python,” 2015, pp. 18–24.
- [45] M. Slaney, “Auditory toolbox,” Interval Research Corporation, Tech. Rep, vol. 10, p. 1194, 1998.
- [46] R. Rehr and T. Gerkmann, “Cepstral noise subtraction for robust automatic speech recognition.” IEEE, 4 2015, pp. 375–378.
- [47] I. R. Galatzer-Levy, D. McDuff, V. Natarajan, A. Karthikesalingam, and M. Malgaroli, “The capability of large language models to measure psychiatric functioning,” 8 2023.
- [48] L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, and A. Talwalkar, “Hyperband: A novel bandit-based approach to hyperparameter optimization,” 3 2016.
- [49] T. O’Malley, E. Bursztein, J. Long, F. Chollet, H. Jin, L. Invernizzi et al., “Kerastuner,” 2019.
- [50] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” 12 2014.
- [51] M. Valstar, J. Gratch, B. Schuller, F. Ringeval, D. Lalanne, M. T. Torres, S. Scherer, G. Stratou, R. Cowie, and M. Pantic, “Avec 2016.” ACM, 10 2016, pp. 3–10.
- [52] S. A. Qureshi, S. Saha, M. Hasanuzzaman, and G. Dias, “Multitask representation learning for multimodal estimation of depression level,” IEEE Intelligent Systems, vol. 34, pp. 45–52, 9 2019.
- [53] Chollet, F. & Others Keras. (GitHub,2015), https://github.com/fchollet/keras
- [54] X. Huang, F. Wang, Y. Gao, Y. Liao, W. Zhang, L. Zhang, and Z. Xu, “Depression recognition using voice-based pre-training model,” Scientific Reports, vol. 14, p. 12734, 6 2024.
- [55] A. Salekin, J. W. Eberle, J. J. Glenn, B. A. Teachman, and J. A. Stankovic, “A weakly supervised learning framework for detecting social anxiety and depression,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 2, pp. 1–26, 7 2018.
- [56] Biehl, M. Webhooks. (Createspace Independent Publishing Platform,2017,12)
- [57] Singhal, K., Tu, T., Gottweis, J., Sayres, R., Wulczyn, E., Hou, L., Clark, K., Pfohl, S., Cole-Lewis, H., Neal, D., Schaekermann, M., Wang, A., Amin, M., Lachgar, S., Mansfield, P., Prakash, S., Green, B., Dominowska, E., Arcas, B., Tomasev, N., Liu, Y., Wong, R., Semturs, C., Mahdavi, S., Barral, J., Webster, D., Corrado, G., Matias, Y., Azizi, S., Karthikesalingam, A. & Natarajan, V. Towards expert-level medical question answering with large language models. (arXiv,2023)

