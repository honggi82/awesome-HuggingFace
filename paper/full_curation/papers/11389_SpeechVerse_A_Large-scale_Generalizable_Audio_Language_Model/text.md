# arXiv:2405.08295v3[cs.CL]24Mar2025

## SpeechVerse: A Large-scale Generalizable Audio Language Model

Nilaksh Das ∗ Saket Dingliwal ∗† Srikanth Ronanki Rohit Paturi

Zhaocheng Huang Prashant Mathur Jie Yuan Dhanush Bekal Xing Niu

Sai Muralidhar Jayanthi Xilai Li Karel Mundnich Monica Sunkara

Sravan Bodapati Sundararajan Srinivasan Daniel Garcia-Romero Kyu J. Han

Katrin Kirchhoff

AWS AI Labs, Amazon

Abstract

Large language models (LLMs) have shown incredible proficiency in performing tasks that require semantic understanding of natural language instructions. Recently, many works have further expanded this capability to perceive multimodal audio and text inputs, but their capabilities are often limited to specific fine-tuned tasks such as automatic speech recognition and translation. We therefore develop SpeechVerse, a robust multi-task training and curriculum learning framework that combines pretrained speech and text foundation models via a small set of learnable parameters, while keeping the pre-trained models frozen during training. The models are instruction finetuned using continuous latent representations extracted from the speech foundation model to achieve optimal zero-shot performance on a diverse range of speech processing tasks using natural language instructions. We perform extensive benchmarking that includes comparing our model performance against traditional baselines across several datasets and tasks. Furthermore, we evaluate the model’s capability for generalized instruction following by testing on out-ofdomain datasets, novel prompts, and unseen tasks. Our empirical experiments reveal that our multi-task SpeechVerse model is even superior to conventional task-specific baselines on 9 out of the 11 tasks.

### 1 Introduction

Large language models (LLMs) [1–3] have achieved remarkable performance on a variety of natural language tasks through self-supervised pre-training on massive text corpora. They have also shown a striking ability to follow open-ended instructions from users through further instruction tuning [4–7], enabling strong generalization capabilities. Despite the success, a significant limitation lies in the inability of language models to perceive non-textual modalities such as images and audio.

∗Equal Contributions †Corresponding author: skdin@amazon.com

[Figure 1]

Figure 1: Schematic diagram of the SpeechVerse framework.

Speech in particular represents the most natural mode of human communication. Empowering LLMs to deeply understand speech could significantly enhance human-computer interaction [8] and multimodal dialog agents [9, 10]. As such, enabling LLMs to comprehend speech has received substantial attention recently. Some approaches first transcribe speech via an automated speech recognition (ASR) system and then process the text with an LLM for improved transcription [11–13]. However, such pipelines cannot capture non-textual paralinguistic and prosodic features like speaker tone, intonation, emotion, valence, etc.

A promising new paradigm directly fuses textual LLMs with speech encoders within an end-toend training framework [14, 15]. Enabling joint modeling of speech and text holds promise for richer speech and audio comprehension versus text-only methods. Particularly, instruction-following multimodal audio-language models [16–18] are increasingly receiving more attention due to their generalization capability. Despite some success, existing works in multi-task audio-language models such as SpeechT5 [19], Whisper [20], VIOLA [15], SpeechGPT [18], and SLM [17] are limited to processing a small number of speech tasks.

We therefore propose SpeechVerse, a robust multi-task framework that leverages supervised instruction finetuning to incorporate various speech tasks (see Figure 1). In contrast to SpeechGPT [18], we propose using continuous representations extracted from a self-supervised pre-trained speech foundation model, focusing on tasks that generate text-only output. More recently, [16] proposed Qwen-Audio, a multi-task audio-language model capable of perceiving human speech and sound signals and is trained on 30 tasks from diverse audio types including music and songs. However, this requires carefully designed hierarchical tagging and a large-scale supervised audio encoder for fusion making it sub-optimal for unseen speech tasks. In contrast, our training paradigm incorporates multi-task learning and supervised instruction finetuning in a unified curriculum, without the need for task-specific tagging, allowing for generalization to unseen tasks using natural language instructions.

We summarize our contributions below:

- 1. Scalable multimodal instruction finetuning for diverse speech tasks. SpeechVerse is a novel LLM-based audio-language framework to scalably exhibit strong performance on as many as 11 diverse tasks. We extensively benchmark our models on public datasets spanning ASR, spoken language understanding and paralinguistic speech tasks.
- 2. Versatile instruction following capability for novel open-ended tasks. We demonstrate the SpeechVerse model’s capability to leverage the robust language understanding of the LLM backbone in order to adapt to open-ended tasks that were unseen during multimodal finetuning.
- 3. Strategies for improving generalization to unseen tasks. We further study prompting and decoding strategies including constrained and joint decoding, that can enhance the model’s ability to generalize to completely unseen tasks, improving absolute metrics by up to 21%.

[Figure 2]

Figure 2: Block diagram of the SpeechVerse architecture.

### 2 Approach

#### 2.1 Architecture

As shown in Figure 2, our multimodal model architecture consists of three main components: (1) a pre-trained audio encoder to encode an audio signal into a feature sequence, (2) a 1-D convolution module that operates over the audio feature sequence to abbreviate the sequence length, and (3) a pre-trained LLM to use these audio features and textual instructions to perform the required task. Details of each of these sub-systems are described below.

Audio encoder: To extract semantic features from a given audio, we use a large pre-trained selfsupervised speech foundation model as an audio encoder. We can represent the audio encoder as a cascading collection of L layers, where each intermediate layer l returns a feature sequence

h(l) = fAE(h(l−1);θAE(l) ), where h(0) = x is the input audio. Here, θAE(l) represents the learned weights for layer l of the pre-trained speech model. To capture a unified representation for multi-

faceted forms of feature semantics, we compute the output of the audio encoder as: AE(x) =

L

1 L

w(l)h(l) (1)

l=1

where, the scalars {w(1),...w(L)} are a set of learnable parameters. As this approach concurrently encodes features from multiple intermediate layers of the speech foundation model, it can simultaneously capture different forms of semantics (higher-order as well as lower-order features), leading to better generalization across a diverse set of tasks. We also perform experiments by only taking the output from the last layer of the audio encoder, i.e., AE(x) = h(L).

Convolution downsampling module: LLMs trained on text-only input encode token sequences which are typically of a considerably shorter length compared to the feature sequences encoded by speech foundation models. To mitigate this large discrepancy in length distribution between audio features and text tokens, we downsample the encoded audio features through a learnable convolution module. This module consists of successive blocks, each having a 1-D convolution layer followed by layer normalization. For the 1-D convolution, we use a kernel size of 3, which ensures that each output frame has both left and right context from the input frames. In our experiments, we use as many number of these downsampling blocks as necessary such that the resulting sampling rate for the audio comes out to be 12.5 Hz, i.e., each output frame corresponds to 80ms of audio. By fine-tuning the convolution downsampling module, the audio encoder output can be transformed from an audio-only feature space to a joint audio-text semantic space. Hence, we set the number of output channels for the 1-D convolution to be equal to the feature dimension of the token embeddings for the downstream LLM. We can represent the output of the convolution downsampling module as CNN(x) = fCNN(AE(x);θCNN), where θCNN represents learnable parameters of the 1-D convolution blocks.

Large Language Model: An LLM typically takes in a sequence of input text tokens z and models the probability of observing an output text sequence y as the possible next tokens for the input text. The text tokens are converted to vectorized embeddings EMB(z) using a lookup matrix that is learned during training. The output of the LLM can be represented as LLM(z) = fLLM(EMB(z);θLLM), where θLLM are the weights of the LLM. In this work, we leverage a pre-trained LLM for multimodal tasks. For formulating the multimodal input with audio x and text sequence z, we simply concatenate the downsampled audio features (CNN(x)) with the token embeddings ((EMB(z)) in the sequence dimension as shown in Figure 2. Hence, we can represent the probability distribution over the output from our multi-modal Spoken Language Model (SLM) as:

SLM(x,z) = fLLM CNN(x),EMB(z) ;θLLM (2)

#### 2.2 Multimodal Instruction Finetuning

τ

Let Dτ = {xτ,yτ}n

1 represent a labelled dataset for a task τ with nτ samples where each sample consists of audio sequence xτ and a corresponding text label sequence yτ. Let Pτ = {pτ}m

τ

1 be a set of mτ textual prompts/instructions sequences pτ describing the task. In our experiments, we do a weighted combination of datasets from each of the M training tasks {wτ,Dτ}M1 where wτ is the weight given to each sample of task τ. This is done to ensure balance between tasks with different complexities and training data sizes. Thus, each sample can be represented as a tuple (xτ,pτ,yτ) where xτ is the audio sample, pτ is a prompt/instruction sampled uniformly from Pτ and yτ is the label. Then the probability for predicting the labels yτ can be defined as:

p(yτ|xτ,pτ;Θ) = SLM(xτ,pτ) (3)

where Θ = {θAE,θCNN,θLLM} are all the parameters of our audio language model. The selfattention layers in the LLM attend to both the audio and the textual instruction to generate the required output on any audio-related task. We use the standard gradient descent method to maximise the likelihood of producing the target label yτ for each sample in the training dataset as defined below:

(yτ|xτ,pτ;Θ) (4)

L(Θ) = −log p(xτ,pτ,yτ)∼{Dτ}M1

#### 2.3 Curriculum Learning with Parameter Efficient Finetuning

To ensure faster convergence and avoid catastrophic forgetting and overfitting by the pre-trained LLM, we adopt a parameter efficient approach based on low-rank adaptation [21], or LoRA, for training our multimodal models. In this work, we freeze the pre-trained audio encoder and LLM, and only train the convolution downsampling module and the LoRA adapter. Since the bulk of the parameters (θAE and θLLM) are never updated throughout the training process, it makes our framework compute efficient and allows us to scale to a large number of diverse datasets and tasks with limited compute resources. Moreover, it enables leveraging the existing capabilities of both pre-trained audio and language models without catastrophic forgetting. However, when training both the downsampling module and the LoRA adapter from scratch on a diverse set of speech tasks, we observe frequent gradient explosion, leading to suboptimal convergence. Hence, we carefully design a curriculum of two stages for training.

In the first stage, we only train the convolution downsampling module and the intermediate layer weights without introducing the LoRA adapters. Further, only the samples from the automatic speech recognition (ASR) task are used in this stage. Since the encoded speech feature vectors can be very different from the token embeddings of the text input, this stage can help align them more easily in a common embedding space by only learning the parameters for the convolution downsampling module in the confined task space of ASR. This enables a pre-trained text-based LLM to attend to the content of the audio sequence and generate the speech transcription.

In the second stage, we now introduce the LoRA adapters for training the model. In this stage, the intermediate layer weights, the downsampling module as well as the LoRA adapters are unfrozen. Since the LoRA adapters are training from scratch, we first allow the adapter weights to warmup by training only on the ASR task, so as to get aligned to the common embedding space learned by the convolution downsampling module in the first stage. Finally, we introduce additional tasks on top of the ASR task and continue training while keeping the pre-trained audio encoder and LLM weights frozen. Since the warmup using only the ASR task enables the model to understand the contents of

- Table 1: Details of our tasks, training datasets and evaluation metrics. ST, IC, SF, KWE and KWS tasks are referred as spoken language understanding (SLU) tasks, while ER, ASC, SC, AC and SNS represents paralinguistic speech processing (PSP) tasks

Training #Hours #Samples

Task Description Metric Dataset

Librispeech [22] 960 281K Mozilla Common Voice 5.1 [23] 1.4K 1.1M

Automatic Speech Recognition

Word Error Rate (WER)↓

ASR

VoxPopuli [24] 385 160K

SLURP [25] 84 119K EuroParl [26] 92 39K

MSP-Podcast 1.11 [27] 135 83K ST Speech Translation BLEU ↑

CoVost2 [28] 426 576K EuroParl [26] 73 90K

IC Intent Classification Accuracy (ACC)↑ SLURP [25] 35 47K SF Slot Filling SLU-F1 ↑ SLURP [25] 25 30K

Librispeech [22] 960 281K Mozilla Common Voice 5.1 [23] 426 288K

KWE Keyword Extraction Macro F1↑

Librispeech [22] 960 281K Mozilla Common Voice 5.1 [23] 426 288K

KWS Keyword Search Accuracy (ACC) ↑

ER Emotion Recognition

MSP-Podcast 1.11 [27] 91 57K

ASC Audio Sentiment Classification MSP-Podcast 1.11 [27] 135 84K SC Speaker Counting Fisher [29, 30] 690 755K AC Accent Classification Mozilla Common Voice 5.1 [23] 190 123K

Unweighted Average Recall (UAR) ↑

SNS Speech/Non-Speech Detection In-house VAD 269 149K

the audio, our curriculum learning approach leads to faster convergence on a variety of speech-based tasks that rely on the spoken contents of the audio.

### 3 Experiments

#### 3.1 Tasks

In this work, we use a large collection of publicly available speech datasets from a diverse set of tasks. A summary of the datasets and evaluation metrics for these tasks is provided in the Table 1, while examples and prompts are covered in the Table 10. Our training tasks include automatic speech recognition (ASR), five spoken language understanding (SLU) tasks, and five paralinguistic speech processing (PSP) tasks. The SLU tasks include those tasks which can be solved by a cascaded system of a ASR model and an LLM, while PSP tasks are classification tasks based on the audio, typically used in audio analytics. For the IC/SL tasks, we split the SLURP dataset into seen and unseen intent/slot label classes and study them separately to understand the generalization capabilities of the model. The KWE task is about finding important keywords from the audio, while in the KWS task, we learn to classify whether a particular keyword was present in the audio or not. The target labels were synthetically created for both these tasks using an LLM. All other tasks are standard and an interested reader can refer to the Appendix A.3 for more details. We create a list of at least 15 prompts per task describing the goal of the task. To further add diversity to the set of tasks, we use a text-to-speech (TTS) version of the Alpaca dataset [31]. This dataset contains a diverse collection of prompt, input, output tuples, where the prompt describes the task, input is the input for the task, and the output contains the target labels. However, there are no corresponding audios associated with the dataset. As in the existing work [17], we use a TTS system (AWS Polly in our case) to generate synthetic audios for the input text using a pool of 10 different speakers.

#### 3.2 Models

We train three different variants of multimodal models using our SpeechVerse framework, namely, (1) Task-FT: These represent a set of models where each model is trained individually for a particular task τ. While most tasks have separate model, datasets from certain related tasks like IC/SF, KWE/KWS, and ER/ASC are trained together. (2) Multitask-WLM: This represents a single multi-task model trained by pooling datasets for all the tasks together. Both (1) and (2) uses a pretrained WavLM Large

- [32] as the backbone audio encoder and only uses the last encoder layer output AE(x) = h(L) as the

- Table 2: Results of ASR and spoken language understanding (SLU) tasks. Datasets are defined as: LTC: Librispeech test-clean; LTO: Librispeech test-other; Vox: Voxpopuli; MCV: Mozilla Common Voice; EN: English; DE: German; FR: French;

Model Training

ASR (WER↓)

IC (ACC↑)

SF (SLU-F1↑)

ST (BLEU↑)

KWE (F1↑)

KWS (ACC↑) LTC LTO Vox MCV SLURP SLURP EN→DE EN→FR MCV MCV

Whisper ASR → LLM

LLM-FT

2.5 4.9 7.0 8.2 86.6 72.3 23.5 30.6 45.1 87.7 SpeechVerse ASR → LLM 2.1 4.4 6.5 10.5 88.8 80.5 24.4 30.9 54.3 87.3

SpeechVerse (E2E–SLM)

Task-FT 2.1 4.4 6.5 10.5 90.4 82.4 27.8 35.7 46.9 98.7 Multitask-WLM 2.5 4.7 6.8 12.0 90.3 82.2 25.9 33.7 46.6 98.5 Multitask-BRQ 3.0 6.7 7.1 11.8 90.0 83.4 25.2 32.5 44.2 98.4

GT → LLM LLM-FT - - - - 94.2 84.3 27.2 34.8 70.3 100

- Table 3: Results of paralinguistic speech processing (PSP) tasks. All reported numbers are the value of the UAR metric.

Model Training

ER ASC SC AC SNS

MSP MSP Fisher MCV VAD WavLM Task-FT 60.3 57.5 98.5 57.9 96.3

SpeechVerse (E2E–SLM)

Task-FT 64.6 61.3 99.6 60.1 96.9 Multitask-WLM 62.0 60.1 98.9 60.0 97.6 Multitask-BRQ 65.1 64.1 99.3 60.4 97.5

representation of the audio. (3) Multitask-BRQ: This model is similar to (2), but it uses the Best-RQ

- [33] architecture for the audio encoder. Because the Best-RQ encoder is trained using a random projection quantizer, the middle layer weights are more suited for downstream tasks if the encoder is frozen during fine-tuning. We therefore used a unified representation that combines representations from all layers via learnable weights for the multitask model trained with the BEST-RQ encoder. The details about the pretraining of this audio encoder are described in the Appendix A.1. All the three variants of our models are trained using curriculum learning as highlighted in the Section 2.3. All the models use Flan-T5-XL [5] as the backbone LLM. The LoRA adapters introduced in Task-FT models have rank (r) = 8, while multitask models use r = 16, allowing more learnable parameters for solving a diverse collection of tasks. The complete list of hyper-parameters and training setting is provided in the Appendix A.2.

Baselines: For the SLU tasks, we compare our models with a cascaded baseline that uses an LLM on ASR hypotheses (ASR → LLM). For a fair comparison, we use a parameter-efficient fine-tuned version of Flan-T5-XL as the LLM for the baseline. The multi-task fine-tuning data is exactly the same between our models and the baseline except that the latter uses ground truth text in place of the audios. We benchmark the cascaded approach with ASR hypotheses from (1) a strong publicly available Whisper-large-v2 [20] ASR model and (2) our ASR Task-FT SpeechVerse model, enabling a true comparison between a multimodal model vs cascaded approach. Finally, we also benchmark the performance of the oracle ASR system by passing the ground truth transcripts to the baseline LLM (GT → LLM). For the KWS task, we use substring search of the keyword in ASR hypotheses as the baseline. For PSP tasks, we train task-specific classifiers that use the last layer representations from WavLM Large. The classifier contains a feed-forward layer, followed by a 2-layer Gated Recurrent Unit (GRU) with mean pooling over frames, followed by another 2 layers of feed-forward network and finally a softmax operator. These models are trained on the same task-specific data thereby allowing for direct comparison with our WavLM-based multimodal models.

- 4 Results

#### 4.1 Evaluation of SpeechVerse models

We evaluate end-to-end trained joint speech and language models (E2E-SLM) leveraging the SpeechVerse framework on 11 unique tasks across multiple domains and datasets. We first evaluate SpeechVerse’s core speech understanding capability through ASR benchmarks. We then evaluate more complex SLU tasks and paralinguistic speech tasks in Tables 2 and 3 respectively.

#### 4.1.1 Performance on ASR and SLU tasks

First, we evaluate the performance of SpeechVerse models on four public ASR benchmark datasets namely libri-test-clean, libri-test-other, Voxpopuli and CommonVoice. The WER numbers for each of these datasets are reported in Table 2. SpeechVerse ASR in row 2 uses same model as task-specific pretrained ASR model (Task-FT) in row 3. When comparing our task-specific pretrained ASR model, which also serves as the initialization for multi-task finetuning, to Whisper ASR, our model achieves slightly better performance on average. However, the WER increases in both multitask models with Multitask-WLM performing similarly to Whisper across three out of the four test sets. The lower performance of the multi-task SpeechVerse model compared to the task-specialized model is likely due to giving lower weight to ASR datasets when constructing batches during multi-task training. This was done to balance the performance across all tasks, since the data distribution is imbalanced between the different tasks.

When it comes to SLU tasks, a frequent question posed is if an end-to-end model can outperform a cascaded pipeline that transcribes speech via ASR and then feeds it to a language model. To investigate this, we conducted experiments on five semantic understanding tasks using the same foundation models as SpeechVerse. The text foundation model was further fine-tuned on data from the five SLU tasks separately, as we found the zero-shot performance of Flan-T5 on these benchmark test sets to be quite poor. We also report performance when feeding ground-truth transcripts into the fine-tuned LLM, to provide upper bound results. On 4 of the 5 tasks, excluding Keyword Extraction, the end-to-end trained models outperform the cascaded pipeline. In particular, the more commonly used tasks like intent classification, slot labeling, and speech translation are performing better than the cascaded system, demonstrating the efficacy of our models trained using SpeechVerse. We also observed that SpeechVerse models on KWS task are outperforming cascaded pipeline by an absolute 10% in accuracy, while performing significantly behind on KWE task. Since the keyword search task requires an attention span focused on a specific word of interest, joint modeling helps improve accuracy by overcoming error propagation present in a cascaded pipeline. We also conducted an ablation study to determine if the KWE task benefits further from joint decoding of ASR transcription and keywords. We noticed an improvement in performance, closing the gap to the cascaded pipeline. The results from this study are detailed further in the sub-section 4.3.2. When comparing the multitask models to the task-specific SpeechVerse models, there is a minor degradation in performance, but the difference is not substantial. Overall, the multitask model trained with either WavLM encoder or Best-RQ encoder outperformed cascaded systems in majority tasks.

#### 4.1.2 Performance on paralinguistic tasks

The results in Table 3 demonstrate clear improvements in performance on various paralinguistic speech processing tasks when using multi-task learning compared to fine-tuning the WavLM model independently for each task. Specifically, the SpeechVerse model trained with multitask learning using Best-RQ audio encoder (Multitask-BRQ) achieves gains over the baseline WavLM model of 4.8% absolute on emotion recognition, 6.6% on audio sentiment classification, and 2.5% on accent classification. More modest gains are seen with the SpeechVerse model trained using multitask learning with WavLM encoder (Multitask-WLM). The unified representation’s adaptive combination of all encoder layers helps multitask BEST-RQ model improve diverse paralinguistic task performance. Overall, multi-task learning provides noticeable improvements in model generalization and effectiveness across a diverse set of speech tasks compared to task-specific fine-tuning of the baseline WavLM model. The results highlight the advantages of learning shared representations across related tasks using multi-task learning techniques.

#### 4.1.3 Comparison against SOTA models

Table 4 benchmarks SpeechVerse models against state-of-the-art (SOTA) models on five diverse tasks: automatic speech recognition (ASR), speech translation (ST), intent classification (IC), slot filling (SF), and emotion recognition (ER). Across these tasks, SpeechVerse demonstrates competitive or superior performance compared to prior specialized models. When comparing our task-specific pretrained ASR model, which also serves as the initialization for multi-task finetuning, to Whisper ASR, our model achieves slightly better performance on average. However, the multitask model (Multitask-WLM) performed similarly to Whisper across three out of the four test sets. When evaluating on speech translation across three language pairs, the task-specialized SpeechVerse model

- Table 4: Comparison of SpeechVerse models to prior specialized SOTA models on five diverse tasks: automatic speech recognition (ASR), speech translation (ST), intent classification (IC), slot filling (SF), and emotion recognition (ER).

Performance Metrics Results

Task Dataset Model

2.5 | 4.9 SLM-FT [17] 2.6 | 5.0

Whisper large-v2 [20]

Librispeech test-clean | test-other

WER ↓

SpeechVerse Task-FT 2.1 | 4.4 SpeechVerse Multitask-WLM 2.5 | 4.7

Whisper large-v2 [20]

7.0 mSLAM-CTC [34] 7.0

ASR

VoxPopuli

WER ↓

SpeechVerse Task-FT 6.5 SpeechVerse Multitask-WLM 6.8

Whisper large-v2 [20]

8.2 SLM-FT [17] 7.5

CommonVoice 5.1

WER ↓

SpeechVerse Task-FT 10.5 SpeechVerse Multitask-WLM 12.0

27.8 | 30.3 | 38.7 XMEF [36] 22.5 | 30.0 | 32.3

SeamlessM4T 1.2B [35]

EuroParl EN→DE | EN→FR | EN→RO

BLEU ↑

ST

SpeechVerse Task-FT 27.8 | 35.7 | 32.2 SpeechVerse Multitask-WLM 25.9 | 33.7 | 30.1

86.9 Frozen-hbt-large [38] 74.4

E2E-SLU CTI [37]

IC SLURP

ACC ↑

PF-hbt-large [38] 89.2 SpeechVerse Task-FT∗ 84.6

74.7 Frozen-hbt-large [38] 60.1

E2E-SLU CTI [37]

SF SLURP

SLU-F1 ↑

PF-hbt-large [38] 78.9 SpeechVerse Task-FT∗ 76.7

w2v2-L-robust[39]

58

SpeechVerse Task-FT 66.7 SpeechVerse Multitask-WLM 61.2

ER MSP-Podcast 1.7

UAR ↑

∗SpeechVerse Task-FT model was re-trained for SLURP by including all intents and slots for comparison with other models.

surpassed SeamlessM4T on two pairs, while the multi-task SpeechVerse model achieved competitive performance compared to prior work on average. Both models did not perform well on English to Romanian pair. The overall performance of the SpeechVerse models on speech translation is heavily limited by the capabilities of the underlying language model FlanT5. The speech translation capabilities cannot exceed the translation quality provided by FlanT5 as the base language model. To evaluate SpeechVerse on spoken language understanding tasks like intent classification (IC) and slot filling (SF), we retrained the task-specialized SpeechVerse model by incorporating all 69 intents (both seen and unseen) as well as all slots. This allowed us to compare SpeechVerse to prior work on the complete intent and slot sets. Our SpeechVerse model achieved competitive performance to the previous SOTA (PF-hbt-large) on slot filling, but was significantly behind on intent classification with 5% lower absolute accuracy. However, SpeechVerse outperformed the same SOTA model (Frozen-hbt-large) by 10% when the encoder weights were frozen during fine-tuning. To further analyze the gap to prior state-of-the-art, we conducted an experiment allowing the audio encoder weights to be tunable during fine-tuning. This achieved 89.5% accuracy, matching the prior SOTA. This suggests the intent classification performance can overfit to the specific acoustic conditions of the SLURP dataset when full fine-tuning is performed. The SpeechVerse model that was trained end-to-end specifically for the task of emotion recognition achieved an 8% absolute improvement in unweighted average recall over the previous state-of-the-art model (w2v2-L-robust). In contrast, the multitask SpeechVerse model performed 3% better than the prior state-of-the-art. However, one key difference is that the previous SOTA work trained on the MSP-Podcast 1.7 dataset, while we used version 1.11 for training. The test set version remained the same between the two approaches. Overall, the SpeechVerse model demonstrated competitive performance compared to prior specialized models in some cases when evaluated across the various tasks.

- Table 5: Generalization to unseen prompts: The performance of each task is assessed on three different prompts, out of which, two are unseen during training.

Prompt (Unseen)

Performance Metric Results

Task Dataset

- P1 (✗) WER↓

6.8

- P2 (✓) 7.1
- P3 (✓) 6.8

ASR Voxpopuli

- P1 (✗) BLEU↑

33.7

- P2 (✓) 33.8
- P3 (✓) 33.5

EuroParl EN→FR

ST

- P1 (✗) UAR↑

60.0

- P2 (✓) 60.7
- P3 (✓) 60.6

AC MCV

#### 4.2 Generalization Across Instructions

We comprehensively study our Multitask-WLM model’s ability to generalize to diverse forms of unseen instructions. As a first, we try to accomplish seen tasks with differently worded instructions than those used for training. We create novel prompts for some of the training tasks and evaluate the robustness of the model to variations in the prompt. Next, we demonstrate the model’s potential to leverage the robust language understanding of the underlying LLM to generalize to completely new tasks that the model has not seen at all during multimodal finetuning.

#### 4.2.1 Measuring robustness to prompt variations

To evaluate the effect of different prompts on the training task performance, we tested our MultiTaskWLM model with additional prompts for 3 different tasks: ASR, ST and AC. We test with three prompts per task, where one is directly taken from the set of prompts used during training, while two additional novel prompts are created by using a different wording and context. As showcased in the Table 5, the model showecased similar performance across prompts for each task. For the ASR task, we only see a minor variation of 0.3 in WER between seen and unseen prompts. Similarly for the ST and the AC task, the metric values deviates very little with the change in prompt. These small differences indicate that our model has generalized its core capabilities well and is not strongly dependent on the specific wording or context of the prompts. While prompt engineering remains important for optimal performance, these results suggest that the model has learned robust taskspecific skills that transfer across the choice of prompts, at least for the tasks seen during training. For the tasks beyond those seen during training, we study the prompt design separately in a later subsection 4.3.1.

#### 4.2.2 Open-ended Instruction Following

To study the model’s ability to understand open-ended text-based as well as speech-based instructions, we prompted the model with several unrestricted creative requests that were not a part of our training curriculum. We enumerate several such examples in the Table 6. In many of these examples, the model is required to exhibit profound comprehension of both the spoken and written directives to successfully execute the task. For example, in the Creative QA task, the model has to understand the spoken request as well as the instruction prompt in order to generate a related response. In the Summarization task, the model has to correctly surmise the spoken content to generate a summary. In the Contextual Biasing task, we observe that the model is even able to correct its own output when provided with hints. The robust responses of the multi-task model with such a distributional shift in the input from the training data demonstrates the adaptability of its core instruction following skills. Rather than overfitting to the training domain, the multi-task learning approach enables the model to learn more versatile capabilities in instruction comprehension and execution that better transfer to new contexts. We provide some quantitative results on unseen tasks and labels in the next section.

- Table 6: Qualitative examples on open-ended tasks showcasing instruction-following ability of the model. DC: Domain Classification; QA: Question Answering; ST: Speech Translation

Task Instruction Prediction

[Figure 3]

[Transcription: are you able to update the personal details on my account]

are you able to update the personal details on my account

Transcribe the preceding audio

Post Processing

are you able to update the personal details on my account?

Transcribe the preceding audio with punctuations

Sure, I can update the personal details on your account. Can you please tell me which account you are referring to and the specific details you would like me to update?

Response Generation

Generate a response to the user query in the preceding audio

ST

Translate the preceding audio from english into spanish

Puedes actualizar los datos personales de mi cuenta?

unseen language

[Figure 4]

[Transcription: i need synthroid supply that will last me a month]

What is being said in the preceding audio? i need cynthroid supply that will last me a month What is being said in the preceding audio? Hints: thyroxine, synthroid, thyroid, hormones

i need synthroid supply that will last me a month

Contextual Biasing

You are an AI assistant for a pharmaceutical company that specializes in thyroid treatment. Based on this context, what is being said in the preceding audio?

i need synthroid supply that will last me a month

Classify the domain of the user audio input into finance, healthcare, travel, internet or food

DC

healthcare

What is the name of the medicine in the preceding audio?

QA

Synthroid

[Figure 5]

A 30-second audio generated using a text-to-speech system with contents of a news article∗

Special counsel jack smith urges us supreme court to let former president donald trump’s election interference case proceed to trial without further delay.

Generate an abstractive summary for the content of the preceding audio in a single line

Summarization

Create a very crisp newspaper headline for the preceding audio

Former President Donald Trump Faces Election Interference Charges in Court

[Figure 6]

[Transcription: Can you explain the process of composting?]

The process of composting is a gradual decomposition of organic matter into soil, with the goal being to decompose the organic matter into nutrients and water.

Factual QA

Generate a short factual answer to the question in the preceding audio

[Figure 7]

[Transcription: Compose a descriptive paragraph about a childhood memory.]

A child’s imagination is a powerful tool for creating memorable images, and one of my favorite memories is the creation of a dollhouse complete with all the pieces.

Creative QA

Generate a short creative answer to the question in the preceding audio

∗https://apnews.com/article/trump-special-counsel-election-interference-january-6-c2dcc83e56a541804d4785f6bb6cd45c

- Table 7: Results on three unseen tasks with and without constrained decoding (CD). We compare the settings wherein the instruction prompt only contained the class labels v/s when descriptions were provided for each class label in the prompt.

IC SL DC

Class Label Description

CD

SLURP∗ SNIPS SLURP∗ Internal (ACC) (ACC) (SD-F1) (SLU-F1) (ACC) No

No 51.7 54.9 73.12 46.45 56.9 Yes 68.8 68.9 73.38 46.49 59.0

No 47.7 61.0 69.79 47.34 44.0 Yes 70.2 75.9 77.47 48.45 62.0

Yes

∗This is a subset of SLURP labels not seen during training

#### 4.3 Strategies for Improving Performance

We further evaluate strategies to improve the multi-task model’s performance specially for unseen tasks and class labels. First, we leverage contrained decoding [40] for tasks that have a pre-defined set of finite outcomes. Next, we also study joint decoding of the output of the task with the ASR hypotheses of the audio for certain complex spoken language understanding tasks.

#### 4.3.1 Constrained Decoding

The work in [40] introduced a model-agnostic technique to enforce domain-specific knowledge and constraints during text generation. Building on this prior approach, we have explored applying decoding constraints to the SpeechVerse model to improve generalization to unseen speech classification tasks. Rather than allowing the model to generate freely in response to a prompt, the decoding is restricted to output from a predefined vocabulary of class names. For example, in an intent classification task, the model would be constrained to only generate intent labels such as “play_radio”, “datetime_query” or “cooking_recipe”. By limiting the output space, the model is more likely to produce the desired class label rather than unrelated text.

We meticulously benchmark the model’s performance on close-ended tasks, such as a diverse set of classification tasks, that have a pre-defined set of finite class labels. To understand the influence of the instruction prompts, we divide this study into two parts: (1) where we only provide the class labels in the prompt, and (2) where we provide an accompanying description of each class label in the prompt. We ensure that none of these class labels were seen during training, and hence these are all novel tasks for the model. Further, we evaluate the efficacy of employing constrained decoding in each of these two parts, as the class labels are known to us beforehand. Note here that the SL task can be considered a harder task as the model has to correctly classify a slot label as well as identify the corresponding slot value from the speech. Hence, we report both, the SLU-F1 metric as well as the SD-F1 (Slot label Detection) metric for SL. The results of this study are presented in Table 7.

We observe that including descriptions in the prompt has inconsistent results, which can be attributed to the quality and subjectivity of the descriptions provided in the prompt, especially as these descriptions were not seen during training. However, we see that constrained decoding improves upon the results in all cases, and most significant gains are observed only when descriptions are provided with constrained decoding. This indicates that providing descriptions indeed steers the model towards better comprehension of the task semantics, but only constrained decoding is able to objectively prune the noise introduced by any prompt bias. This phenomena is further revealed in the SL task, where the SLU-F1 has a lower absolute value compared to SD-F1, as the SLU-F1 metric incorporates both slot label and slot value, whereas constrained decoding can only be applied to the slot label (hence the higher SD-F1). Similarly, for a completely unseen task of Domain Classification (DC), where the goal is to classify the content of the audio into five domains like healthcare, technology etc, we observe a strong performance of 62% accuracy with constrained decoding.

#### 4.3.2 Joint Decoding

Certain SLU tasks require the model to understand the semantics of the audio or perform a operation on the content of the audio. For example, KWE task is about extracting important keywords from

the ASR hypothesis of the audio. Since this is a multi-step reasoning process for the model, we take inspiration from the existing work [41] on Chain-of-Thought (CoT) prompting. We train our model to first decode the ASR hypothesis of the audio, followed by the output of the task. The prompts used for the joint elicitation of ASR hypothesis and the task output are described in the Table 8. For a representative set of SLU tasks including IC, KWE and ER, we re-train the Task-FT models by adding a small portion of such multi-step examples along with single-task examples. We compare the results with and without joint decoding with ASR hypothesis in the Table 9.

Table 8: Example prompt for elicitation of compound goals for KWE and ER task

Instruction Prediction

Task

Perform the following audio-based tasks in the order as described.

=== Task: ASR === Perform speech recognition using the preceding audio.

ASR: paris is the capital of france | KWE: paris, france |

=== Task: KWE === Identify significant keywords in the provided audio. Make sure to format the output as "ASR: ... | KWE: ... |"

KWE

Perform the following audio-based tasks in the order as described.

=== Task: ASR === What is being said in the audio?

ASR: can you shut up for a while | Emotion: angry |

ER

=== Task: Emotion === Classify the tone of the speaker as happy, sad, angry or neutral Make sure to format the output as "ASR: ... | Emotion: ... |"

Table 9: Results on compound goal experiments when decoding performed with (w/) and without (w/o) ASR.

Joint Decoding w/o w/

Task Dataset Metric

IC SLURP ACC↑ 90.4 92.2 KWE

MCV

46.9 49.5 Vox 53.1 56.7

F1↑

ER MSP UAR↑ 64.5 64.7

The results in the table showcase that augmenting the training data with compound goals helps to improve the performance for all three tasks. The improved performance can be attributed to the possibility of self-attention on the already decoded ASR hypothesis in the decoder of our multimodal model. Further, such multi-step training examples brings out the true multimodal capabilities of the model to successfully complete the combination of tasks. Further, such a paradigm can help save crucial inference latency by using a single call of the large multi-modal model for obtaining both the transcript and the task output. A more detailed analysis on understanding the benefits of joint decoding will be conducted in future work.

### 5 Related Work

Multi-task learning. Prior studies have shown that a single deep learning model is capable of jointly learning multiple large-scale tasks across different domains [42]. The key idea in multi-task learning is to leverage shared representations across related tasks to improve overall generalization and efficiency. Following this approach, the T5 model [43] frames all text tasks as text-in to text-out, using a unified text-based framework that facilitates shared representations across textual tasks. Similarly, SpeechNet [44] and SpeechT5 [19] leverage a shared encoder-decoder framework to jointly model speech and text modalities spanning 5 to 6 tasks like TTS, ASR, and Voice Conversion (VC). VIOLA [15], a single auto-regressive Transformer decoder-only network, unifies various cross-modal speech and text tasks as a conditional codec language model via multi-task learning. Whisper [20] also employs large-scale multi-task learning, training on related speech tasks including language identification, speech recognition, and translation. In this work, SpeechVerse utilizes multi-task

Table 10: Qualitative examples of output from all the training tasks on a audio file with transcription "turn on the radio now"

Task Instruction Prediction

Transcribe the preceding audio turn on the radio now What is being said in the preceding audio? turn on the radio now

ASR

Translate the english audio to german schalten sie jetzt den radio auf Generate french translation for the english audio mettez le radio à l’onder maintenant

ST

The previous audio needs to be mapped to exactly one of the following intents:

- - datetime_query: user asks about date, time, schedules, etc;
- - qa_definition: user wants to ask for general fact-based questions to receive definitions, explanations, descriptions, etc;
- - calendar_remove: user wants to remove calendar events by voice;

...

- - play_radio: user wants to request playback of radio, stations, etc;

IC

play_radio

Identify any of the following slots present in the speech:

- - date: any reference to a date, month or day of the week;
- - place_name: any reference of a place e.g., berlin, new york etc.;
- - person: any reference of a person;
- - time: any reference to time e.g., three am, two thirty etc.;

SF

time = now;

KWE Identify important keywords in the preceding recording radio KWS Is the word ’radio’ used in the audio? yes ER Detect the primary emotion conveyed in this audio by the speaker’s

neutral

tone - is it happiness, sadness, anger, or neutral?

ASC Classify the sentiment of the speaker into one of the following positivity, negativity, or neutrality

neutral

AC Analyze the speaker voice and figure out their accent from one of American, Indian, Australian, Canadian or British.

American SC How many distinct speakers are there in this audio clip. one SNS Is their speech in the preceding audio or not? speech

training to transfer knowledge between several related tasks while using natural language instructions to perform each task. Unlike prior work that generated text, speech, or both, our method focuses solely on producing textual output, while taking in audio and text instructions.

Multimodal Large Language Models. Prior work on multimodal LLMs has focused primarily on tasks involving images, such as image generation, visual question answering, and image captioning [4, 9, 45, 46]. Multimodal models incorporating modalities like audio and speech have received relatively less attention compared to vision-and-language models [47–49]. However, there has been growing interest in augmenting large language models with audio data, leading to several proposed approaches [8, 14, 17, 18, 50–52]. SpeechGPT [18] proposed a multimodal LLM combining discrete units of HuBERT with an LLM to solve few understanding tasks like ASR, Spoken QA as well as generation tasks like TTS. [17] introduces the novel capability of zero-shot instruction-following for more diverse tasks such as dialog generation, speech continuation and Question Answering. Most recently, [16] proposed Qwen-Audio, a large-scale audio-language model trained using a multi-task learning approach to handle a diverse range of tasks across various audio types including human speech, natural sounds, music, and songs. Qwen-Audio employs a single audio encoder to process various types of audio whose initialization is based on the Whisper-large-v2 model [20] and performs full finetuning. In contrast, our work utilizes two frozen pretrained models, one each for speech encoder and text decoder to retain their intrinsic strengths. Also, we utilize 30+ instructions for each task during training for improved generalization whereas [17] uses a single fixed instruction. Additionally, SpeechVerse incorporates multi-task learning and instruction finetuning in a single training stage.

### 6 Conclusion

In this work, we propose SpeechVerse, a multimodal framework that enables LLMs to follow natural language instructions for performing diverse speech processing tasks. Through supervised instruction finetuning and combining representations from frozen pre-trained speech and text foundation models, SpeechVerse achieves strong zero-shot generalization on unseen tasks. Extensive benchmarking against conventional baselines show SpeechVerse’s superiority on 9 out of 11 tasks, demonstrating its formidable instruction following capability. Crucially, SpeechVerse maintains robust performance on out-of-domain datasets, unseen prompts, and even unseen tasks. This highlights the efficacy of our proposed training methodology in imbuing the model with a generalizable skill for mapping text-based instructions to speech processing outputs. Moving forward, we aim to expand SpeechVerse’s capabilities to follow even more complex instructions and generalize to new domains. By separating task specification from model design, SpeechVerse represents a versatile framework that can dynamically adapt to new tasks through natural language without retraining.

### Limitations

While this work demonstrated strong instruction following capabilities for the multitask SpeechVerse model across a variety of tasks, some limitations remain. The study relied on a single underlying LLM architecture (FlanT5) rather than exploring more recent models tailored for instruction following. Additionally, there is a trade-off between generalized capabilities on unseen tasks versus specialized performance on original training tasks that poses challenges for a single multitask model. While the model showed promise in handling diverse unseen tasks, its limitations were not fully characterized across the wide scope of possible instructions and the performance on these unseen tasks is not quantitatively measured.

### Ethics Statement

All speech datasets we use have anonymous speakers. We do not have any access to nor try to create any PII (Personal Identifiable Information) of speakers, and our model neither identifies speakers nor uses speaker embeddings. Most of the work used public open-source datasets for both training and testing. The in-house datasets used for pre-training Best-RQ encoder and SNS task are collected via third-party speech data vendors. No additional data collections made concerning to the work carried in this paper.

### References

- [1] T. Brown et al., “Language models are few-shot learners,” Advances in neural information processing systems, vol. 33, pp. 1877–1901, 2020.
- [2] A. Chowdhery et al., “Palm: Scaling language modeling with pathways,” Journal of Machine Learning Research, vol. 24, no. 240, pp. 1–113, 2023.
- [3] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, et al., “Improving language understanding by generative pre-training,” 2018.
- [4] J. Achiam et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.
- [5] H. W. Chung et al., “Scaling instruction-finetuned language models,” arXiv preprint arXiv:2210.11416, 2022.
- [6] L. Ouyang et al., “Training language models to follow instructions with human feedback,” Advances in Neural Information Processing Systems, vol. 35, pp. 27730–27744, 2022.
- [7] H. Touvron et al., “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.
- [8] R. Huang et al., “AudioGPT: Understanding and generating speech, music, sound, and talking head,” arXiv preprint arXiv:2304.12995, 2023.
- [9] T. Gemini et al., “Gemini: A family of highly capable multimodal models,” arXiv preprint arXiv:2312.11805, 2023.
- [10] T. Guo et al., “Large language model based multi-agents: A survey of progress and challenges,” arXiv preprint arXiv:2402.01680, 2024.
- [11] W. R. Huang et al., “Multilingual and fully non-autoregressive asr with large language model fusion: A comprehensive study,” arXiv preprint arXiv:2401.12789, 2024.
- [12] Y. Li, Y. Wu, J. Li, and S. Liu, “Prompting large language models for zero-shot domain adaptation in speech recognition,” in Proc. Automatic Speech Recognition and Understanding Workshop (ASRU), IEEE, 2023, pp. 1–8.
- [13] R. Ma, M. Qian, P. Manakul, M. Gales, and K. Knill, “Can generative large language models perform asr error correction?” arXiv preprint arXiv:2307.04172, 2023.
- [14] P. K. Rubenstein et al., “Audiopalm: A large language model that can speak and listen,” arXiv preprint arXiv:2306.12925, 2023.
- [15] T. Wang et al., “VioLA: Unified Codec Language Models for Speech Recognition, Synthesis, and Translation,” arXiv preprint arXiv:2305.16107, 2023.
- [16] Y. Chu et al., “Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models,” arXiv preprint arXiv:2311.07919, 2023.
- [17] M. Wang et al., “SLM: Bridge the thin gap between speech and text foundation models,” in Proc. Automatic Speech Recognition and Understanding Workshop (ASRU), IEEE, 2023, pp. 1–8.
- [18] D. Zhang et al., “Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities,” arXiv preprint arXiv:2305.11000, 2023.
- [19] J. Ao et al., “SpeechT5: Unified-modal encoder-decoder pre-training for spoken language processing,” arXiv preprint arXiv:2110.07205, 2021.
- [20] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. ICML, 2023, pp. 28492–28518.
- [21] E. J. Hu et al., Lora: Low-rank adaptation of large language models, 2021. arXiv: 2106.09685 [cs.CL].
- [22] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: An asr corpus based on public domain audio books,” in 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), IEEE, 2015, pp. 5206–5210.
- [23] R. Ardila et al., “Common voice: A massively-multilingual speech corpus,” arXiv preprint arXiv:1912.06670, 2019.
- [24] O. Dekel and O. Shamir, “Vox populi: Collecting high-quality labels from a crowd.,” in COLT, 2009.
- [25] E. Bastianelli, A. Vanzo, P. Swietojanski, and V. Rieser, “Slurp: A spoken language understanding resource package,” arXiv preprint arXiv:2011.13205, 2020.
- [26] P. Koehn, “Europarl: A parallel corpus for statistical machine translation,” in Proceedings of machine translation summit x: papers, 2005, pp. 79–86.

- [27] R. Lotfian and C. Busso, “Building naturalistic emotionally balanced speech corpus by retrieving emotional speech from existing podcast recordings,” IEEE Transactions on Affective Computing, vol. 10, no. 4, pp. 471–483, 2017.
- [28] C. Wang, A. Wu, and J. Pino, “Covost 2 and massively multilingual speech-to-text translation,” arXiv preprint arXiv:2007.10310, 2020.
- [29] e. a. Cieri Christopher, “Fisher english training speech part 1 speech ldc2004s13,” Web Download. Philadelphia: Linguistic Data Consortium, 2004.
- [30] e. a. Cieri Christopher, “Fisher english training part 2, speech ldc2005s13,” Web Download. Philadelphia: Linguistic Data Consortium, 2005.
- [31] R. Taori et al., Stanford alpaca: An instruction-following llama model, 2023.
- [32] S. Chen et al., “Wavlm: Large-scale self-supervised pre-training for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, pp. 1505–1518, 2021.
- [33] C.-C. Chiu, J. Qin, Y. Zhang, J. Yu, and Y. Wu, “Self-supervised learning with randomprojection quantizer for speech recognition,” in International Conference on Machine Learning, PMLR, 2022, pp. 3915–3924.
- [34] A. Bapna et al., “Mslam: Massively multilingual joint pre-training for speech and text,” arXiv preprint arXiv:2202.01374, 2022.
- [35] Seamless Communication et al., Seamlessm4t: Massively multilingual & multimodal machine translation, 2023. arXiv: 2308.11596 [cs.CL].
- [36] X. Li et al., “Multilingual speech translation from efficient finetuning of pretrained models,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), C. Zong, F. Xia, W. Li, and R. Navigli, Eds., Online: Association for Computational Linguistics, Aug. 2021, pp. 827–838. DOI: 10.18653/v1/2021.acl-long.68. [Online]. Available: https://aclanthology.org/2021.acl-long.68.
- [37] S. Seo, D. Kwak, and B. Lee, “Integration of pre-trained networks with continuous token interface for end-to-end spoken language understanding,” in Proc. ICASSP, 2022, pp. 7152– 7156.
- [38] Y. Wang, A. Boumadane, and A. Heba, “A fine-tuned wav2vec 2.0/hubert benchmark for speech emotion recognition, speaker verification and spoken language understanding,” arXiv preprint arXiv:2111.02735, 2022.
- [39] A. Derington, H. Wierstorf, A. Özkil, F. Eyben, F. Burkhardt, and B. W. Schuller, “Testing speech emotion recognition machine learning models,” arXiv preprint arXiv:2312.06270, 2023.
- [40] B. T. Willard and R. Louf, “Efficient guided generation for llms,” arXiv preprint arXiv:2307.09702, 2023.
- [41] J. Wei et al., “Chain-of-thought prompting elicits reasoning in large language models,” Advances in Neural Information Processing Systems, vol. 35, pp. 24824–24837, 2022.
- [42] L. Kaiser et al., “One model to learn them all,” arXiv preprint arXiv:1706.05137, 2017.
- [43] C. Raffel et al., “Exploring the limits of transfer learning with a unified text-to-text transformer,” The Journal of Machine Learning Research, vol. 21, no. 1, pp. 5485–5551, 2020.
- [44] Y.-C. Chen et al., “Speechnet: A universal modularized model for speech processing tasks,” arXiv preprint arXiv:2105.03070, 2021.
- [45] J.-B. Alayrac et al., “Flamingo: A visual language model for few-shot learning,” Advances in Neural Information Processing Systems, vol. 35, pp. 23716–23736, 2022.
- [46] J. Li, D. Li, C. Xiong, and S. Hoi, “Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation,” in International Conference on Machine Learning, PMLR, 2022, pp. 12888–12900.
- [47] J. Y. Koh, R. Salakhutdinov, and D. Fried, “Grounding language models to images for multimodal inputs and outputs,” in International Conference on Machine Learning, PMLR, 2023, pp. 17283–17300.
- [48] Z. Peng et al., “Kosmos-2: Grounding multimodal large language models to the world,” arXiv preprint arXiv:2306.14824, 2023.
- [49] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, “Learning to prompt for vision-language models,” International Journal of Computer Vision, vol. 130, no. 9, pp. 2337–2348, 2022.

- [50] S. Deshmukh, B. Elizalde, R. Singh, and H. Wang, “Pengi: An audio language model for audio tasks,” arXiv preprint arXiv:2305.11834, 2023.
- [51] Y. Gong, H. Luo, A. H. Liu, L. Karlinsky, and J. Glass, “Listen, think, and understand,” arXiv preprint arXiv:2305.10790, 2023.
- [52] Y. Shu et al., “Llasm: Large language and speech model,” arXiv preprint arXiv:2308.15930, 2023.
- [53] J. Iranzo-Sánchez et al., “Europarl-st: A multilingual corpus for speech translation of parliamentary debates,” in ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020, pp. 8229–8233.
- [54] R. Lotfian and C. Busso, “Building naturalistic emotionally balanced speech corpus by retrieving emotional speech from existing podcast recordings,” IEEE Transactions on Affective Computing, vol. 10, no. 4, pp. 471–483, Oct. 2019. DOI: 10.1109/TAFFC.2017.2736999.
- [55] R. Paturi, S. Srinivasan, and X. Li, “Lexical Speaker Error Correction: Leveraging Language Models for Speaker Diarization Error Correction,” in Proc. INTERSPEECH 2023, 2023, pp. 3567–3571. DOI: 10.21437/Interspeech.2023-1982.
- [56] e. a. Cieri Christopher, “2000 hub5 english evaluation speech ldc2002s09,” Web Download. Philadelphia: Linguistic Data Consortium, 2002.

Table 11: Various hyper-parameters for our models

Parameter Multitask-WLM Multitask-BRQ

Audio encoder WavLM-Large Best-RQ LLM Flan-T5-XL Flan-T5-XL Convolution blocks 2 2 Kernel sizes [3, 3] [3, 3] Strides [2, 2] [2, 1] Audio encoder ampling rate 50Hz 25Hz Down-sampling factor 4x 2x # parameters 2.9B 3.2B # trainable parameters 28.3M 26.8M LoRA rank(r) 16 16 Learning rate 0.001 0.005 Warm-up steps 100 50000 Effective batch size 768 2048

### A Appendix

#### A.1 Audio Encoder Pre-training

Our audio encoder is a 24-layer Conformer model with feature dimension of 768 and attention head of 8. The total number of parameters of this encoder model is 300M. We adopt the BEST-RQ [33] method, which pre-trains the model to predict the masked speech signals with labels generated from a random-projection quantizer. The quantizer projects the speech inputs with a randomly initialized matrix, and performs a nearest-neighbor lookup in a randomly-initialized codebook. Neither the projection matrix nor the codebook is updated during pre-training. We build an internal pre-training dataset containing 300K hours English audios. The pre-training uses mask span of 10 with total effective masking ratio about 40%. The learning rate schedule follows the transformer learning rate schedule with peak value of 0.0005 and warm-up of 50K steps. AdamW optimizer is adopted with weight decay of 0.01. Since the encoder has 4 times temporal-dimension reduction, the quantization with random projections stacks every 4 frames for projections. We use 16 individual codeboooks, where the vocab size of each codebook is 8192 and the dimension is 16. The model is pre-trained for 500K steps in total.

#### A.2 Hyper-parameters

We train all our models on a cluster of machines with 8 A100 GPUs, each having 40GB of memory. Pytorch Lightning framework3 is used for our implementation. A 5% subset of the complete training dataset for a model is used as a validation set to choose the important hyper-parameters. For the multitask models, the sample weights for each dataset are also chosen using this validation set. Further, all models are trained till the validation loss converges and it does not improve for 5 consecutive epochs. The details of the learning rate, warmup steps, batch size for the Multitask-WLM and Multitask-BRQ are summarized in the Table 11. To adhere to memory constraints of the GPUs, we filter out any training sample where the sequence length of the audio is greater than 900 or the target label is greater than 600. Since, the WavLM Large samples audio features at 50Hz rate, we use two successive 1-D convolution blocks (kernel size=3, stride=2) for the Task-FT and Multitask-WLM model to downsample the audio four times and achieve the desired sampling rate of 12.5Hz. For BestRQ-based audio encoder, the sampling rate is 25Hz and hence the stride of the second convolution block is set to 1 to ensure the output sampling rate is 12.5Hz.

#### A.3 Tasks

We provide the details about our training tasks below as well as provide some qualitative examples in the Table 10 to better understand the tasks.

ASR: We use a combination of 5 publicly available datasets for the ASR task, which totals to 3k hours of paired audio and text data. We evaluate performance on the standard benchmarks for ASR.

ST: We train our models to predict translations in multiple different languages from the audios recorded with English speech. The tokenizer of the backbone LLM limits the choice of what can be a potential target language. For our case, we train and evaluate on German, French, and Romanian

3

https://www.pytorchlightning.ai

translations from the EuroParl dataset [53]. We also augment the training data with German and Catalan translations from the CoVost2 [28] dataset .

IC/SF: We train and evaluate our models on a subset of the SLURP dataset [25] that consists of 10 intent classes and 4 slot labels. This also allows us to study the generalization ability of our models to unseen class labels and we separately study it in the Section 4.2. The intent classes and slot labels that are chosen for the "seen" subset are the ones that occur most frequently in the training data. The training prompt used for this task is designed to contain the description of each class label.

KWE: The goal of this task is to identify important keywords in the content of the speech in the audio. Since no publicly available dataset exists for this task, we synthetically extract keywords from the ground truth transcripts using a text-based keyword extraction model4. These are then used as labels for training and evaluating our models.

KWS: This is a binary classification task to detect whether a specified keyword was spoken in the audio or not. We create positive samples by randomly selecting keywords from the ground truth transcripts and negative samples by choosing a keyword that does not appear in the transcript. Positive and negative examples are created in 70-30 ratio respectively for both training and evaluation.

ER: For emotion recognition, we classify speech into one of four main emotion classes: neutral, happy, sad, and angry, chosen based on the availability of the training samples in the MSP-Podcast v1.11 dataset [54]. We report metrics on the corresponding four-emotion subset of the Test1 split of the dataset.

ASC: For audio sentiment classification, we classify speech as positive, negative, or neutral in sentiment. The sentiment labels were obtained by thresholding the valence scale (annotated from 1 to

- 7) with 3 and 5. We train on the entire training split of the MSP-Podcast v1.11 dataset, and evaluate on the corresponding Test1 split.

SC: For speaker counting, we identify whether one or two speakers are present. We train on segments from Fisher dataset transcripts [29, 30] with one or two speakers, and evaluate on the Fisher test split used in [55].

AC: We train our models to classify speech into five accents of English language: Canadian, Indian, Australian, British, and American, using metadata from the Mozilla Common Voice dataset.

SNS: In this task, we identify whether speech is present in the audio. We collect a diverse set of audios with and without speech for training our models and evaluate them on a combination of speech segments from Hub5 [56] dataset and held-out non-speech segments in our in-house collection.

4

https://huggingface.co/Voicelab/vlt5-base-keywords

