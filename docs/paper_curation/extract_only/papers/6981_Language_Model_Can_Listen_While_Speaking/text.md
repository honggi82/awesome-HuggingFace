# arXiv:2408.02622v1[cs.CL]5Aug2024

## Language Model Can Listen While Speaking

Ziyang Ma1,2 Yakun Song1,2 Chenpeng Du2 Jian Cong2 Zhuo Chen2 Yuping Wang2 Yuxuan Wang2 Xie Chen1∗

1MoE Key Lab of Artificial Intelligence, X-LANCE Lab, Shanghai Jiao Tong University 2 ByteDance Inc.

### Abstract

Dialogue serves as the most natural manner of human-computer interaction (HCI). Recent advancements in speech language models (SLM), have significantly enhanced speech-based conversational AI. However, these models are limited to turn-based conversation, lacking the ability to interact with humans in real-time spoken scenarios, for example, being interrupted when the generated content is not satisfactory. To address these limitations, we explore full duplex modeling (FDM) in interactive speech language models (iSLM), focusing on enhancing real-time interaction and, more explicitly, exploring the quintessential ability of interruption. We introduce a novel model design, namely listening-while-speaking language model (LSLM), an end-to-end system equipped with both listening and speaking channels. Our LSLM employs a token-based decoder-only TTS for speech generation and a streaming self-supervised learning (SSL) encoder for real-time audio input. LSLM fuses both channels for autoregressive generation and detects turn-taking in real time. Three fusion strategies—early fusion, middle fusion, and late fusion—are explored, with middle fusion achieving an optimal balance between speech generation and real-time interaction. Two experimental settings, command-based FDM and voice-based FDM, demonstrate LSLM’s robustness to noise and sensitivity to diverse instructions. Our results highlight LSLM’s capability to achieve duplex communication with minimal impact on existing systems. This study aims to advance the development of interactive speech dialogue systems, enhancing their applicability in real-world contexts2.

Index Terms Full Duplex Modeling, Interactive Speech Language Model

### 1 Introduction

Dialogue is the most natural way of human-computer interaction (HCI). With the rapid development of GPT-style [29] large language models (LLM) and the scaling of Transformer-style [39] architectures, textual conversational AI, such as ChatGPT [27, 1] and LLaMA [36, 37], have become a significant part of daily life. However, these models are limited to text input and output and cannot interact directly with humans in arbitrary scenarios.

Incorporating spoken and auditory interfaces into conversational AI enhances HCI convenience. Leveraging techniques from text LLMs, the speech language model (SLM) processes speech similarly to text. This paradigm involves encoding the speech signal into discrete tokens or continuous embeddings, modeling them with a language model, and decoding the speech tokens or embeddings back to the speech signal. Some studies [19, 17, 26] utilizes this paradigm for speech continuation, generating expressive speech and natural multi-round dialogue. Other research employs this paradigm

∗Corresponding author. 2Demo can be found at https://ddlbojack.github.io/LSLM

Preprint.

to task-specific applications, such as decoder-only high-fidelity TTS [40, 3, 31, 13] and decoder-only streaming ASR [33, 38, 4, 8] Moreover, SpeechGPT [48] and LauraGPT [5] initialize SLMs using LLMs, expanding speech tokens to the LLM vocabulary and continuing training on speech. This empowers SLM to comprehend semantic information and equips SLM with dialogue capability. Despite these advances, all these models are limited to turn-based conversations and cannot handle real-time sound or interruptions, limiting their applicability in real-life scenarios.

Interaction and turn-taking are essential abilities for natural communication among humans. At the dawn of the end-to-end speech dialogue system explosion, we focus on investigating Full Duplex Modeling (FDM) in interactive Speech Language Models (iSLM), a crucial topic affecting user experience. Lin et. al [22] proposes to process real-time audio input with a separate comprehension module. Other works [49, 41] suggest modifying the order in which text tokens are organized in the LLM to tackle the duplex modeling problem. All these models are based on text-centric LLMs that require external ASR and TTS modules for spoken dialogue. As a result, latency remains perceivable and the paralinguistic ability is still lacking. We believe the FDM capability should be an intrinsic capability of SLMs, enabling simultaneous listening and speaking.

To engage FDM capability for iSLM, we propose Listening-while-Speaking Language Model (LSLM), an end-to-end model with both listening and speaking channels. The proposed LSLM uses a token-based decoder-only TTS to model the ability to speak and a streaming self-supervised learning (SSL) encoder to model the ability to listen. LSLM fuses these two channels and detects turn-taking in real time. We explore three strategies for fusing duplex signals: Early Fusion, Middle Fusion, and Late Fusion. Experiments demonstrate that middle fusion achieves a good balance between speech generation and real-time interaction capabilities.

In addition, interactive dialogue systems for realistic scenarios have two important features: 1) Listening channels are not always clean. Users may interact with iSLMs in different scenarios, containing high-frequency noise (e.g., telephone ringing) and low-frequency noise (e.g., white noise). 2) It is possible that the iSLM interacts with an unseen speaker. iSLMs should recognize and respond to new voices and instructions, not dismiss them as noise. Therefore, iSLM should have both robustness to noise and sensitivity to unseen speakers. To test LSLM, we designed two scenarios: Command-based FDM, where LSLM is interrupted by a specific command, and Voice-based FDM, where LSLM can be interrupted by various words from unseen speakers. Experimental results show that LSLM with a listening channel is robust to noisy input and sensitive to turning-taking.

Our contributions are summarized as follows:

- 1. We formulate an important task, Full Duplex Modeling (FDM), applied in the interactive speech language model (iSLM).
- 2. We propose Listening-while-Speaking Language Model (LSLM), an end-to-end single model with the focus of modeling the turn-taking problem. LSLM can listen to the outside signal and provide feedback in real time while speaking.
- 3. We introduce three methods for fusing duplex signals: Early Fusion, Middle Fusion, and Late Fusion, with Middle Fusion providing the optimal tradeoff between speech generation and real-time interaction.
- 4. We tested the FDM ability of the proposed LSLM in two scenarios: Command-based FDM and Voice-based FDM. Experiments indicate that our proposed LSLM can achieve duplexing capability with little impact on the previous system.

### 2 Related Work

Figure 1 illustrates the distinctions between simplex, half duplex, and full duplex speech language models from a telecommunication perspective. An SLM with full duplex modeling (FDM) capability can be referred to as an interactive speech language model (iSLM).

#### 2.1 Simplex and Half Duplex Speech Language Model

Simplex SLMs, depicted in Figure 1(A) and 1(B), are limited to a single channel, either for listening or speaking. With the assistance of LLM, simplex SLMs exhibit strong understanding capabilities.

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

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

- Figure 1: Illustration of simplex, half duplex, and full duplex speech language models. (A): Simplex speech language model with listening ability. (B): Simplex speech language model with speaking ability. (C): Half duplex speech language model with both listening and speaking abilities. (D): Full duplex speech language model can listen while speaking.

Representative works include LLM-based ASR [46, 24, 45, 32], LLM-based speech translation [28, 7, 16, 6], and LLM-based speech emotion understanding [44, 21, 20]. Similarly, simplex SLMs have demonstrated robust generation capabilities, as seen in LLM-based TTS [15, 25, 18, 31]. Some research leverages the powerful in-context learning capabilities of LLMs to extend task-specific abilities to more universal applications, such as speech understanding [11], audio understanding [14], or both [35, 9, 10]. Despite their growing power and versatility, simplex SLMs are limited to one-way communication (either human → machine or machine → human). LLMs have facilitated a paradigm shift from simplex models to half-duplex models, also known as turn-based models, as shown in Figure 1(C). Prominent models include SpeechGPT [48], LauraGPT [5], and VioLA [42]. While these half duplex models can both listen and speak, they are constrained to performing only one action at the same instant, thus failing to address the turn-taking problem.

#### 2.2 Full Duplex Speech Language Model

Full duplex SLMs, as shown in Figure 1(D), have the capability to listen and speak simultaneously, allowing for turn-taking whenever a human interrupts the machine. Recent efforts [49, 41] have attempted to build full duplex capabilities on text-centric LLMs with cascade ASR and TTS modules. Cutting-edge products like GPT-4o 3 and Moshi 4 exhibit full duplex capability in their spoken dialogue systems. Despite these advancements, there are no publicly available open-source models or detailed analyses of full duplex SLMs. This gap highlights the need for further research and development to fully understand and optimize full duplex capability in speech language models.

### 3 Full Duplex Modeling (FDM)

A simplex or half duplex spoken dialogue system can be modeled by finding the parameters θ that maximize the log-likelihood function, formulated as:

max

θ

(C,R)∈D

- 3https://openai.com/index/hello-gpt-4o
- 4https://moshi.chat

log Pθ(R|C), (1)

where (C,R) represents the context-response pairs in the dataset D and Pθ(R|C) is the probability of the response R given the context C and parameters θ. More specifically, if the spoken dialogue system is modeled by an autoregressive language model where the response R is generated token by token, the training loss L(θ) for each sample is expressed as:

T

log Pθ(rt|R1:t−1,C), (2)

L(θ) = −

t=1

where R1:t−1 = [r1,r2,...,rt−1] and T is the sequence length. During the inference phase, the model can only predict the next token autoregressively based on the previous output within the current channel, without information from other channels.

In modeling a full duplex spoken dialogue system within an autoregressive language model, the model needs to predict the next token rt in the response R not only based on the context C and the generated response history R1:t−1 = [r1,r2,...,rt−1] in the current channel, but also by utilizing information S1:t−1 = [s1,s2,...,st−1] from another channel simultaneously. Here we extend the modeling approach used for simplex or half duplex dialogue systems to accommodate the requirements of full duplex modeling (FDM). The training loss L(θ) is now formulated as:

T

log Pθ(rt|R1:t−1,S1:t−1,C) (3)

L(θ) = −

t=1

A key point in FDM is that the sequence S is produced in real time and unpredictably. Taking the full duplex speech language model as an example, at the inference step t − 1, the current speaking channel generates output rt−1 and listening channel acquired input st−1 are fed into the model simultaneously, influencing the prediction of the speaking channel’s next step output rt. This modeling approach endows the system with a full duplex ability, enabling it to effectively leverage the multi-channel information during dialogue, thereby improving the accuracy and fluency of the real-time interaction capability.

### 4 Proposed LSLM

The core difference between LSLM and previous speech language models lies in its capability to simultaneously speak and listen. We first introduce the speaking capability of LSLM, followed by its listening capability, and finally, we discuss various fusion methods that integrate these capabilities, endowing LSLM with full duplex ability.

#### 4.1 Speaking Ability

To simulate the speaking ability of the LSLM, we utilize an autoregressive token-based TTS model. Unlike VALL-E-styled models that combine autoregressive (AR) and non-autoregressive (NAR) approaches with multi-layer residual vector quantization (RVQ) tokens, our model employs a single layer of discrete audio tokens. This design better meets the requirements for real-time interaction, as it eliminates the need to wait for the completion of AR token synthesis before performing NAR operations. Given target speech XR, an SSL encoder Enc is utilized to obtain a continuous embedding R, which can be written as:

R = Enc(XR). (4)

To train an autoregressive TTS model based on discrete tokens, we quantize the speech embedding R, denoted by:

Rq = Qnt(R), (5) where Qnt is the discretization operation and Rq are the discrete tokens. Given the context information C, in this scenario the text content to be synthesized, the model synthesizes the corresponding speech discrete tokens autoregressively. We minimize the negative log-likelihood of the target sequence to train the decoder-only model, conditioned on the preceding tokens and the context. The loss function is defined as:

tEOS

log P(rtq|R1:q t−1,C;θS), (6)

L(θS) = −

t=1

###### Context token Speaking token Listening token

IRQ

LSLM

......

Streaming SSL Encoder

[Figure 35]

Noise Noise + Voice

- Figure 2: Proposed LSLM. The model contains a decoder-only Transformer to generate speaking tokens and a streaming SSL encoder to process listening tokens. An interruption token (IRQ) is added to allow the model to terminate early if a turn-taking occurs.

where θS are the parameters to model speaking ability, tEOS represents the time step at which the end-of-sequence token is reached, rtq is the target discrete token at time step t, R1:q t−1 denotes the sequence of all previous tokens up to time step t − 1, and C is the text content to be synthesized. During inference, the model samples rˆtq from a conditional distribution based on the already generated tokens Rˆ1:q t−1 and the context C. The process is described by the following equation:

rˆtq ∼ P(rtq|Rˆ1:q t−1,C;θS). (7) A vocoder Dec is employed to recover the speech signal XˆR from discrete tokens Rˆq, donated by:

XˆR = Dec(Rˆq,A), (8)

where A is the acoustic prompt providing the timbre of the synthesized speech. This decoupling of timbre from content allows the AR model to focus more on semantic information rather than paralinguistic information.

#### 4.2 Listening Ability

Given the audio input XS of the listening channel, the same SSL encoder Enc in Equation 4 is used to obtain a continuous embedding S, which can be written as:

S = Enc(XS), (9)

where XS can be a variety of sound signals, including environmental noise and human speech. Unlike training the speaking ability, which involves a discretization module, the listening channel embedding S is fed into the neural network end-to-end via a projection module Proj, which can be written as:

Sp = Proj(S), (10) where the listened audio signal is mapped to a space that can be processed by the AR model.

#### 4.3 FDM Ability

LSLM has two channels: speaking and listening. At time step t, all previous information of the speaking channel R1:q t−1 and the processed information of the listening channel S1:pt−1 are considered

by the model simultaneously. Here we revise Equation 6 as follows:

L(θLS) = − ttIRQ=1 log P(rtq|R1:q t−1,S1:pt−1,C;θLS) if turn-taking, − ttEOS=1 log P(rtq|R1:q t−1,S1:pt−1,C;θLS) otherwise.

(11)

where θLS are the parameters to model the proposed LSLM with listening-while-speaking ability. In addition to the EOS token, we add an interruption token IRQ to the tokenizer vocabulary to allow the model to terminate early if turn-taking occurs. For example, if a human interrupts, the model should stop speaking within a detection interval µ seconds after the interruption starts. During inference, the

model samples rˆtq from a conditional distribution based on the already generated tokens Rˆ1:q t−1, the context C, and most important, real-time listened audio tokens S1:pt−1. The revised formula from Equation 8 is written as follows:

rˆtq ∼ P(rtq|Rˆ1:q t−1,S1:pt−1,C;θLS), (12)

in which, an essential requirement for the SSL encoder Enc is that it is streaming. Thus, LSLM can obtain real-time audio features during inference. This is detailed further in Section 5.1.

To comprehensively explore the integration of a listening channel to the proposed LSLM, we try to fuse the listening channel and the speaking channel with early, middle, and late methods, as shown in Figure 3.

Early Fusion integrates the listening and speaking channels at the input embeddings before autoregressive prediction.

Middle Fusion merges the listening and speaking channels at each Transformer block. Specifically, in addition to the hidden states of the speaking channel and positional embeddings, the listening channel is additionally added to the input of each Transformer block.

Late Fusion combines the channels at the output logits before the softmax operation.

###### Listening token Speaking token Hidden feature

| | |
|---|---|
| | |

Transformer Decoder Layer

Transformer Decoder Layer Transformer Decoder Layer

| | |
|---|---|
| | |

###### Early Fusion Middle Fusion Late Fusion

Figure 3: Different model designs to integrate the listening channel to the proposed LSLM.

### 5 Setup

#### 5.1 Model Details

The backbone of the proposed LSLM employs a decoder-only Transformer architecture consisting of 12 Transformer blocks, 12 attention heads, 768 embedding dimensions, and 3072 feed-forward layer dimensions, resulting in 106M parameters. SSL encoder vq-wav2vec [2] is employed to extract audio features and further convert speech features to discrete tokens. vq-wav2vec, a fully convolutional self-supervised pre-trained model with 20 layers of 1D convolutional neural networks with 34M parameters, is naturally suitable for streaming audio feature extraction. A simple linear layer serves as the projection module to adapt the listening channel features to the AR model. A GAN-based token-to-waveform vocoder [12] is utilized to recover discrete audio tokens to speech waveform.

#### 5.2 Data Details

We evaluate the proposed LSLM under two full duplex modeling (FDM) settings: command-based FDM and voice-based FDM. Table 1 summarizes the datasets and experimental settings. For the TTS datasets, we utilize the LibriTTS dataset [47] with 585 hours of speech-text pairs for training and validation. LibriTTS-testsetB [12] is adopted for testing, which contains 500 utterances sampled from the test-clean subset of LibriTTS with 37 unseen speakers. Background noise is uniformly sourced from the Freesound portion of the MUSAN dataset [34], which includes high-frequency noise such as telephone ringing and sounds of the explosion, as well as low-frequency noise such as white noise and traffic noise. The model needs to distinguish the human voice from the noise, so as to avoid turning-taking with any random input signals and avoid trivial solutions. Different interruption data is constructed based on the FDM settings.

Command-based FDM. In this setting, LSLM can only be interrupted by specific keywords. Timbre of 22 boutique speakers from SEED-TTS [31] is used to synthesize the command "Honey" for the command-based FDM.

Voice-based FDM. In this setting, LSLM can be interrupted by a variety of different words. The Speech Commands Dataset [47] is a set of one-second audio, each containing a single spoken English word. We split the dataset into training, validation, and test sets in an 8 : 1 : 1 ratio, resulting in 51,088, 6,798, and 6,835 pieces of data, respectively. In addition, we use a speaker independence setting, which guarantees that the speakers in the test set do not appear in the training set, simulating more challenging and realistic scenarios.

- Table 1: Data details involved in training LSLM. SD means speaker dependence, while SI means speaker independence here.

| |Command-based FDM(SD) Voice-based FDM(SI)<br><br>|
|---|---|
|TTS<br><br>train val test<br><br>|LibriTTS-train [47] LibriTTS-dev-clean/other [47] LibriTTS-testsetB [12]|
|Interruption<br><br>train val test|Say_Honey<br><br>Speech Commands Dataset-train [43] Speech Commands Dataset-dev [43] Speech Commands Dataset-test [43]<br><br>|
|Noise all|Freesound portion of MUSAN [34]|

#### 5.3 Training and Inference Details

We train the model with TTS, interruption, and noise datasets for 20 epochs. For each sample, noise is added with a 50% probability, and interruption with a 50% probability, to the listening tokens. If a sample is selected to include an interruption, we modify the sentence to output the IRQ token µ = 0.5 seconds after the start of the interruption and then stop outputting the remaining speaking tokens. This ensures that the model can correctly handle different audio signal combinations in the listening channel. The optimization strategy involves using AdamW [23] with a max learning rate of 5 × 10−4 without weight decay and a batch size of 4. The learning rate scheduler involves a warm-up phase for the first 5,000 steps, followed by a cosine decay of the learning rate. Validation is performed at the end of each epoch, and the checkpoint with the lowest loss is selected for inference. The generation process employs Top-P sampling with a top-p value of 0.99 and a temperature of 1.0.

### 6 Experiments

#### 6.1 Evaluation Metrics

TTS capability evaluation. We evaluate whether the speech generation capability is affected by the full duplex modeling in the proposed LSLM. The word error rate (WER) comparing the generated

speech to the original text is considered as the TTS capability evaluation metrics using Whisper large v35 [30].

Interactive capability evaluation. Interactivity capability evaluation aims to measure how well the proposed LSLM responds to real-time and unpredictable input from the listening channel. A successful turn-taking is defined as the model stopping speaking within the [0,2µ] interval (1 second in our setting) after the interruption begins. Based on this, we categorize the outcomes into four cases: interruption and hit (TP), interruption and miss (FN), no interruption and hit (FP), and no interruption and miss (TN). From these cases, we construct a confusion matrix and calculate the Precision, Recall, and F1 score. These metrics consider both the success rate of turn-taking (Recall) and the rate of misjudgments (Precision), providing a comprehensive evaluation of the model’s interactivity capabilities.

#### 6.2 Experiments results

We conduct a series of experiments to evaluate the command-based and voice-based FDM for both TTS capability and interactive capability. For TTS capability, we use a test set consisting of 500 utterances, referred to as LibriTTS-testsetB [12], without any interruptions in the listening channel. The primary metric for this evaluation is WER. For the interactive capability evaluation, we employ a set of 1000 utterances divided into two equal parts: 500 utterances with interruptions at a random time step and 500 utterances without interruptions. Interactive capability is measured using Precision, Recall, and F1 Score.

Additionally, we test the models under two listening channel conditions: without noise, donated

- as Clean, and with noise, donated as Noise. For the baseline Vanilla TTS model, since it does not involve a listening channel, the input is inherently clean. By comparing the clean scenarios, we assess whether the intrinsic TTS capability is affected. Additionally, integrating noisy external inputs provides a better simulation of real-world scenarios.

Command-based FDM. For command-based FDM, we test the three architectures described in Section 4.3 to fuse the listening channel and the speaking channel, which are early fusion (LSLMEF), middle fusion (LSLMMF), and late fusion (LSLMLF). The results are shown in Table 2. For TTS capability, The baseline Vanilla TTS model without a listening channel achieves a WER of 4.28%. LSLMMF outperforms LSLMEF and LSLMLF with a WER of 4.05% in clean conditions and maintains a relatively low WER of 4.51% in noisy conditions. The TTS ability of LSLMEF shows a notable decrease, likely due to the fusion of input embeddings, making it difficult for the model to distinguish the information of the listening and speaking channels, negatively impacting the next token prediction. For interactive capability, all three architectures perform well with an oracle clean listening channel. However, LSLMLF shows a notable drop in performance under noisy conditions, with the F1 score falling to 94.89%. Observing that the late fusion method appears to mainly affect the precision score when the listening channel is noisy, suggests that the LSLMLF model reduces the discrimination of noise and human voice, leading to misjudgments of interruptions. In summary, the middle fusion approach demonstrates superior performance in TTS capability and competitive performance in interactive capability. Therefore, LSLMMF is concluded to be the best-performing model among those tested.

Voice-based FDM. We utilized a more diverse set of interruption commands compared to the command-based FDM and involved unseen speakers in the testing procedures. The best configuration from the command-based FDM, the LSLMMF model, was selected to evaluate the voice-based FDM capability. The results are shown in Table 3. LSLM shows a higher WER of 5.33% in clean conditions and 8.50% in noisy conditions compared to the Vanilla TTS model, demonstrating the challenges posed by the real-world turn-taking problem. Comparing the results with the commandbased FDM using the LSLMMF model, we find that the voice-based setting faces greater challenges in maintaining high performance, especially under noisy conditions with Precision at 87.69%, Recall

- at 82.77%, and an F1 score of 85.15%. The diverse set of interruption commands and the involvement of unseen speakers add complexity, resulting in higher error rates.

5https://github.com/openai/whisper

- Table 2: Experiments results on command-based FDM. Early fusion (LSLMEF), middle fusion (LSLMMF), and late fusion (LSLMLF) are considered.

|Model|Listening Channel<br><br>|TTS Capability WER(%) ↓|Interactive Capability Precision(%)↑ Recall(%)↑ F1(%)↑|
|---|---|---|---|
|Vanilla TTS|- (Clean)<br><br>|4.28|- - -|
|LSLMEF<br><br>|Clean Noise|33.56 34.99|98.00 98.20 98.10 97.20 97.20 97.20|
|LSLMMF<br><br>|Clean Noise|4.05 4.51|97.80 98.19 98.00 97.58 97.18 97.38|
|LSLMLF|Clean Noise|4.37 6.87|97.99 97.80 97.89 93.06 96.79 94.89|

- Table 3: Experiments results on voice-based FDM. LSLM here utilizes the architecture of middle fusion.

|Model<br><br>|Listening Channel|TTS Capability WER(%) ↓|Interactive Capability Precision(%)↑ Recall(%)↑ F1(%)↑|
|---|---|---|---|
|Vanilla TTS|- (Clean)|4.28<br><br>|- - -|
|LSLM|Clean Noise|5.33 8.50|95.21 95.78 95.50 87.69 82.77 85.15|

Visualization. To investigate the turn-taking internal mechanism of LSLM, we visualize the probability distribution of IRQ tokens at different time steps during the generation process. Given that the IRQ token probability distribution varies significantly in order of magnitude across different time steps, we utilize a logarithmic scale for probability to enhance the clarity of the visualization. As illustrated in Figure 4, the probability of the IRQ token remains below 1 × 10−3 when the model is not interrupted. When the listening channel starts to receive the real-time turn-taking signal, LSLM senses whether it is an interruption or a noise. After a very short time, the IRQ token probability begins to increase. Shortly thereafter, this probability rises to a level where the IRQ token is sampled by the model during generation.

IRQ Probability

Start/End Interruption Detection Interval

10 1

| |
|---|

10 2

10 3

Probability(LogScale)

10 4

10 5

10 6

10 7

10 8

0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 Time (seconds)

Figure 4: Illustration of the probability distribution of IRQ tokens (being interrupted) over time. The logarithmic scale probability is used for clear visualization.

#### 6.3 Ablation Study

In this section, we conduct an ablation study on LSLM with middle fusion architecture to evaluate the impact of different training methods on the performance of TTS capability and interactive capability. The training methods are categorized as training from scratch (✗), loading the pre-trained model

and fixing the parameters (✓), and loading the pre-trained model and continuing training (✚). The detailed results are presented in Table 4.

The vanilla TTS model, trained from scratch, achieves a WER of 4.28% concerning TTS capability. For the interactive capability, the vanilla TTS model does not have a listening channel, hence no metrics are available. For the LSLM model, the best performance is observed when both the TTS backbone and streaming SSL encoder are loaded and continue training (✚ & ✚), achieving the lowest WER of 4.05% and highest Precision of 97.80%, Recall of 98.19%, and F1 Score of 98.00%. Some conclusions can also be drawn from these experiments. For example, the SSL encoder of the listening channel performs better when it can be continued training than fixed the parameters. One potential reason is that the SSL encoder has not encountered diverse noise during pre-training, creating a bottleneck for extracting audio with mixed human voice and noise when using fixed pre-trained parameters.

- Table 4: Ablation study on LSLM to evaluate the impact of different training methods. ✗ means training from scratch, ✓ means load the pre-training model and fix the parameters, ✚ means load the pre-training model and continue training. LSLM here utilizes the architecture of middle fusion.

|Model<br><br>|Training Method Speaking Listening|TTS Capability WER(%) ↓|Interactive Capability Precision(%)↑ Recall(%)↑ F1(%)↑|
|---|---|---|---|
|Vanilla TTS|✗ -|4.28<br><br>|- - -|
|LSLM|✗ ✓<br>✗ ✚<br><br><br>✓ ✓ ✓ ✚ ✚ ✓ ✚ ✚|4.82 4.67 6.64 4.64 4.46 4.05|97.80 97.99 97.89<br><br>95.60 95.98 95.79 97.89 83.60 90.18 97.60 98.18 97.89<br>96.43 92.54 94.44<br>97.80 98.19 98.00<br>|

### 7 Conclusion

In this paper, we address the challenges of enhancing real-time interaction by introducing full duplex modeling (FDM) in interactive speech language models (iSLM). We introduce listen-while-speaking language model(LSLM), an innovative end-to-end model designed to handle real-time turn-taking. LSLM integrates a token-based decoder-only TTS model for speech generation and a streaming SSL encoder for audio input, enabling simultaneous listening and speaking. We propose three strategies for fusing duplex signals: early fusion, middle fusion, and late fusion. Among these, Middle Fusion demonstrates a superior balance between speech generation and real-time interaction capabilities. The proposed LSLM is evaluated in two settings: command-based FDM and voice-based FDM. Our experiments show that LSLM is robust to noisy environments and responsive to diverse instructions from unseen speakers, achieving effective duplex communication with minimal impact on system performance. Our work is an initial exploration into full duplex interactive speech language models, and there is still a long way to go to achieve smooth human-computer speech interaction. There is a lot to explore in the future, such as developing speech-in speech-out dialogue systems with full duplex modeling ability, incorporating speaker-following capability to identify interrupting speakers, and exploring audiovisual co-guidance for improved turn-taking.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Alexei Baevski, Steffen Schneider, and Michael Auli. vq-wav2vec: Self-supervised learning of discrete speech representations. In Proc. ICLR, 2020.
- [3] Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, et al. AudioLM: a language modeling approach to audio generation. Proc. TASLP, 2023.

- [4] Peikun Chen, Sining Sun, Changhao Shan, Qing Yang, and Lei Xie. Streaming decoder-only automatic speech recognition with discrete speech units: A pilot study. Proc. Interspeech, 2024.
- [5] Qian Chen, Yunfei Chu, Zhifu Gao, Zerui Li, Kai Hu, Xiaohuan Zhou, Jin Xu, Ziyang Ma, Wen Wang, Siqi Zheng, et al. LauraGPT: Listen, attend, understand, and regenerate audio with gpt. arXiv preprint arXiv:2310.04673, 2023.
- [6] Xi Chen, Songyang Zhang, Qibing Bai, Kai Chen, and Satoshi Nakamura. LLaST: Improved end-to-end speech translation system leveraged by large language models. arXiv preprint arXiv:2407.15415, 2024.
- [7] Zhehuai Chen, He Huang, Andrei Andrusenko, Oleksii Hrinchuk, Krishna C Puvvada, Jason Li, Subhankar Ghosh, Jagadeesh Balam, and Boris Ginsburg. SALM: Speech-augmented language model with in-context learning for speech recognition and translation. In Proc. ICASSP, 2024.
- [8] Zhehuai Chen, He Huang, Oleksii Hrinchuk, Krishna C Puvvada, Nithin Rao Koluguri, Piotr Zelasko,˙ Jagadeesh Balam, and Boris Ginsburg. BESTOW: Efficient and streamable speech language model with the best of two worlds in gpt and t5. arXiv preprint arXiv:2406.19954, 2024.
- [9] Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, and Jingren Zhou. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. arXiv preprint arXiv:2311.07919, 2023.
- [10] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024.
- [11] Keqi Deng, Guangzhi Sun, and Philip C Woodland. Wav2prompt: End-to-end speech prompt generation and tuning for llm in zero and few-shot learning. arXiv preprint arXiv:2406.00522, 2024.
- [12] Chenpeng Du, Yiwei Guo, Feiyu Shen, Zhijun Liu, Zheng Liang, Xie Chen, Shuai Wang, Hui Zhang, and Kai Yu. UniCATS: A unified context-aware text-to-speech framework with contextual vq-diffusion and vocoding. In Proc. AAAI, 2024.
- [13] Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, et al. CosyVoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. arXiv preprint arXiv:2407.05407, 2024.
- [14] Yuan Gong, Hongyin Luo, Alexander H Liu, Leonid Karlinsky, and James Glass. Listen, think, and understand. Proc. ICLR, 2024.
- [15] Hongkun Hao, Long Zhou, Shujie Liu, Jinyu Li, Shujie Hu, Rui Wang, and Furu Wei. Boosting large language model for speech synthesis: An empirical study. arXiv preprint arXiv:2401.00246, 2023.
- [16] Chao-Wei Huang, Hui Lu, Hongyu Gong, Hirofumi Inaguma, Ilia Kulikov, Ruslan Mavlyutov, and Sravya Popuri. Investigating decoder-only large language models for speech-to-text translation. Proc. Interspeech, 2024.
- [17] Eugene Kharitonov, Ann Lee, Adam Polyak, Yossi Adi, Jade Copet, Kushal Lakhotia, TuAnh Nguyen, Morgane Rivière, Abdelrahman Mohamed, Emmanuel Dupoux, et al. Text-free prosody-aware generative spoken language modeling. In Proc. ACL, 2022.
- [18] Mateusz Łajszczak, Guillermo Cámbara, Yang Li, Fatih Beyhan, Arent van Korlaar, Fan Yang, Arnaud Joly, Álvaro Martín-Cortinas, Ammar Abbas, Adam Michalski, et al. BASE TTS: Lessons from building a billion-parameter text-to-speech model on 100k hours of data. arXiv preprint arXiv:2402.08093, 2024.
- [19] Kushal Lakhotia, Eugene Kharitonov, Wei-Ning Hsu, Yossi Adi, Adam Polyak, Benjamin Bolte, Tu Anh Nguyen, Jade Copet, Alexei Baevski, Abdelrahman Mohamed, et al. On generative spoken language modeling from raw audio. Proc. TACL, 2021.

- [20] Zheng Lian, Haiyang Sun, Licai Sun, Jiangyan Yi, Bin Liu, and Jianhua Tao. AffectGPT: Dataset and framework for explainable multimodal emotion recognition. arXiv preprint arXiv:2407.07653, 2024.
- [21] Guan-Ting Lin, Cheng-Han Chiang, and Hung-yi Lee. Advancing large language models to capture varied speaking styles and respond properly in spoken conversations. Proc. ACL, 2024.
- [22] Ting-En Lin, Yuchuan Wu, Fei Huang, Luo Si, Jian Sun, and Yongbin Li. Duplex conversation: Towards human-like interaction in spoken dialogue systems. In Proc. SIGKDD, 2022.
- [23] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Proc. ICLR, 2019.
- [24] Ziyang Ma, Guanrou Yang, Yifan Yang, Zhifu Gao, Jiaming Wang, Zhihao Du, Fan Yu, Qian Chen, Siqi Zheng, Shiliang Zhang, et al. An embarrassingly simple approach for llm with strong asr capacity. arXiv preprint arXiv:2402.08846, 2024.
- [25] Paarth Neekhara, Shehzeen Hussain, Subhankar Ghosh, Jason Li, Rafael Valle, Rohan Badlani, and Boris Ginsburg. Improving robustness of llm-based speech synthesis by learning monotonic alignment. Proc. Interspeech, 2024.
- [26] Tu Anh Nguyen, Eugene Kharitonov, Jade Copet, Yossi Adi, Wei-Ning Hsu, Ali Elkahky, Paden Tomasello, Robin Algayres, Benoit Sagot, Abdelrahman Mohamed, et al. Generative spoken dialogue language modeling. Proc. TACL, 2023.
- [27] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Proc. Neurips, 2022.
- [28] Jing Pan, Jian Wu, Yashesh Gaur, Sunit Sivasankaran, Zhuo Chen, Shujie Liu, and Jinyu Li. Cosmic: Data efficient instruction-tuning for speech in-context learning. arXiv preprint arXiv:2311.02248, 2023.
- [29] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.
- [30] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Proc. ICML, 2023.
- [31] SeedSpeechTeam. Seed-TTS: A family of high-quality versatile speech generation models. arXiv preprint arXiv:2406.02430, 2024.
- [32] SeedSpeechTeam. Seed-ASR: Understanding diverse speech and contexts with llm-based speech recognition. arXiv preprint arXiv:2407.04675, 2024.
- [33] Frank Seide, Morrie Doulaty, Yangyang Shi, Yashesh Gaur, Junteng Jia, and Chunyang Wu. Speech ReaLLM–real-time streaming speech recognition with multimodal LLMs by teaching the flow of time. arXiv preprint arXiv:2406.09569, 2024.
- [34] David Snyder, Guoguo Chen, and Daniel Povey. MUSAN: A music, speech, and noise corpus. arXiv preprint arXiv:1510.08484, 2015.
- [35] Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. SALMONN: Towards generic hearing abilities for large language models. In Proc. ICLR, 2024.
- [36] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [37] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

- [38] Emiru Tsunoo, Hayato Futami, Yosuke Kashiwagi, Siddhant Arora, and Shinji Watanabe. Decoder-only architecture for streaming end-to-end speech recognition. Proc. Interspeech, 2024.
- [39] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Proc. Neurips, 2017.
- [40] Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, et al. Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint arXiv:2301.02111, 2023.
- [41] Peng Wang, Songshuo Lu, Yaohua Tang, Sijie Yan, Yuanjun Xiong, and Wei Xia. A full-duplex speech dialogue scheme based on large language models. arXiv preprint arXiv:2405.19487, 2024.
- [42] Tianrui Wang, Long Zhou, Ziqiang Zhang, Yu Wu, Shujie Liu, Yashesh Gaur, Zhuo Chen, Jinyu Li, and Furu Wei. VioLA: Unified codec language models for speech recognition, synthesis, and translation. arXiv preprint arXiv:2305.16107, 2023.
- [43] Pete Warden. Speech commands: A public dataset for single-word speech recognition. Dataset available from http://download. tensorflow. org/data/speech_commands_v0, 2017.
- [44] Yaoxun Xu, Hangting Chen, Jianwei Yu, Qiaochu Huang, Zhiyong Wu, Shi-Xiong Zhang, Guangzhi Li, Yi Luo, and Rongzhi Gu. Secap: Speech emotion captioning with large language model. In Proc. AAAI, 2024.
- [45] Guanrou Yang, Ziyang Ma, Fan Yu, Zhifu Gao, Shiliang Zhang, and Xie Chen. Mala-asr: Multimedia-assisted llm-based asr. Proc. Interspeech, 2024.
- [46] Wenyi Yu, Changli Tang, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. Connecting speech encoder and large language model for ASR. In Proc. ICASSP, 2024.
- [47] Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu. LibriTTS: A corpus derived from librispeech for text-to-speech. Proc. Interspeech, 2019.
- [48] Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, and Xipeng Qiu. SpeechGPT: Empowering large language models with intrinsic cross-modal conversational abilities. In Proc. EMNLP, 2023.
- [49] Xinrong Zhang, Yingfa Chen, Shengding Hu, Xu Han, Zihang Xu, Yuanwei Xu, Weilin Zhao, Maosong Sun, and Zhiyuan Liu. Beyond the turn-based game: Enabling real-time conversations with duplex models. arXiv preprint arXiv:2406.15718, 2024.

