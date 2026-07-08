# arXiv:2601.21181v1[cs.AI]29Jan2026

## MAD: Modality-Adaptive Decoding for Mitigating Cross-Modal Hallucinations in Multimodal Large Language Models

Sangyun Chung, Se Yeon Kim, Youngchae Chee, and Yong Man Ro* Integrated Vision Language Lab, KAIST, South Korea

{jelarum, seyeon.kim, litcoderr, ymro}@kaist.ac.kr

#### Abstract

Multimodal Large Language Models (MLLMs) suffer from cross-modal hallucinations, where one modality inappropriately influences generation about another, leading to fabricated output. This exposes a more fundamental deficiency in modality-interaction control. To address this, we propose Modality-Adaptive Decoding (MAD), a trainingfree method that adaptively weights modality-specific decoding branches based on task requirements. MAD leverages the model’s inherent ability to self-assess modality relevance by querying which modalities are needed for each task. The extracted modality probabilities are then used to adaptively weight contrastive decoding branches, enabling the model to focus on relevant information while suppressing cross-modal interference. Extensive experiments on CMM and AVHBench demonstrate that MAD significantly reduces cross-modal hallucinations across multiple audio-visual language models (7.8% and 2.0% improvements for VideoLLaMA2-AV, 8.7% and 4.7% improvements for Qwen2.5-Omni). Our approach demonstrates that explicit modality awareness through self-assessment is crucial for robust multimodal reasoning, offering a principled extension to existing contrastive decoding methods. Our code is available at https://github.com/top-yun/MAD

#### 1. Introduction

Multimodal Large Language Models (MLLMs) [1–9] have achieved remarkable progress in recent years. They move beyond text-only understanding to integrate diverse modalities such as vision, audio, and language. By jointly processing multiple sensory inputs, these models aim to emulate human-like multimodal perception. They enable a wide range of applications, from video question answering [10– 13] to audio-visual scene understanding [14–16]. This ability to reason across modalities marks a key step toward more holistic and comprehensive AI systems.

Despite their impressive progress, MLLMs still face

*Corresponding author.

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>Video<br><br>Question<br><br>Please describe the video<br><br>and sound in detail<br><br>Sound<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]|
|---|

Input

*visual hallucination *video-driven audio hallucination

[Figure 7]

[Figure 8]

[Figure 9]

In the video, a man is seen standing in front of a boat, giving thumbs up and

talking about the different types of fishing rods. He also demonstrates how to use a fish finder and provides tips on how to catch more fish with your fishing rod. The sound of the man's voice and the occasional splash of a fish jumping out of the water can be heard throughout the video.

Base

[Figure 10]

[Figure 11]

[Figure 12]

In the video, a man is seen standing in front of a red and white boat. He is wearing a blue shirt and talking to the camera. On the table in front of him,

there is a box with various tools. Beside the boat, there is a cooler. The man

MAD

appears to be preparing for some kind of work or project.

(ours)

Figure 1. Cross-modal hallucinations and their mitigation through Modality-Adaptive Decoding (MAD). The base model hallucinates non-existent visual content (red text) and audio events (blue text) when describing audio-visual inputs. MAD eliminates these hallucinations by adaptively suppressing cross-modal interference, producing accurate descriptions grounded in actual content.

a critical challenge beyond single-modality hallucination problems [17–21]. Conventional hallucinations typically involve generating incorrect or fabricated information within a single modality. In contrast, multimodal settings introduce a more subtle yet harmful failure pattern: cross-modal hallucinations [22, 23], where one modality improperly influences content generation about another. Figure 1, shows a clear example of visual hallucination and video-driven audio hallucination occur simultaneously. The input video contains strong visual cues, such as a boat. When asked to describe both video and audio, the base model exhibits visual hallucination—stating that the person “demonstrates how to use a fish finder and provides tips on catching fish.” More critically, because the visual modality exerted an inappropriate influence it fabricates audio events that do not exist in the input, such as “The sound of the man’s voice and the splash of a fish jumping out of the water,” even though such sounds do not exist in the input. Conversely,

audio-driven visual hallucinations occur when auditory signals cause the model to invent corresponding visual events.

Unlike traditional hallucinations which distort content within a single modality, cross-modal hallucinations reveal a deeper failure of modality separation—where information inappropriately influences and corrupts the understanding of another. Consequently, addressing this challenge requires more than just mitigating single-modality hallucination. The model must also possess the modality appropriateness judgment capability—the ability to self-assess the reliability, information content, and contextual relevance of each input modality for a given task. This is crucial because the model must effectively determine which modality is important for a specific task and how to utilize the modalities together. This complexity, involving multiple intertwined requirements, is why cross-modal hallucination is inherently difficult to resolve. The problem stems from a fundamental failure in modality-interaction control, rather than merely inadequate representations within a single modality. Specifically, the failure is rooted in fundamental issues– assigning appropriate modality weights, suppressing misleading modalities, and preserving boundaries.

Recent attempts to mitigate hallucinations in visionlanguage models (VLMs) [24–28] using training-free decoding methods. Contrastive Decoding [29] for visionlanguage models compares the output distribution of the original input with output distribution of a distorted input—typically created by adding noise or masking the visual modality. More recently, this paradigm has been extended to multi-modal settings: Audio-Visual Contrastive Decoding (AVCD) [30] applies similar contrastive mechanisms across both audio and visual modalities to reduce hallucinations in multi-modal language models.

However, these existing methods share a fundamental limitation: they are modality-agnostic, lacking awareness of task-specific modality requirements. Existing multimodal CD methods, designed for single-modality scenarios, assume hallucinations primarily stem from visual corruption. While AVCD extends contrastive decoding to multiple modalities, it still applies uniform distortions across all modalities without considering which modality is actually relevant to the given task.This uniform, non-adaptive approach is insufficient for tackling cross-modal interference. A static, modality-agnostic approach cannot dynamically block irrelevant inputs from causing cross-modal corruption.

In this paper, we address this limitation by introducing Modality-Adaptive Decoding (MAD), a novel trainingfree method that overcomes modality-agnostic limitation. MAD is the first approach to explicitly determine the modality requirements of each task and dynamically adapt the contrastive decoding strategy to mitigate cross-modal hallucinations.

Our core innovation lies in self-assessing the importance of modality and employing task aware modality weights. We first prompt the multimodal model to self-assess the modality relevance on the given task, then extract modality weights that guide adaptive fusion of modality-specific contrastive distributions. This task-conditioned fusion enables the model to concentrate on relevant modality while suppressing inappropriate cross-modal interference.

We evaluate MAD on two comprehensive benchmarks for cross-modal hallucinations: The Curse of MultiModalities (CMM) [23] and AVHBench [22]. Experiments show that MAD effectively mitigates cross-modal hallucinations, achieving significant improvements on both benchmarks while maintaining performance on standard evaluation tasks.

Our main contributions are as follows:

- • We propose a training-free, modality-adaptive decoding method that dynamically determines modality requirements for each task and adjusts contrastive decoding accordingly to mitigate cross-modal hallucinations in audiovisual LLMs.
- • We introduce a task-driven modality weighting scheme that extracts explicit modality preferences and enables adaptive fusion of modality-specific contrastive distributions.
- • We demonstrate significant improvements on crossmodal hallucination benchmarks (CMM and AVHBench), showing that explicit modality-aware fusion effectively reduces cross-modal hallucinations without requiring model retraining.

#### 2. Related Works

##### 2.1. Audio-visual Large Language Models.

Recent progress in multimodal foundation models has improved beyond simply augmenting LLMs with visual perception to incorporating multiple modalities such as audio, vision and text [2, 31–36]. A growing line of work extends language models toward audio-visual understanding, enabling richer reasoning grounded in sound, image, and video. Video-LLaMa [1] enables end-to-end video understanding by jointly modeling visual and auditory signals within a LLM framework. Qwen2.5-omni [37] perceives text, images, audio and video while generating realtime text and speech responses. These Audio-Visual LLMs (AV-LLMs) leverage multiple sensory streams to construct deeper representations of real-world events and improve semantic grounding.

##### 2.2. Mitigating hallucinations in MLLMs.

Hallucination remains a major challenge for multimodal large language models (MLLMs), where generated responses diverge from input evidence. Existing approaches

###### Modality-Adaptive Decoding

[Figure 13]

###### Step 1: Modality-Adaptive Weight Extraction

|Video 𝑋𝑣|
|---|

[Figure 14]

Modality query logits

Modalityadaptive weight

[Figure 15]

[Figure 16]

Modality query prompt 𝑋𝑚

[Figure 17]

- Step 1 Flow

- Step 2 Flow

|`|
|---|
| |
| |

[Figure 18]

softmax

|𝑤𝑎𝑣,𝑤𝑣,𝑤𝑎|
|---|

both video audio

|𝑙𝑜𝑔𝑖𝑡 𝑦|𝑋𝑣,𝑋𝑎,𝑋𝑞,𝑋𝑚<br><br>|
|---|

Audio-Visual Large Language Model

|Audio 𝑋𝑎|
|---|

[Figure 19]

[Figure 20]

###### Step 2: ModalityAdaptive Generation

|[Figure 21]<br><br>Original Prediction:<br><br>Yes, the man is …|
|---|

Audio-Visual logits

Yes No

Modality-

𝑙𝑜𝑔𝑖𝑡 𝑦𝑡 𝑋𝑣,𝑋𝑎,𝑋𝑞, 𝑦<𝑡

Adaptive

| | | |
|---|---|---|

Response

Video-only logits

Yes No

|[Figure 22]<br><br>MAD Prediction: No, the video shows…|
|---|

|𝑙𝑜𝑔𝑖𝑡 𝑦𝑡 𝑋𝑣,𝑋`𝑎ത,𝑋𝑞, 𝑋<𝑡<br><br>|
|---|

|𝐸𝑞9().<br><br>|
|---|

|Question 𝑋𝑞|
|---|

𝐸𝑞9().

| | |
|---|---|
| | |
| | |

| | | |
|---|---|---|

Audio-only logits

Yes No

Did you hear hammer

|𝑙𝑜𝑔𝑖𝑡 𝑦𝑡 𝑋𝑣ത,𝑋𝑎,𝑋𝑞, 𝑋<𝑡<br><br>|
|---|

[Figure 23]

Ground Truth: No

Yes No

hitting in the

| | | |
|---|---|---|

Question-only logits

Yes No

|𝑙𝑜𝑔𝑖𝑡𝑀𝐴𝐷|
|---|

audio?

|𝑙𝑜𝑔𝑖𝑡 𝑦𝑡 𝑋𝑣ത,𝑋𝑎ത,𝑋𝑞, 𝑦<𝑡<br><br>|
|---|

- Figure 2. Overall MAD pipeline. Given audio-visual inputs and a question, MAD extracts modality-adaptive weights by querying the model to identify relevant modalities [Step 1]. During generation, MAD fuses contrastive logits computed from four modality configurations using these weights to dynamically emphasize relevant modalities [Step 2]. In this example, despite a hammer being visible in the video, MAD correctly predicts “No” by prioritizing audio content, whereas the baseline predicts “Yes” due to cross-modal interference. This demonstrates MAD’s effectiveness in mitigating video-driven audio hallucinations through adaptive, question-aware decoding.

aim to reduce hallucination through (i) representation-level debiasing [38, 39], (ii) architectural decoding enhancements [40–43], and (iii) training-free inference methods such as contrastive decoding [24–26, 28, 29, 44]. Among these, training-free approaches are particularly compelling as they require no additional optimization or annotations.

Contrastive Decoding [29] generates outputs by contrasting likelihoods between a more plausible and a less plausible condition. Visual Contrastive Decoding (VCD) [24] extends this idea by defining plausibility through visual distortions. CODE [25] leverages self-generated image descriptions as a contrastive reference to suppress hallucinatory tokens. ICD [26] contrasts distributions obtained from standard versus disturbed instructions to remove hallucinated concepts during inference. DoLa [28] contrasts logits from later versus earlier language model layers, amplifying factual knowledge and reducing false predictions.

More recently, AVCD [30] extends contrastive decoding to audio-visual language models by identifying lessdominant modalities through attention and selectively perturbing them. AVCD [30] reformulates Contrastive Decoding [29] into a trimodal setting—audio, vision, and text—allowing hallucination suppression to operate on cross-modal interactions instead of a single corrupted modality. However, this method does not examine how varying levels of influence from each input modalities (e.g. audio and video) affect decoding. Our findings show that adaptively choosing the appropriate contrastive weight for each modality is crucial for further reducing hallucination.

#### 3. Method

##### 3.1. Preliminaries

###### 3.1.1. Multimodal Input and Generation Process

An Audio-Visual Large Language Model (AV-LLM) takes three modalities as input: a video sequence v, an audio waveform a, and a textual question q. Each modality is processed by its corresponding encoder specialized for that input type— such as a visual encoder (e.g., CLIP [45], SigLIP [46]) for video, an audio encoder (e.g., HuBERT [47], Whisper [48]) for audio, and a text tokenizer for language. These modality-specific encoders transform raw inputs into tokenized representations that are compatible with a shared embedding space. We denote a sequence of input tokens of length n as:

X = {x1,x2,...,xn},

where each token xi represents an element in the multimodal input space. Specifically, the tokenized representations for each modality are given by:

Xv = {xv1,...,xvn

}, Xa = {xa1,...,xan

v

}, Xq = {xq1,...,xqn

a

},

q

where nv, na, and nq denote the number of tokens for the video, audio, and text modalities, respectively.

The AV-LLM, parameterized by θ, is denoted as Mθ. At each decoding step t, the model autoregressively predicts

the next-token distribution over the vocabulary V. The output vector yt ∈ R|V| represents the unnormalized logits for each candidate token, from which the probability distribution is obtained through softmax normalization:

yt ∼ pθ(yt | Xv,Xa,Xq,y<t) ∝ logitθ(yt | Xv,Xa,Xq,y<t),

(1)

where y<t = {y1,...,yt−1} denotes all previously generated tokens, and logitθ(·) represents the model’s raw output logits over V before softmax normalization.

###### 3.1.2. Contrastive Decoding for Vision Language Models

Contrastive Decoding (CD) is a training-free inference method that mitigates hallucinations by contrasting the model’s output on a clean input with a output on a deliberately degraded input. This naturally extends to vision–language models, where the degraded input is obtained by corrupting the image (e.g., noise, masking, or removal) to suppress visually induced hallucinations.

The key intuition is that visually grounded tokens exhibit a significant logit gap between the output from the clean input (with original vision v) and the output from the degraded input (with distorted or absent vision v˜). Conversely hallucinated tokens—arising from spurious correlations or language priors—maintain similar logits in both cases. By amplifying this contrast, the visual extension of CD suppresses hallucinations while preserving visually grounded generation.

Generally, given the vision and question input tokens {Xv,Xq}, the visual extension of CD computes the contrastive logit for each candidate token yt as:

logit(yt) = (1 + α) · logit(yt | Xv,Xq,y<t)

(2)

− α · logit(yt | Xv˜,Xq,y<t)

where Xv˜ denotes the vision input tokens with degraded vision v˜ obtained through techniques such as noise injection, augmentation or masking, and α > 0 controls the contrastive strength. A larger α penalizes vision-independent tokens more heavily, while α = 0 reduces to standard greedy decoding. By subtracting the scaled logits from the degraded branch, this process down-weights tokens relying on language priors, thereby mitigating visual hallucinations.

##### 3.2. Modality-Adaptive Decoding (MAD)

In this section, we introduce Modality-Adaptive Decoding (MAD). MAD extends and applies contrastive decoding to self-assess task-modality relevance and enable a task-aware distribution of modality weights.

###### 3.2.1. Weighted Contrastive Decoding

Following the standard formulation of the visual extension of CD (2), we extend the contrastive objective to an arbi-

trary modality m as:

logit(CDm)(yt) = logit(yt | Xm,Xq,y<t) + αm · ∆m, ∆m = logit(yt | Xm,Xq,y<t) − logit(yt | Xm˜ ,Xq,y<t).

(3) Here, ∆m represents the contrastive signal measuring how much the model’s output depends on modality m—larger values indicate stronger modality-specific contributions. The term αm · ∆m determines the strength of hallucination suppression: αm controls how aggressively we penalize tokens that lack grounding in modality m.

However, conventional contrastive decoding applies a fixed αm uniformly across all tasks, regardless of whether modality m is actually relevant to the task. This is problematic: for a question like “What sound is heard?”, we should apply strong contrastive suppression on the audio modality (high αa) while minimizing visual contrastive effects (low αv). Conversely, for “What color is the car?”, the priorities reverse.

In other words, the contrastive strength αm should be proportional to the importance of modality m for the given task. To formalize this principle, we introduce a modalityadaptive weighting scheme:

αm = γ · wm (4)

where γ > 0 is a fixed base contrastive strength shared across all modalities, and wm ∈ [0,1] is a task-specific weight reflecting the relevance of modality m. By using a unified γ, we ensure that differences in contrastive intensity arise purely from the adaptive weights wm, rather than from inherent modality-specific biases. This decomposition allows the contrastive intensity to scale dynamically: when wm is high (modality is relevant), αm increases, amplifying hallucination suppression; when wm is low (modality is irrelevant), αm decreases, reducing unnecessary penalization.

Substituting Eq. (4) into Eq. (3), we obtain the weighted contrastive decoding formulation:

logit(WCDm) (yt) = logit(yt | Xm,Xq,y<t)+γ·wm·∆m (5)

The key challenge now lies in determining wm dynamically for each question, which we address through a model self-assessment mechanism.

###### 3.2.2. Modality-Adaptive Weight Extraction

A key idea of MAD is to let the multimodal model itself determine which modalities are most relevant to a given task. To estimate the modality-specific weights wm in Eq. (5), we leverage the model’s inherent ability to self-assess modality relevance through the model prediction. Given the inputs Xv, Xa, and Xq, we append a fixed modality query prompt Xm, instructing the model: “To answer this question, which modality is needed (audio, video, or both)?”

|Distribution of 𝒘𝒗,𝒘𝒂,𝒘𝒂𝒗|
|---|

Visual Related

0.565 0.106 0.328

Audio Related

0.237

0.482 0.279

AudioVisual

0.297 0.237 0.464

Related

Avg. Visual Weight Avg. Audio Weight Avg. Audio-Visual Weight

- Figure 3. Distribution of extracted modality weights (wv, wa, wav) across question types on video. The weights align with intuitive modality dependencies, confirming that MAD correctly identifies the required modalities without supervision.

The model then autoregressively predicts the next token, from which we extract the logits corresponding to the tokens ‘video’, ‘audio’, and ‘both’. These logits reflect the model’s confidence in each modality configuration:

zav = logit(‘both’ | Xv,Xa,Xq,Xm),

zv = logit(‘video’ | Xv,Xa,Xq,Xm), za = logit(‘audio’ | Xv,Xa,Xq,Xm),

[wav,wv,wa] = softmax([zav,zv,za]).

(6)

The resulting weights (wav,wv,wa) form normalized probabilities that quantify the model’s self-assessed importance of each modality. These weights are subsequently used in Eq. (5) to adaptively scale the contrastive decoding strength of each modality branch.

Analysis of Modality-Adaptive Weight. To verify whether the extracted modality weights wm accurately reflect the modality relevance implied by each task, we randomly sample 100 videos from VideoMME [49] and construct 300 questions categorized into three types: visual-related, audiorelated, and audio-visual-related. For example, questions included ’Is the folded paper white?’ (visual), ’What kind of instrument is being played?’ (audio), and ’Do hands move faster with the music?’ (audio-visual). More examples are in the supplementary material. For each question, we compute the averaged modality weights (wv,wa,wav) obtained from Eq. (5).

As shown in Figure 3, the distribution of weights aligns well with intuitive modality dependencies: visual-related questions yield dominant wv, audio-related questions yield dominant wa, and audio-visual questions exhibit dominant wav and a balanced combination of wv and wa. This observation confirms that the proposed modality query effectively captures modality relevance without any additional supervision, reinforcing the interpretability and reliability of our adaptive weighting scheme.

###### 3.2.3. Modality-Adaptive Generation

Having established the weighted contrastive formulation in Eq. (5) and the weight extraction mechanism, we now derive the complete generation process for audio-video multimodal inputs.

To simplify notation, we define the following shorthand for logit terms based on modality configuration:

1m2··· ≜ logit(yt | Xm

logitm

,...,Xq,y<t) (7)

,Xm

1

2

where each modality in the superscript can be either standard (m) or perturbed (m˜ ). For brevity, we omit the token argument yt when clear from context.

Instantiating Eq. (2) for two modalities, video (v) and audio (a), and distinguishing between joint audio–visual contrastive strength αav and single-modality strengths αv and αa leads to the following four-branch formulation:

logit(yt) = (1 + αav)logitvaq − αav · logitvaq˜

Visual CD | audio present

###### + (1 + αav)logitvaq − αav · logitvaq˜

Audio CD | visual present

(8)

###### + (1 + αv)logitvaq˜ − αv · logitv˜aq˜

Visual CD | audio absent

###### + (1 + αa)logitvaq˜ − αa · logitv˜aq˜

.

Audio CD | visual absent

Each line implements a contrastive operation targeting a specific modality. When both modalities are available (first two lines), we apply a joint audio–visual contrastive decoding controlled by αav. When one modality is absent (last two lines), the update falls back to single-modality contrastive decoding with modality-specific strengths αv and αa. In this way, the formulation aggregates four distinct contrastive signals, each targeting hallucinations arising from a specific modality configuration.

Building on this four-branch formulation, we replace the fixed contrastive strengths αav, αv, and αa with the modality-adaptive terms γ · wav, γ · wv, and γ · wa derived in Eq. (4). This substitution yields our proposed ModalityAdaptive Decoding (MAD):

logitMAD(yt) = (1 + γ · wav)logitvaq − γ · wav · logitvaq˜

Visual CD | audio present

###### + (1 + γ · wav)logitvaq − γ · wav · logitvaq˜

Audio CD | visual present

###### + (1 + γ · wv)logitvaq˜ − γ · wv · logitv˜aq˜

Visual CD | audio absent

###### + (1 + γ · wa)logitvaq˜ − γ · wa · logitv˜aq˜

###### .

Audio CD | visual absent

(9)

By adaptively scaling the contrastive strength for each modality according to question-specific importance, MAD more precisely suppresses hallucinations arising from irrelevant modalities while preserving information from relevant ones. The complete generation procedure is summarized in Algorithm 1.

Algorithm 1 Modality-Adaptive Decoding (MAD) Require: Video input Xv, Audio input Xa, Question input

Xq, Modality query prompt Xm, Audio-Visual LLM with parameters θ, Contrastive strength γ

Ensure: Decoded output sequence y

- 1: Initialize empty output sequence y ← ∅
- 2: Compute modality query logits: logitm ← pθ(y|Xv,Xa,Xq,Xm)
- 3: Extract modality-adaptive weights:

zav ← logitm[‘both’], zv ← logitm[‘video’], za ← logitm[‘audio’]

wav,wv,wa ← softmax([zav,zv,za])

- 4: while EOS token ∈/ y do
- 5: Compute logits for all modality configurations:

logitvaq ← pθ(yt | Xv,Xa,Xq,y<t) logitvaq˜ ← pθ(yt | Xv˜,Xa,Xq,y<t) logitvaq˜ ← pθ(yt | Xv,Xa˜,Xq,y<t) logitv˜aq˜ ← pθ(yt | Xv˜,Xa˜,Xq,y<t)

- 6: Compute MAD logits using Eq. (9):

logitMAD(yt) = (1 + γ · wav)logitvaq − γ · wav · logitvaq˜ + (1 + γ · wav)logitvaq − γ · wav · logitvaq˜

+ (1 + γ · wv)logitvaq˜ − γ · wv · logitv˜aq˜ + (1 + γ · wa)logitvaq˜ − γ · wa · logitv˜aq˜

- 7: Select next token: yˆt ← arg maxyt∈V logitMAD(yt)
- 8: Append yˆt to y
- 9: end while
- 10: return y

#### 4. Experiments

We evaluate the effectiveness of MAD on cross-modal hallucination benchmarks using multiple state-of-the-art audio-visual language models, comparing against existing contrastive decoding methods.

##### 4.1. Experimental Setup

- 4.1.1. Benchmark Datasets.

We evaluate our method on two benchmarks designed to assess cross-modal hallucinations in multimodal large language models. AVHBench [22] is specifically designed to evaluate audio-visual hallucinations, containing video clips paired with questions that require reasoning about either visual or auditory information. CMM [23] provides a comprehensive evaluation of multimodal understanding, particularly examining how models depend on or bias toward certain modalities under diverse audio-visual scenarios. Both benchmarks enable systematic assessment of how models handle modality-specific queries and resist cross-modal interference.

- 4.1.2. Models.

We conduct experiments on three state-of-the-art audiovisual multimodal large language models: VideoLLaMA2AV-7B [50] and Qwen2.5-Omni-7B [37]. These models represent diverse architectural approaches to multimodal understanding and provide a comprehensive testbed for evaluating cross-modal hallucination mitigation methods.

- 4.1.3. Baselines.

We compare our Modality-Adaptive Decoding (MAD) against existing contrastive decoding methods. VCDExtended applies the Visual Contrastive Decoding [24] principle to the multimodal setting by contrasting against all possible modality distortions:

logitVCD-extended(yt) = (1 + 3α) · logitvaq −α · logitvaq˜ − α · logitvaq˜ −α · logitv˜aq˜ .

(10) AVCD [30] extends contrastive decoding to audio-visual settings but applies uniform distortion without queryspecific adaptation. These baselines represent the current state-of-the-art in training-free hallucination mitigation for multimodal models.

- 4.1.4. Implementation Details.

For all experiments, we set the temperature to 0 for deterministic generation across all models. We determined the optimal values by sampling 100 examples per dataset and varying the strength hyperparameter γ from 0.5 to 3.0 in 0.5 intervals. Consequently, we set the strength hyperparameter γ to 2.5 for all datasets. All experiments are conducted using the same computational environment and random seeds to ensure reproducibility.

##### 4.2. Main Results

Table 1 presents the main experimental results comparing MAD with baseline methods across the CMM and AVHBench benchmarks. Across all models and datasets,

CMM AVHBench Visual Dom. Audio Dom. Language Dom. Overall Acc.

Model

Video-Driven Audio Hall.

Audio-Driven Video Hall.

Overall Acc.

VideoLLaMA2-AV 71.8 80.0 68.8 73.5 75.7 79.0 77.4 VideoLLaMA2-AV + VCDExtended 71.3 83.3 74.8 76.4 66.0 74.8 70.4 VideoLLaMA2-AV + AVCD 71.8 84.0 71.5 75.8 78.3 80.3 79.3 VideoLLaMA2-AV + MAD 82.3 84.3 77.5 81.3 79.7 79.1 79.4 Qwen2.5-Omni-7B 64.5 72.3 81.3 72.7 73.0 80.7 76.9 Qwen2.5-Omni-7B + VCDExtended 62.5 71.3 84.5 72.8 70.3 77.1 73.7 Qwen2.5-Omni-7B + AVCD 66.3 72.8 81.0 73.3 75.8 79.7 77.8 Qwen2.5-Omni-7B + MAD 76.8 84.3 83.3 81.4 78.7 84.4 81.6

Table 1. Main results on cross-modal hallucination benchmarks. We compare the original decoding (Base), VCD-Extended (with Eq (10)), AVCD, and our proposed MAD. MAD consistently outperforms all baselines across three audio-visual LLMs and both benchmarks, validating the effectiveness of query-specific modality selection for mitigating cross-modal hallucinations. All results are reported in accuracy (%).

MAD consistently outperforms existing decoding methods, demonstrating its effectiveness in mitigating cross-modal hallucinations.

Results on CMM. CMM evaluates hallucinations stemming from overreliance on unimodal priors, and MAD achieves clear gains across all three dominance categories. Visual dominance (Visual Dom.) occurs when a model over-relies on visual information while failing to properly incorporate linguistic and auditory cues. Audio dominance (Audio Dom.) occurs when a model places excessive emphasis on auditory input while failing to properly integrate visual or linguistic information. Language dominance (Language Dom.) appears when models follow linguistic priors even when they conflict with multimodal evidence.

For VideoLLaMA2-AV [50], MAD improves visual dominance by +9.3% and language dominance by +5.5%, resulting in an overall accuracy of 81.3%. For Qwen2.5Omni [37], the improvements are even larger: visual dominance increases by +12.3% and audio dominance by +12.0%, bringing the overall accuracy to 81.4%. These consistent gains show that MAD effectively mitigates unimodal-prior-induced hallucinations by accurately identifying which modality each task requires.

Results on AVHBench. MAD also delivers substantial improvements on AVHBench. For VideoLLaMA2-AV, videodriven audio hallucination accuracy increases by +4.0%, while Qwen2.5-Omni achieves a +5.7% gain. Audiodriven video hallucination further improves by +3.7% for Qwen2.5-Omni, confirming that MAD reduces both audiodriven video and video-driven audio hallucinations.

Compared to VCD-Extended and AVCD, which apply uniform distortion strategies without query-specific adaptation, MAD’s adaptive weighting of modality-specific contrastive branches enables more precise suppression of dominance-induced hallucinations. The consistent performance improvements across models and benchmarks underscore the importance of explicit modality-aware decoding

Table 2. Comparison of different modality fusion strategies in the ablation study using VideoLLaMA2-AV as the base model. The proposed adaptive weighting (Weighted) achieves the best overall accuracy across CMM benchmark.

CMM Visual Dom. Audio Dom. Language Dom. Overall Acc.

Fusion Method

Baseline 71.8 80.0 68.8 73.5 Uniform 77.5 83.3 77.5 79.4 Argmax 78.5 80.5 77.0 78.7 Weighted 82.3 84.3 77.5 81.3

for addressing cross-modal hallucinations in multimodal large language models.

##### 4.3. Ablation Studies

We conduct ablation studies to analyze the contribution of individual components in MAD.

4.3.1. Comparison of Weighting Strategies. To examine how different weighting strategies interact with the branch structure of Eq. (9), we compare MAD with two simplified variants: (1) a uniform weighting scheme that assigns equal weights to all modalities (wav = wv = wa = 1 3), and (2) an argmax weighting scheme that computes only the branch corresponding to the most relevant modal-

ity, identified by argmaxmwm. In Eq. (9), MAD is decomposed into four contrastive branches depending on the presence or absence of each modality. The argmax variant activates only one of these branches (e.g., Video CD—audio absent when the video modality is dominant), effectively skipping the others. As shown in Table 2, both the uniform and argmax variants lead to clear performance degradation. The uniform weighting overlooks query-specific modality demands, while the argmax strategy eliminates complementary cues from non-selected modalities. In contrast, our adaptive weighting softly integrates all branches in Eq. (9) according to their estimated relevance, achieving a balanced

Table 3. Ablation on modality-specific weights in MAD using VideoLLaMA2-AV as the base model. We disable each weight individually to evaluate its contribution to mitigating cross-modal hallucinations. Using all three weights yields the best overall accuracy.

Adapted Weights CMM wa wv wav Acc↑

Decoding

Visual Dom.

Audio Dom.

Language Dom.

MAD ✓ ✓ 75.8 83.5 74.8 78.0 MAD ✓ ✓ 81.0 81.3 72.8 78.3 MAD ✓ ✓ 80.3 81.8 74.8 78.9 MAD ✓ ✓ ✓ 82.3 84.3 77.5 81.3

fusion that consistently outperforms both baselines across models and benchmarks.

###### 4.3.2. Contribution of Modality-Specific Weights.

Table 3 reports the effect of removing each modalityspecific weight in MAD. We selectively ablate wa, wv, and wav to assess their individual contributions to mitigating cross-modal hallucinations.

Removing the audio weight (wa) leads to a notable drop to 78.0% accuracy, with a 6.5% decrease in Visual Dominance. Without wa, the model fails to capture essential audio cues, which in turn increases its reliance on visual information and ultimately causes it to hallucinate audio events purely from visual signals (similar to the example in Figure 2).

Eliminating the visual weight (wv) yields 78.3% accuracy and a 3.0% decline in Audio Dominance. Here, the model overweights audio input and insufficiently incorporates visual evidence, limiting its ability to suppress audiodriven interference. The symmetric degradation observed in both cases confirms that wa and wv are jointly essential for preventing dominance-induced hallucinations.

Although using both wa + wv reduces the performance drop compared to single-weight ablations, noticeable drops remain in all three dominance. This configuration suppresses visual dominance on audio-related queries and audio dominance on visual-related ones, yet it still struggles on tasks requiring joint audio-visual reasoning. To address this, wav provides a flexible coupling mechanism between the two modalities, enabling effective fusion when joint reasoning is necessary. When all three weights (wa+wv+wav) are used, MAD achieves the highest accuracy of 81.3

Overall, these results highlight a key insight: crossmodal hallucinations arise bidirectionally-visual dominance induces audio hallucinations, and audio dominance induces visual ones. Effective mitigation therefore requires adaptively balancing modalities based on each query’s modality relevance.

Table 4. Comparison of MLLMs on general AVQA benchmarks (OmniBench [51], Worldsense [52], and MUSIC-AVQA [53]). All results are reported in accuracy (%).

Model OmniBench Worldsense Music-AVQA

VideoLLaMA2-AV 36.3 23.3 78.1 VideoLLaMA2-AV + MAD 36.8 25.6 79.1

###### 4.3.3. Performance on General Audio-Visual QA Tasks.

To evaluate whether MAD also benefits standard audio–visual question answering tasks, we additionally test on general AVQA benchmarks that do not explicitly target cross-modal hallucinations. As shown in Table 4, MAD achieves comparable or slightly improved performance across these tasks. We attribute this improvement to MAD’s ability to suppress subtle hallucination behaviors that can mislead the model even in non-hallucinationfocused benchmarks. These results suggest that adaptive modality selection not only mitigates explicit cross-modal hallucinations but also enhances general AVQA performance by guiding the model toward more reliable evidence.

#### 5. Conclusion

In this work, we addressed cross-modal hallucinations in audio-visual large language models by introducing Modality-Adaptive Decoding (MAD), a simple and training-free decoding strategy that aligns contrastive signals with the modality requirements of each question. MAD employs a self-assessment mechanism to evaluate modality relevance and assign modality-specific weights, adaptively modulating multi modalities to suppress interference from irrelevant modalities and mitigate hallucination. Our experiments, demonstrates that MAD effectively suppresses cross-modal hallucinations, suggesting that applying modality-appropriate contrastive decoding contributes to more reliable multimodal reasoning. Although MAD currently extracts modality weights directly from the underlying MLLM, future work will focus on learning a lightweight, parameter-efficient predictor that estimates modality weights more quickly and accurately. Furthermore, we plan to extend the modality-adaptive framework to richer modality combinations beyond audio –video, including thermal–RGB or other multi-sensor settings, broadening the impact of modality-aware decoding in real-world multimodal systems.

#### References

- [1] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In Conference on Empirical Methods in Natural Language Processing, 2023. doi: 10.48550/arXiv.2306.

02858. 1, 2

- [2] Jiaming Han, Kaixiong Gong, Yiyuan Zhang, Jiaqi Wang, Kaipeng Zhang, Dahua Lin, Yu Qiao, Peng Gao, and Xiangyu Yue. Onellm: One framework to align all modalities

with language. In Computer Vision and Pattern Recognition,

2023. doi: 10.1109/CVPR52733.2024.02510. 2

- [3] Yixuan Su, L¨u Tian, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. arXiv (Cornell University), undefined.
- [4] Jiaming Han, Renrui Zhang, Wenqi Shao, Peng Gao, Peng Xu, Han Xiao, Kaipeng Zhang, Chris Liu, Song Wen, Ziyu Guo, Xudong Lu, Shuai Ren, Yafei Wen, Xiaoxin Chen, Xiangyu Yue, Hongsheng Li, and Y. Qiao. Imagebind-llm: Multi-modality instruction tuning. In arXiv.org, 2023. doi: 10.48550/arXiv.2309.03905.
- [5] Zijia Zhao, Longteng Guo, Tongtian Yue, Sihan Chen, Shuai Shao, Xinxin Zhu, Zehuan Yuan, and Jing Liu. Chatbridge: Bridging modalities with large language model as a language catalyst. arXiv (Cornell University), undefined.
- [6] Guangzhi Sun, Wenyi Yu, Changli Tang, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Yuxuan Wang, and Chao Zhang. video-salmonn: Speech-enhanced audio-visual large language models. arXiv preprint arXiv:2406.15704, 2024.
- [7] Muhammad Maaz, H. Rasheed, Salman H. Khan, and F. Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Annual Meeting of the Association for Computational Linguistics, 2023.
- [8] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In Conference on Empirical Methods in Natural Language Processing, 2023.
- [9] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, pages 396–416. Springer, 2024. 1
- [10] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering,

2019. URL https://arxiv.org/abs/1906.02467. 1

- [11] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa:next phase of question-answering to explaining temporal actions, 2021. URL https://arxiv.org/abs/ 2105.08276.
- [12] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding, 2024. URL https:// arxiv.org/abs/2407.15754.
- [13] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: Benchmarking multi-task long video understanding, 2025. URL https://arxiv.org/abs/2406.04264. 1
- [14] Tiantian Geng, Jinrui Zhang, Qingni Wang, Teng Wang, Jinming Duan, and Feng Zheng. Longvale: Vision-audiolanguage-event benchmark towards time-aware omni-modal perception of long videos, 2025. URL https://arxiv. org/abs/2411.19772. 1
- [15] Yudong Yang, Jimin Zhuang, Guangzhi Sun, Changli Tang, Yixuan Li, Peihan Li, Yifan Jiang, Wei Li, Zejun Ma, and

- Chao Zhang. Audio-centric video understanding benchmark without text shortcut, 2025. URL https://arxiv.org/ abs/2503.19951.
- [16] Jing Liu, Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, and Jinhui Tang. Valor: Vision-audiolanguage omni-perception pretraining model and dataset,

2025. URL https://arxiv.org/abs/2304.08345. 1

- [17] A. Gunjal, Jihan Yin, and Erhan Bas. Detecting and preventing hallucinations in large vision language models. In AAAI Conference on Artificial Intelligence, 2023. doi: 10.48550/ arXiv.2308.06394. 1
- [18] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji rong Wen. Evaluating object hallucination in large vision-language models. In Conference on Empirical Methods in Natural Language Processing, 2023. doi: 10. 48550/arXiv.2305.10355.
- [19] Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang, Zhun Deng, Chelsea Finn, Mohit Bansal, and Huaxiu Yao. Analyzing and mitigating object hallucination in large vision-language models. In International Conference on Learning Representations, 2023. doi: 10.48550/arXiv.2310. 00754.
- [20] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385, 2024.
- [21] Holy Lovenia, Wenliang Dai, Samuel Cahyawijaya, Ziwei Ji, and Pascale Fung. Negative object presence evaluation (nope) to measure object hallucination in vision-language models. undefined. 1
- [22] Kim Sung-Bin, Oh Hyun-Bin, JungMok Lee, Arda Senocak, Joon Son Chung, and Tae-Hyun Oh. Avhbench: A crossmodal hallucination benchmark for audio-visual large language models. arXiv preprint arXiv:2410.18325, 2024. 1, 2, 6
- [23] Sicong Leng, Yun Xing, Zesen Cheng, Yang Zhou, Hang Zhang, Xin Li, Deli Zhao, Shijian Lu, Chunyan Miao, and Lidong Bing. The curse of multi-modalities: Evaluating hallucinations of large multimodal models across language, visual, and audio. arXiv preprint arXiv:2410.12787, 2024. 1, 2, 6
- [24] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Li Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13872– 13882, 2023. 2, 3, 6
- [25] Junho Kim, Hyunjun Kim, Yeonju Kim, and Yonghyun Ro. Code: Contrasting self-generated description to combat hallucination in large multi-modal models. In Neural Information Processing Systems, 2024. 3
- [26] Xintong Wang, Jingheng Pan, Liang Ding, and Christian Biemann. Mitigating hallucinations in large vision-language

- models with instruction contrastive decoding. In Annual Meeting of the Association for Computational Linguistics, 2024. 3
- [27] Fushuo Huo, Wenchao Xu, Zhong Zhang, Haozhao Wang, Zhicheng Chen, and Peilin Zhao. Self-introspective decoding: Alleviating hallucinations for large vision-language models, 2025. URL https://arxiv.org/abs/ 2408.02032.
- [28] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models. arXiv preprint arXiv:2309.03883, 2023. 2, 3
- [29] Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori B Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12286–12312, 2023. 2, 3
- [30] Chaeyoung Jung, Youngjoon Jang, and Joon Son Chung. Avcd: Mitigating hallucinations in audio-visual large language models through contrastive decoding. arXiv preprint arXiv:2505.20862, 2025. 2, 3, 6
- [31] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, Hang Yan, Jie Fu, Tao Gui, Tianxiang Sun, Yu-Gang Jiang, and Xipeng Qiu. Anygpt: Unified multimodal llm with discrete sequence modeling, 2025. URL https://arxiv.org/abs/2402.12226. 2
- [32] Qilang Ye, Zitong Yu, Rui Shao, Xinyu Xie, Philip Torr, and Xiaochun Cao. Cat: Enhancing multimodal large language model to answer questions in dynamic audio-visual scenarios, 2024. URL https://arxiv.org/abs/2403. 04640.
- [33] Chenyang Lyu, Minghao Wu, Longyue Wang, Xinting Huang, Bingshuai Liu, Zefeng Du, Shuming Shi, and Zhaopeng Tu. Macaw-llm: Multi-modal language modeling with image, audio, video, and text integration, 2023. URL https://arxiv.org/abs/2306.09093.
- [34] Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen, and Furu Wei. Beats: Audio pretraining with acoustic tokenizers, 2022. URL https: //arxiv.org/abs/2212.09058.
- [35] Artemis Panagopoulou, Le Xue, Ning Yu, Junnan Li, Dongxu Li, Shafiq Joty, Ran Xu, Silvio Savarese, Caiming Xiong, and Juan Carlos Niebles. X-instructblip: A framework for aligning x-modal instruction-aware representations to llms and emergent cross-modal reasoning, 2024. URL https://arxiv.org/abs/2311.18799.
- [36] Zijia Zhao, Longteng Guo, Tongtian Yue, Sihan Chen, Shuai Shao, Xinxin Zhu, Zehuan Yuan, and Jing Liu. Chatbridge: Bridging modalities with large language model as a language catalyst, 2023. URL https://arxiv.org/abs/ 2305.16103. 2
- [37] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025. 2, 6, 7, 1

- [38] Chaoya Jiang, Haiyang Xu, Mengfan Dong, Jiaxing Chen, Wei Ye, Ming Yan, Qinghao Ye, Ji Zhang, Fei Huang, and Shikun Zhang. Hallucination augmented contrastive learning for multimodal large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27036–27046, 2024. 3
- [39] Yi-Fan Zhang, Weichen Yu, Qingsong Wen, Xue Wang, Zhang Zhang, Liang Wang, Rong Jin, and Tieniu Tan. Debiasing multimodal large language models. arXiv preprint arXiv:2403.05262, 2024. 3
- [40] Yun Xing, Yiheng Li, Ivan Laptev, and Shijian Lu. Mitigating object hallucination via concentric causal attention. Advances in neural information processing systems, 37:92012– 92035, 2024. 3
- [41] Chenxi Wang, Xiang Chen, Ningyu Zhang, Bozhong Tian, Haoming Xu, Shumin Deng, and Huajun Chen. Mllm can see? dynamic correction decoding for hallucination mitigation. arXiv preprint arXiv:2410.11779, 2024.
- [42] Le Yang, Ziwei Zheng, Boxu Chen, Zhengyu Zhao, Chenhao Lin, and Chao Shen. Nullu: Mitigating object hallucinations in large vision-language models via halluspace projection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14635–14645, 2025.
- [43] Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospectionallocation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13418– 13427, 2024. 3
- [44] Yeongjae Cho, Keonwoo Kim, Taebaek Hwang, and Sungzoon Cho. Do you keep an eye on what i ask? mitigating multimodal hallucination via attention-guided ensemble decoding. arXiv preprint arXiv:2505.17529, 2025. 3
- [45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. URL https://arxiv.org/abs/2103.00020. 3
- [46] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training,

2023. URL https://arxiv.org/abs/2303.15343. 3

- [47] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units, 2021. URL https://arxiv.org/abs/2106.07447. 3
- [48] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision, 2022. URL https://arxiv.org/abs/2212.04356. 3
- [49] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, En-

- hong Chen, Caifeng Shan, Ran He, and Xing Sun. Videomme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis, 2025. URL https: //arxiv.org/abs/2405.21075. 5, 1
- [50] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 6, 7, 1
- [51] Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Zekun Wang, Jian Yang, Siwei Wu, Xingwei Qu, Jinjie Shi, Xinyue Zhang, Zhenzhu Yang, Xiangzhou Wang, Zhaoxiang Zhang, Zachary Liu, Emmanouil Benetos, Wenhao Huang, and Chenghua Lin. Omnibench: Towards the future of universal omni-language models, 2025. URL https://arxiv. org/abs/2409.15272. 8
- [52] Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms, 2025. URL https://arxiv.org/abs/2502.04326. 8
- [53] Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, JiRong Wen, and Di Hu. Learning to answer questions in dynamic audio-visual scenarios. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19108–19118, 2022. 8

## MAD: Modality-Adaptive Decoding for Mitigating Cross-Modal Hallucinations in Multimodal Large Language Models

### Supplementary Material

#### 6. Impact of γ

In our modality-adaptive weighting mechanism, the temperature parameter γ is a crucial hyperparameter that controls the contrastive strength between modality-specific distributions and the full distribution. To determine the appropriate γ value, we conduct a systematic ablation study by varying γ from 0.5 to 3.0 with intervals of 0.5. For each γ value, we evaluate the performance of our MAD approach on AVHBench [22] and CMM [23] benchmarks using VideoLLaMA2-AV [50] and Qwen2.5-Omni [37] as the base model.

##### 6.1. Results and Analysis

[Figure 24]

- Figure 4. Impacts of γ in VideoLLaMA2-AV

[Figure 25]

- Figure 5. Impacts of γ in Qwen2.5-Omni

Based on this analysis, we use γ = 2.5 as the default value across all experiments. This value demonstrates consistently robust performance across different models and benchmarks.

#### 7. Qualitative Analysis of Modality-Adaptive Weights

In the main paper, we present a quantitative analysis demonstrating that the extracted modality weights wm accurately reflect the modality relevance implied by each task. Here, we provide additional details on the experimental setup and present qualitative examples to further illustrate the behavior of our modality-adaptive weighting mechanism.

- 7.1. Dataset Construction

We randomly sample 100 videos from VideoMME [49] and construct 300 questions categorized into three types based on their modality dependency:

- • Visual-related questions (100 questions): Questions that require only visual information to answer.
- • Audio-related questions (100 questions): Questions that require only audio information to answer.
- • Audio-visual-related questions (100 questions): Questions that require both audio and visual information to answer.

- 7.2. Question Examples

Table 5 presents representative examples of questions from each category along with their corresponding computed modality weights.

- 7.3. Analysis

As shown in Table 5, our modality-adaptive weighting mechanism successfully assigns higher weights to the relevant modality for each question type. For visual-related questions, wv is consistently higher, indicating strong reliance on video information. Conversely, audio-related questions exhibit higher wa values. For audio-visual questions requiring multimodal reasoning, wav receives the highest weight, demonstrating the model’s ability to recognize the need for integrated multimodal understanding.

#### 8. Modality Weight Distribution Analysis Across Tasks

To validate that our modality-adaptive weighting mechanism appropriately adjusts weights according to task characteristics, we analyze the distribution of modality weights across different question types in AVHBench and CMM benchmarks. All experiments are conducted using VideoLLaMA2-AV as the base model.

Table 5. Representative examples of questions and their modality weights.

Category Question wv wa wav Is the singer a black woman? 0.58 0.17 0.25

Visual Does the man with white hair wear glasses? 0.72 0.02 0.26 Does the character’s surroundings get brighter when the character moves? 0.84 0.15 0.01 Can you hear seaguls? 0.22 0.53 0.25

Audio What kind of music is playing in the background? 0.16 0.59 0.25 Is the background music upbeat and fast-paced? 0.09 0.73 0.18 Is there music when she appears on stage? 0.27 0.16 0.56

Audio-Visual Does the narration match the subtitles? 0.22 0.11 0.67 Does the character move when there is a clicking sound? 0.39 0.11 0.50

##### 8.1. Weight Analysis on AVHBench

AVHBench is specifically designed to evaluate cross-modal hallucinations by presenting questions where one modality can mislead the model’s understanding of another modality. Figure 6 shows the distribution of modality weights (wa, wv, wav) across different task categories.

###### Distribution of 𝒘𝒘𝒗𝒗,𝒘𝒘𝒂𝒂,𝒘𝒘𝒂𝒂𝒗𝒗

VideoDriven

0.176 0.615 0.209

Audio Hall.

AudioDriven

0.517 0.119 0.363

Video Hall.

Avg. Visual Weight Avg. Audio Weight Avg. Audio-Visual Weight

Figure 6. Analysis on AVHBench

Video-Driven Audio Hallucination (V→A). In this task category, visual information can mislead audio-related understanding. Despite the potential interference from video, the questions fundamentally require audio comprehension. Our analysis reveals that the model correctly identifies audio as the primary modality, assigning the highest proportion to wa. This demonstrates that MAD successfully prioritizes the relevant modality even in the presence of misleading cross-modal information.

Audio-Driven Video Hallucination (A→V). Conversely, this task involves audio information potentially interfering with video understanding. The weight distribution shows a predominant wv proportion, indicating that the model appropriately recognizes video as the essential modality for answering these questions.

This symmetric behavior across V→A and A→V tasks validates the adaptability of our weighting mechanism.

##### 8.2. Weight Analysis on CMM

The CMM benchmark categorizes cross-modal hallucinations into three types based on the dominant modality that causes interference. Figure 7 presents the modality weight distributions for each category.

###### Distribution of 𝒘𝒘𝒗𝒗,𝒘𝒘𝒂𝒂,𝒘𝒘𝒂𝒂𝒗𝒗

Visual

Dom. (Audio

0.207 0.639 0.155

Hall.)

Audio

Dom. (Video

0.712 0.161 0.127

Hall.)

Language

Dom. (Video

0.042 0.137

0.821

Hall.)

Avg. Visual Weight Avg. Audio Weight Avg. Audio-Visual Weight

Figure 7. Analysis on CMM

Visual Dominance (Visual Dom.). This category addresses cases where visual information is over-relied upon, leading to the neglect of audio information. Our analysis shows that the model assigns higher weight to wa, effectively preventing audio hallucinations by emphasizing the underrepresented audio modality.

Audio Dominance (Audio Dom.). In contrast, this category involves audio information inappropriately influencing video understanding. The weight distribution demonstrates increased wv proportion, which helps suppress video hallucinations caused by audio interference. This behavior mirrors the Visual Dom. pattern but in the opposite direction, further confirming the mechanism’s adaptability.

Language Dominance (Language Dom.). This category predominantly contains video-related questions that are susceptible to language bias, such as ”Did you see the shape

of the wheel is circular in the video?” These questions can be answered through linguistic priors without genuine visual understanding. The model compensates for this by assigning substantially higher weight to wv, forcing greater reliance on actual visual evidence and thereby mitigating language-based shortcuts.

##### 8.3. Discussion

The consistent patterns observed across both benchmarks demonstrate that our modality-adaptive weighting mechanism successfully captures task-specific modality requirements. The model autonomously adjusts weights to emphasize the most relevant modality while suppressing potentially misleading cross-modal influences, validating the effectiveness of our approach in addressing cross-modal hallucinations.

#### 9. Robustness Analysis of Modality Query Prompts

To evaluate the robustness of our modality-adaptive weighting mechanism, we conduct ablation studies with various modality query prompts. While our main paper uses the prompt Xm = “To answer this question, which modality is needed (audio, video, or both”, we investigate whether alternative prompt formulations affect the model’s ability to generate appropriate modality weights.

##### 9.1. Alternative Prompt Formulations

We design the following alternative modality query prompts that maintain similar semantic intent but differ in phrasing:

- • Xm′

1

: “Identify which modality is required to answer the question (audio, video, or both)”

- • Xm′

2

: “Given this question, select the necessary modality for reasoning (audio, video, or both)”

- • Xm′

3

: “Which modality does this question require (audio, video, or both)”

- • Xm′

: “State the modality relevant for answering this question (audio, video, both)”

4

##### 9.2. Experimental Results

Figure 8 presents the performance statistics across different modality query prompts on both AVHBench and CMM benchmarks. The results demonstrate remarkable consistency across prompt variations, with standard deviations of only 0.26% on AVHBench and 0.31% on CMM. The narrow range between minimum and maximum performance values (0.59% and 0.83% respectively, indicated by the vertical bars) further confirms that our approach is highly robust to variations in prompt formulation.

This robustness suggests that the model’s modality weight generation relies on semantic understanding of the modality dependency concept rather than specific prompt

[Figure 26]

Figure 8. Performance consistency across different modality query prompts on AVHBench and CMM benchmarks using VideoLLaMA2-AV. Circles denote mean ± standard deviation; vertical bars show minimum and maximum accuracy across all five prompt variants.

engineering or surface-level pattern matching. The consistent performance across diverse phrasings validates that MAD captures a fundamental principle of modality-task alignment, making it a principled and reliable approach for addressing cross-modal hallucinations.

#### 10. Computational Analysis

Our MAD method extracts modality-adaptive weights through self-assessment prompts and applies weighted contrastive decoding during inference. To analyze the computational overhead introduced by our approach, we compare the decoding latency (ms/token)↓ with other contrastive decoding-based methods. We conduct all experiments on 8 NVIDIA RTX A6000 GPUs using VideoLLaMA2-AV and Qwen2.5-Omni.

Table 6. Computational efficiency comparison among CD-based methods.

Method VideoLLaMA2-AV Qwen2.5-Omni

VCDextend 3564.46 4431.78 AVCD 4811.13 9489.81 MAD (Ours) 3571.64 6700.56

#### 11. Qualitative Results

We provide quantitative results demonstrating the effectiveness of MAD in Fig 9 and Fig 10. Here, we provide qualitative examples showing how our method mitigates crossmodal hallucinations.

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Video: A situation where a metal duck target plate is being shot with a gun

Sound: The sound of a bullet hitting a metal duck

Question: Please describe what you can hear and see in detail.

*Vision-driven audio hallucination

Base

The video shows a metal box with four metal ducks on it, sitting on the ground. The main source of sound in the video is the quacking of the ducks. Suddenly, a gun is shot, and the ducks start to move, creating a loud quacking sound.

AVCD

The video shows a metal box with four duck cutouts on it. The main source of sound in the video is the quacking of ducks. The person in the video is seen holding a metal box with a duck cutout on it.

MAD

The video shows a metal box with four duck shapes on it, sitting on the ground or a concrete surface. The main source of sound in the video is the sound of shooting, which is heard throughout the video. It appears to be a duck-shaped target or practice aid for shooting

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Video: A situation where a dog is running toward the camera

Sound: The sound of nearby sheep bleating

Question: Did you see any sheep?

*Audio-driven visual hallucination

Base

Yes

AVCD

Yes

MAD

I did not see any sheep in the video.

Figure 9. Qualitative Results in VideoLLaMA2-AV

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Video: A situation where a gun is being loaded at a shooting range

Sound: The sound of a gun being shot by others around

Question: Did he pull the trigger? Please describe what you can hear and see in detail.

Base

- *Vision-driven audio hallucination
- *Audio-driven visual hallucination

Well, from what I can see, there's a guy in a camouflage shirt. He's got a gun in his hand. He seems to be at some kind of shooting range, there's a wooden table in front of him with some stuff on it, like a bag and a couple of other things. He's holding the gun and then he pulls the trigger. You can hear the sound of the gun firing. It's a pretty clear shot. So, yeah, he did pull the trigger. What do you think about it?

AVCD

Well, from what I can tell, it seems like he did pull the trigger. I can hear a distinct click sound, which is usually what you'd hear when a trigger is pulled. And then there's a loud bang, which is likely the sound of the gun firing.

MAD

Well, it seems like the man in the video is preparing for some kind of shooting activity. He's handling a gun and there are other items on the table like a knife and a bottle. He's also wearing a green shirt. But, I'm not sure if he actually pulled the trigger or not. What do you think?

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Video: A situation where a motorcycle with its engine turned off is being filmed

Sound: The sound of mowing grass around

Question: Please describe what you can hear and see in detail.

Base

Well, I can see a motorcycle. It's got a sleek black and white design. The engine looks really powerful, and the wheels are big and sturdy. There's a license plate on the back. As for what I can hear, there's a sound of a motorcycle engine revving up. It's a bit loud, you know? It gives off that feeling of power and speed. What do you think about motorcycles? Do you have a favorite?

AVCD

Well, I can see a motorcycle. It's got a black seat and a shiny engine. There are some stickers on it too. And I can hear a motorcycle engine starting up. It's a bit of a rough start, you know? It's like the engine is trying to get going. What do you think about motorcycles? Do you ride one?

MAD

Well it seems like there's a motorcycle in the video. The front part of the motorcycle is visible and it has a licens e plate. But hmm I can't really tell what's happening with the motorcycle or what the sound is about. Maybe you could give me a bit more information?

Figure 10. Qualitative Results in Qwen2.5-Omni

