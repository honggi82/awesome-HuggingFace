arXiv:2506.13642v2[cs.AI]22Jun2025

# Stream-Omni: Simultaneous Multimodal Interactions with Large Language-Vision-Speech Model

### Shaolei Zhang1,3, Shoutao Guo1,3, Qingkai Fang1,3, Yan Zhou1,3, Yang Feng1,2,3∗

1Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences (ICT/CAS)

2Key Laboratory of AI Safety, Chinese Academy of Sciences 3University of Chinese Academy of Sciences, Beijing, China

zhangshaolei20z@ict.ac.cn, fengyang@ict.ac.cn

## Abstract

The emergence of GPT-4o-like large multimodal models (LMMs) has raised the exploration of integrating text, vision, and speech modalities to support more flexible multimodal interaction. Existing LMMs typically concatenate representation of modalities along the sequence dimension and feed them into a large language model (LLM) backbone. While sequence-dimension concatenation is straightforward for modality integration, it often relies heavily on large-scale data to learn modality alignments. In this paper, we aim to model the relationships between modalities more purposefully, thereby achieving more efficient and flexible modality alignments. To this end, we propose Stream-Omni, a large languagevision-speech model with efficient modality alignments, which can simultaneously support interactions under various modality combinations. Stream-Omni employs LLM as the backbone and aligns the vision and speech to the text based on their relationships. For vision that is semantically complementary to text, Stream-Omni uses sequence-dimension concatenation to achieve vision-text alignment. For speech that is semantically consistent with text, Stream-Omni introduces a CTCbased layer-dimension mapping to achieve speech-text alignment. In this way, Stream-Omni can achieve modality alignments with less data (especially speech), enabling the transfer of text capabilities to other modalities. Experiments on various benchmarks demonstrate that Stream-Omni achieves strong performance on visual understanding, speech interaction, and vision-grounded speech interaction tasks. Owing to the layer-dimensional mapping, Stream-Omni can simultaneously provide intermediate text outputs (such as ASR transcriptions and model responses) during speech interaction, offering users a comprehensive multimodal experience1.

## 1 Introduction

Large multimodal models (LMMs) such as GPT-4o [1] exhibit omni-capabilities across text, vision, and speech modalities, unlocking broad potential across applications. Compared to vision-oriented LMMs [2, 3], omni-modal LMMs can support speech interaction based on visual information. Furthermore, advanced online services like GPT-4o can offer a seamless “see-while-hear” interaction for users by simultaneously providing intermediate text (i.e., transcription of user inputs and model responses) during speech interaction, which highlights the importance of building LMMs that can simultaneously support interactions through various modality combinations.

However, building LMMs that support text, vision, and speech remains a substantial challenge due to the intrinsic representational discrepancies across modalities. Most existing LMMs specialize in

∗Corresponding author: Yang Feng. 1Code: https://github.com/ictnlp/Stream-Omni, Model: https://huggingface.co/ICTNLP/stream-omni-8b.

Preprint. Under review.

Speech Out

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

| | |
|---|---|
|Speech|Decoder|
|Out| |

Text

|Large Language Model|
|---|

| | |
|---|---|
|Vision E + P|ncoder roj|
| | |

| | |
|---|---|
|Speech<br><br>+ P|Encoder roj|
| | |

| | |
|---|---|
|Embe|dding|
| | |

[Figure 5]

|How is the whether?|
|---|

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Vision In Speech In

Text In

sequence-dimension concatenation

(a) Sequence-dimension concatenation for modality alignments in previous works

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Speech Out

Top Speech

|Layers| |
|---|---|
|Transformer Layer 40| |

Simultaneously produce intermediate text results

…

|Transformer Layer 36|
|---|

Text Out

Response

Large Language Model

|It’s sunny today…|
|---|

Transformer Layer 35

layer-dimension mapping

…

[Figure 14]

Optional

for speech-text

|Transformer Layer 4|
|---|

CTC Text Out

Streaming ASR

Vision In

Transformer Layer 3

|How is the whether?|
|---|

…

|How is the whether?|
|---|

|Transformer Layer 1|
|---|

Bottom Speech Layers

Text In

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Speech In

(autoregressive speech units)

sequence-dimension concatenation

for vision-text

(b) Sequence-dimension concatenation for vision-text alignment, and layer-dimension mapping for speech-text alignment

Figure 1: Comparison of modality alignments in Stream-Omni and previous works.

either vision [4, 3, 5–7] or speech [8–11], feeding the extracted modality representations into the context of large language model (LLM) backbone. Recently, some omni-modal LMMs [12–14] aim to integrate text, vision, and speech within a unified framework. Such models typically concatenate representations from individual modality encoders along the sequence dimension before feeding them into the LLM backbone, as shown in Figure 1(a). These concatenation-based approaches simplify modality integration, but they heavily rely on large-scale data to learn modality alignments in a datadriven manner [10, 11, 13, 14], which is not friendly to limited public tri-modal data. Moreover, such concatenation-dimension alignments are not flexible enough to simultaneously produce intermediate text results during speech interactions, as GPT-4o does.

To this end, we aim to model the relationships between modalities more purposefully, thereby achieving more efficient and flexible modality alignments. In multimodal interaction, text, vision, and speech modalities serve different roles, where vision primarily conveys visual information [3], while text and speech focus on language information [9]. As such, directly concatenating all three modalities in sequence-dimension is suboptimal for modality alignments. Ideally, the speech and text should exhibit high semantic consistency, while the vision is semantically complementary to the text. Therefore, vision and speech should be separately aligned to text in different ways.

Along with this idea, we introduce Stream-Omni, a language-vision-speech LMM based on efficient text-centric modality alignments, which can flexibly support interactions under various modality combinations. As shown in Figure 1(b), Stream-Omni is built upon the LLM backbone and aligns the vision and speech modalities to text using different mechanisms. For vision, which is semantically complementary to text, Stream-Omni employs sequence-dimension concatenation for vision-text alignment. For speech, which shares higher semantic consistency with text, Stream-Omni introduces a layer-dimension speech-text mapping for speech-text alignment. Specifically, Stream-Omni takes LLM as the core and introduces bottom and top speech layers to model speech-to-text mapping via Connectionist Temporal Classification (CTC) [15], thereby enabling external interaction through the speech modality and simultaneous internal generation via the text modality. With speech–text mapping, Stream-Omni can transfer the text capability of LLM backbone to the speech modality with less speech data. As a byproduct, Stream-Omni can simultaneously produce intermediate text results (i.e., transcription of instruction and response) during speech interaction, offering a more comprehensive multimodal experience. We evaluate Stream-Omni on various benchmarks covering visual understanding, speech interaction, and vision-grounded speech interaction, and the results demonstrate that Stream-Omni achieves strong performance using only 23,000 hours of speech data.

## 2 Related Work

Existing large multimodal models can be categorized into three types: vision-oriented, speechoriented, and omni-modal. For vision-oriented LMMs, LLaVA [3] is the most widely adopted architecture. In LLaVA, a vision encoder (CLIP [16]) is used to extract visual features from visual inputs, which are then concatenated with the text inputs and fed into LLM to generate text responses.

[Figure 19]

[Figure 20]

[Figure 21]

Interactions under various modality combinations

Speech Out

Stream-Omni

Speech Decoder

###### (a) Text + Vision (Optional) → Text

…

Vision In

<39> <985>

<2542>

<865> <13>

Top Speech Layers

Text In Large Language Model Text Out

[Figure 22]

Text Representation

###### Alignment-based Fusion

Vision Representation

###### (b) Speech + Vision (Optional) → Speech (Text) (simultaneously produce intermediate ASR/output text)

…

Speech Representation

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

…

|It’s sunny today…|
|---|

[Figure 23]

Vision In

It’s sunny today

Text Out

Bottom Speech Layers

Large Language Model

Top Speech Layers

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Large Language Model

[Figure 24]

Speech In Speech Out

…

…

ASR Results Text Out

<blank><blank> How <blank> is <blank> the weather <blank> ?

It’s sunny

<blank><blank>

autoregressive

|<blank> <blank> How <blank> is <blank> the weather <blank>?|
|---|

Projection

| | |
|---|---|
| | |

[Figure 25]

Vision Encoder

###### (c) Text + Vision (Optional) → Speech (Text) (simultaneously produce intermediate output text)

CTC Decoder

ASR Results

| | |
|---|---|
| | |

…

[Figure 26]

Vision In

Vision In Text In

Bottom Speech Layers

[Figure 27]

…

Bottom Speech Layers

Large Language Model

Top Speech Layers

…

| | |
|---|---|
| | |

Speech Out (only output)

Speech Out

<2> <331> <42> <4023> <532> <72> <72> <965> <323> <3424> <39> <985>

<2542> <865>

Text Embdding

Speech Tokenizer

|How is the weather?|
|---|

Text Out

Text In

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

autoregressive

Speech In

Figure 2: Architecture of Stream-Omni. Right: Interactions under various modality combinations.

Based on LLaVA, the following works improve the vision-oriented LMMs through improved training data [5, 17, 18], enhanced image encoding [19, 4, 20], and extended video understanding [21–25].

For speech-oriented LMMs, existing methods rely on either continuous or discrete speech units. Methods based on continuous representations, such as Mini-Omni [8], LLaMA-Omni [9], FreezeOmni [26], SALMONN-Omni [27], and SLAM-Omni [28] use a speech encoder (e.g., Whisper [29]) to extract speech features, which are then projected into the LLM’s embedding space to facilitate speech understanding. These approaches often incorporate a speech decoder to generate speech responses based on LLM’s text outputs. Methods based on discrete units, such as SpeechGPT [30], Moshi [10] and GLM-4-Voice [11], employ a speech tokenizer [31–33] to convert speech into discrete units, allowing the LLM to directly understand and generate speech units, which are finally synthesized into speech using a unit-based speech decoder [34, 33]. Compared to continuous representations, discrete units can be jointly modeled with text in LLM’s context, but they often rely on more speech data for speech pre-training [35, 11].

Existing omni-modal LMMs, such as VITA-1.5 [12], MiniCPM2.6-o [7], Baichuan-Omni [13], Qwen2.5-Omni [14], Megrez-Omni [36], M2-Omni [37], Capybara-Omni [38], EMOVA [39], OpenOmni [40] and Omni-Emotion[41], use various encoders to extract the modality representations, which are then concatenated and fed into the LLM to facilitate multimodal understanding, and finally a speech decoder is employed to synthesize speech from the generated text. Overall, most existing omni-modal LLMs adopt concatenation-based architectures and rely primarily on data-driven approaches to model modality alignments.

In this work, we focus on the efficiency and flexibility of modality alignments, with the goal of leveraging limited tri-modal data to develop a large language-vision-speech model that can simultaneously support multimodal interactions under various modality combinations. Unlike previous approaches that rely solely on sequence-dimension concatenation for modality alignment, Stream-Omni adopts a more deliberate design by employing sequence-dimension concatenation for vision-text alignment and layer-dimension mapping for speech-text alignment, thereby achieving efficient and flexible modality alignments. As an additional advantage, the proposed layer-dimension mapping enables Stream-Omni to produce intermediate text results during speech interaction, thereby offering users a more comprehensive multimodal experience.

## 3 Stream-Omni

We introduce Stream-Omni, a language–vision–speech LMM based on text-centric modality alignments. Stream-Omni aligns vision and speech to the text modality via sequence-dimension concatenation and layer-dimension mapping, respectively, thereby achieving efficient and flexible modality alignments. The architecture, training, and inference of Stream-Omni are introduced as follows.

### 3.1 Architecture

The architecture of Stream-Omni is illustrated in Figure 2. Stream-Omni adopts the LLM as its backbone and progressively aligns the vision and speech to the text, efficiently developing a LMM that supports text, vision, and speech. For vision-text alignment, Stream-Omni applies a vision encoder and projection to extract visual representations, which are then concatenated with the text tokens. For speech-text alignment, Stream-Omni introduces several speech layers at the bottom and top of LLM backbone to respectively map speech to the text and generate speech based on the text.

### 3.1.1 Vision Modality

Given the semantic complementarity between the vision and text modalities, Stream-Omni adopts a sequence-dimension concatenation for vision-text alignment, which is commonly employed in vision-oriented LMMs [3, 5, 42]. Specifically, Stream-Omni introduces the vision encoder and projection to convert visual inputs into visual representations, which are then concatenated with text representations and jointly fed into the LLM to facilitate visual understanding.

### 3.1.2 Speech Modality

Compared to vision, aligning speech and text is more challenging due to the greater variability of speech representations and the relative scarcity of speech data. To address this, Stream-Omni leverages the higher semantic consistency between speech and text, employing a speech-text mapping to facilitate alignment through more direct supervision.

To achieve this, Stream-Omni incorporates an N-layer LLM backbone as the inner core, with Nspeechbottom speech layers added to the bottom for speech-to-text mapping and Nspeechtop speech layers added to the top for text-to-speech mapping. Overall, Stream-Omni extends an N-layer LLM into a (Nspeechbottom + N + Nspeechtop )-layer decoder-only architecture, and leverages multi-task learning to separate different layers into different functions of speech-to-text mapping, text-to-text generation, and text-to-speech mapping. During inference, Stream-Omni autoregressively generates speech at the outermost layer, while relying on the LLM backbone at the inner layers for response generation. In this way, Stream-Omni preserves the generative capabilities and knowledge within the LLM core, while effectively broadening its interaction modalities, avoiding the high cost of using large-scale speech data to relearn textual knowledge. The speech interaction process in Stream-Omni includes speech tokenizer, speech-text mapping, text generation, and streaming speech generation.

Speech Tokenizer To enable the mapping with text token, Stream-Omni employs the pre-trained CosyVoice speech tokenizer [33] to discretize the raw speech S into a sequence of discrete speech units U = (u1,··· ,u|U|):

U = SpeechTokenizer(S), (1)

where SpeechTokenizer(·) denotes speech tokenizer, with the speech units vocabulary VU. To joint modeling speech and text, we extend the vocabulary by merging the speech unit vocabulary VU with the LLM’s text vocabulary VT, and introduce a special blank token ⟨blank⟩, yielding the multimodal vocabulary of Stream-Omni Vomni = VT ∪ VU ∪ {⟨blank⟩}.

Speech-Text Mapping To take advantage of LLM’s capabilities, Stream-Omni introduces the bottom and top speech layers to learn the speech-text mapping, thereby transferring the text capabilities within LLM to the speech modality. Specifically, the bottom and top speech layers consist of Nspeechbottom and Nspeechtop Transformer layers, which share the same configuration as the LLM backbone. The bottom speech layers Fspeechbottom(·) maps the speech units U to the text:

HU = Fspeechbottom(U), (2)

where HU denotes the representation of the speech units. Then, to achieve speech-to-text mapping, Stream-Omni introduces a Connectionist Temporal Classification (CTC) [15] decoder CTCDec(·) to decode the text sequence from HU:

#### DU = CTCDec(HU), (3)

omni| represents the probability distribution over the multimodal vocabulary for each speech unit, which can be decoded into a CTC sequence that includes repeated and blank tokens. During training, this module is optimized using the CTC loss:

where DU ∈ R|U|×|V

p(Z | DU), (4)

LCTC = −log

Z∈Π−1(X)

where Π−1(X) denotes the set of all possible CTC sequences that map to the text sequence X by removing repeated and blank tokens, and p(Z | DU) is the decoding probability of sequence Z from DU. At inference time, Stream-Omni can decode the CTC sequence from DU to produce streaming speech recognition results as an intermediate output for user. More potentially, the CTC decoder holds promise for real-time speech interaction by detecting when the user has stopped speaking based on the consecutive blank tokens in the CTC sequence [43].

Text Generation Through CTC modeling, the bottom speech layers map the speech units into the text representation, achieving speech-text alignment at the representational level. To further bridge the structural gap between speech and text, Stream-Omni removes blank tokens ⟨blank⟩ from HU to produce the refined sequence HˆU. To preserve the model’s understanding of the speech inputs, this blank token removal is only performed during the generation phase (i.e., generated speech).

The processed speech representation HˆU is then concatenated with the visual representation HV (if has visual inputs) and fed into the LLM backbone Fllm(·) to generate the text representation HT:

HT = Fllm([HV : HˆU]), (5)

where [· : ·] is sequence concatenation. Owing to the semantic alignment via CTC modeling, StreamOmni can transfer text intelligence to the speech modality while preserving the text capabilities.

Streaming Speech Generation While autoregressively generating the text outputs, Stream-Omni uses top speech layers to generate the corresponding speech units in a streaming manner. To ensure consistency between the generated speech and text, we introduce an alignment-based fusion to use text information to guide speech unit generation.

As illustrated in Figure 3, the top speech layers take the speech representations HU from bottom speech layers and text representations HT from the LLM backbone as the inputs, where each layer comprises self-attention, alignmentbased fusion, and FFN. The alignment-based fusion module fuses the text representations HT into the speech representations HU, thereby achieving text-to-speech mapping. However, to enable streaming generation, the key challenge lies in accurately identifying which text corresponds to each speech unit, thereby generating the speech units once the related text token is generated.

Mask Attention

× Alignment (CTC)

[Figure 33]

FFN

Text (key)

Alignment-based Fusion

| | |
|---|---|
| | |
| | |
| | |
| | |

Speech(query)

Self-attention

Speech:

Text:

Cross-attention

Figure 3: Diagram of top speech layers.

Fortunately, the CTC decoder introduced in Stream-Omni can naturally capture the positional alignment between speech and text [43], which can be used to guide the alignment-based fusion. Formally, based on the CTC sequence DU, Stream-Omni computes the number of aligned text tokens (excluding duplicate and blank tokens) corresponding to the speech sequence up to unit ui, denoted as Ni. That is, within the first i speech units U≤i, Stream-Omni identifies the first Ni text tokens X≤N

.

i

Accordingly, when autoregressively generating the next speech unit ui+1, Stream-Omni should use the next text token xN

i+1 to guide the generation of speech unit ui+1. In practice, to involve richer text context, Stream-Omni extends the fusion window from the aligned text token xN

i+1 to its preceding W − 1 tokens, where W is the hyperparameter of window size. The alignment-based fusion is implemented via cross-attention [40], with the speech representations attending to the text representations, so the fused representation hfusioni of speech unit ui is:

hfusioni = CrossAttn ui, HNT

i+2−W : Ni+1 , (6) where HNT

i+2−W:Ni+1 are W text representations within the local window (W =5 in Stream-Omni). To reduce generation latency, similar to the widely used wait-k policy in simultaneous translation [44–48], Stream-Omni begins streaming speech generation after lagging K text tokens (K =3 in

Table 1: Training stages and data of Stream-Omni.

Stages Training Tasks Trainable Modules Datasets

- Stage1:

Vision-Text

Vision+Text→Text

Projection LLM Backbone

LLaVA LLaVA-OV LLaVA-zh

- Stage2:

Speech-Text

ASR (CTC Loss in Eq.(6)) Speech→Speech

Bottom Speech Layers Top Speech Layers

LibriSpeech (960h) WenetSpeech (1240h) UltraChattts (6500h) Wikitts (4000h) LLaVAtts (8700h) LLaVA-zhtts (1200h)

- Stage3:

Vision+Text→Text Vision+Speech→Text Vision+Speech→Speech

LLaVAtts (8700h) LLaVA-zhtts (1200h)

LLM Backbone

Text-Vision-Speech

Stream-Omni). Therefore, the first speech unit will be generated immediately after K text tokens have been produced. Using the top speech layers Fspeechtop (·), Stream-Omni can simultaneously generate both text and the corresponding speech units:

Uˆ = Fspeechtop HU,HT , (7)

where Uˆ denotes the generated speech unit sequence. Finally, a CosyVoice speech decoder [33] is used to synthesize the speech waveform from the generated speech units.

### 3.2 Training

Stream-Omni achieves efficient alignment across text, visual, and speech modalities, thus requiring only a small amount of tri-modal training data. Given the scarcity of existing datasets that jointly incorporate all three modalities, we first construct a tri-modal corpus consisting of text, images, and speech through an automated pipeline. Then, Stream-Omni adopts a three-stage training strategy to progressively align the text, visual, and speech modalities.

### 3.2.1 Data Construction

The training of Stream-Omni involves text-vision, text-speech, and text-vision-speech multimodal datasets to support interactions across various modality combinations. For text-vision data, we adopt the LLaVA [3] and the LLaVA-OV dataset [6], while filtering out samples involving maths, code, and other content unsuitable for speech interaction. For text-speech data, we use automatic speech recognition (ASR) corpora from LibriSpeech [49] and WenetSpeech [50] to train bottom speech layers. Given the scarcity of public speech interaction data, we construct speech interaction dataset by converting existing text-only and vision-language instruction datasets into speech interactions datasets using open-source text-to-speech synthesis (TTS) [33], named InstructOmni2. The construction details are introduced in Appendix A. Table 1 summarizes the used training data (only 23K hours of speech), where those marked with superscript ‘tts’ indicate synthesized speech interaction dataset.

### 3.2.2 3-Stage Training

Stream-Omni is initialized using a LLM and adopts a three-stage training strategy, which aligns vision and speech with the text in succession, and then models alignments across three modalities.

- Stage 1: Vision-Text Alignment In this stage, Stream-Omni uses the standard training method used in vision-oriented LMMs such as LLaVA [3].
- Stage 2: Speech-Text Alignment In this stage, the speech-text alignment is achieved by training the bottom and top speech layers using a combination of CTC loss after the bottom speech layers (refer to Eq.( 4)) and cross-entropy loss after the top speech layers. Note that, the text representations

2https://huggingface.co/datasets/ICTNLP/InstructOmni

Algorithm 1 Inference of Stream-Omni

Input: Speech input S, Vision input V , Fusion window size W, Lagging text tokens K Output: Generated speech output S Init: ASR results (CTC sequence) A = [ ]; Generated text tokens Y = [ ]; Generated speech units U = [ ]

- 1: Extract visual representation HV from V using the vision encoder and projection;
- 2: Extract speech units U from S using the speech tokenizer;
- 3: HU ← Fspeechbottom(U); ▷ simultaneously produce ASR results of speech inputs
- 4: while Y [−1] ̸= ⟨eos⟩ do
- 5: y ← Fllm([HV : HˆU : Y ])
- 6: Y .append(y); ▷ simultaneously produce text outputs
- 7: if | Y | < K then continue; ▷ lagging K text tokens
- 8: // Generate speech units corresponding to y until the text token is recognized in the generated speech
- 9: while A[−1] == ⟨blank⟩ or A[−1] == A[−2] do ▷ generate speech for text y
- 10: Generate speech unit u based on HU and Y [−W :] based on Eq.(6);
- 11: U.append(u);
- 12: a ← argmax CTCDec(Fspeechbottom(U)) ; ▷ recognize text from generated speech
- 13: A.append(a);
- 14: Synthesize speech s from U using the speech decoder;
- 15: S.append(s);
- 16: return S

fed into the top speech layers during training (i.e., HT in Eq.(6)) are drawn from ground-truth transcriptions rather than LLM generated text, which aim to avoid text-speech dismatching [40] caused by generating incorrect text, thereby enhancing the consistency of text-to-speech generation.

- Stage 3: Text-Vision-Speech Alignment Finally, we train the LLM backbone of Stream-Omni using constructed tri-modal data through multi-task learning. Specifically, we formulate multiple tasks by combining different modalities, including Vision+Text→Text, Vision+Speech→Text, and Vision+Speech→Speech, which are all optimized using the cross-entropy loss. In this way, StreamOmni is able to flexibly support interactions under various modality combinations.

- 3.3 Inference

Algorithm 1 gives the inference process of Stream-Omni when performing vision-grounded speech interaction. Given vision input V and speech input S, Stream-Omni generates the text token y in an autoregressive manner, and simultaneously synthesizes the corresponding speech of y. During speech synthesis, Stream-Omni autoregressively generates speech units u based on y, until the entire speech corresponding to y is generated. To determine whether the generated speech units for y are complete, Stream-Omni leverages alignment in the CTC decoder (in Eq.(3)). If the CTC decoder identifies a new text token from the generated u (i.e., the semantics of the generated speech are complete), the model proceeds to generate the next text token. Otherwise, the model continues to generate speech units for the current y. Stream-Omni repeats the above process until ⟨eos⟩ is generated.

Besides vision-grounded speech interaction, Stream-Omni also supports interaction of various modality combinations. As shown in Figure 2(right), by flexibly integrating the vision encoder, bottom speech layers, LLM, and top speech layers, Stream-Omni can support various multimodal scenarios.

- 4 Experiments

- 4.1 Benchmarks

We evaluate the multimodal capabilities of Stream-Omni across vision and speech benchmarks. For vision evaluation, we conduct experiments on 11 benchmarks used by LLaVA, including VQA-v2 (VQAv2) [51], GQA [52], VizWiz [53], ScienceQA-IMG (SciQA) [54], TextVQA (VQAT) [55], POPE [56], MME [57], MMBench (MMB) [58], SEED-Bench (SEED) [59], LLaVA-Bench-in-theWild (LLaVAW) [60], and MM-Vet [61]. All evaluations follow LLaVA [3] to ensure comparability. For speech evaluation, we assess the model’s knowledge-grounded speech interaction on spoken

Table 2: Results on visual understanding benchmarks.

Vis Wiz

Sci QA

MMVet

Avg. (%)

Methods LLM VQAv2 GQA

VQAT POPE MME MMB SEED LLaVAW

BLIP-2 Vicuna-13B 65.0 41.0 19.6 61.0 42.5 85.3 1293.8 – 46.4 38.1 22.4 – InstructBLIP Vicuna-7B – 49.2 34.5 60.5 50.1 – – 36.0 53.4 60.9 26.2 – IDEFICS-9B LLaMA-7B 50.9 38.4 35.5 – 25.9 – – 48.2 – – – – Qwen-VL Qwen-7B 78.8 59.3 35.2 67.1 63.8 – – 38.2 56.3 – – – Qwen-VL-Chat Qwen-7B 78.2 57.5 38.9 68.2 61.5 – 1487.5 60.6 58.2 – – – SPHINX LLaMA-13B 78.1 62.6 39.9 69.3 51.6 80.7 1476.1 66.9 56.2 73.5 36.0 56.0 SPHINX-2k LLaMA-13B 80.7 63.1 44.9 70.6 61.2 87.2 1470.6 65.9 57.9 76.9 40.2 59.0 mPLUG-Owl2 LLaMA-7B 79.4 56.1 54.5 68.7 54.3 – 1450.2 64.5 57.8 - 36.2 – LLaVA-1.5 Vicuna-7B 78.5 62.0 50.0 66.8 58.2 85.9 1510.7 64.3 58.6 63.4 30.5 56.3 LLaVA-NeXT Vicuna-7B 81.8 64.2 57.6 70.1 64.9 86.5 1519.0 67.4 70.2 81.6 43.9 62.6 LLaVA-OV Qwen2-7B – – – 96.0 – – 1580.0 80.8 75.4 – – – EMOVA Qwen2.5-7B – – – 96.4 – – – 83.0 75.5 – 59.4 – OpenOmni Qwen2.5-7B – – – – – – – 76.2 – – – – VITA-1.5 Qwen2-7B 78.8 60.6 54.8 90.9 65.0 85.7 1687.7 76.7 70.4 71.0 49.6 64.0 Stream-Omni LLaMA-3.1-8B 79.7 68.3 45.5 93.4 62.7 86.0 1752.7 82.4 76.3 71.2 44.7 64.7

question answering benchmarks, Llama Questions (Llama Q.) [62] and Web Questions (Web Q.) [63], where the metric is the accuracy that whether the model’s response matches the ground-truth answer.

To further assess Stream-Omni’s vision-grounded speech interaction capabilities, we construct a real-world visual-speech interaction benchmark based on the real-world VQA benchmark VisIT [64], named SpokenVisIT3. Following Fang et al. [9], the evaluation for SpokenVisIT employs the GPT model (gpt-4o version) to assign a score ranging from 1 to 5 for response. Appendix B gives the details of SpokenVisIT benchmark. Following previous works [9, 11], all speech evaluations are further divided into speech-to-text (S→T) and speech-to-speech (S→S) settings. For generated speech responses, we use Whisper-large-v34 [29] to transcribe the speech into text for evaluation.

### 4.2 Baselines

We compare Stream-Omni with vision-oriented, speech-oriented, and omni-modal LMMs of similar model scale and training data size. Vision-oriented LMM baselines include models comparable in scale to LLaVA-v1.5 [3], such as BLIP-2 [65], InstructBLIP [66], IDEFICS [67], Qwen-VL [17], Qwen-VL-Chat [17], SPHINX [19], and mPLUG-Owl2 [20]. Speech-oriented LMM baselines include TWIST [68], SpeechGPT [30], Spectron [62], Moshi [10], Freeze-Omni [26], LLaMA-Omni [9], and GLM-4-Voice [11]. Most existing omni-modal LMMs are trained on large-scale proprietary datasets [13, 14, 38]. For a fair comparison, we mainly compare Stream-Omni with VITA-1.5 [12], a text-vision-speech LMM trained on a comparable amount of data, primarily based on LLaVA [3] and LLaVA-OV [6]. Additionally, we also compare Stream-Omni with some methods of similar data scale to demonstrate Stream-Omni’s performance among advanced omni-modal LMMs, such AnyGPT [69], EMOVA [39] and OpenOmni [40]. Note that these models were trained using different datasets and training pipelines with Stream-Omni.

### 4.3 Configuration

Stream-Omni is built upon the LLaMA-3.1-8B-Instruct5 [70], which consists of 32 Transformer layers. For vision, Stream-Omni employs the SigLIP-so400m-patch14-3846 [71] as the vision encoder. For speech, Stream-Omni incorporates the bottom speech layers with 3 Transformer layers and top speech layers with 5 Transformer layers, where all Transformer layers share the same architecture and parameter configuration as those in LLM. The speech tokenizer and flow-matching-based speech decoder are adopted from CosyVoice-300M-25Hz7 [33]. The vocabulary of Stream-Omni comprises 128K text tokens from LLaMA-3.1-8B-Instruct, 4096 speech units from the CosyVoice tokenizer, and a blank token ⟨blank⟩. Stream-Omni is trained using 8 H800 GPUs and tested on 1 A100 GPU.

- 3https://huggingface.co/datasets/ICTNLP/SpokenVisIT
- 4https://huggingface.co/openai/whisper-large-v3
- 5https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- 6https://huggingface.co/google/siglip-so400m-patch14-384
- 7https://modelscope.cn/models/iic/CosyVoice-300M-25Hz

## 5 Results and Analyses

### 5.1 Visual Understanding

We evaluate the visual understanding capabilities of Stream-Omni in Table 2. Compared to advanced vision-oriented LMMs and VITA-1.5 [12], Stream-Omni demonstrates strong visual capabilities on various visual tasks. More importantly, despite being a unified model that simultaneously supports vision, speech, and text, Stream-Omni achieves performance comparable to vision-oriented LMMs, indicating its effectiveness in mitigating modality interference.

##### 5.2 Speech Interaction Table 3: Results on spokenQA benchmarks.

To verify whether Stream-Omni can acquire speech capabilities and knowledge with a small amount of speech data, we conduct experiments on knowledge-based LLaMA Question and Web Question, covering both speech-to-text (S→T) and speech-to-speech (S→S) tasks. As shown in Table 3, Stream-Omni demonstrates strong knowledge-based speech interaction performance. Speech-oriented LMMs based on discrete speech units, such as SpeechGPT, Moshi, and GLM-4Voice, typically rely on speech pretraining to acquire knowledge from large-scale speech data [30, 10, 11], Stream-Omni achieves superior knowledge-based speech interaction with significantly less speech data of 23K hours, particularly in the speech-to-text setting. This advantage primarily stems from the CTC-based speech-to-text mapping in Stream-Omni, which effectively transfers the text knowledge within LLM to the speech modality and thereby supports knowledge-based speech interaction in more efficient manner.

Llama Q. Web Q. Avg.

Methods

S→T S→S S→T S→S S→T S→S

TWIST - 4.0 - 1.5 - 2.8 SpeechGPT 21.6 - 6.5 - 14.1 Spectron 21.9 - 6.1 - 14.0 Moshi 62.3 21.0 26.6 9.2 44.5 15.1 GLM-4-Voice 64.7 50.7 32.2 15.9 48.5 33.3 Freeze-Omni 72.0 - 44.7 - 58.4 LLaMA-Omni 67.7 49.0 33.4 23.7 50.6 36.4 VITA-1.5 76.7 - 42.7 - 59.7 -

Stream-Omni 76.3 65.0 44.2 27.5 60.3 46.3

### 5.3 Vision-grounded Speech Interaction

Most existing benchmarks for evaluating the visiongrounded speech interaction typically use multiple-choice formats, which do not align well with real-world application scenarios. To address this, we constructed SpokenVisIT based on VisIT-Bench [64], a vision-grounded speech interaction benchmark based on real-world scenarios. We evaluate Stream-Omni on the SpokenVisIT benchmark in Table 4. As the omni-modal LMMs with similar training data, Stream-Omni demonstrates superior real-world visual understanding capabilities compared to VITA-1.5. In addition, Stream-Omni supports speech generation, extending its potential for multimodal interaction. Appendix C gives specific case studies, demonstrating the advantages of Stream-Omni’s speech-text mapping in cross-modal consistency.

Table 4: Results on SpokenVisIT (‘V’: vision, ‘T’: text, ‘S’: speech).

SpokenVisIT V+T→T V+S→T V+S→S

Methods

GPT-4V 4.81 - VITA-1.5 3.63 3.45 Stream-Omni 3.93 3.68 2.62

##### 5.4 Quality of Speech-Text Mapping Table 5: Results on LibriSpeech benchmarks.

Stream -ing

test-clean test-other WER

Stream-Omni introduces the auxiliary ASR task to train the bottom speech layers and CTC decoder, thereby learning effective speech-to-text mapping. To evaluate the quality of mapping, we evaluate the ASR performance of Stream-Omni on the LibriSpeech benchmark [49]. As shown in Table 5, Stream-Omni achieves advantages in both accuracy and inference time. SpeechGPT [30], Freeze-Omni [26], and GLM-4-Voice [11] need to forward full LMM to autoregressively generating the ASR results. In contrast, StreamOmni generates the ASR results using its bottom

Methods

Inference Time (ms)

Inference Time (ms)

WER

Whisper × 2.5 692 4.5 616 SpeechGPT × 18.9 794 29.1 755 Moshi ✓ 5.7 - - Mini-Omni × 4.7 196 9.4 148 Freeze-Omni × 3.2 984 7.7 965 GLM-4-Voice × 2.8 756 7.7 701

AnyGPT × 8.5 - - EMOVA × 4.1 - - OpenOmni × 3.1 - 4.1 VITA-1.5 × 3.4 - 7.5 -

Stream-Omni ✓ 3.0 125 7.2 104

speech layers in a non-autoregressive manner, resulting in lower inference time for ASR task. More importantly, this layer-dimension allows Stream-Omni to simultaneously present intermediate ASR results during speech interaction, providing users with a more comprehensive interaction experience.

- 5.5 Effect of Alignment-based Fusion Table 6: Analysis on alignment-based fusion.

Fusion Type

Fusion Window

Llama Q. S→S

Web Q. S→S

Attention 5 65.0 27.5 Add (input) 1 40.3 19.2 Add (per layer) 1 45.3 21.5 Attention 2 54.3 22.1 Attention 10 62.3 25.7 Attention ∞ 60.0 24.3

Stream-Omni generates speech from text in a streaming manner using alignment-based fusion. To evaluate its effectiveness, we conduct the ablation study of alignment-based fusion on Llama Questions and Web Questions benchmarks (S→S) in Table 6, focusing on the fusion type and the fusion window.

Fusion Type For the fusion type, we compare the current cross-attention (named “Attention”) with adding aligned text representations to the input (named “Add (input)”) or each layer (named “Add (per layer)”) of the top speech layers. Results show that the attention-based approach outperforms the others, mainly due to its ability to attend to a broader context rather than merely adding a single text token. Existing speech-oriented LMMs [9, 72] or omni-modal LMMs [40] often mix speech and text representations at the input of the speech decoder to achieve text-to-speech generation. Different with existing methods, Stream-Omni integrates the corresponding textual information into the speech representations at each layer of the speech decoder through alignment-based fusion, enabling high-quality text-to-speech generation.

Fusion Window For the fusion window, we find that attending to either very few or all text tokens during speech generation is less effective than focusing on a moderate window of tokens, which is attributed to the inherent monotonicity and locality in text-to-speech generation. This is also in line with the widely used speech-text interleaved generation methods [35, 11, 73]. The difference lies in that previous methods achieve consistency between generated speech and the current text through interleaving along the sequence dimension, while alignment-based fusion ensures consistency by guiding the speech to attend to the current text along the layer dimension.

- 6 Conclusion

We propose Stream-Omni, a LMM that simultaneously supports various multimodal interactions. Stream-Omni achieves efficient modality alignments via the sequence-dimension concatenation for vision and layer-dimension mapping for speech. Furthermore, Stream-Omni can enhance the multimodal experience by simultaneously providing intermediate text results during speech interaction.

## Limitations

In this paper, we present Stream-Omni, a large multimodal model that supports text, vision, and speech. To address the scarcity of public tri-modal data, we focus on how to model the modality alignment more purposely to achieve efficient and flexible modality alignments. However, beyond the modeling way of modality alignments, high-quality multimodal interaction also rely on other factors, such as speech expressiveness and the degree of human-likeness. These aspects are important but are not the primary focus of Stream-Omni, so we leave them for future work.

## References

- [1] OpenAI. Hello gpt-4o, 2024. URL https://openai.com/index/hello-gpt-4o/.
- [2] OpenAI. Gpt-4v(ision) system card, 2024. URL https://cdn.openai.com/papers/ GPTV_System_Card.pdf.
- [3] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 34892–34916. Curran Associates, Inc., 2023.

- URL https://proceedings.neurips.cc/paper_files/paper/2023/file/ 6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf.
- [4] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=1tZbq88f27.
- [5] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024. URL https: //llava-vl.github.io/blog/2024-01-30-llava-next/.
- [6] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer, 2024. URL https://arxiv.org/abs/2408.03326.
- [7] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.
- [8] Zhifei Xie and Changqiao Wu. Mini-omni: Language models can hear, talk while thinking in streaming, 2024. URL https://arxiv.org/abs/2408.16725.
- [9] Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, and Yang Feng. LLaMAomni: Seamless speech interaction with large language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/ forum?id=PYmrUQmMEw.
- [10] Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. Moshi: a speech-text foundation model for real-time dialogue, 2024. URL https://arxiv.org/abs/2410.00037.
- [11] Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, and Jie Tang. Glm-4-voice: Towards intelligent and human-like end-to-end spoken chatbot, 2024. URL https://arxiv.org/abs/2412.02612.
- [12] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Haoyu Cao, Zuwei Long, Heting Gao, Ke Li, Long Ma, Xiawu Zheng, Rongrong Ji, Xing Sun, Caifeng Shan, and Ran He. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction, 2025. URL https://arxiv.org/abs/2501.01957.
- [13] Yadong Li, Haoze Sun, Mingan Lin, Tianpeng Li, Guosheng Dong, Tao Zhang, Bowen Ding, Wei Song, Zhenglin Cheng, Yuqi Huo, Song Chen, Xu Li, Da Pan, Shusen Zhang, Xin Wu, Zheng Liang, Jun Liu, Tao Zhang, Keer Lu, Yaqi Zhao, Yanjun Shen, Fan Yang, Kaicheng Yu, Tao Lin, Jianhua Xu, Zenan Zhou, and Weipeng Chen. Baichuan-omni technical report, 2024. URL https://arxiv.org/abs/2410.08565.
- [14] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. Qwen2.5-omni technical report, 2025. URL https://arxiv.org/abs/2503.20215.
- [15] Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In Proceedings of the 23rd International Conference on Machine Learning, ICML ’06, page 369–376, New York, NY, USA, 2006. Association for Computing Machinery. ISBN 1595933832. doi: 10.1145/1143844.1143891. URL https://doi.org/10.1145/1143844.1143891.
- [16] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/ radford21a.html.

- [17] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. URL https://arxiv.org/abs/2308. 12966.
- [18] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24185–24198, June 2024.
- [19] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, Jiaming Han, Siyuan Huang, Yichi Zhang, Xuming He, Hongsheng Li, and Yu Qiao. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models, 2023. URL https://arxiv.org/abs/2311.07575.
- [20] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13040–13051, June 2024.
- [21] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, and Yu Qiao. Internvideo: General video foundation models via generative and discriminative learning, 2022. URL https://arxiv.org/abs/2212.03191.
- [22] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding, 2024. URL https: //arxiv.org/abs/2305.06355.
- [23] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Video-ChatGPT: Towards detailed video understanding via large vision and language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12585–12602, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/ v1/2024.acl-long.679. URL https://aclanthology.org/2024.acl-long.679/.
- [24] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models, 2023. URL https://arxiv.org/abs/2311.17043.
- [25] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-LLaVA: Learning united visual representation by alignment before projection. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5971–5984, Miami, Florida, USA, November

2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.342. URL https://aclanthology.org/2024.emnlp-main.342/.

- [26] Xiong Wang, Yangze Li, Chaoyou Fu, Yunhang Shen, Lei Xie, Ke Li, Xing Sun, and Long Ma. Freeze-omni: A smart and low latency speech-to-speech dialogue model with frozen llm, 2024. URL https://arxiv.org/abs/2411.00774.
- [27] Wenyi Yu, Siyin Wang, Xiaoyu Yang, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Guangzhi Sun, Lu Lu, Yuxuan Wang, and Chao Zhang. Salmonn-omni: A codec-free llm for full-duplex speech understanding and generation, 2024. URL https://arxiv.org/abs/2411.18138.
- [28] Wenxi Chen, Ziyang Ma, Ruiqi Yan, Yuzhe Liang, Xiquan Li, Ruiyang Xu, Zhikang Niu, Yanqiao Zhu, Yifan Yang, Zhanxun Liu, Kai Yu, Yuxuan Hu, Jinyu Li, Yan Lu, Shujie Liu, and Xie Chen. Slam-omni: Timbre-controllable voice interaction system with single-stage training,

2024. URL https://arxiv.org/abs/2412.15649.

- [29] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision, 2022.

- [30] Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, and Xipeng Qiu. SpeechGPT: Empowering large language models with intrinsic cross-modal conversational abilities. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 15757–15773, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.1055. URL https://aclanthology.org/2023.findings-emnlp.1055/.
- [31] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3451–3460, 2021. doi: 10.1109/TASLP.2021.3122291.
- [32] Xin Zhang, Dong Zhang, Shimin Li, Yaqian Zhou, and Xipeng Qiu. Speechtokenizer: Unified speech tokenizer for speech language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= AF9Q8Vip84.
- [33] Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, Zhifu Gao, and Zhijie Yan. Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens, 2024. URL https: //arxiv.org/abs/2407.05407.
- [34] Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 17022–17033. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/file/ c5d736809766d46260d816d8dbc9eb44-Paper.pdf.
- [35] Tu Anh Nguyen, Benjamin Muller, Bokai Yu, Marta R. Costa-jussa, Maha Elbayad, Sravya Popuri, Christophe Ropers, Paul-Ambroise Duquenne, Robin Algayres, Ruslan Mavlyutov, Itai Gat, Mary Williamson, Gabriel Synnaeve, Juan Pino, Benoît Sagot, and Emmanuel Dupoux. SpiRit-LM: Interleaved spoken and written language model. Transactions of the Association for Computational Linguistics, 13:30–52, 2025. doi: 10.1162/tacl_a_00728. URL https: //aclanthology.org/2025.tacl-1.2/.
- [36] Boxun Li, Yadong Li, Zhiyuan Li, Congyi Liu, Weilin Liu, Guowei Niu, Zheyue Tan, Haiyang Xu, Zhuyu Yao, Tao Yuan, Dong Zhou, Yueqing Zhuang, Shengen Yan, Guohao Dai, and Yu Wang. Megrez-omni technical report, 2025. URL https://arxiv.org/abs/2502. 15803.
- [37] Qingpei Guo, Kaiyou Song, Zipeng Feng, Ziping Ma, Qinglong Zhang, Sirui Gao, Xuzheng Yu, Yunxiao Sun, Tai-Wei Chang, Jingdong Chen, Ming Yang, and Jun Zhou. M2-omni: Advancing omni-mllm for comprehensive modality support with competitive performance, 2025. URL https://arxiv.org/abs/2502.18778.
- [38] Xingguang Ji, Jiakang Wang, Hongzhi Zhang, Jingyuan Zhang, Haonan Zhou, Chenxi Sun, Yahui Liu, Qi Wang, and Fuzheng Zhang. Capybara-omni: An efficient paradigm for building omni-modal language models, 2025. URL https://arxiv.org/abs/2504.12315.
- [39] Kai Chen, Yunhao Gou, Runhui Huang, Zhili Liu, Daxin Tan, Jing Xu, Chunwei Wang, Yi Zhu, Yihan Zeng, Kuo Yang, Dingdong Wang, Kun Xiang, Haoyuan Li, Haoli Bai, Jianhua Han, Xiaohui Li, Weike Jin, Nian Xie, Yu Zhang, James T. Kwok, Hengshuang Zhao, Xiaodan Liang, Dit-Yan Yeung, Xiao Chen, Zhenguo Li, Wei Zhang, Qun Liu, Jun Yao, Lanqing Hong, Lu Hou, and Hang Xu. Emova: Empowering language models to see, hear and speak with vivid emotions, 2025. URL https://arxiv.org/abs/2409.18042.
- [40] Run Luo, Ting-En Lin, Haonan Zhang, Yuchuan Wu, Xiong Liu, Min Yang, Yongbin Li, Longze Chen, Jiaming Li, Lei Zhang, Yangyi Chen, Xiaobo Xia, Hamid Alinejad-Rokny, and Fei Huang. Openomni: Advancing open-source omnimodal large language models with progressive multimodal alignment and real-time self-aware emotional speech synthesis, 2025. URL https://arxiv.org/abs/2501.04561.

- [41] Qize Yang, Detao Bai, Yi-Xing Peng, and Xihan Wei. Omni-emotion: Extending video mllm with detailed face and audio modeling for multimodal emotion analysis, 2025. URL https://arxiv.org/abs/2501.09502.
- [42] Shaolei Zhang, Qingkai Fang, Zhe Yang, and Yang Feng. LLaVA-mini: Efficient image and video large multimodal models with one vision token. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=UQJ7CDW8nb.
- [43] Shaolei Zhang, Qingkai Fang, Shoutao Guo, Zhengrui Ma, Min Zhang, and Yang Feng. StreamSpeech: Simultaneous speech-to-speech translation with multi-task learning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8964–8986, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. acl-long.485. URL https://aclanthology.org/2024.acl-long.485/.
- [44] Mingbo Ma, Liang Huang, Hao Xiong, Renjie Zheng, Kaibo Liu, Baigong Zheng, Chuanqiang Zhang, Zhongjun He, Hairong Liu, Xing Li, Hua Wu, and Haifeng Wang. STACL: Simultaneous translation with implicit anticipation and controllable latency using prefix-to-prefix framework. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3025–3036, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1289. URL https://www.aclweb.org/anthology/P19-1289.
- [45] Shaolei Zhang, Yang Feng, and Liangyou Li. Future-guided incremental transformer for simultaneous translation. Proceedings of the AAAI Conference on Artificial Intelligence, 35

(16):14428–14436, May 2021. URL https://ojs.aaai.org/index.php/AAAI/ article/view/17696.

- [46] Shaolei Zhang and Yang Feng. Universal simultaneous machine translation with mixture-ofexperts wait-k policy. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7306–7317, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.581. URL https://aclanthology.org/2021.emnlp-main.581.
- [47] Shaolei Zhang and Yang Feng. Information-transport-based policy for simultaneous translation. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 992–1013, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.65. URL https://aclanthology.org/2022. emnlp-main.65/.
- [48] Shaolei Zhang and Yang Feng. End-to-end simultaneous speech translation with differentiable segmentation. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Findings of the Association for Computational Linguistics: ACL 2023, pages 7659–7680, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.485. URL https://aclanthology.org/2023.findings-acl.485/.
- [49] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: An asr corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5206–5210, 2015. doi: 10.1109/ ICASSP.2015.7178964.
- [50] Binbin Zhang, Hang Lv, Pengcheng Guo, Qijie Shao, Chao Yang, Lei Xie, Xin Xu, Hui Bu, Xiaoyu Chen, Chenchen Zeng, Di Wu, and Zhendong Peng. Wenetspeech: A 10000+ hours multi-domain mandarin corpus for speech recognition, 2022. URL https://arxiv.org/ abs/2110.03370.
- [51] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

- [52] Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.
- [53] Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.
- [54] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 2507–2521. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/ 11332b6b6cf4485b84afadb1352d3a9a-Paper-Conference.pdf.
- [55] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.
- [56] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. URL https://openreview.net/forum?id= xozJw0kZXF.
- [57] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2024. URL https://arxiv. org/abs/2306.13394.
- [58] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player?, 2024. URL https://arxiv.org/abs/2307.06281.
- [59] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13299– 13308, June 2024.
- [60] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 34892–34916. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/ 6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf.
- [61] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities,

2023. URL https://arxiv.org/abs/2308.02490.

- [62] Eliya Nachmani, Alon Levkovitch, Roy Hirsch, Julian Salazar, Chulayuth Asawaroengchai, Soroosh Mariooryad, Ehud Rivlin, RJ Skerry-Ryan, and Michelle Tadmor Ramanovich. Spoken question answering and speech continuation using spectrogram-powered LLM. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=izrOLJov5y.
- [63] Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on Freebase from question-answer pairs. In David Yarowsky, Timothy Baldwin, Anna Korhonen, Karen Livescu, and Steven Bethard, editors, Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1533–1544, Seattle, Washington, USA, October

2013. Association for Computational Linguistics. URL https://aclanthology.org/ D13-1160/.

- [64] Yonatan Bitton, Hritik Bansal, Jack Hessel, Rulin Shao, Wanrong Zhu, Anas Awadalla, Josh Gardner, Rohan Taori, and Ludwig Schmidt. Visit-bench: A dynamic benchmark for evaluating instruction-following vision-and-language models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 26898–26922. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/ 5503389dbe070cdae9b48086c4996a59-Paper-Datasets_and_Benchmarks. pdf.
- [65] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 19730–19742. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/li23q.html.
- [66] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26296–26306, June 2024.
- [67] H Laurençon, Daniel van Strien, Stas Bekman, Leo Tronchon, Lucile Saulnier, Thomas Wang, Siddharth Karamcheti, Amanpreet Singh, Giada Pistilli, Yacine Jernite, et al. Introducing idefics: An open reproduction of state-of-the-art visual language model, 2023. URL https://huggingface. co/blog/idefics. Accessed, pages 09–18, 2023.
- [68] Michael Hassid, Tal Remez, Tu Anh Nguyen, Itai Gat, Alexis Conneau, Felix Kreuk, Jade Copet, Alexandre Défossez, Gabriel Synnaeve, Emmanuel Dupoux, Roy Schwartz, and Yossi Adi. Textually pretrained speech language models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id= UlHueVjAKr.
- [69] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, Hang Yan, Jie Fu, Tao Gui, Tianxiang Sun, Yugang Jiang, and Xipeng Qiu. Anygpt: Unified multimodal llm with discrete sequence modeling, 2024. URL https://arxiv.org/abs/2402.12226.
- [70] Meta. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.
- [71] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11975–11986, October 2023.
- [72] Qingkai Fang, Yan Zhou, Shoutao Guo, Shaolei Zhang, and Yang Feng. Llama-omni2: Llmbased real-time spoken chatbot with autoregressive streaming speech synthesis, 2025. URL https://arxiv.org/abs/2505.02625.
- [73] Aohan Zeng, Zhengxiao Du, Mingdao Liu, Lei Zhang, Shengmin Jiang, Yuxiao Dong, and Jie Tang. Scaling speech-text pre-training with synthetic interleaved data, 2024. URL https: //arxiv.org/abs/2411.17607.
- [74] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3029–3051, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. emnlp-main.183. URL https://aclanthology.org/2023.emnlp-main.183/.
- [75] Yao Shi, Hui Bu, Xin Xu, Shaoji Zhang, and Ming Li. Aishell-3: A multi-speaker mandarin tts corpus. In Interspeech 2021, pages 2756–2760, 2021. doi: 10.21437/Interspeech.2021-755.

## A Construction of InstructOmni

Existing publicly available text and vision instruction data are readily accessible, while speech instruction data and tri-modal instruction data involving text, vision, and speech remain relatively scarce. To address this, we propose InstructOmni, an omni-modal dataset automatically constructed using text-to-speech (TTS) synthesis. InstructOmni builds upon existing publicly available textonly and vision-language instruction datasets by generating corresponding speech based on textual instructions and responses, thereby producing both speech-based instruction data and tri-modal instruction data for training. Specifically, we synthesize speech instruction data from the LLaVA visual instruction tuning dataset [3], the UltraChat text instruction tuning dataset [74] (used in LLaMA-Omni [9] and LLaMA-Omni2 [72]), and a subset of Wikipedia entries. The text instructions and responses from these sources are converted into speech using the CosyVoice TTS model [33]. To better simulate the variability of speech input in real-world scenarios, we randomly sample speaker embeddings from LibriSpeech [49] and AISHELL [75], and apply voice cloning techniques to generate speech with diverse speaker characteristics, thereby enhancing the realism and diversity of the speech.

Table 1 summarizes the used datasets during training, where those marked with a superscript ‘tts’ indicate samples with synthesized speech. Overall, Stream-Omni is trained on only 23K hours of speech data, which is significantly less than the large-scale datasets used in previous methods, such as TWIST (150K hours) [68], SpeechGPT (60K hours) [30], Moshi (7M hours) [10], GLM-4-Voice (700K hours) [11], and VITA-1.5 (110K hours) [12], highlighting its advantage in data efficiency.

## B Construction of SpokenVisIT

In Sec.5.3, to align with real-world application scenarios, we construct SpokenVisIT benchmark based on VisIT-Bench [64] to evaluate the vision-grounded speech interaction capability of omni-modal LMMs. Here, we give a detailed introduction to SpokenVisIT.

To better reflect real-world scenarios of vision-based speech interaction, we adopt the VisITBench [64] as the source dataset (an open-ended generation format instead of multi-choice format is much suitable for real-world scenarios). VisIT is a real-world visual question answering benchmark comprising 574 images and 70 types of instructions covering object recognition, visual reasoning, creative writing, and more. Unlike existing vision evaluation benchmarks that mainly use multiplechoice format, all text instructions in the VisIT benchmark are written in a colloquial style, making it particularly well-suited for speech interaction. To adapt VisIT for speech interaction, we employ text-to-speech synthesis [33] to convert each text instruction into a corresponding speech utterance, resulting in a derived benchmark named SpokenVisIT. During construction, eight math-related instructions that were unsuitable for speech interaction were removed. For the evaluation metric, following the open-ended spoken interaction evaluation protocol proposed by Fang et al. [9], we use ChatGPT (gpt-4o version) to assess the quality of responses on a 1-5 scale. The evaluation prompt includes the image caption as a reference, along with the question and the model’s answer.

Prompt of SpokenVisIT Evaluation

I need your assistance in evaluating the performance of several models in a vision-based speech interaction scenario. These models process the user’s spoken input and generate spoken responses. For evaluation purposes, both the user’s speech input and the model’s speech output have been transcribed into text using Automatic Speech Recognition (ASR). Additionally, a brief image caption is provided to help you understand the visual context of the conversation. Your task is to assess the model’s responses based on the given visual context [Image Caption], the transcribed user input [Instruction], and the transcribed model output [Response]. Please evaluate the responses considering factors such as helpfulness, responsiveness, empathy, and suitability for real-world multimodal interaction, and assign a single score on a scale from 1 to 5.

Below are the image caption and the transcriptions of the user’s instruction and the model’s response: ### [Image Caption]: {caption} ### [Instruction]: {question} ### [Response]: {answer}

After evaluating, please output the scores in JSON format: {score: ...}. You don’t need to provide any explanations.

[Figure 34]

Question: Does the stairway from the basement allow traveling to the second floor without moving through another room first?

###### Speech:

Vision + Text → Text GPT-4V: Yes, the stairway from the basement allows traveling to the second floor without moving through another room first, as it is in the same position in the plans for all three levels. VITA-1.5: Based on the image, the stairway from the basement does not allow traveling to the second floor without moving through another room first. The stairway is shown as a direct vertical connection between the basement and the second floor, but there are no rooms depicted between these two levels. Stream-Omni: Based on the image, the stairway from the basement leads directly to the second floor, bypassing the first floor. This suggests that one can travel from the basement to the second floor without moving through another room first.

mismatch

Vision + S𝐩𝐞𝐞𝐜𝐡 → Text VITA-1.5: Yes, the staircase from the basement leads directly to the second floor without requiring passage through another room. Stream-Omni: Yes, the stairway from the basement leads directly to the second floor, allowing for travel between the basement and the second floor without passing through another room first.

Vision + S𝐩𝐞𝐞𝐜𝐡 → Speech Stream-Omni:

10s speech

(Whisper Transcription: Yes, the stairway from the basement leads directly to the second floor, allowing for travel between the basement and the second floor without passing through another room.)

Figure 4: Case Study of Stream-Omni (detail understanding).

## C Case Study

To provide a more intuitive demonstration of Stream-Omni’s multimodal interaction capabilities, we conduct two case studies in Figure 4 and 5, where both the visual and speech inputs are sourced from the constructed SpokenVisIT benchmark. The case in Figure 4 focuses on visual detail understanding, while the case in Figure 5 highlights the model’s ability to generate long speech responses. The red-marked text indicates the incorrect part of the response. In both cases, Stream-Omni demonstrates good performance across different modalities. Specifically, in vision-based text interaction, StreamOmni accurately interprets visual inputs and generates output sequences that closely resemble those produced by GPT-4V [2]. When conditioned on both visual and speech inputs, Stream-Omni outperforms VITA-1.5.

- In the example shown in Figure 4, when the instruction is delivered via text and speech respectively, VITA-1.5 produces two contradictory responses of "does not allow traveling to the second floor" and "leads directly to the second floor". This contradictory response when facing different modal instructions stems from VITA-1.5’s sequence-dimension concatenation of visual, speech, and text representations to achieve multimodal alignment [12], without modeling rigorous semantic alignment between the speech and text modalities. In contrast, Stream-Omni employs the speech-to-text mapping that enables precise semantic alignment between speech and text representations. As a result, Stream-Omni achieves more consistent performance across modalities and can generate similar responses regardless of whether the instruction is delivered via text or speech.

[Figure 35]

Question: How does this device help you decide where to hang a picture?

###### Speech:

Vision + Text → Text GPT-4V: This device helps you decide where to hang a picture by locating the wall studs behind the wallpaper, ensuring a secure and stable spot to anchor the picture frame. VITA-1.5: This device, known as a stud finder, is designed to detect the presence of wall studs behind the surface. These studs are crucial for hanging heavy items like pictures because they provide structural support. By using this tool, you can identify the best spots to hang your picture frame where it will be secure and less likely to fall off the wall. Stream-Omni: The device in the image is a stud finder, which is used to locate the studs behind a wall. To decide where to hang a picture, you would use this device to scan the wall at the spot where you want to hang the picture. The stud finder will detect the presence of a stud, indicated by a beep or a signal, and mark the location. This helps ensure that the picture frame is hung on a sturdy part of the wall, providing support and reducing the risk of the frame falling or the wall damaging.

Vision + S𝐩𝐞𝐞𝐜𝐡 → Text VITA-1.5: This device, known as a spirit level, helps ensure that pictures and other items are hung straight by providing visual feedback on the horizontal or vertical alignment. It's particularly useful for achieving precise and aesthetically pleasing results when hanging artwork or mirrors. Stream-Omni: The device in the image is a stud finder, which helps in determining the location of wooden studs behind a wall. This is useful when hanging a picture because you want to find a spot where the picture will be securely supported by a stud. If you hang a picture on a spot without a stud, it may not be well-supported and could lead to the picture falling. By using a stud finder, you can locate a spot where a stud is present, ensuring that the picture will be securely mounted and safe from falling.

Vision + S𝐩𝐞𝐞𝐜𝐡 → Speech Stream-Omni:

30s speech

(Whisper Transcription: The device in the image is a stud finder which helps to determining the location of the wooden studs behind a wall. This is useful when hanging picture because you want to find a spot where the picture will be securely supported by a stud. If you hang a picture on a spot without a stud, it may not be well-being and could lead to the picture falling. By using a stud finder, you can locate a spot where a stud is present, ensuring that the picture will be securely mounted and safe from falling.)

Figure 5: Case Study of Stream-Omni (long response).

- In the example shown in Figure 5, Stream-Omni exhibits strong speech generation capabilities, producing high-quality speech outputs lasting up to 30 seconds. Notably, the generated speech is highly consistent with the corresponding text outputs, underscoring the effectiveness of the proposed alignment-based fusion module. Overall, Stream-Omni enables high-quality, vision-grounded speech interactions, fulfilling the diverse requirements of multimodal interaction.

