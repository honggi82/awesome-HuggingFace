## Omni-Diffusion: Unified Multimodal Understanding and Generation with Masked Discrete Diffusion

# arXiv:2603.06577v1[cs.CV]6Mar2026

Lijiang Li1 Zuwei Long2 Yunhang Shen2 Heting Gao2 Haoyu Cao2 Xing Sun2 Caifeng Shan1 Ran He3 Chaoyou Fu1†

1 Nanjing University, 2 Tencent Youtu Lab, 3 CASIA

lijiangli@smail.nju.edu.cn, bradyfu24@gmail.com

### Abstract

While recent multimodal large language models (MLLMs) have made impressive strides, they predominantly employ a conventional autoregressive architecture as their backbone, leaving significant room to explore effective and efficient alternatives in architectural design. Concurrently, recent studies have successfully applied discrete diffusion models to various domains, such as visual understanding and image generation, revealing their considerable potential as a promising backbone for multimodal systems. Drawing inspiration from these pioneering research, we introduce Omni-Diffusion, the first any-to-any multimodal language model built entirely on mask-based discrete diffusion models, which unifies understanding and generation across text, speech, and images. Omni-Diffusion employs a unified mask-based discrete diffusion model to directly capture the joint distribution over discrete multimodal tokens. This approach supports not only bimodal tasks but also more complex scenarios involving multiple modalities. On a diverse set of benchmarks, our method outperforms or performs on par with existing multimodal systems that process two or more modalities, highlighting the significant promise of diffusion models in powering the next generation of multimodal foundation models. Project webpage: https://omni-diffusion.github.io.

### 1. Introduction

In the past few years, significant advances have been made in the research of multimodal intelligence. One important direction in this field is designing a unified model that can process tasks involving data from various modalities, including text, images, speech, and beyond. To achieve this goal, many studies have developed multimodal systems by augmenting pretrained large language models (LLMs) with multimodal per-

† Corresponding author.

ception and generation capabilities [5, 10, 12, 40, 44, 46, 54], which exhibit impressive performance thanks to the strong language understanding capabilities of LLMs. However, most existing approaches in multimodal intelligence rely on autoregressive architectures, leaving substantial room to explore alternative probabilistic modeling approaches.

Recent months have witnessed a surge of research interest in applying diffusion models to natural language processing tasks, which has emerged as a promising alternative to classical autoregressive architectures [47, 55]. Diffusion models demonstrate several distinct advantages over autoregressive models [28, 51]. For example, the initial token sequence and the generation trajectory of the diffusion generation process can be steered to control the semantic structure, output format and response style of generated content [43]. Furthermore, diffusion models support parallel decoding, offering the potential for efficient generation [38, 39]. Given these advantages, it is natural to explore the potential of employing diffusion models to build multimodal intelligence systems.

In this work, we introduce Omni-Diffusion, the first anyto-any multimodal language model that builds upon a maskbased discrete diffusion model for unified comprehension and generation. As illustrated in Fig. 1, Omni-Diffusion employs a mask-based discrete diffusion model to learn the joint distribution of multimodal semantic tokens obtained by tokenizing raw text, image, and speech data. In contrast to existing multimodal models that utilize an LLM to generate textual data and rely on additional output models to convert the LLM’s textual hidden states into outputs of other modalities [27, 40], this joint modeling of multimodal discrete tokens enables the model to develop an intrinsically aligned semantic representation space, thereby equipping it with unified comprehension and generation capabilities across various modalities.

We introduce a suite of training and inference techniques tailored to the characteristics of mask-based discrete diffusion models, which facilitate the extension of a pre-trained diffusion language model into a multimodal system. First,

###### Speech Tasks

###### Visual Tasks

###### Speech-Diven Visual Interaction

(Captioning, Visual QA, Text-to-Image, …)

(Speech-to-Image, Spoken Visual Understanding, …)

###### (ASR, TTS, …)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Q: Convert the text to speech. His long hair blown by the pelting storm, and laid flat across his immovable countenance.

Q2: Generate an image of a beautiful stunning plant animal hybrid, which resembles a fantasy creature with elements of a ferret

Q1:

Q2:

Q:

[Text Transcribe: Generate an image of a flamingo stands in the sand at the beach.

Q1: Describe the Image in details

[Text Transcribe: What cat you see in this image]

Text Tokenizer & Speech Encoder & Image Tokenizer

Text Tokens Image Tokens Speech Tokens

Mask-based Discrete Diffusion Model

Text Tokens Image Tokens Speech Tokens

###### Text Tokenizer & Speech Decoder & Image Tokenizer

[Figure 5]

[Figure 6]

A1:

[Figure 7]

[Figure 8]

A2: A2:

A1: A dog is sitting on the back of a white boat over the sea.

A:

[Text Transcribe: A group of elephants are walking across a muddy field.]

[Text Transcribe: His long hair blown by the pelting storm, and laid flat across his immovable countenance.]

Figure 1. Overview of Omni-Diffusion. Our model takes multimodal tokens as input, processes them using a unified mask-based discrete diffusion model, and generates output data of the desired modality. By modeling the joint distribution over discrete multimodal tokens, Omni-Diffusion can handle not only bimodal tasks (e.g., automatic speech recognition, text-to-speech, visual QA, and text-to-image) but also tasks requiring the integration of more than two modalities, such as speech-to-image generation and spoken visual understanding.

we propose a three-stage progressive training pipeline to extend the model to encompass multimodal comprehension and generation capabilities. To equip Omni-Diffusion with the ability to handle any-to-any multimodal conversation, we construct a speech-driven visual interaction (SDVI) dataset that consists of samples requiring both visual and speech capabilities. Furthermore, we employ an attenuated tail-pad masking strategy to enhance the model’s ability to generate responses of variable lengths. Finally, we optimize the inference process based on the characteristics of mask-based discrete diffusion models. For the image modality, we introduce a position penalty to constrain the generation order and improve the visual quality. For the speech modality, we propose a special token pre-infilling strategy that allows the model to incorporate text semantics during speech generation. Additionally, we apply an adaptive token-length initialization strategy to speech understanding and generation tasks to further boost performance.

els. For training, we implement an attenuated tail-pad masking strategy to facilitate variable-length generation and a three-stage progressive training pipeline for effective multi-modality alignment. For inference, we introduce position penalty to constrain generation order and enhance visual quality, alongside a special token pre-infilling strategy to improve spoken dialogue performance.

• We conduct extensive experiments to evaluate the performance of our method. Comprehensive evaluations reveal that Omni-Diffusion achieves performance comparable to or even better than existing autoregressive multimodal system processing two or more modalities on various benchmarks, thereby providing valuable insights into developing discrete diffusion models for multimodal intelligence.

### 2. Related Work

##### 2.1. Multimodal Large Language Models

In summary, our key contributions are as follows:

Recent multimodal research has focused on unified models for diverse input and output modalities. Several studies have developed foundation models for multimodal comprehension. For example, OneLLM [15] aligns eight modalities to an LLM using modality-specific tokenizers and progressive training strategy. Video-SALMONN [32] proposes to connect audio-visual encoders to an LLM via a Q-former for video and speech understanding. The VITA series [10, 12] introduce a duplex communication mechanism to multimodal LLMs for natural multimodal human-computer interaction

- • We introduce Omni-Diffusion, the first any-to-any multimodal language model built on a mask-based discrete diffusion model. By modeling a joint distribution over multimodal discrete tokens, Omni-Diffusion enables the alignment of different modalities in a shared semantic representation space, exhibiting strong capabilities in multimodal comprehension and generation.
- • We develop specialized training and inference techniques based on the characteristic of mask-based diffusion mod-

[Figure 9]

experience. Beyond multimodal comprehension, recent research extends LLMs to accommodate arbitrary input and output modalities, resulting in unified any-to-any frameworks. AnyGPT [54] processes discrete tokens across modalities with a unified LLM to enable any-to-any conversations. NExT-GPT [40] connects pretrained diffusion decoders to a frozen LLM via adapters for multimodal generation. While most existing works use autoregressive architectures, NExTOmni [27] employs a discrete flow matching model to generate multimodal content. Although effective, NExT-Omni is restricted to a text-only backbone and requires additional models for multimodal generation. In contrast to existing methods, we propose modeling the distribution of discrete multimodal tokens directly within a unified, mask-based discrete diffusion model.

[Figure 10]

Full Attention on Text Image Speech multimodal tokens

[Figure 11]

Multimodal Detokenizer

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

SpeechImageText

Mask-based Discrete Diffusion Model

Text Image Speech

|BoI| |EoI| |BoS| |EoS|

Random Mask

Mask Token

|BoI| |EoI| |BoS| |EoS|

Speech Token

Multimodal Tokenizer

Image Token

[Figure 12]

[Figure 13]

Text Image Speech

[Figure 14]

Text Token

Figure 2. Architecture overview. Omni-Diffusion is an any-to-any multimodal system built on the mask token based discrete diffusion model. By modeling a unified distribution of multimodal discrete tokens through the mask token prediction, Omni-Diffusion enables to perform comprehension and generation of various modalities, including text, image, and speech.

- 2.2. Mask-based Discrete Diffusion Models

Mask-based discrete diffusion models (MDMs) are a class of generative models that exhibit impressive performance on various tasks, such as natural language processing [2, 47, 55], image generation [3, 45], and visual understanding [49, 51]. MDMs model the distribution of target discrete token sequences through mask token prediction. During training, MDMs tipically corrupt a clean data sequence sampled from a training dataset by replacing its tokens with a special [MASK] token. Then, a neural network is optimized to predict the original unmasked tokens given the partially masked context. In the inference process, MDMs start with a fully masked sequence and iteratively decode the mask tokens, gradually reconstructing the clean data distribution. While various pioneering works have proposed employing MDMs as the backbone of LLMs [47, 55], we further extend MDMs to a unified multimodal understanding and generation system in this work.

- 3. Method

{tn}N

n=1,{sn}N

n=1,{in}N

n=1 . The token sequences of different modalities are then wrapped with special beginning and end tokens of the corresponding modality, which together form a unified token sequence x0 ∈ RL. Following the common training process of diffusion model [17, 26], we corrupt the token sequence x0 by randomly replacing its tokens with a special mask token [MASK] at a ratio r, where r is derived from the time step t that sampled uniformly from the interval [0,1] at each training iteration. Our model takes the corrupted token sequence at time step t, denoted as xt, as input and predicts the clean token sequence, denoted as xˆ0 = pθ(x0|xt). Therefore, the training loss is the crossentropy between model prediction xˆ0 and the clean token sequence x0:

t

s

i

L

I xit = [MASK] log pθ(xi0|xt)

L = −Et,q(x

t|x0)

i=1

- 3.1. Unified Probabilistic Formulation over Multimodal Discrete Tokens

(1) where I[·] is an indicator function that ensures the crossentropy loss is only calculated for the masked tokens in xt. With this cross-entropy loss, we train our model on text, speech, and image data in a unified mask-token prediction framework, and no modality-specific optimization is employed during training.

As shown in Fig. 2, Omni-Diffusion is built upon a pretrained diffusion language model and performs unified learning over the joint distribution of multimodal discrete tokens. While many existing multimodal systems rely on an additional output model to project the textual features from LLMs into the generated multimodal data [27, 40], our method directly models an intrinsically unified multimodal discrete representation space, thereby achieving effective comprehension and generation of data with various modalities.

##### 3.2. Model Architecture

Omni-Diffusion is built upon a mask-based discrete diffusion language model and is equipped with distinct tokenizers for data of various modalities. The tokenization of different modalities and the model backbone is detailed as follows.

Specifically, we formulate our model as a unified masktoken predictor for discrete tokens of various modalities, including text, speech, and images. Given a data pair (T,S,I) consisting of text T, speech S, and image I, we first tokenize them into discrete representation

Image Tokenization. We leverage the pre-trained MAGVIT-v2 [50] as an image tokenizer, following exist-

ing visual language models [41, 45]. This image tokenizer compresses images into a compact representation with a downsampling factor f = 16 through a visual encoder. Then, a quantizer with a codebook size of 8192 is employed to convert the compact image representation into discrete tokens. We use the resulting discrete image tokens for both visual understanding and generation tasks in our implementation.

Text Image Speech

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Multimodal Detokenizer

[Figure 19]

Mask-based Discrete Diffusion Model

[Figure 20]

Speech Encoder and Decoder. We employ SenseVoiceSmall [1] and the GLM-4-Voice decoder [53] for speech encoding and decoding, respectively, following VITA-Audio [25]. SenseVoiceSmall utilizes a memory-equipped selfattention network to extract semantically rich representations from input speech. These representations are then projected into the hidden dimension of our discrete diffusion model backbone via a lightweight MLP adapter. For speech generation, we leverage the GLM-4-Voice decoder. Its speech tokenizer transforms the speech into discrete tokens at a token rate of 12.5Hz through a finite scalar quantization layer with a codebook size of 16384. Our model is trained to predict the speech tokens conditioned on multimodal input, which are finally reconstructed into waveforms by the GLM-4-Voice decoder.

Multimodal Tokenizer

[Figure 21]

[Figure 22]

[Figure 23]

Text Image Speech

###### Stage1 Stage2 Stage3

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

TextImage Data

TextSpeech Data

TextImage Data

TextImage Data

TextSpeech Data

SDVI Data

Figure 3. Training pipeline of Omni-Diffusion. The first stage pre-aligns the textual capability of pre-trained diffusion language model with the visual modality. The second stage further enhances the multimodal capability of diffusion model by jointly training on the speech and visual data. The last stage optimizes the model on our constructed SDVI datasets that consisting of speech-to-image and image-to-speech tasks, which further enhances the unified multimodal alignment of our model across various modality.

LLM Backbone. We employ Dream-7B [47], a pretrained discrete diffusion language model, as our base model. To enable multimodal processing, we expand the vocabulary to accommodate 16384 speech tokens and 8192 image tokens. Aside from extending the vocabulary as well as the corresponding embedding and output layer, the architecture of discrete diffusion model remain unaltered.

with the semantic space of the pre-trained language model. Stage 2 (Speech–Vision–Language Joint Alignment) focuses on enhancing the alignment between text and other modalities. In this stage, we retain the visual-text datasets from Stage 1 and introduce automatic speech recognition (ASR) and text-to-speech (TTS) data to facilitate speech-text alignment. Stage 3 (Speech-Driven Visual Interaction Capability Improvement) aims to improve unified cross-modal alignment. In this stage, we fine-tune the model on our constructed Speech-Driven Visual Interaction (SDVI) dataset, which comprises samples for spoken visual question answering and speech-to-image generation that requires joint processing of speech and visual data. We also incorporate spoken question answering (SQA) and visual question answering (VQA) data during this final stage.

##### 3.3. Training

To achieve an efficient and stable training process, we propose a three-stage progressive training pipeline to extend the multimodal understanding and generation capabilities of the pre-trained diffusion language model effectively. Besides, we construct a Speech-Driven Visual Interaction Dataset (SDVI) consisting of spoken visual question answering and speech to image data to further improve the unified alignment of our model across various modalities. In addition, we propose an attenuated tail-pad masking strategy to encourage model to generate responses of variable lengths.

Three-Stage Progressive Training Pipeline. As illustrated in Fig. 3, our training pipeline progressively expands the set of modalities and tasks throughout the training process. This strategy ensures stability when training a unified model on data distributions with distinct characteristics. Training datasets are detailed in Tab. 4 of Appendix. Stage 1 (Visual-Language Pre-Alignment) optimizes the pre-trained diffusion language model on text-to-image and image captioning tasks. This stage aims to align the visual modality

SDVI Dataset Construction. We construct the SpeechDriven Visual Interaction (SDVI) dataset that mainly includes spoken visual question answering and speech-toimage generation data to improve the model’s ability for visual interaction via spoken instruction. For spoken visual question answering, we use LLaVA-OneVision [21] as the data source and employ the Cosyvoice2 [9] model to convert

the textual question-answering pairs into speech. We design a processing pipeline prior to speech synthesis to ensure the data quality. Specifically, we first filter out all samples that contain mathematical computation or programming, as these are uncommon in daily spoken dialogue scenarios. Next, we rewrite all multiple-choice questions as open-ended question answering by replacing the answer choices with the corresponding answer words or sentences. Finally, we remove all samples with an answer length greater than 100 words, since humans usually prefer concise response in spoken conversation. To prevent the model from overfitting to a particular voice, we convert the question part of the processed LLaVAOneVision dataset into speech by performing voice cloning conditioned on 1,000 randomly sampled speech samples of the GigaSpeech datasets [4] using the Cosyvoice2 [9] model, while the answer part is converted into speech using a fixed voice. The resulting dataset contains over 30,000 samples, and each sample consists of a spoken input question, an input image, a textual output answer, and the corresponding spoken output response.

For speech to image generation, we select Blip3oPretrain-JourneyDB [5] as the data source for its fluent native text and high-quality images. Similar to the spoken visual question answering task, we convert the text into speech by performing voice clone on 1,000 randomly sampled speech from the Gigaspeech dataset with the Cosyvoice2, resulting in 30,000 speech-image pairs.

Attenuated Tail-Pad Masking. To facilitate variablelength generation, we adapt a tail-pad augmentation strategy that appends a random number of pad tokens to the end of each data sample, consistent with prior diffusion model training methodologies [51]. During model training, both the original and pad tokens are randomly masked and serve as prediction targets. However, we observe that a simple uniform masking strategy leads to overfitting on the special pad token, resulting in the generation of excessive pad token during inference. To solve this issue, we propose Attenuated Tail-Pad Masking, which applies a scaling factor γ (γ < 1) to reduce the mask ratio specifically for pad token. By attenuating the mask ratio, we ensure that the model’s gradient updates are predominantly driven by the regular semantic tokens rather than the pad token, thereby mitigating overfitting and improving generation quality.

##### 3.4. Inference

We utilize an entropy-based decoding strategy consistent with Dream-Instruct-7B [47]. Furthermore, we propose a position penalty to enhance image generation, alongside special token pre-infilling and adaptive token length assignment to enhance spoken dialogue performance. These techniques are detailed in this section.

Entropy-based Decoding Strategy. During inference, we decide which tokens to decode based on the entropy of the token probabilities. To further improve generation quality, we also integrate the repetition penalty and classifier-free guidance into the inference process. Assume the token logits produced by the model backbone at time step t are denoted by zt ∈ RL×V , where L and V represent the sequence length and vocabulary size, respectively. The logits are then adjusted by repetition penalty and classifier-free guidance, and the token probabilities are obtained by computing pt = softmax(zt). We determine the token confidence cit at each position i (i = 1,2,··· ,L) according to the entropy Ht of the token probability pi,vt , which is estimated as follows:

cit = −Hti =

V

pi,vt · log(pi,vt ) (2)

v=1

We select the top-k tokens with the highest confidence and determine their values by sampling from the token probability pt, while the remaining mask tokens are kept unchanged. The sampling process begins with a fully mask token sequence and iterates until all mask tokens are decoded.

Position Penalty. We propose a position penalty strategy to improve the generation quality of image. Specifically, we observe that the model occasionally generates repetitive patterns in images. We hypothesize that these repetitive patterns arise because the model tends to decode mask tokens from the beginning and the end of the sequence towards the center, a phenomenon discussed in prior studies [18]. Since diffusion models typically generates tokens with related and similar semantics within consecutive time steps, simultaneous decoding at both ends of the mask tokens sequence can result in identical patterns appearing in the top and bottom regions of the generated image. To address this problem, we propose the position penalty strategy. During the early stages of inference, we scale down the logits of the last Nt

tokens by a fixed factor γp (γp < 1). This strategy discourages the model from decoding the beginning and end of the sequence simultaneously, therefore reducing repetitive patterns and improve visual quality. It is worth noting that our position penalty differs from semi-autoregressive generation [45], which splits the sequence into blocks and generates each block autoregressively. In contrast, our approach applies a soft constraint on the generation order without regidly forcing the model to generate tokens in specific regions.

Special Token Pre-Infilling. A key advantage of discrete diffusion models is the flexibility to modify the initial mask token sequence to control the output format [43]. Based on this mechanism, we propose Special Token Pre-Infilling to enhance the model performance in spoken dialogue tasks. Specifically, for an initial mask token sequence of length

L, we replace the mask token at index 0.25L with a special token [begin-of-speech]. This method guides the model to generate a text response in the first 0.25L segment and the corresponding speech response in the remaining 0.75L segment simultaneously. Consequently, the model can explicitly attend to text content during speech generation, thereby improving the logic and coherence of the synthesized speech.

Adaptive Token Length Assignment. Similar to the special token pre-infilling strategy, we propose an adaptive assignment of the initial mask token sequence length for ASR and TTS tasks. This approach is motivated by the strong correlation between speech duration and text length, which allows us to approximate the length of one modality given the other. Accordingly, we set the initial token length to 3.5 times the text token length for TTS task, and 0.2 times the speech token length for ASR task. This strategy not only improves the performance of speech understanding and generation but also accelerates the sampling process by decreasing the number of tokens to be decoded.

### 4. Experiment

We evaluate the performance of Omni-Diffusion across various multimodal benchmarks in this section. Furthermore, we assess the model’s capabilities in fast sampling and image inpainting. The experimental settings and implementation details are provided in Sec. A of the Appendix.

##### 4.1. Main Results

Speech Tasks. We evaluate the speech capabilities of our model on ASR and TTS tasks by calculating the word error rate (WER) on the LibriSpeech [29] and LibriTTS [52] benchmarks. We compare our model with existing TTS model CosyVoice [8], the speech LLM GLM-4-Voice [53], and the any-to-any multimodal LLM AnyGPT [54]. To evaluate the WER for TTS, we employ Whisper-Large-V3 [30] to transcribe the generated speech into text. As shown in Tab. 1, compared with the autoregressive any-to-any model AnyGPT [54], our method achieves better performance on speech tasks. In addition, Omni-Diffusion demonstrates comparable performance on the TTS task compared with the TTS expert model and shows significant improvement over the speech-specific LLM.

Visual Tasks We evaluate the visual understanding and generation capabilities of Omni-Diffusion on VQA and textto-image generation tasks, with results shown in Tab. 2. We compare our method with existing visual LLMs, including mPLUG-Owl [48], LLaVA [24], InstructBLIP [6], DreamLLM [7], and Emu [35], as well as the any-to-any multimodal LLM AnyGPT [54] and NExT-GPT [40]. For the VQA task, we evaluate model performance on several widely

Table 1. Performance on ASR and TTS tasks evaluated on the LibriSpeech [29] and LibriTTS [52] benchmarks. Omni-Diffusion exhibits comparable performance on ASR and superior performance on TTS compared to existing specialized speech models.

LibriSpeech LibriTTS Method Model Type

WER ↓ WER ↓ CosyVoice TTS model - 2.89

GLM-4-Voice Speech LLM 2.82 5.64 AnyGPT Any-to-Any 8.50 -

Omni-Diffusion Any-to-Any 7.05 3.07

applied benchmarks, including POPE [22], MME-Perception [11], and Seed-2-Plus [20]. For text-to-image generation, we evaluate CLIP-T and CLIP-I score on 10,000 images randomly sampled from the MSCOCO 2014 validation set [23], following the methodology established by Emu [34]. CLIPT denotes the average cosine similarity between the prompts and the generated image CLIP embeddings, while CLIP-I represents the average cosine similarity between generated and real image CLIP embeddings.

The experimental results in Tab. 2 demonstrate that OmniDiffusion achieves strong performance in both visual understanding and generation. Omni-Diffusion achieves performance comparable to specialized visual LLMs in both domains. However, unlike visual LLMs designed specifically for visual tasks, our method distinguishes itself by supporting a more diverse range of modalities and tasks. Furthermore, our method achieves better visual understanding performance than existing any-to-any models. In the text-to-image task, our method achieves superior text-image alignment compared to other any-to-any models, and visual quality comparable to methods that rely on external pretrained diffusion models.

Speech-Vision Alignment Evaluation We examine the model’s ability to achieve unified alignment across the speech, image, and text modalities by evaluating its performance on speech-to-image generation. Specifically, we randomly sample 10,000 captions from the MSCOCO validation set and convert these captions into speech using the CosyVoice2 model. After that, we employ our model to generate images conditioned on the synthesized speech. As shown in Tab. 3, Omni-Diffusion achieves similar generation quality conditioned on text and speech, highlighting the model’s strong alignment across various modalities.

To demonstrate the effectiveness of Omni-Diffusion for tasks involving spoken interaction with visual content, we present qualitative examples illustrating the model’s ability to generate spoken responses to spoken questions regarding image content in Fig. 4. From a speech perspective, Omni-Diffusion effectively understands the user’s spoken

Table 2. Performance on VQA and text-to-image tasks. VQA performance is evaluated on the POPE, MME (Perception), and Seed-2-Plus benchmark, while text-to-image task is evaluated by the CLIP-T and CLIP-I scores on the MSCOCO dataset. “†” represents visual LLMs capable of understanding only. “‡” denotes models using external pretrained diffusion model. “*” denotes evaluation results using the official released code and model checkpoint.

Image Question Answering Text-to-Image Method Model Type #Params

POPE↑ MME-P↑ Seed-2-Plus↑ CLIP-T↑ CLIP-I↑ mPLUG-Owl Visual LLM† 7B - 976.34 31.8 - -

LLaVA Visual LLM† 7B 76.3 809.6 30.1 - InstructBLIP Visual LLM† 14B 78.9 1212.8 29.2 - DreamLLM Visual LLM 7B 69.2∗ - - 0.238∗ 0.697∗

Emu‡ Visual LLM 14B - - 33.5 0.286 0.656

AnyGPT Any-to-Any 8B 67.7∗ - - - 0.650 NExT-GPT‡ Any-to-Any 7B - - 26.2 0.225∗ 0.691∗ Omni-Diffusion (Ours) Any-to-Any 7B 76.6 1216.7 34.5 0.235 0.667

on the unmasked part of the input image and the prompt. These results demonstrate the advantages of diffusion-based generation systems compared with autoregressive models for downstream visual generation tasks.

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Text Transcription: What can be inferred about the building's purpose based on the presence of residential units?

[Figure 34]

Text Transcription: Based on the way the elephants are positioned, can you infer any relationship or interaction between them?

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

##### 4.3. Sampling Efficiency

Text Transcription: The building is likely used for residential or mixed-use purposes, as suggested by the presence of multiple units and a mix of architectural styles on the building's exterior.

Text Transcription: The elephants are walking together, with the adult and baby elephant following each other. They seem to be in a social setting, possibly as part of a herd.

Table 3. Performance of image generation and TTS across various numbers of inference time steps. Metrics: CLIP-T/CLIP-I(↑) for image generation and WER (↓) for TTS. L denotes the sequence length of TTS task. Our model maintains strong performance even as the number of time steps decreases.

Figure 4. Generated samples of Omni-Diffusion on spoken interaction with visual content.

Task Time steps Metrics∗

input and responds in the speech modality. Regarding visual comprehension, our model is able to capture the semantic information of the image and infer the relationships between objects. Collectively, these results illustrate the comprehensive capabilities of our model across various modalities.

256 0.235 / 0.667 50 0.233 / 0.662 10 0.226 / 0.650

Text-to-Image

256 0.225 / 0.645 50 0.229 / 0.649 10 0.231 / 0.648

Speech-to-Image

##### 4.2. Qualitative Results

0.5L 3.07 0.25L 3.74

We present qualitative examples from Omni-Diffusion for the text-to-image and speech-to-image tasks in Fig. 5. More results are provided in Sec. B of the Appendix. The results demonstrate that our model is capable of generating diverse and vivid images with high-quality details. Furthermore, when conditioned on the text and speech with the same context, Omni-Diffusion is able to generate semantic consistent visual content, showcasing its strong cross-modal alignment.

TTS

0.125L 4.83

Sampling efficiency is a key advantage of discrete diffusion models over autoregressive architectures. Unlike classical autoregressive models that generate tokens sequentially, discrete diffusion models can generate multiple tokens in a single forward pass via parallel decoding. In these experiments, we evaluate the sampling efficiency of OmniDiffusion on text-to-image and TTS tasks. Specifically, for text-to-image generation, we initialize the generation process with 256 [MASK] tokens and evaluate the CLIP score under various numbers of time steps. For TTS, we employ adaptive token length assignment to determine the sequence length and set the number of inference steps as a ratio of the

Owing to the mask-token-prediction mechanism of maskbased discrete diffusion models, our model can perform inpainting without additional fine-tuning or introducing inpainting samples into the training data. Specifically, we replace the unknown regions of the input data with [MASK] tokens and leverage our model to generate the masked part to perform inpainting. As shown in Fig. 6, Omni-Diffusion is capable of generating harmonious visual content conditioned

A snow globe showing a small alpine village surrounded by a glacier that suddenly bursts, causing big explosions and spreading fire everywhere.

This image shows misty blue mountains and trees in a watercolor style, with a pacific northwest look.

A scene made in Unreal Engine showing dark skies, snow-covered mountains, and a blizzard.

A desert view with thin Arizona clouds on the horizon, in anime style.

[Figure 40]

[Figure 41]

|[Figure 42]|
|---|

|[Figure 43]|
|---|

Speech-to-ImageText-to-Image

[Figure 44]

[Figure 45]

|[Figure 46]|
|---|

|[Figure 47]|
|---|

Figure 5. Generated samples of Omni-Diffusion on text-to-image and speech-to-image tasks.

[Figure 48]

[Figure 49]

[Figure 50]

A small tree

[Figure 51]

[Figure 52]

[Figure 53]

A small cat

Figure 6. Output samples of Omni-Diffusion on inpainting task.

total [MASK] tokens. We evaluate TTS performance using the WER metric on the LibriTTS benchmark. As shown in Tab. 3, our model maintains strong generation quality on text-to-image generation when the number of time steps is reduced from 50 to as few as 10. Similarly, for TTS task, Omni-Diffusion maintains consistent performance when the number of time steps exceeds 0.25 times the total number of [MASK] tokens. Additionally, Fig. 7 visualizes the images generated by Omni-Diffusion under various numbers of time steps, demonstrating that our method is able to generate high-quality images with extremely few time steps. These results highlight the potential of discrete diffusion models for efficient multimodal comprehension and generation.

Steps 10 Steps 50 Steps 128 Steps 256

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

Figure 7. Generated samples of Omni-Diffusion under various number of time steps, the input prompt is: This image shows misty blue mountains and trees in a watercolor style, with a pacific northwest look.

### 5. Conclusion

In this work, we present Omni-Diffusion, an any-to-any multimodal language model built on mask-based discrete diffusion models. By modeling the joint distribution over multimodal tokens, Omni-Diffusion performs unified comprehension and generation across various modalities. We carefully design training and inference strategies tailored to the discrete diffusion models, which improve not only the training stability but also generation quality. Extensive experiments show that our method achieves performance comparable to or even better than existing AR-based methods. Overall, our research demonstrates the significant potential of diffusion models to serve as foundation models for multimodal AI systems.

### References

- [1] Keyu An, Qian Chen, Chong Deng, Zhihao Du, Changfeng Gao, Zhifu Gao, Yue Gu, Ting He, Hangrui Hu, Kai Hu, et al. Funaudiollm: Voice understanding and generation foundation models for natural interaction between humans and llms. arXiv preprint arXiv:2407.04051, 2024. 4, 12
- [2] Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. In ICLR, 2025. 3
- [3] Huiwen Chang, Han Zhang, Jarred Barber, Aaron Maschinot, Jos´e Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Patrick Murphy, William T. Freeman, Michael Rubinstein, Yuanzhen Li, and Dilip Krishnan. Muse: Text-to-image generation via masked generative transformers. In ICML, 2023. 3
- [4] Guoguo Chen, Shuzhou Chai, Guan-Bo Wang, Jiayu Du, WeiQiang Zhang, Chao Weng, Dan Su, Daniel Povey, Jan Trmal, Junbo Zhang, Mingjie Jin, Sanjeev Khudanpur, Shinji Watanabe, Shuaijiang Zhao, Wei Zou, Xiangang Li, Xuchen Yao, Yongqing Wang, Zhao You, and Zhiyong Yan. GigaSpeech: An evolving, multi-domain ASR corpus with 10, 000 hours of transcribed audio. In Interspeech, 2021. 5, 12
- [5] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 1, 5
- [6] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In NeurIPS,

2023. 6

- [7] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, and Li Yi. Dreamllm: Synergistic multimodal comprehension and creation. In ICLR, 2024. 6
- [8] Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, et al. Cosyvoice: A scalable multilingual zero-shot textto-speech synthesizer based on supervised semantic tokens. arXiv preprint arXiv:2407.05407, 2024. 6
- [9] Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, et al. Cosyvoice 2: Scalable streaming speech synthesis with large language models. arXiv preprint arXiv:2412.10117,

2024. 4, 5

- [10] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Yuhang Dai, Meng Zhao, Yi-Fan Zhang, Shaoqi Dong, Yangze Li, Xiong Wang, et al. Vita: Towards open-source interactive omni multimodal llm. arXiv preprint arXiv:2408.05211, 2024. 1, 2
- [11] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. In NeurIPS, 2025. 6

- [12] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Haoyu Cao, Zuwei Long, Heting Gao, Ke Li, et al. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. arXiv preprint arXiv:2501.01957,

2025. 1, 2

- [13] Daniel Galvez, Greg Diamos, Juan Torres, Keith Achorn, Juan Felipe Cer´on, Anjali Gopi, David Kanter, Max Lam, Mark Mazumder, and Vijay Janapa Reddi. The people’s speech: A large-scale diverse english speech recognition dataset for commercial usage. In NeurIPS, 2021. 12
- [14] Heting Gao, Hang Shao, Xiong Wang, Chaofan Qiu, Yunhang Shen, Siqi Cai, Yuchen Shi, Zihan Xu, Zuwei Long, Yike Zhang, et al. Lucy: Linguistic understanding and control yielding early stage of her. arXiv preprint arXiv:2501.16327,

2025. 12

- [15] Jiaming Han, Kaixiong Gong, Yiyuan Zhang, Jiaqi Wang, Kaipeng Zhang, Dahua Lin, Yu Qiao, Peng Gao, and Xiangyu Yue. Onellm: One framework to align all modalities with language. In CVPR, 2024. 2
- [16] Haorui He, Zengqiang Shang, Chaoren Wang, Xuyuan Li, Yicheng Gu, Hua Hua, Liwei Liu, Chen Yang, Jiaqi Li, Peiyang Shi, et al. Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation. In IEEE SLT, 2024. 12
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 3
- [18] Pengcheng Huang, Shuhao Liu, Zhenghao Liu, Yukun Yan, Shuo Wang, Zulong Chen, and Tong Xiao. Pc-sampler: Position-aware calibration of decoding bias in masked diffusion models. arXiv preprint arXiv:2508.13021, 2025. 5
- [19] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024. 12
- [20] Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790, 2024. 6
- [21] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 4, 12
- [22] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In EMNLP, 2023. 6
- [23] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In ECCV, 2014. 6
- [24] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 2023. 6
- [25] Zuwei Long, Yunhang Shen, Chaoyou Fu, Heting Gao, Lijiang Li, Peixian Chen, Mengdan Zhang, Hang Shao, Jian Li, Jinlong Peng, et al. VITA-Audio: Fast interleaved crossmodal token generation for efficient large speech-language model. arXiv preprint arXiv:2505.03739, 2025. 4

- [26] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In ICML, 2024. 3
- [27] Run Luo, Xiaobo Xia, Lu Wang, Longze Chen, Renke Shan, Jing Luo, Min Yang, and Tat-Seng Chua. Next-omni: Towards any-to-any omnimodal foundation models with discrete flow matching. arXiv preprint arXiv:2510.13721, 2025. 1, 3
- [28] Jinjie Ni, Qian Liu, Longxu Dou, Chao Du, Zili Wang, Hang Yan, Tianyu Pang, and Michael Qizhe Shieh. Diffusion language models are super data learners. arXiv preprint arXiv:2511.03276, 2025. 1
- [29] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: An ASR corpus based on public domain audio books. In ICASSP, 2015. 6, 12
- [30] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In ICML, 2023. 6
- [31] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION5B: an open large-scale dataset for training next generation image-text models. In NeurIPS, 2022. 12
- [32] Guangzhi Sun, Wenyi Yu, Changli Tang, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Yuxuan Wang, and Chao Zhang. video-salmonn: Speech-enhanced audio-visual large language models. In ICML, 2024. 2
- [33] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, Jifeng Dai, Yu Qiao, Limin Wang, and Hongsheng Li. Journeydb: A benchmark for generative image understanding. In NeurIPS, 2023. 12
- [34] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are incontext learners. In CVPR, 2024. 6
- [35] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. In ICLR, 2024. 6
- [36] Changhan Wang, Morgane Rivi`ere, Ann Lee, Anne Wu, Chaitanya Talnikar, Daniel Haziza, Mary Williamson, Juan Miguel Pino, and Emmanuel Dupoux. VoxPopuli: A large-scale multilingual speech corpus for representation learning, semisupervised learning and interpretation. In ACL/IJCNLP, 2021. 12
- [37] Wenbin Wang, Yang Song, and Sanjay Jha. Globe: A high-quality english corpus with global accents for zero-shot speaker adaptive text-to-speech. arXiv preprint arXiv:2406.14875, 2024. 12
- [38] Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dllm v2: Efficient block-diffusion llm. arXiv preprint arXiv:2509.26328, 2025. 1
- [39] Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie.

- Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025. 1
- [40] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal LLM. In ICML,

2024. 1, 3, 6

- [41] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 4
- [42] Zhifei Xie and Changqiao Wu. Mini-omni: Language models can hear, talk while thinking in streaming. arXiv preprint arXiv:2408.16725, 2024. 12
- [43] Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, et al. Lumina-dimoo: An omni diffusion large language model for multi-modal generation and understanding. arXiv preprint arXiv:2510.06308, 2025. 1, 5
- [44] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, and Junyang Lin. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025. 1
- [45] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809,

2025. 3, 4, 5

- [46] Yan Yang, Haochen Tian, Yang Shi, Wulin Xie, Yi-Fan Zhang, Yuhao Dong, Yibo Hu, Liang Wang, Ran He, Caifeng Shan, et al. A survey of unified multimodal understanding and generation: Advances and challenges. Authorea Preprints,

2025. 1

- [47] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487,

2025. 1, 3, 4, 5, 12

- [48] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023. 6
- [49] Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025. 3
- [50] Lijun Yu, Jos´e Lezama, Nitesh Bharadwaj Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, and Lu Jiang. Language model beats diffusion - tokenizer is key to visual generation. In ICLR, 2024. 3, 12

- [51] Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete diffusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025. 1, 3, 5
- [52] Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu. Libritts: A corpus derived from librispeech for text-to-speech. arXiv preprint arXiv:1904.02882, 2019. 6, 12
- [53] Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, and Jie Tang. Glm4-voice: Towards intelligent and human-like end-to-end spoken chatbot. arXiv preprint arXiv:2412.02612, 2024. 4, 6, 12
- [54] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, Hang Yan, Jie Fu, Tao Gui, Tianxiang Sun, Yu-Gang Jiang, and Xipeng Qiu. Anygpt: Unified multimodal LLM with discrete sequence modeling. In ACL, 2024. 1, 3, 6
- [55] Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, et al. Llada 1.5: Variance-reduced preference optimization for large language diffusion models. arXiv preprint arXiv:2505.19223, 2025. 1, 3

### A. Implementation Details

Our model is initialized with the weights of the pre-trained Dream-7B-Instruct [47] discrete diffusion language model. For modality-specific processing, we incorporate MAGViT-v2 [50] for image tokenization, SenseVoiceSmall [1] for speech encoding, and GLM-4-Voice decoder [53] for speech decoding. The training datasets of Omni-Diffusion in the three-stage progressive training pipeline are detailed in Tab. 4. Optimization is performed using AdamW with hyperparameters set to β1 = 0.9, β2 = 0.95, and ϵ = 1e − 8. We use a learning rate of 1e − 4 at stage 1 and stage 2, while the learning rate is reduced to 1e − 5 at stage 3. The maximum sequence length is set to 3072 tokens across all training stages. We set γ = 0.6 for attenuated tail-pad masking. The position penalty parameters γp and NT are set to 0.5 and L − 100 (L denotes sequence length), respectively.

Table 4. Summary of datasets used in Omni-Diffusion.

Modality Task Name Total Number Training Stages Pure Text Text QA Tulu 3 SFT mixture [19] 670K 1,2,3

|Image Caption Laion-2B [31] 10M<br><br>|1,2|
|---|---|
|Visual QA<br><br>LLaVA-OneVisual [21] 820K In-house Dataset 2000K|2,3<br><br>|
|Text-to-Image<br><br>JourneyDB [33] 4000K JourneyDB [33] 4000K<br><br>|1,2,3|

Text-Image

|ASR<br><br>Librispeech [29] 100 Hours Common Voice 17 [29] 100 Hours<br><br>GigaSpeech [4] 1,000 Hours People’s Speech [13] 100 Hours<br><br>VoxPopuli [36] 54 Hours<br><br>|2,3|
|---|---|
|TTS<br><br>LibriTTS [52] 58 Hours GLOBE [37] 50 Hours<br><br>Emilia [16] 5,000 Hours|2,3<br><br>|
|Speech QA<br><br>VoiceAssistant-400K [42] 250K AudioQA-1.0M [14] 180K<br><br>|2,3|

Text-Speech

Spoken Visual QA SDVI 30K

Text-Image-Speech

3

Speech-to-Image SDVI 30K

#### B. Additional Examples of Image Generation We present additional image generation examples from Omni-Diffusion on Fig. 9 and Fig. 8.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

The image shows a landscape background with double exposure glasses of wine, displaying a hyperealistic and detailed view of the subject.

A hyper-realistic, ultra-detailed picture of a Christian country church lit by a bright, snowy atmosphere.

An image of America's mountains mixed with a grand royal male eagle in double exposure, with dripping red and blue colors.

A cute mushroom roof gnome house in a beautiful romantic house placed in a magical forest by the lake, surrounded by charming fairy tale elements, showing a perfectly round full moon and a pearly mood that adds a human touch to the scene.

###### Figure 8. Generated samples of Omni-Diffusion on text-to-image task.

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Text Transcription: A graphic landscape showing a kingdom castle surrounded by a path, covered in sunlight, in the middle of a lush forest.

Text Transcription: A super realistic and hyper-detailed 8k image showing a fantasy night scene with an amazing beach under the full moon, lit by dramatic lighting.

Text Transcription: A background cloud view with an anime style.

Text Transcription: A wide-shot photograph of Santa's workshop in a winter village, made with a photorealistic 3D render.

###### Figure 9. Generated samples of Omni-Diffusion on speech-to-image task.

