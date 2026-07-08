## Diverse and Aligned Audio-to-Video Generation via Text-to-Video Model Adaptation

### Guy Yariv ♡,♣, Itai Gat ♢, Sagie Benaim ♡, Lior Wolf ♠, Idan Schwartz ♠,♣,∗, Yossi Adi ♡,*

♡The Hebrew University of Jerusalem , ♢Technion ♠Tel-Aviv University, ♣NetApp

# arXiv:2309.16429v1[cs.LG]28Sep2023

##### Abstract

We consider the task of generating diverse and realistic videos guided by natural audio samples from a wide variety of semantic classes. For this task, the videos are required to be aligned both globally and temporally with the input audio: globally, the input audio is semantically associated with the entire output video, and temporally, each segment of the input audio is associated with a corresponding segment of that video. We utilize an existing text-conditioned video generation model and a pre-trained audio encoder model. The proposed method is based on a lightweight adaptor network, which learns to map the audio-based representation to the input representation expected by the text-to-video generation model. As such, it also enables video generation conditioned on text, audio, and, for the first time as far as we can ascertain, on both text and audio. We validate our method extensively on three datasets demonstrating significant semantic diversity of audio-video samples and further propose a novel evaluation metric (AV-Align) to assess the alignment of generated videos with input audio samples. AV-Align is based on the detection and comparison of energy peaks in both modalities. In comparison to recent state-of-the-art approaches, our method generates videos that are better aligned with the input sound, both with respect to content and temporal axis. We also show that videos produced by our method present higher visual quality and are more diverse. Code and samples are available at: https://pages.cs.huji.ac.il/adiyoss-lab/TempoTokens.

### Introduction

Neural generative models have changed the way we create and consume digital content. From generating high-quality images and videos (Ho, Jain, and Abbeel 2020; Rombach

- et al. 2022), speech and audio (Wang et al. 2023a; Sheffer and Adi 2023; Copet et al. 2023; Kreuk et al. 2022; Hassid
- et al. 2023), through generating long textual spans (Touvron et al. 2023a,b; Brown et al. 2020), these models have shown impressive results.

In the context of video generation, progress has been more elusive, with recent work making progress in generating short videos conditioned on text (Singer et al. 2022; Ho et al. 2022). Although audio is tightly connected to videos (e.g., providing important cues for motion in a scene), most of the prior work did not consider audio in the generation process.

*Equal Contribution.

[Figure 1]

[Figure 2]

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

[Figure 13]

[Figure 14]

Figure 1: Generated video frames (above) and input audio signal (below the frames) employing our technique. The input to our model is an audio recording from which a representation is extracted. This representation maintains crucial temporal attributes and is then mapped into a textbased latent space representation incorporating both local and global audio context. Subsequently, this latent representation is fed into a pre-trained text-to-video diffusion generative model, ensuring the synchronized generation of video which is closely aligned with the input audio.

For instance, the action of ‘playing drums’ or the ‘motion of waves’ can be distinctively associated with a naturally occurring sound. Moreover, audio is comprised of structural components such as pitch and envelope that provide important cues for the type of scene and motion depicted.

We tackle the problem of generating diverse and realistic videos guided by natural audio samples. Our generated videos capture diverse and real-life settings from a wide variety of semantic classes and are aligned both globally and temporally with the input audio. Globally, the input audio is semantically associated with the entire output video, and temporally, each segment of the input audio is associated with a corresponding segment of that video. An example generation video can be seen in Figure 1.

Prior work on audio-guided video generation was mainly focused on either global information in the videos (i.e., capturing the semantic class) or specific scenes (e.g., speech). Mama et al. (2021); Park et al. (2022); Kumar et al. (2020)

generate talking heads conditioned on speech, but these are limited to videos of human faces and are conditioned on speech and not natural audio. More closely related to our setting, given an input video and an audio sample, Chatterjee and Cherian (2020) generate a continuation of the video that is aligned with the audio. Our method, however, generates videos from audio-only. Ge et al. (2022) proposed a method for generating aligned videos conditioned on audio. While impressive, generated videos are highly limited in diversity. Other works such as (Chen et al. 2017; Hao, Guan, and Zhang 2022; Ruan et al. 2023) generate videos that are globally aligned to the semantic class of the input audio sample (e.g., dancing, drums, etc.) but are unable to generate videos in which every segment is temporally aligned to each segment in the input audio sample.

In contrast to the above methods, our approach enables the generation of diverse and realistic videos associated and aligned with the input audio from a wide variety of semantic classes. Our work utilizes a pre-trained text-conditioned video generation engine and converts the input audio to a sequence of pseudo tokens. Given an input audio sample, we first encode it using an audio encoder, producing a spectral representation of the audio signal. To capture local-to-global information, we construct the representation considering the i-th segment as well as neighboring segments. In particular, we use windows of varying sizes and average the embeddings corresponding to audio segments in these windows. Next, to produce the N-th video frame, we divide the audio embedding into N consecutive segments. We then train an adapter network to map each of these segments to a set of pseudo-tokens. Lastly, to produce the corresponding video, we feed the output of the audio mapping module into the pretrained text-to-video generation model.

Intuitively, we learn a mapping between the audio representation obtained by the pre-trained audio encoder, to the textual tokens’ representation used for conditioning the pretrained text-to-video model. By that, extending the possible video conditioning to audio tokens. To validate our approach, we consider a number of datasets that exhibit a diverse set of videos and input audio samples. We consider the Landscape dataset (Lee et al. 2022), which captures landscape videos. The AudioSet-Drums dataset (Gemmeke et al. 2017) which captures drums videos, and the VGGSound dataset (Chen et al. 2020) which consists of a diverse set of real-world videos from 309 different semantic classes.

We compare our method to state-of-the-art approaches, both in terms of objective evaluation and human study. We evaluate the audio-video alignment as well as video quality and diversity. To capture temporal alignment, we devise a new metric based on detecting energy peaks in both modalities separately and measuring their alignment. Further, we provide an ablation study where we consider alternative approaches to condition the video model.

Our contributions: (i) A state-of-the-art audio-to-video generation model which captures diverse and naturally occurring real-life settings from a wide variety of input videos of different semantic classes; (ii) We present a method that is based on a lightweight adapter, which learns to map audiobased tokens to pseudo-text tokens. As such, it also allows

video generation conditioned on text, audio, or both text and audio. As far as we are aware, our method is the first to enable video generation conditioned both on audio and text; and (iii) Our method can generate natural videos aligned with the input sound, both globally and temporally. To validate this, we present a novel evaluation function to measure audio-video alignment. Since, as far as we can ascertain, we are the first to generate diverse and natural videos guided by audio inputs, such an evaluation function is critical to making progress in the field.

### Related work

Audio-to-image generation. Text-to-image generation has seen great advances recently, using either autoregressive methods (Ramesh et al. 2021; Gafni et al. 2022; Yu et al. 2022) or diffusion based models (Nichol et al. 2022; Ramesh et al. 2022; Saharia et al. 2022; Rombach et al. 2022; Ramesh et al. 2022; Rombach et al. 2022). This inspired a new line of work concerning audio-to-image generation. Zelaszczyk˙ and Ma´ndziuk (2022); Wan, Chuang, and Lee (2019) proposed to generate images based on audio recordings using a GAN. Zelaszczyk˙ and Ma´ndziuk (2022) present results for generating MNIST digits only and did not generalize to general audio sounds, while Wan, Chuang, and Lee (2019) generate images from general audio. In Wav2Clip (Wu et al. 2022b), the authors learn a Contrastive Language-Image Pre-Training (CLIP) (Radford et al. 2021) like a model for learning joint representation for audioimage pairs. Later on, such representation can be used to generate images using VQ-GAN (Esser, Rombach, and Ommer 2021) under the VQ-GAN CLIP (Crowson et al. 2022) framework. The most relevant related work to ours is AudioToken (Yariv et al. 2023), in which the authors learn an audio token while adapting a diffusion-based text-to-image model to generate images using audio inputs.

Text-to-video generation. Early attempts to establish a connection between text and video relied on conditioned retrieval methods (Ali et al. 2022). Later, Wu et al. (2021) introduces the novel integration of 2D VQVAE and sparse attention in text-to-video generation, facilitating the generation of highly realistic scenes. Wu et al. (2022a) extends this method and presents a unified representation for various generation tasks in a multitask learning scheme. Later on, CogVideo (Hong et al. 2022) is built on top of a frozen text-to-image model by adding additional temporal attention modules. Singer et al. (2022) further improves generation quality following a similar modeling paradigm. Video Diffusion Models (He et al. 2022) uses a space-time factorized U-Net with joint image and video data training. Other approaches, such as Villegas et al. (2022) and Villegas et al. (2022) and Yu et al. (2023) proposed transformerbased approaches to generate long videos or for multi-tasklearning. The most relevant prior work to ours is Wang et al. (2023b), which proposed ModelScope. ModelScope is a latent diffusion-based text-to-video generation model with spatiotemporal blocks. By that, ModelScope enables consistent frame generation and smooth movement transitions.

Audio-to-video generation models can be roughly divided into two: (i) speech-to-video generation (talking heads); and

|[Figure 15]|
|---|

|[Figure 16]|
|---|

Audio-conditioned temporal sequence

AudioMapper

.

.

.

|[Figure 17]|
|---|

|MLP|
|---|

.

.

.

Video Generator

. . .

. . .

BEATs

MLP

. . .

. . .

|[Figure 18]|
|---|

.

.

.

Figure 2: An illustration of the proposed model architecture and method. The input audio is first passed through a pre-trained audio encoder model (BEATs). Then, the resulting representations are fed into a trainable MLP layer, establishing a mapping between audio and text tokens. These text-based representations are then used to condition each frame via a temporal audioconditioned sequence. This sequence effectively takes into account both local and global audio segments. Furthermore, an attentive token (a˜atten) is included to learn the identification of significant audio signals using a pooling attention layer. Lastly, the conditioned components are utilized to generate frames through a pre-trained video generator. Notably, optimization is only applied to the MLP within the AudioMapper model and the pooling attention module.

(ii) general audio-to-video. Under the speech-to-video generation, Mama et al. (2021) proposed learning a discrete latent representation of the video signal using VQ-VAE, which will be later modeled via an auto-encoder conditioned on speech spectrogram. Park et al. (2022) generates talking face focusing a piece of phonetic information via Audio-Lip Memory module, while Kumar et al. (2020) proposed a oneshot approach for fast speaker adaptation.

When considering general audio-to-video generation, Chatterjee and Cherian (2020) first proposed a method of generating aligned videos conditioned on both audio and video prompts. Ge et al. (2022) introduced a transformerbased approach for generating videos conditioned on either audio or textual features. Although providing impressive generations, their videos are not diverse and were demonstrated on drum generation only. Chen et al. (2017) suggest using separate frameworks for audio-to-image and imageto-audio generation. Hao, Guan, and Zhang (2022) also suggest modeling both audio-to-image and image-to-audio using bidirectional transformers, however, using a unified framework. The authors prove it is better than two separate ones. Lastly, Ruan et al. (2023), follows the same modeling paradigm, however, using latent diffusion models.

### Method

The proposed method is composed of three main components: (i) an AudioMapper; (ii) multiple audio-conditioned temporal sequences; and (iii) a text-to-video generation module. As our goal in this study is to enrich video generation models using audio inputs, we leverage a pre-trained diffusion-based text-to-video model and augment it with audio conditioning capabilities. A visual description of the proposed method can be seen in Figure 2.

In contrast to converting audio to image, transforming audio to video presents two additional challenges: (i) ensuring the creation of coherent frames and (ii) synchronization be-

tween the audio and video components. For example, consider the scenario of having an audio recording of a dog barking. In the resulting video, it is crucial not only for the dog’s appearance to remain consistent across all frames but also for the match between the timing of the barking sound and the dog’s motion. In this work, we focus on item (ii) by temporally conditioning the generation of each of the video frames by a contextualized representation of the input audio.

Formally, we are interested in the generation of a video, denoted as v = (v(1),...,v(L)), where v(i) ∈ R3×H×W is an output frame, driven by a corresponding audio condition a = (a1,...,aR), where ai ∈ [−1,1] is an audio sample at a given sampling rate in the time domain. We seek to establish a conditional probabilistic model, pθ(v|a), encompassing the entire frame-set, where each frame v(i) is conditioned on a, which denotes the audio condition.

Note that the conditioning of each frame considers the entire audio input but is built differently for each frame. More details can be found in the paragraph on Audio-conditioned temporal sequence.

AudioMapper maps the audio representation obtained from a pre-trained audio encoder to pseudo-tokens compatible with the pre-trained text-to-video model. We denote the output of the AudioMapper as TEMPOTOKENS.

Formally, the model gets as input embedded audio, which originates from a pre-trained audio encoder h : [−1,1]R → RR

′×H×d, where H is the number of layers the representation is collected from, d is the inner dimension of the encoder, and R′ is the segment length that h operates on. To force both audio and video latent representations to have the same dimension, we fix R′ = L by employing a pooling layer. Specifically, we use the BEATs model (Chen et al. 2022) as the audio encoder h. Different layers encapsulate a range of specificity levels. Representations derived from BEATs’ final layers are strongly tied to class-related attributes, whereas earlier layers encompass low-level audio

features (Gat et al. 2022; Adi et al. 2019). We embed an audio segment into a token representation using a non-linear neural network g : RL×H×d → RL×H×d

t:

a˜(i) = g h(a)(i) , (1) where a˜(i) ∈ RL×H×d

t, and dt is the embedding dimension of the text-conditioned tokens of the video generation process. The network g consists of four sequential linear layers with GELU non-linearity between them. We denote a˜(i) as TEMPOTOKENS. Subsequently, we generate a temporal conditioning sequence for each video frame using TEMPOTOKENS. We provide a detailed description of the process in the following paragraph.

Audio-conditioned temporal sequence. Next, to better capture the local context around each video frame, we apply an expanding context window technique over the obtained TEMPOTOKENS. This approach captures the surrounding sound signals of the i-th frame as follows:

c(i) = a ˜max(1,i−j),min(i+j,K) | j = 2k logk=0K , (2) where

- r
- s=l

1 r − l

a˜(s). (3)

a˜l,r =

This context window expands exponentially with increasing temporal distance from the target position, facilitating consideration of a wider local-to-global audio context range. The exponential expansion effectively balances local and global contexts, encompassing important distant audio components that can provide valuable insights into the audio class and close temporal changes needed for audio-video alignment. Figure 3 visually describes the audio-conditioned temporal sequence. Finally, we consider a context window that encompasses all audio signals. We substitute average operation with a trainable attentive pooling layer (Schwartz et al. 2019). Thus,

a˜atten =

L

p(u)˜a(u), (4)

u=1

where p(u)≥0 ∀u is a probability distribution (i.e., L u=1 p(u) = 1) over the audio components. The proba-

bility distribution takes the form:

p(u) ∝ exp(αlθl(u) + αcθc(u)). (5)

The local potential is θl(u) = vl⊤ relu(Vlau), and the cross potential between the audio components is:

⊤ W2a˜(i) ||W2a˜(i)||

L

W1a˜(u) ||W1a˜(u)||

. (6)

θc(u) =

i=1

The trainable parameters are (i) Vl,W1,W2, which reembed the data to tune the attention, (ii) vl ∈ R(L·H·d

t)×1

that scores the sound component (iii) αl,αc that calibrates the local and cross potentials. The attention mechanism enables learning the significance of the audio components.

... ... ... ... ... ...

Figure 3: Illustration of the audio-conditioned temporal sequence for the case of 24 audio components. For the i-th frame, the window sizes grow exponentially, considering local audio details to aid in aligning audio and video, as well as the broader global information that enhances the differentiation of video classes. Additionally, we introduce a token that encompasses all audio components and identifies the significant ones through an attention pooling layer (a˜atten).

Text-to-video. Lastly, we leverage a pre-trained latent diffusion text-to-video model to learn the aforementioned temporal audio tokens, c = {c(i)}Li=1.

Diffusion models are a family of generative models designed to learn the data distribution p(x). This is done by learning the reverse Markov process of length T. Given a timestamp t ∈ [0,1], the denoising function ϵθ : Rd → Rd learns to predict a clean version of the perturbed xt from the training distribution. The generative process can be conditioned on a given input, i.e., modeling p(x|y) where y is a condition vector. In that case, the objective function is LCLDM ≜,

E(v,a)∼S,t∼U(0,1),ϵ∼N(0,I) ∥ϵ − ϵθ(f (vt,c),t)∥22 , (7)

where each video frame, v(i), is conditioned on a dedicated condition vector c(i).

Specifically, in this work, we set ϵθ to be a state-of-theart text-to-video model, ModelScope, which is comprised of a 3D-UNet integrated with a temporal attention layer as outlined in Wang et al. (2023b). ModelScope was trained on ∼10M text-video pairs and ∼2B text-image pairs (Wang et al. 2023b). Notice the proposed framework is not limited to ModelScope and can be used over any differentiable textto-video model.

Model optimization. We optimize the AudioMapper and the attentive pooling layer only and backpropagate gradients through ϵθ while keeping its parameters unchanged. Optimization minimizes the loss LCLDM for reconstructing a frame v(i) conditioned on c(i) (see Equation (7)), with an added weight decay regularization for the encoded TEMPOTOKENS. Overall, we optimize the following loss function:

log L

λl

a˜(u), (8)

1

L = LCLDM +

L

u=1

is a trade-off hyper-parameter between the loss term and the regularization.

where λl

1

### Evaluation metrics

We evaluate our method on three main axes: video quality and diversity, audio-video alignment, and a human study.

Video quality and diversity. We report standard evaluation metrics in the domain of video generation for assessing quality and diversity. We utilize the following metrics: (i) Frechet Video Distance (FVD) metric, which quantifies the visual disparity between feature embeddings extracted from generated and reference videos (Unterthiner et al. 2019) and is used to assess quality and diversity; (ii) Inception Score (IS), which is computed with a trained C3D model (Tran et al. (2015)) on UCF-101 (Soomro, Zamir, and Shah 2012) and assesses video quality.

Audio-video alignment. We distinguish between two types of audio-video alignment: (i) Semantic (or global) alignment, in which the semantic class (e.g., playing drums) of the input audio is depicted by the output video (e.g., a video of people playing drums). To this end, we consider the CLIP Similarity (CLIPSIM) metric (Wu et al. 2021), which gauges the alignment between generated video content and its corresponding audio label; (ii) Temporal alignment, in which we consider if each input audio segment is synchronized with its corresponding generated video segment. To measure this type of alignment, we introduce a novel evaluation metric.

The new metric is based on detecting energy peaks in both modalities separately and measuring their alignment. The premise behind this metric is that fast temporal energy changes in the audio signal often correspond to an object movement producing this sound. For instance, consider an audio waveform of fireworks. A successful audio-video temporal alignment would ensure that the video frames portraying the fireworks exhibit a noticeable change synchronously. Conversely, when the video exhibits a significant change, a corresponding peak should be observed in the audio waveform at that precise moment.

Our audio-video alignment metric operates as follows. We first detect candidate alignment points by considering each modality separately. We detect audio peaks using an Onset Detection algorithm (B¨ock and Widmer 2013), pinpointing instances of heightened auditory intensity. To detect the changes within the video, we calculate the mean of the Optical Flow (Horn and Schunck 1981) magnitude for each frame and identify rapid changes over time. Then, for each peak in one modality, we validate whether a pick was also detected in the other modality within a three-frame temporal window and vise-verse.

Finally, we normalize by the number of peaks to derive the alignment score ranging between zero and one. Such a metric reflects the model’s proficiency in synchronizing audio and video. More formally, given A and V, audio and video peaks were obtained from the onset detection algorithms and optical flow, respectively. The alignment score is defined as:

1 2|A ∪ V| a∈A

AV-Align =

1[v ∈ A] , (9)

1[a ∈ V] +

v∈V

where we consider a valid peak if placed within a window of three frames in the other modality. The above metric can be interpreted as the Intersection-over-Union metric.

To facilitate comprehension, Figure 4 illustrates the alignment process visually, depicting audio peaks and corresponding video changes, emphasizing the interplay between the auditory and visual domains.

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

MagnitudeAmplitude

[Figure 24]

0.5 1 1.5

Time (s)

Figure 4: Audio-Video alignment metric illustration. The first row presents four frames from a generated video featuring a dog. The second row depicts the mean magnitude of optical flow for each frame, capturing video changes. The bottom row shows the amplitude of the audio waveform. The vertical line in the middle and the bottom graphs marks the onset of the waveform, while the peak of video change is also indicated.

Human study. We perform Mean Opinion Scores (MOS) experiments considering both quality and audio-video alignment. In this setup, human raters are presented with several short video samples and are instructed to evaluate their quality and alignment on a scale between 1–5 with increments of 1.0. Specifically, we ask raters to evaluate the videos considering overall quality, global alignment to the audio file, and local alignment between the visual and sound of the video files. We evaluate 20 videos per method and enforce ten raters per sample. The full questionnaire we asked the raters can be found in the supplemental material.

### Experimental setup

Implementation details. The proposed method contains ∼35M trainable parameters. We optimized the model using two A6000 GPUs for 10K iterations. We use AdamW optimizer with learning rate of 1e-05 using constant learning rate scheduler. Each batch comprises 8 videos with 24 frames per video, sampled randomly for one-second granularity. To enhance training efficiency and mitigate memory consumption, we integrated gradient checkpointing into the training process of the 3D U-net architecture. Code and pretrained models will be publicly available upon acceptance.

Datasets. We utilize the VGGSound dataset (Chen et al. 2020), derived from YouTube videos containing ∼180K clips of 10 seconds duration, annotated across 309 classes. To enhance data quality, we filtered ∼60K videos in which audio-video alignment is weak. During this filtering procedure, we utilized a pre-trained audio classifier to categorize sound events present in each clip. Simultaneously, a pretrained image classifier was employed to classify the middle frame of every video clip. We then computed the CLIP (Rad-

Model FVD (↓) CLIPSIM (↑) IS (↑) AV-Align (↑) VGGSound

ModelScope Text2Vid 801 0.69 15.55 0.27 ModelScope Random 1023 0.47 6.32 0.26

Ours 923 0.57 11.04 0.35 AudioSet-Drums

TATS 303 0.69 2.10 0.28 Ours 299 0.70 2.78 0.61

Landscape

MM-Diffusion 922 0.53 2.85 0.41 Ours 784 0.57 4.49 0.54

- Table 1: Automatic video generation results. We report FVD, CLIPSIM, IS, and Alignment (‘align’) scores for both the proposed method (Ours) and the baselines. For a fair comparison, we compare our method to TATS (Ge et al.

2022) and to MM-Diffusion (Ruan et al. 2023) using the benchmarks reported by the authors in the original paper.

ford et al. 2021) score by comparing the predicted labels from both classifiers. Then, filtering is done by removing videos that do not pass a pre-defined threshold.

Additionally, to have a fair comparison with prior work, we experimented with two additional datasets. (i) The Landscape dataset (Lee et al. 2022), which contains 928 nature videos divided into 10-second clips, covering nine distinct scenes; (ii) The AudioSet-Drum dataset (Gemmeke et al. 2017), contains ∼7k videos of drumming. We used the same split as proposed by Ge et al. (2022), where ∼6k is used as the training set while the rest serves as a test set.

Baselines. We compare the proposed method to previous state-of-the-art models generating videos conditioned on audio inputs. Ge et al. (2022) proposed Time Sensitive Transformer (TATS) model, which projects audio latent embeddings onto video embeddings, enabling cross-modal alignment. Ruan et al. (2023) recently proposed MM-Diffusion, which employs coupled denoising auto-encoders to generate joint audio and video content. Each of the above-mentioned baselines, i.e., TATS and MM-Diffusion, were originally evaluated using different benchmarks, i.e., AudioSet-Drums and Landscape, respectively. For a fair comparison, we evaluate the proposed method using each of the datasets suggested in the original papers.

Moreover, we consider two naive baselines based on textto-video models. In the first one, we generate videos from text description and retrieve random audio from the training set which corresponds to the same class as the generated video, denoted as ModelScope Text-To-Video. For the second one, denoted as ModelScope Random, we generate videos unconditionally (i.e., without any specific textual conditions), and match it with a random audio segment. For both baselines, we use the pre-trained publicly available zeroscope-v2 model 1.

1we use the zeroscope-v2 576w as can be found in the following link: https://huggingface.co/cerspense/zeroscope v2 576w

AudioSet-Drum

Landscape

- 0 Semantic Alignment Temporal Alignment Video Quality

- 1

- 2

- 3

- 4

- 5

- 0 Semantic Alignment Temporal Alignment Video Quality

- 1

- 2

- 3

- 4

- 5

TATS Ours

MM-Diffusion

Ours

Scores

Scores

Figure 5: Human study. We consider the MOS score for three metrics: (i). Semantic alignment, where we ask users to rate how well the video matches the input audio semantic label, (ii). Temporal alignment, where we ask users to rate how well each input audio segment is aligned with the generated video segments, and (iii) Video quality, where we ask users to rate the generated video quality. On the LHS, we consider video models trained on AudioSet-Drum, and on the RHS, we consider video models trained on Landscape.

### Results

We start by presenting results for audio-to-video generation considering both objective metrics presented above and human study. Next, we empirically demonstrate how the proposed method can be used to generate videos conditioned on both text and audio modalities, thus enhancing text-to-video generations. Lastly, we conduct an ablation study to understand better the effect of our audio conditioning technique on generation quality and alignment. Visual results are provided in the supplementary.

#### Audio-to-video generation

Objective evaluation. As can be seen in Tab. 1, our method outperforms the baselines on all metrics for the AudioSetDrums and Landscape datasets. Specifically, our method improves both the quality of the generated videos (FVD and IS scores) together with the audio-video alignment (AV-Align and CLIPSIM scores). As expected, the gap between the methods is larger when considering the alignment scores.

Notice the alignment scores changed significantly when considering different benchmarks. Sound events can also be produced by objects not seen in the video; this is especially noticeable in the VGGSound benchmark, in which the AVAlign score of the original videos is 0.51.

Next, we compare our method to the original ModelScope model, both text-condition (ModelScope Text2Vid) and unconditionally (ModelScope Random). As we do not modify the model, we consider the text-condition setup as a top-line in terms of video quality metrics. Recall the audio in both models is retrieved from our training set, using either the video class for ModelScope Text2Vid or randomly ModelScope Random. As expected, our model outperforms ModelScope Random considering all metrics. The ModelScope Text2Vid is superior to our model for video quality. However, when considering audio-video alignment, our method is significantly better.

Cond. FVD (↓) CLIPSIM (↑) IS (↑) AV-Align (↑) Text 801 0.69 15.55 0.27 Audio 923 0.57 11.04 0.35 Text+Audio 859 0.58 11.66 0.36

- Table 2: Results of the proposed method using different modalities as conditioning. We report results for Text, Audio, and Text+Audio modalities as model conditioning.

|[Figure 25]|[Figure 26]|
|---|---|
|[Figure 27]|[Figure 28]|

|[Figure 29]|[Figure 30]|
|---|---|
|[Figure 31]|[Figure 32]|

|[Figure 33]|[Figure 34]|
|---|---|
|[Figure 35]|[Figure 36]|

WaterFire

"A video of <TemporalAudioTokens>, on the moon"

"a video of <TemporalAudioTokens> in Abstract Colors"

"A video of <TemporalAudioTokens>, with vibrant red and orange foliage"

Figure 6: Examples of added text tokens for altering the output video. We show results for fire and flowing water audio.

Human study. We present results using a human study considering both video quality and alignment (both semantic and temporal). Results are depicted in Figure 5. As can be seen for both the AudioSet-Drum and Landscape datasets, users found our videos significantly more temporally aligned. For semantic alignment, our method improves on both TATS and MM-Diffusion, with a significant gap to MM-Diffusion on the Landscape dataset. Finally, on video quality, users found our videos significantly superior.

#### Joint audio-text to video generation

Utilizing text and audio together to guide generation involves adding text tokens for conditioning. In Tab. 2, we show results using “A video of <class>” for text conditioning and “A video of <TemporalAudioAtokens> <class>” for Text+Audio. Combining text and audio conditioning outperforms audio-only in all metrics, especially FVD. Textonly provides the highest video quality but lacks alignment.

In Fig. 6, we present how we merge text tokens to temporal audio tokens, which enables style manipulation. For example, for the sound of a river, we can depict it flowing over the moon by using the prompt “on the moon”.

#### Ablation study

Recall our method consists of using context windows of varying sizes to capture a local-to-global context of the input audio. In Tab. 3, we assess the effect of using different windows of size K ∈ {1,2,3,4} denoted as win. (K-res.). Note in practice, the window size is determined by log K; we use K for readability. Using only the local context window (K = 1) results in a good alignment. As we increase the global context (i.e., increasing K), the video quality is improved while the alignment scores are comparable.

We additionally consider a single audio conditioning vector (vec) by averaging all the audio components. Despite

Cond. FVD (↓) CLIPSIM (↑) IS (↑) AV-Align (↑) vec. 948 0.57 10.12 0.29

- win. (1-res.) 998 0.56 9.22 0.36

- win. (2-res.) 965 0.56 9.87 0.35

- win. (3-res.) 972 0.56 10.01 0.34

- win. (4-res.) 950 0.56 10.13 0.35

- win. (5-res.) 923 0.57 11.04 0.35

Table 3: An ablation study exploring the different audio conditioning. We report FVD, CLIPSIM, IS, and Alignment scores on VGGSound (Chen et al. 2020) considering singlevector conditioning (vec.), time-dependent condition using one window size (win. (1-res.), and different windows of size k (win. (k-res.)).

high video quality scores, the absence of local temporal information results in a notably worse AV-Align score.

### Limitations

As the proposed method leverages a pre-trained text-tovideo model, the adaptation process between text to audio tokens requires mapping between both latent representations. As both modalities operate at different levels of granularity, it is unclear whether such mapping holds all the relevant information in the audio space. Moreover, at the moment, our method generates relatively short video segments since the temporal conditioning is limited to 24 frames due to hardware limitations.

Lastly, while audio can indeed convey information about a visual scene, discrepancies can also arise between the two modalities. For example, a video might depict a dog in a car while the accompanying audio only features a radio playing. This disparity is particularly noticeable in the context of shorter videos. Such mismatch imposes a general limitation of the domain at large, not specifically to our method.

### Conclusion

In this work, we introduced a state-of-the-art audio-to-video generation model capable of generating diverse and realistic videos aligned to input audio samples. By learning a lightweight adapter to map between the input audio representation and a text-based representation, video generation can be conditioned not only on audio but also on text, enabling, for the first time, the generation of audio aligned to both input audio and text samples. To better capture both local and global context around each frame, we consider an expanding context window technique. We validate our approach extensively, demonstrating significant semantic diversity of audio-video samples, and further propose a novel evaluation metric (AV-Align) to assess the temporal alignment of the input audio and generated video. For future work, we are excited to explore how further modalities, such as depth, images, or IMU can be used jointly with audio and text as further conditions by which video can be generated.

### References

Adi, Y.; Zeghidour, N.; Collobert, R.; Usunier, N.; Liptchinsky, V.; and Synnaeve, G. 2019. To reverse the gradient or not: An empirical comparison of adversarial and multi-task learning in speech recognition. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 3742–3746. IEEE.

Ali, A.; Schwartz, I.; Hazan, T.; and Wolf, L. 2022. Video and text matching with conditioned embeddings. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, 1565–1574.

B¨ock, S.; and Widmer, G. 2013. Maximum Filter Vibrato Suppression for Onset Detection. In DAFx-13.

Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell,

- A.; et al. 2020. Language models are few-shot learners. NeurIPS.

Chatterjee, M.; and Cherian, A. 2020. Sound2sight: Generating visual dynamics from sound and context. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVII 16, 701– 719. Springer.

Chen, H.; Xie, W.; Vedaldi, A.; and Zisserman, A. 2020. Vggsound: A large-scale audio-visual dataset. In ICASSP.

Chen, L.; Srivastava, S.; Duan, Z.; and Xu, C. 2017. Deep cross-modal audio-visual generation. In Proceedings of the on Thematic Workshops of ACM Multimedia 2017, 349–357.

Chen, S.; Wu, Y.; Wang, C.; Liu, S.; Tompkins, D.; Chen, Z.; and Wei, F. 2022. Beats: Audio pre-training with acoustic tokenizers. arXiv preprint arXiv:2212.09058.

Copet, J.; Kreuk, F.; Gat, I.; Remez, T.; Kant, D.; Synnaeve, G.; Adi, Y.; and D´efossez, A. 2023. Simple and Controllable Music Generation. arXiv preprint arXiv:2306.05284.

Crowson, K.; Biderman, S.; Kornis, D.; Stander, D.; Hallahan, E.; Castricato, L.; and Raff, E. 2022. Vqgan-clip: Open domain image generation and editing with natural language guidance. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVII, 88–105. Springer.

Esser, P.; Rombach, R.; and Ommer, B. 2021. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12873–12883.

Gafni, O.; Polyak, A.; Ashual, O.; Sheynin, S.; Parikh, D.; and Taigman, Y. 2022. Make-a-scene: Scene-based text-toimage generation with human priors. In European Conference on Computer Vision, 89–106. Springer.

Gat, I.; Lorberbom, G.; Schwartz, I.; and Hazan, T. 2022. Latent space explanation by intervention. In AAAI.

Ge, S.; Hayes, T.; Yang, H.; Yin, X.; Pang, G.; Jacobs, D.; Huang, J.-B.; and Parikh, D. 2022. Long video generation with time-agnostic vqgan and time-sensitive transformer. In European Conference on Computer Vision, 102– 118. Springer.

Gemmeke, J. F.; Ellis, D. P.; Freedman, D.; Jansen, A.; Lawrence, W.; Moore, R. C.; Plakal, M.; and Ritter, M. 2017. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), 776–780. IEEE.

Hao, W.; Guan, H.; and Zhang, Z. 2022. Vag: A uniform model for cross-modal visual-audio mutual generation. IEEE Transactions on Neural Networks and Learning Systems.

Hassid, M.; Remez, T.; Nguyen, T. A.; Gat, I.; Conneau, A.; Kreuk, F.; Copet, J.; Defossez, A.; Synnaeve, G.; Dupoux, E.; et al. 2023. Textually Pretrained Speech Language Models. arXiv preprint arXiv:2305.13009.

He, Y.; Yang, T.; Zhang, Y.; Shan, Y.; and Chen, Q. 2022. Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221.

Ho, J.; Chan, W.; Saharia, C.; Whang, J.; Gao, R.; Gritsenko, A.; Kingma, D. P.; Poole, B.; Norouzi, M.; Fleet, D. J.; et al. 2022. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. NeurIPS.

Hong, W.; Ding, M.; Zheng, W.; Liu, X.; and Tang, J. 2022. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868.

Horn, B. K.; and Schunck, B. G. 1981. Determining optical flow. Artificial Intelligence, 17(1): 185–203.

Kreuk, F.; Synnaeve, G.; Polyak, A.; Singer, U.; D´efossez, A.; Copet, J.; Parikh, D.; Taigman, Y.; and Adi, Y. 2022. Audiogen: Textually guided audio generation. arXiv preprint arXiv:2209.15352.

Kumar, N.; Goel, S.; Narang, A.; and Hasan, M. 2020. Robust one shot audio to video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 770–771.

Lee, S. H.; Oh, G.; Byeon, W.; Kim, C.; Ryoo, W. J.; Yoon, S. H.; Cho, H.; Bae, J.; Kim, J.; and Kim, S. 2022. Soundguided semantic video generation. In European Conference on Computer Vision, 34–50. Springer.

Mama, R.; Tyndel, M. S.; Kadhim, H.; Clifford, C.; and Thurairatnam, R. 2021. NWT: towards natural audio-to-video generation with representation learning. arXiv preprint arXiv:2106.04283.

Nichol, A. Q.; Dhariwal, P.; Ramesh, A.; Shyam, P.; Mishkin, P.; Mcgrew, B.; Sutskever, I.; and Chen, M. 2022. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. In International Conference on Machine Learning, 16784–16804. PMLR.

Park, S. J.; Kim, M.; Hong, J.; Choi, J.; and Ro, Y. M. 2022. Synctalkface: Talking face generation with precise lip-syncing via audio-lip memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, 2062–2070.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In ICML.

Ramesh, A.; Dhariwal, P.; Nichol, A.; Chu, C.; and Chen, M. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125.

Ramesh, A.; Pavlov, M.; Goh, G.; Gray, S.; Voss, C.; Radford, A.; Chen, M.; and Sutskever, I. 2021. Zero-shot text-toimage generation. In International Conference on Machine Learning, 8821–8831. PMLR.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10684– 10695.

Ruan, L.; Ma, Y.; Yang, H.; He, H.; Liu, B.; Fu, J.; Yuan, N. J.; Jin, Q.; and Guo, B. 2023. Mm-diffusion: Learning multi-modal diffusion models for joint audio and video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10219–10228.

Saharia, C.; Chan, W.; Saxena, S.; Li, L.; Whang, J.; Denton, E.; Ghasemipour, S. K. S.; Gontijo-Lopes, R.; Ayan,

- B. K.; Salimans, T.; et al. 2022. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. In NeurIPS.

Schwartz, I.; Yu, S.; Hazan, T.; and Schwing, A. G. 2019. Factor graph attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2039–2048.

Sheffer, R.; and Adi, Y. 2023. I hear your true colors: Image guided audio generation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1–5. IEEE.

Singer, U.; Polyak, A.; Hayes, T.; Yin, X.; An, J.; Zhang, S.; Hu, Q.; Yang, H.; Ashual, O.; Gafni, O.; et al. 2022. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792.

Soomro, K.; Zamir, A. R.; and Shah, M. 2012. UCF101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402.

Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozi`ere, B.; Goyal, N.; Hambro, E.; Azhar, F.; et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.; Babaei, Y.; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale, S.; et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Tran, D.; Bourdev, L.; Fergus, R.; Torresani, L.; and Paluri, M. 2015. Learning Spatiotemporal Features with 3D Convolutional Networks. arXiv:1412.0767.

Unterthiner, T.; van Steenkiste, S.; Kurach, K.; Marinier, R.; Michalski, M.; and Gelly, S. 2019. Towards Accurate Generative Models of Video: A New Metric Challenges. arXiv:1812.01717.

Villegas, R.; Babaeizadeh, M.; Kindermans, P.-J.; Moraldo, H.; Zhang, H.; Saffar, M. T.; Castro, S.; Kunze, J.; and Erhan, D. 2022. Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399.

Wan, C.-H.; Chuang, S.-P.; and Lee, H.-Y. 2019. Towards audio to scene image synthesis using generative adversarial network. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 496–500. IEEE.

Wang, C.; Chen, S.; Wu, Y.; Zhang, Z.; Zhou, L.; Liu, S.; Chen, Z.; Liu, Y.; Wang, H.; Li, J.; et al. 2023a. Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint arXiv:2301.02111.

Wang, J.; Yuan, H.; Chen, D.; Zhang, Y.; Wang, X.; and Zhang, S. 2023b. ModelScope Text-to-Video Technical Report. arXiv:2308.06571.

Wu, C.; Huang, L.; Zhang, Q.; Li, B.; Ji, L.; Yang, F.; Sapiro, G.; and Duan, N. 2021. Godiva: Generating opendomain videos from natural descriptions. arXiv preprint arXiv:2104.14806.

Wu, C.; Liang, J.; Ji, L.; Yang, F.; Fang, Y.; Jiang, D.; and Duan, N. 2022a. N¨uwa: Visual synthesis pre-training for neural visual world creation. In European conference on computer vision, 720–736. Springer.

Wu, H.-H.; Seetharaman, P.; Kumar, K.; and Bello, J. P.

- 2022b. Wav2clip: Learning robust audio representations from clip. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 4563–4567. IEEE. Yariv, G.; Gat, I.; Wolf, L.; Adi, Y.; and Schwartz, I.
- 2023. AudioToken: Adaptation of Text-Conditioned Diffusion Models for Audio-to-Image Generation. arXiv preprint arXiv:2305.13050.

Yu, J.; Xu, Y.; Koh, J. Y.; Luong, T.; Baid, G.; Wang, Z.; Vasudevan, V.; Ku, A.; Yang, Y.; Ayan, B. K.; et al. 2022. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3): 5.

Yu, L.; Cheng, Y.; Sohn, K.; Lezama, J.; Zhang, H.; Chang, H.; Hauptmann, A. G.; Yang, M.-H.; Hao, Y.; Essa, I.; et al. 2023. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10459–10469.

Zelaszczyk,˙ M.; and Ma´ndziuk, J. 2022. Audio-to-image cross-modal generation. In IJCNN.

