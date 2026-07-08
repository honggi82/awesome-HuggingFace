# DreamTalk: When Emotional Talking Head Generation Meets Diffusion Probabilistic Models

Yifeng Ma, Shiwei Zhang, Jiayu Wang, Xiang Wang, Yingya Zhang, Zhidong Deng

Abstract—Emotional talking head generation has attracted growing attention. Previous methods, which are mainly GANbased, still struggle to consistently produce satisfactory results across diverse emotions and cannot conveniently specify personalized emotions. In this work, we leverage powerful diffusion models to address the issue and propose DreamTalk, a framework that employs meticulous design to unlock the potential of diffusion models in generating emotional talking heads. Specifically, DreamTalk consists of three crucial components: a denoising network, a style-aware lip expert, and a style predictor. The diffusion-based denoising network can consistently synthesize high-quality audio-driven face motions across diverse emotions. To enhance lip-motion accuracy and emotional fullness, we introduce a style-aware lip expert that can guide lip-sync while preserving emotion intensity. To more conveniently specify personalized emotions, a diffusion-based style predictor is utilized to predict the personalized emotion directly from the audio, eliminating the need for extra emotion reference. By this means, DreamTalk can consistently generate vivid talking faces across diverse emotions and conveniently specify personalized emotions. Extensive experiments validate DreamTalk’s effectiveness and superiority. The code is available at https://github.com/ali-vilab/dreamtalk.

Speaking style source

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

or

## arXiv:2312.09767v3[cs.CV]10Aug2024

[Figure 6]

audio video

audio

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

DreamTalk

portrait output video

Fig. 1. Leveraging the powerful diffusion models, DreamTalk can consistently generate high-quality talking heads across diverse speaking styles. Furthermore, DreamTalk can conveniently use audio to specify personalized speaking style, obviating the need for additional style references.

personalized emotions [2]–[4]. These personalized emotions are also called speaking styles [3], [5]. A speaking style is defined as facial motion patterns reflected in a talking video clip. In different video clips, the speakers may have different speaking habits and may display various emotions. Therefore, speaking styles are diverse.

Index Terms—Emotional talking head generation, Diffusion models

Existing methods still struggle to 1) consistently produce high-quality results across diverse speaking styles and 2) conveniently specify desired speaking styles. Existing methods [3], [4], [6], [7] are mainly based on GANs [8]. GANs’ inherent issues, such as mode collapse and unstable training, impair their performance across diverse speaking styles. Although these methods achieve satisfactory results for a limited range of speaking styles, they struggle with more diverse styles, especially ones unseen during training, often resulting in diminished emotional intensity, inaccurate lip motion, or sudden facial distortions [3]. Another issue is that to specify speaking styles, previous methods often rely on extra references, such as videos [2], [4], [6] or texts [9]–[11]. Their acquisition requires extra manual effort and hence is inconvenient.

I. INTRODUCTION

Audio-driven talking head generation, which concerns animating portraits with speech audio, has garnered significant interest due to its diverse applications, such as film dubbing, digital human generation, video conferences in band-limited conditions, and online education. To produce realistic talking heads, it is crucial to generate and control emotions. Recognizing that a single type of emotion can be expressed in diverse ways, recent research focus has shifted from modeling coarsegrained, discrete emotions to modeling more fine-grained,

Yifeng Ma and Zhidong Deng are with Department of Computer Science and Technology, BNRist, THUAI, State Key Laboratory of Intelligent Technology and Systems, Tsinghua University, Beijing 100084, China. (e-mail: mayf18@mails.tsinghua.edu.cn; michael@tsinghua.edu.cn).

Shiwei Zhang, Jiayu Wang and Yiyang Zhang are with Alibaba Group, Hangzhou 310023, China. (e-mail: {zhangjin.zsw, wangjiayu.wjy, yingya.zyy}@alibaba-inc.com).

As a new line of generative technique, diffusion models [12], [13] have shown capability to produce high-quality results in numerous generative areas [14]–[18]. The success of diffusion models, stemming from their superior properties such as powerful distribution learning [14], [19], make them exceptionally promising for exploring emotional talking head generation. However, current diffusion-based talking head approaches [20]–[23] primarily concentrate on generating talking heads with neutral expressions or a limited number of discrete emotions, lacking diverse and fine-grained speaking styles. Therefore, exploring the full potential of diffusion models

Xiang Wang are with Huazhong University of Science and Technology,

Wuhan 430074, China. (e-mail: wxiang@hust.edu.cn). Yifeng Ma and Xiang Wang are interns at Alibaba Group. Note: We would like to exclude the preprint titled "Dreamtalk: When expressive talking head generation meets diffusion probabilistic models" [1] as prior art for the purpose of evaluating novelty, potential plagiarism, and self-plagiarism. This is because the preprint and the submitted manuscript are essentially the same article. We modified the preprint’s title, added content, and then submitted it to the journal, but the core subject matter has not changed.

for generating talking heads with diverse speaking styles represents a promising, yet unexplored, research direction.

In this paper, we propose DreamTalk, an emotional talking head generation framework that takes advantage of diffusion models to consistently deliver high performance across diverse speaking styles and reduce the reliance on expensive style references. Specifically, DreamTalk is composed of a denoising network, a style-aware lip expert, and a style predictor. The diffusion-based denoising network produces audiodriven facial motions with the speaking style specified by a reference video. The great distribution-learning property of diffusion models enable the denoising network to consistently produce high-quality results across diverse speaking styles. To enhance the lip-sync, we design a style-aware lip expert that drives the denoising network to produce accurate lip motions under different speaking styles. We observe that previous lip experts, which neglect emotional information, compromise the intensity of generated emotions. To preserve the emotion intensity, we find it important to integrate style information into the lip expert, thereby making it style-aware. Finally, to eliminate the need for additional style references, a diffusion-based style predictor is incorporated to predict personalized speaking styles directly from audio. To predict more personalized emotions, we find it crucial to leverage the correlation between speaker identity and speaking styles; therefore, we provide identity information by incorporating the portrait as input.

The effectiveness of DreamTalk is demonstrated through comprehensive qualitative and quantitative evaluations. DreamTalk can even generate reasonable results for songs in multiple languages, despite these audios being significantly different from those in the training set. In summary, our contributions are as follows:

- • We propose DreamTalk, a diffusion-based framework that can consistently generate talking faces with precise lipsync as well as rich emotions across diverse speaking styles. We find that diffusion models achieve better results than GANs for more diverse speaking styles.
- • We explore how to use audio alone to predict personalized emotions, making it more convenient than relying on extra videos to specify speaking styles. We discover that incorporating identity information significantly enhances prediction accuracy.
- • We propose a style-aware lip expert that can avoid reducing emotion intensity when providing lip guidance. We find that making the lip expert conditioning on speaking style information is crucial for maintaining emotional fullness.
- • Trained in a classifier-free manner, DreamTalk can use the classifier-free guidance scheme to adjust the intensity of arbitrary speaking styles.

II. RELATED WORK

Audio-Driven Talking Head Generation. Audio-driven methods [24]–[28] fall into two main categories: personspecific and person-agnostic. Person-specific approaches [29]– [33] are constrained to generating videos for speakers seen

during training. Many of these [31], [34]–[40] first craft 3D facial animations, later converting them into realistic videos. Recent advancements [41]–[43], [43]–[45] have employed neural radiance fields for modeling, yielding high-fidelity, realistic videos. Conversely, person-agnostic methods [46]– [49] target generating videos for unseen speakers. Early methods prioritized lip synchronization [49]–[54]. Later works shifted focus to natural facial expressions [21], [55] and head poses [56]–[60]. FROND [61] introduces a fine-grained motion model that captures local facial movement keypoints and embeds overall motion context to predict audio-driven facial movements and achieve smooth temporal transitions. However, this method fails to generate emotional expressions during speech, thereby affecting the video’s realism.

Emotional Talking Head Generation. Early methods [7], [10], [31], [32], [62]–[66] model expressions in discrete emotions. To model fine-grained emotions, recent methods [1], [3], [4], [6], [67] use an expression reference video and transfer the expressions from that video to the generated one. However, these GAN-based methods cannot consistently achieve high performance across diverse emotions. Our work addresses these issues by using diffusion models.

UniFaceGAN [68] introduces a temporally consistent facial video editing framework that handles both face swapping and face reenactment simultaneously by using a 3D reconstruction model and a novel temporal loss constraint. FacialPrior-Guided FME Generation [69] enhances facial microexpression generation by utilizing adaptive weighted prior maps and facial priors to guide motion representation. F3AGAN [70] employs a 3D geometric flow, termed facial flow, to represent natural facial motion for continuous image synthesis. Although all these methods are related to facial expression generation, none of them can generate accurate lip shapes driven by audio in different emotional contexts.

Conveniently specifying desired speaking styles is also important. Most previous methods rely on reference videos [3], [4], [6] or text [7], [9], [10], which needs human labor. A more user-friendly approach is to derive speaking styles from the input audio. Previous methods can only infer a limited number of discrete emotion classes from audio [10], [31], [65]. TH-PAD [21] generates expressions only aligned with the audio rhythm, not aligning with the emotional content of the audio. Besides, previous methods neglect information in the input portrait. In this work, we aim to infer personalized emotions using input audio and portraits.

Diffusion Models. Diffusion models [12], [13] have demonstrated strong performance across multiple vision tasks [14], [16], [71]–[74], including text-to-image generation [75], human motion generation [19], and video generation [17], [76]– [79]. Most diffusion-based methods for talking head generation [20], [21], [80]–[89], including EMO [23], AniPortrait [85] and Hallo [90], mainly generate talking heads with neutral emotions and lacks emotional controllability. Besides, the inference of EMO is slow. VASA [22] can only generate a limited number of emotions, lacking diverse, fine-grained speaking styles. In this work, we aim to harness diffusion models for generating and controlling diverse, fine-grained

!! Audio window

" Diffusion step

&

Style ref. video

[Figure 11]

(Pre-trained)

Mouth Embedder

[Figure 12]

[Figure 13]

[Figure 14]

!

[Figure 15]

Style-aware Lip Expert Guidance

| | |
|---|---|
|Cos Simil|ine arity|
| | |

[Figure 16]

Face motion

Mouth vert.

[Figure 17]

[Figure 18]

[Figure 19]

"!"#$

!(%) !(")∗ Predicted motion

Audio Encoder

Style Encoder

Noisy motion

Style ref. video

Style condition

Style code .

[Figure 20]

[Figure 21]

[Figure 22]

Denoising Decoder

Linear Projection

Audio Embedder

##### C

[Figure 23]

concat. add

Audio

(b) Style-aware Lip Expert

(a) Denoising Network "4

-

[Figure 24]

Output frame

[Figure 25]

!('())∗

Style code predicted using audio

!('(/)∗

Audio

[Figure 26]

Diffuse 0 → (' − 1)

Diffuse 0 → (' − 2)

+(")∗

Renderer

Noisy style code

[Figure 27]

[Figure 28]

Diffusion-based Style Predictor

!(")∗

!(")∗

!(")∗

, Portrait

.($)

Style space

Denosing #"

Denosing #"

Denosing #"

[Figure 29]

Constructed using pre-trained style encoder in #"

.(')∗ - ,% &/+(")∗

.(&) - ,% &/+(")∗

- ,% &/+(")∗

0(0, 3)

~

Portrait

(d) Inference Process

(c) Style Predictor

- Fig. 2. Illustration of DreamTalk. A style-aware lip expert (b) is first trained to provide lip motion guidance for the denoising network (a). The denoising network is then trained to predict emotional audio-driven face motions. Then, A style predictor (c) is trained to use audio to predict the style code. During inference (d), the speaking style can be specified using style codes that are extracted from videos or derived from audio.

speaking styles in talking heads, presenting a more intricate challenge.

The denoising network computes face motion conditioned on the speech and style reference video. The face motion M = [ml]Ll=1 is parameterized as a sequence of expression parameters from 3D Morphable Models [91]. The face motion is rendered into video frames by a renderer [92]. The styleaware lip expert provides lip motion guidance under diverse expressions and thus drives the denoising network to achieve accurate lip-sync while preserving emotion fullness. The style predictor can predict the speaking style aligned with that conveyed in speech.

III. METHOD

- A. Problem Formulation

Given a portrait I, a speech A, and a style reference video R, our method aims to generate a talking head video with lip motions synchronized with the speech and the speaking style reflected in the reference video. The audio A = [ai]Li=1 is parameterized as a sequence of acoustic features. R is a sequence of video frames. The head motions in the generated videos can originate from real videos or be produced by existing methods [5], [60].

Besides, to conveniently specify speaking styles, our method also aims to infer the speaking style using solely the speech and the portrait, obviating the need for extra style references. The inferred speaking style can replace the role of style reference videos in controlling expressions (fig. 1), enabling our method to generate personalized emotions with only speech and portrait.

- B. DreamTalk

Denoising Network. The denoising network synthesizes face motion sequence frame-by-frame in a sliding window manner. It predicts a motion frame ml using an audio window Aw = [ai]li+=wl−w, where w denotes the window size.

The denoising network leverages forward and reverse diffusion processes. The diffusion process is modeled as a Markov noising process. Starting from a motion frame m(0), it incrementally introduces Gaussian noise into the real data, gradually diffusing towards a distribution resembling N(0,I). Consequently, the distribution evolves as follows:

### q(m(t)|m(t−1)) = N(√αnm(t−1),(1 − αn)I), (1)

where m(t) is the motion frame sampled at diffusion step t, t ∈ {1,...,T}, and αn is determined by the variance schedules. Conversely, the reverse diffusion process, or the denoising process, predicts the added noise in a noisy motion

As illustrated in fig. 2, DreamTalk comprises 3 key components: a denoising network, a style-aware lip expert, and a style predictor.

frame. Starting from a random motion frame m(T) ∼ N(0,I), the denoising process incrementally removes the noise and

recovers the original motion m(0).

Instead of predicting the noise as formulated by [12], we follow [93] and predict the signal itself. The denoising network Eθ predicts m(0) based on the noisy motion, the diffusion step, the speech context, and the style reference:

m∗(0) = Eθ(m(t),t,Aw,R). (2) The asterisk(∗) indicates quantities that are generated.

Our denoising network has a transformer architecture [94]. The audio window Aw is first fed into a transformer-based audio encoder and the output is concatenated with the noisy

motion m(t) in the channel dimension. After linearly projected to the same dimension, the concatenated results and the timestep t are summed and served as the key and value of a transformer decoder. To extract the speaking style from the style reference, a style encoder first extracts the sequence of 3DMM expression parameters from R and then feeds them into a transformer encoder. The output tokens are aggregated using a self-attention pooling layer [95] to obtain the style code s. The style code is repeated 2w + 1 times and added with positional encodings. The results serve as the query of the transformer decoder. The middle output token of the decoder is fed into a feed-forward network to predict the signal m(0). Style-aware Lip Expert. We observe that using solely the denoising loss in standard diffusion models results in inaccurate lip motions. We conjecture that the loss alone is insufficient for the denoising network to effectively focus on generating precise lip motions. A typical remedy is to involve a pretrained lip expert [54] that provides lip motion guidance. However, we observe the lip expert reduces the intensity of expressions. This stems from the fact that the lip expert merely focuses on a generic speaking style, which leads to generating face motions in a uniform style.

To address this issue, we introduce a style-aware lip expert. The proposed lip expert is trained to evaluate lip-sync under diverse speaking styles. Therefore, it can provide lip motion guidance under diverse speaking styles and strike a better balance between style expressiveness and lip-sync. The lip expert E computes the probability that a clip of audio and lip motions are synchronous conditioned on style reference R:

Psync = E([ai]li+=nl ,[mi]li+=nl ,R), (3) where n denotes the clip length.

The style-aware lip expert encodes the lip motions and audio into respective embeddings conditioned on style reference and then computes the cosine similarity to represent the sync probability. To obtain lip motion information from face motion m, we first convert m into the corresponding face mesh and select vertices in the mouth area as the representation of the lip motion. The lip motion and audio encoders are mainly implemented by MLPs and 1D-convolutions, respectively. The style condition is fused into embeddings by first extracting style features from style reference using a style encoder, which mirrors the architecture of the one in the denoising network but does not share parameters with it, and then concatenating the

style features with intermediate feature maps from embedding encoders.

Style Predictor. Specifically, the style predictor Sϕ predicts the style code s extracted by the style encoder in the trained denoising network. Since we observe that style codes correlate with speaker identity (section IV-D), the style predictor also integrates the portrait as input. The style predictor is instantiated as a diffusion model and is trained to predict the style code itself:

s∗(0) = Sϕ(s(t),t,A,I), (4) where s(t) is the style code sampled at diffusion step t.

The style predictor Sϕ is a transformer encoder on a sequence consisting of, in order: audio embeddings, an embedding for the diffusion timestep, a speaker info embedding, the noised style code embedding, and a final embedding called learned query whose output is used to predict the unnoised style code. Audio embeddings are audio features extracted using self-supervised pre-trained speech models. To obtain the speaker info embedding, our method first extracts the 3DMM identity parameters, which include the face shape information but removes irrelevant information, such as expressions, from the portrait, and then embeds it into a token using an MLP.

Discussion: Advantages over StyleTalk. Although StyleTalk, a GAN-based baseline, also leverages transformer modules, DreamTalk presents notable advantages: 1) StyleTalk’s modules and loss functions are overly complex, which may cause unstable generation results, while DreamTalk’s are simple, making it more extensible and robust. Since GAN’s modecollapse issue hampers modeling diverse speaking styles, to enhance emotion intensity, StyleTalk uses a complex dynamic network and up to six loss terms. As discussed in section IV-B, the overly complex design may cause unstable and inferior results. In contrast, DreamTalk, leveraging the power of diffusion models, does not need complex modules and only uses two loss terms, which is much simpler. 2) StyleTalk makes incorrect assumptions about the data, which may impair the performance. To apply losses that enhance emotion intensity, StyleTalk assumes the speaking styles are consistent in a predefined video group. However, as discussed in section IV-D, these speaking styles are actually varied. DreamTalk does not need such an assumption. 3) StyleTalk can only specify speaking styles using videos, while DreamTalk, leveraging the style predictor, can specify styles only using input audio, which is more convenient.

C. Training and Inference

Training. The style-aware lip expert is first pre-trained by determining whether randomly sampled audio and lip motion clips are synchronous as in [54] and then frozen during training the denoising network. We use cosine-similarity with binary cross-entropy loss during training. Specifically, we compute cosine-similarity for the face motion embedding em and audio embedding ea to represent the probability that the input audiomotion pair is synchronized. The training loss of the lip expert is:

em · ea max(||em||2 · ||ea||2)

). (5)

Lexpert = BCE(

TABLE I QUANTITATIVE COMPARISONS. WE DO NOT RECEIVE GC-AVT SAMPLES ON VOXCELEB2. SA IS ONLY EVALUATED ON MEAD FOR EMOTIONAL METHODS.

|Methods|MEAD / HDTF / Voxceleb2<br><br>SSIM↑ CPBD↑ F-LMD↓ M-LMD↓ Syncconf ↑ SA↑<br><br>|
|---|---|
|MakeItTalk [55] Wav2Lip [54] PC-AVS [59] AVCT [47] GC-AVT [4] EAMM [6] StyleTalk [3] SadTalker [60] PD-FGC [2] EAT [7]<br><br>|0.73/0.57/0.52 0.11/0.24/0.24 3.97/5.12/6.29 5.32/4.55/5.15 2.10/3.16/2.17 0.80/0.63/0.54 0.18/0.30/0.30 2.72/4.53/5.85 4.05/3.60/4.64 5.26/5.83/5.70 0.50/0.42/0.36 0.07/0.13/0.09 5.83/9.71/12.9 4.97/4.17/7.42 2.18/4.85/4.73 -<br><br>0.83/0.76/0.64 0.14/0.22/0.23 2.92/2.86/3.62 5.52/3.57/3.71 2.53/4.27/3.89 0.34/0.36/ - 0.14/0.28/ - 8.04/10.2/ - 7.10/6.23/ - 2.42/4.72/ - 18.7 0.40/0.40/0.43 0.08/0.14/0.20 6.70/7.03/6.36 6.48/6.86/4.89 1.41/2.54/2.24 20.1<br>0.84/0.81/0.66 0.16/0.30/0.29 2.12/1.96/2.92 3.25/2.41/2.96 3.47/4.82/4.51 74.2 0.69/0.77/0.44 0.16/0.24/0.19 4.12/5.99/9.12 4.37/4.07/6.11 2.76/4.35/4.38 0.49/0.41/0.35 0.05/0.13/0.12 5.50/9.50/12.5 4.10/4.23/8.19 2.27/4.68/4.64 43.9 0.53/0.59/0.47 0.15/0.26/0.20 5.54/3.86/5.53 4.79/4.03/5.88 2.16/4.54/4.35 36.4<br>|
|DreamTalk Ground Truth|0.86/0.85/0.69 0.16/0.31/0.30 1.93/1.80/2.69 2.91/2.15/2.72 3.78/5.17/4.90 86.7 1/1/1 0.22/0.31/0.33 0/0/0 0/0/0 4.13/5.44/5.23 92.5<br><br>|

The denoising network Eθ is trained by sampling random tuples (m(0),t,Aw,R) from dataset, corrupting m(0) into m(t) by adding Gaussian noises, executing denoising steps to m(t), and optimizing the loss:

Lnet = λdenoiseLdenoise + λsyncLsync. (6)

Specifically, the ground-truth motion m(0), and the speech audio window Aw are extracted from the training video of the same moment. t is drawn from the uniform distribution U{1,T}. The style reference R is a video clip randomly drawn from the same video containing m(0).

We first compute the denoising loss of the diffusion models [12] defined as:

### Ldenoise = ∥m(0) − Eθ(m(t),t,Aw,R)∥22. (7)

Then, the denoising network maximizes the synchronous probability via a sync loss on generated clips:

Lsync = −log(Psync). (8)

Classifier-free guidance [96] is used to train our model. Specifically, Eθ is trained to learn both the style-conditional and unconditional distributions via randomly setting R = ∅ by 10% chance during training. ∅ is implemented as a sequence of face motions [mi] with all zero values. For inference, the predicted signal is computed by

m∗(0) = ωEθ(m(t),t,Aw,R)

(9)

+ (1 − ω)Eθ(m(t),t,Aw,∅),

instead of eq. (2). This approach enables controlling the effectiveness of the style reference R through adjustment of the scale factor ω.

When training the style predictor, we draw a random video, then extract audio A and style code s(0) (using the trained style encoder) from it. Since 3DMM identity parameters may leak expression information, the portrait I is sampled

from another video with the same speaker identity. The style predictor Eϕ is trained by optimizing the loss:

### Lpred = ∥s(0) − Sϕ(s(t),t,A,I)∥22, (10)

We utilize PIRenderer [92] as the renderer and fine-tune it on emotional dataset to enable it to generate emotional talking faces.

Inference. Our method can specify speaking styles using either reference videos or solely through input audio and portrait. In the case of reference videos, style codes are derived using the style encoder in the denoising network. When relying solely on input audio and portrait, the style predictor takes these inputs and employs a denoising procedure to obtain the style code.

With the style code, the denoising network utilizes the sampling algorithm of DDPM [12] to produce face motions. It first samples a random motion m∗(T) ∼ N(0,I) then computes denoised sequences {m∗(t)},t = T − 1,...,0 by incrementally removing the noise from m∗(t). Finally, the motion m∗(0) is the generated face motion. The sampling process can be accelerated by leveraging DDIM [97]. The output face motions are then rendered into videos by the renderer.

IV. EXPERIMENTS A. Experimental Setup

Datasets. We train and evaluate the denoising network on MEAD [32], HDTF [57], and Voxceleb2 [98]. Since Voxceleb2 official videos are of low resolution, we redownload the original YouTube videos and re-crop the videos. The styleaware lip expert is trained on MEAD and HDTF. We train the style predictor on MEAD and evaluate it on MEAD and RAVEDESS [99].

Baselines. We compare our method with previous methods: MakeitTalk [55], Wav2Lip [54], PC-AVS [59], AVCT [47],

[Figure 30]

[Figure 31]

Portrait Audio Portrait

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Mouth GT

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

MakeItTalk

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Wav2Lip

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

No Style Reference

No Style Reference

PC-AVS

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

AVCT

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

SadTalker

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Discrete Emotion Disgusted

Discrete Emotion Happy

EAT

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

EAMM

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

GC-AVT

Video Reference

Video Reference

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

StyleTalk

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

PD-FGC

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

DreamTalk

- Fig. 3. Qualitative comparisons. The red arrow indicates inaccurate lip motions.

GC-AVT [4], EAMM [6], StyleTalk [3], DiffTalk [20], SadTalker [60], PD-FGC [2], and EAT [7]. DiffTalk’s released model cannot generate reasonable results until submission, so we perform qualitative comparisons using videos from its demo. For other methods, we generate the samples using released models or with authors’ help. When generating samples, we use the audio and the first image from the test video as inputs. We use a segment of the test video as the reference. Except when evaluating the style predictor, the style of DreamTalk is specified by the video.

Distance on the full face (F-LMD) and a newly proposed metric Style Accuracy (SA). SA is the accuracy obtained from classifying samples using a speaking style classifier. When training the classifier, we divide the MEAD dataset into several groups with approximately consistent speaking styles and train the classifier to sort videos into the correct groups. Therefore, if a method generates accurate expressions, its samples will be classified into correct group and hence it will get higher SA. The details of SA metric are reported in Supp. Mat..

B. Main Results

Metrics. To evaluate video quality, we use SSIM [100] and the CPBD [101]. To evaluate lip-motion accuracy, we use the SyncNet confidence score (Syncconf) [102] and the Landmark Distance around mouth area (M-LMD) [53]. To evaluate the accuracy of generated expressions, we use the Landmark

Quanitative Comparisons. As shown in table I, our method outperforms previous methods across most metrics. Wav2Lip’s SyncNet confidence score is higher than ours, even surpassing the ground truth. This is because Wav2Lip is trained using

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Ground Truth

[Figure 164]

[Figure 165]

[Figure 166]

Portrait

StyleTalk

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

DreamTalk

Style Ref.

- Fig. 4. Comparisons with StyleTalk on in-the-wild style reference. StyleTalk fails to generate accurate emotion.

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Ground Truth

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

StyleTalk

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

DreamTalk

Fig. 5. Sudden face distortion (marked in red box) that frequently occurs in StyleTalk’s output. Better viewed in Supp. Video

TABLE II COMPARISONS OF IDENTITY PRESERVATION ON MEAD. DREAMTALK’S SCORE IS COMPETITIVE TO NON-EMOTIONAL METHODS AND IS THE BEST IN EMOTIONAL METHODS.

Method MakeitTalk Wav2Lip PC-AVS AVCT SadTalker GCAVT EAMM StyleTalk PD-FGC EAT TH-PAD Ours Ground Truth CSIM ↑ 0.77 0.89 0.17 0.79 0.78 0.20 0.36 0.64 0.35 0.61 0.30 0.71 0.80

SyncNet as a discriminator. Notably, our method’s SyncNet confidence score closely aligns with the ground truth, and it achieves the best M-LMD scores, which indicates its capability for precise lip synchronization. Furthermore, our superior performance in the F-LMD and SA metrics demonstrates our method’s proficiency in generating facial expressions consistent with the reference speaking style.

Qualitative comparisons. fig. 3 shows the qualitative comparisons. The portraits, style references, and audio are all unseen during training.

StyleTalk, one of the most competitive baselines, fails to consistently generate high-quality results across diverse speaking styles. Firstly, StyleTalk frequently generates videos with sudden facial distortions (fig. 5), making it unsuitable for large-scale practical applications. We do not observe such phenomena in DreamTalk’s results. We speculate that the reason for Styletalk’s unstable generation is that StyleTalk uses overly complex loss functions and is based on GANs, models that suffer from unstable training. Secondly, StyleTalk fails to generate rich emotions, especially for in-the-wild style references. As shown in fig. 3, the output from StyleTalk exhibits discrepancies with the style reference: the left speaker’s eyes are not as narrowed, and the right speaker’s mouth is not opened as widely. The discrepancies are more pronounced when using in-the-wild style references. As shown in fig. 4, Styletalk fails to generate expressions consistent with those in the style reference, including raised eyebrows, glaring eyes, and widened mouths. Initially, we speculate that these issues are caused by the limited training data of StyleTalk, but we observe that when trained only using the same data as StyleTalk, DreamTalk can still generate expressions consistent with the style reference (the result is also shown in fig. 4). Therefore, we speculate that the reason is that GAN’s modecollapse issue impairs the performance across diverse speaking styles. Thirdly, StyleTalk’s lip-sync is inferior. A notable

example is shown in fig. 3: when the speaker utters "m"; the expected closed-mouth motion is replaced by an open mouth in StyleTalk’s output.

It can be seen that MakeItTalk and AVCT struggle with accurate lip synchronization. While Wav2Lip and PC-AVS synchronize lips accurately, their outputs appear blurry. SadTalker, on the other hand, generally aligns lip movements with audio but occasionally displays unnatural jitters. EAT can only generate discrete emotions, lacking the finesse for nuanced expressions. For example, in the left case, the style reference shows the speaker narrowing his eyes, but EAT merely produces a generic disgusted look with wide-open glaring eyes. Additionally, as shown in the right case, EAT struggles to maintain a consistent face shape during speaker head movements. EAMM, GC-AVT, PD-FGC can produce fine-grained emotions. However, EAMM falls short in lip synchronization, GC-AVT and PD-FGC struggle with preserving speaker identity, and all three have issues rendering a plausible background. As shown in fig. 6, DiffTalk struggles with lip synchronization and produces jitteriness and artifacts in the mouth area.

In contrast, DreamTalk excels in producing realistic talking faces that not only mirror the reference speaking style but also achieve precise lip synchronization and superior video quality. Evaluation of the Identity Preservation. To evaluate the ability to preserve identity, we utilize a widely used metric CSIM. When computing the CSIM score, we utilize an offthe-shelf face recognition network ArcFace [103] to extract the deep identity features from each generated frame and then calculate the cosine similarities between the features of the input portrait and the generated frames. We find that when the ID in the style reference and portrait differ significantly, ID preservation worsens. So, we compute scores for StyleTalk and DreamTalk when the style reference comes from a randomly selected ID different from that in portrait. table II shows the

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Ground Truth

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

DiffTalk

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

DreamTalk

Fig. 6. Comparisons with DiffTalk. DiffTalk fails to achieve lip-sync and produces jitteriness.

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

when light party

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

yume hodo anata

Fig. 7. Emotional results generated from songs in multiple languages (English, Chinese, Japanese).

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Emotion Fear

Emotion Angry

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Ground Truth

Generated results

Results on same portrait

Ground Truth

Generated results

Results on same portrait

Portrait

Portrait

Fig. 8. The results of speaking style prediction. The fourth column displays samples generated with predicted styles applied to the same portrait for clearer comparisons.

result on MEAD dataset. Wav2Lip attains the highest score since it merely changes the mouth region and leaves other parts intact. Non-emotional methods achieve better scores than emotional ones because changing emotions may change the identity perceived by humans and ArcFace. DreamTalk’s score is competitive to non-emotional methods and is the best in emotional methods.

Generalization Capabilities. Leveraging the strong power of diffusion models, DreamTalk shows strong generalization capabilities. As shown in fig. 7 and Supp. Video, DreamTalk can even generate reasonable results for songs in multiple languages, even though our training dataset includes only a few multilingual audio and no song audio. Supp. Video shows that DreamTalk also generalizes well to speech in various languages and noisy audio.

Results of Speaking Style Prediction. fig. 8 presents the results of speaking style predictions. The style predictor, utilizing emotional audio and neutral portraits, adeptly deduces personalized speaking styles aligned with those in the original videos. It can discern subtle expressions within the same emotion. For instance, for samples with angry emotion, the first-row speaker exhibits narrowed eyes, in contrast to the second-row speaker’s intense, glaring stare. For samples with fear emotion, the first-row speaker’s eyes and mouth are open, whereas the second-row speaker combines narrowed eyes with a contorted facial expression.

We analyze the influence of portraits on speaking style prediction by predicting speaking styles with an audio clip and

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

Same Audio Portrait A Portrait B Predicted Style A Predicted Style B

Fig. 9. Analyzing the influence of portraits on style prediction. The audio conveys surprised emotion.

different input portraits. The predicted styles are subsequently applied to an identical portrait for a clearer comparison. As shown in fig. 9, the predicted speaking styles match the subtle identity characteristics, such as gender, of the input portraits. The predicted style A generated more feminine results. This validates the necessity of integrating portrait information during style prediction.

We compare the style predictor with TH-PAD [21], a method that also uses audio to predict expressions. We obtain TH-PAD samples with the authors’ help. For comparisons, we use audio to predict speaking styles and use predicted styles to generate samples. We observe that TH-PAD fails to generate emotions conveyed in audio. TH-PAD only generates neutral expressions aligned with the audio rhythm. fig. 10 shows that TH-PAD fails to generate the sad emotion reflected in the audio. We also conduct a quantitative comparison on MEAD. As shown in table III, TH-PAD’s SA and LMD scores, which measure emotion alignment, are inferior.

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

Sad Audio Portrait TH-PAD DreamTalk Ground Truth

Fig. 10. Comparisons with TH-PAD.

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Style reference Portrait w/o Lip Expert Uncond Lip Exp. Full

Fig. 11. Ablation study results of emotional talking head generation.

- TABLE III COMPARISONS WITH TH-PAD ON MEAD. DREAMTALK(A) USES STYLES PREDICTED FROM AUDIO.

|Method<br><br>|F/M-LMD↓ SA↑|
|---|---|
|TH-PAD DreamTalk(A) Ground Truth|6.38/5.81 5.3<br><br>2.24/3.43 78.6 0 / 0 92.5<br><br>|

- TABLE IV THE RESULTS OF DREAMTALK’S ABLATION STUDY ON MEAD. CPBD IS OMITTED DUE TO NO SIGNIFICANT DIFFERENCES.

|Method<br><br>|SSIM↑ F-LMD↓ M-LMD↓ Syncconf↑|
|---|---|
|w/o Lip Expert Uncond Lip Expert Full<br><br>|0.85 1.90 3.07 2.63 0.83 2.19 3.42 4.51<br>0.86 1.93 2.91 3.78<br>|

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Ground Truth Regression w/o Cross-ID training w/o speaker info Full

- Fig. 12. The qualitative results of style predictor’s ablation study.

- C. Ablation Study

Emotional Talking Head Generation. To analyze the contributions of our designs, we conduct an ablation study with two variants: (1) remove the style-aware lip expert (w/o lip expert); (2) trained with unconditional lip expert (uncond lip expert). Our full model is denoted as Full.

fig. 11 and table IV present our ablation study results. The variant w/o lip expert exhibits a decline in lip-sync accuracy on the emotional dataset MEAD, despite its competitive F-LMD score indicating expressive facial generation. Conversely, uncond lip expert secures a superior SyncNet confidence score at the expense of speaking style expressiveness. The Full model achieves a harmonious balance, ensuring both precise lip synchronization and vivid expressions, thanks to the style-aware lip expert directing the diffusion model’s expressive potential.

Speaking Style Prediction. To evaluate the impact of our design choices, we conduct an ablation study with three variants: (1) omitting speaker information and relying solely on audio for prediction (w/o speaker info); (2) during model training, the speaker info and audio are both obtained from the same video (w/o cross-ID training); (3) employing a regression model instead of a diffusion model for prediction (regression).

TABLE V THE ABLATION STUDY RESULTS OF THE STYLE PREDICTOR.

|Method<br><br>|SCD↓ MD↓ SA↑|
|---|---|
|w/o speaker info w/o cross-ID training regression Full<br><br>|0.49 0.28 64.3 0.68 0.45 28.1 0.56 0.32 55.1 0.42 0.23 78.6|

Our full model is denoted as Full. When generating samples for evaluation, the facial images and audio we use are sourced from videos of the same individual expressing different emotions(e. g. the face image is from a happy video while the audio is from an angry one.). This generation approach better aligns with real-world applications.

How to quantitatively evaluate the performance of speaking style prediction has not been explored before. we devise three metrics:

- • Style Code Distance (SCD) We extract the style codes from the videos that provide the audio input and compute the L2 distance between the predicted style codes and these style codes.
- • Motion Distance (MD) We use the predicted style codes and the audio used for prediction to generate face motions and compute the L2 distance between the generated face motions and the face motions extracted from the ground truth videos.
- • Style Accuracy (SA) We use the SA metric mentioned in the section IV-A. SA is evaluated on 3DMM face motions. The ground truth testing set gets 92.5% accuracy.

We refrain from devising image-level metrics, such as training an image classifier for speaking style classification, due to several critical considerations. Firstly, factors in images that are irrelevant to expression, such as the speaker’s identity and background elements, can adversely impact the accurate prediction of nuanced speaking styles. Secondly, inaccuracies introduced by the rendering process may further additionally hinder the accurate discernment of these subtle speaking styles.

The results are shown in table V and fig. 12. The w/o speaker info variant successfully predicts emotions from audio but occasionally fails to maintain consistency between the predicted speaking style and speaker identity, leading to poor identity preservation. This underscores the importance of speaker information in predicting speaking styles. Although in experiments, we observed that w/o cross-ID training achieves slightly better performance than Full when the input portrait and audio are from the same video, it underperformed, often failing to predict the correct emotion, when inputs were from different videos. This suggests that identity 3DMM parameters

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

- Fig. 13. t-SNE visualization of style codes. Left: Style codes from 15 speakers. Each color indicates style codes from an identical speaker. Right: Style codes from a speaker, with darker hues representing increased emotional intensity.

Style reference ! = 0 ! = 0.5 ! = 1 ! = 1.5

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

- Fig. 14. Modulating style intensity by adjusting the scale ω of classifier-free guidance.

TABLE VI MEAN RATINGS OF USER STUDY WITH 95% CONFIDENCE INTERVALS.

|Methods<br><br>|Lip Sync↑ Realness↑ Style Consistency↑|
|---|---|
|MakeItTalk [55] Wav2Lip [54] PC-AVS [59] AVCT [47] GC-AVT [4] EAMM [6] StyleTalk [3] SadTalker [60] PD-FGC [2] EAT [7]<br><br>|2.73 ± 0.08 3.10 ± 0.08 -<br>3.46 ± 0.07 2.45 ± 0.05 3.29 ± 0.04 2.86 ± 0.07 -<br><br><br>3.43 ± 0.07 3.39 ± 0.08 3.40 ± 0.06 2.21 ± 0.05 2.70 ± 0.05<br><br>2.83 ± 0.08 2.46 ± 0.07 2.43 ± 0.06<br>3.42 ± 0.05 3.48 ± 0.07 2.88 ± 0.07<br><br><br>3.44 ± 0.07 3.51 ± 0.06 3.16 ± 0.06 2.80 ± 0.06 3.06 ± 0.07 3.39 ± 0.07 2.87 ± 0.08 2.91 ± 0.07<br>|
|DreamTalk Ground Truth|3.80 ± 0.07 3.77 ± 0.07 3.34 ± 0.08<br><br>4.21 ± 0.08 4.09 ± 0.08 -<br>|

may convey some expression information, and without cross-

- ID training, the model might derive emotional cues from this leaked information rather than the audio. The regression variant struggles to generate accurate expressions for certain data, highlighting the superior distribution-learning capability of diffusion models in facilitating speaking style prediction.
- D. Style Code Visualization

amplifies or attenuates the designated style. When ω = 0, DreamTalk produces a talking head with a neutral expression. We observed a noticeable decline in lip-sync accuracy when the scale factor ω exceeds 2.

We use t-SNE [104] to map style codes from the MEAD dataset’s 15 speakers into a 2D space. Each speaker exhibits 22 distinct speaking styles, comprising seven emotions at three intensity levels, alongside a neutral style. For each style of each speaker, we randomly select 10 videos to extract style codes for visualization.

F. User Study

Emotional Talking Head Generation. We conduct a user study to further evaluate our method. We generate 30 test samples for each method, which cover diverse speaking styles and speakers, and recruit 22 participants to rate samples. Each participant is required to rate all samples (from 1 to 5, 5 is the best) on three aspects: (1) lip sync quality, (2) video realness, and (3) style consistency between the generated videos and the style reference (This metric is only evaluated on emotional methods. Since Ground Truth videos do not express the expressions reflected in the style references. The score for Ground Truth is omitted.). As shown in table VI, our method outperforms baselines across all aspects. A one-way ANOVA and a post-hoc Tukey test identify a significant difference (p < 0.001) between our method and other baselines on all aspects.

The left in fig. 13 shows that style codes first cluster based on identity rather than emotion, indicating the correlation between speaker identity and speaking styles, hence justifying the rationale for using portrait information to infer speaking styles.

The right in fig. 13 shows that even within the same emotion and the same intensity, a speaker’s expressions can vary significantly. The speaker expresses intense sadness in two ways. In one way, the speaker clenches teeth (top left portrait), similar to happy expressions (bottom left portrait), while in another way, the speaker depresses lip corners (bottom right portrait). The observation suggests StyleTalk’s assumption that speaking styles are consistent in videos with the same emotions is incorrect, which may impair performance.

Speaking Style Prediction. In our user study, we evaluate the alignment between the original and predicted speaking styles. Directly assessing the alignment of speaking styles can be somewhat ambiguous, so we employ a comparative approach for evaluation. Specifically, we create a series of video triplets. Each triplet consisted of a test video from our dataset and two generated videos. The first video was generated using a style

E. Modulating Style Intensity

As elaborated in section III-C, the scale factor ω in the classifier-free guidance scheme can modulate the intensity of any input style, even those unseen during training. As shown in fig. 14, the style reference is in-the-wild and adjusting ω either

code predicted from an input portrait, sharing the same speaker identity as in the test video but displaying a neutral emotion, combined with the audio from the test video. The second video is generated using the style code extracted from videos with the same emotion but from a speaker different from the one in the test video. We recruit 20 participants. Each participant is then asked to evaluate 20 triplets and identify which of the generated videos most accurately reflected the speaking style of the test video. The videos generated using predicted style codes are preferred in 75.8% of all ratings. This indicates that the style predictor is able to infer personalized speaking styles that are aligned with the audio.

V. LIMITATIONS

Despite DreamTalk’s promising advancements in emotional talking head generation, it encounters several challenges that open avenues for future research.

When using a constant style reference, DreamTalk generates expressions that are strictly consistent with the main expressions in the reference but lack expression changes over time, such as eye blinking. Generating emotions with rich temporal variations is, compared to methods that generate neutral or coarse-grained emotions, more difficult for methods that achieve fine-grained emotional control, like DreamTalk. This is because these methods must achieve both precise control of expressions and diverse changes. To address the issue, temporal changes in expressions can be achieved by using temporally changing style references when generating each video frame. As discussed in Supp. Mat., DreamTalk can achieve smooth expression changes by changing the style references. Eye blinks can be achieved by using eye blink loss [60] during training or post-editing 3DMM, a common practice used in previous methods [66], [105] that also aim to control expressions. Specifically, we can obtain the parameter changes of blinking and then edit the generated expression parameters. A video with the eye blinking through post-editing is shown in Supp. Video.

DreamTalk may change the speaker’s identity when the identity in the reference video highly differs from that in the portrait. The reason is that 3DMM expression parameters leak identity information. This leakage leads to the generated identity becoming somewhat similar to the identity in the reference. The issue can be mitigated by adopting 3DMM which decouples expression and identity more effectively.

The style predictor sometimes struggles with accurately identifying emotions in low-emotion-intensity audio clips from the MEAD dataset. The reason is that in some MEAD videos, the audio does not correspond with the expressed emotions. To enhance prediction accuracy, it is beneficial to employ a dataset where the audio closely aligns with the expressed emotions. Another solution is to incorporate textual information from audio during prediction, a strategy commonly employed in speech emotion recognition [106].

DreamTalk occasionally produces artifacts around the mouth area, such as teeth flickering, particularly during intense expressions. The issue comes from the renderer and can be mitigated by using more advanced renderers.

Despite these challenges, DreamTalk marks a significant stride in the realm of emotional talking head generation, paving the way for further innovations.

VI. CONCLUSION

In this work, we propose DreamTalk, a novel diffusionbased framework that can consistently generate high-quality talking heads in diverse speaking styles and conveniently use audio to specify personalized emotions. We develop a denoising network for creating emotional, audio-driven facial motions and introduce a style-aware lip expert to optimize lip-sync while maintaining emotion intensity. Additionally, we devise a style predictor that infers speaking styles directly from audio, eliminating the need for video references. Extensive experiments validate the efficacy of DreamTalk. The results demonstrate that employing diffusion models markedly improves the quality of emotional talking head generation.

VII. ETHICAL CONSIDERATION

DreamTalk can generate vivid talking head videos, opening up diverse applications but also posing risks like promoting hatred or depicting violence. Misuse could harm individuals or groups, perpetuate stereotypes, and spread misinformation. To mitigate these risks, we’ve implemented safeguards like watermarks on all outputs and advising against using images without consent. We remain committed to continuous research to minimize adverse societal impacts.

REFERENCES

- [1] Y. Ma, S. Zhang, J. Wang, X. Wang, Y. Zhang, and Z. Deng, “Dreamtalk: When expressive talking head generation meets diffusion probabilistic models,” arXiv preprint arXiv:2312.09767, 2023.
- [2] D. Wang, Y. Deng, Z. Yin, H.-Y. Shum, and B. Wang, “Progressive disentangled representation learning for fine-grained controllable talking head synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17979–17989.
- [3] Y. Ma, S. Wang, Z. Hu, C. Fan, T. Lv, Y. Ding, Z. Deng, and X. Yu, “Styletalk: One-shot talking head generation with controllable speaking styles,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2023.
- [4] B. Liang, Y. Pan, Z. Guo, H. Zhou, Z. Hong, X. Han, J. Han, J. Liu, E. Ding, and J. Wang, “Expressive talking head generation with granular audio-visual control,” in CVPR, 2022, pp. 3387–3396.
- [5] S. Wang, Y. Ma, Y. Ding, Z. Hu, C. Fan, T. Lv, Z. Deng, and X. Yu, “Styletalk++: A unified framework for controlling the speaking styles of talking heads,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
- [6] X. Ji, H. Zhou, K. Wang, Q. Wu, W. Wu, F. Xu, and X. Cao, “Eamm: One-shot emotional talking face via audio-based emotion-aware motion model,” arXiv preprint arXiv:2205.15278, 2022.
- [7] Y. Gan, Z. Yang, X. Yue, L. Sun, and Y. Yang, “Efficient emotional adaptation for audio-driven talking-head generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 22634–22645.
- [8] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” Advances in neural information processing systems, vol. 27, 2014.
- [9] Y. Ma, S. Wang, Y. Ding, B. Ma, T. Lv, C. Fan, Z. Hu, Z. Deng, and X. Yu, “Talkclip: Talking head generation with text-guided expressive speaking styles,” arXiv preprint arXiv:2304.00334, 2023.
- [10] C. Xu, J. Zhu, J. Zhang, Y. Han, W. Chu, Y. Tai, C. Wang, Z. Xie, and Y. Liu, “High-fidelity generalized emotional talking face generation with multi-modal emotion space learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 6609–6619.

- [11] Y. Wang, J. Guo, J. Bai, R. Yu, T. He, X. Tan, X. Sun, and J. Bian, “Instructavatar: Text-guided emotion and motion control for avatar generation,” arXiv preprint arXiv:2405.15758, 2024.
- [12] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 6840–6851.
- [13] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in ICML, 2015.
- [14] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” NeurIPS, 2021.
- [15] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in CVPR, 2022.
- [16] X. Wang, H. Yuan, S. Zhang, D. Chen, J. Wang, Y. Zhang, Y. Shen, D. Zhao, and J. Zhou, “Videocomposer: Compositional video synthesis with motion controllability,” in NeurIPS, 2023.
- [17] U. Singer, A. Polyak, T. Hayes, X. Yin, J. An, S. Zhang, Q. Hu, H. Yang, O. Ashual, O. Gafni et al., “Make-a-video: Text-to-video generation without text-video data,” arXiv preprint arXiv:2209.14792, 2022.
- [18] T. Ao, Z. Zhang, and L. Liu, “Gesturediffuclip: Gesture diffusion model with clip latents,” arXiv preprint arXiv:2303.14613, 2023.
- [19] G. Tevet, S. Raab, B. Gordon, Y. Shafir, D. Cohen-or, and A. H. Bermano, “Human motion diffusion model,” in The Eleventh International Conference on Learning Representations, 2023.
- [20] S. Shen, W. Zhao, Z. Meng, W. Li, Z. Zhu, J. Zhou, and J. Lu, “Difftalk: Crafting diffusion models for generalized talking head synthesis,” arXiv preprint arXiv:2301.03786, 2023.
- [21] Z. Yu, Z. Yin, D. Zhou, D. Wang, F. Wong, and B. Wang, “Talking head generation with probabilistic audio-to-visual diffusion priors,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7645–7655.
- [22] S. Xu, G. Chen, Y.-X. Guo, J. Yang, C. Li, Z. Zang, Y. Zhang, X. Tong, and B. Guo, “Vasa-1: Lifelike audio-driven talking faces generated in real time,” arXiv preprint arXiv:2404.10667, 2024.
- [23] L. Tian, Q. Wang, B. Zhang, and L. Bo, “Emo: Emote portrait alivegenerating expressive portrait videos with audio2video diffusion model under weak conditions,” arXiv preprint arXiv:2402.17485, 2024.
- [24] D. Das, S. Biswas, S. Sinha, and B. Bhowmick, “Speech-driven facial animation using cascaded gans for learning of motion and texture,” in ECCV. Springer, 2020, pp. 408–424.
- [25] J. Guan, Z. Zhang, H. Zhou, T. Hu, K. Wang, D. He, H. Feng, J. Liu, E. Ding, Z. Liu et al., “Stylesync: High-fidelity generalized and personalized lip sync in style-based generator,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1505–1515.
- [26] J. Wang, K. Zhao, S. Zhang, Y. Zhang, Y. Shen, D. Zhao, and J. Zhou, “Lipformer: High-fidelity and generalizable talking face generation with a pre-learned facial codebook,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13844–13853.
- [27] Y. Sun, H. Zhou, K. Wang, Q. Wu, Z. Hong, J. Liu, E. Ding, J. Wang, Z. Liu, and K. Hideki, “Masked lip-sync prediction by audio-visual contextual exploitation in transformers,” in SIGGRAPH Asia 2022 Conference Papers, 2022, pp. 1–9.
- [28] J. Wang, K. Zhao, Y. Ma, S. Zhang, Y. Zhang, Y. Shen, D. Zhao, and J. Zhou, “Facecomposer: A unified model for versatile facial content creation,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023.
- [29] S. Suwajanakorn, S. M. Seitz, and I. Kemelmacher-Shlizerman, “Synthesizing obama: learning lip sync from audio,” ACM Transactions on Graphics (ToG), vol. 36, no. 4, pp. 1–13, 2017.
- [30] O. Fried, A. Tewari, M. Zollhöfer, A. Finkelstein, E. Shechtman, D. B. Goldman, K. Genova, Z. Jin, C. Theobalt, and M. Agrawala, “Textbased editing of talking-head video,” ACM Transactions on Graphics (TOG), vol. 38, no. 4, pp. 1–14, 2019.
- [31] X. Ji, H. Zhou, K. Wang, W. Wu, C. C. Loy, X. Cao, and F. Xu, “Audio-driven emotional video portraits,” in CVPR, 2021, pp. 14080– 14089.
- [32] K. Wang, Q. Wu, L. Song, Z. Yang, W. Wu, C. Qian, R. He, Y. Qiao, and C. C. Loy, “Mead: A large-scale audio-visual dataset for emotional talking-face generation,” in ECCV. Springer, 2020, pp. 700–717.
- [33] Y. Lu, J. Chai, and X. Cao, “Live speech portraits: real-time photorealistic talking-head animation,” ACM Transactions on Graphics (TOG), vol. 40, no. 6, pp. 1–17, 2021.

- [34] R. Yi, Z. Ye, J. Zhang, H. Bao, and Y.-J. Liu, “Audio-driven talking face video generation with learning-based personalized head pose,” arXiv preprint arXiv:2002.10137, 2020.
- [35] J. Thies, M. Elgharib, A. Tewari, C. Theobalt, and M. Nießner, “Neural voice puppetry: Audio-driven facial reenactment,” in ECCV. Springer, 2020, pp. 716–731.
- [36] L. Song, W. Wu, C. Qian, R. He, and C. C. Loy, “Everybody’s talkin’: Let me talk as you want,” arXiv preprint arXiv:2001.05201, 2020.
- [37] A. Lahiri, V. Kwatra, C. Frueh, J. Lewis, and C. Bregler, “Lipsync3d: Data-efficient learning of personalized 3d talking faces from video using pose and lighting normalization,” in CVPR, 2021, pp. 2755–2764.
- [38] C. Zhang, S. Ni, Z. Fan, H. Li, M. Zeng, M. Budagavi, and X. Guo, “3d talking face with personalized pose dynamics,” IEEE Transactions on Visualization and Computer Graphics, 2021.
- [39] C. Zhang, Y. Zhao, Y. Huang, M. Zeng, S. Ni, M. Budagavi, and X. Guo, “Facial: Synthesizing dynamic talking face with implicit attribute learning,” in ICCV, 2021, pp. 3867–3876.
- [40] A. Tang, T. He, X. Tan, J. Ling, R. Li, S. Zhao, L. Song, and J. Bian, “Memories are one-to-many mapping alleviators in talking face generation,” arXiv preprint arXiv:2212.05005, 2022.
- [41] Y. Guo, K. Chen, S. Liang, Y. Liu, H. Bao, and J. Zhang, “Ad-nerf: Audio driven neural radiance fields for talking head synthesis,” arXiv preprint arXiv:2103.11078, 2021.
- [42] X. Liu, Y. Xu, Q. Wu, H. Zhou, W. Wu, and B. Zhou, “Semantic-aware implicit neural audio-driven video portrait generation,” arXiv preprint arXiv:2201.07786, 2022.
- [43] J. Tang, K. Wang, H. Zhou, X. Chen, D. He, T. Hu, J. Liu, G. Zeng, and J. Wang, “Real-time neural radiance talking portrait synthesis via audio-spatial decomposition,” arXiv preprint arXiv:2211.12368, 2022.
- [44] Z. Ye, Z. Jiang, Y. Ren, J. Liu, J. He, and Z. Zhao, “Geneface: Generalized and high-fidelity audio-driven 3d talking face synthesis,” arXiv preprint arXiv:2301.13430, 2023.
- [45] Z. Peng, W. Hu, Y. Shi, X. Zhu, X. Zhang, H. Zhao, J. He, H. Liu, and Z. Fan, “Synctalk: The devil is in the synchronization for talking head synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 666–676.
- [46] J. S. Chung, A. Jamaludin, and A. Zisserman, “You said that?” arXiv preprint arXiv:1705.02966, 2017.
- [47] S. Wang, L. Li, Y. Ding, and X. Yu, “One-shot talking face generation from single-speaker audio-visual correlation learning,” in AAAI, 2022.
- [48] N. Sadoughi and C. Busso, “Speech-driven expressive talking lips with conditional sequential generative adversarial networks,” IEEE Transactions on Affective Computing, vol. 12, no. 4, pp. 1031–1044, 2019.
- [49] K. Vougioukas, S. Petridis, and M. Pantic, “Realistic speech-driven facial animation with gans,” International Journal of Computer Vision, pp. 1–16, 2019.
- [50] Y. Song, J. Zhu, D. Li, X. Wang, and H. Qi, “Talking face generation by conditional recurrent adversarial network,” arXiv preprint arXiv:1804.04786, 2018.
- [51] L. Chen, Z. Li, R. K. Maddox, Z. Duan, and C. Xu, “Lip movements generation at a glance,” in ECCV, 2018, pp. 520–535.
- [52] H. Zhou, Y. Liu, Z. Liu, P. Luo, and X. Wang, “Talking face generation by adversarially disentangled audio-visual representation,” in AAAI, vol. 33, no. 01, 2019, pp. 9299–9306.
- [53] L. Chen, R. K. Maddox, Z. Duan, and C. Xu, “Hierarchical cross-modal talking face generation with dynamic pixel-wise loss,” in CVPR, 2019, pp. 7832–7841.
- [54] K. Prajwal, R. Mukhopadhyay, V. P. Namboodiri, and C. Jawahar, “A lip sync expert is all you need for speech to lip generation in the wild,” in Proceedings of the 28th ACM International Conference on Multimedia, 2020, pp. 484–492.
- [55] Y. Zhou, X. Han, E. Shechtman, J. Echevarria, E. Kalogerakis, and D. Li, “Makelttalk: speaker-aware talking-head animation,” ACM Transactions on Graphics (TOG), vol. 39, no. 6, pp. 1–15, 2020.
- [56] L. Chen, G. Cui, C. Liu, Z. Li, Z. Kou, Y. Xu, and C. Xu, “Talkinghead generation with rhythmic head motion,” in ECCV. Springer,

- 2020, pp. 35–51.

[57] Z. Zhang, L. Li, Y. Ding, and C. Fan, “Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset,” in CVPR,

- 2021, pp. 3661–3670.

- [58] S. Wang, L. Li, Y. Ding, C. Fan, and X. Yu, “Audio2head: Audiodriven one-shot talking-head generation with natural head motion,” IJCAI, 2021.
- [59] H. Zhou, Y. Sun, W. Wu, C. C. Loy, X. Wang, and Z. Liu, “Posecontrollable talking face generation by implicitly modularized audiovisual representation,” in CVPR, 2021, pp. 4176–4186.

- [60] W. Zhang, X. Cun, X. Wang, Y. Zhang, X. Shen, Y. Guo, Y. Shan, and F. Wang, “Sadtalker: Learning realistic 3d motion coefficients for stylized audio-driven single image talking face animation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8652–8661.
- [61] Z. Sheng, L. Nie, M. Liu, Y. Wei, and Z. Gao, “Towards fine-grained talking face generation,” IEEE Transactions on Image Processing, 2023.
- [62] R. Danˇeˇcek, K. Chhatre, S. Tripathi, Y. Wen, M. J. Black, and T. Bolkart, “Emotional speech-driven animation with content-emotion disentanglement,” arXiv preprint arXiv:2306.08990, 2023.
- [63] S. Gururani, A. Mallya, T.-C. Wang, R. Valle, and M.-Y. Liu, “Space: Speech-driven portrait animation with controllable expression,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 20914–20923.
- [64] S. Tan, B. Ji, and Y. Pan, “Emmn: Emotional motion memory network for audio-driven emotional talking face generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 22146–22156.
- [65] S. Sinha, S. Biswas, R. Yadav, and B. Bhowmick, “Emotioncontrollable generalized talking face generation,” arXiv preprint arXiv:2205.01155, 2022.
- [66] Z. Peng, H. Wu, Z. Song, H. Xu, X. Zhu, J. He, H. Liu, and Z. Fan, “Emotalk: Speech-driven emotional disentanglement for 3d face animation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 20687–20697.
- [67] S. Tan, B. Ji, M. Bi, and Y. Pan, “Edtalk: Efficient disentanglement for emotional talking head synthesis,” arXiv preprint arXiv:2404.01647, 2024.
- [68] M. Cao, H. Huang, H. Wang, X. Wang, L. Shen, S. Wang, L. Bao, Z. Li, and J. Luo, “Unifacegan: a unified framework for temporally consistent facial video editing,” IEEE Transactions on Image Processing, vol. 30, pp. 6107–6116, 2021.
- [69] Y. Zhang, X. Xu, Y. Zhao, Y. Wen, Z. Tang, and M. Liu, “Facial prior guided micro-expression generation,” IEEE Transactions on Image Processing, 2023.
- [70] X. Wu, Q. Zhang, Y. Wu, H. Wang, S. Li, L. Sun, and X. Li, “F3a-gan: Facial flow for face animation with generative adversarial networks,” IEEE Transactions on Image Processing, vol. 30, pp. 8658–8670, 2021.
- [71] S. Zhang, J. Wang, Y. Zhang, K. Zhao, H. Yuan, Z. Qin, X. Wang, D. Zhao, and J. Zhou, “I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models,” arXiv preprint arXiv:2311.04145, 2023.
- [72] X. Gao, Y. Yang, Y. Wu, S. Du, and G.-J. Qi, “Multi-condition latent diffusion network for scene-aware neural human motion prediction,” IEEE Transactions on Image Processing, 2024.
- [73] Y. Wang, H. Liu, Y. Feng, Z. Li, X. Wu, and C. Zhu, “Headdiff: Exploring rotation uncertainty with diffusion models for head pose estimation,” IEEE Transactions on Image Processing, 2024.
- [74] S. Welker, H. N. Chapman, and T. Gerkmann, “Driftrec: Adapting diffusion models to blind jpeg restoration,” IEEE Transactions on Image Processing, 2024.
- [75] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, “Dreambooth: Fine tuning text-to-image diffusion models for subjectdriven generation,” arXiv, 2022.
- [76] Z. Qing, S. Zhang, J. Wang, X. Wang, Y. Wei, Y. Zhang, C. Gao, and N. Sang, “Hierarchical spatio-temporal decoupling for text-to-video generation,” arXiv preprint arXiv:2312.04483, 2023.
- [77] Y. Wei, S. Zhang, Z. Qing, H. Yuan, Z. Liu, Y. Liu, Y. Zhang, J. Zhou, and H. Shan, “Dreamvideo: Composing your dream videos with customized subject and motion,” arXiv preprint arXiv:2312.04433, 2023.
- [78] X. Wang, S. Zhang, H. Zhang, Y. Liu, Y. Zhang, C. Gao, and N. Sang, “Videolcm: Video latent consistency model,” arXiv preprint arXiv:2312.09109, 2023.
- [79] J. Wang, H. Yuan, D. Chen, Y. Zhang, X. Wang, and S. Zhang, “Modelscope text-to-video technical report,” arXiv preprint arXiv:2308.06571, 2023.
- [80] M. Stypułkowski, K. Vougioukas, S. He, M. Zie˛ba, S. Petridis, and M. Pantic, “Diffused heads: Diffusion models beat gans on talkingface generation,” arXiv preprint arXiv:2301.03396, 2023.
- [81] D. Bigioi, S. Basak, H. Jordan, R. McDonnell, and P. Corcoran, “Speech driven video editing via an audio-conditioned diffusion model,” arXiv preprint arXiv:2301.04474, 2023.
- [82] S. Mukhopadhyay, S. Suri, R. T. Gadde, and A. Shrivastava, “Diff2lip: Audio conditioned diffusion models for lip-synchronization,” arXiv preprint arXiv:2308.09716, 2023.

- [83] C. Xu, S. Zhu, J. Zhu, T. Huang, J. Zhang, Y. Tai, and Y. Liu, “Multimodal-driven talking face generation, face swapping, diffusion model,” arXiv preprint arXiv:2305.02594, 2023.
- [84] C. Du, Q. Chen, T. He, X. Tan, X. Chen, K. Yu, S. Zhao, and J. Bian, “Dae-talker: High fidelity speech-driven talking face generation with diffusion autoencoder,” arXiv preprint arXiv:2303.17550, 2023.
- [85] H. Wei, Z. Yang, and Z. Wang, “Aniportrait: Audio-driven synthesis of photorealistic portrait animation,” arXiv preprint arXiv:2403.17694, 2024.
- [86] C. Wang, K. Tian, J. Zhang, Y. Guan, F. Luo, F. Shen, Z. Jiang, Q. Gu, X. Han, and W. Yang, “V-express: Conditional dropout for progressive training of portrait video generation,” arXiv preprint arXiv:2406.02511, 2024.
- [87] T. Liu, F. Chen, S. Fan, C. Du, Q. Chen, X. Chen, and K. Yu, “Anitalker: Animate vivid and diverse talking faces through identitydecoupled facial motion encoding,” arXiv preprint arXiv:2405.03121, 2024.
- [88] Z. Chen, J. Cao, Z. Chen, Y. Li, and C. Ma, “Echomimic: Lifelike audio-driven portrait animations through editable landmark conditions,” arXiv preprint arXiv:2407.08136, 2024.
- [89] T. He, J. Guo, R. Yu, Y. Wang, J. Zhu, K. An, L. Li, X. Tan, C. Wang, H. Hu et al., “Gaia: Zero-shot talking avatar generation,” arXiv preprint arXiv:2311.15230, 2023.
- [90] M. Xu, H. Li, Q. Su, H. Shang, L. Zhang, C. Liu, J. Wang, L. Van Gool, Y. Yao, and S. Zhu, “Hallo: Hierarchical audio-driven visual synthesis for portrait image animation,” arXiv preprint arXiv:2406.08801, 2024.
- [91] V. Blanz and T. Vetter, “A morphable model for the synthesis of 3d faces,” in Proceedings of the 26th annual conference on Computer graphics and interactive techniques, 1999, pp. 187–194.
- [92] Y. Ren, G. Li, Y. Chen, T. H. Li, and S. Liu, “Pirenderer: Controllable portrait image generation via semantic neural rendering,” in ICCV, 2021, pp. 13759–13768.
- [93] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, vol. 1, no. 2, p. 3, 2022.
- [94] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017.
- [95] P. Safari, M. India, and J. Hernando, “Self-attention encoding and pooling for speaker recognition,” arXiv preprint arXiv:2008.01077, 2020.
- [96] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.
- [97] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.
- [98] J. S. Chung, A. Nagrani, and A. Zisserman, “Voxceleb2: Deep speaker recognition,” arXiv preprint arXiv:1806.05622, 2018.
- [99] S. R. Livingstone and F. A. Russo, “The ryerson audio-visual database of emotional speech and song (ravdess): A dynamic, multimodal set of facial and vocal expressions in north american english,” PloS one, vol. 13, no. 5, p. e0196391, 2018.
- [100] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.
- [101] N. D. Narvekar and L. J. Karam, “A no-reference perceptual image sharpness metric based on a cumulative probability of blur detection,” in 2009 International Workshop on Quality of Multimedia Experience. IEEE, 2009, pp. 87–91.
- [102] J. S. Chung and A. Zisserman, “Out of time: automated lip sync in the wild,” in Asian conference on computer vision. Springer, 2016, pp. 251–263.
- [103] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “Arcface: Additive angular margin loss for deep face recognition,” in CVPR, 2019, pp. 4690–4699.
- [104] L. Van der Maaten and G. Hinton, “Visualizing data using t-sne.” Journal of machine learning research, vol. 9, no. 11, 2008.
- [105] D. Cudeiro, T. Bolkart, C. Laidlaw, A. Ranjan, and M. J. Black, “Capture, learning, and synthesis of 3d speaking styles,” in CVPR, 2019, pp. 10101–10111.
- [106] S. Wang, Y. Ma, and Y. Ding, “Exploring complementary features in multi-modal speech emotion recognition,” in ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

