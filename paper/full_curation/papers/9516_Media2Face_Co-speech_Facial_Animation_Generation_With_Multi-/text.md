## Media2Face: Co-speech Facial Animation Generation With Multi-Modality Guidance

# arXiv:2401.15687v2[cs.CV]30Jan2024

Qingcheng Zhao1,2* Pengyu Long1,2∗ Qixuan Zhang1,2† Dafei Qin2,3 Han Liang1 Longwen Zhang1,2 Yingliang Zhang4 Jingyi Yu1‡ Lan Xu1‡

1ShanghaiTech University 2Deemos Technology 3University of Hong Kong 4DGene Digital Technology Co., Ltd.

{zhaoqch1, longpy, zhangqx1, lianghan, zhanglw2, yujingyi, xulan1}@shanghaitech.edu.cn qindafei@connect.hku.hk yingliang.zhang@dgene.com https://sites.google.com/view/media2face

[Figure 1]

Figure 1. Given the speech signal and multi-modal conditions (Left), our system generates personalized and stylized co-speech facial animation and head poses (Middle, Right).

### Abstract

The synthesis of 3D facial animations from speech has garnered considerable attention. Due to the scarcity of high-quality 4D facial data and well-annotated abundant multi-modality labels, previous methods often suffer from limited realism and a lack of flexible conditioning. We address this challenge through a trilogy. We first introduce Generalized Neural Parametric Facial Asset (GNPFA), an efficient variational auto-encoder mapping facial geometry and images to a highly generalized expression latent space, decoupling expressions and identities. Then, we utilize GNPFA to extract high-quality expressions and accurate head poses from a large array of videos. This presents the M2F-D dataset, a large, diverse, and scan-level co-speech 3D facial animation dataset with well-annotated emotional and style labels. Finally, we propose Media2Face, a diffusion model in GNPFA latent space for co-speech facial animation generation, accepting rich multi-modality guid-

ances from audio, text, and image. Extensive experiments demonstrate that our model not only achieves high fidelity in facial animation synthesis but also broadens the scope of expressiveness and style adaptability in 3D facial animation.

### 1. Introduction

Advancements in generative AI, powered by large language models, have brought to life virtual companion AI systems, reminiscent of the film “Her”. The core of these systems is to provide realistic and immersive experiences, especially sustained emotional connections with users. To achieve this goal, it is crucial to generate natural facial animation consistent with rich speech content, subtle voice tones, and complicated underlying emotions.

It has been a long journey for our graphics community to generate realistic co-speech facial animation. Early de-

terministic methods [16,18,19] could generate only limited animation variations from audio. Recently, generative models, especially diffusion models, found their way to nondeterministic generation ranging from 2D image [26,52,55] to 3D human motion [1, 3, 39, 59]. Inspired by them, diffusion-based facial animation generation [2,10,56,57,60] has achieved promising results and hence received substantial attention.

We observe two key factors to explore such diffusionbased generation schemes for facial animations. First, the capability of the diffusion models heavily relies on largescale and high-quality training data. However, most of the existing methods [20,56,70] are trained on small-scale datasets such as VOCASET [14] or BIWI [22]. These datasets only cover limited speech states and lack the diversity of emotional variations and character traits. Some recent methods [15,40,46,57,73] attempt to enrich speaking styles with 2D datasets. Yet, they fall short in authentically replicating natural expressions, since the adopted videobased facial trackers often produce sub-optimal expressions or neglect head motions. Second, it is crucial to enable flexible conditioning and disentangled controls, from diverse modalities like speech, style, or emotion. Some concurrent methods offer keyframe [60] (3DiFace), implicit style [46, 57], or emoji-based [15] controls. Yet, faithful conditioning from more diverse multi-modalities like text and image inputs remains an open challenge.

In this paper, we approach the above challenges through a trilogy. First, we introduce General Neural Parametric Facial Asset (GNPFA), a neural representation of fine-grained facial expressions and head poses in latent space. We train GNPFA on a wide array of multi-identity 4D facial scanning data, including high-resolution images and artists’ refined face geometries, dubbed Range of Motion (RoM) data. As such, we decouple nuanced facial expressions from identity in a latent representation that is generalizable to various talking styles. Then, we utilize GNPFA to extract highquality facial expressions and head poses from a diverse range of videos, including different content, styles, emotions, and languages. This results in the creation of the Media2Face Dataset (M2F-D), a diverse 4D dataset annotated with a variety of emotions and styles, with quality comparable to face scans.

Finally, we propose Media2Face, a latent diffusion model for co-speech facial animation generation using the M2F-D dataset. It generates high-quality lip-sync with speech and expresses nuanced human emotions contained in text, images, and even music. Specifically, we train Media2Face in the latent space of GNPFA to recover finegrained facial animations. It takes both audio features extracted by Wav2Vec2 [4] and text/image prompts encoded by CLIP [49] as conditions and generates the sequential facial expressions and head poses in a multi-classifier-

free guidance manner. We conduct extensive experiments and user studies to demonstrate the effectiveness of Media2Face. We further showcase various applications, i.e., generating vivid and realistic facial animations from diverse audio sources like dialogues, music, and speeches, as well as various text/image-based conditioning and editing. To summarize, our main contributions include:

- • We present Media2Face, a diffusion-based generator that integrates diverse media inputs (audio, image, and text) to drive vivid facial animations including head poses.
- • To train Media2Face, We propose GNPFA, a neural latent representation to capture nuanced facial motion details, enabling the collection of a diverse co-speech 4D facial animation dataset with annotated expressions and styles.
- • We conduct extensive experiments and user studies and demonstrate exciting applications for generating facial animations with multi-modality guidances.

### 2. Related Work

#### 2.1. 3D Facial Animation Representations

The quality of generated 3D facial expressions heavily relies on the chosen representation method for 3D facial animation. Over the years, various representation methods have been proposed in the field [5, 18, 35]. One widely adopted approach is the Facial Action Coding System (FACS) [18], which defines facial movements as combinations of muscle activations based on expert knowledge of human anatomy. Traditionally, these activations have been implemented using blendshape deformers [35]. To automate the process and encompass a broader range of facial expressions, [5] proposes 3D Morphable Models (3DMM). These models capture a linear deformation space directly from face scans, encompassing diverse identities and expressions [6,12,29]. For a comprehensive survey of 3DMM-related methods, we refer readers to [17]. Despite their usefulness, linear methods have limitations in capturing subtle nuances of facial expressions. To address this, FLAME [38] introduces pose-dependent corrective blendshapes, as well as articulated jaw, neck, and eyeballs, to enhance the fidelity of facial animation modeling. Further approaches have been developed to model facial expressions in nonlinear spaces [34,54]. [34] employ a Gaussian mixture model to extend the traditional 3DMM, and [54] introduce the use of multi-scale maps to transfer expression wrinkles from a high-resolution example face model to a target model.

Recent work [7, 11, 37, 48, 63–65] leverages deep neural networks to build the latent expression space from data,

achieving state-of-the-art performance on facial animation tasks. [7] incorporates a specialized convolutional operator designed for 3D meshes, capitalizing on the consistent graph structure inherent to deformable shapes with unchanging topology. [48] learn interpretable and editable latent code over high-fidelity facial deformations, extending its application to various mesh topologies. These methods, however, are constrained by the scope of available datasets, often necessitating a compromise between enhancement of quality and retention of diversity. With the advancements in implicit representations, several work [24, 72, 80] have spurred efforts to employ Signed Distance Fields (SDFs) for modeling human head geometry, coupled with a forward deformation field to articulate facial expressions. However, they fall short in comparison to our approach, which constructs a latent space from a head mesh with a fixed topology. Our technique not only aligns better with practical applications but also ensures compatibility with traditional computer graphics (CG) pipelines.

#### 2.2. Conditional Facial Animation Synthesis

Audio-Driven Facial Animation. Recent studies in audio-driven 3D facial animation have leveraged procedural [19,31] and learning-based methods [9,13,20,21,25,28, 30, 50, 62, 81] to map speech to facial movements. While procedural techniques give artists control over lip shapes, they lack flexibility in capturing individual speech styles and require intensive manual tuning [16]. Learning-based approaches, using blendshapes [46, 47, 58, 67, 75] or 3D meshes [14, 20, 45], better capture speech nuances but can lead to overly smooth lip motion and limited upper face expressiveness. To address the complex speech-to-facialexpression mapping, probabilistic models like Vector Quantized Variational Autoencoders (VQ-VAE) [66] have been introduced, predicting facial expression distributions from speech [44, 70, 74]. Despite their strengths, these models often fail to fully represent the stochastic nature of facial expressions. Diffusion models [26,55], recognized for handling intricate generative tasks [23,51,52,71], show promise in audio-driven facial animation [2,10,56,57,60], facilitating multi-modal generation. Yet, integrating speech with head movements remains a challenge, often yielding less convincing expressions. Our method introduces a novel head motion component within a prompt-guided diffusion model, improving the cohesiveness and expressiveness of generated facial animations, thus advancing the state-ofthe-art in speech-driven 3D facial animation.

Style Control. Current approaches predominantly utilize two methods for style modulation: label-based and contextual-based control. Label-based control employs a fixed set of categories within the data to construct labels, such as emotion [15, 32], speaker style [14, 20], and emotional intensity [46]. However, the pre-defined cat-

egories constrain the model to capture complex, nuanced emotions. Contextual-based control allows for the imitation of styles from given segments through a one-shot approach [57] or learning new speaking styles from given motions [61]. Techniques like 3DiFace [60] adapt to the context of provided keyframes, while DiffPoseTalk [57] and Imitator [61] extract or learn style embeddings from given facial animations and style adaptations, respectively. Yet, these methods fall short of quality and explicit controls, often resulting in animations that lack the depth and complexity of genuine human expressions.

### 3. Reshape Facial Animation Data

Realistic synthesis of 3D facial animations necessitates 4D dynamic facial performance capture, typically reliant on multiview-camera setups [69]. This requirement significantly limits the diversity and scalability of data acquisition due to the labor-intensive nature of capturing and processing such data.

To address these constraints, we propose Generalized Neural Parametric Facial Asset (GNPFA), which is in essence a Variational Auto Encoder, mapping facial geometry and video footprints to the same latent space. We train GNPFA on large-scale 4D facial scanning, including highresolution images and artists’ refined geometries, enabling it to produce nuanced facial animation from videos with diverse identities, languages, emotions, and head poses.

#### 3.1. Expression Latent Space Learning

Training data We first capture a dataset with vast multiidentity 4D scanings. We call this the Range of Motion (RoM) data. RoM consists of 43652 registered meshes and 698432 images from 300 identities across different genders, ages, and ethnicities. In addition, to enhance the robustness of expression-identity disentanglement, we create personalized blendshapes for 200 identities under FACS standards according to [36], and generate random artificial expressions during training for augmentation.

Geometry VAE To learn an expression latent space disentangled from identities, we design a geometry VAE consisting of a geometry encoder Egeo and a geometry generator Ggeo, where Ggeo is conditioned on a neutral geometry and utilizes a UNet architecture. To support traditional blendshape animation, we train two mapping networks M and M′, where the former maps the weight of blendshapes, w to our latent space, and the latter does the inverse. The forward process is illustrated in Fig. 2.

Given the input geometry, GR and its paired neutral geometry G¯R, our geometry encoder encodes it to the expression latent code via VAE sampling: zR = Egeo(GR). Then, the geometry decoder recovers the face geometry:

|[Figure 2]<br><br>|[Figure 3]<br><br>[Figure 4]<br><br>Range of Motion<br><br>Dataset<br><br>[Figure 5]<br><br>[Figure 6]<br><br>4D/Expression| |
|---|---|
| | |
<br><br>|[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>ℊ|
|---|
<br><br>[Figure 10]<br><br>|[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>ℊ<br><br>[Figure 14]|
|---|
<br><br>ℊ<br><br>|[Figure 15]| | | | |
|---|---|---|---|---|
|Weights| | | | |
<br><br>[Figure 16]<br><br>[Figure 17]<br><br>ℇ<br><br>[Figure 18]<br><br>[Figure 19]<br><br>ℊ<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>e<br><br>Neutral Geometry<br><br>BlendShape Expression<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>Neutral Geometry<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>ℇ<br><br>Range of Motion Frame<br><br>ℊ<br><br>|e|
|---|
<br><br>[Figure 29]<br><br>| |[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>ℊ|
|---|---|
| | |
<br><br>e<br><br>Neutral Geometry<br><br>[Figure 34]<br><br>Textures<br><br>Differentiable Rendering<br><br>Captured Image<br><br>Stage 1<br><br>Learning Motion Space<br><br>Stage 2<br><br>Learning Vision Encoders<br><br>|Personalized BlendShape<br><br>Dataset<br><br>[Figure 35]<br><br>[Figure 36]|
|---|
<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>ℇ<br><br>[Figure 40]<br><br>ℇ<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>|Multiview Image<br><br>Dataset<br><br>[Figure 44]|
|---|
<br><br>[Figure 45]<br><br>Paired Data<br><br>Expression Latent code<br><br>Pose<br><br>Expression Latent code<br><br>Expression Latent code<br><br>Expression Latent code<br><br>[Figure 46]<br><br>BlendShape<br><br>Weights<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>| | |
|---|---|
| | |
<br><br>ℒ<br><br>ℒ<br><br>ℒ<br><br>ℒ<br><br>ℒ<br><br>[Figure 57]<br><br>[Figure 58]|
|---|

ℊ

ℊ

Differentiable Rendering

Weights

BlendShape

- Figure 2. GNPFA pipeline. (Left:) We train a geometry VAE to learn a latent space of expression and head pose, disentangling expression with identity. (Right:) Two vision encoders are trained to extract expression latent codes and head poses from RGB images, which enables us to capture a wide array of 4D data.

Table 1. 4D datasets comparison. Notice that DiffposeTalk [57] is a combination of reconstructed TFHP [57] and HDTF. EMOTE [15] is trained on reconstructed MEAD.

G˜R = Ggeo(G¯R,zR). The training objective is simply a reconstruction loss:

Lrecon,R = ∥G˜R − GR∥22. (1)

Hours Emotion Head Pose Language

Given a randomly sampled blendshape wB and neutral face G¯B, we obtain the deformed expression GB using personalized blendshapes. Similar to the real data scenario, we extract the expression latent code z˜B = Mgeo(wB) and reconstruct the geometry through the geometry decoder, G˜B = Ggeo(G¯B,zB). We map the expression back by w˜B = M′(˜zB). The training objective is defined as:

VOCASET 0.5 EN BIWI 1.7 EN MultilFace 2.8 EN UUDaMM 9.6 EN DiffposeTalk 26.5 EN EMOTE 25.3 EN M2F-D (Ours) 60.6 6

Lrecon,B = ∥G˜B −GB∥22 +∥z˜B −zB∥22 +∥w˜B −wB∥22. (2)

vision of both real images of RoM data and rendered images from geometries randomly generated by personalized blendshapes.

We use coordinate maps [77] to represent the geometries, which store the 3D position of each vertex on the 2D geometry map in the UV space. This representation can be converted to and from mesh representation using a fixed topology. Besides CG-compatible, it creates a more realistic and believable animation space than existing parametric facial models, due to its non-linearity and vertex-level granularity.

Specifically, given an image IR in the RoM data with corresponding ground-truth geometry GR and neutral G¯R, we extract the expression latent code zˆR = Eexp(IR), and head pose pˆR = Epose(IR). Then, we reconstruct the face geometry using our pretrained decoder, GˆR = Ggeo(G¯R,zˆR). We utilize a differentiable renderer R, to get the rendered image of the face: ˆIR = R(GˆR,pˆR). We define the training loss, Lexp, as the combination of geometry loss and image loss:

#### 3.2. Image Facial Expression Extraction

In addition to the geometry VAE, we train two vision encoders, Eexp and Epose, to extract unified expression latent code and head pose from RGB images. We freeze the geometry VAE and train the vision encoders under the super-

###### Lexp, R = ∥GˆR − GR∥22 + +∥ˆIR − IR∥22. (3)

Similarly, with a randomly sampled geometry GB, we can extract its head pose pˆB = Epose(R(GB,pB)), and expression code zˆB = Eexp(R(GB,pB)). We render the image by the same differentiable renderer ˆIB = R(GˆB,pˆB). Our training objective is defined as:

Lexp, B = ∥GˆB − GB∥22 + ∥R(GˆB,pˆB) − R(GB,pB)∥22.

After training, Eexp and Epose can capture fine-grained expressions and head poses from in-the-wild videos, represented in the expression latent space, and map them to personalized expressions by Ggeo. Owing to the rapid inference speed of GNPFA, we can efficiently extract high-quality and diverse expressions and head poses from in-the-wild videos.

#### 3.3. Latent-based Facial Animation Dataset

We leverage a large collection of online video facial data with abundant audio and text labels, and use GNPFA to extract exact facial expressions and accurate head poses. This allows us to avoid tedious annotations and thus easily augment the limited 4D facial animation dataset, which presents the Media2Face Dataset (M2F-D).

We retrieve both the expression latent codes and head poses from MEAD [68], CREMA-D [8], RAVDESS [41], HDTF [79] and Acappella [43]. The MEAD dataset comprises talking-face videos of 60 actors and actresses expressing 8 different emotions at 3 varying intensity levels. The CREMA-D dataset comprises 7,442 distinct clips featuring 91 actors who delivered 12 sentences expressing 6 different emotions at 4 different intensity levels. The RAVDESS dataset consists of videos by 24 professional actors who vocalize speeches and songs, with a total of 7 and 5 emotions respectively. The HDTF dataset is a collection of high-quality videos and the Acappella dataset encompasses solo singing videos.

To further increase the diversity of talking languages, we collect a 2-hour dataset from in-the-wild videos which contains 6 different languages: Chinese, French, German, Japanese, Russian, and Spanish. To allow for explicit head pose control under different scenarios, we capture a 1.6hour dataset that consists of 14 speakers performing 14 different head movements including speaking, singing, nodding, shaking head, frowning, winking, etc.

Our M2F-D dataset has a total duration of over 60 hours at 30 fps. As shown in Table 1, and surpasses existing audio-visual datasets both in duration and diversity.

### 4. Media2Face Methods

Media2Face is a transformer-based latent diffusion model conditioning on multi-modal driving signals. It models the joint distribution of sequential head poses and facial expressions, i.e., full facial animation, and thus facilitates

the natural synergy of poses and expressions. It also employs multi-conditioning guidance, enabling highly consistent co-speech facial animation synthesis with CLIP-guided stylization and image-based keyframe editing.

#### 4.1. Facial Animation Latent Diffusion Models

As described in Fig. 3, the expression latent code zie and head pose θi are first extracted from each video frame i. Then, we concatenate them to form a single-frame facial animation state, denoted by xi = [zie,θi]. The facial animation is thus formed by a sequence of states X1:N = [xi]Ni=1.

In the diffusion model, generation is modeled as a

Markov denoising process. Here, X1:t N is obtained by adding noise to the ground truth head motion code

X1:0 N over t steps. Our method models the distribution p X1:0 N|X1:t N to facilitate a stepwise denoising process. Similar to the approach in MDM [59], we predict X1:0 N directly. This prediction method allows us to introduce additional regularization terms to improve action consistency and smoothness.

We employ large-scale pre-trained encoders to incorporate multi-modal conditions. The raw speech audio is encoded by the pre-trained Wav2Vec2 [4] and aligned to the length of the facial animation sequence by linear interpolation, resulting in audio feature A1:N. Besides, a text or an image serving as the prompt for talking style is encoded to CLIP latent code P by the pre-trained CLIP model [49]. Our Transformer-based denoiser learns to predict facial animation X1:0 N conditioning on the concatenation of these multi-modal embeddings via the common style-aware cross-attention layers. At each time step, the denoising process can be formulated as:

Xˆ 1:0 N = G X1:t N,t,A1:N,P . (4)

To enable disentanglement of speech and prompt control, during training, two random masks are introduced for multiconditioning classifier-free guidance [27]. Initially, the CLIP latent code P undergoes the first random mask, which brings both stylized and non-stylized co-speech denoisers and enables style control disentangled with speech signals. Then, this masked code is concatenated with the audio feature A1:N. A second phase of random masking is applied to the final concatenated code, which similarly brings both speech-driven and non-speech-driven denoisers and facilitates adjusting speech content consistency strength.

Training We employ the simple loss [26] as the main objective to train our models, which is defined as:

Lsimple = ∥X1:0 N − Xˆ1:0 N∥22. (5)

Besides, we introduce a velocity loss [14] to enforce the model to produce natural transitions between adjacent

|Concatenate<br><br>𝑋  ~ <br><br>[Figure 59]<br><br>Add noise<br><br>|[Figure 60]|
|---|
<br><br>[Figure 61]<br><br>[Figure 62]<br><br>X num of blocks<br><br>𝑋  ~ <br><br>[Figure 63]<br><br>Multi-Head<br><br>Style-Aware<br><br>Attention<br><br>⊕<br><br>K V<br><br>Q<br><br>Diffusion Step t<br><br>[Figure 64]<br><br>Random<br><br>Mask<br><br>⊕<br><br>[Figure 65]<br><br>Multi-Head<br><br>SelfAttention<br><br>Random<br><br>Mask<br><br>[Figure 66]<br><br>[Figure 67]<br><br>ℇ<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>Video Frames<br><br>Audio<br><br>Image/Text<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>Wav2Vec2<br><br>Encoder CLIP<br><br>Encoder<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>PairedData<br><br>Audio Features<br><br>CLIP Latent code<br><br>Alignment Mask<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>Training Stage<br><br>Inference Stage<br><br>[Figure 91]<br><br>|ℊ|
|---|
<br><br>ℊ<br><br>[Figure 92]<br><br>[Figure 93]<br><br>𝑋  ~ <br><br>𝑋  ~ <br><br>𝑋  ~ <br><br>DDIM<br><br>Sampling<br><br>Head Pose<br><br>....<br><br>[Figure 94]<br><br>Expression Latent code<br><br>...<br><br>...<br><br>....<br><br>....<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>𝐿𝑜𝑠𝑠<br><br>0 0 0<br><br>0 0 0<br><br>0 0 0<br><br>0 0 0<br><br>0 0 0<br><br>0 0 0<br><br>⊕<br><br>[Figure 165]<br><br>ℇ|
|---|

SelfAttention

Style-Aware

Multi-Head

Mask Multi-Head

Attention

Sampling

PairedData

DDIM

ℊ

Wav2Vec2

Encoder CLIP

[Figure 166]

Encoder

Random

Random

Mask

[Figure 167]

- Figure 3. Architecture of Media2Face. Our model takes audio features and CLIP latent code as conditions and denoise the noised sequence of expression latent code together with head pose i.e. head motion code. The conditions are randomly masked and subjected to cross-attention with the noisy head motion code. At inference, we sample head motion codes by DDIM. We feed the expression latent code to the GNPFA decoder to extract the expression geometry, combined with a model template to produce facial animation enhanced by head pose parameters.

where sA and sP are two strength factors to adjust speech and style guidance strength respectively. The last two terms provide both non-stylized and stylized predictions within the same speech inputs, which implies the disentangled style control beyond the speech content.

frames, which is formulated as:

2 2

Lvelocity = X2:0 N − X1:0 N−1 − X ˆ2:0 N − Xˆ01:N−1

.

(6)

Furthermore, a smooth loss [57] is employed to enforce smoothness and reduce abrupt transitions and discontinuities:

Overlapped batching denoising To reduce the inference time for real-time applications, we employ batching denoising, a technique akin to the batching denoising step introduced in StreamDiffusion [33], and further extend it to overlapped batching denoising, to process exceedingly long audios in a single denoising pass, by segmenting audio into overlapped windows. The overlapped batching denoising approach transforms the traditionally multiple, autoregressive sequence generation tasks into a parallelizable endeavor. Within the confines of VRAM capacity, its processing time does not increase linearly with the length of the audio, thereby significantly enhancing the speed of head motion generation.

2 2

Lsmooth = X ˆ3:0 N + Xˆ1:0 N−2 − Xˆ2:0 N−1

. (7)

Overall, the denoiser is trained with the following objective: L = λsimpleLsimple + λvelocityLvelocity + λsmoothLsmooth, (8)

where λsimple, λvelocity, and λsmooth are hyper-parameters serving as loss weights to balance the contributions from these terms.

Inference During the denoising process, our model combines two types of guidance, the main speech audio and the additional text/image style guidance, with the classifier-free guidance technique [27]:

#### 4.2. Conditional Facial Animation Editing

Media2Face achieves fine-grained control of generation through keyframe editing and text/image guidance. As illustrated in Fig. 4, we use GNPFA and CLIP to extract the conditions from face images and text/image prompts and leverage classifier-free guidance to control the diffusion process.

Xˆ 01:N = (1 − sA − sP) · G X1:t N,t

+ sA · G X1:t N,t,A1:N + sP · G X1:t N,t,A1:N,P ,

(9)

[Figure 168]

[Figure 169]

[Figure 170]

|Keyframe<br><br>Editing<br><br>[Figure 171]<br><br>[Figure 172]<br><br>ℇ<br><br>Keyframe<br><br>[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]|
|---|

|StyleEditing<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>Sad<br><br>Happy|
|---|

image prompts. Please refer to the supplementary video for more results.

StyleEditing

Keyframe

Editing

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

#### 5.1. Implementation Details

[Figure 183]

[Figure 184]

For GNPFA, we follow Dreamface [76] to design our geometry VAE, vision encoders share the same architecture as the geometry encoder. Training of the geometry VAE and vision encoders take 10 days and 96 hours to converge respectively on Nvidia A6000 GPU, using an AdaBelief optimizer. GNPFA can inference on Nvidia RTX 3090 GPU at about 500 fps. For Media2Face, we employ an eight-layer transformer decoder for the denoiser, utilizing four attention heads. The feature dimension is 512, and the window size is N = 200 at 30 fps. During training, our models follow a cosine noise schedule with 500 noising steps. Media2Face is trained on Nvidia RTX 3090 GPU for 36 hours using an AdamW optimizer. We set λsmooth = 0.01,λvelocity = 1,λsimple = 1. During inference time, we set sA = 2.5, sP = 1.5. We achieved over 300 fps offline and 30fps in real-time on Nvidia RTX 3090 GPU.

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Generated Animation

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

#### 5.2. Comparisons

- Figure 4. Application show case. We can fine-tune the generated facial animation (Row 2) by 1. extracting key-frame expression latent codes through our expression encoder (Row 3), 2. providing per-frame style prompts through CLIP (Row 4, Left: happy, Right: Sad). The intensity and range of control can be adjusted using diffusion in-betweening techniques.

Keyframe editing We can modify the keyframes of the generated animation and smoothly integrate them with the corresponding lip movements using diffusion inpainting technique [42] in the temporal domain. Fig. 4 illustrates an example of modifying a keyframe retrieved from an image using GNPFA. Similarly, this ability can be generalized as sequential composition in [53] to diffuse animations from different sources together. Please refer to our supplementary video.

CLIP-guided style editing Utilizing an in-betweening technique in [3,59], our approach enables the application of diverse style controls across different frames within an audio segment. By assigning distinct style prompts to individual frames and employing a gradient mask during each diffusion step, we seamlessly and naturally integrate the sampling results of various prompts. This methodology ensures a coherent transition of style influences throughout the audio sequence.

- 5. Results

We compare Media2Face with several state-of-the-art facial animation methods. We separate a two-hour segment from the M2F-D as the test set and keep it with a similar data structure as the training set. For 3D methods, we compared with FaceFormer [20], CodeTalker [70], FaceDiffuser [56] and EmoTalk [46] and use their pre-trained model as the baseline. We unify all results to the same FLAME topology for fair comparisons. We also compared the quality of generated head poses with SadTalker [78], a 2D talking face generation method that incorporates head movements.

##### 5.2.1 Quantitative Comparisons

To measure lip synchronization, we employ Lip Vertex Error (LVE) [50], calculating the maximum L2 error across all lip vertices for each frame. Upper Face Dynamics Deviation (FDD) [70] measures the diversity of expressions by comparing the standard deviation of each upper face motion over time between the synthesized and the ground truth. To evaluate the synchronization between audio and generated head pose, we utilize Beat Alignment (BA) [57,78], which computes the synchronization of head pose beats between the generated and the ground truth. As shown in Table 2, our method surpasses existing methods in terms of lip accuracy, facial expression stylization, and the synthesis of rhythmic head movements.

##### 5.2.2 Qualitative Comparisons

In Fig. 8, we showcase multiple audio-driven animations as well as style-based generation results based on text and

We show our qualitative comparison in Fig. 7. Compared with emotion-blind methods, (FaceFormer, CodeTalker,

Table 2. Quantitative comparisons and evaluations. Notice that the BA metric is not utilized for FaceFormer, CodeTalker, FaceDiffuser, and EmoTalk, as they do not generate head poses. Also, metrics related to vertices are not utilized for SadTalker due to its different facial topology.

Methods LVE(mm)↓ FDD(×10−5m)↓ BA ↑

FaceFormer 18.19 21.37 N/A CodeTalker 16.74 21.95 N/A

FaceDiffuser 16.33 22.38 N/A EmoTalk 14.61 17.84 N/A SadTalker — — 0.219

Ours w/o CFG 10.67 16.69 0.166 Ours w/o GNPFA 14.89 12.81 0.198 Ours w/ 10% data 10.75 20.65 0.170 Ours w/ 40% data 10.55 18.32 0.208 Ours w/ 70% data 10.43 14.98 0.221

Ours 10.44 12.21 0.254

FaceDiffuser), our method can generate not only more accurate lip movement but also micro-expressions under neutral conditions (eye blink, eyebrow gesture). Compared with emotion-aware methods, (EMOTE, EmoTalk), our method demonstrates a more vibrant and natural expression of emotions and facial details while maintaining lip shape accuracy. Notice that our method also generates head poses highly synchronized with the given conditions(raise the head in surprise and lower it in sadness).

#### 5.3. Ablation study

We conduct following ablation experiments to evaluate our key components:(1) Ours w/o GNPFA: We train Media2Face on linear blendshapes obtained from the GNPFA mapping network M′. (2) Ours w/o CFG: We inference Media2Face without classifier-free guidance. As shown in Table 2, the removal of GNPFA leads to a significant degradation in LVE, validating the effectiveness of GNPFA on modeling accurate lip shape. Inference without CFG has bad performance on FDD since the model fails to generate stylized head motions. We also train Media2Face on 10%, 40%, 70% of M2F-D. As shown in Table 2, model performance on FDD and BA increases during dataset scaling up while that on LVE remains steady. This validates our hypothesis that while the model can learn precise lipsync animation on small datasets, it requires learning from a large amount of rich-conditioned data to generate animation with realistic expressions, diverse emotions, and appropriate head movements.

#### 5.4. User Study

We conduct 30 diverse audio samples, including dialogues, speeches, and songs and invert 100 participants. We ensure fair comparisons by employing the same shader

and template for all generated geometries. Participants ensure side-by-side animations with other methods, assessing Media2Face with three conditions: with a specific style prompt for each audio, with a neutral prompt, and without any prompts and head pose animation. Our model demonstrates superior preference ratings: over 90% for general cases, 80% without specific style prompts, and 70% in the absence of specific style prompts and head pose, underscoring the effectiveness of head pose generation and our style prompt.

### 6. Conclusions

In this paper, we present Media2Face, pushing the boundary of diffusion models for realistic co-speech facial animation synthesis with rich multi-modal conditionings.

To enhance the diffusion models with high-quality facial animation data, we introduce GNPFA, a facial VAE with a latent neural representation of facial expressions and head poses, pre-trained on a wide array of facial scanning data. GNPFA is then utilized to extract high-quality expressions and head poses from a mass of accessible facial videos from various resources. It brings M2F-D, a large, diverse, and scan-level 3D facial animation dataset with abundant speech, emotion, and style annotations, over 60 hours. Finally, we train our Media2Face model in GNPFA latent space with M2F-D dataset. Media2Face integrates diverse media inputs as conditions including audio, text, and image, which flexibly control facial emotion and style while preserving high-quality lip-sync with speech. The experimental results demonstrate the effectiveness of Media2Face and showcase various related applications, such as reconstructing dialogue situations and multi-modality conditional editing. We believe Media2Face is a significant step towards realizing realistic human-centric AI virtual companions with strong emotional connection and resonance with us humans.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

- Figure 5. User study result. Note how our method has demonstrated overwhelming superiority in the singing cases, showcasing the model’s ability to generate rich emotions and rhythmic head movements.

|[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]<br><br>[Figure 211]<br><br>[Figure 212]<br><br>[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>[Figure 218]<br><br>[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]<br><br>[Figure 222]<br><br>NeutralExpression2Expression1<br><br>[Figure 223]<br><br>[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]<br><br>[Figure 228]<br><br>[Figure 229]<br><br>[Figure 230]<br><br>Identity 1 Identity 2 Identity 3 Identity 4|
|---|

NeutralExpression1

- Figure 6. Retargeting to various identities. Thanks to GNPFA, we can further generate personalized and nuanced facial mesh, which can fit various identities across different genders, ages, and ethnicities. Note the differences in facial details among different identities, notably the different wrinkles.

|[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]<br><br>[Figure 238]<br><br>Jazz<br><br>OursFaceFormerCodeTalkerFaceDiffuser<br><br>[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]<br><br>[Figure 242]<br><br>[Figure 243]<br><br>[Figure 244]<br><br>[Figure 245]<br><br>[Figure 246]<br><br>Club What No See<br><br>[Figure 247]<br><br>[Figure 248]<br><br>[Figure 249]<br><br>[Figure 250]<br><br>[Figure 251]<br><br>[Figure 252]<br><br>[Figure 253]<br><br>[Figure 254]<br><br>[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]<br><br>[Figure 258]<br><br>[Figure 259]<br><br>[Figure 260]<br><br>[Figure 261]<br><br>[Figure 262]<br><br>[Figure 263]<br><br>[Figure 264]<br><br>[Figure 265]<br><br>[Figure 266]<br><br>[Figure 267]<br><br>[Figure 268]<br><br>[Figure 269]<br><br>[Figure 270]<br><br>[Figure 271]<br><br>[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]<br><br>[Figure 276]<br><br>Hum<br><br>EMOTEOursEmoTalk<br><br>[Figure 277]<br><br>[Figure 278]<br><br>[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]<br><br>[Figure 282]<br><br>Kind Really Save<br><br>[Figure 283]<br><br>[Figure 284]<br><br>[Figure 285]<br><br>[Figure 286]<br><br>[Figure 287]<br><br>[Figure 288]<br><br>[Figure 289]<br><br>[Figure 290]<br><br>[Figure 291]<br><br>[Figure 292]<br><br>[Figure 293]<br><br>[Figure 294]<br><br>|knowing smile|
|---|
<br><br>|pleasant surprise|
|---|
<br><br>|dejected|
|---|
<br><br>|irritated|
|---|
<br><br>[Happy] [Surprise] [Sad] [Anger]|
|---|

OursFaceFormerodeTalkerEmoTalk

Figure 7. Qualitative comparison. Top: Comparing with emotion-blind methods, we use a neutral prompt to feed to Media2Face. Bottom: Comparing with emotion-aware methods. We utilize text prompts for Media2Face and assign corresponding emotion labels to EMOTE. Notice that EmoTalk extracts emotional features from audio that cannot be manually assigned.

||Arguing with his beloved, he felt deeply sad and unjustly wounded.|
|---|
<br><br>[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>[Figure 298]<br><br>[Figure 299]<br><br>[Figure 300]<br><br>[Figure 301]<br><br>[Figure 302]<br><br>[Figure 303]<br><br>[Figure 304]<br><br>[Figure 305]<br><br>[Figure 306]<br><br>This is what I thought u want me to do.<br><br>|She got really mad and her face twisted up while she argued with her partner.|
|---|
<br><br>[Figure 307]<br><br>[Figure 308]<br><br>[Figure 309]<br><br>[Figure 310]<br><br>[Figure 311]<br><br>[Figure 312]<br><br>[Figure 313]<br><br>[Figure 314]<br><br>[Figure 315]<br><br>[Figure 316]<br><br>I do ! It’s because of you !<br><br>[Figure 317]<br><br>[Figure 318]<br><br>[Figure 319]<br><br>[Figure 320]<br><br>[Figure 321]<br><br>[Figure 322]<br><br>[Figure 323]<br><br>[Figure 324]<br><br>[Figure 325]<br><br>[Figure 326]<br><br>You know nothing be- hind this power.<br><br>[Figure 327]<br><br>[Figure 328]<br><br>[Figure 329]<br><br>[Figure 330]<br><br>[Figure 331]<br><br>[Figure 332]<br><br>[Figure 333]<br><br>[Figure 334]<br><br>[Figure 335]<br><br>[Figure 336]<br><br>Oh? … no no no.<br><br>[Figure 337]<br><br>[Figure 338]<br><br>[Figure 339]<br><br>[Figure 340]<br><br>[Figure 341]<br><br>[Figure 342]<br><br>[Figure 343]<br><br>[Figure 344]<br><br>[Figure 345]<br><br>[Figure 346]<br><br>[Figure 347]<br><br>[Figure 348]<br><br>Á voir le monde et sa beau- té.<br><br>[Figure 349]<br><br>[Figure 350]<br><br>[Figure 351]<br><br>[Figure 352]<br><br>[Figure 353]<br><br>[Figure 354]<br><br>[Figure 355]<br><br>[Figure 356]<br><br>[Figure 357]<br><br>[Figure 358]<br><br>Never gonna make you cry , never gonna say goodbye.<br><br>[Figure 359]<br><br>[Figure 360]<br><br>[Figure 361]<br><br>[Figure 362]<br><br>[Figure 363]<br><br>[Figure 364]<br><br>[Figure 365]<br><br>[Figure 366]<br><br>[Figure 367]<br><br>[Figure 368]<br><br>[Figure 369]<br><br>[Figure 370]<br><br>Ari no, mama no sugata miseru no you.<br><br>[Figure 371]<br><br>[Figure 372]<br><br>[Figure 373]<br><br>[Figure 374]<br><br>[Figure 375]<br><br>[Figure 376]<br><br>[Figure 377]|
|---|

Figure 8. Result gallery. We generate vivid dialogue scenes (Row 1,2) through scripted textual descriptions. We synthesize stylized facial animations (Row 3,4) through image prompts, which can be emoji or even more abstract images. We also perform emotional singing in France, English and Japanese(Row 5-7). For more results, please refer to the supplementary video.

### References

- [1] Simon Alexanderson, Rajmund Nagy, Jonas Beskow, and Gustav Eje Henter. Listen, denoise, action! audio-

driven motion synthesis with diffusion models. ACM Transactions on Graphics, 42(4):1–20, July 2023. 2

- [2] Shivangi Aneja, Justus Thies, Angela Dai, and Matthias Nießner. Facetalk: Audio-driven motion diffusion for neural parametric head models, 2023. 2, 3
- [3] Tenglong Ao, Zeyi Zhang, and Libin Liu. Gesturediffuclip: Gesture diffusion model with clip latents. ACM Trans. Graph., 2023. 2, 7
- [4] Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: a framework for self-supervised learning of speech representations. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS’20, Red Hook, NY, USA, 2020. Curran Associates Inc. 2, 5
- [5] Volker Blanz and Thomas Vetter. A morphable model for the synthesis of 3d faces. In Proceedings of the 26th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH ’99, page 187–194, USA, 1999. ACM Press/Addison-Wesley Publishing Co. 2
- [6] James Booth, Epameinondas Antonakos, Stylianos Ploumpis, George Trigeorgis, Yannis Panagakis, and Stefanos Zafeiriou. 3d face morphable models” inthe-wild”. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 48– 57, 2017. 2
- [7] Giorgos Bouritsas, Sergiy Bokhnyak, Stylianos Ploumpis, Stefanos Zafeiriou, and Michael Bronstein. Neural 3d morphable models: Spiral convolutional networks for 3d shape representation learning and generation. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 7212–7221,

2019. 2, 3

- [8] Houwei Cao, David G Cooper, Michael K Keutmann, Ruben C Gur, Ani Nenkova, and Ragini Verma. Crema-d: Crowd-sourced emotional multimodal actors dataset. IEEE transactions on affective computing, 5(4):377–390, 2014. 5
- [9] Yong Cao, Wen C. Tien, Petros Faloutsos, and Fr´ed´eric Pighin. Expressive speech-driven facial animation. ACM Trans. Graph., 24(4):1283–1302, oct

2005. 3

- [10] Peng Chen, Xiaobao Wei, Ming Lu, Yitong Zhu, Naiming Yao, Xingyu Xiao, and Hui Chen. Diffusiontalker: Personalization and acceleration for speech-driven 3d face diffuser, 2023. 2, 3
- [11] Zhixiang Chen and Tae-Kyun Kim. Learning feature aggregation for deep 3d morphable models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13164–13173,

2021. 2

- [12] Byoungwon Choe and Hyeong-Seok Ko. Analysis and synthesis of facial expressions with hand-generated muscle actuation basis. In ACM SIGGRAPH 2006 Courses, pages 21–es. 2006. 2
- [13] Zhaojie Chu, Kailing Guo, Xiaofen Xing, Yilin Lan, Bolun Cai, and Xiangmin Xu. Corrtalk: Correlation between hierarchical speech and facial activity variances for 3d animation, 2023. 3
- [14] Daniel Cudeiro, Timo Bolkart, Cassidy Laidlaw, Anurag Ranjan, and Michael J. Black. Capture, learning, and synthesis of 3d speaking styles. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. 2, 3, 5
- [15] Radek Danˇeˇcek, Kiran Chhatre, Shashank Tripathi, Yandong Wen, Michael Black, and Timo Bolkart. Emotional speech-driven animation with contentemotion disentanglement. In SIGGRAPH Asia 2023 Conference Papers, SA ’23, New York, NY, USA,

2023. Association for Computing Machinery. 2, 3, 4

- [16] Pif Edwards, Chris Landreth, Eugene Fiume, and Karan Singh. Jali: an animator-centric viseme model for expressive lip synchronization. ACM Transactions on graphics (TOG), 35(4):1–11, 2016. 2, 3
- [17] Bernhard Egger, William A. P. Smith, Ayush Tewari, Stefanie Wuhrer, Michael Zollhoefer, Thabo Beeler, Florian Bernard, Timo Bolkart, Adam Kortylewski, Sami Romdhani, Christian Theobalt, Volker Blanz, and Thomas Vetter. 3d morphable face models—past, present, and future. ACM Trans. Graph., 39(5), jun

2020. 2

- [18] Paul Ekman and Wallace V Friesen. Facial action coding system. Environmental Psychology & Nonverbal Behavior, 1978. 2
- [19] Tony Ezzat and Tomaso Poggio. Miketalk: A talking facial display based on morphing visemes. In Proceedings Computer Animation’98 (Cat. No. 98EX169), pages 96–102. IEEE, 1998. 2, 3
- [20] Yingruo Fan, Zhaojiang Lin, Jun Saito, Wenping Wang, and Taku Komura. Faceformer: Speech-driven 3d facial animation with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 3, 7
- [21] Yingruo Fan, Zhaojiang Lin, Jun Saito, Wenping Wang, and Taku Komura. Joint audio-text model for expressive speech-driven 3d facial animation. 5(1), may 2022. 3
- [22] G. Fanelli, J. Gall, H. Romsdorfer, T. Weise, and L. Van Gool. A 3-d audio-visual corpus of affective communication. IEEE Transactions on Multimedia, 12(6):591 – 598, October 2010. 2

- [23] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion,

2022. 3

- [24] Simon Giebenhain, Tobias Kirschstein, Markos Georgopoulos, Martin R¨unz, Lourdes Agapito, and Matthias Nießner. Learning neural parametric head models. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023. 3
- [25] Kazi Injamamul Haque and Zerrin Yumak. Facexhubert: Text-less speech-driven e(x)pressive 3d facial animation synthesis using self-supervised speech representation learning, 2023. 3
- [26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 2, 3, 5
- [27] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 5, 6
- [28] Dong-Yan Huang, Ellensi Chandra, Xiangting Yang, Ying Zhou, Huaiping Ming, Weisi Lin, Minghui Dong, and Haizhou Li. Visual speech emotion conversion using deep learning for 3d talking head. In Proceedings of the Joint Workshop of the 4th Workshop on Affective Social Multimedia Computing and First Multi-Modal Affective Computing of Large-Scale Multimedia Data, ASMMC-MMAC’18, page 7–13, New York, NY, USA, 2018. Association for Computing Machinery. 3
- [29] Patrik Huber, Guosheng Hu, Rafael Tena, Pouria Mortazavian, Willem P Koppen, William J Christmas, Matthias R¨atsch, and Josef Kittler. A multiresolution 3d morphable face model and fitting framework. In International conference on computer vision theory and applications, volume 5, pages 79–86. SciTePress,

2016. 2

- [30] Patrik Jonell, Taras Kucherenko, Gustav Eje Henter, and Jonas Beskow. Let’s face it: Probabilistic multimodal interlocutor-aware generation of facial gestures in dyadic settings. In Proceedings of the 20th ACM International Conference on Intelligent Virtual Agents, IVA ’20, New York, NY, USA, 2020. Association for Computing Machinery. 3
- [31] G.A. Kalberer and L. Van Gool. Face animation based on observed 3d speech dynamics. In Proceedings Computer Animation 2001. Fourteenth Conference on Computer Animation (Cat. No.01TH8596), pages 20– 251, 2001. 3
- [32] Tero Karras, Timo Aila, Samuli Laine, Antti Herva, and Jaakko Lehtinen. Audio-driven facial animation

- by joint end-to-end learning of pose and emotion. ACM Trans. Graph., 36(4), jul 2017. 3
- [33] Akio Kodaira, Chenfeng Xu, Toshiki Hazama, Takanori Yoshimoto, Kohei Ohno, Shogo Mitsuhori, Soichi Sugano, Hanying Cho, Zhijian Liu, and Kurt Keutzer. Streamdiffusion: A pipeline-level solution for real-time interactive generation. 2023. 6
- [34] Paul Koppen, Zhen-Hua Feng, Josef Kittler, Muhammad Awais, William Christmas, Xiao-Jun Wu, and He-Feng Yin. Gaussian mixture 3d morphable face model. Pattern Recogn., 74(C):617–628, feb 2018. 2
- [35] John P Lewis, Ken Anjyo, Taehyun Rhee, Mengjie Zhang, Frederic H Pighin, and Zhigang Deng. Practice and theory of blendshape facial models. Eurographics (State of the Art Reports), 1(8):2, 2014. 2
- [36] Jiaman Li, Zhengfei Kuang, Yajie Zhao, Mingming He, Karl Bladin, and Hao Li. Dynamic facial asset and rig generation from a single scan. ACM Trans. Graph., 39(6):215–1, 2020. 3
- [37] Ruilong Li, Karl Bladin, Yajie Zhao, Chinmay Chinara, Owen Ingraham, Pengda Xiang, Xinglei Ren, Pratusha Prasad, Bipin Kishore, Jun Xing, and Hao Li. Learning formation of physically-based face attributes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020. 2
- [38] Tianye Li, Timo Bolkart, Michael J. Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4d scans. ACM Trans. Graph., 36(6), nov 2017. 2
- [39] Han Liang, Jiacheng Bao, Ruichi Zhang, Sihan Ren, Yuecheng Xu, Sibei Yang, Xin Chen, Jingyi Yu, and Lan Xu. Omg: Towards open-vocabulary motion generation via mixture of controllers. arXiv preprint arXiv:2312.08985, 2023. 2
- [40] Haiyang Liu, Zihao Zhu, Naoya Iwamoto, Yichen Peng, Zhengqing Li, You Zhou, Elif Bozkurt, and Bo Zheng. Beat: A large-scale semantic and emotional multi-modal dataset for conversational gestures synthesis. In Shai Avidan, Gabriel Brostow, Moustapha Ciss´e, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision – ECCV 2022, pages 612–630, Cham, 2022. Springer Nature Switzerland. 2
- [41] Steven R Livingstone and Frank A Russo. The ryerson audio-visual database of emotional speech and song (ravdess): A dynamic, multimodal set of facial and vocal expressions in north american english. PloS one, 13(5):e0196391, 2018. 5
- [42] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilis-

- tic models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11461–11471, June 2022. 7
- [43] Juan F Montesinos, Venkatesh S Kadandale, and Gloria Haro. A cappella: Audio-visual singing voice separation. In 32nd British Machine Vision Conference, BMVC 2021, 2021. 5
- [44] Evonne Ng, Hanbyul Joo, Liwen Hu, Hao Li, Trevor Darrell, Angjoo Kanazawa, and Shiry Ginosar. Learning to listen: Modeling non-deterministic dyadic facial motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20395–20405, June 2022. 3
- [45] Ziqiao Peng, Yihao Luo, Yue Shi, Hao Xu, Xiangyu Zhu, Jun He, Hongyan Liu, and Zhaoxin Fan. Selftalk: A self-supervised commutative training diagram to comprehend 3d talking faces, 2023. 3
- [46] Ziqiao Peng, Haoyu Wu, Zhenbo Song, Hao Xu, Xiangyu Zhu, Jun He, Hongyan Liu, and Zhaoxin Fan. Emotalk: Speech-driven emotional disentanglement for 3d face animation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20687–20697, 2023. 2, 3, 7
- [47] Hai Xuan Pham, Yuting Wang, and Vladimir Pavlovic. End-to-end learning for 3d facial animation from speech. In Proceedings of the 20th ACM International Conference on Multimodal Interaction, ICMI ’18, page 361–365, New York, NY, USA, 2018. Association for Computing Machinery. 3
- [48] Dafei Qin, Jun Saito, Noam Aigerman, Thibault Groueix, and Taku Komura. Neural face rigging for animating and retargeting facial meshes in the wild. arXiv preprint arXiv:2305.08296, 2023. 2, 3
- [49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2, 5
- [50] Alexander Richard, Michael Zollh¨ofer, Yandong Wen, Fernando de la Torre, and Yaser Sheikh. Meshtalk: 3d face animation from speech using cross-modality disentanglement. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 1173–1182, October 2021. 3, 7
- [51] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022. 3

- [52] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models, 2022. 2, 3
- [53] Yonatan Shafir, Guy Tevet, Roy Kapon, and Amit H Bermano. Human motion diffusion as a generative prior. arXiv preprint arXiv:2303.01418, 2023. 7
- [54] Il-Kyu Shin, A. Cengiz Oztireli,¨ Hyeon-Joong Kim, Thabo Beeler, Markus Gross, and Soo-Mi Choi. Extraction and Transfer of Facial Expression Wrinkles for Facial Performance Enhancement. In John Keyser, Young J. Kim, and Peter Wonka, editors, Pacific Graphics Short Papers. The Eurographics Association, 2014. 2
- [55] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2256–2265, Lille, France, 07–09 Jul

2015. PMLR. 2, 3

- [56] Stefan Stan, Kazi Injamamul Haque, and Zerrin Yumak. Facediffuser: Speech-driven 3d facial animation synthesis using diffusion. In ACM SIGGRAPH Conference on Motion, Interaction and Games (MIG ’23), November 15–17, 2023, Rennes, France, New York, NY, USA, 2023. ACM. 2, 3, 7
- [57] Zhiyao Sun, Tian Lv, Sheng Ye, Matthieu Gaetan Lin, Jenny Sheng, Yu-Hui Wen, Minjing Yu, and Yong jin Liu. Diffposetalk: Speech-driven stylistic 3d facial animation and head pose generation via diffusion models, 2023. 2, 3, 4, 6, 7
- [58] Sarah Taylor, Taehwan Kim, Yisong Yue, Moshe Mahler, James Krahe, Anastasio Garcia Rodriguez, Jessica Hodgins, and Iain Matthews. A deep learning approach for generalized speech animation. ACM Trans. Graph., 36(4), jul 2017. 3
- [59] Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit H Bermano. Human motion diffusion model. 2022. 2, 5, 7
- [60] Balamurugan Thambiraja, Sadegh Aliakbarian, Darren Cosker, and Justus Thies. 3diface: Diffusionbased speech-driven 3d facial animation and editing,

2023. 2, 3

- [61] Balamurugan Thambiraja, Ikhsanul Habibie, Sadegh Aliakbarian, Darren Cosker, Christian Theobalt, and Justus Thies. Imitator: Personalized speech-driven 3d facial animation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 20621–20631, October 2023. 3

- [62] Justus Thies, Mohamed Elgharib, Ayush Tewari, Christian Theobalt, and Matthias Nießner. Neural voice puppetry: Audio-driven facial reenactment. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, editors, Computer Vision – ECCV 2020, pages 716–731, Cham, 2020. Springer International Publishing. 3
- [63] Luan Tran, Feng Liu, and Xiaoming Liu. Towards high-fidelity nonlinear 3d face morphable model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1126– 1135, 2019. 2
- [64] Luan Tran and Xiaoming Liu. Nonlinear 3d face morphable model. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7346–7355, 2018. 2
- [65] Luan Tran and Xiaoming Liu. On learning 3d face morphable model from in-the-wild images. IEEE transactions on pattern analysis and machine intelligence, 43(1):157–171, 2019. 2
- [66] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 3
- [67] Monica Villanueva Aylagas, Hector Anadon Leon, Mattias Teye, and Konrad Tollmar. Voice2face: Audio-driven facial and tongue rig animations with cvaes. Computer Graphics Forum, 41(8):255–265,

2022. 3

- [68] Kaisiyuan Wang, Qianyi Wu, Linsen Song, Zhuoqian Yang, Wayne Wu, Chen Qian, Ran He, Yu Qiao, and Chen Change Loy. Mead: A large-scale audio-visual dataset for emotional talking-face generation. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and JanMichael Frahm, editors, Computer Vision – ECCV 2020, pages 700–717, Cham, 2020. Springer International Publishing. 5
- [69] Cheng-hsin Wuu, Ningyuan Zheng, Scott Ardisson, Rohan Bali, Danielle Belko, Eric Brockmeyer, Lucas Evans, Timothy Godisart, Hyowon Ha, Xuhua Huang, et al. Multiface: A dataset for neural face rendering. arXiv preprint arXiv:2207.11243, 2022. 3
- [70] Jinbo Xing, Menghan Xia, Yuechen Zhang, Xiaodong Cun, Jue Wang, and Tien-Tsin Wong. Codetalker: Speech-driven 3d facial animation with discrete motion prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12780–12790, 2023. 2, 3, 7
- [71] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Comput. Surv., 56(4), nov 2023. 3

- [72] Tarun Yenamandra, Ayush Tewari, Florian Bernard, Hans-Peter Seidel, Mohamed Elgharib, Daniel Cremers, and Christian Theobalt. i3dmm: Deep implicit 3d morphable model of human heads. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12803–12813, 2021. 3
- [73] Hongwei Yi, Hualin Liang, Yifei Liu, Qiong Cao, Yandong Wen, Timo Bolkart, Dacheng Tao, and Michael J. Black. Generating holistic 3d human motion from speech, 2023. 2
- [74] Hongwei Yi, Hualin Liang, Yifei Liu, Qiong Cao, Yandong Wen, Timo Bolkart, Dacheng Tao, and Michael J. Black. Generating holistic 3d human motion from speech. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 469–480, June 2023. 3
- [75] Chenxu Zhang, Saifeng Ni, Zhipeng Fan, Hongbo Li, Ming Zeng, Madhukar Budagavi, and Xiaohu Guo. 3d talking face with personalized pose dynamics. IEEE Transactions on Visualization and Computer Graphics, 29(2):1438–1449, 2023. 3
- [76] Longwen Zhang, Qiwei Qiu, Hongyang Lin, Qixuan Zhang, Cheng Shi, Wei Yang, Ye Shi, Sibei Yang, Lan Xu, and Jingyi Yu. Dreamface: Progressive generation of animatable 3d faces under text guidance. arXiv preprint arXiv:2304.03117, 2023. 7
- [77] Longwen Zhang, Chuxiao Zeng, Qixuan Zhang, Hongyang Lin, Ruixiang Cao, Wei Yang, Lan Xu, and Jingyi Yu. Video-driven neural physically-based facial asset for production. volume 41, pages 1–16. ACM New York, NY, USA, 2022. 4
- [78] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. Sadtalker: Learning realistic 3d motion coefficients for stylized audio-driven single image talking face animation, 2022. 7
- [79] Zhimeng Zhang, Lincheng Li, Yu Ding, and Changjie Fan. Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 5
- [80] Mingwu Zheng, Hongyu Yang, Di Huang, and Liming Chen. Imface: A nonlinear 3d morphable face model with implicit neural representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20343–20352, 2022. 3
- [81] Yang Zhou, Zhan Xu, Chris Landreth, Evangelos Kalogerakis, Subhransu Maji, and Karan Singh. Visemenet: Audio-driven animator-centric speech animation. 37(4), jul 2018. 3

