## ISDrama: Immersive Spatial Drama Generation through Multimodal Prompting

Yu Zhang∗

Wenxiang Guo∗

Changhao Pan∗

Zhejiang University, Shanghai AI Lab Hangzhou, Zhejiang, China yuzhang34@zju.edu.cn

Zhejiang University Hangzhou, Zhejiang, China guowx314@zju.edu.cn

Zhejiang University Hangzhou, Zhejiang, China panch@zju.edu.cn

Zhiyuan Zhu∗

Zhou Zhao†

Tao Jin

Zhejiang University Hangzhou, Zhejiang, China schmittzhu@zju.edu.cn

Zhejiang University, Shanghai AI Lab Hangzhou, Zhejiang, China zhaozhou@zju.edu.cn

Zhejiang University Hangzhou, Zhejiang, China jint_zju@zju.edu.cn

# arXiv:2504.20630v6[eess.AS]29Jul2025

#### KEYWORDS

#### ABSTRACT

Multimodal immersive spatial drama generation focuses on creating continuous multi-speaker binaural speech with dramatic prosody based on multimodal prompts, with potential applications in AR, VR, and others. This task requires simultaneous modeling of spatial information and dramatic prosody based on multimodal inputs, with high data collection costs. To the best of our knowledge, our work is the first attempt to address these challenges. We construct MRSDrama, the first multimodal recorded spatial drama dataset, containing binaural drama audios, scripts, videos, geometric poses, and textual prompts. Then, we propose ISDrama, the first immersive spatial drama generation model through multimodal prompting. ISDrama comprises these primary components: 1) Multimodal Pose Encoder, based on contrastive learning, considering the Doppler effect caused by moving speakers to extract unified pose information from multimodal prompts. 2) Immersive Drama Transformer, a flow-based mamba-transformer model that generates high-quality drama, incorporating DramaMOE to select proper experts for enhanced prosody and pose control. We also design a context-consistent classifier-free guidance strategy to coherently generate complete drama. Experimental results show that ISDrama outperforms baseline models on objective and subjective metrics. The demos are available at https://aaronz345. github.io/ISDramaDemo. We provide the dataset and the evaluation code at https://huggingface.co/datasets/AaronZ345/MRSDrama and https://github.com/AaronZ345/ISDrama.

immersive spatial drama generation, spatial audio dataset, prosody modeling, speech synthesis, multimodal prompting

ACM Reference Format:

Yu Zhang, Wenxiang Guo, Changhao Pan, Zhiyuan Zhu, Tao Jin, and Zhou Zhao. 2025. ISDrama: Immersive Spatial Drama Generation through Multimodal Prompting. In Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 17 pages. https://doi.org/10.1145/3746027.3755014

#### 1 INTRODUCTION

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

L R

Immersive Spatial Drama

- Speaker1:你们既不是 注定要观看，也不是 因为被解除了其他⾏ 动⽽来观看...
- Speaker2:你们是... Drama Script Prompt Audio

[Figure 5]

ISDrama

- Speaker1
- Speaker2

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Speaker1, initially stands in the front-right, walks at a slow speed…

[Figure 10]

[Figure 11]

[Figure 12]

#### CCS CONCEPTS

[Figure 13]

Silent Video

Geometric Pose

Textual Prompt

##### • Computing methodologies → Artificial intelligence; • Applied computing → Arts and humanities.

Figure 1: ISDrama uses scripts as content, prompt audio to guide timbre, and spatial information from multimodal prompts, to create continuous multi-speaker binaural speech with dramatic prosody.

∗Equal contribution. †Corresponding Author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Binaural hearing provides localization cues through sound fields, enhancing the human spatial perception of the environment. This capability is critical for applications requiring deep immersion, like movies, VR, and AR [6, 18, 29]. Compared to non-linguistic binaural audio [63], generating binaural speech is more challenging yet promising. Specifically, based on multimodal prompts from diverse contexts, generating continuous multi-speaker binaural speech with

MM ’25, October 27–31, 2025, Dublin, Ireland © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2035-2/2025/10...$15.00 https://doi.org/10.1145/3746027.3755014

dramatic prosody can create immersive spatial drama. This new task enhances storytelling, offering an immersive emotional and spatial engagement, and an integration of virtual and reality.

With deep learning advancements, progress has been made in synthesizing speech with prosody modeling [15, 31] and generating binaural audio from monaural audio based on input multimodal prompts [45, 52]. Cascading these methods to generate continuous multi-speaker binaural speech with prosody modeling seems to be a viable approach. However, this cascading method will disrupt the integration of prosody and spatial modeling, leading to mismatches and error accumulation. This highlights the necessity for a unified framework for multimodal immersive spatial drama generation.

To date, one-stage multimodal immersive spatial drama generation remains an unexplored domain, focusing on creating continuous multi-speaker binaural speech with dramatic prosody. As shown in Figure 1, this task requires inputting scripts as content and prompt audio to guide timbre, along with extracting spatial information in pose (including position, orientation, and movement speed) and scene from multimodal prompts (e.g., silent video, textual prompts, geometric poses) to cover a wider range of applications. Therefore, this task extends beyond video dubbing, enabling precise and flexible spatial control. Additionally, drama demands prosodic expressiveness that far exceeds normal speech. Not only does dramatic prosody require learning accent and articulation from prompt audio, but it also integrates semantics to enhance the modeling of semantically aligned emotion and rhythm.

Currently, multimodal immersive spatial drama generation encounters three major challenges: 1) Lack of high-quality annotated recorded data. Simulated data fails to capture the complex, dramatic prosody and the precise effects of real-world spatial scenes, positions, and orientations. While some datasets [58] use binaural devices to simulate human interaural phase difference (IPD) and interaural level difference (ILD), they suffer from limitations in scale, dramatic prosody, and multimodal prompts. 2) Challenges in extracting unified pose representations from multimodal prompts. Silent video, geometric pose, and textual prompts provide spatial information, including positions, orientations, and movement speed for various scenarios. While some methods extract positional information from visual or positional inputs [39], they cannot learn unified pose representations across more diverse scenarios. 3) Difficulty in one-stage modeling dramatic prosody and spatial immersion. Existing monaural speech models [15] struggle to simultaneously model dramatic prosody, which requires semantic alignment in the temporal-frequency domain, and spatial information, which spans the spatial-temporal dimensions.

To address these challenges, we first introduce MRSDrama, the first multimodal recorded spatial drama dataset, comprising binaural drama audios, scripts, videos, geometric poses, and textual prompts. The dataset includes 97.82 hours of speech data recorded by 21 speakers across three scenes. Next, we propose ISDrama, the first immersive spatial drama generation model based on multimodal prompting. ISDrama generates high-quality, continuous, multi-speaker binaural speech with dramatic prosody and spatial immersion, driven by multimodal prompts. To extract a unified pose representation from multimodal prompts, we design the Multimodal Pose Encoder, a contrastive learning-based framework that encodes not only position and head orientation but also radial

velocity, accounting for the Doppler effect caused by moving speakers. Meanwhile, we develop the Immersive Drama Transformer, a flow-based Mamba-Transformer model capable of generating immersive spatial drama effectively and stably. Within this model, we introduce Drama-MOE (Mixture of Experts), which selects the appropriate experts to enhance prosodic expressiveness and improve pose control. Then, we adopt a context-consistent classifier-free guidance (CFG) strategy to ensure the quality and coherence of complete drama generation. We evaluate ISDrama on quality, speaker similarity, prosodic expressiveness, pose, angle, distance, IPD, and ILD. Experimental results show that ISDrama outperforms baseline models on both objective and subjective metrics, demonstrating its ability to generate immersive spatial drama that adheres to physical principles while exhibiting rich prosodic variation. Overall, our main contributions can be summarized as follows:

- • We develop MRSDrama, the first multimodal recorded spatial drama dataset, accompanying videos, scripts, alignments, positions, and textual prompts.
- • We introduce ISDrama, the first immersive spatial drama generation model through multimodal prompting. We design the Multimodal Pose Encoder to extract pose information from multimodal inputs, while the Immersive Drama Transformer produces high-quality binaural speech.
- • Experimental results show that ISDrama outperforms baseline models on objective and subjective metrics.

#### 2 RELATED WORK

Audio Spatialization with Multimodal Cues. In recent years, the development of deep learning has significantly advanced the exploration of spatial audio. Substantial progress has been made in sound source localization for binaural audio [21, 36, 60, 67]. At the same time, the rise of multimodal research has spurred innovations in spatial audio synthesis. Mono2Binaural [19] devises a deep convolutional neural network that leverages visual cues to convert monaural audio into binaural audio. Sep-Stereo [77] enhances stereo audio by integrating audio-visual features. Garg et al. [22] proposes a multi-task framework that learns geometry-aware features for generating binaural audio. CLUP [45] jointly learns to localize visually sounding objects and generate binaural audio. Beyond visual modalities, BinauralGrad [39] employs positional information for monaural-to-stereo audio generation, while scene depth maps have also been utilized [52]. However, these methods rely on monaural audio input, limiting their applicability for complex and flexible generation tasks. Recently, SpatialSonic [63] employs a Diffusion Transformer to generate binaural audio from both text and image prompts. VISAGE [33] leverages CLIP visual features to generate first-order ambisonics directly from silent video frames. However, these tasks focus on generating short audio clips lacking actual linguistic information. ImmerseDiffusion [27] proposes an end-to-end generative audio model conditioned on textual prompts of spatial, temporal, and environmental factors. However, it lacks support for multimodal prompts, binaural scenes, and expressive prosody modeling. Generating continuous linguistic speech with unified scene and pose information extracted from more diverse prompt modalities remains an unresolved challenge.

##### Table 1: Comparison of open-source recorded spatial speech datasets. Multi-channel speech does not account for the intricate structures of human ears, whereas binaural speech incorporates natural IPD and ILD, ensuring an immersive hearing perception.

Dataset Hours Speakers Audio Paired Type Sweet-Home [64] 47.3 71 multi-channel text, prompt DIRHA [55] 11 24 multi-channel text, pose Voice-Home [4] 2.5 3 multi-channel text, pose, prompt Binaural[58] 2 8 binaural pose MRSDrama (Ours) 97.82 21 binaural text, video, pose, prompt

[Figure 14]

| | |
|---|---|
| | |
| | |

Data Check & Segmentation

Script Audio Recording Preparation

[Figure 15]

[Figure 16]

Denoising wav

Alignment

[Figure 17]

[Figure 18]

L R

[Figure 19]

textgrid

wav

[Figure 20]

[Figure 21]

Video Recording

Geometric Annotation

Prompt Annotation: Speaker1, initially stands in the front-right, walks at a slow speed…

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

json

MRSDrama

txt mp4

json

Step1: Recording Step3: Post-Processing

Step2: Annotation

##### Figure 2: The pipeline of MRSDrama data collection. Human double-checks exist in each process. All data are desensitized.

Speech Synthesis with Prosody Modeling. Prosody modeling aims to synthesize target speech with natural or emotional prosody, which is essential to generate expressive speech in a controlled manner and typically involves transferring prosody from prompt audio [65]. Skerry-Ryan et al. [61] is the first to integrate a prosody reference encoder into a Tacotron-based TTS system, enabling the transfer of prosody for similar-text speech. Attentron [11] introduces an attention mechanism to extract prosody from reference samples. ZSM-SS [37] proposes a Transformer-based architecture with an external speaker encoder using wav2vec 2.0 [3]. Li et al. [42] incorporates global utterance-level and local phoneme-level prosodic features in target speech. Daft-Exprt [68] employs a gradient reversal layer to enhance target speaker fidelity in prosody transfer. Generspeech [28] incorporates the attention mechanism to capture multi-level prosody. HierSpeech++ [38] generates F0 representation based on text representations and prosody prompts, while StyleTTS 2 [44] predicts pitch and energy based on a prosody predictor [43]. NaturalSpeech 3 [31] employs factorized vector quantization to disentangle prosody. CosyVoice [15] incorporates x-vectors into an LLM to disentangle and model prosody. FireRedTTS [25] employs a semantic-aware speech tokenizer to encode speech style. However, these models only focus on monaural speech synthesis and cannot simultaneously model dramatic prosody, which requires semantic alignment in the temporal-frequency domain, and spatial information, which spans the spatial-temporal dimensions.

cannot reflect the nuanced prosody or subtle acoustic effects of realworld spatial environments. Therefore, we propose MRSDrama, the first multimodal recorded spatial drama dataset containing binaural drama audios, videos, scripts, geometric poses, and textual prompts. Our dataset consists of 97.82 hours of speech data recorded by 21 speakers across 3 scenes. Figure 2 shows the construction pipeline. Recording. To construct ISDrama, we first source Chinese translations of renowned theatrical scripts through authorized channels, including Hamlet, Waiting for Godot, Thunderstorm, Hiroshima Mon Amour, Offending the Audience, etc. After a meticulous selection process, we enlist 21 expressive speakers, each provides consent to record for research purposes. Additionally, we select three scenes with varying sizes and acoustical effects to ensure diverse spatial effects. During the recording process, speakers are instructed to read the script expressively while remaining in character and occasionally moving at a constant speed to create dynamic spatial effects. We use professional binaural recording equipment and sound cards to capture the audio, while synchronized video footage is recorded using cameras. All audio files are saved in WAV format with a 48 kHz sampling rate, and videos are captured at 24 fps.

Annotation. We separately annotate the audio and video components. 1) For Audio Annotation, We first perform denoising using FRCRN [75]. Subsequently, MFA [50] is employed for a coarse phoneme-to-audio alignment between the original script and audio. Chinese phonemes are extracted by pypinyin. Next, annotators are asked to use Praat [5] to refine the rough alignment, focusing on correcting word and phoneme boundaries and addressing erroneous words based on auditory perception. 2) For Video Annotation, Speakers occasionally move between many fixed points in each scene. Annotators are asked to record speakers’ arrival times and point coordinates of each movement. They then measure these

#### 3 DATASET: MRSDRAMA

Due to the high costs associated with multimodal spatial speech recording and annotation, as shown in Table 1, existing open-source recorded datasets are inadequate for supporting the generation of multimodal immersive spatial drama. Meanwhile, simulated data

[Figure 27]

[Figure 28]

[Figure 29]

(b) Speaker Information

(a) Theatrical Scripts

[Figure 30]

(c) Phoneme Duration

d) Position Distribution

- Figure 3: The statistics of MRSDrama. The position distribution is plotted on the plane defined by listeners’ direction and ears.

speakers’ head orientation and mouth height while standing and sitting in each frame, extracting 3D position coordinates and quaternion orientation to form frame-level sound source poses. Based on the annotated pose data, we then use GPT-4o [1] to generate textual prompts for each actor’s line by combining the orientation, endpoint, direction, speed, and start time of each motion. Meanwhile, annotators also label the camera pose (position and orientation) and scene prompts, including room sizes and acoustical effects.

Post-Processing. To ensure data quality, we inspect the entire dataset, including the script, alignment, poses, and prompts. Next, we segment the 97.82 hours of speech into 47,958 segments based on speaker transitions of each drama and a maximum duration setting of 16 seconds. MRSDrama is the largest recorded spatial speech corpus to date and the first spatial drama dataset with multimodal annotations. It features continuous multi-speaker binaural speech with dramatic prosody accompanying rich modalities, making it suitable for various tasks like binaural localization and drama generation. Figure 3 presents the statistics of MRSDrama, showcasing the diversity across theatrical scripts, speakers, phonemes, and positions. This highlights the dataset’s variety in content, timbre, and spatial information, demonstrating its potential for generalization. For more details about MRSDrama, please refer to Appendix B.

- 4 METHOD: ISDRAMA

generation process is 𝑚ˆ𝑝𝑟 = 𝐺(𝜖 | 𝐶), where 𝜖 is Gaussian noise and 𝐶 represents the conditions. 𝐶 includes the corresponding content 𝑐, scene 𝑠, pose 𝑝 from multimodal prompts, and specified prompt audio 𝑎. To encode unified pose embedding 𝑧𝑝 from multimodal prompts, we design the Multimodal Pose Encoder based on contrastive learning, accounting for the Doppler effect of moving speakers. In addition to the pose 𝑝 corresponding to𝑦𝑔𝑡, the content 𝑐 is segmented from the complete drama script. The pronunciation and semantics of 𝑐 are encoded as 𝑧𝑐. Next, 𝑧𝑝 and 𝑧𝑐 are fed into the Immersive Drama Transformer, along with scene information 𝑠 (a video frame or textual description) and prompt audio𝑎 for timbre, to generate the predicted binaural speech 𝑦𝑝𝑟 through spatial and prosodic modeling. Then, by combining different segments of 𝑦𝑝𝑟, as shown in Figure 4 (c), we can coherently generate the complete drama through a context-consistent CFG strategy.

#### 4.2 Multimodal Pose Encoder

To support a broader range of application scenarios, the generation of spatial drama must not only accommodate video dubbing but also offer flexibility in textual prompts and precise control over geometric poses, including positions, orientations, and movement speed. To address this, we design the Multimodal Pose Encoder, which predicts a unified pose embedding 𝑧𝑝 from multimodal prompts. As shown in Figure 4(a), our model encodes three types of multimodal prompts and embeds them into a unified space.

#### 4.1 Overview

We aim to generate immersive spatial drama based on scripts, prompt audio for each speaker, and spatial information from multimodal prompts. Let 𝑦𝑔𝑡 represent one of the binaural speech in the ground truth drama, and 𝑚𝑔𝑡 ∈ R2×80×𝑇 represent the mel spectrogram, where𝑇 denotes the target length. Typically, we segment target drama based on speaker transitions in the script. As shown in Figure 4, since the autoencoder compresses 𝑚𝑔𝑡 into 𝑚ˆ𝑔𝑡, the

The correct phase estimation is crucial for binaural audio [57]. Therefore, for geometric pose, we not only encode the head orientation 𝑜𝑟𝑖 and 3D coordinates of the sound source relative to left and right ears, 𝑝𝑜𝑠𝑙 and 𝑝𝑜𝑠𝑟 but also add the radial relative velocity according to Doppler effect [23] for phase estimation [47]. Specifically, we calculate the 3D velocity vector 𝑣 of the moving sound source in the Cartesian coordinate system, then decompose 𝑣 into

F0

𝜖~𝑁(0,𝐼)

𝑣

Timestep 𝑡 𝑁×

⊕

Contrastive Learning

ODE Solver

Drama-MOE

𝑧 𝑧 𝑧

Global Adaptor

Scene 𝑠

Spatial Audio Decoder

⊕

Attention

Camera Pose

Geometric Encoder

Video Encoder

Textual Encoder

𝑧

Vocoder Predicted Audio 𝑦 L R

Global Adaptor

⊕

𝑧

⊕

𝑧

[Figure 31]

[Figure 32]

Speaker1, initially stands in the frontright, walk…

[Figure 33]

Mamba×𝐾 Global Adaptor

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Prompt Encoder

Encoder & DP & LR

GT Audio 𝑦

⊕

[Figure 38]

Spatial Audio Encoder

Phoneme Sequence

Geometric Pose

Textual Prompt

Silent Video

Noising

𝑧 𝑥

Prompt Audio 𝑎

(a) Multimodal Pose Encoder

(b) Immersive Drama Transformer

Context-Consistent CFG

⊕

Immersive Spatial Drama

FAN1 FAN2 … FANn

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

L R L R L R

⊕ ⊕

𝑧

[Figure 45]

[Figure 46]

[Figure 47]

Spatial Router

Speaker 1 Speaker 2 Speaker 1

⊕

𝑧

…

Multimodal 𝑠 Prompts

𝑠 𝑠 IDT

IDT

###### IDT

…

FAN1 FAN2 FANn Prosodic Router

𝑧

Drama Script Phonemes

𝑧 1 𝑧 1

𝑧 2 𝑧 2

𝑧 3 𝑧 3

MPE

(c) Complete Drama Inference Procedure

(d) Drama-MOE

Figure 4: The architecture of ISDrama. In Figure (a), DP & LR is duration predictor and length regulator. In Figure (b), the ODE Solver, autoencoder decoder, and vocoder generate predicted audio from Gaussian noise with conditions during inference. In Figure (c), MPE is the Multimodal Pose Encoder, while IDT is the Immersive Drama Transformer. MPE predicts content duration and segments inputs. Then IDL coherently generates the complete drama. In Figure (d), FAN is Fourier Analysis Networks.

radial velocity components 𝑣𝑟𝑎𝑑−𝑙 and 𝑣𝑟𝑎𝑑−𝑟 in the spherical coordinate systems of left and right ears respectively by 𝑣𝑟𝑎𝑑 = ∥ 𝑝𝑜𝑠 𝑝𝑜𝑠· 𝑣∥ ·rˆ, where rˆ ∈ R1 is the radial unit vector. Finally, we can encode and concatenate ( 𝑝𝑜𝑠𝑙, 𝑝𝑜𝑠𝑟,𝑜𝑟𝑖,𝑣𝑟𝑎𝑑−𝑙,𝑣𝑟𝑎𝑑−𝑟) into 𝑧𝑝−𝑔𝑒𝑜.

then use the contrastive objective [54] for training:

∑︁𝑁

exp(𝑠𝑖𝑚(𝑧𝑝1𝑖,𝑧𝑝2𝑖)/𝜏)

- 1

- 2𝑁

(log

L𝑝1,𝑝2 = −

𝑁 𝑗=1 exp(𝑠𝑖𝑚(𝑧𝑝1𝑖,𝑧𝑝2𝑗)/𝜏)

𝑖=1

(1)

exp(𝑠𝑖𝑚(𝑧𝑝2𝑖,𝑧𝑝1𝑖)/𝜏)

To determine the target embedding length and considering the relationship between semantics, spoken speed, and motion, we encode the pronunciation and semantics of content 𝑐 and expand it with the predicted phoneme duration as 𝑧𝑐. Since the input script is organized in speaker transitions, we can compute predicted phoneme durations to segment the length for each target speech as shown in Figure 4 (c). Then we employ a Transformer to predict 𝑧𝑝−𝑡𝑥𝑡 and 𝑧𝑝−𝑣𝑖𝑑, where 𝑧𝑐 serves as the input and leverages different modal conditions. For textual prompts, we use FLAN-T5 [12] to encode text as the condition. For silent video, we encode and concatenate camera pose, mouth pixel sequences from Co-Tracker3 [32], and embedding from CLIP [54] as the condition.

+ log

),

𝑁 𝑗=1 exp(𝑠𝑖𝑚(𝑧𝑝2𝑖,𝑧𝑝1𝑗)/𝜏)

where 𝑠𝑖𝑚(·) denotes cosine similarity. The final total loss L𝑐𝑜𝑛𝑡𝑟𝑎𝑠 will be L𝑝𝑔𝑒𝑜,𝑝𝑣𝑖𝑑 + L𝑝𝑔𝑒𝑜,𝑝𝑡𝑥𝑡 + L𝑝𝑣𝑖𝑑,𝑝𝑡𝑥𝑡 . After training, these three embeddings are aligned in the same space. The resulting unified 𝑧𝑝 ∈ R2×𝐻×𝑇 , where 𝐻 denotes the hidden size, contains pose information, helping to model natural IPD and ILD of binaural speech. For more details, please refer to Appendix C.4.

#### 4.3 Immersive Drama Transfomer

##### Flow-basedMamba-Transformer.Spatialinformationandprosody

in binaural speech are related to pose, scene, emotion, and rhythm. Their interaction necessitates the simultaneous modeling of both temporal-frequency and spatial-temporal, which becomes more complex with long sequences. The flow matching method facilitates smooth transformations, promoting stable and rapid generation, while the Transformer architecture excels in sequence generation, making the flow-based Transformer particularly suitable for this

After obtaining pose embeddings of these three modalities, we design three types of contrasts for contrastive learning, each focusing on different aspects of the pose to explore diverse physical and spatial features. These include dynamic features (mobility, movement speeds, and movement direction), postural features (posture and orientation), and positional features (distance and angle). We

task. Meanwhile, since Transformer tends to be computationally expensive for long sequences, we add Mamba blocks [24] at an Attention-to-Mamba ratio of 1:K to balance memory usage, efficient training, and the ability to handle long sequences.

As shown in Figure 4 (b), we add Gaussian noise 𝜖 to the autoencoder output 𝑚ˆ𝑔𝑡 to obtain 𝑥𝑡 at timestep 𝑡. We then add content embedding 𝑧𝑐 and pose embedding 𝑧𝑝 to 𝑥𝑡 and concatenate it with prompt audio embedding 𝑧𝑎. Therefore, using the self-attention mechanism and capability of Mamba blocks to capture long-range dependencies, Immersive Drama Transformer can effectively model content, pose, timbre, accent, and articulation. Scene information 𝑠 (e.g., a video frame in silent video or a textual description for other inputs) is encoded and processed using the cross-attention mechanism to simulate the room’s acoustic properties, such as the differences in acoustic effects caused by varying room sizes and acoustical effects. Notably, we use the first block’s output to predict binaural F0, providing extra supervision and additional input for subsequent blocks, helping to model dramatic prosody. Furthermore, we employ RMSNorm [69] and design global adapters with AdaLN [53] to ensure training stability and global consistency in timbre and scene. The final output vector field is trained as:

L𝑓 𝑙𝑜𝑤 = E𝑡,𝑝𝑡 (𝑥𝑡) 𝑣𝑡 (𝑥𝑡,𝑡|𝐶;𝜃) − (𝑚ˆ𝑔𝑡 − 𝜖) 2 , (2)

where 𝑝𝑡 (𝑥𝑡) represents the distribution of 𝑥𝑡 at timestep 𝑡. For more details, please refer to Appendix A and C.6.

Drama-MOE. To enhance prosodic expressiveness and pose control, we propose Drama-MOE (Mixture of Experts), which selects suitable experts based on various input conditions. As shown in Figure 4 (d), our Drama-MOE consists of two expert groups, each focusing on dramatic prosody and spatial information. Prosodic MOE leverages prosody in aligned prompt audio embedding 𝑧𝑎 and semantics in content embedding 𝑧𝑐 to select suitable experts for fine-grained dramatic prosody modeling, such as an expert specialized in a happy, fast, high-pitched tone.𝑧𝑎 is aligned with the inputs by a cross-attention model. The Spatial MOE conditions on 𝑧𝑎, adjusting inputs to match the corresponding spatial information, such as sound source pose-induced changes and binaural differences in phase and loudness. It selects proper experts for different inputs, like one expert specialized in a sound source slowly approaching the listener from the far left-front direction.

Each expert in our design leverages Fourier Analysis Networks (FAN) [14] to decompose frequency components, enabling explicit modeling of binaural spatial dynamics while capturing speech periodicity, rhythm, and intonation. The FAN layer is defined as:

𝜙(𝑥) ≜ [cos(𝑊𝑝𝑥)|| sin(𝑊𝑝𝑥)||𝜎(𝐵𝑝¯ +𝑊𝑝¯𝑥)], (3)

where 𝑊𝑝,𝑊𝑝¯, and 𝐵𝑝¯ are learnable parameters. FAN combines sinusoidal functions and a nonlinear activation, capturing periodic and non-periodic variations. This enhances the power for spatialtemporal and time-frequency auditory modeling, aiding in the synchronized modeling of speech prosody and spatial information.

Our routingstrategiesinDrama-MOE use dense-to-sparse Gumbel-

Softmax [51], enabling dynamic and efficient expert selection for each group. Let ℎ be the hidden representation, and 𝑔(ℎ)𝑖 denote the routing score for expert 𝑖. To prevent overloading and ensure

balance, we apply a load-balancing loss [16]:

∑︁

###### ∑︁𝑁

1 𝐵

𝑔(ℎ)𝑖 , (4)

L𝑏𝑎𝑙𝑎𝑛𝑐𝑒 = 𝛼𝑁

𝑖=1

ℎ∈𝐵

where 𝐵 is the batch size, 𝑁 is the number of experts, and 𝛼 a hyperparameter controlling regularization strength. For more details and the algorithm, please refer to Appendix C.7.

#### 4.4 Complete Drama Inference Procedure

For inference, users can input a complete script and specify each speaker’s timbre with prompt audio 𝑎. We typically segment the target drama based on speaker transitions in the script. As shown in Figure 4 (c), the Multimodal Pose Encoder predicts content duration and provides each target’s embedding of pose 𝑧𝑝 and content 𝑧𝑐. Then, the Immersive Drama Transformer generates each target speech 𝑦𝑝𝑟 from Gaussian noise 𝜖, conditioned on 𝑧𝑐, 𝑧𝑝, scene 𝑠, and 𝑎. To enhance generation quality and ensure contextual prosodic consistency, we design the context-consistent classifierfree guidance (CFG) strategy, which uses both prompt audio 𝑎 and the last predicted audio from the same speaker 𝑦𝑝𝑟−𝑙𝑎𝑠𝑡. During inference, we modify the output vector field as:

𝑣𝑐𝑓 𝑔(𝑥,𝑡|𝑧𝑝;𝜃) = 𝛾𝛼𝑣𝑡 (𝑥,𝑡|𝑎,𝐶;𝜃)+ 𝛾(1 − 𝛼)𝑣𝑡 (𝑥,𝑡|𝑦𝑝𝑟−𝑙𝑎𝑠𝑡,𝐶;𝜃) + (1 −𝛾)𝑣𝑡 (𝑥,𝑡|∅,𝐶;𝜃),

(5)

where𝛾 is the CFG scale that balances creativity and controllability, while𝛼 balances context consistency and controllability. Setting𝛾 = 3 and 𝛼 = 0.4, we improve generation quality and add previously generated audio for the prosody consistency of the same speaker within a single drama act. This ensures coherence while preserving the timbre and accent of the original prompt audio. Moreover, as prosody can be learned from previously generated audio in the same context, this approach also helps the modeling of semantically aligned prosody. For more details, please refer to Appendix C.3.

5 EXPERIMENTS

#### 5.1 Experimental Setup

Implementation Details. Mel-spectrograms are derived from raw binaural waveforms with a 48 kHz sample rate, 1024 window size, 256 hop size, and 80 mel bins. We use four Mamba-Transformer blocks, each of which includes three Mamba blocks. The flowmatching timestep is 1000 for training and 25 for inference with the Euler ODE solver. For the training procedure of Immersive Drama Transformer, we use 8 NVIDIA RTX-4090 GPUs with a batch size of 12K frames per GPU for 100K steps. The Adam optimizer is applied with a learning rate of 5 × 10−5, 𝛽1 = 0.9, 𝛽2 = 0.999, and 10K warm-up steps. Please refer to Appendix C.1 for more details.

Evaluation Metrics. We perform both subjective and objective evaluations on the generated samples. To ensure a fair comparison with existing monaural speech synthesis models, we adopt monaural evaluation metrics and then evaluate binaural metrics for cascade-generated binaural speech with BinauralGrad [39]. 1) For objective evaluation, we use Character Error Rate (CER) and Cosine Similarity (SIM) to assess the content accuracy and speaker similarity with prompt audio. F0 Frame Error (FFE) is used to evaluate the quality of prosody modeling. Since existing binaural metrics

- Table 2: Monaural speech quality comparison. For testing quality and speaker similarity, we evaluate single-sentence speech.

Method

Objective Metrics Subjective Metrics

CER ↓ SIM ↑ FFE ↓ MOS-Q ↑ MOS-S ↑ MOS-E ↑

Ground Truth 2.58% - - 4.43 ± 0.12 4.41 ± 0.07 4.26 ± 0.11 Uniaudio 4.21% 0.94 0.68 3.93 ± 0.12 4.06 ± 0.05 3.65 ± 0.18 StyleTTS 2 4.19% 0.93 0.60 3.89 ± 0.04 4.02 ± 0.10 3.72 ± 0.09 CoisyVoice 3.95% 0.96 0.56 4.05 ± 0.06 4.19 ± 0.09 3.81 ± 0.21 FireRedTTS 3.07 % 0.95 0.60 4.01 ± 0.17 4.11 ± 0.11 3.77 ± 0.10 F5-TTS 3.13% 0.96 0.55 4.12 ± 0.16 4.21 ± 0.08 3.86 ± 0.06 ISDrama (ours) 3.31% 0.96 0.34 4.06 ± 0.14 4.18 ± 0.11 4.01 ± 0.09

- Table 3: Binaural speech quality comparison. We evaluate complete drama. ANG and Dis denote angle and distance. Spatialization refers to the generation of binaural audio directly from the GT monaural audio input based on the geometric pose.

Objective Metrics Subjective Metrics

Method

IPD MAE ↓ ILD MAE ↓ ANG Cos ↑ DIS Cos ↑ MOS-Q ↑ MOS-P ↑

Spatialization 0.007 0.043 0.58 0.79 4.09 ± 0.08 4.26 ± 0.16 Uniaudio 0.012 0.060 0.38 0.64 3.65 ± 0.16 3.89 ± 0.11 StyleTTS2 0.011 0.064 0.33 0.61 3.63 ± 0.05 3.81 ± 0.16 CoisyVoice 0.011 0.055 0.44 0.68 3.72 ± 0.13 4.02 ± 0.09 FireRedTTS 0.010 0.051 0.42 0.65 3.78 ± 0.19 3.97 ± 0.17 F5-TTS 0.009 0.053 0.45 0.70 3.66 ± 0.14 4.11 ± 0.13 ISDrama (geometric) 0.008 0.046 0.51 0.75 4.01 ± 0.14 4.18 ± 0.10 ISDrama (video) 0.009 0.051 0.48 0.73 3.97 ± 0.11 4.09 ± 0.08 ISDrama (textual) 0.011 0.055 0.43 0.68 3.95 ± 0.13 4.41 ± 0.06

are scarce and not suitable for our one-stage binaural speech generation task, we have designed several new metrics. We extract Interaural Phase Difference (IPD) and Interaural Level Difference (ILD) from the binaural mel-spectrograms and compute MAE with GT. We also compute cosine similarity of angle and distance embedding extracted from SPATIAL-AST [76] with GT for spatial evaluation. 2) For subjective evaluation, we conduct Mean Opinion Score (MOS), which is rated from 1 to 5 and reported with 95% confidence intervals. MOS-Q evaluates the synthesized quality (like naturalness, spatial perception, and coherence), MOS-S assesses speaker similarity in timbre and accent, and MOS-E measures the expressiveness of semantically aligned prosody. For binaural metrics, we employ MOS-P to evaluate the pose consistency between the multimodal prompt and the generated audio. In the ablation study, we conduct Comparative Mean Opinion Score (CMOS). For more evaluation details, please refer to Appendix D.

#### 5.2 Results

Comparison of baseline models for monaural speech. To ensure a fair evaluation of the synthesized quality, speaker similarity to the prompt audio, and prosodic expressiveness, we compute monaural speech metrics after averaging the binaural speech generated by ISDrama across channels. For the baseline models, we

employ several strong popular open-source speech synthesis models, including: 1) Uniaudio [66], 2) StyleTTS 2 [44], 3) CosyVoice [15], 4) FireRedTTS [25], and 5) F5-TTS [10]. We use their open-source codes on GitHub and train them using our MRSDrama dataset in monaural speech format. We also provide a detailed introduction for each baseline model in Appendix E.1. To achieve a more precise evaluation of quality and speaker similarity, we conduct the assessment using only single sentences for all models. Table 2 shows that ISDrama performs well across all monaural metrics. In synthesized quality (CER, MOS-Q) and speaker similarity (SIM, MOS-S), it is comparable to the performance of the best speech synthesis baseline model, while it surpasses all baseline models in prosodic expressiveness (FFE, MOS-E). These well-performing results can be attributed to the integration of semantics to model dramatic prosody with semantically aligned emotion and rhythm in Drama-MOE, as well as F0 supervision during training.

Comparison of Baseline Models for Binaural Speech. Based on the subjective and objective spatial metrics we designed, we train a commonly used monaural-to-binaural spatialization model, BinauralGrad [39], on our MRSDrama dataset to convert the generated monaural speech from baseline models, conditioned on geometric poses, into binaural speech for evaluating spatial performance. To better test the coherence of spatial information and the consistency of prosody, we concatenate sentences from the same generated drama for evaluation. As shown in Table 3, with the same

Table 4: Ablation studies on Immersive Drama Transformer. MPE denotes Multimodal Pose Encoder.

Objective Metrics Subjective Metrics

Method

CER ↓ FFE ↓ ANG Cos ↑ DIS Cos ↑ CMOS-Q ↑ CMOS-S ↑ CMOS-E ↑ CMOS-P ↑

ISDrama (Geometric) 3.31% 0.34 0.51 0.75 0.00 0.00 0.00 0.00 Geometric w/o MPE 3.35% 0.37 0.49 0.73 -0.11 -0.03 -0.10 -0.16 Video w/o MPE 3.68% 0.41 0.38 0.63 -0.39 -0.19 -0.29 -0.52 Textual w/o MPE 3.72% 0.44 0.35 0.61 -0.50 -0.20 -0.32 -0.57 w/o Drama-MOE 4.01% 0.49 0.39 0.65 -0.46 -0.27 -0.40 -0.52 w/o Prosodic-MOE 3.55% 0.47 0.47 0.71 -0.28 -0.21 -0.36 -0.46 w/o Spatial-MOE 3.48% 0.39 0.41 0.66 -0.27 -0.12 -0.21 -0.28 w/o FAN 3.79% 0.46 0.40 0.66 -0.43 -0.24 -0.32 -0.39 w/o F0 pred 3.65% 0.46 0.46 0.69 -0.36 -0.19 -0.41 -0.48 w/o Mamba 3.84% 0.49 0.43 0.65 -0.30 -0.08 -0.27 -0.39 w/o CFG 3.73% 0.48 0.45 0.68 -0.38 -0.22 -0.32 -0.38

geometric pose input, the binaural speech generated by ISDrama outperforms all two-stage models across all spatial metrics. The high MOS-Q shows coherence in continuous multi-speaker speech generation, which can be attributed to the effectiveness of our context-consistent CFG in generating complete drama. In more flexible scenarios, including video and textual prompt inputs, ISDrama continues to deliver excellent performance, thanks to the unified pose representation extracted by the Multimodal Pose Encoder. Although the textual prompt leads to lower objective metrics, it achieves a strong MOS-P score, which can be attributed to the general nature of the text descriptions. Specifically, the same textual description can generate multiple pose sequences, resulting in spatial information not fully aligned with the ground truth.

#### 5.3 Ablation Study

Multimodal Pose Encoder. Tables 3 and 4 present the results of feeding pose embeddings of different prompts into the Immersive Drama Transformer, with or without contrastive learning. We observe that the absence of contrastive learning has minimal impact on geometric poses. This is because the geometric pose embeddings include 3D orientation, quaternion orientation, and radial velocity, which provide sufficient information for accurate pose modeling. However, for silent video and textual prompts, omitting contrastive learning significantly reduces spatial performance. This highlights the importance of contrastive learning, as it enhances the model’s ability to generate unified embeddings from multimodal prompts, which is crucial for supporting a wide range of diverse and flexible immersive spatial drama application scenarios.

Drama-MOE. Table 4 shows the results of experiments where we removed the full Drama-MOE, eliminated individual expert groups, and replaced the FAN module with a simple linear layer. The results reveal that removing the full Drama-MOE leads to a decline in performance across all metrics. When examining the individual expert groups, we find that Spatial-MOE significantly affects spatial performance, while Prosodic-MOE influences both prosody expressiveness and speaker similarity. These observations suggest that Drama-MOE plays a key role in enhancing the modeling of

both prosody and pose by utilizing specialized experts tailored to different spatial conditions and semantically aligned prosody. Additionally, we find that the FAN module outperforms the simple linear layer in all aspects, reflecting the benefits brought by frequency decomposition in capturing more nuanced features.

Immersive Drama Transformer. We evaluate the effects of removing the supervision of F0, replacing Mamba with a memoryequivalent Transformer, and eliminating the context-consistent CFG. The results are shown in Table 4. It can be observed that F0 prediction is crucial for accurate prosody modeling, as its absence results in a significant loss of prosodic quality. The Mamba block, being lightweight, allows stacking more layers than a traditional Transformer with the same memory budget, which leads to further improvements in quality. Additionally, removing the contextconsistent CFG reduces the consistency and prosodic expressiveness of the generated outputs. These findings underscore the effectiveness of these design choices in enhancing quality and prosody modeling capabilities of the Immersive Drama Transformer.

#### 6 CONCLUSION

In this paper, we introduce a novel task, Multimodal Immersive Spatial Drama Generation, focusing on creating continuous multispeaker binaural speech with dramatic prosody based on multimodal prompts. To support this task, we present MRSDrama, the first multimodal recorded spatial drama dataset, comprising binaural drama audios, scripts, videos, geometric poses, and textual prompts. Then, we propose ISDrama, the first immersive spatial drama generation model based on multimodal prompting. To extract a unified pose representation from multimodal prompts, we design the Multimodal Pose Encoder, a contrastive learning-based framework. To generate immersive spatial drama effectively and stably, we develop the Immersive Drama Transformer, a flow-based Mamba-Transformer model, incorporating Drama-MOE, which selects proper experts to enhance prosodic expressiveness and pose control. Then, we design a context-consistent CFG strategy to coherently generate complete drama. Experimental results show that ISDrama achieves better performance than baseline models.

#### ACKNOWLEDGEMENTS

This work was supported by the National Key R&D Program of China (2022ZD0162000) and the National Natural Science Foundation of China under Grant No.U24A20326.

#### REFERENCES

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774

(2023).

- [2] Thomas Bachlechner, Bodhisattwa Prasad Majumder, Henry Mao, Gary Cottrell, and Julian McAuley. 2021. Rezero is all you need: Fast convergence at large depth. In Uncertainty in Artificial Intelligence. PMLR, 1352–1361.
- [3] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems 33 (2020), 12449–12460.
- [4] Nancy Bertin, Ewen Camberlein, Emmanuel Vincent, Romain Lebarbenchon, Stéphane Peillon, Éric Lamandé, Sunit Sivasankaran, Frédéric Bimbot, Irina Illina, Ariane Tom, et al. 2016. A French corpus for distant-microphone speech processing in real homes. In Interspeech 2016.
- [5] Paul Boersma. 2001. Praat, a system for doing phonetics by computer. Glot. Int. 5, 9 (2001), 341–345.
- [6] XIE Bosun. 2020. Spatial Sound—History, Principle, Progress and Challenge. Chinese Journal of Electronics 29, 3 (2020), 397–416. https://doi.org/10.1049/cje. 2020.02.016
- [7] Jiawei Chen, Xu Tan, Jian Luan, Tao Qin, and Tie-Yan Liu. 2020. Hifisinger: Towards high-fidelity neural singing voice synthesis. arXiv preprint arXiv:2009.01776

(2020).

- [8] Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al. 2022. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing 16, 6 (2022), 1505–1518.
- [9] Victor C. Chen, F. Li, Shen-Shyang Ho, and Harry Wechsler. 2006. Micro-Doppler effect in radar: phenomenon, model, and simulation study. IEEE Trans. Aerospace Electron. Systems 42 (2006), 2–21.
- [10] Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, Jian Zhao, Kai Yu, and Xie Chen. 2024. F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching. arXiv preprint arXiv:2410.06885 (2024).
- [11] Seungwoo Choi, Seungju Han, Dongyoung Kim, and Sungjoo Ha. 2020. Attentron: Few-shot text-to-speech utilizing attention-based variable-length embedding. arXiv preprint arXiv:2005.08484 (2020).
- [12] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2024. Scaling instruction-finetuned language models. Journal of Machine Learning Research 25, 70 (2024), 1–53.
- [13] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).
- [14] Yihong Dong, Ge Li, Yongding Tao, Xue Jiang, Kechi Zhang, Jia Li, Jing Su, Jun Zhang, and Jingjing Xu. 2024. FAN: Fourier Analysis Networks. arXiv preprint arXiv:2410.02675 (2024).
- [15] Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, et al. 2024. Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. arXiv preprint arXiv:2407.05407 (2024).
- [16] William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research 23, 120 (2022), 1–39.
- [17] Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Youqiang Zhang, and Junshi Huang. 2024. Dimba: Transformer-Mamba Diffusion Models. arXiv preprint arXiv:2406.01159 (2024).
- [18] Tira Nur Fitria. 2023. Augmented reality (AR) and virtual reality (VR) technology in education: Media of teaching and learning: A review. International Journal of Computer and Information System (IJCIS) 4, 1 (2023), 14–25.
- [19] Ruohan Gao and Kristen Grauman. 2019. 2.5 d visual sound. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 324–333.
- [20] Zhifu Gao, Shiliang Zhang, Ian McLoughlin, and Zhijie Yan. 2022. Paraformer: Fast and accurate parallel transformer for non-autoregressive end-to-end speech recognition. arXiv preprint arXiv:2206.08317 (2022).
- [21] Guillermo García-Barrios, Daniel Aleksander Krause, Archontis Politis, Annamaria Mesaros, Juana M Gutiérrez-Arriola, and Rubén Fraile. 2022. Binaural source localization using deep learning and head rotation information. In 2022 30th European Signal Processing Conference (EUSIPCO). IEEE, 36–40.
- [22] Rishabh Garg, Ruohan Gao, and Kristen Grauman. 2021. Geometry-aware multi-task learning for binaural audio generation from video. arXiv preprint

- arXiv:2111.10882 (2021).
- [23] Thomas Perrott Gill. 1965. The Doppler effect: An introduction to the theory of the effect. (No Title) (1965).
- [24] Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 (2023).
- [25] Hao-Han Guo, Kun Liu, Fei-Yu Shen, Yi-Chen Wu, Feng-Long Xie, Kun Xie, and Kai-Tuo Xu. 2024. Fireredtts: A foundation text-to-speech framework for industry-level generative speech applications. arXiv preprint arXiv:2409.03283

(2024).

- [26] Wenxiang Guo, Yu Zhang, Changhao Pan, Rongjie Huang, Li Tang, Ruiqi Li, Zhiqing Hong, Yongqi Wang, and Zhou Zhao. 2025. TechSinger: Technique Controllable Multilingual Singing Voice Synthesis via Flow Matching. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 23978–23986.
- [27] Mojtaba Heydari, Mehrez Souden, Bruno Conejo, and Joshua Atkins. 2025. ImmerseDiffusion: A Generative Spatial Audio Latent Diffusion Model. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 1–5.
- [28] Rongjie Huang, Yi Ren, Jinglin Liu, Chenye Cui, and Zhou Zhao. 2022. Generspeech: Towards style transfer for generalizable out-of-domain text-to-speech synthesis. arXiv preprint arXiv:2205.07211 (2022).
- [29] Xu Huiyu, Jin Shuaifan, Wang Zhibo, Ba Zhongjie, and Wei Tao. 2025. PSANeRF: Personalized Spatial Attention Neural Rendering for Audio-Driven Talking Portraits Generation. Chinese Journal of Electronics 34 (2025), 1–12. https: //doi.org/10.23919/cje.2024.00.095
- [30] Eric Jang, Shixiang Gu, and Ben Poole. 2016. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144 (2016).
- [31] Zeqian Ju, Yuancheng Wang, Kai Shen, Xu Tan, Detai Xin, Dongchao Yang, Yanqing Liu, Yichong Leng, Kaitao Song, Siliang Tang, et al. 2024. Naturalspeech 3: Zero-shot speech synthesis with factorized codec and diffusion models. arXiv preprint arXiv:2403.03100 (2024).
- [32] Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. 2024. CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos. arXiv preprint arXiv:2410.11831 (2024).
- [33] Jaeyeon Kim, Heeseung Yun, and Gunhee Kim. 2025. ViSAGe: Video-to-Spatial Audio Generation. In The Thirteenth International Conference on Learning Representations.
- [34] Diederik P Kingma and Max Welling. 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013).
- [35] Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. 2020. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. Advances in neural information processing systems 33 (2020), 17022–17033.
- [36] Daniel Aleksander Krause, Guillermo García-Barrios, Archontis Politis, and Annamaria Mesaros. 2023. Binaural sound source distance estimation and localization for a moving listener. IEEE/ACM Transactions on Audio, Speech, and Language Processing (2023).
- [37] Neeraj Kumar, Srishti Goel, Ankur Narang, and Brejesh Lall. 2021. Normalization Driven Zero-Shot Multi-Speaker Speech Synthesis.. In Interspeech. 1354–1358.
- [38] Sang-Hoon Lee, Ha-Yeong Choi, Seung-Bin Kim, and Seong-Whan Lee. 2023. Hierspeech++: Bridging the gap between semantic and acoustic representation of speech by hierarchical variational inference for zero-shot speech synthesis. arXiv preprint arXiv:2311.12454 (2023).
- [39] Yichong Leng, Zehua Chen, Junliang Guo, Haohe Liu, Jiawei Chen, Xu Tan, Danilo Mandic, Lei He, Xiangyang Li, Tao Qin, et al. 2022. Binauralgrad: A two-stage conditional diffusion probabilistic model for binaural audio synthesis. Advances in Neural Information Processing Systems 35 (2022), 23689–23700.
- [40] Kai Li, Wendi Sang, Chang Zeng, Runxuan Yang, Guo Chen, and Xiaolin Hu.

2024. SonicSim: A customizable simulation platform for speech processing in moving sound source scenarios. arXiv preprint arXiv:2410.01481 (2024).

- [41] Ruiqi Li, Yu Zhang, Yongqi Wang, Zhiqing Hong, Rongjie Huang, and Zhou Zhao. 2024. Robust singing voice transcription serves synthesis. arXiv preprint arXiv:2405.09940 (2024).
- [42] Xiang Li, Changhe Song, Jingbei Li, Zhiyong Wu, Jia Jia, and Helen Meng. 2021. Towards multi-scale style control for expressive speech synthesis. arXiv preprint arXiv:2104.03521 (2021).
- [43] Yinghao Aaron Li, Cong Han, and Nima Mesgarani. 2022. Styletts: A style-based generative model for natural and diverse text-to-speech synthesis. arXiv preprint arXiv:2205.15439 (2022).
- [44] Yinghao Aaron Li, Cong Han, Vinay Raghavan, Gavin Mischler, and Nima Mesgarani. 2024. Styletts 2: Towards human-level text-to-speech through style diffusion and adversarial training with large speech language models. Advances in Neural Information Processing Systems 36 (2024).
- [45] Zhaojian Li, Bin Zhao, and Yuan Yuan. 2024. Cyclic Learning for Binaural Audio Generation and Localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 26669–26678.
- [46] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. 2022. Flow Matching for Generative Modeling. In The Eleventh International Conference on Learning Representations.

- [47] Jinglin Liu, Zhenhui Ye, Qian Chen, Siqi Zheng, Wen Wang, Qinglin Zhang, and Zhou Zhao. 2022. DopplerBAS: Binaural Audio Synthesis Addressing Doppler Effect. arXiv preprint arXiv:2212.07000 (2022).
- [48] Xingchao Liu, Chengyue Gong, et al. 2022. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In The Eleventh International Conference on Learning Representations.
- [49] Xudong Mao, Qing Li, Haoran Xie, Raymond YK Lau, Zhen Wang, and Stephen Paul Smolley. 2017. Least squares generative adversarial networks. In Proceedings of the IEEE international conference on computer vision. 2794–2802.
- [50] Michael McAuliffe, Michaela Socolof, Sarah Mihuc, Michael Wagner, and Morgan Sonderegger. 2017. Montreal forced aligner: Trainable text-speech alignment using kaldi.. In Interspeech, Vol. 2017. 498–502.
- [51] Xiaonan Nie, Xupeng Miao, Shijie Cao, Lingxiao Ma, Qibin Liu, Jilong Xue, Youshan Miao, Yi Liu, Zhi Yang, and Bin Cui. 2021. Evomoe: An evolutional mixture-of-experts training framework via dense-to-sparse gate. arXiv preprint arXiv:2112.14397 (2021).
- [52] Kranti Kumar Parida, Siddharth Srivastava, and Gaurav Sharma. 2022. Beyond mono to binaural: Generating binaural audio from mono audio with depth and cross modal attention. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 3347–3356.
- [53] William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 4195–4205.
- [54] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.
- [55] Mirco Ravanelli, Luca Cristoforetti, Roberto Gretter, Marco Pellin, Alessandro Sosi, and Maurizio Omologo. 2015. The DIRHA-English corpus and related tasks for distant-speech recognition in domestic environments. In 2015 IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU). IEEE, 275–282.
- [56] Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. 2019. Fastspeech: Fast, robust and controllable text to speech. Advances in neural information processing systems 32 (2019).
- [57] Alexander Richard, Peter Dodds, and Vamsi Krishna Ithapu. 2022. Deep impulse responses: Estimating and parameterizing filters with deep networks. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 3209–3213.
- [58] Alexander Richard, Dejan Markovic, Israel D Gebru, Steven Krenn, Gladstone Alexander Butler, Fernando Torre, and Yaser Sheikh. 2021. Neural synthesis of binaural speech from mono audio. In International Conference on Learning Representations.
- [59] Miguel Sarabia, Elena Menyaylenko, Alessandro Toso, Skyler Seto, Zakaria Aldeneh, Shadi Pirhosseinloo, Luca Zappella, Barry-John Theobald, Nicholas Apostoloff, and Jonathan Sheaffer. 2023. Spatial Librispeech: An augmented dataset for spatial audio learning. arXiv preprint arXiv:2308.09514 (2023).
- [60] Kazuki Shimada, Yuichiro Koyama, Shusuke Takahashi, Naoya Takahashi, Emiru Tsunoo, and Yuki Mitsufuji. 2022. Multi-accdoa: Localizing and detecting overlapping sounds from the same class with auxiliary duplicating permutation invariant training. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 316–320.
- [61] RJ Skerry-Ryan, Eric Battenberg, Ying Xiao, Yuxuan Wang, Daisy Stanton, Joel Shor, Ron Weiss, Rob Clark, and Rif A Saurous. 2018. Towards end-to-end prosody transfer for expressive speech synthesis with tacotron. In international conference on machine learning. PMLR, 4693–4702.
- [62] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568 (2024), 127063.
- [63] Peiwen Sun, Sitong Cheng, Xiangtai Li, Zhen Ye, Huadai Liu, Honggang Zhang, Wei Xue, and Yike Guo. 2024. Both Ears Wide Open: Towards Language-Driven Spatial Audio Generation. arXiv preprint arXiv:2410.10676 (2024).
- [64] Michel Vacher, Benjamin Lecouteux, Pedro Chahuara, François Portet, Brigitte Meillon, and Nicolas Bonnefond. 2014. The Sweet-Home speech and multimodal corpus for home automation interaction. In The 9th edition of the Language Resources and Evaluation Conference (LREC). 4499–4506.
- [65] Michael Wagner and Duane G Watson. 2010. Experimental and theoretical advances in prosody: A review. Language and cognitive processes 25, 7-9 (2010), 905–945.
- [66] Dongchao Yang, Jinchuan Tian, Xu Tan, Rongjie Huang, Songxiang Liu, Xuankai Chang, Jiatong Shi, Sheng Zhao, Jiang Bian, Xixin Wu, et al. 2023. Uniaudio: An audio foundation model toward universal audio generation. arXiv preprint arXiv:2310.00704 (2023).
- [67] Qiang Yang and Yuanqing Zheng. 2022. Deepear: Sound localization with binaural microphones. IEEE Transactions on Mobile Computing 23, 1 (2022), 359–375.
- [68] Julian Zaıdi, Hugo Seuté, BV Niekerk, and M Carbonneau. 2021. Daft-exprt: Robust prosody transfer across speakers for expressive speech synthesis. arXiv preprint arXiv:2108.02271 (2021).

- [69] Biao Zhang and Rico Sennrich. 2019. Root mean square layer normalization. Advances in Neural Information Processing Systems 32 (2019).
- [70] Lichao Zhang, Ruiqi Li, Shoutong Wang, Liqun Deng, Jinglin Liu, Yi Ren, Jinzheng He, Rongjie Huang, Jieming Zhu, Xiao Chen, et al. 2022. M4singer: A multi-style, multi-singer and musical score provided mandarin singing corpus. Advances in Neural Information Processing Systems 35 (2022), 6914–6926.
- [71] Yu Zhang, Wenxiang Guo, Changhao Pan, Zhiyuan Zhu, Ruiqi Li, Jingyu Lu, Rongjie Huang, Ruiyuan Zhang, Zhiqing Hong, Ziyue Jiang, and Zhou Zhao.

2025. Versatile Framework for Song Generation with Prompt-based Control. arXiv preprint arXiv:2504.19062 (2025).

- [72] Yu Zhang, Rongjie Huang, Ruiqi Li, JinZheng He, Yan Xia, Feiyang Chen, Xinyu Duan, Baoxing Huai, and Zhou Zhao. 2024. StyleSinger: Style Transfer for Outof-Domain Singing Voice Synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 19597–19605.
- [73] Yu Zhang, Ziyue Jiang, Ruiqi Li, Changhao Pan, Jinzheng He, Rongjie Huang, Chuxin Wang, and Zhou Zhao. 2024. TCSinger: Zero-Shot Singing Voice Synthesis with Style Transfer and Multi-Level Style Control. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 1960–1975.
- [74] Yu Zhang, Changhao Pan, Wenxiang Guo, Ruiqi Li, Zhiyuan Zhu, Jialei Wang, Wenhao Xu, Jingyu Lu, Zhiqing Hong, Chuxin Wang, et al. 2024. Gtsinger: A global multi-technique singing corpus with realistic music scores for all singing tasks. arXiv preprint arXiv:2409.13832 (2024).
- [75] Shengkui Zhao, Bin Ma, Karn N Watcharasupat, and Woon-Seng Gan. 2022. FRCRN: Boosting feature representation using frequency recurrence for monaural speech enhancement. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 9281–9285.
- [76] Zhisheng Zheng, Puyuan Peng, Ziyang Ma, Xie Chen, Eunsol Choi, and David Harwath. 2024. BAT: Learning to Reason about Spatial Sounds with Large Language Models. arXiv preprint arXiv:2402.01591 (2024).
- [77] Hang Zhou, Xudong Xu, Dahua Lin, Xiaogang Wang, and Ziwei Liu. 2020. Sepstereo: Visually guided stereophonic audio generation by associating source separation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16. Springer, 52–69.

A PRELIMINARIES

- A.1 Doppler Effect

The Doppler effect [23] refers to the change in the frequency of a wave as observed by an observer when the source of the wave is in motion relative to it. Initially applied in radar systems, this effect helps to analyze the features of moving objects, such as targets of interest [9]. The Doppler effect can be expressed as:

𝑓𝑜 =

𝑐 𝑐 ± 𝑣𝑟𝑎𝑑

𝑓𝑠, (6)

where 𝑐 is the wave propagation speed, 𝑣𝑟𝑎𝑑 is the radial velocity of the moving sound source, 𝑓𝑠 is the source frequency, and 𝑓𝑜 is the frequency received by the observer.

- A.2 Rectified flow-matching In generative modeling, the true data distribution is denoted as 𝑞(𝑥1), which can be sampled but lacks an accessible density function. Consider a probability path 𝑝𝑡 (𝑥𝑡), where 𝑥0 ∼ 𝑝0(𝑥) represents a standard Gaussian, and 𝑥1 ∼ 𝑝1(𝑥) approximates the real data distribution. The core of the flow-matching approach [48] is to model this path directly, which is governed by the following ordinary differential equation (ODE):

d𝑥 = 𝑢(𝑥,𝑡)d𝑡, 𝑡 ∈ [0, 1], (7)

where𝑢(𝑥,𝑡) denotes the target vector field, and 𝑡 is the time index. If we have access to the vector field 𝑢, it is possible to recover realistic data by reversing the flow. To approximate 𝑢, we use a vector field estimator𝑣(·), and the flow-matching objective function is:

LFM(𝜃) = E𝑡,𝑝𝑡 (𝑥) ∥𝑣(𝑥,𝑡;𝜃) −𝑢(𝑥,𝑡)∥2 , (8) where 𝑝𝑡 (𝑥) denotes the distribution of 𝑥 at time 𝑡. When conditioning on additional 𝐶, the objective becomes the conditional flow-matching formulation [46]:

LCFM(𝜃) = E𝑡,𝑝1(𝑥1),𝑝𝑡 (𝑥|𝑥1) ∥𝑣(𝑥,𝑡|𝐶;𝜃) −𝑢(𝑥,𝑡|𝑥1,𝐶)∥2 .

(9) The key idea behind flow-matching is to construct a direct transformation path from Gaussian noise to real data. This is achieved by linearly interpolating between Gaussian noise 𝑥0 and the real data 𝑥1 to generate samples at time 𝑡:

𝑥𝑡 = (1 − 𝑡)𝑥0 + 𝑡𝑥1. (10)

Thus, the conditional vector field becomes 𝑢(𝑥,𝑡|𝑥1,𝐶) = 𝑥1 − 𝑥0, and the rectified flow-matching (RFM) loss used for optimization is:

∥𝑣(𝑥,𝑡|𝐶;𝜃) − (𝑥1 − 𝑥0)∥2 . (11) If the vector field 𝑢 is estimated correctly, realistic data can be generated by passing Gaussian noise through an ODE solver at discrete time steps. One effective method to solve the reverse flow is the Euler ODE:

𝑥𝑡+𝜖 = 𝑥 + 𝜖𝑣(𝑥,𝑡|𝐶;𝜃), (12)

where 𝜖 is the step size. Flow-matching models typically require hundreds to thousands of training steps. However, the efficient linear interpolation approach significantly reduces this to 25 steps or fewer during inference, greatly enhancing computational efficiency. Moreover, the seamless transition from noise to data ensures both

stability and high-quality output, which is essential for generating complex data free of artifacts while maintaining consistency across diverse input conditions.

#### A.3 Mamba

Models based on Structured State Space (SSM), such as S4 and Mamba [24], draw inspiration from continuous systems that map a 1D sequence 𝑥(𝑡) ∈ R to another sequence 𝑦(𝑡) ∈ R through a hidden state ℎ(𝑡) ∈ RN. In this system, the matrix A ∈ RN×N serves

- as the state evolution parameter, while B ∈ RN×1 and C ∈ R1×N are used for projection. The continuous system is:

ℎ′(𝑡) = Aℎ(𝑡) + B𝑥(𝑡), 𝑦(𝑡) = Cℎ(𝑡).

(13)

S4 and Mamba are discrete-time versions of this continuous system, incorporating a timescale parameter ∆ to convert the continuous parameters A and B into discrete counterparts A and B. A typical approach to perform this transformation is through zero-order hold (ZOH), which is:

- A = exp (∆A),

- B = (∆A)−1(exp (∆A) − I) · ∆B.

(14)

After discretizing A and B, the discrete-time system can be written as:

ℎ𝑡 = Aℎ𝑡−1 + B𝑥𝑡, 𝑦𝑡 = Cℎ𝑡.

(15)

Finally, these models obtain the output through a global convolution operation, represented as:

K = (CB, CAB, . . ., CAM−1B), y = x ∗ K,

(16)

where M denotes the length of the input sequence x, and K ∈ RM represents the structured convolutional kernel.

B DATASET DETAILS

Simulated data cannot accurately capture the intricate, dramatic prosody or the precise effects of real-world spatial scenes, positions, and orientations. Meanwhile, stimulated spatial audio datasets that expand on monaural audio data also have limited effectiveness in modeling the nuances of real-world scenes, positions, and orientations [40, 59].

As shown in Figure 6, we utilize the 3Dio FS XLR Binaural Microphone1 connected to a Yamaha professional sound card to record binaural audio, effectively modeling interaural phase differences (IPD) and interaural level differences (ILD). The recordings are complemented by 24fps video captured using a camera. The scripts used in our study are sourced from authorized materials. For the recording process, we design 5 to 12 fixed positions in each scene, covering various distances and angles. However, the routes and speeds are not predetermined, allowing speakers to move freely. Speakers may stand and walk through these points or sit on a chair

- at a designated position, which results in height variations. This flexibility also aids in contrastive learning. Speakers read from a script displayed on a screen that is not visible in the camera frame, 1https://3diosound.com/products/free-space-xlr-binaural-microphone

[Figure 48]

[Figure 49]

### (a) Movement Speed (b) Segment Duration

Figure 5: The extended statistics of MRSDrama.

speaker transitions, with each segment having a maximum duration of 16 seconds. Figure 5 shows more statistics of MRSDrama on movement speeds and segment duration.

[Figure 50]

[Figure 51]

#### C MODEL DESIGN C.1 Model Configuration

For the Multimodal Pose Encoder, we use a hidden size of 768. Our semantic information is modeled using BERT (base) [13], which preserves temporal length. The duration decoder adopts the FastSpeech architecture [56]. For the geometric encoder, we employ a Conv1D layer with a kernel size of 5. For the video and textual encoders, the Transformer model utilizes 8 transformer layers and 8 attention heads, with a hidden size of 768. The total number of parameters is 23.32M for each model. We obtain CLIP embeddings at 4 frames per second (FPS). During training, we use the Adam optimizer with a learning rate of 1 × 10−4, 𝛽1 = 0.9, 𝛽2 = 0.999, and 10K warm-up steps.

Figure 6: The binaural recording device we used.

with the requirement that the script has semantically aligned dramatic prosody. The recording of both audio and video for each dramatic act typically lasts between 3 and 15 minutes. We hire the speakers at a rate of $100 per hour, and they agree to make the video and audio data available for research purposes. We assure them that no facial information will be disclosed and then apply masks to their faces in the video using Adobe After Effects.

For the spatial audio encoder and decoder, we use a Variational Autoencoder (VAE) architecture [34]. Mel-spectrograms are derived from binaural waveforms with a 48 kHz sample rate, 1024 window size, 256 hop size, and 80 mel bins. We use HiFi-GAN [35] as the vocoder to synthesize waveforms from mel-spectrograms. Specifically, the model consists of 3 layers for both the encoder and decoder, with a hidden size of 384 and a Conv1D kernel size of 5. The binaural mel-spectrogram with dimensions 𝐵, 2, 80,𝑇 is compressed into dimensions 𝐵, 40,𝑇/4, facilitating further processing by the Transformer. During training, batches of fixed length are used, consisting of 3000 mel-spectrogram frames. The Adam optimizer is employed with a learning rate of 1 × 10−4, 𝛽1 = 0.9, 𝛽2 = 0.999, and 10K warm-up steps.

Annotators label the 3D coordinates and quaternion orientations of the sound source (the speaker’s mouth) based on the video. Using the annotated pose data, we employ GPT-4o [1] to generate textual prompts, such as: "Speaker 1 initially stands in the front-right, then walks at a slow pace to the far front-left..." For binaural speech, we perform denoising using FRCRN [75] and extract phonemes using PyPinyin. Following previous Chinese audio annotation works [26, 41, 70, 72–74]. Coarse alignment between phonemes and audio is achieved using MFA [50]. Annotators then refine the rough alignment using Praat [5], focusing on correcting word and phoneme boundaries and addressing erroneous words based on auditory perception. Each step is double-checked by human annotators to ensure accuracy. We hire annotators at a rate of $30 per hour, and they consent to their contributions being used for research purposes. Finally, the dataset is segmented into 47,958 segments according to

For the Immersive Drama Transformer, we employ four MambaTransformer blocks. Each block uses a hidden size of 768 and 8 attention heads. The Mixture-of-Experts (MoE) module includes 4 experts per expert group. Each Transformer has three Mamba blocks. The total number of parameters is 177 M. The flow-matching training uses 1,000 timesteps, while inference employs 25 timesteps

with the Euler ODE solver. During training, we use 8 NVIDIA RTX4090 GPUs with a batch size of 12K frames per GPU for 100K steps. The Adam optimizer is applied with a learning rate of 5 × 10−5, 𝛽1 = 0.9, 𝛽2 = 0.999, and 10K warm-up steps.

- C.2 Training Procedure As detailed in Section 4.2, the final loss for the Multimodal Pose Encoder includes: 1) L𝑐𝑜𝑛𝑡𝑟𝑎𝑠: the contrastive objective for three

modalities. 2) L𝑑𝑢𝑟: the mean squared error (MSE) duration loss between the predicted and ground truth phoneme-level durations on a logarithmic scale.

In Appendix C.5, the final loss terms for the spatial audio encoder and accomp decoder are as follows: 1) L𝑟𝑒𝑐: the L2 reconstruction loss of mel-spectrograms. 2) L𝑎𝑑𝑣: the LSGAN-styled adversarial loss for the GAN discriminator.

As for Immersive Drama Transformer, the final loss terms during training include: 1) L𝑓 𝑙𝑜𝑤: the flow matching loss, as described in Section 4.3. 2) L𝑝𝑖𝑡𝑐ℎ: the MSE pitch loss between the predicted and ground truth f0 in the log scale. 3) L𝑏𝑎𝑙𝑎𝑛𝑐𝑒: the load-balancing loss for each expert group in Drama-MOE, as discussed in Section C.7.

- C.3 Inference Procedure

For enhanced generation quality and contextual prosody consistency, we utilize the prompt audio 𝑎 and the last predicted audio from the same speaker,𝑦𝑝𝑟−𝑙𝑎𝑠𝑡, using the Classifier-Free Guidance (CFG) strategy. During training, we randomly select prompt audio from the same speaker within the same script to improve generalization, with a 0.4 probability of selecting other audio from the same speaker and a 0.2 probability of dropping the prompt audio entirely. During inference, we modify the output vector field as in Equation 5. When 𝛼 = 1, 𝑣𝑐𝑓 𝑔 relies solely on the input prompt audio. Furthermore, if 𝛼 = 1 and 𝛾 = 1, 𝑣𝑐𝑓 𝑔 is equivalent to the original 𝑣𝑡 (𝑥,𝑡|𝑎,𝐶;𝜃). Setting 𝛾 = 3 and 𝛼 = 0.4, we improve generation quality and incorporate previously generated audio to enhance the prosody consistency of the same speaker within a single drama act. This ensures coherence while preserving the timbre, accent, and articulation of the original prompt audio. Moreover, as prosody can be learned from previous prompt audio in the same context, this method further enhances prosodic expressiveness and alignment with the drama narrative.

- C.4 Multimodal Pose Encoder

For inputs from video and geometric pose, we need to segment the length of each speaker transition. In the case of geometric pose, a sudden jump in position indicates a speaker switch, allowing us to determine the total duration of each actor’s line. For video inputs, we also input the pixel coordinates of the speaker’s lips for the onset of each speech and the corresponding starting frame. This information is utilized by Cotracker3 [32] to track the speaker’s mouth movements, facilitating the modeling of 3D position coordinates and quaternion orientation. This process also helps in accurately determining the total duration of each actor’s line. For duration prediction, we estimate the phoneme durations within each actor’s line based on speaker transitions in the script. In addition to encoding semantic and phonetic information as inputs to the duration

predictor, we incorporate the length from the video segment and the pose sequence as constraints for the actor’s line duration. This ensures a highly precise and consistent duration prediction.

We design three types of contrasts for contrastive learning to explore diverse physical and spatial features. Dynamic features include mobility, which differentiates between moving and stationary states, movement speed to capture variations in velocity, and movement direction to account for distinct directional shifts. Postural features focus on posture, distinguishing between standing and sitting, and orientation, which captures differences in the relative facing direction of the sound source. Finally, positional features emphasize varying distances to explore positional relationships and different angles to capture changes in the viewing perspective. Together, these dimensions comprehensively enhance the robustness of our contrastive learning framework.

#### C.5 Spatial Audio Encoder and Decoder

The Spatial Audio Encoder and Decoder are designed based on the Variational Autoencoder (VAE) model [34]. During pre-training, we optimize the encoder and decoder using the L2 reconstruction loss. To further enhance reconstruction quality, we integrate a GAN discriminator inspired by the architecture of ML-GAN [7]. Specifically, we employ the LSGAN-style adversarial loss [49], L𝑎𝑑𝑣, which minimizes the distributional divergence between the predicted mel-spectrograms and the ground truth mel-spectrograms. Before encoding, we extract the mel-spectrogram using librosa 2. After generating the mel-spectrogram from the decoder’s output, HiFi-GAN [35] is used to convert it back into an audio waveform. To improve the quality of speech generation, we add the reconstructed F0 from our Immersive Drama Transformer during inference, applying the neural source filter (NSF) strategy for enhanced quality.

#### C.6 Mamba-Transformer Block

The integration of Transformer and Mamba layers provides a flexible framework to balance the often competing objectives of low memory consumption, high computational throughput, and output quality [17]. As sequence lengths increase, attention operations progressively dominate computational costs. In contrast, Mamba layers are inherently more compute-efficient, and increasing their proportion within the model improves throughput, particularly for longer sequences. After experimentation, we determine that an optimal balance is achieved with an attention-to-Mamba ratio of 1 : 𝐾, where 𝐾 is set to 3.

To enhance training stability and prevent numerical instability caused by uncontrolled growth in absolute values, we adopt RMSNorm [69]. For encoding scene 𝑠, we encode a video frame or textual scene descriptions into scene embedding 𝑧𝑠. Subsequently, the global embedding 𝑧𝑔 is computed by averaging the prompt audio embedding 𝑧𝑎 and the scene embedding 𝑧𝑠 along the temporal dimension, followed by the addition of the time step embedding 𝑧𝑡. This global embedding is processed through a global adaptor, which modulates the latent representation via adaptive layer normalization (AdaLN) [53] to ensure style consistency. The scale and

2https://github.com/librosa/librosa

Algorithm 1 Pseudo-Code of Drama-MOE Routing Strategy Input: Input hidden representation ℎ, content embedding 𝑧𝑐, prompt audio embedding 𝑧𝑎, pose embedding 𝑧𝑝, time step 𝑡 Output: Output with enhanced quality and control 𝑜final

- 1: Initialize Gumbel-Softmax temperature 𝜏, sample Gumbel noise 𝜁
- 2: for each time step 𝑡 do
- 3: Prosidic MOE:
- 4: Use Cross-Attention modeling prosody for alignment between 𝑧𝑎 and 𝑧𝑝:
- 5: 𝑧𝑝𝑟𝑜 ← CrossAttention(𝑧𝑝(𝑄),𝑧𝑎(𝐾),𝑧𝑎(𝑉))
- 6: Use Gumbel-Softmax for each token in the time channel to select an expert by 𝑧𝑝𝑟𝑜:
- 7: 𝑔prosodic(ℎ) ← GumbelSoftmax(𝑧𝑝𝑟𝑜 ·𝑊prosodic + 𝜁)/𝜏
- 8: Compute prosodic MOE output:
- 9: 𝑜prosodic ← 𝑖 𝑔prosodic,𝑖 · Expert𝑖,prosodic(ℎ)
- 10: Spatial MOE:
- 11: Use Gumbel-Softmax for each token in the time channel to select an expert by 𝑧𝑝:
- 12: 𝑔spatial(𝑜prosodic) ← GumbelSoftmax(𝑧𝑝 ·𝑊spatial + 𝜁)/𝜏
- 13: Compute spatial MOE output:
- 14: 𝑜spatial ← 𝑖 𝑔spatial,𝑖 · Expert𝑖,spatial(𝑜prosodic)
- 15: end for
- 16: Return 𝑜final ← 𝑜spatial as the final routed output

shift parameters are calculated using linear regression: 𝐴𝑑𝑎𝐿𝑁 (ℎ,𝑐) = 𝛾𝑐 × 𝐿𝑎𝑦𝑒𝑟𝑁𝑜𝑟𝑚(ℎ) + 𝛽𝑐, (17)

where ℎ represents the hidden representation. The batch normalization scale factor𝛾 is zero-initialized in each block [53]. Additionally, we utilize rotary positional embeddings (RoPE) [62] as a form of relative positional encoding, injecting temporal positional information into the model. This enhances the model’s ability to capture temporal relationships between sequential frames, leading to notable performance gains in the transformer.

Furthermore, a zero-initialized attention mechanism [2] is employed. Given the queries 𝑄ℎ, keys 𝐾ℎ, and values 𝑉ℎ derived from the hidden states, as well as the scene keys 𝐾𝑠 and values 𝑉𝑠, the final attention output is computed as:

𝑄 ˜ℎ𝐾˜ℎ⊤

𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 = softmax

𝑉ℎ+

√

𝑑

(18)

𝑄 ˜ℎ𝐾𝑠⊤

tanh(𝛼)softmax

√

𝑉𝑠,

𝑑

where 𝑄˜ℎ and 𝐾˜ℎ incorporate RoPE for queries and keys, 𝑑 represents the dimensionality of these vectors, and 𝛼 is a zero-initialized learnable parameter that modulates the cross-attention with the scene embedding.

#### C.7 Drama-MOE

Following previous works [71], our routing mechanism employs the dense-to-sparse Gumbel-Softmax technique [51] to achieve adaptive and efficient expert selection. This method utilizes the Gumbel-Softmax trick, which reparameterizes categorical variables to make sampling differentiable, enabling dynamic routing. For hidden state ℎ, the routing score assigned to expert 𝑖, denoted as 𝑔(ℎ)𝑖, is calculated as:

exp((ℎ ·𝑊𝑔 + 𝜁𝑖)/𝜏)

, (19)

𝑔(ℎ)𝑖 =

𝑁 𝑗=1 exp((ℎ ·𝑊𝑔 + 𝜁𝑗)/𝜏)

where𝑊𝑔 is the trainable gating weight, 𝜁 represents noise sampled from a Gumbel(0, 1) distribution [30], and 𝜏 denotes the softmax temperature. At the beginning of training, 𝜏 is set to a high value, promoting denser routing where multiple experts may contribute to processing the same input. As training advances, 𝜏 is gradually lowered, resulting in more selective routing with fewer experts involved. When 𝜏 approaches zero, the output distribution becomes nearly one-hot, effectively assigning each token to the most relevant expert. Following the approach of [51], we gradually decrease 𝜏 from 2.0 to 0.3 during training to transition from dense to sparse routing. During inference, a deterministic routing mode is applied, ensuring that only one expert is chosen for each token. The complete Drama-MOE algorithm is outlined in Algorithm 1.

To prevent overloading any single expert and to ensure balanced utilization, we integrate a load-balancing loss for each expert group, as described in Section 4.3, following the approach in [16]. For the regularization strength hyperparameter, we set it to 0.1 in our implementation. The load-balancing mechanism promotes a more uniform allocation of tokens across experts, thereby improving training efficiency by mitigating issues such as expert underutilization or excessive workload. Consequently, our routing strategy not only facilitates dynamic and adaptive expert selection but also ensures an even distribution of computational resources. This leads to reduced training time and enhanced performance for the DramaMOE.

#### D EVALUATION METRICS D.1 Objective Evaluation

We evaluate speech intelligibility using the Character Error Rate (CER). The CER is calculated by comparing the transcribed text from speech to the original target text by Paraformer-zh [20].

To objectively evaluate speaker similarity, we employ Cosine Similarity (SIM). SIM measures the resemblance in speaker identity between the synthesized speech and the GT speech by computing the average cosine similarity between the embeddings extracted

[Figure 52]

[Figure 53]

##### Figure 7: Screenshot of monaural speech evaluation.

from the synthesized and GT speech, thus serving as an objective measure of speaker similarity. We use WavLM [8] model fine-tuned for speaker verification 3 to extract speaker embeddings.

where melW is an 𝑀-bin mel filter bank. The IPD is derived from the phase spectrograms of the left and right channels:

We employF0Frame Error (FFE)toevaluate the synthesis prosody

𝑋2(𝑡, 𝑓 ) 𝑋1(𝑡, 𝑓 )

. (22)

𝐼𝑃𝐷(𝑡, 𝑓 ) = ∠

of the test set objectively. FFE combines metrics for voicing decision error and F0 error, capturing essential synthesis quality information.

ILD is extracted from the loudness spectrum of the left and right channels:

For the objective evaluation of IPD and ILD, we first convert the time-domain signal 𝑥(𝑛) into the frequency-domain signal 𝑋 (𝑡, 𝑓 ) using the short-time Fourier transform (STFT):

𝐼𝐿𝐷(𝑡, 𝑓 ) = 20log10 |𝑋2(𝑡, 𝑓 )| + 𝜀 |𝑋1(𝑡, 𝑓 )| + 𝜀

,𝜀 = 1𝑒−10. (23)

𝑁∑︁−1

𝑥𝑖(𝑛) · 𝑤(𝑡 − 𝑛) · 𝑒−𝑗2𝜋𝑓 𝑛,𝑖 ∈ {1, 2}, (20)

We calculate Mean Absolute Error (MAE) metrics based on the IPD and ILD extracted from the ground truth (GT) and the predicted speech. Since the IPD here is in radians and the ILD uses log10, the resulting values are quite small, especially after averaging the MAE over the time dimension. So, we multiply by 100 to make the results more intuitive.

𝑋𝑖(𝑡, 𝑓 ) =

𝑛=0

where 𝑤(𝑡 − 𝑛) is a window function, 𝑁 is the window length, and 𝑖 indicates the channel of the binaural audio. Next, we calculate the mel-spectrogram, IPD, and ILD based on the frequency-domain signals𝑋𝑖(𝑡, 𝑓 ). The mel-spectrogram for each channel is calculated as:

Additionally, we analyze angular and distance metrics using SPATIAL-AST [76]. SPATIAL-AST encodes an angle and a distance embedding for binaural audio. We compute and average the cosine similarity for each 1-second segment based on the GT and predicted audio.

𝑆𝑖(𝑡,𝑚) = log |𝑋𝑖(𝑡, 𝑓 )|2 × melW , (21)

3https://huggingface.co/pyannote/speaker-diarization

[Figure 54]

[Figure 55]

Figure 8: Screenshot of binaural speech evaluation.

#### D.2 Subjective Evaluation

research. The instructions for testers in monaural and binaural evaluation are shown in Figure 7 and Figure 8.

We conduct Mean Opinion Score (MOS) as a subjective evaluation metric. For each task, we randomly select 40 pairs of sentences from our test set for subjective evaluation. Each pair consists of an audio prompt that provides timbre, and a synthesized speech sample, both of which are evaluated by at least 15 professional listeners.

For CMOS, listeners are asked to compare pairs of audio generated by systems A and B and indicate their preference between the two. They are then asked to choose one of the following scores: 0 indicating no difference, 1 indicating a slight difference, 2 indicating a significant difference, and 3 indicating a very large difference.

In the context of MOS-Q evaluations, these listeners are instructed to concentrate on synthesis quality (including clarity, naturalness, rich stylistic details, coherence, and spatial perception for binaural audio). Conversely, during MOS-S evaluations, the listeners are directed to assess speaker similarity (in terms of timbre, accent, and articulation) to the audio prompt. For MOS-E, the listeners are informed to evaluate prosodic expressiveness. For MOS-P, the listeners are instructed to evaluate the consistency between the pose information provided by the multimodal prompt and the generated binaural speech (focusing on angle, distance in 3D space, and changes caused by movement).

#### E EXTENDED EXPERIMENTS E.1 Baseline Models

UniAudio [66] uses LLMs to generate multiple audio types by tokenizing target audio and conditions, concatenating them into a single sequence, and performing next-token prediction. A multiscale Transformer is introduced to handle long sequences caused by neural codec-based VQ. We employ their official code 4.

StyleTTS 2 [44] combines style diffusion and adversarial training with large speech language models (SLMs) for high-quality text-to-speech synthesis. It models style as a latent random variable using diffusion models. We employ their official code 5.

In MOS evaluations, listeners are requested to grade various speech samples on a Likert scale ranging from 1 to 5. Notably, all participants are fairly compensated for their time and effort. We compensated participants at a rate of $12 per hour, with a total expenditure of approximately $300 for participant compensation. Participants are informed that the results will be used for scientific

CosyVoice [15] represents speech with supervised semantic tokens derived from a multilingual speech recognition model, using

- 4https://github.com/yangdongchao/UniAudio
- 5https://github.com/yl4579/StyleTTS2

[Figure 56]

[Figure 57]

[Figure 58]

(a) Left mel-spectrogram (b) Right mel-spectrogram (c) Interaural level differences

##### Figure 9: Visualization results.

Table 5: Ablation study for context-consistent CFG.

vector quantization in the encoder. It employs an LLM for textto-token generation and a conditional flow matching model for token-to-speech synthesis. We employ their official code 6

𝛼 𝛾 CMOS-Q CMOS-S CMOS-E

FireRedTTS [25] is a language-model-based TTS system that encodes speech into discrete semantic tokens via a semantic-aware speech tokenizer. A language model then generates these tokens from text and audio prompts, followed by a two-stage waveform generator for high-fidelity synthesis. We employ their official code 7.

1.00 1.00 -0.62 -0.09 -0.78 0.40 1.00 -0.31 -0.16 -0.43

- 0.40 1.50 -0.36 -0.22 -0.51
- 1.00 3.00 -0.28 -0.02 -0.55 0.40 3.00 0.00 0.00 0.00 0.00 3.00 -0.30 -0.24 -0.22 0.40 5.00 -0.49 0.04 -0.60

F5-TTS [10], a non-autoregressive text-to-speech system based on flow matching with Diffusion Transformer (DiT). Text input is padded with filler tokens to match the length of input speech, followed by denoising for speech generation. We employ their official code 8.

𝛾 > 5, the prosody adheres to the style of the prompt audio but reduces the ability to adapt to semantic learning. On the other hand, when 𝛼 approaches 0, the CMOS-S decreases as the model overrelies on contextual cues, neglecting the fine-grained details of the accent and pronunciation in the prompt audio. Conversely, when 𝛼 approaches 1, the CMOS-E decreases because the model becomes overly dependent on the prompt audio, making it challenging to model semantically aligned dramatic prosody.

#### E.2 Visualization Results

Figure 9 presents a visualization of our binaural speech synthesis results. In Figures (a) and (b), we illustrate the mel spectrogram and F0 contour of a single speech sample for the left and right audio channels. ISDrama effectively captures fine-grained mel details and generates expressive F0 variations, demonstrating high-quality synthesis and rich prosody. Figure (c) visualizes the Interaural Level Difference (ILD). The darker coloration on the left side indicates that the sound source initially originates from the back-left position, while the lighter coloration on the right side corresponds to the source moving toward the back-right. This clearly shows that ISDrama successfully models positional information, accurately reflecting spatial dynamics.

By setting 𝛾 = 3 and 𝛼 = 0.4, we achieve improved generation quality and incorporate previously generated audio to enhance the prosody consistency of the same speaker within a single dramatic act.

#### E.3 Context-Consistent CFG

We conduct experiments to verify the parameter settings of Equation 5, as shown in Table 5. For evaluation, we implement CMOS assessments. When 𝛼 = 1, 𝑣𝑐𝑓 𝑔 relies solely on the input prompt audio. Moreover, if 𝛼 = 1 and𝛾 = 1, 𝑣𝑐𝑓 𝑔 becomes equivalent to the original formulation 𝑣𝑡 (𝑥,𝑡|𝑎,𝐶;𝜃).

When𝛾 increases from 1 to 1.5, the generated speech exhibits inconsistencies between the pronunciation and prosody of the prompt audio, resulting in lower CMOS-S and suboptimal CMOS-E. When 𝛾 ranges from 1.5 to 3, the generated speech aligns with the pronunciation and accent of the prompt audio. Within this range, if 𝛼 is appropriately set, the model effectively utilizes the previous prompt audio to synthesize semantically aligned dramatic prosody. When

- 6https://github.com/FunAudioLLM/CosyVoice
- 7https://github.com/FireRedTeam/FireRedTTS
- 8https://github.com/SWivid/F5-TTS

