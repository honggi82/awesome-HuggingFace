# arXiv:2507.11949v2[cs.GR]2Nov2025

## MOSPA: Human Motion Generation Driven by Spatial Audio

Shuyang Xu∗,1, Zhiyang Dou∗,†,1, Mingyi Shi1, Liang Pan1, Leo Ho1, Jingbo Wang2, Yuan Liu3, Cheng Lin4, Yuexin Ma5, Wenping Wang†,6, Taku Komura†,1

1The University of Hong Kong 2Shanghai AI Lab 3The Hong Kong University of Science and Technology 4Macau University of Science and Technology 5ShanghaiTech University 6Texas A&M University

[Figure 1]

Figure 1: We introduce a novel human motion generation task centered on spatial audio-driven human motion synthesis. Top row: We curate a novel Spatial Audio-Driven Human Motion (SAM) dataset, including diverse spatial audio signals and high-quality 3D human motion pairs. Bottom row: We develop a generative framework for human MOtion generation driven by SPatial Audio (MOSPA) to produce high-quality, responsive human motion driven by spatial audio. We note that the motion generation results are both realistic and responsive, effectively capturing both the spatial and semantic features of spatial audio inputs.

#### Abstract

Enabling virtual humans to dynamically and realistically respond to diverse auditory stimuli remains a key challenge in character animation, demanding the integration of perceptual modeling and motion synthesis. Despite its significance, this task remains largely unexplored. Most previous works have primarily focused on mapping modalities like speech, audio, and music to generate human motion. As of yet, these models typically overlook the impact of spatial features encoded in spatial audio signals on human motion. To bridge this gap and enable high-quality modeling of human movements in response to spatial audio, we introduce the first comprehensive Spatial Audio-Driven Human Motion (SAM) dataset, which contains diverse and high-quality spatial audio and motion data. For benchmarking, we develop a simple yet effective diffusion-based generative framework for human MOtion generation driven by SPatial Audio, termed MOSPA, which faithfully captures the relationship between body motion and spatial audio through an effective fusion mechanism. Once trained, MOSPA can generate diverse realistic human

∗, † denote equal contributions and corresponding authors.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

motions conditioned on varying spatial audio inputs. We perform a thorough investigation of the proposed dataset and conduct extensive experiments for benchmarking, where our method achieves state-of-the-art performance on this task. Our code and model are publicly available at our website.

#### 1 Introduction

Humans exhibit varying responses to different auditory inputs within a given space. For instance, when exposed to sharp, piercing sounds, individuals are likely to cover their ears and move away in the direction opposite to the sound source. Conversely, when the sound is soft and soothing, they may approach it out of curiosity or to investigate further. Therefore, generating realistic human motion for virtual characters to respond realistically to a variety of sounds in their environment is both a highly sought-after feature and is crucial for applications such as virtual reality, human-computer interaction, robotics, etc.

Unfortunately, while previous studies have extensively explored motion generation from action label [85, 31], text [93, 100, 77], music [78, 91, 46, 1], and speech [2, 97, 3, 102], human motion generation driven by spatial audio remains unexplored to the best of our knowledge. Unlike pure audio signals, e.g., music [71, 48, 78], speech [3, 97], the spatial audio signals not only does it encode semantics, but it also captures spatial characteristics that significantly influence body movements, requiring a specialized framework to accurately model motion responses to spatial audio stimuli.

To address this overlooked aspect, we propose to model the complex interactions between spatial audio inputs and human motion using a generative model. Since there is no such dataset tailored for this task, we first introduce the SAM dataset (Spatial Audio Motion dataset), which captures diverse human responses to various spatial audio conditions. This dataset is meticulously curated to include a wide range of spatial audio scenarios, enabling the study of motion conditioned on sound field variations. The SAM dataset has a total of more than 9 hours of motion, covering 27 common spatial audio scenarios and more than 70 audio clips. To ensure the diversity of the spatial audio, around 480 seconds of motion were captured for each audio clip at different positions in the character space. To ensure diverse motion responses to spatial audio, we introduce 20 distinct motion types (excluding motion genres) and 49 in total when including motion genres. We visualize samples from SAM in the top row of Fig. 1. See Appendix A for detailed statistics.

We further conduct benchmarking experiments on the proposed dataset, revealing the limitations of existing methods in this setting. To enable spatial audio-driven human motion generation, we introduce MOSPA, a simple yet effective framework tailored for this task. In real-world scenarios, human responses to sound are inherently influenced by spatial perception, intensity variations, directional cues, temporal dynamics, etc. Motivated by this, we generate motion by incorporating features extracted from the input spatial audio signals using [61]. Specifically, to capture intrinsic features across both temporal and spatial dimensions, we mainly utilize Mel-Frequency Cepstral Coefficients (MFCCs)[17] and Tempograms[28] to model the temporal characteristics of the audio. Additionally, we characterize the spatial audio by analyzing the root mean square (RMS) [61] energy, which quantifies signal intensity in audio processing.

These features enhance the effective modeling of the spatial and intensity variations of the spatial audio. To capture the distribution of spatial audio features and human motion dynamics effectively, we employ a diffusion-based generative model that ensures strong alignment between the two modalities—human motion and spatial audio signals. Leveraging diffusion models, MOSPA excels at modeling the complex interplay between spatial audio features and human motion. Besides, a residual feature fusion mechanism is employed to model the subtle influences of spatial audio on human movement.

Extensive evaluations on the SAM demonstrate that MOSPA achieves state-of-the-art performance on this task, outperforming existing baselines in generating realistic and diverse motion responses to spatial audio. Our contributions are summarized as follows:

- • We introduce a novel task of spatial audio-conditioned motion generation and present the first comprehensive dataset SAM with over 9 hours of motion across diverse scenarios.
- • We conduct extensive benchmarking and propose MOSPA, a diffusion-based generative framework tailored for modeling and generating diverse human motions from spatial audio.

- • We achieve the SOTA performance on motion generation conditioned on spatial audio. Our dataset, code, and models will be publicly released for further research.

#### 2 Related Work

Spatial Audio. Many studies have explored spatial audio modeling [98, 24, 99, 74, 37, 87, 45, 44]. For instance, [98] utilizes the natural synchronization between visual and audio modalities to learn models that jointly parse sounds and images without manual annotations. [24] leverages unlabeled audiovisual data to localize objects, such as moving vehicles, using only stereo sound at inference time. [99] reason about spatial sounds with large language models. Recently, spatial audio generation has been explored from text [74] and video [42]. [87] propose a method to model 3D spatial audio from body motion and speech. [37] presents a framework for spatial audio generation, capable of rendering 3D soundfields generated by human actions, including speech, footsteps, and hand-body interactions. Despite progress on spatial-audio tasks, generating human motion from spatial audio remains largely underexplored.

Conditional Motion Generation. Extensive efforts have been made into motion synthesis conditioning on user control signals [36, 72, 11, 79, 92], text [93, 100, 77, 30, 51, 53, 14], action [85, 31, 9, 39], music [78, 91, 46, 1], speech [2, 68, 40, 97, 3, 102], past trajectories [23, 10, 4, 90, 73], etc. We refer readers to [103] for a detailed survey of motion generation.

Text-to-Motion. Text-to-motion generation has recently gained popularity as an intuitive and userfriendly approach to synthesizing diverse human body motions. Generative pre-trained transformer frameworks have been utilized for text-to-motion generation [93, 57]. Subsequently, various generation techniques have been explored, including diffusion models [77], latent diffusion models [12], autoregressive diffusion model [69, 11], denoising diffusion GANs [100], consistency models [16], and generative masked modeling [29]. Recent advancements include the integration of motion generation with large language models [41, 94] and investigations into the scaling laws for motion generation [58, 22]. Recently, controllable text-to-motion generation has gained attention, enabling motion synthesis conditioned on both text prompts and control signals, e.g., target control points [84, 79].

Music-to-Motion. Recent advancements in Music-to-Motion generation have been made [21, 71, 50, 78, 27, 88, 70]. DanceFormer [47] adopts a two-stage approach, generating key poses for beat synchronization followed by parametric motion curves for smooth, rhythm-aligned movements. Bailando [71] utilizes a VQ-VAE to encode motion features via a choreographic memory module. [1] introduces a diffusion-based probabilistic model for motion generation, using a Conformer-based architecture. EDGE [78] also applies a diffusion model for dance generation and editing. Furthermore, multimodal approaches incorporating language and music enhance generation quality [27, 88, 15].

Speech-to-Motion. We mainly review studies on audio-driven motion (gesture) generation [97, 13,

###### 3, 2, 102]. Early works are mostly based on GAN models [26, 54, 65, 89], while the recent attempts are mainly based on the generative diffusion model [102]. For instance, [97] proposes a generative retrieval framework leveraging a large language model to efficiently retrieve semantically appropriate gesture candidates from a motion library in response to input speech. [2] introduces a co-speech gesture synthesis method by employing a segmentation pipeline for temporal alignment and disentangling speech-motion embeddings to capture both semantics and subtle variations.

While audio cues power many music- and speech-to-motion methods, the use of spatial audio for human motion synthesis is still scarcely studied. As a result, data-driven methods are highly constrained by limited paired data. The goal of this paper is to develop a comprehensive dataset and a novel approach for high-quality spatial audio-driven motion synthesis.

#### 3 SAM Dataset

We first introduce the Spatial Audio-driven Motion (SAM) dataset designed for human motion synthesis conditioned on spatial audio. We focus on binaural audio, a common form of spatial audio that aligns with human and (most) animal perception and can be readily applied to robotic platforms. SAM consists of more than 9 hours of human motions with corresponding binaural audio, and more than 4M frames, covering 27 common spatial audio scenarios and 20 common reaction types in daily life without counting the motion genres. The majority of the audio clips are sourced from the AudioSet [25], while only a small portion is extracted from publicly available YouTube videos

- Table 1: Statistics of the SAM dataset. The SAM dataset encompasses 27 common daily spatial audio scenarios, over 20 reaction types excluding the motion genres, and 49 reaction types (see details in Appendix A). The number of subjects covered in SAM is 12, where 5 of them are female and the remaining 7 are male. It is also the first dataset to incorporate spatial audio information, annotated with Sound Source Location (SSL). The total duration of the dataset exceeds 34K seconds.

Dataset SSL 3DJointpos/rot Model Joints Subjects Seconds

Dance with Melody [75] × ✓/× - 21 - 5640 DanceNet [104] × ✓/× - 55 2 3472 AIST++ [50] × ✓/✓ COCO/SMPL 17/24 30 18694 PopDanceSet [60] × ✓/✓ COCO/SMPL 17/24 132 12819 FineDance [49] × ✓/✓ SMPL & hand joints 52 27 52560

SAM (Ours) ✓ ✓/✓ SMPL-X 55 12 34356

[Figure 2]

(a) Look for sound source and approach upon hearing miaow at the left-hand side.

[Figure 3]

(b) Step away with ears covered upon hearing crowd yelling at the back.

[Figure 4]

- (c) Run away from the sound source upon hearing gunshot at the right hand side.

[Figure 5]

- (d) Walk towards the sound source upon hearing insect noise at the left-hand side.

[Figure 6]

(e) Start dancing upon hearing music in front of the character.

[Figure 7]

(f) Look for sound source upon hearing the phone ring at the left-hand side.

- Figure 2: Visualization of samples from SAM with expected motions annotated. Red dots indicate the actor’s trajectory, while the blue sphere represents the sound source. The SAM dataset ensures high diversity by encompassing a broad spectrum of audio types and varying sound source locations.

or through manual recording. More detailed information can be found in Tab. 1 and Appendix A. Visualization results are in Fig. 2.

[Figure 8]

[Figure 9]

(i) Mocap environment. (ii) The speakers.

- Figure 3: Spatial audio-driven human motion data collection setup. Figure 4: Statistics of action duration in the dataset.

[Figure 10]

Data Capture Settings. We utilize a Vicon motion capture system [56] to collect motion data and spatial audio signals. The motion capture is performed in a semi-open cage having a space of approximately 5m × 10m × 3m, a structure covered by rope nets, with 28 mocap cameras mounted on the ropes and vertical supports recording at a frame rate of 120 Hz; See Fig. 3. The surrounding walls are standard painted concrete, resulting in a setting that resembles a typical indoor environment. In SAM, each audio clip is associated with 16 randomly sampled relative sound source locations, defined by combinations of different speakers and spatial positions relative to the subject. For each location, we capture three motion sequences corresponding to different reaction intensities: dull, neutral, and sensitive, resulting in a total of 48 motion sequences, each lasting 10 seconds. Fig. 4 shows the statistics of the approximate duration of the actions. The three motion genres define the varying degrees of responsiveness, decreasing from sensitive to dull. For instance, upon hearing an explosion, a dull individual might remain largely unreactive, whereas a sensitive one may immediately flee from the sound source. The total number of action types is 49. The percentage of action types covered within the dull, neutral, and sensitive motion genres are 28.57%, 34.69%, and 36.73%, respectively. To capture the binaural sound heard at the position of the actor, we employ two microphones to record the audio at the ear positions of the actors separately; See the inset. The two microphones are connected to a Deity PR-2 recorder [18] that has been synchronized with the Vicon mocap system in advance using a timecode with a frame rate of 30 FPS. With this setting, the stereo sound at the position of the actor can be recorded and has an accurate alignment with the corresponding motion.

[Figure 11]

Data Processing. All motion and audio clips are precisely aligned. The motions are re-targeted and converted from the original BVH format in Vicon to the SMPL-X [63] format. SMPL-X is a parametric 3D human body model that encompasses the body, hands, and face, comprising N = 10,475 vertices and K = 55 joints. Given shape parameters β and pose parameters θ, the SMPL-X model generates the corresponding body shape and pose through forward dynamics. We extract the locations of the sound sources in each motion clip. The sound source locations are then transformed into the local space of the character aligned with the SMPL-X local coordinate system (a.k.a the local frame).

- 4 Method

We introduce MOSPA, a diffusion-based probabilistic model that serves as a baseline for this novel task of spatial audio-driven human motion generation. First, we extract spatial audio features a using a feature extractor [61]. During motion generation, the extracted spatial audio feature a is combined with the sound source location s and the motion genre g as conditioning inputs. These inputs are passed to a denoiser G, which is trained to reconstruct the original clean motion vector xˆ0 by denoising the given noisy motion vector xt at time step t. Mathematically, we have xˆ0 = G(xt,t;a,s,g).

##### 4.1 Feature Representation

The two key vectors are the audio feature vector a and the motion vector p. We carefully designed the structure of the two vectors in MOSPA.

Spatial Audio Feature Extraction. We first extract a range of audio features that capture intensity, temporal dynamics, and spatial characteristics. Inspired by [71], our feature set primarily includes Mel-

frequency cepstral coefficients (MFCCs), MFCC delta, constant-Q chromagram, short-time Fourier transform (STFT) of the chromagram, onset strength, tempogram, and beats [17, 67, 28, 61, 7, 20]. On top of these audio features, we additionally add the root mean square (RMS) energy Erms of the audio [61], and the active frames Factive defined as Factive = Erms > 0.01 to capture the distance information of the audio. The dimension of the audio feature vector for each ear is 1136. By concatenating the features from both ears, we obtain a combined feature vector a of dimension 2272. The detailed construction of the audio vector can be viewed in Appendix B.1.

Motion Representation. In this paper, we focus on body motion and leave the modeling of detailed finger movements to future work. Therefore, we exclude all the finger joints and retain only the first J = 25 body joints of the SMPL-X model [63]. In addition to the essential translation and joint rotations required for human pose representation, we introduce the residual feature fusion mechanism [30] to incorporate the global joint positions and the velocity of the joints to capture the nuanced difference in audio and further improve the accuracy of the generated samples. Each motion vector x is thus composed of the global positions p ∈ RT×(J×3), the local rotations r ∈ RT×(J×6) and the velocities v ∈ RT×(J×3) of the joints (including the root), where T = 240 represents the number of frames in each motion sequence. The joint rotations are represented in the 6d format [101] to guarantee the continuity of the change (x0 = (p0,r0,v0),x ∈ RT×(J×12)). The dimension of each motion vector is therefore 300.

##### 4.2 Framework

[Figure 12]

Following [77, 11], the diffusion is modeled as a Markov chain process which progressively adds noise to clean motion vectors x0 in t time steps, i.e.

q(xt|xt−1) = N(√αtxt−1,(1−αt)I) (1)

where αt ∈ (0,1). The model then learns to gradually denoise a noisy

motion vector xt in t time steps, i.e. p(xt−1|xt). We directly predict the clean sample xˆ0 in each diffusion step xˆ0 = G(xt,t;a,s,g), where a is the audio features, s is the sound source location and g is the motion genre. This strategy, employed by [77, 11, 66, 84], has been proved to be more efficient and accurate than predicting the noise ϵt, suggested by [35]. We employ an encoder-only Transformer to reverse the diffusion process and predict the clean samples. The timestep, motion, and conditioning signals are each projected into the same latent dimension using separate feed-forward networks. Random masks are applied to the audio features a and the sound source location (SSL) s, after which all components are concatenated to form the complete token sequence z. The tokens are positionally embedded afterward and input into a transformer to get the output ˆz. The predicted clean sample is thus extracted from the last T tokens of ˆz by inputting it to another feed-forward network, where T = 240 is the length of the motion; see Fig. 5.

Figure 5: The framework of MOSPA. We perform diffusion-based motion generation given spatial audio inputs. Specifically, Gaussian noise is added to the clean motion sample x0, generating a noisy motion vector xt, modeled as q(xt|xt−1). An encoder transformer then predicts the clean motion from the noisy motion xt, guided by extracted audio features a, sound source location (SSL) s, motion genre g, and timestep t.

##### 4.3 Loss Functions

We train MOSPA using the following loss functions. A simple mean squared error (MSE) loss is applied to the original clean sample and the predicted clean sample as the main objective: E∥xˆ0 − x0∥22. To guarantee the smooth variation on the predicted clean sample across frames, we also apply MSE loss to the rate of change of the vectors across frames: E∥δxˆ0 − δx0∥22. Combining the two simple losses we have Ldata = E∥xˆ0 − x0∥22 + E∥δxˆ0 − δx0∥22. Geometric losses, encompassing position loss and velocity loss, are also incorporated, as we rely solely on joint

[Figure 13]

Figure 6: Qualitative comparison of state-of-the-art methods for the spatial audio-to-motion task. We visualize motion results from five cases. MOSPA produces high-quality movements that closely correspond to the input spatial audio. We provide Expected Motion as a description for reference.

rotations and translations in motion vectors to represent poses: Lgeo = E∥FK(xˆ0) − FK(x0)∥22 + E∥δFK(xˆ0) − δFK(x0)∥22. Furthermore, foot sliding is prevented by introducing the foot contact loss Lfoot that measures the inconsistency in the velocities of the foot joints between the ground truth and the predicted motions.

We also incorporate trajectory loss and joint rotation loss to underscore their importance in achieving the training objectives and accelerate the convergence of the model, defined as Ltraj = E∥trajˆ 0 − traj0∥22 + E∥δtrajˆ 0 − δtraj0∥22 and Lrot = E∥ˆr0 − r0∥22 + E∥δˆr0 − δr0∥22 respectively, where traj is the trajectory vector of the motion sequence and r is the joint rotations represented in the 6d format [101]. Given that trajectory and joint rotations are inherently encoded within the motion vectors, these supplementary losses represent an overlap with the existing loss terms, effectively amplifying the emphasis on trajectory and joint rotation accuracy through increased weighting. Empirically, we observe that this implementation accelerates model convergence and facilitates correct displacement direction generation in motion sequences. In sum, the total loss is given by:

L = λdataLdata + λgeoLgeo + λfootLfoot + λtrajLtraj + λrotLrot (2)

All loss weights (λ) are initialized set to 1. At epoch 5,000 of the total 6,000 training epochs, λtraj and λrot are increased to 3, thereby intensifying the emphasis on trajectory and rotation accuracy.

##### 4.4 Implementation Details

In MOSPA, the diffusion model is a transformer-based diffusion network [11, 77, 100]. The encoder transformer is configured with a latent dimension of 512, 8 heads, and 4 layers. We employ AdamW [55] as the optimizer with an initial value of 1 × 10−4. The number of denoising steps used is 1000, and the noise schedule is cosine. The training phase concludes after 6,000 epochs. Exceeding these recommended epoch counts may degrade model quality due to overfitting. The

- Table 2: Quantitative evaluation on the SAM, where MOSPA achieves higher alignment with the GT motion while maintaining high diversity, as reflected by the metrics. The error bar is the 95% confidence interval assuming normal distribution, and → means the closer to Real Motion the better.

Method

R-precision ↑ FID ↓ Diversity → APD → Top1 Top2 Top3

Real Motion 1.000±0.000 1.000±0.000 1.000±0.000 0.001 23.616±0.188 59.435

EDGE [78] 0.886±0.005 0.960±0.003 0.977±0.002 13.993 23.099±0.196 43.882 POPDG [60] 0.762±0.006 0.886±0.005 0.934±0.003 20.967 22.536±0.170 34.996 LODGE [48] 0.444±0.006 0.594±0.005 0.679±0.004 102.289 21.101±0.141 11.801 Bailando [71] 0.077±0.003 0.134±0.003 0.182±0.004 168.396 17.347±0.247 23.121

MOSPA 0.937±0.005 0.984±0.002 0.996±0.001 7.981 23.575±0.188 53.915

- Table 3: Ablation study on MOSPA on the spatial audio-driven motion generation performance. The error bar is the 95% confidence interval assuming normal distribution, and → means the closer to real motions the better.

Latent Dim

Head Num

Diff Steps

R-precision ↑ FID ↓ Diversity → APD → Top1 Top2 Top3

Genre

Real Motion 1.000±0.000 1.000±0.000 1.000±0.000 0.001 23.616±0.188 59.435

512 8 1000 ✓ 0.937±0.005 0.984±0.002 0.996±0.001 7.981 23.575±0.188 53.915 256 8 1000 ✓ 0.891±0.005 0.952±0.002 0.971±0.001 9.226 23.007±0.198 55.175 512 4 1000 ✓ 0.923±0.004 0.972±0.002 0.986±0.001 9.282 23.232±0.170 56.572 512 8 100 ✓ 0.930±0.004 0.980±0.002 0.991±0.001 8.456 23.351±0.177 49.824 512 8 4 ✓ 0.934±0.004 0.989±0.002 0.998±0.001 8.387 23.474±0.192 49.507 512 8 1000 × 0.889±0.005 0.958±0.003 0.977±0.002 10.930 23.150±0.153 46.807

entire training process requires approximately 18 hours on a single RTX 4090 GPU with a batch size of 128.

#### 5 Experiments

Experiment Setup. We use our SAM dataset to evaluate the spatial audio-driven motion generation task. As detailed in Sec. 3, it contains 9 hours of human motion with paired binaural audio and corresponding sound source locations, covering 27 common spatial audio scenarios and 20 common reaction types. The dataset is split into training, validation, and test sub-datasets at a common ratio of 8:1:1. Consequently, the training sub-dataset comprises 2,400 motion sequences, while the validation and test sub-datasets each contain approximately 300 motion sequences. To keep fair setting [9], the motions and the audio clips are both downsampled to the frame rate of 30 FPS. The character is rotated to face the negative y-axis and initially translated to the origin in the world space in all motion sequences, and the sound source locations (SSL) are transformed to the local space of the character in every single frame.

Baselines and Metrics. Our system is the first work to receive spatial audio as input to generate human motion results. To our best knowledge, as there is no other system achieving this, we made adaptations on other audio2motion methods, such as EDGE [78], POPDG [60], LODGE [48] and Bailando [71] by replacing their original audio input with our spatial audio feature as input. We evaluated four metrics, focusing on motion quality and diversity: 1) R-precision, FID, Diversity These three metrics are calculated using the same setup proposed by [30]. Two bidirectional GRU are trained with a hidden size of 1024 for 1,500 epochs with a batch size of 64 to extract the audio features and the corresponding motion features, as suggested by [30]. Detailed implementation details of the feature extractor are provided in Appendix B.2. 2) APD [19, 33] is calculated by

- 1

- 2

APD(M) = N(N1−1) Ni=1 Nj=1 j̸=i

L t=1 ∥sit − sjt∥2

, where M = {xˆi} is the set of generated motion sequences, N is the number of motion sequences in the set M, L is the number of frames of each motion sequence, and sit ∈ xˆi is a state in the motion sequence xˆi.

##### 5.1 Comparisons

Qualitative Results. We demonstrate the qualitative comparison in Fig. 6. For the same input spatial audio, our methods show the superiority of producing high-quality and realistic response motion. Other methods often exhibit various limitations due to their unique model characteristics. EDGE [78] and POPDG [60] demonstrate relatively strong performance among the four baselines, sharing a diffusion-based foundation with MOSPA, despite differences in their encoding and decoding mechanisms. Their shortcomings in generated samples can primarily be attributed to model size and their strong focus on music-like audio. The bad performance of LODGE [48] is likely due to its specialization in long-term music-like audio, resulting in deficiencies when handling short-term audio information with abrupt feature changes. Similarly, Bailando [71] faces challenges in processing rapidly changing spatial audio. More critically, due to its separate training process for upper and lower body parts, Bailando occasionally produces distorted or disjointed motions when encountering sudden changes in spatial audio. Please watch our supplementary video for more results. Furthermore, we test MOSPA on out-of-distribution audio-source configurations. As shown in Fig. 8, it maintains motion quality and intent alignment, demonstrating robustness to unseen spatial setups.

Quantitative Results. The quantitative results are reported in Tab. 2. MOSPA achieves the best performance as shown by the lowest FID value and the highest R-precision values. Also, our generated motions exhibit the closest diversity and APD [19] values compared with the Real Motion, demonstrating the effectively balanced variation and precision. Bailando [71] has the worst performance among the four baselines in practice, as illustrated by the extremely high FID. The model possibly lacks the ability to perceive commonly heard sounds other than music and also the spatial information of the audio. Our method, overall speaking, still demonstrates competitive performance in spatial audio conditioned motion generation, which is proved by the low values in precision-related metrics and the high values in diversity-related metrics.

[Figure 14]

User Study. We conducted a user study with 25 participants to assess the perceptual quality of motion generation. Participants evaluated five models (MOSPA, EDGE, POPDG, LODGE, Bailando) alongside ground truth (GT), selecting the best motion for: 1) Human Intent Alignment: Does the motion align with real-world intent? 2) Motion Quality: Which has the highest movement quality? 3) GT Similarity:Which best matches the GT motion? We provided GT motion and a textual description for reference. As shown in Fig. 7, MOSPA outperforms all baselines across all criteria, while LODGE and Bailando received the fewest selections, indicating limitations in generating realistic, semantically meaningful motions. See more details in Appendix C.

Figure 7: User study results. MOSPA outperforms other methods in intent alignment, motion quality, and similarity to ground truth. The bar chart shows the vote distribution across methods.

##### 5.2 Ablation Study

We conducted ablation studies on the latent dimension, the number of attention heads, the diffusion step number, and the masking of motion genre, with results summarized in Tab. 3. All ablation experiments maintained consistent training epoch counts throughout.

Latent Dimension. The default latent dimension of MOSPA’s encoder transformer is 512. In our study, reducing it to 256 slightly increases the APD [19] value but also degrades the R-precision and the FID, leading to an overall decline in model performance, as seen in row 1 and row 2 in Tab. 3.

Number of Attention Heads. We reduced the number of attention heads in MOSPA ’s encoder transformer from 8 to 4, observing degradation in almost all of the metrics except a slight improvement in APD [19]. This reduction compromises overall model performance without yielding significant improvements in training efficiency, as seen in row 1 and row 3 in Tab. 3.

Number of Diffusion Steps. We evaluated MOSPA with varying diffusion step numbers, reducing it from 1000 to 100 and further to 4, as detailed in rows 1, 4, and 5 of Tab. 3. Fewer steps slightly degrade the performance as shown by the increase in FID and degradation in diversity, thereby lowering the upper limit of the power of the model.

[Figure 15]

- Figure 8: Test of MOSPA on out-of-distribution spatial audios. Descriptions of motions are provided for reference.

[Figure 16]

- Figure 9: Spatial audio-driven physically simulated humanoid robot control based on [34]. Descriptions of expected motion are provided for reference.

Genre Masking. Masking motion genres leads to a degradation in model performance across all metrics, as demonstrated in row 1 and 6 of Tab. 3. Motion genres are required to provide a guidance for the model on the intensity of the expected motions.

We evaluate the contribution of the extracted audio features by conducting an ablation study on the effectiveness of MFCC [17] and tempogram [28] features. As shown in Tab. 4, improvements in FID and Rprecision—two key metrics for assessing generative quality and correspondence—demonstrate their significance in model.

Table 4: Ablation study on the effect of MFCC [17] and tempogram [28] features.

R-precision ↑ FID ↓ Top-1 Top-2 Top-3

MFCC Tempogram

✓ ✓ 0.937±0.005 0.984±0.002 0.996±0.001 7.981 × ✓ 0.907±0.004 0.967±0.002 0.983±0.002 9.070 ✓ × 0.917±0.004 0.982±0.002 0.994±0.001 10.786

#### 6 Conclusion

This introduces a novel task for enabling virtual humans to respond realistically to spatial auditory stimuli. We present a comprehensive SAM dataset, capturing human movement in response to spatial audio, and propose MOSPA, a diffusion-based generative model with an attention-based fusion mechanism. Once trained, MOSPA synthesizes diverse, high-quality motions that adapt to varying spatial audio inputs with binaural recording. Extensive evaluations show MOSPA achieves state-of-the-art performance on this task. Limitations and Future Works. Physical Correctness: While MOSPA generates diverse and semantically plausible motions, it lacks physical constraints, which may lead to physically implausible artifacts. Integrating physics-based control methods [19, 59, 76, 38, 95, 96, 62, 82] could improve motion realism and embodiment fidelity (see Fig. 9 for spatial audio-driven humanoid robot control). Body Modeling: This work focuses on body motion and omits finer-grained components such as hand gestures and facial expressions supported by SMPL-X [63]. Extending the model to full-body motion generation [57, 52, 86, 64, 5]—including hand motions—remains an important direction for future research. Scene Awareness: The current framework does not incorporate awareness of surrounding environments or physical scene geometry, limiting its ability to produce scene-consistent or contact-aware motions. Future extensions could integrate scene representations or affordance prediction [14, 81, 83, 8, 80] with spatial audio signals to enhance human motion generation.

#### Acknowledgements

This work is partly supported by the Innovation and Technology Commission of the HKSAR Government under the ITSP-Platform grant (Ref: ITS/335/23FP) and the InnoHK initiative (TransGP project). Part of the research was conducted in the JC STEM Lab of Robotics for Soft Materials, funded by The Hong Kong Jockey Club Charities Trust. We are grateful to Yiduo Hao, Chuan Guo, Chen Wang, Zitong Lan, and Peter Qingxuan Wu for their insightful discussions and constructive feedback, which have been helpful to the development of this research.

#### A Dataset Statistics

We present more statistics on our dataset regarding the audio scenarios and the the motion types implemented in the collection of the dataset.

Spatial Audio Scenarios The SAM dataset covers 27 common spatial audio scenarios, as listed in Tab. A5. Each scenario is accompanied by two or three audio clips and for each audio clip 48 10-second motion sequences are collected.

##### Scenario

aircraft bark bicycle bell

bird call cat church bell cough crowd yell

drum engine explosion

fire alarm firework gunshot insect instrument laughter music phone ringing shout

sing siren speech thunder vehicle wind rain

Table A5: The 27 common spatial audio settings covered in SAM, each with two or three 10-second audio clips and around 100 motion sequences.

Motion Types The SAM dataset covers 20 common reactions to spatial audio in daily life, as presented in Tab. A6. Each motion type includes at least 10 minutes of motion sequences, ensuring a balanced dataset.

##### Motion

look at stand still cover ears step aside run look around

shake dance wave hands change trajectory squat nod head

clap hands laugh stretch shout curl cover nose pray jump

Table A6: The 20 common reactions excluding the motion genres covered in SAM.

Motion Genres Three general motion genres are covered in the SAM dataset—dull, neutral, and sensitive. The intensity and reaction speed decreased from sensitive to neutral to dull. The motion types listed in Tab. A6 can thus be further divided into motion types with genres, as listed in Tab. A7, A8, A9.

##### Dull Motion

change trajectory clap hands dance

laugh look around look at nod head pray run

shout stand still step aside stretch wave hands

Table A7: Dull motion types.

##### Neutral Motion

change trajectory clap hands cover ears cover nose curl dance laugh look around look at

nod head pray run shake stand still step aside stretch wave hands

Table A8: Neutral motion types.

##### Sensitive Motion

change trajectory clap hands cover ears cover nose curl dance

jump laugh look around look at nod head pray

run shake squat stand still step aside wave hands

Table A9: Sensitive motion types.

#### B Implementation Details

##### B.1 More Implementation Details

We present more details regarding the audio features extracted. The full list of audio features and their corresponding dimensions are listed in Tab. B10. During feature extraction of the audio, we use a sampling rate of 30,720 Hz, an FFT window length of 2048, and a hop length of 256. The Short-Time Fourier Transform (STFT) windowing function is the Hann window [6]. The final shape of the audio features a is thus a ∈ RT×2272, where T=240 is the number of frames in each motion sequence.

|Feature Name<br><br>|Dimension|
|---|---|
|MFCC MFCC delta constant-Q chromagram STFT chromagram onset strength tempogram one-hot beats RMS energy active frames<br><br>|20 20 12 12 1 1068 1 1 1|

Table B10: The extracted audio features for a single ear comprise a 1136-dimensional vector. The total dimension of the audio feature vector a is 2272.

##### B.2 Feature Extractor Implementation Details

Following the feature extractor architecture from [30], our framework incorporates two bidirectional GRUs (Bi-GRUs) and a motion sequence autoencoder, as illustrated in Fig. B10. The Bi-GRUs are applied to extracting condition and motion features respectively, each employ 4 layers with a hidden size of 1024. The audio features undergo an initial projection to 1020 dimensions, followed by concatenation with the sound source location and genre information, resulting in a 1024-dimensional condition feature vector. The motion autoencoder consists of a transformer encoder-decoder structure, where each component contains 4 layers with 4 attention heads and has a latent dimension of 512.

[Figure 17]

Figure B10: The feature extractor framework consists of a motion autoencoder co-trained with two Bi-GRU modules functioning as feature extractors (condition encoder and motion encoder). A reconstruction loss is applied between the ground-truth motions x and the decoded motions x′ produced by the motion autoencoder. Additionally, a contrastive loss operates on the extracted condition features c and motion features m.

Notably, the Bi-GRU used to extract features from motion sequences takes the motion autoencoder’s output as its input.

The feature extractor is optimized using two loss functions. Consistent with the implementation in [30], we employ a contrastive loss [32] defined as:

Dc,m = ∥c − m∥2 Lcta = (1 − y)Dc,m2 + (y)max(0,m − Dc,m)2,

where c denotes condition features, m represents motion features, and y = 0 for matched conditionmotion pairs, otherwise y = 1. This contrastive loss segregates the feature space by minimizing distances between matched pairs while enforcing a minimum separation margin m between mismatched pairs. We set m = 10 here. Additionally, we apply a reconstruction loss:

###### Lrecon = ∥x′ − x∥22,

to guide the learning of the motion autoencoder, where x′ denotes reconstructed motion sequences from the transformer decoder and x represents ground-truth motion sequences. These sequences maintain the identical RT×300 dimensionality format used during MOSPA’s training.

We employ the Adam optimizer [43] with a learning rate of 5×10−5. The feature extractor undergoes training for 1,500 epochs with a batch size of 64, with the motion autoencoder frozen after the initial 1,000 epochs to concentrate optimization efforts on the two Bi-GRU modules in the later 500 epochs.

R-precision For each motion sequence and its extracted motion features, a condition feature pool is constructed comprising the ground-truth matched condition feature along with 31 randomly selected mismatched condition features from the test dataset serving as distractors. The R-precision metrics are then computed by ranking the Euclidean distances between the extracted condition features and motion features, then determining the probability that the ground-truth condition appears among the top-1, top-2, and top-3 ranked positions.

Fréchet Inception Distance (FID) The Fréchet Inception Distance (FID) is computed between motion features derived from ground-truth motion sequences in the test dataset and corresponding generated motion sequences, serving as a metric for assessing the quality and fidelity of the synthesized motions.

Diversity Following the methodology in [30], diversity is computed to quantify the variance of the generated motions. Two sets of generated motions, each of size Sd = 64, are randomly sampled from the test dataset: {m1,m2,··· ,mS

}. The diversity metric is then calculated as:

##### d} and {m′1,m′2,··· ,m′S

d

Sd

1 Sd

∥mi − m′i∥.

Diversity =

i=1

#### C User Study Design

We conducted a user study to evaluate the perceptual quality of motion generation. A total of 25 participants assessed five models (MOSPA, EDGE, POPDG, LODGE, and Bailando) alongside the ground truth (GT). They were asked to select the best motion based on the following criteria:

- • Human Intent Alignment: Does the motion align with real-world intent?
- • Motion Quality: Which motion exhibits the highest movement quality?
- • GT Similarity: Which motion best matches the GT?

To facilitate evaluation, we provided both the GT motion and a textual description as references. Fig. C11 presents a screenshot of the user study interface.

Further Details on the User Study for Spatial Audio-driven Motion Generation. Before starting the study, users are instructed to:

- • Stay in a quiet environment;
- • Wear binaural headphones;
- • Carefully read all instructions.

Each case presents six videos with spatial audio:

- • Videos 1–5 are generated by different methods.
- • The rightmost video is the Ground Truth (GT), shown in green.

After watching and listening to each row of videos, participants are asked to answer:

- (1) Which video best aligns with real-world human Intent in motion generation?
- (2) Which video exhibits the highest Motion Quality of body movement?
- (3) Which video best Matches the Ground Truth motion (rightmost)?

The following are the 10 testing cases and their corresponding instructions: ...

### User Study: Spatial Audio Driven Human Motion Generation

Before starting the study, ensure you are in a quiet environment, wearing headphones, and carefully read the following instructions:

You will watch six videos with spatial audio: the rightmost is the ground truth (GT) in green, while the remaining five are generated by different methods.

Carefully watch and listen to each row of videos and answer the three corresponding questions.

Binaural headphones are required!

###### Case 1

[Figure 18]

[Figure 19]

[Figure 20]

Video 1 Video 2 Video 3

[Figure 21]

[Figure 22]

[Figure 23]

Video 4 Video 5 Ground Truth Reference Motion

Text Description:

Cover ears and step back when hearing engine (sensitive)

Among Videos 1 to 5, which one best aligns with real-world human Intent in motion generation?

Video 1 Video 2 Video 3 Video 4 Video 5

Among Videos 1 to 5, which one exhibits the highest motion Quality of body movement?

Video 1 Video 2 Video 3 Video 4 Video 5

Figure C11: Screenshot of the user study interface.

#### References

- [1] Simon Alexanderson, Rajmund Nagy, Jonas Beskow, and Gustav Eje Henter. Listen, denoise, action! audio-driven motion synthesis with diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–20, 2023.
- [2] Tenglong Ao, Qingzhe Gao, Yuke Lou, Baoquan Chen, and Libin Liu. Rhythmic gesticulator: Rhythmaware co-speech gesture synthesis with hierarchical neural embeddings. ACM Transactions on Graphics (TOG), 41(6):1–19, 2022.
- [3] Tenglong Ao, Zeyi Zhang, and Libin Liu. Gesturediffuclip: Gesture diffusion model with clip latents. ACM Transactions on Graphics (TOG), 42(4):1–18, 2023.
- [4] German Barquero, Sergio Escalera, and Cristina Palmero. Belfusion: Latent diffusion for behavior-driven human motion prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2317–2327, 2023.
- [5] Yuxuan Bian, Ailing Zeng, Xuan Ju, Xian Liu, Zhaoyang Zhang, Wei Liu, and Qiang Xu. Motioncraft: Crafting whole-body motion with plug-and-play multimodal controls. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 1880–1888, 2025.
- [6] RB Blackman and JW Tukey. The measurement of power spectra dover publications. Inc, New York, 1958.
- [7] Sebastian Böck and Gerhard Widmer. Maximum filter vibrato suppression for onset detection. In Proc. of the 16th Int. Conf. on Digital Audio Effects (DAFx). Maynooth, Ireland (Sept 2013), volume 7, page 4. Citeseer, 2013.
- [8] Zhi Cen, Huaijin Pi, Sida Peng, Zehong Shen, Minghui Yang, Shuai Zhu, Hujun Bao, and Xiaowei Zhou. Generating human motion in 3d scenes from text descriptions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1855–1866, 2024.
- [9] Ling-Hao Chen, Shunlin Lu, Wenxun Dai, Zhiyang Dou, Xuan Ju, Jingbo Wang, Taku Komura, and Lei Zhang. Pay attention and move better: Harnessing attention for interactive motion generation and training-free editing. arXiv preprint arXiv:2410.18977, 2024.
- [10] Ling-Hao Chen, Jiawei Zhang, Yewen Li, Yiren Pang, Xiaobo Xia, and Tongliang Liu. Humanmac: Masked motion completion for human motion prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9544–9555, 2023.
- [11] Rui Chen, Mingyi Shi, Shaoli Huang, Ping Tan, Taku Komura, and Xuelin Chen. Taming diffusion probabilistic models for character control. In ACM SIGGRAPH 2024 Conference Papers, pages 1–10, 2024.
- [12] Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu. Executing your commands via motion diffusion in latent space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18000–18010, 2023.
- [13] Qingrong Cheng, Xu Li, and Xinghui Fu. Siggesture: Generalized co-speech gesture synthesis via semantic injection with large-scale pre-training diffusion models. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024.
- [14] Peishan Cong, Ziyi Wang, Zhiyang Dou, Yiming Ren, Wei Yin, Kai Cheng, Yujing Sun, Xiaoxiao Long, Xinge Zhu, and Yuexin Ma. Laserhuman: language-guided scene-aware human motion generation in free environment. arXiv preprint arXiv:2403.13307, 2024.
- [15] Rishabh Dabral, Muhammad Hamza Mughal, Vladislav Golyanik, and Christian Theobalt. Mofusion: A framework for denoising-diffusion-based motion synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9760–9770, 2023.
- [16] Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang. Motionlcm: Real-time controllable motion generation via latent consistency model. In European Conference on Computer Vision, pages 390–408. Springer, 2024.
- [17] Steven Davis and Paul Mermelstein. Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. IEEE transactions on acoustics, speech, and signal processing, 28(4):357–366, 1980.
- [18] Deity Microphones. Pr-2. https://deitymic.com/products/pr-2/, 2025. Accessed: Feb 5, 2025.
- [19] Zhiyang Dou, Xuelin Chen, Qingnan Fan, Taku Komura, and Wenping Wang. C· ase: Learning conditional adversarial skill embeddings for physics-based characters. In SIGGRAPH Asia 2023 Conference Papers, pages 1–11, 2023.
- [20] Daniel PW Ellis. Beat tracking by dynamic programming. Journal of New Music Research, 36(1):51–60, 2007.

- [21] Di Fan, Lili Wan, Wanru Xu, and Shenghui Wang. A bi-directional attention guided cross-modal network for music based dance generation. Computers and Electrical Engineering, 103:108310, 2022.
- [22] Ke Fan, Shunlin Lu, Minyue Dai, Runyi Yu, Lixing Xiao, Zhiyang Dou, Junting Dong, Lizhuang Ma, and Jingbo Wang. Go to zero: Towards zero-shot motion generation with million-scale data. arXiv preprint arXiv:2507.07095, 2025.
- [23] Yuming Feng, Zhiyang Dou, Ling-Hao Chen, Yuan Liu, Tianyu Li, Jingbo Wang, Zeyu Cao, Wenping Wang, Taku Komura, and Lingjie Liu. Motionwavelet: Human motion prediction via wavelet manifold learning. arXiv preprint arXiv:2411.16964, 2024.
- [24] Chuang Gan, Hang Zhao, Peihao Chen, David Cox, and Antonio Torralba. Self-supervised moving vehicle tracking with stereo sound. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7053–7062, 2019.
- [25] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 776–780. IEEE, 2017.
- [26] Shiry Ginosar, Amir Bar, Gefen Kohavi, Caroline Chan, Andrew Owens, and Jitendra Malik. Learning individual styles of conversational gesture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3497–3506, 2019.
- [27] Kehong Gong, Dongze Lian, Heng Chang, Chuan Guo, Zihang Jiang, Xinxin Zuo, Michael Bi Mi, and Xinchao Wang. Tm2d: Bimodality driven 3d dance generation via music-text integration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9942–9952, 2023.
- [28] Peter Grosche, Meinard Müller, and Frank Kurth. Cyclic tempogram—a mid-level tempo representation for musicsignals. In 2010 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 5522–5525. IEEE, 2010.
- [29] Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, and Li Cheng. Momask: Generative masked modeling of 3d human motions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1900–1910, 2024.
- [30] Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. Generating diverse and natural 3d human motions from text. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5152–5161, 2022.
- [31] Chuan Guo, Xinxin Zuo, Sen Wang, Shihao Zou, Qingyao Sun, Annan Deng, Minglun Gong, and Li Cheng. Action2motion: Conditioned generation of 3d human motions. In Proceedings of the 28th ACM International Conference on Multimedia, pages 2021–2029, 2020.
- [32] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In 2006 IEEE computer society conference on computer vision and pattern recognition (CVPR’06), volume 2, pages 1735–1742. IEEE, 2006.
- [33] Mohamed Hassan, Duygu Ceylan, Ruben Villegas, Jun Saito, Jimei Yang, Yi Zhou, and Michael J Black. Stochastic scene-aware motion prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11374–11384, 2021.
- [34] Tairan He, Jiawei Gao, Wenli Xiao, Yuanhang Zhang, Zi Wang, Jiashun Wang, Zhengyi Luo, Guanqi He, Nikhil Sobanbab, Chaoyi Pan, et al. Asap: Aligning simulation and real-world physics for learning agile humanoid whole-body skills. arXiv preprint arXiv:2502.01143, 2025.
- [35] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [36] Daniel Holden, Taku Komura, and Jun Saito. Phase-functioned neural networks for character control. ACM Transactions on Graphics (TOG), 36(4):1–13, 2017.
- [37] Chao Huang, Dejan Markovi´c, Chenliang Xu, and Alexander Richard. Modeling and driving human body soundfields through acoustic primitives. In European Conference on Computer Vision, pages 1–17. Springer, 2024.
- [38] Yiming Huang, Zhiyang Dou, and Lingjie Liu. Modskill: Physical character skill modularization. arXiv preprint arXiv:2502.14140, 2025.
- [39] Yiming Huang, Weilin Wan, Yue Yang, Chris Callison-Burch, Mark Yatskar, and Lingjie Liu. Como: Controllable motion generation through language guided pose code editing. In European Conference on Computer Vision, pages 180–196. Springer, 2024.
- [40] Yinghao Huang, Leo Ho, Dafei Qin, Mingyi Shi, and Taku Komura. Interact: Capture and modelling of realistic, expressive and interactive activities between two persons in daily scenarios. arXiv preprint arXiv:2405.11690, 2024.

- [41] Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. Advances in Neural Information Processing Systems, 36:20067–20079, 2023.
- [42] Jaeyeon Kim, Heeseung Yun, and Gunhee Kim. Visage: Video-to-spatial audio generation. arXiv preprint arXiv:2506.12199, 2025.
- [43] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [44] Zitong Lan, Yiduo Hao, and Mingmin Zhao. Resounding acoustic fields with reciprocity, 2025.
- [45] Zitong Lan, Chenhao Zheng, Zhiwei Zheng, and Mingmin Zhao. Acoustic volume rendering for neural impulse response fields. arXiv preprint arXiv:2411.06307, 2024.
- [46] Nhat Le, Tuong Do, Khoa Do, Hien Nguyen, Erman Tjiputra, Quang D Tran, and Anh Nguyen. Controllable group choreography using contrastive diffusion. ACM Transactions on Graphics (TOG), 42(6):1–14, 2023.
- [47] Buyu Li, Yongchi Zhao, Shi Zhelun, and Lu Sheng. Danceformer: Music conditioned 3d dance generation with parametric motion transformer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 1272–1279, 2022.
- [48] Ronghui Li, YuXiang Zhang, Yachao Zhang, Hongwen Zhang, Jie Guo, Yan Zhang, Yebin Liu, and Xiu Li. Lodge: A coarse to fine diffusion network for long dance generation guided by the characteristic dance primitives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1524–1534, 2024.
- [49] Ronghui Li, Junfan Zhao, Yachao Zhang, Mingyang Su, Zeping Ren, Han Zhang, Yansong Tang, and Xiu Li. Finedance: A fine-grained choreography dataset for 3d full body dance generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10234–10243, 2023.
- [50] Ruilong Li, Shan Yang, David A Ross, and Angjoo Kanazawa. Ai choreographer: Music conditioned 3d dance generation with aist++. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13401–13412, 2021.
- [51] Zhouyingcheng Liao, Mingyuan Zhang, Wenjia Wang, Lei Yang, and Taku Komura. Rmd: A simple baseline for more general human motion generation via training-free retrieval-augmented motion diffuse. arXiv preprint arXiv:2412.04343, 2024.
- [52] Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. Motionx: A large-scale 3d expressive whole-body human motion dataset. Advances in Neural Information Processing Systems, 36:25268–25280, 2023.
- [53] Jinpeng Liu, Wenxun Dai, Chunyu Wang, Yiji Cheng, Yansong Tang, and Xin Tong. Plan, posture and go: Towards open-world text-to-motion generation. arXiv preprint arXiv:2312.14828, 2023.
- [54] Xian Liu, Qianyi Wu, Hang Zhou, Yinghao Xu, Rui Qian, Xinyi Lin, Xiaowei Zhou, Wayne Wu, Bo Dai, and Bolei Zhou. Learning hierarchical cross-modal association for co-speech gesture generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10462– 10472, 2022.
- [55] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [56] Vicon Motion Systems Ltd. Vicon motion capture systems, 2025. Accessed: Feb 5, 2025.
- [57] Shunlin Lu, Ling-Hao Chen, Ailing Zeng, Jing Lin, Ruimao Zhang, Lei Zhang, and Heung-Yeung Shum. Humantomato: Text-aligned whole-body motion generation. arXiv preprint arXiv:2310.12978, 2023.
- [58] Shunlin Lu, Jingbo Wang, Zeyu Lu, Ling-Hao Chen, Wenxun Dai, Junting Dong, Zhiyang Dou, Bo Dai, and Ruimao Zhang. Scamo: Exploring the scaling law in autoregressive motion generation model. arXiv preprint arXiv:2412.14559, 2024.
- [59] Zhengyi Luo, Jinkun Cao, Kris Kitani, Weipeng Xu, et al. Perpetual humanoid control for real-time simulated avatars. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10895–10904, 2023.
- [60] Zhenye Luo, Min Ren, Xuecai Hu, Yongzhen Huang, and Li Yao. Popdg: Popular 3d dance generation with popdanceset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26984–26993, 2024.
- [61] Brian McFee, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. librosa: Audio and music signal analysis in python. SciPy, 2015:18–24, 2015.
- [62] Liang Pan, Zeshi Yang, Zhiyang Dou, Wenjia Wang, Buzhen Huang, Bo Dai, Taku Komura, and Jingbo Wang. Tokenhsi: Unified synthesis of physical human-scene interactions through task tokenization. arXiv preprint arXiv:2503.19901, 2025.

- [63] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10975–10985, 2019.
- [64] Huaijin Pi, Zhi Cen, Zhiyang Dou, and Taku Komura. Coda: Coordinated diffusion noise optimization for whole-body manipulation of articulated objects. arXiv preprint arXiv:2505.21437, 2025.
- [65] Shenhan Qian, Zhi Tu, Yihao Zhi, Wen Liu, and Shenghua Gao. Speech drives templates: Co-speech gesture synthesis with learned templates. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11077–11086, 2021.
- [66] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- [67] Christian Schörkhuber and Anssi Klapuri. Constant-q transform toolbox for music processing. In 7th sound and music computing conference, Barcelona, Spain, pages 3–64. SMC, 2010.
- [68] Mingyi Shi, Dafei Qin, Leo Ho, Zhouyingcheng Liao, Yinghao Huang, Junichi Yamagishi, and Taku Komura. It takes two: Real-time co-speech two-person’s interaction generation via reactive auto-regressive diffusion model. arXiv preprint arXiv:2412.02419, 2024.
- [69] Yi Shi, Jingbo Wang, Xuekun Jiang, Bingkun Lin, Bo Dai, and Xue Bin Peng. Interactive character control with auto-regressive motion diffusion models. ACM Transactions on Graphics (TOG), 43(4):1–14, 2024.
- [70] Li Siyao, Tianpei Gu, Zhitao Yang, Zhengyu Lin, Ziwei Liu, Henghui Ding, Lei Yang, and Chen Change Loy. Duolando: Follower gpt with off-policy reinforcement learning for dance accompaniment. arXiv preprint arXiv:2403.18811, 2024.
- [71] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. Bailando: 3d dance generation by actor-critic gpt with choreographic memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11050–11059, 2022.
- [72] Sebastian Starke, Ian Mason, and Taku Komura. Deepphase: Periodic autoencoders for learning motion phase manifolds. ACM Transactions on Graphics (TOG), 41(4):1–13, 2022.
- [73] Jiarui Sun and Girish Chowdhary. Comusion: Towards consistent stochastic human motion prediction via motion diffusion. In European Conference on Computer Vision, pages 18–36. Springer, 2024.
- [74] Peiwen Sun, Sitong Cheng, Xiangtai Li, Zhen Ye, Huadai Liu, Honggang Zhang, Wei Xue, and Yike Guo. Both ears wide open: Towards language-driven spatial audio generation. arXiv preprint arXiv:2410.10676, 2024.
- [75] Taoran Tang, Jia Jia, and Hanyang Mao. Dance with melody: An lstm-autoencoder approach to musicoriented dance synthesis. In Proceedings of the 26th ACM international conference on Multimedia, pages 1598–1606, 2018.
- [76] Chen Tessler, Yunrong Guo, Ofir Nabati, Gal Chechik, and Xue Bin Peng. Maskedmimic: Unified physics-based character control through masked motion inpainting. ACM Transactions on Graphics (TOG), 43(6):1–21, 2024.
- [77] Guy Tevet, Sigal Raab, Brian Gordon, Yoni Shafir, Daniel Cohen-or, and Amit Haim Bermano. Human motion diffusion model. In The Eleventh International Conference on Learning Representations, 2023.
- [78] Jonathan Tseng, Rodrigo Castellon, and Karen Liu. Edge: Editable dance generation from music. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 448–458, 2023.
- [79] Weilin Wan, Zhiyang Dou, Taku Komura, Wenping Wang, Dinesh Jayaraman, and Lingjie Liu. Tlcontrol: Trajectory and language control for human motion synthesis. arXiv preprint arXiv:2311.17135, 2023.
- [80] Jingbo Wang, Yu Rong, Jingyuan Liu, Sijie Yan, Dahua Lin, and Bo Dai. Towards diverse and natural scene-aware 3d human motion synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20460–20469, 2022.
- [81] Jingbo Wang, Sijie Yan, Bo Dai, and Dahua Lin. Scene-aware generative network for human motion synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12206–12215, 2021.
- [82] Wenjia Wang, Liang Pan, Zhiyang Dou, Jidong Mei, Zhouyingcheng Liao, Yuke Lou, Yifan Wu, Lei Yang, Jingbo Wang, and Taku Komura. Sims: Simulating stylized human-scene interactions with retrieval-augmented script generation. arXiv preprint arXiv:2411.19921, 2024.
- [83] Zan Wang, Yixin Chen, Baoxiong Jia, Puhao Li, Jinlu Zhang, Jingze Zhang, Tengyu Liu, Yixin Zhu, Wei Liang, and Siyuan Huang. Move as you say interact as you can: Language-guided human motion generation with scene affordance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 433–444, 2024.

- [84] Yiming Xie, Varun Jampani, Lei Zhong, Deqing Sun, and Huaizu Jiang. Omnicontrol: Control any joint at any time for human motion generation. arXiv preprint arXiv:2310.08580, 2023.
- [85] Liang Xu, Ziyang Song, Dongliang Wang, Jing Su, Zhicheng Fang, Chenjing Ding, Weihao Gan, Yichao Yan, Xin Jin, Xiaokang Yang, et al. Actformer: A gan-based transformer towards general actionconditioned 3d human motion generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2228–2238, 2023.
- [86] Sirui Xu, Hung Yu Ling, Yu-Xiong Wang, and Liang-Yan Gui. Intermimic: Towards universal whole-body control for physics-based human-object interactions. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12266–12277, 2025.
- [87] Xudong Xu, Dejan Markovic, Jacob Sandakly, Todd Keebler, Steven Krenn, and Alexander Richard. Sounding bodies: modeling 3d spatial sound of humans using body pose and audio. Advances in Neural Information Processing Systems, 36:44740–44752, 2023.
- [88] Han Yang, Kun Su, Yutong Zhang, Jiaben Chen, Kaizhi Qian, Gaowen Liu, and Chuang Gan. Unimumo: Unified text, music and motion generation. arXiv preprint arXiv:2410.04534, 2024.
- [89] Youngwoo Yoon, Bok Cha, Joo-Haeng Lee, Minsu Jang, Jaeyeon Lee, Jaehong Kim, and Geehyuk Lee. Speech gesture generation from the trimodal context of text, audio, and speaker identity. ACM Transactions on Graphics (TOG), 39(6):1–16, 2020.
- [90] Ye Yuan and Kris Kitani. Diverse trajectory forecasting with determinantal point processes. arXiv preprint arXiv:1907.04967, 2019.
- [91] Canyu Zhang, Youbao Tang, Ning Zhang, Ruei-Sung Lin, Mei Han, Jing Xiao, and Song Wang. Bidirectional autoregessive diffusion model for dance generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 687–696, 2024.
- [92] He Zhang, Sebastian Starke, Taku Komura, and Jun Saito. Mode-adaptive neural networks for quadruped motion control. ACM Transactions on Graphics (TOG), 37(4):1–11, 2018.
- [93] Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Yong Zhang, Hongwei Zhao, Hongtao Lu, Xi Shen, and Ying Shan. Generating human motion from textual descriptions with discrete representations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14730–14740, 2023.
- [94] Mingyuan Zhang, Daisheng Jin, Chenyang Gu, Fangzhou Hong, Zhongang Cai, Jingfang Huang, Chongzhi Zhang, Xinying Guo, Lei Yang, Ying He, et al. Large motion model for unified multi-modal motion generation. In European Conference on Computer Vision, pages 397–421. Springer, 2024.
- [95] Yufei Zhang, Jeffrey O Kephart, Zijun Cui, and Qiang Ji. Physpt: Physics-aware pretrained transformer for estimating human dynamics from monocular videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2305–2317, 2024.
- [96] Yufei Zhang, Jeffrey O Kephart, and Qiang Ji. Incorporating physics principles for precise human motion prediction. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 6164–6174, 2024.
- [97] Zeyi Zhang, Tenglong Ao, Yuyao Zhang, Qingzhe Gao, Chuan Lin, Baoquan Chen, and Libin Liu. Semantic gesticulator: Semantics-aware co-speech gesture synthesis. ACM Transactions on Graphics (TOG), 43(4):1–17, 2024.
- [98] Hang Zhao, Chuang Gan, Andrew Rouditchenko, Carl Vondrick, Josh McDermott, and Antonio Torralba. The sound of pixels. In Proceedings of the European conference on computer vision (ECCV), pages 570–586, 2018.
- [99] Zhisheng Zheng, Puyuan Peng, Ziyang Ma, Xie Chen, Eunsol Choi, and David Harwath. Bat: Learning to reason about spatial sounds with large language models. arXiv preprint arXiv:2402.01591, 2024.
- [100] Wenyang Zhou, Zhiyang Dou, Zeyu Cao, Zhouyingcheng Liao, Jingbo Wang, Wenjia Wang, Yuan Liu, Taku Komura, Wenping Wang, and Lingjie Liu. Emdm: Efficient motion diffusion model for fast and high-quality motion generation. In European Conference on Computer Vision, pages 18–38. Springer, 2025.
- [101] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5745–5753, 2019.
- [102] Lingting Zhu, Xian Liu, Xuanyu Liu, Rui Qian, Ziwei Liu, and Lequan Yu. Taming diffusion models for audio-driven co-speech gesture generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10544–10553, 2023.
- [103] Wentao Zhu, Xiaoxuan Ma, Dongwoo Ro, Hai Ci, Jinlu Zhang, Jiaxin Shi, Feng Gao, Qi Tian, and Yizhou Wang. Human motion generation: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.

###### [104] Wenlin Zhuang, Congyi Wang, Jinxiang Chai, Yangang Wang, Ming Shao, and Siyu Xia. Music2dance: Dancenet for music-driven dance generation. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 18(2):1–21, 2022.

