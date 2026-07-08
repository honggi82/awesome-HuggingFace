## MusicInfuser: Making Video Diffusion Listen and Dance

#### Susung Hong Ira Kemelmacher-Shlizerman Brian Curless Steven M. Seitz University of Washington

# arXiv:2503.14505v3[cs.CV]3May2026

[Figure 1]

“a female dancer wearing a Hawaiian dress dancing on Kuhio Beach, front view”

[Figure 2]

“a male dancer dancing on a roo op at sunset, captured from a front view”

[Figure 3]

“a male dancer wearing a chef's uniform dancing in a busy restaurant kitchen with ﬂames from the grill behind him, captured from a front view”

Figure 1. MusicInfuser adapts video diffusion models to music, making them listen and dance according to the music. This adaptation is done in a prior-preserving manner, enabling it to also accept style through the prompt while aligning the movement to the music.

### Abstract

We introduce MusicInfuser, an approach that aligns pretrained text-to-video diffusion models to generate highquality dance videos synchronized with specified music tracks. Rather than training a multimodal audio-video or audio-motion model from scratch, our method demonstrates how existing video diffusion models can be efficiently adapted to align with musical inputs. We propose a novel layer-wise adaptability criterion based on a guidance-inspired constructive influence function to select adaptable layers, significantly reducing training costs while preserving rich prior knowledge, even with limited, specialized datasets. Experiments show that MusicInfuser effec-

tively bridges the gap between music and video, generating novel and diverse dance movements that respond dynamically to music. Furthermore, our framework generalizes well to unseen music tracks, longer video sequences, and unconventional subjects, outperforming baseline models in consistency and synchronization. All of this is achieved without requiring motion data, with training completed on a single GPU within a day.

### 1. Introduction

Today’s leading open-source video diffusion models often produce silent [46] or speech-focused [29, 49] videos.

[Figure 4]

Music-Driven Dance Video Genera on

[Figure 5]

Music-Driven Dancing Skeleton Genera on

Figure 2. Motivational example. Skeletal motion generation [47] produces simplified movements lacking nuances such as backbone curvature, axial rotation, hand articulation, hair dynamics, and clothing motion, resulting in a more limited range of dance compared to video-based dance generation approaches (ours).

While it is possible to add music after the fact, it is difficult to generate motion that is properly synchronized with a specified music track. Alternatively, some research has begun to explore audio-video generation [39]. However, focusing on the specific application of dance, dance videos well-aligned with their music are far rarer than finding general, unconstrained videos, resulting in sub-optimal quality when training audio-video generative models from scratch.

In this paper, we introduce an approach to align pretrained text-to-video models that have useful ingredients for dance. Our method, called MusicInfuser, generates output videos that are synchronized with the input music, with various components such as styles and appearances controllable via text prompts. We focus on synthesizing dance videos, i.e., generating realistic dancing figures that adjust and synchronize to the music, which poses several difficulties that require extensive knowledge about human motion and physics, music, and choreography.

Automatic dance generation must consider style, beat, and the inherently multimodal nature of dance, where multiple valid sequences can follow a given pose [31]. Computational approaches have drawn on choreographic principles [52] and techniques ranging from graph-based methods [12, 28] to deep neural networks [47, 55, 56]. However, traditional methods rely on motion capture [3] or reconstructed motions [33], which are costly or prone to floating/jitter artifacts. In addition, skeletal representations are underparameterized for dance, lacking nuances such as backbone curvature, axial rotation, hand articulation, hair dynamics, and clothing motion (Fig. 2).

MusicInfuser bypasses these limitations by adapting pre-trained text-to-video models [46] with zero-initialized music-video modules injected into DiT blocks. This approach does not require motion capture or reconstruction, relying instead on existing dance videos for alignment. To address the scarcity of high-quality music-aligned dance datasets and reduce fine-tuning costs, we introduce a layerwise adaptability criterion using a guidance-based constructive influence function. This preserves pre-trained knowledge while establishing correlations between music and movement, allowing training on a GPU within a day.

MusicInfuser retains text-based control, enabling users

[Figure 6]

[Figure 7]

[Figure 8]

Figure 3. Using prompts such as “a {marmot, rabbit, dog (top to bottom rows)} dancing ...,” our method generalizes to unseen dancing subjects.

to specify dance style, setting, and other aesthetics (Fig. 1) as well as the number of dancers (Fig. 4) while maintaining music synchronization. Our method generalizes to longer videos with unseen music and even to unseen subjects such as animals (Figs. 3, 6). For evaluation, we introduce an automatic framework based on Video-LLMs [14, 50] that jointly assesses video, audio, and language alignment, correlating well with human judgment.

Our experiments show that MusicInfuser successfully closes the gap between music and dance without intermediate motion data. By leveraging pre-trained video diffusion models through targeted adaptation, it produces highquality, novel dance movements that respond naturally to musical rhythms, offering flexible dance video generation.

### 2. Related Work

Music-to-Dance Generation Early approaches mapped music primitives to dance elements using Hidden Markov Models [36] and graph-based methods with movement transition graphs [28]. Later research integrated Gaussian processes [15], various neural networks [2, 55, 56], and transformers [33, 55]. Traditional methods often produced

[Figure 9]

Male and Female Dancers

[Figure 10]

Mul ple Dancers

[Figure 11]

Group of Dancers

- Figure 4. We can generate group dance videos aligned with music, based on the text.

beat-synchronized movements lacking contextual meaning or showing excessive repetition [4], with limited choreographic diversity [5]. Recent advances have shifted toward diffusion-based approaches [3, 30, 37, 38, 47], while supporting two- or multi-person dance [42, 43]. Unlike these skeleton-based methods, our framework directly synthesizes dance videos by adapting pre-trained text-to-video diffusion models to musical inputs. Without an intermediate representation, our method avoids rigid body parameterization, requires no motion capture or pose reconstruction, and eliminates post-processing to generate dance videos.

Controllable Approaches Dance generation systems have evolved to incorporate multiple input modalities for richer choreographic control [9, 17, 35], with text emerging as a powerful interface for its zero-shot capability and communicating choreographic ideas [35]. Transformerbased approaches using Vector Quantized-Variational Autoencoders create discrete motion tokens processable alongside text [41], while systems now process both text and music inputs simultaneously [17]. The MusicInfuser framework combines the flexibility of text-based interfaces with precise audio synchronization, allowing users to control stylistic and aesthetic elements of generated dance videos through prompts while ensuring movements remain aligned with musical features.

Audio-to-Video Generation Another domain that is adjacent to our method is audio-driven video generation. Pioneering this domain, Sound2Sight [10] introduced a deep variational encoder-decoder framework that predicts future frames by conditioning on both past frames and audio input. TATS [16] addressed audio-to-video generation challenges

through a combination of time-agnostic VQGAN and timesensitive transformer architectures. More recently, leveraging advances in diffusion models [20, 44], joint audiovideo generation methods like MM-Diffusion [39] have been developed, enabling bidirectional generation where either modality can condition the other.

### 3. Preliminaries

Video Diffusion Models Diffusion models [7, 8, 20, 21, 44, 45, 53] represent a family of generative techniques that restore data via iterative denoising steps. The goal is to generate samples from a video distribution p(x). To this end, we can define a convoluted distribution of p(x) and a Gaussian distribution with standard deviation σ, namely p(x,σ). In this paper, we follow [27] to construct a compact formulation of diffusion models. The denoiser Dθ is optimized with the following L2 objective:

train,n∼N(0,σ2I)∥Dθ(y + n;σ) − y∥22, (1)

L = Ey∼p,σ∼Σ

where Σtrain denotes a noise distribution from which we sample noise during training, which is typically a uniform distribution. To sample with the denoiser Dθ, the ODE representing the change in the sample x with the change in σ can be defined as:

Dθ(x;σ) − x σ

dx dσ

. (2)

= −

Text-Conditional Generation In a similar way, we can construct a conditional denoiser Dθ(x|c;σ) by training with a condition c paired with each y and replace the sampling process with a conditional denoiser. To boost generated content quality and alignment with prompts, classifierfree guidance (CFG) [19] has become widely used. Applying CFG, the modified ODE then becomes the linear combination form:

dx dσ

= −γcfg

Dθ(x|c;σ) − x σ

+ (γcfg − 1)

Dθ(x;σ) − x σ

(3)

In this formulation, Dθ(x;σ) shares the same parameters as Dθ(x|c;σ) but is trained by randomly omitting conditional information during training, and parameter γcfg denotes the guidance scale.

.

### 4. MusicInfuser

Text-to-video models already know how to dance. In contrast to previous multimodal dance generation methods, video diffusion models trained on massive and diverse video datasets have already internalized sophisticated representations of human motion, including intricate and expressive movements. They have implicitly learned choreographic patterns, style variations, and the general physics of

DiT Tokens

DiT Tokens

DiT Tokens DiT Block ❄

DiT Tokens Audio Tokens

[Figure 12]

[Figure 13]

[Figure 14]

DiT Block ❄

DiT Block ❄

[Figure 15]

[Figure 16]

ZICA Block 🔥

ZICA Block 🔥

Projec on

[Figure 17]

DiT Block ❄

[Figure 18]

[Figure 19]

[Figure 20]

DiT Block ❄

DiT Block ❄

ZICA Block 🔥

WQ WK WV

[Figure 21]

ZICA Block 🔥

Layer Adaptability

[Figure 22]

[Figure 23]

DiT Block ❄

DiT Block ❄

Audio Tokens

Audio Tokens

Audio Tokens

[Figure 24]

[Figure 25]

ZICA Block 🔥

DiT Block ❄

[Figure 26]

DiT Block ❄

[Figure 27]

ZICA Block 🔥

Mul modal Cross A en on

[Figure 28]

DiT Block ❄

[Figure 29]

DiT Block ❄

[Figure 30]

DiT Block ❄

[Figure 31]

[Figure 32]

DiT Block ❄

ZICA Block 🔥

Zero WO

[Figure 33]

[Figure 34]

ZICA Block 🔥

DiT Block ❄

[Figure 35]

DiT Block ❄

[Figure 36]

[Figure 37]

[Figure 38]

DiT Block ❄

DiT Block ❄

ZICA Block 🔥

(a)

(b)

(c)

(d)

- Figure 5. Zero-initialized cross-attention (ZICA) block. The output projection is initialized with a zero matrix, making the cross-attention block act as an identity function at the beginning. (b–d) illustrate several baseline layer selection strategies when the number of ZICA blocks is fewer than that of DiT blocks. (b) Attaching cross-attention blocks evenly across DiT blocks. (c) Attaching the blocks evenly to the earliest layers. (d) Attaching the blocks based on pre-computed layer adaptabilities (Sec. 4.1). Table 1 shows the results.

a human body during their extensive training, providing a valuable foundation that can be leveraged for music-driven video generation.

Considering this, our goal is to align the models to musical input a with adaptation parameters ϕ to construct a final denoiser, Dθ,ϕ(x|c,a;σ), while preserving the pre-trained model’s knowledge about dance. For the rest of this paper, we call the probability distribution characterized by the pretrained text-to-video denoiser Dθ(x|c;σ) the prior, since it denotes a learned prior video distribution not conditioned on audio. Accordingly, our new continual optimization objective is as follows:

mm,σ∼Σtrain,n∼N(0,σ2I)∥Dθ,ϕ(y+n|c,a;σ)−y∥22,

L = E(y,c,a)∼p

where pmm is a joint data distribution of video, caption, and audio.

Unfortunately, we recognize that specialized dancing datasets are notably scarcer than general video datasets used for pre-training and thus inevitably contain biases that compromise the prior model’s generalization and denoising capabilities. Moreover, significant resources are required for fine-tuning text-to-video diffusion models. We address this challenge through a carefully balanced adaptation mechanism that preserves the rich prior while establishing robust correlations between musical features and dance movements with a significantly lower cost.

##### 4.1. Measuring Layer Adaptability

Cross-attention is effective for conditioning on auxiliary modalities [13, 18, 40]. However, applying cross-attention mechanisms to all layers of a model poses challenges: 1) it incurs substantial computational costs, and 2) it can degrade the denoising capabilities of pre-trained diffusion models (see “All Layers” in Table 1), especially in low-data regimes such as professional dancing. Consequently, identifying an optimal subset of layers for cross-attention adaptation is im-

portant to preserve both the denoising effectiveness and the generalization capability of the pre-trained model.

Unfortunately, finding this optimal combination through exhaustive search by fine-tuning every possible configuration is infeasible. For example, for a pre-trained model with 48 layers, the number of possible combinations for adapting only one-third of the layers with cross-attention is

48 16 > 2 × 1012. Two intuitive layer selection approaches, shown in (b) and (c) of Fig. 5, reduce computational costs. However, both methods fail to account for the behavior of the layers and compromise the model’s capabilities, specifically degrading video outputs (Table 1).

To address this, we propose a principled, constructive metric for layer selection based on adaptability. Instead of measuring importance by performance degradation when a layer is removed, we measure each layer’s positive influence by using it as guidance during inference. This way, we can use existing evaluation metrics for videos [25] to measure and precompute the influence of each layer without the risk of out-of-distribution issues. Specifically, this criterion quantifies each layer’s influence by performing guided sampling while leveraging the pre-trained model without the layer to provide guidance [1, 22, 26]. The layer skip guidance can be formulated as the derivative of the implicit energy function [23, 26]:

∇xGl = DθL(x|c;σ) − DθL\{l}(x|c;σ) /σ, (4)

where L represents the complete layer set, while DθL and DθL\{l} denote the full-layer diffusion transformer denoiser and the variant skipping layer l ∈ L, respectively. Then, we define the improvement observed in the resulting videos as layer adaptability (see the supplementary material for details). Intuitively, layers that are more intrinsically connected to the structural and perceptual quality of video content exhibit greater performance when excluded and used for guidance than those primarily involved in local denois-

[Figure 39]

- Figure 6. Generalization capabilities in terms of music length and type. MusicInfuser can generate multiple times longer dance videos that are multiple times longer than the videos used for training. For each row, we use synthetic in-the-wild music tracks with a keyword “K-pop,” a type of music not existing in AIST [48], and use a prompt “a professional female dancer dancing K-pop ....” This shows our method is highly generalizable, even extending to longer videos with an unseen category of the music. The beat and style alignment can be more clearly observed in the supplementary video.

ing. Our method thus identifies layers where modulation can effectively influence motion and structure, eliminating the need to train separate video models for each layer combination and avoiding significant deviations from the learned denoising manifold during adaptation.

##### 4.2. Beta-Uniform Scheduling

Diffusion models, including those using LoRA fine-tuning, typically employ a uniform distribution for noise sampling throughout training. For adapter training, we aim to preserve the denoising capability of the pre-trained model by initially focusing on low-noise levels and gradually learning the more substantial components over the course of training. To achieve this, we propose a Beta-Uniform scheduling strategy that evolves the training noise distribution Σtrain from a Beta distribution concentrated on low noise levels to a uniform distribution.

The Beta distribution with parameters α = 1 and β is formally defined by the probability density function:

(1 − x)β−1 B(1,β)

, 0 ≤ x ≤ 1 (5)

f(x;α = 1,β) =

where B(α,β) is the Beta function serving as a normalization constant. When β > 1, the distribution Beta(1,β) concentrates probability mass near zero, which in our diffusion framework corresponds to sampling predominantly smaller noise scales. As β decays toward 1, the distribution gradually flattens, approaching Uniform(0,1), i.e.,

limβ→1 f(x;1,β) = 1, for all 0 ≤ x ≤ 1.

This causes a smooth transition from focusing on highfrequency components at lower noise levels to equally considering all frequencies. By first influencing the taskspecific fine components of the dance and then the fundamental structure of dance movements, our approach preserves the pre-trained knowledge of general physics of human motion and produces more coherent dance sequences.

##### 4.3. Zero-Initialized Adaptation Modules

To extend pre-trained diffusion transformers to new modalities while maintaining stable training, we introduce zeroinitialized adaptation modules that start with zero parameters and gradually learn to influence the model. Specifically, we employ Zero-Initialized Cross-Attention (ZICA) for multimodal conditioning and Low-Rank Adaptors (LoRA) for domain and motion adaptation with the new modality.

Random initialization of cross-attention modules can bias predictions and destabilize continual training. ZICA addresses this by initializing the output projection to zero, so that cross-attention initially behaves as an identity mapping and gradually incorporates information from the conditioning modality. Let A ∈ RN

V ×d denote projected audio and video tokens. The cross-attention with output projection is

A×d and V ∈ RN

Z = V + WO softmax

###### VWQ(AWK)⊤

√

d

###### AWV , (6)

[Figure 40]

[Figure 41]

[Figure 42]

Top row: slowed down, middle row: original speed, bottom row: sped up

- Figure 7. Speed control. The audio input is slowed down (top row) or sped up (bottom row) by factors of 0.75 and 1.25, respectively. This shows that speeding up audio generally results in sped-up movement. Note also the change in dynamics, as speeding up the audio increases the musical tone. More examples of audio speeding up and slowing down are included in the supplementary video.

Style Beat Body Movement Choreography Dance Quality Imaging Aesthetic Overall Video Quality

Model

Overall

Alignment Alignment Representation Realism Complexity Average Quality Quality Consistency Average

Ours 8.95 9.54 10.00 7.36 5.25 8.22 7.08 7.01 9.96 8.02 8.14 All Layers 8.37 9.02 9.55 7.02 5.35 7.86 6.00 7.17 9.95 7.71 7.80 Evenly Distributed Layers 8.15 9.01 9.95 6.59 5.08 7.76 5.93 6.61 9.66 7.40 7.62 First Layers 8.67 9.44 9.90 7.20 5.57 8.16 6.36 6.88 9.89 7.71 7.99 Middle Layers 9.05 9.39 9.05 6.67 5.56 7.94 5.84 6.80 9.78 7.47 7.77 Last Layers 8.60 9.34 9.80 6.91 5.45 8.02 6.03 6.70 9.81 7.51 7.83 Feature Addition 8.60 9.37 9.90 6.92 5.41 8.04 6.29 6.90 9.94 7.71 7.92 No Beta-Uniform Schedule 8.93 9.42 9.40 6.67 5.61 8.01 6.20 6.99 9.93 7.71 7.89 No In-the-Wild Data 8.80 9.52 9.75 7.17 5.39 8.13 6.46 6.56 9.96 7.66 7.95 No Cross Attention Zero Init. 8.64 9.25 9.95 7.05 5.28 8.03 6.56 6.91 9.98 7.82 7.95

- Table 1. Evaluation with Qwen3-Omni [50] shows various baseline cross-attention adaptation strategies. Feature Addition refers to directly adding audio features, inspired by image conditioning in ControlNet [54]. Results using VideoLLaMA 2 [14] are provided in the supplementary material, which show a similar trend in layer selection.

where WO is initialized to zero. The module initially acts as an identity mapping, and as WO moves away from zero during training, it gradually integrates audio features.

Similarly, attention weights are adapted using LoRA [24], which decomposes updates into low-rank matrices initialized to zero. Conventional LoRA ranks (8–16) for image models are often insufficient for video transformers, which must capture temporal dependencies. We employ higher-rank configurations, since adapting temporal transformations generally requires higher rank. For example, full homography transformations requires increasing the rank by at least 8, and modeling complex human motion benefits from even higher ranks. LoRA, which is zero-initialized, parallels ZICA by gradually learning task-specific modifications from an initially neutral state, progressively adapting the diffusion network to the new domain and modality information.

##### 4.4. Utilizing In-the-Wild Data

Training exclusively on datasets in highly constrained settings [33, 48] can lead to reduced generalizability and model degradation when confronted with diverse real-world scenarios. Therefore, we use a mixture of in-the-wild data and the constrained datasets. These videos introduce diversity in terms of camera trajectories, lighting conditions, performance environments, and dance styles. The inclusion of in-the-wild data serves as regularization, preventing overfitting to specific dance patterns or environmental settings. Details are provided in the supplementary material.

##### 4.5. Prompt Diversification

We use caption templates for constrained setting datasets that provide consistent and structured textual descriptions. These templates contain placeholders for key attributes such as dance style, setting, and movement quality, which are populated based on the specific characteristics of each video. For in-the-wild videos, which lack standardized descriptions, we use VideoChat2 [32] for generating captions. VideoChat2 analyzes the visual content and generates detailed captions that capture the contextual information present in these diverse video samples.

In addition, we randomly replace a small portion of detailed captions with basic, simple captions. This allows the adapter network to learn how to respond to music without relying on the text, effectively reducing the model’s dependence on textual cues and encouraging it to develop stronger associations between musical features while still maintaining prompt adherence. This trade-off between style capture and music interpretation of the prompt is reflected in Table 4. The exact prompt templates used for diversification and replacement are provided in the supplementary material.

### 5. Experiments 5.1. Implementation Details

Model Details We train our model on a single NVIDIA A100 GPU (except for experiments requiring more capacity) for 4,000 steps with a learning rate of 1e−4, which takes roughly 20 hours to complete. Our LoRA uses rank 64, providing sufficient capacity to capture complex dance move-

Style Beat Body Movement Choreography Dance Quality Alignment Alignment Representation Realism Complexity Average

Model Modality

AIST Dataset (GT) [48] A+V 7.46 8.95 7.53 8.67 7.45 8.01 MM-Diffusion [39] A+V 7.16 8.56 5.52 7.05 7.53 7.16 Mochi [46] T+V 7.20 8.34 7.47 7.68 7.82 7.70 MusicInfuser (Ours) T+A+V 7.56 8.89 7.16 8.24 7.90 7.95

- Table 2. Dance quality metrics comparing different models. A, V, and T denote audio, video, and text input modalities, respectively. For the models that have text input modality, we report an average of scores using a predefined benchmark of prompts.

Model Modality

Imaging Aesthetic Overall Video Quality Quality Quality Consistency Average

AIST Dataset (GT) [48] A+V 9.76 8.17 9.77 9.23 MM-Diffusion [39] A+V 8.94 6.52 8.38 7.94 Mochi [46] T+V 9.46 7.90 8.98 8.78 MusicInfuser (Ours) T+A+V 9.60 7.87 9.39 8.95

- Table 3. Video quality metrics comparing different models. For the models that have text input modality, we report an average of scores using a predefined benchmark of prompts.

Style Creative Overall Prompt Align Capture Interpretation Satisfaction Average

Model

Mochi [46] 7.98 9.04 9.55 8.86 MusicInfuser (Ours) 7.80 9.27 9.80 8.96 No in-the-Wild Data 6.80 8.69 8.40 7.96 Base Prompt 0% 7.45 8.85 9.43 8.58 Base Prompt 100% 7.33 9.06 9.36 8.58

Table 4. Prompt alignment metrics comparing different models.

which should represent the upper bound for these metrics, and demonstrates its reliability alongside correlation with human evaluation results shown in Fig. 9.

ments while maintaining parameter efficiency. For BetaUniform scheduling, we set the initial β = 3 with exponential decay toward β = 1. We use Mochi [46] as our base model with a classifier-free guidance scale of γcfg = 6.0 during inference and employ Wav2Vec 2.0 [6] as the audio encoder, while using a shallow MLP followed by downsampling to match the temporal dimension of the audio tokens for the audio projector.

##### 5.2. Experimental Results

Music- and Text-Driven Dance Video Generation Fig. 1 showcases the model’s ability to combine textual control with musical synchronization. The generated videos successfully incorporate scene contexts (restaurant kitchen, beach at sunset) and dancer attributes (wearing a leather jacket, chef’s uniform) as specified in the prompts, while simultaneously aligning the choreographic style with the musical input. Figs. 3 and 4 demonstrates that our model is capable of generating unseen subjects or rare settings.

Dataset The AIST dataset [48] includes 13,940 videos with 60 musical pieces, 10 dance genres, and 35 dancers. We extract 2,378 clips and divide the training and test sets with non-overlapping music tracks, following AIST++ [33]. We randomly sample approximately 2.5-second clips from full sequences for training. As mentioned in Sec. 4.4, we supplement AIST with 15,799 in-the-wild dance video clips from 4 YouTube playlists containing over 3.7k videos across various dance styles and settings. These clips are mixed with the AIST data at a 1:1 ratio during training, creating a balanced dataset that combines AIST’s controlled studio environment with diverse real-world performances.

Music Responsiveness In Fig. 7, we show how MusicInfuser generates dance videos, including the movement and outfit of the dancer, based on the music condition, while keeping the prompt fixed. Additionally, we demonstrate the model’s responsiveness to musical features through experiments with tempo modification. By accelerating the music track by 1.25 times or decelerating it by 0.75 times, the generated dance movements appropriately adjust pace while maintaining similar choreographic style, as shown in Fig. 7. Furthermore, acceleration and deceleration also result in changes in tone, which affect the dynamicity of the dance generated by our model. This shows that our model successfully captures the relationship between musical tempo and dance movement dynamicity, a critical aspect of dance-music synchronization.

Quantitative Metrics Evaluating generated content automatically presents challenges. Inspired by VBench’s use of Visual-Language Models for text-to-video assessment [32], we propose a novel metric using VideoLLaMA 2 [14] and Qwen3-Omni [50], which process both video and audio inputs. We formulate targeted queries to assess three components: dance quality (style alignment, beat alignment, body representation, movement realism, choreography complexity), video quality (imaging quality, aesthetic quality, overall consistency), and prompt alignment (style capture, creative interpretation, satisfaction). See the supplementary material for exact prompts and methods. Tables 2 and 3 also display results on AIST test data, where the ground truth data outperforms generated content in metrics like beat alignment and movement realism. This validates that our metric correctly assigns higher scores to ground truth data,

Generalization to In-the-Wild Music and Longer Videos To evaluate generalization beyond the AIST music distribution, we test our model with music tracks generated by SUNO AI. Fig. 6 shows successful generation for these unseen music categories, confirming the model’s ability to map novel audio patterns to appropriate dance movements. In addition, Fig. 6 shows longer video generation results with the same setting but with multiple times as many

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

- Figure 8. By changing the seed, our method can produce diverse results given the same music and text. The generated choreography of each dance is different from each other. We use the fixed prompt “a professional dancer dancing ....”

frames as the videos we used for training, up to 9 seconds.

Baseline Comparison We present several baselines for our layer adaptation in Table 1. Adapting the layers with cross-attention that we selected based on the layer adaptability criterion in Sec. 4.1 significantly outperforms the strategy of selecting evenly distributed layers, first layers, middle layers, and last layers, and even outperforms adapting all layers in the video diffusion model. This demonstrates that our positive influence function for layer selection is crucial for high-performance adaptation.

Tables 2–3 present our quantitative results compared against prior work [39, 46]. For dance quality (Table 2), our method outperforms previous approaches in style alignment, beat alignment, movement realism, and choreography complexity, while maintaining competitive scores across other metrics. Table 3 demonstrates our superiority in video quality metrics, particularly in imaging quality and overall consistency compared to MM-Diffusion [39] and Mochi [46]. In Table 4, MusicInfuser shows improved creative interpretation and overall satisfaction over the baseline Mochi model. For qualitative comparisons with prior work, we refer readers to the supplementary material.

Human Evaluation We conduct human evaluation to validate MusicInfuser’s performance and examine the correlation between Video-LLM-based quantitative assessment (Tables 2 and 3) and human judgments. Fig. 9 presents the

|8.5%<br><br>17.9%<br><br>14.2%<br><br>18.7%<br><br>91.5%<br><br>82.1%<br><br>85.8%<br><br>83.3%<br><br>MM-Diffusion<br><br>MusicInfuser (Ours)| | | | |
|---|---|---|---|---|
| | | | | |

Scores(%)

Video Quality

Music-Dance Alignment

Motion Realism

Choreography Complexity

Figure 9. Human evaluation.

results of our human evaluation study, where we assess generated videos across multiple dimensions including video quality, music-dance alignment, motion realism, and choreography complexity. The human evaluation demonstrates that our approach consistently outperforms previous work, with evaluators particularly noting improvements in video quality and movement naturalness. The details of the human evaluation are provided in the supplementary material.

Ablation Studies In Table 1, we evaluate the contribution of the components in our framework. Using Beta-Uniform scheduling improves body representation and movement realism. The naive feature addition baseline, where instead of using the ZICA adapter we simply spatially expand the audio feature and add it to the corresponding frame, similar to ControlNet [54], performs worse than our approach in most metrics, confirming the effectiveness of our ZICA strategy. Not zero-initializing the cross-attention layers [51] results in a remarkable drop in the video quality metric. Additionally, in Table 4, we show the trade-off between style capture and creative interpretation of the prompt depending on the base prompt ratio, meaning how frequently we replaced the prompt with the basic prompt. More ablation studies and analysis are in the supplementary material.

Diversity of Results By varying the random seed while keeping the prompt and music constant, our model generates diverse choreographies, as shown in Fig. 8, demonstrating that it does not simply memorize routines for particular tracks but is capable of generating diverse dance sequences.

### 6. Conclusion

In this paper, we present MusicInfuser, a novel approach for generating dance videos synchronized with music by leveraging the rich choreographic knowledge embedded in pretrained text-to-video diffusion models. Through our adaptation architecture and strategies, MusicInfuser enables synchronized dance movements with musical inputs while preserving text-based control over style and scene elements. It achieves this without requiring expensive motion capture data, generalizes to novel music tracks and subjects, and supports the generation of diverse choreographies.

### Acknowledgments

We thank Xiaojuan Wang and Jingwei Ma for their valuable feedback. This work was supported by the UW Reality Lab and Google.

### References

- [1] Donghoon Ahn, Hyoungwon Cho, Jaewon Min, Wooseok Jang, Jungwoo Kim, SeonHwa Kim, Hyun Hee Park, Kyong Hwan Jin, and Seungryong Kim. Self-rectifying diffusion sampling with perturbed-attention guidance. In European Conference on Computer Vision, pages 1–17. Springer,

2024. 4

- [2] Omid Alemi, Jules Franc¸oise, and Philippe Pasquier. Groovenet: Real-time music-driven dance movement generation using artificial neural networks. networks, 8(17):26,

2017. 2

- [3] Simon Alexanderson, Rajmund Nagy, Jonas Beskow, and Gustav Eje Henter. Listen, denoise, action! audio-driven motion synthesis with diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–20, 2023. 2, 3
- [4] Andreas Aristidou, Anastasios Yiannakidis, Kfir Aberman, Daniel Cohen-Or, Ariel Shamir, and Yiorgos Chrysanthou. Rhythm is a dancer: Music-driven motion synthesis with global structure. IEEE transactions on visualization and computer graphics, 29(8):3519–3534, 2022. 3
- [5] Ho Yin Au, Jie Chen, Junkun Jiang, and Yike Guo. Choreograph: Music-conditioned automatic dance choreography over a style and tempo consistent dynamic graph. In Proceedings of the 30th ACM International Conference on Multimedia, pages 3917–3925, 2022. 3
- [6] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460, 2020. 7
- [7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [8] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575, 2023. 3
- [9] Caroline Chan, Shiry Ginosar, Tinghui Zhou, and Alexei A Efros. Everybody dance now. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5933–5942, 2019. 3
- [10] Moitreya Chatterjee and Anoop Cherian. Sound2sight: Generating visual dynamics from sound and context. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVII 16, pages 701–719. Springer, 2020. 3
- [11] Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin.

- Videojam: Joint appearance-motion representations for enhanced motion generation in video models. arXiv preprint arXiv:2502.02492, 2025. 18
- [12] Kang Chen, Zhipeng Tan, Jin Lei, Song-Hai Zhang, YuanChen Guo, Weidong Zhang, and Shi-Min Hu. Choreomaster: choreography-oriented music-driven dance synthesis. ACM Transactions on Graphics (TOG), 40(4):1–13, 2021. 2
- [13] Minghao Chen, Iro Laina, and Andrea Vedaldi. Training-free layout control with cross-attention guidance. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 5343–5353, 2024. 4
- [14] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 2, 6, 7, 13, 15, 17
- [15] Satoru Fukayama and Masataka Goto. Automated choreography synthesis using a gaussian process leveraging consumer-generated dance motions. In Proceedings of the 11th Conference on Advances in Computer Entertainment Technology, pages 1–6, 2014. 2
- [16] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and timesensitive transformer. In European Conference on Computer Vision, pages 102–118. Springer, 2022. 3
- [17] Kehong Gong, Dongze Lian, Heng Chang, Chuan Guo, Zihang Jiang, Xinxin Zuo, Michael Bi Mi, and Xinchao Wang. Tm2d: Bimodality driven 3d dance generation via music-text integration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9942–9952, 2023. 3
- [18] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 4
- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 3
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [21] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 3
- [22] Susung Hong. Smoothed energy guidance: Guiding diffusion models with reduced energy curvature of attention. Advances in Neural Information Processing Systems, 37: 66743–66772, 2025. 4
- [23] Susung Hong, Gyuseong Lee, Wooseok Jang, and Seungryong Kim. Improving sample quality of diffusion models using self-attention guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7462– 7471, 2023. 4
- [24] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 6

- [25] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 4, 16
- [26] Junha Hyung, Kinam Kim, Susung Hong, Min-Jung Kim, and Jaegul Choo. Spatiotemporal skip guidance for enhanced video diffusion sampling. arXiv preprint arXiv:2411.18664,

2024. 4, 16, 18

- [27] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 3
- [28] Tae-hoon Kim, Sang Il Park, and Sung Yong Shin. Rhythmic-motion synthesis based on motion-beat analysis. ACM Transactions on Graphics (TOG), 22(3):392–401,

2003. 2

- [29] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 1
- [30] Nhat Le, Tuong Do, Khoa Do, Hien Nguyen, Erman Tjiputra, Quang D Tran, and Anh Nguyen. Controllable group choreography using contrastive diffusion. ACM Transactions on Graphics (TOG), 42(6):1–14, 2023. 3
- [31] Hsin-Ying Lee, Xiaodong Yang, Ming-Yu Liu, Ting-Chun Wang, Yu-Ding Lu, Ming-Hsuan Yang, and Jan Kautz. Dancing to music. Advances in neural information processing systems, 32, 2019. 2
- [32] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 6, 7, 17
- [33] Ruilong Li, Shan Yang, David A Ross, and Angjoo Kanazawa. Ai choreographer: Music conditioned 3d dance generation with aist++. In Proceedings of the IEEE/CVF international conference on computer vision, pages 13401– 13412, 2021. 2, 6, 7, 13, 15, 17
- [34] Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, and Chao Liang. Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models. arXiv preprint arXiv:2502.01061, 2025. 19
- [35] Yimeng Liu and Misha Sra. Dancegen: Supporting choreography ideation and prototyping with generative ai. In Proceedings of the 2024 ACM Designing Interactive Systems Conference, pages 920–938, 2024. 3
- [36] Ferda Ofli, Engin Erzin, Y¨ucel Yemez, and A Murat Tekalp. Learn2dance: Learning statistical music-to-dance mappings for choreography synthesis. IEEE Transactions on Multimedia, 14(3):747–759, 2011. 2
- [37] Qiaosong Qi, Le Zhuo, Aixi Zhang, Yue Liao, Fei Fang, Si Liu, and Shuicheng Yan. Diffdance: Cascaded human motion diffusion model for dance generation. In Proceedings of the 31st ACM International Conference on Multimedia, pages 1374–1382, 2023. 3

- [38] Liangdong Qiu, Chengxing Yu, Yanran Li, Zhao Wang, Haibin Huang, Chongyang Ma, Di Zhang, Pengfei Wan, and Xiaoguang Han. Vimo: Generating motions from casual videos. arXiv preprint arXiv:2408.06614, 2024. 3
- [39] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. Mm-diffusion: Learning multi-modal diffusion models for joint audio and video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10219–10228, 2023. 2, 3, 7, 8, 12, 14, 17
- [40] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 4
- [41] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. Bailando: 3d dance generation by actor-critic gpt with choreographic memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11050– 11059, 2022. 3, 15
- [42] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. Bailando++: 3d dance gpt with choreographic memory. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(12): 14192–14207, 2023. 3
- [43] Li Siyao, Tianpei Gu, Zhitao Yang, Zhengyu Lin, Ziwei Liu, Henghui Ding, Lei Yang, and Chen Change Loy. Duolando: Follower gpt with off-policy reinforcement learning for dance accompaniment. ICLR, 2024. 3
- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. ICLR, 2021. 3
- [45] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. ICLR, 2021. 3
- [46] Genmo Team. Mochi 1. https://github.com/ genmoai/models, 2024. 1, 2, 7, 8, 12, 14, 16, 17
- [47] Jonathan Tseng, Rodrigo Castellon, and Karen Liu. Edge: Editable dance generation from music. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 448–458, 2023. 2, 3, 14, 17
- [48] Shuhei Tsuchida, Satoru Fukayama, Masahiro Hamasaki, and Masataka Goto. Aist dance video database: Multi-genre, multi-dancer, and multi-camera database for dance information processing. In ISMIR, page 6, 2019. 5, 6, 7, 17
- [49] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu,

- Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1
- [50] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025. 2, 6, 7, 15, 17
- [51] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 8

- [52] Zijie Ye, Haozhe Wu, Jia Jia, Yaohua Bu, Wei Chen, Fanbo Meng, and Yanfeng Wang. Choreonet: Towards music to dance synthesis with choreographic action unit. In Proceedings of the 28th ACM International Conference on Multimedia, pages 744–752, 2020. 2
- [53] Sihyun Yu, Kihyuk Sohn, Subin Kim, and Jinwoo Shin. Video probabilistic diffusion models in projected latent space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18456–18466,

2023. 3

- [54] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 6, 8
- [55] Mingao Zhang, Changhong Liu, Yong Chen, Zhenchun Lei, and Mingwen Wang. Music-to-dance generation with multiple conformer. In Proceedings of the 2022 International Conference on Multimedia Retrieval, pages 34–38, 2022. 2
- [56] Wenlin Zhuang, Congyi Wang, Jinxiang Chai, Yangang Wang, Ming Shao, and Siyu Xia. Music2dance: Dancenet for music-driven dance generation. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 18(2):1–21, 2022. 2

## MusicInfuser: Making Video Diffusion Listen and Dance Supplementary Material

[Figure 48]

[Figure 49]

Music

- Track 1

[Figure 50]

Music

- Track 2

Music

- Track 3

[Figure 51]

[Figure 52]

[Figure 53]

MM-Diﬀusion Ours

- Figure 10. Comparison of audio-driven generation with MM-Diffusion [39]. Our method produces fewer artifacts (shown in the first and third rows), while generating more realistic dance videos with more natural movements (first row) and more dynamic motion (second and third rows). Note that we use the same music track for each row, and the spectrogram is stretched for MM-Diffusion since we generate longer videos. For our method, we use the fixed caption “a professional dancer dancing ...” across all music tracks.

[Figure 54]

[Figure 55]

“a male dancer wearing streetwear dancing in an urban skatepark with graﬃ  walls, front view”

[Figure 56]

“a female dancer dancing in a subway sta on, captured from a front view”

[Figure 57]

Mochi

Ours

- Figure 11. MusicInfuser infuses listening capability into the text-to-video model (Mochi [46]), while preserving prompt adherence and improving overall consistency and realism.

### A. Video Results

We present the flattened video results along the time axis and the corresponding spectrograms in the main paper. However, our frame sampling rate does not exceed the Nyquist frequency for the general musical beat, causing the movement to appear slower. Therefore, we encourage readers to view the supplementary video.

[Figure 58]

[Figure 59]

“… clapping her hands …” “… moves to the le  … kicking her leg …”

[Figure 60]

[Figure 61]

“… contemporary dance, moving ﬂuidly across the ﬂoor …” “… a spin move, turning 360 degrees with arms raised …”

Figure 12. Text-based dance control.

|Layer Selection Strategy|Style Beat Body Movement Choreography Dance Quality<br><br>Alignment Alignment Representation Realism Complexity Average<br><br>|Imaging Aesthetic Overall Video Quality Quality Quality Consistency Average|Overall<br><br>|
|---|---|---|---|
|Layer Adaptability Evenly Distributed Layers First Layers Middle Layers Last Layers All Layers<br><br>|7.56 8.89 7.16 8.24 7.90 7.95 7.31 8.81 7.28 7.70 7.96 7.81 7.25 8.82 6.86 7.37 8.05 7.67 7.91 8.87 6.74 7.83 7.98 7.86 7.52 8.81 7.01 7.47 8.00 7.76 7.49 8.53 6.72 8.16 7.85 7.75|9.60 7.87 9.39 8.95 9.33 7.78 9.04 8.72 9.66 7.91 9.27 8.95 9.21 7.97 9.20 8.79 9.45 7.73 9.14 8.77 9.33 7.99 9.11 8.81<br><br>|8.33 8.15 8.15 8.21 8.14 8.15<br><br>|

Table 5. Evaluation of layer selection strategies using VideoLLaMA 2 [14].

|Model|Style Beat Body Movement Choreography Dance Quality<br><br>Alignment Alignment Representation Realism Complexity Average|Imaging Aesthetic Overall Video Quality<br><br>Quality Quality Consistency Average<br><br>|Overall|
|---|---|---|---|
|Full (Ours) No ZICA Layer Selection No Higher Rank No LoRA No Beta-Uniform Schedule Feature Addition<br><br>|7.56 8.89 7.16 8.24 7.90 7.95 7.31 8.81 7.28 7.70 7.96 7.81 7.37 8.76 6.86 7.75 7.98 7.74<br><br>7.48 8.62 7.02 7.53 7.95 7.72<br>8.04 9.07 6.35 7.88 7.91 7.85 7.62 8.90 6.78 7.97 7.88 7.83<br>|9.60 7.87 9.39 8.95 9.33 7.78 9.04 8.72 9.55 7.94 9.49 8.99<br><br>9.43 8.08 9.36 8.96<br><br>9.17 7.85 9.37 8.80<br><br>9.44 7.88 9.31 8.88<br><br><br>|8.33 8.15 8.21 8.18 8.21 8.22|

Table 6. Ablation study. Feature addition denotes that we spatially expand the audio feature and add it to the corresponding frame. We use VideoLLaMA 2 [14] for the evaluation.

### B. Text-Driven Choreography

To demonstrate text-based control of dance sequences, we conducted additional experiments in which text prompts guide the dance dynamics (clapping, kicking, spinning, etc.). The results are shown in Fig. 12. This demonstrates that MusicInfuser can also generate diverse music-synchronized videos that follow the motion directions specified in the text prompt.

### C. Dance Difficulty Control

We demonstrate difficulty control of the choreography in Fig. 13, which is achieved using the same seed and music but with prompts of varying specificity. For basic dance, we use the general prompt “a professional dancing in a studio with a white backdrop.” For styled dance, we additionally specify the dance genre but use “basic dance setting,” and for advanced, we change it to “advanced dance setting.”

### D. Human Evaluation Protocol

For each test music track [33], we conducted fully anonymized A/B testing. We asked 33 participants to evaluate video quality, music-dance alignment, motion realism, and choreography complexity. The following are examples of the questionnaire items:

- 1. Which video has higher visual quality?
- 2. Which video’s dance aligns better with the music?

[Figure 62]

Basic Dance

[Figure 63]

Styled Dance

[Figure 64]

Advanced Dance

Figure 13. Changes in the complexity of choreography.

- 3. Which video’s motion is more realistic?
- 4. Which video’s dance is more complex?

### E. Limitations

Although our method adds listening capability to text-to-video models and improves dance generation, some properties such as style capture from the prompt and imaging quality are bounded by the capabilities of the underlying models. Additionally, our method inherits some problems from text-to-video models. Sometimes, fine details such as fingers and faces fail to be generated properly, especially when our model synthesizes dance videos with fast movements. Furthermore, our model is easily misled by the silhouette of the dancers, meaning that under the same silhouette, it may merge or swap the positions of body parts, which is also a problem in the base model. We include some examples of failure cases in Fig. 19.

### F. Additional Qualitative Analysis

We show more music-and-text-to-video generation examples in Fig. 20. Fig. 10 presents a side-by-side comparison with MM-Diffusion [39]. Unlike MM-Diffusion, which generates shorter videos with limited style control, MusicInfuser produces longer sequences with both musical synchronization and prompt-based style control, while improving the overall consistency of the video and reducing artifacts. We show a comparison with Mochi [46] in Fig. 11. Note that Mochi is not able to hear the music. Compared to Mochi, MusicInfuser produces more consistent human forms, fewer visual artifacts, and more fluid, realistic movements. Our method adds music responsiveness while maintaining or improving video consistency. We also compare with EDGE [47], a state-of-the-art skeleton-based dance generation model. EDGE performs 3D skeleton generation requiring 3D pose reconstruction, whereas our method performs direct video synthesis. Fig. 16 shows a visual comparison under the same music track using our method and EDGE. This demonstrates that our approach captures nuances that skeletonbased methods cannot represent, such as hair dynamics, clothing motion, a flexible backbone, and hand articulation.

We present qualitative results of our ablation study in Fig. 14 and Fig. 15. Our full model successfully generates consistent body shapes that align with the music while preserving prior knowledge without introducing significant artifacts.

[Figure 65]

Full

[Figure 66]

No ZICA Layer Selec on

[Figure 67]

No BetaUniform

[Figure 68]

No Higher Rank

[Figure 69]

No LoRA

[Figure 70]

Feature Addi on

- Figure 14. Ablation study. The prompt is set to “a male dancer dancing in an art gallery with some paintings, captured from a front view”. The seed and music are set the same across all methods.

### G. Additional Quantitative Analysis

Similar to the layer selection baselines and ablation studies in the main paper using Qwen3-Omni [50], we show evaluation using VideoLLaMA 2 [14] in Tables 5 and 6. The full model achieves the highest score. Using a higher rank for LoRA contributes substantially to movement realism, while our Beta-Uniform scheduling improves body representation. The naive feature addition baseline, where instead of using the ZICA adapter we simply spatially expand the audio feature and add it to the corresponding frame, performs worse than our approach on most metrics, confirming the effectiveness of our ZICA strategy. In Table 7, we present comparisons between MM-Diffusion and our method, both trained on the identical AIST++ training dataset without in-the-wild data. This shows that our model trained on the AIST dataset alone already surpasses MM-Diffusion, while in-the-wild data further enhances generalization capability.

In Table 4 in the main paper, we show the trade-off between style capture and creative interpretation of the prompt depending on the base prompt ratio, meaning how frequently we replaced the prompt with the basic prompt.

Additionally, we present commonly used intrinsic metrics from related work, BeatAlign and kinetic diversity [33, 41],

[Figure 71]

Full

[Figure 72]

No ZICA Layer Selec on

[Figure 73]

No BetaUniform

[Figure 74]

No Higher Rank

[Figure 75]

No LoRA

[Figure 76]

Feature Addi on

- Figure 15. Ablation study. The prompt is set to “a male dancer wearing a suit dancing in the middle of a New York City, captured from a front view”. The seed and music are set the same across all methods.

measured after extracting 2D pose sequences from generated videos. Table 8 shows these metrics, demonstrating a comparable score to the AIST test set and superior scores compared to the baselines.

### H. Layer Adaptability

The imaging and aesthetic quality of the base model [46] is presented in Fig. 17. This is analyzed with STG [26], an inferencetime technique, and the score is calculated with VBench [25]. Based on the imaging quality, which is highly related to the structure and noisiness of the video samples, we select the top 16 out of 48 layers in terms of imaging quality.

#### I. Beta-Uniform Scheduling The visualization of the Beta-Uniform scheduling strategy is shown in Fig. 18.

[Figure 77]

Ours

[Figure 78]

EDGE

- Track 1
- Track 2

[Figure 79]

Ours

[Figure 80]

EDGE

Figure 16. Comparison with EDGE [47]. Note that the poses of our method and EDGE need not align.

Style Beat Body Movement Choreography Imaging Aesthetic Overall Alignment Alignment Representation Realism Complexity Quality Quality Consistency

Model

MM-Diffusion [39] 7.16 8.56 5.52 7.05 7.53 8.94 6.52 8.38 Ours (Only AIST) 7.83 9.10 6.89 8.58 7.96 9.55 8.02 9.75

Table 7. Comparisons between MM-Diffusion [39] and our method, both trained on the AIST++ training dataset.

Method BeatAlign↑ Distk↑

AIST Dataset (GT) 0.2448 9.027 MM-Diffusion [39] 0.1553 2.126 Mochi [46] 0.1976 8.886 MusicInfuser (Ours) 0.2432 9.849

Table 8. BeatAlign and kinetic diversity metrics based on 2D poses.

### J. Test Music Tracks

For evaluating our method, we use music tracks that are set aside from the training set [48], following AIST++ [33]. The full list of test music codes is provided in Table 9.

### K. Prompts

As mentioned in our main paper, we use a proper prompt format and base prompt for AIST [48]. The full list is shown in Table 10. Note that since we use VideoChat2 [32] to label YouTube videos, we have only the base prompt for that dataset. We also provide a predefined set of prompts in Table 11 that is used to generate samples for the evaluation, ultimately resulting in 10 × 10 = 100 videos per model configuration. The system prompts for VideoLLaMA 2 [14] and Qwen3-Omni [50] used for evaluation are provided in Table 12.

| |Aesthetic Quality<br><br>Imaging Quality| |
|---|---|---|
| | | |
| | | |
|1234567891011121314151617181920212223242<br><br>Lay<br><br>17. Layer adaptability graph from [26],<br><br>[Figure 81]<br><br>Figure 18. Beta di| |52627282930313233343536373839404142434445464748<br><br>er<br><br>showing imaging and aesthetic quality.<br><br>stributions.|

0.575

0.550

QualityScore

0.525

0.500

0.475

0.450

0.425

0.400

Figure 1 y.

|Test Music Code|Genre|
|---|---|
|mLH4|LA style Hip-hop|
|mKR2|Krump|
|mBR0<br><br>|Break|
|mLO2<br><br>|Lock|
|mJB5<br><br>|Ballet Jazz|
|mWA0<br><br>|Waack|
|mJS3|Street Jazz|
|mMH3|Middle Hip-hop|
|mHO5|House|
|mPO1<br><br>|Pop|

Table 9. List of test music codes with corresponding dance genres.

### L. Concurrent Work

Several concurrent approaches have emerged alongside our research that address related challenges. Notable among these is VideoJAM [11], which enhances motion generation by jointly denoising both the motion maps and the video, an approach

[Figure 82]

Figure 19. Failure cases. Our model inherits some issues from the base model, such as failing to generate fine details (e.g., fingers and faces) and being fooled by the silhouette of the dancers.

|Category|Dataset|Prompt Template|
|---|---|---|
|Prompt Format|AIST<br><br>|{dancers text} dancing {genre name} in a {situation name} setting in a studio with a white backdrop, captured from a {camera view}<br><br>|
|Prompt Format|AIST<br><br>|a {camera view} video of {dancers text} performing {genre name} choreography against a white background in a {situation name} scene<br><br>|
|Prompt Format<br><br>|AIST|{dancers text} executing {genre name} movements in a minimalist studio space in a {situation name} setting, shot from a {camera view}<br><br>|
|Prompt Format|AIST|a {genre name} dance performance by {dancers text} in a pristine white studio, {camera view}, {situation name}<br><br>|
|Base Prompt|AIST<br><br>|a professional dancer dancing in a studio with a white backdrop|
|Base Prompt<br><br>|YouTube<br><br>|a dance video|

Table 10. Dance prompt templates categorized by type and dataset, including parameterized formats and simple base prompts.

|Prompts|
|---|
|a male dancer dancing on a rooftop at sunset, captured from a front view|
|a female dancer dancing in a subway station, captured from a front view|
|a male dancer dancing in an art gallery with some paintings, captured from a front view|
|a female dancer wearing a leather jacket dancing in a studio with a white backdrop, captured from a front view|
|a male dancer wearing a hoodie dancing in a studio with a white backdrop, captured from a front view|
|a female dancer wearing a denim vest dancing in a studio with a white backdrop, captured from a front view|
|a female dancer wearing a Hawaiian dress dancing on Waikiki Beach at sunset with Diamond Head in the background, captured from a front view|
|a male dancer wearing a suit dancing in the middle of a New York City, captured from a front view|
|a male dancer wearing a chef’s uniform dancing in a busy restaurant kitchen with flames from the grill behind him, captured from a front view|
|a female dancer wearing a Renaissance gown dancing in a Venetian masquerade ball with ornate chandeliers overhead, captured from a front view|

Table 11. Collection of dance scene prompts with various subjects, attire, and settings.

that is orthogonal to ours. Another related line of research is OmniHuman-1 [34], which integrates audio and pose inputs into diffusion models. The application of OmniHuman-1 remains primarily confined to scenarios that do not require much creative movement, relies on a private model, and necessitates full fine-tuning procedures, which distinguishes it from our approach.

|Metric<br><br>|Prompt|
|---|---|
|Dance Quality| |
|Style Alignment|Rate the style alignment of the dance to music where: 0 means poor style alignment of the dance to music, 5 means moderate style alignment of the dance to music, and 10 means perfect style alignment of the dance to music. Output only the number.|
|Beat Alignment|Rate the beat alignment of the dance to music where: 0 means poor beat alignment of the dance to music, 5 means moderate beat alignment of the dance to music, and 10 means perfect beat alignment of the dance to music. Output only the number.|
|Body Representation|Rate the body representation of the dancer where: 0 means unrealistic/distorted proportions of the dancer, 5 means minor anatomical issues of the dancer, and 10 means anatomically perfect representation of the dancer. Output only the number.|
|Movement Realism<br><br>|Rate the movement realism of the dancer where: 0 means poor movement realism of the dancer, 5 means moderate movement realism of the dancer, and 10 means perfect movement realism of the dancer. Output only the number.|
|Choreography Complexity<br><br>|Rate the complexity of the choreography where: 0 means extremely basic choreography, 5 means intermediate choreography, and 10 means extremely complex/advanced choreography. Output only the number.|
|Video Quality| |
|Imaging Quality|Rate the imaging quality where: 0 means poor imaging quality, 5 means moderate imaging quality, and 10 means perfect imaging quality. Output only the number.|
|Aesthetic Quality<br><br>|Rate the aesthetic quality where: 0 means poor aesthetic quality, 5 means moderate aesthetic quality, and 10 means perfect aesthetic quality. Output only the number.|
|Overall Consistency|Rate the overall consistency where: 0 means poor consistency, 5 means moderate consistency, and 10 means perfect consistency. Output only the number.|
|Prompt Alignment| |
|Style Capture|How well does the dance video capture the specific style mentioned in the prompt: ’{prompt}’? Rate 0-10 where: 0 means completely missed the style, 5 means some elements of the style are present, and 10 means perfectly captures the style. Output only the number.|
|Creative Interpretation<br><br>|Based on the prompt ’{prompt}’, rate the creativity in interpreting the prompt 0-10 where: 0 means generic/standard interpretation, 5 means moderate creativity, and 10 means highly creative and unique interpretation. Output only the number.|
|Overall Prompt Satisfaction<br><br>|Rate the overall prompt satisfaction 0-10 where: 0 means the video fails to satisfy the prompt ’{prompt}’, 5 means it partially satisfies the prompt, and 10 means it fully satisfies all aspects of the prompt. Output only the number.|

###### Table 12. System prompts for evaluation

[Figure 83]

“a male dancer dancing on a roo op at sunset, captured from a front view”

[Figure 84]

“a female dancer dancing in a Korean palace garden, front view”

[Figure 85]

“a male dancer wearing a tuxedo dancing in an elegant ballroom with crystal chandeliers, front view”

[Figure 86]

“a male dancer wearing a cowboy ou it dancing in a Western saloon with wooden bar and swinging doors, front view”

Figure 20. More music-and-text-to-video generation results.

