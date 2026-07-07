## FLUX that Plays Music

# arXiv:2409.00587v2[cs.SD]20Dec2024

Zhengcong Fei Mingyuan Fan Changqian Yu Junshi Huang1 Kunlun Inc.

### Abstract

This paper explores a simple extension of diffusion-based rectified flow Transformers for text-to-music generation, termed as FluxMusic. Generally, along with design in advanced Flux1 model, we transfers it into a latent VAE space of mel-spectrum. It involves first applying a sequence of independent attention to the double textmusic stream, followed by a stacked single music stream for denoised patch prediction. We employ multiple pre-trained text encoders to sufficiently capture caption semantic information as well as inference flexibility. In between, coarse textual information, in conjunction with time step embeddings, is utilized in a modulation mechanism, while fine-grained textual details are concatenated with the music patch sequence as inputs. Through an in-depth study, we demonstrate that rectified flow training with an optimized architecture significantly outperforms established diffusion methods for the text-to-music task, as evidenced by various automatic metrics and human preference evaluations. Our experimental data, code, and model weights are made publicly available at: https:

//github.com/feizc/FluxMusic.

et al., 2024b). Among these, diffusion models (Song et al.,

- 2020), trained to reverse the process of data transformation from structured states to random noise (Sohl-Dickstein et al., 2015; Song and Ermon, 2020), have shown exceptional effectiveness in modeling high-dimensional perceptual data, including music (Ho et al., 2020; Huang et al., 2023c; Ho et al., 2022; Kong et al., 2020b; Rombach et al., 2022).

Given the iterative nature of diffusion process, coupled with the significant computational costs and extended sampling times during inference, there has been a growing body of research focused on developing more efficient training strategy and accelerating sampling schedule (Karras et al., 2023; Liu et al., 2022a; Lu et al., 2022a; Fei, 2019; Lu et al., 2022b; Kingma and Gao, 2024), such as distillation (Sauer et al., 2024; Song et al., 2023; Song and Dhariwal, 2023). A particularly effective approach involves defining a forward path from data to noise, which facilitates more efficient training (Ma et al., 2024a) as well as better generative performance. One effective method among them is the Rectified Flow (Liu

- et al., 2022a; Albergo and Vanden-Eijnden, 2022; Lipman et al., 2023), where data and noise are connected along a linear trajectory. It offers improved theoretical properties and has shown promising results in image generation(Ma et al., 2024a; Esser et al., 2024), however, its application in music creation remains largely unexplored.

In the design of model architectures, traditional diffusion models frequently employ U-Net (Ronneberger et al., 2015) as the foundational structure. However, the inherent inductive biases of convolutional neural networks inadequately captures the spatial correlations within signals (Esser et al., 2021) and are insensitive to scaling laws (Li et al., 2024b). Transformer-based diffusion models have effectively addressed these limitations (Peebles and Xie, 2023; Bao

- et al., 2023; Fei et al., 2024b;c) by treating images as sequences of concatenated patches and utilizing stacked transformer blocks for noise prediction. The incorporation of cross-attention for integrating textual information has established this approach as the standard for generating high-resolution images and videos from natural language descriptions, demonstrating impressive generalization capabilities (Chen et al., 2023; 2024a; Fei et al., 2024e; Esser
- et al., 2024; Fei et al., 2023b; Ma et al., 2024b;b; Yang et al., 2024). Notably, the recently open-sourced FLUX model,

### 1. Introduction

Music, as a form of artistic expression, holds profound cultural importance and resonates deeply with human experiences (Briot et al., 2017). The task of text-to-music generation, which involves converting textual descriptions of emotions, styles, instruments, and other musical elements into audio, offers innovative tools and new avenues for multimedia creation (Huang et al., 2023b). Recent advancements in generative models have led to significant progress in this area (Yang et al., 2017; Dong et al., 2018; Mittal et al., 2021). Traditionally, approaches to text-to-music generation have relied on either language models or diffusion models to represent quantized waveforms or spectral features (Agostinelli et al., 2023; Lam et al., 2024; Liu et al., 2024; Evans et al., 2024; Schneider et al., 2024; Fei et al., 2024a; 2023c; Chen

1https://github.com/black-forest-labs/flux

Caption

Noised Latent

Patching

CLAP-L T5 XXL

Linear

MLP Linear

Position Embedding

- DoubleStream Block 1

- DoubleStream Block 2

MLP

Sinusoidal Encoding

DoubleStream Block

Timestep

- SingleStream Block 1

...

- SingleStream Block 2

###### ...

SingleStream Block

Unpatching

Output

- Figure 1. Model architecture of FluxMusic. We use frozen CLAP-L and T5-XXL as text encoders for conditioned caption feature extraction. The coarse text information concatenated with timestep embedding y are used to modulation mechanism. The fine-grained text c concatenated with music sequence x are input to a stacked of double stream block and single steam blocks to predict nose in a latent VAE space.

with its well-designed structure, exhibits strong semantic understanding and produces high-quality images, positioning it as a promising framework for conditional generation tasks.

In this work, we explore the application of rectified flow Transformers within noise-predictive diffusion for text-tomusic generation, introducing FluxMusic as a unified and scalable generative framework in the latent VAE space of the mel-spectrogram, as illustrated in Figure 1. Building upon the text-to-image FLUX model, we present a transformerbased architecture that initially integrates learnable double streams attention for the concatenated music-text sequence, facilitating a bidirectional flow of information between the modalities. Subsequently, the text stream is dropped, leaving a stacked single music stream for noised patch prediction. We leverage multiple pre-trained text encoders for extracting conditioned caption features and inference flexibility. Coarse textual information from CLAP-L (Elizalde et al.,

- 2023), combined with time step embeddings, is employed in the modulation mechanism, while fine-grained textual details from T5-XXL (Raffel et al., 2020) are concatenated with music patch sequence as input. We train the model

with rectified flow formulation and investigate its scalability. Through a in-depth study, we compare our new formulation to existing diffusion formulations and demonstrate its benefits for training efficiency and performance enhancement.

The primary contributions of this work are as follows:

- • We introduce a Flux-like Transformer architecture for text-to-music generation, equipped with rectified flow training. To the best of our knowledge, this is the first study to apply rectified flow transformers to text-tomusic generation;
- • We perform a comprehensive system analysis, encompassing network design, rectified flow sampling, and parameter scaling, demonstrating the advantages of FluxMusic architecture in text-to-music generation;
- • Extensive experimental results demonstrate that FluxMusic achieves generative performance on par with other recent models with adequate training on both automatic metrics and human preference ratings. Finally, we make the results, code, and model weights publicly available to support further research.

### 2. Related Works

#### 2.1. Text-to-music Generation

Text-to-music generation seeks to produce music clips that correspond to descriptive or summarized text inputs. Prior approaches have primarily employed language models (LMs) or diffusion models (DMs) to generate quantized waveform representations or spectral features. For generating discrete representation of waveform, models such as MusicLM (Agostinelli et al., 2023), MusicGen (Copet et al., 2024), MeLoDy (Lam et al., 2024), and JEN-1 (Li et al., 2024c) utilize LMs and DMs on residual codebooks derived from quantization-based audio codecs (Zeghidour et al., 2021; D´efossez et al., 2022). Conversely, models like Moˆusai (Schneider et al., 2024), Noise2Music (Huang et al., 2023b), Riffusion (Forsgren and Martiros, 2022), AudioLDM 2 (Liu et al., 2024), MusicLDM (Chen et al., 2024b), and StableAudio (Evans et al., 2024) employ UNet-based diffusion techniques to model mel-spectrograms or latent representations obtained through pretrained VAEs, subsequently converting them into audio waveforms using pretrained vocoders (Kong et al., 2020a). Additionally, models such as Mustango (Melechovsky et al., 2023) and Music Controlnet (Wu et al., 2024) incorporate control signals or personalization (Plitsis et al., 2024; Fei et al., 2023a), including chords and beats, in a manner similar to ControlNet (Zhang et al., 2023). Our method along with this approach by modeling the mel-spectrogram within a latent VAE space.

#### 2.2. Diffusion Transformers

Transformer architecture (Vaswani et al., 2017) has achieved remarkable success in language models (Radford, 2018; Radford et al., 2019; Raffel et al., 2020) and has also demonstrated significant potential across various computer vision tasks, including image classification (DOSOVITSKIY,

- 2020; He et al., 2022; Touvron et al., 2021; Zhou et al., 2021; Yuan et al., 2021; Han et al., 2021), object detection (Liu et al., 2021; Wang et al., 2021; 2022; Carion et al., 2020), and semantic segmentation (Zheng et al., 2021; Xie et al.,
- 2021; Strudel et al., 2021), among others (Sun et al., 2020; Li et al., 2022b; Zhao et al., 2021; Liu et al., 2022b; He et al.,
- 2022; Li et al., 2022a). Building on this success, the diffusion Transformer (Peebles and Xie, 2023; Fei et al., 2024d) and its variants (Bao et al., 2023; Fei et al., 2024b) have replaced the convolutional-based U-Net backbone (Ronneberger et al., 2015) with Transformers, resulting in greater scalability and more straightforward parameter expansion compared to U-Net diffusion models. This scalability advantage has been particularly evident in domains such as video generation (Ma et al., 2024b), image generation (Chen

- et al., 2023), and speech generation (Liu et al., 2023). Notably, recent works such as Make-an-audio 2 (Huang et al.,

- 2023c;a) and StableAudio 2 (Evans et al., 2024) also explored the DiT architecture for audio and sound generation. In contrast, our work investigates the effectiveness of new multi-modal diffusion Transformer structure similar to Flux and optimized it with rectified flow.

### 3. Methodology

FluxMusic is a conceptually simple extension of FLUX, designed to facilitate text-to-music generation within a latent space. An overview of the model structure is illustrated in Figure 1. In the following, we begin with a review of rectified flow as applied to diffusion models, followed by a detailed examination of the architecture for each component. We also discuss considerations regarding model scaling and data quality.

#### 3.1. Rectified Flow Trajectories

In this work, we explore generative models that estabilish a mapping between samples x1 from a noise distribution p1 to samples x0 from a data distribution p0 through a framework of ordinary differential equation (ODE). The connection can be expressed as dyt = vΘ(yt,t)dt where the velocity v is parameterized by the weights Θ of a neural network. (Chen et al., 2018) proposed directly solving it using differentiable ODE solvers. However, it proves to be computationally intensive, particularly when applied to large neural network architectures that parameterize vΘ(yt,t).

A more effective strategy involves directly regressing a vec-

tor field ut that defines a probability trajectory between p0 and p1. To construct such a vector field ut, we consider a forward process that corresponds to a probability path pt transitioning from p0 to p1 = N(0,1). This can be represented as zt = atx0 + btϵ , where ϵ ∼ N(0,I). With the conditions a0 = 1,b0 = 0,a1 = 0 and b1 = 1, the marginals pt(zt) = Eϵ∼N(0,I)pt(zt|ϵ) align with both the data and noise distribution.

Referring to (Lipman et al., 2023; Esser et al., 2024), we can construct a marginal vector field ut that generates the marginal probability paths pt, using the conditional vector fields as follows:

pt(z|ϵ) pt(z)

], (1)

ut(z) = Eϵ∼N(0,I)[ut(z|ϵ)

The conditional flow matching objective can be then formulated as:

t(z|ϵ),p(ϵ)||vΘ(z,t) − ut(z|ϵ)||22 , (2)

L = Et,p

where the conditional vector fields ut(z|ϵ) provides a tractable and equivalent objective. Although there exists different variants of the above formalism, we focus on Rectified Flows (RF) (Liu et al., 2022a; Albergo and Vanden-Eijnden, 2022; Lipman et al., 2023), which define the forward process as straight paths between the data distribution and a standard normal distribution:

zt = (1 − t)x0 + tϵ, (3)

with the loss function L corresponds to wtRF = 1−t t. The network output directly parameterizes the velocity vΘ.

#### 3.2. Model Architecture

To enable text-conditioned music generation, our FluxMusic model integrate both textual and musical modalities. We leverage pre-trained models to derive appropriate representations and then describe the architecture of our Flux-based model in detail.

Music compression. To better represent music, following (Liu et al., 2024), each 10.24-second audio clip, sampled at 16kHz, is first converted into a 64 × 1024 melspectrogram, with 64 mel-bins, a hop length of 160, and a window length of 1024. This spectrogram is then compressed into a 16 × 128 latent representation, denoted as Xspec, using a Variational Autoencoder (VAE) pretrained on AudioLDM 22. This latent space representation serves as the basis for noise addition and model training. Finally, a pretrained Hifi-GAN (Kong et al., 2020a) is employed to reconstruct the waveform from the generated mel-spectrogram.

2https://huggingface.co/cvssp/audioldm2-music/tree/main

#Params #DoubleStream m #SingleStream n Hidden dim. d Head number h Gflops

Small 142.3M 8 16 512 16 194.5G Base 473.9M 12 24 768 16 654.4G Large 840.6M 12 24 1024 16 1162.6G Giant 2109.9M 16 32 1408 16 2928.0G

Table 1. Scaling law of FluxMusic model size. The model sizes and detailed hyperparameters settings for scaling experiments.

Rectified flow transformers for music generation. Our architecture builds upon the MMDiT (Esser et al., 2024) and Flux architecture. Specifically, we first construct an input sequence consisting of embedding of the text and noised music. The noised latent music representation Xspec ∈ Rh×w×c is flatten 2×2 patches to a sequence of length 12h ·

- 1

- 2w. After aligning the dimensionality of the patch encoding and the fine-grained text encoding c, we concatenate the two sequences.

We then forward with two type layers including double stream block and single stream blocks. In the double stream block, we employ two distinct sets of weights for the text and music modalities, effectively treating them as independent transformers that merge during the attention operation. This allows each modality to maintain its own space while still considering the other. In the single stream block, the text component is dropped, focusing solely on music sequence modeling with modulation. It is also found in AuraFlow3 that removing some of MMDiT layers to just be single DiT block were much more scalable and compute efficient way to train these models.

We incorporate embeddings of the timestep t and coarse text y into the modulation mechanism. Drawing from (Esser et al., 2024), we employ multiple text encoders to capture various levels of textual information, thereby enhancing overall model performance and increasing flexibility during inference. By applying individual dropout rates during training, our model allows the use of any subset of text encoders during inference. This flexibility extends to the ability to pre-store blank textual representations, bypassing the need for network computation during inference.

- 3.3. Discussion

Model at Scale. In summary, the hyper-parameters of proposed FluxMusic architecture include the following key elements: the number of double stream blocks m, number of single stream blocks n, hidden state dimension d, and attention head number h. Various configurations of FluxMusic are listed in Table 1, span a broad range of model sizes and computational requirements, from 142M to 2.1B parameters and from 194.5G to 2928.0G Flops. This range

3https://blog.fal.ai/auraflow/

provides a thorough examination of the model’s scalability. Additionally, the Gflop metric, evaluated for a 16×128 text-to-music generation with a patch size of p = 2, i.e., 10s music clips according to blank text, is calculated using the thop Python package.

Synthetic data incorporation. It is widely recognized that synthetically generated captions can greatly improve performance of generative model at scale, i.e., text-to-image generation (Fei et al., 2024e; Chen et al., 2023; Betker et al., 2023; Fei, 2021; Fei et al., 2022). We follow their design and incorporate enriched music captions produced by a fine-tuned large language model. Specifically, we use the LP-MusicCaps model (Doh et al., 2024; 2023), available on Huggingface4. To mitigate the potential risk of the text-tomusic model forgetting certain concepts not covered in the music captioner’s knowledge base, we maintain a balanced input by using a mixture of 20% original captions and 80% synthetic captions.

### 4. Experiments

#### 4.1. Experimental settings

Datasets. We employ several datasets, including AudioSet Music Subset (ASM) (Gemmeke et al., 2017), MagnaTagTune, Million Song Dataset (MSD) (Bertin-Mahieux et al., 2011), MagnaTagTune (MTT) (Law et al., 2009), Free Music Archive (FMA) (Defferrard et al., 2016), Music4All (Santana et al., 2020), and an additional private dataset. Each audio track was segmented into 10-second clips and uniformly sampled at 16 kHz to ensure consistency across the datasets. Detailed captions corresponding to these clips were sourced from Hugging Face datasets5. Additionally, we automatically labeled the remaining music data using LPMusicCaps models. This preprocessing resulted in a comprehensive training dataset encompassing a total of ∼22K hours of diverse music content.

To benchmark our MusicFlux model against prior work, we conducted evaluations using the widely recognized MusicCaps dataset (Agostinelli et al., 2023) and the Song-

- 4https://huggingface.co/seungheondoh/ttmr-pp
- 5https://huggingface.co/collections/seungheondoh/enriching-

music-descriptions-661e9342edcea210d61e981d

[Figure 1]

[Figure 2]

- Figure 2. The loss curve of different model structure with similar parameters. We can see that combine double and single stream block is much more scalable and compute efficient way for music generation model.

Describer-Dataset (Manco et al., 2023). The prior dataset comprises 5.5K clips of 10.24 seconds each, accompanied by high-quality music descriptions provided by ten professional musicians. The latter dataset contains 706 licensed high-quality music recordings.

Figure 3. The loss curve of different model parameters with same structure. We can see that increase model parameters consistently improve the generative performance.

FAD↓ IS↑ CLAP↑

DDIM 7.42 1.67 0.201 RF 5.89 2.43 0.312

Table 2. Effect of rectified flow training in text-to-music generation. We train small version of FluxMusic with different sampling schedule and results show the superiority of RF training with comparable computation burden.

Implementail details. We utilize the last hidden state of FLAN-T5-XXL as fine-grained textual information and the pooler output of CLAP-L as coarse textual features. Referring to (Liu et al., 2024), our training process involves 10-second music clips, randomly sampled from full tracks. The training configuration includes a batch size of 128, a gradient clipping threshold of 1.0, and a learning rate of 1e-4. During inference, we apply a rectified flow with 50 steps and use a guidance scale of 3.5. To ensure model stability and performance, we maintain a secondary copy of the model weights, updated every 100 training batches through an exponential moving average (EMA) with a decay rate of 0.99, following the approach outlined by Peebles and Xie (2023). For unconditional diffusion guidance, we independently set the outputs of each of the two text encoders to null with a probability of 10%.

Evaluation metrics. The generated results are assessed using several objective metrics, including the Fr´echet Audio Distance (FAD) (Kilgour et al., 2018), Kullback-Leibler Divergence (KL), Inception Score (IS). To ensure a standardized and consistent evaluation process, all metrics are calculated utilizing the audioldm eval library (Liu et al.,

- 2024).

#### 4.2. Model Analysis

To assess the effectiveness of the proposed approaches and compare different strategies, we isolate and test only the specific components under consideration while keeping all other parts frozen. Ablation studies are performed on a subset of the training set, specifically utilizing the ASM and FMA datasets. For evaluation purposes, we employ an outof-domain set comprising 1K samples randomly selected from the MTT dataset.

Advantage of model architecture. We examine the architectural design choices for the diffusion network within FluxMusic, focusing on two specific variants: (1) utilizing double stream blocks exclusively throughout the entire network, and (2) employing a combination of both double and single stream blocks. In particular, we use a small version of the model, with one variant comprising 15 double stream blocks, referred to as 15D 0S, and the other combining 8 double stream blocks with 16 single stream blocks, referred to as 8D 16S. These configurations result in parameter counts of approximately 145.5M and 142.3M, respectively. As depicted in Figure 2, the combined double and single modality stream block architecture not only accelerates the training process but also enhances generative

Details MusicCaps Song Describer Dataset Params Hours FAD ↓ KL ↓ IS ↑ CLAP ↑ FAD ↓ KL ↓ IS ↑ CLAP ↑

MusicLM 1290M 280k 4.00 - - - - - - MusicGen 1.5B 20k 3.80 1.22 - 0.31 5.38 1.01 1.92 0.18 Mousai 1042M 2.5k 7.50 1.59 - 0.23 - - - Jen-1 746M 5.0k 2.0 1.29 - 0.33 - - - AudioLDM 2 (Full) 712M 17.9k 3.13 1.20 - - - - - AudioLDM 2 (Music) 712M 10.8k 4.04 1.46 2.67 0.34 2.77 0.84 1.91 0.28 QA-MDT (U-Net) 1.0B 12.5k 2.03 1.51 2.41 0.33 1.01 0.83 1.92 0.30 QA-MDT (DiT) 675M 12.5k 1.65 1.31 2.80 0.35 1.04 0.83 1.94 0.32

FluxMusic 2.1B 22K 1.43 1.25 2.98 0.36 1.01 0.83 2.03 0.35

- Table 3. Evaluation results for text-to-music generation with diffusion-based models and language-based models. We can see that with compettive parameters and training data, FluxMusic achieve best results in most metrics, demonstrating the promising of structure.

Experts Beginners

Model OVL REL OVL REL Ground Truth 4.20 4.15 4.00 3.85 AudioLDM 2 2.55 2.45 3.12 3.74 MusicGen 3.13 3.34 3.06 3.70 FluxMusic 3.35 3.54 3.25 3.80

- Table 4. Evaluation results of text-to-music performances in human evaluation. We denoted for text relevance (REL) and overall quality (OVL), with higher scores indicating better performance.

to 1408 results in further performance gains. It is important to note that the model’s performance has not yet fully converged, as training was conducted for only 200K steps. Nonetheless, across all configurations, substantial improvements are observed at all training stages as the depth and width of the FluxMusic architecture are increased.

#### 4.3. Compared with Previous Methods

We conducted a comparative analysis of our proposed MusicFlux method against several prominent prior text-tomusic approaches, including AudioLDM 2 (Liu et al., 2024), Mousai (Schneider et al., 2024), Jen-1 (Li et al., 2024c), and QA-MDT (Li et al., 2024a), which model music using spectral latent spaces, as well as MusicLM (Agostinelli et al., 2023) and MusicGen (Copet et al., 2024), which employ discrete representations. All the results of these comparisons are summarized in Table 3.

performance, despite maintaining a comparable parameter scale. Consequently, we designate the mixed structure as the default configuration.

Effect of rectified flow. Table 2 presents a comparative analysis of various training strategies employed in FluxMusic, including DDIM and rectified flow, using the small model version. Both strategy training with 128 batch size and 200K training steps to maintain an identical computation cost. As anticipated, and in line with prior research (Esser et al., 2024), rectified flow training demonstrates a positive impact on generative performance within the music domain.

The experimental outcomes highlight the significant advantages of our FluxMusic models, which achieve stateof-the-art performance across multiple objective metrics. These findings underscore the scalability potential of the FluxMusic framework, particularly as model and dataset sizes consistently increase. Although FluxMusic exhibited a slight advantage in FAD and KL metrics on the SongDescriber-Dataset, this may be attributed to instabilities stemming from the dataset’s limited size. Further, our superiority in text-to-music generation was corroborated through additional subjective evaluations.

Effect of model parameter scale. We examine the scaling properties of the FluxMusic framework by analyzing the impact of model depth, defined by the number of double and single stream layers, and model width, characterized by the hidden size dimension. Specifically, we train four variants of FluxMusic using 10-second clips, with model configurations ranging from small to giant, as detailed in Table 1. As the loss cureve depicted in Figure 3, performance improves as the depth of double:single modality block increases from 8:16 to 16:32, and similarly, expanding the width from 512

#### 4.4. Human Evaluation

We laso conducted a human evaluation, following settings outlined in (Li et al., 2024a; Liu et al., 2024), to assess the performance of text-to-music generation. This evaluation focused on two key aspects of the generated audio samples: (i) overall quality (OVL) and (ii) relevance to the textual

An experimental and free-jazz psych-folk tune with jazz inﬂuences from Europe.

A lively and energizing rock anthem that gets audiences pumped up and singing along.

|[Figure 3]|
|---|
|[Figure 4]|
|[Figure 5]|
|[Figure 6]|

|[Figure 7]|
|---|
|[Figure 8]|
|[Figure 9]|
|[Figure 10]|

50K

100K

150K

200K

This song is a unique mix of experimental sounds and styles that push the boundaries of conventional music.

This song is a mashup of punk, progressive, rock, and no wave genres.

|[Figure 11]|
|---|
|[Figure 12]|
|[Figure 13]|
|[Figure 14]|

|[Figure 15]|
|---|
|[Figure 16]|
|[Figure 17]|
|[Figure 18]|

50K

100K

150K

200K

- Figure 4. Generated mel-spectrum cases of different training steps. We plot small version of MusicFlux at every 50K training steps and we can find that the image becomes orderly and fine-grained instead of random and disorderly with the training continues.

input (REL). For the overall quality assessment, human raters were asked to evaluate the perceptual quality of the samples on a scale from 1 to 5. Similarly, the text relevance test required raters to score the alignment between the audio and the corresponding text input, also on a 1 to 5 scale. Our evaluation team comprised individuals from diverse backgrounds, including professional music producers and novices with little to no prior knowledge in this domain. These groups are categorized as experts and beginners. Each randomly selected audio sample was evaluated by at least ten raters to ensure robust results.

As reflected in Table 4, our proposed FluxMusic method significantly enhances both the overall quality of the music and its alignment with the text input. These improvements can

be attributed to the RF training strategy and the advanced architecture of our model. Notably, the feedback from experts indicates substantial gains, highlighting the model’s potential utility for professionals in the audio industry.

#### 4.5. Visualization and Music Examples

For a more convenient understanding, we visualize some generated music clips towards different prompt, from different perspective. These visualizations encompass: (1) different training step, (2) model parameter at scale, (2) setting of classifier-free guidance (CFG) number, the results are presented in Figure 4, 5, and 6, respectively. For more cases and listen intuitively, we recommand to visit the project webpage.

A high-energy punk rock track with grungy garage undertones.

A hip hop and electronic infused tune with strong elements of dubstep.

|[Figure 19]|
|---|
|[Figure 20]|
|[Figure 21]|
|[Figure 22]|

|[Figure 23]|
|---|
|[Figure 24]|
|[Figure 25]|
|[Figure 26]|

small

base

large

giant

A folk song from a British artist with traditional string instruments such as the ﬁddle and guitar.

This song is a combination of pop and experimental pop.

|[Figure 27]|
|---|
|[Figure 28]|
|[Figure 29]|
|[Figure 30]|

|[Figure 31]|
|---|
|[Figure 32]|
|[Figure 33]|
|[Figure 34]|

small

base

large

giant

- Figure 5. Generated mel-spectrum cases of different model parameters. With model size increase, the resulting mel-spectrum becomes more content rich and rhythmically distinct.

### 5. Conclusion

In this paper, we explore an extension of the FLUX framework for text-to-music generation. Our model, FluxMusic, utilizes rectified flow transformers to predict mel-spectra iteratively within a latent VAE space. Experiments demonstrate the advanced performance comparable to existing benchmarks. Moreover, our study yields several notable findings: first, a simple rectified flow transformer performs effectively for audio spectrograms. Then, we identify the optimal strategy and learning approach through an ablation study. Future research will investigate scalability using a mixture-of-experts architecture and distillation techniques to enhance inference efficiency.

### References

Andrea Agostinelli, Timo I Denk, Zal´an Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, et al. Musiclm: Generating music from text. arXiv preprint arXiv:2301.11325, 2023.

Michael S. Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants, 2022.

Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22669–22679, 2023.

|[Figure 35]|[Figure 36]|
|---|---|
|[Figure 37]|[Figure 38]|

|[Figure 39]|[Figure 40]|
|---|---|
|[Figure 41]|[Figure 42]|

|[Figure 43]|[Figure 44]|
|---|---|
|[Figure 45]|[Figure 46]|

|[Figure 47]|[Figure 48]|
|---|---|
|[Figure 49]|[Figure 50]|

cfg=1.0 cfg=3.5 cfg=5.5 cfg=7.5

- Figure 6. Generated mel-spectrum cases of different classifier-free guidance. We plot four clips with diverse textual prompts from small version of FluxMusic model and concatenate them in one figure. It can be seen that increasing the CFG number results in a more pronounced contrast in the generated mel-spectrum. To consistent with previous works, we set CFG=3.5 by default.

Thierry Bertin-Mahieux, Daniel PW Ellis, Brian Whitman, and Paul Lamere. The million song dataset. arxiv, 2011.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Jean-Pierre Briot, Ga¨etan Hadjeres, and Franc¸ois-David Pachet. Deep learning techniques for music generation–a survey. arXiv preprint arXiv:1709.01620, 2017.

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV, 2020.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-a: Fast training of diffusion transformer for photorealistic text-to-image synthesis, 2023.

Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-\sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024a.

Ke Chen, Yusong Wu, Haohe Liu, Marianna Nezhurina, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Musicldm: Enhancing novelty in text-to-music generation using beatsynchronous mixup strategies. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1206–1210. IEEE, 2024b.

Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David Kristjanson Duvenaud. Neural ordinary differential equations. In Neural Information Processing Systems,

2018. URL https://api.semanticscholar. org/CorpusID:49310446.

Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, and Alexandre D´efossez. Simple and controllable music generation. Advances in Neural Information Processing Systems, 36, 2024.

Micha¨el Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson. Fma: A dataset for music analysis. arXiv preprint arXiv:1612.01840, 2016.

Alexandre D´efossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. High fidelity neural audio compression. arXiv preprint arXiv:2210.13438, 2022.

SeungHeon Doh, Keunwoo Choi, Jongpil Lee, and Juhan Nam. Lp-musiccaps: Llm-based pseudo music captioning. arXiv preprint arXiv:2307.16372, 2023.

SeungHeon Doh, Minhee Lee, Dasaem Jeong, and Juhan Nam. Enriching music descriptions with a finetunedllm and metadata for text-to-music retrieval. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 826–830. IEEE, 2024.

Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, and YiHsuan Yang. Musegan: Multi-track sequential generative adversarial networks for symbolic music generation and accompaniment. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018.

Alexey DOSOVITSKIY. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang. Clap learning audio concepts from natural language supervision. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Zach Evans, Julian D Parker, CJ Carr, Zack Zukowski, Josiah Taylor, and Jordi Pons. Stable audio open. arXiv preprint arXiv:2407.14358, 2024.

Zheng-cong Fei. Fast image caption generation with position alignment. arXiv preprint arXiv:1912.06365, 2019.

Zhengcong Fei. Partially non-autoregressive image captioning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1309–1316, 2021.

Zhengcong Fei, Xu Yan, Shuhui Wang, and Qi Tian. Deecap: Dynamic early exiting for efficient image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12216–12226, 2022.

Zhengcong Fei, Mingyuan Fan, and Junshi Huang. Gradientfree textual inversion. In Proceedings of the 31st ACM International Conference on Multimedia, pages 1364– 1373, 2023a.

Zhengcong Fei, Mingyuan Fan, and Junshi Huang. A-jepa: Joint-embedding predictive architecture can listen. arXiv preprint arXiv:2311.15830, 2023b.

Zhengcong Fei, Mingyuan Fan, Li Zhu, Junshi Huang, Xiaoming Wei, and Xiaolin Wei. Masked auto-encoders meet generative adversarial networks and beyond. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24449–24459, 2023c.

Zhengcong Fei, Mingyuan Fan, and Junshi Huang. Music consistency models. arXiv preprint arXiv:2404.13358, 2024a.

Zhengcong Fei, Mingyuan Fan, Changqian Yu, and Junshi Huang. Scalable diffusion models with state space backbone. arXiv preprint arXiv:2402.05608, 2024b.

Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, and Junshi Huang. Diffusion-rwkv: Scaling rwkvlike architectures for diffusion models. arXiv preprint arXiv:2404.04478, 2024c.

Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, and Junshi Huang. Scaling diffusion transformers to 16 billion parameters. arXiv preprint arXiv:2407.11633, 2024d.

Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Youqiang Zhang, and Junshi Huang. Dimba: Transformer-mamba diffusion models. arXiv preprint arXiv:2406.01159, 2024e.

Seth Forsgren and Hayk Martiros. Riffusion-stable diffusion for real-time music generation. URL https://riffusion. com, 2022.

Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 776–780. IEEE, 2017.

Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, and Yunhe Wang. Transformer in transformer. NIPS, 2021.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.

Jiawei Huang, Yi Ren, Rongjie Huang, Dongchao Yang, Zhenhui Ye, Chen Zhang, Jinglin Liu, Xiang Yin, Zejun Ma, and Zhou Zhao. Make-an-audio 2: Temporalenhanced text-to-audio generation. arXiv preprint arXiv:2305.18474, 2023a.

Qingqing Huang, Daniel S Park, Tao Wang, Timo I Denk, Andy Ly, Nanxin Chen, Zhengdong Zhang, Zhishuai Zhang, Jiahui Yu, Christian Frank, et al. Noise2music: Text-conditioned music generation with diffusion models. arXiv preprint arXiv:2302.03917, 2023b.

Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao. Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models. In International Conference on Machine Learning, pages 13916–13932. PMLR, 2023c.

Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. arXiv preprint arXiv:2312.02696, 2023.

Kevin Kilgour, Mauricio Zuluaga, Dominik Roblek, and Matthew Sharifi. Fr\’echet audio distance: A metric for evaluating music enhancement algorithms. arXiv preprint arXiv:1812.08466, 2018.

Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation. Advances in Neural Information Processing Systems, 36, 2024.

Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. Advances in neural information processing systems, 33:17022–17033, 2020a.

Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761, 2020b.

Max WY Lam, Qiao Tian, Tang Li, Zongyu Yin, Siyuan Feng, Ming Tu, Yuliang Ji, Rui Xia, Mingbo Ma, Xuchen Song, et al. Efficient neural music generation. Advances in Neural Information Processing Systems, 36, 2024.

Edith Law, Kris West, Michael I Mandel, Mert Bay, and J Stephen Downie. Evaluation of algorithms using games: The case of music tagging. In ISMIR, pages 387–392. Citeseer, 2009.

Chang Li, Ruoyu Wang, Lijuan Liu, Jun Du, Yixuan Sun, Zilu Guo, Zhenrong Zhang, and Yuan Jiang. Qualityaware masked diffusion transformer for enhanced music generation. arXiv preprint arXiv:2405.15863, 2024a.

Hao Li, Yang Zou, Ying Wang, Orchid Majumder, Yusheng Xie, R Manmatha, Ashwin Swaminathan, Zhuowen Tu, Stefano Ermon, and Stefano Soatto. On the scalability of diffusion-based text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9400–9409, 2024b.

Peike Patrick Li, Boyu Chen, Yao Yao, Yikai Wang, Allen Wang, and Alex Wang. Jen-1: Text-guided universal music generation with omnidirectional diffusion models. In 2024 IEEE Conference on Artificial Intelligence (CAI), pages 762–769. IEEE, 2024c.

Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai. Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. In ECCV, 2022a.

Zhiqi Li, Wenhai Wang, Enze Xie, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, Ping Luo, and Tong Lu. Panoptic segformer: Delving deeper into panoptic segmentation with transformers. In CVPR, 2022b.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=PqvMRDCJT9t.

Haohe Liu, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong, Qiao Tian, Yuping Wang, Wenwu Wang, Yuxuan Wang, and Mark D Plumbley. Audioldm 2: Learning holistic audio generation with self-supervised pretraining. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.

Huadai Liu, Rongjie Huang, Xuan Lin, Wenqiang Xu, Maozong Zheng, Hong Chen, Jinzheng He, and Zhou Zhao. Vit-tts: visual text-to-speech with scalable diffusion transformer. arXiv preprint arXiv:2305.12708, 2023.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow, 2022a.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021.

Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In CVPR, 2022b.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35: 5775–5787, 2022a.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022b.

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740, 2024a.

Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024b.

Ilaria Manco, Benno Weck, Seungheon Doh, Minz Won, Yixiao Zhang, Dmitry Bodganov, Yusong Wu, Ke Chen, Philip Tovstogan, Emmanouil Benetos, et al. The song describer dataset: a corpus of audio captions for music-andlanguage evaluation. arXiv preprint arXiv:2311.10057, 2023.

Jan Melechovsky, Zixun Guo, Deepanway Ghosal, Navonil Majumder, Dorien Herremans, and Soujanya Poria. Mustango: Toward controllable text-to-music generation. arXiv preprint arXiv:2311.08355, 2023.

Gautam Mittal, Jesse Engel, Curtis Hawthorne, and Ian Simon. Symbolic music generation with diffusion models. arXiv preprint arXiv:2103.16091, 2021.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023. doi: 10.1109/iccv51070.2023.00387. URL http://dx.

doi.org/10.1109/ICCV51070.2023.00387.

Manos Plitsis, Theodoros Kouzelis, Georgios Paraskevopoulos, Vassilis Katsouros, and Yannis Panagakis. Investigating personalization methods in text to music generation. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1081–1085. IEEE, 2024.

A Radford. Improving language understanding by generative pre-training. arxiv, 2018.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015.

Igor Andr´e Pegoraro Santana, Fabio Pinhelli, Juliano Donini, Leonardo Catharin, Rafael Biazus Mangolin, Val´eria Delisandra Feltrim, Marcos Aur´elio Domingues, et al. Music4all: A new music database and its applications. In 2020 International Conference on Systems, Signals and Image Processing (IWSSIP), pages 399–404. IEEE, 2020.

Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. arXiv preprint arXiv:2403.12015, 2024.

Flavio Schneider, Ojasv Kamal, Zhijing Jin, and Bernhard Sch¨olkopf. Moˆusai: Efficient text-to-music diffusion models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8050–8068, 2024.

Jascha Narain Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. ArXiv, abs/1503.03585, 2015. URL https://api.

semanticscholar.org/CorpusID:14888175.

Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189, 2023.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution, 2020.

Yang Song, Jascha Narain Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. ArXiv, abs/2011.13456, 2020. URL https://api.semanticscholar.

org/CorpusID:227209335.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.

Robin Strudel, Ricardo Garcia, Ivan Laptev, and Cordelia Schmid. Segmenter: Transformer for semantic segmentation. In ICCV, 2021.

Peize Sun, Jinkun Cao, Yi Jiang, Rufeng Zhang, Enze Xie, Zehuan Yuan, Changhu Wang, and Ping Luo. Transtrack: Multiple object tracking with transformer. In arxiv, 2020.

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herv´e J´egou. Training data-efficient image transformers & distillation through attention. In ICML, 2021.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2017.

Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In ICCV, 2021.

Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pvt v2: Improved baselines with pyramid vision transformer. Computational Visual Media, 2022.

Shih-Lun Wu, Chris Donahue, Shinji Watanabe, and Nicholas J Bryan. Music controlnet: Multiple timevarying controls for music generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32: 2692–2703, 2024.

Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in Neural Information Processing Systems, 34:12077–12090, 2021.

Li-Chia Yang, Szu-Yu Chou, and Yi-Hsuan Yang. Midinet: A convolutional generative adversarial network for symbolic-domain music generation. arXiv preprint arXiv:1703.10847, 2017.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-tovideo diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zi-Hang Jiang, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens-to-token vit: Training vision transformers from scratch on imagenet. In ICCV, 2021.

Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. Soundstream: An end-to-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30:495–507, 2021.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.

Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In ICCV, 2021.

Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng,

Tao Xiang, Philip HS Torr, et al. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. In CVPR, 2021.

Daquan Zhou, Bingyi Kang, Xiaojie Jin, Linjie Yang, Xiaochen Lian, Zihang Jiang, Qibin Hou, and Jiashi Feng. Deepvit: Towards deeper vision transformer. In arxiv, 2021.

