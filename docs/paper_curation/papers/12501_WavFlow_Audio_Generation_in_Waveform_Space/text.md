# arXiv:2605.18749v1[cs.SD]18May2026

## WavFlow: Audio Generation in Waveform Space

Feiyan Zhou1,2, Luyuan Wang1, Shoufa Chen1,∗, Zhe Wang1, Zhiheng Liu1, Yuren Cong1, Xiaohui Zhang1, Fanny Yang1, Belinda Zeng1

1Meta AI, 2Northeastern University

∗Corresponding author

Modern audio generation predominantly relies on latent-space compression, introducing additional complexity and potential information loss. In this work, we challenge this paradigm with WavFlow, a framework that generates high-fidelity audio directly in raw waveform space without intermediate representations. To overcome the inherent difficulties of modeling high-dimensional and low-energy signals, we reshape audio into 2D token grids through waveform patchify and introduce amplitude lifting to align signal scales, enabling stable optimization via direct x-prediction in flow matching. To capture complex semantic alignment and temporal synchronization, we leverage an automated data pipeline to curate 5M high-quality video-text-audio triplets, allowing the model to learn fine-grained acoustic patterns from scratch. Experimental results show that WavFlow achieves competitive results on the video-to-audio benchmark VGGSound (FDPaSST 59.98, ISPANNs 17.40, DeSync 0.44) and the text-to-audio benchmark AudioCaps (FDPANNs 10.63, ISPANNs 12.62), matching or exceeding the performance of established latent-based methods. Our work demonstrates that such intermediate compression is not a prerequisite for high-quality synthesis, offering a simpler and more scalable alternative for multimodal audio generation.

Date: May 19, 2026 Project page: https://facebookresearch.github.io/WavFlow/ Code: https://github.com/facebookresearch/WavFlow

1 Introduction

Video-to-audio synthesis, often referred to as Foley-style generation (Cheng et al., 2025; Zhang et al., 2026; Shan et al., 2025; Wang et al., 2025a), aims to produce environmental and event-based soundscapes temporally and semantically aligned with the visual content. Recent state-of-the-art methods (Polyak et al., 2024; Luo et al., 2023; Zhang et al., 2026; Wang et al., 2024b; Cheng et al., 2025; Shan et al., 2025; Liu et al., 2025a,b; Wang et al., 2025a; Dai et al., 2026; Tian et al., 2025) have made rapid progress by adopting a common latent-space recipe: raw signals are first mapped into a compressed representation by a pretrained tokenizer or VAE (Défossez et al., 2022; Zeghidour et al., 2021; Kumar et al., 2023; Evans et al., 2024; Kong et al., 2020a; Lee et al., 2022), then a multimodal diffusion or flow-matching transformer (Ho et al., 2020; Rombach et al., 2022; Lipman et al., 2022; Esser et al., 2024; Peebles and Xie, 2023) learns their conditional distribution given visual and text features. Finally, a decoder reconstructs the waveform from these generated latents, as shown in Figure 1 (top). This paradigm has become the dominant framework for modern audio generation tasks.

While effective, this approach leaves a foundational question open: is latent-space compression truly necessary for audio generation? Relying on a separate, pretrained tokenizer not only increases pipeline complexity but also constrains the final synthesis quality to the reconstruction fidelity. This motivates us to investigate direct raw-waveform generation as a way to achieve strict temporal and semantic alignment while bypassing the intermediate compression layer.

Doing so, however, is non-trivial, as raw audio differs from latent representations in three fundamental ways. First, raw waveforms are extremely high-dimensional, leading to long sequences that are computationally challenging to model directly. Second, waveform amplitudes exhibit a high dynamic range while heavily concentrating near zero, yielding a poor signal-to-noise ratio during training that makes the flow-matching objective difficult to optimize in raw space. Third, paired video-audio datasets remain relatively scarce. Even

[Figure 1]

[Figure 2]

Video Text

Previous: Latent Space

Visual Encoder Text Encoder

Audio Encoder

Audio Decoder

latent

latent

[Figure 3]

[Figure 4]

N * blocks

WavFlow: Raw Space

Generated Audio

Audio

Transformer Blocks

MMDiT

Waveform Patches

Waveform Patches

- Figure 1 Standard Latent-Space vs. WavFlow. WavFlow eliminates the encoding-decoding bottleneck by processing waveform patches directly in the raw space.

the widely-used VGGSound (Chen et al., 2020a) contains only ∼200K samples (500 hours), a scale insufficient for models that operate directly on raw waveforms, which must learn complex acoustic structures, temporal dynamics, and precise cross-modal alignments end-to-end without the inductive bias provided by encoded audio priors.

In this work, we introduce WavFlow, a generative framework that performs Foley-style audio synthesis directly in raw waveform space. The architecture is deliberately simple: as illustrated in Figure 1 (bottom), we employ waveform patchify to reshape high-dimensional 1D waveforms into 2D token grids, and adopt x-prediction (Li and He, 2025) under conditional flow matching as a more stable training target for raw signals. To bridge the signal intensity mismatch between raw waveforms and the unit-variance Gaussian prior, we incorporate RMS normalization and amplitude scaling, lifting the signal into a range conducive to generative modeling. Finally, to address the data scarcity in raw-space learning, we develop an automated curation pipeline to filter a large-scale media data for audio quality and event diversity, yielding approximately 5M high-quality video-text-audio pairs (Polyak et al., 2024). We train WavFlow on this curated dataset to achieve robust video-conditioned generation and extend the model to text-only audio generation by simply zeroing out the visual conditions.

We evaluate WavFlow on the standard VT2A (VGGSound) and T2A (AudioCaps (Kim et al., 2019)) benchmarks. On VGGSound, WavFlow achieves state-of-the-art FDPaSST (55.82 at 44.1kHz and 59.98 at 16kHz) while demonstrating competitive performance in DeSync (0.44) and ISPANNs (17.40) compared to latent-based models (Wang et al., 2024b,a; Shan et al., 2025; Cheng et al., 2025). These results validate that raw-waveform synthesis can match or even exceed the precision and fidelity of latent-space paradigms. Furthermore, on AudioCaps, our model attains the best FDPANNs (10.63) and ISPANNs (12.62) reported to date, rivaling dedicated T2A systems.

In summary, our contributions to direct raw-space audio generation are three-fold:

- • (i) Streamlined Framework: we introduce WavFlow, a simplified architecture that synthesizes high-fidelity audio directly in the waveform space through waveform patchify, x-prediction flow matching, and specialized signal preprocessing, effectively eliminating the need for audio tokenizers.
- • (ii) Large-scale Data Curation: we identify that direct waveform modeling is exceptionally sensitive to data quality and scale, and thus develop an automated pipeline to harvest high-quality, large-scale supervision consisting of multi-modal VT2A samples.
- • (iii) Empirical Validation: we achieve highly competitive results on the VGGSound (VT2A) and AudioCaps (T2A) benchmarks, demonstrating that end-to-end waveform generation reaches performance on par with established latent-based methods in acoustic richness, fidelity, and synchronization.

### 2 Related Work

- 2.1 Latent-Space Audio Generation

The landscape of latent-space audio generation is characterized by two main paradigms: continuous latent modeling and discrete codec-based synthesis. Models such as AudioLDM (Liu et al., 2023, 2024), TANGO (Ghosal

- et al., 2023), and MMAudio (Cheng et al., 2025) operate on continuous manifolds learned by audio VAEs (Evans
- et al., 2024). These frameworks prioritize spectral reconstruction and often incorporate adversarial discriminators from vocoders like HiFi-GAN (Kong et al., 2020a) or BigVGAN (Lee et al., 2022) to refine the decoded waveforms. Conversely, systems like AudioGen (Kreuk et al., 2022) and V-AURA (Viertola et al., 2025) leverage discrete neural audio codecs (Défossez et al., 2022; Kumar et al., 2023), where generative modeling is performed over quantized tokens.

While this paradigm bypasses the high-dimensionality of raw audio, it imposes a rigid performance ceiling: the output quality is strictly upper-bounded by the reconstruction fidelity of the pretrained backbone. Critical details, such as high-frequency transients and fine-grained phase information, are often compromised during latent bottlenecking and remain irrecoverable through post-processing. This inherent lossiness motivates the exploration of modeling the audio distribution directly in its native, uncompressed space.

2.2 Raw-Space Generative Modeling

Before the dominance of latent-space paradigms, raw waveform modeling was explored through autoregressive and diffusion-based approaches such as WaveNet (Van Den Oord et al., 2016), WaveRNN (Kalchbrenner et al., 2018), WaveGrad (Chen et al., 2020b), and DiffWave (Kong et al., 2020c). These methods prove high-fidelity synthesis is feasible without intermediate compression, yet they primarily function as neural vocoders reconstructing waveforms from local spectral features. Consequently, the lack of a mechanism to map global semantic cues directly to raw waveforms limits their use in large-scale multimodal generation.

In the image domain, while early CNNs relied on specialized noise schedules (Chen, 2023; Hoogeboom et al.,

- 2023), Transformers often suffer from catastrophic degradation in high-dimensional raw space (Li and He,

- 2025). To mitigate this, frameworks like SiD2 (Hoogeboom et al., 2025), PixelFlow (Chen et al., 2025), and PixNerd (Wang et al., 2025b) resort to hierarchical designs or specialized heads. Most recently, JiT (Li and He, 2025) succeeds by revisiting the manifold hypothesis (Chapelle et al., 2006; Vincent et al., 2010): since clean data lies on a low-dimensional manifold while noise or velocity spans the entire high-dimensional space, x-prediction is fundamentally easier to learn than noise or v-prediction. This allows the network to focus on recovering the low-dimensional data structure rather than modeling full-space noise. These advances in vision suggest that the raw-space paradigm, if properly adapted, can overcome the scalability issues previously encountered in audio modeling.

- 2.3 Multimodal DiT for Audio Generation

Video-to-audio (VT2A) generation requires precise temporal synchronization and semantic consistency, leading recent systems to adopt Multimodal Diffusion Transformers (MMDiT) (Esser et al., 2024) for joint modeling of audio, video, and text. The evolution of these architectures reflects a progression from efficient latent-space synthesis in Frieren (Wang et al., 2024b) to the unified joint-attention paradigm of MMAudio (Cheng et al., 2025), which significantly improved cross-modal alignment. More recently, industrial-scale models (Shan et al., 2025; Wang et al., 2025a) have pushed performance limits by scaling architectures to dozens of layers and training on massive datasets, such as the 100k hours of video-text-audio samples used in HunyuanVideo-Foley, while utilizing universal latent codecs and enhanced visual modules for high-fidelity synthesis.

Despite these advancements, existing systems remain confined to the compressed latent space. Our work overcomes this by adopting an MMDiT-based architecture that eliminates the latent stage entirely, enabling high-fidelity synthesis directly on raw waveforms.

### 3 Method

The architecture of WavFlow is built on a MultiModal Diffusion Transformer (MMDiT) (Esser et al., 2024; Cheng et al., 2025) backbone. Given a multimodal conditioning signal c (video and text), the model employs conditional flow matching to generate the raw waveform x ∈ RT directly in observation space. To manage the challenges of high-dimensional audio, we apply waveform patchify to reshape the signal for transformer processing and adopt an x-prediction strategy to ensure stable training.

- 3.1 Flow Matching in Waveform Space

Conditional Flow Matching. We formulate waveform generation using conditional flow matching (Lipman et al., 2022; Liu et al., 2022; Albergo and Vanden-Eijnden, 2023). Let x0 ∼ N(0,I) denote Gaussian noise and x1 denote a clean waveform. A continuous interpolation between noise and data is defined as:

xt = (1 − t)x0 + tx1, t ∈ [0,1], (1)

with the corresponding target velocity v∗(xt,t) = x1 − x0. The goal is to learn a velocity field vθ(xt,t,c) that transports noise to data along this path. While latent-space methods model these flows in a compressed representation, we perform this mapping directly in the waveform space, solving the ODE dx

t

dt = vθ(xt,t,c) during inference to obtain the final waveform.

Prediction Parameterization and Loss. We adopt x-prediction (Li and He, 2025; Salimans and Ho, 2022), where network predicts the clean signal:

xˆ1 = fθ(xt,t,c). (2)

The velocity is recovered as vθ = (ˆx1 − xt)/(1 − t). Our default configuration optimizes this x-prediction through a v-loss:

L = Ex

0,x1,t

x ˆ1 − xt 1 − t −

x1 − xt 1 − t

2

2

. (3)

This combination ensures that while the network focuses on recovering the data manifold (Chapelle et al., 2006), the objective remains anchored to the flow-matching velocity field. We validate this design choice through ablation experiments in Section 4.4.

- 3.2 Model Architecture

Audio Preprocessing. Raw waveforms typically exhibit a sharp, zero-centered distribution with low energy (average RMS often below 0.2), making them easily masked by noise during training. To mitigate this, we apply amplitude lifting by combining RMS normalization and global scaling. Specifically, after converting audio x ∈ RT to mono, the lifted waveform xlift is computed as:

r⋆ rms(x)

xlift = sa · clamp

x,−1,1 , (4)

where we empirically set r⋆ = 0.33 and sa = 3.0 to align the signal scale with the Gaussian noise prior. During inference, the output is rescaled by 1/sa and normalized to −23LUFS (European Broadcasting Union, 2020) to ensure perceptually comfortable playback. A visualization of this shift is provided in Appendix B.

Waveform Patchify. After preprocessing, raw audio is reshaped into a C × D grid via waveform patchify (Figure 2), where each row serves as a token analogous to the image patchify in ViT (Dosovitskiy et al., 2020). The patch dimension D represents the samples per token, defining its temporal granularity. This involves a fundamental trade-off: while smaller D eases the learning of intricate acoustic details, it increases computational complexity (O(C2)); conversely, larger D improves efficiency but increases per-token information density. Ablation studies (Section 4.4) reveal that increasing the data scale effectively compensates for this modeling difficulty, allowing the network to extract sufficient information even from wider patches.

[Figure 5]

- Figure 2 The WavFlow Architecture. Raw audio is represented as a 2D patch grid and processed through a series of joint and fused transformer blocks. The model leverages multimodal conditioning (cg and ce) for precise semantic and temporal control during the flow-matching process.

Our investigations identify D=200 as the saturation point where performance stabilizes. At 16kHz, this yields C=640 tokens for an 8s clip, resulting in a 12.5ms granularity—well below the ∼25 ms human auditory resolution threshold (Petrini et al., 2009). To maintain architectural consistency, we keep D=200 for 44.1kHz signals (C=1,764), resulting in an even finer temporal granularity that further reinforces the model’s highfidelity representation. After generation, the grid is reshaped back to a 1D waveform (waveform unpatchify). This process is entirely parameter-free and lossless, requiring no learned decoders or neural vocoders.

Multimodal DiT Architecture. As shown in Figure 2, we adopt the Multimodal Diffusion Transformer (MMDiT) (Esser et al., 2024; Cheng et al., 2025) as our backbone, consisting of Ljoint joint blocks for multimodal fusion followed by Lfused audio-only blocks for waveform refinement. Three input streams enter the joint attention sequence: audio waveform tokens, visual features from a frozen CLIP (Radford et al., 2021) encoder, and text embeddings from a CLIP text encoder. Audio waveform tokens and visual CLIP features are projected into a shared hidden dimension d via convolutional input blocks, whereas text features use a linear projection.

The model employs dual-level conditioning to capture semantic (“what”) and temporal (“when”) cues. A global condition cg is formed by summing mean-pooled visual and text features with the flow-matching timestep embedding, providing semantic guidance. To capture precise temporal cues, a frozen Synchformer (Iashin et al., 2024) extracts synchronization features from video, augmented with learnable per-segment positional embeddings. A frame-aligned condition ce ∈ RC×d is obtained by adding cg to these synchronization features (upsampled to length C via nearest interpolation), providing frame-level alignment. These conditions are injected into the transformer blocks through AdaLN modulation (Peebles and Xie, 2023), ensuring robust audio-visual correlation and semantic grounding in the raw waveform space.

Following the transformer blocks, a final output block projects the features from d back to D samples per token via AdaLN and a 1D convolution (kernel size 7). The resulting grid is then reconstructed into a 1D waveform via waveform unpatchify. We instantiate two variants based on scale: WavFlow-M (Ljoint=4,Lfused=8,∼624M parameters) and WavFlow-L (Ljoint=7,Lfused=14,∼1.03B parameters), both sharing hidden dimension d=896 and 14 attention heads.

Positional Encoding. We apply RoPE (Su et al., 2024) on the queries and keys to inject relative position information into joint attention; text tokens are excluded since captions encode unordered semantics rather

than temporal structure. Because the audio and visual CLIP streams run at different frame rates, applying identical base frequencies would map equivalent moments in the two streams to mismatched rotary angles. We therefore multiply the visual stream’s RoPE base frequency by the audio-to-visual ratio C/Nclip (e.g., 10× for C=640,Nclip=64), so that tokens at the same relative temporal position receive matching rotary phases.

- 3.3 Training and Inference

Classifier-Free Guidance. During training, we independently replace the visual conditioning (visual CLIP and Synchformer features jointly) and text features with learned null embeddings with a 10% probability. This strategy not only enables classifier-free guidance during inference but also allows WavFlow to support both video-to-audio (VT2A) and text-to-audio (T2A) tasks within a single model. For T2A generation, we simply zero out the visual pathways using the learned null embeddings, reducing the conditioning signal to text alone without any architectural modification.

Inference. Generation begins with Gaussian noise sampled in the waveform token space. We solve the learned ODE using an Euler solver with classifier-free guidance (CFG):

vˆθ = (1 + w)vθ(xt,t,c) − wvθ(xt,t,∅), (5)

where w is the guidance scale and ∅ denotes the null conditions. After the integration, the generated token grid is converted back to a 1D raw waveform via waveform unpatchify, requiring no additional learned decoder.

- 4 Experiments

- 4.1 Dataset

Training generative models directly in raw waveform space imposes significant demands on data scale and quality. In latent-space methods, a pretrained audio encoder leverages extensive audio data (e.g., ∼20K hours in (Evans et al., 2024)) to encode rich acoustic priors, effectively mapping complex waveforms into a compressed manifold that simplifies generative learning. By contrast, waveform-space models must learn intricate acoustic patterns and cross-modal dependencies from scratch, necessitating access to large-scale, high-fidelity audio-visual datasets.

Consequently, we curate a large-scale proprietary media dataset via an automated unified pipeline, constructing robust training sets for both tasks.

Data Curation Pipeline. As illustrated in Figure 3, our pipeline unifies VT2A and T2A samples through three stages. For open-source data, we utilize VGGSound alongside AudioCaps (Kim et al., 2019) and Freesound (Fonseca et al., 2017). Initially, we apply multi-stage filtering across all sources: extracting 8s segments and discarding samples with > 80% silence, low aesthetic scores (PQ < 6.0 via audioboxaesthetics (Tjandra et al., 2025)), or low classification confidence (bottom 10% via PANNs (Kong et al., 2020b)). This process yields roughly 50M filtered media clips, 100K VGGSound samples, and 150K highquality T2A samples.

Subsequently, we balance and augment the filtered data. The curated media clips are category-aligned with VGGSound to form a balanced pool of 5M samples. For the smaller VGGSound and public T2A sets, we apply temporal augmentation by extracting two overlapping 8s chunks starting at 0s and 1s, respectively, to double their size to 200K and 300K. Ultimately, these sources are merged into our final mixtures: a VT2A set combining the 5M balanced media pool with augmented VGGSound, and a T2A set mixing the 300K augmented public T2A samples with 1M clips randomly sampled from the same high-quality media corpus. This ensures a consistent data distribution across both tasks.

- 4.2 Experimental Setup

Training. Training configurations are detailed in Table 6. Models are trained on 8s clips using flow-matching with x-prediction and v-loss, sampling timesteps from a logit-normal distribution (Esser et al., 2024). Raw

###### Stage I: Filtering Stage II: Balancing & Augmentation Stage III: Training Mixture

[Figure 6]

A&F Filtered (150k) A&F Augmented (300k)

AudioCaps & Freesound (T2A)

Discard

A&F +1s (150k)

VGG +1s (100k)

VGG Filtered (100k)

VGGSound (VT2A)

T2A Cleaned Training Dataset(~1m)

Discard MD Sampled 1m

VGG Augmented (200k)

Filtering Criteria

- (1) Silence Filter (<8s or >80% silent)
- (2) Quality Filter(PQ<6.0)
- (3) PANNs Confidence Filter (bottom 10% class)

Media Data (VT2A)

MD Balanced 5m

(Align to VGGSound Distribution)

MD Filtered (50m)

VT2A Cleaned Training Dataset(~5m)

Discard (Imbalanced Data)

- Figure 3 Overview of multi-stage data curation from raw datasets to final high quality, balanced training mixtures.

audio is tokenized via waveform patchify (D=200), resulting in sequence lengths of 640 tokens for 16kHz and 1,764 tokens for 44.1kHz. We instantiate two 16kHz variants (WavFlow-M-16kHz, WavFlow-L-16kHz) and one 44.1kHz model (WavFlow-L-44.1kHz).

The 16kHz models are trained from scratch for 400epochs, a convergence point identified by monitoring validation metrics (see Figure 5). Our primary VT2A model utilizes the ∼5M mixture with a global batch size of 10,752, while the T2A model uses the ∼1M mixture with a batch size of 8,192. For high-fidelity synthesis, the 44.1kHz Large model is fine-tuned (SFT) (Yosinski et al., 2014) from the converged 16kHz checkpoint. All stages use the AdamW optimizer with a constant learning rate of 1 × 10−4 (1 × 10−5 for SFT), a 20-epoch linear warmup, and an EMA decay of 0.9999.

Evaluation Metrics. We evaluate VT2A on the VGGSound test set (15K videos) and T2A on AudioCaps (4.8K samples). We run inference with an ODE solver using 50steps and a CFG scale of 4.5 (see Appendix F). We assess performance across three dimensions:

- • (i) Acoustic Quality: We measure Fréchet Distance (FD), KL divergence, and Inception Score (IS) using features from PANNs (Kong et al., 2020b) and PaSST (Koutini et al., 2021) classifiers.
- • (ii) Semantic Alignment: The semantic alignment is evaluated via ImageBind (IB) (Girdhar et al., 2023) for audio-visual correspondence and CLAP (Wu et al., 2023) for text-audio consistency.
- • (iii) Temporal Synchronization: Timing accuracy between visual events and audio onsets is quantified using the DeSync metric (Iashin et al., 2024).

- 4.3 Comparison with State-of-the-arts

Video-to-AudioGeneration. Table 1 summarizes the results on VGGSound-Test set. Even with a medium-sized backbone at 16kHz, WavFlow surpasses established latent-based systems—including Frieren (Wang et al.,

- 2024b), V2A-Mapper (Wang et al., 2024a), and HunyuanVideo-Foley (Shan et al., 2025)—across multiple acoustic and synchronization metrics (e.g., FDPANNs: 6.37, ISPANNs: 17.24, and DeSync: 0.47), closely

approaching the state-of-the-art MMAudio-L-44.1kHz (Cheng et al., 2025) baseline despite the absence of a pretrained neural codec.

Scaling to WavFlow-L-16kHz yields consistent improvements, surpassing MMAudio-L-44.1kHz in distributional fidelity (FDPaSST: 59.98 vs. 60.60) while matching its performance in perceptual and alignment metrics (ISPANNs: 17.40, DeSync: 0.44). This validates that raw-waveform modeling can attain the same level of temporal precision and acoustic quality as sophisticated latent-space methods. Furthermore, WavFlow-L44.1kHz, fine-tuned from the 16 kHz Large checkpoint, pushes distributional fidelity even further, achieving the best-reported FDPaSST (55.82) across all compared methods while maintaining high synchronization (DeSync: 0.46). Collectively, these results confirm that with high-quality, large-scale data, end-to-end raw waveform synthesis can eliminate the need for intermediate latent representations without sacrificing performance.

- Table 1 Comparison with state-of-the-art methods (Wang et al., 2024b,a; Shan et al., 2025; Cheng et al., 2025) on VGGSound-Test for video-text-to-audio (VT2A) generation. Lower is better for FD, KL and DeSync; higher is better for IS, IB and CLAP. The best and second-best results are highlighted in bold and underlined, respectively.

Method FDPANNs ↓ FDPaSST ↓ KLPANNs ↓ ISPANNs ↑ IB↑ DeSync↓ CLAP↑ Params Frieren† 11.45 106.10 2.73 12.25 0.23 0.85 0.11 159M V2A-Mapper† 8.40 84.57 2.69 12.47 0.23 1.23 0.11 229M HunyuanVideo-Foley∗ 10.53 97.85 2.02 14.99 0.32 0.54 0.23 – MMAudio-L-44.1kHz† 4.72 60.60 1.65 17.40 0.33 0.44 0.22 1.03B WavFlow-M-16kHz 6.37 62.64 1.68 17.24 0.30 0.47 0.21 624M WavFlow-L-16kHz 5.86 59.98 1.66 17.40 0.31 0.44 0.22 1.03B WavFlow-L-44.1kHz 5.25 55.82 1.73 15.05 0.31 0.46 0.19 1.03B

All methods are evaluated on the same VGGSound test split from the MMAudio (Cheng et al., 2025) benchmark, utilizing original videos and native class labels as captions to ensure a fair comparison. Due to the difference in semantic granularity (sparse labels vs. dense captions), we exclude direct comparisons with models relying on LLM-refined captions. †: results taken from the MMAudio paper. ∗: reproduced by using their open-source checkpoints on the same test set.

- Table 2 Comparison with state-of-the-art methods (Liu et al., 2024; Ghosal et al., 2023; Majumder et al., 2024; Huang et al., 2023b,a; Haji-Ali et al., 2026; Cheng et al., 2025) on AudioCaps-Test for text-to-audio (T2A) generation.

Method Params FDPANNs ↓ FDVGG ↓ ISPANNs ↑ CLAP↑

AudioLDM 2-L 712M 32.50 5.11 8.54 0.21 TANGO 866M 26.13 1.87 8.23 0.19 TANGO 2 866M 19.77 2.74 8.45 0.26 Make-An-Audio 453M 27.93 2.59 7.44 0.21 Make-An-Audio 2 937M 15.34 1.27 9.58 0.25 GenAU-Large 1.25B 16.51 1.21 11.75 0.29 MMAudio-L-44.1kHz 1.03B 15.04 4.03 12.08 0.35

WavFlow-M-16kHz 624M 10.63 1.58 12.62 0.24

Baseline results are quoted from (Cheng et al., 2025) for a fair comparison, as we adopt the same test splits.

Text-to-Audio Generation. To further validate the versatility of our VAE-free approach, we evaluate WavFlowM-16kHz on the AudioCaps text-to-audio benchmark. As shown in Table 2, despite being a unified model rather than one specialized for T2A, our system achieves competitive acoustic quality across all compared methods. Specifically, it attains the lowest FDPANNs (10.63) and the highest ISPANNs (12.62), outperforming dedicated latent-space models and the previous state-of-the-art MMAudio. These results demonstrate that the intricate acoustic patterns learned directly in raw waveform space generalize effectively across different input modalities. More importantly, this cross-task success reinforces our primary conclusion: high-fidelity synthesis can be achieved without intermediate latent representations or task-specific vocoders.

- 4.4 Ablation Studies

Unless specified, all ablations are conducted using WavFlow-M-16k trained on a mixture of 1M media data and VGGSound, following the default configurations in Appendix A. All reported results are evaluated on the VGGSound-Test set.

Patchify Granularity Analysis. Figure 4 and Table 3 reveal an interplay between the patch dimension D and data scale. Reducing D refines the temporal resolution of raw waveform tokens by increasing the token count C. In the low-data regime (200K VGGSound), this finer granularity is the most effective lever for quality: shrinking D from 512 to 200 drives a dramatic improvement in FDPaSST (136.45 → 90.24) and DeSync (0.66 → 0.59). This suggests that high-resolution tokenization is essential for capturing intricate acoustic structures when training samples are sparse.

Conversely, increasing data scale can partially compensate for the modeling difficulty of a larger D. For a fixed D=512, expanding the dataset from 200K to 1M significantly improves FDPaSST (136.45 → 81.81). However, at 3M samples, this coarse configuration suffers from a capacity bottleneck, where performance degrades to 89.51, indicating that large D values eventually fail to represent the complex information inherent in larger datasets. In contrast, with sufficient granularity (D ≤ 256), increasing data scale consistently improves results. Notably, at 3M samples, the performance gain from further reducing D from 200 to 160 is marginal (60.75 vs. 59.43), identifying D=200 as the saturation point where computational efficiency and synthesis quality are optimally balanced. Based on these findings, we adopt D=200 (C=640) as our default configuration.

###### FD-PaSST

###### FD-PANNs

###### IB

###### DeSync

140

0.30

Dataset Scale

Dataset Scale

Dataset Scale

- 6

- 7

- 8

- 9

- 10

- 11

- 12

0.650

200K

200K

200K

130

0.29

1M 3M

1M 3M

1M 3M

0.625

120

0.28

0.600

110

FD-PANNs

FD-PaSST

0.27

DeSync

0.575

100

IB

0.26

90

0.550

0.25

80

0.525

Dataset Scale

200K

0.24

70

0.500

1M 3M

60

0.23

0.475

160 200 256 512

160 200 256 512

160 200 256 512

160 200 256 512

Patch Dimension D

Patch Dimension D

Patch Dimension D

Patch Dimension D

- Figure 4 Patchify granularity vs. data scale across key metrics. Note the performance degradation at D=512 when scaling to 3M, indicating a capacity bottleneck for low-granularity tokens.

- Table 3 Impact of patchify granularity (C × D) on waveform synthesis across varying data scales. Results demonstrate the trade-off between temporal resolution and data diversity.

Data Scale C × D FDPaSST ↓ FDPANNs ↓ KLPANNs ↓ ISPANNs ↑ IB↑ DeSync↓

250 × 512 136.45 12.06 1.88 10.86 0.23 0.66 500 × 256 96.91 9.11 1.84 12.56 0.25 0.61 640 × 200 90.24 8.63 1.83 13.45 0.25 0.59

200K (VGGSound)

250 × 512 81.81 6.56 1.78 13.66 0.27 0.55 500 × 256 73.16 7.54 1.91 16.18 0.28 0.51 640 × 200 63.05 6.21 1.76 15.58 0.28 0.50

1M

250 × 512 89.51 8.58 1.96 15.60 0.27 0.50 500 × 256 62.69 5.65 1.68 15.80 0.30 0.49 640 × 200 60.75 6.08 1.74 16.56 0.30 0.48 800 × 160 59.43 6.04 1.73 16.43 0.30 0.49

3M

Prediction Target and Loss Formulation. Table 4 compares prediction targets (x vs. v) and loss formulations. The results show that x-prediction consistently outperforms v-prediction across all metrics, likely because it provides a more structured objective that better conforms to the manifold hypothesis (Chapelle et al., 2006; Li and He, 2025). It is worth noting that both targets remain viable for direct raw-waveform generation within our framework. This universal effectiveness can be attributed to the low per-token dimensionality (D=200) relative to the model’s hidden dimension (d=896), which provides sufficient capacity to represent either target without a latent bottleneck.

Among the x-prediction variants, x-pred with v-loss achieves the best FDPaSST (63.05) and ISPANNs (15.58), whereas x-pred with x-loss shows slight advantages in FDPANNs (4.86). Given that v-loss superiorly balances generative diversity (ISPANNs) and high-frequency fidelity—as reflected by FDPaSST, which leverages the 32kHz-aware PaSST to capture a wider frequency range than PANNs (operates on 16kHz) —we adopt x-prediction with v-loss as our default.

- Table 4 Ablation on WavFlow’s Flow Matching objectives. Results indicate that x-prediction with v-loss provides the optimal balance between high-level feature similarity (FDPaSST) and generative diversity (ISPANNs).

Setting FDPaSST ↓ FDPANNs ↓ KLPANNs ↓ ISPANNs ↑ IB↑ DeSync↓

v-pred + v-loss 77.19 6.38 1.75 13.48 0.27 0.53 x-pred + x-loss 72.70 4.86 1.72 13.99 0.29 0.50 x-pred + v-loss 63.05 6.21 1.76 15.58 0.28 0.50

Raw-waveform Preprocessing. Table 5 validates our preprocessing pipeline, showing that both RMS normalization and amplitude scaling are essential for signal quality. Without scaling (1.0×), omitting RMS normalization severely degrades FDPaSST (65.83 → 81.26) and DeSync (0.49 → 0.57). At 3.0× scale, the enhanced signal strength naturally narrows the performance gap, yet the combination of both techniques still yields the best overall results in diversity and fidelity (e.g., ISPANNs of 15.58). We hypothesize that proper scaling helps the raw signal better align with the Gaussian prior during flow matching, whereas insufficient scaling (1.0×) leads to suboptimal optimization. Consequently, we adopt RMS normalization with a 3.0× scale as our default.

- Table 5 Ablation of raw-waveform preprocessing. Results indicate that RMS normalization with 3.0× scaling yields optimal performance.

Category Setting FDPaSST↓ FDPANNs↓ KLPANNs ↓ ISPANNs ↑ IB↑ DeSync↓ RMS Norm. (at 1.0× scale)

w/ 65.83 6.03 1.73 13.32 0.28 0.49 w/o 81.26 8.69 1.91 11.64 0.24 0.57

w 63.05 6.21 1.76 15.58 0.28 0.50 w/o 64.23 6.93 1.79 13.84 0.26 0.52

RMS Norm. (at 3.0× scale)

- 5 Conclusion and Limitations

We present WavFlow, a flow-matching framework for high-fidelity audio generation directly in raw-waveform space. By combining waveform patchify, x-prediction, and specific signal scaling with extensive data, WavFlow effectively stabilizes training and models complex raw signals. It achieves highly competitive performance on VT2A and T2A benchmarks, matching or exceeding established latent-based systems.

Limitations. WavFlow currently lacks explicit speech or singing synthesis, as generated vocalizations do not constitute meaningful language. Extending to these domains requires finer linguistic granularity and larger speech datasets. By incorporating larger-scale corpora and fine-grained linguistic captions, this framework could be extended to jointly model environmental sounds and human speech, offering a robust and efficient alternative for future generative research.

References

Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The

Eleventh International Conference on Learning Representations, 2023. https://arxiv.org/abs/2209.15571. Olivier Chapelle, Bernhard Schölkopf, and Alexander Zien. Semi-Supervised Learning. MIT Press, 2006. Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audio-visual dataset. In

International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2020a. Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, and William Chan. WaveGrad: Estimating gradients for waveform generation, 2020b. Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, and Ping Luo. Pixelflow: Pixel-space generative models with

flow. arXiv preprint arXiv:2504.07963, 2025. Ting Chen. On the importance of noise scheduling for diffusion models. arXiv preprint arXiv:2301.10972, 2023. Ho Kei Cheng, Masato Ishii, Akio Hayakawa, Takashi Shibuya, Alexander Schwing, and Yuki Mitsufuji. Mmaudio:

Taming multimodal joint training for high-quality video-to-audio synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28901–28911, 2025.

Yusheng Dai, Zehua Chen, Yuxuan Jiang, Baolong Gao, Qiuhong Ke, Jun Zhu, and Jianfei Cai. Omni2sound: Towards unified video-text-to-audio generation, 2026. https://arxiv.org/abs/2601.02731.

Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. High fidelity neural audio compression. arXiv preprint arXiv:2210.13438, 2022.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

European Broadcasting Union. EBU R 128: Loudness normalisation and permitted maximum level of audio signals. Technical report, European Broadcasting Union, 2020. https://tech.ebu.ch/docs/r/r128.pdf.

Zach Evans, CJ Carr, Josiah Taylor, Scott H. Hawley, and Jordi Pons. Fast timing-conditioned latent audio diffusion. In Forty-first International Conference on Machine Learning, 2024. https://openreview.net/forum?id=jOlO8t1xdx.

Eduardo Fonseca, Jordi Pons, Xavier Favory, Frederic Font, Dmitry Bogdanov, Andres Ferraro, Sergio Oramas, Alastair Porter, and Xavier Serra. Freesound datasets: A platform for the creation of open audio datasets. In ISMIR, pages 486–493, 2017.

Deepanway Ghosal, Navonil Majumder, Ambuj Mehrish, and Soujanya Poria. Text-to-audio generation using instruction-tuned LLM and latent diffusion model. In ACM Multimedia, 2023.

Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15180–15190, 2023.

Moayed Haji-Ali, Willi Menapace, Aliaksandr Siarohin, Guha Balakrishnan, and Vicente Ordonez. Taming data and transformers for audio generation. International Journal of Computer Vision, 134(3):87, 2026.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232. PMLR, 2023.

Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans. Simpler diffusion (sid2): 1.5 fid on imagenet512 with pixel-space diffusion, 2025. https://arxiv.org/abs/2410.19324.

Jiawei Huang, Yi Ren, Rongjie Huang, Dongchao Yang, Zhenhui Ye, Chen Zhang, Jinglin Liu, Xiang Yin, Zejun Ma, and Zhou Zhao. Make-an-audio 2: Temporal-enhanced text-to-audio generation. arXiv preprint arXiv:2305.18474, 2023a.

Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao. Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models. In International Conference on Machine Learning, pages 13916–13932. PMLR, 2023b.

Vladimir Iashin, Weidi Xie, Esa Rahtu, and Andrew Zisserman. Synchformer: Efficient synchronization from sparse cues. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5325–5329. IEEE, 2024.

Nal Kalchbrenner, Erich Elsen, Karen Simonyan, Seb Noury, Norman Casagrande, Edward Lockhart, Florian Stimberg, Aaron Oord, Sander Dieleman, and Koray Kavukcuoglu. Efficient neural audio synthesis. In International Conference on Machine Learning, pages 2410–2419. PMLR, 2018.

Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. Audiocaps: Generating captions for audios in the wild. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 119–132, 2019.

Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. Advances in neural information processing systems, 33:17022–17033, 2020a.

Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D Plumbley. Panns: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28:2880–2894, 2020b.

Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761, 2020c.

Khaled Koutini, Jan Schlüter, Hamid Eghbal-Zadeh, and Gerhard Widmer. Efficient training of audio transformers with patchout. arXiv preprint arXiv:2110.05069, 2021.

Felix Kreuk, Gabriel Synnaeve, Adam Polyak, Uriel Singer, Alexandre Défossez, Jade Copet, Devi Parikh, Yaniv Taigman, and Yossi Adi. Audiogen: Textually guided audio generation. arXiv preprint arXiv:2209.15352, 2022.

Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, and Kundan Kumar. High-fidelity audio compression with improved rvqgan. Advances in Neural Information Processing Systems, 36:27980–27993, 2023.

Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, and Sungroh Yoon. Bigvgan: A universal neural vocoder with large-scale training. arXiv preprint arXiv:2206.04658, 2022.

Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720, 2025.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley. Audioldm: Text-to-audio generation with latent diffusion models. arXiv preprint arXiv:2301.12503, 2023.

Haohe Liu, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong, Qiao Tian, Yuping Wang, Wenwu Wang, Yuxuan Wang, and Mark D Plumbley. Audioldm 2: Learning holistic audio generation with self-supervised pretraining. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32:2871–2883, 2024.

Huadai Liu, Kaicheng Luo, Jialei Wang, Wen Wang, Qian Chen, Zhou Zhao, and Wei Xue. Thinksound: Chainof-thought reasoning in multimodal large language models for audio generation and editing. arXiv preprint arXiv:2506.21448, 2025a.

Huadai Liu, Kaicheng Luo, Wen Wang, Qian Chen, Peiwen Sun, Rongjie Huang, Xiangang Li, Jieping Ye, and Wei Xue. Prismaudio: Decomposed chain-of-thoughts and multi-dimensional rewards for video-to-audio generation. arXiv preprint arXiv:2511.18833, 2025b.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Simian Luo, Chuanhao Yan, Chenxu Hu, and Hang Zhao. Diff-foley: Synchronized video-to-audio synthesis with latent diffusion models. Advances in Neural Information Processing Systems, 36:48855–48876, 2023.

Navonil Majumder, Chia-Yu Hung, Deepanway Ghosal, Wei-Ning Hsu, Rada Mihalcea, and Soujanya Poria. Tango 2: Aligning diffusion-based text-to-audio generations through direct preference optimization. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 564–572, 2024.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

Karin Petrini, Sofia Dahl, Davide Rocchesso, Carl Haakon Waadeland, Federico Avanzini, Aina Puce, and Frank E Pollick. Multisensory integration of drumming actions: musical expertise affects perceived audiovisual asynchrony. Experimental brain research, 198(2):339–352, 2009.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Sizhe Shan, Qiulin Li, Yutao Cui, Miles Yang, Yuehai Wang, Qun Yang, Jin Zhou, and Zhao Zhong. Hunyuanvideofoley: Multimodal diffusion with representation alignment for high-fidelity foley audio generation. arXiv preprint arXiv:2508.16930, 2025.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Zeyue Tian, Yizhu Jin, Zhaoyang Liu, Ruibin Yuan, Xu Tan, Qifeng Chen, Wei Xue, and Yike Guo. Audiox: Diffusion transformer for anything-to-audio generation. arXiv preprint arXiv:2503.10522, 2025.

Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hoffman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, et al. Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound. arXiv preprint arXiv:2502.05139, 2025.

Aaron Van Den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu, et al. Wavenet: A generative model for raw audio. arXiv preprint arXiv:1609.03499, 12(1), 2016.

Ilpo Viertola, Vladimir Iashin, and Esa Rahtu. Temporally aligned audio for video with autoregression. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025.

Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, Pierre-Antoine Manzagol, and Léon Bottou. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. Journal of machine learning research, 11(12), 2010.

Heng Wang, Jianbo Ma, Santiago Pascual, Richard Cartwright, and Weidong Cai. V2a-mapper: A lightweight solution for vision-to-audio generation by connecting foundation models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 15492–15501, 2024a.

Jun Wang, Xijuan Zeng, Chunyu Qiang, Ruilong Chen, Shiyao Wang, Le Wang, Wangjing Zhou, Pengfei Cai, Jiahui Zhao, Nan Li, et al. Kling-foley: Multimodal diffusion transformer for high-quality video-to-audio generation. arXiv

- preprint arXiv:2506.19774, 2025a.

Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. Pixnerd: Pixel neural field diffusion. arXiv

- preprint arXiv:2507.23268, 2025b.

Yongqi Wang, Wenxiang Guo, Rongjie Huang, Jiawei Huang, Zehan Wang, Fuming You, Ruiqi Li, and Zhou Zhao. Frieren: Efficient video-to-audio generation network with rectified flow matching. Advances in neural information processing systems, 37:128118–128138, 2024b.

Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.

Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. How transferable are features in deep neural networks? Advances in neural information processing systems, 27, 2014.

Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. Soundstream: An end-to-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30:495–507, 2021.

Yiming Zhang, Yicheng Gu, Yanhong Zeng, Zhening Xing, Yuancheng Wang, Zhizheng Wu, Bin Liu, and Kai Chen. Foleycrafter: Bring silent videos to life with lifelike and synchronized sounds. International Journal of Computer Vision, 134(1):46, 2026.

## Appendix

- A Training Details

- Table 6 summarizes the full training configurations for all WavFlow variants. All models are trained on

NVIDIA H100 GPUs and share the same optimizer (AdamW with β1=0.9, β2=0.95), EMA decay of 0.9999, gradient clipping at 1.0, and BF16 mixed precision. In our main experiments, 16kHz VT2A models are trained from scratch with a learning rate of 1 × 10−4 and a global batch size of 10,752 on the mixture described in Sec. 4.1 (comprising 5M media data and 200K augmented VGGSound). Within each epoch, every dataset is traversed exactly once to ensure balanced supervision. The T2A model follows the same architecture but is trained separately on the 1M T2A mixture with a batch size of 8,192. WavFlow-L-44.1kHz is obtained via supervised fine-tuning from the converged 16kHz checkpoint on 44.1kHz data, using a reduced learning rate of 1 × 10−5 and a batch size of 1,536.

- Table 6 Training hyperparameters for all WavFlow variants. All models use 16 kHz sample rate except WavFlow-L-44k (44.1 kHz).

VT2A T2A Hyperparameter WavFlow-M-16k WavFlow-L-16k WavFlow-L-44k WavFlow-M-16k Backbone† MMDiT-M MMDiT-L MMDiT-L MMDiT-M

Joint layers Ljoint 4 7 7 4 Single layers Lfused 8 14 14 8 Hidden dim d 896 896 896 896 Attention heads 14 14 14 14 Parameters 624M 1.03B 1.03B 624M

Media data 1M + T2A open source 300K Optimizer AdamW AdamW AdamW AdamW (β1, β2) (0.9, 0.95) (0.9, 0.95) (0.9, 0.95) (0.9, 0.95) Learning rate 1e-4 1e-4 1e-5 1e-4 LR schedule Constant Constant Constant Constant Warmup 20 epochs 20 epochs 20 epochs 20 epochs Global batch size 10752 10752 1536 8192 Training epochs 400 400 650 400 Initialization Scratch Scratch SFT from L-16k Scratch EMA decay 0.9999 0.9999 0.9999 0.9999 Gradient clipping 1.0 1.0 1.0 1.0 Precision BF16 BF16 BF16 BF16

Media data 5M + VGG 200K

Media data 5M + VGG 200K

VGG 200K (44.1 kHz)

Training data

Patchify (C × D) 640 × 200 640 × 200 1764 × 200 640 × 200 Audio scale 3.0 3.0 3.0 3.0 CFG drop rate (video / text) 10% / 10% 10% / 10% 10% / 10% – / 10% ODE steps (inference) 50 50 50 50 CFG strength (inference) 4.5 4.5 4.5 4.5

†Multimodal DiT backbone adopted from MMAudio (Cheng et al., 2025). “M” and “L” denote the medium and large configurations.

To determine the appropriate number of training epochs, we budget each convergence study to approximately 84hours and scale the global batch size proportionally with dataset size so that all runs complete within this timeframe. We monitor validation metrics on the VGGSound validation set (∼2K samples) throughout training across four data scales (200K, 1M, 3M, and 5M). As shown in Figure 5, models trained with ≥ 1M data and correspondingly larger batch sizes (5,632–10,752) converge by approximately 400epochs, with FDPANNs, KLPANNs, and IB scores all plateauing beyond this point. We therefore adopt 400epochs as

#### the default for both main and ablation experiments at this scale. The 200K VGGSound-only run requires approximately 650epochs to stabilize due to its much smaller batch size of 1,536, and the 44.1kHz fine-tuning setting similarly uses 650epochs to reach convergence under its reduced learning rate. Additionally, the global batch sizes labeled in Figure 5 correspond to the fixed settings utilized in our ablation studies.

Training Metrics across Data Scales

200K (VGGSound) (BS=1536)

##### 1M (VGG+mediadata) (BS=5632)

##### 3M (VGG+mediadata) (BS=8192)

##### 5M (VGG+mediadata) (BS=10752)

| |converged|
|---|---|
| | |
| | |
| |11.02<br><br>|
| | |
|400 500 55|0 600 650 700|
| |converged|
| | |
| |1.24<br><br>|
| | |
| | |
|400 500 55<br><br>|0 600 650 700<br><br>converged|
| | |
| |0.29|
| | |
| | |
| | |
|400 500 55<br><br>Epo<br><br>Metrics o<br><br>1,536), 1 M ( row: IB ↑<br><br>epochs due BS denotes<br><br>Audio<br><br>Waveform-space<br><br>observ 6 illust<br><br>as Audio 0, Audio 4, 9 an<br><br>negligib smal|0 600 650 700<br><br>ch<br><br>n<br><br>BS=5,632<br><br>. Models to its smaller the global<br><br>Amplitude<br><br>amplitudes ations rates this<br><br>2, and 14, d 11, show<br><br>le magnitudes l loss|

| |converged|
|---|---|
| | |
| | |
| |9.78|
| | |
|240 280 320|360 400 440|
| |converged|
| | |
| |1.23|
| | |
| | |
|240 280 320<br><br>|360 400 440<br><br>converged|
| |0.33<br><br>|
| | |
| | |
| | |
| | |
|240 280 320<br><br>Epo<br><br>(BS=8,192) with ≥ 1 M<br><br>size. Da size.<br><br>Distribution<br><br>a wide d that most<br><br>plotting the am only a m<br><br>yet st and are eas<br><br>insu|360 400 440<br><br>ch<br><br>, and 5 M samples shed lines<br><br>ynamic waveform<br><br>plitude<br><br>inimal ill sharply ily submerged<br><br>fficient|

| |converged|
|---|---|
| | |
| | |
| |9.65<br><br>|
| | |
|240 280 320|360 400 440|
| |converged|
| | |
| | |
| |1.18<br><br>|
| | |
|240 280 320<br><br>0.34|360 400 440<br><br>converged|
| | |
| | |
| | |
| | |
| | |
|240 280 320<br><br>Epo<br><br>,752). Top around 40 the conv<br><br>yet values a levels typi<br><br>of 15 r of the avail<br><br>zero-centered by noise du informatio|360 400 440<br><br>ch<br><br>row: FD 0 epochs, ergence<br><br>re heavily cally remain andomly able dynamic<br><br>distributions. ring training. n and|

| |converged|
|---|---|
| | |
| | |
| |9.90<br><br>|
| | |
|240 280 320|360 400 440|
| |converged|
| | |
| | |
| |1.18|
| | |
|240 280 320<br><br>0.34|360 400 440<br><br>converged|
| | |
| | |
| | |
| | |
| | |
|240 280 320<br><br>Epo<br><br>scales: 200 K middle row<br><br>200 K sett adopted for fi<br><br>nea<br><br>0.2. Th clips. Qu<br><br>range, while o<br><br>low-ene This furthe<br><br>denoisin|360 400 440<br><br>ch<br><br>VGGSound : KLPANNs<br><br>ing requires nal training.<br><br>r zero, with e left panel iet samples,<br><br>thers, such rgy signals r results g|

16

16

16

16

14

14

14

14

FD-PANN

12

12

12

12

10

10

10

10

4

2

2

2

1.40

1.40

1.40

1.40

1.35

1.35

1.35

1.35

##### KL-PANN

1.30

1.30

1.30

1.30

1.25

1.25

1.25

1.25

1.20

1.20

1.20

1.20

1.15

1.15

1.15

1.15

4

2

2

2

0.34

0.34

0.34

0.34

0.32

0.32

0.32

0.32

##### IB-Score

0.30

0.30

0.30

0.30

0.28

0.28

0.28

0.28

0.26

0.26

0.26

0.26

4

2

2

2

- Figure 5 the VGGSound validation set as a function of training epochs across four data s

(BS=1 ), 3 M (BS=10 PANNs↓; ↓; bottom trained converge while the s ∼650 e er batch indicate points ad . Here, batch

- B A Dis Wavef exhibit range, y concentrated empirical indicating RMS in below l of Figure by plo histograms sampled , such a occupy fraction ic ra

- as Au broader zero- . These possess des a ged b ng. T in vanishingly values, providing gradient making the objective difficult to optimize.

In contrast, after RMS normalization to a target level of 0.33 followed by ×3.0 amplitude scaling (right panel), the signals are effectively lifted, with distributions spreading more broadly across the [−3,3] range. While the resulting distributions remain non-uniform, their amplitude statistics are much better aligned with the standard N(0,1) noise prior than the raw waveforms. This redistribution ensures that the signal remains discernible even at high noise levels, rendering the denoising objective significantly more tractable and training more stable. The quantitative impact of this preprocessing is validated through ablation experiments (Table 5).

Raw Waveform Amplitude Distribution Raw Waveform Amplitude after Preprocessing

[Figure 7]

[Figure 8]

- Figure 6 Amplitude histograms of 15 randomly sampled audio clips before (left, blue) and after (right, green) preprocessing (RMS normalization to 0.33 followed by ×3.0 scaling). Raw waveforms exhibit sharp zero-centered peaks with widely varying energy across clips. After preprocessing, the amplitude values are more consistent, lifting the signal to ensure it is not submerged by the Gaussian noise prior during flow matching.

- C Data Mixture and Rationale for VT2A

This section describes our process for selecting the optimal training data mixture for VT2A generation and the reasoning behind our final configuration. We consider three primary data sources: (1) VGGSound (200K), which consists of paired video-audio samples with sparse labels (e.g., “dog barking”); (2) Open-source T2A data (300K), which includes audio-only samples from FreeSound and AudioCaps featuring fine-grained captions (e.g., “a vehicle engine accelerating then running on idle”); and (3) Media data, a large-scale collection of 5M high-quality proprietary video-audio pairs derived from the MovieGen (Polyak et al., 2024) training subset, featuring fine-grained captions comparable in detail to the Open-source T2A data. For experimental efficiency in our ablation studies, we utilize a 1M representative subset of this collection.

Our exploration began by directly mixing VGGSound (Sparse label) with the Open-source T2A data (Finegrained caption). However, this configuration consistently led to training divergence, where the loss would initially decrease but eventually spike. We attribute this failure to a severe semantic mismatch between the two text styles. Without a visual modality in the T2A samples to act as a grounding bridge, the text encoder embeddings for sparse labels and descriptive captions occupy disparate regions of the latent space, preventing the model from establishing a consistent audio-conditioning mapping.

To address this, we utilized Qwen3.0-Omni (30B) (Xu et al., 2025) to generate dense audio-visual descriptions for the VGGSound dataset, rephrasing them to align with the description style of the Open-source T2A data. This “Dense” VGGSound variant successfully stabilized the training when mixed with T2A data. However, as shown in Table 7, the resulting performance was inferior to the baseline trained solely on VGGSound (DeSync 0.52 → 0.57, IB 0.29 → 0.26). This suggests that while semantic alignment in text can prevent divergence, audio-only data cannot substitute for the rich structural information provided by paired video-audio samples in raw-space generation.

Consequently, we introduced the 1M Media data subset to provide explicit visual supervision. Through our experiment, mixing VGGSound (Sparse label) with Media data (Fine-grained caption) converges reliablyunlike the T2A mixture—despite the disparity in text granularity. This confirms that the presence of the visual modality in both datasets allows the model to align concepts across different text styles by using visual features as a common semantic anchor.

Finally, we compared the impact of caption quality within this stabilized mixture by evaluating VGGSound (Sparse label) + Media data against VGGSound (Dense) + Media data. The Dense variant yields higher ISPANNs (17.05 vs. 16.42), indicating that fine-grained training descriptions help the model learn more diverse and complex audio-visual semantic mappings that generalize even when evaluation captions remain sparse.

- Table 7 Effect of data mixture and caption granularity on VT2A generation ( WavFlow-M-16k, 1M training data scale, evaluated on VGGSound-Val). “Open-source T2A” refers to a mixture of FreeSound and AudioCaps. Note: Dense captions are applied to the VGGSound training set only; the VGGSound validation set utilizes native sparse labels for evaluation.

Data Mixture VGG Caption FDPaSST ↓ FDPANNs ↓ KLPANNs ↓ ISPANNs ↑ IB↑ DeSync↓ VGGSound (200K) Sparse label 141.64 11.16 1.26 12.83 0.29 0.52 VGGSound + Open-source T2A Sparse label training diverged (semantic mismatch) VGGSound + Media data (1M) Sparse label 121.09 9.58 1.20 16.42 0.33 0.47 VGGSound + Open-source T2A Dense 139.23 12.97 1.28 12.59 0.26 0.57 VGGSound + Media data (1M) Dense 125.52 11.20 1.35 17.05 0.33 0.49

However, the Sparse label variant achieves superior FD and DeSync scores (e.g., FDPaSST of 121.09 vs. 125.52) due to the better consistency between the training and evaluation text distributions. Given its superior distributional fidelity and temporal synchronization, we select VGGSound (Sparse label) + Media data as our final training mixture.

D Waveform Patchify Configuration

As discussed in Section 3.2 and Section 4.4, we reshape an 8-second, 16kHz waveform (T = 128,000 samples) into a C × D token grid. As illustrated in Figure 7, for audio samples where T is not exactly divisible by D, we apply zero-padding to the waveform prior to patching; this padding is subsequently truncated during the unpatchify process to restore the original waveform length T.

We sweep the patch dimension D from 512 down to 160 to determine the optimal granularity for waveform modeling (Table 8). This sweep includes specific configurations, such as C = 576 (192 × 3) and C = 768 (192 × 4), designed to align the audio token count with the Synchformer feature length (192 tokens) to test if such explicit choice benefits temporal alignment.

Input: Waveform

( 1 , T )

( 1 , C , D)

Zero Padding ( If T mod D ? 0 ) Reshape

( 1 , C*D )

- Figure 7 Waveform patchify illustration. A 1D waveform is reshaped into a 2D token grid of shape C × D. Zero-padding is naturally applied to handle arbitrary waveform lengths, which is then removed during unpatchify to recover the original T samples.

- Table 8 Ablation on patchify granularity (WavFlow-M-16k, 3M training data scale, evaluated on VGGSound-Val). All configurations target an 8 s waveform (T = 128,000).

Patchify (C × D) Token dur. (ms) FDPaSST ↓ FDPANNs ↓ KLPANNs ↓ ISPANNs ↑ IB↑ DeSync↓

250 × 512 32.0 139.45 11.93 1.41 16.14 0.312 0.48 500 × 256 16.0 124.02 9.84 1.18 16.72 0.335 0.48 576 × 224 13.9 125.69 10.02 1.20 16.92 0.339 0.45 640 × 200 12.5 120.88 10.02 1.16 17.20 0.334 0.45 768 × 168 10.4 124.05 9.91 1.18 17.00 0.338 0.47 800 × 160 10.0 121.83 9.90 1.22 17.06 0.340 0.46

The results in Table 8 show that the coarsest configuration (D = 512, 32ms per token) performs significantly worse, indicating that the model requires finer granularity to capture waveform details. Once the patch dimension D is reduced below 256 (16ms), the generative performance and synchronization metrics stabilize

- at a high level.

Notably, we found that configurations specifically designed for sync-alignment (e.g., C = 576 or 768) did not yield substantial improvements over other fine-grained settings. This suggests that the model is robust to various token counts as long as the temporal resolution is sufficient. Considering the balance between generative quality, computational efficiency, and synchronization, we select 640 × 200 as the default configuration for our 16kHz experiments.

- E Effect of Noise-Level Shift

We evaluate the impact of noise-level shifting on the VGGSound test set. Prior work has shown that increasing the noise level during training can improve generation performance, particularly for high-resolution images (Li and He, 2025; Hoogeboom et al., 2023). Motivated by this finding, we investigate whether noise shift similarly benefits waveform-space generation.

We parameterize noise shift as ts = t/(t + s · (1 − t)) where s is the shift factor, which biases training toward higher noise levels and reduces the effective signal-to-noise ratio by a factor of s2.

- Table 9 Effect of noise shift on VT2A generation (WavFlow-M-16k, 1M training data scale, evaluated on VGGSound-Test). All configurations target an 8 s waveform (T = 128,000).

Noise shift s FDPaSST ↓ FDPANNs ↓ KLPANNs ↓ ISPANNs ↑ IB↑ DeSync↓

1.0 63.05 6.21 1.76 15.58 0.28 0.50 3.0 73.17 7.52 1.72 13.64 0.26 0.50 5.0 92.21 10.25 1.80 11.81 0.23 0.55

- As shown in Table 9, unlike in image generation, noise shift provides little benefit for waveform-space modeling

and progressively degrades all metrics as s increases, with FDPaSST rising from 63.05 (s = 1.0) to 73.17 (s = 3.0) and then to 92.21 (s = 5.0). We attribute this to the fundamental difference in signal characteristics: image pixels span a broad value range where high-noise training helps the model capture low-frequency global structure, whereas audio waveforms have inherently low information density even after our preprocessing. In this regime, increasing the noise level makes the already weak audio signal even harder to recover, degrading rather than improving learning. We therefore adopt s = 1.0 (no shift) as our default.

F Inference Hyperparameters

We ablate two key inference-time hyperparameters on VT2A (VGGSound validation set): the classifier-free guidance (CFG) strength and the number of ODE integration steps.

- As shown in Table 10, CFG strength has a pronounced effect on generation quality. Without sufficient guidance (CFG=1.0), the model produces low-fidelity outputs with poor semantic alignment (IB: 0.26) and severe

temporal artifacts (DeSync: 0.71). Increasing CFG to 2.5 yields the best distributional fidelity (FDPaSST: 108.19), while CFG=4.5 achieves the highest per-sample quality (IS: 16.42, IB: 0.33). Beyond 4.5, further increasing guidance to 7.0 causes FDPaSST to degrade sharply, indicating reduced output diversity without meaningful quality gains.

For ODE steps, the transition from 10 to 25 steps brings a significant quality improvement, and going from 25 to 50 steps yields further gains in IS (16.19 → 16.42) and IB (0.32 → 0.33). Beyond 50 steps, metrics plateau entirely, with 100 steps providing no additional benefit. We therefore adopt CFG =4.5 with 50 ODE steps as our default inference configuration, which offers the best generation quality before diminishing returns set in.

- Table 10 Effect of inference hyperparameters on VT2A (WavFlow-M-16k, 1M training data scale, evaluated on VGGSoundVal). All configurations target an 8 s waveform (T = 128,000).

Setting FDPaSST ↓ FDPANNs ↓ KLPANNs ↓ ISPANNs ↑ IB↑ DeSync↓ CFG strength (50 steps)

- 1.0 145.53 16.34 1.54 9.86 0.26 0.71
- 2.5 108.19 9.40 1.26 15.44 0.32 0.51 4.5 121.09 9.58 1.20 16.42 0.33 0.47 7.0 142.00 9.92 1.22 16.19 0.32 0.46

ODE steps (CFG=4.5)

10 118.90 10.73 1.30 14.93 0.31 0.47 25 117.63 9.78 1.23 16.19 0.32 0.47 50 121.09 9.58 1.20 16.42 0.33 0.47 100 124.53 9.41 1.20 16.39 0.33 0.47

G Evaluation on MovieGen-Audio-Bench

To further evaluate the generalization of WavFlow, we conduct additional experiments on the MovieGenAudio-Bench (Polyak et al., 2024). This benchmark is particularly challenging as it consists of AI-generated videos rather than real-world recordings. Performing audio synthesis for such synthetic content requires the model to establish a fundamental understanding of audio-visual correlation.

Since this benchmark does not provide ground-truth audio, we follow the protocol in (Polyak et al., 2024) and report reference-free metrics: Inception Score (IS) for audio quality, CLAP and IB-score for semantic alignment, and DeSync for temporal synchronization. Table 11 presents the comparison between WavFlow, MMAudio, and MovieGen.

- Table 11 Evaluation on MovieGen-Audio-Bench. Best results are in bold, and second-best are underlined. Reference-free metrics are used as no ground-truth audio is available for this benchmark.

Method Params Training Data IS↑ CLAP↑ IB-score↑ DeSync↓

WavFlow (ours) 1.03B ∼ 11.1K h 8.95 0.28 0.24 0.77 MMAudio 1.03B ∼ 8.2K h 8.40 0.28 0.27 0.77 MovieGen 13B ∼ 1,000K h 8.89 0.29 0.36 1.00

Experimental results show that WavFlow generalizes effectively to synthetic visual content. Notably, WavFlow achieves an Inception Score (IS) of 8.95 and a DeSync score of 0.77, outperforming or matching existing latent-based models in audio quality and temporal synchronization. While MovieGen maintains an advantage in semantic richness (IB-score), likely due to its significantly larger model capacity and training scale, WavFlow remains highly competitive. These results suggest that direct raw-waveform synthesis, without the aid of a pretrained VAE, achieves performance comparable to state-of-the-art latent-based paradigms across both real and synthetic benchmarks.

To qualitatively evaluate temporal synchronization, we provide spectrogram visualizations of samples from the MovieGen-Audio-Bench in Figures 8 to 10. Detailed analyses for each case are provided in the respective figure captions.

WavFlowMMAudioMovieGen

[Figure 14]

- Figure 8 Spectrogram comparison on the "Penguin Walking" scenario. All evaluated models demonstrate precise temporal synchronization. WavFlow exhibits sharper and more vertical energy pulses, reflecting its ability to preserve fine-grained acoustic transients through direct raw-waveform modeling.

[Figure 15]

Punches the air

- Figure 9 Spectrogram comparison on the "Boxing" scenario. Acoustically, both WavFlow and MMAudio achieve precise synchronization, while MovieGen exhibits desynchronization. Crucially, during the air-punching segment (no bag contact), WavFlow correctly omits the impact sound, whereas MMAudio and MovieGen erroneously synthesize strike energy. This highlights WavFlow’s precision in discerning subtle visual nuances for accurate raw-waveform synthesis.

WavFlowMMAudioMovieGen

[Figure 21]

- Figure 10 Spectrogram comparison on the "Horse Trotting" scenario. In this scenario featuring consistent, rhythmic hoofbeats, WavFlow achieves precise temporal synchronization, characterized by distinct and sharp vertical energy pulses that correspond to each footfall. While MMAudio maintains some rhythmic alignment, its spectral pulses are noticeably less defined and lack the transient sharpness seen in our model. In contrast, MovieGen fails to establish a clear temporal relationship with the visual motion.

