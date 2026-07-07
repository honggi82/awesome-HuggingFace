# arXiv:2602.09070v2[cs.SD]12Feb2026

## NarraScore: Bridging Visual Narrative and Musical Dynamics via Hierarchical Affective Control

Zhaocheng Liu∗

Yufan Wen

YeGuo Hua

Tsinghua University Shenzhen, China wenyf24@mails.tsinghua.edu.cn

ByteDance Beijing, China lio.h.zen@gmail.com

ByteDance Beijing, China huayeguo@bytedance.com

Ziyi Guo

ByteDance Shenzhen, China ziyi.94@bytedance.com

Lihua Zhang

ByteDance Beijing, China lizhiyu.0@bytedance.com

Chun Yuan∗

Tsinghua University Shenzhen, China yuanc@sz.tsinghua.edu.cn

Jian Wu

ByteDance Beijing, China wujian@bytedance.com

### Abstract

Synthesizing coherent soundtracks for long-form videos remains a formidable challenge, currently stalled by three critical impediments: computational scalability, temporal coherence, and, most critically, a pervasive semantic blindness to evolving narrative logic. To bridge these gaps, we propose NarraScore, a hierarchical framework predicated on the core insight that emotion serves as a high-density compression of narrative logic. Uniquely, we repurpose frozen Vision-Language Models (VLMs) as continuous affective sensors, distilling high-dimensional visual streams into dense, narrative-aware Valence-Arousal trajectories. Mechanistically, NarraScore employs a Dual-Branch Injection strategy to reconcile global structure with local dynamism: a Global Semantic Anchor ensures stylistic stability, while a surgical Token-Level Affective Adapter modulates local tension via direct element-wise residual injection. This minimalist design bypasses the bottlenecks of dense attention and architectural cloning, effectively mitigating the overfitting risks associated with data scarcity. Experiments demonstrate that NarraScore achieves state-of-the-art consistency and narrative alignment with negligible computational overhead, establishing a fully autonomous paradigm for long-video soundtrack generation.

dynamics and evolving emotional arcs. With the rapid advancement

[Figure 1]

Figure 1: Narrative-aware video background music generation. Unlike baselines that rely on surface visuals and fail to capture narrative tension or hidden subtext, our approach leverages global plot 𝑆𝑔𝑙𝑜𝑏𝑎𝑙 and local emotion 𝐸𝑙𝑜𝑐𝑎𝑙 to generate soundtracks that are temporally coherent and narratively resonant.

### CCS Concepts

• Applied computing → Sound and music computing.

### Keywords

Video-to-Music Generation, Long-form Video Soundtrack Generation, Affective Narrative Alignment, Controllable Music Generation

### 1 Introduction

Background music serves as the emotional pulse of multimedia content, functioning as a narrative engine that actively shapes viewer immersion rather than merely accompanying visuals [7, 29, 30]. A professional-grade soundtrack is defined by its organic synchronization with visual progression, mirroring the underlying temporal

∗Corresponding author.

of generative models[16, 27, 31, 41] enabling the transition from short clips to long-form video creation, the core challenge has fundamentally shifted, transcending the synthesis of high-fidelity static loops to necessitate the orchestration of complex, evolving soundtracks that maintain stylistic unity while dynamically responding to the shifting pacing and intensity of the visual narrative.

Recent paradigms, while achieving impressive fidelity on short clips via strict frame-wise guidance [10, 14, 20, 26, 35, 49], encounter systemic bottlenecks when extrapolated to long-form narratives. Physically, maintaining dense frame-level attention across minutelong sequences incurs prohibitive quadratic memory costs and suffers from attention dilution, where critical narrative cues are drowned out by visual redundancy. Structurally, the absence of global semantic anchors in standard autoregressive models [1, 6, 18] leads to severe style drift, causing the musical identity to fragment

over time. Most critically, at a cognitive level, these methods rely on surface-level visual representations [12, 33, 39, 40], resulting in a semantic blindness—an inability to capture the deep narrative logic, such as rising tension or resolution, which is essential for cinematic storytelling [19].

To bridge this semantic gap, we propose NarraScore, a hierarchical framework predicated on the core insight that emotion serves as a high-density compression of deep narrative logic. Moving beyond paradigms that depend on unstable external classifiers or expensive manual annotation, we introduce a Lightweight Latent Affective Decoder designed to probe the rich semantic priors encapsulated within frozen Vision-Language Models (VLMs)[4, 45, 46]. By explicitly disentangling deep narrative cues from surface-level visual redundancy, this module projects high-dimensional video streams into a compact, autonomous affective manifold. Consequently, NarraScore empowers the system to endogenously deduce evolving emotional arcs directly from raw pixels, establishing a new paradigm of autonomous narrative alignment.

Translating this insight into a generative architecture, NarraScore employs a Dual-Branch Injection strategy to reconcile global coherence with local dynamism. At the macro level, a Global Semantic Anchor conditions the model on the overarching genre and atmosphere, ensuring stylistic stability. At the micro level, we introduce a Token-Level Affective Adapter to modulate finer narrative tension [24, 47]. Diverging from prevalent paradigms that rely on heavy architectural cloning, we adopt a minimalist projection strategy: distilled continuous affective cues (Valence & Arousal) are injected as a lightweight additive bias directly into the decoder’s hidden states. This mechanism modulates the semantic manifold via direct additive bias, enabling precise token-level alignment while introducing negligible parameter overhead and preserving the generative priors of the frozen backbone.

In summary, this work makes three key contributions to the field of intelligent content generation. First, we establish a Pioneering Affective-Semantic Bridge, explicitly validating the emergent capability of frozen VLMs to distill complex narrative intents into continuous emotion curves, thus bypassing the data bottleneck of traditional recognition. Second, we propose a data-efficient Dynamic Control mechanism via a token-level adapter. This module enables fine-grained tension modulation with minimal training overhead, achieving precise narrative alignment without disrupting the overarching musical structure. Finally, NarraScore achieves a synergistic balance between global style and local emotion, paving the way for fully automated, high-quality cinematic soundtrack generation.

### 2 Related Work 2.1 Video to Music Generation

The trajectory of video soundtrack generation has evolved from rule-based symbolic mappings to data-driven deep generative modeling. Early paradigms, represented by methods like CMT [10] and Video2Music [20], pioneered the Video-to-Music task by analyzing visual motion features to predict symbolic MIDI events. While foundational, these approaches often necessitated explicit user guidance to bridge the modality gap, resulting in limited expressive diversity and a heavy reliance on manual intervention.

The advent of audio language models marked a shift toward token-based generation. Recent frameworks such as MuVi [25], VMAS [26], GVMGen [49], and VeM [38] leverage adapters to project dense video frames into latent conditions compatible with pre-trained backbones like MusicGen [6]. Although effective for short clips, these methods predominantly rely on dense frame-level attention mechanisms. This design introduces severe scalability bottlenecks: for minute-level videos, the computational cost becomes prohibitive, and the dilution of attention leads to style drift and a loss of long-term coherence [37].

To address these constraints, recent works have explored specific mechanisms for long-form generation. VidMuse [37] proposes specialized adapters to effectively model both long-term and short-term temporal features, employing a sliding-window inference strategy to ensure computational viability. JenBridge [44] adopts a divideand-conquer approach, segmenting videos for independent scoring and stitching them via transition techniques.

However, while these methods achieve acoustic continuity, they overlook the fundamental semantic shift in long-form content. Unlike short clips where a static mood suffices, long narratives possess evolving logic—tension rises, resolves, and shifts. By treating long videos merely as extended sequences, current solutions fail to capture these dynamic arcs, yielding monotonous background ambience rather than responsive, narrative-aligned soundtracks.Consequently,enablingthesoundtrack to dynamically evolve in resonance with the narrative trajectory—transcending mere acoustic continuity—stands as the pivotal challenge for advancing video scoring toward professional-grade viability.

### 2.2 Emotion-Driven Music Generation

Given the intrinsic link between visual storytelling and musical expression, leveraging emotion as a conditioning signal has become a focal point. Early approaches primarily relied on discrete classification or global mapping. Video2Music [20] utilized CLIP for frame-level emotion classification, while EMSYNC [36] advanced this by employing a psychology-driven mapping mechanism to translate discrete categorical predictions into continuous Valence and Arousal (VA) values. Despite these efforts, these methods rely heavily on surface-level visual semantic analysis, where the inherent accuracy limitations of standard CLIP-based classifiers often lead to coarse or noisy affective guidance.

To improve semantic fidelity of emotional control, subsequent workshaveintegratedmorerobustpriors.Methods like M2UGen [28], FilmComposer [43], and JenBridge [44] leverage Large Language Models (LLMs) to analyze visual content and generate descriptive emotion captions. While these methods demonstrate proficiency in synthesizing accurate global emotion labels, they largely circumvent the use of continuous emotion curves. This avoidance stems primarily from the scarcity of continuous affective data and the prohibitive cost of fine-grained annotation. Yet, for long-form video soundtrack generation, such dense temporal control is indispensable for maintaining plot consistency.

Although approaches like VeM [38] and MTCV2M [42] attempt to incorporate fine-grained control, they are fundamentally limited by their reliance on extrinsic guidance. This requirement for

manual intervention renders such paradigms unscalable for autonomous, large-scale applications. Consequently, the automated and parameter-efficient extraction of deep, continuous affective cues solely from visual narratives remains a critical gap in achieving robust, high-fidelity control.

### 2.3 Emotion Recognition from Video

Affective video analysis serves as the perceptual foundation for emotion-driven content generation. However, the high annotative burden of continuous emotion labeling has led to a persistent scarcity of large-scale, high-quality datasets. To mitigate this supervision bottleneck, recent paradigms have shifted towards exploiting the generalized semantic representations of Vision-Language Pretraining (VLP) models. Pioneering works have demonstrated that contrastive models, such as CLIP, possess robust zero-shot capabilities for static image emotion classification. Extending this to the temporal domain, methods like EmoCLIP [13, 48] leverage CLIP embeddings to predict holistic video-level labels, while hybrid architectures integrate frozen visual features with learnable temporal modules to perform continuous Valence-Arousal (VA) regression.

Despite these advancements, directly deploying current Stateof-the-Art (SOTA) affective recognition models as conditioning priors for music generation remains problematic. A fundamental domain misalignment exists within mainstream benchmarks like AFEW-VA [23], Aff-Wild2 [22], and VEATIC [32], which are predominantly face-centric, focusing on decoding the expressed emotion of actors via facial dynamics. In contrast, video soundtrack generation necessitates interpreting the induced affect—the overarching atmosphere and narrative tension perceived by viewers—rather than local facial cues. While datasets like LIRIS-ACCEDE [3] attempt to align with viewer perception, the limited scale across these benchmarks precludes the training of robust, data-hungry models. Furthermore, existing SOTA methods often lack the high-level semantic reasoning required to capture nuanced narrative shifts, resulting in noisy and temporally inconsistent predictions in complex, non-facial scenes. Consequently, relying on off-the-shelf emotion estimators as ground truth risks introducing significant error propagation, underscoring the imperative for approaches that leverage the deep reasoning capabilities of Large Language Models to infer contextually accurate emotion curves from limited supervision.

3 Methodology

### 3.1 Problem Definition

The primary objective of this work is to generate a musical sequence that mirrors the narrative progression of a long-form video. Formally, let V = {𝑣1, . . .,𝑣𝑇𝑣} denote the input video sequence consisting of 𝑇𝑣 frames. The target output is a discrete acoustic sequence A ∈ {1, . . ., 𝑁}𝑇𝑎×𝐾, derived from a neural audio codec [8] with 𝐾 residual codebooks and a vocabulary size of 𝑁. Here, 𝑇𝑎 denotes the sequence length. The acoustic sequence is inherently dense, whereas the visual sequence 𝑇𝑣 is kept sparse align with the memory capacity limits when processing minute-level inputs. Unlike short-clip generation, this task imposes dual constraints: global coherence for unified musical style, and local alignment for frame-level narrative synchrony.

### 3.2 Overview

To effectively reconcile these requirements, we introduce NarraScore, a hierarchical framework that disentangles the video-to-music generation task into two orthogonal dimensions: macro-scale atmospheric modeling and micro-scale tension tracking. Specifically, we bridge the semantic gap between the visual stream V and the acoustic domain A by introducing two decoupled priors: a global semantic anchor S𝑔𝑙𝑜𝑏𝑎𝑙 and a frame-level affective trajectory E𝑙𝑜𝑐𝑎𝑙. Consequently, the generative process is formulated as an auto regressive sequence prediction conditioned on these hierarchical cues:

𝑇𝑎

𝑝(𝑎𝑡 | 𝑎<𝑡, S𝑔𝑙𝑜𝑏𝑎𝑙, E𝑙𝑜𝑐𝑎𝑙) (1)

𝑝(A | V) =

𝑡=1

As illustrated in Figure , the proposed architecture operationalizes this formulation through a cascaded Perception-Synthesis Pipeline. The workflow is anchored by a Unified Visual-Narrative Backbone, instantiated as a hybrid encoder integrating a Vision Transformer (ViT) [11] front-end with a deep contextual reasoning stack. Leveraging a bifurcated projection mechanism, this unified system decomposes the raw visual stream into hierarchical control priors: it simultaneously regresses the frame-level affective trajectory E𝑙𝑜𝑐𝑎𝑙 to delineate the micro-evolution of narrative tension, while abstracting the global semantic anchor S𝑔𝑙𝑜𝑏𝑎𝑙 via a language modeling head to encapsulate the macro-atmospheric context. In the subsequent generative phase, these disentangled representations are integrated into a conditional acoustic transformer via distinct conditioning pathways, where the global anchor establishes the timbral and stylistic foundation, while the affective trajectory modulates the evolving narrative tension and musical dynamics.

### 3.3 Narrative-Aware Affective Reasoning

Deriving accurate emotion from video necessitates transcending static visual perception to capture evolving narrative dynamics. Rather than training a task-specific temporal encoder from scratch, we introduce a paradigm of Latent Semantic Probing. This approach functions as a unified spatiotemporal engine and orchestrates the rich reasoning priors of a frozen Vision-Language Model [45] to lift raw pixel streams into a continuous affective manifold.

Semantically-Anchored Temporal Alignment. To facilitate robust causal reasoning over long-form sequences, our framework adopts a strategy of Semantically-Anchored Temporal Alignment. Thisformulationleveragesthearchitectural priors of the pre-trained backbone by employing linguistically grounded progression instead of rigid temporal embeddings. We discretize the video stream into a uniform 1Hz sequence to balance narrative granularity against the computational constraints of the backbone. These snapshots are interleaved with discrete semantic clocks 𝜏𝑡, formatted textually as “Time:𝑡s”, to strictly conform to the native interleaved schema of the model. The visual tokens𝑉𝑡 retain their spatial fidelity via the intrinsic positional encoding mechanism. This effectively reconstructs the video as a linear causal sequence and establishes a continuous spatiotemporal context for subsequent affective reasoning.

Instruction-Driven Semantic Steering. We bridge the domain gap between generalist object recognition and nuanced affective analysis by repurposing the instruction-following interface of the

#### Figure 2: Overview of our framework

backbone. Rather than introducing external control modules, we optimize a system instruction T𝑖𝑛𝑠𝑡 to serve as a Semantic Primer. By prepending this primer to the aligned video sequence, we formulate the input representation 𝑋 as:

where 𝜆 serves as a balancing coefficient.

### 3.4 Holistic Musical Conceptualization

Complementing the micro-level tension tracking, this component focuses on distilling the visual narrative into a global semantic anchor. The primary objective is to orchestrate the macro-level auditory identity that governs the overarching musical style and structural coherence.

𝑋 = [T𝑖𝑛𝑠𝑡,𝜏1,𝑉1,𝜏2,𝑉2, . . .,𝜏𝑇,𝑉𝑇] (2)

Functionally, T𝑖𝑛𝑠𝑡 modulates the self-attention mechanism to explicitly suppress the activation of low-level object enumeration patterns while activating high-level narrative reasoning pathways. This strategy steers the pre-trained capabilities towards analyzing narrative tension and emotional evolution to ensure that the extracted representations align with the affective task.

To bridge the semantic inference gap between visual features and musical concepts, we re-frame the interpretation task as a cross-modal sensory translation. Instead of relying on raw visual features, we leverage the reasoning priors of the VLM to identify the visual scene while exclusively describing the implied auditory imagery. This strategy achieves modality decoupling by explicitly suppressing references to specific visual objects and cinematography. It forces the model to abstract away from scene-specific visual nouns and project the content directly into acoustic descriptors semantically aligned with the downstream audio synthesis model [6].To guarantee robustness and consistency, we employ a structured instruction paradigm rather than free-form generation. We impose a strict schema that compels the VLM to synthesize a unified natural language description encompassing four essential musical dimensions: genre and stylistic context, instrumentation and timbral texture, emotional atmosphere, and rhythmic pacing. By projecting the visual content onto this predefined semantic subspace, we significantly reduce output variance and mitigate the ambiguity often associated with open-ended captioning. This structured constraint ensures that the generated control signals remain musically coherent and aligned with the intended aesthetic direction.

Latent Affective Probing. To quantify the narrative tension distilled by the backbone, we introduce a lightweight probing head designed to extract the continuous affective trajectory E𝑙𝑜𝑐𝑎𝑙. Let 𝑍𝑡 ⊂ 𝐻(𝐿) denote the set of contextualized hidden states corresponding to the visual tokens of the 𝑡-th frame. We first aggregate these tokens via Spatial Average Pooling to obtain a holistic frame-level representation. This vector is then projected onto the Valence-Arousal plane [34] via a Multi-Layer Perceptron:

∑︁

1 𝑀

𝑒𝑡 = Clip[−1,1] MLP

𝑧 (3)

𝑧∈𝑍𝑡

We freeze the massive backbone and train only this lightweight probe. This effectively distills the inherent ability of the VLM to correlate visual changes with explicit temporal progression into the specific task of affective regression and achieves high-fidelity prediction with minimal computational overhead.

Optimization Objective. We employ a hybrid objective function combining L2 and L1 norms to calibrate the probing head against the target affective curves. This formulation balances the convergence stability provided by the Mean Squared Error with the robustness against outliers offered by the Mean Absolute Error. Let 𝑒ˆ𝑡 ∈ R2 denote the ground-truth Valence-Arousal vector for the 𝑡-th frame. The optimization objective L𝑒𝑚𝑜 is defined as:

### 3.5 Hierarchical Acoustic Synthesis

The synthesis framework transforms the extracted narrative priors into a coherent acoustic waveform. To achieve this translation without compromising the high-fidelity generative distribution of the pre-trained music backbone [6], we employ a Dual-Stream Injection strategy. This mechanism aligns distinct control signals,

∑︁𝑡 = 1𝑇 |𝑒𝑡 − 𝑒ˆ𝑡|22 + 𝜆|𝑒𝑡 − 𝑒ˆ𝑡|1 (4)

1 𝑇

L𝑒𝑚𝑜 =

#### Figure 3: Our Method of Token-Wise Control Injection.

comprising global style and local tension, with the hierarchical levels of the acoustic decoder.

𝐶𝑙𝑜𝑐𝑎𝑙 exclusively to the shallow Transformer blocks. This design choice implies that early layers are utilized to align the generation trajectory with the continuous affective constraints, thereby allowing the deeper layers to focus on the optimization of acoustic fidelity and harmonic coherence without interference. Formally, we

Explicit SemanticBridging. TheglobalsemanticanchorS𝑔𝑙𝑜𝑏𝑎𝑙 is integrated directly into the acoustic decoder via the pre-trained cross-attention mechanism. By utilizing the derived semantic description as the conditioning context, we steer the generative trajectory toward the target genre and atmosphere identified in the visual analysis phase. This standard conditioning approach ensures that the fundamental musical structure and instrumentation align with the narrative intent while preserving the original synthesis capabilities of the backbone without requiring invasive architectural modifications.

apply a learnable additive bias to the hidden states ℎ𝑡(𝑙) at time step 𝑡 and layer 𝑙:

ℎ𝑡(𝑙)′ = ℎ𝑡(𝑙) +𝛾 · 𝐶𝑙𝑜𝑐𝑎𝑙,𝑡, ∀𝑙 ∈ {1, . . .,𝐿𝑠ℎ𝑎𝑙𝑙𝑜𝑤} (6)

Here, 𝛾 is a learnable scalar initialized to zero to ensure the model retains its original distribution at the start of training. This selective injection scheme balances the trade-off between control precision and audio quality.

Dense Affective Projection. A structural challenge in narrative alignment lies in the significant resolution discrepancy between the sparse visual emotion cues and the dense acoustic tokens. Directly injecting these sparse signals can lead to stepped and unnatural transitions in the musical output. To resolve this, we introduce a Temporal Super-resolution Adapter F𝜙. This module first performs linear interpolation to align the discrete affective trajectory E𝑙𝑜𝑐𝑎𝑙 with the target acoustic sequence length. Subsequently, a stack of dilated temporal convolutions[2] is applied to smooth local jitter and expand the receptive field. This ensures that instantaneous emotional shifts are translated into fluid musical evolution. Formally, the dense control signal 𝐶𝑙𝑜𝑐𝑎𝑙 ∈ R𝑇𝑎×𝐷 is derived as:

Optimization Objective. To train the adapter module, we adhere to the native learning objective of the acoustic backbone. Keeping all parameters of the pre-trained autoregressive decoder frozen, we strictly optimize the adapter parameters𝜙 and the gating scalars 𝛾 using the standard autoregressive modeling objective. The process minimizes the Cross-Entropy loss between the predicted probability distribution and the ground-truth acoustic tokens, formulated as:

∑︁𝑇𝑎

1 𝑇𝑎

log𝑝(𝑎𝑡 | 𝑎<𝑡, S𝑔𝑙𝑜𝑏𝑎𝑙,𝐶𝑙𝑜𝑐𝑎𝑙) (7)

L𝑔𝑒𝑛 = −

𝑡=1

During the inference stage, the model utilizes the learned adapter to guide the backbone in generating the sequence of discrete acoustic tokens, which are subsequently reconstructed into the continuous high-fidelity waveform via the EnCodec [8] decoder.

𝐶𝑙𝑜𝑐𝑎𝑙 = F𝜙(Interp(E𝑙𝑜𝑐𝑎𝑙)) (5)

where F𝜙 acts as a learnable mapping function that projects the 2-dimensional Valence-Arousal manifold into the 𝐷-dimensional latent space of the acoustic decoder.

### 3.6 Scalable Long-Form Inference

Token-Level Control Injection. To incorporate the dense affective features into the pre-trained backbone, we employ a residual modulation strategy [47]. Informed by observations in prior studies regarding the hierarchical distribution of information in music generation models [24], we restrict the injection of the control signal

To address the memory constraints inherent in processing minutelevel sequences, we employ an overlapping sliding-window strategy. This approach enables the synthesis of temporally consistent soundtracks for long-form videos by combining global narrative abstraction with local context-dependent continuation.

Global Semantic Reasoning. Prior to sequential processing, we derive the global semantic anchor S𝑔𝑙𝑜𝑏𝑎𝑙 by leveraging the keyframe extraction capability of the VLM. This mechanism compresses the long-form visual content into a sparse sequence of representative frames and allows the model to perform holistic reasoning over the entire narrative arc within a limited context window. By distilling the global context into a unified stylistic description, we ensure that the subsequent music generation maintains a consistent thematic identity throughout the video duration.

Continuous Affective Reasoning. In contrast to the global analysis, the extraction of local affective cues necessitates high temporal density. Consequently, we explicitly eschew temporal compression strategies to prevent the loss of fine-grained narrative dynamics and employ instead an overlapping sliding-window approach to extract the frame-level affective trajectory. The input video is processed in sequential windows that share a defined intersection. By utilizing this temporal overlap, we ensure that the prediction for the current window is contextually aligned with the preceding frames. This continuation approach guarantees that the extracted Valence and Arousal values evolve smoothly across window boundaries and effectively yields a continuous emotional curve despite the segmented processing.

Autoregressive Acoustic Continuation. Similarly, the acoustic synthesis stage operates within this sliding-window framework. We maintain the global semantic anchor S𝑔𝑙𝑜𝑏𝑎𝑙 as a stationary condition to preserve the overarching stylistic foundation while the dynamic narrative progression is governed by the continuous flow of the adapter signals. To ensure musical coherence at the overlap regions, we utilize the final sequence of acoustic tokens generated in the preceding window as the prompt prefix for the current window. This technique prompts the model to logically extend the musical phrase from the previous context. It guarantees seamless rhythmic continuity and acoustic causality throughout the long-form video.

- 4 Experiments

- 4.1 Experimental Setup

Datasets. We utilize publicly available benchmark datasets designed for continuous affective content analysis to train the videoto-emotion and emotion-to-music modules respectively.

For the video-to-emotion prediction task, we employ a continuous movie dataset designed for induced emotion prediction, which provides frame-level Valence-Arousal annotations capturing the emotional responses elicited in viewers during film watching. Unlike facial expression datasets that capture actors’ portrayed emotions, this dataset annotates the holistic atmosphere and narrative tension perceived by audiences—precisely the type of induced affect required for background music generation. This alignment with the viewer’s internal state is essential for generating background music that matches the narrative atmosphere.

For the emotion-to-music generation task, we utilize a music emotion dataset annotated for dynamic affective content, comprising excerpts and full songs with dense per-second valence-arousal labels. This dataset enables learning the mapping from emotional trajectories to musical characteristics, where continuous shifts in

valence and arousal correspond to changes in harmony, tempo, and instrumentation. To ensure the model learns to control emotion independently of specific instrumentation, we aggregate metadata including genre and tags into textual captions. These captions are used as conditioning prompts during fine-tuning to decouple the representation of emotion from genre-specific stylistic conventions.

We apply a unified preprocessing pipeline to both datasets. The source separation model Demucs [9] is utilized to suppress vocal tracks and isolate the instrumental background. We segment the continuous media streams into 30-second clips with a 15-second overlap to preserve temporal context. For the music generation task, we further refine the training set by discarding samples where the silence ratio exceeds 40%. Following these procedures, the effective dataset sizes are approximately 884 minutes for the video dataset and 1351 minutes for the music dataset.

Ethical Considerations. This research is conducted solely for academic and scientific purposes. The proposed method and experimental results are intended to advance the understanding of video-music alignment and affective computing. This work is not integrated into any commercial products or production systems.

Implementation Details. Our framework integrates the pretrained VideoLlama-3 [45] visual backbone and the MusicGenSmall [6] acoustic decoder. The architecture incorporates two trainable modules to facilitate feature alignment, namely the Projector

and the Temporal Adapter.The Projector serves as a semantic interface. It accepts the extracted visual features 𝐹𝑣 ∈ R𝑇×𝐷𝑣 where 𝐷𝑣 denotes the visual embedding dimension. A two-layer MLP with GELU activation maps 𝐹𝑣 to the acoustic feature space R𝑇×𝐷𝑎. We apply a dropout rate of 0.1 during this projection to regularize the mapping process.

The Temporal Adapter models long-term affective dependencies. It utilizes a dilated convolution layer on the sequence dimension to expand the temporal receptive field. This operation maintains the channel dimension 𝐷𝑎 and is followed by LeakyReLU activation and a linear layer to produce the final condition embeddings𝐶 ∈ R𝑇×𝐷𝑎.

We employ a two-stage training strategy to ensure stability. The Projector is first trained for 150 epochs to align the static semantic features from 𝐷𝑣 to 𝐷𝑎. Subsequently, the Adapter is fine-tuned for 50 epochs to capture temporal dynamics.

Baselines. To ensure a comprehensive evaluation, we benchmark NarraScore against a diverse set of five representative approaches spanning different generative paradigms. We first include M2UGEN [28] and video2music [20] as foundational multimodal frameworks that serve as established benchmarks in the field. To assess performance against the current state-of-the-art, we compare our method with VidMuse and GVMGEN, both of which represent the latest advancements in synchronized video-music synthesis. Furthermore, we construct a strong two-stage pipeline baseline named Caption2Music to evaluate the efficacy of disjoint modality processing. This baseline explicitly utilizes VideoLlama3-2B to generate detailed visual captions, which subsequently prompt MusicGen for audio synthesis. Including this cascaded model allows us to rigorously validate whether our proposed hierarchical injection strategy outperforms a naive combination of state-of-the-art vision and audio models.

Method FAD (↓) FD (↓) KLD (↓) IB(↑) GT 0 0 0 0.241 VidMuse [37] 2.459 29.946 0.734 0.202 Video2Music [20] 14.954 115.008 1.269 0.100 GVMGen [49] 2.362 41.466 0.350 0.213 M2UGEN [28] 9.647 74.625 0.953 0.182 NarraScore 1.923 36.411 0.320 0.219

Table 1: Quantitative comparison with state-of-the-art methods on objective metrics. ↑ indicates higher is better, ↓ indicates lower is better.Bold indicates the best performance, and underlined indicates the second best.

### 4.2 Objective Evaluation

To quantitatively verify the efficacy of the proposed model, we conduct a comprehensive evaluation focusing on generation quality, fidelity, and diversity. Specifically, we employ Fréchet Audio Distance (FAD) [21], Fréchet Distance (FD) [17], and Kullback-Leibler Divergence (KLD) to measure the distributional discrepancies between the generated audio and the ground truth. To assess the variation in the synthesized samples, we utilize Density and Coverage metrics. Regarding cross-modal semantic consistency, we align with established baselines by incorporating the ImageBind score. However, it is pertinent to note that while ImageBind [15] provides a metric for static semantic alignment, it may not fully capture the temporal progression and the intricate narrative flow inherent in video content. The quantitative comparisons are detailed in Table 1.

### 4.3 Subjective Evaluation

Recognizing human perception as the definitive benchmark for artistic generation, we conducted a user study involving 10 participants to evaluate the generated soundtracks. The assessment focused on five key dimensions: Emotional Dynamic Consistency (EDC),GlobalStyleMatching(GSM),Long-term Coherence (LTC),Music Quality (MQ),Overall Preference (OP).

As presented in Table 2, our method demonstrates superior performance across all metrics, establishing a significant lead over baseline approaches, particularly in Emotional Consistency. This advantage substantiates the model’s capability to capture and articulate evolving affective content within the video. Conversely, the pipeline-based baseline (Caption2Music) exhibits marked deficiencies in visual–audio correspondence. This result suggests that exclusive reliance on textual captions creates an information bottleneck, filtering out the critical temporal and dynamic cues essential for precise audiovisual alignment.

The comparative analysis detailed in Tables 2 and 3 reveals that performance disparities are intrinsically linked to the temporal modeling capabilities and conditioning mechanisms of each method. While baselines remain competitive in short- and mid-length scenarios due to limited temporal horizons, user feedback indicates a significant degradation in output quality when scaling to long-form videos. In these extended contexts, baseline methods frequently fail to sustain a stable musical trajectory, manifesting as narrative drift, inconsistent motifs, or disjointed segmentation. In contrast, our

Method EDC GSM LTC MQ OP VidMuse [37] 2.29 2.49 2.49 2.13 2.18 Video2Music [20] 1.48 1.56 2.41 3.39 1.88 GVMGen [49] 1.65 1.69 1.41 1.64 1.34 Caption2Music 2.66 2.77 2.88 3.18 2.82 Ours 2.86 3.02 3.15 3.41 3.06

#### Table 2: Subjective Evaluation of Video-to-Music Generation on Long-form Videos.Bold indicates the best performance, and underlined indicates the second best.

Method EDC GSM LTC MQ OVR

VidMuse [37] 3.09 3.27 3.03 3.00 3.02 Video2Music [20] 1.66 1.68 2.38 3.50 1.84 GVMGen [49] 2.68 2.75 2.80 2.87 2.64 Caption2Music 2.83 3.01 3.04 3.21 2.96 Ours 3.11 3.36 3.24 3.52 3.32

#### Table 3: Subjective Evaluation of Video-to-Music Generation on Short- and Mid-length Videos.Bold indicates the best performance, and underlined indicates the second best.

approach achieves the highest overall preference in both settings, with a widening margin in the long-form evaluation. This validates the effectiveness of our design in handling extended temporal dependencies, preserving a consistent global musical identity while adaptively responding to local visual cues.

Further analysis elucidates the specific limitations inherent in competing paradigms. The pipeline baseline, Caption2Music, highlights the limitations of text-only control, where coarse captions may capture the general mood but lack the temporal granularity required for fine-grained emotional transitions. Similarly, MIDI-based baselines such as Video2Music expose a dichotomy between acoustic fidelity and narrative alignment; despite receiving high ratings for audio texture, these tracks are frequently characterized by users as generic and weakly coupled to the visual storyline. Consequently, while short-term settings naturally narrow the performance gap by reducing the burden of long-range consistency, the sustained top ranking of our method across all metrics underscores that structural and affective coherence remain the defining factors in successful narrative music generation.

### 4.4 Ablation Study

To investigate the contribution of individual components and assess the framework’s adaptability across different reasoning backbones, we conduct a comprehensive ablation study, as summarized in Table 4. The experimental results reveal distinct behaviors of the Holistic Musical Conceptualization (HMC) module across different scales of vision-language backbones. When employing lightweight VLMs as the captioner, the absence of HMC leads to noticeable performance degradation, as the model tends to produce literal descriptions of visual scenes without establishing meaningful connections to musical elements. However, an interesting phenomenon emerges when switching to powerful large-scale VLMs: the HMC module appears to constrain rather than enhance the model’s capabilities.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

#### Figure 4: Visualization of the generated spectrograms and the corresponding narrative emotion curves.

This suggests that advanced VLMs possess inherent cross-modal understanding abilities that allow them to directly infer appropriate musical characteristics from visual content without explicit highlevel guidance. Regarding the adaptive in-attention mechanism, the results demonstrate that the 75% injection ratio employed in NarraScore represents an optimal balance. Both reducing and increasing this proportion adversely affect performance, indicating that excessive narrative guidance may overwhelm the acoustic modeling capacity of the diffusion backbone while insufficient injection fails to provide adequate semantic alignment. Furthermore, the consistent improvements brought by the Narrative-Aware Affective Reasoning (NAR) mechanism across all backbone configurations validate its effectiveness as a model-agnostic component. Even when equipped with state-of-the-art VLMs, the incorporation of NAR continues to enhance music quality by bridging the affective gap between visual narratives and auditory expressions, underscoring the importance of explicit emotional reasoning in video-to-music generation tasks.

density, it is plagued by a high concentration of stochastic noise in the high-frequency regions above 8000Hz. This is visualized as disordered pixel-like artifacts, which suggests that vidmuse is confined to literal object-level associations instead of synthesizing a cohesive and narratively-driven soundtrack.

In contrast, our model demonstrates a superior spectral hierarchy that balances harmonic stability with rhythmic precision. The presence of clear fundamental frequencies alongside vertical onset markers indicates that our method successfully synchronizes discrete rhythmic events with the narrative arc. By capturing both the instantaneous motion cues and the global semantic flow, our model generates music that is not only acoustically clear but also contextually and emotionally resonant with the video content.

Setting FAD (↓) FD (↓) KLD (↓) IB (↑) Component Analysis

NarraScore 1.923 36.411 0.320 0.219 w/o HMC 2.235 36.069 0.388 0.203 w/o NAR 3.009 41.146 0.545 0.202

### 4.5 Qualitative Analysis

To qualitatively evaluate the acoustic characteristics and narrative alignment of the generated music, we perform a detailed comparative analysis of Mel-spectrograms produced by various models under identical video guidance. As illustrated in Figure 4, the spectral architectures reveal fundamental differences in how each framework internalizes temporal dynamics and semantic logic. Specifically, the Mel-spectrum of caption2music is dominated by excessively smooth horizontal energy bands with a notable absence of vertical transients. This spectral stagnation indicates a lack of rhythmic pulses and dynamic fluctuations, which results in audio that fails to reflect the temporal energy shifts inherent in the video. In contrast, gvmgen exhibits significant discontinuities in its spectral manifold characterized by fragmented energy distributions and abrupt temporal breaks. Such erratic spectral patterns suggest a failure in maintaining long-term acoustic coherence and a misalignment with the narrative progression of the visual input. The results from v2m further highlight these challenges because its spectrum appears overly monotonic and mechanical. The absence of structured harmonic variations and distinct beat markers implies a deficiency in narrative expression where the model produces a flat auditory output that remains unresponsive to visual climaxes. Furthermore, while vidmuse generates content with high spectral

In-attention Analysis

50% Blocks 2.021 37.282 0.353 0.196 100% Blocks 1.964 36.503 0.318 0.200

Backbone Analysis

Gemini2.5pro [5] 2.322 33.752 0.376 0.226 Gemini (w/o HMC) 1.906 31.299 0.320 0.223 Gemini (w/o NAR) 2.430 32.568 0.403 0.214 Gemini (cap-only) 2.002 32.239 0.324 0.203

Table 4: Ablation study on key components and different LLM backbones. NAR denotes Narrative-Aware Affective Reasoning.Bold indicates the best performance, and underlined indicates the second best.

### 5 Conclusion

We present NarraScore, pioneering the first direct emotional control pathway from video narratives to musical dynamics. Our findings demonstrate that fine-tuning small mLLMs with limited labeled

data enables robust continuous temporal regression. Simultaneously, we confirm that a lightweight adapter is sufficient to steer the complex acoustic backbone. By reconciling global style with local tension, NarraScore achieves state-of-the-art coherence, establishing a strong baseline for autonomous soundtrack generation.

### 6 Limitations

Current limitations stem from the limited temporal granularity of the affective control, which precludes frame-perfect synchronization with rapid visual events. Additionally, the cascaded design risks error propagation from upstream affective reasoning. Future work will focus on end-to-end joint optimization to mitigate these dependencies and explore knowledge distillation to reduce the computational latency of the visual backbone.

### References

- [1] Andrea Agostinelli, Timo I Denk, Zalán Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, et al. 2023. Musiclm: Generating music from text. arXiv preprint arXiv:2301.11325

(2023).

- [2] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. 2018. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271 [cs.LG] https://arxiv.org/abs/1803.01271
- [3] Yoann Baveye, Emmanuel Dellandrea, Christel Chamaret, and Liming Chen. 2015. LIRIS-ACCEDE: A video database for affective content analysis. IEEE Transactions on Affective Computing 6, 1 (2015), 43–55.
- [4] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. 2024. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476 (2024).
- [5] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025).
- [6] Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, and Alexandre Défossez. 2023. Simple and controllable music generation. Advances in Neural Information Processing Systems 36 (2023), 47704–47720.
- [7] Johanna N Dasovich-Wilson, Marc Thompson, and Suvi Saarikallio. 2022. Exploring music video experiences and their influence on music perception. Music & Science 5 (2022), 20592043221117651.
- [8] Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. 2022. High fidelity neural audio compression. arXiv preprint arXiv:2210.13438 (2022).
- [9] Alexandre Défossez, Nicolas Usunier, Léon Bottou, and Francis Bach. 2019. Music source separation in the waveform domain. arXiv preprint arXiv:1911.13254

(2019).

- [10] Shangzhe Di, Zeren Jiang, Si Liu, Zhaokai Wang, Leyan Zhu, Zexin He, Hongming Liu, and Shuicheng Yan. 2021. Video background music generation with controllable music transformer. In Proceedings of the 29th ACM International Conference on Multimedia. 2037–2045.
- [11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv:2010.11929 [cs.CV] https://arxiv.org/abs/2010.11929
- [12] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. 2019. Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision. 6202–6211.
- [13] Niki Maria Foteinopoulou and Ioannis Patras. 2024. Emoclip: A vision-language method for zero-shot video facial expression recognition. In 2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG). IEEE, 1–10.
- [14] Chuang Gan, Deng Huang, Peihao Chen, Joshua B Tenenbaum, and Antonio Torralba. 2020. Foley music: Learning to generate music from videos. In European Conference on Computer Vision. Springer, 758–775.
- [15] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. 2023. ImageBind: One Embedding Space To Bind Them All. arXiv:2305.05665 [cs.CV] https://arxiv.org/abs/2305.05665
- [16] Google DeepMind. 2025. Veo 3 Tech Report. https://storage.googleapis.com/ deepmind-media/veo/Veo-3-Tech-Report.pdf.
- [17] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30

(2017).

- [18] Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Noam Shazeer, Ian Simon, Curtis Hawthorne, Andrew M Dai, Matthew D Hoffman, Monica Dinculescu, and Douglas Eck. 2018. Music transformer. arXiv preprint arXiv:1809.04281 (2018).
- [19] Shulei Ji, Songruoyao Wu, Zihao Wang, Shuyu Li, and Kejun Zhang. 2025. A Comprehensive Survey on Generative AI for Video-to-Music Generation. arXiv preprint arXiv:2502.12489 (2025).
- [20] Jaeyong Kang, Soujanya Poria, and Dorien Herremans. 2024. Video2music: Suitable music generation from videos using an affective multimodal transformer model. Expert Systems with Applications 249 (2024), 123640.
- [21] Kevin Kilgour, Mauricio Zuluaga, Dominik Roblek, and Matthew Sharifi. 2019. Fréchet Audio Distance: A Metric for Evaluating Music Enhancement Algorithms. arXiv:1812.08466 [eess.AS] https://arxiv.org/abs/1812.08466
- [22] Dimitrios Kollias and Stefanos Zafeiriou. 2018. Aff-wild2: Extending the aff-wild database for affect recognition. arXiv preprint arXiv:1811.07770 (2018).
- [23] Jean Kossaifi, Georgios Tzimiropoulos, Sinisa Todorovic, and Maja Pantic. 2017. AFEW-VA database for valence and arousal estimation in-the-wild. Image and Vision Computing 65 (2017), 23–36.

- [24] Yun-Han Lan, Wen-Yi Hsiao, Hao-Chung Cheng, and Yi-Hsuan Yang. 2024. Musicongen: Rhythm and chord control for transformer-based text-to-music generation. arXiv preprint arXiv:2407.15060 (2024).
- [25] Ruiqi Li, Siqi Zheng, Xize Cheng, Ziang Zhang, Shengpeng Ji, and Zhou Zhao.

2024. Muvi: Video-to-music generation with semantic alignment and rhythmic synchronization. arXiv preprint arXiv:2410.12957 (2024).

- [26] Yan-Bo Lin, Yu Tian, Linjie Yang, Gedas Bertasius, and Heng Wang. 2025. Vmas: Video-to-music generation via semantic alignment in web music videos. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 1155–1165.
- [27] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, et al. 2025. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918 (2025).
- [28] Shansong Liu, Atin Sakkeer Hussain, Qilong Wu, Chenshuo Sun, and Ying Shan.

2023. M2UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models. arXiv preprint arXiv:2311.11255 (2023).

- [29] Lin Ma. 2022. Research on the effect of different types of short music videos on viewers’ psychological emotions. Frontiers in public health 10 (2022), 992200.
- [30] Barbara Millet, Juan Chattah, and Soyeon Ahn. 2021. Soundtrack design: The impact of music on visual attention and affective responses. Applied ergonomics 93 (2021), 103301.
- [31] OpenAI. 2024. Video generation models as world simulators. https://openai.com/ index/video-generation-models-as-world-simulators/.
- [32] Zhihang Ren, Jefferson Ortega, Yifan Wang, Zhimin Chen, Yunhui Guo, Stella X Yu, and David Whitney. 2024. Veatic: Video-based emotion and affect tracking in context dataset. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 4467–4477.
- [33] Andrew Rouditchenko, Angie Boggust, David Harwath, Brian Chen, Dhiraj Joshi, Samuel Thomas, Kartik Audhkhasi, Hilde Kuehne, Rameswar Panda, Rogerio Feris, et al. 2020. Avlnet: Learning audio-visual language representations from instructional videos. arXiv preprint arXiv:2006.09199 (2020).
- [34] James A Russell. 1980. A circumplex model of affect. Journal of personality and social psychology 39, 6 (1980), 1161.
- [35] Kun Su, Judith Yue Li, Qingqing Huang, Dima Kuzmin, Joonseok Lee, Chris Donahue, Fei Sha, Aren Jansen, Yu Wang, Mauro Verzetti, et al. 2024. V2meow: Meowing to the visual beat via video-to-music generation. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 4952–4960.
- [36] Serkan Sulun, Paula Viana, and Matthew EP Davies. 2025. Video Soundtrack Generation by Aligning Emotions and Temporal Boundaries. arXiv preprint arXiv:2502.10154 (2025).
- [37] Zeyue Tian, Zhaoyang Liu, Ruibin Yuan, Jiahao Pan, Qifeng Liu, Xu Tan, Qifeng Chen, Wei Xue, and Yike Guo. 2025. Vidmuse: A simple video-to-music generation framework with long-short-term modeling. In Proceedings of the Computer Vision and Pattern Recognition Conference. 18782–18793.
- [38] Xinyi Tong, Yiran Zhu, Jishang Chen, Chunru Zhan, Tianle Wang, Sirui Zhang, Nian Liu, Tiezheng Ge, Duo Xu, Xin Jin, et al. 2025. Video Echoed in Music: Semantic, Temporal, and Rhythmic Alignment for Video-to-Music Generation. arXiv preprint arXiv:2511.09585 (2025).
- [39] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. 2022. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems 35 (2022), 10078–10093.
- [40] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. 2023. Videomae v2: Scaling video masked autoencoders with dual masking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 14549–14560.
- [41] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. 2025. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328 (2025).
- [42] Junxian Wu, Weitao You, Heda Zuo, Dengming Zhang, Pei Chen, and Lingyun Sun. 2025. Controllable video-to-music generation with multiple time-varying conditions. In Proceedings of the 33rd ACM International Conference on Multimedia. 10427–10436.
- [43] Zhifeng Xie, Qile He, Youjia Zhu, Qiwei He, and Mengtian Li. 2025. FilmComposer: LLM-Driven Music Production for Silent Film Clips. In Proceedings of the Computer Vision and Pattern Recognition Conference. 13519–13528.
- [44] Jiashuo Yu, Yao Yao, Boyu Chen, and Alex Wang. [n.d.]. JenBridge: Adaptive Long-Form Video Soundtracking across Scene Transition. ([n.d.]).
- [45] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. 2025. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106 (2025).
- [46] Hang Zhang, Xin Li, and Lidong Bing. 2023. Video-llama: An instructiontuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858 (2023).
- [47] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision. 3836–3847.

- [48] Sitao Zhang, Yimu Pan, and James Z Wang. 2023. Learning emotion representations from verbal and nonverbal communication. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18993–19004.
- [49] Heda Zuo, Weitao You, Junxian Wu, Shihong Ren, Pei Chen, Mingxu Zhou, Yujia Lu, and Lingyun Sun. 2025. Gvmgen: A general video-to-music generation model with hierarchical attentions. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 23099–23107.

