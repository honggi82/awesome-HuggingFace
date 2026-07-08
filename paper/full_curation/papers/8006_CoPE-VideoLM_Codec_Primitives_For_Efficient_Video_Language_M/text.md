## CoPE-VideoLM: Leveraging Codec Primitives For Efficient Video Language Modeling

##### Sayan Deb Sarkar1,2◦, Rémi Pautrat1, Ondrej Miksik1, Marc Pollefeys1,3, Iro Armeni2, Mahdi Rad1∗, Mihai Dusmanu1∗ 1Microsoft Spatial AI Lab, 2Stanford University, 3ETH Zurich

◦Part of work done at Microsoft ∗Equal Supervision

Video Language Models (VideoLMs) enable AI systems to understand temporal dynamics in videos. To fit within the maximum context window constraint, current methods use keyframe sampling which often misses both macro-level events and micro-level details due to the sparse temporal coverage. Furthermore, processing full images and their tokens for each frame incurs substantial computational overhead. We address these limitations by leveraging video codec primitives (specifically motion vectors and residuals) which natively encode video redundancy and sparsity without requiring expensive full-image encoding for most frames. To this end, we introduce lightweight transformer-based encoders that aggregate codec primitives and align their representations with image encoder embeddings through a pre-training strategy that accelerates convergence during end-to-end fine-tuning. Our approach, CoPE-VideoLM, reduces the time-to-first-token by up to 86% and token usage by up to 93% compared to standard VideoLMs. Moreover, by varying the keyframe and codec primitive densities we maintain or exceed performance on 14 diverse video understanding benchmarks spanning general question answering, temporal and motion reasoning, long-form understanding, and spatial scene understanding.

# arXiv:2602.13191v2[cs.CV]30Mar2026

Date: March 30, 2026 Correspondence: Sayan Deb Sarkar at sdsarkar@stanford.edu

Project: https://microsoft.github.io/CoPE

On PerceptionTest

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

-80% tokens

CoPE-VideoLM Standard

VideoLM

[Figure 6]

+2.4% acc

Decoded RGB Video Frames

Existing Methods

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]| |
|---|---|

| |[Figure 10]|
|---|---|

| |[Figure 11]|
|---|---|

Original Codec Video Representation

Ours

Figure 1 CoPE-VideoLM is a codec-aware tokenization framework for Video Language Models that replaces dense RGB frame encoding with lightweight structured representations derived from codec primitives. Instead of treating every frame as a full image, the model processes only sparse I-frames through a vision encoder, while P-frames are converted into compact tokens using their motion vectors and residuals. By leveraging the inherent sparsity and structure of standard video codecs, CoPE-VideoLM avoids redundant RGB processing, reduces visual token usage by up to 93%, and cuts time-to-first-token by up to 86%, compared to standard dense VideoLMs.

### 1 Introduction

Video Language Models (VideoLMs) represent a major advancement in multi-modal AI [1–5], enhancing Vision Language Models (VLMs) with temporal reasoning that allows them to understand how visual narratives, objects, actions, and relationships evolve across video sequences. This enables a wide range of downstream applications, from more natural human-computer interaction [6] through video question-answering to robotics [7, 8], where agents must understand sequential actions [9, 10]. Beyond these applications, VideoLMs represent a step toward AI systems that process visual information not as disconnected snapshots but as continuous, temporally coherent experiences. As video continues to dominate digital content, models that can “watch” and understand video at scale [11–13] become essential for the next generation of AI applications.

However, realizing this vision remains challenging: VideoLMs have a maximum context window limiting the amount of information that can be provided as input. This is not only a training artifact, but also related to hardware constraints, as larger context windows require linearly more memory and quadratically more compute [14]. To fit in the context window, existing VideoLMs select a subset of video frames as keyframes either through hand-crafted heuristics (e.g., uniform temporal sampling [15]) or learned methods (e.g., flexible frame selection [16] or token compression [17]). In this setup, proprietary models with extended context lengths of up to 1 million tokens can process one hour of video at 1 FPS [15]. Open-source models [4] have a much smaller budget and sample a fixed number of frames (e.g., 64) regardless of video length. This is a severe limitation, as the information content naturally scales with video duration.

Moreover, any keyframe sampling approach fundamentally limits VideoLMs’ understanding by providing only sparse temporal coverage: this can miss both macro-level events crucial for high-level comprehension and micro-level details necessary for recognizing fine-grained actions [18–20]. Compounding this, there is generally high redundancy between consecutive frames (even when downsampled to 1 FPS); therefore, using the same token budget for each keyframe is suboptimal. Furthermore, treating each keyframe as a full image also significantly increases the prefill time [21], delaying the time-to-first token (TTFT) of the VideoLM. Low TTFT is critical for user experience and essential for robotics applications requiring real-time responsiveness.

A natural source of structure that addresses both redundancy and sparsity is already present in video processing pipelines: video codecs, a decades-old field designed precisely to solve these problems [22, 23]. Rather than encoding every frame as a full image, codecs encode what moves between frames as motion vectors and the intensity changes as residuals, preserving temporal structure while minimizing redundancy. In a typical streaming setup, codecs select keyframes (I-frames) every 5–10 seconds, either at fixed intervals or when large scene changes are detected, encoding only the changes for all remaining frames (P-frames).

Prior works have explored codec-domain signals, but with important limitations: CoViAR [24] and TeamNet [25] trained separate CNNs on motion vectors and residuals, ignoring cross-modal dependencies for action recognition. Video-LaVIT [26] discretizes motion vectors but discards residuals. EMA [27] aggregates the I-frame and motion vectors from a group of pictures (GOP) into a fixed-length summary, also discarding residuals and losing temporal ordering. Critically, no existing method preserves both dynamics and appearance in a variable-length, temporally ordered representation.

We propose to directly encode motion vectors and residuals (i.e., codec primitives) as temporally aligned tokens compatible with image features, and arrange them in temporal order alongside keyframe tokens to form a codec-aware token sequence during inference. An overview of our approach is shown in Fig. 1. This provides two major advantages: first, we avoid the costly full image encoding for most frames,

and second, we can use far fewer tokens given the sparse nature of these primitives, thus reducing TTFT by a significant margin. Optionally, the motion vectors and residuals of several subsequent frames can be grouped together to strike a trade-off between fine-grained representation and total number of tokens. While we focus on VideoLMs, the methodology is in principle applicable to other tasks such as video retrieval or action recognition. Our formulation addresses all three gaps identified above: we preserve residuals, maintain temporal ordering through interleaved I/P-frame tokens, and allow the token budget to scale with information content.

To this end, we use transformer-based encoders to aggregate the information of all motion vectors and residuals from a given GOP. These encoders are first pre-trained to adapt them to the space of the image encoder and then integrated with a VideoLM for end-to-end fine-tuning. Our model matches the performance of current open-weight VideoLMs and even surpasses them on several benchmarks despite using substantially fewer tokens. We extensively validate our approach across 14 benchmarks, demonstrating consistent gains across general video QA, temporal and motion reasoning, long-form reasoning, and spatial scene understanding tasks. Our main contributions are as follows:

- • We propose to encode videos for VideoLMs by leveraging codec primitives, preserving motion and appearance in temporal order. Codec primitives allow us to skip redundant RGB information, reducing the TTFT by up to 86% and token usage by up to 93%.
- • We introduce a lightweight dual-branch architecture for encoding codec primitives, achieving substantially higher compression rates and lower token counts than traditional image-based encoders.
- • We propose a pre-training framework for codec-primitive encoders that aligns their representation space with that of image encoders, enabling faster training and quicker convergence when integrated with the VideoLM.

### 2 Related Work

Video Language Models. Recent advances in Multimodal Large Language Models (MLLMs) [28–39] have extended image-based architectures into the video domain, giving rise to VideoLMs capable of temporal reasoning over dynamic visual content. MLLMs typically comprise a vision encoder (e.g., CLIP [40] or SigLIP [41]), a modality alignment mechanism also known as adapter (e.g., linear projection, QFormer [42, 43], or gated cross-attention [44]), and an LLM backbone (e.g., LLaMA [45], Vicuna [46], or Qwen [13, 47, 48]) for multimodal decoding. Early VideoLMs such as Video-LLaMA [49] and VideoChat2 [1] were limited by short context and redundant tokenization. Subsequent models improve efficiency through extended context windows [5], token pooling or merging [50–53], timestamp-aware encoding [54], and large-scale instruction tuning [4]. Closed-source systems like Gemini [12, 15, 55], GPT [11, 55] and Claude [56] demonstrate impressive fine-grained and long-context understanding but depend on proprietary data and undisclosed architectures. Despite these advances, open-source VideoLMs still process videos as dense RGB frame collections, overlooking the structured redundancy inherent in standard video codecs. Leveraging codec primitives like motion vectors and residuals, our approach directly encodes temporal dynamics, supporting efficient long-context understanding while retaining fine-grained detail.

Token Compression. Compression techniques reduce the number of visual tokens fed into VideoLMs by removing redundancy while preserving semantic fidelity. Existing methods can be grouped into heuristic and learnable approaches. Heuristic methods apply uniform downsampling [57], spatial or temporal pooling [4, 58, 59], or similarity-guided merging [60–62]. Learnable modules such

as Q-Former [47, 63, 64], Perceiver Resampler [65], and memory-based mechanisms [61, 66, 67] generate compact latent representations before passing them to the LLM. Attention-based methods [68–

- 71] exploit the observation that visual tokens receive diminishing attention in deeper layers [68,
- 72], pruning or modulating token ratios across layers to remove redundancy. Temporal pooling approaches [50, 58, 73, 74] exploit inter-frame redundancy by downsampling at the frame level, while DyCoke [75] and LLaVA-Scissor [17] further leverage spatio-temporal structure for compression (see Supp. for a direct comparative study; codec-native sparsity provides a stronger starting point than post-hoc pruning). More adaptive approaches such as AdaReTake [76] and FlexSelect [77] dynamically allocate compression budgets across layers or leverage cross-modal attention to filter tokens without retraining. However, these methods rely on dense RGB frame encodings and neglect intrinsic temporal redundancy. In contrast, our native codec representation inherently encodes only meaningful temporal changes rather than removing information post hoc.

Compressed Video Representation. There has been growing interest in exploiting motion vectors and residuals directly from compressed video streams for visual understanding, particularly in action recognition, thus bypassing the costly full-frame processing. Early works such as CoViAR [24] and TEAMNet [25] trained separate CNNs on I- and P-frame signals but ignored inter-modal dependencies and temporal ordering, requiring costly multi-clip inference. Later methods extended compressed-domain learning to 3D CNNs [78], optical-flow distillation [79–81], and transformer-based self-attention [82], improving accuracy but still incurring high inference costs or requiring decoded RGB frames during training. CompressedVideoMAE [83] demonstrates that masked autoencoding on motion vectors and residuals alone can match raw-video pretraining at far less compute. More recently, codec-based representations have been explored in VideoLMs. Video-LaVIT [26] discretizes motion vectors into language-like tokens, and EMA [27] aggregates I-frames and motion vectors into a fixed-length GOP summary, similar to Video-VAE [84]. These approaches either discard residuals or collapse temporal ordering; CoPE-VideoLM addresses both by constructing a variable-length, temporally ordered token sequence that preserves fine-grained motion and appearance signals.

### 3 Method

#### 3.1 Preliminaries

Modern video codecs such as MPEG-4, H.264, and HEVC [85, 86] achieve high compression ratios by exploiting temporal redundancy across consecutive frames. Let a video V be a sequence of frames (𝐹(1), . . . , 𝐹(𝑇)). Each of these frames can be an I-frame (intra-coded), a P-frame (predictive), or optionally a B-frame (bi-directional predictive). Consecutive frames are organized in a Group of Pictures (GOP) structure as illustrated in Fig. 2.

I-frames. An I-frame, 𝐼(𝑡), is an RGB image encoded independently without the use of preceding or subsequent frames. It is used as a reference point of the Group of Pictures and provides a full visual representation.

P-frames. A P-frame 𝑃(𝑡) contains only the changes from the previous frame, be it a reference frame 𝐼(𝑡−1) or a preceding P-frame 𝑃(𝑡−1). The difference is defined by two components:

- • Motion vectors 𝜏(𝑡), which describe block-wise displacements from the reference to the target frame, resembling coarse optical flow.
- • Residuals 𝛿(𝑡), which capture block-wise pixel corrections that remain after motion compensation.

[Figure 12]

[Figure 13]

Text Output Answer: The ball almost collides with the pencil.

[Figure 14]

[Figure 15]

Large Language Model

N = 8 tokens M = 196 tokens N = 8 tokens Language tokens

M = 196 tokens N = 8 tokens

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Tokenizer

Vision Encoder 𝚫-Encoder 𝚫-Encoder

Vision Encoder

𝚫-Encoder

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

I-frame P-frame P-frame I-frame P-frame

|[Figure 32]|
|---|

|[Figure 33]|
|---|

| |[Figure 34]|
|---|---|

|[Figure 35]| |
|---|---|

| |[Figure 36]|
|---|---|

[Figure 37]

[Figure 38]

###### Language

Instruction Question: With which object does the ball almost collide?

Motion Vector Residual Motion Vector Residual Motion Vector Residual

Group of Pictures (GOP 1) Group of Pictures (GOP 2)

Codec Representation

xG GOPs

Figure 2 Overview of our pipeline. Given a video in its raw codec representation, our framework leverages the GOP structure for efficient, codec-aware tokenization. I-frames are processed by a standard frozen vision encoder (𝜙RGB) to produce dense RGB tokens. P-frames, however, bypass full RGB decoding. Their raw components, motion vectors and residuals, are instead fed into our lightweight Δ-Encoder (𝜙Δ) to generate a small set of highly compact Δ-tokens. The final token stream, an interleaved sequence of I-frame tokens and Δ-tokens, is consumed by the LLM, enabling dense temporal coverage at a fraction of the standard token count and runtime.

The recurrence [24, 83] for reconstructing a P-frame is:

ˆ𝐼𝑖(𝑡) = ˆ𝐼(𝑡−1) 𝑖−𝜏𝑖(𝑡)

+ 𝛿𝑖(𝑡) , (1)

where 𝑖 is the pixel coordinate index. ˆ𝐼(𝑡−1) is the reference frame at time 𝑡−1: it equals the raw I-frame 𝐼(𝑡−1) when 𝐹(𝑡−1) is intra-coded, and the reconstructed frame from earlier P-frames otherwise. Hence, P-frames contain only incremental temporal information, and are therefore smaller in file size than I-frames.

B-frames. A B-frame leverages both preceding and subsequent frames to encode its differences. While this bidirectional prediction achieves the highest compression efficiency, it increases decoding complexity: B-frames must wait for future I/P-frames, creating a mismatch between decode and display order. Consequently, they are less suited for streaming or real-time use. We therefore focus on P-frames, which depend only on past references and align naturally with the causal processing required by VideoLMs.

Group of Pictures (GOP). A GOP is a cycle structure that comprises one I-frame together with any mixture of P- and/or B-frames, e.g., 𝐼 𝐵 𝑃 𝑃 𝐵 𝑃 etc. The GOP structure and length control the trade-offs between the compression efficiency, quality, and the capability for random access. Typical applications use varying configurations: all three frame types are used by the H.264 codec, while MPEG-4 uses the I- and P-frames (see Supp. for a detailed explanation).

Implication for VideoLMs. Despite this rich structure, current VideoLMs discard codec information and fully decode and tokenize dense RGB frames, ignoring the inherent sparsity of P- and B-frames. This results in unnecessary computation and inflated token counts, as shown in Sec. 4.2.

#### 3.2 CoPE-VideoLM

Our method explicitly leverages the GOP structure and introduces a codec-aware tokenization framework that seamlessly integrates with VideoLMs. Instead of unnecessarily encoding each frame as RGB patches [4] or compressing an entire GOP into a fixed-length summary [27], we retain I-frames as full RGB tokens and encode P-frames into lightweight and compact Δ-tokens (delta for difference) obtained from motion vectors and residuals. This preserves temporal ordering and appearance information without exceeding token budgets. An illustration of this process is presented in Fig. 2.

Given a video V = (𝐹(1), . . . , 𝐹(𝑇)), each frame 𝐹(𝑡) is represented as:

𝐼(𝑡), if 𝐹(𝑡) is an I-frame, 𝑃(𝑡) = (𝜏(𝑡), 𝛿(𝑡)), if 𝐹(𝑡) is a P-frame,

(2)

𝐹(𝑡) =

where 𝜏(𝑡) is a set of block-wise motion vectors (usually one per up to 16 × 16 macroblock), and 𝛿(𝑡) is a set of block-wise residuals. For our downstream processing, we represent these as tensors: 𝜏(𝑡) ∈ ℤ𝐻×𝑊×2 is the sparse tensor constructed from the block-wise motion vectors and 𝛿(𝑡) ∈ ℝ𝐻×𝑊×𝐶

is the sparse tensor of residuals. I-frame processing. I-frames 𝐼(𝑡) are passed through a frozen vision encoder 𝜙RGB, producing 𝑀 dense tokens:

𝑋𝐼(𝑡) = 𝜙RGB(𝐼(𝑡)) ∈ ℝ𝑀×𝑑. (3)

P-frame processing. Each P-frame 𝑃(𝑡) is mapped into a much more compact representation consisting of 𝑁 ≪ 𝑀 tokens by the Δ-Encoder 𝜙Δ:

𝑋𝑃(𝑡) = 𝜙Δ(𝜏(𝑡), 𝛿(𝑡)) ∈ ℝ𝑁×𝑑. (4)

P-frame fusion. In the setup described so far, processing all frames at the native frame rate is mandatory as codec primitives are defined relative to previous frames and skipping frames invalidates these dependencies. For example, let us consider a 30 FPS video with a GOP size of 240 frames (8 seconds). Our technique yields 𝑀+239𝑁 tokens per GOP, compared to 240𝑀 if all frames were encoded

- as RGB images. Fine-grained action recognition may require this exhaustive temporal coverage, but most video understanding tasks can often be done with sparser coverage.

Rather than processing all frames at the native frame rate, we can fuse 𝑠 consecutive P-frames, encoding their combined changes relative to frame 𝐹(𝑡−𝑠) (rather than the immediately preceding frame). The maximum number of P-frames that can be fused is bounded by the GOP size. In the running example, using 𝑠 = 30 P-frames (1 FPS) for fusion reduces the per-GOP token count to 𝑀 + 7𝑁, which is much smaller than 𝑀 + 239𝑁 for full P-frame modeling and 8𝑀 for RGB encoding at 1 FPS. This fusion offers a codec-native way to trade temporal resolution for efficiency and can be tuned to match available compute and task requirements.

For clarity, we treat the temporal index (𝑡) as already incorporating the P-frame fusion, so 𝐹(𝑡) always depends on 𝐹(𝑡−1), though (𝑡) may no longer correspond to the raw frame indices.

Δ-Encoder. The Δ-Encoder 𝜙Δ (Fig. 3) is designed to process the motion vectors 𝜏(𝑡) and residuals 𝛿(𝑡) through two specialized branches. The motion vectors 𝜏(𝑡) are processed via a multi-layer MLP to extract local features over a grid of size 𝐻𝐺 × 𝑊𝐺 yielding features in ℝ(𝐻

𝐺𝑊𝐺)×𝑑. These features are

then compressed via a motion transformer 𝜃motion with a set of 𝐾𝜏 learnable query tokens that can attend to all 𝐻𝐺𝑊𝐺 input tokens and aggregate their information as:

𝜏tok(𝑡) = 𝜃motion(MLP(𝜏(𝑡))) . (5) Only these 𝐾𝜏 compressed motion tokens 𝜏tok(𝑡) ∈ ℝ𝐾

𝜏×𝑑 are used in the VideoLM.

The residual frame 𝛿(𝑡) is embedded by a lightweight ResNet-18 [87] module to extract local features over the same grid size 𝐻𝐺 × 𝑊𝐺. Similarly to above, a residual transformer 𝜃residual aggregates and compresses the raw features to a set of 𝐾𝛿 compressed residual tokens 𝛿tok(𝑡) ∈ ℝ𝐾

𝛿×𝑑 as:

𝛿tok(𝑡) = 𝜃residual(ResNet(𝛿(𝑡))) . (6)

Interleaved token stream. In practice, we set 𝐾𝜏 = 𝐾𝛿 = 4 and thus 𝑁 = 8 (see Supp.; performance plateaus beyond 𝑁 = 8). The final visual sequence 𝑋 = [𝑥(1), . . . , 𝑥(𝑇)] is formed by interleaving I-frame

tokens 𝑋𝐼(𝑡) and P-frame tokens 𝑋𝑃(𝑡) in their natural temporal order, as shown in Eq. 2. The LLM can consume 𝑋𝐼(𝑡) and 𝑋𝑃(𝑡) alongside textual instructions, without any architectural modifications. This approach reduces the redundant data considerably while preserving accurate temporal coverage and enables smooth scalability to long videos by adjusting the I-frame density, P-frame grouping, and token allocation.

#### 3.3 Training Paradigm

The training is done in two stages. First the Δ-encoder is pre-trained in order to render it compatible with the image encoder. Then, this encoder is integrated into a VideoLM and the whole pipeline is fine-tuned end-to-end.

Δ-encoder pre-training. The primary difficulty lies in ensuring that the Δ-tokens 𝑋𝑃(𝑡) are aligned with the image tokens 𝑋𝐼(𝑡). To achieve this, we pre-train the Δ-Encoder as a modality adapter that enables codec-derived primitives (𝜏(𝑡), 𝛿(𝑡)) to be compatible with the embedding space defined by a vision encoder. By aligning the Δ-tokens with this space, P-frames can then be compactly represented and replace standard RGB tokens within a VideoLM. For pre-training, two additional modules are used on top of the outputs of the Δ-encoder. First, a “reference” transformer 𝜃ref uses the image tokens from 𝐼(𝑡−1) and the compressed motion vector tokens 𝜏tok(𝑡) , and its aim is to understand how the information moved in the image and how this displacement transforms the reference tokens. This is akin to the warping in Eq. 1. Second, a “warped” transformer 𝜃warped takes these enriched tokens and the residual tokens 𝛿emb(𝑡) in order to add the residual information as in Eq. 1. These final features 𝑋ˆ𝑃(𝑡) should be similar to the image features extracted from the raw ˆ𝐼(𝑡). The process can be summarized as:

𝑋𝐼(𝑡−1) =𝜙𝑅𝐺𝐵(𝐼(𝑡−1)) (7) 𝑋ˆwarped(𝑡−1) =𝜃ref(𝑋𝐼(𝑡−1), 𝜏tok(𝑡) ) (8)

𝑋ˆ𝑃(𝑡) =𝜃warped(𝑋ˆwarped(𝑡−1) , 𝛿tok(𝑡) ) . (9)

To align Δ-tokens with image tokens, we apply patch-wise regression against the outputs of a frozen vision encoder. Let 𝑋𝐼(𝑡) = 𝜙RGB(ˆ𝐼(𝑡)) ∈ ℝ𝑀×𝑑 denote the tokens of the ground-truth target frame. We minimize:

###### ∑︁𝑀

1

𝑋𝐼(𝑡)(𝑖) − 𝑋ˆ𝑃(𝑡)(𝑖) 22. (10)

LMSE =

𝑀

𝑖=1

|Frame(t-s)𝐼|
|---|

Unlike global contrastive losses, this finegrained objective encourages spatially consistent alignment across patches. As a result, the Δ-Encoder is able to produce representations that are more closely aligned with the RGB token space, aiming to improve the integration of I- and P-frames during downstream VideoLM training (see Supp.; skipping pretraining and single stage end-to-end training weakens performance).

[Figure 39]

[Figure 40]

Frame(t-s)𝐼

[Figure 41]

[Figure 42]

Vision Encoder

[Figure 43]

[Figure 44]

[Figure 45]

###### Reference

[Figure 46]

Transformer

𝚫-Encoder

[Figure 47]

|[Figure 48]| |
|---|---|
| | |

[Figure 49]

MotionResidualτδ(t)(t)

[Figure 50]

[Figure 51]

Motion

[Figure 52]

[Figure 53]

MLP

[Figure 54]

Transformer

[Figure 55]

Query Tokens

𝚫-tokens

[Figure 56]

[Figure 57]

|[Figure 58]|
|---|

###### Warped

[Figure 59]

[Figure 60]

Transformer

###### Residual

[Figure 61]

[Figure 62]

[Figure 63]

ResNet-18

Transformer

Integration into VideoLMs. After pretraining, we integrate the Δ-encoder 𝜙Δ into the VideoLM pipeline for full fine-tuning and interleave the tokens coming from I/ P-frames. Note that the Reference and Warped transformers from the Δ-encoder pre-training stage are not used at this stage, so no RGB reference frames are processed for the P-frames when training the language model. This yields a substantial compute and memory reduction (shown in Sec. 4.4). The LLM architecture and training objective remain unchanged (standard instruction tuning / next-token prediction loss). The Δ-encoder adds fewer than 15M parameters, representing negligible overhead relative to the 7B LLM.

Figure 3 Δ-encoder processes motion vectors and residuals through two lightweight branches designed to extract and compress codec information. The resulting motion and residual tokens are concatenated to form the Δ-tokens used for our efficient P-frame representation, which is projected to the RGB token space during pre-training.

### 4 Experiments

#### 4.1 Experimental Setup

Training Pipeline. For simplicity, we re-encode videos to the MPEG-4 codec at 30 FPS with a GOP size of 240 frames and use a fusion size of 𝑠 = 30, yielding an effective rate of 1 FPS. We use LLaVA-Video7B [4] as the base VideoLM, which consists of SigLIP [41] as the vision encoder and Qwen2-7B [88]

- as the language model. First, we pre-train the Δ-encoder on frame pairs sampled from videos in the LLaVA-Video-178K [4]. Second, we fine-tune the full VideoLM on the same dataset, comprising a total of 1.39M QA instruction tuning samples. We use a learning rate of 1e−5, an effective total batch size of 128, and train for 21K GPU hours (64 A100-80G for 14 days).

Evaluation Benchmarks. We comprehensively evaluate our method across 14 video benchmarks spanning four categories: (i) general video QA: PerceptionTest [89], NextQA [90], ActivityNet-QA [91], and VideoMME [92]; (ii) temporal reasoning: TempCompass [93], TOMATO [94], CVRR-ES [95] and MVBench [96]; (iii) long-form and instruction-following: LongVideoBench [97], LVBench [98], Video-TT [99], and Video-MMMU [100]; and (iv) spatial scene understanding: ScanQA [101] and SQA3D [102]. Our primary comparison is with LLaVA-Video-7B [4], which serves as our base model. We additionally compare with various similar open-source approaches. Following standard practice, we use lmms-eval [103] for evaluation. ActNet-QA [91] uses GPT-based evaluation; we report scores using gpt-4o-2024-11-20 in Tab. 1 and Azure deployed version of gpt-3.5-turbo-0613 in Tab. 2 for fair comparison with prior work.

Frame Sampling. Most open-source VideoLMs with a 32K token window sample 64 frames per video, regardless of length [4, 104]. As a result, unlike proprietary models, they cannot process videos at

- Table 1 Token Efficiency vs. Accuracy in Video QA. We report the performance of LLaVA-Video (7B) at different number of keyframes per GOP, as well as in the default setup of selecting 64 keyframes regardless of video length. For each setting, we also report the performance of our method using the same keyframes as I-frames and the remaining frames in the GOP as P-frames. We report accuracy (Acc., %) and the % of tokens used compared to the default setup (64 keyframes). Our method only adds a low number of Δ-tokens compared to its associated baseline and these help close the gap compared to the next baseline that is using a significantly larger number of tokens.

PerceptionTest NextQA ActNet-QA

Model Sampling

Token (%) Acc. (%) Token (%) Acc. (%) Token (%) Acc. (%)

- LLaVA-Video 1 keyframe / GOP 005.3 60.4 008.3 77.9 021.6 61.7

Ours + 7 P-frames / GOP 006.9 65.5

+5.1

011.2 78.3

+0.4

029.7 62.3

+0.6

- LLaVA-Video 2 keyframes / GOP 009.7 62.1 015.9 79.2 042.5 62.9

+6.6

+1.2

+0.7

Ours + 6 P-frames / GOP 011.3 68.7

018.6 80.4

050.4 63.6

LLaVA-Video 4 keyframes / GOP 018.6 63.6 031.0 80.5 084.3 63.6 Ours + 4 P-frames / GOP 020.1 70.3

+6.7

+1.2

+1.6

033.5 82.1

091.7 64.8

LLaVA-Video 64 keyframes total 100 67.9 100 83.2 100 64.1

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

70

###### PT

65

60

Accuracy(%)

84

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

82

###### NQA

80

78

76

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

64

AN-QA

62

60

0 20 40 60 80 100

|LLaVA-VideoToken (%) Ours<br><br>|
|---|

1 FPS for videos longer than 64s. Our framework decouples frame rate from token budget: we sample

- at 1 FPS (counting both I- and P-frames), with the number of keyframes per GOP controlling the token cost rather than the temporal coverage. Concretely, each GOP contributes one I-frame (𝑀 tokens) plus up to seven P-frames (𝑁 = 8 tokens each), so the per-GOP cost ranges from 𝑀 + 7𝑁 (1 keyframe) to 4𝑀 + 4𝑁 (4 keyframes) depending on configuration. This allows the token budget to scale naturally with video duration. In our experiments, for fair comparison, we sample at 1 FPS up to 64 GOPs. For videos longer than 64 GOPs (= 512s), we perform uniform GOP sampling and encode only I-/P-frames within each sampled GOP.

#### 4.2 Effectiveness of Δ-tokens

We first evaluate whether the proposed Δ-tokens derived from P-frames provide understanding capabilities comparable to dense RGB representations. Tab. 1 summarizes results on three benchmarks under different sampling configurations. Across all settings, our codec-aware representation achieves consistent improvements in accuracy compared to the same-sampling baseline, and helps close the gap to the denser sampling configuration while using significantly fewer tokens. Remarkably, even under aggressive compression (e.g., 1 keyframe per GOP), our model maintains strong performance with over an order-of-magnitude token reduction. This demonstrates that the Δ-encoder successfully captures motion and appearance cues critical for temporal reasoning (see Supp.; zeroing out Δ-tokens at inference time confirms the VideoLM actively utilizes them and the same scaling trend holds when our model serves as the keyframe-only baseline). At higher frame densities (e.g., 4 keyframes per GOP), our model not only matches but often surpasses the performance of the 64-frame LLaVA-Video7B [4] baseline, despite using only a fraction of its tokens. The largest relative gains are observed on PerceptionTest [89] (+6.7%) and NextQA [90] (+1.6%), indicating stronger reasoning under a limited token budget. These results confirm that our codec-aware tokenization preserves temporal semantics and fine-grained dynamics. As shown in the accompanying plots, our method consistently advances the Pareto-optimal frontier across all three benchmarks, achieving competitive or superior accuracy at substantially reduced token counts.

#### 4.3 Comparison with Current Approaches

We benchmark CoPE-VideoLM against a broad range of both open-source and proprietary VideoLMs. We consider four groups of tasks (see Supp. for a detailed description of the benchmarks): (i) general

- Table 2 General video understanding benchmarks. The best results among open-source methods are highlighted as first , second , and third . Our model achieves state-of-the-art performance among open-source video language models of comparable scale. Notably, it does so while operating on a fraction of the visual tokens, confirming that codec primitives preserve the semantic and temporal cues needed for diverse understanding tasks.

PerceptionTest NextQA ActivityNet-QA VideoMME

Model

val mc test w/o sub w sub Proprietary Models

GPT-5 [55] - 86.3 - 83.3 86.9 Gemini 3 Pro [105] - 84.3 - 88.6 87.5 Gemini 2.5 Pro [12] - 85.3 - 87.8 87.8 Claude Sonnet 4.5 [56] - 79.2 - 74.2 80.5

Open-Source VideoLMs Video-LaVIT [26] 47.9 - 50.1 - EMA-7B [27] - - 52.1 53.4 58.4 VILA-40B [57] 54.0 67.9 58.0 60.1 61.1 LongVA-7B [34] - 68.3 50.0 52.6 54.3 IXC-2.5-7B [36] 34.4 71.0 52.8 55.8 58.8 LLaVA-OV-7B [33] 57.1 79.4 56.6 58.2 61.5 Apollo-7B [39] 67.3 - - 61.3 63.3 Oryx-7B [104] 68.6 81.9 - 58.3 62.6 LLaVA-Video-7B [4] 67.9 83.2 56.5 63.3 69.7

Ours-7B 70.3 82.1 60.3 61.9 69.4

video QA (Tab. 2), (ii) temporal and motion reasoning (Tab. 3 (a)), (iii) long-form and instruction following (Tab. 3 (b)), and (iv) spatial scene understanding (see Supp.; our method performs on par with 2D VLMs using only 25% of tokens).

General Video QA. As shown in Tab. 2, despite being trained on a smaller corpus than most competing models, our codec-aware formulation achieves competitive or superior results across all major benchmarks. By leveraging motion vectors and residuals directly from the compressed stream, CoPE-VideoLM encodes substantially more frames within the same token budget, enhancing temporal coverage without sacrificing spatial fidelity. On PerceptionTest and ActivityNet-QA, our model yields the highest accuracy among all open-source models, indicating improved motion and appearance reasoning due to the Δ-encoder’s temporally grounded representation. The performance gap observed on select benchmarks such as VideoMME is attributable to our smaller training corpus rather than the codec formulation itself; we discuss this in detail in the Supp., where matched-data comparisons consistently favor our method. Notably, among codec-based approaches, CoPE-VideoLM substantially outperforms both Video-LaVIT [26] and EMA [27] across all reported benchmarks, validating that preserving both residuals and temporal ordering yields stronger video-language representations than fixed-length GOP summaries or motion-only tokenization.

Temporal And Motion Reasoning. Tab. 3 (a) evaluates benchmarks specifically designed to probe temporal understanding. On TempCompass [93], TOMATO [94] and CVRR-ES [95], CoPE-VideoLM achieves the highest accuracy among all open-source models. This confirms that the explicit encoding of motion vectors and residuals provides a stronger temporal signal than dense RGB frame processing. These gains are notable given that temporal reasoning requires precisely the kind of fine-grained inter-frame dynamics that codec primitives natively capture. On MVBench, which emphasizes pictorial semantic understanding over temporal dynamics, our model improves over LLaVA-Video-7B by 3.0%. The remaining gap to the top-ranked models reflects the base model’s capacity and training data mixture rather than a codec-specific limitation.

Long-form and Instruction-following. Tab. 3 (b) reports results on benchmarks requiring under-

- Table 3 (a) Temporal reasoning and motion understanding benchmarks. CoPE-VideoLM achieves the highest accuracy on TempCompass, TOMATO, and CVRR-ES, confirming that codec primitives provide a strong inductive bias for temporal reasoning. (b) Long-form and instruction-following benchmarks. CoPE-VideoLM performs better than other open-source models on Video-TT, Video-MMMU, and LVBench, demonstrating compact Δ-tokens effectively scale to longer temporal contexts and complex instruction-following.

TempCompass Tomato CVRR-ES MVBench test MCQ test test test Proprietary Models

Video-TT Video-MMMU LVBench LongVideoBench

Model

Model

mc test test val Proprietary Models

GPT-5 [55] 80.4 53.0 - 74.1 Gemini 3 Pro [105] 82.8 48.3 - 70.4 Gemini 2.5 Pro [12] 81.9 48.6 - 70.6 Claude Sonnet 4.5 [56] 72.8 39.6 - 62.1

GPT-5 [55] - - 68.8 72.6 Gemini-3-Pro [105] - - 78.0 75.9 Gemini-2.5-Pro [12] - - 78.4 76.8 Claude-Sonnet-4.5 [56] - - 50.5 65.1

Open-Source VideoLMs LongVA-7B [34] 56.9 - - VideoLLaMA2-7B [106] - 18.5 21.6 54.6 InternVL2-8B [38] 65.3 21.7 - 65.8 VideoChat2-7B [96] 45.5 - - 51.1 VideoCCAM-9B [107] - 27.0 - 64.6 IXC-2.5-7B [36] 67.1 - - 69.1 LLaVA-OV-7B [33] 64.8 25.5 42.6 56.7 Apollo-7B [39] 64.9 - - LLaVA-Video-7B [4] 66.6 24.9 43.6 58.6 Ours-7B 68.9 28.3 49.4 61.9

Open-Source VideoLMs EMA-7B [27] - - - 47.0 LongVA-7B [34] - 23.9 - Kangaroo-8B [108] - - 39.4 54.8 mPLUG-Owl3-7B [109] - - 43.5 52.1 PLLaVA-34B [58] - - 26.1 53.2 InternVL2-8B [38] - 37.4 - 54.6 TimeMarker [37] - - 41.3 56.3 LLaVA-OV-7B [33] 44.0 33.9 38.1 56.5 LLaVA-Video-7B [4] 41.8 36.1 44.2 58.2

Ours-7B 45.5 38.2 46.4 56.9

(a) (b)

standing of extended video sequences and complex instructions. CoPE-VideoLM achieves the best results among open-source models on Video-TT [99], Video-MMMU [100], and LVBench [98], while remaining competitive on LongVideoBench [97]. The improvement is especially pronounced against EMA [27], even though we train on less data (1.39M video-only vs. 2.4M video+image QA samples) and do not need to increase the GOP count at training time to handle longer videos. These results suggest that preserving residuals and explicit temporal ordering plays an important role for long-form understanding. Overall, compressing P-frames into compact Δ-tokens enables substantially more temporal coverage within the same token budget, supporting both fine-grained temporal reasoning and efficient use of the context window.

#### 4.4 Runtime and Memory

Beyond competitive performance in accuracy, CoPE-VideoLM provides critical efficiency gains during inference. Fig. 4 (a) reports time-to-first-token (TTFT) and end-to-end-latency (E2EL) to generate 64 text tokens, measured on a single consumer-grade GPU at 1 FPS video input. Compared to the 64-frame LLaVA-Video-7B baseline, our most compact configuration (1 keyframe per GOP) achieves a 86.2% reduction in TTFT and a 56.1% faster E2EL. This improvement stems from the reduced visual embedding load (only I-frames require full RGB encoding) and the shorter overall sequence length processed by the LLM due to the Δ-tokens. We highlight the scalability of the computational advantage of the Δ-token formulation in Fig. 4 (b). Standard dense RGB sampling saturates quickly, limiting coverage to short sequences as memory constraints are rapidly encountered. In contrast, CoPEVideoLM exhibits a highly efficient relationship between video length and token budget. The token efficiency enables scaling to sequences previously inaccessible to open-source models; our most compact configuration allows for the processing of videos up to 8 hours in duration (at 1 FPS) within a 1M token context, demonstrating an order-of-magnitude increase in processing capability over the baseline. The benefits become increasingly significant for longer sequences, as the quadratic attention cost amplifies reductions in token count. Moreover, the architecture uses standard transformer components with well-established scaling properties. Together, these results confirm that codec-aware tokenization is not just semantically sound but is a necessity for enabling fast inference and comprehensive long-form video coverage without needing architectural modifications or additional hardware.

Model / Sampling TTFT (s) E2EL (s) LLaVA-Video-7B (64 kf) 2.39 3.78 Ours-7B

- 1 keyframe per GOP

(8 kf + 56 P-frames)

0.33 1.66

- 2 keyframes per GOP

0.51 1.71

(16 kf + 48 P-frames)

- 4 keyframes per GOP

(32 kf + 32 P-frames)

0.90 2.28

|Ours|(2I+6P)| | | | |
|---|---|---|---|---|---|
|Ours|(1I+7P)| | | | |
| | | | | |Gemini2.5-Pro|

32K 64K 128K 256K 512K 1M Token Budget

2min

10min

30min

- 1h
- 2h

5h

8h

10h

VideoLength

LLaVA-Video-7B

Gemini2.5-Pro

VideoLM (1FPS) Ours (4I+4P)

(a) (b)

- Figure 4 (a) Runtime comparison: TTFT and E2EL for generating 64 text tokens at 1 FPS on a single GPU. (b) Video length vs. token budget: Token budget is shown on a log scale; dashed lines mark evaluated budgets. Δ-token representation enables scaling to significantly longer videos without exceeding context limits.

- 5 Conclusion

Limitations and Future Work. Our current approach focuses on I- and P-frames, lacking support for B-frames and their complex non-causal dependencies. One option to address this would be using the decode order instead of the render order. Furthermore, we currently operate on a tensorized version of the codec primitives. For future work, it would be interesting to stay closer to the raw codec primitives, by operating directly on sets of block-wise motion vectors and quantized DCT coefficients, which could offer even better computational and token efficiency. Finally, we use a fixed P-frame fusion window, which is suboptimal for tasks with varying motion. Exploring sensitivity to codec type, bitrate, and encoding quality remains an open question.

Conclusion. Through comprehensive evaluation across 14 video understanding benchmarks, we have demonstrated that codec-aware tokenization offers a compelling alternative to traditional keyframe sampling and outperforms prior codec-based approaches for VideoLMs. By leveraging the information natively encoded by video compression algorithms (i.e., motion vectors and residuals), we achieve substantial efficiency gains while maintaining competitive performance. Notably, our approach reduces time-to-first-token by a significant margin (up to 86%) which is essential for real-time applications. As models scale towards larger context windows, our approach becomes increasingly valuable, enabling richer temporal representations with much lower computational overhead than traditional sparse keyframe sampling. This work positions codec-based methods as a practical and efficient foundation for scaling future VideoLMs.

- 6 Acknowledgements

The authors would like to thank (in alphabetical order): Isar Meijer and Krzysztof Waraksa from Microsoft for help with training pipeline setup; Kevin Qu, Tao Sun and Jianhao Zheng from Stanford for feedback at different stages of the project.

### Appendix

In the appendix, we provide the following:

- 1. Video decoding illustration (Sec. A)
- 2. Details about Δ-encoder (Sec. B)
- 3. Additional training details (Sec. C)
- 4. Details on training data and evaluation benchmarks (Sec. D)
- 5. Scale of training data (Sec. E)
- 6. Spatial video question-answering (Sec. F)
- 7. Comparison with token pruning (Sec. G)
- 8. Ablation study (Sec. H)

- • Varying the number of Δ-tokens (Sec. H.1)
- • Two-Stage training (Sec. H.2)
- • Are Δ-tokens used by the VideoLM? (Sec. H.3)
- • Benefits of codec primitives (Sec. H.4)
- • Scaling to higher frame rates (Sec. H.5)
- • Next-frame retrieval using the Δ-encoder (Sec. H.6)

- 9. Qualitative results (Sec. I)

### A Video Decoding Illustration

- As mentioned in Sec. 3.1, the size of each Group of Pictures (GOP) is usually decided adaptively by the codec and video encoder depending on the motion or change. At encoding time, the developer can provide an upper bound for the GOP size, or even fix it. For example, if the GOP size is fixed at 240, the first frame of each GOP is an I-frame and the rest 239 frames are P-frames, each consisting of motion vectors and residuals. During video decoding, the I-frame or reconstructed frame at the previous timestamp 𝑡 − 1 is first moved using the motion vectors and then the residuals are added to get the RGB frame at the current timestep 𝑡, as described in Eq. 1. We visualize an example of the decoding process in Fig. 5.

Frame I(t-1) Motion τ(t) Residual δ(t) Motion Compensated I(t-1) Frame I(t)

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

Figure 5 Codec primer. We visualize from left to right: the previous frame, the motion vectors and residuals between previous and current frame, the intermediate reconstruction after motion compensation, and the final result after adding the residuals.

### B Details about Δ-Encoder

The Δ-Encoder converts codec primitives, motion vectors and residuals, into a compact set of 𝑁 = 𝐾𝜏+𝐾𝛿 tokens aligned with the vision encoder’s embedding space. The entire module is lightweight (< 15M parameters) and operates purely in the compressed domain during VideoLM fine-tuning. Here, we provide additional architectural and implementation details.

Motion Vector Branch. Given motion vectors 𝜏(𝑡) ∈ ℤ𝐻×𝑊×2, we first perform min–max normalization to map all values to [−1, 1]. We then “patchify” the motion-field into non-overlapping 16 × 16 blocks, yielding a grid of size 𝐻𝐺 × 𝑊𝐺 where 𝐻𝐺 = 𝐻/16 and 𝑊𝐺 = 𝑊/16. Each patch is flattened into a vector of dimension 162×2, and a lightweight two-layer MLP with shared weights is applied independently to each patch, producing per-block embeddings with the same feature dimension as the vision encoder (𝑑 = 1152). To aggregate these features, we employ a transformer equipped with 𝐾𝜏 learnable query tokens. These tokens are concatenated with the motion features along the sequence length dimension at the input of the transformer and then processed together using regular multi-headed attention layers. The transformer contains 4 layers with hidden dimension 𝑑, uses 9 attention heads, and adopts PreNorm residual blocks throughout.

Residual Branch. Residuals 𝛿(𝑡) ∈ ℝ𝐻×𝑊×𝐶 are processed through a truncated ResNet-18 backbone (all convolutional layers up to the final global pooling) [87]. The resulting spatial features share the same grid resolution as above (𝐻𝐺 × 𝑊𝐺). A second transformer, architecturally identical to the motion branch but with its own learned 𝐾𝛿 queries, is used to compress these features.

Pre-training the Δ-Encoder. As mentioned in Sec. 3.3, the Δ-Encoder is augmented with two auxiliary transformer modules, reference and warped branches, that enable it to reconstruct the token representation of the target frame without decoding RGB pixels. These transformers are architecturally identical to the ones in the motion vector and residual branch.

### C Additional Training Details

Pre-training. We pre-train the Δ-encoder with the warped and reference branches together for two days using 16×A100 GPUs, running 113K iterations with a global batch size of 1024 and a per-GPU learning rate of 6.25×10−5, optimized with AdamW and a cosine scheduler with warmup steps = 1000.

VLM Training. To better adhere to the original RGB aligned latent space, we train the VideoLM with 4 keyframes + 4 P-frames per GOP, for 10.9K steps on 64×A100 GPUs with a global batch size of 128. We keep the same hyperparameter settings as LLaVA-Video [4].

### D Training Data and Evaluation Benchmarks

Training Data. We train on LLaVA-Video-178K [4], a video instruction-tuning dataset organized by duration (0–30s, 30–60s, 1–2min, 2–3min) with sources from academic benchmarks and YouTube. It covers captioning, open-ended QA, and multiple-choice QA. ActivityNet-QA [91], NextQA [90], and PerceptionTest [89] are included as QA subsets; following prior work, we treat these together with the rest of LLaVA-Video-178K as a single training corpus totalling 1.39M samples.

General Video QA. PerceptionTest [89] tests fine-grained perceptual reasoning (memory, abstraction, physics) through multiple-choice questions over short diagnostic videos. NextQA [90] targets causal, temporal, and descriptive reasoning on short clips via multiple-choice QA. ActivityNet-QA [91] poses open-ended questions about complex web videos spanning diverse activities, scored by GPT. VideoMME [92] covers short to long videos with questions on spatial, temporal, and semantic understanding, reported with and without subtitles.

Temporal and Motion Reasoning. TempCompass [93] probes temporal understanding through questions about speed, direction, order, and attribute change. TOMATO [94] tests action ordering,

duration, and state transition reasoning via multiple-choice QA. CVRR-ES [95] uses compositionally varied questions to expose systematic failure modes in video reasoning. MVBench [96] spans 20 temporal tasks including action sequence, scene transition, and object interaction recognition.

Long-form and Instruction Following. LongVideoBench [97] tests understanding of long, interleaved video-language sequences with multi-step reasoning. LVBench [98] targets long-form video comprehension with videos exceeding several minutes. Video-TT [99] tests complex multi-step video reasoning and holistic understanding. Video-MMMU [100] tests knowledge acquisition from multi-discipline professional videos.

### E Scale of Training Data

- As mentioned in Sec. 4.3, CoPE-VideoLM underperforms LLaVA-Video on select benchmarks, e.g., VideoMME [92]. This gap reflects training data scale and composition rather than the codec formulation. Tab. E.1 isolates this effect by reporting results from the LLaVA-Video paper [4] across its incremental training stages. LLaVA-Video begins with 0.25M samples (LLaVA-Hound) and progressively adds LLaVAVideo-178K (1.58M), three QA datasets (1.64M), and finally 1.1M LLaVA-OneVision images (2.74M). Two observations stand out. First, VideoMME improves substantially with the image alignment data (61.9 → 63.4), which our pipeline lacks entirely. Second, adding the QA datasets reduces VideoMME (63.2 → 61.9). Since this configuration closely mirrors our data mixture, and we observe similar VideoMME accuracy, this confirms a data-mixture effect rather than a codec-specific limitation. We additionally include a “sampled” variant as reported in the LLaVA-Video paper [4], where LLaVA-Video is trained on a randomly sampled 1.08M subset of LLaVA-Video-178K; our method outperforms it by a wide margin across all benchmarks.

- Table E.1 Effect of training data scale. LLaVA-Video rows are reported in [4] across incremental training stages. Against the most comparable configurations (+ 3 QA datasets and “sampled”), our method outperforms or remains on par across benchmarks despite using fewer samples.

Training Data Total Data NextQA PerceptionTest VideoMME LLaVA-Video [4]

LLaVA-Hound 0.25M 64.4 51.4 54.1 + LLaVA-Video-178K 1.58M 80.1 57.1 63.2 + 3 QA datasets 1.64M 80.1 69.0 61.9 + LLaVA-OV (Images) 2.74M 83.2 67.9 63.4

LLaVA-Video-178K (sampled) 1.08M 73.2 55.9 59.6 Ours LLaVA-Video-178K + 3 QA datasets 1.39M 82.1 70.3 61.9

### F Spatial Video Question Answering

We evaluate our method on two standard 3D QA benchmarks: (1) SQA3D [102] for situated reasoning, and (2) ScanQA [101], for spatial understanding. Both datasets require associating multi-view observations with 3D spatial structure, making them a natural testbed for assessing whether compressed-domain video cues can support geometry-aware reasoning.

Training. We follow the same training pipeline used in our video-language experiments. However, since these videos are comparatively shorter in duration (∼ 15s total), we re-encode them with a GOP size of 120 at 30 FPS with accumulation size 𝑠 = 10. This leads to around 3-4 GOPs per video. Following standard practice in 3D LMMs, we fine-tune our base model on ScanQA and SQA3D training

- Table F.1 Evaluation of 3D question-answering on SQA3D [102] and ScanQA [101]. “Expert models” are customized for specific tasks with task-oriented decoders. “EM” stands for top-1 exact match and “EM-R” means the refined exact match following [110]. “–” indicates the number is not available. We show results for our method in zero-shot setup where it performs comparable to state-of-the-art VideoLMs despite using only 25% of tokens. We further show results after fine-tuning where the performance increases significantly, even outperforming a significant number of 3D VLMs which employ additional inputs (e.g., point-clouds, camera poses).

SQA3Dtest ScanQAval EM EM-R CIDEr BLEU-4 METEOR ROUGE EM

Point Encoder

Vision Encoder

Method

Expert Models SQA3D [102] ✓ ✗ 46.6 – – – – – – ScanQA [101] ✓ ✗ – – 64.9 10.1 13.1 33.3 21.1 3D-VLP [111] ✓ ✗ – – – 11.2 13.5 34.5 21.7 3D-VisTA [112] ✓ ✗ – – – – 13.9 35.7 22.4

3D VLMs Chat-3D [113] ✓ ✗ – – 53.2 6.4 11.9 28.5 – 3D-LLM [114] ✓ ✓ – – 69.4 12.0 14.5 35.7 20.5 Scene-LLM [115] ✓ ✓ 53.6 – 80.0 11.7 15.8 35.9 27.2 LL3DA [116] ✓ ✗ – – 76.8 – 15.9 37.3 – LEO [110] ✓ ✓ 50.0 52.4 80.0 11.5 16.2 39.3 21.5 ChatScene [117] ✓ ✓ 54.6 57.5 87.7 14.3 18.0 41.6 21.6 Grounded 3D-LLM [118] ✓ ✓ – – 72.7 13.4 – – – LLaVA-3D [119] ✗ ✓ 55.6 57.6 91.7 14.5 20.7 50.1 27.0 Video-3D-LLM [120] ✗ ✓ 58.6 – 102.1 16.4 20.0 49.3 30.1 Ross3D [121] ✗ ✓ 63.0 65.7 107.0 17.9 20.9 50.7 30.8

VideoLMs (Zero-shot)

InternVL2-8B [38] ✗ ✓ 33.0 45.3 62.5 3.3 14.5 34.3 – Qwen2-VL-7B [48] ✗ ✓ 40.7 46.7 53.9 3.0 11.4 29.3 – LLaVA-Video-7B [4] 100% tokens ✗ ✓ 48.5 – 88.7 3.1 17.7 44.6 –

Ours-7B (Zero-shot) 25.78% tokens ✗ ✓ 46.5 49.8 70.9 7.1 14.7 38.2 – Ours-7B (Fine-tuned) 25.78% tokens ✗ ✓ 56.6 59.3 96.9 14.9 19.1 46.4 27.5

split, without any dataset-specific heuristics or architectural modifications. We train and evaluate with

- 6 keyframes + 6 P-frames per GOP to align better with 32-RGB frame setting of the other models.

Results. We report the results in Tab. F.1. To maintain fairness with VideoLMs, we show results for both our original and fine-tuned model. Despite using 1/4th of the number of tokens compared to LLaVA-Video-7B [4], the original CoPE-VideoLM matches state-of-the-art VideoLMs. With fine-tuning, our performance becomes comparable to the leading 3D VLMs, despite our method not having access to camera poses or 3D point-clouds. However, we note that fine-tuning would likely benefit other VideoLMs similarly; the key takeaway is that codec-aware tokens retain sufficient spatial information for 3D reasoning at a fraction of the token cost.

### G Comparison with Token Pruning

As discussed in Sec. 2, token compression methods operate on dense vision tokens after full RGB encoding. Although they reduce the LLM’s token count, these still pay the full image embedding cost for every frame, which dominates prefill latency. In contrast, CoPE-VideoLM avoids full image encoding for most frames entirely by leveraging codec-level sparsity. Tab. G.1 compares CoPE-VideoLM against FastV [68], DyCoke [75], PLLaVA [58], VisionZip [122], and LLaVA-Scissor [17]. All baselines use 50% token budget of the default frame configuration, while our method enables up to 93% lower token usage, as shown in Sec. 4.1.

- Table G.1 Comparison with token compression methods. CoPE-VideoLM outperforms post-hoc pruning approaches across all three benchmarks while being faster in TTFT, as P-frames bypass the vision encoder entirely.

Method ActNet-QA Next-QA VideoMME w/o sub

FastV [68] 47.95 81.1 57.5 DyCoke [75] 47.88 81.1 57.4 PLLaVA [58] 47.59 81.0 56.9 VisionZip [122] 45.42 78.5 54.2 LLaVA-Scissor [17] 47.89 81.2 57.4 Ours 60.28 82.1 61.9

H Ablation Study

To reduce carbon footprint and computation cost compared to the main high compute run, for the ablation experiments, we train our CoPE-VideoLM on the three QA datasets: PerceptionTest [89], NextQA [90], ActivityNet-QA [91], comprising 60K samples for 2 days on 16×A100. Due to this, the results shown in Sec. H.1, Sec. H.2, and Sec. H.3 are not directly comparable to these reported in Sec. 4. We perform evaluation on the val and test splits of PerceptionTest [89] and NextQA [90] respectively.

- H.1 Varying the Number of Δ-Tokens

The number of Δ-tokens emitted per P-frame controls the expressive capacity of the codec branch. Fewer tokens encourage a more aggressively compressed representation, whereas larger token budgets allow the Δ-Encoder to retain finer motion and appearance cues from the codec primitives. Tab. H.1 reports the effect of varying this value while keeping all other model and sampling configurations fixed. We observe a consistent trend across both benchmarks. Moving from 2 to 4 Δ-tokens yields a noticeable improvement, as the model benefits from an expanded latent space that can more faithfully capture motion and residual structure. Increasing to 8 Δ-tokens further improves performance, reaching the best overall results. Beyond this point, however, allocating more capacity produces diminishing returns: both NextQA and PerceptionTest remain nearly unchanged when increasing to 16 tokens. This indicates that the codec primitives contain a limited amount of signal per fused P-frame, and that 8 tokens are sufficient to encode the relevant temporal variations. For this reason, we adopt 8 Δ-tokens per P-frame as the default configuration throughout, balancing accuracy and token efficiency.

- Table H.1 Number of Δ-tokens per P-frame. We train several versions of our model with a different number of Δ-tokens per P-frame. Performance significantly increases when going from aggressive compression (2 or 4 tokens) up to 8, which is the value we used for the results in the main paper. Going to 16 further improves performance, but with diminishing returns that do not justify the increased token cost.

Δ-tokens per P-Frame PerceptionTest NextQA

1 Keyframe per GOP 2 63.26 75.04 4 65.68

+1.04 8 67.33

+2.42

76.08

+1.29 16 67.69

+1.65

77.37

+0.36

+0.52

77.89

#### H.2 Two-Stage Training

We ablate the contribution of each component in our two-stage training pipeline: (i) pre-training the Δ-Encoder to align codec primitives with the RGB embedding space, and (ii) end-to-end finetuning of the full VideoLM. The results are shown in Tab. H.2. Fine-tuning the VideoLM without any Δ-encoder pre-training yields reasonable performance, though noticeably worse than the full

- Table H.2 Two-stage training. We attempt to directly train the Δ-encoder together with the LLM without our initial pre-training scheme. This yields significantly lower performance than the two stage setup proposed in the main paper: Stage 1 – Pre-train the Δ-encoder and Stage 2 – Fine-tune pre-trained Δ-encoder & pre-trained LLM together.

Pre-train Δ-Encoder Fine-tune LLM PerceptionTest NextQA

1 Keyframe per GOP

- ✓ 63.45 74.56 ✓ ✓ 67.33 77.37

two-stage approach. In this one-stage setting, the model must simultaneously learn motion–residual interpretation, feature alignment, and multimodal reasoning, which slows convergence and leads to weaker temporal understanding, particularly on PerceptionTest. The best results are achieved when both stages are used. Pre-training ensures that Δ-tokens inhabit a well-structured embedding space, while fine-tuning teaches the LLM how to fuse the I- and P-frame tokens together. This combination consistently provides the strongest performance across both datasets. While two-stage training is clearly advantageous in our data regime, we note that the benefit may diminish when training on significantly larger datasets. In such high-data settings, one-stage training may gradually compensate for the missing pre-training at the cost of more compute.

- H.3 Are Δ-tokens used by the VideoLM?

A natural question is whether the VideoLM actually uses the Δ-tokens during inference, rather than ignoring them and relying solely on sparse keyframes. To probe this, we evaluate with 1 keyframe + 7 Pframes per GOP but zero out all P-frame Δ-tokens at inference time, preserving temporal structure while removing all codec information. As shown in Tab. H.3, performance drops substantially, confirming that the model actively leverages Δ-tokens for temporal reasoning rather than merely interpolating between sparse I-frames.

Table H.3 Ablation on the use of Δ-tokens. We compare our method with a version where we set all Δ-tokens to zero, effectively not providing any useful information to the VideoLM. Zeroing out P-frame Δ-tokens leads to a clear performance degradation, confirming that the VideoLM actively attends to and utilizes these tokens.

Sampling Strategy PerceptionTest NextQA

1 Keyframe per GOP

+ Δ-tokens = 0 64.41 74.21 + Δ-tokens 67.33 77.37

- H.4 Benefits of Codec Primitives

In Sec. 4.2, we compare our method against LLaVA-Video-7B at matched keyframe counts. To disentangle the contribution of Δ-tokens at inference from the effect of training with codec primitives, we evaluate our model using only I-frames (Tab. H.4). Two findings emerge. First, our model with I-frames only already outperforms LLaVA-Video-7B at the same keyframe density, indicating that fine-tuning with interleaved codec primitives teaches the model stronger temporal representations that transfer even when Δ-tokens are absent at inference. Second, adding Δ-tokens on top consistently improves accuracy across all benchmarks, confirming that codec primitives provide complementary temporal signals beyond what the model internalizes during training. Together, these results show that codec-aware training and Δ-token inference are both independently beneficial and additive.

- Table H.4 Controlled ablation: codec-aware training with and without Δ-tokens. We evaluate our model using only I-frames at each keyframe density, removing Δ-tokens at inference to isolate their contribution. Even without Δ-tokens, our model outperforms LLaVA-Video at matched keyframe counts, showing that fine-tuning with codec primitives alone strengthens temporal reasoning. Adding Δ-tokens consistently improves accuracy further, confirming they provide complementary temporal signals at inference.

Sampling

PerceptionTest NextQA ActNet-QA TempCompass TOMATO

val mc test test-mc test

- 1 keyframe / GOP 63.1 78.0 61.8 60.3 22.4

+ 7 P-frames / GOP 65.5 78.3 62.3 62.4 26.2

- 2 keyframes / GOP 66.2 79.4 62.5 64.5 26.5

+ 6 P-frames / GOP 68.7 80.4 63.6 65.6 27.1 4 keyframes / GOP 68.1 80.8 63.7 66.2 26.9 + 4 P-frames / GOP 70.3 82.1 64.8 68.9 28.4

- H.5 Scaling to Higher Frame Rates

A key advantage of our tokenization is that P-frames are represented with only 𝑁=8 Δ-tokens, compared to 𝑀=196 tokens for each I-frame, enabling higher effective frame rates under a constrained token budget. To study this trade-off, we increase the effective FPS from 1 to 3 by reducing the P-frame fusion window 𝑠 from 30 to 10 frames, thereby encoding finer-grained temporal changes at increasing token cost. We evaluate on TempCompass [93] and MVBench [96], as both benchmarks are sensitive to temporal coverage. As shown in Tab. H.5, both benchmarks improve from 1 to 2 FPS, with MVBench gaining over 1.8% and TempCompass nearly 2%. However, performance slightly decreases at 3 FPS on both benchmarks. We note that our model is trained exclusively with 𝑠=30 (1 FPS); the performance drop at higher frame rates likely reflects a train-test mismatch rather than a fundamental limitation, and could be addressed by training with varied fusion windows. Importantly, these improvements show that our framework enables a practical and flexible trade-off between temporal resolution and token efficiency that would be prohibitive with RGB representations.

Table H.5 Effect of higher frame rates. We increase the effective FPS by reducing the P-frame fusion window 𝑠, encoding finer temporal changes at modest token cost. All configurations use 1 keyframe per GOP. Each I-frame contributes 196+14 tokens (vision + newline), and each P-frame group contributes 8+2 tokens (Δ-tokens + newline), giving (196+14) × 𝐾 + (8+2) × 𝑃 tokens per GOP, where 𝐾 and 𝑃 are the number of I-frames and P-frame groups respectively.

FPS Fusion 𝑠 P-frames / GOP Num. Tokens / GOP TempCompass MVBench

- 1 30 4 880 66.89 59.98
- 2 15 8 1760 68.86 61.87

- 3 10 12 2640 68.73 61.67

- H.6 Next-Frame Retrieval using Δ-Encoder

To assess the representational quality of our compressed-domain features, we evaluate next-frame retrieval at 1 FPS on PerceptionTest [89]. Formally, the task can be defined as: given a query frame 𝐼(𝑡−1) at time 𝑡−1, the task is to identify its true successor frame 𝐼(𝑡) at time 𝑡 from a database containing all frames from the same video (except itself 𝐼(𝑡−1)). As a baseline, SigLIP [41] processes the raw RGB frame 𝐼(𝑡−1), whereas our model uses 𝐼(𝑡−1) together with the current motion vectors 𝜏(𝑡) and residuals 𝛿(𝑡) to produce 𝑁=8 Δ-tokens per P-frame via the Δ-Encoder and lightweight transformer branches used during pre-training. For this experiment, we use the intermediate checkpoint after pre-training, as it is compatible with the “reference” and “warped” transformers that are used to transform the features of the previous frame. The results are shown in Tab. H.6. Our method outperforms the baseline by a

- Table H.6 Retrieval performance at 1 FPS. We evaluate the task of next-frame retrieval. The baseline uses the SigLIP embedding of the current frame, while we use the current frame along with the motion vectors and residuals to next frame, as processed by our Δ-encoder and pre-training transformers. The significantly better performance of our method shows that the information of the codec primitives is correctly compressed in the Δ-tokens.

Recall @1 @2 @5

@1 FPS

SigLIP 11.12 47.11 78.65 Ours - Δ-Encoder 30.09 77.15 94.86

[Figure 69]

- Figure 6 Qualitative Results. Multi-turn QA on a TV show clip requiring understanding of character relationships and narrative events. Our method achieves 8.2× faster TTFT and 76% fewer tokens.

large margin across the board. When retrieving a single frame, the relatively low absolute performance is expected, as subsequent frames are heavily similar even at 1 FPS, but our method successfully leverages the codec primitives to significantly improve the performance. Furthermore, our method achieves almost perfect performance at 5 frames, with its very high recall of 94.86%. These strong improvements over SigLIP confirm that the Δ-Encoder preserves semantically meaningful motion and appearance cues that are critical for retrieval. This experiment also highlights that codec primitives contain rich temporal information that is often lost when sampling sparse keyframes.

### I Qualitative Results

We present qualitative comparisons between CoPE-VideoLM and LLaVA-Video-7B across several videos spanning diverse scenarios: sports with fast motion (hockey, athletics), indoor scenes (TV show, cooking), and outdoor activities (wood chopping). For each example, we report TTFT, E2EL, and token counts for both methods, alongside CoPE-VideoLM’s multi-turn QA responses. Note that from the second turn onward, both methods benefit from KV-cache reuse, which narrows the absolute

latency gap. However, our method still maintains a consistent relative speedup across all turns due to the shorter cached sequence length from fewer tokens. As shown in Figs. 6–8, CoPE-VideoLM produces correct, fine-grained answers while achieving consistent speedups at substantially reduced token counts.

### References

- [1] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. VideoChat: Chat-Centric Video Understanding. arXiv preprint arXiv:2305.06355, 2023. 2, 3
- [2] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.
- [3] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, et al. Languagebind: Extending video-language pretraining to n-modality by languagebased semantic alignment. arXiv preprint arXiv:2310.01852, 2023.
- [4] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data, 2024. URL https://arxiv.org/abs/2410.02713. 2, 3, 6, 8, 9, 10, 11, 14, 15, 16
- [5] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, Peng Jin, Wenqi Zhang, Fan Wang, Lidong Bing, and Deli Zhao. VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding. arXiv preprint arXiv:2501.13106, 2025. 2, 3
- [6] Jie Gao, Simret Araya Gebreegziabher, Kenny Tsu Wei Choo, Toby Jia-Jun Li, Simon Tangi Perrault, and Thomas W Malone. A taxonomy for human-llm interaction modes: An initial exploration. In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems, CHI ’24, page 1–11. ACM, May

2024. doi: 10.1145/3613905.3650786. URL http://dx.doi.org/10.1145/3613905.3650786. 2

- [7] Kento Kawaharazuka, Jihoon Oh, Jun Yamada, Ingmar Posner, and Yuke Zhu. Vision-language-action models for robotics: A review towards real-world applications. IEEE Access, 13:162467–162504, 2025. doi: 10.1109/ACCESS.2025.3609980. 2
- [8] Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. A survey on vision-language-action models for embodied ai, 2025. URL https://arxiv.org/abs/2405.14093. 2
- [9] Zhenyang Liu, Yongchong Gu, Sixiao Zheng, Yanwei Fu, Xiangyang Xue, and Yu-Gang Jiang. Trivla: A triple-system-based unified vision-language-action model with episodic world modeling for general robot control, 2025. URL https://arxiv.org/abs/2507.01424. 2
- [10] Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang, Hongxu Yin, Sifei Liu, Song Han, Yao Lu, and Xiaolong Wang. Egovla: Learning vision-language-action models from egocentric human videos, 2025. URL https://arxiv.org/abs/2507.12440. 2
- [11] OpenAI. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774, 2023. 2, 3
- [12] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 3, 10, 11
- [13] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923, 2025. 2, 3

[Figure 70]

[Figure 71]

###### Figure 7 Qualitative Results. An athletics race requiring fine-grained action understanding and an outdoor activity video involving spatial and action reasoning. Our method achieves 4.2−6.2× faster TTFT and 84−86% fewer tokens while producing correct, fine-grained answers.

[Figure 72]

[Figure 73]

###### Figure 8 Qualitative Results. A hockey game involving multi-step temporal reasoning (injury, player identification, causal events) and, a cooking video with questions about actions, ingredients, and fine-grained object details. Our method provides accurate responses at 2.4−3.9× faster TTFT and 82% fewer tokens.

- [14] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision. NeurIPS, 2024. 2
- [15] GeminiTeam. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 2, 3
- [16] Shyamal Buch, Arsha Nagrani, Anurag Arnab, and Cordelia Schmid. Flexible Frame Selection for Efficient Video Reasoning. In CVPR, 2025. 2
- [17] Boyuan Sun, Jiaxing Zhao, Xihan Wei, and Qibin Hou. Llava-scissor: Token compression with semantic connected components for video llms. arXiv preprint arXiv:2506.21862, 2025. 2, 4, 16, 17
- [18] Ming Nie, Dan Ding, Chunwei Wang, Yuanfan Guo, Jianhua Han, Hang Xu, and Li Zhang. Slowfocus: enhancing fine-grained temporal understanding in video llm. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9798331314385. 2
- [19] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. LongVU: Spatiotemporal adaptive compression for long video-language understanding. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 54582–54599. PMLR, 13–19 Jul 2025.
- [20] Chuhan Zhang, Ankush Gupta, and Andrew Zisserman. Temporal query networks for fine-grained video understanding. In Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 2
- [21] Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokula Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, et al. Fastvlm: Efficient vision encoding for vision language models. In CVPR, 2025. 2
- [22] Didier Le Gall. Mpeg: a video compression standard for multimedia applications. Commun. ACM, 34

(4):46–58, April 1991. ISSN 0001-0782. doi: 10.1145/103085.103090. URL https://doi.org/10.1145/ 103085.103090. 2

- [23] Thomas Wiegand, Gary J. Sullivan, Gisle Bjøntegaard, and Ajay Luthra. Overview of the h.264/avc video coding standard. IEEE Trans. Circuits Syst. Video Technol., 13:560–576, 2003. URL https://api. semanticscholar.org/CorpusID:3540699. 2
- [24] Chao-Yuan Wu, Manzil Zaheer, Hexiang Hu, R Manmatha, Alexander J Smola, and Philipp Krähenbühl. Compressed video action recognition. In CVPR, 2018. 2, 4, 5
- [25] Zhengwei Wang, Qi She, and Aljosa Smolic. Team-net: Multi-modal learning for video action recognition with partial decoding. ArXiv, abs/2110.08814, 2021. URL https://api.semanticscholar.org/CorpusID:

239015960. 2, 4

- [26] Yang Jin, Zhicheng Sun, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, Kun Gai, and Yadong Mu. Video-lavit: Unified video-language pre-training with decoupled visual-motional tokenization. In International Conference on Machine Learning, pages 22185–22209, 2024. 2, 4, 10
- [27] Zijia Zhao, Yuqi Huo, Tongtian Yue, Longteng Guo, Haoyu Lu, Bingning Wang, Weipeng Chen, and Jing Liu. Efficient motion-aware video mllm. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24159–24168, 2025. URL https://api.semanticscholar.org/CorpusID:277104804. 2, 4, 6, 10, 11
- [28] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, Ali Vosoughi, Chao Huang, Zeliang Zhang, Pinxin Liu, Mingqian Feng, Feng Zheng, Jianguo Zhang, Ping Luo, Jiebo Luo, and Chenliang Xu. Video understanding with large language

- models: A survey. IEEE Transactions on Circuits and Systems for Video Technology, pages 1–1, 2025. doi: 10.1109/TCSVT.2025.3566695. 3
- [29] Xiao Wang, Guangyao Chen, Guangwu Qian, Pengcheng Gao, Xiao-Yong Wei, Yaowei Wang, Yonghong Tian, and Wen Gao. Large-scale multi-modal pre-trained models: A comprehensive survey. Machine Intelligence Research, 20(4):447–482, 2023. ISSN 2731-538X. doi: 10.1007/s11633-022-1410-8. URL https://www.mi-research.net/en/article/doi/10.1007/s11633-022-1410-8.
- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. NeurIPS, 2023.
- [31] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved Baselines with Visual Instruction Tuning. arXiv preprint arXiv:2310.03744, 2023.
- [32] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. LLaVANeXT: Improved reasoning, OCR, and world knowledge, 2024. URL https://llava-vl.github.io/blog/ 2024-01-30-llava-next/.
- [33] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy Visual Task Transfer. arXiv preprint arXiv:2408.03326,

2024. 10, 11

- [34] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. URL https://arxiv.org/abs/2406.16852. 10, 11
- [35] Pan Zhang, Xiaoyi Dong, Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Wenwei Zhang, Hang Yan, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition. arXiv preprint arXiv:2309.15112, 2023.
- [36] Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, Songyang Zhang, Wenwei Zhang, Yining Li, Yang Gao, Peng Sun, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Hang Yan, Conghui He, Xingcheng Zhang, Kai Chen, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. arXiv preprint arXiv:2407.03320, 2024. 10, 11
- [37] Shimin Chen, Xiaohan Lan, Yitian Yuan, Zequn Jie, and Lin Ma. Timemarker: A versatile video-llm for long and short video understanding with superior temporal localization ability. arXiv preprint arXiv:2411.18211,

2024. 11

- [38] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 11, 16
- [39] Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, Serena Yeung-Levy, and Xide Xia. Apollo: An exploration of video understanding in large multimodal models. arXiv preprint arXiv:2412.10360, 2024. 3, 10, 11
- [40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021. URL https://api.semanticscholar.org/CorpusID:231591445. 3
- [41] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 11941–11952,

2023. URL https://api.semanticscholar.org/CorpusID:257767223. 3, 8, 19

- [42] Qiming Zhang, Yufei Xu, Jing Zhang, and Dacheng Tao. Vsa: learning varied-size window attention in vision transformers. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXV, pages 466–483. Springer, 2022. 3

- [43] Qiming Zhang, Jing Zhang, Yufei Xu, and Dacheng Tao. Vision transformer with quadrangle attention. arXiv preprint arXiv:2303.15105, 2023. 3
- [44] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. ArXiv, abs/2204.14198, 2022. URL https://api.semanticscholar.org/CorpusID:248476411. 3
- [45] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aur’elien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and Efficient Foundation Language Models. arXiv preprint arXiv:2302.13971, 2023. 3
- [46] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-ajudge with MT-Bench and Chatbot Arena. arXiv preprint arXiv:2306.05685, 2023. 3
- [47] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. arXiv preprint arXiv:2308.12966, 2023. 3, 4
- [48] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. arXiv preprint arXiv:2409.12191, 2024. 3, 16
- [49] Hang Zhang, Xin Li, and Lidong Bing. Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding. arXiv preprint arXiv:2306.02858, 2023. 3
- [50] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024. 3, 4
- [51] De-An Huang, Shijia Liao, Subhashree Radhakrishnan, Hongxu Yin, Pavlo Molchanov, Zhiding Yu, and Jan Kautz. Lita: Language instructed temporal-localization assistant. In ECCV, 2024.
- [52] Long Qian, Juncheng Li, Yu Wu, Yaobo Ye, Hao Fei, Tat-Seng Chua, Yueting Zhuang, and Siliang Tang. Momentor: Advancing video large language model with fine-grained temporal reasoning, 2024.
- [53] Yueqian Wang, Xiaojun Meng, Jianxin Liang, Yuxuan Wang, Qun Liu, and Dongyan Zhao. Hawkeye: Training video-text llms for grounding text in videos, 2024. 3
- [54] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding. arXiv, abs/2312.02051, 2023. 3
- [55] OpenAI. GPT-5 system card, 2025. URL https://openai.com/index/gpt-5-system-card/. 3, 10, 11
- [56] Anthropic. Claude sonnet 4.5 system card, 2025. URL https://assets.anthropic.com/m/ 12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf. 3, 10, 11
- [57] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. VILA: On Pre-training for Visual Language Models, 2023. 3, 10
- [58] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava : Parameter-free llava extension from images to videos for video dense captioning, 2024. 3, 4, 11, 16, 17
- [59] Yanwei Li, Chengyao Wang, and Jiaya Jia. LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models. In ECCV, 2024. 3

- [60] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your ViT but faster. In ICLR, 2023. 3
- [61] Peng Jin, Ryuichi Takanobu, Caiwan Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. arXiv preprint arXiv:2311.08046, 2023. 4
- [62] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, Yu Qiao, Yali Wang, and Limin Wang. Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574, 2024. 3
- [63] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, 2023. URL https://api.semanticscholar.org/CorpusID:256390509. 4
- [64] Shaolei Zhang, Qingkai Fang, Zhe Yang, and Yang Feng. Llava-mini: Efficient image and video large multimodal models with one vision token, 2025. URL https://arxiv.org/abs/2501.03895. 4
- [65] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800,

2024. 4

- [66] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Xun Guo, Tian Ye, Yan Lu, Jenq-Neng Hwang, et al. Moviechat: From dense token to sparse memory for long video understanding. arXiv preprint arXiv:2307.16449, 2023. 4
- [67] Enxin Song, Wenhao Chai, Tian Ye, Jenq-Neng Hwang, Xi Li, and Gaoang Wang. Moviechat+: Questionaware sparse memory for long video question answering. arXiv preprint arXiv:2404.17176, 2024. 4
- [68] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models, 2024. 4, 16, 17
- [69] Tianyu Fu, Tengxuan Liu, Qinghao Han, Guohao Dai, Shengen Yan, Huazhong Yang, Xuefei Ning, and Yu Wang. Framefusion: Combining similarity and importance for video token reduction on large visual language models. arXiv preprint arXiv:2501.01986, 2024.
- [70] Long Xing, Qidong Huang, Xiao wen Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, and Dahua Lin. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. ArXiv, abs/2410.17247, 2024. URL https://api.semanticscholar. org/CorpusID:273507889.
- [71] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In International Conference on Machine Learning, 2025. 4
- [72] Kele Shao, Keda Tao, Kejia Zhang, Sicheng Feng, Mu Cai, Yuzhang Shang, Haoxuan You, Can Qin, Yang Sui, and Huan Wang. When tokens talk too much: A survey of multimodal long-context token compression across images, videos, and audios. arXiv preprint arXiv:2507.20198, 2025. 4
- [73] Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. Slowfast-llava: A strong training-free baseline for video large language models. arXiv, 2024. 4
- [74] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understanding via large language models. In ECCV, 2024. 4
- [75] Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Dycoke: Dynamic compression of tokens for fast video large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18992–19001, 2025. 4, 16, 17
- [76] Xiao Wang, Qingyi Si, Shiyu Zhu, Jianlong Wu, Li Cao, and Liqiang Nie. AdaReTaKe: Adaptive redundancy reduction to perceive longer for video-language understanding. In Wanxiang Che, Joyce

- Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 5417–5432, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.283. URL https://aclanthology.org/2025.findings-acl.283/. 4
- [77] Yunzhu Zhang, Yu Lu, Tianyi Wang, Fengyun Rao, Yi Yang, and Linchao Zhu. Flexselect: Flexible token selection for efficient long video understanding, 2025. URL https://arxiv.org/abs/2506.00993. 4
- [78] Samuel Felipe dos Santos, Nicu Sebe, and Jurandy Almeida. Cv-c3d: Action recognition on compressed videos with convolutional 3d networks. In 2019 32nd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), pages 24–30, 2019. doi: 10.1109/SIBGRAPI.2019.00012. 4
- [79] Zheng Shou, Xudong Lin, Yannis Kalantidis, Laura Sevilla-Lara, Marcus Rohrbach, Shih-Fu Chang, and Zhicheng Yan. Dmc-net: Generating discriminative motion cues for fast compressed video action recognition, 2019. URL https://arxiv.org/abs/1901.03460. 4
- [80] Marton Havasi, Rodolphe Jenatton, Stanislav Fort, Jeremiah Zhe Liu, Jasper Snoek, Balaji Lakshminarayanan, Andrew M. Dai, and Dustin Tran. Training independent subnetworks for robust prediction. In International Conference on Learning Representations, 2021.
- [81] Hayato Terao, Wataru Noguchi, Hiroyuki Iizuka, and Masahito Yamamoto. Efficient compressed video action recognition via late fusion with a single network. In ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2023. doi: 10.1109/ICASSP49357.2023.10096477. 4
- [82] Jiawei Chen and Chiu Man Ho. Mm-vit: Multi-modal video transformer for compressed video action recognition. 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 786–797,

2021. URL https://api.semanticscholar.org/CorpusID:237266685. 4

- [83] Shristi Das Biswas, Efstathia Soufleri, Arani Roy, and Kaushik Roy. Towards scalable modeling of compressed videos for efficient action recognition, 2025. URL https://arxiv.org/abs/2503.13724. 4, 5
- [84] Yazhou Xing, Yang Fei, Yingqing He, Jingye Chen, Jiaxin Xie, Xiaowei Chi, and Qifeng Chen. Videovae+: Large motion video autoencoding with cross-modal video vae. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. 4
- [85] Dan Grois, Alex Giladi, Kiho Choi, Min Woo Park, Yinji Piao, Min Woo Park, and Kwang Pyo Choi. Performance comparison of emerging evc and vvc video coding standards with hevc and av1. SMPTE Motion Imaging Journal, 130:1–12, 2021. URL https://api.semanticscholar.org/CorpusID:235569476. 4
- [86] Thorsten Laude, Yeremia Gunawan Adhisantoso, Jan Voges, Marco Munderloh, and Jörn Ostermann. A comprehensive video codec comparison. APSIPA Transactions on Signal and Information Processing, 8:e30,

2019. doi: 10.1017/ATSIP.2019.23. 4

- [87] Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2015. URL https://api.semanticscholar.org/CorpusID:206594692. 7, 14
- [88] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. URL https://arxiv.org/abs/2407.10671. 8

- [89] Viorica Pătrăucean, Lucas Smaira, Ankush Gupta, Adrià Recasens Continente, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alex Frechette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and João Carreira. Perception test: A diagnostic benchmark for multimodal video models. In Advances in Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=HYEGXFnPoq. 8, 9, 14, 17, 19
- [90] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 8, 9, 14, 17
- [91] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In AAAI, pages 9127–9134, 2019. 8, 14, 17
- [92] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 8, 14, 15
- [93] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv: 2403.00476, 2024. 8, 10, 14, 19
- [94] Ziyao Shangguan, Chuhan Li, Yuxuan Ding, Yanan Zheng, Yilun Zhao, Tesca Fitzgerald, and Arman Cohan. Tomato: Assessing visual temporal reasoning capabilities in multimodal foundation models, 2024. URL https://arxiv.org/abs/2410.23266. 8, 10, 14
- [95] Muhammad Uzair khattak, Muhammad Ferjad Naeem, Jameel Hassan, Naseer Muzzamal, Federico Tombari, Fahad Shahbaz Khan, and Salman Khan. How good is my video lmm? complex video reasoning and robustness evaluation suite for video-lmms. arXiv:2405.03690, 2024. 8, 10, 15
- [96] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024. 8, 11, 15, 19
- [97] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding, 2024. URL https://arxiv.org/abs/2407.15754. 8, 11, 15
- [98] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Shiyu Huang, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Lvbench: An extreme long video understanding benchmark, 2024. 8, 11, 15
- [99] Yuanhan Zhang, Yunice Chew, Yuhao Dong, Aria Leo, Bo Hu, and Ziwei Liu. Towards video thinking test: A holistic benchmark for advanced video reasoning and understanding, 2025. URL https://arxiv.org/ abs/2507.15028. 8, 11, 15
- [100] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Videommmu: Evaluating knowledge acquisition from multi-discipline professional videos. 2025. URL https: //arxiv.org/abs/2501.13826. 8, 11, 15
- [101] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In CVPR, pages 19129–19139, 2022. 8, 15, 16
- [102] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. Sqa3d: Situated question answering in 3d scenes. In ICLR, 2023. 8, 15, 16
- [103] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models. arXiv preprint arXiv:2407.12772, 2024. 8

- [104] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024. 8, 10
- [105] Google. Gemini 3 Pro model card, 2025. URL https://storage.googleapis.com/deepmind-media/ Model-Cards/Gemini-3-Pro-Model-Card.pdf. 10, 11
- [106] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, and Lidong Bing. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. URL https://arxiv.org/abs/2406.

07476. 11

- [107] Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu, and Hui Wang. Video-ccam: Enhancing video-language understanding with causal cross-attention masks for short and long videos, 2024. URL https://arxiv.org/abs/2408.14023. 11
- [108] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. arXiv preprint arXiv:2408.15542, 2024. 11
- [109] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models,

2024. URL https://arxiv.org/abs/2408.04840. 11

- [110] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. arXiv preprint arXiv:2311.12871, 2023. 16
- [111] Zhao Jin, Munawar Hayat, Yuwei Yang, Yulan Guo, and Yinjie Lei. Context-aware alignment and mutual masking for 3d-language pre-training. In CVPR, pages 10984–10994, 2023. 16
- [112] Ziyu Zhu, Xiaojian Ma, Yixin Chen, Zhidong Deng, Siyuan Huang, and Qing Li. 3d-vista: Pre-trained transformer for 3d vision and text alignment. In ICCV, pages 2911–2921, 2023. 16
- [113] Zehan Wang, Haifeng Huang, Yang Zhao, Ziang Zhang, and Zhou Zhao. Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes. arXiv preprint arXiv:2308.08769, 2023. 16
- [114] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. NeurIPS, 36:20482–20494, 2023. 16
- [115] Rao Fu, Jingyu Liu, Xilun Chen, Yixin Nie, and Wenhan Xiong. Scene-llm: Extending language model for 3d visual understanding and reasoning. arXiv preprint arXiv:2403.11401, 2024. 16
- [116] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning. In CVPR, pages 26428–26438, 2024. 16
- [117] Jiawei Zhang, Chejian Xu, and Bo Li. Chatscene: Knowledge-enabled safety-critical scenario generation for autonomous vehicles. In CVPR, pages 15459–15469, 2024. 16
- [118] Yilun Chen, Shuai Yang, Haifeng Huang, Tai Wang, Ruiyuan Lyu, Runsen Xu, Dahua Lin, and Jiangmiao Pang. Grounded 3d-llm with referent tokens. arXiv preprint arXiv:2405.10370, 2024. 16
- [119] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. arXiv preprint arXiv:2409.18125, 2024. 16
- [120] Duo Zheng, Shijia Huang, and Liwei Wang. Video-3d llm: Learning position-aware video representation for 3d scene understanding. arXiv preprint arXiv:2412.00493, 2024. 16
- [121] Haochen Wang, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, and Zhaoxiang Zhang. Ross3d: Reconstructive visual instruction tuning with 3d-awareness. arXiv preprint arXiv:2504.01901,

2025. 16

###### [122] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. arXiv preprint arXiv:2412.04467, 2024. 16, 17

