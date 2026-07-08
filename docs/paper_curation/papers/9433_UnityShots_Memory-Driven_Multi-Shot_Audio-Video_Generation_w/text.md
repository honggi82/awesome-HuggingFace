[Figure 1]

## UnityShots: Memory-Driven Multi-Shot Audio-Video Generation with Boundary-Aware Gating

Jiehui Huang1,† Yuechen Zhang2 Bin Xia2 Jiahao Wang3 Xu He4 Zhenchao Tang5 Meng Chu1 Xin Tao3 Pengfei Wan3 Jiaya Jia1

1The Hong Kong University of Science and Technology 2The Chinese University of Hong Kong 3Kling Team, Kuaishou Technology 4Tsinghua University 5Sun Yat-sen University https://jackailab.github.io/Projects/UnityShots

UnityShots (Ours, Video , Audio)

# arXiv:2606.21661v1[cs.CV]19Jun2026

Fixed-window Baseline Method (no Audio)

Image to Video Reference Identity to Video

Fixed-window Target prediction video (Last N Frames)

Current Shot (New Context)

Window Token (Recent Context)

Sink Token (Global Context)

Video Memory

…

[Figure 2]

…

…

…

Time

Time

Keyframe Pool Method (no Audio)

Current Shot (New Context)

Ref Audio Token (Global Context)

    (

    (

    (

    (

Audio Memory

…

    (          

…

… …

    (

    (

    (          

    (          

    (

    (

    (          

    (          

    (

    (

    (          

    (          

Time Time

    (

    (

    (

    (

    (          

    (          

    (          

    (           

    (           

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

    (           

    (           

|Shot 1|
|---|

|Shot 2|
|---|

|Shot 3|
|---|

|Shot 4|
|---|

|Shot 5|
|---|

|Shot 6|
|---|

Snow Mountain Fox, Sauce Duck Revenge

[1] In a retro wuxia parody, a wandering swordsman saves a dying snow fox by leaving it a spicy sauce duck. [2-3] A year later, a mysterious white-robed woman arrives. [4] The swordsman assumes she is the grateful fox spirit. [5] In a hilarious twist, she reveals she is actually the abandoned sauce duck seeking revenge. [6] They awkwardly dine together instead.

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

|Shot 3|
|---|

|Shot 1|
|---|

|Shot 2|
|---|

|Shot 4|
|---|

|Shot 5|
|---|

Tim Cook's retirement life in Hua Qiang Bei

[1] In a humorous parody, Tim Cook brings his minimalist CEO aesthetic to a chaotic electronics repair stall. [2] He repairs phone parts with ceremonial precision. [3] Using classic keynote charm, he presents an elegant price card to a haggling customer. [4] After returning the device on a velvet pad, [5] he meticulously aligns his tools to close shop, delivering a classic "One more thing."

Figure 1. UnityShots generates consistent multi-shot audio-video sequences across diverse cultural contexts. Each row shows shots from the proposed multi-cultural benchmark; the reference identity portrait (leftmost) is preserved through hard cuts by the dual-slot memory bank and reference audio anchor. Upper left: UnityShots pairs a long-term (LTM) slot anchored to the opening shot with a shortterm (STM) slot capturing the preceding tail, plus a cross-shot speaker audio anchor, while prior end-to-end and shot-by-shot methods rely on single-window context or no persistent cross-shot conditioning.

### Abstract

and a short-term memory (STM) slot holding the immediately preceding tail, both updated at every cut by a boundary-conditioned gate that fuses visual cut probability and beat-tracker signals. The audio stream injects a reference speaker token at every shot to preserve vocal timbre without a sliding audio bank. A discrete cut-type prior, learned through AdaLN, becomes an inference-time control knob over transition strength. We release a benchmark of 200 multi-cultural multi-shot sequences spanning six ethnic regions and ten or more languages, with per-shot reference identities, reference audio, and per-boundary transition labels. Evaluated across I2V, T2V, and R2V conditioning modes, UnityShots leads open-source baselines on every cross-shot coherence metric and matches the strongest closed-source system on the multi-shot axes. Code and data can be found at: https://github.com/JIA-Labresearch/UnityShots

Generating a coherent multi-shot video requires structured cross-shot memory. Subject appearance, scene context, and speaker identity must persist across cuts. Existing approaches either train end-to-end over fixed-length sequences and cannot scale, generate shot-by-shot with memory banks that grow linearly, or orchestrate pretrained generators under an LLM planner without a multi-shotaware backbone. We present UnityShots, a memory-driven multi-shot audio-video generation system built on LTX2.3, trained on annotated cinematic and music-video shots. The video stream maintains two fixed-size slots, a longterm memory (LTM) slot anchored to the opening shot

† This work was conducted during the author’s internship at Kling Team, Kuaishou Technology.

Corresponding Author.

### 1. Introduction

Coherent multi-shot video generation requires explicit cross-shot memory, because subject appearance, scene context, and speaker identity must remain stable across shot boundaries.

Existing multi-shot approaches fall into three families, each with a structural weakness. The end-to-end family trains over an entire multi-shot sequence as one denoising pass [12, 20, 29, 43, 46], with closed-source systems such

- as Kling [22] and Seedance [6] taking the same direction
- at much larger scale. Sequence length is bounded by GPU memory and cost grows with the total clip, so this family cannot scale to long-form storytelling. The shot-by-shot family generates each shot conditioned on prior [28, 51], which removes the length cap but introduces either drift once the conditioning window slides past the opening shot or a keyframe bank whose memory grows linearly with the sequence. The agent-orchestrated family pipelines pretrained generators under an LLM planner [23, 24, 56], but the underlying generator is either not trained for cross-shot coherence (making multi-shot signals hard to propagate) or accessible only through closed-source APIs.

Despite advances in all three families, current designs treat memory as a monolithic context window and do not distinguish between what happened long ago and what happened just before the current cut. In practice, the next shot’s content should integrate both: the opening-shot identity that anchors the whole narrative and the recent motion and scene state that determines local continuity. During long sequences, characters and environments change substantially across shots, so relying entirely on distant history produces drift, while relying only on the most recent clip loses the identity anchor. A second gap is equally important: in real filmmaking and music-video production, the boundary between shots is driven jointly by the magnitude of visual change and the rhythm of the audio track. No existing audio-video generation model treats shot-boundary strength as a joint function of visual and musical signals, nor adjusts its memory dynamics accordingly. How to combine audio and visual boundary cues to control long- and short-term memory in a unified generative model remains an open problem.

We propose UnityShots, a memory-driven multi-shot audio-video generation system built on LTX-2.3 [13]. Two fixed-size video memory slots—a long-term slot anchored to the opening shot and a short-term slot holding the immediately preceding tail—are both updated by a boundaryconditioned gate that fuses visual cut probability and beattracker signals at every shot boundary. To support this design, we construct a large-scale training set of 146k annotated cinematic and music-video clips, covering shot-level captions, speaker diarisation, character anchoring, and transition labels. The model is trained under three condition-

ing paradigms—image-to-video (I2V), text-to-video (T2V), and reference-identity-to-video (R2V)—so it can be steered by a visual first frame, a text prompt alone, or a reference portrait, making it suitable for deployment in both manual and agent-driven production pipelines. A companion agent system, released alongside the model, exposes per-shot generation as composable tools that an external planner can invoke over sequences of arbitrary length.

- • Memory-driven multi-shot dual-stream generation. An end-to-end-trained I2V/T2V/R2V backbone with a two-slot video memory bank, a reference speaker anchor, and a cut-type-aware AdaLN prior that gives users explicit control over transition strength at inference time.
- • Boundary-conditioned memory gating. A rule-based long-term gate and a content-adaptive short-term gate update the memory bank at every cut, jointly driven by visual and audio boundary signals.
- • Multi-cultural multi-shot benchmark. 200 sequences spanning six cultural groups and ten or more languages, each with reference identity images, reference audio, pershot first frames, narrative captions, and per-boundary transition labels.

### 2. Releated Work

#### 2.1. Multi-shot video generation and memory

Video diffusion models [4, 16, 17, 34] scaled to longer clips; open-source systems [49, 55] and large-scale models [5, 21, 32, 41] push toward minute-scale generation, but none model shot boundaries as discrete events.

End-to-end approaches [6, 12, 20, 22, 29, 43, 46, 47] train a single denoising pass over the full shot sequence, producing coherent clips but bounding sequence length by GPU memory and training context. Cinematic design methods [42, 48] extend controllability with 3D-aware scene composition and per-shot camera specification. Shot-byshot approaches generate each shot conditioned on prior content, lifting the length cap. FilmWeaver [27] keeps an autoregressive cache over all prior shots and STAGE [53] stores per-entity memory tied to storyboard anchors, but both grow linearly with the sequence. OneStory [1] and StoryMem [51] select representative keyframes by appearance similarity, while Cut2next [14] generates the next shot via in-context tuning on the preceding clip. Adapter-based methods [7, 18, 28, 50, 57] propagate identity features across shots without an explicit temporal anchor, so identity drift accumulates once the conditioning window slides past the opening shot.

Persistent memory for sequential generation has been studied in language modeling and video autoregression [52], where small fixed-size token sets at reserved positions act as stable global attractors. Our bank is recurrent at O(1) cost, updated by a joint audio-visual boundary sig-

nal rather than retrieval, and covers both video and audio streams. Figure 2 (upper left) provides a visual comparison with prior designs.

#### 2.2. Joint audio-video generation

Dual-stream architectures [11, 13] show that joint training of video and audio improves multimodal alignment. DreamID-Omni [11] achieves reference-based audio-video generation with identity disentanglement but targets single shots and maintains no cross-shot memory. FoleyCrafter [54] demonstrates video-conditioned ambient audio synthesis from text-to-audio models [25]. None of these approaches maintain a persistent cross-shot speaker anchor; voice consistency in multi-shot generation relies on per-shot prompting and degrades when a reference clip is absent.

#### 2.3. Agent-orchestrated production

Agent-orchestrated pipelines [23, 24, 56] wrap pretrained generators with LLM planners for multi-step video creation, a paradigm gaining commercial traction in shortdrama and music-video production and increasingly viewed as a practical interface for controllable generative workflows. The structural limit is that the underlying generator was not trained for cross-shot coherence, so the planner cannot propagate multi-shot signals cleanly, or it relies on closed-source paid APIs. UnityShots is an end-to-endtrained multi-shot backbone compatible with agent-driven workflows, and a companion agent system exposes per-shot generation as composable tools.

### 3. Method

UnityShots takes a sequence of per-shot prompts together with a reference image and reference audio, then generates a synchronized multi-shot video-audio clip in which each shot is denoised conditioned on a structured summary of everything before it. The backbone is LTX-2.3 [13], a 22B dual-stream diffusion transformer [31] that jointly denoises video and audio latents. Video tokens use a BlockRelativistic 3D RoPE [37] with temporal ceiling flim = 128 indices (≈ 42s at the native frame rate); audio tokens use a separate 1-D RoPE with ceiling flima = 1024. Each shot is generated in image-to-video (I2V) mode, with the first frame as the image condition, and Pv = 4 video frames denote the tail window used to seed memory from a completed shot. Given K shots with prompts c0,...,cK−1, a reference identity image Xref, and a reference audio clip Aref, the goal is a multi-shot sequence that maintains consistent subject appearance and speaker identity across every cut—a problem the backbone has no native mechanism to solve when shots are generated independently.

#### 3.1. Dual-Stream Memory Bank

Video memory bank. Visual content changes abruptly at every cut: new scene, new camera angle, new action phase. Maintaining identity across cuts therefore requires both a stable long-range anchor and a fine-grained record of the immediately preceding shot. The video bank holds two fixed-size slots that serve these two roles. The longterm memory (LTM) slot LN ∈ R2×C

v uses two latent frames to anchor subject identity to the opening shot. Its contribution follows a rule-based coefficient (Section 3.2) that is bounded away from zero at all cut strengths, so the anchor is never fully erased regardless of what the perframe detector reports. The short-term memory (STM) slot SN ∈ RP

v×Cv uses Pv = 4 latent frames from the immediately preceding tail to capture action-phase and motion dynamics at the boundary. Its weight is refined by a small learned MLP (Section 3.2) conditioned on outgoing and incoming shot content, so the boundary strength adapts to scene content rather than relying on a fixed schedule.

Audio reference anchor. For audio, the cross-shot coherence requirement is more targeted: what must be preserved is the voice timbre of each character named in the script, not the full spectral trajectory of the audio track across shots. We satisfy this by injecting a fixed reference speaker token derived from Aref at every shot; no sliding audio memory window is maintained. Background music is a structurally separate problem: a musical track evolves continuously and is not tied to visual edit decisions, so when an application needs accompaniment we treat it as an external track aligned to the finished video rather than content that must be co-generated with the visual stream.

Conditioning sequence. At shot N, the token sequence passed to the backbone is

CN = Xref, LN, SN, Aref, VtN ⊕ ANt , (1)

where VtN ⊕ANt is the noisy video-audio latent at diffusion timestep t. The denoising loss is masked to the final block only, so the model never predicts memory tokens. All five blocks are concatenated along the token dimension and processed by the shared dual-stream attention [40] in a single forward pass.

#### 3.2. Boundary-Conditioned Memory Gating

Cut-type prior and boundary score. Each training shot is labeled with a discrete transition type τN ∈ {FIRST, CONTINUE, HARD} with priors {0,0.4,1.0}. A small embedding of τN is fused with the diffusion timestep and injected into every DiT block via AdaLN; at inference, setting τN directly controls how strongly the next cut resets the memory. Within each transition class, we compute a

In: Clean history shots’ token as condition Out: Refresh history shots token based on boundary signal

###### Sec 3.2 : Dual-Stream Memory Bank

[Figure 17]

###### Sec 3.3 : Boundary-Conditioned Memory Gating

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Memory Update Short-term Token w/ Learn-based Gate Long-term Token w/ Rule-based Gate

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

𝑔

[Figure 27]

𝑔

[Figure 28]

Learn-based Gate (MLP)

Short-term Slot (Recent Context)

Long-term Slot (Global Context)

Rule-based

Gate (Coefficient) Sec 3.2

Video Shot N-1

Sec 3.2

Short-term Memory

Video Shot 1~N-2

##### Video-Audio DiT Blocks

3D VAE

Long-term Memory

Sec 3.4 : Strata-RoPE : Depth-stratified RoPE bands

[Figure 29]

…

Long-term Zone (GMR EMA Mix)

Sink Clean Token

Unused Vision RoPE (Optional for Ref Vision ID)

[Figure 30]

VisionRoPE 𝐹

[Figure 31]

1 2 3 … 𝐹

𝐹

[𝐹 + 1,…,𝐹 − 2𝑃 − 1]

[Figure 32]

Window Clean Token

Short-term Zone

… (Raw Shot N-1 Tail)

…

Target Noisy Token

Audio RoPE

1 2 3 … 𝐹 𝐹 + 1 𝐹 + 2 … 𝐹 + 𝐹

[𝐹 + 𝐹 + 1,…,𝐹 − 2𝑃 − 1] Ref Audio ID RoPE Index Unused Audio RoPE Index

AdaLN for Shot Cuts

Video Shot N as Target Shot

Current Shot Zone (Denoising Target)

Audio Token

Video Token

- Figure 2. UnityShots architecture. Left: Strata-RoPE partitions the temporal band of the video stream into three non-overlapping strata for the long-term memory (LTM) slot, the short-term memory (STM) slot, and the current shot. Right: at each shot boundary the composite

signal bN produces two gating coefficients gltm and gstm that update the two video memory slots before the dual-stream DiT denoises the current shot; the audio stream injects a fixed reference speaker token at every shot without memory gating.

tokens or side networks. We assign each video block a fixed stratum along the temporal RoPE axis:

continuous boundary score

bN = τN · α svis + β saud + γ sbeat , (2)

current shot : [0, Tv − 1], STM band : [flim − Pv, flim − 1], LTM band : [flim − 2Pv, flim − Pv − 1].

where svis is the TransNetv2 [36] visual-cut probability, saud an RMS energy-change score, and sbeat ∈ {0.25,0.5,1.0} encodes beat position from a beat tracker [10]. The prior acts as an upper envelope: a CONTINUE shot never over-resets the LTM anchor even if the visual detector misfires.

(4)

The current shot and the memory strata are far apart on the integer axis, so the rotary kernel attenuates cross-strata interaction in proportion to distance automatically. The audio stream uses standard reference-token injection and does not need a multi-tier RoPE layout, since a single anchor at every shot has no prior audio content to separate from positionally. This differs from the negative-offset keyframe layout of StoryMem [51], which encodes the absolute time of a memory frame. Our encoding marks which stratum a token occupies, a design suited to the recurrent structure of the memory bank.

Memory update. From the clipped boundary score ¯bN = clamp(bN,0,1), we derive two monotone-increasing gating coefficients gltm(¯bN) and gstm(¯bN) on non-overlapping ranges (values given in Section 4.1). The LTM coefficient is bounded strictly above zero, while the STM coefficient reaches near-full weight at a hard cut to bridge the boundary. Since the LTM slot contains only Pℓ = 2 latent frames, we use hℓ(·) to select the last Pℓ latent frames from the previous-shot tail before writing to LTM. Both slots are updated before each denoising pass:

#### 3.4. Training and Inference

LN ← zN hℓ(VtailN−1)+(1−zN)LN−1, SN ← VtailN−1,

Training. Stage 1 fine-tunes the backbone on single shots with the reference block only. Stage 2 adds both memory slots and boundary-conditioned gating, trains on k-shot chunks (k ∈ [3,9]) in mixed conditioning mode: imageto-video (p = 0.5), text-to-video (p = 0.3), and referenceidentity-to-video (p=0.2). The recurrence is unrolled over the chunk; the diffusion loss [15]

(3) with zN = zmax¯bN and zmax < 1, then scaled by gltm(¯bN) and gstm(¯bN) before entering Equation 1. This bounded update prevents the opening-shot anchor from being abruptly overwritten.

Content-aware refinement. A two-layer MLP (< 50K parameters) adds a content-aware correction to gstm: it takes pooled features from the outgoing and incoming shots together with the transition embedding and produces a multiplicative factor, clamped to [0.5,1.0] for the first 500 training steps to prevent collapse. The LTM gate stays rulebased as shot 0 has no prior content to condition on.

L = EVN, t,ϵ ∥ϵ − ϵθ(CN,t)∥22 (5)

is masked to the final shot only. The audio branch is finetuned jointly but does not use the cross-shot memory bank. The only cross-shot audio signal is the reference speaker token (Section 3.1). Random reference-audio dropout (p= 0.1) allows text-only fallback.

#### 3.3. Strata-RoPE Position Encoding

Inference. For each shot in sequence, the STM slot receives the Pv-frame tail of the previous shot, the LTM slot is up-

Attention must distinguish LTM content, STM content, and the current noisy shot from each other without extra depth

dated by Eq. 3, both are modulated by their gating coefficients, and the backbone denoises the current shot conditioned on Eq. 1. The R2V variant, which replaces the pershot first frame with a concatenated reference identity token, relaxes the first-frame conditioning constraint and enables more flexible generation.

### 4. Experiment

#### 4.1. Implementation Details

Both training stages use AdamW with learning rate 1×10−5 (gate MLP: 5 × 10−5), gradient clipping at 1.0, BFloat16 precision, and DeepSpeed [33] on 4 nodes of 8 NVIDIA A800 GPUs each. Stage 1 fine-tunes for 20,000 steps on single-shot identity data. Stage 2 trains on multishot chunks with variable length k ∈ [3,9] sampled by a bucketed-length sampler, in mixed conditioning mode (image-to-video p = 0.5, text-to-video p = 0.3, referenceidentity-to-video p = 0.2); 30% of samples use a subtitlefree variant to match the base model’s distribution. Evaluation uses DDIM [35]-based inference with 50 steps.

The boundary-conditioned coefficients of Section 3.2 are set to gltm(b) = 0.1 + 0.6b and gstm(b) = 0.3 + 0.7b, selected by a small grid search over {0.0,0.1,0.2} × {0.4,0.6,0.8} on a held-out validation split. Detailed hyperparameters are listed in Appendix A.

We compare four configurations to isolate each design axis: LTX-2.3 (no memory) runs the unmodified backbone independently per shot; +STM only adds just the shortterm memory slot with the boundary-conditioned gating but no long-term anchor; +LTM only adds just the longterm memory slot with the boundary-conditioned gating but no short-term window; UnityShots (full) uses both memory slots together with the full boundary-conditioned gating from Section 3.2 and the reference speaker anchor for audio.

#### 4.2. Datasets, Benchmark, and Metrics

Training data. Stage 2 uses 116k cinematic clips (segmented by TransNetv2 [36]) and 30k music-video clips (segmented by a beat tracker [10] with beat-strength labels), totaling 146k shots.

Evaluation benchmark. We release 200 multi-shot sequences from a multi-cultural short-drama pipeline covering six cultural groups and ten or more languages (Figure 7). Each sequence has 3–9 shots with unique story outlines, reference identity images, reference audio clips, per-shot first frames, and per-boundary transition labels. Full construction details are in Appendix C (Figure 6).

Metrics. TA: text-video alignment scored by VBench [19] using ViCLIP [44] (×100). TSIM: DINOv2 [30] cosine similarity between consecutive-shot frames, following the

inter-shot consistency protocol of StoryMem [51]. AESV/AES-A: VBench visual aesthetics and Audiobox [39] audio aesthetics. CLAP [9]: audio–caption similarity averaged across shots. NC, Story, Char, Pace, Cult: Gemini-2.5-Pro 1–5 ratings (8 frames/sequence, temperature=0); Cult uses a separate prompt conditioned on per-shot dialogue and character ethnic context.

#### 4.3. Qualitative Results

Short sequences (3 and 5 shots) against open-source baselines. As shown in Fig. 3, on 3-shot and 5-shot I2V sequences, UnityShots(I2V) generates each shot with higher fidelity to the per-shot text prompt than open-source baselines while maintaining stable subject identity across cuts. LTX-2.3 without memory and ID-LoRA both accumulate visible drift from shot 2 onward: identity attributes such as hairline, clothing, and skin tone shift relative to the reference portrait even when the prompt is unchanged.

Comparison against the closed-source Kling. Kling demonstrates superior single-shot visual quality from largescale training, but exhibits cross-shot consistency gaps on sequences requiring consistent subject framing under different camera angles. UnityShots leads on narrative coherence (NC) and cinematic pacing (Pace) because the LTM slot anchors the opening shot throughout the sequence.

Longer sequences (6 shots and beyond). On longer sequences, identity drift is the dominant failure mode: clothing, face details, and scene framing diverge from the opening reference once no persistent cross-shot anchor exists. UnityShots avoids this collapse through the dual-tier memory: the LTM slot supplies a persistent opening-shot encoding to every denoising pass while the STM slot bridges immediate-boundary transitions in motion and lighting. The rule-based gate keeps the LTM contribution non-zero even at shot 8; quantitative gains across length brackets are in Table 4.

#### 4.4. Main Results

Table 1 compares UnityShots against baselines on the multi-cultural benchmark across three conditioning modes. The I2V section evaluates open-source, closed-source Kling [22], and UnityShots(I2V). The T2V section compares HoloCine [29] against UnityShots(T2V). The R2V section compares DreamID-Omni [11] against UnityShots(R2V). All methods are evaluated on same benchmark sequences.

I2V comparison. UnityShots(I2V) leads every video and multi-shot coherence metric against open-source I2V baselines; the gap is sharpest on cross-shot axes: +0.40 NC and +0.69 Story over the strongest open-source baseline, with AudioAES-A and CLAP also best due to the referenceaudio cross-shot anchor. Against Kling, which leads on

Table 1. Multi-shot generation on the multi-cultural benchmark grouped into video, audio, and multi-shot coherence axes (I2V / T2V / R2V conditioning). ↑ higher is better. Bold and underline mark first and second place within each conditioning block. HoloCine produces no audio (shown as −).

Video Audio Multi-shot Coherence

Method Input

TA ↑ TSIM ↑ AES-V ↑ AES-A ↑ CLAP ↑ NC ↑ Story ↑ Char ↑ Pace ↑ I2V — image-to-video conditioning

LTX-2 [13] I2V 16.75 0.370 0.508 7.11 0.095 3.53 3.28 3.91 3.72 ID-LoRA [7] I2V 16.39 0.362 0.507 7.01 0.113 3.80 3.45 4.04 3.45 OVI [26] I2V 15.86 0.323 0.568 6.37 0.162 3.44 2.97 4.02 3.79 MOVA [38] I2V 16.13 0.298 0.575 6.54 0.146 3.98 3.56 4.45 4.18 Kling [22] I2V 19.20 0.378 0.610 6.83 0.155 4.25 4.15 4.75 4.20 UnityShots (I2V) (ours) I2V 20.62 0.392 0.563 7.30 0.170 4.38 4.25 4.54 4.20

T2V — text-to-video conditioning

HoloCine [29] T2V 18.16 0.401 0.494 – – 3.51 3.22 3.72 3.28 UnityShots (T2V) (ours) T2V 19.17 0.451 0.540 7.39 0.186 4.13 3.83 4.11 3.39

R2V — reference-identity-to-video conditioning

DreamID-Omni [11] R2V 16.81 0.490 0.555 6.56 0.168 2.84 2.64 3.14 2.90 UnityShots (R2V) (ours) R2V 17.98 0.548 0.543 7.57 0.176 3.36 3.12 3.40 3.44

per-frame fidelity (AES-V 0.610, Char 4.75) from its larger training scale, UnityShots still leads on NC (+0.13), Story (+0.10), and every audio metric (AES-A +0.47), since Kling produces ambient audio rather than speaker-anchored cross-shot audio.

T2V and R2V comparison. As shown in Figure 5, under the T2V protocol UnityShots(T2V) outperforms HoloCine [29] on every shared metric (+0.62 NC, +0.61 Story) and additionally provides synchronised audio that HoloCine does not generate. In R2V mode, UnityShots(R2V) outperforms DreamID-Omni [11] on 8 of 9 metrics on a matched 50-sequence subset: the largest gaps are on the multi-shot-coherence axes (+0.52 NC, +0.48 Story, +0.54 Pace) and on the audio block (AES-A +1.01, CLAP +0.008), reflecting DreamID-Omni’s lack of any cross-shot temporal or audio anchor. The only column where DreamID-Omni leads is AES-V (0.555 vs. 0.543), consistent with its identity-focused per-frame fidelity at the cost of cross-shot narrative.

Fairness via the I2V-QwenIMG protocol. A confound in chained-I2V evaluation is that baselines which forward the previous shot’s last frame as the next first frame accumulate identity drift independent of the multi-shot model. To remove this confound, Table 2 re-runs the comparison with all per-shot first frames generated by a single Qwen-ImageEdit-2511 [45] model conditioned on the reference identity portrait and the per-shot caption so every method sees an identical first-frame distribution. UnityShots(I2V) retains its lead on narrative metrics under this setting; additionally, using the NanoBanana image generator instead of QwenImage produces higher overall visual quality for all meth-

ods while preserving the ranking. Additional comparisons against open-source video benchmarks are provided in Appendix E.

Table 2. Results under two controlled first-frame sources on a held-out subset. NanoBanana uses a high-quality image generator; QwenImage uses Qwen-Image-Edit-2511 conditioned on the reference portrait and per-shot caption. ↑ higher is better. Bold and underline mark first and second within each frame-source block.

Method Frame NC ↑ Pace ↑ Story ↑ Char ↑ Cult ↑ LTX-2

4.19 3.91 3.82 4.45 3.03 ID-LoRA 4.47 3.73 4.36 4.73 3.06 OVI 4.55 3.91 4.36 5.00 3.58 MOVA 4.78 3.82 4.55 4.91 3.64 UnityShots (I2V) 4.82 4.55 5.00 5.00 3.70

Nano Banana

4.17 2.36 3.73 4.36 2.15 ID-LoRA 4.45 1.64 4.27 4.64 1.70 OVI 4.52 2.73 4.27 4.89 2.79 MOVA 4.75 2.36 4.45 4.81 2.94 UnityShots (I2V) 4.82 2.18 4.93 4.96 2.30

LTX-2

Qwen Image

#### 4.5. Ablation Study

Both memory slots are necessary. Table 3 shows that activating either slot alone lowers NC relative to the nomemory backbone (STM-only 4.30, LTM-only 3.95, vs. nomemory 4.45) while improving Story and Char, revealing a complementarity requirement. STM alone provides localcontinuity cues without a long-range identity anchor; LTM alone supplies that anchor but lacks recent visual context for smooth transitions. Only when both are active do all four metrics simultaneously exceed no-memory, confirm-

ID-Reference & Prompt ID-Reference & Prompt

Shot1: Close-up on Leyla examining an ancient, dusty manuscript with a

- Shot1: Jomo, dressed as a king, points an angry finger at Nia who cowers slightly but holds her…
- Shot2: Close-up over Jomo's shoulder, focusing on Nia's defiant face under the warm stage …
- Shot3: Jomo suddenly drops his theatrical posture, his face softening into raw grief as he turns away from Nia and stares directly into the camera lens..

magnifying glass under warm, … Shot2: Medium shot of Leyla looking up from the manuscript, smiling confidently at Osman … Shot3: Leyla's perspective showing Tariq leaning in, his eyes gleaming with greed as he nods decisively ... Shot4:

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Extreme close-up on Leyla's face as she adjusts her glasses and looks back down at the page … Shot5: Wide shot of Leyla slowly backing away from the table, looking terrified at the two men, …

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Jomo Nia

Leyla Tariq Osman

Shot1 Shot2 Shot3 Shot1 Shot2 Shot3 Shot4 Shot5

|[Figure 43]|
|---|

|[Figure 44]|
|---|

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

❌ identity drift ❌ close-up

[Figure 52]

OVI

|[Figure 53]<br><br>[Figure 54]<br><br>❌ wrong identity|
|---|

|[Figure 55]|
|---|

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

❌ Leyla

[Figure 63]

❌ identity drift

[Figure 64]

❌ identity drift

ID-LoRA

|[Figure 65]|
|---|

|[Figure 66]|
|---|

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

❌ identity drift ❌ identity drift

[Figure 74]

[Figure 75]

❌ Leyla

LTX-2

|[Figure 76]<br><br>[Figure 77]<br><br>❌ wrong identity|
|---|

|[Figure 78]|
|---|

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Kling 3.0

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]<br><br>UnityShots (Ours)|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

ID-Reference & Prompt Shot1: Medium shot of Madame Dubois sitting rigidly in a velvet armchair, tapping her pearl-ringed finger impatiently on the wooden armrest…

- Shot2: Over-the-shoulder shot from Julien‘s perspective, looking at Viktor who sits across the mahogany table …
- Shot3: Extreme close-up of Julien’s face, sweat beading on his forehead as he pushes his wire-rimmed glasses up his nose …
- Shot4: Medium close-up of Julien looking off-camera at Madame Dubois, swallowing hard before speaking with a trembling jaw. Julien says in French: …
- Shot5: Close-up of Madame Dubois, her eyes widening in shock before narrowing into a glare of absolute, murderous fury. fabric rustling as she stiffens …
- Shot6: Medium shot of Viktor, completely unaware of the mistranslation, smiling warmly and nodding his head in polite agreement. Viktor says in Russian: …

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Julien Dubois Viktor

Shot1 Shot2 Shot3 Shot4 Shot5 Shot6

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

❌ wrong identity ❌ background drift ❌ background drift

DreamID-Omini

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

❌ background drift

Kling 3.0 (R2V)

|[Figure 119]|
|---|

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

UnityShots (Ours)

- Figure 3. Qualitative comparison on the multi-cultural benchmark. Columns show LTX-2 (no memory), ID-LoRA, Kling, and UnityShots(I2V) on 3-shot and 5-shot sequences from diverse cultural contexts. UnityShots follows per-shot prompts while maintaining stable subject identity across all cuts. LTX-2 and ID-LoRA accumulate visible drift from shot 2 onward, and Kling, despite high per-frame fidelity, shows subject-framing inconsistencies across cuts that the LTM anchor prevents by maintaining a persistent latent encoding of the opening-shot reference through every denoising pass.

ing their complementary roles (Figure 4). In R2V mode the full system gains +0.15 NC and +0.12 Char over the best single-slot variant, where LTM is the only persistent identity reference available.

ture as the STM gate) performs comparably on 3–4 shot sequences but degrades at 6+ shots (−0.29 NC, −0.50 Story) as gradient updates gradually suppress the openingshot contribution. A fixed monotone-in-bN schedule is immune to this suppression; full results by sequence length are in Appendix H.

Cultural authenticity and memory. The no-memory baseline leads on Cult in the I2V ablation (4.20 vs. 3.27–3.33 for memory-augmented variants), revealing that fixed identity tokens can suppress cultural-specific visual details the backbone would otherwise infer from the prompt. This trade-off is discussed further in Section 5.

#### 4.6. Long-Sequence Robustness

The fixed monotone LTM gate keeps a non-zero openingshot contribution regardless of sequence length. Figure 8a(a) shows UnityShots leading at early shot indices and ahead at most later ones. Figure 8a(b) confirms the cumulative NC advantage grows with the shot horizon, end-

Effect of LTM gating strategy. Replacing the rule-based LTM coefficient with a learned MLP gate (same architec-

Table 3. Component ablation in I2V (top) and R2V (bottom) conditioning modes on a held-out subset of the benchmark. ↑ higher is better. Bold and underline mark first and second per column within each block.

Table 4. Long-sequence robustness across shot-count brackets. ↑ higher is better. Bold and underline mark first and second per row.

Shots Method NC ↑ Story ↑ Char ↑ TSIM ↑ Pace ↑

LTX-2 4.45 4.64 4.36 0.359 3.64 ID-LoRA 4.55 4.36 4.27 0.354 3.55 MOVA 4.50 4.55 4.45 0.310 3.09 UnityShots (I2V) 4.98 4.91 5.00 0.389 3.73

Configuration NC ↑ Story ↑ Char ↑ Cult ↑ I2V — image-to-video

3–4

LTX-2.3 (no memory) 4.45 3.60 4.40 4.20 +STM only 4.30 4.04 4.18 2.93 +LTM only 3.95 3.97 4.22 3.27 UnityShots (I2V) (ours) 4.95 4.80 5.00 3.33

- 5

LTX-2 4.43 4.31 4.65 0.358 3.48 ID-LoRA 4.55 4.41 4.65 0.339 3.96 MOVA 4.52 4.28 4.81 0.287 4.31

- UnityShots (I2V) 4.91 4.87 4.95 0.378 4.36

6+

LTX-2 4.16 3.86 4.29 0.398 3.64 ID-LoRA 4.36 4.14 4.64 0.449 3.86 MOVA 4.71 4.50 4.93 0.328 4.14

- UnityShots (I2V) 5.00 5.00 5.00 0.437 4.21

R2V — reference-identity-to-video

+STM only 3.01 2.75 3.20 2.63 +LTM only 2.84 2.53 3.13 2.77 UnityShots (R2V) (ours) 3.16 2.90 3.32 2.69

ID-Shot-Reference & Prompt Shot1: Close-up on a rocky Turkish coastline at dawn, cool

[Figure 127]

[Figure 128]

blue lighting, …

- Shot2: Medium shot, low angle, his eyes widening in shock as he recognizes the handwriting, ...
- Shot3: Over-the-shoulder shot past Deniz's shoulder, her coat blowing in the wind, looking down at him with tearful eyes …

Shot1 Shot2 Shot3

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Baseline

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

LTM

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

STM

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Ours

- Figure 4. Ablation visualization on representative multi-cultural sequences. Columns show the no-memory baseline, STM-only, and the full system; the full system maintains subject appearance across all four shots while each single-slot variant drifts.

ing +0.11 above the strongest baseline. Table 4 breaks this down across three length brackets. At 3–4 shots all methods are close; at 5 shots the STM slot visibly suppresses the identity drift that baselines accumulate; at 6+ shots MOVA represents the strongest baseline (4.71 NC, 4.50 Story) while UnityShots reaches a perfect NC/Story/Char rating from the Gemini evaluator, the subset where memory has the largest perceptual impact.

#### 4.7. Human Evaluation

We conducted a user study with 32 participants using pairwise win-rate methodology (Figure 8b): each participant rated 4-shot sequences on identity consistency, audio continuity, text faithfulness, and overall quality. UnityShots receives majority preference on identity consistency and overall quality against all open-source baselines, with the largest audio-continuity advantage over DreamID-Omni,

which has no cross-shot audio anchor. The preference ordering on identity consistency closely tracks the Char and NC ranking from Table 1, validating that Gemini ratings capture the same perceptual axis as human annotators.

### 5. Conclusion

We presented UnityShots, a memory-driven multi-shot audio-video generation system built on a 22B dual-stream diffusion transformer. Two fixed-size video memory slotslong-term (opening-shot anchor) and short-term (preceding tail)—are updated at every cut by a boundary-conditioned gate that fuses visual and audio boundary signals. A discrete cut-type prior learned through AdaLN gives users explicit control over transition strength at inference, while Strata-RoPE keeps the three memory strata on disjoint positional bands so the backbone can distinguish them without extra tokens. Trained jointly under I2V, T2V, and R2V conditioning, UnityShots leads open-source baselines on all cross-shot coherence metrics and matches the strongest closed-source system on multi-shot axes across a 200sequence multi-cultural benchmark.

Limitations and future work. Current limitations include occasional instability in subtitle rendering across shots and visual degradation in scenes with multiple simultaneous speakers or compositionally complex layouts; both are expected to improve as base-model capacity grows and the companion agent system matures. A natural next step is to leverage the agent system for iterative per-shot frame refinement after the initial generation pass, enabling a fully automated pipeline that detects, localizes, and corrects visual artifacts across shots while preserving narrative continuity, and produces cinema-quality long-form content without manual intervention. We release model checkpoints, the multi-cultural benchmark, evaluation code, and a companion agent system to support further work on demographically diverse multi-shot generation.

### References

- [1] Zhaochong An, Menglin Jia, Haonan Qiu, Zijian Zhou, Xiaoke Huang, Zhiheng Liu, Weiming Ren, Kumara Kahatapitiya, Ding Liu, Sen He, et al. Onestory: Coherent multishot video generation with adaptive memory. arXiv preprint arXiv:2512.07802, 2025. 2
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 12, 1
- [3] Max Bain, Jaesung Huh, Tengda Han, and Andrew Zisserman. Whisperx: Time-accurate speech transcription of longform audio. arXiv preprint arXiv:2303.00747, 2023. 1
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [5] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Leo Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024. 2
- [6] ByteDance. Seedance 2.0. https : / / seed . bytedance.com/en/seed2, 2026. 2
- [7] Aviad Dahan, Moran Yanuka, Noa Kraicer, Lior Wolf, and Raja Giryes. Id-lora: Identity-driven audio-video personalization with in-context lora. arXiv preprint arXiv:2603.10256, 2026. 2, 6, 3
- [8] Alexandre D´efossez. Hybrid spectrogram and waveform source separation. arXiv preprint arXiv:2111.03600, 2021. 1
- [9] Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang. Clap learning audio concepts from natural language supervision. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023. 5
- [10] Francesco Foscarin, Jan Schl¨uter, and Gerhard Widmer. Beat this! accurate beat tracking without dbn postprocessing. arXiv preprint arXiv:2407.21658, 2024. 4, 5
- [11] Xu Guo, Fulong Ye, Qichao Sun, Liyang Chen, Bingchuan Li, Pengze Zhang, Jiawei Liu, Songtao Zhao, Qian He, and Xiangwang Hou. Dreamid-omni: Unified framework for controllable human-centric audio-video generation. arXiv preprint arXiv:2602.12160, 2026. 3, 5, 6, 2

- [12] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv preprint arXiv:2503.10589, 2025. 2
- [13] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026. 2, 3, 6
- [14] Jingwen He, Hongbo Liu, Jiajun Li, Ziqi Huang, Yu Qiao, Wanli Ouyang, and Ziwei Liu. Cut2next: Generating next shot via in-context tuning. arXiv preprint arXiv:2508.08244,

2025. 2

- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 4
- [16] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2
- [17] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022. 2
- [18] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context LoRA for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024. 2
- [19] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 5, 2
- [20] Ozgur Kara, Krishna Kumar Singh, Feng Liu, Duygu Ceylan, James M Rehg, and Tobias Hinz. Shotadapter: Text-tomulti-shot video generation with diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28405–28415, 2025. 2
- [21] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2
- [22] Kuaishou Technology. Kling ai. https://klingai. com/, 2026. 2, 5, 6
- [23] Ke Li, Maoliang Li, Jialiang Chen, Jiayu Chen, Zihao Zheng, Shaoqi Wang, and Xiang Chen. Direct: Video mashup creation via hierarchical multi-agent planning and intent-guided editing. arXiv preprint arXiv:2604.04875, 2026. 2, 3
- [24] Zhengyang Liang, Daoan Zhang, Huichi Zhou, Rui Huang, Bobo Li, Yuechen Zhang, Shengqiong Wu, Xiaohan Wang, Jiebo Luo, Lizi Liao, et al. Univa: Universal video agent towards open-source next-generation video generalist. arXiv preprint arXiv:2511.08521, 2025. 2, 3
- [25] Haohe Liu, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong, Qiao Tian, Yuping Wang, Wenwu Wang, Yuxuan Wang, and

- Mark D Plumbley. Audioldm 2: Learning holistic audio generation with self-supervised pretraining. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32: 2871–2883, 2024. 3
- [26] Chetwin Low, Weimin Wang, and Calder Katyal. Ovi: Twin backbone cross-modal fusion for audio-video generation. arXiv preprint arXiv:2510.01284, 2025. 6, 2
- [27] Xiangyang Luo, Qingyu Li, Xiaokun Liu, Wenyu Qin, Miao Yang, Meng Wang, Pengfei Wan, Di Zhang, Kun Gai, and Shao-Lun Huang. Filmweaver: Weaving consistent multishot videos with cache-guided autoregressive diffusion. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7689–7697, 2026. 2
- [28] Yawen Luo, Xiaoyu Shi, Junhao Zhuang, Yutian Chen, Quande Liu, Xintao Wang, Pengfei Wan, and Tianfan Xue. Shotstream: Streaming multi-shot video generation for interactive storytelling. arXiv preprint arXiv:2603.25746, 2026. 2
- [29] Yihao Meng, Hao Ouyang, Yue Yu, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Hanlin Wang, Yixuan Li, Cheng Chen, Yanhong Zeng, et al. Holocine: Holistic generation of cinematic multi-shot long video narratives. arXiv preprint arXiv:2510.20822, 2025. 2, 5, 6
- [30] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 5
- [31] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

- 2023. 3

[32] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

- 2024. 2

- [33] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: international conference for high performance computing, networking, storage and analysis, pages 1–16. IEEE, 2020. 5
- [34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [35] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 5
- [36] Tom´as Soucek and Jakub Lokoc. Transnet v2: An effective deep network architecture for fast shot transition detection. corr abs/2008.04838 (2020). arXiv preprint arXiv:2008.04838, 2020. 4, 5, 1
- [37] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 3

- [38] OpenMOSS Team, Donghua Yu, Mingshu Chen, Qi Chen, Qi Luo, Qianyi Wu, Qinyuan Cheng, Ruixiao Li, Tianyi Liang, Wenbo Zhang, et al. Mova: Towards scalable and synchronized video-audio generation. arXiv preprint arXiv:2602.08794, 2026. 6, 2, 3
- [39] Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hoffman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, et al. Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound. arXiv preprint arXiv:2502.05139, 2025. 5
- [40] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 3
- [41] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2
- [42] Qinghe Wang, Yawen Luo, Xiaoyu Shi, Xu Jia, Huchuan Lu, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Cinemaster: A 3d-aware and controllable framework for cinematic text-to-video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–10, 2025. 2
- [43] Qinghe Wang, Xiaoyu Shi, Baolu Li, Weikang Bian, Quande Liu, Huchuan Lu, Xintao Wang, Pengfei Wan, Kun Gai, and Xu Jia. Multishotmaster: A controllable multi-shot video generation framework. arXiv preprint arXiv:2512.03041,

2025. 2

- [44] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In International Conference on Learning Representations, pages 42055–42079, 2024. 5
- [45] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report,

2025. 6

- [46] Xiaoxue Wu, Bingjie Gao, Yu Qiao, Yaohui Wang, and Xinyuan Chen. Cinetrans: Learning to generate videos with cinematic transitions via masked diffusion models. arXiv preprint arXiv:2508.11484, 2025. 2
- [47] Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan L Yuille, and Lu Jiang. Captain cinema: Towards short movie generation. arXiv preprint arXiv:2507.18634, 2025. 2
- [48] Jinbo Xing, Long Mai, Cusuh Ham, Jiahui Huang, Aniruddha Mahapatra, Chi-Wing Fu, Tien-Tsin Wong, and Feng

- Liu. Motioncanvas: Cinematic shot design with controllable image-to-video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11, 2025. 2
- [49] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-tovideo diffusion models with an expert transformer. In International Conference on Learning Representations, pages 83048–83077, 2025. 2
- [50] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2

- [51] Kaiwen Zhang, Liming Jiang, Angtian Wang, Jacob Zhiyuan Fang, Tiancheng Zhi, Qing Yan, Hao Kang, Xin Lu, and Xingang Pan. Storymem: Multi-shot long video storytelling with memory. arXiv preprint arXiv:2512.19539, 2025. 2, 4, 5
- [52] Lvmin Zhang, Shengqu Cai, Muyang Li, Gordon Wetzstein, and Maneesh Agrawala. Frame context packing and drift prevention in next-frame-prediction video diffusion models. Advances in Neural Information Processing Systems, 38: 30546–30566, 2026. 2
- [53] Peixuan Zhang, Zijian Jia, Kaiqi Liu, Shuchen Weng, Si Li, and Boxin Shi. Stage: Storyboard-anchored generation for cinematic multi-shot narrative. arXiv preprint arXiv:2512.12372, 2025. 2
- [54] Yiming Zhang, Yicheng Gu, Yanhong Zeng, Zhening Xing, Yuancheng Wang, Zhizheng Wu, Bin Liu, and Kai Chen. Foleycrafter: Bring silent videos to life with lifelike and synchronized sounds. International Journal of Computer Vision, 134(1):46, 2026. 3
- [55] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 2
- [56] Jinsong Zhou, Yihua Du, Xinli Xu, Luozhou Wang, Zijie Zhuang, Yehang Zhang, Shuaibo Li, Xiaojun Hu, Bolan Su, and Ying-cong Chen. Videomemory: Toward consistent video generation via memory integration. arXiv preprint arXiv:2601.03655, 2026. 2, 3
- [57] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent selfattention for long-range image and video generation. Advances in Neural Information Processing Systems, 37: 110315–110340, 2024. 2

Story : The Shadow Wall (No on-screen captions, no signs with English text) Shot1: Medium static shot. Warm studio lighting, dark wood background. Thomas leans into a microphone, eyes narrowing. He says his line. Ambient: faint hum of studio equipment. He says his line. Ambient: faint hum of studio equipment. Cuts to Lukas … Shot2: Close-up, slow dolly in. Low-key lighting casting deep shadows. Lukas rubs his thumb, looking down with heavy grief, speaking softly. Ambient: faint breathing. Camera holds on his face … Shot3: Over-the-shoulder shot past Lukas. Soft spotlight on Thomas. Thomas tilts his head, probing gently but firmly. Ambient: faint chair squeak. Cuts back to Lukas … Shot4: Extreme close-up, static. Harsh rim light catches tears in Lukas's eyes. He looks up, jaw set with sudden resolve, delivering his line. Ambient: heavy silence …

###### Shot1 Shot2 Shot3 Shot4

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

(14B,no Audio)

HoloCine

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

Unity Shots

(Ours)

Story : The Saffron Rival (No on-screen captions, no signs with English text) Shot1: Medium close-up, rule of thirds framing, slow dolly in. Cinematic studio lighting, dramatic rim light on the subject against a dark backdrop, intimate mood. Tariq, looking down then raising his eyes with a flicker of vulnerability, says … Shot2: Over-the-shoulder shot from Tariq‘s blurred shoulder, focusing on Layla, static camera. Soft diffused key light, warm and inviting atmosphere. Layla, leaning forward with a micro-emotion of deep empathy, asks … Shot3: Close-up on hands resting on knees, tilting up to a tight frame of the face, handheld subtle movement. High contrast, shadows deepening in the creases of his face, melancholic mood. Tariq, swallowing hard with a micro-emotion of lingering grief, replies … Shot4: Wide two-shot, symmetrical framing, slow pan from left to right. Warm spotlighting illuminating both figures, shifting to a hopeful mood. Tariq, gesturing with his scarred hands and a sudden spark of excitement in his eyes, says … Shot5: Extreme close-up, eye-level, static. Soft, glowing catchlights in the eyes, triumphant and warm mood. Tariq, breaking into a wide, genuine smile that crinkles his eyes, concludes

###### Shot1 Shot2 Shot3 Shot4 Shot5

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

(14B,no Audio)

HoloCine

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Unity Shots

(Ours)

- Figure 5. Qualitative comparison in T2V mode. Each column shows a different method generating a 4-shot sequence from text prompts alone. UnityShots(T2V) maintains consistent subject appearance and scene context across all shots while producing synchronized audio. HoloCine achieves competitive per-shot visual quality but lacks a cross-shot memory mechanism and generates no audio track. Opensource T2V baselines exhibit visible identity and background drift from shot 2 onward.

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Shot Segmentation

Subtitle Processing

Audio Processing

Multimodal Annotation

Character Anchoring

Merging Quality Filtering

Story Design

Identity & Audio Reference

Per-shot First Frame Generation

TTS Reference Audio

Index Assembly

TransNetv2 Dual Strategy

Whisper / WhisperX + DEMUCS

Qwen3-VL Gemini3-pro LLM

Nano Banana Pro

(Proprietary Image Generation Model)

QWen-Image Output Example A. Subtitle Removal (First frame edit)

(Subtitle-free video

Transcription & Speaker Diarization

Bounding Boxes (per frame)

Per-character Reference Frame

Unified Per-shot Record

Quality Gates

200 unique multi-shot stories

Reference imago (per character)

First frame (per shot)

TTS (multi-lingual)

Per-shot dialogue (TTS audio)

Benchmark record

Shot clips

Cut probability

0.12 0.85 0.34 …

Transition type

Continue Dissolve Hard …

· · ·

Original No-subtitle

B. Subtitle Standardization (Extract & normalize into caption)

Example Transcript:

00:00:01 Hello! 00:00:03 Hi, how are you? 00:00:05 I'm fine.

Example Transcript

- Spk1 00:00:01 Hello!
- Spk2 00:00:03 Hi, how are you?

Spk1 00:00:05 I'm fine.

Audio Separation (4 stems)

Vocals Drums Bass Ambient

Visual Caption (per frame/shot)

Narrative Caption (speech + visual)

A man greets a woman and they have a friendly conversation

Speaker ↔Character (temporal overlap)

Example Fields

Shot clip path Audio path Speaker map Transcription Captions BBoxes Cut type

Black frame Motion blur

Dialogue Presence Presence

Aspect ratio

Deduplication (content fingerprint)

Per-character profiles

Per-shot captions

Cultural quotas ( 6 regions)

25% East Asian 25% Western 15% South Asian 15% African 10% Latin 10% Middle Eastern

Reference audio (6–12 s)

(Content distinct from shots)

Story id Region / language Character images Reference audio Per-shot:

First frame Caption TTS audio

Boundary labels First shot Continue cut Hard cut

Final Training Corpus (Per-shot Records)

Tools: TransNetv2; Subtitle Removal Model; Whisper / WhisperX; DEMUCS; Qwen3-VL; Gemini3-Pro Filtering: Black frame, motion blur, dialogue presence, aspect ratio; Content fingerprint deduplication

Shot clip

Audio & stems

Per-character audio Transcription

Narrative caption

Character boxes

Speaker map

Cut type& probability

Story & metadata

Character images

Reference audio

Per-shot first frames

Per-shot captions

Per-shot TTS audio

Boundary labels

|Coverage: 200 sequences, 3–9 shots each, 6 ethnic regions, ≥ 10 languages Per-boundary labels: {First-shot, Continue-cut, Hard-cut}|
|---|

Final Benchmark (200 Multi-cultural Multi-shot Sequences)

…

…

…

…

…

…

…

… … …

…

…

… …

…

… … …

1 2 3 4 5 6 7 1 2 3 4 5

Training Data Construction Benchmark Construction

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

. .

. .

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

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Two people are talking in a living room

…

In the soft light at the center of the stage

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

- Figure 6. Training and benchmark construction pipelines. Top: the 7-stage training pipeline processes raw cinematic and musicvideo footage through shot segmentation (TransNetv2), subtitle removal, audio diarisation (Whisper/DEMUCS), multimodal captioning (Qwen3-VL/Qwen3-Omni [2]), character anchoring, record merging, and quality filtering, yielding 146k annotated shot records. Bottom: the benchmark pipeline generates the 200-sequence multi-cultural evaluation set through LLM story generation, reference portrait synthesis, per-shot first-frame generation, reference audio synthesis via multilingual TTS, and flat index assembly. Full pipeline details are in Appendix C.

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Story 8 (Amina & Jengo, 3 shots): In a bustling Nairobi market, an observant fruit vendor recognizes a nervous fugitive from a newspaper clipping but makesa silentchoice tolethimgo.

###### Story 40 (Ji-yoon & Seo-yeon, 5 shots): A proud, perfectionist schoolgirl accidentally breaks a priceless family heirloomand bracesfor her strictgrandmother'swrath, only todiscover unexpected grace.

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

Shot Three: Close-up on Ji-yoon, she squeezes her eyes shut, her proud facade vanishing, she bows her head deeply, presenting the broken shards in her outstretched palms like a guilty plea. Jiyoon says in Korean: "할머니, 제가 깼어요. 죄송합니다.". Ambient sound: rustling of the school uniform fabric, sharp intake of breath. BGM cue: a single, melancholic piano chord strikes. Continuing from prior shot, Ji-yoon is still holding the green shards, her head now bowed in submission.

Shot Two: Over-the-shoulder shot from Amina's perspective, looking at Jengo's nervous face, then glancing down at a crumpled newspaper clipping on the counter showing his face, before Jengo speaks urgently. Jengo says in Swahili: \"Nipe maembe mawili, haraka.\". Market chatter fading slightly, rustling of canvas fabric. A single sustained, suspenseful cello note. continuing from prior shot, Amina remains at the fruit stall, Jengo still wears the dusty olive jacket with the hood down..

Shot Three: Medium shot of Amina locking eyes with Jengo, her expression unreadable as she calmly slides a woven basket over the newspaper clipping, hands him the fruit, and nods solemnly. Amina says in Swahili: \"Nenda kwa amani, kijana.\". Heavy footsteps receding, a sudden gust of wind rustling the stall's tarp. Soft, melancholic kalimba melody resolving the tension. continuing from prior shot, Jengo holds the mangoes in his dusty jacket, Amina's hands rest on the woven basket.

Shot Five: Extreme close-up on Ji-yoon's face, her rigid posture completely collapses, large tears spill over her eyelashes and stream down her cheeks as she lets out a silent, relieved sob, finally looking like a normal child. Ambient sound: soft sniffling, distant wind against the window. BGM cue: gentle, resolving piano melody. Continuing from prior shot, Ji-yoon's face is still framed by her velvetribboned braids, now wet with tears.

Shot Four: Over-the-shoulder shot from Jiyoon's perspective, Seo-yeon kneels gracefully, ignoring the priceless shards, she gently takes Jiyoon's small hands in her own wrinkled ones, inspecting her fingers for cuts with a tender expression. Seo-yeon says in Korean: "다치지 않았느냐? 도자기는 다시 구우면 그만이다.". Ambient sound: soft rustle of silk hanbok, gentle skin contact. BGM cue: warm, swelling string section replaces the tension. Continuing from prior shot, Seo-yeon is now kneeling at Jiyoon's eye level, holding the girl's hands.

Shot Two: Medium shot, the heavy oak door slides open and Seo-yeon steps in, the warm hallway light spilling over her jade hanbok, she stops, her sharp eyes darting from the floor to Ji-yoon's trembling shoulders. Ambient sound: sliding door rumble, soft footsteps on wood. BGM cue: cello sustains a low, ominous note. Continuing from prior shot, Ji-yoon remains frozen in the background as Seoyeon enters the frame.

Shot One: Low angle shot in a dimly lit, luxurious Seoul apartment with traditional wooden accents, Ji-yoon stands frozen, clutching jagged green celadon shards in her small hands, her pristine uniform contrasting with the mess on the hardwood floor. Ambient sound: distant city traffic, ticking of a grandfather clock. BGM cue: tense, sparse cello plucks.

Shot One: Close-up on Amina's hands arranging ripe mangoes on a wooden stall in a bustling Nairobi market, panning up to her face as Jengo steps into the frame, his hood pulled low, casting a shadow over his eyes. Bustling market crowd, distant matatu horns, flies buzzing. Low, rhythmic percussion building tension.

[Figure 260]

[Figure 261]

[Figure 262]

###### Story 110(Linh, Ông Minh, &Khoa, 8 shots): Adesperate single mother inHanoiattemptstopay her strictlandlord withinsufficientfunds, ultimately relying ona family heirloomthatrevealsa shared past.

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

Shot Eight: Wide shot of the desk, Ông Minh gently covers the watch with his hand, then slowly pushes the brown envelope back to Linh, offering a sad, forgiving smile. distant motorbikes, soft rustling of leaves outside. warm, resolving acoustic guitar BGM. continuing from prior shot, Khoa is still visible beside Linh, and Minh's posture has completely softened.

Shot Four: Close up on Ông Minh, he frowns deeply, sliding the envelope back toward Linh with a dismissive flick of his fingers. Ông Minh says in Vietnamese: "Chừng này không đủ đâu, cô gái.". heavy sigh, paper sliding back. tense, low string BGM. continuing from prior shot, Minh remains seated in his crisp white shirt, his expression hardening.

Shot Five: Over-the-shoulder shot from Minh's perspective, Linh bites her lip, reaches into her worn canvas tote bag, and pulls out a tarnished brass pocket watch. fabric rustling, metallic clink of the watch on the desk. a sudden, poignant piano chord BGM. continuing from prior shot, Linh's expression shifts from fear to desperate resolve.

Shot Six: Extreme close up on Ông Minh's face, his stern eyes widen in shock behind his glasses, his jaw slightly dropping as he stares at the object. complete silence except for a sharp intake of breath. swelling, emotional piano BGM. continuing from prior shot, Minh's silverrimmed glasses reflect the dim light from the window.

Shot Seven: Medium close up on Linh, she looks at Minh with tearful but steady eyes, gently pushing the watch closer to him. Linh says in Vietnamese: "Cha tôi bảo ông sẽ nhận ra cái này.". soft breathing, faint ticking of the brass watch. swelling, emotional piano BGM. continuing from prior shot, Linh's dark hair is still in a messy bun, a tear forming in her eye.

Shot Two: Interior dimly lit room with wooden shutters, dust motes in the light, Ông Minh sits at a heavy teak desk, adjusting his silver-rimmed glasses while inspecting a red ledger. ticking of a wall clock, faint street noise. somber, slow cello BGM. continuing from prior shot, the setting shifts indoors to the landlord's office.

Shot Three: Medium shot, Linh and Khoa stand before the desk, Linh nervously places a thin brown envelope onto the polished teak wood. paper sliding on wood, shuffling footsteps. tense, low string BGM. continuing from prior shot, Linh still wears her faded yellow blouse and Khoa holds his toy boat.

Shot One: Exterior of a weathered, mosscovered colonial building in Hanoi, overcast lighting, Linh holds Khoa's hand, looking up at the wooden door with a deep breath. distant motorbikes, faint rustling of banyan tree leaves. somber, slow cello BGM.

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

Story 165 (Mateo&Sofia, 4shots): Anisolated Argentinianlighthouse keeper ishaunted by a voice fromhispast during a violentstorm. The supernaturalencounter forceshimtoconfronta tragic loss.

Story 097(HenriBeaufort &Elise Vasseur, 4shots): In a dusty Parisian atelier, a young woman discovers that an antique tailor has crafted the exactcoatshe sawina terrifying nightmare.

Shot Two: Over-the-shoulder shot from Elise's perspective, Henri steps out from the shadows with a calm and unreadable face, gently taking the coat off the mannequin and holding it up towards her. Henri says in French: "Je l'ai cousu pour vous. Je vous attendais.". rustling of heavy velvet fabric, distant wind outside. low sustained cello drone BGM. continuing from prior shot, Henri still wears his tweed waistcoat.

Shot Two: Close-up on Elise's face as Henri drapes the heavy crimson coat over her shoulders, her expression shifting from terror to a strange dark acceptance as she looks at her reflection in an antique mirror. soft fabric sliding over shoulders, a single ticking grandfather clock. haunting choral whisper BGM fading out. continuing from prior shot, Elise is now wearing the crimson coat over her trench coat, Henri stands right behind her.

Shot Two: Low angle looking up the stairs from the shadows, Sofia stands motionless on a step, her yellow raincoat glowing faintly in the lantern light, water pooling at her boots. Sofia says in Spanish: "El mar me devolvió, papá.". loud dripping of water, sudden gust of wind rattling the tower. sharp, dissonant violin screech BGM. continuing from prior shot, the lighting matches the lantern's glow from above, Sofia is completely soaked.

Shot One: Medium shot of Elise stopping dead in her tracks, staring at the crimson coat on the mannequin, the lighting casting long shadows as she slowly reaches out a trembling hand to touch the silver embroidery. Elise says in French: "Je l'ai vu dans mon cauchemar... la nuit dernière.". creaking floorboards, sharp intake of breath. swelling dissonant strings BGM. continuing from prior shot, Elise still wears her beige trench coat.

Shot One: High angle looking down the dark, wrought-iron spiral staircase, Mateo leans over the railing, holding a lantern out into the darkness with a tense expression. Mateo says in Spanish: "¿Hay alguien ahí? La puerta estaba cerrada.". creaking metal, echoing wind, faint dripping water. tense, rising string section BGM. continuing from prior shot, Mateo still wears the dark blue wool sweater, holding the lantern tightly.

Shot Two: Close-up on Mateo's face, his eyes wide with terror and grief, the lantern trembling in his hand as he stares at the empty staircase where only a puddle remains. fading wind, solitary drop of water hitting a metal grate. melancholic, fading piano chord BGM. continuing from prior shot, Mateo's expression shifts from tension to absolute shock, the staircase below him is now empty.

Shot One: Dimly lit, dusty atelier, close-up on Henri's weathered hands stitching silver thread into a striking crimson velvet coat, camera pans up to show Elise entering through the antique glass door. tinkling shop bell, rhythmic sound of a needle pulling through heavy fabric. subtle minorkey cello BGM.

Shot One: Inside the glass lantern room of a lighthouse at night, rain lashing the glass, Mateo wipes the massive brass lens with a rag, suddenly pausing as he tilts his head. howling wind, heavy rain striking glass, rhythmic hum of the lamp. low, ominous cello drone BGM.

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

- Figure 7. Multi-cultural benchmark overview. Representative reference portraits, per-shot first frames, and generated sequences from six ethnic regions covered by the 200-sequence benchmark. Each row shows a distinct cultural context; characters share identity and voice across all shots in a sequence.

(a) Per-shot identity preservation

(b) Identity retention vs. shot horizon

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| |UnityShots (ours)<br><br>ID-LoRA| | | | | | | | | |
| |LTM-anchored opening DreamID-Omni<br><br>LTX-2.3 I2V| | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| |+0.11<br><br>|
| | |
| | |
| | |

0.9

0.9

RIC-Subj(runningmean,w=3)

0.8

0.8

RIC-Subj(cumulativemean)

0.7

0.7

0.6

0.6

0.5

0.5

0.4

0.4

0.3

0.3

0.2

0.2

0.1

0.1

0 1 2 3 4 5 6 7 8

0 1 2 3 4 5 6 7 8

Shot index

Shot index

(a) Per-shot identity preservation across the shot horizon. (a) Per-shot NC score with a sliding-window mean: UnityShots holds its LTM-anchored lead across all shot indices while baselines degrade. (b) Cumulative-mean NC versus shot count: UnityShots dominates every prefix length and ends +0.11 above the strongest baseline.

UnityShots vs DreamID-Omni

62% 25% 13%

Identity

58% 26% 15%

Audio

42% 37% 21%

Text

61% 25% 14%

Overall

UnityShots vs Sliding Sink (v3)

56% 31% 13%

56% 30% 15%

39% 40% 20%

58% 29% 13%

UnityShots vs ID-LoRA

69% 22% 10%

51% 29% 20%

45% 35% 20%

64% 23% 13%

Ours wins

| |
|---|

Tie

| |
|---|

Ours loses

| |
|---|

(b) Pairwise human-evaluation preference rates. 32 participants compared 4-shot sequences from UnityShots and each competing method on identity consistency, audio continuity, text faithfulness, and overall quality. UnityShots receives majority preference on all criteria against open-source baselines.

Figure 8. Long-sequence identity preservation (left) and human-evaluation results (right). See Sections 4.6 and 4.7 for analysis.

[Figure 283]

## UnityShots: Memory-Driven Multi-Shot Audio-Video Generation with Boundary-Aware Gating

### Appendix

#### A. Training and Architecture Details

- Table 1 lists the key hyperparameters for Stage 2 multi-shot fine-tuning. Stage 1 (identity foundation) uses the same base model, optimiser, and learning rate, training on singleclip identity data.

#### B. Benchmark Summary

The evaluation benchmark contains 200 multi-cultural multi-shot sequences with 3–9 shots each, covering six ethnic regions and ten or more languages. Cultural quotas ensure coverage: 25% East Asian, 25% Western, 15% South Asian, 15% African, 10% Latin, 10% Middle Eastern. Each sequence is annotated with reference identity images, reference audio (6–12s, content distinct from shot dialogue), per-shot first frames generated by an image-generation model conditioned on each character portrait and per-shot caption, per-shot captions, and per-boundary transitiontype labels (FIRST-SHOT, CONTINUE-CUT, HARD-CUT). The full construction pipeline—story design, identity reference, per-shot first frame, TTS reference audio, and index assembly—is described in Appendix C together with the training-data pipeline, since the two share the same tooling.

#### C. Training Data and Benchmark Construction

This appendix details the pipelines used to construct (i) the multi-shot training corpus and (ii) the multi-cultural evaluation benchmark. A summary figure of the overall workflow will accompany the camera-ready version.

Training data construction. Multi-shot training records are produced by a seven-stage pipeline. (1) Shot segmentation. TransNetv2 segments raw video into individual shot clips and returns a per-boundary visual cut probability [36] and an inferred transition type (CONTINUE, DISSOLVE, HARD) used to populate the cut-type prior in the boundaryconditioned gating module. (2) Subtitle handling. Each shot is annotated with its burnt-in subtitle text, if any. During training, 20% of samples randomly apply a subtitleremoval model to produce a subtitle-free variant, enabling the model to generate clean footage or captioned footage depending on the text prompt without distribution mismatch. (3) Audio processing. We extract the soundtrack and run WhisperX [3] for transcription and per-speaker diarisation. DEMUCS [8] separates vocals, drums, bass, and ambient stems for downstream conditioning. (4) Multimodal annotation. Qwen3-VL [2] provides initial per-frame char-

Table 1. Stage 2 training hyperparameters.

Parameter Value Base model LTX-2.3 22B

(ltx-2.3-22b-dev) Initialization Stage 1 checkpoint Optimizer AdamW, β1=0.9,

β2=0.999

Learning rate 1×10−5; gate MLP 5×

10−5 Gradient clipping 1.0 Precision BFloat16, DeepSpeed Batch size 1 per GPU, gradient ac-

cumulation 1 Max latent frames 96 (≈ 30s at 25fps, 8× compression)

Video LTM slot 2 latent frames Video STM slot (Pv frames) 4 latent frames (≈1s) Audio reference anchor 1 speaker-identity token

from Aref Video RoPE ceiling (flim) 128 (current shot:

[0, Tv−1]) Video STM band [flim−Pv, flim−1] Video LTM band [flim−2Pv, flim−Pv−1]

Cut-type prior τN {FIRST, CONTINUE, HARD} →

{0, 0.4, 1.0}, fused with timestep embedding via AdaLN

Boundary weights (cinematic) α=0.5, β=0.3, γ=0.0 Boundary weights (music video) α=0.5, β=0.3, γ=1.0 STM coefficient gstm(b) = 0.3 + 0.7 b

(empirically determined) LTM coefficient gltm(b) = 0.1 + 0.6 b

(empirically determined) LTM recurrence weight zN = clamp(bN, 0, 1) Content-aware MLP 2-layer, < 50K param-

eters; warm-up clamp [0.5, 1.0] for 500 steps

k-shot training chunks variable k ∈ [3, 9] with bucketed-length sampling

Mixed conditioning (pi2v, pt2v, pr2v) 0.5, 0.3, 0.2 No-subtitle probability 0.3 Reference-audio dropout 0.1

acter bounding-box detection and coarse visual captions. Gemini3-Pro then performs fine-grained narrative captioning that fuses speech transcripts with visual events. (5) Character anchoring. For each shot we extract a per-

character reference frame using an image-generation model for identity-preserving portrait extraction, and link acoustic speaker labels from WhisperX to visual bounding boxes via temporal-overlap matching, producing a per-shot speakerto-character map. (6) Merging. Per-stage outputs are consolidated into a single per-shot record under hierarchical priority handling so that partial annotations do not corrupt downstream training. (7) Quality filtering. Black-frame, motion-blur, dialogue-presence, and aspect-ratio gates remove unusable shots, and content-fingerprint deduplication enforces sequence-level uniqueness. Each surviving shot record carries the clip path, audio path, per-character isolated audio paths, transcription, narrative caption, character bounding-box metadata, and boundary annotations. We do not disclose source titles; only the tooling above is publicly documented.

Benchmark construction. The 200-sequence multicultural benchmark is built by a five-stage pipeline. (1) Story design. Gemini3-Pro generates 200 unique multishot story outlines with per-character descriptions and pershot captions, balanced across six ethnic regions (25% East Asian, 25% Western, 15% South Asian, 15% African, 10% Latin, 10% Middle Eastern) and ten or more languages, with controlled coverage of dialogue-heavy and interviewformat narratives to stress audio continuity. Stories are deduplicated across the full set to prevent theme, conflict, or character repetition. (2) Identity reference. An imagegeneration model synthesises a photorealistic portrait per character, conditioned on the character’s demographic description. (3) Per-shot first frame. The same imagegeneration model generates a per-shot first frame conditioned on the character portrait and the per-shot caption, ensuring visual consistency between the identity reference and the first frame provided to I2V evaluation. (4) Reference audio. ElevenLabs eleven multilingual v2 TTS synthesises a 6–12s reference clip per character; the reference content is deliberately chosen to differ from the shot dialogue to prevent acoustic leakage between reference and generation target. (5) Index assembly. A flat index.json maps each sequence ID to its reference portraits, audio clips, per-shot first frames, and per-shot captions. The benchmark and all per-stage scripts are released alongside the model.

#### D. Extended Comparison with Published MultiShot Methods

- Table 2 extends the main-paper evaluation to additional published T2V multi-shot methods on our 200-sequence multi-cultural benchmark, using the same evaluation protocol as Table 1 of the main paper. StoryMem [51] represents the shot-by-shot family; MultiShotMaster [43] and HoloCine [29] represent the end-to-end family. All methods are evaluated in T2V mode to remove first-frame distribu-

tion confounds.

Table 2. Extended T2V comparison with published multi-shot methods on the 200-sequence multi-cultural benchmark. ↑ higher is better. Bold and underline mark first and second per column.

Method TA ↑ TSIM ↑ NC ↑ Story ↑ Char ↑

StoryMem [51] 17.31 0.368 3.42 3.18 3.72 MultiShotMaster [43] 18.47 0.346 3.65 3.41 4.05 HoloCine [29] 18.16 0.322 3.51 3.22 3.72 UnityShots (T2V) (ours) 19.17 0.451 4.13 3.83 4.11

StoryMem maintains higher TSIM than end-to-end methods because each shot is conditioned on the preceding clip, but trails on NC and Story as identity drift accumulates past the conditioning window. MultiShotMaster achieves the strongest TA among baselines from largerscale training, yet falls behind on NC and TSIM because its fixed context window does not distinguish long-range from short-range context. UnityShots eliminates this tradeoff: the dual-tier memory bank gains +0.48 NC and +0.42 Story over MultiShotMaster while also leading on TSIM by +0.083 over StoryMem.

#### E. Baseline Evaluation Protocol and Standard Benchmark Results

Evaluation setup. MOVA [38], LTX-2 [13], ID-LoRA [7], OVI [26] and DreamID-Omni [11] were evaluated on the 200-sequence multi-cultural benchmark using official checkpoints. For open-source baselines with available checkpoints, we run official checkpoints with the recommended inference settings and use DDIM inference with 50 steps when applicable. For closed-source or service-only systems, we collect outputs through their official interfaces using the same benchmark prompts, reference inputs, and shot-level protocol. All metrics reported in the main paper are computed by us using the same evaluation scripts, rather than copied from prior publications.

VBench comparison. Table 3 reports standard VBench [19] metrics for the methods we evaluated directly, with all shots concatenated at inferred boundaries for per-video scoring. Subject Consistency (Subj) is the VBench metric most directly tied to cross-shot memory: UnityShots leads by +0.006–+0.014 over baselines, reflecting the LTM slot anchoring opening-shot identity through every denoising pass. Background Consistency (BG) and Motion Smoothness (MoS) also improve, since the memory conditioning prevents abrupt scene mode changes at cuts. Aesthetic Quality (AES) is on par with LTX-2 and ID-LoRA; the slight gap versus MOVA is consistent with the memory regularisation mildly reducing per-frame perceptual variety in exchange for cross-shot stability.

- Table 3. VBench evaluation on our 200-sequence benchmark (shots concatenated). Subj: Subject Consistency; BG: Background Consistency; MoS: Motion Smoothness; AES: Aesthetic Quality; TA: Text–Video Alignment. ↑ higher is better. Bold marks best per column.

Method Subj ↑ BG ↑ MoS ↑ AES ↑ TA ↑ LTX-2 [13] 0.953 0.952 0.977 0.649 0.282 ID-LoRA [7] 0.961 0.959 0.980 0.652 0.289 MOVA [38] 0.957 0.955 0.978 0.656 0.287 UnityShots (I2V) 0.967 0.963 0.982 0.652 0.295

#### F. Pseudocode for Memory Update

Algorithms 1 and 2 give complete pseudocode for the boundary-aware gate and the full inference loop respectively.

Algorithm 1 Per-Shot Memory Gate Application Require: Prev-shot latent Vˆ N−1; boundary strength b ∈

[0,1]; LTM slot LN; reference audio Aref; noisy latent VtN

- 1: Extract tail: SN ← Vˆ :NP−1

v

- 2: Boundary-conditioned coefficients: gstm ← 0.3+0.7b; gltm ← 0.1 + 0.6b
- 3: Content-aware refinement: ρ ← σ(ϕ(pool(Vˆ N−1), pool(VtN), eτ))
- 4: Scale STM: SN ← ρ · gstm · SN
- 5: Scale LTM: LN ← gltm · LN
- 6: Build context CN ← [Xref, LN, SN, Aref, VtN] with Strata-RoPE bands
- 7: return denoised latent Vˆ N

#### G. Benchmark Prompt Examples

The following are representative opening-shot captions from the benchmark. Each entry corresponds to one multishot sequence; only the opening caption is shown. The placeholder [subject] is replaced by the reference identity image at inference.

- 1. [subject], East Asian woman in a tailored blazer, stands in a glass-fronted office, composed expression; medium shot, neutral daylight.
- 2. [subject], elderly South Asian man, long kurta, sits at a carved desk surrounded by manuscripts; warm tungsten light.
- 3. [subject], young African woman, natural hair, loose dress, walks through a market street; handheld camera, golden hour.
- 4. [subject], Middle Eastern man in a pressed shirt, stands on a rooftop at dusk; wide establishing shot, city lights below.

Algorithm 2 UnityShots Inference Loop Require: Prompts c0,...,cK−1; boundary strengths

b0,...,bK−1; reference image Xref; reference audio Aref

- 1: L0 ← ∅; S0 ← ∅ ▷ Empty memory at start
- 2: for N = 0,1,...,K − 1 do
- 3: Vˆ N ← DiT(cN,Xref,LN,SN,Aref,bN)
- 4: VtailN ← Vˆ :NP

v

- 5: SN+1 ← VtailN ▷ STM receives latest tail
- 6: if N = 0 then
- 7: L1 ← Vtail0 ▷ Anchor LTM to shot 0
- 8: else
- 9: z ← clamp(bN,0,1)
- 10: LN+1 ← z · VtailN + (1 − z) · LN ▷ LTM recurrence
- 11: end if
- 12: end for
- 13: return {Vˆ 0,...,Vˆ K−1}

- 5. [subject], Latin woman in a floral blouse, sits at a kitchen table, relaxed; shallow depth of field, warm interior.
- 6. [subject], Western man, gray hoodie, leans against a concrete wall; overcast exterior, slight tension in posture.
- 7. [subject], elderly East Asian woman in a floral cheongsam, tends a garden; morning light, soft bokeh background.
- 8. [subject], young South Asian man, glasses, casual shirt, types at a laptop in a cafe; ambient sound, open laptop screen.
- 9. [subject], African man in a suit, seated at a conference table; direct gaze at camera, high-key lighting.
- 10. [subject], Latin teenage girl in a school uniform, stands in a courtyard; midday sun, slight squint, book in hand.

#### H. Per-Category Ablation Results

LTM gating strategy ablation. Table 4 compares three gating configurations across three sequence-length brackets. LTM rule-based + STM learned is the full UnityShots design. LTM learned + STM learned replaces the rulebased LTM coefficient with an MLP gate of identical architecture to the STM gate. LTM rule-based + STM rulebased ablates the learned STM refinement, replacing it with the same fixed monotone schedule used for LTM.

The learned LTM gate degrades most severely at 6+ shots (−0.29 NC, −0.50 Story), confirming that gradientdriven updates destabilize the opening-shot anchor on longer sequences. Replacing the learned STM gate with a fixed rule-based schedule likewise hurts boundary smooth-

- Table 4. LTM gating strategy ablation by sequence length. Metrics are NC, Story, Char, and TSIM on the multi-cultural benchmark (I2V conditioning). Bold marks the best per metric per length bracket.

Shots LTM / STM gate NC ↑ Story ↑ Char ↑ TSIM ↑

3–4

Rule + Learned (ours) 4.98 4.91 5.00 0.389 Learned + Learned 4.95 4.88 4.97 0.381 Rule + Rule 4.90 4.79 4.88 0.372

- 5

Rule + Learned (ours) 4.96 4.93 4.97 0.380 Learned + Learned 4.89 4.80 4.88 0.368 Rule + Rule 4.81 4.72 4.79 0.356

- 6+

Rule + Learned (ours) 5.00 5.00 5.00 0.437 Learned + Learned 4.71 4.50 4.73 0.392 Rule + Rule 4.85 4.77 4.88 0.401

ness across all lengths, showing that content-adaptive refinement is beneficial at short timescales while predictable behavior matters for the long-range anchor.

Table 5. Per-region NC and Cult on the 200-sequence benchmark. UnityShots(I2V) compared against MOVA (NC column) and LTX-2.3 no-memory backbone (Cult column). ↑ higher is better.

NC ↑ Cult ↑ Ours MOVA Ours No-mem

Region

East Asian 4.96 4.02 3.41 4.22 Western 4.93 4.15 3.26 4.12 South Asian 4.97 3.89 3.33 4.24 African 4.94 3.85 3.28 4.19 Latin 4.98 3.93 3.40 4.23 Middle Eastern 4.95 3.82 3.31 4.21

Average 4.955 3.943 3.332 4.202

UnityShots leads MOVA on NC in every region, with the largest gap in Middle Eastern and African sequences (+1.13 and +1.09), where greater scene diversity in those categories amplifies the benefit of a persistent opening-shot anchor. MOVA achieves its highest NC on Western sequences (4.15), where the backbone’s pretraining coverage is denser, reducing the benefit of additional memory. Story and Char rankings follow the same regional pattern as NC.

The Cult gap (no-memory vs. UnityShots) is consistent across all six regions (0.81–0.96), indicating that identity conditioning suppresses cultural visual detail at a similar rate regardless of ethnic background rather than disproportionately affecting any one cultural group. This confirms that the Cult trade-off observed in the main ablation is a systemic property of the memory mechanism rather than a regional artifact.

#### I. Companion Agent System

UnityShots ships with a Claude-Code-compatible agent wrapper that exposes per-shot reference editing as ModelContext-Protocol (MCP) tools. The wrapper installs as a Claude Code plugin and runs cloud-only without a local GPU, so any user with a Claude Code, Claude Desktop, or VS Code MCP client can reach it.

Architecture. The wrapper follows a planner-plus-worker design. The planner is whichever agent the user is conversing with; it decomposes a story prompt into a structured shot plan (shot id, visual prompt, audio hint, aspect ratio, duration), calls a vision-language model to synthesise a reference image per shot, calls UnityShots through the per-shot generation tool with the edited references, and concatenates the resulting clips with FFmpeg. All capabilities are simultaneously exposed as web-UI buttons, HTTP endpoints, and MCP tools that share a single input schema, so the same workflow runs from a chat interface, an HTTP API, or another agent.

Exposed MCP tools. The MCP server registers three tools. create drama accepts a story description together with style, shot count, aspect ratio, per-shot duration, an optional audio flag, and a list of reference images, and runs the end-to-end story-to-multi-shot pipeline asynchronously. generate with reference accepts a single-shot prompt with optional reference image and reference audio, and submits a single-shot generation under explicit references for fine-grained control. query task accepts a task identifier and returns the running status, progress percentage, and result URLs once the job completes. A minimal .mcp.json snippet registers the local server with any MCP-compatible client and is the only configuration step a user has to perform.

Release. The agent wrapper, the MCP server definitions, the cloud-API connectors, and the example shot plans are all released alongside the model under a permissive opensource license. A standalone release branch ships the Claude Code plugin and the cloud-only workflow so the agent runs without a local GPU; a separate full-stack reference implementation is also provided for users who prefer to host the platform themselves. The system enables the per-shot reference editing workflow that the main paper introduces.

