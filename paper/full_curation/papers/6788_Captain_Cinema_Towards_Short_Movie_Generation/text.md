# arXiv:2507.18634v1[cs.CV]24Jul2025

[Figure 1]

[Figure 2]

## Captain Cinema: Towards Short Movie Generation

[Figure 3]

###### Junfei Xiao1,2,⋆, Ceyuan Yang2,†, Lvmin Zhang3, Shengqu Cai2,3, Yang Zhao2, Yuwei Guo2,4, Gordon Wetzstein3, Maneesh Agrawala3, Alan Yuille1, Lu Jiang2,†

1Johns Hopkins University, 2ByteDance Seed, 3Stanford University, 4CUHK ⋆Project Lead, †Corresponding author

###### Abstract

We present Captain Cinema, a generation framework for short movie generation. Given a detailed textual description of a movie storyline, our approach firstly generates a sequence of keyframes that outline the entire narrative, which ensures long-range coherence in both the storyline and visual appearance (e.g., scenes and characters). We refer to this step as top-down keyframe planning. These keyframes then serve as conditioning signals for a video synthesis model, which supports long context learning, to produce the spatio-temporal dynamics between them. This step is referred to as bottom-up video synthesis. To support stable and efficient generation of multi-scene long narrative cinematic works, we introduce an interleaved training strategy for Multimodal Diffusion Transformers (MM-DiT), specifically adapted for long-context video data. Our model is trained on a specially curated cinematic dataset consisting of interleaved data pairs. Our experiments demonstrate that Captain Cinema performs favorably in the automated creation of visually coherent and narrative consistent short movies in high quality and efficiency.

Date: July 25, 2025 Project Page: thecinema.ai

###### 1 Introduction

Narrative is central to how humans communicate, remember, and perceive the world. As Bruner has argued [1], narrative structures are fundamental to organizing human experience, while Harari [2] emphasizes that shared stories underpin the development of societies. In the field of video generation [3–9], recent advances, especially in diffusion-based [10–13] and auto-regressive models [14–17] have enabled impressive progress in synthesizing short video clips for tasks such as image animation [18, 19], video editing [20, 21], and stylization [22]. However, these accomplishments predominantly address visual fidelity and local temporal coherence, leaving the deeper challenge of producing videos that convey coherent, engaging narratives over extended durations. Bridging this gap—from visually plausible snippets to full-length, story-driven videos—constitutes a pivotal frontier for research and practice, and richer human-centered multimodal storytelling.

Automating the generation of feature-length cinematic narratives remains underexplored. Achieving narrative coherence and visual consistency over extended durations demands models that capture long-range dependencies while retaining fine-grained detail. Existing approaches frequently encounter exploding context lengths, storyline discontinuities, and visual drift when scaled to longer videos. To overcome these challenges, we

#### LCT / VideoAuteur …

#### Captain Sora / Kling/ Veo2 … Cinema

[Figure 4]

##### Shot-level Scene-level Movie-level

[Figure 5]

[Figure 6]

[Figure 7]

Single shot

[Figure 8]

[Figure 9]

~8s

Multiple shots

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

~60s

[Figure 15]

Multiple scenes

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

>1,000 s

- Figure 1 Captain Cinema: ‘‘I can film this all day!‘‘ Captain Cinema bridges top-down interleaved keyframe planning with bottom-up interleaved-conditioning video generation, taking a step toward the first multi-scene, whole-movie generation, preserving high visual consistency in scenes and identities. All the movie frames here are generated.

propose Captain Cinema, a framework tailored for story-driven movie synthesis.

CaptainCinema balances global plot structure with local visual fidelity through two complementary modules. A top-down planner first produces a sequence of key narrative frames that outline the storyboard, ensuring coherent high-level guidance. A bottom-up video synthesizer then interpolates full motion conditioned on these keyframes, maintaining both narrative flow and visual detail. Central to this design is GoldenMem, a memory mechanism that selectively retains and compresses contextual information from past keyframes. By summarizing long histories without exceeding memory budgets, GoldenMem preserves character and scene consistency across multiple acts, enabling scalable generation of multi-scene videos.

Additionally, we build a specialized data processing pipeline for processing long video data for movie generation and introduce progressive long-context tuning strategies tailored for Multimodal Diffusion Transformers (MM-DiT). These techniques enable stable and efficient fine-tuning on large-scale, long-form cinematic datasets, addressing the challenges of multi-scene video generation. Extensive experiments and ablation studies demonstrate that Captain Cinema not only achieves strong performance in long-form narrative video synthesis but also enables the automated creation of visually consistent short films that significantly exceed the duration of existing works, setting a new milestone in multimodal video generation capabilities.

###### 2 Related Works

Text-to-Video Generation. Text-to-video models [3–9] now generate 5–10 s clips with high visual fidelity. Most adopt a latent diffusion paradigm [23]; diffusion variants such as DiT [10], Sora [11], and CogVideo [12, 13] extend this design with larger datasets and refined denoisers. Autoregressive alternatives like VideoPoet [14] and the Emu family [15–17] instead predict discrete visual tokens sequentially. These approaches remain clip-centric and lack mechanisms for narrative or visual consistency over longer horizons. We bridge this gap by introducing explicit consistency regularisation for full-length movie generation.

Interleaved Image–Text Modeling & Conditioning. Interleaved image–text generation [24–28] has emerged as a promising paradigm -for producing richly grounded multimodal content. Early efforts [29–31] leveraged large-scale image–text corpora [32, 33] but were largely restricted to single-turn tasks such as captioning or textto-image synthesis. The advent of large language models [34] and unified vision–language architectures [35–37] has enabled more sophisticated interleaved reasoning, yet most existing systems generate content only once and overlook coherence across multiple shots. VideoAuteur [38] addresses this problem through an interleaved VLM director and LCT [39] directly finetunes MM-DiTs to peform long interleaved video generation. Although Diffusion Forcing [40], LTX-Video [41] and Pusa [42] support multi-keyframe conditioning, they remain confined to single-shot scenarios. By contrast, our approach addresses multi-shot interleaved conditioning, explicitly modeling cross-shot coherence for long-form video generation.

Narrative Visual Generation. Existing research on narrative visual generation primarily focuses on ensuring semantic alignment and visual consistency. Recent methods—VideoDirectorGPT [43], Vlogger [44], Animatea-Story [45], VideoTetris [46], IC-LoRA [47], and StoryDiffusion [48]—pursue these goals through diverse architectural and training strategies. While most prior work concentrates on producing semantically consistent image sets [47–49], our objective is to generate fully coherent narrative videos. Although certain approaches condition synthesis on text [19, 44] or augment prompts with keyframes [50], our framework explicitly disentangles long-form generation into (i) interleaved keyframe synthesis and (ii) multi-frame-conditioned video completion. This design achieves state-of-the-art results in long narrative movie generation, offering superior visual fidelity, temporal coherence, and robustness to super-long context conditions.

Visual Generation with Token Compression. The computational demands associated with synthesizing highresolution visual content for video sequences necessitate efficient data representation strategies. FramePack [51] employs context packing in video generation to compress input frames, thereby maintaining a fixed context length for enhanced processing efficiency. FlexTok [52] utilizes an adaptive 1D tokenizer for images, adjusting token sequence length according to image complexity to achieve semantic compression. LTXVideo [53] uses compressed VAEs to improve model efficiency. PyramidFlow [54] performs video diffusion in a pyramid and renoises the multi-level latents. FAR [55] proposes a multi-level causal history structure to establish long-short-term causal memory and discuss KV cache integrations. HiTVideo [56] uses hierarchical tokenizers in video generation models and connect to autoregressive language models. Memory in discrete space and discrete keyframe intervals is less discussed but plays an important role in our framework to process frame contexts in an interleaved discrete space optimized for synthesizing long-form narrative visuals.

###### 3 Method

###### 3.1 Learning Long-Range Context in Movies

Different from [39, 47] that primarily gather video shots from single scenes, our approach directly learn from frames sampled across entire movies. Specifically, our keyframe generation model is trained on interleaved image-text pairs, while our video generation model learns from interleaved video-text pairs, significantly enhancing its generalization across scenes and cinematic contexts.

Data. We collect data from public available sources as our experimental dataset, amounting to a total length of approximately 500 hours. We process whole movies with the following data pipeline to produce interleaved keyframe-text pairs as well as shot-text pairs. After processing these data, we obtained roughly

###### Scene 1 Scene 2 Scene 3

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

|[Figure 30]|[Figure 31]|[Figure 32]|[Figure 33]|
|---|---|---|---|

|[Figure 34]|[Figure 35]|[Figure 36]|[Figure 37]|
|---|---|---|---|

|[Figure 38]|[Figure 39]|[Figure 40]|[Figure 41]|
|---|---|---|---|

[Figure 42]

[Figure 43]

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

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

[Figure 67]

[Figure 68]

[Figure 69]

High-angle shot overlooking the <Dursleys>. <Vernon Dursley> frantically locks a door with multiple padlocks. <Petunia Dursley> is held by <Dudley Dursley> in striped pajamas. The setting appears to be a cramped hallway in the <Dursleys>' house, illuminated with harsh lighting. The atmosphere is frantic and fearful.

A medium shot captures <Harry Potter> levitating a cake over a woman's head, hands outstretched in concentration. A man in a suit sits to her left, oblivious. The setting is a cluttered dining room with striped wallpaper and decorative lamps. The atmosphere is tense and magical, blending the mundane with the extraordinary.

The shot, framed by a barred window, features three characters. <Ron Weasley> is in the front, smiling slightly, while another ginger-haired character is slightly behind him to the left. Another character's face is partially visible behind and to the right of Ron. The light is subdued, creating a mysterious atmosphere. The setting appears to be inside a vehicle.

- Figure 2 Learn from whole movies. Here is an interleaved data sample processed from full-length movies. Our data pipeline extract structured narrative and visual information across scenes. Each frame is annotated with detailed visual description with major <Character Names> of the movie.

300,000 keyframes and video shots to train our top-down interleaved keyframe generation model (§3.2) and the bottom-up interleaved conditioned video generation model (§3.3).

Processing & Filtering. We detect scene cuts with PySceneDetect [57] and extract a mid-clip frame from each resulting segment. After removing black borders, we center-crop to a 2:1 aspect ratio and resize so the shorter side is 400 pixels. We then run Gemini-2.0 Flash [58] to discard low-quality or uninformative frames and to generate detailed captions for the remaining keyframes. To annotate characters, we include consistent <character name> tags to preserve IPs. Appendix B details the prompt used for keyframe annotation.

###### 3.2 Top-Down Interleaved Keyframe Planning

In this section, we introduce our method to finetune a pretrained Text-to-Image Model (i.e., Flux 1.Dev [59]) for interleaved keyframe generation with stability and efficiency.

Hybrid Attention Masking with MM-DiT. Our design builds on Flux, whose model is split into D double-stream (image–text) blocks followed by S single-stream blocks. Given P image–text pairs S = {(xi,yi)}Pi=1, with xi∈RL

I×d and yi∈RL

T×d, we concatenate the tokens as zi = [xi∥yi].

- 1. Local (double-stream) blocks. Each of the first D blocks uses a block–diagonal mask Mlocal = diag(M1,...,MP); such that zi attends only to itself.
- 2. Global (single-stream) blocks. Their outputs are concatenated Z = [z1∥...∥zP] and passed to the next S blocks. A full mask Mglobal ≡ 0 enables bi-directional inter-pair attention during training; swapping it for an upper-triangular mask yields causal, auto-regressive generation.

The two types of masking (cf., Fig. 3b) keeps early computation local and efficient while later blocks aggregate global context, achieving coherent interleaved key-frame generation.

GoldenMem: Compress Long-Context Visual Memory. As we target movie generation, how to design a long-context memory bank of generated visual frames is a challenge when the context growth. Inspired by the golden ratio squares with an upper bound area, we propose GoldenMem to compress long context visual frames through golden ratio downsamling with semantic-oriented context selection. To maintain a long visual

|[Figure 70]<br><br>| | | |[Figure 71]| |[Figure 72]<br><br>| | |[Figure 73]<br><br>| | |[Figure 74]<br><br>| |[Figure 75]| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | |M-Diffusio (Lo| |n Transfor cal Attentio| | |mer Block n)| | | | | | |
| | | | |M-Diffusio (Glo| |n Transfor bal Attenti| | |mer Block on)| | | | | | |

Less tokens but more memory context 2 4 6 10

…

[Figure 76]

|[Figure 77]|
|---|

|[Figure 78]|
|---|

[Figure 79]

| |[Figure 80]|
|---|---|
|[Figure 81]| |

Inverse Fibonacci Downsampling

[Figure 82]

[Figure 83]

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

[Figure 104]

[Figure 105]

(c) GoldenMem: Visual Context Compression

Image-

| | | | | | | |
|---|---|---|---|---|---|---|
|Mu|lti-keyfram|e Conditi (Glo|oned MMbal Attenti|Diffusion T on)|ransform|er|

- Text Pair 1 Image-
- Text Pair 2 Image-
- Text Pair 3 P 1 P2 P3 P 1 P2 P3 P 1 P2 P3

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

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

Intra-pair Attention (Bi-directional)

Inter-pair Attention (Bi-directional)

Inter-pair Attention (Causal)

(b) Hybrid Attention Masking

(a) Bridging Top-down & Bottom-up Interleaved Generation

- Figure 3 Method Overview. Captain Cinema bridges top-down and bottom-up interleaved for one-stage multi-scene movie generation. It introduces a hybrid attention masking strategy with GoldenMem context compression to learn and generate long movies efficiently and effectively. The number of GoldenMem tokens (referring to the short side of encoded image latents) is an example to show the inverse Fibonacci downsampling.

5512

11024

16536

22048

27560

33072

7824 8984 9696 10280 10824

0

10000

20000

30000

40000

4 8 12 16 20 24

Naïve GoldenMem

- Figure 4 GoldenMem compresses the the context length. The x-axis shows the number of image-text pairs, and the y-axis is the total number of tokens. Initial resolution is 400×800 (H×W).

[Figure 124]

Figure 5 Semantic-oriented context retrieval. CLIP text-to-image beats T5 text-totext for history context retrieval.

context without inflating the token budget, we store only the current frame at full resolution and encode earlier frames at progressively coarser scales. Let the golden ratio be φ = (1 + √5)/2 ≈ 1.618 and denote the short side of the newest latent by s0 (e.g., s0 = 25). We downsample the i-th previous frame by

si = si−1/φ , i = 1,...,k,

which yields the inverse-Fibonacci sequence (25,15,9,5,3) when k = 4. Each si × si latent is partitioned into non-overlapping p × p patches, contributing ti = (si/p)2 memory tokens. Because si decays geometrically, the total conditioning cost

k

φ2 φ2 − 1

ti = t0 1 + φ−2 + φ−4 + ... <

t0 ≈ 1.62t0

T =

i=0

remains a constant factor above the single-frame cost t0. Thus GoldenMem preserves a (k+1)-frame history with a fixed small overhead, as visualized in Figs. 3 and 4.

Semantic-Oriented Context Conditioning Beyond Temporal Distance. Although sequential generation is intuitive, films frequently employ non-linear devices such as flashbacks, foreshadowing, and temporal loops.

We therefore retrieve context by semantic similarity rather than temporal order, conditioning each new frame on embeddings obtained with CLIP (text–image) and T5 (text–text). This strategy accommodates complex narrative structures more effectively than strict temporal conditioning. As shown in Fig. 5, CLIP Text-to-Image retrieval consistently yields higher coverage than T5-Text-to-Text retrieval across retrieval depths, leading to better memory recall for extracting important, most relevant history frames.

Progressively Finetuning with Growing Context Length. Directly finetuning on long context interleaved sequences are prone to model collapse and often generate messy backgrounds with broken semantics (§4.2). To facilitate stable training over interleaved sequences, we use a progressive training strategy to finetune the model on interleaved data with growing context length. Specifically, we warm up the model with single image generation and then progressively finetune the model with 8, 16, and 32 interleaved pairs.

Dynamic Stride Sampling for Interleaved Sequences. As our movie data scale is limited, naively sample consecutive interleaved movie keyframes can lead a large MM-DiT [59] models to get overfitted and be less robust. In this way, we use a dynamic stride sampling strategy to sample interleaved data, which provides thousands of times more valid data sequences (a bar of 25% overlap rate) than naive consecutive sampling.

- 3.3 Bottom-Up Keyframe-Conditioned Video Generation Our framework bridges two complementary viewpoints. From a bottom-up perspective, we begin with a base video generator and expand its scope to long-form content by conditioning on a sequence of interleaved shots. From a top-down perspective, we first construct a sparse set of narrative keyframes that act as visual anchors, then instruct the generator to fill the intervals between them. Unifying these perspectives allows us to preserve local visual fidelity while maintaining global narrative coherence throughout the entire video. Given K keyframes {I1,...,IK}, each the first frame of its shot, we condition a diffusion generator on (i) the tiled global caption ctiled and (ii) the visual embeddings of all keyframes:

Vk = Ik, Gθ I1:K, ctiled, t = 2:Tk , k = 1,...,K, where Gθ outputs frames fk,2,...,fk,T

k

for shot k. Multi-key conditioning thus anchors appearance at shot boundaries and enforces seamless motion dynamics across shots, generating videos that preserve narrative intent, visual details, and temporal coherence with strong visual consistency.

4 Experiment

Experimental Setup. We finetune Flux 1.Dev [59] on our interleaved movie dataset (described in §3.1) for interleaved keyframe generation. All keyframe models are trained for 40,000 steps with a batch size of 32 using 32 H100 GPUs. For interleaved video generation, we adopt Seaweed-3B [39, 60] as the base model and finetune it with multi-frame interleaved conditioning for 15,000 steps using 256 H100 GPUs. Additional implementation details are provided in the appendix. The keyframes are cropped with max area to 400x800 resolution without aspect ratio changes while the video generation data processing resizes all the videos to an approximation area of 230,040 total pixels. More implementation details of training our bottom-up and top-down keyframe and video generation models are provided in Appendix A.

- 4.1 Main Results

Qualitative results. Fig. 6 shows a qualitative result with a storyline generated by Gemini 2.5. Specifically, we first prompt Gemini to generate a sequence of long-form captions describing a storyline inspired by Bruce Wayne, Alfred Pennyworth, and Joker, and interstellar travel. We then use our interleaved top-down keyframe generation model to generate shot-wise keyframes based on the narrative. Finally, we construct interleaved text–keyframe pairs from the generated content and use them as conditioning input for the LCT generation model to produce the final movie. The results demonstrates compelling qualitative performance in the aspects of visual quality, consistency and semantic alignment with prompts. We also provide additional qualitative results of keyframe generation in Appendix C and also multi-scene movie results in the supplementary material.

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

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Shot 1] wide shot of Bruce and Alfred [Shot 2] medium shot of Joker standing [Shot 3] close-up of hand on display [Shot 4] medium shot of Bruce sitting

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

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

[Shot 8] B & A looking out viewport

[Shot 5] Joker gazing at a viewscreen [Shot 6] exterior shot of the spaceship [Shot 7] docking of space station

### ...

### ...

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Shot 17] seated opposite in Cafe [Shot 18] Joker facing to camera talking [Shot 19] Bruce looking resolutely

[Shot 20] medium close-up Bruce in helmet

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

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

[Shot 21] Joker takes the control panel [Shot 22] low-angle close-up of Bruce [Shot 23] low-angle shot of spaceship

[Shot 24] wide shot of a spectacular

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Expansion of [Shot 23]

- Figure 6 Qualitative results. From the narrative prompt “An interstellar voyage with Bruce Wayne, the Joker, and Alfred Pennyworth," Gemini 2.5 Pro composes shot-level descriptions that guide our top-down key-frame generator, yielding the storyboard panels shown above. Each text–key-frame pair then conditions our bottom-up video model, which synthesises the full multi-scene film. The figure highlights twenty-four representative shots demonstrating sustained narrative coherence, character fidelity, and visual style across the entire production.

Quantitative Evaluation & User Study. We follow VBench-2.0 [61] to evaluate visual and temporal aspects, and follow the LCT [39] protocol for assessing text–semantic alignment. A user study is conducted using a 4-point scale—Very Good, Good, Poor, and Very Poor—focusing on semantic alignment and overall visual quality. As few works target long-video generation, we primarily compare against two closely related baselines: LCT and IC-LoRA [47] combined with I2V [60]. To ensure a fair comparison, we use GPT-4o to format consistent scene prompts and generate videos across all methods. Tab. 1 shows that our approach performs

|Method<br><br>|Visual<br><br>Aesthetic↑ Quality↑|Temporal<br><br>Consistency↑ Dynamic↑<br><br>|Semantic<br><br>Text↑|User Study<br><br>Quality↑ Semantic↑|
|---|---|---|---|---|
|IC-LoRA [47] + I2V LCT [39]<br><br>Ours (w/o MF-FT) Ours<br><br>|54.1 60.5 56.2 59.9<br><br>56.8 60.9<br><br>57.2 61.7<br><br><br>|88.7 61.1<br><br>94.8* 51.8 91.9 64.4 91.0 65.4<br><br>|23.1 23.9 25.7<br><br>26.1|1.5 2.3<br>2.4 3.1<br><br><br>2.8 3.5<br><br>3.3 3.7<br><br><br>|

- Table 1 Quantitative Evaluation & User Study. We employ automatic metrics and average human ranking (AHR). “Consistency” represents the average score of subject and background consistency. *: Most video clips show low temporal dynamic but evaluated to have high temporal consistency on VBench metrics [61].

| |Consistency|Visual Quality<br><br>|Diversity Narrative Identity|
|---|---|---|---|
|Method Ctx. Pairs<br><br>|Character Scene|Visual Aesthetic|Diversity Nar. Coher. Identity|
|LCT [39] 8 LCT [39] 16 LCT [39] 24<br><br>|4.3 3.5 3.6 3.4 3.1 0.7|4.8 4.1 4.5 3.9 1.6 1.6<br><br>|3.0 2.8 0.43 2.5 2.8 0.31 1.7 0.6 0.14<br><br>|
|Ours 8 Ours 16 Ours 24 Ours 32|4.9 3.9 4.8 3.8 4.6 3.8 4.5 3.8<br><br>|4.9 4.7 4.9 4.6 4.9 4.6 4.9 4.6|4.5 4.0 0.51 4.2 3.8 0.47 4.2 3.6 0.42 3.9 3.4 0.37<br><br>|
|Ours (w/ G.Mem) 16 Ours (w/ G.Mem) 32 Ours (w/ G.Mem) 48|4.6 3.8 4.6 3.5 4.5 3.0<br><br>|4.9 4.6 4.7 4.6 4.7 4.5<br><br>|4.2 3.8 0.44 4.0 3.5 0.35 3.9 3.3 0.31|

- Table 2 Long-Context Stress Test. We exhibit long context stress test to benchmark the robustness of long context generation. We use Gemini 2.5 Flash to rate the generation quality in multiple aspects (detailed in Appendix D), and an automatic identity consistency metric introduced in VBench 2.0. Our interleaved method with GoldenMem can do high-quality long context generation with strong consistency preservation of both characters and scenes.

favorably across most metrics, a finding corroborated by user studies evaluating video quality and semantic relevance. Notably, our advantages are most evident in temporal dynamics, which are crucial for generating coherent and vivid motion over long sequences in cinematic content creation.

Long-context stress test. We evaluate robustness as the context window grows from 8→48 interleaved pairs. For each setting we generate 20 clips and ask Gemini Flash 2.5 to score Consistency, Visual Quality, Diversity, and Narrative Coherence; and another Identity Consistency metric used in VBench-2.0. As context length

[Figure 209]

[Figure 210]

(a) Consistency (b) Visual Quality

[Figure 211]

[Figure 212]

(c) Diversity (d) Narrative Coherence

- Figure 7 Long-Context Stress Test. While context length grows to be very long, our method still maintains strong visual consistency, high visual quality, diversity, and narrative coherence even under extended context lengths.

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

###### No Warmup Over Warmup (Step 80K) Moderate Warmup (Step 40K)

- Figure 8 Progressive long-context finetuning needs modest warmup. As the base model (FLUX 1.Dev) is a distillation model with guidance, it requires progressive finetuning with growing context length to warmup the model. However, over warmup can also lead the model to forget the distilled knowledge for high-quality keyframe generation, generating artifacts or messy textures. Picking a proper warm-up length is therefore important for long context tuning.

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

(a) Multi-frame Finetuning Makes Less Artifacts

[Figure 231]

[Figure 232]

(b) Context-aware “World” Generation

(c) Smooth Camera Movement

- Figure 9 Disentangled task modeling makes robust movie generation. With the high-quality keyframes generated by our top-down keyframe generation model, the bottom-up video generation model can now focus on motion dynamics. The generation performs higher temporal consistency, world-context awareness and smoother camera movement.

increases, LCT quality degrades sharply (Tab. 2), whereas our model retains >93% of its 8-pair consistency even at 48 pairs, validating the effectiveness of GoldenMem and progressive finetuning ( Fig. 7).

###### 4.2 Ablation Studies

From Short to Long: Progressive Long-context Finetuning. As detailed in §3.2, we employ progressive long-context fine-tuning, gradually expanding the context window of the interleaved keyframe generator. Fig. 8 shows that directly fine-tuning the FLUX base model leads to training collapse. On the other hand, fine-tuning from a late checkpoint (step 80,000; half of the target context length) also produces visual artifacts, which is likely caused by the knowledge forgetting for the FLUX distillation base model. So, progressive finetuning with FLUX-like distillation models needs a modest warmup (i.e., step 40,000) to avoid training collapse or forgetting distilled knowledge of inherited guidance scale conditioning.

GoldenMem: Compressed Long-Context Memory. As detailed in §3.2, we downsample historical visual context at the pixel level in inverse Fibonacci sequence, thereby capping the overall number of visual tokens. Tab. 3 ablates this design with different number of interleaved pairs. GoldenMem demonstrates strong compute efficiency when generating with the same visual context with just minor visual quality and consistency degradation. Moreover, GoldenMem enables longer context window (i.e., from 32 to 48 without OOM), which keeps longer context history information for long video generation.

Robust Long Video Generation with Disentangled Task Modeling. As detailed in §3.2, we fine-tune a pretrained MM-DiT video generator [60] with multi-frame interleaved conditioning to serve as our bottom-up component. Coupled with the top-down interleaved keyframe generator, this design improves robustness, yielding fewer artifacts (Fig. 9a), consistent environment dynamics—e.g., burning ruins with rising smoke (Fig. 9b)—and smooth camera motion with stable character identities (Fig. 9c). These results demonstrate that disentangling high-level narrative planning from low-level motion synthesis enables efficient and robust long-video generation.

|# Pairs<br><br>|GoldenMem Compute<br><br># Groups × Images PFLOPS (Est.)<br><br>|Visual Text<br><br>Quality↑ Consistency↑ Alignment↑|
|---|---|---|
|16-frame 16-frame<br><br>|– 30 3 × 4 21<br><br>|4.4 4.7 4.1 4.3 4.3 4.1<br><br>|
|32-frame 32-frame<br><br>|– 55 3 × 8 35<br><br>|4.1 4.3 3.8 3.9 4.2 3.6<br><br>|
|48-frame 48-frame<br><br>|– OOM 3 × 12 52<br><br>|– – – 3.6 4.2 3.5<br><br>|

- Table 3 GoldenMem makes longer context with less computation. The first row of each block is the baseline while the second row is using GoldenMem. GoldenMem uses less computation with performance preservation and processes

longer context window without out-of-memory (OOM). Compute is estimated through #tokens×steps×FLOP/token.

|[Figure 233]<br><br>[Figure 234]<br><br>…<br><br>…<br><br>…<br><br>…|
|---|

[Figure 235]

Condition NoiseScale

- Figure 10 Visual context conditioning needs moderate noise injection. Each conditioning frame has individually been injected with random noise at varying levels: 1st: 751∼1000; 2nd: 501∼750; 3rd: 251∼500; 4th: 1∼250. Context conditioning with moderate levels of noise lead to better character and scene consistency.

Dynamic Strided Data Sampling. Because the scale of our movie dataset is relative small to the data used for base-model pre-training, naïve sampling induces rapid overfitting and weak generalization, particularly under sparse keyframe conditioning and limited interleaved samples. We mitigate this by introducing a dynamic strided sampling scheme that systematically offsets the sampling stride across epochs. This strategy yields 100 times more valid data when sample 16-frame sequence from 32 frames (25% keyframe-overlap threshold).

Noisy visual context conditioning. During our interleaved keyframes generation model training, each frame is applied with individual level of noise. So, we ablate the noise level for keyframe extending generation with a given context of noisy latents. As shown in Fig. 10, our interleaved keyframe generation model fails to keep character consistency when conditioning on context with large noise injection (i.e. 501∼1000 steps). However, our model shows very strong character consistency when using moderate noise level (i.e. 1∼500 steps) and shows best character facial perservation when using a small noise (i.e. adding 1∼250 timesteps of noise) for long context conditioning. We also evaluate out-of-domain generalization to unseen characters in Appendix E.

###### 4.3 Generalization Abilities

Creative Scene Generation. Our method could synthesize novel scenes that were never present in the training corpus by recombining familiar characters, settings, and plot elements. Leveraging semantic-oriented retrieval and interleaved conditioning, it produces imaginative environments, e.g., a prison-bound Batman confronting the Gotham under Joker’s control, while preserving visual fidelity and narrative plausibility through one-stage generation. Our model effectively understands character traits, typical settings, and plausible interactions, enabling seamless integration of new creative ideas into cohesive, realistic scenes. Our model could extend any scene through context conditioning and continue story-telling with characters and scenes consistency.

Cross-Movie Character Swapping. Our pipeline also enables seamless swapping of characters across unrelated cinematic universes. Leveraging identity-preserving embeddings, the system can insert Bruce Wayne and

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

(a) Creative scene generation. Batman is locked in prison while Joker takes control of Gotham.

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

(b) Cross-movie character swapping. Bruce Wayne and Alfred Pennyworth appear in “Interstellar.”

- Figure 11 Captain Cinema generalization abilities. Trained on high-quality, richly annotated interleaved data, our model offers creative control at the movie, scene, and character levels, allowing users to generate limitless “parallel universes” that maintain coherent narratives and high visual fidelity.

Alfred Pennyworth into the science-fiction environment of Interstellar, where they interact convincingly with the new setting and its supporting cast. This capability demonstrates a clear separation between character identity and scene context, providing a flexible platform for counterfactual narrative exploration.

###### 5 Limitations & Conclusion

Our method still faces several constraints. (1) Absence of Image-to-Video End-to-End Training. Although end-to-end optimization is theoretically attainable, prevailing memory and infrastructure constraints compel us to train the frame-level and video-level modules separately, subsequently employing the generated frames as conditioning inputs for video synthesis. (2) Dependence on external prompts. The model is not yet capable of autonomously devising narratives akin to state-of-the-art multi-modal LLMs; it therefore relies on text supplied by human authors or large language models. (3) Data hunger. The method’s capacity to generalize is curtailed by the scarcity of high-quality, feature-length movie datasets and thus demands extensive real-world validation efforts, larger corpora, and additional architectural refinements.

Conclusion. We propose Captain Cinema, based on top-down interleaved keyframe planning with bottom-up, multi-keyframe-conditioned video synthesis to generate a short movie. Leveraging the GoldenMem compressed visual context, progressive long-context fine-tuning, and dynamic strided sampling training strategies, our model sustains global narrative coherence while preserving local visual fidelity throughout feature-length videos. Captain Cinema also demonstrates generalization ability for creative scene generation and cross-movie character swapping. Despite the limitations discussed above, we believe our work represents a concrete step toward fully automated, story-driven movie generation and inspires future cinematic research.

Broader Impact. Long-form video generation could democratize high-quality animation, documentary creation, education, and simulation, enabling content production that previously demanded extensive budgets and expertise. This technology also provides valuable tools for individuals with limited mobility or resources and offers new avenues for reinforcement learning and robotics simulations. However, accessible video generation also raises concerns around hyper-realistic misinformation, non-consensual media, intellectualproperty violations, and environmental impact due to high computational costs. To mitigate these risks, we will adopt gated model releases, implement robust watermarking, provide model documentation, and enforce clear usage policies. Additionally, we will develop watermark verification tools, conduct external red-team audits, and collaborate with stakeholders on content detection benchmarks.

###### References

- [1] Jerome Bruner. The narrative construction of reality. Critical inquiry, 18(1):1–21, 1991.

- [2] Yuval Noah Harari. Sapiens: A brief history of humankind. Random House, 2014.

- [3] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NeurlPS, 2022.

- [4] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. In arXiv, 2022.

- [5] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. In arXiv, 2023.

- [6] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, 2024.

- [7] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. In arXiv, 2022.

- [8] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. In arXiv, 2023.

- [9] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. In arXiv, 2022.

- [10] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.

- [11] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators, 2024.
- [12] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In arXiv, 2022.

- [13] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In arXiv, 2024.

- [14] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. In ICML, 2024.

- [15] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. In ICLR, 2023.

- [16] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In CVPR, 2024.

- [17] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. In arXiv, 2024.

- [18] Xi Chen, Zhiheng Liu, Mengting Chen, Yutong Feng, Yu Liu, Yujun Shen, and Hengshuang Zhao. Livephoto: Real image animation with text-guided motion control. In ECCV, 2025.

- [19] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In CVPR, 2024.

- [20] Ruoyu Feng, Wenming Weng, Yanhui Wang, Yuhui Yuan, Jianmin Bao, Chong Luo, Zhibo Chen, and Baining Guo. Ccedit: Creative and controllable video editing via diffusion models. In CVPR, 2024.

- [21] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In CVPR, 2023.

- [22] Nisha Huang, Yuxin Zhang, and Weiming Dong. Style-a-video: Agile diffusion for arbitrary text-based video style transfer. IEEE Signal Processing Letters, 2024.

- [23] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.

- [24] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. In arXiv, 2023.

- [25] Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, et al. Mm-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronizer. In arXiv, 2024.

- [26] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. In arXiv, 2024.

- [27] Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen. Seed-story: Multimodal long story generation with large language model. In arXiv, 2024.

- [28] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurlPS, 2022.

- [29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

- [30] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking. In CVPR, 2023.

- [31] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. In arXiv, 2023.

- [32] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. In NeurlPS, 2024.

- [33] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurlPS, 2022.

- [34] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- [35] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023.

- [36] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurlPS, 2024.

- [37] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. In Advances in Neural Information Processing Systems, 2024.

- [38] Junfei Xiao, Feng Cheng, Lu Qi, Liangke Gui, Jiepeng Cen, Zhibei Ma, Alan Yuille, and Lu Jiang. Videoauteur: Towards long narrative video generation. arXiv preprint arXiv:2501.06173, 2024.

- [39] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv preprint arXiv:2503.10589, 2025.

- [40] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.

- [41] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

- [42] Yaofang Liu and Rui Liu. Pusa: Thousands timesteps video diffusion model, 2025.
- [43] Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. In COLM, 2024.

- [44] Shaobin Zhuang, Kunchang Li, Xinyuan Chen, Yaohui Wang, Ziwei Liu, Yu Qiao, and Yali Wang. Vlogger: Make your dream a vlog. In CVPR, 2024.

- [45] Yingqing He, Menghan Xia, Haoxin Chen, Xiaodong Cun, Yuan Gong, Jinbo Xing, Yong Zhang, Xintao Wang, Chao Weng, Ying Shan, et al. Animate-a-story: Storytelling with retrieval-augmented video generation. arXiv preprint arXiv:2307.06940, 2023.

- [46] Ye Tian, Ling Yang, Haotian Yang, Yuan Gao, Yufan Deng, Jingmin Chen, Xintao Wang, Zhaochen Yu, Xin Tao, Pengfei Wan, et al. Videotetris: Towards compositional text-to-video generation. arXiv preprint arXiv:2406.04277, 2024.

- [47] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024.

- [48] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. arXiv preprint arXiv:2405.01434, 2024.

- [49] Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen. Seed-story: Multimodal long story generation with large language model. arXiv preprint arXiv:2407.08683, 2024.

- [50] Canyu Zhao, Mingyu Liu, Wen Wang, Jianlong Yuan, Hao Chen, Bo Zhang, and Chunhua Shen. Moviedreamer: Hierarchical generation for coherent long visual sequence. arXiv preprint arXiv:2407.16655, 2024.

- [51] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation, 2025.
- [52] Roman Bachmann, Jesse Allardice, David Mizrahi, Enrico Fini, Oğuzhan Fatih Kar, Elmira Amirloo, Alaaeldin El-Nouby, Amir Zamir, and Afshin Dehghan. FlexTok: Resampling images into 1d token sequences of flexible length, 2025.
- [53] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

- [54] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling, 2024.
- [55] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction, 2025.
- [56] Ziqin Zhou, Yifan Yang, Yuqing Yang, Tianyu He, Houwen Peng, Kai Qiu, Qi Dai, Lili Qiu, Chong Luo, and Lingqiao Liu. Hitvideo: Hierarchical tokenizers for enhancing text-to-video generation with autoregressive large language models. arXiv preprint arXiv:2503.11513, 2025.

- [57] Breakthrough. Pyscenedetect v0.6. https://github.com/Breakthrough/PySceneDetect, 2021. Python-based video scene detection library.
- [58] Google Gemini Team. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- [59] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [60] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, et al. Seaweed-7b: Cost-effective training of video generation foundation model. arXiv preprint arXiv:2504.08685, 2025.

###### [61] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

## Appendix

###### A Implementation Details

We provide the training hyperparameters for the top-down keyframe generation model and the interleaved conditioned video generation model in Tab. 4. The interleaved auto-regressive model is trained on interleaved image-text pairs with a default resolution of 400 × 800, using a batch size of 32 and bfloat16 precision. It employs AdamW as the optimizer, with a peak learning rate of 2 × 10−4 and a cosine decay schedule, training for 2,500 steps. Training context pairs vary between 2 and 8, while inference always uses 8 pairs for consistency. The visual-conditioned video generation model processes video data at an area of 480x480 ( 230,400 total pixels), with a batch size of 64 and bfloat16 precision. It uses AdamW with a peak learning rate of 1 × 10−5 and a constant decay schedule, training for 20,000 steps to handle temporal conditioning effectively. We provide the training hyperparameters for the top-down keyframe generation model and the interleaved conditioned video generation model in Tab. 4 The interleaved auto-regressive model is trained on interleaved image-text pairs with a default resolution of 400 × 800, using a batch size of 32 and bfloat16 precision. It employs AdamW as the optimizer, with a peak learning rate of 2 × 10−4 and a cosine decay schedule, training for 2,500 steps. Training context pairs vary between 2 and 8, while inference always uses 8 pairs for consistency. The visual-conditioned video generation model processes video data at an area of 480x480 ( 230,400 total pixels), with a batch size of 64 and bfloat16 precision. It uses AdamW with a peak learning rate of 1 × 10−5 and a constant decay schedule, training for 20,000 steps to handle temporal conditioning effectively.

|Configuration Keyframe Generation Model<br><br>|Video Generation Model|
|---|---|
|Resolution 400 × 800 × L Optimizer AdamW Optimizer hyperparameters β1=0.9, β2=0.999, ϵ=10−8 Peak learning rate 4 × 10−4 Learning-rate schedule Linear warm-up, cosine decay Gradient clip 1.0 Training steps 20,000 (per stage) Warm-up steps 500 Batch size 16 Numerical precision bfloat16 Computation 32 × H100, ∼72 h<br><br>|(480P, native AR) × T AdamW β1=0.9, β2=0.95, ϵ=10−8 1 × 10−4 Linear warm-up, cosine decay 1.0 10,000 (total) 1,000 ≈ 64 bfloat16 256 × H100, ∼200 h|

Table 4 Implementation details of our models. The left column lists the interleaved keyframe generator settings, and the right column lists the visual-conditioned video generator settings.

###### B Data Annotation Prompt

You are a film-director assistant annotating a single frame from the movie.

###### 1. Detailed Caption

- • Describe visual composition, lighting, and camera angles.
- • Note character actions, expressions, and positioning.
- • Mention setting details and atmosphere.

###### 2. Character Identification

- • Identify characters only when absolutely certain, enclosing names in angle brackets (e.g. <Character Name ). Allowed list: <Character List>.

- • If uncertain, describe by appearance, role, or action—do not use angle brackets.

- • For dual-identity roles, choose the name that matches the on-screen persona.

###### 3. Level of Detail

- • Craft concise yet thorough sentences capturing technical aspects (framing, lighting) and narrative elements (interactions, mood).
- • Focus on specifics that make the frame distinctive.

###### C Additional Qualitative Results for Keyframe Generation. We provide non-cherry pick qualitative keyframe generation results in figures below.

[Figure 244]

- Figure 12 Qualitative Result. Our method shows superior generation quality with character and scene consistency. 32 frames are generated by our method through diffusion forcing with bidirectional masking strategy. The prompt here is from the validation split with ChatGPT-4o re-imagined.

[Figure 245]

###### Figure 13 Qualitative Result. Our method shows superior generation quality with character and scene consistency. 32 frames are generated by our method through diffusion forcing with bidirectional masking strategy. The prompt here is from the validation split with ChatGPT-4o re-imagined.

[Figure 246]

###### Figure 14 Qualitative Result. Our method shows superior generation quality with character and scene consistency. 32 frames are generated by our method through diffusion forcing with bidirectional masking strategy. The prompt here is from the validation split with ChatGPT-4o re-imagined.

[Figure 247]

###### Figure 15 Qualitative Result. Our method shows superior generation quality with character and scene consistency. 32 frames are generated by our method through diffusion forcing with bidirectional masking strategy. The prompt here is from the validation split with ChatGPT-4o re-imagined.

###### D Long-Context Stress Test Details

We evaluate the robustness of our model under long-range dependencies by scoring sequences of keyframes—both ground-truth and those sampled from our generated clips— with Gemini 2.5 Flash. Guided by the system prompt in Appendix D, the evaluator rates each sequence along six dimensions: character consistency, scene consistency, visual quality, aesthetics, diversity, and narrative coherence. An overview of the per-aspect trends is visualised in Fig. 7, and the complete results for a range of context lengths are reported in Tab. 2.

###### SYSTEM_PROMPT

You are an expert visual analyst. You will receive a sequence of movie keyframes (static images) and must evaluate them in six aspects:

- 1. Character Consistency
- 2. Scene Consistency
- 3. Visual Quality
- 4. Aesthetics
- 5. Diversity (no duplicate or near-duplicate frames)
- 6. Narrative Coherence (does the sequence form a sensible mini-story?) For each aspect, assign a 0–5 score and provide a one-sentence justification.

Score meanings: | Score | Meaning | |-------|---------------------------------------------|

- | 0 | Unacceptable: completely fails expectations |
- | 1 | Very Poor: major problems |
- | 2 | Poor: noticeable flaws |
- | 3 | Fair: meets basic expectations |
- | 4 | Good: solid, minor imperfections |
- | 5 | Excellent: flawless or near-perfect | Scoring rules

- - Character Consistency: clothing, hairstyle, facial features, and proportions remain constant.
- - Scene Consistency: lighting and background elements stay coherent across frames.
- - Visual Quality: frames are sharp, well-exposed, and color-balanced.
- - Aesthetics: composition, color palette, and contrast are deliberate and pleasing.
- - Diversity: frames are meaningfully distinct; penalize repetition.
- - Narrative Coherence: frames suggest a logical progression; penalize abrupt shifts. Return valid JSON in the following structure:

{

"character_consistency": { "score": <0-5>, "justification": "<text>" }, "scene_consistency": { "score": <0-5>, "justification": "<text>" }, "visual_quality": { "score": <0-5>, "justification": "<text>" }, "aesthetics": { "score": <0-5>, "justification": "<text>" }, "diversity": { "score": <0-5>, "justification": "<text>" }, "narrative_coherence": { "score": <0-5>, "justification": "<text>" }

}

- E Out-of-domain Identity Preservation Generalizability We assess our model’s ability to preserve unseen identities under noisy conditioning. For each out-of-domain individual, we collect 16 photographs with accompanying captions. Twelve images serve as interleaved context; their visual embeddings are perturbed with Gaussian noise for robust conditioning. The remaining four captions are then used to prompt the model, which must synthesize four new frames of the same person. Qualitative results are shown in Figs. 16 and 18 (generation with different dressing) and Figs. 17 and 19 (generation with different dressing). Our interleaved keyframe generator effectively maintains identity and accurately follows the given prompting instruction (appearance descriptions), demonstrating strong generalization to unseen characters.

[Figure 248]

[Figure 249]

- Figure 16 Out-of-domain Identity-Preservation Test : Different Dressing. Each test case begins with 12 randomly sampled context images that are injected with noise; the model then generates another four frames conditioned on textual prompts as well as the pretext interleaved conditions. Our interleaved key-frame generator accurately maintains the character’s identity, demonstrating strong out-of-domain generalizability.

[Figure 250]

[Figure 251]

###### Figure 17 Out-of-domain Identity-Preservation Test : Same Dressing. Each test case begins with 12 randomly sampled context images that are injected with noise; the model then generates another four frames conditioned on textual prompts as well as the pretext interleaved conditions. Our interleaved key-frame generator accurately maintains the character’s identity, demonstrating strong out-of-domain generalizability.

[Figure 252]

[Figure 253]

###### Figure 18 Out-of-domain Identity-Preservation Test : Different Dressing. Each test case begins with 12 randomly sampled context images that are injected with noise; the model then generates another four frames conditioned on textual prompts as well as the pretext interleaved conditions. Our interleaved key-frame generator accurately maintains the character’s identity while follows the instruction prompts, demonstrating strong identity preservation ability to out-of-domain characters.

[Figure 254]

[Figure 255]

###### Figure 19 Out-of-domain Identity-Preservation Test : Same Dressing. Each test case begins with 12 randomly sampled context images that are injected with noise; the model then generates another four frames conditioned on textual prompts as well as the pretext interleaved conditions. Our interleaved key-frame generator accurately maintains the character’s identity, demonstrating strong out-of-domain generalizability.

