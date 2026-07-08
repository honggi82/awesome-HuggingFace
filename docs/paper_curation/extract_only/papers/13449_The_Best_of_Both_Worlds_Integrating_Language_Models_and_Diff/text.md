The Best of Both Worlds: Integrating Language Models and Diffusion Models
for Video Generation
Aoxiong Yin * 1 Kai Shen * 2 Yichong Leng 2 Xu Tan 2
Xinyu Zhou 2
Juncheng Li 1 Siliang Tang 1
https://github.com/LanDiff/LanDiff
Abstract
Recent advancements in text-to-video (T2V) gen-
eration have been driven by two competing
paradigms: autoregressive language models and
diffusion models. However, each paradigm has
intrinsic limitations: language models struggle
with visual quality and error accumulation, while
diffusion models lack semantic understanding
and causal modeling. In this work, we propose
LanDiff, a hybrid framework that synergizes the
strengths of both paradigms through coarse-to-
fine generation. Our architecture introduces three
key innovations: (1) a semantic tokenizer that
compresses 3D visual features into compact 1D
discrete representations through efficient seman-
tic compression, achieving a ∼14,000× com-
pression ratio; (2) a language model that gen-
erates semantic tokens with high-level semantic
relationships; (3) a streaming diffusion model
that refines coarse semantics into high-fidelity
videos. Experiments show that LanDiff, a 5B
model, achieves a score of 85.43 on the VBench
T2V benchmark, surpassing the state-of-the-art
open-source models Hunyuan Video (13B) and
other commercial models such as Sora, Kling, and
Hailuo. Furthermore, our model also achieves
state-of-the-art performance in long video gen-
eration, surpassing other open-source models in
this field. Our demo can be viewed at https:
//landiff.github.io/.
1. Introduction
Text-to-video (T2V) (Blattmann et al., 2023; Kondratyuk
et al., 2024; Wang et al., 2024a; Yang et al., 2024b) has
*Equal contribution 1College of Computer Science and Tech-
nology, Zhejiang University, Hangzhou, China 2Moonshot AI, Bei-
jing, China. Correspondence to: Xu Tan <tanxu@moonshot.cn>,
Juncheng Li <junchengli@zju.edu.cn>.
Perceptual  Feature
Video
Semantic Feature
Little boy blowing out
birthday candles
Our Video Generation Process
Low Rate
High Rate
bits/dim
High Distortion
Low Distortion
RMSE
LLM
Diffusion
VAE
Figure 1. The rate-distortion curve illustrates how visual distortion
decreases as the number of transmitted bits increases. With just a
small number of bits representing high-level semantic features, we
can already achieve relatively low visual distortion. Building on
this information-theoretic insight, LanDiff combines the strengths
of both paradigms: LLMs efficiently generate compact semantic
features in the first stage, followed by diffusion models that add
perceptual details in the second stage, before final decoding to pix-
els via VAE. Data from Ho et al. (2020), illustration is conceptual.
made significant progress in recent years, becoming an im-
portant research direction in the fields of computer vision
and artificial intelligence. Recent works in T2V models
have primarily revolved around two predominant paradigms:
autoregressive large language model (LLM)-based (Kon-
dratyuk et al., 2024; Wang et al., 2024a) frameworks and
diffusion-based architectures (Blattmann et al., 2023; Yang
et al., 2024b). However, each paradigm has their own ad-
vantages and limitations, as shown in Table 1.
From a representation perspective, LLM-based methods
leverage discrete tokenization to explicitly encode high-level
semantics through vector quantization, effectively prioritiz-
ing conceptual abstraction and narrative coherence. How-
ever, this discretization inherently sacrifices low-level visual
fidelity due to information compression, resulting in low re-
construction quality. In contrast, diffusion-based approaches
employ continuous latent representations to preserve much
more perceptual details, enabling superior reconstruction
quality at the cost of diluted semantic interpretability, as hier-
1
arXiv:2503.04606v3  [cs.CV]  29 Apr 2025

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
Table 1. The comparison between LLM based and diffusion based
T2V systems. The advantages and disadvantages are marked by
and
, respectively.
Methods
Representations
Modeling
LLM
Discrete Tokens
Low Reconstruction Quality
Highlight Semantic Information
Autoregressvie
No Refinement
Causal Modeling
Diffusion
Continuous Vectors
High Reconstruction Quality
Lack Semantic Information
Non-Autoregressvie
Progressive Refinement
Non-causal Modeling
archical features remain entangled in the latent space. From
a generative modeling perspective, LLM-based systems
adopt autoregressive modeling to enforce causal dependen-
cies between video frames, ensuring strong temporal co-
herence. However, the autoregressive generation inherently
risks error propagation across time steps, where inaccuracies
in early predictions amplify during decoding. In contrast,
diffusion-based methods employ non-autoregressive genera-
tion, refining outputs in parallel through iterative denoising
steps. Although this design mitigates error accumulation
and enhances generation flexibility, the absence of explicit
causal constraints often leads to temporal inconsistencies or
semantic hallucinations.
In this work, we propose a hybrid architecture that syner-
gizes the strengths of both Language models and Diffusion
models, named LanDiff, through a coarse-to-fine generation
paradigm. As shown in Figure 1, inspired by the human
creation of video which will generate the high-level story-
line first and then add low-level visual details based on the
storyline to form the video, we design a two-stage video gen-
eration process with the number of bits gradually increasing
and carefully design the autoregressive model and the dif-
fusion model to be responsible for different stages of T2V
generation, so as to play to their strengths and avoid their
weaknesses. In detail, 1) at the low-bit position “semantic
feature”, the low-bit information ensures that the token se-
quence is not too long, and the high-level information makes
it easier for the model to capture the overall semantic entity
motion of the video, so as to play to the strengths of the au-
toregressive model and avoid its weaknesses. Thus we use
LLM to generate a coarse-grained video in the first stage;
2) at the high-bit position “perceptual feature”, since we
have already obtained the coarse-grained with rich semantic
and time-serial information, we only need to focus on how
to add details to the coarse-grained video. Thus we apply
a diffusion model in the second stage. Finally, a VAE de-
coder transforms the generated “perceptual feature” into the
final RGB video output. By unifying these complementary
mechanisms, we demonstrate that hybrid architectures can
overcome the inherent limitations of isolated approaches,
enabling coherent, semantically faithful, and visually com-
pelling video generation from textual descriptions, as shown
Table 2. The comparison between LanDiff and previous large-scale
T2V systems. The compression rates of LLM-based and diffusion-
based models are illustrated using VideoPoet (Kondratyuk et al.,
2024) and CogVideoX (Yang et al., 2024b) as examples, respec-
tively.
Models
LLM
Diffusion
LanDiff
Highlight Semantic Information?
✓
✗
✓
Progressive Refinement?
✗
✓
✓
Causal Modeling?
✓
✗
✓
High Visual Quality?
✗
✓
✓
Compression Ratio ↑
∼256
∼1024
∼14000
Long video Generation?
✗
✗
✓
in Table 2.
With this design, the ideal semantic feature should contain
high-level semantic information and motion information and
only require a few bits to represent. We achieve this goal by
performing extreme compression on video representations
rich in high-level semantics. For video representation, we
select the Theia model (Shang et al., 2024) as our visual
representation backbone, which has been distilled from mul-
tiple visual understanding and self-supervised representation
models, ensuring the encoded features contain rich semantic
information. To achieve extreme compression and reduce
the number of bits, we design an efficient tokenizer to com-
press 3D visual features into 1D discrete representations.
The tokenizer is based on the Transformer (Vaswani et al.,
2017) structure, uses query embedding to aggregate visual
features, and has a higher compression rate than CNN-based
structures (Yu et al., 2024b). To further compress the video
by fully utilizing the temporal redundancy of the video, in-
spired by the MP4 video encoding algorithm (Le Gall, 1991),
we divide the video into keyframes and non-keyframes, and
set more numbers of tokens for keyframes. The detailed de-
sign is shown in subsection 2.1. For the diffusion model, we
use generated semantic tokens as conditions and generate
the target video by gradually removing the noises. To better
support the long-video generation, we train a chunk-wise
streaming diffusion model that only uses a limited number
of historical frames as conditions, thereby greatly reducing
the computational cost of training and inference.
Thanks to these designs, our LanDiff has made significant
progress in spatial relationship compliance, action coher-
ence, visual quality, etc. Specifically, 1) for short video
generation, our 5B model achieved a score of 85.43 on
the VBench T2V benchmark, surpassing the state-of-the-
art open-source models Hunyuan Video (13B) and other
commercial models such as Sora, Kling, and Hailuo; 2)
for long video generation, after testing by the VBench
T2V benchmark, our model also achieved state-of-the-
art performance, surpassing other open-source models in
this field. Our video examples can be viewed at https:
//landiff.github.io/.
2

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
LLM-based Semantic Token Generator
Diffusion-based Perceptual Feature Generator
Text
Prompt
Text Encoder
Text Embedding
Semantic Token
A man wearing orange clothes is
preparing to squeeze orange juice.
Video
VAE
Figure 2. The architecture of LanDiff. Given text inputs, we first extract text embeddings and employ an LLM to generate semantic tokens
in the first stage. Subsequently, we utilize a diffusion model to synthesize perceptual features conditioned on these semantic tokens,
followed by a VAE decoder that transforms these features into the final video frames.
2. Method
In this work, we propose a novel text-to-video generation
framework that synergistically integrates the strengths of
autoregressive modeling and diffusion processes while cir-
cumventing their respective limitations. The framework
mainly consists of the following components: 1) an efficient
tokenizer that transforms 3D visual features into compact
1D discrete representations while preserving and enhanc-
ing their semantic information; 2) a language model that
performs temporal sequence modeling to generate semantic
tokens representing video blueprints from textual descrip-
tions; 3) a streaming diffusion model that progressively
refines coarse semantic videos by adding fine-grained de-
tails to produce high-quality VAE features; and 4) a VAE
decoder that reconstructs the final video frames from the
refined VAE features.
2.1. Video Semantic Tokenizer
In this section, we introduce a novel video semantic tok-
enizer that efficiently compresses the video into semantic in-
formation. Firstly, we will introduce the video semantic rep-
resentation used in tokenization. Then, considering the high
redundancy of video in both spatial and temporal dimen-
sions, we introduce two compression strategies: 1) a query-
based causal tokenization that efficiently reduces spatial
redundancy while preserving essential semantic information
through vector quantization; 2) inspired by MP4 (Le Gall,
1991), we implement video frame grouping to minimize
temporal redundancy by treating grouped frames as a unit,
where the first frame (I-frame) is fully encoded while subse-
quent frames (P-frames) only capture the temporal changes
by referencing content from previous frames.
Video Semantic Representation. Generally, video rep-
resentations can be divided into two categories: 1) some
methods (Wang et al., 2024a; Yu et al., 2024a) directly uti-
lize an autoencoder to learn the video representations. 2)
Some works (Koh et al., 2023; Jin et al., 2023; Sun et al.,
2024; Jin et al., 2024) use pre-trained visual self-supervised
learning features (SSL) as video representations. Compared
with the first directly learned autoencoder latents which con-
tain lots of visual details, the second SSL features maintain
much more semantic features, which enable the LLM to
focus more on the high-level semantic information of the
video.
Based on these thoughts, we choose the pretrained SSL
features as the video representations. For better usage of
different SSL models, we choose Theia model (Shang et al.,
2024), which is a unified visual feature extractor distilled
from multiple visual task models, including CLIP (Radford
et al., 2021), SAM (Kirillov et al., 2023), DINOv2 (Oquab
et al., 2024), ViT (Dosovitskiy et al., 2021), and Depth-
Anything (Yang et al., 2024a).
Tokenizer Design. In this part, we introduce the design of
our tokenizer, which leverages the query tokens to compress
the video semantic features and uses quantization to discrete
the video semantic representation while minimizing the
reconstruction loss.
In detail, firstly, we extract the semantic features using the
Theia model and flatten it to obtain F ∈R(T ×H×W )×D.
Inspired by TiTok (Yu et al., 2024b), we use N randomly ini-
tialized tokens as query tokens Q ∈RN×D and concatenate
3

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
Encoder Attention Map
 IFrame
 PFrame
Query Tokens
Decoder Attention Map
 IFrame
 PFrame
Query Tokens
KV
Raw Video Sequence
Semantic Feature Extractor
Semantic Features
...
...
Query Tokens
Encoder
...
...
Mask Tokens
....
Quantizer
0
1
2
3
k
Decoder
Figure 3. Proposed architecture of the video semantic tokenizer. We use query tokens to compress the semantic sequence length.
Furthermore, we group the frames into groups (3 frames in a group in this figure). In a group, the first frame is the IFrame and the rest
frames are PFrames. We use different query token numbers for them. The attention mask design is shown in the right.
them with the semantic features F. Then we use a trans-
former encoder to encode them and only take the encoded
features of the query tokens.
ZQ = Enc([F; Q]),
(1)
where [; ] represents the concatenation operation and ZQ ∈
RN×D. We then apply vector quantization on ZQ by train
a VQ-VAE model and obtain the quantized feature ˆZQ. In
the decoding stage, the quantized feature ˆZQ is used as
a condition, and then a sequence of mask tokens M ∈
RT ×H×W are added in front of the ˆZQ to form the inputs of
decoder. Then we only take the features of the mask tokens
as follows:
ˆF = Dec([M; ˆZQ]),
(2)
where ˆF ∈R(T ×H×W )×D represents the reconstructed
feature, and M represents the mask tokens. Inspired by
Wang et al. (2024c); Huang et al. (2023), we minimize the
reconstruction loss of the video semantic feature during
training the VQ-VAE.
For the VQ-VAE, we follow the method of Yu et al. (2022);
Wang et al. (2024c). We update the codebook using ex-
ponential moving average (EMA). We also optimize the
codebook with the video semantic feature reconstruction
loss. The loss function is shown as follows:
L = λrec|| ˆF −F||2 + λcommit||sg( ˆZQ) −ZQ||2,
(3)
where sg() is the stop-gradient operation, || · ||2 is the L2
loss.
Video Frame Grouping. For video, a straightforward ob-
servation is the redundancy in time series (i.e., the difference
between adjacent video frames is minimal). Intuitively, we
can achieve a better compression rate by modeling the differ-
ence between adjacent video frames instead of tokenizing all
frames equally. Inspired by the popular video compression
method MP4 (Le Gall, 1991), given N frames of a video, we
first split them into N/T groups (i.e., T frames as a group.
For clarification, as shown in Figure 3, we use T = 3 as an
example.). Then: 1) we will model different groups inde-
pendently; 2) for each group, we will fully encode the first
frame (Intra-coded Frame, IFrame), while for the remaining
[1, T −1] frames (the Predictive-coded Frame, PFrame), we
encode them by referencing their previous frames.
To achieve this, within a group, the first frame (IFrame) will
only see itself and have a large number of query tokens (e.g.,
3 query tokens in Figure 3) to achieve better reconstruction
quality. For the rest frame (PFrame) i ∈[1, T −1], it will
see the previous frames (i.e., frames j ∈[0, i −1]) and
have a small number of query tokens (e.g., 1 query token
which is 1/3 of the IFrame) to force the model to learn
the difference. Technically, as shown in the attention mask
on the right side of Figure 3, we use frame-level causal
masks for the feature sequence during encoding. The query
tokens are also divided according to the frame, and each
token can only attend to the features of the corresponding
frame and the previous frames. During decoding, the mask
token corresponding to each frame can see the previous
4

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
features, the corresponding query tokens, and the previous
query tokens.
2.2. Language Model for Semantic Token Generation
As shown in Figure 2, after training an efficient tokenizer,
we use a language model to generate semantic tokens au-
toregressively based on text. Specifically, we first use a
pre-trained text encoder T5-XXL (Raffel et al., 2020) to
extract text features X. We use the video tokenizer encoder
trained in subsection 2.1 to convert the video into a 1D dis-
crete token sequence Y . To enhance the controllability of
the generated results, we introduce additional control condi-
tions CC. It includes: 1) frames condition for requiring the
model to generate videos with a specified number of frames;
2) motion score condition, which is a value between 0 and 1,
used to control the degree of motion in the generated video.
The discrete tokens are converted into embedding vectors
during generation and added with positional encoding, and
then concatenated with the control conditions as input. The
structure of the language model follows the typical LLaMA
model (Touvron et al., 2023) network structure. We train
the model from scratch, using cross-entropy loss as the loss
function.
LLM = E[−log p(Yi|X, CC, Y<i)]
(4)
2.3. Diffusion Model for Perceptual Feature Generation
The video detokenizer in LanDiff is responsible for convert-
ing semantic tokens into VAE latent vectors. As shown in
Figure 2 we use a conditional diffusion model to complete
this task. Especially to support long-video generation, we
also design the streaming inference strategy. In this section,
we will first introduce the architecture for a single seman-
tic token chunk. Then we will introduce the chunk-wise
streaming strategy.
Architecture. Our diffusion model employs an architecture
similar to MMDiT(Esser et al., 2024; Yang et al., 2024b).
Specifically, 1) we use the video tokenizer decoder trained
in subsection 2.1 to decode the semantic tokens into seman-
tic features ˆF. Then we use the semantic features ˆF as a
condition to guide the diffusion model to generate videos; 2)
we inject the control signals into the model in a similar way
to ControlNet (Zhang et al., 2023b), as shown in Figure 4.
In detail, during training, the parameters of the main model
are not updated, the control module copies the parameters
of the first half layers of the main model, and adds them to
the output of the main model after a linear layer initialized
with zeros; 3) to make the semantic features match the target
VAE features in the space dimension, we additionally add
an upsampling module.
During training, given the input video chunk V
∈
Dit block_1 🔥
（trainable copy）
Dit block_2 🔥
（trainable copy）
Dit block_15 🔥
（trainable copy）
zero linear 🔥
upsample 🔥
...
Control Module
Time t
Dit block_1 ❄️
Dit block_2 ❄️
Dit block_15 ❄️
...
Backbone Module
Time t
zero linear 🔥
Dit block_16 ❄️
zero linear 🔥
Dit block_30 ❄️
...
Inputs
Semantic Features
Trainable Weights
  Frozen Weights
🔥
❄️
Figure 4. Proposed diffusion model structure.
We use a
ControlNet-style control module to guide the model to generate
perceptual feature based on semantic features.
R(T ×H×W )×D, where T is the frame length of the VAE
latent and D is the VAE latent feature dimension, the dif-
fusion algorithm progressively adds noise to the video and
produces a noisy video V t, where t represents the number of
times noise is added. The diffusion algorithm uses time step
t, and semantic features ˆF ∈R(T ×H×W )×D as conditions,
and then uses a network ϵθ to predict the noise added to the
noisy video Vt through:
L = EV,t,cs,ϵ∼N(0,1)
h
∥ϵ −ϵθ(V t, t, ˆF))∥2
2
i
(5)
Chunk-wise Streaming Strategy. To support long video
generation scenario whose semantic token sequence is very
long and difficult to generate as a whole, we propose the
chunk-wise streaming strategy. During training, given video
latent chunk V ∈R(T ×H×W )×D, to maintain the appear-
ance continuity in the video, we use the first half of V (Vl)
as the prompt and generate the second half of Vr. In detail,
we do not add noise to the first half chunk and give time
condition t to 0.999. We also randomly mask the first half
chunk with a ratio of 20%. The loss function for training at
this time is:
L = EVr,t,cs,ϵ∼N(0,1)
h
∥ϵ −ϵθ(Vl, V t
r , t, cs))∥2
2
i
.
(6)
During inference, the first L/2 VAE latents will be gener-
ated without the prompt. For the following VAE latents, we
will accumulate L/2 tokens to form a chunk and use the
previous chunk as prompt.
5

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
3. Experiments and Results
3.1. Experimental Settings
Datasets. For the video tokenizer and language model,
we use an internal dataset with 200M video-text pairs for
training. All videos with a duration of less than 6s are
filtered, and the videos are kept in the aspect ratio and then
scaled to around 480x720 resolution for center cropping.
Consistent with CogVideoX (Yang et al., 2024b), we set
the fps of all videos to 8. For the diffusion model, we
select a dataset containing 3M high-quality video-text pairs
for training. To evaluate the performance of the text-to-
video generation model, we use the prompts provided by the
widely used VBench (Huang et al., 2024) T2V benchmark
to generate videos.
Implementation Details. We use interpolation of posi-
tional encoding to enable the encoder of the Theia (Shang
et al., 2024) model to handle 480x720 resolution videos.
For the video tokenizer, we set the group size T mentioned
in subsection 2.1 to 13. We set the number of tokens I
corresponding to the IFrame to 330, and the number of to-
kens P corresponding to the PFrame to 74. On average, for
a 480x720 resolution video of one second, our tokenizer
generates about 200 tokens. In contrast, common tokenizers
such as MagViT2 (Yu et al., 2024a) generate about 10,000
tokens per second for videos of this resolution, and our se-
quence length is about 1/50 of MagViT2. The dimension
of the tokenizer’s codebook is set to 16, and the vocabulary
size is set to 2048. For the language model, we use the
LLaMA (Touvron et al., 2023) structure, with 2B model
parameters, and use 1D RoPE (Su et al., 2024) positional en-
coding. We set the text, motion score, and frames conditions
in subsection 2.2 to null with probabilities of 10%, 50%,
and 50%, respectively. We apply classifier-free guidance
(Ho & Salimans, 2022) for better generation quality, and
the guidance scale is set to 6.5. We do not use top-k and
top-p sampling. We use the 2B version of the CogVideoX1
(Yang et al., 2024b) model as the base model for our video
detokenizer. We copy the first 15 layers of the base model as
the proposed trainable control module in subsection 2.3. We
use a structure similar to the VQ-GAN (Esser et al., 2021)
decoder as the upsampling module and change the upsam-
pling method to pixelshuffle (Shi et al., 2016). The total
number of parameters of the video detokenizer is 3B, and
the number of parameters of the trainable control module is
1B. During inference, we follow the same sampling strategy
as Yang et al. (2024b). We list the structural settings and
training details of each module in the appendix.
Evaluation Metrics. To evaluate the text-to-video genera-
tion task, we use the metrics proposed in the VBench and
1https://huggingface.co/THUDM/
CogVideoX-2b
VBench-Long (Huang et al., 2024) benchmarks.
Evaluation Baselines. We compare LanDiff with baselines:
1) Sora (Sor, 2024). 2) Jimeng (Jim, 2024). 3) Hailuo (Hai,
2024). 4) OpenSoraPlan V1.1 (Lin et al., 2024a). 5) Kling
(Luo et al., 2023). 6) InstructVideo (Yuan et al., 2024). 7)
Gen-3 (Gen, 2024). 9) Latte-1 (Ma et al., 2024) 9) HiGen
(Qing et al., 2024). 10) AnimateDiff-V2 (Guo et al., 2024).
11) Show-1 (Zhang et al., 2023a). 12) Pika (Pik, 2024). 13)
VideoCrafter-2.0 (Chen et al., 2024). 14) OpenSora V1.2
(Zheng et al., 2024). 15) LTX-Video (HaCohen et al., 2024).
16) Mochi-1 (Team, 2024). 17) CogVideoX (Yang et al.,
2024b). 18) Vchitect-2.0 (Fan et al., 2025). 19) RepVideo
(Si et al., 2025). 20) HunyuanVideo (Kong et al., 2024). 21)
ARLON (Li et al., 2024). 22) Emu3 (Wang et al., 2024a).
3.2. Experimental Results
A group of silver-colored fish with darker fins swim among green aquatic plants in an
aquarium setting. The fish move gracefully through the water, navigating around the
plants, which are of various sizes and shades of green. The aquarium environment
is designed to mimic a natural habitat, with rocks and shadows in the background
contributing to the underwater scene.
A life-sized ice sculpture of a playful dog, with intricate details and a joyful
expression, stands in the middle of a sunlit, grassy field on a sweltering summer
day. The ice dog, initially solid and vibrant, begins to melt under the relentless
heat, with droplets of water forming on its surface. As the day progresses, the
ice dog's form gradually diminishes, with its once sharp features becoming
blurred and distorted. The melting process accelerates, and the ice dog's body
starts to collapse, pooling into a puddle of water on the ground. By the end of
the day, all that remains is a shallow puddle, reflecting the cloudless sky, with
the memory of the once majestic ice dog now just a memory.
CogVideoX-5B
Ours
CogVideoX-5B
Ours
Figure 5. Comparison of qualitative results for text-to-video gener-
ation.
Text to Video Generation. As shown in Table 3, we con-
ducted a quantitative comparison between LanDiff and other
state-of-the-art open-source models on the VBench bench-
mark. Our model achieves the highest semantic score and
quality score, indicating that our model can follow the text
well and generate high-quality videos. Compared with the
same-sized CogVideoX-5B, our model achieves better re-
sults in almost all dimensions. To eliminate the interference
of the training data volume, we train a 7B DiT-based diffu-
sion model with the same training data recipe as LanDiff.
We mark it as DiT in the table. This indicates that with
the same data volume, by reasonably dividing the task load,
we can combine the advantages of autoregressive models
6

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
Table 3. Performance comparison of Text-to-video (T2V) generation between our LanDiff and other state-of-the-art models on
VBench benchmark. We selected 8 out of the 16 evaluation dimensions from VBench, along with Total Score, Quality Score, and
Semantic Score, for presentation. The complete results with all 16 evaluation dimensions can be found in the appendix Table 8. The best
and second-best scores are highlighted in bold and underline, respectively. † indicates the scores we reproduced, while ‡ indicates the
scores from the original papers, and other scores are from the VBench benchmark.
Model
Name
Type
Model
Size
Total
Score
Quality
Score
Semantic
Score
background
consistency
dynamic
degree
motion
smoothness
multiple
objects
object
class scene
spatial
relationship
subject
consistency
Open Sourced Models
InstructVideo
Diffusion
1.3B
76.61
81.56
56.81
96.97
69.72
96.62
21.57
73.26 22.21
43.49
95.30
Latte-1
Diffusion
0.7B
77.29
79.72
67.58
95.40
68.89
94.63
34.53
86.53 36.26
41.53
88.88
OpenSoraPlan V1.1
Diffusion
2.7B
78.00
80.91
66.38
96.73
47.72
98.28
40.35
76.30 27.17
53.11
95.73
Show-1
Diffusion
6.3B
78.93
80.42
72.98
98.02
44.44
98.24
45.47
93.07 47.03
53.50
95.53
OpenSora V1.2
Diffusion
1.1B
79.76
81.35
73.39
97.61
42.39
98.50
51.83
82.22 42.44
68.56
96.75
LTX-Video
Diffusion
1.9B
80.00
82.30
70.79
97.20
54.35
98.96
45.43
83.45 51.07
65.43
96.56
Mochi-1
Diffusion
10B
80.13
82.64
70.08
97.28
61.85
99.02
50.47
86.51 36.99
69.24
96.99
AnimateDiff-V2
Diffusion
1.3B
80.27
82.90
69.75
97.68
40.83
97.76
36.88
90.90 50.19
34.60
95.30
VideoCrafter-2.0
Diffusion
1.7B
80.44
82.20
73.42
98.22
42.50
97.73
40.66
92.55 55.29
35.86
96.85
CogVideoX-2B
Diffusion
2B
80.91
82.18
75.83
96.63
59.86
97.73
62.63
83.37 51.14
69.90
96.78
Emu3 ‡
LLM
8B
80.96
N/A
N/A
97.69
79.27
98.93
44.64
86.17 37.11
68.73
95.32
Vchitect-2.0-2B
Diffusion
2B
81.57
82.51
77.79
96.53
58.33
97.76
69.35
87.81 57.51
54.64
96.42
CogVideoX-5B
Diffusion
5B
81.61
82.75
77.04
96.52
70.97
96.92
62.11
85.23 53.20
66.35
96.23
DiT †
Diffusion
7B
81.85
82.70
78.42
97.65
51.02
98.94
69.63
93.40 57.10
60.89
97.05
RepVideo
Diffusion
2B
81.94
82.70
78.91
96.56
57.78
98.13
71.18
87.83 52.96
74.74
96.25
Vchitect-2.0[E]
Diffusion
2B
82.24
83.54
77.06
96.66
63.89
98.98
68.84
86.61 56.57
57.55
96.83
HunyuanVideo
Diffusion
13B
83.24
85.09
75.82
97.76
70.83
98.99
68.55
86.10 53.88
68.68
97.37
Close Sourced Models
Pika-1.0
Diffusion
N/A
80.69
82.92
71.77
97.36
47.50
99.50
43.08
88.72 49.83
61.03
96.94
Kling
Diffusion
N/A
81.85
83.39
75.68
97.60
46.94
99.40
68.05
87.24 50.86
73.03
98.33
Jimeng
Diffusion
N/A
81.97
83.29
76.69
98.39
38.43
98.09
69.08
89.62 44.94
77.45
97.25
Gen-3
Diffusion
N/A
82.32
84.11
75.17
96.62
60.14
99.23
53.64
87.81 54.57
65.09
97.10
Hailuo
Diffusion
N/A
83.41
84.85
77.65
97.05
64.91
99.22
76.04
87.83 50.68
75.50
97.51
Sora
Diffusion
N/A
84.28
85.51
79.35
96.35
79.91
98.74
70.85
93.93 56.95
74.29
96.23
ARLON ‡
LLM+Diffusion
1.5B
N/A
N/A
N/A
97.10
52.77
98.92
N/A
89.80 54.43
N/A
93.41
ARLON †
LLM+Diffusion
5B
82.31
83.58
77.27
98.65
72.22
97.56
74.49
90.60 52.07
62.53
95.59
LanDiff
LLM+Diffusion
5B
85.43
86.13
82.61
98.73
92.71
97.08
86.69
94.94 53.79
73.74
96.11
and diffusion models to achieve better performance than
diffusion models alone. Figure 5 shows the qualitative
comparison between the videos generated by LanDiff and
CogVideoX-5B. In the first example, it can be seen that
in the video generated by CogVideoX-5B, one fish dis-
appears after the fish meet, while in the video generated
by our model, the fish can still remain intact after meet-
ing. This demonstrates that our model can understand the
concept of fish as an entity and maintain consistency over
time. Additionally, our model more faithfully adheres to
the background elements described in the prompt “with
rocks and shadows in the background”, rendering these en-
vironmental details with greater accuracy and consistency.
In the second example, our model accurately captures the
temporal dynamics described in the prompt, successfully
rendering a melting ice sculpture of a dog. LanDiff properly
depicts the progressive melting process, with the ice dog
gradually losing its form and eventually transforming into
a puddle of water. In contrast, CogVideoX-5B generates a
static representation of the ice dog sculpture that remains
largely unchanged throughout the sequence, failing to cap-
ture the crucial temporal narrative of melting described in
the prompt. This highlights our model’s superior capability
in understanding and visualizing complex temporal transfor-
mations and physical processes. For more video examples,
please refer to our demo website2.
Long Video Generation. We compare our model with
other open-source text-to-long video generation models: 1)
FreeNoise (Qiu et al., 2024). 2) StreamingT2V (Henschel
et al., 2024). 3) OpenSora-V1.2 (Zheng et al., 2024). 4)
ARLON (Li et al., 2024). As shown in Table 4, our LanDiff
achieves state-of-the-art performance in long video genera-
tion tasks with the highest Total Score of 68.34, outperform-
ing other open-source models across almost all dimensions.
While StreamingT2V exhibits the highest Dynamic Degree
(85.64), this comes at a clear cost to consistency metrics,
as evidenced by its significantly lower Subject Consistency
(87.31) and Background Consistency (94.64). An optimal
video generation model should maximize dynamic content
while maintaining temporal coherence. Compared with
models that excel in consistency metrics but show limited dy-
namism (such as ARLON with 50.42 Dynamic Degree), our
approach demonstrates superior dynamism (72.86) while
preserving strong consistency scores and achieving the best
results in Aesthetic Quality (60.96) and Overall Consistency
(27.29). As illustrated in Figure 6, LanDiff effectively gen-
erates complex motion dynamics when prompted with “a
car accelerating to gain speed,” successfully depicting the
progressive increase in vehicle velocity while preserving
2https://landiff.github.io/
7

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
Table 4. Long video generation results of LanDiff and other models on VBench. The best and second-best scores are highlighted in bold
and underline, respectively.
Models
Total
Score
Subject
Consist
Background
Consist
Motion
Smooth
Dynamic
Degree
Aesthetic
Quality
Imaging
Quality
Overall
Consist
FreeNoise
62.60
96.59
97.48
98.36
17.44
47.39
63.88
25.78
StreamingT2V
62.92
87.31
94.64
93.83
85.64
44.57
53.64
23.65
OpenSora-V1.2
64.24
96.30
97.39
98.94
44.79
56.68
51.64
26.36
ARLON
65.09
97.11
97.56
98.50
50.42
56.85
53.85
26.55
LanDiff
68.34
95.41
97.88
97.38
72.86
60.96
63.00
27.29
Streaming t2v
Freenoise
OpenSora
Ours
"a car accelerating to gain speed."
@1s
@25s
@7s
@13s
@19s
Figure 6. Qualitative comparison of long video generation results between LanDiff and other state-of-the-art models (FreeNoise, Stream-
ingT2V and OpenSora-V1.2).
visual consistency across frames. In contrast, FreeNoise,
StreamingT2V, and OpenSora-V1.2 all struggle to properly
render the acceleration motion, exhibiting either static ve-
hicles, inconsistent car appearances, or unrealistic motion
patterns that fail to convey the sense of increasing speed.
Additional video samples are available on our demo site2.
Reference 
 Video
Ours
MagViT2 
Figure 7. Visualization results of video reconstruction using video
tokenizer.
Video Tokenization. We present visualization results of our
model’s video reconstruction in Figure 7. We extract seman-
tic tokens from the reference video and subsequently convert
these tokens back into video using the video detokenizer.
The results demonstrate that through our careful design, our
video tokenizer can accurately reconstruct videos with accu-
rate semantics and actions using only approximately 1/50th
of the sequence length required by MagViT2 (Yu et al.,
2024a). While some minor discrepancies in clothing details
exist between the reconstructed and reference videos, these
differences remain within acceptable limits for practical
video generation applications. Additional video samples are
available on our demo site2. Figure 8 shows the reconstruc-
tion loss trajectories for IFrame and PFrame components
during training. Despite allocating significantly different
token quantities (330 vs. 74), both frame types converge to
comparable reconstruction quality. This validates our video
frame grouping strategy that prioritizes key frames while
minimizing tokens for intermediate frames. The results
confirm our approach successfully balances reconstruction
fidelity with computational efficiency.
8

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
0
20000
40000
60000
80000
100000 120000
Steps
0.3
0.4
0.5
0.6
0.7
0.8
Loss Value
Tokenizer Training Reconstruction Loss
losses/Iframe_loss (Raw)
losses/Iframe_loss (Smoothed)
losses/Pframe_loss (Raw)
losses/Pframe_loss (Smoothed)
Figure 8. Training loss comparison of the video tokenizer. The
plot illustrates the reconstruction loss trajectories for IFrame and
PFrame components over training iterations. Despite the different
token allocation strategies (330 tokens for IFrame vs. 74 tokens
for PFrame), both frame types achieve comparable reconstruction
quality.
Table 5. The ablation study of video tokenizer and classifier free
guidance (cfg) on Vbench benchmark.
Models
Total
Score
Quality
Score
Semantic
Score
LanDiff
85.43
86.13
82.61
-video tokenizer
82.31
83.58
77.27
-cfg
81.06
83.04
73.13
3.3. Ablation Study
In this section, we conducted ablation experiments on the
video tokenizer and classifier-free guidance. To fairly evalu-
ate the effectiveness of our video tokenizer, we implemented
an ARLON-like method as a comparison baseline. Specif-
ically, for the control group without our proposed video
tokenizer, we followed ARLON’s architectural approach by
training a large language model to predict quantized VAE
features, then using CogVideoX-5B as the base diffusion
model conditioned on these LLM-predicted VAE features
to generate videos. This baseline model was trained with
exactly the same dataset and training recipe as LanDiff, en-
suring strict experimental control variables and effectively
eliminating interference factors such as model parameter
count and training data scale. As shown in Table 5, the
experimental results clearly demonstrate that our proposed
video tokenizer significantly improves both the quality score
and semantic score of generated videos, validating the im-
portance of this component in our framework. In the case
of not using classifier-free guidance for text, our model has
decreased in quality and semantic scores.
4. Related Work
Video Tokenization. Video tokenization plays a crucial
role in video understanding and generation tasks. Since a
video can be represented as a sequence of continuous frames,
some works directly use image tokenizers to process videos
frame by frame. For example, Wang et al. (2024a) directly
use SBER-MoVQGAN as the video tokenizer. However,
this method ignores the temporal redundancy in videos,
resulting in a low compression rate. To reduce temporal
redundancy, some works (Yan et al., 2021; Yu et al., 2024a)
try to extend image tokenizers based on 2D convolution
to 3D convolution, which can process both temporal and
spatial information simultaneously. These methods encode
videos in the original RGB space and perform video recon-
struction tasks, which are more about perceptual compres-
sion. In addition, some works (Ge et al., 2023; Jin et al.,
2023) try to train video tokenizers on features extracted from
pre-trained visual encoders. These tokenizers can achieve
good performance in understanding and generation tasks
while maintaining a high compression rate. The video tok-
enizer we propose belongs to this category. Unlike previous
feature-based tokenizers that achieved limited compression
rates and primarily focused on image processing, our video
tokenizer delivers significantly higher compression while
handling both images and videos in a unified framework.
LLM based Video Generation. LLM-based video genera-
tion methods are usually based on the Transformer (Vaswani
et al., 2017) structure, learning the mapping from text to
video through next token prediction. TATS (Ge et al., 2022)
uses VQ-GAN as the tokenizer and predicts video tokens
using a GPT-like model structure. VideoGPT (Yan et al.,
2021) uses 3D convolution to extract features and quantize
them, and predicts the quantized video discrete tokens using
a GPT-like model. Recently, VideoPoet (Yu et al., 2024a)
uses MagViT2 (Yu et al., 2024a) as the tokenizer and uni-
fies multiple modalities as input to a large language model
(LLM) to conditionally generate video tokens. In addition,
Emu3 (Wang et al., 2024a) uses SBER-MoVQGAN as the
tokenizer to perform video understanding and generation
by predicting the next token. These works all use LLMs
to directly generate “perceptual features” that contain rich
visual details with high bit rates. Recently, ARLON (Li
et al., 2024) attempts to discretize VAE features into a small
number of tokens to reduce the bit rate required for LLMs’
prediction. In this way, the tokens retain low-frequency
visual information such as blurry contours rather than high-
level semantic information. In contrast, our method employs
tokens containing high-level semantic information as pre-
diction targets for LLMs, which enables us to fully leverage
LLMs’ advantages in causal modeling to precisely generate
high-quality videos.
Diffusion based Video Generation. Diffusion-based meth-
ods have achieved great success in image generation, and
recently many people (Chen et al., 2024; Ho et al., 2022a;b;
Singer et al., 2023; Zhou et al., 2023) have tried to apply
them to video generation tasks. VDM (Ho et al., 2022b)
9

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
extends the 3D U-Net structure for video generation. Wang
et al. (2024b) propose to generate high-quality and aestheti-
cally pleasing videos in a cascaded manner. Benefiting from
the success of the text-to-image (T2I) field, some works
such as Animatediff (Guo et al., 2024), SVD (Blattmann
et al., 2023), and PixelDance (Zeng et al., 2023) try to use
pre-trained T2I models as initialization, and then add mod-
ules for temporal modeling to capture motion information
for video generation. Ma et al. (2024) explore the gener-
ation capabilities of multiple different structures of latent
diffusion transformer. After the release of SORA, a series
of video generation methods based on the DiT (Peebles
& Xie, 2023) model have been proposed, including Open-
Sora (Zheng et al., 2024), OpenSoraPlan (Lin et al., 2024a),
Cogvideox (Yang et al., 2024b), Hunyuan Video (Kong
et al., 2024), Mira (Ju et al., 2024) and STIV (Lin et al.,
2024b) etc. These methods can only generate short videos
of a few seconds. Recently, StreamingT2V (Henschel et al.,
2024) generates long videos by block-wise generation on
a pre-trained short video generation model, and then uni-
formly performs mixed augmentation. In addition, some
works improve the consistency of long video generation by
leveraging noise rescheduling techniques (Qiu et al., 2024;
Lu et al., 2024). Our method employs diffusion models as
renderers for semantic features, enabling us to leverage their
superior visual generation quality while circumventing their
limitations in causal modeling.
5. Conclusion
In this paper, we propose a new text-to-video generation
model, LanDiff. It combines the advantages of autoregres-
sive models and diffusion models, including: 1) an efficient
1D video tokenizer to extract videos into semantic tokens;
2) a language model to generate semantic tokens based
on text; 3) a video detokenizer to convert semantic tokens
into videos. LanDiff outperforms the state-of-the-art open-
source models on the VBench benchmark, surpassing other
open-source models in quality and semantic scores. At the
same time, LanDiff also achieves state-of-the-art perfor-
mance in long video generation tasks.
6. Broader Impact
Text-to-video generation models such as LanDiff offer sub-
stantial potential for creative applications across entertain-
ment, education, and content creation domains. Neverthe-
less, these technologies introduce ethical considerations and
potential risks that warrant attention. The capability to gen-
erate photorealistic videos from textual descriptions could
potentially be exploited to create misleading or deceptive
content, including sophisticated deepfakes or videos that
misrepresent individuals or events. Such misuse raises sig-
nificant concerns regarding misinformation propagation, pri-
vacy violations, and potential harm to individuals or commu-
nities. To mitigate these risks, we recommend several safe-
guards: (1) incorporating visible watermarks or robust dig-
ital signatures in generated content to ensure transparency
regarding its synthetic nature; (2) advancing and deploy-
ing sophisticated detection systems capable of identifying
AI-generated content with high accuracy; (3) establishing
comprehensive usage policies that explicitly prohibit harm-
ful applications; and (4) implementing accessible reporting
mechanisms for suspected misuse cases. Furthermore, we
advocate for continued research into technical safeguards
and responsible deployment frameworks specifically de-
signed for generative video models. We emphasize that our
research contribution aims to advance the field of multi-
modal generation for socially beneficial applications while
acknowledging the necessity of addressing potential nega-
tive impacts through complementary technical innovations
and policy measures.
7. Limitation and Future Works
While LanDiff demonstrates significant advancements in
text-to-video generation, several limitations remain to be
addressed in future work. First, the scale of our language
model (2B parameters) is substantially smaller than state-
of-the-art text-only LLMs, potentially limiting the semantic
understanding and generation capabilities. Future work
will explore scaling our language models to larger param-
eter counts to enhance performance. Second, our choice
of CogVideoX-2B as the underlying diffusion model estab-
lishes an upper bound on the quality of generated videos.
We plan to investigate the development of more sophisti-
cated diffusion backbones specifically optimized for video
generation tasks to overcome this constraint. Third, we
observe that LanDiff struggles with accurate text render-
ing within generated videos. This limitation likely stems
from insufficient supervision of text features in the current
semantic token representation. Future research will focus
on developing more comprehensive video semantic tokens
with enhanced text-specific supervision. Our current work
primarily addresses text-to-video generation, but we envi-
sion extending LanDiff’s capabilities to broader application
scenarios. These include image-to-video generation, uni-
fied models for video understanding and generation, and
interactive controllable video synthesis. These extensions
would significantly expand the utility of semantic token-
based language models in multimodal generation tasks and
provide more flexible creative tools for users across various
domains.
10

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
References
Gen-3.
https://runwayml.com/research/
introducing-gen-3-alpha, 2024.
Hailuo.
https://platform.minimaxi.com/,
2024.
Jimeng. https://jimeng.jianying.com/, 2024.
Pika 1.0. https://www.pika.art/, 2024.
Openai sora. https://openai.com/index/sora,
2024.
Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D.,
Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V.,
Letts, A., Jampani, V., and Rombach, R. Stable Video
Diffusion: Scaling Latent Video Diffusion Models to
Large Datasets, November 2023.
Chen, H., Zhang, Y., Cun, X., Xia, M., Wang, X., Weng, C.,
and Shan, Y. VideoCrafter2: Overcoming data limitations
for high-quality video diffusion models. In IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp.
7310–7320. IEEE, 2024. doi: 10.1109/CVPR52733.2024.
00698.
D´efossez, A., Copet, J., Synnaeve, G., and Adi, Y. High
Fidelity Neural Audio Compression, October 2022.
Dong, J., Feng, B., Guessous, D., Liang, Y., and He, H.
Flex Attention: A Programming Model for Generating
Optimized Attention Kernels, December 2024.
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn,
D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer,
M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N.
An image is worth 16x16 words: Transformers for image
recognition at scale. In 9th International Conference
on Learning Representations, ICLR 2021, Virtual Event,
Austria, May 3-7, 2021. OpenReview.net, 2021.
Esser, P., Rombach, R., and Ommer, B. Taming Transform-
ers for High-Resolution Image Synthesis. In Proceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 12873–12883, 2021.
Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller,
J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F.,
Podell, D., Dockhorn, T., English, Z., and Rombach, R.
Scaling rectified flow transformers for high-resolution
image synthesis. In Forty-first International Conference
on Machine Learning, ICML 2024, Vienna, Austria, July
21-27, 2024. OpenReview.net, 2024. URL https://
openreview.net/forum?id=FPnUhsQJ5B.
Fan, W., Si, C., Song, J., Yang, Z., He, Y., Zhuo, L., Huang,
Z., Dong, Z., He, J., Pan, D., Wang, Y., Jiang, Y., Wang,
Y., Gao, P., Chen, X., Li, H., Lin, D., Qiao, Y., and Liu, Z.
Vchitect-2.0: Parallel Transformer for Scaling Up Video
Diffusion Models, January 2025.
Ge, S., Hayes, T., Yang, H., Yin, X., Pang, G., Jacobs, D.,
Huang, J.-B., and Parikh, D. Long Video Generation with
Time-Agnostic VQGAN and Time-Sensitive Transformer,
September 2022.
Ge, Y., Ge, Y., Zeng, Z., Wang, X., and Shan, Y. Planting a
SEED of Vision in Large Language Model, August 2023.
Guo, Y., Yang, C., Rao, A., Liang, Z., Wang, Y., Qiao, Y.,
Agrawala, M., Lin, D., and Dai, B. AnimateDiff: Animate
your personalized text-to-image diffusion models without
specific tuning. In The Twelfth International Conference
on Learning Representations, ICLR 2024, Vienna, Austria,
May 7-11, 2024. OpenReview.net, 2024.
HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D.,
Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari,
N., Gordon, O., Panet, P., Weissbuch, S., Kulikov, V.,
Bitterman, Y., Melumian, Z., and Bibi, O. LTX-Video:
Realtime Video Latent Diffusion, December 2024.
Henschel, R., Khachatryan, L., Hayrapetyan, D., Poghosyan,
H., Tadevosyan, V., Wang, Z., Navasardyan, S., and Shi,
H. StreamingT2V: Consistent, Dynamic, and Extendable
Long Video Generation from Text, March 2024.
Ho, J. and Salimans, T. Classifier-Free Diffusion Guidance,
July 2022.
Ho, J., Jain, A., and Abbeel, P.
Denoising diffusion
probabilistic models. In Larochelle, H., Ranzato, M.,
Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances
in Neural Information Processing Systems 33:
An-
nual Conference on Neural Information Processing
Systems 2020, NeurIPS 2020, December 6-12, 2020,
virtual,
2020.
URL https://proceedings.
neurips.cc/paper/2020/hash/
4c5bcfec8584af0d967f1ab10179ca4b-Abstract.
html.
Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko,
A. A., Kingma, D. P., Poole, B., Norouzi, M., Fleet, D. J.,
and Salimans, T. Imagen video: High definition video
generation with diffusion models. CoRR, abs/2210.02303,
2022a. doi: 10.48550/ARXIV.2210.02303.
Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M.,
and Fleet, D. J. Video Diffusion Models, June 2022b.
Huang, Z., Meng, C., and Ko, T. Repcodec: A speech repre-
sentation codec for speech tokenization. arXiv preprint
arXiv:2309.00169, 2023.
11

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang,
Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen,
X., Wang, L., Lin, D., Qiao, Y., and Liu, Z. VBench:
Comprehensive benchmark suite for video generative
models. In IEEE/CVF Conference on Computer Vision
and Pattern Recognition, CVPR 2024, Seattle, WA, USA,
June 16-22, 2024, pp. 21807–21818. IEEE, 2024. doi:
10.1109/CVPR52733.2024.02060.
Jin, Y., Xu, K., Xu, K., Chen, L., Liao, C., Tan, J., Huang,
Q., Chen, B., Lei, C., Liu, A., Song, C., Lei, X., Zhang,
D., Ou, W., Gai, K., and Mu, Y. Unified Language-
Vision Pretraining in LLM with Dynamic Discrete Visual
Tokenization, September 2023.
Jin, Y., Sun, Z., Xu, K., Xu, K., Chen, L., Jiang, H., Huang,
Q., Song, C., Liu, Y., Zhang, D., Song, Y., Gai, K.,
and Mu, Y. Video-LaVIT: Unified Video-Language Pre-
training with Decoupled Visual-Motional Tokenization,
February 2024.
Ju, X., Gao, Y., Zhang, Z., Yuan, Z., Wang, X., Zeng, A.,
Xiong, Y., Xu, Q., and Shan, Y. MiraData: A Large-
Scale Video Dataset with Long Durations and Structured
Captions, July 2024.
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C.,
Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo,
W.-Y., Doll´ar, P., and Girshick, R. Segment Anything,
April 2023.
Koh, J. Y., Fried, D., and Salakhutdinov, R. Generating
images with multimodal language models. In Oh, A.,
Naumann, T., Globerson, A., Saenko, K., Hardt, M., and
Levine, S. (eds.), Advances in Neural Information Pro-
cessing Systems 36: Annual Conference on Neural In-
formation Processing Systems 2023, NeurIPS 2023, New
Orleans, LA, USA, December 10 - 16, 2023, 2023.
Kondratyuk, D., Yu, L., Gu, X., Lezama, J., Huang, J.,
Schindler, G., Hornung, R., Birodkar, V., Yan, J., Chiu,
M.-C., Somandepalli, K., Akbari, H., Alon, Y., Cheng, Y.,
Dillon, J. V., Gupta, A., Hahn, M., Hauth, A., Hendon,
D., Martinez, A., Minnen, D., Sirotenko, M., Sohn, K.,
Yang, X., Adam, H., Yang, M.-H., Essa, I., Wang, H.,
Ross, D. A., Seybold, B., and Jiang, L. VideoPoet: A
large language model for zero-shot video generation. In
Forty-First International Conference on Machine Learn-
ing, ICML 2024, Vienna, Austria, July 21-27, 2024. Open-
Review.net, 2024.
Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J.,
Xiong, J., Li, X., Wu, B., Zhang, J., Wu, K., Lin, Q.,
Yuan, J., Long, Y., Wang, A., Wang, A., Li, C., Huang,
D., Yang, F., Tan, H., Wang, H., Song, J., Bai, J., Wu,
J., Xue, J., Wang, J., Wang, K., Liu, M., Li, P., Li, S.,
Wang, W., Yu, W., Deng, X., Li, Y., Chen, Y., Cui, Y.,
Peng, Y., Yu, Z., He, Z., Xu, Z., Zhou, Z., Xu, Z., Tao,
Y., Lu, Q., Liu, S., Zhou, D., Wang, H., Yang, Y., Wang,
D., Liu, Y., Jiang, J., and Zhong, C. HunyuanVideo:
A Systematic Framework For Large Video Generative
Models, December 2024.
Le Gall, D. MPEG: A video compression standard for
multimedia applications. Communications of the ACM,
34(4):46–58, 1991.
Li, Z., Hu, S., Liu, S., Zhou, L., Choi, J., Meng, L., Guo, X.,
Li, J., Ling, H., and Wei, F. ARLON: Boosting Diffusion
Transformers with Autoregressive Models for Long Video
Generation, October 2024.
Lin, B., Ge, Y., Cheng, X., Li, Z., Zhu, B., Wang, S., He,
X., Ye, Y., Yuan, S., Chen, L., Jia, T., Zhang, J., Tang,
Z., Pang, Y., She, B., Yan, C., Hu, Z., Dong, X., Chen,
L., Pan, Z., Zhou, X., Dong, S., Tian, Y., and Yuan, L.
Open-Sora Plan: Open-Source Large Video Generation
Model, November 2024a.
Lin, Z., Liu, W., Chen, C., Lu, J., Hu, W., Fu, T.-J., Al-
lardice, J., Lai, Z., Song, L., Zhang, B., Chen, C., Fei,
Y., Jiang, Y., Li, L., Sun, Y., Chang, K.-W., and Yang,
Y. STIV: Scalable Text and Image Conditioned Video
Generation, December 2024b.
Loshchilov, I. Decoupled weight decay regularization. arXiv
preprint arXiv:1711.05101, 2017.
Lu, Y., Liang, Y., Zhu, L., and Yang, Y. FreeLong: Training-
Free Long Video Generation with SpectralBlend Tempo-
ral Attention, July 2024.
Luo, Z., Chen, D., Zhang, Y., Huang, Y., Wang, L., Shen, Y.,
Zhao, D., Zhou, J., and Tan, T. Videofusion: Decomposed
diffusion models for high-quality video generation. In
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), June 2023.
Ma, X., Wang, Y., Jia, G., Chen, X., Liu, Z., Li, Y.-F., Chen,
C., and Qiao, Y. Latte: Latent Diffusion Transformer for
Video Generation, January 2024.
Oquab, M., Darcet, T., Moutakanni, T., Vo, H. V.,
Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D.,
Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba,
W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat,
M., Sharma, V., Synnaeve, G., Xu, H., J´egou, H., Mairal,
J., Labatut, P., Joulin, A., and Bojanowski, P. DINOv2:
Learning robust visual features without supervision. 2024,
2024.
Peebles, W. and Xie, S. Scalable Diffusion Models with
Transformers, March 2023.
12

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
Qing, Z., Zhang, S., Wang, J., Wang, X., Wei, Y., Zhang,
Y., Gao, C., and Sang, N. Hierarchical spatio-temporal
decoupling for text-to-video generation. In CVPR, 2024.
Qiu, H., Xia, M., Zhang, Y., He, Y., Wang, X., Shan, Y., and
Liu, Z. FreeNoise: Tuning-free longer video diffusion
via noise rescheduling. In The Twelfth International Con-
ference on Learning Representations, ICLR 2024, Vienna,
Austria, May 7-11, 2024. OpenReview.net, 2024.
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G.,
Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark,
J., Krueger, G., and Sutskever, I. Learning Transferable
Visual Models From Natural Language Supervision. In
Proceedings of the 38th International Conference on Ma-
chine Learning, pp. 8748–8763. PMLR, July 2021.
Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S.,
Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring
the limits of transfer learning with a unified text-to-text
transformer. Journal of Machine Learning Research, 21:
140:1–140:67, 2020.
Shang, J., Schmeckpeper, K., May, B. B., Minniti, M. V., Ke-
lestemur, T., Watkins, D., and Herlant, L. Theia: Distill-
ing Diverse Vision Foundation Models for Robot Learn-
ing, October 2024.
Shi, W., Caballero, J., Huszar, F., Totz, J., Aitken, A. P.,
Bishop, R., Rueckert, D., and Wang, Z. Real-time sin-
gle image and video super-resolution using an efficient
sub-pixel convolutional neural network. In 2016 IEEE
Conference on Computer Vision and Pattern Recognition,
CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016,
pp. 1874–1883. IEEE Computer Society, 2016.
doi:
10.1109/CVPR.2016.207.
Si, C., Fan, W., Lv, Z., Huang, Z., Qiao, Y., and Liu, Z.
RepVideo: Rethinking Cross-Layer Representation for
Video Generation, January 2025.
Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang,
S., Hu, Q., Yang, H., Ashual, O., Gafni, O., Parikh, D.,
Gupta, S., and Taigman, Y. Make-a-video: Text-to-video
generation without text-video data. In The Eleventh Inter-
national Conference on Learning Representations, ICLR
2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net,
2023.
Su, J., Ahmed, M. H. M., Lu, Y., Pan, S., Bo, W., and Liu,
Y. RoFormer: Enhanced transformer with rotary position
embedding. Neurocomputing, 568:127063, 2024. doi:
10.1016/J.NEUCOM.2023.127063.
Sun, Q., Cui, Y., Zhang, X., Zhang, F., Yu, Q., Wang, Y.,
Rao, Y., Liu, J., Huang, T., and Wang, X. Generative
multimodal models are in-context learners. In IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp.
14398–14409. IEEE, 2024. doi: 10.1109/CVPR52733.
2024.01365.
Team,
G.
Mochi 1.
https://github.com/
genmoai/models, 2024.
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,
M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E.,
Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lam-
ple, G. LLaMA: Open and Efficient Foundation Language
Models, February 2023.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention
is All you Need. In Advances in Neural Information
Processing Systems, volume 30. Curran Associates, Inc.,
2017.
Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J.,
Zhang, F., Wang, Y., Li, Z., Yu, Q., Zhao, Y., Ao, Y.,
Min, X., Li, T., Wu, B., Zhao, B., Zhang, B., Wang, L.,
Liu, G., He, Z., Yang, X., Liu, J., Lin, Y., Huang, T., and
Wang, Z. Emu3: Next-Token Prediction is All You Need,
September 2024a.
Wang, Y., Chen, X., Ma, X., Zhou, S., Huang, Z., Wang,
Y., Yang, C., He, Y., Yu, J., Yang, P., et al. Lavie: High-
quality video generation with cascaded latent diffusion
models. International Journal of Computer Vision, pp.
1–20, 2024b.
Wang, Y., Zhan, H., Liu, L., Zeng, R., Guo, H., Zheng, J.,
Zhang, Q., Zhang, X., Zhang, S., and Wu, Z. Maskgct:
Zero-shot text-to-speech with masked generative codec
transformer. arXiv preprint arXiv:2409.00750, 2024c.
Yan, W., Zhang, Y., Abbeel, P., and Srinivas, A. VideoGPT:
Video Generation using VQ-VAE and Transformers,
September 2021.
Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., and Zhao,
H. Depth anything: Unleashing the power of large-scale
unlabeled data. In IEEE/CVF Conference on Computer
Vision and Pattern Recognition, CVPR 2024, Seattle, WA,
USA, June 16-22, 2024, pp. 10371–10381. IEEE, 2024a.
doi: 10.1109/CVPR52733.2024.00987.
Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J.,
Yang, Y., Hong, W., Zhang, X., Feng, G., Yin, D., Gu, X.,
Zhang, Y., Wang, W., Cheng, Y., Liu, T., Xu, B., Dong,
Y., and Tang, J. CogVideoX: Text-to-Video Diffusion
Models with An Expert Transformer, August 2024b.
Yu, J., Li, X., Koh, J. Y., Zhang, H., Pang, R., Qin, J., Ku, A.,
Xu, Y., Baldridge, J., and Wu, Y. Vector-quantized image
13

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
modeling with improved VQGAN. In The Tenth Inter-
national Conference on Learning Representations, ICLR
2022, Virtual Event, April 25-29, 2022. OpenReview.net,
2022.
Yu, L., Lezama, J., Gundavarapu, N. B., Versari, L., Sohn,
K., Minnen, D., Cheng, Y., Gupta, A., Gu, X., Haupt-
mann, A. G., Gong, B., Yang, M.-H., Essa, I., Ross,
D. A., and Jiang, L. Language model beats diffusion -
tokenizer is key to visual generation. In The Twelfth Inter-
national Conference on Learning Representations, ICLR
2024, Vienna, Austria, May 7-11, 2024. OpenReview.net,
2024a.
Yu, Q., Weber, M., Deng, X., Shen, X., Cremers, D., and
Chen, L.-C. An Image is Worth 32 Tokens for Recon-
struction and Generation, June 2024b.
Yuan, H., Zhang, S., Wang, X., Wei, Y., Feng, T., Pan,
Y., Zhang, Y., Liu, Z., Albanie, S., and Ni, D.
In-
structVideo: Instructing video diffusion models with hu-
man feedback. In IEEE/CVF Conference on Computer
Vision and Pattern Recognition, CVPR 2024, Seattle, WA,
USA, June 16-22, 2024, pp. 6463–6474. IEEE, 2024. doi:
10.1109/CVPR52733.2024.00618.
Zeng, Y., Wei, G., Zheng, J., Zou, J., Wei, Y., Zhang, Y.,
and Li, H. Make Pixels Dance: High-Dynamic Video
Generation, November 2023.
Zhang, D. J., Wu, J. Z., Liu, J.-W., Zhao, R., Ran, L., Gu, Y.,
Gao, D., and Shou, M. Z. Show-1: Marrying Pixel and
Latent Diffusion Models for Text-to-Video Generation,
October 2023a.
Zhang, L., Rao, A., and Agrawala, M. Adding conditional
control to text-to-image diffusion models. In IEEE/CVF
International Conference on Computer Vision, ICCV
2023, Paris, France, October 1-6, 2023, pp. 3813–3824.
IEEE, 2023b. doi: 10.1109/ICCV51070.2023.00355.
Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou,
Y., Li, T., and You, Y. Open-sora: Democratizing efficient
video production for all, March 2024. URL https:
//github.com/hpcaitech/Open-Sora.
Zhou, D., Wang, W., Yan, H., Lv, W., Zhu, Y., and Feng,
J. MagicVideo: Efficient Video Generation With Latent
Diffusion Models, May 2023.
14

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
A. Implementation Details
Table 6. The detailed model configurations of Video Tokenizer.
Module
Configuration
Value
#Parameters
Encoder
Transformer Layer
12
85M
Hidden Size
768
Attention Heads
12
IFrame Query Tokens
330
PFrame Query Tokens
74
Decoder
Transformer Layer
12
85M
Hidden Size
768
Attention Heads
12
Mask Tokens
1
Quantizer
Codebook size
2048
25.4K
Codebook Dimension
16
Similarity Metric
Cosine
Total Parameters
170M
Table 7. The detailed model configurations of LanDiff.
Module
Configuration
Value
#Parameters
LLM
Transformer Layer
24
2B
Hidden Size
2048
Attention Heads
16
MLP Dimension
11008
Activation
GELU
RoPE θ
10000
Text Drop Rate
0.1
Micro Conditioner Hidden Size
512
Diffusion Backbone Module (Frozen)
Transformer Layer
30
2B
Attention Heads
8
Hidden Size
1920
Time Embedding Size
256
Diffusion Control Module
Transformer Layer
15
1B
Attention Heads
8
Hidden Size
1920
Time Embedding Size
256
Trainable Parameters
3B
Total Parameters
5B
A.1. Details of Video Tokenizer
The video tokenizer model follows a similar structure to TiTok (Yu et al., 2024b). However, to flexibly encode videos
with different numbers of frames, we replace absolute position encoding with 3D RoPE position encoding (Su et al.,
2024). The model configuration of the tokenizer and the parameters of each part are shown in Table 6. The model is a
Transformer structure, with 12 layers in both the encoder and decoder, a hidden layer size of 768, and 12 heads. To improve
the computational efficiency of the attention mechanism in the tokenizer, we employ flex attention(Dong et al., 2024). In
addition, inspired by EnCodec (D´efossez et al., 2022), to avoid discontinuities when encoding videos, we set a 20% overlap
between groups. During training, the batch size is 96, we use the AdamW (Loshchilov, 2017) optimizer, the learning rate is
constant at 1e −4, and the learning rate decay factor is 0. During training, we use Model Exponential Moving Average
(EMA) to smooth the model parameters, and the decay rate of EMA is 0.8. The weights of the reconstruction loss and the
15

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
commitment loss are both 1.
A.2. Details of LLM
We use a model structure similar to LLaMA (Touvron et al., 2023) as the LLM. The model has 24 layers, a hidden layer size
of 2048, 16 heads, and an MLP hidden layer size of 11008. The batch size for model training is 4096, we use the AdamW
optimizer, the learning rate is 1e −3, the learning rate decay factor is 0.1, and we use a warm-up strategy for the first 1000
steps of training. We use a cosine learning rate decay strategy.
A.3. Details of Diffusion Model
The batch size for training is 128, we use the AdamW optimizer, the learning rate is 1e −4, and the learning rate decay
factor is 1e −4. To speed up training, we first train directly on the original features extracted from Theia. Then we use the
quantized reconstructed features for training.
Table 8. Performance comparison of Text-to-video (T2V) generation between our LanDiff and other state-of-the-art models on
VBench benchmark. The remaining 8 evaluation dimensions of VBench that are not shown in the main text. The best and second-best
scores are highlighted in bold and underline, respectively. † indicates the scores we reproduced, while ‡ indicates the scores from the
original papers, and other scores are from the VBench benchmark.
Model
Name
Type
Model
Size
Total
Score
Quality
Score
Semantic
Score
aesthetic
quality
appearance
style
color human
action
imaging
quality
overall
consistency
temporal
flickering
temporal
style
Open Sourced Models
InstructVideo
Diffusion
1.3B
76.61
81.56
56.81
52.55
20.16
77.14
85.20
68.01
19.91
98.19
21.26
Latte-1
Diffusion
0.7B
77.29
79.72
67.58
61.59
23.74
85.31
90.00
61.92
27.33
98.89
24.76
OpenSoraPlan V1.1
Diffusion
2.7B
78.00
80.91
66.38
56.85
22.90
89.19
86.80
62.28
26.52
99.03
23.87
Show-1
Diffusion
6.3B
78.93
80.42
72.98
57.35
23.06
86.35
95.60
58.66
27.46
99.12
25.28
OpenSora V1.2
Diffusion
1.1B
79.76
81.35
73.39
56.85
23.95
90.08
91.20
63.34
26.85
99.53
24.54
LTX-Video
Diffusion
1.9B
80.00
82.30
70.79
59.81
21.47
81.45
92.80
60.28
25.19
99.34
22.62
Mochi-1
Diffusion
10B
80.13
82.64
70.08
56.94
20.33
79.73
94.60
60.64
25.15
99.40
23.65
AnimateDiff-V2
Diffusion
1.3B
80.27
82.90
69.75
67.16
22.42
87.47
92.60
70.10
27.04
98.75
26.03
VideoCrafter-2.0
Diffusion
1.7B
80.44
82.20
73.42
63.13
25.13
92.92
95.00
67.22
28.23
98.41
25.84
CogVideoX-2B
Diffusion
2B
80.91
82.18
75.83
60.82
24.80
79.41
98.00
61.68
26.66
98.89
24.36
Emu3 ‡
LLM
8B
80.96
N/A
N/A
59.64
20.92
N/A
77.71
N/A
N/A
N/A
N/A
Vchitect-2.0-2B
Diffusion
2B
81.57
82.51
77.79
61.47
24.93
86.87
97.00
65.60
28.01
98.45
25.56
CogVideoX-5B
Diffusion
5B
81.61
82.75
77.04
61.98
24.91
82.81
99.40
62.90
27.59
98.66
25.38
DiT †
Diffusion
7B
81.85
82.70
78.42
60.00
24.95
78.62
98.20
63.80
27.85
99.13
26.10
RepVideo
Diffusion
2B
81.94
82.70
78.91
62.40
25.12
82.51
98.00
63.16
26.96
99.16
25.31
Vchitect-2.0[E]
Diffusion
2B
82.24
83.54
77.06
60.41
23.73
87.04
97.20
65.35
27.57
98.57
25.01
HunyuanVideo
Diffusion
13B
83.24
85.09
75.82
60.36
19.80
91.60
94.40
67.56
26.44
99.44
23.89
Close Sourced Models
Pika-1.0
Diffusion
N/A
80.69
82.92
71.77
62.04
22.26
90.57
86.20
61.87
25.94
99.74
24.22
Kling
Diffusion
N/A
81.85
83.39
75.68
61.21
19.62
89.90
93.40
65.62
26.42
99.30
24.17
Jimeng
Diffusion
N/A
81.97
83.29
76.69
68.80
22.27
89.05
90.10
67.09
27.10
99.03
24.70
Gen-3
Diffusion
N/A
82.32
84.11
75.17
63.34
24.31
80.90
96.40
66.82
26.69
98.61
24.71
Hailuo
Diffusion
N/A
83.41
84.85
77.65
63.03
20.06
90.36
92.40
67.17
27.10
99.10
25.63
Sora
Diffusion
N/A
84.28
85.51
79.35
63.46
24.76
80.11
98.20
68.28
26.26
98.87
25.01
ARLON ‡
LLM+Diffusion
1.5B
N/A
N/A
N/A
61.01
N/A
N/A
N/A
60.98
27.27
99.37
25.33
ARLON †
LLM+Diffusion
5B
82.31
83.58
77.27
60.58
25.26
85.86
92.20
62.72
26.14
99.39
24.07
LanDiff
LLM+Diffusion
5B
85.43
86.13
82.61
64.78
25.60
91.09
97.20
65.69
27.43
99.43
25.26
B. More Analysis on VBench Benchmark
As shown in Table 8, we conduct a comprehensive comparison with state-of-the-art text-to-video generation models on
VBench benchmark. The benchmark evaluates models across multiple dimensions including quality, semantics, aesthetics,
and temporal consistency. Our LanDiff achieves superior performance in most metrics, particularly excelling in overall
quality (86.13) and semantic accuracy (82.61).
Among open-sourced models, there is a clear trend of performance improvement with model size, from Latte-1 (0.7B)
to HunyuanVideo (13B). However, our hybrid LLM+Diffusion approach (5B) demonstrates that architectural innovation
can be more impactful than simply scaling up model parameters. Notably, LanDiff outperforms much larger models like
HunyuanVideo (13B) and Mochi-1 (10B) across most metrics.
As visualized in Figure 9, we compare LanDiff with five representative models across different evaluation dimensions. The
16

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
aesthetic quality
appearance style
background consistency
color
dynamic degree
human action
imaging quality
motion smoothness
multiple objects
object class
overall consistency
scene
spatial relationship
subject consistency
temporal flickering
temporal style
0.2
0.4
0.6
0.8
1.0
Video Generation Models Performance Comparison
LanDiff
Sora
Hailuo
HunyuanVideo
Kling
CogVideoX-5B
Figure 9. Radar chart visualization of performance comparison across different dimensions on VBench. The plot compares LanDiff
against five competitive baselines: Sora, Hailuo, HunyuanVideo, Kling, and CogVideoX-5B. For better readability, the values in the radar
chart have been normalized to a scale ranging from 0.3 to 0.8. The normalization was performed using the min-max scaling formula:
normalized = 0.3 + 0.5 ×
value−min value
max value−min value. The original raw performance data can be found in Table 3 and Table 8.
17

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
radar chart reveals that LanDiff (shown in red) demonstrates well-balanced performance across all metrics, with notably
strong results in quality score and semantic accuracy. While Sora (shown in blue) achieves competitive scores in imaging
quality and scene, and HunyuanVideo excels in certain visual aspects, LanDiff maintains consistently superior performance
across the entire spectrum of metrics. Notably, while there is typically a trade-off between dynamic degree and motion
smoothness/subject consistency, LanDiff achieves a high level of dynamism while maintaining strong performance in stability
metrics - with subject consistency and motion smoothness scores within 2.3% and 2.5% of the best-performing models
respectively. Notably, while text-to-video models typically exhibit a trade-off between dynamic expressiveness and temporal
stability, LanDiff successfully balances these competing objectives—achieving high dynamism while maintaining robust
stability metrics, with subject consistency and motion smoothness scores deviating by only 2.3% and 2.5% respectively from
the state-of-the-art in each category. The comprehensive comparison with these strong baselines, including both commercial
(Sora, Hailuo, Kling) and open-source models (HunyuanVideo, CogVideoX-5B), further validates the effectiveness of our
hybrid LLM+Diffusion approach.
C. More Examples
A group of silver-colored fish with darker fins swim among green aquatic plants in an aquarium setting.
The fish move gracefully through the water, navigating around the plants, which are of various sizes
and shades of green. The aquarium environment is designed to mimic a natural habitat, with rocks
and shadows in the background contributing to the underwater scene.
CogVideoX-5B
Ours
Figure 10. Examples of text to videos generation of LanDiff and CogVideoX-5B.
18

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
A life-sized ice sculpture of a playful dog, with intricate details and a joyful expression, stands in
the middle of a sunlit, grassy field on a sweltering summer day. The ice dog, initially solid and
vibrant, begins to melt under the relentless heat, with droplets of water forming on its surface.
As the day progresses, the ice dog's form gradually diminishes, with its once sharp features
becoming blurred and distorted. The melting process accelerates, and the ice dog's body starts
to collapse, pooling into a puddle of water on the ground. By the end of the day, all that remains
is a shallow puddle, reflecting the cloudless sky, with the memory of the once majestic ice dog
now just a memory.
CogVideoX-5B
Ours
Figure 11. Examples of text to videos generation of LanDiff and CogVideoX-5B.
A close-up view of a Christmas tree reveals a variety of decorations including a purple
ornament with a gold pattern, a gold textured ornament, a small white house-shaped ornament
with red roof and gold details, and a brown pine cone. The tree branches are dense and green,
providing a natural backdrop for the ornaments. The camera pans slightly across the scene,
maintaining focus on the ornaments while subtly shifting the perspective.
CogVideoX-5B
Ours
Figure 12. Examples of text to videos generation of LanDiff and CogVideoX-5B.
19

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
Two vibrant hot air balloons, one red and the other blue, are seen soaring through a clear blue
sky, their baskets gently bumping against each other mid-air. The red balloon features intricate
gold patterns on its surface, while the blue balloon boasts a white and silver design. As they
collide, the passengers in the baskets, dressed in casual attire, react with surprise and
excitement. The scene is set against a backdrop of a picturesque landscape, with lush green
hills and a sparkling river below. The balloons' vibrant colors contrast beautifully with the azure
sky, creating a visually stunning and dynamic scene.
CogVideoX-5B
Ours
Figure 13. Examples of text to videos generation of LanDiff and CogVideoX-5B.
A colossal, human-shaped cloud towers over the earth, its massive form casting a shadow
across the landscape. The cloud man's features are distinct, with a stern expression and
outstretched arms. Suddenly, the cloud man releases a barrage of lightning bolts, illuminating
the sky as they streak towards the earth. The scene is set against a backdrop of a stormy sky,
with dark clouds and distant thunder adding to the dramatic atmosphere.
CogVideoX-5B
Ours
Figure 14. Examples of text to videos generation of LanDiff and CogVideoX-5B.
20

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
A sleek white sailboat glides gracefully across a calm, azure sea, its sails billowing gently in the
breeze. Above, a silver airplane soars through a clear blue sky. The boat's hull reflects the
sunlight, creating a shimmering effect on the water's surface. The airplane, seen in a high-
altitude flyover, casts a shadow that momentarily aligns with the boat's path, creating a fleeting
connection between sea and sky. The scene is captured in a wide shot, ensuring both the boat
and airplane are prominently centered, emphasizing their contrasting yet harmonious presence
in the vast expanse.
CogVideoX-5B
Ours
Figure 15. Examples of text to videos generation of LanDiff and CogVideoX-5B.
Streaming
t2v
Freenoise
OpenSora
Ours
"a dog drinking water."
@1s
@25s
@7s
@13s
@19s
s
Figure 16. Examples of text to long video generation.
21

The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation
Reference 
 Video
Ours
MagViT2 
Figure 17. Visualization results of video reconstruction using video tokenizer.
Reference 
 Video
Ours
MagViT2 
Figure 18. Visualization results of video reconstruction using video tokenizer.
22
