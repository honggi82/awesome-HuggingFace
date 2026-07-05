[Figure 1]

## Out of Sight but Not Out of Mind: Hybrid Memory for Dynamic Video World Models

Kaijin Chen1, Dingkang Liang1, Xin Zhou1, Yikang Ding2, Xiaoqiang Liu2, Pengfei Wan2, and Xiang Bai1

1 Huazhong University of Science and Technology 2 Kling Team, Kuaishou Technology {kjchen, dkliang}@hust.edu.cn Project Page: Hybrid-Memory-in-Video-World-Models

# arXiv:2603.25716v2[cs.CV]28Mar2026

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Static Consistency

| |
|---|

T1

Appearance Consistency

[Figure 6]

T2

[Figure 7]

T3

[Figure 8]

[Figure 9]

T4

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

In Sight Out of Sight

T5

Subject Re-entering

Subject Exiting the Frame

T1 T2 T3 the Frame T4 T5

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Motion Consistency

Fig. 1: Hybrid Memory demands the model to maintain static consistency in backgrounds, while simultaneously preserving the motion and appearance consistency of dynamic subjects during out-of-view intervals.

[Figure 24]

(b) Hybrid Memory Motion Consistency (Ours)

Abstract. Video world models have shown immense potential in simulating the physical world, yet existing memory mechanisms primarily t environments as static canvases. When dynamic subjects hide out o sight and later re-emerge, current methods often struggle, leading to frozen, distorted, or vanishing subjects. To address this, we introduce Hybrid Memory, a novel paradigm requiring models to simultaneously act as precise archivists for static backgrounds and vigilant trackers for dynamic subjects, ensuring motion continuity during out-of-view intervals. To facilitate research in this direction, we construct HM-World, the first large-scale video dataset dedicated to hybrid memory. It features 59K high-fidelity clips with decoupled camera and subject trajectories,

Subject Exiting the Frame Subject Re-entering the Frame Appearance Consistency

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

|treat of|
|---|

[Figure 30]

[Figure 31]

| |
|---|

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Work done during an internship at Kling Team, Kuaishou Technology.

encompassing 17 diverse scenes, 49 distinct subjects, and meticulously designed exit-entry events to rigorously evaluate hybrid coherence. Furthermore, we propose HyDRA, a specialized memory architecture that compresses memory into tokens and utilizes a spatiotemporal relevancedriven retrieval mechanism. By selectively attending to relevant motion cues, HyDRA effectively preserves the identity and motion of hidden subjects. Extensive experiments on HM-World demonstrate that our method significantly outperforms state-of-the-art approaches in both dynamic subject consistency and overall generation quality. Code is publicly available at https://github.com/H-EmbodVis/HyDRA.

Keywords: World Models · Spatiotemporal Consistency · Memory

#### 1 Introduction

World Models [6, 10, 23, 36] have recently garnered significant research attention for their ability to generate high-fidelity environments that align with the real world. These models have demonstrated immense potential across diverse downstream domains, including autonomous driving [9, 21, 41] and embodied intelligence [15,30]. The latest advancements in video generation [18,29,34] further validate the feasibility of modeling the physical world. Crucially, memory mechanisms have emerged as a critical frontier in advancing world models, as memory capacity dictates the spatial and temporal consistency of generated content. Specifically, it is the cognitive anchor that allows the model to retain historical context during viewpoint shifts or long-term extrapolation. Without robust memory, a simulated world quickly unravels into disconnected, chaotic frames.

While recent studies [12,20,31,33,37] have enhanced the memory capacity through advanced retrieval retrieval [20,33,37] and compression [31] techniques, they share a common blind spot: treating the world as a static canvas. They excel at memorizing and reconstructing motionless environments, but the physical world is a bustling, dynamic stage populated by subjects (e.g., walking pedestrians, running animals) governed by their independent motion logic. When dynamic subjects hide outside the camera’s field of view, these models lose track of them, often rendering the returning subjects as frozen statues, distorted phantoms, or simply letting them vanish into the air. To bridge this gap, we introduce a novel memory paradigm: Hybrid Memory, which requires the model to simultaneously perform precise memorization and viewpoint reconstruction of static backgrounds, while continuously seeking and predicting the motion of dynamic subjects. As illustrated in Fig. 1, when a subject hides out of view, the model must not only remember its appearance but also mentally predict its unseen trajectory, ensuring both visual coherence and motion consistency when they re-enter the frame.

To investigate and validate this new hybrid memory paradigm, constructing a specialized dataset and designing corresponding memory mechanisms are imperative. In this work, we introduce HM-World, the first large-scale video

dataset purpose-built to train and evaluate Hybrid Memory capabilities. HMWorld possesses two core properties: 1) meticulously designed shots with dynamic subjects exiting and entering the frame, and 2) highly diverse scenarios, subjects, and motion patterns. Comprising 59K video clips, the dataset deliberately decouples camera trajectories from subject movements, creating countless natural instances where subjects slip into the unseen margins before re-emerging. Furthermore, HM-World exhibits exceptional diversity, encompassing 17 distinctively styled scenes, 49 different subjects (including humans of various appearances and multiple animal species), 10 motion paths for subjects, and 28 types of camera trajectories.

Based on the proposed dataset HM-World, we evaluate existing methods and observe that they tend to either immobilize moving objects or distort dynamic content, lacking the hybrid memory capacity to track unseen motion. To equip models with this capacity, we propose HyDRA (Hybrid Dynamic Retrieval Attention), a memory approach designed to seek the hidden subjects and preserve dynamic consistency. HyDRA employs a Memory Tokenizer that compresses memory latents into tokens with richer information. When a subject is poised to re-enter the frame, HyDRA utilizes a spatiotemporal relevancedriven retrieval mechanism to actively scan these tokens, pulling the most crucial motion and appearance cues into the current denoising process. This allows the model to effectively rediscover the hidden subject, seamlessly picking up its trajectory where it left off. Extensive experiments on HM-World demonstrate that HyDRA significantly outperforms state-of-the-art approaches in preserving dynamic subject consistency and overall generation quality. Ablation studies further verify the robustness of our design. We hope our dataset and method can offer a fresh perspective for the community.

Our main contributions can be summarized as follows: 1) We identify the limitations of existing static-centric memory mechanisms and propose Hybrid Memory, a novel paradigm that requires models to simultaneously maintain spatial consistency for static backgrounds, and motion continuity for dynamic subjects, especially during out-of-view intervals. 2) We introduce HM-World, the first large-scale video dataset dedicated to hybrid memory research. Featuring 59K clips with diverse scenes, subjects, and motion patterns, it provides a rigorous benchmark for evaluating spatiotemporal coherence in complex, dynamic environments. 3) We propose HyDRA, a specialized memory architecture that utilizes a spatiotemporal relevance-driven retrieval mechanism with memory tokens. By attending to relevant motion cues, HyDRA effectively seeks and rediscovers hidden subjects and preserves its identity and motion, significantly outperforming existing state-of-the-art methods.

#### 2 Related Works

##### 2.1 Video World Models

Recent advances in video generation models [13,18,29,34,39,40] have demonstrated their potential in modeling the real world and synthesizing high-fidelity

clips, increasingly serving as the foundation for world models. Building on this progress, multiple video world models have been introduced [3,4,10,11,19,23,27]. GameGen-X [4] explores interactive video world models within game-like environments. Yume [23] further increases the length of generated videos through autoregressive generation. Matrix-Game 2 [10] constructs a large-scale dataset based on GTA-V and Unreal Engine 5 [8] and incorporates autoregressive denoising [13] to achieve controllability and visual quality comparable to video games. RELIC [11] focuses on static scene consistency and distills long-video generation with replayed back-propagation, enabling stable, long-duration generation. Worldplay [27] leverages large-scale, high-quality data and context forcing technique to deliver both exceptional visual quality and consistency while supporting real-time generation.

Despite significant progress, video world models continue to confront several challenges, with generation consistency being a prominent one. Current models still struggle to maintain both static and dynamic consistency across generated sequences. This issue is particularly pronounced during long-duration generation and under camera motion, where models frequently lose track of previously generated content or contextual input, leading to inconsistent outputs. Our work aims to tackle this challenge from the perspective of hybrid memory, enabling spatiotemporally consistent generation.

##### 2.2 Memory in Video Generation

Existing memory approaches primarily focus on processing the context and optimizing the interaction and propagation of contextual information during the generation process. Vmem [20] employs a 3D surfel-indexed memory structure to retrieve context, while Context-as-Memory [37] adopts Field-of-View (FOV) overlap. Worldmem [33] combines FOV-based retrieval for an external memory bank with Diffusion Forcing [5] on Minecraft data. Memory Forcing [12] further incorporates temporal memory to balance exploration and consistency. Similarly, WorldPlay [27] enhances long-term generation consistency through a context-forcing approach. Inspired by FramePack [38], MemoryPack [31] introduces an updatable semantic pack throughout the generation process, retaining semantically relevant memory. In parallel, RELIC [11] applies uniform spatial down-sampling to compress context memory.

Existing studies have achieved notable results. However, most of these methods are designed for static scenes [11,20,37] or relatively simple dynamic environments [12,31,33], and have not been specifically optimized for complex dynamic scenes involving moving subjects and dynamic elements. Although Genie

###### 3 [2] demonstrates remarkable dynamic consistency, it is a closed-source model with technical details remaining undisclosed. This research gap persists in both dataset construction and method design. To address this, our work focuses on hybrid memory in complex dynamic scenes, tackling the challenge from both methodological and dataset perspectives.

[Figure 37]

###### TrajectoryCamera Exiting the frame Re-entering the frame

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

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

Fig. 2: Instances of exit-entry camera motion.

#### 3 HM-World: Dataset

To address the research gap in hybrid memory, we conduct an in-depth analysis of its definition and inherent challenges for current video world models in Sec.3.1. Building upon this analysis, we introduce HM-World, a large-scale dataset constructed for Hybrid Memory in Video World Models, and detail its characteristics in Sec. 3.2.

##### 3.1 Hybrid Memory

Memory refers to the model’s ability to retain information from inputs or generated content, ensuring consistency throughout the generation process. Static memory ensures the consistency of immobile elements (e.g., buildings, roads), and is typically evaluated by assessing whether a scene looks identical when the camera returns to a previous pose [37]. Hybrid memory, however, demands a far more sophisticated cognitive leap. It requires the model to simultaneously anchor the static background while tracking the dynamic subjects (e.g., pedestrians, running dogs). As illustrated in Fig. 2, when a subject exits and re-enters the frame, hybrid memory dictates that it must not only retain its original visual identity but also reappear at a plausible location with a consistent motion state.

Achieving hybrid memory is challenging for several reasons: 1) Need for spatiotemporal decoupling. Unlike static memory, which merely maps camera poses to a fixed 3D space, hybrid memory forces the model to independently untangle the camera’s ego-motion from the subject’s independent trajectory. 2) Out-of-view extrapolation. Once a subject steps off-stage, the model loses direct visual evidence and must implicitly simulate the subject’s movement in the latent space. 3) Feature entanglement. In standard diffusion latents, static background features and subject features are heavily coupled. Retrieving historical context without isolating the dynamic cues often causes the subjects to freeze into the background or distort unnaturally.

[Figure 61]

###### 6 K. Chen et al.

[Figure 62]

[Figure 63]

| |[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]|
|---|---|
| |[Figure 67]<br><br>[Figure 68]|

|[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]| | |
|---|---|---|
| | | |

[Figure 74]

(a) 3D UE Scenes (b) Subjects

|[Figure 75]<br><br>[Figure 76]|
|---|

|[Figure 77]<br><br>[Figure 78]|
|---|

(c) Subject Trajectories (d) Camera Trajectories

Fig. 3: Construction Procedure of HM-World. We combine (a) 3D scenes, (b) subjects, (c) subject trajectories, and (d) camera trajectories to render data containing dynamics in Unreal Engine 5.

To conquer these complex dynamics and bridge the research gap, a dedicated testing ground is essential. As natural videos with perfectly captured, unoccluded exit-and-re-entry events are remarkably scarce, we constructed HM-World, a dataset explicitly tailored for hybrid memory.

##### 3.2 Dataset Characteristics

Since videos with exit-entry events are rarely found on the Internet, we construct the dataset by implementing a data rendering pipeline within Unreal Engine 5 [8]. As depicted in Fig. 3, our data generation process is structured along four dimensions: scenes, subjects, subject trajectories, and camera trajectories. We first collect 17 stylistically diverse scenes to serve as the environmental background. Then, 49 distinct subjects, encompassing people of varied appearances and animals of multiple species, are combined into groups of 1 to 3. Each combination is procedurally placed within a scene. Furthermore, each subject is associated with its own motion animation and follows a randomly selected trajectory from a set of 10 predefined paths.

To guarantee a rich density of exit-entry events, we meticulously designed the camera motions. Moving beyond simple unidirectional tracking, our camera trajectories incorporate deliberate back-and-forth camera motions, as illustrated in Fig. 2, to actively induce hide-and-reappear dynamics. For instance, a leftward pan followed by a rightward pan typically causes a captured subject to leave and re-enter the frame. Following this principle, we designed 28 distinct camera trajectories. Additionally, each camera movement is assigned multiple initial positions, further enhancing the diversity of camera motion sequences.

Table 1: The comparison between existing datasets and HM-World dataset. "Dynamic Subject" means including moving subjects, "Exit-Enter" refers to containing exit-entry events in clips, and "Subject Pose" denotes including annotated 3D poses of subjects.

Dynamic Subject

Subject Exit-Enter

Subject Pose

Camera Movable

Total Num.

Dataset Reference

WorldScore [7] ICCV 25 ✓ ✗ ✗ ✓ 3K Context-As-Memory [37] SIGGRAPH Asia 25 ✗ ✗ ✗ ✓ 10K Multi-Cam Video [1] ICCV 25 ✓ ✗ ✗ ✓ 136K 360°-Motion [32] ICLR 25 ✓ ✗ ✓ ✗ 5.4K

HM-World (ours) - ✓ ✓ ✓ ✓ 59K

After procedurally combining elements from all four dimensions and filtering clips that lack exit-entry events, we obtain a final collection of 59,225 high-fidelity video clips. Every sample is comprehensively annotated with the rendered video, a descriptive caption generated by MiniCPM-V [35], corresponding camera poses, per-frame positions of all subjects, and precise timestamps marking each subject’s exit from and entry into the frame. Tab. 1 highlights the comparison between HM-World and existing datasets. Specifically, the Contextas-Memory dataset only contains static scenes. WorldScore includes numerous real-world scenes with certain dynamic elements, but its scale is limited to only

- 3K. Multi-Cam Video features dynamic subjects, but they only perform actions in place. 360 °-Motion contains moving subjects, but the camera remains static, and the subjects are always within the field of view. In contrast, our HM-World not only features rich, dynamic subjects and complex camera trajectories, but also includes specific in-and-out-of-frame events for hybrid memory.

4 Hybrid Dynamic Retrieval Attention

Given a sequence of context frames Xctx ∈ RN×C×H×W and a full sequence of camera trajectory P = {Pctx,Ptgt} spanning both historical and future timestamps, our goal is to predict the target frames Xtgt ∈ RM×C×H×W. Unlike static scene generation, the context frames Xctx feature dynamic subjects governed by their independent motion. As the camera viewpoint shifts according to Ptgt (e.g., panning or rotation), these subjects frequently hide and re-enter the camera’s field of view. To synthesize high-fidelity future frames Xtgt, the model must preserve the static background while seeking the moving subjects to maintain their appearance and motion consistency. To achieve this, we introduce HyDRA (Hybrid Dynamic Retrieval Attention), a memory method designed to decouple and preserve consistency of dynamic subjects.

- 4.1 Base Architecture and Camera Injection

Overall Architecture. As depicted in Fig. 4, our approach is built upon a fullsequence video diffusion model, comprising a causal 3D VAE [17] and a Diffusion Transformer (DiT) [24]. Each DiT block integrates dynamic retrieval attention, a projector, cross-attention, and a feedforward network (FFN). The diffusion timestep is encoded via a Multi-Layer Perceptron (MLP) to modulate the DiT

blocks. The model follows Flow Matching [22]. Given a sequence of video frames x, the 3D VAE encodes it into video latent z0 ∈ RC×f×h×w, compressing both temporal and spatial dimensions. During the training phase, the noised latent zt at timestep t is obtained through linear interpolation between z0 and Gaussian noise z1 ∼ N(0,I). The model u learns to predict the ground-truth velocity vt = z0 − z1 at timestep t ∈ [0,1], with the loss function defined as:

0,z1,t||u(zt,t;θ) − vt||2, (1)

Lθ = Ez

where θ represents the model parameters. During the inference phase, randomly sampled Gaussian noise is progressively denoised to yield a clean latent, which is then decoded by the 3D VAE Decoder to reconstruct the video sequence.

Camera Injection. To enable precise spatial control of generated content, we inject camera trajectories into the model as an explicit condition. Suppose the camera pose sequence of length f is denoted as P = {(Ri,ti)}fi=1, where Ri ∈ R3×3 and ti ∈ R3 represent the rotation matrix and the translation vector for the i-th frame, respectively. We flatten and concatenate these parameters to form a unified camera condition ccam ∈ Rf×12. Following ReCamMaster [1], we employ a camera encoder Ecam(·), implemented as a MLP layer to encode ccam. The encoded camera features are then broadcast spatially and added element-wise to the latent features. Formally, let Hin be the sequence features fed into the DiT blocks, the camera-injected feature Hout is formulated as:

[Figure 79]

❄

[Figure 80]

×N

Feed Forward Network

[Figure 81]

❄

[Figure 82]

Cross Attention

[Figure 83]

🔥

[Figure 84]

Projector

[Figure 85]

🔥

[Figure 86]

Dynamic Retrieval Attention

[Figure 87]

[Figure 88]

Memory Tokenizer 🔥

[Figure 89]

[Figure 90]

[Figure 91]

Camera Encoder 🔥

[Figure 92]

3D VAE Encoder ❄

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Context Frames Target Frames

Camera Sequence

Fig. 4: Model architecture.

Hout = Hin + Ecam(ccam), (2) where Ecam(ccam) is projected to match the exact channel dimension of Hin.

[Figure 101]

[Figure 102]

Retrieval Attention Map

[Figure 103]

[Figure 104]

Target Frame

[Figure 105]

[Figure 106]

[Figure 107]

𝑘 𝑘

Attention Calculation

[Figure 110]

[Figure 111]

[Figure 112]

𝑞

[Figure 118]

Selected Tokens

##### 4.2 Memory Tokenization for Retrieval

Top-K Retrieval

[Figure 121]

Memory Tokens Target Query

In our framework, the encoded memory latents, denoted as Zmem, serve as the primary representation of memory. A naive approach to memory utilization would involve injecting the entire Zmem into the generation process. However, this not only incurs computational overhead but also floods the model with irrelevant noise. Such noise can easily mislead the model’s reasoning pathways, ultimately resulting in spatially and temporally inconsistent generation. Therefore, a retrieval mechanism is essential to filter the memory and accurately recall the hidden subject outside the current frame.

𝒒𝒊 Affinity Computation

[Figure 124]

[Figure 126]

[Figure 130]

[Figure 135]

[Figure 136]

[Figure 139]

Memory Tokenizer

[Figure 140]

Memory Tokenizer

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

| | |
|---|---|
|[Figure 147]<br><br>[Figure 148]| |

| | |
|---|---|
|<br><br>| |

| | |
|---|---|
|<br><br>| |

Context Frames Input Noise

|[Figure 153]<br><br>[Figure 154]|
|---|

|<br><br>|
|---|

…

(a) Memory Tokenization

(b) Dynamic Retrieval Attention

❄

×N

Feed Forward Network

❄

Cross Attention

🔥

Projector

🔥

Dynamic Retrieval Attention

Memory Tokenizer🔥

Noise

[Figure 167]

[Figure 169]

Camera Encoder

[Figure 170]

🔥

3D VAE Encoder ❄

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Context Frames Target Frames Camera Sequence

Hybrid Memory for Dynamic Video World Models 9

[Figure 180]

[Figure 181]

###### Retrieval Attention Map

[Figure 182]

[Figure 183]

Target Frame Attention Calculation

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

𝑘 𝑘

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

𝑞

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

Selected Tokens

Top-K Retrieval

Reshape

[Figure 200]

[Figure 201]

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | |[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]| | | |
| | | | | | | | |
| | | | | | | | |

Memory Tokens Target Query

𝒒𝒊 Affinity Computation

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

Memory Tokenizer

Memory Tokenization

[Figure 223]

[Figure 224]

[Figure 225]

| | |
|---|---|
|[Figure 226]<br><br>[Figure 227]| |

| | |
|---|---|
|[Figure 228]<br><br>[Figure 229]| |

| | |
|---|---|
|[Figure 230]<br><br>[Figure 231]| |

Memory Frames Input Noise

[Figure 232]

[Figure 233]

|[Figure 234]<br><br>[Figure 235]|
|---|

|[Figure 236]<br><br>[Figure 237]|
|---|

…

[Figure 238]

(a) Memory Tokenization

(b) Dynamic Retrieval Attention

Fig. 5: Overview of HyDRA. (a) Memory Tokenization Module. (b) Dynamic retrieval attention computes relevance between the target query and memory tokens to retrieve the top-k relevant tokens, enabling the model to recall associated hybrid memory.

Nevertheless, performing retrieval directly on the latent representation could be sub-optimal. Under our proposed hybrid memory paradigm, the task involves highly dynamic subjects and complex spatial relationships driven by camera movements. Direct retrieval from raw, uncoupled latents can lack the expressiveness needed to fully capture the underlying motion of dynamic subjects and the associated camera transformations, potentially undermining spatiotemporal consistency in the generated content.

To overcome this limitation, we introduce a 3D-convolution-based memory tokenizer, designed to process both spatial and temporal dimensions simultaneously. We argue that facilitating spatiotemporal interaction on the latents yields memory tokens with much deeper, motion-aware representations. This enriched representation is crucial for optimizing the retrieval process and ensuring consistent generation, which is validated by our extensive empirical experiments.

Specifically, the Memory Tokenizer Tmem processed the latents Zmem into compact memory tokens M. By employing 3D convolutions, the tokenizer expands the spatiotemporal receptive field to capture long-duration motion information. Formally, this transformation is defined as:

′×fmem′ ×h×w, (3)

M = Tmem(Zmem), M ∈ RC

where fmem′ represents the temporal dimension, and h×w denotes the downsampled spatial resolution. By compressing the raw latents into dense, spatiotemporally-

aware memory tokens M, the model effectively filters out irrelevant context while preserving the essential motion and appearance cues. These refined tokens M then serve as the foundation for our dynamic retrieval attention module, which will be detailed in the following section.

##### 4.3 Dynamic Retrieval Attention

As discussed in Sec. 4.2, indiscriminately injecting all historical context degrades video consistency and inflates computational cost. To tackle this, a retrieval mechanism is imperative for optimizing the information flow. Building upon the principles of attention [28], we propose Dynamic Retrieval Attention, a spatiotemporal-informed retrieval method and memory mechanism that directly replaces the standard 3D self-attention layers within the base model.

′×ftgt×H′×W′ and the memory tokens M ∈ RC

Given the denoising target latents Ztgt ∈ RC

′×fmem′ ×h×w, we first project them into their respective Query, Key, and Value. Concretely, the target latents are projected into queries Q, while the memory tokens are projected into keys Kmem and values Vmem.

To perform dynamic retrieval, we process the query set qi corresponding to each target latent i ∈ {1,...,ftgt} sequentially. Because qi and Kmem operate at different spatial resolutions, we first apply spatial pooling to downsample qi into q˜i ∈ RC

′×h×w, aligning it with the memory tokens. We then compute a spatiotemporal affinity metric between the downsampled query q˜i and each temporal slice of the memory key kmem,j (where j ∈ {1,...,fmem′ }). Since they share the same spatial resolution and channel dimension, the affinity Si,j is calculated by taking the element-wise product across the spatial dimensions:

- w
- x=1

h

1 √

⟨q˜i(x,y),kmem,j(x,y)⟩, (4)

Si,j =

d

y=1

where ⟨·,·⟩ denotes the channel-wise inner product, and d is the channel dimension for scaling.

The affinity metric effectively quantifies the spatiotemporal correspondence between the current target latent and the memory token. Based on these affinities, we employ a Top-K selection strategy to filter the memory tokens, isolating the subset of memory that exhibits the strongest correlation with qi:

Ii = TopK(Si,K), Ksel = {kmem,j | j ∈ Ii}, Vsel = {vmem,j | j ∈ Ii}, (5) where Ii represents the indices of the K most relevant memory tokens.

While retrieving historical memory is crucial for long-term consistency, maintaining local denoising stability is equally important. To preserve the structural integrity of the original self-attention, we forcefully include the queries’ own local temporal window into the attention computation. Let Kloc and Vloc denote the keys and values derived from the adjacent latents within a local window Wi centered around frame i in the target sequence. We first flatten these local features and the retrieved memory features, then concatenate them to form the final keys Ki′ = [Ksel,Kloc] and values Vi′ = [Vsel,Vloc].

Finally, after flattening the query qi, the dynamic retrieval attention for the i-th latent is computed using the standard attention formulation:

Attention(qi,Ki′,Vi′) = Softmax

qi(Ki′)T

√

d

Vi′. (6)

By iterating this process across all queries in the denoising sequence, the model selectively attends to the most pertinent motion and appearance cues of the outof-sight subjects. Extensive experiments validate that this method successfully tracks hidden subjects, preserves spatiotemporal consistency, and substantially decreases the computational burden.

#### 5 Experiments

##### 5.1 Experiment Setup

Implement Details. We build our method on Wan2.1-T2V-1.3B [29]. The model encodes 77 context frames and temporally downsamples them by a factor of 4 via a 3D VAE. For our proposed modules, the memory tokenizer employs a 3D convolution with a kernel size of 2 × 4 × 4. In the Dynamic Retrieval Attention, the retrieval token length is set to 10, and the local window for the denoising latent is 5. We train our model on the proposed HM-World dataset for 10K iterations using 32 GPUs, with a total batch size of 32.

Evaluation Protocol. To evaluate our method, we construct a test set comprising 1000 video samples randomly selected from the HM-World dataset, including scenes and subjects that are unseen during training to assess generalization. Our evaluation metrics span three categories: 1) General Memory Capacity. PSNR, SSIM, and LPIPS analyze pixel-wise differences across frames to measure overall reconstruction fidelity. 2) Frame-level Consistency. We adopt Subject Consistency and Background Consistency from the Vbench [14] to measure frame-level coherence. 3) Dynamic Subject Consistency (DSC). To isolate and evaluate the motion and appearance consistency of moving subjects, especially in re-entering events. We propose a new metric DSC (Dynamic Subject Consistency). Specifically, we utilize bounding boxes of moving subjects, which are obtained via YOLOv11 [16], to crop the subject regions from the predicted video, the GT video, and the context video. We then extract semantic features from these cropped regions using a pretrained CLIP [25] model. After spatial alignment and temporal normalization, we calculate the feature similarities to yield two scores DSCctx and DSCGT, formulated as:

DSCGT = sim Fpred,Fgt , DSCctx = sim Fpred,Fctx , (7)

where sim(·,·) refers to the spatially averaged cosine similarity across the feature channels, Fpred, Fgt, and Fctx denote subject features from predicted video, GT video, and context video. DSCGT evaluates motion and appearance fidelity against the ground truth, while DSCctx evaluates against historical context.

##### 5.2 Main Results

In this section, we evaluate the performance of our proposed method against a baseline and state-of-the-art approaches, including DFoT [26] and Context-asMemory [37]. The baseline is built upon a Wan2.1-T2V-1.3B model equipped

###### Table 2: Quantitative comparison with other methods.

Subj. Cons.

Bg. Cons.

Method Reference PSNR SSIM LPIPS DSCctx DSCGT

Baseline - 18.696 0.517 0.356 0.812 0.837 0.903 0.925 DFoT [26] ICML 25 17.693 0.482 0.410 0.803 0.826 0.893 0.913 Context-as-Memory [37] SIGGRAPH Asia 25 18.921 0.530 0.342 0.816 0.839 0.911 0.922

HyDRA (ours) - 20.357 0.606 0.289 0.827 0.849 0.926 0.932

Table 3: Quantitative comparison against the state-of-the-art commercial model.

Subject Consistency

Background Consistency

Method PSNR SSIM LPIPS DSCctx DSCGT

WorldPlay [27] 14.855 0.355 0.500 0.822 0.832 0.910 0.925 HyDRA (ours) 20.357 0.606 0.289 0.827 0.849 0.926 0.932

with a camera encoder, which directly concatenates the context latents and the noisy latents as the input of the DiT. For fair comparisons, these models are trained on our dataset, strictly adhering to the same training configurations used for our approach. Furthermore, we include a zero-shot evaluation of WorldPlay [27], a cutting-edge commercial known for its exceptional consistency. The comparison results are summarized in Tab. 2, Tab. 3 and Fig. 6.

Quantitative Comparison. As shown in Tab. 2, HyDRA consistently outperforms competing approaches across all evaluation metrics. Compared to the baseline, our model achieves significant improvements, lifting PSNR from 18.696 to 20.357 and SSIM from 0.517 to 0.606. This demonstrates that HyDRA achieves superior reconstruction accuracy for future frames. Crucially, our method attains the highest DSCctx and DSCGT scores of 0.827 and 0.849, respectively, proving its robust capability to track subjects and maintain their appearance and motion consistency, both in aligning with historical context and predicting future states. The Subject Consistency of 0.926 and Background Consistency of 0.932 further corroborate that it successfully anchors the static stage while preserving overall visual coherence. While DFoT relies on a neighbor context window, yielding a PSNR of 17.693, and Context-as-Memory utilizes FOV-based context filtering, yielding 18.921, our method surpasses them both, likely because we leverage retrieval over richer token representations and fuse spatiotemporal relationships via dynamic retrieval attention. Tab. 3 presents the comparison with the zero-shot performance of WorldPlay. Our method surpasses WorldPlay across all metrics, with a notable PSNR gap of 5.502. Although WorldPlay exhibits lower performance on GT-referenced metrics (e.g., PSNR of 14.855, DSCGT of 0.832) due to domain distribution gap and lack of specific finetuning, it demonstrates remarkable robustness on context-referenced metrics by achieving a DSCctx of 0.822. This observation not only confirms that extensively trained models possess fair hybrid consistency but also indirectly validates the rationality of our proposed DSC metrics in reflecting dynamic subject consistency. Ultimately, these impressive results highlight the exceptional capabilities of our model, demonstrating its superiority even over established commercial models.

Qualitative Comparison. We present a qualitative comparison in Fig. 6. In the case of complex exit-and-entry events, the baseline and Context-as-Memory exhibit severe subject distortion and motion incoherence. DFoT fails to maintain

Context Frame

Frame 10 Frame 20 Frame 40 Frame 60

Exit then Re-enter

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

GTBaselineDFoTHyDRA(ours)Context-as-Memory

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

| | | |
|---|---|---|

Baseline的图换⼀个 Ours换⼀帧

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

WorldPlay

Incoherent Motion

Fig. 6: Qualitative comparison with other methods. The green boxes in the figure represent consistently generated subjects, while the red boxes stand for failure cases.

Table 4: Kernel Size of Memory Tokenizer.

Subject Consistency

Background Consistency

T H × W PSNR SSIM LPIPS DSCctx DSCGT

2 2 × 2 20.113 0.599 0.299 0.820 0.843 0.919 0.929 2 4 × 4 20.357 0.606 0.289 0.827 0.849 0.926 0.932 2 8 × 8 20.230 0.610 0.292 0.822 0.843 0.923 0.927 1 4 × 4 19.076 0.554 0.337 0.819 0.841 0.912 0.925

subject integrity, leading to complete vanishing. While WorldPlay manages to preserve the subject’s appearance consistency, it suffers from stuttering movements and unnatural actions. In contrast, our method successfully maintains hybrid consistency, preserving both the subject’s identity and motion coherence after the subject re-enters the frame. Due to space limitations, more generation results are provided in the supplementary materials.

##### 5.3 Ablation Study

In this section, we conduct comprehensive ablation studies to validate the effectiveness of the core components in our method.

Kernel Size of Memory Tokenizer. We first evaluate the impact of different kernel sizes in the memory tokenizer, with the results summarized in Tab. 4. The kernel size is denoted as T × H × W, representing the temporal, height, and width dimensions, respectively. The results indicate that our model exhibits strong robustness to variations in the spatial dimensions. The performance differences among spatial dimensions’ settings are marginal, as transitioning from the optimal 4×4 configuration to 2×2 or 8×8 results in a minor PSNR decrease of only 0.244 and 0.127, respectively. In contrast, when the temporal dimension

###### Table 5: Number of retrieved tokens.

Subject Consistency

Background Consistency

Setting PSNR SSIM LPIPS DSCctx DSCGT

5 19.309 0.566 0.339 0.817 0.836 0.913 0.927 10 20.357 0.606 0.289 0.827 0.849 0.926 0.932 15 20.333 0.612 0.291 0.828 0.842 0.925 0.935

###### Table 6: Approaches to retrieve tokens.

Subject Consistency

Background Consistency

Method PSNR SSIM LPIPS DSCctx DSCGT

FOV Overlap 19.776 0.586 0.300 0.820 0.844 0.908 0.930 Dynamic Affinity 20.357 0.606 0.289 0.827 0.849 0.926 0.932

is reduced to 1, we observe a significant performance drop of 1.281 in PSNR and 0.014 in DSCGT, which demonstrates the necessity of temporal interaction within the tokenizer for capturing long-term dynamic information.

Number of Retrieved Tokens. We investigate the effect of the retrieved memory token length in Tab. 5. Retrieving only 5 tokens yields suboptimal performance with a PSNR of 19.309, indicating that an overly restricted token count leads to severe information loss. Conversely, increasing the number to 10 and 15 generates better and more stable results, with negligible differences between the two. This suggests that a moderate number of tokens is sufficient to provide the necessary spatiotemporal information without introducing redundant noise.

Token Retrieval Approaches. We ablate the token retrieval mechanism by comparing our dynamic affinity retrieval with FOV overlap retrieval in Tab. 6. Since a single memory token in our architecture aggregates information from multiple frames with varying camera poses, we average the camera poses of the source frames to represent the token’s pose. We then follow Context-asMemory [37] to calculate the FOV overlap between the token and the target frame to perform retrieval. Experimental results demonstrate that our method outperforms the FOV-based approach across all metrics, notably improving Subject Consistency from 0.908 to 0.926. This superiority stems from leveraging QK interactions to assess fine-grained spatiotemporal relevance, whereas the FOVbased approach relies solely on static geometry overlap.

#### 6 Conclusion

In this paper, we introduce the novel paradigm of Hybrid Memory, challenging models to simultaneously maintain static background consistency and dynamic subject coherence, particularly during complex exit-and-re-entry events. To systematically facilitate research in this field, we construct HM-World, the first large-scale video dataset dedicated to hybrid memory, featuring highly diverse scenarios and complex dynamic processes. To tackle the challenge of hybrid memory, we propose HyDRA, an advanced memory architecture specifically designed to effectively extract and retrieve motion and appearance cues for consistent generation. Extensive experiments demonstrate that HyDRA significantly

outperforms existing methods. We hope that the hybrid memory paradigm, alongside the HM-World dataset and the HyDRA framework, will inspire new research and provide a solid foundation for advancing video world models.

Limitations and Future Work. Despite the promising results, our work still presents certain limitations. Specifically, HyDRA’s performance in maintaining consistent generation tends to degrade in highly complex scenes involving three or more subjects or severe occlusions. In future work, we plan to explore more advanced and robust memory mechanisms to handle intricate multi-subject dynamics and scale our approach to unconstrained real-world environments.

#### Acknowledgements

We express our sincere gratitude to Jichao Wang, Xiaole Xiong, Siyuan Luo, Mengyuan Li, Boyu Zheng, and Yike Yin from Kuaishou Technology for their invaluable assistance in developing the HM-World dataset.

#### References

- 1. Bai, J., Xia, M., Fu, X., Wang, X., Mu, L., Cao, J., Liu, Z., Hu, H., Bai, X., Wan, P., et al.: Recammaster: Camera-controlled generative rendering from a single video. In: ICCV (2025)
- 2. Ball, P.J., Bauer, J., Belletti, F., Brownfield, B., Ephrat, A., Fruchter, S., Gupta, A., Holsheimer, K., Holynski, A., Hron, J., Kaplanis, C., Limont, M., McGill, M., Oliveira, Y., Parker-Holder, J., Perbet, F., Scully, G., Shar, J., Spencer, S., Tov, O., Villegas, R., Wang, E., Yung, J., Baetu, C., Berbel, J., Bridson, D., Bruce, J., Buttimore, G., Chakera, S., Chandra, B., Collins, P., Cullum, A., Damoc, B., Dasagi, V., Gazeau, M., Gbadamosi, C., Han, W., Hirst, E., Kachra, A., Kerley, L., Kjems, K., Knoepfel, E., Koriakin, V., Lo, J., Lu, C., Mehring, Z., Moufarek, A., Nandwani, H., Oliveira, V., Pardo, F., Park, J., Pierson, A., Poole, B., Ran, H., Salimans, T., Sanchez, M., Saprykin, I., Shen, A., Sidhwani, S., Smith, D., Stanton, J., Tomlinson, H., Vijaykumar, D., Wang, L., Wingfield, P., Wong, N., Xu, K., Yew, C., Young, N., Zubov, V., Eck, D., Erhan, D., Kavukcuoglu, K., Hassabis, D., Gharamani, Z., Hadsell, R., van den Oord, A., Mosseri, I., Bolton, A., Singh, S., Rocktäschel, T.: Genie 3: A new frontier for world models (2025), https://deepmind.google/models/genie/
- 3. Bar, A., Zhou, G., Tran, D., Darrell, T., LeCun, Y.: Navigation world models. In: CVPR (2025)
- 4. Che, H., He, X., Liu, Q., Jin, C., Chen, H.: Gamegen-x: Interactive open-world game video generation. In: ICLR (2025)
- 5. Chen, B., Martí Monsó, D., Du, Y., Simchowitz, M., Tedrake, R., Sitzmann, V.: Diffusion forcing: Next-token prediction meets full-sequence diffusion. In: NeurIPS

(2024)

- 6. Cui, Y., Chen, H., Deng, H., Huang, X., Li, X., Liu, J., Liu, Y., Luo, Z., Wang, J., Wang, W., et al.: Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583 (2025)
- 7. Duan, H., Yu, H.X., Chen, S., Fei-Fei, L., Wu, J.: Worldscore: A unified evaluation benchmark for world generation. In: ICCV (2025)

- 8. Epic Games: Unreal engine 5. https://www.unrealengine.com/en-US/unrealengine-5 (2022), accessed: 2025-10-22
- 9. Gao, S., Yang, J., Chen, L., Chitta, K., Qiu, Y., Geiger, A., Zhang, J., Li, H.: Vista: A generalizable driving world model with high fidelity and versatile controllability. In: NeurIPS (2024)
- 10. He, X., Peng, C., Liu, Z., Wang, B., Zhang, Y., Cui, Q., Kang, F., Jiang, B., An, M., Ren, Y., et al.: Matrix-game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009 (2025)
- 11. Hong, Y., Mei, Y., Ge, C., Xu, Y., Zhou, Y., Bi, S., Hold-Geoffroy, Y., Roberts, M., Fisher, M., Shechtman, E., et al.: Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040 (2025)
- 12. Huang, J., Hu, X., Han, B., Shi, S., Tian, Z., He, T., Jiang, L.: Memory forcing: Spatio-temporal memory for consistent scene generation on minecraft. arXiv preprint arXiv:2510.03198 (2025)
- 13. Huang, X., Li, Z., He, G., Zhou, M., Shechtman, E.: Self forcing: Bridging the train-test gap in autoregressive video diffusion. In: NeurIPS (2025)
- 14. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al.: Vbench: Comprehensive benchmark suite for video generative models. In: CVPR (2024)
- 15. Jiang, Y., Chen, S., Huang, S., Chen, L., Zhou, P., Liao, Y., He, X., Liu, C., Li, H., Yao, M., et al.: Enerverse-ac: Envisioning embodied environments with action condition. arXiv preprint arXiv:2505.09723 (2025)
- 16. Khanam, R., Hussain, M.: Yolov11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725 (2024)
- 17. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)
- 18. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.: Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024)
- 19. Li, J., Tang, J., Xu, Z., Wu, L., Zhou, Y., Shao, S., Yu, T., Cao, Z., Lu, Q.: Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201 (2025)
- 20. Li, R., Torr, P., Vedaldi, A., Jakab, T.: Vmem: Consistent interactive video scene generation with surfel-indexed view memory. In: ICCV (2025)
- 21. Liang, D., Zhang, D., Zhou, X., Tu, S., Feng, T., Li, X., Zhang, Y., Du, M., Tan, X., Bai, X.: Unifuture: A 4d driving world model for future generation and perception. In: ICRA (2026)
- 22. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022)
- 23. Mao, X., Lin, S., Li, Z., Li, C., Peng, W., He, T., Pang, J., Chi, M., Qiao, Y., Zhang, K.: Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744

(2025)

- 24. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: ICCV (2023)
- 25. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021)
- 26. Song, K., Chen, B., Simchowitz, M., Du, Y., Tedrake, R., Sitzmann, V.: Historyguided video diffusion. In: ICML (2025)
- 27. Sun, W., Zhang, H., Wang, H., Wu, J., Wang, Z., Wang, Z., Wang, Y., Zhang, J., Wang, T., Guo, C.: Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614 (2025)

- 28. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: NeurIPS (2017)
- 29. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., et al.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)
- 30. Wang, X., Liu, L., Cao, Y., Wu, R., Qin, W., Wang, D., Sui, W., Su, Z.: Embodiedgen: Towards a generative 3d world engine for embodied intelligence. arXiv preprint arXiv:2506.10600 (2025)
- 31. Wu, X., Zhang, G., Xu, Z., Zhou, Y., Lu, Q., He, X.: Pack and force your memory: Long-form and consistent video generation. arXiv preprint arXiv:2510.01784 (2025)
- 32. Xiao, F., Liu, X., Wang, X., Peng, S., Xia, M., Shi, X., Yuan, Z., Wan, P., Zhang, D., Lin, D.: 3dtrajmaster: Mastering 3d trajectory for multi-entity motion in video generation. In: ICLR (2024)
- 33. Xiao, Z., Yushi, L., Zhou, Y., Ouyang, W., Yang, S., Zeng, Y., Pan, X.: Worldmem: Long-term consistent world simulation with memory. In: NeurIPS (2025)
- 34. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al.: Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024)
- 35. Yao, Y., Yu, T., Zhang, A., Wang, C., Cui, J., Zhu, H., Cai, T., Li, H., Zhao, W., He, Z., et al.: Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800 (2024)
- 36. Ye, D., Zhou, F., Lv, J., Ma, J., Zhang, J., Lv, J., Li, J., Deng, M., Yang, M., Fu, Q., et al.: Yan: Foundational interactive video generation. arXiv preprint arXiv:2508.08601 (2025)
- 37. Yu, J., Bai, J., Qin, Y., Liu, Q., Wang, X., Wan, P., Zhang, D., Liu, X.: Context as memory: Scene-consistent interactive long video generation with memory retrieval. In: ACM SIGGRAPH Asia (2025)
- 38. Zhang, L., Cai, S., Li, M., Wetzstein, G., Agrawala, M.: Frame context packing and drift prevention in next-frame-prediction video diffusion models. In: NeurIPS

(2025)

- 39. Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., You, Y.: Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404 (2024)
- 40. Zhou, X., Liang, D., Chen, K., Feng, T., Chen, X., Lin, H., Ding, Y., Tan, F., Zhao, H., Bai, X.: Less is enough: Training-free video diffusion acceleration via runtime-adaptive caching. arXiv preprint arXiv:2507.02860 (2025)
- 41. Zhou, X., Liang, D., Tu, S., Chen, X., Ding, Y., Zhang, D., Tan, F., Zhao, H., Bai, X.: Hermes: A unified self-driving world model for simultaneous 3d scene understanding and generation. In: ICCV (2025)

## Out of Sight but Not Out of Mind: Hybrid Memory for Dynamic Video World Models

### Supplementary Material

This file provides additional information about our work, mainly from more generation results and ablation studies.

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

Fig. 1: The results generated by HyDRA.

#### A Qualitative Analysis

- A.1 Generation Results

Fig. 1 shows HyDRA’s generation results across multiple scenes, subjects, and trajectories. HyDRA effectively implements memorization of both background and subjects in complex dynamic scenarios with exit-entry events, maintaining appearance and motion consistency.

- A.2 Open-Domain Results

We collect open-domain videos featuring subject motion from the Internet and apply back-and-forth camera movements for inference. The results in Fig 2 demonstrate that even in entirely unseen scenes, HyDRA exhibits good capacity of hybrid memory.

#### B More Ablation Studies

In this section, we further conduct comprehensive ablation analyses on our proposed method and core designs.

Analysis of Retrieval Approaches. We first compare our dynamic-affinitybased retrieval method with the traditional Field of View (FOV) overlap filtering

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

###### Fig. 2: Open-domain results of HyDRA.

Frames selected by FOV Overlap Frames selected by Dynamic Affinity

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

Retrieval Approaches

Context Frame 10 Context Frame 20 Frame 20 Frame 40 Frame 60

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

FOV Overlap

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

Dynamic Affinity

- Fig. 3: Qualitative comparison between retrieval methods. The upper displays frames selected by different methods, while the lower shows the generation results. Selected frames are the source frames of the selected tokens.

approach. As illustrated in Fig. 3, during a long camera movement involving complex exit-and-re-entry events, the FOV-based method merely selects the nearest camera poses corresponding to the re-entry clip. Consequently, it mistakenly retrieves empty shots, leading to a severe loss of critical appearance information and inconsistent generation. In contrast, our dynamic affinity approach filters memory tokens based on feature-level correlations. It successfully retrieves keyframes containing rich subject details, thereby maintaining the appearance and motion consistency of the subject after re-entry. Furthermore, we investigate the distribution of the retrieved tokens across different filtering strategies in Fig. 4. The FOV overlap method relies on static 3D geometric calculations, meaning the selected memory tokens remain fixed throughout the entire inference stage. In contrast, our dynamic affinity method computes feature-level correlations dynamically. As a result, it adaptively selects different tokens at different timesteps and across different DiT layers. This dynamic mechanism grants

[Figure 339]

[Figure 340]

(a) Token distribution selected by FOV Overlap. (b) Token distribution selected by Dynamic Affinity.

- Fig. 4: Distribution comparison of different retrieval methods. The x-axis and y-axis represent the token index and DiT layers, respectively. The bubble size and color reflect the selection frequency of each token during the entire denoising process. (a) The FOV overlap method yields a fixed token selection. (b) Our dynamic affinity method exhibits a diverse retrieval distribution, enabling the perception of richer memory contexts.

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

Kernel Size (𝑇×𝐻×𝑊)

2×4×4

2×8×8

2×16×16

1×4×4

Frame 1 Frame 15 Frame 30 Frame 45 Frame 60

- Fig. 5: Qualitative comparison between different kernel sizes of the Memory Tokenizer. The red bounding boxes annotate the inconsistent region.

the model a broader memory receptive field and superior flexibility during the generation process.

Ablation on Kernel Size of Memory Tokenizer. We provide further qualitative ablation results regarding the kernel size of the Memory Tokenizer. As shown in Fig. 5, when the temporal dimension of the kernel size is set to 2, the generated results maintain spatiotemporal consistency due to effective temporal interaction. However, when the temporal kernel size is reduced to 1 (i.e., no temporal interaction during tokenization), noticeable inconsistencies emerge in the generated subjects. These qualitative observations further corroborate the quantitative ablation results presented in the main paper.

Ablation on Number of Retrieved Tokens. We qualitatively ablate the number of retrieved tokens. As depicted in Fig. 6, restricting the token length to 5 results in a substantial loss of context information, which misleads the model into generating severe artifacts (e.g., hallucinating two giraffes instead of one). In

Number of Retrieved Tokens

Frame 1 Frame 15 Frame 30 Frame 45 Frame 60

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

5

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

10

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

15

- Fig. 6: Qualitative comparison between the number of retrieved tokens. The red bounding boxes annotate the inconsistent region.

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

Fig. 7: Additional examples of HM-World dataset.

###### contrast, other settings with an adequate number of retrieved tokens successfully maintain subject consistency and physical plausibility.

#### C Additional Examples from the HM-World Dataset

To further illustrate the challenges present in the proposed HM-World dataset, we provide additional examples in Fig. 7.

