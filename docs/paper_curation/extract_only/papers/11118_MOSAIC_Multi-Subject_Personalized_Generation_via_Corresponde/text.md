# arXiv:2509.01977v1[cs.CV]2Sep2025

[Figure 1]

## MOSAIC: Multi-Subject Personalized Generation via Correspondence-Aware Alignment and Disentanglement

Dong She♠∗, Siming Fu♠∗†, Mushui Liu♡∗, Qiaoqiao Jin♠∗, Hualiang Wang♢♠∗, Mu Liu♠, Jidong Jiang( )

♠ByteDance Fanqie, ♡Zhejiang University, ♢The Hong Kong University of Science and Technology

∗Equal contributions, †Project lead, Corresponding authors

### Abstract

Multi-subject personalized generation presents unique challenges in maintaining identity fidelity and semantic coherence when synthesizing images conditioned on multiple reference subjects. Existing methods often suffer from identity blending and attribute leakage due to inadequate modeling of how different subjects should interact within shared representation spaces. We present MOSAIC, a representation-centric framework that rethinks multi-subject generation through explicit semantic correspondence and orthogonal feature disentanglement. Our key insight is that multi-subject generation requires precise semantic alignment at the representation level—knowing exactly which regions in the generated image should attend to which parts of each reference. To enable this, we introduce SemAlign-MS, a meticulously annotated dataset providing fine-grained semantic correspondences between multiple reference subjects and target images, previously unavailable in this domain. Building on this foundation, we propose the semantic correspondence attention loss to enforce precise point-to-point semantic alignment, ensuring high consistency from each reference to its designated regions. Furthermore, we develop the multi-reference disentanglement loss to push different subjects into orthogonal attention subspaces, preventing feature interference while preserving individual identity characteristics. Extensive experiments demonstrate that MOSAIC achieves SOTA performance on multiple benchmarks. Notably, while existing methods typically degrade beyond 3 subjects, MOSAIC maintains high fidelity with 4+ reference subjects, opening new possibilities for complex multi-subject synthesis applications.

Project Page: https://bytedance-fanqie-ai.github.io/MOSAIC

### 1 Introduction

Multi-subject personalized generation in controllable image synthesis faces significant challenges in maintaining identity consistency while preventing attribute entanglement. Recent approaches have explored various strategies to address this problem: MS-Diffusion [29] and SSR-Encoder [36] incorporate spatial layout guidance into cross-attention layers to bind subjects to dedicated regions, while DreamO [21] embeds routing constraints directly into DiT blocks for architectural-level control. Most recently, XVerse [2] introduces token-specific modulation offsets for independent subject representation control.

However, these methods share a critical limitation: they lack explicit optimization of the underlying diffusion

[Figure 2]

|[Figure 3]|
|---|

[Figure 4]

|[Figure 5]|
|---|

[Figure 6]

[Figure 7]

|[Figure 8]|
|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

A man and a woman are standing on either side of a canvas with the logo on it.

A man, a woman, and a boy sit behind a table with a cake on it, while balloons floats near the wall.

A woman is holding up a wooden board with the logo on it.

A man wearing a t-shirt with the logo on it.

[Figure 17]

|[Figure 18]|
|---|

[Figure 19]

[Figure 20]

|[Figure 21]|
|---|

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

[Figure 27]

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

[Figure 32]

|[Figure 33]|
|---|

|[Figure 34]|
|---|

|[Figure 35]|
|---|

| |
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

A boy sits next to a girl, gently holding a teddy bear as she lovingly strokes a cat.

A can, a cup, a tower and a toy car are placed on a wooden table.

Five cartoon characters stand together, creating a lively atmosphere.

A man wearing headphones and a T-shirt stands beside a woman wearing a cap and a dress.

- Figure 1 Our proposed MOSAIC demonstrates capabilities in single-subject and multi-subject driven generation tasks.

representations for both precise multi-subject alignment with target images and effective disentanglement between reference subjects. This shortcoming becomes increasingly severe as subject count grows, with most methods experiencing significant degradation beyond 3-4 subjects due to compounding feature interference. To understand why existing approaches fail, we analyze the fundamental requirements for multi-subject generation and identify two key deficiencies. The first concerns the correspondence problem: without explicit modeling of which specific reference image regions should attend to target latent parts, models cannot maintain semantic coherence across multiple subjects. The second involves multi-reference feature integration: when multiple subjects share the same latent space, their representations naturally interfere, yet existing methods provide no mechanism to explicitly disentangle these conflated features. These observations lead to a critical question: how can we design optimization objectives that simultaneously preserve individual subject fidelity while enforcing inter-subject separability between different subjects’ representations?

To address these fundamental limitations, we propose MOSAIC, a principled framework that reformulates multi-subject personalized generation as a representation optimization problem. The foundation of MOSAIC is the establishment of explicit semantic correspondences through a carefully curated dataset featuring densely annotated alignment points across reference-target image pairs. This correspondence foundation enables direct supervision of attention mechanisms for both reference-target alignment and inter-subject differentiation—a critical capability absent in existing approaches. Leveraging this semantic point correspondence, MOSAIC implements two complementary optimization objectives. The alignment objective employs cross-entropy supervision over attention distributions conditioned on correspondence labels, enforcing precise spatial mappings between reference and target representations. This mechanism ensures semantically coherent feature propagation from reference images to their designated target locations. Simultaneously, the disentanglement objective maximizes inter-subject attention divergence via symmetric KL regularization, promoting orthogonal attention patterns across reference subjects. This formulation effectively mitigates cross-subject feature interference—a persistent challenge in multi-reference scenarios. Notably, our method enables high-quality consistent generation with 4+ reference subjects, a capability that current approaches cannot achieve. Our contributions are as follows:

• We propose MOSAIC, a representation-centric framework that learns subject-consistent, disentangled representations by explicitly supervising attention correspondence between target and reference images, combining alignment and disentanglement objectives.

###### 1.Prompt Generation 2.Image Synthesis 3.Reference Extraction 4.Reference Generation 5.Semantic Point Correspondence

Single-Subject Generation

people

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

seg map

[Figure 46]

[Figure 47]

A woman in a hat stands on an empty platform beside a seated dog, with a worn suitcase at her feet and a violin in her hands, as they wait together in quiet urgency.

(multi-subjects)

animal

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Flux Kontext

[Figure 52]

[Figure 53]

object

[Figure 54]

[Figure 55]

[Figure 56]

- Figure 2 SemAlign-MS Dataset Construction Pipeline. Five-stage systematic pipeline for generating high-quality multi-reference training data with validated semantic correspondences.

- • We introduce SemAlign-MS, the first large-scale, comprehensively annotated dataset with fine-grained semantic correspondences specifically tailored for multi-subject driven generation.
- • We demonstrate significant improvements over existing methods, particularly in challenging scenarios with 4+ subjects, while maintaining computational efficiency through our plug-and-play design.

### 2 Related works

Subject-Driven Image Generation. Subject-driven image generation aims to synthesize images that preserve the identity and appearance of reference subjects while following textual descriptions. Recent advances have focused on improving reference encoding and attention mechanisms within diffusion frameworks. OminiControl [27] exploits the generative model itself as a reference image encoder, demonstrating the capability of diffusion transformers in maintaining subject consistency. For multi-subject scenarios, UNO [31] proposes a systematic data generation pipeline, while DreamO [20] constructs a router mechanism to focus attention on target subjects. XVerse [2] adopts text-stream modulation to transform reference images into token-specific offsets, enabling better integration of subject-specific cues. However, existing methods primarily rely on global feature matching, lacking explicit fine-grained detail constraints between reference and target regions. This limitation often results in imprecise spatial alignment, reduced reference fidelity, and attribute entanglement when handling multiple subjects simultaneously. Our work addresses these fundamental challenges through explicit semantic point correspondences that enable precise spatial alignment and cross-reference disentanglement, substantially enhancing subject consistency while preventing inter-subject interference.

Visual Correspondence for Generation. Visual correspondence establishes spatial relationships between semantically similar regions across images, serving as a foundation for various computer vision tasks [14, 15, 18]. Traditional methods rely on handcrafted features like SIFT [17] and SURF [1], while recent deep learning approaches leverage supervised learning with annotated datasets [6, 19, 32]. A promising direction has emerged using diffusion models for correspondence estimation. Methods such as DIFT [28], SD-DINO [33], and GeoAware-SC [34] demonstrate that pre-trained diffusion features can establish reliable correspondences across diverse images without extensive supervision. However, these correspondences have not been effectively utilized for multi-subject generation tasks. Our work bridges this gap by being the first to leverage semantic point correspondences for multi-subject-driven generation. We establish a systematic pipeline that constructs high-quality correspondences and explicitly incorporates them into the generation process through our semantic corresponding attention alignment and multi-reference disentanglement mechanisms.

### 3 SemAlign-MS: A High-Quality Multi-Subject Dataset with Semantic Point Correspondences

Our data construction follows a systematic five-stage pipeline designed to ensure both diversity and quality. We first leverage GPT-4o [9] with carefully designed templates to automatically generate diverse prompts containing multi-subject, encompassing various combinations of people, animals, objects, and their interactions to ensure comprehensive coverage of multi-subject scenarios commonly encountered in real-world applications.

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

A dog, a human and a woman standing in a room

Text Target Ref1 Ref2 RefK

Ref-to-Target Attention

Semantic Point Lable

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

1.0

[Figure 68]

[Figure 69]

0.0

1

K

tgt

1.0

[Figure 70]

[Figure 71]

[Figure 72]

Align

[Figure 73]

tgt

[Figure 74]

[Figure 75]

0.0

1

[Figure 76]

1.0

K

[Figure 77]

[Figure 78]

0.0

[Figure 79]

[Figure 80]

ScoreDistribution

[Figure 81]

[Figure 82]

[Figure 83]

ScoreDribution

[Figure 84]

Attention

Attention

Disentangle

[Figure 85]

[Figure 86]

tokens of

tokens of

- Figure 3 Overview of MOSAIC Framework. MOSAIC introduces two key supervisions: (1) Semantic Correspondence Attention Loss (blue region) enforces precise point-to-point alignment between reference tokens and their corresponding locations in the target latent, ensuring high consistency; (2) Multi-Reference Disentanglement Loss (green region) maximizes the divergence between different references’ attention distributions, pushing each subject into orthogonal representational subspaces.

The generated prompts are then processed through state-of-the-art T2I models to synthesize images, where we implement a multi-criteria automated filtering strategy that evaluates image quality, subject clarity, and compositional coherence to retain only the highest-quality synthetic images. Subsequently, we employ LangSAM [11] for robust open-vocabulary detection and segmentation of all subjects within the synthesized images, enabling precise identification and isolation of individual subjects regardless of their semantic category and providing the foundation for subsequent correspondence establishment. Finally, we utilize FLUX Kontext [13] for viewpoint correction while maintaining semantic consistency, significantly enhancing dataset diversity and ensuring comprehensive appearance coverage across different viewpoints and poses.

Once we obtain the large-scale data, we establish semantic point correspondences between each target image and multiple reference images, where each reference image contributes P(k) sampled semantic points from the reference space that are mapped to corresponding locations in the target latent space, and P(k) ∈ N is the num-

ber of semantic correspondence points for the k-th reference image. Formally, let D = {({Iref(i,k)}Kk=1,Itgt(i))}Ni=1 denote our dataset, where N is the total number of target images in the training batch, M is the maximum

number of reference images, Itgt(i) represents the i-th target image, and Iref(i,k) denotes the k-th reference image for the i-th target. When fewer than K reference images are available, we pad the remaining slots with

black placeholder images to maintain consistent batch dimensions. For each pair (Iref(i,k),Itgt(i)), we define the semantic correspondence set as:

(k)

C(i,k) = {(ui,j,vi,j)}P

j=1 (1)

where ui,j represents the j-th sampled point coordinate in reference image Iref(k), and vi,j denotes its corresponding semantic point in the target latent space. To prevent inter-reference conflicts where references map

to the same target token, we enforce correspondence disjointness across different reference images:

V(i,k) ≜ {vi,j | (ui,j,vi,j) ∈ C(i,k)}, s.t. V(i,k

(2)

1) ∩ V(i,k

2) = ∅, ∀k1 ̸= k2,

where V(i,k) represents the token position corresponding to the j-th sampled token of k-th reference image for the i-th training sample. This constraint ensures that each target latent location vi,j is associated with at most one reference image, preventing ambiguous supervision where multiple references compete for the same target region. Through this systematic pipeline, we successfully collect 1.2M high-quality image pairs with validated semantic correspondences that form the foundation of SemAlign-MS dataset.

### 4 Methodology

#### 4.1 Overview of MOSAIC

Architecture. As shown in Fig. 3, given K reference images {I(refk)}Kk=1 depicting different subjects, a text prompt T, and a target image Itgt to be generated. During training, noise is applied to the target image to obtain Itgt = Itgt + ϵ, where ϵ ∼ N(0,1). The VAE encoder then transforms both the noisy image and all reference images into latent representations:

ztgt = VAEenc(Itgt), z(refk) = VAEenc(I(refk)) (3)

where k = 1,...,K. The text prompt is encoded using T5 [24] to obtain text embeddings Ct = T5(T). To ensure spatial disentanglement between reference and target latents, modified Rotary Position Embeddings (RoPE) [26] with distinct frequency bases are applied.

Multi-Reference within MM-Attention. The core of MOSAIC lies in how multiple reference latent and target latent interact through attention. Following OmniControl [27], we employ a LoRA-augmented branch for reference processing while maintaining the original model weights for the denoising branch. Crucially, we concatenate all reference latents into a unified representation to enable joint processing. In each transformer block l, the attention computation proceeds as:

Qtgt(l) ,K(tgtl) ,Vtgt(l) = fθ(z(tgtl) ), Q(textl) ,K(textl) ,Vtext(l) = fϕ(Ct),

(4)

where fθ and fϕ mean the mm-attention projection weights for target latent and text embedding. For the reference images, we first concatenate their latent representations:

z(refl) = [zref(1,l);z(2ref,l);...;z(refK,l)]. (5) Then process them through the LoRA branch of fθ, which denoted as ∆θLoRA:

(z(refl)). (6) These features are then concatenated across modalities and processed through multi-head attention:

Q(refl),K(refl),Vref(l) = fθ+∆θ

LoRA

Q(l) = [Q(tgtl) ;Q(textl) ;Q(refl)], K(l) = [K(tgtl) ;K(textl) ;K(refl)], V(l) = [Vtgt(l) ;Vtext(l) ;Vref(l)].

(7)

√

The attention scores are computed as A(l) = softmax(Q(l)(K(l))T/

d), where d is the feature dimension. Note that within A(l), the reference-to-target latent attention sub-matrix A(refl)→tgt ∈ RN

ref×Ntgt captures how each token from all reference images attends to the noisy latent, where Nref is the total number of tokens length across all reference images and Ntgt is the token length of the target latent.

#### 4.2 Semantic Corresponding Attention Alignment

To faithfully preserve fine-grained semantic details, especially in regions requiring precise structural correspondence, we design the semantic correspondence attention loss (SCAL) to explicitly enforce point-wise semantic alignment within the reference-to-target attention mechanism.

As shown in Fig. 3, the reference-to-target latent attention allows each token from the reference images to attend to all token positions of the noised target latent. For a reference token at position u and a target latent position v, the average reference to target latent attention across all DiT blocks Nblock is computed as:

√

exp QuKv⊤/

d

Nblock

1 Nblock

(8)

Aref→tgt[u,v] =

√

Ntgt v=1 exp QuKv⊤/

d

l=1

where Qu and Kv are the query and key embeddings for the reference token at position u and latent position v, respectively. To map local reference coordinates to global token indices in concatenated reference representation, we define:

k−1

G(u(i,jk)) =

+u(i,jk) (9)

N(idx)

idx=1

offset

where idx denotes the reference subject index. For each correspondence pair (u(i,jk),vi,j(k)), to supervise the attention from the reference token at position u(i,jk) that focused on its corresponding latent position vi,j(k), we define:

P(k)

K

1 K

1 P(k)

log Aref→tgt[G(u(i,jk)),vi,j(k)] (10)

LSCA = −

j=1

k=1

By integrating LSCA into the training objective, our model is effectively encouraged to learn precise semantic mappings between reference and generated images. This leads to significantly improved preservation of local structure, textures, and fine details, going beyond the limitations of global similarity or implicit feature alignment.

#### 4.3 Multi-Reference Corresponding Disentanglement

While alignment ensures high consistency, a crucial aspect of multi-subject driven generation is the potential interference among different reference images. We introduce multi-reference disentanglement loss (MDL), which promotes distinct attention patterns across references. MDL emphasizes the differentiation of attention maps from various subjects, thus preventing feature conflation.

Specifically, for k-th reference image, we collect the attention patterns at correspondence locations. For each correspondence point (uj,vj) ∈ C(i,k):

a(jk) = [Aref→tgt[uj,t] | t ∈ Ntgt] ∈ RN

(11)

tgt

Subsequently, we aggregate these attention responses for each reference:

1 P(k)

a(k) = ||

P(k)

a(jk) ∈ RN

tgt

j=1

|| (12)

where || · || denotes the normalization operation. Then the distance between a(i) and a(j) is:

- 1

- 2DKL(aˆ(i)||aˆ(j)) +

- 1

- 2DKL(aˆ(j)||aˆ(i)) (13)

dist(a(i),a(j)) =

where DKL is KL divergence. Subsequently, we enforce distinct attention patterns across references:

K

K

1 K(K − 1)

dist(a(i),a(j)) (14)

LMD = −

i=1

j=1,j̸=i

By maximizing attention pattern divergence, this loss prevents references from competing for the same attention regions, mitigating cross-reference feature interference. The overall loss is defined as:

L = Ldiff + αLSCA + βLMD (15) where Ldiff is the flow-matching loss used in [5], α and β are two factors that balance the weight.

Reference Method CLIP-I ↑ CLIP-T ↑ DINO ↑

|Single-Subject<br><br>DreamBooth BLIP-Diffusion SSR-Encoder MS-Diffusion UNO DreamO XVerse<br><br>MOSAIC|80.30 30.52 66.81 80.47 30.24 69.82<br><br>82.10 30.79 61.22 80.82 31.05 70.32<br><br>83.50 30.41 75.97 83.35 30.61 76.03<br><br><br>83.20 30.20 75.44<br>84.30 31.64 77.40<br><br><br>|
|---|---|
|Multi-Subject<br><br>MS-Diffusion UNO DreamO XVerse<br><br>MOSAIC|72.60 31.91 52.50<br>73.29 32.23 54.22 73.32 32.10 52.17 73.47 31.20 53.71 76.30 32.40 56.83<br><br><br>|

Single-SubjectMulti-Subject

- Table 1 Quantitative comparison for single-subject and multi-subject on DreamBench benchmark.

Method

Single-Subject Multi-Subject

Overall

DPG ID-Sim IP-Sim AES AVG DPG ID-Sim IP-Sim AES AVG

|MS-Diffusion 96.89 6.52 55.71 59.63 54.69 UNO 89.65 47.91 80.40 55.90 68.47 DreamO 96.93 75.48 70.84 54.57 74.46 OmniGen2 92.60 62.41 74.08 52.34 70.36 XVerse 93.69 79.48 76.86 56.84 76.72<br><br>|87.21 3.77 46.21 55.91 48.28 85.28 31.82 67.00 54.24 59.59<br>88.80 50.24 64.63 52.47 64.04 91.55 40.81 67.15 51.40 62.73 88.26 66.59 71.48 53.97 70.08<br><br><br>|51.49 64.03 69.25 66.55 73.40|
|---|---|---|
|MOSAIC 96.55 81.98 80.92 60.77 80.05|88.94 69.90 74.27 55.02 72.03<br><br>|76.04|

- Table 2 Quantitative results of single-subject and multi-subject driven generation on XVerseBench.

### 5 Experiments

#### 5.1 Experimental Details

- 5.1.1 Implementation Details

In this work, we adopt FLUX-1.0-DEV [12] as our base model. The rank of additional LoRA [7] is set as 128. During training, we employ the AdamW [10] optimizer with a learning rate of 1e-4 and train the model for 100K steps, using a batch size of 1 per GPU. The α and β in Eq. 15 are set to 0.4 and 0.6.

- 5.1.2 Evaluation Setting

We evaluate our proposed method on DreamBench [25] and XVerseBench [2], focusing on both single-subject and multi-subject generation scenarios. For comparison, we include DreamBooth [25], BLIP-Diffusion [16], SSR-Encoder [35], MS-Diffusion [29], UNO [31], DreamO [20], OmniGen2 [30], and XVerse [2]. DreamBench utilizes the CLIP-I [23], DINO [22], and CLIP-T [23] to evaluate the semantic and text-image alignment. XVerseBench utilizes DPG score [8], ID-Sim [3], IP-Sim [22], AES [4] to further evaluate the alignment and image quality.

#### 5.2 Main Results

##### 5.2.1 Quantitative Results

The results on DreamBench and XVerseBench are shown in Tab.1 and Tab.2, respectively. On DreamBench, MOSAIC achieves consistently strong results across all metrics. In the single-subject setting, it achieves 84.30 (CLIP-I), 31.64 (CLIP-T), and 77.40 (DINO), surpassing the second-best method UNO by 0.80 on CLIP-I, and the second-best method DreamO by 1.37 on DINO, indicating better image-text alignment and

Reference Prompt

MOSAIC(Ours) MS-Diffusion UNO DreamO XVerse OmniGen2

|[Figure 87]|
|---|

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

A snacker is placed on the ground.

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

|[Figure 100]|
|---|

|[Figure 101]|
|---|

A single sneaker rests next to a beer can.

[Figure 102]

| |
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

A Siamese cat is stretched out comfortably beside a leather handbag and a clock.

|[Figure 110]|
|---|

|[Figure 111]|
|---|

| |
|---|

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

A boy is using a camera to take a picture of a toucan in a park.

|[Figure 119]|
|---|

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

|[Figure 126]|
|---|

|[Figure 127]|
|---|

A woman plays a guitar while sitting on the grass, with her hat and backpack resting quietly at her side.

|[Figure 128]|
|---|

|[Figure 129]|
|---|

|[Figure 130]|
|---|

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

|[Figure 136]|
|---|

|[Figure 137]| | | | |
|---|---|---|---|---|
|[Figure 138]| | |[Figure 139]| |

A man and a woman stand in a room with a vase, a bowl, and an ornament on the table.

|[Figure 140]|
|---|

- Figure 4 Qualitative comparison on single and multi-subject driven generation. DreamO does not support 3+ reference images; masked regions show where comparison is unavailable, with visible results generated from two randomly selected references.

semantic fidelity. In the more challenging multi-subject setting, it maintains strong performance with 76.30 (CLIP-I), 32.40 (CLIP-T), and 56.83 (DINO), outperforming second-best competitors by approximately 3.0, 0.2, and 2.6 points, respectively. On XVerseBench, MOSAIC achieves the highest overall average score of 76.04, surpassing XVerse’s 73.40. And MOSAIC achieves a DPG score that is on par with the top-performing DreamO (96.93). It also demonstrates clear advantages in identity preservation, with ID-Sim scores of 81.98 for single-object and 69.90 for multi-object scenarios, as well as in perceptual similarity, with IP-Sim scores of 80.92 and 74.27, respectively. Overall, these demonstrate the robustness of our proposed method under varying subject complexity.

##### 5.2.2 Qualitative Results

Fig. 4 presents qualitative results across scenes with varying numbers of reference objects, demonstrating MOSAIC’s superior performance in multi-subject generation. Appearance consistency. Our method maintains strong visual coherence across different subjects. For instance, the snackers in rows 1-2 and the can in row 2 preserve consistent textures and shapes throughout the generated scenes, indicating effective feature transfer from reference images. Multi-subject handling. The advantages of our approach become particularly evident with three or more reference subjects. In row 3, competing methods suffer from object omission and duplication: MS-Diffusion and OmniGen2 fail to render the clock entirely, while XVerse produces duplicated and deformed cats. In contrast, MOSAIC accurately represents all three objects without such artifacts. Scalability to more complex scenes. For scenes involving four or more reference objects, existing methods exhibit substantial degradation. As shown in row 6, only MOSAIC preserves facial consistency across multiple subjects, while other approaches show significant quality loss and identity confusion. Overall, these results highlight MOSAIC’s ability to generate high-quality, consistent outputs across diverse compositional scenarios,

###### LSCA LMD CLIP-I CLIP-T DINO ✗ ✗ 73.45 29.90 52.03 ✓ ✗ 75.89 31.10 55.99 ✓ ✓ 76.30 32.40 56.83

###### Table 3 Impacts of different losses on DreamBench benchmark in multi-subject scenario.

Reference Prompt w/o 𝐿 & 𝐿 w/o 𝐿 MOSAIC

[Figure 141]

[Figure 142]

[Figure 143]

arequietlyplacedona

withacatonthefloor Ateddybearandatoy

[Figure 144]

| |
|---|

[Figure 145]

parkbench

[Figure 146]

Amanwearingat-shirt

issittingonthesofa,

[Figure 147]

[Figure 148]

| |
|---|

[Figure 149]

[Figure 150]

[Figure 151]

- Figure 5 Ablation study of MOSAIC. Zoom in for details.

[Figure 152]

| |
|---|

[Figure 153]

[Figure 154]

| |
|---|

A teddy bear and a toy are quietly placed on a park bench

Prompt w/o 𝐿 MOSAIC

[Figure 155]

[Figure 156]

[Figure 157]

| |
|---|
| |

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

| |
|---|
| |

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

w/o 𝐿 & 𝐿

Figure 6 Visualization of attention maps from specific reference regions to generated image regions.

especially in challenging multi-subject settings where other methods fall short.

5.3 Ablation Studies

- 5.3.1 Impact of Different Losses

We conduct ablation studies to evaluate each component of our method on multi-subjects of Dreambench, as shown in Tab.3. Adding the semantic correspondence attention loss LSCA leads to consistent gains across all metrics, with CLIP-I improving from 73.45 to 75.89, CLIP-T from 29.90 to 31.10, and DINO from 52.03 to 55.99. This confirms that explicit point-wise semantic supervision aids fine-grained detail retention. Moreover, introducing LMD further improves CLIP-I to 76.30, CLIP-T to 32.40, and DINO to 56.83, suggesting that suppressing cross-reference feature conflation strengthens semantic fidelity and compositional coherence. Moreover, visual results in Fig. 5 improve progressively: the baseline yields blurry, inconsistent outputs, while adding LSCA enhances composition and detail. The full MOSAIC model achieves both global scene coherence and local identity preservation by preventing semantic interference across references.

- 5.3.2 Visualization of attention maps

As shown in Fig. 6, progressive improvement in attention alignment and disentanglement through our proposed losses. We visualize attention maps from specific reference regions (teddy bear’s goggles and Minion’s "001" text) to demonstrate the effectiveness of our semantic corresponding attention loss (LSCA) and multi-reference disentanglement loss (Lrd). From left to right: baseline method shows scattered, unfocused attention with cross-reference interference; LSCA improves semantic alignment but attention remains diffuse across multiple regions; LSCA + Lrd achieves both precise attention alignment to semantically corresponding regions and effective disentanglement between different reference subjects. Blue and green lines connect reference regions to their attention peaks in the target latent space.

- 6 Conclusion

In this paper, we propose MOSAIC, a representation-centric approach for multi-subject driven image generation. Our method features a semantic correspondence pipeline, along with explicit attention alignment and disentanglement mechanisms, effectively addressing the key challenges of identity preservation and attribute entanglement in multi-subject personalized generation scenarios. We also curate and will release SemAlign-MS, a large-scale multi-subject dataset with fine-grained semantic point correspondences to facilitate future research in controllable generation. Extensive experiments demonstrate that MOSAIC achieves state-of-the-art performance in both identity fidelity and semantic consistency across established benchmarks. Notably, our approach exhibits robust scalability, successfully handling complex compositions with 4+ reference subjects.

### References

- [1] Herbert Bay, Tinne Tuytelaars, and Luc Van Gool. Surf: Speeded up robust features. In ECCV, pages 404–417, 2006.

- [2] Bowen Chen, Mengyi Zhao, Haomiao Sun, Li Chen, Xu Wang, Kang Du, and Xinglong Wu. Xverse: Consistent multi-subject control of identity and semantic attributes via dit modulation. arXiv preprint arXiv:2506.21416, 2025.

- [3] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, pages 4690–4699, 2019.

- [4] discus0434. Aesthetic predictor v2.5: Siglip-based aesthetic score predictor. https://github.com/discus0434/ aesthetic-predictor-v2-5, 2024. Accessed: 2024-12-08.
- [5] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

- [6] Bumsub Ham, Minsu Cho, Cordelia Schmid, and Jean Ponce. Proposal flow: Semantic correspondences from object proposals. IEEE TPMAI, 40:1711–1725, 2017.

- [7] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In ICLR, 2022.

- [8] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

- [9] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

- [10] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

- [11] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, pages 4015–4026, 2023.

- [12] Black Forest Labs. Flux-dev-1.0. https://github.com/black-forest-labs/flux, 2024.
- [13] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

- [14] Junghyup Lee, Dohyung Kim, Jean Ponce, and Bumsub Ham. Sfnet: Learning object-aware semantic correspondence, 2019. URL https://arxiv.org/abs/1904.01810.
- [15] Junsoo Lee, Eungyeup Kim, Yunsung Lee, Dongjun Kim, Jaehyuk Chang, and Jaegul Choo. Reference-based sketch image colorization using augmented-self reference and dense semantic correspondence, 2020. URL https: //arxiv.org/abs/2005.05207.
- [16] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. In NeurIPS, pages 30146–30166, 2023.

- [17] David G Lowe. Distinctive image features from scale-invariant keypoints. IJCV, 60:91–110, 2004.

- [18] Jingyi Lu, Xinghui Li, and Kai Han. Regiondrag: Fast region-based image editing with diffusion models, 2024. URL https://arxiv.org/abs/2407.18247.
- [19] Juhong Min, Jongmin Lee, Jean Ponce, and Minsu Cho. Hyperpixel flow: Semantic correspondence with multilayer neural features. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3395–3404, 2019.

- [20] Chong Mou, Yanze Wu, Wenxu Wu, Zinan Guo, Pengze Zhang, Yufeng Cheng, Yiming Luo, Fei Ding, Shiwen Zhang, Xinghui Li, et al. Dreamo: A unified framework for image customization. arXiv preprint arXiv:2504.16915, 2025.

- [21] Chong Mou, Yanze Wu, Wenxu Wu, Zinan Guo, Pengze Zhang, Yufeng Cheng, Yiming Luo, Fei Ding, Shiwen Zhang, Xinghui Li, et al. Dreamo: A unified framework for image customization. arXiv preprint arXiv:2504.16915, 2025.

- [22] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

- [23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.

- [24] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer, 2023. URL https://arxiv.org/abs/1910.10683.
- [25] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pages 22500–22510, 2023.

- [26] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

- [27] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. In ICCV, 2025.

- [28] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion, 2023. URL https://arxiv.org/abs/2306.03881.
- [29] Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. In ICLR, 2025.

- [30] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.

- [31] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. In ICCV, 2025.

- [32] Hang Yu, Yufei Xu, Jing Zhang, Wei Zhao, Ziyu Guan, and Dacheng Tao. Ap-10k: A benchmark for animal pose estimation in the wild. arXiv preprint arXiv:2108.12617, 2021.

- [33] Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, and MingHsuan Yang. A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence,

2023. URL https://arxiv.org/abs/2305.15347.

- [34] Junyi Zhang, Charles Herrmann, Junhwa Hur, Eric Chen, Varun Jampani, Deqing Sun, and Ming-Hsuan Yang. Telling left from right: Identifying geometry-aware semantic correspondence, 2024. URL https://arxiv.org/ abs/2311.17034.
- [35] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In CVPR, pages 8069–8078, 2024.

- [36] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In CVPR, pages 8069–8078, 2024.

