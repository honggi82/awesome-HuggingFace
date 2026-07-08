### Text-Guided 3D Face Synthesis - From Generation to Editing

Yunjie Wu†1 Yapeng Meng†1,2 Zhipeng Hu†1 Lincheng Li∗1 Haoqian Wu1 Kun Zhou3 Weiwei Xu3 Xin Yu4 1Netease Fuxi AI Lab 2Tsinghua University 3Zhejiang University 4University of Queensland

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

# arXiv:2312.00375v1[cs.CV]1Dec2023

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

“Let him wear a purple Zorro mask”

“Mark Zuckerberg” Relighting

Animation

“Make his eyes big”

“Make his lips black”

[Figure 13]

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

[Figure 24]

Animation Relighting

“Elon Musk” “Turn him Asian” “Make him bald”

“Give him mustache”

(a) High-fidelity generation and flexible editing (b) Semlessly integrated to existing CG pipeline

Figure 1. (a) Our approach enables the high-fidelity generation and flexible editing of 3D faces from textual input. It facilitates sequential editing for creating customized details in 3D faces. (b) The produced 3D faces can be seamlessly integrated into existing CG pipelines.

#### Abstract

#### 1. Introduction

Text-guided 3D face synthesis has achieved remarkable results by leveraging text-to-image (T2I) diffusion models. However, most existing works focus solely on the direct generation, ignoring the editing, restricting them from synthesizing customized 3D faces through iterative adjustments. In this paper, we propose a unified text-guided framework from face generation to editing. In the generation stage, we propose a geometry-texture decoupled generation to mitigate the loss of geometric details caused by coupling. Besides, decoupling enables us to utilize the generated geometry as a condition for texture generation, yielding highly geometry-texture aligned results. We further employ a finetuned texture diffusion model to enhance texture quality in both RGB and YUV space. In the editing stage, we first employ a pre-trained diffusion model to update facial geometry or texture based on the texts. To enable sequential editing, we introduce a UV domain consistency preservation regularization, preventing unintentional changes to irrelevant facial attributes. Besides, we propose a self-guided consistency weight strategy to improve editing efficacy while preserving consistency. Through comprehensive experiments, we showcase our method’s superiority in face synthesis. Project page: https://faceg2e.github.io/.

Modeling 3D faces serves as a fundamental pillar for various emerging applications such as film making, video games, and AR/VR. Traditionally, the creation of detailed and intricate 3D human faces requires extensive time from highly skilled artists. With the development of deep learning, existing works [8, 10, 47, 56] attempted to produce 3D faces from photos or videos with generative models. However, the diversity of the generation remains constrained primarily due to the limited scale of training data. Fortunately, recent large-scale vision-language models (e.g., CLIP [33], Stable Diffusion [35]) pave the way for generating diverse 3D content. Through the integration of these models, numerous text-to-3D works [23, 28, 29, 50, 52] can create 3D content in a zero-shot manner.

Many studies have been conducted on text-to-3D face synthesis. They either utilize CLIP or employ score distillation sampling (SDS) on text-to-image (T2I) models to guide the 3D face synthesis. Some methods [46, 53] employ neural fields to generate visually appealing but low-quality geometric 3D faces. Recently, Dreamface [54] has demonstrated the potential for generating high-quality 3D face textures by leveraging SDS on facial textures, but their geometry is not fidelitous enough and they overlooked the subsequent face editing. A few works [2, 12, 27] enable textguided face editing, allowing coarse-grained editing (e.g.

†Equal contribution

*Corresponding author

overall style), but not fine-grained adjustments (e.g., lips color). Besides, the lack of design in precise editing control leads to unintended changes in their editing, preventing the synthesis of customized faces through sequential editing.

To address the aforementioned challenges, we present text-guided 3D face synthesis - from generation to editing, dubbed FaceG2E. We propose a progressive framework to generate the facial geometry and textures, and then perform accurate face editing sequentially controlled by text. To the best of our knowledge, this is the first attempt to edit a 3D face in a sequential manner. We propose two core components: (1) Geometry-texture decoupled generation and (2) Self-guided consistency preserved editing.

To be specific, our proposed Geometry-texture decoupled generation generates the facial geometry and texture in two separate phases. By incorporating texture-less rendering in conjunction with SDS, we induce the T2I model to provide geometric-related priors, inciting details (e.g., wrinkles, lip shape) in the generated geometry. Building upon the generated geometry, we leverage ControlNet to force the SDS to be aware of the geometry, ensuring precise geometry-texture alignment. Additionally, we fine-tune a texture diffusion model that incorporates both RGB and YUV color spaces to compute SDS in the texture domain, enhancing the quality of the generated textures.

The newly developed Self-guided consistency preserved editing enables one to follow the texts, performing efficient editing in specific facial attributes without causing other unintended changes. Here, we first employ a pre-trained image-edit diffusion model to update the facial geometry or texture. Then we introduce a UV domain consistency preservation regularization to prevent unexpected changes in faces, enabling sequential editing. To avoid the degradation of editing effects caused by the regularization, we further propose a self-guided consistency weighting strategy. It adaptively determines the regularization weight for each facial region by projecting the cross-attention scores of the T2I model to the UV domain. As shown in Fig. 1, our method can generate high-fidelity 3D facial geometry and textures while allowing fine-grained face editing. With the proposed components, we achieve better visual and quantitative results compared to other SOTA methods, as demonstrated in Sec. 4. In summary, our contributions are:

- • We propose FaceG2E, facilitating a full pipeline of textguided 3D face synthesis, from generation to editing. User surveys confirm that our synthesized 3D faces are significantly preferable than other SOTA methods.
- • We propose the geometry-texture decoupled generation, producing faces with high-fidelity geometry and texture.
- • We design the self-guided consistency preservation, enabling the accurate editing of 3D faces. Leveraging precise editing control, our method showcases some novel editing applications, such as sequential and geometry-

texture separate editing.

#### 2. Related Work

Text-to-Image generation. Recent advancements in visual-language models [33] and diffusion models [9, 14, 43] have greatly improved text-to-image generation [4, 34, 35, 38]. These methods, trained on large-scale image-text datasets [41, 42], can synthesize realistic and complex images from text descriptions. Subsequent studies have made further efforts to introduce additional generation process controls [17, 49, 55], fine-tuning the pre-trained models for specific scenarios [11, 16, 36], and enabling image editing capabilities [6, 13, 24]. However, generating high-quality and faithful 3D assets, such as 3D human faces, from textual input still poses an open and challenging problem.

Text-to-3D generation. With the success of text-to-image generation in recent years, text-to-3D generation has attracted significant attention from the community. Early approaches [15, 21, 31, 39, 51] utilize mesh or implicit neural fields to represent 3D content, and optimized the CLIP metrics between the 2D rendering and text prompts. However, the quality of generated 3D contents is relatively low.

Recently, DreamFusion [32] has achieved impressive results by using a score distillation sampling (SDS) within the powerful text-to-image diffusion model [38]. Subsequent works further enhance DreamFusion by reducing generation time [28], improving surface material representation [7], and introducing refined sampling strategies [19]. However, the text-guided generation of high-fidelity and intricate 3D faces remains challenging. Building upon DreamFusion, we carefully design the form of score distillation by exploiting various diffusion models at each stage, resulting in high-fidelity and editable 3D faces.

Text-to-3D face synthesis. Recently, there have been attempts to generate 3D faces from text. Describe3D [48] and Rodin [46] propose to learn the mapping from text to 3D faces on pairs of text-face data. They solely employ the mapping network trained on appearance descriptions to generate faces, and thus fail to generalize to out-of-domain texts (e.g., celebrities or characters). On the contrary, our method can generalize well to these texts and synthesize various 3D faces.

Other works [12, 18, 22, 27, 54] employ SDS on the pretrained T2I models. Dreamface [54] utilizes CLIP to select facial geometry from candidates. Then they perform the SDS with a texture diffusion network to generate facial textures. Headsculpt [12] employs Stable Diffusion [35] and InstructPix2Pix [6] for computing the SDS, and relies on the mixture of SDS gradients for constraining the editing process. These approaches can perform not only generation but also simple editing. However, they still lack the design in precise editing control, and unintended changes in the editing results often occur. This prevents them from synthe-

###### "a DSLR photo of Scarlett Johansson"

###### "let her wear a batman eyemask"

[Figure 25]

condition

a (1) a (2) b

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

𝑰

shared view

SDS

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Instruct pix2pix

3DMM-based mesh texture 𝒅

geometry 𝒈

𝒅𝒐 𝒅𝒆

generated face edited face

𝑰′

[Figure 36]

[Figure 37]

norm(att(𝑰 , "mask" ))

UV unwrap

UV unwrap

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Self-guided Consistency Regularization

⊙

=

## ,

normal grey 𝑰

depth

shared view

𝒑𝒐 𝒑𝒆

𝑪

𝒅𝒐 𝒅𝒆

(b) Self-guided Consistency Preserved Editing

SDS

SDS stable diffusion

SDS

[Figure 47]

[Figure 48]

[Figure 49]

depth controlnet

Tex diffusion

[Figure 50]

update

noise

condition

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

SD/ControlNet/ InsP2P/ TexDiffusion

[Figure 55]

[Figure 56]

"facial texture"

(1) Geometry Phase

(2) Texture Phase (a) Geometry-Texture Decoupled Generation

rendered

condition (optional)

𝐿

Description/Instruction/Keyword

(c) Score Distillation Sampling (SDS)

Figure 2. Overview of FaceG2E. (a) Geometry-texture decoupled generation, including a geometry phase and a texture phase. (b) Selfguided consistency preserved editing, in which we utilize the built-in cross-attention to obtain the editing-relevant regions and unwrap them to UV space. Then we penalize inconsistencies in the irrelevant regions. (c) Our method exploits multiple score distillation sampling.

sizing highly customized 3D faces via sequential editing. On the contrary, our approach facilitates accurate editing of 3D faces, supporting sequential editing.

embedding of input text.

Facial Geometry and Texture is represented with parameters θ = (β,u) in FaceG2E. β denotes the identity coefficient from the parametric 3D face model HIFI3D [5], and u denotes a image latent code for facial texture. The geometry g can be achieved by the blendshape function M(·):

#### 3. Methodology

FaceG2E is a progressive text-to-3D approach that first generates a high-fidelity 3D human face and then performs finegrained face editing. As illustrated in Fig. 2, our method has two main stages: (a) Geometry-texture decoupled generation, and (b) Self-guided consistency preserved editing. In Sec. 3.1, we introduce some preliminaries that form the fundamental basis of our approach. In Sec. 3.2 and Sec.

βiSi, (2)

g = M(β) = T +

i

where T is the mean face and S is the vertices offset basis. As to the texture, the facial texture map d is synthesized with a decoder: d = D(u). We take the decoder from VAE of Stable Diffusion [35] as D(·).

- 3.3, we present the generation and editing stages.

###### 3.2. Geometry-Texture Decoupled Generation

###### 3.1. Preliminaries

The first stage of FaceG2E is the geometry-texture decoupled generation, which generates facial geometry and texture from the textual input. Many existing works have attempted to generate geometry and texture simultaneously in a single optimization process, while we instead decouple the generation into two distinct phases: the geometry phase and the texture phase. The decoupling provides two advantages: 1) It helps enhance geometric details in the generated faces. 2) It improves geometry-texture alignment by exploiting the generated geometry to guide the texture generation.

Score distillation sampling has been proposed in DreamFusion [32] for text-to-3D generation. It utilizes a pretrained 2D diffusion model ϕ with a denoising function ϵϕ (zt;y,t) to optimize 3D parameters θ. SDS renders an image I = R(θ) and embeds I with an encoder E(·), achieving image latent z. Then it injects a noise ϵ into z, resulting in a noisy latent code zt. It takes the difference between the predicted and added noise as the gradient:

∂z ∂I

∂I ∂θ

, (1)

∇θLSDS(I) = Et,ϵ w(t)(ϵϕ (zt;y,t) − ϵ)

Geometry Phase. An ideal generated geometry should possess both high quality (e.g., no surface distortions) and a

where w(t) is a time-dependent weight function and y is the

good alignment with the input text. The employed facial 3D morphable model provides strong priors to ensure the quality of generated geometry. As to the alignment with the input text, we utilize SDS on the network ϕsd of Stable Diffusion [35] to guide the geometry generation.

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

noise

texutre diffusion1

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Face texture datasets

a face UV texture

[Figure 66]

Previous works [22, 27, 53] optimize geometry and texture simultaneously. We observe this could lead to the loss of geometric details, as certain geometric information may be encapsulated within the texture representation. Therefore, we aim to enhance the SDS to provide more geometry-centric information in the geometry phase. To this end, we render the geometry g with texture-less rendering I˜ = R˜(g), e.g., surface normal shading or diffuse shading with constant grey color. The texture-less shading attributes all image details solely to geometry, thereby allowing the SDS to focus on geometry-centric information. The geometry-centric SDS loss is defined as:

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

noise

texutre diffusion2

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

YUV color space

Figure 3. Training the texture diffusion model is performed on the collected facial textures in both RGB and YUV color space.

prior SDS is formulated with the trained ϕtd1 and ϕtd2 as: ∇uLprtex = Lrgbtex + λyuvLyuvtex , Lrgbtex = Et,ϵ w(t) ϵϕ

∂ztd ∂d

∂d ∂u

ztd;y∗,t − ϵ

,

td1

∂I˜ ∂g

∂g ∂β

∂zt ∂I˜

′

. (3)

∇βLgeo=Et,ϵ w(t)(ϵϕ

∂zd

(zt;y,t) − ϵ)

∂d ∂u

′

Lyuvtex = Et,ϵ w(t) ϵϕ

t ;y∗,t − ϵ

t

zd

sd

,

td2

∂d

Texture Phase. Many works [27, 54] demonstrate that texture can be generated by minimizing the SDS loss. However, directly optimizing the standard SDS loss could lead to geometry-texture misalignment issues, as shown in Fig .9. To address this problem, we propose the geometry-aware texture content SDS (GaSDS). We resort to the ControlNet [55] to endow the SDS with awareness of generated geometry, thereby inducing it to uphold geometry-texture alignment. Specifically, we render g into a depth map e. Then we equip the SDS with the depth-ControlNet ϕdc, and take e as a condition, formulating the GaSDS:

(5)

′

where ztd and zd

t denote the noisy latent codes of the texture d and the converted YUV texture d′. The y∗ is the text embedding of the fixed text keyword. We combine the Lgatex and Lprtex as our final texture generation loss:

Ltex = Lgatex + λprLprtex, (6) where λpr is a weight to balance the gradient from Lprtex.

###### 3.3. Self-guided Consistency Preserved Editing

∂zt ∂I

∂I ∂d

∂d ∂u

To attain the capability of following editing instructions instead of generation prompts, a simple idea is to take the text-guided image editing model InstructPix2Pix [6] ϕip2p as a substitute for Stable Diffusion to form the SDS:

∇uLgatex = Et,ϵ w(t)(ϵϕ

(zt;e,y,t) − ϵ)

. (4)

dc

With the proposed GaSDS, the issue of geometric misalignment is addressed. However, artifacts such as local color distortion or uneven brightness persist in the textures. This is because the T2I model lacks priors of textures, which hinders the synthesis of high-quality texture details.

∂zt′ ∂β,∂u

(zt′;zt,y∗,t) − ϵ

∇β,uLedit = Et,ϵ w(t) ϵϕ

,

ip2p

(7)

where zt′ denotes the latent for the rendering of the edited face, and the original face is embedded to zt as an extra conditional input, following the setting of InstructPix2Pix.

Hence we propose texture prior SDS to introduce such priors of textures. Inspired by DreamFace [54], we train a diffusion model ϕtd1 on texture data to estimate the texture distribution for providing the prior. Our training dataset contains 500 textures, including processed scanning data and selected synthesized data [3]. Different from DreamFace, which uses labeled text in training, we employ a fixed text keyword (e.g., ‘facial texture’) for all textures. Because the objective of ϕtd1 is to model the distribution of textures as a prior, the texture-text alignment is not necessary. We additionally train another ϕtd2 on the YUV color spaces to promote uniform brightness, as shown in Fig 3. We finetune both ϕtd1 and ϕtd2 on Stable Diffusion. The texture

Note that our geometry and texture are represented by separate parameters β and u, so it is possible to independently optimize one of them, enabling separate editing of geometry and texture. Besides, when editing the texture, we integrate the Lprtex to maintain the structural rationality of textures.

Self-guided Consistency Weight. The editing SDS in Eq. 7 enables effective facial editing, while fine-grained editing control still remains challenging, e.g., unpredictable and undesired variations may occur in the results, shown as Fig.

Iteration

dict the attention scores, the norm(·) denotes the normalization operation, and the proj denotes the unwrapping projection from image to UV domain. As C˜i is related to the viewpoint, we establish a unified consistency weight Ci to fuse C˜i from different viewpoints. The initial state of Ci is a matrix of all ‘one’, indicating the highest level of consistency applied to all regions. The updating of Ci at each step is informed by the C˜i. Specifically, we select the regions where the values in C˜i are lower than Ci to be updated. Then we employ a moving average strategy to get the Ci:

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

edited face

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

cross-attention score

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Ci = Ci−1 ∗ w + C˜i ∗ (1 − w), (9)

consistency weight 𝑪𝒊

- Figure 4. Visualization of the edited face, the cross-attention score

where w is a fixed moving average factor. We take the Ci as a weight to perform region-specific consistency.

for token “mask” and the consistency weight Ci during iterations in editing. Note the viewpoints vary due to random sampling in iterations.

Consistency Preservation Regularization. With the consistency weight Ci in hand, we propose a region-specific consistency preservation regularization in the UV domain to encourage consistency between faces before and after editing in both texture and geometry:

10. This hinders sequential editing, as earlier edits can be unintentionally disrupted by subsequent ones. Therefore, consistency between the faces before and after the editing should be encouraged.

Ltexreg = ∥(do − de) ⊙ Ci∥22 , Lgeoreg = ∥(po − pe) ⊙ Ci∥22 ,

However, the consistency between faces during editing and the noticeability of editing effects, are somewhat contradictory. Imagine a specific pixel in texture, encouraging consistency inclines the pixel towards being the same as the original pixel, while the editing may require it to take on a completely different value to achieve the desired effect.

(10)

where do, de denote the texture before and after the editing, po, pe denote the vertices position map unwrapped from the facial geometry before and after the editing, and ⊙ denotes the Hadamard product.

A key observation in addressing this issue is that the weight of consistency should vary in different regions: For regions associated with editing instructions, a lower level of consistency should be maintained as we prioritize the editing effects. Conversely, for irrelevant regions, a higher level of consistency should be ensured. For instance, given the instruction “let her wear a Batman eyemask”, we desire the eyemask effect near the eyes region while keeping the rest of the face unchanged.

With the consistency preservation regularization, we propose the final loss for our self-guided consistency preserved editing as:

LfinalEdit = Ledit + λregLreg, (11) where λreg is the balance weight.

To locate the relevant region for editing instructions, we propose a self-guided consistency weight strategy in the UV domain. We utilize the built-in cross-attention of the InstructPix2Pix model itself. The attention scores introduce the association between different image regions and specific textual tokens. An example of the consistency weight is shown in Fig 4. We first select a region-indicating token T∗ in the instruction, such as “mask”. At each iteration i, we extract the attention scores between the rendered image I of the editing and the token T∗. The scores are normalized and unwrapped to the UV domain based on the current viewpoint, and then we compute temporal consistency weight C˜i from the unwrapped scores:

#### 4. Experiments

###### 4.1. Implementation Details

Our implementation is built upon Huggingface Diffusers [45]. We use stable-diffusion [37] checkpoint for geometry generation, and sd-controlnet-depth [30] for texture generation. We utilize the official instruct-pix2pix [44] in face editing. The RGB and YUV texture diffusion models are both fine-tuned on the stable-diffusion checkpoint. We utilize NVdiffrast [26] for differentiable rendering. Adam [25] optimizer with a fixed learning rate of 0.05 is employed. The generation and editing for geometry/texture require 200/400 iterations, respectively. It takes about 4 minutes to generate or edit a face on a single NVIDIA A30 GPU. We refer readers to the supplementary material for more implementation details.

C˜i = 1 − (proj(norm(att(I′,T∗))))2 , (8) where att(·,·) denotes the cross-attention operation to pre-

Geometry Editing Texture Editing

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

Caucasian teenager girl, thin eyebrows, pink lipstick

Elderly Asian man, gray beard, deep wrinkles

Anne Hathaway Turn her a man Benedict Cumberbatch

Turn him a woman

[Figure 105]

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

Turn him a cropper-made robot

Scarlett Johansson

Jason Statham

Make his eyes big

Make his lips red

Tom Cruise

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Neteyam in Avatar Thanos in the Marvel

Make his eyemask blue

The Joker Spider-man

Make him chubby

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

Captain America Hulk Make him young Turn him female Let her wear Batman eyemask

Kratos in God of War

- Figure 5. FaceG2E enables the generation of highly realistic and diverse 3D faces (on the left), as well as provides flexible editing capabilities for these faces (on the right). Through sequential editing, FaceG2E achieves the synthesis of highly customized 3D faces, such as ‘A female child Hulk wearing a Batman mask’. Additionally, independent editing is available for geometry and texture modification.

- 4.2. Synthesis Results We showcase some synthesized 3D faces in Fig. 1 and Fig.
- 5. As depicted in the figures, FaceG2E demonstrates exceptional capabilities in generating a wide range of visually diverse and remarkably lifelike faces, including notable celebrities and iconic film characters. Furthermore, it enables flexible editing operations, such as independent manipulation of geometry and texture, as well as sequential editing. Notably, our synthesized faces can be integrated into existing CG pipelines, enabling animation and relighting applications, as exemplified in Fig. 1. More animation and relighting results are in the supplementary material.

[Figure 142]

[Figure 143]

[Figure 144]

Describe3DDreamfaceTADAOurs

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

###### 4.3. Comparison with the state-of-the-art

“Let her wear a Geisha makeup”

“Let him wear a black glasses”

“Make his eyemask made of gold”

“Emma Watson” “Will Smith” “Deadpool”

We compare some state-of-the-art methods for text-guided 3D face generation and editing, including Describe3D [48], DreamFace [54] and TADA [27]. Comparisons with some other methods are contained in the supplementary material.

Figure 6. The comparison on text-guided 3D face synthesis. We present both the generation and editing results of each method.

###### 4.3.1 Qualitative Comparison

The qualitative results are presented in Fig. 6. We can observe that: (1) Describe3D struggles to generate 3D faces

following provided texts due to its limited training data and inability to generalize beyond the training set. (2) TADA produces visually acceptable results but exhibits shortcomings in (i) generating high-quality geometry (e.g., evident

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

###### Describe3DDreamfaceTADAOurs

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

Figure 8. Quantitative results of user study. Our results are more favored by the participants compared to the other methods.

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

editing tasks, to all methods for face synthesis. All results are rendered with the same pipeline, except DreamFace, which takes its own rendering in the web demo [20]. A fixed prefix ‘a realistic 3D face model of ’ is employed for all methods when calculating the CLIP score. We report the CLIP Score [40] and Ranking-1 in Tab. 1. CLIP Ranking1 calculates the ratio of a method’s created faces ranked as top-1 among all methods. The results validate the superior performance of our method over other SOTA methods.

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

“Make her big eyes”

“Turn her skin dark”

“Make her

###### → → → lipspurple”

“Scatlett Johansson”

Figure 7. The comparison on sequential face editing.

|Method|Generation<br><br>|Editing|
|---|---|---|
| |Score ↑ Ranking-1 ↑<br><br>|Score ↑ Ranking-1 ↑|
|Describe3D [48]<br><br>Dreamface [54] TADA [27] Ours|29.81 0%<br><br>33.22 10%<br><br>34.85 10%<br><br><br>36.95 80%<br><br>|28.83 0% 33.14 10% 33.73 20% 35.50 70%|

###### 4.3.3 User Study

We perform a comparative user study involving 100 participants to evaluate our method against state-of-the-art (SOTA) approaches. Participants are presented with 10 face generation examples and 10 face editing examples, and are asked to select the best method for each example based on specific criteria. The results, depicted in Fig. 8, unequivocally show that our method surpasses all others in terms of both geometry and texture preference.

- Table 1. The CLIP evaluation results on the synthesized 3D faces.

geometric distortion in its outputs), and (ii) accurately following editing instructions (e.g., erroneously changing black glasses to blue in case 2). (3) Dreamface can generate realistic faces but lacks editing capabilities. Moreover, its geometry fidelity is insufficient, hindering the correlation between the text and texture-less geometry. In comparison, our method is superior in both generated geometry and texture and allows for accurate and flexible face editing.

###### 4.4. Ablation Study

Here we present some ablation studies. Extra studies based on user surveys are provided in the supplementary material.

We further provide a comparison of sequential editing in Fig. 7. Clearly, the editing outcomes of Describe3D and Dreamface in each round lack prominence. Although TADA performs well with single-round editing instructions, it struggles in sequence editing due to unintended changes that impact the preceding editing effects influenced by subsequent edits. For instance, in the last round, TADA mistakenly turns the skin purple. In contrast, our FaceG2E benefits from the proposed self-guided consistency preservation, allowing for precise sequence editing.

###### 4.4.1 Effectiveness of GDG

To evaluate the effectiveness of geometry-texture decoupled generation (GDG), we conduct the following studies.

Geometry-centric SDS (GcSDS). In Fig. 9(a), we conduct an ablation study to assess the impact of the proposed GcSDS. We propose a variation that takes standard textured rendering as input for SDS and simultaneously optimizes both geometry and texture variables. The results reveal that without employing the GcSDS, there is a tendency to generate relatively planar meshes, which lack geometric details such as facial wrinkles. We attribute this deficiency to the misrepresentation of geometric details by textures.

###### 4.3.2 Quantitative Comparison

Geometry-aligned texture content SDS (GaSDS). In Columns 3 and 4 of Fig. 9(b), we evaluate the effectiveness of GaSDS. We replace the depth-ControlNet in GaSDS

We quantitatively compare the fidelity of synthesized faces to text descriptions using the CLIP evaluation. We provide a total of 20 prompts, evenly split between generation and

Ours w/o GcSDS Ours w/o 𝓛𝒕𝒆𝒙𝒚𝒖𝒗 w/o 𝓛𝒕𝒆𝒙𝒑𝒓 w/o 𝓛𝒕𝒆𝒙𝒈𝒂 & 𝓛𝒕𝒆𝒙𝒑𝒓

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

(a) Geometry Generation (b) Texture Generation

- Figure 9. The ablation study of our geometry-texture decoupled generation. The input texts are ‘Scarlett Johansson’ and ‘Will Smith’.

Original face Ours full w/o SC-weight w/o Reg

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

Scarlett Johansson

Hulk

→ Makeherlipspurple

→ LethimwearBatmaneyemask

- Figure 10. Analysis of the proposed self-guided consistency preservation (SCP) in 3D face editing.

irregularities. The complete Lprtex yields the best results.

###### 4.4.2 Effectiveness of SCP

To evaluate the effectiveness of the proposed self-guided consistency preservation (SCP) in editing, we conduct the following ablation study. We make two variants: One variant, denoted as w/o Reg, solely relies on Ledit for editing without employing consistency regularization. The other variant, denoted as w/o SC-weight, computes the consistency preservation regularization without using the selfguided consistency weight.

The results are shown in Fig. 10. While w/o Reg shows noticeable editings following the instructions, unexpected alterations occur, such as the skin and hair of Scarlett turning purple, and Hulk’s skin turning yellow. This inadequacy can be attributed to the absence of consistency constraints. On the other hand, w/o SC-weight prevents undesirable changes in the results but hampers the effectiveness of editing, making it difficult to observe significant editing effects. In contrast, the full version of SCP achieves evident editing effects while preserving consistency in unaffected regions, thereby ensuring desirable editing outcomes.

#### 5. Conclusion

with the standard Stable-Diffusion model to compute Lgatex. The results demonstrate a significant problem of geometrytexture misalignment. This issue arises because the standard Stable Diffusion model only utilizes text as a conditional input and lacks perception of geometry, therefore failing to provide geometry-aligned texture guidance.

Texture prior SDS. To assess the efficacy of our texture prior SDS, we compared it with two variants: one that solely relies on geometry-aware texture content SDS, denoted as w/o Lprtex, and another that excludes the use of Lyuvtex , denoted as w/o Lyuvtex . As shown in Columns 1,2 and 3 of Fig. 9(b), the results demonstrate that the w/o Lprtex pipeline generates textures with significant noise and artifacts. The w/o Lyuvtex pipeline produces textures that generally adhere to the distribution of facial textures, but may exhibit brightness

We propose FaceG2E, a novel approach for generating diverse and high-quality 3D faces and performing facial editing using texts. With the proposed geometry-texture decoupled generation, high-fidelity facial geometry and texture can be produced. The designed self-guided consistency preserved editing enabling us to perform flexible editing, e.g., sequential editing. Extensive evaluations demonstrate that FaceG2E outperforms SOTA methods in 3D face synthesis.

Despite achieving new state-of-the-art results, we notice some limitations in FaceG2E. (1) The geometric representation restricts us from generating shapes beyond the facial skin, such as hair and accessories. (2) Sequential editing enables the synthesis of customized faces, but it also leads to a significant increase in time consumption. Each round of editing requires additional time.

#### References

- [1] Stable-dreamfusion. https : / / github . com / ashawkey/stable-dreamfusion, 2022. 12
- [2] Shivangi Aneja, Justus Thies, Angela Dai, and Matthias Nießner. Clipface: Text-guided editing of textured 3d morphable models. arXiv preprint arXiv:2212.01406, 2022. 1
- [3] Haoran Bai, Di Kang, Haoxian Zhang, Jinshan Pan, and Linchao Bao. Ffhq-uv: Normalized facial uv-texture dataset for 3d face reconstruction. arXiv preprint arXiv:2211.13874,

2022. 4

- [4] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2
- [5] Linchao Bao, Xiangkai Lin, Yajing Chen, Haoxian Zhang, Sheng Wang, Xuefei Zhe, Di Kang, Haozhi Huang, Xinwei Jiang, Jue Wang, Dong Yu, and Zhengyou Zhang. Highfidelity 3d digital human head creation from rgb-d selfies. ACM Transactions on Graphics, 2021. 3
- [6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, pages 18392–18402, 2023. 2, 4
- [7] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. arXiv preprint arXiv:2303.13873, 2023. 2
- [8] Rahul Dey and Vishnu Naresh Boddeti. Generating diverse 3d reconstructions from a single occluded face image. In CVPR, pages 1547–1557, 2022. 1
- [9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 34:8780–8794,

2021. 2

- [10] Abdallah Dib, Junghyun Ahn, Cedric Thebault, PhilippeHenri Gosselin, and Louis Chevallier. S2f2: Self-supervised high fidelity face reconstruction from monocular image. In 2023 IEEE 17th International Conference on Automatic Face and Gesture Recognition (FG), pages 1–8. IEEE, 2023. 1
- [11] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2
- [12] Xiao Han, Yukang Cao, Kai Han, Xiatian Zhu, Jiankang Deng, Yi-Zhe Song, Tao Xiang, and Kwan-Yee K. Wong. Headsculpt: Crafting 3d head avatars with text. arXiv preprint arXiv:2306.03038, 2023. 1, 2
- [13] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2
- [14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. 2
- [15] Fangzhou Hong, Mingyuan Zhang, Liang Pan, Zhongang Cai, Lei Yang, and Ziwei Liu. Avatarclip: zero-shot text-

- driven generation and animation of 3d avatars. ACM TOG, 41(4):1–19, 2022. 2
- [16] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2
- [17] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023. 2
- [18] Xin Huang, Ruizhi Shao, Qi Zhang, Hongwen Zhang, Ying Feng, Yebin Liu, and Qing Wang. Humannorm: Learning normal diffusion model for high-quality and realistic 3d human generation. arXiv preprint arXiv:2310.01406, 2023. 2
- [19] Yukun Huang, Jianan Wang, Yukai Shi, Xianbiao Qi, ZhengJun Zha, and Lei Zhang. Dreamtime: An improved optimization strategy for text-to-3d content creation. arXiv preprint arXiv:2306.12422, 2023. 2
- [20] Deemos. Inc. dreamface web demo. https:// hyperhuman.deemos.com/, 2023. 7
- [21] Ajay Jain, Ben Mildenhall, Jonathan T Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields. In CVPR, pages 867–876, 2022. 2
- [22] Ruixiang Jiang, Can Wang, Jingbo Zhang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Avatarcraft: Transforming text into neural human avatars with parameterized shape and pose control. arXiv preprint arXiv:2303.17606, 2023. 2, 4, 12
- [23] Zutao Jiang, Guansong Lu, Xiaodan Liang, Jihua Zhu, Wei Zhang, Xiaojun Chang, and Hang Xu. 3d-togo: Towards text-guided cross-category 3d object generation. In AAAI, pages 1051–1059, 2023. 1
- [24] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, pages 6007–6017, 2023. 2
- [25] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014. 5

- [26] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives for high-performance differentiable rendering. ACM Transactions on Graphics, 39(6), 2020. 5
- [27] Tingting Liao, Hongwei Yi, Yuliang Xiu, Jiaxiang Tang, Yangyi Huang, Justus Thies, and Michael J. Black. TADA! Text to Animatable Digital Avatars. In International Conference on 3D Vision (3DV), 2024. 1, 2, 4, 6, 7
- [28] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In CVPR, pages 300–309, 2023. 1, 2
- [29] Zhengzhe Liu, Yi Wang, Xiaojuan Qi, and Chi-Wing Fu. Towards implicit text-guided 3d shape generation. In CVPR, pages 17896–17906, 2022. 1

- [30] lllyasviel. Controlnet. https://huggingface.co/ runwayml/lllyasviel/sd-controlnet-depth,

2023. 5

- [31] Oscar Michel, Roi Bar-On, Richard Liu, Sagie Benaim, and Rana Hanocka. Text2mesh: Text-driven neural stylization for meshes. In CVPR, pages 13492–13502, 2022. 2
- [32] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2, 3, 11
- [33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 2
- [34] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2

- [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 1, 2, 3, 4
- [36] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pages 22500–22510, 2023. 2
- [37] RunwayML. Stable diffusion v1.5. https : / / huggingface.co/runwayml/stablediffusionv1-5, 2022. 5
- [38] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 35:36479–36494, 2022. 2
- [39] Aditya Sanghi, Hang Chu, Joseph G Lambourne, Ye Wang, Chin-Yi Cheng, Marco Fumero, and Kamal Rahimi Malekshan. Clip-forge: Towards zero-shot text-to-shape generation. In CVPR, pages 18603–18613, 2022. 2
- [40] Aditya Sanghi, Rao Fu, Vivian Liu, Karl DD Willis, Hooman Shayani, Amir H Khasahmadi, Srinath Sridhar, and Daniel Ritchie. Clip-sculptor: Zero-shot generation of high-fidelity and diverse shapes from natural language. In CVPR, pages 18339–18348, 2023. 7
- [41] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 2
- [42] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 35:25278– 25294, 2022. 2
- [43] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2020. 2

- [44] timbrooks. Instructpix2pix. https://huggingface. co/runwayml/timbrooks/instruct- pix2pix,

2023. 5

- [45] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, and Thomas Wolf. Diffusers: State-of-the-art diffusion models, 2022. 5
- [46] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, et al. Rodin: A generative model for sculpting 3d digital avatars using diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4563–4573, 2023. 1, 2
- [47] Erroll Wood, Tadas Baltruˇsaitis, Charlie Hewitt, Matthew Johnson, Jingjing Shen, Nikola Milosavljevi´c, Daniel Wilde, Stephan Garbin, Toby Sharp, Ivan Stojiljkovi´c, et al. 3d face reconstruction with dense landmarks. In ECCV, pages 160–

177. Springer, 2022. 1

- [48] Menghua Wu, Hao Zhu, Linjia Huang, Yiyu Zhuang, Yuanxun Lu, and Xun Cao. High-fidelity 3d face generation from natural language descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4521–4530, 2023. 2, 6, 7
- [49] Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. pages 7452–7461, 2023. 2
- [50] Jiale Xu, Xintao Wang, Weihao Cheng, Yan-Pei Cao, Ying Shan, Xiaohu Qie, and Shenghua Gao. Dream3d: Zero-shot text-to-3d synthesis using 3d shape prior and text-to-image

- diffusion models. In CVPR, pages 20908–20918, 2023. 1

[51] Jiale Xu, Xintao Wang, Weihao Cheng, Yan-Pei Cao, Ying Shan, Xiaohu Qie, and Shenghua Gao. Dream3d: Zero-shot text-to-3d synthesis using 3d shape prior and text-to-image

- diffusion models. In CVPR, pages 20908–20918, 2023. 2

- [52] Kim Youwang, Kim Ji-Yeon, and Tae-Hyun Oh. Clip-actor: Text-driven recommendation and stylization for animating human meshes. In ECCV, pages 173–191. Springer, 2022. 1
- [53] Huichao Zhang, Bowen Chen, Hao Yang, Liao Qu, Xu Wang, Li Chen, Chao Long, Feida Zhu, Kang Du, and Min Zheng. Avatarverse: High-quality & stable 3d avatar creation from text and pose. arXiv preprint arXiv:2308.03610,

2023. 1, 4

- [54] Longwen Zhang, Qiwei Qiu, Hongyang Lin, Qixuan Zhang, Cheng Shi, Wei Yang, Ye Shi, Sibei Yang, Lan Xu, and Jingyi Yu. Dreamface: Progressive generation of animatable 3d faces under text guidance. arXiv preprint arXiv:2304.03117, 2023. 1, 2, 4, 6, 7
- [55] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 2, 4
- [56] Wojciech Zielonka, Timo Bolkart, and Justus Thies. Towards metrical reconstruction of human faces. In ECCV, pages 250–269. Springer, 2022. 1

#### A. Appendix

Editing

|ours|w/o SC-weight w/o Reg<br><br>|
|---|---|
|3.95<br><br>|2.55 2.28|

###### A.1. Implementation Details

Camera settings. During the optimization, We employ a camera with fixed intrinsic parameters: near=0.1, far=10, fov=12.59, rendering image size=224. For the camera extrinsics, we defined a set of optional viewing angles and randomly selected one of these angles as the rendering viewpoint for optimization in each iteration. The elevation angle x ∈ 0,10,30, the azimuth angle y ∈ {0,30,60,300,330}, and the camera distance d ∈ {1.5,3}. We set these extrinsics to ensure that the rendering always includes the facial region.

Table 3. Ablation study of face editing based on user ratings.

[Figure 234]

Light settings. We utilize spherical harmonic (SH) to represent lighting. We pre-define 16 sets of spherical harmonic 3-band coefficients. In each iteration of rendering, we randomly select one set from these coefficients to represent the current lighting.

Figure 11. Relighting of our synthesized 3D faces.

Prompt engineering. In the generation stage, for the face description prompt of a celebrity or a character, we add the prefix ‘a zoomed out DSLR photo of ’. We also utilize the view-dependent prompt enhancement. For the azimuth in (0,45) and (315,360), we add a suffix ‘ from the front view’, for the azimuth in (45,135) and (225,315), we add a suffix ‘ from the side view’.

Generation with composed prompt

Generation + Sequential editing

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

Turn him a cropper-made robot

Jason Statham Make his eyes big

Make his lips red

SDS Time schedule. Following the Dreamfusion [32], we set the range of t to be between 0.98 and 0.02 in the SDS computation process. Besides, we utilize the linearly decreasing schedule for t, which is crucial for the stability of synthesis. As the iteration progresses from 0 to the final (e.g. iteration 400), our t value linearly decreases from 0.98 to 0.02.

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

Let him wear a purple Zorro mask

Mark Zuckerberg Make his lips black

Make his eyes big

Figure 12. Generation with composed prompt leads to the loss of concepts in prompts.

###### A.2. User survey as ablation

###### A.3. More Relighting Results

We conduct a user survey as ablation to further validate the effectiveness of our key design. A total of 100 volunteers participated in the experiment. We presented the results of our method and different degradation versions, alongside the text prompts. Then we invited the volunteers to rate the facial generation and editing. The ratings ranged from 1 to 5, with higher scores indicating higher satisfaction. The user rating results are shown in Tab. 2 and Tab. 3. The results indicate that removing any of our key designs during the face generation or face editing leads to a decrease in user ratings. This suggests that our key designs are necessary for synthesizing high-quality faces.

We present some more relighting results in Fig 11. We recommend referring to the supplementary video or project page, where the video results can better demonstrate our animation and relighting effects.

###### A.4. Generation with composed prompt

Our sequential editing can synthesize complex 3D faces, an alternative approach is to combine all editing prompts into a composed prompt and generate the face in one step.

In Fig.12, we showcase the results generated from a composed prompt with our generation stage. It can be observed that directly generating with the composed prompt leads to the loss of certain concepts and details present in the prompts (e.g., the cropped-made effect in row 1, or the black lips in row 2). This underscores the necessity of the editing technique we propose for synthesizing customized faces.

Generation

|ours|w/o Lyuvtex w/o Lprtex w/o Lgatex & Lprtex|
|---|---|
|3.82<br><br>|3.77 2.59 1.78|

- Table 2. Ablation study of face generation based on user ratings.

###### A.5. More Comparison Results

We conduct more comparisons with more baseline methods. We add two baselines: a public implementation [1] for the Dreamfusion, and AvatarCraft [22], a SOTA text-to3D avatar method that utilizes the implicit neutral field representation. We compare text-guided 3D face generation, single-round 3D face editing, and sequential 3D face editing. Note that baseline methods are not capable of directly editing 3D faces with text instruction (e.g., ‘make her old’), so we let them perform the editing by generating a face with the composed prompt. For example, ‘an old Emma Watson’ is the composed prompt of ‘Emma Watson’ and ‘Make her old’.

We present the 3D face generation results in Fig 13 and Fig 14. The 3D face editing results are contained in Fig 15 and Fig 16. The comparisons on sequential editing are presented in Fig 17 and Fig 18. It should be noted that Dreamfusion [1] and Avatarcraft [22] occasionally fail to produce meaningful 3D shapes and instead output a white background for some prompts. This issue could potentially be addressed by resetting the random seed, however, due to time constraints, we did not attempt repeated trials. We have labeled these examples as ‘Blank Result’ in the figures.

###### Dreamfusion Describe3D Avatarcraft TADA Dreamface Ours

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 251]

A teenager boy, Asian, yellow skin, round face

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 256]

[Figure 257]

[Figure 258]

Elderly Asian man, gray beard, deep wrinkles

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 265]

Caucasian teenager girl, thin eyebrows, pink lipstick

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

Taylor Swift

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

Tom Cruise

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Barack Hussein Obama

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

Jake Gyllenhaal

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 300]

Cate Blanchett

- Figure 13. Comparison on text-guided 3D face generation.

[Figure 303]

###### Dreamfusion Describe3D Avatarcraft TADA Dreamface Ours

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

Hellboy in Hellboy series

[Figure 314]

[Figure 315]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

Kratos in the God of War

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 325]

[Figure 326]

[Figure 327]

Thanos in the Marvel

[Figure 328]

[Figure 329]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

Captain America

[Figure 335]

[Figure 336]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

Neteyam in Avatar

[Figure 342]

[Figure 343]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

The Flash

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 353]

[Figure 354]

[Figure 355]

Voldemort

[Figure 356]

[Figure 357]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

Red Skull

- Figure 14. Comparison on text-guided 3D face generation.

###### Dreamfusion Describe3D AvatarCraft TADA Dreamface Ours

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

Original: Emma Watson

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

Editing

Editing: Make her old

Blank Result

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

Original

Original: Scarlett Johansson

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

Editing

Editing:

Turn her into Harley Quinn

[Figure 390]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

Original

Original: Scarlett Johansson

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

Editing

Editing: Let her wear a Geisha makeup

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

Original

Blank Result

Original: Brad Pitt

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

Editing

Editing: Make him made of wood

Blank Result

- Figure 15. Comparison on text-guided single-round 3D face editing.

##### Dreamfusion Describe3D AvatarCraft TADA Dreamface Ours

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 422]

[Figure 423]

Original

Original: Benedict Cumberbatch

[Figure 424]

[Figure 425]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

Editing

Editing: Turn him a girl

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 437]

Original

Original: Anne Hathaway

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 443]

Editing

Editing: Turn her a man

Blank Result

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 450]

Original

Original: Spider-man

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

Editing

Editing: Make him chubby

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

Original

Original: The Joker

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

Editing

Editing: Make his eyemask blue

- Figure 16. Comparison on text-guided single-round 3D face editing.

###### Dreamfusion Describe3D AvatarCraft TADA Dreamface Ours

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

Original: Jason Statham

Original

[Figure 479]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

Editing step 1

- Editing step 1: Make his eyes big
- Editing step 2: Turn him a cropper-made robot
- Editing step 3: Make his lips red

[Figure 486]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

Editing

- step 2

Editing

- step 3

[Figure 493]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

Original: Hulk

Original

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

Editing step 1

- Editing step 1: Make him young
- Editing step 2: Turn him female
- Editing step 3: Let her wear Batman eyemask

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

Editing

- step 2

Editing

- step 3

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

Figure 17. Comparison on text-guided sequential 3D face editing.

###### Dreamfusion Describe3D AvatarCraft TADA Dreamface Ours

[Figure 528]

[Figure 529]

[Figure 531]

[Figure 532]

[Figure 533]

Original: Mark Zuckerberg

Blank Result

Original

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 540]

Editing step 1

- Editing step 1: Make his eyes big
- Editing step 2: Let him wear a purple Zorro mask
- Editing step 3: Make his lips black

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 545]

[Figure 546]

[Figure 547]

Editing

- step 2

Editing

- step 3

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 560]

[Figure 561]

Original: Elon Musk

Original

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 567]

[Figure 568]

Editing step 1

- Editing step 1: Turn him Asian
- Editing step 2: Give him thick mustache
- Editing step 3: Make him bald

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 574]

[Figure 575]

Editing

- step 2

Editing

- step 3

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 581]

[Figure 582]

Figure 18. Comparison on text-guided sequential 3D face editing.

