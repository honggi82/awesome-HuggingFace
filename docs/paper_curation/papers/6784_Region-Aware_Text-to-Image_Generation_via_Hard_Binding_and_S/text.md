# arXiv:2411.06558v2[cs.CV]15Nov2024

## Region-Aware Text-to-Image Generation via Hard Binding and Soft Refinement

Zhennan Chen1⋆ Yajie Li1⋆ Haofan Wang2,3 Zhibo Chen3 Zhengkai Jiang4 Jun Li1 Qian Wang5 Jian Yang1 Ying Tai1

1Nanjing University 2InstantX Team 3Liblib AI 4HKUST 5China Mobile

https://github.com/NJU-PCALab/RAG-Diffusion

[Figure 1]

[Figure 2]

Flux

[Figure 3]

Ours

RPG

On the left, Einstein is painting the Mona Lisa; in the center, Elon Reeve Musk is participating in the U.S. presidential election; on the right, Trump is hosting a Tesla product launch.

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Ours Ours

Flux Flux

RPG RPG

Seven ceramic mugs in different colors are placed on a wooden table,

A cylindrical glass, obscuring the right half of the apple behind it.

with numbers from 1 to 7 written on the cups, and a bunch of white

roses on the left.

Figure 1. RAG decouples a raw prompt into regional prompts, processes different regions separately and strengthens the interaction between adjacent regions. It enables precise control over object relationship, action and attributes, achieving more harmonious and consistent complex compositional generation compared to competing models like Flux and RPG.

### Abstract

Regional prompting, or compositional generation, which enables fine-grained spatial control, has gained increasing attention for its practicality in real-world applications. However, previous methods either introduce additional trainable modules, thus only applicable to specific models, or manipulate on score maps within cross-attention layers using attention masks, resulting in limited control strength when the number of regions increases. To handle these limitations, we present RAG, a Regional-Aware text-to-image Generation method conditioned on regional descriptions for precise layout composition. RAG decouple the multi-region generation into two sub-tasks, the con-

struction of individual region (Regional Hard Binding) that ensures the regional prompt is properly executed, and the overall detail refinement (Regional Soft Refinement) over regions that dismiss the visual boundaries and enhance adjacent interactions. Furthermore, RAG novelly makes repainting feasible, where users can modify specific unsatisfied regions in the last generation while keeping all other regions unchanged, without relying on additional inpainting models. Our approach is tuning-free and applicable to other frameworks as an enhancement to the prompt following property. Quantitative and qualitative experiments demonstrate that RAG achieves superior performance over attribute binding and object relationship than previous tuning-free methods.

### 1. Introduction

Recent advancements in diffusion models [12, 21, 24, 28– 30, 32–34, 37, 41, 42] have substantially enhanced the aesthetic appeal and prompt adherence in text-to-image synthesis. The prevailing trend sees the denoising architecture shifting from UNet [27] to the Diffusion Transformer (DiT) [21], which excels in scaling with large datasets. Transformer-based diffusion models like PixArt-α [5], Stable Diffusion 3/3.5 [8], and Flux.1 [3] have set a new benchmark, surpassing the quality of earlier UNet-based models such as Stable Diffusion 1.5 [26] and SDXL [22]. Furthermore, employing more robust text encoders, including T5-XXL [23], has demonstrated the ability to render visual text and significantly enhance prompt adherence. Some innovative studies even leverage large language models (LLMs) for text representation, with examples like Kolors [31] utilizing GLM [6] and Playground V3 [18] employing Llama3-8B [7]. Despite this significant progress in generating high-quality images from prompts, achieving precise fine-grained spatial control remains elusive. In essence, current generative models still struggle with comprehending the quantity and spatial arrangement of objects.

To address these limitations, the concept of regional prompting, also known as regional control, regional grounding, or composition generation, has emerged. Unlike providing a single global description, achieving fine-grained region-controllable generation requires users to supply not only the spatial location (e.g., a segmentation mask or bounding box) but also a corresponding description for each region. Several approaches have been proposed under this setting, broadly falling into two categories: tuning-based and tuning-free. For tuning-based methods, they often necessitate the training of an additional module to handle explicit conditions like bounding boxes. For example, GLIGEN [17] integrates regional inputs into new trainable layers through a gated mechanism, where each grounding token is a combination of the semantics of the grounded entity and its spatial location. Similarly, InstanceDiffusion [35] and MS-Diffusion [36] also incorporate learnable blocks to handle per-instance conditioning. These methods generally deliver strong performance in precise regional control but are limited to specific base models due to the introduction of extra trainable components. On the other hand, tuning-free methods, such as Multidiffusion [1], RPG [39], and Omost [20], operate on the denoised latent space or attention score map with a mask for each region. They frequently employ a split-and-merge strategy but face challenges in maintaining precise control as the number of regions increases.

In this paper, we adopt a tuning-free manner and aim to improve its control strength and coherence when dealing with multiple regions. Specifically, we present RAG, a novel Regional-Aware text-to-image Generation method for precise regional control, which is composed of two sub-

tasks, Regional Hard Binding and Regional Soft Refinement. First, we implement region-aware hard binding at the beginning of the denoising process to ensure that each regional prompt is executed accurately. This step breaks down the input prompt into several regional prompts, each with its respective spatial position, and then merges the individually denoised regional latents into the original image latent. Second, to dismiss the visual boundaries and enhance interaction between adjacent regions, regional soft refinement is applied within the cross-attention layers at the subsequent steps to obtain a regional latent, where K and V are from regional text tokens while Q is from original image latent, followed by a weighted recombination of base image latent and regional latent. Furthermore, leveraging robust control and fusion capabilities, our framework supports users to refine specific unsatisfactory regions from the last generation while keeping all other regions intact. Both quantitative and qualitative results demonstrate our superior performance over attribute binding and object relationship than previous state-of-the-art tuning-free methods. Our contributions are summarized as follows:

- • We propose RAG, a tuning-free Regional-Aware text-toimage Generation framework on the top of DiT-based model (FLUX.1-dev), with two novel components, Regional Hard Binding and Regional Soft Refinement, for precise and harmonious regional control.
- • RAG novelly makes image repainting feasible, allowing users to modify specific unsatisfactory regions in the previous generation while keeping all other regions intact without need for additional inpainting models.
- • Extensive qualitative and quantitative experiments demonstrate that RAG shows superior performance over attribute binding, object relationship and complex composition on T2ICompBench benchmark, in comparison with previous state-of-the-arts.

### 2. Related Work

Tuning-based Regional Control. Conventional text-toimage generation only uses text as conditional input and injects control signals through cross-attention. In order to handle spatial conditions, some works introduce additional training modules, such as ControlNet [43], to handle new control conditions, including depth maps, sketches, human poses, etc. For regional control, additional module is also introduced for training-based methods [4, 11, 17, 35, 36, 38, 40, 44] to process spatial positions, such as bounding boxes (coordinates) or segmentation masks. GLIGEN [17] adopts Fourier embedding to encode box coordinates and adds trainable gated self-attention layer at each transformer block to accept new grounding input. InstanceDiffusion [35] further allows diverse ways to specify region positions such as simple single points, scribbles, bounding boxes or segmentation masks. MS-Diffusion [36] integrates

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

… …

Regional Hard Binding (𝒓 steps) Regional Soft Refinement (𝑻-𝒓 steps)

“Spring in the Forbidden City” “Summer in the Forbidden City” “Autumn in the Forbidden City” “Winter in the Forbidden City”

From left to right, showcasing spring, summer, autumn, and winter in the Forbidden City.

Regional Hard Binding

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

| | |[Figure 16]| |
|---|---|---|---|

𝒙𝒕−𝒓 𝒙𝒕−𝒓

|[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>𝒙𝒕−(𝒓+𝟏) 𝒙𝒕−(𝒓+𝟏)<br><br>“Spring in the Forbidden City, where cherry blossoms bloom in abundance amidst the ancient architecture, painting a picture of hope and renewal.” “Summer in the Forbidden City, alive with vibrant green trees.” “Autumn in the Forbidden City, a canvas of orange and gold as leaves drift lazily to the ground, creating an atmosphere of reflection and peacefulness.” “Winter in the Forbidden City, adorned with a blanket of snow and stillness, presenting a scene of serene beauty and quietude.”<br><br>[Figure 21]<br><br>[Figure 22]<br><br>Regional Soft Refinement|
|---|

𝒘s𝒑𝒍𝒊𝒕𝒊

|| |
|---|
<br><br>| |
|---|
<br><br>𝒎𝒐𝒇𝒇𝒔𝒆𝒕𝒊<br><br>𝒏𝒐𝒇𝒇𝒔𝒆𝒕𝒊<br><br>𝒎𝒔𝒄𝒂𝒍𝒆𝒊<br><br>𝒏𝒔𝒄𝒂𝒍𝒆𝒊|
|---|

𝒉𝒔𝒑𝒍𝒊𝒕𝒊

- Figure 2. The overall framework of RAG, which divides regional-aware generation into two stages: (1) Regional Hard Binding ensures the proper response of regional prompts by processing each region individually with its fundamental description, and bound at the first r steps to ensure accurate attribute representation and entity localization. (2) Regional Soft Refinement improves the harmony of adjacent region via enabling the interaction of regional local conditions with global image latent within the cross-attention layers at the later T − r steps. The lower left corner shows the definition of spatial region in regional hard binding and regional soft refinement.

grounding tokens with its grounding resampler to correlate specific entities and spatial constraints. Other works, such

- as BoxDiff [38] and Attend-and-Excite [4], apply regional constraints via gradient optimization within the denoising process to ensure all regional tokens are attended.

Tuning-Free Regional Control. Although tuning-based methods show strong performance, collecting training data is time-consuming and labor-intensive, and they are limited to specific models due to the introduction of additional modules on the top of base models. To address these challenges, model agnostic approaches [1, 10, 14, 16, 20, 39] are proposed. DenseDiffusion [16] and Omost [20] directly adjust attention scores within cross-attention layers to ensure that the activations within masked regions are promoted. Mixture of Diffusers [14] and MultiDiffusion [1] denoise different locations separately and then combine the denoised latents using regional masks. Recently, RPG [39] introduces complementary regional diffusion through a resizeand-concatenate approach for region-specific compositional generation, where regional denoised latents are resized and merged as a single concatenated latent at each step. However, the control strength of these methods decreases signif-

icantly when the number of regions increases. Thus, in our work, we aim to absorbs the advantages of model-agnostic tuning-free methods and improves their control capabilities when dealing with multiple regions.

### 3. Method

#### 3.1. Preliminaries

Diffusion Transformer (DiT). DiT [21] is a novel architecture integrating transformer as the backbone network within Latent Diffusion Model (LDM), and has become a dominated choice in recent text-to-image generation models like Stable Diffusion 3/3.5 [8] and Flux [3]. By leveraging transformer, DiT efficiently captures complex dependencies in data, resulting in high-quality image generation. Consistent with the design philosophy of LDM, DiT also operates directly in the latent space, allowing the model to generate high-fidelity images that adhere to specified conditions while reducing computational overhead.

Attention Mechanism. The attention mechanism is a crucial component in DiT, enabling effective interactions between the diffusion network and additional control sig-

nals such as text or image. During the diffusion process, the attention mechanism incrementally captures feature representations in the latent space, facilitating efficient denoising

- at each step while preserving global consistency and detail accuracy in the generated results. The attention weights are calculated as follows:

Attention(Q,K,V ) = softmax

QKT √

d · V, (1)

where the query matrix Q, key matrix K, and value matrix V are derived from the input feature embedding vectors in self-attention layers, while K and V are from conditioned embedding vectors in cross-attention layers.

#### 3.2. Overview of RAG

We briefly illustrate the idea of RAG as shown in Figure 2, which decouples the compositional generation process into the construction of individual regions and detail refinement. The implementation of RAG consists of the following steps:

- (1) Regional Hard Binding (Sec. 3.3): This step involves decomposing the raw and complex input prompt containing multiple objects into a subset of fundamental descriptions for each individual region or object, along with their corresponding spatial positions. This process can be accomplished either through a finetuned MLLM as done in [20, 39] or through manual definition. Then, each region is processed individually with its fundamental description and bound only at the early stage of denoising to ensure accurate attribute representation and entity localization.
- (2) Regional Soft Refinement (Sec. 3.4): At this step, highly descriptive sub-prompts are generated for each region along with a global prompt. Similarly, this process can be automated using an MLLM or defined manually, further enriching the definition of each object and promote the fusion between adjacent regions. Instead of manipulating the image latent, the refinement step achieves the interaction of regional local conditions and global image latent within the cross-attention layers.

It is worth noting that based on our setting, RAG can novelly support image repainting (Sec. 3.5) in a free lunch manner by only re-initialize the initial noise in target areas, thereby enabling accurate modification of previously generated images while maintaining the overall generation quality without the need for an additional inpainting model.

#### 3.3. Regional Hard Binding

To ensure the proper response of regional prompts and mitigate object omission when the number of region or object increases, we apply regional hard binding in the early steps of denoising process as illustrated in Figure 2, which involves separately denoising the regions with their short fundamental descriptions and then binding local regional latents into global latent.

Specifically, we first decompose the long input prompt P containing multiple objects into a set of fundamental descriptions pˆi with their position sets mi = {mioffset,miscale} and ni = {nioffset,niscale} by MLLM or manually. Subsequently, we perform text encoding on P and pˆi to obtain y and yˆi. The the individual latent xˆi utilizes yˆi as a text condition, while origin latent x takes the long input prompt P as condition. The formulaic process is as follows:

xt−r = xt−r+1 − ϵθ (xt−r+1,y) (2)

xˆit−r = xˆit−r+1 − ϵθ x ˆit−r+1,yˆi , (3) where i ∈ [1,k], k is the number of regions. ϵθ is the noise predicted. For each denoising step, we bind xˆit−r to the latent space in the rectangular area given by mi,ni as follows:

xt−r = Replace(xt−r,xˆit−r,mi,ni), (4) where Replace(·) represents the process of pasting the individual latents back to the corresponding area in origin latent. The binding is only executed in the early steps within denoising process. We find that a few steps of binding is sufficient for regional completeness, whereas full-steps binding results in either clear visual boundaries adjacent regions or poor interactivity.

#### 3.4. Regional Soft Refinement

Images generated by direct regional hard binding allow for precise control over positioning, effectively preventing object omission. However, the rendering of attributes tends to be relatively coarse, and there is a tendency for noticeable boundaries between adjacent regions. Therefore, we apply regional soft refinement in the later steps of the denoising process to improve the harmony of adjacent regions.

Specifically, similar to the previous binding step, we break down the original long prompt P, but instead of breaking it down into short fundamental descriptions, we break it down into highly descriptive sub-prompts p˜i, with a set of global regions oi = {hisplit,wspliti }, and obtain the corresponding representation y˜i through the text encoder. Then, the text condition y˜i is projected into Ki and V i, while Qi is derived from current image latent xt−(r+1), within cross-attention layers:

Qi = ℓQ xt−(r+1) ,Ki = ℓK(˜yi),V i =ℓV (˜yi)

T

QiKi

V i,

xit−(r+1) = Softmax

√

d

(5)

where ℓQ, ℓk, ℓV are linear projections, xit−(r+1) is globally-informed latent. We crop xit−(r+1) to the region corresponding to different objects to obtain x˜it−(r+1) with rich attributes generated in the global region:

x˜it−(r+1) = Crop xit−(r+1),oi . (6)

Attribute Binding Object Relationship Color ↑ Shape ↑ Texture ↑ Spatial ↑ Non-Spatial ↑ Complex ↑

Model

|Stable v1.4 [26] [CVPR2022] Composable v2 [19] [ECCV2022] Structured v2 [9] [ICLR2023] Stable v2 [26] [CVPR2022] Stable XL [2] [2022] Attn-Exct v2 [4] [TOG2023] GORS [13] [Neurips2023] Pixart-α-ft [5] [ICLR2024] RPG* [39] [ICML2024] Flux.1-dev* [3] [2024]|0.3765 0.3576 0.4156 0.1246 0.3079 0.3080 0.4063 0.3299 0.3645 0.0800 0.2980 0.2898 0.4990 0.4218 0.4900 0.1386 0.3111 0.3355 0.5065 0.4221 0.4922 0.1342 0.3127 0.3386 0.5879 0.4687 0.5299 0.2133 0.3119 0.3237 0.6400 0.4517 0.5963 0.1455 0.3109 0.3401 0.6603 0.4785 0.6287 0.1815 0.3193 0.3328 0.6690 0.4927 0.6477 0.2064 0.3197 0.3433 0.7476 0.5640 0.6724 0.4017 0.3032 0.3702 0.7680 0.5078 0.6195 0.2606 0.3078 0.3650<br><br>|
|---|---|
|Ours<br><br>|0.8039 0.6016 0.7085 0.5193 0.3263 0.4377|

- Table 1. Comparison of alignment evaluation on T2ICompBench [13]. The best results are highlighted in bold, second-best in underline. The basic data is sourced from [13]. * indicates results we reproduced using the official open-source codes and configurations.

specific areas? By leveraging the robust control and fusion capabilities of regional hard binding and regional soft refinement, we can reinitialize noise in a specific region requiring modification, enabling repainting of a region without affecting the overall layout or attributes of other areas. Different from typical post-processing inpainting task that usually requires additional inpainting models, repainting regenerates with the same parameters from the last generation, with only the prompts in the target area modified.

Image Image Encoder

Diffusion Model

Last Denoising Parameters

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

###### A pink shirt with a bird

Last generation Image

Repainting Image

As shown in Figure 3, users are allowed to modify the description of a specific region within P, pˆi, p˜i to obtain new edited prompts Pe, pˆie, p˜ie. Subsequently, we encode them as text features ye, yˆei, y˜ei. Given a total denoising step T, the initial noisy latent x

- Figure 3. Illustration of Re-painting. Different from regular image-to-image inpainting, repainting inherits from last generation with only the target area re-initialized (upper). Given the parameters in previous generation, users are allowed to specify a target area with a new prompt and repaint the image, without relying on additional inpainting models (bottom).

′

T is inherited from the previous generation xT, only the target area indicated by mask is re-initialized.

′

T = init(xT,mask). (9)

x

Then, we also execute Replace(·) to splice these regions:

To ensure the other regions intact, we simultaneously perform denoising on xT and x

′

T. At each timestep t, we replace the corresponding portions of xt with the masked areas of the edited region from x

xrt−(r+1) = Replace xt−(r+1),x˜it−(r+1),oi . (7)

′

t for repainting: xt = Repaint(xt,x

To further improve the alignment between each region with the original prompt and enhance adjacent interactions, we perform a linear weighted recombination of xrt−(r+1) and xt−(r+1) as follows:

′

t,mask). (10)

Through the regional complementarity enabled by regional soft refinement, the repainting region can be seamlessly integrated with the surrounding areas of other regions. This approach not only makes image repainting feasible, but also ensures the coherence and consistency of the overall scene.

xt−(r+1) = xt−(r+1) ∗ (1 − δ)+xrt−(r+1) ∗ δ, (8) where δ controls the fusion strength of image latent xt−(r+1) and region-aware global latent xrt−(r+1).

#### 3.5. Image Repainting

### 4. Experiments

Based on our task setting, where a long complex prompt containing multiple objects is decoupled into several regional sub-prompts for individual processing, a natural question is raised: Can this approach be used to repaint

#### 4.1. Experiment Setting

Implementation Details. We implement our approach on the top of Flux.1-dev [3] for its superior performance. The

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

A balloon on the bottom of a dog.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

A small elephant on the left and a huge rabbit on the right.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

A woman walks toward us in the forest, holding a dog on a leash

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

A glass vase with sunflowers positioned to the right of a white ceramic teapot. To their right is a large mirror, showing the reflection of the vase and the flowers, while the teapot remains slightly out of reflection.

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Three cans of Sprite and two cans of Coke, alternately arranged.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

A two-tier cabinet: the top shelf has two pears, and the bottom shelf has three apples.

Ours Flux RPG SDXL

- Figure 4. Qualitative comparisons on compositional text-to-image generation. From top to bottom, we show 6 examples of different prompts and regions. Compared with previous methods, we demonstrate excellent regional control capabilities.

inference process is set to 20 steps, with guidance scale of 3.5. For large-scale quantitative evaluations, we employ Chain-of-Thought (CoT) template from [39] and leverage GPT-4 to automatically decouple multi-object scenes. All experiments are conducted on a single A6000 GPU.

Compared Methods. To comprehensively evaluate the generation quality, we compare our RAG with several stateof-the-art text-to-image approaches, including: Stable v1.4 [26], Composable v2 [19], Structured v2 [9], Stable v2 [26], Stable XL [2], Attn-Exct v2 [4], GORS [13], Pixart-α-ft [5], RPG [39], and Flux.1-dev [3].

#### 4.2. Main Results

Quantitative Comparison. Table 1 presents our alignment evaluation on the T2ICompBench benchmark, comparing RAG with other state-of-the-art methods. Our approach outperforms competitors in key aspects such as attribute binding, object relationships, and complex composition. Notably, RAG achieves a 29% improvement over RPG in handling prompts with spatial relationships, underscoring its effectiveness in spatially accurate generation. Quantitative results demonstrate our superior performance in handling complex multi-region prompts.

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Ours BrushNet

BrushNet

Ours

A boy holding a

A Corgi lying on the grass.

A Siamese cat lying on the grass.

A boy holding a soccer in one hand.

basketball in one

hand.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

BrushNet BrushNet

Ours

Ours

A man on the left, a woman on the right.

Two plush toys and one

A man on the left, an anime woman on the right.

Three plush toys on the table.

balloon on the table.

- Figure 5. Qualitative comparisons on image repainting between our RAG and the state-of-the-art inpainting model BrushNet. Our results are more region-aware with harmonious effect with the surrounding, revealing diverse potential for applications.

Qualitative Comparison. Figure 4 illustrates visual comparisons, highlighting our superiority in complex multiregion generation. While RPG [39] also excels at regional control, its lack of precise positional control may lead to object omission or fusion. In contrast, our method enable accurate multi-region control via hard binding and soft refinement mechanisms, faithfully conveying details on position, quantity, and attributes based on the input text. Figure 5 shows the comparison with the state-of-the-art inpainting model BrushNet [15], showcasing our strength in repainting both single and contacted regions without conflict. Even when the repainted object differs greatly in style or shape, our method successfully generates the intended target while maintaining the layout and regional attributes, enhancing the flexibility and control of the generation process.

[Figure 73]

[Figure 74]

| |
|---|

|[Figure 75]|
|---|

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

w/o Hard Binding w/o Soft Refinement

w/ Hard Binding w/o Soft Refinement

w/o Hard Binding w/ Soft Refinement

w/ Hard Binding w/ Soft Refinement

Figure 6. Qualitative analysis of Hard Binding and Soft Refinement. The former ensures the proper responses of each region, while the latter enhances the coherence among regions.

Methods Aesthetic ↑ Alignment ↑

|RPG [39] [ICML2024] Stable v3 [8] [ICML2024] Flux.1-dev [3] [2024]|16.1% 14.8% 13.2% 13.4% 18.8% 17.8%<br><br>|
|---|---|
|Ours<br><br>|51.9% 54.0%|

- Table 2. User study on aesthetics and text-image alignment. Our approach shows a significant improvement.

User Study. We randomly sampled 109 prompts from the T2ICompBench test set to evaluate the aesthetic quality and text-image consistency (including attributes and spatial positions) of the generated images. During the evaluation, we invited 24 users to compare the generation results of RPG [39] and Stable v3 [8], Flux.1-dev [3], RAG, selecting the most suitable generation results for each prompt. Through this blind test, we obtained 2,616 user votes. The results are shown in Table 2. RAG is far ahead in terms of both aesthetics and alignment.

#### 4.3. Ablation Study

Effectiveness of Hard Binding & Soft Refinement. The proposed RAG comprises two key components: hard binding and soft refinement. We conducted ablation experiments on these two parts, with visual results shown in Figure 6. Results reveal that hard binding achieves precise object positioning but yields coarse attribute rendering and limited integration among objects. With soft refinement, object attributes are richly detailed and relationships are harmonious, though precise positioning may be compromised, and object omission may sometimes occur.

Effectiveness of Parameter r. We introduced a parameter r to control the frequency of hard binding applications. As shown in Figure 7, excessive hard binding limits the opportunity for subsequent soft refinement, weakening its impact. This may diminish interactions between adjacent objects, potentially causing noticeable boundaries to reappear. By adjusting r, we aim to balance precise object placement

a blue jay on the left and a green parrot on the right.

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

𝒓 = 𝟏 𝒓 = 𝟑 𝒓 = 𝟓 𝒓 = 𝟕 𝒓 = 𝟗

From left to right: forest, penguin, peacock, glacier.

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

𝜹 = 𝟎.𝟎 𝜹 = 𝟎.𝟑 𝜹 = 𝟎.𝟓 𝜹 = 𝟎.𝟕 𝜹 = 𝟏.𝟎

- Figure 7. Qualitative analysis of hard binding steps r and soft refinement strength δ. A few steps of binding is sufficient for regional completeness, and appropriate soft refinement intensity leads to improved regional coherence.

[Figure 90]

[Figure 91]

RAG RAG with Hyper-Flux

From left to right: a red cake, an orange cake, a yellow

cake, and a green cake.

- Figure 8. RAG, accelerated by Hyper-Flux, is 2.5× faster than the original version, maintaining visual quality.

existing acceleration frameworks, such as Hyper-Flux [25], remain compatible with RAG. As shown in Figure 8, combining RAG with Hyper-Flux results in inference being 2.5fold faster while preserving the stability of accuracy and detail control in regional generation. This demonstrates that, although inference speed presents a challenge, with appropriate acceleration strategies, RAG can efficiently generate multi-region images with high quality and maintain control over interactions between regions.

### 5. Conclusion

In this paper, we introduce RAG, a novel tuning-free Region-Aware text-to-image Generation method designed to address challenges posed by regional prompts. RAG operates through two key stages. First, regional hard binding independently constructs the content of each region. Then, regional soft refinement enhances interactions between adjacent regions and improves attribute generation. Furthermore, RAG bypasses the need for external inpainting models, enabling direct image repainting to modify unsatisfactory regions from last generations. Extensive experimental results demonstrate the superiority of RAG on compositional generation compared to prior tuning-free methods.

with smooth integration, optimizing both the structure and cohesion of the generated image.

Effectiveness of Coefficient δ. We introduced the coefficient δ to modulate the intensity of regional soft refinement. As illustrated in Figure 7, excessively low or high δ values may result in noticeable partitioning or slight misalignment between text and image. Setting δ to an optimal level enhances the fusion of image regions, yielding a more coherent and natural overall output.

Limitation and Future Work. RAG offers precise regional control and flexible image repainting but has limitations. Its multi-region decoupling increases inference time as region count grows. Future work will focus on improving RAG’s inference efficiency and integrating with other diffusion models for enhanced scalability and performance.

Analysis of Inference Time. In RAG, the decoupled multi-region generation operation processes each region independently, leading to increased inference time as the number of regions grows. However, because RAG does not modify the underlying architecture of the original model,

### References

- [1] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. 2023. 2, 3
- [2] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 5, 6
- [3] BlackForest. Black forest labs; frontier ai lab, 2024. 2, 3, 5, 6, 7
- [4] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–10, 2023. 2, 3, 5, 6
- [5] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 2, 5, 6
- [6] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. arXiv preprint arXiv:2103.10360, 2021. 2
- [7] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 2

- [8] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 2, 3, 7
- [9] Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. Training-free structured diffusion guidance for compositional text-to-image synthesis. arXiv preprint arXiv:2212.05032, 2022. 5, 6
- [10] Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems, 36, 2024. 3
- [11] Yutong Feng, Biao Gong, Di Chen, Yujun Shen, Yu Liu, and Jingren Zhou. Ranni: Taming text-to-image diffusion for accurate instruction following. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4744–4753, 2024. 2
- [12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [13] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image genera-

- tion. Advances in Neural Information Processing Systems, 36:78723–78747, 2023. 5, 6
- [14] Alvaro´ Barbero Jim´enez. Mixture of diffusers for scene composition and high resolution image generation. arXiv preprint arXiv:2302.02412, 2023. 3
- [15] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. arXiv preprint arXiv:2403.06976, 2024. 7
- [16] Yunji Kim, Jiyoung Lee, Jin-Hwa Kim, Jung-Woo Ha, and Jun-Yan Zhu. Dense text-to-image generation with attention modulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7701–7711, 2023. 3
- [17] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22511–22521, 2023. 2
- [18] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-toimage alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024. 2
- [19] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In European Conference on Computer Vision, pages 423–439. Springer, 2022. 5, 6
- [20] Omost-Team. Omost github page, 2024. 2, 3, 4
- [21] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 2, 3

- [22] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2
- [23] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 2
- [24] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2

- [25] Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, Xing Wang, and Xuefeng Xiao. Hyper-sd: Trajectory segmented consistency model for efficient image synthesis. arXiv preprint arXiv:2404.13686, 2024. 8
- [26] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 5, 6
- [27] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted

- intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 2
- [28] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 2
- [29] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.
- [30] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2
- [31] Kolors Team. Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint,

2024. 2

- [32] Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024. 2
- [33] Haofan Wang, Peng Xing, Renyuan Huang, Hao Ai, Qixun Wang, and Xu Bai. Instantstyle-plus: Style transfer with content-preserving in text-to-image generation. arXiv preprint arXiv:2407.00788, 2024.
- [34] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 2
- [35] Xudong Wang, Trevor Darrell, Sai Saketh Rambhatla, Rohit Girdhar, and Ishan Misra. Instancediffusion: Instancelevel control for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6232–6242, 2024. 2
- [36] X Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. arXiv preprint arXiv:2406.07209, 2024. 2
- [37] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Yujun Lin, Zhekai Zhang, Muyang Li, Yao Lu, and Song Han. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024. 2
- [38] Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7452–7461, 2023. 2, 3
- [39] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and CUI Bin. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In Forty-first International Conference on Machine Learning, 2024. 2, 3, 4, 5, 6, 7
- [40] Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Linjie Li, Kevin Lin, Chenfei Wu, Nan Duan, Zicheng Liu, Ce Liu, Michael

- Zeng, et al. Reco: Region-controlled text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14246–14255, 2023. 2
- [41] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2

- [42] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022. 2
- [43] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2
- [44] Dewei Zhou, You Li, Fan Ma, Xiaoting Zhang, and Yi Yang. Migc: Multi-instance generation controller for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6818– 6828, 2024. 2

