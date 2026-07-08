LayerFlow: A Unified Model for Layer-aware Video Generation

SIHUI JI∗, The University of Hong Kong, DAMO Academy, Alibaba Group, China HAO LUO, DAMO Academy, Alibaba Group, Hupan Laboratory, China XI CHEN, The University of Hong Kong, Hong Kong YUANPENG TU, The University of Hong Kong, Hong Kong YIYANG WANG, The University of Hong Kong, Hong Kong HENGSHUANG ZHAO†, The University of Hong Kong, Hong Kong

### Multi-layer Generation

### Multi-layer Decomposition

Input foreground prompt: Soft, fluffy clouds float lazily across the sky by wind … Input background prompt: A bright balloon floats high against the sky … Input blended prompt: Clouds sweep gently by the wind with a balloon floating …

[Figure 1]

[Figure 2]

[Figure 3]

Input blended video

arXiv:2506.04228v1[cs.CV]4Jun2025

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Input prompt

[Figure 11]

[Figure 12]

[Figure 13]

A kite surfer … A surfboard on beach…

### Foreground-conditioned Layer Generation

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Input foreground video

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Input foreground prompt: A beautifully crafted glass cup … Input background prompt: The table is adorned with delicious dishes … Input blended prompt: A glass cup on the table filled with delicious dishes …

Input prompt

A grassland… A duck walks on grass…

[Figure 24]

[Figure 25]

[Figure 26]

### Background-conditioned Layer Generation

[Figure 27]

[Figure 28]

[Figure 29]

##### Input background video

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Input prompt

A bird soars … A bird flies over hills …

- Fig. 1. Demonstration for the applications of LayerFlow. Given layer-wise prompts, our method produces videos for a transparent foreground, a clean background, and a blended scenario. It also supports different user-provided conditions, enabling users to decompose and recompose videos creatively.

We present LayerFlow, a unified solution for layer-aware video generation. Given per-layer prompts, LayerFlow generates videos for the transparent foreground, clean background, and blended scene. It also supports versatile variants like decomposing a blended video or generating the background for the given foreground and vice versa. Starting from a text-to-video diffusion transformer, we organize the videos of different layers as sub-clips, and leverage layer embeddings to distinguish each clip and the corresponding layer-wise prompts. In this way, we seamlessly support the aforementioned

∗Work is done when Sihui Ji worked as interns in Alibaba DAMO Academy. †Corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-1540-2/2025/08...$15.00 https://doi.org/10.1145/3721238.3730662

SIGGRAPH Conference Papers ’25, August 10–14, 2025, Vancouver, BC, Canada

variants in one unified framework. For the lack of high-quality layer-wise training videos, we design a multi-stage training strategy to accommodate static images with high-quality layer annotations. Specifically, we first train the model with low-quality video data. Then, we tune a motion LoRA to make the model compatible with static frames. Afterward, we train the content LoRA on the mixture of image data with high-quality layered images along with copy-pasted video data. During inference, we remove the motion LoRA thus generating smooth videos with desired layers.

CCS Concepts: • Computing methodologies → Computer vision. Additional Key Words and Phrases: Video generation, multi-layer content generation ACM Reference Format:

Sihui Ji, Hao Luo, Xi Chen, Yuanpeng Tu, Yiyang Wang, and Hengshuang Zhao. 2025. LayerFlow: A Unified Model for Layer-aware Video Generation. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (SIGGRAPH Conference Papers ’25), August 10–14, 2025, Vancouver, BC, Canada. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3721238.3730662

1 INTRODUCTION

Video generation is a burgeoning topic with broad real-world applications [Brooks et al. 2024; Guo et al. 2023; Hong et al. 2022]. Previous works tackle text-to-video (T2V) generation with various approaches, and make exciting advancements [Chen et al. 2024; Yang et al. 2024b]. Some works further explore enhancing the controllability of the generated videos with sequential inputs like optical flows, or depth maps, bringing more convenience and imagination to video content creation [Ma et al. 2023; Wang et al. 2024; Yang et al. 2024a]. In this work, we focus on layer-aware video generation, which means the simultaneous synthesis of foreground, background, and blended videos, with the prompt of each layer.

Layer-wise generation has potential to support flexible decomposition and recomposition of visual assets, and independent editing in layer level. Furthermore, the inclusion of foreground layer with transparency channel allows realistic effects to be seamlessly integrated into background, benefiting visual production workflows and applications. In the field of layered image synthesis, several progress has been made like LayerDiffuse [Zhang and Agrawala 2024] and Alfie [Quattrini et al. 2024], which investigate text-to-RGBA image with transparent effects represented by alpha channel. However, layer-wise video synthesis remains relatively underdeveloped attributed to two primary challenges. First, representations of different layers and alpha mattes in videos remain largely unexplored. The inclusion of temporal dimension of videos increases the complexity of incorporating transparency channels. Second, the scarcity of multilayer video data significantly hinders progress. High-quality layerd video datasets are rare and difficult to construct, posing a barrier to the generalization and diversity of generated videos. As a result of these two difficulties, there are currently no accessible works synthesizing videos of multiple layers to the best of our knowledge.

Facing the aforementioned challenges, in this work, we propose LayerFlow, which supports generating independent layers of a transparent foreground, a clean background and a compositional scenario. To represent different video layers efficiently, the video segment of each layer including alpha channel, and their corresponding prompts are separately stitched together into a long sequence. Layer

embeddings are also inserted into the sequence to endow the model with layer-awareness. Equipped with such a framework, multi-layer videos with transparency can be synthesized simultaneously. The scarcity and challenges in creation of compliant training dataset force us to devise an effective training pipeline to make full use of accessible data, thus we propose a three-stage training strategy based on two well-designed LoRAs [Hu et al. 2021] to allow joint image-video data training. To be specific, first, we finetune the pretrained text-to-video model [Yang et al. 2024b] with video data roughly made by inpainting for initial ability of layer-aware generation. Second, Motion LoRA is trained on static multi-layer videos made by copy and paste for adapting the model to image data. Thus at the final stage, we tune Content LoRA with joint image-video data by turning on or off Motion LoRA according to the type (static or dynamic) of input training data. During inference, Content LoRA is applied for refining layer-aware generation quality while Motion LoRA is removed for restoring video dynamics. By borrowing knowledge from high-quality images and preserving the motion prior through videos, we are capable of generating layered videos including transparent foreground and undisturbed backgrounds.

LayerFlow demonstrates promising abilities in both generation quality and semantic alignment over other solutions such as trained solely on videos or generating then animating. Moreover, by removing the noise from the specified video clip, we can condition on that clip for generating the remaining segments, thus achieving multiple derivative applications in fig. 1 in a unified framework. For example, background-conditioned generation draws foreground on the input video and vice versa, while multi-layer decomposition subtracts independent video layer from a given video. LayerFlow shows the potential to act as a foundational solution for layered video creation with multi-modal conditions to energize more fancy applications.

2 RELATED WORK

Video generation and editing. The fast-paced growth of T2V models has been phenomenal driven by both the Transformer architecture [Vaswani et al. 2017] and diffusion model [Ho et al. 2020]. Conventional diffusion-based T2V models usually start from pretrained text-to-image (T2I) models or leverage large-scale image datasets for training. AnimateDiff [Guo et al. 2023] inflates T2I models by training a plug-and-play motion module to learn transferable motion priors from real-world videos. Using Transformers as the backbone of diffusion models (DiT) [Peebles and Xie 2023] has shown great promise since Sora [Brooks et al. 2024] presented impressive performances. CogVideoX [Yang et al. 2024b] is a novel DiT-based model, achieving long-term consistent video generation with dynamic plots. Existing works have also explored video editing. AVID [Zhang et al. 2024b] conducts inpainting on videos of any duration following a similar architecture as AnimateDiff [Guo et al. 2023], and UniEdit [Bai et al. 2024] proposes a unified tuning-free framework for video motion and appearance editing. All the above approaches fail to generate layered videos, while our method focuses on layer-aware video creation with all related applications.

Layered content generation. Layered image or video generation is an emerging and challenging topic. LayerDiffuse [Zhang and Agrawala 2024] encodes alpha channel transparency into latent

🔥Trainable ❄ Frozen

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

🔥

🔥

Frozen copy-and-paste video

Generated Video

Motion LoRA

Motion LoRA

3D VAE Decoder

×N

[Figure 44]

[Figure 45]

×(𝑇 − 1)

[Figure 46]

[Figure 47]

❄

Transformer Blocks

[Figure 48]

[Figure 49]

# Transformer Blocks

[Figure 50]

🔥

- (b) Motion LoRA Training
- (c) Content LoRA Training

Copy-and-paste video

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Motion LoRA ❄ Motion LoRA ❄

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

+ Layer Embedding

3D VAE Encoder

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

❄

Transformer Blocks

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Multi-layer image

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Text Encoder

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

×N

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Content LoRA🔥 Content LoRA🔥

0, A panda… 1, Woods… 2, A panda in the woods…

Coarse multi-layer video

[Figure 91]

[Figure 92]

(a) Base Model Training

- Fig. 2. Overall pipeline of LayerFlow, which allows for the production of multi-layer videos including transparent foreground, undisturbed background and blended sequences. We organize videos of different layers as sub-clips and concatenate them to form a whole sequence to be encoded by VAE encoder. At the same time, index modification is conducted before prompts are processed by the𝑇5 encoder, then layer embedding is added to text embeddings to impart layer awareness. All the visual patches and text embeddings are fed into transformer blocks as a long tensor. In the process of training, a base model is firstly trained on crudely made multi-layer video data for initial layered generation ability. Motion LoRA tuning prepares the model with accommodations for frozen video and Content LoRA can then borrow knowledge from both high-quality duplicated multi-layer images and copy-pasted video data for improving layer-aware synthesis quality as well as maintenance of motion dynamics.

space of a pre-trained latent diffusion model for the generation of single transparent images or multiple transparent layers. Alfie [Quattrini et al. 2024] modifies the inference behavior of a pretrained DiT for fully automated obtaining of RGBA illustrations. TransPixar [Wang et al. 2025] extends pretrained video diffusion models to generate RGBA videos by jointly modeling RGB and alpha channels through alpha-aware tokens and LoRA-based finetuning, while TransAnimate [Chen et al. 2025] combines pre-trained transparent image models with temporal modules and introduces motion-guided control mechanism for controllable RGBA video generation. Several works [Gu et al. 2023; Lee et al. 2024; Lu et al. 2021] also explored generative layered video decomposition, aiming to separate objects and their associated effects into multiple layers. However, there are no available methods of omni solution for various simultaneous multi-layer video generation tasks to the best of our knowledge, and our work manages to unify solutions of layered video generation and its derived applications in a single framework.

- 3 METHODS

align visual contents of each layer with corresponding text prompts by insertion of layer embeddings on layer-wise text embeddings. Based on two delicately designed LoRAs [Chefer et al. 2024] we can fine-tune the model in a multi-stage scheme which will be detailed in section 3.2 on joint image-video data to synthesize high-quality video. We can also achieve conditional layered video generation with such a unified framework to support versatile variants.

Layer-wise video representation. We propose a simple but effective formulation to represent different layers in videos, which concatenates visual embeddings of each layer including alpha-matte as a long sequence. The following 3D attention associates between text and video and shares information among layers, contributing to inter-layer coherence. Note that we divide foreground into RGB sequence and alpha sequence for transparency representation, thus it can be combined as “RGBA video” for further re-composition.

Layer-wise text prompts. To make the generated video clips separately refer to corresponding prompts, we propose to conduct textual modification in both text input and encoded embeddings to impart layer-awareness. Specifically, before the description of each layer encoded by 𝑇5 [Raffel et al. 2020], we attach an index number to the prompt in the format of "index number, layer description". After encoding, a learnable layer embedder projects the index number to a layer embedding of the same size as text embedding before being added to the corresponding text embedding. All the above modifications together with position embedding link each layer to its description by building the correspondence between pairs of textual and visual embedding explicitly and implicitly.

The pipeline of LayerFlow is demonstrated in fig. 2. Given three text inputs that seperately describe content of the target layer, our model can generate foreground, alpha channel, background, and blended video with high fidelity and per-layer prompt alignment. We first give a brief introduction to the overall framework in section 3.1. Following that, our comprehensive training pipeline is outlined in section 3.2 on training strategy and dataset organization.

- 3.1 Overall Framework

We first adapt a DiT-based T2V model to a layer-aware generation setting by concatenating sub-clips of all layers as a whole, then

Conditional layer generation. We can also make simple modifications to this framework to support variants of conditional layer

generation, including foreground/background-conditioned generation as well as layer decomposition. More specifically, by removing noise from the foreground clip of visual embeddings and departing it from loss calculation in training process, foreground sequence acts as condition for synthesis of remaining video segments and is only used in attention sharing for modeling relationships among layers. Thus the framework becomes a foreground-conditioned video generator. The other two applications also share similar modifications and we demonstrate all these variations in experiments.

- 3.2 Training Pipeline

The layer-aware video generation model requires high-quality multilayer video data, which is often difficult to obtain or construct. To address this, we propose a three-stage training scheme combining static image and dynamic video data, leveraging motion and Content LoRA to overcome data scarcity, resulting in improved inter-layer coherence, aesthetic quality, and motion dynamics.

First stage: base model training. We first train a base model on low-quality multi-layer video datasets to empower the model with initial layer-aware generation ability. The denoising network 𝜖𝜃 learns to predict the added noise, encouraged by an MSE loss:

√1 − 𝛼¯𝑡𝜖,𝑡,

𝐿(𝜃) := E𝑡,𝑥0,𝑦,𝑖𝑙,𝜖 𝜖 − 𝜖𝜃 √𝛼¯𝑡𝑥0 +

2

, (1)

𝜏𝜃 (𝑦) + 𝜏𝑙 (𝑖𝑙))

where 𝑥0 is the target video sequence and 𝑡 is uniformly distributed between 1 and 𝑇. Pre-defined 𝛼¯𝑡 determines the noise strength at step 𝑡 and 𝑖𝑙 represent the layer index, thus 𝜏𝑙 (𝑖𝑙) represent the layer embedding.

To prepare the data for this stage, we first segment the foreground video sequence from raw video via SAM-Track [Cheng et al. 2023] under the guidance of a predicted prompt list. Then we filter out duplicate mask sequences and invalid foreground objects. The filtered mask sequences are used to guide the video inpainting model [Zhou et al. 2023] in generating background videos. At last, we utilize CogVLM2 [Hong et al. 2024] to separately caption layer-wise video data. As a result, we acquire the {𝑓𝑜𝑟𝑒𝑔𝑟𝑜𝑢𝑛𝑑,𝑎𝑙𝑝ℎ𝑎,𝑏𝑎𝑐𝑘𝑔𝑟𝑜𝑢𝑛𝑑,

𝑏𝑙𝑒𝑛𝑑𝑒𝑑} pairs of both textual and visual input. Note that due to the weaknesses of segmentation, inpainting models themselves, and poor-quality of raw videos, video frames can contain motion blur or ambiguous foreground edges, and alpha channel is binary. So there is a non-negligible quality gap between the base model output after first-stage training and current state-of-the-art video generation models as shown in fig. 3.

Second stage: Motion LoRA training. A key idea to improve the quality of layered video generation is to train on high-quality but static multi-layer image data. To prevent loss of motion dynamics due to directly trained on static videos made by duplicated images, we propose to employ Motion LoRA as shown in fig. 2 to adapt the base model to image data. Take query (𝑄) projection as an example. The internal feature z after projection becomes

𝑄 = 𝑊𝑄𝑧 + 𝑀𝑜𝑡𝑖𝑜𝑛𝐿𝑜𝑅𝐴(𝑧) = 𝑊𝑄𝑧 + 𝛼 · 𝐴𝐵𝑇𝑧. (2)

The implementation is conducted on all 𝑊 ∈ 𝑊𝑄,𝑊𝐾,𝑊𝑉 to accommodate the motion mode between static and dynamic by adjusting the scalar 𝛼 to 1 and 0. In other words, after optimized on duplicated frames, Motion LoRA enables the model to generate frozen videos when 𝛼 is set to 1, accommodating the model to image data in the third stage.

For better alignment with initial feature distribution of transformer backbone trained on video dataset in the first stage, we sample static frame randomly from video for duplication rather than directly duplicating image data for training. On the other hand, the quality of video data used in this stage should be considerably high since the optimized Motion LoRA is finally employed on highquality image data in last stage. An alternative method to meet both requirements is to copy attainable foreground video matting data with transparency and paste on background videos to form self-made multi-layer video datasets. We need to mention that copypasted video datasets in this stage are only used for Motion LoRA training, thus the problem of incoherence due to random collage will exert no influence on the quality of layered video generation.

Third stage: Content LoRA training. With Motion LoRA adapts the model to joint image-video training, we introduce Content LoRA on the same blocks of transformer backbone in the first two stages, which is also implemented as a multiplication of low-rank matrices:

𝑄 = 𝑊𝑄𝑧 + 𝑀𝑜𝑡𝑖𝑜𝑛𝐿𝑜𝑅𝐴(𝑧) +𝐶𝑜𝑛𝑡𝑒𝑛𝑡𝐿𝑜𝑅𝐴(𝑧)

= 𝑊𝑄𝑧 + 𝛼 · 𝐴𝐵𝑇𝑧 +𝐶𝐷𝑇𝑧. (3)

We optimize the Content Lora with the same diffusion reconstruction loss in Eq. 1, with 𝛼 = 0 for the copy-pasted videos, and 𝛼 = 1 for multi-layer dupilated images as frozen videos. In this way, the model can learn high fidelity from image matting data without losing the motion prior of the base model. After all three stages of training, we drop the Motion LoRA at inference time and reserve the Content LoRA for refinement, showing that this training pipeline helps reduce the negative effects inherited from the weak base model (e.g., defects in background filling, ambiguous foreground boundaries) and achieve transparency, high fidelity, and inter-layer harmony of multi-layer video generation.

The multi-layer image datasets used in this stage have two main sources, one is accessible multi-layer annotated datasets like MULAN [Tudosiu et al. 2024], the other is image matting datasets postprocessed by image inpainting and captioning. Compared to coarse video datasets used in the first stage, multi-layer image datasets possess transparent foregrounds and harmonious backgrounds without obvious motion blur or artifacts, thus playing a decisive role in refining multi-layer generation. Note that copy-pasted videos are also used in this stage with a small proportion of joint data, which has a weak influence on generation coherence but effectively improves fidelity and prevents the model from loss of dynamic property.

4 EXPERIMENTS 4.1 Implementation Details

Training configurations. We implement LayerFlow based on the T2V model CogVideoX [Yang et al. 2024b] with 2B parameters. We

|Foreground: An elderly gentleman, his silver hair glistening in the golden light, sits peacefully. With a brush in hand, he is deeply focused on his canvas Background: The background features a stunning shoreline where soft waves lap gently against the sand, with seabirds flying overhead. Blended: In the video, an elderly gentleman sits at the water's edge, immersed in his painting as the sun sets over the horizon, casting a warm glow.|
|---|

inputforegroundbackgroundblended

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

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

## Ours (joint image and video data) Ours (purely video data)

- Fig. 3. Ablation for training data. We visualize the results for models trained on purely video data and joint image and video data. Without high-quality image data, the model tends to generate a fuzzy background with obvious blur and low fidelity, while joint image-video data training contributes to undisturbed background synthesis and a higher level of text alignment (e.g., "colorful flowers") and generation quality.

[Figure 111]

- Fig. 4. Qualitative comparison for multi-layer video generation with generation then animation pipeline, i.e., composition of LayerDiffuse [Zhang and Agrawala 2024] and motion module [Guo et al. 2023], where LayerFlow achieves better layer-level coherence and clearer separation of layers.

while in the inference process, number of sampling steps is set as 50 with classifier-free guidance scale of 6.

Coarse multi-layer video dataset construction. First, the prompt list used for foreground segmentation is produced by Recognize Anything [Zhang et al. 2024a], composed of possible foreground subjects of the video. With the generated foreground masks, the filtering is then conducted according to the similarity among all masks to avoid duplicated foreground objects. Qwen-VL [Bai et al. 2023] further helps to check whether the segmented sequence is a “real” foreground to exclude samples commonly viewed as background like sky, lake, etc. Based on the above steps, we completed the construction of coarse layered video dataset.

Evaluation protocols. We conduct user studies and utilize four metrics from VBench [Huang et al. 2024] for quantitative assessment: Frame Consistency measures consistency between adjacent frames; Aesthetic Quality considers artistic value; Text Alignment reflects both semantics and style consistency on text prompts; Dynamic Degree depends on whether a video contains large motion or not. The evaluation prompt sets are specially designed with help of GPT4 and CogVLM2 [Hong et al. 2024] as text inputs. Both qualitative and quantitative comparisons are carried out on those prompt sets. Since we have different prompts for foreground, background, and blended videos, the performance is assessed separately on each category. The subtask of text-guided layered video decomposition which encompasses both segmentation and inpainting, enables the reconstruction of foreground and background regions occluded by one another. This distinguishes it from conventional segmentation tasks and makes traditional segmentation metrics unsuitable for its evaluation. As a result, the same metrics except for Dynamic Degree as those used for generation are applied to the decomposed

sample training videos of each layer with 16 frames, thus the frame number becomes 64 after concatenation, and resize each frame to 480 × 720 pixels as CogVideoX [Yang et al. 2024b] does. We only use a simple MSE loss to train the model in all three stages and adopt Adam optimizer with a learning rate of 1𝑒−4 for base model finetuning, 1𝑒−3 for Motion LoRA and 5𝑒−3 for Content LoRA training. Motion LoRA and Content LoRA are also attached on the trainable 1/6 transformer blocks in base model. The model is optimized on 8 NVIDIA A800 GPUs with batch size of 12 for each GPU in training,

Multi-layer decomposition Background-conditioned Layer Generation

|[Figure 112]<br><br>[Figure 113]<br><br>Decomposed Background|[Figure 114]<br><br>[Figure 115]<br><br>Background-conditioned Foreground|[Figure 116]<br><br>[Figure 117]<br><br>Background-conditioned Blended|
|---|---|---|

[Figure 118]

|[Figure 119]<br><br>[Figure 120]<br><br>Decomposed Foreground|[Figure 121]<br><br>[Figure 122]<br><br>Foreground-conditioned Background|[Figure 123]<br><br>[Figure 124]<br><br>Foreground-conditioned Blended|
|---|---|---|

Multi-layer decomposition

Foreground -conditioned Layer Generation

- Fig. 5. Qualitative results for iterative multi-layer video generation, which iteratively implement multi-layer decomposition and conditioned layer generation to recompose video assets.

foreground and background videos for quantitative evaluation. The raw videos for decomposition are from DAVIS [Perazzi et al. 2016] comprised of 50 sequences.

solution like combination of pipelines, it has limitations in thoroughly evaluating the model. In the absence of an appropriate metric to evaluate inter-layer consistency and coherence, we resort to a user study for further quantitative comparison. 30 annotators are required to rate the generated videos from five key aspects. Artistic quality considers the overal quality of three layered videos like the richness and harmony of colors and layout. Foreground quality evaluates the completeness and clarity of the foreground layer; Background quality measures completeness and adherence to the objective of the physical world of the background video. Blended quality assesses harmony and naturalness of blended video while text alignment checks whether the motion and content of videos follow each text description. The results are shown in table 2. We select 25 groups of descriptions including subjects like humans, animals and still objects from prompt sets, then the participants are invited to rank the 25 groups of generated results by LayerFlow and alternative pipelines (generated results of our model trained by purely video data and another framework is also included for ablative analysis, which will be further explained in section 4.3), then the preference scores for each group rated from 1 (worst) to 4 (best) are summed up with a full mark of 100 as the testing metric. The second row and bottom row of table 2 show that LayerFlow performs significantly better text consistency and overall quality over the alternative pipeline.

4.2 Comparisons with Existing Alternatives

It is worth clarifying that since we are studying a novel problem, there is no prior work operating under the exact same setting to the best of our knowledge. We hence compare with a multi-stage alternative achieving the same goal, which also demonstrates that the formulation and overall pipeline is one of the core contributions of our work. We compare LayerFlow with the alternative via both qualitative analysis and user study on the core functionality of multi-layer video generation.

Qualitative analysis. Specifically, we compare our method with a generation then animation baseline. The alternative architecture starts from AnimateDiff [Guo et al. 2023], a prominent T2V design where the video model is built over a text-to-image (T2I) model inflation. We replace the base T2I model in AnimateDiff [Guo et al. 2023] with LayerDiffuse [Zhang and Agrawala 2024] and try to synthesize multi-layer videos by inflating it with motion module [Guo et al. 2023]. However naive plugging of motion module doesn’t work for LayerDiffuse [Zhang and Agrawala 2024] since a conflict exists between the attention sharing in dimension of layer and time. Thus we modify the architecture, where we separately conduct temporal attention within each layer of video frames and layered attention among three layer sequences to resolve this conflict. Although multilayer video generation is achieved, the composition of two attention mechanisms without tuning leads to lack of inter-layer coherence as shown in fig. 4. Besides, our model also demonstrates significant advantages in motion dynamics and text alignment in comparison. Moreover, our model completes the entire pipeline in a single pass, allowing layer-wise interaction for generation coherence and showcasing text-aligned distinguishment between layers.

4.3 Ablation Studies

We ablate crucial designs of our method from two broad perspectives, the training mechanism and the framework architecture.

Training mechanism. As demonstrated in fig. 3, before applying Motion LoRA and Content LoRA for further tuning, the generation results suffer from unclear boundaries of foreground and severe blurring of background. While after joint image-video training based on two LoRAs, the beauty value of foreground is improved and the background is out of fuzziness. The quantitative comparison results on two subtasks, i.e., multi-layer generation and decomposition, are separately listed in table 1 and table 3 (top row and bottom row), both consistently confirming the effectiveness of our training

User study. Although qualitative comparison has demonstrated significant advantages of multi-layer video generation over ad-hoc

- Table 1. Quantitative analysis for model framework and training data of multi-layer generation. Two groups of comparison are included, one is between our model trained without (top row) or with (bottom row) image data, and the other is among different architectures including our framework, LayerDiffuse [Zhang and Agrawala 2024]+motion module [Guo et al. 2023], and “Channel-concatenate” architecture (first three rows). Here, “FG”, "BG", and "BL" refer to foreground, background, and blended layer.

Methods

Dynamic Degree Aesthetic Quality Text Alignment Frame Consistency FG BG BL FG BG BL FG BG BL FG BG BL

Ours (purely video data) 0.32 - 0.60 0.4189 0.4364 0.4959 0.1757 0.1861 0.2428 0.9612 0.9643 0.9608 LayerDiffuse+motion module 0.05 - 0.36 0.4487 0.3701 0.4868 0.1841 0.1695 0.1674 0.9669 0.9888 0.9747 Channel-concatenate 0.02 - 0.01 0.3454 0.2899 0.3871 0.1348 0.0404 0.1585 0.9493 0.9543 0.9549 Ours (joint image-video data) 0.62 - 0.66 0.4859 0.5753 0.5742 0.1972 0.2312 0.2350 0.9621 0.9777 0.9633

- Table 2. User study for multi-layer video generation on LayerFlow and existing alternatives. “Quality” and “T-A” measure overall synthesis quality and text alignment respectively, “FG”, “BG”, and “BL” refer to the detailed evaluation of the foreground, background, and blended video.

Methods Aesthetic (↑) FG (↑) BG (↑) BL (↑) T-A (↑)

Channel-concatenate 40.69 34.85 42.13 38.23 40.61 LayerDiffuse+motion module 46.58 56.07 56.60 45.92 37.92 Ours (purely video data) 73.58 67.33 56.81 74.68 75.57 Ours (joint image-video data) 89.15 91.75 94.46 91.17 95.90

- Table 3. Quantitative analysis for model framework and training data of multi-layer video decomposition. Two groups of comparison are included, one is between our model trained without (top row) and with (bottom row) images, and the other is between different architectures of our base model (top row), and “Channel-concatenate” (middle row).

User study also demonstrates obvious superiorities of our training strategies as shown in the top row and bottom row of table 2, We also carry out a similar ablation analysis on the multi-layer decomposition task as in top and bottom row of table 3 Frame consistency and Text Alignment are all improved, verifying the benefits of joint data training in making layer separation clearer and semantically aligned. We also find that the quality of aesthetics undergoes considerable improvement.

Overal framework. We analyze the generated results of different model architectures with the conclusion that our framework design outperforms all other explorations. The first alternative architecture is training-free by composition of LayerDiffuse [Zhang and Agrawala 2024] and motion module [Guo et al. 2023] as detailed in section 4.2. It only works for video generation, unable to accomplish the decomposition task. The second alternative architecture is more similar to ours where the video sequences are not concatenated in dimension of time but in dimension of layer. It takes 6D video tensors 𝑧 ∈ R𝑏×𝑙×𝑡×𝑐×ℎ×𝑤 as input, where b stands for batch axis, 𝑙 and 𝑡 represent layer axis and time axis respectively. When the internal feature maps go through transformer blocks, the layer axis is reshaped into the channel axis 𝑐 to get expanded feature maps 𝑧 ∈ R𝑏×𝑡×(𝑙×𝑐)×ℎ×𝑤. With the following layer projecting the tensors to original size, such a framework is also ready to complete the generation and decomposition tasks after fine-tuning, which is denoted as “Channel-concatenate”.

Aesthetic Quality Text Alignment Frame Consistency FG BG FG BG FG BG

Methods

Ours (purely video data) 0.3973 0.2822 0.1604 0.0436 0.9371 0.9370 Channel-concatenate 0.3265 0.2782 0.1400 0.0388 0.9280 0.9301 Ours (joint image and video data) 0.4240 0.3020 0.1872 0.0471 0.3973 0.9424

mechanism. As for generation, scores of Frame Consistency reflect that the introduction of joint data allows high-level of appearance consistency of adjacent frames, Image data also contributes to artistic and beauty value perceived by humans according to Aesthetic Quality. Reported text alignment scores witness increasing consistency of semantics on prompts in foreground and background, but decent performance of text alignment in blended videos. It is explainable that joint data, which has a clearer and more distinct separation between foreground and background, will help improve the association between text and video of foreground and background. However, the textual alignment in blended videos may be sacrificed due to differences in distribution in text-video pairs and text-image pairs. Dynamic Degree is introduced as an additional reference for frame consistency since static video will also score well in this temporal metric. In other words, with the dynamic degrees increase in the foreground and blended video, the frame consistency produced by our joint data training model is still proved to be improved, showcasing that the contents do remain consistent throughout the whole video. Note that the prompts of background often describe a nearly static scene, thus the dynamic degree is not measured on background videos.

We compare LayerFlow trained by purely video data with the training-free pipeline and the “Channel-concatenate" architecture trained on same datasets for a fair comparison. “LayerDiffuse + motion module” performs worse than our model on all metrics except for frame consistency due to particularly low dynamic degree. “Channel-concatenate” architecture fails to show promising performance in both setting, possibly due to the difficulty for lightweight projection layer in preserving sufficient visual information among layers as showcased in table 1 and table 3. In contrast, even LayerFlow purely trained on video data (top row) owns a dominant superiority over other alternative architectures for both generation and decomposition, joint data training mechanism further improves the generation quality across all aspects, demonstrating the efficacy of our whole model design.

4.4 Versatile Variants

A set of generated multi-layer videos are shown in fig. 7. Using three descriptions corresponding to each layer, LayerFlow shows

generation capability in both text alignment and artistic quality. For versatile variants, we prove our model’s ability by presenting samples in figs. 6, 8 and 9, demonstrating LayerFlow’s extended capabilities for harmonious foreground or background-conditioned layer generation and multi-layer video decomposition. We also demonstrate results of iterative layered video generation in fig. 5, by conduct multi-layer decomposition followed by conditioned multi-layer video generation with a raw video as input, showing our model’s ability in video recomposition, and we have reasons to believe its potential for future applications.

5 CONCLUSION

We introduce LayerFlow, a unified multi-layer video generation model guided by textual input and optional video conditions. LayerFlow is built upon a novel framework with inherent layer awareness, enabling effective multi-layer generation, decomposition, and foreground/background-conditioned video synthesis. To address the challenges of limited training data, we propose a joint training strategy that integrates both video and image data. This is achieved using two specially designed components: Motion LoRA, which adjusts dynamic motion synthesis, and Content LoRA, which enhances layer separation and improves overall visual quality. Our pipeline demonstrates impressive performance, making it well-suited for various applications in video creation and editing.

Limitations. Our limitation lies in the model’s inability to support multi-layer generation with a variable number of layers. We expect to address this in future work by enabling video generation with a flexible number of layers, allowing for more dynamic and complex scene compositions.

ACKNOWLEDGMENTS

This work is supported by the National Natural Science Foundation of China (No. 62441615, 62422606, 624B2124) and Damo Academy through Damo Academy Research Intern Program.

REFERENCES

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv:2308.12966 (2023).

Jianhong Bai, Tianyu He, Yuchi Wang, Junliang Guo, Haoji Hu, Zuozhu Liu, and Jiang Bian. 2024. Uniedit: A unified tuning-free framework for video motion and appearance editing. arXiv preprint arXiv:2402.13185 (2024).

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. 2024. Video generation models as world simulators. 2024. URL https://openai. com/research/video-generation-modelsas-world-simulators (2024).

Hila Chefer, Shiran Zada, Roni Paiss, Ariel Ephrat, Omer Tov, Michael Rubinstein, Lior Wolf, Tali Dekel, Tomer Michaeli, and Inbar Mosseri. 2024. Still-moving: Customized video generation without customized video data. ACM Transactions on Graphics (TOG) 43, 6 (2024), 1–11.

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. 2024. VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models. arXiv:2401.09047 [cs.CV]

Xuewei Chen, Zhimin Chen, and Yiren Song. 2025. TransAnimate: Taming Layer Diffusion to Generate RGBA Video. arXiv preprint arXiv:2503.17934 (2025). Yangming Cheng, Liulei Li, Yuanyou Xu, Xiaodi Li, Zongxin Yang, Wenguan Wang, and Yi Yang. 2023. Segment and track anything. arXiv:2305.06558 (2023). Zeqi Gu, Wenqi Xian, Noah Snavely, and Abe Davis. 2023. Factormatte: Redefining video matting for re-composition tasks. TOG (2023).

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2023. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv:2307.04725 (2023).

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. In NeurIPS.

Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. 2022. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv:2205.15868 (2022).

Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. 2024. Cogvlm2: Visual language models for image and video understanding. arXiv:2408.16500 (2024).

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv:2106.09685 (2021).

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. 2024. Vbench: Comprehensive benchmark suite for video generative models. In CVPR.

Yao-Chih Lee, Erika Lu, Sarah Rumbley, Michal Geyer, Jia-Bin Huang, Tali Dekel, and Forrester Cole. 2024. Generative Omnimatte: Learning to Decompose Video into Layers. arXiv preprint arXiv:2411.16683 (2024).

Erika Lu, Forrester Cole, Tali Dekel, Andrew Zisserman, William T Freeman, and Michael Rubinstein. 2021. Omnimatte: Associating objects and their effects in video. In CVPR.

Wan-Duo Kurt Ma, John P Lewis, and W Bastiaan Kleijn. 2023. Trailblazer: Trajectory control for diffusion-based video generation. arXiv:2401.00896 (2023). William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In ICCV.

Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. 2016. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition. 724–732.

Fabio Quattrini, Vittorio Pippi, Silvia Cascianelli, and Rita Cucchiara. 2024. Alfie: Democratising RGBA Image Generation With No $$$. arXiv:2408.14826 (2024). Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR (2020).

Petru-Daniel Tudosiu, Yongxin Yang, Shifeng Zhang, Fei Chen, Steven McDonagh, Gerasimos Lampouras, Ignacio Iacobacci, and Sarah Parisot. 2024. MULAN: A Multi Layer Annotated Dataset for Controllable Text-to-Image Generation. In CVPR. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In NeurIPS.

Luozhou Wang, Yijun Li, Zhifei Chen, Jui-Hsien Wang, Zhifei Zhang, He Zhang, Zhe Lin, and Yingcong Chen. 2025. TransPixeler: Advancing Text-to-Video Generation with Transparency. CoRR (2025).

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. 2024. Motionctrl: A unified and flexible motion controller for video generation. In SIGGRAPH.

Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. 2024a. Direct-a-video: Customized video generation with user-directed camera movement and object motion. In SIGGRAPH.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024b. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv:2408.06072 (2024).

Lvmin Zhang and Maneesh Agrawala. 2024. Transparent image layer diffusion using latent transparency. arXiv:2402.17113 (2024).

Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. 2024a. Recognize anything: A strong image tagging model. In CVPR.

Zhixing Zhang, Bichen Wu, Xiaoyan Wang, Yaqiao Luo, Luxin Zhang, Yinan Zhao, Peter Vajda, Dimitris Metaxas, and Licheng Yu. 2024b. AVID: Any-Length Video Inpainting with Diffusion Model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7162–7172.

Shangchen Zhou, Chongyi Li, Kelvin CK Chan, and Chen Change Loy. 2023. Propainter: Improving propagation and transformer for video inpainting. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 10477–10486.

|FG : A man…; BG: A chair on grassland…; BL: A man stands near the chair….|
|---|

|FG : A red fish…; BG: Sea…; BL: A red fish swims in the sea ….|
|---|

backgroundforegroundinputblended

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

- Fig. 6. Demonstrations for background-conditioned layer generation, where we take three layer-wise descriptions and a background sequence as input (top row) and show generated results of foreground (middle row) and blended video (bottom row).

[Figure 143]

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

[Figure 160]

|FG : Campfires flicker …; BG: Forest…; BL: Campfires flicker in the forest….|
|---|

foreground

|FG : A mystical light …; BG : Dark sky…; BL : A mystical light appears in….|
|---|

blendedinputbackground

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

|FG: Hair flows…; Background: Room…; Blended: Hair flows in the room….|
|---|

inputblendedforegroundbackground

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

|Foreground: A cat…; Background: Dog …; Blended: A cat walks on grass ….|
|---|

- Fig. 7. Demonstrations for multi-layer video generation. For each example, we take three layer-wise descriptions as input and show generated results of foreground (top row), background (middle row) and blended video (bottom row).

|FG : A horserider…; BG: A equestrian arena…; BL: A woman guides horse….<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]|
|---|

|FG : A man dances…; BG: Crowd…; BL: A man dances in the crowd….<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]|
|---|

inputbackgroundblendedforeground

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

- Fig. 8. Demonstrations for foreground-conditioned layer generation, where we take three layer-wise descriptions and a foreground sequence as input (top row) and show generated results of background (middle row) and blended video (bottom row).

|FG : A car…; BG: A road…; BL: A car navigates on the road ….<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]|
|---|

|FG : A girl…; BG: Building…; BL: A girl is roller skating in front of ….<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]|
|---|

|FG : A man…; BG: Playground…; BL: A man on the playground….<br><br>[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]|
|---|

|FG : A goat…; BG: Hillside…; BL: A goat on the hillside….<br><br>[Figure 206]<br><br>[Figure 207]<br><br>[Figure 208]|
|---|

foregroundforegroundinputinputblendedblendedbackgroundbackground

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

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

- Fig. 9. Demonstrations for multi-layer video decomposition, where we take three layer-wise descriptions and a blended sequence as input (top row) and show generated results of background (middle row) and foreground video (bottom row).

