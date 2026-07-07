arXiv:2506.01853v1[cs.CV]2Jun2025

## ShapeLLM-Omni: A Native Multimodal LLM for 3D Generation and Understanding

###### Junliang Ye1,3∗ Zhengyi Wang1,3∗ Ruowen Zhao1∗ Shenghao Xie2 Jun Zhu1,3†

Tsinghua University1 Peking University2 ShengShu 3 https://github.com/JAMESYJL/ShapeLLM-Omni/

### Abstract

Recently, the powerful text-to-image capabilities of GPT-4o have led to growing appreciation for native multimodal large language models. However, its multimodal capabilities remain confined to images and text. Yet beyond images, the ability to understand and generate 3D content is equally crucial. To address this gap, we propose ShapeLLM-Omni—a native 3D large language model capable of understanding and generating 3D assets and text in any sequence. First, we train a 3D vector-quantized variational autoencoder (VQVAE), which maps 3D objects into a discrete latent space to achieve efficient and accurate shape representation and reconstruction. Building upon the 3D-aware discrete tokens, we innovatively construct a large-scale continuous training dataset named 3D-Alpaca, encompassing generation, comprehension, and editing, thus providing rich resources for future research and training. Finally, by performing instruction-based training of the Qwen-2.5-vl-7B-Instruct model on the 3D-Alpaca dataset. Our work provides an effective attempt at extending multimodal models with basic 3D capabilities, which contributes to future research in 3D-native AI.

### 1 Introduction

Large language models have made significant achievements, including text-only language models (LLMs) [1, 42, 3, 72], Multimodal Large Language Models (MLLMs) that can understand images [34, 25, 71], video [26, 19, 51, 40] and 3D [82, 63, 8, 14] content. These models employ similar transformer architectures, using dedicated encoders to model each modality independently, thereby integrating images, video, and 3D modalities into existing LLMs.

Recently, ChatGPT-4o [34] has demonstrated remarkable performance. By natively incorporating image generation and understanding into the large language model (LLM) architecture, it enables more fine-grained and precise control through human instructions. However, its multimodal capabilities remain confined to images and text, limiting its potential in more complex spatial domains.

In this work, we propose a unified approach to integrate 3D generation and understanding into a pre-trained multimodal large language model (MLLM). Enhancing LLMs with native 3D capabilities is crucial for downstream applications such as 3D content creation, robotics, digital twins, and immersive virtual environments.

Our method adopts a fully next-token prediction paradigm, which ensures natural compatibility with joint training and large-scale scalability. We leverage a VQVAE to encode 3D meshes into compact discrete tokens, enabling a unified representation. These tokens are utilized for both understanding and generating 3D meshes, following a format analogous to language modeling.

∗Equal contribution †Corresponding author.

Preprint. Under review.

3D mesh

[Figure 1]

This is a 3D mesh of “Beige Teddy Bear “

Continuous Image feature

Discrete Voxel Tokens

3D VQVAE Decoder

Discrete Text Tokens

Text Detokenizer

# ShapeLLM-Omni

Text Tokenizer

3D VQVAE Encoder

Vision Encoder

Text Tokenizer

[Figure 2]

[Figure 3]

Generate a 3D mesh using the image i gave you.

Provide a brief analysis of the 3D asset shown here:

3D Generation 3D Understanding

Image

3D mesh

Figure 1: ShapeLLM-Omni inherits Qwen2.5-vl’s strong multimodal capabilities and additionally supports text-to-3D, image-to-3D, 3D captioning, and 3D editing using text instruction.

To enable LLMs with 3D ability, we construct a comprehensive training dataset using 3D shapes from a mixture of 3D datasets [22, 23, 21, 7]. We construct interleaved 710k text/image-3D pairs to enable the model for basic 3D understanding ability and text/image to 3D generation ability.

Furthermore, to enable interactive 3D mesh editing, we introduce a novel dataset of 62k paired 3D meshes and corresponding text-based editing instructions. This facilitates fine-grained manipulation of 3D assets through natural language, making real-time editing more intuitive and controllable.

After that, we train an LLM on the corpus. We resume from Qwen-2.5-VL-Instruct-7B [4] to utilize the effective of its large-scale pre-training on text and images. Our model demonstrates a wide range of capabilities, including: (1) generating 3D content from language instructions; (2) generating 3D objects from image inputs; (3) interactively editing 3D assets using natural language; (4) understanding and interpreting 3D meshes for semantic and geometric reasoning.

In all, our contributions are:

- • We propose a novel framework for unified 3D object generation and understanding based on a fully autoregressive next-token prediction paradigm.
- • We present the 3D-Alpaca dataset for training large language models (LLMs) with 3D capabilities. Comprising 3.46 billion tokens, it covers three core tasks: 3D generation, 3D understanding, and 3D editing.
- • Our experimental results provide strong empirical evidence supporting the effectiveness of the proposed method.

### 2 Related Work

###### 2.1 3D Mesh Generation

The remarkable achievement of 2D diffusion models [30, 59] has facilitated the exploration of 3D generative models. Early 3D generation methods [54, 81, 10, 41, 58, 38, 66, 17, 75, 68, 101] often rely on SDS-based optimization to distill 3D content due to the limited 3D data, but encounter challenges such as long optimization time and Janus problem. Subsequent works such as [76, 62, 80, 100, 56, 9] enhance semantic consistency across different views during multi-view image synthesis. To minimize

0 1 2 8191

[Figure 4]

…

[Figure 5]

Voxel-Decoder

Voxel-Encoder 3D-Unet

[Figure 6]

3D-Unet

codebook

[Figure 7]

[Figure 8]

quantization

[Figure 9]

[Figure 10]

z

Figure 2: The pipeline of 3D VQVAE, which can compress voxels into discrete tokens.

generation time, more recent approaches [49, 105, 48, 47, 61, 85, 46, 89, 18, 74, 98, 43] adopt a two-stage pipeline that integrates multi-view image prediction with 3D reconstruction to produce

- 3D models. LRM [31] and other works [69, 84, 109, 37, 95, 77, 64, 102, 103, 110, 93, 53, 83] build on a feed-forward reconstruction model and predict 3D structures within seconds. Additionally, native 3D diffusion models [107, 78, 90, 97, 33, 97, 104, 91, 15, 39, 90, 99] encode 3D objects into a VAE latent and adapt a latent diffusion model on the resulting representations for comprehensive 3D understanding. Nevertheless, the above methods treat 3D objects as numerical fields [52, 36] and extract meshes using Marching Cubes [50], which are not easily represented as discrete tokens.

###### 2.2 Autoregressive 3D Generation

Inspired by the success of auto-regressive models in language and image synthesis, some pioneering works [63, 12, 86] have explored their use in 3D shape generation. They adopt VQVAE [73] to compress 3D shapes into latent spaces, which are subsequently quantized into discrete tokens for learning via an auto-regressive transformer. Instead of employing VQVAE, other studies [13, 11, 87, 70, 27, 106] have proposed specialized mesh tokenization techniques that transform mesh vertices and faces into compact discrete token sequences, while preserving the original complex geometric details. These approaches enable the auto-regressive model to effectively generate meshes in a faceby-face manner. Building on 3D auto-regressive models, LLaMA-Mesh [82] explores the integration of natural language instructions with mesh generation and understanding, enabling interactive 3D content creation through a unified framework. However, it treats the 3D OBJ mesh file as text for language model to process, which overlooks the inherent topological structures of 3D data.

###### 2.3 Unified Models for Multimodal Understanding and Generation

Extending large language models (LLMs) to process, generate, and comprehend multiple modalities—such as vision and language—within a unified framework has become a major research frontier. Previous studies [3, 16, 2] have advanced this direction by equipping LLMs with visual understanding capabilities for multimodal tasks. Concurrently, other works [71, 44, 79, 92, 108] have proposed the integration of image and text generation through specialized visual tokenizers. More recently, ChatGPT-4o has further propelled this progress, achieving state-of-the-art performance in both visual comprehension and image synthesis. Beyond 2D modalities, a growing body of research [32, 94, 55, 96] has extended LLMs to 3D content understanding, primarily through point cloud representations. However, point clouds often lack fine-grained geometric detail and are challenging to acquire in real-world settings, limiting their applicability for interactive generation. Despite these advancements, there remains a notable gap: very few models are capable of jointly processing and generating text, images, and 3D data in an integrated manner. To bridge this gap, we introduce a 3D VQVAE module that encodes 3D shapes into discrete representations, enabling autoregressive models to perform unified multimodal understanding and generation across text, images, and 3D content.

### 3 Method

###### 3.1 Overview

Figure 1 provides an overview of our native Multimodal LLM framework, which can handle mixed sequences of text, images, and 3D data and produce corresponding text or 3D outputs. We begin

- Table 1: Modality comparison. In contrast to the task-specific model architectures of SAR3D and Trellis, ShapeLLM-Omni achieves cross-modal alignment by jointly modeling text and 3D representations in a shared latent space, enabling unified understanding and generation capabilities.

| |Input Modality Text Image 3D Unified model<br><br>|Output Modality Text Image 3D|
|---|---|---|
| | | |
|SAR3D [14]|✓ ✓ ✓<br><br>|✓ ✓<br><br>|
|Trellis [91]|✓ ✓<br><br>|✓|
|PointLLM [94]|✓ ✓ ✓<br><br>|✓|
|LLaMA-Mesh [82]|✓ ✓ ✓<br><br>|✓ ✓<br><br>|
|ChatGPT-4o [35]|✓ ✓ ✓<br><br>|✓ ✓<br><br>|
|Qwen-2.5vl [4]|✓ ✓ ✓<br><br>|✓|
|ShapeLLM-Omni (ours)|✓ ✓ ✓ ✓<br><br>|✓ ✓<br><br>|

by converting 3D assets into discrete tokens using a 3D VQVAE (Sec. 3.3), which allows us to leverage the same transformer architecture for both 3D and text token sequences. Subsequently, we assemble a comprehensive 3D supervised fine-tuning dataset, 3D-Alpaca (Sec. 3.4), covering text-to-3D generation, image-to-3D generation, 3D captioning, and 3D editing.

###### 3.2 Architecture

As shown in Figure 1, we represent both text and 3D data as sequences of discrete tokens, enabling fully autoregressive multi-modal generation. This design allows for flexible input and output across modalities in any order. While we adopt token-based representations for both text and 3D modalities, we use continuous features for images. This is because images are only involved in understanding tasks, whereas 3D data supports both understanding and generation. Such a unified modeling approach—based on early fusion—facilitates better modality integration within the language model. Compared to prior work in the 3D domain Table 1, our model is the first unified auto-regressive framework that supports text-to-3D, image-to-3D, 3D understanding, and 3D editing in a single system. It also marks the first attempt at a ChatGPT-4o-style model tailored for 3D tasks.

###### 3.3 3D VQVAE

In this section, we introduce our 3D representation—voxels—explain why we chose voxels, and how we compress voxels into discrete tokens using a 3D VQVAE. Finally, we describe how to reconstruct high-quality 3D meshes from voxels.

Voxel-Based Representation 3D assets can be represented in various ways—such as voxels, point clouds, or Gaussian splats [36]. In this work, we adopt low-resolution voxels as our representation: on one hand, they compress complex 3D information into a much smaller space, which facilitates subsequent training; on the other hand, voxels effectively preserve an asset’s essential shape and skeletal structure, providing sufficient information for a language model. Moreover, we can leverage open-source models to reconstruct coarse-resolution voxels into high-quality, detail-rich meshes.

Model Architecture We adopt a 643 voxel grid resolution, as voxels at this resolution strike the optimal balance for modeling 3D skeletons, preserving essential structural details while avoiding excessive redundancy [91]. Although voxel representations are compact, even modeling a single 3D object with a 643 voxel grid still requires 643 tokens—far beyond what a large language model can handle. Therefore, we further compress voxels using a 3D VQVAE [91]: first, we encode the 643 grid into a 163 latent grid; then we serialize it into 4096 tokens. However, 4096 tokens remain too long. Inspired by [71], which represents images as 1024 tokens, we concatenate every four neighboring tokens along the channel dimension—transforming the original 4096 tokens with 8 channels into 1024 tokens with 32 channels. Finally, we employ an 8192-entry codebook to compress the voxels into 1024 discrete tokens. In all, we represent a single 3D object using 1024 discrete tokens, for both generation and understanding.

Shape Reconstruction Although we employ voxel-based representations for 3D shape generation, practical deployment often necessitates converting voxels into meshes for downstream applications.

3D-Alpaca Dataset

3D-Editing

Text-to-3D Image-to-3D 3D-Captioning

###### Prompt:

Prompt: Generate a 3D mesh

Prompt: Please put a blue and white porcelain vase in the center of the table.

Prompt: Please Provide a concise

Generate a 3D mesh

using the following image: <image>

using the following

caption for the 3D

Prompt: “ A rotating chair ”.

mesh: <meshbos>…<mesh-eos>

[Figure 11]

<mesh-bos>…<mesh-

eos>.

[Figure 12]

[Figure 13]

Response: <mesh-bos>…<mesh-

###### Response:

###### Response:

###### Response:

[Figure 14]

[Figure 15]

[Figure 16]

This is a 3D mesh of A flying eagle.

<mesh-bos>…<mesh-eos>

<mesh-bos>…<mesheos>

eos>

- Figure 3: Our proposed 3D-Alpaca dataset comprises 3D generation, 3D understanding, and 3D editing components, providing a comprehensive foundation for training and evaluating 3D large language models.

To address this, we adopt the approach proposed by Xiang et al. [91], which utilizes a Rectified Flow model to refine and complete voxel information, enabling high-quality mesh reconstruction. By first generating 3D shapes in the voxel domain and then converting them into meshes using this method, our framework achieves a balance between precision and efficiency. This hybrid representation allows large language models to exert fine-grained control over 3D content generation while avoiding the computational burden associated with high-resolution geometry.

###### 3.4 3D-Alpaca Dataset Construction

Although a wealth of datasets has been developed for the supervised fine-tuning of multimodal large-language models, dialogue data within the 3D LLM [32, 14, 94] domain remains relatively scarce. To bridge this gap, we introduce 3D-alpaca, a comprehensive dataset encompassing tasks in 3D content generation, comprehension, and editing.

3D Generation and Understanding Dataset We select a high-quality subset of approximately 712k 3D assets from Trellis [91] and internal collection. For the image collection, each asset is rendered into a 2D image, and a random offset is applied to the frontal view to create the input. Moreover, these rendered images also underpin the construction of the editing dataset in the Sec. 3.4. To generate the text collection and enable early fusion across all three modalities, we render four orthogonal views—front, back, left, and right—of each asset. These multi-view images are then input into the base model Qwen-2.5-VL-Instruct [4] to generate descriptive captions. The resulting captions are utilized both as prompts for text-to-3D generation and as ground-truth targets for 3D-to-text captioning tasks.

- 3D Edited Dataset We aim to build a 3D asset-editing dataset composed of paired 3D assets, where each pair is linked to a specific editing instruction. Despite recent advances in 3D content creation, the field still lacks a model capable of performing consistent edits on 3D assets. In light of the promising performance of current image-editing models, we therefore adopt an image-mediated pipeline: first rendering each 3D asset into images and applying an image-editing model, then reconstructing the edited images back into 3D assets via an image-to-3D generation method. Based on the multimodal alignment demonstrated and with the aim of equipping the model with ChatGPT-4o–level editing capabilities, we follow a six-step pipeline.

- (1) Category: We reference the data distribution of Objaverse-XL [22] and manually selected the 100 most representative and frequent object categories, such as cars, tables, cabinets, human figures, etc.
- (2) Asset Classification: Using ChatGPT-4o, we classify the 3D assets in our dataset into fine-grained subcategories, with the frontal view renderings of each asset as input. From the 3D asset dataset, we filtered 311k assets belonging to the predefined 100 major categories.
- (3) Editing-Prompt Definition: We provide the category names to ChatGPT-4o and instruct it to generate 20 feasible editing-prompts for each category. The instruction given to ChatGPT-4o is: "For each given category name, suggest potential image editing operations that could be applied to objects of that category." Next, we manually review each generated editing prompt and retain only those that meet both our technical feasibility and visual engagement criteria, resulting in 371 unique editing prompts (e.g: “Replace the chair’s backrest with a mesh frame”).
- (4) Asset Sampling & Annotation: Due to time and resource constraints, we build a compact, highquality dataset of editing prompts rather than applying every possible editing prompt to each asset. Specifically, we allocate 200 assets to each editing prompt.
- (5) Editing-Image Pair Collection: For each sampled asset, we provide ChatGPT-4o with its frontal render plus the chosen editing-prompt, and ChatGPT-4o produces the corresponding edited image, yielding image-level editing pairs. After filtering out erroneous cases, we end up with 70k valid editing samples.
- (6) 3D reconstruction: Finally, we employ Trellis [91] to convert the curated images into 3D assets, resulting in 3D pairs before/after editing.

Dialogue Data Construction We define 25 dialogue templates per task (e.g., “Generate a 3D asset of prompt/images”) and encode all 3D assets into discrete token sequences with our pre-trained 3D VQVAE (Sec. 3.3). For each 3D-edit instance, we randomly select 6 templates from a pool of 25; for all other instances, we randomly assign one template each. By merging the tokens with these templates, we create a training corpus of 2.5 million 3D dialogues.

General Conversation To ensure the model’s general conversational capability, we adopt UltraChat [24] as our text-only dataset, with its data distribution shown in the Table 2. For additional details, please refer to the Appendix.

Putting these together After data processing and construction, we finally arrive at the 3D-Alpaca dataset. As shown in the Table 2, the dataset includes four types of tasks: image-to-3D, text-to-3D, 3D-to-caption, and 3D-editing. Together, these four subsets form a total of 2.56 million samples, comprising 3.46 billion tokens. To ensure the large language model retains its original reasoning and dialogue capabilities, we additionally include the UltraChat [24] dataset, a high-quality, large-scale multi-turn dialogue corpus.

- Table 2: Corpus Data Proportions An overview of token and item counts in the training corpus, covering two datasets: the 3D-Alpaca dataset, which includes four task types—Text-to-3D, Image-to3D, 3D-to-Caption, and 3D-Editing—and the text-only UltraChat dataset [24]

Text-To-3D Image-To-3D 3D-to-Caption 3D-Edit 3D-All Text-Only

|Token count<br><br>|0.77B 1.01B 0.77B 0.91B 3.46B|2.16B|
|---|---|---|
|Item count<br><br>|712k 712k 712k 420k 2.56M|1.47M|

### 4 Experiments

###### 4.1 Implementation Details

For training our 3D VQVAE, we adopt a 3D U-Net VAE architecture introduced in Trellis [91]. Our training follows a two-stage strategy: In Stage 1, we freeze the VAE’s pre-trained parameters and train only the codebook. In Stage 2, we unfreeze the VAE and jointly fine-tune it with the codebook. Concretely, each stage runs for 1000 steps on 48 NVIDIA H100 GPUs with a batch size of 25, while the learning rate decays from 5 × 10−3 to 5 × 10−5. For the training of ShapeLLM-Omni,

we use Qwen-2.5-VL-Instruct-7B [4], a multimodal large language model (MLLM) with imageunderstanding capability, as our backbone. Specifically, we extend its base architecture by adding the 8192 3D VQVAE codebook. To preserve its original image-understanding skills, we freeze the parameters of Qwen2.5-vl’s visual encoder. While training, the learning rate decays from 5 × 10−5 to 5 × 10−6, with a per-GPU batch size of 2 and gradient accumulation over 2 steps. The model is trained for 15 epochs on 48 NVIDIA H100 GPUs.

###### 4.2 Quantitative comparisons

Language and Conversational Abilities Table 3 presents quantitative results evaluating language abilities. The table provides a comparison with models: LLaMA-Mesh [82], Chameleon [71], and Qwen2.5-vl [4]. The metrics include SIQA [60], PIQA [6], MMLU [28], and GSM8K [20]. Fine-tuned on 3D-Alpaca for both 3D mesh generation and comprehension, our ShapeLLM-Omni maintains language understanding and reasoning performance on par with baseline models. The result demonstrates that ShapeLLM-Omni effectively extends the MLLM’s capabilities to 3D content generation while preserving its native language capabilities.

Table 3: Language capabilities comparison. We provide a comparison with models: LLaMAMesh [82], Chameleon [71], and Qwen2.5-vl [4]. The metrics include SIQA [60], PIQA [6], MMLU [28], and GSM8K [20]. Fine-tuned on 3D-Alpaca for both 3D mesh generation and comprehension, our ShapeLLM-Omni maintains language understanding and reasoning performance. The table highlights the optimal values in bold and the suboptimal values with underlining.

Metric Qwen2.5-vl-7B ShapeLLM-Omni-7B Chameleon-7B LLaMA-Mesh-8B MMLU 66.9 63.9 59.4 57.4

PIQA 81.0 78.6 79.6 78.9 GSM8K 42.9 55.1 66.9 33.1

SIQA 40.7 41.0 57 40.4

###### 3D Generation We compare our methods on both text-to-3D and image-to-3D generation tasks against CRM [83], SAR3D [14], 3DTopia-XL [15], and TRELLIS [91]. When evaluating the generation performance of ShapeLLM-Omni, we set the model’s top-k parameter equal to the size of the 3D vocabulary (8192), with top-p=0.7 and temperature=0.7. Regarding the dialogue templates, the image-to-3D template is formulated as: "Create a 3D asset using the following image: <image>", while the text-to-3D template is expressed as: "Please generate a 3D mesh based on the prompt I provided: <prompt>". Quantitative evaluations are conducted using image and text prompts sampled from the Toys4K [65] test dataset, with the results summarized in Table 4. To assess the overall quality of the generated 3D outputs, following [91], we compute Frechet Distance (FD) [29] and Kernel Distance (KD) [5] using Inception-V3 [67] features. Additionally, we report the CLIP score [57] to measure the semantic alignment between the generated outputs and their input prompts. As shown in the Table 4, our generation results outperform all baseline methods except for Trellis.

Compared with Trellis Our results are not as good as Trellis for a few reasons. First, Trellis uses separate models for text-to-3D and image-to-3D tasks. In contrast, our ShapeLLM-Omni handles both tasks in a single model, and it also supports 3D editing, understanding, and interactive conversation. This all-in-one training comes with trade-offs and can reduce generation quality. Second, Trellis is built on a Rectified Flow model, while ours is a discrete autoregressive model. From an architectural point of view, it’s expected that this may lead to some performance differences.

3D-to-Caption Following the evaluation settings provided by PointLLM [94], we test the same metrics on the benchmark dataset used by PointLLM. We adopt the same curated test set to assess the 3D-to-caption task. The dialogue prompt is structured as: “<mesh>. Caption this 3D model in detail.”. As shown in Table 5, our ShapeLLM-Omni demonstrates strong 3D understanding capabilities, with performance second only to PointLLM, which is specifically tailored for single-task 3D understanding.

- Table 4: Comparison of methods on Text-to-3D and Image-to-3D tasks. We scale KD by (×102).

Text-to-3D Image-to-3D CLIP↑ FDincep ↓ KDincep ↓ CLIP↑ FDincep ↓ KDincep ↓

Method

CRM - - - 76.1 14.7 0.12 3DTopia-XL - - - 76.5 49.5 1.63 SAR3D 23.9 27.2 0.28 84.70 20.6 0.17 Trellis 30.8 18.3 0.19 85.0 8.31 0.07 ShapeLLM-Omni (ours) 26.7 25.9 0.25 84.5 12.2 0.09

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Image Ours Sar3D Trellis CRM 3Dtopia-XL

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

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

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

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

- Figure 4: Comparisons with other baselines on the image-to-3D task. Our results demonstrate more complete geometry and high-fidelity textures compared to baselines, enabling photorealistic image-to-3D generation.

###### 4.3 Qualitative comparisons

3D Generation To evaluate the effectiveness of our image-conditioned generation, we compare against baselines including SAR3D, TRELLIS, CRM, and 3Dtopia-XL. As illustrated in Figure 4, the baselines exhibit notable limitations in capturing fine-grained visual features, suffering from geometric distortions and texture misalignments. In contrast, our method generates high-quality 3D meshes that accurately preserve both geometry and appearance details. Moreover, our generation quality matches that of TRELLIS, which is our base model and performance upper bound, due to the integration of a well-trained 3D VQVAE and a carefully constructed 3D image-to-3d dataset for LLM fine-tuning. For text-to-3D tasks, Figure 5 presents qualitative comparisons among different baselines. The input prompts are randomly generated by ChatGPT-4o to cover a diverse range of objects. Since 3Dtopia-XL does not support text-to-3D tasks, we use ChatGPT-4o to generate reference images from the test prompts. These images are then used as input for image-to-3D generation. It is evident that our method achieves precise alignment with the given text prompts and excels at generating intricate, coherent details.

3D Editing Compared with traditional generative models, native multimodal LLMs not only enhance image understanding capabilities, but also exhibit significantly improved comprehension of text instructions. Therefore, it introduces a more powerful language-driven interactive 3D asset manipulation paradigm for artists, offering a more flexible and accessible alternative to conventional

- Table 5: 3D object captioning results [94] on Objaverse [22]. As can be seen from the table, our model achieves better performance on 3D understanding/caption tasks. "*" indicate PointLLM was prompted for shorter captions with no more than 20 words.

Model BLUE-1 ROUGE-L METEOR Sentence-BERT SimCSE

InstructBLIP-13B [88] 4.65 8.85 13.23 45.90 48.86 LLaVA-13B [45] 4.02 8.15 12.58 46.37 45.90 3D-LLM [32] 16.91 19.48 19.73 44.48 43.68 PointLLM-13B [94] 3.38 7.23 12.26 47.91 49.12 PointLLM-13B* [94] 17.09 20.99 16.45 50.15 50.83

###### ShapeLLM-Omni (ours) 18.51 21.37 19.89 48.34 49.72

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Prompt Ours Trellis Sar3D 3Dtopia-XL

[Figure 101]

[Figure 102]

A storyed building

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

A iguana with shirt yellow

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

A cartoon robotic white toy

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Green coloured toy van with rectangular curved surface yellow front lights and red seats with grey coloured wheels

[Figure 124]

[Figure 125]

- Figure 5: Comparisons with other baselines on text-to-3d task. Compared to other methods, our results show better text alignment, with generated 3D shapes accurately reflecting the input descriptions.

software-based 3D content creation pipelines, which are usually time-consuming. As shown in Figure 6, ShapeLLM-Omni can edit 3D assets according to user-provided instructions while maintaining good identity consistency.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

+“open the cabinet doors” +“add lid on top” +“add wings” +“grow a tail”

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

BeforeAfter

- Figure 6: Some cases of 3D editing result from our method. Our method enables the editing of 3D assets based on textual instructions while preserving their original identity and visual consistency.

### 5 Conclusion

In this work, we introduce ShapeLLM-Omni, a novel framework that advances both 3D generation and understanding through a 3D VQVAE. By constructing a comprehensive 3D-Alpaca dataset, we provide a data foundation to support future research on native 3D-modality large language models.

Limitation Constrained by limited resources, we possess only 70k 3D-editing pairs—far too few to achieve ChatGPT-4o–level results in 3D editing. Due to limited computing resources, our ShapeLLMOmni only has 7B parameters. As a result, our performance hasn’t yet reached the level of a true “3D version of ChatGPT-4o”.

### A More details about 3D-Alpaca

- A.1 3D Editing Prompt List

As shown in Table 7 and Table 8, we present 70 out of the 100 categories from the 3D editing dataset, along with their corresponding editing prompts.

- A.2 3D Editing Data

As shown in Figure 11, we present several examples from our 3D editing dataset. The figure illustrates that our 3D editing data pairs support effective modifications while preserving subject consistency between the original and edited versions.

### B More Experiments

###### B.1 More Implementation details

[Figure 142]

- Figure 7: About how to generate 3d mesh from voxel. The upper part illustrates the process of reconstructing a textured mesh from voxel inputs using a texture transformer [91] and mesh decoder. In contrast, the lower part demonstrates the pipeline for reconstructing a non-textured mesh directly from voxels. Both reconstruction pathways are optional and can be flexibly applied based on the needs of specific applications.

Decoding Voxel into 3D Mesh As illustrated in the upper part of Figure 7, we first utilize a texture transformer, named Sparse-Flow Transformer [91], to extract texture latents from the voxel representation. These latents are then fed into a voxel-to-mesh decoder, which generates a mesh with associated texture information. Interestingly, we observe that the geometry of the output mesh is entirely determined by the input voxel representation, regardless of the presence of texture information. Inspired by this observation, and as illustrated in the lower part of Figure 7, we define a grey texture latent to support the generation of non-textured meshes.

More Details about Training The model is trained on 48 H100 GPUs for 60k iterations. We conduct full parameter fine-tuning. We use the AdamW optimizer, with a learning rate of 1e-5, a warm-up of 400 steps with cosine scheduling, and a global batch size of 192. The total training time is around 5 days. We present the training and testing loss curve in Figure 8, which demonstrates that the model achieves rapid convergence on the new modality, reflecting effective knowledge adaptation. Throughout training, the loss remains smooth and stable, with no noticeable spikes or instability.

[Figure 143]

###### Figure 8: Training Loss Curve and Testing Loss Curve

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

0ea33b6617174530b97d6b7a92c275fb InstructBLIP UID

de8ec2a724f14fc4b54624512f80f13e A black insect A small, black spider with a long tail.

An appleavatar 3d model A 3D model of a red apple. This is a 3D model of a unique apple, distinctively adorned with a single, vibrant green leaf at the top.

3D-LLM PointLLM

This 3D model depicts a realistic, jet-black insect with a pair of striking, golden brown eyes.

ShapeLLM-Omni An apple with a stem and leaf.

A spider with multiple legs and a segmented body.

Figure 9: Qualitative results on Objaverse.

- B.2 More Qualitative comparisons

In Figure 12, Figure 13, and Figure 14, we showcase additional Image-to-3D generation results. To maintain consistency with the training setup, all input images are resized to 512×512 resolution with a white background. This preprocessing step is crucial, as our base model, Qwen-VL [4], encodes images into token sequences whose length depends on the input resolution. Additional Text-to-3D generation examples are presented in Figure 15. The visual results clearly demonstrate that our model is capable of producing high-fidelity 3D assets through a unified architecture. Furthermore, Figure 10 provides additional 3D-to-caption generation results, and Figure 9 shows two caption examples from Objaverse [22]. The generated captions demonstrate that our ShapeLLM-Omni exhibits robust 3D understanding capabilities.

- B.3 More Quantitative comparisons

Ablation Study To determine the optimal codebook size for our 3D VQVAE model, we train several variants with different codebook sizes. We randomly sample 1000 meshes from the test dataset, voxelize them, and encode them into discrete tokens using each model. These tokens are then decoded into voxel grids and converted back to meshes through a voxel-to-mesh decoder. We evaluate reconstruction quality using Chamfer Distance (CD) and Hausdorff Distance (HD). As shown in Table 6, larger codebooks lead to better reconstruction performance. However, the improvement levels off beyond a codebook size of 8192, indicating saturation. We therefore choose 8192 as the final codebook size to strike a balance between quality and efficiency.

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Renderingsof

Captioninpputmesh

A robot with a rectangular head and four limbs.

A futuristic vehicle with multiple sections and protrusions.

A lighthouse with a cylindrical base and a tower.

A fantasy creature with wings and a pointed head.

Figure 10: Some cases of 3D editing result from our method. Table 6: Ablation study on the codebook Size of 3D VQVAE Vocabulary Size Chamfer Distance↓ Hausdorff Distance↓

4096 0.0102 0.0561 8192 0.0094 0.0525

16384 0.0095 0.0534

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.
- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [5] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. arXiv preprint arXiv:1801.01401, 2018.
- [6] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.
- [7] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.
- [8] Guangyan Chen, Meiling Wang, Yi Yang, Kai Yu, Li Yuan, and Yufeng Yue. Pointgpt: Auto-regressively generative pre-training from point clouds. Advances in Neural Information Processing Systems, 36: 29667–29679, 2023.
- [9] Luxi Chen, Zhengyi Wang, Chongxuan Li, Tingting Gao, Hang Su, and Jun Zhu. Microdreamer: Zeroshot 3d generation in 20 seconds by score-based iterative reconstruction. arXiv e-prints, pages arXiv–2404, 2024.
- [10] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. arXiv preprint arXiv:2303.13873, 2023.
- [11] Sijin Chen, Xin Chen, Anqi Pang, Xianfang Zeng, Wei Cheng, Yijun Fu, Fukun Yin, Billzb Wang, Jingyi Yu, Gang Yu, et al. Meshxl: Neural coordinate field for generative 3d foundation models. Advances in Neural Information Processing Systems, 37:97141–97166, 2025.
- [12] Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Xin Chen, Zhongang Cai, Lei Yang, Gang Yu, et al. Meshanything: Artist-created mesh generation with autoregressive transformers. arXiv preprint arXiv:2406.10163, 2024.

- [13] Yiwen Chen, Yikai Wang, Yihao Luo, Zhengyi Wang, Zilong Chen, Jun Zhu, Chi Zhang, and Guosheng Lin. Meshanything v2: Artist-created mesh generation with adjacent mesh tokenization. arXiv preprint arXiv:2408.02555, 2024.
- [14] Yongwei Chen, Yushi Lan, Shangchen Zhou, Tengfei Wang, and Xingang Pan. Sar3d: Autoregressive 3d object generation and understanding via multi-scale 3d vqvae. In CVPR, 2025.
- [15] Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao, Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie, Tong Wu, Shunsuke Saito, et al. 3dtopia-xl: Scaling high-quality 3d asset generation via primitive diffusion. arXiv preprint arXiv:2409.12957, 2024.
- [16] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024.
- [17] Zilong Chen, Feng Wang, Yikai Wang, and Huaping Liu. Text-to-3d using gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21401– 21412, 2024.
- [18] Zilong Chen, Yikai Wang, Feng Wang, Zhengyi Wang, and Huaping Liu. V3d: Video diffusion models are effective 3d generators. arXiv preprint arXiv:2403.06738, 2024.
- [19] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024.
- [20] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems, 2021. URL https://arxiv. org/abs/2110.14168, 9, 2021.
- [21] Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, et al. Abo: Dataset and benchmarks for real-world 3d object understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21126–21136, 2022.
- [22] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36:35799–35813, 2023.
- [23] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36:35799–35813, 2023.
- [24] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233, 2023.
- [25] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793, 2024.
- [26] Yanan Guo, Wenhui Dong, Jun Song, Shiding Zhu, Xuan Zhang, Hanqing Yang, Yingbo Wang, Yang Du, Xianing Chen, and Bo Zheng. Fila-video: Spatio-temporal compression for fine-grained long video understanding. arXiv preprint arXiv:2504.20384, 2025.
- [27] Zekun Hao, David W Romero, Tsung-Yi Lin, and Ming-Yu Liu. Meshtron: High-fidelity, artist-like 3d mesh generation at scale. arXiv preprint arXiv:2412.09548, 2024.
- [28] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.
- [29] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [30] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

- [31] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023.
- [32] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems, 36:20482–20494, 2023.
- [33] Zixuan Huang, Mark Boss, Aaryaman Vasishta, James M Rehg, and Varun Jampani. Spar3d: Stable point-aware reconstruction of 3d objects from single images. arXiv preprint arXiv:2501.04689, 2025.
- [34] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [35] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [36] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
- [37] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214, 2023.
- [38] Weiyu Li, Rui Chen, Xuelin Chen, and Ping Tan. Sweetdreamer: Aligning geometric priors in 2d diffusion for consistent text-to-3d. arxiv:2310.02596, 2023.
- [39] Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979, 2024.
- [40] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pages 323–340. Springer, 2024.
- [41] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.
- [42] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [43] Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Reconx: Reconstruct any scene from sparse views with video diffusion model, 2024. URL https://arxiv.org/abs/2408.16767.
- [44] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with blockwise ringattention. arXiv preprint arXiv:2402.08268, 2024.
- [45] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [46] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-23-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems, 36:22226–22246, 2023.
- [47] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298–9309, 2023.
- [48] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023.
- [49] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9970–9980, 2024.

- [50] William E Lorensen and Harvey E Cline. Marching cubes: A high resolution 3d surface construction algorithm. In Seminal graphics: pioneering efforts that shaped the field, pages 347–353. 1998.
- [51] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023.
- [52] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.
- [53] Piotr Nawrot, Szymon Tworkowski, Michał Tyrolski, Łukasz Kaiser, Yuhuai Wu, Christian Szegedy, and Henryk Michalewski. Hierarchical transformers are more efficient language models. arXiv preprint arXiv:2110.13711, 2021.
- [54] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv, 2022.
- [55] Zekun Qi, Runpei Dong, Shaochen Zhang, Haoran Geng, Chunrui Han, Zheng Ge, Li Yi, and Kaisheng Ma. Shapellm: Universal 3d object understanding for embodied interaction. In European Conference on Computer Vision, pages 214–238. Springer, 2024.
- [56] Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, and Xiaoguang Han. Richdreamer: A generalizable normal-depth diffusion model for detail richness in text-to-3d. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9914–9925, 2024.
- [57] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [58] Amit Raj, Srinivas Kaza, Ben Poole, Michael Niemeyer, Nataniel Ruiz, Ben Mildenhall, Shiran Zada, Kfir Aberman, Michael Rubinstein, Jonathan Barron, et al. Dreambooth3d: Subject-driven text-to3d generation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2349–2359, 2023.
- [59] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [60] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.
- [61] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023.
- [62] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023.
- [63] Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias Nießner. Meshgpt: Generating triangle meshes with decoder-only transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19615–19625, 2024.
- [64] Yawar Siddiqui, Tom Monnier, Filippos Kokkinos, Mahendra Kariya, Yanir Kleiman, Emilien Garreau, Oran Gafni, Natalia Neverova, Andrea Vedaldi, Roman Shapovalov, and David Novotny. Meta 3d assetgen: Text-to-mesh generation with high-quality geometry, texture, and pbr materials. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 9532–9564. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper_files/paper/2024/file/ 123cfe7d8b7702ac97aaf4468fc05fa5-Paper-Conference.pdf.
- [65] Stefan Stojanov, Anh Thai, and James M Rehg. Using shape to categorize: Low-shot learning with an explicit shape bias. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1798–1808, 2021.

- [66] Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, and Yebin Liu. Dreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior. arXiv preprint arXiv:2310.16818, 2023.
- [67] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016.
- [68] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653, 2023.
- [69] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. arXiv preprint arXiv:2402.05054, 2024.
- [70] Jiaxiang Tang, Zhaoshuo Li, Zekun Hao, Xian Liu, Gang Zeng, Ming-Yu Liu, and Qinsheng Zhang. Edgerunner: Auto-regressive auto-encoder for artistic mesh generation. arXiv preprint arXiv:2409.18114, 2024.
- [71] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. doi: 10.48550/arXiv.2405.09818. URL https://github.com/ facebookresearch/chameleon.
- [72] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [73] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.
- [74] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision, pages 439–457. Springer, 2024.
- [75] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A. Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. arXiv preprint arXiv:2212.00774, 2022.
- [76] Peng Wang and Yichun Shi. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201, 2023.
- [77] Peng Wang, Hao Tan, Sai Bi, Yinghao Xu, Fujun Luan, Kalyan Sunkavalli, Wenping Wang, Zexiang Xu, and Kai Zhang. Pf-lrm: Pose-free large reconstruction model for joint pose and shape prediction. arXiv preprint arXiv:2311.12024, 2023.
- [78] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, et al. Rodin: A generative model for sculpting 3d digital avatars using diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4563–4573, 2023.
- [79] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [80] Xinzhou Wang, Yikai Wang, Junliang Ye, Zhengyi Wang, Fuchun Sun, Pengkun Liu, Ling Wang, Kai Sun, Xintong Wang, and Bin He. Animatabledreamer: Text-guided non-rigid 3d model generation and reconstruction with canonical score distillation. arXiv preprint arXiv:2312.03795, 2023.
- [81] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
- [82] Zhengyi Wang, Jonathan Lorraine, Yikai Wang, Hang Su, Jun Zhu, Sanja Fidler, and Xiaohui Zeng. Llama-mesh: Unifying 3d mesh generation with language models. arXiv preprint arXiv:2411.09595, 2024.
- [83] Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu, Chongxuan Li, Hang Su, and Jun Zhu. Crm: Single image to 3d textured mesh with convolutional reconstruction model. arXiv preprint arXiv:2403.05034, 2024.

- [84] Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan, Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. Meshlrm: Large reconstruction model for high-quality meshes. arXiv preprint arXiv:2404.12385, 2024.
- [85] Haohan Weng, Tianyu Yang, Jianan Wang, Yu Li, Tong Zhang, CL Chen, and Lei Zhang. Consistent123: Improve consistency for one image to 3d object synthesis. arXiv preprint arXiv:2310.08092, 2023.
- [86] Haohan Weng, Yikai Wang, Tong Zhang, CL Chen, and Jun Zhu. Pivotmesh: Generic 3d mesh generation via pivot vertices guidance. arXiv preprint arXiv:2405.16890, 2024.
- [87] Haohan Weng, Zibo Zhao, Biwen Lei, Xianghui Yang, Jian Liu, Zeqiang Lai, Zhuo Chen, Yuhong Liu, Jie Jiang, Chunchao Guo, et al. Scaling mesh generation via compressive tokenization. arXiv preprint arXiv:2411.07025, 2024.
- [88] D Wenliang, L Junnan, L Dongxu, T Anthony Meng Huat, Z Junqi, W Weisheng, L Boyang, F Pascale, and H Steven. Instructblip: Towards general-purpose vision-language models with instruction tuning [c]. Advances in Neural Information Processing Systems, 36, 2023.
- [89] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.
- [90] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. arXiv preprint arXiv:2405.14832, 2024.
- [91] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506, 2024.
- [92] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.
- [93] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191, 2024.
- [94] Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. In European Conference on Computer Vision, pages 131–147. Springer, 2024.
- [95] Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Jiahao Li, Zifan Shi, Kalyan Sunkavalli, Gordon Wetzstein, Zexiang Xu, et al. Dmv3d: Denoising multi-view diffusion using 3d large reconstruction model. arXiv preprint arXiv:2311.09217, 2023.
- [96] Le Xue, Mingfei Gao, Chen Xing, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1179–1189, 2023.
- [97] Xianghui Yang, Huiwen Shi, Bowen Zhang, Fan Yang, Jiacheng Wang, Hongxu Zhao, Xinhai Liu, Xinzhou Wang, Qingxiang Lin, Jiaao Yu, et al. Hunyuan3d 1.0: A unified framework for text-to-3d and image-to-3d generation. arXiv preprint arXiv:2411.02293, 2024.
- [98] Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, and Xiaoguang Han. Stablenormal: Reducing diffusion variance for stable and sharp normal. ACM Transactions on Graphics (TOG), 2024.
- [99] Chongjie Ye, Yushuang Wu, Ziteng Lu, Jiahao Chang, Xiaoyang Guo, Jiaqing Zhou, Hao Zhao, and Xiaoguang Han. Hi3dgen: High-fidelity 3d geometry generation from images via normal bridging. arXiv preprint arXiv:2503.22236, 3, 2025.
- [100] Junliang Ye, Fangfu Liu, Qixiu Li, Zhengyi Wang, Yikai Wang, Xinzhou Wang, Yueqi Duan, and Jun Zhu. Dreamreward: Text-to-3d generation with human preference. In European Conference on Computer Vision, pages 259–276. Springer, 2024.

- [101] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6796–6807, 2024.
- [102] Chubin Zhang, Hongliang Song, Yi Wei, Yu Chen, Jiwen Lu, and Yansong Tang. Geolrm: Geometry-aware large reconstruction model for high-quality 3d gaussian generation. arXiv preprint arXiv:2406.15333, 2024.
- [103] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In European Conference on Computer Vision, pages 1–19. Springer, 2024.
- [104] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 43(4):1–20, 2024.
- [105] Ruowen Zhao, Zhengyi Wang, Yikai Wang, Zihan Zhou, and Jun Zhu. Flexidreamer: single image-to-3d generation with flexicubes. arXiv preprint arXiv:2404.00987, 2024.
- [106] Ruowen Zhao, Junliang Ye, Zhengyi Wang, Guangce Liu, Yiwen Chen, Yikai Wang, and Jun Zhu. Deepmesh: Auto-regressive artist-mesh creation with reinforcement learning. arXiv preprint arXiv:2503.15265, 2025.
- [107] Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, and Shenghua Gao. Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation. Advances in neural information processing systems, 36:73969–73982, 2023.
- [108] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.
- [109] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Long-sequence large reconstruction model for wide-coverage gaussian splats. arXiv preprint arXiv:2410.12781, 2024.
- [110] Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Yan-Pei Cao, and Song-Hai Zhang. Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10324–10335, 2024.

Table 7: Edited Prompt Collection: Part One

ID Category Edited prompt

- 1 Car

Add a cannon to the front, Open the door, Add a roof rack, Add a rear wing, Lengthen the car body, Shorten the car body, Convert into a convertible, Change wheels to square shape, Bend the roof, Add air vents on the sides, Install a spotlight on the roof, Open the hood, Install a rear-view camera

- 2 Tricycle Add a wheel, Install a small trumpet

- 3 Bicycle Raise the seat, Add a wheel, Install a basket

- 4 Traffic light Add an extra light, Install a surveillance camera

- 5 Spaceship

Add wings, Add jet flames, Add solar panels, Install radar antenna, Shorten fuselage, Bend the tail fins downward, Bend the tail fins upward, Widen the wingspan, Narrow the wingspan, Tilt the whole body, Mount small missiles on wings

- 6 Tank Rotate cannon to the side, Mount a telescope on the turret top

- 7 Character

Raise both hands, Raise left hand, Raise right hand, Hold a sword, Enlarge the head, Sit cross-legged, Wear a backpack, Wear a shoulder bag, Change to running pose, Grow a pair of wings, Stand on wind-fire wheels, Step on rocket launchers, Wear glasses, Wear a tall hat, Spread arms, High knee movement, Stand on one leg, Add a cape, Hold a shield, Grow a tail, Twist the waist, Stand on a skateboard, Change hairstyle to a bun, Enlarge the ears, Bend the elbows, Wear armor, Kneel on both legs, Cross both arms, Add halo above the head

- 8 Robot

Turn feet into wheels, Turn hands into bayonets, Wear an Iron Man helmet, Lengthen the arms, Mount mechanical wings on the back, Add antenna to the head, Add springs to the soles, Mount a rocket booster on the back, Lengthen the legs, Turn hands into cannons, Turn hands into claws, Turn arms into chainsaws, Add solar panels to the back, Transform into spider legs

- 9 Table

Put a vase on the table, Change table shape to round, Lay a tablecloth, Spiralshaped table legs, Add a drawer under the tabletop, Jagged edges on the tabletop, Dig a hole in the center, Put a cup on the table, Add wheels under table legs, Put a fruit plate on the table

- 10 Chair

Place a cushion, Extend the legs, Shorten the legs, Add wheels to the feet, Install a footrest, Place a seat cushion, Add storage bags on the sides, Put a speaker on it, Turn into a rocking chair

- 11 Cabinet

Add cabinet doors, Open the cabinet doors, Add drawers, Pull out a drawer, Put a table lamp on top, Add a lock, Add internal shelves, Place a potted plant on top Bowl: Change to square, Put an egg inside, Add a pair of chopsticks

- 12 Bed

Add a pillow, Change to round shape, Add bed curtains, Place a kitten on the bed, Convert into a bunk bed

- 13 Sofa Place a blanket, Place a teddy bear, Add a throw pillow

- 14 Bowl Change to square, Put an egg inside, Add a pair of chopsticks

- 15 Backpack Transform into a jetpack, Transform into a rolling backpack

- 16 Gun

Lengthen the barrel, Add barrels on both sides, Mount a scope on top, Add a magazine slot on the left, Attach a bayonet under the muzzle

- 17 Shoes

Extend the upper part, Thicken the sole, Attach wind-fire wheels

- 18 Clothes Convert to short-sleeve, Convert to long-sleeve, Add a scarf

- 19 Hat Raise the crown, Add wings to the sides, Turn the top into animal ears

- 20 Glasses Change to round frames, Add a head strap, Remove the frames

- 21 Ring Add a diamond, Remove the diamond

- 22 Knife Extend the blade, Turn into "Zangetsu" from Bleach

- 23 Sword

Lengthen the blade, Wrap the blade in flames, Make the blade serrated, Add a ring guard to the hilt, Embed gems in the blade

- 24 Teapot

Change the spout length, Open the lid, Turn the spout into a chainsaw, Add a heater at the bottom

- 25 Bottle

Only upper half remains, Insert a rose, Pour tea into the bottle, Replace cap with cork, Tie a label around the neck

- 25 Cup Turn into conical flask, Add a handle, Add a lid, Insert a straw, Add a cup heater

- 26 Cat

Jumping pose, Skating on a skateboard, Add a pair of wings, Wear clothes, Wear a bow on the head

- 27 Dog Hold a bone in mouth, Add a dog leash, Wear clothes, Wear a Christmas hat

- 28 Insect Remove wings, Remove antennae, Add an antenna, Add a pair of wings

- 29 Fish Wear goggles

- 30 Block-shaped Object Be stretched

- 31 Ball-shaped Object Change to oval

Table 8: Edited Prompt Collection: Part Two

ID Category Edited prompt

- 32 House

Add chimney on roof, Add and open a door, Change roof to dome, Change door to arch, Add canopy on the door, Add garage on the side, Add a balcony, Add a street lamp next to house, Add a fence, Add a mailbox at entrance, Install solar panels on roof

- 33 Tower

Shorten height, Add flag on top, Add door at base, Add spotlight at tip, Add fence around, Add antenna on top, Add spiral staircase outside, Add window in middle, Add vines on surface, Keep only lower half, Add observation deck at top

- 34 Tree

Grow two giant hands, Grow giant flowers on top, Grow stars at top, Grow two long legs, Grow large wings on sides, Butterfly perching on tree, Add a door on trunk, Hang lanterns on branches

- 35 Flower Add more petals, Insert into vase, Bee perching on it

- 36 Fruit Put in fruit plate, Peel skin, Insert small umbrella on surface

- 37 Vegetable Be stretched

- 38 Phone Turn into tri-fold screen, Add stylus on edge

- 39 Computer Grow wheels

- 40 TV Add two antennas, Install base stand

- 41 Keyboard Change to round keycaps

- 42 Book Grow two arms and legs, Grow wings

- 43 Building

Add arched entrance in front, Install antenna on roof, Add chimney on roof, Add external staircase, Add billboard on top, Helicopter parked on roof, Add fence in front, Make building round, Install solar panels on roof, Add flag on roof, Change door to revolving door, Add a clock on wall, Hang string lights on wall

- 44 Building Structure

Remove one column, Change to flat roof, Convert to castle top, Add cable support structure

- 45 Statue

Add a pair of wings, Wear sunglasses, Wear headphones, Wear a tall hat, Add halo above, Add fence around, Add multiple arms, Change head to Medusa, Wear a flower crown, Be wrapped in chains

- 46 Lamp

Change bulb to square, Change lampshade shape, Add more lamp heads, Change lamp head direction, Add hanging chains

- 47 Door

Replace rectangle door with arch, Add doorbell, Add surveillance camera, Add door lock, Add steps at entrance, Open the door, Wrap door with vines, Add peephole Bird: Claw grasping branch, Wings spread, Pecking downward, Lengthen beak, Shorten beak, Wear top hat, Hold a branch in beak, Wear goggles

- 48 Sculpture Wear crown, Wear armor, Wear mask, Hold scepter

- 49 Weapon Add hook at front, Make blade wavy, Change to double-headed, Be chained

- 50 Helmet Add goggles, Add visor, Change to pointed top, Unfold side wings

- 51 Bridge

Convert to suspension bridge, Add pillars, Make multi-level, Add street lights, Add toy cars

- 52 Vase Insert flowers, Place on table, Add handles on sides

- 53 Mechanical Arm Replace hand with clamp, Arm rotates

- 54 Plant Add fruits, Broken branches, Grow upwards

- 55 Shield Change to octagonal, Embed gem in center, Insert an arrow, Wrap in vines

- 56 Chest Be flattened, Open lid, Lock with chains

- 57 Airplane

Mount missiles under wings, Retract landing gear, Extend landing gear, Add more engines

- 58 Castle

Add drawbridge at entrance, Attach a dragon on wall, Connect towers with bridges, Hang flags on walls

- 59 Mythical Creature Add saddle, Grow spikes on back, Sleep curled on ground

- 60 Pillar Change to polygonal, Bend to one side, Add grooves to body

- 61 Tool Lengthen handle, Replace tool head with bayonet, Bend the handle

- 62 Lighthouse Add radar antenna on top, Add spiral staircase outside, Add window

- 63 Box Be flattened, Open the lid, Punch a hole

- 64 Monument Change top to pointed, Add flag on top, Add steps at base

- 65 Animal Grow antennae

- 66 Stairs Add more steps, Change to spiral stairs, Remove handrails

- 67 Tent Extend awning, Change to dome-shaped

- 68 Street Light Add signboard on pole, Add camera on pole

- 69 Trophy Add lid, Add handles

- 70 Machine Add wheels

Before Editing Action After

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

add storage bags on the sides

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

add wheels under table legs

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

change hairstyle to a bun

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

enlarge the ears

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Wear a tal hat

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

mount a scope on top

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

put a vase on the table

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

turn the spout into a chainsaw

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Grow a pair of wings

###### Figure 11: Some cases of our 3D-Editing Data

###### Input 3D Mesh Output

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

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

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

###### Input 3D Mesh Output

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

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

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

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

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

Figure 13: More cases of Image-to-3D result from our method.

##### Input 3D Mesh Output

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

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

#### Input 3D Mesh Output

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

A truck

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

A sword with a hilt and a pointed tip

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

A smal house with a snowcovered roof

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

A pair of pants with a tied waistband

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

A large building with a covered porch and multiple doors

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

A handgun with a suppressor

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

A handgun with a grip and barrel

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

A guitar with a pickguard and bridge

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

A hammer with a handle and head

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

A building with a staircase and a balcony

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

A chair with a curved backrest and four legs

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

A female figure in a futuristic outfit with high heels

###### Figure 15: More cases of Text-to-3D result from our method.

