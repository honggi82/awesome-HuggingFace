# arXiv:2606.06042v2[cs.CV]5Jun2026

[Figure 1]

[Figure 2]

## LoomVideo: Unifying Multimodal Inputs into Video Generation and Editing

### Peking University, Alibaba Group

[Figure 3]

Project Page: https://msalab-pku.github.io/projects/LoomVideo/index.html Github: https://github.com/MSALab-PKU/LoomVideo Model: https://huggingface.co/MSALab/LoomVideo

[Figure 4]

### Abstract

Developing unified video generation and editing models capable of interpreting interleaved multimodal inputs is a promising yet challenging frontier field. Existing unified frameworks predominantly rely on massive models (typically 13B parameters or more) and incorporate source video conditions for editing by concatenating sequence tokens. This concatenation inevitably doubles the sequence length, quadrupling the computational complexity of the self-attention mechanism and introducing prohibitive overhead. To address these bottlenecks, we present LoomVideo, a highly efficient 5B-parameter unified architecture for both video generation and editing. LoomVideo replaces the standard text encoder with a Multimodal Large Language Model (MLLM) and employs a Deepstack injection mechanism to align multi-layer MLLM features with the Diffusion Transformer (DiT). Crucially, we introduce a zero-overhead Scale-and-Add conditioning approach for video editing. By scaling and directly adding the clean source video latent to the noised target latent, this elegant design eliminates the need for token concatenation, drastically reducing computational cost while maintaining robust capabilities for complex, non-rigid edits. Furthermore, a Negative Temporal RoPE strategy is seamlessly integrated to handle multiple reference images. Extensive experiments demonstrate that our compact 5B model achieves state-of-the-art or highly competitive performance across comprehensive benchmarks, exhibiting exceptional superiority in e-commerce and fashion generation scenarios. Benefiting from the zero-overhead conditioning mechanism, LoomVideo achieves at least a 5.41× acceleration in inference speed compared to models of similar capabilities, paving the way for highly practical and efficient video foundation models.

### 1 Introduction

The demand for versatile and highly controllable video generation and editing is rapidly increasing across diverse practical applications, from digital entertainment to e-commerce. To meet these complex requirements, developing a unified architecture capable of interpreting and adhering to interleaved multimodal inputs (e.g., video, images, and fine-grained instructions) is of paramount importance. The paradigm of unifying multiple vision-language tasks within a single foundational model first achieves remarkable success in the image domain, evidenced by pioneering works such as Qwen-Image Wu et al. (2025b) and the OmniGen series Xiao et al. (2025); Wu et al. (2025c). Recently, this unified modeling approach has been extended to the video domain, though existing works remain relatively sparse due to the inherent complexity of spatiotemporal dynamics.

A few pioneering works have explored unified video generation and editing, including UniVideo Wei et al. (2025), OmniWeaving Pan et al. (2026), VINO Chen et al. (2026b), and OmniVideo Tan et al. (2025); Yang et al. (2026). While these frameworks have demonstrated the feasibility of multimodal conditioning, they predominantly suffer from two major limitations. First, most of these frameworks rely heavily on massive base models (typically 13B parameters or more), making them exceptionally resource-intensive. Second, to incorporate source video condisource-video conditions for editing tasks, the prevailing practice is to concatenate the source-video tokens with the target-videosequence dimension. This design inevitably doubles the token sequence length, which in turn quadruples the computational complexity of the self-attention mechanism. Consequently, these approaches introduce prohibitive computational overhead and significantly elevate both training and inference costs.

To address these bottlenecks, we introduce LoomVideo, an efficient unified architecture for video generation and editing. Built upon the foundation of the 5B-parameter Wan 2.2 Text-Image-to-Video (TI2V) model Wan et al. (2025), we replace its standard T5 text encoder with Qwen3-VL Bai et al. (2025b), a powerful Multimodal Large Language Model (MLLM), to handle interleaved multimodal inputs. To maximize multimodal interaction and minimize computational overhead, we incorporate three key effective architectural designs. First, instead of solely utilizing the final-layer embedding of the vision-language model, we employ a Deepstack injection mechanism.

Text-to-Video

Prompt: A jellyfish floating through the ocean, with bioluminescent tentacles.

Prompt: Snow rocky mountains peaks canyon. ..The canyons twist and bend through the high elevated mountain peaks.

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Instruction Editing

Prompt: Replace the man's casual green t-shirt with a sharp dark navy blue business suit, white shirt, and burgundy tie, maintaining the same position and pose within the scene.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Prompt: Apply the Gongbi animation style to this video, ensuring seamless temporal consistency. The result should mirror the elegance of traditional Chinese ink painting, with fluid brushstroke transitions…

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Instruction-Image Editing

Prompt: Replace the background with a serene anime-style landscape, featuring a massive, fluffy cumulonimbus cloud towering over a windswept grassy hill…

Ref Image

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Prompt: Add the woman in the image to stand to the left of the man and look at the man.

Ref Image

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Multi-Image-to-Video

Prompt: The girl (@Image 2), wearing the denim jacket (@Image 3), black inner top, and black shorts, wearing sunglasses and carrying the handbag, walks down the street (@Image 1). Then, the girl (@Image 2) stops walking and turns her head to look to one side, followed by the girl (@Image 2) crossing her arms over her chest and striking a confident pose.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Image1 Image2

Image3

Prompt: Generate a video with reference images: The girl (@Image 2) wearing a Chinese-style vest with frog buttons, top, and skirt (@Image 3), holding a handbag, standing in front of the background (@Image 1).

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Image1 Image2 Image3

- Figure 1: Showcase of LoomVideo across diverse video generation and editing scenarios, including foundational video generation & editing, reference-image-guided video generation and editing.

We extract feature embeddings from every layer of Qwen3-VL and inject them into the corresponding layers of the Diffusion Transformer (DiT) via cross-attention, thereby enhancing the deep semantic alignment between the multimodal controls and the generative process. Second, to overcome the severe efficiency bottleneck caused by token concatenation in editing tasks, we utilize a Scale-and-Add conditioning approach. Rather than appending source tokens, we simply scale the clean source video latent by the current timestep and add it directly to the noised target latent. This elegant design introduces zero additional tokens to the sequence length, drastically reducing generation time. More importantly, we demonstrate that this lightweight conditioning mechanism is fully capable of driving highly complex, non-rigid editing operations (e.g., changing human actions or camera angles). Third, to seamlessly integrate multiple reference images, we employ a Negative Temporal RoPE index strategy. This positional encoding effectively distinguishes the reference inputs from the video frames, providing robust multi-image guidance without disrupting the spatiotemporal dynamics of the generative process.

We devise a progressive three-stage training strategy: Stage 1 focuses on semantic alignment at a low resolution, where we extract and inject MLLM embeddings into the DiT. Stage 2 scales up the resolution, training simultaneously on fundamental image/video generation and reconstruction/editing tasks. Stage 3 incorporates diverse reference images and advanced text instructions to drive complex generation and editing tasks. Additionally, we further enhance the model’s overall performance via reinforcement learning post-training, which effectively improves its instruction-following capability and generation fidelity. Through this progressive multimodal alignment, task adaptation, and targeted post-training, our relatively compact 5B model efficiently performs various generation and editing tasks that previously required 13B-scale models, unifying them within a single framework.

In summary, the main contributions of this work are as follows:

- • We propose LoomVideo, a highly efficient 5B-parameter unified architecture that seamlessly integrates visual-language multimodal input controls for both video generation and editing.
- • Our method achieves state-of-the-art (SOTA) or on-par SOTA performance across multiple comprehensive benchmarks. It demonstrates particularly superior capabilities in reference-image-guided video editing and controllable generation tasks within e-commerce and fashion product domains.
- • Benefiting from the compact parameter scale and the zero-overhead Scale-and-Add conditioning mechanism, LoomVideo achieves at least 5.41× acceleration in inference speed compared to existing concatenationbased unified models of similar capabilities.

### 2 Related Work

Video Generation and Editing. The landscape of text-to-video generation has been reshaped by diffusion models. This evolution has catalyzed a wave of highly popular and powerful foundational video models, such as CogVideoX Yang et al. (2024), Wan Wan et al. (2025), and the HunyuanVideo series Kong et al. (2024); Wu et al. (2025a), which have garnered widespread attention and extensive open-source community support. Recently, stateof-the-art models like Seedance 2.0 Seedance et al. (2026), SkyReelsV4 Chen et al. (2026a), and Sora 2 OpenAI

- (2024) have pushed the boundaries of visual quality, physical world complexity, and generation duration. Building upon robust video generation priors, video editing has naturally emerged as the next frontier. Early video editing methods predominantly focused on zero-shot adaptations or rigid structural preservation, which often struggled with complex, non-rigid transformations. The introduction of instruction-following and unified frameworks has significantly advanced this field. Models like InsViE Wu et al. (2025d), Ditto Bai et al. (2025a), OpenVE He et al.
- (2025), and Kiwi-Edit Lin et al. (2026) have demonstrated the feasibility of performing diverse editing tasks via instruction and reference guidance. Despite these remarkable strides, most existing unified models either rely on massive parameter scales (13B or more). In contrast to these approaches, our work demonstrates that a relatively compact 5B-scale Diffusion Transformer (coupled with an 8B VLM) can achieve highly efficient and high-quality multi-task unification. Particularly in complex e-commerce and fashion generation scenarios.

Multimodal Input for Video Generation. As the demand for precise control over video generation and editing grows, the field is rapidly moving beyond pure text-to-video generation toward multimodal conditioning. Current prominent frameworks, such as VINO Chen et al. (2026b), VACE Jiang et al. (2025), OmniWeaving Pan et al. (2026) , UniVideo Wei et al. (2025), and Omni-Video Tan et al. (2025); Yang et al. (2026), have explored various ways to fuse interleaved multimodal inputs into video generation. For the input video, existing methods often concatenate the source video tokens with the target tokens Wei et al. (2025); Pan et al. (2026). While effective, this design inevitably leads to a drastic increase in token sequence length, resulting in severe computational bottlenecks. Our architecture takes a more efficient approach to multimodal conditioning. 1) To achieve deep semantic alignment between the multimodal input and the generative prior, we do not merely use the final output of the VLM. Instead, we extract the feature embeddings from every layer form the VLM and inject them into the corresponding layers of the DiT via cross-attention. 2) When incorporating source video conditions for editing tasks, we simply scale and add the clean source video latent to the target noised video latent. This elegant design introduces zero additional tokens to the sequence length, ensuring optimal computational efficiency. Surprisingly, despite this lightweight

Hidden States for Layer 𝑁𝑢𝑛𝑑 …

DiT

- Hidden States for Layer 1

MLP

Cross-Attention

KV

- Hidden States for Layer 2

Extract Each Layer’s Output

Temporal RoPE Indices

… …

-2𝜏

-𝜏

0 1 2 𝑇

|Merged Embedding|
|---|

Negative Temporal RoPE Index

+

×

…References for the male's face (@Figure 2), full-body physique (@Figure 2), and hairstyle (@Figure 2); brown jacket (@Figure 3); beige trousers (@Figure 4); and the

###### MLLM

+

Timestep

| |Scale|
|---|---|
|Source Video Latent| |

|Target Video<br><br>Latent|
|---|

|Noise|
|---|

|Ref. 2|
|---|

|Ref. 1|
|---|

exhibition hall (@Figure

1).The male (@Figure 2) wearing a brown jacket (@Figure 3) and beige trousers (@Figure 4), holding a scarf, stands in the center of the exhibition hall (@Figure 1), turning to show

VAE Ref. Image(s) Source Video Instruction Ref. Image(s) Source Video Target Video

ViT Text Tokenizer

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Replace the coffee cup with a patterned teacup

[Figure 59]

[Figure 60]

[Figure 61]

details of the back of

the clothing.

Example Instruction with

Multiple Reference Images

- Figure 2: Overall Architecture of LoomVideo. LoomVideo seamlessly processes interleaved multimodal inputs using an MLLM. It employs two key designs: (1) A Deepstack injection mechanism, which extracts feature embeddings from every layer of the MLLM and injects them into the corresponding layers of the DiT via crossattention. (2) A zero-overhead Scale-and-Add conditioning approach for video editing, which scales the clean source video latent by the current timestep and directly adds it to the noised target latent, completely bypassing the severe inefficiency of token concatenation. (3) Negative Temporal RoPE index for multiple reference images.

conditioning scheme, our model is fully capable of executing complex, non-rigid editing tasks (e.g., changing human actions or camera directions).

- 3 Method

#### 3.1 Architecture

In this section, we introduce the architecture and key designs of LoomVideo, as illustrated in fig. 2.

Deepstack mechanism of MLLM features. To handle interleaved multimodal inputs (e.g., text, reference images, and source videos), we replace the default T5 text encoder with an MLLM. However, relying solely on the final-layer embedding of the MLLM restricts the generative prior from fully utilizing the rich, hierarchical semantic information captured at different layers. To maximize multimodal interaction, we propose a Deepstack injection mechanism. Rather than using only the final layer’s output, we extract hidden states from every layer of the MLLM. These representations are transformed by a newly added MLP module and then injected into the corresponding layers of the DiT via cross-attention, serving as key and value. This process can be written as:

cl = MLP(hmllml ), oditl = CrossAttn(hditl ,cl,cl), (1)

where hmllml ) denotes the hidden states extracted from the l-th layer of the MLLM, and hditl represents the intermediate features of the corresponding l-th layer in the DiT. The MLP module projects the MLLM hidden states into the

condition features cl, which are then served as keys and values in the cross-attention module. The MLP module shares parameters across layers. This layer-to-layer conditioning approach guarantees deep semantic alignment between the fine-grained multimodal controls and the spatiotemporal generative process, enhancing the model’s instruction-following capabilities without introducing heavy adapter networks.

Scale-and-Add of Source Video Latent. Existing unified video generation and editing frameworks predominantly incorporate source video conditions by concatenating the source video tokens with the target video tokens along the sequence dimension. This paradigm inevitably doubles the token sequence length, quadrupling the computational complexity of the self-attention mechanism and causing severe efficiency bottlenecks during both training and inference. To overcome this limitation, LoomVideo introduces a zero-overhead Scale-and-Add conditioning approach. Instead of appending source tokens, we directly manipulate the latent representations. Specifically, we scale the clean source video latent by a timestep-dependent factor and add it directly to the noised target latent at the current timestep. This merged embedding is then fed into the DiT. This process can be formulated as:

hmerged = ϕ(ztarget) + t · ϕ′(zsource), (2)

where ztarget denotes the noised target video latent at the current timestep t ∈ [0,1), and zsource represents the clean latent of the source video. Here, the timestep t directly acts as the scaling factor. As t increases, the scale

factor proportionally amplifies the contribution of zsource, ensuring stronger guidance during the early stages of the denoising process, while leaving purer target latent during later steps. ϕ and ϕ′ are the original and newly added patch embedding layers, respectively. ϕ′ is zero-initialized. This lightweight conditioning introduces zero additional tokens to the sequence length, drastically reducing the generation time (achieving at least 5.41× acceleration in inference speed compared to concatenation-based models). Despite its simplicity, empirical results in section 4 demonstrate that this mechanism provides sufficient guidance to drive highly complex, non-rigid editing operations, such as modifying human actions or camera angles, while preserving optimal computational efficiency.

Negative Temporal RoPE Index for Reference Images. Target video frames in DiT are assigned normal positive temporal RoPE index 0,1,2, . . . , T, where T is the number of latent frames. For the reference images, we assign them negative temporal indices −τ, −2τ, . . . , −Nrefτ, where τ is a hyperparameter, and Nref is the number of reference images. By doing so, the model robustly distinguishes the reference images from the target video frames, providing strong multi-image guidance with minimal extra tokens. Furthermore, when the task involves detailed textual descriptions of multiple reference images (e.g., “the person in @Figure 1”, “the clothes in @Figure 2”), this negative indexing strategy enables the model to explicitly align the i-th image specified in the prompt with its corresponding visual representation based on the absolute value of the assigned negative RoPE index.

#### 3.2 Training Data

We train LoomVideo using a mixture of image and video datasets. Although our primary objective is unified video generation and editing, incorporating a massive and semantically diverse collection of image generation and editing data effectively bolsters the foundational capabilities of the model.

Text-to-Image/Video. For text-to-image generation, we utilize approximately O(10M) high-quality internal image-text pairs. Since we replace the default T5 text encoder with the Qwen3-VL MLLM, establishing accurate multimodal alignment from scratch is crutial. Our internal dataset features exceptionally detailed and accurate textual descriptions across a diverse range of visual domains, which significantly aids in maintaining robust textimage alignment performance. For text-to-video generation, we employ around O(10M) open-source video-text pairs, primarily sourced from Koala 36M Wang et al. (2025a) and OpenVid-1M Nan et al. (2024). This large-scale temporal data is sufficient to seamlessly extend the fundamental text-image alignment capabilities learned in the static domain to the video domain with complex spatiotemporal dynamics.

Instruction-based Image/Video Editing. For instruction-based image editing, we curate a comprehensive blend of open-source and internal datasets. The open-source collection comprises approximately O(10M) samples from various high-quality datasets, including SEED-Data-Edit part 2 and 3 Ge et al. (2024), NHR-Edit Kuprashevich et al. (2026), OmniGen2 Wu et al. (2025c), Pico-Banana-400k Qian et al. (2025), CrispEdit-2M Chow et al. (2025), ShareGPT-4o-Image Chen et al. (2025a), and GPT-Image-Edit-1.5M Wang et al. (2025b). We further incorporate O(10M) internal editing samples specifically focusing on the Taobao fashion domain. For instruction-based video editing, our training set consists of O(3M) samples from the open-source Kiwi-Edit dataset Lin et al. (2026), supplemented by O(1M) internal instruction-guided video editing pairs.

Instruction-and-Reference-based Image/Video Editing. To equip the model with fine-grained visual condition following capabilities, we incorporate reference-guided editing tasks. For the image domain, we predominantly rely on an internal dataset of O(10M) instruction-and-reference pairs. For the video domain, we utilize approximately O(0.5M) reference-guided video editing samples from the RefVIE dataset Lin et al. (2026), alongside O(1M) internal samples tailored for reference-based video editing.

Multiple-Reference-Image-to-Image/Video. For tasks requiring multiple reference images, we utilize the opensource Phantom dataset Chen et al. (2025b) alongside a large-scale internal collection of multi-reference image and video data, particularly emphasizing the Taobao e-commerce domain. To ensure the model explicitly distinguishes and follows instructions involving multiple reference images, we adopt an interleaved text-image prompt format inspired by Seedance 2.0 Seedance et al. (2026). Specifically, we use explicit textual pointers such as “The person in @Figure 1” and “The clothes in @Figure 2” to bind textual descriptions with their corresponding visual reference inputs. An illustrative example of this interleaved multi-reference instruction format is provided in fig. 2.

#### 3.3 Training Pipeline

- Stage1: MLLM Alignment. To enable our model to seamlessly process interleaved text, image, and video inputs, we must leverage the powerful multimodal understanding capabilities of the MLLM. The primary objective of the first stage is to replace the original condition encoder (i.e., the T5 text encoder) of the pretrained DiT with the MLLM. In this stage, we exclusively utilize text-to-image and text-to-video datasets. To ensure the generalizability of semantic alignment, we employ a large batch size (approximately 640) while maintaining a relatively low spatial resolution of 256p. The sample-level ratio of image to video data is set to 4:1. It is worth noting that since a single video contains significantly more tokens than an image, and we adopt a dynamic 1D sequence concatenation

Table 1: Training recipe of LoomVideo.

Stage1 Stage2 Stage3

Hyperparameters

MLLM Alignment Reconstruction & Editing Multi Task

Learning rate 2.0 × 10−5 2.0 × 10−5 2.0 × 10−5 LR scheduler Constant Constant Constant Weight decay 0.0 0.0 0.0 Gradient norm clip 1.0 1.0 1.0 Optimizer AdamW AdamW AdamW GPUs 16× H20s 64× H20s 64× H20s Batch Size ∼ 640 ∼ 128 ∼ 128 Warm-up steps 100 1000 1000 Training steps 50K 40K 36K EMA weight decay None 0.9999 0.9999 EMA interval None 1 1 Resolution 256p 480p 480p Text dropout ratio 0 0.1 0.1 All dropout ratio 0 0.1 0.1 Dynamic timestep shift scale 0.65 0.5 0.5

###### Data sampling ratio

Text-to-Image 0.80 0.59 0.14 Text-to-Video 0.20 0.04 0.08 Image Reconstruction 0.00 0.07 0.02 Video Reconstruction 0.00 0.01 0.01 Text-Image-to-Image 0.00 0.28 0.17 Text-Video-to-Video 0.00 0.01 0.18 Text-Image-Ref-to-Image 0.00 0.00 0.15 Text-Video-Ref-to-Video 0.00 0.00 0.13 Text-Multi-Image-to-Image 0.00 0.00 0.06 Text-Multi-Image-to-Video 0.00 0.00 0.07

batching strategy bounded by a maximum token limit, the actual proportion of video tokens in a batch is substantially higher than that of image tokens.

Furthermore, we ablated to retain the T5 text encoder, feeding both the T5 embeddings and the newly introduced MLLM embeddings into the DiT via cross-attention. Although this approach converged faster initially, we empirically observed in subsequent stages that the model heavily relied on the T5 embeddings and largely ignored the MLLM conditionings. Consequently, to force the generative prior to fully align with the MLLM representations, the T5 encoder must be completely discarded.

- Stage2: Reconstruction and Editing. Following the first stage, the model demonstrates robust semantic alignment with the MLLM. However, its visual generation quality remains limited due to the low training resolution, often resulting in blurry details in intricate regions such as human faces. Therefore, the primary objective of the second stage is to scale up the resolution (to 480p) and enhance generation quality. Simultaneously, we introduce a portion of instruction-based editing data to facilitate a smooth transition toward the multi-task training in the final stage. Importantly, we also introduce a reconstruction task utilizing the text-to-image and text-to-video datasets, prompted simply by “reconstruct this image/video”. This design is motivated by the intuition that reconstruction serves as the fundamental prerequisite for editing; robust and accurate video editing capabilities must be built upon the model’s ability to precisely reconstruct the source visual content.
- Stage3: Multi Task. In the final stage, we incorporate the full spectrum of our datasets to perform comprehensive multi-task fine-tuning. During this phase, the data sampling distribution is deliberately skewed toward newly introduced and more challenging tasks, such as reference-guided image/video editing and multi-reference generation. This targeted sampling strategy ensures that the model devotes sufficient capacity to mastering complex, fine-grained control and non-rigid transformations.

Post-training: Reinforcement Learning (RL). Current RL methods Liu et al. (2025a); Tong et al. (2026); Liu et al. (2026); Zheng et al. (2025) have demonstrated promising performance in visual generation tasks. We adopt DiffusionNFT Zheng et al. (2025) for our experiments, employing PickScore Kirstain et al. (2023) as the reward model to optimize human aesthetic preferences and perceptual quality. The training data is curated from our in-house dataset. We follow the FlowFactory Ping et al. (2026) implementation for the training pipeline setup.

- 3.4 Implementation Details Model Initialization and Architecture. We initialize our model with the pretrained weights of Wan 2.2 TI2V
- 5B Wan et al. (2025) and Qwen3-VL-8B-Instruct Bai et al. (2025b). Because the Qwen3-VL-8B model consists of 36 transformer layers while the DiT contains only 30 layers, we extract the hidden states from the last 30 layers of the MLLM and inject them into the corresponding 30 layers of the DiT. To transform these MLLM hidden states for cross-attention, our newly introduced MLP module follows the architectural design of the original T5 text projection layer in DiT, which comprises two linear layers and an activation function. Notably, to ensure training stability, we prepend an RMSNorm layer at the beginning of this module to properly normalize the extracted Qwen3-VL hidden states before projection.

Dynamic Batching and Multi-Resolution Training. During training, we employ a 1D sequence concatenation batching strategy. Specifically, all spatial and temporal tokens from the images and videos within a batch are flattened into 1D sequences and concatenated together. We apply block-diagonal attention masks to strictly prevent tokens from different samples within the same batch from attending to one another. Under this paradigm, the batch size is dynamically constrained by a predefined maximum total token limit rather than a fixed number of samples, meaning the batch sizes reported in our settings are empirical averages. Furthermore, we adopt a dynamic multi-resolution training approach. Images and videos are grouped into predefined buckets based on their total pixel counts and aspect ratios, and are then resized to the nearest bucket resolution. By flattening and concatenating these dynamically sized samples, our framework natively supports efficient multi-resolution training.

Dynamic Timestep Shift Scale. Since our training paradigm involves a mixture of multi-resolution images and videos, we adopt the dynamic timestep shift technique recommended by Stable Diffusion 3 (SD3) Esser et al. (2024). This technique dynamically adjusts the noise scheduling by increasing the timestep shift alongside the growth of data resolution and temporal length. Specifically, the shift value is calculated as follows:

Ntarget Nbase

shift = scale × log2

(3)

where Ntarget denotes the total token count of the target sample, and Nbase is the base token number. In our implementation, we set Nbase = 64, which exactly corresponds to the sequence length of a standard 256 × 256 image after VAE encoding and patchification. Under this formulation, by setting the hyperparameter scale = 0.5, the calculated shift value for a high-resolution video (e.g., 480 × 832 × 97) automatically reaches approximately 4.5. This adaptive scaling ensures optimal noise scheduling across diverse input dimensions and modalities during joint training.

Training Configurations. Regarding the training resolutions listed in our recipe, “256p” denotes the bucket with a standard video dimension of 256 × 256 × 33 frames (and its corresponding 1-frame image counterpart), while “480p” refers to a standard dimension of 480 × 832 × 97 frames for videos (and corresponding 1-frame images). Finally, to maintain robust training stability and achieve better convergence, we maintain an Exponential Moving Average (EMA) of the model weights exclusively during Stage 2 and Stage 3. The detailed hyperparameters and specific data sampling ratios across all three stages are summarized in table 1.

### 4 Experiments

#### 4.1 Experimental Settings

Benchmarks. For the Text-to-Video (T2V) generation task, we employ VBench Huang et al. (2024), selecting Imaging Quality, Overall Consistency, and Subject Consistency as our primary evaluation metrics, and we report their average scores. For instruction-based video editing, we adopt OpenVE-Bench He et al. (2025). For instruction-andreference-based video editing, we utilize RefVIE-Bench Lin et al. (2026). To assess multi-reference image-to-video (MI2V) capabilities, we evaluate our model on IntelligentVBench Pan et al. (2026), with a specific focus on its TIV2V and MI2V sub-tasks.

Furthermore, to comprehensively demonstrate our model’s specialized capabilities in the e-commerce domain, we construct a novel benchmark, FashionVideoBench. Curated from internal, held-out data (strictly excluded from the training set), this benchmark encompasses six core tasks: Product Edit, Model Edit, Freeform Edit, Product-Reference Edit (PRef Edit), Model-Reference Edit (MRef Edit), and MI2V. Specifically, we sample 50 test cases for each sub-task, resulting in a total of 300 evaluation samples. We employ Gemini 2.5 Pro as an automated judge to assess the generated outputs across three dimensions: Subject Consistency, Prompt Following, and Video Quality. Detailed evaluation protocols and prompt templates are provided in Appendix C.

Baseline Methods. We primarily benchmark our approach against state-of-the-art open-source unified video generation and editing models, including UniVideo Wei et al. (2025), OmniWeaving Pan et al. (2026), VINO Chen et al. (2026b), and VACE Jiang et al. (2025). Depending on the task, we also compare against several specialized models. This includes editing-specific models such as Kiwi-Edit Lin et al. (2026), Ditto Bai et al. (2025a), OpenVEEdit He et al. (2025), InsViE Wu et al. (2025d), and OmniVideo Tan et al. (2025), as well as prominent foundational

Table 2: Quantitative results on VBench.

Model #Params Imaging Quality Overall Consistency Subject Consistency Average

Wan 2.2 5B 69.82 22.41 95.28 62.50 UniVideo 13B 69.34 22.62 97.08 63.01 OmniWeaving 8.3B 61.78 22.46 94.12 59.45

LoomVideo (Stage 3) 5B 67.13 23.74 94.60 61.82 LoomVideo (RL) 5B 70.92 23.59 94.93 63.15

Table 3: Quantitative results on OpenVE-Bench. We re-evaluate methods with † for all seven criteria.

Model #Params Global Style Background Change Local Change Local Remove Local Add Subtitle Edit Creative Edit Overall Specialized Video Editing Models

OmniVideo 1.3B 1.11 1.18 1.14 1.14 1.36 1.00 2.26 1.31 InsViE 2B 2.20 1.06 1.48 1.36 1.17 2.18 2.02 1.64 Ditto 14B 4.01 1.68 2.03 1.53 1.41 2.81 1.23 2.10 OpenVE-Edit 5B 3.16 2.36 2.98 1.85 2.15 2.91 2.31 2.53 Kiwi-Edit† 5B 3.62 2.57 3.76 3.36 2.57 2.91 3.08 3.12

Unified Video Generation Models

VACE 14B 1.49 1.55 2.07 1.46 1.26 1.48 1.47 1.54 VINO† 13B 3.95 2.39 3.51 3.20 2.68 2.65 3.01 3.07 UniVideo† 13B 3.47 2.58 3.41 2.99 2.83 2.87 3.07 3.05 OmniWeaving† 8.3B 3.68 2.16 3.78 2.68 1.83 3.48 2.8 2.92

- LoomVideo (Stage 2) 5B 3.81 2.46 3.04 3.33 2.21 3.64 3.54 3.15

- LoomVideo (Stage 3) 5B 3.62 2.26 3.32 2.82 2.40 3.30 2.86 2.94 LoomVideo (RL) 5B 3.85 2.37 3.41 3.12 2.19 3.42 3.23 3.05

video generation models like Wan Wan et al. (2025), HunyuanVideo Kong et al. (2024); Wu et al. (2025a), CogVideoX Yang et al. (2024), SkyReels-A2 Fei et al. (2025), SkyReels-V3 Li et al. (2026), and Phantom Liu et al. (2025b). In addition, for certain benchmarks, we include comparisons with leading closed-source commercial models Runway ML (2025); Team et al. (2025). Importantly, most state-of-the-art models in this domain operate with massive parameter counts (typically 13B or larger), which are significantly larger than our efficient 5B-parameter architecture.

#### 4.2 Quantitative Comparison

Results on VBench. As presented in table 2, our fully trained LoomVideo outperforms the foundational Wan 2.2 model in terms of average score and obtains the highest scores on both Imaging Quality and Overall Consistency. This demonstrates the effectiveness of our training strategy to completely replace the standard T5 text encoder with Qwen3-VL, thereby fully exploiting the rich, hierarchical multimodal alignment capabilities of the MLLM. Furthermore, it verifies that our unified architecture delivers consistent enhancements in both video generation and editing capabilities without incurring any performance degradation.

Results on OpenVE-Bench. For instruction-based video editing, we evaluate our model on OpenVE-Bench, with results detailed in table 3. Our Stage 2 model, which primarily focuses on reconstruction and editing tasks, achieves the highest overall score. This demonstrates that our compact 5B LoomVideo can achieve state-of-the-art editing performance, underscoring its strong practicality and efficiency. Specifically, our model excels in the “Creative Edit” metric, indicating exceptional semantic alignment with the MLLM and a robust ability to accurately recognize and execute diverse editing intents. While our final model experiences a slight performance drop on pure instruction-based editing due to the integration of more diverse and challenging multi-modal tasks, it still remains competitive with larger-parameter baselines.

Results on RefVIE-Bench. We further evaluate our model’s performance on instruction-and-reference-guided video editing (incorporating a single subject or background reference image) using RefVIE-Bench. As shown in

- table 4, LoomVideo achieves the highest overall score among all open-source baseline methods, outperforming the second-place VINO by a significant margin of 7%. This substantial improvement highlights our model’s robust capability and precision in handling fine-grained visual conditions.

Results on IntelligentVBench. To assess our model’s multimodal conditioning capabilities, we report results on the Text-Image-to-Video (TIV2V) and Multi-Image-to-Video (MI2V) sub-tasks of IntelligentVBench, detailed in

- table 5 and table 6. The TIV2V task shares inherent similarities with RefVIE, and our 5B model consistently secures the top position, outperforming the second-best OmniWeaving by 8%. For the highly complex compositional MI2V task, LoomVideo achieves performance comparable to the SOTA UniVideo (query) model. We attribute this competitive but non-leading performance primarily to the inherent capacity limits of our 5B-parameter scale, as well as to domain shift, given that our MI2V training data predominantly consists of Taobao e-commerce scenarios rather than the open-domain distribution of IntelligentVBench.

Table 4: Quantitative comparison on RefVIE-Bench. The top and the second-best scores are only compared among open-source models.

Model Subject Reference Background Reference Overall

Identity Temporal Physical Reference Sim Matting Quality Video Quality Closed-Source Models

Runway Aleph 3.79 3.65 3.58 3.33 2.81 2.58 3.29 Kling-O1 4.75 4.66 4.60 3.95 3.21 2.75 3.99

Open-Source Models

Kiwi-Edit (All data) 3.51 2.96 2.91 3.40 2.58 2.40 2.96 Kiwi-Edit (Ref. data only) 3.98 3.40 3.34 3.72 2.90 2.51 3.31 VINO 4.18 4.03 3.74 2.93 2.60 2.40 3.53 UniVideo 4.19 3.80 3.61 2.90 2.22 2.12 3.38 OmniWeaving 3.29 2.96 2.82 3.45 2.55 2.35 2.94

LoomVideo (Stage 3) 4.29 3.90 2.72 3.75 2.65 2.38 3.62 LoomVideo (RL) 4.50 3.98 3.90 3.88 2.90 2.50 3.78

Table 5: Quantitative comparison of the TIV2V task in IntelligentVBench.

###### TIV2V IF ↑ CP ↑ VQ ↑ AVG

Model #Params

VACE-Wan2.1 14B 1.46 1.42 1.71 1.53 VACE-LTX 2B 1.43 1.36 1.25 1.35 VINO 13B 2.86 2.90 2.52 2.76 UniVideo (query) 13B 3.22 3.91 3.26 3.46 UniVideo (hidden) 13B 3.13 4.01 2.93 3.36 OmniWeaving 8.3B 4.00 4.04 3.65 3.89

LoomVideo (Stage 3) 5B 4.35 4.08 3.99 4.14 LoomVideo (RL) 5B 4.39 4.23 4.08 4.24

Results on FashionVideoBench. Recognizing that existing methods primarily focus on general domains and lack specialized evaluations for e-commerce applications, we introduce FashionVideoBench, constructed entirely from held-out internal data. As demonstrated in table 7, LoomVideo consistently achieves the highest overall scores across all six core sub-tasks compared to open-source counterparts. This comprehensive superiority strongly validates our model’s exceptional practicality, controllability, and generation quality in complex e-commerce and fashion generation scenarios.

Ablation Study on Reinforcement Learning Post-Training. To validate the necessity of RL post-training, we conduct comprehensive comparisons between our Stage 3 model and the final model across all evaluation benchmarks. As evidenced by the quantitative results presented in all aforementioned tables, our final model achieves significant performance improvements on all benchmarks except FashionVideoBench, where our model has already attained a high level of editing quality with limited room for further enhancement. This result strongly validates that RL post-training effectively boosts the model’s video generation and editing capabilities, serving as a critical component for our training pipeline.

#### 4.3 Qualitative Comparison

As illustrated in fig. 3, we qualitatively compare four methods, including Kiwi-Edit, VINO, UniVideo and OmniWeaving. In the traditional video editing tasks on OpenVE-Bench, our model exhibits superior instruction following capability and editing quality compared with all baseline methods. For the reference-guided editing tasks on RefVIE-Bench, our model can more accurately restore reference images and produce high-fidelity editing results. In the MI2V tasks on FashionVideoBench, our model achieves precise restoration of each reference image while maintaining favorable overall video generation quality. More qualitative results of our method are provided in fig. 5, fig. 6, fig. 7, and fig. 8 in Appendix A.

#### 4.4 Efficiency Comparison

As shown in table 8, our model achieves significantly faster inference speeds compared to counterpart models across both T2V generation and video editing tasks. Compared to the fastest baseline, OmniWeaving, LoomVideo demonstrates a remarkable speedup of 6.24× for T2V generation and 5.41× for video editing. Furthermore, for models such as UniVideo and VINO that rely on the token concatenation strategy, the inference time required to edit a single video is approximately four times their respective T2V generation time. This observation aligns with our analysis that token concatenation inherently quadruples the computational complexity of the self-attention

Table 6: Quantitative comparison of the Compositional MI2V task in IntelligentVBench.

###### 1Subject (with BKG) 2Subjects (with BKG) 3Subjects (with BKG) IF ↑ CP ↑ VQ ↑ AVG IF ↑ CP ↑ VQ ↑ AVG IF ↑ CP ↑ VQ ↑ AVG

Model #Params

Specialized Video Generation Models

SkyReels-A2 14B 3.51 4.08 4.46 4.02 3.22 3.76 4.37 3.78 1.64 1.76 2.50 1.97 SkyReels-V3 14B 3.46 3.71 4.65 3.94 3.28 3.84 4.44 3.86 2.59 3.10 4.30 3.33 Phantom 14B 3.21 2.95 4.29 3.48 2.88 3.42 4.38 3.55 2.36 2.79 4.21 3.12

Unified Video Generation Models

VACE-Wan2.1 14B 3.88 4.48 4.68 4.35 3.31 4.03 4.51 3.95 2.60 3.03 4.40 3.34 VACE-LTX 2B 2.74 2.86 2.89 2.83 2.12 2.26 2.49 2.29 1.94 2.06 2.41 2.14 VINO 13B 3.72 4.22 4.46 4.13 3.56 4.34 4.58 4.16 2.63 2.97 4.24 3.28 UniVideo(query) 13B 3.35 3.90 4.41 3.89 2.98 3.73 4.18 3.63 2.30 2.50 3.89 2.89 UniVideo(hidden) 13B 3.33 4.18 4.38 3.97 3.22 4.12 4.36 3.90 2.31 2.83 3.94 3.03 OmniWeaving 8.3B 4.35 4.53 4.58 4.49 4.08 4.22 4.52 4.27 3.53 4.01 4.54 4.03

LoomVideo (Stage 3) 5B 3.81 3.78 3.91 3.83 3.30 3.32 3.69 3.44 2.40 2.70 3.61 2.90 LoomVideo (RL) 5B 3.95 3.90 4.20 4.02 3.51 3.44 4.06 3.67 2.47 2.67 3.67 2.94

Table 7: Quantitative comparison on FashionVideoBench.

Split by Metrics Split by Task

Model

Overall SC PF VQ Product Edit Model Edit Freeform Edit PRef Edit MRef Edit MI2V

UniVideo 4.08 4.34 4.37 3.84 4.93 4.20 4.24 4.05 4.29 4.26 OmniWeaving 3.28 3.71 3.70 3.67 4.04 3.49 2.95 3.49 3.72 3.56 VINO 4.18 4.51 4.45 4.02 4.83 4.27 4.27 4.43 4.45 4.38

LoomVideo (Stage 3) 4.45 4.74 4.61 4.59 4.95 4.47 4.51 4.37 4.70 4.60 LoomVideo (RL) 4.44 4.71 4.62 4.59 4.92 4.45 4.51 4.37 4.70 4.59

mechanism. Overall, these results compellingly highlight the exceptional efficiency and practical viability of our proposed architecture in this domain.

### 5 Conclusion and Future Work

In this work, we introduce LoomVideo, a highly efficient unified framework capable of handling diverse video generation and editing tasks via interleaved multimodal inputs. By replacing the standard text encoder with an MLLM, we unlock rich, hierarchical semantic understanding. To fully leverage this, we proposed the Deepstack injection mechanism, ensuring deep alignment between multimodal controls and the spatiotemporal generative prior. Furthermore, we address the severe computational bottleneck of existing editing frameworks by introducing the zero-overhead Scale-and-Add conditioning approach. By fundamentally bypassing the conventional token concatenation paradigm, this elegant design enables LoomVideo to achieve over 5.4× inference speedup while maintaining robust capabilities for complex, non-rigid edits. Coupled with the Negative Temporal RoPE strategy for multi-reference guidance, extensive experiments demonstrate that our compact 5B-parameter model achieves state-of-the-art or highly competitive performance across comprehensive open-domain benchmarks. Notably, it exhibits exceptional superiority and practicality in complex e-commerce and fashion generation scenarios.

Several avenues remain for future exploration. We aim to scale up the diffusion transformer’s parameter size and extend our multi-resolution training pipeline to support higher-definition (e.g., 720p or 1080p) and longer-duration video generation, ultimately pushing the boundaries of physical world simulation and visual fidelity.

### 6 Contributions and Acknowledgments

Authors are organized by their affiliation. ∗ indicates equal contribution, § indicates corresponding authors, and † indicates the project leader.

Peking University: Jianzong Wu∗†, Hao Lian∗, Jiongfan Yang, Dachao Hao, Ye Tian, Yunhai Tong§ Alibaba Group: Jingyuan Zhu, Biaolong Chen, Qiaosong Qi, Aixi Zhang, Wanggui He, Mushui Liu, Jinlong Liu, Pipei Huang, Hao Jiang§ We would like to thank MSALab at Peking University and Alibaba Group for their support and discussions throughout this project.

OpenVE-Bench

|Input|[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>Prompt: Apply the Cartoon Style to this video, ensuring seamless temporal consistency across all frames…|[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>Prompt: Transform the shotgun instantly into a polished wooden acoustic guitar, with strings and tuning pegs…|
|---|---|---|
|OmniWeaving|[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]|[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]|
|Kiwi-Edit|[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]|[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]|
|VINO|[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]|[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]|
|UniVideo|[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]|[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]|
|Ours|[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]|[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]|

RefVIE-Bench

|Input|[Figure 110]<br><br>Ref Image<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>Prompt: Replace the background with a Chinese ink painting, featuring a large golden mountain peak rising above swirling clouds…|Ref Image<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>Prompt: Replace the balck suit of the girl in the left with traditional Chinese Mamianqun skirt. All other parts of the video must remain unchanged.|
|---|---|---|
|OmniWeaving|[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]|[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]|
|Kiwi-Edit|[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]|[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]|
|VINO|[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]|[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]|
|UniVideo|[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]|[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]|
|Ours|[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]|[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]|

FashionVideoBench

|Input|[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>Ref Image1 Ref Image2 Ref Image3<br><br>Prompt: The woman (@Image 2), wearing the shirt (@Image 3) and trousers, and wearing earrings, stands indoors (@Image 1), gently swings her arms, then raises her right hand to brush the hair on her right side, closing her eyes and revealing a smile.|Ref Image1 Ref Image2 Ref Image3 Ref Image4<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>Prompt: The girl (@Image 2) is wearing a long-sleeved shirt (@Image 3) and jeans (@Image 4), wearing sneakers and holding a handbag, leaning against the white wall and green leaves (@Image 1), slightly twisting her body and looking to the side.|
|---|---|---|
|OmniWeaving|[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]|[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]|
|UniVideo|[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]|[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]|
|Ours|[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]|[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]|

Figure 3: Qualitative comparison of LoomVideo and other baselines.

Table 8: Comparison of inference time. We test all the models on the same type of GPU. We measure the average time required to generate/edit a single 480×832×97 video for both Text-to-Video (T2V) and Instruction-based Video Editing (TV2V) tasks. The speedup ratio is calculated based on OmniWeaving.

Model #Params Source Token Injection T2V (s) TV2V (s) Wan 2.2 5B + 5.68B (UMT5-XXL) - 138.61 UniVideo (hidden) 13B+ 7B Token Concat 1792.65 6140.18 OmniWeaving 8.3B+ 7B Channel Concat 824.93 899.32 VINO 13B+ 4B Token Concat 2793.52 9555.13 Ours 5B + 8B Scale-and-Add 132.23 (6.24×) 166.30 (5.41×)

### References

Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, et al. Scaling instruction-based video editing with a high-quality synthetic dataset. arXiv preprint arXiv:2510.15742, 2025a.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025b.

Guibin Chen, Dixuan Lin, Jiangping Yang, Youqiang Zhang, Zhengcong Fei, Debang Li, Sheng Chen, Chaofeng Ao, Nuo Pang, Yiming Wang, et al. Skyreels-v4: Multi-modal video-audio generation, inpainting and editing model. arXiv preprint arXiv:2602.21818, 2026a.

Junyi Chen, Tong He, Zhoujie Fu, Pengfei Wan, Kun Gai, and Weicai Ye. Vino: A unified visual generator with interleaved omnimodal context. arXiv preprint arXiv:2601.02358, 2026b.

Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv:2506.18095, 2025a.

Zhuowei Chen, Bingchuan Li, Tianxiang Ma, Lijie Liu, Mingcong Liu, Yi Zhang, Gen Li, Xinghui Li, Siyu Zhou, Qian He, et al. Phantom-data: Towards a general subject-consistent video generation dataset. arXiv preprint arXiv:2506.18851, 2025b.

Wei Chow, Linfeng Li, Lingdong Kong, Zefeng Li, Qi Xu, Hang Song, Tian Ye, Xian Wang, Jinbin Bai, Shilin Xu, et al. Editmgt: Unleashing potentials of masked generative transformers in image editing. arXiv preprint arXiv:2512.11715, 2025.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

Zhengcong Fei, Debang Li, Di Qiu, Jiahua Wang, Yikun Dou, Rui Wang, Jingtao Xu, Mingyuan Fan, Guibin Chen, Yang Li, et al. Skyreels-a2: Compose anything in video diffusion transformers. arXiv preprint arXiv:2504.02436, 2025.

Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007, 2024.

Haoyang He, Jie Wang, Jiangning Zhang, Zhucun Xue, Xingyuan Bu, Qiangpeng Yang, Shilei Wen, and Lei Xie. Openve-3m: A large-scale high-quality dataset for instruction-guided video editing. arXiv preprint arXiv:2512.07826, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In ICCV, 2025.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. In NeurIPS, pp. 36652–36663, 2023.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Maksim Kuprashevich, Grigorii Alekseenko, Irina Tolstykh, Georgii Fedorov, Bulat Suleimanov, Vladimir Dokholyan, and Aleksandr Gordeev. Nohumansrequired: Autonomous high-quality image editing triplet mining. In WACV, 2026.

Debang Li, Zhengcong Fei, Tuanhui Li, Yikun Dou, Zheng Chen, Jiangping Yang, Mingyuan Fan, Jingtao Xu, Jiahua Wang, Baoxuan Gu, et al. Skyreels-v3 technique report. arXiv preprint arXiv:2601.17323, 2026.

Yiqi Lin, Guoqiang Liang, Ziyun Zeng, Zechen Bai, Yanzhe Chen, and Mike Zheng Shou. Kiwi-edit: Versatile video editing via instruction and reference guidance. arXiv preprint arXiv:2603.02175, 2026.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025a.

Jinlong Liu, Wanggui He, Peng Zhang, Mushui Liu, Hao Jiang, and Pipei Huang. Promptecho: Annotation-free reward from vision-language models for text-to-image reinforcement learning. arXiv preprint arXiv:2604.12652, 2026.

Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. In ICCV, 2025b.

Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371, 2024.

OpenAI. Sora 2. https://openai.com/sora, 2024. Kaihang Pan, Qi Tian, Jianwei Zhang, Weijie Kong, Jiangfeng Xiong, Yanxin Long, Shixue Zhang, Haiyi Qiu,

Tan Wang, Zheqi Lv, et al. Omniweaving: Towards unified video generation with free-form composition and reasoning. arXiv preprint arXiv:2603.24458, 2026.

Bowen Ping, Chengyou Jia, Minnan Luo, Hangwei Qian, and Ivor Tsang. Flow-factory: A unified framework for reinforcement learning in flow-matching models. arXiv preprint arXiv:2602.12529, 2026.

Yusu Qian, Eli Bocek-Rivele, Liangchen Song, Jialing Tong, Yinfei Yang, Jiasen Lu, Wenze Hu, and Zhe Gan. Pico-banana-400k: A large-scale dataset for text-guided image editing. arXiv preprint arXiv:2510.19808, 2025.

Runway ML. Introducing Runway Aleph: A state-of-the-art in-context video model. https://runwayml.com/research/introducing-runway-aleph, July 2025.

Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.

Zhiyu Tan, Hao Yang, Luozheng Qin, Jia Gong, Mengping Yang, and Hao Li. Omni-video: Democratizing unified video understanding and generation. arXiv preprint arXiv:2507.06119, 2025.

Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He, Kang He, et al. Kling-omni technical report. arXiv preprint arXiv:2512.16776, 2025.

Yunze Tong, Mushui Liu, Canyu Zhao, Wanggui He, Shiyi Zhang, Hongwei Zhang, Peng Zhang, Jinlong Liu, Ju Huang, Jiamang Wang, Hao Jiang, and Pipei Huang. Alleviating sparse rewards by modeling step-wise and long-term sampling effects in flow-based grpo. In ICML, 2026.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In CVPR, 2025a.

Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. Gpt-image-edit-1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033, 2025b.

Cong Wei, Quande Liu, Zixuan Ye, Qiulin Wang, Xintao Wang, Pengfei Wan, Kun Gai, and Wenhu Chen. Univideo: Unified understanding, generation, and editing for videos. arXiv preprint arXiv:2510.08377, 2025.

Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025a.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025b.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025c.

Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi Xie, and Lei Zhang. Insvie-1m: Effective instruction-based video editing with elaborate dataset construction. In ICCV, 2025d.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In CVPR, 2025.

Hao Yang, Zhiyu Tan, Jia Gong, Luozheng Qin, Hesen Chen, Xiaomeng Yang, Yuqing Sun, Yuetan Lin, Mengping Yang, and Hao Li. Omni-video 2: Scaling mllm-conditioned diffusion for unified video generation and editing. arXiv preprint arXiv:2602.08820, 2026.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.

Appendix

- A More Experimental Results

More qualitative results across various tasks are shown in fig. 5, fig. 6, fig. 7, and fig. 8. The results demonstrate that LoomVideo unifies diverse video generation capabilities within a single model. Notably, it achieves this versatility while maintaining high efficiency, powered by a compact 5B parameter architecture.

- B Limitation

As shown in fig. 4, despite achieving competitive performance across various video generation tasks, our model still presents several limitations for future exploration. First, noticeable visual artifacts including distorted eyes and unnatural limb movements emerge under highly dynamic scenarios. This problem stems from our relatively compact model size (5B parameters), which constrains the model’s capability to learn fine-grained temporal dynamics. Second, our training data is primarily curated for e-commerce scenarios and lacks sufficient diversity across realworld scenes. This constraint hinders accurate background reconstruction on general MI2V benchmarks such as IntelligentVBench. In future work, we plan to enhance model capacity and expand the coverage of training data to address the above issues.

- C FashionVideoBench Evaluation Prompts

This section presents the evaluation prompt templates for each task in FashionVideoBench. Specifically, the evaluation prompt template for product editing is detailed in fig. 9. The evaluation prompt template for model editing is detailed in fig. 10. The evaluation prompt template for freeform editing is detailed in fig. 11. Finally, we design evaluation prompt templates for the distinct sub-tasks of reference-guided video editing in fig. 12, fig. 13, and fig. 14.

[Figure 197]

Figure 4: Low-quality generation cases of LoomVideo.

Text-to-Video

Prompt: A cat wearing sunglasses at a pool. Prompt: A cute happy Corgi playing in park, sunset.

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

Prompt: A beautiful coastal beach in spring, waves lapping on sand by Vincent van Gogh.

Prompt: Pacific coast, carmel by the sea ocean and waves.

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

Prompt: A motorcycle gliding through a snowy field.

Prompt: Sunset time lapse at the beach with moving clouds and colors in the sky.

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

Prompt: This is how I do makeup in the morning. Prompt: Yoda playing guitar on the stage.

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Insturction Editing (Image)

Prompt: Transfer the image into an 8-bit pixel-art videogame style.

Prompt: Transfer the image into a clean graphite pencilsketch style.

Prompt: Change the tiny house in the picture from the forest to the beach.

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Prompt: Add a car in the foreground to the right side of the image.

Prompt: Change the priest’s coat color to dark brown.

Prompt: Replace the bird in the image with a small rabbit.

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

Prompt: Replace the tent in the image with a small wooden cabin.

Prompt: Remove the person on the left side of the image.

Prompt: Extract the yacht and the water surrounding it from the image.

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

Figure 5: Qualitative results for LoomVideo on Text-to-Video and Instuction Editing tasks.

Instruction Editing (Video)

Prompt: Apply the Impressionist aesthetic to this video…The result should emulate the fluid brushstroke techniques and atmospheric focus of 19th-century Impressionist art…

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Prompt: Replace the man's dark chef coat with a light blue short-sleeve button-up shirt, keeping the same position and pose within the video scene.

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Prompt: Replace the background with a dynamic desert highway scene. Heat waves should shimmer above the asphalt, dust occasionally drifts across the road… The man and SUV remain perfectly still.

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Prompt: Remove the curved, crescent-shaped piano keyboard with alternating black and white keys from the entire video. The background must be reconstructed with temporal consistency…

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

Prompt: Overlay an animated colorful kite flying in the upper right quadrant of the sky. The kite should move gently with the wind, swaying and fluttering its tail realistically...

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Prompt: Given the video of a man's natural hands typing on a black laptop keyboard, transform the hands instantly into sleek, metallic robotic hands with articulated joints and glowing blue lights.

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

Prompt: Given the video of the freshly baked seeded bread loaf on the cutting board with a hand holding a knife, transform the entire bread surface into a miniature lush forest with green trees, moss, and tiny flowers shaped exactly like the loaf.

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Prompt: Remove the subtitles at the bottom of the video.

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

Figure 6: Qualitative results for LoomVideo on Instuction Editing task.

Instruction-Image Editing

Prompt: Replace the background with a whimsical fairytale landscape, featuring colorful mushroom houses and a floating castle in a magical forest…

Ref Image

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

Ref Image Prompt: Replace the background with the scene shown in the image.

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

Ref Image Prompt: Replace the background with the scene shown in the image.

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

Prompt: Replace the girl's outfit from a black suit with a matte black techwear tactical jacket. All other Ref Image parts of the video must remain unchanged.

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

Prompt: Replace the bus with a modern Italian bright red sports car. Sleek aerodynamic curves, low Ref Image ground clearance, and large alloy wheels with red brake calipers…

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

Ref Image Prompt: Replace the pink flower with the white flower in the image.

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

Prompt: Add the black dog in the image, sitting attentively on the pavement in front of the man.

Ref Image

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

Prompt: Add a static DIY toy robot made from brown corrugated cardboard boxes and toilet paper rolls Ref Image next to the window. All other parts of the video must remain unchanged.

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

Ref Image Prompt: Add the man in the image, standing on the left and introducing the car.

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

Figure 7: Qualitative results for LoomVideo on Instuction-Image Editing task.

FashionVideoBench

Prompt: Remove the large white lace collar from the shirt.

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

Prompt: Change the subject's floral dress to a black blazer dress.

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

Prompt: Replace the person: The person in the video is a dark-skinned woman with short, dark curly hair, wearing large sunglasses with light brown frames. She is dressed in a light grey and white vertical-striped, long-sleeved button-down shirt worn open over a simple white crop top. Her bottoms consist of dark navy blue, wide-leg pleated pants…She completes the outfit with a pair of white sneaker...

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

Prompt: Edit this video: The video features the same woman as the original video, wearing the same large square sunglasses with white frames and dark lenses, and the same large silver hoop earrings. She is dressed in the same long-sleeved purple polo shirt with a white collar peeking out, the same black pants…Additionally, she is carrying a black handbag with a gold chain strap on her right arm.

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

Prompt: Edit this video with reference images: Replace the floral embroidered boots with classic black leather ankle boots.

Ref Image

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

Prompt: Generate a video that follows the movement of people in the reference video and the people and Ref Image background in the reference image:

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

Prompt: Show the man wearing the overcoat, shirt, suit pants, tie, sunglasses (@Image2), and watch. Have him carry the briefcase, grasp the overcoat lapel with one hand, and stride through the courtyard (@Image1)…

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

Image1 Image2

Prompt: Position the man (@Image 2) in the street background (@Image 1) wearing the sunglasses, turtleneck, leather jacket (@Image 3), and trousers, with the watch on his left wrist. Have the man (@Image 2) pull and adjust the hem of the leather jacket (@Image 3) with both hands…

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

Image1 Image2 Image3

Figure 8: Qualitative results for LoomVideo on our benchmark FashionVideoBench.

Evaluation Prompt Template for Product Editing

You are an expert data rater specializing in grading video product editing. You will be provided with an original video, the edited video, and the editing instruction. The instruction specifies changes to be made to a product in the video. Your task is to evaluate the editing performance on a 5-point scale across three key dimensions:

- 1. The first score: Subject Consistency Objective: Evaluate how well the edited product maintains its identity and appearance throughout the video.

- 5: Perfect Preservation. Product identity is flawlessly maintained with all attributes correctly edited. - 4: High Preservation. Product is highly consistent with only minor detail loss. - 3: Moderate Preservation. Product is generally recognizable but with noticeable flaws. - 2: Low Preservation. Product suffers from severe drift or incorrect edits. - 1: Complete Failure. Product is completely irrelevant or incorrectly edited.

- 2. The second score: Prompt Following Objective: Assess how accurately the video executes the editing instruction.

- 5: Perfect Alignment. Every detail of the instruction is perfectly rendered. - 4: Good Alignment. The core semantics of the prompt are successfully captured. - 3: Partial Alignment. The model captures the main idea but ignores specific nuances. - 2: Weak Alignment. The video mostly ignores the prompt. - 1: No Alignment. The video content has no relevance to the instruction.

- 3. The third score: Visual Quality Objective: Evaluate the general aesthetic quality, temporal consistency, and motion smoothness.

- 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. - 4: Good. High overall quality with only minor artifacts. - 3: Fair. Acceptable but noticeably flawed. - 2: Poor. Significantly degraded quality. 1: Unacceptable. Completely collapsed visual integrity.

Example Response Format: You are required to return a dictionary structured as follows: {”Subject Consistency”: [A number from 1 to 5], ”Prompt Following”: [A number from 1 to 5], ”Visual Quality”: [A number from 1 to 5]}.

The editing instruction is: <edit prompt> Below are the original video and edited video:

Figure 9: Evaluation prompt template for video product editing.

Evaluation Prompt Template for Video Model Replacement Editing

You are an expert data rater specializing in grading video model replacement editing. You will be provided with an original video, the edited video, and the editing instruction. The instruction specifies replacing a model or person in the video. Your task is to evaluate the editing performance on a 5-point scale across three key dimensions:

1. The first score: Subject Consistency Objective: Evaluate how well the new model maintains consistency throughout the video.

- 5: Perfect Consistency. Model identity is flawlessly maintained across all frames. - 4: High Consistency. Model is highly consistent with only minor flickering. - 3: Moderate Consistency. Model is recognizable but shows some instability. - 2: Low Consistency. Model suffers from significant identity drift. - 1: Complete Failure. Model is completely inconsistent or unrecognizable.

2. The second score: Prompt Following Objective: Assess how accurately the video executes the model replacement instruction.

- 5: Perfect Alignment. The model is perfectly replaced as specified. - 4: Good Alignment. The model is correctly replaced with minor deviations. - 3: Partial Alignment. The model is partially replaced but with noticeable errors. - 2: Weak Alignment. The replacement is attempted but poorly executed. - 1: No Alignment. The model was not replaced or replaced incorrectly.

3. The third score: Visual Quality Objective: Evaluate the general aesthetic quality, temporal consistency, and motion smoothness.

- 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. - 4: Good. High overall quality with only minor artifacts. - 3: Fair. Acceptable but noticeably flawed. - 2: Poor. Significantly degraded quality. 1: Unacceptable. Completely collapsed visual integrity.

Example Response Format: You are required to return a dictionary structured as follows: {”Subject Consistency”: [A number from 1 to 5], ”Prompt Following”: [A number from 1 to 5], ”Visual Quality”: [A number from 1 to 5]}.

The editing instruction is: <edit prompt> Below are the original video and edited video:

Figure 10: Evaluation prompt template for video model replacement editing.

Evaluation Prompt Template for Freeform Video Editing

You are an expert data rater specializing in grading freeform video editing. You will be provided with an original video, the edited video, and the editing instruction. The instruction specifies general editing changes to be made to the video. Your task is to evaluate the editing performance on a 5-point scale across three key dimensions:

1. The first score: Subject Consistency Objective: Evaluate how well the edited content maintains consistency throughout the video.

- 5: Perfect Consistency. Edited content is flawlessly maintained across all frames. - 4: High Consistency. Content is highly consistent with only minor flickering. - 3: Moderate Consistency. Content is recognizable but shows some instability. - 2: Low Consistency. Content suffers from significant identity drift. - 1: Complete Failure. Content is completely inconsistent or unrecognizable.

2. The second score: Prompt Following Objective: Assess how accurately the video executes the editing instruction.

- 5: Perfect Alignment. Every detail of the instruction is perfectly rendered. - 4: Good Alignment. The core semantics of the prompt are successfully captured. - 3: Partial Alignment. The model captures the main idea but ignores specific nuances. - 2: Weak Alignment. The video mostly ignores the prompt. - 1: No Alignment. The video content has no relevance to the instruction.

3. The third score: Visual Quality Objective: Evaluate the general aesthetic quality, temporal consistency, and motion smoothness.

- 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. - 4: Good. High overall quality with only minor artifacts. - 3: Fair. Acceptable but noticeably flawed. - 2: Poor. Significantly degraded quality. 1: Unacceptable. Completely collapsed visual integrity.

Example Response Format: You are required to return a dictionary structured as follows: {”Subject Consistency”: [A number from 1 to 5], ”Prompt Following”: [A number from 1 to 5], ”Visual Quality”: [A number from 1 to 5]}.

The editing instruction is: <edit prompt> Below are the original video and edited video:

Figure 11: Evaluation prompt template for freeform video editing.

Evaluation Prompt Template for Video Item Replacement with Reference Image

You are an expert data rater specializing in grading video item replacement with reference image. You will be provided with an original video, a reference image, the edited video, and the editing instruction. The instruction specifies replacing an item in the video using the reference image. Your task is to evaluate the editing performance on a 5-point scale across three key dimensions:

- 1. The first score: Subject Consistency Objective: Evaluate how well the replaced item matches the reference image and maintains consistency throughout the video.

- 5: Perfect Match. The replaced item is identical to the reference image and consistent across all frames. - 4: High Match. The replaced item closely matches the reference with minor differences. - 3: Moderate Match. The replaced item is similar but has noticeable differences. - 2: Low Match. The replaced item has significant differences from the reference. - 1: No Match. The replaced item is completely different from the reference.

- 2. The second score: Prompt Following Objective: Assess how accurately the video executes the item replacement instruction.

- - 5: Perfect Alignment. The item is perfectly replaced as specified. - 4: Good Alignment. The item is correctly replaced with minor deviations. - 3: Partial Alignment. The item is partially replaced but with noticeable errors.
- - 2: Weak Alignment. The replacement is attempted but poorly executed. - 1: No Alignment. The item was not replaced or replaced incorrectly.

- 3. The third score: Visual Quality Objective: Evaluate the general aesthetic quality, temporal consistency, and motion smoothness.

- 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. - 4: Good. High overall quality with only minor artifacts. - 3: Fair. Acceptable but noticeably flawed. - 2: Poor. Significantly degraded quality. 1: Unacceptable. Completely collapsed visual integrity.

Example Response Format: You are required to return a dictionary structured as follows: {”Subject Consistency”: [A number from 1 to 5], ”Prompt Following”: [A number from 1 to 5], ”Visual Quality”: [A number from 1 to 5]}.

The editing instruction is: <edit prompt> Below are the original video, reference image, and edited video:

Figure 12: Evaluation prompt template for video item replacement with reference image.

Evaluation Prompt Template for Motion-Transfer Video Generation

You are an expert data rater specializing in grading motion-transfer video generation. You will be provided with: (1) A source video (providing the motion/pose reference), (2) A reference image (providing the target person appearance and background), and (3) The generated video (which should combine the source motion with the reference appearance). Your task is to evaluate the generated video on a 5-point scale across three key dimensions:

- 1. The first score: Subject Consistency Objective: Evaluate both (a) how faithfully the generated video reproduces the person’s appearance (face, body, clothing) and background from the reference image, and (b) how consistently these elements are maintained across all frames of the generated video.

- 5: Perfect. Reference appearance is faithfully reproduced and flawlessly consistent across all frames. - 4: High Consistency. Appearance closely matches the reference with only minor detail loss or flickering. - 3: Moderate. Person is generally recognizable from the reference but with noticeable differences or temporal instability. - 2: Low Consistency. Significant identity drift from the reference or severe frame-to-frame inconsistency. - 1: Complete Failure. Generated appearance is unrelated to the reference or completely inconsistent.

- 2. The second score: Prompt Following Objective: Assess how faithfully the generated video reproduces the movements, poses, and motion trajectory from the source video as specified by the instruction.

- - 5: Perfect Alignment. All movements, poses, and temporal dynamics from the source are flawlessly reproduced.
- - 4: Good Alignment. Core motion is accurately captured with only minor timing or pose deviations. - 3: Partial Alignment. General movement pattern is recognizable but with noticeable errors in pose or timing. - 2: Weak Alignment. Motion is largely distorted or only partially follows the source. - 1: No Alignment. Generated motion bears no resemblance to the source video.

- 3. The third score: Visual Quality Objective: Evaluate the general aesthetic quality, temporal consistency, and motion smoothness.

- 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. - 4: Good. High overall quality with only minor artifacts. - 3: Fair. Acceptable but noticeably flawed. - 2: Poor. Significantly degraded quality. 1: Unacceptable. Completely collapsed visual integrity.

Example Response Format: You are required to return a dictionary structured as follows: {”Subject Consistency”: [A number from 1 to 5], ”Prompt Following”: [A number from 1 to 5], ”Visual Quality”: [A number from 1 to 5]}.

The editing instruction is: <edit prompt> Below are the source video, reference image, and generated video:

Figure 13: Evaluation prompt template for motion-transfer video generation.

Evaluation Prompt Template for Multi-Reference-Image-to-Video Generation

You are an expert data rater specializing in grading multi-reference-image-to-video generation. You will be provided with multiple reference images and the generated video. The reference images provide visual elements (person appearance, clothing items, background, etc.) that should appear in the generated video. A text instruction describes how to compose these elements into a coherent video. Your task is to evaluate the generated video on a 5-point scale across three key dimensions:

- 1. The first score: Subject Consistency Objective: Evaluate both (a) how faithfully the generated video reproduces the visual elements from the reference images (person identity, clothing details, background, product appearance, etc.), and (b) how consistently these elements are maintained across all frames of the generated video.

- - 5: Perfect. All referenced elements are faithfully reproduced and flawlessly consistent across all frames. - 4: High Consistency. Most elements accurately match the references with only minor detail loss or flickering. - 3: Moderate. Elements are generally recognizable but with noticeable differences from references or temporal instability. - 2: Low Consistency. Significant mismatch with references or severe frame-to-frame inconsistency.
- - 1: Complete Failure. Generated content bears no resemblance to the reference images or is completely inconsistent.

- 2. The second score: Prompt Following Objective: Assess how accurately the video follows the text instruction (scene composition, actions, camera movements, timing, etc.).

- 5: Perfect Alignment. Every detail of the instruction is perfectly rendered. - 4: Good Alignment. The core semantics of the prompt are successfully captured. - 3: Partial Alignment. The model captures the main idea but ignores specific nuances. - 2: Weak Alignment. The video mostly ignores the prompt. - 1: No Alignment. The video content has no relevance to the instruction.

- 3. The third score: Visual Quality Objective: Evaluate the general aesthetic quality, temporal consistency, and motion smoothness.

- 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. - 4: Good. High overall quality with only minor artifacts. - 3: Fair. Acceptable but noticeably flawed. - 2: Poor. Significantly degraded quality. 1: Unacceptable. Completely collapsed visual integrity.

Example Response Format: You are required to return a dictionary structured as follows: {”Subject Consistency”: [A number from 1 to 5], ”Prompt Following”: [A number from 1 to 5], ”Visual Quality”: [A number from 1 to 5]}.

The instruction is: <edit prompt> Below are the reference images and generated video:

Figure 14: Evaluation prompt template for multi-reference-image-to-video generation.

