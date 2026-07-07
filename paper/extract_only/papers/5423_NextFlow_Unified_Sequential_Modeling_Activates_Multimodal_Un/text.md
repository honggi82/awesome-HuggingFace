[Figure 1]

## NextFlow: Unified Sequential Modeling Activates Multimodal Understanding and Generation

Huichao Zhang*1, Liao Qu*1, Yiheng Liu*1, Hang Chen*1, Yangyang Song*1, Yongsheng Dong*1, Shikun Sun*1,2, Xian Li*1, Xu Wang1,†, Yi Jiang1,†, Hu Ye1, Bo Chen1, Yiming Gao1, Peng Liu1, Akide Liu1,3, Zhipeng Yang1, Qili Deng1, Linjie Xing1, Jiyang Liu1, Zhao Wang1, Yang Zhou1, Mingcong Liu1, Yi Zhang1, Qian He1, Xiwei Hu1, Zhongqi Qi1, Jie Shao1, Zhiye Fu1, Shuai Wang1, Fangmin Chen1, Xuezhi Chai1, Zhihua Wu1, Yitong Wang1, Zehuan Yuan1, Daniel K. Du1, Xinglong Wu1

# arXiv:2601.02204v1[cs.CV]5Jan2026

1ByteDance, 2TsingHua University, 3Monash University ∗Core Contributors, †Project Lead

### Abstract

We present NextFlow, a unified decoder-only autoregressive transformer trained on 6 trillion interleaved text-image discrete tokens. By leveraging a unified vision representation within a unified autoregressive architecture, NextFlow natively activates multimodal understanding and generation capabilities, unlocking abilities of image editing, interleaved content and video generation. Motivated by the distinct nature of modalities—where text is strictly sequential and images are inherently hierarchical—we retain next-token prediction for text but adopt next-scale prediction for visual generation. This departs from traditional raster-scan methods, enabling the generation of 1024 × 1024 images in just 5 seconds—orders of magnitude faster than comparable AR models. We address the instabilities of multi-scale generation through a robust training recipe. Furthermore, we introduce a prefix-tuning strategy for reinforcement learning. Experiments demonstrate that NextFlow achieves state-of-the-art performance among unified models and rivals specialized diffusion baselines in visual quality.

Official Page: https://github.com/ByteVisionLab/NextFlow

Aa

Image Decoder

Decoder-only Transformer

Aa

Text

Decoder Aa

Figure 1 Architecture of NextFlow. NextFlow processes interleaved text-image discrete token sequences as input and generates interleaved multimodal outputs. Text tokens are predicted via next-token modeling, while visual tokens are generated through next-scale prediction.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

###### Figure 2 NextFlow visualization. Our approach generates high-fidelity images via a pure discrete autoregressive framework, achieving production-grade visual quality.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

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

###### Figure 3 NextFlow visualization. Utilizing next-scale prediction for visual generation, our model can efficiently synthesizes high-quality 1024 × 1024 images under 5 seconds.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Edit/global_edit/background Change the clear blue sky in the background to a sky filled with soft, scattered white clouds

Edit/local_edit/inpainting Fill the hollow center of the bundt cake with a smooth vanilla cream filling

Edit/global_edit/modality Convert the dog silhouette into a realistic photograph of a dog with detailed fur texture, natural colors, and realistic shading

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Edit/local_edit/human_age Make the man look 20 years older, adding wrinkles, graying his hair, and adjusting his facial features to show advanced age while keeping the workshop background unchanged

Edit/local_edit/object_material Change the material of the black hanging swing on the wooden porch to rustic cedar wood

Edit/local_edit/object_rotation Rotate the left Under Armour soccer cleat 90 degrees clockwise around its vertical axis so that its toe faces directly toward the camera instead of to the left

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Edit/local_edit/object_color Change the color of the leaf-shaped pillow from light pink to soft blue

Edit/local_edit/object_outfit Change the woman's black tank top to a red flannel shirt

Edit/global_edit/enhance_image Enhance the quality to make the details of the palm tree leaves, sunset glow, and ocean waves clearer and more high-definition

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Edit/local_edit/object_position Move the white plastic anchor to be directly below the drill bit

Edit/style_edit/real_to_unreal Convert this real-world port train scene with cranes, freight cars, and the red-and-yellow locomotive into a vibrant 2D cartoon animation style

Edit/style_edit/unreal_to_real Convert this stylized, painting-like mountain landscape into a realistic photographic image with natural mountain textures, detailed snow coverage, lifelike sky gradients, and authentic forest foliage.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Edit/local_edit/object_remove Remove the rear spoiler from the gray Lamborghini sports car.

Edit/global_edit/lighting_direction Alter the direction of the sunset light so that it comes from the left side of the horizon instead of the center, adjusting the shadows on the barbed wire and hand to match the new light angle.

Edit/local_edit/object_add Add a black desk lamp with a white fabric shade to the center of the wooden desk

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Edit/global_edit/lighting_add Add a warm directional light source from the upper left to cast soft shadows and accentuate the wood grain texture and craftsmanship of the desk.

Edit/local_edit/object_size Reduce the size of the large pink bow on the puppy's head to make it more proportional to the puppy's face.

Edit/global_edit/adjust_color Adjust the overall color of the image to a cooler tone with reduced saturation to create a more muted and minimalist visual effect.

Edit/local_edit/object_motion Change the rugby player's posture from pointing with his right arm extended to running forward with both arms bent at the elbows

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Edit/text_edit/replace Replace the cardholder name 'CHRIS L MARTIN' with 'JANE A DOE'

Edit/view_edit/camera_view Change the camera view to a bird's-eye perspective of the wooden house, showing the entire roof structure and the surrounding forest and lawn from above

Edit/view_edit/camera_zoom_in Zoom in on the central glass coffee table with the orange floral arrangement to focus on the table's clear surface, the vibrant flowers, and the surrounding chairs with teal patterned pillows.

###### Figure 4 Edit results of NextFlow on EditCanvas benchmark.

### Contents

- 1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 2 Model Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 2.1 Tokenizer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.2 Decoder-Only Transformer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.3 Optional Diffusion Decoder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 3 Training Odyssey . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 3.1 Tokenizer Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 3.2 Decoder-only Transformer Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- 3.2.1 Before the Journey Begins . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 3.2.2 Alignment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 3.2.3 Pre-Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 3.2.4 Continue-Training and Supervised Fine-Tuning . . . . . . . . . . . . . . . . . . . . . . 13
- 3.2.5 Reinforcement Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- 3.3 Diffusion Decoder Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 4 Infrastructure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 5 Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 5.1 Visual Understanding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 5.2 Image Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 5.3 Image Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- 5.3.1 Traditional Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 5.3.2 Subject-driven Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- 5.4 Interleaved Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 5.5 Text Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- 6 Model Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 6.1 Image Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 6.2 Image Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 6.3 Interleaved Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- 6.4 CoT Reasoning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- 6.5 In-context Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- 6.6 Image Reconstruction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- 6.7 Multimodal Understanding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- 7 Conclusion, Limitations and Future Directions . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

##### A Additional Implementation Details and Analysis . . . . . . . . . . . . . . . . . . . . . . . . . 38

- A.1 Predefined Scale Schedules . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- A.2 Inference Efficiency Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39

##### B EditCanvas benchmark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40

### 1 Introduction

The pursuit of artificial general intelligence (AGI) has long envisioned a unified system capable of perceiving, reasoning, and creating across diverse modalities. While Large Language Models (LLMs) have achieved mastery in text understanding and reasoning, and Diffusion Models [17, 35, 57] have revolutionized visual generation, these two distinct paradigms remain largely separated. This separation creates a friction: diffusion models excel at pixel-level fidelity but lack the inherent logical reasoning and in-context learning capabilities of LLMs, whereas traditional multimodal LLMs are often restricted to perception only.

Recently, AR-Diffusion hybrid architecture including Transfusion [93] and Bagel [16] demonstrated promising results for unified understanding and generation. However, the reliance on two different representations creates a gap between generation and understanding. This separation imposes re-encoding overheads for interleaved tasks and might fundamentally constrain the potential for deep multimodal integration. On the other side, pure autoregressive efforts like Chameleon [66], EMU3 [74], and EMU3.5 [13] remain constrained by two fundamental bottlenecks that hinder their practical deployment and multimodal capability. First, the reliance on the standard raster-scan next-token prediction paradigm for visual generation incurs a prohibitive computational cost at high resolutions. Unlike text, the sequence length of flattened visual tokens grows quadratically with resolution. Consequently, generating a single 1024 × 1024 image via raster-scan autoregression can take over 10 minutes [13, 74], making these models significantly slower than their diffusion counterparts and impractical for interactive applications. Second, the visual representations in these models are typically derived from reconstruction-oriented VQ tokenizers. While these tokenizers optimize for pixel-level fidelity, the resulting discrete codes often lack high-level semantic density. This semantic gap fundamentally limits the model’s performance on multimodal understanding tasks, as the visual tokens fail to capture the abstract concepts necessary for complex reasoning and alignment with the textual latent space.

In this work, we introduce NextFlow, a unified sequential modeling framework that activates both multimodal understanding and generation within a single decoder-only transformer. To address the efficiency bottleneck, NextFlow departs from raster-scan generation, adopting a next-scale prediction paradigm [69]. This hierarchical approach generates visual content from coarse structural layouts to fine-grained details, significantly reducing inference latency to just 5 seconds for a 1024 × 1024 image—orders of magnitude faster than rasterscan counterparts. To bridge the semantic gap, we employ a dual-codebook tokenizer [54] that decouples semantic and pixel-level features, ensuring both high-level conceptual alignment and fine-grained visual fidelity within a dynamic resolution framework.

Introduction of next-scale prediction in a unified decoder-only structure is non-trivial and presents unique challenges. We scale NextFlow on a massive corpus of 6 trillion tokens, comprising text, image-text pairs, and interleaved multimodal data. Throughout this journey, we identify and resolve key instabilities inherent to next-scale AR generation. Furthemore, we implement a rigorous post-training pipeline to boost the model performance. Uniquely, we propose a prefix-tuning strategy for Group Reward Policy Optimization (GRPO), which focuses optimization on the coarse-scale "prefixes" that determine global structure. This approach stabilizes RL training and effectively aligns the model with downstream objectives. For scenarios demanding hyper-realistic detail, we further integrate an optional diffusion-based decoder that refines the discrete output, pushing the boundaries of visual fidelity without compromising the unified architecture.

NextFlow demonstrates that a unified AR model can rival state-of-the-art diffusion models in visual quality while retaining the reasoning power of LLMs. Our 7B parameter model achieves competitive performance on text-to-image benchmarks and outperforms specialized models in image editing. Crucially, the unified architecture naturally supports interleaved text-image tasks. NextFlow can perform Chain-of-Thought (CoT) reasoning to refine prompts before generation or enable in-context learning for zero-shot image editing. Analysis reveals that NextFlow is highly efficient, requiring 6× fewer FLOPs during inference compared to MMDiT-based diffusion models [17] at 10242 resolution, achieving a generation speed of 5s per image.

Our contributions are summarized as follows:

- • We propose NextFlow, a unified decoder-only Transformer that activates multimodal understanding, generation, and editing. It employs next-scale prediction for efficient generation and a dual-codebook tokenizer to ensure high semantic density.

- • We present a robust training recipe validated on 6 trillion tokens, introduce a novel RL strategy for multi-scale generation that focuses optimization on coarse scales, and an optional diffusion decoder for enhanced visual detail, establishing a comprehensive training pipeline.
- • Extensive experiments demonstrate that NextFlow achieves state-of-the-art performance, outperforming specialized image editing models and rivaling top-tier diffusion models in visual quality. Crucially, we show that a unified AR architecture can be both computationally efficient and structurally simple, offering a superior alternative to complex hybrid architectures.

### 2 Model Architecture

#### 2.1 Tokenizer

The NextFlow tokenizer adopts a dual-codebook architecture building upon TokenFlow [54], which simultaneously achieves high-fidelity image reconstruction and semantically rich discrete representations for multimodal understanding.

Specifically, this design decouples the learning of semantic and pixel-level features while maintaining their alignment via a shared-mapping mechanism. Unlike standard VQ-VAE that relies solely on pixel reconstruction, our quantization process is jointly constrained by both reconstruction fidelity and semantic consistency. By minimizing the weighted summation of semantic and pixel-level distances during codebook lookup, we ensure that the discrete tokens encapsulate both high-level concepts (distilled from the semantic teacher) and fine-grained visual details.

To accommodate dynamic resolution processing, we upgrade the semantic encoder initialization from siglip-so400m1 to siglip2-so400m-naflex2, enabling variable resolution and aspect ratio processing. When combined with our CNN-based pixel branch, this architecture enables fully dynamic spatial processing, allowing our AR model to train directly at native resolutions without the constraints of fixed input ratios. We employ multi-scale VQ [69] to further enhance the quantization quality. The scale settings are detailed in the appendix A.1.

#### 2.2 Decoder-Only Transformer

Our autoregressive framework builds upon a standard decoder-only Transformer architecture, initialized from Qwen2.5-VL-7B [1] to leverage its strong multimodal priors. We extend this architecture to support visual token prediction using a next-scale prediction paradigm. For the newly added visual codebook, we initialize the embeddings directly from the tokenizer’s codebook embeddings. Empirically, we find that a unified prediction head achieves comparable performance to separate text and image heads while maintaining architectural simplicity. Therefore, we adopt a single output head for both modalities. The model is trained with cross-entropy loss to predict codebook indices across both modalities.

Scale 1

Scale 2

###### Scale 1

Scale 2

(16,16, 2) (16,36, 2) (16,56, 2)

(16,16, 11) (16,36, 11) (16,56, 11)

(14,14, 1) (14,34, 1)

(14,14, 10) (14,34, 10)

(36,16, 2) (36,36, 2) (36,56, 2)

(36,16, 11) (36,36, 11) (36,56, 11)

<boi>

###### <eoi>

<eoi>

Change

the

dog

to

cat

<boi>

(34,14, 1) (34,34, 1)

(34,14, 10) (34,34, 10)

(3,3,3)

(4,4,4)

(5,5,5)

(6,6,6)

(7,7,7)

(8,8,8)

(9,9,9) (12,12,12)

(0,0,0)

(56,16, 2) (56,36, 2) (56,56, 2)

(56,16, 11) (56,36, 11) (56,56, 11)

Figure 5 Multi-Scale 3D RoPE for interleaved text-image sequences. Text tokens employ diagonal positions (e.g., position 3 → [3,3,3]), while vision tokens utilize normalized spatial coordinates with augmented scale indices. For clarity, we show 2 scales using square images as examples. Note that the next-scale prediction paradigm excludes 1×1 features in the input, beginning with 2×2 feature maps.

Positional Encoding. We introduce Multiscale 3D RoPE to handle interleaved text and multiscale vision tokens, as shown in Figure 5. For text tokens at position t, we simply replicate the position across all three

- 1https://huggingface.co/google/siglip-so400m-patch14-384
- 2https://huggingface.co/google/siglip2-so400m-patch16-naflex

dimensions: (t,t,t). For vision tokens, we encode spatial and scale information explicitly: Each patch at scale s with grid coordinates (i,j) receives position

C √

C √

(j + 0.5),s) (1)

(px,py,ps) = (

(i + 0.5),

HW

HW

where H × W is the grid size and C is a constant range factor (we omit this constant in 5 for simplicity). The 0.5 offset means to center the positional encoding within each patch.

This normalized formulation enables resolution-invariant training: by mapping all spatial positions to a fixed range [0,C], grids of different resolutions (e.g. 16 × 16 and 32 × 32) share the same coordinate space. This eliminates positional extrapolation during high-resolution fine-tuning, as the model encounters only previously learned position ranges.

Following [69], we incorporate additional learnable scale embeddings for vision tokens. To enhance the model’s ability to adapt to varying resolutions during both training and inference, we further introduce scale length positional embeddings. Specifically, we employ sinusoidal encoding over the number of scales, which serves to disambiguate the target output resolution. This design is particularly critical in the VAR framework, where the generative process for images of different sizes follows distinct scale-wise patterns. By explicitly encoding scale length, the model can accordingly adjust its generation paradigm, leading to more coherent and resolution-aware synthesis.

Scale Reweight. In the next-scale prediction paradigm, earlier scales play a crucial role in determining the overall layout and structure of the image. However, these scales contain significantly fewer tokens compared to later scales, creating an imbalanced loss landscape during training. With uniform token weighting, the model tends to prioritize abundant fine-scale tokens at the expense of coarse-scale structural information, leading to layout degradation, particularly severe at high resolutions where the token count disparity becomes extreme.

To address this imbalance, we introduce scale-aware loss reweighting while maintaining the total vision loss constant. Specifically, we assign scale-dependent weights as:

1 (hs × ws)α

(2)

ks =

where hs ×ws represents the spatial resolution at scale s and α is a hyperparameter controlling the reweighting strength. This formulation substantially increases the importance of early-scale predictions, ensuring stable structural generation.

Self Correction with Residual Features. During inference in the next-scale prediction paradigm, tokens within each scale are sampled independently using top-k/top-p sampling. This independence can introduce local conflicts: adjacent positions may sample semantically redundant content due to the lack of joint modeling. More fundamentally, autoregressive models suffer from exposure bias [2, 9, 29, 56]: during training with teacher forcing, the model always conditions on ground-truth tokens from previous steps, but at inference time, it must condition on its own predictions. This train-test mismatch means that errors from earlier scales, which the model was never exposed to during training, tend to propagate and amplify through the generation process [46, 54].

To address this, we introduce a self-correction mechanism during training, inspired by [25, 46]. Rather than deterministically selecting the closest codebook index during encoding, we sample from a multinomial distribution over the top-k nearest indices, while the model continues to predict the top-1 index as the target. This trains the model to correct suboptimal choices from previous scales. In original VAR framework, the input features for each scale are accumulated from all previous scales. However, we find that applying self-correction to accumulated features in our decoder-only architecture leads to performance degradation. We hypothesize that self-correction significantly complicates the input feature space, creating a mismatch with text features that are directly retrieved from the codebook.

We modify the visual input to use residual features directly from the codebook without accumulation. The features of each scale are independently retrieved and upsampled as needed. This approach significantly constrains the complexity of visual input feature space, yielding substantial performance improvements and reducing local artifacts in generated images.

#### 2.3 Optional Diffusion Decoder

Input caption Text encoder

Diffusion Model

Vision indices

Codebook Emb.

Semantic Decoder

Projection

Semantic Feat.

Figure 6 Architecture of the optional diffusion decoder.

While our default VQ decoder achieves satisfactory reconstruction fidelity with high inference efficiency, the discrete quantization process inevitably results in the loss of high-frequency details. To push the boundaries of visual quality, particularly for generation tasks requiring photo-realistic details, we introduce an optional diffusion-based decoder acting as a refinement module.

Conditioning Mechanism. After the next-scale visual index prediction, we obtain the corresponding embeddings from both the semantic and pixel codebooks. The semantic embeddings can be processed through the tokenizer’s semantic decoder to yield high-dimensional semantic features—representations that are explicitly aligned with ground-truth semantic features during tokenizer training [54]. We concatenate these three elements (semantic embeddings, pixel embeddings, and decoded semantic features), pass them through a linear projection layer, and feed the result into the diffusion model as a visual condition. Simultaneously, the diffusion model integrates the image caption via the text branch.

Usage and Trade-offs. Within the NextFlow framework, we observe that the diffusion decoder significantly mitigates detail degradation, particularly in challenging regions such as small-scale faces and text. However, this generative refinement introduces a trade-off: the stochastic nature of the diffusion process may alter fine-grained structures, potentially impacting performance in tasks requiring strict spatial consistency, such as local editing or identity preservation. Unless explicitly stated otherwise, the diffusion decoder is disabled in our reported experiments, including both quantitative evaluations and qualitative visualizations.

### 3 Training Odyssey

Experimental progress is rarely linear. Throughout training, we encountered a variety of challenges. In this section, we present our training odyssey—a chronological account of our training recipes, the issues that arose at each stage, and the solutions we developed to address them. The overall training pipeline is shown in 7.

#### 3.1 Tokenizer Training

The original TokenFlow framework [54] initializes the semantic encoder from pretrained model, while leaving the pixel encoder train from scratch. This leads to semantic features dominating early optimization and ultimately limits reconstruction quality. To mitigate this, we adopt a multi-stage training strategy: We first independently train the pixel branch to establish strong reconstruction capabilities. We then initialize all components—semantic encoder, pixel encoder/decoder, and pixel codebook—from their pre-trained checkpoints and train them jointly. This ensures that both branches begin from a favorable initialization, significantly accelerating convergence and improving final performance. Finally, we double the capacity of the pixel decoder, following architectural insights from [8, 54], and fine-tune it separately. This stage substantially reduces local artifacts and enhances fine details such as small faces and text.

The tokenizer is trained on high-fidelity images with oversampling of face-containing samples to improve the preservation of facial detail. We find that randomly dropping 50% of VAR scales enhances the robustness of numerical distributions in the codebook and results in better reconstruction. Additionally, we enforce fp32 precision during quantization to maintain numerical stability.

Pre-training Post-Training

###### Stage 1: Alignment

###### PT Phase 1

###### PT Phase 2

###### PT Phase 3

###### Stage 4

###### Stage 3

NextFlow

256-res large-scale pretrain

512-res +scale reweight

1024-res HQ subset

RL / GRPO Prefix-tuning

CT & SFT

256-res

0.9

ALIGNMENT PT-256 PT-512 PT-1024 CT&SFT RL

Optional Difusion Decoder (Detail Refinement)

0.8

Geneval Score

0.7

0.6

0.5

###### 0

###### 3.4T

###### 5.2T

###### 5.4T

###### 5.6T+

Start

Tokens

Tokens

Tokens

Final

- Figure 7 Overall training pipeline and the evolution of the Geneval score. The plot below tracks the Geneval score [21] against the cumulative number of training tokens (in trillions). Shaded regions distinguish the distinct stages of our pipeline.

Advantage of TokenFlow-style tokenizer. We validate the efficacy of the TokenFlow-style dual-codebook design by comparing it against a standard single-branch VQGAN. While the single-branch baseline yields marginally higher raw reconstruction fidelity (+0.5 PSNR at 2562), it proves less effective for generative modeling. As shown in Figure 8, under identical training protocols (40k steps, ∼40M samples), the dual-branch tokenizer achieves significantly lower vision loss and consistently superior GenEval scores. We attribute this to the semantic constraints inherent in the dual-branch architecture; these constraints shape a latent space that is structurally easier for the autoregressive model to learn, aligning with findings in REPA [85] and VA-VAE [82].

0 5000 10000 15000 20000 25000 30000 35000 40000

Training Steps

6.5

7.0

7.5

8.0

8.5

9.0

9.5

LossVision

Loss Vision

dual TokenFlow branch

single VQGAN branch

0 5000 10000 15000 20000 25000 30000 35000 40000

Training Steps

0.96

0.98

1.00

1.02

1.04

1.06

1.08

LossText

Loss Text

dual TokenFlow branch

single VQGAN branch

0M 4M 8M 12M 16M 20M 24M 28M 32M 36M 40M

Training data amount (M)

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

Genevalscore

Geneval Performance

dual TokenFlow branch

single VQGAN branch

- Figure 8 Tokenizer Comparison. We compare the training loss and generative performance (GenEval [21]) of our dual branch tokenizer against a single branch VQGAN baseline over 40k steps. Despite slightly lower reconstruction PSNR, the dual-branch architecture demonstrates faster convergence and superior generative capability due to semantically aligned latent structures.

- 3.2 Decoder-only Transformer Training

- 3.2.1 Before the Journey Begins

Output Head Design. Prior to large-scale training, we evaluated two architectural variants: a shared output head that predicts both text and visual tokens within a unified vocabulary, and separate modality-specific heads that employ distinct prediction mechanisms for text and vision. This design choice reflects a common trade-off

in multimodal modeling: shared heads maximize parameter sharing and simplify training [39, 66], while separate heads allow modality-specific optimization [34, 80, 81]. To systematically compare these approaches in our decoder-only framework, we conducted a controlled ablation under a lightweight setting using 5M alignment and 5M supervised fine-tuning (SFT) samples. During alignment, we fine-tuned the adapter and the expanded shared output head for the single-head model, keeping the remaining parameters frozen. For the dual-head model, we fine-tuned the additional vision head along with the adapter. All parameters were fine-tuned during the SFT phase.

As illustrated by the training losses in Figure 9, the single-head architecture consistently demonstrates superior performance. Specifically, it achieves lower total loss and vision loss throughout both the alignment and SFT phases. Furthermore, its text loss during the SFT phase is comparable to that of the dual-head counterpart. Given its architectural simplicity and empirically better performance, we adopted the shared, single-head design for all subsequent large-scale experiments.

###### Loss

###### Vision Loss

###### Text Loss

One Head Alignment Two Head Alignment

One Head Alignment Two Head Alignment

One Head Alignment Two Head Alignment

20

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

22

One Head SFT Two Head SFT

One Head SFT Two Head SFT

One Head SFT Two Head SFT

18

20

16

18

LossValue

14

16

14

12

12

10

10

8

8

0 1000 2000 3000 4000 5000 6000

0 1000 2000 3000 4000 5000 6000

0 1000 2000 3000 4000 5000 6000

Global Step

Global Step

Global Step

- Figure 9 Loss curves for total loss, vision loss and text loss during the lightweight alignment/SFT experiments. Losses are computed using cross-entropy loss.

Impact of Self Correction. We validate self-correction strategies on a 2B parameter model. All experiments are conducted using a high-quality subset of LAION [59] and utilize a tokenizer with a 16,384-size codebook. Each 10,000 training steps corresponds to 5M image-text pairs.

- Figure 10 reveals an intriguing phenomenon: directly applying self-correction with accumulated features (green line, p = 1.0, 30%) degrades performance below the non-corrected baseline (origin line). However, replacing accumulated VAR features with residual features leads to significant improvements. We attribute this to the fundamental difference in feature space complexity: in our decoder-only architecture, self-correction on accumulated features excessively complicates the input space, creating a mismatch with text tokens that are directly retrieved from the codebook. Residual features effectively constrain this complexity while maintaining consistency with the text modality, enabling self-correction to function as intended. Further ablations on self-correction intensity reveal that applying correction with probability p = 1.0 to 60% of visual tokens per scale achieves optimal performance (0.56 at 50k steps).

Impact of Text-only Data. To preserve the original model’s text capabilities and evaluate whether mixing text-only data harms text-to-image performance, we conducted a small-scale ablation. As shown in Table 1, incorporating 25% text data during training does not adversely affect text-to-image generation quality.

Table 1 Geneval scores comparing the pure-t2i baseline and the setting with 25% text mixed.

Setting 12M (iter-5k) 24M (iter-10k) 36M (iter-15k) 48M (iter-20k)

t2i data only 0.266 0.384 0.441 0.505 +25% text 0.265 0.404 0.454 0.499

[Figure 76]

- Figure 10 Ablations of self-correction strategy. Direct application of self-correction degrades performance, while combining it with residual features yields substantial improvements. Using self-correction on 100% training samples and 60% tokens per scale achieves optimal performance.

##### 3.2.2 Alignment

During the alignment stage, we replace the original ViT in Qwen2.5-VL-7B [1] with our vision tokenizer and expand the vocabulary to include vision codes and image boundary tokens ⟨boi⟩, ⟨eoi⟩. We explore two training strategies to align the tokenizer with the language backbone: (1) a two-stage approach that first aligns the connector module and subsequently fine-tunes the output projection layer, and (2) a joint training strategy that optimizes both components simultaneously. To compare these strategies, we monitor performance metrics after an identical supervised fine-tuning (SFT) phase. Under controlled experimental conditions, both methods yield comparable performance. For simplicity, we adopt the joint alignment strategy in our main experiments. We used 10 million image-text pairs for bidirectional alignment tasks (image captioning and text-to-image generation) during this phase. Training is conducted at 256-level resolution.

##### 3.2.3 Pre-Training

During pre-training, all model parameters except those of the tokenizer are trainable. The training corpus consisted of approximately 6 trillion tokens drawn from various sources, including pure text, image-text pairs, editing data, and interleaved multimodal data. We adopt a progressive resolution curriculum across three sub-stages: 256-level, 512-level, and 1024-level pre-training.

256-level Pre-training: In this stage, we conducted large-scale pre-training using around 2 billion text-to-image samples to enable the model to learn a wide range of visual semantics and establish fundamental image generation capabilities. We mixed pure text and multimodal understanding data to maintain the model’s original language and visual comprehension abilities. Additionally, we incorporate 147M interleaved samples, which teach the model to understand relationships across multiple images, laying the foundation for advanced editing capabilities.

512-level Pre-training: Upon transitioning to 512 × 512 resolution, we observed a significant increase in artifacts and structural degradation in generated images. Although overall vision loss decreased, lower-scale losses exhibited an upward trend (Fig. 11). The Geneval score also dropped from 0.67 to 0.57, which we attribute to the substantial increase in the number of tokens per image (approximately 3000 additional tokens) at higher resolution. In VAR architecture, earlier scales play a critical role in the determination of global layout [33, 54, 69]. Equal weighting of tokens disproportionately suppressed learning on these foundational scales. Inspired by flow matching techniques that emphasize difficult intermediate timesteps [17, 36], we introduced a scale-reweighting strategy (Eq. 2) with α = 0.9. This adjustment ensured stable loss reduction on all scales and eliminated localized artifacts.

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

(a) (b)

- Figure 11 Motivation for scale-reweight strategy. Left: Comparison of vision loss curves at 256-level and 512-level pre-training stages, including the first 9 VAR scales. A loss increase is observed in lower scales during 512-level training. Right: (a) Initial state before 512-level training; (b) Generation results after training on 200M tokens. Image quality degrades with increased artifacts and structural anomalies, indicating a need for reweighted loss scaling.

1024-level Pre-training: At 1024 × 1024 resolution, the exponential growth in token count necessitated the use of a carefully curated subset of 40 million high-quality samples. Despite the reduced size of the data set, the tokenizer achieved significantly finer reconstruction of local details. This stage required minimal data to enable high-resolution generation while substantially improving visual fidelity compared to previous stages.

##### 3.2.4 Continue-Training and Supervised Fine-Tuning

After pre-training, we employ a two-phase post-training strategy to refine the models capabilities. First, we conduct continued training (CT) [19, 23] on a curated subset of high-quality data to improve the aesthetic quality of the generated images. While pretrained models demonstrate strong general capabilities, they often produce outputs with inconsistent aesthetic standards due to the heterogeneous nature of pre-training datasets. The CT phase addresses this limitation by fine-tuning aesthetically superior samples while preserving prompt adherence and structural accuracy. Subsequently, we performed supervised fine-tuning (SFT) using a small set of high-quality conversational data. During SFT, we format the data in a dialogue structure and apply supervision exclusively to the model’s responses, enabling more natural and contextually appropriate interactions while further improving generation quality.

##### 3.2.5 Reinforcement Learning

As our architecture follows a pure autoregressive approach, unlike Bagel-like unified models [16, 41, 93], we can directly employ Group Reward Policy Optimization (GRPO) [60] for reinforcement learning [31, 47, 70, 71, 86, 88]. The generation of a VAR sequence can be formulated as a multi-step Markov Decision Process (MDP), where each action at corresponds to generating the token grid for the next resolution level. The policy is defined as:

πtθ(at | st) =

πt,θ(i,j)(at,(i,j) | st), (3)

(i,j)

where at,(i,j) denotes the token at position (i,j), and st represents the prefix {ai}ti=1−1. However, applying RL to a multi-scale VAR generation process introduces unique challenges.

1.2588

Vanilla GRPO

GRPO with prefix 512 GRPO with prefix 256 GRPO with prefix 128 GRPO with prefix 64

1.1844

RewardValue

1.1100

1.0357

0.9613

0 50 100 150 200 250 300 350 400

Training Step

- Figure 12 Prefix-tuning strategy for RL fine-tuning in NextFlow. Only the policies for the coarse scales are optimized, while policies for the finer scales remain frozen. This approach stabilizes training by focusing high-variance RL updates on the most semantically critical generation steps.

As discussed in 3.2.3, early steps in VAR architecture produce coarse grids with few tokens, while later steps generate fine-grained grids with orders of magnitude more tokens. This imbalance causes learning signals from later steps to dominate optimization, destabilizing training and hindering effective policy updates for the initial steps that define global structure. To address this, we introduce two targeted strategies to stabilize the RL fine-tuning process. The first is scale reweight, same as the strategy adopted at the pretraining stage. The second is prefix-tuning strategy. Since low-resolution steps are most critical for the global layout and semantics of the generated image, we concentrate our RL updates on these formative stages. Given the condition c, which may also contains condition images, we roll out a group of image token sequences {siT}Gi=1, and the corresponding decoded images {xi}Gi=1. Within each group, the advantage can be calculated by:

R(xi,c) − mean({R(xi,c)}Gi=1) std({R(xi,c)}Gi=1)

, (4)

Ai =

where R is the reward model. Then the GRPO optimization target is that

m

1 G

LGRPO(θ) =Ec∼C,{si

T }Gi=1∼πθ

t=1

− βDKL(πθ,πref).

kt min

pθ(sit+1|sit,c) pθ

Ai,clip

(sit+1|sit,c)

old

pθ(sit+1|sit,c) pθ

,1 − ϵ,1 + ϵ Ai

(sit+1|sit,c)

old

(5)

As shown in Fig. 12, we find that fine-tuning only the policies for the first m (e.g., m = 8) scales, while keeping the policies for the remaining T − m finer scales frozen, is highly effective. This strategy, termed "prefix-tuning," focuses the limited and high-variance RL signal on the most impactful decisions, avoiding noisy updates to later-stage policies that primarily refine local details. This not only accelerates convergence but also better preserves the generative quality of the pretrained model while adapting its high-level attributes. Together, these two techniques enable stable and efficient RL-based fine-tuning of NextFlow, allowing for precise alignment with downstream objectives without sacrificing generative coherence.

#### 3.3 Diffusion Decoder Training

We explore three model variants: a 1B parameter UNet-based model, and two Transformer-based models with 12B and 18B parameters, respectively. To preserve the text-conditional capabilities of the pre-trained backbones, we inject the discrete tokens strictly via the visual conditioning branch. We avoid concatenating discrete tokens with text embeddings, as our preliminary experiments indicated this would corrupt the textual semantic information. Structurally, the UNet variant adapts its input convolution to match the input channel dimensions, while the Transformer variants utilize an MLP layer to project the discrete tokens into the model’s native dimension.

Training Strategy. We adopt different training strategies based on model scale and computational constraints. The 1B model undergoes full parameter fine-tuning, whereas the 12B and 18B models are trained using Low-Rank Adaptation (LoRA) [27]. Data selection is also tailored to model capacity: the 1B model is trained on high-quality synthetic data, as its limited capacity struggles to fit the complex distribution of real-world data, often resulting in generation artifacts. Conversely, the larger 12B and 18B models are trained on real-world data to maximize reconstruction fidelity. We employ a two-stage curriculum: models are first trained at the base resolution, followed by fine-tuning on a 2 × 2 upsampling task to yield sharper high-frequency details. We rely exclusively on the standard diffusion loss, explicitly avoiding pixel-level losses (e.g., MSE), which we found to inflate PSNR metrics while degrading perceptual quality.

### 4 Infrastructure

###### NextFlow is pre-trained on a cluster equipped with 1024 GPUs. We employ DeepSpeed ZeRO [55] with gradient checkpointing to enable efficient distributed training at scale.

interleaved edit t2i t2i txt edit t2i i2t i2t t2i txt txt

interleaved edit t2i t2i edit t2i i2t i2t t2i txt txt txt

(a) Fixed max length packing

(b) Fixed max computation budget packing

- Figure 13 Illustration of different packing strategy. Fixed max computation budget packing reduces inter-GPU idle time and improving overall training throughput.

Workload Balance. The pre-training of NextFlow involves heterogeneous data types (pure text vs. textto-image vs. interleaved, varying resolution), which introduces significant computational imbalance across GPUs. We address this through a workload balancing strategy during data packing, as shown in Figure 13. Specifically, we precompute the TFLOPS for all sequence lengths and balance workloads accordingly. As demonstrated in Table 3, this strategy significantly reducing inter-GPU idle time, achieving a 4.1× speedup compared to naive approach.

High-Performance Kernel. The large codebook size results in significant memory consumption when computing output logits during training. To address this challenge, we adapt FusedLinearCrossEntropy [26], which fuses the final linear projection and cross-entropy loss computation into a single kernel, reducing peak memory usage by ∼ 20GB per GPU by avoiding the storage of the full logit tensor in memory. We further identified memory-bound operations and fused them into single kernels to minimize redundant memory access, such as RoPE, RMS norm and Flash-Attention [14]. These fused kernels store intermediate results in registers or shared memory, significantly improving arithmetic intensity.

Pre-extract image indices. To minimize computational overhead during large-scale training, we pre-extract image indices offline across all training stages, thereby eliminating online encoding latency. For each image-text sequence, we precompute and store the index sequence along with the original ordering of the data. This approach allows the encoders to remain offloaded from the GPU during training, significantly reducing memory requirements. To support self-correction during training, we incorporate a predefined self-correction probability

Table 2 Training recipe of NextFlow. "M" refers to million. "T" refers to trillion.

Data volume

Stage Resolution Epochs Learning rate

Train tokens Text T2I I2T Editing Interleaved

Alignment 256-level 1 1e-3 - 5M 5M - - 0.01T Pretrain 256-level 1 1e-4 662M 1891M 520M - 147M 3.4T Pretrain 512-level 1 1e-4 700M 345M - 20M 5M 1.8T Pretrain 1024-level 1 1e-4 47M 10M - 2M 0.5M 0.2T

CT 1024-level 1 5e-5 47M 9M - 2M - 0.2T SFT 1024-level 1 1e-5 5M 1M - 0.1M - 0.02T

###### Table 3 Throughput comparison of different packing strategies on 512-level pretraining stage. Tpt denotes throughput per GPU.

Strategy Tpt (tokens/s) Speedup

Batch Padding 620.3 1.0× Fixed Length Packing 2109.5 3.4× Fixed computation budget packing 2517.4 4.1×

(100% samples, 60% tokens per scale). The corresponding input indices and ground-truth indices are stored accordingly. Once pre-extraction is complete, we perform offline data packing using these precomputed samples, further improving training efficiency as detailed in Sec. 4.

### 5 Data

NextFlow is trained on a large scale multimodal dataset which covers a wide range of tasks involving images and texts, including visual understanding, image generation, image editing, image and text interleaved documents generation, text generation. We elaborate how each component of our dataset is built in the following sections.

#### 5.1 Visual Understanding

In the pre-training stage, the model learns to perceive and understand images from text supervision. And thus we collect a large number of image captioning samples from open-source image-text datasets. In addition, we further supplement it with 1) text-rich images, comprising scene text and documents of various types such as table, chart, plot, and so on, to enhance the OCR capability of our model; 2) images with associated world knowledge. The image captions are rewritten using Vision-Language Models (VLM) to improve the text quality and comprehensiveness of the description.

[Figure 89]

[Figure 90]

[Figure 91]

Pl a

A n

L a n

C h a

n e

nts

m a

D

ds c a p e

s

a e

i

i

t

F i n e A rt

N ature

3 3 %

%

I l l u s t r a t i o n

P e r s o n 3 8 . 7 %

Portrait General Objects Mixed Layout

Group

[Figure 92]

[Figure 93]

[Figure 94]

###### ArtandStyles 16.7%

s

- a

- d
- e

O

- b

e

- c

D i g i t a l

M

%

n

A c t i o n

- 2

8

- 3

a

M

Group Digital Ilustration

[Figure 95]

V e h i c l e s

[Figure 96]

rc hite cture

[Figure 97]

[Figure 98]

h aracters

G e n e

a

Ob je c s

Action Fod Architecture Characters

Hierarchical Distribution

###### Visual Samples

I ner: Primary Categories | Outer: Sub-categories

Top 10 categories organized by prominence

- Figure 14 Hierarchical distribution and visual samples of the pre-training dataset. The left panel illustrates the data composition, where the inner ring represents primary categories and the outer ring details fine-grained sub-categories. The right panel presents representative visual samples across top categories.

#### 5.2 Image Generation

We curate a billion scale dataset for text to image generation, with great diversity in terms of image content and image types. The images are gathered from open-source datasets, which exhibit a broad coverage over the content on the Internet; and Megalith [4], CommonCatalog [22], which provide valuable high-quality photographs; as well as our in-house image gallery which further extends the distribution of our data. We first apply a set of heuristic filters to the images and run aesthetic models to rule out images of poor visual quality. For sources whose images we observe have noticeably imbalanced content distribution, we employ a image topic classifier using pretrained SigLip2 in zero-shot manner, to categorize images into topics like landscape, human, animal, plant, food&drinks and so on. Then data resample is performed to obtain a balanced distribution. As shown in recent works [17], caption plays a crucial role in training image generation models with strong instruction following ability. Therefore we caption all the images using VLM to obtain accurate and detailed descriptions. In the CT stage, we introduce a small amount of synthetic data to further boost the aesthetics of our model’s generation.

Traditional Editing Subject-Driven Editing

[Figure 99]

[Figure 100]

Figure 15 Task distribution of the image editing dataset. The data is mainly categorized into two streams: Traditional Editing (left) and Subject-Driven Editing (right). The left panel illustrates the frequency of specific editing operations (e.g., Add, Remove, Replace) colored by their high-level semantic types (Local, Global, View, Text, Style). The right panel displays the distribution of subject-driven tasks, highlighting the diversity in preserving object identity (ID Keep), common objects, abstract content, and visual styles.

#### 5.3 Image Editing

For clarity, we categorize image editing into two distinct tasks: traditional editing and subject driven generation. Traditional editing tackles the problem of transforming an image either locally or globally, while the resulted image always follows the input image in spirit, for example, local editing like object removal, object insertion and global editing like lighting modification, stylization, as well as view editing like zooming in/out, changing the camera’s view angle. In contrast, subject driven image generation aims to create an image of new content which only shares the target subject with the input image, for example, generating an image of the dog running on a grass field, given its close-up photo as input.

##### 5.3.1 Traditional Editing

We start with open-source traditional editing datasets, including UniWorld-V1 [42], RealEdit [65] and ShareGPT-4o-Image [10]. First, we filter edit pairs with mismatched image resolution. Then we identify and remove low quality subsets which contain over 5% bad edits from these datasets through manual examination, followed by a VLM assessment at a granularity of samples. An edit is considered as "bad" if the target image does not respond to or closely follow the edit instruction. The resulted dataset exhibits a task distribution biased towards common and simple editing tasks such as adding, removing an object or replacing an object

with another, which substantially limits the model’s capability to perform complex, varied real-world editing. As a result, we build an additional synthetic dataset which span over a diverse spectrum of traditional editing tasks. The task distribution is significantly more balanced than the open-source counterpart as illustrated in Figure 15.

##### 5.3.2 Subject-driven Generation

We observe that subjects like human, animal, objects and even abstract content, often reoccur across real world data. And this fact indicates there are numerous natural demonstrations for diverse tasks of subject-driven generation. We collect images that co-occur on the Internet and utilize VLM to identify image pairs sharing common subjects, which turns out to be a scalable way to build the training data. Compared to workflow methodology which heavily relies on generative models, our method inherently produces data with a significant advantage in terms of task diversity as well as subject consistency.

#### 5.4 Interleaved Generation

To endow the model with the ability to generate coherent interleaved image-text sequences, we construct a large-scale video-text dataset. We treat video clips as sequences of interleaved frames and leverage the temporal continuity of video to model narrative progression. Our data pipeline aggregates sources including OmniCorpus-CC, OmniCorpus-YT [40], and Koala36M [72], processed through a rigorous multi-stage filtering strategy.

Quality and Heuristic Filtering. We apply strict quality controls to the raw video corpus, specifically targeting the Koala36M subset. To minimize fragmentation errors during training, we discard long clips exceeding 20 seconds. Visual quality is ensured by filtering for high aesthetic scores (> 4.3, retaining the top 30%) and clarity (> 0.7, retaining the top 50%). Furthermore, to avoid degenerate solutions where the model generates static images, we require a motion score greater than 4. This pipeline removes approximately 75% of the raw data, resulting in a refined subset of 8M clips free from blur, static scenes, and low-aesthetic content.

Semantic Balancing. To prevent the model from overfitting to dominant categories, we employ SigLIP [87] to classify video content. We downsample overrepresented classes, removing 50% of clips containing "person" (reducing the total volume by 20%) and 50% of "television news broadcast" clips (reducing the total by 10%). This results in a balanced corpus of approximately 5.3M clips.

Motion-Adaptive Frame Selection. We extract frames at a baseline rate of 0.5 FPS (one frame every 2 seconds). However, uniform sampling often captures redundant frames in static scenes or blurry frames during rapid camera movement. We introduce an optical flow-based filtering mechanism using RAFT [68] to refine frame selection:

- • Static Filtering: Frames with negligible optical flow magnitude are discarded.
- • Camera vs. Object Motion: We analyze the variance of flow directions. Low variance implies global camera movement (pan/zoom), which we discard to focus on semantic content changes. High variance indicates independent object motion, which is preserved.
- • Large Displacements: Frames exhibiting significant structural changes (high flow magnitude) are retained to capture scene transitions.

During training, we constrain the input to a maximum of 5 frames at 512px resolution or 3 frames at 1k resolution.

Transition Text Generation. Finally, to bridge the visual gaps between selected frames, we utilize VLMs to generate coherent transition texts, effectively converting the video clips into interleaved image-text documents.

#### 5.5 Text Generation

To ensure NextFlow maintains strong pure-text instruction following and reasoning capabilities, we incorporate high-quality text-only data into the training mix. We source general-purpose instruction data from Nemotron-

###### Table 4 Comparison of text-to-image generation ability on DPG [28] benchmark. For AR Models, the best results are in bold and the second-best are underlined.

Model Global Entity Attribute Relation Other Overall↑ Diffusion Models

SDXL [53] 83.27 82.43 80.91 86.76 80.41 74.65 DALL-E 3 [3] 90.97 89.61 88.39 90.58 89.83 83.50 FLUX.1 [Dev] [35] 74.35 90.00 88.96 90.87 88.33 83.84 SD3 Medium [17] 87.90 91.01 88.83 80.70 88.68 84.08 GPT Image 1 [High] [50] 88.89 88.94 89.84 92.63 90.96 85.15 HiDream-I1-Full [6] 76.44 90.22 89.48 93.74 91.83 85.89 Seedream 3.0 [19] 94.31 92.65 91.36 92.78 88.24 88.27 Qwen-Image [75] 91.32 91.56 92.02 94.31 92.73 88.32

AR Models

TokenFlow-XL [54] 73.38 78.72 79.22 81.29 85.22 71.20 Emu3-Gen [74] 85.21 86.68 86.84 90.22 83.15 80.60 Janus-Pro-7B [11] 86.90 88.90 89.40 89.32 89.48 84.19 NextStep-1 [67] - - - - - 85.28 EMU3.5 [13] - - - - - 88.26 NextFlow 92.40 90.05 90.51 92.72 91.14 86.00 NextFlow-RL 92.09 93.48 91.92 92.08 93.22 88.32

CC HQ [64] to maintain conversational fluency and general knowledge. Additionally, we integrate mathematical reasoning data from MegaMath [94] to enhance the model’s logical reasoning and problem-solving abilities.

### 6 Model Performance

#### 6.1 Image Generation

We conduct a comprehensive evaluation of NextFlow across multiple dimensions, including prompt following, world knowledge, and aesthetic quality. The quantitative results demonstrate that our autoregressive approach, particularly when enhanced with reinforcement learning, achieves state-of-the-art performance comparable to or exceeding top-tier models.

As shown in Tables 5 and 4, NextFlow demonstrates exceptional prompt following capabilities, with our RL-finetuned model achieving state-of-the-art scores (0.84 on GenEval [21] and 88.32 on DPG [28]) that outperform strong diffusion baselines like FLUX.1-dev [35] and match top-tier performers.

A critical advantage of our large-scale pre-training on diverse multimodal data is reflected in the WISE benchmark [48] (Table 6). NextFlow RL achieves an overall score of 0.62, matching the performance of Qwen-Image and significantly outperforming other autoregressive models like Show-o (0.30) and Janus-Pro-7B (0.35). The model demonstrates robust understanding across cultural, spatial, and physical domains, suggesting that our unified architecture effectively internalizes complex world knowledge.

On PRISM-Bench [18] (Table 7), which evaluates broader generative aspects including style and affect, NextFlow RL achieves an overall score of 78.8. This result places our model on par with top-tier systems like Seedream 3.0 [20] and Qwen-Image [75], demonstrating that our unified architecture achieves competitive aesthetic quality and text rendering capabilities alongside its strong instruction following.

#### 6.2 Image Editing

We assess the image editing capabilities of NextFlow across a diverse set of benchmarks, ranging from traditional instruction-based editing to complex subject-driven generation. As evidenced by the quantitative results, our unified autoregressive framework, particularly when enhanced with reinforcement learning, establishes new state-of-the-art performance levels.

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

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

###### Figure 16 NextFlow text-to-image visualization.

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Edit/text_edit/position Move the 'STARCRAFT' text from the top right corner to the top center of the image

Edit/text_edit/add Add the word 'Happy' in a white cursive font above the existing 'hello' text to form the phrase 'Happy hello Fall'

Edit/text_edit/remove Remove the 'PARADOX' text from the side of the black and yellow heavy machinery

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Edit/view_edit/camera_zoom_out Zoom out the camera to reveal more of the surrounding pastries and the wooden table surface, providing a wider view of the entire pastry arrangement.

SubjectDrivenGeneration/common_object/object_extract Extract the wooden double door from the reference image and isolate it on a plain white background.

SubjectDrivenGeneration/common_object/object_quantity Generate an image of three identical turquoise rings (matching the reference) placed on a black velvet jewelry tray.

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

SubjectDrivenGeneration/abstract_content/logo_extract Extract the Longform logo (red lighthouse icon with 'LONGFORM' text below it) from the reference email image and isolate it on a plain white background.

SubjectDrivenGeneration/abstract_content/material_apply Apply the soft pink fabric material from the reference image to a sleeveless maxi dress, and place the dress on a black mannequin in a modern boutique window with gold accents, sunlight filtering through sheer curtains, and a small table with a vase of roses nearby.

SubjectDrivenGeneration/abstract_content/logo_extract Extract and isolate the "My Hero One's Justice 2" logo from the input image and place it on a plain white background.

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

SubjectDrivenGeneration/abstract_content/photo_place Generate an image of the reference landscape photo framed in a rustic wooden frame, placed on a weathered oak desk inside a cozy mountain cabin with a stone fireplace, a wool blanket draped over a nearby armchair, and pine trees visible through the cabin windows.

SubjectDrivenGeneration/common_object/object_view Generate an image of the rear view of the classic car parked on a cobblestone street in a quaint French village, with pastel-colored cottages, flower-adorned balconies, and a small caf\u00e9 with outdoor seating across the street.

SubjectDrivenGeneration/abstract_content/design_apply Apply the decorative black metal scrollwork design from the bed frame to a garden gate leading into a cottage garden with pink climbing roses, a weathered wooden fence, and a stone pathway lined with lavender.

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

SubjectDrivenGeneration/common_object/object_state Generate an image of the chocolate cake from the reference image with a slice removed, placed on a rustic wooden cutting board. Include a silver fork resting beside the cake and a folded linen napkin underneath the cutting board, with a neutral-colored kitchen wall in the background.

SubjectDrivenGeneration/human/face_id_keep Generate an image of the man from the reference image (preserving his face identity) dressed in a dark formal suit, standing on a conference stage with a microphone in hand, presenting to an audience while a projection screen behind him displays voice research slides.

SubjectDrivenGeneration/abstract_content/logo_place Place the Young Workers Toolbox logo on a red cotton tote bag that is hanging from a metal fence at a community garden

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

SubjectDrivenGeneration/common_object/object_extract Extract the potted bamboo plant from the reference image and place it in a bright, sunlit living room with a beige leather sofa, a wooden coffee table, and large floor-to-ceiling windows overlooking a green garden

SubjectDrivenGeneration/abstract_content/material_apply Use the copper metallic material from the lipstick packaging in the reference image to create a small cylindrical candle holder, and place it on a dark walnut shelf next to a jar of sea salt and a sprig of rosemary.

SubjectDrivenGeneration/common_object/object_quantity Generate three identical silver necklaces with intricate floral designs and a central teardrop-shaped gemstone, displayed on a plush black velvet jewelry stand with a soft gold trim, arranged in a cascading formation.

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

SubjectDrivenGeneration/visual_style/style_apply Generate a black and white halftone engraving-style image of a sunflower growing in a meadow, using the visual style of the reference botanical illustration of a camellia branch.

SubjectDrivenGeneration/human/character_place Generate an image of the comic queen character in the purple and gold gown from the reference image placed in a mystical forest with glowing silver trees, a crystal-clear lake reflecting the moon, and bioluminescent blue flowers. She stands by the lake holding a glowing blue orb in her right hand, with a calm expression, instead of being held by the red-haired character as in the original.

SubjectDrivenGeneration/abstract_content/photo_extract Extract the still life painting featuring a wine bottle, a stemmed wine glass, and a lemon on a tray from the reference image and present it as a standalone image

SubjectDrivenGeneration/common_object/object_place Generate an image of the pink tart with whipped cream and decorative sprinkles in a new setting: placed on a red and white checkered picnic blanket spread over green grass, surrounded by small yellow dandelions and purple clover, with a wicker picnic basket holding a linen napkin in the background.

SubjectDrivenGeneration/abstract_content/ph oto_extract Extract the photograph from the pink ornate frame and present it as a standalone image

###### Figure 17 Edit results of NextFlow on EditCanvas benchmark.

- Table 5 Comparison of text-to-image generation ability on GenEval [21] benchmark. † refer to the methods using LLM rewriter. For AR Models, the best results are in bold and the second-best are underlined.

Model Single Obj. Two Obj. Count. Colors Pos. Color Att. Overall↑ Diffusion Models

SDXL [53] 0.98 0.74 0.39 0.85 0.15 0.23 0.55 DALL-E 3 [3] 0.96 0.87 0.47 0.83 0.43 0.45 0.67 SD3-Medium [17] 0.99 0.94 0.72 0.89 0.33 0.60 0.74 FLUX.1-dev† [35] 0.98 0.93 0.75 0.93 0.68 0.65 0.82 HiDream-I1-Full [6] 1.00 0.98 0.79 0.91 0.60 0.72 0.83 GPT Image 1 [High] [50] 0.99 0.92 0.85 0.92 0.75 0.61 0.84 Seedream 3.0 [19] 0.99 0.96 0.91 0.93 0.47 0.80 0.84 Qwen-Image [75] 0.99 0.92 0.89 0.88 0.76 0.77 0.87

AR + Diffusion Models

Transfusion [93] - - - - - - 0.63 BAGEL† [16] 0.98 0.95 0.84 0.95 0.78 0.77 0.88

AR Models

Chameleon [66] - - - - - - 0.39 Emu3-Gen [74] 0.98 0.71 0.34 0.81 0.17 0.21 0.54 TokenFlow-XL† [54] 0.95 0.60 0.41 0.81 0.16 0.24 0.55 Show-o [80] 0.98 0.80 0.66 0.84 0.31 0.50 0.68 NextStep-1† [67] - - - - - - 0.73 Janus-Pro-7B [11] 0.99 0.89 0.59 0.90 0.79 0.66 0.80 Infinity-8B† [25] 0.99 0.89 0.59 0.90 0.79 0.66 0.80 EMU3.5† [13] - - - - - - 0.86 NextFlow† 0.98 0.92 0.73 0.90 0.77 0.69 0.83 NextFlow-RL† 1.00 0.92 0.75 0.90 0.76 0.70 0.84

On the ImgEdit benchmark [83], which covers a wide spectrum of editing operations including addition, removal, and style transfer, NextFlow RL achieves the highest overall score of 4.49, surpassing strong baselines such as Qwen-Image (4.27) and Emu3.5 (4.41). Notably, our model demonstrates exceptional precision in the Adjust (4.68) and Remove (4.67) sub-tasks, indicating a robust ability to manipulate specific image regions without disrupting the global context. OmniContext benchmark [76] evaluate the in-context editing and subject preservation ability. In the single-subject setting (Table 9), NextFlow RL achieves a Subject Consistency (SC) score of 9.22, significantly outperforming models like OmniGen2 (8.34) and even surpassing the proprietary GPT-4o (9.03) in maintaining subject fidelity. This result highlights the effectiveness of our interleaved pre-training in learning robust identity representations. On GEdit-Bench [45] (Table 10), which explicitly measures the trade-off between semantic consistency and perceptual quality, NextFlow RL achieves the best overall score 7.87.

Comprehensive Evaluation on EditCanvas. Finally, we report results on our proposed EditCanvas benchmark (Table 11). To address the limitations of existing datasets that often focus on isolated capabilities or lack granularity, EditCanvas establishes both Traditional Editing and Subject-Driven Generation across 56 finegrained tasks with over 5,000 high-quality samples (please refer to Appendix B for detailed dataset statistics and comparisons). On this rigorous benchmark, NextFlow RL demonstrates balanced excellence, achieving an overall score of 8.04. It shows particular strength in Subject-Driven Generation (8.78). These results confirm that our "next-scale prediction" paradigm allows for precise local modifications while maintaining the high aesthetic quality required for creative workflows, fully leveraging the comprehensive evaluation scope provided by EditCanvas.

#### 6.3 Interleaved Generation

To evaluate the unified modeling capabilities of NextFlow, we test its performance on interleaved multimodal generation tasks. Unlike pipeline approaches that rely on separate text and image models, our unified autoregressive framework treats text and visual tokens as a single continuous sequence. Figure 18 demonstrates the model’s ability to generate coherent, alternating sequences of text and images across distinct domains.

- Table 6 Comparison of text-to-image generation ability on WISE [48] benchmark. WISE examines the complex semantic understanding and world knowledge for T2I generation. For AR Models, the best results are in bold and the second-best are underlined.

Model Cultural Time Space Bio. Phys. Chem. Overall↑ Diffusion Models

SDXL [53] 0.43 0.48 0.47 0.44 0.45 0.27 0.43 SD3-Medium [17] 0.42 0.44 0.48 0.39 0.47 0.29 0.42 FLUX.1-dev [35] 0.48 0.58 0.62 0.42 0.51 0.35 0.50 GPT Image 1 [High] [50] 0.81 0.71 0.89 0.83 0.79 0.74 0.80 Qwen-Image [75] 0.62 0.63 0.77 0.57 0.75 0.40 0.62

AR + Diffusion Models

BAGEL [16] 0.44 0.55 0.68 0.44 0.60 0.39 0.52

AR Models

Show-o [80] 0.28 0.36 0.40 0.23 0.33 0.22 0.30 Emu3-Gen [74] 0.34 0.45 0.48 0.41 0.45 0.27 0.39 Janus-Pro-7B [11] 0.30 0.37 0.49 0.36 0.42 0.26 0.35 Liquid [77] 0.38 0.42 0.53 0.36 0.47 0.30 0.41 NextFlow 0.62 0.60 0.70 0.54 0.58 0.38 0.59 NextFlow-RL 0.63 0.63 0.77 0.58 0.67 0.39 0.62

###### Visualization Results

###### Figure 1: Comparison of generated samples. (Click to edit title)

"In a sun-drenched farmyard nestled beside a glistening pond, a mother duck sits proudly on her nest. Among the broken shells of hatched eggs, one large, peculiar egg finally cracks open. Out steps a clumsy, grey, and oversized duckling, looking vastly different from his small, yellow, and fluffy siblings. The farm animals, including chickens and a grumpy dog, stare at the newcomer with confusion and disdain, highlighting his awkward appearance against the idyllic rural backdrop.",

"As the seasons turn, the grey duckling wanders alone through a marshy wetland, his head hung low in sadness. He is being pecked at and chased away by other ducks and geese who mock his strange looks. The sky is overcast, reflecting his gloomy mood. He gazes at his own distorted reflection in a puddle, feeling like an outcast, while a group of elegant birds flies high in the distance, ignoring his plight.",

"Winter has arrived with a vengeance, covering the landscape in a thick blanket of white snow and ice. The poor duckling, now shivering and huddled near a frozen reed bed, struggles to survive the biting cold. The wind howls around him, and the world seems desolate and unforgiving. He is curled up, trying to preserve warmth, a small speck of grey life amidst the vast, freezing blue and white wilderness.",

"Spring returns, melting the ice and bringing vibrant green life back to the pond. The duckling, who has survived the winter, steps into the water and stretches his wings. To his astonishment, he notices his reflection in the crystal-clear water has changed completely. He is no longer grey and awkward; his feathers are now pure white, his neck is long and graceful, and his wings are powerful and majestic.",

"Now a magnificent swan, he glides effortlessly across the shimmering lake, surrounded by other beautiful swans who welcome him as one of their own. Children playing by the water's edge point in awe at his beauty, cheering for him. The sun sets in a warm, golden glow behind him, illuminating his white feathers as he realizes he was never ugly, just different, and has finally found where he truly belongs."

|[Figure 161]|
|---|

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

"On a clean wooden countertop, all the raw ingredients for a tomato basil pasta are laid out in an organized manner. In the center, there is a pile of dry spaghetti. Surrounding it are three bright red tomatoes, a bunch of fresh green basil leaves, a bulb of garlic, a bottle of olive oil, and a block of parmesan cheese. The lighting is neutral and bright, showing a topdown view of the preparation stage.",

"The scene shifts to the chopping phase. A wooden cutting board takes center stage on the counter. A sharp chef's knife rests beside a pile of roughly diced red tomatoes and finely minced garlic cloves. To the side, the fresh basil has been picked from the stems and sits in a small bowl. The view remains a direct overhead shot, focusing on the prepared ingredients ready for cooking.",

"Now, the cooking process begins. A large stainless steel pan is placed on a stovetop burner. Inside the pan, the diced tomatoes and minced garlic are sizzling in hot olive oil, creating a rich, chunky sauce. Steam rises slightly from the pan. Next to the pan, a large pot of boiling water bubbles vigorously, waiting for the pasta. The perspective is a straightdown look into the cookware.",

"The pasta has been cooked and combined with the sauce. The stainless steel pan now holds the cooked spaghetti, which is being tossed with the tomato and garlic sauce. The sauce coats the noodles evenly, turning them a vibrant reddish-orange. Fresh basil leaves are being scattered on top of the hot pasta, wilting slightly from the heat. The top-down view captures the action of mixing the dish.",

"The final dish is plated and ready to serve. A white ceramic plate sits on the wooden table, holding a neat mound of the tomato basil pasta. Grated parmesan cheese has been sprinkled generously over the top, adding a snowy texture. A fork rests on the side of the plate, and a glass of water is placed nearby. The overhead shot presents the completed meal clearly and simply."

|[Figure 166]|
|---|

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

[Figure 170]

"A sleek silver sedan is cruising down a coastal highway during the golden hour. The sun is low on the horizon, casting long, warm shadows across the asphalt. The car is positioned in the center of the lane, with the ocean visible on the right side, shimmering under the orange light. The wheels are spinning in a blur of motion, and the reflection of the setting sun glints sharply off the car's polished hood.",

"The silver sedan continues its forward motion on the coastal highway, moving just a fraction further along the road. The golden sunlight shifts slightly across the curvature of the car's body as the angle changes. The ocean waves in the background have rolled slightly closer to the shore, and the white lane markers on the asphalt have blurred backwards just a few feet, emphasizing the speed.",

"Driving steadily, the silver sedan's position remains focused, but the background scenery has panned slightly to the left. The reflection of the sun on the hood intensifies as the car aligns more directly with the light source. The shadow of the car on the road stretches marginally longer, and the distant cliffs on the coastline appear just a tiny bit larger as the vehicle approaches them.",

"The car speeds along the smooth asphalt, the tires kicking up a very faint, almost invisible dust from the road surface. The golden light of the sunset is now reflecting off the side windows, creating a warm glow inside the cabin. The white dashed line on the road has moved further back, and a seagull in the distant sky has flapped its wings once, changing its position slightly against the clouds.",

"As the silver sedan drives on, the perspective remains locked on the vehicle's steady momentum. The sunlight flare on the windshield shifts position slightly as the car follows the gentle curve of the road. The ocean on the right is now passing by a bit faster in the peripheral vision, and the warm orange hues of the sky have deepened ever so slightly as the sun dips a fraction lower."

|[Figure 171]|
|---|

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

###### Figure 18 NextFlow interleaved generation results. Our unified sequential modeling framework can produce text descriptions and corresponding images in an alternating sequence. We show diverse applications including storytelling (top), recipe instructions (middle), and dynamic scene generation (bottom).

- Table 7 Quantitative results on PRISM-Bench evaluated by GPT-4.1. PRISM-Bench [18] measures image generation performance on multiple aspects including imagination, entity accuracy, style, and long-text alignment. Imag., Comp., and Text Rend. denote Imagination, Composition, and Text Rendering, respectively. For AR Models, the best result is in bold and the second best is underlined.

Model Imag. Entity Text Rend. Style Affect. Comp. Long Text Overall↑ Diffusion Models

SDXL [52] 58.2 70.0 25.4 73.9 78.0 75.4 41.9 60.4 SD3-Medium [17] 63.3 60.6 43.0 75.2 79.5 82.3 53.8 65.4 SD3.5-Large [63] 72.3 74.3 58.9 80.7 86.2 85.9 58.0 73.7 FLUX.1-dev [35] 71.1 71.0 56.3 76.4 89.7 86.8 64.6 73.7 HiDream-I1-Dev [7] 69.0 69.5 58.8 73.7 83.7 83.7 52.8 70.2 HiDream-I1-Full [7] 75.0 73.4 64.3 83.1 89.5 87.8 57.9 75.9 SEEDream 3.0 [20] 76.9 77.0 63.2 85.7 89.8 89.8 75.0 79.6 Qwen-Image [75] 79.6 76.3 61.6 86.6 90.4 90.3 74.5 79.9

AR + Diffusion Models

Bagel [16] 68.7 54.6 37.4 69.6 81.6 81.8 61.7 65.1 Bagel-CoT [16] 71.3 61.2 31.7 67.3 83.8 83.2 58.0 65.2

AR Models

JanusPro-7B [12] 68.1 59.5 26.1 72.6 75.4 72.4 51.1 60.7 NextFlow 84.3 71.2 50.3 80.2 82.8 71.2 70.8 74.7 NextFlow-RL 87.1 79.3 49.8 83.8 88.8 90.5 72.4 78.8

#### 6.4 CoT Reasoning

A distinct advantage of our NextFlow architecture is its native support for interleaved text-image generation, which allows for the seamless integration of Chain-of-Thought (CoT) reasoning prior to visual synthesis. To investigate the potential of this paradigm in handling implicit constraints and cultural nuances, we conducted an exploratory study using a curated dataset of 1M instruction-reasoning-image triplets.

As illustrated in Figure 19, standard generation often struggles with prompts containing latent ambiguities. For instance, baseline models may render a Red Panda for "China’s national treasure" or generate autumnal red leaves for a "maple leaf in summer," failing to resolve logical conflicts. By fine-tuning NextFlow to articulate a reasoning trace—analyzing the prompt for cultural context and physical constraints—before predicting visual tokens, the model effectively self-corrects these semantic discrepancies.

We evaluated this on WISE [48]. This experiment is conducted as a controlled study on an intermediate model checkpoint. The integration of CoT yielded a substantial performance boost, raising the WISE score from 0.60 to 0.70. This significant relative improvement demonstrates that enabling the model to "think" before "drawing" creates a strong inductive bias for logical consistency and superior prompt adherence.

#### 6.5 In-context Learning

Benefiting from large-scale pre-training on interleaved image-text data, NextFlow demonstrates the capability to perform in-context learning. As shown in Fig. 20, when prompted with instructions to "infer the transformation pattern from the preceding examples and apply it to the target image," the model is able to capture the stylistic mapping and modifies the input accordingly.

#### 6.6 Image Reconstruction

Quantitative Evaluation. As detailed in Table 12, our optimized training recipe yields substantial gains over the baseline TokenFlow architecture. On ImageNet-1K (5122), NextFlow achieves a marked improvement of +2.08 dB in PSNR, validating the efficacy of our multi-stage training and scale dropout strategies. Scaling to 10242 further enhances fidelity, reaching a PSNR of 28.04 on our internal benchmark, demonstrating the

[Figure 176]

[Figure 177]

[Figure 178]

A powerful marsupial, a national symbol of Australia.

The user wants an image of a powerful marsupial that is a national symbol of Australia. The key elements are "powerful," "marsupial," and "national symbol of Australia." Common national symbols of Australia include kangaroos, koalas, and maybe the platypus, but kangaroos are the most iconic. Since "powerful" is mentioned, perhaps a large kangaroo, maybe a red kangaroo, which is the largest. Need to describe the kangaroo in detail, its posture to show power, maybe in a natural Australian setting like the outback, with elements that highlight its status as a national symbol, such as a clear sky or iconic Australian landscape.

The final prompt is: A majestic, powerful red kangaroo standing tall in the Australian outback, its muscular hind legs planted firmly on the red earth, chest puffed out to showcase its strength. The kangaroo's thick fur glows in the warm sunlight, with its large, powerful tail balancing its massive body, embodying the resilience and iconic status of Australia as a national symbol. In the background, a vast, arid landscape stretches to the horizon, dotted with sparse eucalyptus trees and a clear blue sky, emphasizing the kangaroo's connection to the land it calls home.

[Figure 179]

[Figure 180]

[Figure 181]

A national treasure animal of China.

The user wants an image of a national treasure animal of China. Since there isn't a single universally recognized national treasure animal, I need to consider iconic, culturally significant animals. Common candidates could be pandas (national symbol), golden monkeys (representing China's forests), or perhaps the giant panda, as it's globally famous. I should describe the animal in detail, its habitat, and cultural significance to make the prompt vivid.

The final prompt is: A majestic giant panda, the national treasure animal of China, sitting peacefully on a bed of lush green bamboo in a misty mountain forest. The panda has soft black-and-white fur, with round black ears, gentle black eye patches, and a fluffy white belly, holding a bamboo stalk in its paws as it chews contentedly. Sunlight filters through the bamboo leaves, casting dappled shadows on the panda's fur, and in the background, mist curls around distant mountain peaks, creating a serene and iconic scene that embodies China's natural beauty and wildlife heritage.

[Figure 182]

[Figure 183]

[Figure 184]

A close-up of a maple leaf in summer.

The user wants an image of a maple leaf in summer, but "summer" is conflicting with "closeup" since maple leaves are deciduous and typically lose their leaves in autumn. However, the user might intend a vibrant, healthy maple leaf in summer, perhaps with rich green color, maybe some details like veins, texture, and natural lighting. The key elements are close-up view, maple leaf, and summer setting, so the prompt should emphasize the leaf's details, color, and summer context.

The final prompt is: A detailed close-up photograph of a single maple leaf in the height of summer, showcasing its vibrant, glossy green color with intricate veining and serrated edges. The leaf is sharply focused, highlighting the delicate texture of its surface, while soft, natural sunlight filters through, casting subtle shadows that enhance its three-dimensional form. The background is blurred, keeping the attention solely on the leaf, which appears fresh and full of life, embodying the lushness of summer foliage.

###### Figure 19 Qualitative comparison of text-to-image generation with and without reasoning. Left: Baseline outputs generated directly from prompts, often failing to capture implicit constraints or cultural context (e.g., generating a Red Panda instead of a Giant Panda for "China’s national treasure"). Middle: The input prompt coupled with the generated reasoning traces. Right: Final outputs leveraging the thinking process.

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

###### Figure 20 Visualization of in-context learning. The model can learn the transformation pattern in the first two example pairs and applies a similar modification to the input image.

- Table 8 Quantitative comparison results on ImgEdit [83]. All metrics are evaluated by GPT-4.1. “Overall” is calculated by averaging all scores across tasks.

Model Add Adjust Extract Replace Remove Background Style Hybrid Action Overall↑

MagicBrush [89] 2.84 1.58 1.51 1.97 1.58 1.75 2.38 1.62 1.22 1.90 Instruct-Pix2Pix [5] 2.45 1.83 1.44 2.01 1.50 1.44 3.55 1.20 1.46 1.88 AnyEdit [84] 3.18 2.95 1.88 2.47 2.23 2.24 2.85 1.56 2.65 2.45 UltraEdit [92] 3.44 2.81 2.13 2.96 1.45 2.83 3.76 1.91 2.98 2.70 OmniGen [79] 3.47 3.04 1.71 2.94 2.43 3.21 4.19 2.24 3.38 2.96 ICEdit [91] 3.58 3.39 1.73 3.15 2.93 3.08 3.84 2.04 3.68 3.05 Step1X-Edit [45] 3.88 3.14 1.76 3.40 2.41 3.16 4.63 2.64 2.52 3.06 BAGEL [16] 3.56 3.31 1.70 3.30 2.62 3.24 4.49 2.38 4.17 3.20 UniWorld-V1 [43] 3.82 3.64 2.27 3.47 3.24 2.99 4.21 2.96 2.74 3.26 OmniGen2 [76] 3.57 3.06 1.77 3.74 3.20 3.57 4.81 2.52 4.68 3.44 FLUX.1 Kontext [Pro] [38] 4.25 4.15 2.35 4.56 3.57 4.26 4.57 3.68 4.63 4.00 GPT Image 1 [High] [50] 4.61 4.33 2.90 4.35 3.66 4.57 4.93 3.96 4.89 4.20 Qwen-Image [75] 4.38 4.16 3.43 4.66 4.14 4.38 4.81 3.82 4.69 4.27 Emu3.5 [13] 4.61 4.32 3.96 4.84 4.58 4.35 4.79 3.69 4.57 4.41 NextFlow 4.34 4.68 4.23 4.47 4.58 4.30 4.61 4.20 4.56 4.44 NextFlow-RL 4.33 4.68 4.22 4.50 4.67 4.38 4.74 4.11 4.77 4.49

- Table 9 Comparison on task type SINGLE from OmniContext [76]. Prompt Following (PF), Subject Consistency (SC), and Overall scores are reported (higher is better, ↑).

SINGLE↑

Model

Character Object Average PF SC Overall PF SC Overall PF SC Overall

Flux.1 Kontext max [38] 7.98 9.24 8.48 8.78 8.76 8.68 8.38 9.00 8.58 Gemini-2.0-flash [24] 5.54 5.98 5.06 6.17 5.89 5.17 5.86 5.93 5.11 GPT-4o [49] 8.89 9.03 8.90 9.40 8.74 9.01 9.14 8.88 8.95

InfiniteYou [32] 7.81 5.15 6.05 - - - - - UNO [78] 7.56 6.48 6.60 7.78 6.65 6.83 7.67 6.56 6.72 BAGEL [16] 7.72 4.86 5.48 8.56 6.06 7.03 8.14 5.46 6.25 OmniGen [79] 7.12 7.58 7.21 7.66 5.04 5.71 7.39 6.31 6.46 OmniGen2 [76] 8.04 8.34 8.05 8.44 7.26 7.58 8.24 7.80 7.81 NextFlow 7.66 9.04 8.23 8.96 8.94 8.89 8.31 8.99 8.56 NextFlow-RL 7.96 9.22 8.50 9.00 8.76 8.84 8.48 8.99 8.67

tokenizer’s robustness in high-resolution scenarios.

Qualitative Comparison. Figure 21 presents the visual comparison. The default VQ decoder demonstrates robust reconstruction capabilities, faithfully preserving the structural layout and semantic integrity of the input. The optional diffusion decoder acts as a perceptual enhancer, further refining high-frequency textures in complex regions, such as small faces and texts. However, we observe that this generative refinement involves a trade-off: while it boosts perceptual realism, the VQ decoder ensures higher fidelity to the original signal, avoiding the potential detail alteration inherent in diffusion-based re-synthesis.

#### 6.7 Multimodal Understanding

We observe that simultaneously supporting multimodal understanding and image generation within a dense 7B decoder-only model imposes a significant capacity bottleneck. Furthermore, due to the scarcity of our high-quality pre-training data for multimodal understanding, we minimize the understanding training in the late pre-training stage. We only employed a task inversion where text-to-image samples were converted into image captioning samples with a probability of 10%.

To evaluate the understanding capabilities of our base model, we conducted SFT experiments with increasing

- Table 10 Comparison on GEdit-Bench [45]. G_SC: Semantic Consistency, G_PQ: Perceptual Quality, G_O: Overall Score (geometric mean of G_SC and G_PQ). All metrics evaluated by GPT-4.1.

Model G_SC G_PQ G_O↑

Instruct-Pix2Pix [5] 3.58 5.49 3.68 AnyEdit [84] 3.18 5.82 3.21 MagicBrush [89] 4.68 5.66 4.52 UniWorld-v1 [43] 4.93 7.43 4.85 OmniGen [79] 5.96 5.89 5.06 OmniGen2 [76] 7.16 6.77 6.41 Gemini 2.0 [15] 6.73 6.61 6.32 BAGEL [16] 7.36 6.83 6.52 FLUX.1 Kontext [Pro] [38] 7.02 7.60 6.56 Step1X-Edit [45] 7.66 7.35 6.97 GPT Image 1 [50] 7.85 7.62 7.53 Qwen-Image [75] 8.00 7.86 7.56 EMU3.5 [13] 8.11 7.70 7.59 NextFlow 8.16 7.90 7.60 NextFlow-RL 8.37 8.10 7.87

- Table 11 Comparison on our proposed EditCanvas benchmark. We report the overall score for each sub-category. Abbreviations for subject-driven sub-categories are: Abs. (abstract content), Obj. (common object), Hum. (human), and Sty. (visual style). For the aggregate metrics, we report Prompt Following (PF), Perceptual Quality (PQ), Subject Consistency (SC), and the Overall score. The best results are in bold and the second-best are underlined.

Edit Task Subject-Driven Generation

Method

Sub-Categories Aggregate Sub-Categories Aggregate Overall Global Local Style Text View PF PQ Ov. Abs. Obj. Hum. Sty. PF SC Ov.

Bagel [16] 5.72 5.50 5.08 3.79 6.89 5.66 7.27 5.28 6.91 7.92 8.43 7.86 8.16 7.50 7.69 6.49 HiDream [7] 5.30 5.25 5.80 3.12 4.58 5.21 5.86 4.81 5.53 6.30 8.04 7.32 6.94 6.38 6.52 5.67 OmniGen2 [76] 5.04 4.99 5.17 3.39 4.50 4.91 6.78 4.65 5.25 6.55 8.37 7.09 7.14 6.37 6.58 5.62 Qwen-Image [75] 7.32 6.78 7.56 5.86 8.19 7.34 7.83 6.83 7.90 8.45 8.92 8.49 8.83 8.12 8.38 7.60 Flux.1 Kontext [dev] [37] 6.19 5.35 6.47 5.32 5.92 6.08 7.64 5.59 7.48 8.06 8.51 7.89 8.32 7.79 7.96 6.77 Nano-Banana 6.69 6.49 5.30 5.62 7.18 6.83 8.05 6.34 9.22 9.13 9.21 9.05 9.51 8.91 9.17 7.76 GPT-Image-1 8.37 7.96 8.09 7.59 8.75 8.35 8.35 8.02 9.41 9.31 9.21 9.10 9.72 8.96 9.31 8.67 Emu3.5 [13] 8.06 7.61 7.71 7.17 8.92 8.34 7.80 7.70 8.86 9.16 9.18 8.70 9.50 8.68 9.04 8.37

NextFlow 7.82 7.10 7.45 6.17 8.01 7.91 7.26 7.13 8.69 8.73 8.82 8.58 9.09 8.46 8.73 7.93 NextFlow-RL 7.92 7.28 7.79 6.41 8.16 8.11 7.26 7.31 8.72 8.77 8.91 8.54 9.17 8.47 8.78 8.04

data scales: (1) 0.7M standard LLaVA-1.5 SFT data [44]; (2) 21M LLaVA-OneVision SFT data; and (3) 40M composite data, consisting of a 19M high-quality captioning mid-train followed by the 21M SFT. As shown in Table 13, despite the reduced model size, our 7B model fine-tuned on just 0.7M data delivers performance comparable to the 13B LLaVA-1.5 baseline. While scaling to 21M significantly boosts document-oriented tasks (e.g., ChartQA, OcrBench), the integration of the 19M high-quality captioning data (40M total) yields the most robust improvements across all benchmarks, demonstrating the potential of our base model.

### 7 Conclusion, Limitations and Future Directions

In this work, we introduced NextFlow, a unified sequential modeling that activates multimodal understanding and generation. By shifting from the traditional raster-scan paradigm to a next-scale prediction framework, we have demonstrated that autoregressive models can achieve inference speeds comparable to, or exceeding, state-of-the-art diffusion models without compromising on visual fidelity or reasoning capabilities. Validated on a massive corpus of 6 trillion tokens, our contributions—ranging from the dual-codebook tokenizer and scale-reweighting training objective to the prefix-tuning GRPO—establish a robust recipe for training our unified multimodal systems. NextFlow serves as a proof of concept that a single decoder-only transformer can effectively perceive, reason, and create.

Despite these advancements, limitations remain. While our dual-codebook tokenizer significantly improves semantic density, the discrete nature of vector quantization inevitably imposes an information bottleneck compared to continuous latent spaces, occasionally necessitating our optional diffusion decoder for hyper-

###### Table 12 Tokenizer reconstruction metrics on ImageNet-1K validation set [58] and in-house reconstruction benchmark.

Benchmark Method Resolution PSNR ↑ SSIM ↑ ImageNet

Original TokenFlow [54] 512 23.147 0.761 NextFlow 512 25.228 0.820

- NextFlow 1024 27.410 0.884

In-house benchmark

Original TokenFlow [54] 512 23.320 0.760 NextFlow 512 26.472 0.870

- NextFlow 1024 28.038 0.900

###### Table 13 Quantitative comparison results on multimodal understanding benchmarks. We compare different training data configurations.

Model #Training Data Model Size MMStar ChartQA OCRBench MME MME-P MMB MMMU TextVQA

LLaVA 1.5 [44] 0.7M 13B 36.7 20.4 36.2 1826.7 1531.3 67.7 36.4 61.3

0.7M 7B 44.0 18.7 35.6 1752.1 1462.1 65.3 35.1 49.5 21M 7B 42.8 50.4 44.8 1744.9 1386.3 61.7 35.8 52.7 40M 7B 53.0 57.7 55.1 1897.7 1453.1 66.7 37.1 58.9

NextFlow

realistic refinement. Furthermore, balancing the objectives of text generation and visual synthesis within a shared parameter space remains a non-trivial optimization challenge, particularly at lower parameter counts.

Looking forward, we identify three critical avenues to further scale and refine unified autoregressive modeling:

- • Data Scaling for Advanced Understanding. While our current pre-training corpus is extensive, the ratio of high-quality, dense reasoning data remains a limiting factor for complex multimodal comprehension. Future iterations will prioritize the integration of more diverse and high-quality understanding data, specifically dense image captions and complex interleaved reasoning chains. We hypothesize that scaling the density of semantic supervision will unlock emergent reasoning capabilities in the visual domain, mirroring the trajectory of text-only LLMs.
- • Model Scaling and Mixture-of-Experts (MoE). Following neural scaling laws, we anticipate that increasing model capacity will yield predictable gains in both generation fidelity and instruction following. We conducted a toy experiment and proved that transition from dense architectures to Mixture-of-Experts (MoE) frameworks significantly improves the overall generation quality.
- • Next-Generation Tokenization. The tokenizer remains the fundamental upper bound on autoregressive visual generation performance. We aim to develop improved tokenization strategies that achieve higher compression rates without sacrificing reconstruction quality. This includes exploring variablerate quantization and semantic-aware compression that can further reduce the sequence length for high-resolution images, thereby compounding the efficiency gains of our next-scale prediction paradigm.
- • Native Multimodal Chain-of-Thought and Unified RL. Our decoder-only architecture naturally extends the "Chain-of-Thought" paradigm to "Thinking with Images," enabling the model to reason via intermediate visual generation. Unlike hybrid architectures such as Transfusion [93]—which grapple with disjoint optimization objectives for discrete text and continuous visual latents—our fully autoregressive framework allows for seamless reinforcement learning across modalities. This unique structural advantage permits the direct application of standard LLM RL techniques to multimodal reasoning paths, ensuring consistent alignment and unlocking the potential for self-improving multimodal intelligence.

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

Input VQ decoder 1B diffusion decoder 12B diffusion decoder 18B diffusion decoder

- Figure 21 Qualitative comparison of VQ decoder versus diffusion decoders on image reconstruction. The VQ decoder yields satisfactory reconstruction results, although its performance on fine details is moderate. Serving as an optional enhancement, the diffusion decoder improves the reconstruction quality on local details, such as small text and small faces, as its size increases.

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

(a) (b) (c) (d)

- Figure 22 Qualitative comparison of VQ decoder versus diffusion decoders on text-to-image scenario. (a) shows the reconstruction from the baseline VQ decoder. (b)-(d) display results from our diffusion decoders with varying capacities: (b) the 1B parameter UNet-based model, (c) the 12B parameter Transformer-based model, and (d) the 18B parameter Transformer-based model. Note the progression in detail fidelity as model size increases.

Add a dog walking beside the person on the snow-covered path.

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

Add a person sitting at the edge of the wooden dock, facing the water, as if watching the sunset.

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Transfer the image into a colourful ceramic mosaic-tile style.

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

Extract the animal in the image.

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

Replace the tiny house on wheels in the image with a vintage car.

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

Add a small brown dog sitting inside the open trunk of the antique car.

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Input VQ decoder 1B diffusion decoder 12B diffusion decoder 18B diffusion decoder

###### Figure 23 Qualitative comparison of VQ decoder versus diffusion decoders on image editing scenario.

### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [2] Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28, 2015.

- [3] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. OpenAI blog, 2023.

- [4] Ollin Boer Bohan. Megalith-10m. https://huggingface.co/datasets/madebyollin/megalith-10m, June 2024. URL https://huggingface.co/datasets/madebyollin/megalith-10m. Accessed: 2024-10-07.
- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.

- [6] Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, et al. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025.

- [7] Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, et al. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025.

- [8] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.

- [9] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.

- [10] Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv:2506.18095, 2025.

- [11] Xiaokang Chen, Chengyue Wu, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, and Ping Luo. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

- [12] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

- [13] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025.

- [14] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.

- [15] Google DeepMind. Gemini 2.0. https://gemini.google.com/, 2025.
- [16] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

- [17] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

- [18] Rongyao Fang, Aldrich Yu, Chengqi Duan, Linjiang Huang, Shuai Bai, Yuxuan Cai, Kun Wang, Si Liu, Xihui Liu, and Hongsheng Li. Flux-reason-6m & prism-bench: A million-scale text-to-image reasoning dataset and comprehensive benchmark. arXiv preprint arXiv:2509.09680, 2025.

- [19] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.

- [20] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.

- [21] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

- [22] Aaron Gokaslan, A Feder Cooper, Jasmine Collins, Landan Seguin, Austin Jacobson, Mihir Patel, Jonathan Frankle, Cory Stephenson, and Volodymyr Kuleshov. Commoncanvas: An open diffusion model trained with creative-commons images. arXiv preprint arXiv:2310.16825, 2023.

- [23] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025.

- [24] Google. Gemini 2.0 flash. https://developers.googleblog.com/en/ experiment-with-gemini-20-flash-native-image-generation, 2025.
- [25] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15733–15744, 2025.

- [26] Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, Yanning Chen, and Zhipeng Wang. Liger-kernel: Efficient triton kernels for LLM training. In Championing Open-source DEvelopment in ML Workshop @ ICML25, 2025. URL https://openreview.net/ forum?id=36SjAIT42G.

- [27] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

- [28] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

- [29] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

- [30] Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990, 2024.

- [31] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025.

- [32] Liming Jiang, Qing Yan, Yumin Jia, Zichuan Liu, Hao Kang, and Xin Lu. Infiniteyou: Flexible photo recrafting while preserving your identity. arXiv preprint arXiv:2503.16418, 2025.

- [33] Siyu Jiao, Gengwei Zhang, Yinlong Qian, Jiancheng Huang, Yao Zhao, Humphrey Shi, Lin Ma, Yunchao Wei, and Zequn Jie. Flexvar: Flexible visual autoregressive modeling without residual prediction. arXiv preprint arXiv:2502.20313, 2025.

- [34] Siqi Kou, Jiachun Jin, Zhihong Liu, Chang Liu, Ye Ma, Jian Jia, Quan Chen, Peng Jiang, and Zhijie Deng. Orthus: Autoregressive interleaved image-text generation with modality-specific heads. arXiv preprint arXiv:2412.00127, 2024.

- [35] Black Forest Labs. Flux, 2024. URL https://github.com/black-forest-labs/flux.
- [36] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [37] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith.

- Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. URL https: //arxiv.org/abs/2506.15742.
- [38] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

- [39] Han Li, Xinyu Peng, Yaoming Wang, Zelin Peng, Xin Chen, Rongxiang Weng, Jingang Wang, Xunliang Cai, Wenrui Dai, and Hongkai Xiong. Onecat: Decoder-only auto-regressive model for unified understanding and generation. arXiv preprint arXiv:2509.03498, 2025.

- [40] Qingyun Li, Zhe Chen, Weiyun Wang, Wenhai Wang, Shenglong Ye, Zhenjiang Jin, Guanzhou Chen, Yinan He, Zhangwei Gao, Erfei Cui, et al. Omnicorpus: A unified multimodal corpus of 10 billion-level images interleaved with text. arXiv preprint arXiv:2406.08418, 2024.

- [41] Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.

- [42] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.

- [43] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld-v1: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.

- [44] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024.

- [45] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

- [46] Yiheng Liu, Liao Qu, Huichao Zhang, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Xian Li, Shuai Wang, Daniel K Du, et al. Detailflow: 1d coarse-to-fine autoregressive image generation via next-detail prediction. arXiv preprint arXiv:2505.21473, 2025.

- [47] Xiaoxiao Ma, Haibo Qiu, Guohui Zhang, Zhixiong Zeng, Siqi Yang, Lin Ma, and Feng Zhao. Stage: Stable and generalizable grpo for autoregressive image generation. arXiv preprint arXiv:2509.25027, 2025.

- [48] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng Ning, Chaoran Feng, Bin Zhu, and Li Yuan. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

- [49] OpenAI. Gpt-4o. https://openai.com/index/introducing-4o-image-generation, 2025.
- [50] OpenAI. Gpt-image-1, 2025. URL https://openai.com/index/introducing-4o-image-generation/.
- [51] Yulin Pan, Xiangteng He, Chaojie Mao, Zhen Han, Zeyinzi Jiang, Jingfeng Zhang, and Yu Liu. Ice-bench: A unified and comprehensive benchmark for image creating and editing. arXiv preprint arXiv:2503.14482, 2025.

- [52] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

- [53] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024.

- [54] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2545–2555, 2025.

- [55] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.

- [56] Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Beyond next-token: Next-x prediction for autoregressive visual generation. arXiv preprint arXiv:2502.20388, 2025.

- [57] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

- [58] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211–252, 2015.

- [59] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.

- [60] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [61] Ying Shen, Yizhe Zhang, Shuangfei Zhai, Lifu Huang, Joshua M Susskind, and Jiatao Gu. Many-to-many image generation with auto-regressive diffusion models. arXiv preprint arXiv:2404.03109, 2024.

- [62] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879, 2024.

- [63] Stability-AI. Stable diffusion 3.5, 2024. URL https://github.com/Stability-AI/sd3.5.
- [64] Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. Nemotron-cc: Transforming common crawl into a refined long-horizon pretraining dataset. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2459–2475, 2025.

- [65] Peter Sushko, Ayana Bharadwaj, Zhi Yang Lim, Vasily Ilin, Ben Caffee, Dongping Chen, Mohammadreza Salehi, Cheng-Yu Hsieh, and Ranjay Krishna. Realedit: Reddit edits as a large-scale empirical dataset for image transformations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13403–13413, 2025.

- [66] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

- [67] NextStep Team, Chunrui Han, Guopeng Li, Jingwei Wu, Quan Sun, Yan Cai, Yuang Peng, Zheng Ge, Deyu Zhou, Haomiao Tang, et al. Nextstep-1: Toward autoregressive image generation with continuous tokens at scale. arXiv preprint arXiv:2508.10711, 2025.

- [68] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, pages 402–419. Springer, 2020.

- [69] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024.

- [70] Chengzhuo Tong, Ziyu Guo, Renrui Zhang, Wenyu Shan, Xinyu Wei, Zhenghao Xing, Hongsheng Li, and Pheng-Ann Heng. Delving into rl for image generation with cot: A study on dpo vs. grpo. arXiv preprint arXiv:2505.17017, 2025.

- [71] Junke Wang, Zhi Tian, Xun Wang, Xinyu Zhang, Weilin Huang, Zuxuan Wu, and Yu-Gang Jiang. Simplear: Pushing the frontier of autoregressive visual generation through pretraining, sft, and rl. arXiv preprint arXiv:2504.11455, 2025.

- [72] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8428–8437, 2025.

- [73] Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi Pont-Tuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J Fleet, Radu Soricut, et al. Imagen editor and editbench: Advancing and evaluating text-guided image inpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18359–18369, 2023.

- [74] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arxiv:2409.18869, 2024.

- [75] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

- [76] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.

- [77] Junfeng Wu, Yi Jiang, Chuofan Ma, Yuliang Liu, Hengshuang Zhao, Zehuan Yuan, Song Bai, and Xiang Bai. Liquid: Language models are scalable and unified multi-modal generators. arXiv preprint arXiv:2412.04332, 2024.

- [78] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160, 2025.

- [79] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13294–13304, 2025.

- [80] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arxiv:2408.12528, 2024.

- [81] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.

- [82] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025.

- [83] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.

- [84] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26125–26135, 2025.

- [85] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.

- [86] Shihao Yuan, Yahui Liu, Yang Yue, Jingyuan Zhang, Wangmeng Zuo, Qi Wang, Fuzheng Zhang, and Guorui Zhou. Ar-grpo: Training autoregressive image generation models via reinforcement learning. arXiv preprint arXiv:2508.06924, 2025.

- [87] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

- [88] Guohui Zhang, Hu Yu, Xiaoxiao Ma, JingHao Zhang, Yaning Pan, Mingde Yao, Jie Xiao, Linjiang Huang, and Feng Zhao. Group critical-token policy optimization for autoregressive image generation. arXiv preprint arXiv:2509.22485, 2025.

- [89] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.

- [90] Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, et al. Hive: Harnessing human feedback for instructional visual editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9026–9036, 2024.

- [91] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer. arXiv preprint arXiv:2504.20690, 2025.

- [92] Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024.

- [93] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arxiv:2408.11039, 2024.

- [94] Fan Zhou, Zengzhi Wang, Nikhil Ranjan, Zhoujun Cheng, Liping Tang, Guowei He, Zhengzhong Liu, and Eric P Xing. Megamath: Pushing the limits of open math corpora. arXiv preprint arXiv:2504.02807, 2025.

## Appendix

### A Additional Implementation Details and Analysis

#### A.1 Predefined Scale Schedules

We design a comprehensive set of scale schedules to enable flexible image generation across diverse aspect ratios. As detailed in Table 14, for each target aspect ratio r, we define a progressive scale schedule Sr = (hr1,w1r),(hr2,w2r),...,(hrK,wKr ), where K = 18 represents the total number of scale progression steps.

In this design, each tuple (hrk,wkr) maintains an aspect ratio approximately equal to r, with particular emphasis on preserving this ratio at higher scales where visual fidelity is critical. For any given scale index k, we ensure

that the spatial area hrk × wkr remains approximately constant across different aspect ratios r. This design choice guarantees uniform training sequence lengths and balanced computational requirements.

For 256-scale images, we utilize the first 12 scales from the corresponding schedule as the scale setting. For 512-scale images, we utilize the first 16 scales as the scale setting. During inference, our method generates photorealistic images spanning an extensive range of aspect ratios from 1:4 to 4:1, following these predefined scale schedules. This approach enables consistent quality across diverse output dimensions while maintaining computational efficiency.

- Table 14 Predefined scale schedules for progressive image generation across 40 aspect ratios. Each row represents a complete 18-step scale progression from initial patch (hr1, w1r) to final resolution (hr18, w18r ).

Aspect Ratio Resolution Scale Schedule

0.250 (1:4) 512×2048 (1,1) (1,4) (2,6) (2,8) (2,10) (3,12) (4,14) (4,16) (5,20) (6,24) (7,28) (8,32) (10,40) (12,48) (14,56) (16,64) (24,96) (32,128) 0.260 (13:50) 544×2080 (1,1) (1,4) (2,6) (2,8) (2,10) (3,12) (4,14) (4,16) (5,19) (6,23) (7,27) (8,31) (10,39) (12,46) (14,54) (16,62) (24,93) (32,124) 0.267 (4:15) 576×2160 (1,1) (1,4) (2,6) (2,8) (2,9) (3,11) (4,13) (4,15) (5,19) (6,22) (7,26) (8,30) (10,38) (12,45) (14,52) (16,60) (24,90) (32,120) 0.276 (10:36) 576×2088 (1,1) (1,4) (2,5) (2,7) (2,9) (3,11) (4,13) (4,14) (5,18) (6,22) (7,25) (8,29) (10,36) (12,44) (14,51) (16,58) (24,87) (32,116) 0.295 (11:37) 592×1984 (1,1) (1,4) (2,5) (2,7) (3,9) (3,10) (4,12) (4,14) (6,18) (7,21) (8,24) (9,28) (11,35) (14,42) (16,49) (18,56) (27,84) (36,112) 0.311 (9:29) 624×2001 (1,1) (1,3) (2,5) (2,6) (3,8) (3,10) (4,11) (4,13) (6,16) (7,20) (8,23) (9,26) (11,32) (14,39) (16,46) (18,52) (27,78) (36,104) 0.333 (1:3) 576×1728 (1,1) (1,3) (2,5) (2,7) (3,8) (3,10) (4,12) (4,14) (6,17) (7,20) (8,24) (9,27) (11,34) (14,40) (16,47) (18,54) (27,81) (36,108) 0.345 (10:29) 640×1856 (1,1) (1,3) (2,5) (2,6) (3,8) (4,9) (4,11) (5,12) (6,16) (8,19) (9,22) (10,25) (12,31) (15,38) (18,44) (20,50) (30,75) (40,100) 0.417 (5:12) 800×1920 (1,1) (1,3) (2,4) (2,6) (3,8) (4,9) (4,10) (5,12) (6,15) (8,18) (9,21) (10,24) (12,30) (15,36) (18,42) (20,48) (30,72) (40,96) 0.478 (11:23) 880×1840 (1,1) (1,3) (2,4) (3,6) (3,7) (4,9) (5,10) (6,12) (7,14) (8,17) (10,20) (11,23) (14,29) (16,34) (19,40) (22,46) (33,69) (44,92)

- 0.500 (1:2) 880×1760 (1,1) (1,3) (2,4) (3,6) (3,7) (4,8) (5,10) (6,11) (7,14) (8,16) (10,19) (11,22) (14,28) (16,33) (19,38) (22,44) (33,66) (44,88)

- 0.524 (11:21) 896×1709 (1,1) (1,3) (2,4) (3,5) (3,7) (4,8) (5,9) (6,10) (7,13) (8,16) (10,18) (11,21) (14,26) (16,32) (19,37) (22,42) (33,63) (44,84)

- 0.571 (4:7) 896×1568 (1,1) (2,3) (2,4) (3,5) (4,7) (4,8) (5,9) (6,10) (8,13) (9,16) (10,18) (12,21) (15,26) (18,32) (21,37) (24,42) (36,63) (48,84)

- 0.600 (3:5) 960×1600 (1,1) (2,2) (2,4) (3,5) (4,6) (4,8) (5,9) (6,10) (8,12) (9,15) (10,18) (12,20) (15,25) (18,30) (21,35) (24,40) (36,60) (48,80)

- 0.685 (13:19) 816×1192 (1,1) (2,2) (2,4) (3,5) (4,6) (5,7) (6,8) (6,10) (8,12) (10,14) (11,17) (13,19) (16,24) (20,28) (23,33) (26,38) (39,57) (52,76)

- 0.722 (13:18) 864×1196 (1,1) (2,2) (2,3) (3,4) (4,6) (5,7) (6,8) (6,9) (8,11) (10,14) (11,16) (13,18) (16,22) (20,27) (23,32) (26,36) (39,54) (52,72)

- 0.781 (25:32) 992×1270 (1,1) (2,2) (3,3) (4,4) (4,6) (5,7) (6,8) (7,9) (9,11) (10,14) (12,16) (14,18) (18,22) (21,27) (24,32) (28,36) (42,54) (56,72)

- 0.824 (14:17) 1008×1224 (1,1) (2,2) (3,3) (4,4) (4,5) (5,6) (6,7) (7,8) (9,11) (10,13) (12,15) (14,17) (18,21) (21,26) (24,30) (28,34) (42,51) (56,68)

- 0.882 (15:17) 960×1088 (1,1) (2,2) (3,3) (4,4) (5,5) (6,6) (7,7) (8,8) (9,11) (11,13) (13,15) (15,17) (19,21) (22,26) (26,30) (30,34) (45,51) (60,68)

- 0.937 (15:16) 960×1024 (1,1) (2,2) (3,3) (4,4) (5,5) (6,6) (7,7) (8,8) (9,10) (11,12) (13,14) (15,16) (19,20) (22,24) (26,28) (30,32) (45,48) (60,64)
- 1.000 (1:1) 1024×1024 (1,1) (2,2) (3,3) (4,4) (5,5) (6,6) (7,7) (8,8) (10,10) (12,12) (14,14) (16,16) (20,20) (24,24) (28,28) (32,32) (48,48) (64,64)

- 1.067 (16:15) 1024×960 (1,1) (2,2) (3,3) (4,4) (5,5) (6,6) (7,7) (8,8) (10,9) (12,11) (14,13) (16,15) (20,19) (24,22) (28,26) (32,30) (48,45) (64,60)

- 1.133 (17:15) 1088×960 (1,1) (2,2) (3,3) (4,4) (5,5) (6,6) (7,7) (8,8) (11,9) (13,11) (15,13) (17,15) (21,19) (26,22) (30,26) (34,30) (51,45) (68,60)

- 1.214 (17:14) 1224×1008 (1,1) (2,2) (3,3) (4,4) (5,4) (6,5) (7,6) (8,7) (11,9) (13,10) (15,12) (17,14) (21,18) (26,21) (30,24) (34,28) (51,42) (68,56)

- 1.286 (9:7) 1270×992 (1,1) (2,2) (3,3) (4,4) (6,4) (7,5) (8,6) (9,7) (11,9) (14,10) (16,12) (18,14) (22,18) (27,21) (32,24) (36,28) (54,42) (72,56)

- 1.385 (18:13) 1196×864 (1,1) (2,2) (3,2) (4,3) (6,4) (7,5) (8,6) (9,6) (11,8) (14,10) (16,11) (18,13) (22,16) (27,20) (32,23) (36,26) (54,39) (72,52)

1.462 (19:13) 1192×816 (1,1) (2,2) (4,2) (5,3) (6,4) (7,5) (8,6) (10,6) (12,8) (14,10) (17,11) (19,13) (24,16) (28,20) (33,23) (38,26) (57,39) (76,52) 1.667 (5:3) 1600×960 (1,1) (2,2) (4,2) (5,3) (6,4) (8,4) (9,5) (10,6) (12,8) (15,9) (18,10) (20,12) (25,15) (30,18) (35,21) (40,24) (60,36) (80,48) 1.750 (7:4) 1568×896 (1,1) (3,2) (4,2) (5,3) (7,4) (8,4) (9,5) (10,6) (13,8) (16,9) (18,10) (21,12) (26,15) (32,18) (37,21) (42,24) (63,36) (84,48) 1.909 (21:11) 1709×896 (1,1) (3,1) (4,2) (5,3) (7,3) (8,4) (9,5) (10,6) (13,7) (16,8) (18,10) (21,11) (26,14) (32,16) (37,19) (42,22) (63,33) (84,44) 2.000 (2:1) 1760×880 (1,1) (3,1) (4,2) (6,3) (7,3) (8,4) (10,5) (11,6) (14,7) (16,8) (19,10) (22,11) (28,14) (33,16) (38,19) (44,22) (66,33) (88,44) 2.091 (23:11) 1840×880 (1,1) (3,1) (4,2) (6,3) (7,3) (9,4) (10,5) (12,6) (14,7) (17,8) (20,10) (23,11) (29,14) (34,16) (40,19) (46,22) (69,33) (92,44) 2.400 (12:5) 1920×800 (1,1) (3,1) (4,2) (6,2) (8,3) (9,4) (10,4) (12,5) (15,6) (18,8) (21,9) (24,10) (30,12) (36,15) (42,18) (48,20) (72,30) (96,40) 2.500 (5:2) 1856×640 (1,1) (3,1) (5,2) (6,2) (8,3) (9,4) (11,4) (12,5) (16,6) (19,8) (22,9) (25,10) (31,12) (38,15) (44,18) (50,20) (75,30) (100,40)

- 2.889 (26:9) 2001×624 (1,1) (3,1) (5,2) (6,2) (8,3) (10,3) (11,4) (13,4) (16,6) (20,7) (23,8) (26,9) (32,11) (39,14) (46,16) (52,18) (78,27) (104,36)
- 3.000 (3:1) 1728×576 (1,1) (3,1) (5,2) (7,2) (8,3) (10,3) (12,4) (14,4) (17,6) (20,7) (24,8) (27,9) (34,11) (40,14) (47,16) (54,18) (81,27) (108,36)

- 3.111 (28:9) 1984×592 (1,1) (4,1) (5,2) (7,2) (9,3) (10,3) (12,4) (14,4) (18,6) (21,7) (24,8) (28,9) (35,11) (42,14) (49,16) (56,18) (84,27) (112,36)

- 3.625 (29:8) 2088×576 (1,1) (4,1) (5,2) (7,2) (9,2) (11,3) (13,4) (14,4) (18,5) (22,6) (25,7) (29,8) (36,10) (44,12) (51,14) (58,16) (87,24) (116,32)

- 3.750 (15:4) 2160×576 (1,1) (4,1) (6,2) (8,2) (9,2) (11,3) (13,4) (15,4) (19,5) (22,6) (26,7) (30,8) (38,10) (45,12) (52,14) (60,16) (90,24) (120,32)

- 3.875 (31:8) 2080×544 (1,1) (4,1) (6,2) (8,2) (10,2) (12,3) (14,4) (16,4) (19,5) (23,6) (27,7) (31,8) (39,10) (46,12) (54,14) (62,16) (93,24) (124,32)
- 4.000 (4:1) 2048×512 (1,1) (4,1) (6,2) (8,2) (10,2) (12,3) (14,4) (16,4) (20,5) (24,6) (28,7) (32,8) (40,10) (48,12) (56,14) (64,16) (96,24) (128,32)

#### A.2 Inference Efficiency Analysis

- 1e+06

- 1e+07

- 1e+08

- 1e+09

- 1e+10

6.5x

MMDiT (Diffusion) 9.2x

MMDiT (Diffusion)

MMDiT (Diffusion) 7.0x

TokenFlow++ (Uniform Steps)

TokenFlow++ (Uniform Steps)

TokenFlow++ (Uniform Steps)

TokenFlow++ (Realistic Steps)

TokenFlow++ (Realistic Steps)

TokenFlow++ (Realistic Steps)

7.2x

1e+05

6.0x

- 1e+11

- 1e+12

- 1e+13

4.7x

TotalEstimatedFLOPs

QuadraticComplexity

(FFN/Projections)

LinearComplexity

(Attention)

4.3x

3.7x

2.0x

1e4

3.1x

2.9x

1.4x

1000

256 512 1024 2048

256 512 1024 2048

256 512 1024 2048

Image Resolution

Image Resolution

Image Resolution

Figure 24 Inference efficiency comparison between NextFlow and MMDiT across different image resolutions.

Both models use the same number of sampling steps for fair comparison. We evaluate two scale scheduling strategies for NextFlow: (1) Uniform Steps, where the scale resolution increases linearly with each step; (2) Realistic Steps, which reflects our actual training configuration that allocates more steps to lower scales and fewer steps to higher resolutions. The annotations show the FLOPs ratio of MMDiT to NextFlow. As resolution increases, the efficiency advantage of NextFlow becomes more pronounced, achieving up to 6× reduction in total FLOPs.

In this section, we provide a comprehensive theoretical analysis comparing the computational costs of NextFlow, which adopts the next-scale prediction paradigm from VAR [69], against the Multi-Modal Diffusion Transformer (MMDiT) [17] baseline. We focus primarily on the inference phase, where efficiency is critical for real-world deployment.

Inference Efficiency. To evaluate the inference cost fairly, we compare the Floating Point Operations (FLOPs) required to generate a 1024 × 1024 resolution image (approximately 64 × 64 latent tokens). Both models utilize a similar hyperparameter setup: a hidden size h (tested at 2048 and 4096) and a total of T = 18 sampling steps.

For a standard Transformer layer, the FLOPs for the Attention mechanism and the Feed-Forward Network (FFN) are defined as [14, 69]:

FLOPsattn ≈ 4sqh · (2h) + 4sqskvh = 8sqh2 + 4sqskvh (6) FLOPsMLP ≈ 2 · 2sqh2 = 4sqh2 (7) where sq is the query sequence length and skv is the key/value sequence length. The total FLOPs per layer is: FLOPslayer = 12sqh2 + 4sqskvh (8)

MMDiT Cost Analysis. MMDiT [17] operates on a fixed sequence length s = 64 × 64 = 4096 across all T = 18 denoising steps. Since sq = skv = s, the total inference cost is:

FLOPsMMDiT = T · L · (12sh2 + 4s2h) ≈ 884,736 · h2 + 1,207,959,552 · h (9)

NextFlow Cost Analysis. In contrast, NextFlow employs the next-scale prediction paradigm [69]. The sequence length sq varies dynamically at each step (e.g., 1,4,9,...,4096), and thanks to the autoregressive nature, we utilize KV-caching. This means sq is the number of new tokens generated in the current step, while skv is the cumulative sum of all previous tokens. The total inference cost is:

FLOPsNextFlow =

T

L · (12s(qt)h2 + 4s(qt)s(kvt)h) ≈ 122,208 · h2 + 256,215,912 · h (10)

t=1

Comparative Results. By substituting typical hidden sizes (e.g., h = 4096), the ratio of computational cost is:

FLOPsMMDiT FLOPsNextFlow ≈

6.18 × 1012 1.04 × 1012 ≈ 5.96 (11)

This analysis demonstrates that MMDiT requires approximately 6× more FLOPs than NextFlow to generate an image of the same resolution. The efficiency gain stems from two key factors: (1) FFN and Linear Projections: These costs depend linearly on the current step’s token count (sq). Since the next-scale prediction paradigm generates fewer tokens in early steps, the average computational load is significantly lower. In our implementation, MMDiT consumes ≈ 7.2× more FLOPs in these components. (2) Attention Mechanism: While memory access for Keys/Values grows cumulatively, the computational cost for Queries is minimal in early steps. Overall, MMDiT’s attention computation is ≈ 4.7× that of NextFlow.

In summary, for deployment, NextFlow offers a definitive advantage, reducing inference FLOPs by a factor of 6 compared to state-of-the-art diffusion transformers [17, 35].

- Figure 24 visualizes this efficiency comparison across different image resolutions. For fair comparison, both MMDiT and NextFlow use the same number of sampling steps at each resolution. We evaluate two scale scheduling strategies: (1) Uniform Steps, where scale resolution increases linearly from 1 × 1 to the target resolution; (2) Realistic Steps, which reflects our actual training configuration—allocating more steps to lower scales for better structural modeling and fewer steps to higher resolutions where fine details are refined. The realistic schedule which we use achieves slightly better efficiency while maintaining generation quality.

### B EditCanvas benchmark

To comprehensively evaluate the capabilities of modern image editing models, we introduce EditCanvas, a novel and meticulously structured benchmark. EditCanvas is designed to address the limitations of existing benchmarks by offering a more systematic and fine-grained assessment framework. A key innovation of EditCanvas is its hierarchical, three-level labeling system, which categorizes editing tasks with high granularity. At the highest level, we distinguish between two fundamental paradigms: Traditional Editing and Subject-Driven Generation.

- • Traditional Editing focuses on direct modifications to a reference image, akin to benchmarks like GEditBench. This category is further divided into five sub-categories: Text Editing, Global Editing, Local Editing, Viewpoint Editing, and Style Editing.
- • Subject-Driven Generation, similar in spirit to OmniContext, involves extracting a specific subject from a reference image to generate a new scene. This is broken down into four sub-categories: Style, Character, Object, and Abstract Concept.

This structure extends to a third level with 56 fine-grained tasks (40 for Traditional Editing and 16 for Subject-Driven Generation), such as object addition, removal, replacement, and repositioning. This detailed taxonomy allows for a nuanced analysis of a model’s performance across a wide spectrum of editing scenarios, providing deeper insights than previously possible. The EditCanvas dataset comprises 5,221 high-quality samples. We began by sourcing high-resolution images (over 1K) from open datasets like LAION and COYO, filtering out those with plain white backgrounds. We then employed a state-of-the-art Vision-Language Model (VLM) to automatically generate corresponding editing instructions for each image. To ensure the benchmark’s reliability and quality, these samples underwent a rigorous two-round manual filtering process, where we assessed factors such as image clarity, content appropriateness, and the suitability of the generated instructions. For evaluation, EditCanvas adopts a hybrid metric approach inspired by its predecessors. For Traditional Editing tasks, we follow the methodology of GEdit-Bench, using GPT-4.1 to assess Instruction Following and the presence of Artifacts. For Subject-Driven Generation tasks, we align with OmniContext’s protocol, evaluating Instruction Following and Content Consistency using GPT-4.1.

Compared to existing benchmarks such as GEdit-Bench and OmniContext, EditCanvas presents several key advantages:

Benchmarks Size Real Image Human Filtering Sub-tasks

EditBench [73] 240 ✓ ✗ 1 EmuEdit [62] 3,055 ✓ ✗ 7 HIVE [90] 1,000 ✓ ✓ 1 HQ-Eidt [30] 1,640 ✗ ✗ 7 MagicBrush [89] 1,053 ✓ ✓ 7 AnyEdit [61] 1,250 ✓ ✗ 25 ICE-Bench [51] 6,538 ✓ ✓ 31 Gedit-Bench [45] 606 ✓ ✓ 11

EditCanvas (Ours) 5,221 ✓ ✓ 56

- Table 15 Key Attributes of Open-source Edit Benchmarks. The reliance of existing open-source benchmarks on synthetic user inputs and minimal human involvement highlights the necessity of our proposed EditCanvas.

1.6%object_remove 1.6%object_outfit

1.6%object_color

style_apply

face_id_keep

character_place

object_pose1.6% outpainting1.6%

3.6%

object_texture

1.6% object_expression

human_id_place

1.9%

2.1%

1.5% object_material

4.3%

1.5% object_replace

design_apply

logo_extract

visual_style

photo_extract

material_apply

0.4%

1.4% object_size

0.5%

1.1%

1.4% human_makeup

3.6%

1.1%

logo_place

human 8.4%

1.2% human_age

1.6%

photo_place

local_edit 26.3%

1.1% object_add

abstract_content

1.6%

1.1% human_gender

pattern_apply

1.1% human_hair

2.6%

9.0%

SubjectDrivenEdit

1.1% object_position

object_extract 0.5%

object_state 1.1%

1.1% object_shape

32.2%

1.1% object_rotation

object_view 1.8%

1.1%

Edit Canvas

inpainting 1.0%

object_quantity 2.1%

common_object 11.3%

object_motion 1.0%

Edit 67.8%

rotation 3.3%

5.8%object_place

2.5%style_edit

text_edit20.6%

style 3.3%

4.9%view_edit

1.0%unreal_to_real

1.5%real_to_unreal

replace 3.2%

global_edit

13.5%

1.5%camera_view

camera_zoom_out

1.6%

size 3.1%

camera_zoom_in

1.8%

shape2.3%

lighting_direction

1.6%

lighting_add

1.7%

add 2.2%

lighting_remove

1.8%

remove2.0%

enhance_image

2.0%

position1.3%

background

modality2.3%

2.1%

adjust_color

2.1%

- Figure 25 Distribution of EditCanvas Tasks. The chart illustrates the hierarchical structure of the dataset, dividing tasks into Traditional Editing (blue, 67.8%) and Subject-Driven Editing (orange, 32.2%). The outer layers represent the sub-categories and specific fine-grained editing tasks.

- 1. Unified and Comprehensive Framework: While GEdit-Bench focuses on genuine user instructions for direct editing and OmniContext specializes in in-context subject generation, EditCanvas is the first benchmark to systematically integrate both paradigms into a single, unified framework. This allows for a holistic evaluation of a model’s diverse editing capabilities, from minor tweaks to complex generative tasks.
- 2. Granular, Hierarchical Taxonomy: EditCanvas’s three-level classification system is significantly more detailed than the taxonomies of previous benchmarks. While GEdit-Bench offers 11 categories and OmniContext defines 8 subtasks, EditCanvas provides 57 distinct third-level tasks. This granularity enables precise identification of a model’s specific strengths and weaknesses, for instance, distinguishing its performance in "object addition" versus "object repositioning."
- 3. Large-Scale, High-Quality Dataset: With over 5,000 carefully curated samples, EditCanvas offers a larger and more diverse dataset compared to GEdit-Bench’s 606 examples. The semi-automated generation process, combining a powerful VLM with multi-round human verification, ensures both scale and quality, creating a robust testbed for modern models.

By providing a structured, comprehensive, and large-scale benchmark, EditCanvas sets a new standard for evaluating image editing models, enabling a more thorough and insightful understanding of their real-world performance.

