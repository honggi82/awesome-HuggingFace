# arXiv:2507.21033v1[cs.CV]28Jul2025

## GPT-IMAGE-EDIT-1.5M A Million-Scale, GPT-Generated Image Dataset

Yuhan Wang1 Siwei Yang1 Bingchen Zhao2 Letian Zhang1 Qing Liu3 Yuyin Zhou1 Cihang Xie1

1University of California, Santa Cruz 2The University of Edinburgh 3Adobe

[Figure 1]

Project Page: https://ucsc-vlaa.github.io/GPT-Image-Edit Code: https://github.com/wyhlovecpp/GPT-Image-Edit Dataset: https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M

[Figure 2]

[Figure 3]

### Abstract

Recent advancements in large multimodal models like GPT-4o have set a new standard for high-fidelity, instruction-guided image editing. However, the proprietary nature of these models and their training data creates a significant barrier for open-source research. To bridge this gap, we introduce GPT-IMAGE-EDIT1.5M, a publicly available, large-scale image-editing corpus containing more than 1.5 million high-quality triplets {instruction, source image, edited image}. We systematically construct this dataset by leveraging the versatile capabilities of GPT-4o to unify and refine three popular image-editing datasets: OmniEdit, HQEdit, and UltraEdit. Specifically, our methodology involves 1) regenerating output images to enhance visual quality and instruction alignment, and 2) selectively rewriting prompts to improve semantic clarity. To validate the efficacy of our dataset, we fine-tune advanced open-source models on GPT-IMAGE-EDIT-1.5M. The empirical results are exciting — e.g., the fine-tuned FluxKontext achieves highly competitive performance across a comprehensive suite of benchmarks, including 7.24@GEdit-EN, 3.80@ImgEdit-Full, and 8.78@Complex-Edit, showing stronger instruction following and higher perceptual quality while maintaining identity. These scores markedly exceed all previously published open-source methods and substantially narrow the gap to leading proprietary models. We hope the full release of GPT-IMAGE-EDIT-1.5M can help to catalyze further open research in instruction-guided image editing.

### 1 Introduction

Instruction-guided image editing has rapidly emerged as a key research direction in generative AI, with a series of diffusion- and inversion-based methods demonstrating that natural-language instructions can drive high-quality, controllable edits (e.g., InstructPix2Pix (Brooks et al., 2023), Promptto-Prompt (Hertz et al., 2022), SDEdit (Meng et al., 2021), Imagic (Kawar et al., 2023)). Among all models, proprietary systems, exemplified by GPT-4o (Hurst et al., 2024), have established a particularly high benchmark for performance, executing edits with impressive semantic understanding and photorealism. However, the closed-source nature of these leading models (Shi et al., 2024; Wang et al., 2025) and the datasets they are trained on limits the progress of the broader research community, creating a disparity between proprietary capabilities and open-source alternatives.

A primary bottleneck for developing powerful open-source image editing models is the absence of sufficiently large, diverse, and high-quality training data. Existing public resources (e.g., OmniEdit (Wei et al., 2025), HQ-Edit (Hui et al., 2025), UltraEdit (Zhao et al., 2024)) sometimes contain noisy or simplistic instructions, misaligned input–output pairs, or limited editing diversity between the user’s intent and the resulting edited image. This data gap makes it challenging to

[Figure 4]

|[Figure 5]|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

[Figure 9]

###### GPT-IMAGE-EDIT-1.5M

[Figure 10]

[Figure 11]

Transform the bed into a floating cloud in the sky with a rainbow arching above it

7.5

###### 7.24 GEdit-EN-full

Make the cat turn its head to the side and widen its eyes

6.97

7.0

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

6.60

6.42

6.5

6.26

6.0

5.01

5.0

[Figure 16]

[Figure 17]

Replace the guitar's texture with a darker wood tone and change the shirt's color to navy blue. Modify the background to feature a cozy interior with books and plants

Apply a comic book-style filter to the image, adding bold outlines, vibrant colors, and texture resembling halftone dots

4.0

- Figure 1: An overview of the GPT-IMAGE-EDIT-1.5M dataset. The figure presents qualitative examples from our dataset, showcasing its ability to handle complex and diverse instruction-guided edits. The bar chart on the right demonstrates the effectiveness of our data; a model fine-tuned on GPT-IMAGE-EDIT-1.5M achieves a new state-of-the-art score of 7.24 on the GEdit-EN-full benchmark, outperforming existing open-source methods.

train models that can rival the nuanced understanding and execution quality of their closed-source counterparts. (Wei et al., 2025; Zhao et al., 2024; Lin et al., 2025)

To address this challenge, we introduce GPT-IMAGE-EDIT-1.5M, a large-scale dataset of over 1.5 million samples designed to propel open-source image editing forward. An overview of GPTIMAGE-EDIT-1.5M is presented in Fig 1. Our core strategy is to leverage a frontier model, GPT-4o, as a data generation and refinement tool. We unify and enhance three prominent datasets — OmniEdit (Wei et al., 2025), HQ-Edit (Hui et al., 2025), and UltraEdit (Zhao et al., 2024) — through a systematic, multi-faceted approach. Specifically, our initial step involved regenerating only the output images for existing pairs to improve visual quality and alignment. This alone provided a significant boost; for example, on the OmniEdit dataset (Wei et al., 2025), fine-tuning Flux 1.0 dev (Labs, 2024) on this regenerated data (‘omniedit100k-gpt’) improved the imgedit score from 2.94 to

- 3.24 compared to the baseline.

Nonetheless, we observed that GPT-4o would sometimes interpret instructions creatively, causing a subtle semantic drift between the original instruction and the new output. To correct this instructionimage mismatch, we further developed two more sophisticated data refinement techniques. First, we introduced instruction regeneration, where we prompted GPT-4o to write a new, more accurate instruction describing the transformation from the original input to the newly generated output. This ‘gpt-rewrite’ variant further improved performance, raising the imgedit (Ye et al., 2025) score on OmniEdit to 3.40. Second, inspired by the HQ-Edit (Hui et al., 2025) methodology, we implemented full pair regeneration, where both the input and output images are synthesized by the model. This ‘pair-regen’ strategy, applied to the HQ-Edit subset (Hui et al., 2025), also demonstrated consistent gains, lifting the GEdit-EN (Liu et al., 2025) score from 5.67 to 5.73 on the Flux model. This iterative refinement process—from output regeneration to instruction rewriting and full pair synthesis—was crucial for creating a dataset with high-fidelity alignment between text and images.

Lastly, we systematically demonstrate the profound impact of our GPT-IMAGE-EDIT-1.5M dataset by fine-tuning a powerful open-source model, FluxKontext dev (Labs et al., 2025). By additionally enhancing its architecture with Qwen-VL-7b (Bai et al., 2025) embeddings for improved conditionimage alignment, our model achieves exciting performance on a wide array of benchmarks. As shown in our experiments, training on GPT-IMAGE-EDIT-1.5M yields substantial performance gains over models trained on the original datasets, with our final model achieving an average score of 7.236 on GEdit-EN (Liu et al., 2025) and 3.80 on ImgEdit-Full (Ye et al., 2025), placing it among the top-performing open-source models. Our ablation studies confirm that data regenerated by GPT-4o is the key driver of this improvement. Interestingly, our findings also suggest that simply increasing instruction complexity without ensuring identity preservation can be counterproductive, highlighting the critical importance of data quality in the training process. By making this GPTIMAGE-EDIT-1.5M dataset and the trained model publicly available, we hope to provide a valuable resource for the community to keep pushing the frontier of open research in image editing.

### 2 Related Works

Instruction-Guided Image Editing. The paradigm of instruction-guided image editing was largely established by InstructPix2Pix (Brooks et al., 2023), which framed the task as a supervised learning problem. This seminal work introduced a scalable, two-stage pipeline: first, generating a large synthetic dataset of {instruction, source image, edited image} triplets by prompting a large language model (GPT-3) to create editing instructions (Brown et al., 2020), and then using a textto-image model (Stable Diffusion) with Prompt-to-Prompt controls to generate the corresponding image pairs (Hertz et al., 2022). While foundational, the performance of InstructPix2Pix was inherently capped by the generative capabilities of its underlying models — latent-diffusion-based architectures trained with CLIP — which struggled with photorealism and complex semantic understanding (Rombach et al., 2022). This limitation spurred subsequent research to focus on two primary vectors of improvement: the quality and complexity of training data, and the power of the underlying generative architecture.

Data-Centric Advancements. Recognizing that data quality is a critical bottleneck, recent efforts have shifted towards more sophisticated data curation strategies. Works like HQ-Edit (Hui et al., 2025) leveraged more powerful proprietary models, namely GPT-4V (Hurst et al., 2024) and DALL-E 3 (OpenAI, 2023), to generate higher-fidelity and better-aligned image–instruction pairs from scratch. A parallel and highly effective trend is the direct distillation of capabilities from frontier models. ShareGPT-4o-Image (Chen et al., 2025) exemplifies this by using GPT-4o’s own image generation API to create a high-quality dataset of over 90,000 text-to-image and image-editing samples, with the explicit goal of transferring its advanced skills to smaller, open-source models. Our work aligns with this data-centric philosophy, utilizing GPT-4o not just for generation, but for the systematic refinement and enhancement of existing large-scale datasets.

Architectural Evolution: From Diffusion to Flow Matching. The backbone of generative models has undergone significant evolution. Early methods were built upon U-Net-based diffusion models (Rombach et al., 2022), which, while effective, have limitations in modeling long-range dependencies. A major architectural shift came with Diffusion Transformers (DiT) (Peebles & Xie, 2023), which replaced the U-Net with a more scalable transformer architecture operating on latent patches and demonstrated superior performance as model size and compute increase. More recently, flow matching (FM) models have emerged as a more efficient alternative to traditional diffusion (Lipman et al., 2022). Instead of learning a multi-step denoising process, FM models learn to predict a continuous velocity field that transforms a simple prior distribution directly into the target data distribution by solving an ordinary differential equation. Our choice to fine-tune FLUX.1 Kontext (Labs et al., 2025) is motivated by its state-of-the-art FM-based architecture, which is a rectified flow transformer that unifies image generation and editing tasks with remarkable speed and consistency, making it an ideal foundation for our work.

Enhancing Semantic Control with MLLM Encoders. A final crucial advancement lies in improving the model’s comprehension of user instructions. The standard CLIP text encoders used in early models often fail to parse complex spatial, relational, or compositional commands (Radford et al., 2021). To overcome this, a clear trend has emerged towards replacing or augmenting these encoders with powerful multimodal large language models (MLLMs). State-of-the-art open-source models like Step1X-Edit (Liu et al., 2025) and UniWorld-V1 (Lin et al., 2025) have converged on a similar, highly effective architecture: both employ a potent MLLM, such as Qwen-VL (Bai et al., 2025) or LLaVA (Liu et al., 2023), to process the input image and textual instruction, extracting a rich semantic embedding that conditions a DiT or FLUX-based generative backbone. This approach leverages the MLLM’s superior cross-modal reasoning to achieve more precise semantic control. Our work adopts this SOTA paradigm by integrating Qwen-VL as a text and image encoder, demonstrating that the combination of an advanced flow matching architecture, enhanced semantic understanding via an MLLM, and a large-scale, high-quality refined dataset leads to significant performance gains.

Evaluation Benchmarks. We evaluate on four complementary instruction-editing benchmarks: GEdit-Bench-EN (Full), ImgEdit (Full), Complex-Edit, and OmniContext. GEdit-Bench-EN covers 11 edit types (background, color, material/texture, motion, portrait, style, add/remove/replace subject, text, tone) with MLLM-based scoring of instruction following and perceptual quality (Liu

Instruction re-write

Original

Apply a comic book-style filter to the image, adding bold outlines, vibrant colors, and

Make it a pop art piece.

texture resembling halftone dots.

Image regeneration

More accurate!

[Figure 18]

[Figure 19]

Re-write & Re-edit

[Figure 20]

[Figure 21]

Better alignment!

Original Image Edited Image

- Figure 2: An overview of GPT-IMAGE-EDIT-1.5M data curation pipeline. We applied mutiple methods to collect high-quality image-editing data. We used GPT-4o to re-write 10% instructions of the original OmniEdit dataset to make them more accurate, and the input images originally generated by DALL-E in HQ-Edit were re-synthesized by GPT-Image-1 for higher alignment.

et al., 2025); ImgEdit spans 9 task families (Add, Adjust, Extract, Replace, Remove, Background, Style, Hybrid, Action) with a unified evaluation pipeline (Ye et al., 2025); Complex-Edit composes chains of atomic edits to probe compositional reasoning and reports Instruction Following (IF), Identity Preservation (IP), and Perceptual Quality (PQ) (Yang et al., 2025); OmniContext targets context-aware in-context generation/editing (single/multiple/scene) and uses GPT-4.1 for interpretable, metric-driven assessment (Wu et al., 2025).

- 3 Data Curation

As the core motivation of this work is to explore the plausibility of replacing currently widely-used data collection methods for instruction-based image editing, which depend on predefined editing operation types and manually crafted expert pipelines, with a minimalist pipeline that utilizes a preexisting image editing model for various editing operations. The data curation pipeline is shown in Fig 2.

#### 3.1 Generation and Alignment Pipeline

To create the final GPT-IMAGE-EDIT-1.5M corpus, we merged OmniEdit, HQ-Edit, and UltraEdit, unifying their metadata into a consistent format. The core of our data generation process relied on the gpt-image-1 API, which we used to regenerate or augment all image-instruction pairs. To maintain consistency, all generated images were created in high-quality mode and constrained to one of three fixed aspect ratios: 1:1 (1024×1024), 3:2 (1536×1024), or 2:3 (1024×1536).

A key challenge was adapting source images with varying aspect ratios without introducing distortion or losing content. Na¨ıve resizing would stretch objects, while simple cropping could remove semantically relevant regions. Our solution was a simple pad-and-crop procedure: we snap each source image’s aspect ratio to the nearest supported ratio and add minimal white padding to match the target dimensions. After generation, this padding is precisely cropped away. Finally, we apply automatic quality filtering, rejecting any outputs with artifacts or residual padding. Further datasetspecific details are provided in the Appendix.

#### 3.2 Dataset Specific Processes

Instruction Re-writing For re-edited OmniEdit, we used GPT-4o to rewrite part of the instructions with the original input image and the updated output image in GPT-IMAGE-EDIT-1.5M. Under our cost limitation, we managed to rewrite ∼ 10% instructions of the original OmniEdit dataset.

Input Image Regeneration Since the input image in HQ-Edit is originally generated by DALL-E

##### 3, which is an arguably outdated image generation model, we regenerated about ∼ 50% of the input images with GPT-Image-1 and produced corresponding output images based on these re-generated input images.

Table 1: Comparison on the GEdit-EN-full benchmark. Our model achieves the highest average score among open-source methods.

BG Change

Color Alt.

Mat. Mod.

Model

Motion Portrait Style Add Remove Replace Text Tone Avg Open-Sourced Models

AnyEdit 4.31 4.25 2.64 0.67 1.90 1.95 3.72 3.75 3.23 0.77 4.21 2.85 MagicBrush 6.17 5.41 4.75 1.55 2.90 4.10 5.53 4.13 5.10 1.33 5.07 4.19 Instruct-Pix2Pix 3.94 5.40 3.52 1.27 2.62 4.39 3.07 1.50 3.48 1.13 5.10 3.22 OmniGen 5.23 5.93 5.44 3.12 3.17 4.88 6.33 6.35 5.34 4.31 4.96 5.01 Step1X-Edit 7.03 6.26 6.46 3.66 5.23 7.24 7.17 6.42 7.39 7.40 6.62 6.44 Bagel 7.44 6.99 6.26 5.09 4.82 6.04 7.94 7.37 7.31 7.16 6.17 6.60 Bagel-thinking 7.22 7.24 6.69 7.12 6.03 6.17 7.93 7.44 7.45 3.61 6.36 6.66 Ovis-U1 7.49 6.88 6.21 4.79 5.98 6.46 7.49 7.25 7.27 4.48 6.31 6.42 OmniGen2 - - - - - - - - - - - 6.42 Step1X-Edit(v1.1) 7.45 7.38 6.95 4.73 4.70 7.11 8.20 7.59 7.80 7.91 6.85 6.97 FluxKontext dev 7.06 7.03 5.52 5.62 4.68 5.55 6.95 6.76 6.13 6.10 7.48 6.26

Proprietary Models

Gemini 7.11 7.14 6.47 5.67 3.99 4.95 8.12 6.89 7.41 6.85 7.01 6.51 Doubao 8.07 7.36 7.20 5.38 6.28 7.20 8.05 7.71 7.87 4.01 7.67 6.98 GPT-4o 6.96 6.85 7.10 5.41 6.74 7.44 7.51 8.73 8.55 8.45 8.69 7.49

Ours 7.80 7.54 7.12 7.75 7.09 6.74 8.04 7.95 7.17 5.45 6.95 7.24

Table 2: Comparison on the ImgEdit-Full benchmark.

Model Add Adjust Extract Replace Remove Background Style Hybrid Action Overall

MagicBrush 2.84 1.58 1.51 1.97 1.58 1.75 2.38 1.62 1.22 1.90 Instruct-Pix2Pix 2.45 1.83 1.44 2.01 1.50 1.44 3.55 1.20 1.46 1.88 AnyEdit 3.18 2.95 1.88 2.47 2.23 2.24 2.85 1.56 2.65 2.45 UltraEdit 3.44 2.81 2.13 2.96 1.45 2.83 3.76 1.91 2.98 2.70 OmniGen 3.47 3.04 1.71 2.94 2.43 3.21 4.19 2.24 3.38 2.96 Step1X-Edit 3.88 3.14 1.76 3.40 2.41 3.16 4.63 2.64 2.52 3.06 ICEdit 3.58 3.39 1.73 3.15 2.93 3.08 3.84 2.04 3.68 3.05 BAGEL 3.56 3.31 1.70 3.30 2.62 3.24 4.49 2.38 4.17 3.20 UniWorld-V1 3.82 3.64 2.27 3.47 3.24 2.99 4.21 2.96 2.74 3.26 OmniGen2 3.57 3.06 1.77 3.74 3.20 3.57 4.81 2.52 4.68 3.44 Ovis-U1 4.13 3.62 2.98 4.45 4.06 4.22 4.69 3.45 4.61 4.00 FluxKontext dev 3.76 3.45 2.15 3.98 2.94 3.78 4.38 2.96 4.26 3.52

GPT-4o 4.61 4.33 2.90 4.35 3.66 4.57 4.93 3.96 4.89 4.20 Ours 4.07 3.79 2.04 4.13 3.89 3.90 4.84 3.04 4.52 3.80

Complex-Edit style instruction One key benefit of a data collection pipeline that utilizes a general-purpose image-editing model instead of a set of expert pipelines, each for one of the manually crafted editing operations, is that it can easily handle complex, composed instructions. Under this motivation, we generate Complex-Edit style instructions with ∼ 50% input images of OmniEdit. As we noticed that edited images by GPT-Image-1 with very complex instructions tend to lose the realistic feel, we opted for C3 level of complexity, that is each complex instruction is composed with 3 atomic instructions.

### 4 Experiments

#### 4.1 Experimental Setup

Models Our primary model, which we refer to as Ours, is built upon the state-of-the-art FluxKontext dev framework. We enhance its performance by replacing the original CLIP-based encoders with Qwen-VL-7b embeddings for both image and instruction conditioning, which improves semantic alignment. For our ablation studies, we also evaluate a SD3-Medium model, which uses a DiT-style architecture with channel-wise conditioning, and the original Flux 1.0 dev model, which uses token-wise control with SigLIP features (Zhai et al., 2023).

Benchmarks We evaluate our models on a comprehensive suite of benchmarks to assess various editing capabilities. These include GEdit-EN-full and ImgEdit-Full for general editing performance, Complex-Edit for compositional understanding, and OmniContext for context-aware editing.

- Table 3: Comparison on the Complex-Edit benchmark. Method IF IP PQ Overall

AnyEdit 1.60 8.15 7.25 5.67 UltraEdit 6.56 5.93 7.29 6.59 OmniGen 6.25 6.42 7.54 6.74 FluxKontext Dev 8.56 8.39 8.51 8.49

Imagen3 7.56 6.55 7.67 7.26 SeedEdit 8.49 6.91 8.74 8.04 GPT-4o 9.29 7.51 9.47 8.76

Ours 8.99 8.41 8.93 8.78

- Table 4: Results on the OmniContext SINGLE benchmark.

Character Object Average PF SC Overall PF SC Overall PF SC Overall

Method

InfiniteYou 7.81 5.15 6.05 — — — — — UNO 7.56 6.48 6.60 7.78 6.65 6.83 7.67 6.56 6.72 BAGEL 7.72 4.86 5.48 8.56 6.06 7.03 8.14 5.46 6.25 OmniGen 7.12 7.58 7.21 7.66 5.04 5.71 7.39 6.31 6.46 OmniGen2 8.04 8.34 8.05 8.44 7.26 7.58 8.24 7.80 7.81 Flux.1 Kontext (dev) 7.70 8.72 8.07 8.76 8.22 8.33 8.23 8.47 8.20

Flux.1 Kontext (max) 7.98 9.24 8.48 8.78 8.76 8.68 8.38 9.00 8.58 Gemini-2.0-Flash 5.54 5.98 5.06 6.17 5.89 5.17 5.86 5.93 5.11 GPT-4o 8.89 9.03 8.90 9.40 8.74 9.01 9.14 8.88 8.95

Ours 8.10 8.36 8.11 8.50 7.68 7.87 8.30 8.02 7.99

#### 4.2 Main Results

As demonstrated in Tables 1, 2, 3 and 4, our model, trained on the GPT-IMAGE-EDIT-1.5M dataset, achieves state-of-the-art performance among open-source methods and is highly competitive with leading proprietary models like GPT-4o. On GEdit-EN-full, our model scores an average of 7.236, outperforming all other open-source models. Similarly, on ImgEdit-Full, our model achieves an overall score of 3.80, again surpassing existing methods. In the challenging Complex-Edit benchmark with C8 complexity, which decomposes evaluation into Instruction Following (IF), Identity Preservation (IP), and Perceptual Quality (PQ), our model scores an impressive 8.78 overall, demonstrating a strong balance between accurately following instructions (8.99 IF) and preserving unchanged content (8.41 IP).

#### 4.3 Ablation Studies

We conduct a series of ablation studies to dissect the sources of performance improvement, isolating the effects of our data curation strategies and our model architecture choices.

Impact of Data Curation Strategies Table 5 shows the impact of our different data refinement strategies. Across all experiments, training on the baseline datasets yields the lowest scores. Our first refinement step, regenerating only the output image (-gpt or -output-regen), provides a substantial performance uplift. For instance, with Flux 1.0 dev on OmniEdit, this step improves the GEdit-EN score from 4.93 to 5.98. Our second step, regenerating the instruction to match the new output (-gpt-rewrite), provides a further boost to instruction-following capabilities, raising the imgedit score from 3.24 to 3.40. This validates our meticulous, multi-step approach to data curation.

Across all experiments, training on the baseline datasets yields the lowest scores. Our first refinement step, regenerating only the output image (-gpt or -output-regen), provides a substantial performance uplift. For instance, with Flux 1.0 dev on OmniEdit, this step improves the GEditEN score from 4.93 to 5.98. Our second step, regenerating the instruction to match the new output (-gpt-rewrite), provides a further boost to instruction-following capabilities, raising the imgedit score from 3.24 to 3.40.

- Table 5: Ablation studies on data curation strategies. We report imgedit and GEdit-EN scores for models trained on different 100k data variants.

Method Dataset Variant imgedit GEdit-EN OmniEdit Ablations

SD3-Medium omniedit100k-base 2.54 3.92 SD3-Medium omniedit100k-gpt 3.13 4.91 SD3-Medium omniedit100k-gpt-rewrite 3.32 4.89

Flux 1.0 dev omniedit100k-base 2.94 4.93 Flux 1.0 dev omniedit100k-gpt 3.24 5.98 Flux 1.0 dev omniedit100k-gpt-rewrite 3.40 5.88

HQ-Edit Ablations

SD3-Medium hqedit100k-base 2.19 2.00 SD3-Medium hqedit100k-output-regen 3.02 4.45 SD3-Medium hqedit100k-pair-regen 3.08 4.75

Flux 1.0 dev hqedit100k-base 3.12 4.34 Flux 1.0 dev hqedit100k-output-regen 3.44 5.67 Flux 1.0 dev hqedit100k-pair-regen 3.45 5.73

Complex-Edit Instruction Ablation Flux 1.0 dev Complex-Edit 2.89 5.39

- Table 6: Ablation on the inclusion of the Complex-Edit data subset on the GEdit-EN-full benchmark.

BG Change

Color Alt.

Mat. Mod.

Dataset

Motion Portrait Style Add Remove Replace Text Tone Avg

Ours w/o complex 7.62 7.55 6.77 7.08 6.74 6.74 7.68 7.74 6.82 5.36 7.23 7.03 Ours (full) 7.80 7.54 7.12 7.75 7.09 6.74 8.04 7.95 7.17 5.45 6.95 7.24

- Table 7: Ablation on the inclusion of the Complex-Edit data subset on the ImgEdit benchmark.

Dataset Add Adjust Extract Replace Remove Background Style Hybrid Action Overall

Ours w/o complex 4.07 3.69 1.94 4.17 3.93 3.73 4.74 2.91 4.19 3.71 Ours (full) 4.07 3.79 2.04 4.13 3.89 3.90 4.84 3.04 4.52 3.80

- Table 8: Ablation on different text encoder configurations on the GEdit-EN-full benchmark.

BG Change

Color Alt.

Mat. Mod.

Text Encoder

Motion Portrait Style Add Remove Replace Text Tone Avg

FluxKontext dev (T5) 7.06 7.03 5.52 5.62 4.68 5.55 6.95 6.76 6.13 6.10 7.48 6.26 Finetuned with T5 7.39 7.43 7.07 6.29 6.91 6.62 7.84 7.36 7.17 6.22 8.04 7.12 Finetuned with QwenVL 6.45 7.27 5.04 6.53 7.26 5.88 7.03 7.20 4.31 1.20 6.64 5.89 QwenVL + T5 (Ours) 7.80 7.54 7.12 7.75 7.09 6.74 8.04 7.95 7.17 5.45 6.95 7.24

Table 9: Ablation on different text encoder configurations on the ImgEdit benchmark.

Text Encoder Add Adjust Extract Replace Remove Background Style Hybrid Action Overall

FluxKontext dev (T5) 3.76 3.45 2.15 3.98 2.94 3.78 4.38 2.96 4.26 3.52 Finetuned with T5 4.19 3.79 2.09 4.22 3.96 3.90 4.76 3.23 4.49 3.85 Finetuned with QwenVL 3.92 3.58 1.95 3.62 3.89 3.72 4.64 3.22 3.82 3.60 QwenVL + T5 (Ours) 4.07 3.79 2.04 4.13 3.89 3.90 4.84 3.04 4.52 3.80

Interestingly, our ablation on using raw complex instructions (Complex-Edit) resulted in relatively poor performance (5.39 on GEdit-EN). Visual inspection of the outputs revealed a significant failure in identity preservation, where the model would alter uninstructed parts of the image. This critical finding underscores that merely increasing instruction complexity is insufficient and can even be detrimental if not paired with high-quality, aligned image pairs. This validates our meticulous, multi-step approach to data curation, which explicitly optimizes for instruction following, identity preservation, and perceptual quality.

###### Subject Addition Subject Replacement

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Replace the eagle with a parrot.

Add more hair to the front, making it long and soft for a gentle look.

###### Subject Removal

[Figure 26]

[Figure 27]

Remove the sticky notes next to the monitor.

###### Motion Change

Text Change

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Replace the text 'TOFEL' with 'IELTS'. Change the action of cat to sleeping.

Portrait Beautification

[Figure 32]

[Figure 33]

Generate my adult appearance.

###### Material Modification

[Figure 34]

[Figure 35]

Craft the outerwear from full-grain calfskin leather.

###### Background Change

[Figure 36]

[Figure 37]

Adjust the background to a glass wall.

Style Transfer

[Figure 38]

[Figure 39]

Modify this image in a Ghibli style.

###### Tone Transformation Color Alteration

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Change the color of jacket to purple.

Restore and colorize this old photo in high definition.

Figure 3: The qualitative results of our method on G-Edit-Benchmark-EN.

Impact of Complex-Edit Data and Text Encoders We further analyze the impact of including the Complex-Edit subset and the choice of text encoder. As shown in Table 6 and 7, including the Complex-Edit data provides a consistent, measurable boost across both GEdit-EN and ImgEdit benchmarks, improving the average scores from 7.03 to 7.24 and 3.71 to 3.80, respectively. This highlights the value of training on more challenging, compositional instructions.

Tables 8 and 9 ablate the choice of instruction text encoder with all text encoders frozen (T5, QwenVL, and CLIP). We fine-tune the rest of FluxKontext end-to-end. Using frozen T5 already lifts GEdit-EN from 6.26 to 7.12. Using frozen Qwen-VL alone underperforms on the Text category 1.20, likely due to tokenizer merges that hinder isolating target strings. Concatenating frozen QwenVL and frozen T5 features yields the best GEdit-EN average 7.24 and competitive ImgEdit overall

- 3.80, while retaining CLIP’s pooled global features for architectural parity.
- 4.4 Qualitative Results

In Fig 3, 4, 5, and 6, we present the qualitative editing results of the model trained on GPT-IMAGEEDIT-1.5M on different types editing scenarios as defined by these benchmarks. It is clear that the model trained on GPT-IMAGE-EDIT-1.5M demonstrates a good understanding of the editing instruction as well as generating realistic images while keeping elements that are not instructed to change the same.

### 5 Conclusion

In this work, we addressed the critical need for a large-scale, high-quality dataset to advance opensource instruction-based image editing. We introduced GPT-IMAGE-EDIT-1.5M, a corpus of over 1.5 million samples created by systematically refining and unifying existing datasets using GPT4o. Our data curation process was explicitly guided by the core principles of a successful edit: instruction following, identity preservation, and perceptual quality. Our experiments demonstrate

###### Addition

###### Removement

###### Replacement

###### Object Extraction

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Add a set of colorful beach towels hanging over the railing on the right side of the pier.

Remove the transport object in the image.

Replace the yacht in the image with a hot air balloon floating just above the ocean surface.

Extract the bird from the image, keeping it isolated from the surrounding apples and background.

###### Attribute Alter Motion Change

Background Change

Style Transfer

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Change the color of the vehicle to red.

Raise the person's left arm.

Change the background color from black to a light blue sky with fluffy white clouds.

Transfer the image into a vibrant graffiti street-mural style.

Hybrid

[Figure 60]

[Figure 61]

Remove the plant on the right side of the image, and adjust the man's suit to a darker shade of blue.

##### Figure 4: The qualitative results of our method on Img-Edit.

[Figure 62]

[Figure 63]

In a cozy studio, the person sits in a chair and speaks into a microphone.

[Figure 64]

[Figure 65]

A woman is playing with colorful beads and small cars on the red carpet.

[Figure 66]

[Figure 67]

[Figure 68]

Display the intricate mandala artwork on the wall above the marble fireplace in the elegant living room, enhancing the serene atmosphere as the fire crackles softly.

[Figure 69]

[Figure 70]

Show the man in a suit embracing his partner in a lavender field, both smiling and holding bouquets.

[Figure 71]

[Figure 72]

Dance on the stage beneath the red curtain.

[Figure 73]

[Figure 74]

In the cozy corner of a rustic tropical restaurant, a Steller's sea eagle perches gracefully on a wicker chair, its sharp eyes scanning the warm ambiance filled with greenery and soft hanging lights.

Figure 5: The qualitative results of our method on OmniContext.

the significant impact of this new dataset. By fine-tuning a state-of-the-art open-source model on GPT-IMAGE-EDIT-1.5M, we achieved new SOTA performance across multiple benchmarks, substantially narrowing the gap with proprietary models. The ablation studies validated our multifaceted data generation strategy, showing that each refinement step—from output regeneration to instruction rewriting—provides tangible benefits. Crucially, we also highlighted that data quality, specifically the tight alignment between instruction and image while preserving identity, is more important than mere instruction complexity. By releasing the GPT-IMAGE-EDIT-1.5M dataset and our fine-tuned models, we provide a powerful resource for the research community. We hope this work will catalyze further innovation in open-source image editing, enabling the development of models with even greater capabilities. Future work could explore applying this data curation methodology to other modalities like video or 3D, or investigate more automated techniques for detecting and correcting subtle misalignments in generative data pipelines.

[Figure 75]

[Figure 76]

Modify the mirror reflections to display a window with a garden view and increase the ambient lighting. Replace the black faucet with a stainless-steel one, and change the countertop texture to a white marble finish. Remove the soapdish near the right sink. Resize the ornamental vase to make it slimmer, and add a small potted plant next to the left sink. Apply a warm filter to the image.

[Figure 77]

[Figure 78]

Transform the scene into a moody, rainy vintage depiction by replacing the clear sky with a cloudy, stormy one, colorizing with muted vintage tones, and adding falling raindrops. Introduce a vintage car near the curb, apply a wet, reflective texture to the road, and dim the overall lighting while adding subtle fog at ground level. Conclude with a sepiatone filter to enhance the vintage atmosphere.

[Figure 79]

[Figure 80]

Replace the wall with an artistic mural and change the wooden bunk bed frame to white. Add a hammock hanging above the bunk beds diagonally, and modify the bedding to have floral patterns. Remove the chair near the flowers. Illuminate the scene with warm golden light, add a glowing aura to the bouquet, and finish with a vintage-style filter.

[Figure 81]

[Figure 82]

Change the wall color to a vibrant sky blue and the floor to a polished marble texture. Replace the TV with a wooden shelf with books and move the coffee table slightly closer to the couch. Add a vase with colorful flowers on the dining table, increase the brightness of the lamp, and apply a warm sepia tone across the image. Introduce softly glowing light trails along the ceiling corners.

Figure 6: The qualitative results of our method on Complex-Edit.

### 6 Acknowledge

We would like to thank Ashwin Nagarajan, Tejas Polu, Jiawei Mao, Zeyu Wang and Haoqin Tu for the early discussion and exploration of this project. We would like to thank the Microsoft Accelerate Foundation Models Research Program for supporting our computing needs.

### References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 18392–18402, 2023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv:2506.18095, 2025.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Cihang Xie, and Yuyin Zhou. HQ-edit: A high-quality dataset for instruction-based image editing. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=mZptYYttFj.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6007–6017, 2023.

Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.

OpenAI. GPT-4v System Card. https://openai.com/research/dall-e-3-system-card, 2023. William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of

the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Yichun Shi, Peng Wang, and Weilin Huang. Seededit: Align image re-generation to image editing. arXiv preprint arXiv:2411.06686, 2024.

Peng Wang, Yichun Shi, Xiaochen Lian, Zhonghua Zhai, Xin Xia, Xuefeng Xiao, Weilin Huang, and Jianchao Yang. Seededit 3.0: Fast and high-quality generative image editing. arXiv preprint arXiv:2506.05083, 2025.

Cong Wei, Zheyang Xiong, Weiming Ren, Xeron Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=Hlm0cga0sv.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.

Siwei Yang, Mude Hui, Bingchen Zhao, Yuyin Zhou, Nataniel Ruiz, and Cihang Xie. Complexedit: Cot-like instruction generation for complexity-controllable image editing benchmark. arXiv preprint arXiv:2504.13143, 2025.

Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11975–11986, 2023.

Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id=9ZDdlgH6O8.

### A Dataset-Specific Processing Details

#### A.1 UltraEdit Downscale Workflow

The UltraEdit dataset originally uses 512×512 inputs. Our workflow was to regenerate both the input (where applicable) and the output at 1024×1024 using our standard procedure, and then downsample both images back to 512×512 using bicubic interpolation to maintain compatibility with the original benchmark’s expectations.

#### A.2 OmniEdit Alignment Procedure

For each sample in OmniEdit, we applied our standard geometric alignment: compute the image’s ratio, pad to the nearest supported generation ratio, and run the edit. After generation, we crop the padding and resize the image back to its original dimensions to ensure comparable pixel density. We implemented a strict quality filter, rejecting any sample if more than 0.5% of its border consisted of uniform padding after the process.

#### A.3 Complex-Edit Subset

The Complex-Edit subset was handled similarly to OmniEdit regarding the geometry and padding procedure. It contains only the canonical complex instructions. We applied a stringent filter, discarding any output that had a detectable padding error after the crop step.

##### A.4 HQ-Edit Dual Splits The HQ-Edit portion of our dataset was processed in two distinct splits:

Edit Split For existing pairs, we pad the original input image, generate the edit, crop the padding,

and restore the image to its original resolution.

Generate Split For generation tasks, we first synthesize a new reference input image from the textual instruction. We then apply the same edit instruction to this newly generated input. For these tasks, the aspect ratio was chosen randomly from our three supported options (1:1, 2:3, 3:2) to increase diversity.

