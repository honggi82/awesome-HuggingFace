# arXiv:2512.04810v5[cs.CV]15Dec2025

## EMMA: Efficient Multimodal Understanding, Generation, and Editing with a Unified Architecture

Xin He∗, Longhui Wei∗†‡, Jianbo Ouyang∗, Minghui Liao, Lingxi Xie, Qi Tian‡ ∗ Equal Contribution † Project Lead ‡ Corresponding Author weilh2568@gmail.com, tian.qi1@huawei.com Huawei Inc. Project Page: https://emma-umm.github.io/emma/

### Abstract

We propose EMMA, an efficient and unified architecture for multimodal understanding, generation and editing. Specifically, EMMA primarily consists of 1) An efficient autoencoder with a 32x compression ratio, which significantly reduces the number of tokens required for generation. This also ensures the training balance between understanding and generation tasks by applying the same compression ratio to images. 2) Channel-wise concatenation instead of token-wise concatenation among visual understanding and generation tokens, which further reduces the visual tokens in unified architectures. 3) A shared-and-decoupled network that enables mutual improvements across tasks while meeting the task-specific modeling requirements. 4) A mixture-of-experts mechanism adopted for visual understanding encoder, which substantially improves perceptual capabilities with a few parameters increase. Extensive experiments have shown that EMMA-4B can significantly outperform state-of-the-art unified multimodal approaches (e.g., BAGEL-7B) in both efficiency and performance, while also achieving competitive results compared to recent multimodal understanding and generation experts (e.g., Qwen3-VL and Qwen-Image). We believe that EMMA lays a solid foundation for the future development of unified multimodal architectures.

#### 1. Introduction

Recently, unified multimodal architectures have garnered unprecedented attention. A growing number of researchers have recognized that pushing the boundaries and limitations of unified multimodal foundation models is crucial to the progress of multimodal research, embodied artificial intelligence, and even artificial general intelligence [13, 14, 18, 24, 47, 51, 53]. Consequently, unified multimodal models have seen rapid development.

Currently, unified multimodal approaches [9, 11, 14, 26, 47, 53, 59] can be broadly categorized into three main directions: 1) Unification of architecture formats: These approaches (such as BAGEL [14] and JanusFlow [33]) facilitate the deep interaction and integration between understanding and generation tasks. They aim to train unified multimodal architectures in an end-to-end manner to unlock more powerful capabilities and emerge potential intelligence. 2) Unification of task formats: These methods usually connect a generation decoder to an

- Table 1 | Comparison between different unified multimodal approaches across multimodal understanding, text-to-image generation, and image editing benchmarks. Params represents the number of parameters of utilized LLM. † denotes the methods using prompt rewriting. More results are shown in Section 5.

Model Params Understanding T2I Generation Image Editing

| | |MMBench [29] MMMU [65] MMVet [64]|GenEval [17] DPG-Bench [21]|GEdit-Bench-EN [28]|
|---|---|---|---|---|
|Janus-Pro [11] Mogao [25] OmniGen2 [55] BLIP3-o [9] UniWorld-V1 [26] BAGEL [14]|7B 7B 3B 7B 7B 7B<br><br>|79.2 41.0 50.0 75.0 44.2 79.1 53.1 61.8<br><br>83.5 58.6 66.6<br>83.5 58.6 67.1 85.0 55.3 67.2<br>|0.80 84.19 0.89 84.33<br><br>0.80 / 0.86† 83.57<br><br>0.84† 81.60 0.80 / 0.84† 81.38 0.82 / 0.88† 85.07|6.42 4.85 6.52|
|EMMA|4B|85.8 62.5 73.0<br><br>|0.91 / 0.93† 85.63<br><br>|6.53|

existing multimodal understanding foundation model via specific bridging mechanisms, thereby enabling multimodal generation and complex instruction-based editing. Representative works include MetaQuery [38], OminiGen2 [55], and etc. 3) Unification of learning paradigms: These approaches (such as EMU3 [47] and D-DiT [24]) focus on optimization consistency in the unified model, i.e., employing the same learning paradigm (e.g., next-token prediction or denoising diffusion loss) for both generation and understanding tasks.

The above technical directions have not yet converged, and each exhibiting distinct advantages and limitations. In this work, EMMA is proposed to solve the inherent issues in the "unification of architecture formats". Specifically, EMMA first uses a higher-compression autoencoder (DCAE [7], 32x compression), that aligns the compression ratio in the visual understanding branch. This alignment enables the training balances between understanding and generation tasks within the same architecture. Furthermore, benefiting from the same compression ratio used in both the understanding and generation branches, visual tokens extracted from these branches can be concatenated along the channel dimension instead of the token level like BAGEL [14]. This strategy effectively reduces visual context tokens while maintaining the information fusion and complementarity between understanding and generation tasks. In addition, EMMA incorporates a shared-and-decoupled mechanism in its network design. Unlike BAGEL, EMMA introduces cross-task parameter sharing at shallow layers to facilitate mutual improvements between multimodal tasks, while employing parameter decoupling at deeper layers to meet the distinct modeling requirements of understanding and generation. Inspired by Mixpert [19], EMMA also integrates a mix-of-experts (MoE) strategy into the visual understanding encoder to improve its perceptual ability across various types of input images.

Using the above designs, EMMA significantly surpasses recent unified multimodal models in extensive multimodal understanding, generation, and editing benchmarks [17, 28, 32, 35], despite using relatively fewer training data and parameters than previous approaches [14, 25]. As shwon in Table 1, EMMA achieves a score of 73.0 on MMVet [64] and 0.93 on GenEval [17] based on 4B LLM, compared to the larger unified model BAGEL-7B achieving with 67.2 on MMVet and 0.88 on GenEval. Notably, EMMA only needs 20% visual context tokens compared to BAGEL while handling cross-model interaction generation tasks (e.g. image editing). Moreover, compared to the recent open-source state-of-the-art multimodal understanding models, such as InternVL3.5 [46] and Qwen3-VL [2], EMMA also achieves competitive results on multiple understanding benchmarks [32, 35, 36]. In addition, EMMA also demonstrates superior performance over Qwen-Image [54] on GenEval [17] (0.91 vs. 0.87 without using prompt rewriting or reinforcement learning strategies). More visualization results are shown in Figure 1.

[Figure 1]

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

[Figure 16]

[Figure 17]

[Figure 18]

###### Figure 1 | Showcase of EMMA’s text-to-image generation capability.

The contributions of this work are summarized as follows:

- • Efficiency. Through a high-compression autoencoder and channel-wise concatenation mechanism, EMMA significantly reduces the visual tokens (e.g., reducing by 5x in image editing task) compared to recent unified models like BAGEL, boosting end-to-end training and inference efficiency.
- • Performance. Benefiting from the shared-and-decoupled architecture design and billions of training data, EMMA achieves superior performances compared to recent unified multimodal models.
- • Potential. EMMA shows competitive results, achieved by a unified architecture training with the end-to-end manner, compared to understanding and generation state-of-the-arts. The above results further highlight its potential for cross-modal tasks.

#### 2. Method

The overall architecture of EMMA is shown in Figure 2. The core of EMMA lies in its visual encoder module and network architecture. The encoder ensures effective visual information fusion, while the network architecture enables cross-task parameter sharing and meets the distinct modeling requirements of understanding (semantics modeling) and generation (semantics and high-frequency details modeling [34]) tasks. The details of these two modules are shown below.

###### 2.1. Visual Encoder

Choice of Visual Understanding and Generation Encoder. For unified multimodal architectures, there are usually two types of visual encoder: understanding encoder (denoted as Und-Enc) and generation encoder (denoted as Gen-Enc). For Und-Enc, recent works generally adopt SigLIP [44, 66] to encode images as visual tokens and extract the corresponding semantic information, followed by a 2x2 pixel shuffle strategy [12] to further reduce these tokens to one quarter before feeding them into the Large Language Model (LLM). In this work, we directly utilize SigLIP2-so400m-patch16-512 [44] as Und-Enc. Meanwhile, we extend its capability to support native resolutions of input images by interpolating the positional embeddings. As a result, Und-Enc, through the patchfy operation in SigLIP2 and pixel shuffling strategy, achieves a 32× compression ratio of the input image. For example, a 1024×1024 resolution image is compressed into 1024 visual tokens. For Gen-Enc, we employ a high-compression autoencoder, DCAE [8], with a 32× compression ratio. Compared to other unified architectures that generally adopt the autoencoder (AE) with an 8x compression ratio (e.g., the AE in FLUX [22]) and 2x2 token merging strategy, EMMA only requires 1/4 visual tokens for the generation task. Experimental results in Section 5 demonstrate that using the above high-compression AE still yields competitive generation quality.

Fusion of Visual Information. Furthermore, since both Und-Enc and Gen-Enc utilize the same compression ratio (32×), EMMA can directly perform channel-wise concatenation of the corresponding visual tokens, rather than the token-wise concatenation used in previous approaches like BAGEL [14]. This allows EMMA to effectively fuse semantic information (from Und-Enc) with detailed information (from Gen-Enc) without increasing the total number of visual tokens, thereby supporting more efficient training and inference of unified models. Consequently, compared to recent state-of-the-art unified models (e.g., BAGEL [14]), EMMA significantly reduces the visual tokens (e.g., reducing by 5x in image editing task), while still achieving superior performances across various multimodal understanding, generation, and editing benchmarks [17, 28, 65].

###### Next Token Prediction Velocity Prediction

|||Und FFN|
|---|
<br><br>|Gen FFN|
|---|
|
|---|
| |Unified Multimodal Architecture|
|---|---|---|
||Self Attention|
|---|
<br><br>X M| | |
||Und QKV|
|---|
<br><br>|Gen QKV|
|---|
| |||Und Adapter|
|---|
<br><br>|Gen Adapter|
|---|
|
|---|
|
|||Und FFN|
|---|
<br><br>|Gen FFN|
|---|
<br><br>|Self Attention|
|---|
|
|---|
<br><br>X N| |Gen Concat<br><br>Tokens Und Tokens<br><br>Router|
||Und V|
|---|
<br><br>|Shared QK|
|---|
<br><br>|Gen V|
|---|
| |STEM Expert<br><br>[Figure 19]<br><br>|[Figure 20]|
|---|
<br><br>[Figure 21]<br><br>[Figure 22]|
||Text Tokenizer|
|---|
<br><br>|Und Adapter|
|---|
<br><br>|Gen Adapter|
|---|
| |Versatile Expert<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>|[Figure 26]|
|---|
|
|Mixpert Please describe the image in details<br><br>DCAE| |Router Shared Component<br><br>Mixpert|

Text Modality Vision Modality

- Figure 2 | The overall architecture of our EMMA. "Und Adapter" and "Gen Adapter" represent understanding adapter and generation adapter for projecting visual tokens into unified multimodal architecture, respectively.

Mixture of Experts. In addition, the types of input images in multimodal understanding tasks are highly different. Inspired by Mixpert [20], EMMA further extends SigLIP2 into a mixture-ofexperts architecture. As shown in Figure 2, EMMA additionally introduces a STEM (science, technology, engineering, and math) expert to process STEM images, with a router module dynamically selecting the appropriate expert for each input. Specifically, when the input image is identified as STEM data by the router, it would be fed into the STEM expert; otherwise, it defaults to the versatile expert. Given that STEM data appear almost exclusively in multimodal understanding tasks, the weights of STEM expert are first initialized from the versatile expert and only trained in the last tuning phase with the mutlimodal understanding data (the training details are presented in Section 4). Experimental results in Section 5 demonstrate that by adding only approximately 50M additional parameters and few fine-tuning steps, EMMA improves performance on multimodal understanding benchmarks while preserving original generation and editing capabilities.

###### 2.2. Network Architecture Design

Network Design. As shown in Figure 2, EMMA consists of a shared-and-decoupled network. This design is mainly motivated by understanding (denoted as Und) tasks focusing on semantics modeling and generation (denoted as Gen) tasks focusing on semantics and high-frequency information (details) modeling, thereby there are both common and independent characteristics. Considering that deeper layers exhibit stronger relevance to specific tasks, EMMA adopts a mechanism of sharing shallow layers while keeping deeper layers independent. Additionally, in shallow layes, EMMA still maintains a portion of parameters (e.g., the value projection layers)

as task-specific module to ensure task independence. Moreover, cross-modal interactions are achieved through self-attention layers. In particular, the weights of all layers in both Und and Gen branches are initialized from Qwen3-4B [62].

Data Processing and Optimization Objective. Therefore, for each task, textual inputs are processed through the Und branch. Visual inputs first undergo semantic feature and details extraction via the modified Mixpert and DCAE. These extracted information are then fused through channel concatenation, and then feed into Und or Gen branch according to the corresponding tasks. As for Und tasks, EMMA utilizes the next-token prediction mechanism to guide overall learning. For Gen tasks, EMMA utilizes flow matching with velocity prediction. Notably, before feeding visual tokens into the LLM, the 2D positional encoding is applied to incorporate spatial priors. Subsequently, all textual and visual tokens are treated uniformly and processed using the positional embedding of 1D RoPE [43].

Attention Strategy. Following Transfusion [68], EMMA adopts a hybrid attention strategy. Specifically, pure casual masking is utilized to both textual and visual tokens in Und tasks. As for Gen tasks, textual tokens are still restricted to attend previous tokens, and visual tokens can attend both of previous tokens and other visual tokens within the same image.

#### 3. Data Construction

EMMA is trained with end-to-end manner, and the training data consists of mutlimodal understanding data, text-to-image generation data and instruction editing data. The detailed statistics are presented in Table 2.

Multimodal Understanding Data (I2T). This dataset is primarily constructed from six types of data: alignment data, pre-training (PT) data, supervised fine-tuning (SFT) data, quality tuning (QT) data, STEM expert tuning data (ET), and router tuning data (RT). Alignment data are used to align the extracted visual tokens with LLM, and this work directly uses LLaVA-558K [27]. PT data mainly consists of text-image pairs, sourced predominantly from large-scale open-source datasets such as LAION [41], as well as internal data. Additionally, re-captioning models [4, 12] are employed to regenerate captions for these images. SFT data are largely composed of imagequestion-answer triplets. These are mainly derived from large-scale, high-quality open-source datasets (including LLaVA-OneVision-Data [23], FineVision [52] and etc.), along with the internal constructed datasets. SFT data cover a wide range of types, such as document parsing, chart recognition, optical character recognition, mathematical problem solving, etc. QT data are built by selecting higher-quality samples from SFT data and performing balanced sampling across each task. ET data consist of 15M STEM data sourced from SFT data, and RT data are constructed by combining 3M data with both STEM and general data.

Text-to-Image Generation Data (T2I). This dataset is composed of three data types: PT, SFT, and QT data. PT data are mainly filtered from large-scale open-source datasets (such as LAION [41]) as well as internal datasets, based on aesthetic quality. SFT data are selected according to the filter criterias such as image resolution (1K resolution and above) and aesthetic score, with the balanced sampling between general images and portrait images. To address the scarcity of text rendering data, we have also synthesized text rendering images with current state-of-the-art generation models [54].

Image Editing Data (IT2I). Currently, several high-quality image editing datasets have been released, such as X2I2 [55] and OmniEdit [50]. In addition to leveraging these existing opensource datasets, we also use state-of-the-art editing models (e.g., Qwen-Image-Edit [54]) to

- Table 2 | The training data and hyperparameters of EMMA. I2T, T2I and IT2I represents the multimodal understanding, text-to-image generation and image editing data, respectively. LR denotes the learning rate.

##### Alignment PT SFT QT ET RT Training Data(M)

I2T Data 0.56 400 120 1 15 3 T2I Data - 600 105 0.15 - IT2I Data - - 12 0.35 - -

Hyperparameters Und resolution (512,512) (512,512) native native native native Gen resolution - (512,512) 1K 1K - LR (×10−4) 10.0 1.0 0.4 0.1 0.04 1.0 LR scheduler Cosine Constant Constant Cosine Cosine Constant Optimizer AdamW (𝛽1 = 0.9, 𝛽2 = 0.999, 𝜖 = 1.0 × 10−8)

synthesize a number of high-quality editing pairs. The construction and filtering pipeline is illustrated in Figure 3. The above pipeline is designed to produce data covering various editing types, including object addition, removal, replacement, background transfer, tone transfer, and virtual try-on. Especially, we also construct a data pipeline for text editing tasks, and the details are shown in Figure 4. The final SFT data are constructed by combining the open-source datasets with our synthesized editing pairs. Furthermore, the synthesized editing data (the reference images are sourced from T2I QT data) are regarded as the final editing QT data. It should be noted that although the recently released GPT-Image-Edit-1.5M [48] can significantly improve performance on image editing benchmarks, it severely disrupts the character consistency. Therefore, we believe that this dataset is harmful to region-based editing tasks and excludes it from our training corpus. This observation also highlights the need for more accurate evaluation metrics in image editing (such as subject consistency), which is left for future work.

#### 4. Training Details

- Stage 0: Alignment. This stage aims to align the visual tokens with unified model. Therefore, the parameters of visual encoders and unified model are frozen, and only the adapter in Und branch is tuned. The input image is fixed at 512x512. The learning rate is set as 1𝑒−3. As for Gen branch, the adapter is randomly initialized and left for training during the pre-training stage.
- Stage 1: Pre-Training (PT). In this stage, all parameters of EMMA (expect for visual generation encoder, i.e., DCAE) are trained. The input image is set as 512x512 for both Und and Gen branches, and their corresponding batch sample ratio is set as 1:1. The learning rate is set as 1𝑒−4.
- Stage 2: Supervised Finetuning (SFT). Consistent with PT stage, all parameters are still tuned except for DCAE in this stage. Differently, Und branch supports training with the native resolution of input images, while Gen branch supports training with input images of which are scaled to the nearest pre-defined bucket size of 1K resolution based on their original aspect ratios. Furthermore, after training with one epoch on the initial I2T and T2I SFT data, EMMA is further fine-tuned by using a balanced sampling strategy. Specifically, for T2I SFT data, we select ∼50M samples with a 1:1 ratio between portrait and general images, while ensuring the

[Figure 27]

[Figure 28]

Editing Pair Generation

| | | |Editing| |
|---|---|---|---|---|
| |VLM|Add a dog on the sandy land|model| |

Reverse instruction

Remove the dog on the sandy land

Editing Pair Filter

VLM

Matching score Face similarity

- Figure 3 | Overview of the general editing data construction pipeline. The data pipeline consists of two parts. The first is to generate image editing pairs: firstly, we use a VLM to generate editing instruction for the input image, then use an image editing model to generate the edited image as well as reversing the editing instruction to obtain the reversed pair. The second part is to filter the image editing pairs: we use a VLM to determine whether the edited image follows the editing instruction, and if the image contains portraits, we further filter the pair using face similarity.

balances across aspect ratios and incorporating aesthetic-based ranking strategy. Similarly, in I2T SFT data, we sample ∼50M instances with a 1:1 ratio between STEM and general categories. Towards the end of its training, we further incorporate IT2I SFT data and train EMMA with a balanced mixture ratio of 1:1:1 between all three tasks.

- Stage 3: Quality Tuning (QT). Consistent with SFT stage, we conduct joint training among T2I, I2T, and IT2I tasks with a balanced batch ratio of 1:1:1. The initial learning rate is set as 1𝑒−5 in this stage.
- Stage 4: STEM Expert and Router Tuning (ET&RT). In these two stages, only the parameters of STEM expert or router module are trained and other parameters are frozen. The STEM expert is trained with 12M STEM data, and the router is trained with the specially chosen 3M data as described in Section 3. The initial learning rate is set as 4𝑒−6 in ET stage and 1𝑒−4 in RT stage.

#### 5. Experiments

###### 5.1. Comparisons on Understanding Tasks

We compare EMMA against current state-of-the-art methods on multiple multimodal understanding benchmarks. As shown in Table 3, EMMA significantly outperforms existing unified multimodal architectures. For example, EMMA, built on 4B LLM, can significantly surpass BAGEL-7B by a notable margin, e.g., achieving with a 5.8% improvement on the MMVet benchmark.

Furthermore, reconstructing the vision encoder with a mixture-of-experts (MoE) design further improves the accuracy of EMMA in understanding benchmarks, yielding an average gain of 0.4% across 11 evaluation datasets. In particular, EMMA remains highly competitive results compared to current multimodal understanding state-of-the-arts, e.g., it outperforms Qwen3-VL

[Figure 29]

[Figure 30]

Text Editing Pair Generation

| |OCR &|Change “BEACH” to “EMMA”|Editing| |
|---|---|---|---|---|
| |Select a word “BEACH”| |model| |

Reverse instruction

Change “EMMA” to “BEACH”

Text Editing Pair Filter

OCR

Match or not

- Figure 4 | Overview of the text editing data construction pipeline. The data pipeline consists two parts. The first is to generate text-edited image pairs: firstly, text detection is performed on the input image to extract text information, and then one or more words are randomly selected for replacement or removal, while generating the corresponding editing instruction. We further use a image editing model to produce the edited image. The second part is to filter the image pairs, where the OCR model is used to determine whether the edited image follows the instruction or not.

by an average of 0.4% and exceeds InternVL3.5 by an average of 2.6% on 11 evaluation datasets. The above results clearly demonstrate the effectiveness of EMMA.

###### 5.2. Comparisons on Text-to-Image Generation Tasks

To further evaluate the text-to-image generation capability of EMMA, we conduct comprehensive comparisons on GenEval [28] and DPG-Bench [21] against both unified multimodal architectures and current text-to-image (T2I) generation approaches. As shown in Table 4 and Table 5, EMMA significantly outperforms existing unified approaches in T2I generation benchmarks. For example, on GenEval, EMMA achieves a score of 0.91 without prompt rewriting and reinforcement learning, surpassing BAGEL-7B (0.82) and Qwen-Image (0.87), despite the latter having a larger model size (4B vs. 20B). It is worth noting that, to our best knowledge, EMMA is the first to reach a score of 0.91 on GenEval without relying on prompt rewriting or reinforcement learning. This further demonstrates the benefits of unified architecture for T2I generation task. As shown in Figure 1, EMMA is capable of generating high-quality images following the given textual instruction.

###### 5.3. Comparisons on Editing Tasks

We also compare EMMA with current unified architectures on image editing task. As shown in Table 4, EMMA has a slight advantage over the existing unified multimodal models on GEdit [28]. This marginal improvement may be attributed to two reasons: 1) the image-text interaction data used in our approach is relatively limited (12M in EMMA vs. 65M in BAGEL); 2) image editing is a more complex task, and addressing the corner cases should require larger model. It is worth noting that in image editing, EMMA only needs 1/5 of the visual tokens for reference images but achieves better results than BAGEL-7B.

In addition, we have observed the limitation of evaluation metrics in GEdit. Specifically,

- Table 3 | Comparisons with state-of-the-arts on multimodal understanding benchmarks. Params represents the number of parameters of utilized LLM. § denotes the results tested by ourself with VLMEvalKit [15] using the officially released instruct-version checkpoint. CQA, DVQA, TVQA, IVQA, OCRB, MMB. MVista, MMS and MMV represents ChartQA [35], DocVQA [36], TextVQA [42], InfoVQA [37], OCRBench [30], MMBench [29], MathVista [31], MMstar [10] and MMVet [64], respectivelly. W/O represents the method without extending SigLIP2[44] into a mixture-of-experts architecture.

Method Params CQA DVQA TVQA IVQA OCRB MMB MMMU MVista AI2D MMS MMV Avg

|LLaVA-OV-1.5 [3] InternVl-3.5§ [46] Qwen3-VL§ [2]|4B 4B 4B|87.1 94.4 - 76.1 80.0 84.2 52.7 67.9 83.6 64.9 86.3 91.9 77.8 77.5 81.7 81.4 60.0 68.4 82.0 64.0 76.1 82.8 95.2 80.9 79.9 88.1 83.6 67.4 73.7 83.9 69.8 66.2<br><br>|77.0 79.2<br><br>|
|---|---|---|---|
|Janus Pro [11] EMU3 [47] VILA-U [56] ILLUME [45] MUSE-VL [61] OminiGen2 [55] UniWolrd-V1 [26] BLIP3-o [9] Mogao [25] Show-o2 [60] BAGEL [14]|7B<br>8B 7B 7B 7B 3B 7B 7B 7B 7B 7B<br>|- - - - - 79.2 41.0 - - - 50.0 68.6 76.3 64.7 43.8 68.7 58.5 31.6 - 70.0 - 37.2<br>- - 60.8 - - - - - - - 33.5 66.7 76.0 - 45.5 66.9 75.1 38.2 - 71.4 - 37.0<br>- - - - - 72.1 39.7 51.3 69.8 49.6 -<br>- - - - - 79.1 53.1 - - - 61.8<br>- - - - - 83.5 58.6 - - - 67.1<br>- - - - - 83.5 58.6 - - - 66.6<br>- - - - - 75.0 44.2 - - - -<br>- - - - - 79.3 48.9 - 78.6 56.6 -<br>- - - - - 85.0 55.3 73.1 - - 67.2<br><br><br>|-|
|EMMA (W/O) EMMA|4B 4B|87.8 95.5 79.0 77.4 88.4 85.6 62.0 75.4 83.6 64.2 72.5<br>88.0 95.9 79.5 77.7 89.0 85.8 62.5 75.8 83.9 64.8 73.0<br><br><br>|79.2 79.6|

- Table 4 | Quantitative evaluations of the text-to-image generation capacity on GenEval [17]. † denotes the methods using prompt rewriting.

Method Single object Two object Counting Colors Position Color attribution Overall↑

|LUMINA-Next [69] SDXL [39] FLUX.1-dev [22] FLUX.1-dev† [22] SD3-medium [16] SANA-1.5 [58] HiDream-I1-Full [6] Qwen-Image [54] Qwen-Image-RL [54]|0.92 0.46 0.48 0.70 0.09 0.13<br><br>0.98 0.74 0.39 0.85 0.15 0.23<br>0.99 0.81 0.79 0.74 0.20 0.47<br><br><br>0.98 0.93 0.75 0.93 0.68 0.65<br>0.99 0.94 0.72 0.89 0.33 0.60<br><br><br>0.99 0.93 0.86 0.84 0.59 0.65<br>1.00 0.98 0.79 0.91 0.60 0.72<br><br><br>0.99 0.92 0.89 0.88 0.76 0.77<br>1.00 0.95 0.93 0.92 0.87 0.83<br><br><br>|0.46 0.55 0.67 0.82 0.74 0.81 0.83 0.87 0.91|
|---|---|---|
|ILLUME [45] Emu3-Gen† [47] Show-o [59] Janus Pro [11] MetaQuery-XL† [38] UniWorld-V1 [26] BLIP3-o† [9] OminiGen2 [55] OminiGen2† [55] BAGEL [14] BAGEL† [14] Mogao [25]<br><br>|0.99 0.86 0.45 0.71 0.39 0.28 0.99 0.81 0.42 0.80 0.49 0.45<br><br>0.98 0.80 0.66 0.84 0.31 0.50<br>0.99 0.89 0.59 0.90 0.79 0.66<br><br><br>- - - - - -<br><br>0.99 0.93 0.79 0.89 0.49 0.70<br><br>- - - - - -<br><br>1.00 0.95 0.64 0.88 0.55 0.76 0.99 0.96 0.74 0.98 0.71 0.75 0.99 0.94 0.81 0.88 0.64 0.63<br><br><br>0.98 0.95 0.84 0.95 0.78 0.77<br>1.00 0.97 0.83 0.93 0.84 0.80<br><br><br>|0.61 0.66 0.68 0.80 0.80 0.80 0.84 0.80 0.86 0.82 0.88 0.89|
|EMMA EMMA†|1.00 0.98 0.83 0.96 0.83 0.85<br><br>1.00 0.99 0.87 0.98 0.86 0.87<br><br><br>|0.91 0.93|

- Table 5 | Quantitative evaluations of the text-to-image generation capacity on DPG-Bench [21].

###### Method Global Entity Attribute Relation Other Overall↑

|SD v1.5 [40] LUMINA-Next [69] SDXL [39] SD3-medium [16] FLUX.1-dev [22] SANA-1.5 [58] GPT Image 1 [High] [1] HiDream-I1-Full [6] Qwen-Image [54]|74.63 74.23 75.39 73.49 67.81<br><br>82.82 88.65 86.44 80.53 81.82<br>83.27 82.43 80.91 86.76 80.41<br><br><br>87.90 91.01 88.83 80.70 88.68 74.35 90.00 88.96 90.87 88.33<br><br>- - - - -<br><br>88.89 88.94 89.84 92.63 90.96 76.44 90.22 89.48 93.74 91.83 91.32 91.56 92.02 94.31 92.73<br><br><br>|63.18 74.63 74.65 84.08 83.84 84.7 85.15 85.89 88.32<br><br>|
|---|---|---|
|Show-o [59] Emu3-Gen [47] UniWorld-V1 [26] BLIP3-o [9] OmniGen2 [55] Janus Pro [11] Mogao [25] BAGEL [14]<br><br>|79.33 75.44 78.02 84.45 60.80<br><br>85.21 86.68 86.84 90.22 83.15 83.64 88.39 88.44 89.27 87.22<br><br>- - - - 88.81 88.83 90.18 89.37 90.27<br><br>86.90 88.90 89.40 89.32 89.48 82.37 90.03 88.26 93.18 85.40 88.94 90.37 91.29 90.82 88.67<br>|67.27 80.60 81.38 81.60 83.57 84.19 84.33 85.07|
|EMMA|91.24 91.71 90.59 92.23 90.02<br><br>|85.63|

GEdit relies on a vision-language model to assess editing results, which causes the failure to evaluate subject consistency. Although many recent methods [48, 49] incorporate the GPTImage-Edit-1.5M dataset [48] (generated via ChatGPT [1]) to significantly boost GEdit scores, we observed that such data severely disrupt subject consistency. As a result, we exclude this dataset from our final training corpus. This observation also highlights the need for more accurate evaluation criteria for image editing, which is left for future work.

###### 5.4. Emergence of Capabilities

During the evaluation of EMMA, we have observed several interesting phenomena: 1) without incorporating Chinese T2I generation and editing data during training, EMMA has the ability to directly support T2I generation and editing based on Chinese instructions, as shown in

- Figure 5. This capability is likely attributed to the inclusion of Chinese data in multimodal understanding datasets, which makes the understanding branch of EMMA capable of handling Chinese instructions. 2) while trained solely on single-instruction editing data, EMMA is still capable of performing the editing followed by complex instructions, as illustrated in Figure 5. The emergence of this ability is probably due to the multimodal chain-of-thought data, which enables the unified model to comprehend complex instructions and successfully execute the corresponding editing tasks.
- 6. Conclusion

In this work, we propose EMMA, an efficient multimodal unified architecture capable of performing multimodal understanding, generation, and editing tasks. Specifically, EMMA significantly reduces the number of visual tokens through an efficiently compressed autoencoder

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Add a black T-shirt to the man Add a pair of colorful sunglasses for the woman Add a bow hair accessory to the smiling woman

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Replace the pen with a fruit knife Replace the crab with a frog Change the background to a snow-capped mountain Change the weather into rainy weather

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Remove the man Remove the stethoscope from the doctor's neck Remove the text

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

去掉耳机，将头发换成蓝色，带上墨镜 Take off the headphones, change the hair to blue and put on sunglasses

将男人的衣服换成白色毛衣，桌子上加一杯水，头发换成黄色 Change the man's clothes to a white sweater, add a glass of water to the table, and change his hair to yellow

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Replace the glasses with sunglasses, add a hat, and change the clothes to a black dress

Remove the hat, change the clothes to purple, and change the background to the beach

Figure 5 | Showcase of EMMA’s image editing capability. EMMA effectively adheres to editing instructions while maintaining the character consistency. Furthermore, it natively supports both Chinese editing instructions and complex instructions. It is worth emphasizing that almost all editing training data used for EMMA consisted solely of single English instructions. The above results indicate that EMMA shows strong generalization capabilities, enabling it to handle complex multimodal tasks effectively.

- Table 6 | Comparisons between recent approaches on GEdit-Bench-EN [28]. The methods trained with the GPT-Image-Edit-1.5M [48] are not included.

GEdit-Bench-EN Semantic Consistency↑ Perceptual Quality↑ Overall Score ↑

Method

|Instruct-Pix2Pix [5] AnyEdit [63] MagicBrush [67] OmniGen [57] FLUX.1 Kontext [Pro] [22] Step1X-Edit [28] Qwen-Image [54]<br><br>|3.58 5.49 3.68<br><br>3.18 5.82 3.21<br>4.68 5.66 4.52<br>5.96 5.89 5.06 7.02 7.60 6.56<br><br><br>7.09 6.76 6.70<br>8.00 7.86 7.56<br><br><br>|
|---|---|
|UniWorld-V1 [26] OmniGen2 [55] BAGEL [14]|4.93 7.43 4.85 7.16 6.77 6.41 7.36 6.83 6.52<br><br>|
|EMMA|7.12 6.85 6.53|

and a channel-wise token concatenation mechanism. Additionally, it employs a mixture-ofexperts strategy in the visual understanding encoder to enhance visual perception. A sharedand-decoupled unified architecture is designed to mutually benefit various multimodal tasks while meeting their distinct modeling requirements. Experimental results demonstrate that EMMA outperforms current unified architectures across multiple benchmarks in multimodal understanding, generation, and editing tasks.

#### References

- [1] 2025. URL https://openai.com/zh-Hans-CN/index/introducing-4o-image-g eneration/.
- [2] 2025. URL https://github.com/QwenLM/Qwen3-VL.
- [3] X. An, Y. Xie, K. Yang, W. Zhang, X. Zhao, Z. Cheng, Y. Wang, S. Xu, C. Chen, C. Wu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025.

- [4] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [5] T. Brooks, A. Holynski, and A. A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.

- [6] Q. Cai, J. Chen, Y. Chen, Y. Li, F. Long, Y. Pan, Z. Qiu, Y. Zhang, F. Gao, P. Xu, et al. Hidreami1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025.

- [7] J. Chen, H. Cai, J. Chen, E. Xie, S. Yang, H. Tang, M. Li, Y. Lu, and S. Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024.

- [8] J. Chen, H. Cai, J. Chen, E. Xie, S. Yang, H. Tang, M. Li, Y. Lu, and S. Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024.

- [9] J. Chen, Z. Xu, X. Pan, Y. Hu, C. Qin, T. Goldstein, L. Huang, T. Zhou, S. Xie, S. Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025.

- [10] L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao, D. Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024.

- [11] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling, 2025.
- [12] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 67(12):220101, 2024.

- [13] Y. Cui, H. Chen, H. Deng, X. Huang, X. Li, J. Liu, Y. Liu, Z. Luo, J. Wang, W. Wang, Y. Wang, C. Wang, F. Zhang, Y. Zhao, T. Pan, X. Li, Z. Hao, W. Ma, Z. Chen, Y. Ao, T. Huang, Z. Wang, and X. Wang. Emu3.5: Native multimodal models are world learners, 2025. URL https://arxiv.org/abs/2510.26583.
- [14] C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, G. Shi, and H. Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

- [15] H. Duan, J. Yang, Y. Qiao, X. Fang, L. Chen, Y. Liu, X. Dong, Y. Zang, P. Zhang, J. Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11198– 11201, 2024.

- [16] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

- [17] D. Ghosh, H. Hajishirzi, and L. Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

- [18] X. He, L. Wei, L. Xie, and Q. Tian. Incorporating visual experts to resolve the information loss in multimodal large language models. arXiv preprint arXiv:2401.03105, 2024.

- [19] X. He, X. Han, L. Wei, L. Xie, and Q. Tian. Mixpert: Mitigating multimodal learning conflicts with efficient mixture-of-vision-experts. arXiv preprint arXiv:2505.24541, 2025.

- [20] X. He, X. Han, L. Wei, L. Xie, and Q. Tian. Mixpert: Mitigating multimodal learning conflicts with efficient mixture-of-vision-experts. arXiv preprint arXiv:2505.24541, 2025.

- [21] X. Hu, R. Wang, Y. Fang, B. Fu, P. Cheng, and G. Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

- [22] B. F. Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [23] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

- [24] Z. Li, H. Li, Y. Shi, A. B. Farimani, Y. Kluger, L. Yang, and P. Wang. Dual diffusion for unified image generation and understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2779–2790, 2025.

- [25] C. Liao, L. Liu, X. Wang, Z. Luo, X. Zhang, W. Zhao, J. Wu, L. Li, Z. Tian, and W. Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.

- [26] B. Lin, Z. Li, X. Cheng, Y. Niu, Y. Ye, X. He, S. Yuan, W. Yu, S. Wang, Y. Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.

- [27] H. Liu, C. Li, Y. Li, and Y. J. Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024.

- [28] S. Liu, Y. Han, P. Xing, F. Yin, R. Wang, W. Cheng, J. Liao, Y. Wang, H. Fu, C. Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

- [29] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.

- [30] Y. Liu, Z. Li, M. Huang, B. Yang, W. Yu, C. Li, X.-C. Yin, C.-L. Liu, L. Jin, and X. Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024.

- [31] P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

- [32] P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.

- [33] Y. Ma, X. Liu, X. Chen, W. Liu, C. Wu, Z. Wu, Z. Pan, Z. Xie, H. Zhang, X. Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7739–7751, 2025.

- [34] Z. Ma, L. Wei, S. Wang, S. Zhang, and Q. Tian. Deco: Frequency-decoupled pixel diffusion for end-to-end image generation. arXiv preprint arXiv:2511.19365, 2025.

- [35] A. Masry, X. L. Do, J. Q. Tan, S. Joty, and E. Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 2263–2279, 2022.

- [36] M. Mathew, D. Karatzas, and C. Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.

- [37] M. Mathew, V. Bagal, R. Tito, D. Karatzas, E. Valveny, and C. Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022.

- [38] X. Pan, S. N. Shukla, A. Singh, Z. Zhao, S. K. Mishra, J. Wang, Z. Xu, J. Chen, K. Li, F. JuefeiXu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

- [39] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

- [40] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022.

- [41] C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta, T. Coombes, J. Jitsev, and A. Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million imagetext pairs. arXiv preprint arXiv:2111.02114, 2021.

- [42] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.

- [43] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

- [44] M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans, L. Beyer, Y. Xia, B. Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

- [45] C. Wang, G. Lu, J. Yang, R. Huang, J. Han, L. Hou, W. Zhang, and H. Xu. Illume: Illuminating your llms to see, draw, and self-enhance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21612–21622, 2025.

- [46] W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

- [47] X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

- [48] Y. Wang, S. Yang, B. Zhao, L. Zhang, Q. Liu, Y. Zhou, and C. Xie. Gpt-image-edit-1.5m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033, 2025.

- [49] Z. Wang, Z. Chen, C. Gou, F. Li, C. Deng, D. Zhu, K. Li, W. Yu, H. Tu, H. Fan, et al. Lightbagel: A light-weighted, double fusion framework for unified multimodal understanding and generation. arXiv preprint arXiv:2510.22946, 2025.

- [50] C. Wei, Z. Xiong, W. Ren, X. Du, G. Zhang, and W. Chen. Omniedit: building image editing generalist models through specialist supervision. In ICLR, 2025.

- [51] L. Wei, L. Xie, W. Zhou, H. Li, and Q. Tian. Mvp: Multimodality-guided visual pre-training. In European conference on computer vision, pages 337–353. Springer, 2022.

- [52] L. Wiedmann, O. Zohar, A. Mahla, X. Wang, R. Li, T. Frere, L. von Werra, A. R. Gosthipaty, and A. Marafioti. Finevision: Open data is all you need. arXiv preprint arXiv:2510.17269, 2025.

- [53] C. Wu, X. Chen, Z. Wu, Y. Ma, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, C. Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024.

- [54] C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S.-m. Yin, S. Bai, X. Xu, Y. Chen, et al. Qwenimage technical report. arXiv preprint arXiv:2508.02324, 2025.

- [55] C. Wu, P. Zheng, R. Yan, S. Xiao, X. Luo, Y. Wang, W. Li, X. Jiang, Y. Liu, J. Zhou, Z. Liu, Z. Xia, C. Li, H. Deng, J. Wang, K. Luo, B. Zhang, D. Lian, X. Wang, Z. Wang, T. Huang, and Z. Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.

- [56] Y. Wu, Z. Zhang, J. Chen, H. Tang, D. Li, Y. Fang, L. Zhu, E. Xie, H. Yin, L. Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.

- [57] S. Xiao, Y. Wang, J. Zhou, H. Yuan, X. Xing, R. Yan, C. Li, S. Wang, T. Huang, and Z. Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13294–13304, 2025.

- [58] E. Xie, J. Chen, Y. Zhao, J. Yu, L. Zhu, C. Wu, Y. Lin, Z. Zhang, M. Li, J. Chen, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025.

- [59] J. Xie, W. Mao, Z. Bai, D. J. Zhang, W. Wang, K. Q. Lin, Y. Gu, Z. Chen, Z. Yang, and M. Z. Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

- [60] J. Xie, Z. Yang, and M. Z. Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.

- [61] R. Xie, C. Du, P. Song, and C. Liu. Muse-vl: Modeling unified vlm through semantic discrete encoding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 24135–24146, 2025.

- [62] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

- [63] Q. Yu, W. Chow, Z. Yue, K. Pan, Y. Wu, X. Wan, J. Li, S. Tang, H. Zhang, and Y. Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26125–26135, 2025.

- [64] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

- [65] X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

- [66] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pretraining. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

- [67] K. Zhang, L. Mo, W. Chen, H. Sun, and Y. Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.

- [68] C. Zhou, L. Yu, A. Babu, K. Tirumala, M. Yasunaga, L. Shamis, J. Kahn, X. Ma, L. Zettlemoyer, and O. Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.

- [69] L. Zhuo, R. Du, H. Xiao, Y. Li, D. Liu, R. Huang, W. Liu, X. Zhu, F.-Y. Wang, Z. Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. Advances in Neural Information Processing Systems, 37:131278–131315, 2024.

