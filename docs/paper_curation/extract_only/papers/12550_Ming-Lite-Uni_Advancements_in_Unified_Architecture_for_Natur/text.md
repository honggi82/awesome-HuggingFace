arXiv:2505.02471v3[cs.CV]13Jun2025

Ming-Lite-Uni: Advancements in Unified Architecture for Natural Multimodal Interaction

Inclusion AI, Ant Group∗

∗See Contributions section (Sec. 5) for full author list.

We introduce Ming-Lite-Uni, an open-source multimodal framework featuring a newly designed unified visual generator and a native multimodal autoregressive model tailored for unifying vision and language. Specifically, this project provides an open-source implementation of the integrated MetaQueries and M2-omni framework, while introducing the novel multi-scale learnable tokens and multi-scale representation alignment strategy. By leveraging a fixed MLLM and a learnable diffusion model, Ming-Lite-Uni enables native multimodal AR models to perform both text-toimage generation and instruction based image editing tasks, expanding their capabilities beyond pure visual understanding. Our experimental results demonstrate the strong performance of Ming-LiteUni and illustrate the impressive fluid nature of its interactive process. All code and model weights are open-sourced to foster further exploration within the community. Notably, this work aligns with concurrent multimodal AI milestones - such as ChatGPT-4o with native image generation updated in March 25, 2025 - underscoring the broader significance of unified models like Ming-Lite-Uni on the path toward AGI. Ming-Lite-Uni is in alpha stage and will soon be further refined.

[Figure 1]

Date: May 05, 2025 Code: https://github.com/inclusionAI/Ming/tree/Ming-Lite-Omni-Preview/Ming-unify

# 1 Introduction

The release of GPT-4o OpenAI (2025) in March 2025, which introduced native image generation, has made the impact of unified models Tong et al. (2024); Team (2024); Pan et al. (2025) increasingly apparent. Users can now perform complex visual tasks such as image editing Gong et al. (2024a); Feng et al. (2024), multi-view synthesis Shi et al. (2023), style transfer Li et al. (2024), and even 3D rendering Mildenhall et al. (2021) purely through natural language conversations. These capabilities, once dependent on specialized models Gong et al. (2024b); Wang et al. (2024b); Huang et al. (2024a); Tan et al. (2024b), can now be achieved fluently and with high quality, marking a major advance in perceived intelligence. In this context, our work pursues two primary objectives: (1) to demonstrate the strong potential of a unified auto-regressive multimodal model built upon the multi-scale learnable tokens with fine-tuned diffusion models, and (2) to accelerate community engagement by open-sourcing an alpha-version of our code and model weights.

A major challenge in unifying multimodal understanding and generation lies in the inconsistency of visual feature spaces. Recent models such as TokenFlow Qu et al. (2024) and Janus Wu et al. (2024a) integrate diffusion-based decoders with token-based understanding models, achieving strong image generation but often at the cost of precise understanding. These methods prioritize pixel fidelity, leading to a mismatch between visual features and semantic understanding. Ming-Lite-Uni adopts a lightweight bridging approach, serving as an open-source implementation and improvement over the integrated MetaQueries Pan et al. (2025) and M2-omni Guo et al. (2025) framework. Unlike

an apple placed on the table

a photo of a person a photo of a toaster a photo of a stop sign a photo of a dog a photo of a cake a potted plant

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

ImageeditingStyletransferImagegeneration

make the action of the horse to galloping add a teddy bear sitting on the bed make the sheep wear tiny sunglasses remove two of the flowers in the image

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Abstract Expressionism Hayao Miyazaki's style Kawaii Adorable 3D

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

D E M O

[Figure 25]

Make the sheep wear tiny sunglasses. The wool color turns green.

[Figure 26]

OK, here is the result.

###### ……

- Figure 1 The output results and multimodal interactive demos of Ming-Lite-Uni. Our model supports basic multimodal chatting, text-to-image generation, image editing, and image style transfer based on textual instructions.

previous works, which focused on understanding performance and model structure, Ming-Lite-Uni fixes the MLLM and fine-tunes the diffusion model through the newly designed multi-scale learnable tokens, multi-scale representation alignment mechanism, and connector. In text-to-image generation and instruction based image editing tasks, autoregressive models excel at semantic understanding, providing robust contextual guidance. Meanwhile, finetuned diffusion models leverage multi-scale learnable tokens and tailored loss functions to achieve high-fidelity, fine-grained image synthesis.

A second core component of Ming-Lite-Uni focuses on enhancing the visual generation capacity of its auto-regressive backbone, which is achieved by integrating a FlowMatching loss Esser et al. (2024a) directly into the separate diffusion model Tan et al. (2024a); Shi et al. (2024); Ma et al.

- (2024). Such an approach allows generation quality to improve in tandem with end-to-end training, enabling Ming-Lite-Uni to effectively optimize the diffusion model while keeping the MLLM frozen. Additional efforts such as new multimodal AR architecture, scaling-up rule, pre-training and post-training strategy of the auto-regressive module help balance the model’s visual and language capacity, ensuring that Ming-Lite-Uni remains a unified prototype with high potential and no compromises between modalities and tasks. Further improvements to the auto-regressive component are already underway and will be included in the next release.

Finally, we curate a multimodal dataset covering tasks such as image editing Tan et al. (2024c) to train Ming-Lite-Uni. Despite limited resources, Ming-Lite-Uni exhibits strong control fluency and contextual understanding, handling diverse fine-grained tasks like image-to-text and text-to-image QA through natural language dialogue. All code and weights have been open-sourced, with a full experimental evaluation to follow in the next release.

# 2 Approach

Ming-Lite-Uni compresses image representations into a sequence of continuous tokens, which are combined with discrete text tokens and processed by a scaled auto-regressive Transformers Bai et al.

- (2025); Lu et al. (2024a) for end-to-end multimodal context learning. The generation capability is provided by an externally trainable diffusion model (e.g., SANA Xie et al. (2024a)), conditioned on tokens produced by auto-regressive Transformers. In this Section, we first introduce our newly designed multi-scale learnable tokens and the multi-scale representation alignment strategy in Sec. 2.1, followed by our previous designed Native Multimodal Auto-regressive Model in Sec. 2.2.

## 2.1 Ming-Lite-Uni with Multi-Scale Learnable Tokens

We propose Multi-Scale Learnable Query Tokens to enhance multi-resolution image understanding and generation within a unified framework. Additionally, we introduce a Multi-Scale Representation Alignment strategy to align intermediate and output representations across different scales. The framework is shown in Fig. 2.

Training DenoisingObjective Image@256px Image@512px

Inference Image@512px

Image@128px

Shared Shared

Multi-scale Representation Alignment

DiT Blocks DiT Blocks DiT Blocks

🔥 🔥 🔥

[Figure 27]

[Figure 28]

[Figure 29]

Response

DiT Block DiT Block DiT Block

🔥

Connector

[Figure 30]

❄

[Figure 31]

Multi-modal Large Language Model

Image tokens Text tokens

Noisy inputs

Multi-scale queries

- Figure 2 The framework of Ming-Lite-Uni. Our model fixes the MLLM and fine-tunes the diffusion model through the newly designed multi-scale learnable tokens, multi-scale representation alignment, and connector.

Multi-Scale Learnable Tokens Construction Given an input image x, we define a set of scales S = {s1,s2,...,sK}, where each sk corresponds to a spatial resolution, e.g., sk ∈ {4×4,8×8,16×16}. Each scale sk is associated with a dedicated set of learnable query tokens Qs

sk×d, where Ns

k ∈ RN

k

is the number of tokens for scale sk, and d is the hidden dimension size. Formally, we initialize the multi-scale query tokens as:

= Learnable Parameters. (1) Each Qs

Q = {Qs

K}, Qs

,Qs

,...,Qs

1

2

k

is designed to capture information at different granularity levels: (1) Low resolution (4 × 4), which captures global layout and color distribution. (2) Medium resolution (8 × 8), which models major objects and mid-level structures. (3) High resolution (16 × 16), which encodes fine textures and detailed patterns.

k

Multi-Scale Learnable Tokens Fusion and Processing To preserve scale-specific semantics, we introduce explicit scale boundary markers. For each scale sk, we prepend and append special tokens:

], (2) where STARTs

Inputs

= [STARTs

,ENDs

,Qs

k

k

k

k

are learnable tokens indicating the boundaries of sk. Each query token also receives a dedicated positional encoding. Let Ps

and ENDs

k

k

sk×d denote the positional grid encoding for scale sk, constructed according to the spatial grid associated with that resolution. The complete multi-scale token sequence fed into the transformer encoder is thus:

k ∈ RN

Zinput = Concat Inputs

,Inputs

,...,Inputs

+ Concat(Ps

). (3)

## ,Ps

,...,Ps

1

2

K

1

2

K

The transformer encoder fθ(·) processes Zinput to produce the hidden representations:

H = fθ(Zinput). (4)

Multi-Scale Representation Alignment To unify the feature representations used for both image understanding and generation, we introduce a simple yet effective Multi-Scale Representation Alignment strategy (i.e., scale wised consistency loss). Specifically, we align the intermediate hidden states from the DiT backbone with the final semantic representations by minimizing the mean squared error between them. The alignment loss encourages consistency between hierarchical representations and final outputs through native-resolution optimization which directly enhances the high-res reconstruction quality (>2dB PSNR) and boosts GenEval by 1.5%.

## 2.2 Native Multimodal Auto-regressive Model (The AR part of Ming-Lite-Uni)

###### MultiModal Step Balance Strategy

MultiModal Multi-Task Balanced Training Strategy

Text 𝑎 , 𝑥 + 𝑏 , 

Image-Text 𝑎 , 𝑥 + 𝑏 , 

[Figure 32]

Update

[Figure 33]

Text Update

[Figure 34]

[Figure 35]

F1

Text Image Video Audio

|𝐿|
|---|

|𝐿|
|---|

× 𝑤 × 𝑤 × 𝑤 × 𝑤

× 𝑤 ,  × 𝑤 ,  × 𝑤 ,  × 𝑤 , 

Modality weight

+ + +

Forward

[Figure 36]

[Figure 37]

- F2
- F3
- F4

Image Video Audio

|𝐿|
|---|

|𝐿|
|---|

M2omni

M2omni

Audio-Text 𝑎 , 𝑥 + 𝑏 , 

Video-Text 𝑎 , 𝑥 + 𝑏 , 

[Figure 38]

[Figure 39]

|𝐿|
|---|

|𝐿|
|---|

Backward

[Figure 40]

[Figure 41]

|𝐿|
|---|

|𝐿|
|---|

Backward once

Historical validation losses

Fn: n-th Forward

- Figure 3 The AR part of Ming-Lite-Uni. Our model reuses the M2-omni MLLM as a frozen token prediction module, retaining only its text and image branches. The pretraining procedure and dataset of the AR model are consistent with our previous work, please refer to Guo et al. (2025) for details.

Input Output

Models

Text Img Video Audio Text Img Audio VITA-1.5 Mini-omni2 Baichuanomni MiniCPM-o GPT-4o M2-omni

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Vision Encoder We utilize a NaViT Dehghani et al. (2024) as the vision encoder, capable of processing images of arbitrary resolution. To reduce the length of visual tokens, we concatenate adjacent 2 × 2 tokens into a single token and use an MLP to reduce the dimension to the original dimension, thereby downsampling the visual representation.

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

M2-omni LLM As shown in Fig. 3, the M2-omni LLM integrates the multimodal information and outputs the embedding for multimodal understanding. Our M2-omni LLM is initialized with pre-trained weights from the Llama3 Touvron et al. (2023); Dubey et al. (2024) series, specifically Llama3.1-8B or Llama3.3-70B. To facilitate unified positional encoding across textual, image, video, and audio modalities, and to enable the model to generalize to longer sequences during inference, we substitute the original 1D-RoPE Su et al. (2024) in Llama with M-RoPE Wang et al. (2024a).

# 3 Training Data

The training dataset consists of two parts: basic image-text pairs and image generation data. The basic image-text pairs are primarily sourced from public datasets (e.g., Laion Schuhmann et al.

- (2022), COYO Byeon et al. (2022)) as well as additional images and corresponding captions collected and filtered from the internet (e.g., Midjourney, Google Search). The image generation data mainly comes from commonly used training datasets for various downstream image style transfer tasks or image editing tasks.

## 3.1 Basic Image-Text Pairs

We use LAION-5B (1,555,342,102 samples), Zero Guo et al. (2025) (151,766,788 samples), and COYO (61,915,700 samples) as the basic pretrain image-text datasets. Additionally, we collect high-quality image-caption pairs from Wukong Gu et al. (2022), Midjourney, and web search engines, commonly used for diffusion model training. We apply aspect ratio (≤ 2.5), watermark detection (≤ 0.5), and CLIP alignment (≥ 0.45) thresholds for filtering. After preprocessing, the final sample counts are 34,826,982 (Wukong), 5,421,512 (Midjourney), and 440,951,902 (others). Fig. 4 shows sample visualizations. Finally, our tarining dataset also includes a small amount of aesthetic evaluation data collected from AVA Murray et al. (2012) (255,000 samples), TAD66K He et al. (2022) (66,000 samples), AesMMIT Huang et al. (2024b) (21,904 samples), and APDD Jin et al. (2024) (10,023 samples). Although limited in amount, these data help the model learn humandefined aesthetic standards, improving image generation quality and enhancing the model’s ability to assess visual aesthetics.

[Figure 84]

Figure 4 Basic image-text training pairs of Ming-Lite-Uni.

## 3.2 Image Generation Datasets

This part of training data includes InstructPix2Pix-clip-filtered Brooks et al. (2023), where each pair of edited images is generated 100 times, and the best examples are chosem based on CLIP metrics (Sec.3.1.2 in InstructPix2Pix), SEED-Data-Edit-part2/3 Ge et al. (2024a), excluding part1 due to the poor visual quality, Ultra-edit Zhao et al. (2024), SynCD Kumari et al. (2025), Subjects200k Tan et al. (2024c), HQ-edit Hui et al. (2024), and MagicBrush Zhang et al. (2023). It consists of 5,008,795 samples. Tab. 1 reports the number of sequences with two or more consecutive edits. Representative examples are shown in Fig. 5.

- Table 1 Statistics of multi-round editing data. We report the number of samples with 2, 3, 4, and 5 or more consecutive edits. Dataset 2-step Edit 3-step Edit 4-step Edit 5-step Edit and above MagicBrush Zhang et al. (2023) 1,151 1,572 - SynCD Kumari et al. (2025) 25,438 - - SEED-Data-Edit-part3 Ge et al. (2024a) 472 7,453 8,783 4,669

In addition, our training data includes publicly available datasets commonly used for style transfer tasks, along with synthesized data generated using style prompts. The style data comprises a high-quality subset of WikiArt, covering 27 painting styles such as Impressionism, Realism, and Expressionism, and the StyleBooth Han et al. (2024), featuring 67 styles including cartoon and 3D, each with 717 image pairs. These two datasets contain 81,444 and 80,922 samples, respectively. Representative examples are shown in Fig. 6.

[Figure 85]

Figure 5 Ming-Lite-Uni image editing data examples in the training set.

[Figure 86]

Figure 6 Ming-Lite-Uni image style transfer data examples in the training set.

# 4 Benchmark Evaluations

We conduct separate quantitative evaluations of Ming-Lite-Uni on multimodal understanding and text-to-image generation using public benchmarks. For multimodal understanding, we compare against traditional models that take images and text as input and output text, as well as against recent models with visual generative capabilities. For multimodal generation, we evaluate text-to-image performance on GenEval Ghosh et al. (2024).

## 4.1 Multimodal Understanding

To evaluate the effectiveness of our Ming-Lite-Uni in image-text understanding, we benchmark it against state-of-the-art MLLMs on 7 different multimodal benchmarks, including complex VQA (MMB Liu et al. (2025), MMS Chen et al. (2024a), MMMU Yue et al. (2024), AI2D Kembhavi et al. (2016), and MM-Vet Yu et al. (2024)), multimodal reasoning (MathV Lu et al. (2024b)), and hallucination evaluation (Hall Guan et al. (2024)). Tab. 2 shows the overall results. Our model achieves top-tier performance on most benchmarks, surpassing closed-source models like GPT-4o and Gemini-1.5-Pro. Furthermore, our model exhibits competitive performance among models of similar size, showcasing its robust capabilities in image-text understanding tasks.

Table 2 Quantitative results on parts of OpenCompass Contributors (2023) multimodal leaderboard. ‡ denotes closed-source models. Hall denotes HallusionBench. “Und.” and “Gen.” denote “understanding” and “generation”, respectively.

Type Model Avg. MMB↑ MMS↑ MMMU↑ MathV↑ Hall↑ AI2D↑ MM-Vet↑

Und. Only LLaVA-72B Xie et al. (2024b) 68.0 84.5 65.8 56.6 68.4 47.9 86.2 60.6 Qwen2-VL-72B Bai et al. (2023) 74.8 85.9 68.6 64.3 69.7 58.7 88.3 73.9 Qwen2.5-VL-7B Bai et al. (2025) 76.2 87.8 71.1 67.9 70.8 58.8 88.2 76.7 Emu3-Chat Wang et al. (2024c) - 58.5 - 31.6 - - - 37.2 InternVL2.5-8B Chen et al. (2024b) 70.3 82 65.2 54.8 67.9 51.7 84.5 68.1 InternVL2.5-38B Chen et al. (2024b) 73.5 85.4 68.5 64.6 72.4 57.9 87.6 67.2 InternVL2.5-78B Chen et al. (2024b) 75.2 87.5 69.5 70 71.4 57.4 89.1 71.8 DeepSeek-VL2 Wu et al. (2024c) 66.4 81.2 61.0 50.7 59.4 51.5 84.5 60.0 GPT-4o-20241120‡ OpenAI (2024) 72.0 84.3 65.1 70.7 59.9 56.2 84.9 74.5 Step-1o‡ StepFun (2025) 77.7 87.3 69.3 69.9 74.7 55.8 89.1 82.8

Und. and Gen. DreamLLM Dong et al. (2023) - - - - - - - 36.6 MetaMorph Tong et al. (2024) - 75.2 - - - - - Show-o-256 Xie et al. (2024b) - - - 25.1 - - - Show-o-512 Xie et al. (2024b) - - - 26.7 - - - TokenFlow-XL Qu et al. (2024) - 68.9 - 38.7 - - - 40.7 Chameleon Team (2024) - - - 22.4 - - - 8.3 Janus Wu et al. (2024a) - 69.4 - 30.5 - - - 34.3 Janus-Pro-7B Chen et al. (2025) - 79.2 - 41.0 - - - 50.0 Ours (Ming-Lite-Uni) 69.7 80.7 60.5 51.2 68.3 51.8 84.5 72.3

## 4.2 Text-to-Image Generation

We report visual generation performance on GenEval Ghosh et al. (2024). As shown in Tab. 3, our Ming-Lite-Uni obtains 0.62 overall accuracy on GenEval, which outperforms all the other unified or generation-only methods, e.g., MetaQueries Pan et al. (2025) (0.61), DALL-E 3 Betker et al.

- (2023) (0.67), and SDXL Podell et al. (2023) (0.55). These results demonstrate that our model matches the performance of state-of-the-art diffusion models in generating single-subject images.

#### Table 3 Evaluation of text-to-image generation ability on GenEval benchmark Ghosh et al. (2024). “Und.” and “Gen.” denote “understanding” and “generation”, respectively.

##### Type Method Single Obj.↑ Two Obj.↑ Counting↑ Colors↑ Position↑ Color Attri.↑ Overall↑

LlamaGen Sun et al. (2024) 0.71 0.34 0.21 0.58 0.07 0.04 0.32 LDM Rombach et al. (2022) 0.92 0.29 0.23 0.70 0.02 0.05 0.37 SDv1.5 Rombach et al. (2022) 0.97 0.38 0.35 0.76 0.04 0.06 0.43 PixArt-α Chen et al. (2023) 0.98 0.50 0.44 0.80 0.08 0.07 0.48 SDv2.1 Rombach et al. (2022) 0.98 0.51 0.44 0.85 0.07 0.17 0.50

Gen. Only

- DALL-E 2 Ramesh et al. (2022) 0.94 0.66 0.49 0.77 0.10 0.19 0.52 Emu3-Gen Wang et al. (2024c) 0.98 0.71 0.34 0.81 0.17 0.21 0.54 SDXL Podell et al. (2023) 0.98 0.74 0.39 0.85 0.15 0.23 0.55

- DALL-E 3 Betker et al. (2023) 0.96 0.87 0.47 0.83 0.43 0.45 0.67 SD3-Medium Esser et al. (2024b) 0.99 0.94 0.72 0.89 0.33 0.60 0.74

Chameleon Team (2024) - - - - - - 0.39 LWM Liu et al. (2024) 0.93 0.41 0.46 0.79 0.09 0.15 0.47 SEED-X Ge et al. (2024b) 0.97 0.58 0.26 0.80 0.19 0.14 0.49 Show-o Xie et al. (2024b) 0.95 0.52 0.49 0.82 0.11 0.28 0.53 TokenFlow-XL Liu et al. (2024) 0.95 0.60 0.41 0.81 0.16 0.24 0.55 Janus Wu et al. (2024b) 0.97 0.68 0.30 0.84 0.46 0.42 0.61 Janus-Pro-1B Chen et al. (2025) 0.98 0.82 0.51 0.89 0.65 0.56 0.73 MetaQueries Pan et al. (2025) - - - - - - 0.61 Ours (Ming-Lite-Uni) 0.99 0.76 0.53 0.87 0.26 0.30 0.62

Und. and Gen.

## 4.3 Instruction Based Image Editing

As shown in Fig. 7, we conducted a qualitative analysis on a wider range of interactive image editing tasks, including style transfer and object addition, deletion, and modification.

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Origin

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Edited

Change the action of the skier to jumping

Change the action of cat to jumping

Change the cat to a squirrel

Change the color of the bird to tan

Make the dog sleeping

Change the sheep into gold

Let the image on a cartoon style

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Origin

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Edited

Remove the giraffes Replace the train to a bus

Alter the background to a beachside

Replace the elephant with a giraffe

Replace the pizza with a cake

Include a rainbow in the sky

Alter the style of the image to sketch

Figure 7 Instruction based image editing results outputted by Ming-Lite-Uni.

- 5 Contributors Authors are listed alphabetically by the first name.

Biao Gong Cheng Zou Dandan Zheng Hu Yu Jingdong Chen Jianxin Sun

Junbo Zhao Jun Zhou Kaixiang Ji Lixiang Ru Libin Wang Qingpei Guo

Rui Liu Weilong Chai Xinyu Xiao Ziyuan Huang

# References

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. https://arxiv.org/abs/2502.13923.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.

Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024a.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024b.

OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https://github.com/

open-compass/opencompass, 2023.

Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36, 2024.

Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499, 2023.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis, 2024. URL https://arxiv. org/abs/2403.03206, 2, 2024a.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024b. https://arxiv.org/abs/2403.03206.

Yutong Feng, Biao Gong, Di Chen, Yujun Shen, Yu Liu, and Jingren Zhou. Ranni: Taming text-to-image diffusion for accurate instruction following. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4744–4753, 2024.

Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007, 2024a.

Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024b.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36, 2024.

Biao Gong, Siteng Huang, Yutong Feng, Shiwei Zhang, Yuyuan Li, and Yu Liu. Check locate rectify: A training-free layout calibration system for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6624–6634, 2024a.

Biao Gong, Shuai Tan, Yutong Feng, Xiaoying Xie, Yuyuan Li, Chaochao Chen, Kecheng Zheng, Yujun Shen, and Deli Zhao. Uknow: A unified knowledge protocol with multimodal knowledge graph datasets for reasoning and vision-language pre-training. Advances in Neural Information Processing Systems, 37:9612–9633, 2024b.

Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Niu Minzhe, Xiaodan Liang, Lewei Yao, Runhui Huang, Wei Zhang, Xin Jiang, et al. Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark. Advances in Neural Information Processing Systems, 35:26418–26431, 2022.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14375–14385, June 2024.

Qingpei Guo, Kaiyou Song, Zipeng Feng, Ziping Ma, Qinglong Zhang, Sirui Gao, Xuzheng Yu, Yunxiao Sun, Tai-Wei Chang, Jingdong Chen, et al. M2-omni: Advancing omni-mllm for comprehensive modality support with competitive performance. arXiv preprint arXiv:2502.18778, 2025.

Zhen Han, Chaojie Mao, Zeyinzi Jiang, Yulin Pan, and Jingfeng Zhang. Stylebooth: Image style editing with multimodal instruction. arXiv preprint arXiv:2404.12154, 2024.

Shuai He, Yongchang Zhang, Rui Xie, Dongxiang Jiang, and Anlong Ming. Rethinking image aesthetics assessment: Models, datasets and benchmarks. IJCAI, 2022.

Siteng Huang, Biao Gong, Yutong Feng, Xi Chen, Yuqian Fu, Yu Liu, and Donglin Wang. Learning disentangled identifiers for action-customized text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7797–7806, 2024a.

Yipo Huang, Xiangfei Sheng, Zhichao Yang, Quan Yuan, Zhichao Duan, Pengfei Chen, Leida Li, Weisi Lin, and Guangming Shi. Aesexpert: Towards multi-modality foundation model for image aesthetics perception. In Proceedings of the 32nd ACM International Conference on Multimedia, MM ’24, pages 5911–5920, 2024b.

Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990, 2024.

Xin Jin, Qianqian Qiao, Yi Lu, Huaye Wang, Heng Huang, Shan Gao, Jianfei Liu, and Rui Li. Apddv2: Aesthetics of paintings and drawings dataset with artist labeled scores and comments, 2024. https://arxiv.org/abs/2411.08545.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images, 2016.

Nupur Kumari, Xi Yin, Jun-Yan Zhu, Ishan Misra, and Samaneh Azadi. Generating multi-image synthetic data for text-to-image customization. arXiv preprint arXiv:2502.01720, 2025.

Wen Li, Muyuan Fang, Cheng Zou, Biao Gong, Ruobing Zheng, Meng Wang, Jingdong Chen, and Ming Yang. Styletokenizer: Defining image style by a single instance for controlling diffusion models. In European Conference on Computer Vision, pages 110–126. Springer, 2024.

Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint arXiv:2402.08268, 2024.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2025.

Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024a.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024b.

Shuailei Ma, Kecheng Zheng, Ying Wei, Wei Wu, Fan Lu, Yifei Zhang, Chen-Wei Xie, Biao Gong, Jiapeng Zhu, and Yujun Shen. Learning visual generative priors without text. arXiv preprint arXiv:2412.07767, 2024.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.

Naila Murray, Luca Marchesotti, and Florent Perronnin. Ava: A large-scale database for aesthetic visual analysis. In 2012 IEEE

Conference on Computer Vision and Pattern Recognition, pages 2408–2415, 2012. doi: 10.1109/CVPR.2012.6247954. OpenAI. Gpt-4o system card, 2024. https://arxiv.org/abs/2410.21276. OpenAI. Introducing 4o image generation. https://openai.com/index/introducing-4o-image-generation/, 2025. Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen,

Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. arXiv preprint arXiv:2412.03069, 2024. Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022.

Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023.

Shuwei Shi, Biao Gong, Xi Chen, Dandan Zheng, Shuai Tan, Zizheng Yang, Yuyuan Li, Jingwen He, Kecheng Zheng, Jingdong Chen, et al. Motionstone: Decoupled motion intensity modulation with diffusion transformer for image-to-video generation. arXiv preprint arXiv:2412.05848, 2024.

StepFun. step-1o. https://www.stepfun.com/, 2025. Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary

position embedding. Neurocomputing, 568:127063, 2024. Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. Shuai Tan, Biao Gong, Yutong Feng, Kecheng Zheng, Dandan Zheng, Shuwei Shi, Yujun Shen, Jingdong Chen, and Ming Yang. Mimir: Improving video diffusion models for precise text understanding. arXiv preprint arXiv:2412.03085, 2024a. Shuai Tan, Biao Gong, Xiang Wang, Shiwei Zhang, Dandan Zheng, Ruobing Zheng, Kecheng Zheng, Jingdong Chen, and Ming Yang.

Animate-x: Universal character image animation with enhanced motion representation. arXiv preprint arXiv:2410.10306, 2024b. Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for

diffusion transformer. arXiv preprint arXiv:2411.15098, 2024c. Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie,

and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

Wen Wang, Qiuyu Wang, Kecheng Zheng, Hao Ouyang, Zhekai Chen, Biao Gong, Hao Chen, Yujun Shen, and Chunhua Shen. Framer: Interactive frame interpolation. arXiv preprint arXiv:2410.18978, 2024b.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024c.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, and Ping Luo. Janus: Decoupling visual encoding for unified multimodal understanding and generation, 2024a. https: //arxiv.org/abs/2410.13848.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024b.

Zhiyu Wu, Xiaokang Chen, Zizheng Pan, and et al. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding, 2024c. https://arxiv.org/abs/2412.10302.

Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024a.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024b.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In International conference on machine learning. PMLR, 2024.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.

Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37: 3058–3093, 2024.

