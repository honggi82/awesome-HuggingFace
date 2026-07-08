## Xmodel-VLM: A Simple Baseline for Multimodal Vision Language Model

Xu Wanting Liu Yang He Langping Huang Xucheng Jiang Ling XiaoduoAI {xuwanting,liuyangfoam,helangping}@xiaoduotech.com

# arXiv:2405.09215v3[cs.CV]20Jun2024

### Abstract

We introduce Xmodel-VLM, a cutting-edge multimodal vision language model. It is designed for efficient deployment on consumer GPU servers. Our work directly confronts a pivotal industry issue by grappling with the prohibitive service costs that hinder the broad adoption of large-scale multimodal systems. Through rigorous training, we have developed a 1B-scale language model from the ground up, employing the LLaVA paradigm for modal alignment. The result, which we call Xmodel-VLM, is a lightweight yet powerful multimodal vision language model. Extensive testing across numerous classic multimodal benchmarks has revealed that despite its smaller size and faster execution, Xmodel-VLM delivers performance comparable to that of larger models. Our model checkpoints and code are publicly available on GitHub at http s://github.com/XiaoduoAILab/XmodelVLM.

### 1. Introduction

In recent years, the integration of natural language processing (NLP) and computer vision has spurred significant innovation and breakthroughs within the field of multimodal learning. Notably, advanced visual language models (VLMs) such as GPT-4V [1] and Gemini [39] leverage the synergy between text and visual data to achieve advanced understanding and interaction with the world. With their powerful capabilities, they have demonstrated excellent performance in a variety of downstream visual language tasks. At the same time, the rapid development of open source large language models (LLMs) has broken the bottleneck of limited access to technical details of proprietary models, and also laid a solid foundation for the progress of visual language models.

The current mainstream open source visual language models often show excellent performance, which usually relies on the language model components with no less than 7B parameters behind them. For example, Openflamingo [2] achieves the best results when using the language model of MPT-7B [40]. BLIP-2 [19] uses the 2.7B and 6.7B ver-

sions of the OPT [44] series as well as FlanT5XL (3B) and FlanT5XXL (11B) [10]for exploration, and shows the best performance when the language model is configured as FlanT5XXL (11B) in its architecture. LLaVA-1.5 [25] works best when using the language model of Vicuna13B [8]. However, the escalating complexity and resource intensity of these large visual language models have cast a spotlight on one of the primary obstacles hindering their widespread adoption: the considerable operational costs.

In this context, research on small-scale visual language models has become increasingly popular. Existing studies have shown that small-scale visual language models can still perform as well as larger-scale models, such as LLaVAPhi [48], which combines the open source multi-modal model LLaVA-1.5 and the open source small language model Phi-2(2.7B) [21] to improve multi-modal model resource efficiency. Tiny-LLaVA [45] demonstrates that with better training recipes and data quality, smaller-scale Large Multimodal Models (LMMs) can achieve comparable performance to larger LMMs. MobileVLM [9] is designed for efficient operation on mobile devices and includes MobileLLaMA(2.7B), CLIP ViT-L/14 [32], and lightweight downsample projector (LDP). Table 1 provides a meticulous comparative overview of the architectures, cross-modal design strategies, and training data sources employed in each model.

Despite the encouraging advancements made in the realm of visual language models, the pursuit of a genuinely optimal harmony between performance and efficiency remains an active and ongoing challenge. To this end, we trained a 1B scale language model (Xmodel-LM1B [41]) from scratch, and guided by the cross-modal alignment principle advocated by the LLaVA paradigm, we have delved deeply into various facets of model architecture and training, including the selection of image encoders, the design of image-text connectors, and the exploitation of diverse datasets, all with the aim of pushing the boundaries of what is achievable with a smaller-scale model.

In this paper, we present Xmodel-VLM, an innovative vision-language assistant driven by a compact language model. Our contributions are outlined as follows:

Model Vision Encoder Language Model Cross-modality Design Multimodal Training Corpora CLIP [32] ViT, ResNet Transformer Linear-Projection WebImageText [33] (400M image-text pairs) BLIP [20] ViT MED Cross-Attention COCO [24], VG [17], CC3M [35], CC12M [4], LAION [34] BLIP-2 [19] CLIP/Eva-CLIP ViT OPT, Flan-T5 Q-Former same as BLIP InstructBLIP [11] ViT-G/14 Flan-T5, Vicuna Q-Former w/ FC 13 held-in out of 26 public datasets Openflamingo [2] CLIP ViT-L/14 MPT/RedPajama Cross-Attention LAION, Multimodal C4 [47] LLaVA [26] CLIP ViT-L/14 Vicuna 7B/13B Linear-Projection filtered CC-595K from CC3M, LLaVA-Instruct-158K LLaVA-1.5 [25] CLIP ViT-L@336 Vicuna-7B/13B MLP a subset of InstructBLIP (1.2M) MiniGPT-4 [46] EVA-CLIP ViT-G/14 Vicuna-7B Q-Former LAION, CC, SBU [31] QWen-VL [3] Openclip ViT-bigG [16] Qwen-7B Cross-Attention multi-tasking datasets (Captioning, VQA, Grounding, etc.) ShareGPT4V [7] CLIP ViT-L/14@336 Vicuna-7B MLP ShareGPT4V (100K by GPT-4V, 1.2M by its learned model) LLaVA-Phi [48] CLIP ViT-L/14@336 Phi-2-2.7B MLP same as LLaVA-1.5 [25] Tiny-LLaVA [45] CLIP/SigLIP TinyLlama

MLP same as LLaVA-1.5 and ShareGPT4V

StableLM-2 Phi-2

MobileVLM [9] CLIP ViT-L/14@336 MobileLLaMA LDP same as LLaVA-1.5 [25] Mini-Gemini [23] CLIP/ConvNeXt [28] Gemma-2B

MLP Mini-Gemini-Pretrain, Mini-Gemini-Instruction

Vicuna-7B/13B Mixtral-8x7B Hermes-2-Yi-34B

Xmodel-VLM (ours) CLIP ViT-L/14@336 Xmodel-LM-1B XDP same as LLaVA-1.5 [25]

Table 1. Comparison of open-source VLM architectures and their training corpora.

- 1. We delve into the performance and capabilities of smaller Chinese and English language models, painstakingly trained on terabytes of data.
- 2. We conduct comprehensive ablation studies on the design of Xmodel-VLM, meticulously evaluating various design choices and their respective impacts.

These contributions not only shed light on the efficacy of compact language models but also offer insights into optimizing the design of vision-language models for enhanced performance and efficiency.

### 2. Related Work

#### 2.1. Vision Transformer

The Vision Transformer (ViT) represents a pioneering model that adopts a Transformer-like architecture applied to patches extracted from the image. Initially, the image is segmented into fixed-size patches, followed by linear embedding of each patch. Position embeddings are subsequently incorporated, and the resultant sequence of vectors is fed into a standard Transformer architecture, as depicted in Figure 1. This innovative approach marks a departure from conventional convolutional neural networks (CNNs), offering promising avenues for enhanced image understanding and representation learning.

#### 2.2. LLaVA

LLaVA represents a pioneering approach to constructing cost-effective universal multimodal assistants. It embod-

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

Figure 1. Vision Transformer

ies a novel end-to-end trained large-scale multimodal model that ingeniously integrates visual encoders with Vicuna [8] for comprehensive vision and language understanding. Inspired by the multimodal GPT-4 framework, LLaVA not only excels in conversational interactions but also achieves state-of-the-art accuracy in Science QA tasks.

LLaVA leverages ChatGPT/GPT-4 for multimodal instruction-following data collection, taking advantage of abundant image-pair data. Given an image Xv and its caption Xc, the model prompts GPT-4 to genenrate a series of questions aimed at instructing the assistant to describe the image content. Consequently, an image-text pair is expanded into its instruction-following version, as illustrated

below: Human : Xq Xv<STOP> Assistant : Xc<STOP>.

The sequence above illustrates the unified format for the multimodal instruction-following sequence. This transformative process elevates conventional image-text pairing into a comprehensive instructional scenario, facilitating deeper cross-modal understanding and responsiveness in the model.

For each input image Xv, the vision encoder g extracts visual features Zv = g(Xv). Subsequently, a trainable projection network converts Zv into language embedding tokens Hv, aligning their dimensions with the word embedding space in the language model:

###### Hv = P(Zv), with;Zv = g(Xv) (1)

Thus, a sequence of visual tokens Hv is obtained. For generating multi-turn conversation data (Xq1,Xa1,··· ,XqT,XaT) corresponding to each image Xv, all answers are considered as the assistant’s responses, resulting in a unified structure for the multimodal instruction-following sequence. The LLM then instructiontuned on prediction tokens using its original auto-regressive training objective.

Specifically, for a sequence of length L, the probability of target answers Xa is computed as follows:

p(Xa|Xv,Xq) =

L

pθ(xi|Xv,Xq,<i,Xa,<i) (2)

i=1

where θ represents trainable parameters, and Xq,<i and Xa,<i denote instruction and answer tokens preceding the current prediction token xi, respectively.

The LLaVA model training involves a two-stage instruction-tuning procedure:

- Stage 1: Pre-training for Feature Alignment. By utilizing 595K image-text pairs filtered from CC3M, which are reformatted into an instruction-following format, this stage

aims to align image features Hv with the pre-trained LLM word embeddings. During this stage, the visual encoder and the LLM weights remain frozen. Training is focused on the trainable parameters within the projector. This is equivalent to create a visual tokenizer compatible with the frozen LLM embeddings.

- Stage 2: Fine-tuning End-to-End. The visual encoder weights remain frozen while continuing to update both pretrained projector and LLM weights.

#### 2.3. MobileVLM

MobileVLM is a multimodal vision language model designed explicitly for deployment on mobile devices. Central

to its efficiency is the Lightweight Downsample Projector (LDP), as depicted in Figure 2. The LDP reduces the number of visual tokens by 75%, leading to a significant increase in inference speed.

𝑓 ∈ ℝ𝑁𝑣×𝐷𝑣

[Figure 31]

Pointwise GELU

Pointwise

- Depthwise s=1 Pointwise

[Figure 32]

[Figure 33]

- Depthwise s=2 Pointwise

[Figure 34]

[Figure 35]

[Figure 36]

H𝒗 ∈ ℝ(𝑁𝑣/4)×𝐷𝑣

LDP

Layer Normalization

[Figure 37]

Pixel-wise addition

Figure 2. LDP architecture

### 3. Model architecture

#### 3.1. Overall Architectural Design

The architecture of our network, illustrated in Figure 3, closely mirrors that of LLaVA-1.5. It consists of three key components: 1) a vision encoder, 2) a lightweight language model (LLM), and 3) a projector responsible for aligning the visual and textual spaces.

#### 3.2. Vision Encoder

Our model utilizes the pre-trained CLIP ViT-L/14 with a resolution of 336×336 as the visual encoder.

#### 3.3. Language Model

To reduce operational costs, we trained an lightweight language model Xmodel-LM 1.1B from scratch. To ensure seamless integration with existing inference frameworks, our language model is designed to closely resemble LLaMA. The detailed settings of our LLM are provided in Table 3.

Xsystem-message <STOP> Human : X1instruct <STOP> Assistant: X1a <STOP> Human : X2instruct <STOP> Assistant: X2a <STOP> · · ·

Table 2. Input sequence used for model training. Only two conversation turns are shown here; in practice, the number of turns varies depending on the instruction-following data. The model is trained to predict assistant responses and determine stopping points, with only green sequences/tokens used to compute loss in the auto-regressive model.

𝐗𝐚

Language Response

Language Model

𝐇𝐯 𝐇𝐪

| | |
|---|---|
|Tokenizer| |

|Projector|
|---|

|𝐙𝐯| |
|---|---|
|Vision Encoder| |

𝐗𝐪

Language Instruction

𝐗𝐯

Image

Figure 3. Xmodel-VLM architecture

# of Params Hidden size Heads Layers Context Len 1.1B 2048 32 24 4096

Table 3. Detailed settings of our language model.

To tokenize text data, we employ the unigram algorithm [18], utilizing the implementation provided by SentencePiece [38].

In Table 4, we evaluate our model using standard natural language benchmarks, focusing on language understanding and common sense reasoning. The evaluation is conducted utilizing the Language Model Evaluation Harness tool [13]. Our experimental results demonstrate that our Xmodel-LM 1.1B model performs comparably to recent open-source models, including OPT 1.3B, Pythia 1.4B, MobileLLaMA 1.4B and TinyLLaMA 1.1B.

#### 3.4. Projector

We employ a two-layer MLP to enhance the connection between the vision encoder and LLM, and use the Mish [30] function for activation. Notably, this innovative projector also serves as a downsampling mechanism, effectively reducing the number of visual tokens by 75%. Our projector architecture, denoted as XDP, exemplifies a paradigm of

simplicity and efficacy, as shown in Figure 4.

16 × 576 × 1024 (batch size × visual tokens × feature size)

𝐗𝐯 Images

| |
|---|
| |

|Vision Encoder| |
|---|---|
|𝐙𝐯| |

reshape

16 × 144 × 4096

| |
|---|
| |

|Projector| |
|---|---|
|𝐇𝐯| |

…

4096

| |
|---|

…

2048

| | | | | | | |
|---|---|---|---|---|---|---|
|Mish| | | | | | |

16 × 144 × 2048

(batch size × language tokens × hidden size)

…

2048

Figure 4. XDP architecture

### 4. Experiment

#### 4.1. Training

Our training process involves two main steps: pretraining and instruction tuning, as illustrated in Figure 5.

[Figure 38]

[Figure 39]

Xmodel-LM

Xmodel-LM

Projector

Projector

Xq

##### Xq Xv

Vision

Vision

Xv

Encoder

Encoder

Stage 1: Pre-training Stage 2: Instruction Tuning

Figure 5. Training strategy

Initially, we focus on training efficient projector while keeping the visual encoder and Xmodel-LM frozen. Subsequently, we conduct comprehensive fine-tuning of both the projector and the language model (LLM) to enhance their visual understanding and language processing capabilities.

ARC(c/e) BoolQ HellaSwag OpenBookQA PIQA SciQ TriviaQA Winogrande Avg. OPT 1.3B [44] 23.29 / 57.07 57.89 41.52 23.40 71.65 84.20 7.43 59.51 47.33 Pythia 1.4B [37] 26.02 / 60.69 63.09 40.43 22.40 70.73 86.40 5.67 57.30 48.08 MobileLLaMA 1.4B [9] 26.28 / 61.32 57.83 42.87 23.60 71.33 87.30 12.02 58.25 48.98 TinyLLaMA 1.1B (3T) [43] 27.82 / 60.31 57.83 44.98 21.80 73.34 88.90 11.30 59.12 49.49

Xmodel-LM 1.1B 28.16 / 62.29 61.44 45.96 24.00 72.03 89.70 18.46 60.62 51.41

Table 4. Comparison with 1B scale state-of-the-art language models on prominent language benchmarks

Method LLM Res. VizWiz SQAI VQAT POPE GQA MMB MMBCN MM-Vet MME Openflamingo [2] MPT-7B 336 – – 33.6 – – 4.6 – – – BLIP-2 [19] Vicuna-13B 224 19.6 61.0 42.5 85.3 41.0 – – 22.4 1293.8 MiniGPT-4 [46] Vicuna-7B 224 – – – – 32.2 23.0 – 22.1 581.7 InstructBLIP [11] Vicuna-7B 224 34.5 60.5 50.1 – 49.2 36 23.7 26.2 – InstructBLIP [11] Vicuna-13B 224 33.4 63.1 50.7 78.9 49.5 – – 25.6 1212.8 Shikra [6] Vicuna-13B 224 – – – – – 58.8 – – – Qwen-VL [3] Qwen-7B 448 35.2 67.1 63.8 – 59.3 38.2 7.4 – 1487.6 MiniGPT-v2 [5] LLaMA-7B 448 – – – – 60.3 12.2 – – – LLaVA-v1.5-13B [25] Vicuna-13B 336 53.6 71.6 61.3 85.9 63.3 67.7 63.6 35.4 1531.3 MobileVLM 1.7 [39] MobileLLaMA 1.4B 336 26.3 54.7 41.5 84.5 56.1 53.2 16.67 21.7 1196.2

Xmodel-VLM Xmodel-LM 1.1B 336 41.7 53.3 39.9 85.9 58.3 52.0 45.7 21.8 1250.7

Table 5. Comparison with SOTA methods on 9 VLM benchmarks. VizWiz [14]; SQAI: ScienceQA-IMG [29]; VQAT: TextVQA [36]; POPE [22]; GQA [15]; MMB: MMBench [27]; MMBCN: MMBench-Chinese [27]; MM-Vet [42]; MME [12]; Column Res. is the image resolution of vision model.

For pretraining, we utilize a filtered subset of the CC595K dataset for one epoch, with an initial learning rate of 1e-3 and a batch size of 64. We then fine-tune the model for one epoch on the LLaVA-Instruct-150K dataset, using a learning rate of 4e-5 and a batch size of 32. We employ a weight decay of 0 and utilize the AdamW optimizer with momentum parameters of 0.9 and 0.999, along with an epsilon value of 1e-8. During fine-tuning, all parameters in LLM are updated without using LoRA.

#### 4.2. Evaluation of Xmodel-VLM

We evaluate the multimodal performance across a variety of datasets: VizWiz [14], SQAI [29], VQAT [36], POPE [22], GQA [15], MMB [27], MMBCN [27], MMVet [42], and MME [12]. Our analysis, as depicted in Table 5, illustrates that despite having a smaller parameter count, our proposed Xmodel-VLM 1.1B demonstrates competitive performance.

We evaluated the inferential latency of our model, comparing it with LLAVA-7B and Mobile-VLM models. Utilizing the SQAI dataset on a single NVIDIA GeForce RTX 3090 GPU with 24GB memory, we measured inference speed, excluding preprocessing time. Results in Table 6 show our model’s faster inference compared to LLAVA-7B, although MobileVLM-1.7B remains faster. This highlights

the advantage of compact models in delivering expedited inference results.

Model Size(GB) Samples(token/s) Total(s) LLaVA-7B 7B 19.06 1090.32 MoblieVLM-1.7B 1.7B 919.25 579.88 Xmodel-VLM 1.1B 415.69 1360.25

Table 6. Lantency comparison of small VLMs. Smaples reflects a model can predict the number of token per second, and Total denotes the total time of inference on the SQAI dataset.

### 5. Ablation Study

#### 5.1. Projectors

With the language model fixed to Xmodel-LM 1.1B, vision encoder to clip-vit-large-patch14-336, and token number to 144, we examine different projector architectures and their impact on multimodal performance, as depicted in Table 7.

#### 5.2. Token Numbers

In order to explore the impact of different visual token numbers on model performance, we keep the language

Projector GQA SQAI VQAT POPE MME MMBdev

Linear 59.2 31.6 38.7 85.7 1274.0 42.7 MLP 60.1 49.2 40.1 86.8 1294.1 49.1 LDP 57.5 52.5 37.2 85.5 1230.5 45.3 LDPv2 58.8 52.8 38.4 86.2 1275.6 48.6

XDP 58.3 53.3 39.9 85.9 1250.7 52.0

Table 7. Comparison with different projector architectures

model as Xmodel-LM 1.1B, vision encoder as clip-vitlarge-patch14-336, and projector architecture as XDP, we change the number of tokens and assess multimodal performance, as illustrated in Table 8.

Tokens GQA SQAI VQAT POPE MME MMBdev

576 60.1 53.6 40.3 86.9 1259.4 48.3 288 59.1 42.8 39.8 86.8 1226.2 48.5 144 58.3 53.3 39.9 85.9 1250.7 52.0 72 57.0 46.4 37.7 84.9 1234.0 45.3 64 56.9 43.6 35.3 84.9 1234.7 43.6 36 55.2 52.9 33.8 84.3 1214.0 47.1 18 54.4 44.7 32.1 83.3 1247.1 40.2 8 52.3 40.8 29.8 80.7 1120.5 41.2 4 51.2 43.2 29.2 80.1 1040.1 36.2 2 49.7 44.0 28.8 79.6 1006.6 33.0 1 49.4 44.7 27.5 79.2 1070.6 32.1

Table 8. Comparison with different token numbers

LLM GQA SQAI VQAT POPE MME MMBdev

LLaMA-3 8B 60.8 68.8 51.4 86.7 1458.7 65.0 Phi-3 4B 59.4 71.2 51.3 86.8 1388.7 65.8

Xmodel-LM 1.1B 57.4 54.4 38.9 86.1 1251.5 48.5

Table 9. Comparison with different Language Models

### 6. Conclusion

In summary, we present a high-performance visual language model achieved through careful selection of visual encoders, efficient projector design, and a two-stage training strategy. Extensive experiments on popular VLM benchmarks demonstrate its effectiveness. We anticipate our technology will unlock new possibilities across various applications, including customer service robots.

Surprisingly, as the token count decreases, although there is an overall downward trend in various evaluation metrics, the change is remarkably gradual. Especially when only a small number of visual tokens are used, such as 1 and 2, there is no significant difference in the evaluation results. Of course, this may be related to the fact that there is no fine-grained description of the image in the datasets, which is the limitation of the vision encoder in modality alignment.

#### 5.3. Language Model

To investigate the potential of our approach, we compared the impact of different sizes of LLM on performance based on the structure of the vision encoder as clip-vitlarge-patch14-336 and projector architecture as XDP with 144 tokens, and the results are shown in Table 9. This result shows that larger language models are an effective way to improve the performance of our model, which provides a direction for further improvement of our method in the future.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1

- [2] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. arXiv preprint arXiv:2308.01390, 2023. 1, 2, 5
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023. 2, 5
- [4] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pretraining to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3558–3568, 2021. 2
- [5] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023. 5
- [6] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 5
- [7] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 2
- [8] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6,

2023. 1, 2

- [9] Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, et al. Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886, 2023. 1, 2, 5
- [10] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instructionfinetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024. 1
- [11] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning,

2023. 2, 5

- [12] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2024. 5
- [13] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023. 4
- [14] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617,

2018. 5

- [15] Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering, 2019. 5
- [16] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, et al. Openclip. If you use this software, please cite it as below, page 1, 2021. 2
- [17] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017. 2
- [18] Taku Kudo. Subword regularization: Improving neural network translation models with multiple subword candidates. arXiv preprint arXiv:1804.10959, 2018. 4
- [19] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 1, 2, 5
- [20] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–

12900. PMLR, 2022. 2

- [21] Yuanzhi Li, S´ebastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023. 1
- [22] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models, 2023. 5
- [23] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models, 2024. 2
- [24] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence

- Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 2
- [25] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 1, 2, 5
- [26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 2
- [27] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player?, 2024. 5
- [28] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986,

2022. 2

- [29] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering, 2022. 5
- [30] Diganta Misra. Mish: A self regularized non-monotonic activation function. arXiv preprint arXiv:1908.08681, 2019. 4
- [31] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. Advances in neural information processing systems, 24, 2011. 2
- [32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages

- 8748–8763. PMLR, 2021. 1, 2

[33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages

- 8748–8763. PMLR, 2021. 2

- [34] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 2
- [35] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 2
- [36] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read, 2019. 5
- [37] Quentin Gregory Anthony Herbie Bradley Kyle O’Brien Eric Hallahan Mohammad Aflah Khan Shivanshu Purohit

- USVSN Sai Prashanth Edward Raff et al. Stella Biderman, Hailey Schoelkopf. Pythia: A suite for analyzing large language models across training and scaling. arXiv preprint arXiv:2304.01373, 2023. 5
- [38] John Richardson Taku Kudo. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226,

2018. 4

- [39] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1, 5
- [40] MosaicML NLP Team et al. Introducing mpt-7b: A new standard for open-source, commercially usable llms, 2023. URL www. mosaicml. com/blog/mpt-7b. Accessed, pages 05– 05, 2023. 1
- [41] Yichuan Wang, Yang Liu, Yu Yan, Xucheng Huang, and Ling Jiang. Xmodel-lm technical report. arXiv preprint arXiv:2406.02856, 2024. 1
- [42] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 5
- [43] Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model, 2024. 5
- [44] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068,

2022. 1, 5

- [45] Baichuan Zhou, Ying Hu, Xi Weng, Junlong Jia, Jie Luo, Xien Liu, Ji Wu, and Lei Huang. Tinyllava: A framework of small-scale large multimodal models. arXiv preprint arXiv:2402.14289, 2024. 1, 2
- [46] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 2, 5
- [47] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billion-scale corpus of images interleaved with text. Advances in Neural Information Processing Systems, 36, 2024. 2
- [48] Yichen Zhu, Minjie Zhu, Ning Liu, Zhicai Ou, Xiaofeng Mou, and Jian Tang. Llava-phi: Efficient multi-modal assistant with small language model, 2024. 1, 2

