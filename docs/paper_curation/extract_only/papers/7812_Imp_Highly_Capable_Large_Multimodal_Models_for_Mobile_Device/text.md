[Figure 1]

# arXiv:2405.12107v2[cs.CV]30May2024

#### Imp: Highly Capable Large Multimodal Models for Mobile Devices

Zhenwei Shao, Zhou Yu†, Jun Yu, Xuecheng Ouyang, Lihao Zheng, Zhenbiao Gai, Mingyang Wang, Jiajun Ding

1Hangzhou Dianzi University

https://imp-vl.github.io

##### Abstract

By harnessing the capabilities of large language models (LLMs), recent large multimodal models (LMMs) have shown remarkable versatility in open-world multimodal understanding. Nevertheless, they are usually parameter-heavy and computation-intensive, thus hindering their applicability in resource-constrained scenarios. To this end, several lightweight LMMs have been proposed successively to maximize the capabilities under constrained scale (e.g., 3B). Despite the encouraging results achieved by these methods, most of them only focus on one or two aspects of the design space, and the key design choices that influence model capability have not yet been thoroughly investigated. In this paper, we conduct a systematic study for lightweight LMMs from the aspects of model architecture, training strategy, and training data. Based on our findings, we obtain Imp—a family of highly capable LMMs at the 2B∼4B scales. Notably, our Imp-3B model steadily outperforms all the existing lightweight LMMs of similar size, and even surpasses the state-of-the-art LMMs at the 13B scale. With low-bit quantization and resolution reduction techniques, our Imp model can be deployed on a Qualcomm Snapdragon 8Gen3 mobile chip with a high inference speed of about 13 tokens/s.

###### 1. Introduction

“A very small man can cast a very large shadow.”1

—— George R. R. Martin, "A Clash of Kings"

The revolution of large language models (LLMs) has greatly changed the landscape of artificial intelligence in recent years [4, 9, 26, 53, 61]. The latest LLMs like GPT-4 [49] and Gemini1.5 [60] exhibit remarkable versatilities and capabilities across a variety of domains and tasks. Meanwhile, there has been an increasing interest in expanding the language-only LLMs to large multimodal models (LMMs), which aims to jointly handle more modalities beyond language, such as vision [3, 33, 50, 64, 78], speech [12, 74], and their combinations [18, 43, 59, 75, 77]. These LMMs significantly extend the capabilities of language-only LLMs, enabling more natural and flexible interactions to accomplish complex real-world tasks.

1This description is about a dwarf character named Tyrion Lannister in the fiction, whose nickname is Imp.

† Project lead & Corresponding author: Zhou Yu (yuz@hdu.edu.cn)

Despite the remarkable progress achieved by these LMMs, they are usually parameter-heavy and computation-intensive in both training and inference stages, which hinder the participation of academic researchers with limited resources and pose challenges for applications in resourceconstrained environments like PCs and mobile devices. In light of the limitations above, building lightweight yet strong LMMs has rapidly drawn attention from both academia and industry. Built upon the lightweight LLMs like Phi-2 (2.7B) [25] and MiniCPM [79], some representative LMMs around 3B scale have achieved comparative results to the state-of-the-art counterparts at 7B scale [3, 37] by introducing carefully-designed training strategies [65, 71], advanced architectures [10], and enriched training data [22, 79], respectively.

Despite the prominence of these lightweight LMMs, most of them only focus on one or two aspects of the whole design space. It still remains unclear what the key design choices are that influence the capabilities of a lightweight LMM. Although some systematic studies have been made for the 7B/13B LMMs [29, 37], there is no clear evidence that these empirical design choices can be directly transferred to the LMMs of much smaller sizes. To this end, we conduct a thorough study to investigate the impact of design choices of the lightweight LMMs in controlled settings. Originating from the commonly-used LLaVA-1.5 [37] model, we construct a comprehensive roadmap by evaluating the impacts of different model architecture, training strategy, and training data progressively. Building upon the open-source lightweight LLMs like Qwen-1.5 [2], Phi-2 [25], and Phi-3 [47], we obtained a family of highly capable yet lightweight LMMs named Imp-2B, Imp-3B and Imp-4B, respectively. Notably, Our Imp3B model significantly outperforms the counterparts of similar size on a broad range of LMM benchmarks and also steadily surpasses the state-of-the-art LMMs at the 13B scale. Moreover, we do not use any proprietary pretrained models or private training data to ensure reproducibility. With low-bit quantization and reduced image resolution techniques, the optimized Imp-3B model can run efficiently at about 13 tokens per second on a mobile phone equipped with a Qualcomm Snapdragon chip. The code and pretrained models are publicly available at Github2 and HuggingFace3, respectively. We hope that our work may serve as a solid baseline for future research in lightweight LMMs.

###### 2. Related Work

Large language models (LLMs). Due to the remarkable versatilities and capabilities across various language tasks, LLMs have revolutionized the field of artificial intelligence in the past few years. By scaling the parameter sizes of Transformers [63] to tens or even hundreds of billions, LLMs have exhibited emergent properties that have not been witnessed in previous small pretrained language models [13, 52, 55]. The prominent success of LLMs is first brought by commercial models like the GPT series [4, 49, 51] and ChatGPT [48], and further accelerated by the open-sourced models such as LLaMA [61, 62], Mistral [26], Qwen [2], and Baichuan [67]. The success of LLMs has also promoted the research interest in large multimodal models (LMMs), which aim to empower LLMs with the ability to handle multiple modalities.

Large multimodal models (LMMs). The research of LMMs can be roughly categorized into two lines: loosely coupled and tightly coupled approaches. Loosely coupled approaches, e.g., Visual ChatGPT [66], MM-REACT [68], and HuggingGPT [56], leverage the LLMs to coordinate with multiple external vision models to understand and express visual information. These methods

2Code: https://github.com/MILVLG/imp 3Models: https://huggingface.co/MILVLG

are also known as multimodal agent approaches since the LLMs can autonomously plan and invoke vision models as tools to tackle the multimodal tasks. The tightly coupled approaches aim to train end-to-end multimodal models by aligning pretrained vision models with LLMs through multimodal interaction modules. Flamingo introduces a gated cross-attention mechanism to align the visual and language modalities [1]. LLaVA uses a simple linear projection layer to map the visual representations into the embedding space of the LLM [38]. BLIP-2 [32] and MiniGPT-4 [78] learn a heavier Q-Former modules to establish alignment across modalities effectively. Subsequent studies construct diverse and high-quality multimodal instruction datasets to enhance specific capabilities of the learned LMMs [3, 6, 11, 36, 37, 64, 73].

Lightweight LLMs and LMMs. Currently, most popular LLMs and LMMs are parameterheavy and computation-intensive, which hinders their applicability in resource-constrained scenarios like PC and mobile devices. It is worth noting that there has been an increasing interest in developing lightweight yet powerful LLMs and LMMs below the 7B scale. For example, Phi (1.3B and 2.7B) [20, 25], Gemma (2B) [17], Qwen (1.8B and 4B) [2], TinyLlama (1.1B) [76], MobileLLaMA (1.4B and 2.7B) [10] and MiniCPM (2B) [79] are representatives LLMs of this size. The open-source of these lightweight LLMs also facilitates the exploration of lightweight LMMs, thus a series of early attempts have been made successively, such as TinyGPT-V (3B) [71], LLaVA-Phi (3B) [80], Vary-Toy (1.8B) [65], MiniCPM-V (3B), Bunny (3B) [22] and MobileVLM (3B) [10]. However, there is a lack of through study of the model architecture, training strategy, and training data to explore the potential capabilities of lightweight LMMs.

###### 3. Preliminaries

Our Imp models are derived from LLaVA-1.5 [37], a highly capable 7B LMM trained on curated publicly available data. To better describe our modifications, we first revisit LLaVA-1.5’s model architecture and training recipe, as illustrated in Figure 1.

Model architecture. As shown in Figure 1a, the architecture of LLaVA-1.5 consists of three key components: a pretrained visual encoder, a pretrained LLM, and an intermediate multimodal connector trained from scratch. For an input image, it is represented as a sequence of visual embeddings in the same word (token) embedding space of the LLM. To achieve this goal, LLaVA introduces a visual encoder and a multimodal projector. Specifically, LLaVA-1.5 utilizes a ViT-based visual encoder pretrained by multimodal contrastive learning [14], which refers to the CLIP ViT-L/14@336 model (0.3B) [54], to encode the image into a sequence of 576 (24×24) flattened visual features. After that, these visual features are fed through a multimodal connector module implemented by a two-layer MLP, which transforms the visual embeddings to the same dimensionality as the word embedding. The obtained visual embeddings are concatenated with the word embeddings of input language instruction to form a multimodal input, which is then fed into a pretrained LLM (e.g., Vicuna [8]) to generate a language response token by token.

Training recipe. As shown in Figure 1b, LLaVA-1.5 utilizes a two-stage training scheme, namely multimodal alignment pretraining and multimodal instruction tuning, to guarantee sufficient learning of each network component. Specifically, in the first stage, only the multimodal connector is trained while the rest of the model are frozen. This stage uses a collection of 558K paired image-caption data, which aims to learn the alignment of visual embeddings and word embeddings. In the second stage, the LLM and multimodal connector are optimized

language response

Large Language Model (e.g., Vicuna-7B)

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

...

Multimodal Connector

Visual Encoder (e.g., CLIP ViT-L)

tokenize & embed

image input language instruction

(a) Model architecture

###### Stage-1

multimodal alignment pretraining

[Figure 2]

Large Language Model

[Figure 3]

Multimodal Connector

[Figure 4]

Visual Encoder

###### Stage-2

multimodal instruction tuning

[Figure 5]

Large Language Model

[Figure 6]

Multimodal Connector

[Figure 7]

Visual Encoder

(b) Training recipe

Figure 1 | LLaVA-1.5’s model architecture and its two-stage training recipe.

jointly while the visual encoder is still frozen. This stage aims to endow the LMM with the instruction-following ability by training on 665K image-instruction-response triplets collected from a mixture of academic VQA datasets and GPT-generated datasets.

###### 4. A Roadmap from LLaVA to Imp

In this section, we introduce a detailed roadmap to obtain our Imp-3B model from LLaVA-1.5-7B. To remedy the capability degradation of smaller models, we conduct a thorough study of the design space of LMMs, namely model architecture, training strategy, and training data. An intuitive roadmap is shown in Figure 2, and its extended results are shown in Table 1. Detailed analyses are provided below.

###### 4.1. Optimized Model Architectures

We start the exploration by searching for optimal model architectures, which consists of the choices of the LLM and visual encoder.

Choice of the LLM. We adopt the LLaVA-1.5-7B model [37] trained with LoRA [23] as our reference model. Based on calculations, the majority of parameters in LLaVA come from its LLM backbone Vicuna-7B [8]. In order to obtain a lightweight LMM, our first step is to replace Vicuna with a smaller yet strong LLM. To make a trade-off between efficiency and efficacy, we choose candidate LLMs at a 2.7B scale for comparison, namely, Phi-2 [25] and MobileLLaMA [10].

From the results in section 1.1 of Table 1, we can see that: 1) when the same visual encoder is used (i.e., CLIP ViT-L), replacing Vicuna-7B with a smaller LLM (Phi-2 or MobileLLaMA) brings notable performance degradation in terms of average score over the six benchmarks. This suggests that the LMM’s performance is highly dependent on its supporting LLM, and the capability of larger LLMs usually outperforms smaller ones according to the scaling law [28]. 2) With the same 2.7B model size, the LMM with Phi-2 significantly outperforms the counterpart with MobileLLaMA, showing the superiority of Phi-2 due to its meticulously organized training data. Therefore, we choose Phi-2 as our default LLM in the following design.

Choice of the visual encoder. Apart from the LLM, the choice of the visual encoder also impacts the capability of the LMM. LLaVA-1.5 uses the powerful CLIP-ViTL/14@336 model as its visual encoder, which is contrastively pretrained on 400M curated image-text pairs [54]. To validate the capability of different visual encoders, we experiment with two pretrained visual encoders, namely, IN1k-ViT-L/16@336 [14] and SigLIP-SO400M/14@384 [72]. The former one is the original ViT model pretrained on ImageNet-21K and then finetuned on ImageNet-1K, and the latter one uses the same training methodology as CLIP but a shape-optimized ViT architecture with a slightly larger image resolution.

[Figure 8]

Figure 2 | An overview roadmap from LLaVA-1.57B to Imp-3B. The average score is calculated on six commonly-used LMM benchmarks, namely VQAv2 [19], GQA [24], TextVQA [58], ScienceQAIMG [42], POPE [34], and MMB (dev) [40]. More detailed results can be referred to Table 1.

The results in section 1.2 of Table 1 show that: 1) with similar model architecture and the same input resolution, the LMM with CLIP-ViT-L significantly outperforms the counterpart with IN1k-ViT-L/16@336. This can be explained by the fact that large-scale image-text contrastive learning can facilitate generalization of the visual encoder and obtain more fine-grained visual representations. 2) Replacing CLIP-ViT with SigLIP brings consistent performance improvement on all the benchmarks, which can be explained by the synergistic effect of improved model capability and increased visual tokens (576 in CLIP vs. 729 in SigLIP). Therefore, we choose SigLIP as our default visual encoder in the following. It is worth noting that the current 3B-scale LMM has already surpassed the reference 7B counterpart in terms of the average score, showing the feasibility and potentiality of lightweight LMMs.

###### 4.2. Improved Training Strategies

After exploring the model architecture, we fix the default settings and then investigate the training strategies, including the finetuning mechanism and number of training epochs.

Finetuning mechanism. As mentioned above, LLaVA is trained using a two-stage manner. In the first stage, the visual encoder and LLM are kept frozen and only the multimodal connector is trained, while in the second stage, the LLM and multimodal connector are finetuned jointly. As the first stage only acts as an initialization, it is of less importance compared to the second stage. Therefore, we maintain the first-stage training settings in LLaVA, and explore different finetuning mechanisms in the second stage.

Our exploration is carried out on two levels. On the macro level, we compare the traditional full-parameter finetuning and LoRA finetuning [23] mechanisms. On the micro level, we explore the LoRA finetuned models with different ranks (i.e., 128, 256, and 512). From the results in section 2.1 of Table 1, we can see that: 1) the model trained with full-parameter funetuning is inferior to the models with LoRA finetuning while requiring much more GPU memory. Therefore, we use LoRA finetuning as our training mechanism for the second stage. 2) For LoRA finetuning, increasing rank from 128 to 256 brings a 0.2 point average score improvement, while

- Table 1 | A detailed roadmap from LLaVA-1.5-7B [37] to our Imp-3B model. Each section is separated by a line, which refers to one of the three aspects of the design space. The ablation studies are conducted in a progressive manner that each selected choice (marked with a purple background) will be used in the following experiments. The best result on each dataset is bold.

VQAv2 GQA VQAT SQAI POPE MMB Avg. Score

###### 1. Optimized Model Architectures

- 1.1. Choice of the LLM Vicuna-1.5 (7B) [8] (LLaVA-1.5 [37]) 79.1 63.0 58.2 68.4 86.4 66.1 70.2 MobileLLaMA (2.7B) [10] 76.0 60.6 49.9 58.7 87.0 58.4 65.1 Phi-2 (2.7B) [25] 76.5 57.2 52.3 69.8 87.2 68.6 68.6

- 1.2. Choice of the visual encoder CLIP-ViT-L/14@336 (0.3M) [54] 76.5 57.2 52.3 69.8 87.2 68.6 68.6 IN1K-ViT-L/16@336 (0.3M) [14] 67.5 55.2 38.1 63.1 83.2 50.5 59.6 SigLIP-SO400M/14@384 (0.4M) [72] 79.9 62.9 57.9 70.3 87.8 69.4 71.4

- 2. Improved Training Strategies

- 2.1. Finetuning mechanism Full-parameter finetuning 79.3 61.8 57.1 70.9 87.4 70.5 71.2 LoRA finetuning, rank=128 79.9 62.9 57.9 70.3 87.8 69.4 71.4 LoRA finetuning, rank=256 79.9 63.0 57.9 71.0 87.8 70.0 71.6 LoRA finetuning, rank=512 80.0 62.6 57.4 71.2 87.7 70.4 71.5

- 2.2. Number of training epochs

- 1 epoch 79.9 63.0 57.9 71.0 87.8 70.0 71.6

- 2 epochs 81.2 63.8 59.4 71.2 87.8 69.3 72.1

- 3 epochs 81.5 64.0 57.7 70.0 87.5 69.7 71.7

- 3. Augmented Stage-2 Training Data

- 3.1. OCR & chart-oriented data

Original-665K [37] 81.2 63.8 59.4 71.2 87.8 69.3 72.1 643K (665K−22K TextCaps) 81.2 64.2 54.6 69.5 87.9 70.4 71.3 675K (643K+32K OCR & chart data) 81.2 63.9 57.6 71.0 87.6 69.6 71.8

- 3.2. GPT4V-annotated data 675K 81.2 63.9 57.6 71.0 87.6 69.6 71.8 705K (675K+30K captioning data) 81.2 64.1 58.5 73.2 88.0 70.4 72.6 1M (705K+300K conversation data) 81.2 63.5 59.8 72.8 88.9 72.9 73.2

further increasing it to 512 leads to a 0.1 point decrease. Based on the above observations, we adopt LoRA finetuning with rank 256 as our finetuning mechanism in the following experiments.

Number of training epochs. LLaVA-1.5 is trained for one epoch by default. However, whether the model is sufficiently trained remains unclear. To this end, we experimented with different numbers of training epochs to verify the impact of this factor.

As shown in section 2.2 of Table 1, increasing training epochs from 1 to 2 brings a 0.5 point improvement in average score. This observation verifies our hypothesis that the model trained for one epoch may be undertrained. Meanwhile, further increasing training epochs from 2 to 3 leads to a 0.4 point decrease, suggesting that the 2 epochs are the optimal setting. Likewise, we set the number of training epochs to 2 in the following explorations.

- 4.3. Augmented Stage-2 Training Data

In addition to the model architecture and training strategy, the quality and diversity of training data, especially the instruction tuning data in the second stage, plays a key role in the capabilities

- Table 2 | Statistical information of the instruction tuning dataset for Imp. Based on LLaVA’s original 665K data, we remove the TextCaps dataset in line with [39] and then append about 32K OCR & chart-oriented datasets and 330K GPT4V-annotated datasets, resulting in 1M mixed instruction-tuning data in total.

LLaVA-1.5 (trimmed) OCR & chart-oriented datasets (32K) GPT4V-annotated datasets (330K) dataset

original-665K [37] (w/o TextCaps [57])

DVQA [27]

ChartQA [44]

DocVQA [45]

AI2D [30]

InfographicVQA [46]

ShareGPT-4V [7]

LAION-GPT-V [31]

ALLaVA [5]

size 643K 10K 4K 10K 4K 4K 20K 10K 300K

of LMMs [37, 41]. LLaVA-1.5 elaborately constructs a 665K mixed dataset from several academictask-oriented VQA datasets, which has been widely used as the stage-2 training data for many open-source LMMs. Inspired by [39], we have considered two types of training data to augment the original 665K data, namely OCR & chart-oriented data and GPT4V-annotated data. Detailed statistical information of the augmented dataset is illustrated in Table 2.

OCR & chart-oriented data. Similar to [35, 39], we introduce DVQA [27], ChartQA [44], DocVQA [45], AI2D [30], and InfographicVQA [46], which are human annotated VQA datasets that focus on reasoning about of OCR and chart in images. Meanwhile, we remove the 22K TextCaps data [57] from LLaVA-1.5’s 665K dataset, which uses the same set of training images as TextVQA [58]. This allows us to better evaluate the zero-shot performance on the TextVQA benchmark. As a result, we obtain an augmented dataset with 675K samples (665K+32K−22K).

From the results in section 3.1 of Table 1, we can see that: 1) removing TextCaps leads to a 4.8 point significant decline on TextVQA and a 0.8 point performance drop in the average score. This reflects the actual zero-shot performance on TextVQA. 2) Introducing OCR & chart-oriented data brings notable improvements to TextVQA and ScienceQA, both of which require understanding and reasoning about the texts in the image.

GPT4V-annotated data. Apart from the academic VQA datasets, high-quality instructiontuning data about images is valuable yet often inaccessible. As an alternative, one can harness the state-of-the-art LMM (e.g., GPT-4V) to annotate responses according to input images and predefined prompts. We utilize three typical GPT4V-annotated datasets, namely ShareGPT-4V [7], LAION-GPT-V [31], and ALLaVA [5]. The first two are visual captioning datasets (30K in total), and the last one is a conversation dataset for general-purpose multimodal tasks. Note that the original ALLaVA contains 708K samples. To avoid its dominance in our mixed dataset, we randomly sample a 300K subset from ALLaVA. Finally, we obtain about an augmented dataset with 1M samples (675K+30K+300K) in total.

From the results in section 3.2 of Table 1, we observe that both the captioning and conversation data facilitate the model capability. Their synergistic effect enables us to obtain a highly capable LMM with a 73.2 average score, which is 3 points higher than LLaVA-1.5-7B on average. We name this model Imp-3B and will compare it with the state-of-the-art LMMs next.

###### 5. Main Results

The roadmap above ends with a combination of design choices that is generic and transferrable. In addition to Phi-2, we also apply these design choices to different lightweight LLMs, namely,

- Table 3 | Comparison of our Imp models (with purple backgrounds) and open-sourced stateof-the-art LMMs on ten commonly-used VQA and LMM benchmarks. #PT and #FT denote the number of pretrainng and instruction-tuning samples, respectively. Benchmark names are abbreviated due to space limits: VQA-v2 [19]; GQA [24]; VisWiz [21]; SQAI: ScienceQA-IMG [42]; VQAT: TextVQA [58]; POPE [34]; MMEP: MME-Perception [15]; MMB: MMBench (dev)[40]; MMBCN: MMBench-Chinese (dev)[40]; MM-Vet [70]. ∗: The training images of the datasets are observed during training. The compared LMMs are first categorized into the normal-sized and lightweight groups using a cut-off of 6B. Among the lightweight LMMs, we further categorize them into three groups based on their model sizes. Within each group of the lightweight LMMs, the best result on each benchmark is bold.

Method LLM Visual Enc@Res. #PT #FT VQAv2 GQA Vizwiz SQAI VQAT POPE MMEP MMB MMBCN MM-Vet Normal-sized LMM (>6B) Yi-VL-6B Yi-6B CLIP-H@224 100M 26M - - - - - - - 68.4 68.6 mPLUG-Owl2-8B LLaMA2-7B CLIP-L@448 400M 1.2M 79.4 56.1 54.5 68.7 54.3 - 1450.2 64.5 - 36.2 SPHINX-X-8B InternLM2-7B CLIP&Dino@224 0 15M 75.5* 56.2* 49.6 70.4 58.1 86.9 1260.4 57.9 - InstructBLIP-8B Vicuna-7B ViT-G/14@224 129M 1.2M - 49.2* 34.5 60.5 50.1 - - - - LLaVA-1.5-7B Vicuna-1.5-7B CLIP-L@336 558K 665K 79.1∗ 63.0∗ 50.0 68.4 58.2∗ 86.4 1476.9 66.1 58.9 30.2 LLaVA-1.5-13B Vicuna-1.5-13B CLIP-L@336 558K 665K 80.0∗ 63.3∗ 53.6 71.2 60.2∗ 86.7 1541.7 68.5 61.5 38.3 Lightweight LMM (<6B) Bunny-4B Phi-3-3.8B SigLIP-SO@384 2M 695K 81.5* 63.5* - 75.1 - 86.7 1495.2 73.5 - Imp-4B Phi-3-3.8B SigLIP-SO@384 558K 1M 81.5* 63.5* 51.2 78.3 60.2 86.9 1507.7 73.3 61.1 44.6

TinyGPT-V-3B Phi-2-2.7B E-CLIP-G@448 23M 1M - 38.9∗ 37.8 - - - - - - LLaVA-Phi-3B Phi-2-2.7B CLIP-L@336 558K 665K 71.4∗ - 35.9 68.4 48.6∗ 85.0 1335.1 59.8 - 28.9 MobileVLM-3B MobileLLaMA-2.7B CLIP-L@336 558K 665K - 59.0∗ - 61.0 47.5∗ 84.9 1288.9 59.6 - MiniCPM-V-3B MiniCPM-SFT-2B SigLIP-SO@384 300M 8M - - - - - - 1452 .0 67.9 65.3 Bunny-3B Phi-2-2.7B SigLIP-SO@384 2M 695K 79.8∗ 62.5∗ - 70.9 - 86.8 1488.8 68.6 - Imp-3B Phi-2-2.7B SigLIP-SO@384 558K 1M 81.2* 63.5* 54.1 72.8 59.8 88.0 1446.4 72.9 46.7 43.3

Bunny-2B Qwen1.5-1.8B SigLIP-SO@384 2M 695K 76.6* 59.6* - 64.6 - 85.8 1300.8 59.1 58.5 Mini-Gemini-2B Gemma-2B CLIP-L@336 1.2M 1.5M - - - - 56.2 - 1341 59.8 - 31.1

- Imp-2B Qwen-1.5-1.8B SigLIP-SO@384 558K 1M 79.2* 61.9* 39.6 66.1 54.5 86.7 1304.8 63.8 61.3 33.5

Qwen-1.5 (1.8B) [2] and Phi-3 (3.8B) [47], to obtain a family of lightweight LMMs Imp-2B/3B/4B4. We conduct comprehensive quantitative and qualitative comparisons with state-of-the-art LMMs to validate the efficacy and efficiency of our Imp models. Each Imp model is trained on a server with 8 A100 GPUs (40GB) and finished in less than 32 hours.

###### 5.1. Quantitative Comparisons with the State-of-the-art LMMs

In Table 3, we compare our Imp models to the state-of-the-art lightweight LMMs, namely TinyGPT-V-3B [71], LLaVA-Phi-3B [80], MobileVLM-3B [10], MiniCPM-V-3B [79], Bunny-3B [22], and Mini-Gemini-2B [35]. Moreover, we also add comparisons to the state-of-the-art normal-sized LMMs, namely LLaVA-1.5-7B/13B [37], InstructBLIP-8B [11], mPlug-Owl2-8B [69], SPHINX-X-8B [16], to further demonstrate the superior capability of our Imp models.

From the results, we can see that Imp models steadily outperform all the counterparts at the same model size and is even on par with LLaVA-1.5-13B, reflecting the effectiveness of our elaborated model design. Compared with the strong competitor like MiniCPM-V-3B and Mini-Gemini-2B, Imp uses much less training data. This suggests that for lightweight LMMs, the quality of training data is more important than the quantity. Moreover, although Imp-2B exhibits

4We have previously released an Imp model named Imp-v1-3B on HuggingFace (https://huggingface.co /MILVLG/imp-v1-3b). To avoid ambiguity, the Imp models obtained in this paper are termed Imp-v1.5-2B/3B/4B when released on HuggingFace. To be more precise, Imp-v1 and Imp-v1.5 use slightly different image preprocess strategies (“resize-to-square” for Imp-v1 and “resize-then-pad” for Imp-v1.5) and different training data (665K for Imp-v1 and 1M for Imp-v1.5).

- Table 4 | Latency and performance comparisons of MobileVLM-3B@336 [10], Imp-3B@384 and Imp-3B@196 on different hardware platforms and quantization precision. All models are evaluated using llama.cpp framework with 16/8/4-bit quantilization precision. SD denotes the Snapdragon mobile chip from Qualcomm. 𝑇VE indicates the visual encoding time. 𝑆prompt and 𝑆gen measure the speed (tokens/s) of the prompt encoding and response generation stages, repetitively. 𝑇𝑡𝑜𝑡𝑎𝑙 refers to the entire latency to infer one sample.

Quant. Precision

Size (GB)

𝑆prompt

𝑆gen

𝑇total

𝑇VE

SQAI MMB

Model@resolution Hardware

(s)

(tokens/s)

(tokens/s)

(s)

RTX 3090 16-bit 5.6 0.027 5919.84 102.57 0.70 60.19 56.70 RTX 3090

0.027 3937.64 137.26 0.57 60.19 56.63

SD 8Gen3 4.93 17.19 13.79 21.10 SD 888 9.70 11.18 6.85 37.09

8-bit 3.3

MobileVLM-3B@336

RTX 3090

0.027 4006.48 176.17 0.47 58.70 55.76

SD 8Gen3 4.93 15.73 13.07 22.23 SD 888 9.63 10.73 8.15 35.85

4-bit 2.1

RTX 3090 16-bit 6.0 0.045 6125.18 97.91 0.83 73.13 71.05 RTX 3090

0.043 4731.30 138.17 0.68 73.03 71.05

SD 8Gen3 8.57 15.73 11.91 63.57 SD 888 17.52 10.10 5.82 105.80

8-bit 3.6

Imp-3B@384

RTX 3090

0.043 4814.34 167.17 0.59 72.88 70.88

SD 8Gen3 8.88 14.02 11.02 70.12 SD 888 16.9 9.89 6.74 105.05

4-bit 2.3

RTX 3090 16-bit 6.0 0.018 6325.2 101.71 0.69 68.37 63.40 RTX 3090

0.017 3744.14 137.57 0.57 68.42 63.66

- SD 8Gen3 2.09 17.12 13.63 21.24 SD 888 4.02 11.68 6.86 34.65

RTX 3090

4-bit 2.3

0.017 3930.54 170.42 0.48 68.47 63.14

- SD 8Gen3 2.10 15.64 12.73 22.72 SD 888 4.02 10.84 8.31 34.37

8-bit 3.6

Imp-3B@196

consistently worse performance than the rest two Imp models, its performance on the Chinese MMBCN benchmark is prominent. As our training data is purely in English, this phenomenon implies that the bilingual understanding capability of the LLM (i.e., Qwen-1.5) is successfully inherited by the LMM.

###### 5.2. Quantitative Comparisons on Mobile Devices

The lightweight nature of the Imp models enables us to deploy them on mobile devices. We choose two mobile phones equipped with Snapdragon 8Gen3 and Snapdragon 888 chips, respectively. To measure the latency on different devices with different quantization precision (16/8/4-bit), we resort to the open-source llama.cpp inference framework5.

The inference process of these LMMs consists of three key stages: visual encoding, prompt encoding, and response generation. The total inference time can be calculated as follows:

𝑇total = 𝑇VE + 𝑆prompt/𝑁prompt + 𝑆gen/𝑁gen +𝑇other (1) where 𝑇total and 𝑇VE refer to the entire time and visual encoding time consumed by a single

5https://github.com/ggerganov/llama.cpp

inference. 𝑆prompt and 𝑆gen measure the speed (tokens/s) of the prompt encoding and response generation stages, repetitively. 𝑁prompt and 𝑁gen refer to the number of prompt tokens (text tokens plus image embeddings) and generated tokens, respectively. 𝑇other means the rest negligible overheads. Note that the model loading time is not counted in the total time.

We compare the latency and performance of MobileVLM-3B@336 and Imp-3B@384 in Table

- 4. To further accelerate our Imp-3B model, we retrain a variant Imp-3B@196 with a reduced image resolution of 196×196 pixels. For latency measurement, all models are fed with the same text prompt (41 tokens) and generate the response with the same length (stops at the 64th token). For performance evaluation on SQAI [42] and MMB [40], the results of models with different quantization precision are computed using a RTX 3090 GPU for efficiency6. Note that the number of image embeddings is different for the three compared models, which are 576, 729, and 196 for MobileVLM-3B@336, Imp-3B@384, and Imp-3B@196, respectively. From the results, we have the following observations.

First, the inference latency of models on GPUs and mobile devices differs significantly in magnitude, which is about 100× difference in visual encoding and prompt encoding, and about 10× difference in response generation. Similarly, there exists a 2∼3× difference in inference speed between two mobile devices. These divergences reveal the challenges for the deployment of LMM at the edge.

Second, low-bit quantization techniques can effectively reduce model size, making it easier to be stored and run on mobile devices. Moreover, compared with the 16-bit unquantized version, 4-bit or 8-bit quantization lead to subtle performance and latency degradation.

Third, our Imp-3B@384 is slower than MobileVLM-3B@336 at each stage, which is mainly caused by more visual embeddings (729 vs. 576). To address this issue, we simply reduce the input image resolution and obtain an Imp-3B@196 variant. Imp-3B@196 achieves a comparable speed to MobileVLM-3B@336, while performing better in terms of accuracies on ScienceQA-IMG [42] and MMB [40].

To summarize, Imp-3B@196 with 4-bit quantization achieves a good balance in model storage, latency, and capability. We regard it as our default model for the following deployment.

- 5.3. Qualitative Results

To demonstrate the comprehensive capabilities of our Imp models and understand how they behave in real-world scenarios, we provide some visualized examples by comparing the response of our Imp models with the larger LLaVA-v1.5-7B model.

- Figure 3 demonstrates different skills of Imp models, including code generation, math problem solving, Chinese conversation, and medical image understanding. While the LLaVAv1.5-7B model generates unsatisfactory outputs in these cases, the Imp-3B model provides more reasonable responses, showing its superiority in VL understanding and reasoning, as well as the completeness of knowledge. For Chinese conversation, only the Imp-2B (Qwen-1.5) responds in the expected language, while the other two models fail to generate Chinese responses. This indicates the importance of the multilingual LLMs when building more user-friendly LMMs. For medical image understanding, although Imp-3B has not been deliberately trained on medical images, it exhibits impressive zero-shot capabilities.
- Figure 4 shows more diverse skill demonstrations of our Imp models, including camouflage

6Tolerable numerical differences are observed in the evaluated results obtained by llama.cpp and PyTorch frameworks.

🤠 😈

Code Generation

[Figure 9]

Convert the website in the image to HTML.

Imp-3B (Phi-2)

[Figure 10]

[Figure 11]

<!DOCTYPE html><html><head><title>My Personal Blog </title> </ head><body><nav><a href="#">Home </a><a href="#">My Trip to Gibalar </a><a href="#">How to Capture Pokemon </a> </nav><h1>My Personal Blog </h1><div id="post1"><h2>My Trip to Gibalar </ h2><p>Date: 2014-04-01 </p><img src="trip.jpg"> </div><div id="post2"><h2>How to Capture Pokemon </h2><p>Date: 2014-04-02 </ p><img src="capture.jpg"> </div> </body> </html>

[Figure 12]

🌋

LLaVA-1.5-7B

[Figure 13]

<!DOCTYPE html><html><head>\t<title>My Personal Blog </title> </ head><body>\t<h1>My Personal Blog </h1>\t<p>My trip to Gibraltar </p>\t<p>How to capture Pokemon </p> </body> </html>

[Figure 14]

## 🤠

Math

[Figure 15]

Solve the promblem in the image.

## 😈 🌋

Imp-3B (Phi-2) LLaVA-1.5-7B

[Figure 16]

[Figure 17]

To solve the quadratic equation ( x^2 + x 12 = 0 ), we can use the quadratic formula: [ x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} ] In this equation, ( a = 1 ), ( b = 1 ), and ( c =

The image shows a white background with a black equation written on it. The equation is x + 2x = 12. To solve this problem, we need to isolate the variable x by moving all the terms containing x to one side of the equation and constants to the other side. First, we can simplify the equation by combining the terms with the same variable: x + 2x = 12 Next, we can move the constant 12 to the right side of the equation: x + 2x = 12 Now, we can isolate the variable x by subtracting 2x from both sides of the equation: x = 12 - 2x Finally, we can simplify the equation by dividing both sides by 2: x = 6 Therefore, the solution to the equation is x = 6.

-12 ). Plugging these values into the formula gives us: [ x = \frac{-1 \pm \sqrt{1^2 - 4(1)(-12)}} {2(1)} ] [ x = \frac{-1 \pm \sqrt{1 + 48}}{2} ] [ x = \frac{-1 \pm \sqrt{49}}{2} ] [ x = \frac{-1 \pm 7}{2} ] This gives us two solutions: [ x = \frac{-1 + 7}{2} = 3 ] [ x = \frac{-1 - 7} {2} = -4 ] Therefore, the solutions to the quadratic equation ( x^2 + x - 12 = 0 ) are ( x = 3 ) and ( x = -4 ).

[Figure 18]

[Figure 19]

🤠 Chinese 😈

[Figure 20]

你看到了什么？

Imp-3B (Phi-2)

[Figure 21]

[Figure 22]

[Figure 23]

The image is a digital illustration that features two anthropomorphic bear characters, likely designed to represent a person and a pet. The person character is depicted as studious, with a book and a pen in hand, suggesting a scene of learning or studying. The pet character, with its head tilted to the side and a book in its paws, appears to be in a state of relaxation or contemplation, possibly enjoying a moment of leisure or daydreaming. The text "2000 YEARS LATER" implies a humorous or ironic commentary on the passage of time, contrasting the youthful and carefree demeanor of the pet with the more mature and serious attitude of the person. The Chinese characters at the bottom of the image could be providing additional context or commentary related to the illustration.

😈 🌋

Imp-2B (Qwen1.5)

[Figure 24]

🇨🇳

我看到了两个可爱的卡通⼈物，⼀个在阅读，另⼀个在写作。

[Figure 25]

Translation:

I see two cute cartoon characters, one reading and the other writing.

LLaVA-1.5-7B

[Figure 26]

The image features two cartoon characters sitting at a table, each holding a book. The characters are white and have big eyes, giving them a cute appearance. The table is surrounded by several books, with some placed on the table and others scattered around the area. The scene appears to be a study or reading environment, with the characters engaging in a shared activity.

## 🤠 😈

Medical What's notable in this X-ray image?

[Figure 27]

Imp-3B (Phi-2)

[Figure 28]

[Figure 29]

[Figure 30]

In this X-ray image, there is a notable presence of a pacemaker, which is a small device that helps regulate the heartbeat. The pacemaker is located in the upper left side of the chest, just below the collarbone.

## 🌋

LLaVA-1.5-7B

[Figure 31]

The notable aspect of this X-ray image is the presence of a pacemaker in the chest area of the person.

- Figure 3 | Comprehensive skill demonstrations of Imp, including code generation, math problem solving, Chinese conversation, and medical image understanding.

🤠

[Figure 32]

Camouflaged Object Perception

Is there a face in the image?

😈

Imp-3B (Phi-2)

[Figure 33]

[Figure 34]

Yes, there is a face in the image. It is formed by the tree's branches, which are arranged to resemble the shape of a face.

🌋

LLaVA-1.5-7B

[Figure 35]

Yes, there is a face in the image. It is a tree with a face carved into it, and it is located on a beach.

🤠

[Figure 36]

Celebrity Recognition

What's this guy's name?

😈

Imp-3B (Phi-2)

[Figure 37]

The individual in the image is Tyrion Lannister, a character from the television series "Game of Thrones." Tyrion is known for his wit, intelligence, and complex character, often portrayed as a master manipulator and a key player in the political intrigue of the series. He is one of the main characters and is recognized for his distinctive hairstyle, which is often styled in a way that frames his face, and his attire, which is a mix of practical and ornate clothing that reflects his status and personality.

[Figure 38]

🌋

LLaVA-1.5-7B

[Figure 39]

The guy's name is Gendry Baratheon, as he is portrayed by actor Joe Dempsie in the HBO series "Game of Thrones."

### 🤠 😈

Chart Understanding

According to this chart, which smartphone vendor had the highest market share in Q3, 2022?

[Figure 40]

[Figure 41]

Imp-3B (Phi-2)

[Figure 42]

Based on the chart, Apple had the highest market share in Q3, 2022, with a black bar that reaches the 50% mark.

### 🌋

LLaVA-1.5-7B

[Figure 43]

In Q3, 2022, Apple had the highest market share among smartphone vendors, with 32.5% of the market.

🤠

Object Grounding

Provide the bounding box coordinate of the car with the least common color.

[Figure 44]

😈 Imp-3B (Phi-2) 🤖 LLaVA-1.5-7B

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

- Figure 4 | More diverse skill demonstrations of Imp, including camouflage perception, celebrity recognition, chart understanding, and object grounding.

How many bananas are there?

[Figure 50]

🤠

What does this plaque say?

[Figure 51]

🤠

[Figure 52]

😈

There are a total of twelve bananas visible in the image.

The plaque in the image is inscribed with the following text: "IN MEMORY OF THE MEN WHO SERVED IN THE WORLD WAR II. A NATION CONSCIOUS IN LIBERTY AND JUSTICE." This text commemorates the individuals who served in World War II and reflects on the values of liberty and justice that the nation holds dear. The plaque is a tribute to the sacrifices made by these individuals and serves as a reminder of the nation's commitment to these principles.

Imp-3B (Phi-2)

[Figure 53]

😈 Imp-3B (Phi-2)

[Figure 54]

[Figure 55]

Failed

HERE IN THE PRESENCE OF WASHINGTON AND LINCOLN, ONE THE EIGHTEENTH CENTURY FATHER AND THE OTHER THE NINETEENTH CENTURY PRESERVER OF OUR NATION, WE HONOR THOSE TWENTIETH CENTURY AMERICANS WHO TOOK UP THE STRUGGLE DURING THE SECOND WORLD WAR AND MADE THE SACRIFICES TO PERPETUATE THE GIFT OUR FOREFATHERS ENTRUSTED TO US: A NATION CONCEIVED IN LIBERTY AND JUSTICE.

The text reads:

- Figure 5 | Failure cases of Imp, including the hard examples about dense object counting, and long-text OCR.

[Figure 56]

[Figure 57]

- (a) ImpChat-Web (b) ImpChat-Mobile
- Figure 6 | The UI demonstrations of ImpChat on multiple platforms.

perception, celebrity recognition, chart understanding, and object grounding. In these cases,

- Imp-3B can consistently provide correct and coherent responses, while LLaVA-v1.5-7B gets responses of relatively poorer quality. Those comparisons demonstrate the strength of Imp models in solving diverse tasks and assisting users with various intentions.

Finally, we showcase some failure cases of Imp models in Figure 5, including the hard examples about dense object counting, and long-text OCR. These failure cases demonstrate the weakness of Imp models and point out the directions for future improvement.

###### 6. ImpChat: LMM Assistants on Multiple Platforms

Based on the Imp models we trained and the deployment strategies we explored, we have developed a software suite of conversational AI interfaces named ImpChat, which is dedicated to providing users with an easy-to-use LMM assistant in offline environments. ImpChat is capable of understanding text and images in a wide range of conversational contexts, and generating coherent responses that are engaging and informative. As shown in Figure 6, ImpChat provides specialized interfaces for different platforms, i.e., web and mobile devices.

For web users, we offer ImpChat-Web, a web-based chat assistant. The frontend of ImpChatWeb is implemented by ChuanhuChat7, a popular UI framework for open-source LLM deployment, which is also developed by our team. The backend is served by a server with multiple GPUs, with an optimized load balancing strategy to offer a high throughput and low latency for concurrent user accesses.

For mobile phone users, we develop ImpChat-Mobile based on the MLC framework8. ImpChat-Mobile is an application that can be installed across different platforms such as Android and iOS. ImpChat-Mobile is run in an offline environment without an Internet connection,

- 7https://github.com/GaiZhenbiao/ChuanhuChatGPT
- 8https://github.com/mlc-ai/mlc-llm

which is more favorable for data privacy scenarios. We currently implement ImpChat-Mobile for Android only.

###### 6.1. Conclusion and Future Work

In this paper, we present a family of lightweight LMMs named Imp. To balance the model efficiency and efficacy, we conduct a systematic exploration and establish a comprehensive roadmap from the aspects of model architecture, training strategy, and training data. Based on the empirical studies and different lightweight LLMs, we obtained Imp models at the 2B∼4B scales. Notably, our Imp-3B model steadily outperforms all the existing lightweight LMMs of similar size and even surpasses the state-of-the-art LMMs at the 13B scale. Our optimized Imp models can run on a mobile phone with a high inference speed.

In the future, we consider improving the model capability from following aspects: 1) appending high-quality training data for specific tasks, such as Chinese OCR and set-of-marks instruct-tuning data, 2) introducing more effective training strategy such as knowledge distillation and human-aligned preference optimization, 3) exploring more efficient model compression algorithms such as 1-bit and 1.58-bit quantization methods, and 4) supporting broader input and output modalities such as audio and 3D. On the application level, we will explore the usage Imp models in more diverse scenarios such as UI agent and robot manipulation. We hope our imp project could establish an ecosystem to facilitate future research and applications with lightweight LMMs.

###### Acknowledgements

This work was supported in part by the National Natural Science Foundation of China under Grants No.62125201, No.62072147, No.62020106007, No.62206082, and in part by the Zhejiang Provincial Natural Science Foundation of China under Grants No.LR22F020001 and No.DT23F020007.

###### References

- [1] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

- [2] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al. Qwen technical report. Technical report, Alibaba Group, 2023. https://arxiv.org/abs/2303

.08774.

- [3] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.

- [4] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan,

- R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin,
- S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, pages 1877–1901, 2020.

- [5] G. H. Chen, S. Chen, R. Zhang, J. Chen, X. Wu, Z. Zhang, Z. Chen, J. Li, X. Wan, and B. Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model. arXiv preprint arXiv:2402.11684, 2024.

- [6] K. Chen, Z. Zhang, W. Zeng, R. Zhang, F. Zhu, and R. Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023.

- [7] L. Chen, J. Li, X. Dong, P. Zhang, C. He, J. Wang, F. Zhao, and D. Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023.

- [8] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, and E. P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https://lmsys.org/blog/2023-03-30-vic una/.
- [9] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.

- [10] X. Chu, L. Qiao, X. Lin, S. Xu, Y. Yang, Y. Hu, F. Wei, X. Zhang, B. Zhang, X. Wei, and C. Shen. Mobilevlm : A fast, strong and open vision language assistant for mobile devices, 2023.
- [11] W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. A. Li, P. Fung, and S. C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. ArXiv, abs/2305.06500, 2023. URL https://api.semanticscholar.org/CorpusID: 258615266.

- [12] S. Deshmukh, B. Elizalde, R. Singh, and H. Wang. Pengi: An audio language model for audio tasks. Advances in Neural Information Processing Systems, 36:18090–18108, 2023.

- [13] J. Devlin, M. Chang, K. Lee, and K. Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT 2019, pages 4171–4186, 2019.

- [14] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

- [15] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, Z. Qiu, W. Lin, J. Yang, X. Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

- [16] P. Gao, R. Zhang, C. Liu, L. Qiu, S. Huang, W. Lin, S. Zhao, S. Geng, Z. Lin, P. Jin, et al. Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. arXiv preprint arXiv:2402.05935, 2024.

- [17] G. Gemma team. Gemma: Open models based on gemini research and technology, 2024.
- [18] R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, and I. Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15180–15190, 2023.

- [19] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017.

- [20] S. Gunasekar, Y. Zhang, J. Aneja, C. C. T. Mendes, A. Del Giorno, S. Gopi, M. Javaheripi, P. Kauffmann, G. de Rosa, O. Saarikivi, et al. Textbooks are all you need. arXiv preprint arXiv:2306.11644, 2023.

- [21] D. Gurari, Q. Li, A. J. Stangl, A. Guo, C. Lin, K. Grauman, J. Luo, and J. P. Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617, 2018.

- [22] M. He, Y. Liu, B. Wu, J. Yuan, Y. Wang, T. Huang, and B. Zhao. Efficient multimodal learning from data-centric perspective. arXiv preprint arXiv:2402.11530, 2024.

- [23] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

- [24] D. A. Hudson and C. D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.

- [25] M. Javaheripi, S. Bubeck, M. Abdin, J. Aneja, S. Bubeck, C. C. T. Mendes, W. Chen, A. Del Giorno, R. Eldan, S. Gopi, et al. Phi-2: The surprising power of small language models. Microsoft Research Blog, 2023.

- [26] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

- [27] K. Kafle, B. Price, S. Cohen, and C. Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5648–5656, 2018.

- [28] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

- [29] S. Karamcheti, S. Nair, A. Balakrishna, P. Liang, T. Kollar, and D. Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. arXiv preprint arXiv:2402.07865, 2024.

- [30] A. Kembhavi, M. Salvato, E. Kolve, M. Seo, H. Hajishirzi, and A. Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–

251. Springer, 2016.

- [31] LAION-GPT-V. https://huggingface.co/datasets/laion/gpt4v-dataset, 2023.
- [32] J. Li, D. Li, S. Savarese, and S. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.

- [33] K. Li, Y. He, Y. Wang, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang, and Y. Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.

- [34] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.

- [35] Y. Li, Y. Zhang, C. Wang, Z. Zhong, Y. Chen, R. Chu, S. Liu, and J. Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814, 2024.

- [36] J. Lin, H. Yin, W. Ping, Y. Lu, P. Molchanov, A. Tao, H. Mao, J. Kautz, M. Shoeybi, and S. Han. Vila: On pre-training for visual language models. arXiv preprint arXiv:2312.07533, 2023.

- [37] H. Liu, C. Li, Y. Li, and Y. J. Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.

- [38] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.

- [39] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024. URL https://llava-vl.github.io/blog/2 024-01-30-llava-next/.
- [40] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.

- [41] H. Lu, W. Liu, B. Zhang, B. Wang, K. Dong, B. Liu, J. Sun, T. Ren, Z. Li, Y. Sun, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.

- [42] P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.-C. Zhu, O. Tafjord, P. Clark, and A. Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.

- [43] C. Lyu, M. Wu, L. Wang, X. Huang, B. Liu, Z. Du, S. Shi, and Z. Tu. Macaw-llm: Multimodal language modeling with image, audio, video, and text integration. arXiv preprint arXiv:2306.09093, 2023.

- [44] A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E. Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.

- [45] M. Mathew, D. Karatzas, and C. Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.

- [46] M. Mathew, V. Bagal, R. Tito, D. Karatzas, E. Valveny, and C. Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022.

- [47] Microsoft. Phi-3 technical report: A highly capable language model locally on your phone, 2024.
- [48] OpenAI. Chatgpt: Optimizing language models for dialogue. https://openai.com/b log/chatgpt, 2023. Accessed: 2023.
- [49] OpenAI. Gpt-4 technical report. Technical report, OpenAI, 2023.
- [50] OpenAI. Gpt-4v(ision) system card. Technical report, OpenAI, 2023.
- [51] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

- [52] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, et al. Improving language understanding by generative pre-training. 2018.
- [53] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

- [54] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

- [55] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020.

- [56] Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 36, 2024.

- [57] O. Sidorov, R. Hu, M. Rohrbach, and A. Singh. Textcaps: a dataset for image captioning with reading comprehension. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 742–758. Springer, 2020.

- [58] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.

- [59] Y. Su, T. Lan, H. Li, J. Xu, Y. Wang, and D. Cai. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355, 2023.

- [60] G. Team. Gemini: A family of highly capable multimodal models. Technical report, Google,

2023. https://storage.googleapis.com/deepmind-media/gemini/gemini_1_r eport.pdf.

- [61] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- [62] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

- [63] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and

I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

- [64] W. Wang, Q. Lv, W. Yu, W. Hong, J. Qi, Y. Wang, J. Ji, Z. Yang, L. Zhao, X. Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079,

- 2023.

[65] H. Wei, L. Kong, J. Chen, L. Zhao, Z. Ge, E. Yu, J. Sun, C. Han, and X. Zhang. Small language model meets with reinforced vision vocabulary. arXiv preprint arXiv:2401.12503,

- 2024.

- [66] C. Wu, S. Yin, W. Qi, X. Wang, Z. Tang, and N. Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023.

- [67] A. Yang, B. Xiao, B. Wang, B. Zhang, C. Bian, C. Yin, C. Lv, D. Pan, D. Wang, D. Yan, et al. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023.

- [68] Z. Yang, L. Li, J. Wang, K. Lin, E. Azarnasab, F. Ahmed, Z. Liu, C. Liu, M. Zeng, and L. Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023.

- [69] Q. Ye, H. Xu, J. Ye, M. Yan, H. Liu, Q. Qian, J. Zhang, F. Huang, and J. Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. arXiv preprint arXiv:2311.04257, 2023.

- [70] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

- [71] Z. Yuan, Z. Li, and L. Sun. Tinygpt-v: Efficient multimodal large language model via small backbones. arXiv preprint arXiv:2312.16862, 2023.

- [72] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pretraining. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023.

- [73] A. Zhang, L. Zhao, C.-W. Xie, Y. Zheng, W. Ji, and T.-S. Chua. Next-chat: An lmm for chat, detection and segmentation. arXiv preprint arXiv:2311.04498, 2023.

- [74] D. Zhang, S. Li, X. Zhang, J. Zhan, P. Wang, Y. Zhou, and X. Qiu. Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities. arXiv preprint arXiv:2305.11000, 2023.

- [75] H. Zhang, X. Li, and L. Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.

- [76] P. Zhang, G. Zeng, T. Wang, and W. Lu. Tinyllama: An open-source small language model, 2024.
- [77] B. Zhu, B. Lin, M. Ning, Y. Yan, J. Cui, H. Wang, Y. Pang, W. Jiang, J. Zhang, Z. Li, et al. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. arXiv preprint arXiv:2310.01852, 2023.

- [78] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

- [79] M. Zhu, Y. Zhu, X. Liu, N. Liu, Z. Xu, C. Shen, Y. Peng, Z. Ou, F. Feng, and J. Tang. A comprehensive overhaul of multimodal assistant with small language models. arXiv preprint arXiv:2403.06199, 2024.

- [80] Y. Zhu, M. Zhu, N. Liu, Z. Ou, X. Mou, and J. Tang. Llava-phi: Efficient multi-modal assistant with small language model, 2024.

