# arXiv:2504.00595v2[cs.CL]2Apr2025

## Open-Qwen2VL: Compute-Efficient Pre-Training of Fully-Open Multimodal LLMs on Academic Resources

Weizhi Wang1, Yu Tian2, Linjie Yang2, Heng Wang3, Xifeng Yan1 1UC Santa Barbara, 2 Seed Vision Team, ByteDance, 3 Nvidia Research

### Abstract

The reproduction of state-of-the-art multimodal LLM pre-training faces barriers at every stage of the pipeline, including high-quality data filtering, multimodal data mixture strategies, sequence packing techniques, and training frameworks. We introduce Open-Qwen2VL, a fully open-source 2B-parameter Multimodal Large Language Model pre-trained efficiently on 29M image-text pairs using only 220 A100-40G GPU hours. Our approach employs low-to-high dynamic image resolution and multimodal sequence packing to significantly enhance pretraining efficiency. The training dataset was carefully curated using both MLLM-based filtering techniques (e.g., MLM-Filter) and conventional CLIP-based filtering methods, substantially improving data quality and training efficiency. The Open-Qwen2VL pre-training is conducted on academic level 8xA100-40G GPUs at UCSB on 5B packed multimodal tokens, which is 0.36% of 1.4T multimodal pre-training tokens of Qwen2-VL. The final instruction-tuned Open-Qwen2VL outperforms partially-open state-ofthe-art MLLM Qwen2-VL-2B on various multimodal benchmarks of MMBench, SEEDBench, MMstar, and MathVista, indicating the remarkable training efficiency of Open-Qwen2VL. We open-source all aspects of our work, including compute-efficient and data-efficient training details, data filtering methods, sequence packing scripts, pre-training data in WebDataset format, FSDP-based training codebase, and both base and instruction-tuned model checkpoints. We redefine "fully open" for multimodal LLMs as the complete release of: 1) the training codebase, 2) detailed data filtering techniques, and 3) all pre-training and supervised finetuning data used to develop the model.

Website https://victorwz.github.io/Open-Qwen2VL

Code https://github.com/Victorwz/Open-Qwen2VL

Models https://huggingface.co/weizhiwang/Open-Qwen2VL

Data https://huggingface.co/datasets/weizhiwang/Open-Qwen2VL-Data

#### 1. Introduction

The Multimodal Large Language Models (MLLMs) [4, 9, 35, 44, 48] present strong emergent capabilities on multimodal understanding and visual reasoning, eliciting the Artificial Intelligence Applications to comprehend and analyze images, charts, and PDF documents. Different from conventional Vision-Language Models (VLMs) [19, 37], trained on image-text caption data from scratch with small model size, the MLLMs are typically constructed on a well-trained text-only LLM and then continually pre-trained on diverse large-scale multimodal data. However, recent state-of-the-art MLLMs are neither “fully-open“ to the community for reproduction nor compute-friendly to academic institutions with limited GPUs. In Table 1, we compare the openness of recent SOTA MLLMs of VILA [28], MM1 [54], Ideflics [24], BLIP-3 [49], Llama-3.2Vision [13], Phi-3.5-Vision [1], and Qwen2VL [44]. Even if most of the SOTA MLLMs release their base or instruction-tuned model checkpoints, their killer secrets of data filtering techniques, sequence packing scripts, pre-training data, training codebase, etc are completely close-source, in which they even hide such technical details in their technical reports.

In this work, we introduce Open-Qwen2VL, a 2B-parameter MLLM which outperforms closesource Qwen2-VL-2B on various multimodal benchmarks and achieves outstanding compute efficiency. Open-Qwen2VL is pre-trained on approximately 5B well-curated high-quality caption data tokens, which is 0.36% of 1.4T multimodal tokens of Qwen2-VL [44] pre-training. Such remarkable data-efficiency enables us to perform the pre-training on academic-level computing resources of 8*A100-40G GPUs. In addition, we conduct compressive visual projector [50] to scale-down 729 image patches to 144 visual tokens and perform multimodal sequence packing to further enhance the pre-training efficiency.

We perform comprehensive ablation studies on pre-training data mixture strategies and data filtering models. The best pre-training data consists of CC3M-CC12M-SBU [36, 40] caption dataset curated by CLIP and DataComp-Medium caption dataset curated by both the DFN-CLIP and MLM-Filter. Adopting efficient MLLM as the data filtering model significantly enhances the model capabilities on various benchmarks. Additionally, we scale up the visual supervised fine-tuning (SFT) data to 10M level [16] to further enhance the model capabilities.

We open-source everything to the community to help easy and convenient reproductions to our model Open-Qwen2VL, including the data filtering details, sequence packing scripts, pre-training data in webdataset format, training codebase based on FSDP, and both base model and instruction-tuned model checkpoints. Meanwhile, our open-source codebase is the first comprehensive solution that supports all stages of Multimodal Large Language Model training, including large-scale caption data preparation, quality score generation, data filtering, multimodal sequence packing, pre-training, supervised fine-tuning, and evaluations on multimodal benchmarks. We redefine "fully open" for multimodal LLMs as the complete release of: 1) the training codebase, 2) detailed data filtering techniques, and 3) all pre-training and supervised fine-tuning data used to develop the model. We wish to demonstrate that the research on pretraining is not only a game for giant tech companies and encourage the academic community to work on pre-training data and pipeline research even with very limited computing resources.

#### 2. Compute-Efficient Multimodal Pre-Training

##### 2.1. Dataset Choices and High-Quality Data Filtering

The current advanced multimodal LLMs are continually pre-trained on a large-scale imagetext caption dataset. In addition to image-text caption dataset, some of latest MLLMs like VILA[28], MM1[34], DeepSeek-VL2 [48] also mix the image-text interleaved data with caption data for multimodal pre-training. Mixing image-text caption data and interleaved data will

Data Filtering Techniques

Sequence Packing Scripts

Pre-Training Data

Pre-Training Codebase

Base Model Checkpoint

SFT Data

Instruct Model Checkpoint

Models

VILA None None Open Open Open Open Open MM1 Closed Closed Closed Closed Closed Closed Closed Ideflics Open Open Open Open Open Open Open BLIP-3 Closed Closed Open Closed Open Closed Open Llama-3.2-Vision Closed Closed Closed Closed Open Closed Open Phi-3.5-Vision Closed Closed Closed Closed Closed Closed Open Qwen2VL Closed Closed Closed Closed Open Closed Open

Open-Qwen2VL Open Open Open Open Open Open Open

- Table 1 | Comparisons of openness between several state-of-the-art MLLMs.

enhance the multimodal in-context learning and multi-image reasoning capabilities of MLLMs. However, MM1 [34] demonstrates introducing image-text interleaved documents into pretraining data will reduce the zero-shot single-image reasoning and understanding capabilities of base MLLMs. Thus, to control the scale of pre-training data and ensure the pre-training efficiency, Open-Qwen2VL focuses on the pre-training paradigm on image-text caption data only.

To motivate the easy reproduction to our work from the community, we choose 4 most popular image-text caption datasets shown in Table 2, which are widely used in open-source vision-language model pre-training. BLIP-1 [26] releases the high-quality caption data curated from a combination of CC3M-CC12M-SBU (CCS) using CLIP-based [17, 37] filtering. LAION400M [39] implements a strict 0.3 threshold based on CLIP image-text cosine similarity to curate its high-quality dataset of 400M image-text caption pairs. We download the CCS and LAION caption datasets based on the release image-urls using img2dataset [7] tool. We only download 15M LAION data to perform controlled-size data mixture ablation studies in Section 2.5.

Secondly, we choose DataComp-Medium-128M [15] as another pre-training data choice. Based on the leaderboard of DataComp medium filtering track performance, Data-FilteringNetwork (DFN) [14] is the top-1 independent data filter on the leaderboard1. We successfully download 99.8M out of 128M original released 128M DataComp-Medium data. Then we adopt the official resharder script2 to select the DFN-curated high-quality subset based on the released top-15% data uids from DFN3. It is worth noting that the DFN only releases the top-15% curated data rather than the DFN model checkpoint. Thus, it is impossible to change the retained data fraction based on the quality scores generated by DFN-model. For DataComp-DFN high-quality dataset, finally we get 15M image-text caption data.

The MLLM-based data filtering method emerges since the introduction of MLM-Filter [46], in which these methods adopt efficient MLLM as the high-quality data filter instead of CLIP model. MLM-Filter provides four distinct image-text data quality metric for high-quality data filtering, including image-text matching (ITM), object detail fulfillment (ODF), caption text quality (CTQ), and semantic understanding (SU). Based on the conclusions from ATIQE [18], the Semantic Understanding (SU) quality metric yields the best performance for MLLMs trained on the high-quality data curated from such metric. Thus, we generate the SU quality scores for DataComp-Medium data using mlm-filter-qwen2.5-1.5b-gpt4o4 data filtering model and set filtering score threshold as 85 out of 100. With such threshold, We get 8M MLM-Filter

- 1https://www.datacomp.ai/dcclip/leaderboard.html
- 2https://github.com/mlfoundations/datacomp/blob/main/resharder.py
- 3https://huggingface.co/datasets/apf1/datafilteringnetworks_2b/blob/main/datacomp_med

ium_dfn_20m_inds.npy

- 4https://huggingface.co/weizhiwang/mlm-filter-qwen2.5-1.5b-gpt4o

Filtering Model

#Image-Text Pairs

ID Dataset

Resources

- 1 CCS CLIP 8.5M https://github.com/salesforce/BLIP
- 2 DataComp DFN 15M huggingface:apf1/datafilteringnetworks_2b
- 3 LAION CLIP 15M https://github.com/salesforce/BLIP
- 4 DataComp MLMFilter & DFN 19.9M huggingface:weizhiwang/Open-Qwen2VL-Data

- Table 2 | Image-Text Caption Datasets for Open-Qwen2VL pre-training.

curated data and union them with DFN-15M. After deduplication, we get 19.9M high-quality data.

##### 2.2. Model Architecture with Low-to-High Image Resolution

We adopt a simple architecture with Qwen2.5-1.5B-Instruct LLM Backbone [43], Adaptive Average-Pooling Visual Projector [50], and SigLIP-SO-400M Vision Encoder [53]. Specifically, the Adaptive Average-Pooling Visual Projector contains an Adaptive Average-Pooling layer followed by a two-layer MLP. With the Adaptive Average-Pooling layer, we can scale the 729 output visual patches from SigLIP to any resolution. We adopt 144 visual tokens for representing an image in the pre-training stage and scale up the resolution to vanilla 729 visual tokens in SFT stage. Such low-to-high image resolution significantly enhances the MLLM pre-training efficiency and does not hurt the high-resolution image understanding of the final MLLM after SFT stage.

Open-Qwen2VL does not adopt advanced designs of 2d-Multimodal RoPE [44] and naive dynamic resolution [44] to save computes and ensure the training efficiency. Moreover, for academic computing resources, downloading images and saving in original resolution require huge disk space, which is unavailable in most of academic institutions. During our data downloading process with img2dataset [7], we resize the smaller side of the image to 512 pixels and keep the aspect ratio, which makes us not able to adopt naive dynamic resolution in the pre-training stage.

For both the pre-training and SFT stages, we freeze the parameters of vision encoder and make the parameters of projector and LLM backbone trainable to save more computes. However, recent studies [44, 54] demonstrate that making the vision encoder trainable can further enhance the visual understanding capabilities of the MLLMs. We leave it as an ablation study for investigations.

CE Loss

CE Loss

Large Language Model

Large Language Model

AvgPooling Projector

Word Embeddings

MLP Projector

Word Embeddings

| | |
|---|---|
|Caption| |

| | |
|---|---|
|Caption| |

SigLIP ViT Image Encoder

SigLIP ViT Image Encoder

| | |
|---|---|
|[Figure 1]| |

| | |
|---|---|
|[Figure 2]| |

Pre-Training Stage SFT Stage

- Figure 1 | Model Architecture of Open-Qwen2VL. Here 𝑀 = 729, 𝑁 = 144 are the number of image patch tokens and number of projected visual tokens during the pre-training stage, respectively.

Algorithm 1 Multimodal Sequence Packing

Require:

- 1: D: Set of image-text caption data
- 2: 𝐿: Maximum context length
- 3: 𝑝: Padding token

Ensure: Packed sequences within context length 𝐿

- 4: function PA C KSE Q U E N C E S(D, 𝐿, 𝑝)
- 5: 𝐼 ← ∅ ⊲ Initialize items dictionary
- 6: for all 𝑑 ∈ D do
- 7: (𝑇𝑑,𝑉𝑑) ← ProcessCaption(𝑑) ⊲ Get text tokens and visual tokens
- 8: 𝑙𝑒𝑛𝑑 ← |𝑇𝑑| + |𝑉𝑑| ⊲ |𝑉𝑑| = 144
- 9: 𝐼[𝑑] ← 𝑙𝑒𝑛𝑑
- 10: end for
- 11: 𝐼 ← SortByLength(𝐼) ⊲ Sort in descending order
- 12: 𝐵 ← ∅ ⊲ Initialize bins
- 13: for all (𝑑, 𝑙𝑒𝑛𝑑) ∈ 𝐼 do
- 14: 𝑝𝑙𝑎𝑐𝑒𝑑 ← false
- 15: for all 𝑏𝑖𝑛 ∈ 𝐵 do
- 16: if (𝑑′,𝑙𝑒𝑛′)∈𝑏𝑖𝑛 𝑙𝑒𝑛′+ 𝑙𝑒𝑛𝑑 ≤ 𝐿 then
- 17: 𝑏𝑖𝑛.append((𝑑, 𝑙𝑒𝑛𝑑))
- 18: 𝑝𝑙𝑎𝑐𝑒𝑑 ← true
- 19: break
- 20: end if
- 21: end for
- 22: if ¬𝑝𝑙𝑎𝑐𝑒𝑑 then
- 23: 𝐵.append({(𝑑, 𝑙𝑒𝑛𝑑)})
- 24: end if
- 25: end for
- 26: 𝑃 ← ∅ ⊲ Initialize packed sequences
- 27: for all 𝑏𝑖𝑛 ∈ 𝐵 do
- 28: 𝑇𝑏𝑖𝑛 ← ∅ ⊲ Concatenated text tokens
- 29: 𝑉𝑏𝑖𝑛 ← ∅ ⊲ Concatenated PIL images
- 30: for all 𝑑 ∈ 𝑏𝑖𝑛 do
- 31: 𝑇𝑏𝑖𝑛.append(𝑇𝑑)
- 32: 𝑉𝑏𝑖𝑛.append(𝑉𝑑)
- 33: end for
- 34: if |𝑇𝑏𝑖𝑛| < 𝐿 then
- 35: 𝑇𝑏𝑖𝑛.pad(𝑝, 𝐿) ⊲ Pad to context length
- 36: end if
- 37: 𝑃.append((𝑇𝑏𝑖𝑛,𝑉𝑏𝑖𝑛))
- 38: end for return 𝑃
- 39: end function

##### 2.3. Multimodal Sequence Packing

Because the large-scale image-text data are varied in its length, simply batchfying a set of examples based on similar length and padding them to the longest sequence will lead to a large portion of padding tokens in each training batch. Such a large amount of padding tokens will result in heavy compute-waste and training inefficiency. Thus, we introduce multimodal sequence packing to regroup the image-text caption data into sequence groups closest to 4096 context length.

The algorithm for the multimodal sequence packing is presented in Algorithm 1. Since we download and pack all image-text caption data into webdataset format and each webdataset tar file contains exactly 10k image-text caption data, the proposed multimodal sequence packing

Data Mixture 1 + 2 1 + 3 1 + 2 + 3 1 + 4 (23.5M) (23.5M) (38.5M) (28.4M) General Benchmark

Benchmark

MMMUval 38.9 37.2 36.7 38.0 MMBenchdev 75.6 75.9 75.9 77.3 SEEDBench − Imgdev 68.9 69.6 68.9 68.7 MMStar 39.6 41.7 41.7 41.3

OCR VQA

AI2Dtest 56.3 55.5 57.3 56.8 TextVQAval 55.1 57.4 57.4 57.0

Math Reasoning MathVistatestmini 28.6 28.1 28.7 28.6 Hallucination

POPE 79.2 80.1 77.7 80.1 Average 55.3 55.4 55.5 56.0

- Table 3 | Benchmark performance of MLLMs pre-trained on different pre-train data mixture and fine-tuned on controlled LLaVA-665k instructions. Remarks for each dataset: 1: CCS-CLIP; 2: DataComp-DFN; 3: LAION-CLIP; 4: DataComp-MLM-Filter & DFN.

intends to regroup such 10k pairs into a set of multimodal sequences with 4096 context length.

The multimodal sequence packing involves three major steps: computing the multimodal length of each image-text caption sample, regroup data into several bins in which the total length of each bin is closest to 4096, and concatenate the input_ids vectors and pillow-format images. We adopt First-fit-decreasing (FFD) bin packing algorithm [20] to pack each imagetext caption data into several bins. We also follows LLaVA to insert an <image> placeholder token at the beginning of each image-text caption. We use the default end-of-sequence token, <|im_end|> token as the separator between each image caption text.

We store each packed multimodal sequence to a pickle file, because pickle support storing data in different formats like pillow-image and torch input_ids tensor in one file. Finally, each pickle file contains the following dictionary for each packed multimodal sequence:

- • “images“: a list of pillow image objects;
- • “input_ids“: torch Long Tensor with image placeholder token;
- • “lengths“: a list of integers to record the multimodal length of each image-text caption data.

##### 2.4. Training Infrastructure and Codebase

We develop our training codebase based on Prismatic-VLM [21]. The original codebase only supports SFT on single-image instructions and we heavily modify its dataloader and batch preparation to support the multimodal packed sequences with multiple-images in one sequence. We retain its Fully-Sharded Distributed Parallel (torch-FSDP) trainer, which we find that it significantly accelerates the training compared with LLaVA codebase using DeepSpeedZero3. Although FSDP and DeepSpeed-Zero3 utilize the same model sharding algorithm, our FSDP-based implementation achieves approximately 17% faster for each training step than the DeepSpeed implementation, consistent with findings reported by Karamcheti et al. [21].

##### 2.5. Ablations on the Data Mixture

After preparing and sequentially-packing the 4 image-text caption dataset, we conduct ablation studies to investigate the effects of different data mixtures to the performance of final MLLMs. Since there is 16 combinations between the four datasets, we only consider 4 combinations. The CCS-CLIP data is fixed and we incrementally add other three datasets with it. For each dataset group, we pre-train the MLLM for one epoch on packed multimodal sequences and then fine-tune the base MLLM on LLaVA-665k instruction dataset [30]. The training details and hyperparameters are available in Appendix Table 7. Then we evaluate each model ablations on multimodal benchmarks of AI2D-test [22], TextVQA-val [42], POPE [27], MMMU-val [52], MMBench-v1.0-dev [31], SEEDBench-imge-dev [25], MMStar [8], and MathVista-test-mini [32].

Results. The benchmark results of each pre-trained and fine-tuned MLLM using different data mixtures are presented in Table 3. Since the DataComp-DFN and LAION are both web-crawled data and adopt similar CLIP-based data filtering techniques, the data mixtures of these two dataset with CCS achieves very similar model performance. Moreover, simply mixing three CCS-CLIP, DataComp-DFN, and LAION-CLIP does not achieve better performance, which might be caused by the high data homogeneity between DataComp-DFN data and LAIONCLIP data. Surprisingly, adding very small amount of high-quality data (5M) curated by a different efficient-MLLM based data filter, MLM-Filter can significantly enhance the model performance, achieving +0.5 average performance improvements. We suppose that MLLMbased data filter may introduce a different data distribution into the pre-training set, which brings new knowledge for enhancing the MLLM capabilities.

Finally, the pre-training of Open-Qwen2VL on the best data mixture takes about 220 A100-40G GPU hours, and the SFT on LLaVA-665k instructions takes 48 A100-40G GPU hours.

#### 3. Scaling-Up Supervised Fine-Tuning

##### 3.1. SFT Dataset

After the ablation studies on the pre-training mixture, we further scale-up the visual SFT data from LLaVA-665k [30] to MAmmoTH-VL-10M [16] to further enhance the understanding and reasoning capabilities of MLLM. We only use the 10M single-image subset for visual SFT and do not include the additional LLaVA-OneVision-2M for further SFT on mixed image and video data. The MAmmoth-VL-10M data requires over 200GB CPU memory if the original LLaVA-style dataloader is adopted to load the full 10M json file data into memory in distributed multi-process. To comply with the limited CPU memory of our server, we store each data sample in the 10M full json data into single json file, and meanwhile generate a 10M-indices file for loading into the memory. Each index contains the path to the data sample json, the boolean value of text-only or image-text data, and the pre-computed image-text data length for batchfying. The SFT hyperparameters also follow Appendix Table 7.

##### 3.2. Scaling Effects and Results

We save the checkpoints for every 2M sft instructions, which is 15625 steps under the batch size of 128. We illustrate the benchmark performance of each saved checkpoint in Figure 2. We can conclude that scaling-up SFT remarkably improves model performance on various multimodal benchmarks. Most of the benchmarks like POPE, MMMU, MMBench, and SEEDBench performance converges at the SFT scale of 8M instructions and do not improve for the final 2M data. The curves of TextVQA and MathVista performance vary from others, which present a steady

###### AI2D

TextVQA

###### POPE

###### MMMU

- 36

- 37

- 38

- 39

- 40

- 41

- 42

- 79

- 80

- 81

- 82

- 83

- 84

- 85

- 86

64

58

62

56

60

Accuracy

Accuracy

Accuracy

Avg-F1

54

58

LLaVA-665k

LLaVA-665k

56

52

LLaVA-665k

LLaVA-665k

54

50

MAmmoTH-VL-10M

MAmmoTH-VL-10M

MAmmoTH-VL-10M

MAmmoTH-VL-10M

52

2M 4M 6M 8M 10M Training Data Scale

2M 4M 6M 8M 10M Training Data Scale

2M 4M 6M 8M 10M Training Data Scale

2M 4M 6M 8M 10M Training Data Scale

MMBench

SEEDBench

MMStar

MathVista

55

52

- 67

- 68

- 69

- 70

- 71

- 72

- 73

- 74

- 76

- 77

- 78

- 79

- 80

- 81

- 82

50

50

48

45

Accuracy

Accuracy

Accuracy

Accuracy

46

MAmmoTH-VL-10M

40

44

LLaVA-665k

LLaVA-665k

35

42

LLaVA-665k

30

LLaVA-665k

MAmmoTH-VL-10M

MAmmoTH-VL-10M

MAmmoTH-VL-10M

40

2M 4M 6M 8M 10M Training Data Scale

2M 4M 6M 8M 10M Training Data Scale

2M 4M 6M 8M 10M Training Data Scale

2M 4M 6M 8M 10M Training Data Scale

- Figure 2 | The scaling effects of visual SFT data from LLaVA-665k to MAmmoTH-VL-si-10M. We evaluate the checkpoints every 2M training samples.

improvement over the data scale. It might be caused by the lack of such pre-training math or OCR data in our curated pre-training caption dataset, making the visual math reasoning and text-based VQA become out-of-distribution tasks. For the general-pupose knowledge-based benchmarks of MMMU, SEEDBench, and MMStar, we even observe the slight performance degradation in the final 6M instruction tuning data.

We compare the final Open-Qwen2VL model with the state-of-the-art partially-open MLLMs of InternVL2.5-2B-MPO [45], DeepSeekVL-2-Tiny [48], and Qwen2-VL-2B-Instruct [44] on a set of general multimodal benchmarks, OCR VQA datasets, multimodal math reasoning benchmark, and hallucination benchmark. Concluded from the results in Table 4, Open-Qwen2VL demonstrates competitive performance across benchmarks compared to other 2B-parameter state-of-the-art MLLMs. It particularly excels in MMBench, achieving the highest score of 80.9, while maintaining comparable performance in SEEDBench and MMStar benchmarks. Moreover, Open-Qwen2VL outperforms the most relevant competitor Qwen2-VL-2B-Instruct on MMBench, SEEDBench-img, MMStar, and MathVista, while it is only trained on 0.35% of tokens of Qwen2VL. However, it shows relatively weaker results in OCR VQA tasks of AI2D and TextVQA. This is because the pre-training data of Open-Qwen2VL does not include the OCR-specific caption dataset like SynthDoG [23] or LAIONCOCO-OCR [38]. Simply introducing such OCR-related pre-training data will significantly enhance the OCR-VQA task performance of MLLMs.

#### 4. Analysis

##### 4.1. Impacts of Sequence Packing on Multi-Image In-Context Learning and Reasoning.

Flamingo [3] proposes MultiModal MassiveWeb (M3W) dataset to construct pseudo interleaving data structure using caption data to elicit the multimodal in-context learning capabilities of MLLMs. The multimodal sequence packing also constructs similar pseudo image-text interleaving data strcture. Thus, we conduct experiments to evaluate the few-shot multimodal in-context learning capabilities of the pre-trained base MLLM trained on packed multimodal sequences. We evaluate the base non-sft MLLM trained on the caption data mixture of CCSCLIP and DataComp-MLM-Filter & DFN on GQA, VQA-v2, VizWiz, OKVQA, and Text-VQA datasets. This base model is the best model we get based on the ablation studies on the pretraining data mixture in Table 3. We select 5 random seeds for the 8-shot multimodal in-context

Models InternVL2.5-2B-MPO DeepSeekVL-2-Tiny Qwen2-VL-2B-Ins Open-Qwen2VL # Pretrain Tokens 277B 8.1T 1.4T 5B

General Benchmark

MMMUval 41.2 39.6 41.1 39.8 MMBenchdev 72.5 68.3 68.8 80.9 SEEDBenchdev 73.2 72.5 72.0 72.5 MMStar 54.3 49.9 46.3 49.7

OCR VQA

AI2Dtest 75.3 74.6 72.3 66.3 TextVQAval 77.2 80.5 78.8 63.3

Math Reasoning MathVistatestmini 55.3 54.5 48.0 53.1 Hallucination POPE 89.8 88.8 87.6 84.4

- Table 4 | Benchmarks performance of Open-Qwen2VL and other 2B-parameter state-of-the-art MLLMs.

learning experiments and report the average performance over the 5 random seeds. The results in Table 5 demonstrate that the base MLLM trained with packed multimodal sequences can learn from multimodal demonstration examples for completing the task well. The 8-shot incontext learning can gain +3% to +12% performance improvements on VQA dataset compared with 0-shot reasoning. This also demonstrates the necessity and significance of performing multimodal sequence packing as it can enhance the multimodal in-context learning capabilities of the MLLMs.

- 4.2. Impact of Unfreezing Vision Encoder Parameters.

Most of the state-of-the-art MLLMs like InternVL-2.5 [9] demonstrate that making the vision encoder trainable during the SFT stages will enhance the multimodal understanding capabilities of MLLMs. Therefore, we perform such ablation studies on our base non-sft MLLM pre-trained on CCS+DataComp-MLM-Filter&DFN data mixture. We use the LLaVA-665k data as the visual SFT dataset and evaluate the two SFT-ed MLLMs with frozen and trainable vision encoder during the SFT stage. The results in Table 6 demonstrates that make the vision encoder parameters trainable during SFT stage can achieve better average performance while there is significant performance degradation on MMMU benchmark.

- 5. Related Work

Open-Source Multimodal Large Language Models. Close-source MLLMs like GPT-4o [2] and Claude-3.7-Sonnet [4] have strong multimodal understanding and reasoning capabilities. To replicate the strong capabilities of close-source MLLMs, research teams from industry develop the partially open-source strong MLLMs including InternVL-2.5 [9], DeepSeek-VL2 [48], and Qwen2.5-VL [6], which can achieve equivalent capability with close-source models. However, the training data, codebase and data filtering details of such models are not open-sourced for reproduction.

Large-Scale Image Text Data. Starting from ImageNet [11], the large-scale image dataset has significantly driven advances in both computer vision and multimodal foundational models.

###### # shots GQA VQA-v2 VizWiz OKVQA Text-VQA

0 27.1 40.2 26.1 24.7 30.4 8 35.4 51.8 31.2 27.1 30.6

- Table 5 | Results of the 0-shot and 8-shot multimodal in-context learning capabilities of pretrianed base MLLMs.

Vision Encoder

AI2D test

TextVQA POPE

MMMU val

MMBench Dev

SEEDBench Img-Dev

MMStar

MathVista test-mini

Avg.

Frozen 56.8 57.0 80.1 38.0 77.3 68.7 41.3 28.6 56.0 Trainable 57.4 57.6 82.3 36.1 76.5 69.7 41.4 29.3 56.3

- Table 6 | Ablation study on the trainable or frozen vision encoder during the SFT stage on LLaVA-665k data. We freeze vision encoder for pre-training stage due to limited computing resources.

MSCOCO [29], SBU [36], Conceptual Captions (CC) [41] scales up the image dataset size to near million level, which significantly enhances the image captioning performance of VLMs. OpenAI pre-trained contrastive VLM, CLIP with 400M WebImage data without releasing them. Then LAION-400M and COYO-700M are open-source efforts to further scale up the image-text dataset to hundreds of million level. Then LAION-5B and DataComp-commonpool-12.8B scale up the image-text dataset size to billions level for supporting the data-intensive MLLM pretraining. Most of SOTA MLLMs like DeepSeek-VL [48], Qwen-VL [5], Intern-VL [10], SAIL [12] construct and curate their own large-scale image-text dataset with more then 10B image-text data, while such dataset will not be released for public research.

High-Quality Image-Text Data Filtering. Beyond the conventional rule-based or heuristicbased data filtering methods in constructing image-text dataset, image-text dataset with much larger scale for training contrastive VLMs adopts CLIPScore-based filtering methods for highquality data curation. LAION-400M [39] set a hard filtering threshold using OpenAI-CLIP model for its data filtering. Later, DataComp [15] is the first effective benchmark for fairly evaluating the effectiveness of each data filtering method on selecting high-quality data for CLIP pre-training. Various methods [33, 47, 51] try to combine CLIPScore filteirng with other metrics to achieve better filtering performance on DataComp, while DFN [14] directly scales up the CLIP-based data filtering model and achieves the top-1 performance. Moreover, another line of data filtering method based on efficient MLLM [18, 46] emerges and presents better capabilities in selecting high-quality data for MLLM pre-training.

#### 6. Conclusion

We demonstrate that the efficient MLLM-based high-quality data filtering techniques and welldesigned data mixture strategies can achieve compute-efficient pretraining for developing SOTA MLLMs. Adopting the multimodal sequence packing and dynamic image token number with average-pooling layer can further enhance such pre-training efficiency. The induced final MLLM, Open-Qwen2VL outperforms the partially-open MLLM Qwen2-VL-2B on various multimodal benchmarks, in which Open-Qwen2VL is trained on only 0.36% pre-training tokens of Qwen2VL. The training is conducted on academic-level computing resources and demonstrates that the advanced training pipeline and data filtering can overcome the limitation of computing resources. We wish Open-Qwen2VL can motivate the fully-open compute-efficient multimodal pre-training research from academic community.

#### Acknowledgments

We would like to thank for Facebook (now Meta) for donating the 8xA100-40G GPUs for conducting the experiments. We appreciate the codebase of prismatic-vlms [21] and vlmevaluation5, on which we build our codebase.

#### References

- [1] M. Abdin et al. “Phi-3 technical report: A highly capable language model locally on your phone”. In: arXiv preprint arXiv:2404.14219 (2024).
- [2] J. Achiam et al. “Gpt-4 technical report”. In: arXiv preprint arXiv:2303.08774 (2023).
- [3] J.-B. Alayrac et al. “Flamingo: a visual language model for few-shot learning”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 23716–23736.
- [4] Anthropic. Claude 3.7 Sonnet and Claude Code. https://www.anthropic.com/news/ claude-3-7-sonnet. Accessed on Feb 24, 2025. 2025.
- [5] J. Bai et al. “Qwen-vl: A frontier large vision-language model with versatile abilities”. In: arXiv preprint arXiv:2308.12966 (2023).
- [6] S. Bai et al. “Qwen2. 5-vl technical report”. In: arXiv preprint arXiv:2502.13923 (2025).
- [7] R. Beaumont. img2dataset: Easily turn large sets of image urls to an image dataset. https: //github.com/rom1504/img2dataset. 2021.
- [8] L. Chen et al. “Are we on the right way for evaluating large vision-language models?” In: arXiv preprint arXiv:2403.20330 (2024).
- [9] Z. Chen et al. “Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling”. In: arXiv preprint arXiv:2412.05271 (2024).
- [10] Z. Chen et al. “InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks”. In: arXiv preprint arXiv:2312.14238 (2023).
- [11] J. Deng et al. “Imagenet: A large-scale hierarchical image database”. In: 2009 IEEE conference on computer vision and pattern recognition. Ieee. 2009, pp. 248–255.
- [12] H. Dong et al. “Scalable vision language model training via high quality data curation”. In: arXiv preprint arXiv:2501.05952 (2025).
- [13] A. Dubey et al. “The llama 3 herd of models”. In: arXiv preprint arXiv:2407.21783 (2024).
- [14] A. Fang et al. “Data filtering networks”. In: arXiv preprint arXiv:2309.17425 (2023).
- [15] S. Y. Gadre et al. “DataComp: In search of the next generation of multimodal datasets”. In: arXiv preprint arXiv:2304.14108 (2023).
- [16] J. Guo et al. “Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale”. In: arXiv preprint arXiv:2412.05237 (2024).
- [17] J. Hessel et al. “Clipscore: A reference-free evaluation metric for image captioning”. In: arXiv preprint arXiv:2104.08718 (2021).
- [18] H. Huang et al. “Beyond Filtering: Adaptive Image-Text Quality Enhancement for MLLM Pretraining”. In: arXiv preprint arXiv:2410.16166 (2024).
- [19] C. Jia et al. “Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision”. In: ICML. 2021.
- [20] D. S. Johnson. “Near-optimal bin packing algorithms”. PhD thesis. Massachusetts Institute of Technology, 1973.

5https://github.com/TRI-ML/vlm-evaluation

- [21] S. Karamcheti et al. “Prismatic vlms: Investigating the design space of visually-conditioned language models”. In: Forty-first International Conference on Machine Learning. 2024.
- [22] A. Kembhavi et al. “A diagram is worth a dozen images”. In: Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14. Springer. 2016, pp. 235–251.
- [23] G. Kim et al. “OCR-Free Document Understanding Transformer”. In: European Conference on Computer Vision (ECCV). 2022.
- [24] H. Laurençon et al. “What matters when building vision-language models?” In: arXiv preprint arXiv:2405.02246 (2024).
- [25] B. Li et al. “Seed-bench: Benchmarking multimodal llms with generative comprehension”. In: arXiv preprint arXiv:2307.16125 (2023).
- [26] J. Li et al. “Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation”. In: International Conference on Machine Learning. PMLR. 2022, pp. 12888–12900.
- [27] Y. Li et al. “Evaluating object hallucination in large vision-language models”. In: arXiv preprint arXiv:2305.10355 (2023).
- [28] J. Lin et al. “Vila: On pre-training for visual language models”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024, pp. 26689–26699.
- [29] T.-Y. Lin et al. “Microsoft coco: Common objects in context”. In: Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. Springer. 2014, pp. 740–755.
- [30] H. Liu et al. “Improved baselines with visual instruction tuning”. In: arXiv preprint arXiv:2310.03744 (2023).
- [31] Y. Liu et al. “Mmbench: Is your multi-modal model an all-around player?” In: European conference on computer vision. Springer. 2024, pp. 216–233.
- [32] P. Lu et al. “Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts”. In: arXiv preprint arXiv:2310.02255 (2023).
- [33] P. Maini et al. “T-mars: Improving visual representations by circumventing text feature learning”. In: arXiv preprint arXiv:2307.03132 (2023).
- [34] B. McKinzie et al. “Mm1: Methods, analysis & insights from multimodal llm pre-training”. In: arXiv preprint arXiv:2403.09611 (2024).
- [35] OpenAI. “GPT-4V(ision) Technical Work and Authors”. In: (2023).
- [36] V. Ordonez, G. Kulkarni, and T. Berg. “Im2text: Describing images using 1 million captioned photographs”. In: Advances in neural information processing systems 24 (2011).
- [37] A. Radford et al. “Learning Transferable Visual Models From Natural Language Supervision”. In: ICML. 2021.
- [38] C. Schuhmann et al. “Laion coco: 600m synthetic captions from laion2b-en”. In: URL https://laion. ai/blog/laion-coco 5 (2022).
- [39] C. Schuhmann et al. “Laion-400m: Open dataset of clip-filtered 400 million image-text pairs”. In: arXiv preprint arXiv:2111.02114 (2021).
- [40] P. Sharma et al. “Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning”. In: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018, pp. 2556–2565.
- [41] P. Sharma et al. “Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning”. In: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018, pp. 2556–2565.

- [42] A. Singh et al. “Towards vqa models that can read”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019, pp. 8317–8326.
- [43] Q. Team. Qwen2.5: A Party of Foundation Models. Sept. 2024.
- [44] P. Wang et al. “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution”. In: arXiv preprint arXiv:2409.12191 (2024).
- [45] W. Wang et al. “Enhancing the reasoning ability of multimodal large language models via mixed preference optimization”. In: arXiv preprint arXiv:2411.10442 (2024).
- [46] W. Wang et al. “Finetuned multimodal language models are high-quality image-text data filters”. In: arXiv preprint arXiv:2403.02677 (2024).
- [47] Y. Wang et al. “Cliploss and norm-based data selection methods for multimodal contrastive learning”. In: Advances in Neural Information Processing Systems 37 (2024), pp. 15028– 15069.
- [48] Z. Wu et al. “Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding”. In: arXiv preprint arXiv:2412.10302 (2024).
- [49] L. Xue et al. “xGen-MM (BLIP-3): A Family of Open Large Multimodal Models”. In: arXiv preprint arXiv:2408.08872 (2024).
- [50] L. Yao et al. “DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models”. In: arXiv preprint arXiv:2405.20985 (2024).
- [51] H. Yu et al. “The devil is in the details: A deep dive into the rabbit hole of data filtering”. In: arXiv preprint arXiv:2309.15954 (2023).
- [52] X. Yue et al. “Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024, pp. 9556–9567.
- [53] X. Zhai et al. “Sigmoid loss for language image pre-training”. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023, pp. 11975–11986.
- [54] H. Zhang et al. “Mm1. 5: Methods, analysis & insights from multimodal llm fine-tuning”. In: arXiv preprint arXiv:2409.20566 (2024).

#### A. Training Settings of MLLM Pre-Training

The training details and hyperparameters for MLLM pre-training are presented in Tab. 7. The pre-training is only one-stage process. We do not follow Qwen-VL or DeepSeek-VL to split the MLLM pre-training into two stages of VL alignment and VL pre-training.

Details Pre-Training Visual SFT

Vision Encoder SigLIP-so400m-384px SigLIP-so400m-384px Visual Projector 2d Adaptive Average Pooling + MLP MLP LLM Backbone Qwen2.5-1.5B-Instruct Qwen2.5-1.5B-Instruct # Tokens per Image 144 729 Context Length 4096 4096 Sequence Packing Yes No

Precision BF16 BF16 Global Batch Size 256 128 # Training Epoch 1 1 # GPUs 8 A100-40G 8 A100-40G Peak LR 5e-5 2e-5 # Warmup Steps Ratio 3% 3% LR Scheduler linear-warmup+cosine-decay linear-warmup+cosine-decay Weight Decay 0.01 0.1

Table 7 | Training details and hyper-parameters for MLLM pre-training and visual supervised fine-tuning.

#### B. Examples

Prompt

[Figure 3]

Can you introduce this painting?

Open-Qwen2VL Response

This painting is a beautiful representation of Vincent van Gogh's famous work, "Irises." The painting is a vibrant and detailed depiction of a garden filled with irises, showcasing van Gogh's unique style of using bold colors and expressive brushstrokes. The irises are painted in various shades of blue, with some flowers appearing more saturated than others, creating a sense of depth and movement. The background features a mix of green foliage and hints of yellow and orange, adding to the overall vibrancy of the scene. The painting captures the essence of the irises' natural beauty and the serene atmosphere of the garden, making it a captivating piece of art.

Claude-3.7 Sonnet Response

This is Vincent van Gogh's "Irises," painted in 1889. It's one of his most famous works, created during his stay at the asylum of Saint Paul-de-Mausole in Saint-Rémy-de-Provence, France, shortly before his death in 1890.

The painting depicts a garden filled with vibrant purple-blue irises against a backdrop of green foliage and reddish-brown soil. Notice the single white iris standing out among the blue ones in the upper left portion of the canvas. Van Gogh painted this work from nature in the asylum's garden, capturing the flowers with his characteristic thick, expressive brushstrokes and bold use of color.

"Irises" reflects the Japanese ukiyo-e woodblock print influence on Van Gogh's work during this period, with its strong outlines, unusual angles, and flattened perspective. The painting is now housed in the J. Paul Getty Museum in Los Angeles and is considered one of the most valuable paintings in the world.

###### Figure 3 | An example for generating a caption towards a famous painting, which requires the background knowledge.

Prompt

Please describe the image.

Open-Qwen2VL Response

The image is a collage of four distinct scenes, each showcasing different aspects of a vibrant and culturally rich environment.

[Figure 4]

- 1. **Top Left Image**: This image captures a group of people dressed in traditional Indian attire, likely celebrating a festival or cultural event. The individuals are adorned in colorful and elaborate costumes, with some wearing traditional Indian garments such as sarees and lehengas. They are standing in front of a stage with a banner that reads "PONGAL FESTIVAL." The setting appears to be a festive venue, possibly a temple or a cultural center, decorated with vibrant decorations and lights. The atmosphere suggests a celebratory and joyous occasion, likely related to the Pongal festival, which is a harvest festival celebrated in South India.
- 2. **Top Right Image**: This image depicts a bustling urban street scene at night. The street is lined with numerous colorful lanterns, creating a festive and lively ambiance. The street is illuminated by these lanterns, which are traditional Chinese lanterns, adding to the cultural richness of the scene. The street is busy with vehicles, including cars and buses, indicating a well-connected and lively city environment. The presence of the lanterns and the busy street scene suggest that this could be a popular area for celebrations or festivals, possibly during the festival season.
- 3. **Bottom Left Image**: This image shows a young boy in traditional Indian attire, specifically a red kurta and a matching shawl, holding a blue object, possibly a small flag or a decorative item. He is smiling and appears to be in a cheerful mood, indicating a sense of pride or joy in his attire. The background includes a woman in traditional Indian clothing, possibly a mother or a guardian, who is handing over the object to the boy. This scene likely captures a moment of cultural heritage and tradition being passed down to the younger generation.
- 4. **Bottom Right Image**: This image is a close-up of a woman in traditional Indian clothing, possibly a saree, standing in front of a colorful display of lanterns and other decorative items. The woman is smiling and appears to be engaged in a conversation or looking at something in her hands. The background is filled with a variety of colorful lanterns and other decorative items, creating a festive and vibrant atmosphere. The setting suggests a market or a festival stall, where traditional items and decorations are being sold or displayed.

In summary, the collage showcases a rich tapestry of cultural celebrations and traditions, highlighting the importance of heritage and community in South Asian cultures. The images collectively depict a vibrant and lively environment, filled with traditional attire, festive decorations, and bustling urban life, reflecting the joy and pride associated with cultural festivals and celebrations.

Claude-3.7 Sonnet Response

These images showcase various cultural celebrations and festivities in Southeast Asia: Top left: A Pongal Open House 2024 celebration featuring performers in traditional Indian attire. Several women are dressed in colorful saris and traditional costumes, with some holding decorative horse figurines. In the background, there are performers with peacock feather displays, and a banner for the "Pongal Open House 2024" event is visible.

Top right: A nighttime street scene showing elaborate lantern and light decorations spanning across a busy road. The street is illuminated with large flower-shaped lanterns in red and gold colors, creating a festival atmosphere. Traffic flows underneath the decorations, likely during a cultural celebration or holiday.

Bottom left: A young child wearing a traditional Malay outfit consisting of a red top and black songkok (cap) is receiving or exchanging what appears to be a small gift or envelope, possibly "duit raya" (money given during Hari Raya/Eid celebrations).

Bottom right: A vibrant marketplace or bazaar filled with colorful ornaments, decorations, and hanging items for sale. Shoppers, including a person in purple attire, are browsing through the densely packed stalls of festive goods and decorations.

These images represent various cultural festivals and celebrations across Southeast Asia, including Pongal (Tamil harvest festival), possibly Chinese New Year or Mid-Autumn Festival (lanterns), Hari Raya celebrations, and festive marketplaces.

###### Figure 4 | An example for generating long captions for a complicated 2x2 image.

