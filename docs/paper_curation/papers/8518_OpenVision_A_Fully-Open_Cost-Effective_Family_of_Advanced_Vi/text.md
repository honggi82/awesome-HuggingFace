## OpenVision : A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning

[Figure 1]

# arXiv:2505.04601v1[cs.CV]7May2025

Xianhang Li* Yanqing Liu* Haoqin Tu Hongru Zhu Cihang Xie University of California, Santa Cruz

[Figure 2]

Project Page: https://ucsc-vlaa.github.io/OpenVision Model Training: https://github.com/UCSC-VLAA/OpenVision Model Zoo: click me

[Figure 3]

[Figure 4]

Data Training Evalua>on Model Num. Training Time

### Abstract

Model

[Figure 5]

Short Long

|4|
|---|

OpenAI’s CLIP

Closed Closed Open

[Figure 6]

[Figure 7]

OpenAI’s CLIP, released in early 2021, have long been the go-to choice of vision encoder for building multimodal foundation models. Although recent alternatives such as SigLIP have begun to challenge this status quo, to our knowledge none are fully open: their training data remains proprietary and/or their training recipes are not released. This paper fills this gap with OpenVision, a fullyopen, cost-effective family of vision encoders that match or surpass the performance of OpenAI’s CLIP when integrated into multimodal frameworks like LLaVA. OpenVision builds on existing works—e.g., CLIPS for training framework and Recap-DataComp-1B for training datawhile revealing multiple key insights in enhancing encoder quality and showcasing practical benefits in advancing multimodal models. By releasing vision encoders spanning from 5.9M to 632.1M parameters, OpenVision offers practitioners a flexible trade-off between capacity and efficiency in building multimodal models: larger models deliver enhanced multimodal performance, while smaller versions enable lightweight, edge-ready multimodal deployments.

|10|
|---|

Closed

Open

Open

Google’s SigLIP

[Figure 8]

[Figure 9]

|>25|
|---|

OpenVision

Open Open Open

[Figure 10]

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Figure 1: The top table compares our OpenVision series to OpenAI’s CLIP and Google’s SigLIP. The bottom figure showcases that OpenVision attain competitive or even superior multimodal performance than OpenAI’s CLIP and Google’s SigLIP.

### 1. Introduction

Recent advances in multimodal foundation models rely almost exclusively on the same visual backbone: OpenAI’s CLIP encoders [39]. From early open-source efforts such as LLaVA [30] and Mini-GPT-4 [55], to the most recent advanced models such as Falcon2 VLM [35] and Eagle [41], OpenAI’s CLIP-L/336 has consistently been the default choice, even as the language components have evolved rapidly. This dependence, however, imposes several issues. First, OpenAI CLIP’s training data and detailed framework remain undisclosed, limiting transparency and reproducibil-

ity. Moreover, OpenAI’s CLIP is available only in two parameter scales—Base and Large—hindering both the deployment of lightweight models on edge devices and the exploration of higher-capacity encoders for complex tasks. Finally, OpenAI’s CLIP suffers from documented weaknesses, including poor spatial-relation understanding and object-counting hallucinations [45, 44, 46]. These shortcomings call for a vision encoder whose architecture, data, and training recipe are fully open.

*Equal contribution.

In response, the open-source community has mounted a concerted effort to replicate and surpass OpenAI’s CLIP, notably through (1) fully open CLIP training frameworks [17], (2) billion-scale open datasets such as Laion [40], DataComp [15], and DFN [12], and (3) improved training methodologies [24, 23, 21, 52]. Yet a crucial gap persists: no fully open, from-scratch vision encoder of comparable capacity and resolution consistently matches—or surpasses—OpenAI’s CLIP when used as the visual backbone of multimodal foundation models. For example, popular OpenCLIP [17] checkpoints achieve superior zero-shot performance, but they fall markedly short on multimodal benchmarks such as MME [14], ChartQA [36] and TextVQA [42] (see Tables 1 and 2).

In this work, we address this gap with OpenVision, a fully-open, cost-effective family of vision encoders that excel in multimodal learning scenarios (Figure 11). OpenVision builds on two recent advances: (i) Recap-DataComp-

- 1B [22], which re-captions the entire DataComp-1B corpus [15] using a LLaVA model powered by Llama-3 [37]; and (ii) CLIPS [31], an enhanced CLIP training pipeline that incorporates synthetic captions. Leveraging these resources, we conduct a systematic analysis to identify key design elements that improve overall training efficiency and enhance the quality of vision encoders, as well as showcasing their practical benefits in the development of different multimodal models.

Extensive experiments show that OpenVision matches—and sometimes exceeds—OpenAI’s CLIP across a suite of multimodal evaluations when used as the visual backbone of multimodal models such as LLaVA-1.5 and Open-LLaVA-Next. To accommodate diverse deployment needs, we release more than 25 checkpoints ranging from 5.9 million to 632.1 million parameters, enabling smooth accuracy–efficiency trade-offs from edge devices to high-capacity servers. By openly releasing datasets, training recipes, and checkpoints, we hope OpenVision can set a new standard for transparency and flexibility, enabling the community to push multimodal research beyond the constraints of proprietary encoders.

### 2. OpenVision Training and Evaluation

This section outlines the pipeline for building and assessing the OpenVision family. We provide details about the vision encoder pre-training, multimodal large language model (MLLM) instruction tuning, and MLLM evaluation.

#### 2.1. Vision Encoder Pre-training

Recent studies have revealed multiple key aspects in advancing MLLMs, including model architecture and training

1Note that we normalize OCR and MME scores to the range of 0 to 100 following previous research [13].

strategies [10, 7, 44], yet the discussion about its vision encoder training remains lacking. Our objectives are therefore two-fold: (i) to publish a fully reproducible “from-scratch” recipe for training strong vision encoders, and (ii) to isolate the design choices that matter most once these encoders are paired with an LLM.

We leverage CLIPS [31]—a recent variant of CLIPas our building foundation. CLIPS employs the standard two-tower architecture with a contrastive objective, but extends it with a multi-positive loss that treats both the original and synthetic captions of an image as positives. A lightweight text decoder is trained jointly to generate new captions, further enriching the training signal. While CLIPS attains state-of-the-art zero-shot retrieval performance, its suitability as an MLLM perception module remains underexplored—a gap we fill in this work. Additionally, following CLIPS, we use Recap-DataComp-1B [22], a re-captioned version of the billion-scale DataComp corpus [15] [37], for training. Both CLIPS codebase 2 and Recap-DataComp-1B dataset 3 are fully open-sourced.

Training Stages and Resolution. Following the efficient training curriculum of CLIPA [24, 23], we pre-train every encoder in three successive resolution stages. Specifically, the Large, SoViT-400M, and Huge variants are trained at 84 × 84, 224 × 224, and finally 336 × 336 or 384 × 384. Smaller models such as Tiny, Small, and Base start at a larger resolution of 160 × 160, and then continues with 224×224, and 336×336 or 384×384. This staged approach substantially improves efficiency and naturally yields model variants capable of handling different input resolutions. After pre-training, we discard the text tower and decoder, retaining only the vision backbone.

Training Details. Across three stages, each models processes 12.8B, 1.024B, and 256M image–text pairs, respectively. The global batch sizes are 32K, 16K, and 8K, with cosine-decayed base learning rates of 8 × 10−6, 4 × 10−7, and 1 × 10−7. The text encoder uses 80 input tokens, and the text decoder generates 128 tokens, consistent with CLIPS [31]. For experiments involving different patch sizes, we only modify the patch size to 8; fixed sinecosine positional embeddings allow adaptation to varying sequence lengths.

#### 2.2. Visual Instruction Fine-tuning and Evaluation

To assess the quality of visual encoders from the MLLM perspective, we benchmark them on general VQA tasks, which require generating free-form text answers based on visual inputs. Following prior practice [30, 29, 19], we attach a lightweight MLP projector to the vision encoder, concatenate the resulting visual tokens to the language tokens, and perform visual instruction tuning. Unlike prior

- 2https://ucsc-vlaa.github.io/CLIPS/
- 3https://www.haqtu.me/Recap-Datacomp-1B/

- Table 1: Comparison of OpenVision encoders with existing CLIP variants on CLIP benchmarks and multimodal downstream tasks under the LLaVA-1.5 framework. Cls./Retr.: zero-shot classification accuracy on ImageNet or image and text retrieval on MSCOCO. OpenVision outperforms OpenAI-CLIP significantly across multiple settings.

Method Vision Encoder # Res.

|CLIP-Bench| |
|---|---|
|Cls.|Retr.|

Text VQA Chart QA OCR. MME SEED MMVet SQA GQA POPE OpenAI-CLIP [39] B/16 224 68.3 33.1/52.4 53.1 11.9 153 1444/325 63.7 28.3 72.5 59.9 83.4

SigLIP [52] B/16 224 76.0 47.8/65.7 53.3 12.2 238 1421/318 65.5 31.3 73.8 60.3 84.2 OpenVision B/16 224 73.9 51.1/71.6 54.1 11.8 262 1496/293 68.2 30.9 74.4 61.6 86.6

SigLIP [52] B/16 384 78.5 49.9/67.7 57.3 13.9 285 1411/266 67.7 33.6 73.2 62.0 86.0 OpenVision B/16 384 74.5 52.0/72.3 57.9 14.5 293 1432/333 69.8 33.2 73.5 62.8 87.8

OpenAI-CLIP [39] L/14 224 75.5 36.5/56.3 56.1 13.2 177 1443/306 66.0 32.8 73.4 60.8 85.0 LAION-2B-CLIP [17] L/14 224 75.3 46.5/63.4 54.2 12.8 165 1434/298 65.5 31.4 76.0 59.0 84.5

DataComp-1B-CLIP [15] L/14 224 79.2 45.7/63.3 53.0 12.3 131 1382/312 62.4 28.9 74.2 57.8 83.0 DFN-2B-CLIP [12] L/14 224 81.4 48.6/65.6 53.2 12.4 246 1447/306 65.6 29.4 76.3 59.1 85.0 MetaCLIP-5B [48] L/14 224 79.2 47.1/64.4 55.6 12.8 313 1552/315 67.4 34.6 78.0 61.3 85.4

OpenVision L/14 224 78.4 55.3/75.2 57.7 13.9 315 1487/317 69.5 35.2 73.6 62.9 86.4 OpenAI-CLIP [39] L/14 336 76.6 37.1/57.9 59.1 13.8 201 1475/288 67.5 35.2 73.1 61.1 85.7

OpenVision L/14 336 78.8 55.9/75.2 61.2 15.7 339 1525/315 70.5 36.2 75.1 63.7 87.2

SigLIP [52] SoViT-400M/14 384 83.2 52.0/70.2 62.6 14.5 338 1481/347 69.4 35.1 76.7 63.3 87.0 OpenVision SoViT-400M/14 384 79.9 57.6/77.5 62.4 16.1 357 1493/320 70.4 35.3 72.4 63.8 88.0

- Table 2: Comparison of OpenVision encoders with existing CLIP variants on CLIP benchmarks and multimodal downstream tasks under the Open-LLaVA-Next framework. Cls./Retr.: zero-shot classification accuracy on ImageNet or image and text retrieval on MSCOCO. OpenVision achieves comparable or even better performance than existing models.

|CLIP-Bench| |
|---|---|
|Cls.|Retr.|

Text VQA Chart QA OCR. MME SEED MMVet SQA GQA POPE OpenAI-CLIP [39] B/16 224 68.3 33.1/52.4 58.7 57.5 379 1497/321 70.0 38.6 74.0 62.7 86.6

Method vision Encoder # Res.

SigLIP [52] B/16 224 76.0 47.8/65.7 58.4 53.6 377 1430/332 69.5 33.6 75.9 62.4 85.8 OpenVision B/16 224 73.9 51.1/71.6 60.7 59.2 405 1520/336 71.8 38.8 73.1 63.1 86.4

SigLIP [52] B/16 384 78.5 49.9/67.7 64.2 63.3 476 1540/326 71.3 38.7 69.0 62.6 87.6 OpenVision B/16 384 74.5 52.0/72.3 66.3 67.4 499 1501/330 72.9 40.6 69.8 64.0 87.7

OpenAI-CLIP [39] L/14 224 75.5 36.5/56.3 62.8 60.7 459 1600/334 70.6 41.5 75.0 62.8 86.9 LAION-2B-CLIP [17] L/14 224 75.3 46.5/63.4 59.4 50.8 396 1533/323 70.0 36.2 72.9 62.7 86.4

DataComp-1B-CLIP [15] L/14 224 79.2 45.7/63.3 58.1 48.5 373 1524/348 70.2 37.2 75.6 62.3 86.2 DFN-2B-CLIP [12] L/14 224 81.4 48.6/65.6 57.0 42.7 303 1486/328 68.3 34.5 70.6 61.7 86.0 MetaCLIP-5B [48] L/14 224 79.2 47.1/64.4 63.0 62.9 493 1590/335 72.3 41.8 77.1 64.0 86.8

OpenVision L/14 224 78.4 55.3/75.2 65.7 61.5 503 1567/332 73.1 41.4 73.1 64.7 87.8 OpenAI-CLIP [39] L/14 336 76.6 37.1/57.9 69.4 70.0 535 1591/351 73.3 40.8 76.9 64.5 87.6

OpenVision L/14 336 78.8 55.9/75.2 68.3 68.0 547 1520/310 73.3 45.3 75.4 64.4 88.1

SigLIP [52] SoViT-400M/14 384 83.2 52.0/70.2 68.2 61.3 494 1539/325 72.9 40.5 74.7 62.9 86.8 OpenVision SoViT-400M/14 384 79.9 57.6/77.5 67.4 63.1 540 1500/353 72.2 43.7 73.5 63.4 87.8

work that studies off-the-shelf checkpoints [44], we compare our from-scratch OpenVision models with CLIP-style baselines at different sizes. All experiments use Llama-38B as the language backbone and adopt two LLaVA setups:

- 1. LLaVA-1.5 [28]. In this low-compute regime the vision encoder is kept frozen; only the lightweight projector and the language model are updated. This setup allows us to assess the quality of the pre-trained vision features. We train with the standard LCS-558K and LLaVA-665K datasets.
- 2. Open-LLaVA-Next [6]. This high-compute regime gauges the encoder’s capacity for further learning and scaling. Roughly one million image–instruction pairs are used, and the vision backbone, projector, and LLM are all finetuned. The setup also employs the “any-resolution” strategy [29] to tackle larger inputs: each image is resized to several aspect-ratio variants (e.g., 672×672, 336×1344) generated from a base size of 336 × 336.

Evaluation benchmarks. Performance is reported on a broad suite, including: MME [14], GQA [16], ChartQA [36], POPE [25], TextVQA [42], OCR [32], SEED [18], MMVet [51], and SQA [34]. We follow the lmms-eval protocol [53] for prompt formatting and use greedy decoding as the text generation strategy in all tasks.

### 3. Main Results

#### 3.1. OpenVision vs. Proprietary

We compare our OpenVision family against popular proprietary and open-source vision encoders under the LLaVA-1.5 and Open-LLaVA-Next frameworks. To ensure fairness, all runs employ the original hyper-parameters provided by CLIPS [31], LLaVA-1.5 [28], and OpenLLaVA-Next [6]. Figure 1 offers a high-level view: across nine representative benchmarks, OpenVision con-

Table 3: Performance of OpenVision encoders at different scales with Llama3-8B under LLaVA-1.5.

|CLIP-Bench| |
|---|---|
|Cls.|Retr.|

Vision Encoder # Res. # Params.

Text VQA Chart QA OCR. MME SEED MMVet SQA GQA POPE

OpenAI-CLIP-L/14 224 303.7M 75.5 36.5/56.3 56.1 13.2 177 1443/306 66.0 32.8 73.4 60.8 85.0 L/14 224 303.7M 78.4 55.3/75.2 57.7 13.9 315 1487/317 69.5 35.2 73.6 62.9 86.4 H/14 224 632.1M 80.4 57.4/77.0 57.9 13.6 330 1501/308 69.3 35.8 75.9 61.9 87.0

B/16 224 87.4M 73.7 51.1/71.6 54.1 11.8 262 1496/293 68.2 30.9 74.4 61.6 86.6 S/16 224 22.4M 65.9 43.6/64.5 51.8 11.0 202 1348/264 65.5 24.6 71.8 60.1 84.6 Ti/16 224 5.9M 49.6 50.0/30.4 48.9 11.7 128 1273/282 59.9 21.8 71.8 57.4 82.0

Table 4: Performance of OpenVision encoders with Qwen2.5-0.5B under LLaVA-1.5.

|CLIP-Bench| |
|---|---|
|Cls.|Retr.|

Vision Encoder # Res. # Params.

Text VQA Chart QA OCR MME SEED MMVet SQA GQA POPE

OpenAI-CLIP-B/16 224 87.4M 68.3 33.1/52.4 33.5 10.0 69 1059/255 49.1 13.6 55.8 48.5 82.3 B/16 224 87.4M 73.9 51.1/71.6 34.8 10.1 132 1063/252 51.4 16.1 56.0 49.6 84.4 B/16 384 87.4M 74.5 52.0/72.3 38.2 10.3 174 1171/280 53.9 15.9 56.0 51.8 85.8 S/16 384 22.4M 67.1 45.0/66.2 32.8 9.9 78 1071/246 50.5 11.6 54.7 49.1 84.3 Ti/16 384 5.9M 51.4 32.2/53.0 27.4 9.4 27 843/263 40.9 11.0 54.1 42.8 79.1

sistently matches—or surpasses—the performance of OpenAI’s CLIP and Google’s SigLIP.

A more comprehensive comparison is presented in Table 1 and Table 2, which also include results for LAION-2B-CLIP [40], DataComp-1B-CLIP [15], DFN-

- 2B-CLIP [12], and MetaCLIP-5B [48]. At 224×224 resolution, our OpenVision-B/16 and OpenVision-L/14 checkpoints significantly outperform their counterparts on most tasks under both MLLM setups. At 336 × 336 resolution, OpenVision-L/14-336 either closely matches or exceeds OpenAI’s CLIP-L/14-336 under Open-LLaVA-Next setup, establishing a new benchmark for open-source visual encoders.

These findings confirm that vision models trained entirely from public data and code can rival—and often outdo—proprietary alternatives, providing the research community with competitive, transparent, and flexible backbones for future multimodal work.

#### 3.2. More OpenVision Variants

The full transparency of OpenVision allows us to freely craft a spectrum of vision encoders (see Appendix A.2 for architecture details) tailored to different resource or accuracy demands. Specifically, we illustrate this versatility by scaling OpenVision up/down and varying patch size for different application scenarios, and by showcasing its competitiveness even with an ultra-small language model.

Scale Up for Superior Multimodal Performance. For applications demanding strong multimodal performance, larger vision encoders are beneficial as they can encode richer semantics and align more precisely with language. To this end, we release OpenVision-H/14, a 632.1 Mparameter vision encoder—significantly larger than the largest models from OpenAI’s CLIP and Google’s SigLIP.

As shown in Table 3 under the LLaVA-1.5 setup, this variant delivers substantial gains over OpenAI CLIP-L/14 in multimodal understanding, particularly in high-resolution VQA, OCR, and retrieval tasks, confirming the value of additional capacity for challenging multimodal tasks.

Scale Down for Resource-Limited Scenarios. To meet the memory and latency budgets of mobile or low-power devices, we train two compact variants, i.e., OpenVisionS/16 and OpenVision-Ti/16. In the same LLaVA-1.5 setting (Table 3), S/16 retains 94% of CLIP-L/14’s average score while using more than 13× fewer parameters, and Ti/16 keeps 87% at nearly 50× smaller size.

We further pair these encoders with a 0.5 B-parameter Qwen-2.5 LLM [49]. Firstly, simply replacing the baseline CLIP-B/16 with OpenVision-B/16 boosts accuracy on nearly every benchmark (Table 4). Then, by scaling down the size of vision encoder and meanwhile increasing the resolution from 224 × 224 to 384 × 384, the smaller S/16 and Ti/16 manage to maintain very competitive performance. These results confirm that lightweight, fully open vision backbones can power practical, high-quality edgeready multimodal systems.

Variable Patch Sizes. In a ViT, the patch size determines the spatial resolution at which an image is tokenized [47], i.e., smaller patches supply finer details when encoding visual features (while at the cost of significantly increased computational budget). To assess the impact of patch size, we therefore pre-trained two otherwise identical OpenVision models with 8 × 8 and 16 × 16 patches.

Table 5 summarizes performance comparisons on a range of multimodal benchmarks under the LLaVA-1.5 setup. We can observe that the 8×8 variant delivers consistent and significant gains across all tasks, especially on finegrained understanding tasks like TextVQA (e.g., +4.4% for

Table 5: Impact of different patch sizes in LLaVA-1.5. Smaller patch sizes generally improve performance.

Vision Encoder Patch Size Text VQA Chart QA OCR. MME SEED MMVet SQA GQA POPE

Ti 16 50.2 11.6 139 1329/280 62.0 21.4 73.1 58.0 82.8 Ti 8 54.6 12.9 223 1383/310 66.3 25.1 73.1 59.7 85.3

S 16 54.3 12.0 235 1393/343 67.5 28.8 73.2 61.6 85.7 S 8 59.3 15.9 310 1449/303 70.3 32.5 74.7 62.0 87.1

B 16 57.9 14.5 293 1432/333 69.8 33.2 73.5 62.8 87.8 B 8 61.2 17.2 345 1545/299 71.8 35.5 74.0 63.0 87.0

- Table 6: By pairing with a small LM (Smol-150M), we use OpenVision-B/16-384 to create a ∼250M multimodal model. We show scaling behavior across Stage 2 data size, input resolution, and Stage 3 data size. We report performance on the following benchmarks: TextVQA, ChartQA, OCR-VQA, MME, SEED-Bench, MMVet, SQA, GQA, and POPE.

Stage 2 Res. Stage 3 Data Scale TextVQA ChartQA OCR-VQA MME SEED-Bench MMVet SQA GQA POPE (1) Scale Stage 2 Data: ×1, ×2, ×4, ×6, ×8 (fix resolution=384, Stage 3=LLaVA (665K))

- ×1 384

LLaVA (665K)

33.2 10.3 194 743/212 48.8 15.8 38.2 54.2 85.0

- ×2 384 34.2 10.6 200 785/204 50.0 16.4 37.0 54.3 85.1 ×4 384 34.7 10.2 204 760/210 48.2 16.3 33.9 54.4 84.7 ×6 384 34.7 10.1 223 806/201 47.4 15.8 37.5 53.9 84.6 ×8 384 35.4 10.8 234 788/215 45.1 16.4 35.6 54.2 84.7

- (2) Scale Stage 3 Data: LLaVA (665K), LLaVA-Next (1M), LLaVA-One (3M) (fix Stage 2=×8, Res=384)

×8 384 LLaVA-Next (1M) 34.5 26.1 284 869/219 50.8 16.4 39.0 53.9 84.5 ×8 384 LLaVA-OneVision (3M) 36.3 31.3 319 1051/248 41.6 20.7 37.6 53.3 84.6

- (3) Scale Input Resolution: 384→448→512→672→768 (fix Stage 2=×8, Stage 3=LLaVA-OneVision (3M))

×8 448

37.0 34.9 333 907/246 41.3 18.1 36.8 53.5 85.0 ×8 512 38.2 37.2 347 886/226 39.3 20.8 39.0 53.9 86.0 ×8 672 38.3 43.2 355 1126/203 46.6 18.8 43.7 53.3 85.5 ×8 768 40.6 44.7 382 1080/242 45.8 22.0 39.5 53.2 86.3

LLaVA-OneVision (3M)

Tiny, +5.0% for Small, and +3.3% for Base). However, we would also like to point out that these gains come at a cost: the finer patchification substantially increases the number of visual tokens, leading to much higher memory consumption and latency.

#### 3.3. OpenVision-Smol: Tuning with a 150M LM

To push the portability of our vision backbones, we pair OpenVision with smol-LM—a 150 M-parameter language model (LM), currently the smallest available on Hugging Face [3]. Specifically, we pair OpenVision-B/16-384 with this Smol-150M, creating a multimodal system of fewer than 250M parameters—smaller than a ViT-L vision encoder on its own.

Three-stage training protocol. Following the training recipe of LLaVA-OneVision [19], we first pre-train the models with image-caption alignment (Stage 1), and then perform additional vision-language pre-training using synthetic instructions (Stage 2); lastly, we fine-tune on curated multimodal instruction datasets (Stage 3). To probe scaling behavior, we systematically vary three knobs while holding all other hyper-parameters fixed: (1) the size of the Stage2 instruction corpus, (2) the size of the Stage-3 instruction corpus, and (3) the input image resolution.

Main results. Table 6 reports the scaling results. Firstly, we can observe that enlarging the corpus in Stage 2 from ×1 to ×8 provides consistent gains on text-centric tasks such as TextVQA and OCR-Bench; although the gains flatten on reasoning-oriented suites like SEED-Bench and MMVet. Secondly, we notice that increasing data size in Stage 3 delivers a strong boost, especially in document-centric and chart reasoning tasks (e.g., ChartQA, OCR-Bench). Lastly, raising the input resolution from 384 px to 768 px leads to the largest overall improvements, particularly for OCR and complex reasoning benchmarks.

These results collectively confirm that our fully open OpenVision backbones retain strong scalability even when paired with a tiny 150 M-parameter language model. The resulting model family competitively offers a practical path to ultra-lightweight yet capable multimodal systems for real-world, resource-constrained deployments.

### 4. Ablation Studies

The results in Section 3 show that OpenVision rivals, and sometimes surpasses, proprietary vision encoders such as OpenAI’s CLIP and Google’s SigLIP. We now dissect the model to pinpoint the design choices that drive this performance.

100

Ours w/o Decoder

95.0

0.0

-1.0

| |
|---|

87.8

90

-0.4

-0.4

w/ Org. Cap.

80

+2.2

+0.9

73.1

73.1

-0.1

-0.4

Score

70

65.7

-0.8

-0.8

64.7

-0.2

-0.5

61.5

-1.1

-1.5

60

50.3

50

-2.8

-3.1

41.4

0.0

-0.1

40

TextVQA ChartQA OCR MME SEED MMVet SQA GQA POPE

Figure 2: Ablations on the impact of an auxiliary decoder and synthetic captions. Results show that both contribute to better performance across multimodal benchmarks. We present performance gaps between different variants and our setting.

- Table 7: Ablation study on our OpenVision visual encoder with different input resolutions resulting from the three-stage training pipeline, evaluated under the Open-LLaVA-Next setting.

Res. Text VQA Chart QA OCR. MME SEED MMVet SQA GQA POPE

84×84 64.4 63.1 508 1479/296 71.5 38.6 72.5 63.5 87.4 224×224 65.7 61.5 503 1567/332 73.1 41.4 73.1 64.7 87.8 336×336 68.3 68.0 547 1520/310 73.3 45.3 75.4 64.4 88.1

#### 4.1. Auxiliary Decoder and Synthetic Caption

Following CLIPS, OpenVision augments the standard contrastive objective with an auxiliary text decoder trained on the re-captioned Recap-DataComp-1B corpus. Although CLIPS demonstrated that this generative signal improves cross-modal retrieval, its impact on multimodal reasoning has not been examined. We close this gap with two ablations: 1) w/o Decoder: remove the text decoder and train with pure contrastive loss; and 2) w/ Orig. Caps: keep the decoder but replace synthetic captions with the original DataComp-1B captions.

- Figure 2 summarizes the findings. We can observe that

removing the decoder consistently degrades performance across most multimodal benchmarks, confirming that the generative objective supplies essential semantic supervision that the contrastive loss alone cannot provide. Additionally, replacing synthetic captions with the original, often noisy captions produces a similar drop, indicating that the richer, LLM-generated descriptions in Recap-DataComp-1B offer superior guidance for learning transferable visual features.

Takeaway. With the results above, we can confirm that both components—the auxiliary text decoder and the highquality synthetic captions—are critical to the strong multimodal performance of OpenVision.

#### 4.2. Progressive Resolution Pre-training

To significantly accelerate pre-training, OpenVision follows a three-stage curriculum that begins with very small crops and ends at 336/384 px. Prior works have shown that such schedules can accelerate CLIP training without hurting

performance [24, 23, 21, 26], but their downstream effect on multimodal performance—especially the contribution of the main low-resolution stages—has not been analyzed.

To investigate this, we assess the performance of OpenVision encoders produced at the end of each stage using our multimodal evaluation pipeline (see Table 7 and more details in Appendix A.1). Note that the LLaVA models built on these encoders use the same native resolution. Figure 3 also provides averaged multimodal performance against estimated training time and includes OpenAI’s CLIP as a reference point.

We highlight two key findings from these results. First, we can see that low-resolution pre-training is able to achieve competitive performance at a significantly reduced training cost. For example, the OpenVision encoder trained only at 84 × 84 resolution outperforms OpenAI’s CLIP that is trained at 224 × 224 resolution in the Open-LLaVA-Next setting while requiring roughly only half of the pre-training compute. Second, training with progressively increasing resolutions (OpenVision 336 × 336) not only yields better performance than training at high resolution from scratch (OpenAI-CLIP 336 × 336) but is also 3× more efficient in pre-training.

Takeaway. These results confirm that progressive resolution training yields vision encoders that are both performant and computationally efficient for multimodal learning.

#### 4.3. Extended High-Resolution Fine-Tuning

The next interesting question we explore here is how much additional compute should be invested in the highresolution stage. Using the number of image–text pairs

- Table 8: Ablation study on extending schedule higher-resolution fine-tuning in CLIPS pre-training as illustrated in Section 2. Doubling fine-tuning samples improves performance, especially in high-resolution tasks like OCR and ChartQA.

224 × 224 336 × 336 Text VQA Chart QA OCR. MME SEED MMVet SQA GQA POPE

512M 128M 68.6 66.1 513 1574/326 73.4 40.7 73.4 65.0 88.1 1024M 256M 68.3 68.0 547 1520/310 73.3 45.3 75.4 64.4 88.1

512M 512M 68.9 68.6 550 1548/323 74.0 44.9 73.9 64.6 88.3 0M 768M 69.1 68.2 554 1553/332 74.2 41.6 71.9 64.7 88.5

- Table 9: Ablation study on the Stage 1 & Stage 2 training data of small VLM. Results show that both contribute to better performance across multimodal benchmarks.

Stage 1 Stage 2 Stage 3 Text VQA Chart QA OCR. MME SEED MMVet SQA GQA POPE LCS-558K LLaVA-1.5 19.5 9.1 92 555/199 25.1 8.4 35.5 33.0 61.8

Recap-DataComp-558K LLaVA-1.5 20.7 9.0 59 600/211 24.1 9.5 35.0 33.6 68.9

LCS-558K OneVision-4M LLaVA-1.5 19.2 11.2 191 503/227 24.0 12.5 34.8 35.1 65.0 LCS-558K Recap-DataComp-4M LLaVA-1.5 24.9 10.6 213 720/210 25.5 15.3 34.6 37.6 72.9

Recap-DataComp-558K Recap-DataComp-4M LLaVA-1.5 26.5 10.5 136 618/242 26.2 16.7 36.5 38.6 72.7

##### Training Time vs Average Performance (LLaVA-1.5 vs Open-LLaVA-Next)

70.0

67.5

AveragePerformance

65.0

62.5

60.0

57.5

OpenVision ViT-L/14 (LLaVA-1.5)

OpenVision SoViT (LLaVA-1.5)

55.0

OpenAI CLIP (LLaVA-1.5)

SigLIP (LLaVA-1.5)

OpenVision ViT-L/14 (Open-LLaVA-Next)

OpenVision SoViT (Open-LLaVA-Next)

OpenAI CLIP (Open-LLaVA-Next)

SigLIP (Open-LLaVA-Next)

500 1000 1500 2000 2500 3000 3500

Training Time (hours)

- Figure 3: Comparison of training time and average multimodal performance between our OpenVision and OpenAICLIP on both LLaVA-1.5 and LLaVA-Next. Larger markers correspond to vision encoders with higher input resolutions. As a fully open and cost-effective vision encoder, OpenVision achieves higher performance with significantly less pre-training time.

processed as a proxy for training cost, our default CLIPSstyle schedule fine-tunes on 512 M samples at 224 × 224, followed by 128 M samples at 336 × 336. Doubling the budget, we compare three alternatives: (1) fine-tuning with 1024M samples at 224×224, followed by 256M samples at 336×336 (2) fine-tuning with 512 M samples at 224×224, followed by 512M samples at 336×336, and (3) fine-tuning entirely with 768M samples at 336 × 336.

As reported in Table 8, all three strategies lead to consistent improvements over the baseline. The largest gains are observed in fine-grained tasks such as OCR and ChartQA, where high-resolution details are especially critical. Interestingly, while all extended training strategies yield improvements, diminishing returns emerge when training exclusively at 336 × 336.

Takeaway. These results suggest that a balanced allocation across resolutions is more efficient, as lower-resolution finetuning helps establish general visual representations while high-resolution tuning refines fine-grained understanding capabilities.

#### 4.4. OpenVision + Smol-LM

Building on Section 3.3, we further analyze a tiny multimodal model that couples OpenVision-B/16-384 with Smol-LM (150 M parameters). To deeper our understanding of its training dynamics, we hereby probe two factors: data source and learning rate.

Regarding data source, Table 9 demonstrates that increasing the amount of data consistently improves performance, regardless of whether the corpus is OneVision or Recap-DataComp. Data quality, however, has a stronger effect than quantity. Substituting the 4 Msample OneVision subset [19] with an equally sized slice of Recap-DataComp [22] in Stage 2 yields substantial gains: TextVQA improves from 19.2 to 24.9, OCR-Bench from 191 to 213, and POPE from 65.0 to 72.9. Moreover, even a 558K-sample slice of Recap-DataComp outperforms the same-sized LCS baseline (e.g., TextVQA 19.5 → 20.7). This suggests that Recap-DataComp not only scales better but is also a more effective source in multimodal learning.

For hyperparameter tuning, we present detailed results in Appendix A.3. We can see that excessively low or higher learning rates degrade model accuracy, while an appropriately tuned learning rate is essential for maximizing performance, echoing the findings of [54].

Takeaway. In summary, our ablation studies emphasize that high-quality synthetic captions from Recap-DataComp and a moderate learning rate are critical is critical for maximizing the performance of this tiny multimodal models.

### 5. Discussions

From these experiments, we summarize three interesting observations on the vision encoder design when paired for multimodal learning:

- 1. Limited predictive value of CLIP benchmarks. Traditional CLIP evaluation tasks—such as ImageNet [11] classification accuracy and MSCOCO [27] image-text retrieval—do not reliably predict a vision encoder’s performance in multimodal models. For instance, as shown in Tables 2 and Table 1, despite achieving a superior MSCOCO retrieval performance compared to OpenAI-CLIP, both LAION-2B-CLIP and DataComp-1B-CLIP do not exhibit corresponding advantages on multimodal benchmarks. Additionally, DFN-2B-CLIP, which attains state-of-the-art accuracy on ImageNet, similarly fails to translate this strength into improved multimodal task performance. These results suggest that strong image classification or retrieval metrics fail to capture the qualities needed for a vision encoder to be effective in multimodal foundation models.
- 2. Crucial role of generative training (auxiliary decoder). The inclusion of an auxiliary text decoder with a generative loss (e.g., caption prediction) is essential for a vision encoder’s semantic understanding in multimodal models. To validate this observation, we conduct ablation experiments in Figure 2, comparing the performance of vision encoder when trained with and without the auxiliary decoder. Results clearly demonstrate that removing the decoder significantly deteriorates the multimodal performance, indicating that generative training substantially enriches the encoder’s learned visual representations beyond contrastive image-text learning alone. Specifically, the auxiliary decoder provides essential semantic supervision, allowing the encoder to acquire deeper visual insights beneficial for downstream multimodal reasoning tasks.
- 3. Benefits of training with synthetic captions. Utilizing synthetic captions during pre-training is beneficial for enhancing the vision encoder’s multimodal capabilities. We conduct ablation experiments in Figure 2 and demonstrate that replacing synthetic captions with original web-crawled captions results in a noticeable decline in multimodal performance, indicating that synthetic captions substantially enrich the learned visual representations beyond traditional web-crawled captions. Specifically, synthetic captions provide richer and more precise semantic supervision, enabling the vision encoder to achieve deeper visual understanding crucial for downstream multimodal reasoning tasks.

### 6. Related Works

Vision-Language Pre-training. Vision-language pretraining serves as a foundational strategy for multimodal learning. The popular archiectures include ViLBERT [33], CLIP [39], and ALBEF [20], which employ independent

encoders to separately process visual and textual inputs. Recent advances in vision-language pre-training have been driven primarily by the development of innovative loss functions. CoCa [50] combines contrastive and generative training objectives within a unified encoder-decoder framework. SigLIP [52] further improves the original CLIP model by adopting a pairwise sigmoid loss. AIMV2 [13] employs a multimodal autoregressive pre-training strategy, enabling large vision encoders to jointly model image and text tokens. CLOC [4] strengthens localized vision-language alignment by introducing region-level contrastive learning. Our work builds upon the recently proposed, fully-open CLIPS [31] framework, which enhances CLIP by utilizing synthetic captions to enrich textual representations.

Open Vision Encoder for Multimodal Learning. Advanced closed-source multimodal models, such as OpenAI’s GPT-4o [1, 38], Google’s Gemini [43], exhibit exceptionally strong vision language capabilities. However, because of their proprietary nature, the specifics of their visual processing mechanisms remain entirely unknown. Recently open-source community make efforts to proposed fully-opened multimodal large language models which even achieve better performance like InternVL [8]and LLaVAOneVision [19]. To develop high-performing MLLMs, the open-source community primarily focuses on curating high-quality, large-scale datasets, including visionlanguage alignment datasets [5, 22] and visual instruction datasets [30, 19, 44]. Meanwhile, others like [8, 9] concentrate on novel architectural designs to better integrate stateof-the-art vision encoders with LLMs. However, the selection of vision encoders is largely restricted to open-weight models such as CLIP [39] and SigLIP [52]. The challenge of training a fully open and high-performing visual encoder for MLLMs remains an open question.

### 7. Conclusion

This paper introduces OpenVision, a fully-open and cost-effective family of vision encoders designed to support the development of multimodal foundation models. Through extensive experiments, our OpenVision encoders demonstrate performance comparable to or surpassing widely used proprietary models like OpenAI’s CLIP and Google’s SigLIP. Furthermore, OpenVision scales flexibly in both model size and input resolution, making it suitable for deployment in diverse environments, ranging from large-scale computing infrastructures to edge devices. By releasing all model weights, code, and training data, we aim to foster research flexibility and drive further innovation in the community, paving the way for more transparent and adaptable multimodal foundation models.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4V(ision) system card. OpenAI Research Blog, 2023. 8
- [2] Ibrahim M Alabdulmohsin, Xiaohua Zhai, Alexander Kolesnikov, and Lucas Beyer. Getting vit in shape: Scaling laws for compute-optimal model design. Advances in Neural Information Processing Systems, 36:16406–16425, 2023. 11
- [3] Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Mart´ın Bl´azquez, Guilherme Penedo, Lewis Tunstall, Andr´es Marafioti, Hynek Kydl´ıˇcek, Agust´ın Piqueres Lajar´ın, Vaibhav Srivastav, et al. Smollm2: When smol goes big–data-centric training of a small language model. arXiv preprint arXiv:2502.02737, 2025. 5
- [4] Hong-You Chen, Zhengfeng Lai, Haotian Zhang, Xinze Wang, Marcin Eichner, Keen You, Meng Cao, Bowen Zhang, Yinfei Yang, and Zhe Gan. Contrastive localized languageimage pre-training. arXiv preprint arXiv:2410.02746, 2024. 8
- [5] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 8
- [6] Lin Chen and Long Xing. Open-llava-next: An opensource implementation of llava-next series for facilitating the large multi-modal model community. https://github. com/xiaoachen98/Open-LLaVA-NeXT, 2024. 3
- [7] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 2
- [8] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 8
- [9] Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nvlm: Open frontier-class multimodal llms. arXiv preprint, 2024. 8
- [10] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024. 2
- [11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 8
- [12] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023. 2, 3, 4
- [13] Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor Guil-

- herme Turrisi da Costa, Louis B´ethune, Zhe Gan, et al. Multimodal autoregressive pre-training of large vision encoders. arXiv preprint arXiv:2411.14402, 2024. 2, 8
- [14] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 2, 3
- [15] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. arXiv preprint arXiv:2304.14108, 2023. 2, 3, 4
- [16] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 3
- [17] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip. github, July 2021. 2, 3
- [18] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 3
- [19] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2, 5, 7, 8
- [20] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34:9694–9705, 2021. 8
- [21] Runze Li, Dahun Kim, Bir Bhanu, and Weicheng Kuo. Reclip: Resource-efficient clip by training with small images. arXiv preprint arXiv:2304.06028, 2023. 2, 6
- [22] Xianhang Li, Haoqin Tu, Mude Hui, Zeyu Wang, Bingchen Zhao, Junfei Xiao, Sucheng Ren, Jieru Mei, Qing Liu, Huangjie Zheng, et al. What if we recaption billions of web images with llama-3? arXiv preprint arXiv:2406.08478,

2024. 2, 7, 8

- [23] Xianhang Li, Zeyu Wang, and Cihang Xie. Clipa-v2: Scaling clip training with 81.1arXiv preprint arXiv:2306.15658,

2023. 2, 6

- [24] Xianhang Li, Zeyu Wang, and Cihang Xie. An inverse scaling law for clip training. In NeurIPS, 2023. 2, 6
- [25] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. 3
- [26] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking. In CVPR, 2023. 6
- [27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In

- ECCV, 2014. 8
- [28] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following, 2023. 3
- [29] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge. https://llavavl.github.io/blog/2024-01-30-llava-next/, January 2024. 2, 3
- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

- 2023. 1, 2, 8

[31] Yanqing Liu, Xianhang Li, Zeyu Wang, Bingchen Zhao, and Cihang Xie. Clips: An enhanced clip framework for learning with synthetic captions. arXiv preprint arXiv:2411.16828,

- 2024. 2, 3, 8

- [32] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12), Dec. 2024. 3
- [33] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 32, 2019. 8
- [34] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022. 3
- [35] Quentin Malartic, Nilabhra Roy Chowdhury, Ruxandra Cojocaru, Mugariya Farooq, Giulia Campesan, Yasser Abdelaziz Dahou Djilali, Sanath Narayan, Ankit Singh, Maksim Velikanov, Basma El Amel Boussaha, et al. Falcon2-11b technical report. arXiv preprint arXiv:2407.14885, 2024. 1
- [36] Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, May 2022. Association for Computational Linguistics. 2, 3
- [37] Meta LLaMA Team. Introducing Meta Llama 3: The most capable openly available LLM to date, 2024. 2
- [38] OpenAI. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 8
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 1, 3, 8
- [40] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 2, 4
- [41] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree

- Radhakrishnan, Yilin Zhao, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. In The Thirteenth International Conference on Learning Representations, 2025. 1
- [42] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 2, 3
- [43] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 8
- [44] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2025. 1, 2, 3, 8
- [45] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578, 2024. 1
- [46] Haoqin Tu, Chenhang Cui, Zijun Wang, Yiyang Zhou, Bingchen Zhao, Junlin Han, Wangchunshu Zhou, Huaxiu Yao, and Cihang Xie. How many unicorns are in this image? a safety evaluation benchmark for vision llms. arXiv preprint arXiv:2311.16101, 2023. 1
- [47] Feng Wang, Yaodong Yu, Guoyizhe Wei, Wei Shao, Yuyin Zhou, Alan Yuille, and Cihang Xie. Scaling laws in patchification: An image is worth 50,176 tokens and more. ICML,

2025. 4

- [48] Hu Xu, Saining Xie, Xiaoqing Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. In ICLR, 2023. 3, 4
- [49] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024. 4
- [50] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 8
- [51] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In ICML, 2024. 3
- [52] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training.

- In ICCV, 2023. 2, 3, 8
- [53] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmmseval: Reality check on the evaluation of large multimodal models, 2024. 3
- [54] Bingchen Zhao, Haoqin Tu, Chen Wei, Jieru Mei, and Cihang Xie. Tuning layernorm in attention: Towards efficient multi-modal llm finetuning. arXiv preprint arXiv:2312.11420, 2023. 7
- [55] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 1

### A. Appendix

#### A.1. Ablation w.r.t. Input Resolutions of the Vision Encoder

Following Sec. 4.2, we present model performance under the LLaVA 1.5 setting with varied input resolutions in Table 10. We draw a similar conclusion as the findings from the LLaVA-Next setting: a higher resolution into the vision encoder during training always help boost model performance on vision-language benchmarks.

- Table 10: Ablation study on our OpenVision visual encoder with different input resolutions resulting from the three-stage training pipeline, evaluated under the LLaVA1.5 setting.

Res. Text VQA Chart QA OCR. MME SEED MMVet SQA GQA POPE

84×84 50.4 12.1 231 1372/290 63.5 28.8 76.6 58.8 83.9 224×224 57.7 13.9 315 1487/317 69.5 35.2 73.6 62.9 86.4 336×336 61.2 15.7 339 1525/315 70.5 36.2 75.1 63.7 87.2

A.2. Visual Encoder Configuration

We present detailed visual encoder configurations in Table 11. We demonstrate the flexibility of our approach by scaling OpenVision up/down and varying patch size for different application scenarios, and by showcasing its adaptability even with very small language models.

- Table 11: Visual encoder configurations used in our paper. Model Size Patch Size Layers Width Heads #Params (M)

Tiny 16 or 8 12 192 3 5 Small 16 or 8 12 384 6 22 Base 16 or 8 12 768 12 86 Large 14 24 1024 16 303 SoViT-400M [2] 14 27 1152 16 412 Huge 14 32 1280 16 631

#### A.3. Ablation w.r.t. Learning Rate

We also conduct comprehensive ablations w.r.t. the learning rate and other hyper-parameters during VLLM training. In Table 12 shows that a mid-range learning rate setting of 5×10−5 (Stage 2 ViT) and 5×10−4 (Stage 3 LLM) achieves the best overall scores—TextVQA 33.2, MME 743/212, POPE 85.0 — whereas overly lower or higher rates degrade accuracy. Careful hyperparameter tuning is essential to maximize performance for practical and extensible multimodal pipelines.

Table 12: Ablation study on the Stage 2 & Stage 3’s learning rate. Results show that both contribute to better performance across multimodal benchmarks.

Stage 2 Stage 3 LLM Stage 3 ViT Text VQA Chart QA OCR MME SEED MMVet SQA GQA POPE

26.5 10.5 136 618/242 26.2 16.7 36.5 38.6 72.7 1e-5

1e-5 1e-5

32.8 10.2 171 806/213 48.7 16.7 37.8 54.2 85.4 3e-5 33.2 10.6 173 759/215 47.8 17.0 38.1 54.9 84.7 5e-5 33.2 10.3 194 743/212 48.8 15.8 38.2 54.2 85.0 7e-5 32.6 10.2 184 845/205 42.0 14.4 32.8 54.2 85.7 1e-4 32.5 9.4 165 734/211 48.1 14.4 37.8 53.1 85.4 3e-4 29.2 9.2 149 649/205 44.5 14.1 35.5 50.5 83.3

Frozen

5e-4

- 5e-4 25.4 9.8 86 684/205 27.1 10.7 34.9 49.4 81.1 7e-4 23.0 9.2 22 812/210 28.0 14.4 35.0 47.5 79.3 1e-3 22.5 9.2 20 656/206 27.5 11.2 34.3 44.7 77.5

- 5e-5

1e-5

26.1 10.0 147 672/221 24.8 15.8 34.1 39.4 78.2 3e-5 29.1 10.0 178 769/259 26.9 16.0 35.6 44.2 80.7 5e-5 29.6 10.0 176 797/240 27.3 15.8 35.7 46.3 82.1 7e-5 30.4 10.1 185 836/235 27.2 13.9 35.3 47.7 83.3 1e-4 31.7 9.8 185 876/260 27.4 13.9 35.5 9.4 84.3 3e-4 32.8 10.4 198 717/210 44.5 13.3 36.9 53.2 84.7 5e-4 33.2 10.3 194 743/212 48.8 15.8 38.2 54.2 85.0 7e-4 32.4 10.3 191 793/215 49.5 15.1 35.9 54.8 86.3 1e-3 32.1 10.8 202 808/247 50.2 15.3 31.6 55.4 85.5

Frozen

1e-6 21.7 9.3 32 705/223 27.2 12.5 34.9 46.2 79.23

5e-5 5e-4

5e-6 21.7 9.3 31 706/223 27.3 12.8 34.8 46.1 79.2

