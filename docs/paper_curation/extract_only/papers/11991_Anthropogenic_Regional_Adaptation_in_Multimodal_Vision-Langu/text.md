# arXiv:2604.11490v2[cs.AI]16Apr2026

## Anthropogenic Regional Adaptation in Multimodal Vision-Language Model

Samuel CahyawijayaQ1,2, Peerat LimkonchotiwatQ3,2, Tack Hwa WongQ4,2, Hitesh Laxmichand PatelQ5, Amit AgarwalQ5, Manuel Antonio RufinoQ6, Carlos Rafael CatalanQ6, Muhammad Reza QoribR7, Vicky FelirenR8, Holy LoveniaR2, Aye Hninn KhineR9,2, Frederikus HudiR10,2, David AnugrahaR11, Alham Fikri AjiR12,2, Romrawin ChumpuR13, Viet-Thanh PhamN14, Minghan WangN14, Mohamed Fazli ImamN12, Ruochen ZhangN15,2, Joseph Marvin ImperialN16,31, Khumaisa Nur’ainiN8, Do Xuan LongN13, Musa Izzanardi WijanarkoN8, Joel Ruben Antony MonizN17, Patrick Amadeus IrawanN12, Hanif Muhammad ZhafranN18, Isaiah FloresN19, Salsabila Zahirah PranidaN12, Jun KevinN20, Jostin Jerico RosalN21, Patricia Nicole MonderinN6, Kun Kerdthaisong22, Ahmad Mustafid23, My Chiffon Nguyen2, Natchapon Jongwiriyanurak24, Siva Worajitwannakul16, Haochen Li25, Adrian Xuan Wei Lim13, Bin Wang26, Muhammad Ravi Shulthan Habibi27,2, Lynnette Hui Xian Ng7, Mithil Bangera28, Yeshil Bangera28, Priyaranjan Pattnayak23, Dun Li Chan29, Sherissa Caren Djuniwar30, Cho Chan Myei Oo32, and Hee Ming Shan12

1Cohere 2SEACrowd 3AI Singapore 4Universiti Teknologi PETRONAS 5Oracle 6Samsung R&D Institute Philippines 7Carnegie Mellon University 8Monash University, Indonesia 9King Mongkut’s University of Technology Thonburi 10Nara Institute of Science and Technology 11Stanford University 12MBZUAI 13National University of Singapore 14Monash University, Australia 15Brown University 16University of Bath 17Mila - Quebec AI Institute 18Institut Teknologi Bandung 19Ateneo de Manila University 20Universitas Pelita Harapan 21Seoul National University of Science and Technology 22Thammasat University 23Independent 24University College London 25Nanyang Technological University 26MiroMind AI 27University of Indonesia 28University of New Haven 29INTI International University and Colleges 30Binus University 31National University Philippines 32ThoughtFull

Abstract. While the field of vision-language (VL) has achieved remarkable success in integrating visual and textual information across multiple languages and domains, there is still no dedicated framework for assessing human-centric alignment in vision-language systems. We offer two contributions to address this gap. First, we introduce Anthropogenic Regional Adaptation: a novel paradigm that aims to optimize model relevance to specific regional contexts while ensuring the retention of global generalization capabilities. Second, we present a simple, but effective

Q Main Contributors, R Major Contributors, N Notable Contributors

We release all artifacts generated within our work including training corpora, evaluation datasets, and models at https://huggingface.co/collections/SEACrowd/ sea-vl-phase-2-multimodal-vision-language-models-for-sea.

2 Cahyawijaya et al.

adaptation method named Geographical-generalization-made-easy (GGEZ), which utilizes regional data filtering and model merging. Through comprehensive experiments on 3 VL architectures: large vision-language models, text-to-image diffusion models, and vision-language embedding models, and a case study in Southeast Asia (SEA) regional adaptation, we demonstrate the importance of Anthropogenic Regional Adaptation and the effectiveness of GG-EZ, showing 5-15% gains in cultural relevance metrics across SEA while maintaining over 98% of global performance and even occasionally surpassing it. Our findings establish Anthropogenic Regional Alignment as a foundational paradigm towards applicability of multimodal vision-language models in diverse regions and demonstrate a simple-yet-effective baseline method that optimizes regional value alignment while preserving global generalization.

### 1 Introduction

Representation alignment in underrepresented regional contexts of foundation models has been a longstanding problem in AI [6,29,62], limiting their effectiveness and applicability across diverse global populations. This challenge is particularly acute in under-developed and developing regions – such as African [1,57], Indian [32, 55], Middle Eastern, Southeast Asian (SEA) [12, 15, 38, 60], etc – where existing resources exhibit severe underrepresentation. Building on this critical gap, we observe that existing vision-language (VL) AI solutions, despite their capabilities to generalize across different languages and domains, often exhibit cultural insensitivity [11,42,48], stereotypical outputs [16,36], and reduced task performance when deployed in underrepresented regions [3, 17, 31]. This misalignment stems from training data dominated by certain regions with limited exposure to diverse contextual nuances prevalent in other regions [2,10,13], undermining their real-world applicability.

To bridge this divide, we propose Anthropogenic Regional Adaptation, a foundational paradigm designed to systematically evaluate human-centric alignment of VL models across regional contexts while preserving global generalization capabilities. Within this framework, we identify two primary model archetypes:

(1) global models that achieve strong performance across broad geographic contexts but struggle with underrepresented regions, and (2) regionally specialized models that excel on locale-specific metrics, yet falter when confronted with broader global contexts. Neither archetype is ideal, creating a critical gap in the utilization of existing systems for targeted regional applications.

Overcoming this limitation, we specifically designed a simple yet effective method, Geographical-generalization-made-easy (GG-EZ), which adapts an existing global model to a regional-specific context with minimal degradation on the global context. Inspired by recent advancements in training strategies of large language models (LLMs) [18, 23, 33, 67], GG-EZ operationalizes regional adaptation through a two-level approach: (1) regional data filtering to curate culturally relevant training subsets, and (2) model merging to integrate regionspecific adaptations without catastrophic forgetting of global knowledge.

We validate Anthropogenic Regional Adaptation and GG-EZ through rigorous experiments across three VL architectures—large vision-language models (Gemma-3 27B), text-to-image diffusion models (SDXL), and vision-language embedding models (SigLIP-2) – focusing on SEA as a case study. Our results demonstrate that GG-EZ achieves 5-15% improvements in cultural relevance metrics, including cultural context accuracy and local visual context understanding, while retaining over 98% of global performance on standard benchmarks. These findings establish Anthropogenic Regional Adaptation as a critical step toward equitable vision-language solutions and position GG-EZ as a practical baseline for anthropogenic regional adaptation in vision-language systems.

### 2 Related Work

Regional-specific models have been reported to outperform one-size-fits-all models on regional context [45]. Several datasets [15,38] and language models [43,44] have been developed for the Southeast Asia (SEA) region. However, in computer vision, work on regional adaptation has focused more on remote sensing [47,56], which results in limited understanding of a region’s unique cultural and anthropomorphic characteristics. Cahyawijaya et al. [15] report that existing multilingual vision-language models such as MAYA-8B [5], PaliGemma-2-10B [50], Pangea-7B [66], Qwen2-VL-7B [58], Gemma-3 [52] fail in generating culturallyrelevant responses when benchmarked against regional-specific cultural benchmarks such as SEAVQA [54], CVQA [40] and World Cuisines [61]. On the other hand, recent development of regional-specific vision-language models such as VIOLET [41] and Baseer [27], VARCO-VISION [30], and SEA-LION-VL [49], have strong performance on regional-specific benchmark, but fall short on a broader and more general global context. Motivated by these limitations, we present a generic framework to systematically assess the gap of anthropogenic regional alignment in different models and introduce a simple-yet-effective approach, GGEZ, to reduce the regional gap of existing multi-modal vision-language models, while maintaining its performance on the broader global contexts. 1

### 3 Anthropogenic Regional Adaptation

#### 3.1 Overview

Let Rglobal denote the comprehensive spatial domain representing our global context of interest. Within this universal space, we establish a discrete partitioning into meaningful geographical or conceptual units through the definition of R, a finite collection of k distinct regions. Formally, R = {R1,R2,...,Rk} where each individual region Ri constitutes a measurable subset of Rglobal. This decomposition enables systematic analysis of adaptation strategies across heterogeneous spatial contexts, recognizing that the world is rarely homogeneous in its characteristics or response patterns to anthropogenic influences.

1 We provide comparison of our SEA-VLM models with other models in Appendix D.

[Figure 1]

###### 4 Cahyawijaya et al.

Global models: Broad coverage, gaps in underrepresented regions

Regionally specialized models: Local excellence, failure in broader global contexts

[Figure 2]

| |
|---|

[Figure 3]

| |
|---|

Well-aligned Low performance

###### Well-aligned Low performance

- Fig. 1: Through anthropogenic regional adaptation, we identify two primary model archetypes: (left) Global model with strong overall global performance, but struggle to represent certain regions appropriately, and (right) Regional-specific model that has a strong representation towards certain regions, but fall short on the global context.

Building upon this foundational regional partitioning, we introduce a critical distinction between regions based on strategic importance. The target region Rregional is defined as a specific, non-empty subset of Rglobal representing our primary focus region where adaptation efforts should be maximized. Complementarily, the remaining regions are collectively designated as Rothers, defined through set difference as Rothers = Rglobal \Rreg. This ensures complete domain coverage while maintaining clear analytical separation between priority and nonpriority regions, with each point in Rglobal belonging to exactly one region.

Building upon this theoretical framework, as depicted in Figure 1, existing vision-language models predominantly follow two archetypal approaches. Global models capture broad, universal patterns across Rglobal but sacrifice nuanced regional specificity. Conversely, regional-specific models focus on particular Rregional subsets with high local performance but reduced global coherence. This dichotomy reflects a fundamental trade-off in model design between universal applicability and specialized excellence. Most existing models, commit to one of the paradigm: global optimization for comprehensive coverage or regional specialization for enhanced regional performance.

#### 3.2 Global-Regional Parity (GRP) Optimization

= {eR

1 ,eR

For each region Ri, we have a set of evaluation metrics ER

mi}.

2 ,...,eR

i

i

i

We define a quality metric qR of these quality metrics forms the set QR

- i
- j for each evaluation metric eR

- i
- j , and the collection

= {qR

1 ,qR

mi}. This multidimensional approach captures the complexity of regional characteristics influencing adaptation outcomes. The evaluation framework extends to our partitioned structure, yielding aggregated evaluation sets QR

2 ,...,qR

i

i

i

global

regional

that consolidate metrics across regions while maintaining connections to individual characteristics. To balance competing objectives, we introduce a globalization factor α ∈ [0,1] that explicitly controls the trade-off between optimizing the target region and maintaining global generalization. The formal optimization

and QR

###### Phase1:RegionalQualityFiltering Phase2:Global-RegionalRefinement

Global model

Global VL data in regional languages

Global VL data in English

Anthropogenic regional adapted model

[Figure 4]

[Figure 5]

[Figure 6]

Region-specific SFT VL corpora

Translate

| |
|---|

###### GG-EZ

Supervised fine-tuning

Merge

[Figure 7]

[Figure 8]

Regional filter

[Figure 9]

+

| |
|---|

Regional model

Quality filter

Worldwide cultural VL data

Region-specific cultural VL data

- Fig. 2: Overview of our Geographical-generalization-made-easy (GG-EZ) framework. Our framework consist of 3 constituents: (1) High-quality regional data filtering pipeline; (2) supervised fine-tuning to create a high quality regional-specific model; and (3) model merging to capture the best combination between regional-specific and global represention while also maintaining the generalization capabilities of the model.

objective is constructed as a weighted combination: max

global

regional

α · QR

+ (1 − α) · QR

θ

This formulation creates a single scalar objective that can be optimized using standard techniques, with α serves as a critical control mechanism that can be adjusted based on the specific focus and characteristics of each region. For regions with strong global connections or dependencies, α can be set closer to 1 to emphasize maintenance of Rglobal conditions, while for regions where local impacts dominate, α can approach 0 to prioritize Rregional optimization. The solution achieves "best parity" between regions, representing the most favorable equilibrium where neither aspect is disproportionately favored according to our specified weighting scheme.

### 4 Geographical Generalization Made Easy (GG-EZ)

Geographical-generalization-made-easy (GG-EZ) is a general framework for regional adaptation of global multimodal models that achieves effective regionspecific performance while preserving original generalization capabilities. As il-

- lustrated in Figure 2, the core mechanism of GG-EZ consists of two phases: regional quality filtering and global-regional refinement. GG-EZ not only enables adaptation across local textual and imagery contexts while maintaining base model capabilities, but also better yet being architecture-agnostic, allowing effective regional adaptation across different vision-language architectures.

- 4.1 Regional Quality Filtering

The regional quality filtering phase refines the training dataset to focus on regionally relevant examples while preserving general knowledge. Let D = {Dreg,Dgen}

represent the complete dataset, where Dreg contains region-specific examples and Dgen contains general-domain examples.

Regional Filter (Frf): We define a boolean regional filter function that selects examples from the target region:

Frf(x,r) =

1 if region(x) = r 0 otherwise

where r denotes the target region and region(x) returns the geographical region associated with example x.

Multilingual Reward (Frm): To further refine data filtering, we employ reward models that scores data quality and relevance on a continuous scale:

Frm(x) = θreward(x)

where θreward is the weight of the reward model and θreward(x) measure the quality of the data x using the reward model θreward. The filtered regionalspecific dataset is then obtained by applying a threshold τ:

Dfiltered = {x ∈ Dreg | Frf(x,r) = 1 and Frm(x) > τ}

Language Translation Augmentation: Beyond regional filtering of existing cultural-relevant corpora, a further data enrichment is done through translating English high-quality datasets to multiple target regional languages. Let T represent the translation function mapping English to regional languages Lr:

Dtranslation = {T(x,English → l) | x ∈ Dfiltered,l ∈ Lr}

#### 4.2 Global-Regional Refinement

Using a pre-trained global model and the resulting corpora from the regional quality filtering phase, two steps of global-regional refinements are incorporated: 1) supervised fine-tuning, which turns the global model into a highlt performant region-specific model, and 2) model merging to ensure the preservation of the original global generalization capabilities. We first trained the global model θglobal into a quality regional-specific model θregional using Dsft ⊆ {Dfiltered ∪ Dtranslation}. After supervised fine-tuning on Dsft, a linear model merging technique is applied to combine the global model with the region-specific model. The merged model parameters θmerged are defined as:

θmerged(β) = β · θregional + (1 − β) · θglobal

where β controls the interpolation between the region-specific and the original global model. For beta interpolation, we first select a certain value for the globalregional parity factor α (e.g., α = 0.65) to balance region-specific performance and general capabilities. The optimal β value is selected based on evaluation metrics:

regional

global

θmerged(β) + (1 − α) · QR

β∗ = argmax

α · QR

θmerged(β)

α∈[0.0,1.0]

regional

where QR

θmerged(β) measures regional-specific performance from the evaluation sets ER

regional

global

and QR

θmerged(β) measures global performance from the evaluation sets ER

global

. This approach ensures minimal degradation of base model capabilities while achieving strong regional adaptation.

### 5 Case Study on Southeast Asian (SEA) Adaptation

We explore GG-EZ on 3 distinct multimodal architectures: a large-scale VisionLanguage Model (VLM) with 27B parameters, a contrastive Vision-Language Embedding Model (VL-Embed) with 1B parameters, and a Contextualized Diffusion Model (SEA-ImageGen) with 3.5B parameters. Our experiments focus on the SEA region, encompassing 11 countries—Singapore, Indonesia, Malaysia, Brunei, Thailand, Philippines, Vietnam, Myanmar, Cambodia, Laos, and East Timor—with a total population of approximately 700 million. In this section, we provide detailed configurations for applying GG-EZ and the evaluation strategy used to assess its effectiveness.

#### 5.1 Data Curation

For data curation, we leveraged prefiltered SEA-specific content from SEA-VL, culturally relevant imagery from CulturalGround [46], and translated instruction data from MAmmoTH-VL [24] (5 shards converted to major SEA languages, including Indonesian, Malaysian, Thai, Vietnamese, Filipino, Khmer, Lao, Chinese, and Tamil). To ensure high-quality translation on the underrepresented languages like Khmer and Lao, we ablate different translation models for each language and select the one with optimal quality and cost 2. The data filtering process applied a regional filter to isolate SEA-specific examples, utilized the UnifiedReward [59] model for quality assessment with a threshold set at 3 or above 3, and combined regional and translated instruction data to create the final fine-tuning dataset. This dataset curation is crucial to enable adaptation of a general global model into a strong regional-specific model as explained in §6.2.

#### 5.2 Regional Supervised Fine-tuning

All models underwent supervised fine-tuning using the AdamW optimizer [37] with linear learning rate decay. The SEA-adapted VLM (SEA-VLM) was trained from the Gemma-3 model as the initial checkpoint with a batch size of 64, learning rate of 2e-5, weight decay of 0.01, and 3 epochs. Before the supervised fine-tuning, we perform continuous pre-training on the Gemma-3 checkpoint using SEA-VL, XM3600, and Flickr30k, where we translated these datasets into

- 2 See Appendix A for more detail.
- 3 We pick the best reward model based on our ablations on a human-annotated SEAspecific human preference dataset. The comparison of different reward model quality on the SEA regional test set is shown on Appendix B.

SEA languages similar to Section 5.1. For the contextualized diffusion model (SEA-ImageGen), the model was fine-tuned from SDXL with a batch size of 32, learning rate of 1e-5, weight decay of 0.01, and 4 epochs. We only finetuned the UNet module the model while keeping the VAE module as is. For the VL embedding model (SEA-VL Embed), the model was fully fine-tuned from SigLIP2-SO400m with a batch size of 128, learning rate of 5e-6, weight decay of 0.001, and 2 epochs. The resulting regionally-adapted models are denoted as SEA-Gemma-3, SEA-SDXL, and SEA-SigLIP2 for the VLM, diffusion, and embedding architectures respectively.

#### 5.3 Global-Regional Merging

After fine-tuning, we performed linear model merging to combine region-specific adaptations with the original base models. For VL-Embed and SEA-ImageGen, we explored interpolation weights β in the range [0.25, 0.5, 0.75], while for the SEA-VLM we tested a broader range of β values [0.05, 0.10, 0.5, 0.7]. The optimal β was determined by maximizing a weighted combination of regional and general performance metrics, where RegionAcc measures performance on SEA-specific validation data and GeneralAcc evaluates performance on broad-domain benchmarks. This formulation prioritizes regional adaptation while ensuring minimal degradation of foundational capabilities. We denote the resulting merged model as “<MODEL> (X%)” – e.g., SEA-Gemma-3 10% – where X% denotes the weight percentage β of the regionally-specific model used in the linear merging.

#### 5.4 Evaluation

For the SEA-VLM model, evaluation included the SEAVQA regional visual question answering benchmark, CVQA for culturally nuanced queries, WorldCuisine for food identification tasks, and human evaluation to assess cultural relevance and accuracy. 4. The SEA-VL Embed was evaluated on SEAVQA and CVQA to measure regional and cultural performance in embedding space. The SEAImageGen model underwent assessment using a DPGBench benchmark [28] as a proxy for the global quality of the models and diffusion model human evaluation set from SEA-VL [16] to gauge cultural relevance and visual quality on the SEA region. All evaluations were conducted on held-out test sets to ensure fair comparison across architectures and adaptation strategies. For human evaluation we use a 3-level likert scoring with native speaker annotators for each of the evaluated regional language. Finally, to compute a fair and practical global-regional parity (GRP) score from both of the global and region-specific evaluations, we set the globalization factor α = 0.43 5 derived from 2023 KOF Globalization Index [25,26] which measures the level of globalization across 190 countries.

- 4 Due to limited evaluation resources for the region, we develop two human evaluation sets: (1) SEA-AYA a human translated AyaVisionBench [19]; and (2) SEA-VL VQA, a human annotated VQA prompts from SEA imagery. Details in Appendix C.
- 5 α = 0.43 indicates a moderate tendency toward SEA-focused quality QRregional with a fair amount of consideration for global quality QRglobal.

- Table 1: SEA-VLM evaluation results on CVQA, SEA-VQA, and WorldCuisine (WC). <MODEL> (X%) denotes a merged model where X% is the weight β use in the linear merging process of the region-specific model.

Model GRP

Global SEA-Specific Avg. WC CVQA Avg. SEAVQA WC CVQA

Google Gemma-3 59.4 63.5 59.8 67.2 56.3 41.0 60.1 67.8 SEA-Gemma-3 5% 64.0 64.3 60.0 68.7 63.7 61.2 60.3 69.5 SEA-Gemma-3 10% 64.1 64.4 60.0 68.8 63.8 61.7 60.2 69.5 SEA-Gemma-3 50% 57.3 56.7 51.6 61.8 57.8 59.5 51.4 62.6 SEA-Gemma-3 70% 56.1 56.3 51.9 60.6 56.0 54.0 52.6 61.3 SEA-Gemma-3 (w/o merge) 42.2 42.1 48.5 35.6 42.2 41.9 48.6 36.2

- Table 2: SEA-VLM human evaluation on SEA-AYA and SEA-VL VQA datasets. We report the average rank across different models (Higher is better). <MODEL> (X%) denotes a linear merge where X% is the weight β of the region-specific model.

Overall Language Breakdown

Model

GRP Global SEA fil ind tha vie zsm Google Gemma-3 2.29 2.54 2.09 1.69 2.15 2.17 2.37 2.07 SEA-Gemma-3 10% 2.31 2.42 2.22 1.88 2.07 2.29 2.61 2.25 SEA-Gemma-3 (w/o merge) 1.74 1.18 2.23 2.75 2.29 2.33 1.76 2.00

### 6 Result and Discussion

#### 6.1 Overall Results

GG-EZ on SEA-VLM We observe two major trends in SEA-Gemma-3 performance, as illustrated in Table 1. First, without any model merging, SEAGemma-3 generally performs worse compared to the original Gemma-3 model as evidenced in both the CVQA and World-Cuisine benchmarks, possibly due to some loss of task generalization capability after the supervised fine-tuning with regional data . Nevertheless, SEA-Gemma-3 is still able to outperform the original Gemma-3 model when faced with a challenging SEA-specific regional evaluation, i.e., SEA-VQA. This signifies the only applying supervised fine-tuning to perform regional adaptation might not results in an ideal performance as pre-trained general VLMs are usually already trained on large subset of diverse high-quality datasets across languages, regions, and tasks, and fine-tuning them for with smaller regional-specific data might harm their generalization capability.

Interestingly, when model merging is applied into SEA-Gemma-3„ we observe significant performance improvements across all evaluation scenarios with 50:50 mix of the original Gemma-3 and SEA-Gemma-3 models (SEA-Gemma-3 50%), we attain a turning point, where the global performance is only slightly lower, while the SEA-specific performance surpasses the original Gemma-3 model. Pushing the merging weight β further to 10% brings further improvement, with the SEA-Gemma-3 10% outperforms the original Gemma-3 model on all evaluation

- Table 3: SEA-SDXL Performance on Image Generation Benchmark DPGBench (Higher is better). <MODEL> (X%) denotes a linear merge where X% is the weight β of the region-specific model.

Overall Aspects Score Attribute Relation Entity Other Global

Model

StabilityAI SDXL 73.75 79.18 86.38 81.07 60.00 83.89 SEA-SDXL 25% 74.75 80.43 86.34 81.86 61.60 85.41 SEA-SDXL 50% 74.61 80.93 87.12 81.15 60.00 86.02 SEA-SDXL 75% 74.39 79.98 85.96 81.36 63.20 82.67 SEA-SDXL (w/o merge) 74.32 80.79 86.92 81.57 65.20 81.16

sets, improving the averaged global performance by 1% while significantly improve the average SEA-specific performance by 7.5% leading to a much higher GRP compared the source global and region-specific models. This result underscores the advantage of GG-EZ to enable regional-adapted models that are strong on both region-specific aspect and generalization capability.

The human evaluation results in Table 2 further validate the automatic evaluation trends of GG-EZ. The result reveals that SEA-Gemma-3 (w/o merge) achieves superior regional specialization, securing the best average rank across Southeast Asian languages—including Filipino (2.75), Indonesian (2.29), Thai (2.33) and the broader SEA region (2.23). These findings corroborate our earlier observation that SEA-Gemma-3 excels on region-specific evaluations where Gemma-3’s global optimization proves to be less effective. Conversely, Google Gemma-3 maintains dominance in global performance (2.54), confirming that SEA-Gemma-3’s regional advantages come at the cost of broader generalizability. The SEA-Gemma-3 10% variant occupies an interesting middle ground, achieving the best Vietnamese score (2.61) and Malaysian performance (2.25) with minimal degradation on the global performance (2.42) compared to (2.54) on the original Gemma-3 model yielding highest GRP score of 2.31. This suggests that even modest SEA integration yields measurable regional benefits. 6.

GG-EZ on SEA-ImageGen As illustrated in Table 3, the resulting SEASDXL models also demonstrate strong regional adaptation while preserving strong general image generation performance on DPGBench which is used as the proxy of the global performance. the fully fine-tuned SEA-SDXL model shows a slight improvement in most aspects, achieving an overall performance of 74.32 compared to the original SDXL with 73.75. Similar with SEA-VLM, the linear merging incorporated in GG-EZ further improve the image generation capability, with the SEA-SDXL 25% and 50% models surpassing the general model and the regional-specific SEA-SDXL model achieving 74.75 and 74.61 on DPGBench, respectively, with competitive scores across all aspect categories.

- 6 We provide more detailed results and samples outputs from different models from

- our experiments in Appendix E and Appendix F.

- Table 4: SEA-SDXL human evaluation results on 3 distinct cultural aspects. “T” denotes Tradition, “L” denotes Landmark, and “C” denotes Cuisine

Correctness Naturalness Overall T L C Overall T L C

Model

StabilityAI SDXL 1.491 1.470 1.636 1.387 1.675 1.436 2.023 1.613 SEA-SDXL 25% 1.569 1.587 1.729 1.413 1.767 1.473 2.124 1.753 SEA-SDXL (w/o merge) 1.431 1.473 1.527 1.307 1.557 1.340 1.806 1.560

The regional-specific human evaluation results (Table 4) further validate the effectiveness of the merging strategy for regional context. The SEA-SDXL 25% model demonstrates superior performance across all cultural aspects (Tradition, Landmark, Cuisine) in both correctness and naturalness metrics, outperforming both the original SDXL and the fully fine-tuned SEA-SDXL This indicates that the merging process successfully recovers the generalization capability lost during regional specialization, allowing the resulting SEA-SDXL models to maintain or enhance their performance on general image generation benchmarks while simultaneously excelling in regional context evaluations. 7

GG-EZ on SEA-VL Embed Aligned to the result of SEA-ImageGen, the developed SEA-SigLIP2 demonstrate an interesting performance trend where, as shown in Table 5, the regional fine-tuned SEA-SigLIP2 model perform better on both SEA and non-SEA (other) regions in both CVQA and SEA-VQA evaluations compared to the original Google SigLIP2 model. Moreover, merging the SEA-SigLIP2 model back with the original SigLIP2 model brings additional boost of the evaluation performance. The SEA-SigLIP2 50% model excels in general knowledge transfer, achieving the highest global CVQA score (27.52) and strongest performance on non-SEA regions (27.97), While the SEA-SigLIP2 75% model demonstrates superior regional specialization with the highest SEAVQA overall score (29.66) and peak performance for Indonesia (30.05) and Vietnam (28.75), while at the same time maintaining a high global performance (27.12) yielding the highest GRP score of 27.96 compared to the original SigLIP2 model (25.17) and the fine-tuned SEA-SingLIP2 (26.96). 8 The merging approach effectively balances the trade-off between regional adaptation and generalization, allowing the models to preserve and enhance the original SigLIP2 broad capabilities while incorporating specialized regional knowledge from SEA-SigLIP2.

#### 6.2 Impact of Regional Quality Filtering

The quality and relevance of data utilized for region-specific model adaptation significantly dictate performance outcomes [13,14]. Our ablation study on SEA-

- 7 We do not measure the GRP score for image generation as there is no comparable test sets with the comparable value scale. However, it is clear that the resulting model improves both regional-specific and general image generation scores.
- 8 We provide more detailed breakdown of all the evaluations in Appendix E.

- Table 5: SEA-VL Embedding Zero-Shot Performance on Cultural VQA Benchmarks. <MODEL> (X%) denotes a linear merge where X% is the weight β of the regionspecific model. For SEA-VQA, we provide additional breakdown of 3 of the biggest country data in the dataset with “Khm”=Cambodia, “Ind”=Indonesia, “Vie”=Vietnam.

CVQA SEAVQA Global SEA Others SEA Khm Ind Vie

Model GRP

Google SigLIP2 25.17 25.51 24.02 25.84 25.81 28.95 25.00 25.56 SEA-SigLIP2 25% 24.96 24.38 24.44 24.36 26.36 27.63 25.40 23.96 SEA-SigLIP2 50% 27.10 27.52 25.50 27.97 28.06 31.25 26.99 27.16 SEA-SigLIP2 75% 27.96 27.12 27.51 27.03 29.66 28.62 30.05 28.75 SEA-SigLIP2 (w/o merge) 26.96 26.75 25.29 27.07 28.96 28.29 28.59 29.07

Gemma-3 demonstrates that naive data augmentation does not guarantee improvements; rather, performance gains depend heavily on the specific structure, scope, and provenance of the integrated data.

As shown in Figure 3, data volume acts as a primary bottleneck. Utilizing only a 20% subset of the baseline dataset, which consists of the SEA-Mammoth 250k dataset translated into 10 regional languages, leads to a severe 70% degradation in overall model performance relative to the full dataset, emphasizing the necessity of maintaining adequate training scale. As for subsequent integration of culture-specific knowledge, i.e., the SEA-filtered Cultural-Ground dataset [46], reveals that performance is highly contingent on the VQA task format. As il-

- lustrated in Figure 3, the open-ended VQA formulation effectively enhances the model’s regional understanding from the baseline to 41.9%, albeit with high sensitivity to training prompt design. Conversely, appending the multiple-choice variant of the same data source actively degrades performance to 21.6%. Similarly, incorporating the SEA-filtered WorldCuisine dataset [62] proves harmful, resulting in a ∼42% performance reduction compared to the baseline. We hypothesize that this degradation stems from WorldCuisine’s specialization in regional cuisine, which excessively narrows the model’s representational focus and fails to provide the comprehensive knowledge required for adaptation to other cultural aspects, e.g., landmarks, history, traditional clothing, and others.

These findings highlight the critical importance of strategic data curation. Indiscriminately appending datasets without rigorous, format- and domain-aware evaluation can compromise the model’s capacity for generalizable learning.

#### 6.3 Globalization Factor in Anthropogenic Regional Adaptation

The globalization factor α plays a huge role in the GRP optimization of Anthropogenic Regional Adaptation. This factor determine how a model should be optimized to achieve the optimal characteristic of representing a particular target region. An appropriate globalization factor α would reflect the actual condition on how society within the targeted regions interact with others, whether

Country

Only used a subset (20%)

ID KH LA

SEA-Mammoth250k

Modificationsto

| | | |
|---|---|---|
| | | |

Added Cultural-Ground Multiple-Choice

MY PH SG TH VN

Added World Cuisine

Added Cultural-Ground Open-Ended

10.15 15.15 20.15 25.15 30.15 35.15 40.15

45.15 50.15

(Baseline dataset)

SEA-VQA Accuracy (%)

Fig. 3: Impact of regional-specific data curation strategy on SEA-Gemma-3.

individuals prefer a single unified global perspective or a more unique and localized regional perspective. The misalignment of globalization α could result in undesirable behavior of the adapted model as shown in Figure 4 (left).

To accommodate different preferences across different regions, we derive the globalization factor α based on globalization index [7, 21, 39] which has been widely explored in the field of social science to measure the degree of globalization across different regions or countries. More specifically, we employ one prominent globalization index known as Konjunkturforschungsstelle Globalization Index (KOF-GI) [25,26] . This is done by calculating the expectation of the KOF-GI across all countries within the target region Rregional. 9. This formulation of the globalization factor α provides a timely, quantifiable, and standardized measure of globalization intensity for each region. By doing so, we establish a fair globalregional parity baseline that reflects how embedded globalization is within that specific region. This has significant implications: 1) as the globalization index evolves over time – as shown in Figure 4 (right) –, the GRP value naturally shifted, dynamically adjusting the optimization balance; and 2) the KOF-GI derived α provides an objective, external metric that reduces subjectivity in determining regional importance, making the process more data-driven.

The integration of KOF-GI-based α directly addresses the model archetype dichotomy illustrated in Figure 1. By providing a globalization-informed weighting scheme, it helps bridge the gap between purely global models (which struggle with regional specificity) and purely regional models (which lack global coherence). Models optimized with KOF-GI derived α can better adjust their focus—emphasizing regional performance in various regions with lower globalization parity while maintaining stronger global consistency in highly interconnected regions. This creates an adaptive system that can simultaneously satisfy the competing demands of universal applicability and specialized excellence leading to solutions that perform robustly across heterogeneous regional contexts. 10

- 9 Specifically, we use the “de facto interpersonal” component of KOF Globalization Index as our anchor for GRP, as this aspect measure the globalization in terms of person-to-person interaction which is a significant aspect for human-centric systems.
- 10 To help future research working on Anthropogenic Regional Adaptation in visionlanguage, we provide the breakdown of the Globalization Index in Appendix G.

Google Gemma 3 SEA-Gemma 10% SEA-Gemma (w/o merge)

2.6

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

GRPScore(HumanEval)

2.4

2.2

- 1.8
- 2

1.6

1.4

1.2

0 0.2 0.4 0.6 0.8 1

GRP Factor

Google Gemma 3 SEA-Gemma 10% SEA-Gemma 50% SEA-Gemma 70%

- 56
- 57
- 58
- 59
- 60
- 61
- 62
- 63
- 64
- 65

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

GRPScore(AutomaticEval)

0 0.2 0.4 0.6 0.8 1

GRP Factor

East Asia & Pacific Europe & Central Asia Latin America Middle East & North Africa North America South Asia Southeast Asian Sub-Saharan Africa World

| | | |
|---|---|---|
| | | |

80

70

GlobalizationIndex

60

50

40

30

1993 1998 2003 2008 2013 2018 2023

Year

Fig. 4: (left) Impact of globalization factor alpha to GRP across different models. Optimizing on a misaligned α can lead to suboptimal performance. (right) We derive α from the KOF globalization index [25,26] to better reflect the degree of globalization across regions. The globalization index is distinct across regions and evolves over time.

### 7 Conclusion

We propose Anthropogenic Regional Adaptation framework which addresses the critical challenge of cultural insensitivity and reduced performance in underrepresented regions by establishing a foundational paradigm for human-centric alignment in vision-language systems. The proposed approach, GG-EZ, effectively implements this framework, demonstrating through rigorous testing across three major architectures—Gemma-3 27B, SDXL, and SigLIP-2— on a comprehensive SEA case study, which specifically demonstrated 5-15% gains in cultural relevance metrics across SEA contexts while maintaining or even improving its generalization capabilities, enabling equitable deployment of multimodal models across diverse populations without sacrificing core generalization. Our work demonstrates the importance of GRP in human-centric alignment allowing for a data-driven and dynamically adjusted optimization.

### 8 Acknowledgements

We extend our gratitude to all affiliated institutions, particularly Oracle and AISingapore, for providing the compute resources essential to this project. We express our deepest appreciation to Sarana Nutanong, Genta Indra Winata, Fajri Koto, Derry Wijaya, Muhamad Risqi U Saputra, Gholamreza Haffari, Enshiun Annie Lee, and William Tjhi for their invaluable assistance throughout the vari-

- ous phases of this project, including idea discussion, information dissemination, financial support, and other invaluable supports. Additionally, we wish to acknowledge Daryl Peralta, Robert Wijaya, Pume Tuchinda, and Ong Lip Wei for their dedicated support and expertise in dataset annotations and human evaluations. We acknowledge the support of the Google Cloud Research Credits

program (Award No. GCP19980904), which provided cloud computing credits used for data set translation. This research is also supported by the National Research Foundation, Singapore, under its National Large Language Models Funding Initiative. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of the National Research Foundation, Singapore.

### References

- 1. Adelani, D.I., Abbott, J., Neubig, G., D’Souza, D., Kreutzer, J., Lignos, C., PalenMichel, C., Rijhwani, S., et al.: Masakhaner: Named entity recognition for african languages. Transactions of the Association for Computational Linguistics 9, 1116– 1131 (2021)
- 2. Adilazuarda, M.F., Mukherjee, S., Lavania, P., Singh, S.S., Aji, A.F., O’Neill, J., Modi, A., Choudhury, M.: Towards measuring and modeling “culture” in LLMs: A survey. In: Al-Onaizan, Y., Bansal, M., Chen, Y.N. (eds.) Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. pp. 15763–15784. Association for Computational Linguistics, Miami, Florida, USA (Nov 2024). https://doi.org/10.18653/v1/2024.emnlp-main.882, https: //aclanthology.org/2024.emnlp-main.882/
- 3. Agarwal, A., Meghwani, H., Patel, H.L., Sheng, T., Ravi, S., Roth, D.: Aligning LLMs for multilingual consistency in enterprise applications. In: Potdar, S., RojasBarahona, L., Montella, S. (eds.) Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track. pp. 117–137. Association for Computational Linguistics, Suzhou (China) (Nov 2025). https://doi.org/10. 18653/v1/2025.emnlp-industry.9, https://aclanthology.org/2025.emnlpindustry.9/
- 4. Agrawal, P., Antoniak, S., Hanna, E.B., Bout, B., Chaplot, D., Chudnovsky, J., Costa, D., Monicault, B.D., Garg, S., Gervet, T., Ghosh, S., Héliou, A., Jacob, P., Jiang, A.Q., Khandelwal, K., Lacroix, T., Lample, G., Casas, D.L., Lavril, T., Scao, T.L., Lo, A., Marshall, W., Martin, L., Mensch, A., Muddireddy, P., Nemychnikova, V., Pellat, M., Platen, P.V., Raghuraman, N., Rozière, B., Sablayrolles, A., Saulnier, L., Sauvestre, R., Shang, W., Soletskyi, R., Stewart, L., Stock, P., Studnia, J., Subramanian, S., Vaze, S., Wang, T., Yang, S.: Pixtral 12b (2024), https://arxiv.org/abs/2410.07073
- 5. Alam, N., Kanjula, K.R., Guthikonda, S., Chung, T., Vegesna, B.K.S., Das, A., Susevski, A., Chan, R.S.Y., Uddin, S.M.I., Islam, S.B., Santhosh, R., A, S., Sharma, D., Liu, C., Chaturvedi, I., Winata, G.I., S, A., Mukherjee, S., Aji, A.F.: Maya: An instruction finetuned multilingual multimodal model (2024), https://arxiv. org/abs/2412.07112
- 6. Anugraha, D., Irawan, P.A., Singh, A., Lee, E.S.A., Winata, G.I.: M4-rag: A massive-scale multilingual multi-cultural multimodal rag. arXiv preprint arXiv:2512.05959 (2025)
- 7. Axel, D., Noel, G., Pim, M., Lotte, V.B.: Measuring globalization opening the black box. a critical analysis of globalization indices. Journal of Globalization Studies 1(1), 166–185 (2010)
- 8. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D.,

- Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., Zhu, K.: Qwen3-vl technical report (2025), https://arxiv.org/abs/2511.21631
- 9. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report (2025), https://arxiv.org/abs/2502.13923
- 10. Cahyawijaya, S.: Llm for everyone: Representing the underrepresented in large language models (2024), https://arxiv.org/abs/2409.13897
- 11. Cahyawijaya, S., Chen, D., Bang, Y., Khalatbari, L., Wilie, B., Ji, Z., Ishii, E., Fung, P.: High-dimension human value representation in large language models. In: Chiruzzo, L., Ritter, A., Wang, L. (eds.) Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). pp. 5303–5330. Association for Computational Linguistics, Albuquerque, New Mexico (Apr 2025). https://doi.org/10.18653/v1/2025.naacl-long.274, https: //aclanthology.org/2025.naacl-long.274/
- 12. Cahyawijaya, S., Lovenia, H., Aji, A.F., Winata, G., Wilie, B., Koto, F., Mahendra, R., Wibisono, C., Romadhony, A., Vincentio, K., Santoso, J., Moeljadi, D., Wirawan, C., Hudi, F., Wicaksono, M.S., Parmonangan, I., Alfina, I., Putra,

- I.F., Rahmadani, S., Oenang, Y., Septiandri, A., Jaya, J., Dhole, K., Suryani, A., Putri, R.A., Su, D., Stevens, K., Nityasya, M.N., Adilazuarda, M., Hadiwijaya, R., Diandaru, R., Yu, T., Ghifari, V., Dai, W., Xu, Y., Damapuspita, D., Wibowo, H., Tho, C., Karo Karo, I., Fatyanosa, T., Ji, Z., Neubig, G., Baldwin, T., Ruder, S., Fung, P., Sujaini, H., Sakti, S., Purwarianti, A.: NusaCrowd: Open source initiative for Indonesian NLP resources. In: Rogers, A., Boyd-Graber,
- J., Okazaki, N. (eds.) Findings of the Association for Computational Linguistics: ACL 2023. pp. 13745–13818. Association for Computational Linguistics, Toronto, Canada (Jul 2023). https://doi.org/10.18653/v1/2023.findings-acl.868, https://aclanthology.org/2023.findings-acl.868/

- 13. Cahyawijaya, S., Lovenia, H., Koto, F., Adhista, D., Dave, E., Oktavianti, S., Akbar, S., Lee, J., Shadieq, N., Cenggoro, T.W., Linuwih, H., Wilie, B., Muridan, G., Winata, G., Moeljadi, D., Aji, A.F., Purwarianti, A., Fung, P.: NusaWrites: Constructing high-quality corpora for underrepresented and extremely low-resource languages. In: Park, J.C., Arase, Y., Hu, B., Lu, W., Wijaya, D., Purwarianti, A., Krisnadhi, A.A. (eds.) Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 921–945. Association for Computational Linguistics, Nusa Dua, Bali (Nov 2023). https://doi.org/10.18653/v1/2023.ijcnlp-main.60, https://aclanthology.org/2023.ijcnlp-main.60/
- 14. Cahyawijaya, S., Lovenia, H., Koto, F., Putri, R.A., Dave, E., Lee, J., Shadieq, N., Cenggoro, W., Akbar, S.M., Mahendra, M.I., Putri, D.A., Wilie, B., Winata, G.I., Aji, A.F., Purwarianti, A., Fung, P.: Cendol: Open instruction-tuned generative large language models for Indonesian languages. In: Ku, L.W., Martins, A., Srikumar, V. (eds.) Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 14899–

14914. Association for Computational Linguistics, Bangkok, Thailand (Aug 2024).

https://doi.org/10.18653/v1/2024.acl-long.796, https://aclanthology. org/2024.acl-long.796/

- 15. Cahyawijaya, S., Lovenia, H., Moniz, J.R.A., Wong, T.H., Farhansyah, M.R., Maung, T.T., Hudi, F., Anugraha, D., Habibi, M.R.S., Qorib, M.R., Agarwal, A., Imperial, J.M., Patel, H.L., Feliren, V., Nasution, B.I., Rufino, M.A., Winata, G.I., Rajagede, R.A., Catalan, C.R., Imam, M.F.M., Pattnayak, P., Pranida, S.Z., Pratama, K., Bangera, Y., Na-Thalang, A., Monderin, P.N., Song, Y., Simon, C., Ng, L.H.X., Sapan, R.L., Rafi, T.H., Wang, B., Supryadi, Veerakanjana, K., Ittichaiwong, P., Roque, M.T., Vincentio, K., Kreangphet, T., Artkaew, P., Palgunadi, K.H., Yu, Y., Hastuti, R.P., Nixon, W., Bangera, M., Lim, A.X.W., Khine, A.H., Zhafran, H.M., Ferdinan, T., Izzani, A.A., Singh, A., Evan, E., Krito, J.A., Anugraha, M., Ilasariya, F.A., Li, H., Daniswara, J.A., Tjiaranata, F.A., Yulianrifat, E.P., Udomcharoenchaikit, C., Ansori, F.R., Ihsani, M.K., Nguyen, G., Barik, A.M., Velasco, D.J., Genadi, R.A., Saha, S., Wei, C., Flores, I.E.W., Han, K.C.K., Santos, A.G.D., Lim, W.S., Phyo, K.S., Santos, T., Dwiastuti, M., Luo, J., Cruz, J.C.B., Hee, M.S., Hanif, I.A., Hakim, M.A., Sya’ban, M.R., Kerdthaisong, K., Miranda, L.J.V., Koto, F., Fatyanosa, T.N., Aji, A.F., Rosal, J.J., Kevin, J., Wijaya, R., Kampman, O.P., Zhang, R., Karlsson, B.F., Limkonchotiwat, P.: Crowdsource, crawl, or generate? creating SEA-VL, a multicultural vision-language dataset for Southeast Asia. In: Che, W., Nabende, J., Shutova, E., Pilehvar, M.T. (eds.) Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 18685–18717. Association for Computational Linguistics, Vienna, Austria (Jul 2025). https://doi.org/10.18653/v1/2025.acllong.916, https://aclanthology.org/2025.acl-long.916/
- 16. Cahyawijaya, S., Lovenia, H., Moniz, J.R.A., Wong, T.H., Farhansyah, M.R., Maung, T.T., Hudi, F., Anugraha, D., Habibi, M.R.S., Qorib, M.R., et al.: Sea-vl: A multicultural vision-language dataset for southeast asia. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) pp. 18685–18717 (2025)
- 17. Cecilia Liu, C., Koto, F., Baldwin, T., Gurevych, I.: Are multilingual LLMs culturally-diverse reasoners? an investigation into multicultural proverbs and sayings. In: Duh, K., Gomez, H., Bethard, S. (eds.) Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). pp. 2016–

2039. Association for Computational Linguistics, Mexico City, Mexico (Jun 2024). https://doi.org/10.18653/v1/2024.naacl-long.112, https://aclanthology. org/2024.naacl-long.112/

- 18. Cohere, T., :, Aakanksha, Ahmadian, A., Ahmed, M., Alammar, J., Alizadeh, M., Alnumay, Y., Althammer, S., Arkhangorodsky, A., Aryabumi, V., Aumiller,

- D., Avalos, R., Aviv, Z., Bae, S., Baji, S., Barbet, A., Bartolo, M., Bebensee, B., Beladia, N., Beller-Morales, W., Bérard, A., Berneshawi, A., Bialas, A., Blunsom, P., Bobkin, M., Bongale, A., Braun, S., Brunet, M., Cahyawijaya, S., Cairuz, D., Campos, J.A., Cao, C., Cao, K., Castagné, R., Cendrero, J., Currie, L.C., Chandak, Y., Chang, D., Chatziveroglou, G., Chen, H., Cheng, C., Chevalier, A., Chiu, J.T., Cho, E., Choi, E., Choi, E., Chung, T., Cirik, V., Cismaru, A., Clavier, P., Conklin, H., Crawhall-Stein, L., Crouse, D., Cruz-Salinas, A.F., Cyrus, B., D’souza, D., Dalla-Torre, H., Dang, J., Darling, W., Domingues, O.D., Dash, S., Debugne, A., Dehaze, T., Desai, S., Devassy, J., Dholakia, R., Duffy, K., Edalati, A., Eldeib, A., Elkady, A., Elsharkawy, S., Ergün, I., Ermis, B., Fadaee, M., Fan, B., Fayoux, L., Flet-Berliac, Y., Frosst, N., Gallé, M., Galuba, W., Garg, U., Geist, M., Azar, M.G.,

- Gilsenan-McMahon, E., Goldfarb-Tarrant, S., Goldsack, T., Gomez, A., Gonzaga, V.M., Govindarajan, N., Govindassamy, M., Grinsztajn, N., Gritsch, N., Gu, P., Guo, S., Haefeli, K., Hajjar, R., Hawes, T., He, J., Hofstätter, S., Hong, S., Hooker, S., Hosking, T., Howe, S., Hu, E., Huang, R., Jain, H., Jain, R., Jakobi, N., Jenkins, M., Jordan, J., Joshi, D., Jung, J., Kalyanpur, T., Kamalakara, S.R., Kedrzycki, J., Keskin, G., Kim, E., Kim, J., Ko, W.Y., Kocmi, T., Kozakov, M., Kryściński, W., Jain, A.K., Teru, K.K., Land, S., Lasby, M., Lasche, O., Lee, J., Lewis, P., Li, J., Li, J., Lin, H., Locatelli, A., Luong, K., Ma, R., Mach, L., Machado, M., Magbitang, J., Lopez, B.M., Mann, A., Marchisio, K., Markham, O., Matton, A., McKinney, A., McLoughlin, D., Mokry, J., Morisot, A., Moulder, A., Moynehan, H., Mozes, M., Muppalla, V., Murakhovska, L., Nagarajan, H., Nandula, A., Nasir, H., Nehra, S., Netto-Rosen, J., Ohashi, D., Owers-Bardsley, J., Ozuzu, J., Padilla, D., Park, G., Passaglia, S., Pekmez, J., Penstone, L., Piktus, A., Ploeg, C., Poulton, A., Qi, Y., Raghvendra, S., Ramos, M., Ranjan, E., Richemond, P., Robert-Michon, C., Rodriguez, A., Roy, S., Ruder, S., Ruis, L., Rust, L., Sachan, A., Salamanca, A., Saravanakumar, K.K., Satyakam, I., Sebag, A.S., Sen, P., Sepehri, S., Seshadri, P., Shen, Y., Sherborne, T., Shi, S.S., Shivaprasad, S., Shmyhlo, V., Shrinivason, A., Shteinbuk, I., Shukayev, A., Simard, M., Snyder, E., Spataru, A., Spooner, V., Starostina, T., Strub, F., Su, Y., Sun, J., Talupuru, D., Tarassov, E., Tommasone,
- E., Tracey, J., Trend, B., Tumer, E., Üstün, A., Venkitesh, B., Venuto, D., Verga, P., Voisin, M., Wang, A., Wang, D., Wang, S., Wen, E., White, N., Willman, J., Winkels, M., Xia, C., Xie, J., Xu, M., Yang, B., Yi-Chern, T., Zhang, I., Zhao, Z., Zhao, Z.: Command a: An enterprise-ready large language model (2025), https: //arxiv.org/abs/2504.00698

- 19. Dash, S., Nan, Y., Dang, J., Ahmadian, A., Singh, S., Smith, M., Venkitesh, B., Shmyhlo, V., Aryabumi, V., Beller-Morales, W., Pekmez, J., Ozuzu, J., Richemond, P.H., Locatelli, A., Frosst, N., Blunsom, P., Gomez, A., Zhang,

I., Fadaee, M., Govindassamy, M., Roy, S., Gallé, M., Ermis, B., Üstün, A., Hooker, S.: Aya vision: Advancing the frontier of multilingual multimodality (2026), https://openreview.net/forum?id=DinXMuw6ED

- 20. Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J.S., Salehi, M., Muennighoff, N., Lo, K., Soldaini, L., Lu, J., Anderson, T., Bransom, E., Ehsani, K., Ngo, H., Chen, Y., Patel, A., Yatskar, M., Callison-Burch, C., Head, A., Hendrix, R., Bastani, F., VanderBilt, E., Lambert, N., Chou, Y., Chheda, A., Sparks, J., Skjonsberg, S., Schmitz, M., Sarnat, A., Bischoff, B., Walsh, P., Newell, C., Wolters, P., Gupta, T., Zeng, K.H., Borchardt, J., Groeneveld, D., Nam, C., Lebrecht, S., Wittlif, C., Schoenick, C., Michel, O., Krishna, R., Weihs, L., Smith, N.A., Hajishirzi, H., Girshick, R., Farhadi, A., Kembhavi, A.: Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models (2024), https://arxiv.org/abs/2409.17146
- 21. Figge, L., Martens, P.: Globalisation continues: The maastricht globalisation index revisited and updated. Globalizations 11(6), 875–893 (2014). https://doi.org/ 10.1080/14747731.2014.887389, https://doi.org/10.1080/14747731.2014. 887389
- 22. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., Korenev, A., Hinsvark, A., Rao, A., Zhang, A., Rodriguez, A., Gregerson, A., Spataru, A., Roziere, B., Biron,

- B., Tang, B., Chern, B., Caucheteux, C., Nayak, C., Bi, C., Marra, C., McConnell,
- C., Keller, C., Touret, C., Wu, C., Wong, C., Ferrer, C.C., Nikolaidis, C., Allon-

sius, D., Song, D., Pintz, D., Livshits, D., Wyatt, D., Esiobu, D., Choudhary, D., Mahajan, D., Garcia-Olano, D., Perino, D., Hupkes, D., Lakomkin, E., AlBadawy,

- E., Lobanova, E., Dinan, E., Smith, E.M., Radenovic, F., Guzmán, F., Zhang, F., Synnaeve, G., Lee, G., Anderson, G.L., Thattai, G., Nail, G., Mialon, G., Pang,

- G., Cucurell, G., Nguyen, H., Korevaar, H., Xu, H., Touvron, H., Zarov, I., Ibarra,

- I.A., Kloumann, I., Misra, I., Evtimov, I., Zhang, J., Copet, J., Lee, J., Geffert, J., Vranes, J., Park, J., Mahadeokar, J., Shah, J., van der Linde, J., Billock, J., Hong,
- J., Lee, J., Fu, J., Chi, J., Huang, J., Liu, J., Wang, J., Yu, J., Bitton, J., Spisak, J., Park, J., Rocca, J., Johnstun, J., Saxe, J., Jia, J., Alwala, K.V., Prasad, K., Upasani, K., Plawiak, K., Li, K., Heafield, K., Stone, K., El-Arini, K., Iyer, K., Malik, K., Chiu, K., Bhalla, K., Lakhotia, K., Rantala-Yeary, L., van der Maaten, L., Chen, L., Tan, L., Jenkins, L., Martin, L., Madaan, L., Malo, L., Blecher, L., Landzaat, L., de Oliveira, L., Muzzi, M., Pasupuleti, M., Singh, M., Paluri, M., Kardas, M., Tsimpoukelli, M., Oldham, M., Rita, M., Pavlova, M., Kambadur,

- M., Lewis, M., Si, M., Singh, M.K., Hassan, M., Goyal, N., Torabi, N., Bashlykov,
- N., Bogoychev, N., Chatterji, N., Zhang, N., Duchenne, O., Çelebi, O., Alrassy, P., Zhang, P., Li, P., Vasic, P., Weng, P., Bhargava, P., Dubal, P., Krishnan, P., Koura, P.S., Xu, P., He, Q., Dong, Q., Srinivasan, R., Ganapathy, R., Calderer, R., Cabral,

- R.S., Stojnic, R., Raileanu, R., Maheswari, R., Girdhar, R., Patel, R., Sauvestre,

- R., Polidoro, R., Sumbaly, R., Taylor, R., Silva, R., Hou, R., Wang, R., Hosseini,
- S., Chennabasappa, S., Singh, S., Bell, S., Kim, S.S., Edunov, S., Nie, S., Narang,

- S., Raparthy, S., Shen, S., Wan, S., Bhosale, S., Zhang, S., Vandenhende, S., Batra,

- S., Whitman, S., Sootla, S., Collot, S., Gururangan, S., Borodinsky, S., Herman,
- T., Fowler, T., Sheasha, T., Georgiou, T., Scialom, T., Speckbacher, T., Mihaylov,

- T., Xiao, T., Karn, U., Goswami, V., Gupta, V., Ramanathan, V., Kerkez, V., Gonguet, V., Do, V., Vogeti, V., Albiero, V., Petrovic, V., Chu, W., Xiong, W., Fu, W., Meers, W., Martinet, X., Wang, X., Wang, X., Tan, X.E., Xia, X., Xie,

- X., Jia, X., Wang, X., Goldschlag, Y., Gaur, Y., Babaei, Y., Wen, Y., Song, Y., Zhang, Y., Li, Y., Mao, Y., Coudert, Z.D., Yan, Z., Chen, Z., Papakipos, Z., Singh,

- A., Srivastava, A., Jain, A., Kelsey, A., Shajnfeld, A., Gangidi, A., Victoria, A., Goldstand, A., Menon, A., Sharma, A., Boesenberg, A., Baevski, A., Feinstein, A., Kallet, A., Sangani, A., Teo, A., Yunus, A., Lupu, A., Alvarado, A., Caples, A., Gu, A., Ho, A., Poulton, A., Ryan, A., Ramchandani, A., Dong, A., Franco, A., Goyal, A., Saraf, A., Chowdhury, A., Gabriel, A., Bharambe, A., Eisenman, A., Yazdan, A., James, B., Maurer, B., Leonhardi, B., Huang, B., Loyd, B., Paola,
- B.D., Paranjape, B., Liu, B., Wu, B., Ni, B., Hancock, B., Wasti, B., Spence, B., Stojkovic, B., Gamido, B., Montalvo, B., Parker, C., Burton, C., Mejia, C., Liu,
- C., Wang, C., Kim, C., Zhou, C., Hu, C., Chu, C.H., Cai, C., Tindal, C., Feichtenhofer, C., Gao, C., Civin, D., Beaty, D., Kreymer, D., Li, D., Adkins, D., Xu,
- D., Testuggine, D., David, D., Parikh, D., Liskovich, D., Foss, D., Wang, D., Le,

D., Holland, D., Dowling, E., Jamil, E., Montgomery, E., Presani, E., Hahn, E., Wood, E., Le, E.T., Brinkman, E., Arcaute, E., Dunbar, E., Smothers, E., Sun, F., Kreuk, F., Tian, F., Kokkinos, F., Ozgenel, F., Caggioni, F., Kanayet, F., Seide, F., Florez, G.M., Schwarz, G., Badeer, G., Swee, G., Halpern, G., Herman, G., Sizov, G., Guangyi, Zhang, Lakshminarayanan, G., Inan, H., Shojanazeri, H., Zou,

- H., Wang, H., Zha, H., Habeeb, H., Rudolph, H., Suk, H., Aspegren, H., Goldman,

- H., Zhan, H., Damlaj, I., Molybog, I., Tufanov, I., Leontiadis, I., Veliche, I.E., Gat,
- I., Weissman, J., Geboski, J., Kohli, J., Lam, J., Asher, J., Gaya, J.B., Marcus, J., Tang, J., Chan, J., Zhen, J., Reizenstein, J., Teboul, J., Zhong, J., Jin, J., Yang,
- J., Cummings, J., Carvill, J., Shepard, J., McPhie, J., Torres, J., Ginsburg, J.,

Wang, J., Wu, K., U, K.H., Saxena, K., Khandelwal, K., Zand, K., Matosich, K., Veeraraghavan, K., Michelena, K., Li, K., Jagadeesh, K., Huang, K., Chawla, K., Huang, K., Chen, L., Garg, L., A, L., Silva, L., Bell, L., Zhang, L., Guo, L., Yu, L., Moshkovich, L., Wehrstedt, L., Khabsa, M., Avalani, M., Bhatt, M., Mankus, M., Hasson, M., Lennie, M., Reso, M., Groshev, M., Naumov, M., Lathi, M., Keneally, M., Liu, M., Seltzer, M.L., Valko, M., Restrepo, M., Patel, M., Vyatskov, M., Samvelyan, M., Clark, M., Macey, M., Wang, M., Hermoso, M.J., Metanat,

- M., Rastegari, M., Bansal, M., Santhanam, N., Parks, N., White, N., Bawa, N., Singhal, N., Egebo, N., Usunier, N., Mehta, N., Laptev, N.P., Dong, N., Cheng,
- N., Chernoguz, O., Hart, O., Salpekar, O., Kalinli, O., Kent, P., Parekh, P., Saab, P., Balaji, P., Rittner, P., Bontrager, P., Roux, P., Dollar, P., Zvyagina, P., Ratanchandani, P., Yuvraj, P., Liang, Q., Alao, R., Rodriguez, R., Ayub, R., Murthy, R., Nayani, R., Mitra, R., Parthasarathy, R., Li, R., Hogan, R., Battey, R., Wang,

- R., Howes, R., Rinott, R., Mehta, S., Siby, S., Bondu, S.J., Datta, S., Chugh, S., Hunt, S., Dhillon, S., Sidorov, S., Pan, S., Mahajan, S., Verma, S., Yamamoto, S., Ramaswamy, S., Lindsay, S., Lindsay, S., Feng, S., Lin, S., Zha, S.C., Patil, S., Shankar, S., Zhang, S., Zhang, S., Wang, S., Agarwal, S., Sajuyigbe, S., Chintala,
- S., Max, S., Chen, S., Kehoe, S., Satterfield, S., Govindaprasad, S., Gupta, S., Deng, S., Cho, S., Virk, S., Subramanian, S., Choudhury, S., Goldman, S., Remez,
- T., Glaser, T., Best, T., Koehler, T., Robinson, T., Li, T., Zhang, T., Matthews,

- T., Chou, T., Shaked, T., Vontimitta, V., Ajayi, V., Montanez, V., Mohan, V., Kumar, V.S., Mangla, V., Ionescu, V., Poenaru, V., Mihailescu, V.T., Ivanov, V., Li, W., Wang, W., Jiang, W., Bouaziz, W., Constable, W., Tang, X., Wu, X., Wang, X., Wu, X., Gao, X., Kleinman, Y., Chen, Y., Hu, Y., Jia, Y., Qi, Y., Li,

- Y., Zhang, Y., Zhang, Y., Adi, Y., Nam, Y., Yu, Wang, Zhao, Y., Hao, Y., Qian, Y., Li, Y., He, Y., Rait, Z., DeVito, Z., Rosnbrick, Z., Wen, Z., Yang, Z., Zhao, Z., Ma, Z.: The llama 3 herd of models (2024), https://arxiv.org/abs/2407.21783

- 23. Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z.F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Xu, H., Ding, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Chen, J., Yuan, J., Tu, J., Qiu, J., Li, J., Cai, J.L., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., You, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Zhou, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen, Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R.J., Jin, R.L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye, S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S.S., Zhou, S., Wu, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Liu, W., Liang, W., Gao,

- W., Yu, W., Zhang, W., Xiao, W.L., An, W., Liu, X., Wang, X., Chen, X., Nie,
- X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang, X., Li, X., Su, X., Lin, X., Li, X.Q., Jin, X., Shen, X., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X., Wang, X., Shan, X., Li, Y.K., Wang, Y.Q., Wei, Y.X., Zhang, Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang,
- Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Xiong, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Zhu, Y.X., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z.Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang, Z., Zhang, Z.: Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature

645(8081), 633–638 (Sep 2025). https://doi.org/10.1038/s41586-025-09422-z, http://dx.doi.org/10.1038/s41586-025-09422-z

- 24. Guo, J., Zheng, T., Bai, Y., Li, B., Wang, Y., Zhu, K., Li, Y., Neubig, G., Chen, W., Yue, X.: Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale (2024), https://arxiv.org/abs/2412.05237
- 25. Gygli, S., Haelg, F., Potrafke, N., Sturm, J.E.: The kof globalisation index – revisited. The Review of International Organizations 14(3), 543–574 (Jan 2019). https://doi.org/10.1007/s11558-019-09344-2, http://dx.doi.org/10.1007/ s11558-019-09344-2
- 26. Haelg, F.: The kof globalisation index – a multidimensional approach to globalisation. Jahrbücher für Nationalökonomie und Statistik 240(5), 691–696 (Sep 2019). https://doi.org/10.1515/jbnst-2019-0045, http://dx.doi.org/10. 1515/jbnst-2019-0045
- 27. Hennara, K., Hreden, M., Hamed, M.M., Bastati, A., Aldallal, Z., Chrouf, S., AlModhayan, S.: Baseer: A vision-language model for arabic document-tomarkdown ocr (2025), https://arxiv.org/abs/2509.18174
- 28. Hu, X., Wang, R., Fang, Y., Fu, B., Cheng, P., Yu, G.: Ella: Equip diffusion models with llm for enhanced semantic alignment (2024), https://arxiv.org/abs/2403. 05135
- 29. Jain, A., et al.: Ai4bharat indicbert: A monolingual bert model for indian languages pp. 10685–10706 (2024)
- 30. Ju, J., Kim, D., Park, S., Kim, Y.: Varco-vision: Expanding frontiers in korean vision-language models (2024), https://arxiv.org/abs/2411.19103
- 31. Kabra, A., Liu, E., Khanuja, S., Aji, A.F., Winata, G., Cahyawijaya, S., Aremu, A., Ogayo, P., Neubig, G.: Multi-lingual and multi-cultural figurative language understanding. In: Rogers, A., Boyd-Graber, J., Okazaki, N. (eds.) Findings of the Association for Computational Linguistics: ACL 2023. pp. 8269–8284. Association for Computational Linguistics, Toronto, Canada (Jul 2023). https: //doi.org/10.18653/v1/2023.findings-acl.525, https://aclanthology.org/ 2023.findings-acl.525/
- 32. Khan, F., et al.: Ai4bharat indicbert: A monolingual bert model for indian languages pp. 10685–10706 (2024)
- 33. Kocmi, T., Arkhangorodsky, A., Berard, A., Blunsom, P., Cahyawijaya, S., Dehaze, T., Fadaee, M., Frosst, N., Galle, M., Gomez, A., Govindarajan, N., Ko, W.Y., Kreutzer, J., Marchisio, K., Üstün, A., Vincent, S., Zhang, I.: Command-atranslate: Raising the bar of machine translation with difficulty filtering. In: Haddow, B., Kocmi, T., Koehn, P., Monz, C. (eds.) Proceedings of the Tenth Conference on Machine Translation. pp. 789–799. Association for Computational Linguistics, Suzhou, China (Nov 2025). https://doi.org/10.18653/v1/2025.wmt-1.55, https://aclanthology.org/2025.wmt-1.55/
- 34. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., Stoica, I.: Efficient memory management for large language model serving with pagedattention. In: Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles (2023)
- 35. Labs, C.: Aya vision benchmark (2025), https://huggingface.co/datasets/ CohereLabs/AyaVisionBench
- 36. Liu, C.C., Gurevych, I., Korhonen, A.: Culturally aware and adapted NLP: A taxonomy and a survey of the state of the art. Transactions of the Association for Computational Linguistics 13, 652–689 (2025). https://doi.org/10.1162/tacl_ a_00760, https://aclanthology.org/2025.tacl-1.31/

- 37. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv (2017). https://doi.org/10.48550/arxiv.1711.05101
- 38. Lovenia, H., Mahendra, R., Akbar, S.M., Miranda, L.J.V., Santoso, J., Aco, E., Fadhilah, A., Mansurov, J., Imperial, J.M., Kampman, O.P., Moniz, J.R.A., Habibi, M.R.S., Hudi, F., Montalan, R., Ignatius, R., Lopo, J.A., Nixon, W., Karlsson, B.F., Jaya, J., Diandaru, R., Gao, Y., Amadeus, P., Wang, B., Cruz, J.C.B., Whitehouse, C., Parmonangan, I.H., Khelli, M., Zhang, W., Susanto, L., Ryanda, R.A., Hermawan, S.L., Velasco, D.J., Kautsar, M.D.A., Hendria, W.F., Moslem, Y., Flynn, N., Adilazuarda, M.F., Li, H., Lee, J., Damanhuri, R., Sun, S., Qorib, M.R., Djanibekov, A., Leong, W.Q., Do, Q.V., Muennighoff, N., Pansuwan, T., Putra,

I.F., Xu, Y., Chia, T.N., Purwarianti, A., Ruder, S., Tjhi, W., Limkonchotiwat, P., Aji, A.F., Keh, S., Winata, G.I., Zhang, R., Koto, F., Yong, Z.X., Cahyawijaya, S.: SEACrowd: A multilingual multimodal data hub and benchmark suite for Southeast Asian languages. In: Al-Onaizan, Y., Bansal, M., Chen, Y.N. (eds.) Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. pp. 5155–5203. Association for Computational Linguistics, Miami, Florida, USA (Nov 2024). https://doi.org/10.18653/v1/2024.emnlp-main.296, https://aclanthology.org/2024.emnlp-main.296/

- 39. Martens, P., Raza, M.: The maastricht globalisation index: An update, pp. 279–

309. Nova Science Publishers, United States (2009)

- 40. Mogrovejo, D.O.R., Lyu, C., Wibowo, H.A., Góngora, S., Mandal, A., Purkayastha, S., Ortiz-Barajas, J.G., Cueva, E.V., Baek, J., Jeong, S., Hamed, I., Yong, Z.X., Lim, Z.W., Silva, P.M., Dunstan, J., Jouitteau, M., MEUR, D.L., Nwatu, J., Batnasan, G., Otgonbold, M.E., Gochoo, M., Ivetta, G., Benotti, L., Alemany, L.A., Maina, H., Geng, J., Torrent, T.T., Belcavello, F., Viridiano, M., Cruz, J.C.B., Velasco, D.J., Ignat, O., Burzo, Z., Whitehouse, C., Abzaliev, A., Clifford, T., Caulfield, G., Lynn, T., Salamea-Palacios, C., Araujo, V., Kementchedjhieva, Y., Mihaylov, M.M., Azime, I.A., Ademtew, H.B., Balcha, B.F., Etori, N.A., Adelani, D.I., Mihalcea, R., Tonja, A.L., Cabrera, M.C.B., Vallejo, G., Lovenia, H., Zhang, R., Estecha-Garitagoitia, M., Rodríguez-Cantelar, M., Ehsan, T., Chevi, R., Adilazuarda, M.F., Diandaru, R., Cahyawijaya, S., Koto, F., Kuribayashi, T., Song, H., Khandavally, A.N.K., Jayakumar, T., Dabre, R., Imam, M.F.M., Nagasinghe, K.R.Y., Dragonetti, A., D’Haro, L.F., NIYOMUGISHA, O., Gala, J., Chitale, P.A., Farooqui, F., Solorio, T., Aji, A.F.: CVQA: Culturally-diverse multilingual visual question answering benchmark. In: The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2024), https://openreview.net/forum?id=E18kRXTGmV
- 41. Mohamed, A., Alwajih, F., Nagoudi, E.M.B., Inciarte, A., Abdul-Mageed, M.: Violet: A vision-language model for Arabic image captioning with gemini decoder. In: Sawaf, H., El-Beltagy, S., Zaghouani, W., Magdy, W., Abdelali, A., Tomeh, N., Abu Farha, I., Habash, N., Khalifa, S., Keleg, A., Haddad, H., Zitouni, I., Mrini, K., Almatham, R. (eds.) Proceedings of ArabicNLP 2023. pp. 1–11. Association for Computational Linguistics, Singapore (Hybrid) (Dec 2023). https://doi.org/10. 18653/v1/2023.arabicnlp-1.1, https://aclanthology.org/2023.arabicnlp1.1/
- 42. Naous, T., Ryan, M.J., Ritter, A., Xu, W.: Having beer after prayer? measuring cultural bias in large language models. In: Ku, L.W., Martins, A., Srikumar, V. (eds.) Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 16366–16393. Association for Computa-

- tional Linguistics, Bangkok, Thailand (Aug 2024). https://doi.org/10.18653/ v1/2024.acl-long.862, https://aclanthology.org/2024.acl-long.862/
- 43. Ng, R., Nguyen, T.N., Yuli, H., Chia, T.N., Yi, L.W., Leong, W.Q., Yong, X., Ngui, J.G., Susanto, Y., Cheng, N., Rengarajan, H., Limkonchotiwat, P., Hulagadri, A.V., Teng, K.W., Tong, Y.Y., Siow, B., Teo, W.Y., Meng, T.C., Ong, B., Ong, Z.H., Montalan, J.R., Chan, A., Antonyrex, S., Lee, R., Choa, E., Tat-Wee, D.O., Liu, B.J.D., Tjhi, W.C., Cambria, E., Teo, L.: SEA-LION: Southeast Asian languages in one network. In: Inui, K., Sakti, S., Wang, H., Wong, D.F., Bhattacharyya, P., Banerjee, B., Ekbal, A., Chakraborty, T., Singh, D.P. (eds.) Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics. pp. 512–526. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics, Mumbai, India (Dec 2025), https://aclanthology.org/2025.ijcnlp-long.30/
- 44. Nguyen, T.S., Qorib, M.R., Ng, H.T.: Openseal: Good, fast, and cheap construction of an open-source southeast asian llm via parallel data (2026), https://arxiv. org/abs/2602.02266
- 45. Nguyen, X.P., Zhang, W., Li, X., Aljunied, M., Hu, Z., Shen, C., Chia, Y.K., Li, X., Wang, J., Tan, Q., Cheng, L., Chen, G., Deng, Y., Yang, S., Liu, C., Zhang, H., Bing, L.: SeaLLMs - large language models for Southeast Asia. In: Cao, Y., Feng, Y., Xiong, D. (eds.) Proceedings of ACL. pp. 294–304 (Aug 2024). https://doi.org/10.18653/v1/2024.acl-demos.28, https://aclanthology. org/2024.acl-demos.28/
- 46. Nyandwi, J.D.D., Song, Y., Khanuja, S., Neubig, G.: Grounding multilingual multimodal LLMs with cultural knowledge. In: Christodoulopoulos, C., Chakraborty, T., Rose, C., Peng, V. (eds.) Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 24187–24231. Association for Computational Linguistics, Suzhou, China (Nov 2025). https://doi.org/10.18653/v1/2025. emnlp-main.1232, https://aclanthology.org/2025.emnlp-main.1232/
- 47. Nyborg, J., Pelletier, C., Lefèvre, S., Assent, I.: Timematch: Unsupervised crossregion adaptation by temporal shift estimation. ISPRS Journal of Photogrammetry and Remote Sensing 188, 301–313 (2022). https://doi.org/https: //doi.org/10.1016/j.isprsjprs.2022.04.018, https://www.sciencedirect. com/science/article/pii/S0924271622001216
- 48. Patel, H.L., Agarwal, A., Das, A., Kumar, B., Panda, S., Pattnayak, P., Rafi, T.H., Kumar, T., Chae, D.K.: Sweeval: Do llms really swear? a safety benchmark for testing limits for enterprise use. In: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: Industry Track). pp. 558–582 (2025)
- 49. Singapore, A.: Sea-lion (southeast asian languages in one network): A family of large language models for southeast asia. https://github.com/aisingapore/ sealion (2024)
- 50. Steiner, A.e.a.: Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555 (2024)
- 51. Team, G., Anil, R., Borgeaud, S., Alayrac, J.B., Yu, J., Soricut, R., Schalkwyk,

- J., Dai, A.M., Hauth, A., Millican, K., Silver, D., Johnson, M., Antonoglou, I., Schrittwieser, J., Glaese, A., Chen, J., Pitler, E., Lillicrap, T., Lazaridou, A., Firat,

- O., Molloy, J., Isard, M., Barham, P.R., Hennigan, T., Lee, B., Viola, F., Reynolds, M., Xu, Y., Doherty, R., Collins, E., Meyer, C., Rutherford, E., Moreira, E., Ayoub,

- K., Goel, M., Krawczyk, J., Du, C., Chi, E., Cheng, H.T., Ni, E., Shah, P., Kane,

- P., et al.: Gemini: A family of highly capable multimodal models (2023), https: //arxiv.org/abs/2312.11805

- 52. Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ramé, A., Rivière, M., Rouillard, L., Mesnard, T., Cideron, G., bastien Grill, J., Ramos, S., Yvinec, E., Casbon, M., Pot, E., Penchev, I., Liu, G., Visin, F., Kenealy, K., Beyer, L., Zhai, X., Tsitsulin, A., Busa-Fekete, R., Feng, A., Sachdeva, N., Coleman, B., Gao, Y., Mustafa, B., Barr, I., Parisotto, E., Tian, D., Eyal, M., Cherry, C., Peter, J.T., Sinopalnikov, D., Bhupatiraju, S., Agarwal, R., Kazemi, M., Malkin, D., Kumar, R., Vilar, D., Brusilovsky, I., Luo, J., Steiner, A., Friesen, A., Sharma, A., Sharma, A., Gilady, A.M., Goedeckemeyer, A., Saade, A., Kolesnikov, A., Bendebury, A., Abdagic, A., Vadi, A., György, A., Pinto, A.S., Das,

- A., Bapna, A., Miech, A., Yang, A., Paterson, A., Shenoy, A., Chakrabarti, A., Piot,
- B., Wu, B., Shahriari, B., Petrini, B., Chen, C., Lan, C.L., Choquette-Choo, C.A., Carey, C., Brick, C., Deutsch, D., Eisenbud, D., Cattle, D., Cheng, D., Paparas, D., Sreepathihalli, D.S., Reid, D., Tran, D., Zelle, D., Noland, E., Huizenga, E., Kharitonov, E., Liu, F., Amirkhanyan, G., Cameron, G., Hashemi, H., KlimczakPlucińska, H., Singh, H., Mehta, H., Lehri, H.T., Hazimeh, H., Ballantyne, I., Szpektor, I., Nardini, I., Pouget-Abadie, J., Chan, J., Stanton, J., Wieting, J., Lai,

- J., Orbay, J., Fernandez, J., Newlan, J., yeong Ji, J., Singh, J., Black, K., Yu,
- K., Hui, K., Vodrahalli, K., Greff, K., Qiu, L., Valentine, M., Coelho, M., Ritter, M., Hoffman, M., Watson, M., Chaturvedi, M., Moynihan, M., Ma, M., Babar, N., Byrd, N., Roy, N., Momchev, N., Chauhan, N., Bunyan, O., Botarda, P., Caron, P., Rubenstein, P.K., Culliton, P., Schmid, P., Sessa, P.G., Xu, P., Stanczyk, P., Tafti, P., Shivanna, R., Wu, R., Pan, R., Rokni, R., Willoughby, R., Vallu, R., Mullins, R., Jerome, S., Smoot, S., Girgin, S., Iqbal, S., Reddy, S., Sheth, S., Põder, S., Bhatnagar, S., Eiger, S., Zhang, S., Liu, T., Yacovone, T., Liechty, T., Kalra, U., Evci, U., Misra, V., Roseberry, V., Feinberg, V., Kolesnikov, V., Han, W., Kwon, W., Chen, X., Chow, Y., Zhu, Y., Wei, Z., Egyed, Z., Cotruta, V., Giang, M., Kirk, P., Rao, A., Black, K., Babar, N., Lo, J., Moreira, E., Martins, L.G., Sanseviero, O., Gonzalez, L., Gleicher, Z., Warkentin, T., Mirrokni, V., Senter, E., Collins, E., Barral, J., Ghahramani, Z.: Gemma 3 technical report. arXiv:2503.19786 (2025), https://arxiv.org/abs/2503.19786

- 53. Team, N., Costa-jussà, M.R., Cross, J., Çelebi, O., Elbayad, M., Heafield, K., Heffernan, K., Kalbassi, E., Lam, J., Licht, D., Maillard, J., Sun, A., Wang, S., Wenzek, G., Youngblood, A., Akula, B., Barrault, L., Gonzalez, G.M., Hansanti, P., Hoffman, J., Jarrett, S., Sadagopan, K.R., Rowe, D., Spruit, S., Tran, C., Andrews, P., Ayan, N.F., Bhosale, S., Edunov, S., Fan, A., Gao, C., Goswami, V., Guzmán, F., Koehn, P., Mourachko, A., Ropers, C., Saleem, S., Schwenk, H., Wang, J.: No language left behind: Scaling human-centered machine translation. arXiv:2207.04672

(2022), https://arxiv.org/abs/2207.04672

- 54. Urailertprasert, N., Limkonchotiwat, P., Suwajanakorn, S., Nutanong, S.: SEAVQA: Southeast Asian cultural context dataset for visual question answering. In: Gu, J., Fu, T.J.R., Hudson, D., Celikyilmaz, A., Wang, W. (eds.) Proceedings of the 3rd Workshop on Advances in Language and Vision Research (ALVR). pp. 173–

185. Association for Computational Linguistics, Bangkok, Thailand (Aug 2024). https://doi.org/10.18653/v1/2024.alvr-1.15, https://aclanthology.org/ 2024.alvr-1.15/

- 55. Verma, S., Khanuja, M.S.U.R., Kumar, V., Murthy, R., Sen, J.: Milu: A multi-task indic language understanding benchmark. arXiv preprint arXiv:2411.02538 (2025)

- 56. Wang, H., Yao, Y., Liu, J., Zhang, X., Zhao, Y., Li, S., Liu, Z., Zhang, X., Zeng, Y.: Unsupervised cross-regional and cross-year adaptation by climate indicator discrepancy for crop classification. Journal of Remote Sensing 5 (Jan 2025). https://doi.org/10.34133/remotesensing.0439, http://dx.doi.org/ 10.34133/remotesensing.0439
- 57. Wang, J., Adelani, D.I., et al.: Afrimte and africomet: Enhancing comet to embrace under-resourced african languages pp. 5997–6023 (2024)
- 58. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., Fan, Y., Dang, K., Du, M., Ren, X., Men, R., Liu, D., Zhou, C., Zhou, J., Lin, J.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)
- 59. Wang, Y., Zang, Y., Li, H., Jin, C., Wang, J.: Unified reward model for multimodal understanding and generation (2026), https://arxiv.org/abs/2503.05236
- 60. Winata, G.I., Aji, A.F., Cahyawijaya, S., Mahendra, R., Koto, F., Romadhony, A., Kurniawan, K., Moeljadi, D., Prasojo, R.E., Fung, P., Baldwin, T., Lau, J.H., Sennrich, R., Ruder, S.: NusaX: Multilingual parallel sentiment dataset for 10 Indonesian local languages. In: Vlachos, A., Augenstein, I. (eds.) Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics. pp. 815–834. Association for Computational Linguistics, Dubrovnik, Croatia (May 2023). https://doi.org/10.18653/v1/2023.eacl-main.57, https: //aclanthology.org/2023.eacl-main.57/
- 61. Winata, G.I., Hudi, F., Irawan, P.A., Anugraha, D., Putri, R.A., Yutong, W., Nohejl, A., Prathama, U.A., Ousidhoum, N., Amriani, A., et al.: Worldcuisines: A massive-scale benchmark for multilingual and multicultural visual question answering on global cuisines. In: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). pp. 3242–3264 (2025)
- 62. Winata, G.I., et al.: Worldcuisines: A benchmark dataset for multilingual and multicultural image classification. arXiv preprint arXiv:2405.14133 (2024)
- 63. Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., Li, H.: Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis (2023), https://arxiv.org/abs/2306.09341
- 64. Xu, J., Huang, Y., Cheng, J., Yang, Y., Xu, J., Wang, Y., Duan, W., Yang, S., Jin, Q., Li, S., Teng, J., Yang, Z., Zheng, W., Liu, X., Zhang, D., Ding, M., Zhang, X., Gu, X., Huang, S., Huang, M., Tang, J., Dong, Y.: Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation (2026), https://arxiv.org/abs/2412.21059
- 65. Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., Dong, Y.: Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems 36, 15903–15935 (2023)
- 66. Yue, X., Song, Y., Asai, A., Kim, S., de Dieu Nyandwi, J., Khanuja, S., Kantharuban, A., Sutawika, L., Ramamoorthy, S., Neubig, G.: Pangea: A fully open multilingual multimodal llm for 39 languages. arXiv preprint arXiv:2410.16153

(2024), https://arxiv.org/abs/2410.16153

- 67. Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu, D., Lin, J., Huang, F., Zhou, J.: Qwen3 embedding: Advancing text embedding and reranking through foundation models (2025), https://arxiv.org/abs/2506. 05176

## Appendix

### A Assessment for SEA languages Translation Quality

Table 6: Human evaluation of English→5 SEA languages translation quality using a five-point scale (mean ± standard deviation)

##### Language Model Grammar Naturalness

NLLB 4.6 ± 0.8 4.1 ± 1.1 Gemma-3-27b 4.1 ± 1.2 3.6 ± 1.2 Gemini-2.5-flash 4.8 ± 0.6 4.6 ± 0.8 Gemini-2.5-pro 4.9 ± 0.4 4.9 ± 0.5

Burmese (mya)

NLLB 3.3 ± 1.1 3.0 ± 1.2 Gemma-3-27b 4.2 ± 0.9 4.1 ± 0.9 Gemini-2.5-flash 4.8 ± 0.5 4.7 ± 0.5

Thai (tha)

- Gemini-2.5-pro 4.7 ± 0.5 4.6 ± 0.5

Filipino (fil)

NLLB 4.4 ± 0.6 4.3 ± 0.9 Gemma-3-27b 4.8 ± 0.6 4.6 ± 0.8 Gemini-2.5-flash 4.7 ± 0.4 4.4 ± 0.7

- Gemini-2.5-pro 4.8 ± 0.4 4.5 ± 0.6

NLLB 4.3 ± 0.9 3.8 ± 1.1 Gemma-3-27b 4.6 ± 0.6 4.2 ± 0.9 Gemini-2.5-flash 4.5 ± 0.6 4.1 ± 0.9 Gemini-2.5-pro 4.6 ± 0.5 4.3 ± 0.8

Indonesian (ind)

NLLB 4.7 ± 0.5 4.4 ± 0.8 Gemma-3-27b 5 ± 0.1 4.9 ± 0.3 Gemini-2.5-flash 5 ± 0.1 4.9 ± 0.2 Gemini-2.5-pro 5 ± 0.1 5 ± 0.1

Vietnamese (vie)

We conducted a systematic evaluation on 100 randomly selected humangenerated English captions from the SEA-VL [15] dataset for Burmese (mya), Thai (tha), Filipino (fil), Indonesian (ind), and Vietnamese (vie). The evaluation employed a five-point scale (1–5) assessing grammatical accuracy and naturalness, using NLLB [53], Gemma-3-27b [52], Gemini-2.5-flash, and Gemini-2.5pro [51] models. Table 6 presents the results of the translation quality evaluation in five Southeast Asian languages using a five-point scale (mean ± standard deviation). Overall, the Gemini-2.5 models consistently achieve the highest scores for Burmese (mya) and Thai (tha), with Gemini-2.5-Pro obtaining the best performance for Burmese and Gemini-2.5-Flash performing slightly better for Thai. For Filipino (fil), Indonesian (ind), and Vietnamese (vie), Gemma3-27B achieves competitive or top performance, often matching or exceeding

the Gemini models in grammatical accuracy while maintaining high naturalness scores. Based on these findings, Gemma-3-27b was used for Indonesian (ind), Vietnamese (vie), Standard Malay (zsm), Filipino (fil), and Chinese (zho), where it demonstrated strong and stable performance. Gemini-2.5-flash was used for Thai (tha), Burmese (mya), Lao (lao), Khmer (khm), and Tamil (tam), as the Gemini family showed superior robustness and higher translation quality for these languages.

### B Reward Model Ablation

We evaluated four reward models: HPSv2 [65], HPSv2 [63], VisionReward-Image [64], and UnifiedReward [59]. In order to pick a reward model to use for data curation as described in section 5.1, we measured how much the reward models agreed with human evaluations from SEA-VL [16]. For each image In, we normalized the scores via min-max normalization and then took the average across categories to get a human annotator score snhuman ∈ [0,1]. Notably, the absolute score range of ImageReward-style models varies across datasets, making direct normalization and comparison with human ground-truth scores unreliable. Therefore, instead of comparing absolute scores, we reformulated the evaluation as a pairwise preference task. Specifically, We randomly sampled 500 pairs of images Ia and Ib such that sahuman > sbhuman. For each reward model, we then compute the rate at which it gives Ia a higher score than Ib i.e. sarm > sbrm. Results are summarized in table 7. Although VisionReward-Image achieved the highest pairwise agreement with human judgments, we ultimately selected UnifiedReward by considering both computational efficiency and predictive performance. While its accuracy is slightly lower, UnifiedReward offers a more favorable trade-off overall, as it supports efficient batched inference through vLLM [34], substantially improving throughput for large-scale data filtering.

Table 7: Rate at which sarm > sbrm for each reward model under consideration.

Reward Model Rate where sarm > sbrm

ImageReward 0.394 HPSv2 0.384 VisionReward-Image 0.466 UnifiedReward 0.442

### C Annotated Human Evaluation Test Sets

We conduct human evaluation on two datasets: SEA-VL VQA and SEA AYA.

#### C.1 SEA-VL VQA

SEA-VL VQA contains ∼1.1k visual questions spanning 9 official languages in Southeast Asia: Malay (zsm), Vietnamese (vie), Thai (tha), Indonesian (ind), Filipino (fil), Tamil (tam), Khmer (khm), Chinese (cmn), and Burmese (mya). The culturally grounded images were originally collected via crowdsourcing by [16]. For each image, a native speaker authored a visual question in the target language, which was subsequently reviewed by two additional native speakers, with revisions made when needed to ensure quality.

To promote consistency and cultural relevance, we provided annotators with detailed instructions for writing high-quality visual questions. Specifically, questions were required to: (1) be directly grounded in the visual content of the image; (2) go beyond low-level perception and instead target culturally salient elements, such as traditional attire, local cuisine, religious practices, architectural styles, or social activities; (3) be clear, grammatically correct, and contextually appropriate; and (4) be answerable solely from the image.

We also specified several categories of questions to avoid. These included: (1) overly generic questions (e.g., "What is this?" or "What color is the shirt?"); (2) purely factual questions lacking cultural grounding (e.g., "How many people are in the image?"); (3) questions that cannot be answered from the image alone (e.g., "What is her name?" unless explicitly shown); (4) questions that fail to reflect the Southeast Asian context without clear justification; and (5) questions that can be answered without the visual context.

#### C.2 SEA AYA

We construct SEA AYA by manually translating 135 visual questions from the Aya Vision Benchmark [35] into 6 official Southeast Asian languages: Thai (tha), Malay (zsm), Filipino (fil), Tamil (tam), Chinese (cmn), and Burmese (mya). Combined with the English (eng) and Indonesian (ind) subsets already available in Aya Vision Benchmark, the resulting benchmark contains ∼1.2k instances covering 9 vision-language tasks: image captioning, chart and figure understanding, image difference detection, general VQA, OCR, document understanding, text transcription, visual reasoning, and screenshot-to-code generation. For each example, annotators were first given a rough machine-translated draft as a starting point. They were then asked to revise it to ensure that the final translation is fluent, culturally appropriate, and semantically faithful to the English source.

For quality and consistency, our annotation guidelines emphasized several principles: (1) preserve the meaning, tone, and intent of the original English prompt; (2) produce text that is natural and fluent for native speakers of the target language; (3) use grammar, vocabulary, and syntax that conform to standard usage in the target language; (4) match the level of formality of the source; (5) respect language-specific cultural and pragmatic norms, including idiomatic usage and politeness conventions; and (6) appropriately translate, retain, or normalize named entities and technical terms when necessary.

We also requested annotators to avoid common translation errors, including semantic drift, omission of essential information, overly literal adherence to machine-translated phrasing, literal translation of idiomatic expressions, overtranslation or under-translation, awkward or unidiomatic wording, and inconsistent terminology, particularly across multi-sentence examples.

### D Comparison of SEA-Gemma-3 and Other VLMs

We provide the full comparison per language on CVQA and per region on SEAVQA in Table 8 and Table 9, respectively. The merged global-regional model SEA-Gemma-3 10% achieves strongest performance on SEA regions and other regions on CVQA, producing a solid performance even hgher compared to other smaller and similar sized models like Llama-3.2 Vision [22], Pixtral [4], Gemma-

- 3 [52], and even outperform larger models such as Aya-Vision (32B) [19], Qwen-3 VL (35B) [8], Qwen-2.5-VL (72B) [9], and Molmo (72B) [20]. Similarly on SEAVQA, SEA-Gemma-3 10% also achieves the best average regional score, although on several regions it is still outperformed by large models especially Aya-Vision (32B) [19] which showcase strong performance on multiple SEA regions but falls short on Indonesia, Thailand, and Laos.

### E Results Breakdown

We provide the results breakdown in Table 10, Table 12, Table 11, and Table 15 for SEA-VLM; Table 13 for SEA-ImageGen, and Table 14 fo SEA-VL Embedding.

### F Sample Outputs of SEA-VLM and SEA-ImageGen

We provide samples generated by our best SEA-VLM and SEA-ImageGen compared to the baselines on Figure 5 and Figure 6, respectively. As shown in Figure 6, the SEA-SDXL 25% model produces the most accurate representation among others, while retaining the image naturalness and overall quality of the original SDXL. For example, while the baselines also capture the component of nasi lemask such as nasi putih (white rice) and telur (egg), the SEA-SDXL 25% model incorporates more detailed factors including timun (cucumber), kacang (peanut), ikan bilis (anchovy). Similarly for cultural and tradition categories, SEA-SDXL 25% captures a more accurate illustration of playing angklung and Hung king festival in Phu Tho, Vietnam.

### G List of Globalization Index by Region by Year

We provide the detailed per year per region globalization index which we used for deciding the globalization factor α in GG-EZ in Table 16.

- Table 8: Comparison of our best model (SEA-Gemma-3 10%) againsts other competitor models and the regionally-adapted Gemma-3 (SEA-Gemma-3) model on CVQA.

Llama-3.2 Pixtral Gemma-3 SEA-Gemma-3 SEA-Gemma-3 Aya-Vision Qwen-3 VL Qwen-2.5-VL Molmo Language Vision (11B) (12B) (27B) (27B) 10% (27B) (32B) (35B) (72B) (72B)

Indonesian 56.31 62.86 66.50 21.60 66.26 67.88 66.26 66.50 63.83 Filipino 51.72 64.53 66.01 24.14 70.94 66.34 64.04 65.02 64.53 Malay 56.19 61.90 67.62 29.84 71.43 72.38 70.16 62.50 69.84 Chinese 62.80 68.93 72.66 40.73 75.14 77.48 78.28 80.98 79.29 Tamil 58.41 51.87 73.36 40.65 74.30 61.68 61.68 58.88 58.41 Javanese 47.81 51.18 62.96 22.22 63.64 55.56 54.55 55.22 54.88 Minangkabau 51.79 51.39 64.54 23.90 67.73 61.75 54.98 51.79 58.17 Sundanese 44.00 49.00 59.00 12.50 63.50 52.53 55.50 56.50 52.00

SEA Avg. 53.63 57.71 66.58 26.95 69.12 64.45 63.18 62.17 62.62 Amharic 39.32 32.91 62.82 26.07 62.39 29.18 42.74 36.48 45.30 Bengali 55.59 48.25 72.03 33.57 75.87 64.31 64.34 61.27 68.88 Breton 34.57 35.80 42.96 27.65 45.68 39.36 34.81 37.78 35.06 Bulgarian 49.06 22.91 62.26 34.23 64.42 56.49 54.45 61.99 57.68 Arabic 49.26 43.35 62.07 32.02 65.02 68.47 57.64 61.08 58.62 Hindi 68.16 30.85 80.60 42.79 81.09 78.11 75.62 75.12 78.11 Igbo 44.00 41.50 43.50 25.50 47.00 38.00 34.50 36.55 41.50 Irish 53.99 57.67 64.11 31.29 65.95 56.13 55.52 57.98 57.06 Japanese 50.74 49.26 53.20 29.06 53.69 59.11 58.62 58.62 57.14 Kinyarwanda 35.32 34.47 53.62 28.51 53.19 40.43 32.77 38.30 40.43 Korean 59.66 73.45 76.90 42.76 77.93 80.00 77.24 77.59 74.14 Marathi 48.02 31.19 74.75 34.16 77.23 66.17 63.86 61.39 68.81 Mongolian 39.42 39.74 50.32 23.40 50.64 36.01 44.23 39.10 47.76 Norwegian 54.52 64.21 66.89 38.80 69.57 66.22 61.20 68.56 69.90 Oromo 34.11 35.51 45.79 24.77 47.20 36.45 38.32 35.05 42.06 Portuguese 57.75 73.59 80.63 41.20 80.63 78.01 76.06 76.76 77.46 Romanian 58.94 67.88 75.17 40.73 76.49 74.17 70.20 75.83 70.20 Russian 66.50 37.00 80.50 45.50 81.50 80.00 76.50 79.00 84.00 Sinhala 48.00 28.44 65.33 29.78 65.78 39.56 43.56 45.50 45.78 Spanish 57.02 69.13 69.34 37.61 69.92 73.25 68.20 71.44 71.55 Swahili 53.11 60.07 76.19 36.26 80.95 66.18 60.07 55.31 67.77 Telugu 55.50 32.50 71.50 32.00 74.50 57.79 61.00 58.50 57.00 Urdu 55.98 48.02 73.39 36.93 75.69 66.69 66.99 64.92 69.50 Other Avg. 50.81 45.99 65.39 33.68 67.06 58.70 57.32 58.01 60.25 Global Avg. 51.53 49.01 65.69 31.94 67.59 60.18 58.83 59.08 60.86

- Table 9: Comparison of our best model (SEA-Gemma-3 10%) againsts other competitor models and the regionally-adapted Gemma-3 (SEA-Gemma-3) model on SEA-VQA.

Gemma-3 SEA-Gemma-3 SEA-Gemma-3 Aya-Vision Qwen-3 VL Region (27B) (27B) 10% (27B) (32B) (35B)

Cambodia 50.7 40.5 52.3 56.25 50.99 Indonesia 43.5 49.3 69.0 66.09 66.09 Laos 0.0 50.0 68.1 45.83 63.89 Malaysia 37.5 44.4 62.4 69.31 60.32 Philippines 63.4 46.6 58.8 63.40 56.21 Singapore 56.3 46.9 68.8 71.88 65.63 Thailand 51.6 39.1 52.2 49.46 52.72 Vietnam 18.9 18.5 57.5 63.26 53.04

SEA Avg. 40.2 41.9 61.1 60.68 58.61

- Table 10: SEA-VLM Zero-shot performance on SEA-VQA. The result is broken down per country. “Camb.” = Cambodia, “Indo.” = Indonesia, “Mala.” = Malaysia, ‘Phil.’ = Phillipines, “Sing.”=Singapore, “Thai.” = Thailand, “Viet.” = Vietname

Avg. Country Model Score Camb. Indo. Laos Mala. Phil. Sing. Thai. Viet.

Google Gemma-3 41.0 50.7 43.5 0.0 37.5 63.4 56.3 51.6 18.9 SEA-Gemma-3 5% 61.2 50.3 68.8 66.7 62.4 59.5 65.6 51.1 57.8 SEA-Gemma-3 10% 61.7 52.3 69.0 68.1 62.4 58.8 68.8 52.2 57.5 SEA-Gemma-3 50% 59.5 52.0 67.3 62.5 57.1 56.9 53.1 52.7 55.0 SEA-Gemma-3 70% 54.0 51.3 67.3 63.9 56.6 58.2 53.1 51.6 20.5 SEA-Gemma-3 (w/o merge) 41.9 40.5 49.3 50.0 44.4 46.6 46.9 39.1 18.5

- Table 11: SEA-VLM Zero-shot performance on World Cuisine. (NC) = No Context, (C) = Context, (Adv.) = Adversarial

Overall SEA By Prompt Type Model Score Avg. Task 1 (NC) Task 2 Task 1 (C) Task 1 (Adv.) Google Gemma-3 59.8 60.1 64.8 48.3 78.5 47.5 SEA-Gemma-3 5% 60.0 60.3 64.5 48.1 79.2 48.1 SEA-Gemma-3 10% 60.0 60.2 64.5 48.0 79.1 48.1 SEA-Gemma-3 50% 51.6 51.4 53.7 42.0 67.0 43.6 SEA-Gemma-3 70% 51.9 52.6 53.5 43.5 67.1 43.4 SEA-Gemma-3 (w/o merge) 48.5 48.6 49.3 43.5 61.9 39.4

Table 12: SEA-VLM Zero-shot performance on CVQA.

Overall SEA Prompt Type Score Avg. English Local Lang.

Model

Gemma-3 67.2 67.83 67.96 66.44 SEA-Gemma-3 5% 68.72 69.45 69.39 68.05 SEA-Gemma-3 10% 68.76 69.45 69.42 68.09 SEA-Gemma-3 50% 61.84 62.62 62.9 60.79 SEA-Gemma-3 70% 60.58 61.27 61.51 59.66 SEA-Gemma-3 (w/o merge) 35.58 36.16 38.0 33.16

- Table 13: Detailed per aspect breakdown (Attribute, Entity, Other, and Relation) of the DPGBench evaluation.

Attribute Entity Other Relation Model color other shape size texture part state whole count text non-spatial spatial

StabilityAI SDXL 80.88 79.49 69.87 67.36 79.98 78.52 74.23 82.71 56.00 76.00 81.13 86.73 SEA-SDXL 25% 82.34 81.49 75.11 63.22 80.82 82.81 73.39 83.45 59.50 70.00 77.36 86.93 SEA-SDXL 50% 82.39 80.49 73.80 66.53 82.52 80.08 74.13 82.67 58.50 66.00 76.73 87.80 SEA-SDXL 75% 81.59 79.49 72.49 67.36 81.19 79.49 73.71 83.09 61.50 70.00 81.76 86.23 SEA-SDXL 100% 82.14 81.26 77.73 64.46 81.73 79.69 74.45 83.20 62.00 78.00 84.91 87.06

- Table 14: Detailed per country zero-shot performance of the SEAVQA evaluation.

Model SEAVQA Zero-Shot Performance

Overall Cambodia Indonesia Laos Malaysia Phillipines Singapore Thailand Vietnam

Google SigLIP2 25.81 28.9 25.0 22.2 24.3 28.8 21.9 25.5 25.6 SEA-SigLIP2 25% 26.36 27.6 25.4 29.2 29.1 28.8 28.1 26.1 24.0 SEA-SigLIP2 50% 28.06 31.3 27.0 22.2 29.6 27.5 21.9 31.0 27.2 SEA-SigLIP2 75% 29.66 28.6 30.1 19.4 32.3 34.0 25.0 29.9 28.8 SEA-SigLIP2 100% 28.96 28.3 28.6 29.2 26.5 32.7 31.3 30.4 29.1

- Table 15: SEA-VLM Zero-shot performance on SEA-VQA. The result is broken down per country. “Khm” = Cambodia, “Lao” = Laos, “Ind” = Indonesia, “Mys” = Malaysia, ‘Phl’ = Phillipines, “Sgp”=Singapore, “Th.” = Thailand, “Vie” = Vietnam. “CG” denotes CulturalGround, “WC” denotes WorldCuisine, while “OE” and “MC” denote openended and multiple-choice, respectively.

Avg. Country Model Score Khm Ind Lao Mys Phl Sgp Tha Vie

SEA-Mammoth 50k 13.0 9.9 18.8 11.1 11.1 11.8 18.8 14.67 2.88 SEA-Mammoth 250k 39.6 13.8 52.9 52.8 42.9 47.1 46.9 44.0 20.8 SEA-Mammoth 250k + CG OE 41.9 40.5 49.3 50.0 44.4 46.6 46.9 39.1 18.5 SEA-Mammoth 250k + CG MC 21.6 26.6 20.9 26.4 16.4 24.8 28.1 28.3 14.1 SEA-Mammoth 250k + WC 30.4 10.2 39.9 34.7 37.6 34.0 50.0 32.6 16.9

[Figure 10]

[Figure 11]

[Figure 12]

###### Fig. 5: Generated responses using different model architypes. From left-to-right: global model (Gemma-3), our regional model (SEA-Gemma-3), our merged model (SEAGemma-3 10%) along with the prompts. Our model produces the most correct image among others, while retaining the image naturalness and overall quality of the original Gemma-3.

[Figure 13]

###### Fig. 6: Generated image using different model architypes. From left-to-right: global model (SDXL), our regional model (SEA-SDXL), our merged model (SEA-SDXL 25%), and reference of natural images.

- Table 16: Globalization index by region by yea from 1993 to 2023. “EAP” = East Asia and Pacific , ‘ECA” = Europe and Central Asia,“LAC” = Latin America and Caribbean, “MENA” Middle East and North Africa, “NA” = North America, “SA” = South Asia, “SEA” = Southeast Asian, “SSA” = Sub-Saharan Africa.

###### Year EAP ECA LAC MENA NA SA SEA SSA World

- 1993 49.32 58.82 51.20 57.00 68.59 28.05 36.04 34.37 48.86
- 1994 51.07 59.15 51.00 56.85 68.58 28.25 36.40 34.26 49.15
- 1995 51.40 59.72 51.67 56.62 68.66 28.39 37.34 34.13 49.42
- 1996 52.02 60.23 51.67 56.74 69.43 28.37 37.89 33.99 49.64
- 1997 51.68 60.43 51.75 56.79 69.48 28.35 37.76 33.43 49.53
- 1998 51.63 60.71 51.86 56.89 69.83 28.33 37.70 33.55 49.66
- 1999 52.00 61.00 52.20 56.94 70.64 28.26 38.59 33.45 49.85
- 2000 52.48 61.65 53.33 57.03 71.09 28.94 39.23 33.63 50.38
- 2001 52.61 62.06 53.88 57.43 71.74 28.96 39.12 33.80 50.70
- 2002 52.63 62.83 54.53 57.66 72.50 30.16 40.06 33.92 51.15
- 2003 52.75 63.55 55.32 58.04 73.02 30.84 39.76 34.82 51.79
- 2004 53.41 64.48 55.88 58.42 73.52 31.06 40.63 34.95 52.33
- 2005 53.93 65.47 56.19 58.48 74.17 32.29 41.61 35.24 52.86
- 2006 54.32 66.19 56.86 59.34 74.39 32.87 41.96 35.70 53.46
- 2007 55.16 67.67 58.04 59.78 75.24 33.99 42.90 36.40 54.46
- 2008 55.55 68.76 58.41 60.78 75.57 36.58 43.59 36.72 55.17
- 2009 55.70 68.42 58.59 61.54 75.41 37.98 44.27 37.16 55.38
- 2010 55.94 68.92 58.37 62.05 75.87 39.49 43.19 37.37 55.68
- 2011 56.40 70.04 58.50 62.51 76.43 40.59 44.17 38.87 56.53
- 2012 56.55 70.52 58.75 63.24 76.64 42.27 44.50 38.92 56.88
- 2013 56.50 70.93 59.06 63.55 77.25 42.87 45.33 39.14 57.16
- 2014 56.77 71.07 59.00 63.44 77.80 43.90 46.15 39.54 57.36
- 2015 57.13 70.32 58.76 62.50 77.86 44.75 45.79 38.99 56.98
- 2016 57.09 70.13 58.97 62.44 78.44 44.38 45.58 37.91 56.70
- 2017 57.04 70.88 59.00 60.91 78.81 44.14 45.12 37.71 56.75
- 2018 57.34 71.10 58.85 60.79 78.82 43.25 46.21 37.15 56.65
- 2019 56.36 70.84 58.51 60.92 78.95 40.53 44.29 36.54 56.12
- 2020 54.54 69.16 56.97 59.02 77.72 38.80 41.33 34.97 54.45
- 2021 53.94 69.60 57.23 59.66 78.02 37.97 39.87 34.86 54.54
- 2022 54.96 70.78 58.30 60.00 79.04 39.20 42.11 35.36 55.42
- 2023 55.74 71.03 58.91 60.13 79.36 40.06 43.40 35.46 55.79

