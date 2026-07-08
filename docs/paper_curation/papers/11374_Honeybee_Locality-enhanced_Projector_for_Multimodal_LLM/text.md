## Honeybee: Locality-enhanced Projector for Multimodal LLM

Junbum Cha* Wooyoung Kang* Jonghwan Mun* Byungseok Roh Kakao Brain

{junbum.cha, edwin.kang, jason.mun, peter.roh}@kakaobrain.com

# arXiv:2312.06742v2[cs.CV]1Apr2024

### Abstract

In Multimodal Large Language Models (MLLMs), a visual projector plays a crucial role in bridging pre-trained vision encoders with LLMs, enabling profound visual understanding while harnessing the LLMs’ robust capabilities. Despite the importance of the visual projector, it has been relatively less explored. In this study, we first identify two essential projector properties: (i) flexibility in managing the number of visual tokens, crucial for MLLMs’ overall efficiency, and (ii) preservation of local context from visual features, vital for spatial understanding. Based on these findings, we propose a novel projector design that is both flexible and locality-enhanced, effectively satisfying the two desirable properties. Additionally, we present comprehensive strategies to effectively utilize multiple and multifaceted instruction datasets. Through extensive experiments, we examine the impact of individual design choices. Finally, our proposed MLLM, Honeybee, remarkably outperforms previous state-of-the-art methods across various benchmarks, including MME, MMBench, SEED-Bench, and LLaVA-Bench, achieving significantly higher efficiency. Code and models are available at https://github. com/kakaobrain/honeybee.

### 1. Introduction

Large Language Models (LLMs) have made great progress in recent years, mainly thanks to instruction tuning. Visual instruction tuning [34] has been proposed to extend LLMs into Multimodal LLMs (MLLMs) to perceive and understand visual signals (e.g., images). The main idea for MLLMs is to introduce a projector connecting the vision encoder and LLM, and to learn the projector using visual instruction data while keeping the parameters of the vision encoder and LLM. Such a simple technique allows to preserve and leverage the pre-trained knowledge and abilities in vision encoder and LLM, making resulting MLLMs unlock new capabilities, such as generating stories, poems,

*Equal contribution

74

|Linear<br><br>Resamp|ler| | | | |
|---|---|---|---|---|---|
|Honeyb|ee (Ours)| | | | |
| | | | | | |
| | | | |# visua|l tokens|
| | | | |64<br><br>144|256 400<br><br>|
| | | | | | |

72

70

AvgN

68

66

64

1.5 2.0 2.5 3.0 3.5 4.0 4.5

Step time (s/step)

Figure 1. Performance vs. efficiency for projectors where AvgN means an average of normalized benchmark scores (MME, MMBench, and SEED-Bench) and step time is a single step execution time during pre-training. Honeybee with the locality-enhanced projector (i.e., C-Abstractor) offers a more favorable balance between efficiency and performance over existing projectors.

MMB SEEDI MMEP MME LLaVAW

Previous SoTA 67.7 [33] 68.1 [33] 1531 [33] 1848 [2] 70.7 [33] Honeybee (Ours) 73.6 (+5.9) 68.6 (+0.5) 1661 (+130) 1977 (+129) 77.5 (+6.8)

Table 1. Comparison with SoTA. The proposed Honeybee outperforms the previous state-of-the-art MLLMs on various benchmarks with significant gaps.

advertisements, code, and more from given images; those tasks have traditionally been considered challenging for conventional vision-language foundation models [56, 59]. Such success leads to increasing attention for research into MLLMs taking multimodal inputs (e.g., videos [28], audio [13], 3d world [17], point cloud [52]) beyond text.

For MLLMs, the projector plays a critical role in the following two aspects: 1) performance: as it bridges the vision and language models by translating visual features into visual tokens so that the language model can understand, the quality of conveyed visual tokens directly impacts the overall performance of the MLLM; and 2) efficiency: as most of the computational burden lies with the language model, the efficiency of MLLMs is heavily influenced by the number of resulting visual tokens. However, despite its critical importance, the projector has been relatively underexplored in the literature and most MLLMs simply adopt either linear projectors [7, 34] or abstractors [2, 11, 27, 54, 66].

Notably, recent MLLMs prefer abstractors (e.g., resampler, Q-former) to linear projectors; this is primarily due to their flexibility in handling the number of resulting visual tokens, thus offering versatile design options for achieving a preferable balance between efficiency and performance. However, as shown in Fig. 3, the abstractors face more difficulties in learning spatial understanding tasks compared to the linear projectors. This difficulty stems from the abstraction process lacking a locality-aware design, which causes it to primarily focus on a few regions, leading to a loss of finer details essential for spatial comprehension. In contrast, linear projectors excel at preserving all local contexts of visual features via one-to-one transformation. This strong preservation of locality allows effective spatial understanding.

Motivated by this, we propose novel locality-enhanced projectors, which exhibit a more favorable balance between performance (by locality preservation) and efficiency (by abstraction capability) as presented in Fig. 1. To be specific, we introduce two locality-enhanced projectors by employing two powerful operations in locality modelingconvolution and deformable attention. Such injection of locality-aware design into the abstraction process not only promotes the overall performance improvement of MLLMs in handling intricate visual information but also capitalizes on computational efficiency during the subsequent response generation phase of LLMs.

On top of the MLLM with a locality-enhanced projector, named Honeybee, we offer a hidden recipe for cutting-edge MLLMs. Notably, a prevalent strategy in recent MLLM training involves multiple instruction data: 1) GPT-assisted instruction-following dataset like LLaVA [34] and 2) vision-language task datasets with instructization1 process [11]. To take maximized advantage from these datasets, we present important but less explored design choices for 1) how to utilize multifaceted instruction data and 2) the effective way for an instructization process. We perform extensive experiments to verify the impact of individual design choices on diverse benchmarks and hope to offer valuable insights into training strong MLLMs.

Our main contributions are summarized as follows:

- • We identify two crucial properties of projector, 1) locality preservation of visual features and 2) flexibility to manage the number of visual tokens, and propose localityenhanced abstractors to achieve the best of both worlds.
- • We propose a (hidden) effective way to tackle multifaceted datasets as well as the instructization process, maximizing the benefit from instruction data.
- • With the locality-enhanced projector and explored hidden recipes, our Honeybee achieves state-of-the-art performances across the various MLLM benchmarks—MME, MMBench, SEED-Bench, and LLaVA-Bench (Table 1).

1Instructization denotes conversion of raw data into instructionfollowing format using pre-defined templates.

### 2. Related Work

#### 2.1. Multimodal Large Language Models

The remarkable instruction-following and generalization abilities of recent LLMs have ushered in extending LLMs to Multimodal LLMs (MLLMs). Early works such as Flamingo [1] and BLIP-2 [27] successfully adapted LLMs to visual tasks, showing notable zero-shot generalization and in-context learning capabilities. More recently, MLLMs are further advanced mainly through visual instruction tuning, which includes utilizing vision-language (VL) datasets [2, 11, 61] and enhancing visual instructionfollowing data [32, 34, 40, 63, 65, 66]. Also, several studies focus on grounding capabilities of MLLMs by utilizing additional datasets specifically designed for these tasks [7, 45, 53, 55]. However, recent MLLMs have not yet deeply explored visual projectors, despite the proper design of projectors is critical in both the effectiveness and efficiency of MLLMs.

#### 2.2. Multimodal Instruction-following Data

The breakthrough from GPT-3 [4] to ChatGPT [43] highlights the importance of instruction-following data in empowering LLM to understand and follow natural language instructions. Similarly, integrating visual instruction data is essential for training MLLMs to handle various instructions, thus increasing their versatility. Several studies employ a powerful LLM, e.g., GPT-4 [44], to generate visual instruction data for complex VL tasks, such as generating stories, poems, detailed captions from given images [32, 34, 63, 65, 66]. Another line of studies has explored transforming existing VL task datasets into an instruction-following format using pre-defined templates, called instructization [2, 11, 33, 61]. While there is active development and expansion of instruction-following datasets, the research focusing on how to combine and utilize these datasets remains underexplored.

#### 2.3. Benchmarks for MLLM

MME [14], MMBench [35], and SEED-Bench [25] have been introduced as comprehensive benchmarks for the objective evaluation of MLLMs with yes/no or multiplechoice questions. These benchmarks encompass a broad spectrum of evaluation tasks, ranging from coarse- and finegrained perceptual analysis to visual reasoning tasks. On the other hand, as the capabilities of MLLMs evolve to handle more complex VL tasks such as visual storytelling and instruction-following in an open-set manner with free-form text, other types of benchmarks have been proposed, i.e., subjective evaluation. Following NLP studies [9, 36], several studies leverage powerful LLMs, e.g., GPT-4 [44], to assess the response quality of MLLMs [3, 34, 58]. This approach aims for a more detailed evaluation of the profi-

[Figure 1]

Vision Encoder

Projector Large Language Model

Visual Features Visual Tokens Text Tokens

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

|Resampler|
|---|

|D-Abstractor|
|---|

|C-Abstractor|
|---|

D-Abstractor

C-Abstractor

Resampler

Linear Projector

[Figure 8]

[Figure 9]

[Figure 10]

✗ Flexibility

###### ✓ Flexibility

✓ Flexibility

✗ Locality Preservation

✓ Locality Preservation

✓ Locality Preservation

(a) Linear Projector

(b) Abstractor

(c) Locality-enhanced Abstractor

- Figure 2. Conceptual comparison between projectors in terms of how to convert visual features into visual tokens. (a) Linear projector performs a one-to-one transformation, thus effective in preserving all local contexts of visual features, but limited in flexibility. (b) Abstractor such as resampler offers flexibility by abstracting the visual features into a smaller number of visual tokens but is limited in local context preservation by focusing on salient regions. (c) Our locality-enhanced abstractors can achieve both flexibility and locality preservation.

ciency of MLLMs. In this paper, we aim to provide valuable insights into training a robust and high-performing MLLM through extensive analysis.

same LLM, the efficiency of the MLLM—in terms of computation, memory consumption, and throughput—is mainly affected not by the efficiency of the visual encoder and projector, but by the number of resulting visual tokens fed into the LLM. This is also shown in Fig. 1 and Appendix A.

### 3. Honeybee: Locality-enhanced MLLM

Revisiting existing projectors. The projector takes the N visual features and converts them into M visual tokens. For the projector, MLLMs adopt an operation between a linear projection and an abstraction of visual features. The linear projection is simple yet effective, particularly in preserving knowledge and understanding of vision encoder (e.g., the locality of visual features), but faces challenges in scalability and efficiency, primarily due to its inherent constraint of one-to-one transformation between visual features and tokens (i.e., M = N). On the other hand, the abstraction offers a more adaptable approach to determining the quantity of visual tokens (M). For example, resampler and Q-former utilize M (generally < N for efficiency) learnable queries and cross-attention to extract visual cues from visual features [1, 2, 11, 54, 66]. While such flexibility by abstraction allows better efficiency, but it can inherently suffer from a risk of information loss from the vision encoder.

#### 3.1. Overview

Generally, the goal of Multimodal Large Language Models (MLLMs) is to learn a model that can produce instructionfollowing responses for the given multimodal inputs. In this paper, we consider images as an additional modality input to MLLMs. Thus, the language model becomes a receiver of both visual and text (instruction) tokens while generating text responses in an autoregressive manner. Formally, a multimodal input consists of two types of tokens: image tokens Ximg and text tokens Xtext. Then, the language model predicts the response Y = {wi}Li=1 conditioned on the multimodal input where L means the number of tokens in the response. Therefore, the response is predicted by

L

p(wi|Ximg,Xtext,w<i). (1)

p(Y|Ximg,Xtext) =

i=1

Architecture. MLLMs are generally composed of three networks: 1) vision encoder, 2) projector, and 3) large language model (LLM). The vision encoder provides a sequence of region-level visual features for detailed image understanding. The projector is in charge of transferring the visual features to visual tokens for the subsequent language model. Then, the LLM processes the fused visual and instruction tokens and produces a response autoregressively.

#### 3.2. Locality-enhanced Projector

In this section, we first describe our motivation for localityenhanced projectors. Then, we present two types of localityenhanced projectors (C-Abstractor and D-Abstractor) and describe the training pipeline.

##### 3.2.1 Motivation

The projector is crucial as it bridges visual and language models, translating image features into a format that is comprehensible and utilizable by the language model. Considering its role, when designing a projector, the most impor-

Efficiency of MLLMs. In the MLLM architecture, the LLM predominantly accounts for the entire computation and memory consumption of the MLLM. Thus, with the

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

|ResBlock|
|---|

|ResBlock|
|---|

Adaptive AvgPool

ResBlock

ResBlock

× L

× L

+

- (a) C-Abstractor
- (b) D-Abstractor

- Figure 3. (Left) an example of an attention map from the resampler and (Right) a comparison of spatial understanding capability for the resampler and linear projector where AvgN is computed using six spatial understanding tasks from MME, MMB, and SEEDI.

× L

[Figure 16]

[Figure 17]

|Self Attention|
|---|

|Deformable Attention|
|---|

Deformable Attention

+

Self Attention

Adaptive AvgPool

Visual features

Positional Embedding

tant factor is flexibility in deciding the number of resulting visual tokens. As described above, the number of visual tokens produced by the projector determines the overall efficiency and computational amount of MLLM. Considering the scenario of handling multiple or large images, improving efficiency through flexibility in reducing the number of visual tokens is highly required for scalability. This requirement has led to the preference for abstractors like resamplers and Q-formers over linear projectors in recent MLLMs [2, 11, 27, 54].

Figure 4. Conceptual architecture of our proposed projectors.

other L ResNet blocks. This design allows to abstract visual features to any squared number of visual tokens, and even project to more visual tokens than the original number of visual features. We also tested several variants [37, 49] in Appendix B, but ResNet [51] shows the best performance.

However, we observe the resampler suffers from tackling spatial understanding tasks compared to the linear projector. Note that a linear projector retains all the local context of visual features through a one-to-one projection without loss. In contrast, in Fig. 3, the resampler tends to summarize information primarily from a few regions (e.g., man) while potentially overlooking details in some local regions (e.g., meals, cups, background people). We believe that this difference between two models in the preservation of all local contexts (during abstraction) significantly impacted spatial understanding performance.

D-Abstractor. While convolution is a successful concept in local context modeling, one can argue that it introduces overly strict inductive biases for locality. Hence, we propose Deformable attention-based Abstractor, D-Abstractor, enhancing the locality-awareness of the resampler during abstraction while keeping its flexibility. Specifically, the deformable attention [67] benefits in preserving local context; each learnable query gathers visual features via a 2-D coordinate-based sampling process using reference points and sampling offsets focusing on near the reference points. Here, we propose an advanced initialization method of reference points where the reference points are manually initialized, distributing uniformly over the whole feature map. This additional technique allows D-Abstractor to capture fine-grained and comprehensive information for a given image. More detailed explanations are given in Appendix B.

Stemming from these observations, we propose two novel visual projectors, C-Abstractor and D-Abstractor, under two key design principles: (i) enabling flexibility over the number of visual tokens and (ii) effectively preserving the local context. These new projectors are designed to maintain the strengths of the abstractor, such as computational efficiency via flexibility in managing visual token numbers, while also improving the preservation of local features. This enhancement not only boosts the overall performance of MLLMs in handling complex visual information but also benefits from the computational efficiency during the subsequent response generation phase of LLMs. The conceptual comparison between the existing and proposed projectors is illustrated in Fig. 2.

#### 3.3. Training

We train Honeybee in the two-stage pipeline. In the first stage, we freeze the vision encoder and LLM, focusing on training the proposed locality-enhanced projector. In the second stage, we train both the projector and LLM to enhance deeper visual understanding and generation abilities.

Pre-training for vision-language alignment. The goal of pre-training is to learn a newly introduced visual projector to build connections between the vision encoder and LLM. Using the image-text data (e.g., BlipCapFilt [26], COYO [5]), the pre-training enables MLLM to develop a nuanced understanding of how visual cues align with textual descriptions. During pre-training, the vision encoder and LLM are frozen to keep the fundamental understanding already established in vision and language models.

##### 3.2.2 Architecture

C-Abstractor. In deep learning, convolution has been the most successful architecture for modeling local context [24, 49, 51]. Thus, we design Convolutional Abstractor, C-Abstractor, for effective local context modeling. Fig. 4a depicts the entire architecture, comprising L ResNet blocks [51] followed by adaptive average pooling and an-

Task Datasets #samples

Captioning BlipCapFilt [26], COYO100M [5] 200M VQA (Open) VQAv2 [16], GQA [20], OCRVQA [42], VSR [31] 2.2M VQA (MC) ScienceQA [39], A-OKVQA [48] 0.03M REC RefCOCO [21], RefCOCO+ [57], RefCOCOg [41], VG [23] 5.7M Instruction LLaVA150K [34], ShareGPT [10] 0.2M

Table 2. List of all training datasets.

Visual instruction tuning. After the pre-training of the projector for vision-language alignment, in the second stage, we jointly train the projector and LLM to enhance instruction-following capabilities and achieve a more profound visual understanding. For instruction-following, we utilize two GPT-assisted instruction-following datasets, LLaVA [34] and ShareGPT [10]. In addition, to enhance visual understanding, we instructize a wide range of existing datasets using templates, as listed in Table 2. Specifically, our approach includes: 1) employing a range of tasks such as open-ended VQA [16, 20, 31, 42], multiple-choice VQA [39, 48], captioning [5, 26], and referring expression comprehension (visual grounding and grounded captioning) [21, 23, 41, 57]; 2) using multiple datasets for each task; 3) applying a fine-grained but single template for each dataset. Detailed examples and descriptions are in Appendix E. We thoroughly explore template-based instructization strategies and the utilization of multifaceted datasets in Section 4.

### 4. Hidden Recipe for Visual Instruction Tuning

In Section 3, we examine the limitations of current projectors and propose methods for enhancing locality. However, a clear recipe for training cutting-edge Multimodal LLMs (MLLMs) remains unclear. While it is widely known that instruction tuning using existing datasets with the templatebased instructization is beneficial [2, 11, 33], the details of the instructization process are still underexploredquestions persist regarding dataset selection, utilization, and combination strategies. In this section, we aim to clarify these aspects via following the five research questions: (i) To what extent does each dataset contribute to the performance of specific tasks? (ii) What is an effective balancing strategy between diverse datasets? (iii) What is the appropriate granularity for the templates? (iv) How significant is the diversity of the templates? (v) Do conversation-like multi-turn templates provide additional benefits?

Dataset combination. In recent MLLM studies, a diverse range of datasets has been employed for training powerful MLLMs [2, 6, 11, 33, 61]. This prevalent practice, however, is not accompanied by comprehensive analysis to identify which datasets are critical for specific tasks. To offer an indepth analysis of this, we design a systematic ablation experiment. As outlined in Table 2, we categorize the datasets into several task groups. Then, we examine the variations in benchmark performances by sequentially excluding each task group during instruction tuning. Through these ablation

experiments, we hope to offer valuable insights into the key factors for design choice regarding the dataset combination.

Dataset balancing. While a wide range of datasets are available for training MLLMs, their sizes differ substantially, as shown in Table 2. Also, when training MLLMs, it is common practice to restrict the number of training iterations to preserve the knowledge of a pre-trained LLM. Consequently, properly balancing the training datasets is crucial to maximize learning diverse skills within the short training schedule. To examine this, we compare five different balancing strategies: 1) per-dataset: uniform sampling for each dataset, 2) per-task: uniform sampling for each task, 3) per-sample-100k: uniform sampling for each sample with clipping the maximum size of each dataset to 100k [50], 4) per-dataset-tuned: empirically tuned balancing based on per-dataset strategy.

Template granularity. While the use of pre-defined templates for transforming existing datasets into an instruction format is widely recognized [11, 33, 50, 61], the appropriate granularity for applying these templates is not clearly established. We design the experiments to compare two approaches with different template granularity: 1) finegrained: applying unique templates for each dataset [50], and 2) coarse-grained: applying the shared templates across datasets within the same task category [11, 33].

Template diversity. Prior to the emergence of GPTassisted conversation datasets, securing template diversity was critical, often achieved by employing a range of diverse pre-defined templates alongside input inversion strategies2 [22, 38, 61]. However, the introduction of GPTassisted datasets has seemingly diminished the emphasis on the diversity of templates [33]. The exact role and significance of employing multiple templates and input inversion techniques in the context of GPT-assisted datasets remain less understood. To investigate this, we compare three distinct approaches utilizing: 1) a single template, 2) multiple templates, and 3) multiple templates with input inversion.

Multi-turn template. When utilizing existing datasets, it’s common to find multiple input-target pairs for a single image, as seen in VQA datasets with several QA pairs per image. The multi-turn strategy merges these pairs into a single, conversation-like multi-turn example. However, this approach can merge semantically overlapped input-target pairs into one example, potentially encouraging simplistic shortcuts in finding answers, particularly in the autoregressive training of MLLMs. To mitigate this, we introduce an additional de-duplication strategy, which removes semantically duplicate input-target pairs from the multi-turn examples, thereby preventing shortcut training. We detail this strategy with examples in Appendix E.

2Input inversion is a task augmentation strategy by reversing input and target, e.g., inversion of VQA generating questions from image and answer.

### 5. Experiments

- 5.1. Settings

Benchmarks. We adopt four benchmarks specifically designed for Multimodal LLM (MLLM) evaluation, including MME [14], MMBench [35], SEED-Bench [25] and LLaVA-Bench (In-the-Wild) [34]. The first three assess various capabilities of MLLMs, such as perceptual understanding and visual reasoning, using binary yes/no questions (MME) or multiple-choice questions (MMBench, SEEDBench). Note that we use splits of MME with perception tasks (MMEP), MMBench-dev (MMB), and SEEDBench Image-only (SEEDI), respectively. Our focus on perception tasks in MME are explained in Appendix F. On the other hand, LLaVA-Bench (In-the-Wild), LLaVAW, exploits GPT-4 to assess MLLM’s descriptive responses, providing a comprehensive view of the model’s performance in natural language generation and human preference.

Metrics. We report the official metrics computed using official implementation for individual benchmarks by default; we also report the normalized average AvgN [8, 29] across benchmarks, defined as the average of scores normalized by their respective upper bound scores, facilitating straightforward comparisons.

Implementation details. We use 7B and 13B Vicunav1.5 [10] as LLM. We leverage the pre-trained CLIP ViTL/14 [46] with 224 and 336 resolutions for 7B- and 13BLLM, respectively; we use features from the second-last layer of CLIP instead of the last layer. Any image indicator tokens, e.g., special tokens enclosing visual tokens, are not used. We train the entire LLM instead of parameter-efficient fine-tuning. For in-depth ablations, we use a short training schedule (50k pre-training, 4k instruction tuning) with Vicuna-7B, CLIP ViT-L/14, and C-Abstractor with M=144 visual tokens unless stated otherwise. For the final models, we adopt a long training schedule (200k pre-training, 10k instruction tuning). More details are in Appendix C.

- 5.2. Analysis on Locality-Enhanced Projector To showcase the value of the proposed projector, we assess and compare both performance and efficiency against existing projectors in Table 3 using six spatial understanding tasks from MME, MMBench, and SEED-Bench. First, Resampler (B2, B5) shows poor performance due to its lack of consideration for local context preservation, despite being flexible to the number of visual tokens M. Second, Linear projector is limited to M=256 (B4) due to its inflexibility (B1), leading to intractable computational costs in high-resolution of larger M. Third, for the same computational budget (M=256), our C-Abstractor offer significantly improved performance compared to linear one (52.6 (B4) vs. 56.3 (B6)). Lastly, with fewer visual tokens (M=144), our CAbstractor demonstrate improved performance (+0.9 point)

MME MMB SEED

AvgN

Projector M s/step

POS SR OL PR SR IL

|B1<br><br>B2<br><br>B3<br><br><br>|Linear Resampler C-Abstractor|144 -<br><br>144 2.28 144 2.23<br><br>|Unavailable due to inflexiblity 75.0 22.2 43.2 62.5 47.5 50.6 135.0 24.4 54.3 66.7 49.0 58.8|43.9 53.5<br><br>|
|---|---|---|---|---|
|B4<br>B5<br>B6<br>|Linear<br><br>Resampler<br><br>C-Abstractor|256 3.04 256 3.12 256 3.07<br><br>|140.0 24.4 40.7 70.8 48.9 60.9<br><br>73.3 24.4 37.0 79.2 44.4 51.8 136.7 26.7 55.6 75.0 52.7 59.3|52.6 45.6 56.3<br><br>|

Table 3. Comparison of spatial understanding capability between projectors. The abbreviations for task names mean Position (POS) for MME, Spatial Relationship (SR), Object Localization (OL), and Physical Relation (PR) for MMBench, Spatial Relation (SR) and Instance Location (IL) for SEED-Bench. AvgN indicates the normalized average over six tasks. M means the number of visual tokens and s/step indicates the execution time for a single step during pre-training.

and greater efficiency (3.04 (B4) vs. 2.23 (B3) s/step). This improvement suggests our locality-enhanced projector excels at abstracting visual features where it integrates local contexts from neighboring features and provides contextenriched visual tokens, thus enabling our projectors to outperform linear counterparts even with fewer visual tokens.

#### 5.3. Hidden Recipe for Visual Instruction Tuning

Dataset combination. Table 4 shows a comprehensive ablation study to identify the individual impact of datasets on various multimodal benchmarks. First, we investigate the impact of dataset diversity within each task by leveraging only a single dataset for each task group (D1 vs. D2). The overall performance drop highlights the importance of the dataset diversity within each task. Second, we explore the impact of each task by sequentially excluding specific tasks (D1 vs. D3-8). This reveals that task diversity is crucial for learning how to handle a variety of tasks; each task improves the performance of relevant benchmarks, VQA (Open) → MME, VQA (MC) → MMB and SEEDI, and captioning and instruction-following data → LLaVAW. Third, we inspect the impact of using existing vision-language data (D9 vs. D10). Excluding such data leads to significant decreases in MME, MMB and SEEDI benchmarks. This suggests that rich knowledge in existing vision-language datasets enhances MLLM’s perception understanding or visual reasoning capabilities. In summary, these experiments emphasize the importance of diversity in both tasks and datasets within each task.

Dataset balancing. The necessity of hand-crafted dataset balancing is addressed in previous studies [11, 38]. Based on our observations in Table 4, we tune the balance of each dataset with the two principles: limiting epochs for smaller datasets and allowing up to about a few epochs for key datasets. Table 5a demonstrates the effectiveness of our manually tuned per-dataset-tuned approach. Without handcrafting, the per-dataset can be a reliable alternative. More details are provided in Appendix C.

| |Task type<br><br>| |MLLM benchmark| | |
|---|---|---|---|---|---|
| |Template-based VQA (Open) VQA (MC) REC Cap|GPT-assisted V-Inst T-Inst<br><br>|Multiple choice MMB SEEDI<br><br>|Binary yes/no MMEP MME<br><br>|GPT eval LLaVAW|

D1 ✓ ✓ ✓ ✓ ✓ ✓ 69.2 64.2 1568 1861 64.5 D2 ✓∗ ✓∗ ✓∗ ✓∗ ✓∗ ✓∗ 67.4 (↓1.8) 63.1 1454 (↓114) 1754 (↓107) 62.2 (↓2.3) D3 ✓ ✓ ✓ ✓ ✓ 68.8 62.4 (↓1.8) 1310 (↓258) 1605 (↓256) 67.0 D4 ✓ ✓ ✓ ✓ ✓ 30.4 (↓38.8) 20.8 (↓43.4) 1536 1829 65.4 D5 ✓ ✓ ✓ ✓ ✓ 68.5 63.5 1524 1787 67.0 D6 ✓ ✓ ✓ ✓ ✓ 69.7 63.9 1540 1846 59.8 (↓4.7) D7 ✓ ✓ ✓ ✓ ✓ 70.0 64.0 1507 1805 51.9 (↓12.6) D8 ✓ ✓ ✓ ✓ ✓ 68.7 64.5 1559 1851 62.7 (↓1.8) D9 ✓ ✓ ✓ ✓ 70.0 64.5 1527 1800 26.1 (↓38.4)

D10 ✓ ✓ 43.7 (↓25.5) 0.0 (↓64.2) 1123 (↓445) 1441 (↓420) 67.0

- Table 4. The impact of data mixtures during instruction tuning. Abbreviations for instruction data types stand for VQA (Open): openended visual question answering, VQA (MC): visual question answering with multiple choice, REC: referring expression comprehension, Cap: captioning, V-Inst: visual instruction, T-Inst: text-only instruction-following. The ✓∗ indicates that only one dataset from each task type is used to train a model, including GQA, ScienceQA, RefCOCO, COYO100M, LLaVA150k, and ShareGPT for each task.

|Mixture type|MMB SEEDI MMEP<br><br>|AvgN|
|---|---|---|
|per-dataset per-task per-sample-100k per-dataset-tuned<br><br>|68.7 64.1 1543.2<br><br>65.7 62.1 1488.9 63.6 62.8 1494.8<br><br>69.2 64.2 1568.2<br><br><br>|70.0 67.4 67.1 70.6<br><br>|

(a) Dataset balancing. Hand-crafted balancing is the best, with perdataset strategy serving as an effective starting point for tuning.

|Type Identifier<br><br>|MMB SEEDI MMEP AvgN|LLaVAW|
|---|---|---|
|Inst. instruction Multi. dataset name Multi. task name<br><br>|69.2 64.2 1568.2 70.6 66.8 64.2 1483.1 68.4 68.4 64.1 1507.5 69.3<br><br>|64.5 64.3 64.2|

M M

(b) Instruction tuning vs. Multi-task learning. Instruction tuning (inst.) is more effective compared to multi-task learning (multi.).

|Granularity Diversity<br><br>|MMB SEEDI MMEP AvgN<br><br>|LLaVAW|
|---|---|---|
|fine single coarse single fine multi fine multi+flip<br><br>|69.2 64.2 1568.2 70.6 68.9 64.0 1553.8 70.2 68.1 64.2 1581.2 70.5 67.4 63.3 1575.9 69.8<br><br>|64.5 64.3 61.0 62.7|

c f f

(c) Template granularity and diversity. The fine-grained and single template works the best for instructization.

|MT Dedup|MMB SEEDI MMEP<br><br>|AvgN|
|---|---|---|
|✓<br><br>✓ ✓<br><br>|69.1 63.5 1518.2<br><br>67.8 63.7 1546.1<br><br>69.2 64.2 1568.2<br><br><br>|69.5 69.6 70.6<br><br>|

(d) Multi-turn and de-duplication strategies. Employing both strategies results in the best score.

- Table 5. Ablations on dataset balancing and instructization. AvgN indicates normalized average of MMB, SEEDI, and MMEP. Default settings are marked in gray .

Instruction tuning vs. multi-task learning. Table 5b shows the advantages of instruction tuning with templatebased formatting over multi-task learning using simple identifiers. This result aligns with prior studies [11, 50], showing the efficacy of instruction tuning in our setting.

Template granularity. Table 5c demonstrates that the fine-grained template (first row) consistently outperforms the coarse-grained template (second row) across all benchmarks. We observe that in datasets such as RefCOCO and RefCOCO+, while the input distribution p(Ximg,Xtext) is similar, the answer distribution p(Y|Ximg,Xtext) differs. In this scenario, the coarse-grained template makes the model suffer from differentiating answers for similar inputs. Template diversity. To compare the effect of template diversity on model performance, we evaluate three scenarios with different diversities: using a single template (single), employing 10 templates for each dataset (multi), and inverting 3 out of 10 templates (multi+flip). Interestingly, our experiments reveal that increasing template diversity does not guarantee a performance boost, as shown in Table 5c. This

is consistent results with recent studies [33], showing that effective zero-shot generalization is achievable even without using multiple templates.

Multi-turn template. Table 5d shows the effectiveness of both multi-turn template and de-duplication strategies. The results imply removing the semantically overlapping pairs in each example is effective for mitigating shortcut training. Additional recipes. Apart from datasets and instructization strategies, training recipes also incorporate several subtle yet crucial design choices, including the selection of features in vision encoder, LLMs, LLM training techniques, image indicators, pre-training and instruction tuning iterations. These recipes are detailed in Appendix D.

Final recipe. In summary, our final recipe is summarized as 1) adopting flexible, locality-preserving C-Abstractor or D-Abstractor; 2) leveraging diverse datasets for various tasks (Table 4); 3) applying selected ablation options in Table 5 and Appendix D—the application of per-dataset balancing with hand-crafted tuning, fine-grained templates, and multi-turn interactions with deduplication.

Method LLM Projector Vision Encoder Res. MMB MMEP MME SEEDI LLaVAW Approaches using 7B LLM LLaVA (v1) [34] LLaMA-7B Linear CLIP ViT-L/14 224 38.7 502.8 717.5 33.5 MiniGPT-4 [66] Vicuna-7B Resampler EVA-CLIP ViT-G 224 24.3 581.7 726.0 47.4 LLaMA-AdapterV2 [15] LLaMA-7B LLaMA-Adapter CLIP ViT-L/14 224 41.0 972.7 1221.6 32.7 mPLUG-Owl [54] LLaMA-7B Resampler CLIP ViT-L/14 224 49.4 967.3 1243.4 34.0 InstructBLIP [11] Vicuna-7B Q-former EVA-CLIP ViT-G 224 36.0 - - 58.8 60.9 IDEFICS LLaMA-7B Flamingo OpenCLIP ViT-H/14 224 48.2 - - 44.5 Shikra [7] Vicuna-7B Linear CLIP ViT-L/14 224 58.8 - - Qwen-VL [2] Qwen-7B Resampler OpenCLIP ViT-bigG 448 38.2 - - 62.3 Qwen-VL-Chat [2] Qwen-7B Resampler OpenCLIP ViT-bigG 448 60.6 1487.5 1848.3 65.4 LLaVA-1.5 [33] Vicuna-7B Linear CLIP ViT-L/14 336 64.3 1510.7 - - 63.4 Honeybee (M=144) Vicuna-7B

70.1 1584.2 1891.3 64.5 67.1 D-Abstractor 70.8 1544.1 1835.5 63.8 66.3

C-Abstractor

CLIP ViT-L/14 224

Approaches using 13B LLM MiniGPT-4 [66] Vicuna-13B Resampler EVA-CLIP ViT-G 224 - 866.6 1158.7 - BLIP-2 [27] Vicuna-13B Q-former EVA-CLIP ViT-G 224 - 1293.8 - - 38.1 InstructBLIP [11] Vicuna-13B Q-former EVA-CLIP ViT-G 224 44.0 1212.8 1504.6 - 58.2 LLaVA-1.5 [33] Vicuna-13B Linear CLIP ViT-L/14 336 67.7 1531.3 1826.7 68.1 70.7

73.2 1629.3 1944.0 68.2 75.7 D-Abstractor 73.5 1632.0 1950.0 66.6 72.9

C-Abstractor

Honeybee (M=256) Vicuna-13B

CLIP ViT-L/14 336

- Table 6. Comparison with other state-of-the-art MLLMs. Res. and M indicate the image resolution and the number of visual tokens, respectively. We highlight the best results and second-best results in bold and underline.

#### 5.4. Putting It Altogether

Comparison with existing MLLMs. In Table 6, we compare our Honeybee, trained using the final recipe and a long training schedule, with other state-of-the-art MLLMs. Honeybee outperforms comparable 7B-scale MLLMs in all benchmarks, except for SEEDI. It is worth noting that competing methods like Qwen-VL [2] and LLaVA-1.5 [33] use larger vision encoders (e.g., ViT-bigG for Qwen-VL) or larger images (448 and 336) with more visual tokens (M=256 and 576). In contrast, Honeybee employs ViT-L/14 with 224 resolution and 144 visual tokens striking a balance between performance and efficiency (Figure 8). For tasks requiring detailed visual understanding, such as SEEDI (see Appendix F), using larger images or more visual tokens can be beneficial. When the number of visual tokens is increased from 144 to 256, Honeybee achieves the best score in SEEDI (65.5) among 7B-scale LLMs, as shown in Table 7. When scaled up to 13B, Honeybee surpasses all previous methods in every benchmark. The detailed scores are available in Appendix G.1.

Pushing the limits. In our final 7B and 13B models, we use 144 and 256 visual tokens (M), respectively, balancing efficiency and performance. As indicated in Fig. 1 and Appendix A, increasing M consistently improves performance. Our experiments, aligning M in Honeybee with that of linear projector (Table 7), show performance enhancement at the cost of efficiency. Additional comparisons with previous methods are in Appendix G.2.

LLM Res. M s/step MMB MMEP MME SEEDI LLaVAW

144 2.23 70.1 1584.2 1891.3 64.5 67.1 256 3.07 71.0 1592.7 1951.3 65.5 70.6

7B 224

256 5.52 73.2 1629.3 1944.0 68.2 75.7 576 9.80 73.6 1661.1 1976.5 68.6 77.5

13B 336

Table 7. Pushing the limits with C-Abstractor by increasing the number of visual tokens (M). s/step is pre-training step time.

Additional results. We additionally present (i) the detailed scores for MME, MMB, SEEDI, and LLaVAW in Appendix G.1, (ii) ScienceQA [39] results in Appendix G.3, (iii) additional benchmark (MM-Vet [58], MMMU [60], POPE [30]) results in Appendix G.4, and (iv) qualitative examples in Appendix H.2.

### 6. Conclusion

The advent of visual instruction tuning has brought remarkable advances in MLLMs. Despite these strides, areas such as projector design and the approach in handling multifaceted data with instructization processes remain underexplored or unclear. Inspired by this, we identify the desirable but overlooked projector property, i.e., locality preservation, and propose the locality-enhanced projector that offers a preferable performance-efficiency balance. In addition, we provide extensive experiments to identify the impact of individual design choices in handling multifaceted instruction data, unveiling hidden recipes for high-performing MLLM development. Finally, Honeybee remarkably outperforms previous state-of-the-art methods on various benchmarks.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikoł aj Bi´nkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Kar´en Simonyan. Flamingo: a Visual Language Model for Few-Shot Learning. In NeurIPS, 2022. 2, 3
- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. arXiv preprint arXiv:2308.12966, 2023. 1, 2, 3, 4, 5, 8, 14, 17
- [3] Shuai Bai, Shusheng Yang, Jinze Bai, Peng Wang, Xingxuan Zhang, Junyang Lin, Xinggang Wang, Chang Zhou, and Jingren Zhou. TouchStone: Evaluating Vision-Language Models by Language Models. arXiv preprint arXiv:2308.16890,

2023. 2

- [4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language Models are Few-shot Learners. In NeurIPS, 2020. 2
- [5] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. COYO-700M: Image-Text Pair Dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 4, 5
- [6] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechu Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. MiniGPT-v2: Large Language Model as A Unified Interface for Vision-Language Multi-task Learning. arXiv preprint arXiv:2310.09478, 2023. 5
- [7] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing Multimodal LLM’s Referential Dialogue Magic. arXiv preprint arXiv:2306.15195, 2023. 1, 2, 8
- [8] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Universal Image-Text Representation Learning. In ECCV,

2020. 6

- [9] Cheng-Han Chiang and Hung-yi Lee. Can Large Language Models Be an Alternative to Human Evaluations? arXiv preprint arXiv:2305.01937, 2023. 2
- [10] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing.

- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality, 2023. 5, 6, 18, 19
- [11] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards Generalpurpose Vision-Language Models with Instruction Tuning. arXiv preprint arXiv:2305.06500, 2023. 1, 2, 3, 4, 5, 6, 7, 8
- [12] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023. 13
- [13] Soham Deshmukh, Benjamin Elizalde, Rita Singh, and Huaming Wang. Pengi: An Audio Language Model for Audio Tasks. arXiv preprint arXiv:2305.11834, 2023. 1
- [14] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv preprint arXiv:2306.13394,

2023. 2, 6

- [15] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, Hongsheng Li, and Yu Qiao. LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model. arXiv preprint arXiv:2304.15010, 2023. 8
- [16] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering. In CVPR, 2017. 5
- [17] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3D-LLM: Injecting the 3D World into Large Language Models. arXiv preprint arXiv:2307.12981, 2023. 1
- [18] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR, 2022. 14
- [19] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In CVPR, 2018. 12
- [20] Drew A Hudson and Christopher D Manning. GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. In CVPR, 2019. 5, 14
- [21] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to Objects in Photographs of Natural Scenes. In EMNLP, 2014. 5
- [22] Dohwan Ko, Ji Soo Lee, Wooyoung Kang, Byungseok Roh, and Hyunwoo J. Kim. Large Language Models are Temporal and Causal Reasoners for Video Question Answering. In EMNLP, 2023. 5
- [23] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations. IJCV, 2017. 5
- [24] Yann LeCun, Yoshua Bengio, et al. Convolutional networks for images, speech, and time series. The handbook of brain theory and neural networks, 3361(10):1995, 1995. 4

- [25] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-Bench: Benchmarking Multimodal LLMs with Generative Comprehension. arXiv preprint arXiv:2307.16125, 2023. 2, 6
- [26] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. In

- ICML, 2022. 4, 5

[27] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. In

- ICML, 2023. 1, 2, 4, 8, 14

- [28] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 1
- [29] Linjie Li, Jie Lei, Zhe Gan, Licheng Yu, Yen-Chun Chen, Rohit Pillai, Yu Cheng, Luowei Zhou, Xin Eric Wang, William Yang Wang, et al. VALUE: A Multi-Task Benchmark for Video-and-Language Understanding Evaluation. arXiv preprint arXiv:2106.04632, 2021. 6
- [30] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In EMNLP, 2023. 8, 18
- [31] Fangyu Liu, Guy Edward Toh Emerson, and Nigel Collier. Visual Spatial Reasoning. Transactions of the Association for Computational Linguistics, 2023. 5
- [32] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning Large Multi-Modal Model with Robust Instruction Tuning. arXiv preprint arXiv:2306.14565, 2023. 2
- [33] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 1, 2, 5, 7, 8, 12, 14, 17
- [34] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. In NeurIPS, 2023. 1, 2, 5, 6, 8, 17, 18, 19
- [35] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. MMBench: Is Your Multi-modal Model an All-around Player? arXiv preprint arXiv:2307.06281, 2023. 2, 6
- [36] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. arXiv preprint arXiv:2303.16634, 2023. 2
- [37] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In CVPR, 2022. 4, 12
- [38] Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. In ICML, 2023. 5, 6
- [39] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to Explain: Multimodal Reasoning

- via Thought Chains for Science Question Answering. In NeurIPS, 2022. 5, 8, 17
- [40] Yadong Lu, Chunyuan Li, Haotian Liu, Jianwei Yang, Jianfeng Gao, and Yelong Shen. An Empirical Study of Scaling Instruct-tuned Large Multimodal Models. arXiv preprint arXiv:2309.09958, 2023. 2
- [41] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and Comprehension of Unambiguous Object Descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016. 5
- [42] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. OCR-VQA: Visual Question Answering by Reading Text in Images. In ICDAR, 2019. 5
- [43] OpenAI. ChatGPT, 2023. 2
- [44] OpenAI. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774, 2023. 2, 17
- [45] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding Multimodal Large Language Models to the World. arXiv preprint arXiv:2306.14824, 2023. 2
- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning Transferable Visual Models From Natural Language Supervision. In ICML, 2021. 6, 14
- [47] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020. 13
- [48] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge. In ECCV, 2022. 5
- [49] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 4
- [50] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In ICLR, 2022. 5, 7
- [51] Saining Xie, Ross Girshick, Piotr Doll´ar, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In CVPR, 2017. 4, 12
- [52] Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. PointLLM: Empowering Large Language Models to Understand Point Clouds. arXiv preprint arXiv:2308.16911, 2023. 1
- [53] Shiyu Xuan, Qingpei Guo, Ming Yang, and Shiliang Zhang. Pink: Unveiling the Power of Referential Comprehension for Multi-modal LLMs. arXiv preprint arXiv:2310.00582, 2023. 2
- [54] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality. arXiv preprint arXiv:2304.14178, 2023. 1, 3, 4, 8, 14, 18, 19

- [55] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and Ground Anything Anywhere at Any Granularity. arXiv preprint arXiv:2310.07704, 2023. 2
- [56] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. Transactions on Machine Learning Research, 2022. 1
- [57] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling Context in Referring Expressions. In ECCV, 2016. 5
- [58] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities. arXiv preprint arXiv:2308.02490, 2023. 2, 8, 18
- [59] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A New Foundation Model for Computer Vision. arXiv preprint arXiv:2111.11432, 2021. 1
- [60] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023. 8, 18
- [61] Yan Zeng, Hanbo Zhang, Jiani Zheng, Jiangnan Xia, Guoqiang Wei, Yang Wei, Yuchen Zhang, and Tao Kong. What Matters in Training a GPT4-Style Language Model with Multimodal Inputs? arXiv preprint arXiv:2307.02469, 2023. 2, 5
- [62] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention. arXiv preprint arXiv:2303.16199,

2023. 17

- [63] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. LLaVAR: Enhanced Visual Instruction Tuning for Text-rich Image Understanding. arXiv preprint arXiv:2306.17107, 2023. 2
- [64] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal Chain-ofThought Reasoning in Language Models. arXiv preprint arXiv:2302.00923, 2023. 17
- [65] Bo Zhao, Boya Wu, and Tiejun Huang. SVIT: Scaling up Visual Instruction Tuning. arXiv preprint arXiv:2307.04087,

2023. 2

- [66] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models. arXiv preprint arXiv:2304.10592, 2023. 1, 2, 3, 8
- [67] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable DETR: Deformable Transformers for End-to-End Object Detection. In ICLR, 2021. 4, 12

Projector M s/step MMB SEEDI MMEP AvgN Linear 256 3.04 67.1 65.1 1556.5 70.0

64 1.69 65.9 58.9 1394.7 64.8 144 2.28 66.0 57.0 1389.6 64.2 256 3.12 67.1 59.9 1489.6 67.2 400 4.27 67.7 61.5 1502.5 68.1

Resampler

64 1.65 69.2 62.9 1528.1 69.5 144 2.23 69.2 64.2 1568.2 70.6 256 3.07 70.2 65.3 1586.8 71.6 400 4.15 70.8 65.5 1615.0 72.3

C-Abstractor

Table 8. Detailed scores of projectors by the number of visual tokens (M). s/step indicates the time spent to perform one step in pre-training.

### A. Efficiency of MLLMs

As described in Section 3 of the main text, the efficiency of MLLMs is predominantly affected not by the efficiency of the vision model or projector, but by the number of visual tokens (i.e., the number of output tokens of the projector). Table 8 demonstrates this description, complementing Fig. 1. Notably, while the resampler has substantially larger parameters than linear (105M vs. 4M parameters), MLLM with resampler with M = 144 is more efficient than MLLM with linear (M = 256), as shown by lower step times (2.28 vs. 3.04). Our C-Abstractor, adhering to our design principles of flexibility and locality preservation, stands out as a Pareto-front model compared to both resampler and linear.

### B. Details on Projectors

In this section, we provide further ablations and descriptions for design choices of individual projectors.

#### B.1. Linear Projector

In the recent study, LLaVA (v1.5) [33] utilizes a 2-layer MLP instead of a single linear projection for enhancing the vision-language connector’s representation power. This approach led to an investigation of how varying the number of MLP layers impacts overall performance. As shown in Table 9, the 2-layer MLP-based projector marginally improves the overall performance compared to the linear projector. However, we observe a slight performance drop when further increasing the number of MLP layers (i.e., 6-layer MLP). We note that our C-Abstractor and D-Abstractor achieve better or comparable benchmark scores while using fewer visual tokens, indicating our projectors’ superiority regarding the balance of efficiency and performance.

#### B.2. Resampler

As described in the main text, our design focuses on two principles: 1) flexibility in visual token counts, which is the

Architectures MMB SEEDI MMEP AvgN Linear 67.1 65.1 1556.5 70.0 2-layer MLP 68.3 64.5 1557.2 70.2 6-layer MLP 68.5 63.5 1509.2 69.1 Resampler 66.0 57.0 1389.6 64.2 Resamplerw/pos-emb 65.9 58.0 1384.7 64.4 ResNet (C-Abstractor) 69.2 64.2 1568.2 70.6 ConvNext 66.2 61.9 1525.4 68.1 StandardConv 67.4 57.1 1409.7 65.0 Deformable (D-Abstractor) 68.6 63.2 1548.3 69.7 Deformablew/ov-pooledQ 68.4 63.1 1521.7 69.2 Deformablew/oM-RP 68.5 62.9 1497.0 68.7

Table 9. Ablations for various architectural design choices in each projector. We use 144 visual tokens (M=144) for all architectures except for Linear and MLPs (M=256) due to their inflexibility.

key factor to the efficiency of MLLM, and 2) preservation of local context, which is critical for spatial understanding. Our first try is augmenting visual features with positional embeddings in the resampler framework, but it does not yield notable improvements (See Resamplerw/ pos-emb in Table 9). This leads us to design two novel projectors, CAbstractor and D-Abstractor.

#### B.3. C-Abstractor

Under our design principles on flexibility and locality, we introduce convolution layers and adaptive average pooling into the projector. The overall architecture is illustrated in Fig. 4. We compare three convolution blocks: 1) ResNet bottleneck block [51] with squeeze-excitation [19], 2) ConvNext block [37], and 3) a standard convolution block (3×3 convolution layer). Table 9 shows ResNet block outperforms ConvNext and standard convolution (StandardConv) blocks. Hence, we employ ResNet block for C-Abstractor. While further architectural variations are explorable under the proposed design principles, we leave them for future investigation.

#### B.4. D-Abstractor

We first describe how deformable attention [67] works in D-Abstractor. The core components of deformable attention include (i) 2-D reference points p, (ii) 2-D sampling offsets ∆o, and (iii) attention weights A. For individual learnable queries z, the feature aggregation from the visual feature map Xfeat is formulated by3:

K

zl+1 =

Alk · Xfeat(p + ∆olk), (2)

k=1

where K is the number of sampling offsets per reference point, and l is the index of the attention layer. All the ref-

3We recommend reading [67] for more details.

Ablated setting Default value Changed value MMB SEEDI MMEP MME AvgN LLaVAW

(Default) Honeybee with short training schedule 69.2 64.2 1568.2 1860.7 70.6 64.5

- (i) Image indicator ✗ ✓ 67.4 62.5 1543.4 1809.5 69.0 60.5

- (ii) Visual feature layer Second-last Last 69.2 63.7 1566.1 1839.3 70.4 62.1

- (iii) LLM Vicuna-v1.5 LLaMA-2-chat 70.0 63.6 1551.7 1822.0 70.4 62.8

- (iv) LLM tuning Full

LoRA (r = 64) 35.0 48.9 1016.1 1156.1 44.9 59.2 LoRA (r = 256) 47.3 49.9 959.1 1217.3 48.4 64.0

- (v) Pre-training steps 50k 200k 69.1 63.8 1586.6 1855.2 70.7 66.4

- (vi) Instruction tuning steps 4k

10k 69.3 64.3 1586.8 1868.6 71.0 66.6 16k 70.9 63.8 1550.6 1856.7 70.7 66.0

- Table 10. Additional recipes. The default value indicates the choice used in our default ablation setting with the short training schedule.

Configuration Pre-training Instruction Tuning Trainable modules Abstractor Abstractor, LLM Batch size 256 128 Learning rate 3e-4 2e-5 Minimum LR 1e-5 1e-6 LR schedule Cosine decay Warmup steps 2000 150 Training steps 200k 10k Weight decay 0.01 1e-4 Optimizer AdamW Optimizer HPs β1 = 0.9, β2 = 0.98, ϵ = 1e − 6 Gradient clipping 1.0

- Table 11. Training hyperparameters. HP and LR indicate hyperparameter and learning rate, respectively. Note that we use LR of 1e-4 for D-Abstractor.

Task Dataset Ratio Task Dataset Ratio

VQA (Open) VQAv2 10.3% REC RefCOCO 10.3% GQA 10.3% RefCOCO+ 10.3% OCRVQA 5.1% RefCOCOg 10.3% VSR 2.6% VG 5.1%

VQA (MC) ScienceQA 5.1% Instruction LLaVA150K 10.3%

A-OKVQA 10.3% ShareGPT 2.6% Captioning COYO100M 7.7%

Table 12. Sampling ratio during instruction tuning.

### C. Implementation Details

The detailed hyperparameters (HPs) are summarized in Table 11. Additionally, we utilize total six blocks in both CAbstractor and D-Abstractor (i.e., L = 3 for C-Abstractor and L = 6 for D-Abstractor in Fig. 4). We use a single node with A100 80GB × 8, employing deepspeed zero-2 [47] and flash-attention v2 [12] for all experiments, except for the pre-training of long schedule where we use multi-node setups.

erence points, sampling offsets, and attention weights are obtained via linear projection over the learnable queries z; that is, they are all learnable values. The introduction of reference points and sampling offsets for learnable queries allows locality modeling by enabling the collection of features near reference points via the sampling offsets.

Sampling ratio for datasets. As described in Section 4, balancing the wide range of datasets is important to train precise MLLMs. To maximize the learning of diverse knowledge from multifaceted datasets, we manually determine the sampling ratios of these datasets during training. In pre-training, COYO100M and BlipCapFilt are used in a 1:1 ratio. For instruction tuning, the specific sampling ratios of each dataset, determined through short schedule ablations, are detailed in Table 12. Notably, datasets such as VSR, ShareGPT, ScienceQA, OCRVQA, and Visual Genome (VG) have lower sampling ratios. The restricted scale of ShareGPT, VSR, and ScienceQA is due to their small dataset sizes, limited to a maximum of 3 epochs in short schedule criteria. On the other hand, the sampling ratio for OCRVQA and VG is set to 5.1%, derived empirically from ablation experiments. The exclusion of BlipCapFilt in instruction tuning stems from computational resource con-

On top of the deformable attention, we additionally present two techniques to improve local context modeling: 1) learnable query initialization through adaptive average pooling to the visual feature map instead of random initialization (v-pooled Q), and 2) a manual initialization of reference points uniformly distributing on visual feature maps instead of centralized initialization (M-RP). With these techniques, we can make reference points cover the whole region of an image, which results in offering more benefits in preserving local context with fine-grained information for a given image. The results in Table 9 demonstrate that two techniques provide overall performance improvements of MLLMs.

|Task|Dataset|Template|
|---|---|---|
| | | |
|Captioning|BlipCapFilt|AI: {caption}<br><br>|
| |COYO100M<br><br>|AI: {caption}|
|VQA (Open)<br><br>|VQAv2<br><br>|Human: Answer the question using a single word or phrase. {question} AI: {answer}|
| |GQA|Human: Answer the question using a single word or phrase. {question} AI: {answer}<br><br>|
| |OCRVQA<br><br>|Human: Answer the question using a single word or phrase. {question} AI: {answer}|
| |VSR|Human: Answer the question using a single word or phrase. {question} Please answer yes or no. AI: {answer}|
|VQA (MC)|ScienceQA|Human: Answer with the option’s letter from the given choices directly. {question} Context: {context} There are several options: {option} AI: {answer}<br><br>|
| |A-OKVQA<br><br>|Human: Answer with the option’s letter from the given choices directly. {question} There are several options: {option} AI: {answer}|
|REC|RefCOCO<br><br>[Figure 18]|Human: Provide the bounding box coordinate of the region this sentence describes: {phrase} AI: {bbox}<br><br>|
| | |Human: Provide a description for the region {bbox}, utilizing positional words to refer to objects. Example: ‘The large blue teddy bear next to the red balloon’ AI: {phrase}|
| |RefCOCO+<br><br>|Human: Provide the bounding box coordinate of the region this sentence describes: {phrase} AI: {bbox}<br><br>|
| | |Human: Provide a description for the region {bbox}, focusing on the appearance of objects without using positional words. Example: ‘The large blue teddy bear holding a red balloon.’ AI: {phrase}|
| |RefCOCOg<br><br>|Human: Provide the bounding box coordinate of the region this sentence describes: {phrase} AI: {bbox}<br><br>|
| | |Human: Provide a description for the region {bbox}, using detailed and descriptive expressions to refer to objects. Example: ‘The large blue teddy bear holding a red balloon with a joyful expression.’ AI: {phrase}|
| |Visual Genome<br><br>|Human: Provide the bounding box coordinate of the region this sentence describes: {phrase} AI: {bbox}<br><br>|
| | |Human: Provide a short description for this region: {bbox} AI: {phrase}|
|Instruction|LLaVA150k|Human: {instruction} AI: {response}<br><br>|
| |ShareGPT|Human: {instruction} AI: {response}|

- Table 13. Templates for individual dataset. We develop the templates based on LLaVA (v1.5) [33]. {*} is replaced depending on dataset examples where red-colored one means a target output. Note that bbox is expressed as normalized coordinates [xmin, ymin, xmax, ymax].

|Single-turn|
|---|

|Multi-turn|multi-turn|
|---|---|
|Human: Answer the question using a single word or phrase. What’s on the beach?<br><br>AI: pillow Human: Answer the question using a single word or phrase. What is on the beach?<br><br>AI: pillow Human: Answer the question using a single<br><br>word or phrase. What kind of furniture is to the right of the table?| |
| | |
| |de-duplication|

Human: Answer the question using a single word or phrase. What’s on the beach? AI: pillow

|Multi-turn w/ de-duplication|
|---|

Human: Answer the question using a single word or phrase. What’s on the beach?

AI: pillow Human: Answer the question using a single

word or phrase. What kind of furniture is to the right of the table? AI: dresser

AI: dresser

- Figure 5. The construction process of a multi-turn example with de-duplication. This example is sampled from the GQA [20] dataset.

periment with visual feature sources from the CLIP vision model [46]. The results show that utilizing features from the second-last layer rather than the last layer yields better performance [27]. (iii) LLaMA-2-chat and Vicuna-v1.5 show similar results, with Vicuna marginally outperforming, thus we use Vicuna. (iv) We applied LoRA to every query and value layer of attention following the original paper [18], yet found full tuning of LLM to be superior. While there may be ways to better utilize LoRA, such as increas-

straints, not from ablation results; we observe that including it does not notably affect the average performance.

### D. Additional Recipes

Table 10 presents additional ablation studies for our design choices. (i) There are several studies employing image indicator tokens [2, 54], yet they do not demonstrate the effectiveness of the indicator tokens. Our experiments show that omitting indicator tokens improves performance. (ii) We ex-

[Figure 19]

[Figure 20]

[Figure 21]

Q: What item is hanging on the wall behind the person in the image? A. Picture B. Clock C. Shelf D. Cabinet

Q: What color are the socks of the player nearest to the ball in the image? A. Yellow and blue B. Red C. Black and white D. Blue and yellow

Q: In the image, where is the person surfing?

- A. On a surfboard riding a large wave
- B. In a group of surfers riding wave
- C. Close to the shore
- D. In the middle of the ocean

- Figure 6. Examples of SEED-Bench. The examples require in-depth visual understanding; we highlight the regions (yellow boxes) that we need to focus on to get the correct answer (red-colored option).

[Figure 22]

[Figure 23]

[Figure 24]

Q: The image shows a python code. Is the output of the code ‘Hello’? Please answer yes or no. A. Yes

Q: The image shows a python code. Is the output of the code ‘a dog’? Please answer yes or no. A. No

Q: The image shows a python code. Is the output of the code ‘12’? Please answer yes or no. A. Yes

Q: The image shows a python code. Is the output of the code ‘2’? Please answer yes or no. A. No

(a) Code reasoning task

[Figure 25]

Q:Is the answer to the arithmetic question in the image 1511? Please answer yes or no. A. No

Q:Is the answer to the arithmetic question in the image 17? Please answer yes or no. A. No

Q:Is the answer to the arithmetic question in the image 65? Please answer yes or no. A. Yes

Q:Is the answer to the arithmetic question in the image 33? Please answer yes or no. A. Yes

(b) Numerical calculation task

[Figure 26]

Q:Is it appropriate to translate the Chinese in the image into English ‘classic taste’ in the picture? Please answer yes or no. A. Yes

Q:Is it appropriate to translate the Chinese in the image into English ‘a delicious dinner’ in the picture? Please answer yes or no. A. Yes

Q:Is it appropriate to translate the Chinese in the image into English ‘cold weather’ in the picture? Please answer yes or no. A. No

Q:Is it appropriate to translate the Chinese in the image into English ‘run very slow’ in the picture? Please answer yes or no. A. No

(c) Text translation task

Figure 7. Examples of MME with cognition taks.

### E. Details on Templates

ing its application scope or rank, we did not explore these further in this study. Experiments (v) and (vi) pertain to the long training schedule employed for our final model (Table 6). (v) In pre-training, we freeze the LLM and train only the projector. Here, extending pre-training, a feasible option with more computational resources, is beneficial, albeit with marginal improvements. (vi) When increasing instruction tuning steps, a broader consideration is necessary as continued LLM training can diminish its pre-trained knowledge and capabilities. Our experiments reveal that excessively long training is counterproductive, with around 10k training iterations being the most effective.

Templates. Detailed templates for individual datasets are presented in Table 13. For captioning tasks, MLLMs are encouraged to generate directly output captions without any instructional phrase as the standard captioning task. For VQA and REC tasks, we adopt fine-grained templates to favorably adapt LLM’s outputs for individual datasets. For the VSR dataset, we rephrase the declarative captions into questions to suit a VQA context. For instance, a caption “The cat is inside the refrigerator” marked as False is converted into “Is the cat inside the refrigerator?” with the answer No. Finally, for the instruction task, we use the original instruc-

[Figure 27]

[Figure 28]

Perception Cognition Model Existence Count Position Color Poster Celebrity Scene Landmark Artwork OCR Sum

Commonsense reasoning

Numerical calculation

Text translation

Code reasoning

Sum Total

C-7B 185.0 145.0 161.7 180.0 166.7 152.4 157.3 174.5 129.3 132.5 1584.2 112.1 37.5 100.0 57.5 307.1 1891.3 D-7B 175.0 153.3 143.3 175.0 155.4 148.2 153.3 163.3 129.8 147.5 1544.1 111.4 47.5 72.5 60.0 291.4 1835.5 C-13B 185.0 141.7 173.3 170.0 178.2 172.4 160.3 173.5 142.5 132.5 1629.3 127.1 47.5 80.0 60.0 314.6 1944.0 D-13B 195.0 175.0 146.7 168.3 168.0 164.7 156.5 174.5 131.0 152.5 1632.2 130.0 62.5 82.5 42.5 317.5 1949.7

(a) MME scores. Maximum scores are 200 for each subcategory, and 2000, 800, and 2800 for perception, cognition, and total, respectively.

Scene understanding

Instance identity

Instance attributes

Instance location

Instances counting

Spatial relation

Instance interaction

Visual reasoning

Text understanding

Model

Total

C-7B 73.4 67.8 64.6 59.8 55.6 48.4 73.2 74.9 41.2 64.5 D-7B 73.1 67.9 62.3 60.8 55.0 49.8 67.0 73.1 27.1 63.5 C-13B 75.4 74.0 68.1 65.5 59.2 54.2 71.1 79.5 38.8 68.2 D-13B 74.8 71.2 65.4 64.6 59.3 51.6 69.1 78.5 24.7 66.6

(b) SEEDI accuracies.

Model LR AR RR FP-S FP-C CP Total C-7B 41.7 78.1 69.6 74.1 53.8 80.2 70.1 D-7B 44.2 75.1 73.0 73.1 58.6 81.2 70.8 C-13B 45.8 77.6 77.4 76.8 57.9 83.6 73.2 D-13B 45.0 75.6 81.7 76.4 62.1 82.9 73.5

(c) MMB accuracies. Abbreviations stand for LR: Logic Reasoning, AR: Attribute Reasoning, RR: Relation Reasoning, FP-S: Fine-grained Perception (Single-instance), FP-C: Fine-grained Perception (Cross-instance), CP: Coarse Perception.

Model Complex Conv Detail All

C-7B 84.6 50.3 55.1 67.1 D-7B 79.6 49.4 62.6 66.3 C-13B 82.5 72.9 66.7 75.7 D-13B 84.1 68.6 57.8 72.9

(d) LLaVAW scores.

- Table 14. Detailed scores. C- and D- in Model column indicate C-Abstractor and D-Abstractor, respectively. 7B and 13B indicate LLM size. For the input images, we use 224 resolution for 7B and 336 for 13B.

tions and responses rather than using templates.

Multi-turn with de-duplication. For data such as VQA datasets where multiple input-target pairs exist for a single image, we make conversation-like multi-turn examples by simply concatenating the input-target pairs. Additionally, we perform a de-duplication strategy which remains only one from the duplicates (having the same target). The process is illustrated in Fig. 5.

### F. Benchmark Characteristics

Throughout this study, we observe specific characteristics in benchmarks, particularly in SEED-Bench and MME with cognition tasks (MME-cognition). SEED-Bench tends to require fine-grained visual comprehension, while MMEcognition is highly text-oriented, resulting in substantial dependency on the capabilities of LLMs. In this section, we investigate these distinctive benchmark characteristics.

SEED-Bench. We present examples of SEED-Bench, in Fig. 6, to show one of the major characteristics of the benchmark; we observe that the examples frequently require finegrained visual understanding, e.g., details from small regions. Such characteristics suggest that using large images

or more visual tokens is critical in achieving higher performance in this benchmark. Notably, in Table 6, Honeybee achieves competitive performance over comparative models even with smaller images or fewer visual tokens.

MME-cognition. We present examples of MMEcognition in Fig. 7. Notably, three out of four cognition tasks are text-oriented reasoning tasks, such as code reasoning, numerical calculation, and text translation. Consequently, the performance of these cognition tasks is predominantly influenced by which LLM is used, rather than the visual comprehension capabilities of MLLM. Furthermore, our analysis reveals a distinct bias in the text translation task towards Chinese-English translation. While only four examples are shown in Fig. 7, all instances of text translation tasks are observed to be Chinese-English translations. Considering such characteristics, we prioritize the MME with perception tasks (MMEP) over cognition tasks for model comparisons.

### G. Additional Results

#### G.1. Detailed Benchmark Scores

We report the detailed scores of our final models for all categories in MME, MMB, SEEDI, and LLaVAW in Table 14.

Subject Context Modality Grade

Model

Average

NAT SOC LAN TXT IMG NO G1-6 G7-12

Human [39] 90.23 84.97 87.48 89.60 87.50 88.10 91.59 82.42 88.40 GPT-3.5 [39] 75.44 70.87 78.09 74.68 67.43 79.93 78.23 69.68 75.17 GPT-4 [34] 84.06 73.45 87.36 81.87 70.75 90.73 84.69 79.10 82.69

###### Specialist Models

LLaMA-Adapter [62] 84.37 88.30 84.36 83.72 80.32 86.90 85.83 84.05 85.19 MM-CoT [64] 95.91 82.00 90.82 95.26 88.80 92.89 92.44 90.31 91.68 LLaVA [34] 90.36 95.95 88.00 89.49 88.00 90.66 90.93 90.90 90.92 LLaVA+GPT-4 (judge) [34] 91.56 96.74 91.09 90.62 88.99 93.52 92.73 92.16 92.53 Generalist Models

Honeybee (M=256) 93.12 96.63 90.55 92.52 91.77 92.26 93.72 92.22 93.19 Honeybee (M=576) 95.20 96.29 91.18 94.48 93.75 93.17 95.04 93.21 94.39

- Table 15. Evaluation results on the Science QA test split. Question classes: NAT = natural science, SOC = social science, LAN = language science, TXT = text context, IMG = image context, NO = no context, G1-6 = grades 1-6, G7-12 = grades 7-12. Despite specialist models being tailored explicitly for the ScienceQA benchmark, e.g., further fine-tuning solely on ScienceQA, Honeybee achieves state-ofthe-art scores under a generalist approach. We highlight the best results and second-best results in bold and underline.

76

Honeybee (13B, M=576)

74

Honeybee (7B, M=256) Honeybee (13B, M=256)

72

AvgN

Honeybee (7B, M=144)

70

LLaVA-1.5 (13B, M=576)

68

66

Qwen-VL-Chat (7B, M=256)

64

2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0

Step time (s/step)

- Figure 8. Comparison between Honeybee variants and current state-of-the-art methods. AvgN denotes the normalized average score of MMB, MMEP, and SEEDI.

#### G.2. Pushing the Limits

Table 7 in the main text shows the performance of Honeybee with the increased number of visual tokens, matching them to the linear projector. Here, we further provide the comparison between the Honeybee variants and the current stateof-the-art methods, namely Qwen-VL-Chat [2] and LLaVA1.5 [33], in Fig. 8. This figure highlights the efficiency and effectiveness of the proposed Honeybee.

#### G.3. Science QA

The Science QA dataset [39] is specifically designed to evaluate the broadness of domain knowledge and multihop reasoning skills of AI systems, which is essential for MLLMs to perform a wider range of tasks requiring more complex reasoning. Thus, in this section, we additionally provide the evaluation results of the Science QA benchmark. From Table 15, recent MLLMs, i.e., LLaMAadapter [62], MM-CoT [64], and LLaVA [34], show remarkable performance in this benchmark via further fine-

MM-Vet MMMU POPE Approaches using 7B LLM

|LLaVA (v1) MiniGPT-4 LLaMA-AdapterV2 mPLUG-Owl InstructBLIP Qwen-VL-Chat LLaVA-1.5<br><br>|23.8 - 66.5 22.1 - 31.4 29.8 -<br><br>- - 67.4 26.2 - -<br>- 35.9 30.5 - 85.9<br><br><br>|
|---|---|
|Honeybee (C-7B-144M) Honeybee (C-7B-256M)|34.9 35.3 83.2<br><br>35.6 36.4 84.3<br>|

Approaches using 13B LLM

|MiniGPT-4 BLIP-2 InstructBLIP LLaVA-1.5<br><br>|24.4 26.8 74.5 22.4 35.4 85.3<br>25.6 35.7 83.8 35.4 36.4 85.9<br><br><br>|
|---|---|
|Honeybee (C-13B-256M) Honeybee (C-13B-576M)<br><br>|38.1 37.3 85.5 42.2 36.2 85.6<br><br>|

Table 16. Additional benchmark comparison. Numbers are collected from each paper and official leaderboard, selecting the best results for each method when multiple exist.

tuning on the Science QA dataset; we refer to these finetuned models as Specialist Models in Table 15. Especially, in LLaVA+GPT-4 (judge), they achieved state-of-theart scores by utilizing the GPT-4 [44] as a judge; whenever GPT-4 and LLaVA produce different answers, they prompt GPT-4 again, asking it to provide a final answer based on the question and two outcomes. Remarkably, Honeybee, with C-Abstractor and vicuna-13B, outperforms the LLaVA+GPT-4 (judge) and achieves new state-of-the-art scores in this benchmark without the assist of GPT-4 or the task-specific fine-tuning process. These results highlight the effectiveness of our contributions: 1) architectural improvement of the projector and 2) thoroughly explored training recipe.

[Figure 30]

[Figure 31]

[Figure 32]

- Figure 9. Visualization of attention maps. (Left) the input image, (Middle) the attention map from the resampler, and (Right) the attention map from D-Abstractor. Our locality-aware projector (D-Abstractor) effectively preserves local contexts, while the resampler extracts visual information mainly from a few regions and loses some details.

#### G.4. Additional Benchmark Results

In addition to four benchmarks used in the main paper, we perform further evaluation using three additional benchmarks—1) MM-Vet [58] and MMMU [60] for visual understanding capability evaluation, and 2) POPE [30] for evaluation of object hallucination. In Table 16, similar to the experimental results in the main text, Honeybee shows superior comprehensive visual understanding on MM-Vet and MMMU benchmarks. On the other hand, in the hallucination aspect, the performance of 7B-scale Honeybee slightly falls short compared to LLaVA-1.5. However, when using larger images (336 resolution) with 13B LLM, it achieves competitive performance, suggesting the importance of higher visual understanding and reasoning to tackle hallucination issue.

### H. Qualitative Analysis

#### H.1. Attention Comparison between Resampler and D-Abstractor

As discussed in Section 3.2.1, the vanilla abstractor (i.e., resampler) tends to primarily focus on salient regions, whereas our locality-enhanced abstractor (i.e., DAbstractor) is designed to preserve local contexts effectively. To further validate this, we examined attention maps from both the resampler and the D-Abstractor for their every learnable query (M=144). From Fig. 9, we observe that queries of the resampler only attend to specific salient areas, suggesting the potential loss of detailed information. On the other hand, in the case of D-Abstractor, each learnable query locally abstracts visual features across the whole feature map, which provides fine-grained and comprehensive

information about the image. This result indicates that our proposed projectors benefit in improving the performance of spatial understanding tasks that require to capture diverse relationships and objects in an image.

#### H.2. Qualitative Examples

In Fig. 10, we present generation results for various user instructions using samples of OwlEval [54]. To verify the importance of instruction-following data for human preference, we report both results generated from two different models: 1) a model trained without instruction-following data such as LLaVA [34] and ShareGPT [10], called Honeybee w/o Inst. (D9 in Table 4), and 2) a model trained with all available data, including the instruction-following data, called Honeybee w/ Inst. (D1 in Table 4). From Fig. 10, while the Honeybee w/o Inst. captures objects included in given images exactly, the answers tend to be short or not follow the instructions. In contrast, Honeybee w/ Inst. accurately responds to user instructions and generates detailed answers preferred by humans.

[Figure 33]

User

Why would a person find this image funny?

Honeybee w/o Inst.

Squirrel with camera

Honeybee w/ Inst.

A person might find this image funny because it features a squirrel, which is a small, cute, and often mischievous creature, holding a camera and seemingly posing for a photo. The idea of a rodent holding human-like objects, such as a Camera, can be amusing and whimsical. Furthermore, the squiggly-eyed squirming squid expression adds to the humor by making the scene even more exaggerated and light-hearted. People often find anthropomorphic representations of animals and unexpected situations like this entertaining.

[Figure 34]

User

What is unusual about this image?

Honeybee w/o Inst.

Man drying clothes on car

Honeybee w/ Inst.

In the image, a man is ironing clothes while standing on the back of a car parked on a street. This is unusual because iron is not typically used in outdoor settings, and it is even more uncommon to see someone iron clothes on top of an automobile. Furthermore, the presence of the man iron and the car with its hood open create a unique and unexpected scene that captures the attention of passersby, as they may not commonly witness such an unusual event.

[Figure 35]

User

How many championship rings did the player on the left win in his career?

Honeybee w/o Inst.

5

Honeybee w/ Inst.

The player on the left, wearing a yellow jersey, is Kobe Bryant. Kobi Bryant won five championship rings in his career, all of which were won with the Los Angeles Lakers.

[Figure 36]

User

Is this image common in real world?

Honeybee w/o Inst.

No

Honeybee w/ Inst.

The image depicts an unusual and surreal scene of a cactus surrounded by ice cubes. This is not a common sight in the real world, as cacti and ice are generally found in different environments and climates. Cacti typically grow in hot and arid desert regions, while ice is formed in colder temperatures and environments. The combination of the two elements in this image is likely a result of photo manipulation or staging, creating an interesting and unconventional visual effect.

[Figure 37]

User

Write a conversation between the two animals.

Honeybee w/o Inst.

Cat on dog

Honeybee w/ Inst.

In the image, a cat is sitting on the back of a dog that is lying in the snow. It is an unusual and amusing scene. If the cat and the dog could communicate, they might have a conversation something like this: Cat: purrs "Hey there, big guy. Thanks for letting me ride you." Dog: wag wags "You're welcome, feline friend. I love having you on my back."

[Figure 38]

User

Help me write an advertisement for this.

Honeybee w/o Inst.

Bring home a piece of art with this gorgeous golden bird lamp

Honeybee w/ Inst.

Introducing our unique and whimsical bird-shaped table lamp. This charming piece features a beautifully crafted bird with a large, long beak, perched on one foot and holding a light bulb in its beak. The lamp's design adds a touch of creativity and playfulness to any space, making it an ideal addition to a living room, bedroom, or office. With its eye-catching appearance, this bird lamp will spark conversation and bring a sense of joy and charm to your room. Place it on a table or desk to create a cozy ambiance with its warm glow. Order yours today!

- Figure 10. Qualitative examples with various user inputs. w/ Inst. and w/o Inst. indicate results from models trained with or without instruction-following data, i.e., LLaVA [34] and ShareGPT [10], respectively. The example images are selected from OwlEval [54].

