|[Figure 1]<br><br>Chimera: Improving Generalist Model with Domain-Specific Experts<br><br>Tianshuo Peng1,2,∗, Mingsheng Li3,∗, Jiakang Yuan3, Hongbin Zhou1, Renqiu Xia1,4, Renrui Zhang2 Lei Bai1, Song Mao1, Bin Wang1, Aojun Zhou2, Botian Shi1<br><br>Tao Chen3,5, Bo Zhang1,‡, , Xiangyu Yue2,<br><br>1Shanghai Artificial Intelligence Laboratory, 2MMLab, The Chinese University of Hong Kong 3College of Future Information Technology, Fudan University, 4Shanghai Jiao Tong University 5Shanghai Innovation Institute<br><br>* Equal Contribution, Corresponding Authors, ‡ Project Lead<br><br>Abstract<br><br>Large Multi-modal Models (LMMs), trained on webscale datasets predominantly composed of natural images, have demonstrated remarkable performance on general tasks. However, these models often exhibit limited specialized capabilities for domain-specific tasks that require extensive domain prior knowledge. An intuitive solution is to post-train LMMs on a specific domain, but often suffers from the labor-intensive annotating process and the inaccessibility of private training data. Directly integrating expert models tailored for those tasks is also challenging due to representational gaps and imbalanced optimization. To address these challenges, we introduce Chimera, a scalable and low-cost multi-modal pipeline designed to boost the ability of existing LMMs with domain-specific experts. Specifically, we design a progressive training strategy to integrate features from expert models into the input of a<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>arXiv:2412.05983v3[cs.CV]26Jul2025|
|---|

generalist LMM. To address the imbalanced optimization caused by the well-aligned general visual encoder, we introduce a novel Generalist-Specialist Collaboration Masking (GSCM) mechanism. This results in a versatile model that excels across the chart, table, math, and document domains, achieving state-of-the-art performance on multi-modal reasoning and visual content extraction tasks, both of which are challenging tasks for assessing existing LMMs. We will release model weights, along with the data used for training and evaluation, to facilitate future research on LMMs.

Figure 1. Performance comparison of different models on multimodal reasoning (MathVista, MathVerse) and visual structural extraction (ChartQA-SE, Table-SE) tasks.

trieval [7, 84], demonstrating their potential as a technical pathway towards a general-purpose AI assistant. Although proficient in a wide range of tasks, their performance still lags behind that of models fine-tuned with target-domain data, especially in specialized tasks such as multi-modal reasoning and visual content extraction. As depicted in Fig. 1, state-of-the-art general-purpose LMMs demonstrate significant limitations in addressing these tasks, highlighting the necessity for further research to bridge this gap.

###### 1. Introduction

The past year has witnessed the remarkable success of Large Multi-modal Models (LMMs) in handling a variety of general domain tasks, such as image captioning [10, 14, 41], visual dialog [4, 11, 12, 59, 78], and cross-modal re-

Current research on LMMs [10, 12, 40, 59, 60, 62, 85] has extensively invested in scaling up by collecting webscale image-text pairs and employing multi-task instruction tuning to develop generalist models, following a “One for

All” paradigms [12, 77]. However, the pursuit of generality often results in suboptimal performance in domain-specific tasks, such as Chart [84], Table [83], and Math [20, 89, 93]. This is mainly due to the substantial differences between natural images and those found in specialized fields [37]. For instance, domain-specific tasks such as multi-modal reasoning and visual structural extraction often involve content that includes charts, tables, geometric figures, and function graphs [17, 83, 84, 92]. These tasks are characterized by higher text density and more abstract content [75, 80]. As a result, general LMMs, which are primarily trained on web-scale natural images, struggle to adapt effectively to these specialized contexts [30, 84, 89].

To enhance performance in target domains, numerous studies have focused on developing tailored models or taskspecific architectures for downstream tasks [17, 67, 80, 83, 84, 89, 92], adopting a “One for One” paradigm where models are trained on a single scene type. While these expert models exhibit strong capabilities in specialized tasks, they are often criticized for being designed to address individual scenarios. This phenomenon arises from significant distribution gaps across various sub-domains, such as tables, charts, functions, and geometry, potentially sacrificing their generalizability across broader applications using specialized models.

To push the boundary further for the existing LMMs and improve their performance in specialized domains, an intuitive solution is to post-train LMMs on data relevant to the target domain. However, a common challenge is that the vast amounts of domain-specific data necessary for specialist models are often proprietary and inaccessible. On the other hand, integrating specialist experts that contain specialized prior knowledge presents a promising approach to address this issue [66, 94]. Moreover, directly combining specialist experts with the generalist model could result in unsatisfactory performance, due to the following factors: 1) large distribution shifts between cross-domain encoders, 2) imbalanced optimization for generalists and specialists.

To address these challenges, this work introduces Chimera: a flexible and scalable pipeline that can effectively scale up off-the-shelf experts into LMMs at low cost. Specifically, we utilize a lightweight routing module that dynamically selects tokens from the most suitable experts based on visual content, enabling tailored input to the LLM. Through cost-effective training aimed at feature alignment, we integrate multiple encoders from different expert models into a single LMM, effectively merging diverse specialized knowledge without requiring vast amounts of target-domain data. Besides, we observed alignment imbalances during the cross-modal encoder fusion and propose a General-Expert Collaboration Masking mechanism to facilitate better model fusion. Our method easily adapts LMMs, such as InternVL [12, 13], to a range

of domain-specific tasks, including advanced mathematical reasoning, table/chart QA & extraction, and document structural extraction tasks. By aggregating multiple expert models into a single general LMM, Chimera develops a versatile model endowed with multiple specialized capabilities. During inference, Chimera employs a simple routing module to determine whether to invoke the corresponding domain expert model based on the visual input, resulting in a versatile model that excels across the chart, table, math, and document domains, as well as tasks involving multi-modal reasoning and extraction.

We conduct extensive experiments to evaluate Chimera’s capabilities in multi-modal reasoning and visual content extraction, both of which are challenging domains for assessing existing LMMs. With the introduction of domain knowledge from expert models and supervised fine-tuning, Chimera achieves overall accuracies of 64.9 and 32.4 on the multi-modal reasoning benchmarks MathVista [48] and MathVerse [90], setting a new State-Of-The-Art (SOTA) for LMMs of comparable scale. Direct preference optimization can further boost Chimera’s reasoning capabilities, allowing it to achieve superior performance with a small amount of data. It also surpasses or matches the performance of representative expert models in visual content extraction tasks across chart, table, and document domains.

Our contributions can be summarized as follows:

- 1. We introduce Chimera, a scalable pipeline that integrates specialist models into generalist LMMs, facilitating their adaptation to many specialized tasks.
- 2. We present a lightweight routing module that dynamically selects the most relevant experts based on visual input, coupled with Generalist-Specialist Collaboration Masking (GSCM) aimed at facilitating representation alignment between the generalist and domain experts.
- 3. Chimera achieves SOTA performance on challenging benchmarks for reasoning, including MathVista and MathVerse. Furthermore, it achieves near-specialistlevel results in visual structural extraction on benchmarks like ChartQA-SE, Table-SE, Doc-SE, etc.

###### 2. Related Work

Generalist Large Multi-modal Models. Following the remarkable success of Large Language Models (LLMs) [6, 73, 74], researchers have made great efforts in adapting LLMs for multi-modal tasks in a general context, contributing to the flourishing of Large Multi-modal Models (LMMs) [10, 12, 34, 40, 51, 59, 60, 62, 85]. Recent LMMs typically utilize a cross-modal connector [12, 32] and perform the pre-training on large-scale natural imagetext datasets [35, 83] to alleviate the modality gap between the visual encoder and the LLMs. For instance, BLIP series [31, 32] utilizes captions from datasets like COCO [39], CC3M [65], SBU [61], and LAION [64], while the LLaVA

series [44, 45] constructs complex instruction-following datasets based on natural images from COCO [39] to further enhance their understanding of visual content.

However, the pursuit of generality often results in limited performance in specialized scenarios, such as geometric and function reasoning [17, 82, 86, 89], table and chart understanding [75, 84, 92], of which the visual content differing differs significantly from natural images. Moreover, finetuning LMMs on specialized domains remains challenging due to inaccessible private data and potential degradation in general performance.

Expert Models on Specialized Scenarios. Expert tasks in multimodal settings, such as geometric and function reasoning [82, 89, 90], table and chart understanding [84, 92], and document information extraction [75], often require specialized designs to achieve optimal task performance. For example, Math-LLaVA [67], and MAVIS [89] train LMMs on carefully curated mathematical datasets using natural language descriptions. Table-LLaVA [92] constructs a large-scale multimodal table understanding dataset, while StructEqTable [83] uses extensive table format transformation data to build a highly specialized expert model with limited generality. Similarly, ChartGemma [55] and ChartInstruct [54] train LMMs on diverse chart instructionfollowing data, and ChartVLM [84] employs a router structure to selectively engage different decoders for base perception tasks and cognition tasks. GOT [80] trains a specialist model on million-scale private data specifically for document structural extraction task. Besides, some work executes complex and challenging tasks, including automated scientific discovery, in the form of multi-agents through carefully designed and comprehensive workflows [71].

While expert models excel in specific domains, they struggle with tasks outside their specialization. In contrast, Chimera integrates specialized knowledge into a generalist LMM, achieving superior performance across both multimodal reasoning and document context extraction tasks.

###### 3. Methodology

To develop an assistant that can adapt to challenging domains at a low cost, we propose Chimera, a scalable multimodal model built upon existing LMMs and pretrained expert models. In this section, we first introduce an overview of Chimera in Sec. 3.1. Sec. 3.2 discusses the integration of generalists and domain-specific experts, and Sec. 3.3 details Generalist-Specialist Collaboration Masking (GSCM), an effective method for collaboration between the generalist model and specialists. Furthermore, we present the training recipe for Chimera in Sec. 3.4.

###### 3.1. Overview

As illustrated in Fig 2, Chimera consists of: a general visual encoder Eg, a general projector Pg together with a language Model f initialized from a pretrained LMM, a router

R, an expert model set Se with Ne expert models and corresponding expert projector set Sp. Assuming expert models from the domains of table, chart, and math as aggregation targets, we have:

Se = {Etable,Echart,Emath} Sp = {Ptable,Pchart,Pmath}.

(1)

Generalist Branch. For visual input Xv, Eg privides the general visual features Zv = Eg(Xv), Pg projects general visual features into word embedding space, yielding general visual tokens Hv = Pg(Zv). During training, we apply the GSCM mechanism on Hv as Hvm to replace Hv.

Specialist Branch. The linear layer R first predicts routing value Hr ∈ RN

e+1 as Hr = R(Zvcls), where Zvcls represents the classification token of Zv, determining whether to invoke an expert model and which specific expert model to call. Consequently, the expert visual tokens He can be formulated as:

(Hr)i

i = arg max

i

(2)

∅, if i == 0, Sip (Sie (Xv)), otherwise.

He =

Given the text embedding Ht of instruction Xt, the input sequence during training is formulated as:

Hinput = concat([Hvm : He : Ht]). (3)

We validated Chimera’s capability and adaptability in two distinctly different task scenarios: multi-modal reasoning and visual content extraction, both of which are challenging domains for assessing existing LMMs. The former scenario requires integrating expert models including those for tables, math, and charts. The latter scenario requires integrating expert models that specialize in document structural extraction.

###### 3.2. Integration of Generalist and Specialist

There are two intuitive ideas to adapt an generalist LMM into specialized domain: 1) performing supervised finetuning on domain-specific data (naive finetune) and 2) sequentially appending features from different encoders (naive concat). The primary difference between Chimera and these two approaches lies in the definition of the input sequence during training for the language model f. Let Hinputnf and Hinputnc denote the input sequences in the naive finetune and naive concat methods, respectively. They can be formulated as:

Hinputnf = concat([Hv : Ht]), Hinputnc = concat([Hv : He : Ht]).

(4)

The former approach attempts to use a single visual encoder to handle all visual content, which refuses to incorporate domain-specific knowledge from expert models. Finetuning on subtasks in several specialized domains can also

[Figure 23]

Language Model

[Figure 24]

###### ... ... ...

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Masking Router

[Figure 30]

[Figure 31]

Projector

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Visual Encoder

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

###### CHIMERA

[Figure 45]

[Figure 46]

[Figure 47]

：Frozen parameters

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

：Visual input

[Figure 53]

：Text instruction input

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

：Target response sequence

[Figure 59]

：Expert visual tokens

General Table Chart Math

[Figure 60]

：General visual features

- Figure 2. Overview of our Chimera framework. Chimera uses Generalist-Specialist Collaboration Masking to facilitate the alignment with expert models. During inference, the Router R decides expert invocations based on the visual input, resulting in a versatile model that excels across multiple specialized domains and tasks.

reduce generalizability, leading to trade-offs or suboptimal performance across subtasks. The latter approach incorporates encoded features from various domains, but applying this directly to a well-aligned LMM may lead to misalignment between the generalist and specialist models—a limitation we will discuss in the next section.

###### 3.3. Generalist-Specialist Collaboration Masking

Although naive concat method with input Hinputnc is intuitive, we still concern that since the general visual encoder Eg is well-aligned with language Model f, it may cause the model to overly rely on Eg to complete tasks, which leads to ineffective alignment with the expert models. To better align domain knowledge and general world knowledge, we propose a simple yet effective learning mechanism called Generalist-Specialist Collaboration Masking, designed to boost the synergy between general-purpose and domain-specific capabilities.

During training, we sample a subset of general visual tokens from Hv at a certain ratio and mask them to build the masked general visual tokens Hvm. In practice, this is achieved by setting the attention mask corresponding to the sampled subset to False. We consider a simple sampling strategy: randomly sampling tokens without replacement according to a uniform distribution. Applying mask to information provided by general encoder Eg produces a limi-

tation on Eg, which will force the model to utilize domainspecific information provided by expert models as supplements for vision-language tasks. The uniform distribution helps prevent bias that may arise from masking predominantly in the image center or specific regions.

###### 3.4. Training Recipe

To equip multi-modal generalists with rich domain-specific knowledge, we apply a progressive training strategy, including Domain-General Knowledge Alignment and Visual Instruction Tuning. Through two-stage training, we develop variants for different scenarios respectively. All the datasets used are publicly available. Details about the datasets can be found in the supplementary material.

Domain-General Knowledge Alignment. To initially align domain-specific knowledge with the semantic space of the generalist LMM, we train the model using tasks that directly perceive diverse image content. The tasks include natural image description, table format transformation, chart structural extraction and summarization, math diagram captioning, and paragraph-level OCR.

With guidance from image-text pairs across different domains, the model is able to leverage domain knowledge from expert models to accurately recognize visual content in each domain and describe its spatial arrangement. This marks the first step toward deeper integration. In this stage,

we freeze the general visual encoder Eg, expert model set Se and language model f, only train the router R, general projector Pg and expert projector set Sp.

Visual Instruction Tuning. To further align model with domain knowledge from expert models and enhance its performance on specialized tasks across different domains, we take instruction-following datasets from various domains to perform visual instruction tuning with the proposed GSCM. During this stage, we unfreeze router R, general projector Pg, expert projector set Sp and language model f, perform a thorough instruction-following tuning, which finally results in the versatile Chimera models.

Training Objective. Our primary training objective is to optimize the trainable parameters θ, so that the likelihood of target response sequence Xa is maximized given the visual input Xv and instruction Xt as follows:

θ∗ = arg max

P(Xa|Xv,Xt;θ). (5)

θ

To accomplish this, we utilize token-wise cross-entropy loss to train the model in an auto-regressive manner. For target Xa of length L, the auto-regressive modeling loss Lm is represented as follows:

Lm = −

i=1

L

log P(xi|Xv,Xt,Xa;<i,θ), (6)

where Xa;<i are the tokens before the current prediction token xi.

Besides, we add classification loss to guide the Router to accurately call different expert models based on image content,which can be represented as follows:

Ne+1

log P(ci|Xv,θ), (7)

Lc = −

i=0

where ci represents the expert domain category that the current image requires for invocation (including category 0, which means no expert model is invoked). Finally, the optimization objective is formulated as follows:

L = Lc + Lm. (8)

###### 4. Experiments

To evaluate the capabilities of Chimera, we begin by detailing the datasets and metrics used for evaluation, along with the implementations for multi-modal reasoning and visual content extraction in Sec. 4.1. Then, we provide a comparison of Chimera models against previous generalist models across various benchmarks, including multi-modal reasoning (Sec. 4.2) and visual structural extraction (Sec. 4.3). Besides, we conduct further analysis on the model design and training strategy in Sec. 4.4.

- 4.1. Datasets, Metrics and Implementation Details Datasets. In this paper, we conduct quantitative evaluations of Chimera model across a range of challenging multimodal benchmarks, categorized into the following areas:

- • Multi-modal Reasoning. We evaluate the Chimera on the MathVista [48] to determine its visual reasoning capabilities. Besides, we extend our evaluation on the MathVerse [90], which is specifically designed for mathematical problem-solving, to gauge its performance in multimodal mathematical reasoning.
- • Visual Structural Extraction (SE). Evaluations of Chimera on chart domain are conducted on the challenging ChartQA-SE [52] and PlotQA-SE [84] benchmarks. Following the protocol of StructChart [81], we utilize the test sets of both the ChartQA [52] and PlotQA [57] to ensure a fair comparison. For table and document, we manually collected and annotated a table format transformation benchmark called Table-SE and a document structural extraction benchmark called Doc-SE. Details regarding the data collection and annotation process can be found in the supplementary material.

Metrics. In evaluations, we adhere to the default metrics used by benchmarks, such as MathVista, MathVerse, ChartQA-SE, and PlotQA-SE. For the assessment of the table format transformation, we use Tree-EditDistance-based Similarity (TEDS) score and Edit Distance for evaluation. For document structural extraction, we take Edit Distance, Precision, BLEU and METEOR as evaluation metrics.

Implementation Details. We initialize Chimera using the InternVL2 series. Specifically, we use InternVL2-2B, 4B, 8B to construct Chimera-2B, 4B, and 8B for multi-modal reasoning, while using InternVL-1B to build Chimera for visual content extraction. In each training phase, we train the model for one epoch on the public datasets (refer to supplementary materials for more details). For multimodal reasoning scenario, we take StructEqTable [83], ChartVLM [84] and Math-CLIP [89] as table, chart and math expert respectively. For visual content extraction scenario, we employ GOT [80] as the document expert. More detailed implementation specifics can be found in the supplementary material.

- 4.2. Comparison on Multi-modal Reasoning Comparison with Generalist Models. LMMs, such as LLaVA-OneVision [30], Qwen2-VL [77],InternVL2 [12] and GPT-4o demonstrate powerful multi-modal reasoning abilities on general purpose scenarios. However, these generalist models always exhibit limited performance when handling tasks under professional scenarios. Our model demonstrates exceptional performance on the challenging multi-modal reasoning benchmarks MathVista and MathVerse, significantly outperforming existing generalist models. As shown in Tab. 1 and Tab. 2, Chimera-8B achieved overall accuracies of 64.9 and 32.4, respectively, setting

Model #Params. ALL FQA GPS MWP TQA VQA ALG ARI GEO LOG NUM SCI STA

Close Source LMMs InternVL2-Pro [12] - 66.8 70.6 65.4 76.9 71.5 48.0 66.5 62.3 63.6 27.0 40.3 65.6 81.1 Gemini 1.5 Pro [70] - 63.9 - - - - - - - - - - - GPT-4o - 63.8 - - - - - - - - - - - Grok-1.5V - 52.8 - - - - - - - - - - - Claude 3 Opus [1] - 50.5 - - - - - - - - - - - GPT-4V (Playground) - 49.9 43.1 50.5 57.5 65.2 38.0 53.0 49.0 51.0 21.6 20.1 63.1 55.8

Open Source LMMs LLaVA-OneVision [30] 72B 67.5 - - - - - - - - - - - Math-LLaVA∗ [67] 13B 46.6 37.2 57.7 56.5 51.3 33.5 53.0 40.2 56.5 16.2 33.3 49.2 43.9 Pixtral [2] 12B 58.0 - - - - - - - - - - - SPHINX-MoE [40] 8×7B 42.7 - - - - - - - - - - - InternLM-XComposer2 [15] 7B 57.6 55.0 63.0 73.7 56.3 39.7 56.6 52.4 62.3 8.1 42.4 59.0 64.1 LLaVA-OneVision [30] 7B 63.2 - - - - - - - - - - - Math-PUMA-DeepSeek-Math∗ [93] 7B 44.7 42.8 39.9 67.7 42.4 31.3 39.2 41.9 41.4 8.1 36.8 48.4 52.5

2B 43.0 - - - - - - - - - - - -

Qwen2-VL [77]

- 7B 58.2 - - - - - - - - - - - -

IntenrVL2 [12]

2B 48.3 51.3 45.7 40.9 50.6 52.5 43.4 47.3 42.3 13.5 28.5 53.3 56.8 4B 57.0 58.0 58.2 62.4 57.0 48.6 55.9 53.8 55.2 13.5 30.6 59.0 65.1

- 8B 61.6 62.5 64.4 61.3 64.6 54.7 63.0 58.9 61.9 18.9 34.0 59.0 70.1

2B 53.1 52.4 56.7 62.9 51.9 40.8 52.7 47.6 56.1 10.8 34.0 52.5 61.1 4B 61.3 58.4 66.8 72.0 61.4 48.0 63.3 54.7 65.7 24.3 39.6 60.7 66.4 8B 64.9 62.8 71.6 72.6 65.2 52.0 67.6 57.8 69.5 21.6 45.8 61.5 69.4

Chimera

LMMs with Preference Optimization InternVL-MPO [79] 8B 67.3 72.5 73.6 69.9 66.5 50.3 70.1 57.5 71.5 27.0 43.1 65.6 79.1 Ovis1.6-Gemma2 [50] 9B 67.2 - - - - - - - - - - - -

Chimera† 8B 68.3 66.5 76.9 80.1 60.8 55.3 69.7 64.8 74.5 13.5 49.3 62.2 77.7 Human performance - 60.3 59.7 48.4 73.0 63.2 55.9 50.9 59.2 51.4 40.7 53.8 64.9 63.9

Table 1. Accuracy scores on the testmini subset of MathVista. Task types: FQA: figure QA, GPS: geometry problem solving, MWP: math word problem, TQA: textbook QA, VQA: visual QA. Math reasoning types: ALG: algebraic, ARI: arithmetic, GEO: geometry, LOG: logical , NUM: numeric, SCI: scientific, STA: statistical. Chimera† represents post-trained Chimera model. ∗ represents the domain expert model. Chimera set a new SOTA results of 64.9 among open source LMMs under 70B scale. Direct preference optimization further boost Chimera’s performance to 68.3, outperforming the latest LMMs with preference optimization.

a new state-of-the-art (SOTA) for LMMs under the 70B scale. Chimera-2B and Chimera-4B both significantly outperform the baseline InternVL2 series and achieve results comparable to models of much larger scales. On MathVista, Chimera-8B stands out among both closed-source and open-source LMMs of the same size, leading GPT-4o by 1.1%, and outperforming Qwen2-VL and InternVL by 6.7% and 3.9%, respectively. On MathVerse, Chimera-8B is only slightly behind GPT-4V, and surpasses the hundredbillion-scale LLaVA-NeXT by 7.9 points. This demonstrates that our approach, by integrating domain knowledge from different expert models, effectively enhances performance in specialized domains.

Comparison with Specialist Models. Compared to specialist models such as Math-LLaVA [67], MathPUMA [93], MAVIS [89], and GeoX [82], Chimera demonstrates outstanding performance. As reported in Tab. 1 and Tab. 2, Chimera-8B outperforms the previous best expert models by 18.3% and 4.9% on MathVista and MathVerse,

respectively. In contrast, the latest expert model, MathPUMA [93], achieves performances of 44.7 and 31.8 on the two benchmarks, which is notably inferior to our method. It is worth noting that these expert models have limited generalization ability and cannot handle tasks from other domains. In contrast, our model excels across various tasks in the table, chart, and document domains, proving Chimera’s powerful versatility.

Further Improve Reasoning Ability with Preference Optimization. The emergence of preference optimization like RLHF has revolutionized model enhancement strategies. Chimera is able to capitalize on this trend through seamless preference optimization training integration, demonstrating remarkable scalability. We construct 60K preference pairs using publicly available datasets and Chimera’s outputs and perform a naive Direct Preference Optimization to develop Chimera†. Specific data construction process can be found in the supplementary material. Chimera†, with naive post-train, surpasses SOTA post-training methods by

Model #Params. All Acc Text Dominant Text Lite Vision Intensive Vision Dominant Vision Only Closed-source MLLMs

Gemini-Pro [69] - 23.5 26.3 23.5 23.0 22.3 22.2 Qwen-VL-Max [5] - 25.3 30.7 26.1 24.1 24.1 21.4 GPT-4V - 39.4 54.7 41.4 34.9 34.4 31.6

Open-source MLLMs SPHINX-Plus [40] 13B 14.0 16.3 12.8 12.9 14.7 13.2 SPHINX-MoE [40] 8×7B 15.0 22.2 16.4 14.8 12.6 9.1 LLaVA-NeXT [29] 110B 24.5 31.7 24.1 24.0 22.1 20.7 LLaVA-NeXT [29] 8B 19.3 24.9 20.9 20.8 16.1 13.8 InternLM-XComposer2 [15] 7B 16.5 22.3 17.0 15.7 16.4 11.0 Math-LLaVA∗ [67] 13B 19.0 21.2 19.8 20.2 17.6 16.4 MAVIS-7B∗ [89] 7B 27.5 41.4 29.1 27.4 24.9 14.6 Math-PUMA-DeepSeek-Math∗ [93] 7B 31.8 43.4 35.4 33.6 31.6 14.7

2B 21.4 24.1 22.5 22.8 21.1 16.6 4B 26.3 32.0 28.6 28.0 24.4 18.8

InternVL2 [12]

- 8B 31.3 38.8 34.5 33.6 32.6 17.0

Chimera

2B 22.6 27.3 23.9 22.3 22.8 16.9 4B 27.2 31.4 30.8 29.7 25.7 18.2

- 8B 32.4 39.6 35.8 34.8 32.7 19.3

Table 2. Performance Comparison on MathVerse with the accuracy metric. ∗ represents the domain expert model.

Model ALL General Chart Table Math

InternVL2-2B [12] 48.3 45.3 58.9 50.0 44.2 InternVL2-4B [12] 57.0 50.1 66.2 65.7 58.3 InternVL2-8B [12] 61.6 52.7 71.2 67.1 66.5

Chimera-2B 53.1 46.0 60.3 62.9 56.1 Chimera-4B 61.3 54.0 64.8 72.9 66.9 Chimera-8B 64.9 57.5 71.2 62.9 71.9

- Table 3. Accuracy scores of different visual content domain on the testmini subset of MathVista.Those do not belong to the last three domains are uniformly classified as General for simplicity.

1.0%, shows significant 3.4% improvement over base models without post-training, highlighting the framework’s scalability and potential.

Fine-Grained Analysis. We manually classified MathVista questions by the domain of visual content and present the model’s performance across domains in Tab. 3. In most cases, the model outperforms its baseline in each domain, further demonstrating that incorporating expert models enhances the generalist model’s performance on specialized tasks. We also observed that expert models improve performance in general scenarios, suggesting that domain knowledge provides diverse insights for language model in handling visual information, thereby enhancing the model even on scenarios where experts are not activated.

Chimera-8B performs similarly in the chart domain but slightly worse in the table domain. This is due to the overspecialized function of the table expert, which, despite sup-

porting comprehensive extraction, may introduces noise for the 8B baseline model because of the considerable task gap. In contrast, the chart expert’s pre-training task covers both extraction and perception, with minimal impact. The math expert consistently improves performance across models due to its alignment with reasoning tasks.

###### 4.3. Comparison on Visual Structural Extraction

Comparison with Generalist Models. For specialized tasks beyond VQA, generalist models similarly show limited performance. Tab. 4 and Tab. 5 present the results of visual structural extraction in the Chart and Table domains, respectively. On ChartQA-SE and PlotQA-SE, Chimera outperforms representative generalist models such as GPT-4V, Qwen-VL [5], and InternVL-2 [12] across the AP@strict, AP@slight, and AP@high metrics. In Table-SE, Chimera leads InternVL2 and Qwen2-VL by a larger margin in Edit Distance and TED scores, demonstrating strong domainspecific capability.

Tab. 6 and Fig. 3 exhibit results on Doc-SE, Chimera significantly outperforms InternVL2 [12] across four metrics in bilingual tasks and shows balanced performance across different document categories.

Comparison with Specialist Models. Compared to specialist models majoring in single task, Chimera still demonstrates strong performance. Specifically, for ChartQA-SE and PlotQA-SE, Chimera achieves excellent or competitive results across three metrics compared to the SOTA expert model GOT. In Table-SE, Chimera also achieves comparable TED scores and outperforms with a lower Edit Distance

Task Metric Deplot∗ [42] UniChart∗ [53] ChartVLM∗ [84] GPT-4V Qwen-VL [5] GOT [80] InternVL-2 [12] Chimera

AP@strict 61.4 42.3 71.8 50.4 58.6 74.7 73.7 74.1 AP@slight 70.9 53.1 81.4 60.6 68.5 84.5 83.9 84.4

ChartQA-SE

AP@high 72.9 56.0 84.2 64.3 72.7 86.7 87.2 87.6

AP@strict 3.1 10.5 3.8 7.3 0.5 13.3 5.7 5.9 AP@slight 16.5 26.0 46.8 19.4 4.2 59.6 55.0 62.1

PlotQA-SE

AP@high 26.5 26.9 54.0 22.3 12.0 64.0 61.8 71.0

- Table 4. Performance on ChartQA-SE and PlotQA-SE. Metrics include Average Precision (AP) at strict, slight, and high levels. ∗ represents the domain expert model.

Method Edit Distance↓ TEDS↑ TEDS (structure only)↑

InternVL-2 [12] 0.229 0.676 0.762 Qwen2-VL [77] 0.231 0.690 0.773

StructEqTable∗ [83] 0.226 0.706 0.787 GOT∗ [80] 0.257 0.745 0.830 Chimera 0.165 0.740 0.828

- Table 5. Comparison of performance on Table-SE across different methods: TEDS, TEDS (structure only), and Edit Distance. ∗ represents the domain expert model.

Method

Edit Distance↓ Precision↑ BLEU↑ METEOR↑ en zh en zh en zh en zh

InternVL [12] 0.504 0.604 65.4 66.0 38.4 33.1 52.6 50.6 GOT∗ [80] 0.355 0.510 67.9 71.2 52.5 34.3 65.3 53.9 Chimera 0.304 0.461 69.6 66.1 49.8 40.5 64.8 56.9

- Table 6. Comparison of performance metrics across different methods on Doc-SE. Metrics include Edit Distance (lower is better), Precision, BLEU, and METEOR (higher is better). ∗ represents the domain expert model.

GT \Prediction General Table Chart Math

General – 0 16 6 Table 1 – 0 0 Chart 1 0 – 0 Math 22 0 0 –

Table 7. Error statistics of router on MathVista testmini.

[Figure 61]

[Figure 62]

General visual tokens Expert visual tokens

[Figure 63]

[Figure 64]

Figure 4. Attention distribution on Chimera with (Top) & without (Bottom) masking.

visual tokens Hv and expert visual tokens He. As illustrated in Fig. 4, Chimera places greater emphasis on domain features during inference, demonstrating improved alignment between the specialist models and the generalist LMM. In contrast, Chimera w/o masking exhibits significantly weaker utilization of domain features. This occurs because the general encoder, which is already well-aligned with the language model, often results in imbalanced optimization between the general encoder and expert models.

Analysis about Expert Router Error. Due to the significant differences in visual inputs across different domains, Chimera can achieve effective classification using a simple linear layer, thereby guiding expert model selection. As shown in Tab. 7, it achieves 95.4% accuracy on MathVista, proving the router’s effectiveness. All errors stem from confusion between the general and expert domains, not between experts. This issue arises because training labels are dataset-based, and the “general” category includes mixed-domain images. Explicit domain annotation for each image could better address this.

| | | |
|---|---|---|
| | | |

- Figure 3. Comparison of Edit Distance ↓ across different document categories on Document Structural Extraction (Doc-SE) task.

by 0.092. As for Doc-SE, Chimera leads in most metrics for both English and Chinese documents, showing better overall generalization across document categories than GOT.

- 4.4. Further analysis

###### 5. Conclusion

We present Chimera, a scalable pipeline that integrates specialist models into generalist LMMs, enabling adaptation to specialized tasks. Our approach transforms LMMs, like InternVL-2, into versatile models capable of handling tasks across tables, math, documents, etc. Chimera pioneers new directions for bridging generalist and specialist models.

Effects of GSCM. To assess whether our proposed GSCM enhances alignment with the expert model, we analyze the attention distribution of the model’s output across general

###### Acknowledgements

The research was supported by Shanghai Artificial Intelligence Laboratory, a locally commissioned task from the Shanghai Municipal Government, the Shanghai Municipal Science and Technology Major Project, and Shanghai Rising Star Program (Grant No. 23QD1401000).

This work is supported by National Key Research and Development Program of China (No. 2022ZD0160101), Shanghai Natural Science Foundation (No. 23ZR1402900), Shanghai Municipal Science and Technology Major Project (No.2021SHZDZX0103). The computations in this research were performed using the CFFF platform of Fudan University.

[Figure 65]

###### Chimera: Improving Generalist Model with Domain-Specific Experts

###### Supplementary Material

Due to the eight-page limitation of the main text, we provide more details and visualizations from the following aspects:

- Stage 1

General:

ShareGPT4v [10], ShareGPT4-o [10] Table:

TableX [83] Chart:

ChartQA [52], PlotQA [57],ChartX [84], SimChart [81] Math:

MAVIS-Caption [89]

- Stage 2:

- • Sec. 6: Selection strategy for pre-training tasks and expert models.
- • Sec. 7: Dataset Details in training.
- • Sec. 8: Chimera’s performance on general tasks.
- • Sec. 9: Details about preference optimization.
- • Sec. 10: Experiment results on mask ratio selection.
- • Sec. 11: Introduction of Table-SE and Doc-SE.
- • Sec. 12: Experiments of scaling up more experts.
- • Sec. 13: Comparison with existing works.
- • Sec. 14: More information of implementation details.
- • Sec. 15: Visualization of Chimera’s visual content extraction performance.

Language:

Kaggle-science-exam [38], MathInstruct [87],

MathQA [3], SciInstruct [88], Orcamath [58] General:

ShareGPT4v [10], ShareGPT4-o [10], LLaVAR [91],

AI2D (GPT4V) [30], AI2D (InternVL [12]), AI2D (Original) [27],

MathVision [76], IconQA [47], MapQA [8], ScienceQA [63],

ArxivQA [33], TQA [28], CLEVR-Math [21], Super-CLEVR [36],

Cambrian Data Engine [72] Table:

TableX [83], TabMWP [49], MMTab [92] Chart:

###### 6. Pre-training Tasks and Expert Models

PlotQA [57],ChartX [84], SimChart [81], Chart2Text [25],

ChartQA [52], LRV Chart [43], ChartGemma [55], DVQA [23],

###### Low Level High Level

FigureQA [24], VisText [68] Math:

| | | |
|---|---|---|
| | | |
| | | |

Table format transformation

TableChartMathDocument

| |
|---|

MAVIS-Caption [89], Geo170K [17], GeoMVerse [26],

MAVIS Manual Collection [89], MAVIS Data Engine [89]

Geometry3K [46], GeoQA+ [9], InterGPS [46]

Chart structural extraction

Chart type&title prediction

- Table 8. Dataset used for multi-modal reasoning scenario. Stage 1 and Stage 2 represent Domain-General Knowledge Alignment and Visual Instruction Tuning separately.

- Stage 1 ChartQA [52], PlotQA [57],ChartX [84],, SimChart [81], TableX [83]

- Stage 2 DocGenome [83], DocStruct4M [19], DocVQA [56]

- Table 9. Datasets used for visual content extraction scenario.

| | |
|---|---|
| | |

Math caption contrastive learning

| |
|---|

| |
|---|

Document content extraction

| |
|---|

- Stage 1 represents Domain-General Knowledge Alignment, and
- Stage 2 represents Visual Instruction Tuning.

Figure 5. Pre-training tasks of expert models considered by Chimera.

tion caption data. For document structural extraction, we employ the encoder from the latest model, GOT [80].

The type of pre-training task significantly affects model performance, which we consider when selecting expert models. As shown in Fig. 5, we categorize low-level tasks as the precise extraction of domain-specific visual content and structure (e.g., Table2LaTeX, Chart2Markdown, Doc2Markdown), while high-level tasks involve understanding and summarizing image content. We select expert models with diverse pre-training task configurations. For the table expert, we use the encoder from StructEqTable [83], which effectively converts table images into LaTeX/HTML. For the chart expert, we choose the encoder from ChartVLM [84], which excels in structural extraction and chart type classification. For the math expert, we adopt Math-CLIP [89], trained on extensive geometry and func-

###### 7. Dataset Details

The datasets used for Chimera is presented in Tab. 8 and Tab. 9. All the datasets we used come from publicly accessible datasets.

###### 8. Evaluation on General Tasks

We evaluate Chimera’s general capabilities using the perception set from the general benchmark MME [16], with results presented in Tab. 10. Across different model sizes, Chimera and InternVL exhibit varying strengths across different tasks, achieving overall comparable performance.

InternVL2 Chimera

4B 8B 4B 8B

Existence 200.00 190.00 200.00 195.00 Count 123.33 158.33 130.00 155.00 Position 143.33 155.00 123.33 148.33 Color 165.00 175.00 160.00 190.00 Posters 158.84 168.03 159.86 164.97 Celebrity 125.00 148.53 145.29 162.65 Scene 158.75 152.50 163.50 157.75 Landmark 167.25 178.25 167.25 177.75 Artwork 144.75 154.50 144.00 153.00 OCR 147.50 162.50 117.50 132.50 Table 10. Performance on perception sub-tasks of MME.

This suggests that the Chimera framework introduces minimal degradation to the model’s general task capabilities. Meanwhile, Chimera demonstrates strong expertise in domains such as tables, math, charts, and documents, further validating that our proposed approach effectively enhances a generalist LMM’s domain-specific knowledge without compromising its general performance.

###### 9. Details about preference optimization

For preference optimization, we adopt a commonly used approach: we randomly sample 10k problems from MathV360K, generating 16 responses per problem using Chimera. Each response is classified based on correctness using rulebased answer matching, and after filtering, we construct 60k preference pairs for Direct Preference Optimization (DPO) training. Then we perform DPO training on 60K data for 1 epoch.

###### 10. Mask Ratio Selection

Model Ratio ALL General Chart Table Math InternVL2-4B [12] N/A 57.0 50.1 66.2 65.7 58.3 InternVL2-4B-NF [12] N/A 58.5 51.5 67.1 74.3 58.6 Chimera-4B-0.0 0.0 59.4 50.8 66.2 67.1 65.5 Chimera-4B 0.3 61.3 54.0 64.8 72.9 66.9

- Chimera-4B-0.5 0.5 60.4 51.3 68.5 70.0 65.8

- Chimera-4B-1.0 1.0 56.2 51.5 63.5 72.9 53.6

Table 11. Ablation results on different visual content domain on the testmini subset of MathVista. InternVL2-4B-NF represents naive finetune of baseline with same settings, Chimera-4B-R means Chimera model trained with mask ratio R in GSCM.

We conducted an ablation study on 4B scale models to assess our approach’s effectiveness, as shown in Table 11. It should be noted that model with mask ratio 1.0 does not have access to the general encoder during training, contrary to our intentions. Thus, we modified this case to give the model an 80% probability of masking all general features. The results show that naively finetuning the LMM leads to limited performance improvement. By incorporat-

ing domain knowledge from expert models, even the case without GSCM still yields better results than naive finetuning. As the mask ratio increases, the model’s performance improves initially and then declines. This indicates that slightly masking helps balance encoder optimization, leading to better alignment. However, as the mask ratio increase, we believe excessive masking prevents the model from effectively learning to utilize both features for reasoning. Based on the above observations, we set the mask ratio to 0.3 in Chimera’s implementation. We also observed that performance trends vary across domains as the mask ratio changes, suggesting that the alignment difficulty of expert models differs by domain and task, which we leave for future exploration.

###### 11. Details of Table-SE and Doc-SE

In Tables 7 and 8 of the main text, we conduct the experiments on Table Structural Extraction (Table-SE) task and Document Structural Extraction (Doc-SE) task, respectively. In this section, we primarily introduce the evaluation dataset construction method and provide detailed information about the dataset.

###### 11.1. Data Source

###### Count

Document Categories PPT2PDF 43 Academic Literature 42 Book 13 Colorful Textbook 37 Magazine 30 Exam Paper 7 Note 18 Newspaper 15

Language Simplified Chinese 128 English 77

###### Layout

1 and More Column 27 Single Column 91 Other Layout 43 Double Column 40 Three Column 4

# Total 205 Table 12. Statistical information of Doc-SE.

Our benchmark was developed through a systematic sampling process from an initial collection of 200,000 PDF documents sourced from Common Crawl, Google, Baidu search engines, and internal repositories. We initially extracted visual features using ResNet-50 [18] and performed clustering algorithm using Faiss [22] to identify diverse document patterns. From the 10 cluster centers, we sampled 6,000 visually diverse pages, which were then manu-

Count Background

w/o Background 80 w/ Background 20

###### Equation

w/o Equation 78 w/ Equation 22

###### Language

English 45 English & Chinese Mixed 5 Chinese 50

###### Table Format

Three-line Table 47 Full-bordered Table 39 Partial-bordered Table 14 w/o Merged Cells 58 w/ Merged Cells 42

###### Layout

Horizontal 97 Vertical 3

# Total 100 Table 13. Statistical information of Table-SE.

ally annotated with attributes such as page type, layout type, and language. As illustrated in Table 12 and Table 13, the final benchmark includes 205 page-level PDF images and 100 table images, ensuring comprehensive representation of real-world document scenarios with various layouts and attributes.

###### 11.2. Annotation Process

For ensuring annotation quality and efficiency, we design separate standardized processes for page-level PDF documents and tables.

For page-level PDF documents, our process consists of three stages: (1) We first employ fine-tuned LayoutLMv3 for layout detection and PaddleOCR for text recognition as intelligent pre-annotation. (2) Professional annotators then refine the detection boxes, verify text content accuracy, and enhance annotations with reading order and affiliation details. (3) Finally, researchers review the annotations to ensure overall quality and accuracy.

For table annotations, we follow a similar but specialized three-stage approach: (1) We utilize GPT-4o and PaddleOCR for initial table annotations. (2) Annotators then verify and correct the table structure and content, using specialized tools like Tables Generator for verification. (3) Finally, experts through table annotations re-rendering to ensure correct HTML and LaTeX code labels.

###### 11.3. Showcase

We provide several visualization examples of Table-SE in Fig. 6 and Fig. 7, where each item contains a visual table and its corresponding LaTeX code.

###### 12. Experiments of Scaling Up More Experts

Model ALL General Chart Table Math InternVL2-4B 57.0 50.1 66.2 65.7 58.3

InternVL2-4B w/ Chart Expert 59.4 52.0 68.0 72.9 60.8 Chimera-4B 61.3 54.0 64.8 72.9 66.9

Table 14. Accuracy scores of different visual content domain on the testmini subset of MathVista.Those do not belong to the last three domains are uniformly classified as General for simplicity. InternVL2-4B w/ Chart Expert represent the case only integrating chart expert model.

To further validate the impact of scaling up the number of expert models, we provide ablation results introducing only the chart expert. In this case, non-chart data is encoded solely by the general encoder during training. As shown in Table 14, incorporating only the chart expert obtains lower MathVista [48] overall score by 1.9 points than Chimera4B.

Specifically, InternVL2-4B w/ Chart Expert also shows improvements in general scenarios, though less significant than Chimera, which integrates three expert models. In the chart domain, InternVL2-4B w/ Chart Expert achieves notable gains by avoiding conflicts among multiple experts with large task gaps. However, Chimera’s integration of multiple experts enhances performance across diverse domains, boosting overall results. In the math domain, InternVL2-4B w/ Chart Expert scores 6.1 points lower than Chimera, demonstrating the strong mathematical reasoning capabilities derived from integrating the math expert.

###### 13. Comparison with existing works

Integrating specialist experts that contain specialized prior knowledge presents a promising approach to improve the specific capabilities of generalist model. MoVA [94] proposes a multi-turn method that relies on the language model to call an expert in the first round and generates responses in the second, which reduces conciseness and efficiency. MoME [66] uses soft-weighting to fuse multiple visual encoders, enabling VLMs to benefit from leveraging representations from different encoders. However, this approach lacks explicit guidance for encoder selection and introduces additional concerns such as inference efficiency and uniform visual feature sizes.

###### 14. Training Configuration

The training strategy is summarized in Tab 15 and Tab 16. During the two-stage training process, we gradually increase the maximum image resolution and the number of visual tokens of the general visual encoder Eg. In the Domain-General Knowledge Alignment stage, we use

###### Visual Table Ground Truth: LaTeX

\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|l|l|l|} \hline \multirow{2}{*}{} & \multicolumn{4}{c}{Beef meat} & \multicolumn{4}{c}{Chicken meat} & \multicolumn{4}{c}{Pork meat} \\ \cline{2-13} & Cases & Controls & OR & 95\% CI & Cases & Controls & OR & 95\% CI & Cases & Controls & OR & 95\% CI \\ \hline Never Exposed (Ref group) & 1,823 & 2,273 & 1.00 & --- & 1,823 & 2,273 & 1.00 & --- & 1,823 & 2,273 & 1.00 & --- \\ \hline Ever Exposed & 117 & 108 & 1.22 & 0.90-1.67 & 1,36 & 129 & 1.19 &

- 0.91-1.55 & 145 & 143 & 1.09 & 0.83-1.42 \\ \hline Duration of exposure & & & & & & & & & & & & \\ \hline $\leq$5 years & 40 & 37 & 1.45 & 0.92-2.31 & 30 & 40 & 0.97 & 0.60-
- 1.58 & 39 & 41 & 1.25 & 0.80-1.96 \\ \hline 6-15 years & 29 & 43 & 0.79 & 0.47-1.31 & 42 & 41 & 1.21 & 0.78-1.88 & 44 & 58 & 0.84 & 0.55-1.28 \\ \hline $\geq$16 years & 48 & 28 & 1.63 & 0.93-2.88 & 64 & 48 & 1.36 & 0.90-
- 2.06 & 61 & 43 & 1.28 & 0.81-2.03 \\ \hline p-value of test for linear trend (with ref cat) & & & 0.23 & & & & 0.11 & & & & 0.54 & \\ \hline Intensity of exposure & & & & & & & & & & & & \\ \hline Low & 60 & 59 & 1.26 & 0.86-1.83 & 71 & 68 & 1.24 & 0.88-1.75 & 70 & 72 & 1.15 & 0.82-1.62 \\ \hline Medium & 42 & 35 & 1.22 & 0.73-2.04 & 47 & 46 & 1.11 & 0.72-1.71 & 55 & 52 & 1.03 & 0.68-1.58 \\ \hline High & 15 & 14 & 0.91 & 0.35-2.40 & 18 & 15 & 1.22 & 0.56-2.65 & 20 & 19 & 0.89 & 0.40-1.94 \\ \hline p-value of test for linear trend (with ref cat) & & & 0.36 & & & & 0.29 & & & & 0.78 & \\ \hline \end{tabular}

[Figure 66]

[Figure 67]

\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|l|l|} \hline \multicolumn{12}{l}{Table l. Anticonyulsant activity and protective index of intraperitoneal AEDs in mice} \\ \hline

& Rotorod test & \multicolumn{2}{l}{MES test} & \multicolumn{2}{l}{Pentylenetetrazole} & \multicolumn{2}{l}{Bicuculline} & \multicolumn{2}{l}{Picrotoxin} & \multicolumn{2}{l}{Strychnine} \\ \hline

AED& TD_{50}(95\% CI) mg/kg & ED_{50}(95\%CI) mg/kg & PI & ED_{50}(95\%CI) mg/kg & PI & ED_{50}(95\%CI) mg/kg & PI & ED_{50}(95\%CI) mg/kg & PI & ED_{50} & PI \\ \hline Rufinamide & >500<1,000 & 15.5(12.5-18.1) & >32.2 & 54.0(38.1-74.9) & >9.3 & 50.5(23.9-87.8) & >9.9 & 76.3 (64.0-90.7) & >6.6 & 125\textasciicircum{}{a} & NA \\ Phenytoin & 65.5(52.5-72.1) & 9.5(8.1-10.4) & 6.9 & 300 no protection & <0.2 & 100 no protection & <0.7 & 100 no protection & <0.7 & 55100b & <0.7 \\ Phenobarbital & 69.0(62.8-72.9) & 21.8(15.0-25.5) & 3.2 & 13.2(5.9-15.9) & 5.2 & 37.7(26.5-47.4) & 1.8 & 27.5 (20.9-34.8) & 2.5 & 95.3 (91.399.5) & 0.7 \\ Valproate & 425.8(369-450) & 272(247-338) & 1.6 & 148.6(123-177) & 2.9 & 359.9(294-439) & 1.2 & 387.2 (341-444) & 1.1 & 292.9 (261-323) & 1.5 \\ Ethosuximide & 440.8(383-485) & 1,000 no protection & 0.4 & 130.3(111-151) & 3.4 & 459.0(350-633) & 1.0 & 243 (228-255) & 1.8 & 250-1,000c & <0.4 \\

- \multicolumn{12}{l}{a: Maximum protection,37.5\%}\\
- \multicolumn{12}{l}{b: protection, 50.0\%}\\
- \multicolumn{12}{l}{c: Maximum protection, 62.5\%}\\ \multicolumn{12}{l}{AED, antiepileptic drug; MEs, maximal electroshock; TD_{50}, the dose eliciting evidence of minimal neurotoxicityin 50\% of animals; Cl, conidence interva; ED_{50},the dose of drugrequired to produce the desired end point in 50\% of animals; and Pl, protective index (ratio of TD_{50} to ED_{50})}\\ \\ \hline \end{tabular}

- Figure 6. Showcase of Table-SE.

###### Visual Table Ground Truth: LaTeX

\begin{tabular}{|l|l|l|l|l|} \hline 企业类型 & 目的 & 模式和特点 & 优势 & 典型企业 \\ \hline 云服务提供商 & $\cdot$ 以物联网为抓手带动上层应用服务业绩增长

& $\cdot$ 目前多以提供底层计算资源、提供应用使能平台为主 & $\cdot$ 在互联网领域中积累了丰富的技术、商业、生态优势经验 $\cdot$ 底层IaaS能力突出、共性技术能力提炼 & 阿里云、腾讯云、

百度云、亚马逊AWS IoT等 \\ \hline

通信领域厂商 & $\cdot$ 获得流量业务收入，战略布局物联网，把握 新增市场机遇 & $\cdot$ 多以连接管理、应用使能为平台主要功能服 务为主 & $\cdot$ 在连接管理平台具有绝对优势，具有全球通用连接 能力 & 电信运营商、通信设备厂商，中国电信天翼物联、如中国移 动ONENet、中国联通物联网平台、华为云IoT等 \\ \hline

[Figure 68]

软件系统服务商 & $\cdot$ 解决内部开发效率的问题，优化产品服务 & $\cdot$ 以应用开发平台为主要服务内容为主 & $\cdot$ 擅长软件设 计、生产、管理、运维等服务，具备丰富的行业软件开发及服务经 验 & 紫光云、广联达筑联等 \\ \hline

垂直领域传统厂商 & $\cdot$ 利用自身对行业的理解和经验，打造垂 直型平台，实现传统企业的转型升级 & $\cdot$ 垂直专业领域的物联 网平台 & $\cdot$ 深刻的行业理解和行业技术、对行业有深度应用， 拥有行业数据和客户资源 & 西门子、工业富联、美的M-Smart等企 业 \\ \hline

初创企业 & $\cdot$ 看好物联网未来的发展潜能 & $\cdot$ 目前阶段 很多初创型平台企业多以SaaS解决方案公司的形式存在 & $\cdot$ 拥 有与选定细分行业相关的软件、硬件经验 $\cdot$ 服务延伸到通用型 平台厂商难以触及的细分领域，形成错位竞争 & 涂鸦智能、云智易、 机智云、艾拉物联等 \\ \hline

\end{tabular}

[Figure 69]

\begin{tabular}{|l|l|l|l|l|} \hline Site-B & Site-E & DEV & DD & Model \\ \hline

- 3-HCY & 11-D G G & 1.49 & 66.5 & \\ \hline
- 3-HCY & 12-G G S & 1.49 & 66.5 & \\ \hline

- 4-CNY & 1-G G H & 1.29 & 74.8 & d-1 \\ \hline

- 4-CNY & 11-D G G & 1.29 & 74.8 & \\ \hline
- 4-CNY & 12-G G S & 1.29 & 74.8 & \\ \hline

- 5-YNT & 1-G G H & 1.39 & 70.3 & d-2 \\ \hline

- 5-YNT & 11-D G G & 1.39 & 70.3 & \\ \hline
- 5-YNT & 12-G G S & 1.39 & 70.3 & \\ \hline 5-YNT & 18-G G C & 1.39 & 76.6 & \\ \hline 9-NND & 1-G G H & 1.85 & 69.2 & d-3 \\ \hline 9-NND & 12-G G S & 1.85 & 69.2 & \\ \hline 9-NND & 18-G G C & 1.85 & 73.9 & \\ \hline \end{tabular}

- Figure 7. Showcase of Table-SE.

thumbnail images as input for Eg without employing the widely-used Dynamic High Resolution (DHR) technique [12, 30]. In the Visual Instruction Tuning stage, DHR is introduced, allowing up to six times more visual tokens. At this stage, we apply the Generalist-Specialist Collaboration Masking (GSCM) mechanism with a masking ratio of 0.3 to constrain Eg, encouraging the model to leverage domain-specific information from expert models. For train-

able modules, the Domain-General Knowledge Alignment stage updates only the General Projector Pg and Expert Projector Set Se. In subsequent stages, the General Projector Pg, Expert Projector Set Se, and Language Model f are updated.

###### Domain-General Knowledge Alignment Visual Instruction Tuning

General Encoder Eg 448 448×{{1,2,3,4,5,6}×1, 1×{2,3,4,5,6}, 2× {2,3}, 3× 2 }}

Table Encoder Etable N/A N/A Chart Encoder Echart N/A N/A Math Encoder Emath 336 336

Resolution

Vision

General Encoder Eg 256 Max 256×6

Table Encoder Etable 2048 2048 Chart Encoder Echart 2048 2048 Math Encoder Emath 576 576

#Tokens

Total Tokens 256 + {0, 2048, 2048, 576} Max 256×6 + {0, 2048, 2048, 576}

#Samples 1.1M 2.6M

GSCM ratio N/A 0.3

Dynamic High Res [12] False True

Trainable General Projector Pg, Expert Projector Set Se General Projector Pg, Expert Projector Set Se, Language Model f

Batch Size 256/128 128

Training

LR 4e-5/2e-5 2e-5/1e-5 Warm Up 100 steps 0.03 ratio

LR Scheduler Consine Consine Max Length 4096 8192

Weight Decay 0.01 0.01

Epoch 1 1

- Table 15. Detailed configuration for each training stage of Chimera in multi-modal reasoning scenario. The table outlines the progression of vision parameters, dataset characteristics and training hyperparameters. For elements containing “/”, the left side represents configurations used by the 2B and 4B model, while the right side represents configurations used by the 8B model.

Domain-General Knowledge Alignment Visual Instruction Tuning

Vision

Resolution

General Encoder Eg 448 448×{{1,2,3,4,5,6}×1, 1×{2,3,4,5,6}, 2× {2,3}, 3× 2 }}

Document Encoder Edocument 1024 1024

#Tokens

General Encoder Eg 256 Max 256×6

Document Encoder Edocument 256 256

Total Tokens 256 + 256 Max 256×6 + 256

Training

#Samples 995K 275K

GSCM ratio N/A 0.3

Dynamic High Res [12] False True

Trainable General Projector Pg, Expert Projector Set Se General Projector Pg, Expert Projector Set Se, Language Model f

Batch Size 256 128

LR 4e-5 2e-5 Warm Up 100 steps 0.03 ratio

LR Scheduler Consine Consine Max Length 4096 8192

Weight Decay 0.01 0.01

Epoch 1 1

- Table 16. Detailed configuration for each training stage of Chimera in visual content extraction scenario. The table outlines the progression of vision parameters, dataset characteristics and training hyperparameters.

###### 15. Visualization of Chimera on Visual Content Extraction

###### 15.1. Table Format Transformation

We provide the rendered table of the output results of Chimera-8B to show its table format transformation performance. As shown in Fig. 8, Fig. 9 and Fig. 10, Chimera excels in extracting and formatting table content from both Arxiv-style and more diverse table layouts with high accuracy.

###### 15.2. Chart Structural Extraction

We provide the rendered table of the output results of Chimera-8B to show its chart structural extraction performance. As shown in Fig. 11, Fig. 12 and Fig. ??, Chimera can identify and extract information from various types of charts, such as pie charts, line graphs, bar charts, etc., and output this information in a structured format accurately.

## Input: Visual Table Output: LaTeX

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Figure 8. Output of Chimera-8B on Table Format Transformation.

###### 15.3. Document Context Extraction

pabilities on both single-column and double-column documents, effectively extracting structured information endto-end from text-dense visual inputs.

We provide the rendered page of the output results of Chimera to show its document content extraction performance. As shown in Fig. 14, Fig. 15, Fig. 16 and Fig. 17, Chimera demonstrates exceptional content extraction ca-

##### Input: Visual Table Output: LaTeX

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

- Figure 9. Output of Chimera-8B on Table Format Transformation.

#### Input: Visual Table Output: LaTeX

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

- Figure 10. Output of Chimera-8B on Table Format Transformation.

### Input: Visual Chart Output: Markdown

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

###### Input: Visual Chart Output: Markdown

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Figure 12. Output of Chimera-8B on Chart Structural Extraction.

###### Input: Visual Chart Output: Markdown

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

###### Input: Document Page Output: Markdown

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

###### Input: Document Page Output: Markdown

[Figure 110]

[Figure 111]

[Figure 112]

Figure 15. Output of Chimera

on Document Context Extraction.

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

###### References

- [1] The claude 3 model family: Opus, sonnet, haiku. 6
- [2] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Devendra Chaplot, Jessica Chudnovsky, Saurabh Garg, Theophile Gervet, Soham Ghosh, Am´elie H´eliou, Paul Jacob, et al. Pixtral 12b. arXiv preprint arXiv:2410.07073,

2024. 6

- [3] Aida Amini, Saadia Gabriel, Peter Lin, Rik KoncelKedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. arXiv preprint arXiv:1905.13319, 2019. 1
- [4] Anthropic. The claude 3 model family: Opus, sonnet, haiku. https://www.anthropic.com,, 2024. 1
- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 1(2):3, 2023. 7, 8
- [6] Tom B Brown. Language models are few-shot learners. arXiv preprint ArXiv:2005.14165, 2020. 2
- [7] Davide Caffagni, Federico Cocchi, Nicholas Moratelli, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, and Rita Cucchiara. Wiki-llava: Hierarchical retrieval-augmented generation for multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1818–1826, 2024. 1
- [8] Shuaichen Chang, David Palzer, Jialin Li, Eric FoslerLussier, and Ningchuan Xiao. Mapqa: A dataset for question answering on choropleth maps. arXiv preprint arXiv:2211.08545, 2022. 1
- [9] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. arXiv preprint arXiv:2105.14517, 2021. 1
- [10] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 1, 2
- [11] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26428–26438, 2024. 1
- [12] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 1, 2, 5, 6, 7, 8
- [13] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 2

- [14] Hongyuan Dong, Jiawen Li, Bohong Wu, Jiacong Wang, Yuan Zhang, and Haoyuan Guo. Benchmarking and improving detail image caption. arXiv preprint arXiv:2405.19092,

2024. 1

- [15] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, et al. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024. 6, 7
- [16] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 1
- [17] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370, 2023. 2, 3, 1
- [18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 2
- [19] Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl2: High-resolution compressing for ocrfree multi-page document understanding. arXiv preprint arXiv:2409.03420, 2024. 1
- [20] Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanwei Li, Yu Qi, Xinyan Chen, Liuhui Wang, Jianhan Jin, Claire Guo, Shen Yan, et al. Mme-cot: Benchmarking chain-of-thought in large multimodal models for reasoning quality, robustness, and efficiency. arXiv preprint arXiv:2502.09621, 2025. 2
- [21] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017. 1
- [22] Jeff Johnson, Matthijs Douze, and Herv´e J´egou. Billionscale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535–547, 2019. 2
- [23] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5648–5656,

2018. 1

- [24] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Akos´ K´ad´ar, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300, 2017. 1
- [25] Shankar Kantharaj, Rixie Tiffany Ko Leong, Xiang Lin, Ahmed Masry, Megh Thakkar, Enamul Hoque, and Shafiq Joty. Chart-to-text: A large-scale benchmark for chart summarization. arXiv preprint arXiv:2203.06486, 2022. 1
- [26] Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. Geomverse: A systematic evalu-

- ation of large models for geometric reasoning. arXiv preprint arXiv:2312.12241, 2023. 1
- [27] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–

251. Springer, 2016. 1

- [28] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In Proceedings of the IEEE Conference on Computer Vision and Pattern recognition, pages 4999–5007, 2017. 1
- [29] Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. Llava-next: Stronger llms supercharge multimodal capabilities in the wild, 2024. 7
- [30] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2, 5, 6, 1
- [31] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022. 2
- [32] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 2

- [33] Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models. arXiv preprint arXiv:2403.00231, 2024. 1
- [34] Mingsheng Li, Xin Chen, Chi Zhang, Sijin Chen, Hongyuan Zhu, Fukun Yin, Gang Yu, and Tao Chen. M3dbench: Let’s instruct large models with multi-modal 3d prompts. arXiv preprint arXiv:2312.10763, 2023. 2
- [35] Qingyun Li, Zhe Chen, Weiyun Wang, Wenhai Wang, Shenglong Ye, Zhenjiang Jin, Guanzhou Chen, Yinan He, Zhangwei Gao, Erfei Cui, et al. Omnicorpus: An unified multimodal corpus of 10 billion-level images interleaved with text. arXiv preprint arXiv:2406.08418, 2024. 2
- [36] Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14963–14973, 2023. 1
- [37] Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. Advances in Neural Information Processing Systems, 35:17612–17625, 2022. 2
- [38] Will Lifferth, Walter Reade, and Addison Howard. Kaggle - llm science exam. https://kaggle.com/

competitions / kaggle - llm - science - exam,

2023. Kaggle. 1

- [39] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 2, 3
- [40] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023. 1, 2, 6, 7
- [41] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-toimage alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024. 1
- [42] Fangyu Liu, Julian Martin Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, and Yasemin Altun. Deplot: One-shot visual language reasoning by plot-to-table translation, 2022. 8
- [43] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning large multi-modal model with robust instruction tuning. arXiv preprint arXiv:2306.14565, 2023. 1
- [44] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 3
- [45] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 3
- [46] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In The 59th Annual Meeting of the Association for Computational Linguistics (ACL), 2021. 1
- [47] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214, 2021. 1
- [48] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 2, 5, 3
- [49] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, SongChun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In International Conference on Learning Representations (ICLR), 2023. 1
- [50] Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Han-Jia Ye. Ovis: Structural embedding alignment for multimodal large language model. arXiv:2405.20797, 2024. 6

- [51] Yiting Lu, Jiakang Yuan, Zhen Li, Shitian Zhao, Qi Qin, Xinyue Li, Le Zhuo, Licheng Wen, Dongyang Liu, Yuewen Cao, et al. Omnicaptioner: One captioner to rule them all. arXiv preprint arXiv:2504.07089, 2025. 2
- [52] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 5, 1
- [53] Ahmed Masry, Parsa Kavehzadeh, Xuan Long Do, Enamul Hoque, and Shafiq Joty. Unichart: A universal visionlanguage pretrained model for chart comprehension and reasoning. arXiv preprint arXiv:2305.14761, 2023. 8
- [54] Ahmed Masry, Mehrad Shahmohammadi, Md Rizwan Parvez, Enamul Hoque, and Shafiq Joty. Chartinstruct: Instruction tuning for chart comprehension and reasoning. arXiv preprint arXiv:2403.09028, 2024. 3
- [55] Ahmed Masry, Megh Thakkar, Aayush Bajaj, Aaryaman Kartha, Enamul Hoque, and Shafiq Joty. Chartgemma: Visual instruction-tuning for chart reasoning in the wild. arXiv preprint arXiv:2407.04172, 2024. 3, 1
- [56] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 1
- [57] Nitesh Methani, Pritha Ganguly, Mitesh M Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1527–1536, 2020. 5, 1
- [58] Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. Orca-math: Unlocking the potential of slms in grade school math, 2024. 1
- [59] OpenAI. Gpt-4v. https://openai.com/index/ gpt-4v-system-card/, 2023. 1, 2
- [60] OpenAI. Hello gpt-4o. https://openai.com/ index/hello-gpt-4o/, 2024. 1, 2
- [61] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. Advances in neural information processing systems, 24, 2011. 2
- [62] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1, 2
- [63] Tanik Saikh, Tirthankar Ghosal, Amish Mittal, Asif Ekbal, and Pushpak Bhattacharyya. Scienceqa: A novel resource for question answering on scholarly articles. International Journal on Digital Libraries, 23(3):289–301, 2022. 1
- [64] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 2
- [65] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for

- Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 2
- [66] Leyang Shen, Gongwei Chen, Rui Shao, Weili Guan, and Liqiang Nie. Mome: Mixture of multimodal experts for generalist multimodal large language models. arXiv preprint arXiv:2407.12709, 2024. 2, 3
- [67] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Mathllava: Bootstrapping mathematical reasoning for multimodal large language models. arXiv preprint arXiv:2406.17294,

2024. 2, 3, 6, 7

- [68] Benny J Tang, Angie Boggust, and Arvind Satyanarayan. Vistext: A benchmark for semantically rich chart captioning. arXiv preprint arXiv:2307.05356, 2023. 1
- [69] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 7
- [70] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 6
- [71] NovelSeek Team, Bo Zhang, Shiyang Feng, Xiangchao Yan, Jiakang Yuan, Zhiyin Yu, Xiaohan He, Songtao Huang, Shaowei Hou, Zheng Nie, et al. Novelseek: When agent becomes the scientist–building closed-loop system from hypothesis to verification. arXiv preprint arXiv:2505.16938,

2025. 3

- [72] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 1
- [73] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2
- [74] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 2
- [75] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839, 2024. 2, 3
- [76] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset, 2024. 1
- [77] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2, 5, 6, 8

- [78] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1
- [79] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, and Jifeng Dai. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442,

2024. 6

- [80] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv preprint arXiv:2409.01704,

2024. 2, 3, 5, 8, 1

- [81] Renqiu Xia, Bo Zhang, Haoyang Peng, Hancheng Ye, Xiangchao Yan, Peng Ye, Botian Shi, Yu Qiao, and Junchi Yan. Structchart: Perception, structuring, reasoning for visual chart understanding. arXiv preprint arXiv:2309.11268,

2023. 5, 1

- [82] Renqiu Xia, Mingsheng Li, Hancheng Ye, Wenjie Wu, Hongbin Zhou, Jiakang Yuan, Tianshuo Peng, Xinyu Cai, Xiangchao Yan, Bin Wang, et al. Geox: Geometric problem solving through unified formalized vision-language pretraining. arXiv preprint arXiv:2412.11863, 2024. 3, 6
- [83] Renqiu Xia, Song Mao, Xiangchao Yan, Hongbin Zhou, Bo Zhang, Haoyang Peng, Jiahao Pi, Daocheng Fu, Wenjie Wu, Hancheng Ye, et al. Docgenome: An open largescale scientific document benchmark for training and testing multi-modal large language models. arXiv preprint arXiv:2406.11633, 2024. 2, 3, 5, 8, 1
- [84] Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Min Dou, Botian Shi, Junchi Yan, et al. Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning. arXiv preprint arXiv:2402.12185, 2024. 1, 2, 3, 5, 8
- [85] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13040–13051, 2024. 1, 2
- [86] Jiakang Yuan, Tianshuo Peng, Yilei Jiang, Yiting Lu, Renrui Zhang, Kaituo Feng, Chaoyou Fu, Tao Chen, Lei Bai, Bo Zhang, et al. Mme-reasoning: A comprehensive benchmark for logical reasoning in mllms. arXiv preprint arXiv:2505.21327, 2025. 3
- [87] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653, 2023. 1
- [88] Dan Zhang, Ziniu Hu, Sining Zhoubian, Zhengxiao Du, Kaiyu Yang, Zihan Wang, Yisong Yue, Yuxiao Dong, and Jie Tang. Sciglm: Training scientific language models with selfreflective instruction annotation and tuning. arXiv preprint arXiv:2401.07950, 2024. 1

- [89] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al. Mavis: Mathematical visual instruction tuning. arXiv preprint arXiv:2407.08739, 2024. 2, 3, 5, 6, 7, 1
- [90] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2025. 2, 3, 5
- [91] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding. arXiv preprint arXiv:2306.17107, 2023. 1
- [92] Mingyu Zheng, Xinwei Feng, Qingyi Si, Qiaoqiao She, Zheng Lin, Wenbin Jiang, and Weiping Wang. Multimodal table understanding. arXiv preprint arXiv:2406.08100,

2024. 2, 3, 1

- [93] Wenwen Zhuang, Xin Huang, Xiantao Zhang, and Jin Zeng. Math-puma: Progressive upward multimodal alignment to enhance mathematical reasoning. arXiv preprint arXiv:2408.08640, 2024. 2, 6, 7
- [94] Zhuofan Zong, Bingqi Ma, Dazhong Shen, Guanglu Song, Hao Shao, Dongzhi Jiang, Hongsheng Li, and Yu Liu. Mova: Adapting mixture of vision experts to multimodal context. arXiv preprint arXiv:2404.13046, 2024. 2, 3

