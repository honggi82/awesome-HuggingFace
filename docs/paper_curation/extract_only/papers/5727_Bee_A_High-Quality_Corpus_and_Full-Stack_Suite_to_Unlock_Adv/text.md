# arXiv:2510.13795v4[cs.CV]8Mar2026

[Figure 1]

## BEE: A HIGH-QUALITY CORPUS AND FULL-STACK SUITE TO UNLOCK ADVANCED FULLY OPEN MLLMS

### Yi Zhang1,2, Bolin Ni2, Xin-Sheng Chen1, Heng-Rui Zhang1, Yongming Rao2, Houwen Peng2∗, Qinglin Lu2, Han Hu2, Meng-Hao Guo1∗ , Shi-Min Hu1† 1Tsinghua University, 2Tencent Hunyuan Team

: https://open-bee.github.io

ABSTRACT

Fully open multimodal large language models (MLLMs) currently lag behind proprietary counterparts, primarily due to a significant gap in data quality for supervised fine-tuning (SFT). Existing open-source datasets are often plagued by widespread noise and a critical deficit in complex reasoning data, such as Chain-of-Thought (CoT), which hinders the development of advanced model capabilities. Addressing these challenges, our work makes three primary contributions. First, we introduce Honey-Data-15M, a new SFT dataset comprising approximately 15 million QA pairs, processed through multiple cleaning techniques and enhanced with a novel dual-level (short and long) CoT enrichment strategy. Second, we introduce HoneyPipe, the data curation pipeline, and its underlying framework DataStudio, providing the community with a transparent and adaptable methodology for data curation that moves beyond static dataset releases. Finally, to validate our dataset and pipeline, we train Bee-8B, an 8B model on Honey-Data15M. Experiments show that Bee-8B establishes a new state-of-the-art (SOTA) for fully open MLLMs, achieving performance that is competitive with, and in some cases surpasses, recent semi-open models such as InternVL3.5-8B. A comprehensive ablation study further dissects the impact of our data curation process, revealing that each stage provides significant performance gains across a wide range of benchmarks. Our work delivers to the community a suite of foundational resources, including: the Honey-Data-15M corpus; the full-stack suite comprising HoneyPipe and DataStudio; training recipes; an evaluation harness; and the model weights. This effort demonstrates that a principled focus on data quality is a key pathway to developing fully open MLLMs that are highly competitive with their semi-open counterparts.

1 INTRODUCTION

Massive datasets have been a cornerstone of the success seen in today’s powerful multimodal large language models (MLLMs) (Li et al., 2025a; Liu et al., 2023c; Wang et al., 2024b; Hurst et al., 2024; Guo et al., 2025b). However, as the field matures, a new consensus is emerging that data quality is as critical as data quantity, especially for the supervised fine-tuning (SFT) stage (Li et al., 2024d; Wang et al., 2023b). The success of top-tier proprietary models such as Gemini 2.5 (Comanici et al., 2025) and GPT-5 (OpenAI, 2025) relies heavily on highly curated and refined SFT datasets. Such private resources pose a formidable barrier to entry—one that has split the MLLMs field into a distinctly tiered structure: (1) top proprietary models (Guo et al., 2025b; Jaech et al., 2024; Comanici et al., 2025), (2) semi-open MLLMs that release weights but keep data private (Yue et al., 2025b; Bai et al., 2025), and (3) the fully open MLLMs (Li et al., 2025a; Guo et al., 2025c; Deitke et al., 2025), which lag behind the former two. For the fully open community, competing on sheer data volume is not an appropriate strategy. Therefore, focusing on data quality is the most viable path forward.

∗Project lead. Contact: henryllpeng@tencent.com, gmh@tsinghua.edu.cn †Corresponding author. Contact: shimin@tsinghua.edu.cn

However, existing open-source SFT datasets suffer from fundamental quality issues that hinder model development. The first of these is widespread data noise (Chen et al., 2024d; Guo et al., 2025c). Existing datasets are frequently contaminated with not only content-level issues such as factual inaccuracies and image-instruction mismatches, but also structural and format-level flaws, such as excessive text repetition, incorrect tags in instructions, and low-quality images with improper sizes or aspect ratios. During training, this noisy data misleads the model into learning spurious correlations, which systematically undermines its core capabilities. This manifests not only as factual hallucinations (Bai et al., 2024) but also as degraded reasoning, poor instruction-following ability, and a tendency to generate low-quality responses. Another critical issue is the gap in complex problem-solving abilities. The capacity of a model for complex reasoning has become a key determinant of its overall capability and a primary manifestation of its advanced intelligence (Kahneman, 2011; Guo et al., 2025e;d; Jaech et al., 2024; Guo et al., 2025a; Yue et al., 2025b). This is clearly demonstrated by leading proprietary and semi-open models, which excel at handling complex instructions, often leveraging long Chain-of-Thought (CoT) processes. The fully open community, however, lags significantly in this area. This weakness is rooted in data deficiencies: the community not only lacks large-scale, high-quality long CoT datasets, but it also remains difficult to identify which instructions actually require these deep, multistep reasoning paths (Kojima et al., 2022). This data deficit represents the primary bottleneck preventing fully open MLLMs from developing the advanced problem-solving skills necessary to compete.

The root of these data quality issues lies not only in the raw data but also in the lack of transparent and reproducible data curation pipelines. Proprietary models benefit from sophisticated, undisclosed recipes for data filtering and cleaning (Guo et al., 2025b; Yue et al., 2025b). A similar opacity exists within the open-source community. Previous work has focused on sharing the final dataset rather than the methodologies behind its creation. Pioneering projects (Li et al., 2025a; Guo et al., 2025c), despite their contributions, often release static datasets. Their curation pipelines—the code, prompts, and filtering logic remain opaque black boxes. This one-off release model is critically flawed. Proprietary efforts constantly refine their internal data recipes. To truly compete, the opensource community needs access to evolving methodologies. The absence of shared, adaptable data curation methods is a fundamental roadblock.

To address these multifaceted challenges, we introduce Honey-Data-15M, a large-scale, highquality SFT dataset designed to serve as a new cornerstone resource for the fully open MLLM community. Its construction is guided by two core principles. First, we conducted a comprehensive data refinement process, collecting from diverse projects and systematically cleaning the corpus to purge widespread data noise, thereby significantly enhancing overall data quality and reliability. Second, we implemented a nuanced, dual-level CoT enrichment strategy that tailors response depth to instruction complexity. For instructions requiring moderate reasoning, we constructed short CoT responses, creating a massive corpus of 12.1M instruction-response pairs. For the most complex instructions, we generated detailed long CoT responses, yielding a high-quality set of 2.9M pairs. This targeted, dual-level approach provides tailored reasoning depth across the entire dataset and inherently solves the critical challenge of identifying which instructions warrant more elaborate, multi-step solutions. Honey-Data-15M was created using our data pipeline, HoneyPipe, which is an instantiation of our self-developed data curation framework, DataStudio. This pipeline leverages MLLMs to automate the entire curation workflow, from cleaning to enrichment. As a scalable and economical alternative to costly human annotation, this model-driven process makes high-quality data construction feasible for the open-source community.

To validate the effectiveness of our curated Honey-Data-15M, we also contribute a new model to the fully open MLLM ecosystem. Experiments with our final model, Bee-8B, trained on the full Honey-Data-15M dataset, establish a new state-of-the-art (SOTA) among fully open MLLMs. Its performance is highly competitive, standing on par with, and in some cases surpassing, several recent semi-open models such as InternVL3.5-8B (Wang et al., 2025b). This significant leap is directly attributable to our data curation strategy. A comprehensive ablation study quantifies this impact, showing that our curation process yields significant improvements across multiple benchmarks compared to using the original, unprocessed data. These results confirm that our focus on data quality is a crucial strategy for closing the performance gap between fully open MLLMs and recent semi-open models.

In summary, our key contributions are threefold:

[Figure 2]

[Figure 3]

[Figure 4]

|?|
|---|

General Chart STEM

- 1. Is the picture relevant to the question?
- 2. Is there any grammatical error in the question…….

[Figure 5]

###### Q: What is the measure of angle GJH?

Q: AB is the diameter of circle O, and ∠D is 40°. What is the degree measure of ∠CAB? (A) 20° (B) 40° (C) 50° (D) 70°

First think, then Step By Step Answer….

Rule-Based

Language Caption

Grounding &Counting

OCR

A: To determine the measure of \angle GJH …

[Figure 6]

Document

A: The answer is C

- Step1: ( GJ ) is a radius of …
- Step2 : ( HJ ) is a tangent to… Final Answer: (90^\circ)

Data Classification Noise and Irrelevance Filtering Short CoT Enrichment

[Figure 7]

[Figure 8]

[Figure 9]

- 1. Is the open question the same in meaning?
- 2. Are the values the

</think>

<think> Final Datasets

[Figure 10]

same?

So, answer is

Q: What is the main functional group in this molecule? A: <think> So, let’s analyze the molecule. First, identify functional groups. Let’s list the groups…. Wait…. Let’s confirm. Yes, the carboxylic acid (-COOH) is a functional group here, and that’s likely the main one. </think> carboxylic acid (-COOH).

Data Aggregation and Deduplication

Long CoT Enrichment Loop Fidelity Verification

Figure 1: Overview of the HoneyPipe. After initial data preparation and filtering, the foundational path generates short CoT responses, which are then checked by Fidelity Verification. Complex instructions that fail this check are routed to the long CoT Enrichment loop. Here, a more powerful model generates a detailed long CoT response, which then undergoes the same Fidelity Verification. This dual-level architecture systematically builds the Honey-Data-15M.

- • Honey-Data-15M: A dataset with 15M QA pairs, systematically cleaned of noise and enriched with a dual-level CoT reasoning, serving as a new cornerstone for the open community.
- • HoneyPipe: The data curation pipeline and its underlying framework, DataStudio, offering the community a transparent and adaptable methodology that moves beyond static dataset releases.
- • Bee-8B: An 8B model trained on Honey-Data-15M that achieves SOTA among fully open models and competes with semi-open counterparts, validating data quality and pipeline effectiveness.

- 2 HONEY-DATA-15M AND HONEYPIPE

To construct our Honey-Data-15M dataset, we introduce the data pipeline, HoneyPipe, as shown in Fig. 1. It is an automated and reproducible workflow designed to systematically address the twin challenges of widespread data noise and the gap in complex reasoning abilities identified as critical roadblocks for the open-source community. The entire pipeline is constructed from the modular components of our DataStudio. Its architecture features a nuanced, dual-level reasoning enrichment strategy, comprising a foundational path for large-scale short CoT enrichment and a specialized loop for generating long CoT responses to the most complex instructions. This structure allows us to transform a vast, raw data pool into a high-quality, dual-level CoT supervised fine-tuning (SFT) dataset. By offering the community a transparent and adaptable methodology, we move beyond static dataset releases. Each stage of this pipeline is detailed below.

- 2.1 STAGE 1: DATA AGGREGATION AND PREPARATION

The pipeline begins by assembling an initial pool of approximately 24 million image-text pairs from diverse community datasets, including LLaVA-OneVision (Li et al., 2025a), PixMo (Deitke et al., 2025), and MAmmoth-VL (Guo et al., 2025c), etc., where content overlap is a key challenge. To maximize data diversity and enhance processing efficiency, we performed rigorous deduplication at the pair level. Specifically, a sample was considered a duplicate and removed only if both its image (represented by a perceptual hash) and its textual instruction (represented by a simhash) were identical to those of another sample. This critical process significantly reduced redundancy, yielding a clean and unique set of image-instruction-response triplets. Finally, to guide subsequent processing, each sample was assigned a domain label (e.g., General, Chart, OCR, STEM). Instead of performing computationally expensive per-instance classification, we employed an efficient coarse-grained strategy at the data source level. Specifically, for each aggregated data source, we manually in-

spected approximately five representative image-text pairs to determine a single overarching topic label, which was then uniformly assigned to all samples originating from that source.

- 2.2 STAGE 2: NOISE AND IRRELEVANCE FILTERING

This stage is designed to purge the widespread data noise endemic to open-source datasets by integrating both rule-based and model-based filtering operators. The rule-based operators address specific formatting issues, such as removing samples with very small images, extreme aspect ratios, or repeated text in instructions. Orchestrated alongside these is a model-based filtering operator, which leverages the powerful Qwen2.5-VL-72B (Bai et al., 2025) model to ensure image-instruction consistency. This operator is prompted to assess whether an instruction is logical and answerable, and whether it is semantically related to the visual content. For example, it will flag an instruction such as “solve the function problem” as irrelevant to an image containing only oranges.

By integrating these different types of operators within this stage, we effectively pruned flawed samples, producing a clean set of image-instruction pairs ready for enrichment.

- 2.3 STAGE 3: SHORT COT ENRICHMENT AND VERIFICATION

With a clean data foundation, this stage constitutes the foundational path of our dual-level CoT enrichment strategy. It targets instructions that require moderate reasoning by generating explicit, step-by-step CoT explanations. The process begins with data triage before proceeding to two main phases: enrichment and verification.

First, we identify data sources unsuitable for reasoning enhancement, such as samples from computer vision tasks like OCR or object detection. These samples bypass the enrichment process and are added to the final dataset. For all other samples, we proceed as follows:

Short CoT Enrichment. For samples slated for enrichment, we begin by preprocessing their instructions. We remove any head or tail prompts that might discourage detailed reasoning (e.g., “Answer directly”) to elicit the model to produce a comprehensive, step-by-step response. We then use powerful open-source MLLMs (Qwen2.5-VL-72B/32B) to transform the simple, short-form responses into detailed reasoning paths. Notably, we do not add extra system prompts. This is because the models are already adept at generating multi-step responses, and a prompt could constrain their output diversity. This step is the primary source of the approximately 12.1 million short CoT samples in our dataset.

Fidelity Verification. To ensure the fidelity of the generated content, we employ a verification stage that operates on the “LLM-as-a-Judge” (Gu et al., 2024; Zheng et al., 2023). We use a verifier model (Qwen2.5-VL-72B) to perform a semantic comparison between the final conclusion of the newly generated CoT and the original response. The evaluation is twofold: for factual queries (objective questions), the final responses must match precisely; for descriptive or open-ended queries (subjective questions), thematic relevance and semantic consistency are required. Samples that pass this check are added to the final dataset. Samples that fail are not discarded but are instead routed to the long CoT enrichment loop for more specialized enrichment.

- 2.4 STAGE 4: LONG COT ENRICHMENT LOOP

This loop constitutes the second level of the enrichment strategy, designed specifically for the most complex instructions that demand deep, multi-step problem-solving. This secondary path processes three primary types of inputs: 1) the samples that failed the fidelity verification in the previous stage Sec. 2.3; 2) samples from select data sources identified as inherently complex during our initial classification (e.g., VisualWebInstruct (Jia et al., 2025)), for which we proactively generate a long CoT version in addition to their short CoT counterpart; and 3) samples from datasets that have been validated in prior research (e.g., Vision-R1 (Huang et al., 2025)) as being particularly suitable for generating deep reasoning chains.

We leverage the top proprietary MLLMs to generate a more detailed solution. When tasked with the instructions, the model first generates a deep reasoning, often structured with tags like < think >< /think >, before outputting the final response. This specialized reasoning level can handle complex instructions beyond the reach of the initial models.

Following this enrichment, each new long CoT response undergoes the same fidelity verification described in Sec. 2.3. This final validation ensures the correctness of the enriched response. The samples that successfully pass this check constitute the approximately 2.9 million long CoT data points in Honey-Data-15M. We discard any sample that fails this final verification, assuming it is erroneous, unsolvable, or too costly to annotate, even if top proprietary models cannot solve it.

- 2.5 HONEY-DATA-15M: HIGH QUALITY CORPUS WITH THE DUAL-LEVEL COT

The primary output of our pipeline is Honey-Data-15M, a large-scale, multimodal SFT dataset comprising 15 million meticulously curated samples. Our primary objective in developing Honey-Data15M is to furnish the research community with a high-quality, reproducible resource that can serve as a new cornerstone for the fully open MLLM community. The compositional breakdown, illustrated in Fig. 2, reveals a diverse amalgamation of data sources, strategically chosen to cover a wide spectrum of domains and complexities. A more granular description of the source-specific statistics can be found in Fig. 3.

A defining feature of Honey-Data-15M is its enrichment with dual-level CoT reasoning, which forms the backbone of the dataset. We have integrated approximately 12.1 million short CoT samples, designed to instill foundational, step-by-step logical inference. Complementing these are 2.9 million long CoT samples, which challenge models with more intricate, multi-step reasoning problems that require a deeper synthesis of information. This dualpronged approach ensures that models trained on Honey-Data-15M can develop both precision in concise reasoning and depth in complex problem-solving.

###### OCR

###### Grounding & Counting

4.9%

###### Document

5.1%

5.9%

###### General

###### STEM

36.8%

7.6%

###### Honey-Data 15M

15.1%

Caption

24.6%

To forge powerful and well-rounded models, Honey-Data-15M applies its dual-level reasoning across a spectrum of critical domains, such as “General” for foundational visual understanding and “STEM” for symbolic reasoning, thereby ensuring comprehensive skill development.

Chart

Figure 2: Distributions of Honey-Data-15M.

- 3 BEE-8B TRAINING RECIPE

To validate the effectiveness of our Honey-Data-15M, we developed Bee-8B, an 8B multimodal large language model trained with a multi-stage recipe. This training process is intended to showcase the strengths of our Honey-Data-15M dataset, particularly its capacity to foster advanced complex reasoning.

- 3.1 MODEL ARCHITECTURE

Our model, Bee-8B, adopts a proven and effective MLLM architecture, drawing inspiration from leading open-source models like LLaVA-OneVision (Li et al., 2025a). It is built upon the powerful Qwen3-8B (Yang et al., 2025a) large language model, which serves as the foundation for reasoning and text generation. For visual understanding, we employ the open-source SigLIP2-so400mpatch14-384 (Tschannen et al., 2025) as our core vision encoder. To effectively handle images of varying resolutions and preserve fine-grained details, we use the Anyres strategy (Liu et al., 2024b). A simple two-layer MLP with a GELU activation function serves as the projector, mapping these aggregated visual features into the LLM’s token embedding space.

General (36.8%) SVIT-mix-665K(1019.1K)* ALLaVA(721.1K)* LLaVA-NeXT-Data(711.5K) LLaVA-Instruct-300k(470.8K)* Vision FLAN(458.8K)*

idefics375k(406.1K)* SVIT-core-150K(269.4K)* PixMo-CapQA(237.3K)* LVIS-InstructV4(145.4K)* ShareGPT4V(SAM)(122.5K)* Cambrian (Filter)(99.6K)*

PixMo-AskModelAnything(96.3K) ShareGPT4o(73.9K)* NLVR2(54.1K) GQA(45.7K)* PixMo-Point-Explanations(44.9K) LRV Normal(39.5K)*

HQ-Edit(36.4K) ALFWorld(31.9K)* Visual7W(30.7K)* Co-Instruct(30.5K) Cauldron(mulberry)(28.1K)* A-OKVQA(27.0K)*

IconQA(26.5K) VIST(21.1K) KVQA(17.7K) ContrastiveCaption(17.2K) FlintstonesSV(17.0K) InternVL-SA-1B-Caption(16.9K)

IDK(16.6K)* COCO(16.3K)* EST-VQA(13.5K) Birds-to-Words(11.2K) ART500K(11.0K) DreamSim(11.0K)

ScanQA(10.2K) MagicBrush(9.8K) KonIQ-10k(9.4K) Hateful Memes(7.3K)* WebQA(6.6K) nuScenes(6.5K)

Other(5.7K)* Objects365(5.2K) MMChat-Twitter-Post(4.3K) NextQA(3.1K) VSR(2.6K)* New Yorker Caption(2.6K)

ViQuAE(2.4K)* TQA(2.1K) ShareGPT4V(Knowledge)(2.0K)* WildVision(1.9K) Chart (24.6%) TinyChart(1014.1K)* DVQA(473.8K)* UniChart(330.0K) CoSyn(chart)(308.2K)* ArxivQA(287.0K)*

FigureQA(206.8K)* MMTab(165.6K) PlotQA(138.7K)* UReader QA(121.4K)* RobuT WikiSQL(110.9K)* CoSyn(table)(86.5K)*

CoSyn(diagram)(77.1K)* TabMWP(46.3K)* RobuT WTQ(44.2K)* UReader KG(38.2K)* Chart2Text(34.5K)* ChartQA(34.4K)*

MMC-Instruction(27.1K) RobuT SQA(26.6K)* MAVIS-Function(25.3K) CoSyn(graphic)(19.4K)* VisText(15.6K)* SciTSR(9.0K)

MultiHiertt(4.6K)* SimChart9K(4.5K) Other(3.9K)* HiTab(3.6K)* LRV Chart(3.3K)* Infographic(2.2K)* Caption (15.1%) COYO-Recaption(1091.8K) PixMo-Cap(706.9K) WIT(419.9K)* Sherlock(16.9K)* ST-VQA(13.5K)*

Other(1.1K)*

STEM (7.6%) VisualWebInstruct(filtered)(190.4K)* MapQA(142.2K)* VisualWebInstruct(127.1K)* MetaMathQA(110.0K) Geo170K(106.6K)* MAVIS-Metagen(62.5K)* GeoQA+(60.3K)* MAVIS-Geo(43.9K)* CoSyn(math)(37.4K)* AI2D(27.8K)* PMC-VQA(25.1K)*

RAVEN(23.1K)* PathVQA(21.8K)* CoSyn(music)(21.2K)* CoSyn(chemical)(18.2K)* CoSyn(circuit)(17.0K)* MathV360K(TQA)(14.2K)*

ScienceQA(12.6K)* Geometry3K(11.4K)* AI2D(InternVL)(10.6K)* MMChem(10.0K) WebSight(7.6K)* UniGeo(7.6K)*

Other(7.4K)* GeomVerse(6.5K)* AI2D(GPT4V)(5.5K)* VizWiz(5.3K)* VQA-RAD(1.6K)* CMM-Math(1.5K)

InterGPS(1.3K)* Document (5.9%) Ureader Chart(371.0K)* OCR-VQA(107.8K)* CoSyn(document)(101.5K)* ScreenQA(64.9K)* FinTabNet(50.1K)

TextVQA(39.1K)* EATEN(30.0K) DocVQA(23.8K)* LLaVAR GPT4(23.3K)* Docmatix(19.1K) CoSyn(nutrition)(13.0K)*

InfoVQA(9.8K)* UreaderOCR(5.5K)* Other(4.3K)* DocReason(4.0K) InfographicVQA(4.0K)* VisualMRC(3.3K)*

POIE(2.2K) Grounding & Counting (5.1%) CLEVR(504.2K)* TallyQA(91.6K)* VisualGenome(77.4K) IconQA(35.9K)* TQA(20.6K)*

MovieNet(7.4K) CLEVR-Math(7.2K)* Super-CLEVR(6.8K)* MathV360K(VQA-AS)(4.1K)* CLEVR-Change(3.0K) Other(1.7K)*

OCR (4.9%) K12 Printing(211.1K) ArXiv OCR(159.3K) HME(93.6K) VCR-Wiki(77.9K) TextOCR(47.7K) Sroie(31.2K) ICDAR-LSVT-zh(28.2K) ReCTs(17.0K) CTW(13.6K) Rendered Text(10.0K) ICDAR2017(9.9K)

Chrome-Writing(9.2K) MTWI(zh)(8.3K) IAM(5.7K) ICDAR2019(3.4K) Orand-Car-A(2.0K) IIIT 5K(2.0K)

- Figure 3: Data collection of Honey-Data-15M. A detailed breakdown of our dataset’s composition across seven major categories. The number of samples (in thousands) is listed for each source. The

* denotes that the data contains the long CoT response.

- 3.2 MODEL TRAINING

Our training methodology is a five-stage process, detailed in Tab. 1, which progressively builds the model’s capabilities from basic perception to complex reasoning. The overarching goal is to first establish a robust vision-language foundation, then instill deep reasoning abilities using our Honey-Data-15M dataset, and finally polish the model’s outputs for enhanced reliability.

Table 1: Detailed configuration for each training stage of Bee-8B.

Stages Stage 1 Stage 2 Stage 3 Stage 4 Stage 5 Purpose

MLP Warmup

Vision-Language Alignment

Multimodal SFT

Efficient Refinement SFT

Policy Optimization RL (GRPO)

Batch size 512 256 256 256 512 Learning Rate 1e-3 4e-5 5e-5 3e-5,5e-6 2e-6 Dataset Items 1M 14M 15M 1M 50K Packed Sequence Length 8192 16384 16384 16384 Training Epochs 1 1 1 1 Trainable Components MLP All All All All

The training begins with two foundational stages. The first stage consists of an MLP warmup, where only the projector is trained to efficiently align the visual and language feature spaces without altering the pretrained backbones. This is followed by a full-parameter vision-language alignment stage, where the model is trained on a large corpus of image-text pairs and text-only data to build core multimodal capabilities while preserving the LLM’s intrinsic cognitive abilities.

The subsequent three stages focus on advanced instruction tuning and refinement. The pivotal third stage is a large-scale SFT on the entire Honey-Data-15M dataset, designed to instill the complex reasoning patterns from our dual-level CoT data. The fourth stage is an efficient refinement SFT on Honey-Data-1M, a subset of Honey-Data-15M curated for a more rational topic distribution, thereby further polishing the model’s capabilities. Finally, the fifth stage employs Group Relative Policy Optimization (GRPO) to improve response quality and reliability by mitigating common generation issues like text repetition. The efficacy of this final optimization phase hinges on the high-quality model produced by the SFT stages, which in turn validates our curated data. Further details on the data composition, sources, and specific configurations for each stage are provided in Sec. F.

- 4 EXPERIMENTS

To validate the effectiveness of our Honey-Data-15M, we conduct a comprehensive experimental evaluation of Bee-8B, beginning with our evaluation setup (Sec. 4.1). We then benchmark Bee8B against leading fully open and semi-open MLLMs to demonstrate its capabilities (Sec. 4.2) and conclude with ablation studies to validate the high quality of our Honey-Data-15M and quantify the impact of our data curation strategy (Sec. 4.3).

- 4.1 EVALUATION SETUP

We evaluated our model using a customized VLMEvalKit (Duan et al., 2024), which we adapted to enable LLM-based judging on benchmarks like DocVQA. Our model was evaluated with the thinking mode and a maximum response length of 16,384 tokens. For LLM-based judging, we employed Qwen3-32B (Yang et al., 2025a) in a non-thinking mode. Additional details are provided in the Sec. H.

- 4.2 EVALUATION OF MODEL CAPABILITY

As shown in the comprehensive benchmark results in Tab. 2, Bee-8B achieves across-the-board performance improvements over existing fully open models (Li et al., 2025a; Deitke et al., 2025) and stands as a highly competitive alternative to recent semi-open models (Bai et al., 2025; Team et al., 2025c; Wang et al., 2025b). Its most significant advantages are observed in factual accuracy and complex multi-step reasoning, directly reflecting the strengths of our Honey-Data-15M dataset. More results for all benchmarks are available in Sec. A.1. Key results are detailed as follows:

General VQA Tasks: Bee-8B showcases a robust and well-rounded performance across a wide array of general visual question answering tasks. On comprehensive, multi-domain benchmarks such as MMMU (Yue et al., 2024) and MMStar (Chen et al., 2024c), it achieves highly competitive scores of 66.8 and 71.4, respectively, demonstrating its strong general knowledge base. The model’s superior performance becomes particularly apparent on benchmarks that test specific VQA capabilities. For example, on MMMU-Pro (Yue et al., 2025a), a benchmark for professional-level knowledge, Bee-8B obtains a top score of 50.7. This establishes a commanding lead of 3.6% over the nextbest competitor, Qwen2.5-VL-7B (Bai et al., 2025), underscoring its advanced cognitive abilities. Similarly, on CountBench (Paiss et al., 2023), it ranks first with an exceptional score of 93.0. This leading performance extends to other tasks, including a top score of 83.9 on the MMVet (Yu et al.,

- 2024b). Furthermore, it demonstrates superior real-world knowledge by securing the top rank on RealWorldQA (xAI, 2024) with a score of 73.1. This suite of results confirms that Bee-8B not only possesses a broad understanding of general multimodal information but also excels in a variety of core visual skills.

Document, Chart, and OCR Tasks: In tasks involving structured visual content like documents, tables, and charts, Bee-8B demonstrates strong performance. Its skill in chart analysis is evident on ChartQA (Masry et al., 2022), where it achieves a highly competitive score of 86.7, confirming its robust data parsing abilities. This proficiency is most pronounced in the challenging area of scientific document analysis. On the CharXiv benchmark (Wang et al., 2024c), Bee-8B secures the top rank in both descriptive questions (DQ) and reasoning questions (RQ) with scores of 84.8 and 57.3, respectively. For the reasoning task, its score establishes an impressive lead of nearly 12% over the closest competitor, Keye-VL (45.4) (Team et al., 2025c), underscoring its advanced capacity for deep semantic inference. These results validate Bee-8B’s powerful abilities to not only extract precise information but also to deeply comprehend the context of structured visual data.

Math and Reasoning Tasks: The most significant advancements delivered by Bee-8B are in complex math and reasoning, directly validating the effectiveness of our CoT-enriched Honey-Data-15M dataset. It consistently delivers exceptional performance across a suite of benchmarks designed to test quantitative and logical problem-solving. The model’s leadership is most evident on MathVerse (Zhang et al., 2024b), where the RL-tuned version scores a top-ranking 67.0. This represents a clear improvement of 5.5% over the strong semi-open model InternVL3.5-8B (Wang et al., 2025b), showcasing its superior visual-mathematical skills. Furthermore, on LogicVista (Xiao et al., 2024), Bee-8B also achieves the top score of 61.3, surpassing the next-best competitor by 4% and demonstrating its versatile reasoning abilities. Its robustness is further confirmed on DynaMath (Zou et al.,

Table 2: Evaluation of Bee-8B against other MLLMs. We distinguish between fully open (*) and semi-open (†) models. The top and second-best scores for each benchmark are highlighted.

Qwen2.5 -VL-7B†

Keye-VL -8B†

InternVL3.5 -8B†

LLaVA OneVision-7B∗

Molmo -7B-D∗

Bee-8B -SFT∗

Bee-8B -RL∗

Task Benchmark

AI2D 81.4 81.0 84.3 86.7 84.0 83.8 85.3 BLINKval 48.2 49.7 56.4 52.0 59.5 52.5 55.0 CountBench – 84.8 74.1 78.0 – 90.5 93.0 HallusionBenchavg 31.6 46.4 52.9 67.0 54.5 59.8 58.2 MMBench-CNdev – – 81.3 92.0 – 81.2 84.2 MMBench-ENdev 80.8 – 82.1 91.5 – 83.0 85.5 MMMUval 48.8 45.3 58.6 71.4 73.4 66.8 66.1 MMMU-Prostandard 29.5 – 34.7 47.1 – 50.4 50.7 MMStar 61.7 56.1 63.9 75.5 69.3 69.0 71.4 MMT-Benchval 59.3 56.3 63.6 65.9 66.7 64.6 67.0 MMVet 57.5 41.5 67.1 79.0 83.1 83.3 83.9 MMVP – – 73.3 79.0 – 80.7 82.0 POPEavg 88.4 89.0 86.4 86.0 88.7 84.0 84.8 RealWorldQA 66.3 70.7 68.5 67.7 67.5 70.1 73.1 VisuLogic – – 20.0 25.6 – 24.4 26.5 VLMs are Blind 39.2 – 37.4 57.1 – 55.8 56.5

General VQA

CharXivDQ – – 73.9 77.7 72.2 84.7 84.8 CharXivRQ – – 42.5 45.4 44.4 55.3 57.3 ChartQAtest 80.0 84.1 87.3 86.3 86.7 86.7 86.1 DocVQAval – – 95.5 88.5 – 87.2 87.0 InfoVQAval – – 81.4 67.4 – 72.3 72.9 OCRBench 62.2 65.6 86.4 85.1 84.0 83.1 82.5 SEED-Bench2-Plus 65.4 67.6 70.4 69.4 70.8 67.7 68.5

Table & Chart & OCR

DynaMathworst 9.0 – 21.0 37.3 37.7 41.3 40.5 LogicVista 33.3 – 44.1 54.8 57.3 56.8 61.3 MathVersevision_only 26.2 4.2 25.1 59.8 61.5 61.9 67.0 MathVision 18.3 16.2 25.4 46.0 56.8 46.8 50.0 MathVistamini 63.2 51.6 68.2 80.7 78.4 78.6 81.4 WeMath 20.9 – 35.2 60.7 57.0 55.0 59.8

Math & Reasoning

- 2025), where it again secures the top score of 41.3. This consistent pattern of leading performance on challenging reasoning tasks confirms that our high-quality data strategy has successfully instilled the model with complex reasoning capabilities.

In summary, Bee-8B establishes a new performance bar for fully open models, particularly in factual accuracy and complex reasoning, and proves highly competitive with recent semi-open models. These findings confirm our core thesis: a focus on high-quality data curation is critical for creating models that can rival leading semi-open models.

- 4.3 ABLATION STUDY

Ablation for HoneyPipe: To precisely isolate the effects of our data curation pipeline, we conducted an ablation study with three data subsets, visualized in Fig. 4. We established a baseline with Draw (1.2M raw samples). Our main curated set is Dcurated (960K samples after full filtering and short CoT enrichment). To isolate the impact of the enriched responses themselves, we created Dno-CoT, an identical 960K set to Dcurated, but with the new CoT responses replaced by the original, simple responses. The results reveal a clear hierarchy (Dcurated > Dno-CoT > Draw). The significant improvement from Draw to Dno-CoT demonstrates the large combined benefit of noise filtering and our data selection process. More importantly, the performance leap from Dno-CoT to Dcurated clearly demonstrates the direct impact of the CoT enrichment itself. The superior performance, especially in reasoning-heavy benchmarks like MathVista and CharXiv-RQ, is therefore directly attributable to training on detailed, step-by-step rationales. This study provides powerful evidence that data cleaning and CoT enrichment are critical.

Ablation for Honey-Data-1M: To validate the effectiveness of our curated 1M data subset, we conducted an ablation study with results visualized in Fig. 5. We compare the original Qwen2.5-VL7B base model against two variants fine-tuned from our Stage 2 checkpoint: one trained on Random1M (a randomly sampled 1M subset) and another on Honey-Data-1M (our meticulously curated 1M subset). The results clearly demonstrate the superiority of our selection strategy, as the model fine-tuned on Honey-Data-1M outperforms the variant trained on the randomly sampled Random1M. Notably, fine-tuning on just this 1M curated subset was sufficient for our model to surpass the

MMBench-ENdev MMMU-Prostandard

MMBench-CNdev

81

HallusionBenchavg

79

42

OCRBench

76

53

74

37

81

47

MMStar

72

70

76

32

64

42

59

71

InfoVQAval

68

66

27

61

54

37

57

67

52

49

47

MMMUval

43 48 53 59

69 74

9

61

79 83

13

14

27

21

54

DocVQAval

66

20

19

25

32

71

30

60

DynaMathworst

26

77

37

38

32

66

CharXivDQ

42

46

MathVision

72

Draw(1200K)

CharXivRQ

Dno-CoT(960K)

MathVersevision_only

MathVistamini

Dcurated(960K)

- Figure 4: Visual comparison of model performance in our ablation study. The radar chart illustrates the step-by-step impact of our curation pipeline. The performance lift from Draw (baseline) to Dno-CoT shows the benefit of noise filtering, while the larger leap to Dcurated highlights the critical contribution of Chain-of-Thought enrichment, especially in reasoning-heavy domains.

MathVistamini MathVision MathVersevision_only DynaMathworst WeMath

20

40

60

80

Score(%)

68.2

25.4 25.1

21.0

35.2

73.1

34.7

48.5

25.9

39.6

75.0

38.6

33.5

25.8

40.8

75.5

45.1

54.4

29.7

50.6

Math Benchmarks Qwen2.5-VL-7B

Random-1M(N-T) Honey-Data-1M(N-T) Honey-Data-1M(T)

| |
|---|

| |
|---|

MMMUval MMStar HallusionBenchavg MMBench-ENdev MMMU-Prostandard

30

40

50

60

70

80

Score(%)

58.6

63.9

52.9

82.1

34.7

60.2

63.3

50.9

79.0

41.9

61.1

66.3

51.5

80.7

44.8

63.3

66.0

56.9

79.3

47.0

General Benchmarks

OCRBench DocVQAval InfoVQAval CharXivDQ CharXivRQ

40

60

80

100

Score(%)

86.4

95.5

81.4

73.9

42.5

81.1 83.0

59.9

77.2

40.0

80.1

83.4

63.0

80.2

42.9

80.1

84.4

67.9

79.2

47.9

Chart & OCR Benchmarks

Figure 5: Performance comparison of models fine-tuned on different 1M data subsets. The model trained on our curated Honey-Data-1M consistently and outperforms both the Qwen2.5VL-7B base model and the model trained on a Random-1M subset, highlighting the efficiency of our data selection strategy.

original Qwen2.5-VL-7B on nearly half of the evaluated benchmarks. This study provides powerful evidence for both the high quality of our data and the effectiveness of our selection strategy in cultivating advanced reasoning skills.

- 5 RELATED WORK

Multimodal Large Language Models (MLLMs) have demonstrated immense potential across diverse real-world scenarios (Li et al., 2026; Xu et al., 2026; Ning et al., 2026; Tang et al., 2025a; Wei et al., 2025; Tang et al., 2025b). The modern wave of MLLMs began with GPT-4V (OpenAI, 2023), which showed that a general LLM (OpenAI, 2023; Touvron et al., 2023; Bai et al., 2023a) could process visual inputs. In the open-source community, LLaVA (Liu et al., 2023c) established a standard paradigm of a frozen vision encoder (Radford et al., 2021; Tschannen et al., 2025; Zhai et al., 2023), a lightweight connector, and visual instruction tuning, proving that GPT-generated instructions were sufficient for building a practical multimodal assistant. Subsequently, the field’s focus shifted to data, which has produced a significant data gap. Proprietary models (OpenAI, 2025; Comanici et al., 2025; Guo et al., 2025b) leverage massive, ever-evolving private datasets. Semi-open models (Wang et al., 2024b; Yao et al., 2024; Zhang et al., 2025a; Bai et al., 2025; 2023b; Yue et al., 2025b; Team et al., 2025c;b; Yang et al., 2025b; Team et al., 2025a) retain an edge through private data curation, even with released weights. In contrast, fully open models (Liu et al., 2024b; Wang et al., 2025c; Guo et al., 2025c; Deitke et al., 2025; Li et al., 2025a) are constrained in both the scale and quality of their public data. This performance hierarchy stems more from data access and pipelines than from architectural tweaks. Accordingly, our work centers not on novel architectures but on a high-quality, reproducible data curation pipeline designed to narrow this gap for fully open models at a reasonable cost.

- 6 AVAILABILITY

We are committed to open science and the reproducibility of our results. We will publicly release the full suite of resources developed in this work, including:

- • Dataset: The complete Honey-Data-15M corpus.
- • Model: The weights for the final Bee-8B model.
- • Codebase: The source code for our data curation pipeline (HoneyPipe) and the underlying framework (DataStudio). And all scripts, configurations, and the code required to train and evaluate.
- • Training Resources: To support detailed analysis and community research, we will also release the intermediate model checkpoints for each stage of our training recipe, along with the specific training data subsets used for each stage.

- 7 CONCLUSION

In this work, we addressed the critical data quality gap that hinders the progress of open-source MLLMs. We introduced Honey-Data-15M, a 15M-sample SFT dataset systematically cleaned of noise and enriched with a dual-level Chain-of-Thought (CoT) reasoning structure, all constructed using our open-source, model-driven pipeline, HoneyPipe. To validate our approach, we trained Bee-8B, which establishes a new state-of-the-art among fully open models and proves highly competitive with several leading semi-open models. Our experiments reveal standout performance in complex math and reasoning, a direct outcome of our targeted CoT enrichment, with ablation studies confirming the significant impact of our curation process. Ultimately, our work suggests a clear path forward for the community: prioritizing data quality through transparent, reproducible methods is a more effective strategy than competing on data volume. The full-stack suite of our dataset, pipeline, and model aims to facilitate this data-centric approach and empower further innovation within the open-source ecosystem. By open-sourcing not just our dataset but the entire curation methodology, we hope to provide a cornerstone for a new wave of collaborative and competitive open-source MLLM development.

Acknowledgement. This work was supported by Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (No. JYB2025XDXM101), the National Natural Science Foundation of China (project No. 62495060, 623B2057), and the Research Grant of Tsinghua-Tencent Joint Laboratory for Internet Innovation Technology.

REFERENCES

Manoj Acharya, Kushal Kafle, and Christopher Kanan. Tallyqa: Answering complex counting questions. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The ThirtyFirst Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019, pp. 8076–8084. AAAI Press, 2019. doi: 10.1609/AAAI. V33I01.33018076. URL https://doi.org/10.1609/aaai.v33i01.33018076.

Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 19107–19117. IEEE, 2022. doi: 10.1109/CVPR52688.2022.01854. URL https://doi.org/10.1109/ CVPR52688.2022.01854.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. CoRR, abs/2309.16609, 2023a. doi: 10.48550/ARXIV.2309.16609. URL https://doi.org/10.48550/arXiv. 2309.16609.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. CoRR, abs/2308.12966, 2023b. doi: 10.48550/ARXIV.2308.12966. URL https: //doi.org/10.48550/arXiv.2308.12966.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Ming-Hsuan Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. CoRR, abs/2502.13923, 2025. doi: 10.48550/ARXIV.2502.13923. URL https://doi.org/ 10.48550/arXiv.2502.13923.

Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024.

Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, Ido Shahaf, Oren Tropp, Ehud Karpas, Ran Zilberstein, Jiaqi Zeng, Soumye Singhal, Alexander Bukharin, Yian Zhang, Tugrul Konuk, Gerald Shen, Ameya Sunil Mahabaleshwarkar, Bilal Kartal, Yoshi Suhara, Olivier Delalleau, Zijia Chen, Zhilin Wang, David Mosallanezhad, Adi Renduchintala, Haifeng Qian, Dima Rekesh, Fei Jia, Somshubra Majumdar, Vahid Noroozi, Wasi Uddin Ahmad, Sean Narenthiran, Aleksander Ficek, Mehrzad Samadi, Jocelyn Huang, Siddhartha Jain, Igor Gitman, Ivan Moshkov, Wei Du, Shubham Toshniwal, George Armstrong, Branislav Kisacanin, Matvei Novikov, Daria Gitman, Evelina Bakhturina, Jane Polak Scowcroft, John Kamalu, Dan Su, Kezhi Kong, Markus Kliegl, Rabeeh Karimi, Ying Lin, Sanjeev Satheesh, Jupinder Parmar, Pritam Gundecha, Brandon Norick, Joseph Jennings, Shrimai Prabhumoye, Syeda Nahida Akter, Mostofa Patwary, Abhinav Khattar, Deepak Narayanan, Roger Waleffe, Jimmy Zhang, Bor-Yiing Su, Guyue Huang, Terry Kong, Parth Chadha, Sahil Jain, Christine Harvey, Elad Segal, Jining Huang, Sergey Kashirsky, Robert McQueen, Izzy Putterman, George Lam, Arun Venkatesan, Sherry Wu, Vinh Nguyen, Manoj Kilaru, Andrew Wang, Anna Warno, Abhilash Somasamudramath, Sandip Bhaskar, Maka Dong, Nave Assaf, Shahar Mor, Omer Ullman Argov, Scot Junkin, Oleksandr Romanenko, Pedro Larroy, Monika Katariya, Marco Rovinelli, Viji Balas, Nicholas Edelman, Anahita Bhiwandiwalla, Muthu Subramaniam, Smita Ithape, Karthik Ramamoorthy, Yuting Wu, Suguna Varshini Velury, Omri Almog, Joyjit Daw, Denys Fridman, Erick Galinkin, Michael Evans, Katherine Luna, Leon Derczynski, Nikki Pope, Eileen Long, Seth Schneider, Guillermo Siman, Tomasz Grzegorzek, Pablo Ribalta, Monika Katariya, Joey Conway, Trisha Saar, Ann Guan, Krzysztof Pawelec, Shyamala Prayaga, Oleksii Kuchaiev, Boris Ginsburg, Oluwatobi Olabiyi, Kari Briski, Jonathan Cohen, Bryan Catanzaro, Jonah Alben, Yonatan Geifman, Eric Chung, and Chris Alexiuk. Llama-nemotron: Efficient reasoning models, 2025. URL https://arxiv.org/abs/2505.00949.

Ali Furkan Biten, Rubèn Tito, Andrés Mafla, Lluís Gómez, Marçal Rusiñol, Minesh Mathew, C. V. Jawahar, Ernest Valveny, and Dimosthenis Karatzas. ICDAR 2019 competition on scene text visual question answering. In 2019 International Conference on Document Analysis and Recognition, ICDAR 2019, Sydney, Australia, September 20-25, 2019, pp. 1563–1570. IEEE, 2019a. doi: 10.1109/ICDAR.2019.00251. URL https://doi.org/10.1109/ICDAR.2019.00251.

Ali Furkan Biten, Rubèn Tito, Andrés Mafla, Lluís Gómez i Bigorda, Marçal Rusiñol, C. V. Jawahar, Ernest Valveny, and Dimosthenis Karatzas. Scene text visual question answering. In 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019, pp. 4290–4300. IEEE, 2019b. doi: 10.1109/ICCV.2019.00439. URL https://doi.org/10.1109/ICCV.2019.00439.

Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/ coyo-dataset, 2022.

Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pp. 11618–11628. Computer Vision Foundation / IEEE, 2020. doi: 10.1109/CVPR42600.2020.01164. URL https: //openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_ A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html.

Sungguk Cha, Jusung Lee, Younghyun Lee, and Cheoljong Yang. Visually dehallucinative instruction generation: Know what you don’t know. CoRR, abs/2402.09717, 2024. doi: 10.48550/ ARXIV.2402.09717. URL https://doi.org/10.48550/arXiv.2402.09717.

Shuaichen Chang, David Palzer, Jialin Li, Eric Fosler-Lussier, and Ningchuan Xiao. Mapqa: A dataset for question answering on choropleth maps. CoRR, abs/2211.08545, 2022a. doi: 10. 48550/ARXIV.2211.08545. URL https://doi.org/10.48550/arXiv.2211.08545.

Yingshan Chang, Guihong Cao, Mridu Narang, Jianfeng Gao, Hisami Suzuki, and Yonatan Bisk. Webqa: Multihop and multimodal QA. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 16474–16483. IEEE, 2022b. doi: 10.1109/CVPR52688.2022.01600. URL https://doi.org/10.1109/ CVPR52688.2022.01600.

Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for A lite vision-language model. CoRR, abs/2402.11684, 2024a. doi: 10.48550/ARXIV. 2402.11684. URL https://doi.org/10.48550/arXiv.2402.11684.

Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P. Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021, volume ACL/IJCNLP 2021 of Findings of ACL, pp. 513–523. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.FINDINGS-ACL.46. URL https://doi.org/10.18653/ v1/2021.findings-acl.46.

Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pp. 3313–3323. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.218. URL https://doi.org/10.18653/v1/2022. emnlp-main.218.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (eds.), Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XVII, volume 15075 of Lecture Notes in Computer Science, pp. 370–387. Springer, 2024b. doi: 10.1007/978-3-031-72643-9\_22. URL https://doi.org/10. 1007/978-3-031-72643-9_22.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024c. URL http://papers.nips.cc/paper_files/paper/2024/hash/ 2f8ee6a3d766b426d2618e555b5aeb39-Abstract-Conference.html.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. CoRR, abs/2312.14238, 2023. doi: 10.48550/ARXIV.2312.14238. URL https://doi.org/10. 48550/arXiv.2312.14238.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271,

- 2024d.

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Chao Xu, Bin Wang, Xingjian Wei, Wei Li, Wenjian Zhang, Bo Zhang, Pinlong Cai, Licheng Wen, Xiangchao Yan, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. CoRR, abs/2404.16821,

- 2024e. doi: 10.48550/ARXIV.2404.16821. URL https://doi.org/10.48550/arXiv. 2404.16821.

Zhoujun Cheng, Haoyu Dong, Zhiruo Wang, Ran Jia, Jiaqi Guo, Yan Gao, Shi Han, Jian-Guang Lou, and Dongmei Zhang. Hitab: A hierarchical table dataset for question answering and natural language generation. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pp. 1094–1110. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.ACL-LONG.78. URL https://doi.org/10.18653/v1/2022.acl-long.78.

Zewen Chi, Heyan Huang, Heng-Da Xu, Houjin Yu, Wanxuan Yin, and Xianling Mao. Complicated table structure recognition. CoRR, abs/1908.04729, 2019. URL http://arxiv.org/abs/ 1908.04729.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit S. Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, Krishna Haridasan, Ahmed Omran, Nikunj Saunshi, Dara Bahri, Gaurav Mishra, Eric Chu, Toby Boyd, Brad Hekman, Aaron Parisi, Chaoyi Zhang, Kornraphop Kawintiranon, Tania Bedrax-Weiss, Oliver Wang, Ya Xu, Ollie Purkiss, Uri Mendlovic, Ilaï Deutel, Nam Nguyen, Adam Langley, Flip Korn, Lucia Rossazza, Alexandre Ramé, Sagar Waghmare, Helen Miller, Nathan Byrd, Ashrith Sheshan, Raia Hadsell Sangnie Bhardwaj, Pawel Janus, Tero Rissa, Dan Horgan, Sharon Silver, Ayzaan Wahid, Sergey Brin, Yves Raimond, Klemen Kloboves, Cindy Wang, Nitesh Bharadwaj Gundavarapu, Ilia Shumailov, Bo Wang, Mantas Pajarskas, Joe Heyward, Martin Nikoltchev, Maciej Kula, Hao Zhou, Zachary Garrett, Sushant Kafle, Sercan Arik, Ankita Goel, Mingyao Yang, Jiho Park, Koji Kojima, Parsa Mahmoudieh, Koray Kavukcuoglu, Grace Chen, Doug Fritz, Anton Bulyenov, Sudeshna Roy, Dimitris Paparas, Hadar Shemtov, Bo-Juen Chen, Robin Strudel, David Reitter, Aurko Roy, Andrey Vlasov, Changwan Ryu, Chas Leichner, Haichuan Yang, Zelda Mariet, Denis Vnukov, Tim Sohn, Amy Stuart, Wei Liang, Minmin Chen, Praynaa Rawlani, Christy Koh, JD CoReyes, Guangda Lai, Praseem Banzal, Dimitrios Vytiniotis, Jieru Mei, and Mu Cai. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. CoRR, abs/2507.06261, 2025. doi: 10.48550/ARXIV.2507.06261. URL https://doi.org/10.48550/arXiv.2507.06261.

Erfei Cui, Yinan He, Zheng Ma, Zhe Chen, Hao Tian, Weiyun Wang, Kunchang Li, Yi Wang, Wenhai Wang, Xizhou Zhu, Lewei Lu, Tong Lu, Yali Wang, Limin Wang, Yu Qiao, and Jifeng Dai. Sharegpt-4o: Comprehensive multimodal annotations with gpt-4o, 2024. URL https: //sharegpt4o.github.io/.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, Yen-Sung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross B. Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp. 91–104. Computer Vision Foundation / IEEE, 2025. doi: 10.1109/CVPR52734.2025.00018. URL https://openaccess.thecvf.com/content/CVPR2025/html/Deitke_

Molmo_and_PixMo_Open_Weights_and_Open_Data_for_State-of-the-Art_ CVPR_2025_paper.html.

Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. Vlmevalkit: An opensource toolkit for evaluating large multi-modality models. In Jianfei Cai, Mohan S. Kankanhalli, Balakrishnan Prabhakaran, Susanne Boll, Ramanathan Subramanian, Liang Zheng, Vivek K. Singh, Pablo César, Lexing Xie, and Dong Xu (eds.), Proceedings of the 32nd ACM International Conference on Multimedia, MM 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November 2024, pp. 11198–11201. ACM, 2024. doi: 10.1145/3664647.3685520. URL https://doi.org/10.1145/3664647.3685520.

Maxwell Forbes, Christine Kaeser-Chen, Piyush Sharma, and Serge J. Belongie. Neural naturalist: Generating fine-grained image comparisons. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pp. 708–717. Association for Computational Linguistics, 2019. doi: 10.18653/V1/D19-1065. URL https: //doi.org/10.18653/v1/D19-1065.

Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/ hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: multimodal large language models can see but not perceive. In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (eds.), Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XXIII, volume 15081 of Lecture Notes in Computer Science, pp. 148–166. Springer, 2024. doi: 10.1007/978-3-031-73337-6\_9. URL https://doi.org/10.1007/978-3-031-73337-6_9.

Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. G-llava: Solving geometric problem with multi-modal large language model. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=px1674Wp3C.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Yuanzhuo Wang, and Jian Guo. A survey on llm-as-a-judge. CoRR, abs/2411.15594, 2024. doi: 10.48550/ARXIV.2411.15594. URL https://doi.org/ 10.48550/arXiv.2411.15594.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 14375–14385. IEEE, 2024. doi: 10.1109/CVPR52733.2024.01363. URL https://doi.org/10.1109/CVPR52733. 2024.01363.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025a.

Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, Jingji Chen, Jingjia Huang, Kang Lei, Liping Yuan, Lishu Luo,

Pengfei Liu, Qinghao Ye, Rui Qian, Shen Yan, Shixiong Zhao, Shuai Peng, Shuangye Li, Sihang Yuan, Sijin Wu, Tianheng Cheng, Weiwei Liu, Wenqian Wang, Xianhan Zeng, Xiao Liu, Xiaobo Qin, Xiaohan Ding, Xiaojun Xiao, Xiaoying Zhang, Xuanwei Zhang, Xuehan Xiong, Yanghua Peng, Yangrui Chen, Yanwei Li, Yanxu Hu, Yi Lin, Yiyuan Hu, Yiyuan Zhang, Youbin Wu, Yu Li, Yudong Liu, Yue Ling, Yujia Qin, Zanbo Wang, Zhiwu He, Aoxue Zhang, Bairen Yi, Bencheng Liao, Can Huang, Can Zhang, Chaorui Deng, Chaoyi Deng, Cheng Lin, Cheng Yuan, Chenggang Li, Chenhui Gou, Chenwei Lou, Chengzhi Wei, Chundian Liu, Chunyuan Li, Deyao Zhu, Donghong Zhong, Feng Li, Feng Zhang, Gang Wu, Guodong Li, Guohong Xiao, Haibin Lin, Haihua Yang, Haoming Wang, Heng Ji, Hongxiang Hao, Hui Shen, Huixia Li, Jiahao Li, Jialong Wu, Jianhua Zhu, Jianpeng Jiao, Jiashi Feng, Jiaze Chen, Jianhui Duan, Jihao Liu, Jin Zeng, Jingqun Tang, Jingyu Sun, Joya Chen, Jun Long, Junda Feng, Junfeng Zhan, Junjie Fang, Junting Lu, Kai Hua, Kai Liu, Kai Shen, Kaiyuan Zhang, and Ke Shen. Seed1.5vl technical report. CoRR, abs/2505.07062, 2025b. doi: 10.48550/ARXIV.2505.07062. URL https://doi.org/10.48550/arXiv.2505.07062.

He Guo, Xiameng Qin, Jiaming Liu, Junyu Han, Jingtuo Liu, and Errui Ding. EATEN: entity-aware attention for single shot visual text extraction. In 2019 International Conference on Document Analysis and Recognition, ICDAR 2019, Sydney, Australia, September 20-25, 2019, pp. 254– 259. IEEE, 2019. doi: 10.1109/ICDAR.2019.00049. URL https://doi.org/10.1109/ ICDAR.2019.00049.

Jiawei Guo, Tianyu Zheng, Yizhi Li, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Graham Neubig, Wenhu Chen, and Xiang Yue. Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 August 1, 2025, pp. 13869–13920. Association for Computational Linguistics, 2025c. URL https://aclanthology.org/2025.acl-long.680/.

Meng-Hao Guo, Xuanyu Chu, Qianrui Yang, Zhe-Han Mo, Yiqing Shen, Pei-lin Li, Xinjie Lin, Jinnian Zhang, Xin-Sheng Chen, Yi Zhang, Kiyohiro Nakayama, Zhengyang Geng, Houwen Peng, Han Hu, and Shi-Min Hu. Rbench-v: A primary assessment for visual reasoning models with multi-modal outputs. CoRR, abs/2505.16770, 2025d. doi: 10.48550/ARXIV.2505.16770. URL https://doi.org/10.48550/arXiv.2505.16770.

Meng-Hao Guo, Jiajun Xu, Yi Zhang, Jiaxi Song, Haoyang Peng, Yi-Xuan Deng, Xinzhi Dong, Kiyohiro Nakayama, Zhengyang Geng, Chen Wang, Bolin Ni, Guo-Wei Yang, Yongming Rao, Houwen Peng, Han Hu, Gordon Wetzstein, and Shi-Min Hu. R-bench: Graduatelevel multi-disciplinary benchmarks for LLM & MLLM complex reasoning evaluation. CoRR, abs/2505.02018, 2025e. doi: 10.48550/ARXIV.2505.02018. URL https://doi.org/10. 48550/arXiv.2505.02018.

Tanmay Gupta, Dustin Schwenk, Ali Farhadi, Derek Hoiem, and Aniruddha Kembhavi. Imagine this! scripts to compositions to videos. In Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and Yair Weiss (eds.), Computer Vision - ECCV 2018 - 15th European Conference, Munich, Germany, September 8-14, 2018, Proceedings, Part VIII, volume 11212 of Lecture Notes in Computer Science, pp. 610–626. Springer, 2018. doi: 10.1007/978-3-030-01237-3\_37. URL https://doi.org/10.1007/978-3-030-01237-3_37.

Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pp. 3608–3617. Computer Vision Foundation / IEEE Computer Society, 2018. doi: 10.1109/CVPR. 2018.00380. URL http://openaccess.thecvf.com/content_cvpr_2018/html/ Gurari_VizWiz_Grand_Challenge_CVPR_2018_paper.html.

Mengchao He, Yuliang Liu, Zhibo Yang, Sheng Zhang, Canjie Luo, Feiyu Gao, Qi Zheng, Yongpan Wang, Xin Zhang, and Lianwen Jin. ICPR2018 contest on robust reading for multi-type web images. In 24th International Conference on Pattern Recognition, ICPR 2018, Beijing, China, August 20-24, 2018, pp. 7–12. IEEE Computer Society, 2018. doi: 10.1109/ICPR.2018.8546143. URL https://doi.org/10.1109/ICPR.2018.8546143.

Xuehai He, Yichen Zhang, Luntian Mou, Eric P. Xing, and Pengtao Xie. Pathvqa: 30000+ questions for medical visual question answering. CoRR, abs/2003.10286, 2020. URL https://arxiv. org/abs/2003.10286.

Jack Hessel, Jena D. Hwang, Jae Sung Park, Rowan Zellers, Chandra Bhagavatula, Anna Rohrbach, Kate Saenko, and Yejin Choi. The abduction of sherlock holmes: A dataset for visual abductive reasoning. In Shai Avidan, Gabriel J. Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner (eds.), Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XXXVI, volume 13696 of Lecture Notes in Computer Science, pp. 558–575. Springer, 2022. doi: 10.1007/978-3-031-20059-5\_32. URL https: //doi.org/10.1007/978-3-031-20059-5_32.

Jack Hessel, Ana Marasovic, Jena D. Hwang, Lillian Lee, Jeff Da, Rowan Zellers, Robert Mankoff, and Yejin Choi. Do androids laugh at electric sheep? humor “understanding” benchmarks from the new yorker caption contest. In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pp. 688–714. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.ACL-LONG.41. URL https: //doi.org/10.18653/v1/2023.acl-long.41.

Vlad Hosu, Hanhe Lin, Tamás Szirányi, and Dietmar Saupe. Koniq-10k: An ecologically valid database for deep learning of blind image quality assessment. IEEE Trans. Image Process., 29: 4041–4056, 2020. doi: 10.1109/TIP.2020.2967829. URL https://doi.org/10.1109/ TIP.2020.2967829.

Yu-Chung Hsiao, Fedir Zubach, Gilles Baechler, Srinivas Sunkara, Victor Carbune, Jason Lin, Maria Wang, Yun Zhu, and Jindong Chen. Screenqa: Large-scale question-answer pairs over mobile app screenshots. In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.), Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2025 - Volume 1: Long Papers, Albuquerque, New Mexico, USA, April 29 - May 4, 2025, pp. 9427–9452. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.NAACL-LONG.477. URL https://doi.org/10.18653/v1/2025.naacl-long.477.

Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, Florida, USA, November 12-16, 2024, pp. 3096–3120. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.FINDINGS-EMNLP.175. URL https://doi.org/10.18653/ v1/2024.findings-emnlp.175.

Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and Dahua Lin. Movienet: A holistic dataset for movie understanding. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm (eds.), Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part IV, volume 12349 of Lecture Notes in Computer Science, pp. 709–727. Springer, 2020. doi: 10.1007/978-3-030-58548-8\_41. URL https://doi.org/10.1007/ 978-3-030-58548-8_41.

Ting-Hao (Kenneth) Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Aishwarya Agrawal, Jacob Devlin, Ross B. Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, C. Lawrence Zitnick, Devi Parikh, Lucy Vanderwende, Michel Galley, and Margaret Mitchell. Visual storytelling. In Kevin Knight, Ani Nenkova, and Owen Rambow (eds.), NAACL HLT 2016, The 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, San Diego California, USA, June 12-17, 2016, pp. 1233–1239. The Association for Computational Linguistics, 2016. doi: 10.18653/V1/N16-1147. URL https://doi.org/10.18653/v1/n16-1147.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. CoRR, abs/2503.06749, 2025. doi: 10.48550/ARXIV.2503.06749. URL https://doi.org/ 10.48550/arXiv.2503.06749.

Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and C. V. Jawahar. ICDAR2019 competition on scanned receipt OCR and information extraction. CoRR, abs/2103.10213, 2021. URL https://arxiv.org/abs/2103.10213.

Drew A. Hudson and Christopher D. Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pp. 6700– 6709. Computer Vision Foundation / IEEE, 2019. doi: 10.1109/CVPR.2019.00686. URL http://openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_ A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_ CVPR_2019_paper.html.

Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Cihang Xie, and Yuyin Zhou. Hq-edit: A high-quality dataset for instruction-based image editing. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id= mZptYYttFj.

Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander Madry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, Alex Paino, Alex Renzin, Alex Tachard Passos, Alexander Kirillov, Alexi Christakis, Alexis Conneau, Ali Kamali, Allan Jabri, Allison Moyer, Allison Tam, Amadou Crookes, Amin Tootoonchian, Ananya Kumar, Andrea Vallone, Andrej Karpathy, Andrew Braunstein, Andrew Cann, Andrew Codispoti, Andrew Galu, Andrew Kondrich, Andrew Tulloch, Andrey Mishchenko, Angela Baek, Angela Jiang, Antoine Pelisse, Antonia Woodford, Anuj Gosalia, Arka Dhar, Ashley Pantuliano, Avi Nayak, Avital Oliver, Barret Zoph, Behrooz Ghorbani, Ben Leimberger, Ben Rossen, Ben Sokolowsky, Ben Wang, Benjamin Zweig, Beth Hoover, Blake Samic, Bob McGrew, Bobby Spero, Bogo Giertler, Bowen Cheng, Brad Lightcap, Brandon Walkin, Brendan Quinn, Brian Guarraci, Brian Hsu, Bright Kellogg, Brydon Eastman, Camillo Lugaresi, Carroll L. Wainwright, Cary Bassin, Cary Hudson, Casey Chu, Chad Nelson, Chak Li, Chan Jun Shern, Channing Conger, Charlotte Barette, Chelsea Voss, Chen Ding, Cheng Lu, Chong Zhang, Chris Beaumont, Chris Hallacy, Chris Koch, Christian Gibson, Christina Kim, Christine Choi, Christine McLeavey, Christopher Hesse, Claudia Fischer, Clemens Winter, Coley Czarnecki, Colin Jarvis, Colin Wei, Constantin Koumouzelis, and Dane Sherburn. Gpt-4o system card. CoRR, abs/2410.21276, 2024. doi: 10.48550/ARXIV.2410.21276. URL https://doi.org/10.48550/arXiv.2410. 21276.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, and Ilge Akkaya. Openai o1 system card. CoRR, abs/2412.16720, 2024. doi: 10.48550/ARXIV.2412.16720. URL https://doi.org/10.48550/arXiv.2412.16720.

Yiming Jia, Jiachen Li, Xiang Yue, Bo Li, Ping Nie, Kai Zou, and Wenhu Chen. Visualwebinstruct: Scaling up multimodal instruction data through web search. CoRR, abs/2503.10582, 2025.

doi: 10.48550/ARXIV.2503.10582. URL https://doi.org/10.48550/arXiv.2503. 10582.

Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image instruction tuning. Trans. Mach. Learn. Res., 2024, 2024. URL https: //openreview.net/forum?id=skLtdUVaJa.

Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross B. Girshick. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pp. 1988–1997. IEEE Computer Society, 2017. doi: 10.1109/CVPR.2017.215. URL https://doi.org/10.1109/CVPR.2017.215.

Kushal Kafle, Brian L. Price, Scott Cohen, and Christopher Kanan. DVQA: understanding data visualizations via question answering. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pp. 5648– 5656. Computer Vision Foundation / IEEE Computer Society, 2018. doi: 10.1109/CVPR. 2018.00592. URL http://openaccess.thecvf.com/content_cvpr_2018/html/ Kafle_DVQA_Understanding_Data_CVPR_2018_paper.html.

Daniel Kahneman. Thinking, fast and slow. macmillan, 2011.

Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Ákos Kádár, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Workshop Track Proceedings. OpenReview.net, 2018. URL https://openreview. net/forum?id=H1mz0OyDz.

Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. Geomverse: A systematic evaluation of large models for geometric reasoning. CoRR, abs/2312.12241, 2023. doi: 10.48550/ARXIV.2312.12241. URL https://doi.org/10.48550/arXiv. 2312.12241.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Min Joon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling (eds.), Computer Vision - ECCV 2016 - 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part IV, volume 9908 of Lecture Notes in Computer Science, pp. 235–251. Springer, 2016. doi: 10.1007/978-3-319-46493-0\_15. URL https://doi.org/10.1007/978-3-319-46493-0_15.

Aniruddha Kembhavi, Min Joon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pp. 5376–5384. IEEE Computer Society, 2017. doi: 10.1109/CVPR.2017.571. URL https://doi.org/10.1109/CVPR.2017. 571.

Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, and Davide Testuggine. The hateful memes challenge: Detecting hate speech in multimodal memes. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1b84c4cee2b8b3d823b30e2d604b1878-Abstract.html.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 8bb0d291acd4acf06ef112099c16f326-Abstract-Conference.html.

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Li Fei-Fei. Visual genome: Connecting language and vision using crowdsourced dense image annotations. Int. J. Comput. Vis., 123(1):32–73, 2017. doi: 10.1007/S11263-016-0981-7. URL https: //doi.org/10.1007/s11263-016-0981-7.

Jianfeng Kuang, Wei Hua, Dingkang Liang, Mingkun Yang, Deqiang Jiang, Bo Ren, and Xiang Bai. Visual information extraction in the wild: Practical dataset and end-to-end solution. In Gernot A. Fink, Rajiv Jain, Koichi Kise, and Richard Zanibbi (eds.), Document Analysis and Recognition - ICDAR 2023 - 17th International Conference, San José, CA, USA, August 21-26, 2023, Proceedings, Part VI, volume 14192 of Lecture Notes in Computer Science, pp. 36–53. Springer, 2023. doi: 10.1007/978-3-031-41731-3\_3. URL https://doi.org/10.1007/ 978-3-031-41731-3_3.

Jason J Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinically generated visual questions and answers about radiology images. Scientific data, 5(1):1–10, 2018.

Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. OBELICS: an open web-scale filtered dataset of interleaved image-text documents. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/ 2023/hash/e2cfb719f58585f779d0a4f9f07bd618-Abstract-Datasets_ and_Benchmarks.html.

Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. Building and better understanding vision-language models: insights and future directions. CoRR, abs/2408.12637, 2024a. doi: 10.48550/ARXIV.2408.12637. URL https://doi.org/10.48550/arXiv.2408. 12637.

Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024b. URL http://papers.nips.cc/paper_files/paper/2024/hash/ a03037317560b8c5f2fb4b6466d4c439-Abstract-Conference.html.

Hugo Laurençon, Léo Tronchon, and Victor Sanh. Unlocking the conversion of web screenshots into HTML code with the websight dataset. CoRR, abs/2403.09029, 2024c. doi: 10.48550/ARXIV. 2403.09029. URL https://doi.org/10.48550/arXiv.2403.09029.

Paul Lerner, Olivier Ferret, Camille Guinaudeau, Hervé Le Borgne, Romaric Besançon, José G. Moreno, and Jesús Lovón-Melgarejo. Viquae, a dataset for knowledge-based visual question answering about named entities. In Enrique Amigó, Pablo Castells, Julio Gonzalo, Ben Carterette, J. Shane Culpepper, and Gabriella Kazai (eds.), SIGIR ’22: The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, Madrid, Spain, July 11 15, 2022, pp. 3108–3120. ACM, 2022. doi: 10.1145/3477495.3531753. URL https://doi. org/10.1145/3477495.3531753.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. Trans. Mach. Learn. Res., 2025, 2025a. URL https://openreview.net/forum?id= zKv8qULV6n.

Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. CoRR, abs/2404.16790, 2024a. doi: 10.48550/ARXIV.2404.16790. URL https://doi.org/10. 48550/arXiv.2404.16790.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. CoRR, abs/2407.07895, 2024b. doi: 10.48550/ARXIV.2407.07895. URL https://doi.org/10. 48550/arXiv.2407.07895.

Junxian Li, Di Zhang, Xunzhi Wang, Zeying Hao, Jingdi Lei, Qian Tan, Cai Zhou, Wei Liu, Yaotian Yang, Xinrui Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Wei Li, Mao Su, Shufei Zhang, Wanli Ouyang, Yuqiang Li, and Dongzhan Zhou. Chemvlm: Exploring the power of multimodal large language models in chemistry area. In Toby Walsh, Julie Shah, and Zico Kolter (eds.), AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, pp. 415–423. AAAI Press, 2025b. doi: 10.1609/ AAAI.V39I1.32020. URL https://doi.org/10.1609/aaai.v39i1.32020.

Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 14369–14387. Association for Computational Linguistics, 2024c. doi: 10.18653/V1/2024.ACL-LONG.775. URL https://doi.org/10. 18653/v1/2024.acl-long.775.

Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, Jianzong Wang, Tianyi Zhou, and Jing Xiao. From quantity to quality: Boosting LLM performance with self-guided data selection for instruction tuning. In Kevin Duh, Helena Gómez-Adorno, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024, pp. 7602–7635. Association for Computational Linguistics, 2024d. doi: 10.18653/V1/2024.NAACL-LONG.421. URL https://doi.org/ 10.18653/v1/2024.naacl-long.421.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pp. 292–305. Association for Computational Linguistics, 2023a. doi: 10.18653/V1/2023.EMNLP-MAIN.20. URL https://doi.org/10.18653/v1/2023.emnlp-main.20.

Yunheng Li, Hengrui Zhang, Meng-Hao Guo, Wenzhao Gao, Shaoyong Jia, Shaohui Jiao, Qibin Hou, and Ming-Ming Cheng. Towards universal video mllms with attribute-structured and quality-verified instructions. arXiv preprint arXiv:2602.13013, 2026.

Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L. Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 14963–14973. IEEE, 2023b. doi: 10. 1109/CVPR52729.2023.01437. URL https://doi.org/10.1109/CVPR52729.2023. 01437.

Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In David J. Fleet, Tomás Pajdla, Bernt Schiele, and Tinne Tuytelaars (eds.), Computer Vision - ECCV 2014 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V, volume 8693 of Lecture Notes in Computer Science, pp. 740–755. Springer, 2014. doi: 10.1007/ 978-3-319-10602-1\_48. URL https://doi.org/10.1007/978-3-319-10602-1_ 48.

Adam Dahlgren Lindström and Savitha Sam Abraham. Clevr-math: A dataset for compositional language, visual and mathematical reasoning. In Artur S. d’Avila Garcez and Ernesto JiménezRuiz (eds.), Proceedings of the 16th International Workshop on Neural-Symbolic Learning and Reasoning as part of the 2nd International Joint Conference on Learning & Reasoning (IJCLR 2022), Cumberland Lodge, Windsor Great Park, UK, September 28-30, 2022, volume 3212 of

CEUR Workshop Proceedings, pp. 155–170. CEUR-WS.org, 2022. URL https://ceur-ws. org/Vol-3212/paper11.pdf.

Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning. Trans. Assoc. Comput. Linguistics, 11:635–651, 2023a. doi: 10.1162/TACL\_A\_00566. URL https://doi.org/ 10.1162/tacl_a_00566.

Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning large multi-modal model with robust instruction tuning. CoRR, abs/2306.14565, 2023b. doi: 10.48550/ ARXIV.2306.14565. URL https://doi.org/10.48550/arXiv.2306.14565.

Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, and Dong Yu. MMC: advancing multimodal chart understanding with large-scale instruction tuning. In Kevin Duh, Helena Gómez-Adorno, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024, pp. 1287–1310. Association for Computational Linguistics, 2024a. doi: 10.18653/V1/2024.NAACL-LONG.70. URL https://doi.org/10.18653/v1/2024. naacl-long.70.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023c. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 26286–26296. IEEE, 2024b. doi: 10.1109/CVPR52733. 2024.02484. URL https://doi.org/10.1109/CVPR52733.2024.02484.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024c. URL https:// llava-vl.github.io/blog/2024-01-30-llava-next/.

Wentao Liu, Qianjun Pan, Yi Zhang, Zhuo Liu, Ji Wu, Jie Zhou, Aimin Zhou, Qin Chen, Bo Jiang, and Liang He. Cmm-math: A chinese multimodal math dataset to evaluate and enhance the mathematics reasoning of large multimodal models. CoRR, abs/2409.02834, 2024d. doi: 10. 48550/ARXIV.2409.02834. URL https://doi.org/10.48550/arXiv.2409.02834.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (eds.), Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part VI, volume 15064 of Lecture Notes in Computer Science, pp. 216–233. Springer, 2024e. doi: 10.1007/978-3-031-72658-3\_13. URL https://doi.org/10.1007/978-3-031-72658-3_13.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of OCR in large multimodal models. Sci. China Inf. Sci., 67(12), 2024f. doi: 10.1007/S11432-024-4235-6. URL https://doi.org/10.1007/s11432-024-4235-6.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pp. 6774–6786. Association for Computational Linguistics, 2021a. doi: 10.18653/V1/2021.ACL-LONG.528. URL https://doi.org/10. 18653/v1/2021.acl-long.528.

Pan Lu, Liang Qiu, Jiaqi Chen, Tanglin Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021b. URL https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/ hash/d3d9446802a44259755d38e6d163e820-Abstract-round2.html.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 11332b6b6cf4485b84afadb1352d3a9a-Abstract-Conference.html.

Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https: //openreview.net/forum?id=DHyHRBwJUTN.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024a. URL https://openreview.net/forum?id=KUNzEQMWU7.

Yujie Lu, Dongfu Jiang, Wenhu Chen, William Yang Wang, Yejin Choi, and Bill Yuchen Lin. Wildvision: Evaluating vision-language models in the wild with human preferences. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024b. URL http://papers.nips.cc/paper_files/paper/ 2024/hash/563991b5c8b45fe75bea42db738223b2-Abstract-Datasets_ and_Benchmarks_Track.html.

Hui Mao, Ming Cheung, and James She. Deepart: Learning joint representations of visual arts. In Qiong Liu, Rainer Lienhart, Haohong Wang, Sheng-Wei "Kuan-Ta" Chen, Susanne Boll, YiPing Phoebe Chen, Gerald Friedland, Jia Li, and Shuicheng Yan (eds.), Proceedings of the 2017 ACM on Multimedia Conference, MM 2017, Mountain View, CA, USA, October 23-27, 2017, pp. 1183–1191. ACM, 2017. doi: 10.1145/3123266.3123405. URL https://doi.org/10. 1145/3123266.3123405.

Urs-Viktor Marti and Horst Bunke. The iam-database: an english sentence database for offline handwriting recognition. Int. J. Document Anal. Recognit., 5(1):39–46, 2002. doi: 10.1007/ S100320200071. URL https://doi.org/10.1007/s100320200071.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Findings of the Association for Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022, pp. 2263–2279. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.FINDINGS-ACL.177. URL https://doi.org/10.18653/v1/2022.findings-acl.177.

Ahmed Masry, Parsa Kavehzadeh, Do Xuan Long, Enamul Hoque, and Shafiq Joty. Unichart: A universal vision-language pretrained model for chart comprehension and reasoning. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pp. 14662–14684. Association for Computational Linguistics, 2023. doi:

10.18653/V1/2023.EMNLP-MAIN.906. URL https://doi.org/10.18653/v1/2023. emnlp-main.906.

Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for VQA on document images. In IEEE Winter Conference on Applications of Computer Vision, WACV 2021, Waikoloa, HI, USA, January 3-8, 2021, pp. 2199–2208. IEEE, 2021. doi: 10.1109/WACV48630. 2021.00225. URL https://doi.org/10.1109/WACV48630.2021.00225.

Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. Infographicvqa. In IEEE/CVF Winter Conference on Applications of Computer Vision, WACV 2022, Waikoloa, HI, USA, January 3-8, 2022, pp. 2582–2591. IEEE, 2022. doi: 10. 1109/WACV51458.2022.00264. URL https://doi.org/10.1109/WACV51458.2022. 00264.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In IEEE Winter Conference on Applications of Computer Vision, WACV 2020, Snowmass Village, CO, USA, March 1-5, 2020, pp. 1516–1525. IEEE, 2020. doi: 10.1109/WACV45572.2020.9093523. URL https://doi.org/10.1109/WACV45572. 2020.9093523.

Anand Mishra, Karteek Alahari, and C. V. Jawahar. Scene text recognition using higher order language priors. In Richard Bowden, John P. Collomosse, and Krystian Mikolajczyk (eds.), British Machine Vision Conference, BMVC 2012, Surrey, UK, September 3-7, 2012, pp. 1–11. BMVA Press, 2012. doi: 10.5244/C.26.127. URL https://doi.org/10.5244/C.26.127.

Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. OCR-VQA: visual question answering by reading text in images. In 2019 International Conference on Document Analysis and Recognition, ICDAR 2019, Sydney, Australia, September 20-25, 2019, pp. 947– 952. IEEE, 2019. doi: 10.1109/ICDAR.2019.00156. URL https://doi.org/10.1109/ ICDAR.2019.00156.

Dhruv Nathawani, Igor Gitman, Somshubra Majumdar, Evelina Bakhturina, Ameya Sunil Mahabaleshwarkar, , Jian Zhang, and Jane Polak Scowcroft. Nemotron-PostTraining-Dataset-v1, 2025. URL https://huggingface.co/datasets/nvidia/ Nemotron-Post-Training-Dataset-v1.

Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating video-based large language models. Computational Visual Media, 12(1):71–84, 2026. doi: 10.26599/CVM.2025.9450516.

Jason Obeid and Enamul Hoque. Chart-to-text: Generating natural language descriptions for charts by adapting the transformer model. In Brian Davis, Yvette Graham, John D. Kelleher, and Yaji Sripada (eds.), Proceedings of the 13th International Conference on Natural Language Generation, INLG 2020, Dublin, Ireland, December 15-18, 2020, pp. 138–147. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.INLG-1.20. URL https: //doi.org/10.18653/v1/2020.inlg-1.20.

OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303.08774. URL https://doi.org/10.48550/arXiv.2303.08774.

- OpenAI. Gpt-4v(ision) system card, 9 2023. URL https://cdn.openai.com/papers/ GPTV_System_Card.pdf. Accessed 2025-09-17.
- OpenAI. Gpt-5 system card. 2025. URL openai.com/index/gpt-5-system-card.

Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching CLIP to count to ten. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pp. 3147–3157. IEEE, 2023. doi: 10.1109/ICCV51070. 2023.00294. URL https://doi.org/10.1109/ICCV51070.2023.00294.

Dong Huk Park, Trevor Darrell, and Anna Rohrbach. Robust change captioning. In 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 November 2, 2019, pp. 4623–4632. IEEE, 2019. doi: 10.1109/ICCV.2019.00472. URL https: //doi.org/10.1109/ICCV.2019.00472.

Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Jiapeng Wang, Zhuoma Gongque, Shanglin Lei, Yifan Zhang, Zhe Wei, Miaoxuan Zhang, Runfeng Qiao, Xiao Zong, Yida Xu, Peiqing Yang, Zhimin Bao, Muxi Diao, Chen Li, and Honggang Zhang. Wemath: Does your large multimodal model achieve human-like mathematical reasoning? In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pp. 20023–20070. Association for Computational Linguistics, 2025. URL https://aclanthology.org/2025. acl-long.983/.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pp. 8748–8763. PMLR, 2021. URL http://proceedings.mlr. press/v139/radford21a.html.

Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind[inline-graphic not available: see fulltext]. In Minsu Cho, Ivan Laptev, Du Tran, Angela Yao, and Hongbin Zha (eds.), Computer Vision - ACCV 2024 - 17th Asian Conference on Computer Vision, Hanoi, Vietnam, December 8-12, 2024, Proceedings, Part V, volume 15476 of Lecture Notes in Computer Science, pp. 293–309. Springer, 2024. doi: 10.1007/ 978-981-96-0917-8\_17. URL https://doi.org/10.1007/978-981-96-0917-8_ 17.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael S. Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. Int. J. Comput. Vis., 115(3): 211–252, 2015. doi: 10.1007/S11263-015-0816-Y. URL https://doi.org/10.1007/ s11263-015-0816-y.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: an open large-scale dataset for training next generation image-text models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/ 2022/hash/a1859debfb3b59d094f3504d5ebb6c25-Abstract-Datasets_ and_Benchmarks.html.

Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-OKVQA: A benchmark for visual question answering using world knowledge. In Shai Avidan, Gabriel J. Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner (eds.), Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part VIII, volume 13668 of Lecture Notes in Computer Science, pp. 146–162. Springer, 2022. doi: 10.1007/978-3-031-20074-8\_9. URL https://doi.org/10.1007/ 978-3-031-20074-8_9.

Sanket Shah, Anand Mishra, Naganand Yadati, and Partha Pratim Talukdar. KVQA: knowledgeaware visual question answering. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019, pp. 8876–8884. AAAI Press,

2019. doi: 10.1609/AAAI.V33I01.33018876. URL https://doi.org/10.1609/aaai. v33i01.33018876.

Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 November 2, 2019, pp. 8429–8438. IEEE, 2019. doi: 10.1109/ICCV.2019.00852. URL https: //doi.org/10.1109/ICCV.2019.00852.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024. doi: 10.48550/ARXIV.2402.03300. URL https://doi.org/10.48550/arXiv.2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys 2025, Rotterdam, The Netherlands, 30 March 2025 - 3 April 2025, pp. 1279–1297. ACM, 2025. doi: 10.1145/3689031.3696075. URL https://doi.org/10.1145/3689031.3696075.

Baoguang Shi, Cong Yao, Minghui Liao, Mingkun Yang, Pei Xu, Linyan Cui, Serge J. Belongie, Shijian Lu, and Xiang Bai. ICDAR2017 competition on reading chinese text in the wild (RCTW17). In 14th IAPR International Conference on Document Analysis and Recognition, ICDAR 2017, Kyoto, Japan, November 9-15, 2017, pp. 1429–1434. IEEE, 2017. doi: 10.1109/ICDAR. 2017.233. URL https://doi.org/10.1109/ICDAR.2017.233.

Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, Florida, USA, November 12-16, 2024, pp. 4663–4680. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.FINDINGS-EMNLP.268. URL https://doi.org/10.18653/ v1/2024.findings-emnlp.268.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew J. Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id= 0IOX0YcCdTn.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 1620, 2019, pp. 8317–8326. Computer Vision Foundation / IEEE, 2019. doi: 10.1109/CVPR. 2019.00851. URL http://openaccess.thecvf.com/content_CVPR_2019/html/ Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.html.

Amanpreet Singh, Guan Pang, Mandy Toh, Jing Huang, Wojciech Galuba, and Tal Hassner. Textocr: Towards large-scale end-to-end reasoning for arbitrary-shaped scene text. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pp. 8802– 8812. Computer Vision Foundation / IEEE, 2021. doi: 10.1109/CVPR46437.2021.00869. URL https://openaccess.thecvf.com/content/CVPR2021/html/ Singh_TextOCR_Towards_Large-Scale_End-to-End_Reasoning_for_ Arbitrary-Shaped_Scene_Text_CVPR_2021_paper.html.

Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. WIT: wikipedia-based image text dataset for multimodal multilingual machine learning. In Fernando Diaz, Chirag Shah, Torsten Suel, Pablo Castells, Rosie Jones, and Tetsuya Sakai (eds.), SIGIR ’21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021, pp. 2443–2449. ACM, 2021. doi: 10.1145/ 3404835.3463257. URL https://doi.org/10.1145/3404835.3463257.

Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, and Yoav Artzi. A corpus for reasoning about natural language grounded in photographs. In Anna Korhonen, David R. Traum, and Lluís Màrquez (eds.), Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pp. 6418–6428. Association for Computational Linguistics, 2019. doi: 10.18653/V1/P19-1644. URL https://doi.org/10.18653/v1/p19-1644.

Yipeng Sun, Dimosthenis Karatzas, Chee Seng Chan, Lianwen Jin, Zihan Ni, Chee Kheng Chng, Yuliang Liu, Canjie Luo, Chun Chet Ng, Junyu Han, Errui Ding, and Jingtuo Liu. ICDAR 2019 competition on large-scale street view text with partial labeling - RRC-LSVT. In 2019 International Conference on Document Analysis and Recognition, ICDAR 2019, Sydney, Australia, September 20-25, 2019, pp. 1557–1562. IEEE, 2019. doi: 10.1109/ICDAR.2019.00250. URL https://doi.org/10.1109/ICDAR.2019.00250.

Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. Visualmrc: Machine reading comprehension on document images. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, ThirtyThird Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pp. 13878–13888. AAAI Press, 2021. doi: 10.1609/AAAI.V35I15.17635. URL https://doi.org/10.1609/aaai.v35i15.17635.

Benny J. Tang, Angie W. Boggust, and Arvind Satyanarayan. Vistext: A benchmark for semantically rich chart captioning. In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pp. 7268–7298. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.ACL-LONG.401. URL https://doi.org/10.18653/v1/2023.acl-long.401.

Fei Tang, Zhangxuan Gu, Zhengxi Lu, Xuyang Liu, Shuheng Shen, Changhua Meng, Wen Wang, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Gui-g2: Gaussian reward modeling for gui grounding, 2025a. URL https://arxiv.org/abs/2507.15846.

Fei Tang, Haolei Xu, Hang Zhang, Siqi Chen, Xingyu Wu, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Zeqi Tan, Yuchen Yan, Kaitao Song, Jian Shao, Weiming Lu, Jun Xiao, and Yueting Zhuang. A survey on (m)llm-based gui agents, 2025b. URL https://arxiv.org/abs/ 2504.13865.

Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025a.

Kwai Keye Team, Biao Yang, Bin Wen, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, Fan Yang, Guorui Zhou, Hao Peng, Haojie Ding, Jiaming Huang, Jiangxia Cao, Jiankang Chen, Jingyun Hua, Jin Ouyang, Kaibing Chen, Kaiyu Jiang, Kaiyu Tang, Kun Gai, Shengnan Zhang, Siyang Mao, Sui Huang, Tianke Zhang, Tingting Gao, Wei Chen, Wei Yuan, Xiangyu Wu, Xiao Hu, Xingyu Lu, Yang Zhou, Yifan Zhang, Yiping Yang, Yulong Chen, Zhenhua Wu, Zhenyu Li, Zhixin Ling, Ziming Li, Dehua Ma, Di Xu, Haixuan Gao, Hang Li, Jiawei Guo, Jing Wang, Lejian Ren, Muhao Wei, Qianqian Wang, Qigen Hu, Shiyao Wang, Tao Yu, Xinchen Luo, Yan Li, Yiming Liang, Yuhang Hu, Zeyi Lu, Zhuoran Yang, and Zixing Zhang. Kwai keye-vl technical report. CoRR, abs/2507.01949, 2025b. doi: 10.48550/ ARXIV.2507.01949. URL https://doi.org/10.48550/arXiv.2507.01949.

V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyue Fan, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan

An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhao Xue, Zhenyu Hou, Zhengxiao Du,

- Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025c. URL https://arxiv.org/abs/2507.01006.

Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Iyer, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, Xichen Pan, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024a. URL http://papers.nips.cc/paper_files/paper/2024/hash/ 9ee3a664ccfeabc0da16ac6f1f1cfe59-Abstract-Conference.html.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 9568–9578. IEEE, 2024b. doi: 10.1109/CVPR52733.2024.00914. URL https: //doi.org/10.1109/CVPR52733.2024.00914.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971, 2023. doi: 10.48550/ARXIV.2302.13971. URL https://doi.org/10.48550/arXiv.2302.13971.

Michael Tschannen, Alexey A. Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier J. Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. Siglip 2: Multilingual visionlanguage encoders with improved semantic understanding, localization, and dense features. CoRR, abs/2502.14786, 2025. doi: 10.48550/ARXIV.2502.14786. URL https://doi.org/ 10.48550/arXiv.2502.14786.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vlrethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025a.

Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, and Yu-Gang Jiang. To see is to believe: Prompting GPT-4V for better visual instruction tuning. CoRR, abs/2311.07574, 2023a. doi: 10.48550/ARXIV.2311.07574. URL https://doi.org/10.48550/arXiv.2311. 07574.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with mathvision dataset. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024a. URL http://papers.nips.cc/paper_files/paper/2024/ hash/ad0edc7d5fa1a783f063646968b7315b-Abstract-Datasets_and_ Benchmarks_Track.html.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. CoRR, abs/2409.12191, 2024b. doi: 10.48550/ARXIV. 2409.12191. URL https://doi.org/10.48550/arXiv.2409.12191.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin

Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, JingJing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. CoRR, abs/2508.18265, 2025b. doi: 10.48550/ARXIV.2508.18265. URL https://doi.org/10.48550/arXiv.2508.18265.

Weizhi Wang, Yu Tian, Linjie Yang, Heng Wang, and Xifeng Yan. Open-qwen2vl: Compute-efficient pre-training of fully-open multimodal llms on academic resources. CoRR, abs/2504.00595, 2025c. doi: 10.48550/ARXIV.2504.00595. URL https://doi.org/10. 48550/arXiv.2504.00595.

Xinyu Wang, Yuliang Liu, Chunhua Shen, Chun Chet Ng, Canjie Luo, Lianwen Jin, Chee Seng Chan, Anton van den Hengel, and Liangwei Wang. On the general value of evidence, and bilingual scene-text visual question answering. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pp. 10123– 10132. Computer Vision Foundation / IEEE, 2020. doi: 10.1109/CVPR42600.2020.01014. URL https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_On_ the_General_Value_of_Evidence_and_Bilingual_Scene-Text_Visual_ CVPR_2020_paper.html.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pp. 13484–13508. Association for Computational Linguistics, 2023b. doi: 10.18653/V1/2023.ACL-LONG.754. URL https://doi.org/10.18653/ v1/2023.acl-long.754.

Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, Alexis Chevalier, Sanjeev Arora, and Danqi Chen. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024c. URL http://papers.nips.cc/paper_files/paper/ 2024/hash/cdf6f8e9fd9aeaf79b6024caec24f15b-Abstract-Datasets_ and_Benchmarks_Track.html.

Yuxiang Wei, Yiheng Zheng, Yabo Zhang, Ming Liu, Zhilong Ji, Lei Zhang, and Wangmeng Zuo. Personalized image generation with deep generative models: A decade survey. Computational Visual Media, 11(6):1141–1194, 2025. doi: 10.26599/CVM.2025.9450495.

Chris Wendler. wendlerc/renderedtext, 2023. URL https://huggingface.co/datasets/ wendlerc/RenderedText.

Haoning Wu, Hanwei Zhu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Annan Wang, Wenxiu Sun, Qiong Yan, Xiaohong Liu, Guangtao Zhai, Shiqi Wang, and Weisi Lin. Towards open-ended visual quality comparison. In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (eds.), Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part III, volume 15061 of Lecture Notes in Computer Science, pp. 360–377. Springer, 2024. doi: 10.1007/ 978-3-031-72646-0\_21. URL https://doi.org/10.1007/978-3-031-72646-0_ 21.

xAI. Grok-1.5 vision preview: Connecting the digital and physical worlds with our first multimodal model., 4 2024. URL https://x.ai/blog/grok-1.5v. Accessed 2025-09-19.

Renqiu Xia, Bo Zhang, Haoyang Peng, Ning Liao, Peng Ye, Botian Shi, Junchi Yan, and Yu Qiao. Structchart: Perception, structuring, reasoning for visual chart understanding. CoRR, abs/2309.11268, 2023. doi: 10.48550/ARXIV.2309.11268. URL https://doi.org/10. 48550/arXiv.2309.11268.

Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pp. 9777–9786. Computer Vision Foundation / IEEE, 2021. doi: 10.1109/CVPR46437.2021.00965. URL https://openaccess.thecvf.com/content/CVPR2021/html/Xiao_ NExT-QA_Next_Phase_of_Question-Answering_to_Explaining_Temporal_ Actions_CVPR_2021_paper.html.

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal LLM logical reasoning benchmark in visual contexts. CoRR, abs/2407.04973, 2024. doi: 10.48550/ARXIV.2407.04973. URL https://doi.org/10.48550/arXiv.2407.04973.

Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, Wenhai Wang, Jifeng Dai, and Jinguo Zhu. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. CoRR, abs/2504.15279, 2025. doi: 10.48550/ARXIV.2504.15279. URL https://doi.org/ 10.48550/arXiv.2504.15279.

Zhen Xu, Hongyu Zhou, Sida Peng, Haotong Lin, Haoyu Guo, Jiahao Shao, Peishan Yang, Qinglin Yang, Sheng Miao, Xingyi He, Yifan Wang, Yue Wang, Ruizhen Hu, Yiyi Liao, Xiaowei Zhou, and Hujun Bao. Towards depth foundation models: Recent trends in vision-based depth estimation. Computational Visual Media, pp. 1–29, 2026. doi: 10.26599/CVM.2025.9450517.

Zhiyang Xu, Chao Feng, Rulin Shao, Trevor Ashby, Ying Shen, Di Jin, Yu Cheng, Qifan Wang, and Lifu Huang. Vision-flan: Scaling human-labeled tasks in visual instruction tuning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pp. 15271–15342. Association for Computational Linguistics, 2024. doi: 10. 18653/V1/2024.FINDINGS-ACL.905. URL https://doi.org/10.18653/v1/2024. findings-acl.905.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. CoRR, abs/2505.09388, 2025a. doi: 10.48550/ARXIV.2505.09388. URL https://doi.org/10.48550/arXiv.2505.09388.

Qi Yang, Bolin Ni, Shiming Xiang, Han Hu, Houwen Peng, and Jie Jiang. R-4b: Incentivizing general-purpose auto-thinking capability in mllms via bi-mode annealing and reinforce learning, 2025b. URL https://arxiv.org/abs/2508.21113.

Yue Yang, Ajay Patel, Matt Deitke, Tanmay Gupta, Luca Weihs, Andrew Head, Mark Yatskar, Chris Callison-Burch, Ranjay Krishna, Aniruddha Kembhavi, and Christopher Clark. Scaling text-rich image understanding via code-guided synthetic multimodal data generation. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pp. 17486–17505. Association for Computational Linguistics, 2025c. URL https://aclanthology.org/2025.acl-long.855/.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong

Sun. Minicpm-v: A GPT-4V level MLLM on your phone. CoRR, abs/2408.01800, 2024. doi: 10. 48550/ARXIV.2408.01800. URL https://doi.org/10.48550/arXiv.2408.01800.

Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, Qin Jin, Liang He, Xin Lin, and Fei Huang. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pp. 2841–2858. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-EMNLP.187. URL https://doi.org/10.18653/v1/2023.findings-emnlp.187.

Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, Jiayi Lei, Quanfeng Lu, Runjian Chen, Peng Xu, Renrui Zhang, Haozhe Zhang, Peng Gao, Yali Wang, Yu Qiao, Ping Luo, Kaipeng Zhang, and Wenqi Shao. Mmtbench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask AGI. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview. net/forum?id=R4Ng8zYaiz.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024a. URL https://openreview.net/forum?id=N8N0hgNDRt.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024b. URL https://openreview.net/forum?id= KOTutrSR2y.

Tailing Yuan, Zhe Zhu, Kun Xu, Cheng-Jun Li, Tai-Jiang Mu, and Shi-Min Hu. A large chinese text dataset in the wild. J. Comput. Sci. Technol., 34(3):509–521, 2019. doi: 10.1007/ S11390-019-1923-Y. URL https://doi.org/10.1007/s11390-019-1923-y.

Ye Yuan, Xiao Liu, Wondimu Dikubab, Hui Liu, Zhilong Ji, Zhongqin Wu, and Xiang Bai. Syntaxaware network for handwritten mathematical expression recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 1824, 2022, pp. 4543–4552. IEEE, 2022. doi: 10.1109/CVPR52688.2022.00451. URL https: //doi.org/10.1109/CVPR52688.2022.00451.

Xiang Yue, Yuansheng Ni, Tianyu Zheng, Kai Zhang, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 9556–9567. IEEE, 2024. doi: 10.1109/CVPR52733.2024.00913. URL https://doi.org/10.1109/CVPR52733. 2024.00913.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pp. 15134–15186. Association for Computational Linguistics, 2025a. URL https://aclanthology.org/2025.acl-long.736/.

- Zihao Yue, Zhenru Lin, Yifan Song, Weikun Wang, Shuhuai Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, Lei Li, Kainan Bao, Hao Tian, Hailin Zhang, Xiao-Gang Wang, Dawei Zhu, Cici, Chenhong He, Bowen Ye, Bowen Shen, Zihan Zhang, Zihan Jiang, Zhixian Zheng, Zhichao Song, Zhenbo Luo, Yue Yu, Yudong Wang, Yuanyuan Tian, Yu Tu, Yihan Yan, Yi Huang, Xu Wang,

Xinzhe Xu, Xingchen Song, Xing Zhang, Xing Yong, Xin Zhang, Xiangwei Deng, Wenyu Yang, Wenhan Ma, Weiwei Lv, Weiji Zhuang, Wei Liu, Sirui Deng, Shuo Liu, Shimao Chen, Shihua Yu, Shaohui Liu, Shande Wang, Rui Ma, Qiantong Wang, Peng Wang, Nuo Chen, Menghang Zhu, Kangyang Zhou, Kang Zhou, Kai Fang, Jun Shi, Jinhao Dong, Jiebao Xiao, Jiaming Xu, Huaqiu Liu, Hongshen Xu, Heng Qu, Haochen Zhao, Hanglong Lv, Guoan Wang, Duo Zhang, Dong Zhang, Di Zhang, Chong Ma, Chang Liu, Can Cai, and Bingquan Xia. Mimo-vl technical report. CoRR, abs/2506.03569, 2025b. doi: 10.48550/ARXIV.2506.03569. URL https:// doi.org/10.48550/arXiv.2506.03569.

Niccolò Zanichelli. nz/arxiv-ocr-v0.2, 2024. URL https://huggingface.co/datasets/ nz/arxiv-ocr-v0.2.

Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pp. 11941–11952. IEEE, 2023. doi: 10.1109/ICCV51070. 2023.01100. URL https://doi.org/10.1109/ICCV51070.2023.01100.

Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. RAVEN: A dataset for relational and analogical visual reasoning. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pp. 5317– 5327. Computer Vision Foundation / IEEE, 2019a. doi: 10.1109/CVPR.2019.00546. URL http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_RAVEN_ A_Dataset_for_Relational_and_Analogical_Visual_REasoNing_CVPR_ 2019_paper.html.

Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, Sam Dodge, Keen You, Zhen Yang, Aleksei Timofeev, Mingze Xu, Hong-You Chen, Jean-Philippe Fauconnier, Zhengfeng Lai, Haoxuan You, Zirui Wang, and et al. MM1.5: methods, analysis & insights from multimodal LLM fine-tuning. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025a. URL https://openreview.net/forum? id=HVtu26XDAA.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023a. URL http://papers.nips.cc/paper_files/paper/2023/ hash/64008fa30cba9b4d1ab1bd3bd3d57d61-Abstract-Datasets_and_ Benchmarks.html.

Liang Zhang, Anwen Hu, Haiyang Xu, Ming Yan, Yichen Xu, Qin Jin, Ji Zhang, and Fei Huang. Tinychart: Efficient chart understanding with program-of-thoughts learning and visual token merging. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pp. 1882–1898. Association for Computational Linguistics, 2024a. doi: 10.18653/V1/2024.EMNLP-MAIN.112. URL https://doi.org/10.18653/ v1/2024.emnlp-main.112.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, Peng Gao, and Hongsheng Li. MATHVERSE: does your multi-modal LLM truly see the diagrams in visual math problems? In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (eds.), Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part VIII, volume 15066 of Lecture Notes in Computer Science, pp. 169–186. Springer, 2024b. doi: 10.1007/978-3-031-73242-3\_10. URL https://doi.org/10. 1007/978-3-031-73242-3_10.

Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Ziyu Guo, Yichi Zhang, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Shanghang Zhang, Peng Gao, and Hongsheng Li. MAVIS: mathematical visual instruction tuning with an automatic data engine. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025b. URL https://openreview.net/forum?id=MnJzJ2gvuf.

Rui Zhang, Mingkun Yang, Xiang Bai, Baoguang Shi, Dimosthenis Karatzas, Shijian Lu, C. V. Jawahar, Yongsheng Zhou, Qianyi Jiang, Qi Song, Nan Li, Kai Zhou, Lei Wang, Dong Wang, and Minghui Liao. ICDAR 2019 robust reading challenge on reading chinese text on signboard. In 2019 International Conference on Document Analysis and Recognition, ICDAR 2019, Sydney, Australia, September 20-25, 2019, pp. 1577–1581. IEEE, 2019b. doi: 10.1109/ICDAR.2019. 00253. URL https://doi.org/10.1109/ICDAR.2019.00253.

Tianyu Zhang, Suyuchen Wang, Lu Li, Ge Zhang, Perouz Taslakian, Sai Rajeswar, Jie Fu, Bang Liu, and Yoshua Bengio. VCR: visual caption restoration. CoRR, abs/2406.06462, 2024c. doi: 10. 48550/ARXIV.2406.06462. URL https://doi.org/10.48550/arXiv.2406.06462.

Xiaoman Zhang, Chaoyi Wu, Ziheng Zhao, Weixiong Lin, Ya Zhang, Yanfeng Wang, and Weidi Xie. PMC-VQA: visual instruction tuning for medical visual question answering. CoRR, abs/2305.10415, 2023b. doi: 10.48550/ARXIV.2305.10415. URL https://doi.org/10.

- 48550/arXiv.2305.10415.

Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding. CoRR, abs/2306.17107, 2023c. doi: 10.48550/ARXIV.2306.17107. URL https://doi.org/10.

- 48550/arXiv.2306.17107.

Bo Zhao, Boya Wu, and Tiejun Huang. SVIT: scaling up visual instruction tuning. CoRR, abs/2307.04087, 2023a. doi: 10.48550/ARXIV.2307.04087. URL https://doi.org/10.

- 48550/arXiv.2307.04087.

Yilun Zhao, Yunxiang Li, Chenying Li, and Rui Zhang. Multihiertt: Numerical reasoning over multi hierarchical tabular and textual data. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pp. 6588–6600. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.ACL-LONG.454. URL https://doi.org/10.18653/v1/2022.acl-long.454.

Yilun Zhao, Chen Zhao, Linyong Nan, Zhenting Qi, Wenlin Zhang, Xiangru Tang, Boyu Mi, and Dragomir Radev. Robut: A systematic study of table QA robustness against human-annotated adversarial perturbations. In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pp. 6064–6081. Association for Computational Linguistics, 2023b. doi: 10.18653/V1/2023.ACL-LONG.334. URL https: //doi.org/10.18653/v1/2023.acl-long.334.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/ hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_ Benchmarks.html.

Mingyu Zheng, Xinwei Feng, Qingyi Si, Qiaoqiao She, Zheng Lin, Wenbin Jiang, and Weiping Wang. Multimodal table understanding. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 9102–9124. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.493. URL https://doi.org/10.18653/v1/2024.acl-long.493.

Xinyi Zheng, Douglas Burdick, Lucian Popa, Xu Zhong, and Nancy Xin Ru Wang. Global table extractor (GTE): A framework for joint table identification and cell structure recognition using visual context. In IEEE Winter Conference on Applications of Computer Vision, WACV 2021, Waikoloa, HI, USA, January 3-8, 2021, pp. 697–706. IEEE, 2021. doi: 10.1109/WACV48630. 2021.00074. URL https://doi.org/10.1109/WACV48630.2021.00074.

Yuke Zhu, Oliver Groth, Michael S. Bernstein, and Li Fei-Fei. Visual7w: Grounded question answering in images. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pp. 4995–5004. IEEE Computer Society, 2016. doi: 10.1109/CVPR.2016.540. URL https://doi.org/10.1109/CVPR.2016.540.

Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/ forum?id=VOAMTA8jKu.

APPENDIX

- A ADDITIONAL EXPERIMENTS

In this section, we provide a series of additional experiments to offer a more granular understanding of our model’s performance and the key components of our training methodology. We investigate the specific contributions of different training stages and analyze the model’s behavior under various inference conditions. These analyses serve to validate our design choices and highlight the critical factors that enable the model’s advanced reasoning capabilities.

- A.1 ABLATION OF DIFFERENT REASONING MODES AND STAGES

To further analyze our model’s performance and the impact of our training methodology, we conducted a comprehensive ablation study comparing the model at three key checkpoints: the initial SFT model (Stage 3), the model after refinement SFT (Stage 4), and the final model after reinforcement learning (Stage 5). For the SFT models, we also assessed performance across both short CoT (N-T) and long CoT (T) inference modes. This multifaceted analysis aims to understand the progressive impact of each training stage and the model’s behavior across different reasoning complexities.

As detailed in Tab. 3 to 6, we assessed performance across a wide array of benchmarks in two distinct inference modes:

- Table 3: Performance comparison of our model after Stage 3, Stage 4, and Stage 5 on general VQA

- benchmarks (Part 1). The top and second-best scores for each benchmark are highlighted.

MMMU

MMMU -Prostandard

MMBench -ENdev

BLINK

Count Bench

Hallusion Benchavg

MMBench -CNdev

Model Mode AI2D

val

val

- Stage3 N-T 82.7 53.3 93.2 54.2 79.6 81.3 58.9 43.2

- Stage3 T 83.7 52.6 91.4 57.0 83.0 81.9 66.7 48.5

- Stage4 N-T 81.2 54.9 92.4 56.8 80.7 82.7 59.8 43.9

- Stage4 T 83.8 52.5 90.5 59.8 81.2 83.0 66.8 50.4

- Stage5 T 85.3 55.0 93.0 58.2 84.2 85.5 66.1 50.7

Short CoT (N-T) mode: We prompt the model with <think>\n\n</think> to elicit a concise reasoning path with fewer steps (short CoT).

Long CoT (T) mode: We use the prompt <think>\n to encourage the model to produce a more detailed and in-depth reasoning process (long CoT).

The model exhibits strong performance across both inference modes, reflecting an effective alignment between its vision module and the dual-level reasoning capabilities inherent in the Qwen3 LLM. The results show a clear progression of improvements. First, the Stage 4 refinement yields

- Table 4: Performance comparison of our model after Stage 3, Stage 4, and Stage 5 on general VQA

- benchmarks (Part 2). The top and second-best scores for each benchmark are highlighted.

Model Mode MMStar

MMT -Benchval

MMVet MMVP

POPE

avg

RealWorldQA VisuLogic

VLMs are Blind

- Stage3 N-T 66.5 62.3 69.3 81.0 88.0 70.7 26.1 56.9

- Stage3 T 68.0 63.8 84.1 80.0 84.4 72.5 24.0 53.9

- Stage4 N-T 67.5 63.2 69.2 79.0 86.2 70.5 24.2 52.9

- Stage4 T 69.0 64.6 83.3 80.7 84.0 70.1 24.4 55.8

- Stage5 T 71.4 67.0 83.9 82.0 84.8 73.1 26.5 56.5

- Table 5: Performance comparison on benchmarks for table, chart, and document understanding. The top and second-best scores for each benchmark are highlighted.

Model Mode

CharXiv

DQ

CharXiv

RQ

ChartQA

test

DocVQA

val

InfoVQA

val

OCRBench

SEED -Bench2-Plus

- Stage3 N-T 81.7 47.0 82.1 86.3 66.5 84.2 66.7

- Stage3 T 83.0 53.4 86.8 87.3 72.5 84.1 67.7

- Stage4 N-T 84.4 48.3 79.4 86.3 66.7 83.0 67.2

- Stage4 T 84.7 55.3 86.7 87.2 72.3 83.1 67.7

- Stage5 T 84.8 57.3 86.1 87.0 72.9 82.5 68.5

discernible gains, which we attribute to the high quality of the curated 1M subset. Subsequently, the Stage 5 reinforcement learning provides a further significant boost. This stage primarily enhances the model’s reliability by mitigating common generation issues such as response repetition, which in turn elevates its final performance on the majority of benchmarks, particularly in complex math and reasoning. This finding underscores that our multi-stage recipe, combining high-quality data curation with final policy optimization, is critical for unlocking a model’s most advanced reasoning abilities.

A.2 SENSITIVITY ANALYSIS OF JUDGE MODELS

As our primary evaluation pipeline utilizes a Qwen-based judge (Qwen3-32B) to assess a Qwenbased model (Bee-8B), there is a potential risk of “self-preference” bias. To rigorously quantify the sensitivity of our results to the choice of the evaluator, we conducted a cross-family evaluation using GLM-4.5-FP8 (Zeng et al., 2025) as an independent judge across all 29 benchmarks.

Tab. 7 presents a comprehensive comparison of the performance of both Bee-8B-SFT and Bee-8BRL when evaluated by the original Qwen judge versus the GLM judge. The results indicate a high degree of consistency between the two evaluators.

While we observe discrepancies in specific benchmarks, most notably in ChartQA, DocVQA, and InfoVQA, where the GLM judge tends to be stricter, the overall impact on the global average is minimal. For Bee-8B-RL, the Global Average score shifts only slightly by -0.3 points (from 70.2 with the Qwen judge to 69.9 with the GLM judge). This minimal deviation confirms that while samefamily preference may exist to a minor degree or judges may have different calibration thresholds

- Table 6: Performance comparison on mathematical and logical reasoning benchmarks. The top and second-best scores for each benchmark are highlighted.

DynaMath

MathVerse

MathVista

LogicVista

MathVision

###### WeMath

Model Mode

vision_only

worst

mini

- Stage3 N-T 34.3 52.6 61.9 37.8 78.1 43.1

- Stage3 T 35.9 54.1 63.3 42.8 78.8 52.7

- Stage4 N-T 31.1 52.1 62.9 40.6 78.6 45.5

- Stage4 T 41.3 56.8 61.9 46.8 78.6 55.0

- Stage5 T 40.5 61.3 67.0 50.0 81.4 59.8

- Table 7: Sensitivity analysis of Bee-8B performance using different LLM judges (Qwen3-32B vs. GLM-4.5-FP8). We report the scores for both SFT and RL stages. The Average row shows the mean score across all 29 benchmarks. The minimal delta in the global average confirms the robustness of our evaluation.

Bee-8B-SFT Bee-8B-RL Qwen Judge GLM Judge ∆ Qwen Judge GLM Judge ∆

Task Benchmark

AI2D 83.8 83.7 -0.1 85.3 85.3 0.0

BLINKval 52.5 52.7 +0.2 55.0 55.2 +0.2 CountBench 90.6 90.3 -0.3 93.0 92.8 -0.2

HallusionBenchavg 59.8 59.4 -0.4 58.2 58.6 +0.4

MMBench-CNdev 81.2 81.2 0.0 84.2 84.2 0.0

MMBench-ENdev 83.0 83.0 0.0 85.5 85.7 +0.2 MMMUval 66.8 66.0 -0.8 66.1 65.6 -0.5 MMMU-Prostandard 50.5 49.0 -1.5 50.7 50.3 -0.4 MMStar 69.0 68.9 -0.1 71.4 71.4 0.0

General VQA

MMT-Benchval 64.6 64.6 0.0 67.0 67.1 +0.1 MMVet 83.3 85.0 +1.7 83.9 83.7 -0.2 MMVP 80.7 80.0 -0.7 82.0 82.0 0.0 POPEavg 84.0 84.0 0.0 84.8 84.8 0.0 RealWorldQA 70.1 70.1 0.0 73.1 72.8 -0.3 VisuLogic 24.4 24.4 0.0 26.5 26.5 0.0 VLMs are Blind 55.8 55.8 0.0 56.5 56.5 0.0

CharXivDQ 84.7 83.6 -1.1 84.8 84.8 0.0 CharXivRQ 55.3 54.0 -1.3 57.3 55.8 -1.5 ChartQAtest 86.7 84.2 -2.5 86.1 83.8 -2.3 DocVQAval 87.2 83.6 -3.6 87.0 83.4 -3.6 InfoVQAval 72.3 69.9 -2.4 72.9 70.7 -2.2 OCRBench 83.1 83.1 0.0 82.5 82.5 0.0 SEED-Bench2-Plus 67.7 67.8 +0.1 68.5 68.5 0.0

Table & Chart & OCR

DynaMathworst 41.3 41.5 +0.2 40.5 40.5 0.0 LogicVista 56.8 56.2 -0.6 61.3 60.9 -0.4

Math & Reasoning

MathVersevision_only 61.9 61.5 -0.4 67.0 66.9 -0.1 MathVision 46.8 47.6 +0.8 50.0 50.2 +0.2

MathVistamini 78.6 78.7 +0.1 81.4 81.4 0.0

###### WeMath 55.0 54.8 -0.2 59.8 60.0 +0.2 Average All Benchmarks 68.6 68.2 -0.4 70.2 69.9 -0.3

for specific tasks, our model’s strong performance is fundamentally robust and not an artifact of a specific evaluator.

- A.3 EVALUATION STABILITY ANALYSIS

To verify the stability of our results, we repeated the evaluation for both Bee-8B-SFT and Bee-8BRL across 5 independent inference runs. We report the scores for each individual run, the calculated mean, and the standard deviation (Std) in Tab. 8. We also provide the average score across all 29 benchmarks for each run. The extremely low standard deviation values across all benchmarks and the stable global average demonstrate that the model’s performance is highly consistent and reproducible.

- A.4 HUMAN EVALUATION OF DATA QUALITY

To verify that our automated curation aligns with human standards, we conducted a rigorous blind evaluation. We recruited 14 valid evaluators to perform 532 pairwise comparisons between Original and Enriched responses across 38 randomly sampled questions. The assessment covered four key dimensions: Accuracy (factual correctness), Reasoning & Explanation (logical depth), Instruction Following (constraint adherence), and Expression Style (fluency and structure).

The results, summarized in Tab. 9, reveal a strong alignment with human preferences. The dominant Tie rate (83.65%) in Accuracy validates that our Fidelity Verification module effectively maintains

- Table 8: Robustness analysis of Bee-8B (SFT and RL) across 5 independent runs. The table reports the score for each run, the mean, and the standard deviation.

Task Benchmark

Bee-8B-SFT (5 Runs) Bee-8B-RL (5 Runs)

R1 R2 R3 R4 R5 Mean Std R1 R2 R3 R4 R5 Mean Std

General VQA

AI2D 83.8 83.5 83.7 84.1 84.0 83.8 0.2 85.3 85.1 85.7 84.8 84.5 85.1 0.4 BLINKval 52.5 53.1 51.8 51.8 50.3 51.9 1.0 55.0 55.1 56.1 55.0 53.2 54.9 1.0 CountBench 90.6 92.2 91.0 92.2 91.4 91.5 0.7 93.0 92.2 91.8 93.8 93.4 92.9 0.9 HallusionBenchavg 59.8 57.9 59.0 59.3 59.9 59.2 0.8 58.2 58.5 60.6 59.7 60.9 59.6 1.2 MMBench-CNdev 81.2 82.2 81.8 82.0 81.7 81.8 0.4 84.2 83.8 84.4 83.8 84.1 84.1 0.3 MMBench-ENdev 83.0 82.6 82.7 83.4 83.9 83.1 0.6 85.5 84.6 85.3 84.6 84.6 84.9 0.5 MMMUval 66.8 65.3 65.4 65.7 64.8 65.6 0.7 66.1 68.0 67.7 67.9 67.1 67.4 0.8 MMMU-Prostandard 50.5 48.8 50.0 50.0 48.7 49.6 0.8 50.7 52.1 49.7 50.8 50.7 50.8 0.8 MMStar 69.0 68.7 70.3 68.5 69.4 69.2 0.7 71.4 72.9 70.3 70.9 71.9 71.5 1.0 MMT-Benchval 64.6 65.0 65.2 64.6 65.6 65.0 0.4 67.0 66.6 67.3 67.5 67.2 67.1 0.3 MMVet 83.3 81.7 82.6 78.9 82.8 81.8 1.8 83.9 85.9 83.6 82.1 82.2 83.5 1.5 MMVP 80.7 82.7 80.7 79.3 79.3 80.5 1.4 82.0 80.7 83.0 84.0 79.7 81.9 1.7 POPEavg 84.0 83.3 84.2 83.3 84.1 83.8 0.5 84.8 85.0 84.7 84.9 84.4 84.8 0.2 RealWorldQA 70.1 68.6 70.8 70.2 69.9 69.9 0.8 73.1 71.0 70.8 73.6 72.9 72.3 1.3 VisuLogic 24.4 25.4 23.7 22.7 24.8 24.2 1.0 26.5 25.4 25.3 26.1 26.6 26.0 0.6 VLMs are Blind 55.8 53.7 54.0 54.5 54.5 54.5 0.8 56.5 57.3 57.5 56.6 56.6 56.9 0.5

Table & Chart & OCR

CharXivDQ 84.7 84.1 84.6 84.2 84.9 84.5 0.4 84.8 84.7 84.8 84.8 84.8 84.8 0.1 CharXivRQ 55.3 56.4 53.4 56.3 55.1 55.3 1.2 57.3 57.2 54.6 55.8 56.6 56.3 1.1 ChartQAtest 86.7 86.6 87.4 86.9 86.5 86.8 0.3 86.1 87.3 87.0 87.1 87.4 87.0 0.5 DocVQAval 87.2 87.2 87.5 87.7 87.2 87.4 0.3 87.0 87.2 87.0 87.3 87.0 87.1 0.2 InfoVQAval 72.3 71.8 72.1 72.2 72.3 72.1 0.2 72.9 73.5 72.8 72.5 72.5 72.9 0.4 OCRBench 83.1 82.9 83.5 83.5 82.8 83.2 0.3 82.5 83.3 83.6 82.4 83.7 83.1 0.6 SEED-Bench2-Plus 67.7 67.3 67.7 67.2 68.4 67.7 0.5 68.5 68.6 69.1 68.3 68.8 68.7 0.3

Math & Reasoning

DynaMathworst 41.3 40.3 39.9 38.7 40.1 40.1 0.9 40.5 39.5 41.3 40.9 40.7 40.6 0.7 LogicVista 56.8 56.6 60.0 58.2 59.1 58.1 1.4 61.3 60.2 57.7 59.3 57.7 59.2 1.6

MathVersevision_only 61.9 63.7 62.6 64.5 64.0 63.3 1.0 67.0 67.1 66.2 65.4 66.6 66.5 0.7 MathVision 46.8 46.9 47.8 47.7 47.4 47.3 0.4 50.0 50.0 50.4 50.4 50.1 50.2 0.2

MathVistamini 78.6 78.7 79.0 79.1 78.7 78.8 0.2 81.4 79.6 79.9 79.9 81.3 80.4 0.9 WeMath 55.0 55.6 53.6 54.5 56.2 55.0 1.0 59.8 58.5 57.3 60.6 57.8 58.8 1.4

Average All Benchmarks 68.6 68.5 68.6 68.5 68.7 68.6 0.1 70.2 70.2 70.0 70.2 70.0 70.1 0.1

factual consistency with the ground truth. Crucially, evaluators overwhelmingly preferred the enriched data in Reasoning (72.74%) and Expression Style (69.92%). These findings confirm that our pipeline successfully transforms raw data into high-utility, logically structured content that meets human expectations for advanced reasoning tasks.

- Table 9: Blind human evaluation results comparing Original vs. Enriched responses (532 pairwise comparisons). The high Tie rate in Accuracy confirms fidelity, while the strong preference for Enriched in Reasoning and Style demonstrates that our data aligns better with human standards for high-quality responses.

Metric Accuracy Reasoning & Explanation Instruction Following Expression Style

Prefer Enriched 42 (7.89%) 387 (72.74%) 163 (30.64%) 372 (69.92%) Prefer Original 45 (8.46%) 75 (14.10%) 45 (8.46%) 87 (16.35%) Tie 445 (83.65%) 70 (13.16%) 324 (60.90%) 73 (13.72%)

- A.5 DATA CONTAMINATION ANALYSIS

To rigorously assess potential data leakage, we performed a comprehensive decontamination analysis against all 66,682 samples in our evaluation sets. We utilized Perceptual Hashing (pHash) for visual content and SimHash for textual instructions, calculating Hamming distances to identify matches ranging from exact duplicates (Distance = 0) to high-similarity near-duplicates (Distance ≤ 3).

As detailed in Tab. 10, the detected leakage is extremely minimal. Even under the relaxed threshold, only 29 samples (< 0.05%) were identified as potential overlaps, with merely 2 exact matches. No-

tably, these overlaps are concentrated in specific benchmarks like MathVision, while our primary comprehensive benchmarks—including MMMU, MMMU-Pro, MMStar, and CharXiv—remain completely free of any overlap.

We conclude that the identified overlap of 29 samples across all evaluation sets is statistically negligible and insufficient to influence the performance metrics or the validity of our conclusions. We strictly clarify that evaluation data was never intentionally included in our training set. Given that our dataset aggregates open-source collections, these few overlapping samples likely originated from the web-crawled nature of the upstream data sources.

- Table 10: Data decontamination analysis results. We report the number of overlapping samples between Honey-Data-15M and the evaluation sets (total 66,682 samples) at different Hamming Distance thresholds.

Threshold Overlaps Ratio Benchmarks with Overlap (Count) Exact Match (= 0) 2 0.003% MathVision (1), MathVista (1)

- Diff ≤ 1 4 0.006% MathVision (1), MathVista (1), DocVQA (1), InfoVQA (1)
- Diff ≤ 2 14 0.021% MathVision (10), DocVQA (2), MathVista (1), InfoVQA (1)
- Diff ≤ 3 29 0.043%

MathVision (19), MathVista (4), DocVQA (3), ChartQA (1), InfoVQA (1), OCRBench (1)

A.6 INTER-MODEL AGREEMENT ON VERIFICATION

To validate the reliability of our data curation pipeline and address potential concerns regarding the bias of using a single verifier model, we conducted a comprehensive experiment evaluating intermodel agreement using a diverse set of verifier models. We sampled a random subset of approximately 83,800 raw samples from our pool and performed the Fidelity Verification process independently using three different models with varying scales and capabilities: the original verifier utilized in our pipeline (Qwen2.5-VL-72B), a smaller, efficiency-focused variant (Qwen2.5-VL-32B), and a significantly larger Mixture-of-Experts model (Qwen3-VL-235B-A22B-Instruct) serving as a strong reference capability upper bound.

The number of samples retained by each model and the intersection of samples retained by all three models are reported in Tab. 11. The results demonstrate a high degree of inter-model consistency. The intersection of 63,203 samples constitutes 96.5% of the data retained by the powerful reference model (Qwen3-VL-235B-A22B-Instruct) and 91.3% by our chosen verifier. This significant overlap across model scales (spanning from 32B to 235B) confirms that the identified inconsistencies are largely objective flaws rather than artifacts of specific model biases. These findings validate Qwen2.5-VL-72B as a robust and reliable verifier for our pipeline.

- Table 11: Inter-model agreement analysis for Fidelity Verification across distinct model scales. The high intersection count demonstrates that our filtering criteria are robust and not heavily biased by a specific model.

Metric Qwen2.5-VL-72B (Ours) Qwen2.5-VL-32B Qwen3-VL-235B Intersection (All 3) Retained Samples 69,198 65,054 65,468 63,203

- B THE USE OF LARGE LANGUAGE MODELS

In preparing this manuscript, we utilized Large Language Models (LLMs) as a general-purpose assistive tool. Specifically, we employed LLMs to refine and polish the language, improve clarity, and perform comprehensive grammar checks. The models were also used to ensure that our phrasing and word choices were idiomatic and aligned with standard scientific discourse. Additionally, we received assistance from the LLM for minor LaTeX formatting adjustments and engaged in discussions with it to brainstorm and select an appropriate name for our dataset and model.

- C REPRODUCIBILITY STATEMENT

We are committed to the full reproducibility of our work and will publicly release our full-stack suite. This includes the Honey-Data-15M corpus, the final Bee-8B model weights, the intermediate checkpoints from our training recipe, the specific training data used for each intermediate stage, and the complete code for our HoneyPipe data pipeline and DataStudio framework.

- D ETHICS STATEMENT

This work adheres to the ICLR Code of Ethics and involved no human or animal subjects. Our work relies on public academic datasets and MLLMs. We acknowledge that Honey-Data-15M may inherit biases from its public sources, despite our HoneyPipe pipeline’s design to systematically filter noise, factual errors, and image-instruction mismatches. Users should be aware of these potential limitations. Similarly, Bee-8B may generate inaccurate or harmful content. We are publicly releasing our full suite (dataset, pipeline, model weights, etc.) to promote open academic research and strongly discourage malicious use. We are committed to transparency, releasing our full methodology to help the community understand, build upon, and mitigate the limitations of such models.

- E LIMITATIONS

- E.1 LIMITATION OF FIDELITY VERIFICATION

Our data curation pipeline employs an automated “LLM-as-a-Judge” mechanism for Fidelity Verification. We explicitly acknowledge that this design necessitates a strategic trade-off between recall (the coverage of valid samples) and computational cost.

Specifically, relying on a single-pass model-based verifier to enforce consistency with the original ground truth functions as a strict filter. High-quality, logically correct CoT samples may be inadvertently discarded if they conflict with the original answer due to valid variations (e.g., formatting differences, unit conversions, or synonyms) that the verifier fails to align. While massive-scale human annotation or extensive rejection sampling (e.g., generating and verifying dozens of candidates per sample) could theoretically resolve this ambiguity and achieve near-perfect recall, such approaches are prohibitively expensive and computationally infeasible at the scale of tens of millions of samples.

Therefore, we made a conscious design choice to prioritize precision (ensuring the retained data is factually consistent) and cost-effectiveness over recall. This pragmatic approach sacrifices a fraction of valid enriched data but ensures that the pipeline remains scalable, reproducible, and accessible to the broader academic community without requiring industrial-scale resources.

Furthermore, regarding the potential bias of using a single verifier model, we acknowledge that an ensemble of diverse judges could theoretically offer higher robustness. However, for the specific task of checking factual consistency against a ground truth, strong MLLMs generally exhibit a high degree of inter-model agreement. The inconsistencies identified by our verifier typically reflect objective flaws or significant deviations rather than model-specific artifacts. Thus, employing a single, strong open-source model as a verifier represents an optimal balance point, minimizing engineering complexity while maintaining high data quality standards.

- E.2 LIMITATION OF EVALUATION PARADIGM

As MLLMs evolve from simple pattern matchers to models capable of complex CoT reasoning, traditional deterministic evaluation methods like Regex have become insufficient. Advanced models now generate verbose outputs that standard extraction scripts often fail to parse correctly, leading to false negatives even when the reasoning is sound. Consequently, adopting “LLM-as-a-Judge” to perform semantic consistency checking has become a necessary evolution for the field.

However, this shift introduces a new layer of uncertainty: evaluation becomes inherently nondeterministic and dependent on the chosen judge’s capabilities. Unlike rigid code-based metrics, an LLM judge introduces subjectivity regarding formatting strictness and semantic equivalence, and may exhibit subtle biases such as self-preference. We acknowledge that performance metrics in the

era of reasoning models are no longer absolute but are bounded by the subjectivity of the evaluator. Establishing standardized, objective, and automated metrics for rapidly evolving reasoning models remains a critical open challenge for the community.

- F DETAILED TRAINING STAGE CONFIGURATIONS

This section provides an expanded description of the five-stage training recipe for Bee-8B, as introduced in Sec. 3. Our methodology follows a progressive curriculum, starting with foundational vision-language alignment, advancing to complex instruction-based SFT, and concluding with targeted refinement and reinforcement learning. For the purpose of clarity and reproducibility, the following subsections detail the specific data sources, and key configurations for each of the five training stages.

- F.1 STAGE 1: MLP WARMUP

This initial stage is dedicated to bridging the visual and language modalities. For this, we train only the MLP projector on a curated dataset of approximately one million image-caption pairs, keeping the vision encoder and LLM backbone frozen. This training set is a carefully assembled collection, combining roughly 560K samples from LLaVA-OneVision (Li et al., 2025a) with a highquality subset of around 440K samples derived from COYO (Byeon et al., 2022). To generate this high-quality subset, we enriched the original responses by recaptioning them with the powerful Qwen2.5-VL-72B (Bai et al., 2025), followed by rule-based filtering. This entire process efficiently maps visual features into the language model’s token space without disturbing the powerful pretrained weights of the core components.

- F.2 STAGE 2: VISION-LANGUAGE ALIGNMENT

In the second stage, we unfreeze all model components to build foundational multimodal capabilities. The training data is a large-scale composite resource, mixing approximately 12.6 million vision-language pairs with 1.43 million text-only samples to teach visual understanding while preserving language skills. The vision-language component is a meticulously curated collection from three main sources: a high-quality filtered subset of LAION (Schuhmann et al., 2022) (~6.9M), a refined selection from COYO (Byeon et al., 2022) (~5.4M), and additional specialized data from the stage 1.5 of LLaVA-OneVision (Li et al., 2025a) (~300K). All vision-language data was constructed using the same recaptioning and filtering methodology detailed in Sec. F.1. Crucially, to preserve the LLM’s intrinsic reasoning abilities and mitigate catastrophic forgetting, we integrate a blend of text-only data from the Nemotron dataset (Bercovich et al., 2025; Nathawani et al., 2025). This blend includes long CoT samples for deep, multi-step problem-solving, sourced from reasoning-focused subsets like Nemotron-STEM (~458K), Nemotron-Math (~229K), and a reasoning-intensive version of Nemotron-Chat (~367K). This is complemented by the short CoT Nemotron-Chat subset (~376K) to maintain broad conversational proficiency. This mixed-modality approach enables the model to learn robust visual-language correlations while ensuring its core reasoning engine remains intact for subsequent stages.

- F.3 STAGE 3: MULTIMODAL SFT

In this stage, we train the model on the entire Honey-Data-15M to develop its advanced instructionfollowing and reasoning capabilities with vision. A detailed breakdown of the dataset’s composition, including its sources and dual-level CoT distribution, is provided in Fig. 3 and Tab. 12. Training for one full epoch ensures complete exposure to this diverse data distribution. This is critical for learning from the rarer but more complex long CoT samples and allowing the model to learn their intricate reasoning patterns.

- F.4 STAGE 4: EFFICIENT REFINEMENT SFT

This final stage of SFT is designed for dual purposes: to conduct targeted refinement of the model’s capabilities and to provide an accessible, efficient training option for researchers with limited computational resources. To this end, we curated a high-quality 1M subset from our full 15M dataset

Table 12: A detailed breakdown of the datasets used in our collection, categorized by task.

|Task|Dataset|
|---|---|
|General|SVIT-mix-665K (Zhao et al., 2023a), ALLaVA (Chen et al., 2024a), TQA (Kembhavi et al., 2017),<br><br>LLaVA-NeXT-Data (Liu et al., 2024c), IconQA (Lu et al., 2021b), Objects365 (Shao et al., 2019),<br><br>Vision FLAN (Xu et al., 2024), idefics375k (Laurençon et al., 2023), ViQuAE (Lerner et al., 2022),<br><br>Co-Instruct (Wu et al., 2024), LVIS-InstructV4 (Wang et al., 2023a), DreamSim (Fu et al., 2023),<br><br>SVIT-core-150K (Zhao et al., 2023a), Visual7W (Zhu et al., 2016), ShareGPT4V(SAM) (Li et al., 2025a),<br><br>Cambrian (Filter) (Tong et al., 2024a), PixMo-CapQA (Deitke et al., 2025), HQ-Edit (Hui et al., 2025),<br><br>PixMo-AskModelAnything (Deitke et al., 2025), ShareGPT4o (Cui et al., 2024), IDK (Cha et al., 2024), EST-VQA (Wang et al., 2020), COCO (Lin et al., 2014), PixMo-Point-Explanations (Deitke et al., 2025), Birds-to-Words (Forbes et al., 2019), GQA (Hudson & Manning, 2019), NextQA (Xiao et al., 2021),<br><br>ALFWorld (Shridhar et al., 2021), Cauldron(mulberry) (Laurençon et al., 2024b), VSR (Liu et al., 2023a).<br><br>A-OKVQA (Schwenk et al., 2022), KVQA (Shah et al., 2019), ContrastiveCaption (Jiang et al., 2024),<br><br>WebQA (Chang et al., 2022b), WildVision (Lu et al., 2024b), New Yorker Caption (Hessel et al., 2023),<br><br>InternVL-SA-1B-Caption (Chen et al., 2023; 2024e), ShareGPT4V(Knowledge) (Chen et al., 2024b), LLaVA-Instruct-300k (Liu et al., 2023c), VIST (Huang et al., 2016), LRV Normal (Liu et al., 2023b), KonIQ-10k (Hosu et al., 2020), Hateful Memes (Kiela et al., 2020), ART500K (Mao et al., 2017),<br><br>NLVR2 (Suhr et al., 2019), ScanQA (Azuma et al., 2022), MMChat-Twitter-Post (Li et al., 2024b),<br><br>nuScenes (Caesar et al., 2020), FlintstonesSV (Gupta et al., 2018), MagicBrush (Zhang et al., 2023a)|
|Chart|TinyChart (Zhang et al., 2024a), DVQA (Kafle et al., 2018), UniChart (Masry et al., 2023), CoSyn(chart, table, diagram, graphic) (Yang et al., 2025c), ArxivQA (Li et al., 2024c), FigureQA (Kahou et al., 2018), MMTab (Zheng et al., 2024), PlotQA (Methani et al., 2020), UReader QA (Ye et al., 2023), RobuT WikiSQL (Zhao et al., 2023b), TabMWP (Lu et al., 2023), RobuT WTQ (Zhao et al., 2023b), UReader KG (Ye et al., 2023), Chart2Text (Obeid & Hoque, 2020), ChartQA (Masry et al., 2022), MMC-Instruction (Liu et al., 2024a), RobuT SQA (Zhao et al., 2023b), MAVIS-Function (Zhang et al., 2025b), VisText (Tang et al., 2023), MultiHiertt (Zhao et al., 2022), SciTSR (Chi et al., 2019), LRV Chart (Liu et al., 2023b), SimChart9K (Xia et al., 2023), Infographic (Mathew et al., 2022), HiTab (Cheng et al., 2022)<br><br>|
|Caption|PixMo-Cap (Deitke et al., 2025), WIT (Srinivasan et al., 2021), ST-VQA (Biten et al., 2019b),<br><br>COYO-Recaption (Byeon et al., 2022), Sherlock (Hessel et al., 2022)|
|STEM|VisualWebInstruct(filtered) (Li et al., 2025a), MapQA (Chang et al., 2022a), VizWiz (Gurari et al., 2018), MetaMathQA (Yu et al., 2024a), Geo170K (Gao et al., 2025), VisualWebInstruct (Jia et al., 2025), MathV360K(TQA) (Shi et al., 2024), AI2D (Kembhavi et al., 2016), GeomVerse (Kazemi et al., 2023), GeoQA+ (Chen et al., 2021), MAVIS-Geo (Zhang et al., 2025b), CMM-Math (Liu et al., 2024d), CoSyn(math, music, chemical, circuit) (Yang et al., 2025c), MAVIS-Metagen (Zhang et al., 2025b), PMC-VQA (Zhang et al., 2023b), PathVQA (He et al., 2020), InterGPS (Lu et al., 2021a), VQA-RAD (Lau et al., 2018), RAVEN (Zhang et al., 2019a), AI2D(GPT4V) (Kembhavi et al., 2016), Geometry3K (Lu et al., 2021a), AI2D(InternVL) (Kembhavi et al., 2016), MMChem (Li et al., 2025b), WebSight (Laurençon et al., 2024c), UniGeo (Chen et al., 2022), ScienceQA (Lu et al., 2022)<br><br>|
|Document<br><br>|Ureader Chart (Ye et al., 2023), OCR-VQA (Mishra et al., 2019), InfographicVQA (Mathew et al., 2022), CoSyn(document, nutrition) (Yang et al., 2025c), POIE (Kuang et al., 2023), EATEN (Guo et al., 2019), FinTabNet (Zheng et al., 2021), UreaderOCR (Ye et al., 2023), InfoVQA (Mathew et al., 2022), Docmatix (Laurençon et al., 2024a), DocVQA (Mathew et al., 2021), ScreenQA (Hsiao et al., 2025), TextVQA (Singh et al., 2019), DocReason (Hu et al., 2024), VisualMRC (Tanaka et al., 2021), LLaVAR GPT4 (Zhang et al., 2023c)|
|Grounding & Counting<br><br>|CLEVR (Johnson et al., 2017), TallyQA (Acharya et al., 2019), VisualGenome (Krishna et al., 2017), TQA (Kembhavi et al., 2017), MovieNet (Huang et al., 2020), MathV360K(VQA-AS) (Shi et al., 2024), CLEVR-Math (Lindström & Abraham, 2022), Super-CLEVR (Li et al., 2023b), IconQA (Lu et al., 2021b), CLEVR-Change (Park et al., 2019)<br><br>|
|OCR|K12 Printing (Li et al., 2025a), ArXiv OCR (Zanichelli, 2024), HME (Yuan et al., 2022),<br><br>VCR-Wiki (Zhang et al., 2024c), Sroie (Huang et al., 2021), IIIT 5K (Mishra et al., 2012),<br><br>ICDAR-LSVT-zh (Sun et al., 2019), TextOCR (Singh et al., 2021), ReCTs (Zhang et al., 2019b),<br><br>Orand-Car-A (Russakovsky et al., 2015), Rendered Text (Wendler, 2023), ICDAR2017 (Shi et al., 2017),<br><br>Chrome-Writing (Wendler, 2023), MTWI(zh) (He et al., 2018), IAM (Marti & Bunke, 2002),<br><br>ICDAR2019 (Biten et al., 2019a), CTW (Yuan et al., 2019)|

through a meticulous, multi-faceted selection strategy. Our process began by defining target proportions for various topics to ensure a balanced and comprehensive training corpus. We established a more rational distribution among topics such as STEM, Chart, Document, Grounding, and OCR, among others, while also maintaining a substantial portion for General. Furthermore, we aimed for

an approximate 1:1 ratio between long-chain and short-chain conversations to balance depth and breadth.

The core of our selection methodology was a quality-driven quota system. We first manually assigned a quality score (on a scale of 1 to 5) to each data source. This score was the primary factor used to determine each source’s proportional contribution towards the overall target for its topic. This approach ensured that higher-quality sources contributed a proportionally larger number of samples, while still guaranteeing that every source was represented in the final subset to preserve diversity. Within each source’s assigned quota, we employed a hybrid sampling strategy to balance difficulty and variety. Specifically, 60% of the data was selected by prioritizing conversations with the longest responses, based on the hypothesis that longer responses often correlate with more complex and challenging user queries. The remaining 40% was chosen via random sampling from the source to maintain broad diversity and prevent overfitting on specific types of difficult instructions. Through this stratified approach, we successfully constructed a 1M subset that is not only computationally efficient for training but also features a more rational topic distribution, preserving the difficulty and diversity of the original, larger dataset.

- F.5 STAGE 5: REINFORCEMENT LEARNING WITH GRPO

In this final stage, we employ the Group Relative Policy Optimization (GRPO) (Shao et al., 2024) algorithm to address persistent SFT issues such as text repetition, incomplete responses, and improper formatting. The reinforcement learning (RL) process is implemented with the verl framework (Sheng et al., 2025), where each rollout uses a batch size of 512, and the policy model updates its gradients with a batch size of 128.

The training data was constructed using prompts from the open-source MMK12 (Meng et al., 2025) and ViRL39K (Wang et al., 2025a) datasets. To improve data quality, the ViRL39K dataset was preprocessed by removing multi-image samples and randomly splitting 95% of the remaining data as the training set. For each prompt, we generated a set of candidate responses, enabling GRPO to refine the model’s policy by learning to distinguish high-quality responses from flawed ones.

To guide this optimization, we adopt a rule-based reward function consisting of two components: a format reward (with a weight of 0.2) that enforces the presence of \boxed{} in the final output, and an accuracy reward (with a weight of 0.8) that evaluates whether the extracted content inside \boxed{} matches the ground-truth answer. This targeted reinforcement learning step provides a final polish, significantly improving the model’s overall reliability and output quality.

- G DATA LICENSING AND GOVERNANCE

To ensure rigorous compliance and promote responsible adoption within the community, we have established a transparent licensing framework and governance policy for Honey-Data-15M. Our approach is modeled after industry standards, such as The Cauldron Dataset (Laurençon et al., 2024b), to clarify rights and usage boundaries. Honey-Data-15M is primarily aggregated from established, large-scale open-source collections, including LLaVA-OneVision and MAmmoth-VL, which typically operate under permissive licenses like Apache 2.0. While we accessed data through these aggregates, we performed fine-grained tracing to generate the source distribution statistics, ensuring maximum credit is given to the original content creators.

We implement a clear dual-layered licensing structure for the repository. First, we explicitly state that Honey-Data-15M is a composite collection where each sub-dataset remains governed by its specific original license. Users are strictly mandated to adhere to the terms and restrictions of each original source. Second, to the extent that we hold rights in the modified prompts, structural formatting, and the newly generated CoT responses, these contributions are licensed under CC-BY-NC-4.0 (Creative Commons Attribution-NonCommercial 4.0 International).

Regarding specific materials such as OCRed textbooks or K-12 content, we clarify that we did not independently digitize physical materials; we utilized them strictly as they exist in public academic datasets. For Personally Identifiable Information (PII), we primarily rely on the anonymization measures performed by the original dataset curators (e.g., SROIE explicitly addresses privacy). To address any potential future concerns regarding copyright or privacy, we have implemented a strict

Notice and Takedown Policy. We have established a dedicated communication channel for these concerns, and upon receiving a valid notice regarding infringement or PII exposure, we are committed to promptly reviewing the request and removing the contested content from our distribution.

From Tab. 13 to Tab. 19 provides a detailed enumeration of the license terms, usage restrictions (e.g., Non-Commercial) for the constituent source categories. For instances where definitive licensing information remained unverifiable despite exhaustive search, entries are left blank to maintain accuracy.

- Table 13: Detailed overview of the licensing terms, usage restrictions for the General Category

|Dataset|License|Restrictions|
|---|---|---|
|SVIT-mix-665K<br><br>|CC BY 4.0<br><br>|Appropriate credit must be given when sharing or adapting the work.|
|SVIT-core-150K|CC BY 4.0<br><br>|Appropriate credit must be given when sharing or adapting the work.|
|ALLaVA|CC BY-NC 4.0|Appropriate credit must be given and no commercial use is permitted when sharing or adapting the work.|
|LLaVA-NeXT-Data|Apache-2.0<br><br>|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|LLaVA-Instruct-300k<br><br>|CC BY 4.0|Appropriate credit must be given when sharing or adapting the work.|
|Vision FLAN|–<br><br>|–|
|idefics375k|CC BY 4.0<br><br>|Appropriate credit must be given when sharing or adapting the work.|
|PixMo-CapQ|ODC-By 1.0|Appropriate credit must be given when using, sharing, or creating works from the database.|
|PixMo-AskModelAnything<br><br>|ODC-By 1.0|Appropriate credit must be given when using, sharing, or creating works from the database.|
|PixMo-Point-Explanations|ODC-By 1.0<br><br>|Appropriate credit must be given when using, sharing, or creating works from the database.|
|LVIS-Instruct4V<br><br>|–|–|
|ShareGPT4V(SAM)<br><br>|Apache-2.0<br><br>|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|Cambrian (Filter)|Apache-2.0<br><br>|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|ShareGPT4o|MIT|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|NLVR2|–|–|
|GQA|–<br><br>|–|
|LRV Normal<br><br>|BSD-3-Clause<br><br>|Appropriate credit must be given, notices must be retained, and the author’s name cannot be used for endorsement.|
|HQ-Edit|CC BY-NC 4.0<br><br>|Appropriate credit must be given and no commercial use is permitted when sharing or adapting the work.|
|ALFWorld<br><br>|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|Visual7W<br><br>|MIT|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|Co-Instruct|MIT|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|Cauldron(mulberry)<br><br>|–|–|
|A-OKVQA<br><br>|Apache-2.0|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|Continued on next page| | |

###### Table 13 – continued from previous page

|Dataset|License|Restrictions|
|---|---|---|
|IconQA<br><br>|CC BY-NC-SA 4.0<br><br>|Appropriate credit must be given, no commercial use is permitted, and adaptations must be shared under the same license.|
|VIST|–<br><br>|–|
|KVQA|–<br><br>|–|
|ContrastiveCaption|Apache-2.0<br><br>|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|FlintstonesSV<br><br>|–|–|
|InternVL-SA-1B-Caption|MIT|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|IDK|BSD-3-Clause<br><br>|Appropriate credit must be given, notices must be retained, and the author’s name cannot be used for endorsement.|
|COCO|CC BY 4.0|Appropriate credit must be given when sharing or adapting the work.|
|EST-VQA<br><br>|–|–|
|Birds-to-Words<br><br>|CC BY-SA 4.0<br><br>|Appropriate credit must be given and adaptations must be shared under the same license.|
|ART500K<br><br>|–<br><br>|Use is limited to non-commercial research, and no redistribution or commercial exploitation is permitted.|
|DreamSim<br><br>|MIT|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|ScanQA|CC BY-NC-SA 3.0|Appropriate credit must be given, no commercial use is permitted, and adaptations must be shared under the same license.|
|MagicBrush<br><br>|CC BY 4.0<br><br>|Appropriate credit must be given when sharing or adapting the work.|
|KonIQ-10k|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|Hateful Memes<br><br>|–<br><br>|–|
|WebQA<br><br>|CC0-1.0<br><br>|No restrictions are imposed, and the work may be used freely for any purpose.|
|nuScenes<br><br>|CC BY-NC-SA 4.0<br><br>|Appropriate credit must be given, no commercial use is permitted, and adaptations must be shared under the same license.|
|Objects365|CC BY 4.0|Appropriate credit must be given when sharing or adapting the work.|
|MMChat-Twitter-Post|CC BY 4.0<br><br>|Appropriate credit must be given when sharing or adapting the work.|
|NextQA|MIT|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|VSR|CC BY 4.0|Appropriate credit must be given when sharing or adapting the work.|
|New Yorker Caption<br><br>|CC BY 4.0|Appropriate credit must be given when sharing or adapting the work.|
|ViQuAE|CC BY 4.0<br><br>|Appropriate credit must be given when sharing or adapting the work.|
|TQA<br><br>|CC BY-NC 3.0<br><br>|Appropriate credit must be given and no commercial use is permitted when sharing or adapting the work.|
|ShareGPT4V(Knowledge)|CC BY-NC 4.0|Appropriate credit must be given and no commercial use is permitted when sharing or adapting the work.|
|WildVision<br><br>|CC BY 4.0|Appropriate credit must be given when sharing or adapting the work.|

##### Table 14: Detailed overview of the licensing terms, usage restrictions for the Chart Category

|Dataset|License|Restrictions|
|---|---|---|
|TinyChart<br><br>|Apache-2.0<br><br>|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|DVQA<br><br>|CC BY-NC 4.0<br><br>|Appropriate credit must be given and no commercial use is permitted when sharing or adapting the work.|
|UniChart|–<br><br>|–|
|CoSyn(chart)<br><br>|ODC-By 1.0|Appropriate credit must be given when using, sharing, or creating works from the database.|
|CoSyn(table)|ODC-By 1.0<br><br>|Appropriate credit must be given when using, sharing, or creating works from the database.|
|CoSyn(diagram)<br><br>|ODC-By 1.0|Appropriate credit must be given when using, sharing, or creating works from the database.|
|CoSyn(graphic)<br><br>|ODC-By 1.0<br><br>|Appropriate credit must be given when using, sharing, or creating works from the database.|
|ArxivQA<br><br>|CC BY-SA 4.0|Appropriate credit must be given and adaptations must be shared under the same license.|
|FigureQA|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|MMTab|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|PlotQA|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|UReader QA|Apache-2.0|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|UReader KG|–|–|
|RobuT WikiSQL<br><br>|MIT|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|RobuT SQA<br><br>|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|RobuT WTQ|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|TabMWP<br><br>|CC BY-NC-SA 4.0<br><br>|Appropriate credit must be given, no commercial use is permitted, and adaptations must be shared under the same license.|
|Chart2Text<br><br>|–|–|
|ChartQA|GPL-3.0<br><br>|Appropriate credit must be given and any modified versions must be released under the same license when distributing the work.|
|MMC-Instruction<br><br>|CC BY-SA 4.0|Appropriate credit must be given and adaptations must be shared under the same license.|
|MAVIS-Function<br><br>|MIT|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|VisText|GPL-3.0|Appropriate credit must be given and any modified versions must be released under the same license when distributing the work.|
|SciTSR|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|MultiHiertt<br><br>|MIT|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|Continued on next page| | |

- Table 14 – continued from previous page

|Dataset|License|Restrictions|
|---|---|---|
|SimChart9K|–|–|
|HiTab|C-UDA 1.0|Use is limited to computational purposes, attribution must be retained, and any redistributed data must remain under the same terms.|
|LRV Chart<br><br>|BSD-3-Clause|Appropriate credit must be given, notices must be retained, and the author’s name cannot be used for endorsement.|
|Infographic|–<br><br>|–|

- Table 15: Detailed overview of the licensing terms, usage restrictions for the Caption Category

|Dataset|License|Restrictions|
|---|---|---|
|COYO-Recaption<br><br>|CC BY 4.0<br><br>|Appropriate credit must be given when sharing or adapting the work.|
|PixMo-Cap<br><br>|ODC-By 1.0|Appropriate credit must be given when using, sharing, or creating works from the database.|
|WIT|CC BY-SA 3.0|Appropriate credit must be given and adaptations must be shared under the same license.|
|Sherlock<br><br>|CC BY 4.0<br><br>|Appropriate credit must be given when sharing or adapting the work.|
|ST-VQA|–|–|

- Table 16: Detailed overview of the licensing terms, usage restrictions for the STEM Category

|Dataset|License|Restrictions|
|---|---|---|
|VisualWebInstruct(filtered)<br><br>|Apache-2.0<br><br>|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|MapQA|CC BY-NC-SA 4.0|Appropriate credit must be given, no commercial use is permitted, and adaptations must be shared under the same license.|
|VisualWebInstruct<br><br>|Apache-2.0|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|MetaMathQA|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|Geo170K<br><br>|–|–|
|MAVIS-Metagen|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|GeoQA+<br><br>|–<br><br>|–|
|MAVIS-Geo|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|CoSyn(math)|ODC-By 1.0<br><br>|Appropriate credit must be given when using, sharing, or creating works from the database.|
|CoSyn(music)|ODC-By 1.0|Appropriate credit must be given when using, sharing, or creating works from the database.|
|CoSyn(chemical)|ODC-By 1.0<br><br>|Appropriate credit must be given when using, sharing, or creating works from the database.|
|CoSyn(circuit)<br><br>|ODC-By 1.0|Appropriate credit must be given when using, sharing, or creating works from the database.|
|AI2D|–<br><br>|–|
|PMC-VQA<br><br>|CC BY-SA<br><br>|Appropriate credit must be given and adaptations must be shared under the same license.|
|Continued on next page| | |

Table 16 – continued from previous page

|Dataset|License|Restrictions|
|---|---|---|
|RAVEN<br><br>|GPL-3.0<br><br>|Appropriate credit must be given and any modified versions must be released under the same license when distributing the work.|
|PathVQA|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|MathV360K(TQA)|Apache-2.0<br><br>|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|ScienceQA|CC BY-SA 4.0|Appropriate credit must be given and adaptations must be shared under the same license.|
|Geometry3K|MIT|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|AI2D(InternVL)<br><br>|–|–|
|MMChem|–<br><br>|–|
|WebSight|CC BY 4.0<br><br>|Appropriate credit must be given when sharing or adapting the work.|
|UniGeo<br><br>|–<br><br>|–|
|GeomVerse|–<br><br>|–|
|AI2D(GPT4V)|–<br><br>|–|
|VizWiz<br><br>|CC BY 4.0<br><br>|Appropriate credit must be given when sharing or adapting the work.|
|VQA-RAD<br><br>|CC0-1.0|No restrictions are imposed, and the work may be used freely for any purpose.|
|CMM-Math<br><br>|BSD-3-Clause<br><br>|Appropriate credit must be given, notices must be retained, and the author’s name cannot be used for endorsement.|
|InterGPS|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|

- Table 17: Detailed overview of the licensing terms, usage restrictions for the Document Category

|Dataset|License|Restrictions|
|---|---|---|
|Ureader Chart|Apache-2.0|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|OCR-VQA<br><br>|–<br><br>|–|
|CoSyn(document)|ODC-By 1.0<br><br>|Appropriate credit must be given when using, sharing, or creating works from the database.|
|CoSyn(nutrition)<br><br>|ODC-By 1.0|Appropriate credit must be given when using, sharing, or creating works from the database.|
|ScreenQA|CC BY 4.0|Appropriate credit must be given when sharing or adapting the work.|
|FinTabNet<br><br>|–<br><br>|–|
|TextVQA|CC BY 4.0<br><br>|Appropriate credit must be given when sharing or adapting the work.|
|EATEN|–<br><br>|–|
|DocVQA|–|–|
|LLaVAR GPT4<br><br>|CC-BY-SA 4.0|–|
|Docmatix<br><br>|MIT|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|InfoVQA<br><br>|–<br><br>|–|
|UreaderOCR|Apache-2.0<br><br>|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|Continued on next page| | |

Table 17 – continued from previous page

|Dataset|License|Restrictions|
|---|---|---|
|DocReason<br><br>|Apache-2.0<br><br>|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|InfographicVQA|–<br><br>|–|
|VisualMRC|–<br><br>|–|
|POIE<br><br>|–<br><br>|–|

- Table 18: Detailed overview of the licensing terms, usage restrictions for the Grounding & Counting Category

|Dataset|License|Restrictions|
|---|---|---|
|CLEVR<br><br>|CC BY 4.0<br><br>|Appropriate credit must be given when sharing or adapting the work.|
|TallyQA|Apache-2.0|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|VisualGenome<br><br>|CC BY 4.0|Appropriate credit must be given when sharing or adapting the work.|
|IconQA<br><br>|CC BY-NC-SA 4.0|Appropriate credit must be given, no commercial use is permitted, and adaptations must be shared under the same license.|
|TQA<br><br>|CC BY-NC 3.0|Appropriate credit must be given and no commercial use is permitted when sharing or adapting the work.|
|MovieNet<br><br>|–<br><br>|–|
|CLEVR-Math|CC BY 4.0|Appropriate credit must be given when sharing or adapting the work.|
|Super-CLEVR|BSD<br><br>|Appropriate credit must be given and the copyright notice must be retained when redistributing the work.|
|MathV360K(VQA-AS)|Apache-2.0<br><br>|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|CLEVR-Change<br><br>|–|–|

Table 19: Detailed overview of the licensing terms, usage restrictions for the OCR Category

|Dataset|License|Restrictions|
|---|---|---|
|K12 Printing<br><br>|Apache-2.0<br><br>|Appropriate credit must be given and any modified versions must be licensed under the same terms when sharing or adapting the work.|
|ArXiv OCR<br><br>|–<br><br>|–|
|HME<br><br>|–|–|
|VCR-Wiki|CC BY-SA 4.0<br><br>|Appropriate credit must be given and adaptations must be shared under the same license.|
|TextOCR|CC BY 4.0<br><br>|Appropriate credit must be given when sharing or adapting the work.|
|Sroie<br><br>|MIT|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|ICDAR-LSVT-zh<br><br>|–|–|
|ReCTs<br><br>|–|–|
|CTW|CC BY-NC-SA 4.0<br><br>|Appropriate credit must be given, no commercial use is permitted, and adaptations must be shared under the same license.|
|Rendered Text<br><br>|–|–|
|ICDAR2017<br><br>|–|–|
|Chrome-Writing<br><br>|–<br><br>|–|
|MTWI(zh)|–|–|
|Continued on next page| | |

Table 19 – continued from previous page

|Dataset|License|Restrictions|
|---|---|---|
|IAM<br><br>|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|ICDAR2019|MIT<br><br>|Appropriate credit must be given and the copyright and permission notice must be included when distributing the work.|
|Orand-Car-A|CC BY-NC-ND 4.0|Appropriate credit must be given, no commercial use is permitted, and no adaptations are allowed.|
|IIIT 5K<br><br>|–|–|

- H EVALUATION

We comprehensively evaluate our model across a wide range of benchmarks, which are grouped into three main categories: General VQA tasks, Table & Chart & OCR tasks, and Math & Reasoning tasks. The full list of benchmarks and their corresponding categories is presented in Tab. 20.

- H.1 EVALUATION DETAIL

Our model is evaluated under two distinct inference configurations to assess its capabilities under different conditions. For the non-thinking mode, we employ a deterministic setting with the temperature set to 0 and a maximum output length of 8,192 tokens. Conversely, for the thinking mode, which is designed to elicit more detailed reasoning, we set the temperature to 0.6 and increase the maximum token length to 16,384.

All experiments are conducted using our customized version of the VLMEvalKit (Duan et al., 2024) framework. To ensure a more comprehensive and accurate evaluation, we introduced several key modifications. First, we extended the framework’s LLM-based judging capabilities to several benchmarks that originally lacked this support, including ChartQA (Masry et al., 2022), InfoVQA (Mathew et al., 2022), DocVQA (Mathew et al., 2021), and CountBench (Paiss et al., 2023). Second, for the MathVerse (Zhang et al., 2024b) benchmark, we identified and addressed issues of inaccurate answer extraction and judging errors present in the original implementation. To enhance its robustness, we supplemented its judge with a broader set of test cases, thereby improving the reliability of the evaluation for this benchmark.

- H.2 PROMPT FOR LLM-BASED EVALUATION

Finally, to ensure transparency and reproducibility in our evaluation process, we provide the exact prompt from VLMEvalkit official (Duan et al., 2024) used for our LLM-based judging across the aforementioned benchmarks. This prompt constitutes a rigorous instruction set that directs the judge model to strictly evaluate the correctness of the candidate’s final answer against the standard answer, ignoring intermediate reasoning errors and accommodating various valid output formats. The full prompt is detailed below:

### Prompt for LLM-based Answer Grading

Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. Here are some evaluation criteria:

- 1. Please refer to the given standard answer. You don’t need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate’s answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
- 2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
- 3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate’s answer is correct. If the standard answer does not specify a unit, but the candidate’s answer includes a unit that is correct for the value given, consider it correct.
- 4. Some answers may consist of multiple items, such as multiple-choice questions, multipleselect questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
- 5. If the prediction is given with \boxed{}, please ignore the \boxed{} and only judge whether the candidate’s answer is consistent with the standard answer.
- 6. If the candidate’s answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can’t answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).

Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:

- A: CORRECT
- B: INCORRECT
- C: INVALID

Just return the letters “A”, “B”, or “C”, with no text around it. Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don’t apologize or correct yourself if there was a mistake; we are just trying to grade the answer.

<Original Question Begin>: {question} <Original Question End> <Standard Answer Begin>: {gold_answer} <Standard Answer End> <Candidate’s Answer Begin>: {llm_response} <Candidate’s Answer End>

Judging the correctness of the candidate’s answer:

- H.3 BENCHMARKS

We comprehensively evaluate our model across a wide range of benchmarks, which are grouped into three main categories: General VQA tasks, Table & Chart & OCR tasks, and Math & Reasoning tasks. The full list of benchmarks and their corresponding categories is presented in Tab. 20.

Table 20: The evaluation benchmarks used in our study, grouped into three main categories: General VQA, Table & Chart & OCR, and Math & Reasoning.

|Task|Benchmark|
|---|---|
|General VQA<br><br>|MMMU (Yue et al., 2024), AI2D (Kembhavi et al., 2016), MMStar (Chen et al., 2024c), MMVet (Yu et al., 2024b), HallusionBench (Guan et al., 2024), MMBench (Liu et al., 2024e), MMMU-Pro (Yue et al., 2025a), MMVP (Tong et al., 2024b), POPE (Li et al., 2023a) VisuLogic (Xu et al., 2025), RealWorldQA (xAI, 2024), CountBench (Paiss et al., 2023), BLINK (Fu et al., 2024), MMT-Bench (Ying et al., 2024), VLMs are Blind (Rahmanzadehgervi et al., 2024)|
|Table & Chart & OCR<br><br>|DocVQA (Mathew et al., 2021), InfoVQA (Mathew et al., 2022), CharXiv (Wang et al., 2024c), SEED-Bench2-Plus (Li et al., 2024a), ChartQA (Masry et al., 2022), OCRBench (Liu et al., 2024f)<br><br>|
|Math & Reasoning<br><br>|MathVista (Lu et al., 2024a), MathVision (Wang et al., 2024a), MathVerse (Zhang et al., 2024b), LogicVista (Xiao et al., 2024), WeMath (Qiao et al., 2025), DynaMath (Zou et al., 2025)|

- I PROMPT

In this section, we present the complete prompts used in our HoneyPipe. Specifically, two distinct prompts are utilized: one is for the Noise and Irrelevance Filtering process in Sec. 2.2, and the other is for the Fidelity Verification process in Sec. 2.3.

Prompt for Noise and Irrelevance Filtering Given the image and the [QUESTION x], your task is to determine:

- 1. Whether the image and question are relevant to each other;
- 2. Whether they form an appropriate question;
- 3. Whether there are any obvious issues with the question. Below are some examples of evaluation scenarios:

- 1. **Relevance Check**: Is the image related to the question?

- - Example 1: If the question is about “the dog in the photo,” the image should clearly depict a dog or something directly related to dogs.
- - Example 2: If the question asks about the diameter of a circle in the image, but the image does not contain a circle → filter out.
- - Example 3: If the image is completely unrelated to the question → filter out.

- 2. **Ambiguous or Vague Question**: If the question is too vague, unclear, or lacks sufficient context to establish a clear connection with the image → filter out.

- - Example 1: “What do you think about this?” without specifying what “this” refers to in the image’s context.
- - Example 2: Ambiguous question (e.g., unclear referents) → filter out.
- - Example 3: The question is not a question. (e.g., a list of numbers, a table, etc.) → filter out.

- 3. **Language Filter**: If the question contains languages other than Chinese or English → filter out. After evaluating the image and question, provide a clear decision:

- - false: If the image and question are relevant, the relationship is clear, and no issues are found.
- - true: If the image and question do not match, contain issues (e.g., ambiguity, contradiction, irrelevance, language errors), or fail any evaluation criteria.

**Output Format(json format)**: ```json {

“q0”: “true”, “q0_reason”: “Briefly explain the reason for filtering”,

...

} ```

where the returned result index starts from 0. If there are N questions, return the results of all questions(q0, q1, ..., qN). Here are the specific [QUESTION x] and image:

### Prompt for Fidelity Verification

**Task:** Evaluate whether the new answer [ANSWER x] should be retained or filtered compared to the original answer [ORI_ANSWER x] for [QUESTION x] according to the following guidelines. Provide the evaluation results in JSON format (where x is an index starting from 0).

**Evaluation Guidelines:**

- 1. **Open-Ended/Descriptive Questions**

- **Rule:** Answers are inherently diverse and subjective. Retain the new answer unless it is clearly irrelevant to the question.

- 2. **Precise Answer Questions**

- **Rule:** Retain the new answer if it is numerically equivalent to the original answer (e.g., different formats, minor rounding errors within 0.1, valid unit conversions). Filter it if there is a fundamental conflict in numerical values.

- 3. **Factual Questions**

- **Rule:** Retain the new answer if it conveys the same fact (rephrasing is allowed). Filter it if it introduces fabricated or irrelevant content.

- 4. **Conceptual or Common Sense Questions**

- **Rule:** Retain the new answer if the final conclusion or reasoning result is consistent, regardless of wording. Filter it if there is a logical contradiction or opposite judgment.

- 5. **Chart Analysis Questions**

- **Rule:** Retain the new answer if both reference consistent content in the chart. Filter it if it conflicts with the factual data in the chart.

- 6. **Logical Reasoning or Hypothetical Questions**

- **Rule:** Retain the new answer if the final conclusion is consistent, even if the steps differ. Filter it if the results are contradictory or contain logical errors.

- 7. **Sorting/Priority/Comparison Questions**

- - **Rule:** Retain the new answer if the relative order or judgment is consistent. Filter it if the new answer reverses or distorts the sorting/comparison result.

**Judgment Criteria:**

- - **false:** The new answer meets consistency requirements, supplements information, and should be retained.
- - **true:** The new answer is inconsistent, irrelevant, or contains factual errors, and should be filtered.

**Output Format (JSON):** ```json {

“q0”: “false”, “q0_reason”: “The new answer effectively rephrases the same factual information.”,

... “qN”: “true”, “qN_reason”: “The new answer introduces numerical values conflicting with the original precise answer.”

} ```

**Explanation:**

- - “false” indicates that the new answer should be retained (consistent), and “true” indicates that the new answer should be filtered (inconsistent or erroneous).
- - The index starts from 0 and must cover all questions (q0, q1, ..., qN).
- - ... Here are the specific [QUESTION x], [ANSWER x] and [ORI_ANSWER x]:

- J NOISE DATA CASE STUDY

- J.1 FORMAT FLAW

Some data exhibit format flaws, such as missing instructions or responses. Below is an example of data filtered out due to a missing response.

Case: Format flaw

[Figure 11]

User Question: What might be the implications of this lighting for a photographer?

Response: (None)

Filtering Reason: Only half conversation.

- J.2 LOW QUALITY IMAGE

Low-quality images, such as those with excessively low resolution, overly large aspect ratios, or blurriness, are detrimental to the model’s ability to extract high-quality visual features and semantic information, thereby impairing the model’s performance and effectiveness. Below is an example of an image filtered out due to excessively low resolution.

- J.3 TEXT REPETITION

Some responses generated by MLLMs may suffer from the issue of text repetition. Below is an example of data filtered out due to the inclusion of repeatedly occurring segments in the response. The first occurrence of a repeatedly appearing pattern is marked in red, and the second occurrence is marked in blue. More repeated occurrences have been omitted.

Case: Low-quality image

[Figure 12]

User Question:

In this task, you will classify traffic signs in the given image. The image is in color and contains various traffic signs captured in real-world conditions. Your goal is to accurately identify the type of traffic sign present in the image. Consider factors such as the sign’s shape, color, and symbols to ensure correct classification. A sample output should look like this: ‘Speed limit (80km/h)’ Options: (a) Go straight or right (b) Right-of-way at the next intersection (c) Dangerous curve to the left (d) Bicycles crossing (e) Speed limit (30km/h) (f) Speed limit (120km/h) (g) Speed limit (50km/h) (h) End of speed limit (80km/h) (i) Pedestrians (j) Ahead only

Response: (j) Ahead only

Filtering Reason: Image size too small: [26, 27]

Case: Text repetition

[Figure 13]

User Question:

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. Question: Determine the value of x. Options: (A) 12 (B) 15 (C) 18 (D) 21

### Response (part):

To determine the value of x, we need to analyze the given geometric configuration. The image shows a triangle with segments marked with expressions involving x and y.

...

x = −156

However, this result seems incorrect based on the provided options. Let’s re-evaluate the steps. Revisiting the substitution step:

3x − 12

5 3

2 − 9 Multiply every term by 6:

x + 11 =

10x + 66 = 9x − 36 − 54 ··· x = −156

This result is still incorrect. Let’s re-evaluate the initial setup and equations. Revisiting the initial equations:

5 3

x + 11 = 3y − 9 ···

3x − 12

5 3

2 − 9 Multiply every term by 6:

x + 11 =

10x + 66 = 9x − 36 − 54 ··· x = −156

This result is still incorrect. Let’s re-evaluate the initial setup and equations. Revisiting the initial equations:

5 3

x + 11 = 3y − 9

...

- J.4 IMAGE-INSTRUCTION MISMATCH

Some data contain instructions irrelevant to the associated images, which is detrimental to the MLLM’s ability to align visual features with semantic information. Below is an example of data filtered out due to the irrelevance between the image and its corresponding instruction.

Case: Image-instruction mismatch

[Figure 14]

User Question:

What natural elements contribute to the sense of tranquility in this landscape, and how does the play of light enhance the overall atmosphere?

Response:

The natural elements that contribute to the tranquility include the lush green meadow, vibrant wildflowers, and the gently flowing stream. These elements evoke a sense of peace and harmony with nature. The play of light, with sunlight filtering through the tree branches, creates a dynamic interplay of shadows and highlights, further enhancing the serene atmosphere by adding depth and visual interest to the landscape.

Filtering Reason:

The image shows a person on an escalator with a bag labeled ‘bloke,’ which is unrelated to natural elements or tranquility in a landscape. The question and image are not relevant to each other.

- J.5 UNANSWERABLE INSTRUCTION

Some data contain overly vague instruction descriptions, or the image-instruction pairs lack sufficient information, rendering the instructions unanswerable. However, it is important to note that well-articulated open-ended instructions will still be retained. Below is an example of an imageinstruction pair that was filtered out due to insufficient information caused by partial corruption of the image.

- J.6 LOW QUALITY INSTRUCTION

Some data contain images that are relevant to their corresponding instructions, but the instructions can be answered without extracting information from the images. Such data only leverages the LLM component of MLLMs and provides no benefit to the training of the vision encoder and projector. Below is an example of data filtered out because the instruction can be answered without extracting information from the image.

Case: Unanswerable instruction

[Figure 15]

User Question:

Based on the pie chart representing the distribution of parliamentary seats in Italy, which party has the second-largest share, and what percentage of seats does it hold?

Response: M5S, 25%

Filtering Reason:

The image is a pie chart, but it lacks labels or a legend to identify which colors correspond to which parties. Without this information, it is impossible to determine which party has the second-largest share or the percentage of seats it holds.

Case: Low quality instruction

[Figure 16]

User Question:

Who discovered the moon of Saturn called Titan in 1655? Answer the question using a single word or phrase.

Response: Huygens

Filtering Reason:

The image shows Saturn, but the question is about the discovery of Titan, which is not depicted in the image. The image does not provide any information relevant to answering the question.

- K QUALITATIVE ANALYSIS OF HONEY-DATA-15M

- K.1 FIDELITY VERIFICATION-FAILED DATA

The fidelity verification process in Sec. 2.3 filters out data where there are conflicts between the enriched response and the original response. Below is an example of a factual query being filtered out because the enriched response differs from the original response.

Case: Factual inaccuracy

[Figure 17]

User Question:

Hint: Please answer the question and provide the final answer at the end. Question: What is the radius of the circle O if the length of arc AB is 4*pi and the measure of angle AOB is 40 degrees?

Original Response: The answer is 9 Rewritten Response:

To find the radius of circle O, we can use the formula for the length of an arc. The formula for the length of an arc s is:

θ 360◦ × 2πr

s =

where θ is the central angle in degrees, and r is the radius of the circle. We know that the length of arc AB is 4π and the central angle ∠AOB is 40◦. Plugging these values into the formula:

40◦

360◦ × 2πr First, simplify 40

4π =

◦

360◦ to 19:

1 9 × 2πr

4π =

Simplify the right-hand side:

2πr 9

4π =

To solve for r, multiply both sides of the equation by 9:

36π = 2πr Then divide both sides by 2π:

36π 2π

r =

= 18

Final answer: The radius of circle O is 18. Filtering Reason:

The new answer introduces a numerical value 18 that conflicts with the original precise answer 9.

- K.2 SHOWCASE OF HIGH-QUALITY ENRICHED DATA

In this section, we showcase data from diverse topics within our Honey-Data-15M. A subset of these responses with long Chain-of-Thought reasoning. The enriched responses incorporate more in-depth knowledge, feature more accurate responses, and present more detailed reasoning processes.

### Case: Caption (with short CoT)

[Figure 18]

User Question: Can you describe the scene depicted in the image? Original Response: The image shows a tree in the desert with a sand dune in the background. Enriched Response:

The image depicts a serene and somewhat desolate desert scene with the following elements:

### 1. Desert Landscape:

- • The background features rolling sand dunes with a warm, orange hue, characteristic of a desert environment.
- • The sky is clear and light, suggesting a calm and open atmosphere.

### 2. Tree:

• In the foreground, there is a bare, leafless tree with twisted branches. The tree appears to be struggling or enduring harsh conditions, symbolizing resilience or waiting.

### 3. Text Overlay:

- • The text “Wait on the Lord” is prominently displayed in the center of the image, written in a clean, white font.
- • Below the main text, there is additional information: “Joshua 3 | October 11th, 2020,” indicating a reference to a biblical passage and a specific date, likely for a church service or event.
- • In the top left corner, the text “Immanuel Baptist Church” is visible, suggesting the image is related to a church event or sermon.

Interpretation: The scene conveys a sense of patience, endurance, and faith, as the desert and the bare tree symbolize challenging conditions. The text “Wait on the Lord” reinforces the theme of waiting and trusting in divine guidance, aligning with the biblical reference to Joshua 3. The overall tone is reflective and spiritual, encouraging viewers to find strength and hope in difficult times.

### Case: General (with short CoT)

[Figure 19]

User Question: Can you tell what the people are doing?

### Original Response:

The people appear to be either walking or standing on the street, perhaps waiting to cross the intersection or attending a public event.

### Enriched Response:

The image shows a large group of people gathered in a public space, likely on a street or near a building. Here’s what can be inferred about their activities:

- 1. Crowd Gathering: The people are standing in a large group, suggesting they are attending an event or gathering. The density of the crowd indicates it might be a public or organized event.
- 2. Event or Parade: The presence of traffic cones, barricades, and the organized nature of the crowd suggest this could be a parade, festival, or public demonstration. The “BEGIN” sign on the pole might indicate the start of a route or event.
- 3. Traffic Management: The traffic lights and signs (e.g., “ONLY,” “NO TURN ON RED”) indicate that the area is being managed to control traffic flow, likely to accommodate the event.
- 4. Casual Attire: The people are dressed in casual clothing, which is typical for outdoor public events.
- 5. Balloons and Decorations: The balloons visible in the background suggest a festive or celebratory occasion.

Overall, the people appear to be participating in or observing a public event, such as a parade, festival, or community gathering. The setup indicates careful planning to manage the crowd and traffic.

Case: Grounding & Counting (with short CoT)

[Figure 20]

User Question:

Think out loud before giving the final conclusion. Is there a large blue cube made of the same material as the small sphere?

### Original Response: yes

Enriched Response: To answer the question, let’s analyze the image step by step:

### 1. Identify the objects in the image:

- • There are two red cylinders.
- • There is a red metallic cube.
- • There is a small red sphere.
- • There is a brown matte cube.
- • There is a large blue cube.

### 2. Examine the material of the small sphere:

• The small sphere is red and appears to have a matte finish. This can be inferred from its lack of reflective properties and its uniform color.

### 3. Examine the material of the large blue cube:

• The large blue cube also appears to have a matte finish. It does not reflect light in a way that suggests a metallic or glossy surface.

### 4. Compare the materials:

- • The small red sphere has a matte finish.
- • The large blue cube also has a matte finish.
- • Since both objects share the same matte appearance, it is reasonable to conclude that they are made of the same material.

### 5. Final conclusion:

• There is a large blue cube, and it appears to be made of the same material as the small red sphere.

Final Answer: Yes, there is a large blue cube made of the same material as the small sphere.

Case: Chart (with long CoT)

[Figure 21]

User Question:

Which algorithm has the smallest accuracy summed across all the datasets? Please conclude your answer as Answer: xxx at the end if possible.

Enriched Response (Part): < think > So, let’s solve this problem. First, we need to find the accuracy of each algorithm (artery and steel) across all datasets (bird and poet), sum them, and then compare. First, look at the “artery” algorithm. The black bar (bird dataset) has a height of 1, and the gray bar (poet dataset) has a height of 7 (since total for artery is 7, and bird is 1, so poet is 7-1=6... < /think > To determine which algorithm has the smallest accuracy summed across all datasets, we analyze the two algorithms (artery and steel) by calculating the total accuracy across both datasets (bird and poet) for each:

- 1. Artery algorithm:

- • Accuracy on the bird dataset: The height of the black bar is 1.
- • Accuracy on the poet dataset: The height of the gray bar is 7 − 1 = 6 (since the total height of the artery bar is 7).
- • Sum of accuracies: 1 + 6 = 7.

- 2. Steel algorithm:

- • Accuracy on the bird dataset: The height of the black bar is 9.
- • Accuracy on the poet dataset: The height of the gray bar is 17 − 9 = 8 (since the total height of the steel bar is 17).
- • Sum of accuracies: 9 + 8 = 17.

Comparing the two sums (7 for artery and 17 for steel), the smaller sum belongs to the artery algorithm. Answer: artery

Case: STEM (with long CoT)

[Figure 22]

User Question:

In the right triangle ABC, ∠ACB = 90°, CD is the height from C to side AB, CE is the median to side AB, AD = 2, CE = 5, then CD = ( ) A. 2 B. 3 C. 4 D. 2√3 .

Enriched Response (Part): < think > Got it, let’s try to solve this problem step by step. First, let’s recall some properties of right triangles, medians, and altitudes. We have a right triangle ABC with ∠ACB = 90◦. CE is the median to AB, so in a right triangle, the median to the hypotenuse is half the hypotenuse. That’s a key property: in a right triangle, the median from the right angle to the hypotenuse is equal to half the

hypotenuse. So CE = 12AB. We know CE = 5, so AB = 2 × CE = 10... < /think > To solve the problem, we can use properties of right triangles, medians, and altitudes. Let’s

break down the steps:

- Step 1: Use the property of the median in a right triangle In a right triangle, the median to the hypotenuse is half the length of the hypotenuse. Given △ABC is a right triangle with ∠ACB = 90◦, and CE is the median to AB, we have:

CE =

- 1

- 2

AB Since CE = 5, substitute to find AB:

5 =

- 1

- 2

AB =⇒ AB = 10

- Step 2: Analyze segments on AB E is the midpoint of AB (because CE is a median), so AE = BE = AB2 = 5. We know AD = 2, so the segment DE (between D and E) is:

DE = AE − AD = 5 − 2 = 3

- Step 3: Use the Pythagorean theorem in △CDE CD is the altitude from C to AB, so CD ⊥ AB, meaning ∠CDE = 90◦. Thus, △CDE is a right triangle. In △CDE:

- • Leg DE = 3.
- • Hypotenuse CE = 5 (given)
- • Leg CD (unknown, let’s call it x)

By the Pythagorean theorem:

CD2 + DE2 = CE2 Substitute DE = 3 and CE = 5:

x2 + 32 = 52 =⇒ x2 + 9 = 25 =⇒ x2 = 16 =⇒ x = 4 Thus, CD = 4, and the answer is C.

### Case: Document (with long CoT)

[Figure 23]

User Question: Which service has the next renewal date after Spotify? Enriched Response: < think > So, let’s look at the renewal dates for each service. First, Spotify’s renewal date is November 1, 2023. Now we need to find the next one after that. Let’s list all renewal dates:

- • Spotify Premium: November 1, 2023
- • Netflix Standard Plan: November 7, 2023
- • Hulu (Ad-Supported): November 12, 2023
- • Adobe Creative Cloud: November 15, 2023
- • Amazon Prime: November 20, 2023

So after November 1, the next date is November 7, which is Netflix Standard Plan. < /think > To determine which service has the next renewal date after Spotify, we first identify Spotify’s renewal date and then compare it with the renewal dates of other services:

- 1. Locate Spotify Premium’s renewal date: November 1, 2023.
- 2. List the renewal dates of all other services:

- • Netflix Standard Plan: November 7, 2023
- • Hulu (Ad - Supported): November 12, 2023
- • Adobe Creative Cloud: November 15, 2023
- • Amazon Prime: November 20, 2023

Identify the earliest date after November 1, 2023: November 7, 2023, which corresponds to Netflix Standard Plan. Thus, the service with the next renewal date after Spotify is Netflix Standard Plan.

- L QUALITATIVE SHOWCASE OF BEE-8B

Case: STEM

[Figure 24]

User Question:

Given: In △ABC, the perpendicular bisectors of AB and AC intersect BC at points M and N, respectively. AB = 4, AC = 7, BC = 10. Find the perimeter of △AMN.

### Bee-8B Response:

< think > Got it, let’s try to solve this problem step by step. First, let’s recall what a perpendicular bisector is. A perpendicular bisector of a segment is a line that is perpendicular to the segment and divides it into two equal parts. So, if M is on the perpendicular bisector of AB, that means MA = MB. Similarly, N is on the perpendicular bisector of AC, so NA = NC. Now, we need to find the perimeter of triangle AMN. The perimeter of a triangle is the sum of its sides, so that would be AM + MN + AN. Let’s substitute the equal segments we just found. Since MA = MB and NA = NC, we can replace AM with MB and AN with NC. So the perimeter becomes MB + MN + NC... < /think > To solve the problem, we use the property of perpendicular bisectors: any point on the perpendicular bisector of a segment is equidistant from the segment’s two endpoints.

### Step 1: Analyze the Perpendicular Bisectors

- • M lies on the perpendicular bisector of AB, so MA = MB (by the perpendicular bisector theorem).
- • N lies on the perpendicular bisector of AC, so NA = NC (by the perpendicular bisector theorem).

#### Step 2: Express the Perimeter of △AMN The perimeter of △AMN is AM + MN + AN. Substitute AM = MB and AN = NC (from Step 1):

Perimeter of △AMN = MB + MN + NC

##### Step 3: Relate to BC Points M and N lie on BC, so MB + MN + NC is exactly the length of BC. We know BC = 10 (given). Thus:

Perimeter of △AMN = BC = 10 Answer: The perimeter of △AMN is 10.

Case: General

[Figure 25]

User Question:

This is an illusion image. What kind of visual illusions can it make people perceive? And why is it able to produce such effects?

Bee-8B Response: < think >... < /think > To analyze the visual illusions in this image and their underlying mechanisms, let’s break it down:

- 1. Types of Visual Illusions Perceived This image triggers ambiguous shape perception (a type of multistable illusion). Viewers can perceive the same visual contour in two distinct ways:

- • Interpretation 1: Two face profiles The left and right sides of the gray shape resemble the side profiles of two human faces (from forehead to chin). The central “connection” between the two profiles creates the illusion of two faces facing each other.
- • Interpretation 2: A goblet (chalice) shape The central curve and the extended top/bottom of the gray shape can also be read as the outline of a goblet: the central curve as the goblet’s bowl, and the top/bottom as the rim and stem.

- 2. Why It Produces These Effects The illusion arises from cognitive biases and visual perception principles, rooted in how the brain processes shape and familiarity:

- • Shape Similarity & Prior Experience Human faces and goblets share structural similarities in their curved contours (e.g., the smooth, continuous curves of a face’s profile match the curve of a goblet’s bowl). The brain prioritizes interpreting ambiguous shapes using prior experience—we evolved to recognize faces (a “biologically significant” shape for survival) and are also trained to recognize everyday objects like goblets. When a single contour matches both sets of familiar shapes, the brain generates multiple plausible interpretations.
- • Gestalt Psychology: “Ambiguity” and “Continuity” Gestalt principles explain how the brain organizes visual information into coherent wholes. Here, the continuity of the gray shape’s curve is split into two competing “wholes”: one resembling two faces (due to the brain’s bias for face-like patterns) and one resembling a goblet (due to the brain’s bias for object-like patterns). This split creates the illusion of simultaneous, conflicting interpretations.

### In short, the image leverages one contour’s ability to fit two familiar shapes (faces

+ goblet) and the brain’s innate tendency to prioritize familiar, evolutionarily important shapes (like faces) to create a multistable, ambiguous perception.

### Case: Chart

###### OCR

###### Grounding & Counting

4.9%

###### Document

5.1%

5.9%

###### General

### User Question:

###### STEM

36.8%

7.6%

If you want to build a new dataset based on this dataset, how would you adjust the distribution of the data?

Honey-Data 15M

15.1%

Caption

24.6%

Chart

Bee-8B Response: < think > So, let’s tackle this question. First, I need to understand the current distribution from the pie chart. The Honey-Data 15M dataset has segments: General (36.8%), Chart (24.6%), Caption (15.1%), STEM (7.6%), Document (5.9%), Grounding & Counting (5.1%), OCR (4.9%).... < /think > To adjust the data distribution when building a new dataset based on Honey-Data 15M, follow this structured approach, which accounts for the current distribution (from the pie chart) and the task’s requirements:

- 1. Analyze the Current Distribution ...
- 2. Define the New Dataset’s Purpose The distribution adjustment depends on the task the new dataset will support. For example:

- • Niche application: If the task targets a rare category (e.g., OCR for document analysis), prioritize increasing that category’s share.
- • Balanced general-purpose: If the task requires broad coverage, rebalance dominant categories (General, Chart) and enhance smaller ones.
- • Multi-modal focus: If the task combines categories (e.g., Chart + Caption for image-text models), balance those segments.

- 3. Select Categories to Adjust Determine which categories need oversampling (more data) or undersampling (less data):

- • Oversample underserved categories: If a category is critical but underrepresented (e.g., OCR at 4.9%), increase its proportion.
- • Undersample overrepresented categories: If a dominant category (General at 36.8%) is less relevant to the task, reduce its share.

- 4. Implement Adjustment Methods ...
- 5. Validate the Adjusted Distribution ... Example: Adjusting for an OCR-Focused Task If the new dataset targets OCR (currently 4.9%):

- 1. Oversample OCR: Collect more OCR samples (e.g., scanned documents, handwritten notes) and augment existing ones (e.g., add noise, change fonts).
- 2. Undersample General: Reduce the General segment to free up space for OCR (e.g., scale General to 25% instead of 36.8%).
- 3. Balance remaining categories: Adjust other segments (e.g., Chart to 25%, Caption to 10%) to fit the new total (100%).

By aligning the distribution with the task’s goals and using data-driven methods, the new dataset will better support the intended application while maintaining robustness.

### Case: Grounding & Counting

[Figure 26]

User Question: How many monitors are there?

Bee-8B Response: < think >... < /think > To determine the number of monitors, we analyze the image by counting each distinct monitor from left to right:

- 1. The first monitor (leftmost, with a textured black front).
- 2. The second monitor (black).
- 3. The third monitor (silver).
- 4. The fourth monitor (white - bordered).
- 5. The fifth monitor (black, behind the white - bordered one).
- 6. The sixth monitor (black).
- 7. The seventh monitor (black).
- 8. The eighth monitor (rightmost, black).

Answer: 8

