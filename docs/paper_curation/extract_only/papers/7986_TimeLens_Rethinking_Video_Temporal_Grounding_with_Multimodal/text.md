# arXiv:2512.14698v2[cs.CV]26Mar2026

#### TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs

Jun Zhang1,2,* Teng Wang2, Yuying Ge2 Yixiao Ge2 Xinhao Li1 Ying Shan2 Limin Wang1,3,

1Nanjing University 2ARC Lab, Tencent PCG 3Shanghai AI Lab https://timelens-arc-lab.github.io/

###### Abstract

This paper does not introduce a novel method but instead establishes a straightforward, incremental, yet essential baseline for video temporal grounding (VTG), a core capability in video understanding. While multimodal large language models (MLLMs) excel at various video understanding tasks, the recipes for optimizing them for VTG remain under-explored. In this paper, we present TimeLens, a systematic investigation into building MLLMs with strong VTG ability, along two primary dimensions: data quality and algorithmic design. We first expose critical quality issues in existing VTG benchmarks and introduce TimeLens-Bench, comprising meticulously re-annotated versions of three popular benchmarks with strict quality criteria. Our analysis reveals dramatic model re-rankings compared to legacy benchmarks, confirming the unreliability of prior evaluation standards. We also address noisy training data through an automated re-annotation pipeline, yielding TimeLens-100K, a large-scale, high-quality training dataset. Building on our data foundation, we conduct in-depth explorations of algorithmic design principles, yielding a series of meaningful insights and effective yet efficient practices. These include interleaved textual encoding for time representation, a thinking-free reinforcement learning with verifiable rewards (RLVR) approach as the training paradigm, and carefully designed recipes for RLVR training. These efforts culminate in TimeLens models, a family of MLLMs with state-of-the-art VTG performance among open-source models and even surpass proprietary models such as GPT-5 and Gemini-2.5-Flash. All codes, data, and models will be released to facilitate future research.

###### 1. Introduction

Recent multimodal large language models (MLLMs) have excelled at understanding “what” happens in a video, yet they largely fail when asked “when.” This limitation is central to

Corresponding author. * Work done during internship at ARC Lab, Tencent PCG.

Data Quality Algorithmic Design

[Figure 1]

[Figure 2]

l Position embedding l Visual overlay l Textual encoding

✘ Low-quality annotation

Benchmark Diagnosis

Time Representation

✘ Unreliable evaluation

[Figure 3]

l Manual refinement l Cross validation l Re-Annotation

[Figure 4]

l RLVR vs. SFT? l w/ or w/o thinking? l Multi-stage training?

Benchmark Refinement

Training Paradigm

[Figure 5]

[Figure 6]

[Figure 7]

Event uniqueness Event existence Query clarity Annotation accuracy

###### Reliable Evaluation Suite

l Training data l How long to train ? l How to sample data?

[Figure 8]

Training Recipe

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Best Practices

TimeLens Models

TimeLens-Bench & TimeLens-Train-100K

Figure 1. Overview of the proposed TimeLens framework. We systematically explore the key factors for building performant video temporal grounding models, dissecting our efforts along two primary dimensions: data quality and algorithmic design. For data quality, we focus on benchmark diagnosis, benchmark refinement, and creating a reliable evaluation suite. For algorithmic design, we study various aspects including time encoding, training recipes, and optimization strategies to establish best practices and develop the TimeLens models.

the task of video temporal grounding (VTG). The challenge is twofold: 1) VTG necessitates a fundamental shift from coarse semantic aggregation to fine-grained time-aware perception; 2) Distinguishing queried events requires modeling long-term visual dynamics over appearance-centric features, which are notoriously difficult to annotate and learn. As MLLMs become integral to perception [43, 44, 55, 58] and reasoning systems [6, 13, 37, 39, 40, 66], equipping them with robust temporal awareness is no longer optional, but essential [26, 35, 46, 49, 54].

This work focuses on post-training MLLMs with leading temporal grounding ability. This investigation is a straightforward extension given the recent progress in pretrained foundation MLLMs [2, 3, 53]. Different from heavily studied general understanding tasks, recipes for fine-grained grounding tasks are not yet to be established. This paper aims to systematically investigate core components of building timeaware MLLMs (Fig. 1) along two primary dimensions: data quality and algorithmic design.

Our investigation starts by exposing critical flaws in

60

###### mIoUonRefinedBenchmark(%)

50

Proprietary Models

Open-Source Models

40

30

25 35 45 55 65

mIoU on Original Benchmark (%)

(a) (b)

- Figure 2. (a) Impact of data quality on model evaluation. A comparison of Mean IoU on original versus our refined Charades-STA benchmarks. The deviation from the diagonal line shows that legacy benchmarks are misleading, as they inflate the results of some open-source models while underestimating proprietary ones. (b) Cumulative performance gains of TimeLens explorations. This analysis shows how each component boosts the model’s average performance on TimeLens-Bench. From data curation to thinking-free RLVR with early stopping and difficulty-based data sampling, each step demonstrates a clear positive impact towards our final TimeLens model. TimeLens-7B and TimeLens-8B are based on Qwen2.5-VL-7B and Qwen3-VL-8B, respectively.

evaluation benchmarks. We find that existing VTG benchmarks [11, 25, 27] not only lack a clear comparison between leading proprietary and open-source models but are also rife with low-quality queries and erroneous timestamps. This noisy data may render current leaderboards misleading and misguide research efforts. To rectify this, we undertook a meticulous data overhaul. We first defined strict criteria for query and timestamp quality, in terms of uniqueness, existence, clarity, and accuracy. We then manually re-annotated three popular datasets (Charades-STA [11], ActivityNet Captions [25], QVHighlights [27]) to create TimeLens-Bench, a rigorously cross-validated benchmark. As shown in Fig. 2a, the necessity of this correction is confirmed by a dramatic re-ranking of models on TimeLens-Bench compared to their performance on legacy benchmarks, proving the unreliability of prior evaluation standards. Beyond evaluation, we also fix the noisy training data by automated re-annotation, yielding TimeLens-100K, a large-scale, high-quality training dataset.

With our curated data suite as a solid foundation, we conduct in-depth explorations on the algorithmic design principles from three key aspects. First, for timestamp representation, we discover that a simple yet effective interleaved textual encoding strategy outperforms more complex alternatives. Second, we determine that VTG is fundamentally a perception-driven task, and thus employ a pure thinkingfree reinforcement learning with verifiable rewards (RLVR) approach that outperforms other training paradigms in both efficiency and performance. Finally, our detailed analysis of RLVR training reveals two key recipes for both performance and training efficiency: (1) early stopping when re-

ward metrics plateau, and (2) difficulty-based data sampling. By integrating these insights and design principles, we ultimately develop TimeLens models, a family of MLLMs with superior VTG capability. As shown in Fig. 2b, our model achieves state-of-the-art performance among opensource models and even surpasses proprietary models such as GPT-5 and Gemini-2.5-Flash.

Through these efforts, we identified and addressed longoverlooked quality issues in existing datasets, and derived a series of insights and best practices in algorithmic design. We hope TimeLens can serve as a solid foundation in both data curation and algorithmic design principles, to facilitate future research on building MLLMs with strong VTG capabilities. Our code, data, and models will be open-sourced.

###### 2. Related Work

Temporal Grounding Datasets. Numerous VTG datasets have been proposed, spanning diverse domains [14, 22, 25, 27, 41, 45, 50]. Early works [11, 38, 65] trained and evaluated models on the training and test splits of a single benchmark [25, 45] to assess their ability to fit single-domain data distribution. In recent works [17, 37, 46], large diverse corpuses composed of multiple different source datasets [1, 22, 36, 41, 50, 60] are aggregated for training, and a suite of distinct benchmarks [11, 25, 27] are used to probe the models’ real-world cross-domain generalizability.

However, the critical issue of data quality has been overlooked. There lacks a systematic examination on whether existing datasets are reliable enough for training and evaluation. In this paper, we manually inspect existing datasets,

###### Error Case: Multiple Event Occurences

###### Error Case: No Event Occurrence

###### Error Case: Duplicate Queries

[Figure 14]

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Query 1: A person sits in a chair.

✘ Original Query: A man is running down a track by a field.

✘ Original Query: Man in gray top walks from outside to inside.

✘ Query 2 (Duplicate): The person sits in the chair momentarily.

Re-annotated: A man kneels on the ground with both knees.

Re-annotated: Three people are crossing the crosswalk.

[Figure 27]

[Figure 28]

Re-annotated: A man moves the position of a chair.

[Figure 29]

###### Error Case: Inaccurate Annotation

###### Error Case: Unclear Query

Query: Some friends have a birthday meal together around a large table at a restaurant.

|[Figure 30]|
|---|

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

|[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]|
|---|

[Figure 39]

[Figure 40]

[Figure 41]

✘ Original Query: When all of the lather is gone, the process continues until the boot is finally dry.

✘ Original Annotation: 122s 132s

Re-annotated: A man is tying shoelaces on a shoe.

[Figure 42]

Refined Annotation: 121s 150s

[Figure 43]

- Figure 3. Qualitative examples of errors and fixes. We present representative errors identified in existing datasets, spanning different error types, including multiple event occurrences, no event occurrence, duplicate queries for the same video, unclear query, and inaccurate annotation. Through our rigorous manual refinement, these errors have been properly corrected, significantly improving data quality.

identify and correct errors, and produce quality-improved training and evaluation suites for developing more practical VTG models.

- • Event existence. The event described in the text query must genuinely exist within the video content.
- • Query uniqueness. All queries must be unique in a single video. The presence of multiple nearly identical queries describing the same event is equivalent to duplicating or weighting certain samples, leading to biased metrics. Indeed, this issue is severe in Charades-STA dataset.
- • Avoid information leakage in queries. Queries like “ending credits” leak their temporal position, allowing the model to answer via shortcut, without truly “grounding” the query over the entire video. However, annotators tend to label such queries since they are easy to identify.

MLLMs for Temporal Grounding. Substantial works focus on algorithmic designs to improve MLLMs’ VTG capability. One line of research explores model architectures, including token compression methods to reduce computation on long videos [46, 61], timestamp encoding strategies to align the timestamps of each frame with its corresponding features [5, 12, 28, 34, 51, 55, 62]. Another line of works investigate training strategies: introducing various supervised fine-tuning tasks to improve VTG performance [7, 61], or designing verifiable rewards to improve performance via reinforcement learning [4, 33, 54, 59].

Output Criteria. The temporal segment should satisfy:

- • Annotation precision. The annotated event boundaries should be precise, excluding any subsegments that do not conform to the query’s description.
- • Annotation exhaustiveness. There should be no other time segments outside the annotated one that also satisfy the query’s description.

Despite the abundance of proposed designs, their inconsistent experimental settings make it difficult to fairly compare their relative merits and establish best practices. In this paper, we systematically analyze these design choices using our quality-assured training and evaluation suites, offering key insights for improving MLLMs’ VTG capability.

###### 3.2. Manual Auditing and Refinement

###### 3. Towards Reliable, High-Quality VTG Data

We introduce a rigorous and efficient pipeline for auditing and refining existing temporal grounding datasets.

###### 3.1. Annotation Criteria

Diagnose-then-Refine. Our pipeline follows a diagnosethen-refine workflow. Given a video-query pair from existing datasets, annotators first carefully review the video to identify potential errors against the criteria in Sec. 3.1. If an error is detected, they select the error category, then either revise the query or choose a new valid event to describe. Subsequently, the precise temporal segment is annotated. The core principle is that the same annotator performs both error detection and subsequent correction, which not only improves efficiency but also strengthens annotators’ awareness

Task Formulation. For temporal grounding, a model takes as input a video v and a text query q, localizes the event E described by q, and outputs the corresponding temporal segment S = (tstart,tend). In practice, a video is typically annotated with one or more query-segment pairs {(qi,Si)}ni=1. Input Criteria. The input video and query should satisfy: • Query clarity and specificity. The query must be clear, pre-

cise, and unambiguous for accurate and definitive grounding (A counterexample like “the game continues”).

Charades-STA ActivityNet Captions QVHighlights

Figure 4. Statistics of errors indicating alarmingly high proportion of errors in existing datasets.

of potential errors, thereby reducing the risk of introducing similar ones.

Error Identification. Directly applying the abstract criteria from Sec. 3.1 for error detection proves overly challenging for annotators. Therefore, as shown in Fig. 3, we derive from these criteria a set of concrete, easily identifiable error types with clear illustrations. Annotators check whether each error type is present and fill in the corresponding information. Additionally, we group all queries from the same video together to detect violations of “query uniqueness” and improve annotation efficiency. During the process, we do not provide original temporal segments to annotators.

Quality Control. Upon completion of each small data batch, every sample is assigned to a different annotator for cross-validation and error correction. If the error rate in a batch exceeds a threshold, the entire batch is rejected for re-annotation and then validated again. For annotator selection and training, we sampled a small subset of data for trial annotation with over a dozen vendors, then selected the vendor with the highest quality and consistency. Before formal annotation, we provided a detailed handbook and conducted several training sessions. The annotation interface and detailed manual are provided in Sec. G of the appendix.

###### 3.3. Empirical Analysis on TimeLens-Bench

In this section, we present our efforts and findings by applying the above annotation pipeline to existing datasets. We focus on three most widely-used temporal grounding benchmarks: Charades-STA [11], ActivityNet Captions [25], and QVHighlights [27]. These datasets exhibit diversity across video domains, video durations, and query semantics. They are all manually annotated and generally considered the highest-quality VTG datasets available. Therefore, analyzing them offers a representative view of the quality and issues prevalent in existing data. Through diagnosis and refinement, we release TimeLens-Bench, comprising refined versions of the three aforementioned benchmarks: Charades-TimeLens, ActivityNet-TimeLens, and QVHighlights-TimeLens. Together, they form a comprehensive evaluation suite that combines diversity with high quality. Detailed statistics for these

benchmarks are provided in Sec. B.

Finding 1: Widely-used benchmarks have an alarmingly high proportion of errors.

Error Statistics and Analysis. As shown in Fig. 4, we observe an alarmingly high proportion of errors across different categories in these benchmarks. The distribution of error composition varies across different datasets, yet all datasets exhibit consistently high overall error rates. For example, in Charades-STA, we find that 20.6% of samples violate query uniqueness, while 34.9% exhibit annotation accuracy issues. Such severe errors will lead to unreliable evaluation results and misguide research efforts.

Qualitative Examples of Errors and Fixes. As shown in Fig. 3, various error examples are identified in existing datasets, including multiple event occurrences, no event occurrence, duplicate queries for the same video, unclear query, and inaccurate annotation. Through our rigorous manual refinement, these detected errors have been properly corrected, significantly improving data quality. Our refined datasets provide more reliable evaluation results.

Finding 2: Low-quality evaluation data inflates performance of open-source models, while underestimating proprietary models.

Counter-intuitive Evaluation Results. We evaluate various frontier models on both the original and refined benchmarks, observing drastically contrasting performance trends. As illustrated in Fig. 2a, on the original benchmarks, we observe a surprising phenomenon: frontier proprietary models like Gemini-2.5-Pro [8] receive poor scores, whereas opensource models [3, 54] attain significantly higher ones. Conversely, on our refined benchmarks, this trend reverses. The proprietary models exhibit much better results, though with room for improvement, while the open-source models suffer a substantial performance degradation, lagging far behind their proprietary counterparts. This reversal indicates that the original benchmarks produce misleading results due to

Charades-TimeLens ActivityNet-TimeLens QVHighlights-TimeLens Model

R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoU Proprietary Models

- GPT-4o [23] 60.6 44.5 23.5 41.8 55.2 41.4 25.8 40.4 69.0 54.8 38.5 52.1

- GPT-5 [42] 59.3 42.0 22.0 40.5 57.4 44.9 30.4 42.9 72.4 60.4 46.4 56.8 Gemini-2.0-Flash [8] 66.4 53.5 27.1 46.7 62.9 54.0 37.7 49.3 76.2 66.4 48.3 60.8 Gemini-2.5-Flash [8] 68.7 56.1 30.6 48.6 66.8 57.5 41.3 52.5 78.2 69.4 55.0 64.3 Gemini-2.5-Pro [8] 74.1 61.1 34.0 52.8 72.3 64.2 47.1 58.1 84.1 75.9 61.1 70.4

Open-Source Models

VideoChat-Flash-7B [31] 60.2 37.9 17.8 39.7 35.5 21.8 10.5 24.8 45.2 30.6 16.7 32.7 VideoChat-R1-7B [32] 51.9 30.8 11.7 33.7 35.0 23.9 11.3 25.0 29.3 19.1 9.4 21.5 Time-R1-7B [54] 57.9 32.0 16.9 36.6 44.8 31.0 19.0 33.1 65.8 51.5 36.1 49.2 TRACE [17] 37.2 21.8 9.6 27.1 43.4 33.9 22.0 32.7 49.7 39.1 28.1 39.0 TRACE-uni [17] 38.2 22.9 10.4 28.1 44.3 35.1 22.6 33.6 49.9 40.0 29.2 39.8 TimeSuite [61] 56.3 35.5 18.0 38.1 27.1 17.5 8.6 19.8 27.1 16.9 9.9 21.7 Grounded-VideoLLM [51] 43.3 28.7 13.5 30.0 39.2 29.6 19.5 30.0 43.7 33.8 22.5 33.4 MiMo-VL-7B [9] 57.9 42.6 20.5 39.6 49.3 38.7 22.4 35.5 57.1 42.6 28.4 41.5 Qwen2.5-VL-7B [3] 59.7 37.8 16.6 39.3 44.1 31.0 16.1 31.4 41.5 27.8 15.2 31.6

- TimeLens-7B 70.5 55.6 28.4 48.8 62.8 51.0 32.6 46.2 74.1 62.7 43.1 56.0 Qwen3-VL-235B-A22B [2] 71.7 50.8 24.5 47.8 69.0 57.5 39.3 52.2 79.6 70.2 54.5 64.6 Qwen3-VL-8B [2] 69.2 53.4 27.5 48.3 62.1 51.2 34.4 46.8 74.2 64.6 49.3 59.4

- TimeLens-8B 76.6 63.0 35.2 55.2 68.9 58.4 40.6 53.2 80.2 71.6 55.5 65.5

- Table 1. Main Results. We benchmark the performance of various state-of-the-art proprietary and open-source models on TimeLens-Bench. Our TimeLens models are built upon their respective baseline models (preceding rows in the table). Our TimeLens-7B not only delivers substantial improvements over the Qwen2.5-VL baseline but also closes the gap with the more powerful Qwen3-VL-8B model. Building upon the stronger Qwen3-VL baseline, our TimeLens-8B pushes performance even further, setting a new state-of-the-art among open-source models and surpassing prominent proprietary models like GPT-5 and Gemini-2.5-Flash.

inherent quality flaws, while our refined benchmarks yield results that align more closely with real-world user experience, providing reliable evaluation for developing better VTG models.

###### 3.4. Training Data Re-annotation

By applying our manual pipeline from Sec. 3.2 to a sampled subset of existing VTG training corpus [1, 22, 41, 50, 60], we found that the training data exhibits an even higher error rate compared to the evaluation benchmarks. This motivated us to refine training data based on scalable re-annotation. Given the vast scale of the training sets, we employ an automated pipeline to improve their quality based on advanced multimodal models. Owing to the poor quality of these training datasets, especially the high proportion of queries that fail to meet our criteria in Sec. 3.1, we re-annotate the videos rather than refining existing labels. Through this process, we curate TimeLens-100K, a large-scale, high-quality, and diverse VTG training set. Additional details are provided in Sec. H.

Finding 3: Improved annotation quality in training data yields stronger grounding ability.

As presented in Fig. 2b, models trained on TimeLens-

100K demonstrate substantially improved performance on

- our refined evaluation benchmarks. This performance gain serves as a direct validation of the data’s enhanced quality. Notably, our automated re-annotation for training data is developed entirely independently of the manual benchmark refinement process, ensuring an unbiased evaluation.

4. Benchmarking Grounding MLLMs

In this section, we benchmark the performance of vari-

- ous state-of-the-art proprietary and open-source models on TimeLens-Bench, including our TimeLens models derived from the exploration in Sec. 5.

Evaluation Metrics. We evaluate VTG performance using the “R1@m” metric, which measures the proportion of test instances where the highest-ranked predicted segment achieves an IoU exceeding threshold m (where m takes values from 0.3, 0.5, 0.7). Additionally, we employ mIoU as a primary measure, computing the mean IoU across the entire test set for conciseness.

Evaluation Results. As shown in Tab. 1, we observe a significant performance gap between existing open-source and proprietary models, and our TimeLens models substantially narrow this gap. TimeLens-7B delivers substantial improvements over its baseline, demonstrating the effective-

LLM

LLM

LLM

…

0

4

8

RoPE IDs

###### …

…

| | | | |
|---|---|---|---|
|Vision|Encoder|& Proje<br><br>|ctor|
| | | | |

| | | | |
|---|---|---|---|
|Visi|on Enco|der & P|rojector|
| | | | |

| | | | |
|---|---|---|---|
|Visi|on Enco|der & P|rojector|
| | | | |

[Figure 44]

[Figure 45]

[Figure 46]

… 1s 2s 3s

Text Tokenizer

[Figure 47]

[Figure 48]

[Figure 49]

Align

[Figure 50]

[Figure 51]

[Figure 52]

…

…

2s 3s …

1s …

…

2s 3s

1s

1s

2s

3s

(a) Interleaved Textual Timestamp Encoding.

(b) Visual Timestamp Overlay.

(c) Position-embedding-based Time Encoding.

- Figure 5. Illustration of different timestamp encoding schemes. (a) Textual Encoding uses the text tokenizer of LLMs to tokenize textual timestamps into textual tokens. (b) Visual Overlay directly overlays timestamps as visual text onto the corresponding frames. (c) Position-embedding-based Methods aligns the positional encodings of visual tokens in the LLM with the sampling time of each frame.

Charades-TimeLens ActivityNet-TimeLens QVHighlights-TimeLens Method

Timestamp

Format R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoU Position Embed. [3] - 57.9 32.0 16.9 36.6 44.8 31.0 19.0 33.1 65.8 51.5 36.1 49.2

Frame Index 65.5 48.0 22.2 44.0 47.5 34.0 17.8 33.3 61.4 43.6 24.1 42.3 Visual Overlay

Raw Timestamp 67.6 50.7 25.7 46.3 54.0 42.2 26.3 39.8 70.0 58.3 42.1 53.6

Not-Interleaved Textual Prefix

Raw Timestamp 64.9 49.4 27.3 45.8 48.2 35.5 21.4 35.2 59.7 45.8 26.4 42.8

Interleaved Frame Index 66.0 51.6 25.6 45.6 51.0 39.1 23.1 36.9 64.4 52.1 32.3 47.2 Textual Prefix Raw Timestamp 70.0 53.9 28.1 48.3 57.9 46.3 30.5 43.1 73.0 62.2 46.1 56.7

- Table 2. Ablation on timestamp encoding methods. For each method, we experiment with two timestamp formats: raw timestamps (e.g., “10.2s”) or frame indices (e.g., “1, 2, 3”). “Position Embed.” means “Position Embedding”. Results show that interleaved textual prefix with raw timestamps is the most effective approach, while maintaining simplicity.

ness of the insights and best practices obtained from our experiments in Sec. 5. It surpasses strong open-source competitors such as Time-R1-7B [54] and MiMo-VL-7B [9], as well as proprietary models like GPT-4o [23] and GPT-5 [42]. More remarkably, on the already stronger baseline Qwen3VL-8B, our TimeLens-8B model still achieves substantial performance gains, establishing a new state-of-the-art among open-source models and even surpassing frontier proprietary models like Gemini-2.5-Flash [8].

Bench for evaluation and TimeLens-100K for training. To ensure rigor, all ablation studies are based on the final, bestperforming model configuration, isolating the impact of a single design choice at a time. Due to limited computational resources, we adopt a lower per-frame resolution for our ablation experiments. More implementation details are provided in Sec. C of the appendix.

###### 5.1. Timestamp Encoding

Finding 4: Encoding timestamps as interleaved textual prefix is the most effective while maintaining simplicity.

###### 5. Exploring Algorithmic Designs

In this section, we conduct a systematic study on the algorithmic designs for improving MLLMs’ VTG performance, covering various aspects from model architectures to training strategies. Leveraging our high-quality training and evaluation suites as a reliable testbed, we derive several novel and valuable insights. As shown in Fig. 2b, each of our findings contributes a non-trivial performance gain, ultimately culminating in our TimeLens model.

To enable MLLMs to perform temporal grounding, a critical design decision is timestamp encoding (i.e., aligning the timestamp of each frame with its corresponding features). Effective timestamp encoding allows the model to accurately perceive the absolute temporal position of each frame and the relative order between frames, thereby producing precise localization results. As illustrated in Fig. 5, various timestamp encoding strategies have been proposed:

Experimental Setup. Our experiments use Qwen2.5-VL7B [3] as the baseline. For RLVR experiments, we employ GRPO [47] as optimization method. We use TimeLens-

• Position-embedding based. These methods adapt position embeddings in LLMs to represent the temporal position of each frame. For example, MRoPE [3, 9] and 3D RoPE [48]

Training Charades-TimeLens ActivityNet-TimeLens QVHighlights-TimeLens Paradigm

Training

Time R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoU SFT (32K Data) 1.0× 68.8 53.0 26.2 47.4 53.3 42.6 27.5 39.9 65.8 54.8 40.6 52.0 SFT (100K Data) 2.4× 70.6 54.9 27.1 48.6 53.2 43.1 27.2 39.7 63.1 51.1 36.9 49.0 Thinking-based RLVR 1.9× 60.3 46.4 24.7 42.7 54.3 44.2 29.1 41.2 72.1 62.7 48.2 57.8 SFT + Thinking-free RLVR

2.9× 71.7 56.7 29.8 50.1 56.9 46.1 30.1 42.7 72.2 60.6 43.8 55.9

###### Thinking-free RLVR 1.0× 70.0 53.9 28.1 48.3 57.9 46.3 30.5 43.1 73.0 62.2 46.1 56.7

- Table 3. Ablation on different training paradigms. We compare the performance and efficiency of different training paradigms, showing that thinking-free RLVR achieves the best performance while maintaining high efficiency. All training is conducted on our quality-improved TimeLens-100K training data. Training time is measured on 8× H20 GPUs, where 1.0× corresponds to approximately 4h10m. As described in Sec. 5.3, before RLVR training, offline inference on the training data is required to select samples with appropriate difficulty; this time is also included in the reported RLVR training time.

extend pure-text RoPE to multimodal scenarios, encoding the spatial and temporal dimensions of video frame tokens.

- • Visual overlay. These methods [7, 12, 55] directly overlay timestamps or frame index onto each frame, enabling MLLMs to “read” the temporal position through their OCR capabilities.
- • Textual encoding. These methods convert timestamps into text tokens using the MLLM’s text tokenizer. There are two main variants: the Interleaved approach [5, 15, 20, 34, 56] in Fig. 5a inserts timestamp tokens before the visual tokens of each frame. In contrast, the Non-interleaved approach [29, 31, 52] adds an instruction like “This video

samples N frames of a T-second video at t1,t2,... seconds.” into the prompt.

We conduct a comprehensive comparison of different timestamp encoding methods. For each method, we experiment with two timestamp formats: raw timestamps (e.g., “10.2s”) or frame indices (e.g., “1, 2, 3”), which are simpler but neglects the temporal interval between frames. As shown in Tab. 2, our results reveal: Position-embedding based methods yield unsatisfactory results. Given that they require fundamental modifications to the RoPE mechanism in LLMs, their practicality is limited without large-scale retraining. Instead, interleaved textual prefix with raw timestamps achieves the best performance among all approaches, while remaining simple and intuitive.

###### 5.2. Optimization Paradigms

Finding 5: For the optimization paradigm, a pure thinking-free RLVR approach achieves superior performance and efficiency. Both SFT and thinking-based RLVR are not necessary.

In this section, we review different training paradigms and conduct systematic experiments to compare their effectiveness and efficiency for VTG, seeking insights into the opti-

mal training paradigm.

Earlier works [17, 18, 22, 46, 61] employ supervised finetuning (SFT) to improve MLLMs’ VTG capability. Recently, some works [4, 54] utilize reinforcement learning with verifiable rewards (RLVR), following a “think-then-answer” approach [16] (details in Sec. C): during sampling, the model first generates an explicit thinking process and then produces the final answer. The task-specific VTG accuracy reward is computed only on the final answer. Despite these efforts, there lacks a systematic comparison of the respective merits of these methods, leaving some key questions unanswered:

- • Is RLVR superior to SFT? While the pioneering work Time-R1 [54] demonstrates that RLVR outperforms SFT, they compare the two methods using the same amount of training data, despite RLVR requiring significantly more training time. A fair comparison under equal training budgets remains absent.
- • Is explicit “thinking” necessary for RLVR? Recent works suggest that the thinking process is not essential when applying RLVR to visual perception such as counting [30, 43]. Whether this holds for VTG, a predominantly perception-oriented task, remains unanswered.
- • Does a preceding SFT phase benefit RLVR? An SFT phase prior to RLVR is typically employed to enhance the model’s capability and facilitate subsequent RLVR training [9, 48]. However, whether this preceding SFT phase actually improves final performance in the VTG scenario remains unexplored.

In Tab. 3, we compare the performance and efficiency of different training paradigms. Our results reveal that thinkingfree RLVR surpasses both SFT and thinking-based RLVR in performance while being more efficient. Adding a preceding SFT phase before RLVR yields no significant performance gain compared to pure RLVR. Overall, a pure thinking-free RLVR approach maintains simplicity, superior performance, and high efficiency.

0.6

60

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

55

0.5

###### EvaluationMetrics(%)

###### Reward/RewardStd

50

0.4

45

0.3

40

0.2

35

0.1

30

200 400 600 800 1.0K 1.2K 1.4K

Training Step

Reward Reward Std Avg R1@0.5 Avg mIOU

- Figure 6. The effectiveness of early stopping for RLVR. We show the trends of training reward and evaluation metrics during RLVR training. When the temporal IoU reward and the withingroup reward standard deviation plateau, performance reaches its peak. Continued training beyond this point leads to performance degradation. Therefore, performing early stopping when the reward plateaus ensures optimal training efficiency and performance. Training is conducted on ∼12K samples selected from TimeLens100K via difficulty-aware sampling.

###### 5.3. Recipes for RLVR Training

Building on the finding in Sec. 5.2 that thinking-free RLVR is the optimal training paradigm, in this section, we further explore effective recipes for RLVR training, focusing on two key questions: (i) How long should we train? (ii) How to effectively sample training data?

Finding 6: For RLVR training, performing early stopping when reward metrics plateau saves computational cost, while preventing performance degradation.

How long should we train? In SFT, the prevailing wisdom is “train longer, generalize better” [19]. Given training data with sufficient scale and quality, we typically train MLLMs for at least one full epoch over the entire dataset, ensuring the model sees as much data as possible to enhance generalization. However, whether this strategy is optimal for RLVR remains to be explored.

In Fig. 6, we conduct RLVR training on ∼12K data from TimeLens-100K, tracking the reward and evaluating model checkpoints at different training steps on our evaluation benchmarks. When the temporal IoU reward and the withingroup reward standard deviation plateau, model performance has reached its peak. Continued training beyond this point leads to performance degradation. Therefore, in RL training, even with sufficiently high data quality, training for a full epoch over all available data is suboptimal. A good practice

65

60

Avg R1@0.3 Avg R1@0.5

Avg R1@0.7

55

Avg mIOU

Performance(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

45

40

35

30

0.4 0.5 0.6 0.7 0.8

Average Difficulty of Filtered Training Data

Figure 7. The importance of difficulty-based training data sampling for RLVR. We investigate the impact of training data with different difficulty levels on performance by adjusting the mean of the Gaussian distribution for difficulty-based data sampling. Model performance improves as the average sample difficulty increases, and eventually plateaus when difficulty becomes high. This demonstrates that selecting samples with sufficiently high difficulty is crucial for achieving optimal performance.

is performing early stopping when reward metrics plateau, which not only saves computational cost but also prevents performance degradation.

Finding 7: For RLVR training, sampling training data with sufficiently high difficulty relative to the model is crucial for performance.

How to sample training data? For RLVR training, it is crucial to select samples with appropriate difficulty relative to the model, and many works propose assessing training sample difficulty and employing difficulty-aware sampling [21, 54, 57]. To evaluate the impact of sample difficulty on video temporal grounding, we conduct experiments on our TimeLens-100K high-quality training corpus. Following prior works [54, 57], we use the model to be trained to perform offline inference on the training data, compute IoU metrics to estimate sample difficulty, and then perform Gaussian sampling based on sample difficulty (details in Sec. C). By varying the mean of the Gaussian distribution, we obtain training sets with different difficulty levels relative to the model, and conduct RLVR training on each set independently.

As shown in Fig. 7, model performance improves as the average sample difficulty increases, and eventually plateaus when difficulty becomes sufficiently high (over 0.75). This trend demonstrates that selecting training samples with sufficiently high difficulty relative to the model is crucial for achieving optimal performance.

Acknowledgements. This work is supported by the National Key R&D Program of China (No. 2022ZD0160900), the Basic Research Program of Jiangsu (No. BK20250009), the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (No. JYB2025XDXM118), and the Collaborative Innovation Center of Novel Software Technology and Industrialization.

###### References

- [1] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, pages 5803– 5812, 2017. 2, 5, 17
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631,

2025. 1, 5, 15, 18

- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 4, 5, 6, 13, 14, 15, 16, 18
- [4] Ruizhe Chen, Zhiting Fan, Tianze Luo, Heqing Zou, Zhaopeng Feng, Guiyang Xie, Hansheng Zhang, Zhuochen Wang, Zuozhu Liu, and Huaijian Zhang. Datasets and recipes for video temporal grounding via reinforcement learning. arXiv preprint arXiv:2507.18100, 2025. 3, 7
- [5] Shimin Chen, Xiaohan Lan, Yitian Yuan, Zequn Jie, and Lin Ma. Timemarker: A versatile video-llm for long and short video understanding with superior temporal localization ability. arXiv preprint arXiv:2411.18211, 2024. 3, 7
- [6] Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, et al. Scaling rl to long videos. arXiv preprint arXiv:2507.07966, 2025. 1
- [7] Jen-Hao Cheng, Vivian Wang, Huayu Wang, Huapeng Zhou, Yi-Hao Peng, Hou-I Liu, Hsiang-Wei Huang, Kuang-Ming Chen, Cheng-Yen Yang, Wenhao Chai, et al. Tempura: Temporal event masked prediction and understanding for reasoning in action. arXiv preprint arXiv:2505.01583, 2025. 3, 7
- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing

- the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 4, 5, 6, 14, 15, 16, 17
- [9] Team Core, Zihao Yue, Zhenru Lin, Yifan Song, Weikun Wang, Shuhuai Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, et al. Mimo-vl technical report. arXiv preprint arXiv:2506.03569, 2025. 5, 6, 7, 14, 15, 18
- [10] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24108–24118, 2025. 16
- [11] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, pages 5267–5275, 2017. 2, 4, 12
- [12] Yuying Ge, Yixiao Ge, Chen Li, Teng Wang, Junfu Pu, Yizhuo Li, Lu Qiu, Jin Ma, Lisheng Duan, Xinyu Zuo, et al. Archunyuan-video-7b: Structured video comprehension of realworld shorts. arXiv preprint arXiv:2507.20939, 2025. 3, 7
- [13] Sara Ghazanfari, Francesco Croce, Nicolas Flammarion, Prashanth Krishnamurthy, Farshad Khorrami, and Siddharth Garg. Chain-of-frames: Advancing video understanding in multimodal llms via frame-aware reasoning. arXiv preprint arXiv:2506.00318, 2025. 1
- [14] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022. 2
- [15] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025. 7
- [16] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 7, 13
- [17] Yongxin Guo, Jingyu Liu, Mingda Li, Qingbin Liu, Xi Chen, and Xiaoying Tang. Trace: Temporal grounding video llm via causal event modeling. arXiv preprint arXiv:2410.05643,

- 2024. 2, 5, 7

[18] Yongxin Guo, Jingyu Liu, Mingda Li, Dingxin Cheng, Xiaoying Tang, Dianbo Sui, Qingbin Liu, Xi Chen, and Kevin Zhao. Vtg-llm: Integrating timestamp knowledge into video llms for enhanced video temporal grounding. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3302–3310,

- 2025. 7, 12

- [19] Elad Hoffer, Itay Hubara, and Daniel Soudry. Train longer, generalize better: closing the generalization gap in large batch training of neural networks. Advances in neural information processing systems, 30, 2017. 8

- [20] Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500,

2024. 7

- [21] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507, 2025. 8, 13
- [22] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14271–14280, 2024. 2, 5, 7, 17
- [23] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 5, 6, 14, 16, 17
- [24] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions. Advances in Neural Information Processing Systems, 37:48955–48970, 2024. 17
- [25] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision, pages 706–715, 2017. 2, 4, 12
- [26] Xiaohan Lan, Yitian Yuan, Xin Wang, Zhi Wang, and Wenwu Zhu. A survey on temporal sentence grounding in videos. ACM Transactions on Multimedia Computing, Communications and Applications, 19(2):1–33, 2023. 1
- [27] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. Advances in Neural Information Processing Systems, 34: 11846–11858, 2021. 2, 4, 12
- [28] Hongyu Li, Jinyu Chen, Ziyu Wei, Shaofei Huang, Tianrui Hui, Jialin Gao, Xiaoming Wei, and Si Liu. Llava-st: A multimodal large language model for fine-grained spatialtemporal understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8592–8603, 2025. 3
- [29] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 7
- [30] Ming Li, Jike Zhong, Shitian Zhao, Yuxiang Lai, Haoquan Zhang, Wang Bill Zhu, and Kaipeng Zhang. Think or not think: A study of explicit thinking in rule-based visual reinforcement fine-tuning. arXiv preprint arXiv:2503.16188,

2025. 7

- [31] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574, 2024. 5, 7, 14, 18
- [32] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang.

- Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025. 5, 14, 16, 18
- [33] Yunheng Li, Jing Cheng, Shaoyong Jia, Hangyi Kuang, Shaohui Jiao, Qibin Hou, and Ming-Ming Cheng. Tempsamp-r1: Effective temporal sampling with reinforcement fine-tuning for video llms. arXiv preprint arXiv:2509.18056, 2025. 3
- [34] Zeqian Li, Shangzhe Di, Zhonghua Zhai, Weilin Huang, Yanfeng Wang, and Weidi Xie. Universal video temporal grounding with generative multi-modal large language models. arXiv preprint arXiv:2506.18883, 2025. 3, 7
- [35] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. Univtg: Towards unified video-language temporal grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2794–2804,

2023. 1

- [36] Ye Liu, Zongyang Ma, Zhongang Qi, Yang Wu, Ying Shan, and Chang W Chen. Et bench: Towards open-ended eventlevel video-language understanding. Advances in Neural Information Processing Systems, 37:32076–32110, 2024. 2
- [37] Ye Liu, Kevin Qinghong Lin, Chang Wen Chen, and Mike Zheng Shou. Videomind: A chain-of-lora agent for long video reasoning. arXiv preprint arXiv:2503.13444, 2025. 1, 2, 12
- [38] Chujie Lu, Long Chen, Chilie Tan, Xiaolin Li, and Jun Xiao. Debug: A dense bottom-up grounding approach for natural language video localization. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5144–5153,

2019. 2

- [39] Arsha Nagrani, Mingda Zhang, Ramin Mehran, Rachel Hornung, Nitesh Bharadwaj Gundavarapu, Nilpa Jha, Austin Myers, Xingyi Zhou, Boqing Gong, Cordelia Schmid, et al. Neptune: The long orbit to benchmarking long video understanding. arXiv preprint arXiv:2412.09582, 2024. 1
- [40] Arsha Nagrani, Sachit Menon, Ahmet Iscen, Shyamal Buch, Ramin Mehran, Nilpa Jha, Anja Hauth, Yukun Zhu, Carl Vondrick, Mikhail Sirotenko, et al. Minerva: Evaluating complex video reasoning. arXiv preprint arXiv:2505.00681,

2025. 1

- [41] Andreea-Maria Oncescu, Joao F Henriques, Yang Liu, Andrew Zisserman, and Samuel Albanie. Queryd: A video dataset with high-quality text and audio narrations. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2265–

2269. IEEE, 2021. 2, 5, 17

- [42] OpenAI. Introducing gpt-5, 2025. Available from OpenAI announcement, August 7, 2025. 5, 6, 14, 17
- [43] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36:42748–42761, 2023. 1, 7
- [44] Yukun Qi, Yiming Zhao, Yu Zeng, Xikun Bao, Wenxuan Huang, Lin Chen, Zehui Chen, Jie Zhao, Zhongang Qi, and

- Feng Zhao. Vcr-bench: A comprehensive evaluation framework for video chain-of-thought reasoning. arXiv preprint arXiv:2504.07956, 2025. 1
- [45] Michaela Regneri, Marcus Rohrbach, Dominikus Wetzel, Stefan Thater, Bernt Schiele, and Manfred Pinkal. Grounding action descriptions in videos. Transactions of the Association for Computational Linguistics, 1:25–36, 2013. 2
- [46] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14313–14323, 2024. 1, 2, 3, 7
- [47] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 6, 12
- [48] Kwai Keye Team, Biao Yang, Bin Wen, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl technical report. arXiv preprint arXiv:2507.01949, 2025. 6, 7
- [49] Vidi Team, Celong Liu, Chia-Wen Kuo, Dawei Du, Fan Chen, Guang Chen, Jiamin Yuan, Lingxi Zhang, Lu Guo, Lusha Li, et al. Vidi: Large multimodal models for video understanding and editing. arXiv preprint arXiv:2504.15681, 2025. 1, 16
- [50] Alex Jinpeng Wang, Linjie Li, Kevin Qinghong Lin, Jianfeng Wang, Kevin Lin, Zhengyuan Yang, Lijuan Wang, and Mike Zheng Shou. Cosmo: Contrastive streamlined multimodal model with interleaved pre-training. arXiv preprint arXiv:2401.00849, 2024. 2, 5, 17
- [51] Haibo Wang, Zhiyang Xu, Yu Cheng, Shizhe Diao, Yufan Zhou, Yixin Cao, Qifan Wang, Weifeng Ge, and Lifu Huang. Grounded-videollm: Sharpening fine-grained temporal grounding in video large language models. arXiv preprint arXiv:2410.03290, 2024. 3, 5
- [52] Jiankang Wang, Zhihan Zhang, Zhihang Liu, Yang Li, Jiannan Ge, Hongtao Xie, and Yongdong Zhang. Spacevllm: Endowing multimodal large language model with spatio-temporal video grounding capability. arXiv preprint arXiv:2503.13983,

2025. 7

- [53] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, pages 396–416. Springer, 2024. 1
- [54] Ye Wang, Ziheng Wang, Boshen Xu, Yang Du, Kejun Lin, Zihan Xiao, Zihao Yue, Jianzhong Ju, Liang Zhang, Dingyi Yang, et al. Time-r1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025. 1, 3, 4, 5, 6, 7, 8, 13, 14, 17, 18
- [55] Yongliang Wu, Xinting Hu, Yuyang Sun, Yizhou Zhou, Wenbo Zhu, Fengyun Rao, Bernt Schiele, and Xu Yang. Number it: Temporal grounding videos like flipping manga. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13754–13765, 2025. 1, 3, 7
- [56] Linli Yao, Haoning Wu, Kun Ouyang, Yuanxing Zhang, Caiming Xiong, Bei Chen, Xu Sun, and Junnan Li. Generative

- frame sampler for long video understanding. arXiv preprint arXiv:2503.09146, 2025. 7
- [57] Ruifeng Yuan, Chenghao Xiao, Sicong Leng, Jianyu Wang, Long Li, Weiwen Xu, Hou Pong Chan, Deli Zhao, Tingyang Xu, Zhongyu Wei, et al. Vl-cogito: Progressive curriculum reinforcement learning for advanced multimodal reasoning. arXiv preprint arXiv:2507.22607, 2025. 8, 13
- [58] Yitian Yuan, Xiaohan Lan, Xin Wang, Long Chen, Zhi Wang, and Wenwu Zhu. A closer look at temporal sentence grounding in videos: Dataset and metric. In Proceedings of the 2nd international workshop on human-centric multimedia analysis, pages 13–21, 2021. 1
- [59] Feng Yue, Zhaoxing Zhang, Junming Jiao, Zhengyu Liang, Shiwen Cao, Feifei Zhang, and Rong Shen. Tempo-r0: A video-mllm for temporal video grounding through efficient temporal sensing reinforcement learning. arXiv preprint arXiv:2507.04702, 2025. 3
- [60] Abhay Zala, Jaemin Cho, Satwik Kottur, Xilun Chen, Barlas Oguz, Yashar Mehdad, and Mohit Bansal. Hierarchical videomoment retrieval and step-captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23056–23065, 2023. 2, 5, 17
- [61] Xiangyu Zeng, Kunchang Li, Chenting Wang, Xinhao Li, Tianxiang Jiang, Ziang Yan, Songze Li, Yansong Shi, Zhengrong Yue, Yi Wang, et al. Timesuite: Improving mllms for long video understanding via grounded tuning. arXiv preprint arXiv:2410.19702, 2024. 3, 5, 7
- [62] Yingsen Zeng, Zepeng Huang, Yujie Zhong, Chengjian Feng, Jie Hu, Lin Ma, and Yang Liu. Distime: Distribution-based time representation for video large language models. arXiv preprint arXiv:2505.24329, 2025. 3
- [63] Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, et al. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. arXiv preprint arXiv:2407.03320, 2024. 17
- [64] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 17
- [65] Songyang Zhang, Houwen Peng, Jianlong Fu, and Jiebo Luo. Learning 2d temporal adjacent networks for moment localization with natural language. In Proceedings of the AAAI conference on artificial intelligence, pages 12870–12877, 2020. 2
- [66] Yongheng Zhang, Xu Liu, Ruihan Tao, Qiguang Chen, Hao Fei, Wanxiang Che, and Libo Qin. Vitcot: Video-text interleaved chain-of-thought for boosting video understanding in large language models. arXiv preprint arXiv:2507.09876,

2025. 1

#### TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs Supplementary Material

###### A. Conclusion

In this work, we presented TimeLens, a systematic investigation into building MLLMs with robust video temporal grounding capabilities. On the data front, we first exposed severe quality issues in existing VTG benchmarks. Through meticulous manual re-annotation guided by strict quality criteria, we created TimeLens-Bench, a reliable evaluation suite that dramatically reshapes model rankings and provides trustworthy evaluation for future research. We also developed an automated re-annotation pipeline for noisy training data, yielding TimeLens-100K, a large-scale, high-quality training dataset. On the algorithmic front, our comprehensive exploration yielded several key insights, which culminate in TimeLens models, a family of MLLMs with state-of-theart VTG performance among open-source models and even surpasses leading proprietary models like GPT-5 and Gemini2.5-Flash. By open-sourcing our code, data, and models, we hope TimeLens can serve as a strong foundation for building MLLMs with stronger temporal video grounding capability.

###### B. Details of TimeLens-Bench

In this section, we provide details on our proposed TimeLensBench, a comprehensive, high-quality evaluation benchmark for video temporal grounding (VTG), comprising refined versions of three mainstream benchmarks: Charades-TimeLens, ActivityNet-TimeLens, and QVHighlights-TimeLens.

Source datasets. We construct our TimeLens-Bench based on Charades-STA [11], ActivityNet Captions [25], and QVHighlights [27]. For Charades-STA and ActivityNet Captions, we utilize their test splits, while for QVHighlights, we use the validation split, as its test split annotations are not publicly available and prior works [18, 37] also adopt the validation split. Since the test set of ActivityNet Captions is excessively large, resulting in prohibitively high evaluation cost, we uniformly partition videos based on their duration and sample an equal number of videos from each duration bin. This yields a subset with a video count comparable to Charades-STA and QVHighlights, while maintaining a balanced distribution of video durations. Although QVHighlights was originally annotated for both video temporal grounding and video highlight detection, our work focuses exclusively on the temporal grounding task.

Detailed Statistics. In Tab. 4, we present detailed statistics of our TimeLens-Bench and its source dataset counterparts. For each source dataset, annotations with high-quality queries had their corresponding temporal segments refined. For the remaining annotations with low-quality queries, we

Avg. Duration

# Rewritten Queries

# Refined Time Segments

Dataset # Videos

# Annotations

Domain

Charades-STA [11] 1334 29.5 3720 - - Daily Life Charades-TimeLens 1313 29.6 3363 2467 896 Daily Life

ActivityNet Captions [25] 4885 118.1 17031 - - Activity ActivityNet-TimeLens 1455 134.9 4500 3137 1363 Activity

QVHighlights [27] 1529 149.6 1542 - - Mixed QVHighlights-TimeLens 1511 149.6 1541 859 682 Mixed

Total (TimeLens-Bench) 4279 107.8 9404 6463 2941 Mixed

Table 4. Statistics of the datasets in our proposed TimeLens-Bench, compared against their original versions (Charades-STA [11], ActivityNet Captions [25] and QVHighlights [27]).

either revised the queries or rewrote them entirely. A small fraction of queries that were deemed unfixable were subsequently discarded. Overall, TimeLens-Bench comprises a total of 4,279 videos with an average duration of 107.8 seconds, and 9,404 annotations.

###### B.1. Evaluation Metrics

We evaluate model performance on TimeLens-Bench using four standard metrics: Recall@1 at IoU thresholds of 0.3, 0.5, and 0.7 (denoted as R1@0.3, R1@0.5, R1@0.7), and mean Intersection over Union (mIoU).

- • mIoU is defined as the average of the temporal Intersection over Union (IoU) scores between the predicted and groundtruth segments across all test samples.
- • R1@m measures the percentage of samples for which the temporal IoU of the prediction exceeds a given threshold m.

While TimeLens-Bench can be treated as a single unified benchmark for computing the aforementioned metrics, we compute and report metrics separately on its three constituent benchmarks to enable a more fine-grained analysis of model performance across different domains. We encourage future works to adopt this evaluation protocol.

###### C. More Implementation Details. C.1. Preliminaries for RL

Thinking-based vs. Thinking-free RLVR.. We formalize the distinction between thinking-based and thinking-free RLVR paradigms using GRPO [47] as the reinforcement learning algorithm.

In the task of video temporal grounding, given a video v and query q, the model generates a response y. For GRPO training, for each input pair (v,q) in the training set D, we sample a group of G responses {y(i)}Gi=1 from the policy πθ, compute their rewards {r(y(i))}Gi=1, and optimize the policy

Charades-TimeLens ActivityNet-TimeLens QVHighlights-TimeLens Training Data

R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoU

Original Noisy Data 52.6 30.4 14.0 35.6 45.0 29.5 16.0 31.3 61.3 46.1 29.1 44.6 TimeLens-100K 70.0 53.9 28.1 48.3 57.9 46.3 30.5 43.1 73.0 62.2 46.1 56.7

Table 5. Ablation on training data. Our TimeLens-100K training set significantly improves model performance, validating its enhanced quality.

Charades-TimeLens ActivityNet-TimeLens QVHighlights-TimeLens Model

R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoU Qwen2.5-VL-3B [3] (Baseline) 51.3 30.5 14.2 33.9 25.3 18.2 9.4 18.4 27.4 19.3 10.6 20.9

+TimeLens 63.5 48.1 23.9 43.3 56.6 44.9 27.4 41.2 71.9 58.9 37.8 52.9 Qwen2.5-VL-7B [3] (Baseline) 59.7 37.8 16.6 39.3 44.1 31.0 16.1 31.4 41.5 27.8 15.2 31.6

###### +TimeLens 70.5 55.6 28.4 48.8 62.8 51.0 32.6 46.2 74.1 62.7 43.1 56.0

- Table 6. Results across different model sizes. The best and second-best results are highlighted in bold and underlined, respectively.

to maximize the relative advantage within the group:

LGRPO = −E(v,q)∼DEy(i)∼πθ A(i) log πθ(y(i)|v,q) ,

(1) where the advantage is computed as:

1 G

A(i) = r(y(i)) −

G

r(y(j)). (2)

j=1

The key distinction between the two paradigms lies in the response structure and reward computation. In thinking-based RLVR, following the “think-then-answer” approach [16], the response consists of two parts:

y = [ythinking,yanswer], (3)

where ythinking represents the explicit reasoning process and yanswer contains the predicted temporal segment Sˆ = (tˆstart,tˆend). The reward function combines accuracy and format compliance:

###### r(y) = racc(yanswer) + rformat(y), (4)

where racc(yanswer) = IoU(S,Sˆ ∗) measures the temporal intersection-over-union with the ground truth segment S∗, and rformat(y) ensures proper output formatting following the “think-then-answer” structure.

In contrast, our thinking-free RLVR directly generates the answer without explicit reasoning:

y = yanswer, (5) with a simplified reward based solely on grounding accuracy: r(y) = racc(y) = IoU(S,Sˆ ∗). (6)

As shown in Tab. 3, the thinking-free paradigm eliminates the need for explicit reasoning generation and format

reward engineering, leading to simpler mplementation, faster training and inference, and superior performance.

Difficulty-aware Sampling. To formalize difficulty-aware sampling [21, 54, 57], we first perform offline inference with the model to be trained on the training dataset D = {(vi,qi,Si∗)}Ni=1. For each sample, we obtain the predicted segment Sˆi and compute the difficulty estimate as:

di = 1 − IoU(Sˆi,Si∗), (7)

where higher values indicate more challenging samples for the current model.

We then compute sampling weights for each sample based on its difficulty. Following [54, 57], we employ Gaussian sampling to construct a training subset where samples with difficulty around a target mean µ are more likely to be selected. Let g(d;µ,σ2) denote the target Gaussian distribution:

g(d;µ,σ2) =

(d − µ)2 2σ2

- 1

√

- 2πσ2

exp −

. (8)

To ensure that samples with difficulty d are selected with probability proportional to g(d), we compute the sampling weight for each sample i as:

g(di;µ,σ2) pˆ(di)

, (9)

wi =

where pˆ(di) is the empirical density of samples with difficulty di in the original dataset. This density correction ensures that the difficulty distribution of the sampled subset follows the target Gaussian distribution, rather than being biased by the original difficulty distribution in D.

By varying the mean µ of the Gaussian distribution, we obtain training sets with different average difficulty levels and conduct RLVR training on each independently to evaluate the impact of sample difficulty on final model performance in Sec. 5.3.

Charades-TimeLens Charades-STA Model

R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoU Proprietary Models

- GPT-4o [23] 60.6 44.5 23.5 41.8 51.6 27.9 11.7 34.7

- GPT-5 [42] 59.3 42.0 22.0 40.5 39.7 18.3 6.2 28.4 Gemini-2.0-Flash [8] 66.4 53.5 27.1 46.7 55.6 29.0 9.5 35.1 Gemini-2.5-Flash [8] 68.7 56.1 30.6 48.6 47.0 21.8 7.1 31.1 Gemini-2.5-Pro [8] 74.1 61.1 34.0 52.8 53.9 25.5 8.8 34.6

Open-Source Models

VideoChat-Flash-7B [31] 60.2 37.9 17.8 39.7 72.5* 51.4* 26.4* 45.2* VideoChat-R1-7B [32] 51.9 30.8 11.7 33.7 - 71.7* 50.2* 60.8* Time-R1-7B [54] 57.9 32.0 16.9 36.6 78.1* 60.8* 35.3* 58.1* MiMo-VL-7B [9] 57.9 42.6 20.5 39.6 - - - 50.0* Qwen2.5-VL-7B [3] (Baseline) 59.7 37.8 16.6 39.3 59.4 38.2 18.1 43.6* TimeLens-7B 70.5 55.6 28.4 48.8 70.7 39.8 14.5 42.3

- Table 7. Results on our refined Charades-TimeLens and the original Charades-STA benchmark. * indicates results from the original paper, other results are from our evaluation.

ActivityNet-TimeLens ActivityNet-Captions Model

R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoU Proprietary Models

Gemini-2.0-Flash [8] 62.9 54.0 37.7 49.3 50.4 33.2 19.9 36.5 Gemini-2.5-Flash [8] 66.8 57.5 41.3 52.5 51.2 34.7 21.0 37.4

Open-Source Models VideoChat-R1-7B [32] 35.0 23.9 11.3 25.0 - 33.3* 16.7* 35.5* Time-R1-7B [54] 44.8 31.0 19.0 33.1 58.1* 39.0* 21.4* 40.5* MiMo-VL-7B [9] 49.3 38.7 22.4 35.5 39.3 24.3 12.9 28.1 Qwen2.5-VL-7B [3] (Baseline) 44.1 31.0 16.1 31.4 34.5 20.8 11.2 26.4 TimeLens-7B 62.8 51.0 32.6 46.2 53.5 35.2 19.7 37.7

- Table 8. Results on our refined ActivityNet-TimeLens and the original ActivityNet-Captions benchmark. * indicates results from the original paper, other results are from our evaluation.

###### C.2. Experimental Setup

Unless otherwise specified, all experiments are conducted using Qwen2.5-VL-7B [3] as the base model. Under the Qwen2.5-VL architecture, every two consecutive video frames are merged during the patch-embedding stage of the vision encoder.

Model Configuration. We sample video frames at 2 FPS. For all ablation experiments, we set the minimum number of tokens per frame (i.e., every two merged frames) to min tokens = 16, and the maximum number of tokens for the entire video to total tokens = 3584. Under this configuration, the model adaptively adjusts the spatial resolution based on the video’s duration and raw resolution. For the final TimeLens models presented in Tab. 1, we scale the resolution budget to min tokens = 64 and total tokens = 14336. Under this setting, for a 110second video, the maximum resolution per frame is about 320 × 320 pixels.

Timestamp Encoding. In Tab. 2 and Sec. 5.1, we experiment with different timestamp encoding methods. For

position-embedding methods, we directly adopt the original MRoPE implementation from Qwen2.5-VL [3]. For Visual Overlay, we render timestamps in red with a font size of 40pt, overlaid at the bottom-left corner of each frame. For Non-interleaved Textual Encoding, an instruction like “This video samples N frames of a T-second video at t1,t2,... seconds” is prepended to the prompt.

For Interleaved Textual Encoding, timestamps are converted to text with one decimal place retained (e.g., 10.2 seconds), then tokenized and prepended to the tokens of each frame. As described above, when processing videos, every two consecutive frames are merged during the patchembedding stage of the vision encoder, while for images, each image is duplicated into two identical copies for merging. Since we insert textual timestamp tokens into the original video tokens to form an interleaved visual-text sequence, we treat each video frame as an independent image and duplicate it for processing. This approach allows us to bypass the original MRoPE mechanism entirely, enabling an isolated comparison between Interleaved Textual Encoding and MRoPE. Meanwhile, we adopt 1 FPS sampling to ensure the

QVHighlights-TimeLens QVHighlights Model

R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoU Proprietary

Gemini-2.0-Flash [8] 76.2 66.4 48.3 60.8 72.1 58.1 41.9 54.9 Gemini-2.5-Flash [8] 78.2 69.4 55.0 64.3 76.9 62.6 46.7 59.1

Open-Source

MiMo-VL-7B [9] 57.1 42.6 28.4 41.5 59.6 43.3 24.7 41.8 Qwen2.5-VL-7B [3] 41.5 27.8 15.2 31.6 29.2 19.2 11.9 27.4 TimeLens-7B 74.1 62.7 43.1 56.0 78.4 63.6 43.7 57.4

###### Table 9. Results on our refined QVHighlights-TimeLens and the original QVHighlights benchmark. All results are from our evaluation.

[Figure 53]

- (a)

|<think> The user is navigating through the settings menu oftheir LG smartphone during the time frame from57.3s to 99.7s. This is evident from the navigationthrough various settings pages and selections onthe screen. </think>|
|---|

|<think> The blue tractor driving through the vineyard is akey element of the scene, indicating the harvestingprocess. This vehicle's presence is consistentthroughout the relevant timestamp, clearly showingits action within the vineyard context. </think>|
|---|

- (b)

Figure 8. Observations during thinking-based RLVR training. (a) The model’s thinking length gradually decreases during training. (b) The model generates simple thinking content, primarily perception-related, without exhibiting any complex reasoning.

computational cost matches that of other temporal encoding methods using 2 FPS sampling.

Training Configuration. For all training procedures, we freeze the vision encoder while updating all other parameters, and train for one epoch. For supervised fine-tuning (SFT) experiments, we use a batch size of 128 and a learning rate of 1 × 10−5. For reinforcement learning (RL) experiments, we perform difficulty-aware data sampling with a Gaussian distribution where µ = 0.05 and σ = 0.2. The training batch size is 8, each prompt samples 8 roll-outs, the learning rate is 1 × 10−6, and the KL coefficient β is set to 0. We train until we observe that the reward plateaus and then perform early stopping. In practice, this corresponds to approximately 310 training steps (∼2.5K training examples) for Qwen2.5-VL

models.

Throughout the development of this work, our experiments were conducted based on Qwen2.5-VL. More recently, the more powerful Qwen3-VL [2] models have been released, so we also validated the effectiveness of our data and recipe on Qwen3-VL. We observed that directly applying RL training on Qwen3-VL fails to yield improvements, likely because Qwen3-VL has undergone large-scale multi-task RL training that includes VTG data, preventing the model from generating rollouts with sufficient diversity on VTG task during our continual RL. Therefore, we first perform a small SFT stage to, in a sense, revert the model back to the “base model” state before RL. This is merely a workaround specific to Qwen3-VL, a model that has already acquired

strong VTG capabilities through an RL stage similar to that proposed in this paper. In the common scenario, our recipes are designed to enhance the VTG capabilities of a “base MLLM”, where this trick is not required.

###### D. More Experimental Results

Results across different model sizes. In Tab. 6, we demonstrate the effectiveness of our proposed design principles across various model sizes. Across base models of varying sizes, TimeLens consistently delivers significant performance gains. Remarkably, despite having fewer parameters, TimeLens-3B substantially surpasses even the larger Qwen2.5-VL-7B model.

Generalization to an external benchmark. To further validate that our recipe generalizes beyond TimeLensBench, we additionally evaluate on VUE-TR, a high-quality human-annotated VTG benchmark introduced concurrently in Vidi [49]. As shown in Tab. 10, TimeLens-7B achieves the best result among the compared models, indicating that the gains from our data curation and algorithmic recipe are not limited to our own refined benchmark suite.

Model IoU @ VUE-TR (0-200s)

Qwen2.5-VL-7B [3] 36.0 GPT-4o [23] 34.5 Gemini-2.5-Pro [8] 41.6 TimeLens-7B 45.1

- Table 10. Results on the VUE-TR benchmark from Vidi [49]. TimeLens-7B achieves the best IoU among the compared models on this external benchmark.

Comparison of results on TimeLens-Bench and original benchmarks. In Tab. 7, Tab. 8, and Tab. 9, we compare the evaluation results of various models on TimeLens-Bench and the original benchmarks. On the original benchmarks, due to data quality issues, open-source models deceptively surpass state-of-the-art proprietary models like Gemini-2.5Pro. On our refined benchmarks, model capabilities are more reliably evaluated, with proprietary models maintaining a significant advantage over open-source models. Remarkably, our TimeLens model substantially narrows the performance gap between open-source and proprietary models.

Results on general video understanding. In Tab. 11, we evaluate TimeLens-7B’s general video understanding capabilities on VideoMME [10], the most comprehensive and widely-adopted video understanding benchmark. The results demonstrate that TimeLens-7B maintains the strong general video understanding capability of its base model. This validates that our proposed design principles can effectively enhance video temporal grounding capabilities without sacrificing general-purpose video understanding abilities.

Video-MME Model

Short Medium Long All

Qwen2.5-VL-7B [3] (Baseline) 64.3* 75.2* 55.1* 65.1† TimeLens-7B 66.4 76.7 54.1 65.7

Table 11. Results on the general video understanding benchmark Video-MME [10]. The results show that TimeLensQwen2.5-VL-7B maintains strong general video understanding capability while achieving substantial improvements in video temporal grounding. * Results reproduced by us. † Results reported in original paper.

###### E. Discussion on Thinking-free vs. Thinkingbased RLVR

We analyze the possible reasons why thinking-based RLVR underperforms thinking-free methods in our experiments, from both intuitive and empirical perspectives.

Intuitive Analysis.. When manually examining and refining existing grounding datasets, we observe that queries in the grounding task are relatively straightforward, primarily testing the model’s perception capability: whether the model can accurately localize the corresponding event in a long video containing massive information. From a human perspective, completing existing grounding tasks indeed relies mainly on intuition and instinct, rather than complex reasoning.

Empirical Observations.. In our experiments, when training with thinking-based RLVR, the model’s thinking length gradually decreases and converges to simple, non-reasoning thinking processes, as shown in Fig. 8. This suggests that the model learns to bypass explicit reasoning when it provides no benefit to the task.

Implications and Future Work.. We believe that most samples in existing video temporal grounding data do not require complex reasoning capabilities, but rather demand sufficiently robust long-video perception and localization abilities. Due to the high cost and corresponding low quality of existing data annotation, as well as limitations in current algorithmic designs, existing MLLMs cannot yet achieve this perfectly. Therefore, in this work, we focus on addressing these two fundamental issues. Meanwhile, we believe that certain grounding tasks do require reasoning capabilities, and we leave the exploration of reasoning-intensive VTG scenarios to future work.

###### F. Additional Discussions

Why do VideoChat-R1 and Time-R1 behave differently across benchmarks?. VideoChat-R1 [32] is trained on single-domain data with relatively limited quality control, and it does not incorporate the training recipes that we find most effective for VTG, such as difficulty-aware sampling and early stopping. As a result, it underperforms even the

Qwen2.5-VL baseline on our refined benchmarks. TimeR1 [54] by contrast, benefits from a stronger RL-oriented training recipe and therefore transfers better to ActivityNetTimeLens and QVHighlights-TimeLens, where long-video localization is a major challenge. However, its training data is still noisier than ours, which likely limits temporal precision and makes it less competitive on Charades-TimeLens, where videos are shorter but localization accuracy must be much higher.

Training setting for timestamp encoding.. All methods in Tab. 2 are compared after RLVR training rather than in a training-free setting. This controlled setup is important because the temporal encoding method interacts with the optimization paradigm: the final comparison reflects how well each representation supports post-training for VTG, instead of only measuring zero-shot prompting behavior.

Early stopping.. In our single-task VTG setting, reward plateauing is a reliable signal that continued RLVR is unlikely to improve grounding quality and may even degrade it, so early stopping saves computation while preventing over-training. We regard this as a practical recipe rather than a universal rule. In more general multi-task training, a single stopping criterion may be harder to define because different capabilities can peak at different stages, requiring more sophisticated recipes (e.g., remove training data for a single task when reward on this task peaks, while continuing training for other tasks).

###### G. Annotation Interface and Manual

We present our annotation interface in Fig. 14 and our annotation manual in Fig. 15.

###### H. Details of Curating TimeLens-100K

As described in Sec. H, we perform automated re-annotation on existing training datasets, resulting in TimeLens-100K, a large-scale, high-quality VTG training set comprising approximately 20K videos and 100K VTG annotations.

We begin by sampling videos from numerous existing VTG datasets, including CosMo-Cap [50], InternVidVTime [22], DiDeMo [1], QuerYD [41], HiREST [60], etc. These datasets already cover sufficiently diverse video domains. Additionally, we perform uniform sampling based on video duration to ensure sampled videos are approximately uniformly distributed within 0–240 seconds, with a small portion of longer videos included.

Given that most queries in the original annotations either lack clarity and specificity, or describe events that do not exist in the video, we directly use MLLMs for re-annotation. First, we prompt the MLLM to identify distinct events in the video and ensure these events are distributed across different time periods rather than being concentrated in a particular segment. Then, we have the model describe each event to

generate queries and output the corresponding timestamps. Finally, we prompt the model to verify the quality of the queries and annotations.

Specifically, we use Gemini-2.5-Pro [8], currently the best-performing VTG model, for re-annotation. The annotation prompt is provided in Fig. 10. Notably, although this prompt appears simple, it is the result of extensive prompt engineering and optimization. We find that a concise and intuitive prompt is sufficient, as the model possesses adequate common sense and reasoning capabilities to understand the task. Overly complex and detailed prompts are unnecessary and can actually degrade annotation quality. During its reasoning process, the model can automatically verify and ensure the uniqueness and uniform distribution of events throughout the video.

As shown in Tab. 5, our training data substantially improves model performance, validating the enhanced quality of our training set. Notably, the construction of our training data is independent of our manual benchmark refinement process, ensuring a fair comparison.

###### I. Implementation Details for Benchmarking Existing MLLMs

In this section, we present the implementation details for evaluating existing MLLMs on our TimeLens evaluation suite, yielding the results reported in Fig. 2a and Tab. 1.

TimeLens Models. The prompt for training and evaluating TimeLens models is illustrated in Fig. 9. Implementation details are provided in Sec. C.

GPT-5 [42] and GPT-4o [23]. Since GPT models only support multi-image sequences as input, we sample frames from videos at 1 FPS and prepend textual timestamps (i.e., “Frame at 2.5s:”) to each frame. As the Azure OpenAI API we use does not support more than 50 images for a single request, we adopt different strategies for videos longer than 50 seconds: for videos lasting 50-80 seconds, we uniformly sample 50 frames; for videos longer than 80 seconds, we sample at 1 FPS and arrange every 4 consecutive frames into a 2 × 2 grid within a single image, following previous works [24, 63, 64]. For GPT-5, which is a thinking model, we use the default value for the reasoning.effort parameter. The evaluation prompt is shown in Fig. 11.

Gemini models [8]. Although Gemini models support audio input, we remove the audio from videos to ensure fair comparison with other vision-only models and maintain consistency with our benchmarks, which features vision-only, audio-free annotations. Following the best practices outlined in the official Gemini API documentation, we prompt the models to output timestamps in “MM:SS” format. For other hyperparameters, we use the default settings: 1 FPS sampling and default mediaResolution, which tokenizes each frame into 258 tokens. For thinking models, we do

### Prompt for TimeLens Models

You are given a video with multiple frames. The numbers before each video frame indicate its sampling timestamp (in seconds). Please find the visual event described by the sentence ‘{query}’, determining its starting and ending times. The format should be: ‘The event happens in <start time> - <end time> seconds’.

Figure 9. Prompt for training and evaluating TimeLens.

## Prompt for Annotating TimeLens-100K

First, design five queries related to video grounding, for example: “When can we observe ‘a woman cooking in the kitchen’? Please specify the time range.” The queries can be flexible and diverse in form, but the answer to each query must correspond to exactly one time range. Then provide the answers. The answers should be time ranges (timestamps). Please provide the queries and answers in JSON format, with 'query' for the question and 'timestamps' for the answer, in the format ["MM:SS", "MM:SS"], for example ["00:59", "01:02"].

Figure 10. Prompt for annotating TimeLens-100K.

not impose any limit on the thinkingBudget parameter. The evaluation prompt is shown in Fig. 12.

Qwen3-VL [2], Qwen2.5-VL [3] and MiMo-VL [9]. TimeLens models, Qwen2.5-VL-7B, and MiMo-VL share approximately the same model architecture and hyperparameter configurations. Therefore, when evaluating these models, we adopt the same settings to ensure fair comparisons in Tab. 1. Specifically, consistent with Sec. C.2, we sample video frames at 2 FPS and set the resolution budget to min tokens = 64 and total tokens = 14,336. For MiMo-VL, we evaluate their best-performing model, MiMoVL-7B-RL. The evaluation prompt is shown in Fig. 13.

Other Open-source Models. When evaluating TimeR1 [54], VideoChat-Flash [31] and VideoChat-R1 [32], we directly use their original codebases. Please refer to their papers and code repositories for details.

##### Prompt for Evaluating GPT-5/GPT-4o Models

Prompt for videos shorter than 80 seconds: You are given multiple frames from a video. Each frame is preceded by its timestamp (e.g., ‘Frame at 2.5s:’). Please find the visual event described by the sentence ‘{query}’, determining its starting and ending times. Answer in the format ‘The event happens in x - y seconds’, where x is the start time and y is the end time (x < y).

|Prompt for videos longer than 80 seconds: You are given multiple frames from a video. Every four adjacent frames are stacked into a 2x2 grid (topleft, top-right, bottom-left, bottom-right in timestamp order). Each stacked frame is preceded by its timestamps (e.g., ‘Stacked frames at 1.0s, 2.0s, 3.0s, 4.0s:’). Please find the visual event described by the sentence ‘{query}’, determining its starting and ending times. Answer in the format 'The event happens in x - y seconds', where x is the start time and y is the end time (x < y).|
|---|

Figure 11. Prompts for evaluating GPT-5 and GPT-4o.

###### Prompt for Evaluating Gemini Models

Please find the visual event described by the sentence ‘{query}’, determining its starting and ending times. Answer in the format ‘The event happens in MM:SS - MM:SS’.

Figure 12. Prompt for evaluating Gemini models.

###### Prompt for Evaluating Qwen/MiMo Models

Please find the visual event described by the sentence ‘{query}’, determining its starting and ending times. The format should be: ‘The event happens in <start time> - <end time> seconds’.

Figure 13. Prompt for evaluating Qwen3-VL, Qwen2.5-VL and MiMo-VL models.

[Figure 54]

Video Temporal Localization Submit Result Event Description: A boy is talking to a camera.

Labels

###### Figure 14. Illustration of our annotation interface.

###### Video Temporal Localization Annotation Guidelines

###### Task Overview

The annotation page contains a video and a description of a specific event. You need to watch the video to confirm whether the event occurred, how many times it occurred, select the corresponding option, and fill in the subsequent information (rewriting the description, filling in the start and end times of the event, etc.).

###### Annotation Page Description

The annotation page is shown below.

Left side: The video page where you can play the video.

Right side (Red Box): The "Event Description" area. This provides a sentence describing the event that needs to be annotated.

Right side (Green Box): The options that need to be selected.

Image Here Instructions: Watch the video, select the corresponding option, and the information required to be filled in will appear below. Fields marked with a red asterisk (*) are mandatory. Image Here

###### Important Note on Time Entry

When filling in the "Event Start Time" and "Event End Time" in the image above, you need to move your mouse over the video timeline. You will see the specific time of the current moment in the white text box (indicated by the Red Box in the diagram below; e.g., 0 min 29 sec).

Please Note: The time displayed on the right side of the video timeline (indicated by the Green Box area below) is NOT the time of the current moment! Do not enter this time!

(a)

Image Here

###### Detailed Step-by-Step Instructions

First, you need to watch the entire video in mute mode. Pay attention to observe in which time segments the "Event Description" occurred, and complete the first multiplechoice question on the page:

Image Here

- 1. If the event occurs multiple times in the video (i.e., occurs in several noncontinuous time segments):
- 2. If the event described by the sentence occurs in only one continuous time segment, and you can clearly determine the start and end times of the event:
- 3. If the event described by the sentence does not occur at all in the entire video:
- 4. If for various reasons (description is unclear, video is too blurry, etc.) you cannot judge whether the event occurred, or cannot accurately determine the start and end times:

a) You need to select the "Event occurs multiple times" option.

a) You need to select the "Event occurs only once, and start and end times can be accurately determined" option.

a) You need to select the "Event did not occur" option.

- a) You need to select the "Unable to determine if event occurred, or unable to accurately determine start and end times" option.

- b) Then, you need to fill in the specific reason in the "Why unable to determine" field.

Image Here

Next, complete the second multiple-choice question: If you have already annotated the same event for the same video, select "Yes"; otherwise, select "No".

(b)

Image Here

###### Scenario A: You selected "Event occurs only once..."

If you selected the second option in the first question ("Event occurs only once..."), and selected "No" in the second question:

- 1. Please perform appropriate Polishing on the event description. Make the description clearer and smoother without changing the semantic meaning of the sentence. You must ensure that after polishing, the start and end times of the event remain unchanged, and the event still occurs only once in the video. (If the original description already meets the requirements, you do not need to fill this in).
- 2. Then, near the start and end times of the event, pause the video and drag the progress bar to determine the start and end times as accurately as possible. Fill in the minutes and seconds for the start and end.

Note: For detailed instructions on "Polishing," please refer to the "Precautions" section of this document.

###### Scenario B: You selected other options If you selected any other option in the first question (Multiple times, Did not occur, etc.):

- 1. You need to Rewrite the description of the event (if it is truly impossible to rewrite, choose a new event to describe). Ensure the modified event occurs only once in the video and that you can accurately determine its start and end times.

- Example 1: "A person walking" occurs multiple times. It can be rewritten as "A man wearing blue clothes is walking," "A person walking on the crosswalk," "A person walking while holding an ice cream," or "A man walking with two women."

- Example 2: "A man walking" did not occur, but there is a woman walking. It can be rewritten as "A woman is walking."

- 2. Then, near the start and end times of the rewritten event, pause the video and drag the progress bar to determine the start and end times as accurately as possible. Fill in the minutes and seconds for the start and end.

(c)

Note: For detailed instructions on "Rewriting," please refer to the "Precautions" section of this document.

###### Final Check

Afterwards, please watch the entire video again to check the filled content:

- 1. Ensure the event described by the modified sentence occurs only once.
- 2. Ensure the start and end times of the event are correct and accurate.

After checking, you can proceed to the next annotation.

###### Annotation Examples

......

###### Precautions / Important Notes

###### I. Instructions regarding "Rewriting"

When rewriting the original sentence, please follow these steps:

......

###### II. Instructions regarding "Polishing"

......

###### FAQ (Frequently Asked Questions)

###### Question 1:

......

###### Question 2:

.......

(d)

Figure 15. Illustration of our annotation manual. Some figures and details are removed for confidentiality and safety reasons.

