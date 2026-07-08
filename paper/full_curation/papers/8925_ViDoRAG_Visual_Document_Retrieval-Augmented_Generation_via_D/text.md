## ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents

### Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, Feng Zhao*

[Figure 1]

Tongyi Lab, Alibaba Group Dataset & Code: https://github.com/Alibaba-NLP/ViDoRAG

### Abstract

- (a) Comparison between our ViDoSeek and the traditional VQA datasets.
- (b) Comparison between our ViDoRAG and the traditional RAG approach.

Traditional VQA Dataset ViDoSeek(Ours): Q&A with the Large Corpus

Understanding information from visually rich documents remains a significant challenge for traditional Retrieval-Augmented Generation (RAG) methods. Existing benchmarks predominantly focus on image-based question answering (QA), overlooking the fundamental challenges of efficient retrieval, comprehension, and reasoning within dense visual documents. To bridge this gap, we introduce ViDoSeek, a novel dataset designed to evaluate RAG performance on visually rich documents requiring complex reasoning. Based on it, we identify key limitations in current RAG approaches: (i) purely visual retrieval methods struggle to effectively integrate both textual and visual features, and (ii) previous approaches often allocate insufficient reasoning tokens, limiting their effectiveness. To address these challenges, we propose ViDoRAG, a novel multi-agent RAG framework tailored for complex reasoning across visual documents. ViDoRAG employs a Gaussian Mixture Model (GMM)-based hybrid strategy to effectively handle multi-modal retrieval. To further elicit the model’s reasoning capabilities, we introduce an iterative agent workflow incorporating exploration, summarization, and reflection, providing a framework for investigating test-time scaling in RAG domains. Extensive experiments on ViDoSeek validate the effectiveness and generalization of our approach. Notably, ViDoRAG outperforms existing methods by over 10% on the competitive ViDoSeek benchmark.

What is the peak value of the data?

What is the profit difference between the highest and lowest selling products in Apple's 2024 quarterly reports?

arXiv:2502.18017v2[cs.CV]3Jun2025

Answer: 529

[Figure 2]

[Figure 3]

Answer: 32°C

Unique Answer: $24.7Billion

[Figure 4]

[Figure 5]

[Figure 6]

###### . . .

[Figure 7]

…

[Figure 8]

Retrieval-Augmented Generation System…

VLMs, LLMs,DocLMs …

###### Traditional RAG

###### ViDoRAG(Ours)

[Figure 9]

[Figure 10]

Limited by retrieval precision, reasoning capability, etc.

A novel multi-agent, coarse-to-fine RAG framework with dynamic retrieval and task-specific agents.

[Figure 11]

Inspector Agent

[Figure 12]

[Figure 13]

[Figure 14]

###### Top-K

Seeker Agent

[Figure 15]

Multi-modal Retrieval

Answer Agent

Vector Retrieval

Generation

Figure 1: Comparison of our work with the existing datasets and methods. (a) In traditional datasets, each query must be paired with specific images or documents. In our ViDoSeek, each query can obtain a unique answer within the large corpus. (b) Our ViDoRAG is a multiagent, coarse-to-fine framework specifically optimized for visually rich documents.

elements make information easier to understand and are widely used in education, finance, law, and other fields. Therefore, researching RAG within visually rich documents is highly valuable.

In practical applications, RAG systems often need to retrieve information from a large collection consisting of hundreds of documents, amounting to thousands of pages. As shown in Fig. 1, existing Visual Question Answering (VQA) benchmarks aren’t designed for such large corpus. The queries in these benchmarks are typically paired with one single image(Methani et al., 2020; Masry et al., 2022; Li et al., 2024; Mathew et al., 2022) or document(Ma et al., 2024), which is used for evaluating Q&A tasks but not suitable for evaluating RAG systems. The answers to queries in these datasets may not be unique within the whole corpus.

### 1 Introduction

Retrieval-Augmented Generation (RAG) enhances Large Models (LMs) by enabling them to use external knowledge to solve problems. As the expression of information becomes increasingly diverse, we often work with visually rich documents that contain diagrams, charts, tables, etc. These visual

To address this gap, we introduce ViDoSeek, a novel dataset designed for visually rich document retrieval-reason-answer. In ViDoSeek, each query

*Corresponding author

has a unique answer and specific reference pages. It covers the diverse content types and multi-hop reasoning that most VQA datasets include. This specificity allows us to better evaluate retrieval and generation performance separately.

Moreover, to enable models to effectively reason over a large corpus, we propose ViDoRAG, a multi-agent, coarse-to-fine retrieval-augmented generation framework tailored for visually rich documents. Our approach is based on two critical observations: (i) Inefficient and Variable Retrieval Performance. Traditional OCR-based retrieval struggles to capture visual information. With the development of vision-based retrieval, it is easy to capture visual information(Faysse et al., 2024; Yu et al., 2024a; Zhai et al., 2023). However, there lack of an effective method to integrate visual and textual features, resulting in poor retrieval of relevant content. (ii) Insufficient Activation of Reasoning Capabilities during Generation. Previous studies on inference scaling for RAG focus on expanding the length of retrieved documents(Jiang et al., 2024; Shao et al., 2025; Xu et al., 2023). However, due to the characteristics of VLMs, only emphasizing on the quantity of knowledge without providing further reasoning guidance presents certain limitations. There is a need for an effective inference scale-up method to efficiently utilize specific action spaces, such as resizing and filtering, to fully activate reasoning capabilities.

Building upon these insights, ViDoRAG introduces improvements in both retrieval and generation. We propose Multi-Modal Hybrid Retrieval, which combines both visual and textual features and dynamically adjusts results distribution based on Gaussian Mixture Models (GMM) prior. This approach achieves the optimal retrieval distribution for each query, enhancing generation efficiency by reducing unnecessary computations. During generation, our framework comprises three agents: the seeker, inspector, and answer agents. The seeker rapidly scans thumbnails and selects relevant images with feedback from the inspector. The inspector reviews, then provides reflection and offers preliminary answers. The answer agent ensures consistency and gives the final answer. This framework reduces exposure to irrelevant information and ensures consistent answers across multiple scales.

Our major contributions are as follows:

• We introduce ViDoSeek, a benchmark specifically designed for visually rich document

retrieval-reason-answer, fully suited for evaluation of RAG within large document corpus.

- • We propose ViDoRAG, a novel RAG framework that utilizes a multi-agent, actor-critic paradigm for iterative reasoning, enhancing the noise robustness of generation models.
- • We introduce a GMM-based multi-modal hybrid retrieval strategy to effectively integrate visual and textual pipelines.
- • Extensive experiments demonstrate the effectiveness of our method. ViDoRAG significantly outperforms strong baselines, achieving over 10% improvement, thus establishing a new state-of-the-art on ViDoSeek.

### 2 Related Work

Visual Document Q&A Benchmarks. Visual Document Question Answering is focused on answering questions based on the visual content of documents(Antol et al., 2015; Ye et al., 2024; Wang et al., 2024). While most existing research (Methani et al., 2020; Masry et al., 2022; Li et al., 2024; Mathew et al., 2022) has primarily concentrated on question answering from single images, recent advancements have begun to explore multi-page document question answering, driven by the increasing context length of modern models (Mathew et al., 2021; Ma et al., 2024; Tanaka et al., 2023). However, prior datasets were not wellsuited for RAG tasks involving large collections of documents. To fill this gap, we introduce ViDoSeek, the first large-scale document collection QA dataset, where each query corresponds to a unique answer across a collection of ∼ 6k images.

Retrieval-augmented Generation. With the advancement of large models, RAG has enhanced the ability of models to incorporate external knowledge (Lewis et al., 2020; Chen et al., 2024b; Wu et al., 2025). In prior research, retrieval often followed the process of extracting text via OCR technology (Chen et al., 2024a; Lee et al., 2024; Robertson et al., 2009). Recently, the growing interest in multimodal embeddings has greatly improved image retrieval tasks (Faysse et al., 2024; Yu et al., 2024a). Additionally, there are works that focus on In-Context Learning in RAG(Agarwal et al., 2025; Yue et al., 2024; Team et al., 2024; Weijia et al., 2023). Our work builds upon these developments by combining multi-modal hybrid retrieval with a

Fast but rough

Document Database

###### Semantic Filter By LLM

[Figure 16]

Initial Query

Refined Query

Final Output

[Figure 17]

Human Experts

Refine with Golden Image

[Figure 18]

Sample

Slow but careful

[Figure 19]

[Figure 20]

[Figure 21]

Qualified Query

Unqualified Query

Document Candidate

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Retrieve within Document Collection

Visual Filter By VLM

(a) Document Collecting

(b) Query Creation (c) Quality Review (d) Multimodal Refine

Figure 2: Data Construction pipeline. (a) We sample and filter documents according to the requirements to obtain candidates. (b) Then experts construct the initial query from different contents. (c) After that, we prompt GPT-4 to directly determine whether the query is a general query. The remaining queries are carefully reviewed with top-K recall images. (d) Finally, unqualified queries are refined paired with golden image by GPT-4o.

coarse-to-fine multi-agent generation framework, seamlessly integrating various embedding and generation models into a scalable framework.

the open-source dataset, we initiate the query refinement starting from the third step of our pipeline. For the dataset we build from scratch, we follow the entire pipeline beginning with document collection. The following outlines our four-step pipeline:

### 3 Problem Formulation

Given a query as q, and we have a collection of documents C = {D1,D2,...,DM} which contains

- Step 1. Document Collecting. As slides are a widely used medium for information transmission today, we selected them as our document source. We began by collecting English-language slides containing 25 to 50 pages, covering 12 domains such as economics, technology, literature, and geography. And we filtered out 300 slides that simultaneously include text, charts, tables, and twodimensional layouts which refer to flowcharts, diagrams, or any visual elements composed of various components and are a distinctive feature of slides.
- Step 2. Query Creation. To make the queries more suitable for RAG over a large-scale collection, our experts were instructed to construct queries that are specific to the document. Additionally, we encouraged constructing queries in various forms and with different sources and reasoning types to better reflect real-world scenarios.
- Step 3. Quality Review. In large-scale retrieval and generation tasks, relying solely on manual annotation is challenging due to human brain limitations. To address this, we propose a review module that automatically identifies problematic queries.
- Step 4. Multimodal Refine. In this final step, we refine the queries that did not meet our standards during the quality review. We use carefully designed VLM-based agents to assist us throughout the entire dataset construction pipeline.

- M documents. Each document Dm consists of
- N pages, each image representing an individual

page, defined as Dm = {I1,I2,...,IN}. The total number of images included in the collection is

M m=1 |Dm|. We aim to retrieve the most relevant

information efficiently and accurately and generate the final answer a to the query q.

### 4 ViDoSeek Dataset

Existing VQA datasets typically consist of queries paired with a single image or a few images. However, in practical application scenarios, users often pose questions based on a large-scale corpus rather than targeting an individual document or image. To better evaluate RAG systems, we prefer questions that have unique answers when retrieving from a large corpus. To address this need, we introduce a novel Visually rich Document dataset specifically designed for RAG systems, called ViDoSeek. Below we provide the pipeline for constructing the dataset(§4.1) and a detailed analysis of the dataset(§4.2).

#### 4.1 Dataset Construction.

To construct the ViDoSeek dataset, we developed a four-step pipeline to ensure that the queries meet our stringent requirements. As illustrated in Figure

- 2, our dataset comprises two parts: one annotated from scratch by our AI researchers, and the other derived from refining queries in the existing opensource dataset SlideVQA (Tanaka et al., 2023). For

#### 4.2 Dataset Analysis

Dataset Statistics. ViDoSeek is the first dataset specifically designed for question-answering over

##### Table 1: Comparison of existing dataset with ViDoSeek.

###### DATASET DOMAIN CONTENT TYPE REFERENCE TYPE LARGE DOCUMENT COLLECTION

PlotQA(Methani et al., 2020) Academic Chart Single-Image ✗ ChartQA(Masry et al., 2022) Academic Chart Single-Image ✗ ArxivQA(Li et al., 2024) Academic Chart Single-Image ✗ InfoVQA(Mathew et al., 2022) Open-Domain Text, Chart, Layout Single-Image ✗ DocVQA(Mathew et al., 2021) Open-Domain Text, Chart, Table Single-Document ✗ MMLongDoc(Ma et al., 2024) Open-Domain Text, Chart, Table, Layout Single-Document ✗ SlideVQA(Tanaka et al., 2023) Open-Domain Text, Chart, Table, Layout Single-Document ✗

ViDoSeek(Ours) Open-Domain Text, Chart, Table, Layout Multi-Documents ✓

large-scale document collections. It comprises approximately ∼ 1.2k questions across a wide array of domains, addressing four key content types: Text, Chart, Table, and Layout. Among these, the Layout type poses the greatest challenge and represents the largest portion of the dataset. Additionally, the queries are categorized into two reasoning types: single-hop and multi-hop. Further details of the dataset can be found in the Appendix B and C.

Comparative Analysis. Table 1 highlights the limitations of existing datasets, which are predominantly tailored for scenarios involving single images or documents, lacking the capacity to handle the intricacies of retrieving relevant information from large collections. ViDoSeek bridges this gap by offering a dataset that more accurately mirrors real-world scenarios. This facilitates a more robust and scalable evaluation of RAG systems.

### 5 Method

In this section, drawing from insights and foundational ideas, we present a comprehensive description of our ViDoRAG framework, which integrates two modules: Multi-Modal Hybrid Retrieval (§5.1) and Multi-Scale View Generation (§5.2).

#### 5.1 Multi-Modal Hybrid Retrieval

For each query, our approach involves retrieving information through both textual and visual pipelines, dynamically determining the optimal value of topK using a Gaussian Mixture Model (GMM), and merging the retrieval results from both pipelines.

Adaptive Recall with Gaussian Mixture Model. Traditional methods rely on a static hyperparameter, K, to retrieve the top-K images or text chunks from a corpus. A smaller K might fail to capture sufficient references needed for accurate responses,

- as the most relevant nodes are not always ranked
- at the top. Conversely, a larger K can slow down inference and introduce inaccuracies due to noise.

Additionally, manually tuning K for different scenarios is troublesome.

Our objective is to develop a straightforward yet effective method to automatically determine K for each modality, without the dependency on a fixed value. We utilize the similarity S of the embedding E to quantify the relevance between the query and the document collection C:

S(q,C) = {si|cos(Eq,Epi),pi ∈ C} (1)

where si represents the cosine similarity between the query Q and page pi. In the visual pipeline, a page corresponds to an image, whereas in the textual pipeline, it corresponds to chunks of OCR text. We propose that the distribution of S follows a GMM and we consider they are sampled from a bimodal distribution P(s) shown in Fig.3:

P(s) = wF · N(s | µF, σF2 ) + wT · N(s | µT, σT2 ) (2)

where N represents a Gaussian distribution, with w,µ,σ2 indicating the weight, mean, and variance, respectively. The subscripts T and F refer to the distributions of pages with high and low similarity. The distribution with higher similarity is deemed valuable for generation. The ExpectationMaximization (EM) algorithm is utilized to estimate the prior probability P(T|s,µT,σT2 ) for each modality. The dynamic value of K is defined as:

K = |{pi ∈ C | pi ∼ N(µT,σT2 )}| (3)

Considering that the similarity score distribution for different queries within a document collection may not strictly follow a standard distribution, we establish upper and lower bounds to manage outliers. The EM algorithm is employed sparingly, less than ∼ 1% of the time. Dynamically adjusting K enhances generation efficiency compared to a static setting. Detailed analysis is available in §7.2.

Textual and Visual Hybrid Retrieval. In the previous step, nodes were retrieved from both

###### Multi-Modal Hybrid Retrieval

DynamicLength

Multi-modal Embedding

Parser Similarity

Modality-Wise GMM

[Figure 26]

Measure

[Figure 27]

…

[Figure 28]

N

[Figure 29]

[Figure 30]

[Figure 31]

Nodes Fusion

[Figure 32]

Multi-modal Retrieval

# … !!

Similarity

[Figure 33]

[Figure 34]

1 2

###### Inspector Agent: Detailed review and reflect

[Figure 35]

Seeker Agent: Hunting for relevant images

###### When information is insufficient to answer:

Initial Step:

###### ℱ!

[Figure 36]

Each Round

[Figure 37]

Human Query Retrieval Results

###### Reason about what to do next.

Reflection on the images selected by the Seeker

When there is reflection from inspector：

Each Round

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

“I have Known {Knowledge from Image}, My next step is to {Future Plan}.”

[Figure 42]

!!"

[Figure 43]

[Figure 44]

Reason with reflection of inspector.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Feedback for additional information.

!!%

Reflection

“After reflecting on {Reflection}, My current thought is {Current Thought}.”

Reflection: “I need more information about…”

Extract the useful global information, Retrieve the relevant Image.

Retain relevant images after detailed review

Select the useful images to retain.

[Figure 51]

[Figure 52]

[Figure 53]

!!#$"

[Figure 54]

[Figure 55]

“ Next, information about {Image description} is needed, Useful information might be found in {Selected Image}.”

“I choose {Retained Image} as reference.”

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

When the information is sufficient to answer:

3

Answer Agent: Synthesize the final answer

$#

Summarize and output draft answer.

Consistency check and return the final answer.

!&'(

[Figure 60]

“The final answer is {Draft Answer}, and the reference is {Reference Image}.”

[Figure 61]

“Based on the {Draft Answer} and {Reference Image}, my final answer is {Final Answer}.”

Figure 3: ViDoRAG Framework.

extracts clues, reflects, and answers in a coarse-tofine manner from a multi-scale perspective. More details are provided in Appendix D.

pipelines. In this phase, we integrate them:

Rhybrid = Sort[F(RText,RV isual)] (4)

where RText and RV isual denote the retrieval results from the textual and visual pipelines, respectively. The function F(·) signifies a union operation, and Sort(·) arranges the nodes in their original sequence, as continuous pages often exhibit correlation (Yu et al., 2024b).

Seeker Agent: Hunting for relevant images. The Seeker Agent is responsible for selecting from a coarse view and extracting global cues based on the query and reflection from the Inspector Agent. We have made some improvements to ReAct(Yao et al., 2022) to facilitate better memory management. The action space is defined as the selection of the images. Initially, the agent will reason only based on the query Q and select the most relevant images Is0 from the candidate images Ic0, while the initial memory M0 is empty. In step t, the candidate images Ict+1 are the complement of previously selected images Ist, defined as Ict+1 = Ict \ Ist. The seeker has received the reflection Ft−1 from the inspector, which includes an evaluation of the selected images and a more detailed description of the requirements for the images. The Seeker integrates feedback Ft−1 from the Inspector, which includes an evaluation of the selected images and a description of image requirements, to further refine

The textual and visual retrieval pipelines demonstrate varying levels of performance for different features. Without adaptive recall, the combined retrieval Rhybrid can become excessive. Adaptive recall ensures that effective retrievals are concise, while traditional pipelines yield longer recall results. This strategy optimizes performance relative to context length, underscoring the value of adaptive recall in hybrid retrieval.

- 5.2 Multi-Agent Generation with Iterative Reasoning

During the generation, we introduce a multi-agent framework which consists of three types of agents: the Seeker Agent, the Inspector Agent, and the Answer Agent. As illustrated in Fig. 3, this framework

the selection Ist and update the memory Mt+1:

Ict+1, Mt+1 = Θ(Ict,Q,Mt,Ft−1) (5)

where Mt+1 represents the model’s thought content in step t under the ReAct paradigm, maintaining a constant context length. The process continues until the Inspector determines that sufficient information is available to answer the query, or the Seeker concludes that no further relevant images exist among the candidates.

Inspector Agent: Review in detail and Reflect. In baseline scenarios, increasing the top-K value improves recall@K, but accuracy initially rises and then falls. This is attributed to interference from irrelevant images, referred to as noise, affecting model generation. To address this, we use Inspector to perform a more fine-grained inspection of the images. In each interaction with the Seeker, the Inspector’s action space includes providing feedback or drafting a preliminary answer. At step t, the inspector reviews images at high resolution, denoted as Θ(Ict ∪ Irt−1,Q) where Irt−1 are images retained from the previous step and Ict are from the Seeker. If the current information is sufficient to answer the query, a draft answer Aˆ is provided, alongside a reference to the relevant image:

Aˆ, Iref = Θ(Ict ∪ Irt−1,Q) (6)

Conversely, if more information is needed, the Inspector offers feedback Ft to guide the Seeker in better image selection and identifies images Irt to retain for further review in the next step t + 1:

Ft, Irt = Θ(Ict ∪ Irt−1,Q) (7)

The number of images the Inspector reviews is typically fewer than the Seeker’s, ensuring robustness in reasoning, particularly for Visual Language Models with moderate reasoning abilities.

Answer Agent: Synthesize the final answer. In our framework, the Seeker and Inspector engage in a continuous interaction, and the answer agent provides the answer in the final step. To balance accuracy and efficiency, the Answer Agent verifies the consistency of the Inspector’s draft answer Aˆ. If the reference image matches the Inspector’s input, the draft answer is accepted as the final answer A = Aˆ. If the reference image is a subset of the input image, the answer agent should check for consistency between the draft answer Aˆ and the

reference image, then give the final answer A: If the reference image is a subset of Inspector’s the input, the Answer Agent ensures consistency between the draft answer Aˆ and the reference image before finalizing the answer A:

A = Θ(Iref,Q,Aˆ) (8)

The Answer Agent utilizes the draft answer as prior knowledge to refine the response from coarse to fine. The consistency check between the Answer Agent and Inspector Agent enhances the depth and comprehensiveness of the final answer.

### 6 Experiments

#### 6.1 Experimental Settings

Evaluation Metric For our end-to-end evaluation, we employed a model-based assessment using GPT-4o, which involved assigning scores from 1 to 5 by comparing the reference answer with the final answer. Answers receiving scores of 4 or above were considered correct, and we subsequently calculate accuracy as the evaluation metric. For retrieval evaluation, we use recall as the metric.

Baselines and Oracle. We selecte Nv-embedV2(Lee et al., 2024) and ColQwen2(Faysse et al., 2024) as the retrievers for the TextRAG and VisualRAG baselines, respectively. Based on their original settings, we choose the top-5 recall results as the generation input, which equals the average length of dynamic recall results. This ensures a fair comparison and highlights the advantages of our method. The Oracle serves as the upper bound performance, where the model responds based on the golden page without retrieval or other operations.

#### 6.2 Main Results

As shown in Table. 2, we conducted experiments on both closed-source and open-source models: GPT-4o, Qwen2.5-7B-Instruct, Qwen2.5-VL7B(Yang et al., 2024)-Instruct, Llama3.2-Vision90B-Instruct. Closed-source models generally outperform open-source models performance. It is worth mentioning that the qwen2.5-VL-7B has shown excellent instruction-following and reasoning capabilities within our framework. In contrast, we found that the llama3.2-VL requires 90B parameters to accomplish the same instructions, which may be related to the model’s pre-training domain. The results suggest that while API-based models offer strong baseline performance, our method is also

##### Table 2: Overall Generation performance.

METHOD Single-hopREASONINGMulti-hopTYPE Text TableANSWERChartTYPE Layout OVERALL Llama3.2-Vision-90B-Instruct

Upper Bound 83.1 78.7 88.7 73.1 68.1 85.1 81.1 TextRAG 42.6 45.7 67.6 41.8 25.4 45.9 43.9 VisualRAG 61.8 60.5 82.5 48.5 52.2 63.9 61.2 ViDoRAG (Ours) 73.3 68.5 85.1 65.6 56.1 74.7 71.2

Qwen2.5-VL-7B-Instruct

Upper Bound 77.5 78.2 88.4 77.1 69.4 78.8 77.9 TextRAG 59.6 55.7 78.7 53.8 40.7 60.5 57.6 VisualRAG 66.8 64.3 84.9 61.1 52.8 67.5 65.7 ViDoRAG (Ours) 70.4 67.3 81.9 65.2 57.7 71.3 69.1

GPT-4o (Closed-Sourced Models)

Upper Bound 88.8 86.3 97.5 85.7 77.1 89.4 87.7 TextRAG 64.3 62.6 78.7 61.0 48.4 66.1 63.5 VisualRAG 75.7 66.1 90.1 62.4 58.5 75.4 72.1 ViDoRAG (Ours) 83.5 74.1 88.5 73.6 76.4 80.4 79.4

Table 3: Retrieval Performance on ViDoSeek.

retrieval across queries, we use the average length of results for analysis. Our goal is to incorporate more relevant information within a shorter context while minimizing the impact of noise and reducing computational cost without losing valuable information. Dynamic retrieval can achieve better recall performance with a smaller context length, while hybrid retrieval combines the results of two pipelines achieving state-of-the-art performance.

Retriever Recall@1 Recall@3 Recall@5 MRR@5

BM25 55.2 77.4 84.5 66.5 BGE-M3(Chen et al., 2024a) 60.2 79.3 87.6 70.5 NV-Embed-V2(Lee et al., 2024) 64.1 83.5 90.3 74.7

VisRAG-Ret(Yu et al., 2024a) 64.4 84.1 91.2 75.2 ColPali(Faysse et al., 2024) 70.6 87.9 92.8 79.6 ColQwen2(Faysse et al., 2024) 75.4 89.7 95.1 83.3

effective in enhancing the performance of opensource models, offering promising potential for future applications. To further demonstrate the robustness of the framework, we constructed a pipeline using data to rewrite queries from SlideVQA(Tanaka et al., 2023), making the queries suitable for scenarios involving large corpora. The experimental results are presented the analysis.

### 7 Analysis

#### 7.1 Ablations

Table 4 presents the impact of different retrievers and generation methods on performance. We have decomposed the dynamic retrieval into two components, Dynamic and Hybrid. Naive refers to the method of direct input, which is most commonly used as baselines. Dynamic indicates using GMM to fit the optimal recall distribution based solely on the visual pipeline. Hybrid refers to merging the visual and the textual retrieval results directly, which leads to suboptimal results due to long contexts. Experiments demonstrate that the effectiveness and scalability of our improvements on retrieval and generation modules, as well as their combination, can comprehensively enhance end-to-end performance from various perspectives.

Recall

1.00

0.95

Hybrid Retrieval : Avg. K = 8.1 ColQwen2 /w GMM: Avg. K = 6.9 NV-Embed-V2 w/GMM: Avg. K = 5.7

0.90

0.85

0.80

0.75

ColQwen2

NV-Embed-V2 ColQwen2 w/ GMM NV-Embed-V2 w/ GMM Hybrid Retrieval (Ours)

0.70

0.65

1 3 5 7 9 Top-K

Figure 4: Retrieval performance across different retrievers and hybrid retrieval, along with ablations on GMM.

#### 7.2 Time Efficiency

#### 6.3 Retrieval Evaluation

How does dynamic retrieval balance latency and accuracy? In traditional RAG systems, using a small top-K value may result in missing critical

In Table 3, we report the detailed performance for various retrievers, including OCR-based and visual-based. Due to the uncertainty of dynamical

- Table 4: Ablation study on ViDoSeek benchmark.

RETRIEVAL GENERATION Accuracy Naive Dynamic Hybrid Naive Multi-Agent

✓ ✓ 72.1 ✓ ✓ 72.8 ✓ ✓ 74.1 ✓ ✓ ✓ 74.3 ✓ ✓ 77.3 ✓ ✓ ✓ 79.4

- Table 5: Evaluation of Dynamic Retrieval Methods.

speaking, due to models’ inherent characteristics, the reasoning ability of LLMs is stronger than that of VLMs. However, the lack of visual information makes it difficult for models to identify the intrinsic connections between pieces of information. This also poses a challenge for the generation of content based on visually rich documents. While obtaining visual information, VidoRAG further enhances the reasoning capabilities of VLMs, striking a balance between accuracy and computational load.

Method Accuracy ↑ Avg. Pages ↓

w/o GMM 72.1 10 w/ GMM 72.8 6.76

Text

MultiHop

MultiHop

NonSpan

SingleHop

SingleHop

information, whereas employing a larger value can introduce noise and increase computational overhead. ViDoRAG dynamically determines the number of documents to retrieve based on the similarity distribution between the query and the corpus. This approach ensures that only the most relevant documents are retrieved, thereby reducing unnecessary computations from overly long contexts and accelerating the generation process. As shown in Table 5, we compare retrieval with and without GMM based on the Naive method. The experiments indicate that GMM may reduce recall due to distribution bias. However, because it significantly shortens the generation context, it effectively improves performance in end-to-end evaluations.

Table

SingleSpan

TextRAG VisualRAG ViDoRAG(Ours)

MultiSpan

Chart Layout

(a) Performance on ViDoSeek (b) Performance on SlideVQA-Refined

###### Figure 6: Performance across different types of queries on our ViDoSeek and the refined SlideVQA datasets.

Accuracy

80

GPT-4o

Qwen2.5-VL-72B

70

Qwen2.5-VL-7B

Llama3.2-Vision-90B

Latency Analysis of the Multi-Agent Generation. There is an increase in delay due to the iterative nature of the multi-agent system, as shown in Fig. 5. Each agent performs specific tasks in a sequential manner, which adds a small overhead compared to traditional straightforward RAG. However, despite the increase in latency, the overall performance improves due to the higher quality of generated answers, making the trade-off between latency and accuracy highly beneficial for complex RAG tasks.

60

Llama3.2-Vision-11B

2 3 Avg. Reasoning Iterations

Figure 7: Scaling behavior with ViDoRAG.

#### 7.4 Performance with Test-time Scaling

Fig. 7 illustrates the number of interaction rounds between the seeker and inspector within ViDoRAG based on different models. Due to the limited instruction capabilities of some models, we sampled 200 queries for the experiment. Models with stronger performance require fewer reasoning iterations, while weaker models often need additional time to process and reach a conclusion. Conditioning the model on a few demonstrations of the task at inference time has been proven to be a computationally efficient approach to enhance model performance(Brown et al., 2020; Min et al., 2021). The results indicate that predefining tasks and breaking down complex tasks into simpler ones is an effective method for scaling inference.

Baseline

Seeker

Traditional RAG

Inspector

Answer

ViDoRAG

Avg. Query Latency

Figure 5: Latency Analysis on Generation.

#### 7.3 Modalities and Strategies of Generation

As shown in Fig. 6, the vision-based pipeline outperforms the text-based pipeline across all types, even for queries related to text content. Generally

### 8 Conclusion

In this work, we introduced ViDoRAG, a novel multi-agent RAG framework tailored for visually rich documents. By proposing a coarse-to-fine reasoning process and a multi-modal retrieval strategy, ViDoRAG significantly outperforms existing methods, achieving new SOTA on the ViDoSeek benchmark. Future work will focus on further optimizing the framework’s efficiency while maintaining high accuracy, and exploring its potential in diverse realworld applications, such as education and finance, where visually rich document RAG is crucial.

### Limitations

In addition to the advanced improvements mentioned above, our work has several limitations:

- (1) Potential Bias in Query Construction. The queries in ViDoSeek were constructed by human experts, which may introduce bias in the types of questions and the way they are phrased. This could affect the model’s ability to handle more diverse and natural language queries from real-world users.
- (2) Computational Overhead of ViDoRAG. The multi-agent framework, while effective in enhancing reasoning capabilities, introduces additional computational overhead due to the iterative interactions between the seeker, inspector, and answer agents. This may limit the scalability of the framework in scenarios with strict latency requirements.
- (3) Model Hallucinations. Despite the improvements in retrieval and reasoning, the models used in ViDoRAG can still generate hallucinated answers that are not grounded in the retrieved information. This issue can lead to incorrect or misleading responses, especially when the model is overconfident in its generated content.

In summary, while ViDoRAG demonstrates significant improvements in visually rich document retrieval and reasoning, there are still areas for further enhancement, particularly in terms of generalization to diverse document types, reducing potential biases in query construction, optimizing the computational efficiency of the multi-agent framework, and addressing the issue of model hallucinations. Future work will focus on addressing these limitations to further improve the robustness and applicability of the model.

### Ethical Considerations

Our data does not contain any private or sensitive information, and all content is derived from publicly

available sources. Additionally, the construction and refinement of the dataset were conducted in a manner that respects copyright and intellectual property rights.

### References

Rishabh Agarwal, Avi Singh, Lei Zhang, Bernd Bohnet, Luis Rosias, Stephanie Chan, Biao Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, et al. 2025. Many-shot in-context learning. Advances in Neural Information Processing Systems, 37:76930–76966.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. 2015. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024a. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216.

Zehui Chen, Kuikun Liu, Qiuchen Wang, Jiangning Liu, Wenwei Zhang, Kai Chen, and Feng Zhao. 2024b. Mindsearch: Mimicking human minds elicits deep ai searcher. arXiv preprint arXiv:2407.20183.

Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2024. Colpali: Efficient document retrieval with vision language models. arXiv preprint arXiv:2407.01449.

Ziyan Jiang, Xueguang Ma, and Wenhu Chen. 2024. Longrag: Enhancing retrieval-augmented generation with long-context llms. arXiv preprint arXiv:2406.15319.

Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2024. Nv-embed: Improved techniques for training llms as generalist embedding models. arXiv preprint arXiv:2405.17428.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474.

Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. 2024. Multimodal

arxiv: A dataset for improving scientific comprehension of large vision-language models. arXiv preprint arXiv:2403.00231.

Yanjun Ma, Dianhai Yu, Tian Wu, and Haifeng Wang. 2019. Paddlepaddle: An open-source deep learning platform from industrial practice. Frontiers of Data and Domputing, 1(1):105–115.

Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. 2024. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. Preprint, arXiv:2407.01523.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244.

Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. 2022. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706.

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209.

Nitesh Methani, Pritha Ganguly, Mitesh M Khapra, and Pratyush Kumar. 2020. Plotqa: Reasoning over scientific plots. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1527–1536.

Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2021. Metaicl: Learning to learn in context. arXiv preprint arXiv:2110.15943.

Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389.

Rulin Shao, Jacqueline He, Akari Asai, Weijia Shi, Tim Dettmers, Sewon Min, Luke Zettlemoyer, and Pang Wei W Koh. 2025. Scaling retrieval-based language models with a trillion-token datastore. Advances in Neural Information Processing Systems, 37:91260– 91299.

Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. 2023. Slidevqa: A dataset for document visual question answering on multiple images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 13636–13645.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al.

2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Minzheng Wang, Longze Chen, Fu Cheng, Shengyi Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan Xu, Lei Zhang, Run Luo, et al. 2024. Leave no document behind: Benchmarking long-context llms with extended multi-doc qa. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5627–5646.

Shi Weijia, Min Sewon, Yasunaga Michihiro, Seo Minjoon, James Rich, Lewis Mike, and Yih Wen-tau. 2023. Replug: Retrieval-augmented black-box language models. ArXiv: 2301.12652.

Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Deyu Zhou, Pengjun Xie, and Fei Huang. 2025. Webwalker: Benchmarking llms in web traversal. arXiv preprint arXiv:2501.07572.

Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro. 2023. Retrieval meets long context large language models. arXiv preprint arXiv:2310.03025.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. 2024. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models. arXiv preprint arXiv:2408.04840.

Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. 2024a. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. arXiv preprint arXiv:2410.10594.

Tan Yu, Anbang Xu, and Rama Akkiraju. 2024b. In defense of rag in the era of long-context language models. arXiv preprint arXiv:2409.01666.

Zhenrui Yue, Honglei Zhuang, Aijun Bai, Kai Hui, Rolf Jagerman, Hansi Zeng, Zhen Qin, Dong Wang, Xuanhui Wang, and Michael Bendersky. 2024. Inference scaling for long-context retrieval augmented generation. arXiv preprint arXiv:2410.04343.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986.

### A Additional Experiments Details

Backbones. To thoroughly validate the effectiveness of ViDoRAG, we conducted experiments on various models across various baselines, including both closed-source and open-source models: GPT-4o, Qwen2.5-7B, Llama3.2-3B, Qwen2.5-VL7B(Yang et al., 2024), Llama3.2-Vision-90B. For OCR-based pipelines, we use PPOCR(Ma et al., 2019) to recognize text within documents. Optionally, VLMs can also be employed for text recognition, as their OCR capabilities are quite strong.

Experimental Environments. We conducted our experiments on a server equipped with 8 A100 GPUs and 96 CPU cores. Open-source models require substantial computational resources.

Retrieval Implementation Details. Due to the context length limitations of the model, we use the Top-2K pages to fit the GMM and we restrict the output chunks of the GMM algorithm to be between K/2 and K, we set K = 10 in practice.

### B More Details on Datasets

#### B.1 Annotation Case

Annotated Data Format

- 1 ## JSON Format
- 2 {
- 3 "uid": "04d8bb0db929110f204723c56e5386c1d8d21587_2",
- 4 "query": "What is the temperature of Steam explosion of Pretreatment for Switchgrass and Sugarcane bagasse preparation?",
- 5 "reference_answer": "195-205 Centigrade",
- 6 "meta_info": {
- 7 "file_name": "04d8bb0db929110f204723c586c1d8d21587.pdf ",
- 8 "reference_page": [
- 9 10
- 10 ], # may contain multiple pages
- 11 "source_type": "2d_layout",
- 12 "query_type": "Multi-Hop"
- 13 }
- 14 }

Figure 8: Annotation case in ViDoSeek.

#### B.2 Details on ViDoSeek

More Dataset Statistics. The statistical about ViDoSeek is presented in Table 7. We categorize queries from a logical reasoning perspective into single-hop and multi-hop. Text, Table, Chart and Layout represent different sources of reference.

Dataset Difficulty. ViDoSeek sets itself apart with its heightened difficulty level, attributed to the multi-document context and the intricate nature of

Table 6: Statistics of ViDoSeek.

STATISTIC NUMBER Total Questions 1142

Single-Hop 645 Multi-Hop 497

Pure Text 80 Chart 157 Table 175 Layout 730

its content types, particularly the Layout category. The dataset contains both single-hop and multihop queries, presenting a diverse set of challenges. Consequently, ViDoSeek serves as a more comprehensive and demanding benchmark for RAG systems compared to previous works.

#### B.3 Details on SlideVQA-Refined

Dataset Statistics. We supplemented our experiments with the SlideVQA dataset to demonstrate the scalability of our method. SlideVQA categorizes queries from a logical reasoning perspective into single-hop and multi-hop. Non-span, singlespan, and multi-span respectively refer to answers derived from a single information-dense sentence, reference information that is sparse but located on the same page, and reference information distributed across different pages. The statistical information about dataset is presented in Table 7.

Table 7: Statistics of SlideVQA-Refined.

STATISTIC NUMBER Total Questions 2020 Single-Hop 1486 Multi-Hop 534 Non-Span 358 Single-Spin 1347 Multi-Span 315

Dataset Difficulty. The SlideVQA dataset focuses on evaluating the RAG system’s ability to understand both visually sparse and visually dense information. When multi-hop questions involve reference information spread across different pages, it presents a significant challenge to the RAG system, further demonstrating the effectiveness of our approach.

### C Data Construction Details

To construct the ViDoSeek dataset, we developed a four-step pipeline to ensure that the queries meet our requirements.

- Step 1. Document Collecting. We collected English-language slides containing 25 to 50 pages, covering 12 domains such as economics, technology, literature, and geography, etc.
- Step 2. Query Creation. To make the queries more suitable for RAG over a large-scale collection, our experts constructed queries based on the following requirements: (i) Each query must have a unique answer when paired with the document. (ii) The query must include unique keywords that point to the specific document and pages. (iii) The query should require external knowledge. Additionally, we encouraged constructing queries in various forms and with different sources and reasoning types to better reflect real-world scenarios. Our queries not only focus on types of references, including text, tables, charts, and layouts, but also provide a classification of reasoning types, including single-hop and multi-hop.
- Step 3. Quality Review. To effectively evaluate the generation and retrieval quality of our RAG system, we require queries that yield unique answers, preferably located on a specific page or within a few pages. However, in large-scale retrieval and generation tasks, relying solely on manual annotation is challenging due to human cognitive limitations. To address this, we propose a review module that automatically identifies problematic queries. This module consists of two steps: (i) We prompt LLMs to filter out queries that may have multiple answers across the document collection; for example, the question What is the profit for this company in 2024? might have a unique answer within a single document but could yield multiple answers in a multi-document setting. (ii) For the remaining queries, we retrieve the top-k slides for each query and use a VLM to determine whether each slide can answer the query. If only the golden page can answer the question, we consider it to meet the requirements. If pages other than the golden page can answer the query, we have experts manually evaluate and refine them.
- Step 4. Multimodal Refine. In this final step, we refine the queries that did not meet our standards during the quality review. The goal is to

adjust these queries so they satisfy the following requirements: (i) The refined query should point to specific pages within the large collection with minimal additional information; (ii) The refined query must retain its original meaning. We use carefully designed VLM-based agents to assist us throughout the entire dataset construction pipeline. The prompt is presented in Fig. 9 and Fig. 10, respectively. We will first perform filtering based on semantics, and then conduct a fine-grained review using a multimodal reviewer.

### D More Details about Multi-Agent Generation with Iterative Reasoning

We designed prompts to drive VLMs-based agents, and through our experiments, we found that some open-source models require the design of few-shot examples to learn specific thought patterns. See detailed prompts in Fig. 12, Fig.13 and Fig.14.

Query Reviewer Prompt.

### System Prompt:

Task I have some QA data here, and you can observe that the questions can be divided into two categories:

- The category #A: When you see this question alone without a given document, you are sure to find a unique document in a corpus to provide a unique answer. The question having some key words to help you locate the document from corpus.
- The category #B: When you see this question alone without a given document, you will find hard to locate a document to give a deterministic answer for this question, because you will find multiple candidate documents in a corpus, which may lead to different answers for this question. The question do not have any special key words to help you locate the document from corpus. Examples The number mentioned on the right of the leftside margin? #B What is the date mentioned in the second table? #B What is the full form of PUF? #A What is the number at the bottom of the page, in bold? #B Who presented the results on cabin air quality study in commercial aircraft? #A What is the name of the corporation? #B Which part of Virginia is this letter sent from? #B who were bothered by cigarette odors? #A which cigarette would be better if offered on a thicker cigarette? #A Cigarettes will be produced and submitted to O/C Panel for what purpose? #A What is the heading of first table? #B What is RIP-6 value for KOOL KS? #A Which test is used to evaluate ART menthol levels that has been shipped? #A How much percent had not noticed any difference in the odor of VSSS? #A What is the cigarette code of RIP-6(W/O Filter) 21/4SE? #A what mm Marlboro Menthol were subjectively smoked by the Richmond Panel? #A What are the steps of Weft Preparation between Spinning bobbin and Weaving? #A What level comes between Middle Managers and Non-managerial Employees? #A What are the six parts of COLLABORATION MODEL of the organization where James has a role of leading the UK digital strategy? #A

### User Prompt:

#### Query: {Query Description}

Figure 9: Prompt of Query Reviewer.

Multi-Modal Reviewer Prompt.

### System Prompt:

Please check the image, tell me whether the image can answer my question.

### User Prompt:

#### Query: {Query Description} Image: {Relevant Image}

Figure 10: Prompt of Multi-Modal Reviewer.

Multi-Modal Query Refiner Prompt.

### System Prompt:

#### Task

Rewrite the following question so that it contains specific keywords that clearly point to the provided document, ensuring that it would likely match this document alone within a larger corpus.

#### Instruction

- - Do not add any additional information or context to the question.
- - You should not change the meaning of the question.
- - If the question is already specific and unique, you may leave it unchanged.
- - Please make the sentences you have rewritten more diverse and fluent. Examples
- - Original question: GIS data integration is part of which process?
- - Rewritten question: Citizen Science shows which process the GIS data integration is part of?
- - Original question: What percentage of apps ranked in the top five for including what resulted in a 10,3% Ranking Increase?
- - Rewritten question: According to the App Store Optimization what percentage of apps ranked in the top five for including what resulted in a 10,3% Ranking Increase?
- - Original question: Who is the author of the book, the title of which is the same as the section title of the presentation?
- - Rewritten question: Who is the author of the book, the title of which is the same as the section title of the presentation by Michael Sahota and Olaf Lewitz?
- - Original question: Which region of the world accounts for the highest percentage of revenues in the year 12% GROWTH is achieved?
- - Rewritten question: Which region of the world accounts for the highest percentage of revenues in the year 12% GROWTH is achieved?
- - Original question: What directly follows "conduct market research to refine" in the figure?
- - Rewritten question: What directly follows "conduct market research to refine" in the figure within the Social Velocity Strategic Plan Process?
- - Original question: How can the company which details 24 countries in the report be contacted?
- - Rewritten question: How can the company which details 24 countries in the Global Digital Statistics 2014 report, be contacted?
- - Original question: What substances are involved in the feeding of substrates?
- - Rewritten question: What substances are involved in the feeding of substrates during the production of penicillin?

### User Prompt:

#### Query: {Query Description} Document: {Document Description} Image: {Image File}

Figure 11: Prompt of Multi-Modal Refiner.

Seeker Agent Prompt.

### System Prompt:

#### Character Introduction

You are an artificial intelligence assistant with strong ability to find references to problems through images. The images are numbered in order, starting from zero and numbered as 0, 1, 2 ... Now please tell me what information you can get from all the images first, then help me choose the number of the best picture that can answer the question.

Response Format The number of the image is starting from zero, and counting from left to right and top to bottom, and you should response with the image number in the following format:

{

"reason": Evaluate the relevance of the image to the question step by step , "summary": Extract the information related to the problem , "choice": List[int]

}

Response Example # open-source models sometimes need few-shot instructions.

- Example 1: Question: Who is the person playing a musical instrument in restaurant?

- Response to Example 1: {

"reason": "Image 0 shows that KFC on Renmin Road has a birthday party on February 3rd. I can know that there are musical instruments playing in Shanghai hotels during meals from Image 1. Image 2 shows that this is an invitation letter for the music performance of the New Year ’s Concert at Qintai Art Museum on December 31st. The question is related to the restaurant , and Image 2 is not relevant to the question.", "summary": "KFC on Renmin Road has a birthday party on February 3rd;Shanghai hotels have musical instruments playing during meals;The Qintai Art Museum will hold a New Year ’s concert on December 31st.", "choice": [0, 1]

} Example 2: Question: What time is the train departing from hangzhou to beijing?

- Response to Example 2: {

"reason": "Image 0 shows that Beijing has a temperature of 18 degrees Celsius. Image 0 is a train ticket from hangzhou to beijing showing a departure time of 14:30. Image 1 is a photo of a train station clock , but it’s blurry and hard to read the exact time. Image 2 shows a train schedule with multiple departure times listed. Image 3 is the timetable of Hangzhou Xiaoshan International Airport , and this image is not related to the issue. I think Image 0 is the most relevant to the question.", "summary": "The train ticket shows a departure time of 14:30;The train station clock is blurry;Train schedule shows time.", "choice": [0]

} Example 3: Question: Where can I find a bookstore that sells rare books?

- Response to Example 3: {

"reason": "Image 0 is a street view of a shopping mall with various stores , but no bookstores are visible. Image 1 shows a sign for a bookstore called "Rare Finds Bookstore" specializing in rare books. Image 2 is a map with multiple bookstores marked , but it doesn ’t specify if they sell rare books. Image 3 is a photo of a library , which is not a place to buy books. Image 5 is a rare books list , which includes the names and prices of various books. ", "summary": "The shopping mall has no visible bookstores;Rare Finds Bookstore specializes in rare books;Map shows multiple bookstores but doesn ’t specify rarity;Library is not for buying books;The price list includes the prices and names of rare books.", "choice": [1, 5]

}

### User Prompt:

#### Query: {Query Description} Images: {Candidate Images} Reflection: {Feedback From Inspector}

Figure 12: Prompt of Seeker Agent.

Inspector Agent Prompt.

### System Prompt:

Character Introduction You are an artificial intelligence assistant with strong ability to answer questions through images. Please provide the answer to the question based on the information provided. Task Description

- - If the images can answer the question, please answer the question directly.
- - If the images are not enough to answer the question, please tell me which pictures are related to the question. Response Format
- - If the images can answer the question, please answer the question directly: {

"reason": Solve the question step by step , "answer": Answer the question briefly with several words , "reference": List[int]

}

- - If the images are not enough to answer the question, please tell me what additional information

you need, and tell me which pictures are related to the question:

{

"reason": Evaluate the relevance of the image to the question one by one, and solve the question step by step , "information": Carefully clarify the information required , "choice": List[int]

}

Response Example # open-source models sometimes need few-shot instructions.

- - Example 1: {

"reason": "The image only provides information about the Bohr Model and does not include details about subshells in the Modern Quantum Cloud Model.", "information": "More information about the Bohr Model.", "choice": []

}

- - Example 2: {

"reason": "The images provide information about the #swallowaware campaign , including its aims and how they were measured. However , specific details on the success metrics are not clearly visible in the provided images.", "information": "More information about the success metrics of the #swallowaware campaign.", "choice": [0, 1]

}

- - Example 3: {

"reason": "We first found the restaurant name on the menu , and then we located the restaurant in the city center on the map.", "answer": "city center", "reference": [2, 3]

}

- - Example 4: {

"reason": "The entire process , from input , processing to output , ultimately produces a product with a purity of 42%.", "answer": "42%", "reference": [0]

}

### User Prompt:

#### Query: {Query Description} Plan: {Thought From Last Step.} Images: {Images Pending Review.}

Figure 13: Prompt of Inspector Agent.

Answer Agent Prompt.

### System Prompt:

#### Character Introduction

You are an artificial intelligence assistant with strong ability to answer questions through images. Please provide the answer to the question based on the information provided and tell me which pictures are your references.

Response Format Please provide the answer in JSON format:

{

"reason": Solve the question step by step , "answer": Answer the question briefly with several words , "reference": List[int]

}

### User Prompt:

#### Query: {Query Description} Draft Answer: {Draft Answer From Inspector} Images: {Reference Images}

Figure 14: Prompt of Answer Agent.

