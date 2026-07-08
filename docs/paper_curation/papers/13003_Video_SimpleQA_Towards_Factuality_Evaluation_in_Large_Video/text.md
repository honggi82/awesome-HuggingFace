## Video SimpleQA: Towards Factuality Evaluation in Large Video Language Models

### Meng Cao1,2*, Pengfei Hu1,2*, Yingyao Wang2, Jihao Gu2, Haoran Tang1, Haoze Zhao1, Chen Wang2, Jiahua Dong2, Wangbo Yu3, Ge Zhang4, Jun Song2, Xiang Li2, Bo Zheng2, Ian Reid1, Xiaodan Liang1,5† https://videosimpleqa.github.io/

1MBZUAI 2Taobao&Tmall Group 3Peking University 4MAP 5Sun Yat-sen University

[Figure 1]

[Figure 2]

∗Equal contribution. †Corresponding author.

# arXiv:2503.18923v2[cs.CV]13Aug2025

|[Figure 3]<br><br>Question: What could the demonstrated performance imply for future abilities …? Answer: It implies that Atlas can perform assisting tasks to a human facilitator and may also be able to ….. [Subjective speculation]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]|
|---|

(a)

Climate Engineering

Biochemical Engineering

Polymer Engineering

[Figure 9]

Environmental

Entertainment

Geophysical

Hydrology

Literature

Aerospace Engineering

Music

Art

Sports

Dance

Process

Theatre

Chemical

Photography

Environmental & Geophysical

Mechatronics

Film

Arts & Recreation

Chemical & Process

Games

Electrical

[Figure 10]

[Figure 11]

Belief

Mechanical

Religion

UrbanPlanning

StructuralEngineering

Philosophy

Mechanical & Electrical

Ethics

|[Figure 12]<br><br>Question: Assume that 2.24 liters of gas fully participates in the reaction shown in the video …., how many grams of precipitate are produced approximately? Answer: 10.0 [Entangled fact grounding and reasoning]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]|
|---|

Politics

Civil

Architecture

Civil&Architecture

Beliefs&Institutions

Flags

Government

Forecasting

[Figure 19]

AtmosphericPhenomena

Law

Engineering

Society & Culture

Meteorology

Climate

People

Language

Weather

EndangeredSpecies

Objects Food

Nature

Microorganisms

Language&MaterialCulture

Flora&Fauna

Clothing

Fungi

[Figure 20]

[Figure 21]

Transportation

Plantae

Science

(b) Existing knowledge-based video benchmark (MMWorld, MMVU)

Instruments

MarineOrganisms

Geology&Landscapes

Animalia

Tools

History&Heritage

History

CoastalLandforms

|(c) Our Video SimpleQA benchmark<br><br>Question: Which national rugby union team is renowned for performing the same dance shown in the video before international matches? Answer: The New Zealand national rugby union team<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]|
|---|

Events

VolcanicFeatures

Geomorphology

Places

Applied sciences

[Figure 28]

Physical sciences

Archaeology

Rocks&Minerals

Formal sciences

HeritageSites

Landscapes

Genealogy

Fossils

Life sciences

Physics

Data Science

Chemistry

Agricultural Science

Astronomy

Robotics

Earth sciences

Technology

Materials Science

Logic

Statistics

Atmospheric Science

Geophysics

Computer Science

Biology

Mathematics

Neuroscience

Medicine

Ecology

Genetics

(c) Our Video SimpleQA benchmark

Figure 1: (a) The taxonomy of Video SimpleQA benchmark; (b) Illustrations of existing knowledge-based video benchmarks (Zhao et al. 2025; He et al. 2024a) which may contain subjective speculation or conflate factual grounding with reasoning skills (i.e., mathematical calculation); (c) Illustrations of our Video SimpleQA benchmark with the fact-seeking question and definitive & short-form answer with multi-hop external facts verified.

###### Abstract

in a short format with minimal scoring variance; 4) Temporal grounded required: Requiring answers to rely on one or more temporal segments in videos, rather than single frames. We extensively evaluate 33 state-of-the-art LVLMs and summarize key findings as follows: 1) Current LVLMs exhibit notable deficiencies in factual adherence, with the best-performing model o3 merely achieving an F-score of 66.3%; 2) Most LVLMs are overconfident in what they generate, with self-stated confidence exceeding actual accuracy; 3) Retrieval-augmented generation demonstrates consistent improvements at the cost of additional inference time overhead; 4) Multi-hop QA demonstrates substantially degraded performance compared to single-hop sub-QAs, with first-hop object/event recognition emerging as the primary bottleneck. We position Video SimpleQA as the cornerstone benchmark for video factuality assessment, aiming to steer LVLM development toward verifiable grounding in real-world contexts.

Recent advancements in Large Video Language Models (LVLMs) have highlighted their potential for multi-modal understanding, yet evaluating their factual grounding in videos remains a critical unsolved challenge. To address this gap, we introduce Video SimpleQA, the first comprehensive benchmark tailored for factuality evaluation in video contexts. Our work differs from existing video benchmarks through the following key features: 1) Knowledge required: demanding integration of external knowledge beyond the video’s explicit narrative; 2) Multi-hop fact-seeking question: Each question involves multiple explicit facts and requires strict factual grounding without hypothetical or subjective inferences. We also include per-hop single-fact-based sub-QAs alongside final QAs to enable fine-grained, stepby-step evaluation; 3) Short-form definitive answer: Answers are crafted as unambiguous and definitively correct

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

|sub-question-1: Which country is the tourist attraction in the video located in?<br><br>sub-answer-1: Greece<br><br>ref. source-1: https://commons.wikimedia.org/wiki/Template:Motd/2011-01-16<br><br>sub-question-2: What is the capital of Greece? sub-answer-2: Athens<br><br>ref. source-2: https://en.wikipedia.org/wiki/Greece<br><br>sub-question-3: What peninsula is Athens located on? sub-answer-3: Balkan peninsula<br><br>ref. source-3: https://en.wikipedia.org/wiki/Athens<br><br>sub-question-4: Who coined the term "Balkan Peninsula"? sub-answer-4: August Zeune<br><br>ref. source-4: https://en.wikipedia.org/wiki/Balkans multi-hop-question: Who coined the term for the peninsula where the capital of the country containing the tourist attraction in the video is located? multi-hop-answer: August Zeune<br><br><br><br><br>|
|---|

|sub-question-1: Which short science-fiction film is excerpted in the video?<br><br>sub-answer-1: Tears of Steel<br><br>ref. source-1: https://commons.wikimedia.org/wiki/Template:Motd/2013-01-26<br><br>sub-question-2: Who is the director of Tears of Steel? sub-answer-2: Ian Hubert<br><br>ref. source-2: https://en.wikipedia.org/wiki/Tears_of_Steel<br><br>sub-question-3: Which country is Ian Hubert from? sub-answer-3: USA<br><br>ref. source-3: https://baike.baidu.com/item/Ian%20Hubert<br><br>sub-question-4: Who is the first holder of Chief Justice of USA? sub-answer-4: John Jay<br><br>ref. source-4: https://en.wikipedia.org/wiki/John_Jay<br><br><br><br><br><br><br>multi-hop-question: Who is the first holder of Chief Justice of the country which is the director of the short science-fiction excerted in the video from? multi-hop-answer: John Jay|
|---|

Figure 2: Four-hop examples in Video SimpleQA including the final multi-hop QA and the decomposed per-fact sub-QAs.

### Introduction

The substantial advancements in Large Language Models (LLMs) (Achiam et al. 2023; Reid et al. 2024; Touvron et al.

- 2023) over the past few years have inaugurated a new frontier in artificial intelligence. Despite their remarkable capabilities, the factuality concern (Wang et al. 2024c; Akhtar et al. 2023; Wang et al. 2023) remains a critical challenge, i.e., how to ensure that the generated contents are consistent with factual knowledge and grounded in credible sources.1

Existing research has primarily focused on evaluating factuality in text-based (Yu et al. 2022; Pan et al. 2024; Lin, Hilton, and Evans 2022; Chern et al. 2023; Gou et al. 2023) and image-based (Marino et al. 2019; Wang et al. 2015, 2017; Zellers et al. 2019; Jain et al. 2021) scenarios. Recently, the SimpleQA benchmark (Wei et al. 2024) introduced by OpenAI and its subsequent works (He et al. 2024b; Gu et al. 2025; Cheng et al. 2025) streamline the factuality evaluation by considering only concise and fact-seeking questions, which enables standardized and tractable assessments. However, extending this paradigm to video contexts is under-explored and presents unique challenges due to the inherent temporal dynamics and procedural knowledge.

To bridge this gap, we present Video SimpleQA, a comprehensive factuality evaluation benchmark tailored for Large Video Language Models (LVLMs). As shown in Figure 1 and Figure 2, Video SimpleQA is composed of multi-hop fact-seeking questions and short-form definitive answers. Compared to previous video benchmarks, Video SimpleQA stands out with the following advancements:

- • Knowledge required: Beyond comprehending the visual content, Video SimpleQA necessitates the integration of external knowledge that is not explicitly presented in the video narrative, e.g., domain-specific information, contextual background, commonsense.
- • Multi-hop fact-seeking question: Questions necessitate strict adherence to factual grounding principles, eliminating any hypothetical or subjective inferences. In addition,

1Please refer to (Wang et al. 2023, 2024c) for the differentiation between the factuality and the similar hallucination concepts.

each question is constructed to involve multiple explicitly identifiable facts. To achieve this, beyond the final multi-hop question-answer (QA) pairs, we additionally provide per-fact specific sub-QA annotations, which facilitate fine-grained evaluation of model performance at each fact-grounding hop and help pinpoint exactly which hop fails in factual grounding (c.f. Figure 2).

- • Short-form definitive answer: All the answers are unambiguous, universally agreed upon, consistent over time, and invariant to individual perspectives. Following SimpleQA (Wei et al. 2024), the answers also advocate the short-form paradigm, which establishes reliable factual assessment with low run-to-run variance.
- • Temporal grounded: Answering questions in Video SimpleQA should refer to one or more temporal segments in the video, rather than relying on a single frame.

While existing knowledge-based (Garcia et al. 2020; Zhang et al. 2024b; Hu et al. 2025) and recent disciplinebased (Zhao et al. 2025; He et al. 2024a) benchmarks may appear similar to our Video SimpleQA , our benchmark features several distinct characteristics (c.f. Table 1):

- • Open-domain: While KnowIT-VQA (Garcia et al. 2020) is constrained to TV shows, and MMVU (Zhao et al.

2025) as well as MMWorld (He et al. 2024a) focus on discipline-specific knowledge, our Video SimpleQA encompasses open-domain video types and questions.

- • Objective QA: Our benchmark is explicitly designed for factuality evaluation through objective factual assertions, in contrast to existing benchmarks that often involve varying degrees of subjectivity, even those focusing on disciplinary knowledge. For instance, as shown in Figure 1(b) top, MMWorld (He et al. 2024a) includes cases requiring predictions about a robot’s future capabilities—introducing subjective speculation and personal judgment, which deviates from our goal of evidencebased and objective evaluation.2
- • Factuality exclusive: Discipline-based benchmarks of2More examples are available in the supplementary material.

- Table 1: Benchmark comparisons across key dimensions: video domain scope, knowledge-driven focus, objective factuality focus, exclusivity of factual evaluation, multi-hop fact decomposition, and external evidence source.

Video Knowledge Objective Factuality Multi-hop Evidence domain driven QA exclusive fact decomp. source

Benchmarks

Video-MME (Fu et al. 2024) Open ✗ ✗ ✗ ✗ ✗ MMBench-Video (Fang et al. 2024) Open ✗ ✗ ✗ ✗ ✗ Video-MMMU (Hu et al. 2025) Professional ✓ ✗ ✗ ✗ ✗ MMVU (Zhao et al. 2025) Discipline ✓ ✗ ✗ ✗ ✓ MMWorld (He et al. 2024a) Discipline ✓ ✗ ✗ ✗ ✗ WorldQA (Zhang et al. 2024b) Open ✓ ✗ ✗ ✗ ✗ KnowIT-VQA (Garcia et al. 2020) TV shows ✓ ✗ ✗ ✗ ✗ Video SimpleQA Open ✓ ✓ ✓ ✓ ✓

ten conflate external knowledge retrieval with reasoning skills (e.g., numerical calculations). For example, the case in Figure 1(b) bottom from MMVU (Zhao et al. 2025) requires LVLMs to both recognize a chemical reaction in the video and perform numeric computations based on the question context. This coupling makes it difficult to pinpoint the error source—whether due to incorrect fact identification (e.g., failing to detect the reaction) or faulty reasoning (e.g., miscalculating). In contrast, Video SimpleQA exclusively focuses on fact identification, providing a clearer assessment of LVLMs’ fact-grounding ability.2

• Multi-hop fact decomposition: As shown in Figure 2, Video SimpleQA includes not only the final multihop QA pairs but also the decomposed per-fact sub-QAs, enabling fine-grained evaluations. While some cases in MMVU (Zhao et al. 2025) also involve knowledge from multiple external sources, they do not provide such explicit per-fact decomposition, making it difficult to assess how each individual fact contributes to the final answer.

We conduct comprehensive evaluations of 33 state-of-theart LVLMs on Video SimpleQA, revealing several critical insights: 1) Significant performance gap: Both proprietary and open-source LVLMs substantially underperform compared to human expertise; 2) Overconfidence bias: Most LVLMs exhibit systematic overconfidence in their predictions despite output inaccuracies, with notable variations in calibration quality (c.f. Figure 7); 3) Efficiencyperformance tradeoff: Retrieval-Augmented Generation (RAG) yields significant gains at the cost of inference efficiency (c.f. Table 3); 4) Multi-hop performance bottleneck: Multi-hop QA performance significantly lags behind single-hop sub-tasks, with the initial video-grounded hop acting as the primary bottleneck (c.f. Table 4). More experimental findings are available in the supplementary material.

### Related Work

Factuality Benchmarks. Factuality is the capability of LLMs to generate content that aligns with factual information, which can be substantiated by authoritative sources such as Wikipedia or textbooks (Akhtar et al. 2023; Wang et al. 2024c). Evaluating LLM factuality presents a nontrivial challenge and various benchmarks are proposed in

the text-based (Lin, Hilton, and Evans 2022; Chern et al.

- 2023; Gou et al. 2023; Wei et al. 2024; He et al. 2024b) and image-based scenarios (Marino et al. 2019; Wang et al. 2017; Jain et al. 2021; Gu et al. 2025). As one of the pioneering works, TruthfulQA (Lin, Hilton, and Evans 2022) specifically targets imitative falsehoods in LLM-generated responses, which stem from erroneous preconceptions or knowledge gaps. Recently, the SimpleQA series of works (Wei et al. 2024; He et al. 2024b; Gu et al. 2025; Cheng et al. 2025) facilitate factuality evaluation by constraining the scope to short, fact-seeking questions with simple answers, making factuality assessment more tractable compared to previous long, open-ended model outputs. Despite of this, the community urgently needs a standard benchmark for trustworthy factuality evaluation in video contexts. To address this gap, our Video SimpleQA emerges.

Video Understanding Benchmarks. Recently, video benchmarks have been designed for evaluations in comprehensive tasks, including temporal perception (Li et al. 2024b), reasoning (Cai et al. 2024; Chen et al. 2024), navigation (Yang et al. 2024), long-form comprehension (Song et al. 2024; Chandrasegaran et al. 2024), etc. However, current video benchmarks largely overlook factuality evaluation, resulting in a lack of assessment for LVLMs’ ability to generate factually accurate responses. Compared to video hallucination benchmarks (Wang et al. 2024d; Guan et al.

- 2024; Zhang et al. 2024a), which primarily assess models’ adherence to video contents, our focused factuality evaluation emphasizes the model’s alignment with verifiable external world knowledge (Wang et al. 2023, 2024c).

Differentiation from Knowledge-based and Disciplinebase Benchmarks. Existing knowledge-based (Zhang et al. 2024b; Hu et al. 2025) and discipline-based benchmarks (Zhao et al. 2025; He et al. 2024a) either contain hypothetical/subjective reasoning (e.g., the categories of societal norms and social interactions in WorldQA (Zhang et al. 2024b)) or narrow their scopes to single TV show (Garcia et al. 2020) or discipline-related knowledge (Zhao et al. 2025; He et al. 2024a). Our Video SimpleQA addresses these limitations by enforcing objective factuality verification and ensuring diversity across various categories. In addition, we introduce per-hop factgrounded QA for fine-grained evaluations.

###### Video & Encyclopedia Collection QAAnnotation Quality Control

[Figure 51]

[Figure 52]

[Figure 53]

Video Download

Retrieval-Augmented LLM-based Iterative Generation Generation

###### Human-in-the-loop Verification

Difficulty Filtering

Human CrossVerification

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Generator Critic

Wikimedia Commons:Motd Search Engines

Knowledge required Fact-seeking question

Validation by expert annotators

Filter out questions answerable by all models

Cross-validation by expert annotators

Definitive answer

……

###### Figure 3: An overview of the construction pipeline of Video SimpleQA .

[Figure 67]

LLM-based Iterative Generation: The iterative generation process involves two LLMs, a generator LLM for initial QA pair synthesis and a critic LLM for quality assessment. Specifically, the generator receives videos and encyclopedic knowledge to generate candidate QA pairs. Subsequently, the critic LLM systematically evaluates output compliance with predefined quality criteria, providing targeted feedback for refinement. This iterative process continues for up to three refinement cycles, with non-compliant outputs being discarded post-final iteration to ensure rigorous quality control. Both generator and critic are implemented as GPT-4o.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Associated Descriptions

Barbary Ground Squirrel ( Atlantoxerus getulus ) at Paseo Maritimo in Antigua, Fuerteventura , Canary Islands.

- 1. Barbary Ground Squirrel: The Barbary ground squirrel (Atlantoxerus getulus) is a species of rodent in the family Sciuridae. It is monotypic within the genus Atlantoxerus. ……
- 2. Fuerteventura: Fuerteventura (Spanish: [ˌfweɾteβenˈtuɾa]) is one of the Canary Islands, in the Atlantic Ocean, geographically part of Macaronesia, and politically part of Spain. ……

RAG Results

The explicit construction criteria are as follows: 1) Knowledge required: The questions should necessitate both video content and relevant external factual knowledge. Those that can be answered solely based on either source should be excluded. For example, two questions that should be eliminated are: What color is the insect in the video? (which relies solely on video content) and Which president of the United States was Obama? (which relies solely on external knowledge); 2) Fact-seeking question: The generated question should be factually grounded without any hypothetical or subjective reasoning; 3) Definitive answer: To ensure a rigorous evaluation, each question must have a single, unambiguous, and time-invariant answer. To achieve this, we explicitly define the level of granularity in question phrasing. For example, we use “which year” instead of “when” and “which city” instead of “where” to eliminate ambiguity; 4) Short-form answer: The answers should be in a short-form format; 5) Multi-hop facts: To answer the question, it requires involving multiple factual sources; 6) Temporal grounded: The questions are grounded in one or more video segments rather than specific frames.

Figure 4: The encyclopedia collection process including the raw associated descriptions in Wikimedia and the RAG results3 for the specialized terms extracted by GPT-4o.

### Video SimpleQA

The construction pipeline of Video SimpleQA is illustrated in Figure 3, which includes video & encyclopedia collection, QA annotations, and quality control.

##### Video & Encyclopedia Collection

Video Collection: To ensure broad coverage, we curate the knowledge-intensive videos from the “Media of the Day” page of Wikimedia Commons3 together with the accompanied brief descriptions or scientific illustrations. Note that files on the “Media of the Day” page are freely licensed, which avoids introducing any potential copyright concerns.

Encyclopedia Collection: As shown in Figure 4, although the associated textual descriptions on the Wikimedia page provide related descriptions, the explanations for the specialized terms (e.g., Barbary Ground Squirrel, Fuerteventura) still lack formal definitions. To construct a more comprehensive encyclopedia, we leverage GPT-4o (OpenAI 2024) to extract key terms from the original descriptions and then obtain detailed explanations for these terms via RAG. Specifically, we apply LlamaIndex (Liu 2022) as the RAG method, with search results from Google and Bing as data sources.

Human-in-the-loop Verification: Through the iterative generation, we obtain QA annotations of reasonable quality. To further enhance the reliability, we train expert annotators to refine the LLM-generated QA annotations. The expert annotators are first required to watch the complete video and examine the collected encyclopedic knowledge. They then evaluate whether the LLM-generated QA annotations meet the specified criteria and manually revise them if necessary.

##### QA Annotations

To ensure multi-hop fact grounding, annotators were additionally instructed to decompose each multi-hop QA into a series of sub-QAs (c.f. Figure 2). The decomposition follows two rules: 1) Single fact per sub-QA: Each sub-QA targets a single fact that is independently verifiable; 2) Referential dependency: Each sub-QA builds upon the answer to the previous one, forming the step-by-step fact chaining.

The annotation pipeline for Video SimpleQA follows a two-stage process: (1) automated LLM-based iterative generation and (2) human-in-the-loop verification refinement.

3https://commons.wikimedia.org/wiki/Commons: Media of the day

- Table 2: Evaluation results (%) of open-source and proprietary multi-modal LLMs on Video SimpleQA . For metrics, CO, NA, IN, and CGA denote “Correct”, “Not attempted”, “Incorrect”, and “Correct given attempted”, respectively. For subtopics, ENG, NAT, SCI and SAC represent “Engineering”, “Nature”, “Science” and “Society and Culture”.

Overall results on 5 metrics F-score on 4 primary categories CO IN↓ NA↓ CGA F-score ENG NAT SCI SAC

Model

Human Performance Human Open-book 87.0 5.0 8.0 94.6 90.6 85.2 83.7 89.1 96.8 Human Closed-book 59.0 18.5 22.5 76.1 66.5 58.4 52.8 54.2 80.6 Proprietary Multi-modal LLMs

- o4-mini (OpenAI 2025b) 53.7 45.3 0.9 54.2 54.0 44.3 59.4 56.8 54.0
- o3 (OpenAI 2025b) 66.3 33.6 0.1 66.4 66.3 63.0 71.3 63.5 68.8 GPT-4.5 (OpenAI 2025a) 52.9 42.5 4.6 55.4 54.1 49.5 57.5 57.9 51.4 GPT-4o (OpenAI 2024) 47.7 45.9 6.4 51.0 49.3 45.1 57.1 52.7 45.4 Claude Sonnet 4 (Anthropic 2025b) 32.8 51.2 16.0 39.0 35.6 33.0 34.3 37.9 35.0 Claude 3.7 Sonnet (Anthropic 2025a) 32.6 47.3 20.1 40.8 36.2 24.2 40.3 41.9 34.5 Claude 3.5 Sonnet2 (Anthropic 2024) 33.7 58.2 8.1 36.7 35.2 26.1 37.2 38.5 35.4 Claude 3.5 Sonnet (Anthropic 2024) 31.5 53.5 15.0 37.0 34.0 26.8 36.0 38.1 32.5 Gemini 2.5 Pro (Comanici et al. 2025) 61.2 34.3 4.5 64.1 62.6 53.5 65.8 67.1 61.5 Gemini 2.5 Flash (Comanici et al. 2025) 53.7 34.9 11.3 60.6 57.0 46.0 61.6 61.4 56.2 Qwen-VL-Max (Bai et al. 2023) 39.2 57.1 3.7 40.7 39.9 27.4 46.2 48.8 35.1 Qwen-VL-Plus (Bai et al. 2023) 21.9 63.3 14.7 25.7 23.7 10.5 25.6 30.5 21.8

Open-source Multi-modal LLMs InternVL3-78B (Zhu et al. 2025) 33.7 65.6 0.7 33.9 33.8 25.4 41.2 38.6 30.6 InternVL3-38B (Zhu et al. 2025) 31.4 67.7 0.9 31.7 31.5 21.3 33.3 35.7 31.8 InternVL3-14B (Zhu et al. 2025) 24.9 73.3 1.8 25.4 25.2 14.6 32.3 28.4 24.6 InternVL3-9B (Zhu et al. 2025) 22.6 72.9 4.5 23.7 23.1 12.8 33.2 27.9 19.9 InternVL3-8B (Zhu et al. 2025) 23.3 75.2 1.5 23.7 23.5 16.2 30.7 25.6 22.4 Qwen2.5-VL-72B (Bai et al. 2025) 38.7 57.3 4.0 40.3 39.5 26.1 47.0 48.2 34.7 Qwen2.5-VL-32B (Bai et al. 2025) 30.3 67.1 2.7 31.1 30.7 18.1 39.3 37.4 27.0 Qwen2.5-VL-7B (Bai et al. 2025) 24.7 71.2 4.1 25.8 25.3 13.8 25.6 30.8 25.1 Qwen2-VL-72B (Wang et al. 2024a) 32.7 59.0 8.3 35.7 34.2 20.2 39.0 40.0 33.2 Qwen2-VL-7B (Wang et al. 2024a) 22.4 69.4 8.2 24.4 23.4 15.9 23.9 25.1 25.0 LLaVA-1.5-13B (Liu et al. 2023) 19.3 76.7 4.1 20.1 19.7 11.6 21.2 21.9 20.8 LLaVA-1.5-7B (Liu et al. 2023) 16.1 78.5 5.4 17.1 16.6 8.9 19.2 19.0 17.1 LLaVa-NeXT-Video-34B (Liu et al. 2024) 11.2 83.8 4.9 11.8 11.5 7.6 11.5 10.3 14.5 LLaVa-NeXT-Video-7B (Liu et al. 2024) 9.3 52.9 37.8 14.9 11.4 7.4 15.1 14.5 8.7 LLaVA-OneVision-72B (Li et al. 2024a) 25.4 73.6 1.0 25.7 25.5 15.9 25.3 28.5 27.3 LLaVA-OneVision-7B (Li et al. 2024a) 18.9 76.6 4.5 19.8 19.3 12.1 26.3 21.3 18.4 DeepSeek-VL2 (Wu et al. 2024) 3.2 49.1 47.7 6.1 4.2 3.0 4.4 3.0 5.9 DeepSeek-VL2-Small (Wu et al. 2024) 5.9 52.1 42.1 10.1 7.4 3.9 10.1 9.5 6.2 DeepSeek-VL2-Tiny (Wu et al. 2024) 16.1 75.6 8.3 17.6 16.8 9.6 27.1 17.5 16.0 Kimi-VL (Team et al. 2025a) 18.3 44.4 37.3 29.1 22.4 14.9 19.8 25.2 24.5 Keye-VL (Team et al. 2025b) 25.4 53.9 20.7 32.0 28.3 15.3 23.1 37.2 26.6

##### Quality Control

Difficulty Filtering. To ensure an appropriate level of assessment difficulty, we establish filtering rules to exclude questions that are easy to answer. In particular, questions correctly answered by all four state-of-the-art models, including GPT-4o (OpenAI 2024), Claude Sonnet 4 (Anthropic 2025b), Gemini 2.5 Pro (Comanici et al. 2025), and Qwen-VL-Max (Bai et al. 2023) are deemed insufficiently challenging and consequently excluded from our benchmark. This filtering strategy ensures our dataset maintains a sufficient level of difficulty for meaningful evaluations.

Human Cross-verification. To further enhance the dataset quality, a rigorous human validation process is implemented. Each question is independently evaluated by two annotators for compliance with our predefined criteria. Questions are discarded if either annotator deems them noncompliant. Meanwhile, annotators are required to verify answers against authoritative sources (such as Wikipedia). Finally, the final dataset undergoes security auditing to address potential security issues. All these stringent human verification processes ensure both the accuracy of our dataset and its adherence to established criteria.

Flora &

Formal sciences

Applied sciences

Identity Certification

Statistics of Video SimpleQA Value 2/3/4-hop QA pairs 928 / 469 / 107 Total QA pairs 1504 Question Length (avg/max) 15.64 / 38 Answer Length (avg/max) 1.28 / 9 Unique Videos 1079 Video Length (Seconds, avg/max) 201 / 8763 Number of primary category 4 Number of secondary category 15 Number of tertiary category 84

Possession Tracing

Fauna Geology & Landscapes

Physical sciences

Meteorology Arts & Recreation

Life sciences

Causal Relationship

Relational Attribution

Chemical & Process

History & Heritage Beliefs & Institutions Language & Material Culture

Mechanical…

Process Verification

Spatial Localization Temporal Grounding

Factual Identification

Environmental & Geophysical

Civil & Architecture

(a)

(b)

(c)

###### Figure 5: (a) Video distribution at the secondary level; (b) Question type distribution; (c) Key statistics.

Dataset Statistics. The key statistics of Video SimpleQA are demonstrated in Table 5c. As shown, it consists of 1079 videos with 1504 expert-annotated QA pairs. The videos span 4 primary categories, 15 secondary categories and 84 tertiary categories. The average lengths of questions and answers are 15.64 and 1.28 words, respectively, aligning with our intended short-form design. The video distribution at the secondary level is demonstrated in Figure 5a. The question type distribution is visualized in Figure 5b.

### Experiments

##### Experimental Setup

Evaluated Models. We benchmark comprehensive stateof-the-art LLMs, including 12 proprietary models, including o4-mini, o3, GPT-4.5, GPT-4o, Claude Sonnet 4, Claude 3.7 Sonnet, Claude 3.5 Sonnet series, Gemini 2.5 series, and Qwen-VL series, and 21 open-source models, including InternVL3 series, Qwen2.5-VL series, Qwen2VL series, LLaVA-1.5 series, LLaVA-NeXT-Video series, LLaVA-OneVision series, DeepSeek-VL2 series, Kimi-VL, and Keye-VL. Following Video-MME (Fu et al. 2024), we maximize frame utilization of each model by inputting the maximum frames that fit within its context window.

Evaluation Metrics. Following SimpleQA (Wei et al.

- 2024), we set up five evaluation metrics: (1) Correct: The predicted answer comprehensively contains all key information from the reference answer while containing no contradictory elements. (2) Incorrect: The predicted answer contradicts the reference answer. The indirect or equivocal responses (e.g., “possibly”, “I think, although I’m not sure”) are also considered incorrect. (3) Not attempted: The reference answer is not fully given in the predicted answer, and no statements in the answer contradict the gold target. (4) Correct given attempted: The ratio of correctly answered questions among attempted ones. (5) F-score: The harmonic mean values between correct and correct given attempted metrics. We follow the paradigm of LLM-as-a-Judge (Gu

- et al. 2024) and employ o3 as the judge model.

##### Experimental Findings4

The evaluation results on Video SimpleQA are presented in Table 2, and key findings are summarized as follows:

4More experiments are available in the supplementary material.

Video SimpleQA is challenging: To assess human performance on Video SimpleQA , we sample 200 instances and recruit five participants to independently complete the tasks under two distinct conditions: with access to external resources (e.g., Internet, textbooks) and without such access. These configurations correspond to the human open-book and human closed-book settings documented in Table 2.

Compared to the human open-book performance, both open-source and proprietary models demonstrate suboptimal performance. Specifically, the top-performing proprietary model, o3 (OpenAI 2025b), achieves an F-score of 66.3%. Open-source models exhibit even poorer results, with the best-performing one, Qwen2.5-VL-72B (Wang et al. 2024a) attaining only 39.5% F-score. This demonstrates that LVLMs still exhibit limited capability in factuality adherence within video contexts, while also highlighting the necessity of establishing Video SimpleQA .

LVLMs are overconfident in what they generate: All models exhibit higher IN values (incorrect predictions) than NA values (non-attempted responses), indicating a prevalent tendency to generate answers despite insufficient factual knowledge. To further investigate this overconfidence phenomenon, we conduct calibration experiments (Guo et al. 2017) to examine whether language models “know what they know”, i.e., whether the model’s assessed confidence scores align with the actual likelihood of its responses being correct. Specifically, we instruct LVLMs to self-assess confidence scores (0-100) for their predictions. Responses are grouped into confidence intervals (10-point bins), and we calculate interval accuracy (correct predictions per bin). As shown in Figure 7, except for o3, which demonstrates superior calibration, all other models mostly fall below the perfect calibration line, indicating systematic overconfidence.

RAG yields significant gains at the cost of inference efficiency: We explore RAG to facilitate Video SimpleQA benchmark comprehension in a three-step approach: 1) Prompting GPT-4o with video and questions to extract key textual entities; 2) Applying LlamaIndex with Google and Wikipedia as sources to retrieve relevant information based on these extracted key entities; 3) Augmenting the input question with the retrieved information.

As shown in Table 3, RAG achieves consistent and significant F-score improvements over vanilla models. For instance, when integrated with Claude Sonnet 4 (Anthropic

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

|sub-question-1: What is the training program in the video?<br><br>sub-answer-1: Dog agility o3 answer-1: Dog agility training<br><br>sub-question-2: Which country can the origin of dog agile be traced back to?<br><br>sub-answer-2: The United Kingdom o3 answer-2: England<br><br>sub-question-3: What is the primary language in the United Kingdom?<br><br>sub-answer-3: English o3 answer-3: English is the primary<br><br>sub-question-4: What language family does English belong to?<br><br>sub-answer-4: Indo-European languages o3 answer-4: Germanic<br><br><br><br><br>multi-hop-question: What language family does the language mainly spoken in the country where the training program depicted in the video originates belong to? multi-hop-answer: Indo-European languages o3 answer: Germanic Lack of Knowledge<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]|
|---|

|sub-question-1: What's the name of the film in the video?<br><br>sub-answer-1: Caminandes- Llama Drama o3 answer-1: Cammande<br><br>sub-question-2: Who directed the film "Caminandes- Llama Drama”<br><br>sub-answer-2: Pablo Vazquez o3 answer-2: Pablo Vazquez<br><br>sub-question-3: Which country is Pablo Vazquez originally from?<br><br>sub-answer-3: Argentina o3 answer-3: Argentina<br><br>sub-question-4: What is Argentina's motto?<br><br>sub-answer-4: En unión y libertad o3 answer-4: En unión y libertad<br><br><br><br><br>multi-hop-question: What is the motto of the country which the director of the film in the video comes from? multi-hop-answer: En unión y libertad o3 answer: Order and Progress Perception Error<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]|
|---|

Figure 6: Visualizations of per-hop evaluation results of o3 (OpenAI 2025b).

Claude Sonnet 4

GPT-4.5

o3

Qwen-VL-Max

Gemini 2.5 Pro

Perfect Calibration

1.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.8

0.6

Accuracy

0.4

0.2

0.0

0 20 40 60 80 100

Stated Confidence

Figure 7: Calibration curves based on the self-stated confidence scores and interval-level accuracy.

- 2025b), RAG delivers an absolute improvement of 23.2% (35.6% vs. 58.8%). However, this performance gain comes with substantial computational overhead. Table 3 also quantifies the total inference time, demonstrating that RAG significantly impairs inference efficiency. Our findings highlight the critical trade-off between performance gains and computational practicality. Per-hop factual evaluation: In addition to multi-hop factual QA, our Video SimpleQA benchmark also incorporates decomposed per-fact sub-QAs to facilitate fine-grained evaluations. As shown in Table 4, we present F-scores for both the final multi-hop QA and the sub-QAs across all 4hop questions in Video SimpleQA . Our analysis reveals:

- 1) Multi-hop challenge: The final multi-hop QAs achieve substantially lower F-scores than most single-hop sub-QAs, underscoring the difficulty of multi-hop fact grounding;
- 2) First-hop bottleneck: The F-score for the first hop is markedly lower than those of later hops, likely due to its reliance on accurate object or event recognition, which poses a key challenge. In contrast, LVLMs perform better on sub-

###### Table 3: Comparisons between vanilla models and models with RAG by F-score (%) and the inference time (min).

F-score Inference Time

Model

vanilla w/ RAG vanilla w/ RAG

o3 66.3 69.0 27.8 54.7 GPT-4o 49.3 61.3 30.1 53.9 Gemini 2.5 Pro 62.6 66.2 33.2 60.1 Claude Sonnet 4 35.6 58.8 29.9 56.4 Qwen-VL-Max 39.9 57.0 24.2 61.2

###### Table 4: Per-hop factual evaluations for 4-hop questions in terms of F-score (%). Q1-Q4 denote the decomposed perhop questions. Refer to Figure 6 for an illustrative case.

###### Model QA1 QA2 QA3 QA4 Multi-hop

o3 74.9 89.7 95.3 88.8 78.5 GPT-4o 59.1 85.4 92.5 80.9 47.0 Claude Sonnet 4 43.5 85.7 80.8 61.6 47.7 Gemini 2.5 Pro 65.1 78.7 60.4 28.3 69.8 Qwen-VL-Max 37.9 69.2 83.6 72.3 40.7

sequent hops (Q2–Q4) given clearer contextual grounding.

In Figure 6, we visualize the per-hop evaluation results of the o3 model. This allows us to clearly identify which specific piece of factual knowledge the model lacks. For instance, in the left case, o3 lacks knowledge about the language family of English, while in the right case, it fails to recognize the film Caminandes: Llama Drama.

### Conclusions

We present Video SimpleQA, the first benchmark explicitly designed for evaluating factual grounding in video contexts. Distinct from prior works, our framework introduces the following diagnostic dimensions: knowledge integration, multi-hop fact-seeking questioning, short-form definitive answering, and temporal grounded demands. Through the extensive evaluation of 33 state-of-the-art LVLMs, we reveal notable deficiencies in factual adherence, uncover prevalent model overconfidence, trade-offs associated with RAG, and the critical performance bottleneck.

### Supplementary Material

This supplementary material is organized as follows. We begin with a detailed description of the experimental setups, then present additional experimental findings, followed by the analysis of error types and the visualization results.

Specifically, the experiment setups include the following aspects:

- • Experimental configurations.
- • Evaluation prompts.
- • Video distributions.
- • Biographies of annotators.
- • Copyright clarifications.

###### More experimental findings include the following aspects:

- • Results with different judge models.
- • Performance across secondary categories.
- • Model size scaling remains effective.
- • Frame number scaling remains effective.
- • Performance of smaller LVLMs.
- • Test-time compute yields limited benefits.
- • Temporal scope analysis.
- • More per-hop factual evaluation results.

##### Experiment Setup

Experimental configurations. Table 5 details the configuration of each evaluated model. We used the default settings from the official implementation of each model. All experiments were reproducible on a workstation equipped with 8 NVIDIA A100 GPUs. Across all experiments, the temperature is set to 1.0, with a maximum output length of 1024 tokens. Each experiment was repeated three times to ensure reproducibility and statistical reliability.

Evaluation prompts. The prompts for the grader, along with instructions guiding the model to output answers and confidence scores, are illustrated in Figure 17, Figure 18 and Figure 19, respectively.

Video distributions. In Table 10 and Table 11, we present the video taxonomy distributions of Video SimpleQA including 4 primary categories, 15 secondary categories, and 84 tertiary categories.

Biographies of annotators. The detailed information of annotators who participated in the construction of Video SimpleQA can be found in Table 9. All annotators come from universities ranked in the top 500 of the 2026 QS Global Rankings5, and they are all fluent in English.

Copyright clarifications. The videos of our Video SimpleQA benchmark are sourced from the “Media of the Day” page of Wikipedia, which are freely licensed. Therefore, the construction of Video SimpleQA avoids introducing any potential copyright concerns.

To better align with ethical standards, we have implemented the following measures: 1) Data desensitization: To reduce possible security threats during the evaluation stage, the finalized dataset was independently reviewed by

5https://www.topuniversities.com/world-university-rankings

75

o3 Gemini 2.5 Pro Gemini 2.5 Flash Qwen-VL-Max Claude Sonnet 4

65

55

F-score

45

35

Judge Model Claude Sonnet 4

Judge Model Gemini 2.5 Pro

Judge Model Gemini 2.5 Flash

Judge Model Qwen-VL-Max

Judge Model o3

###### Figure 8: Results with different judge models.

Applied sciences

Arts & Recreation

Physical sciences

Meteorology

Beliefs & Institutions

Mechanical & Electrical

Chemical & Process

Civil & Architecture

Life sciences

Environmental & Geophysical

Language & Material Culture

History & Heritage Geology & Landscapes

Flora & Fauna

Formal sciences

Figure 9: The performance of different models across 15 secondary categories in Video SimpleQA .

six professional security auditors. Each data instance was examined by at least two reviewers, focusing on potential issues such as gender bias, offensive language, or politically sensitive content. Only the samples that passed all checks were included in the final release; 2) Non-commercial research: We release our code and dataset under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License. They are available strictly for non-commercial research; 3) Opt-out mechanism: We release our dataset with an opt-out mechanism, allowing content owners and individuals appearing in the videos to request the removal of their video references.

##### More Experimental Findings

Results with different judge models. The short-form answer paradigm of Video SimpleQA enables automated evaluation through LLM-as-a-judge frameworks with low run-to-run variance. To demonstrate this, we select five typical LVLMs, and evaluate them using various judge models including o3 (OpenAI 2025b), Gemini 2.5 Pro (Comanici

Table 5: Configurations of evaluated LVLMs in Video SimpleQA .

Input Frames Proprietary Multi-modal LLMs

Organization Model Release Version

- o4-mini 2025-4 o4-mini-2025-04-16 32
- o3 2025-4 o3-2025-04-16 32 GPT-4.5 2025-2 gpt-4.5-2025-02-27 32 GPT-4o 2024-5 gpt-4o-2024-08-06 32

OpenAI

Claude Sonnet 4 2025-5 claude-sonnet-4 32 Claude 3.7 Sonnet 2025-2 claude-3.7-sonnet 16 Claude 3.5 Sonnet2 2024-10 claude-3.5-sonnet-v2 16 Claude 3.5 Sonnet 2024-6 claude-3.5-sonnet 16

Anthropic

Gemini 2.5 Pro 2025-3 gemini-2.5-pro-2025-06-17 32 Gemini 2.5 Flash 2025-3 gemini-2.5-flash-2025-06-17 32

Google

Qwen-VL-Max 2024-1 qwen-vl-max 32 Qwen-VL-Plus 2023-11 qwen-vl-plus 32

Alibaba

###### Open-source Multi-modal LLMs

InternVL3-78B 2025-4 InternVL3-78B 16 InternVL3-38B 2025-4 InternVL3-38B 16 InternVL3-14B 2025-4 InternVL3-14B 16 InternVL3-9B 2025-4 InternVL3-9B 16 InternVL3-8B 2025-4 InternVL3-8B 16 InternVL3-2B 2025-4 InternVL3-2B 16 InternVL3-1B 2025-4 InternVL3-1B 16

Shanghai AI Lab

Qwen2.5-VL-72B 2025-2 Qwen2.5-VL-72B-Instruct 32 Qwen2.5-VL-32B 2025-2 Qwen2.5-VL-32B-Instruct 32 Qwen2.5-VL-7B 2025-2 Qwen2.5-VL-7B-Instruct 32 Qwen2.5-VL-3B 2025-2 Qwen2.5-VL-3B-Instruct 32 Qwen2-VL-72B 2024-8 Qwen2-VL-72B-Instruct 16 Qwen2-VL-7B 2024-8 Qwen2-VL-7B-Instruct 16 Qwen2-VL-2B 2024-8 Qwen2-VL-2B-Instruct 16

Alibaba

LLaVA-1.5-13B 2023-9 llava-1.5-13b-hf 4 LLaVA-1.5-7B 2023-9 llava-1.5-7b-hf 4 LLaVA-NeXT-Video-34B 2024-4 LLaVA-NeXT-Video-34B-hf 8 LLaVA-NeXT-Video-7B 2024-4 LLaVA-NeXT-Video-7B-hf 4 LLaVA-OneVision-72B 2024-8 llava-onevision-qwen2-72b-ov-hf 16 LLaVA-OneVision-7B 2024-8 llava-onevision-qwen2-7b-ov-hf 16 LLaVA-OneVision-0.5B 2024-8 llava-onevision-qwen2-0.5b-ov-hf 16

Llava Hugging Face

DeepSeek-VL2 2024-12 deepseek-vl2 2 DeepSeek-VL2-Small 2024-12 deepseek-vl2-small 2 DeepSeek-VL2-Tiny 2024-12 deepseek-vl2-tiny 2

DeepSeek

MoonshotAI Kimi-VL 2025-4 Kimi-VL-A3B-Instruct 32 Kwai-Keye Keye-VL 2025-6 Keye-VL-8B-Preview 32

- et al. 2025), Gemini 2.5 Flash (Comanici et al. 2025), QwenVL-Max (Bai et al. 2023), and Claude Sonnet 4 (Anthropic 2025b). As shown in Figure 8, while the specific scores from different judge models vary, the relative rankings of the evaluated models remain consistent. This demonstrates the robustness of our evaluation under the short-form paradigm. Performance across secondary categories. Figure 9 demonstrates the F-score performance across 15 secondary categories. As shown, we observe distinct performance patterns among the compared LVLMs.

• Capability distribution: o3 (OpenAI 2025b) demonstrates the most consistent performance with superior F-

scores across domains, particularly excelling in the category of Meteorology. Gemini 2.5 Pro (Comanici et al. 2025) follows with complementary strengths and exhibits notable advantages in the category of Life sciences compared to other LVLMs.

• Imbalanced performance profiles: Qwen2.5-VL-72B (Bai et al. 2025) and Qwen-VL-MAX (Bai et al. 2023) show significant performance variance, with Qwen-VLMAX (Bai et al. 2023) severely underperforming in Formal Sciences compared to its moderate capabilities in other domains. Kimi-VL (Team et al. 2025a) and Keye-VL (Team et al. 2025b) consistently underper-

| | | |o3 Gemini|2.5 Pro|
|---|---|---|---|---|
| | | |GPT-|4.5|
| | | |Qwen2.5-VL-72B<br><br>Qwen-VL-Max<br><br>Claude S|onnet 4|
| | | |InternVL3-38B<br><br>Intern<br><br>Qwen2.5-VL-32B Qwen2-VL-72B|VL3-78B|
| | |In Inter<br><br>Qwen2.5-VL-7B Qwen2.5-VL-3B|ternVL3-14B nVL3-8B| |
| | |InternVL3-2B<br><br>Qwen2-VL-2B<br><br>Qwen2-VL-7B Int|ernVL3-9B| |
| |In|ternVL3-1B| | |

Figure 10: The relationship between model size and F-score.

0.38

0.63

0.37

0.62

F-score

0.36

0.61

0.35

0.60

y = 0.0006x + 0.35, R²: 0.13

y = 0.0015x + 0.59, R²: 0.77

0.34

0.59

5 10 15 20 25 30

5 10 15 20 25 30

Number of Frames (Claude Sonnet 4)

Number of Frames (Gemini 2.5 Pro)

0.57

0.56

0.66

F-score

0.55

0.54

0.64

0.53

y = 0.0018x + 0.62, R²: 0.71

y = 0.0020x + 0.51, R²: 0.98

0.52

0.62

5 10 15 20 25 30

5 10 15 20 25 30

Number of Frames (o3)

Number of Frames (Gemini 2.5 Flash)

Figure 11: Frame number scaling experiments.

form across most domains, with particularly low F-scores in the technical areas (e.g., Physical Sciences, Applied Sciences).

• Disciplinary performance gap: All models exhibit systematically lower F-scores in scientific domains (e.g., Formal/Applied Sciences) compared to humanities-oriented categories (History & Heritage, Civil & Architecture), with Kimi-VL (Team et al. 2025a) showing the most pronounced disparities.

Model size scaling remains effective: As evidenced by the F-score across various model sizes in Figure 10, model size scaling continues to demonstrate effectiveness, where larger architectures exhibit consistently superior performance. This observation aligns with the widely recognized scaling law principle.

Frame number scaling remains effective: In Figure 11, we demonstrate the impact of the number of sampled frames on performance. The results for Gemini 2.5 Pro, o3, and Gemini 2.5 Flash reveal a positive correlation between the video frame number and the F-score, thereby validating the effectiveness of frame number scaling. For the results of Claude Sonnet 4, the optimum result is achieved when setting the frame number to 10.

o3

Gemini 2.5 Pro

Qwen-VL-Max

Claude Sonnet 4

GPT-4o

Gemini 2.5 Flash

0.7

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
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.6

0.5

Accuracy

0.4

0.3

0.2

1 2 4 8 16

0 1 2 3

N (Best of N)

#lteration (Self-refine)

Figure 12: Evaluations of test-time compute including Best-of-N and Self-refine.

Performance of smaller LVLMs: In addition to the evaluation results of the 33 LVLMs presented in the main paper, we have also included results for smaller models (with parameter counts of 0.5B, 1B, 2B and 3B) in Table 6. This serves as a valuable reference for users with limited resources who wish to utilize the proposed Video SimpleQA dataset.

Test-Time compute yields limited benefits: We empirically investigate the effectiveness of test-time compute (Snell et al. 2024) strategies on Video SimpleQA by conducting experiments with 200 randomly sampled instances. Two approaches are evaluated: 1) Best-of-N: The model independently generates N responses and selects the one it considers the best as the final answer; 2) Self-refine: The model is prompted to iteratively refine the initial outputs using self-generated feedback (Madaan et al. 2023).

Figure 12 presents the accuracy (i.e., the proportion of correct answers) under varying inference-time N (for Bestof-N) and different refinement iterations (for Self-refine). Experimental results reveal that both strategies fail to produce significant or consistent accuracy improvements. In some cases, these strategies even degrade performance. For instance, when increasing N from 2 to 4 in Best-of-N trials, o3 (OpenAI 2025b) and Gemini-2.5-Pro (Reid et al. 2024) exhibit reduced accuracy, suggesting that these models struggle to reliably select the best answer from multiple inferences. These findings highlight the challenges in improving factuality through post-hoc test-time compute strategies.

Temporal scope analysis: Video SimpleQA is designed to be temporally grounded, i.e., answering questions should refer to one or more temporal segments in the video, rather than relying on a single frame. To demonstrate this, we randomly sampled 200 QA pairs and instructed expert annotators to categorize them based on the necessary temporal scope needed for accurate answers. We categorize the results as follows: 1) short-term scope required (<10 seconds); 2) medium-term scope required (10s-1min); or 3) long-term scope required (>1min). It should be emphasized that our definitions of short/medium/long-term specifically denote

- Table 6: Evaluation results (%) of small LVLMs on Video SimpleQA . For metrics, CO, NA, IN, and CGA denote “Correct”, “Not attempted”, “Incorrect”, and “Correct given attempted”, respectively. For subtopics, ENG, NAT, SCI and SAC represent “Engineering”, “Nature”, “Science” and “Society and Culture”.

Model

Overall results on 5 metrics F-score on 4 primary categories CO IN↓ NA↓ CGA F-score ENG NAT SCI SAC

InternVL3-2B (Zhu et al. 2025) 15.5 80.5 4.0 16.1 15.8 10.9 23.4 16.4 14.8 InternVL3-1B (Zhu et al. 2025) 11.3 82.5 6.1 12.1 11.7 6.1 19.2 12.1 11.2 Qwen2.5-VL-3B (Bai et al. 2025) 22.3 74.5 3.2 23.0 22.6 12.1 30.3 29.2 18.6 Qwen2-VL-2B (Wang et al. 2024a) 16.3 73.3 10.4 18.2 17.2 12.9 24.9 16.4 17.4 LLaVA-OneVision-0.5B (Li et al. 2024a) 7.8 85.7 6.5 8.3 8.0 6.1 11.6 5.8 10.0

- Table 7: The proportion and performance (%) of QA pairs by different temporal scopes.

###### o3

###### Gemini 2.5 Pro

Qwen2.5-VL-72B

Type Prop CO IN↓ NA↓ CGA F-score

5% 4%

6%

9%

5%

Short-term 50.5 56.6 33.1 10.3 63.1 59.7 Medium-term 32.1 48.5 38.8 12.7 55.6 51.8 Long-term 17.4 43.2 41.5 15.3 51.0 46.8

9%

23%

19%

14%

70%

68%

68%

the temporal scope required to correctly answer QA pairs, distinct from the absolute video duration referenced in existing long-form video understanding benchmarks (Wang et al. 2024b; Fu et al. 2024).

Perception Error Lack of Knowledge RefusalRefusetotoAnswerAnswer Failure to Follow Instructions

Figure 13: Error type distributions across o3 (OpenAI 2025b), Gemini 2.5 Pro (Comanici et al. 2025) and Qwen2.5-VL-72B (Bai et al. 2025).

- Table 7 summarizes the proportion and performance un-

der different temporal scopes, which reveals that 82.6% of cases require either short-term or medium-term temporal understanding to answer correctly. This distribution demonstrates that our Video SimpleQA benchmark indeed necessitates temporal understanding capabilities rather than simple frame-level analysis. Furthermore, as indicated in Table 7, videos with long-term temporal scope exhibit significantly lower performance metrics compared to the other two categories, which highlights the importance of long-context temporal modeling.

single-hop, fact-based question.

##### Error Analysis

This section presents a case study analyzing error patterns in o3 (OpenAI 2025b), Gemini 2.5 Pro (Comanici et al. 2025) and Qwen2.5-VL-72B (Bai et al. 2025) through a systematic examination of all available samples per model spanning diverse question types. We categorize observed errors into four primary classes, each illustrated with representative examples:

More per-hop factual evaluation results: In the main paper, we have presented the performance of per-fact sub-QA for all the 4-hop questions Video SimpleQA . Here, we present the per-fact sub-QA performance across all 3-hop questions. As shown in Table 8, we still observe that final multi-hop QAs achieve substantially lower F-scores than most single-hop sub-QAs, and the performance of the first hop is markedly lower than that of subsequent hops.

- • Perception error: Incorrect identification of objects. This occurs when LVLMs misidentify or fail to detect key visual elements in input videos (c.f. Figure 15 (a)).
- • Lack of knowledge: Correct identification but lacking relevant knowledge. LVLMs accurately perceive the visual content but cannot provide accurate information due to knowledge limitations (c.f. Figure 15 (b)).
- • Refusal to answer: LVLMs recognize their inability to make a confident determination and opt to abstain from answering (c.f. Figure 16 (c)).
- • Failure to follow instructions: LVLMs understand the input but fail to properly execute the given instructions. This typically manifests in two ways: 1) the generated outputs do not conform to the specified format requirements, or 2) the responses are irrelevant to the question posed, e.g., addressing different topics than what was actually requested (c.f. Figure 16 (d)).

We observe that the accuracy of multi-hop questions is higher than that of sub-QA1. Through analysis, we found this occurs because in certain cases, even if sub-QA1 is answered incorrectly, both the ground-truth answer and the incorrect answer lead to a consistent final answer for the multihop question. For example in Figure 14, for the model o3, the kata type is incorrectly identified as Heian Nidan. Yet, whether based on this incorrect answer or the correct answer (Heian Godan), the final answer for the multihop question is consistently Japan. This further highlights the value of our proposed fine-grained per-hop evaluation, i.e., it requires the model not only to correctly answer the final multi-hop question, but also to accurately answer each

- Table 8: Per-hop factual evaluations for 3-hop questions in terms of F-score (%). Q1-Q3 denote the decomposed perhop questions.

###### Model QA1 QA2 QA3 Multi-hop

o3 64.9 94.5 90.5 74.9 GPT-4o 55.0 88.0 87.4 56.7 Claude Sonnet 4 39.7 83.3 76.1 54.6 Gemini 2.5 Pro 63.4 81.7 59.5 71.7 Qwen-VL-Max 42.6 77.5 77.5 49.5

##### Visualizations

Evaluation results: The visualization results of three typical LVLMs (o3 (OpenAI 2025b), Gemini 2.5 Pro (Comanici et al. 2025) and Qwen2.5-VL-72B (Bai et al. 2025)) are illustrated in Figure 22 – Figure 26.

Limitations of existing discipline-based methods: As analyzed in the main paper, current discipline-based methods suffer from the following issues: 1) Inclusion of hypothetical or subjective speculation: For instance, in Figure 20, the questions often contain hypothetical or subjective expressions such as “might”, “infer”, “seem to”, or “probably”. In contrast, our Video SimpleQA requires answers to be crafted in an unambiguous and definitively correct manner; 2) Entangling fact grounding and reasoning: To answer the questions in the examples shown in Figure 21, the model needs to possess reasoning abilities such as matrix computation, minimum spanning tree planning, and time complexity analysis. This makes it difficult to independently evaluate the model’s fact-grounding capability. In contrast, our Video SimpleQA exclusively focuses on fact identification, providing a clearer assessment of LVLMs’ factgrounding ability.

### References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Akhtar, M.; Schlichtkrull, M.; Guo, Z.; Cocarascu, O.; Simperl, E.; and Vlachos, A. 2023. Multimodal Automated Fact-Checking: A Survey. In Findings of the Association for Computational Linguistics: EMNLP 2023, 5430–5448.

- Anthropic. 2024. Claude 3.5 Sonnet.
- Anthropic. 2025a. Claude 3.7 Sonnet. Anthropic. 2025b. Claude 4.

Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin, J.; Zhou, C.; and Zhou, J. 2023. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; et al. 2025. Qwen2. 5-VL Technical Report. arXiv preprint arXiv:2502.13923.

Cai, M.; Tan, R.; Zhang, J.; Zou, B.; Zhang, K.; Yao,

- F.; Zhu, F.; Gu, J.; Zhong, Y.; Shang, Y.; et al. 2024.

Temporalbench: Benchmarking fine-grained temporal understanding for multimodal video models. arXiv preprint arXiv:2410.10818.

Chandrasegaran, K.; Gupta, A.; Hadzic, L. M.; Kota, T.; He, J.; Eyzaguirre, C.; Durante, Z.; Li, M.; Wu, J.; and Fei-Fei, L. 2024. HourVideo: 1-Hour Video-Language Understanding. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Chen, X.; Lin, Y.; Zhang, Y.; and Huang, W. 2024. Autoevalvideo: An automatic benchmark for assessing large vision language models in open-ended video question answering. In European Conference on Computer Vision, 179–195. Springer.

Cheng, X.; Zhang, W.; Zhang, S.; Yang, J.; Guan, X.; Wu, X.; Li, X.; Zhang, G.; Liu, J.; Mai, Y.; et al. 2025. SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models. arXiv preprint arXiv:2502.13059. Chern, I.; Chern, S.; Chen, S.; Yuan, W.; Feng, K.; Zhou, C.; He, J.; Neubig, G.; Liu, P.; et al. 2023. FacTool: Factuality Detection in Generative AI–A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios. arXiv preprint arXiv:2307.13528.

Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang, D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Fang, X.; Mao, K.; Duan, H.; Zhao, X.; Li, Y.; Lin, D.; and Chen, K. 2024. MMBench-Video: A Long-Form MultiShot Benchmark for Holistic Video Understanding. arXiv preprint arXiv:2406.14515.

Fu, C.; Dai, Y.; Luo, Y.; Li, L.; Ren, S.; Zhang, R.; Wang, Z.; Zhou, C.; Shen, Y.; Zhang, M.; et al. 2024. VideoMME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis. arXiv preprint arXiv:2405.21075.

Garcia, N.; Otani, M.; Chu, C.; and Nakashima, Y. 2020. KnowIT VQA: Answering knowledge-based questions about videos. In Proceedings of the AAAI conference on artificial intelligence, volume 34, 10826–10834.

Gou, Z.; Shao, Z.; Gong, Y.; Shen, Y.; Yang, Y.; Duan, N.; and Chen, W. 2023. Critic: Large language models can self-correct with tool-interactive critiquing. arXiv preprint arXiv:2305.11738.

Gu, J.; Jiang, X.; Shi, Z.; Tan, H.; Zhai, X.; Xu, C.; Li, W.; Shen, Y.; Ma, S.; Liu, H.; et al. 2024. A Survey on LLM-asa-Judge. arXiv preprint arXiv:2411.15594.

Gu, J.; Wang, Y.; Bu, P.; Wang, C.; Wang, Z.; Song, T.; Wei, D.; Yuan, J.; Zhao, Y.; He, Y.; et al. 2025. “See the World, Discover Knowledge”: A Chinese Factuality Evaluation for Large Vision Language Models. arXiv preprint arXiv:2502.11718.

Guan, T.; Liu, F.; Wu, X.; Xian, R.; Li, Z.; Liu, X.; Wang, X.; Chen, L.; Huang, F.; Yacoob, Y.; et al. 2024. HallusionBench: an advanced diagnostic suite for entangled language

hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14375–14385.

Guo, C.; Pleiss, G.; Sun, Y.; and Weinberger, K. Q. 2017. On calibration of modern neural networks. In International conference on machine learning, 1321–1330. PMLR.

- He, X.; Feng, W.; Zheng, K.; Lu, Y.; Zhu, W.; Li, J.; Fan, Y.; Wang, J.; Li, L.; Yang, Z.; et al. 2024a. MMWorld: Towards Multi-discipline Multi-faceted World Model Evaluation in Videos. arXiv preprint arXiv:2406.08407.
- He, Y.; Li, S.; Liu, J.; Tan, Y.; Wang, W.; Huang, H.; Bu, X.; Guo, H.; Hu, C.; Zheng, B.; et al. 2024b. Chinese simpleqa: A chinese factuality evaluation for large language models. arXiv preprint arXiv:2411.07140.

Hu, K.; Wu, P.; Pu, F.; Xiao, W.; Zhang, Y.; Yue, X.; Li, B.; and Liu, Z. 2025. Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos. arXiv preprint arXiv:2501.13826.

Jain, A.; Kothyari, M.; Kumar, V.; Jyothi, P.; Ramakrishnan,

- G.; and Chakrabarti, S. 2021. Select, substitute, search: A new benchmark for knowledge-augmented visual question answering. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2491–2498.

Li, B.; Zhang, Y.; Guo, D.; Zhang, R.; Li, F.; Zhang, H.; Zhang, K.; Li, Y.; Liu, Z.; and Li, C. 2024a. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326. Li, K.; Wang, Y.; He, Y.; Li, Y.; Wang, Y.; Liu, Y.; Wang, Z.; Xu, J.; Chen, G.; Luo, P.; et al. 2024b. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22195–22206.

Lin, S.; Hilton, J.; and Evans, O. 2022. TruthfulQA: Measuring How Models Mimic Human Falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 3214–3252.

Liu, H.; Li, C.; Li, Y.; Li, B.; Zhang, Y.; Shen, S.; and Lee, Y. J. 2024. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual instruction tuning. Advances in neural information processing systems, 36: 34892–34916.

Liu, J. 2022. LlamaIndex.

Madaan, A.; Tandon, N.; Gupta, P.; Hallinan, S.; Gao, L.; Wiegreffe, S.; Alon, U.; Dziri, N.; Prabhumoye, S.; Yang, Y.; et al. 2023. Self-refine: Iterative refinement with selffeedback. Advances in Neural Information Processing Systems, 36: 46534–46594.

Marino, K.; Rastegari, M.; Farhadi, A.; and Mottaghi, R. 2019. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, 3195–3204.

- OpenAI. 2024. Hello GPT-4o. OpenAI Blog.
- OpenAI. 2025a. Introducing GPT-4.5. OpenAI Blog.

OpenAI. 2025b. Introducing OpenAI o3 and o4-mini. OpenAI Blog.

Pan, S.; Luo, L.; Wang, Y.; Chen, C.; Wang, J.; and Wu, X. 2024. Unifying large language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and Data Engineering.

Reid, M.; Savinov, N.; Teplyashin, D.; Lepikhin, D.; Lillicrap, T.; Alayrac, J.-b.; Soricut, R.; Lazaridou, A.; Firat, O.; Schrittwieser, J.; et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Snell, C.; Lee, J.; Xu, K.; and Kumar, A. 2024. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314.

Song, E.; Chai, W.; Wang, G.; Zhang, Y.; Zhou, H.; Wu, F.; Chi, H.; Guo, X.; Ye, T.; Zhang, Y.; et al. 2024. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18221–18232.

Team, K.; Du, A.; Yin, B.; Xing, B.; Qu, B.; Wang, B.; Chen, C.; Zhang, C.; Du, C.; Wei, C.; et al. 2025a. Kimi-vl technical report. arXiv preprint arXiv:2504.07491.

Team, K. K.; Yang, B.; Wen, B.; Liu, C.; Chu, C.; Song, C.; Rao, C.; Yi, C.; Li, D.; Zang, D.; et al. 2025b. Kwai KeyeVL Technical Report. arXiv preprint arXiv:2507.01949.

Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.; Babaei, Y.; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale, S.; et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Wang, C.; Liu, X.; Yue, Y.; Tang, X.; Zhang, T.; Jiayang, C.; Yao, Y.; Gao, W.; Hu, X.; Qi, Z.; et al. 2023. Survey on factuality in large language models: Knowledge, retrieval and domain-specificity. arXiv preprint arXiv:2310.07521.

Wang, P.; Bai, S.; Tan, S.; Wang, S.; Fan, Z.; Bai, J.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; et al. 2024a. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Wang, P.; Wu, Q.; Shen, C.; Dick, A.; and Van Den Hengel, A. 2017. Fvqa: Fact-based visual question answering. IEEE transactions on pattern analysis and machine intelligence, 40(10): 2413–2427.

Wang, P.; Wu, Q.; Shen, C.; Hengel, A. v. d.; and Dick, A. 2015. Explicit knowledge-based reasoning for visual question answering. arXiv preprint arXiv:1511.02570.

Wang, W.; He, Z.; Hong, W.; Cheng, Y.; Zhang, X.; Qi, J.; Huang, S.; Xu, B.; Dong, Y.; Ding, M.; et al. 2024b. LVBench: An Extreme Long Video Understanding Benchmark. arXiv preprint arXiv:2406.08035.

Wang, Y.; Wang, M.; Manzoor, M. A.; Liu, F.; Georgiev, G.; Das, R.; and Nakov, P. 2024c. Factuality of large language models: A survey. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 19519–19529.

Wang, Y.; Wang, Y.; Zhao, D.; Xie, C.; and Zheng, Z. 2024d. Videohallucer: Evaluating intrinsic and extrinsic hallucinations in large video-language models. arXiv preprint arXiv:2406.16338.

Wei, J.; Karina, N.; Chung, H. W.; Jiao, Y. J.; Papay, S.; Glaese, A.; Schulman, J.; and Fedus, W. 2024. Measuring short-form factuality in large language models. arXiv

- preprint arXiv:2411.04368.

Wu, Z.; Chen, X.; Pan, Z.; Liu, X.; Liu, W.; Dai, D.; Gao, H.; Ma, Y.; Wu, C.; Wang, B.; et al. 2024. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302.

Yang, J.; Yang, S.; Gupta, A. W.; Han, R.; Fei-Fei, L.; and Xie, S. 2024. Thinking in space: How multimodal large language models see, remember, and recall spaces. arXiv

- preprint arXiv:2412.14171.

Yu, W.; Iter, D.; Wang, S.; Xu, Y.; Ju, M.; Sanyal, S.; Zhu, C.; Zeng, M.; and Jiang, M. 2022. Generate rather than retrieve: Large language models are strong context generators. arXiv preprint arXiv:2209.10063.

Zellers, R.; Bisk, Y.; Farhadi, A.; and Choi, Y. 2019. From recognition to cognition: Visual commonsense reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 6720–6731.

Zhang, J.; Jiao, Y.; Chen, S.; Chen, J.; and Jiang, Y.-G. 2024a. Eventhallusion: Diagnosing event hallucinations in video llms. arXiv preprint arXiv:2409.16597.

Zhang, Y.; Zhang, K.; Li, B.; Pu, F.; Setiadharma, C. A.; Yang, J.; and Liu, Z. 2024b. WorldQA: Multimodal World Knowledge in Videos through Long-Chain Reasoning. arXiv preprint arXiv:2405.03272.

Zhao, Y.; Xie, L.; Zhang, H.; Gan, G.; Long, Y.; Hu, Z.; Hu, T.; Chen, W.; Li, C.; Song, J.; et al. 2025. MMVU: Measuring Expert-Level Multi-Discipline Video Understanding. arXiv preprint arXiv:2501.12380.

Zhu, J.; Wang, W.; Chen, Z.; Liu, Z.; Ye, S.; Gu, L.; Tian,

- H.; Duan, Y.; Su, W.; Shao, J.; et al. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479.

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

|sub-question-1: Which kata is the person performing in the video?<br><br>sub-answer-1: Heian Godan<br><br>o3-answer-1: Heian Nidan<br><br>sub-question-2: Which Okinawan karate master developed the series of kata that includes Heian Godan? sub-answer-2: Anko Itosu<br><br>o3-answer-2: Anko Itosu<br><br>sub-question-3: Which country is Anko Itosu from? sub-answer-3: Japan<br><br>o3-answer-3: Japan<br><br><br><br><br><br><br>multi-hop-question: Which country did the Okinawan karate master who developed the series of kata being performed in the video come from? multi-hop-answer: Japan o3-answer: Japan<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]|
|---|

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

|sub-question-1: What insects are in the video?<br><br>sub-answer-1: Box Bug<br><br>o3-answer-1: Leaf-footed bugs (family Coreidae)<br><br>sub-question-2: What order does the Box bug belong to? sub-answer-2: Hemiptera<br><br>o3-answer-2: Hemiptera<br><br>sub-question-3: What phylum does Hemiptera belong to? sub-answer-3: Arthropoda<br><br>o3-answer-3: Arthropoda<br><br><br><br><br><br><br>multi-hop-question: What phylum does that The order of insects in the video belong to? multi-hop-answer: Arthropoda o3-answer: Arthropoda<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]|
|---|

Figure 14: Case studies where the incorrect sub-QA1 still yields the correct final multi-hop answer.

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

|sub-question-1: What is the scientific name of the animal shown in the video?<br><br>sub-answer-1: Bufo bufo o3 answer-1: Rana temporaria<br><br>sub-question-2: To which taxonomic family does the European common toad, Bufo bufo, belong?<br><br>sub-answer-2: Bufonidae o3 answer-2: Bufonidae<br><br><br><br><br>multi-hop-question: To which taxonomic family does the animal shown in the video belong? multi-hop-answer: Bufonidae o3 answer: Ranidae<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]|
|---|
|error type: Perception Error error reason: The model misidentified the animal in the video as a frog instead of a toad, leading to incorrect taxonomic classification.|

[Figure 147]

[Figure 148]

[Figure 149]

#### (a)

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

|sub-question-1: Which movie is this scene from in the video?<br><br>sub-answer-1: What Happened on Twenty-Third Street, New York City o3 answer-1: What Happened on Twenty-Third Street, New York City<br><br>sub-question-2: What is the release month of the movie "What Happened on Twenty-Third Street, New York City"?<br><br>sub-answer-2: August 1901 o3 answer-2: July<br><br><br><br><br>multi-hop-question: What is the release month of the movie clip shown in the video? multi-hop-answer: August 1901 o3 answer: July<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]|
|---|
|error type: Lack of Knowledge error reason: The model had incorrect factual information about the movie's release date, believing it was released in July instead of August 1901.|

#### (b)

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

|sub-question-1: What is the grain in the video?<br><br>sub-answer-1: Rice Gemini 2.5 Pro answer-1: The grain being threshed in the video is most likely rice.<br><br>sub-question-2: What is the scientific name of rice?<br><br>sub-answer-2: Oryza sativa Gemini 2.5 Pro answer-2: Oryza sativa<br><br><br><br><br>multi-hop-question: What is the scientific name of the grain in the video? multi-hop-answer: Oryza sativa Gemini 2.5 Pro answer: The specific type of grain cannot be identified.<br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]|
|---|
|error type: Refusal to Answer Incorrect error reason: The model was overly cautious and refused to identify the grain type despite having correctly identified it as rice in the sub-questions.|

[Figure 175]

[Figure 176]

[Figure 177]

- (c)

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

|sub-question-1: What is the name of the bridge shown in the video?<br><br>sub-answer-1: The Rolling Bridge Qwen2.5-VL-72B answer-1: Rolling Bridge<br><br>sub-question-2: What is the total length of The Rolling Bridge?<br><br>sub-answer-2: 12 metres Qwen2.5-VL-72B answer-2: 12 metres<br><br><br><br><br>multi-hop-question: What is the total length of the bridge shown in the video? multi-hop-answer: 12 metres Qwen2.5-VL-72B answer: The bridge in the video is Rolling Bridge.<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]|
|---|
|error type: Failure to Follow Instructions error reason: The model answered with the bridge name instead of the requested length measurement, failing to properly integrate information from sub-questions.|

(d)

|Your job is to look at some video frames generated from the video, a question generated from the video, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]. First, I will give examples of each grade, and then you will grade a new example.<br><br>The following are examples of CORRECT predicted answers. ``` Question: What is the name of the man's child in the video? Gold target: Malia Obama and Sasha Obama<br><br>Predicted answer 1: sasha and malia obama<br>Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check<br>Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001. ``` These predicted answers are all CORRECT because:<br><br>- They fully contain the important information in the gold target.<br>- They do not contain any information that contradicts the gold target.<br>- Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.<br>- Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.<br><br><br>The following are examples of INCORRECT predicted answers. ``` Question: What is the name of the man's child in the video? Gold target: Malia and Sasha<br><br>Predicted answer 1: Malia.<br>Predicted answer 2: Malia, Sasha, and Susan.<br>Predicted answer 3: Barack Obama does not have any children.<br>Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.<br><br><br>Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.<br>Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?<br>Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information. ``` These predicted answers are all INCORRECT because:<br><br><br>- A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.<br><br>The following are examples of NOT_ATTEMPTED predicted answers. ``` Question: What is the name of the man's child in the video? Gold target: Malia and Sasha<br><br>Predicted answer 1: I don't know.<br>Predicted answer 2: I need more context about which Obama you are talking about.<br>Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.<br>Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one. ``` These predicted answers are all NOT_ATTEMPTED because:<br><br><br>- The important information in the gold target is not included in the answer.<br>- No statements in the answer contradict the gold target.<br>|
|---|

###### Figure 17: Prompt for grading: Part 1.

|Also note the following things:<br><br>- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".<br>- Predicted answers "120k", "124k", and 115k" are all CORRECT.<br>- Predicted answers "100k" and "113k" are INCORRECT.<br>- Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.<br>- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.<br>- For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.<br>- Do not punish predicted answers if they omit information that would be clearly inferred from the question.<br>- For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".<br>- Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.<br>- For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.<br>- For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.<br>- Do not punish for typos in people's name if it's clearly the same name.<br>- For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung". Grade the predicted answer of the question as one of:<br><br><br>A: CORRECT<br>B: INCORRECT<br>C: NOT_ATTEMPTED Just return the letter "A", "B", or "C", with no text around it.<br>|
|---|

- Figure 18: Prompt for grading: Part 2.

|Do not generate any intermediate reasoning process. Based on the video frames, directly output a short, accurate answer to the user's question and include a confidence score (0-100) in the following JSON format: {"answer": "Your answer here", "confidence_score": number} Do not include any additional text or explanations outside this JSON format.|
|---|

Figure 19: Prompt for calibration experiments.

Table 9: Biographies of annotators involved in Video SimpleQA construction.

###### ID Background Language Proficiency Avg. Time (per QA) Agreement Rate (%)

- 1 CS Undergraduate English (Fluent) 22s 92.3%
- 2 NLP Master English (Native) 25s 94.1%
- 3 NLP PhD English (Fluent) 23s 93.2%
- 4 CV PostDoc English (Fluent) 21s 91.7%
- 5 Robotics Undergraduate English (Native) 26s 89.4%
- 6 ML PhD English (Fluent) 24s 92.8%
- 7 CS Undergraduate English (Fluent) 28s 88.6%
- 8 CV PhD English (Native) 22s 93.7%
- 9 Robotics Undergraduate English (Fluent) 27s 90.1%
- 10 NLP PhD English (Native) 20s 94.5%
- 11 ML Master English (Fluent) 25s 91.3%
- 12 CS Master English (Native) 29s 87.9%
- 13 CV PhD English (Fluent) 21s 93.8%
- 14 Robotics Master English (Fluent) 26s 90.7%
- 15 ML PhD English (Native) 23s 92.4%
- 16 NLP Master English (Fluent) 28s 89.2%
- 17 CS PhD English (Fluent) 24s 91.8%
- 18 Robotics Undergraduate English (Native) 30s 86.5%
- 19 CV Master English (Fluent) 25s 92.1%
- 20 ML PhD English (Fluent) 22s 93.6%
- 21 CS Undergraduate English (Native) 27s 88.7%
- 22 NLP Master English (Fluent) 26s 90.4%
- 23 Robotics PhD English (Fluent) 23s 92.9%
- 24 CV PhD English (Native) 24s 91.2%
- 25 ML PhD English (Fluent) 21s 94.1%
- 26 CS Master English (Fluent) 28s 89.8%
- 27 NLP Master English (Native) 25s 92.3%
- 28 Robotics Undergraduate English (Fluent) 29s 87.6%
- 29 CV Undergraduate English (Fluent) 27s 90.5%
- 30 ML PhD English (Native) 22s 93.4%
- 31 CS Master English (Fluent) 26s 91.7%
- 32 NLP PhD English (Native) 23s 93.1%
- 33 Robotics PhD English (Fluent) 24s 92.6%
- 34 CV Undergraduate English (Fluent) 28s 88.9%
- 35 ML Master English (Native) 25s 91.4%
- 36 CS PhD English (Fluent) 23s 92.8%
- 37 NLP Undergraduate English (Fluent) 29s 87.3%
- 38 Robotics Master English (Native) 26s 90.9%
- 39 CV Master English (Fluent) 27s 89.7%
- 40 ML PostDoc English (Native) 21s 94.2%

Table 10: The video distribution of Video SimpleQA benchmark (part 1).

Primary Category Secondary Category Tertiary Category Count

Fossils 43 Landscapes 19

Nature Geology & Landscapes Rocks & Minerals 7

Geomorphology 5 Volcanic Features 7

Coastal Landforms 6 Animalia 18

Marine Organisms 91 Nature Flora & Fauna Plantae 61

Fungi 37 Microorganisms 82

Endangered Species 90

Weather 3 Nature Meteorology Climate 6

Atmospheric Phenomena 6 Forecasting 3 Architecture 56

Civil 65 Engineering Civil & Architecture Structural Engineering 56 Urban Planning 46 Mechanical 52 Electrical 47 Engineering Mechanical & Electrical Mechatronics 50 Aerospace Engineering 43

Chemical 4 Process 3 Engineering Chemical & Process Biochemical Engineering 3 Polymer Engineering 3

Environmental 22 Geophysical 21 Engineering Environmental & Geophysical Hydrology 26 Climate Engineering 20

Physics 2 Chemistry 4 Astronomy 5

Science Physical sciences Earth sciences 2 Materials Science 9 Atmospheric Science 3 Geophysics 6

Biology 10 Medicine 11 Science Life sciences Ecology 11 Genetics 10

Neuroscience 7 Mathematics 3

Science Formal sciences Computer Science 2 Statistics 3

Logic 1 Technology 6

Science Applied sciences Robotics 3 Agricultural Science 4 Data Science 3

Table 11: The video distribution of Video SimpleQA benchmark (part 2).

Primary Category Secondary Category Tertiary Category Count

Art 16 Literature 6

Music 30 Entertainment 20

Society & Culture Arts & Recreation Sports 12

Dance 12 Theatre 42

Film 25

Photography 10 Games 31 Belief 5 Religion 20 Philosophy 7

Ethics 9 Society & Culture Beliefs & Institutions Politics 11

Flags 8 Government 3

Law 3 People 5 History 32 Events 9 Places 9

Society & Culture History & Heritage Archaeology 24

Heritage Sites 9 Genealogy 9 Language 2

Objects 9 Food 7 Society & Culture Language & Material Culture Clothing 2 Transportation 5 Instruments 3 Tools 3

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

|Question: What might be the cause of the scene in the video? Answer: Long-term imbalance …. [Subjective speculation]<br><br>[Figure 202]|
|---|

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

|Question: What can you infer about the bird in the video? Answer: He is potentially a healthy bird with better genes. [Subjective speculation]<br><br>[Figure 216]|
|---|

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

|Question: This video game seems to belong to which genre? Answer: RPG. [Subjective speculation]<br><br>[Figure 230]|
|---|

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

|Question: What might probablly happen then? Answer: The robot arm might try to open the bottom drawer. [Subjective speculation]<br><br>[Figure 244]|
|---|

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

|Question: According to the video, which of the following statements can be inferred? Answer: Al has a lower specific heat than Brass [Subjective speculation]<br><br>[Figure 258]|
|---|

- Figure 20: Examples of the inclusion of hypothetical or subjective speculation in existing discipline-based (Zhao et al. 2025; He et al. 2024a) benchmarks.

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

|Question: Why is the (2,3) element in the last matrix -1? Answer: ….The value should be -2. [Entangled fact grounding and reasoning]<br><br>[Figure 272]|
|---|

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

|Question: What is the total weight of the minimum spanning tree formed using the demonstrated algorithm? Answer: 26 [Entangled fact grounding and reasoning]<br><br>[Figure 286]|
|---|

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

|Question: Given that n represents the number of elements and k denotes the range of input values, what is the worst-case time complexity of the demonstrated algorithm? Answer: O(n + k) [Entangled fact grounding and reasoning]<br><br>[Figure 300]|
|---|

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

|Question: In the experiment described in the video, …. Between the two readings, how much did the number of photons reaching the photodiode change? Answer: 3.86% [Entangled fact grounding and reasoning]<br><br>[Figure 314]|
|---|

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

|Question: What is his new angular velocity ( omega_f ) in terms of ( I_0 ), ( m ), ( L ), and ( omega_0 )? Answer: dfrac{I_0}{I_0 + 2mL^2} omega_0 [Entangled fact grounding and reasoning]<br><br>[Figure 328]|
|---|

- Figure 21: Examples of the entangled fact grounding and reasoning cases in existing discipline-based (Zhao et al. 2025; He et al. 2024a) benchmarks.

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

|[Figure 339]<br><br>sub-question-1: What species of butterfly is shown in the video?<br><br>sub-answer-1: The Common Blue<br><br>o3-answer-1: Common blue butterfly (Polyommatus icarus)<br><br>Gemini 2.5 Pro-answer-1: Common Blue butterfly (Polyommatus icarus).<br><br>Qwen2.5-VL-72B-answer-1: Common Blue<br><br>reference source-1: https://commons.wikimedia.org/wiki/Template:Motd/2010-11-09<br><br>sub-question-2: Which country honors the Common Blue as its national butterfly? sub-answer-2: Israel o3-answer-2: Ireland Gemini 2.5 Pro-answer-2: Finland Qwen2.5-VL-72B-answer-2: Sweden<br><br>reference source-2: https://en.wikipedia.org/wiki/Common_blue#Physiology<br><br>sub-question-3: Where is the capital of Israel? sub-answer-3: Jerusalem o3-answer-3: Jerusalem Gemini 2.5 Pro-answer-3: Jerusalem Qwen2.5-VL-72B-answer-3: Jerusalem<br><br>reference source-3: https://en.wikipedia.org/wiki/Israel<br><br>sub-question-4: In which year was Jerusalem declared the capital of Arabic culture? sub-answer-4: 2009 o3-answer-4: 2009 Gemini 2.5 Pro-answer-4: 2009 Qwen2.5-VL-72B-answer-4: 2008<br><br>reference source-4: https://en.wikipedia.org/wiki/Jerusalem<br><br><br><br><br><br><br><br><br><br><br><br><br>multi-hop-question: In which year was the capital of the country that honors the butterfly shown in the video as its national butterfly declared the capital of Arabic culture? multi-hop-answer: 2009 o3-answer: 2002 Gemini 2.5 Pro-answer: 2002 Qwen2.5-VL-72B-answer: 2016<br><br>[Figure 340]<br><br>[Figure 341]<br><br>[Figure 342]<br><br>[Figure 343]<br><br>[Figure 344]<br><br>[Figure 345]<br><br>[Figure 346]<br><br>[Figure 347]<br><br>[Figure 348]<br><br>[Figure 349]<br><br>[Figure 350]<br><br>[Figure 351]<br><br>[Figure 352]<br><br>[Figure 353]<br><br>[Figure 354]|
|---|

- Figure 22: Sampled examples in Video SimpleQA and the responses of typical LVLMs: part 1.

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

|sub-question-1: What is the name of the steam tram in the video?<br><br>sub-answer-1: Feuriger Elias<br><br>o3-answer-1: Feuriger Elias<br><br>Gemini 2.5 Pro-answer-1: Feuriger Elias<br><br>Qwen2.5-VL-72B-answer-1: Feuriger Elias<br><br>reference source-1: https://commons.wikimedia.org/wiki/Template:Motd/2010-12-02<br><br>sub-question-2: Which company does Feuriger Elias belong to? sub-answer-2: Henschel & Sohn o3-answer-2: Henschel & Sohn Gemini 2.5 Pro-answer-2: Henschel & Sohn Qwen2.5-VL-72B-answer-2: Henschel & Sohn<br><br>reference source-2: https://en.wikipedia.org/wiki/Tram_engine<br><br>sub-question-3: Which country is Henschel & Sohn from? sub-answer-3: Germany o3-answer-3: Germany Gemini 2.5 Pro-answer-3: Germany Qwen2.5-VL-72B-answer-3: Germany<br><br>reference source-3: https://en.wikipedia.org/wiki/Henschel_%26_Son<br><br>sub-question-4: What is the political system of Germany? sub-answer-4: Parliamentary republic o3-answer-4: Germany is a federal parliamentary republic Gemini 2.5 Pro-answer-4: Parliamentary republic Qwen2.5-VL-72B-answer-4: Parliamentary republic<br><br>reference source-4: https://en.wikipedia.org/wiki/Germany<br><br><br><br><br><br><br><br><br><br><br><br><br>multi-hop-question: What is the political system of the country which the company of the steam tram in the video belongs to? multi-hop-answer: Parliamentary republic o3-answer: Federal parliamentary republic Gemini 2.5 Pro-answer: Parliamentary republic Qwen2.5-VL-72B-answer: Parliamentary republic<br><br>[Figure 366]<br><br>[Figure 367]<br><br>[Figure 368]<br><br>[Figure 369]<br><br>[Figure 370]<br><br>[Figure 371]<br><br>[Figure 372]<br><br>[Figure 373]<br><br>[Figure 374]<br><br>[Figure 375]<br><br>[Figure 376]<br><br>[Figure 377]<br><br>[Figure 378]<br><br>[Figure 379]<br><br>[Figure 380]|
|---|

- Figure 23: Sampled examples in Video SimpleQA and the responses of typical LVLMs: part 2.

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

|sub-question-1: What is the name of the movie shown in the video?<br><br>sub-answer-1: Out West<br><br>o3-answer-1: Out West<br><br>Gemini 2.5 Pro-answer-1: Out West<br><br>Qwen2.5-VL-72B-answer-1: The Gold Rush<br><br>reference source-1: https://commons.wikimedia.org/wiki/Template:Motd/2015-10-01<br><br>sub-question-2: Who is the director of the movie Out West? sub-answer-2: Roscoe Arbuckle o3-answer-2: Roscoe Arbuckle Gemini 2.5 Pro-answer-2: Roscoe Arbuckle Qwen2.5-VL-72B-answer-2: Buster Keaton<br><br>reference source-2: https://en.wikipedia.org/wiki/Out_West_(1918_film)<br><br>sub-question-3: Which country was Roscoe Arbuckle born in? sub-answer-3: USA o3-answer-3: USA Gemini 2.5 Pro-answer-3: USA Qwen2.5-VL-72B-answer-3: USA<br><br>reference source-3: https://en.wikipedia.org/wiki/Roscoe_Arbuckle<br><br><br><br><br><br><br><br><br><br><br><br><br>multi-hop-question: Which country was the director of the movie shown in the video born in? multi-hop-answer: USA o3-answer: USA Gemini 2.5 Pro-answer: USA Qwen2.5-VL-72B-answer: USA<br><br>[Figure 392]<br><br>[Figure 393]<br><br>[Figure 394]<br><br>[Figure 395]<br><br>[Figure 396]<br><br>[Figure 397]<br><br>[Figure 398]<br><br>[Figure 399]<br><br>[Figure 400]<br><br>[Figure 401]<br><br>[Figure 402]<br><br>[Figure 403]|
|---|

- Figure 24: Sampled examples in Video SimpleQA and the responses of typical LVLMs: part 3.

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

|sub-question-1: Which mountain is shown in the video?<br><br>sub-answer-1: Huangshan<br><br>o3-answer-1: Huangshan (Yellow Mountain) in China<br><br>Gemini 2.5 Pro-answer-1: Huangshan (Yellow Mountain) in China.<br><br>Qwen2.5-VL-72B-answer-1: Huangshan<br><br>reference source-1: https://commons.wikimedia.org/wiki/Template:Motd/2016-08-08<br><br>sub-question-2: In which province of China is Huangshan located? sub-answer-2: Anhui o3-answer-2: Anhui province Gemini 2.5 Pro-answer-2: Huangshan is located in the Anhui province of China. Qwen2.5-VL-72B-answer-2: Anhui<br><br>reference source-2: https://en.wikipedia.org/wiki/Huangshan<br><br>sub-question-3: From the names of which two cities does the name 'Anhui' derive? sub-answer-3: Anqing and Huizhou o3-answer-3: Anqing and Huizhou Gemini 2.5 Pro-answer-3: Anqing and Huizhou Qwen2.5-VL-72B-answer-3: Hangzhou and Ningbo<br><br>reference source-3: https://en.wikipedia.org/wiki/Anhui<br><br><br><br><br><br><br><br><br><br><br><br><br>multi-hop-question: From the names of which two cities does the name of the province in China, where the mountain shown in the video is located, derive? multi-hop-answer: Anqing and Huizhou o3-answer: Anqing and Huizhou Gemini 2.5 Pro-answer: Anqing and Huizhou Qwen2.5-VL-72B-answer: Hangzhou and Ningbo<br><br>[Figure 415]<br><br>[Figure 416]<br><br>[Figure 417]<br><br>[Figure 418]<br><br>[Figure 419]<br><br>[Figure 420]<br><br>[Figure 421]<br><br>[Figure 422]<br><br>[Figure 423]<br><br>[Figure 424]<br><br>[Figure 425]<br><br>[Figure 426]|
|---|

- Figure 25: Sampled examples in Video SimpleQA and the responses of typical LVLMs: part 4.

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

|sub-question-1: What is the title of the movie whose trailer is shown in the video?<br><br>sub-answer-1: Lawrence of Arabia<br><br>o3-answer-1: Lawrence of Arabia<br><br>Gemini 2.5 Pro-answer-1: The title of the movie is Lawrence of Arabia.<br><br>Qwen2.5-VL-72B-answer-1: Lawrence of Arabia<br><br>reference source-1: https://commons.wikimedia.org/wiki/Template:Motd/2018-10-30<br><br>sub-question-2: In which year was Lawrence of Arabia released? sub-answer-2: 1962 o3-answer-2: 1962 Gemini 2.5 Pro-answer-2: 1962 Qwen2.5-VL-72B-answer-2: 1962<br><br>reference source-2: https://en.wikipedia.org/wiki/Lawrence_of_Arabia_(film)<br><br><br><br><br><br><br><br><br><br><br><br><br>multi-hop-question: Which year was the movie in the video released? multi-hop-answer: 1962 o3-answer: 1962 Gemini 2.5 Pro-answer: 1962 Qwen2.5-VL-72B-answer: 1962<br><br>[Figure 438]<br><br>[Figure 439]<br><br>[Figure 440]<br><br>[Figure 441]<br><br>[Figure 442]<br><br>[Figure 443]<br><br>[Figure 444]<br><br>[Figure 445]<br><br>[Figure 446]|
|---|

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

|sub-question-1: What type of gymnastics movements are demonstrated in the video?<br><br>sub-answer-1: Rhythmic gymnastics<br><br>o3-answer-1: Passes through a hoop in rhythmic gymnastics<br><br>Gemini 2.5 Pro-answer-1: Rhythmic gymnastics<br><br>Qwen2.5-VL-72B-answer-1: Rhythmic gymnastics<br><br>reference source-1: https://commons.wikimedia.org/wiki/Template:Motd/2017-07-24<br><br>sub-question-2: Which year did rhythmic gymnastics first appear as an Olympic medal event? sub-answer-2: 1984 o3-answer-2: 1984 Gemini 2.5 Pro-answer-2: 1984 Qwen2.5-VL-72B-answer-2: 1903<br><br>reference source-2: https://en.wikipedia.org/wiki/Lawrence_of_Arabia_(film)<br><br><br><br><br><br><br><br><br><br><br><br><br>multi-hop-question: Which year did the type of gymnastics movements demonstrated in the video first appear as an Olympic medal event? multi-hop-answer: 1984 o3-answer: 1984 Gemini 2.5 Pro-answer: 1984 Qwen2.5-VL-72B-answer: 1903<br><br>[Figure 458]<br><br>[Figure 459]<br><br>[Figure 460]<br><br>[Figure 461]<br><br>[Figure 462]<br><br>[Figure 463]<br><br>[Figure 464]<br><br>[Figure 465]<br><br>[Figure 466]|
|---|

- Figure 26: Sampled examples in Video SimpleQA and the responses of typical LVLMs: part 5.

