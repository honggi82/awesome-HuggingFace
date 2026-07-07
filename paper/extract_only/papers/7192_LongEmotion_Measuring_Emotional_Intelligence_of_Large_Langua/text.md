[Figure 1]

## LongEmotion: Measuring Emotional Intelligence of Large Language Models in Long-Context Interaction

# arXiv:2509.07403v2[cs.CL]11Jan2026

### Weichu Liu1∗, Jing Xiong2∗, Yuxuan Hu3, Zixuan Li4, Minghuan Tan5, Ningning Mao6, Hui Shen2,7,

Wendong Xu2, Chaofan Tao2, Min Yang5†, Chengming Li1†, Lingpeng Kong2, Ngai Wong2 1Shenzhen MSU-BIT University, 2The University of Hong Kong, 3City University of Hong Kong, 4Institute of Automation, Chinese Academy of Sciences, 5Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, 6Beijing Normal University, 7University of Michigan, Ann Arbor

Github: https://longemotion.github.io/ Huggingface: LongEmotion

[Figure 2]

### Abstract

Large language models (LLMs) make significant progress in Emotional Intelligence (EI) and long-context modeling. However, existing benchmarks often overlook the fact that emotional information processing unfolds as a continuous long-context process. To address the absence of multidimensional EI evaluation in long-context inference and explore model performance under more challenging conditions, we present LONGEMOTION, a benchmark that encompasses a diverse suite of tasks targeting the assessment of models’ capabilities in Emotion Recognition, Knowledge Application, and Empathetic Generation, with an average context length of 15,341 tokens. To enhance performance under realistic constraints, we introduce the Collaborative Emotional Modeling (COEM) framework, which integrates Retrieval-Augmented Generation (RAG) and multi-agent collaboration to improve models’ EI in long-context scenarios. We conduct a detailed analysis of various models in longcontext settings, investigating how reasoning mode activation, RAG-based retrieval strategies, and context-length adaptability influence their EI performance.

### 1 Introduction

Large Language Models (LLMs) are increasingly adopted in the domain of Emotional Intelligence (EI) (Wang et al., 2023). By leveraging their advanced language understanding and generation capabilities, LLMs become valuable tools for facilitating emotional expression (Ishikawa and Yoshino, 2025; Lu et al., 2025), with recent work showing

*Equal contribution. †Corresponding author.

their capacity to simulate specified emotional states in accordance with established models such as Russell’s Circumplex (Russell, 1980, 2003). LLMs are increasingly serving in roles ranging from mental health assistants (Guo et al., 2024; Malgaroli et al., 2025; Fu et al., 2024) to everyday conversational companions (Fu et al., 2024; Duan et al., 2024; Zhang et al., 2025). This growing integration into emotionally sensitive domains places greater demand on LLMs to maintain emotional coherence over time — not only to understand but also to remember, adapt, and respond empathetically in prolonged inference (Zhong et al., 2024).

Although existing benchmarks make considerable progress in measuring the EI of LLMs (Sabour et al., 2024; Huang et al., 2024), current evaluation still suffers from the following limitations: (i) As articulated by Affective Information Processing Theory (Lang and Cuthbert, 1984), humans continuously receive, process, organize, and respond to emotional information, which can manifest unique patterns of emotional intelligence within a longcontext setting. Existing studies often overlook the gap between idealized conditions and real-world scenarios: in realistic settings, the processing of emotional information is a continuous and enduring process. To bridge this gap, models should be evaluated on their EI in long context, which can be further decomposed into three key abilities: accurate emotion recognition, appropriate knowledge application, and affectively empathetic expression in long-context inference. (ii) Current research predominantly focuses on measuring a single aspect of the model’s capabilities, such as classification, expression, etc. According to the Mayer-Salovey-Caruso Emotional Intelligence Test

(MSCEIT) (Mayer et al., 2002), an individual’s EI encompasses multiple dimensions. Assessing only one specific capability is insufficient to fully represent the model’s EI. (iii) The emotional behavior of recent state-of-the-art techniques in long-context scenarios remains unexplored, especially for reasoning models with think mode, RAG-based agent methods, and other emerging approaches.

To bridge realistic scenarios and long-context evaluation, we introduce LONGEMOTION, a benchmark designed to comprehensively evaluate the EI of LLMs in long-context inference. LONGEMOTION comprises six complementary tasks. Two Emotion Recognition tasks, Emotion Classification and Emotion Detection, measure the model’s reasoning ability when key emotional information is located in noisy, long-context scenarios; two Empathetic Generation tasks, Emotion Conversation and Emotion Expression, evaluate the model’s empathy and expression abilities in the context of expansive multi-turn conversations or self-narratives; two Knowledge Application tasks, Emotion QA and Emotion Summary, probe how effectively the model leverages and applies emotional knowledge in authentic scenarios. Figure 1 depicts the dataset’s distribution.

To handle these realistic settings, we develop a Retrieval-Augmented Generation (RAG) approach as well as a novel multi-agent emotional modeling framework called Collaborative Emotional Modeling (COEM). Unlike standard RAG systems that pull from static, external corpora, our method treats the conversation history itself as a dynamic vector store to capture aspect-level sentiment terms. To further enhance EI in long context, we introduce COEM, where the context is divided into coherent chunks, initially ranked by relevance, and then processed by multiple collaborating agents (e.g., an auxiliary GPT-4o instance (OpenAI, 2024b)). After a second-stage re-ranking, these agents collectively generate an emotional “ensemble” response. This architecture captures the uncertainty and fluidity of real-world dialogue, allowing emotionally salient information to be continuously extracted, re-contextualized, and articulated. To further investigate the applicability of RAG techniques to long-context-based emotional tasks, we also adapt Self-RAG (Asai et al., 2024) and Search-o1 (Li et al., 2025) methods to LongEmotion by replacing their retrieval corpus with conversational context, thereby exploring broader possibilities of RAG in the domain of Emotional Intelligence. Our contri-

butions are summarized as:

- • We present LONGEMOTION, a long-context EI benchmark with six diverse tasks targeting recognition, generation, and psychological knowledge application.
- • We propose CoEM framework to enhance performance by retrieving and enriching contextually relevant information.
- • We perform extensive experiments across all settings and comprehensive case study, offering detailed analyses of LLMs’ EI in longcontext scenarios.

### 2 Related Work

Emotional Intelligence Benchmarks. Many benchmarks are developed to assess LLMs’ Emotional Intelligence (EI). Emobench (Sabour et al., 2024) draws on psychological theories to evaluate both emotional understanding and application across 400 English–Chinese handcrafted questions, exposing significant gaps between model and human EI levels. EQ-Bench (Paech, 2023) measures LLMs’ ability to rate emotional intensity in dialogues through 60 English queries, showing strong correlation with multi-domain reasoning benchmarks. More recently, EmotionQueen (Chen et al., 2024b) offers a specialized benchmark for empathy, requiring LLMs to recognize key events, implicit emotions, and generate empathetic responses. Despite their strengths, all of these focus on short or synthetic interactions and lack the long contextual depth critical for assessing EI in extended conversational or narrative settings.

Long-Context Modeling. LLMs make strides in processing long documents, yet robust evaluation remains an open challenge. LongBench (Bai et al., 2023) introduces a bilingual, multi-task benchmark covering QA, summarization, and code tasks with average context lengths over 6,000 words, revealing that even state-of-the-art models struggle with extended inputs. Complementing this, LooGLE (Li et al., 2023) evaluates longcontext reasoning using realistic documents exceeding 24k tokens, uncovering dependencies that span across distant spans. For extreme-length evaluation, XL2Bench (Ni et al., 2024) includes tasks on fiction, law, and scientific papers with inputs up to 100k+ words—yet LLMs still fall short in handling long-range dependencies. Beyond these,

[Figure 3]

[Figure 4]

(a) Token distributions across tasks. (b) Distribution of sample counts.

- Figure 1: (a) Sequence length denotes average model output length for Emotion Expression, and average input context length for other tasks. (b) Distribution of sample counts across the six tasks, illustrating the overall composition of the dataset.

RULER (Chen et al., 2023) focuses on complex reasoning chains in long-form texts via fine-grained question types and inter-paragraph dependencies, providing a valuable diagnostic lens into model reasoning depth. InfiniteBench (Sun et al., 2024), meanwhile, evaluates LLMs’ abilities on openended, unbounded contexts with theoretically unlimited input lengths, highlighting model degradation as input exceeds trained context windows. Survey work such as Liu et al. (2025) offers a broad overview of long-context modeling and evaluation paradigms but emphasizes that most benchmarks primarily target information retrieval or general comprehension—not emotional intelligence or affective computing.

### 3 LongEmotion: Construction and Task

A visual overview is shown in Figure 2. Appendix D provides a detailed explanation of metrics used in tasks where LLMs act as evaluators. We summarize the advantages of LongEmotion in enhancing LLMs’ EI in Appendix B.3

#### 3.1 Task Design

Emotion Classification. This task requires the model to identify the emotional category of a target entity within long-context texts that contain lengthy spans of context-independent noise (Kamradt, 2023). Model performance is evaluated by its accuracy against the ground truth.

Emotion Detection. The model is given N+1 emotional segments. Among them, N segments express the same emotion, while one segment expresses a unique emotion. The model is required to identify the single distinctive emotional segment.

During evaluation, the model’s score depends on whether the predicted index matches the groundtruth index.

Emotion QA. In this task, the model is required to answer questions grounded in long-context psychological literature. Model performance is evaluated using the F1 score between its responses and the ground truth answers.

Emotion Summary. In this task, the model is required to summarize the following aspects from long-context psychological pathology reports: (i) causes, (ii) symptoms, (iii) treatment process, (iv) illness characteristics, and (v) treatment effects. After generating the model’s response, we employ GPT-4o to evaluate its factual consistency, completeness, and clarity with respect to the reference answer. These three evaluation criteria are validated in CPsyExam (Zhao et al., 2024).

Emotion Conversation. In our four-stage longcontext counseling dialogue dataset, we select the quartile, half, and three-quarter points of each stage as evaluation checkpoints to assess the model’s EI capabilities. We introduce 12 specialized metrics informed by five major therapeutic frameworks: Cognitive Behavioral Therapy (CBT) (Beck, 2021), Acceptance and Commitment Therapy (ACT) (Waltz and Hayes, 2010), Humanistic Therapy (Elliott, 2002), Existential Therapy (May, 1994), and Satir Family Therapy (Rebner, 1972), which can be seen in Appendix D. The scoring is performed by GPT-4o, which serves as the evaluator to ensure consistency and scalability.

Emotion Expression. In this task, the model is situated within a specific emotional context and

[Figure 5]

- Figure 2: An illustrative overview of the LongEmotion dataset. To comprehensively evaluate the EI of LLMs in long-context interaction, we design six tasks: Emotion Classification, Emotion Detection, Emotion QA, Emotion Conversation, Emotion Summary, and Emotion Expression.

Task ID Source Construction Metric Avg len Count Emotion Recognition

Emotion Classification EC Emobench, FinEntity Segment Insertion Accuracy 30139 400 Emotion Detection ED Covid-worry Reorganization Accuracy 4106 136

###### Knowledge Application

Emotion QA QA Literature Human Annotation F1 11207 120 Emotion Summary ES CPsycoun Human Annotation LLM as Judge 15341 150

###### Empathetic Generation

Emotion Conversation MC CPsycoun Expansion LLM as Judge 4856 100 Emotion Expression EE EmotionBench Reorganization LLM as Judge 8546* 428

- Table 1: A statistical overview of the LONGEMOTION dataset. ID denotes task abbreviations. EC, ED, QA, MC, and ES involve long-text input, with Avg len showing average context length. EE is a long-text generation task—Avg len here refers to average output length (marked with *).

prompted to produce a long-form emotional selfnarrative. Models first complete a psychometric self-assessment (e.g., PANAS), followed by the generation of a structured narrative spanning five phases: (i) Immediate Reaction, (ii) Cognitive Appraisal, (iii) Emotional and Physiological Expression, (iv) Regulation Strategies, and (v) Reflective Integration. The evaluation encompasses six dimensions: emotional consistency, content redundancy, expressive richness, cognition–emotion interplay, self-reflectiveness, and narrative coherence. All dimensions are assessed by GPT-4o, which serves as the evaluator to score the model’s capacity for emotional expression.

#### 3.2 Data Construction

The statistical overview of LongEmotion dataset can be found in Table 1. EC and ED tasks focus on evaluating the model’s ability in emotional recognition. QA and ES tasks emphasize the model’s capability to apply knowledge within long-context scenarios. MC and EE tasks aim to measure the model’s generative ability.

Reorganization from Existing Datasets. In Emotion Classification, we embed short excerpts from Emobench (Sabour et al., 2024) and FinEntity (Tang et al., 2023) into BookCorpus passages (Zhu et al., 2015), by randomly inserting snippets and manually adjusting proper nouns for coherence. In Emotion Detection, we build contrast sets by grouping texts from Covid-worry (Klein-

berg et al., 2020; van der Vegt and Kleinberg, 2023) by emotion label and inserting mismatched segments. In Emotion Expression, we use situations from EmotionBench (Huang et al., 2024) to provide models with specific emotional contexts.

Expansion and Human Annotation For Emotion Conversation, based on CPsyCoun (Zhang et al., 2024), we construct 100 emotionally rich dialogues by expanding seed prompts into four functional stages: (i) Reception and Inquiry, (ii) Diagnostic, (iii) Consultation, and (iv) Consolidation and Ending. Dataset quality is evaluated through two parallel protocols: (i) manual scoring by psychology experts and (ii) automated assessment with GPT-4o. As reported in Figure 3, the Pearson correlation between LLM and human scores reaches 0.934 (p = 0.066), indicating a relatively high alignment. In addition, we use the same prompts and GPT model for evaluation as those employed in the quality assessment, which further validates the rationality of our LLM-as-Judge setting. Annotator qualifications are detailed in Appendix A.

[Figure 6]

- Figure 3: Quality Evaluation on Emotion Conversation.

In Emotion Summary, drawing on CPsyCounR dataset, we first expand the experience and reflection section of the dataset to meet our requirements for long-context inputs. Next, psychology annotators label each sample across five standardized dimensions: (i) Causes, (ii) Symptoms, (iii) Treatment Process, (iv) Illness Characteristics, and (v) Treatment Effect. Finally, by filtering samples based on format, content richness, and precision, we select a final set of 150 samples. To further extend the dataset length while preserving the original semantic integrity, we employ DeepSeekV3 (DeepSeek-AI, 2024) to perform structured decomposition and subsequent content augmentation. In Appendix B.3, we discuss the annotation disci-

pline for the annotation process of Emotion Summary.

In constructing Emotion QA, the annotation pipeline is illustrated in Figure 4. The construction process on psychological literature involves: (i) expert-written questions targeting emotional understanding, (ii) refinement of reference answers for clarity and consistency with F1-based evaluation, and (iii) filtering based on model performance to exclude overly ambiguous or trivial examples. Through this series of manual annotation and selection, we finally obtain 120 high-quality pairs of psychological knowledge questions and answers.

[Figure 7]

Figure 4: Annotation process of Emotion QA.

### 4 Collaborative Emotional Modeling

Figure 5 illustrates the pipeline of CoEM. To address EI tasks involving long contexts, we propose a hybrid retrieval-generation architecture that combines Retrieval-Augmented Generation (RAG) with modular multi-agent collaboration. For the parameter settings and application details, please refer to Appendix C. For the case analysis of RAG and CoEM, please refer to Appendix B.2. The framework consists of five key stages:

Chunking. The context is segmented into tokenlength-constrained chunks, whereas in Emotion Detection, each segment is considered as an individual chunk. We set different chunk sizes based on the characteristics of each task. We demonstrate the parameter settings in Appendix C.

[Figure 8]

Figure 5: The pipeline of Collaborative Emotional Modeling (CoEM).

Initial Ranking. A retrieval agent, implemented as CoEM-Rank, evaluates the relevance of each chunk to the query using dense semantic similarity, with relevance scores computed based on cosine similarity. Top-ranked chunks are passed forward for enhancement. By ranking the original context chunks, the factual relevance of the retrieved information is ensured.

Multi-Agent Enrichment. A reasoning agent called CoEM-Sage, functioning as a knowledge assistant, enriches the selected chunks by incorporating external knowledge or latent emotional signals through our task-specific prompts. Specifically, in Emotional Recognition tasks, CoEM-Sage identifies subtle emotional cues; in Knowledge Application tasks, it provides summaries based on psychological knowledge; and in Empathetic Generation tasks, it enhances CoEM-Core’s empathy and expression through emotional analysis. These signals, derived from psychological theories or curated priors, are incorporated into the original chunks without task-specific leakage.

Re-Ranking. The enriched chunks, now augmented with emotional features, are then reevaluated by CoEM-Rank for their semantic relevance to the query, measured by cosine similarity. This final ranking ensures that the selected context is not only factually grounded but also affectively coherent. By ranking the enriched chunks, the emotional relevance of the retrieved information is ensured, as these chunks contain not only the original text but also external emotional information.

Emotional Ensemble Generation. The selected and enriched chunks, along with the context and prompt, is fed into a generation model denoted as CoEM-Core. This model (e.g., a long-context LLM or an instruction-tuned model) produces the final task-specific output, whether it be classification,

summarization, or dialogue generation.

This modular approach encourages interpretability, emotional awareness, and task robustness. The CoEM setting encompasses all five stages, while the RAG setting only comprises Chunking, onetime Ranking, and Emotional Ensemble Generation. We conduct an empirical case study of the framework, which can be found in Appendix B.2.

### 5 Experiment

#### 5.1 Experiment Setup

In our experiments, for closed-source models, we choose GPT-4o-mini (OpenAI, 2024a) and GPT-4o, while for open-source models, we select DeepSeek-V3 (DeepSeek-AI, 2024), Llama3.18B-Instruct (Grattafiori et al., 2024), and Qwen38B (Team, 2025). For tasks employing automatic evaluation, we adopt GPT-4o as the evaluator. Under the base setting, we compare a broader range of advanced open-source and closed-source models. For comparison, we have the performance of GPT5 (OpenAI, 2025), Qwen3-14B and Qwen3-32B under the Base setting.

To accelerate inference, we use vllm library (Kwon et al., 2023) as the inference engine and set temperature=0.8 and top_p=0.9 for all open-source models. For Qwen3 series models, we enable its thinking capabilities and manually remove the reasoning process between <think> and </think> to keep the answers concise. All experiments are conducted using NVIDIA A800 80G GPUs, with open-source models under 14B parameters running on a single GPU and the 32B models utilizing two GPUs. In the EC (Emobench as needle), ED, and EE, we employ GPT-4o as the CoEM-Sage, while DeepSeek-V3 is used for the EC (Finentity as needle), QA, MC and ES in the same role. For the retrieval and ranking components across both the RAG and CoEM

Recognition Knowledge Generation Overall EC ED QA ES MC-4 EE Avg

Method Model

GPT-4o-mini 37.00 16.42 48.61 4.54 3.75 86.77 59.10

- GPT-4o 50.09 19.12 50.12 4.60 3.77 81.03 61.29 DeepSeek-V3 56.50 24.51 45.53 4.62 3.99 81.75 63.42 Qwen3-8B 48.00 18.14 44.75 4.51 3.97 73.40 58.98 Llama3.1-8B-Instruct 39.34 9.80 44.56 4.29 4.00 75.61 55.85 (Extended Comparison Models)
- GPT-5 73.75 22.79 43.22 4.42 4.67 86.77 68.06 Qwen3-14B 50.00 20.83 46.35 4.55 3.95 84.49 61.95 Qwen3-32B 58.25 20.59 43.11 4.53 4.17 84.81 63.46

Base

GPT-4o-mini 51.67 21.57 50.72 4.53 3.78 80.41 61.76 ↑2.66 GPT-4o 61.34 22.55 51.81 4.52 3.80 79.49 63.60 ↑2.31 DeepSeek-V3 62.59 23.53 50.44 4.63 4.34 81.83 66.30 ↑2.88 Qwen3-8B 41.59 19.12 44.34 4.54 4.14 73.28 58.65 ↓0.33 Llama3.1-8B-Instruct 44.00 11.27 43.21 4.26 3.94 75.16 56.27 ↑0.42

RAG

GPT-4o-mini 59.50 20.59 49.12 4.52 3.77 80.38 62.57 ↑3.47 GPT-4o 61.42 25.00 51.07 4.53 3.81 80.41 64.12 ↑2.83 DeepSeek-V3 64.17 23.04 50.39 4.65 4.34 82.83 66.70 ↑3.28 Qwen3-8B 62.92 18.14 51.11 4.55 4.14 73.59 63.26 ↑4.28 Llama3.1-8B-Instruct 55.09 11.27 44.79 4.17 4.00 75.71 58.38 ↑2.53

CoEM

- Table 2: Experiment result across Base, RAG and CoEM. MC-4 represents the fourth stage of Emotion Conversation. By aligning the MC-4 and ES scores with the 100-point scale, the overall score is computed as (EC + ED + EE + QA + MC-4×20 + ES×20)/6, where the numbers to the right indicate the score change relative to the Base setting.

settings, we adopt bge-m3 (Chen et al., 2024a) as the CoEM-Rank. The generation models listed in Table 2 are used as the CoEM-Core. Configuration details for both the RAG and CoEM frameworks are in Appendix C.

#### 5.2 Results on LongEmotion

The overall experimental results can be seen in Table 2. We evaluate the performance of each model on all tasks under the Base, RAG, and CoEM settings. As the first three stages of the dialogue are relatively brief, RAG and CoEM are only applied in the fourth stage of the Emotion Conversation.

Overall Analysis of Experimental Results. As shown in Table 2, DeepSeek-V3 and GPT models exhibit generally strong EI capabilities, achieving stable performance gains even with vanilla RAG. In contrast, Qwen3-8B and Llama-3.1-8B-Instruct perform less effectively under the RAG setting, suggesting that some models struggle to effectively integrate retrieved chunks within long-context reasoning. This limitation can be mitigated by CoEM, which enhances contextual alignment and emotional reasoning through multi-agent collaboration.

Ablation Experiments. To evaluate the effectiveness of RAG-based methods in enhancing EI, we integrate Self-RAG and Search-o1 into LongEmotion using Qwen3-8B as the base model. In the Self-RAG setting, retrieved chunks are rescored by Self-RAG-7B for relevance, with irrelevant ones filtered out before concatenation with the prompt. The additional use of Self-RAG-7B outputs in the ES task further improves performance, showing that selectively enriching retrieved information benefits emotional intelligence. In the Search-o1 setting, Qwen3-8B autonomously generates queries and retrieves relevant chunks via Bge-m3 within five search turns. The observed performance drop indicates that small-scale models struggle with autonomous search-based reasoning in emotional tasks. Results are reported in Table 3.

To investigate how the reasoning processes of models affect their Emotional Intelligence in longcontext scenarios, we perform ablation studies on the Qwen3 model series using two emotion recognition tasks—Emotion Classification (Emobench as Needle) and Emotion Detection—along with one empathetic generation task, Emotion Expres-

#### Method EC ED QA ES MC-4

RAG 41.59 19.12 44.34 4.51 4.14 Self-RAG 44.00 16.18 44.02 4.57 4.15 Search-o1 45.25 16.18 45.12 4.50 3.72 CoEM 62.92 18.14 51.11 4.55 4.14

- Table 3: Ablation experiment results on methods.

sion, under the Base setting. By analyzing Table 4, we can observe that through thinking, Qwen3-8B achieve the most significant improvement, while the improvement of Qwen3-14B is not substantial.

Task

Qwen3-8B Qwen3-14B Qwen3-32B think w/o think w/o think w/o

EC-E 38.50 28.67 31.00 30.75 48.00 37.50

- ED 18.14 12.01 20.83 20.83 20.59 20.10
- EE 73.40 70.32 84.49 83.13 84.81 84.02

- Table 4: Ablation experiments of the thinking process in the Qwen3 series models.

Furthermore, to examine how the capability of CoEM-Sage affects the overall framework, we perform ablation experiments on the MC-4 task. As shown in Table 2, DeepSeek-V3 outperforms GPT-

- 4o under the base setting. Consistently, when used as the CoEM-Sage, DeepSeek-V3 also drives higher performance than GPT-4o, as can be seen

- in Figure 6. These results further demonstrate the soundness and scalability of CoEM.

[Figure 9]

Figure 6: Impact of CoEM-Sage models on MC-4.

To explore models’ ability in emotion recognition across different context lengths, we evaluate their performance on the Emotion Classification (Finentity as Needle) under Base setting, as shown

- in Figure 7. DeepSeek-V3 and Qwen3-8B exhibit both high stability and strong overall performance, whereas GPT-based models show weaker robust-

ness in long-context settings, in some cases even performing below Llama-3.1-8B-Instruct.

[Figure 10]

- Figure 7: Model accuracy by context length on EC.

We further conduct ablation experiments on RAG with varying chunk sizes and retrieval counts, as shown in Figure 8. GPT-4o-mini performs best with 128-token chunks and eight retrieved segments, while larger settings introduce noise and reduce overall performance.

[Figure 11]

- Figure 8: Impact of chunk size and retrieved count on GPT-4o-mini’s RAG performance on Emotion QA.

Case Study. (i) First, we qualitatively compare the GPT model series across all tasks under the Base setting, revealing that GPT-5 is theoretically stronger but more mechanical and prone to hallucination, GPT-4o-mini exhibits more human-like behavior yet lacks theoretical grounding, while GPT4o achieves a balanced trade-off. (ii) Furthermore, we visualize the CoEM framework and empirically analyze its influence on emotional information. (iii) Finally, we analyze the advantages of the LongEmotion dataset in advancing Emotional Intelligence. For complete details of case study, please refer to Appendix B.

### 6 Conclusion

In this work, we introduce LONGEMOTION, a benchmark for measuring models’ Emotional Intelligence in long-context scenarios. LONGEMOTION comprises six tasks that comprehensively challenge

models across emotion recognition, knowledge application and empathetic generation. Beyond constructing the dataset, we also build RetrievalAugmented Generation (RAG) and Collaborative Emotional Modeling (CoEM) frameworks for each task, achieving improvements on the vast majority of them. We conduct exhaustive experiments and a detailed case study to analyze models’ EI in long-context scenarios.

### 7 Limitations

In this work, we propose LongEmotion, a benchmark for evaluating the emotional intelligence of LLMs in long-context inference. However, all the datasets in our benchmark are based solely on the text modality and are restricted to the psychological and emotional domains. Similarly, the proposed CoEM framework focuses only on textual inputs and does not extend to other modalities such as vision or audio. In addition, our dataset includes only English texts and does not cover other languages. It remains uncertain whether the same level of quality can be preserved when the data are translated into other languages.

### 8 Ethical Considerations

Data Privacy In this work, all the datasets we adopt are formally published in academic venues and comply with data privacy and ethical protection standards. Through data augmentation and manual inspection, we ensure that no ethical risks are introduced. In addition, all annotators involved in our dataset construction possess academic backgrounds in computer science or psychology, ensuring the reliability of the data annotation process. We adhere to the intended use and license terms of all source datasets. The datasets in LongEmotion are intended solely for academic research and will not be used for any other purposes.

Potential Risks All models evaluated in our experiments, including both open-source and closedsource ones, are officially released models, which helps ensure that no harmful or unsafe content is generated. In addition, all prompts used in our evaluation are fully disclosed in this paper, which can be seen in Appendix F, and these prompts are carefully designed to ensure a high level of safety.

### References

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, and 1 others. 2023. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508.

Judith S. Beck. 2021. Cognitive Behavior Therapy: Basics and Beyond, 3rd edition. The Guilford Press.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024a. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. Preprint, arXiv:2402.03216.

Yifan Chen, Yankai Lin, Jie Zhou, and Minlie Huang. 2023. Ruler: A diagnostic benchmark for longcontext reasoning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Yuyan Chen, Hao Wang, Songzhou Yan, Sijia Liu, Yueze Li, Yi Zhao, and Yanghua Xiao. 2024b. Emotionqueen: A benchmark for evaluating empathy of large language models. arXiv preprint arXiv:2409.13359.

DeepSeek-AI. 2024. Deepseek-v3 technical report. Preprint, arXiv:2412.19437.

Jinhao Duan, Xinyu Zhao, Zhuoxuan Zhang, Eunhye Grace Ko, Lily Boddy, Chenan Wang, Tianhao Li, Alexander Rasgon, Junyuan Hong, Min Kyung Lee, and 1 others. 2024. An exploration of llm-guided conversation in reminiscence therapy. In GenAI for Health: Potential, Trust and Policy Compliance.

Robert Elliott. 2002. The effectiveness of humanistic therapies: A meta-analysis.

Yumeng Fu, Junjie Wu, Zhongjie Wang, Meishan Zhang, Lili Shan, Yulin Wu, and Bingquan Li. 2024. Laerc-s: Improving llm-based emotion recognition in conversation with speaker characteristics. arXiv preprint arXiv:2403.07260.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, and 1 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Z Guo, A Lai, JH Thygesen, J Farrington, T Keen, and K Li. 2024. Large language model for mental health: A systematic review. arxiv 2024. arXiv preprint arXiv:2403.15401.

Jen-tse Huang, Man Ho Lam, Eric John Li, Shujie Ren, Wenxuan Wang, Wenxiang Jiao, Zhaopeng Tu, and Michael R. Lyu. 2024. Apathetic or empathetic? evaluating LLMs’ emotional alignments with humans. In Advances in Neural Information Processing Systems 37.

Shin-nosuke Ishikawa and Atsushi Yoshino. 2025. Ai with emotions: Exploring emotional expressions in large language models. arXiv preprint arXiv:2504.14706.

Greg Kamradt. 2023. Needle in a haystack - pressure testing llms. https://github.com/ gkamradt/LLMTest_NeedleInAHaystack. Accessed: 2025-07-23.

Bennett Kleinberg, Isabelle van der Vegt, and Maximilian Mozes. 2020. Measuring Emotions in the

COVID-19 Real World Worry Dataset. In Proceedings of the 1st Workshop on NLP for COVID-19 at ACL 2020, Online. Association for Computational Linguistics.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Peter J Lang and Bruce N Cuthbert. 1984. Affective information processing and the assessment of anxiety. Journal of Behavioral Assessment, 6(4):369–395.

Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. 2023. Loogle: Can long-context language models understand long contexts? arXiv preprint arXiv:2311.04939.

Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic searchenhanced large reasoning models. arXiv preprint arXiv:2501.05366.

Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun Wang, Chenchen Zhang, Ge Zhang, Jiebin Zhang, and 1 others. 2025. A comprehensive survey on long context language modeling. arXiv preprint arXiv:2503.17407.

Haifeng Lu, Jiuyi Chen, Feng Liang, Mingkui Tan, Runhao Zeng, and Xiping Hu. 2025. Understanding emotional body expressions via large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 1447–1455.

Matteo Malgaroli, Katharina Schultebraucks, Keris Jan Myrick, Alexandre Andrade Loch, Laura OspinaPinillos, Tanzeem Choudhury, Roman Kotov, Munmun De Choudhury, and John Torous. 2025. Large language models for the mental health community: framework for translating code to care. The Lancet Digital Health, 7(4):e282–e285.

Rollo May. 1994. Discovery of being: Writings in existential psychology. WW Norton & Company.

John D Mayer, Peter Salovey, and David R Caruso.

2002. Mayer-salovey-caruso emotional intelligence test (msceit) users manual.

Xuanfan Ni, Hengyi Cai, Xiaochi Wei, Shuaiqiang Wang, Dawei Yin, and Piji Li. 2024. Xl 2 bench: A benchmark for extremely long context understanding with long-range dependencies. arXiv preprint arXiv:2404.05446.

OpenAI. 2024a. GPT-4o Mini: Advancing CostEfficient Intelligence. https://openai.com/ zh-Hans-CN. Accessed: 2025-07-24.

- OpenAI. 2024b. OpenAI: Hello GPT-4o. https://openai.com/zh-Hans-CN/ index/hello-gpt-4o/. Accessed: 2025-0724.
- OpenAI. 2025. OpenAI: GPT-5. https://openai. com/zh-Hans-CN/gpt-5/. Accessed: 202508-24.

Samuel J Paech. 2023. Eq-bench: An emotional intelligence benchmark for large language models. arXiv preprint arXiv:2312.06281.

I. Rebner. 1972. Conjoint family therapy. Psychotherapy: Theory, Research & Practice, 9(1):62–66.

James A Russell. 1980. A circumplex model of affect. Journal of personality and social psychology, 39(6):1161.

James A Russell. 2003. Core affect and the psychological construction of emotion. Psychological review, 110(1):145.

Sahand Sabour, Siyang Liu, Zheyuan Zhang, June M Liu, Jinfeng Zhou, Alvionna S Sunaryo, Juanzi Li, Tatia Lee, Rada Mihalcea, and Minlie Huang. 2024. Emobench: Evaluating the emotional intelligence of large language models. arXiv preprint

- arXiv:2402.12071.

Maosong Sun, Liangming Gao, and 1 others. 2024. Infinitebench: Towards evaluating llms on unbounded long-context tasks. arXiv preprint

- arXiv:2403.07486.

Yixuan Tang, Yi Yang, Allen Huang, Andy Tam, and Justin Tang. 2023. Finentity: Entity-level sentiment classification for financial texts. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 15465–15471.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Isabelle van der Vegt and Bennett Kleinberg. 2023. A multi-modal panel dataset to understand the psychological impact of the pandemic. Scientific data, 10(1):537.

T. J. Waltz and S. C. Hayes. 2010. Acceptance and commitment therapy. In N. Kazantzis, M. A. Reinecke, and A. Freeman, editors, Cognitive and Behavioral Theories in Clinical Practice, pages 148–192. The Guilford Press.

Xuena Wang, Xueting Li, Zi Yin, Yue Wu, and Jia Liu. 2023. Emotional intelligence of large language models. Journal of Pacific Rim Psychology, 17:18344909231213958.

Chenhao Zhang, Renhao Li, Minghuan Tan, Min Yang, Jingwei Zhu, Di Yang, Jiahao Zhao, Guancheng Ye, Chengming Li, and Xiping Hu. 2024. Cpsycoun: A report-based multi-turn dialogue reconstruction and evaluation framework for chinese psychological counseling. arXiv preprint arXiv:2405.16433.

Xue Zhang, Mingjiang Wang, Xuyi Zhuang, Xiao Zeng, and Qiang Li. 2025. Cdea: Causality-driven dialogue emotion analysis via llm. Symmetry, 17(4):489.

Jiahao Zhao, Jingwei Zhu, Minghuan Tan, Min Yang, Renhao Li, Di Yang, Chenhao Zhang, Guancheng Ye, Chengming Li, Xiping Hu, and 1 others. 2024. Cpsyexam: A chinese benchmark for evaluating psychology using examinations. arXiv preprint arXiv:2405.10212.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19724–19731.

Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In The IEEE International Conference on Computer Vision (ICCV).

### A Qualifications of Annotators

Our annotation team consists of psychology researchers and computer science researchers. In the psychology research team, there is a postdoctoral fellow expert specializing in psychology and seven Master’s students majoring in the same field. The theoretical foundation of our dataset and metrics involves deep participation from the psychology team. Under the guidance of the expert, the seven psychology Master’s students carry out the annotation work. In the computer science research team, there are three Master’s students and one PhD student majoring in computer science. Their main responsibility is to modify, adjust, and organize the data annotated by the psychology team according to the characteristics of the tasks. All student annotators and researchers involved in the annotation and data processing work receive reasonable financial compensation for their time and effort, commensurate with local standards and the complexity of the tasks.

### B Case Study

#### B.1 Comparison of GPT series models

From Table 2, it can be seen that GPT-5’s overall capabilities surpass those of GPT-4o and GPT-4omini. In the tasks of Emotion Classification and Emotion Detection, we only prompt the models to output the final label. The results show that GPT5’s reasoning ability is significantly better than that of GPT-4o and GPT-4o-mini.

In the Emotion QA task, GPT-4o and GPT-4omini tend to respond more literally based on the original text, which can be seen in Figure 9. In contrast, GPT-5 modifies content according to its own understanding, which leads to a lower F1 score due to reduced alignment with the ground truth.

[Figure 12]

Figure 9: Comparison of the performance of different versions of GPT models on Emotion QA.

In the Emotion Conversation task, GPT-5 achieved higher scores based on our psychology theory-driven metrics. However, by examining the model outputs in Figure 10, we can see that GPT-5 merely makes better use of psychological knowl-

edge to offer advice to the patient, rather than genuinely demonstrating empathy toward the client.

[Figure 13]

- Figure 10: Comparison of the performance of different versions of GPT models on Emotion Conversation.

In the Emotion Expression task, GPT-4o-mini performed more like a real person, with the generated content closely resembling what an actual individual might say in a given situation. In contrast, GPT-4o’s expressions were more like a rigidly told story, lacking natural fluidity. Meanwhile, GPT-5’s generation was more comprehensive and balanced, providing a well-rounded and objective description of emotions across various features, as clearly shown in Figure 11.

[Figure 14]

- Figure 11: Comparison of the performance of different versions of GPT models on Emotion Expression.

In the Emotion Summary task, both GPT-4omini and GPT-4o directly analyze various features of the case, while GPT-5 structures its analysis based on psychological theories. However, GPT-5 exhibits hallucinations, often adding non-existent facts. For instance, in Figure 12, the term "slapping" is highlighted in red, but the source data never mentions such an action.

From the tasks above, we can conclude that GPT-

[Figure 15]

Figure 12: Comparison of the performance of different versions of GPT models on Emotion Summary.

4o-mini behaves more like a human, with richer emotional features, but its application of psychological theory is somewhat lacking. On the other hand, GPT-5 has a better understanding of psychological theories, but the output is too rigid and mechanical, which might lead to a less empathetic user experience in practice. Additionally, GPT-5 tends to exhibit hallucinations, often adding non-existent facts. GPT-4o strikes a more balanced approach between theoretical understanding and emotional features.

#### B.2 Case Analysis of RAG and CoEM

We conduct a concrete analysis of how the information retrieved by the RAG and CoEM methods affects model performance. In models’ final generation prompts, the Base setting includes none of the information; the RAG setting includes only the Chunk information; and the CoEM setting includes both the Chunk and Summary information.

Emotion Classification. In this task, the model is given a long context in which an emotional segment is embedded within unrelated noise. The RAG method enables the model to retrieve a more accurate segment, leading to improved performance; CoEM further conducts emotional analysis on the retrieved segment, resulting in the greatest performance improvement, as shown in Figure 13.

[Figure 16]

- Figure 13: Case analysis of RAG and CoEM in Emotion Classification.

Emotion Detection. In this task, the model receives multiple emotional segments. The RAG method ranks the original segments based on their relevance, while CoEM further enhances the emotional features of the segments and ranks the enriched packs. This relevance-based ranking approach significantly boosts the model’s ability to distinguish emotions. We skip the Initial-Ranking to capture richer emotional features. After enhancing the chunks with Multi-Agent Enrichment, we perform Re-Ranking to select the chunks that are least similar to others, as shown in Figure 14.

[Figure 17]

- Figure 14: Case analysis of RAG and CoEM in Emotion Detection.

Emotion QA. In this task, we evaluate the model’s responses based on the F1 similarity with the ground truth. RAG helps the model retrieve more relevant source content, thereby improving its performance. Next, CoEM-Sage performs extraction on each retrieved chunk based on the query, retaining only the parts that are relevant to the query, as shown in Figure 15.

[Figure 18]

- Figure 15: Case analysis of RAG and CoEM in Emotion QA.

Emotion Conversation. In this task, the model is placed within a multi-turn dialogue context. The RAG method ranks the context chunks based on

their relevance to the previous three dialogue turns. CoEM, after the initial ranking, generates a summary by combining the previous three turns with the initially selected chunks, and then performs a second round of relevance ranking between the initially filtered chunks and this summary, further ensuring the accuracy of the relevance assessment, as shown in Figure 16.

[Figure 19]

- Figure 16: Case analysis of RAG and CoEM in Emotion Conversation.

Emotion Summary. In this task, the model is required to summarize specific characteristics of a psychological counseling report. RAG ranks the chunks based on their similarity to the target characteristics. CoEM further injects the analysis of these chunks provided by CoEM-Sage, as shown in Figure 17.

[Figure 20]

- Figure 17: Case analysis of RAG and CoEM in Emotion Summary.

Emotion Expression. In this task, the model is placed in an emotional situation, where it is required to answer the PANAS scale and express its emotions. RAG ranks the context chunks based on the query at each stage, while CoEM performs a finer-grained emotional analysis of these chunks. The CoEM-Sage model, with its stronger emotional intelligence (EI) capabilities, captures emotional cues more precisely, which in turn helps the tested CoEM-Core model better understand and express its own emotions, as shown in Figure 18.

- B.3 Advantages of LongEmotion in Enhancing Emotional Intelligence

In this section, we discuss the advantages of the LongEmotion benchmark in promoting the full utilization of LLMs’ Emotion Intelligence capabilities in long-context interaction.

[Figure 21]

Figure 18: Case analysis of RAG and CoEM in Emotion Expression.

Psychological theories guided benchmark design. In the Emotion Conversation task, we design scientifically rigorous evaluation metrics based on various psychological therapies and stages of dialogue data. For the Emotion Summary task, annotators summarize key elements of patient records considering physiological factors, personal growth history, and social factors, which can be seen in Table 5. In the Emotion Expression task, under given scenarios, models are guided to perform staged long-text self-expression in the rigorously designed framework.

- i) Biological, Genetic & Medical Factors. e.g., family medical history.
- ii) Lifestyle Habits. e.g., sleep, diet, and exercise patterns.

Physiological Factors

- i) Quality of interpersonal relationships during development.
- ii) Academic and occupational performance during development.

Growth History

- i) Family support system. e.g., emotional and financial support.
- ii) Peer support system. e.g., friendship, social belonging and trust.
- iii) Stressful life events. e.g., bereavement, job loss and daily stress.

Social Factors

Table 5: Annotation discipline for the annotation process of Emotion Summary.

Quality-guaranteed synthetic translation data. We employ the two-stage generation framework of CPsyCoun to generate Emotion Conversation dataset, and compare it with the direct use of a single-stage straightforward generation without the counseling note and the detailed skills in the prompt. The prompt we use can be found in Figure 19, and the comparison of experimental results can be seen in Table 6.

Comprehensive Experiments and In-Depth Case Studies. We conducted extensive experiments on Base, RAG, and CoEM frameworks, accompanied by detailed case studies based on model outputs. Under the LongEmotion benchmark, various models exhibited distinct limitations—even the most advanced GPT-5 demonstrated issues such as overly mechanical responses despite its stronger theoretical capabilities.

### C Details of RAG and CoEM

We present the application details of the CoEM framework in Table 7. To ensure the accuracy of

###### Metric One-Stage Two-Stage

Establishing the Therapeutic Alliance 4.88 4.92 Emotional Acceptance and Exploration Guidance 4.36 4.38 Systematic Assessment 3.86 3.79 Recognizing Surface-Level Reaction Patterns 4.13 4.10 Deep Needs Exploration 4.13 4.32 Pattern Interconnection Analysis 3.66 3.77 Adaptive Cognitive Restructuring 3.60 3.73 Emotional Acceptance and Transformation 4.12 3.96 Value-Oriented Integration 3.94 3.69 Consolidating Change Outcomes and Growth Narrative 4.52 4.63 Meaning Integration and Future Guidance 4.16 4.19 Autonomy and Resource Internalization 4.84 4.86

Avg 4.18 4.20

- Table 6: The comparison experiment results of synthetic data. One-Stage represents straightforward generation without the counseling note and the detailed skills. TwoStage represents our generation method.

the ranking, in the Emotion Detection task, we skip the initial ranking and directly carry out multi-agent enrichment. The Chunking and Re-Ranking in the table are also applicable to the RAG framework.

Task Chunking Initial Ranking Multi-Agent Enrichment

Re-Ranking

EC Chunk by length Compute chunk-query similarity

External injection into each chunk

Compute chunk-query similarity

- ED Each segment as a chunk

Skip this stage External injection into each chunk

Select chunks with lowest similarity scores

QA Chunk by length Compute chunk-query similarity

External injection into each chunk

Compute chunk-query similarity

MC-4 Chunk by length Compute chunk-query similarity

Generate an overall summary

Compute chunk-query similarity

ES Chunk by length Compute chunk-query similarity

External injection into each chunk

Compute chunk-query similarity

- EE Chunk by length Compute chunk-query similarity

External injection into each chunk

Compute chunk-query similarity

- Table 7: Application details in the CoEM framework.

We also report the chunk size and retrieved count for each task in Table 8. In QA, models use different chunk sizes. For EE, the retrieved counts correspond to stages 2–5. The retrieved count of the one-time ranking in RAG is the same as the parameter settings for Re-Ranking in the table.

### D LLM as Judge Metrics Design

In this section, we provide a detailed presentation of the metric designs that employ large models as evaluators.

Emotion Summary. In the Emotion Summary, we design three metrics—consistency, completeness, and clarity—with respect to the reference answer. Table 9 shows the explanations of these metrics:

Emotion Conversation. In the Emotion Conversation task, we design metrics for each dialogue stage based on Cognitive Behavioral Ther-

Task Chunk Size Initial Ranking Re-Ranking EC 128 1 1

- ED Num of segs – 8 QA GPT-4o-mini 128 16 8 GPT-4o 128 16 4 Deepseek-V3 512 8 4 Qwen3-8B 128 16 4 Llama-3.1-8B-Instruct 512 8 4 MC-4 128 16 4 ES 128 8 4

- EE 128 4,8,8,8 2,4,4,4

- Table 8: Parameter settings applied to CoEM. Initial Ranking and Re-Ranking denote the number of chunks retrieved in each respective stage.

Metric Description Factual Consistency Is the model output factually aligned

with the ground truth? Completeness Does the model include all key details found in the ground truth? Clarity Is the expression clear and coherent?

- Table 9: Design of Emotion Summary evaluation metrics.

apy (CBT), Acceptance and Commitment Therapy (ACT), Humanistic Therapy, Existential Therapy, and Satir Family Therapy. The description and theoretical foundations for the design of each metric can be found in Table 10.

Emotion Expression. In the Emotion Expression task, we design six metrics—emotional consistency, content redundancy, expressive richness, cognition–emotion interplay, self-reflectiveness, and narrative coherence. Table 11 shows the detailed explanations of these six metrics.

### E Unified Format of Data

We present data samples for each task in Figures 20 to 25. Emotion Detection requires the model to identify segments that carry distinct emotional expressions. In the Emotion Classification task, the model analyzes the subject’s emotional state based on the given context. In Emotion QA, the model answers questions grounded in contextual information. The Emotion Conversation task places the model in the role of a psychological counselor, responding to the client’s previous turn. Emotion Summary challenges the model to generate a struc-

###### Stage Metric Name Description

Establish initial trust through empathy and a nonjudgmental attitude, providing a safe foundation for further interventions.

Establishing the Therapeutic Alliance

Reception & Inquiry

Emotional Acceptance and Exploration Guidance

Guide the client to express emotions (e.g., anxiety, helplessness) in a safe atmosphere, demonstrating acceptance.

Integrate cognitive, behavioral, emotional, relational, and existential factors into a multidimensional assessment.

Systematic Assessment

Recognizing Surface-Level Reaction Patterns

Identify the client’s automatic cognitive, emotional, and behavioral responses.

Diagnostic

Reveal unmet psychological needs such as security, autonomy, connection, or meaning.

Deep Needs Exploration

Understanding the interaction of problems within the individual’s internal systems and external systems; integrating findings from various dimensions to present a panoramic view of how the problem is maintained.

Pattern Interconnection Analysis

By examining the truthfulness and constructiveness of thoughts, build a more adaptive cognitive framework.

Adaptive Cognitive Restructuring

Consultation

Emotional Acceptance and Transformation

Developing Emotional Awareness, Acceptance, and Transformation Skills.

Value-Oriented Integration

Anchor change to the life dimension beyond symptoms.

Consolidating Change and Growth Narrative

Review therapeutic progress and reinforce positive change through a coherent personal narrative.

Consolidation & Ending

Meaning Integration and Future Guidance

Internalize therapy gains into a life philosophy and create a value-driven future plan.

Autonomy and Resource Internalization

Strengthen the client’s internal coping resources and ability to continue growth independently.

- Table 10: Design of Emotion Conversation evaluation metrics.

Metric Description

Consistency Between Emotional Ratings and Generated Text

Evaluate whether the emotional ratings from the scale align with the content in the model’s self-description. Are the emotions rated in the scale accurately reflected in the model’s self-description? Also, assess whether the intensity of the ratings matches the emotional expression in the generated text.

Repetition of Content Check if there is noticeable repetition in the generated text, especially in the emotional descriptions. Are there repeated emotional, thought, or behavioral descriptions that make the text feel redundant or unnatural? Also, evaluate whether the generated text avoids repeating the same emotional descriptions and provides a multi-dimensional analysis.

Richness and Depth of Content Assess whether the generated text thoroughly explores the different dimensions of emotions (e.g., psychological, physical, and behavioral responses). Examine whether it delves into the origins, progression, and impact of the emotions, and whether it uses sufficient detail and examples to enrich emotional expression.

Interaction Between Emotion and Cognition

Determine whether the generated text effectively showcases the interaction between emotions and cognition. For example, does it demonstrate how the protagonist adjusts emotional reactions based on thoughts and situation evaluations? Also, check whether the emotions and behaviors in the text are consistent.

Emotional Reflection and Selfawareness

Evaluate whether the protagonist reflects on their emotional reactions. Does the text explore personal growth, self-awareness, or suggest strategies for emotional improvement?

Overall Quality and Flow of the Text

Assess whether the generated text flows smoothly and has a clear structure. Is there a natural progression from emotional reaction to evolution and reflection? Also, does the text use varied sentence structures and expressions to avoid monotony?

- Table 11: Design of Emotion Expression evaluation metrics.

tured summary of a counseling session, including the cause, symptoms, treatment process, illness characteristics, and treatment effect. Finally, in the Emotion Expression task, the model is immersed in an emotional situation, responds to the PANAS scale, and articulates its emotional state.

### F Comprehensive Prompt Collections

This section presents the complete set of prompts used throughout the framework, encompassing Evaluation, Multi-agent Enrichment, and Emotional Ensemble Generation stages across all tasks. For tasks adopting automatic evaluation as the metric, we utilize GPT-4o as the evaluation model, with detailed evaluation prompts illustrated in Figures 26 to 31. During the Multi-Agent Enrichment stage, task-specific prompts are designed to guide agent collaboration and reasoning, as shown in Figures 32 to 37. Finally, in the Emotional Ensemble Generation stage, we employ carefully constructed prompts to support emotional diversity and coherence in response generation, with the full set depicted in Figures 38 to 43.

[Figure 22]

###### Figure 19: Dataset generation prompt for Emotion Conversation.

[Figure 23]

###### Figure 20: Emotion Detection dataset example.

[Figure 24]

###### Figure 21: Emotion Classification dataset example.

[Figure 25]

Figure 22: Emotion QA dataset example.

[Figure 26]

- Figure 23: Emotion Conversation dataset example.

[Figure 27]

###### Figure 24: Emotion Summary dataset example.

[Figure 28]

###### Figure 25: Emotion Expression dataset example.

[Figure 29]

###### Figure 26: Evaluation prompt for the first stage of Emotion Conversation.

[Figure 30]

###### Figure 27: Evaluation prompt for the second stage of Emotion Conversation.

[Figure 31]

###### Figure 28: Evaluation prompt for the third stage of Emotion Conversation.

[Figure 32]

###### Figure 29: Evaluation prompt for the fourth stage of Emotion Conversation.

[Figure 33]

###### Figure 30: Evaluation prompt for Emotion Summary.

[Figure 34]

###### Figure 31: Evaluation prompt for Emotion Expression.

[Figure 35]

###### Figure 32: Multi-agent enrichment prompt for Emotion Classification.

[Figure 36]

###### Figure 33: Multi-agent enrichment prompt for Emotion Detection.

[Figure 37]

###### Figure 34: Multi-agent enrichment prompt for Emotion Conversation.

[Figure 38]

Figure 35: Multi-agent enrichment prompt for Emotion QA.

[Figure 39]

- Figure 36: Multi-agent enrichment prompt for Emotion Summary.

[Figure 40]

- Figure 37: Multi-agent enrichment prompt for Emotion Expression.
- Figure 38: Emotional ensemble generation prompt for Emotion Classification.

[Figure 41]

[Figure 42]

- Figure 39: Emotional ensemble generation prompt for Emotion Detection.

[Figure 43]

- Figure 40: Emotional ensemble generation prompt for Emotion Conversation.
- Figure 41: Emotional ensemble generation prompt for Emotion QA.

[Figure 44]

[Figure 45]

###### Figure 42: Emotional ensemble generation prompt for Emotion Summary.

[Figure 46]

###### Figure 43: Emotional ensemble generation prompt for Emotion Expression. The prompt for the Emotion Expression task was originally structured in multiple stages; for better clarity and intuitive understanding, it has been consolidated into a single prompt.

