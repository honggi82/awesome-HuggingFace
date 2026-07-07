# arXiv:2411.07140v2[cs.CL]13Nov2024

## CHINESE SIMPLEQA: A CHINESE FACTUALITY EVALUATION FOR LARGE LANGUAGE MODELS

Yancheng He∗, Shilong Li∗, Jiaheng Liu∗†, Yingshui Tan, Weixun Wang, Hui Huang, Xingyuan Bu, Hangyu Guo, Chengwei Hu, Boren Zheng, Zhuoran Lin, Xuepeng Liu, Dekai Sun, Shirong Lin, Zhicheng Zheng, Xiaoyong Zhu, Wenbo Su, Bo Zheng

Taobao & Tmall Group of Alibaba

ABSTRACT

New LLM evaluation benchmarks are important to align with the rapid development of Large Language Models (LLMs). In this work, we present Chinese SimpleQA, the first comprehensive Chinese benchmark to evaluate the factuality ability of language models to answer short questions, and Chinese SimpleQA mainly has five properties (i.e., Chinese, Diverse, High-quality, Static, Easy-to-evaluate). Specifically, first, we focus on the Chinese language over 6 major topics with 99 diverse subtopics. Second, we conduct a comprehensive quality control process to achieve high-quality questions and answers, where the reference answers are static and cannot be changed over time. Third, following SimpleQA, the questions and answers are very short, and the grading process is easy-to-evaluate based on OpenAI API. Based on Chinese SimpleQA, we perform a comprehensive evaluation on the factuality abilities of existing LLMs. Finally, we hope that Chinese SimpleQA could guide the developers to better understand the Chinese factuality abilities of their models and facilitate the growth of foundation models.

1 INTRODUCTION

A significant challenge in AI development is to ensure language models generate factually accurate responses. Current frontier models sometimes produce false outputs or answers that are not substantiated by evidence. This is the problem known as “hallucinations”, which greatly hinders the extensive use of general AI technologies, such as large language models (LLMs). Besides, it is difficult to evaluate the factuality abilities of the existing LLMs. For example, LLMs usually generate lengthy responses containing numerous factual claims. Recently, to address the aforementioned evaluation problem, OpenAI has released the SimpleQA benchmark (Wei et al., 2024) with 4,326 concise and fact-seeking questions, which makes measuring factuality simple and reliable.

However, the SimpleQA benchmark primarily targets the English language, resulting in a limited understanding of LLMs’ capabilities in other languages. Moreover, inspired by several recent Chinese LLM benchmarks (e.g., C-Eval (Huang et al., 2023), CMMLU (Li et al., 2023b)), to evaluate the factuality abilities of LLMs in Chinese, we present the Chinese SimpleQA benchmark1, which consists of 3000 high-quality questions spanning 6 major topics, ranging from humanities to science and engineering, as shown in Figure 1. Specifically, the distinct main features of our proposed Chinese SimpleQA dataset are as follows:

- • Chinese: Our Chinese SimpleQA focuses on the Chinese language, which provides a comprehensive evaluation of the factuality abilities of existing LLMs in Chinese.
- • Diverse: Chinese SimpleQA covers 6 topics (i.e., “Chinese Culture”, “Humanities”, “Engineering, Technology, and Applied Sciences”, “Life, Art, and Culture”, “Society”, and “Natural Science”), and these topic includes 99 fine-grained subtopics in total, which demonstrates the diversity of our Chinese SimpleQA.
- • High-quality: We conduct a comprehensive and rigorous quality control process to ensure the quality and accuracy of our Chinese SimpleQA.

* First three authors contributed equally. † Corresponding Author: Jiaheng Liu. 1https://openstellarteam.github.io/ChineseSimpleQA/

2024/11/13 19:10 sunburst-drink (6).html

C h

C h

C h

C h

C

n e s e

n e s e

CC

- n e s e F
- o k ore

- m

F o

k ore a

- n d

Tra d t

- o n

C h

- n e s e F
- o k ore

n e

n

- n

T

e

ev

s

- o

C

e

M e

M e

B u d d h s

B u d d h s m

o

i

My

i

My

s

- d c n
- e

- d c n
- e

cs

cs C

T a o s m

T a o s m

ev

h o

m

F m F

m

m

e

o

- n

- m

at

o

- n

- n

- m

at

o

- n

M

T

H

lli

o

A

A

o

o g

o g

i

i

C

rti l

- G

m

- G

a

m

e

i

i

i

D

m D

m

l

t

r l

usiic

ulture

a n d

Tra ditio n

Festivals

stiv

ls

Cultural

Relics ultural

elics

C olectibles

C olle ctible s C r a fts

C r a fts

L if e s t y l e

C uisin e

C uisin e

T o u ris m

T o u ri s m

F a s h i o n

F a s h i o n

E n t e r t a i n m e n t

E n t e r t a i n m e n t T o o l s T o o l s

- H

u

- m

a

- n

e

- H

hinese

i

i

l

R e

rt

- d

E

n

- e

H

i

i

g

pera

r

Industry

Industry

ediia

a n d

n

E cono m y

E c o n o

U r b a n

U r b a n

a

P h

a

I

S tu die s

d

S t u die s

f

y

e

os

r

M

N atio n

N a ti o n

ti

E t h nicit y

E t h n i c it y

O r g a n i z a ti o n

O r g a n i z a ti o n

S o c. S tr u c t. a n d

S o c. S tr u c t. a n d

S e c u rit y

S e c u r i t y

Lif e, A rt, a n d

i

C h i n e s e C u l.

C h i n e s e C u l.

LL aa ww

E c o n.

E c o n.

G o v e r n m e n t

G o v e r n m e n t

AA rr tt ss

P o l i t i c s a n d L a w

L i n g u i s t i c s

L i n g u i s t i c s Literature Literature PPhhiilloossoopphhyy

P o l i t i c s

C ultu r e

P o l i t i c s

RReelliiggiioonn

Cultural Studies

Cultural Studies

TTrraaddiittiioonn

S Culture and Education

CultureCulture

HistoryHistory

e

EEdduuccaattiioonn

EEtthhnnoollooggyy Anthropology

- m

a

- n

C&L Sciences

Bioengineering

Bioengineering

Anthropology S o c i o l o g y

E n e r g y a n d E n v i r o n .

u

C h e m i c a l E n g i n e e r i n g

C h e m i c a l E n g i n e e r i n g

S o c i o l o g y

T r a n s p o r . a n d L o g i s .

E n e r g y S c i e n c e

D e m o g r a p h y

E n e r g y S c i e n c e

D e m o g r a p h y P s y c h o l o g y

S o c i a l S c i e n c e s

Natural Science

Natural Science

E n v i r o n m e n t a l S c i e n c e

E n v i r o n m e n t a l S c i e n c e

P s y c h o l o g y J o u r n a li s m

a n u. a n d In d u s.

a n u. a n d In d u s.

T r a n s p o r t a ti o n

J o u r n a li s m

T r a n s p o r t a t i o n

A e r o s p a c e

A e r o s p a c e

Mi lit a r y S c i e n c e

Mi lit a r y S c i e n c e

M inin g

C o m m u nic a ti o n S t u d ie s

M i n i n g

C o m m u n i c a ti o n S t u d i e s

.

.

f

f

P rintin g

P olitic al S cie n c e

P rin tin g

P o litic al S cie n c e

I

I

i

S cie n c e

LIS La w

S cie n c e ate rials

LIS

E a

E a

l

s

s

.

M etalurgy

.

etallurg y a t e rials

L a

e

e

e

e

c

c

h

h

c

c

l

l

Engineering

E n gineerin g

n

n

i

n

n

anage

e

S c

e

ana ge

S c

e

e

Science t

Education

c

C

c

c

c

S

S

y

e n c

e n c

&

S

S

i

ti cono

A

A

echnolo

ent

l

ent

fe

fe

&

&

es

ri

es

ri

ttrr ii

echanical

echanical

puter

r

L

L

ii

Ma

Ma

ics

i

C

i

i

i

CC

G e o

e

G o

r

G e

e

G e

e

unications

o

c

M e

gy

M e

gy

B o o gy

ny Z

B o o gy

ota ny B

M e M e

cy B

cy

m

m

n

m

m

u

u

i

l

l

ti

- o g

a

- p hy

e

- o g

a

- p hy

o

o

ota

a

a

o g

o g

- d c n
- e

- d c n
- e

m

m

e o

e o

u

u

c

i

o

o

tri

tri

i

c

S

c

o

o

ar

ar

Z

o o

o o

- m

a

o

- n

li

g

g

h

h

E

l

l

A

A

P

P

gy

gy

o

- m

o

S

- n
- o

CC

o

Figure 1: Overview of Chinese SimpleQA. “Chinese Cul.” and “ETAS” represent “Chinese Culture” and “Engineering, Technology, and Applied Sciences”, respectively.

- • Static: Following SimpleQA, to preserve the evergreen property of Chinese SimpleQA, all reference answers would not change over time.
- • Easy-to-evaluate: Following SimpleQA, as the questions and answers are very short, the grading procedure is fast to run via existing LLMs (e.g., OpenAI API).

Moreover, we perform a comprehensive evaluation and analysis of existing LLMs on Chinese SimpleQA, and several insightful findings are as follows:

- • Chinese SimpleQA is challenging. Only o1-preview and Doubao-pro-32k achieve the passing score (63.8% and 61.9% on the correct metric), and there is a long way to improve for many closed-source and open-source LLMs.
- • Larger models lead to better results. Based on the results of the Qwen2.5 series, InternLM series, Yi-1.5 series, etc, we observe that better performance is obtained when the model is larger.
- • Larger models are more calibrated. We observe that o1-preview is more calibrated than o1-mini, and GPT-4o is more calibrated than GPT-4o-mini.
- • RAG matters. When introducing the RAG strategy into existing LLMs, the performance gaps between different LLMs decrease a lot. For example, for GPT-4o and Qwen2.5-3B, the performance gap decreases from 42.4% to 9.3% within using RAG.
- • Alignment tax exists. Existing alignment or post-training strategies usually decrease the factuality of language models.
- • Rankings of SimpleQA and Chinese SimpleQA are different. The performance of several LLMs focusing on Chinese (Doubao-pro-32k, and GLM-4-Plus) is close to the highperformance o1-preview. In particular, in the “Chinese Culture” topic, these Chinese community LLMs are significantly better than GPT or o1 series models.

2

ﬁle:///Users/heyancheng/Downloads/sunburst-drink (6).html 1/1

###### Benchmark Data Size Language Data Source Domain Reasoning Metric

WebQA (Li et al., 2016) 42187 Chinese Real World Knowledge ✓ Accuracy MMLU (Hendrycks et al., 2021) 15,908 English Exams & Textbooks Knowledge ✓ Accuracy CMMLU (Li et al., 2023a) 11,528 Chinese Exams Knowledge ✓ Accuracy GSM8K (Cobbe et al., 2021) 8,792 English Human Writers Math ✓ Accuracy AlpacaEval (Li et al., 2023d) 805 English Alpaca Data General ✓ LLM-as-a-Judge MT-Bench (Zheng et al., 2023) 80 English Self-constructed General ✓ LLM-as-a-Judge Arena-Hard (Li et al., 2024) 500 English Human Writers General ✓ LLM-as-a-Judge C-Eval (Huang et al., 2023) 13,948 Chinese Exams Knowledge ✓ Accuracy SimpleQA (Wei et al., 2024) 4,326 English Human Writers Knowledge × LLM-as-a-Judge

Self-constructed &Human Writers

Chinese SimpleQA (Ours) 3000 Chinese

Knowledge × LLM-as-a-Judge

Table 1: Comparisons between our Chinese SimpleQA and other benchmarks.

|Wikipedia Contents<br><br>|Question-Answer Pair Criteria<br><br>[Figure 1]<br><br>1. Answers must be objective and unique.<br>2. Answers must not change over time.<br>3. Questions must be challenging.<br>4. Questions must be answerable as of 2023.<br>|
|---|
<br><br>[Figure 2]<br><br>Extraction & Filtering<br><br>Quality Control<br><br>[Figure 3]<br><br>• Independent Answering<br>• Criteria Rechecking<br>• Dual Annotation<br>• Third-Party Check<br>• Supply Multiple Sources …<br><br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>1<br><br>2 QA-pair Generation<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>3 LLM Verification<br><br>[Figure 15]<br><br>|[Figure 16]<br><br>[Figure 17]<br><br>Search Engines<br><br>[Figure 18]|
|---|
<br><br>RAG<br><br>4 Verification<br><br>[Figure 19]<br><br>6 Human Verification<br><br>[Figure 20]<br><br>|[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>Large Language Models|
|---|
<br><br>1 5 Difficulty Filtering<br><br>[Figure 25]<br><br>[Figure 26]<br><br>Chinese SimpleQA<br><br>[Figure 27]|
|---|

Figure 2: An overview of the data construction process of Chinese SimpleQA.

- 2 CHINESE SIMPLEQA

- 2.1 OVERVIEW

Figure 1 shows the category distribution of Chinese SimpleQA, which contains six primary topics: “Chinese Culture”, “Humanities”, “Engineering, Technology and Applied Sciences”, “Life, Art and Culture”, “Society”, and “Natural Science”. Each primary topic includes multiple secondary subtopics. In Table 1, we also compare Chinese SimpleQA with several mainstream LLMs evaluation benchmarks, which demonstrates that Chinese SimpleQA is the first benchmark to focus on the evaluation of the boundaries of Chinese knowledge in LLMs.

- 2.2 DATA COLLECTION

As shown in Figure 2, the data collection process for Chinese SimpleQA involves both automated construction and human verification. The automated phase includes: (1) extracting and filtering relevant knowledge content, (2) generating question-answer pairs automatically, (3) verifying these pairs using an LLM based on predefined criteria, (4) performing Retrieval-Augmented Generation (RAG) verification, and (5) filtering for difficulty level.

Specifically, first, we collect a large amount of knowledge-rich text content from various knowledge fields (e.g., Wikipedia), and we utilize a quality assessment model to filter out low-quality data. Then, we prompt the LLM to generate question-answer pairs using these high-quality knowledge contents. After that, to ensure the quality of Chinese SimpleQA, we use LLM to remove samples, which cannot meet the requirements of our predefined criteria. In this way, we can obtain a large set of initially filtered knowledge question-answer pairs. Meanwhile, to improve the quality of answers, we deploy external retrieval tools (i.e., search engines) to gather more diverse information, which guides the LLM to evaluate the factual correctness of answers based on the RAG system. Specifically, we apply LlamaIndex 2 as the retrieval method, with search results from Google and Bing as data sources. Details on the generation and validation can be found in Appendix A. In

2https://github.com/run-llama/llama_index

addition, we filter some simple samples to discover the knowledge boundaries of the LLMs and improve the difficulty of Chinese SimpleQA. Specifically, if a question could be correctly answered by all four powerful models3, it is considered as a simple question and will be discarded.

Notably, the construction of question-answer pairs is based on the following criteria:

Answers must be objective and unique. Questions should relate to factual knowledge about the objective world and remain uninfluenced by personal subjective views. For example, questions beginning with “What do you think about” or “How would you evaluate” are inappropriate. Furthermore, the answer to each question must be unique, precluding the possibility of multiple correct responses. For instance, the query “In what year did Zhu Qizhen ascend to the throne?” is inadequate because it has two possible answers: 1435 and 1457.

Answers must not change over time. Answers should consistently reflect timeless facts, unaffected by the time the question is posed. For example, “What is the atomic number of carbon?”, and the answer “6” remains unchanged. In contrast, questions regarding current affairs, such as “Who is the current president of a certain country?” are inappropriate, as their answers are subject to change.

Questions must be challenging. Questions should not be overly simplistic, and the designed queries need to thoroughly assess the model’s depth of knowledge.

Questions must be answerable as of 2023. Each question must be answerable by December 31, 2023, ensuring fair evaluation for models trained on data available post this date.

- 2.3 QUALITY CONTROL

Following automated data collection, we employ human verification to enhance dataset quality. Specifically, each question is independently assessed by two human annotators. Initially, annotators determine whether the question adheres to the aforementioned predefined criteria. If either annotator deems the question non-compliant, this sample is discarded. Subsequently, both annotators utilize search engines to retrieve pertinent information and formulate answers. During this stage, the annotators are supposed to use content from authoritative sources (e.g., Wikipedia5, Baidu Baike6), and each annotator must provide at least two supporting URLs. In cases where the annotators’ answers are inconsistent, a third annotator reviews the sample. The final annotation is determined by the third annotator, referencing the initial two assessments. Finally, the human annotation results are compared with responses generated by the LLM, retaining only question-answer pairs that are entirely consistent. This rigorous human verification process ensures that our dataset maintains high accuracy and meets established standards.

In the entire process of constructing and annotating Chinese SimpleQA, many low-quality questionanswer pairs are discarded. Specifically, 10,000 pairs are initially generated. After difficulty evaluation through testing with different models, roughly 6,310 pairs are retained, with about 37% of the easier data being discarded. Following this, another 2,840 samples are removed after rule-based validation and model-based RAG validation, which means that only about 35% of the original generated data remains. Finally, after a thorough and rigorous manual review, only about 3,000 samples are kept, which is approximately 30% of the original dataset.

- 2.4 DATASET STATISTICS

- Table 2 presents the statistics of Chinese SimpleQA. With a total of 3000 samples, the data distribution across the six primary topics in Chinese SimpleQA is relatively balanced, which can effectively assess the knowledge boundaries of LLMs in various fields. Furthermore, the length distribution of both questions and reference answers in this dataset is very short, which is characteristic of knowledge-based queries. Notably, evaluating models using Chinese SimpleQA requires minimal input and output tokens, resulting in very low evaluation computation and time costs.

3GPT-4o (OpenAI, 2023), Meta-Llama-3-70B-Instruct (Dubey et al., 2024), Qwen2.5-72B-Instruct (Team, 2024d), and GLM-4-Plus4.

- 5https://www.wikipedia.org/
- 6https://baike.baidu.com/

Table 2: Dataset statistics of Chinese SimpleQA.

Statistics Number Statistics Number #Problems 3000 Length Primary Topics Question Length

- - Chinese Culture 323 - maximum length 81

- - Humanities 623 - minimum length 8

- - Engineering, Technology 473 - avg length 23.6 and Applied Sciences Reference Answer Length

- - Life, Art and Culture 602 - maximum length 47

- - Society 450 - minimum length 1

- - Natural Science 529 - avg length 6.1

- 2.5 EVALUATION METRICS Following SimpleQA, we also adopt the following five evaluation metrics.

- • Correct (CO): The predicted answer fully includes the reference answer without introducing any contradictory elements.
- • Not attempted (NA): The reference answer is not fully given in the predicted answer, and there are no contradictory elements with the reference answer.
- • Incorrect (IN): The predicted answer contradicts the reference answer, even if the contradiction is solved.
- • Correct given attempted (CGA): The metric is the proportion of accurately answered questions among those attempted questions.
- • F-score: The metric represents the harmonic mean between correct and correct given attempted.

- 3 EXPERIMENTS

- 3.1 BASELINE MODELS

We evaluate 17 closed-source LLMs (i.e., o1-preview 7, Doubao-pro-32k8, GLM-4-Plus9, GPT-4o10, Qwen-Max (Team, 2024c), Gemini-1.5-pro (Team, 2024a), DeepSeek-V2.5 (DeepSeek-AI, 2024b), Claude-3.5-Sonnet 11, Yi-Large12, moonshot-v1-8k13, GPT-4-turbo (OpenAI, 2023), GPT-4 (OpenAI, 2023), Baichuan3-turbo14, o1-mini15, Doubao-lite-4k16, GPT-4o-mini17, GPT-3.5 (Brown et al., 2020), and 24 open-source LLMs (i.e., Qwen2.5 series (Team, 2024d), InternLM2.5 series (Team, 2024b), Yi-1.5 series (AI et al., 2024), LLaMA3 (AI@Meta, 2024) series, DeepSeek Series (DeepSeek-AI, 2024a), Baichuan2 series (Baichuan, 2023), Mistral series (Jiang et al., 2023), ChatGLM3 and GLM-4 (GLM et al., 2024; Du et al., 2022)).

- 3.2 MAIN RESULTS

As shown in Table 3, we provide the performance results of different LLMs on our Chinese SimpleQA. Specifically, following SimpleQA, we provide the overall results on 5 evaluation metrics.

- 7https://openai.com/index/introducing-openai-o1-preview/
- 8https://www.volcengine.com/product/doubao
- 9https://bigmodel.cn/dev/api/normal-model/glm-4
- 10https://openai.com/index/hello-gpt-4o/
- 11https://www.anthropic.com/news/claude-3-5-sonnet
- 12https://platform.lingyiwanwu.com/
- 13https://platform.moonshot.cn/
- 14https://platform.baichuan-ai.com/
- 15https://openai.com/o1/
- 16https://www.volcengine.com/product/doubao
- 17https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/

Overall results on 5 metrics F-score on 6 topics CO NA IN CGA F-score CC HU ETAS LAC SO NS

Models

Closed-Source Large Language Models

o1-preview 63.8 12.2 24.0 72.7 67.9 45.7 69.8 72.4 65.0 73.5 72.3 Doubao-pro-32k 61.9 10.3 27.8 69.1 65.3 61.8 69.3 69.0 56.1 64.2 70.4

GLM-4-Plus 58.7 7.4 33.9 63.4 60.9 56.5 64.1 64.9 50.7 66.6 62.8 GPT-4o 59.3 1.4 39.3 60.1 59.7 39.4 64.0 65.1 53.3 68.6 62.0

Qwen-Max 54.1 11.3 34.6 61.0 57.4 47.8 59.9 63.5 49.9 61.2 59.3 Gemini-1.5-pro 54.4 8.0 37.6 59.1 56.7 41.4 59.1 60.8 52.2 56.3 64.3 DeepSeek-V2.5 54.1 5.9 40.0 57.5 55.7 50.4 57.6 58.8 50.1 59.4 56.9

Claude-3.5-Sonnet 46.2 27.4 26.4 63.6 53.5 28.7 61.3 60.4 42.2 59.8 57.7

Yi-Large 47.3 16.4 36.3 56.6 51.5 41.1 56.5 55.1 41.7 57.6 53.8 moonshot-v1-8k 48.7 5.4 45.9 51.5 50.1 49.8 54.1 56.8 41.4 53.0 46.6

GPT-4-turbo 45.6 14.2 40.2 53.1 49.1 24.2 55.2 58.9 43.9 52.5 50.8

GPT-4 45.4 8.4 46.2 49.6 47.4 25.2 54.0 52.8 41.8 52.8 50.6 Baichuan3-turbo 45.2 9.0 45.8 49.6 47.3 32.3 52.5 54.0 35.4 54.6 50.9

o1-mini 39.5 20.6 39.9 49.7 44.1 21.3 49.2 55.9 33.8 48.8 46.8 Doubao-lite-4k 36.7 31.2 32.1 53.3 43.4 40.2 44.8 51.0 31.1 41.4 50.4

GPT-4o mini 37.6 0.9 61.5 37.9 37.8 19.0 42.4 46.4 31.0 42.2 39.8 GPT-3.5 29.7 2.9 67.4 30.6 30.1 13.3 35.8 35.2 25.6 32.7 31.7

Open-Source Large Language Models

Qwen2.5-72B 48.4 7.1 44.5 52.1 50.2 36.3 56.1 57.9 37.1 53.3 56.4 Qwen2.5-32B 38.8 11.1 50.1 43.6 41.1 33.7 45.8 48.7 27.3 44.7 44.9 Qwen2.5-14B 35.4 9.6 55.0 39.2 37.2 30.2 41.8 46.1 24.1 38.8 41.0

Qwen2.5-7B 26.6 9.9 63.5 29.5 27.9 20.1 32.7 33.8 18.0 28.6 32.0 Qwen2.5-3B 16.2 12.8 71.0 18.6 17.3 13.4 17.9 26.1 9.3 15.6 20.8

Qwen2.5-1.5B 11.1 14.6 74.3 13.1 12.0 11.0 11.3 18.7 6.7 12.2 12.9

GLM4-9B 25.9 12.5 61.6 29.6 27.6 28.8 32.1 32.0 17.6 28.9 27.8 ChatGLM3-6B 11.2 13.6 75.2 12.9 12.0 12.1 13.8 12.4 8.8 13.4 11.8

InternLM2.5-20B 31.5 7.7 60.8 34.1 32.8 32.0 37.1 37.7 21.2 35.7 34.3

InternLM2.5-7B 24.7 7.5 67.8 26.7 25.7 25.5 29.4 31.0 16.4 26.9 25.8 InternLM2.5-1.8B 5.3 31.1 63.6 7.6 6.2 6.1 8.7 7.2 3.3 4.5 7.4

Yi-1.5-34B 30.9 5.8 63.3 32.8 31.8 28.2 36.9 36.8 24.4 32.8 31.4 Yi-1.5-9B 18.2 2.9 78.9 18.7 18.4 17.2 20.2 24.3 10.2 20.1 19.8 Yi-1.5-6B 15.9 2.8 81.3 16.3 16.1 14.2 17.9 21.3 10.3 16.8 16.5

LLaMA3.1-70B 38.3 9.4 52.3 42.3 40.2 22.9 47.2 49.3 34.5 49.6 40.4 LLaMA3.1-8B 16.9 8.8 74.3 18.6 17.7 8.5 20.7 23.4 9.7 20.5 20.7 DeepSeek-67B 43.5 14.8 41.7 51.1 47.0 34.3 54.5 50.3 42.3 49.0 46.2

DeepSeek-V2-Lite-Chat 33.7 12.8 53.5 38.6 36.0 35.3 38.5 41.7 32.2 37.5 31.2

DeepSeek-7B 23.2 13.2 63.6 26.7 24.8 24.5 27.2 28.9 20.6 27.0 21.5 Baichuan2-13B 19.1 24.9 56.0 25.4 21.8 24.0 25.8 23.3 16.8 23.0 18.7

Baichuan2-7B 12.5 21.8 65.7 16.0 14.0 14.6 16.1 15.4 11.1 13.8 13.3 Mixtral-8x22B-Instruct-v0.1 27.3 2.2 70.5 27.9 27.6 10.6 32.3 36.0 21.0 34.1 26.9

Mixtral-8x7B-Instruct-v0.1 20.4 7.2 72.4 22.0 21.2 5.2 26.5 29.0 13.0 25.0 23.3 Mistral-7B-Instruct-v0.2 15.0 8.8 76.2 16.4 15.6 4.5 18.2 22.2 9.5 21.4 15.7

- Table 3: Results of different models on Chinese SimpleQA. For metrics, CO, NA, IN, and CGA denote “Correct”, “Not attempted”, “Incorrect”, and “Correct given attempted”, respectively. For subtopics, CC, HU, ETAS, LAC, SO and NS represent “Chinese Culture”, “Humanities”, “Engineering, Technology, and Applied Sciences”, “Life, Art, and Culture”, “Society”, and “Natural Science”, respectively.

Additionally, we also report the F-score for 6 topics to analyze the fine-grained factuality abilities of these LLMs. In Table 3, we have the following insightful and interesting observations:

- • o1-preview achieves the best performance on Chinese SimpleQA, and the performance results of several recent closed-source LLMs focusing on Chinese (Doubao-pro-32k and GLM-4-Plus) are very close to o1-preview.
- • It is obvious that the “mini” series models (o1-mini, GPT-4o-mini) achieve lower results than the corresponding larger models (o1-preview, GPT-4o), which also indicates these “mini” series models do not pay attention to memorize factuality knowledge.

###### CO of CC

###### CO of HU

###### CO of ETAS

###### CO of LAC

###### CO of SO

###### CO of NS

GLM-4-Plus

GLM-4-Plus

GLM-4-Plus

GLM-4-Plus

GLM-4-Plus

GLM-4-Plus

65.3

64.1

62.4

60.7

Baichuan3-Turbo

Baichuan3-Turbo

Baichuan3-Turbo

Baichuan3-Turbo

Baichuan3-Turbo

Baichuan3-Turbo

o1-preview

o1-preview

o1-preview

o1-preview

o1-preview

o1-preview

52.3

47.7

47.7

71.3

70.4

70.1

65.0

58.8

52.9

52.9

49.9

49.1

39.9

33.3 36.4

29.7 36.5

43.3

46.0

51.4

51.8

52.3

53.6

55.4

58.0

58.4

61.8

Yi-Large

Yi-Large

Yi-Large

Yi-Large

Yi-Large

Yi-Large

Qwen-Max

Qwen-Max

Qwen-Max

Qwen-Max

Qwen-Max

Qwen-Max

47.1 56.7

51.7 47.7

55.1 64.4

56.7 69.4

58.4 67.0

62.0 58.7

Doubao-pro-32k DeepSeek-V2.5

Doubao-pro-32k DeepSeek-V2.5

Doubao-pro-32k DeepSeek-V2.5

Doubao-pro-32k DeepSeek-V2.5

Doubao-pro-32k DeepSeek-V2.5

Doubao-pro-32k DeepSeek-V2.5

###### CGA of CC

###### CGA of HU

###### CGA of ETAS

###### CGA of LAC

###### CGA of SO

###### CGA of NS

GLM-4-Plus

GLM-4-Plus

GLM-4-Plus

GLM-4-Plus

GLM-4-Plus

GLM-4-Plus

67.9

67.9

65.7

63.2

61.5

Baichuan3-Turbo

Baichuan3-Turbo

Baichuan3-Turbo

Baichuan3-Turbo

Baichuan3-Turbo

Baichuan3-Turbo

o1-preview

o1-preview

o1-preview

o1-preview

o1-preview

o1-preview

54.0

75.9

75.4

74.6

74.4

72.6

56.5

56.5

55.1

53.5

51.9

37.7

35.3

47.0

48.9

53.2

54.5

56.0

58.3

60.7

62.3

62.9

64.1

65.2

65.3

Yi-Large

Yi-Large

Yi-Large

Yi-Large

Yi-Large

Yi-Large

Qwen-Max

Qwen-Max

Qwen-Max

Qwen-Max

Qwen-Max

Qwen-Max

52.7 61.3

54.3 68.0

57.1 71.4

59.3 71.0

66.6 60.1

60.5 75.0

Doubao-pro-32k DeepSeek-V2.5

Doubao-pro-32k DeepSeek-V2.5

Doubao-pro-32k DeepSeek-V2.5

Doubao-pro-32k DeepSeek-V2.5

Doubao-pro-32k DeepSeek-V2.5

Doubao-pro-32k DeepSeek-V2.5

Figure 3: Results (CO and CGA metrics) of different models for six topics.

[Figure 28]

Figure 4: Left: Calibration of LLMs based on their stated confidence. Right: Improvement in accuracy with increased test-time compute using Best-of-N.

- • A Larger LLM leads to better performance, where we can draw this conclusion based on many model series (e.g., GPT, Qwen2.5, InternLM2.5, Yi-1.5)
- • Small LLMs usually lead to higher scores on “not attempted (NA)”. The NA scores for o1mini, InternLM2.5-1.8B are 20.5 and 31.2, respectively, which are larger than the scores of corresponding larger LLMs a lot (o1-preview with 12.2, InternLM2.5-20B with 7.7).
- • There is a significant performance difference among different subtopics for different LLMs. Notably, the Chinese community LLMs (e.g., Doubao-pro-32k, GLM-4-Plus, Qwen-Max, Deepseek) are significantly better than the GPT or o1 models in the Chinese Culture (CC) subtopic. In contrast, the o1 has significant advantages in science-related subtopics (e.g., Engineering, Technology, and Applied Sciences (ETAS), and Natural Science (NS)).

In addition, we also provide the detailed results (CO and CGA metrics) on 6 topics in Figure 3.

- 3.3 FURTHER ANALYSIS

- 3.3.1 ANALYSIS OF CALIBRATION

For the calibration of different LLMs, following SimpleQA, we instruct the model to provide a corresponding confidence level (from 0 to 100) when answering questions to measure the model’s confidence in its answers (See the prompt in Appendix B). We know that a perfectly calibrated

70

Pretrain Model Aligned Model

56.7

60

| |
|---|

50.2

48.2

47.0

50

F-score

40

36.1

35.1

32.8

31.8

30.3

30

26.3

25.7

25.4

25.3

25.1

25.0

21.8 19.3

18.4

17.3

20

14.0

10

0

Qwen2.5-3B Qwen2.5-72B Yi-1.5-9B Yi-1.5-34B Deepseek-7B Deepseek-67BInternLM2.5-7BInternLM2.5-20B Baichuan2-7BBaichuan2-13B

Figure 6: The effect of alignment in post-training.

model’s confidence (%) should match its answers’ actual accuracy. The left plot in Figure 4 illustrates the alignment performance, which indicates that GPT-4o aligns better than GPT-4o-mini and

- o1-preview aligns better than o1-mini. For the Qwen2.5 series, the alignment order is Qwen2.5-72B > Qwen2.5-32B > Qwen2.5-7B > Qwen2.5-3B, which suggests that larger model sizes result in better calibration. Furthermore, for all evaluated models, their confidence in the range of confidence > 50 falls below the line of perfect alignment, which means that they all overestimate the accuracy
- of their responses and overconfidence exists.

- 3.3.2 ANALYSIS OF TEST-TIME COMPUTE

We also evaluate the relationship between increased test-time compute and response accuracy for different models. Specifically, we randomly sample 50 samples from Chinese SimpleQA, and for each sample, the model is asked to independently answer 100 times. Then, we obtain the model’s response accuracy using the Best-of-N method as the inference counts increase. The results are shown in the right plot of Figure 4. We observe that as the times of inferences increase, the response accuracy of all models improves and eventually reaches a ceiling. This is reasonable for Chinese SimpleQA, which is specifically designed to probe the boundaries of a model’s knowledge.

- 3.3.3 ANALYSIS ON THE EFFECT OF RAG

In this study, we explore the effectiveness of the Retrieval-Augmented Generation (RAG) strategy in enhancing the factual accuracy of large language models (LLMs) on the Chinese SimpleQA dataset. Specifically, we reproduce a RAG system based on LlamaIndex (Liu, 2022), incorporating Google search APIs. As illustrated in Figure 5, all models demonstrate a substantial improvement in accuracy with RAG. For example, the performance of Qwen2.5-3B improved more than threefold. Notably, nearly all models with RAG outperform the native GPT4-o model. Meanwhile, the application of RAG also leads to a marked reduction in performance disparities among models. For example, the F-score difference between the Qwen2.5-3B with RAG and the Qwen2.5-72B with RAG is only 6.9%. This suggests that RAG reduces the performance gaps on models greatly, enabling even smaller ones to achieve high performance when augmented with RAG. Overall, this suggests that RAG serves as an effective shortcut for enhancing the factuality of LLMs.

Performance Comparison of Models with and without RAG

100

1.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Baseline with RAG

| |
|---|

81.8

79.3 79.4 81.2

79.3

80

0.8

75.7

72.5

59.7

60

0.6

57.4

F-score

50.2

41.1

37.8

40

0.4

27.9

20

0.2

17.3

0

0.0

0.0 0.2 0.4 0.6 0.8 1.0

GPT-4o GPT-4ominiQwen2.5-3BQwen2.5-7BQwen2.5-32BQwen2.5-72B Qwen-Max

Figure 5: The effect of RAG strategy.

###### o1-preview

###### Doubao-pro-32k

###### GLM-4-Plus

###### moonshot-v1-8K

Education

Education

Education

Education

Entertainment

Entertainment

Entertainment

Entertainment

82.10

73.70 92.10

71.20 78.10

66.70

Mathematics

Mathematics

Mathematics

Mathematics

59.00

52.50

51.40

87.50

59.40

30.40

84.20

72.40

73.30

56.70

Medicine

Medicine

Medicine

Medicine

53.60

71.40

78.60

44.70

82.10

54.10

63.20

Law

Law

Law

Law

75.70

66.10

68.90

71.80

71.80

Computer Science

Computer Science

Computer Science

Computer Science

Economics

Economics

Economics

Economics

###### Qwen-Max

###### DeepSeek-V2.5

###### Yi-Large

###### Baichuan3-Turbo

Education

Education

Education

Education

Entertainment

Entertainment

Entertainment

Entertainment

85.70

73.70

69.00 75.00

66.70 81.30

Mathematics

Mathematics

Mathematics

Mathematics

51.30

81.30

79.40

42.10

40.00

31.60

69.00

60.00

55.20

67.80

Medicine

Medicine

Medicine

Medicine

67.90

69.10

73.10

75.00

44.70

48.70

52.10

59.50

Law

Law

Law

Law

59.50

64.40

65.60

67.80

Computer Science

Computer Science

Computer Science

Computer Science

Economics

Economics

Economics

Economics

Figure 7: Detailed results on some selected subtopics.

- 3.3.4 ANALYSIS ON THE ALIGNMENT TAX

Recently, prior studies (OpenAI, 2023; Song et al., 2023) have found that the alignment can lead to a decrease in the abilities of language models as known as the “alignment tax”. To illustrate the effect of alignment on factuality, we conduct a comparative performance analysis between pre-trained models and aligned models that are trained with Supervised Fine-Tuning (SFT) or Reinforcement Learning from Human Feedback (RLHF). As illustrated in Figure 6, different models exhibit varying trends after post-training, but most models have a significant decline. Among these, the Baichuan2 series models show the most significant decreases, with Baichuan2-7B and Baichuan2-13B experiencing F-score reductions of 47% and 28%, respectively. This reflects that the alignment training of most current LLMs still has obvious drawbacks to produce knowledge hallucinations, which further reflects the necessity of our dataset.

- 3.3.5 ANALYSIS ON THE RESULTS OF SUBTOPICS

As mentioned in Section 2.2, the benchmark covers a total of 99 subtopics, which can comprehensively detect the knowledge level of the model in various domains. Figure 7 illustrates the performance comparison between the o1 model and seven notable Chinese community models within several common domains. Firstly, from an overall perspective, the o1-preview model exhibits the most comprehensive performance across these domains, with the Doubao model following closely. In contrast, the Moonshot model demonstrates the weakest overall performance. Secondly, when examining specific domains, a significant disparity emerges between the Chinese community models and the o1 model in areas such as Computer Science and Medicine. However, this gap is minimal in domains like Education and Economics. Notably, in Education, some Chinese community models outperform the o1-preview, highlighting their potential for achieving success in specific vertical domains. Lastly, when examining specific models, the Moonshot model is notably weaker in Mathematics, Law, and Entertainment, while the Baichuan model also underperforms in Entertainment. The Yi-Large model excels in Education, and the o1 model maintains the strongest performance across other domains. Evaluating the performance of the models across diverse domains within the benchmark dataset enables users to identify the most suitable model for their specific needs.

- 3.3.6 COMPARISON BETWEEN CHINESE SIMPLEQA AND SIMPLEQA

We also compare the ranking differences of various models on the SimpleQA and the Chinese SimpleQA. As illustrated in Figure 8, there are notable discrepancies in model performance across these two benchmarks. For instance, Doubao-pro-32k ranks significantly higher on the Chinese SimpleQA, moving from 12th to 2nd place (+10). Conversely, GPT-4 shows a decline in performance

Comparison of Model Rankings on the SimpleQA and Chinese SimpleQA Datasets Ranking on the SimpleQA Ranking on the Chinese SimpleQA

20

| |
|---|

15

12

12

11

11

10

10

Rank

10

9

9

8

8

7

7

6

6

5

5

+10

+1 -2 -4

5

4

4

+3

-3 -6

3

3

+2 -2

2

2

+3

1

1

-2

+0

0

o1-previewDoubao-pro-32kGLM-4-PlusGPT-4oQwen-MaxGemini-1.5-proYi-LargeGPT-4-turboGPT-4o1-miniGPT-4ominiGPT-3.5

Figure 8: The rankings of different LLMs on SimpleQA and Chinese SimpleQA.

on the Chinese SimpleQA, dropping from 3rd to 9th place (-6). These differences emphasize the importance of evaluating models on datasets in various languages and the need for research into optimizing model performance across different linguistic environments. Notably, the o1-preview maintains its top position consistently across both datasets, indicating its robustness and adaptability to different linguistic contexts. In addition, most Chinese community-developed models (e.g., Qwen-Max, GLM-4-Plus, Yi-Large, Doubao-pro-32k) perform better on the Chinese SimpleQA than on the SimpleQA, demonstrating their competitive results on Chinese-language tasks.

- 4 RELATED WORKS

LLM Factuality. LLM factuality is the capability of large language models to produce contents that follow factual content, including commonsense, world knowledge, and domain facts, and the factual content can be substantiated by authoritative sources (e.g., Wikipedia, textbooks). Recent works have explored the potential of LLMs to serve as factual knowledge bases (Yu et al., 2023; Pan et al., 2023). Specifically, existing studies have primarily focused on qualitative assessments of LLM factuality (Lin et al., 2022; Chern et al., 2023), investigations into knowledge storage mechanisms (Meng et al., 2022; Chen et al., 2023), and analyses on knowledge-related issues (Gou et al., 2023).

Factuality Benchmarks. Many factuality benchmarks (Hendrycks et al., 2021; Zhong et al., 2023; Huang et al., 2023; Li et al., 2023b; Srivastava et al., 2023; Yang et al., 2018) have been proposed. For example, MMLU Hendrycks et al. (2021) is to measure the multitask accuracies on a diverse set of 57 tasks. TruthfulQA (Lin et al., 2022) focuses on assessing the truthfulness of a language model’s generated answers. Additionally, HaluEval (Li et al., 2023c) is to examine the tendency of LLMs to produce hallucinations. Recently, SimpleQA (Wei et al., 2024) has been proposed to measure the short-form factuality in LLMs. However, SimpleQA only focuses on the English domain. In contrast, our Chinese SimpleQA aims to evaluate factuality in Chinese comprehensively.

- 5 CONCLUSION

In this paper, to evaluate the factuality abilities of existing LLMs, we propose the first Chinese short-form factuality benchmark (i.e., Chinese SimpleQA), which mainly has five important features (i.e., Chinese, diverse, high-quality, static, and easy-to-evaluate). Based on Chinese SimpleQA, we comprehensively evaluate the performance of existing 40+ LLMs on factuality and provide detailed analysis to demonstrate the advantage and necessity of our Chinese SimpleQA. In the future, we will investigate how to improve the LLMs’ factuality and explore how to extend Chinese SimpleQA to multilingual, multimodal, and domain-specific (e.g., code, e-commerce) settings.

REFERENCES

01. AI, :, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. Yi: Open foundation models by 01.ai, 2024. URL https://arxiv.org/ abs/2403.04652.

AI@Meta. Llama 3 model card. 2024. URL https://github.com/meta-llama/ llama3/blob/main/MODEL_CARD.md.

Baichuan. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023. URL https://arxiv.org/abs/2309.10305.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Yuheng Chen, Pengfei Cao, Yubo Chen, Kang Liu, and Jun Zhao. Journey to the center of the knowledge neurons: Discoveries of language-independent knowledge neurons and degenerate knowledge neurons, 2023.

I-Chun Chern, Steffi Chern, Shiqi Chen, Weizhe Yuan, Kehua Feng, Chunting Zhou, Junxian He, Graham Neubig, and Pengfei Liu. Factool: Factuality detection in generative ai – a tool augmented framework for multi-task and multi-domain scenarios, 2023.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv: Arxiv2110.14168, 2021.

DeepSeek-AI. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024a. URL https://github.com/deepseek-ai/ DeepSeek-LLM.

DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024b.

Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 320–335, 2022.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Laurens van der

Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Olivier Duchenne, Onur C¸elebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, Danny Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzm´an, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Igor Molybog, Igor Tufanov, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Karthik Prasad, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev, Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lind-

say, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and Zhiwei Zhao. The llama 3 herd of models. arXiv preprint arXiv: 2407.21783, 2024.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and Zihan Wang. Chatglm: A family of large language models from glm-130b to glm-4 all tools, 2024.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing, 2023.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. arXiv preprint arXiv:2305.08322, 2023.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Tim Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese. ArXiv, abs/2306.09212, 2023a. URL https://api.semanticscholar.org/CorpusID: 259164635.

Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese, 2023b.

Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. Halueval: A largescale hallucination evaluation benchmark for large language models, 2023c.

Peng Li, Wei Li, Zhengyan He, Xuguang Wang, Ying Cao, Jie Zhou, and Wei Xu. Dataset and neural recurrent sequence labeling model for open-domain factoid question answering. arXiv preprint arXiv:1607.06275, 2016.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. ArXiv, abs/2406.11939, 2024. URL https://api.semanticscholar. org/CorpusID:270562889.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca_eval, 2023d.

Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3214–3252, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.229. URL https: //aclanthology.org/2022.acl-long.229.

Jerry Liu. LlamaIndex, 11 2022. URL https://github.com/jerryjliu/llama_index. Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual

associations in GPT. Advances in Neural Information Processing Systems, 36, 2022. OpenAI. Gpt-4 technical report. PREPRINT, 2023. Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large

language models and knowledge graphs: A roadmap, 2023. Ziang Song, Tianle Cai, Jason D Lee, and Weijie J Su. Reward collapse in aligning large language models. arXiv preprint arXiv:2305.17608, 2023.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adri`a Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/ forum?id=uyTL5Bvosj.

Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of con-

text, 2024a. URL https://arxiv.org/abs/2403.05530. InternLM2 Team. Internlm2 technical report, 2024b. Qwen Team. Introducing qwen1.5, February 2024c. URL https://qwenlm.github.io/

blog/qwen1.5/. Qwen Team. Qwen2.5: A party of foundation models, September 2024d. URL https: //qwenlm.github.io/blog/qwen2.5/.

Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, and William Fedus. Measuring short-form factuality in large language models.

2024. URL https://api.semanticscholar.org/CorpusID:273877483.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering, 2018.

Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang. Generate rather than retrieve: Large language models are strong context generators. In International Conference for Learning Representation (ICLR), 2023.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv: 2306.05685, 2023.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models, 2023.

### A GENERATION AND VALIDATION OF QUESTION-ANSWER PAIRS

The generation and validation of question-answer pairs both use OpenAI’s gpt-4o-0806. The specific prompts are shown in Figures 9, 10, and 11.

现在需要你根据给定的文档生成一个事实类问题和对应的标准答案，需要满足下列 要求:

- 1. 生成的问题必须关联到客观世界的知识，例如可以询问“2024年诺贝尔物理学 奖的获得者是谁？”不得构造涉及个人观点或感受相关的主观问题，如“你如何看 待xxx？”。
- 2. 所提出的问题应该有且只有一个明确且无争议的实体作为答案，且问题表述中 不应存在任何形式的模糊性或歧义。例如，避免提问“巴拉克和米歇尔·奥巴马在哪 里会面？”因为无法确定是指哪一次会面；同样不要问“白民国人身体的特点是什 么？”因为这个问题过于模糊，没有明确的答案。“周汝昌最为人熟知的著作是哪 个？”也是不合格问题，因为“最熟知”可能是有争议的。
- 3. 问题的答案应当是时间不变的，不会随着时间的推移而改变。例如，“美国现任总 统是谁？”就不是一个合适的问题，因为总统身份会随选举结果改变。
- 4. 问题应该具有一定的难度，以体现出一定的挑战性。例如：电影《脱衣舞娘》是 由同名小说改编的，该小说的作者是谁？
- 5. 如果问题的答案为英文人名，请给出中文翻译后的名字和括号里带上英文原名， 格式如：雅各布·福格（Jakob Fugger）。
- 6. 生成的问题需要与给定的类目相关 请将生成的问题和答案以JSON格式返回，具体格式如下： {”question”: ”这里填写生成的问题”, ”answer”: ”这里填写对应的标准答案”} ###以下是一些示例### ### 示例一 类目：娱乐 文档内容：2022年国际足联世界杯为第22届国际足联世界杯，于2022年11月20日 至12月18日在卡塔尔举行[2][3]，成为全球爆发防疫后首个终结限制的大型国际体育 盛事。考量到气候因素，本届世界杯亦是首次于11月至12月北半球秋季[注 1]举行 之世界杯。决赛于卡塔尔的卢赛尔体育场举行，由阿根廷队对阵卫冕冠军法国队。 双方先于比赛正规时间踢至加时赛以3–3赛和，后在点球大战中阿根廷以4–2击败法 国，赢得了此届世界杯，也是阿根廷继1986年世界杯后，相隔36年再度于世界杯夺 冠，继巴西，意大利及德国后第四支三次冠军的球队，也成为继巴西后第二支在亚 洲夺冠的南美洲球队。克罗地亚则以2比1击败该年黑马摩洛哥赢得季军[4][5]。 返回结果：{”question”: ”2022年世界杯决赛点球大战中阿根廷队是以多少击败法国 队”, ”answer”: ”4–2”} ### 示例二 类目：政治 文档内容：中国与世界贸易组织（英语：China and the World Trade Organization） 指中华人民共和国与世界贸易组织的关系。在部长级会议达成协议后，中国 于2001年12月11日成为世界贸易组织成员。[1][2]在承认这一点之前，双方进行了 漫长的谈判，并且需要对中国经济进行重大改革。世贸组织的成员资格一直存在争 议，对其它国家产生了重大的经济和政治影响(也被称为“中国冲击”) ，对世贸组织 框架与中国经济模式之间的不匹配也存在争议。[3][4]评估和执行合规已成为中美贸 易关系中的问题，包括中国的不合规行为如何为本国经济创造利益。[5][6] 返回结果：{”question”: ”中国是哪一年正式成为世界贸易组织成员？”, ”answer”: ”2001”} ### 让我们开始吧！

Figure 9: The prompt for generating question-answer pairs.

### B ANALYSIS OF MODEL CALIBRATION

你是一个数据质量检查员，现在需要你检查下面生成的问题是否满足以下要求：

- 1.生成的问题必须对客观世界的知识的提问，例如可以询问“2024年诺贝尔物理学 奖的获得者是谁？”不得构造涉及个人观点或感受相关的主观问题，如“你如何看 待xxx？”。
- 2. 问题应该有且只有一个明确且无争议的实体作为答案，且问题表述中不应存在任 何形式的模糊性或歧义。例如，避免提问“巴拉克和米歇尔·奥巴马在哪里会面？”因 为无法确定是指哪一次会面；同样不要问“白民国人身体的特点是什么？”因为这 个问题过于模糊，没有明确的答案。注意如果回答是多个实体也不满足要求，例 如：“软体动物、腕足动物及被囊动物”
- 3. 问题的答案应当是时间不变的，不会随着时间的推移而改变。例如，“美国现任总 统是谁？”就不是一个合适的问题，因为总统身份会随选举结果改变。 如果问题不合格则解释并输出“【否】”， 如果问题合格则直接输出“【是】” #### 以下是一些示例： 问题：《黄帝内经》中, 援物比类思维方式包括哪些核心概念? 评价：该问题不是只有一个确切答案，【否】

问题：建筑理论主要研究什么内容？ 评价：该问题不具体，回答不是只有一个实体，【否】

问题：成立了程派高氏八卦掌的高义盛原籍是哪里？ 评价：回答范围不明确，不清楚是回答到城市还是省份，【否】

问题：自由恋爱主义最初的目标是将哪些事务与国家分离？ 评价：该问题不是只有一个答案，【否】

问题：汉十高速公路连接的武汉市和哪两个城市？ 评价：【是】 #### 如果问题不合格则输出原因并最后输出“【否】”， 如果问题合格则直接输 出“【是】”，注意如果认为问题不合格需要输出原因 让我们开始吧！

Figure 10: The prompt for validating criteria.

请结合检索材料判断给定的问题是否满足以下要求以及答案是否正确：

- 1. 问题应该有且只有一个明确且无争议的实体作为答案，且问题表述中不应存在任 何形式的模糊性或歧义。例如，避免提问“巴拉克和米歇尔·奥巴马在哪里会面？”因 为无法确定是指哪一次会面；同样不要问“白民国人身体的特点是什么？”因为这 个问题过于模糊，没有明确的答案。注意如果回答是多个实体也不满足要求，例 如：“软体动物、腕足动物及被囊动物”
- 2. 问题的答案应当是时间不变的，不会随着时间的推移而改变。例如，“美国现任总 统是谁？”就不是一个合适的问题，因为总统身份会随选举结果改变。
- 3. 检索材料应该能够验证给定答案的正确性，并且这些材料中不应存在其他与之相 矛盾的答案。如果检索材料中存在不同的答案，则该问题被视为无法验证或答案不 明确，不满足要求。 请依据上述标准，审查并确保提出的问题及其答案符合要求，如果不满足上面任一 要求，则输出原因，最后输出“【不合格】”， 否则输出检索材料对应的片段，最后 输出“【合格】”

Figure 11: The prompt for verifying factual accuracy based on RAG.

请阅读以下问题： {question} 请基于此问题提供你的最佳答案，并用0到100的分数表示你对该答案的信心（置信 度）。请以如下的JSON格式给出回复： {

”answer”: ”你的答案”, ”confidence score”: 你的置信度

#### }

Figure 12: The prompt for guiding the model to output confidence.

