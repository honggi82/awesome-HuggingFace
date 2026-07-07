# arXiv:2412.12606v1[cs.AI]17Dec2024

MULTI-DIMENSIONAL INSIGHTS: BENCHMARKING REAL-WORLD PERSONALIZATION IN LARGE MULTIMODAL MODELS

YiFan Zhang1∗, Shanglin Lei2∗, Runqi Qiao1∗, Zhuoma GongQue1, Xiaoshuai Song1, Guanting Dong1, Qiuna Tan1, Zhe Wei1 Peiqing Yang1, Ye Tian1, Yadong Xue1, Xiaofei Wang1, Honggang Zhang1† 1Beijing University of Posts and Telecommunications 2Huazhong University of Science and Technology

ABSTRACT

The rapidly developing field of large multimodal models (LMMs) has led to the emergence of diverse models with remarkable capabilities. However, existing benchmarks fail to comprehensively, objectively and accurately evaluate whether LMMs align with the diverse needs of humans in real-world scenarios. To bridge this gap, we propose the Multi-Dimensional Insights (MDI) benchmark, which includes over 500 images covering six common scenarios of human life. Notably, the MDI-Benchmark offers two significant advantages over existing evaluations: (1) Each image is accompanied by two types of questions: simple questions to assess the model’s understanding of the image, and complex questions to evaluate the model’s ability to analyze and reason beyond basic content. (2) Recognizing that people of different age groups have varying needs and perspectives when faced with the same scenario, our benchmark stratifies questions into three age categories: young people, middle-aged people, and older people. This design allows for a detailed assessment of LMMs’ capabilities in meeting the preferences and needs of different age groups. With MDI-Benchmark, the strong model like GPT4o achieve 79% accuracy on age-related tasks, indicating that existing LMMs still have considerable room for improvement in addressing real-world applications. Looking ahead, we anticipate that the MDI-Benchmark will open new pathways for aligning real-world personalization in LMMs. The MDI-Benchmark data and evaluation code are available at https://mdi-benchmark.github.io/

1 INTRODUCTION

Developing personalized artificial intelligence (AI) assistants to address the diverse needs of different users has long been a significant pursuit for humanity (Kobsa & Schreck, 2003; Xiao et al.,

- 2018; Kocaballi et al., 2019; Rafieian & Yoganarasimhan, 2023; Pesovski et al., 2024). In realworld scenarios, an ideal AI assistant should be capable of precisely meeting the specific demands of individuals across various age groups, cultural backgrounds, and professional fields.

Recently, the field of artificial intelligence has undergone a significant paradigm shift, transitioning from specialized small models designed for specific simple tasks (Rawat & Wang, 2017; Zhao et al.,

- 2019; Minaee et al., 2021; Singh et al., 2017) to unified large multimodal models (LMMs) capable of handling complex tasks (Zhang et al., 2024). This paradigm shift marks a crucial step toward achieving Artificial General Intelligence (AGI) and underscores the potential for LMMs to become personalized human assistants.

To comprehensively evaluate the capabilities of LMMs, researchers have constructed several common visual question-answering benchmarks (Goyal et al., 2017; Chen et al., 2015; Marino et al., 2019; Mishra et al., 2019; Biten et al., 2019) that assess general image-text comprehension and

∗The first three authors contribute equally. †Corresponding author

[Figure 1]

- Figure 1: The MDI-Benchmark includes real needs of different age groups in six major real-world scenarios.

dialogue capabilities of LMMs. However, these benchmarks merely compare answers to standard solutions, offering limited insights into the fine-grained capabilities of models. To address this limitation, subsequent multimodal understanding benchmarks are developed (Yu et al., 2023b; Liu et al., 2023; Fu et al., 2024a; Ying et al., 2024), covering a broader range of tasks and a larger number of test samples. This refinement enables a more precise evaluation of model capabilities, fostering the development of more robust LMMs. Nevertheless, current benchmarks focus primarily on technical metrics for specific tasks, neglecting two critical research questions:

- Q1: Can these LMMs truly align with the actual needs of humans in real-world scenarios?
- Q2: Can these LMMs subsequently address the diverse needs of distinct groups?

To tackle these challenges, we introduce a novel ”Multi-Dimensional Insights” (MDI) benchmark, which encompasses various real-world scenarios, different problem complexities, and diverse age groups. In detail, the MDI-Benchmark consists of more than 500 real-world images and 1.2k humanposed questions. As shown in Figure 2, it covers six major scenarios of human life: Architecture, Education, Housework, Social Services, Sport, and Transport. Furthermore, MDI-Benchmark focuses on evaluating LMMs from the following two dimensions:

Question Complexity Dimension. This dimension categorizes human-posed problems into two levels of complexity. The first level assesses the basic capabilities of LMMs, such as object detection and optical character recognition (OCR), etc. The second level evaluates more complex capabilities, including logical reasoning, mathematical calculation, and knowledge application.

Age Dimension. Age is a fundamental criterion for evaluating individual differences, as people of different ages have diverse needs. We categorize individuals into three age groups: young people, middle-aged people, and older people, to assess the effectiveness of LMMs in addressing the varying needs and preferences across these groups. Our goal is to comprehensively assess whether LMMs can meet the diverse needs of humans in practical situations.

In summary, our major contributions are listed:

- • To align with the actual needs of humans for Large Multimodal Models, we are the first to propose a multi-modal benchmark for providing a thorough assessment of the capacities of LMMs in practical, real-world scenarios.
- • The MDI-Benchmark includes over 500 real-world images and 1.2k human-posed questions, spanning six real-world multimodal scenarios. Each scenario is divided into 3 sub-domains with 2

- levels of complexity. Additionally, we incorporate age factors into the evaluation to guide LMMs in personalizing their responses for different demographic groups.
- • With the MDI-Benchmark, we conduct a comprehensive evaluation of several mainstream LMMs. Specifically, GPT-4o achieved the best results across all indicators, but there is still significant room for improvement in addressing the needs of different age groups. Further analysis across dimensions such as Scenario, Complexity and Age provides valuable insights for developing reliable, personalized human assistants.

We hope our research will advance the application of multimodal large models in real-world scenarios and pave the way for the development of multi-dimensional personalization.

2 RELATED WORK

- 2.1 MULTIMODAL DATASET AND BENCHMARK

To evaluate the capabilities of LMMs, a variety of benchmarks from past research have been applied. Among them, Flickr30k (Young et al., 2014), COCO Captions (Chen et al., 2015), and Nocaps (Agrawal et al., 2019) are utilized to evaluate LMMs’ text generation and image description abilities. Vizwiz (Bigham et al., 2010), VQA (Goyal et al., 2017), GQA (Hudson & Manning, 2019), and OK-VQA (Marino et al., 2019) are used to assess LMMs’ comprehension of image information and question-answering abilities. For evaluating OCR capabilities, benchmarks like ST-VQA (Biten et al., 2019) and OCR-VQA (Mishra et al., 2019) are employed. DocVQA (Mathew et al., 2021) is specifically used to evaluate a model’s ability to understand and identify documents.

To further explore the fine-grained capabilities of LMMs, recent benchmarks have significantly expanded the types of tasks assessed. Examples of such benchmarks include LVLM-eHub (Xu et al., 2023), MM-Vet (Yu et al., 2023b), MMBench (Liu et al., 2023), SEED-Bench (Li et al., 2023), MME (Fu et al., 2024a), MMT-Bench (Ying et al., 2024), Video-MME (Fu et al., 2024b), MMMU (Yue et al., 2023), MMMU-Pro (Yue et al., 2024a), MathVista (Lu et al., 2024b), Mathverse (Zhang et al., 2025), We-Math(Qiao et al., 2024a), and MMEvol(Luo et al., 2024). Nevertheless, it should be noted that these benchmarks have not fully explored the capability of LMMs to address the diverse needs of different individuals. Therefore, we hope to better explore this ability through the MDI-Benchmark.

- 2.2 LARGE MULTIMODAL MODELS

Building on the success of many large language models (LLMs) (Brown et al., 2020; Touvron et al., 2023; Chiang et al., 2023), recent research has combined large language models with visual encoders to form LMMs with powerful visual understanding and semantic generation capabilities. Many excellent open-source (Hong et al., 2023; Wang et al., 2023; Hu et al., 2024; Lu et al., 2024a; Liu et al., 2024b; Ye et al., 2023; Abdin et al., 2024) and closed-source (Team et al., 2023; Bai et al.,

- 2023; OpenAI, 2023; 2024) projects have been developed. This development has further enhanced the potential for realizing personalized AI assistants.

2.3 PERSONALIZED RESEARCH

To achieve personalized AI assistants, large language models (LLMs) are currently attempting to combine with users’ personalized outputs to enhance their personalization capabilities and enable them to generate outputs that conform to users’ preferences (Wo´zniak et al., 2024; Zhuang et al.,

- 2024b; Baek et al., 2024; Tan et al., 2024). Simultaneously, to further expand the understanding ability of LLMs in the face of different needs, personalized data generation is also crucial(Chan et al., 2024). In this work, we utilize the MDI-Benchmark to evaluate the ability of existing large multimodal models to address personalized needs and provide our insights for future LMMs research.

[Figure 2]

- Figure 2: The overview of the MDI Benchmark’s six real-world multimodal scenarios, each comprising three sub-domains.
- 3 MDI-BENCHMARK

The MDI-Benchmark sample design emphasizes the real-world complexity of information, scene variability, and age differences. People’s information concerns often vary by scenario. As shown in Figure 1, a family buying a new house may focus on practical issues that are closely related to them, such as kitchen type, garage capacity, and bedroom amenities. Spectators at sports events may concern themselves with game details, player achievements, and game progress.

- 3.1 EVALUATION DIMENSION

In contrast to existing work, MDI-Benchmark emphasizes the model’s performance on real-world problems across various ages and complexities within specific task scenarios, it is structured along three different dimensions: scenario, age, and problem complexity.

Scenario Dimension. From the perspective of the scenario, the MDI-Benchmark aims to closely align with the real needs of human life. Unlike the capability evaluation focus of previous LMMs evaluation benchmarks, the MDI-Benchmark is constructed based on real-life scenarios.

In response to the various scenarios that humans face in real life, we have drawn on the definitions provided in sociological literature (Tajfel, 1979; Birmingham et al., 2008; Spears, 2021) and expanded upon them to identify 30 sub-domain scenarios. On this basis, we conducted a one-month questionnaire survey covering people of different ages, genders, and occupations. A total of 2,500 questionnaires were distributed, and 2,374 valid responses were collected. Based on the frequency of sub-domain selection in the questionnaires, we selected the top 18 sub-domains, which were ultimately summarized into six main scenarios: architecture, education, housework, social service, sports, and transport. We collected images from these subdomains to ensure this benchmark is rich in scenario information. Examples are in the Appendix C.1.

Problem Complexity Dimension. In the realm of everyday human activities, the level of complexity varies significantly, and the definition of difficulty is often subjective. To streamline this definition, we have quantified the problems hierarchically based on the fundamental capabilities of the model as the atomic units. Based on this criterion, we have filtered survey questions and refined previous evaluation standards. Furthermore, the MDI-Benchmark is categorized into two levels: (1) The first level involves relatively straightforward problem types that mainly evaluate the model’s ability to extract scenario information. This includes tasks such as detection, optical character recognition, position recognition, color recognition, and other fundamental capacities. (2) The second level demands that the model skillfully analyze both scenario information and user semantic information with logical acuity while integrating relevant knowledge to effectively meet user requirements. Examples are in the Appendix C.2.

Age Dimension. Age is a universal and specific criterion for group classification, making it more objective compared to classifications based on culture and religious beliefs. As a fundamental attribute possessed by everyone, age is easy to quantify and compare. By using age as a classification dimension, we can better understand the needs of various groups and assess the capability of LMMs to meet these diverse needs. For the purposes of assessment and quantification, we identified three

distinct age groups: young people (ages 10-25), middle-aged people (ages 35-50), and old people (ages 60-75). We engaged individuals from these age brackets in real-life scenarios to inquire about their needs. The results of these surveys informed the creation of the initial version of the MDI-Benchmark. Examples are in the Appendix C.3.

- 3.2 DATA COLLECTION

Data Source. Existing LMMs evaluation benchmarks have been widely used to evaluate and train new models. To ensure the accuracy of the evaluation results, we collected over 500 new images that were not included in existing datasets and recruited 120 volunteers from three age groups. From each group, we sampled 10 volunteers to form a 30-person data construction team. The main data collection process was as follows: First, after determining the scenario dimension information, the data construction team wrote detailed scenario information based on their interests. Meanwhile, we input the scenario dimension information into open-source models (e.g., GPT-4o, Gemini 1.5 Pro) and closed-source models (e.g., LLaVA-Next, MiniCPM) to generate more personalized, diverse, and detailed scenario descriptions. Furthermore, the descriptions created by both humans and models were used as keywords to search for relevant images on the Internet. Meanwhile, We paid volunteers a sufficient wage, approximately seven dollars per hour. These volunteers were tasked with categorizing the images into six scenario dimensions. To ensure data balance and minimize bias, we ensured diversity within each age group in terms of gender, occupation, and other factors. Detailed classification standards and guidelines were provided to ensure consistency in categorization. We employed a cross-validation approach, whereby each group of volunteers screened the images, and we retained only those images that were categorized identically by all three groups. Additionally, multiple iterations of validation were conducted. This comprehensive process helped to construct a balanced and reliable data source.

Question and Answer Generation. After obtaining the collected images, we used a heuristic method to manually generate questions and problems. The specific process is as follows: (1) Construction of Knowledge Base. Specifically, multiple open-source and closed-source models are first used to describe the scenario content in the image and are summarized by human experts. Subsequently, additional information related to the scenario content was found through an Internet search, and the image and this information were combined to form a knowledge base. (2) Generation of Difficult Multi-Choice Questions. To ensure the consistency of the generated questions with the image content, we invited volunteers from three different age groups who participated in the data collection phase to submit questions. These volunteers posed questions of varying complexity based on the image scenarios and knowledge base content and created confusing incorrect options. (3) Question Format. The image-question pairs provided by the volunteers had to follow the format: [Level]-[Age]-[Scenario]. Here, Level includes level 1 and level 2; Age includes old, mid, and young; Scenario includes architecture, education, housework, social services, sports, and transport. Finally, a team of experts screened and evaluated the questions submitted by the volunteers to finalize the construction of the questions.

Data Statistics. The MDI-Benchmark is collected from three different dimensions: scenarios, age groups, and abilities. It includes a total of 514 images and 1298 questions, all newly collected. Meanwhile, we strived to ensure a balance of data across different scenarios, ages, and question complexities. The detailed information is presented in the Table 1. As shown in Figure 1, the dataset covers six domains, each with three sub-domains, providing a comprehensive and structured construction of data across various fields.

Table 1: Statistical details of MDI-Benchmark.

Scenarios Number of images Number of L1 questions Number of L2 questions Number of old questions Number of mid questions Number of young questions Architecture 85 121 112 77 74 82

Education 85 114 115 80 79 70 Housework 86 103 109 71 74 67

Social services 86 95 108 65 66 72

Sports 86 107 103 70 73 67 Transport 86 109 102 73 70 68

Total 86 649 649 436 436 426

Table 2: LMMs Performance on MDI-Benchmark in Terms of Level and Scenario. Vertically, the table is composed of a model score and two Level sub-tables, where the model score is obtained from Formula 1. Each sub-table consists of seven columns showing the accuracy rates of LMMs in different scenarios. The first column of each sub-table represents the mean value of the subsequent six columns, reflecting the overall performance at different levels. The annotations for Level and Scenario are as follows: Level 1: assessment questions that focus only on basic perceptual ability; Level 2: assessment questions that involve logical reasoning. The scenarios are abbreviated as follows: Arc (architecture), Edu (education), Hou (housework), Soc (social service), Spo (sport), Tra (transport). Horizontally, the table is divided into two blocks. For better statistics and analysis, we will display the blocks as closed-source model statistics and open-source model statistics. The best performance in each block is highlighted in blue and green.

Level 1 Level 2 Avg Arc Edu Hou Soc Spo Tra Avg Arc Edu Hou Soc Spo Tra

Model Final Score

#### Closed-source

GPT-4o 78.46 87.46 76.47 94.12 92.16 90.20 86.27 94.12 69.45 70.59 70.59 78.43 82.35 54.90 66.67 GPT-4V 74.92 87.46 86.27 92.16 86.27 90.20 88.24 90.20 62.38 72.55 70.59 74.51 60.78 45.10 56.86

Gemini 1.5 Pro 69.13 82.32 68.63 92.16 76.47 88.24 86.27 90.20 55.95 52.94 56.86 54.90 74.51 43.14 58.82 Qwen-VL-Plus 43.57 56.59 43.14 64.71 62.75 78.43 50.98 45.10 30.55 35.29 41.18 37.25 25.49 23.53 23.53

#### Open-source

LLaVA-NeXT-110B 65.59 79.10 60.78 92.16 78.43 84.31 78.43 88.24 52.09 66.67 56.86 54.90 64.71 31.37 43.14 LLaVA-NeXT-72B 63.67 76.21 68.63 88.24 80.39 82.35 70.59 74.51 51.13 66.67 54.90 52.94 60.78 33.33 43.14

MiniCPM-LLaMA3-V 2.5 55.95 72.67 52.94 86.27 70.59 82.35 70.59 80.39 39.23 45.10 49.02 49.02 31.37 27.45 37.25 mPLUG-Owl2-7B 52.57 64.63 49.02 70.59 74.51 70.59 58.82 70.59 40.51 41.18 41.18 47.06 39.22 29.41 49.02 DeepSeek-VL-7B 52.09 68.49 49.02 70.59 74.51 80.39 62.75 80.39 35.69 41.18 33.33 39.22 41.18 21.57 41.18 Phi3-Vision-4.2B 50.80 67.20 50.98 76.47 60.78 80.39 62.75 78.43 34.41 37.25 33.33 41.18 43.14 21.57 33.33

CogVLM-chat 49.84 60.77 49.02 72.55 62.75 56.86 68.63 60.78 38.91 49.02 33.33 43.14 41.18 27.45 43.14 DeepSeek-VL-1.3B 46.30 58.20 45.10 56.86 66.67 56.86 66.67 62.75 34.41 35.29 29.41 29.41 39.22 27.45 49.02 CogAgent-vqa 41.16 49.52 35.29 45.10 66.67 54.90 56.86 43.14 32.80 31.37 35.29 35.29 37.25 25.49 35.29 LLaVA-NeXT-7B 33.60 43.09 31.37 52.94 43.14 49.02 39.22 47.06 24.12 35.29 13.73 37.25 23.53 9.80 27.45

- 4 EXPERIMENTS

- 4.1 EXPERIMENTAL SETTINGS

Evaluation Protocols. To effectively evaluate the model’s output, we require the model to provide the correct answer in its response. Based on this, the accuracy of the response was calculated. This means that if the model articulates the correct concept but fails to produce the precise answer, it will be classified as incorrect. This approach underscores the model’s ability to follow instructions accurately, highlighting any deficiencies in this capacity. In addition, since the prompt input format varies across different models, we investigated the input format for each model. We then endeavored to maintain consistency in the prompts, adhering to the official input format provided by each model. This approach aims to minimize the impact of prompt differences on model performance.

Prompt Template. Table 4 report the prompt templates in our experiments.

Evaluation Models. We studied the performance of two different categories of base models on the MDI-Benchmark. (a) Closed-source models: GPT-4o(OpenAI, 2024), GPT-4V(OpenAI, 2023), Qwen-VL-Plus(Bai et al., 2023), Gemini 1.5 Pro(Team et al., 2023) (b) Open-source models: LLaVA-NeXT-110B(Liu et al., 2024a), LLaVA-NeXT-70B(Liu et al., 2024a), LLaVANeXT-7B(Liu et al., 2024b), DeepSeek-VL-7B, DeepSeek-VL-1.3B(Lu et al., 2024a), Phi3-Vision4.2B(Abdin et al., 2024), MiniCPM-LLaMA3-V 2.5(Hu et al., 2024), CogVLM-chat(Wang et al., 2023), CogAgent-vqa(Hong et al., 2023), mPLUG-Owl2-7B(Ye et al., 2023)

Scoring Metric. Table 2 shows the overall performance of different LMMs under two levels of problem complexity and across six scenarios. To better assess the capabilities demonstrated by the model, we defined the scoring metric:

Scorefinal = α · ScoreL1 + (1 − α) · ScoreL2 (1)

where ScoreL1, ScoreL2 denotes the average performance of LMMs in various fields at the first and second tiers, respectively and we set the default value of α to 0.5.

100

Level1 Level2

80

60

Score(%)

40

20

0

CogVLM -chat

CogAgent -vqa

GPT-4o GPT-4V Gemini 1.5 Pro

Qwen-VL -Plus

LLaVANeXT-110B

LLaVANeXT-72B

MiniCPMLLaMA3-V 2.5

mPLUGOwl2-7B

DeepSeekVL-7B

Phi3Vision-4.2B

DeepSeek -VL-1.3B

LLaVA -NeXT-7B

- Figure 3: The average performance of different LMMs on different difficulty levels of the MDIBenchmark.

- 4.2 MAIN RESULTS

Table 2 illustrates the overall performance of different LMMs on MDI-benchmark. We find out the following insights:

GPT family demonstrate an absolute advantage. GPT-4o leads all models and receives the highest performance score. It can also be observed that closed-source models generally outperform open-source models. However, some powerful open-source models are struggling to catch up with closed-source models. For example, the LLaVA-NeXT-110B, and LLaVA-NeXT-72B performed slightly worse than the Gemini 1.5 Pro and better than the Qwen-VL-Plus.

Scaling phenomenon of model performance. Furthermore, due to the limited data available for the closed-source models, we observed some interesting trends among the open-source models. We selected the best-performing open-source models in various sizes, from LLaVA-NeXT-110B and LLaVA-NeXT-72B to MiniCPM-LLaMA3-V 2.5, DeepSeek-VL-7B, Phi3-Vision-4.2B and DeepSeek-VL-1.3B. As shown in Figure 4 (the Leaderboard of different LMMs), the final scores for these models showed that the larger the model parameters, the better its ability to solve problems in real scenarios. This is consistent with human experience: larger language model parameters mean more text logic training samples and less model distillation. When faced with more complex logical reasoning tasks, these models can leverage more underlying knowledge and fundamental capabilities.

- 4.3 SCENARIO DIMENSION ANALYSIS

The performance of LMMs in daily scenarios still has great room for improvement. To observe the specific performance of different models in various scenarios, as shown in Figure 3, we calculated the accuracy of different models across different fields. We found that these 14 LMMs achieved good performance in Level 1 for the education scenario. The performance is more balanced in the architecture, housework, transport and social service scenarios. However, there are some shortcomings in the performance of sports scenarios, which we believe are closely related to the current training data of LMMs. At present, LMMs research groups focus more on achieving better training and testing levels using existing Internet text data and high-quality textbook data, but they neglect the improvement of datasets and capabilities in everyday life fields. This is where the MDI-Benchmark comes into play. We believe that the types of problems related to logical reasoning and the required background knowledge in the fields of sport and transport are richer and broader than those in architecture, resulting in increased problem difficulty and a significant gap in reasoning performance.

- 4.4 COMPLEXITY DIMENSION ANALYSIS

Decreased performance with increased complexity. As the complexity of the problems increases, the model’s performance in every scenario noticeably decreases. The accuracy of answering questions in the same scenario can also change significantly for the same model. For instance, in the case

[Figure 3]

- Figure 4: Performance of the model at different difficulty levels and the overall performance results of the model under the score metric.

of GPT-4o, the accuracy in the best-performing educational scenario dropped from 94.12 to 70.59. This highlights the significant impact of problem complexity on model performance.

The complexity of questions presents a rich diversity in generalization when it comes to different scenarios. To analyze the detailed performance of these LMMs across multiple levels, we create radar charts (Figure 4) that display the performance of 14 LMMs in various scenarios under Level 1 and Level 2. To illustrate macro performance changes due to varying problem complexity, we also generate statistics of performance variance and summation, plotting average and variance data on different axes to highlight macro trends (Figure 5). Generally, models with high averages and low variances exhibit better and more comprehensive capabilities.

Upon examination at Level 1, it is evident that the majority of models exhibit a balanced performance, as indicated by Figure 4. Notable exceptions to this trend are observed in models such as CogAgent-vqa and LLaVA-NeXT-7B. Under Level 2, GPT-4o’s variance increases significantly, with only the GPT series and Gemini 1.5 Pro maintaining balanced performance. As shown in Figure 4, only the GPT series shows slight performance degradation, while other LMMs exhibit a steep decline in the sports scenario.

Compared to advanced closed-source LMMs, open-source LMMs require further research on specific daily life capabilities and complex problem scenarios to bridge the significant gap. Notably, As shown in Figure 5, LLaVA-NeXT-72B performs similarly to the optimal model LLaVA-NeXT110B at Level 2 but with decreased variance, suggesting that effective distillation to achieve better performance with smaller parameters is a worthy area for further investigation.

We believe that the research community’s lack of focus on enhancing LMMs datasets and capabilities in these areas, along with the diverse and extensive types of problems associated with logical reasoning and required background knowledge, is more pronounced compared to simpler tasks. This diversity results in significant gaps in the model’s inference performance as the complexity of the problems increases. Therefore, further research is needed to address these gaps and improve LMM performance in complex problem scenarios.

- 4.5 AGE DIMENSION ANALYSIS

For a more direct and macro-level performance analysis, we only presented the average performance statistics in the main table, as shown in Table 3, which primarily represents the performance of LMMs across three age stratification. Furthermore, we analyzed the model’s performance in detail based on age groups and scenario dimensions, as shown in the Appendix D. We have the following observations.

All the models to follow under the level evaluation dimensions, but there are differences in performance between different age. As shown in Table 3, GPT-4o remains the top-performing model in the age dimension, demonstrating a performance advantage of 13 points over the highest-ranked open-source model and 35 points over the lowest-ranked closed-source model. This dominant performance in the age-stratified evaluation highlights GPT-4o’s strong generalization ability and its leadership in daily use scenarios. However, when evaluating the model’s capabilities from the perspective of the age dimension, it provides insights into the model’s effectiveness across different

[Figure 4]

Figure 5: The average accuracy and variance of LLMs across six domains at Level 1 and Level 2

Table 3: Performance of Various Models Across Different Age Groups.

Model Avg old middle-aged young Closed-source

GPT-4o 79.74 77.94 78.43 82.84 GPT-4V 76.14 75.49 75.49 77.45

Gemini 1.5 Pro 70.26 70.10 68.63 72.06 Qwen-VL-Plus 44.28 41.67 40.20 50.98

### Open-source

LLaVA-NeXT-110B 66.67 69.12 63.24 67.65 LLaVA-NeXT-72B 64.71 66.67 63.73 63.73

MiniCPM-LLaMA3-V 2.5 56.86 55.88 54.90 59.80 mPLUG-Owl2-7B 53.43 55.39 50.98 53.92 DeepSeek-VL-7B 52.94 53.43 51.96 53.43 Phi3-Vision-4.2B 51.63 53.43 49.02 52.45

CogVLM-chat 50.65 52.94 51.96 47.06 DeepSeek-VL-1.3B 47.06 49.02 39.71 52.45

CogAgent-vqa 41.83 44.12 42.65 38.73 LLaVA-NeXT-7B 34.15 37.75 33.82 30.88

groups in various real-world scenarios. Given the multitude of situations individuals encounter in daily life, a model’s capabilities must be comprehensive to address diverse human needs. The observed decline in accuracy across age groups indicates that there is significant room for improvement in the overall performance of all models within this dimension. This finding underscores the need for further research focusing on age-related issues and highlights both the necessity and innovation of our work.

Models exhibit insufficient overall generalization across different age dimensions. As shown in Figure 6, we further visualize the model’s performance across different age group, including old, middle-aged, young. By summing the model’s results across age dimensions, we find that the old group achieves a total of 856.38, the middle-aged group 764.72, and the young group 902.94. This distribution highlights the actual difficulty order of questions across age levels: middle-aged >old >young. In real-world scenarios, questions posed by middle-aged individuals tend to encompass more aspects and require greater logical reasoning and background knowledge than those from older

or younger individuals. Therefore, multi-modal LMMs need to have robust and comprehensive capabilities to effectively handle such questions. GPT-4o demonstrates strong performance in this aspect, exhibiting smaller performance gaps across all three age-related categories. Interestingly, the Cog-series model, despite having the largest visual encoder, shows a noticeable performance drop in the young group, suggesting that its large visual encoder does not generalize as effectively as CLIP-ViT/L14.

| |37.75%<br><br>44.12%<br><br>49.02%<br><br>52.94%<br>53.43%<br><br><br>53.43%<br><br>55.39%<br><br>55.88%<br><br>66.67%<br><br>69.12%<br><br>41.67%<br><br>70.1%<br><br><br>75.49%<br><br>77.94%<br><br>33|.82%<br><br>42.65%<br><br>39.71%<br><br>51.96%<br><br>49.02%<br><br>51.96%<br><br>50.98%<br><br><br>54.9%<br><br>63.7<br><br>63.2<br><br>40.2%<br><br><br>68<br><br>30.88%<br><br>38<br><br>5|3%<br>4%<br><br><br>.63%<br><br>75.49%<br><br>78.43%<br><br>.73%<br><br>52.45%<br><br>47.06%<br><br>52.45%<br>53.43%<br><br><br>53.92%<br><br>59.8%<br><br>0.98%|63.73%<br><br>67.65%<br><br>72.06%<br><br>77.45%<br><br>82.84|%<br><br>Old<br><br>Middle<br><br>Young|
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
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

GPT-4o

GPT-4V

Gemini 1.5 Pro

Qwen-VL-Plus

LLaVA-NeXT-110B

LLaVA-NeXT-72B

MiniCPM-LLaMA3-V 2.5

mPLUG-Owl2-7B

DeepSeek-vl-7B

Phi3-Vision-4.2B

CogVLM-chat

DeepSeek-VL-1.3B

CogAgent-vqa

LLaVA-NeXT-7B

0 50 100 150 200 250

Figure 6: Performance of different LMMs across the age dimension.

In the age dimension, the scaling performance of language models is evident, but model compression shows great potential. We find that at each model layer, the model with the largest language model parameters achieved the best performance. Empirically, we believe that language models play a more important role in LMMs than visual encoders. Additionally, we are surprised to find that Phi3Vision-4.2B exceed the macro performance of the closed-source model Qwen-VL-Plus using only about 4.2B parameters. This indicates that LMMs still have significant room for exploration in terms of model parameter compression.

- 5 CONCLUSION

In this paper, we propose the MDI-Benchmark, a tool designed to evaluate the capability of Large Multimodal Models (LMMs) in addressing real-world human demands within multi-dimensional scenarios. The MDI-Benchmark comprises over 500 images and 1.2k corresponding requirements, encompassing six major aspects of human life. Additionally, we introduce the concept of age stratification and sampling questions based on the needs of elderly, middle-aged, and young individuals to ensure comprehensive evaluation. Using the MDI-Benchmark, we evaluated 14 existing LMMs, revealing their performance preferences in different scenarios. While GPT-4o performed best across a variety of metrics, there were gaps in performance across all age groups and scenarios. Therefore, we suggest that future studies should focus on improving the adaptability of LMM to human needs and its ability to generalize across different domains and age groups. This will pave the way for the next generation of LMMs that can effectively meet human needs.

REFERENCES

Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024.

Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 8948–8957, 2019.

Jinheon Baek, Nirupama Chandrasekaran, Silviu Cucerzan, Allen Herring, and Sujay Kumar Jauhar. Knowledge-augmented large language models for personalized contextual query suggestion. In Proceedings of the ACM on Web Conference 2024, pp. 3355–3366, 2024.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.

Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samual White, et al. Vizwiz: nearly real-time answers to visual questions. In Proceedings of the 23nd annual ACM symposium on User interface software and technology, pp. 333–342, 2010.

Elina Birmingham, Walter F Bischof, and Alan Kingstone. Social attention and real-world scenes: The roles of action, competition and social content. Quarterly journal of experimental psychology, 61(7):986–998, 2008.

Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Mar¸cal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4291–4301, 2019.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. Scaling synthetic data creation with 1,000,000,000 personas. arXiv preprint arXiv:2406.20094, 2024.

Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023.

Guanting Dong, Daichi Guo, Liwen Wang, Xuefeng Li, Zechen Wang, Chen Zeng, Keqing He, Jinzheng Zhao, Hao Lei, Xinyue Cui, Yi Huang, Junlan Feng, and Weiran Xu. PSSAT: A perturbed semantic structure awareness transferring method for perturbation-robust slot filling. In Nicoletta Calzolari, Chu-Ren Huang, Hansaem Kim, James Pustejovsky, Leo Wanner, Key-Sun Choi, Pum-Mo Ryu, Hsin-Hsi Chen, Lucia Donatelli, Heng Ji, Sadao Kurohashi, Patrizia Paggio, Nianwen Xue, Seokhwan Kim, Younggyun Hahm, Zhong He, Tony Kyungil Lee, Enrico Santus, Francis Bond, and Seung-Hoon Na (eds.), Proceedings of the 29th International Conference on Computational Linguistics, COLING 2022, Gyeongju, Republic of Korea, October 12-17, 2022, pp. 5327–5334. International Committee on Computational Linguistics, 2022. URL https://aclanthology.org/2022.coling-1.473.

Guanting Dong, Rumei Li, Sirui Wang, Yupeng Zhang, Yunsen Xian, and Weiran Xu. Bridging the kb-text gap: Leveraging structured knowledge-aware pre-training for KBQA. In Ingo Frommholz, Frank Hopfgartner, Mark Lee, Michael Oakes, Mounia Lalmas, Min Zhang, and

Rodrygo L. T. Santos (eds.), Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, CIKM 2023, Birmingham, United Kingdom, October 21-25, 2023, pp. 3854–3859. ACM, 2023. doi: 10.1145/3583780.3615150. URL https: //doi.org/10.1145/3583780.3615150.

Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, and Jingren Zhou. Self-play with execution feedback: Improving instruction-following capabilities of large language models. CoRR, abs/2406.13542, 2024a. doi: 10.48550/ARXIV.2406.13542. URL https://doi.org/10.48550/arXiv.2406.13542.

Guanting Dong, Xiaoshuai Song, Yutao Zhu, Runqi Qiao, Zhicheng Dou, and Ji-Rong Wen. Toward general instruction-following alignment for retrieval-augmented generation. arXiv preprint arXiv:2410.09584, 2024b.

Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, and Jingren Zhou. How abilities in large language models are affected by supervised fine-tuning data composition. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 177–198. Association for Computational Linguistics, 2024c. doi: 10.18653/V1/2024. ACL-LONG.12. URL https://doi.org/10.18653/v1/2024.acl-long.12.

Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen Wang, Zhicheng Dou, and Ji-Rong Wen. Understand what LLM needs: Dual preference alignment for retrieval-augmented generation. CoRR, abs/2406.18676, 2024d. doi: 10.48550/ARXIV.2406.18676. URL https://doi.org/10. 48550/arXiv.2406.18676.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2024a.

Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024b.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Minlie Huang, Nan Duan, Weizhu Chen, et al. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv:2309.17452, 2023.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6904–6913, 2017.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In International conference on machine learning, pp. 3929–3938. PMLR, 2020.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021.

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. arXiv preprint arXiv:2312.08914, 2023.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6700–6709, 2019.

Alfred Kobsa and J¨org Schreck. Privacy through pseudonymity in user-adaptive systems. ACM Transactions on Internet Technology (TOIT), 3(2):149–183, 2003.

Ahmet Baki Kocaballi, Shlomo Berkovsky, Juan C Quiroz, Liliana Laranjo, Huong Ly Tong, Dana Rezazadegan, Agustina Briatore, and Enrico Coiera. The personalization of conversational agents in health care: systematic review. Journal of medical Internet research, 21(11):e15360, 2019.

Shanglin Lei, Guanting Dong, Xiaoping Wang, Keheng Wang, and Sirui Wang. Instructerc: Reforming emotion recognition in conversation with a retrieval multi-task llms framework. arXiv preprint arXiv:2309.11911, 2023.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33: 9459–9474, 2020.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.

Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 36, 2024.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024b.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024c.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.

Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024a.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024b.

Run Luo, Haonan Zhang, Longze Chen, Ting-En Lin, Xiong Liu, Yuchuan Wu, Min Yang, Minzheng Wang, Pengpeng Zeng, Lianli Gao, et al. Mmevol: Empowering multimodal large language models with evol-instruct. arXiv preprint arXiv:2409.05840, 2024.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pp. 3195–3204, 2019.

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2200–2209, 2021.

Shervin Minaee, Yuri Boykov, Fatih Porikli, Antonio Plaza, Nasser Kehtarnavaz, and Demetri Terzopoulos. Image segmentation using deep learning: A survey. IEEE transactions on pattern analysis and machine intelligence, 44(7):3523–3542, 2021.

Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pp. 947–952. IEEE, 2019.

OpenAI. Hello gpt-4o, 2024. URL https://openai.com/index/hello-gpt-4o/. R OpenAI. Gpt-4v (ision) system card. Citekey: gptvision, 2023.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744, 2022.

Ivica Pesovski, Ricardo Santos, Roberto Henriques, and Vladimir Trajkovik. Generative ai for customizable learning experiences. Sustainability, 16(7):3034, 2024.

Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024a.

Runqi Qiao, Lan Yang, Kaiyue Pang, and Honggang Zhang. Making visual sense of oracle bones for you and me. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12656–12665, 2024b.

Omid Rafieian and Hema Yoganarasimhan. Ai and personalization. Artificial Intelligence in Marketing, pp. 77–102, 2023.

Waseem Rawat and Zenghui Wang. Deep convolutional neural networks for image classification: A comprehensive review. Neural computation, 29(9):2352–2449, 2017.

Shashi Pal Singh, Ajai Kumar, Hemant Darbari, Lenali Singh, Anshika Rastogi, and Shikha Jain. Machine translation using deep learning: An overview. In 2017 international conference on computer, communications and electronics (comptelix), pp. 162–167. IEEE, 2017.

Xiaoshuai Song, Muxi Diao, Guanting Dong, Zhengyang Wang, Yujia Fu, Runqi Qiao, Zhexu Wang, Dayuan Fu, Huangxuan Wu, Bin Liang, Weihao Zeng, Yejie Wang, Zhuoma Gongque, Jianing Yu, Qiuna Tan, and Weiran Xu. Cs-bench: A comprehensive benchmark for large language models towards computer science mastery. CoRR, abs/2406.08587, 2024. doi: 10.48550/ARXIV.2406. 08587. URL https://doi.org/10.48550/arXiv.2406.08587.

Russell Spears. Social influence and group identity. Annual review of psychology, 72(1):367–390, 2021.

Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355, 2023.

Henri Tajfel. Individuals and groups in social psychology. British Journal of social and clinical psychology, 18(2):183–190, 1979.

Zhaoxuan Tan, Qingkai Zeng, Yijun Tian, Zheyuan Liu, Bing Yin, and Meng Jiang. Democratizing large language models via personalized parameter-efficient fine-tuning. arXiv preprint arXiv:2402.04401, 2024.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239, 2022.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023.

Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-toend model. arXiv preprint arXiv:2409.01704, 2024.

Stanisław Wo´zniak, Bartłomiej Koptyra, Arkadiusz Janz, Przemysław Kazienko, and Jan Koco´n. Personalized large language models. arXiv preprint arXiv:2402.09269, 2024.

Yuxiang Wu, Guanting Dong, and Weiran Xu. Semantic parsing by large language models for intricate updating strategies of zero-shot dialogue state tracking. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pp. 11093–11099. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-EMNLP.741. URL https://doi.org/10. 18653/v1/2023.findings-emnlp.741.

Jun Xiao, Minjuan Wang, Bingqian Jiang, and Junli Li. A personalized recommendation system with combinational algorithm for online learning. Journal of ambient intelligence and humanized computing, 9:667–677, 2018.

Peng Xu, Wenqi Shao, Kaipeng Zhang, Peng Gao, Shuo Liu, Meng Lei, Fanqing Meng, Siyuan Huang, Yu Qiao, and Ping Luo. Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models. arXiv preprint arXiv:2306.09265, 2023.

Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. arXiv preprint arXiv:2311.04257, 2023.

Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, et al. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. arXiv preprint arXiv:2404.16006, 2024.

Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023a.

- Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023b.

Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, and Chang Zhou. Scaling relationship on learning mathematical reasoning with large language models. CoRR, abs/2308.01825, 2023. doi: 10.48550/ARXIV.2308.01825. URL https://doi.org/10. 48550/arXiv.2308.01825.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Ming Yin, Botao Yu, Ge Zhang, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024a.

Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web, 2024b. URL https://arxiv.org/abs/2405.03548.

Zhiyuan Zeng, Jiatong Yu, Tianyu Gao, Yu Meng, Tanya Goyal, and Danqi Chen. Evaluating large language models at evaluating instruction following. arXiv preprint arXiv:2310.07641, 2023.

Duzhen Zhang, Yahan Yu, Chenxing Li, Jiahua Dong, Dan Su, Chenhui Chu, and Dong Yu. Mmllms: Recent advances in multimodal large language models. arXiv preprint arXiv:2401.13601, 2024.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pp. 169–186. Springer, 2025.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023.

Zhong-Qiu Zhao, Peng Zheng, Shou-tao Xu, and Xindong Wu. Object detection with deep learning: A review. IEEE transactions on neural networks and learning systems, 30(11):3212–3232, 2019.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36, 2024.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

Wenwen Zhuang, Xin Huang, Xiantao Zhang, and Jin Zeng. Math-puma: Progressive upward multimodal alignment to enhance mathematical reasoning, 2024a. URL https://arxiv. org/abs/2408.08640.

Yuchen Zhuang, Haotian Sun, Yue Yu, Qifan Wang, Chao Zhang, and Bo Dai. Hydra: Model factorization framework for black-box llm personalization. arXiv preprint arXiv:2406.02888, 2024b.

- A MORE DETAILS ON EXPERIMENT SETUP

- A.1 DETAILS OF THE PROMPT INFORMATION The specific prompt information is shown in Table 4.

Table 4: Prompt templates for response generations.

Type Prompt Template

Multiple Choice

Now, we require you to solve a multiple-choice real-world question. Please briefly describe your thought process and provide the final answer(option). Question: <Question> Option: <Option> Regarding the format, please answer following the template below, and be sure to include two <> symbols: <Thought process>: <<your thought process>> <Answer>: <<your option>>

- A.2 DETAILS OF THE EVALUATED MODELS

- Table 5 shows the release times and model sources of the LMMs we evaluated at MDI-Benchmark.

Table 5: The release time and model source of LMMs used in MDI-Benchmark

Model Release Time Source

GPT-4o (OpenAI, 2024) 2024-05 https://gpt4o.ai/ GPT-4V (OpenAI, 2023) 2024-04 https://openai.com/index/gpt-4v-system-card/ Gemini 1.5 Pro (Team et al., 2023) 2024-05 https://deepmind.google/technologies/gemini/pro/ Qwen-VL-Plus (Bai et al., 2023) 2024-01 https://huggingface.co/spaces/Qwen/Qwen-VL-Plus/ LLaVA-NeXT-110B (Liu et al., 2024a) 2024-05 https://huggingface.co/lmms-lab/llava-next-110b/ LLaVA-NeXT-72B (Liu et al., 2024a) 2024-05 https://huggingface.co/lmms-lab/llava-next-72b/ MiniCPM-LLaMA3-V 2.5 (Hu et al., 2024) 2024-05 https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5/ mPLUG-Owl2-7B (Ye et al., 2023) 2023-11 https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b DeepSeek-VL-7B (Lu et al., 2024a) 2024-03 https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat/ Phi3-Vision-4.2B (Abdin et al., 2024) 2024-05 https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/ CogVLM-chat (Wang et al., 2023) 2023-12 https://huggingface.co/THUDM/cogvlm-chat-hf/ DeepSeek-VL-1.3B (Lu et al., 2024a) 2024-03 https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat/ CogAgent-vqa (Hong et al., 2023) 2023-12 https://huggingface.co/THUDM/cogagent-vqa-hf/ LLaVA-NeXT-7B (Liu et al., 2024a) 2024-03 https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf/

- B MORE RELATED WORKS OF MDI-BENCHMARK

The advent of large language models (LLMs) has driven significant advancements in natural language processing (NLP) Zhao et al. (2023) such as instruction following (Ouyang et al., 2022; Zhou et al., 2023; Su et al., 2023; Zeng et al., 2023; Dong et al., 2024a), RAG (Lewis et al., 2020; Guu et al., 2020; Liu et al., 2024c; Dong et al., 2023; 2024d;b; Gao et al., 2023), reasoning (Yuan et al.,

- 2023; Yu et al., 2023a; Yue et al., 2024b; Gou et al., 2023), information extraction and dialogue system (Thoppilan et al., 2022; Dong et al., 2022; Wu et al., 2023; Lei et al., 2023). Moreover, several studies focuse on how to comprehensively evaluate the various capabilities and robustness of LLMs (Hendrycks et al., 2021; Zheng et al., 2024; Song et al., 2024; Dong et al., 2024c). Building on this success, recent works have combined LLMs with visual encoders to form large multi-modal models (LMMs) with powerful visual understanding and semantic generation capabilities. Both open-source (Hong et al., 2023; Wang et al., 2023; Hu et al., 2024; Lu et al., 2024a; Liu et al.,
- 2024b; Ye et al., 2023; Abdin et al., 2024) and closed-source (Team et al., 2023; Bai et al., 2023; OpenAI, 2023; 2024) works have significantly expanded the capabilities of AI systems across diverse applications. Additionally, studies such as (Li et al., 2024; Zhuang et al., 2024a; Wei et al., 2024; Qiao et al., 2024b) have demonstrated the effectiveness of multi-modal models in domains like medical imaging, mathematical reasoning, and general-purpose understanding, and some intriguing applications.

- C MORE DETAIL ON MDI-BENCHMARK

- C.1 EXAMPLE OF SCENARIO DIMENSION

In this section, we present a selection of images from the MDI-Benchmark for visual demonstration purposes.

- 1. Architecture: Including house planning, work scenes, measuring, etc. As shown in Figure 7.
- 2. Education: Including campus facilities, studying activities, teaching, etc. As shown in Figure 8.
- 3. Housework: Including home arrangements, housework activities, household appliances, etc. As shown in Figure 9.
- 4. Social service: Including travel, shopping, communal facilities, etc. As shown in Figure 10.
- 5. Sport: Including ball sports, racing sports, powerlifting, etc. As shown in Figure 11.
- 6. Transport: including signpost, rail transit, airport, etc. As shown in Figure 12.

[Figure 5]

- Figure 7: Examples of Architecture Scenario.

[Figure 6]

- Figure 8: Examples of Education Scenario.

[Figure 7]

## Figure 9: Examples of Housework Scenario.

[Figure 8]

## Figure 10: Examples of Social Service.

[Figure 9]

## Figure 11: Examples of Sport Scenario.

[Figure 10]

## Figure 12: Examples of Transport Scenario.

- C.2 EXAMPLE OF PROBLEM COMPLEXITY DIMENSION

In this section, we present questions of varying difficulties across six scenario dimensions, as shown in Figures 13 to Figure 18. It is evident that Level 1 questions are relatively simple, while Level 2 questions require LMMs to use more advanced abilities to answer.

[Figure 11]

- Figure 13: Examples of Architecture Scenario Questions.

[Figure 12]

- Figure 14: Examples of Education Scenario Questions.

[Figure 13]

## Figure 15: Examples of Housework Scenario Questions.

[Figure 14]

## Figure 16: Examples of Social Service Scenario Questions.

[Figure 15]

## Figure 17: Examples of Sport Scenario Questions.

[Figure 16]

## Figure 18: Examples of Transport Scenario Question.

- C.3 EXAMPLE OF AGE DIMENTION

In this section, we have sampled various concerns and issues from people across three different age groups within the six major scenarios. These concerns have been categorized by scenario and are visually presented in Figures 19 through 24.

[Figure 17]

Figure 19: Example of Architecture Scenario Age Questions.

[Figure 18]

Figure 20: Example of Education Scenario Age Questions.

[Figure 19]

Figure 21: Example of Housework Scenario Age Questions.

[Figure 20]

Figure 22: Example of Social Service Scenario Age Questions.

[Figure 21]

Figure 23: Example of Sport Scenario Age Questions.

[Figure 22]

Figure 24: Example of Transport Scenario Age Questions.

- D MORE DETAILS ON EXPERIMENT RESULTS We present the performance of models across different age groups in Table 6.

- Table 6: Performance of models across different age groups. The best performance in each block is highlighted in blue and green.

Model

Avg Arc Edu Hou Soc Spo Tra Old Mid Young Old Mid Young Old Mid Young Old Mid Young Old Mid Young Old Mid Young Old Mid Young

Closed-source

GPT-4o 77.94 78.43 82.84 79.41 67.65 73.53 85.29 79.41 82.35 82.35 82.35 91.18 88.24 79.41 91.18 64.71 76.47 70.59 67.65 85.29 88.24 GPT-4V 75.49 75.49 77.45 79.41 76.47 82.35 82.35 76.47 85.29 76.47 85.29 79.41 76.47 73.53 76.47 67.65 61.76 70.59 70.59 79.41 70.59

Gemini 1.5 Pro 70.10 68.63 72.06 58.82 47.06 76.47 73.53 79.41 70.59 67.65 64.71 64.71 85.29 70.59 88.24 55.88 67.65 70.59 79.41 82.35 61.76 Qwen-VL-Plus 41.67 40.20 50.98 38.24 32.35 47.06 44.12 52.94 61.76 50.00 38.24 61.76 50.00 47.06 58.82 32.35 38.24 41.18 35.29 32.35 35.29

Open-source

LLaVA-NeXT-110B 69.12 63.24 67.65 73.53 52.94 64.71 76.47 76.47 70.59 70.59 67.65 61.76 76.47 64.71 82.35 50.00 55.88 58.82 67.65 61.76 67.65 LLaVA-NeXT-72B 66.67 63.73 63.73 73.53 58.82 70.59 73.53 73.53 67.65 67.65 67.65 64.71 73.53 61.76 79.41 52.94 55.88 47.06 58.82 64.71 52.94

MiniCPM-LLaMA3-V 2.5 55.88 54.90 59.80 50.00 44.12 52.94 64.71 67.65 70.59 58.82 52.94 67.65 55.88 50.00 64.71 47.06 50.00 50.00 58.82 64.71 52.94 mPLUG-Owl2-7B 55.39 50.98 53.92 47.06 38.24 50.00 73.53 44.12 50.00 58.82 64.71 58.82 58.82 52.94 52.94 38.24 47.06 47.06 55.88 58.82 64.71 DeepSeek-VL-7B 53.43 51.96 53.43 41.18 41.18 52.94 61.76 50.00 44.12 55.88 55.88 58.82 61.76 44.12 76.47 41.18 52.94 32.35 58.82 67.65 55.88 Phi3-Vision-4.2B 53.43 49.02 52.45 44.12 41.18 47.06 58.82 52.94 52.94 52.94 44.12 55.88 64.71 58.82 61.76 50.00 38.24 38.24 50.00 58.82 58.82

CogVLM-chat 52.94 51.96 47.06 44.12 58.82 44.12 61.76 50.00 47.06 52.94 55.88 50.00 50.00 50.00 47.06 41.18 52.94 50.00 67.65 44.12 44.12 DeepSeek-VL-1.3B 49.02 39.71 52.45 41.18 29.41 50.00 50.00 32.35 47.06 50.00 47.06 47.06 58.82 35.29 50.00 29.41 52.94 58.82 64.71 41.18 61.76

CogAgent-vqa 44.12 42.65 38.73 32.35 41.18 26.47 38.24 47.06 35.29 50.00 52.94 50.00 52.94 35.29 50.00 41.18 47.06 35.29 50.00 32.35 35.29 LLaVA-NeXT-7B 37.75 33.82 30.88 32.35 32.35 35.29 35.29 38.24 26.47 44.12 47.06 29.41 41.18 26.47 41.18 32.35 26.47 14.71 41.18 32.35 38.24

- E CORRECT RESPONDS FROM GPT-4O

In view of GPT-4o’s leading position in each scene and age dimension, we selected the correct answers and their reasoning processes for each scenario to display. The results are shown in Figures 25 through 30.

[Figure 23]

- Figure 25: Example of GPT-4o Architecture Scenario Correct Answers.

[Figure 24]

## Figure 26: Example of GPT-4o Education Scenario Correct Answers.

[Figure 25]

## Figure 27: Example of GPT-4o Housework Scenario Correct Answers.

[Figure 26]

- Figure 28: Example of GPT-4o Social Service Scenario Correct Answers.
- Figure 29: Example of GPT-4o Sport Scenario Correct Answers.

[Figure 27]

[Figure 28]

## Figure 30: Example of GPT-4o Trans Scenario Correct Answers.

- F BAD CASE

In this section, we will conduct a case study of the types of errors that different models make in each dimension of MDI-Benchmark. We classify errors into three categories: information extraction errors, lack of knowledge errors, and reasoning errors. Errors are highlighted in red.

Information Extraction Error. As shown in Figure 31. It occurs most frequently. This is because the visual encoder of LMMs often fails to correctly capture the content information in the images, leading to incorrect answers.

[Figure 29]

- Figure 31: Example of Information Extraction Error.

Knowledge Deficiency Error. As shown in Figure 32. Because LMMs lack the ability to associate and search for relevant knowledge within certain contexts. For example, when presented with an image of a past sports event, the model fails to provide the final score.

[Figure 30]

- Figure 32: Example of Knowledge Deficiency Error.

Reasoning Error. As shown in Figure 33. LMMs correctly extract relevant visual information from the image but make mistakes during the reasoning process, leading to incorrect answers.

[Figure 31]

Figure 33: Example of Reasoning Error.

