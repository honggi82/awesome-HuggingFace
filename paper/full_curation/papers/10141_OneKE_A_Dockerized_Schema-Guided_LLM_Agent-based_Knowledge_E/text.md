# arXiv:2412.20005v2[cs.CL]6Feb2025

## OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System

Yujie Luo

luo.yj@zju.edu.cn Zhejiang University ZJU-Ant Group Joint Research Center for Knowledge Graphs Hangzhou, China

Xiangyuan Ru Kangwei Liu

Zhejiang University ZJU-Ant Group Joint Research Center for Knowledge Graphs Hangzhou, China

Lin Yuan Mengshu Sun

ZJU-Ant Group Joint Research Center for Knowledge Graphs Ant Group Hangzhou, China

Ningyu Zhang*

Zhejiang University ZJU-Ant Group Joint Research Center for Knowledge Graphs Hangzhou, China

Lei Liang Zhiqiang Zhang

ZJU-Ant Group Joint Research Center for Knowledge Graphs Ant Group Hangzhou, China

Jun Zhou*

ZJU-Ant Group Joint Research Center for Knowledge Graphs Ant Group Hangzhou, China

Lanning Wei Da Zheng

ZJU-Ant Group Joint Research Center for Knowledge Graphs Ant Group Hangzhou, China

Haofen Wang

Tongji University Shanghai, China

Huajun Chen*

Zhejiang University ZJU-Ant Group Joint Research Center for Knowledge Graphs Hangzhou, China

### Abstract

We introduce OneKE, a dockerized schema-guided knowledge extraction system, which can extract knowledge from the Web and raw PDF Books, and support various domains (science, news, etc.). Specifically, we design OneKE with multiple agents and a configure knowledge base. Different agents perform their respective roles, enabling support for various extraction scenarios. The configure knowledge base facilitates schema configuration, error case debugging and correction, further improving the performance. Empirical evaluations on benchmark datasets demonstrate OneKE’s efficacy, while case studies further elucidate its adaptability to diverse tasks across multiple domains, highlighting its potential for broad applications. We have open-sourced the Code1 and released a Video2.

### Keywords

Information Extraction; Natural Language Processing; Large Language Models

- 1https://github.com/zjunlp/OneKE
- 2http://oneke.openkg.cn/demo.mp4

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

WWW Companion ’25, April 28-May 2, 2025, Sydney, NSW, Australia © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-1331-6/2025/04 https://doi.org/10.1145/3701716.3715189

##### ACM Reference Format:

Yujie Luo, Xiangyuan Ru, Kangwei Liu, Lin Yuan, Mengshu Sun, Ningyu Zhang*, Lei Liang, Zhiqiang Zhang, Jun Zhou*, Lanning Wei, Da Zheng, Haofen Wang, and Huajun Chen*. 2025. OneKE: A Dockerized SchemaGuided LLM Agent-based Knowledge Extraction System. In Companion Proceedings of the ACM Web Conference 2025 (WWW Companion ’25), April 28-May 2, 2025, Sydney, NSW, Australia. ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/3701716.3715189

### 1 Introduction

Knowledge extraction–obtaining knowledge from data, is a critical component for a wide range of practical systems such as Knowledge Graph (KG) construction [1], Retrieval Augmentation (RAG) [3], and domain-specific applications like scientific discovery [2] and intelligence analysis [7]. The last decades have witnessed the development of various knowledge extraction systems [5, 9, 10]. In particular, with the emergence of Large Language Models (LLMs), new works such as InstructUIE [8], iText2KG [4] and AgentRE [6] have been continuously emerged. However, previous approaches still struggle to effectively extract information from raw data following complex schemas and face challenges in debugging and correcting errors when they occur.

Note that previous efforts have primarily focused on the capabilities of individual models while neglecting the design of a comprehensive system to address the knowledge extraction task. To this end, we introduce OneKE, a dockerized schema-guided knowledge extraction system. We adopt a multi-agent design with a configure knowledge base to provide knowledge extraction support for various scenarios and error debugging, aiming to meet the practical needs of users as much as possible. Following [6],

|Source Text|
|---|

###### General IE

[Figure 1]

[Figure 2]

[Figure 3]

###### Extraction Agent Reflection Agent

Extract entities belonging to the entity types: […].

###### Schema Agent

[Figure 4]

[Figure 5]

Direct IE IE with Case Retrieval

No Reflection Case Reflection

Default Schema Predefined Schema Self Schema Deduction

Website Dataset

Science IE

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Extract all chemical reagents from the text.

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Book Dialog

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Extracted Result

News IE

[Figure 23]

Reflected Result

###### Output Schema

[Figure 24]

[Figure 25]

Extract key information of the given news.

According to the reference examples and output schema, the answer is as follows : ```

HTML, PDF, Word

[Figure 26]

Current Trial

Previous Trial

{

________ ________ ________

________ ________ ________

"chemical_list": [{ "name": "chemical_name", "form": "pyisical_state"}]

Extracted Result ```

}

[Figure 27]

[Figure 28]

Final Extraction Result

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Predefined Schema Correct Cases Bad Cases Update

###### General

News

###### Science

|[Figure 35]<br><br>Schema Repository<br><br>[Figure 36]<br><br>NER_Schema RE_Schema<br><br>News_Schema Paper_Schema<br><br>Correct Cases Bad Cases<br><br>Case Repository<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>Correct Answer Incorrect Answer Reasoning Steps Reflections & Issues<br><br>[Figure 41]<br><br>| | |
|---|---|---|
| |Configure Knowledge Base| |

[Figure 42]

London: Location

Chemicals

Title: … … … Event: … … Person: … …

Name: … … … Form: … … …

UK: Country

… …

#### Figure 1: The overview of the OneKE system, supporting various domains (science, news, etc.) and data (Web HTML, PDF, etc.).

we design three agents: Schema Agent for schema analysis with various data types, Extraction Agent for extracting knowledge with various LLMs, and Reflection Agent to debug and handle erroneous cases. Based on this design, OneKE can efficiently process source texts of varying lengths and formats, such as HTML and PDF, and demonstrates a robust capability to adapt to diverse task configurations, yielding a comprehensive range of output schemas tailored to specific requirements.

Qwen and ChatGLM, as well as proprietary models like GPT-4, enabling effective knowledge extraction.

(3) Debugging and fixing errors. Most previous works require retraining the model when encountering error cases. In contrast, we integrate Case Repository into OneKE to equip the model with reflective and error-correcting capabilities, enabling its continuous improvement in knowledge extraction tasks.

### 2.1 Schema Agent

We evaluate OneKE using two benchmark datasets for Named Entity Recognition (NER) and Relation Extraction (RE), demonstrating the effectiveness of our framework. Furthermore, to explore the versatility of OneKE in practical applications, we conduct case analyses on specific extraction tasks. These scenarios encompass extracting structured information from the Web news articles and raw PDF book chapters, highlighting OneKE’s capability to manage diverse data formats and varying task contexts effectively. This flexible framework, which operates without the necessity of finetuning, is adept at swift adaptation to forthcoming LLMs, thereby amplifying their capabilities and elevating their overall efficacy.

To support various task settings and data types, we develop the Schema Agent based on LLMs to generate the corresponding output schema for each task. The primary goal is to preprocess the data, standardize its format and schema, and prepare it for the subsequent information extraction step. To support various real-world data, we utilize the document_loaders module provided by the Langchain, to preprocess the data and perform chunking on long texts. Users can also define new data types, and add custom preprocessing methods. Next, if the user has defined a schema, given a task description and raw data input, the Schema Agent will select the appropriate pre-defined schema from the schema repository in the configure knowledge base. If the user does not provide a schema, the model will generate a unified output schema based on the user’s instructions, such as “Extract characters and background setting”. Users can customize schemas using simple text by updating the schema repository in the Configure Knowledge Base.

### 2 Design and Implementation

OneKE is thoughtfully designed to address the complexities and challenges inherent in knowledge extraction. As shown in Figure 1, the framework is guided by several key considerations that enhance its functionality and adaptability in real-world scenario:

### 2.2 Extraction Agent

#### (1) Adaptability to real-world data. Real-world information

extraction tasks often handle raw data, like HTML, PDF, etc. Based on this, the OneKE framework supports a variety of data types rather than pure text. We also reserve a user-defined interface to support new data types in the future.

Upon receiving the unified output schema from the Schema Agent, we design the Extraction Agent to utilize LLMs for extracting knowledge, thereby generating the preliminary extraction results. Specifically, this module supports a variety of models, including locally deployed open-source models such as LLaMA, Qwen, and ChatGLM, as well as API services like OpenAI and DeepSeek. To enhance performance, the Extraction Agent learns from similar cases and applies this knowledge to the extraction process. Relevant cases are retrieved from the Case Repository using semantic similarity

#### (2) Generalization for complex schemas. Practical knowledge

extraction scenarios should handle diverse and complex schemas, or even no schema. Thus, we design the OneKE-specific Schema Agent to support both pre-defined schemas and self-schema deduction using LLMs. OneKE also supports various LLMs, including LLaMA,

OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System WWW Companion ’25, April 28-May 2, 2025, Sydney, NSW, Australia

combined with string matching (via the all-MiniLM-L6-v2 model and the FuzzyWuzzy package, we set Top two as default). These cases are then incorporated as few-shot examples into the original context to form the prompt, after which the LLM is called to obtain the extraction results. After the above steps, we obtain the preliminary extraction results; however, errors often occur. To address this, we design a reflection agent to debug and fix these errors. We use self-consistency to filter out the cases where the model is uncertain and pass these cases to the Reflection Agent.

### 2.3 Reflection Agent

To enable debugging and error correction, allowing OneKE to learn from past mistakes, we follow [6] to design the Reflection Agent to facilitate reflection and optimization. By leveraging prior knowledge, the agent refines and improves the initial output from the previous module, ultimately producing the final extraction result. Concretely, the agent leverages external Case Repository, specifically relevant bad cases, to facilitate reflection and correction. In a manner similar to the retrieval approach discussed in Section 2.2, the Reflection Agent fetches bad cases that are most relevant to the current task and extraction text. These relevant bad cases, along with their associated reflective analyses, are subsequently incorporated into the LLM. In this way, the agent can effectively learn from past mistakes, enabling its error correction capabilities to generate accurate answers.

### 2.4 Configure Knowledge Base

The Configure Knowledge Base provides essential information for the three agents, including manually defined schemas for various tasks, and historical extraction cases, enhancing the performance and error correction capabilities:

Schema Repository. In the OneKE system, the Schema Repository provides the pre-defined schema for the Schema Agent, supporting the subsequent extraction process. Specifically, the Schema Repository includes pre-defined output schemas for NER, RE, and EE tasks, along with templates for various data scenarios such as scientific academic papers and news reports. The schemas in the Schema Repository are structured as Pydantic objects, enabling seamless serialization into JSON format for the Extraction Agent. Moreover, this structure allows users to customize new schemas within the repository, thus enhancing adaptability and extensibility.

Case Repository. To enable debugging and error correction in OneKE, we design the Case Repository, which primarily stores traces of past knowledge extraction cases. This repository supports the Extraction Agent in performing extractions and assists the Reflection Agent in reflecting on and correcting errors. Specifically, knowledge extraction cases stored in Case Repository can be divided into two categories: Correct Cases and Bad Cases. Correct Cases provide the Extraction Agent with reasoning steps of successful extraction, while Bad Cases offer the Reflection Agent warnings about avoidable mistakes. The Case Repository will be automatically updated once a knowledge extraction task is completed. Concretely, this module first generates reasoning steps derived from the correct answer, storing both the correct answer and its reasoning steps in the Correct Case Repository to enhance task understanding. Additionally, the agent compares its answer with

the correct one and reflects on its original response to identify potential issues. It then stores the original answer along with the corresponding reflections in the Bad Case Repository for future reference.

Vanilla

Case Reflection

Vanilla

Case Reflection

Case Retrieval

Case Retrieval

80

40

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

60

30

F1Score

F1Score

40

20

10

20

0

0

LLaMA-3-8B-Instruct GPT-4-turbo

LLaMA-3-8B-Instruct GPT-4-turbo

NYT-11-HRL

CrossNER

Figure 2: Performance of different components in OneKE.

### 3 Evaluation

Experimental Settings. We evaluate OneKE on the CrossNER and NYT-11-HRL datasets. CrossNER is a cross-domain NER dataset, while NYT-11-HRL focuses on the RE task within the news domain. The performance of OneKE is evaluated on both the LLaMA-3-8BInstruct and GPT-4-turbo models.

Main Results. As depicted in Figure 2, the various methods employed in OneKE confer performance enhancements across both NER and RE. Notably, the Case Retrieval method of the Extraction Agent achieves the most significant improvements. Through analysis, we observe that the agent effectively applies the reasoning paths in the provided samples, thereby facilitating accurate extraction. Additionally, by comparing the two different tasks, we observe that the aforementioned Case Retrieval method is more effective for the more challenging RE task, as the intermediate reasoning steps are essential in such complex scenarios. The Case Reflection primarily emphasizes the model’s ability to recognize known errors and its capacity for transfer learning regarding those errors, leading to similar improvements across both tasks. We provide case studies of specific application scenarios in the following Section.

### 4 Application

In practical applications, the OneKE framework supports diverse data formats (HTML, PDF, Word), accommodating both short and long contexts for seamless integration into various downstream applications. We provide case analyses in the following two representative extraction scenarios.

Web News Extraction. In the news domain, OneKE enhances the knowledge extraction of Web news content, thereby facilitating downstream tasks such as effective sentiment monitoring, proactive risk management, as well as a variety of additional applications. As illustrated in Figure 3, the extraction task starts with Extracting key information from news articles on a randomly selected raw HTML page from the web, aiming to identify the overall nature of the news and obtain structured key insights. After parsing the HTMLformatted text, the Schema Agent first identifies its domain and genre as a Politics News Report, offering crucial guidance. Utilizing this metadata, the Schema Agent generates a structured Output Schema in code format that effectively captures the key information

[Figure 43]

###### Web News Extraction

###### Extracted Result

###### Text: (Field: Politics, Grene: News Report) + [HTML] Output Schema:

###### Instruction: Extract key information from this article. Text: [News.html]

Json

Python

{

class Person(BaseModel): name: str = Field(description="...")

“title”: “Tulsi Gabbard: Trump's DNI pick", "publication_date": "2024-12-04T17:06:00Z", "key_words": ["Tulsi Gabbard", "Donald Trump"], "events": [ {

Html

[Figure 44]

[Figure 45]

<!DOCTYPE html> <html lang="en"> <head>

class Event(BaseModel): name: str = Field(description="...") people: [List[Person]]=Field(description="...")

"name": "Intel Director Nomination", "people": [

<title> Tulsi Gabbard: Trump&#x27;s DNI pick</title> </head>

class ExtractionTarget(BaseModel): title: str = Field(description="...") events: List[Event] = Field(description="...")

<body> ... ... </body> </html>

{"name": "Tulsi Gabbard", "role" : "Nominee"}, {"name": "Donald Trump", "role": "President"}]

[Figure 46]

[Figure 47]

...

}

[Figure 48]

###### Book Knowledge Extraction

Text: (Field: Literature, Grene: Novel) + [PDF] Output Schema:

###### Extracted Result

###### Instruction: Extract characters and background setting. Text: [Chapter1.pdf]

Json

Python

{

[Figure 49]

THE BOY WHO LIVED

class Character(BaseModel): name: str = Field(description="...")

"main_characters": [ {"name": "Mr. Dursley"}, {"name": "Harry Potter}, ...],

[Figure 50]

[Figure 51]

Mr.andMrs.Dursley,ofnumberfour,PrivetDrive,were

class Setting(BaseModel): location: str = Field(description="...") time_period: [str] = Field(description="...")

"background_setting": {

proud to say that they were perfectly normal, thank you very much. They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense.

class ExtractionTarget(BaseModel):

"location": "Privet Drive, suburban", "time_period": "Late 20th century"

characters: List[Character] = Field(...) background_setting: Setting = Field(...)

[Figure 52]

} }

[Figure 53]

[...16 pages left...]

#### Figure 3: Using OneKE on Web News Extraction and Book Knowledge Extraction.

### Acknowledgments

of the news. Once the Output Schema has been serialized into a JSON format description, the Extraction Agent and the Reflection Agent collaborate to undertake subsequent extraction and reflective optimization tasks. This cooperative effort of the agents culminates in a JSON output that captures the key information and structure of the news report.

This work was supported by the National Natural Science Foundation of China (No. 62206246, No. U23B2057, No. 62176185), the Fundamental Research Funds for the Central Universities (226-202300138), Yongjiang Talent Introduction Programme (2021A-156-G), CIPSC-SMP-Zhipu Large Model Cross-Disciplinary Fund, Ningbo Science and Technology Special Projects under Grant No. 2023Z212. This work was supported by Ant Group and Zhejiang University Ant Group Joint Laboratory of Knowledge Graph.

Book Knowledge Extraction. Another application of OneKE lies in extracting structured knowledge from extensive corpora, including books, documents, or manuals. Specifically, we use the first chapter of the Harry Potter series (with a total length of 17 pages in PDF format) as the target text. The extraction focuses on the main characters and the background setting within this chapter. As shown in Figure 3, the generated Output Schema accurately identified the two target extraction objects: “main_characters” and “background_setting”. Subsequently, with the collaboration of the Extraction Agent and the Reflection Agent, OneKE successfully extracted relevant information.

### References

- [1] Xiang Chen, Ningyu Zhang, Xin Xie, Shumin Deng, Yunzhi Yao, Chuanqi Tan, Fei Huang, Luo Si, and Huajun Chen. 2022. Knowprompt: Knowledge-aware prompt-tuning with synergistic optimization for relation extraction. In WWW.
- [2] John Dagdelen, Alexander Dunn, Sanghoon Lee, Nicholas Walker, Andrew S Rosen, Gerbrand Ceder, Kristin A Persson, and Anubhav Jain. 2024. Structured information extraction from scientific text with large language models. Nature Communications 15, 1 (2024), 1418.
- [3] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. 2023. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997 (2023).
- [4] Yassir Lairgi, Ludovic Moncla, Rémy Cazabet, Khalid Benabdeslem, and Pierre Cléau. 2024. iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models. CoRR abs/2409.03284 (2024). arXiv:2409.03284
- [5] Bo Li, Gexiang Fang, Yang Yang, Quansen Wang, Wei Ye, Wen Zhao, and Shikun Zhang. 2023. Evaluating ChatGPT’s Information Extraction Capabilities: An Assessment of Performance, Explainability, Calibration, and Faithfulness. arXiv preprint arXiv:2304.11633 (2023).
- [6] Yuchen Shi, Guochao Jiang, Tian Qiu, and Deqing Yang. 2024. AgentRE: An Agent-Based Framework for Navigating Complex Information Landscapes in Relation Extraction. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, CIKM 2024. ACM.
- [7] Nan Sun, Ming Ding, Jiaojiao Jiang, Weikang Xu, Xiaoxing Mo, Yonghang Tai, and Jun Zhang. 2023. Cyber threat intelligence mining for proactive cybersecurity defense: a survey and new perspectives. IEEE Communications Surveys & Tutorials 25, 3 (2023), 1748–1774.
- [8] Xiao Wang, Weikang Zhou, Can Zu, et al. 2023. InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction. CoRR abs/2304.08085 (2023). arXiv:2304.08085
- [9] Xiang Wei, Xingyu Cui, Ning Cheng, Xiaobin Wang, Xin Zhang, Shen Huang, Pengjun Xie, Jinan Xu, Yufeng Chen, Meishan Zhang, et al. 2024. Chatie: Zero-shot information extraction via chatting with chatgpt. arXiv preprint arXiv:2302.10205 (2024).
- [10] Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, and Tong et al. Xu. 2024. Large language models for generative information extraction: A survey. Frontiers of Computer Science (2024).

### 5 Conclusion and Future Work

In this paper, we introduce OneKE, a Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System. OneKE is designed for flexible application across a spectrum of extraction tasks in real-world scenarios. It can handle source texts of varying lengths and formats (such as HTML and PDF) while demonstrating the capability to adapt to diverse task configurations, generating a broad range of output schemas tailored to specific requirements. Moreover, the integration of a self-reflection mechanism enables iterative improvement informed by external feedback, thereby improving both accuracy and adaptability.

Long-term Maintenance. We will maintain OneKE over the long term, adding new features and fixing bugs. To advance document data extraction and comprehension, we plan to develop methodologies for the integration and analysis of diverse chart types and content. We hope OneKE can serve as a helpful tool for researchers and engineers engaging in knowledge extraction with LLMs.

