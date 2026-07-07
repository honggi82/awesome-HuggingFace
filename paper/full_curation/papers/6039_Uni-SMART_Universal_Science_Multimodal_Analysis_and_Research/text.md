# arXiv:2403.10301v2[cs.CL]15Jun2024

[Figure 1]

## UNI-SMART: UNIVERSAL SCIENCE MULTIMODAL ANALYSIS AND RESEARCH TRANSFORMER

Hengxing Cai1, Xiaochen Cai1, Shuwen Yang1, Jiankun Wang1, Lin Yao1, Zhifeng Gao1, Junhan Chang1, Sihang Li1, Mingjun Xu1, Changxin Wang1, Hongshuai Wang1, Yongge Li1, Mujie Lin1, Yaqi Li1, Yuqi Yin1, Zheng Cheng2, Zifeng Zhao2, Linfeng Zhang1,2 and Guolin Ke1 1DP Technology 2AI for Science Institute, Beijing {caihengxing, caixiaochen, yangsw, wangjiankun, yaol, gaozf, changjh, lisihang, xumj, wangchangxin, wanghongshuai, liyongge, linmujie, liyq, yinyuqi, zhanglf, kegl}@dp.tech {chengz, zhaozf}@aisi.ac.cn

http://uni-smart.dp.tech/

### ABSTRACT

In scientific research and its application, scientific literature analysis is crucial as it allows researchers to build on the work of others. However, the fast growth of scientific knowledge has led to a massive increase in scholarly articles, making in-depth literature analysis increasingly challenging and timeconsuming. The emergence of Large Language Models (LLMs) has offered a new way to address this challenge. Known for their strong abilities in summarizing texts, LLMs are seen as a potential tool to improve the analysis of scientific literature. However, existing LLMs have their own limits. Scientific literature often includes a wide range of multimodal elements, such as tables, charts, and molecule, which are hard for text-focused LLMs to understand and analyze. This issue points to the urgent need for new solutions that can fully understand and analyze multimodal content in scientific literature. To answer this demand, we present Uni-SMART (Universal Science Multimodal Analysis and Research Transformer), an innovative model designed for in-depth understanding of multimodal scientific literature. Through rigorous quantitative evaluation across several domains, Uni-SMART demonstrates superior performance over other text-focused LLMs. Furthermore, our exploration extends to practical applications, including patent infringement detection and nuanced analysis of charts. These applications not only highlight Uni-SMART’s adaptability but also its potential to revolutionize how we interact with scientific literature.

### 1 Introduction

Scientific literature, encompassing patents and academic papers, constitutes a rich science data resource, including but not limited to drug properties and activities, reaction pathways, and manufacturing processes. However, extracting target information from this extensive corpus is a laborious and time-intensive task. It necessitates meticulous manual review, analysis, and extraction – processes that are inherently slow and prone to human error [1, 2]. To enhance the efficiency of information retrieval, specialized databases like Sci-Finder [3] and Reaxys [4] have been developed. However, their utility is constrained to document retrieval for molecule and reaction queries, lacking the capabilities of information extraction and knowledge comprehension to function as domain assistants. Consequently, users must still engage in the tedious tasks of reading and analyzing the retrieved documents to extract definitive answers. This limitation poses a significant bottleneck in the utilization of scientific data, hindering research progress and the rapid application of discoveries. Thus, researchers and practitioners require an intelligent navigator that can swiftly guide through the complexities of the latest scientific data, identify relevant information with precision, and present it in a digestible format.

The emergence of Large Language Models (LLMs), represented by LLaMA [5], Gemini [6], and GPT [7, 8, 9], has marked a significant milestone in the evolution of natural language processing. These models have revolutionized the extraction of textual information from documents, enabling direct responses to queries using the extracted content. Despite their proficiency, current LLMs are primarily designed for text extraction and often struggle with the multimodal aspects inherent in scientific literature, which include a large number of tables, charts, and reactions. The extraction and interpretation of such multimodal data require an understanding that beyond texts and delves into visual and structural contents.

[Figure 2]

[Figure 3]

Patent Infringement Analysis

###### Uni-SMART

[Figure 4]

[Figure 5]

[Figure 6]

Alloy Materials

###### Multimodal Ability

Chemical Reactions

- •Molecules
- •Chemical Reactions
- •Tables
- •……

Application Scope

- •Patent Infringement Analysis
- •Chart Analysis
- •……

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Texts

Organic Material

Chart Analysis

Tables

[Figure 13]

[Figure 14]

Molecules

[Figure 15]

Charts

Biology

[Figure 16]

Drug Discovery

Figure 1: Uni-SMART overview. It interprets multimodal data (e.g., texts, molecule structures, chemical reactions, charts, and tables), thereby facilitating a broad range of applications such as patent infringement analysis, chart analysis, and more.

To address these challenges, we developed UniSMART (Universal Science Multimodal Analysis and Research Transformer), which extends the capabilities of LLMs beyond text, allowing for the interpretation of the multimodal content that is crucial in scientific literature. As illustrated in Figure 1, it is designed to recognize and analyze multimodal data, such as molecule structures, chemical reactions, charts, and tables, alongside textual content, facilitating a comprehensive understanding of scientific literature. Such ability not only augments automated and precise information extraction but also enriches the interaction between researchers and the vast expanse of scientific knowledge.

To rigorously assess the multimodal capabilities of Uni-SMART, a comparative analysis was conducted against several LLMs, such as GPT-4o, Gemini, and Claude3. Our assessment targets several data types critical to the comprehension of scientific documents: tables, charts, molecular structures, and chemical reactions. The results demonstrate UniSMART’s superior performance in all tested areas, especially in understanding and analyzing complex multimodal contents, thus highlighting its potential as a helpful assistant for scientific literature analysis.

In the following sections, Section 2 details Uni-SMART’s data sources and iterative training approach. Subsequently, Section 3 – Evaluation, presents detailed comparisons of Uni-SMART with several LLMs across a variety of modalities, showcasing its advanced capabilities in multimodal data interpretation. Then, Section 4 – Application showcases some specific applications of Uni-SMART in the scientific domain. Finally, Section 5 discusses Uni-SMART’s limitations and future research avenues, along with its potential impact on scientific research and technological advancement.

### 2 Method

As depicted in Figure 2, Uni-SMART sources training data from a wide range of scientific literature from global patents, news articles, scientific publications, and market reports. In particular, It adopts a cyclical, iterative approach to enhance its multimodal understanding capabilities, comprising the following key components:

- • Multimodal Learning: During the initialization phase, the parsing model is trained with a limited set of multimodal data to recognize and extract diverse information elements from scientific literature. The output is formatted in a custom text format, similar to html format, to effectively represent multimodal elements.
- • LLM SFT: A series of valuable queries are constructed, particularly in multimodal scenarios. With Multimodal Retrieval-Augmented Generation (RAG), relevant content is recalled from the literature based on queries. Answers are built based on the queries and retrieved content. The query-answer data is then used to fine-tune the LLM. This process helps the LLM adapt to the custom input format and improves its ability to follow instructions.
- • User Feedback: The parsing model and SFT-enhanced LLM are deployed in real applications, facilitating the collection of user feedback. Samples receiving positive feedback are subsequently filtered and incorporated into the data enhancement, while those with negative feedback are subject to expert annotation before being integrated into the data enhancement process.
- • Expert Annotation: Samples with negative feedback are carefully annotated by human experts. This step ensures that these models learns from their mistakes, with semi-automated tools assisting in this process to enhance efficiency. Negative feedback cases typically fall into the following categories:

1. Multimodal element recognition errors;

Data Source

Tasks

LLM SFT

###### Multimodal Learning

[Figure 17]

Boost LLM with multimodal RAG Based fine-tuning

[Figure 18]

Capture multimodal

Information Extraction

information in scientific literature

Patents

Uni-SMART

[Figure 19]

[Figure 20]

[Figure 21]

Complex Element Identification

###### Data Enhancement

News

###### User Feedback

Enrich training data, refine model

[Figure 22]

[Figure 23]

Deploy and collect Expert feedback.

Scientific Literature Understanding / Analysis

Scientific Literature

###### Annotation

Annotate samples with negative feedback

[Figure 24]

[Figure 25]

Multimodal Content

Multimodal Understanding / Reasoning

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Market Report

[Figure 30]

Chemical Reactions

Molecules Texts Charts Tables

Figure 2: Uni-SMART model architecture.

- 2. Recall content errors;
- 3. Poor instruction-following.

Detailed analysis of these error types facilitates targeted improvements.

- • Data Enhancement: Finally, the annotated data, along with partial samples that received positive feedback, are added to the training dataset for data enhancement. The pipeline is optimized based on different types of negative feedback:

- 1. Multimodal element recognition errors: expand the training data for the parsing model;
- 2. Recall content errors: optimize the Multimodal RAG scheme;
- 3. Poor instruction-following: enlarge and refine the dataset used for LLM SFT.

This iterative process is repeated to continually optimize Uni-SMART’s overall performance. And significantly enhances Uni-SMART’s performance in a variety of challenging tasks, such as information extraction, complex element identification, scientific literature understanding/analysis, and multimodal understanding/reasoning.

### 3 Evaluation

In this section, we perform a detailed quantitative evaluation on the capabilities of Uni-SMART and various available LLMs across modalities. Table 1 presents the statistics of the benchmarks. For details of the specific evaluation tasks, please refer to SciAssess [10].

For our experiments, different methods are employed depending on the source of the models. Closed-source models are accessed via API calls, while open-source models are obtained from Hugging Face, deployed, and tested. Evaluation tasks that require an article as context involve converting the PDF content to text for input into the LLMs. If an LLM offers a built-in PDF parsing interface, we utilize this feature; otherwise, PyPDF2 is used for PDF parsing. Notably, Uni-SMART can directly read PDF files, allowing us to upload the original documents and pose questions directly to the model. Due to input length limitations of the LLMs, tasks that require the full text of an article as context are executed in a zero-shot manner, whereas tasks that do not require such extensive context are tested using few-shot approaches.

##### Table 1: Statistics of evaluation datasets

Modality Domain Task Question Type Metric # Question

Alloy Materials Composition Extraction Table Extraction Table Recall 244 Drug Discovery Affinity Extraction Table Extraction Table Recall 40

Electrolyte Table QA Multi Choices Accuracy 100 OLED Property Extraction Table Extraction Table Recall 13 Polymer Property Extraction Table Extraction Table Recall 109

Table

Organic Materials

Solubility Extraction Table Extraction Table Recall 100

Alloy Materials Alloy Chart QA Multi Choices Accuracy 15 Biology Biology Chart QA Multi Choices Accuracy 99

Chart

Drug Discovery Drug Chart QA Multi Choices Accuracy 15 Organic Materials Polymer Chart QA Multi Choices Accuracy 15

Affinity Data Extraction Table Extraction Table Recall 40 Tag to Molecule Molecule Generation Molecule Similarity 50

Drug Discovery

Markush to Molecule Molecule Generation Molecule Similarity 37 Molecule in Document YES/NO Accuracy 50

Molecule

Organic Materials OLED Property Extraction Table Extraction Table Recall 13

Drug Discovery Reaction QA Multi Choices Accuracy 95 Organic Materials Reaction Mechanism QA Multi Choices Accuracy 22

Reaction

##### Table 2: Performance Comparison of LLMs on Diverse Tasks.

Modality Task Uni-SMART GPT-4o GPT-4 GPT-3.5 Moonshot Claude3 Doubao Gemini Llama3 DeepSeek Qwen2 Command R+

Composition Extraction 0.511 0.484 0.458 0.112 0.127 0.495 0.304 0.239 0.212 0.389 0.423 0.128 Affinity Extraction 0.200 0.072 0.042 0.025 0.040 0.097 0.050 0.040 0.064 0.017 0.075 0.043 Electrolyte Table QA 0.850 0.890 0.790 0.370 0.670 0.870 0.710 0.880 0.460 0.720 0.620 0.450 OLED Property Extraction 0.490 0.336 0.406 0.201 0.037 0.477 0.259 0.093 0.263 0.292 0.392 0.234 Polymer Property Extraction 0.705 0.692 0.681 0.329 0.705 0.629 0.514 0.606 0.536 0.652 0.636 0.171 Solubility Extraction 0.468 0.435 0.440 0.410 0.363 0.426 0.371 0.397 0.399 0.432 0.400 0.351

Table

Alloy Chart QA 0.933 0.533 0.600 0.333 0.333 0.400 0.467 0.667 0.467 0.333 0.400 0.200 Biology Chart QA 0.616 0.580 0.480 0.390 0.545 0.505 0.480 0.616 0.520 0.545 0.515 0.535 Drug Chart QA 0.600 0.333 0.400 0.067 0.400 0.200 0.533 0.533 0.400 0.400 0.400 0.533 Polymer Chart QA 0.933 0.800 0.667 0.400 0.800 0.467 0.867 0.800 0.867 0.733 0.933 0.800

Chart

Affinity Extraction 0.200 0.072 0.042 0.025 0.040 0.097 0.050 0.040 0.064 0.017 0.075 0.043 Tag to Molecule 0.188 0.040 0.022 0.000 0.016 0.035 0.094 0.169 0.034 0.014 0.000 0.031 Markush to Molecule 0.686 0.642 0.654 0.431 0.504 0.675 0.239 0.526 0.491 0.470 0.379 0.376 Molecule in Document 0.720 0.580 0.700 0.500 0.460 0.480 0.560 0.640 0.680 0.460 0.460 0.460 OLED Property Extraction 0.490 0.336 0.406 0.201 0.037 0.477 0.259 0.093 0.263 0.292 0.392 0.234

Molecule

Reaction QA 0.768 0.705 0.674 0.442 0.253 0.663 0.442 0.305 0.611 0.368 0.442 0.316 Reaction Mechanism QA 0.682 0.545 0.636 0.455 0.545 0.455 0.636 0.727 0.500 0.545 0.591 0.591

Reaction

The final results of these experiments are systematically presented in Table 2, illustrating the performance of each model across the evaluated tasks.

#### 3.1 Table

Tables play a pivotal role in scientific literature, presenting complex data and findings in a highly structured manner and thereby contributing significantly to scientific discovery and the dissemination of knowledge [11]. They facilitate the intuitive display of experimental data and enable the efficient summarization and comparison of research outcomes, becoming an indispensable component of scientific investigation. Consequently, enhancing the capability to understand tables is crucial for the automated processing and analysis of scientific documents. The utilization of table data spans a wide array of applications, such as trend analysis, which can reveal developmental trajectories in research fields, and comparative studies, which can elucidate differences in experimental outcomes under varying research methodologies or conditions [12].

To assess the table understanding capabilities of our model, Uni-SMART, compared to other LLMs, we designed a diverse set of tasks across different domains. These tasks were specifically tailored to evaluate how well each model could interpret and extract information from tables.

In our thorough assessment, the Uni-SMART showcased its exceptional proficiency in understanding and extracting table data from the scientific literature. Among six assessed tasks, it surpassed other models in five and delivered a competitive performance in the remainder. Notably, in the "Composition Extraction" and "Solubility Extraction" tasks, Uni-SMART achieved "Table Recall" of 0.511 and 0.468, respectively, significantly outperforming its counterparts. The objectives of these two tasks are to extract necessary information from tables within articles and organize it into a specified format. These results indicate that Uni-SMART excels in handling and understanding tabular data, particularly in information extraction and formatting tasks.

However, in the Electrolyte Table QA task, Uni-SMART’s performance was slightly below the state-of-the-art model (GPT-4o). In this task, LLMs are required to answer questions about details in the table with the question type of multiple-choice, indicating that Uni-SMART still has room for improvement in understanding the details within tables.

Moreover, all models under-performed in the "Affinity Extraction" task, indicating the high complexity of these tasks. This task requires LLMs to organize molecules and corresponding experimental result from articles. It necessitates the model’s ability to integrate molecules with table information and demands strong long-context understanding capabilities. To better address these issues in the future, a possible direction is to enhance LLMs’ abilities in long context comprehension and information matching.

#### 3.2 Chart

Charts are essential tools in scientific literature, offering a visual representation of data that can significantly enhance the comprehension and communication of complex information. By succinctly illustrating trends, comparisons, and patterns, charts enable researchers to convey their findings more effectively and intuitively [13]. Therefore, the ability to accurately interpret and analyze charts is vital for the automated processing and understanding of scientific documents.

To assess the chart interpretation capabilities of Uni-SMART, we conducted a series of ChartQA tasks spanning various scientific domains. These tasks rigorously evaluated models’ ability to analyze and clarify data presented in charts, with a particular focus on identifying trends and extracting meaningful insights.

Analysis of the results presented in Figure 3 indicates that Uni-SMART significantly outperforms existing models in Chart QA tasks across diverse scientific domains, particularly excelling in the tasks of Alloy Chart QA and Drug Chart QA, with leading scores of 0.933 and 0.600, respectively. However, both in the Biology Chart QA, where the Gemini model scored 0.616, and in the Polymer Chart QA task, where Qwen2 also achieved a top score of 0.933—matching Uni-SMART’s performance in both cases—it is evident that these models have strong capabilities in analyzing scientific charts. These results underscore the intense competition among different models in the context of chart understanding. The future direction for development likely focuses on enhancing accuracy and applicability across various scientific domains.

#### 3.3 Molecule

In scientific literature, molecule or molecular structures hold fundamental importance, especially within fields such as chemistry, pharmacology, and biology. Comprehending molecular structures is crucial for analyzing research outcomes, predicting chemical behaviors, and innovating novel compounds [14, 15]. Deciphering the molecular structure of a newly discovered drug compound, for instance, can provide essential insights into its therapeutic potential and biological interactions, which is crucial for enhancing our understanding of related scientific literature.

To evaluate our model’s ability to understand molecular structures, we carried out a series of tasks involving molecules, polymers, and Markush structures, which are common in chemistry and medicine studies. These tasks were designed to assess the model’s capability in deciphering information from representations of molecular structures.

The Uni-SMART model showed outstanding performance in molecular structure understanding tasks, particularly in processing complex molecular structures within the scientific literature. For example, in the "Molecule in Document" task, Uni-SMART achieved an accuracy of 0.720, significantly outperforming other models. This task requires LLMs to determine whether a molecular structure has appeared in the literature, demonstrating Uni-SMART’s strong capabilities in accurate molecule identification and information retrieval.

Such achievements highlight Uni-SMART’s considerable advantage in parsing molecular structure, potentially attributed to its access to richer, more specialized training data sources, effective preprocessing of molecular structure, and architecture designed specifically for multimodal data. Despite these strengths, there remains room for improvement in Uni-SMART’s comprehension of molecular structure. Future enhancements may include strengthening the model’s understanding of molecule details and expanding the training dataset to encompass a broader range of molecule types. These efforts are expected to further augment Uni-SMART’s applicability in reading and interpreting scientific literature, especially those involving molecular structure.

#### 3.4 Chemical Reactions

Understanding chemical reactions is crucial in scientific literature, particularly within chemistry and its related disciplines [16]. The ability to accurately parse and analyze the details of chemical reactions enhances readers’ grasp of complex material, enabling a deeper understanding of experimental results and theoretical discussions.

[Figure 31]

Figure 3: Comparative Performance of LLMs Across Different Modalities.

To assess Uni-SMART’s capability to comprehend chemical reactions, we designed two tasks aimed at challenging the model’s proficiency in interpreting chemical reactions. These tasks focus on understanding the reactants, products, and conditions of chemical reactions, as well as grasping the underlying mechanisms and clarifying the significance of these reactions within a broader scientific context.

Uni-SMART exhibited outstanding performance in "Reaction QA" task, achieving an accuracy of 0.768. However, in the "Reaction Mechanism QA" task, Uni-SMART’s performance was slightly below that of the Gemini. This task requires LLMs to accurately parse and interpret the step-by-step mechanisms of chemical reactions from the article, indicating that Uni-SMART still has room for improvement in understanding complex reactions.

### 4 Application

In this section, we showcase how Uni-SMART significantly improves the efficiency and quality of scientific research. Through several selected case studies, we will present the specific applications of Uni-SMART in the scientific domain. These cases allow us to qualitatively assess the model’s capabilities in addressing various research challenges. For more details and updates, please refer to the Uni-SMART website at http://uni-smart.dp.tech/.

#### 4.1 Patent infringement analysis

In today’s research and industrial domains, the correct understanding and application of patent information have become increasingly crucial, especially in the fields of chemistry and drug development [17]. Accurately determining whether a compound is protected by existing patents is essential for avoiding potential patent infringements. However, this task presents numerous technical challenges, particularly the need for precise analysis of complex chemical structures and their comparison with descriptions in patent documents. This requires not only a deep understanding of chemical structures but also the ability to process and analyze information from different modalities, such as chemical structure and patent text descriptions.

In this case study, we utilize Uni-SMART to analyze whether a specific chemical structure is protected by a particular patent. The process involves receiving the patent document and compound uploaded by the user, analyzing the molecular skeleton and substituents of the compound, and then comparing this information with the range given in the patent document.

As shown in Figure 4, Uni-SMART analyzes the provided compound structure and accurately determines whether the compound falls within the patent protection scope by considering multiple factors, such as the molecular skeleton and substituents. This process not only demonstrates Uni-SMART’s efficient handling of cross-modal information between chemical structure and patent text but also highlights its strong capability in understanding and applying patent

[Figure 32]

[Figure 33]

Prompt

Response

Is the molecule O=C1N([C@H](COC)C2=CC3=NC([C@@H](NC(C4=NON=C4C )=O)C5CCC(F)(CC5)F)=CN3N=C2)C[C@@H](C(F)(F)F)N1 protected by this patent?

- 1. The molecular skeleton as described in the patent document is:

*C1=NO*C1C(=O)N[C@H](c1cn2ncc(C(*)N3CC(C(F)(F)F)NC3=O)cc2n1) C1CCC(F)(F)CC1<sep><a>0:R[1]</a><a>4:X</a><a>17:R[2]</a>.

- 2. The molecule provided has the same molecular skeleton as the one described in the patent document, fulfilling the requirements for the molecular skeleton specified therein.
- 3. The possible substituents for the structure as given in the original patent text are:

- • X can be CH or N.
- • R1 can be various alkyl or cycloalkyl groups, including but not limited to -CH3, -CH2F, -CHF2, -CF3, -CH2CH3, -CH2CF3, -CH(CH3)2,

-CH2CHF2, -CH2CH2F, -CF(CH3)2, -CF2CH3, -OCH3, multiple cyclopropyl substitutions, and others as listed.

- • R2 can be -H or -CH2OCH3

- 4. The substituents for the provided molecular structure are:

- • R1 is represented as ‘C’, which would correspond to a -CH3 group if we ignore hydrogen atoms in the interpretation.
- • X is ‘N’, clearly indicated in the structure.
- • R2 is ‘COC’ which corresponds to a structure like -CH2OCH3 when hydrogen atoms are ignored.

- 5. Compare the substituents of the molecular structure with the range of substituents given in the patent.

You need to

- 1). analyse the skeletons of the provided molecule and the molecule in the patent,
- 2). compare the substituents of the molecular structure with the range of substituents given in the patent, to determine whether the molecule is protected by the patent.

[Figure 34]

- • The value of X in the provided molecular structure is ‘N’, which is directly listed as one of the options in the patent description.
- • For R1, the provided structure has ‘C’, which, based on the note given, would correspond to a methyl group (-CH3). This is within the list provided in the patent.
- • R2 in the molecular structure is indicated as ‘COC’, which translates to a methoxyethyl group (-CH2OCH3) when considering the note about ignoring hydrogen atoms. This is also explicitly listed as an option in the patent.

[Figure 35]

Given that all substituents in the provided molecular structure fall within the ranges specified in the patent, we can conclude that the provided molecular structure is within the patent protection scope.

Figure 4: Patent infringement analysis.

information. Applying Uni-SMART in practical scenarios like patent infringement protection can help users avoid potential patent risks, which is of significant importance for advancing scientific research and industrial development.

#### 4.2 Chart analysis

In scientific literature, charts serve as a crucial tool for conveying complex data and experimental results. They play an essential role in the deep understanding of technical details by presenting data trends in an intuitive form. However, the effective understanding and analysis of charts in scientific literature encounter significant challenges. These include but are not limited to, the simultaneous parsing of visual elements and textual descriptions within charts, the accurate identification of data points, understanding the scientific processes behind the data, and performing subsequent complex calculations and logical reasoning.

In this case study, Uni-SMART was utilized to analyze a chart showing the temperature control curve during the Metal Injection Molding (MIM) process. As illustrated in Figure 5, Uni-SMART accurately described the temperature control curve of the MIM process, precisely identifying the key data points in the chart. Following user instructions, it successfully converted this information into a tabular format. Furthermore, the model demonstrated its powerful capability in performing mathematical calculations and logical reasoning, especially in calculating the quenching rate.

[Figure 36]

User

[Figure 37]

Outline the temperature control steps.

This diagram illustrates the temperature control curve during the Metal Injection Molding (MIM) process, highlighting three main stages:

- 1. Dewaxing: On the left side of the graph, we see the temperature initially rising over time. This stage typically involves heating the molded parts to remove the wax-based binder. From the starting point to about 35 minutes, the temperature rises from room temperature to approximately 350°C, and then it remains at this temperature until about 65 minutes.
- 2. Sintering: Next, the process enters the sintering phase, beginning around 65 minutes, where the temperature quickly rises to a peak of approximately 500°C and is maintained until about 117 minutes.
- 3. Quenching: After completing the sintering stage, the temperature rapidly drops in a process known as quenching. In the diagram, this stage occurs from approximately 117 minutes to 237 minutes, with the temperature quickly dropping from the sintering temperature to room temperature. This helps the material achieve better mechanical properties. The diagram shows a typical thermal treatment curve for the MIM process, including key steps like dewaxing, sintering, and quenching. The total sintering time is approximately 237 minutes.

Uni-SMART

[Figure 38]

User

Convert the content of this diagram into a table format.

[Figure 39]

The content of this diagram can be transformed into a table that includes the name of each process, the start time, the end time, and the corresponding temperature range. Please note, as the chart does not provide specific start and end temperatures, I will use approximate values. Uni-SMART

[Figure 40]

[Figure 41]

User

What is the rate of the quenching process?

[Figure 42]

To calculate the rate of the quenching process, we need to determine the amount of temperature decrease during this stage and the time taken for this process. From the diagram, it is evident that the temperature drops from the sintering temperature of approximately 500 ◦C to room temperature, which we can assume to be about 20 ◦C. The quenching process starts at approximately 117 minutes and ends at 237 minutes.

Uni-SMART

The formula for calculating the quenching rate is: Quenching Rate =

[Figure 43]

Temperature Change Time Change Based on the chart, the temperature change (∆T ) is 500 ◦C − 20 ◦C = 480 ◦C, and the time change (∆t) is 237 minutes − 117 minutes = 120 minutes.

480 ◦C 120 minutes

Quenching Rate =

The rate of the quenching process is 4 ◦C per minute. This means that during the quenching stage, the temperature drops at a rate of 4

◦C per minute.

Figure 5: Understanding the Metal Injection Molding (MIM) Process.

### 5 Discussion and Conclusion

In this report, we introduce Uni-SMART for in-depth understanding of multimodal information within the scientific literature. Through rigorous quantitative evaluation, Uni-SMART demonstrates significant performance gain in interpreting and analyzing multimodal contents in scientific documents, such as tables, charts, molecular structures, and chemical reactions, compared with other competitors. The success of Uni-SMART lies in its innovative cyclic iterative process that continuously refines its multimodal understanding capabilities, leveraging a robust dataset and a

combination of multimodal learning, supervised fine-tuning, user feedback, expert annotation, and data enhancement to achieve superior performance in scientific literature analysis.

Beyond quantitative assessment, we are particularly excited about Uni-SMART’s potential to address scientific challenges through practical applications. From patent infringement analysis to complex material science chart interpretation, Uni-SMART’s cross-modal understanding capabilities offer new perspectives and tools for research and technological development, showcasing its potential to facilitate research processes and accelerate discovery phases.

Despite Uni-SMART’s strong ability in multimodal scientific literature understanding, we acknowledge that there is still room for improvement. This includes enhancing the model’s understanding of highly complex and specialized content, as well as reducing hallucinations. We believe that through continuous research and development, these limitations will be addressed, making Uni-SMART an even more powerful and flexible tool for scientific research assistance.

In summary, the research and development of Uni-SMART mark a significant advancement in the field of multimodal scientific literature understanding. By providing scientists and researchers with an efficient tool for deep understanding and analysis of scientific documents, Uni-SMART not only facilitates the accumulation and innovation of scientific knowledge but also paves the way for future scientific work, technological development, and potential commercial applications. As we continue to improve and expand Uni-SMART, we look forward to its greater role in promoting scientific discovery and technological innovation.

### References

- [1] Zhi Hong, Logan Ward, Kyle Chard, Ben Blaiszik, and Ian Foster. Challenges and advances in information extraction from scientific literature: a review. JOM, 73(11):3383–3400, 2021.
- [2] Zara Nasar, Syed Waqar Jaffry, and Muhammad Kamran Malik. Information extraction from scientific articles: a survey. Scientometrics, 117:1931–1990, 2018.
- [3] Stephen Walter Gabrielson. Scifinder. Journal of the Medical Library Association: JMLA, 106(4):588, 2018.
- [4] Jonathan Goodman. Computer software review: Reaxys, 2009.
- [5] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971, 2023.
- [6] Gemini Team Google. Gemini: A family of highly capable multimodal models. CoRR, abs/2312.11805, 2023.
- [7] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS, 2020.
- [8] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In NeurIPS, 2022.
- [9] OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023.
- [10] Hengxing Cai, Xiaochen Cai, Junhan Chang, Sihang Li, Lin Yao, Changxin Wang, Zhifeng Gao, Yongge Li, Mujie Lin, Shuwen Yang, Jiankun Wang, Yuqi Yin, Yaqi Li, Linfeng Zhang, and Guolin Ke. Sciassess: Benchmarking llm proficiency in scientific literature analysis, 2024.
- [11] Harsh Desai, Pratik Kayal, and Mayank Singh. Tablex: a benchmark dataset for structure and content information extraction from scientific tables. In Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part II 16, pages 554–569. Springer, 2021.
- [12] Nikola Milosevic, Cassie Gregson, Robert Hernandez, and Goran Nenadic. A framework for information extraction from tables in biomedical literature. International Journal on Document Analysis and Recognition (IJDAR), 22:55–78, 2019.
- [13] Jagadish S Kallimani, KG Srinivasa, and Reddy B Eswara. Extraction and interpretation of charts in technical documents. In 2013 International Conference on Advances in Computing, Communications and Informatics (ICACCI), pages 382–387. IEEE, 2013.
- [14] Christian Blaschke, Lynette Hirschman, and Alfonso Valencia. Information extraction in molecular biology. Briefings in Bioinformatics, 3(2):154–165, 2002.
- [15] Matthew C Swain and Jacqueline M Cole. Chemdataextractor: a toolkit for automated extraction of chemical information from the scientific literature. Journal of chemical information and modeling, 56(10):1894–1904, 2016.
- [16] Jiang Guo, A Santiago Ibanez-Lopez, Hanyu Gao, Victor Quach, Connor W Coley, Klavs F Jensen, and Regina Barzilay. Automated chemical reaction extraction from scientific literature. Journal of chemical information and modeling, 62(9):2035–2045, 2021.
- [17] Srikumaran Melethil. Patent issues in drug development: perspectives of a pharmaceutical scientist-attorney. The AAPS journal, 7:E723–E727, 2005.

