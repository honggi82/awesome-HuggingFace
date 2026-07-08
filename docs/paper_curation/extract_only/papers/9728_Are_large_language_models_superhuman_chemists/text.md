# arXiv:2404.01475v2[cs.LG]1Nov2024

## Are large language models superhuman chemists?

Adrian Mirza 1,2,⋆, Nawaf Alampara 1,⋆, Sreekanth Kunchapu 1,⋆, Martino˜ R´ıos-Garc´ıa 1,3 ⋆, Benedict Emoekabu , Aswanth Krishnan 4, Tanya Gupta 5,6, Mara Schilling-Wilhelmi 1, Macjonathan Okereke 1, Anagha Aneesh 1, Mehrdad Asgari 7, Juliane Eberhardt 8, Amir Mohammad Elahi 9, Hani M. Elbeheiry 10, Mar´ıa Victoria Gil 3, Christina Glaubitz , Maximilian Greiner1, Caroline T. Holick 1,14, Tim Hoffmann 1, 14, Abdelrahman Ibrahim 1, Lea C. Klepsch 1, 14, Yannik K¨oster 1, Fabian Alexander Kreth 11, 12, Jakob Meyer1, Santiago Miret 13, Jan Matthias Peschel 1, Michael Ringleb 1, 14, Nicole Roesner 1, 14, Johanna Schreiber 1, 14, Ulrich S. Schubert 1,2, 10, 14, Leanne M. Stafast 1, 14, Dinga Wonanke 15, Michael Pieler 16,17, Philippe Schwaller 5, 6, and

Kevin Maik Jablonka 1,2, 11, 14,

1Laboratory of Organic and Macromolecular Chemistry (IOMC), Friedrich Schiller University Jena, Humboldtstrasse 10, 07743 Jena, Germany 2Helmholtz Institute for Polymers in Energy Applications Jena (HIPOLE Jena), Lessingstrasse 12-14, 07743 Jena, Germany 3Institute of Carbon Science and Technology (INCAR), CSIC, Francisco Pintado Fe 26, 33011 Oviedo, Spain 4QpiVolta Technologies Pvt Ltd 5Laboratory of Artificial Chemical Intelligence (LIAC), Institut des Sciences et Ing´enierie Chimiques, Ecole Polytechnique F´ed´erale de Lausanne (EPFL), Lausanne, Switzerland 6National Centre of Competence in Research (NCCR) Catalysis, Ecole Polytechnique F´ed´erale de Lausanne (EPFL), Lausanne, Switzerland 7Department of Chemical Engineering & Biotechnology, University of Cambridge, Philippa Fawcett Drive, Cambridge CB3 0AS, United Kingdom 8Macromolecular Chemistry, University of Bayreuth, 95447 Bayreuth, Germany 9Laboratory of Molecular Simulation (LSMO), Institut des Sciences et Ing´enierie Chimiques, Ecole Polytechnique F´ed´erale de Lausanne (EPFL), Sion, Switzerland 10Institute for Inorganic and Analytical Chemistry (IAAC), Friedrich Schiller University Jena, Humboldtstrasse 8, 07743 Jena, Germany 11Center for Energy and Environmental Chemistry Jena (CEEC Jena), Friedrich Schiller University Jena, Philosophenweg 7a, 07743 Jena, Germany 12Institute for Technical Chemistry and Environmental Chemistry (ITUC), Friedrich Schiller University Jena, Philosophenweg 7a, 07743 Jena, Germany 13Intel Labs 14Jena Center for Soft Matter (JCSM), Friedrich Schiller University Jena, Philosophenweg 7, 07743 Jena, Germany 15Theoretical Chemistry, Technische Universit¨at Dresden, Dresden 01062, Germany 16OpenBioML.org 17Stability.AI

mail@kjablonka.com ⋆These authors contributed equally.

November 4, 2024

###### Abstract

Large language models (LLMs) have gained widespread interest due to their ability to process human language and perform tasks on which they have not been explicitly trained.

However, we possess only a limited systematic understanding of the chemical capabilities of LLMs, which would be required to improve models and mitigate potential harm. Here, we introduce “ChemBench,” an automated framework for evaluating the chemical knowledge and reasoning abilities of state-of-the-art LLMs against the expertise of chemists.

We curated more than 2,700 question-answer pairs, evaluated leading openand closed-source LLMs, and found that the best models outperformed the best human chemists in our study on average. However, the models struggle with some basic tasks and provide overconfident predictions.

These findings reveal LLMs’ impressive chemical capabilities while emphasizing the need for further research to improve their safety and usefulness. They also suggest adapting chemistry education and show the value of benchmarking frameworks for evaluating LLMs in specific domains.

### 1 Introduction

Large language models (LLMs) are machine learning (ML) models trained on massive amounts of text to complete sentences. Aggressive scaling of these models has led to a rapid increase in their capabilities,1,2 with the leading models now being able to pass the United States Medical Licensing Examination3 or other professional licensing exams. They also have been shown to design and autonomously perform chemical reactions when augmented with external tools such as web search and synthesis planners.4–7 While some see “sparks of artificial general intelligence (AGI)” in them,8 others consider them as “stochastic parrots”—i.e., systems that only regurgitate what they have been trained on9 and that show inherent limitations due to the way they are trained.10 Nevertheless, the promise of these models is that they have shown the ability to solve a wide variety of tasks they have not been explicitly trained on.11–13

Chemists and materials scientists have quickly caught on to the mounting attention given to LLMs, with some voices even suggesting that “the future of chemistry is language.”14 This statement is motivated by a growing number of reports that use LLMs to predict properties of molecules or materials,2,15–19 optimize reactions,20,21 generate materials,22–25 extract information,26–33 or to even prototype systems that can autonomously perform experiments in the physical world based on commands provided in natural language.5–7

In addition, since a lot—if not most—of the information about chemistry is currently stored and communicated in text, there is a strong reason to believe that there is still a lot of untapped potential in LLMs for chemistry and materials science.34 For instance, most insights in chemical research do not directly originate from data stored in databases but rather from the scientists interpreting the data. Many of these insights are in the form of text in scientific publications. Thus, operating on such texts might be our best way of unlocking these insights and learning from them. This might ultimately lead to general copilot systems for chemists that can provide answers to questions or even suggest new experiments based on vastly more information than a human could ever read.

However, the rapid increase in capabilities of chemical ML models led (even before the recent interest in LLMs) to concerns about the potential for dual use of these technologies, e.g., for the design of chemical weapons.35–40 To some extent, this is not surprising as any technology that, for instance, is used to design non-toxic molecules can also be used inversely to predict toxic ones (even though the synthesis would still require access to controlled physical resources and facilities). Still, it is essential to realize that the user base of LLMs is broader than that of chemistry and materials science experts who can critically reflect on every output these models produce. For

example, many students frequently consult these tools—perhaps even to prepare chemical experiments.41 This also applies to users from the general public, who might consider using LLMs to answer questions about the safety of chemicals. Thus, for some users, misleading information—especially about safety-related aspects—might lead to harmful outcomes. However, even for experts, chemical knowledge and reasoning capabilities are essential as they will determine the capabilities and limitations of their models in their work, e.g., in copilot systems for chemists. Unfortunately, apart from exploratory reports such as by prompting leading models with various scientific questions,13 there is little systematic evidence on how LLMs perform compared to expert (human) chemists.

Thus, to better understand what LLMs can do for the chemical sciences and where they might be improved with further developments, evaluation frameworks are needed to allow us to measure progress and mitigate potential harms systematically. For the development of LLMs, evaluation is currently primarily performed via standardized benchmark suites such as BigBench42 or the LM Eval Harness.43 Among 204 tasks (such as linguistic puzzles), the former contains only two tasks classified as “chemistry related”, whereas the latter contains no specific chemistry tasks. Due to the lack of widely accepted standard benchmarks, the developers of chemical language models16,44–47 frequently utilize language-interfaced48 tabular datasets such as the ones reported in MoleculeNet,49 Therapeutic Data Commons50 or MatBench.51 In these cases, the models are evaluated on predicting very specific properties of molecules (e.g., solubility, toxicity, melting temperature or reactivity) or on predicting the outcome of specific chemical reactions. This, however, only gives a very limited view of the general chemical capabilities of the models.

While some benchmarks based on university entrance exams52,53 or automatic text mining54–56 have been proposed, none of them have been widely accepted. This is likely because they cannot automatically be used with black box (or tool-augmented) systems, do not cover a wide range of topics and skills, or are not carefully validated by experts. On top of that, the existing benchmarks are not designed to be used with models that support special treatment of molecules or equations and do not provide insights on how the models compare relative to experts49.

In this work, we report a novel benchmarking framework (Figure 1), which we call ChemBench, and use it to reveal limitations of current frontier models for use in the chemical sciences. Our benchmark consists of 2788 question-answer pairs compiled from diverse sources (1039 manually generated, and 1749 semi-automatically generated). Our corpus measures reasoning, knowledge and intuition across a large fraction of the topics taught in undergraduate and graduate chemistry curricula. It can be used to evaluate any system that can return text (i.e., including tool-augmented systems).

To contextualize the scores, we also surveyed 19 experts in chemistry on a subset of the benchmark corpus to be able to compare the performance of current frontier

(>2700 total questions)

###### 19 respondents automatically updated 236 diverse questions

Question: What is the number of signals in the 1H NMR spectrum of the molecule on the right?

HO OH Knowledge Reasoning O

0.61

Intuition

Answer:

semantic annotation curation

0.57

[Figure 1]

chembench.org

0.51

peer-reviewed

... Question: What is the number of signals in the 1H NMR spectrum of a molecule with the SMILES [START_SMILES] OCC1C2CC1(O)C2=O[END_SMILES]?

topic leaders overall leaders

closed-source models open-weight models diverse settings

corpusinBIG-benchformat

... Answer:

[Figure 2]

- Figure 1: Overview of the ChemBench framework. The figure shows the different components of the ChemBench framework. The framework’s foundation is the benchmark corpus comprising thousands of questions and answers that we manually or semi-automatically compiled from various sources (see Section 4.1). Questions are classified based on topics, required skills (reasoning, calculation, knowledge, intuition), and difficulty levels. We then used this corpus to evaluate the performance of various models and tool-augmented systems using a custom framework. To provide a baseline, we built a web application that we used to survey experts in chemistry. The results of the evaluations are then compiled in publicly accessible leaderboards (Appendix A.15), which we propose as a foundation for evaluating future models.

models with (human) chemists of different specializations. In parts of the survey, the volunteers were also allowed to use tools such as web search to create a realistic setting.

### 2 Results and Discussion

#### 2.1 Benchmark corpus

To compile our benchmark corpus, we utilized a broad list of sources (see Section 4.1), ranging from completely novel, manually crafted questions over university exams to semi-automatically generated questions based on curated subsets of data in chemical databases. For quality assurance, all questions have been reviewed by at least two scientists in addition to the original curator and automated checks. Importantly, our large pool of questions encompasses a wide range of topics and question types (Figure 2). The topics range from general chemistry to more specialized fields such as inorganic, analytical or technical chemistry. We also classify the questions based on what skills are required to answer them. Here, we distinguish between questions that require knowledge, reasoning, calculation, intuition, or a combination of these.

Moreover, the annotator also classifies the questions by difficulty to allow for a more nuanced evaluation of the models’ capabilities.

[Figure 3]

- Figure 2: Distribution of topics and required skills. This circular plot illustrates the distribution of questions across various chemistry topics, along with the primary skills required to address them. The topics were manually classified, showing a varied representation across different aspects of chemistry. Each topic is associated with a combination of three key skills: Calculation, Reasoning, and Knowledge, as indicated by the colored bars. ChemBench samples diverse topics and diverse skills, setting a high bar for LLMs to demonstrate human-competitive performance across a wide range of chemistry tasks.

While many existing benchmarks are designed around multiple-choice question (MCQ), this does not reflect the reality of chemistry education and research. For this reason, ChemBench samples both MCQ and open-ended questions (2544 MCQ questions and 244 open-ended questions). In addition, ChemBench samples different skills on various difficulty levels: From basic knowledge questions (as knowledge underpins reasoning processes57,58) to complex reasoning tasks (such as finding out which ions are in a sample given a description of observations). We also include questions about chemical intuition, as showing human-aligned preferences is relevant for applications such as hypothesis generation or optimization tasks.59

ChemBench-Mini It is important to note that a smaller subset of the corpus might be more practical for routine evaluations.60 For instance, Liang et al. [61] report costs of more than $10,000 for application programming interface (API) calls for a single evaluation on the widely used Holistic Evaluation of Language Models (HELM) benchmark. To address this, we also provide a subset (ChemBench-Mini, 236 questions) of the corpus that was curated to be a diverse and representative subset of the full corpus. While it is impossible to comprehensively represent the full corpus in a subset, we aimed to include a maximally diverse set of questions and a more balanced distribution of topics and skills (see Section 4.4 for details on the curation process). Our human volunteers answered all the questions in this subset.

#### 2.2 Model evaluation

Benchmark suite design Because the text used in scientific settings differs from typical natural language, many models have been developed that deal with such text in a particular way. For instance, the Galactica model62 uses special encoding procedures for molecules and equations. Current benchmarking suites, however, do not account for such special treatment of scientific information. To address this, ChemBench encodes the semantic meaning of various parts (e.g., chemicals, units, equations) of the question or answer. For instance, molecules represented in simplified molecular input line-entry system (SMILES) are enclosed in [START SMILES][\END SMILES] tags. This allows the model to treat the SMILES string differently from other text. ChemBench can seamlessly handle such special treatment in an easily extensible way because the questions are stored in an annotated format.

Since many widely utilized LLM systems only provide access to text completions (and not the raw model outputs), ChemBench is designed to operate on text completions. This is also important given the growing number of tool-augmented systems that are deemed essential for building chemical copilot systems. Such systems can augment the capabilities of LLMs through the use of external tools such as search APIs or code executors.63–65 In those cases, the LLM that returns the probabilities for various tokens, i.e., text fragments, is only a part of the whole system, and it is not clear how to interpret the probabilities in the context of the whole system. The text completions, however, are the system’s final outputs, which would also be used in a real-world application. Hence, we use them for our evaluations.66

Overall system performance To understand the current capabilities of LLMs in the chemical sciences, we evaluated a wide range of leading models67 on the ChemBench corpus, including systems augmented with external tools. An overview of the results of this evaluation is shown in Figure 3 (all results can be found in Table A3 and Table A4). In this figure, we show the percentage of questions that the models answered correctly. Moreover, we show the worst, best, and average performance of

lowest human scoreaverage human scorehighest human score

| | | | |
|---|---|---|---|
| | | | |

o1

Claude-3.5 (Sonnet)

GPT-4o

Llama-3.1-405B-Instruct

Mistral-Large-2

PaperQA2

Llama-3.1-70B-Instruct

Llama-3.1-8B-Instruct

GPT-3.5 Turbo

0.0 0.2 0.4 0.6 0.8

fraction of correct answers

- Figure 3: Performance of models and humans on ChemBench-Mini. The figure shows the percentage of questions that the models answered correctly. We use horizontal bars to indicate the performance of various models and highlight statistics of the human performance. The evaluation we use here is very strict as it only considers a question answered correctly or incorrectly, partially correct answers are also considered incorrect. Figure A3 provides an overview of the performance of various models on the entire corpus. PaperQA233 is an agentic system that can also search the literature to obtain an answer. We find that the best models outperform all humans in our study when averaged over all questions (even though humans had access to tools such as web search and ChemDraw for a subset of the questions).

the experts in our study, which we obtained via a custom web application (chembench.

org) that we used to survey the experts. Remarkably, the figure shows that the leading LLM, o1, outperforms the best human in our study in this overall metric by almost a factor of two. Many other models also outperform the average human performance. Interestingly, Llama-3.1-405B-Instruct shows performance that is close to the leading proprietary models, indicating that new open-source models can be competitive with the best proprietary models also in chemical settings.

Notably, we find that models are still limited in their ability to answer knowledgeintensive questions (Table A4); that is, they did not memorize the relevant facts.

Our results indicate that this is not a limitation that could be overcome by simple application of retrieval augmented generation (RAG) systems such as PaperQA2. This is likely because the required knowledge cannot easily be accessed via papers (which is the only external knowledge PaperQA2 has access to) but rather by lookup in specialized databases (e.g., PubChem, Gestis), which also the humans in our study used to answer such questions (Figure A17). This indicates that there is still room for improving chemical LLMs by training them on more specialized data sources or integrating them with specialized databases.

In addition, our analysis shows that the performance of models is correlated with their size (see Figure A11). This is in line with observations in other domains but also indicates that chemical LLMs could, to some extent, be further improved by scaling them up.

Performance per topic To obtain a more detailed understanding of the performance of the models, we also analyzed the performance of the models in different subfields of the chemical sciences. For this analysis, we defined a set of topics (see Section 4.5) and classified all questions in the ChemBench corpus into these topics. We then computed the percentage of questions the models or experts answered correctly for each topic and show them in Figure 4. In this spider chart, the worst score for every dimension is zero (no question answered correctly), and the best score is one (all questions answered correctly). Thus, a larger colored area indicates a better performance.

One can observe that this performance varies across models and topics. While general and technical receive relatively high scores for many models, this is not the case for topics such as toxicity and safety or analytical chemistry.

In the subfield of analytical chemistry, the prediction of the number of signals observable in a nuclear magnetic resonance (NMR) spectrum proved difficult even for the best models (e.g., 22 percent correct answers for o1). Importantly, while the human experts are given a drawing of the compounds, the models are only shown the SMILES string of a compound and have to use this to reason about the symmetry of the compound (i.e., to identify the number of diasterotopically distinct protons, which requires reasoning about the topology and structure of a molecule).

These findings also shine an interesting light on the value of textbook-inspired questions. A subset of the questions in ChemBench are based on textbooks targeted at undergraduate students. On those questions, the models tend to perform better than on some of our semi-automatically constructed tasks (see Figure A5). For instance, while the overall performance in the chemical safety topic is low, the models would pass the certification exam according to the German Chemical Prohibition Ordinance based on a subset of questions we sampled from the corresponding question bank (e.g., 71% correct answers for GPT-4, 61% for Claude-3.5 (Sonnet), and 3% for the human experts). While those findings are impacted by the subset of questions we

Analytical Chemistry

o1 Claude-3.5 (Sonnet) GPT-4o Llama-3.1-405B-Instruct PaperQA2 Mistral-Large-2 Llama-3.1-70B-Instruct Llama-3.1-8B-Instruct GPT-3.5 Turbo Human (Average)

General Chemistry

Toxicity/Safety

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

Inorganic Chemistry

Technical Chemistry

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Materials Science

Physical Chemistry

Organic Chemistry

- Figure 4: Performance of the models and humans on the different topics on ChemBench-Mini. The radar plot shows the performance of the models and humans on the different topics of ChemBench-Mini. The performance is measured as the fraction of questions that were answered correctly by the models. The best score for every dimension is one (all questions answered correctly), and the worst is zero (no question answered correctly). A larger colored area indicates a better performance. This figure shows the performance on ChemBench-Mini. The performance of models on the entire corpus is shown in Figure A3.

sampled, the results still highlight that good performance on such question bank or textbook questions does not necessarily translate to good performance on other questions that require more reasoning or are further away from the training corpus.10 The findings also underline that such exams might have been a good surrogate for the general performance of skills for humans, but their applicability in the face of systems that can consume vast amounts of data is up for debate.

We also gain insight into the models’ struggles with chemical reasoning tasks by examining their performance as a function of molecular descriptors. If the model would answer questions after reasoning about the structures, one would expect the performance to depend on the complexity of the molecules. However, we find that the models’ performance does not correlate with complexity indicators (see Appendix A.5). This indicates that the models may not be able to reason about the structures of the molecules (in the way one might expect) but instead rely on the proximity of the molecules to the training data.10

It is important to note that the model performance for some topics, however, is slightly underestimated in the current evaluation. This is because models provided via APIs typically have safety mechanisms that prevent them from providing answers that the provider deems unsafe. For instance, models might refuse to provide answers about cyanides. Statistics of the frequency of such refusals are shown in Table A7. To overcome this, direct access to the model weights would be required, and we strive to collaborate with the developers of frontier models to overcome this limitation in the future. This is facilitated by the tooling ChemBench provides, thanks to which contributors can automatically add new models in an open science fashion.

Judging chemical preference One interesting finding of recent research is that foundation models can judge interestingness or human preferences in some domains.59,68 If models could do so for chemical compounds, this would open opportunities for novel optimization approaches. Such open-ended tasks, however, depend on an external observer defining what interestingness is.69 Here, we posed models the same question Choung et al. [70] asked chemists at a drug company: “Which of the two compounds do you prefer?” (in the context of an early virtual screening campaign setting, see Table A2 for an example). Despite chemists demonstrating a reasonable level of interrater agreement, our models largely fail to align with expert chemists’ preferences. Their performance is often indistinguishable from random guessing, even though these same models excel in other tasks in ChemBench (Table A4). This indicates that using preference tuning for chemical settings is a promising approach to explore in future research.

Confidence estimates One might wonder whether the models can estimate if they can answer a question correctly. If they could do so, incorrect answers would be less problematic.

To investigate this, we prompted66 some of the top-performing models to estimate, on an ordinal scale, their confidence in their ability to answer the question correctly (see Appendix A.12 for details on the methodology and comparison to logit-based approaches).

In Figure 5, we show that for some models, there is no significant correlation between the estimated difficulty and whether the models answered the question correctly or not. For applications in which humans might rely on the models to provide answers with trustworthy uncertainty estimates, this is a concerning observation highlighting the need for critical reasoning in the interpretation of the model’s outputs.34,71 For example, for the questions about the safety profile of compounds, GPT-4 reported a confidence of 1.0 (on a scale of 1–5) for the 1 questions it answered correctly and 4.0 for the 6 questions it answered incorrectly. While, on average, the verbalized confidence estimates from Claude 3.5 seem better calibrated (Figure 5), they

###### GPT-4

GPT-4o

1.00

Average Performance

Average Performance

0.75

0.50

0.25

0.00

fractioncorrect

Claude-3.5 (Sonnet)

Llama-3.1-8B-Instruct

1.00

Average Performance

Average Performance

0.75

0.50

0.25

0.00

1 2 3 4 5

1 2 3 4 5

confidence estimate

- Figure 5: Reliability and distribution of confidence estimates. For this analysis, we used verbalized confidence estimates from the model. We prompted the models to return a confidence score on an ordinal scale to obtain those estimates. The line plot shows the average fraction of correctly answered questions for each confidence level. The bar plot shows the distribution of confidence estimates. A confidence estimate would be well-calibrated if the average fraction of correctly answered questions increases with the confidence level. The dashed black line indicates this ideal behavior, which would be monotonically increasing correctness with higher levels of confidence. We find that most models are not well-calibrated and provide misleading confidence estimates.

are still misleading in some cases. For example, for the questions about the globally harmonized system of classification and labelling of chemicals (GHS) pictograms Claude 3.5 returns an average score of 2.0 for correct answers and 1.83 for incorrect answers.

### 3 Conclusions

On the one hand, our findings underline the impressive capabilities of LLMs in the chemical sciences: Leading models outperform domain experts in specific chemistry questions on many topics. On the other hand, there are still striking limitations. For very relevant topics, the answers that models provide are wrong. On top of that,

many models are not able to reliably estimate their own limitations. Yet, the success of the models in our evaluations perhaps also reveals more about the limitations of the questions we use to evaluate models—and chemists—than about the models themselves. For instance, while models perform well on many textbook questions, they struggle with questions requiring more reasoning about chemical structures (e.g., number of isomers or NMR peaks). Given that the models outperformed the average human in our study, we need to rethink how we teach and examine chemistry. Critical reasoning is increasingly essential, and rote solving of problems or memorization of facts is a domain in which LLMs will continue to outperform humans (when trained on the right training corpus).

Our findings also highlight the nuanced trade-off between breadth and depth of evaluation frameworks. The analysis of model performance on different topics shows that models’ performance varies widely across the subfields they are tested on. However, even within a topic, the performance of models can vary widely depending on the type of question and the reasoning required to answer it.

The current evaluation frameworks for chemical LLMs are primarily designed to measure the performance of the models on specific property prediction tasks. They cannot be used to evaluate reasoning or systems built for scientific applications. Thus, we had little understanding of the capabilities of LLMs in the chemical sciences. Our work shows that carefully curated benchmarks can provide a more nuanced understanding of the capabilities of LLMs in the chemical sciences. Importantly, our findings also illustrate that more focus is required in developing better human-model interaction frameworks, given that models cannot estimate their limitations.

While our findings indicate many areas for further improvement of LLM-based systems, such as for agents (more discussion in Appendix A.11), it is also important to realize that clearly defined metrics have been the key to the progress of many fields of ML, such as computer vision. Although current systems might be far from reasoning like a chemist, our ChemBench framework will be a stepping stone for developing systems that might come closer to this goal.

### 4 Methods

#### 4.1 Curation workflow

For our dataset, we curated questions from existing exams or exercise sheets but also programmatically created new questions (Table 1). Questions were added via Pull Requests on our GitHub repository and only merged into the corpus after passing manual review (Figure 6) as well as automated checks (e.g., for compliance with a standardized schema).

To ensure that the questions do not enter a training dataset, we use the same canary string as the BigBench project. This requires that LLM developers filter their training dataset for this canary string.4,42

- Table 1: Overview of sources of the curated questions. The table provides an overview of the types of sources the questions have been curated from. Detailed sources are available in the source data on GitHub. Questions without a source have been curated completely from scratch. Questions based on lecture notes or URLs have been curated based on content presented in those resources. All questions have been rephrased, annotated, and reviewed before being added to the corpus.

Source Count Semiautomatically generated 1749 URL 375 Textbook 206 Exam 149 IChO 149 No source 139 Lectures 21

Manually curated questions Manually curated questions were sourced from various sources, including university exams, exercises, and question banks.

Semi-programmatically generated questions In addition to the manually curated questions, we also generated questions programmatically. An overview of the sources of the semi-programmatically generated questions is provided in Table 2.

###### Manually curated

###### Reactions

Chemistry olympiads University exams University exercise sheets

[Figure 4]

[START_RXNSMILES] [END_RXNSMILES]

Manual inspection

[Figure 5]

Factual correctness Clarity and phrasing

###### Molecules

Manually created

[Figure 6]

[START_SMILES] [END_SMILES] \ce{C6H6}

###### Semi-programatically curated

Error analysis

GHS pictograms Daily allowed intakes Hazard statements Number of NMR peaks Electron counts IUPAC-SMILES questions Oxidation states Point groups

Automatic checks Schema

###### Units

\pu{m^{-3}}

Invariance to shuffling

Spelling

###### Equations

$a^2 + b^2 = c^2$

- Figure 6: Overview of the workflow for the assembly of the ChemBench corpus. To assemble the ChemBench corpus, we first collected questions from various sources. Some tasks were manually curated, others semi-programmatically. We added semantic annotations for all questions to make them compatible with systems that use special processing for modalities that are not conventional natural text. We reviewed the questions using manual and automatic methods before adding them to the corpus.

- Table 2: Sources of semi-programatically generated questions. The table shows the sources and a brief description as well as the number of the semi-programatically generated questions.

source description question count

Number of isomers MAYGEN72 was used to compute the number of isomers for a set of SMILES extracted from the ZINC dataset73

24

Total electron count of molecules

Electron counts based on the data from https://www.cheminfo.org/

25

Oxidation states Oxidation states questions based on the data from https://www.cheminfo. org/

10

Chemical reactivity Questions are framed based on the information from the Cameo Chemicals website

276

Number of NMR signals

Molecules are sampled from the ZINC database73, OpenChemLib74 is used to compute the number of diasterotopically distinct hydrogen atoms

50

Point group of molecules

Our ChemCaption tool is used to assign the point group using spglib,75 and then each case was manually checked to select well-defined cases

16

IUPAC-SMILES pairs Sampled from the PubChem76 database 10 + 10

Daily allowable intakes according to the World Health Organization

10

PubChem76 safety data

Definitions of hazard statements 10 GHS classification of chemicals mined through the API

7

Materials’ compatibility 20 Chemical compatibility 296

Safety

Chemical preference data These questions assess the ability to establish a “preference”, such as favoring a specific molecule. Chemical preference is of major importance in drug discovery projects, where the optimization process to reach the desired molecular properties is a process that takes several years within a chemist’s career. Our data corpus is adapted from the published dataset by Choung et al. [70], which consists of more than 5000 question-answer pairs about chemical intuition. To build the dataset, they presented 35 medicinal chemists with two different molecules, asking them what molecule they would like to continue with when imaging an early virtual screening campaign setting. The question was designed so the scientists do not spend much time answering it, relying only on their feelings or “chemical preference”.

To understand whether the capabilities of the leading models align with the preferences of professional chemists, we randomly selected 1000 data points from the original dataset to create a meaningful evaluation set, where molecules are represented as SMILES. To ablate the effect of different molecular representations, we only considered questions for which we could obtain International Union of Pure and Applied Chemistry (IUPAC) names for both molecules present.

- 4.2 Model evaluation workflow A graphical overview of the pipeline is shown in Figure A12.

Prompting We employ distinct prompt templates tailored for completion and instruction-tuned models to maintain consistency with the training. As explained below, we impose constraints on the models within these templates to receive responses in a specific format so that robust, fair, and consistent parsing can be performed. Certain models are trained with special annotations and LATEX syntax for scientific notations, chemical reactions, or symbols embedded within the text. For example, all the SMILES representations are encapsulated within [START SMILES][\END SMILES] in Galactica62. Our prompting strategy consistently adheres to these details in a model-specific manner by post-processing LATEX syntax, chemical symbols, chemical equations, and physical units (by either adding or removing wrappers). This step can be easily customized in our codebase, and we provide presets for the models we evaluated.

Parsing Our parsing workflow is multistep and primarily based on regular expressions. Inthe caseofinstruction-tunedmodels, wefirstidentify the[ANSWER][\ANSWER] environment we prompt the model to report the answer in. In the case of completion models, this step is skipped. From there, we attempt to extract the relevant enumeration letters (for multiple-choice questions) or numbers. In the case of numbers, our regular expression was engineered to deal with various forms of scientific notation. As initial tests indicated that models sometimes return integers in the form of words, e.g., “one” instead of “1”, we also implemented a word-to-number conversion using regular expressions. If these hard-coded parsing steps fail, we use a LLM, e.g., Claude-3.5 (Sonnet), to parse the completion (Appendix A.8 provides more details on this step).

Models For all models, we performed inference using greedy decoding (i.e., temperature 0). We used the API endpoints provided by the model developers and those provided by Groq. PaperQA2 was used (in August 2024) via an API provided by FutureHouse.

- 4.3 Confidence estimate

To estimate the models’ confidence, we prompted them with the question (and answer options for MCQ) and the task to rate their confidence to produce the correct answer on a scale from 1 to 5. We decided to use verbalized confidence estimates66 since we found those closer to current practical use cases than other prompting strategies, which might be more suitable when implemented in systems. In addition, this approach captures semantic uncertainty, which is not the same as the probability

of a token being given a sequence of tokens (i.e., the uncertainty one obtains from logit-based approaches). On top of that, many proprietary models do not provide access to the logits, making this approach more general. In Appendix A.12, we provide more details and comparisons with a logit-based approach.

#### 4.4 Human baseline

Question selection Several design choices were made when selecting ChemBenchMini. Firstly, from the full dataset, we kept all the questions labeled as advanced. In this way, we can obtain a deeper insight into the capabilities of LLMs on advanced tasks when compared to actual chemists. Secondly, we sample a maximum of three questions across all possible combinations of categories (i.e., knowledge or reasoning) and topics (e.g., organic chemistry, physical chemistry). Thirdly, we do not include any intuition questions in this subset because the intended use of ChemBench-Mini is to provide a fast and fair evaluation of LLMs independent of any human baseline. In total, 236 questions have been sampled for ChemBench-Mini. Then, this set is divided into two subsets based on the aforementioned combinations. One of the question subsets allows tool use, and the other does not.

Study design Human volunteers were asked the questions in a custom-built web interface (see Appendix A.10), which rendered chemicals and equations. Questions were shown in random order, and volunteers were not allowed to skip questions. For a subset of the questions, the volunteers were allowed to use external tools (excluding other LLM or asking other people) to answer the questions. Prior to answering questions, volunteers were asked to provide information about their education and experience in chemistry. The study was conducted in English.

Human volunteers Users were open to reporting about their experience in chemistry. Overall, 16 did so. Out of those, 2 are beyond a first postdoc, 13 have a master’s degree (and are currently enrolled in Ph.D. studies), and 1 has a bachelor’s degree. For the analysis, we excluded volunteers with less than two years of experience in chemistry after their first university-level course in chemistry.

Comparison with models For the analysis, we treated each human as a model. We computed the topic aggregated averages per human for analyses grouped by topic and then averaged over all humans. The performance metrics reported for models in the main text are computed on the same questions the humans answered. Metrics for the entire corpus are reported in the appendix (Appendix A.4).

#### 4.5 Data annotation

In the curation of our dataset, we manually assigned difficulty levels and required skills to each question. We used the following guidelines for these annotations: calculation is required if answering a question would require the use of a calculator, knowledge is required if answering a question requires non-trivial knowledge of facts (e.g., the H/P statements of chemicals). Reasoning is required if answering a question requires multiple reasoning steps. Basic questions only require those skills up to the high school level. Advanced questions would require an expert multiple minutes up to hours to answer.

### Data and code availability

The code and data for ChemBench are available at https://github.com/lamalaborg/chem-benchandarchivedonZenodounderhttps://zenodo.org/records/14010212. The code for the app for our human baseline study is available at https://github.

com/lamalab-org/chem-bench-app. To ensure reproducibility, this manuscript was generated using the show your work! framework.77 The code to rebuild the paper (including code for all figures and numbers next to which there is a GitHub icon) can be found at https://github.com/lamalab-org/chembench-paper. To facilitate reproduction, some intermediate analysis results are cached at http://dx.

doi.org/10.5072/zenodo.34706.

### Acknowledgements

This work was supported by the Carl Zeiss Foundation, and a “Talent Fund” of the “Life” profile line of the Friedrich Schiller University Jena.

In addition, M.S-W.’s work was supported by Intel and Merck via the AWASES programme.

Parts of A.M.’s work was supported as part of the “SOL-AI” project funded by the Helmholtz Foundation model initative.

K.M.J.is partoftheNFDIconsortium FAIRmatfundedbythe Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – project 460197019.

K.M.J. thanks FutureHouse (a non-profit research organization supported by the generosity of Eric and Wendy Schmidt) for supporting PaperQA2 runs via access to the API. We also thank Stability.AI for the access to its HPC cluster.

M.R.G. and M.V.G. acknowledge financial support from the Spanish Agencia Estatal de Investigaci´on (AEI) through grants TED2021-131693B-I00 and CNS2022135474, funded by MICIU/AEI/10.13039/501100011033 and by the European Union NextGenerationEU/PRTR. M.V.G. acknowledges support from the Spanish National

Research Council (CSIC) through Programme for internationalization i-LINK 2023 (Project ILINK23047).

A.A. gratefully acknowledges financial support for this research by the Fulbright U.S. Student Program, which is sponsored by the U.S. Department of State and German-American Fulbright Commission. Its contents are solely the responsibility of the author and do not necessarily represent the official views of the Fulbright Program, the Government of the United States, or the German-American Fulbright Commission.

M.A. expresses gratitude to the European Research Council (ERC) for evaluating the project with the reference number 101106377 titled “CLARIFIER” and accepting it for funding under the HORIZON TMA MSCA Postdoctoral Fellowships - European Fellowships. Furthermore, M.A. acknowledges the funding provided by UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee (Grant Reference: EP/Y023447/1; Organization Reference: 101106377).

M.R. and U.S.S. thank the “Deutsche Forschungsgemeinschaft” for funding under the regime of the priority programme SPP 2363 “Utilization and Development of Machine Learning for Molecular Applications – Molecular Machine Learning” (SCHU 1229/63-1; project number 497115849).

In addition, we thank the OpenBioML.org community and their ChemNLP project team for valuable discussions. Moreover, we thank Pepe M´arquez for discussions and support and Julian Kimmig for feedback on the web app. In addition, we acknowledge support from Sandeep Kumar with an initial prototype of the web app. We thank Bastian Rieck for developing the LATEX-credit package (https://github.

com/Pseudomanifold/latex-credits) and thank Berend Smit for feedback on an early version of the manuscript.

### Statement of ethical compliance

The authors confirm to have complied with all relevant ethical regulations, according to the Ethics Commission of the Friedrich Schiller University Jena (which decided that study is ethically safe). Informed consent was obtained from all volunteers.

### Conflicts of interest

K.M.J. was a paid consultant for OpenAI (as part of the red teaming network). M.P. is an employee of Stability.AI, and A.M. and N.A. were paid contractors of Stability.AI.

### Author contributions

Mara Schilling-WilhelmiMacjonathan Okereke Anagha AneeshMehrdad AsgariJuliane Eberhardt Amir Mohammad Elahi Hani M. Elbeheiry

Fabian Alexander Krethoster

Abdelrahman Ibrahim LeaC.Klepsch

Sreekanth Kunchapu

Jan Matthias Peschel Michael RinglebNicole C. RoesnerJohanna SchreiberUlrich S. SchubertLeanne M. StafastDinga WonankeMichaelPieler Philippe SchwallerKevin Maik Jablonka

Benedict EmoekabuAswanth Krishnanıos-Garc´ıa TanyaGupta

ıaVictoriaGilChristina GlaubitzMaximilian GreinerCaroline T. HolickTimHoffmann

Nawaf Alampara

JakobMeyerSantiagoMiret

AdrianMirza

MartinoR˜ ´

YannikK¨

Mar´

Conceptualization Data curation Formal analysis

Funding acquisition Investigation Methodology

Project administration Resources Software Supervision

Validation Visualization

Writing – original draft Writing – review & editing

### References

- 1. Brown, T. B. et al. Language Models are Few-Shot Learners. arXiv preprint arXiv:2005.14165 (2020).
- 2. Zhong, Z., Zhou, K. & Mottin, D. Benchmarking Large Language Models for Molecule Prediction Tasks. arXiv preprint arXiv:2403.05075 (2024).
- 3. Kung, T. H. et al. Performance of ChatGPT on USMLE: potential for AI-assisted medical education using large language models. PLoS digit. health 2, e0000198

(2023).

- 4. OpenAI et al. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774 (2024).
- 5. Boiko, D. A., MacKnight, R., Kline, B. & Gomes, G. Autonomous chemical research with large language models. Nature 624, 570–578 (Dec. 20, 2023).
- 6. M. Bran, A. et al. Augmenting large language models with chemistry tools. Nat. Mach. Intell. 6, 525–535 (2024).
- 7. Darvish, K. et al. ORGANA: A Robotic Assistant for Automated Chemistry Experimentation and Characterization. arXiv preprint arXiv:2401.06949 (2024).
- 8. Bubeck, S. et al. Sparks of Artificial General Intelligence: Early experiments with GPT-4. arXiv preprint arXiv:2303.12712 (2023).
- 9. Bender, E. M., Gebru, T., McMillan-Major, A. & Shmitchell, S. On the dangers of stochastic parrots: Can language models be too big? in Proceedings of the 2021 ACM conference on fairness, accountability, and transparency (2021), 610–623.

- 10. McCoy, R. T., Yao, S., Friedman, D., Hardy, M. & Griffiths, T. L. Embers of Autoregression: Understanding Large Language Models Through the Problem They are Trained to Solve. arXiv preprint arXiv:2309.13638 (2023).
- 11. Bommasani, R. et al. On the Opportunities and Risks of Foundation Models. arXiv preprint arXiv:2108.07258 (2021).
- 12. Anderljung, M. et al. Frontier AI regulation: Managing emerging risks to public safety. arXiv preprint arXiv:2307.03718 (2023).
- 13. Microsoft Research AI4Science and Microsoft Azure Quantum. The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4. arXiv preprint arXiv:2311.07361 (2023).
- 14. White, A. D. The future of chemistry is language. Nat. Rev. Chem. 7, 457–458 (May 19, 2023).
- 15. Jablonka, K. M. et al. 14 examples of how LLMs can transform materials science and chemistry: a reflection on a large language model hackathon. Digit. Discov. 2, 1233–1250 (2023).
- 16. Jablonka, K. M., Schwaller, P., Ortega-Guerrero, A. & Smit, B. Leveraging large language models for predictive chemistry. Nat. Mach. Intell. 6, 161–169 (2024).
- 17. Xie, Z. et al. Fine-tuning GPT-3 for machine learning electronic and functional properties of organic molecules. Chem. Sci. 15, 500–510 (2024).
- 18. Liao, C., Yu, Y., Mei, Y. & Wei, Y. From Words to Molecules: A Survey of Large Language Models in Chemistry. arXiv preprint arXiv:2402.01439 (2024).
- 19. Zhang, D. et al. ChemLLM: A Chemical Large Language Model. arXiv preprint arXiv:2402.06852 (2024).
- 20. Ramos, M. C., Michtavy, S. S., Porosoff, M. D. & White, A. D. Bayesian Optimization of Catalysts With In-context Learning. arXiv preprint arXiv:2304.05341

- (2023).

21. Kristiadi, A. et al. A Sober Look at LLMs for Material Discovery: Are They Actually Good for Bayesian Optimization Over Molecules? arXiv preprint arXiv:2402.05015

- (2024).

- 22. Rubungo, A. N., Arnold, C., Rand, B. P. & Dieng, A. B. Llm-prop: Predicting physical and electronic properties of crystalline solids from their text descriptions. arXiv preprint arXiv:2310.14029 (2023).
- 23. Flam-Shepherd, D. & Aspuru-Guzik, A. Language models can generate molecules, materials, and protein binding sites directly in three dimensions as XYZ, CIF, and PDB files. arXiv preprint arXiv:2305.05708 (2023).
- 24. Gruver, N. et al. Fine-Tuned Language Models Generate Stable Inorganic Materials as Text. arXiv preprint arXiv:2402.04379 (2024).

- 25. Alampara, N., Miret, S. & Jablonka, K. M. MatText: Do Language Models Need More than Text & Scale for Materials Modeling? arXiv preprint arXiv:2406.17295

(2024).

- 26. Patiny, L. & Godin, G. Automatic extraction of FAIR data from publications using LLM. ChemRxiv preprint doi:10.26434/chemrxiv-2023-05v1b-v2 (2023).
- 27. Dagdelen, J. et al. Structured information extraction from scientific text with large language models. Nat. Commun. 15 (2024).
- 28. Zheng, Z. et al. Image and data mining in reticular chemistry powered by GPT-4V. Digit. Discov. 3, 491–501 (2024).
- 29. L´ala, J. et al. PaperQA: Retrieval-Augmented Generative Agent for Scientific Research. arXiv preprint arXiv:2312.07559 (2023).
- 30. Caufield, J. H. et al. Structured Prompt Interrogation and Recursive Extraction of Semantics (SPIRES): a method for populating knowledge bases using zero-shot learning. Bioinformatics 40 (ed Wren, J.) (2024).
- 31. Gupta, T., Zaki, M., Krishnan, N., et al. DiSCoMaT: distantly supervised composition extraction from tables in materials science articles. arXiv preprint arXiv:2207.01079 (2022).
- 32. Schilling-Wilhelmi, M. et al. From Text to Insight: Large Language Models for Materials Science Data Extraction. arXiv preprint arXiv:2407.16867 (2024).
- 33. Skarlinski, M. D. et al. Language agents achieve superhuman synthesis of scientific knowledge. arXiv preprint arXiv:2409.13740 (2024).
- 34. Miret, S. & Krishnan, N. Are LLMs Ready for Real-World Materials Discovery? arXiv preprint arXiv:2402.05200 (2024).
- 35. Gopal, A. et al. Will releasing the weights of future large language models grant widespread access to pandemic agents? arXiv preprint arXiv:2310.18233 (2023).
- 36. Ganguli, D. et al. Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned. arXiv preprint arXiv:2209.07858 (2022).
- 37. Urbina, F., Lentzos, F., Invernizzi, C. & Ekins, S. Dual use of artificial-intelligencepowered drug discovery. Nat. Mach. Intell. 4, 189–191 (Mar. 7, 2022).
- 38. Campbell, Q. L., Herington, J. & White, A. D. Censoring chemical data to mitigate dual use risk. arXiv preprint arXiv:2304.10510 (2023).
- 39. Moulange, R., Langenkamp, M., Alexanian, T., Curtis, S. & Livingston, M. Towards ResponsibleGovernanceofBiologicalDesignTools.arXivpreprint arXiv:2311.15936

(2023).

- 40. Urbina, F., Lentzos, F., Invernizzi, C. & Ekins, S. A teachable moment for dual-use. Nat. Mach. Intell. 4, 607–607 (July 12, 2022).

- 41. Intelligent.com. One-third of college students used CHATGPT for schoolwork during the 2022-23 academic date https://www.intelligent.com/one-third-of-collegestudents-used-chatgpt-for-schoolwork-during-the-2022-23-academic-date/. Oct. 2023.
- 42. Srivastava, A. et al. Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615 (2022).
- 43. Gao, L. et al. A framework for few-shot language model evaluation version v0.4.0. Dec. 2023. https://zenodo.org/records/10256836.
- 44. Guo, T. et al. What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks. arXiv preprint arXiv:2305.18365 (2023).
- 45. Ahmad, W., Simon, E., Chithrananda, S., Grand, G. & Ramsundar, B. ChemBERTa2: Towards Chemical Foundation Models. arXiv preprint arXiv:2209.01712 (2022).
- 46. Cai, X. et al. Comprehensive evaluation of molecule property prediction with ChatGPT. Methods 222, 133–141 (Feb. 2024).
- 47. Frey, N. C. et al. Neural scaling of deep chemical models. Nat. Mach. Intell. 5, 1297–1305 (Oct. 23, 2023).
- 48. Dinh, T. et al. Lift: Language-interfaced fine-tuning for non-language machine learning tasks. Adv. Neur. In. 35, 11763–11784 (2022).
- 49. Wu, Z. et al. MoleculeNet: a benchmark for molecular machine learning. Chem. Sci. 9, 513–530 (2018).
- 50. Huang, K. et al. Therapeutics data commons: Machine learning datasets and tasks for drug discovery and development. arXiv preprint arXiv:2102.09548v2

(2021).

- 51. Dunn, A., Wang, Q., Ganose, A., Dopp, D. & Jain, A. Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. npj Comp. Mater. 6 (Sept. 2020).
- 52. Zaki, M., Jayadeva, Mausam & Krishnan, N. M. A. MaScQA: investigating materials science knowledge of large language models. Digit. Discov. 3, 313–327

(2024).

- 53. Arora, D., Singh, H. G. & Mausam. Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark For Large Language Models. arXiv preprint arXiv:2305.15074 (2023).
- 54. Song, Y., Miret, S., Zhang, H. & Liu, B. HoneyBee: Progressive Instruction Finetuning of Large Language Models for Materials Science in Findings of the Association for Computational Linguistics: EMNLP 2023 (Association for Computational Linguistics, 2023).

- 55. Wei, Z.et al. ChemistryQA: A Complex Question Answering Dataset from Chemistry

2021. https://openreview.net/forum?id=oeHTRAehiFF.

- 56. Song, Y., Miret, S. & Liu, B. MatSci-NLP: Evaluating Scientific Language Models on Materials Science Language Tasks Using Text-to-Schema Modeling in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (eds Rogers, A., Boyd-Graber, J. & Okazaki, N.) (Association for Computational Linguistics, Toronto, Canada, July 2023), 3621–3639.
- 57. Hu, X. et al. Towards Understanding Factual Knowledge of Large Language Models in The Twelfth International Conference on Learning Representations (2024). https://openreview.net/forum?id=9OevMUdods.
- 58. Bloom, B. Taxonomy of Educational Objectives: The Classification of Educational Goals Taxonomy of Educational Objectives: The Classification of Educational Goals v. 1. isbn: 9780679302094 (Longmans, Green, 1956).
- 59. Zhang, J., Lehman, J., Stanley, K. & Clune, J. OMNI: Open-endedness via Models of human Notions of Interestingness. arXiv preprint arXiv:2306.01711 (2024).
- 60. Polo, F. M. et al. tinyBenchmarks: evaluating LLMs with fewer examples. arXiv preprint arXiv:2402.14992 (2024).
- 61. Liang,P.etal.Holistic EvaluationofLanguageModels.arXivpreprint arXiv:2211.09110

(2022).

- 62. Taylor, R. et al. Galactica: A Large Language Model for Science. arXiv preprint arXiv:2211.09085 (2022).
- 63. Schick, T. et al. Toolformer: Language models can teach themselves to use tools. Adv. Neur. In. 36 (2024).
- 64. Karpas, E. et al. MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning. arXiv preprint arXiv:2205.00445 (2022).
- 65. Yao, S. et al. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629 (2022).
- 66. Xiong, M. et al. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. arXiv preprint arXiv:2306.13063 (2023).
- 67. Beeching, E.etal. OpenLLMLeaderboard https://huggingface.co/spaces/HuggingFaceH4/open llm

leaderboard. 2023.

- 68. Argyle, L. P. et al. Out of One, Many: Using Language Models to Simulate Human Samples. Polit. Anal. 31, 337–351 (2023).
- 69. Hughes, E. et al. Open-Endedness is Essential for Artificial Superhuman Intelligence. arXiv preprint arXiv:2406.04268 (2024).

- 70. Choung, O.-H., Vianello, R., Segler, M., Stiefl, N. & Jim´enez-Luna, J. Extracting medicinal chemistry intuition via preference machine learning. Nat. Commun.

14. http://dx.doi.org/10.1038/s41467-023-42242-1 (2023).

- 71. Li, B. et al. Trustworthy AI: From Principles to Practices. ACM Comput. Surv. 55, 1–46 (Jan. 16, 2023).
- 72. Yirik, M. A., Sorokina, M. & Steinbeck, C. MAYGEN: an open-source chemical structure generator for constitutional isomers based on the orderly generation principle. J. Cheminf. 13 (July 3, 2021).
- 73. Irwin, J. J., Sterling, T., Mysinger, M. M., Bolstad, E. S. & Coleman, R. G. ZINC: A Free Tool to Discover Chemistry for Biology. J. Chem. Inf. Model. 52, 1757–1768 (June 2012).
- 74. Actelion. OpenChemLib https://github.com/actelion/openchemlib.
- 75. Togo, A., Shinohara, K. & Tanaka, I. spglib: a software library for crystal symmetry search. arXiv preprint arXiv:1808.01590 (2018).
- 76. Kim, S. et al. PubChem 2023 update. Nucleic Acids Res. 51, D1373–D1380 (Oct. 28, 2022).
- 77. Luger, R. et al. Mapping stellar surfaces III: An Efficient, Scalable, and OpenSource Doppler Imaging Model. arXiv preprint arXiv:2110.06271 (2021).

### A Appendix

#### A.1 Desired properties of a chemistry benchmark

- • End-to-end automation. For model development, the evaluations must be run many times (e.g., on regular intervals of a training run). Approaches that rely on humans scoring the answers of a system1–3 can thus not be used.
- • Careful validation by experts. Manual curation is needed to minimize the number of incorrect or unanswerable questions.4 This is motivated by the observation that many widely used benchmarks are plagued by noisiness.5,6
- • Usable with models that support special treatment of molecules. Some models, such as Galactica7, use special tokenization or encoding procedures for molecules or equations. The benchmark system must encode the semantic meaning of various parts of the question or answer to support this.
- • Usable with black box systems. Many relevant systems do not provide access to model weights or raw logits. This might be the case because the systems are proprietary or because they involve not only LLMs but also external tools such as search APIs or code executors.8–10 Thus, a benchmark should not assume access to the raw model outputs but be able to operate on text completions.
- • Probing capabilities beyond answering of MCQs. In real-world chemistry, as well as higher-level university education, multiple-choice questions are seldom utilized. Yet, most benchmarking frameworks focus on the MCQ setting because of the ease of evaluation. Realistic evaluations must measure capabilities beyond answering MCQ.
- • Cover a diverse set of topics. Chemistry, as the “central science”, bridges multiple disciplines.11 To even just approximate “chemistry capabilities”, the topics covered by a chemistry benchmark must be very diverse.
- • Cover diverse skills. To holistically judge performance, it is important to cover questions that rely on reasoning, calculation, knowledge, and intuition.
- • Cover a range of difficulty levels. To allow for a continuous measure of improvement for a range of different (evolving) systems, a benchmark should cover a wide range of difficulty levels.
- • Impossible to completely solve with current models. A benchmark should contain questions that are impossible to solve with current models. The benchmark provides no useful signal if current models can solve all questions.

#### A.2 Related work

Existing benchmarks such as those from Guo et al. [12], Sun et al. [13], Schulze Balhorn et al. [1], Cai et al. [14], Rein et al. [15] fail to comply with most of the requirements stipulated above. While these benchmarks could provide valuable insights in the short term, they cannot follow the rapid additions to the LLM space. ChemBench aims to correct this through a set of developments: compatibility with BigBench, end-to-end automation, a particular focus on chemical safety, employment of diverse prompting strategies, and specialized notation for molecules and mathematical symbols. Moreover, our robust framework, including the platform chembench.org, will engage the community in open-source contributions.

#### A.3 Benchmark corpus

To ensure maximal interoperability with existing benchmarks or tools, we curated the data in an extended form of the widely used BigBench format.16 This also implies that future baselines can be built on top of our infrastructure if saved in the same format.

###### A.3.1 Curation workflow

Questions were added via pull requests to the ChemBench repository on GitHub. This allowed for a manual review of each question by expert reviewers (with backgrounds in chemistry, materials science, chemical engineering, and physics). The reviews were conducted directly on the GitHub platform, where our entire curation history is also publicly available.

The general guidelines followed by the reviewer are the following:

- • Originality: Questions should not be readily findable online or in other easily accessible sources (example https://github.com/lamalab-org/chembench/pull/392#discussion_r1694299474)
- • Ambiguity: Questions with unclear wording or multiple interpretations (example https://github.com/lamalab-org/chem-bench/pull/420#discussion_ r1698147159 )
- • Factual or heuristic Errors: Questions containing factual inaccuracies or misconceptions are not included (example https://github.com/lamalab-org/ chem-bench/pull/389#discussion_r1686187301)
- • Clarity and Difficulty: They should pose a challenge and encourage exploration within the chemical domain (example https://github.com/lamalab-org/ chem-bench/pull/391#discussion_r1679276714)
- • Out of Scope: Questions outside the realm of chemistry are rejected.
- • Contribution to Dataset Diversity: Questions should cover a wide range of chemical concepts and sub-disciplines. They should add value by expanding the breadth of the dataset. That is, questions already multiple (¿10) times in the corpus in a similar form are rejected.

Reviewers also solved the questions to verify the answers. They also performed web searches to ensure questions were not easily found online. The reviewers often guided the revision process to ensure the question aligned with the guidelines. Questions that don’t meet the criteria are either rejected or suggested for revision, and

most often, they are modified to a new question. Reviewers also provide feedback on the skill and difficulty annotations.

In addition to the manual review, we also performed automated checks to ensure

the quality of the questions. The schemas, LATEX templating, and other formatting aspects are checked automatically using GitHub Actions.

While adding questions from existing benchmarks might seem to be another good source of semi-automatically generated data, we prioritized the diversity of the data and avoided data contamination while addressing the guidelines above and in Appendix A.1. However, even though we decided not to include questions from other previously published chemistry-focused benchmarks into the ChemBench corpus, the framework is flexible enough to be readily extended with questions from other benchmarks.

###### A.3.2 Composition

Figure A1 shows the distribution of topics and required skills in the human subset of the ChemBench corpus.

The corpus of the questions in ChemBench, as shown in Table A1, can be divided according to which chemical topic they belong.

- Table A1: Examples for each of the topics considered in the evaluation of the ChemBench corpus. The table shows the percentage of questions in the corpus that belong to each topic, as well as example questions.

###### Analytical Chemistry 148 Questions (38.51%)

Which of the following analytical methods is most appropriate for performing a survey analysis of a solid sample containing various metals?

- A. X-ray fluorescence analysis
- B. Differential pulse polarography
- C. Flame-atomic absorption spectroscopy
- D. Gas chromatography with flame ionization detector
- E. Hydride generation atomic absorption spectroscopy Chemical Preference 1001 Questions (51.15%)

Imagine an early virtual screening campaign setting (accounting for simple aspects such as oral availability and small molecular profile, but no other modalities such as covalency or bifunctionality). Which of the following two candidates would you prefer for further development?

- A. [START SMILES]N#Cc1ccc(OCCCN2CC3CN(CCNS(=O)(=O)c4ccc(F)cc4)CC(C2)O3)cc1[END SMILES]

- B. [START SMILES]O=C1CC(c2ccc(CC(NS(=O)(=O)c3cc(Cl)cc(Cl)c3)c3nc4ccccc4[nH]3)cc2)S(=O)(=O)N1[END SMILES]

General Chemistry 150 Questions (50.67%) Which of the following salts is an acidic salt?

- A. NH4Cl
- B. Na2CO3
- C. NaH2PO4
- D. Zn(OH)Cl Inorganic Chemistry 101 Questions (56.44%)

What is the oxidation number of the metal in the compound [ZrF7]3–?

Materials Science 10 Questions (30.00%)

For NMR analysis, you need to digest the MOF in a strong acid to remove the linker and leave the metal clusters intact. Why would one choose HF over HCl for this purpose?

- A. F– forms a stable bonds to the metal ions
- B. HF has a better water solubility than HCl
- C. HF has a higher boiling point than HCl
- D. HF is a weaker acid than HCl Organic Chemistry 334 Questions (59.58%)

What is the reaction mechanism that describes the following reaction (represented using reaction SMILES) [START RXNSMILES]CCCl.CO[Na]¿¿[Na]Cl.CCOC[END RXNSMILES]?

- A. E1
- B. Ecb
- C. SN1
- D. SN2 Physical Chemistry 142 Questions (53.52%)

The Born-Oppenheimer (BO) approximation is widely used in computational chemistry, but its accuracy can vary depending on the system. Among the following options, for which system is the Born-Oppenheimer approximation likely to be least applicable?

- A. C60
- B. CH4
- C. Fe(CO)5
- D. H2+
- E. NaCl

Technical Chemistry 40 Questions (47.50%) Which of the following statements is true about the different types of ideal reactors?

- A. In a batch reactor, the composition is uniformly mixed and remains the same throughout the reactor and at the exit
- B. In a batch reactor, the fluid passes through the reactor with no mixing of earlier and later entering fluid
- C. In a mixed flow reactor, the composition changes with time but is uniform everywhere within the reactor
- D. In a plug flow reactor, the fluid moves in single flow through the reactor with no mixing and no overtaking

Toxicity/Safety 675 Questions (33.19%)

Pindolol and propranolol are (relatively nonselective) antagonists at β1- and β2adrenoceptors. However, pindolol is a partial agonist, whereas propranolol is a pure antagonist. What follows from this?

- A. Pindolol has a greater therapeutic range than propranolol
- B. Pindolol has a longer half-life than propranolol
- C. Pindolol has intrinsic activity
- D. Pindolol is more lipophilic than propranolol
- E. Pindolol is more potent than propranolol

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

- Figure A1: Composition of the human subset. The circular plot shows the distribution of topics and required skills in the human subset of the ChemBench corpus. The human subset is a representative subset of the full corpus, with a balanced distribution of topics and skills.

[Figure 24]

###### Figure A2: Composition of the required skills considered in the ChemBench corpus. The circular plot shows the distribution of required skills in the ChemBench corpus.

In addition, as shown in Figure A2 the ChemBench corpus can be divided considering the different skills needed to solve the questions. The plot shows a balanced distribution of the required skills in the ChemBench corpus. Table A2 shows an example question for each skill category.

- Table A2: Examples for each required skill considered in the ChemBench corpus. The table shows the number of questions for each skill and an example question. Note that the total count in this table is bigger than the ChemBench corpus. This is because the same question can be annotated with two different skills, e.g., Reasoning and Calculation.

Knowledge 886 Questions Which of the following salts is an acidic salt?

- A. NH4Cl
- B. Na2CO3

- C. NaH2PO4
- D. Zn(OH)Cl Reasoning 955 Questions

You make the following experimental observations: The sample consisted of small colorless/white crystals, wine-red transparent crystals, turquoise transparent crystals, and violet flakes. Most components dissolved in cold water, with turquoise crystals dissolving upon heating. Dimethylglyoxime produced a raspberry-red color with dissolved turquoise crystals. Violet flakes turned blue on contact and showed red/violet colors in flame tests. An oxidation melt turned green. No red coloration occurred with KSCN, but a blue organic phase formed with acetone. Potassium ferrocyanide showed no changes. KOH produced a brown/black precipitate. Alizarin-S testing resulted in red coloration after acidification. Heating with cobalt nitrate produced a bluish color without luster. Lead dioxide and sulfuric acid oxidation formed a violet solution. No changes occurred in chromate oxidation or titanium detection attempts. BaCl2 formed precipitates in both original solution and soda extract. Lunge’s reagent showed red coloration around zinc granules. Silver nitrate formed white precipitates that dissolved in NH3 and reappeared with HNO3 acidification. The separation procedure began with the aqueous solution being treated with (NH4)2CO3 solution until a persistent turbidity appeared, followed by addition of dilute HCl until the turbidity disappeared. The pH was then adjusted to 5-6 with NH3 (2 M) and heated on a water bath. Three spatula tips of urotropin were added, resulting in a white, rather flocculent precipitate. This precipitate was centrifuged off and washed with warm water. In the Urotropin group, the residue was dissolved in dilute HCl and diluted with water. An alkaline precipitation was performed, resulting in a brown coloration even after repeated washing of the urotropin precipitation. This precipitate dissolved in 2 M HCl, but showed no change when treated with either KSCN or potassium ferrocyanide. In the (NH4)2S group, addition of dimethylglyoxim produced a very flocculent red precipitate that rose to the top of the test tube. When H2S was introduced to the centrifugate, a black precipitation occurred. The precipitates did not completely dissolve in HCl. Treatment with KSCN resulted in a blue organic phase when layered with acetone. An alkaline precipitation produced a brown/black residue. No precipitation was observed when H2S was introduced to the supernatant liquid after the alkaline precipitation. What ions are in the sample?

- A. Al3+
- B. Co2+
- C. Cr3+
- D. Fe3+
- E. Mn2+
- F. Ni2+

- G. Zn2+ Intuition 1001 Questions

Imagine an early virtual screening campaign setting (accounting for simple aspects such as oral availability and small molecular profile, but no other modalities such as covalency or bifunctionality). Which of the following two candidates would you prefer for further development?

- A. [START SMILES]CC1(C)Oc2ccc([N+](=O)[O])cc2[C@@H](N2CCOCC2)[CH]1O[END SMILES]

- B. [START SMILES]Cc1ccccc1N=C(S)N1CCC(NC(=O)c2ccco2)CC1[END SMILES] Calculation 118 Questions

Given that the average molar mass of the polymer chains in this sample of poly(lactic acid) (PLA) is 595gmol−1 using end-group analysis, where 0.1619g of PLA was dissolved in 25cm3 of benzyl alcohol and titrated with 0.0400moldm−3 KOH solution. The volume of KOH solution required to reach the endpoint was 6.81cm3. What is the average number of monomer units in each polymer chain of this sample?

#### A.4 Model performance

###### We also evaluated the model performance on the entire ChemBench corpus. Figure A3 shows the fraction of questions the models answered correctly.

o1

Claude-3.5 (Sonnet)

GPT-4o

Llama-3.1-405B-Instruct

Mistral-Large-2

PaperQA2

Llama-3.1-70B-Instruct

Llama-3.1-8B-Instruct

GPT-3.5 Turbo

0.0 0.2 0.4 0.6

fraction of correct answers

- Figure A3: Overall performance of the models on the ChemBench corpus. The bar plot shows the fraction of questions the models answered correctly. Scores computed on the entire ChemBench corpus.

Figure A4 shows the performance of the models on the different topics of the ChemBench corpus. The general pattern of performance varies significantly between the different topics and is also observed when the models are evaluated on the entire corpus. However, since some subjects are composed of questions from different sources, the ranking of the models is, in some instances, different from the one on ChemBench-Mini.

To further investigate the performance of the models, we also compared the performance on different data sources. Compared to topics, this is a more fine-grained analysis, as topics can be composed of questions from different sources. In Figure A5, we see that the performance of the models varies significantly between the different data sources. Interestingly, the performance of the models on questions sourced based on textbooks seems to be better for our models than some semi-programmatically created tasks, such as questions about the number of signals in an NMR spectrum.

- Figure A6 shows the same analysis on ChemBench-Mini.
- Figure A7 shows the performance of the models on the ChemBench corpus and

the ChemBench-Mini subset. The relative performance difference between both is quite similar across most models. This makes ChemBench-Mini subset a reliable

Analytical Chemistry

Chemical Preference

Toxicity/Safety o1 Claude-3.5 (Sonnet) GPT-4o Llama-3.1-405B-Instruct Mistral-Large-2 PaperQA2 Llama-3.1-70B-Instruct Llama-3.1-8B-Instruct GPT-3.5 Turbo

| |
|---|

| |
|---|

| |
|---|

General Chemistry

Technical Chemistry

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Inorganic Chemistry

Physical Chemistry

| |
|---|

Organic Chemistry

Materials Science

- Figure A4: Performance of the models on the different topics of the ChemBench corpus. The radar plot shows the performance of the models on the different topics of the ChemBench corpus. The performance is measured as the fraction of questions answered correctly by the models. A score of 1 (full coverage until the outer line of this plot) indicates that all questions were answered correctly, while a score of 0 indicates that none were answered correctly.

subset for human baseline comparison and particularly valuable for rapid prototyping and initial model assessment phases.

An interesting observation is the significant impact of chemical preference tasks on GPT-4’s scores. A detailed breakdown of overall accuracy into scores on different skills and difficulty levels is provided in Table A3 and Table A4 for ChemBench corpus and the ChemBench-Mini subset, respectively.

- Table A3: Performance of the models on the ChemBench corpus. The table shows the fraction of questions answered correctly by the models for different skills and difficulty levels. Models with “T-one” in the name were run for a temperature of 1, which allows us to study the temperature effect in the benchmark. Systems with “ReAct” in the name are tool augmented, i.e., they can call external tools such as web search or a calculator to answer the questions better. However, we limit those systems to ten calls to the LLM. This constraint often led the systems to not find the correct answer within the specified number of calls. In this case, we consider the answer as incorrect (see Appendix A.11).

###### Requires Difficulty

Overall Accuracy Calculation Knowledge Reasoning Intuition Basic Intermediate Advanced

Model

Claude-2 0.31 0.37 0.54 0.51 0.63 0.41 0.36 0.47

Claude-3 (Opus) 0.59 0.47 0.67 0.57 0.77 0.50 0.43 0.57 Claude-3.5 (Sonnet) 0.65 0.53 0.77 0.58 0.83 0.55 0.64 0.63

Claude-3.5 (Sonnet) React 0.71 0.50 0.77 0.60 0.81 0.54 0.75 0.62 Command-R+ 0.19 0.35 0.50 0.51 0.55 0.40 0.20 0.45 Galactica-120b 0.02 0.02 0.03 0.00 0.04 0.00 0.00 0.02

Gemini-Pro 0.25 0.36 0.51 0.50 0.58 0.40 0.35 0.45

- Gemma-1.1-7B-it 0.20 0.24 0.36 0.00 0.43 0.09 0.10 0.19 Gemma-1.1-7B-it-T-one 0.19 0.23 0.36 0.01 0.44 0.10 0.14 0.19
- Gemma-2-9B-it 0.40 0.36 0.52 0.55 0.61 0.43 0.38 0.48 Gemma-2-9B-it-T-one 0.37 0.35 0.51 0.56 0.61 0.43 0.43 0.48 GPT-3.5 Turbo 0.25 0.35 0.53 0.53 0.60 0.42 0.36 0.47 GPT-4 0.48 0.46 0.65 0.16 0.74 0.28 0.57 0.41

GPT-4o 0.72 0.51 0.73 0.59 0.81 0.53 0.62 0.61 GPT-4o React 0.54 0.44 0.67 0.42 0.73 0.41 0.60 0.51

- Llama-2-70B Chat 0.10 0.14 0.15 0.49 0.20 0.30 0.00 0.27
- Llama-3-8B-Instruct 0.25 0.37 0.50 0.52 0.55 0.41 0.52 0.46

Llama-3-8B-Instruct-T-one 0.23 0.36 0.51 0.52 0.55 0.42 0.57 0.46

Llama-3-70B-Instruct 0.47 0.42 0.60 0.53 0.69 0.45 0.29 0.52 Llama-3-70B-Instruct-T-one 0.46 0.42 0.59 0.53 0.70 0.45 0.29 0.52

Llama-3.1-8B-Instruct 0.34 0.36 0.52 0.53 0.61 0.41 0.38 0.47 Llama-3.1-8B-Instruct-T-one 0.35 0.35 0.50 0.52 0.59 0.41 0.64 0.46

Llama-3.1-70B-Instruct 0.50 0.44 0.65 0.52 0.74 0.45 0.33 0.53 Llama-3.1-70B-Instruct-T-one 0.50 0.43 0.57 0.54 0.65 0.46 0.36 0.51 Llama-3.1-405B-Instruct 0.64 0.49 0.71 0.54 0.79 0.50 0.57 0.58

Mistral-Large-2 0.65 0.47 0.70 0.55 0.78 0.48 0.57 0.57 Mixtral-8x7b-Instruct 0.21 0.31 0.42 0.54 0.51 0.39 0.25 0.42

Mixtral-8x7b-Instruct-T-one 0.21 0.31 0.41 0.52 0.51 0.38 0.25 0.42

o1 0.80 0.57 0.80 0.56 0.86 0.55 0.85 0.64 PaperQA2 0.62 0.49 0.66 0.56 0.75 0.49 0.57 0.57

Phi-3-Medium-4k-Instruct 0.34 0.36 0.53 0.53 0.61 0.42 0.50 0.47

###### Table A4: Performance of the models on ChemBench-Mini. The table shows the fraction of questions answered correctly by the models for different skills and difficulty levels.

###### Requires Difficulty

Overall Accuracy Calculation Knowledge Reasoning Basic Intermediate Advanced

Model

Claude-2 0.29 0.52 0.51 0.57 0.45 0.36 0.49

Claude-3 (Opus) 0.62 0.64 0.67 0.78 0.61 0.43 0.67 Claude-3.5 (Sonnet) 0.62 0.75 0.75 0.79 0.70 0.64 0.73

Claude-3.5 (Sonnet) React 0.73 0.75 0.75 0.83 0.71 0.75 0.76 Command-R+ 0.25 0.49 0.41 0.46 0.42 0.20 0.42 Galactica-120b 0.01 0.02 0.01 0.04 0.01 0.00 0.02

Gemini-Pro 0.27 0.49 0.46 0.50 0.43 0.35 0.45

- Gemma-1.1-7B-it 0.23 0.40 0.36 0.42 0.33 0.10 0.35 Gemma-1.1-7B-it-T-one 0.21 0.39 0.35 0.41 0.31 0.14 0.34
- Gemma-2-9B-it 0.44 0.46 0.45 0.56 0.44 0.40 0.49 Gemma-2-9B-it-T-one 0.40 0.44 0.42 0.55 0.41 0.43 0.46 GPT-3.5 Turbo 0.31 0.46 0.45 0.51 0.41 0.36 0.44 GPT-4 0.47 0.63 0.64 0.67 0.62 0.57 0.64

GPT-4o 0.68 0.72 0.74 0.81 0.67 0.65 0.72 GPT-4o React 0.60 0.59 0.63 0.69 0.57 0.60 0.62

Human 0.28 0.23 0.27 0.32 0.24 0.27 0.27

- Llama-2-70B Chat 0.12 0.18 0.13 0.19 0.13 0.00 0.14
- Llama-3-8B-Instruct 0.26 0.54 0.44 0.43 0.45 0.50 0.44

Llama-3-8B-Instruct-T-one 0.28 0.54 0.45 0.43 0.46 0.57 0.45

Llama-3-70B-Instruct 0.49 0.63 0.57 0.67 0.59 0.30 0.60 Llama-3-70B-Instruct-T-one 0.49 0.61 0.57 0.67 0.57 0.29 0.59

Llama-3.1-8B-Instruct 0.33 0.47 0.44 0.55 0.39 0.40 0.46 Llama-3.1-8B-Instruct-T-one 0.38 0.47 0.40 0.51 0.38 0.64 0.44

Llama-3.1-70B-Instruct 0.52 0.62 0.63 0.73 0.62 0.35 0.64 Llama-3.1-70B-Instruct-T-one 0.50 0.55 0.58 0.66 0.54 0.36 0.57 Llama-3.1-405B-Instruct 0.66 0.69 0.70 0.72 0.69 0.57 0.69

Mistral-Large-2 0.66 0.65 0.68 0.72 0.66 0.60 0.68 Mixtral-8x7b-Instruct 0.23 0.40 0.37 0.45 0.32 0.25 0.36

Mixtral-8x7b-Instruct-T-one 0.21 0.39 0.35 0.46 0.29 0.25 0.36

o1 0.77 0.79 0.81 0.82 0.77 0.85 0.80 PaperQA2 0.63 0.69 0.65 0.70 0.67 0.55 0.67

Phi-3-Medium-4k-Instruct 0.34 0.49 0.52 0.57 0.47 0.50 0.51

- Table A5: Performance of the models on the ChemBench corpus. The table shows the fraction of questions answered correctly by the models for different topics. Models with “T-one” in the name were run for a temperature of 1, which allows us to study the temperature effect in the benchmark. Systems with “ReAct” in the name are tool augmented, i.e., they can call external tools such as web search or a calculator to answer the questions better. However, we limit those systems to ten calls to the LLM. This constraint often led the systems to not find the correct answer within the specified number of calls. In this case, we consider the answer as incorrect (see Appendix A.11).

Model Analytical Chemical Preference General Inorganic Materials Science Organic Physical Technical Toxicity/Safety Overall Accuracy

- Claude-2 0.39 0.51 0.51 0.56 0.30 0.60 0.54 0.47 0.33 0.47
- Claude-3 (Opus) 0.48 0.57 0.77 0.75 0.50 0.69 0.69 0.70 0.41 0.57 Claude-3.5 (Sonnet) 0.57 0.58 0.83 0.82 0.50 0.82 0.81 0.85 0.44 0.63 Claude-3.5 (Sonnet) React 0.58 0.60 0.87 0.79 0.50 0.83 0.80 0.80 0.41 0.62 Command-R+ 0.36 0.51 0.49 0.50 0.30 0.56 0.37 0.50 0.31 0.45 Galactica-120b 0.00 0.00 0.05 0.05 0.00 0.00 0.06 0.00 0.02 0.02 Gemini-Pro 0.41 0.50 0.49 0.44 0.50 0.58 0.46 0.47 0.31 0.45 Gemma-1.1-7B-it 0.21 0.00 0.33 0.43 0.40 0.39 0.34 0.38 0.23 0.19 Gemma-1.1-7B-it-T-one 0.21 0.01 0.35 0.41 0.40 0.38 0.35 0.38 0.22 0.19 Gemma-2-9B-it 0.32 0.55 0.55 0.53 0.50 0.57 0.54 0.53 0.34 0.48 Gemma-2-9B-it-T-one 0.30 0.56 0.56 0.51 0.40 0.57 0.53 0.47 0.34 0.48 GPT-3.5 Turbo 0.40 0.53 0.49 0.53 0.50 0.60 0.44 0.40 0.31 0.47 GPT-4 0.44 0.16 0.70 0.68 0.50 0.68 0.69 0.70 0.41 0.41 GPT-4o 0.56 0.59 0.81 0.80 0.60 0.75 0.76 0.75 0.44 0.61 GPT-4o React 0.47 0.42 0.76 0.73 0.40 0.72 0.62 0.72 0.37 0.51

- Llama-2-70B Chat 0.07 0.49 0.13 0.23 0.20 0.15 0.18 0.12 0.14 0.27
- Llama-3-8B-Instruct 0.43 0.52 0.45 0.48 0.40 0.57 0.40 0.60 0.32 0.46 Llama-3-8B-Instruct-T-one 0.42 0.52 0.44 0.52 0.30 0.55 0.38 0.62 0.32 0.46 Llama-3-70B-Instruct 0.43 0.53 0.61 0.64 0.60 0.65 0.64 0.62 0.37 0.52 Llama-3-70B-Instruct-T-one 0.39 0.53 0.61 0.66 0.60 0.64 0.65 0.60 0.37 0.52 Llama-3.1-8B-Instruct 0.41 0.53 0.51 0.50 0.40 0.58 0.55 0.45 0.33 0.47 Llama-3.1-8B-Instruct-T-one 0.38 0.52 0.53 0.48 0.40 0.58 0.48 0.40 0.32 0.46 Llama-3.1-70B-Instruct 0.42 0.52 0.69 0.74 0.50 0.68 0.68 0.65 0.38 0.53 Llama-3.1-70B-Instruct-T-one 0.36 0.54 0.67 0.67 0.50 0.58 0.55 0.55 0.39 0.51 Llama-3.1-405B-Instruct 0.52 0.54 0.79 0.78 0.70 0.74 0.73 0.70 0.42 0.58 Mistral-Large-2 0.50 0.55 0.79 0.77 0.40 0.73 0.73 0.68 0.40 0.57 Mixtral-8x7b-Instruct 0.28 0.54 0.43 0.51 0.40 0.50 0.37 0.33 0.27 0.42 Mixtral-8x7b-Instruct-T-one 0.28 0.52 0.45 0.50 0.40 0.50 0.39 0.33 0.27 0.42 o1 0.62 0.56 0.93 0.90 0.80 0.82 0.89 0.85 0.48 0.64 PaperQA2 0.48 0.56 0.73 0.71 0.60 0.67 0.73 0.70 0.42 0.57 Phi-3-Medium-4k-Instruct 0.35 0.53 0.48 0.58 0.40 0.56 0.49 0.55 0.33 0.47

- Table A6: Performance of the models on ChemBench-Mini. The table shows the fraction of questions answered correctly by the models for different topics.

Model Analytical General Inorganic Materials Science Organic Physical Technical Toxicity/Safety Overall Accuracy Claude-2 0.77 0.46 0.47 0.00 0.62 0.42 0.52 0.60 0.49

Claude-3 (Opus) 0.77 0.73 0.76 0.50 0.62 0.62 0.74 0.65 0.67 Claude-3.5 (Sonnet) 0.86 0.69 0.68 0.50 0.79 0.79 0.87 0.55 0.73

Claude-3.5 (Sonnet) React 0.77 0.81 0.71 0.50 0.88 0.79 0.83 0.55 0.76 Command-R+ 0.45 0.35 0.44 0.00 0.33 0.25 0.57 0.60 0.42 Galactica-120b 0.00 0.00 0.03 0.00 0.00 0.12 0.00 0.05 0.02

Gemini-Pro 0.55 0.38 0.41 0.00 0.58 0.46 0.43 0.45 0.45

- Gemma-1.1-7B-it 0.36 0.31 0.44 0.50 0.46 0.29 0.39 0.40 0.35 Gemma-1.1-7B-it-T-one 0.32 0.31 0.41 0.50 0.42 0.33 0.39 0.40 0.34
- Gemma-2-9B-it 0.45 0.54 0.47 0.50 0.54 0.58 0.61 0.50 0.49 Gemma-2-9B-it-T-one 0.45 0.50 0.44 0.50 0.58 0.54 0.52 0.45 0.46 GPT-3.5 Turbo 0.55 0.35 0.44 0.50 0.58 0.50 0.43 0.55 0.44 GPT-4 0.77 0.69 0.59 1.00 0.79 0.67 0.83 0.60 0.64 GPT-4o 0.73 0.69 0.71 0.50 0.75 0.75 0.78 0.60 0.72 GPT-4o React 0.55 0.58 0.68 0.50 0.75 0.58 0.74 0.45 0.62 Human 0.31 0.41 0.30 0.24 0.36 0.26 0.20 0.22 0.27

- Llama-2-70B Chat 0.00 0.19 0.15 0.50 0.21 0.17 0.13 0.25 0.14
- Llama-3-8B-Instruct 0.68 0.27 0.44 0.00 0.50 0.29 0.61 0.50 0.44

Llama-3-8B-Instruct-T-one 0.64 0.27 0.53 0.00 0.46 0.25 0.65 0.50 0.45 Llama-3-70B-Instruct 0.64 0.62 0.62 0.50 0.75 0.58 0.74 0.50 0.60 Llama-3-70B-Instruct-T-one 0.59 0.62 0.62 0.50 0.71 0.58 0.70 0.60 0.59 Llama-3.1-8B-Instruct 0.59 0.35 0.53 0.00 0.50 0.50 0.52 0.60 0.46 Llama-3.1-8B-Instruct-T-one 0.59 0.38 0.47 0.50 0.50 0.42 0.39 0.55 0.44 Llama-3.1-70B-Instruct 0.59 0.62 0.74 0.50 0.75 0.71 0.70 0.60 0.64 Llama-3.1-70B-Instruct-T-one 0.45 0.58 0.59 0.50 0.67 0.62 0.65 0.45 0.57 Llama-3.1-405B-Instruct 0.68 0.73 0.74 1.00 0.71 0.75 0.70 0.60 0.69

Mistral-Large-2 0.82 0.65 0.68 1.00 0.75 0.71 0.70 0.50 0.68 Mixtral-8x7b-Instruct 0.41 0.35 0.47 0.50 0.33 0.33 0.30 0.40 0.36

Mixtral-8x7b-Instruct-T-one 0.45 0.42 0.41 0.50 0.38 0.38 0.30 0.35 0.36

o1 0.91 0.92 0.79 1.00 0.79 0.79 0.91 0.70 0.80 PaperQA2 0.68 0.73 0.62 1.00 0.58 0.71 0.78 0.65 0.67

Phi-3-Medium-4k-Instruct 0.45 0.42 0.56 0.50 0.62 0.54 0.57 0.50 0.51

[Figure 25]

Claude-3.5 (Sonnet) GPT-3.5 Turbo

0.8

GPT-4o Llama-3.1-405B-Instruct

0.6

##### model

Llama-3.1-70B-Instruct Llama-3.1-8B-Instruct Mistral-Large-2 PaperQA2 o1

0.4

0.2

0.0

DAI(dailyallowedintake)

toxicology

chemicalpreference

textbook

polymerchemistry

chemistryolympiad

numberofisomers

pointgroup

GFK(chemicalsafety)

chemicalcompatibility

electroncounts

organicreactivity

materialscompatibility

nametoSMILES

GHSpictograms

classificationtask

numberofNMRsignals

subset

- Figure A5: Fraction of correctly answered questions per data source on ChemBench-Mini. The heatmap shows, in color, the fraction of questions answered correctly by different systems for some of our data sources. The performance is measured as the fraction of questions answered correctly by the models. A score of one (red) indicates that all questions were answered correctly, while a score of zero (blue) indicates that none of the questions were answered correctly. We see that the performance of the models varies significantly between the different data sources. For instance, it is interesting to observe that questions sourced based on textbooks seem easier for our leading models than for humans. However, this performance does not correlate with performance on other sources, e.g., semi-programmatically created tasks such as questions about the number of signals in an NMR spectrum.

Claude-3.5 (Sonnet) GPT-3.5 Turbo

[Figure 26]

0.8

GPT-4o Llama-3.1-405B-Instruct

0.6

##### model

Llama-3.1-70B-Instruct Llama-3.1-8B-Instruct Mistral-Large-2 PaperQA2 human o1

0.4

0.2

0.0

toxicology

textbook

polymerchemistry

chemistryolympiad

numberofisomers

pointgroup

GFK(chemicalsafety)

chemicalcompatibility

electroncounts

organicreactivity

classificationtask

numberofNMRsignals

subset

- Figure A6: Fraction of correctly answered questions per data source on ChemBench-Mini. The heatmap shows, in color, the fraction of questions answered correctly by different systems for some of our data sources. The performance is measured as the fraction of questions answered correctly by the models. A score of one (red) indicates that all questions were answered correctly, while a score of zero (blue) indicates that none were answered correctly. We see that the performance of the models varies significantly between the different data sources. For instance, it is interesting to observe that questions sourced based on textbooks seem easier for the leading models than for humans. However, this performance does not correlate with performance on other sources, e.g., semi-programmatically created tasks such as questions about the number of signals in an NMR spectrum.

ChemBench ChemBench-Mini

0.8

0.6

accuracy

0.4

0.2

0.0

Claude-3.5 (Sonnet) ReactClaude-3.5 (Sonnet)Llama-3.1-405B-Instructo1 Mistral-Large-2GPT-4oLlama-3.1-70B-InstructClaude-3 (Opus)Llama-3-70B-Instruct-T-oneLlama-3.1-70B-Instruct-T-oneLlama-3-70B-InstructPaperQA2 Phi-3-Medium-4k-InstructGemma-2-9B-it-T-oneGPT-4o ReactGemma-2-9B-itLlama-3.1-8B-InstructLlama-3.1-8B-Instruct-T-oneLlama-3-8B-Instruct-T-oneClaude-2GPT-3.5 TurboLlama-3-8B-InstructMixtral-8x7b-Instruct-T-oneMixtral-8x7b-InstructGemini-ProCommand-R+Llama-2-70B ChatGemma-1.1-7B-it-T-oneGemma-1.1-7B-itGPT-4Galactica-120b

- Figure A7: Performance of the models on the ChemBench corpus and ChemBench-Mini subset. The bar plot shows the fraction of questions that were answered completely correctly, highlighting the relative performance of different models on both the ChemBench corpus and the ChemBench-Mini subset. We see that the model ranking remains fairly consistent across both sets

#### A.5 Performance as a function of molecular features

To better understand if the performance of the models is correlated with specific features of the molecules, we analyzed the performance of the models as a function of the number of atoms and the complexity of the molecules. Figure A8 shows that the performance of the models is not correlated with the complexity of the molecules but rather with the number of atoms (Figure A9, similar trivial correlation for Figure A10).

GPT-3.5 GPT-4o o1 Claude-3.5 Llama-3.1-405B

10

5

0

metricsmae

Mistral-Large-2

Llama-3.1-8B

Llama-3.1-70B

PaperQA2

10

5

0

0 100

0 100

0 100

0 100

Böttcher complexity

- Figure A8: Dependence of the mean absolute error in predicting the number of NMR signals on the B¨ottcher complexity of the molecules. The complexity measure proposed by B¨ottcher [17] is an information-theoretic additive measure of compound complexity that follows chemical intuitions. The plot shows that for the LLMs, the predictive performance (measured as the mean absolute error in the prediction of the number of NMR signals) is not correlated with the complexity of the molecules (that is, molecules tend to not to be able to predict the number of NMR signals regardless of molecular complexity). For inference based on reasoning, one would expect that the complexity of the molecule is a good predictor of the difficulty of the question.

GPT-3.5 GPT-4o o1 Claude-3.5 Llama-3.1-405B

10

5

0

metricsmae

Mistral-Large-2

Llama-3.1-8B

Llama-3.1-70B

PaperQA2

10

5

0

5 10 15

5 10 15

5 10 15

5 10 15

number of atoms

###### Figure A9: Dependence of the mean absolute error in predicting the number of NMR signals on the number of atoms.

GPT-3.5 GPT-4o o1 Claude-3.5 Llama-3.1-405B

100

50

0

metricsmae

Mistral-Large-2

Llama-3.1-8B

Llama-3.1-70B

PaperQA2

100

50

0

0 10

0 10

0 10

0 10

number of atoms

###### Figure A10: Dependence of the mean absolute error in predicting total electron counts on the number of atoms. The plot shows that for the LLMs, the predictive performance (measured as the mean absolute error in the prediction of the total electron counts) is sometimes correlated with the number of atoms in the molecule.

#### A.6 Influence of model scale

- Figure A11 shows the performance of the models as a function of the number of parameters in the model. We see that the performance of the models correlates with their size for the models of the LLama-3 and Llama-3.1 herd of models.

Llama-3

performance

0.55

Llama-3.1

0.50

0 1 2 3 4

number of parameters 1e11

- Figure A11: Performance of models as a function of model size. The plot shows the performance of the models as a function of the parameter count. The performance is measured as the fraction of questions answered correctly by the models. We see that the performance of the models correlates with scale for the models of the LLama3 and Llama-3.1 herd of models.

#### A.7 Refusal detection

LLMs typically undergo refusal training to prevent harmful or undesirable outputs. As a result, models may decline to answer questions perceived as potentially adversarial prompts. To automatically detect refusals, we use a modified regular expression reported by LLM Guard18 to detect commonly used refusal phrases.

Table A7 lists how many refusals were detected for the responses of different models on ChemBench. Overall, we find that refusals do not majorly affect the performance measured by ChemBench.

#### A.8 LLM Parsing

Our parsing workflow uses pipelines based on regular expressions to extract the answers. In some cases, however, the answers are not directly extractable from the responses, for instance, when the model does not follow the formatting instructions. In these cases, we use a fallback mechanism to extract the answers. The fallback mechanism uses an LLM to extract the answers from the responses. The LLM is provided with the response and the question and is prompted to only extract but not generate the answer. We used Llama-3-70B-Instruct, accessed via the Groq API.

- Table A7 tabulates the number of times the fallback mechanism was used for each model.

Table A7: Refusal counts and parsing. The table shows the number of refusals detected and the number of times the LLM fallback parsing mechanism was used for each model.

###### Refusal LLM Extraction

Model

Nº of Questions Fraction Nº of Questions Fraction

- Claude 2 100 0.035868 0 0.000000
- Claude 3 9 0.003228 0 0.000000 Claude 3.5 Sonnet 31 0.011119 3 0.001076 Claude-3.5 ReAct 0 0.000000 129 0.046270 Command R+ 0 0.000000 5 0.001793 Gemini Pro 0 0.000000 4 0.001435 Gemma 1.1 7B 0 0.000000 18 0.006456 Gemma 1.1 7B Temp=1 1 0.000359 22 0.007891 Gemma 2 9B 5 0.001793 25 0.008967 Gemma 2 9B Temp=1 5 0.001793 37 0.013271 GPT-3.5 Turbo 0 0.000000 8 0.002869 GPT-4 0 0.000000 3 0.001076 GPT-4o 0 0.000000 7 0.002511 GPT-4o ReAct 0 0.000000 667 0.239240

- Llama 2 70B Chat 1 0.000359 3 0.001076
- Llama 3 8B 0 0.000000 9 0.003228 Llama 3 8B Temp=1 6 0.002152 11 0.003945 Llama 3 70B 0 0.000000 16 0.005739 Llama 3 70B Temp=1 1 0.000359 17 0.006098 Llama 3.1 8B 0 0.000000 35 0.012554 Llama 3.1 8B Temp=1 3 0.001076 30 0.010760 Llama 3.1 70B 0 0.000000 53 0.019010 Llama 3.1 70B Temp=1 1 0.000359 245 0.087877 Llama 3.1 405B 0 0.000000 22 0.007891 Mistral Large 2 123B 0 0.000000 2 0.000717 Mixtral 8x7B 4 0.001435 54 0.019369 Mixtral 8x7B Temp=1 7 0.002511 58 0.020803 o1 0 0.000000 2 0.000717 Paper QA 35 0.012554 52 0.018651 Phi 3 Medium 4K 0 0.000000 7 0.002511

- A.9 Implementation An overview of the benchmarking pipeline implemented in ChemBench is shown in

- Figure A12. More detailed information can be found in the online documentation of the ChemBench package at https://lamalab-org.github.io/chem-bench/.

JSON Files

Create Task objects

Initialize ChemBenchmark

Create Prompter objects

InstructionPrompter CompletionPrompter

MCQ not MCQ

Task iterator

Prompt with MCQ answer template

Prompt with numeric answer template

LLM call

LLM call

regex-based numeric answer parsing and conversion to float

regex-based MCQ answer parsing

Parsing successfull

no LLM-based extraction

yes

Metric computation, refusal detection

Report

- Figure A12: Overview of the benchmarking pipeline implemented in ChemBench. The process begins with JSON files containing task data, which are used to create Task objects and initialize the ChemBenchmark. Prompter objects are then created to handle different types of prompts for instruction-tuned and completion models. The Task Iterator differentiates between MCQ and other question types. For each task type, appropriate prompts are generated and passed to the LLM. The responses are then processed using regex-based parsing methods specific to MCQ or numeric answers (after obtaining the relevant part of the response from the instruction-tuned models). The regex-based parsing is elaborate and can handle special cases such as scientific notation or Roman numerals. If the initial parsing is unsuccessful, the system employs an LLM-based extraction method as a fallback. The parsed or extracted answers then undergo metric computation and refusal detection.

#### A.10 Human baseline

App To facilitate the collection of responses, we developed a responsive web application in Typescript using the Next.js19 app router framework. This application handles serving the user interface and exposes various representational state transfer (REST) APIs for relevant operations. We utilize a Postgresql. The web application is styled with Tailwind CSS20 using the shadcn/ui component library and uses NextAuth21 for easy and secure user authentication. The application is hosted on the Vercel web hosting platform.

In the applications, human participants were presented with molecules as ren-

dered drawings and SMILES strings. LATEX equations and chemical equations were rendered using MathJax (Figure A13).

Statistics Figure A14 shows the distribution of scores our human scorers achieved.

We also recorded the time humans took to answer the questions (Figure A15). This time is the time from the question being displayed to the human to the human submitting the answer.

Additionally, we prompted users to provide additional information about their experience in chemistry. While we recorded fine-grained information, e.g., their specialization, we focused on the number of years since the first university-level chemistry course. Figure A16 shows that the experience of the human scorers was weakly correlated with the correctness of their answers (Figure A16, Spearman’s ρ ≈ 0.19 , and p ≈ 0.31 ).

Tool use In our study, humans were allowed to use tools for answering some questions. They could also report what tools they used for answering questions. As Figure A17 shows, the most common tool was some form of web search (which, according to the free text responses, often was a multi-step process).

[Figure 27]

- (a) A physcial chemistry question.

[Figure 28]

- (b) An organic chemistry question.

54

- Figure A13: Examples of how questions were shown to the human participants.

20

/w tools

w/o tools

count

10

0

0.2 0.3 0.4

fraction correct

- Figure A14: Distribution of human scores.The histogram and kernel density estimates show the fraction of questions answered correctly by the human volunteers. Since the best possible score for each question is one and the worst possible score is zero, the values on this plot are between zero and one. A score of one would mean that a volunteer answered all questions correctly. A score of zero would mean that no question was answered correctly. We find that the scores for the questions that the human volunteers answered with tools are generally lower than the scores for the questions that the human volunteers answered without tools.

False True

all correct

101

103

time/s

tools allowed False True

| |
|---|

| |
|---|

- Figure A15: Time taken by human scorers to answer questions vs. correctness of their answers. From the plot, it is clear that there is no clear dependence of the correctness of the answers on the time taken by the human scorers to answer the questions. However, we see that human scorers typically took longer to correctly answer questions with tool use.

without tools

fractioncorrect

0.4

with tools

0.3

0.2

0.0 2.5 5.0 7.5 10.0

experience in chemistry / y

###### Figure A16: Experience of human scorers vs. correctness of their answers. The experience (in the number of years since the first university-level chemistry course) of the human scorers wasp correlated with the correctness of their answers.

Web search

100

Wikipedia Calculator ChemDraw

80

percentageoftoolusage

PTE

Database

60

Other

Textbook

40

20

0

Analytical ChemistryGeneral ChemistryInorganic ChemistryMaterials ScienceOrganic ChemistryPhysical ChemistryTechnical ChemistryToxicity/Safety

###### Figure A17: Tool usage by human scorers. The plot shows the most commonly used tools by human participants.

#### A.11 Tool augmented models

In addition to directly prompting LLMs, we also investigated the performance of tool-augmented systems. For this, we investigated the two models that showed the best overall results, GPT-4o and Claude-3.5 (Sonnet) (o1 overperformed both models but is not recommended using this model with such “reasoning” prompts such as ReAct10 or chain of thought (CoT)22). For both models, we created a ReAct-style tool augmentation environment in which models had access to WolframAlpha, the ArXiv API, Wikipedia, and web search (using Brave search API). We based the selection of these tools on the most used tools by humans (see Figure A17). Additionally, we added two specific tools to convert IUPAC names to SMILES, and SMILES to IUPAC names. These conversion tools allow us to understand better how the agents perform for specific questions in this agent environment configuration. We implemented the systems using Langchain23 with the default ReAct prompt and constrained the system to a maximum of ten LLM calls.

While for Claude-3.5 (Sonnet) the overall performance did not change, we observe a decrease in performance for GPT-4o compared with the LLM without tools (Table A3). If we study the results by each type of question, we observe an improvement for questions regarding electron counts or point groups of compounds. However, the scores decreased for questions about the number of isomers or GHS pictograms. For the specific questions about converting IUPAC names to SMILES, the results decreased notably despite the models having access to specific tools prepared for those questions. By studying the reasoning path for these cases, we found that the error results from the models responding in a format that the LangChain framework with the default ReAct loop cannot handle. This indicates that agent frameworks need optimization to be more robust. It involves not only equipping the LLMs with tools but also necessitates engineering efforts to create robust systems.

#### A.12 Confidence estimates

Since it is important to understand if models can provide an indication of whether their answer might likely be incorrect, we prompted some of our top performing LLMs to return the confidence in providing a correct answer on an ordinal scale. This is similar to the verbalized confidence scores reported by Xiong et al. [24]. We find that the models show different distributions of confidence scores, which, for some, are skewed to the extremes.

In addition, we also analyzed the confidence estimated via the log probabilities of the answer tokens. This probability of a token given the context is not necessarily the same as the confidence in the correctness of the answer. However, it is still often used as a proxy.

Our analysis of both log probabilities and prompting confidence reveals distinct calibration behaviors across different language models. GPT-4o demonstrates an overconfident tendency, often assigning high probabilities even to incorrect answers. However, when GPT-4o displays high confidence, it accurately predicts correct answers approximately 80% of the time. In contrast, Llama-3.1-8B-Instruct confidence distribution is more evenly spread, with a majority of predictions centered around 0.5. High-confidence predictions from Llama-3.1-8B-Instruct are less frequent compared to GPT-4o, and unlike GPT-4o, high confidence does not necessarily correlate with a higher chance of correct answers.

Llama-3.1-8B-Instruct

GPT-4o

1.00

1.00

fractionofpositives

ECE: 0.1391

ECE: 0.2286

1000

0.75

0.75

400

Count

0.50

0.50

500

200

0.25

0.25

0.00

0

0.00

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

predicted probability

- Figure A18: Reliability diagram of logit-based confidence estimates. For this analysis we obtained the linear probability from the logprobs of the models. Only logprobs of the tokens corresponding to the answers were considered. Linear probability was computed by taking exponential of logprobs (for sequences with multiple tokens, values were multiplied). The plot shows the average predicted probability and the fraction of correct answers for each bin of linear probabilities. The ideal scenario is a diagonal line, indicating perfect calibration where the model’s confidence aligns perfectly with the actual correctness. The expected calibration error (ECE) value quantifies the overall calibration performance, with a lower ECE indicating better calibration.

#### A.13 Impact of sampling temperature

We also investigated the impact of sampling temperature (i.e. temperature 0 means always sampling the most likely token, higher temperatures introduce some randomness in the generation process) on the performance of the models. Figure A19 shows that, generally, the performance of models tends to decrease with increasing temperature.

0.6

Llama-3.1-70B-Instruct

Mixtral-8x7b-Instruct

score

Llama-3.1-8B-Instruct

0.5

Gemma-2-9B-it

Llama-3-8B-Instruct

Llama-3-70B-Instruct

0.4

T=0 T=1

temperature

- Figure A19: Impact of sampling temperature on the performance of the models. The plot shows the performance of the models at zero temperature (i.e., greedy decoding) and temperature of one. The performance is measured in terms of the fraction of questions answered correctly.

#### A.14 Summary of trends and recommendations

o1 consistently leads across both the ChemBench corpus and ChemBench-Mini. Claude-3.5 (Sonnet), GPT-4o, Llama-3.1-405B-Instruct form the next tier of strong performers. These models show robust performance across all skills and difficulty levels.

Larger models generally perform better (Llama-3.1-405B-Instruct ¿ Llama-3.1-70BInstruct ¿ Llama-3.1-8B-Instruct). However, smaller but well-designed models can compete with larger models. For example, Gemma-2-9B-it has an overall accuracy of 0.48 on the ChemBench corpus, which is only 10% less accurate than the much larger Llama-3.1-405B-Instruct model.

One can observe a clear progression of performance across model families (e.g., Claude-3.5 (Sonnet) ¿ Claude-3 ¿ Claude-2 or GPT-4o ¿ GPT-4 ¿ GPT-3.5 Turbo Zero-T). Newer versions consistently outperform their predecessors. Scores on knowledgeintensive tasks are typically lower than those on calculation and reasoning-intensive questions.

Based on these findings, some scope for improving these models could be to focus on enhancing knowledge-based training, possibly through improved pre-training on chemistry-specific texts or with chemistry knowledge-base integration. However, innovation is also needed to incorporate this information into the systems. For example, even PaperQA2 could not outperform the LLM it is used for directing the tools, even though the agent has access to literature evidence. This suggests we might need to not only focus on building chemistry-specific retrieval datasets since current systems fail to retrieve the relevant papers and should be coupled with more domain-specific databases. However, the observation that our ReAct agents were too fragile to answer questions for which they had custom-made tools available correctly suggests that we also must invest in building more robust agent frameworks (see Appendix A.11 for further discussion about the ReAct environment).

Models show the strongest performance in reasoning tasks on the ChemBench corpus, which suggests that current training approaches are good at developing logical/reasoning capabilities for basic problem-solving of university-level exams. However, performance drops significantly for advanced tasks, suggesting that more advanced chemistry problems with complex, multi-step solutions (e.g., solving analytical chemistry problems) should be included in training or finetuning.

The post-training (e.g., reinforcement learning with human feedback (RLHF)) of the current models does not equip them with human-like “chemical taste.” This, however, will be essential for future discovery research (e.g., in combination with genetic algorithms).

#### A.15 Leaderboard

Our leaderboard is based on the tool chain developed for Matbench.25 Briefly, the ChemBench pipeline produces standardized files in json format that contributors can add via pull requests to the ChemBench repository. The Markdown tables and interactive plots are automatically generated and updated on the ChemBench website. The leaderboard is available at https://lamalab-org.github.io/chem-

bench/leaderboard/.

### Acronyms

AGI artificial general intelligence. API application programming interface. CoT chain of thought. ECE expected calibration error. GHS globally harmonized system of classification and labelling of chemicals. HELM Holistic Evaluation of Language Models. IUPAC International Union of Pure and Applied Chemistry. LLM large language model. MCQ multiple-choice question. ML machine learning. NMR nuclear magnetic resonance. RAG retrieval augmented generation. REST representational state transfer. RLHF reinforcement learning with human feedback. SMILES simplified molecular input line-entry system.

#### References

- 1. Schulze Balhorn, L. et al. Empirical assessment of ChatGPT’s answering capabilities in natural science and engineering. Sci. Rep. 14 (Feb. 2024).
- 2. Microsoft Research AI4Science and Microsoft Azure Quantum. The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4. arXiv preprint arXiv:2311.07361 (2023).
- 3. Castro Nascimento, C. M. & Pimentel, A. S. Do Large Language Models Understand Chemistry? A Conversation with ChatGPT. J. Chem. Inf. Model. 63, 1649–1655 (Mar. 16, 2023).
- 4. Northcutt, C. G., Athalye, A. & Mueller, J. Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks. arXiv preprint arXiv:2103.14749 (2021).
- 5. Frye, C. PubMedQA noisy Dec. 2023. https://twitter.com/charles%5C_ irl/status/1731854677711507650.
- 6. Awg. Broken benchmark: MMLU 2023. https://www.lesswrong.com/posts/ rQBaftqKMfG2uMiWb/broken-benchmark-mmlu.
- 7. Taylor, R. et al. Galactica: A Large Language Model for Science. arXiv preprint arXiv:2211.09085 (2022).
- 8. Schick, T. et al. Toolformer: Language models can teach themselves to use tools. Adv. Neur. In. 36 (2024).
- 9. Karpas, E. et al. MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning. arXiv preprint arXiv:2205.00445 (2022).
- 10. Yao, S. et al. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629 (2022).
- 11. Aspuru-Guzik, A., Lindh, R. & Reiher, M. The Matter Simulation (R)evolution. ACS Cent. Sci. 4, 144–152 (Feb. 6, 2018).
- 12. Guo, T. et al. What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks. arXiv preprint arXiv:2305.18365 (2023).
- 13. Sun, L. et al. SciEval: A Multi-Level Large Language Model Evaluation Benchmark for Scientific Research. arXiv preprint arXiv:2308.13149 (2023).
- 14. Cai, X. et al. Comprehensive evaluation of molecule property prediction with ChatGPT. Methods 222, 133–141 (Feb. 2024).
- 15. Rein, D. et al. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. arXiv preprint arXiv:2311.12022 (2023).
- 16. Srivastava, A. et al. Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615 (2022).

- 17. B¨ottcher, T. An Additive Definition of Molecular Complexity. J. Chem. Inf. Model. 56, 462–470 (Feb. 2016).
- 18. AI, P. LLM Guard https://github.com/protectai/llm-guard.
- 19. Vercel. nextjs https://nextjs.org/.
- 20. tailwindcss. Tailwind CSS https://tailwindcss.com/.
- 21. NextAuth.js. NextAuth.js https://next-auth.js.org/.
- 22. Wei, J. et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv preprint arXiv:2201.11903 (2023).
- 23. Chase, H. LangChain Oct. 2022. https://github.com/langchain- ai/ langchain.
- 24. Xiong, M. et al. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. arXiv preprint arXiv:2306.13063 (2023).
- 25. Dunn, A., Wang, Q., Ganose, A., Dopp, D. & Jain, A. Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. npj Comp. Mater. 6 (Sept. 2020).

