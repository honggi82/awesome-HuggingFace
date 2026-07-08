# arXiv:2502.16457v4[cs.CL]19Mar2025

## Towards Fully-Automated Materials Discovery via Large-Scale Synthesis Dataset and Expert-Level LLM-as-a-Judge

Heegyu Kim1∗, Taeyang Jeon1∗, Seungtaek Choi, Ji Hoon Hong3,4, Dong Won Jeon3,4, Ga-Yeon Baek5, Gyeong-Won Kwak5, Dong-Hee Lee5, Jisu Bae5, Chihoon Lee5, Yunseo Kim5, Seon-Jin Choi5, Jin-Seong Park5, Sung Beom Cho3,4, Hyunsouk Cho1,2, 1Department of Artificial Intelligence, 2Department of Software and Computer Engineering, 3 Department of Energy Systems Research, 4 Department of Materials Science and Engineering, Ajou University, Suwon 16499, Republic of Korea 5Division of Materials Science and Engineering, Hanyang University, Seoul 04763, Republic of Korea

Correspondence: hyunsouk@ajou.ac.kr

### Abstract

[Figure 1]

Materials synthesis is vital for innovations such as energy storage, catalysis, electronics, and biomedical devices. Yet, the process relies heavily on empirical, trial-and-error methods guided by expert intuition. Our work aims to support the materials science community by providing a practical, data-driven resource. We have curated a comprehensive dataset of 17K expert-verified synthesis recipes from openaccess literature, which forms the basis of our newly developed benchmark, AlchemyBench. AlchemyBench offers an end-to-end framework that supports research in large language models applied to synthesis prediction. It encompasses key tasks, including raw materials and equipment prediction, synthesis procedure generation, and characterization outcome forecasting. We propose an LLM-as-aJudge framework that leverages large language models for automated evaluation, demonstrating strong statistical agreement with expert assessments. Overall, our contributions offer a supportive foundation for exploring the capabilities of LLMs in predicting and guiding materials synthesis, ultimately paving the way for more efficient experimental design and accelerated innovation in materials science.

Figure 1: An overview of our contributions, featuring the Open Materials Guide Dataset for large-scale synthesis recipes and AlchemyBench for scalable, expertlevel evaluation.

tracting and generating synthesis procedures from unstructured scientific literature (Song et al., 2023; Dunn et al., 2020). However, practical adoption is hampered by several challenges. Existing datasets are often small, domain-specific, and noisy, limiting model generalizability. Moreover, the absence of comprehensive benchmarks makes it difficult to assess the performance of synthesis prediction methods, while expert evaluations remain too costly and time-consuming for large-scale use.

### 1 Introduction

Materials synthesis underpins advances in energy storage, catalysis, electronics, and biomedical devices (Olivetti et al., 2020). Despite its importance, synthesis processes remain largely empirical, relying on trial-and-error approaches guided by expert intuition (Merchant et al., 2023). This inefficiency highlights the need for systematic, datadriven approaches to predict synthesis workflows and optimize experimental design (Huang et al., 2023).

To address these challenges, we introduce Open Materials Guide (OMG), a dataset comprising 17K high-quality, expert-verified synthesis recipes curated from open-access literature. This dataset is the foundation for our benchmark, AlchemyBench, which evaluates synthesis prediction across multiple facets—from inferring raw ma-

Recent progress in machine learning and large language models has opened new avenues for ex-

*These authors contributed equally to this work

[Figure 2]

[Figure 3]

(a) The periodic table of logarithmic frequency of elements. (b) The distribution of synthesis techniques.

- Figure 2: The periodic table (left) demonstrates that OMG covers diverse elements used in target materials, with darker colors indicating higher usage frequencies. A pie chart (right) illustrates the diversity of synthesis methods, highlighting the contributions of prior studies (white) and our dataset (white + green).

terials and recommending appropriate synthesis equipment to generating detailed procedural steps and forecasting suitable characterization techniques.

Additionally, we investigate an LLM-as-aJudge framework to automate the evaluation process. Our systematic comparisons reveal a strong statistical agreement between LLM-based assessments and expert judgments, underscoring the potential of LLMs to serve as scalable, automated evaluators.

Our work makes the following key contributions:

- • Open Materials Guide (OMG), the most significant materials synthesis dataset, comprises 17K high-quality recipes from openaccess literature. We demonstrated that various models can improve their performance with the proposed data-driven RetrievalAugmented Generation (RAG) (Lewis et al.,

2020) experiments. From the improvement, we validate the applicability of our data.

- • AlchemyBench, the first end-to-end benchmark for ML-driven synthesis prediction utilizing LLM-as-a-Judge, a scalable framework for evaluating synthesis predictions, demonstrating strong alignment with expert assessments. This framework enables automated benchmarking of synthesis prediction models, significantly reducing the reliance on costly and time-intensive expert evaluations while maintaining high evaluation reliability.
- • Extensive experimental insights into model

performance, identifying key challenges, potential capabilities, and future directions to utilize LLM for fully-automated materials synthesis.

To enhance reproducibility and accessibility, we release the dataset and code as an open-source resource for the research community1.

### 2 Data Collection and Preparation

#### 2.1 Motivation

Previous large-scale datasets for extracting synthesis procedures from materials science literature have faced several critical challenges (Kononova et al., 2019; Wang et al., 2022). The most significant limitation involves common extraction errors—such as missing reagent concentrations, incorrect reaction temperatures, and misordered procedural steps—which have rendered many outputs unreliable for downstream synthesis prediction (Sun and David, 2025). We analyzed existing datasets and revealed that over 92% of records in Kononova et al. and 98% in Wang et al. lacked essential synthesis parameters (e.g., heating temperature, duration, mixing media). Additionally, these datasets are narrowly focused on a few synthesis techniques (such as solid-state and solutionbased). At the same time, real-world materials innovation employs a broader range of specialized techniques (Xu et al., 2023). Finally, copyright restrictions from commercial journals have limited the legal redistribution of textual synthesis procedures (Authors Alliance, 2024).

1https://github.com/HeegyuKim/ AlchemyBench

[Figure 4]

- Figure 3: An example of extracted recipe from Zhao et al. demonstrates structured annotation of materials, equipment, procedures, and characterization methods.

To overcome these limitations, we propose OMG with three innovations: an LLMdriven parsing approach that improves extraction accuracy, a systematic collection covering more than ten distinct synthesis techniques (including vapor deposition, hydrothermal, and hybrid material systems), and the exclusive use of open-access publications to enable legal distribution of the dataset.

#### 2.2 Dataset Construction

Our pipeline begins by retrieving 28,685 openaccess articles from a pool of 400K search results using the Semantic Scholar API with 60 domain-specific search terms (e.g., “solid state sintering process”, “metal organic CVD”) recommended by domain experts. We convert PDFs to structured Markdown using PyMuPDFLLM (Artifex Software, 2024) and then employ GPT-4o in a multi-stage annotation process. First, articles are categorized based on whether they contain synthesis protocols, target materials, synthesis techniques, and applications. For articles confirmed to include synthesis procedures, the text is segmented into five key components, as illustrated in

- Figure 3:

- • X: A summary of the target material, synthesis method, and application.
- • YM: Raw materials, including quantitative details.
- • YE: Equipment specifications.
- • YP: Step-by-step procedural instructions.
- • YC: Characterization methods and results.

This systematic extraction yielded a dataset of 17,667 high-quality recipes (approximately a 62% yield) covering 10 diverse synthesis methods. Figure 2 demonstrates our dataset’s broad coverage of materials systems and synthesis techniques. Detailed LLM prompts and search keywords are provided in Appendix A.

#### 2.3 Quality Verification

To ensure the accuracy of our automatically extracted recipes, we assembled a panel of eight domain experts from three institutions 2. The experts manually reviewed a representative sample of ten recipes, evaluating them based on the following criteria:

- • Completeness: Capturing the full scope of the reported recipe (X, YM, YE, YP, and YC).
- • Correctness: Extracting critical details such as temperature values and reagent amounts accurately.
- • Coherence: Retaining a logical, consistent narrative without contradictions or abrupt transitions.

Table 1 presents our expert evaluation results using a five-point Likert scale (1 = poor, 5 = excellent). To measure expert agreement, we computed the Intraclass Correlation Coefficient (ICC) (Shrout and Fleiss, 1979), utilizing a twoway mixed-effects model (ICC (3,k)) that quantifies agreement among evaluators, ensuring reliability in subjective scoring. The extracted data exhibited high mean scores, but inter-rater reliability

2Appendix B describes the details about domain experts

- Table 1: Data verification by eight domain experts.

#### Criteria Mean σ ICC (3,k)p−value

Completeness 4.2 0.81 0.695 0.00 Correctness 4.7 0.58 0.258 0.23 Coherence 4.8 0.46 0.429 0.10

varied across criteria, particularly for articles with well-structured experimental sections.

Completeness showed moderate agreement (ICC = 0.695), while correctness (ICC = 0.258) and coherence (ICC = 0.429) had lower agreement due to variations in naming conventions and missing characterization details. Although the completeness score (4.2/5.0) was slightly lower than those for correctness (4.7/5.0) and coherence (4.8/5.0), correctness and coherence exhibited lower inter-rater reliability (ICC = 0.258 and 0.429, respectively), suggesting inconsistencies in how evaluators interpreted minor details. Variability in scores for correctness and coherence arose from differences in how evaluators weighted minor inconsistencies, such as variations in equipment naming or missing characterization information. Some considered these negligible, while others applied stricter criteria, underscoring the need for refined annotation guidelines.

While manual verification confirms the effectiveness of our extraction process, it cannot fully ensure consistent performance across the diverse range of synthesis procedures. In the following section (Section 3), we present a structured evaluation framework for tasks such as raw materials and equipment inference, procedure generation, and characterization outcome forecasting.

### 3 AlchemyBench

We present AlchemyBench, a comprehensive benchmark for evaluating materials synthesis prediction models. This framework addresses key challenges in synthesis recipe evaluation through structured tasks, expert-aligned metrics, and scalable assessment strategies.

#### 3.1 Motivation

Evaluating synthesis predictions presents several fundamental challenges:

• Lack of Benchmarks: No standardized evaluation framework exists, making it challenging to compare synthesis models systemati-

cally. Prior datasets lack critical synthesis parameters and structured ground truth labels, making meaningful comparisons difficult.

- • Limitations of Traditional Metrics: Traditional metrics, such as BLEU (Papineni et al.,

2002) and ROUGE (Lin, 2004) prioritize lexical overlap but fail to capture the procedural correctness of synthesis recipes. Na et al. introduced the Jaccard score to measure set overlap in synthesis procedures, yet it lacks sensitivity to sequential dependencies critical in procedural texts. BERTScore (Zhang et al., 2019) improves contextual similarity measurement but struggles with domain-specific dependencies unique to materials synthesis. Moreover, these metrics do not account for experimental feasibility, limiting their applicability in real-world synthesis.

- • High Cost of Human Evaluation: Expertbased assessments require significant time and resources, averaging 23 minutes per prediction (σ = 7.57) in our experiment. This cost makes large-scale benchmarking impractical, requiring an automated evaluation system.
- • Scalability Requirements: Large-scale benchmarking necessitates an automated yet reliable evaluation system, which LLMs can provide (Gu et al., 2025). However, prior attempts to use LLMs for evaluation lacked systematic validation against human expert assessments in materials science, raising concerns about reliability.

#### 3.2 Task Definition

AlchemyBench simulates real-world synthesis workflows, where models must predict the following components given input X (target material, synthesis method, application domain):

- • PM: Raw materials (e.g., reagents, solvents) with quantities.
- • PE: Required equipment (e.g., furnace, autoclave).
- • PP: Synthesis procedures (e.g., reaction steps, temperatures).
- • PC: Characterization methods and expected outcomes.

- Table 2: Seven evaluation criteria used to evaluate synthesis recipes, categorized into materials, equipment, procedure, characterization, and overall score. Each criterion is rated on a 1–5 scale to reflect the quality and practicality of the predicted recipes.

|Category|Criteria|Description|
|---|---|---|
|[Figure 5]<br><br>Materials|Appropriateness|Are the selected materials suitable for the target synthesis?|
|[Figure 6]<br><br>Equipment|Appropriateness|Is the selected equipment suitable?|
|[Figure 7]<br><br>Procedure|Completeness Similarity Feasibility<br><br>|Is the procedure well-organized and logically structured? How closely does it match the ground truth procedure? Can this procedure be realistically executed in a lab?|
|[Figure 8]<br><br>Characterization|Appropriateness Similarity<br><br>|Are the methods and metrics suitable for validating the success of the synthesized material? How well do predicted properties match actual results?|
|Overall Score|-|Average score considering the recipe’s overall quality and practicality.|

Predictions PX = {PM,PE,PP,PC} are evaluated against ground truth YX = {YM,YE,YP,YC} using the LLM-as-a-Judge framework. Unlike prior benchmarks that rely on lexical similarity, AlchemyBench assesses procedural correctness and experimental feasibility. The evaluation criteria are described in Table 2.

The scoring function is computed as:

NC i=1 Ci

Score(PX,YX) =

NC

where Ci represents the score for criterion i, and NC is the total number of evaluation criteria. These criteria were developed in collaboration with domain experts to ensure alignment with realworld synthesis evaluation.

#### 3.3 Dataset Splits and Distribution

We divided OMG to three splits to ensure robust evaluation:

- • Training Set: 16,026 articles published before 2024.
- • Test - Standard Impact: 1,472 articles (2024 and beyond) from journals with Impact Factor (IF) < 10.
- • Test - High Impact: 169 articles (2024 and beyond) from journals with IF ≥ 10.

The temporal split ensures that models are evaluated on unseen future research, mitigating data contamination. Additionally, stratification by journal impact allows assessment of a model’s ability to process high-impact findings, often introducing novel and complex synthesis

techniques. This split design evaluates both generalizability and the ability to meet the rigorous standards of top-tier journals3.

### 4 LLM as a Judge

A reliable evaluation framework is essential for benchmarking synthesis prediction models. This section examines the alignment between LLMbased and human expert judgments, evaluating inter-rater agreement and assessing the effectiveness of LLMs as automated evaluators.

#### 4.1 Evaluation Metrics

We employ two metrics for evaluating the reliability of metrics: BLEU, ROUGE-L, BERTScore, and our LLM-as-a-Judge approach. Pearson Correlation Coefficient measures how closely LLM scores align with expert ratings on a continuous scale, capturing linear relationships. Finally, the Spearman’s Rank Correlation assesses rankorder consistency, beneficial when the relative ranking of recipes is more informative than absolute scores.

#### 4.2 Human Expert Evaluation Setup

Before evaluating whether the reliability of AlchemyBench assessment aligns with expert evaluations, we enlisted eight materials science researchers from three institutions to establish reliable ground truth. Each evaluator had prior experience in experimental synthesis and was selected based on their publication record and domain expertise. Experts independently assessed model-generated recipes using seven criteria (Table 2) on a 1–5 scale. To ensure high-quality assessments, we collected expert confidence scores

3Table 6 describes the detailed list of high-impact journals utilized for our test-set split.

and highlighted the agreement of one organization of three experts with the highest confidence levels on average (denoted as ‘High’). ICC(3,k) ensures the annotators’ consensus’s reliability.

The dataset for evaluation included ten representative synthesis workflows selected by a senior materials scientist to ensure diversity. Prediction recipes were generated using two models (GPT4o-mini and o1-mini), resulting in 20 unique predictions evaluated by human experts and LLM judges. Appendix B.2 and C describe the experts annotation details and hyperparameters.

#### 4.3 Inter-Expert Agreement Analysis

- Table 3: ICC (3,k) for each criterion. High denotes the highest confidence organization on average, and subscripts denote the p-value.

|Criteria|High (3) All (8)<br><br>|
|---|---|
|Material Appropriateness Equipment Appropriateness Procedure Completeness Procedure Similarity Procedure Feasibility Characterization Appropriateness Characterization Similarity Overall Score (Average)<br><br>|0.61 0.01 0.80 0.00 0.63 0.00 0.63 0.00 0.46 0.05 0.23 0.19 0.34 0.14 0.13 0.31 0.70 0.00 −0.58 0.88 0.45 0.06 0.78 0.00 0.37 0.11 0.45 0.03 0.75 0.00 0.68 0.00|

The comparison between the High Confidence group and the All group in Table 3 highlights key differences in inter-rater reliability. The All group achieves higher ICC values for “Material Appropriateness” (0.80) and “Characterization Appropriateness” (0.78) compared to the High group (0.61 and 0.45, respectively), indicating better consensus among the broader panel for these criteria. However, the High group shows significantly stronger agreement on “Procedure Feasibility” (ICC = 0.70) than the All group, which exhibits a negative ICC value (-0.58), suggesting inconsistencies in feasibility evaluations within the larger group. Both groups display similar reliability for “Equipment Appropriateness” (ICC = 0.63). Overall, while larger panels may enhance agreement on straightforward criteria, smaller high-confidence subgroups provide more consistent evaluations for complex aspects like procedural feasibility.

#### 4.4 LLM-Expert Agreement Analysis

In Table 4, the traditional metrics (BLEU, ROUGE-L, and BERTScore) exhibit low or even

Table 4: Pearson correlation coefficients between evaluation metrics and domain expert consensus for overall score. Subscripts denote the p-values. We set reasoning effort to high for o3-mini.

|Model|High (3) All (8)<br><br>|
|---|---|
|BLEU ROUGE-L BERTScore GPT-4o-mini GPT-4o-Aug GPT-4o-Nov o3-mini (high)|−0.16 0.50 −0.23 0.34 0.06 0.80 −0.12 0.60 −0.30 0.19 −0.24 0.31<br><br>0.61 0.00 0.45 0.05<br><br>0.80 0.00 0.61 0.01 0.63 0.00 0.75 0.00<br><br>0.62 0.02 0.47 0.03<br>|

negative correlations with the domain expert consensus regardless of the evaluator group, whereas the LLM-based scores consistently yield higher and statistically significant correlations. The values obtained for GPT-4o-mini, GPT-4o-Aug, and

- o3-mini (high) are notably higher in the high confidence subgroup—0.61, 0.80, and 0.62 respectively—compared to 0.45, 0.61, and 0.47 for the full panel, suggesting that evaluations from the more confident experts are more tightly aligned with these models. In contrast, GPT-4o-Nov shows a higher correlation with all eight experts (0.75) than with the high confidence subset (0.63), indicating that its performance remains robust even when considering a broader range of expert opinions. Overall, the comparison underscores the influence of expert group composition on evaluation
- outcomes and highlights the superior alignment of advanced LLM evaluators with expert assessments over traditional similarity metrics4.

Our experiment confirms that LLM-generated scores correlate significantly better with expert assessments, supporting their use as scalable synthesis evaluators.

#### 4.5 Summary and Implications

Our findings demonstrate that LLM-based evaluation provides a scalable and effective alternative to traditional synthesis assessment methods. GPT4o-Aug exhibits strong agreement with expert ratings, outperforming traditional NLP metrics.

However, challenges remain, as LLMs can be sensitive to ambiguous phrasing and domain-

4Spearman correlation scores are described in Appendix B.3.

specific biases, affecting evaluation consistency. Future work should explore hybrid approaches integrating expert feedback with LLM scoring. Reinforcement learning from human feedback (Ouyang et al., 2022) (RLHF) and domainspecific fine-tuning (Anisuzzaman et al., 2025) may improve alignment with expert reasoning.

Future work should investigate methods for mitigating biases and inconsistencies to enhance reliability, such as integrating expert validation into LLM-based evaluation pipelines. This study highlights the potential of LLMs as automated evaluators, paving the way for AI-driven, context-aware benchmarking frameworks in materials science.

### 5 Experiments

This section evaluates five LLMs on our benchmark using the LLM-as-a-Judge framework based on GPT-4o-Aug, analyzing performance across multiple metrics and the impact of retrievalaugmented generation (RAG).

#### 5.1 Experiment Setup

To comprehensively evaluate the models, we conducted experiments with the following setup:

Base LLMs We evaluated four LLMs, including reasoning-based models (o3-mini) and generalpurpose models (GPT-4o variants). The knowledge cutoff of these models is October 2023, minimizing potential data contamination in our test sets, which contain 2024 and beyond5. We prompt the LLM with a fixed one-shot example from our train set to predict all components (PX) well6. Moreover, we varied the reasoning effort of o3mini to ensure the effectiveness of reasoning effort in materials synthesis.

Evaluation Framework Each model generated synthesis recipes for both the High Impact set and Standard Impact set. Recipes were evaluated using our LLM-as-a-Judge method based on GPT4o-Aug. The evaluation criteria focused on material appropriateness, procedural feasibility, and overall recipe quality.

Retrieval-Augmented Generation (RAG) To evaluate the impact of retrieval on recipe generation, we implemented a RAG pipeline using

- 5The knowledge cutoff of OpenAI’s models is described in this documentation.
- 6Details about LLM prompt and hyperparameters are described in Appendix C.

OpenAI’s text-embedding-3-large model (OpenAI, 2022). For each input X, we retrieved the top-K most similar recipes from the train set based on cosine similarity and included them as references in LLM prompts. We evaluated K = {0,1,5,10,25} to assess the effect of contextual information.

This experimental setup ensures a thorough evaluation of both baseline performance and improvements achieved through retrieval augmentation. Due to computational constraints, RAG experiments were conducted on three representative models (GPT-4o-Nov, GPT-4o-mini, and o3-mini) using only the High Impact set.

#### 5.2 Insights from Experimental Results

Table 5: Base LLMs’ overall score evaluated by LLMas-a-judge. Subscripts denote the standard deviation.

High Impact Standard Impact Model Mean σ Max Mean σ Max GPT-4o-mini 3.238 0.432 4.50 3.412 0.412 4.71 GPT-4o-Aug 3.362 0.405 4.50 3.508 0.397 4.71 GPT-4o-Nov 3.709 0.410 4.71 3.398 0.397 4.71

- o3-mini (medium) 3.714 0.411 4.64 3.822 0.387 4.80
- o3-mini (high) 3.759 0.407 4.71 3.885 0.377 4.80

The experimental results provide valuable insights into the challenges and opportunities in materials synthesis prediction, structured around the following research questions:

- RQ1: Is High-Impact Set More Challenging than Standard-Impact Set? The results confirm that High Impact set is indeed more challenging than Standard Impact set. Across models without GPT-4o-Nov, average scores on the High Impact set were generally lower than those on the Standard Impact set. For example, o3-mini (high) achieved an average score of 3.759 ± 0.407 on High Impact set compared to 3.885 ± 0.377 on Standard Impact set (Table 5). This discrepancy highlights the increased complexity of HighImpact synthesis workflows, which often involve novel materials or cutting-edge techniques requiring greater reasoning and contextual understanding.
- RQ2: Does Increasing Reasoning Effort Improve Recipe Quality in Materials Science? Materials science relies heavily on trial-anderror experimentation, making reasoning-based approaches particularly relevant for complex syn-

thesis tasks. To explore this, we evaluated the o3mini model using low, medium, and high reasoning efforts. As shown in Table 5, o3-mini (high) achieved the highest mean scores across both High-Impact (3.759) and Standard-Impact (3.885) sets, outperforming both its low and medium reasoning efforts and general-purpose models like GPT-4o-Nov. o3-mini (high) exhibited superior performane in step-by-step instructions compared to GPT-4o-Nov, which achieved a lower mean score (3.709) on the High Impact set despite matching o3-mini (high)’s maximum score (4.71). These findings demonstrate the importance of structured problem-solving capabilities in materials synthesis tasks.

[Figure 9]

- Figure 4: Impact of the Retrieval Augmented Generation (RAG) in High Impact set.

RQ3: Does Recipes of Similar Contributions Improve Prediction Performance? RAG with similar recipes significantly enhances model performance through domain-relevant examples. As shown in Figure 4, increasing K improves scores across all models, with different patterns between reasoning-based and general-purpose models. For

- o3-mini (high), performance diminish after K = 5, reaching a maximum score of 4.0. In contrast, GPT-4o-Nov shows continuous improvement up to K = 25, achieving 3.98. The close performance between GPT-4o-Nov and o3-mini (high) suggests RAG can effectively bridge the gap between general-purpose and reasoning-based models. These results highlight two key points: (1) RAG benefits general-purpose models that rely
- on external context, and (2) retrieval effectiveness depends on model architecture and training data quality. Our findings suggest that K = 5 provides the best trade-off between context enrichment and cognitive load, as additional references beyond this point yield minimal improvements.

### 6 Related Work

Materials Synthesis Datasets Existing materials synthesis datasets, such as those focusing on solid-state (Kononova et al., 2019) and solutionbased (Wang et al., 2022) methods, have provided valuable resources for machine learning applications. However, these datasets often suffer from issues of incompleteness and low quality, with many synthesis procedures lacking critical parameters necessary for reproducibility or predictive modeling. For instance, only 28% of solid-state synthesis paragraphs yield complete reactions, and over 90% of recipes are missing key parameters. These limitations hinder their utility in guiding novel synthesis workflows.

LLM-based Generation for Materials Science Large language models (LLMs) have shown promise in accelerating materials discovery by automating hypothesis generation (Kumbhar et al., 2025), property prediction (Chiang et al., 2024), and evaluation (Mishra et al.). However, their effectiveness is constrained by the lack of highquality domain-specific datasets and the need for retrieving or fine-tuning to handle complex synthesis workflows. Our work addresses these gaps and introduces a large-scale dataset and benchmark tailored to the real-world synthesis workflow, enabling rigorous evaluation of LLM capabilities in materials science.

### 7 Conclusion

This study presents a comprehensive benchmark for evaluating LLMs in materials synthesis prediction, addressing key challenges in data-driven materials science. By curating a large-scale dataset and designing tasks that mirror real-world synthesis workflows, we provide a robust framework to assess model capabilities in raw materials selection, equipment inference, procedure generation, and characterization prediction. Our experiments reveal the potential of reasoning-based models, such as o3-mini, outperforming general-purpose models like GPT-4o variants in generating coherent and feasible synthesis recipes. Furthermore, integrating retrieval-augmented generation (RAG) enhances recipe quality by grounding predictions in domain-relevant examples, with optimal performance gains observed at K = 5. These findings underscore the importance of combining advanced reasoning architectures with adaptive re-

trieval strategies for materials science tasks, laying the foundation for interdisciplinary innovation and accelerating progress in data-driven and fullyautomated materials discovery.

### 8 Limitations

While our benchmark represents a significant step toward integrating LLMs into materials synthesis, several limitations remain. First, the dataset, derived from open-access articles, may exhibit biases in domain coverage, overrepresenting fields like battery materials while underrepresenting others like biomaterials. Second, GPT-4o for recipe extraction and evaluation introduces potential inaccuracies and biases, particularly in complex or ambiguous texts. Third, while practical, reliance on LLM-based scoring may oversimplify the nuanced requirements of tasks like procedure generation and characterization prediction. Additionally, the sequential dependencies between tasks (e.g., precursor prediction influencing procedure generation) pose challenges for current models, which may overfit dataset-specific patterns rather than learning generalizable principles. Finally, the lack of interpretability in model outputs limits their applicability in critical experimental workflows. Addressing these issues through improved data curation, expanded evaluation frameworks, and developing more interpretable models will be vital for future progress in this domain.

### References

DM Anisuzzaman, Jeffrey G Malins, Paul A Friedman, and Zachi I Attia. 2025. Fine-tuning large language models for specialized use cases. Mayo Clinic Proceedings: Digital Health, 3(1).

Artifex Software. 2024. Pymupdf4llm: Pdf text extraction library for llm applications. https:// github.com/artifex-com/pymupdf4llm. Accessed: January 2024.

Authors Alliance. 2024. Text and data mining under u.s. copyright law: Landscape, flaws & recommendations. Technical report.

BerriAI. Litellm. https://github.com/ BerriAI/litellm.

Yuan Chiang, Elvis Hsieh, Chia-Hong Chou, and Janosh Riebesell. 2024. Llamp: Large language model made powerful for high-fidelity materials knowledge retrieval and distillation. Preprint, arXiv:2401.17244.

Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazar´e, Maria Lomeli, Lucas Hosseini, and Herv´e J´egou. 2024. The faiss library.

Alexander Dunn, Qi Wang, Alex Ganose, Daniel Dopp, and Anubhav Jain. 2020. Benchmarking materials property prediction methods: the matbench test set and automatminer reference algorithm. npj Computational Materials, 6(1):138.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. 2025. A survey on llm-as-a-judge. Preprint, arXiv:2411.15594.

Guannan Huang, Yani Guo, Ye Chen, and Zhengwei Nie. 2023. Application of machine learning in material synthesis and property prediction. Materials, 16(17):5977.

Olga Kononova, Haoyan Huo, Tanjin He, Ziqin Rong, Tiago Botari, Wenhao Sun, Vahe Tshitoyan, and Gerbrand Ceder. 2019. Text-mined dataset of inorganic materials synthesis recipes. Scientific data, 6(1):203.

Shrinidhi Kumbhar, Venkatesh Mishra, Kevin Coutinho, Divij Handa, Ashif Iquebal, and Chitta Baral. 2025. Hypothesis generation for materials discovery and design using goal-driven and constraint-guided llm agents. Preprint, arXiv:2501.13299.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474.

Quentin Lhoest, Albert Villanova del Moral, Yacine Jernite, Abhishek Thakur, Patrick von Platen, Suraj Patil, Julien Chaumond, Mariama Drame, Julien Plu, Lewis Tunstall, Joe Davison, Mario Saˇˇ sko, Gunjan Chhablani, Bhavitvya Malik, Simon Brandeis, Teven Le Scao, Victor Sanh, Canwen Xu, Nicolas Patry, Angelina McMillan-Major, Philipp Schmid, Sylvain Gugger, Cl´ement Delangue, Th´eo Matussi`ere, Lysandre Debut, Stas Bekman, Pierric Cistac, Thibault Goehringer, Victor Mustar, Fran¸cois Lagunas, Alexander Rush, and Thomas Wolf. 2021. Datasets: A community library for natural language processing. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 175–184, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Amil Merchant, Simon Batzner, Samuel S Schoenholz, Muratahan Aykol, Gowoon Cheon, and Ekin Dogus Cubuk. 2023. Scaling deep learning for materials discovery. Nature, 624(7990):80–85.

Vaibhav Mishra, Somaditya Singh, Mohd Zaki, Hargun Singh Grover, Santiago Miret, NM Anoop Krishnan, et al. Llamat: Large language models for materials science. In AI for Accelerated Materials Design-Vienna 2024.

Gyoung S Na. 2023. Artificial intelligence for learning material synthesis processes of thermoelectric materials. Chemistry of Materials, 35(19):8272–8280.

Pengcheng Xu, Xiaobo Ji, Minjie Li, and Wencong Lu.

2023. Small data machine learning in materials science. npj Computational Materials, 9(1):42.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675.

Yu Zhao, Dongru Gao, Ruxin Guan, Hongwei Li, Ning Li, Guixian Li, and Shiyou Li. 2020. Synthesis of a three-dimensional cross-linked ni–v 2 o 5 nanomaterial in an ionic liquid for lithium-ion batteries. RSC advances, 10(64):39137–39145.

Elsa A Olivetti, Jacqueline M Cole, Edward Kim, Olga Kononova, Gerbrand Ceder, Thomas Yong-Jin Han, and Anna M Hiszpanski. 2020. Data-driven materials research enabled by natural language processing and information extraction. Applied Physics Reviews, 7(4).

OpenAI. 2022. Introducing text and code embeddings. https://openai.com/index/ introducing-text-and-code-embeddings/. [Accessed 11-02-2025].

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318.

Semantic Scholar. 2023. Semantic scholar academic graph api. https://www. semanticscholar.org/product/api. Accessed: January 2025.

Patrick E Shrout and Joseph L Fleiss. 1979. Intraclass correlations: uses in assessing rater reliability. Psychological bulletin, 86(2):420.

Yu Song, Santiago Miret, and Bang Liu. 2023. Matscinlp: Evaluating scientific language models on materials science language tasks using text-to-schema modeling. arXiv preprint arXiv:2305.08264.

Wenhao Sun and Nicholas David. 2025. A critical reflection on attempts to machine-learn materials synthesis insights from text-mined literature recipes. Faraday Discussions.

Zheren Wang, Olga Kononova, Kevin Cruse, Tanjin He, Haoyan Huo, Yuxing Fei, Yan Zeng, Yingzhi Sun, Zijian Cai, Wenhao Sun, et al. 2022. Dataset of solution-based inorganic materials synthesis procedures extracted from the scientific literature. Scientific data, 9(1):231.

[Figure 10]

- Figure 5: Yearly distribution of collected material synthesis papers

### A Additional Dataset Information

This section provides extended details on dataset statistics and collection methodology that were not included in the main text for brevity.

- A.1 Keyword Selection Rationale

Figure 9 describes the search keywords to retrieve 400K articles using Semantic Scholar API (Semantic Scholar, 2023). We collect these keywords guided by our eight domain experts.

- A.2 Downloading PDFs

We downloaded open-access papers exclusively from the following six publishers, most frequent in our retrieval result: pubs.rsc.org, mdpi.com, nature.com, link.springer.com, pubs.acs.org, onlinelibrary.wiley.com.

- A.3 Dataset Details

Figure 6a, 6c, and 6b demonstrate the distributions of venue for train, test-standard-impact, and testhigh-impact respectively. Table 6 describes the high-impact venues (IF ≥ 10) that at least ten papers are included in OMG. Figure 5 demonstrates the dataset distribution of the published year, indicating the latest data, 2020 and beyond, accounts for a large percentage.

### B Annotation Pipeline and Quality Checks

Here, we elaborate on the annotation workflow and the validation methods used to ensure the reliability of extracted recipes.

#### B.1 GPT-4 Prompts for Classification and Extraction

Figure 10 demonstrates the prompt to categorize the literature and Figure 11 demonstrates the prompt to extract the synthesis recipe from the literature.

#### B.2 Expert Review Protocol

Table 7 describes the anonymized details about the eight domain experts in materials science. They participated as volunteers and received no evaluation fees. Figure 7 and 8 demonstrate the web UI screenshots for evaluating LLM predictions by domain experts. Domain experts evaluated 20 LLM predictions with seven criteria in Table 2 and recorded the results in a spreadsheet. We aggregated those eight spreadsheets and calculated the agreement.

#### B.3 LLM-Expert Agreement Details

The agreement analysis between expert groups (entire 8-member panel vs. 3-member highconfidence group) and GPT-4o-Nov reveals distinct patterns across evaluation criteria, as shown in Tables 8 and 9. These results highlight the critical influence of expert group composition on LLM alignment assessment. Compared to the entire panel, the high-impact subgroup demonstrates enhanced agreement on procedural elements but reduced consensus on characterization tasks, suggesting domain-specific expertise differentially weights evaluation criteria. The stability of non-significant results across both groups for equipment and feasibility judgments implies fundamental challenges in consistently operationalizing these metrics.

### C Experimental and Implementation Details

This section thoroughly describes the LLM prompts and hyperparameter settings to facilitate reproducibility.

#### C.1 Hyperparameters

We use a temperature of zero, top-p of 1.0, and max tokens of 4096 for GPT-4o-mini and GPT-4o variants. o3-mini variants use max completion tokens of 16384, and OpenAI does not allow to set temperature and top-p hyperparameters for o3-mini models.

Table 6: A list of high-impact journals (IF ≥ 10) that at least ten papers are included in OMG.

|Publisher|Journals<br><br>|
|---|---|
|ACS RSC Nature<br><br>Wiley<br><br>Springer|ACS Applied Materials and Interfaces, ACS Nano, ACS Energy Letters, Journal of the American Chemical Society, ACS Catalysis Journal of Materials Chemistry A, Chemical Society Reviews, Energy Environmental Science Nature Communications, Nature Materials, Nature Nanotechnology, Nature Energy, Nature Reviews Materials, Nature Catalysis, Nature, Nature Electronics, Nature Methods, Nature Chemistry, Nature Physics, Light: Science Applications Advanced Materials, Advanced Energy Materials, Small, Angewandte Chemie, Advanced Science, ChemSusChem, Advanced Functional Materials Nano-Micro Letters, Journal of Advanced Ceramics, Advanced Composites and Hybrid Materials|

- Table 7: Anonymized details for the domain experts in our study. Conf. denotes the average confidence for evaluating LLM predictions in Section 4. Group C is the highest confidence group.

|Group|Expertise<br><br>|Conf.|
|---|---|---|
|A<br>B<br>C<br>|One master and two PhD candidates specialized in:<br><br>- Thin film transistors<br>- 3D nano-semiconductor thin films<br>- atomic layer deposition Two master’s student specialized in:<br><br>- Materials modeling<br>- DFT & MD simulations<br>- NLP for materials science Three master’s student specialized in:<br><br>- Transparent electrodes<br>- electrospinning<br>- 2D materials<br>|1.90<br><br>3.15<br><br>4.47|

- Table 8: A agreement between entire expert consensus and GPT-4o-Nov for each criterion. Subscripts denote the p−value.

|Category<br><br>|Pearson Spearman|
|---|---|
|Material Appropriateness Equipment Appropriateness Procedure Completeness Procedure Similarity Procedure Feasibility Characterization Appropriateness Characterization Similarity<br><br>|0.59 0.01 0.59 0.01<br><br>-0.25 0.29 -0.25 0.28 0.05 0.83 0.09 0.71 0.41 0.07 0.40 0.08<br>-0.04 0.86 -0.04 0.86 0.43 0.06 0.42 0.07 0.45 0.05 0.47 0.04<br>|

- Table 9: A agreement between the expert consensus of the high-confidence group and GPT-4o-Nov for each criterion. Subscripts denote the p−value.

|Category<br><br>|Pearson Spearman|
|---|---|
|Material Appropriateness Equipment Appropriateness Procedure Completeness Procedure Similarity Procedure Feasibility Characterization Appropriateness Characterization Similarity|0.44 0.05 0.41 0.07 0.10 0.68 0.13 0.58 0.23 0.33 0.20 0.39 0.56 0.01 0.50 0.02<br><br>-0.04 0.86 -0.04 0.86 0.43 0.06 0.42 0.07 0.09 0.72 0.16 0.50<br><br>|

- C.2 LLM Prompt

Figure 14 describes the LLM-as-a-Judge prompt. The LLM outputs the JSON-formatted judgment of seven criteria and an overall score for extraction.

- C.3 Other Artifacts

We utilized LiteLLM (BerriAI) and FAISS (Douze et al., 2024), Huggingface Datasets (Lhoest et al., 2021). We confirmed that all models, datasets, and frameworks are allowed for research use.

- D Additional Results and Analysis

Table 10 describes the detailed result of RAG experiments in Figure 4 for four base LLMs.

- E Ethical Considerations and Potential Risks

Our data collection approach exclusively utilized open-access publications from six major publishers to ensure copyright compliance. Additionally, we verified the 12958 articles through keywordbased content filtering and selenium-confirmed articles of CC-BY licensing status, supplemented by

Table 10: A full experiment result of RAG prediction in Section 5

|Model K<br><br>|Mean σ Min Max|
|---|---|
|o3-mini-high<br><br>0<br><br>1 5<br><br><br>10 25<br><br>|3.759 0.407 2.86 4.71<br><br>3.937 0.401 2.80 4.86<br>4.001 0.384 3.00 4.80 3.939 0.359 3.00 4.80 3.986 0.383 3.00 4.71<br>|
|o3-mini-medium<br><br>0<br>1 5<br><br><br>10 25|3.714 0.411 2.86 4.64 3.867 0.381 3.00 4.80 3.934 0.349 2.86 4.80 3.937 0.390 2.57 4.80 3.975 0.413 2.50 4.80<br><br>|
|o3-mini-low<br><br>0<br><br>1 5<br><br><br>10 25<br><br>|3.676 0.407 2.50 4.57 3.848 0.411 2.90 4.80 3.904 0.393 2.93 4.86 3.917 0.397 2.86 4.86 3.961 0.388 3.00 4.80|
|GPT-4o Nov<br><br>0<br><br>1 5<br><br><br>10 25<br><br>|3.709 0.410 2.75 4.71 3.824 0.417 2.57 4.80 3.949 0.391 2.93 4.80 3.954 0.366 3.00 4.80 3.976 0.375 2.93 4.86|

a manual sampling of 100 randomly selected articles to validate redistribution rights. While this strategy mitigates legal risks, two potential limitations warrant consideration: First, the open-access corpus may exhibit selection bias toward wellfunded research domains (e.g., energy materials) versus proprietary industrial methods. Second, automated extraction via GPT-4o risks propagating subtle errors from source documents, particularly in stoichiometric ratios and procedural sequencing, despite our expert validation protocol. All dataset derivatives will be distributed under original CC-BY licenses.

### F AI Assitant

We use Microsoft Copilot as a coding assistant and Grammarly and Writefull as a writing assistant.

[Figure 11]

(a) A venue distribution of the train set

[Figure 12]

- (b) A venue distribution of the test high-impact set

[Figure 13]

- (c) A venue distribution of the test standard impact set

Figure 6: Venue distributions across datasets: (a) training set, (b) test high-impact set, and (c) test standard impact set. The distributions illustrate the diversity and focus of venues in each subset.

[Figure 14]

##### Figure 7: A web UI screenshot for domain experts annotation (1/2).

[Figure 15]

##### Figure 8: A web UI screenshot for domain experts annotation (2/2).

|# Solid-State Processing solid state sintering process, reactive sintering synthesis, pressure-assisted sintering, spark plasma sintering, hot pressing synthesis, hot isostatic pressing, cold isostatic pressing, flash sintering technique, field-assisted sintering, microwave sintering process,<br><br># Mechanochemical Methods high energy ball milling, mechanical alloying synthesis, mechanochemical activation, planetary ball milling, cryogenic milling process, attrition milling synthesis, mechanical grinding method, mechanofusion process, mechano-chemical reaction, solid-state mechanical synthesis,<br><br># Vapor Deposition Techniques atomic layer deposition, plasma enhanced CVD, metal organic CVD, low pressure CVD, atmospheric pressure CVD, electron beam PVD, magnetron sputtering deposition, pulsed laser deposition, thermal evaporation method, molecular beam epitaxy,<br><br># Advanced Thermal Methods combustion synthesis process, self-propagating synthesis, plasma spray synthesis, flame spray pyrolysis, laser ablation synthesis, thermal plasma synthesis, microwave-assisted synthesis, ultrasonic spray pyrolysis, radio frequency thermal plasma, arc discharge synthesis,<br><br># Electrochemical Approaches electrochemical co-deposition, pulse electrodeposition, electroless deposition method, anodic oxidation synthesis, cathodic reduction process, electrochemical etching, electrochemical polymerization, electrophoretic deposition, galvanic replacement reaction, electrochemical exfoliation,<br><br># Novel Processing Methods freeze drying synthesis, spray freeze drying, supercritical fluid process, template-assisted synthesis, biomimetic processing method, sol-gel electrospinning, ionothermal synthesis route, microemulsion technique, sonochemical processing, continuous flow synthesis<br><br>|
|---|

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27
- 28
- 29
- 30
- 31
- 32
- 33
- 34
- 35
- 36
- 37
- 38
- 39
- 40
- 41

##### Figure 9: 60 search keywords to retrieve the literature, including materials synthesis recipes using Semantic Scholar API.

|Analyze the given scientific text and provide classifications in the following order :<br><br>1. Synthesis Recipe Classification: Determine if the text contains detailed synthesis procedures. Return only "YES" or "NO". If "NO", stop here. If "YES", continue with the following classifications.<br><br>2. Target Classification: Classify the synthesized target as one of:<br><br>- Material (e.g., nanoparticles, compounds, composites)<br><br>- Device (e.g., sensors, batteries, transistors)<br><br>- Molecule (e.g., organic compounds, polymers)<br><br><br>3. Material Identification: Provide:<br><br>- Chemical formula (if applicable)<br><br>- Material name<br><br>- Material class (e.g., metal oxide, polymer, semiconductor)<br><br><br>4. Application Domain: List the primary applications mentioned in the text:<br><br>- Energy (e.g., batteries, solar cells)<br><br>- Electronics (e.g., transistors, sensors)<br><br>- Healthcare (e.g., drug delivery, imaging)<br><br>- Environmental (e.g., catalysis, filtration)<br><br>- Others (specify)<br><br><br>5. Synthesis Process Classification: Classify the given synthesis method into one of these categories. If it combines multiple methods, label it as "Hybrid". If it doesn’t fit any category, label it as "Others". Categories:<br><br><br>1. Solid-State: solid-state reaction, ceramic method, sintering<br><br>2. Vapor Deposition: CVD, PVD, sputtering, evaporation<br><br>3. Mechanochemical: ball milling, mechanical alloying<br><br>4. Hydrothermal: solvothermal, pressurized solution<br><br>5. Pyrolysis: thermal decomposition, spray pyrolysis<br><br>6. Melt Quenching: rapid solidification, glass formation<br><br>7. Electrochemical: electrodeposition, anodization<br><br>8. Self-Assembly: molecular assembly, biomineralization<br><br>9. Solution-Based: precipitation, sol-gel, wet chemical synthesis<br><br>10. Biological: biomimetic, enzyme-mediated, microbial synthesis<br><br>11. Hybrid: combination of multiple methods<br><br>12. Others: novel or unconventional methods<br><br><br>Format the output as a structured list only if Step 1 is "YES". For not available, use "N/A". Do not provide explanations or additional commentary.<br><br>Example Output: For a paper titled "Hydrothermal Synthesis of LiFePO4/C Composites for HighPerformance Lithium-Ion Batteries":<br><br>1. Synthesis Recipe: YES<br><br>2. Target: Material<br><br>3. Material Identification:<br><br>- Chemical Formula: LiFePO4/C<br><br>- Material Name: Carbon-coated lithium iron phosphate<br><br>- Material Class: Phosphate composite<br><br><br>4. Application Domain: Energy (lithium-ion batteries)<br><br>5. Synthesis Process: Hydrothermal (solvothermal)<br><br><br>Scientific Paper: {text}<br><br>|
|---|

- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27
- 28
- 29
- 30
- 31
- 32
- 33
- 34
- 35
- 36
- 37
- 38
- 39
- 40
- 41
- 42
- 43
- 44
- 45
- 46
- 47
- 48
- 49
- 50
- 51
- 52
- 53
- 54
- 55
- 56
- 57
- 58
- 59
- 60
- 61
- 62
- 63
- 64

##### Figure 10: System prompt to categorize the literature converted to markdown format.

|You are a materials science expert. Your task is to extract ONLY the explicitly stated synthesis information from the provided research paper. Do not generate, assume, or infer any information not directly presented in the paper. If the provided paper does not contain any synthesis information, please indicate " NOT A MATERIAL SYNTHESIS PAPER" and do not provide any further details.<br><br>## Key Contributions Summarize the key contributions of the paper:<br><br>- Novel materials or compounds: <summary><br><br>- Unique synthesis methods: <summary><br><br>- Specific applications or domains: <summary> ## Materials Extract and list:<br><br>- All precursor materials with:<br><br>* Exact quantities and concentrations<br><br>* Molar ratios or stoichiometric proportions<br><br>* Purity grades and specifications<br><br>* Supplier information if provided<br><br><br>- Solvents, reagents, catalysts, and any other materials such as carrier gases. ## Synthesis Equipment<br><br>- All equipment and apparatus with:<br><br><br>* Model numbers if specified<br><br>* Operating parameters<br><br>* Special configurations or modifications<br><br><br>## Synthesis Procedure Extract and organize:<br><br>- Chronological step-by-step synthesis method<br><br>- All processing parameters:<br><br>* Temperature ranges and ramp rates<br><br>* Time durations for each step<br><br>* Pressure conditions<br><br>* pH values if applicable<br><br>* Mixing speeds and durations<br><br><br>- Critical control points and special conditions ## Characterization Methods and Equipment List all:<br><br>- Analytical techniques used<br><br>- Specific measurement conditions<br><br>- Sample preparation methods<br><br>- Equipment models and settings<br><br>- Standards or references used ## Product Characteristics Document:<br><br>- Final product properties and specifications (include both numerical values and literal descriptions if provided)<br><br>- Yield calculations and actual yields<br><br>- Purity levels and impurity content<br><br>- Performance metrics with measured values<br><br>- Morphological characteristics IMPORTANT RULES:<br><br><br>1. DO NOT generate or assume any missing information<br><br>2. If specific details are not mentioned in the paper, indicate "N/A"<br><br>3. Use exact numbers and units as presented in the paper<br><br>4. Maintain original measurement units<br><br>5. Quote unusual or specific procedures directly when necessary<br><br>6. Format all information using proper markdown with headers (##) and bullet points<br><br><br>Remember: Accuracy and authenticity are crucial. Only include information explicitly stated in the paper.<br><br>Scientific Paper: {text}<br><br>|
|---|

- 18
- 19
- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27
- 28
- 29
- 30
- 31
- 32
- 33
- 34
- 35
- 36
- 37
- 38
- 39
- 40
- 41
- 42
- 43
- 44
- 45
- 46
- 47
- 48
- 49
- 50
- 51
- 52
- 53
- 54
- 55
- 56
- 57
- 58
- 59
- 60
- 61
- 62
- 63

##### Figure 11: System prompt to extract the recipe from literature converted to markdown format.

|You are a materials science expert. Your task is to extract ONLY the explicitly stated synthesis information from the provided research paper. Do not generate, assume, or infer any information not directly presented in the paper. If the provided paper does not contain any synthesis information, please indicate " NOT A MATERIAL SYNTHESIS PAPER" and do not provide any further details.<br><br>## Key Contributions Summarize the key contributions of the paper:<br><br>- Novel materials or compounds: <summary><br><br>- Unique synthesis methods: <summary><br><br>- Specific applications or domains: <summary> ## Materials Extract and list:<br><br>- All precursor materials with:<br><br>* Exact quantities and concentrations<br><br>* Molar ratios or stoichiometric proportions<br><br>* Purity grades and specifications<br><br>* Supplier information if provided<br><br><br>- Solvents, reagents, catalysts, and any other materials such as carrier gases. ## Synthesis Equipment<br><br>- All equipment and apparatus with:<br><br><br>* Model numbers if specified<br><br>* Operating parameters<br><br>* Special configurations or modifications<br><br><br>## Synthesis Procedure Extract and organize:<br><br>- Chronological step-by-step synthesis method<br><br>- All processing parameters:<br><br>* Temperature ranges and ramp rates<br><br>* Time durations for each step<br><br>* Pressure conditions<br><br>* pH values if applicable<br><br>* Mixing speeds and durations<br><br><br>- Critical control points and special conditions ## Characterization Methods and Equipment List all:<br><br>- Analytical techniques used<br><br>- Specific measurement conditions<br><br>- Sample preparation methods<br><br>- Equipment models and settings<br><br>- Standards or references used ## Product Characteristics Document:<br><br>- Final product properties and specifications (include both numerical values and literal descriptions if provided)<br><br>- Yield calculations and actual yields<br><br>- Purity levels and impurity content<br><br>- Performance metrics with measured values<br><br>- Morphological characteristics IMPORTANT RULES:<br><br><br>1. DO NOT generate or assume any missing information<br><br>2. If specific details are not mentioned in the paper, indicate "N/A"<br><br>3. Use exact numbers and units as presented in the paper<br><br>4. Maintain original measurement units<br><br>5. Quote unusual or specific procedures directly when necessary<br><br>6. Format all information using proper markdown with headers (##) and bullet points<br><br><br>Remember: Accuracy and authenticity are crucial. Only include information explicitly stated in the paper.<br><br>Scientific Paper: {text}<br><br>|
|---|

18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46

47 48 49 50 51 52 53 54 55 56 57 58 59 60

61 62 63

##### Figure 12: A prompt to predict the recipe with a one-shot example.

|You are an expert in material science and chemical synthesis. Your task is to design a detailed material synthesis recipe, considering the following **key contributions<br><br>**: Your output should follow the exact structure and format of the below references. Provide precise details for each section, including materials, equipment, and stepby-step procedures.<br><br>Here are recipes from research papers with similar contributions.<br><br>-- References -{references}<br><br>-- Prediction Input -Based on these references, predict the synthesis recipe for the given key contributions: {contributions}<br><br>|
|---|

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10

##### Figure 13: A prompt to predict the recipe using retrieval-augmented generation

|You are an expert materials scientist tasked with evaluating an AI-generated synthesis recipe. You will be provided with:<br><br>1. Target material description<br><br>2. AI-generated recipe<br><br>3. Ground truth recipe from literature<br><br><br>Please evaluate the AI-generated recipe according to the following criteria on a scale of 1-5 (1.0: poor, 5.0: excellent, 0.5 step). Provide detailed justification for each score.<br><br>Guidelines for evaluation:<br><br>1. Materials<br><br>- Appropriateness: Are the selected materials suitable for the target synthesis?<br><br>2. Equipment<br><br>- Appropriateness: Is the selected equipment suitable?<br><br>3. Procedure<br><br>- Completeness: Are all necessary steps included with sufficient detail?<br><br>- Similarity: How closely does it match the ground truth procedure?<br><br>- Feasibility: Can this procedure be realistically executed in a lab?<br><br><br>4. Characterization<br><br><br>- Appropriateness: Are the metrics suitable for validating success?<br><br>- Similarity: How well do predicted properties match actual results? Your Output format must be as following:<br><br><br>1. Step-by-step reasoning first<br><br>2. JSON score result ‘‘‘json {<br><br><br>"materials_appropriateness_score": float, "equipment_appropriateness_score": float, "procedure_completeness_score": float, "procedure_similarity_score": float, "procedure_feasibility_score": float, "characterization_appropriateness_score": float, "characterization_similarity_score": float, "overall_score": float<br><br>} ‘‘‘<br><br>|
|---|

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27
- 28
- 29
- 30
- 31
- 32
- 33
- 34
- 35
- 36
- 37
- 38
- 39

##### Figure 14: A prompt to judge the prediction recipe using LLM-as-a-Judge

