## SearchInstruct: Enhancing Domain Adaptation via Retrieval-Based Instruction Dataset Creation

Iman Barati1 Mostafa Amiri2 Heshaam Faili2 1 Iran University of Science and Technology 2 University of Tehran iman_barati@comp.iust.ac.ir {mostafa.amiri,hfaili}@ut.ac.ir

# arXiv:2509.10708v1[cs.CL]12Sep2025

### Abstract

Supervised Fine-Tuning (SFT) is essential for training large language models (LLMs), significantly enhancing critical capabilities such as instruction-following and in-context learning. Nevertheless, creating suitable training datasets tailored for specific domains remains challenging due to unique domain constraints and data scarcity.

In this paper, we propose SearchInstruct, an innovative method explicitly designed to construct high-quality instruction datasets for SFT. Our approach begins with a limited set of domain-specific, human-generated questions, which are systematically expanded using a large language model. Subsequently, domainrelevant resources are dynamically retrieved to generate accurate and contextually appropriate answers for each augmented question.

Experimental evaluation demonstrates that SearchInstruct notably enhances both the diversity and quality of SFT datasets, leading to measurable improvements in LLM performance within specialized domains. Additionally, we demonstrate that beyond dataset generation, the proposed method can also effectively facilitate tasks such as model editing, enabling efficient updates to existing models.

To facilitate reproducibility and community adoption, we provide full implementation details, the complete set of generated instruction–response pairs, and the source code in a publicly accessible Git repository.1

### 1 Introduction

The rapid progress of LLMs and their success in solving general-purpose natural language processing (NLP) tasks (Wei et al., 2021; Sanh et al., 2022) have led to a growing focus on adapting these models to specialized domains. Achieving this goal

1https://github.com/mostafaamiri/ SearchInstruct

requires not only access to cleaned raw corpora, but also the construction of structured datasets suitable for the SFT stage (Ouyang et al., 2022).

Generating high-quality SFT data—especially for expert domains—presents several major challenges. The data must be both diverse and capable of reflecting the inherent complexity of realworld user queries. Existing approaches either rely on extensive manual annotation, as in InstructGPT (Ouyang et al., 2022), suffer from limited coverage and task diversity (Ziegler et al., 2024), or fail to simulate real user needs accurately. Furthermore, many methods depend solely on the internal knowledge of LLMs, which may be outdated or insufficient for domain-specific adaptation (Lewis et al., 2020; Lazaridou et al., 2022).

At the same time, LLMs have shown that—when provided with appropriate context—they can generate accurate and informative responses to even complex domain-specific questions (Izacard et al., 2022). This observation suggests that, by systematically generating diverse and realistic queries and retrieving relevant, up-to-date contextual information, it is possible to construct powerful training datasets without relying on human annotation. Such an approach enables the integration of external domain-specific knowledge into the training process, offering a path to updating and extending model capabilities (Guu et al., 2020; Wang et al., 2024).

In this paper, we introduce SearchInstruct, a novel framework for automatically generating supervised training data. The method operates by expanding a small set of seed questions using an LLM, then dynamically retrieving relevant documents to construct accurate, context-grounded answers. Our empirical results demonstrate that this approach not only increases the quality and diversity of SFT data, but also effectively enhances and updates LLM behavior in specialized domains by leveraging up-to-date external knowledge.

Our contributions are as follows:

- 1. We introduce SearchInstruct, an automated framework for constructing SFT datasets by combining targeted document retrieval with grounded answer generation;
- 2. We design and implement a four-stage pipeline for query expansion, diversification, document retrieval, and answer synthesis;
- 3. We empirically show that, in certain specialized domains, this pipeline leads to improvements in model performance and instructionfollowing accuracy;
- 4. We demonstrate the effectiveness of SearchInstruct in model editing scenarios, enabling efficient updates to model outputs using newly acquired knowledge.

### 2 Related Work

Instruction tuning has emerged as a key strategy for adapting LLMs to follow user instructions. The FLAN approach by Wei et al. (Wei et al., 2021) demonstrated that fine-tuning on a broad range of instruction-formatted tasks leads to strong zero-shot generalization. Similarly, T0 by Sanh et al. (Sanh et al., 2022) employed multitask prompted training to enable cross-task generalization.

Wang et al. (Wang et al., 2022b) extended this paradigm with the Super-NaturalInstructions dataset, collecting over 1,600 crowdsourced tasks paired with declarative instructions to promote task diversity. InstructGPT (Ouyang et al., 2022) introduced a method for aligning LLMs with human preferences using instruction–response pairs ranked via human feedback. Despite their utility, these datasets are static, expensive to construct, and often insufficient for domain-specific applications or ongoing model updates.

To address limitations in scalability and recency, Self-Instruct (Wang et al., 2022a) bootstraps a base LLM to generate new instructions and responses from a small seed set, filtering outputs using heuristics. While it enhances general instructionfollowing capabilities, it still relies entirely on the model’s internal knowledge, which may be outdated or incomplete for specialized domains.

Recent work has focused on automating instruction dataset creation to reduce dependence on manual annotation. The Alpaca project (Taori et al., 2023) used GPT-3.5 to generate 52,000 instruc-

tion–response pairs, enabling effective fine-tuning of a LLaMA model. Evol-Instruct by Xu et al. (Xu et al., 2023) introduced an evolutionary strategy to rewrite and complicate instructions, leading to the creation of WizardLM. InstructZero (Chen et al., 2023) proposed a label-free tuning method using reinforcement learning with black-box models. While these approaches improve scale and diversity, they generally lack grounding in real-world evidence and do not facilitate domain-specific adaptation.

Retrieval-augmented methods aim to improve factuality and relevance by incorporating external documents. RAG (Lewis et al., 2020) combines dense retrieval with generation to condition responses on retrieved passages. REALM (Guu et al., 2020) pre-trains language models with latent document retrieval to support open-domain QA. More recently, InstructRetro (Wang et al., 2024) integrates retrieval with instruction tuning to enhance grounding, and RA-DIT (Lin et al., 2024) jointly trains a retriever and generator using dual instruction signals. However, these methods often require costly retriever training and may not be ideal for rapid, low-resource domain updates.

SearchInstruct addresses the limitations of static and hallucinated instruction datasets by incorporating real-time document retrieval during data generation. Unlike Self-Instruct, which depends solely on internal knowledge, our method grounds both the instruction and the response in up-to-date, domain-specific documents. Unlike methods such as RAG and REALM, which apply retrieval at inference time, SearchInstruct applies it during dataset construction—allowing the fine-tuned model to internalize retrieved knowledge. This enables scalable, lightweight domain adaptation and continual knowledge refresh, bridging the gap between static SFT and real-world evolving needs.

### 3 Methodology

The core idea behind our method is that an LLM, when provided with relevant contextual information—even for domain-specific queries—can generate accurate and informative answers. These instruction–response pairs can then be used as highquality data for SFT, improving the model’s ability to follow instructions and generalize in specialized domains.

While prior work such as Self-Instruct (Wang et al., 2022a) demonstrated the value of gen-

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>humans seeds GPT Instructions<br><br>GPT<br><br>Web Search<br><br>RAG<br><br>API<br><br>[Figure 10]<br><br>Documents Response<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>Generating initial seeds<br><br>Types of retrieval methods<br><br>Document-based response construction<br><br>Increasing query variety<br><br>Retrieving documents related to the query|
|---|

- Figure 1: The four-stage SearchInstruct pipeline: seed generation, query expansion, document retrieval, and response construction.

erating large quantities of instruction–response examples, we argue that in the current landscape—where LLMs often outperform crowdannotated answers—the primary focus should shift from generating more answers to diversifying the questions themselves. A broader and more realistic query distribution enables better coverage of domain-specific tasks, especially when answers are grounded in retrieved, up-to-date evidence.

To operationalize this idea, we propose a fourstage pipeline, illustrated in Figure 1, consisting of the following steps:

- 1. Seed Generation: Start with a small set of domain-relevant, human-written or curated instruction prompts;
- 2. Query Expansion: Use an LLM to generate diverse variations and paraphrases of the seed prompts;
- 3. Document Retrieval: Retrieve domainspecific and up-to-date documents relevant to each expanded query (using web search, RAG, or external APIs);
- 4. Response Construction: Generate accurate, context-aware answers grounded in the retrieved evidence.

In the following sections, we describe each stage of the pipeline in detail. 3.1 SearchInstruct Dataset Formulation

We formally define our instruction dataset D as a set of instruction–response pairs constructed through a four-stage pipeline.

Let Q = {q1,q2,...,q|Q|} denote the initial set of domain-specific seed queries. From Q, we generate a set of additional instructions via LLM-based expansion. Let I denote the set of newly generated instructions, and define the final instruction pool as:

Itotal = Q ∪ I

For each instruction i ∈ Itotal, we retrieve a set of relevant textual contexts (e.g., documents, passages), denoted Ci = {ci,1,...,ci,k}. We then use an LLM to generate an answer Ai grounded in the retrieved context:

Ai = LLManswer(i,Cifiltered) The final instruction dataset is constructed as:

D = (i,Ai) i ∈ Itotal

During training, the model M learns the conditional instruction-following mapping:

M(i) → Ai

Note that while Ci is essential for answer generation during dataset construction, it is not required at inference or training time. This enables M to internalize external knowledge through SFT and improve response quality in dynamic or specialized domains.

#### 3.2 Stage 1: Seed Generation

The SearchInstruct pipeline begins with the creation of a diverse set of seed queries Q =

{q1,q2,...,q|Q|} that serve as the foundation for subsequent query expansion. Because all downstream instructions are generated based on these seeds, ensuring high quality and topical diversity in Q is critical.

We consider two approaches for seed construction:

Human-Crafted Seeds. In the fully manual setup, domain experts are given a detailed instruction-writing guideline that includes:

- • Requirements for seed diversity and realism,
- • Types of questions that are difficult to synthesize from retrieved documents (e.g., abstract reasoning, subjective judgment),
- • Subcategories within the target domain.

Each annotator is asked to write several domainspecific instructions based on this guideline. This ensures that Q contains high-coverage, realistic questions that may not emerge from automatic generation methods.

Human–LLM Collaborative Seeds. In the hybrid setup, annotators are guided to use powerful LLMs (e.g., GPT-4o, Claude 3.5 Sonnet, Gemini 2 Pro) to generate seeds. A separate guideline is provided, which instructs annotators to:

- 1. Ask the LLM to suggest a comprehensive list of subtopics within the target domain;
- 2. For each subtopic, prompt the LLM to generate multiple instruction-style queries based on predefined types;
- 3. Select, edit, or rewrite these outputs into final seeds.

This method enables rapid scaling while retaining human supervision and correction. In practice, we collected approximately twice as many seeds per annotator compared to the fully manual setting.

After initial training, we identify weaknesses in specific subdomains or instruction types. Additional seeds are then generated—using either method above—to target these weaknesses and enrich low-performing areas.

Appendix C provides a detailed example of how seed queries were constructed in the domain of Iranian culinary and tourism.

#### 3.3 Stage 2: Query Expansion

To increase the diversity and coverage of the instruction dataset, we expand the initial seed set Q using a LLM. This process is inspired by the

Self-Instruct framework (Wang et al., 2022a), but adapted to focus exclusively on instruction generation rather than full instruction–response pairs.

At each expansion step, a subset of n seed queries S = {q1,q2,...,qn} ⊂ Q is selected, and the LLM is prompted to synthesize k new instruction-style queries:

IS = {i1,i2,...,ik}

Each instruction ij ∈ IS is either a paraphrase of a single qi or a novel combination of multiple seed queries in S. This process is repeated over m iterations to generate a sufficiently diverse instruction pool.

We denote the union of all generated instructions across iterations as:

m

I =

ISt

t=1

The final instruction set used for downstream processing is defined as:

Itotal = Q ∪ I

Appendix A.1 shows the prompt template used to diversify queries during expansion.

#### 3.4 Stage 3: Document Retrieval

In this stage, we retrieve domain-relevant content to ground each instruction i ∈ Itotal in external evidence. The goal is to identify high-quality sources that can support accurate and informative answer generation in the next stage.

Depending on the application setting and available resources, we consider multiple retrieval strategies:

- • RAG-style retrieval: If access to a structured document collection is available, we apply dense or hybrid retrieval to identify the top-

k text chunks Ci = {ci,1,...,ci,k} that are semantically similar to the instruction i. This is done using retriever models such as DPR or OpenAI embeddings.

- • Web search: For open-domain instructions or in the absence of curated corpora, we use web search engines to collect a set of relevant URLs and associated snippets as external context.

• External APIs: In some cases, domainspecific APIs (e.g., tourism databases, legal knowledge graphs) are queried to extract structured or semi-structured evidence.

To improve the effectiveness of retrieval, we do not use the instruction i directly. Instead, we construct a search-oriented query qisearch from i using an LLM-based rewriting module:

qisearch = LLMrewriter(i)

This transformation helps reduce ambiguity and match the query to the structure and vocabulary of the underlying retrieval source.

Prompt template used for query rewriting is provided in Appendix A.2.

#### 3.5 Stage 4: Response Construction

In the final stage of the SearchInstruct pipeline, we synthesize high-quality answers for each instruction i ∈ Itotal using retrieved evidence. Each instruction is paired with its corresponding set of retrieved contexts Ci = {ci,1,...,ci,k} obtained from Stage 3.

To ensure efficient and accurate generation, we pre-process the retrieved documents to filter out irrelevant or noisy content. This step is especially important when dealing with long documents, as excessive input length can degrade LLM performance and increase computational cost. We apply one of the following filtering strategies:

- • LLM-based chunk filtering: A lightweight language model is used to rank or extract segments of each document that are most relevant to the instruction;
- • Rule-based filtering: Heuristics are applied to remove common noise sources such as HTML tags, ads, metadata, or user comments.

After filtering, we concatenate the instruction

i with the cleaned context Cifiltered and prompt a powerful LLM to generate a contextually grounded

answer:

##### Ai = LLManswer(i,Cifiltered)

This produces the final instruction–response pair (i,Ai) to be included in the training dataset D.

Prompt templates for this stage are provided in Appendix A.3.

### 4 Applications of SearchInstruct

The advantages of our proposed approach can be summarized in three key aspects:

- 1. It enables the generation of accurate and contextually grounded responses for questions that require domain-specific knowledge—particularly in cases where no existing LLM is capable of providing a correct answer.
- 2. The instruction–response pairs generated by this method provide a close approximation of real end-user queries.
- 3. The method remains effective even when using smaller or open-source LLMs. As long as relevant documents are retrieved and appropriately incorporated, it is possible to construct high-quality responses to complex queries without relying on large proprietary models.

In this work, we apply SearchInstruct in two primary settings:

- • Constructing SFT datasets in the domain of Iranian culture, with a focus on two subdomains: traditional cuisine and domestic tourism;
- • Updating model knowledge with up-to-date information, to enhance its ability to respond to recent or evolving queries in specialized domains.

In the following sections, we provide a detailed analysis of both application scenarios.

- 4.1 SFT Dataset Construction for Iranian Culture

Iran is a vast and culturally diverse country, home to a wide range of historical sites, regional customs, and intangible heritage. Many of these cultural and geographical details are either underrepresented or entirely absent in current LLMs. To support this claim, Appendix B presents examples of domainspecific queries for which existing models failed to provide correct answers, while the SearchInstruct framework successfully generated accurate responses.

Another challenge is that Persian speakers, particularly in informal contexts, often use colloquial and context-rich language that differs significantly from formal documents. Except for a few major companies that serve Persian-speaking users, there

Win

Tie

Lose

| |
|---|

| |
|---|

| |
|---|

Tourism Domain With vs. Without SearchInstruct

| | | |63%| | | |34|%| |
|---|---|---|---|---|---|---|---|---|---|
| | | |68%| | | | |30%| |
| | | | | | | | | | |

Matina 70B

Matina 8B

0 10 20 30 40 50 60 70 80 90 100

Outcome Share (%)

(a) Tourism domain

Win

Tie

Lose

| |
|---|

| |
|---|

| |
|---|

Culinary Domain With vs. Without SearchInstruct

| | |6|2%| | | |35%| | |
|---|---|---|---|---|---|---|---|---|---|
| | | |65%| | | |31|%| |
| | | | | | | | | | |

Matina 70B

Matina 8B

0 10 20 30 40 50 60 70 80 90 100

Outcome Share (%)

(b) Culinary domain

- Figure 2: Human evaluation results comparing baseline models with those fine-tuned using the SearchInstruct framework across two domains.

is limited access to large-scale, real-world Persian queries. Attempting to synthesize such instructions purely from documents using LLMs often fails to capture the diversity of real user intent and task formulations. For example, a typical user might describe their own travel constraints and ask for a multi-day itinerary for a specific region of Iran—a type of instruction that is rarely (if ever) found explicitly in documents or websites.

To evaluate the practical effectiveness of our method, we applied SearchInstruct in two cultural subdomains: traditional Iranian cuisine and domestic tourism. As discussed in the following sections, our results demonstrate the value of this approach in generating realistic, high-quality SFT data in resource-limited domains.

- 4.1.1 Improvements in Cuisine and Tourism In the MATINA project (Hosseinbeigi et al., 2025), multiple LoRA-tuned LLaMA-based models were trained across various domains, including traditional cuisine and tourism. The training data was generated using a mix of document-based question–answering and Evol-Instruct methods.

However, during human evaluation, it became evident that the models failed to perform adequately in several cultural subdomains. Upon analysis, these shortcomings were largely attributed to gaps in training data—particularly the absence of certain types of queries. These included questions such as:

- • “List multiple examples of...”
- • “Imagine the following scenario...”
- • “Recommend something based on personal constraints...”

Such instructions are rarely present in documents and are not well-covered by typical synthetic data pipelines.

To address these gaps, we curated a specialized seed set as described in Section 3.2 and Appendix C. The SearchInstruct pipeline was then applied to produce high-quality SFT data specifically targeting underrepresented question types and subdo-

mains. The number of generated instances for each category using this method is reported in Table 1. Each stage in the table corresponds to one full iteration of the SearchInstruct pipeline—including query expansion, document retrieval, and answer generation—which was repeated in response to human feedback or model performance analysis. In each stage, the pipeline was refined to specifically address the issues and gaps identified during the previous round, leading to incremental improvements in coverage, quality, and diversity.

Domain Stage 1 Stage 2 Stage 3 Overall Culinary 4273 2886 1773 8932 Tourism 3932 2378 1250 7560

Table 1: Number of SFT samples generated in culinary and tourism domains across three iterative stages. Each stage reflects one cycle of the SearchInstruct pipeline applied to address feedback from the previous round.

To evaluate the effectiveness of the proposed method, we conducted a blind human evaluation. Five independent annotators—none of whom were involved in seed construction—were asked to each design 20 diverse questions, resulting in a 100question benchmark. This benchmark was used to evaluate two versions of the MATINA model: one trained prior to SearchInstruct augmentation, and one trained with our additional SFT data. Annotators were blind to the model identities2.

As shown in Figure 2, models trained with SearchInstruct data performed significantly better on previously weak areas. These results validate the method’s utility in real-world settings and highlight its ability to fill content gaps that standard data generation techniques miss.

This setup also enables iterative refinement: once specific weaknesses in the model’s behavior

2Model identifiers were hidden from annotators during evaluation.

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

seeds

Question Author

seeds SFT data Create SFT data with

Generating initial seeds for feed backs

SearchInstruct

FeedBack

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Updated Model

Evaluator

Updated Model

SFT data

Evaluation Phase

SFT training phase

- Figure 3: Iterative refinement loop enabled by the SearchInstruct framework. After initial fine-tuning, specific model weaknesses are identified through targeted evaluation. New instruction–response data is then generated to directly address these shortcomings, creating a feedback loop that leads to focused and incremental improvements in model performance.

are identified through human evaluation or domainspecific testing, additional instruction–response pairs can be generated using the SearchInstruct framework to target those areas. As shown in Figure 3, this creates a continuous improvement loop in which domain-specific shortcomings are systematically addressed. The process can be repeated until performance stabilizes or marginal gains diminish, allowing for efficient and focused model enhancement over time.

#### 4.2 Model Updating via SearchInstruct

Another practical use case for the SearchInstruct framework is model updating—specifically, targeted modification of model outputs based on recent information. The underlying intuition is that by retrieving up-to-date documents relevant to a query, it is possible to revise the model’s response in a lightweight and localized manner.

To constrain the task for evaluation purposes, we focused on a narrow update scope: tracking recent political changes and selected current events. Our goal was to update only specific knowledge while preserving the rest of the model’s internal representations, effectively performing controlled model editing.

We applied this strategy to the Gemma 27B model. First, using a set of manually curated seed instructions (see Appendix C.3), we generated query–response pairs via SearchInstruct. For each instruction, we obtained the model’s original output and retrieved relevant, up-to-date docu-

ments. These documents included new facts—such as political appointments, resignations, or verified current affairs.

To incorporate the updated knowledge, we used a secondary model such as DeepSeek to minimally revise the original answer from Gemma. The editing model was guided to only modify incorrect or outdated information based on the retrieved documents, without altering unrelated content. The prompt system used for editing is described in Appendix A.4.

As a result, only selected spans within the model’s responses were modified, ensuring that the updates were localized, accurate, and minimally invasive. A schematic overview of the updatespecific data generation process is shown in Figure 4, and examples of before-and-after outputs are provided in Appendix D.

#### 4.2.1 Evaluation of the Edited Model

To evaluate the effectiveness of model updating via SearchInstruct, we constructed a preference dataset for fine-tuning using the ORPO (Hong et al., 2024). For each instruction i ∈ Iedit (instructions related to recent knowledge), we defined a pair:

##### (i,Arejecti ,Achoseni )

where Arejecti is the original output from the base model (e.g., Gemma), and Achoseni is the edited response generated by a secondary editing model (e.g., DeepSeek) based on up-to-date documents.

Chat bot

[Figure 28]

[Figure 29]

[Figure 30]

Instructions

Question Author

Generating instruction

Retriever updated Information

[Figure 31]

[Figure 32]

Updated documents

Web Search

[Figure 33]

[Figure 34]

[Figure 35]

Correct answer

Deepseek-v3

[Figure 36]

Edite answer with Update information

Gemma3 27B it

Wrong answer

Create answer from base model

- Figure 4: Pipeline for constructing update-specific instruction data used in model editing. Starting from userprovided queries, relevant documents are retrieved and used to construct grounded instruction–answer pairs.

Category Gemma3 27B (original) Gemma3 27B (updated)

STEM 66.04 62.86 (-3.18) Social Sciences 79.30 77.87 (-1.43) Humanities 60.60 59.02 (-1.58) Other 73.66 71.65 (-2.01) Average 68.88 66.89 (-1.99)

Table 2: Performance of Gemma3 27B before and after the update. Score decreases are highlighted in red.

Using this preference data, we fine-tuned the model with ORPO to align its outputs toward the corrected information. We then evaluated the updated model in two ways:

- • Targeted evaluation: A new set of instructions related to the same domain (e.g., political changes) was submitted to both the original and edited models. Human evaluation confirmed that updated responses were correct and aligned with recent information.
- • General evaluation: We also measured model performance on the MMLU benchmark (Hendrycks et al., 2021) to assess whether general knowledge was degraded. Results in Table 2 show no significant drop in accuracy, suggesting that the editing process preserved the broader knowledge of the model.

However, further analysis revealed a critical limitation. In some cases, the model was only capable of answering specific updated questions correctly (e.g., "Who is the current president of Iran?"), but lacked deeper knowledge about the subject. For instance, although the edited model correctly answered “Masoud Pezeshkian” as the current president, it failed to provide accurate information about his past roles or political background—facts not present in the updated dataset and not previously encoded in the model’s internal knowledge.

This indicates that the editing process effectively

replaces surface-level facts for specific queries but does not deeply integrate new knowledge into the model’s reasoning capabilities. While model editing is efficient and minimally invasive, its ability to introduce deeply connected knowledge remains limited.

### 5 Conclusion

In this work, we introduced SearchInstruct, a novel framework for constructing high-quality instruction datasets for SFT. The method combines LLMdriven instruction generation with targeted document retrieval to address challenges related to diversity, realism, and recency—especially in specialized domains.

We demonstrated the effectiveness of this approach in two key scenarios: (1) enhancing model performance in cultural domains such as Iranian cuisine and tourism, and (2) updating factual knowledge in large language models using minimally invasive editing strategies. Experimental results showed that SearchInstruct can generate contextually accurate and domain-aligned data, leading to measurable improvements without degrading existing knowledge.

Overall, SearchInstruct provides a flexible and scalable solution for data construction and model editing, making it a promising tool for the continued development of adaptive and domain-aware

language models.

### 6 Limitations

Despite its effectiveness, the SearchInstruct framework has several limitations:

- • Dependency on retrieved documents: The quality and accuracy of generated answers are directly tied to the reliability of retrieved content. In domains with sparse or noisy resources, this can limit performance.
- • Shallow model editing: Our findings suggest that editing responses through surface-level substitution does not lead to deep conceptual integration. The model may memorize specific updates without understanding broader context.
- • Limited scalability: Although semiautomated, applying the method across large or fast-changing domains still requires substantial resources for document retrieval, filtering, and validation.
- • Reliance on strong LLMs: Several stages, including query expansion and response construction, depend on high-quality LLMs. In low-resource settings, performance may degrade.
- • Risk of bias propagation: Web-based retrieval introduces potential biases from source documents, which can affect the neutrality and fairness of generated data.

### References

Lichang Chen, Jiuhai Chen, Tom Goldstein, Heng Huang, and Tianyi Zhou. 2023. Instructzero: Efficient instruction optimization for black-box large language models. arXiv preprint arXiv:2306.03082.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. Realm: Retrieval-augmented language model pre-training. In Proceedings of the 37th International Conference on Machine Learning (ICML 2020), pages 3929–3938. PMLR.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In Proceedings of the International Conference on Learning Representations (ICLR 2021). Also available as arXiv:2009.03300.

Jiwoo Hong, Noah Lee, and James Thorne. 2024. Orpo: Monolithic preference optimization without reference model. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 11170–11189, Miami, Florida, USA. Association for Computational Linguistics.

Sara Bourbour Hosseinbeigi, MohammadAli SeifKashani, Javad Seraj, Fatemeh Taherinezhad, Ali Nafisi, Fatemeh Nadi, Iman Barati, Hosein Hasani, Mostafa Amiri, and Mostafa Masoudi. 2025. Matina: A culturally-aligned Persian language model using multiple LoRA experts. In Findings of the Association for Computational Linguistics: ACL 2025, pages 20874–20889, Vienna, Austria. Association for Computational Linguistics.

Gautier Izacard, Patrick S. H. Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2022. Atlas: Few-shot learning with retrieval augmented language models. arXiv preprint arXiv:2208.03299.

Angeliki Lazaridou, Elena Gribovskaya, Wojciech Stokowiec, and Nikolai Grigorev. 2022. Internet-augmented language models through search. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8460–8478, Dublin, Ireland. Association for Computational Linguistics.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020), pages 9459–9474. Curran Associates, Inc.

Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, and Scott Wen-tau Yih. 2024. Ra-dit: Retrieval-augmented dual instruction tuning. In Proceedings of the Eighth International Conference on Learning Representations (ICLR 2024). OpenReview.net / ICLR. Originally published as arXiv:2310.01352 (Oct2,2023).

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155.

Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M. Saiful Bari, Canwen Xu, Urmish

Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, and 22 others. 2022. Multitask prompted training enables zero-shot task generalization. In The Tenth International Conference on Learning Representations (ICLR), Virtual Event.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Xinyang Geng, Deepak Narayanan, Percy Liang, and Tatsunori B. Zhang. 2023. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford_alpaca.

Boxin Wang, Wei Ping, Lawrence McAfee, Peng Xu, Bo Li, Mohammad Shoeybi, and Bryan Catanzaro. 2024. Instructretro: Instruction tuning post retrieval-augmented pretraining. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024), pages 51255–51272. PMLR. Originally available as arXiv:2310.07713 (Oct 11, 2023).

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022a. Self-instruct: Aligning language models with self-generated instructions. arXiv preprint arXiv:2212.10560.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, and 19 others. 2022b. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5085– 5109, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2021. Finetuned language models are zero-shot learners. In Advances in Neural Information Processing Systems (NeurIPS).

Canwen Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, and Daxin Jiang. 2023. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244.

Zachary Ziegler, Ethan Perez, Tiago Pimentel, Emily Reif, Tianyi Zhang, Christopher Manning, and Colin Raffel. 2024. Craft: Conceptual retrieval-augmented fine-tuning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024). Association for Computational Linguistics.

### Appendix A System Prompt Designs

Each stage of our pipeline uses a dedicated system prompt tailored to the specific objective of that component. This appendix presents the design of each prompt through structured illustrations.

- A.1 System Prompt for Query Expansion

The system prompt used in this stage instructs the model to generate diverse and semantically rich variations of an initial instruction seed. This system prompt is illustrated in Figure 5

- A.2 System Prompt for Search-Oriented Rewriting

This prompt transforms general natural language instructions into search-optimized queries that enhance document retrieval performance.

|Search-Oriented Rewriting System Prompt|
|---|
|Given an instruction, respond with a search term query which can be efficiently used to find the answer in search engines. Do not include any explanations.<br><br>|

- Figure 6: System prompt design for search-oriented query rewriting.

A.3 System Prompt for Response Construction This prompt guides the model to synthesize a grounded and context-aware response using retrieved documents as input.

|Response Construction System Prompt|
|---|
|Provide accurate and well-explained answers to the user's questions, based on the content provided below. Ensure your response is as complete and comprehensive as possible.<br><br>Do not include information or assumptions outside of this content.|

- Figure 7: System prompt design for evidence-grounded response construction.

- A.4 System Prompt for Answer Updating

The final system prompt is designed to apply local edits to existing answers when new or corrected information is introduced.

### Appendix B Failure Cases of Current LLM Models

As discussed in the main body of the paper, current language models often fail to provide accurate

answers to questions in specialized domains due to insufficient domain knowledge. However, when provided with relevant supporting documents, their performance improves significantly. This observation reflects one of the core ideas behind our proposed method, SearchInstruct.

To demonstrate the effectiveness of our approach in situations where the base model fails to produce a reliable response, we present a set of representative examples in Table 3. For each case, we include the model’s original response as well as the revised answer generated using the SearchInstruct pipeline. These examples highlight how retrievalaugmented prompting can compensate for gaps in model knowledge and substantially improve response quality in specialized domains.

### Appendix C Seeds Generation

To generate high-quality SFT data, we begin with a curated set of instruction seeds Q designed to ensure realism, topic diversity, and domain coverage. We employ two complementary strategies for constructing these seeds:

(1) Human-Crafted Seeds. In the fully manual setup, domain experts were given a detailed instruction-writing guideline that included:

- • Requirements for seed diversity and contextual realism;
- • Emphasis on question types that are difficult to synthesize from retrieved documents (e.g., abstract reasoning, subjective judgment);
- • A comprehensive taxonomy of subcategories within each domain.

Each annotator was tasked with writing several domain-specific instructions that reflect real-world user needs and domain-specific reasoning challenges. This ensures that Q contains high-coverage, high-quality questions that may not emerge from automatic generation methods.

(2) Human–LLM Collaborative Seeds. In the hybrid setup, annotators used powerful LLMs (e.g., GPT-4o, Claude 3.5 Sonnet, Gemini 2 Pro) to assist with ideation and rewriting. Annotators were instructed to:

1. Ask the LLM to suggest a list of subtopics within the domain;

|Query Expansion System Prompt|
|---|
|You are an expert in question generation. You will be provided with sample questions. Your task is to create exactly **{args.number_created_questions}** new questions inspired by the sample questions. Each question should retain the structure and intent of the original sample questions but include variations by modifying<br><br>specific details. The generated questions must:<br><br>1. Be realistic, logical, and coherent.<br><br>2. Showcase creativity while maintaining a challenging nature.<br><br>3. Avoid being overly simple, nonsensical, or repetitive.<br><br><br>Provide the output in JSON format with the following structure: {{<br><br>"questions": [<br><br>"Generated question 1",<br><br>"Generated question 2",<br><br><br>... ]<br><br>}} Do not include any additional explanations, comments, or extra content. Only return the JSON output as specified.<br><br>|

Figure 5: System prompt design for query expansion.

- 2. For each subtopic, prompt the LLM to generate instruction-style questions based on predefined types;
- 3. Select, edit, or rewrite the outputs into final seeds that match our quality criteria.

This collaborative setup enabled faster idea generation, controlled diversity, and linguistic variation, while maintaining human oversight to ensure instructional clarity and domain relevance.

In the following sections, we describe the seed construction process for each target domain in detail.

#### C.1 Culinary Domain

To construct seeds for the culinary domain, we followed a dual strategy: fully human-written instructions and human–LLM collaborative generation. Our objective was to capture real-world cooking challenges that require reasoning beyond standard recipes, while ensuring topical diversity and realistic phrasing.

#### C.1.1 Human-Crafted Seeds

In the manual setup, domain experts wrote a wide range of instruction-style questions grounded in common cooking situations. Rather than relying on simple recipe queries, the focus was placed on mistakes, decision-making, ingredient constraints,

and reasoning-oriented prompts. Example questions include:

- • “My Fesenjan stew turned out too sour. How can I fix the taste?”
- • “How can I grill kebab without using a charcoal grill?”

To encourage topic diversity, annotators were provided with a guide listing more than 30 thematic categories. These included examples such as food preservation, regional dishes, dietary limitations, scientific explanations of cooking phenomena, ingredient reuse, children-oriented meals, and presentation aesthetics. These categories served purely as inspiration, not constraints, and annotators were encouraged to propose novel or cross-cutting question types that go beyond the listed themes.

#### C.1.2 Human–LLM Collaborative Seeds

In the collaborative setup, annotators were encouraged to interact with large language models (e.g., GPT-4o, Claude 3.5) to co-develop seeds. This included using the model to generate candidate questions, rewrite drafts in different tones, or suggest prompts within a topic area.

Common interaction patterns included:

• “Suggest a question about common mistakes in cooking rice.”

|Answer Updating System Prompt|
|---|
|You will receive two string variables:<br><br>• CURRENT_ANSWER: {current_answer} – the existing answer text<br><br>• NEW_CONTEXT: {new_context} – newly retrieved context that may confirm, update, or extend the original answer ## Instructions (Minimal-Change Policy) Your job is to update CURRENT_ANSWER using NEW_CONTEXT with the **least possible modification**. Follow these rules:<br><br><br>1. **Preserve the wording, tone, and sentence structure** of CURRENT_ANSWER unless NEW_CONTEXT explicitly requires a change.<br><br>2. If only names, dates, numbers, or similar facts change, **replace those elements only**—leave the rest untouched.<br><br>3. If NEW_CONTEXT introduces new, non-conflicting information, **append** it naturally using the same writing style.<br><br>4. If a statement in CURRENT_ANSWER is contradicted, **overwrite only the smallest necessary part** to reflect the updated fact. Always favor information from NEW_CONTEXT.<br><br>5. Do **not** rewrite, paraphrase, or restructure unless absolutely necessary for grammatical correctness after an edit. ### Temporal Reference When interpreting date expressions (e.g., "this year"), assume the current year is **2025 CE / 1404 SH**. ## Output<br><br><br>Return a **single plain-text string** with the fully updated answer. Do **not** include any metadata, brackets, or explanations—only the final revised answer.<br><br>|

Figure 8: System prompt design for answer refinement and updates.

• “Rewrite this to be more natural: Why is the center of my cake undercooked?”

Human reviewers selected and refined the outputs to ensure clarity, relevance, and natural phrasing. This collaborative process enhanced topical coverage and linguistic diversity, while accelerating seed creation.

#### C.2 Tourism Domain

To construct seeds in the tourism domain, we aimed to capture real-world questions that reflect the complex needs, constraints, and reasoning challenges of travelers within Iran. The objective was to go beyond basic location lookups or generic suggestions and instead focus on experiential, cultural, seasonal, and logistical aspects of domestic tourism.

We adopted a dual strategy: (1) fully humanauthored seeds and (2) human–LLM collaborative construction.

- C.2.1 Human-Crafted Seeds In the manual setup, expert annotators were instructed to write instruction-style questions that simulate authentic user concerns in diverse travel scenarios. These included comparisons between destinations, cultural experiences, regional infrastructure challenges, weather-related planning, and cost-aware decisions.

To ensure coverage, annotators were given a guide listing more than 30 example cate-

gories. These included: historical landmarks, local cuisines, nature tourism, rural and nomadic travel, eco-tourism, handicrafts and souvenirs, accommodation types, seasonal destinations, transportation, adventure activities, underexplored sites, wellness tourism, and even environmental concerns such as climate impact on desert and coastal areas.

These categories were intended for inspiration only, not as limitations. Annotators were encouraged to mix themes (e.g., cultural + seasonal + budget-focused) and explore niche or unconventional travel scenarios.

Representative examples of well-formed questions include:

- • “If I want to travel in winter to a warm region with eco-lodges and low tourist density, where should I go?”
- • “What makes staying in a traditional Yazdi house different from a modern hotel, and which one would be better for a three-day cultural trip?”
- • “During my visit to Kandovan, the cave-like architecture stood out. How does it compare to stepped villages like Masuleh in terms of tourism experience?”
- • “Why is Hormuz Island, although less famous than Qeshm or Kish, often described as more unique and immersive?”

|index Language Instruction<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]| | |gpt4o|serachInstruct (gpt4o-mini)|
|---|---|---|---|---|
|One|Persian<br><br>ﯽﺠﻨﮐﻪﻠﮐيﻪﯿﻬﺗزﺮﻃ<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]| |ﻪﯿﻬﺗ زﺮﻃ ﻪﻣادا رد .دﻮﺷ ﯽﻣ ﻪﯿﻬﺗ درآ و وﺎﮔ ﺎﯾ ﺪﻨﻔﺳﻮﮔ ﺮﺳ ناﻮﺨﺘﺳا و ﺖﺷﻮﮔ زا و دراد قﺮﻓﻪﭼﺎﭘﻪﻠﮐ ﺎﺑ ﺮﺘﺸﯿﺑ ﻪﮐ ﺖﺳا نﺎﻣﺮﮐ نﺎﺘﺳا ﯽﺘﻨﺳ يﺎﻫاﺬﻏ زا ﯽﮑﯾ ﯽﺠﻨُﮐﻪﻠﮐ :ﻢﻫدﯽﻣ ﺢﯿﺿﻮﺗ ﺖﯾاﺮﺑ ار ﯽﺠﻨﮐ ﻪﻠﮐ ﯽﺘﻨﺳ<br><br>:مزﻻ داﻮﻣ<br><br>(...و ﻪﻧﻮﮔ ،ترﻮﺻ ﺖﺷﻮﮔ ،نﺎﺑز ،ﺰﻐﻣ ﻞﻣﺎﺷ) دﺪﻋ 1 :وﺎﮔ ﺎﯾ ﺪﻨﻔﺳﻮﮔ ﻪﻠﮐ • ناﻮﯿﻟ 2 دوﺪﺣ :راد سﻮﺒﺳ مﺪﻨﮔ درآ • گرﺰﺑ دﺪﻋ 2 :زﺎﯿﭘ •<br><br>|ﻪﮐ ﯽﯾﻻﺎﺑ يژﺮﻧا و مﺮﮔ ﻊﺒﻃ ﻞﯿﻟد ﻪﺑ ﯽﻨﯾﺮﯿﺷ ﻦﯾا. دﻮﺷﯽﻣ ﻪﯿﻬﺗ ﺎﻣﺮﺧ و ﺪﺠﻨﮐ ﺐﯿﮐﺮﺗ زا ﻪﮐ ﺖﺳا مﻼﯾا نﺎﺘﺳا يﻮﻘﻣ و ﯽﺘﻨﺳ يﺎﻫ ﯽﻨﯾﺮﯿﺷ زا ﯽﮑﯾ (ﻪﮕﻤﮔ ﺎﯾ) ﯽﺠﻨﮐ ﻪﻠﮐ :ماهدروآ نﺎﺘﯾاﺮﺑ ار هﺰﻤﺷﻮﺧ ﯽﻨﯾﺮﯿﺷ ﻦﯾا ﻪﯿﻬﺗ زﺮﻃ ،ﻪﻣادا رد. ﺖﺳا بﻮﺒﺤﻣ زورﻮﻧ ﺪﯿﻋ و اﺪﻠﯾ ﺐﺷ هﮋﯾوﻪﺑ لﺎﺳ دﺮﺳ لﻮﺼﻓ رد ،دراد<br><br>:مزﻻ داﻮﻣ<br><br>مﺮﮔ 500 :ﺪﺠﻨﮐ • مﺮﮔﻮﻠﯿﮐ 1 :(يز ﺎﯾ يﺪﻫاز) ﮏﺸﺧ يﺎﻣﺮﺧ •<br><br>|
| |Translatedinto<br><br>English<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]| |Kaleh Konji is a traditional dish from Kerman Province in Iran. It differs from regular kaleh pacheh and is made from the meat and bones of a sheep’s or cow’s head, along with whole wheat flour. Below is the traditional method of preparing Kaleh Konji:<br><br>Ingredients:<br><br>• Sheep or cow head: 1 (including brain, tongue, cheek meat, etc.)<br><br>• Whole wheat flour: about 2 cups<br><br>• Onion: 2 large<br>|Kaleh Konji (or Gomgeh) is a traditional and nutritious sweet from Ilam Province in Iran. Made from sesame seeds and dates, it’s especially popular during cold seasons—particularly on Yalda Night and Nowruz—due to its warming nature and high energy content. Below is the method for making this delicious sweet:<br><br>Ingredients:<br><br>• Sesame seeds: 500 grams<br><br>• Dried dates (Zahedi or similar variety): 1 kilogram<br><br><br>|
|Tow<br><br>|[Figure 205]<br><br>[Figure 206]<br><br>Persian<br><br>ﻪﭼمرادﻮﺑگﺮﺑومادﺎﺑلﻼﺧ،غﺮﻣﻢﺨﺗ،ﺰﻣﺮﻗﺖﺷﻮﮔﻦﻣ<br><br>شﺎﻫﺎﺑﯽﻧاﺮﯾاياﺬﻏ؟ﻢﻨﮐﺖﺳرد<br><br>[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]<br><br>[Figure 211]<br><br>[Figure 212]<br><br>[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>[Figure 218]<br><br>[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]<br><br>[Figure 222]<br><br>[Figure 223]<br><br>[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]<br><br>[Figure 228]<br><br>[Figure 229]<br><br>[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]<br><br>[Figure 238]<br><br>[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]<br><br>[Figure 242]<br><br>[Figure 243]<br><br>[Figure 244]<br><br>[Figure 245]<br><br>[Figure 246]<br><br>[Figure 247]<br><br>[Figure 248]<br><br>[Figure 249]<br><br>[Figure 250]<br><br>[Figure 251]<br><br>[Figure 252]<br><br>[Figure 253]<br><br>[Figure 254]<br><br>[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]<br><br>[Figure 258]<br><br>[Figure 259]<br><br>[Figure 260]<br><br>[Figure 261]<br><br>[Figure 262]<br><br>[Figure 263]<br><br>[Figure 264]<br><br>[Figure 265]<br><br>[Figure 266]<br><br>[Figure 267]<br><br>[Figure 268]<br><br>[Figure 269]<br><br>[Figure 270]<br><br>[Figure 271]<br><br>[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]<br><br>[Figure 276]<br><br>[Figure 277]<br><br>[Figure 278]<br><br>[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]<br><br>[Figure 284]<br><br>[Figure 285]<br><br>[Figure 286]<br><br>[Figure 287]<br><br>[Figure 288]<br><br>[Figure 289]<br><br>[Figure 290]<br><br>[Figure 291]<br><br>[Figure 292]<br><br>[Figure 293]<br><br>[Figure 294]<br><br>[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>[Figure 298]<br><br>[Figure 299]<br><br>[Figure 300]<br><br>[Figure 301]<br><br>[Figure 302]<br><br>[Figure 303]<br><br>[Figure 304]<br><br>[Figure 305]<br><br>[Figure 306]<br><br>[Figure 307]<br><br>[Figure 308]<br><br>[Figure 309]<br><br>[Figure 310]<br><br>[Figure 311]<br><br>[Figure 312]<br><br>[Figure 313]<br><br>[Figure 314]<br><br>[Figure 315]<br><br>[Figure 316]<br><br>[Figure 317]<br><br>[Figure 318]<br><br>[Figure 319]<br><br>[Figure 320]<br><br>[Figure 321]| |اﺬﻏ ﻦﯾا .ﺪﯿﻨﮐ ﺖﺳرد ﯽﻫﺎﺸﻧﺎﻣﺮﮐ مادﺎﺑ لﻼﺧ شرﻮﺧ مﺎﻧ ﻪﺑ ﯽﻧاﺮﯾا هﺰﻤﺷﻮﺧ ﯽﺘﻨﺳ ياﺬﻏ ﮏﯾ ﺪﯿﻧﻮﺗ ﯽﻣ ،(ﻮﺑ گﺮﺑ ،مادﺎﺑ لﻼﺧ ،غﺮﻣ ﻢﺨﺗ،ﺰﻣﺮﻗ ﺖﺷﻮﮔ) ﺪﯾراد ﻪﮐ يداﻮﻣ ﺎﺑ .هﺮﻄﻌﻣ و ﯽﺴﻠﺠﻣ رﺎﯿﺴﺑ و ﺖﺴﻫ (هﺎﺸﻧﺎﻣﺮﮐ هﮋﯾوﻪﺑ) ناﺮﯾا بﺮﻏ ﻖﻃﺎﻨﻣ صﻮﺼﺨﻣ<br><br>:مزﻻ داﻮﻣ<br><br>مﺮﮔ 300 – (ﯽﺘﺷرﻮﺧ) ﺰﻣﺮﻗ ﺖﺷﻮﮔ • ﻪﻧﺎﻤﯿﭘ 2/ 1 – مادﺎﺑ لﻼﺧ •<br><br>گرﺰﺑ دﺪﻋ 1 – زﺎﯿﭘ • يرﻮﺧاﺬﻏ ﻖﺷﺎﻗ 2 – هدﺮﮐ مد ناﺮﻔﻋز •<br><br>(يرﺎﯿﺘﺧا) يرﻮﺧاﺬﻏ ﻖﺷﺎﻗ 1 – ﯽﮕﻧﺮﻓ ﻪﺟﻮﮔ بر • يرﻮﺧاﺬﻏ ﻖﺷﺎﻗ 1 – بﻼﮔ •<br><br>دﺪﻋ 2 ﺎﺗ 1 – ﻮﺑ گﺮﺑ • ( يرﺎﯿﺘﺧا ،ﻦﯿﺋﺰﺗ ياﺮﺑ ) هدرز ﻂﻘﻓ – غﺮﻣ ﻢﺨﺗ •<br><br>|نﺎﺘﺳا زا ﯽﺴﻠﺠﻣ و ﯽﺘﻨﺳ شرﻮﺧ ﻦﯾا .ﺪﯿﻨﮐ ﻪﯿﻬﺗ (ﯽﻠﯿﺑدرا مادﺎﺑ شرﻮﺧ) «ﻪﻤﯿﻗ قﺎﭽﯿﭘ» مﺎﻧ ﻪﺑ هﺰﻤﺷﻮﺧ ﯽﻧاﺮﯾا ياﺬﻏ ﮏﯾ ﺪﯿﻧاﻮﺗﯽﻣ ،ﺪﯾراد رﺎﯿﺘﺧا رد ﻪﮐ يداﻮﻣ ﻪﺑ ﻪﺟﻮﺗ ﺎﺑ .دﻮﺷﯽﻣ وﺮﺳ ﺲﻟﺎﺠﻣ و ﺎﻫﯽﻧﺎﻤﻬﻣ رد دﻮﺧ دﺮﻓ ﻪﺑ ﺮﺼﺤﻨﻣ ﻢﻌﻃ ﺎﺑ و ﺖﺳا ﻞﯿﺑدرا<br><br>ﺮﻔﻧ 4 ياﺮﺑ مزﻻ داﻮﻣ<br><br>مﺮﮔ 400 :ﯽﺘﺷرﻮﺧ يﺪﻨﻔﺳﻮﮔ ﺖﺷﻮﮔ •<br><br>دﺪﻋ 4 :ﻂﺳﻮﺘﻣ زﺎﯿﭘ • ﻪﻧﺎﻤﯿﭘ 2/1 :مادﺎﺑ لﻼﺧ •<br><br>دﺪﻋ 4 :غﺮﻣ ﻢﺨﺗ •<br><br>دﺪﻋ 2 :ﻮﺑ گﺮﺑ • مﺮﮔ 100 :ﯽﻠﺤﻣ هﺮﮐ •<br><br>يرﻮﺧاﺬﻏ ﻖﺷﺎﻗ 3 :ﻆﯿﻠﻏ هدﺮﮐ مد ناﺮﻔﻋز • يرﻮﺧاﺬﻏ ﻖﺷﺎﻗ 1 :ﻮﻤﯿﻠﺑآ ﺎﯾ هرﻮﻐﺑآ •<br><br>|
| |TranslatedintoEnglish<br><br>[Figure 322]<br><br>[Figure 323]<br><br>[Figure 324]<br><br>[Figure 325]<br><br>[Figure 326]<br><br>[Figure 327]<br><br>[Figure 328]<br><br>[Figure 329]<br><br>[Figure 330]<br><br>[Figure 331]<br><br>[Figure 332]<br><br>[Figure 333]<br><br>[Figure 334]<br><br>[Figure 335]<br><br>[Figure 336]<br><br>[Figure 337]<br><br>[Figure 338]<br><br>[Figure 339]<br><br>[Figure 340]<br><br>[Figure 341]<br><br>[Figure 342]<br><br>[Figure 343]<br><br>[Figure 344]<br><br>[Figure 345]<br><br>[Figure 346]<br><br>[Figure 347]<br><br>[Figure 348]<br><br>[Figure 349]<br><br>[Figure 350]<br><br>[Figure 351]<br><br>[Figure 352]<br><br>[Figure 353]<br><br>[Figure 354]<br><br>[Figure 355]<br><br>[Figure 356]<br><br>[Figure 357]<br><br>[Figure 358]<br><br>[Figure 359]<br><br>[Figure 360]<br><br>[Figure 361]<br><br>[Figure 362]<br><br>[Figure 363]<br><br>[Figure 364]<br><br>[Figure 365]<br><br>[Figure 366]<br><br>[Figure 367]<br><br>[Figure 368]<br><br>[Figure 369]<br><br>[Figure 370]<br><br>[Figure 371]<br><br>[Figure 372]<br><br>[Figure 373]<br><br>[Figure 374]<br><br>[Figure 375]<br><br>[Figure 376]<br><br>[Figure 377]<br><br>[Figure 378]<br><br>[Figure 379]<br><br>[Figure 380]<br><br>[Figure 381]<br><br>[Figure 382]<br><br>[Figure 383]<br><br>[Figure 384]<br><br>[Figure 385]<br><br>[Figure 386]<br><br>[Figure 387]<br><br>[Figure 388]<br><br>[Figure 389]<br><br>[Figure 390]<br><br>[Figure 391]<br><br>[Figure 392]<br><br>[Figure 393]<br><br>[Figure 394]<br><br>[Figure 395]<br><br>[Figure 396]<br><br>[Figure 397]<br><br>[Figure 398]<br><br>[Figure 399]<br><br>[Figure 400]<br><br>[Figure 401]<br><br>[Figure 402]<br><br>[Figure 403]<br><br>[Figure 404]<br><br>[Figure 405]<br><br>[Figure 406]<br><br>[Figure 407]<br><br>[Figure 408]<br><br>[Figure 409]<br><br>[Figure 410]<br><br>[Figure 411]<br><br>[Figure 412]<br><br>[Figure 413]<br><br>[Figure 414]<br><br>[Figure 415]<br><br>[Figure 416]<br><br>[Figure 417]<br><br>[Figure 418]<br><br>[Figure 419]<br><br>[Figure 420]<br><br>[Figure 421]<br><br>[Figure 422]<br><br>[Figure 423]| |With the ingredients you have (red meat, eggs, slivered almonds, bay leaves), you can make a delicious traditional Persian dish called Khoresht-e Khalal Badam Kermanshahi. This dish is native to western Iran (especially Kermanshah) and is considered very festive and aromatic.<br><br>Ingredients:<br><br>• Red meat (stewing cuts) – 300 grams<br><br>• Slivered almonds – 1/2 cup<br><br>• Onion – 1 large<br><br>• Brewed saffron – 2 tablespoons<br><br>• Tomato paste – 1 tablespoon (optional)<br><br>• Rose water – 1 tablespoon<br><br>• Bay leaves – 1 to 2 leaves<br><br>• Eggs – yolks only (for garnish, optional)<br>|Based on the ingredients you have, you can make a delicious Persian dish called "Pichagh Gheimeh" (Ardabili Almond Stew). This is a traditional and festive stew from Ardabil Province, known for its unique flavor and typically served at gatherings and special occasions.<br><br>Ingredients for 4 people:<br><br>• Lamb stew meat: 400 grams<br><br>• Medium onions: 4<br><br>• Slivered almonds: 1/2 cup<br><br>• Eggs: 4<br><br>• Bay leaves: 2<br><br>• Local butter: 100 grams<br><br>• Strong brewed saffron: 3 tablespoons<br><br>• Verjuice or lemon juice: 1 tablespoon<br>|
|Three<br><br>|[Figure 424]<br><br>[Figure 425]<br><br>Persian<br><br>ﺶﮐﺦﻧﻞﻘﻧﯽﻧوﺪﯿﻣ<br><br>؟سﺎﺠﮐتﺎﻏﻮﺳ<br><br>[Figure 426]<br><br>[Figure 427]<br><br>[Figure 428]<br><br>[Figure 429]<br><br>[Figure 430]<br><br>[Figure 431]<br><br>[Figure 432]<br><br>[Figure 433]<br><br>[Figure 434]<br><br>[Figure 435]<br><br>[Figure 436]<br><br>[Figure 437]<br><br>[Figure 438]<br><br>[Figure 439]<br><br>[Figure 440]<br><br>[Figure 441]<br><br>[Figure 442]<br><br>[Figure 443]<br><br>[Figure 444]<br><br>[Figure 445]<br><br>[Figure 446]<br><br>[Figure 447]<br><br>[Figure 448]<br><br>[Figure 449]<br><br>[Figure 450]<br><br>[Figure 451]<br><br>[Figure 452]<br><br>[Figure 453]<br><br>[Figure 454]<br><br>[Figure 455]<br><br>[Figure 456]<br><br>[Figure 457]<br><br>[Figure 458]<br><br>[Figure 459]<br><br>[Figure 460]<br><br>[Figure 461]<br><br>[Figure 462]<br><br>[Figure 463]<br><br>[Figure 464]<br><br>[Figure 465]<br><br>[Figure 466]<br><br>[Figure 467]<br><br>[Figure 468]<br><br>[Figure 469]<br><br>[Figure 470]<br><br>[Figure 471]<br><br>[Figure 472]<br><br>[Figure 473]<br><br>[Figure 474]<br><br>[Figure 475]<br><br>[Figure 476]<br><br>[Figure 477]<br><br>[Figure 478]<br><br>[Figure 479]<br><br>[Figure 480]<br><br>[Figure 481]<br><br>[Figure 482]<br><br>[Figure 483]| |. ﺖﺳا ﺰﯾﺮﺒﺗ ﺮﻬﺷ فوﺮﻌﻣ يﺎﻫ ﯽﺗﺎﻏﻮﺳ زا ﯽﮑﯾ "ﺶﮐﺦﻧ ﻞﻘﻧ" ،ﻪﻠﺑ<br><br>رد .دﻮﺷﯽﻣ ﺮﻄﻌﻣ بﻼﮔ ﺎﯾ ﻞﻫ ﺎﺑ ﯽﻫﺎﮔ و ﻦﯾﺮﯿﺷ ًﻻﻮﻤﻌﻣ نآ ﻢﻌﻃ .ﺖﺳا ﮏﭼﻮﮐ يﺎﻫﻞﻘﻧ زا ياهﺮﯿﺠﻧز ﺎﯾ ﺢﯿﺒﺴﺗ ﻪﺑ ﻪﯿﺒﺷ نﺎﺷﺮﻫﺎﻇ و ﺪﻧاﻞﺼﺘﻣ ﻢﻫ ﻪﺑ ﯽﯾﺎﻫﺦﻧ ﺎﺑ ﺎﻫﻞﻘﻧ ﻦﯾا .ﺪﺷﯽﻣ هدﺎﻔﺘﺳا ﺰﯾﺮﺒﺗ رد ﺎﻫﯽﺳوﺮﻋ و ﯽﺘﻨﺳ ﻢﺳاﺮﻣ رد ﺮﺘﺸﯿﺑ ﻪﺘﺷﺬﮔ<br><br>|،دﻮﺧ مﺮﻧ ﺖﻓﺎﺑ و ﻦﯾﺮﯿﺷ ﻢﻌﻃ ﺎﺑ ﯽﺘﻨﺳ ﯽﻨﯾﺮﯿﺷ ﻦﯾا .دﻮﺷ ﯽﻣ ﺪﯿﻟﻮﺗ ناﺪﻤﻫ و ﺮﯾﻼﻣ يﺎﻫﺮﻬﺷ رد هﮋﯾوﻪﺑ ﻪﮐ ﺖﺳا ناﺪﻤﻫ نﺎﺘﺳا فوﺮﻌﻣ يﺎﻫﯽﺗﺎﻏﻮﺳ زا ﯽﮑﯾ ﺶﮐ ﺦﻧ ﻞﻘﻧ ،ﻪﻠﺑ .دﻮﺷﯽﻣ ﻪﺘﺧﺎﻨﺷ ﻪﻘﻄﻨﻣ ﻦﯾا فوﺮﻌﻣ ﯽﺘﺳد ﻊﯾﺎﻨﺻ زا ﯽﮑﯾ ناﻮﻨﻋﻪﺑ و دراد ﯽﻧاواﺮﻓ ﺖﯿﺑﻮﺒﺤﻣ<br><br>|
| |Translatedinto<br><br>English|[Figure 484]<br><br>[Figure 485]<br><br>[Figure 486]<br><br>[Figure 487]<br><br>[Figure 488]<br><br>[Figure 489]<br><br>[Figure 490]<br><br>[Figure 491]<br><br>[Figure 492]<br><br>[Figure 493]<br><br>[Figure 494]<br><br>[Figure 495]<br><br>[Figure 496]<br><br>[Figure 497]<br><br>[Figure 498]<br><br>[Figure 499]<br><br>[Figure 500]<br><br>[Figure 501]<br><br>[Figure 502]<br><br>[Figure 503]<br><br>[Figure 504]<br><br>[Figure 505]<br><br>[Figure 506]<br><br>[Figure 507]<br><br>[Figure 508]<br><br>[Figure 509]<br><br>[Figure 510]<br><br>[Figure 511]<br><br>[Figure 512]<br><br>[Figure 513]<br><br>[Figure 514]<br><br>[Figure 515]<br><br>[Figure 516]<br><br>[Figure 517]<br><br>[Figure 518]<br><br>[Figure 519]<br><br>[Figure 520]<br><br>[Figure 521]<br><br>[Figure 522]<br><br>[Figure 523]<br><br>[Figure 524]<br><br>[Figure 525]<br><br>[Figure 526]<br><br>[Figure 527]<br><br>[Figure 528]<br><br>[Figure 529]<br><br>[Figure 530]<br><br>[Figure 531]<br><br>[Figure 532]<br><br>[Figure 533]<br><br>[Figure 534]<br><br>[Figure 535]<br><br>[Figure 536]<br><br>[Figure 537]<br><br>[Figure 538]<br><br>[Figure 539]<br><br>[Figure 540]<br><br>[Figure 541]<br><br>[Figure 542]|Yes, Nogh-e Nakhkesh (thread-pulled nougat) is one of the famous souvenirs of Tabriz.<br><br>These sweets are connected with threads and resemble a rosary or a chain of small sugar-coated pieces. Their taste is usually sweet and sometimes flavored with cardamom or rose water. In the past, they were commonly used in traditional ceremonies and weddings in Tabriz.<br><br>|Yes, Nogh-e Nakhkesh (thread-pulled nougat) is one of the well-known souvenirs of Hamadan Province, especially produced in the cities of Malayer and Hamadan. This traditional sweet is highly popular for its sweet flavor and soft texture, and it is recognized as one of the region’s famous handmade delicacies.<br><br>|
|Four|[Figure 543]<br><br>[Figure 544]<br><br>Persian<br><br>يﻮﺗﯽﮑﺳاﺖﺴﯿﭘ<br><br>ﻦﮐﯽﻓﺮﻌﻣﻢﻬﺑدﺰﯾ<br><br>[Figure 545]<br><br>[Figure 546]<br><br>[Figure 547]<br><br>[Figure 548]<br><br>[Figure 549]<br><br>[Figure 550]<br><br>[Figure 551]<br><br>[Figure 552]<br><br>[Figure 553]<br><br>[Figure 554]<br><br>[Figure 555]<br><br>[Figure 556]<br><br>[Figure 557]<br><br>[Figure 558]<br><br>[Figure 559]<br><br>[Figure 560]<br><br>[Figure 561]<br><br>[Figure 562]<br><br>[Figure 563]<br><br>[Figure 564]<br><br>[Figure 565]<br><br>[Figure 566]<br><br>[Figure 567]<br><br>[Figure 568]<br><br>[Figure 569]<br><br>[Figure 570]<br><br>[Figure 571]<br><br>[Figure 572]<br><br>[Figure 573]<br><br>[Figure 574]<br><br>[Figure 575]<br><br>[Figure 576]<br><br>[Figure 577]<br><br>[Figure 578]<br><br>[Figure 579]<br><br>[Figure 580]<br><br>[Figure 581]<br><br>[Figure 582]<br><br>[Figure 583]<br><br>[Figure 584]<br><br>[Figure 585]<br><br>[Figure 586]<br><br>[Figure 587]<br><br>[Figure 588]<br><br>[Figure 589]<br><br>[Figure 590]<br><br>[Figure 591]<br><br>[Figure 592]<br><br>[Figure 593]| |ﻪﮐ نﺎﺠﯾﺎﺑرذآ ﺎﯾ ناﺪﻤﻫ ،زﺮﺒﻟا ﺪﻨﻧﺎﻣ ناﺮﯾا ﯽﺑﺮﻏ ﺎﯾ ﯽﻟﺎﻤﺷ يﺎﻫ نﺎﺘﺳا فﻼﺧﺮﺑ .دراﺪﻧ ﯽﻤﺋاد ﺎﯾ ﯽﻌﯿﺒﻃ ﯽﮑﺳا ﺖﺴﯿﭘ ،ﮏﺸﺧ و مﺮﮔ ﻢﯿﻠﻗا و يﺮﯾﻮﮐ ياﻮﻫوبآ ﻞﯿﻟدﻪﺑ دﺰﯾ .دراﺪﻧ ار ﯽﻓﺮﺑ ﯽﮑﺳا ﺖﺴﯿﭘ دﺎﺠﯾا ياﺮﺑ مزﻻ ﯽﻤﯿﻠﻗا ﻂﯾاﺮﺷ دﺰﯾ ،ﺪﻨﺘﺴﻫ ﯽﻌﻗاو ﯽﮑﺳا يﺎﻫﺖﺴﯿﭘ ياراد<br><br>|ﻞﺼﻓ رد فﺮﺑ يﺎﻫ شرﺎﺑ ﻞﯿﻟد ﻪﺑ و دراد راﺮﻗ هﻮﮐﺮﯿﺷ هﻮﮐ ﯽﺑﻮﻨﺟ يﺎﻫﻪﻨﻣاد رد ﻪﻘﻄﻨﻣ ﻦﯾا .ﺖﺳا هﺪﺷ ﻊﻗاو ﺖﻔﺗ نﺎﺘﺳﺮﻬﺷ رد ﺪﯾﻮﺨﺳ ﯽﮑﺳا ﺖﺴﯿﭘ ، دﺰﯾ نﺎﺘﺳا رد دراﺪﻧﺎﺘﺳا ﯽﮑﺳا ﺖﺴﯿﭘ ﮏﯾ ناﻮﻨﻋﻪﺑ ﯽﻤﺳر رﻮﻃ ﻪﺑ نﺎﮑﻣ ﻦﯾا ﻪﮐ ﺖﺷاد ﻪﺟﻮﺗ ﺪﯾﺎﺑ ،لﺎﺣ ﻦﯾا ﺎﺑ .دﻮﺷﯽﻣ ﻞﯾﺪﺒﺗ ﯽﻧﺎﺘﺴﻣز تﺎﺤﯾﺮﻔﺗ ياﺮﺑ بﻮﺒﺤﻣ ﯽﻠﺤﻣ ﻪﺑ ،نﺎﺘﺴﻣز . ﺖﺳا ﺐﺳﺎﻨﻣ يراﻮﺳبﻮﯿﺗ ﺪﻨﻧﺎﻣ ﯽﻓﺮﺑ يﺎﻫيزﺎﺑ ياﺮﺑ ﺮﺘﺸﯿﺑ و دﻮﺷﯽﻤﻧ ﻪﺘﺧﺎﻨﺷ<br><br>|
| |Translated<br><br>intoEnglish<br><br>[Figure 594]<br><br>[Figure 595]<br><br>[Figure 596]<br><br>[Figure 597]<br><br>[Figure 598]<br><br>[Figure 599]<br><br>[Figure 600]<br><br>[Figure 601]<br><br>[Figure 602]<br><br>[Figure 603]<br><br>[Figure 604]<br><br>[Figure 605]<br><br>[Figure 606]<br><br>[Figure 607]<br><br>[Figure 608]<br><br>[Figure 609]<br><br>[Figure 610]<br><br>[Figure 611]<br><br>[Figure 612]<br><br>[Figure 613]<br><br>[Figure 614]<br><br>[Figure 615]<br><br>[Figure 616]<br><br>[Figure 617]<br><br>[Figure 618]<br><br>[Figure 619]<br><br>[Figure 620]<br><br>[Figure 621]<br><br>[Figure 622]<br><br>[Figure 623]<br><br>[Figure 624]| |Due to its desert climate and hot, arid conditions, Yazd does not have a natural or permanent ski resort. Unlike northern or western provinces of Iran such as Alborz, Hamedan, or Azerbaijan, which have actual ski resorts, Yazd lacks the necessary climatic conditions for establishing a snow-based ski resort.<br><br>|In Yazd Province, the Sakhvid ski slope is located in Taft County. This area lies on the southern slopes of Shirkooh Mountain and, due to snowfall in the winter season, has become a popular spot for winter recreation. However, it's important to note that this location is not officially recognized as a standard ski resort, and is more suitable for snow-related activities such as snow tubing.<br><br>|

[Figure 625]

[Figure 626]

ﯽﺠﻨﮐﻪﻠﮐيﻪﯿﻬﺗزﺮﻃ

Persian

One

RecipeforKalehKonji

Translatedinto

English

ﻪﭼمرادﻮﺑگﺮﺑومادﺎﺑلﻼﺧ،غﺮﻣﻢﺨﺗ،ﺰﻣﺮﻗﺖﺷﻮﮔﻦﻣ

شﺎﻫﺎﺑﯽﻧاﺮﯾاياﺬﻏ؟ﻢﻨﮐﺖﺳرد

Persian

Tow

meat,eggs,sliveredalmonds,and

s.WhatPersiandishcanImake

anslatedintoEnglish

withthem?

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

- Table 3: Examples of partial outputs from GPT-4o and SearchInstruct (GPT4o mini):This demonstrates that in certain instructions, the SearchInstruct method is capable of producing better responses than larger models, as it benefits from access to highly relevant contextual information, even when using a smaller model

These questions require the model to reason across multiple dimensions, including cultural insight, regional comparison, travel logistics, and user preference.

#### C.2.2 Human–LLM Collaborative Seeds

In the collaborative setup, annotators interacted with large language models (e.g., GPT-4o, Claude

- 3.5) to generate, refine, or extend questions. They could prompt the model with a general topic or theme and request multiple formulations, rewrites, or angle shifts.

Typical prompts used in the process included:

• “Generate questions about seasonal travel in northern Iran.”

• “Give me a few ideas on how to ask about nomadic tourism experiences.”

Annotators reviewed and adjusted the model’s output for realism, clarity, and tone. This approach supported exploration of lesser-covered subdomains, stylistic variation, and faster seed creation, especially in cases where human inspiration plateaued.

Together, these two methods enabled the creation of a rich, diverse, and high-quality set of instruction seeds for the tourism domain, covering both typical and edge-case user needs.

#### C.3 Model Updating Domain

Language models are often unable to respond accurately to questions involving real-world facts that have changed after the model’s training cut-off. These include updates to public figures, international events, institutional changes, or recent sports outcomes. In this section, we focus on designing instruction seeds that explicitly aim to identify and correct outdated model knowledge in such contexts.

The goal was to create seed questions that prompt the model to revise its internal responses, particularly in scenarios where factual updates are essential. These seeds support fine-tuning or model editing workflows and serve as a targeted evaluation tool for temporal generalization.

Seeds were manually constructed by identifying areas where the model is likely to return outdated or incorrect answers. Each question was designed to be direct, fact-seeking, and anchored in wellknown public events. Special care was taken to select topics that are newsworthy and important yet unlikely to be politically sensitive in the context of scientific publication.

Representative examples include:

- • “Who won the 2024 U.S. presidential election?”
- • “Who is currently the president of Iran in 1404

(2025)?”

- • “Has the war between Russia and Ukraine ended or is it still ongoing?”
- • “What was Iran’s medal ranking in the 2024 Summer Olympics?”

Such questions are simple in structure but powerful in purpose: they help surface blind spots in the model’s temporal awareness and provide clear entry points for knowledge injection. The resulting dataset serves both as an editing benchmark and as fine-tuning input for time-sensitive model correction.

Appendix D Examples from the Updated Dataset

This section presents examples from the final dataset constructed for model updating. These instances are designed to help the model revise outdated or incorrect knowledge in a localized and controlled manner, without disrupting its broader behavior.

Each entry consists of a time-sensitive question accompanied by two responses:

- • Rejected: An outdated, incorrect, or uncertain response—typically resembling what an unedited, pre-trained model might produce.
- • Chosen: A minimally edited and factually updated version of the rejected answer.

A key principle in constructing the chosen responses is that the edits are minimal and highly targeted. Instead of rewriting the entire answer, only the incorrect portion is replaced with updated information, while preserving the original structure, phrasing, and tone as much as possible. This design allows the model to learn localized corrections through fine-tuning or model editing, rather than general retraining.

Table 4 provides several representative samples. Each row shows a real-world question, the outdated (rejected) model response, and the minimally revised, accurate version (chosen). These examples span topics such as elections, leadership updates, international conflicts, and sports results.

### Appendix E – Training Hyperparameters

This appendix provides a consolidated overview of the most relevant training hyperparameters used across different stages of our experiments. We categorize the configurations into three groups:

- • SFT (Supervised Fine-Tuning) on domainspecific tasks using the Matina-8B and Matina-70B models, applied separately to culinary and tourism instruction datasets.
- • ORPO (Optimal Response Preference Optimization) training on the Gemma-3-27B-IT model using LoRA adapters and a custom ORPO objective.

Table 5 presents a unified summary of these hyperparameters, including key factors such as LoRA configuration, learning rate, dropout rate, batch size, training duration, and preference optimization parameters (where applicable)

### Appendix F: Future Work

While SearchInstruct demonstrates promising results, several directions remain open for future exploration:

|index Language Instruction<br><br>[Figure 634]<br><br>[Figure 635]<br><br>[Figure 636]<br><br>[Figure 637]<br><br>[Figure 638]<br><br>[Figure 639]<br><br>[Figure 640]<br><br>[Figure 641]<br><br>[Figure 642]<br><br>[Figure 643]<br><br>[Figure 644]<br><br>[Figure 645]<br><br>[Figure 646]<br><br>[Figure 647]<br><br>[Figure 648]<br><br>[Figure 649]<br><br>[Figure 650]<br><br>[Figure 651]<br><br>[Figure 652]<br><br>[Figure 653]<br><br>[Figure 654]<br><br>[Figure 655]<br><br>[Figure 656]<br><br>[Figure 657]<br><br>[Figure 658]<br><br>[Figure 659]<br><br>[Figure 660]<br><br>[Figure 661]| | |Gemma27b-it Response|Response Updated|
|---|---|---|---|---|
|One|Persian<br><br>زوﺮاشزروناﺮ<br><br>لﺎﺳرد1403<br><br>ﮐﯿ؟ﺖﺴ<br><br>[Figure 662]<br><br>[Figure 663]<br><br>[Figure 664]<br><br>[Figure 665]<br><br>[Figure 666]<br><br>[Figure 667]<br><br>[Figure 668]<br><br>[Figure 669]<br><br>[Figure 670]<br><br>[Figure 671]<br><br>[Figure 672]<br><br>[Figure 673]<br><br>[Figure 674]<br><br>[Figure 675]<br><br>[Figure 676]<br><br>[Figure 677]<br><br>[Figure 678]<br><br>[Figure 679]<br><br>[Figure 680]<br><br>[Figure 681]<br><br>[Figure 682]<br><br>[Figure 683]<br><br>[Figure 684]<br><br>[Figure 685]<br><br>[Figure 686]<br><br>[Figure 687]<br><br>[Figure 688]<br><br>[Figure 689]<br><br>[Figure 690]<br><br>[Figure 691]<br><br>[Figure 692]<br><br>[Figure 693]<br><br>[Figure 694]<br><br>[Figure 695]<br><br>[Figure 696]<br><br>[Figure 697]| |ﺮﯾ زو ناﻮﻨﻋ ﻪﺑ ﻢﻫدﺰﯿ ﺳ ﺖﻟود رد و ﺪﻧراد هﺪﻬﻋ ﺮﺑ ار ﺖﻤﺳ ﻦﯾ ا 1401 دادﺮﻣ زا نﺎﺸﯾ ا .ﺖﺳا ناﺮﯾ ا نﺎﻧاﻮﺟ و شزرو ﺮﯾ زو ﯽ ﻤﺷﺎﻫ ثﺮﻣﻮﯿﮐ ،(1403 دادﺮﺧ) ﺮﺿﺎﺣ لﺎﺣ رد .ﺪﻨﺘﺴﻫ رﺎﮐ ﻪﺑ لﻮﻐﺸﻣ شزرو<br><br>|ﺮﯾ زو ناﻮﻨﻋ ﻪﺑ ﻢﻫدﺰﯿ ﺳ ﺖﻟود رد و ﺪﻧراد هﺪﻬﻋ ﺮﺑ ار ﺖﻤﺳ ﻦﯾ ا 1403 دادﺮﻣ 31 زا نﺎﺸﯾ ا .ﺖﺳا ناﺮﯾ ا نﺎﻧاﻮﺟ و شزرو ﺮﯾ زو ﯽﻟﺎﻣﺎﯿﻧد ﺪﻤﺣا ،( 1403 رﻮﯾﺮﻬﺷ) ﺮﺿﺎﺣ لﺎﺣ رد . دﻮﺑ ﺖ ﻤﺳ ﻦﯾا راد هﺪﻬﻋ 1403 دادﺮﻣ 31 ﺎﺗ 1402 رﻮﯾ ﺮﻬﺷ 27 زا ﯽ ﻤﺷﺎﻫ ثﺮﻣﻮﯿ ﮐ ،نﺎﺸﯾ ا زا ﺶﯿﭘ .ﺪﻨﺘﺴﻫ رﺎﮐ ﻪﺑ لﻮﻐﺸﻣ شزرو<br><br>|
| |Translated<br><br>into<br><br>English<br><br>[Figure 698]<br><br>[Figure 699]<br><br>[Figure 700]<br><br>[Figure 701]<br><br>[Figure 702]<br><br>[Figure 703]<br><br>[Figure 704]<br><br>[Figure 705]<br><br>[Figure 706]<br><br>[Figure 707]<br><br>[Figure 708]<br><br>[Figure 709]<br><br>[Figure 710]<br><br>[Figure 711]<br><br>[Figure 712]<br><br>[Figure 713]<br><br>[Figure 714]<br><br>[Figure 715]<br><br>[Figure 716]<br><br>[Figure 717]<br><br>[Figure 718]<br><br>[Figure 719]<br><br>[Figure 720]<br><br>[Figure 721]<br><br>[Figure 722]<br><br>[Figure 723]<br><br>[Figure 724]<br><br>[Figure 725]<br><br>[Figure 726]<br><br>[Figure 727]<br><br>[Figure 728]<br><br>[Figure 729]|1404(20242025) |At present (June 2024), Kioumars Hashemi is the Minister of Sports and Youth Affairs of Iran. He has held this position since August 2022 and is serving as the Minister of Sports in the thirteenth administration.<br><br>|At present (August–September 2024), Ahmad Donyamali is the Minister of Sports and Youth Affairs of Iran. He has held this position since August 21, 2024, and is serving as the Minister of Sports in the thirteenth administration. Before him, Kioumars Hashemi held this position from September 18, 2023, to August 21, 2024.<br><br>|
|Tow|[Figure 730]<br><br>Persian<br><br>هﺪﺤﺘﻣتﻻﺎﯾارﻮﻬﻤﺟﺲﯿﯾر<br><br>؟ﻪﯿﮐ<br><br>[Figure 731]<br><br>[Figure 732]<br><br>[Figure 733]<br><br>[Figure 734]<br><br>[Figure 735]<br><br>[Figure 736]<br><br>[Figure 737]<br><br>[Figure 738]<br><br>[Figure 739]<br><br>[Figure 740]<br><br>[Figure 741]<br><br>[Figure 742]<br><br>[Figure 743]<br><br>[Figure 744]<br><br>[Figure 745]<br><br>[Figure 746]<br><br>[Figure 747]<br><br>[Figure 748]<br><br>[Figure 749]<br><br>[Figure 750]<br><br>[Figure 751]<br><br>[Figure 752]<br><br>[Figure 753]<br><br>[Figure 754]<br><br>[Figure 755]<br><br>[Figure 756]<br><br>[Figure 757]<br><br>[Figure 758]<br><br>[Figure 759]<br><br>[Figure 760]<br><br>[Figure 761]<br><br>[Figure 762]<br><br>[Figure 763]<br><br>[Figure 764]<br><br>[Figure 765]<br><br>[Figure 766]<br><br>[Figure 767]<br><br>[Figure 768]<br><br>[Figure 769]<br><br>[Figure 770]<br><br>[Figure 771]<br><br>[Figure 772]<br><br>[Figure 773]<br><br>[Figure 774]<br><br>[Figure 775]<br><br>[Figure 776]<br><br>[Figure 777]<br><br>[Figure 778]<br><br>[Figure 779]<br><br>[Figure 780]<br><br>[Figure 781]<br><br>[Figure 782]<br><br>[Figure 783]<br><br>[Figure 784]<br><br>[Figure 785]| |. دراد هﺪﻬﻋ ﺮﺑ ار ﺖﻤﺳ ﻦﯾا 2021 ﻪﯾﻮﻧاژ 20 زا وا .ﺖﺳا نﺪﯾﺎﺑ ﻮﺟ ﺎﮑﯾﺮﻣآ هﺪﺤﺘﻣ تﻻﺎﯾا ﯽﻠﻌﻓ رﻮﻬﻤﺟ ﺲﯿﺋر<br><br>:ﺪﯿﻨﮐ ﻪﻌﺟاﺮﻣ ﺮﯾز ﻊﺑﺎﻨﻣ ﻪﺑ ﺪﯿﻧاﻮﺗﯽﻣ ﺮﺘﺸﯿﺑ تﺎﻋﻼﻃا ياﺮﺑ<br><br>: https://www.whitehouse.gov/ﺪﯿﻔﺳ خﺎﮐ ﯽﻤﺳر ﺖﯾﺎﺳ بو •<br><br>: https://www.usa.gov/presidentﺎﮑﯾﺮﻣآ يرﻮﻬﻤﺟ ﺖﺳﺎﯾر ﯽﻤﺳر ﺖﯾﺎﺳبو <br><br>|20 ﺦﯾرﺎﺗ زا يو .ﺪﺷﺎﺑﯽﻣ ﭗﻣاﺮﺗ ﺪﻟﺎﻧود ﺎﮑﯾﺮﻣآ هﺪﺤﺘﻣ تﻻﺎﯾا ﯽﻧﻮﻨﮐ رﻮﻬﻤﺟﺲﯿﺋر .دﺮﮐ زﺎﻏآ ار دﻮﺧ ﺖﯿﻟﺎﻌﻓ ﺎﮑﯾﺮﻣآ رﻮﻬﻤﺟﺲﯿﺋر ﻦﯿﻤﺘﻔﻫوﻞﻬﭼ ناﻮﻨﻋ ﻪﺑ 2025 ﻪﯾﻮﻧاژ<br><br>|
| |Translated<br><br>into<br><br>English<br><br>[Figure 786]<br><br>[Figure 787]<br><br>[Figure 788]<br><br>[Figure 789]<br><br>[Figure 790]<br><br>[Figure 791]<br><br>[Figure 792]<br><br>[Figure 793]<br><br>[Figure 794]<br><br>[Figure 795]<br><br>[Figure 796]<br><br>[Figure 797]<br><br>[Figure 798]<br><br>[Figure 799]<br><br>[Figure 800]<br><br>[Figure 801]<br><br>[Figure 802]<br><br>[Figure 803]<br><br>[Figure 804]<br><br>[Figure 805]<br><br>[Figure 806]<br><br>[Figure 807]<br><br>[Figure 808]<br><br>[Figure 809]<br><br>[Figure 810]<br><br>[Figure 811]<br><br>[Figure 812]<br><br>[Figure 813]<br><br>[Figure 814]| |The current President of the United States is Joe Biden. He has held this position since January 20, 2021. For more information, you can visit the following sources:<br><br>• Official White House website: https://www.whitehouse.gov/<br><br>• Official U.S. government presidential page: https://www.usa.gov/president<br><br><br>|The current President of the United States is Donald Trump. He began his term as the 47th President of the United States on January 20, 2025.<br><br>|
|Three<br><br>|[Figure 815]<br><br>[Figure 816]<br><br>Persian<br><br>ﯽﻧﺎﺘﺴﺑﺎﺗﮏﯿﭙﻤﻟايراﺰﮔﺮﺑ2024 رﻮﮐرﻪﭼﺲﯾرﺎﭘرد؟ﺪﻧﺎﺳرﺖﺒﺛﻪﺑاريﺪﯾﺪﺟيﺎﻫد<br><br>[Figure 817]<br><br>[Figure 818]<br><br>[Figure 819]<br><br>[Figure 820]<br><br>[Figure 821]<br><br>[Figure 822]<br><br>[Figure 823]<br><br>[Figure 824]<br><br>[Figure 825]<br><br>[Figure 826]<br><br>[Figure 827]<br><br>[Figure 828]<br><br>[Figure 829]<br><br>[Figure 830]<br><br>[Figure 831]<br><br>[Figure 832]<br><br>[Figure 833]<br><br>[Figure 834]<br><br>[Figure 835]<br><br>[Figure 836]<br><br>[Figure 837]<br><br>[Figure 838]<br><br>[Figure 839]<br><br>[Figure 840]<br><br>[Figure 841]<br><br>[Figure 842]<br><br>[Figure 843]<br><br>[Figure 844]<br><br>[Figure 845]<br><br>[Figure 846]<br><br>[Figure 847]<br><br>[Figure 848]<br><br>[Figure 849]<br><br>[Figure 850]<br><br>[Figure 851]<br><br>[Figure 852]<br><br>[Figure 853]<br><br>[Figure 854]<br><br>[Figure 855]<br><br>[Figure 856]<br><br>[Figure 857]<br><br>[Figure 858]<br><br>[Figure 859]<br><br>[Figure 860]<br><br>[Figure 861]<br><br>[Figure 862]<br><br>[Figure 863]<br><br>[Figure 864]<br><br>[Figure 865]<br><br>[Figure 866]<br><br>[Figure 867]<br><br>[Figure 868]<br><br>[Figure 869]<br><br>[Figure 870]<br><br>[Figure 871]<br><br>[Figure 872]<br><br>[Figure 873]<br><br>[Figure 874]<br><br>[Figure 875]<br><br>[Figure 876]<br><br>[Figure 877]<br><br>[Figure 878]<br><br>[Figure 879]<br><br>[Figure 880]<br><br>[Figure 881]<br><br>[Figure 882]<br><br>[Figure 883]<br><br>[Figure 884]<br><br>[Figure 885]<br><br>[Figure 886]<br><br>[Figure 887]<br><br>[Figure 888]<br><br>[Figure 889]<br><br>[Figure 890]<br><br>[Figure 891]<br><br>[Figure 892]<br><br>[Figure 893]<br><br>[Figure 894]<br><br>[Figure 895]<br><br>[Figure 896]<br><br>[Figure 897]<br><br>[Figure 898]<br><br>[Figure 899]<br><br>[Figure 900]<br><br>[Figure 901]<br><br>[Figure 902]<br><br>[Figure 903]<br><br>[Figure 904]<br><br>[Figure 905]<br><br>[Figure 906]<br><br>[Figure 907]<br><br>[Figure 908]<br><br>[Figure 909]<br><br>[Figure 910]<br><br>[Figure 911]<br><br>[Figure 912]<br><br>[Figure 913]<br><br>[Figure 914]<br><br>[Figure 915]<br><br>[Figure 916]<br><br>[Figure 917]<br><br>[Figure 918]<br><br>[Figure 919]<br><br>[Figure 920]<br><br>[Figure 921]<br><br>[Figure 922]<br><br>[Figure 923]<br><br>[Figure 924]<br><br>[Figure 925]<br><br>[Figure 926]<br><br>[Figure 927]<br><br>[Figure 928]<br><br>[Figure 929]<br><br>[Figure 930]<br><br>[Figure 931]<br><br>[Figure 932]<br><br>[Figure 933]<br><br>[Figure 934]<br><br>[Figure 935]<br><br>[Figure 936]<br><br>[Figure 937]<br><br>[Figure 938]<br><br>[Figure 939]<br><br>[Figure 940]<br><br>[Figure 941]<br><br>[Figure 942]<br><br>[Figure 943]<br><br>[Figure 944]<br><br>[Figure 945]<br><br>[Figure 946]<br><br>[Figure 947]<br><br>[Figure 948]<br><br>[Figure 949]<br><br>[Figure 950]<br><br>[Figure 951]<br><br>[Figure 952]<br><br>[Figure 953]<br><br>[Figure 954]<br><br>[Figure 955]<br><br>[Figure 956]<br><br>[Figure 957]<br><br>[Figure 958]<br><br>[Figure 959]<br><br>[Figure 960]<br><br>[Figure 961]<br><br>[Figure 962]<br><br>[Figure 963]<br><br>[Figure 964]<br><br>[Figure 965]<br><br>[Figure 966]<br><br>[Figure 967]<br><br>[Figure 968]<br><br>[Figure 969]<br><br>[Figure 970]<br><br>[Figure 971]<br><br>[Figure 972]<br><br>[Figure 973]<br><br>[Figure 974]<br><br>[Figure 975]<br><br>[Figure 976]<br><br>[Figure 977]<br><br>[Figure 978]<br><br>[Figure 979]<br><br>[Figure 980]<br><br>[Figure 981]<br><br>[Figure 982]<br><br>[Figure 983]<br><br>[Figure 984]<br><br>[Figure 985]<br><br>[Figure 986]<br><br>[Figure 987]<br><br>[Figure 988]<br><br>[Figure 989]<br><br>[Figure 990]<br><br>[Figure 991]<br><br>[Figure 992]<br><br>[Figure 993]<br><br>[Figure 994]<br><br>[Figure 995]<br><br>[Figure 996]<br><br>[Figure 997]<br><br>[Figure 998]<br><br>[Figure 999]<br><br>[Figure 1000]<br><br>[Figure 1001]<br><br>[Figure 1002]<br><br>[Figure 1003]<br><br>[Figure 1004]<br><br>[Figure 1005]<br><br>[Figure 1006]<br><br>[Figure 1007]<br><br>[Figure 1008]<br><br>[Figure 1009]<br><br>[Figure 1010]<br><br>[Figure 1011]<br><br>[Figure 1012]<br><br>[Figure 1013]<br><br>[Figure 1014]<br><br>[Figure 1015]<br><br>[Figure 1016]<br><br>[Figure 1017]<br><br>[Figure 1018]<br><br>[Figure 1019]<br><br>[Figure 1020]<br><br>[Figure 1021]<br><br>[Figure 1022]<br><br>[Figure 1023]<br><br>[Figure 1024]<br><br>[Figure 1025]<br><br>[Figure 1026]<br><br>[Figure 1027]<br><br>[Figure 1028]<br><br>[Figure 1029]<br><br>[Figure 1030]<br><br>[Figure 1031]<br><br>[Figure 1032]<br><br>[Figure 1033]<br><br>[Figure 1034]<br><br>[Figure 1035]<br><br>[Figure 1036]<br><br>[Figure 1037]<br><br>[Figure 1038]<br><br>[Figure 1039]<br><br>[Figure 1040]<br><br>[Figure 1041]<br><br>[Figure 1042]<br><br>[Figure 1043]<br><br>[Figure 1044]<br><br>[Figure 1045]<br><br>[Figure 1046]<br><br>[Figure 1047]<br><br>[Figure 1048]<br><br>[Figure 1049]<br><br>[Figure 1050]<br><br>[Figure 1051]<br><br>[Figure 1052]| |:ﻢﯾزادﺮﭘﯽﻣ ﺎﻫدرﻮﮐر ﻦﯾا ﮏﯿﮑﻔﺗ ﻪﺑ ﺎﺠﻨﯾا رد .دز ﻢﻗر ار ﯽﺨﯾرﺎﺗ و ﯽﻧﺪﻧﺎﻣ دﺎﯾ ﻪﺑ هرود ﮏﯾ ،ﻒﻠﺘﺨﻣ يﺎﻫ ﻪﺒﻨﺟ رد يدﺪﻌﺘﻣ يﺎﻫدرﻮﮐر ﺖﺒﺛ ﺎﺑ ﺲﯾرﺎﭘ 2024 ﯽﻧﺎﺘﺴﺑﺎﺗ ﮏﯿﭙﻤﻟ<br><br>(Sporting Records):ﯽﺷزرو يﺎﻫدرﻮﮐر<br><br>:ﺎﻫ لاﺪﻣ داﺪﻌﺗ • . دﺮﮐ ﺖﺒﺛ دﻮﺧ مﺎﻧ ﻪﺑ ار ﮏﯿﭙﻤﻟا هرود ﮏﯾ رد هﺪﺷ ﺐﺴﮐ لاﺪﻣ داﺪﻌﺗ ﻦﯾﺮﺘﺸﯿﺑ درﻮﮐر (ﺰﻧﺮﺑ 43 ،هﺮﻘﻧ 41 ،ﻼﻃ 37 ) لاﺪﻣ 121 ﺐﺴﮐ ﺎﺑ: ﺎﮑﯾﺮﻣآ هﺪﺤﺘﻣ تﻻﺎﯾا • . ﺖﺷاﺬﮔ ﺶﯾﺎﻤﻧ ﻪﺑ ار ﯽﺑﻮﺧ رﺎﯿﺴﺑ دﺮﮑﻠﻤﻋ و ﺖﻓﺮﮔ راﺮﻗ مود ﻪﺒﺗر رد (ﺰﻧﺮﺑ 18 ،هﺮﻘﻧ 32 ،ﻼﻃ 38 ) لاﺪﻣ 88 ﺎﺑ: ﻦﯿﭼ • . دﺎﺘﺴﯾا مﻮﺳ ﻪﺒﺗر رد (ﺰﻧﺮﺑ 24 ،هﺮﻘﻧ 23 ،ﻼﻃ 20 ) لاﺪﻣ 67 ﺎﺑ: ﺎﯿﻧﺎﺘﯾﺮﺑ • :يداﺮﻔﻧا يﺎﻫ ﯽﻨﮑﺷدرﻮﮐر • . ﺖﺴﮑﺷ ﻪﯿﻧﺎﺛ 46,97 نﺎﻣز ﺎﺑ ار دازآ ﺮﺘﻣ 100 ﯽﻧﺎﻬﺟ درﻮﮐر: (ﺎﻨﺷ) ﺰﻧﻮﺟ ﻞﯿﻧد • . دﺮﮐ ﺖﺒﺛ دﻮﺧ مﺎﻧ ﻪﺑ ﺮﺘﻣ 21,42 ﺎﺑ ار ﻪﻧزو بﺎﺗﺮﭘ ﯽﻧﺎﻬﺟ درﻮﮐر: (ﯽﻧاﺪﯿﻣوود) ﯽﻧﺎﯾرآ ﺎﻧﺎﯾرآ • .ﺪﻧﺎﺳر ﺖﺒﺛ ﻪﺑ ار يﺪﯾﺪﺟ درﻮﮐر BMX ﻪﺘﺷر رد: (يراﻮﺳﻪﺧﺮﭼود) نﻮﻨﯿﮐ ﮏﻣ ﯽﻠﯾﺎﮐ • .ﺪﻧﺪﯿﺸﺨﺑ دﻮﺒﻬﺑ ار دﻮﺧ ﯽﻠﻣ يﺎﻫدرﻮﮐر ﺎﯾ و ﮏﯿﭙﻤﻟا يﺎﻫدرﻮﮐر ... و ﯽﻧاﺮﻘﯾﺎﻗ ،يرادﺮﺑﻪﻧزو ،يزاﺪﻧاﺮﯿﺗ ﺪﻨﻧﺎﻣ ﻒﻠﺘﺨﻣ يﺎﻫ ﻪﺘﺷر رد: ﺮﮕﯾد نارﺎﮑﺷزرو زا يرﺎﯿﺴﺑ •<br><br>(.ﺖﺴﯿﻧ ﺮﯾﺬﭘنﺎﮑﻣا ﺦﺳﺎﭘ ﻦﯾا رد ﺎﻫنآ ﻪﻤﻫ ﺮﮐذ ،دﺎﯾز داﺪﻌﺗ ﻞﯿﻟد ﻪﺑ)<br><br>: ﯽﻤﯿﺗ ﯽﻨﮑﺷدرﻮﮐر • . ﺖﺴﮑﺷ ﻪﻘﯿﻗد 3:07,45 نﺎﻣز ﺎﺑ ار ﯽﻧﺎﻬﺟ درﻮﮐر: ﺎﮑﯾﺮﻣآ نادﺮﻣ دازآ ﺮﺘﻣx100 4 ﺎﺘﻓﺎﺘﺳا ﻢﯿﺗ • . ﺖﺴﮑﺷ ﻪﻘﯿﻗد 7:33,15 نﺎﻣز ﺎﺑ ار ﯽﻧﺎﻬﺟ درﻮﮐر: ﺎﯿﻟاﺮﺘﺳا نﺎﻧز دازآ ﺮﺘﻣx200 4 ﺎﺘﻓﺎﺘﺳا ﻢﯿﺗ •<br><br>.ﺪﻨﺘﻓﺎﯾ ﺖﺳد يﺪﯾﺪﺟ يﺎﻫدرﻮﮐر ﻪﺑ و ﺪﻨﺘﺷاد تارﺎﻈﺘﻧا زا ﺮﺗاﺮﻓ يدﺮﮑﻠﻤﻋ ﺰﯿﻧ ... و لﺎﺑﺪﻨﻫ ،لﺎﺒﺘﮑﺴﺑ ،لﺎﺒﯿﻟاو ﺪﻨﻧﺎﻣ ﯽﻤﯿﺗ يﺎﻫﻪﺘﺷر رد ﻒﻠﺘﺨﻣ يﺎﻫ ﻢﯿﺗ •<br><br>:يروﺎﻨﻓ و ﯽﻫﺪﻧﺎﻣزﺎﺳ ﻪﺑ طﻮﺑﺮﻣ يﺎﻫدرﻮﮐر<br><br>ﻪﺑ ﯽﻄﯿﺤﻣﺖﺴﯾز يراﺪﯾﺎﭘ ﻪﻨﯿﻣز رد يدرﻮﮐر ،ﺮﯾﺬﭘﺪﯾﺪﺠﺗ يﺎﻫ يژﺮﻧا زا هدﺎﻔﺘﺳا و ﻦﺑﺮﮐ ﺶﻫﺎﮐ ﺎﺑ و ﺪﺷ ﻪﺘﺧﺎﻨﺷ "ﺰﺒﺳ ﮏﯿﭙﻤﻟا" ناﻮﻨﻋ ﻪﺑ ﺲﯾرﺎﭘ ﮏﯿﭙﻤﻟا: يراﺪﯾﺎﭘ •<br><br>.ﺪﻧﺎﺳر ﺖﺒﺛ ﻪﻨﯿﻣز رد درﻮﮐر ﮏﯾ ،نارﺎﮑﺷزرو و ناﺮﮔﺎﺷﺎﻤﺗ ﻪﺑﺮﺠﺗ دﻮﺒﻬﺑ ياﺮﺑ ﺎﯿﺷا ﺖﻧﺮﺘﻨﯾا و يزﺎﺠﻣ ﺖﯿﻌﻗاو ،ﯽﻋﻮﻨﺼﻣ شﻮﻫ ﺪﻨﻧﺎﻣ ﻦﯾﻮﻧ يﺎﻫيروﺎﻨ ﻓ زا هدﺎﻔﺘﺳا: يروﺎﻨﻓ •<br><br>.دﻮﺷﯽﻣ بﻮﺴﺤﻣ ﮏﯿﭙﻤﻟا رد يروﺎﻨﻓ زا هدﺎﻔﺘﺳا ﺰﯿﻧ ﻦﯾا ﻪﮐ ﺪﺷ راﺰﮔﺮﺑ ﯽﺘﯿﻨﻣا ﻪﺛدﺎﺣ ناﺰﯿﻣ ﻦﯾﺮﺘﻤﮐ ﺎﺑ ﺲﯾرﺎﭘ ﮏﯿﭙﻤﻟا ،ﯽﺘﯿﻨﻣا يﺎﻫوﺮﯿﻧ هدﺮﺘﺴﮔ يرﺎﮑﻤﻫ و ﻪﺘﻓﺮﺸﯿﭘ ﯽﺘﯿﻨﻣا يﺎﻫﻢﺘﺴﯿﺳ يﺮﯿﮔرﺎﮐ ﻪﺑ ﺎﺑ: ﺖﯿﻨﻣا •<br><br>.دﻮﺷﯽﻣ بﻮﺴﺤﻣ درﻮﮐر ﮏﯾ ،ﺮﻣا ﻦﯾا ﻪﮐ دﺮﮐ يزﺎﺳزﺎﺑ ار ﯽﻤﯾﺪﻗ يﺎﻫ هﺎﮕﺷزرو و ﺖﺧﺎﺳ ﺪﯾﺪﺟ هﺎﮕﺷزرو يداﺪﻌﺗ ،ﮏﯿﭙﻤﻟا يراﺰﮔﺮﺑ ياﺮﺑ ﺲﯾرﺎﭘ: هﺪﺷ يزﺎﺳزﺎﺑ و ﺪﯾﺪﺟ يﺎﻫ هﺎﮕﺷزرو داﺪﻌﺗ •<br><br>.دورﯽﻣ رﺎﻤﺷ ﻪﺑ ﯽﺷزرو يﺎﻫﺖﺧﺎﺳﺮﯾز ﻪﻨﯿﻣز رد يدرﻮﮐر<br><br>|. دﻮﺑ داﺪﯾور 329 و ﻪﺘﺷر 32 رد ﮏﯿﭙﻤﻟا ﯽﻠﻣ ﻪﺘﯿﻤﮐ 206 زا ﯽﻧارﺎﮑﺷزرو نﺎﺑﺰﯿﻣ و ﺪﺷ راﺰﮔﺮﺑ ﺲﯾرﺎﭘ رد 2024 توا 11 ﺎﺗ ﻪﯿﺋوژ 26 زا 2024 ﺲﯾرﺎﭘ ﯽﻧﺎﺘﺴﺑﺎﺗ ﮏﯿﭙﻤﻟ<br><br>(Sporting Records):ﯽﺷزرو يﺎﻫدرﻮﮐر<br><br>:ﺎﻫ لاﺪﻣ داﺪﻌﺗ • عﻮﻤﺠﻣ ﻦﯾﺮﺗﻻﺎﺑ زا ﯽﮑﯾ ،درﻮﮐر ﻦﯾا .داد صﺎﺼﺘﺧا دﻮﺧ ﻪﺑ ار لوﺪﺟ رﺪﺻ (ﺰﻧﺮﺑ 42 ،هﺮﻘﻧ 44 ،ﻼﻃ 40 ) لاﺪﻣ 126 ﺐﺴﮐ ﺎﺑ: ﺎﮑﯾﺮﻣآ هﺪﺤﺘﻣ تﻻﺎﯾا o<br><br>.دورﯽﻣ رﺎﻤﺷ ﻪﺑ ﮏﯿﭙﻤﻟا ﺦﯾرﺎﺗ يﺎﻫلاﺪﻣ<br><br>. ﺖﻓﺮﮔ راﺮﻗ مود هدر رد (ﺰﻧﺮﺑ 24 ،هﺮﻘﻧ 27 ،ﻼﻃ 40 ) لاﺪﻣ 91 ﺎﺑ: ﻦﯿﭼ o . دﺎﺘﺴﯾا مﻮﺳ ﻪﺒﺗر رد (ﺰﻧﺮﺑ 29 ،هﺮﻘﻧ 22 ،ﻼﻃ 14 ) لاﺪﻣ 65 ﺎﺑ: ﺎﯿﻧﺎﺘﯾﺮﺑ o<br><br>: ﮏﯿﭙﻤﻟا و ﯽﻧﺎﻬﺟ يﺎﻫدرﻮﮐر •<br><br>. ﺪﺷ ﺖﺒﺛ هرود ﻦﯾا رد ﯽﻧﺎﻬﺟ درﻮﮐر 32 دوﺪﺣ و ﮏﯿﭙﻤﻟا درﻮﮐر 125 زا ﺶﯿﺑ o : ﻪﺘﺴﺟﺮﺑ يﺎﻫدرﻮﮐر ﻪﻠﻤﺟ زا o<br><br>.داد ءﺎﻘﺗرا ا ر دﻮﺧ ﯽﻧﺎﻬﺟ درﻮﮐر هﺰﯿﻧ ﺎﺑ شﺮﭘ رد (ﺪﺋﻮﺳ) ﺲﯿﺘﻧﻼﭘود ﺪﻧﺎﻣرا  . ﺖﺴﮑﺷ ار ﯽﻧﺎﻬﺟ درﻮﮐر ﻪﯿﻧﺎﺛ 50,37 نﺎﻣز ﺖﺒﺛ ﺎﺑ ،ﻊﻧﺎﻣ ﺎﺑ ﺮﺘﻣ 400 رد (ﺎﮑﯾﺮﻣآ) نوروﻮﻟ-ﻦﯿﻠﮐﻻ ﮏﻣ ﯽﻧﺪﯿﺳ  . ﺪﻧﺎﺳر ﺖﺒﺛ ﻪﺑ ار ﮏﯿﭙﻤﻟا درﻮﮐر 3:07,41 نﺎﻣز ﺎﺑ ﯽﺗﺎﻣﺪﻘﻣ ﻪﻠﺣﺮﻣ رد ﺎﮑﯾﺮﻣآ ﻂﻠﺘﺨﻣ ﺮﺘﻣ 400×4 ﯽﺒﯿﮐﺮﺗ ﻪﻟر ﻢﯿﺗ  . دﺮﮐ ﺖﺒﺛ هدﺎﻣ ﻦﯾا رد يﺪﯾﺪﺟ درﻮﮐر 2:06:26 نﺎﻣز ﺎ ﺑ ،نادﺮﻣ ﻦﺗارﺎﻣ يود رد (ﯽﭘﻮﯿﺗا) ﻻﻮﺗ تاﺮﯿﻣﺎﺗ  . ﺖﺴﮑﺷ ار ﮏﯿﭙﻤﻟا درﻮﮐر 3:51,29 نﺎﻣز ﺎﺑ نﺎﻧز ﺮﺘﻣ 1500 رد (ﺎﯿﻨﮐ) ﻦﮕﭙﯿﮐ ﺖﯿﻓ  . دﺮﮐ ﺖﺒﺛ ار يﺪﯾﺪﺟ ﯽﻧﺎﻬﺟ درﻮﮐر 14:30,67 نﺎﻣز ﺎﺑ ،نادﺮﻣ دازآ ﺮﺘﻣ 1500 يﺎﻨﺷ رد (ﺎﮑﯾﺮﻣآ) ﮏﻨﯿﻓ ﯽﺑﺎﺑ <br><br>.ﺪﯿﺳر ﺖﺒﺛ ﻪﺑ يدﺪﻌﺘﻣ ﯽﻠﻣ ﺎﯾ ﮏﯿﭙﻤﻟا ،ﯽﻧﺎﻬﺟ يﺎﻫدرﻮﮐر ،ﺎﻫﻪﺘﺷر ﺮﮕﯾد و ﯽﻧاﺮﻘﯾﺎﻗ ،يرادﺮﺑﻪﻧزو ،يزاﺪﻧاﺮﯿﺗ ،يراﻮﺳﻪﺧﺮﭼود ﺪﻨﻧﺎﻣ ﯽﯾﺎﻫ ﻪﺘﺷر رد ﻦﯿﻨﭽﻤﻫ o<br><br>:يراﺪﯾﺎﭘ و يروﺎﻨﻓ ،ﯽﻫﺪﻧﺎﻣزﺎﺳ ﻪﺑ طﻮﺑﺮﻣ يﺎﻫدرﻮﮐر<br><br>. ﺪﺷ ﻪﺘﺧوﺮﻓ نآ ياﺮﺑ ﻂﯿﻠﺑ راﺰﻫ 326 دوﺪﺣ و ﺪﺷ راﺰﮔﺮﺑ ﻦﺳ ﻪﻧﺎﺧدور رﺎﻨﮐ رد و هﺎﮕﺷزرو زا جرﺎﺧ ﮏﯿﭙﻤﻟا ﺦﯾرﺎﺗ رد رﺎﺑ ﻦﯿﺘﺴﺨﻧ ياﺮﺑ ﻪﯿﺣﺎﺘﺘﻓا ﻢﺳاﺮﻣ •<br><br>. ﺪﺷ ﻪﺘﺧﺎﻨﺷ "ﺰﺒﺳ ﮏﯿﭙﻤﻟا" ناﻮﻨﻋ ﻪﺑ ،ﮏﯿﭙﻤﻟا هﺪﮑﻫد رد ﯽﻫﺎﯿﮔ يﺎﻫاﺬﻏ ﻪﺋارا و ،ﺮﯾﺬﭘﺪﯾﺪﺠﺗ يﺎﻫيژﺮﻧا زا هدﺎﻔﺘﺳا ،ﻦﺑﺮﮐ رﺎﺸﺘﻧا ﺶﻫﺎﮐ فﺪﻫ ﺎﺑ ﺲﯾرﺎﭘ ﮏﯿﭙﻤﻟا • ﻪﺘﺴﺟﺮﺑ يﺎﻫﯽﮔﮋﯾو زا ﯽﮑﯾ ،ﺎﻫﻪﻧﺎﺳر و نارﺎﮑﺷزرو ،ناﺮﮔﺎﺷﺎﻤﺗ ﻪﺑﺮﺠﺗ يﺎﻘﺗرا ياﺮﺑ ﺎﯿﺷا ﺖﻧﺮﺘﻨﯾا و يزﺎﺠﻣ ﺖﯿﻌﻗاو ،ﯽﻋﻮﻨﺼﻣ شﻮﻫ ﺪﻨﻧﺎﻣ ﻦﯾﻮﻧ يﺎﻫ يروﺎﻨﻓ زا هدﺎﻔﺘﺳا •<br><br>.دﻮﺑ هرود ﻦﯾا<br><br>.ﺪﺷ راﺰﮔﺮﺑ ﯽﺘﯿﻨﻣا يﺎﻫداﺪﺧر ناﺰﯿﻣ ﻦﯾﺮﺘﻤﮐ ﺎﺑ ﮏﯿﭙﻤﻟا ﻦﯾا ،ﯽﻠﻠﻤﻟاﻦﯿﺑ يﺎﻫوﺮﯿﻧ يرﺎﮑﻤﻫ و ﻪﺘﻓﺮﺸﯿﭘ ﯽﺘﯿﻨﻣا يﺎﻫﻢﺘﺴﯿﺳ زا يﺮﯿﮔ هﺮﻬﺑ ﺎﺑ •<br><br>|
| |TranslatedintoEnglish<br><br>WhatnewrecordsweresetduringtheParis2024SummerOlympics?<br><br>[Figure 1053]<br><br>[Figure 1054]<br><br>[Figure 1055]<br><br>[Figure 1056]<br><br>[Figure 1057]<br><br>[Figure 1058]<br><br>[Figure 1059]<br><br>[Figure 1060]<br><br>[Figure 1061]<br><br>[Figure 1062]<br><br>[Figure 1063]<br><br>[Figure 1064]<br><br>[Figure 1065]<br><br>[Figure 1066]<br><br>[Figure 1067]<br><br>[Figure 1068]<br><br>[Figure 1069]<br><br>[Figure 1070]<br><br>[Figure 1071]<br><br>[Figure 1072]<br><br>[Figure 1073]<br><br>[Figure 1074]<br><br>[Figure 1075]<br><br>[Figure 1076]<br><br>[Figure 1077]<br><br>[Figure 1078]<br><br>[Figure 1079]<br><br>[Figure 1080]<br><br>[Figure 1081]<br><br>[Figure 1082]<br><br>[Figure 1083]<br><br>[Figure 1084]<br><br>[Figure 1085]<br><br>[Figure 1086]<br><br>[Figure 1087]<br><br>[Figure 1088]<br><br>[Figure 1089]<br><br>[Figure 1090]<br><br>[Figure 1091]<br><br>[Figure 1092]<br><br>[Figure 1093]<br><br>[Figure 1094]<br><br>[Figure 1095]<br><br>[Figure 1096]<br><br>[Figure 1097]<br><br>[Figure 1098]<br><br>[Figure 1099]<br><br>[Figure 1100]<br><br>[Figure 1101]<br><br>[Figure 1102]<br><br>[Figure 1103]<br><br>[Figure 1104]<br><br>[Figure 1105]<br><br>[Figure 1106]<br><br>[Figure 1107]<br><br>[Figure 1108]<br><br>[Figure 1109]<br><br>[Figure 1110]<br><br>[Figure 1111]<br><br>[Figure 1112]<br><br>[Figure 1113]<br><br>[Figure 1114]<br><br>[Figure 1115]<br><br>[Figure 1116]<br><br>[Figure 1117]<br><br>[Figure 1118]<br><br>[Figure 1119]<br><br>[Figure 1120]<br><br>[Figure 1121]<br><br>[Figure 1122]<br><br>[Figure 1123]<br><br>[Figure 1124]<br><br>[Figure 1125]<br><br>[Figure 1126]<br><br>[Figure 1127]<br><br>[Figure 1128]<br><br>[Figure 1129]<br><br>[Figure 1130]<br><br>[Figure 1131]<br><br>[Figure 1132]<br><br>[Figure 1133]<br><br>[Figure 1134]<br><br>[Figure 1135]<br><br>[Figure 1136]<br><br>[Figure 1137]<br><br>[Figure 1138]<br><br>[Figure 1139]<br><br>[Figure 1140]<br><br>[Figure 1141]<br><br>[Figure 1142]<br><br>[Figure 1143]<br><br>[Figure 1144]<br><br>[Figure 1145]<br><br>[Figure 1146]<br><br>[Figure 1147]<br><br>[Figure 1148]<br><br>[Figure 1149]<br><br>[Figure 1150]<br><br>[Figure 1151]<br><br>[Figure 1152]<br><br>[Figure 1153]<br><br>[Figure 1154]<br><br>[Figure 1155]<br><br>[Figure 1156]<br><br>[Figure 1157]<br><br>[Figure 1158]<br><br>[Figure 1159]<br><br>[Figure 1160]<br><br>[Figure 1161]<br><br>[Figure 1162]<br><br>[Figure 1163]<br><br>[Figure 1164]<br><br>[Figure 1165]<br><br>[Figure 1166]<br><br>[Figure 1167]<br><br>[Figure 1168]<br><br>[Figure 1169]<br><br>[Figure 1170]<br><br>[Figure 1171]<br><br>[Figure 1172]<br><br>[Figure 1173]<br><br>[Figure 1174]<br><br>[Figure 1175]<br><br>[Figure 1176]<br><br>[Figure 1177]<br><br>[Figure 1178]<br><br>[Figure 1179]<br><br>[Figure 1180]<br><br>[Figure 1181]<br><br>[Figure 1182]<br><br>[Figure 1183]<br><br>[Figure 1184]<br><br>[Figure 1185]<br><br>[Figure 1186]<br><br>[Figure 1187]<br><br>[Figure 1188]<br><br>[Figure 1189]<br><br>[Figure 1190]<br><br>[Figure 1191]<br><br>[Figure 1192]<br><br>[Figure 1193]<br><br>[Figure 1194]<br><br>[Figure 1195]<br><br>[Figure 1196]<br><br>[Figure 1197]<br><br>[Figure 1198]<br><br>[Figure 1199]<br><br>[Figure 1200]<br><br>[Figure 1201]<br><br>[Figure 1202]<br><br>[Figure 1203]<br><br>[Figure 1204]<br><br>[Figure 1205]<br><br>[Figure 1206]<br><br>[Figure 1207]<br><br>[Figure 1208]<br><br>[Figure 1209]<br><br>[Figure 1210]<br><br>[Figure 1211]<br><br>[Figure 1212]<br><br>[Figure 1213]<br><br>[Figure 1214]<br><br>[Figure 1215]<br><br>[Figure 1216]<br><br>[Figure 1217]<br><br>[Figure 1218]<br><br>[Figure 1219]<br><br>[Figure 1220]<br><br>[Figure 1221]<br><br>[Figure 1222]<br><br>[Figure 1223]<br><br>[Figure 1224]<br><br>[Figure 1225]<br><br>[Figure 1226]<br><br>[Figure 1227]<br><br>[Figure 1228]<br><br>[Figure 1229]<br><br>[Figure 1230]<br><br>[Figure 1231]<br><br>[Figure 1232]<br><br>[Figure 1233]<br><br>[Figure 1234]<br><br>[Figure 1235]<br><br>[Figure 1236]<br><br>[Figure 1237]<br><br>[Figure 1238]<br><br>[Figure 1239]<br><br>[Figure 1240]<br><br>[Figure 1241]<br><br>[Figure 1242]<br><br>[Figure 1243]<br><br>[Figure 1244]<br><br>[Figure 1245]<br><br>[Figure 1246]<br><br>[Figure 1247]<br><br>[Figure 1248]<br><br>[Figure 1249]<br><br>[Figure 1250]<br><br>[Figure 1251]<br><br>[Figure 1252]<br><br>[Figure 1253]| |The Paris 2024 Summer Olympics set numerous new records across various aspects, making it a memorable and historic event. Here's a breakdown of those records:<br><br>Sporting Records:<br><br>• Medal Counts:<br><br>o United States: With 121 medals (37 gold, 41 silver, 43 bronze), the U.S. set the record for the most medals won in a single Olympic Games.<br><br>o China: Secured second place with 88 medals (38 gold, 32 silver, 18 bronze), showcasing an outstanding performance.<br><br>o Great Britain: Ranked third with 67 medals (20 gold, 23 silver, 24 bronze).<br><br><br>• Individual World Records:<br><br>o Daniel Jones (Swimming): Set a new world record in the 100m freestyle with a time of 46.97 seconds.<br><br>o Ariana Ariani (Athletics): Broke the world record in shot put with a distance of 21.42 meters.<br><br>o Kylie McKinnon (Cycling): Set a new record in BMX.<br><br>o Many other athletes also broke Olympic or national records in sports such as shooting, weightlifting, rowing, and more (due to the high number, not all can be listed here).<br><br><br>• Team World Records:<br><br><br>o USA Men’s 4x100m Freestyle Relay Team: Broke the world record with a time of 3:07.45 minutes.<br><br>o Australia Women’s 4x200m Freestyle Relay Team: Set a world record with a time of 7:33.15 minutes.<br><br>o Teams in sports such as volleyball, basketball, and handball also exceeded expectations and achieved new records.<br><br><br>Organizational and Technological Records:<br><br>• Sustainability: Known as the “Green Olympics,” Paris 2024 set a benchmark in environmental sustainability by reducing carbon emissions and using renewable energy.<br><br>• Technology: Advanced technologies like artificial intelligence, virtual reality, and the Internet of Things enhanced the experience for athletes and spectators — a first in Olympic history.<br><br>• Security: With cutting-edge security systems and coordinated efforts among security forces, Paris hosted one of the safest Games in history.<br><br>• New and Renovated Venues: Numerous new sports venues were built and existing ones renovated, marking a record in sports infrastructure development.<br><br><br>|The Paris 2024 Summer Olympics were held from July 26 to August 11, 2024, hosting athletes from 206 National Olympic Committees across 32 sports and 329 events.<br><br>Sporting Records:<br><br>• Medal Counts:<br><br>o United States: Topped the medal table with 126 medals (40 gold, 44 silver, 42 bronze), marking one of the highest total medal counts in Olympic history.<br><br>o China: Ranked second with 91 medals (40 gold, 27 silver, 24 bronze).<br><br>o Great Britain: Finished third with 65 medals (14 gold, 22 silver, 29 bronze).<br><br><br>• World and Olympic Records:<br><br>o Over 125 Olympic records and approximately 32 world records were set during the Games.<br><br>o Notable highlights include:<br><br> Armand Duplantis (Sweden) improved his own world record in pole vault.<br><br> Sydney McLaughlin-Levrone (USA) broke the world record in the women’s 400m hurdles with a time of 50.37 seconds.<br><br> USA’s mixed 4×400m relay team set a new Olympic record in the heats with a time of 3:07.41.<br><br> Tamirat Tola (Ethiopia) set a new record in the men’s marathon with a time of 2:06:26.<br><br> Faith Kipyegon (Kenya) broke the Olympic record in the women’s 1500m with a time of 3:51.29.<br><br> Bobby Finke (USA) set a new world record in the men’s 1500m freestyle swimming with a time of 14:30.67.<br><br><br>o Many other world, Olympic, and national records were also broken in disciplines such as cycling, shooting, weightlifting, rowing, and more.<br><br><br>Records in Organization, Technology, and Sustainability:<br><br>• The opening ceremony was held outside a stadium for the first time in Olympic history, along the Seine River, with about 326,000 tickets sold.<br><br>• Paris 2024 was recognized as the "Green Olympics" due to efforts to reduce carbon emissions, use renewable energy, and provide plant-based food options in the Olympic Village.<br><br>• The use of advanced technologies such as artificial intelligence, virtual reality, and the Internet of Things significantly enhanced the experience for spectators, athletes, and media.<br><br>• With cutting-edge security systems and international cooperation, this edition of the Olympics had the lowest number of security incidents recorded<br><br><br>|

[Figure 1254]

[Figure 1255]

زوﺮاشزروناﺮ

لﺎﺳرد1403

Persian

ﮐﯿ؟ﺖﺴ

One

MinisterofSports

ofIranintheyear

Translated

1404(20242025)

Whoisthe

English

into

هﺪﺤﺘﻣتﻻﺎﯾارﻮﻬﻤﺟﺲﯿﯾر

Persian

؟ﻪﯿﮐ

Tow

Presidentofthe

Translated

UnitedStates?

Whoisthe

English

into

ﯽﻧﺎﺘﺴﺑﺎﺗﮏﯿﭙﻤﻟايراﺰﮔﺮﺑ2024رﻮﮐرﻪﭼﺲﯾرﺎﭘردﺖﺒﺛﻪﺑاريﺪﯾﺪﺟيﺎﻫد

Persian

[Figure 1256]

[Figure 1257]

- Table 4: Representative examples of minimally revised responses with factual updates. Each row includes a user instruction, a rejected response containing outdated or incorrect information, and a chosen response with concise, targeted edits. The examples cover domains such as politics, international affairs, and sports, demonstrating factual correction with minimal changes to phrasing and tone.

- • Deeper exploration of RAG methods and external APIs: Our current pipeline only briefly references the use of retrievalaugmented generation (RAG) techniques and external APIs, without a thorough analysis of their practical impact. Future work could investigate how different RAG configurations, API sources, and retrieval strategies affect data diversity and factual accuracy across domains.
- • Leveraging agent-based multi-step interactions for RL training: Large language model

agents are increasingly used to perform multistep tasks such as guided retrieval, source evaluation, answer synthesis, and refinement. Capturing these rich, interactive sequences offers an opportunity to create environments or trajectories for reinforcement learning (RL). This could facilitate training of more capable agents in real-world, information-seeking scenarios.

• Integration with structured knowledge sources: Incorporating structured resources—such as knowledge graphs,

###### Model / Task LoRA Rank LoRA Alpha DeepSpeed Stage Learning Rate Dropout Batch Size Epochs Pref.β

Matina-8B (SFT)

Culinary 128 256 Z0 1 × 10−4 0.05 64 3 – Tourism 256 512 Z0 1 × 10−4 0.05 64 4 –

Matina-70B (SFT)

Culinary 128 256 Z3 1 × 10−4 0.05 8 2 – Tourism 256 512 Z3 1 × 10−4 0.05 8 3 –

Gemma-3-27B-IT (ORPO) Update 16 32 Z3 2 × 10−5 0.005 4 6 0.05

- Table 5: Unified view of key hyperparameters for all training stages. Rows are grouped into three sections: SFT with Matina-8B, SFT with Matina-70B, and ORPO on Gemma-3-27B-IT.

ontologies, or semantic lexicons—could improve answer precision and coherence, especially in knowledge-intensive domains like law, medicine, or education. Future pipelines may combine retrieved text with structured representations to enhance factual grounding.

- • Automated optimization of the feedback loop: Our current framework includes an iterative improvement loop driven by human feedback and manual adjustments. A promising future direction is to formalize this loop using reinforcement learning or other optimization techniques, allowing the system to continuously refine its query generation, retrieval, and synthesis strategies based on performance signals.
- • Automated quality assessment tools: Evaluating the quality of generated training data remains a bottleneck. Future work could explore automatic metrics or learned models that assess dimensions such as evidential coverage, factual accuracy, and stylistic consistency, reducing reliance on manual review.

