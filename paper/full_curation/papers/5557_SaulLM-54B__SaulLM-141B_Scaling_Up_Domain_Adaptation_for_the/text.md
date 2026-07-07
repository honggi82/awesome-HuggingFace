# arXiv:2407.19584v1[cs.CL]28Jul2024

## SaulLM-54B & SaulLM-141B: Scaling Up Domain Adaptation for the Legal Domain

[Figure 1]

#### Pierre Colombo

Equall MICS - CentraleSupelec

[Figure 2]

Telmo Pires Equall

[Figure 3]

#### Dominic Culver

Equall

[Figure 4]

#### Sofia Morgado

Equall

Johanne Charpentier CINES

[Figure 5]

Malik Boudiaf

Equall

[Figure 6]

#### Rui Melo

Equall

Etienne Malaboeuf CINES

Gabriel Hautreux CINES

[Figure 7]

Michael Desa

Equall

[Figure 8]

### Abstract

In this paper, we introduce SaulLM-54B and SaulLM-141B, two large language models (LLMs) tailored for the legal sector. These models, which feature architectures of 54 billion and 141 billion parameters, respectively, are based on the Mixtral architecture. The development of SaulLM-54B and SaulLM-141B is guided by large-scale domain adaptation, divided into three strategies: (1) the exploitation of continued pretraining involving a base corpus that includes over 540 billion of legal tokens, (2) the implementation of a specialized legal instruction-following protocol, and (3) the alignment of model outputs with human preferences in legal interpretations. The integration of synthetically generated data in the second and third steps enhances the models’ capabilities in interpreting and processing legal texts, effectively reaching state-of-the-art performance and outperforming previous open-source models on LegalBench-Instruct. This work explores the trade-offs involved in domain-specific adaptation at this scale, offering insights that may inform future studies on domain adaptation using strong decoder models. Building upon SaulLM-7B, this study refines the approach to produce an LLM better equipped for legal tasks. We are releasing base, instruct, and aligned versions on top of SaulLM-54B and SaulLM-141B under the MIT License to facilitate reuse and collaborative research.

Preprint. Under review.

### 1 Introduction

LLMs have demonstrated exceptional capabilities across various domains [1, 65, 58, 73, 37, 38, 74, 7, 26], excelling in tasks such as language translation [6], medical diagnostics [16, 11, 12], and automated code generation [4, 40, 30], among others. These achievements highlight the potential for human-like communication through large language models (LLMs). Despite the significant potential benefits, the adaptation of most recent LLMs for legal tasks has not been extensively examined, with only two recent studies cited from [18, 53, 83], and its impact on society could be substantial. Indeed, at a time when legal systems in many countries are overburdened [69], the development of robust and high-performing legal LLMs could provide critical support to lawyers and judicial systems [63, 10]. However, adapting LLMs to the legal domain presents unique challenges, particularly because of the vast scale involved, with hundreds of billions of existing legal data available.

Previous efforts to tailor LLMs to the legal sector have encountered significant challenges [18, 53, 83]: first, a limited model scale, capped at 7/12B parameters, which is considerably smaller than the largest open-source models [7, 38]; second, training datasets restricted to no more than 30 billion tokens, significantly fewer than potentially available tokens [28, 52]. Given the importance of scale and breadth in effectively adapting LLMs to new domains, this paper aims to answer the following research question:

How much can we improve the specialization of generic LLMs for legal tasks by scaling up both model and corpus size?

In this paper, we present an empirical study on the scalability and domain adaptation of LLMs in the legal sector. Relying on a corpus exceeding 500B tokens and models up to 141B parameters, our research seeks to address the gaps in the examination of legal applications. A novel aspect of our approach is the adaptation of large-scale Mixture of Experts (MoE) models with 54B and 141B parameters, which have gained significant traction in recent months [90, 22, 43, 87, 84, 59]. Formally, this study makes two principal contributions:

- 1. Comprehensive Analysis of Domain Adaptation Strategies for Legal LLMs Domain adaptation for legal LLMs remains a challenging and somewhat underexplored area. This work advances the field by specializing each step in the process of developing modern LLMs, from continued pretraining to instruction fine-tuning and alignment, relying on both synthetic and real data. This paper offers a fresh perspective on the efficacy of each step and its value for adapting to the legal domain, potentially guiding further research in the legal domain as well as in other expert domains.
- 2. SaulLM-54B & SaulLM-141B: Joining SaulLM-7b to form a Family of Legal LLMs under Permissive License1 We specialize general-purpose, large-scale LLMs for the law. This work represents an ambitious advancement in terms of scale and leveraging the increasingly popular MoE architecture. While this architecture is widely used, its specific applications within focused domains, particularly the legal sector, are still largely unexplored. By releasing these models, we aim to foster further research in legal NLP and contribute to unlocking the full potential of LLMs.

### 2 Related Work

#### 2.1 Domain Specialization For Large Language Models

The process of domain specialization for LLMs has demonstrated promising results in areas such as medicine [16], science [71], translation [6, 5], or code [64, 40, 4]. Models like SciBERT [9], PubMedBERT [72], Galactica [71] and Meditron [16] have been specifically trained on domainrelated corpora to enhance their performance. Studies have identified that both the scale of the model and the size of the in-domain data are crucial for achieving strong domain adaptation [16, 64].

In the legal domain, earlier models such as LegalBERT [15], InCaseLawBERT [57], and SaulLM-7B [18], among others, while pioneering, have been constrained by their relatively small scale and the specificity of their training data, which covers a limited number of documents and jurisdictions. Our work aims to build on these efforts by deploying LLMs at an unprecedented scale, utilizing models of up to 141B parameters and a base corpus exceeding 500 billion tokens to significantly enhance the depth and breadth of legal language comprehension and generation.

1Model will be made available at https://huggingface.co/.

#### 2.2 Legal Domain Adaptation for Modern LLM

The field of legal domain adaptation has traditionally concentrated on refining models through pretraining on specialized corpora [15, 18, 21]. Yet, in the current paradigm, pretraining represents just one aspect of the solution, as LLMs often utilize techniques like instruction fine-tuning and alignment, employing algorithms such as DPO [61], PPO [66] or RLHF [55, 42].

Recent domain-adapted models, such as SaulLM or Legal-FLAN-T5 (a closed model), have tried to improve alignment with legal instructions. However, SaulLM is a smaller model, and LegalFLAN-T5, is based on an outdated architecture and does not leverage the extreme scale pretraining that contemporary models do. Moreover, it not being publicly available stymies progress vital for advancing research and applications in the legal sector.

We believe this work pioneers a holistic approach to domain adaptation by training modern LLMs specifically for the legal domain, from pretraining to instruction fine-tuning and legal preference alignment. We demonstrate that synthetic data can be effectively utilized for alignment, advancing beyond SaulLM-7B’s use solely of instruction fine-tuning. The resulting models, SaulLM-54B and SaulLM-141B, lay the groundwork for further research and development, and expand access to high-performance legal LLMs.

### 3 Data Collection and Corpus Construction

This section outlines our approach to assembling and refining a comprehensive legal text corpus tailored for training large language models in the legal domain.

#### 3.1 Pretraining Corpora

The diversity of legal systems worldwide, from common law to civil law traditions, presents unique challenges and opportunities [52, 33]. To address this, we compiled an extensive English-language corpus from various jurisdictions including the U.S., Europe, Australia, and others [3, 31], which comprises 500 billion tokens before cleaning and deduplication.

#### 3.1.1 Legal Sources

Our base corpus combines various legal datasets [51] with newly sourced public domain documents. It includes significant collections such as the FreeLaw subset and the MultiLegal Pile, augmented with extensive web-scraped content. Table 1 summarizes the composition and scale of our dataset.

#### 3.1.2 Other Sources

Replay Sources. To mitigate the risk of catastrophic forgetting during model training, we reintroduced data from earlier training distributions [46, 16, 70]. This replay strategy incorporates general data sources such as Wikipedia, StackExchange, and GitHub, and makes up approximately 2% of the total training mix. These datasets are sampled from SlimPajama [67, 19, 68].

Additionally, we included 5% of math datasets in the pretraining mix using commercially available math sources. We found this approach usefull for retaining the reasoning performance of the final model and avoiding the weaker performance observed in previous research attempts like SaulLM-7B2.

In our experiments, model annealing with high-quality, domain-relevant data significantly enhanced performance. Conversely, repetitive synthetic data from initial instruction fine-tuning harmed performance. Therefore, we used the commercial portion of the LawInstruct dataset for model annealing, which proved more effective than for instruction finetuning. We also included UltraChat [24] as generic instructions during the annealing phase.

##### 3.1.3 Data Preprocessing Our data processing pipeline closely follows [18]. In particular, we do:

- 1. Text extraction: a significant fraction of the collected data is in PDF format. We used Poppler to 2These findings also align with the high percentage of math and STEM in the pretraining mix from [54].

#### Table 1: Sources of Legal Pretraining Data Source Name Tokens (B)

FreeLaw Subset from The Pile 15 EDGAR Database 5 English MultiLegal Pile 50 English EuroParl 6 GovInfo Statutes, Opinions & Codes 11 Law Stack Exchange 0.019 Comm Open Australian Legal Corpus 0.5 EU Legislation 0.315 UK Legislation 0.190 Court Transcripts 0.350 UPSTO Database 4.7 Web Data (legal) 400 Other 30

Total 520

Instruction Sources. We found that incorporating conversational data during pretraining is advantageous, drawing inspiration from recent breakthroughs in neural machine translation [6]. Studies suggest that the enhanced translation capabilities of LLMs can be attributed to the presence of accidental parallel data within their training corpora. Accordingly, we have integrated the Super Natural Instruction [80] and FLAN collection [44] into our pretraining mix, enriching the dataset with diverse instructional content.

Data for Model Annealing. Model annealing is primarily achieved through a methodical reduction of the learning rate [56, 36], known as learning rate annealing.

extract the text.

- 2. Data cleaning: extraction from PDF files creates some artifacts like page and line numbers in the middle of sentences, as well as broken lines of text, non-normalized Unicode characters, etc.

- • Text normalization. We normalize all text using the NFKC method, available through the unicodedata Python package.
- • Rule-based filters. We created regex rules for filtering commonly undesirable but commonly recurring patterns, like page and line numbers in the middle of the text, HTML tags, etc. Following [18], we found that some of the most common 10-grams in our dataset were repeated characters and whitespace and removed them.
- • Perplexity filtering. Similarly to [18] we used a KenLM [32] model trained on a small subset of carefully cleaned legal data to filter documents with high perplexity. Concretely, we filtered any document whose normalized perplexity was larger than 1500.

- 3. Text deduplication: we used [48] to remove duplicates and near-duplicate examples from the training set. We used default parameters except for the similarity threshold, which we set to 0.5.

Finally, we packed the individual documents together to build 8192 tokens-long training examples. Documents longer than this value were chunked into several examples.

#### 3.2 Instruction Data

Instruction fine-tuning is essential for making an LLM follow instructions and optimize the performance of pretrained models across a variety of tasks [78, 81, 17, 27, 24, 79]. To this end, we employ a strategic mix of general and domain-specific (legal) instructions, aimed at enhancing the model’s ability to precisely interpret and execute commands, with a particular focus on legal scenarios.

General Instructions Our methodology for sourcing general instructions involves the integration of a diverse array of datasets, each selected to augment different aspects of the model’s capabilities across various domains [14, 89]:

- 1. General Instruction from UltraInteract [85]: UltraInteract is an extensive, high-quality dataset designed to foster complex reasoning, featuring structured instructions that include preference trees, reasoning chains, and multi-turn interaction trajectories.
- 2. General Instruction from Dolphin 3: This dataset provides additional conversational data, further broadening the model’s exposure to diverse communication styles and contexts.

Each dataset is subjected to rigorous filtering, deduplication, and curation processes, culminating in a refined compilation of approximately 1,000,000 instructions meticulously prepared for the instruction fine-tuning phase.

Legal Instruction Construction For legal instructions, we synthesize dialogues and ques-

3https://huggingface.co/datasets/cognitivecomputations/dolphin

[Figure 9]

[Figure 10]

- Figure 1: Domain adaptation method model for turning a Mixtral to a SaulLM-141B. Training involves different stages: legal domain pretraining, instruction filtering, and preference filtering.

tion/answer pairs that capture key legal concepts and document types to emulate legal analysis. In accordance with the model scale, we used Mistral-54B-Instruct for SaulLM-54B and Mistral-141B-Instruct for SaulLM-141B. The generation follows the recipe from [18] and begins with a three-turn sequence: (1) a user inquires about a legal document, (2) the assistant reformulates this inquiry by integrating metadata such as document type or issue date, and (3) the user asks for further explanation of the assistant’s reasoning. The dialogue progressively deepens, with the assistant methodically unpacking the legal reasoning in response to increasingly nuanced questions from the user.

#### 3.3 Preference Data

We enhance our models’ adaptability and precision by incorporating preference data from both general and legal-specific sources [75, 60, 50, 77]. General datasets are UltraFeedback [20] and Orca. For the legal domain, we employ synthetic scenarios crafted to simulate complex legal reasoning and generate accepted/rejected responses. The Mixtral-142B-Instruct model evaluates these responses based on factual accuracy, relevance, and logical coherence, selecting the most appropriate responses as preferred outcomes (similar to [88]).

4 Implementation Details & Evaluation Protocol

#### 4.1 Model Selection

We used Mixtral models [38], which are built on a Transformer architecture [76] enhanced with a Mixture of Experts to improve computational efficiency and adaptability for handling extensive contexts. The Mixtral-54B and Mixtral-141B architecture respectively consists of 32 (resp. 56) layers, a model dimension of 4096 (resp. 6144), and a hidden dimension of 14,336 (resp. 16384). Although it supports a context length of up to 32,768 (resp. 65536) tokens, we continue pretraining on 8,192 tokens. Extending the context length is beyond the scope of this paper. The MoE layers in Mixtral rely on 8 experts with 2 active experts selectively based on the input, efficiently utilizing computational resources and providing significant model capacity. Interestingly, Mixtral is the only model available in dual configurations (Mixtral-54B and Mixtral-141B), allowing us to evaluate the scalability of our domain adaptation approaches.

At the time of the training, Mixtral was the most powerful decoder in its class, surpassing all competitors including Llama [73, 74, 86], Yi, Qwen [7], and CroissantLLM [26] in terms of both cost-effectiveness and performance.

#### 4.2 Engineering Details

Codebase Configuration Our training framework uses PyTorch. The integration of DeepSpeed [62] (level 3) and Flash attention [23] mechanisms enhances our training efficiency and scalability. We make our models available through the Huggingface hub [82].

Compute Infrastructure The computational backbone for the continuous pretraining phase of our project consists of 384 AMD MI250 GPUs. We can reach 40% GPU utilization with our implementation. For instruction fine-tuning and preference optimization, we rely on 64 AMD MI250

GPUs. Evaluation protocols are executed on a single node of AMD MI250 GPU. Synthetic Data Generation For synthetic data generation, we used vLLM on a node of NVIDIAA100, primarily due to limited support of libraries on MI2504.

#### 4.3 Training Details

The model training process is divided into three distinct phases: continued pretraining, instruction finetuning (IFT), and preference alignment using domain-specific optimization (DPO). A full schema of the pipeline can be found in Figure 1.

Continued Pretraining For continued pretraining, we use the AdamW [39, 45, 8] optimizer with hyperparameters β1 = 0.99, β2 = 0.90, and a learning rate of 2 × 10−5. We utilize a cross-entropy loss function to optimize model predictions. The training protocol sets gradient accumulation to

- 4, with tailored batch sizes of 8 for SaulLM-54B and 4 for SaulLM-141B, optimizing both GPU utilization and training efficiency. Instruction Fine-Tuning (IFT) IFT uses the AdamW optimizer (learning rate of 1 × 10−5), reinitialized to reset training states and maintain training stability. We limit this phase to a single training epoch, as our experiments suggest this maximizes performance gains.

Preference Training Using DPO We adjust the learning rate of the AdamW optimizer to 1 × 10−6 during DPO. Our choice of DPO over IPO [13], KTO [25] or ORPO [35] was based on preliminary experiments.

- 4.4 Evaluation Protocol

LegalBench-Instruct We rely on LegalBench-Instruct [18], which refined the prompts from LegalBench [29] by eliminating distracting elements and specifying a response format to enhance precision. Like LegalBench, it evaluates LLMs across six types of legal reasoning: issue-spotting, rule-recall, rule-application, rule-conclusion, interpretation, and rhetorical understanding. Grounded in American legal frameworks but applicable globally, these categories provide a comprehensive evaluation of the models’ legal reasoning capabilities. This structured approach helps in accurately assessing and guiding the enhancement of LLMs within and beyond American legal contexts. We follow previous work and rely on balanced accuracy as the primary metric across all tasks.

Massive Multitask Language Understanding (MMLU) Previous works utilize MMLU [34], a widely-recognized benchmark, focusing on its legal-specific tasks in international law, professional law, and jurisprudence, with 120, 1500, and 110 examples respectively. These tasks are crucial for assessing our models’ understanding and application of complex legal concepts, highlighting their proficiency in nuanced legal environments.

Choice of Baseline & Model Naming For our evaluation, we aim for a direct, apples-to-apples comparison of models. It is important to note that not all competing models are open source, and detailed information on their alignment procedures and instruction fine-tuning processes is not available. This lack of transparency complicates the establishment of fully equitable baseline comparisons. In what follows, we use OpenAI’s GPT-4 as of 10 May 2024, Meta’s Llama3 (the Instruct variant) and the Instruct variants of Mixtral-54B, and Mixtral-141B. Additionally, SaulLM-54B-IFT is the IFT version built on SaulLM-54B-base and SaulLM-medium for the DPO version based on SaulLM-54B-IFT. SaulLM-large is the final version DPO and IFT based on SaulLM-141B.

5 Experimental Results

- 5.1 Global Results

Domain adaptation works across scales and MoE models. The results from SaulLM-medium and SaulLM-large confirm previous findings from [18] and confirm that domain adaptation is effective across different scales, including on MoE models. Interestingly, most of the data collected for this work comes from public sources, which were likely seen during the pretraining of the base models.

4VLLM is not supported on AMD-MI250 and HuggingFace’s text-generation-inference had a few bugs that prevented its use.

80.0

77.5

75.34%

MeanBalancedAccuracy

75.0

73.32% 73.35%

72.5

70.20%

70.0

67.35%

67.5

66.02%

63.59%

65.0

62.5

60.0

###### Mixtral-54BGPT3.5Mixtral-140BLlama3Saul-mediumGPT4 Saul-large

Model

- Figure 2: Overall Results. Comparison of SaulLM-large and SaulLM-medium with existing models.

Figure 2 presents the results of SaulLM-large and SaulLM-medium on LegalBench-Instruct, from which we make several observations:

Our domain adaptation strategy is achieving strong results. SaulLM-medium outperforms Mixtral-54B, and similar findings are observed with SaulLM-large compared to Mixtral-141B. Interestingly, domain adaptation at both the instruction tuning stage and preference alignment enables our smaller models to outperform larger ones, such as GPT-4 and LLama3-70B. These results validate our approach and demonstrate that specializing the entire pipeline (i.e., from continued pretraining to preference alignment) is a promising direction for improving performance in legal-related tasks.

Mixtral-54 IFT

Mixtral-54 Mixtral-54 IFT+DPOSaulLM-54B

IFT

SaulLM-med

Model

0.50

0.55

0.60

0.65

0.70

0.75

0.80

MeanBalancedAccuracy

0.59

0.64

0.65

0.66

0.73

- Figure 3: Global Analysis. Role of continued pretraining.

Rules

Rhetoric

Saul-54B - IFT

Mixtral-54 - IFT

Saul-med Mixtral-54

Issue

IFT + DPO

Mixtral-54

Interpretation

Conclusion

0.4 0.5 0.6 0.7 0.8 0.9

Balanced Accuracy

Figure 4: Category Analysis: Role of continue pretraining.

A Path Towards Stronger Models. The results of LLama3-70B and the scalability of our methods suggest that applying the same approach to the LLama3-70B base model could lead to even better performance than our best model, SaulLM-141B. It is worth noting that SaulLM-141B has only 44B active parameters making it appealing for efficient serving.

- Table 2: Quantifying the role of DPO. We report the percentage of tasks where the difference in performance (∆) between SaulLM-54B and SaulLM-54B-IFT is positive (resp. negative).

Category ∆ ≥ 0 ∆ ≤ 0 Conclusion 37.5% 62.5%

Interpretation 65.1% 34.9%

Rhetoric 44.4% 55.6% Rules 60.0% 40.0% Issue 100.0% -

Table 3: Quantifying the role of scaling. We report the percentage of tasks where the difference in performance (∆) between SaulLM-medium and SaulLM-large is positive (resp. negative).

Category ∆ ≥ 0 ∆ ≤ 0 Conclusion 18.2% 81.8%

Interpretation 23.7% 76.3% Rules 25.0% 75.0% Issue - 100.0%

Rhetoric - 100.0%

#### 5.2 How much does continued pretraining help for the legal domain?

Previous works on domain adaptation via continued pretraining primarily focused on instruction finetuning [16, 18, 53]. In Figure 3 and Figure 4, we report the performance of Mixtral-54B trained with the IFT mix described in subsection 3.2 (Mixtral-54-IFT) and subsequently aligned using the DPO dataset (Mixtral-54-IFT+DPO), as described in subsection 3.3. We also compare these results to the instruct version of Mixtral (Mixtral-54B), as outlined in [38].

Continuing pretraining significantly enhances model performance in the legal domain, benefiting both the IFT and DPO stages. From Figure 3, we observe that both IFT and DPO benefit from a notable improvement (approximately +7%). Interestingly, this improvement is consistent across all five categories, as shown in Figure 4.

Adding legal data to the IFT and DPO datasets improves the model’s legal capabilities. By comparing the performance of Mixtral-54-IFT+DPO and Mixtral-54, we observe that the mix used for IFT and DPO enhanced with legal data leads to stronger legal performance than that of Mixtral-54, which does not publicly describe the alignment methods used. This result aligns with findings reported in [53, 18].

#### 5.3 How Much Does Legal Preference Alignment Help?

Our findings from Figure 3 indicate that alignment significantly improves the results. In particular, DPO improvement is mostly consistent across tasks and categories. As shown in Table 2, the alignment version SaulLM-medium demonstrates significant improvements over the IFT version across most tasks, including conclusion, rhetoric, rules, and issue tasks. We observe, however, a drop in performance in some interpretation tasks. Upon closer examination, we found that this decline is often due to the model becoming more verbose, which causes the evaluation process to fail in correctly parsing the answers , i.e., this issue primarily arises from a benchmark limitation. Addressing model verbosity and the challenge of more reliable benchmarks is beyond the scope of this work, but it is a well-known problem identified in many concurrent studies [27]. Enhancing the evaluation process is one of the key improvements we plan to contribute to in the future.

1.30

SaulLM-base-140B

NormalizedLogLoss

SaulLM-base-54B

1.18

1.07

0.97

0.88

0.80

0.0 0.2 0.4 0.6 0.8 1.0

Normalized Epochs

Figure 5: Continue Pretraining Analysis. Training loss for SaulLM-141B-base and SaulLM-54B-base over normalized epochs.

2500

Weighted Mean Power per Node: 1361.97 W

MeanPowerperNode(W)

2250

2000

1750

1500

1250

1000

750

500

0 250 500 750 1000 1250 1500 1750

Job ID

Figure 6: Energy Consumption Analysis. Mean Power per Node for training jobs on the ADASTRA Supercomputer.

#### 5.4 Can We Achieve Further Improvements by Continuing Pretraining?

Training longer can potentially improve the results. Figure 5 illustrates the normalized log loss over normalized epochs for both model sizes, SaulLM-54B-base and SaulLM-141B-base. The figure presents both the raw and smoothed loss curves, which exhibit a clear downward trend throughout the training period, with no indication of saturation.

This observation suggests that continuing the pretraining process beyond the current SaulLM-base can lead to further improvements. The consistent decrease in loss implies that the models have not yet reached their full potential and that additional pretraining could enhance their performance further, which is consistent with findings from other works [58, 2, 49].

#### 5.5 How Much Does Scaling Help?

- Table 3 quantifies the impact of scaling the model and compares the performance between SaulLM-medium and SaulLM-large.

The main takeaway is that scaling generally improves overall results, but we also observe inverse scaling on some legal tasks [47, 41]. Unsurprisingly, for the majority of tasks across all categories,

increasing the model size leads to improvements, but for tasks involving conclusion, interpretation, and rules, we observe a proportion of tasks (20%) that follow inverse scaling laws.

#### 5.6 Energy Consumption

The training was conducted on Adastra, ranked 3rd in the Green500 since November 20225, as one of the world’s most efficient machines in terms of performance per watt.

Experiments for training SaulLM were performed between February 20th and May 15th. Energy consumption for each job was meticulously tracked, and we calculated and displayed the average power used per node for each job involved in this training in Figure 66. The mean power usage ranged from 600W to 2500W, reflecting the varying utilization of the GPUs. Each node contains four MI250X GPUs, which have a theoretical Thermal Design Power (TDP) of 560W. This configuration explains the maximum consumption of 2500W during high-intensity GPU usage.

Overall, the project involves over 160,000 hours of MI250 for debugging, continued pretraining, instruction finetuning and preference alignment. The total energy consumed was 65,480.4kWh. Consumption is significantly lower than the typical energy requirements for full LLM training, showing that continued pretraining is an effective strategy for specializing in new LLMs while optimizing energy efficiency.

### 6 Conclusion & Limitations

#### 6.1 Conclusion

This study released two new legal LLMs under the MIT license: SaulLM-54B and SaulLM-141B. They leverage the Mixtral architecture and continued pretraining on a large legal corpus. Our findings show significant advancements in processing and understanding complex legal documents. Through continued pretraining, instruction fine-tuning, and preference alignment using domainspecific optimization, we have demonstrated substantial improvements compared to GPT-4, Llama3 and original Mixtral models as measured on LegalBench-Instruct.

#### 6.2 Limitations

Our experiments suggest that the instruction finetuning and alignment processes utilized by Mixtral-Instruct and Llama3 are advanced and challenging to replicate. These processes often rely on proprietary datasets and significant computational resources that are not readily available in open-source frameworks. Although both SaulLM-54B and SaulLM-141B achieve stronger performances than Llama3 and Mixtral Instruct on legal benchmarks, we found that they are slightly weaker at following generic instructions.

Looking forward, we aim to continue our work on enhancing the SaulLM family, particularly focusing on integrating Llama3 and improving the alignment procedure. Our goal is to improve the alignment of these models with legal tasks, refining their ability to process and understand legal language with even greater accuracy and relevance. This future work will strive to address the current limitations by developing more robust methods for instruction finetuning and alignment that are accessible to the broader research community.

### Acknowledgments and Disclosure of Funding

This research was supported by computing grants from Adastra and Jeanzay. We extend our special thanks to Michael Robert, head of CINES, for his invaluable support and confidence in our work.

Our models have been trained on ADASTRA, with minor experimentation conducted on Jeanzay. The utilization of HPC resources was made possible through the Jeanzay grants 101838, 103256, and 103298, as well as the Adastra grants C1615122, CAD14770, and CAD15031.

- 5https://top500.org/lists/green500/2023/11/
- 6The high number of jobs (over 1800) for this project arise from (i) the technical difficulties in efficiently

using AMD GPUs, and (ii) numerous hardware failures when scaling up the number of nodes.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Alon Albalak, Yanai Elazar, Sang Michael Xie, Shayne Longpre, Nathan Lambert, Xinyi Wang, Niklas Muennighoff, Bairu Hou, Liangming Pan, Haewon Jeong, et al. A survey on data selection for language models. arXiv preprint arXiv:2402.16827, 2024.
- [3] Nikolaos Aletras, Dimitrios Tsarapatsanis, Daniel Preot¸iuc-Pietro, and Vasileios Lampos. Predicting judicial decisions of the european court of human rights: A natural language processing perspective. PeerJ computer science, 2:e93, 2016.
- [4] Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al. Santacoder: don’t reach for the stars! arXiv preprint arXiv:2301.03988, 2023.
- [5] Duarte M Alves, Nuno M Guerreiro, João Alves, José Pombal, Ricardo Rei, José GC de Souza, Pierre Colombo, and André FT Martins. Steering large language models for machine translation with finetuning and in-context learning. arXiv preprint arXiv:2310.13448, 2023.
- [6] Duarte M. Alves, José Pombal, Nuno M. Guerreiro, Pedro H. Martins, João Alves, Amin Farajian, Ben Peters, Ricardo Rei, Patrick Fernandes, Sweta Agrawal, Pierre Colombo, José G. C. de Souza, and André F. T. Martins. Tower: An open multilingual large language model for translation-related tasks. 2024.
- [7] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report, 2023.
- [8] Anas Barakat and Pascal Bianchi. Convergence and dynamical behavior of the adam algorithm for nonconvex stochastic optimization. SIAM Journal on Optimization, 31(1):244–274, 2021.
- [9] Iz Beltagy, Kyle Lo, and Arman Cohan. Scibert: A pretrained language model for scientific text. arXiv preprint arXiv:1903.10676, 2019.
- [10] Rohan Bhambhoria, Samuel Dahan, Jonathan Li, and Xiaodan Zhu. Evaluating ai for law: Bridging the gap with open-source solutions. arXiv preprint arXiv:2404.12349, 2024.
- [11] Elliot Bolton, Abhinav Venigalla, Michihiro Yasunaga, David Hall, Betty Xiong, Tony Lee, Roxana Daneshjou, Jonathan Frankle, Percy Liang, Michael Carbin, et al. Biomedlm: A 2.7 b parameter language model trained on biomedical text. arXiv preprint arXiv:2403.18421, 2024.
- [12] Elliot Bolton, Betty Xiong, Vijaytha Muralidharan, Joel Schamroth, Vivek Muralidharan, Christopher D Manning, and Roxana Daneshjou. Assessing the potential of mid-sized language models for clinical qa. arXiv preprint arXiv:2404.15894, 2024.
- [13] Daniele Calandriello, Daniel Guo, Remi Munos, Mark Rowland, Yunhao Tang, Bernardo Avila Pires, Pierre Harvey Richemond, Charline Le Lan, Michal Valko, Tianqi Liu, et al. Human alignment of large language models through online preference optimisation. arXiv preprint arXiv:2403.08635, 2024.
- [14] Yihan Cao, Yanbin Kang, and Lichao Sun. Instruction mining: High-quality instruction data selection for large language models. arXiv preprint arXiv:2307.06290, 2023.
- [15] Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. Legal-bert: The muppets straight out of law school. arXiv preprint arXiv:2010.02559, 2020.

- [16] Zeming Chen, Alejandro Hernández Cano, Angelika Romanou, Antoine Bonnet, Kyle Matoba, Francesco Salvi, Matteo Pagliardini, Simin Fan, Andreas Köpf, Amirkeivan Mohtashami, et al. Meditron-70b: Scaling medical pretraining for large language models. arXiv preprint arXiv:2311.16079, 2023.
- [17] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022.
- [18] Pierre Colombo, Telmo Pessoa Pires, Malik Boudiaf, Dominic Culver, Rui Melo, Caio Corro, Andre F. T. Martins, Fabrizio Esposito, Vera Lúcia Raposo, Sofia Morgado, and Michael Desa. Saullm-7b: A pioneering large language model for law. 2024.
- [19] Together Computer. Redpajama: an open dataset for training large language models, 2023.
- [20] Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models with high-quality feedback. arXiv preprint arXiv:2310.01377, 2023.
- [21] Jiaxi Cui, Zongjian Li, Yang Yan, Bohua Chen, and Li Yuan. Chatlaw: Open-source legal large language model with integrated external knowledge bases. arXiv preprint arXiv:2306.16092, 2023.
- [22] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. arXiv preprint arXiv:2401.06066, 2024.
- [23] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.
- [24] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233, 2023.
- [25] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.
- [26] Manuel Faysse, Patrick Fernandes, Nuno Guerreiro, António Loison, Duarte Alves, Caio Corro, Nicolas Boizard, João Alves, Ricardo Rei, Pedro Martins, et al. Croissantllm: A truly bilingual french-english language model. arXiv preprint arXiv:2402.00786, 2024.
- [27] Manuel Faysse, Gautier Viaud, Céline Hudelot, and Pierre Colombo. Revisiting instruction fine-tuned model evaluation to guide industrial applications. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2023.
- [28] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The pile: An 800gb dataset of diverse text for language modeling. 2020.
- [29] Neel Guha, Daniel E Ho, Julian Nyarko, and Christopher Ré. Legalbench: Prototyping a collaborative benchmark for legal reasoning. arXiv preprint arXiv:2209.06120, 2022.
- [30] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming–the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024.
- [31] Asier Gutiérrez-Fandiño, Jordi Armengol-Estapé, Aitor Gonzalez-Agirre, and Marta Villegas. Spanish legalese language model and corpora. arXiv preprint arXiv:2110.12201, 2021.
- [32] Kenneth Heafield. KenLM: Faster and smaller language model queries. In Chris Callison-Burch, Philipp Koehn, Christof Monz, and Omar F. Zaidan, editors, Proceedings of the Sixth Workshop on Statistical Machine Translation, pages 187–197, Edinburgh, Scotland, July 2011. Association for Computational Linguistics.

- [33] Peter Henderson, Mark Krass, Lucia Zheng, Neel Guha, Christopher D Manning, Dan Jurafsky, and Daniel Ho. Pile of law: Learning responsible data filtering from the law and a 256gb opensource legal dataset. Advances in Neural Information Processing Systems, 35:29217–29234, 2022.
- [34] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.
- [35] Jiwoo Hong, Noah Lee, and James Thorne. Reference-free monolithic preference optimization with odds ratio. arXiv preprint arXiv:2403.07691, 2024.
- [36] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.
- [37] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023.
- [38] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts, 2024.
- [39] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [40] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161, 2023.
- [41] Xianhang Li, Zeyu Wang, and Cihang Xie. An inverse scaling law for clip training. Advances in Neural Information Processing Systems, 36, 2024.
- [42] Zihao Li, Zhuoran Yang, and Mengdi Wang. Reinforcement learning with human feedback: Learning dynamic choices via pessimism. arXiv preprint arXiv:2305.18438, 2023.
- [43] Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Junwu Zhang, Munan Ning, and Li Yuan. Moe-llava: Mixture of experts for large vision-language models. arXiv preprint arXiv:2401.15947, 2024.
- [44] Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. arXiv preprint arXiv:2301.13688, 2023.
- [45] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [46] Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. volume 24 of Psychology of Learning and Motivation, pages 109–165. Academic Press, 1989.
- [47] Ian R McKenzie, Alexander Lyzhov, Michael Pieler, Alicia Parrish, Aaron Mueller, Ameya Prabhu, Euan McLean, Aaron Kirtland, Alexis Ross, Alisa Liu, et al. Inverse scaling: When bigger isn’t better. arXiv preprint arXiv:2306.09479, 2023.
- [48] Chenghao Mou, Chris Ha, Kenneth Enevoldsen, and Peiyuan Liu. Chenghaomou/text-dedup: Reference snapshot, September 2023.

- [49] Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel. Scaling data-constrained language models. Advances in Neural Information Processing Systems, 36, 2024.
- [50] Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Andrea Michi, et al. Nash learning from human feedback. arXiv preprint arXiv:2312.00886, 2023.
- [51] Joel Niklaus and Daniele Giofré. Budgetlongformer: Can we cheaply pretrain a sota legal language model from scratch? arXiv preprint arXiv:2211.17135, 2022.
- [52] Joel Niklaus, Veton Matoshi, Matthias Stürmer, Ilias Chalkidis, and Daniel E. Ho. Multilegalpile: A 689gb multilingual legal corpus. 2023.
- [53] Joel Niklaus, Lucia Zheng, Arya D McCarthy, Christopher Hahn, Brian M Rosen, Peter Henderson, Daniel E Ho, Garrett Honke, Percy Liang, and Christopher Manning. Flawn-t5: An empirical examination of effective instruction-tuning data mixtures for legal reasoning. arXiv preprint arXiv:2404.02127, 2024.
- [54] Aitor Ormazabal, Che Zheng, Cyprien de Masson d’Autume, Dani Yogatama, Deyu Fu, Donovan Ong, Eric Chen, Eugenie Lamprecht, Hai Pham, Isaac Ong, et al. Reka core, flash, and edge: A series of powerful multimodal language models. arXiv preprint arXiv:2404.12387, 2024.
- [55] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [56] Jupinder Parmar, Shrimai Prabhumoye, Joseph Jennings, Mostofa Patwary, Sandeep Subramanian, Dan Su, Chen Zhu, Deepak Narayanan, Aastha Jhunjhunwala, Ayush Dattagupta, et al. Nemotron-4 15b technical report. arXiv preprint arXiv:2402.16819, 2024.
- [57] Shounak Paul, Arpan Mandal, Pawan Goyal, and Saptarshi Ghosh. Pre-trained language models for the legal domain: A case study on indian law. In Proceedings of 19th International Conference on Artificial Intelligence and Law - ICAIL 2023, 2023.
- [58] Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: Outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116, 2023.
- [59] Maciej Pióro, Kamil Ciebiera, Krystian Król, Jan Ludziejewski, and Sebastian Jaszczur. Moe-mamba: Efficient selective state space models with mixture of experts. arXiv preprint arXiv:2401.04081, 2024.
- [60] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023.
- [61] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.
- [62] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3505–3506, 2020.
- [63] Richard M Re and Alicia Solow-Niederman. Developing artificially intelligent justice. Stan. Tech. L. Rev., 22:242, 2019.

- [64] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.
- [65] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022.
- [66] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [67] Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Joel Hestness, Natalia Vassilieva, Daria Soboleva, and Eric Xing. Slimpajama-dc: Understanding data combinations for llm training. arXiv preprint arXiv:2309.10818, 2023.
- [68] Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. Slimpajama: A 627b token cleaned and deduplicated version of redpajama, 2023.
- [69] Tania Sourdin, Bin Li, and Donna Marie McNamara. Court innovations and access to justice in times of crisis. Health policy and technology, 9(4):447–453, 2020.
- [70] Jingyuan Sun, Shaonan Wang, Jiajun Zhang, and Chengqing Zong. Distill and replay for continual language learning. In Proceedings of the 28th international conference on computational linguistics, pages 3569–3579, 2020.
- [71] Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic. Galactica: A large language model for science. arXiv preprint arXiv:2211.09085, 2022.
- [72] Robert Tinn, Hao Cheng, Yu Gu, Naoto Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng Gao, and Hoifung Poon. Fine-tuning large neural language models for biomedical natural language processing. Patterns, 4(4), 2023.
- [73] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [74] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.
- [75] Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, et al. Zephyr: Direct distillation of lm alignment. arXiv preprint arXiv:2310.16944, 2023.
- [76] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [77] Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, and Shengyi Huang. Trl: Transformer reinforcement learning. https://github. com/huggingface/trl, 2020.

- [78] Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A Smith, Iz Beltagy, et al. How far can camels go? exploring the state of instruction tuning on open resources. arXiv preprint arXiv:2306.04751, 2023.
- [79] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions, 2023.
- [80] Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. arXiv preprint arXiv:2204.07705, 2022.
- [81] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.
- [82] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Huggingface’s transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771, 2019.
- [83] Yunzhi Yao, Shaohan Huang, Wenhui Wang, Li Dong, and Furu Wei. Adapt-and-distill: Developing small, fast and effective pretrained language models for domains. arXiv preprint arXiv:2106.13474, 2021.
- [84] Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Dong Wang, Huchuan Lu, and You He. Boosting continual learning of vision-language models via mixture-of-experts adapters. arXiv preprint arXiv:2403.11549, 2024.
- [85] Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, Zhenghao Liu, Bowen Zhou, Hao Peng, Zhiyuan Liu, and Maosong Sun. Advancing llm reasoning generalists with preference trees, 2024.
- [86] Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model. arXiv preprint arXiv:2401.02385, 2024.
- [87] Rongyu Zhang, Yulin Luo, Jiaming Liu, Huanrui Yang, Zhen Dong, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, Yuan Du, et al. Efficient deweahter mixture-of-experts with uncertainty-aware feature-wise linear modulation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 16812–16820, 2024.
- [88] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36, 2024.
- [89] Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. arXiv preprint arXiv:2305.11206, 2023.
- [90] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. Mixture-of-experts with expert choice routing. Advances in Neural Information Processing Systems, 35:7103–7114, 2022.

