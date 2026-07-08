# arXiv:2503.00955v3[cs.CL]5Oct2025

## SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking

### Dien X. Tran1,*, Nam V. Nguyen1,*, Thanh T. Tran1, Anh T. Hoang1, Tai V. Duong1, Di T. Le1, Phuc-Lu Le2 1Industrial University of Ho Chi Minh City, Vietnam 2University of Science, VNU-HCM, Vietnam

Correspondence: lplu@fit.hcmus.edu.vn

### Abstract

Recent advances in LLMs have accelerated both information generation and misinformation, especially in low-resource languages like Vietnamese, motivating robust fact-checking systems. Existing methods struggle with semantic ambiguity, homonyms, and complex linguistic structures, often trading accuracy for efficiency. We introduce SemViQA, a novel Vietnamese fact-checking framework integrating Semantic-based Evidence Retrieval (SER) and Two-step Verdict Classification (TVC). Our approach balances precision and speed, achieving state-of-the-art results with 78.97% strict accuracy on ISE-DSC01 and 80.82% on ViWikiFC, securing 1st place in the UIT Data Science Challenge. Additionally, SemViQA Faster improves inference speed 7× while maintaining competitive accuracy. SemViQA sets a new benchmark for Vietnamese fact verification, advancing the fight against misinformation. The source code is available at: https://github. com/DAVID-NGUYEN-S16/SemViQA.

### 1 Introduction

The rapid advancement of large language models (LLMs), such as OpenAI’s ChatGPT, Google Gemini (Team et al., 2024), Llama3.1 (Touvron et al., 2023), Qwen2.5 (Qwen et al., 2025), DeepSeek V3, (DeepSeek-AI et al., 2024), Phi3.5 (Abdin et al., 2024) has significantly improved information retrieval and processing across various domains. However, a major challenge with these systems is their tendency to generate factually incorrect or hallucinated content seemingly plausible information that lacks factual grounding (Soleimani et al., 2020). This issue is particularly critical in domains requiring high accuracy, such as healthcare, law, and journalism, where misinformation can have serious consequences. Consequently, developing reliable fact-checking systems capable of retrieving

*Equal contribution. Preprint

Context

Above all, the most dangerous weapon that North Korea could deploy in combat is its nuclear arsenal. [...] In the event of a U.S. attack, North Korea may launch nuclear-armed missiles in retaliation. [...] North Korea has stated that it will only agree to dismantle its nuclear weapons program if the United States signs a peace treaty first; conversely, the U.S. demands that North Korea abandon its nuclear weapons before any peace agreement is signed.

[Figure 1]

Claim

Nuclear-armed missiles are prohibited from being deployed in North Korea, even in hypothetical scenarios involving a U.S. military strike.

SemViQA

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Semantic-based Evidence Retrieval

Two-step Verdict Classification

Evidence Verdict

In the event of a U.S. attack, North Korea may launch nuclear-armed missiles in retaliation.

[Figure 7]

###### REFUTED

Figure 1: Overview of a Sample Information FactChecking Task

and evaluating evidence from real-world sources has become an urgent need in Natural Language Processing (NLP).

Although fact verification has been widely studied in high-resource languages like English, applying these methods to low-resource languages such as Vietnamese remains a significant challenge. Transformer-based models, including BERT (Devlin et al., 2019) and RoBERTa (Liu et al., 2019), have demonstrated strong performance but their adaptation to Vietnamese is still limited. ViNSV (Tran et al., 2024b) employs BM25 and SBERT (Reimers and Gurevych, 2019a) for evidence retrieval but suffers from SBERT’s 256token input constraint, making it ineffective for complex, long-context claims. Graph-based reasoning methods (Zhong et al., 2020) offer promising semantic inference but are often computationally expensive. Traditional retrieval methods like TFIDF and BM25, while efficient, rely heavily on exact keyword matching, limiting their ability to capture nuanced semantics. Recent large language model (LLM) approaches (Huo et al., 2023; Schimanski et al., 2024) show potential but typically

require substantial computational resources, creating a trade-off between speed and accuracy.

To address these challenges, we propose SemViQA, a Vietnamese fact-checking framework that balances semantic accuracy and computational efficiency. As shown in Figure 1, SemViQA comprises three key components:

- 1. Semantic-based Evidence Retrieval (SER): Includes a preprocessing step that efficiently handles long-token contexts by splitting them into manageable subcontexts (e.g., 400 tokens). It combines fast TF-IDF retrieval with selective Question Answering Token Classification (QATC) to strike a balance between speed and semantic accuracy.
- 2. Two-step Verdict Classification (TVC): Employs a hierarchical classification strategy with both three-class and binary classification stages to enhance robustness and improve performance on challenging claim verification tasks.

SemViQA achieves 78.97% strict accuracy on ISE-DSC01 1 and 80.82% on ViWikiFC (Le et al., 2024), outperforming existing baselines (see Table 1). These results validate SemViQA’s potential to enhance Vietnamese fact verification, supporting misinformation mitigation and improved transparency.

The rest of this paper is organized as follows: Section 2 reviews related work, Section 3 presents the methodology, Section 4 reports experimental results, and Section 5 concludes the paper with future directions.

### 2 Related Works

Advances in Natural Language Processing (NLP) have driven rapid progress in fact verification and evidence extraction. Early BiLSTM-based models such as the Neural Semantic Matching Network (NSMN) (Nie et al., 2018)—augmented with WordNet features improved accuracy but struggled with complex sentence relations due to sequential limitations (Graves and Schmidhuber, 2005). Transformer models, notably BERT (Devlin et al., 2019), introduced bidirectional contextual encoding and achieved state-of-the-art results on FEVER (Soleimani et al., 2020; Zhou et al., 2019; Malon, 2018; Aly et al., 2021; Lin

1https://codalab.lisn.upsaclay.fr/competitions/ 15497#results

et al., 2024; Yuan and Vlachos, 2024; DeHaven and Scott, 2023), yet their input-length cap hinders long-document fact-checking. Graph-based reasoning further enhances multi-hop verification GCN variants (Zhong et al., 2020; Thorne et al.,

- 2018) and AdMIRaL’s logic-driven retrieval (Aly and Vlachos, 2022) boost evidence sufficiency but at considerable computational cost. Vietnamese research is sparse: ViNSV (Tran et al., 2024b) pairs BM25 with SBERT (Reimers and Gurevych,
- 2019a) yet falters on complex reasoning due to static embeddings.

TF-IDF remains the industry standard for document retrieval due to its speed, simplicity, interpretability, and ability to handle long contexts effectively, making it suitable for rapid and scalable retrieval (Reddy et al., 2018; Qaiser and Ali, 2018; Li, 2021; Azevedo et al., 2022). However, its reliance on surface-level keyword matching limits its capability to handle paraphrases, contextual nuances, and multi-hop reasoning, reducing recall accuracy on complex queries. Ensemble learning (Hannichenko et al., 2023; Wang et al., 2021; Liu et al., 2024; Ganaie et al., 2022) mitigates individual model weaknesses by aggregating diverse architectures and training signals, yielding robust gains. Building on these insights, SemViQA fuses fast TF-IDF retrieval, semantic reasoning via QATC, and hierarchical classification, delivering high accuracy, low latency, and practical scalability for Vietnamese fact verification.

### 3 SemViQA - Semantic Vietnamese Question Answering

We formulate Vietnamese fact verification as a multi-output classification task, where the input is a pair (C,X), with C being a claim and X its corresponding context paragraph or document. The objective is to (i) identify the most relevant evidence sentence(s) from X and (ii) predict the veracity label of the claim as one of three categories: Supported, Refuted, or Not Enough Information (NEI). To address challenges such as long input sequences and semantic ambiguity, we propose SemViQA, a three-stage framework consisting of data preprocessing, semantic-based evidence retrieval, and two-step verdict classification. An overview of the architecture is shown in Figure 2, with detailed descriptions provided in the following subsections.

Data Input

[Figure 8]

Context

Claim

###### SemViQA

SER TVC

No

SUPPORTED REFUTED NEI

No

###### QATC

Yes

Confidence Threshold

Binary Classification

Yes

| | |
|---|---|
| | |

- Evi 1

Candidate evidence

0.856 0.748 0.525 0.254 0.125

- Evi 2
- Evi 3
- Evi 4
- Evi 5

| | | |
|---|---|---|
| | | |

No

Evidence

Yes

Three-Class Classification

TF-IDF

Figure 2: SemViQA: A Two-Stage Method for Semantic-based Evidence Retrieval (SER) and Two-step Verdict Classification (TVC), where P2 and P3 represent the probabilities of the two-class and three-class classifications, respectively, and yˆ2 and yˆ3 denote their corresponding predictions.

- 3.1 Evidence Extraction via Question Answering with Token Classification

- 3.1.1 Data Processing

Context

evidence

Split context into sentences

Combining Segments into New Sub-Contexts

- Sub 1

- Sub 2

Sub n

Max 400 Tokens

Figure 3: Long context processing solution.

To effectively support evidence retrieval and claim classification, we apply distinct preprocessing strategies to the context based on the specific requirements of each downstream task (see Appendix A). A key challenge arises from the considerable length of many context passages, frequently exceeding the token limits of Vietnamese BERTbased models. Figure 3 illustrates our approach for handling long input contexts. First, the context is segmented into individual sentences. Next, sentences are sequentially aggregated into subcontexts until reaching approximately 400 tokens. Each completed subcontext is then processed by the QATC model to identify potential evidence. The next subcontext begins from the subsequent sentence, and this process continues until all sentences are processed. However, processing subcontexts sequentially can be time-consuming. Therefore, we developed SemViQA Faster, which batches and

processes subcontexts in parallel, significantly accelerating the retrieval process. 3.1.2 Question Answering with Token

#### Classification (QATC)

Traditional Question Answering models typically predict the start and end positions of an answer span. In our framework, we enhance this approach by introducing a token-level classification objective, enabling the model to focus explicitly on tokens within evidence sentences in the context. This dual formulation provides improved supervision for evidence extraction. Drawing inspiration from rationale tagging (Ju et al., 2019), we treat token labeling as a binary classification task: tokens within evidence sentences receive a label of 1, and all other tokens receive a label of 0. In cases marked NEI (Not Enough Information), every token is labeled as 0. We employ a feed-forward classification layer on the token representations:

pt = σ (W2 · ReLU(W1ht)), (1)

where ht is the contextual representation of token t, W1,W2 are learnable weights in the neural network and σ(·) denotes the sigmoid function. The loss for this task is the Binary Cross-Entropy (BCE) loss:

1 T

LRT = −

T

BCE(yt,pt), (2)

t=1

3.2 Semantic-based Evidence Retrieval (SER) Accurate claim verification requires reliable evidence. To improve both efficiency and robustness,

we adopt a two-stage evidence retrieval strategy combining TF-IDF with a QATC.

#### Stage 1: TF-IDF-based Retrieval. We segment

the context X into smaller passages and pair each with the claim C. Preprocessing includes noise removal and tokenization using ViTokenizer2. TFIDF is effective for simple claims particularly refuted ones but struggles with semantically complex cases due to its reliance on keyword overlap. To enrich short segments (i.e., those with fewer than 60% of C’s tokens), we merge them with preceding segments to improve evidence completeness. Retrieved segments are then ranked, and a confidence threshold is applied to identify easy cases (handled by TF-IDF) and hard cases (passed to QATC).

#### Stage 2: QATC-based Refinement. For com-

plex cases, QATC is applied to segmented subcontexts rather than the full context due to the input length limitation of BERT models. The detailed processing approach is described in Section 3.1.1. At this time, we consider three scenarios: (1) If multiple subcontexts yield conflicting answers, we collect all predicted spans and re-rank them using TF-IDF. (2) If a single evidence span consistently appears, it is selected directly. (3) If no evidence is found, fallback to TF-IDF is used.

This hybrid approach balances speed and semantic accuracy, improving evidence selection for downstream verdict classification. Examples are provided in Appendix E.

- 3.3 Two-step Verdict Classification (TVC) We adopt a two-stage classification framework to enhance claim verification robustness and mitigate label imbalance, especially the overrepresentation of NEI.

- Stage 1: Three-Class Classification. Given a claim-evidence pair (C,E), a BERT-based model

f3-class predicts a probability distribution over three labels: Supported, Refuted, and Not Enough Information (NEI):

P3 = f3-class(C,E), yˆ3 = arg max

k

P3. (3)

This step is optimized using Cross-Entropy Loss.

- Stage 2: Binary Classification. If yˆ3 ̸= NEI, we apply a refined binary classifier f2-class to distinguish between Supported and Refuted:

##### P2. (4)

P2 = f2-class(C,E), yˆ2 = arg max

k

2https://github.com/trungtv/pyvi

This model uses Focal Loss (Lin et al., 2018) to address class imbalance.

Final Prediction Rule. The final verdict yˆ ∈ {1,2,3} where 1 = NEI, 2 = Supported, 3 = Refuted is determined by comparing the confidence scores from both classifiers. Here, yˆ represents the index of the predicted label, where each index corresponds to a specific class description.

 

yˆ3, if yˆ3 = 1, yˆ3, if v3max > v2max, yˆ2, otherwise,

(5)

yˆ =



where v3max = max(P3), v2max = max(P2) represents the highest probability.

This hybrid strategy allows the three-class model to handle general cases, especially detecting NEI early, while the binary model specializes in distinguishing difficult SUP/REF cases.

#### 3.4 SemViQA Pipeline System

We now describe the full SemViQA pipeline, as illustrated in Figure 2. First, we prepare input for TF-IDF by splitting the context paragraph X into sentences, then concatenating each sentence with the claim C. We calculate the matching score for each sentence and select the one with the highest probability. If this score exceeds the threshold t, we directly use this sentence as the evidence. If the score is below t, we proceed to prepare input for the QATC model. We segment X into subcontexts (as detailed in Section 3.1.1). Each subcontext is sequentially processed by the QATC model. If QATC identifies zero or multiple candidate evidence spans, we collect all predicted spans and re-rank them using TF-IDF. If QATC finds exactly one candidate evidence span, we confidently select it as the final evidence. Finally, we move to the twostep verdict classification (TVC) stage. We prepare input for TVC by concatenating the claim C with the final evidence. We first apply the three-class model. If it predicts NEI, the process ends. Otherwise, we use an ensemble method to combine the weights from both the three-class and binary models to make the final prediction. All coefficients used in our system are provided in the ablation study, as shown in Appendix C.

4 Experiments

#### 4.1 Dataset

We use two Vietnamese fact verification datasets: ISE-DSC01 from the UIT Challenge 2023 and

ViWikiFC (Le et al., 2024), which contains over 20,000 Wikipedia-based claims, including annotated evidence for the “nei” label. Dataset details are provided in Appendix A.1.

#### 4.2 Experimental Setup

We conducted extensive experiments on NVIDIA A100 GPUs, fine-tuning key hyperparameters while keeping consistent settings across runs. The final configuration, selected via rigorous validation, improved both accuracy and strict accuracy on ISE-DSC01 and ViWikiFC. Full details are provided in Appendix D. For fair evaluation, all methods were tested on a Kaggle instance with an NVIDIA T4 GPU.

The large language model was fine-tuned in a distributed A100 setup using a structured promptbased reformulation. Raw data were converted into prompt format to align with LLM training objectives and maximize task-specific performance. Training setup, prompt design, and preprocessing are also detailed in Appendix D.

- 4.3 Main Results The results in Table 1 demonstrate that SemViQA outperforms previous methods in Vietnamese factchecking tasks. Specifically, our model achieves the highest Strict Accuracy, reaching 80.82% on ViWikiFC and 78.97% on ISE-DSC01, establishing a new benchmark for automated fact-checking systems in Vietnamese language.

#### 4.3.1 Performance Comparison

a) Handling Long Token Sequences in FactChecking A major limitation of conventional Question Answering (QA) models in fact verification is their inability to process long-context claims due to the 512-token input limit of transformer-based models such as ViMRClarge3, InfoXLMlarge (Chi et al., 2021), XLMRlarge (Conneau et al., 2020), and ErnieMlarge (Ouyang et al., 2021). Real-world datasets like ISE-DSC01 often contain contexts exceeding 4800 tokens, severely degrading QA-based performance by limiting access to full evidence. To overcome this, SemViQA employs an efficient retrieval-based strategy (see Section 3.1.1) that handles long-token sequences effectively. On ISEDSC01, SemViQA outperforms traditional QA models by fully leveraging extended contexts, confirming that the long-token constraint is a criti-

3https://huggingface.co/nguyenvulebinh/ vi-mrc-large

Figure 4: Comparison of methods in terms of peak performance and total inference time across datasets. Each retrieval approach is evaluated by its best score, while overall efficiency is reflected through cumulative inference time. See Table 1 for details.

cal bottleneck. Conversely, on ViWikiFC, where contexts average around 512 tokens, QA models perform competitively. Yet, even in this setting, integrating our Semantic-based Evidence Retrieval (SER) yields a 1.86% improvement in evidence retrieval accuracy, demonstrating the versatility and efficiency of our approach. These findings emphasize that long-token limitations significantly hinder fact verification, and SemViQA successfully mitigates this issue while enhancing QA models across varying dataset conditions.

b) Performance and Inference Time Optimization SemViQA significantly reduces inference time while maintaining high accuracy, making it highly practical for real-world applications. Key highlights include:

- • On ISE-DSC01, SemViQA averages 5200s per run, over 6 times faster than large LLM-based models like Qwen2.5-3BInstruct (Qwen et al., 2024), which require over 31,000s.
- • Compared to ViMRClarge (9800s), SemViQA halves inference time while achieving superior Strict Accuracy and Veracity Classification Accuracy.
- • Although BM25 and SBERT (Reimers and Gurevych, 2019b) are faster, they struggle with complex, multi-step reasoning where SemViQA maintains a strong balance between speed and accuracy.

SemViQA Faster: We further introduce SemViQA Faster, which accelerates inference by

|Method<br><br>|ViWikiFC<br><br>|ISE-DSC01<br><br>|Avg Strict Acc|
|---|---|---|---|
|ER VC<br><br>|Strict Acc VC Acc ER Acc Time (s)<br><br>|Strict Acc VC Acc ER Acc Time (s)| |

Traditional Baselines

|TF-IDF<br><br>InfoXLMlarge XLM-Rlarge Ernie-Mlarge|75.56 82.21 90.15 131<br><br>76.47 82.78 90.15 134<br><br><br>75.56 81.83 90.15 144<br><br>|73.59 78.08 76.61 378 75.61 80.50 78.58 366 78.19 81.69 80.65 403<br><br>|74.58 76.04 76.88|
|---|---|---|---|
|BM25<br><br>InfoXLMlarge XLM-Rlarge Ernie-Mlarge<br><br>|70.44 79.01 83.50 130 70.97 78.91 83.50 132 70.21 78.29 83.50 141|72.09 77.37 75.04 320<br><br>73.94 79.37 76.95 333<br><br><br>76.58 80.76 79.02 381|71.27 72.46 73.40<br><br>|
|SBert<br><br>InfoXLMlarge XLM-Rlarge Ernie-Mlarge<br><br>|74.99 81.59 89.72 195<br><br>75.80 82.35 89.72 194 75.13 81.44 89.72 203<br><br><br>|71.20 76.59 74.15 915<br>72.85 78.78 75.89 835 75.46 79.89 77.91 920<br>|73.10 74.33 75.30<br><br>|

QA-based Approaches

|ViMRClarge<br><br>InfoXLMlarge XLM-Rlarge Ernie-Mlarge|77.28 81.97 92.49 3778<br><br>78.29 82.83 92.49 3824<br><br><br>77.38 81.92 92.49 3785|54.36 64.14 56.84 9798 53.98 66.70 57.77 9809 56.62 62.19 58.91 9833<br><br>|65.82 66.14 67.00<br><br>|
|---|---|---|---|
|InfoXLMlarge<br><br>InfoXLMlarge XLM-Rlarge Ernie-Mlarge|78.14 82.07 93.45 4092<br><br>79.20 83.07 93.45 4096<br><br><br>78.24 82.21 93.45 4102<br><br>|53.50 63.83 56.17 10057 53.32 66.70 57.25 10066 56.34 62.36 58.69 10078<br><br>|65.82 66.26 67.29|

LLMs

|Qwen2.5-1.5-Instruct Qwen2.5-3B-Instruct<br><br>|51.03 65.18 78.96 7665 44.38 62.31 71.35 12123|59.23 66.68 65.51 19780<br><br>60.87 66.92 66.10 31284<br>|55.13 52.63<br><br>|
|---|---|---|---|
|Qwen2.5-1.5-Instruct<br><br>InfoXLMlarge XLM-Rlarge Ernie-Mlarge<br><br>|66.14 76.47 78.96 7788<br><br>67.67 78.10 78.96 7789 66.52 76.52 78.96 7794<br><br><br>|64.40 68.37 66.49 19970<br><br>64.66 69.63 66.72 19976<br><br>65.70 68.37 67.33 20003<br><br><br>|65.27 66.17 66.11|
|Qwen2.5-3B-Instruct<br><br>InfoXLMlarge XLM-Rlarge Ernie-Mlarge<br><br>|59.88 72.50 71.35 12246<br><br>60.74 73.08 71.35 12246 60.02 72.21 71.35 12251<br><br><br>|65.72 69.66 67.51 31477<br><br>66.12 70.44 67.83 31483<br><br>67.48 70.77 68.75 31512<br><br><br>|62.80 63.43 63.80|

Ours: SER Faster + TVC ViMRClarge

###### 79.44 82.93 94.60 410 78.32 81.91 80.26 995 78.88 InfoXLMlarge 79.77 83.07 95.03 487 78.37 81.91 80.32 925 79.07 Ours: Full SER + TVC

Ernie-Mlarge

|ViMRClarge<br><br>InfoXLMlarge XLM-Rlarge Ernie-Mlarge<br><br>|80.25 83.84 94.69 2731 80.34 83.64 94.69 2733 79.53 82.97 94.69 2733<br><br>|75.13 79.54 76.87 5191<br>76.71 81.65 78.91 5219 78.97 82.54 80.91 5225<br>|77.69 78.53 79.25<br><br>|
|---|---|---|---|
|InfoXLMlarge<br><br>InfoXLMlarge XLM-Rlarge Ernie-Mlarge<br><br>|80.68 83.98 95.31 3860 80.82 83.88 95.31 3843 80.06 83.17 95.31 3891<br><br>|75.13 79.60 76.87 5175<br>76.74 81.71 78.95 5200 78.97 82.49 80.91 5297<br>|77.91 78.78 79.52<br><br>|

- Table 1: Performance comparison on the ViWikiFC test set and the ISE-DSC01 private-test dataset. The results highlight differences among models based on several criteria: Strict Accuracy (Strict Acc), Veracity Classification Accuracy (VC Acc), and Evidence Retrieval Accuracy (ER Acc). Time represents the total inference time required to generate the complete results.

up to 7× through batch processing of subcontexts (see Section 3.1.1). As shown in Figure 4, this variant achieves inference speeds comparable to traditional methods while retaining competitive accuracy. The minor performance trade-off is acceptable given the substantial time savings, making SemViQA Faster ideal for scalable, real-world factchecking systems.

4.3.2 Comparison with other results in the competition

Methods Strict Acc VC Acc ER Acc SemViQA 78.97 82.54 80.91 DS@UIT Dynasty4 78.05 84.76 80.13 URA_FNU5 77.87 83.71 79.96 ViNSV (Tran et al., 2024b) 76.33 81.67 78.11 (Tran et al., 2024a) 75.11 82.30 76.82

- Table 2: Private leaderboard comparison of top systems in the ISE-DSC01 competition.

mation processing and verification. This achievement highlights SemViQA’s capability to deliver significantly more accurate and reliable results.

### 5 Conclusion and Future Works

We introduced SemViQA, a Vietnamese factchecking framework that integrates Semantic-based Evidence Retrieval (SER) and Two-step Verdict Classification (TVC) to enhance claim verification. Our approach outperforms existing methods, including LLMs, TF-IDF, BM25, SBERT, and QAbased models, particularly in handling long-token sequences and complex reasoning tasks. Extensive experiments demonstrated SemViQA’s stateof-the-art performance on ISE-DSC01 and ViWikiFC. Additionally, the SemViQA Faster variant accelerates inference by up to 7×, improving its practicality for real-world applications. By addressing key challenges such as semantic ambiguity and multi-step reasoning, SemViQA lays the groundwork for advancing Vietnamese NLP, with potential applications in misinformation detection and low-resource language fact-checking.

The results presented in Table 2 indicate that our SemViQA approach outperforms other competing teams, achieving the highest Strict Accuracy and demonstrating exceptional effectiveness in infor-

### Limitations

While SemViQA demonstrates strong performance in Vietnamese fact verification, several limitations remain. First, our reliance on TF-IDF for initial evidence retrieval, while efficient, limits the model’s ability to capture deep semantic relationships and retrieve implicit evidence. To mitigate this, we employ a threshold-based mechanism to identify hard samples and process them with a more advanced retrieval model. However, this approach relies on manually defined thresholds, which may not generalize well across different datasets, underscoring the need for adaptive and data-driven retrieval strategies in future work. Second, our Two-step Verdict Classification (TVC) framework improves claim verification accuracy but requires multiple classification stages, increasing inference time compared to single-step approaches. This additional computational cost is particularly significant in three-class classification tasks, where optimizing model efficiency without compromising accuracy remains a key challenge. Future work should focus on refining retrieval mechanisms and classification strategies to enhance efficiency and robustness, ensuring broader applicability of SemViQA in real-world fact verification scenarios.

### References

Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. 2024. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219.

Rami Aly, Zhijiang Guo, Michael Sejr Schlichtkrull, James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Oana Cocarascu, and Arpit Mittal. 2021. The fact extraction and VERification over unstructured and structured information (FEVEROUS) shared task. In Proceedings of the Fourth Workshop on Fact Extraction and VERification (FEVER), pages 1–13, Dominican Republic. Association for Computational Linguistics.

Rami Aly and Andreas Vlachos. 2022. Natural logicguided autoregressive multi-hop document retrieval for fact verification. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6123–6135, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Pedro Azevedo, Gil Rocha, Diego Esteves, and Henrique Lopes Cardoso. 2022. Towards better evidence extraction methods for fact-checking systems. In

IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology, WIIAT ’21, page 277–284, New York, NY, USA. Association for Computing Machinery.

Zewen Chi, Li Dong, Furu Wei, Nan Yang, Saksham Singhal, Wenhui Wang, Xia Song, Xian-Ling Mao, Heyan Huang, and Ming Zhou. 2021. InfoXLM: An information-theoretic framework for cross-lingual language model pre-training. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3576–3588, Online. Association for Computational Linguistics.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo,

Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. 2024. Deepseek-v3 technical report. Preprint, arXiv:2412.19437.

Mitchell DeHaven and Stephen Scott. 2023. BEVERS: A general, simple, and performant framework for automatic fact verification. In Proceedings of the Sixth Fact Extraction and VERification Workshop (FEVER), pages 58–65, Dubrovnik, Croatia. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Mudasir A Ganaie, Minghui Hu, Ashwani Kumar Malik, Muhammad Tanveer, and Ponnuthurai N Suganthan. 2022. Ensemble deep learning: A review. Engineering Applications of Artificial Intelligence, 115:105151.

Alex Graves and J¨urgen Schmidhuber. 2005. Framewise phoneme classification with bidirectional lstm and other neural network architectures. Neural networks : the official journal of the International Neural Network Society, 18:602–10.

Tetyana Hannichenko, Peter Bidyuk, Irina Kalinina, and Oleksandr Zhebko. 2023. Classification system based on ensemble methods for solving machine learning tasks.

Siqing Huo, Negar Arabzadeh, and Charles Clarke. 2023. Retrieving supporting evidence for generative question answering. In Proceedings of the Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, SIGIR-AP ’23, page 11–20. ACM.

Ying Ju, Fubang Zhao, Shijie Chen, Bowen Zheng, Xuefeng Yang, and Yunfeng Liu. 2019. Technical report on conversational question answering. Preprint, arXiv:1909.10772.

Hung Tuan Le, Long Truong To, Manh Trong Nguyen, and Kiet Van Nguyen. 2024. Viwikifc: Fact-checking for vietnamese wikipedia-based textual knowledge source. Preprint, arXiv:2405.07615.

Kun Li. 2021. Haha at fakedes 2021: A fake news detection method based on tf-idf and ensemble machine learning. In IberLEF@ SEPLN, pages 630–638.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2018. Focal loss for dense object detection. Preprint, arXiv:1708.02002.

Ying-Jia Lin, Chun-Yi Lin, Chia-Jen Yeh, Yi-Ting Li, Yun-Yu Hu, Chih-Hao Hsu, Mei-Feng Lee, and Hung-Yu Kao. 2024. Cfever: A chinese fact extraction and verification dataset. Preprint, arXiv:2402.13025.

Yichen Liu, Abhijit Dasgupta, and Qiwei He. 2024. Music genre classification: Ensemble learning with subcomponents-level attention. Preprint, arXiv:2412.15602.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. Preprint, arXiv:1907.11692.

Christopher Malon. 2018. Team papelo: Transformer networks at FEVER. In Proceedings of the First Workshop on Fact Extraction and VERification (FEVER), pages 109–113, Brussels, Belgium. Association for Computational Linguistics.

Yixin Nie, Haonan Chen, and Mohit Bansal. 2018. Combining fact extraction and verification with neural semantic matching networks. Preprint, arXiv:1811.07039.

Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. 2021. ERNIE-M: Enhanced multilingual representation by aligning cross-lingual semantics with monolingual corpora. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 27–38, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Shahzad Qaiser and Ramsha Ali. 2018. Text mining: Use of tf-idf to examine the relevance of words to documents. International Journal of Computer Applications, 181.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji

Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Aniketh Janardhan Reddy, Gil Rocha, and Diego Esteves. 2018. Defactonlp: Fact verification using entity recognition, tfidf vector comparison and decomposable attention. arXiv preprint arXiv:1809.00509.

- Nils Reimers and Iryna Gurevych. 2019a. SentenceBERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.
- Nils Reimers and Iryna Gurevych. 2019b. SentenceBERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Tobias Schimanski, Jingwei Ni, Mathias Kraus, Elliott Ash, and Markus Leippold. 2024. Towards faithful and robust LLM specialists for evidence-based question-answering. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1913– 1931, Bangkok, Thailand. Association for Computational Linguistics.

Amir Soleimani, Christof Monz, and Marcel Worring. 2020. Bert for evidence retrieval and claim verification. In Advances in Information Retrieval: 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14–17, 2020, Proceedings, Part II, page 359–366, Berlin, Heidelberg. SpringerVerlag.

Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul R. Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, Jack Krawczyk, Cosmo Du, Ed Chi, HengTze Cheng, Eric Ni, Purvi Shah, Patrick Kane, Betty Chan, Manaal Faruqui, Aliaksei Severyn, Hanzhao Lin, YaGuang Li, Yong Cheng, Abe Ittycheriah, Mahdis Mahdieh, Mia Chen, Pei Sun, Dustin Tran, Sumit Bagri, Balaji Lakshminarayanan, Jeremiah Liu, Andras Orban, Fabian G¨ura, Hao Zhou, Xinying Song, Aurelien Boffy, Harish Ganapathy, Steven

Zheng, HyunJeong Choe, Ágoston Weisz, Tao Zhu, Yifeng Lu, Siddharth Gopal, Jarrod Kahn, Maciej Kula, Jeff Pitman, Rushin Shah, Emanuel Taropa, Majd Al Merey, Martin Baeuml, Zhifeng Chen, Laurent El Shafey, Yujing Zhang, Olcan Sercinoglu, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Ana¨ıs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, Alexandre Frechette, Charlotte Smith, Laura Culp, Lev Proleev, Yi Luan, Xi Chen, James Lottes, Nathan Schucher, Federico Lebron, Alban Rrustemi, Natalie Clay, Phil Crone, Tomas Kocisky, Jeffrey Zhao, Bartek Perz, Dian Yu, Heidi Howard, Adam Bloniarz, Jack W. Rae, Han Lu, Laurent Sifre, Marcello Maggioni, Fred Alcober, Dan Garrette, Megan Barnes, Shantanu Thakoor, Jacob Austin, Gabriel Barth-Maron, William Wong, Rishabh Joshi, Rahma Chaabouni, Deeni Fatiha, Arun Ahuja, Gaurav Singh Tomar, Evan Senter, Martin Chadwick, Ilya Kornakov, Nithya Attaluri, I˜naki Iturrate, Ruibo Liu, Yunxuan Li, Sarah Cogan, Jeremy Chen, Chao Jia, Chenjie Gu, Qiao Zhang, Jordan Grimstad, Ale Jakse Hartman, Xavier Garcia, Thanumalayan Sankaranarayana Pillai, Jacob Devlin, Michael Laskin, Diego de Las Casas, Dasha Valter, Connie Tao, Lorenzo Blanco, Adrià Puigdomènech Badia, David Reitter, Mianna Chen, Jenny Brennan, Clara Rivera, Sergey Brin, Shariq Iqbal, Gabriela Surita, Jane Labanowski, Abhi Rao, Stephanie Winkler, Emilio Parisotto, Yiming Gu, Kate Olszewska, Ravi Addanki, Antoine Miech, Annie Louis, Denis Teplyashin, Geoff Brown, Elliot Catt, Jan Balaguer, Jackie Xiang, Pidong Wang, Zoe Ashwood, Anton Briukhov, Albert Webson, Sanjay Ganapathy, Smit Sanghavi, Ajay Kannan, MingWei Chang, Axel Stjerngren, Josip Djolonga, Yuting Sun, Ankur Bapna, Matthew Aitchison, Pedram Pejman, Henryk Michalewski, Tianhe Yu, Cindy Wang, Juliette Love, Junwhan Ahn, Dawn Bloxwich, Kehang Han, Peter Humphreys, Thibault Sellam, James Bradbury, Varun Godbole, Sina Samangooei, Bogdan Damoc, Alex Kaskasoli, Sébastien M. R. Arnold, Vijay Vasudevan, Shubham Agrawal, Jason Riesa, Dmitry Lepikhin, Richard Tanburn, Srivatsan Srinivasan, Hyeontaek Lim, Sarah Hodkinson, Pranav Shyam, Johan Ferret, Steven Hand, Ankush Garg, Tom Le Paine, Jian Li, Yujia Li, Minh Giang, Alexander Neitz, Zaheer Abbas, Sarah York, Machel Reid, Elizabeth Cole, Aakanksha Chowdhery, Dipanjan Das, Dominika Rogozi´nska, Vitaliy Nikolaev, Pablo Sprechmann, Zachary Nado, Lukas Zilka, Flavien Prost, Luheng He, Marianne Monteiro, Gaurav Mishra, Chris Welty, Josh Newlan, Dawei Jia, Miltiadis Allamanis, Clara Huiyi Hu, Raoul de Liedekerke, Justin Gilmer, Carl Saroufim, Shruti Rijhwani, Shaobo Hou, Disha Shrivastava, Anirudh Baddepudi, Alex Goldin, Adnan Ozturel, Albin Cassirer, Yunhan Xu, Daniel Sohn, Devendra Sachan, Reinald Kim Amplayo, Craig Swanson, Dessie Petrova, Shashi Narayan, Arthur Guez, Siddhartha Brahma, Jessica Landon, Miteyan Patel, Ruizhe Zhao, Kevin Villela, Luyu Wang, Wenhao Jia, Matthew Rahtz, Mai Giménez, Legg Yeung, James

Keeling, Petko Georgiev, Diana Mincu, Boxi Wu, Salem Haykal, Rachel Saputro, Kiran Vodrahalli, James Qin, Zeynep Cankara, Abhanshu Sharma, Nick Fernando, Will Hawkins, Behnam Neyshabur, Solomon Kim, Adrian Hutter, Priyanka Agrawal, Alex Castro-Ros, George van den Driessche, Tao Wang, Fan Yang, Shuo yiin Chang, Paul Komarek, Ross McIlroy, Mario Luˇci´c, Guodong Zhang, Wael Farhan, Michael Sharman, Paul Natsev, Paul Michel, Yamini Bansal, Siyuan Qiao, Kris Cao, Siamak Shakeri, Christina Butterfield, Justin Chung, Paul Kishan Rubenstein, Shivani Agrawal, Arthur Mensch, Kedar Soparkar, Karel Lenc, Timothy Chung, Aedan Pope, Loren Maggiore, Jackie Kay, Priya Jhakra, Shibo Wang, Joshua Maynez, Mary Phuong, Taylor Tobin, Andrea Tacchetti, Maja Trebacz, Kevin Robinson, Yash Katariya, Sebastian Riedel, Paige Bailey, Kefan Xiao, Nimesh Ghelani, Lora Aroyo, Ambrose Slone, Neil Houlsby, Xuehan Xiong, Zhen Yang, Elena Gribovskaya, Jonas Adler, Mateo Wirth, Lisa Lee, Music Li, Thais Kagohara, Jay Pavagadhi, Sophie Bridgers, Anna Bortsova, Sanjay Ghemawat, Zafarali Ahmed, Tianqi Liu, Richard Powell, Vijay Bolina, Mariko Iinuma, Polina Zablotskaia, James Besley, Da-Woon Chung, Timothy Dozat, Ramona Comanescu, Xiance Si, Jeremy Greer, Guolong Su, Martin Polacek, Rapha¨el Lopez Kaufman, Simon Tokumine, Hexiang Hu, Elena Buchatskaya, Yingjie Miao, Mohamed Elhawaty, Aditya Siddhant, Nenad Tomasev, Jinwei Xing, Christina Greer, Helen Miller, Shereen Ashraf, Aurko Roy, Zizhao Zhang, Ada Ma, Angelos Filos, Milos Besta, Rory Blevins, Ted Klimenko, Chih-Kuan Yeh, Soravit Changpinyo, Jiaqi Mu, Oscar Chang, Mantas Pajarskas, Carrie Muir, Vered Cohen, Charline Le Lan, Krishna Haridasan, Amit Marathe, Steven Hansen, Sholto Douglas, Rajkumar Samuel, Mingqiu Wang, Sophia Austin, Chang Lan, Jiepu Jiang, Justin Chiu, Jaime Alonso Lorenzo, Lars Lowe Sj¨osund, Sébastien Cevey, Zach Gleicher, Thi Avrahami, Anudhyan Boral, Hansa Srinivasan, Vittorio Selo, Rhys May, Konstantinos Aisopos, Léonard Hussenot, Livio Baldini Soares, Kate Baumli, Michael B. Chang, Adrià Recasens, Ben Caine, Alexander Pritzel, Filip Pavetic, Fabio Pardo, Anita Gergely, Justin Frye, Vinay Ramasesh, Dan Horgan, Kartikeya Badola, Nora Kassner, Subhrajit Roy, Ethan Dyer, Víctor Campos Campos, Alex Tomala, Yunhao Tang, Dalia El Badawy, Elspeth White, Basil Mustafa, Oran Lang, Abhishek Jindal, Sharad Vikram, Zhitao Gong, Sergi Caelles, Ross Hemsley, Gregory Thornton, Fangxiaoyu Feng, Wojciech Stokowiec, Ce Zheng, Phoebe Thacker, C¸a˘glar Unl¨¨ u, Zhishuai Zhang, Mohammad Saleh, James Svensson, Max Bileschi, Piyush Patil, Ankesh Anand, Roman Ring, Katerina Tsihlas, Arpi Vezer, Marco Selvi, Toby Shevlane, Mikel Rodriguez, Tom Kwiatkowski, Samira Daruki, Keran Rong, Allan Dafoe, Nicholas FitzGerald, Keren Gu-Lemberg, Mina Khan, Lisa Anne Hendricks, Marie Pellat, Vladimir Feinberg, James Cobon-Kerr, Tara Sainath, Maribeth Rauh, Sayed Hadi Hashemi, Richard Ives, Yana Hasson, Eric Noland, Yuan Cao, Nathan Byrd, Le Hou, Qingze Wang, Thibault Sottiaux, Michela Paganini, Jean-Baptiste Lespiau, Alexandre Mou-

farek, Samer Hassan, Kaushik Shivakumar, Joost van Amersfoort, Amol Mandhane, Pratik Joshi, Anirudh Goyal, Matthew Tung, Andrew Brock, Hannah Sheahan, Vedant Misra, Cheng Li, Nemanja Raki´cevi´c, Mostafa Dehghani, Fangyu Liu, Sid Mittal, Junhyuk Oh, Seb Noury, Eren Sezener, Fantine Huot, Matthew Lamm, Nicola De Cao, Charlie Chen, Sidharth Mudgal, Romina Stella, Kevin Brooks, Gautam Vasudevan, Chenxi Liu, Mainak Chain, Nivedita Melinkeri, Aaron Cohen, Venus Wang, Kristie Seymore, Sergey Zubkov, Rahul Goel, Summer Yue, Sai Krishnakumaran, Brian Albert, Nate Hurley, Motoki Sano, Anhad Mohananey, Jonah Joughin, Egor Filonov, Tomasz, Yomna Eldawy, Jiawern Lim, Rahul Rishi, Shirin Badiezadegan, Taylor Bos, Jerry Chang, Sanil Jain, Sri Gayatri Sundara Padmanabhan, Subha Puttagunta, Kalpesh Krishna, Leslie Baker, Norbert Kalb, Vamsi Bedapudi, Adam Kurzrok, Shuntong Lei, Anthony Yu, Oren Litvin, Xiang Zhou, Zhichun Wu, Sam Sobell, Andrea Siciliano, Alan Papir, Robby Neale, Jonas Bragagnolo, Tej Toor, Tina Chen, Valentin Anklin, Feiran Wang, Richie Feng, Milad Gholami, Kevin Ling, Lijuan Liu, Jules Walter, Hamid Moghaddam, Arun Kishore, Jakub Adamek, Tyler Mercado, Jonathan Mallinson, Siddhinita Wandekar, Stephen Cagle, Eran Ofek, Guillermo Garrido, Clemens Lombriser, Maksim Mukha, Botu Sun, Hafeezul Rahman Mohammad, Josip Matak, Yadi Qian, Vikas Peswani, Pawel Janus, Quan Yuan, Leif Schelin, Oana David, Ankur Garg, Yifan He, Oleksii Duzhyi, Anton Algmyr,¨ Timothée Lottaz, Qi Li, Vikas Yadav, Luyao Xu, Alex Chinien, Rakesh Shivanna, Aleksandr Chuklin, Josie Li, Carrie Spadine, Travis Wolfe, Kareem Mohamed, Subhabrata Das, Zihang Dai, Kyle He, Daniel von Dincklage, Shyam Upadhyay, Akanksha Maurya, Luyan Chi, Sebastian Krause, Khalid Salama, Pam G Rabinovitch, Pavan Kumar Reddy M, Aarush Selvan, Mikhail Dektiarev, Golnaz Ghiasi, Erdem Guven, Himanshu Gupta, Boyi Liu, Deepak Sharma, Idan Heimlich Shtacher, Shachi Paul, Oscar Akerlund, Franc¸ois-Xavier Aubet, Terry Huang, Chen Zhu, Eric Zhu, Elico Teixeira, Matthew Fritze, Francesco Bertolini, Liana-Eleonora Marinescu, Martin B¨olle, Dominik Paulus, Khyatti Gupta, Tejasi Latkar, Max Chang, Jason Sanders, Roopa Wilson, Xuewei Wu, Yi-Xuan Tan, Lam Nguyen Thiet, Tulsee Doshi, Sid Lall, Swaroop Mishra, Wanming Chen, Thang Luong, Seth Benjamin, Jasmine Lee, Ewa Andrejczuk, Dominik Rabiej, Vipul Ranjan, Krzysztof Styrc, Pengcheng Yin, Jon Simon, Malcolm Rose Harriott, Mudit Bansal, Alexei Robsky, Geoff Bacon, David Greene, Daniil Mirylenka, Chen Zhou, Obaid Sarvana, Abhimanyu Goyal, Samuel Andermatt, Patrick Siegler, Ben Horn, Assaf Israel, Francesco Pongetti, Chih-Wei "Louis" Chen, Marco Selvatici, Pedro Silva, Kathie Wang, Jackson Tolins, Kelvin Guu, Roey Yogev, Xiaochen Cai, Alessandro Agostini, Maulik Shah, Hung Nguyen, Noah Ó Donnaile, Sébastien Pereira, Linda Friso, Adam Stambler, Adam Kurzrok, Chenkai Kuang, Yan Romanikhin, Mark Geller, ZJ Yan, Kane Jang, Cheng-Chun Lee, Wojciech Fica, Eric Malmi, Qijun Tan, Dan Banica, Daniel Balle, Ryan Pham,

Yanping Huang, Diana Avram, Hongzhi Shi, Jasjot Singh, Chris Hidey, Niharika Ahuja, Pranab Saxena, Dan Dooley, Srividya Pranavi Potharaju, Eileen O’Neill, Anand Gokulchandran, Ryan Foley, Kai Zhao, Mike Dusenberry, Yuan Liu, Pulkit Mehta, Ragha Kotikalapudi, Chalence Safranek-Shrader, Andrew Goodman, Joshua Kessinger, Eran Globen, Prateek Kolhar, Chris Gorgolewski, Ali Ibrahim, Yang Song, Ali Eichenbaum, Thomas Brovelli, Sahitya Potluri, Preethi Lahoti, Cip Baetu, Ali Ghorbani, Charles Chen, Andy Crawford, Shalini Pal, Mukund Sridhar, Petru Gurita, Asier Mujika, Igor Petrovski, Pierre-Louis Cedoz, Chenmei Li, Shiyuan Chen, Niccolò Dal Santo, Siddharth Goyal, Jitesh Punjabi, Karthik Kappaganthu, Chester Kwak, Pallavi LV, Sarmishta Velury, Himadri Choudhury, Jamie Hall, Premal Shah, Ricardo Figueira, Matt Thomas, Minjie Lu, Ting Zhou, Chintu Kumar, Thomas Jurdi, Sharat Chikkerur, Yenai Ma, Adams Yu, Soo Kwak, Victor Ahdel,¨ Sujeevan Rajayogam, Travis Choma, Fei Liu, Aditya Barua, Colin Ji, Ji Ho Park, Vincent Hellendoorn, Alex Bailey, Taylan Bilal, Huanjie Zhou, Mehrdad Khatir, Charles Sutton, Wojciech Rzadkowski, Fiona Macintosh, Konstantin Shagin, Paul Medina, Chen Liang, Jinjing Zhou, Pararth Shah, Yingying Bi, Attila Dankovics, Shipra Banga, Sabine Lehmann, Marissa Bredesen, Zifan Lin, John Eric Hoffmann, Jonathan Lai, Raynald Chung, Kai Yang, Nihal Balani, Arthur Braˇzinskas, Andrei Sozanschi, Matthew Hayes, Héctor Fernández Alcalde, Peter Makarov, Will Chen, Antonio Stella, Liselotte Snijders, Michael Mandl, Ante K¨arrman, Paweł Nowak, Xinyi Wu, Alex Dyck, Krishnan Vaidyanathan, Raghavender R, Jessica Mallet, Mitch Rudominer, Eric Johnston, Sushil Mittal, Akhil Udathu, Janara Christensen, Vishal Verma, Zach Irving, Andreas Santucci, Gamaleldin Elsayed, Elnaz Davoodi, Marin Georgiev, Ian Tenney, Nan Hua, Geoffrey Cideron, Edouard Leurent, Mahmoud Alnahlawi, Ionut Georgescu, Nan Wei, Ivy Zheng, Dylan Scandinaro, Heinrich Jiang, Jasper Snoek, Mukund Sundararajan, Xuezhi Wang, Zack Ontiveros, Itay Karo, Jeremy Cole, Vinu Rajashekhar, Lara Tumeh, Eyal Ben-David, Rishub Jain, Jonathan Uesato, Romina Datta, Oskar Bunyan, Shimu Wu, John Zhang, Piotr Stanczyk, Ye Zhang, David Steiner, Subhajit Naskar, Michael Azzam, Matthew Johnson, Adam Paszke, Chung-Cheng Chiu, Jaume Sanchez Elias, Afroz Mohiuddin, Faizan Muhammad, Jin Miao, Andrew Lee, Nino Vieillard, Jane Park, Jiageng Zhang, Jeff Stanway, Drew Garmon, Abhijit Karmarkar, Zhe Dong, Jong Lee, Aviral Kumar, Luowei Zhou, Jonathan Evens, William Isaac, Geoffrey Irving, Edward Loper, Michael Fink, Isha Arkatkar, Nanxin Chen, Izhak Shafran, Ivan Petrychenko, Zhe Chen, Johnson Jia, Anselm Levskaya, Zhenkai Zhu, Peter Grabowski, Yu Mao, Alberto Magni, Kaisheng Yao, Javier Snaider, Norman Casagrande, Evan Palmer, Paul Suganthan, Alfonso Casta˜no, Irene Giannoumis, Wooyeol Kim, Mikołaj Rybi´nski, Ashwin Sreevatsa, Jennifer Prendki, David Soergel, Adrian Goedeckemeyer, Willi Gierke, Mohsen Jafari, Meenu Gaba, Jeremy Wiesner, Diana Gage Wright, Yawen Wei, Harsha Vashisht, Yana Kulizhskaya, Jay

Hoover, Maigo Le, Lu Li, Chimezie Iwuanyanwu, Lu Liu, Kevin Ramirez, Andrey Khorlin, Albert Cui, Tian LIN, Marcus Wu, Ricardo Aguilar, Keith Pallo, Abhishek Chakladar, Ginger Perng, Elena Allica Abellan, Mingyang Zhang, Ishita Dasgupta, Nate Kushman, Ivo Penchev, Alena Repina, Xihui Wu, Tom van der Weide, Priya Ponnapalli, Caroline Kaplan, Jiri Simsa, Shuangfeng Li, Olivier Dousse, Fan Yang, Jeff Piper, Nathan Ie, Rama Pasumarthi, Nathan Lintz, Anitha Vijayakumar, Daniel Andor, Pedro Valenzuela, Minnie Lui, Cosmin Paduraru, Daiyi Peng, Katherine Lee, Shuyuan Zhang, Somer Greene, Duc Dung Nguyen, Paula Kurylowicz, Cassidy Hardin, Lucas Dixon, Lili Janzer, Kiam Choo, Ziqiang Feng, Biao Zhang, Achintya Singhal, Dayou Du, Dan McKinnon, Natasha Antropova, Tolga Bolukbasi, Orgad Keller, David Reid, Daniel Finchelstein, Maria Abi Raad, Remi Crocker, Peter Hawkins, Robert Dadashi, Colin Gaffney, Ken Franko, Anna Bulanova, Rémi Leblond, Shirley Chung, Harry Askham, Luis C. Cobo, Kelvin Xu, Felix Fischer, Jun Xu, Christina Sorokin, Chris Alberti, Chu-Cheng Lin, Colin Evans, Alek Dimitriev, Hannah Forbes, Dylan Banarse, Zora Tung, Mark Omernick, Colton Bishop, Rachel Sterneck, Rohan Jain, Jiawei Xia, Ehsan Amid, Francesco Piccinno, Xingyu Wang, Praseem Banzal, Daniel J. Mankowitz, Alex Polozov, Victoria Krakovna, Sasha Brown, MohammadHossein Bateni, Dennis Duan, Vlad Firoiu, Meghana Thotakuri, Tom Natan, Matthieu Geist, Ser tan Girgin, Hui Li, Jiayu Ye, Ofir Roval, Reiko Tojo, Michael Kwong, James Lee-Thorp, Christopher Yew, Danila Sinopalnikov, Sabela Ramos, John Mellor, Abhishek Sharma, Kathy Wu, David Miller, Nicolas Sonnerat, Denis Vnukov, Rory Greig, Jennifer Beattie, Emily Caveness, Libin Bai, Julian Eisenschlos, Alex Korchemniy, Tomy Tsai, Mimi Jasarevic, Weize Kong, Phuong Dao, Zeyu Zheng, Frederick Liu, Fan Yang, Rui Zhu, Tian Huey Teh, Jason Sanmiya, Evgeny Gladchenko, Nejc Trdin, Daniel Toyama, Evan Rosen, Sasan Tavakkol, Linting Xue, Chen Elkind, Oliver Woodman, John Carpenter, George Papamakarios, Rupert Kemp, Sushant Kafle, Tanya Grunina, Rishika Sinha, Alice Talbert, Diane Wu, Denese Owusu-Afriyie, Cosmo Du, Chloe Thornton, Jordi Pont-Tuset, Pradyumna Narayana, Jing Li, Saaber Fatehi, John Wieting, Omar Ajmeri, Benigno Uria, Yeongil Ko, Laura Knight, Amélie Héliou, Ning Niu, Shane Gu, Chenxi Pang, Yeqing Li, Nir Levine, Ariel Stolovich, Rebeca Santamaria-Fernandez, Sonam Goenka, Wenny Yustalim, Robin Strudel, Ali Elqursh, Charlie Deck, Hyo Lee, Zonglin Li, Kyle Levin, Raphael Hoffmann, Dan Holtmann-Rice, Olivier Bachem, Sho Arora, Christy Koh, Soheil Hassas Yeganeh, Siim Põder, Mukarram Tariq, Yanhua Sun, Lucian Ionita, Mojtaba Seyedhosseini, Pouya Tafti, Zhiyu Liu, Anmol Gulati, Jasmine Liu, Xinyu Ye, Bart Chrzaszcz, Lily Wang, Nikhil Sethi, Tianrun Li, Ben Brown, Shreya Singh, Wei Fan, Aaron Parisi, Joe Stanton, Vinod Koverkathu, Christopher A. ChoquetteChoo, Yunjie Li, TJ Lu, Abe Ittycheriah, Prakash Shroff, Mani Varadarajan, Sanaz Bahargam, Rob Willoughby, David Gaddy, Guillaume Desjardins,

Marco Cornero, Brona Robenek, Bhavishya Mittal, Ben Albrecht, Ashish Shenoy, Fedor Moiseev, Henrik Jacobsson, Alireza Ghaffarkhah, Morgane Rivière, Alanna Walton, Clément Crepy, Alicia Parrish, Zongwei Zhou, Clement Farabet, Carey Radebaugh, Praveen Srinivasan, Claudia van der Salm, Andreas Fidjeland, Salvatore Scellato, Eri Latorre-Chimoto, Hanna Klimczak-Pluci´nska, David Bridson, Dario de Cesare, Tom Hudson, Piermaria Mendolicchio, Lexi Walker, Alex Morris, Matthew Mauger, Alexey Guseynov, Alison Reid, Seth Odoom, Lucia Loher, Victor Cotruta, Madhavi Yenugula, Dominik Grewe, Anastasia Petrushkina, Tom Duerig, Antonio Sanchez, Steve Yadlowsky, Amy Shen, Amir Globerson, Lynette Webb, Sahil Dua, Dong Li, Surya Bhupatiraju, Dan Hurt, Haroon Qureshi, Ananth Agarwal, Tomer Shani, Matan Eyal, Anuj Khare, Shreyas Rammohan Belle, Lei Wang, Chetan Tekur, Mihir Sanjay Kale, Jinliang Wei, Ruoxin Sang, Brennan Saeta, Tyler Liechty, Yi Sun, Yao Zhao, Stephan Lee, Pandu Nayak, Doug Fritz, Manish Reddy Vuyyuru, John Aslanides, Nidhi Vyas, Martin Wicke, Xiao Ma, Evgenii Eltyshev, Nina Martin, Hardie Cate, James Manyika, Keyvan Amiri, Yelin Kim, Xi Xiong, Kai Kang, Florian Luisier, Nilesh Tripuraneni, David Madras, Mandy Guo, Austin Waters, Oliver Wang, Joshua Ainslie, Jason Baldridge, Han Zhang, Garima Pruthi, Jakob Bauer, Feng Yang, Riham Mansour, Jason Gelman, Yang Xu, George Polovets, Ji Liu, Honglong Cai, Warren Chen, XiangHai Sheng, Emily Xue, Sherjil Ozair, Christof Angermueller, Xiaowei Li, Anoop Sinha, Weiren Wang, Julia Wiesinger, Emmanouil Koukoumidis, Yuan Tian, Anand Iyer, Madhu Gurumurthy, Mark Goldenson, Parashar Shah, MK Blake, Hongkun Yu, Anthony Urbanowicz, Jennimaria Palomaki, Chrisantha Fernando, Ken Durden, Harsh Mehta, Nikola Momchev, Elahe Rahimtoroghi, Maria Georgaki, Amit Raul, Sebastian Ruder, Morgan Redshaw, Jinhyuk Lee, Denny Zhou, Komal Jalan, Dinghua Li, Blake Hechtman, Parker Schuh, Milad Nasr, Kieran Milan, Vladimir Mikulik, Juliana Franco, Tim Green, Nam Nguyen, Joe Kelley, Aroma Mahendru, Andrea Hu, Joshua Howland, Ben Vargas, Jeffrey Hui, Kshitij Bansal, Vikram Rao, Rakesh Ghiya, Emma Wang, Ke Ye, Jean Michel Sarr, Melanie Moranski Preston, Madeleine Elish, Steve Li, Aakash Kaku, Jigar Gupta, Ice Pasupat, Da-Cheng Juan, Milan Someswar, Tejvi M., Xinyun Chen, Aida Amini, Alex Fabrikant, Eric Chu, Xuanyi Dong, Amruta Muthal, Senaka Buthpitiya, Sarthak Jauhari, Nan Hua, Urvashi Khandelwal, Ayal Hitron, Jie Ren, Larissa Rinaldi, Shahar Drath, Avigail Dabush, Nan-Jiang Jiang, Harshal Godhia, Uli Sachs, Anthony Chen, Yicheng Fan, Hagai Taitelbaum, Hila Noga, Zhuyun Dai, James Wang, Chen Liang, Jenny Hamer, Chun-Sung Ferng, Chenel Elkind, Aviel Atias, Paulina Lee, Vít Listík, Mathias Carlen, Jan van de Kerkhof, Marcin Pikus, Krunoslav Zaher, Paul M¨uller, Sasha Zykova, Richard Stefanec, Vitaly Gatsko, Christoph Hirnschall, Ashwin Sethi, Xingyu Federico Xu, Chetan Ahuja, Beth Tsai, Anca Stefanoiu, Bo Feng, Keshav Dhandhania, Manish Katyal, Akshay Gupta, Atharva Parulekar, Divya Pitta, Jing Zhao, Vivaan

Bhatia, Yashodha Bhavnani, Omar Alhadlaq, Xiaolin Li, Peter Danenberg, Dennis Tu, Alex Pine, Vera Filippova, Abhipso Ghosh, Ben Limonchik, Bhargava Urala, Chaitanya Krishna Lanka, Derik Clive, Yi Sun, Edward Li, Hao Wu, Kevin Hongtongsak, Ianna Li, Kalind Thakkar, Kuanysh Omarov, Kushal Majmundar, Michael Alverson, Michael Kucharski, Mohak Patel, Mudit Jain, Maksim Zabelin, Paolo Pelagatti, Rohan Kohli, Saurabh Kumar, Joseph Kim, Swetha Sankar, Vineet Shah, Lakshmi Ramachandruni, Xiangkai Zeng, Ben Bariach, Laura Weidinger, Tu Vu, Alek Andreev, Antoine He, Kevin Hui, Sheleem Kashem, Amar Subramanya, Sissie Hsiao, Demis Hassabis, Koray Kavukcuoglu, Adam Sadovsky, Quoc Le, Trevor Strohman, Yonghui Wu, Slav Petrov, Jeffrey Dean, and Oriol Vinyals. 2024. Gemini: A family of highly capable multimodal models. Preprint, arXiv:2312.11805.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809–819, New Orleans, Louisiana. Association for Computational Linguistics.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971.

Bao Tran, T. N. Khanh, Khang Nguyen Tuong, Thien Dang, Quang Nguyen, Nguyen T. Thinh, and Vo T. Hung. 2024a. BERT-Based Model for Vietnamese Fact Verification Dataset, page 219–231. Springer Nature Switzerland.

Quang-Duy Tran, Thai-Hoa Tran, and Khanh Quoc Tran. 2024b. Advancing vietnamese fact extraction and verification through multi-stage text ranking. In 2024 International Conference on Multimedia Analysis and Pattern Recognition (MAPR), pages 1–7.

Guangtao Wang, Qinbao Song, and Xiaoyan Zhu. 2021. Ensemble learning based classification algorithm recommendation. Preprint, arXiv:2101.05993.

Moy Yuan and Andreas Vlachos. 2024. Zero-shot fact-checking with semantic triples and knowledge graphs. In Proceedings of the 1st Workshop on Knowledge Graphs and Large Language Models (KaLLM 2024), pages 105–115, Bangkok, Thailand. Association for Computational Linguistics.

Wanjun Zhong, Jingjing Xu, Duyu Tang, Zenan Xu, Nan Duan, Ming Zhou, Jiahai Wang, and Jian Yin. 2020. Reasoning over semantic-level graph for fact checking. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages

6170–6180, Online. Association for Computational Linguistics.

Jie Zhou, Xu Han, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, and Maosong Sun. 2019. GEAR: Graph-based evidence aggregating and reasoning for fact verification. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 892–901, Florence, Italy. Association for Computational Linguistics.

### A Comprehensive Data Processing

- A.1 Dataset Statistics

This section presents the detailed statistics of the two datasets used in our experiments: ISE-DSC01 and ViWikiFC. The table below summarizes the number of samples in the training, development, and test sets for both corpora. These statistics help contextualize the scale and evaluation coverage of our methods.

ISE-DSC01 ViWikiFC Train 37,967 16,738 Dev 4,794 2,090 Test 5,396 2,091

Table 3: Dataset statistics for ISE-DSC01 and ViWikiFC.

- A.2 Data Analysis on Context Lengths

Figure 5 illustrates the distribution of token lengths across input contexts in the ViWikiFC and ISEDSC01 datasets. As shown, many samples significantly exceed the 512-token input limitation of standard Transformer models. The ISE-DSC01 dataset, in particular, contains several contexts with over 4,000 tokens. This analysis highlights the necessity for effective context segmentation strategies to ensure full coverage of relevant evidence while maintaining compatibility with model constraints.

Comparison of Token Counts in Contexts

5000

Max: 4803

4000

3000

TokenLength

2000

1000

Max: 598

0

ViWiki ISE-DSC Dataset

Figure 5: Graph representing the lengths of contexts.

Figure 3 illustrates our strategy for handling long input contexts. First, the context is segmented into individual sentences. Next, sentences are sequentially aggregated into subcontexts until reaching approximately 400 tokens. Each completed subcontext is then passed to the QATC model to identify potential evidence. The next subcontext begins from the subsequent sentence, continuing until all sentences have been processed. However, processing subcontexts sequentially can be time-consuming. Therefore, we developed SER Faster, which batches subcontexts and processes them in parallel, significantly accelerating the retrieval process.

### B Strict Accuracy in Fact-Checking

Strict Accuracy: This metric is a stringent measure that requires both the verdict and the evidence to be predicted correctly compared to the ground truth sample.

Verdict (v and v′): refers to the verdict of the sample and the predicted verdict (supported, refuted, nei). Evidence (e and e′): refers to the evidence of the sample and the predicted evidence.

StrAcc = f(v,v′).f(e,e′) (6) Where:

1 if v = v′ 0 otherwise

f(v,v′) =

(7)

f(e,e′) =

1 if e = e′ 0 otherwise

Strict accuracy is the average of all StrAcc values.

### C Ablation Study

#### C.1 Ablation Experiments

(8)

|Configuration|ViWikiFC ISE-DSC01<br><br>|
|---|---|
|ER Model VC Model|Strict Acc VC Acc ER Acc Strict Acc VC Acc ER Acc|

SemViQA (Full)

InfoXLMlarge 80.68 83.98 95.31 75.13 79.60 76.87 XLM-Rlarge 80.82 83.88 95.31 76.74 81.71 78.95 Ernie-Mlarge 80.06 83.17 95.31 78.97 82.49 80.91

InfoXLMlarge

w/o Binary Classification (One-step VC)

InfoXLMlarge 79.63 82.88 95.31 73.87 78.35 76.89 XLM-Rlarge 80.73 83.69 95.31 75.96 80.80 78.97 Ernie-Mlarge 79.91 83.07 95.31 78.47 81.89 80.93

InfoXLMlarge

w/o QATC (TF-IDF-based ER)

InfoXLMlarge 76.57 83.26 90.15 74.89 79.36 76.61 XLM-Rlarge 76.47 82.93 90.15 76.39 81.41 78.58 Ernie-Mlarge 75.75 81.97 90.15 78.71 82.28 80.65

TF-IDF

Table 4: Ablation results of SemViQA.

- Table 4 presents the ablation results to evaluate the contribution of each component in the SemViQA

framework. When employing the full model with both the QATC-based evidence retrieval and the two-step verdict classification (TVC), SemViQA achieves the best performance across both datasets. Notably, using Ernie-Mlarge yields the highest strict accuracy (78.97%) and evidence retrieval accuracy (80.91%) on the ISE-DSC01 dataset. Removing the binary classification stage (i.e., using only one-step VC) leads to a noticeable performance drop, especially on ISE-DSC01, indicating that the binary classifier enhances the distinction between Supported and Refuted labels. Furthermore, replacing the QATC module with a TF-IDF based retriever results in a significant decline (about 5%) in evidence retrieval accuracy, which subsequently affects the overall performance. These findings highlight the critical role of both QATC and the two-step classification scheme in improving SemViQA’s effectiveness on fact verification in Vietnamese.

#### C.2 Analysis of Confidence Threshold in SemViQA

The confidence threshold plays a crucial role in balancing accuracy and inference time in SemViQA’s evidence retrieval process. Analysis from Figure 6 indicates that as the threshold increases from 0.0 to 0.5, evidence retrieval accuracy improves significantly, particularly on ViWikiFC ( 95%) and ISE-DSC01 ( 80.8%). However, beyond 0.5, accuracy gains plateau, while inference time decreases sharply due to the system filtering out low-confidence evidence more aggressively. Setting an optimal threshold in the range of 0.4 - 0.5 achieves a trade-off between efficiency and accuracy, ensuring that SemViQA operates swiftly while maintaining precise evidence retrieval.

ViWikiFC

###### ISE-DSC01

- 89

- 90

- 91

- 92

- 93

- 94

- 95

- 96

81.6

InfoXLM Acc

InfoXLM Acc

4.0

| |
|---|

viMRC Acc

ViMRC Acc

95.31

95.22

InfoXLM Time

InfoXLM Time

81.4

| |
|---|

viMRC Time

ViMRC Time

3.5

94.69

94.36

81.2

3.0

81.0

EvidenceAccuracy(\%)

EvidenceAccuracy(\%)

80.91

80.91

93.21

93.11

| |
|---|

2.5

80.86

80.82

3Time(10s)

- 2

4

6

8

- 3Time(10s)

80.78

80.76

80.8

2.0

80.65

80.65

80.6

1.5

80.4

1.0

| |
|---|

90.15

90.15

80.2

| |
|---|

0.5

80.0

| |
|---|

| |
|---|

0.0

0.0 0.25 0.5 0.75 Confidence Threshold

0.0 0.25 0.5 0.75 Confidence Threshold

Figure 6: Impact of confidence threshold on evidence retrieval accuracy in SemViQA.

|SemViQA|ViWikiFC ISE-DSC01<br><br>|
|---|---|
|SER TVC|Strict Acc VC Acc ER Acc Strict Acc VC Acc ER Acc|

QATC-based ER

|InfoXLMlarge<br><br>InfoXLMlarge XLM-Rlarge Ernie-Mlarge|80.68 83.98 95.31 75.13 79.60 76.87 80.82 83.88 95.31 76.74 81.71 78.95 80.06 83.17 95.31 78.97 82.49 80.91<br><br>|
|---|---|
|ViMRClarge<br><br>InfoXLMlarge XLM-Rlarge Ernie-Mlarge|80.25 83.84 94.69 75.13 79.54 76.87 80.34 83.64 94.69 76.71 81.65 78.91 79.53 82.97 94.69 78.97 82.54 80.91<br><br>|

QA-based ER

|InfoXLMlarge<br><br>InfoXLMlarge XLM-Rlarge Ernie-Mlarge|79.96 83.50 94.45 74.02 78.95 75.83<br><br>80.11 83.60 94.45 75.61 80.95 77.91<br><br><br>79.24 82.74 94.45 77.82 81.76 79.82<br><br>|
|---|---|
|ViMRClarge<br><br>InfoXLMlarge XLM-Rlarge Ernie-Mlarge<br><br>|79.77 83.84 94.26 74.05 78.93 75.87 79.87 83.79 94.26 75.65 80.93 77.95 79.01 82.78 94.26 77.84 81.73 79.86|

- Table 5: Comparison of QATC-based vs. QA-based evidence retrieval in SemViQA. QATC consistently improves all metrics.

#### C.3 Effectiveness of QATC over Traditional QA-based Evidence Retrieval

To compare the learning capabilities of QATC with traditional QA models, we construct two versions of the SemViQA pipeline that are identical in structure, differing only in the evidence retrieval model. As shown in Table 5, using QATC consistently outperforms QA-based retrieval across both datasets. Specifically, on ViWikiFC, QATC achieves up to 80.82% Strict Accuracy and 95.31% ER Accuracy, while QA-based models peak at 80.11% and 94.45%, respectively. The improvement is even more evident on ISE-DSC01, where QATC reaches 78.97% Strict Accuracy and 80.91% ER Accuracy. These results confirm that QATC is more effective at learning to identify relevant evidence, leading to superior performance across the entire fact-checking pipeline.

#### C.4 Analysis of Confusion Matrix in Verdict Classification

To evaluate the effectiveness of our Two-step Verdict Classification (TVC) strategy, we analyze the confusion matrices in Figure 7, which compare the standard three-class classification with our proposed two-step approach, focusing solely on Supported and Refuted claims. The three-class classifier achieves strong performance, with accuracy of 95.4% for Supported and 94.6% for Refuted; however, it still exhibits notable confusion between the two classes, likely due to their semantic proximity and shared evidence patterns. In contrast, the two-step approach—by isolating NEI cases early and applying a dedicated

Three-Class Classification

Two-step Verdict Classification

[Figure 9]

[Figure 10]

95.4% 4.6%

98.0% 2.0%

SUPPORTED

SUPPORTED

Truelabel

Truelabel

5.4% 94.6%

3.6% 96.4%

REFUTED

REFUTED

SUPPORTED REFUTED Predicted label

SUPPORTED REFUTED Predicted label

Figure 7: Confusion matrix of the Three-Class and Two-Step Verdict Classification (TVC) on the ISE-DSC01 dataset.

binary classifier further improves accuracy to 98.0% for Supported and 96.4% for Refuted. These results support our hypothesis that decomposing the verdict classification task into semantically coherent subtasks enhances the model’s precision in detecting factual consistency.

### D Hyperparameter and LLM Training Configuration

In this section, we present the detailed hyperparameter settings and training configurations for both our SemViQA models and the Large Language Model (LLM) fine-tuning process. Table 6 consolidates all hyperparameters used across different models, including Binary Classification (BC), Three-Class Classification (TC), Question Answering with Token Classification (QATC), and LLM fine-tuning.

|Hyperparameter<br><br>|BC TC QATC LLM|
|---|---|
|Epochs RT Loss Cross-Entropy Loss Focal Loss Learning Rate Batch Size Gradient Accumulation Optimizer (AdamW) Max Token Length GPUs Zero LR Schedule Mixed Precision<br><br>|20 20 20 1<br><br>- - ✓ -<br>- ✓ ✓ -<br><br><br>✓ - - 1e−5 1e−5 2e−6 5e−5<br><br>104 104 36 2 1 1 2 1 ✓ ✓ ✓ ✓ 256 256 512 4096<br><br>A100 A100 A100 A100 - - - Zero3 Linear Linear Cyclic Cosine<br><br>- - - bf16|

- Table 6: Consolidated hyperparameter and training configuration for SemViQA models and LLM fine-tuning.

We fine-tune a Large Language Model (LLM) using a restructured version of the original datasets, ViWikiFC and ISE-DSC01, as detailed in Figure 7. These datasets have been carefully adapted for training to improve performance and ensure compatibility with our model. For training, we utilize the official Qwen LLM implementation from the QwenLM repository6. Our training setup follows the full configuration outlined in Table 6, ensuring optimal efficiency and alignment with best practices.

6https://github.com/QwenLM/Qwen

|Question: You are tasked with verifying the correctness of the following statement.<br><br>- We provide you with a claim and a context. Please classify the claim into one of three labels: “Supported”, “REFUTED”, or “NEI” (Not Enough Info).<br>- Your answer should include the classification label and the most relevant evidence sentence from the context.<br>- Remember, the evidence must be a full sentence, not part of a sentence or less than one sentence.<br><br><br>Given a claim and context as follows:<br><br>Context: The actress revealed her secrets to maintaining a youthful appearance as follows: Eating three balanced meals a day. For dinner, Ivy Chen usually eats early to ensure her body has enough time to digest food, metabolize energy, and avoid putting pressure on the stomach and other organs. A recent study published in *Frontiers in Nutrition* suggests that eating dinner earlier can lead to a longer lifespan, with the ideal time being 7 PM. If this is not possible, experts recommend having the last meal of the day 2-3 hours before bedtime. Drinking ginger tea: To keep her body warm, promote blood circulation, and enhance circulation, Ivy Chen drinks ginger tea daily. Her ginger tea is typically made with ground ginger, black tea, turmeric powder, and brown sugar. This drink is a natural remedy that not only boosts the immune system and reduces inflammation but also fights oxidation, supports weight loss, improves skin health, and helps maintain a youthful look. Regular exercise: Ivy Chen is a fitness enthusiast who loves physical activities and exercises daily, even during pregnancy. The Taiwanese actress shared that if she is not busy with work, she runs for at least 30 minutes every day. Even when traveling abroad, she maintains her running habit. A recent study published in *Progress in Cardiovascular Disease* found that regular runners live three years longer than non-runners. Running significantly helps with weight loss, maintaining a balanced physique, toning muscles, relaxing the mind, and benefiting heart health. Besides running, Ivy Chen also swims, practices yoga, and hikes to maintain physical fitness and endurance. Skincare: Regarding her skincare routine, the actress emphasized the importance of hydration. The Taiwanese beauty revealed that she always carries a facial mist to ensure her skin stays hydrated while outdoors.<br><br>Claim: Even when traveling abroad, Ivy Chen maintains her running habit. Answer: This claim is classified as Supported. The evidence is: Even when traveling abroad, she maintains her running habit.|
|---|

- Table 7: Example of a fact-checking task prompt used for LLM training. Note: Some parts of the Context and Claim were originally in Vietnamese. In this paper, we have translated them into English for better readability. Sentences highlighted in blue indicate the evidence.

We present the complete training progress of the LLM models and QATC in Figure 9 and Figure 8, respectively. Figure 9 illustrates the training dynamics of Qwen 1.5B and Qwen 3B, supporting the results presented in Table 1. Notably, the Qwen 1.5B model demonstrates more stable training dynamics compared to the Qwen 3B model during the initial stage. Meanwhile, Figure 8 showcases the completion of QATC training, depicting the loss curves of ViMRClarge and InfoXLMlarge. These results highlight the convergence behavior of QATC training across different architectures, further supporting the robustness of our approach.

ViWikiFC

###### ISE-DSC01

ViMRC

ViMRC

| |
|---|

InfoXLM

InfoXLM

3.0

3.0

2.5

2.5

| |
|---|

2.0

2.0

Loss

Loss

1.5

1.5

| |
|---|

1.0

1.0

| |
|---|

0.5

0.5

| |
|---|

| | |
|---|---|

0.0

0.0

0 2000 4000 6000 8000 Training Steps

0 2000 4000 6000 8000 Training Steps

Figure 8: Training progress of the ViMRClarge and InfoXLMlarge models.

Qwen2.5 1.5B Qwen2.5 3B

ViWikiFC

2.5

2.0

1.5

Loss

1.0

0.5

0.0

0 200 400 600 800 1000

Epoch

ISE-DSC01

2.0

1.5

Loss

1.0

0.5

0.0

0 200 400 600 800 1000 1200

Epoch

Figure 9: Training progress of the Qwen 1.5B and Qwen 3B models.

### E Comparison of TF-IDF and QATC in Fact-Checking: Examples of Incorrect vs. Correct Evidence Selection

|Claim|Evidence<br><br>|TF-IDF<br><br>|QATC|
|---|---|---|---|
|Du lịch Triều Tiên là điều mà chỉ có một số người được đi đến. (Traveling to North Korea is something only a few people can do.)|Theo nguyên tắc, bất kỳ ai cũng được phép du lịch tới Triều Tiên, và những ai có thể hoàn thành quá trình làm thủ tục thì đều không bị Triều Tiên từ chối cho nhập cảnh. (In principle, anyone is allowed to travel to North Korea, and those who complete the process are not denied entry.)<br><br>|Khách du lịch không được đi thăm thú bên ngoài vùng đã được cho phép trước mà không được hướng dẫn viên người Triều Tiên cho phép nhằm tránh các điệp viên nằm vùng. (Tourists are not allowed to visit areas outside of the designated zones without a North Korean guide to prevent undercover spies.)|Theo nguyên tắc, bất kỳ ai cũng được phép du lịch tới Triều Tiên, và những ai có thể hoàn thành quá trình làm thủ tục thì đều không bị Triều Tiên từ chối cho nhập cảnh. (In principle, anyone is allowed to travel to North Korea, and those who complete the process are not denied entry.)|
|Nó có độ nóng chảy ở mức gần 30 độ C. (It has a melting point of about 30°C.)|Nó là một kim loại kiềm mềm, màu bạc, và với điểm nóng chảy là 28 °C (83 °F) khiến cho nó trở thành một trong các kim loại ở dạng lỏng tại hay gần nhiệt độ phòng. (It is a soft, silvery alkali metal with a melting point of 28°C (83°F), making it one of the metals that is liquid at or near room temperature.)<br><br>|Nó là nguyên tố có độ âm điện thấp thứ hai sau franci, và chỉ có một đồng vị bền là caesi-133. (It is the second least electronegative element after francium, and has only one stable isotope, cesium-133.)<br><br>|Nó là một kim loại kiềm mềm, màu bạc, và với điểm nóng chảy là 28 °C (83 °F) khiến cho nó trở thành một trong các kim loại ở dạng lỏng tại hay gần nhiệt độ phòng. (It is a soft, silvery alkali metal with a melting point of 28°C (83°F), making it one of the metals that is liquid at or near room temperature.)|

- Table 8: Comparison of TF-IDF and QATC in Fact-Checking: TF-IDF selects irrelevant evidence (Incorrect), while QATC selects accurate evidence (Correct).

