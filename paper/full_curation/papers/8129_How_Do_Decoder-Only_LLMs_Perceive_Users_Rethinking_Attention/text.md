# arXiv:2602.10622v1[cs.CL]11Feb2026

## How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning

Jiahao Yuan1,2†, Yike Xu1†, Jinyong Wen1, Baokun Wang1,∗, Yang Chen1, Xiaotong Lin1, Wuliang Huang1, Ziyi Gao1, Xing Fu1, Yu Cheng1, Weiqiang Wang1

1DeepFind Team, Ant Group, 2East China Normal University ∗Corresponding author, †Jiahao Yuan and Yike Xu contributed equally to this work. The work was completed during Jiahao(ECNU)’s internship at Ant Group. jhyuan.cs@gmail.com

Decoder-only large language models are increasingly used as behavioral encoders for user representation learning, yet the impact of attention masking on the quality of user embeddings remains underexplored. In this work, we conduct a systematic study of causal, hybrid, and bidirectional attention masks within a unified contrastive learning framework trained on large-scale real-world Alipay data that integrates long-horizon heterogeneous user behaviors. To improve training dynamics when transitioning from causal to bidirectional attention, we propose Gradient-Guided Soft Masking, a gradient-based pre-warmup applied before a linear scheduler that gradually opens future attention during optimization. Evaluated on 9 industrial user cognition benchmarks covering prediction, preference, and marketing sensitivity tasks, our approach consistently yields more stable training and higher-quality bidirectional representations compared with causal, hybrid, and scheduler-only baselines, while remaining compatible with decoder pretraining. Overall, our findings highlight the importance of masking design and training transition in adapting decoder-only LLMs for effective user representation learning. Our code is available at https://github.com/JhCircle/Deepfind-GGSM.

[Figure 1]

Correspondence: yike.wbk@antgroup.com

#### 1 Introduction

User embeddings integrate large-scale heterogeneous signals including textual profiles, interaction histories, and tabular attributes into compact representations that enable robust user understanding in digital marketing Zhang et al. (2024), recommendation Feng et al., and personalization systems Dou et al. (2025); Gao et al. (2025). Existing works leverage self-supervised contrastive learning to align augmented views of user activity via contextual consistency within sequences Oord et al.

- (2018); Lin et al. (2022) or cross-view coherence across sessions Zou et al. (2022). Yet as behaviors grow increasingly sequential and context-sensitive, effective representations must anticipate future actions Dou et al. (2025), demanding stronger semantic integration and long-range reasoning. Bidirectional pre-trained language models (PLMs) address this via full self-attention Devlin et al.
- (2019); Raffel et al. (2020), enabling holistic embeddings that have dominated general-purpose Wang et al. (2024a) and user-centric tasks Sun et al. (2019); Dou et al. (2025). However, their static, batch-oriented design requires the full context upfront—making them impractical for interactive settings where signals arrive incrementally Dou et al. (2025). Decoder-only large language models

(LLMs), by contrast, support autoregressive interaction, and recent work shows they can be effective for user modeling when adapted with contrastive objectives Zhang et al. (2025); Gao et al. (2025). Crucially, such adaptation hinges on the attention masking strategy: while decoder-only LLMs are pretrained with causal attention, they can be trained and evaluated under three distinct recipes—(i) Causal: standard unidirectional mask Zhang et al. (2025); (ii) Bidirectional: full self-attention over the entire input Hu et al. (2025); Li et al. (2025); (iii) Hybrid: bidirectional attention over a designated user segment followed by causal attention for downstream tokens. Despite their prevalence, no study systematically compares how these masking choices affect user representation quality under a unified contrastive training framework.

To address this gap, we systematically investigate the role of attention masking and its training dynamics when adapting decoder-only LLMs for user representation learning. Rather than treating masking as a fixed design choice, we highlight the importance of the transition from causal to bidirectional attention and propose a practical warm-up mechanism to stabilize this process within a contrastive learning framework. We evaluate our approach on twelve discriminative user understanding benchmarks derived from Alipay’s real-world user cognition system. Our key contributions are summarized as follows:

- • We conduct a unified empirical study of causal, hybrid, and bidirectional attention masks for LLM-based user representation learning under a controlled contrastive framework.
- • We demonstrate that the training transition from causal to bidirectional attention is a key factor affecting optimization stability and representation quality.
- • We propose Gradient-Guided Soft Masking (GG-SM) as a gradient-informed pre-warmup that facilitates a smoother causal-to-bidirectional transition and leads to stronger final bidirectional representations evaluated on 9 user-centric classification benchmarks.

#### 2 Related Work

LLM for User Embedding. Large language models (LLMs) are increasingly employed for user representation learning due to their ability to integrate behavioral sequences, textual profiles, and structured attributes into unified embeddings. Encoder-based models such as BERT4Rec Sun et al. (2019) and FOUND Dou et al. (2025) treat user histories as pseudo-sentences and capture rich contextual dependencies, but their bidirectional attention necessitates full input visibility and limits applicability in streaming or interactive scenarios. Decoder-only LLMs overcome this limitation via autoregressive processing, enabling continual updates and dynamic context integration. Recent systems, including Qwen3-embedding Zhang et al. (2025) and InstructUE Gao et al. (2025), adapt causal LLMs for embedding tasks through contrastive or instruction-based objectives, yet the impact of attention masking strategies remains underexplored. Existing approaches follow one of three paradigms: causal masking, ensuring compatibility with generative inference; bidirectional masking, maximizing representational completeness but forfeiting autoregressiveness; and hybrid masking, combining bidirectional attention within the history block with causal attention thereafter. Conan Li et al. (2025) introduces a progressive scheduler transitioning from causal to bidirectional masking, narrowing the gap between pretraining dynamics and embedding requirements. However, no systematic comparison across these strategies under identical training conditions exists. We fill this gap via large-scale evaluation on 9 real-world user cognition benchmarks, finding that bidirectional masking yields the highest representational quality, while hybrid masking offers the best trade-off between completeness and generative compatibility.

Synthetic Data for User Embedding. High-quality labeled data for user modeling remains scarce, motivating growing interest in synthetic data generation. Early methods relied on heuristic augmentation or retrieval-based pseudo-labels Nogueira and Cho (2019). Recent approaches leverage large language models to generate realistic behavior traces or user intents Gao et al. (2025). However, most pipelines depend on proprietary APIs such as GPT-4 Choi et al. (2024); Chen et al. (2025); Yuan et al. (2025), which raises concerns about cost, reproducibility, and domain alignment. Alternatives based on small open-source LLMs often suffer from low fidelity due to insufficient semantic alignment with target user behaviors Wang et al. (2024b). To improve the quality and scalability of hard positive samples in training data, we propose a training-free synthesis framework that directly leverages an off-the-shelf base LLM to probe hard-to-align user&query–answer pairs. By applying post-hoc chain-of-thought reasoning, we identify the underlying difficult patterns in these pairs and use these insights to refine prompt for QA synthesis, enabling scalable generation of high-fidelity synthetic hard positives.

#### 3 Training Data

Following Dou et al. (2025); Gao et al. (2025), we contrust and employ two types of alignment data for embedding training based on real-world Alipay user interactions: (1) Rule-based Behavioral Trajectories Dataset Dbehavior: Composed of user behavior sequences, Dbehavior = {ui, bi}iN=1, where bi denotes the user i’s real future behavior. (2) LLM-synthesized Query–Answer Alignments Dataset Dqa: Represents user intent and language understanding, denoted as Dqa = {ui ⊕ qi, ai}, where qi is the query generated from ui, and ai is the corresponding LLM-generated answer. Here, ui = {Billi, Minii, Spmi, Appi, Searchi, Tabulari} ∈ U represent the user i’s multi-modal interaction profile over the past 90 days, where Billi denotes PayBill transactions, Minii represents Mini Program interactions, Spmi represents super position model (SPM) paths Si refers to superposition model paths, Si captures search queries, and Ti ∈ RF×D includes tabular features with F features and D-dimensional embeddings.

###### 3.1 Rule-based Behavioral Trajectories Dataset

Following Dou et al. (2025), we construct behavioral trajectory pairs using a rule-based alignment strategy. The left tower encodes the user’s raw interaction sequence over the past three months. For the right tower, we first aggregate all interactions from the subsequent one-month window (e.g., by action type or temporal bins), then randomly sample a representative subset to serve as the future prediction target. This aggregated-and-sampled future signal is aligned with the historical sequence during embedding training.

###### 3.2 LLM-Synthesized Query-Answer Alignments Dataset

Building on the insights from Robinson et al.; Lee et al. (2024) that challenging negative samples can enhance embedding learning, we extend this idea by focusing on generating challenging positive samples as anchors through a post-rule improvement mechanism, optimizing data synthesis by pregenerating challenging query-answer pairs to avoid the embedding-based real-time post-mining of negative samples constrained by data quality Gao et al. (2025) and computational limitations Li et al. (2025) during training.

- Step (1) Synthesis Pipeline and Calibration Set Generation. We begin by initializing our synthesis pipeline with Qwen-Max to generate diverse user-understanding scenarios as a seed pool Pool for

the subsequent synthesis of varied and generalizable QA pairs. Given each user i, we prompt LLM LLM to retrieve the top 10 most relevant seed scenario seedtop10 according to user behavior history ui via Pretrieve and then instantiate ui and seedtop10 to synthesize QA pairs that reflect diverse user understanding through prompting LLM with Pqa. From these, we construct a calibration set Dc of 1,000 user&query-answer pairs {(ui ⊕ qi, ai)}1000i=1 , ensuring diverse coverage of user behavior topics, formally:

seedtop10(ui, P) = LLM(ui, Pool, Pretrieve), (1) Dc = {LLM(seedtop10, ui, Pqa)}1000j=1 . (2)

- Step (2) Alignment Difficulty Probing on Dc. Inspired by difficulty probing in Team (2024), for each user&query-answer pair (ui ⊕ qi, ai), we evaluate its alignment difficulty by computing the

similarity between ui ⊕ qi and ai as hard-to-align score Sd via a strong and size-efficient embedding model Emb 1, formally:

Sd = 1 − Sim(Emb(ui ⊕ qi), Emb(ai)) (3) where Sim(v1, v2) represents the cosine similarity between v1 and v2, computed as: Sim(v1, v2) =

v1·v2

∥v1∥∥v2∥. Higher Sd values indicate more difficult alignments, meaning the pair (ui ⊕ qi, ai) is harder to align even for a strong original model. We further set a threshold Tfilter to retain only the challenging samples Dhard = {(ui ⊕ qi, ai) | Sd ≥ Tfilter} with high alignment difficulty.

- Step (3) Inductive Feature Completion. Inspired by post-cot interpretation for feature interpretation Singh et al. (2024), for the remaining challenging pairs, we apply an inductive feature completion rule using Qwen-Max to extract common rules Prule from hard-to-align positive QA pairs Dhard and further integrate them into prompt Pqa for qa synthesis as described in Eq. 1.
- Step (4) Scaling and Posterior Rewriting. After enriching the challenging query-answer pairs,

we scale the step (1) by applying the optimized Pqa to generate a larger set of query-answer pairs following the input template specified in Appendix A, which standardizes modality delimiters, instruction formatting, and the placement of the special <USER> token. These pairs undergo posterior rewriting to ensure enhanced clarity, context alignment, and semantic consistency with the user’s historical behavior. This step refines the dataset, preparing the pairs for embedding model training and improving their alignment with real-world user interactions.

#### 4 LLMs as Encoders: Training Recipe

Training Architecture. As illustrated in Figure 1, our framework follows Dou et al. (2025) to process user pair ui ⊕ qi via modality-specific encoders whose outputs are projected into the LLM’s (M) embedding space via lightweight adapters. In parallel, the corresponding answer ai through the same decoder-only LLM M as a dual-tower alignment architecture. Both towers share the LLM backbone but operate independently during encoding, enabling efficient, modalityaware representation learning while maintaining compatibility with the LLM’s token semantics for downstream contrastive alignment. Implement details are provided in Sec. 5 and representation learning procedures are deferred to Appendix A.

1https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

[Figure 2]

Figure 1 Architechure Overview of Our Find-Embedding (w / GGSM).

Gradient-Guided Soft Masking. To endow causal LLMs with bidirectional reasoning capabilities during encoding, we extend the causal-to-bidirectional scheduler of Li et al. (2025) with a gradientguided warmup phase. Let Twarm and Ttotal denote the warmup and total training steps, respectively. For a sequence of user&query length L with hidden states H = [h1, . . . , hL] ∈ RL×d, we define the soft attention mask Msoft(t) ∈ RL×L at training step t as:

0 if j ≤ i, log wij(t) if j > i,

Mijsoft(t) =

(4)

σ ∥∇hjL∥ if t < Twarm, (1 − αt) · σ ∥∇hjLwarm∥ + αt if Twarm ≤ t < Ttotal.

(5)

wij(t) =

where αt = T t−Twarm

total−Twarm ∈ [0,1] and Lwarm denotes the loss computed at the final warmup step t = Twarm −1, and σ(·) is the sigmoid function ensuring wij(t) ∈ (0, 1]. During warmup (t < Twarm), future attention weights are set adaptively via the instantaneous gradient norm ∥∇hjL∥: tokens that strongly influence the loss receive higher visibility. At the end of warmup, these gradientderived weights are frozen. In the scheduler phase (t ≥ Twarm), we linearly interpolate between the frozen soft mask and full bidirectionality (i.e., wij = 1), and at inference, M employs a fully bidirectional attention mask to maximize contextual integration, thereby better bridge the gap between token-level pretraining with sentence-level representation learning Li et al. (2025).

Training Objective. We employ contrastive learning, following Li et al. (2023); Zhang et al. (2025), to align user and answer embeddings. The goal is to learn discriminative user embeddings by pulling semantically related user-answer pairs closer and pushing apart negative samples, facilitating comprehensive user profile extraction and accurate future action prediction. The objective is defined using the InfoNCE loss Lcl across a batch of size B, formally:

B

1 B

### ∑

Lcl = −

i=1

es(uˆi,aˆi+)/τ Zi

log

, (6)

where uˆi and aˆi are the normalized embeddings of user i and its answer. s(uˆi, aˆi) is the cosine similarity between the user and answer embeddings, and τ controls the similarity smoothness. Zi

is the normalization factor, aggregating positive and negative pair similarities:

Zi =es(uˆi,aˆi+)/τ + ∑

mijes(uˆi,aˆj)/τ

j̸=i

mijes(uˆi,uˆj)/τ + ∑

mijes(aˆi,aˆj)/τ, (7)

### + ∑

j̸=i

j̸=i

where ai+ and aj are the positive and other in-batch answer embeddings, respectively, and uj represents other in-batch user embeddings. To mitigate the effect of false negatives, we follow

Zhang et al. (2025) and introduce a mask factor mij, which ensures that negative samples are sufficiently distinct. The mask factor is computed as:

mij =

- 0 if sij > s(uˆi, aˆi+) + cmargin,
- 1 otherwise.

(8)

where sij represents the similarity between user embeddings uˆi, uˆj or mixed embeddings uˆi, aˆj, and cmargin is a margin hyperparameter that ensures adequate separation between positive and negative pairs. Incorporating same-side negative samples enhances the distinctiveness of user representations across different instructions and improves the separability of answer embeddings, ultimately boosting model performance in embedding-based tasks.

#### 5 Experiments

Models and Implementation. For training data, we follow our pipeline (Sec. 3.2) via Qwen3-30B-A3B 2 Team (2025) for efficiency. For training architecture, we adopt dedicated instances of gte-base-zh Li et al. (2023) to encode heterogeneous behavioral inputs into modality-specific embeddings, which are concatenated and prepended to the input of Qwen2.5-0.5B-Instruct Team (2024), serving as the LLM backbone for contrastive user representation learning. And we fine-tune this decoder-only LLM under a contrastive learning objective with distinct attention masking strategies (detailed in Sec. 4), using identical training configurations across all variants: a global batch size of 2,048, 7w fine-tuning steps, an AdamW optimizer with initial learning rate 2 × 10−4 and cosine decay, LoRA Hu et al. with rank=64 and α=64. All experiments are trained on 64 A100-80GB GPUs using data parallelism, while inference is performed on single A100-80GB GPU for subsequent evaluation.

Baselines and Tasks. We select Qwen2.5-0.5B-Instruct as the oracle backbone and compare its performance under three attention mask training recipes: (1) Causal: contrastive learning with the original causal attention mask; (2) Hybrid: three strategies for opening the upper triangular attention matrix: (a) gradient-guided soft masking using left tower gradients to compute importance scores that control the future mask, (b) applying an MLP for direct attention opening, and (c) introducing a global query in a CLS-like fashion to guide attention; (3) Bidirectional: (a) contrastive learning with the bidirectional mask, and transitioning from unidirectional to bidirectional training via (b) scheduler or (c) gradient-guided soft mask pre-warmup and scheduling (Ours). All recipes are detailed in Appendix A.4. Additionally, we evaluate inference performance using the top three embedding models 3 from the MTEB leaderboard including KaLM-Embedding-Gemma3-12B-2511 Hu et al. (2025), llama-embed-nemotron-8b Babakhin et al. (2025), Qwen3-Embedding-8B Zhang et al. (2025). For broader comparison, we also include representative traditional user modeling

2https://huggingface.co/Qwen/Qwen3-30B-A3B 3Valid during the working period until December 31, 2025.

baselines, where U-MLP One4all Shin et al. (2021) extends a general-purpose One4all representation with an additional MLP decoder for user targeting, while MSDP Fu et al. (2023) and CPC Oord et al. (2018) adopt contrastive learning to learn robust user representations from augmented views of behavior sequences, together with LLM-based user representation models such as FOUND Dou et al. (2025). All models are evaluated under consistent training hyperparameters, with the evaluation performed on a binary classification task across 9 real-world Alipay user scenarios, as listed in Table 1:

DatasetDomain Scenario Number Dtrain General (3.1 & 3.2) General ≈ 1.433 × 108

❶ User Prediction Concert Click Prediction (Concert), User Log-in Prediction (User), MAU Loss Prediction (MAU)

≈ 50w per task

Dtest

❷ Behavior Preference Public Transit Preference (Transit), Consumption Power (Power), Food Interest (Food), Movie Interest (Movie)

≈ 50w per task

❸ Marketing Sensitivity Achievement Preference (Achiev.), Physical Preference (Physical)

≈ 50w per task

Table 1 Data information for user pretraining and test benchmarks, with number of tests per task.

Evaluation Metrics. We assess user representations via linear probing on 9 annotated binary classification tasks from Alipay’s user cognition system, reporting AUC (Area Under the ROC Curve Bradley (1997)) for discriminative performance.

#### 6 Main Results

[Figure 3]

Figure 2 Average AUC performance across 9 downstream tasks under different attention masking strategies (left) and comparison with general embedding, user embedding (right).

In this section, we evaluate the effectiveness of our proposed GG-SM training strategy (Sec. 4) across 9 downstream user-centric tasks. Figure 2 illustrates the average AUC across tasks under different attention masking strategies (left) and a comparison with general embeddings, user

User Prediction Behavior Preference Marketing Sensitivity Method Concert User MAU Transit Power Food Movie Achiev. Physical Avg General Embedding Models

Qwen3-Embedding-0.6B 0.5226 0.7294 0.9197 0.6098 0.8078 0.6656 0.6641 0.5529 0.5759 0.6720 Llama-embed-nemotron 0.5627 0.7735 0.9351 0.6915 0.9308 0.7936 0.7692 0.5879 0.5768 0.7357 KaLM-Embedding 0.5359 0.7609 0.9272 0.6400 0.8623 0.7443 0.7812 0.5787 0.6099 0.7156

###### User Embedding Models

MSDP Fu et al. (2023) 0.5155 0.9504 0.9633 0.6367 0.8480 0.7928 0.7645 0.6151 0.5900 0.7418 One4all Shin et al. (2021) 0.5568 0.9509 0.9639 0.6276 0.8393 0.7984 0.7526 0.6016 0.5957 0.7430 CPC Oord et al. (2018) 0.5314 0.9506 0.9654 0.6376 0.8415 0.8009 0.7526 0.6256 0.5952 0.7445 FOUND Dou et al. (2025) 0.5670 0.8330 0.9574 0.6824 0.9669 0.8513 0.8472 0.6102 0.6059 0.7690 InstructUE Gao et al. (2025) 0.5712 0.8394 0.9661 0.6964 0.9695 0.8534 0.7927 0.6071 0.6594 0.7728

###### Qwen2.5-0.5B-Instruct (Causal)

Oracle 0.5173 0.7219 0.9202 0.5642 0.7638 0.6561 0.6435 0.5415 0.5592 0.6542 w/ Causal 0.5716 0.8313 0.9669 0.6967 0.9678 0.8473 0.7922 0.6054 0.6589 0.7709

###### Qwen2.5-0.5B-Instruct (Hybrid)

w/ Hybridmask 0.5748 0.8311 0.9671 0.6951 0.9653 0.8520 0.7913 0.6056 0.6565 0.7710 w/ Hybridgq 0.5647 0.8382 0.9665 0.6945 0.9678 0.8528 0.7887 0.6044 0.6582 0.7706 w/ Hybridmlp 0.5750 0.8410 0.9667 0.6965 0.9649 0.8484 0.7886 0.6042 0.6608 0.7718 Qwen2.5-0.5B-Instruct (Bidirectional)

w/ Bidirectional 0.5707 0.8390 0.9673 0.6983 0.9671 0.8505 0.7906 0.6043 0.6607 0.7721 w/ Scheduler 0.5742 0.8419 0.9664 0.6973 0.9688 0.8540 0.7908 0.6056 0.6605 0.7733 w/ GG-SM (Ours) 0.5767 0.8438 0.9674 0.6978 0.9689 0.8554 0.7913 0.6078 0.6615 0.7745

Table 2 Comparison of AUC performance for general embeddings, user embeddings, and our GG-SM method across all downstream tasks.

embeddings, Oracle, and Ours (right). Table 2 provides detailed numerical results across three major domains: User Prediction, Behavior Preference, and Marketing Sensitivity.

Parameter Efficiency and Domain-Specific Alignment. A primary finding from Table 2 is that our GG-SM-enhanced Qwen2.5-0.5B-instruct achieves an average AUC of 0.7745, consistently outperforming massive general-purpose embeddings such as Llama-embed-nemotron (0.7357) and KaLM-Embedding (0.7156). With significantly fewer parameters, GG-SM still outperforms on task-specific metrics (Transit: 0.6978; Power: 0.9689), verifying that raw parameter scale yields diminishing returns when applied to industrial behavioral logs with high sparsity and non-linguistic distributions. While 8B+ models possess broader natural language priors, they introduce redundant noise in discrete behavioral sequences. In contrast, GG-SM maximizes information extraction density, proving that gradient-based attention calibration is more critical than raw scaling for aligning an LLM’s latent space with domain-specific behavioral structures.

From Local Contrast to Contextual Priors. The results highlight a decisive performance gap between traditional user modeling and LLM-based approaches. Traditional baselines like MSDP, One4all, and CPC excel in specific tasks like Achiev. (0.6256), likely due to their effective capture of local feature matches. However, they struggle with tasks requiring global contextual transfer, such as Food or Movie preferences. Comparing our model against recent LLM-based baselines like FOUND (0.7690) and InstructUE (0.7728), we observe that GG-SM provides more consistent gains. While these models utilize LLMs as static extractors, GG-SM treats the attention mechanism as an evolvable bottleneck, effectively leveraging pre-trained contextual priors to model long-range user dependencies more holistically.

Efficacy of Gradient-Guided Attention Evolution. The internal comparison of masking strategies reveals that the path to bidirectionality determines the quality of the final embedding. Standard Causal masks (Oracle) are too restrictive for representation tasks, while Hybrid strategies (Hybridmask, Hybridgq, Hybridmlp) provide only marginal gains as they introduce additional parameters that are difficult to align with a frozen or pre-trained backbone. GG-SM outperforms both naive Bidirectional and Scheduler-based methods by using instantaneous gradient norms as a dynamic signal for token importance. This ensures that the model does not merely see more tokens, but learns to prioritize the most informative ones during the crucial early stages of bidirectional adaptation as shown in Fig. 3. As evidenced by the Behavior Preference results, this leads to a significantly sharper separation of user interests compared to static masking recipes.

[Figure 4]

Figure 3 Training loss convergence: GG-SM (Ours) vs. scheduler.

Domain Robustness and Transferability. The robustness of GG-SM is evidenced by its consistent lead across three distinct domains: User Prediction, Behavior Preference, and Marketing Sensitivity. While traditional contrastive models often exhibit high variance—e.g., performing well in high-frequency behavior tasks but degrading in sensitivity tasks—our model maintains a stable performance advantage. Notably, in the Marketing Sensitivity domain, where latent intent is hardest to capture, GG-SM achieves peak AUC. This suggests that guiding the attention mechanism through the model’s own internal learning pressure (gradients) captures more transferable user traits than manually engineered data augmentations or fixed architectural modifications.

#### 7 Conclusion

In this work, we revisit user representation learning with decoder-only LLMs through the lens of attention masking, systematically comparing causal, hybrid, and bidirectional masks under a unified contrastive framework on large-scale real-world Alipay data and 9 industrial user-centric tasks. We show that not only the final mask but also the transition path from causal to bidirectional modeling critically affects training stability and embedding quality. To this end, we introduce Gradient-Guided Soft Masking as a pre-warmup before a linear scheduler, which consistently improves optimization behavior and yields stronger bidirectional representations while remaining compatible with decoder pretraining. Overall, our findings highlight that careful masking design and transition dynamics are key to effectively adapting decoder-only LLMs as practical user encoders.

#### References

Yauhen Babakhin, Radek Osmulski, Ronay Ak, Gabriel Moreira, Mengyao Xu, Benedikt Schifferer, Bo Liu, and Even Oldridge. Llama-embed-nemotron-8b: A universal text embedding model for multilingual and cross-lingual tasks. arXiv preprint arXiv:2511.07025, 2025.

Andrew P Bradley. The use of the area under the roc curve in the evaluation of machine learning algorithms. Pattern recognition, 30(7):1145–1159, 1997.

Haonan Chen, Liang Wang, Nan Yang, Yutao Zhu, Ziliang Zhao, Furu Wei, and Zhicheng Dou. Little giants: Synthesizing high-quality embedding data at scale. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1392–1411, 2025.

Chanyeol Choi, Junseong Kim, Seolhwa Lee, Jihoon Kwon, Sangmo Gu, Yejin Kim, Minkyung Cho, and Jy-yong Sohn. Linq-embed-mistral technical report. arXiv preprint arXiv:2412.03223, 2024.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.

Bin Dou, Baokun Wang, Yun Zhu, Xiaotong Lin, Yike Xu, Xiaorui Huang, Yang Chen, Yun Liu, Shaoshuai Han, Yongchao Liu, et al. Transferable and forecastable user targeting foundation model. In Companion Proceedings of the ACM on Web Conference 2025, pages 181–190, 2025.

Ningya Feng, Junwei Pan, Jialong Wu, Baixu Chen, Ximei Wang, Xian Hu, Jie Jiang, Mingsheng Long, et al. Longsequence recommendation models need decoupled embeddings. In The Thirteenth International Conference on Learning Representations.

Chilin Fu, Weichang Wu, Xiaolu Zhang, Jun Hu, Jing Wang, and Jun Zhou. Robust user behavioral sequence representation via multi-scale stochastic distribution prediction. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, pages 4567–4573, 2023.

Ziyi Gao, Yike Xu, Jiahao Yuan, Baokun Wang, Jinyong Wen, Xiaotong Lin, Yun Liu, Xing Fu, Yu Cheng, Yongchao Liu, et al. Instruction-aware user embedding via synergistic language and representation modeling. arXiv preprint arXiv:2510.11016, 2025.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Xinshuo Hu, Zifei Shan, Xinping Zhao, Zetian Sun, Zhenyu Liu, Dongfang Li, Shaolin Ye, Xinyuan Wei, Qian Chen, Baotian Hu, et al. Kalm-embedding: Superior training data brings a stronger embedding model. arXiv preprint arXiv:2501.01028, 2025.

Unggi Lee, Sungjun Yoon, Joon Seo Yun, Kyoungsoo Park, YoungHoon Jung, Damji Stratton, and Hyeoncheol Kim. Difficulty-focused contrastive learning for knowledge tracing with a large language model-based difficulty prediction. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 4891–4900, 2024.

Shiyu Li, Yang Tang, Ruijie Liu, Shi-Zhe Chen, and Xi Chen. Conan-embedding-v2: Training an llm from scratch for text embeddings. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 15011–15027, 2025.

Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281, 2023.

Guanyu Lin, Chen Gao, Yinfeng Li, Yu Zheng, Zhiheng Li, Depeng Jin, and Yong Li. Dual contrastive network for sequential recommendation. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval, pages 2686–2691, 2022.

Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with bert. arXiv preprint arXiv:1901.04085, 2019. Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv

preprint arXiv:1807.03748, 2018.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

Joshua David Robinson, Ching-Yao Chuang, Suvrit Sra, and Stefanie Jegelka. Contrastive learning with hard negative samples. In International Conference on Learning Representations.

Kyuyong Shin, Hanock Kwak, Kyung-Min Kim, Minkyu Kim, Young-Jin Park, Jisu Jeong, and Seungjae Jung. One4all user representation for recommender systems in e-commerce. arXiv preprint arXiv:2106.00573, 2021.

Chandan Singh, Jeevana Priya Inala, Michel Galley, Rich Caruana, and Jianfeng Gao. Rethinking interpretability in the era of large language models. arXiv preprint arXiv:2402.01761, 2024.

Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management, pages 1441–1450, 2019.

- Qwen Team. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.
- Qwen Team. Qwen3 technical report, 2025. https://arxiv.org/abs/2505.09388.

Jiajia Wang, Jimmy Xiangji Huang, Xinhui Tu, Junmei Wang, Angela Jennifer Huang, Md Tahmid Rahman Laskar, and Amran Bhuiyan. Utilizing bert for information retrieval: Survey, applications, resources, and challenges. ACM Computing Surveys, 56(7):1–33, 2024a.

Ke Wang, Jiahui Zhu, Minjie Ren, Zeming Liu, Shiwei Li, Zongye Zhang, Chenkai Zhang, Xiaoyu Wu, Qiqi Zhan, Qingjie Liu, et al. A survey on data synthesis and augmentation for large language models. arXiv preprint arXiv:2410.12896, 2024b.

Jiahao Yuan, Zhiqing Cui, Hanqing Wang, Yuansheng Gao, Yucheng Zhou, and Usman Naseem. Kardia-r1: Unleashing llms to reason toward understanding and empathy for emotional support via rubric-as-judge reinforcement learning. arXiv preprint arXiv:2512.01282, 2025.

Wei Zhang, Dai Li, Chen Liang, Fang Zhou, Zhongke Zhang, Xuewei Wang, Ru Li, Yi Zhou, Yaning Huang, Dong Liang, et al. Scaling user modeling: Large-scale online user representations for ads personalization in meta. In Companion Proceedings of the ACM Web Conference 2024, pages 47–55, 2024.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.

Ding Zou, Wei Wei, Xian-Ling Mao, Ziyang Wang, Minghui Qiu, Feida Zhu, and Xin Cao. Multi-level cross-view contrastive learning for knowledge-aware recommender system. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval, pages 1358–1368, 2022.

#### A Details of Representation Training

###### A.1 Standardized Input Template

The following presents heterogeneous user data collected from multiple sources, including PayBill transactions (Bi), Mini Program interaction logs (Mi), Super Position Model paths (Si), App interaction records (Ai), homepage search queries (Ri), and structured tabular features (Ti): <bill> { Bill data Bi } </bill>

<minipro> { Mini Program logs Mi } </minipro> <spm> { Super Position Model paths Si } </spm> <app> { App interaction records Ai } </app> <search> { Search queries Ri } </search> <tabular> { Tabular features Ti ∈ RF×D } </tabular> Instruction: { optional user query qi } <USER>

Figure 4 Input format of Training Data. <USER> token serves as the anchor for extracting user embedding.

To ensure consistency across modalities and reproducibility of representation learning, we adopt a unified input template for all synthesized and real-world alignment data. As illustrated in Figure 4, each user instance ui is represented as a heterogeneous sequence of multimodal records collected over the past 90 days, formally:

ui = {Billi, Minii, Spmi, Appi, Searchi, Ti} ∈ U, (9)

where each modality is enclosed by explicit semantic boundary tokens, e.g., <bill>...</bill>, <minipro>...</minipro>, etc., to preserve modality structure and facilitate modality-aware encoding.

For each user instance, a (optional) user instruction qi may be appended after the user profile. The complete model input is formulated as:

xi = ui ⊕ [qi] ⊕ <USER>, (10)

where ⊕ denotes sequence concatenation, and [qi] indicates that the instruction is optional. The special token <USER> signals the model to aggregate all preceding multimodal information (and the instruction, if provided) into a unified user representation.

###### A.2 User Embedding Extraction

We adopt a decoder-only causal LLM M as the backbone encoder. Given an input sequence xi of length L, the model produces a sequence of final-layer hidden states:

Hi = [h1, . . . , hL] = M(xi), (11) where ht ∈ Rd denotes the hidden state of the t-th token. Let tuser denote the position of the special token <USER> in xi. The unified user embedding is defined as the corresponding hidden state:

uˆi = htuser. (12) To ensure stable contrastive training, we apply L2 normalization:

###### A.3 Answer Embedding Extraction

uˆi ∥uˆi∥2

u˜i =

. (13)

For each answer ai, we feed it independently into the same LLM backbone M and append an end-of-sequence token <EOS>. The answer embedding is extracted analogously:

aˆi ∥aˆi∥2

aˆi = htEOS, a˜i =

. (14)

These normalized embeddings u˜i and a˜i are then used for contrastive alignment via the InfoNCE objective in Sec. 4.

###### A.4 Training Recipes Across Causal, Hybrid, and Bidirectional Masking

We release the complete implementation details for all baselines to ensure transparency and reproducibility in this technique report, with particular emphasis on our hybrid masking variants. We explicitly frame the hybrid approach as user-centric, designed to better capture contextual directionality in user representation learning. We regard hybrid masking—especially the progressive hybrid-to-bidirectional transition—as a promising research direction, and we encourage the community to further explore and advance this paradigm. Below, w e outline the exact definitions:

Causal Masking. The causal masking strategy employs standard autoregressive attention, where each token ti attends only to tokens {tj | j ≤ i}. Formally, the attention mask Mcausal ∈ RL×L for a sequence of length L is defined as:

0 if j ≤ i, −∞ otherwise.

Mijcausal =

This enforces strict left-to-right information flow, preserving compatibility with generative inference and the pretraining dynamics of decoder-only LLMs. We apply the contrastive learning objective directly on representations extracted from this causal encoder.

Hybrid Masking. Hybrid masking selectively relaxes causality over the user-history segment while maintaining causal constraints for future tokens. We implement three user-centric variants:

- (a) Gradient-Guided Soft Masking: During training, we compute importance scores for future positions using gradients from a frozen left-tower encoder as illustrated in Sec. 4.
- (b) MLP-Driven Attention Opening: A lightweight MLP predicts attention bias for j > i, dynamically enabling direct future token access based on hi.
- (c) Global-Query Guidance: A learnable [CLS]-like token qglobal attends bidirectionally to all history tokens; its attention weights supervise block-level contextual integration without violating causality for downstream generation.

Bidirectional Masking. Bidirectional masking grants full self-attention (all-to-all token visibility) and is instantiated in three ways:

- (a) Direct Bidirectional Contrastive: The model uses a fully unmasked attention matrix Mijbi = 0 for all i, j from initialization, trained with the same contrastive objective as other variants.
- (b) Scheduler-Based Transition: The attention span grows from causal to bidirectional via a deterministic schedule; e.g., at epoch e, the mask allows attention up to position min(i + ∆(e), L), where ∆(e) increases linearly or cosinely with e.
- (c) Gradient-Guided Soft-Mask Warm-Up (Ours): Building on the hybrid approach, we first warm up with gradient-derived soft masks (as in Hybrid (a)), then linearly interpolate toward full bidirectionality. Specifically, for step t ≥ Twarm, the future mask weight is:

t − Twarm Ttotal − Twarm

##### wij(t) = (1 − αt) · σ ∥∇hjLwarm∥ + αt, αt =

(15)

where Lwarm is the loss at the end of warm-up. This data-driven transition enables stable convergence to a fully bidirectional encoder while leveraging task-specific signal during adaptation.

