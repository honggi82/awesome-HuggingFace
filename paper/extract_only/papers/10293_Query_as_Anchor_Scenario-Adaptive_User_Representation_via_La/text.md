# arXiv:2602.14492v2[cs.CL]17Feb2026

## Query as Anchor: Scenario-Adaptive User Representation via Large Language Model

Jiahao Yuan1,2,∗, Yike Xu1,∗, Jinyong Wen1, Baokun Wang1,†, Ziyi Gao1, Xiaotong Lin1 Yun Liu1, Xing Fu1, Yu Cheng1, Yongchao Liu1, Weiqiang Wang1, Zhongle Xie3

1Ant Group, 2East China Normal University, 3Zhejiang University China

### Abstract

##### ACM Reference Format:

Jiahao Yuan1,2,∗, Yike Xu1,∗, Jinyong Wen1, Baokun Wang1,†, Ziyi Gao1, Xiaotong Lin1 and Yun Liu1, Xing Fu1, Yu Cheng1, Yongchao Liu1, Weiqiang Wang1, Zhongle Xie3. 2018. Query as Anchor: Scenario-Adaptive User Representation via Large Language Model. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 15 pages. https: //doi.org/XXXXXXX.XXXXXXX

Industrial-scale user representation learning requires balancing robust universality with acute task-sensitivity. However, existing paradigms primarily yield static, task-agnostic embeddings that struggle to reconcile the divergent requirements of downstream scenarios within unified vector spaces. Furthermore, heterogeneous multi-source data introduces inherent noise and modality conflicts, degrading representation. We propose Query-as-Anchor, a framework shifting user modeling from static encoding to dynamic, queryaware synthesis. To empower Large Language Models (LLMs) with deep user understanding, we first construct UserU, an industrialscale pre-training dataset that aligns multi-modal behavioral sequences with user understanding semantics, and our Q-Anchor Embedding architecture integrates hierarchical coarse-to-fine encoders into dual-tower LLMs via joint contrastive-autoregressive optimization for query-aware user representation. To bridge the gap between general pre-training and specialized business logic, we further introduce Cluster-based Soft Prompt Tuning to enforce discriminative latent structures, effectively aligning model attention with scenario-specific modalities. For deployment, anchoring queries at sequence termini enables KV-cache-accelerated inference with negligible incremental latency. Evaluations on 10 Alipay industrial benchmarks show consistent SOTA performance, strong scalability, and efficient deployment. Large-scale online A/B testing in Alipay’s production system across two real-world scenarios further validates its practical effectiveness. Our code is prepared for public release and will be available at: https://github.com/JhCircle/Q-Anchor.

### 1 Introduction

User representation learning underpins modern industrial intelligence systems by enabling personalized, data-driven decision making across applications such as recommendation [5], digital marketing [31], and risk management [7]. In practice, users frequently engage in multiple business scenarios, each characterized by distinct behavioral patterns and decision objectives, which calls for representations that are both transferable across tasks and adaptable to scenario-specific contexts. Accordingly, existing systems typically compress large-scale, heterogeneous user signals, including textual profiles, interaction histories, and structured attributes, into compact embeddings to support downstream modeling and inference.

Most existing user embedding methods learn static, task-specific representations, either trained from scratch [8] or based on pretrained language model (PLM) encoders optimized with contrastive objectives [22, 25]. While effective at capturing historical behavioral patterns, these approaches are inherently retrospective and lack the flexibility to adapt to diverse downstream decision scenarios. As a result, industrial systems often rely on multiple task-specific user models for different applications, increasing system complexity, deployment overhead, and long-term maintenance cost, while still struggling with cross-domain generalization [15].

### CCS Concepts

• Information systems → Collaborative filtering.

### Keywords

Recent research has explored large language models (LLMs) as user encoders, enhancing sequential and semantic understanding through fine-tuning on behavior sequences [20, 25] or by enriching user and item semantics [12]. However, real-world user behaviors are typically sparse, symbolic, and highly heterogeneous, which diverges significantly from the dense, language-centric data used in LLM pretraining [1, 13, 32]. This modality and semantic gap limits the ability of LLMs to generate adaptive and anticipatory user representations, especially in specialized domains where behavior distributions differ substantially from pretraining corpora. While instruction-aware embeddings [32] offer limited task conditioning, they remain insufficient for capturing the complex, dynamic, and noisy nature of industrial user data.

User representation modeling, User Embedding, Large language model

*Both authors contributed equally to this research. This work was completed during Jiahao’s research internship at Ant Group. †Corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Overall, existing user representation learning methods face three key challenges: (1) Limited scenario adaptability and cross-task generalization: Static embeddings cannot flexibly support diverse

Conference acronym ’XX, Woodstock, NY © 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/XXXXXXX.XXXXXXX

downstream tasks such as credit assessment, marketing optimization, and risk management, each of which requires scenario-specific user understanding [26, 33]. (2) Modality and semantic gap: The symbolic sparsity and heterogeneous structure of behavioral logs are misaligned with language-centric pretraining, constraining LLM-based user representations from effectively generalizing across domains. (3) Heterogeneous data integration at scale: Industrial user data varies significantly in relevance across scenarios, requiring mechanisms to selectively attend to scenario-relevant signals, suppress noise, and compress large-scale behaviors without introducing prohibitive inference overhead.

To address these challenges, we propose a unified user representation learning framework that integrates industrial-scale pretraining with efficient scenario adaptation. We introduce Queryas-Anchor, a query-conditioned mechanism that separates user behavior encoding from scenario-specific objectives. Our key contributions are summarized as follows:

- • We construct UserU, an industrial-scale pretraining dataset that aligns heterogeneous user behaviors with user understanding semantics via future behavior prediction and QAbased supervision, providing strong behavioral and semantic priors for user embedding learning.
- • We proposeQuery-as-Anchor,a query-conditionedframework that generates scenario-adaptive user embeddings by re-anchoring the same behavioral profile under different downstream decision contexts, enabling reuse across multiple business scenarios.
- • We introduce soft-prompt tuning and KV-cache–aware acceleration, enabling efficient scenario specialization and low-latency multi-scenario inference without retraining the backbone model.
- • Extensive offline experiments and large-scale online A/B testing on Alipay demonstrate consistent performance gains across user engagement, risk control, and marketing scenarios, while significantly reducing system complexity and deployment overhead.

### 2 Related Work

LLM for User Embedding. Large language models (LLMs) have demonstrated significant potential in user representation learning by integrating diverse data types, such as textual profiles, behavioral sequences, and structured attributes, into unified embeddings. Early works like BERT4Rec [25] and FOUND [7] fine-tuned LMs [23] on behavioral data to capture temporal dependencies via masked prediction and contrastive objectives [9], but they rely on static user embeddings [18], which limits adaptability to dynamic or evolving contexts. More recent instruction-aware models, such as Qwen3embedding [32] and KaLM-Embedding [13], have extended LLMs for task-specific embeddings [2, 10]. However, these models still face challenges when transitioning from general-purpose pretraining on large text corpora to sparse, symbolic, and highly contextual user behavior data, creating a gap in both structure and semantics. In contrast, our Query as Anchor framework bridges this gap by introducing dynamic, query-aware embeddings that adapt to the evolving nature of user behaviors, improving adaptability and contextual sensitivity across diverse tasks and domains.

[Figure 1]

Figure 1: Comparison between (A) General User Embedding [7] and (B) our Query-as-anchor. (A) learns transferable user representations across domains but generates fixed embeddings regardless of downstream context. (B) extends (A) with query-as-anchor modulation, enabling a single model to produce adaptive, domain-specific embeddings via natural language instructions.

Synthetic Data for User Embedding. Despite advancements in user representation learning, a significant gap remains due to the lack of a large-scale, comprehensive dataset specifically designed for user embedding pretraining. This limitation has spurred interest in generating synthetic data to supplement the training process. Early methods primarily focused on heuristic data augmentation and pseudo-labeling approaches [21, 29], which aimed to simulate user behaviors and interactions. More recently, large language models (LLMs) have been employed to generate realistic user behavior, intent patterns, and interaction sequences [7], offering scalable and diverse data generation capabilities. However, these techniques still face challenges in capturing the full complexity of user behaviors. The absence of a dedicated pretraining dataset for user embeddings remains a critical obstacle. To overcome this, we introduce UseU, a large-scale dataset designed specifically for user embedding pretraining. By incorporating rule-based future behavior prediction and QA-understanding tasks, UseU enables more effective, contextaware learning of user representations.

### 3 Methodology

We propose Q-Anchor Embedding, a large-scale pretraining framework for user understanding within the Alipay ecosystem. Here, u𝑖 = {𝐵𝑖𝑙𝑙𝑖,𝑀𝑖𝑛𝑖𝑖,𝑆𝑝𝑚𝑖,𝐴𝑝𝑝𝑖,𝑆𝑒𝑎𝑟𝑐ℎ𝑖,𝑇𝑎𝑏𝑢𝑙𝑎𝑟𝑖} ∈ U denotes user 𝑖’s multi-modal interaction profile over the past 90 days, where 𝐵𝑖𝑙𝑙𝑖 are PayBill transactions, 𝑀𝑖𝑛𝑖𝑖 are Mini Program interactions, 𝑆𝑝𝑚𝑖 are SPM navigation paths, 𝐴𝑝𝑝𝑖 are app-level list, 𝑆𝑒𝑎𝑟𝑐ℎ𝑖 are onplatform search queries, and𝑇𝑎𝑏𝑢𝑙𝑎𝑟𝑖 ∈ R1×𝐹 contains 𝐹 structured features. This unified profile constitutes the input to our UserU pretraining dataset (§ 3.1) and the Query-as-Anchor framework (§ 3.2), which aligns heterogeneous behaviors via query-driven anchors. To support efficient transfer, we further apply soft prompt tuning to adapt anchor embeddings to diverse downstream tasks with minimal parameter updates.

### 3.1 UserU Pretraining Dataset

To enhance user embedding performance across diverse tasks, we introduce the User Understanding (UserU) Pretraining Dataset, which integrates dynamic, context-aware user behavior with task

[Figure 2]

#### Figure 2: Overview of Query-as-Anchor Framework for our Q-Anchor Embedding.

adaptability for real-world applications. Extending future prediction pretraining [7], we augment it with synthesized query-answer pairs to capture deeper user understanding. UserU combines two key components: a behavior prediction dataset for future user actions 𝐷𝑓𝑢𝑡𝑢𝑟𝑒 and a LLM-generated UserQA dataset 𝐷𝑢𝑞𝑎 for comprehensive user representation via query-answer synthesis. Both data examples are detailed in Appendix A.

Behavior-basedInteractionDataset D𝑓𝑢𝑡𝑢𝑟𝑒. We constructabehaviorgrounded supervision signal by pairing each user’s historical profile with a future-action summary in a unified query–answer format (Appendix A). Concretely, for each user𝑖, we build a three-month behavior profile u𝑖 from multi-modal logs. We then derive a target 𝑎𝑖future by aggregating subsequent interactions into temporal bins and action categories, and selecting the frequency- and diversity-aware actions as the representative subset. Using a fixed template query 𝑞future (e.g., “What are the user’s most likely actions in the next period?”), we form training pairs D𝑓𝑢𝑡𝑢𝑟𝑒 = {(u𝑖 ⊕𝑞future, 𝑎𝑖future)}𝑖𝑁=1, where ⊕ denotes contextual concatenation. This dataset encourages the embedding to capture temporal regularities and encode predictive signals for near-future behaviors.

Synthetic Query-Answer Dataset D𝑢𝑞𝑎. To address the scarcity of user understanding training data and ensure better decoupling between pretraining data and downstream tasks for improved generalization, we propose a self-reflect synthetic data generation approach to construct a general-purpose user-query-answer alignment dataset. Inspired by the cold-start strategy in [6], we first initialize a seed pool P comprising 72 life-related user-understanding topics (e.g., financial planning, health management) via Qwen-Max. Given a user profile u𝑖, we prompt an LLM M to retrieve the top-10 most relevant topics from P, and instantiate each topic into a naturalistic query 𝑞𝑖 grounded in u𝑖. Conditioned on (u𝑖,𝑞𝑖), the LLM then generates an answer𝑎𝑖. To improve faithfulness and contextual

validity, we apply a post-generation reflection step in which M rechecks the draft answer against u𝑖 and revises unsupported or inconsistent statements. The resulting dataset is D𝑢𝑞𝑎 = {(u𝑖 ⊕𝑞𝑖,𝑎𝑖)}𝑖𝑀=1, where ⊕ denotes contextual concatenation.

### 3.2 Query as Anchor

3.2.1 Hierarchical Coarse-to-fine User Encoder. To effectively reconcile the inherent sparsity of multi-source behavioral signals with the dense semantic requirements of Large Language Models (LLMs), we propose a hierarchical encoding architecture that distills raw interactions into a multi-granularity representation space. Specifically, for each modality𝑚 ∈ M (where M = {𝐵𝑖𝑙𝑙,𝑀𝑖𝑛𝑖,𝑆𝑃𝑀,𝐴𝑝𝑝, 𝑆𝑒𝑎𝑟𝑐ℎ,𝑇𝑎𝑏𝑢𝑙𝑎𝑟}), raw event sequences are first projected into initial embeddings {h𝑚,𝑡} via encoder and subsequently refined by modality-specific event adapters:

𝑚 LayerNorm(h𝑚,𝑡) , (1) thereby preserving fine-grained atomic features of individual actions. To capture intra-modality trends while mitigating idiosyncratic noise, these event-level embeddings are aggregated through mean-pooling into a summary vector z¯𝑚(𝑒𝑣𝑡), which is further transformed by a shared modal adapter to yield a unified modality embedding z𝑚(𝑚𝑑𝑙) = MLP(𝑚𝑑𝑙)(LayerNorm(z¯𝑚(𝑒𝑣𝑡))). At the apex of this hierarchy, a global user-level representation z(𝑢𝑠𝑟) is derived by consolidating all modality-specific vectors through a dedicated user adapter, capturing the holistic behavioral profile. The final comprehensive input tokens e𝑖 is constructed via the structured concatenation of representations across all three levels:

z𝑚,𝑡(𝑒𝑣𝑡) = MLP(𝑒𝑣𝑡)

e𝑖 = z(usr) ; {z𝑚(mdl)}𝑚∈M ; {z¯𝑚(evt)}𝑚𝐾𝑚∈M , (2)

where [·; ·] denotes the concatenation operation and 𝐾𝑚 is the number of event tokens retained for modality 𝑚. This hierarchical design lets the LLM attend to either fine-grained events or high-level behavior summaries conditioned on the query, while remaining compatible with its native embedding space.

#### Input Template of UserU

The following are heterogeneous user data from multiple sources, including PayBill transactions, Mini Program interaction logs, Super Position Model paths, App list, homepage search queries, and structured tabular features:

Hierarchical User Tokens: e𝑖 Query: { User query or directive } <USER_EMB>

- Figure 3: Input Template of UserU. The Hierarchical User Tokens: e𝑖 injects the precomputed hierarchical embedding e𝑖 (Eq. 2). An optional instruction is followed by the special <USER_EMB> token, which signals the model to extract a unified user embedding (Sec. 3.2.2).

3.2.2 Q-Anchor Pretraining Architecture.

Query-as-Anchor Dual-Tower Architecture. Building upon the hierarchical user representations, we propose a dual-tower training architecture that operationalizes the Query-as-Anchor paradigm by aligning multi-modal behavioral signals with semantic task directives as illustrated in Fig. 2. The primary Anchor Tower ingests the hierarchical user tokens e𝑖 and appends the natural language query 𝑞𝑖 as a trailing semantic anchor. By positioning the query at the sequence terminus, the LLM backbone acts as a query-aware aggregator that selectively distills intent-relevant features from the latent space of e𝑖, ultimately projecting them into the anchored representation u𝑖,𝑞 = LLM𝑎𝑛𝑐(e𝑖,𝑞𝑖). This structural decoupling is specifically engineered for industrial scalability; the computationally intensive hierarchical prefix e𝑖 is computed once and stored via KV-cache mechanisms, allowing the model to generate diverse, scenario-adaptive embeddings for multiple queries with negligible incremental latency. Concurrently, an asymmetric Semantic Tower projects the target answer 𝑎𝑖 into a dense vector v𝑎𝑖 = LLM𝑠𝑒𝑚(𝑎𝑖), providing a high-fidelity linguistic target that serves as the groundtruth for user behavior synthesis and intent modeling. Specially, both towers share the same LLM parameters, which ensures that the behavioral features and semantic labels are mapped into a unified latent space.

JointContrastive–GenerativeOptimization. OurQuery-as-Anchor framework is trained with a joint objective that combines discriminative contrastive alignment with generative grounding to produce user embeddings that are both distinctive and semantically rich. For each user 𝑖, the anchor tower generates a query-anchored embedding u𝑖,𝑞 from the hierarchical profile e𝑖 and the task-specific query 𝑞𝑖, while the semantic tower encodes the answer 𝑎𝑖 into v𝑎𝑖. Contrastive alignment is implemented via a query-conditioned

InfoNCE loss:

∑︁𝐵

exp(sim(u𝑖,𝑞, v𝑎𝑖 )/𝜏) 𝑍𝑖

1 𝐵

, (3)

log

L𝑐𝑙 = −

𝑍𝑖 = exp(sim(u𝑖,𝑞, v𝑎𝑖 )/𝜏) + ∑︁ 𝑗≠𝑖

𝑖=1

𝑚𝑖𝑗 exp(sim(u𝑖,𝑞, v𝑎𝑗 )/𝜏)

+ ∑︁

𝑚𝑖𝑗 exp(sim(u𝑖,𝑞, u𝑗,𝑞)/𝜏) + ∑︁ 𝑗≠𝑖

𝑚𝑖𝑗 exp(sim(v𝑎𝑖, v𝑎𝑗 )/𝜏),

𝑗≠𝑖

(4)

where sim(·) is cosine similarity, 𝜏 is a temperature, 𝐵 is the batch size, and 𝑚𝑖𝑗 is a margin-based mask that removes potential false negatives inspired by [32]. Specifically, we discard a candidate negative 𝑗 for anchor 𝑖 if either its user embedding or its answer embedding is too similar to u𝑖,𝑞:

- 0, if ∃ 𝑥 ∈ {u𝑗,𝑞, v𝑗 } s.t. sim(u𝑖,𝑞,𝑥) > sim(u𝑖,𝑞, v𝑖) + 𝑐margin,
- 1, otherwise.

𝑚𝑖𝑗 =

(5)

To bridge the granularity gap between sentence-level alignment and token-level grounding [16], we propose a joint optimization objective that mitigates representation collapse while enhancing semantic density. While the contrastive loss L𝑐𝑙 enforces global discriminativeness by aligning positive pairs in the latent space, it often overlooks the fine-grained linguistic nuances essential for complex intent modeling. To counteract this, we introduce an auxiliary Next-Token Prediction (NTP) task that requires the anchor tower to autoregressively reconstruct the target answer 𝑎𝑖:

∑︁𝑇

log𝑃 (𝑦𝑡 | 𝑦<𝑡,𝑒𝑖,𝑞𝑖), (6)

L𝑛𝑡𝑝 = −

𝑡=1

where𝑇 is the sequence length. The total objective is a weighted sum L𝑡𝑜𝑡𝑎𝑙 = L𝑐𝑙 + L𝑛𝑡𝑝, enabling the query to anchor the hierarchical user profile e𝑖 into a compact, scenario-adaptive embedding that is both discriminative for downstream tasks and semantically rich for answer reconstruction.

3.2.3 Soft Prompt Tuning. To further bridge the semantic gap between general-purpose user understanding and specialized downstream business logic, we introduce a cluster-based soft prompt tuning mechanism as a post-training adaption inspired by [19, 28]. As illustrated in Fig. 2 (C), while the LLM backbone and the coarseto-fine user encoder remain frozen to preserve the foundational multi-modal alignment, we introduce a set of learnable prompt tokens that function as differentiable task controllers. These to-

kens modulate the latent space of the hierarchical embeddings u𝑖,𝑞 to better align with downstream class-specific logic. We optimize

these learnable tokens and a set of class prototypes {p𝑘}𝑘𝐾=1 using a prototypical contrastive loss L𝑝𝑡, which pulls user embeddings toward their respective category centers while pushing them away from irrelevant clusters:

u⊤

𝑖,𝑞p𝑦𝑖 𝜏

exp

∑︁𝐵

1 𝐵

log

, (7)

L𝑝𝑡 = −

u⊤

𝑖,𝑞p𝑘 𝜏

𝐾 𝑘=1 exp

𝑖=1

where u𝑖,𝑞 denotes the prompt-conditioned user embedding for sample𝑖,𝑦𝑖 is its ground-truth label, and p𝑘 represents the learnable prototype (center) for class𝑘. By maximizing the similarity between

u𝑖,𝑞 and its corresponding class center p𝑦𝑖 relative to all 𝐾 prototypes, the framework enforces a discriminative clustering structure

[Figure 3]

- Figure 4: KV-Cache optimized Query-as-Anchor inference: pre-computed user prefixes enable sequence, low-latency reanchoring across diverse tasks.

in the latent space. This strategy enables the model to bridge the semantic gap between general pretraining and specialized business labels—such as high-risk vs. low-risk users—without the catastrophic forgetting associated with full parameter fine-tuning, thereby maintaining the structural integrity and deployment efficiency of the Query-as-Anchor paradigm.

- 3.2.4 Query-as-Anchor: Accelerating Multi-Scenario Inference. To meet the demands of industrial deployment, Q-Anchor adopts a KV-cache optimization that decouples user encoding from task querying. As shown in Fig. 4, for a given user 𝑖, the hierarchical profile e𝑖 is encoded once to produce a shared-context KV cache. This cache is kept fixed as a persistent semantic prefix during inference. Given a set of downstream queries {𝑞1, . . .,𝑞𝑛}, we process them sequentially while reusing the same shared-context cache. For each query 𝑞𝑗, the model only computes the incremental hidden states for the short query tokens, resulting in an amortized complexity of 𝑂(𝐿𝑞𝑗 ) per task (and 𝑂( 𝑛𝑗=1 𝐿𝑞𝑗 ) for all queries). This design enables a single comprehensive user representation to be efficiently re-anchored to many business scenarios with negligible per-scenario incremental latency, supporting high-throughput embedding generation in the Alipay ecosystem. Deployment details are provided in Appendix E.
- 4 Experiments

ModelsandImplementation. We adoptQwen2.5-0.5B-Instruct[27] as a decoder-only backbone LLM for user representation learning. Heterogeneous user behaviors are encoded by modality-specific gte-base [17] encoders into dense embeddings, which are aggregated into unified user representations and fed into the backbone LLM for both training and inference. All baselines and ablations are trained under identical conditions: 50k fine-tuning steps with a global batch size of 2,048, using AdamW with an initial learning rate of 2 × 10−4 and cosine decay. We employ LoRA [11] with rank 64 and 𝛼 = 32 for pretraining inspired by [32], and user representations are fixed at 128 dimensions. Pretraining is conducted on 64 A100-80G GPUs with data parallelism, inference is performed on single A100-80G GPU for evaluation, and soft prompt tuning with 6 learnable tokens is carried out on single A100-80G GPU.

Pretraining scalability (data/model scale) and prompt tuning scalability (training steps/prompt tokens) are detailed in Appendix C.2 and C.3, respectively.

Baselines and Tasks. We compare against two categories of baselines. (1) General text embeddings encode natural language descriptions of user behaviors, including Qwen2.5-0.5B-Instruct without user-specific fine-tuning and several top-ranked embedding models on MTEB including KaLM-Embedding-Gemma3-12B-2511 [13], llama-embed-nemotron-8b [1], Qwen3-Embedding-8B [32]. (2) User representation models include contrastive approaches such as MSDP [8], One4all [24] and CPC [22], as well as LLM-based foundation models FOUND [7]. To study scalability, we also evaluate larger variants of our architecture via under the same training protocol. We evaluate all methods on 10 real-world binary classification tasks from Alipay’s production systems, grouped into three domains (Table 1). Detailed positive and negative label definitions for each task are provided in Appendix B (Table 4).

#### Table 1: Data information for user pretraining and test benchmarks, with number of tests per task.

Dataset Domain Scenario Number D𝑡𝑟𝑎𝑖𝑛 General (3.1) General ≈1.024×108 D𝑡𝑒𝑠𝑡

❶ User Engagement

Interest Community Active User Identification (Active), Concert Click Prediction (Concert), User Log-in Prediction (Login), Ant Forest Engagement (Forest)

≈ 50w per task

❷ Risk Fraud Detection (Fraud), Money Laundering Detection (Money)

≈ 50w per task

❸ Marketing Sensitivity

Takeout Interest (Takeout), Brand Sensitivity (Brand), Big Sale Sensitivity (Promo), Cost-Performance Sensitivity (Value)

≈ 50w per task

Evaluation Metrics. Following [7], we assess representation via linear probing on 10 binary classification tasks from Alipay’s user cognition system. We report AUC (Area Under the ROC Curve [4]) for discriminative performance and KS (Kolmogorov-Smirnov [3]) for critical decision-boundary separation.

### 5 Main Results 5.1 In-depth Performance Analysis

Q-Anchor delivers the strongest and most stable AUC and KS across all scenarios. As shown in Table 2 and Fig. 5, Q-Anchor (Prompt Tuned) achieves the best performance across all 10 benchmarks, with an average AUC of 0.8225 and KS of 0.5267. It surpasses the strongest general-purpose baseline, Llama-Embed-Nemotron-8B (AUC: 0.7488, KS: 0.3805), by +0.0737 (+9.84%) in AUC and +0.1462 (+38.4%) in KS, consistently across User Engagement, Risk, and Marketing domains (see Appendix C.1). These results indicate that the key limitation in industrial user modeling lies not in semantic capacity, but in representation alignment: generic text embeddings struggle to reconcile sparse, symbolic, multi-source behavioral logs. By contrast, Q-Anchor, pretrained on UserU with hierarchical behavior encoding, maps heterogeneous events into query-relevant signals without relying on massive parameterization.

#### Table 2: AUC performance on 10 key scenarios where our Q-Anchor (base version: pretrained) and (prompt-tuning version: post-trained) shows consistent improvement. KS results are detailed in Appendix C.1.

User Engagement Risk Marketing Sensitivity Avg. AUC Method Active Concert Login Forest Fraud Money Takeout Brand Promo Value General Embedding Models

Qwen2.5-0.5B-Instruct 0.5269 0.5173 0.7219 0.7161 0.6969 0.6885 0.6361 0.5855 0.5483 0.7134 0.6351 Qwen3-Embedding-0.6B 0.5378 0.5226 0.7294 0.7287 0.7106 0.7123 0.6914 0.6076 0.5508 0.7172 0.6508 Llama-Embed-Nemotron-8B 0.5882 0.5627 0.7735 0.8372 0.8632 0.8708 0.8176 0.7525 0.5918 0.8303 0.7488 KaLM-Embed.-Gemma3-12B 0.5597 0.5359 0.7609 0.7729 0.8229 0.8174 0.7812 0.6564 0.589 0.7609 0.7058

###### User Embedding Models

MSDP [8] 0.6415 0.5155 0.9504 0.9580 0.9152 0.8746 0.7814 0.6318 0.5850 0.8389 0.7692 One4all [24] 0.6515 0.5568 0.9509 0.9614 0.9203 0.8782 0.7609 0.6289 0.5761 0.8207 0.7706 CPC [22] 0.6506 0.5314 0.9506 0.9608 0.9171 0.8736 0.7817 0.6235 0.6101 0.8259 0.7725 FOUND [7] 0.6201 0.5527 0.8131 0.9573 0.9083 0.9235 0.8528 0.7294 0.6202 0.8544 0.7832

###### Q-Anchor Embedding (Ours)

Q-Anchor (Base) 0.6568 0.5739 0.8420 0.9700 0.9218 0.9382 0.8799 0.7979 0.6189 0.9049 0.8104 Q-Anchor (Prompt Tuned) 0.6678 0.5844 0.8443 0.9716 0.9242 0.9439 0.8811 0.8535 0.6350 0.9194 0.8225

[Figure 4]

#### Figure 5: Average KS performance of Q-Anchor and baselines across 10 Alipay scenarios. Per-scenario results are provided in Appendix C.1.

Robust cross-domain representation validates Query-asAnchor as a “one-model-for-many” paradigm. The same encoder generalizes well across three heterogeneous domains– Engagement, Risk, and Marketing—without task-specific architectures. In Risk, Q-Anchor (Prompt Tuned) achieves 0.9439 on Money, outperforming strong user-embedding baselines such as FOUND (0.9235) and MSDP (0.8746), suggesting that the anchor query suppresses high-entropy transactional noise and amplifies decision-relevant patterns. In Marketing, the gains are even more pronounced: for Brand, AUC jumps from 0.7979 (Base) to 0.8535 (Prompt Tuned), indicating that the model not only transfers across domains but also adapts to domain-specific decision boundaries where subtle preference signals matter. This supports a scalable alternative to maintaining many scenario-specific models: a universal representation plus lightweight scenario conditioning.

Q-Anchor (Base) scales more with data than with parameters. Figure 6 characterizes Q-Anchor (Base) under data scaling (20.48M→102.4M pretraining pairs) and model scaling (Qwen2.50.5B→3B) with a fixed training/data budget. Increasing data yields consistentgains(Avg. AUC0.8029→0.8105,Avg.KS 0.4895→0.5044). In contrast, model scaling is non-monotonic: Table 8 shows 0.5B

[Figure 5]

[Figure 6]

(a) Pretraining Data Scale (b) Impact of Model Size

#### Figure 6: Scalability analysis of Q-Anchor Embedding (Base). (a) Performance increases with more pretraining data. (b) Performance vs. model parameters (0.5B to 3.0B).

performs best (Avg. AUC/KS=0.8105/0.5044), while 1.5B/3B bring no gains and sometimes regress (e.g., Brand, Promo), consistent with prior embedding-scaling observations [14]. To better understand this, we analyze larger models’ training gradient in Appendix C.2 (Fig. 11), showing that embedding quality depends more on pretraining data than model scale. Accordingly, we adopt 50k-step pretraining with the 0.5B backbone for the optimal accuracy–efficiency trade-off; per-scenario results are provided in Appendix C.2.

Q-Anchor (Prompt-tuned) scales efficiently with prompt tokens and tuning steps, saturating early. Figure 8 evaluates QAnchor Embedding (Prompt Tuned) under token scaling (1→16 tokens) and step scaling (100→500 steps). Performance rises rapidly up to6 tokens(Avg.AUC0.8146→0.8225,Avg.KS0.5140→0.5267), then saturates with minor fluctuations at 8/16 tokens. Increasing tuning steps yields steady gains (Avg. AUC 0.8159→0.8225, Avg. KS 0.5141→0.5267). Overall, prompt tuning is efficiency-friendly, achieving most gains with a small prompt budget and modest optimization, motivating 6 tokens and 500 steps as our default. Perscenario results are in Appendix C.3.

Scenario-adaptive tuning as a performance multiplier with interpretable re-anchoring. Post-training with cluster-based soft prompts yields a consistent lift in all scenarios (avg. 0.8104 to 0.8225), confirming that the base model captures general behavioral

[Figure 7]

[Figure 8]

- Figure 7: Modal attention shift after prompt tuning. Left: Takeout Interest—Bill attention increases by +26.0%, capturing dining purchasing power. Right: Ant Forest—SPM attention rises by +6.4%, consistent with navigation-heavy usage. Red arrows mark positive attention gains, highlighting QAnchor’s dynamic feature re-anchoring.

[Figure 9]

(a) Prompt Tokens Scale (b) Training Steps Scale

[Figure 10]

- Figure 8: Scalability analysis of Q-Anchor Embedding (Prompt Tuned). (a) Performance vs. the number of learnable prompt tokens (1-16, Ours: 6). (b) Performance vs. the training-step budget (100–500, Ours: 500).

semantics while prompts specialize the embedding space to scenario boundaries. Importantly, the improvement is not a black box: Figure 7 shows that prompt tuning changes where the model looks. For Takeout Interest, attention to the Bill modality increases by +26.0%, aligning with purchasing power as the dominant signal; for Ant Forest, the SPM modality rises by +6.4%, matching navigationintensive usage. The t-SNE visualizations (Figure 9) further corroborate this behavior: prompt-tuned representations form clearer scenario-consistent groupings than the universal space, reflecting better separation of positives vs. negatives. To avoid visualization bias, we corroborate t-SNE findings with PCA (Appendix C.4), which consistently shows sharper cluster separation under prompt tuning. Together, the quantitative gains and qualitative shifts indicate that Query-as-Anchor functions as a semantic lens—guiding the representation to focus more on semantically relevant multimodal evidence through minimal parameter updates—enabling fast, deployable specialization in changing industrial environments.

### 5.2 Ablation Study

We ablate Q-Anchor (Base) by removing one component at a time while keeping the backbone, data, and training budget fixed. We evaluate (i) structural tokens (User token, modality tokens, or both), (ii) pretraining ingredients (margin-mask filtering, NTP, and contrastive alignment), and (iii) post-training (prompt tuning, and

[Figure 11]

(a) User Engagement

[Figure 12]

(b) Risk

[Figure 13]

(c) Marketing Sensitivity

#### Figure 9: t-SNE Visualization of universal and prompt-tuned representation of 10 scenarios. PCA visualizations are provided in Appendix C.4.

prompt tuning without pretraining). AUC results are in Table 3, with KS results in Appendix C.1.

Ablation on Modality Token: Explicit structure helps DeepFind attribute evidence to the correct source. Removing the User or modality tokens slightly lowers Avg. AUC (0.8104 to 0.8086/ 0.8088), and removing both produces the largest drop in this block (0.8065). The effect concentrates on modality-sensitive marketing scenarios, especially Brand (0.7979 to 0.7819), where performance depends on which modality carries the signal rather than overall activity. These results support our design: user/modal markers inject a minimal inductive bias about log structure and improve cross-modal aggregation; consistent degradations are also observed in KS (Appendix C.1), indicating reduced ranking separability.

Ablation on Training Method: Contrastive alignment is the primary signal; auxiliary objectives act as regularizers. Removing contrastive learning causes the largest regression, reducing Avg. AUC from 0.8104 to 0.7667 and sharply degrading fine-grained scenarios (e.g., Brand 0.7979→0.6512; Money 0.9382→0.9091). This indicates that token-level modeling alone cannot impose queryconditioned separability, whereas contrastive supervision explicitly structures the embedding space. In comparison, margin-mask filtering and next-token prediction are supportive: disabling filtering

#### Table 3: Ablation study (AUC) of Q-Anchor Embedding across modality, training method and prompt tuning. KS results are detailed in Appendix C.1.

User Engagement Risk Marketing Sensitivity Avg. AUC Method Active Concert Login Forest Fraud Money Takeout Brand Promo Value Q-Anchor (Base) 0.6568 0.5739 0.8420 0.9700 0.9218 0.9382 0.8799 0.7979 0.6189 0.9049 0.8104 Ablation on Modality Token

w / o User Tok. 0.6525 0.5757 0.8424 0.9702 0.9219 0.9358 0.8785 0.7884 0.6195 0.9006 0.8086 w / o Modal Tok. 0.6546 0.5701 0.8423 0.9700 0.9214 0.9372 0.8778 0.7903 0.6210 0.9036 0.8088 w / o Modal & User Tok. 0.6544 0.5703 0.8403 0.9694 0.9211 0.9374 0.8783 0.7819 0.6141 0.8981 0.8065

###### Ablation on Training Method

w / o Filter 0.6461 0.5693 0.8412 0.9685 0.9197 0.9357 0.8728 0.7867 0.6218 0.8854 0.8047 w / o NTP 0.6577 0.5692 0.8392 0.9701 0.9208 0.9369 0.8798 0.7858 0.6079 0.8936 0.8061 w / o Contrastive 0.6408 0.5456 0.8454 0.9584 0.9071 0.9091 0.7967 0.6512 0.5980 0.8143 0.7667

Q-Anchor (Prompt Tuned) 0.6678 0.5844 0.8443 0.9716 0.9242 0.9439 0.8811 0.8535 0.6350 0.9194 0.8225 Ablation on Prompt Tuning w / o pretrain 0.6557 0.5432 0.7546 0.9338 0.9189 0.8153 0.8793 0.7937 0.5833 0.9044 0.7782

(“w/o Filter”) lowers Avg. AUC to 0.8047, consistent with increased false negatives in noisy logs, and disabling NTP (“w/o NTP”) yields 0.8061, suggesting weaker local event modeling. KS shows an even steeper drop without contrastive alignment (Appendix C.1), and the same trend is reflected qualitatively by less coherent scenario clusters in Fig. 9. Overall, contrastive alignment defines the geometry, while filtering and NTP mainly stabilize training and improve robustness.

Ablation on Pretraining: Pre-trained weights provide the essential behavioral prior for intent discovery. Excluding the pretraining phase (w/o pretrain) triggers a systemic performance collapse: Average AUC falls from 0.8225 to 0.7781, while Average KS undergoes a sharper relative regression of 11.2% (0.5267→0.4679). The degradation is most pronounced in complex scenarios (e.g., Money AUC: 0.9439→0.8153), where prompt tuning alone fails to reconstruct the high-dimensional latent structures necessary for fine-grained separation. This confirms that pretraining is not a mere optimization aid but a foundational requirement for distilling robust behavioral priors, without which the model cannot effectively generalize across noisy, long-tail user trajectories.

### 5.3 Industrial Online A/B Testing

We evaluate Q-Anchor embeddings in two large-scale Alipay A/B tests over two weeks, with users randomly assigned to treatment (policy with embeddings) or control (fixed-time/rule-based). See Appendix D for pre-deployment offline results; online A/B outcomes are reported below:

Scenario I: Interactive Voice Response (IVR) Cash-Reserve Outreach. Q-Anchor embeddings were deployed in the live cashreserve outreach system to optimize send-time decisions according to user availability and responsiveness. In production, this representation-aware timing increased the drawdown rate by 12.5% and the average outstanding balance per user by 5.3%. Early-funnel engagement also improved: the cash-reserve product visit rate rose by 4.2%, and drawdown-page visits increased by 17.7%, reflecting stronger activation and smoother progression through the credit funnel. These results demonstrate that embedding-informed timing decisions enhance both immediate credit utilization and upstream engagement.

#### ScenarioII:CreditDelinquencyRisk Identification.Q-Anchor

embeddings were integrated into the credit risk scoring pipeline, improving the business-critical KS score by 1.96%. Results show that embedding-informed scoring enhances predictive performance and risk-aware credit allocation.

Deployment at Alipay Scale. We run daily Q-Anchor embedding refresh for hundreds of millions of users on a 100×L20 cluster. For multi-scenario serving, Query-as-Anchor with shared prefix KV-cache encodes each user prefix once and reuses it across queries, so adding a scenario only requires query-suffix computation (one extra L20 to maintain the same SLA), rather than replicating the entire 100-GPU pipeline for every scenario.

### 6 Conclusion

We present Q-Anchor Embedding, a unified framework for industrial user representation that closes the gap between sparse, noisy, heterogeneous behavior logs and LLM-level semantic representatin. Q-Anchor contributes: (1) UserU, an industrial-scale pretraining corpus that couples rule-based future behavior supervision with reflection-verified LLM-synthesized user QA to inject temporal dynamics and semantic understanding; (2) Query-asAnchor, a hierarchical coarse-to-fine user encoder and query- conditioned alignment paradigm that distills multi-source signals into task-adaptive embeddings; and (3) semantic re-anchoring via lightweight soft prompts that re-anchor modalities to match downstream decision boundaries. Across 10 real-world benchmarks and 2 online A/B tests, Q-Anchor achieves SOTA AUC and KS over LLM embeddings and strong baselines, while enabling interpretable, scenario-adaptive, low-cost, and transferable industrial user representations.

### Ethical Considerations

All experiments comply with Alipay’s data-governance, privacy, and security policies, with encrypted data/embeddings, strict access control, and audit logging to prevent unauthorized access or linkage. We present Q-Anchor Embedding to advance responsible user representation learning and industrial systems.

### References

- [1] Yauhen Babakhin, Radek Osmulski, Ronay Ak, Gabriel Moreira, Mengyao Xu, Benedikt Schifferer, Bo Liu, and Even Oldridge. 2025. Llama-Embed-Nemotron8B: A Universal Text Embedding Model for Multilingual and Cross-Lingual Tasks. arXiv preprint arXiv:2511.07025 (2025).
- [2] Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. [n.d.]. LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders. In First Conference on Language Modeling.
- [3] Vance W Berger and YanYan Zhou. 2014. Kolmogorov–smirnov test: Overview. Wiley statsref: Statistics reference online (2014).
- [4] Andrew P Bradley. 1997. The use of the area under the ROC curve in the evaluation of machine learning algorithms. Pattern recognition 30, 7 (1997), 1145–1159.
- [5] Zheng Chai, Zhihong Chen, Chenliang Li, Rong Xiao, Houyi Li, Jiawei Wu, Jingxu Chen, and Haihong Tang. 2022. User-aware multi-interest learning for candidate matching in recommenders. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval. 1326–1335.
- [6] Haonan Chen, Liang Wang, Nan Yang, Yutao Zhu, Ziliang Zhao, Furu Wei, and Zhicheng Dou. 2025. Little giants: Synthesizing high-quality embedding data at scale. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 1392–1411.
- [7] Bin Dou, Baokun Wang, Yun Zhu, Xiaotong Lin, Yike Xu, Xiaorui Huang, Yang Chen, Yun Liu, Shaoshuai Han, Yongchao Liu, et al. 2025. Transferable and Forecastable User Targeting Foundation Model. In Companion Proceedings of the ACM on Web Conference 2025. 181–190.
- [8] Chilin Fu, Weichang Wu, Xiaolu Zhang, Jun Hu, Jing Wang, and Jun Zhou.

2023. Robust user behavioral sequence representation via multi-scale stochastic distribution prediction. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 4567–4573.

- [9] Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 6894–6910.
- [10] Yingzhi He, Xiaohao Liu, An Zhang, Yunshan Ma, and Tat-Seng Chua. 2025. Llm2rec: Large language models are powerful embedding models for sequential recommendation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 896–907.
- [11] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. [n. d.]. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.
- [12] Jun Hu, Wenwen Xia, Xiaolu Zhang, Chilin Fu, Weichang Wu, Zhaoxin Huan, Ang Li, Zuoli Tang, and Jun Zhou. 2024. Enhancing sequential recommendation via llm-based semantic embedding learning. In Companion Proceedings of the ACM Web Conference 2024. 103–111.
- [13] Xinshuo Hu, Zifei Shan, Xinping Zhao, Zetian Sun, Zhenyu Liu, Dongfang Li, Shaolin Ye, Xinyuan Wei, Qian Chen, Baotian Hu, et al. 2025. Kalm-embedding: Superior training data brings a stronger embedding model. arXiv preprint arXiv:2501.01028 (2025).
- [14] Ting Jiang, Shaohan Huang, Zhongzhi Luan, Deqing Wang, and Fuzhen Zhuang.

2024. Scaling sentence embeddings with large language models. In Findings of the association for computational linguistics: EMNLP 2024. 3182–3196.

- [15] Chenglin Li, Yuanzhen Xie, Chenyun Yu, Bo Hu, Zang Li, Guoqiang Shu, Xiaohu Qie, and Di Niu. 2023. One for all, all for one: Learning and transferring user embeddings for cross-domain recommendation. In Proceedings of the sixteenth ACM international conference on web search and data mining. 366–374.
- [16] Shiyu Li, Yang Tang, Ruijie Liu, Shi-Zhe Chen, and Xi Chen. 2025. Conanembedding-v2: Training an llm from scratch for text embeddings. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 15011–15027.
- [17] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281 (2023).
- [18] Guanyu Lin, Chen Gao, Yinfeng Li, Yu Zheng, Zhiheng Li, Depeng Jin, and Yong Li. 2022. Dual contrastive network for sequential recommendation. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval. 2686–2691.
- [19] Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. 2022. P-tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 61–68.
- [20] Lin Ning, Luyang Liu, Jiaxing Wu, Neo Wu, Devora Berlowitz, Sushant Prakash, Bradley Green, Shawn O’Banion, and Jun Xie. 2025. User-llm: Efficient llm contextualization with user embeddings. In Companion Proceedings of the ACM on Web Conference 2025. 1219–1223.
- [21] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085 (2019).

- [22] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018).
- [23] Letian Peng, Yuwei Zhang, Zilong Wang, Jayanth Srinivasa, Gaowen Liu, Zihan Wang, and Jingbo Shang. 2024. Answer is All You Need: Instruction-following Text Embedding via Answering the Question. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 459–477.
- [24] Kyuyong Shin, Hanock Kwak, Kyung-Min Kim, Minkyu Kim, Young-Jin Park, Jisu Jeong, and Seungjae Jung. 2021. One4all user representation for recommender systems in e-commerce. arXiv preprint arXiv:2106.00573 (2021).
- [25] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang.

2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management. 1441–1450.

- [26] Yixuan Tang and Yi Yang. 2024. Do we need domain-specific embedding models? An empirical investigation. arXiv preprint arXiv:2409.18511 (2024).
- [27] Qwen Team et al. 2024. Qwen2 technical report. arXiv preprint arXiv:2407.10671 2, 3 (2024).
- [28] Sishi Xiong, Yu Zhao, Jie Zhang, Li Mengxiang, Zhongjiang He, Xuelong Li, and Shuangyong Song. 2024. Dual prompt tuning based contrastive learning for hierarchical text classification. In Findings of the association for computational linguistics ACL 2024. 12146–12158.
- [29] Jiahao Yuan, Zhiqing Cui, Hanqing Wang, Yuansheng Gao, Yucheng Zhou, and Usman Naseem. 2025. Kardia-R1: Unleashing LLMs to Reason toward Understanding and Empathy for Emotional Support via Rubric-as-Judge Reinforcement Learning. arXiv preprint arXiv:2512.01282 (2025).
- [30] Biao Zhang, Zhongtao Liu, Colin Cherry, and Orhan Firat. [n.d.]. When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method. In The Twelfth International Conference on Learning Representations.
- [31] Wei Zhang, Dai Li, Chen Liang, Fang Zhou, Zhongke Zhang, Xuewei Wang, Ru Li, Yi Zhou, Yaning Huang, Dong Liang, et al. 2024. Scaling user modeling: Largescale online user representations for ads personalization in meta. In Companion Proceedings of the ACM Web Conference 2024. 47–55.
- [32] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. 2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv preprint arXiv:2506.05176 (2025).
- [33] Yuying Zhao, Minghua Xu, Huiyuan Chen, Yuzhong Chen, Yiwei Cai, Rashidul Islam, Yu Wang, and Tyler Derr. 2024. Can one embedding fit all? a multiinterest learning paradigm towards improving user interest diversity fairness. In Proceedings of the ACM web conference 2024. 1237–1248.

### A Data Example of UserU Dataset

In this section, we present illustrative toy examples curated from the two core subsets of the UserU dataset: the Behavior-based Interaction Dataset (Dfuture) and the Synthetic Query-Answer Dataset (Duqa). These samples are designed to provide a qualitative intuition of the data modality and the alignment tasks.

Privacy and Demonstration Note: It is important to emphasize that the cases provided below are synthetic toy examples constructed solely for demonstration purposes. They do not represent the actual raw records of any specific individual.

### B Details of Downstream Benchmarks

As shown in Table 1 & Table 4, we evaluate on D𝑡𝑒𝑠𝑡 containing 10 real-world binary classification tasks, grouped into three domains.

Common protocol. Each benchmark is a user-level binary classification task. For task 𝜏, we define a prediction window and assign the label 𝑦𝑢𝜏 ∈ {0, 1} by whether user 𝑢 triggers at least one target event associated with 𝜏 during that window. We report task definitions at the scenario level for privacy and compliance; all methods are evaluated under identical data construction and labeling rules.

#### Table 4: Downstream test benchmarks and scenario-level binary label definitions. Labels are assigned by whether a user triggers the target event within the prediction window. Each task contains ≈50w test data.

###### Domain Scenario (Task) Target Positive (𝑦=1) Negative (𝑦=0)

Interest Community Active User Identification (Active)

Whether a user will be active in the Alipay interest community.

Any qualified interest-community activity occurs (e.g., community visit/app open, content browse, like, comment, post).

No qualified interest-community activity occurs.

User Engagement

Concert Click Prediction (Concert)

Whether a user will click concert-related content.

Any click on items categorized as concertrelated occurs.

No click on concert-related items occurs.

User Log-in Prediction (Login) Whether a user will log in. At least one login event occurs. No login event occurs. Ant Forest Engagement (Forest) Whether a user will engage

Any Ant Forest interaction occurs (e.g., enter/collect/participate).

No Ant Forest interaction occurs.

with Ant Forest.

Fraud Detection (Fraud) Whether a user will be associated with a fraud-related risk event.

User is confirmed/flagged by the risk-control pipeline as fraud-related.

User is not flagged as fraud-related.

Risk

Money Laundering Detection (Money)

Whether a user will be associated with a money-launderingrelated risk event.

User is confirmed/flagged by compliance/risk-control as moneylaundering-related.

User is not flagged as money-launderingrelated.

Takeout Interest (Takeout) Whether a user will show interest in takeout-related services.

Any click/visit/conversion on takeoutcategory content occurs.

No click/visit/conversion on takeoutcategory content occurs.

Marketing Sensitivity

Brand Sensitivity (Brand) Whether a user will respond to brand-related marketing stimuli.

Any interaction with brand campaigns occurs (e.g., click/claim/participate).

No interaction with brand campaigns occurs.

Big Sale Sensitivity (Promo) Whether a user will respond to big-sale promotions.

Any interaction with big-sale promotional content occurs.

No interaction with big-sale promotional content occurs.

Cost-Performance Sensitivity (Value)

Whether a user will respond to cost-performance offers.

Any interaction with value-for-money themed content or items tagged as costperformance occurs.

No such interaction occurs.

[Figure 14]

- (a) Alignment between raw multi-modal behavioral signals and future user trajectories.

[Figure 15]

- (b) High-level semantic reasoning derived from implicit behavioral patterns.

Figure 10: Illustrative toy instances from the UserU dataset. Figure (a) illustrates the behavior-to-behavior prediction task, while Figure (b) showcases the behavior-to-semantic reasoning capability enabled by query-aware alignment.

C Supplementary Experiments and Analysis C.1 Experimental Results on Alipay

Benchmarks: KS Performance and Ablation Study

KS Performance. As illustrated in Table 5, our Q-Anchor embedding achieves state-of-the-art KS performance across all 10 industrial scenarios, demonstrating superior discriminative power in ranking positive vs. negative user behaviors. As shown in Table 2, Q-Anchor (Prompt Tuned) attains the highest average KS (0.5267), outperforming not only general-purpose embedding models (e.g., Llama-Embed-Nemotron-8B: 0.3805) but also specialized user representation methods such as FOUND (0.4529) and CPC (0.4556).

KS Ablation Study. Table 6 presents the KS ablations of QAnchor (Base) by removing one component at a time while keeping the backbone, data, and training budget fixed. Overall, removing structural tokens (user/modality token) causes mild yet consistent degradations in Avg. KS, indicating that explicit log structure helps the encoder attribute evidence to the correct source and improves ranking stability. In contrast, the contrastive objective is the primary driver of discriminability: without contrastive learning, Avg. KS drops substantially from 0.5044 to 0.4215, with particularly large losses on fine-grained marketing scenarios (e.g., Brand: 0.4527 → 0.2169), showing that token-level modeling alone is insufficient to enforce query-conditioned separation. Disabling margin-mask filtering or NTP yields smaller but systematic decreases, suggesting they mainly stabilize training under noisy, sparse behavioral logs. Notably, the "w/o pretrain" ablation reveals that skipping the initial pretraining phase significantly weakens the model’s performance (Avg. KS drops by 11.16%), even with subsequent prompt tuning.

#### Table 5: KS performance on 10 key scenarios where our Q-Anchor (base version: pretrained) and (prompt-tuning version: post-trained) shows consistent improvement.

User Engagement Risk Marketing Sensitivity Avg. KS Method Active Concert Login Forest Fraud Money TakeOut Brand Promo Value General Embedding Models

Qwen2.5-0.5B-Instruct 0.0421 0.0280 0.3326 0.3174 0.3034 0.2958 0.2066 0.1324 0.0880 0.3262 0.2073 Qwen3-Embedding-0.6B 0.0524 0.0330 0.3459 0.3274 0.3102 0.3086 0.275 0.1581 0.0814 0.3270 0.2219 Llama-Embed-Nemotron-8B 0.1302 0.0857 0.4003 0.5201 0.5758 0.5868 0.4918 0.3685 0.1364 0.5091 0.3805 KaLM-Embed.-Gemma3-12B 0.0880 0.0533 0.3830 0.4036 0.4944 0.4879 0.4207 0.2139 0.1312 0.3937 0.3070

##### User Embedding Models

MSDP [8] 0.2165 0.0608 0.7731 0.8104 0.7026 0.6069 0.4229 0.2107 0.1299 0.5345 0.4469 One4all [24] 0.2403 0.1075 0.7749 0.8191 0.7160 0.6113 0.3886 0.2135 0.1261 0.4980 0.4495 CPC [22] 0.2402 0.0921 0.7729 0.8197 0.7077 0.6028 0.4249 0.2068 0.1824 0.5061 0.4556 FOUND [7] 0.1669 0.0780 0.4660 0.7967 0.6739 0.7084 0.5762 0.3348 0.1753 0.5521 0.4529

##### Q-Anchor Embedding (Ours)

Q-Anchor (Base) 0.2228 0.1023 0.5216 0.8437 0.7054 0.7450 0.6210 0.4527 0.1727 0.6567 0.5044 Q-Anchor (Prompt Tuned) 0.2412 0.1177 0.5225 0.8480 0.7086 0.7573 0.6212 0.5564 0.1962 0.6978 0.5267

This underscores that pretraining effectively encodes a robust behavioral prior that serves as a necessary foundation for downstream adaptation. These patterns closely match the AUC ablations, confirming that our design choices jointly improve both classification performance and the ranking quality measured by KS.

### C.2 Scalability of Q-Anchor Embedding (Base)

Scalability performance across pretraining data scales. Table 7 evaluates Q-Anchor (Base) under increasing pretraining data scales (10k–50k steps, i.e., 20.5M–102.4M samples with batch size 2048). Overall, performance improves steadily as we scale up pretraining, with the best overall results achieved at 50k steps: Avg. AUC increases from 0.8029 (10k) to 0.8105, and Avg. KS rises from 0.4895 to 0.5044. Gains are particularly clear in high-signal domains such as Risk and Marketing (e.g., Money AUC: 0.9340 → 0.9382; Value AUC: 0.8869 → 0.9049), suggesting that larger pretraining exposes the encoder to more diverse long-tail behavioral patterns and improves query-conditioned ranking separability. While a few scenarios saturate early (e.g., Forest), the overall trend indicates that scaling pretraining data consistently strengthens both classification (AUC) and ranking quality (KS); therefore, we adopt 50k as our default setting in all subsequent experiments.

Scalability performance across model scales. Table 8 compares Q-Anchor (Base) under different backbone scales (0.5B–3B). We observe that scaling up the model does not monotonically improve performance: the 0.5B backbone achieves the best overall results (Avg. AUC/KS: 0.8105/0.5044), while larger models (1.5B and 3B) provide no consistent gains and even slightly regress on several scenarios (e.g., Brand and Promo). This non-monotonic scaling behavior is aligned with prior findings [14] that, under fixed data and training budgets, larger encoders may not translate into better embedding quality and can be harder to optimize for sentencelevel embedding objectives [14]. From an industrial perspective, the 0.5B configuration is also preferable for deployment due to its substantially lower latency and serving cost, making it a better

[Figure 16]

[Figure 17]

(a) Performance (Avg. AUC/KS) vs. model size (0.5B–3B).

(b) Gradient magnitude decreases with model scale.

Figure 11: Scalability analysis of Q-Anchor (Base). (a) Smaller 0.5B backbone achieves best Avg. AUC/KS; larger models show no consistent gains. (b) Gradient magnitude diminishes as model size increases, explaining the nonmonotonic performance trend.

accuracy–efficiency trade-off for high-throughput online Alipay financial and risk systems.

To investigate the non-monotonic performance observed across different model scales, we analyze the gradient dynamics during the late stages of training. As shown in Figure 11, both the average and maximum gradient magnitudes decrease sharply as the model size increases: the average gradient drops from 0.082 (0.5B) to 0.035 (1.5B) and further to 0.028 (3B), while the maximum gradient similarly decreases from 0.824 to 0.231 and 0.164, respectively. This empirical evidence indicates that under fixed data budgets, larger encoders produce increasingly flat optimization landscapes for sentence-level embedding objectives, leading to a “saturation” of learning signals. Consistent with prior findings [14], this explains why larger models, despite their generative capacity, are harder to optimize for discriminative embedding alignment. From an industrial perspective, the 0.5B configuration achieves the best accuracy–efficiency trade-off, offering stronger gradients, better embedding quality, and significantly lower serving latency and

#### Table 6: Ablation study (KS) of Q-Anchor Embedding across modality, training method and prompt tuning.

User Engagement Risk Marketing Sensitivity Avg. KS Method Active Concert Login Forest Fraud Money TakeOut Brand Promo Value Q-Anchor (Base) 0.2228 0.1023 0.5216 0.8437 0.7054 0.7450 0.6210 0.4527 0.1727 0.6567 0.5044 Ablation on Modality Token

w / o User Tok. 0.2176 0.1021 0.5204 0.8440 0.7056 0.7387 0.6200 0.4309 0.1744 0.6478 0.5002 w / o Modal Tok. 0.2186 0.0982 0.5192 0.8415 0.7039 0.7407 0.6187 0.4321 0.1775 0.6574 0.5008 w / o Modal & User Tok. 0.2181 0.0966 0.5169 0.8411 0.7030 0.7402 0.6184 0.4246 0.1654 0.6414 0.4966

##### Ablation on Training Method

w / o Filter 0.2082 0.0982 0.5176 0.8370 0.6986 0.7386 0.6071 0.3824 0.1755 0.6133 0.4877 w / o NTP 0.2255 0.0952 0.5150 0.8432 0.7024 0.7404 0.6225 0.4298 0.1524 0.6346 0.4961 w / o Contrastive 0.2028 0.0662 0.5213 0.7977 0.6690 0.6752 0.4534 0.2169 0.1423 0.4702 0.4215

Q-Anchor (Prompt Tuned) 0.2412 0.1177 0.5225 0.8480 0.7086 0.7573 0.6212 0.5564 0.1962 0.6978 0.5267 Ablation on Prompt Tuning w / o pretrain 0.2200 0.0931 0.4522 0.7822 0.6953 0.5645 0.6202 0.4436 0.1528 0.6550 0.4679

#### Table 7: Scalability performance of Q-Anchor Embedding (Base) across pretraining steps (10k–50k) and corresponding data scales (20.5M–102.4M samples, batch size=2048) (AUC & KS).

User Engagement Risk Marketing Sensitivity Avg. Steps / Data scale Active Concert Login Forest Fraud Money TakeOut Brand Promo Value AUC Performance

w/ 10k / 20.48M 0.6443 0.5630 0.8419 0.9707 0.9170 0.9340 0.8759 0.7750 0.6206 0.8869 0.8029 w/ 20k / 40.96M 0.6481 0.5713 0.8429 0.9705 0.9198 0.9362 0.8776 0.7988 0.6259 0.8999 0.8091 w/ 30k / 61.44M 0.6472 0.5735 0.8429 0.9706 0.9203 0.9365 0.8782 0.8027 0.6229 0.9041 0.8099 w/ 40k / 81.92M 0.6532 0.5742 0.8419 0.9704 0.9204 0.9369 0.8792 0.7986 0.6159 0.9045 0.8095 w/ 50k / 102.4M 0.6571 0.5739 0.8420 0.9700 0.9218 0.9382 0.8799 0.7979 0.6190 0.9049 0.8105

##### KS Performance

w/ 10k / 20.48M 0.2036 0.0857 0.5166 0.8447 0.6950 0.7337 0.6140 0.4099 0.1761 0.6153 0.4895 w/ 20k / 40.96M 0.2110 0.0988 0.5205 0.8445 0.7012 0.7369 0.6167 0.4534 0.1848 0.6469 0.5015 w/ 30k / 61.44M 0.2099 0.1008 0.5211 0.8459 0.7030 0.7391 0.6173 0.4606 0.1789 0.6554 0.5032 w/ 40k / 81.92M 0.2183 0.1015 0.5204 0.8447 0.7025 0.7405 0.6201 0.4536 0.1685 0.6572 0.5027 w/ 50k / 102.4M 0.2228 0.1023 0.5216 0.8437 0.7054 0.7450 0.6210 0.4527 0.1727 0.6567 0.5044

cost, making it ideal for high-throughput deployment in Alipay’s production systems.

### C.3 Scalability performance of Q-Anchor Embedding (Prompt Tuned)

Scalability performance across learnable token scales. Table 9 studies the scalability of prompt tuning by varying the number of learnable prompt tokens from 1 to 16. Overall, performance improves rapidly when increasing tokens from 1 to 6, and then largely saturates: the best average performance is achieved at 6 prompt tokens (Avg. AUC/KS: 0.8225/0.5267). This suggests that a small number of learnable tokens is sufficient to provide scenario-specific conditioning, while additional tokens bring diminishing returns and may introduce optimization noise on some tasks. Given this clear accuracy–efficiency trade-off, we adopt 6 prompt tokens as the default configuration in all subsequent experiments.

Scalability performance across learning step scales. Table 10 reports the scalability of prompt-tuned Q-Anchor embedding under

different training-step budgets (100–500, step size = 100). As the number of steps increases, performance improves consistently on both AUC and KS, with the average AUC rising from 0.8159 (100 steps) to 0.8225 (500 steps) and the average KS from 0.5141 to 0.5267. Notably, 500 steps achieves the best overall results, delivering the top average AUC/KS and leading most tasks, while 400 steps is typically the strongest runner-up. These results indicate that our method benefits from longer optimization yet remains robust under smaller step budgets, showing stable scalability across training steps.

### C.4 PCA Visualization of Universal and Prompt-tuned Representation

To complement our t-SNE analysis and strengthen the reliability of qualitative insights, we further visualize the universal and prompt-tuned representations using PCA (Fig. 12). Across all three domains—User Engagement, Risk, and Marketing Sensitivity—the prompt-tuned embeddings consistently exhibit tighter intra-class

#### Table 8: Scalability performance of Q-Anchor Embedding (Base) across model scales (0.5B–3B).

User Engagement Risk Marketing Sensitivity Avg. Model Active Concert Login Forest Fraud Money TakeOut Brand Promo Value AUC Performance

- Q-Anchor-0.5B (Base) 0.6571 0.5739 0.8420 0.9700 0.9218 0.9382 0.8799 0.7979 0.6190 0.9049 0.8105

- Q-Anchor-1.5B (Base) 0.6436 0.5693 0.8444 0.9706 0.9220 0.9376 0.8759 0.7864 0.6156 0.9000 0.8065 Q-Anchor-3B (Base) 0.6404 0.5626 0.8449 0.9712 0.9220 0.9370 0.8777 0.7824 0.6066 0.8989 0.8044 KS Performance

- Q-Anchor-0.5B (Base) 0.2228 0.1023 0.5216 0.8437 0.7054 0.7450 0.6210 0.4527 0.1727 0.6567 0.5044
- Q-Anchor-1.5B (Base) 0.2010 0.0969 0.5223 0.8442 0.7052 0.7428 0.6151 0.4295 0.1657 0.6486 0.4971 Q-Anchor-3B (Base) 0.1957 0.0875 0.5234 0.8457 0.7045 0.7407 0.6171 0.4231 0.1493 0.6454 0.4932

- Table 9: Scalability performance of Q-Anchor Embedding (Prompt Tuned) across learnable token scales (1–16) for prompt tuning (Ours: 6 prompt tokens).

User Engagement Risk Marketing Sensitivity Avg. Token Active Concert Login Forest Fraud Money TakeOut Brand Promo Value AUC Performance

- w / 1 prompt tok. 0.6513 0.5746 0.8408 0.9702 0.9215 0.9424 0.8798 0.8290 0.6250 0.9114 0.8146
- w / 2 prompt tok. 0.6576 0.5755 0.8408 0.9696 0.9221 0.9407 0.8810 0.8380 0.6269 0.9170 0.8169 w / 4 prompt tok. 0.6625 0.5792 0.8450 0.9715 0.9220 0.9430 0.8804 0.8501 0.6321 0.9155 0.8201 w / 6 prompt tok. 0.6678 0.5844 0.8443 0.9716 0.9242 0.9439 0.8811 0.8535 0.6350 0.9194 0.8225 w / 8 prompt tok. 0.6612 0.5774 0.8432 0.9704 0.9228 0.9409 0.8801 0.8575 0.6310 0.9205 0.8205 w / 16 prompt tok. 0.6649 0.5768 0.8435 0.9740 0.9230 0.9389 0.8793 0.8585 0.6351 0.9205 0.8215 KS Performance

- w / 1 prompt tok. 0.2176 0.1031 0.5208 0.8444 0.7053 0.7576 0.6228 0.5152 0.1790 0.6742 0.5140

- w / 2 prompt tok. 0.2240 0.1032 0.5202 0.8433 0.7054 0.7529 0.6233 0.5292 0.1865 0.6893 0.5177 w / 4 prompt tok. 0.2323 0.1068 0.5222 0.8509 0.7056 0.7593 0.6208 0.5537 0.1928 0.6833 0.5228 w / 6 prompt tok. 0.2412 0.1177 0.5225 0.8480 0.7086 0.7573 0.6212 0.5564 0.1962 0.6978 0.5267 w / 8 prompt tok. 0.2317 0.1078 0.5229 0.8432 0.7073 0.7520 0.6219 0.5646 0.1904 0.6995 0.5241 w / 16 prompt tok. 0.2351 0.1058 0.5229 0.8577 0.7082 0.7461 0.6209 0.5694 0.1975 0.6988 0.5262

- Table 10: Scalability of Q-Anchor Embedding (Prompt Tuned) across training steps (100–500) for prompt tuning (Ours: 500 steps).

User Engagement Risk Marketing Sensitivity Avg. Steps Active Concert Login Forest Fraud Money TakeOut Brand Promo Value AUC Performance

w / 100 0.6541 0.5775 0.8441 0.9703 0.9230 0.9382 0.8787 0.8329 0.6308 0.9091 0.8159 w / 200 0.6601 0.5774 0.8432 0.9701 0.9237 0.9403 0.8796 0.8454 0.6303 0.9139 0.8184 w / 300 0.6622 0.5806 0.8434 0.9698 0.9234 0.9428 0.8810 0.8484 0.6281 0.9167 0.8196 w / 400 0.6657 0.5821 0.8434 0.9718 0.9236 0.9433 0.8799 0.8513 0.6348 0.9167 0.8213 w / 500 0.6678 0.5844 0.8443 0.9716 0.9242 0.9439 0.8811 0.8535 0.6350 0.9194 0.8225

#### KS Performance

w / 100 0.2212 0.1059 0.5225 0.8429 0.7064 0.7425 0.6191 0.5195 0.1913 0.6701 0.5141 w / 200 0.2287 0.1041 0.5204 0.8433 0.7076 0.7488 0.6210 0.5452 0.1912 0.6797 0.5190 w / 300 0.2317 0.1100 0.5227 0.8418 0.7076 0.7543 0.6227 0.5478 0.1864 0.6866 0.5212 w / 400 0.2364 0.1116 0.5227 0.8466 0.7079 0.7553 0.6214 0.5538 0.1958 0.6917 0.5243 w / 500 0.2412 0.1177 0.5225 0.8480 0.7086 0.7573 0.6212 0.5564 0.1962 0.6978 0.5267

[Figure 18]

(a) User Engagement

[Figure 19]

(b) Risk

[Figure 20]

(c) Marketing Sensitivity

#### Figure 12: PCA Visualization of universal and prompt-tuned representation of 10 scenarios.

clustering and clearer separation between positive and negative samples compared to the base universal representation. This effect is especially pronounced in high-stakes scenarios such as Brand and Money, where subtle behavioral signals determine model performance. The convergence of t-SNE and PCA visualizations with consistent quantitative improvements in AUC and KS underscores that lightweight scenario-specific conditioning effectively reshapes the embedding geometry to better align with downstream decision boundaries—without modifying the underlying architecture.

Table 11: Offline performance of Q-Anchor embeddings. Base uses universal embeddings; Prompt Tuned adds lightweight, scenario-specific soft prompts. Both scenarios report AUC and KS, with Delinquency primarily optimizing KS.

Method AUC KS Scenario: IVR Response

Business SOTA 0.6362 0.2411 Q-Anchor (Base) 0.6534 0.2682 Q-Anchor (Prompt Tuned) 0.6863 0.3016

Scenario: Delinquency

Business SOTA - 0.1499 Q-Anchor (Base) - 0.1534 Q-Anchor (Prompt Tuned) - 0.1700

### D Pre-Deployment Offline Performance on Downstream Business

Before online deployment, we evaluate Q-Anchor embeddings againstcurrentproductionSOTA(highly optimized, domain-specific handcrafted features and specialized models) across two disparate tasks: IVR Response Prediction and Credit Delinquency Identification. Table 11 summarizes the results.

Robustness of Universal Representations. Even in its Base form, Q-Anchor consistently outperforms the Business SOTA across all metrics. Specifically, in the IVR scenario, it achieves a 2.71% absolute lift in KS over handcrafted features. This confirms that our pre-training objective effectively distills a foundational understanding of user behavior that generalizes across different financial contexts without any scenario-specific adaptation.

Steering Performance with Minimal Overhead. By introducing scenario-specific soft prompts, Q-Anchor (Prompt Tuned) further unlocks the latent potential of the universal backbone. It reaches a 0.3016 KS (+6.05% over SOTA) in IVR and a 0.1700 KS (+2.01% over SOTA) in delinquency prediction.

Crucially, this substantial performance gain is achieved with negligible incremental cost. As Q-Anchor reuses the expensive user-prefix KV-cache, the prompt tuning only involves optimizing a few learnable vectors at the query suffix. This architecture allows us to "steer" the same multi-billion parameter representation toward diverse business goals, achieving a superior balance between high-precision task adaptation and industrial-scale computational efficiency.

### E Deployment Detail of Q-Anchor Embedding

To serve embedding requests at Alipay scale, we deploy Q-Anchor Embedding with an incremental update pipeline that decouples heavy user-history encoding from lightweight scenario querying. Each day, the system ingests newly arrived multi-modal behaviors and updates the user profile by re-encoding only the affected modalities (i.e., modalities that receive new events) and refreshing their corresponding modality- and period-level summary tokens, rather than reprocessing all modalities and all raw events from the past 90 days end-to-end. For each modality𝑚∈M, a frozen backbone with a modality-specific event adapter encodes the day’s events into event-level tokens, which are further aggregated into a modality

summary token z𝑚(mdl) (Eq. 2). Newly generated tokens are updated into the historical token buffer to form the updated hierarchical

prefix e𝑖, while tokens outside the rolling window are expired. This design keeps representations fresh, bounded, and cost-stable.

For online inference, we exploit Query-as-Anchor with prefix KV-cache sharing (Fig. 4). For each user, the prefix e𝑖 is encoded once to build a shared KV cache that is reused across downstream scenarios. Given scenario queries {𝑞1, . . .,𝑞𝑛} (where 𝑛 is the number of scenarios), we process the queries sequentially while reusing the same shared prefix cache. For each query𝑞𝑗, the model only computes the short query suffix on top of the cached prefix, reducing the marginal per-scenario cost to 𝑂(𝐿𝑞𝑗 ) (and 𝑂( 𝑛𝑗=1 𝐿𝑞𝑗 ) for all scenarios) and enabling efficient multi-scenario re-anchoring (e.g., risk, marketing, engagement). Overall, the deployment provides (i) efficient daily refresh via delta updates and (ii) high-throughput multi-scenario serving by amortizing prefix encoding with KVcache reuse.

### F Future Work

While standard LLM scaling laws [30] suggest performance improves with model size, our experiments reveal a Scaling Paradox in user embeddings: larger backbones (1.5B–3B) exhibit gradient attenuation and performance stagnation compared to the 0.5B model, consistent with prior observations [14]. This indicates that embedding quality may follow a discriminative scaling trajectory, governed by the signal-to-parameter ratio rather than raw parameter count. Future work will investigate gradient recovery and adaptive parameter tuning to overcome optimization plateaus and systematically extend scaling benefits to larger models. Within the standardized, fair comparison setup of this study—without additional hyperparameter or learning-rate tricks—Q-Anchor achieves optimal performance at minimal cost on the 0.5B backbone. Accordingly, this base model will remain the primary configuration for our deployment and large-scale adoption across diverse Alipay scenarios.

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009

