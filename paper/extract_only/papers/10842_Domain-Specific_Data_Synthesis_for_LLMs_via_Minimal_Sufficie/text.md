[Figure 1]

[Figure 2]

[Figure 3]

## Domain-Specific Data Synthesis for LLMs via Minimal Sufficient Representation Learning

Tong Ye∗, Hang Yu∗†, Tengfei Ma, Xuhong Zhang, Jianguo Li, Peng Di, Peiyu Liu, Jianwei Yin, Wenhai Wang†

vivo AI Lab, Ant Group, Zhejiang University tongye0820@gmail.com, hyu1@e.ntu.edu.sg

# arXiv:2605.30039v2[cs.AI]29May2026

https://github.com/tongye98/DOMINO

Abstract. Large Language Models have demonstrated remarkable progress in general-purpose capabilities and can achieve strong performance in specific domains through fine-tuning on domainspecific data. However, acquiring high-quality data for target domains remains a significant challenge. Existing data synthesis approaches follow a deductive paradigm, heavily relying on explicit domain descriptions expressed in natural language and careful prompt engineering, limiting their applicability in real-world scenarios where domains are difficult to describe or formally articulate. In this work, we tackle the underexplored problem of domain-specific data synthesis through an inductive paradigm, where the target domain is defined only through a set of reference examples, particularly when domain characteristics are difficult to articulate in natural language. We propose a novel framework, DOMINO, that learns a minimal sufficient domain representation from reference samples and leverages it to guide the generation of domain-aligned synthetic data. DOMINO integrates prompt tuning with a contrastive disentanglement objective to separate domain-level patterns from sample-specific noise, mitigating overfitting while preserving core domain characteristics. Theoretically, we prove that DOMINO expands the support of the synthetic data distribution, ensuring greater diversity. Empirically, on challenging coding benchmarks where domain definitions are implicit, fine-tuning on data synthesized by DOMINO improves Pass@1 accuracy by up to 4.63% over strong, instruction-tuned backbones, demonstrating its effectiveness and robustness. This work establishes a new paradigm for domain-specific data synthesis, enabling practical and scalable domain adaptation without manual prompt design or natural language domain specifications.

Accepted by KDD 2026. ∗Core contribution; †Corresponding authors; Main contact: tongye0820@gmail.com Work partially done during an internship at Ant Group.

###### Contents

- 1 Introduction 3
- 2 Related Work 4
- 3 Methodology 5

- 3.1 Problem Definition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 3.2 Implicit Domain Representation Learning via Prompt Tuning . . . . . . . . . . . . . . . . . . 6
- 3.3 Minimal Sufficient Domain Representation Learning . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.4 Data Synthesis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 4 Experiments 8

- 4.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 4.2 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 4.3 Deeper Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- 4.3.1 Visualizing Synthetic Sample Distribution. . . . . . . . . . . . . . . . . . . . . . . . . 10
- 4.3.2 Interpretability of 𝐷∗. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 4.3.3 Generalization of DOMINO to More Task Domains. . . . . . . . . . . . . . . . . . . . 11
- 4.3.4 Impact of Temperature T. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 4.3.5 Impact of Domain-level Soft Token Counts. . . . . . . . . . . . . . . . . . . . . . . . . 12
- 4.3.6 Impact of Reference Data Percentage. . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 4.3.7 Impact of Hyperparameter 𝜆. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 4.3.8 Further Analysis. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- 5 Conclusion 13
- 6 Acknowledgements 14 A Appendix 17

- A.1 Theoretical Guarantee . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- A.1.1 The Proof of Proposition 1. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.1.2 The Proof of Proposition 2. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.1.3 The Proof of Proposition 3. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.1.4 The Proof of Proposition 4. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- A.2 Experimental Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- A.2.1 Target Domain Data Statistics. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- A.2.2 Case Study: Interpretability of 𝐷∗. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- A.2.3 Generalization of DOMINO to More Task Domains. . . . . . . . . . . . . . . . . . . . 20
- A.2.4 Comparison of Synthetic Samples vs. Reference Samples (Equal Size). . . . . . . . . . 21
- A.2.5 Impact of Reference Data Percentage on Live Code Execution. . . . . . . . . . . . . . 21
- A.2.6 Impact of more reference samples for MAGPIE. . . . . . . . . . . . . . . . . . . . . . 21
- A.2.7 Baseline Prompts. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- A.2.8 Post-processing. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- A.2.9 Domain Adaptation Configurations. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- A.2.10 A Case Study of Domain-Specific Synthetic Samples. . . . . . . . . . . . . . . . . . . . 23

- A.3 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- A.4 Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

###### 1. Introduction

Recent advances in Large Language Models (LLMs) (DeepSeek-AI, 2024; Gemini Team, 2024; OpenAI, 2022; Qwen et al., 2025) have revolutionized natural language processing, allowing for unprecedented performance in coding (Jain et al., 2024; Zhuo et al., 2025), mathematics (Cobbe et al., 2021; Ye et al., 2025), and various downstream tasks (Nie et al., 2024; Plaat et al., 2024; Zhang

- et al., 2024a). In real-world applications, the prevailing approach to downstream adaptation involves leveraging high-quality, domain-specific data for supervised instruction tuning (Ding et al., 2023; Zhang et al., 2024b). This approach rests on a crucial assumption that practitioners are able to obtain a sufficient number of annotated examples to explicitly characterize the target domain.

However, current data synthesis methods share a fundamental, yet often overlooked, limitation: they presuppose the existence of an explicit, human-articulable domain definition. These methods, whether based on Instruction Evolution (Wang et al., 2023a; Xu et al., 2024), Key-point-driven synthesis (Huang et al., 2024a,b), or System Prompts (Xu et al., 2025), all require translating domain knowledge into textual instructions. (Details of these related methods are shown in Section 2.) This approach is analogous to deductive reasoning—starting from a general rule (the prompt) to generate specific instances. This paradigm breaks down when we consider how humans often learn. We excel at inductive reasoning: when confronted with a new concept—be it a genre of music, a style of art, or a type of logical puzzle—we do not begin with a formal, axiomatic definition. Instead, we learn by observing a handful of canonical examples and inducing the underlying rules, patterns, and boundaries of the domain. This raises a critical question for machine learning:

“How can we synthesize domain-specific data when the target domain is defined implicitly by a set of reference examples, rather than explicitly in natural language?”

This problem is not a niche concern but is central to practical domain adaptation. It arises in high-value scenarios such as emerging scientific fields where terminology is still fluid, rapidly evolving cultural trends like online slang, or with proprietary business logic that is confidential or poorly documented. In all these cases, the domain is governed by subtle conventions and emergent patterns that resist simple verbalization. While providing examples is easy, writing a formal definition is nearly impossible. A tempting but flawed approach is to simply use the available examples for supervised fine-tuning (SFT). With a limited dataset, however, SFT encourages the model to memorize superficial features of the training data rather than generalize the underlying domain principles (Chu et al., 2025). This leads to poor performance on novel, out-of-distribution tasks. This risk is not merely theoretical; our own experiments (Table 1) confirm that naive SFT on a reference set can fail to improve, or even degrade, results, highlighting its inadequacy for true domain adaptation.

The challenge, therefore, is not merely one of data quantity—a problem that synthesis can, in principle, solve by generating as many samples as desired—but of learning quality. To truly learn a domain from examples, a model must distinguish its core, generalizable principles from the idiosyncratic details of each sample. Our key insight is that this can be achieved by learning a minimal sufficient representation—a latent embedding that is sufficient to capture the essential characteristics of the domain while being minimal enough to discard irrelevant, sample-specific noise that leads to overfitting. As shown in Figure 1, we propose a framework that derives this minimal representation from reference samples and then uses it to guide an LLM in generating novel, diverse samples that adhere to the domain’s induced principles.

We call our framework DOMINO (DOmain-specific data synthesis through MINimal sufficient representatiOn learning). Drawing on information-theoretic principles, DOMINO operates through two synergistic components. First, it uses prompt tuning to learn a continuous domain embedding that

Reference Samples Synthetic Samples

|You are given two integers, n and k. An array of distinct positive integers is called a k-avoiding array if there does not exist any pair of distinct elements that sum to k. Return the minimum possible sum of a k-avoiding array of length n.<br><br>Example 1: Input: n = 5, k = 4 Output: 18 Explanation: Consider the k-avoiding array [1,2,4,5,6], which has a sum of 18. It can be proven that there is no kavoiding array with a sum less than 18.<br>Example 2: Input: n = 2, k = 6 Output: 3 Explanation: We can construct the array [1,2], which has a sum of 3. It can be proven that there is no k-avoiding array with a sum less than 3.<br><br><br>Constraints: 1 <= n, k <= 50| |
|---|---|
| | |

You are given an array of integers A (size N). If the sum of the subarray A[i] to A[j] (inclusive) is an even number, return the number of elements in this subarray. Note that a subarray is a contiguous sequence of elements of the array.

Input: The input is given from Standard Input in the following format: N A1 A2 ... AN

Output: Print the sum of the number of subarrays for each i and j.

Example Input

- 3 1 2 3 Output
- 4 Explanation: The following four subarrays will be evensum:

- * [1,2] * [2,3] * [1,2,3] * [3] Constraints
- * 1 \le N \le 100
- * 0 \le A_i \le 100

𝑑 , 𝑑 , ···, 𝑑

#### LLMs

Domain Representation

- Figure 1 | DOMINO encodes domain characteristics into latent embeddings derived from reference samples, which are then used to guide LLMs in generating new samples. While the domain characteristics are not explicitly defined in natural language, a comparison between reference and synthetic samples at least reveals a shared structural pattern (e.g., description, examples, constraints), even though the specific content differs, highlighting the DOMINO’s ability to capture the domain’s distribution—at least its visible format.

maximizes the likelihood of the reference examples, ensuring sufficiency. Second, it introduces a contrastive disentanglement mechanism that explicitly separates shared domain-level patterns from unique sample-level features, enforcing minimality. By optimizing a combined objective that enforces both data reconstruction fidelity and information minimality, DOMINO learns to abstract the domain’s “first principles” from the examples provided. In summary, the primary contributions are:

- • We formalize and tackle the problem of domain synthesis from implicit examples, mirroring human inductive reasoning. We develop DOMINO, a framework that learns a minimal sufficient representation by combining prompt tuning with contrastive disentanglement to capture essential domain patterns and promote generalization.
- • Theoretically, we prove that DOMINO expands the support of the synthetic data distribution compared to baselines, ensuring greater diversity in generated examples.
- • Empirically, we show DOMINO’s effectiveness and robustness across diverse tasks. On coding tasks with implicit domains, it improves Pass@1 by up to 4.63% over strong instruction tuned models, and consistently outperforms baselines on both coding and instruction-following tasks.

###### 2. Related Work

In recent years, with the advancement of LLMs, data synthesis has attracted increasing research attention due to its unique advantages (Huang et al., 2025b; Li et al., 2024; Long et al., 2024). Broadly,

existing LLM-driven synthetic data approaches can be categorized into three main categories:

Instruction Evolution: This category involves leveraging iterative and carefully crafted prompt engineering to guide LLMs in expanding instruction sets. Specifically, Self-Instruct (Wang et al., 2023a) employs a curated seed pool and task-specific prompts to generate additional instruction data. Meanwhile, Evol-Instruct (Xu et al., 2024) uses strategically designed prompts to guide LLMs in modifying existing instructions or creating new ones from both depth and breadth perspectives. While effective, these methods rely heavily on complex natural prompt engineering within the domain.

Key-Point-Driven: These methods guide LLMs to synthesize data by extracting knowledge and key points from the target domain. Li et al. (2024) employ GPT-4 to construct a taxonomy of concepts; however, the resulting synthetic data distribution deviates significantly from real data. Huang et al. (2024a) build an explicit concept pool from extracted domain concepts. In contrast, KPDDS (Huang

- et al., 2024b) constructs topic–key point pairs from a large number of seed examples to capture the frequency of domain-specific topics. It then samples multiple topic–key point combinations to guide the generation of new samples. Despite their effectiveness, these approaches rely heavily on extensive prior knowledge and assume that the target domain possesses a hierarchical conceptual structure. Moreover, they require such internal structures and key points can be explicitly described in natural language.

System Prompt Guided: Recently, MAGPIE (Xu et al., 2025) introduced a method that constructs instruction data by prompting aligned LLMs with a pre-defined query template. By incorporating a domain-specific system prompt, MAGPIE can synthesize large-scale domain data. However, it necessitates the deliberate design of the system prompt to guide the LLM toward the target domain.

However, all three categories of methods break down in real-world scenarios when the target domain cannot be explicitly described in natural language and no prior domain knowledge is available. In contrast, our proposed method, DOMINO, effectively overcomes this limitation by enabling the synthesis of a virtually unlimited quantity of target-domain data without requiring any prior knowledge or manual prompt engineering.

###### 3. Methodology

###### 3.1. Problem Definition

We consider the problem of synthesizing data for a specific domain, denoted by D, with a corresponding observation space X. We assume access to a limited set of reference data within X, that is, 𝑿(1:𝑛) = {𝑿(1), 𝑿(2), · · · , 𝑿(𝑛)}. Each 𝑿(𝑖) consists of both input and output sequences expressed as plain text (i.e., the data for supervised fine-tuning of LLMs). The size of the reference set, 𝑛, typically ranges from tens to hundreds of examples, a scale limited by high collection and annotation costs. Even within this range, distinct challenges require a carefully designed representation. When 𝑛 is on the lower end (e.g., tens of examples), the primary risk is overfitting to the idiosyncratic details of the few available samples. As 𝑛 grows larger (e.g., hundreds), the challenge evolves. A representation that is merely sufficient for reconstruction but not minimal may still overfit in a more subtle way. It might encode spurious correlations or incidental stylistic details common across the dataset by chance, mistaking them for core domain principles. For example, it might learn that problem descriptions frequently use a specific turn of phrase, rather than capturing the abstract structural pattern of the domain itself. Therefore, our objective is to learn a minimal sufficient representation that is robust across this spectrum: it must be general enough to avoid overfitting on smaller datasets, yet discerning enough to separate the essential, generalizable domain characteristics from coincidental, sample-specific artifacts in larger ones. The following details DOMINO.

###### 3.2. Implicit Domain Representation Learning via Prompt Tuning

To accomplish our goal of synthesizing domain-specific data, we need to (i) learn a suitable representation of the domain D and (ii) leverage that representation to generate new data within the domain. A promising approach is to employ a pre-trained LLM as an implicit encoder to represent the domain, capitalizing on its extensive knowledge base. Mathematically, let 𝑫 = [𝒅1, · · · , 𝒅𝑘] denote the domain representation comprising 𝑘 domain soft tokens, each with dimension 𝑑. We aim to optimize 𝑫 such that 𝑝(𝑫|X) = ∏𝑛𝑖=1 𝑝(𝑫|𝑿(𝑖)) is maximized. According to Bayes’ rule,

𝑝(𝑫)𝑝(𝑿(𝑖)|𝑫) 𝑝(𝑿(𝑖))

∝ 𝑝(𝑿(𝑖)|𝑫), (1)

𝑝(𝑫|𝑿(𝑖)) =

where 𝑝(𝑿(𝑖)|𝑫) signifies the probability of generating the data example 𝑿(𝑖) given the domain representation 𝑫, as determined by an LLM. In this context, we assume a uniform prior distribution for the domain representation, 𝑝(𝑫) ∝ 1, reflecting our lack of prior knowledge about the domain. Consequently, the problem amounts to soft prompt tuning, where our objective is to identify the domain-level soft token prompt 𝑫 that maximizes the likelihood of all reference data 𝑿(1:𝑛):

∑𝑛

1

log 𝑝(𝑿(𝑖)|𝑫). (2)

L1 = −

𝑛

𝑖=1

Unlike explicit encoders that require structured inputs (e.g., labeled features), LLMs leverage their pre-training on vast corpora to infer domain-specific representations directly from natural language examples. Moreover, soft prompt tuning avoids full model fine-tuning, preserving the LLM’s general capabilities while specializing its behavior for D. In addition, the same framework applies to diverse domains without architectural changes, as the domain representation 𝑫 adapts to each domain’s unique characteristics. Subsequently, new data can be synthesized by inputting the estimated 𝑫 into the LLM, drawing samples according to 𝑝(𝑿|𝑫), which will be discussed in Section 3.4.

###### 3.3. Minimal Sufficient Domain Representation Learning

While the soft prompt tuning approach described above seems promising, it suffers from a critical issue: overfitting. Given a small number of reference examples (tens to hundreds), the learned domain representation 𝑫 can become overly tailored to the training data. This may result in synthetic data that closely resembles the reference examples, failing to capture the broader range of possibilities within the observation space X of the domain D. Indeed, this approach primarily aims to learn a sufficient representation for the domain D, which we define as follows:

- Definition 1. (Sufficiency) A representation 𝑫 is considered sufficient for domain D if the conditional distribution 𝑝(𝑿(𝑖)|𝑫, D) = 𝑝(𝑿(𝑖)|𝑫) for all 𝑿(𝑖) ∈ X. In other words, 𝐼(D; 𝑿(𝑖)|𝑫) = 0, indicating that 𝑫 captures all information about D contained in 𝑿(𝑖), where 𝐼 is mutual information.

However, sufficiency alone does not guarantee generalization. A representation that retains unnecessary details (e.g., unique attributes of individual examples) risks overfitting. To mitigate this, we seek a minimal sufficient representation 𝑫∗:

- Definition 2. (Minimal Sufficiency) A sufficient representation 𝑫∗ is minimal if and only if it is a function of every other sufficient representation 𝑫, i.e., 𝑫∗ = 𝑓 (𝑫). According to the deterministic function property of mutual information, 𝐼(𝑫∗, 𝑿(𝑖)) = 𝐼( 𝑓 (𝑫), 𝑿(𝑖)) ≤ 𝐼(𝑫, 𝑿(𝑖)), indicating that 𝑫∗ discards all information irrelevant to D.

Minimal sufficiency ensures 𝑫∗ focuses on domain-wide patterns (e.g., the general topic, task type, or style) rather than memorizing sample-specific traits (e.g., specific facts, word choices, or constraints). This process is the computational analog to the inductive leap humans make: identifying

the essential, shared “rules” of a concept while ignoring the incidental details of any single example. This enables the generation of diverse synthetic data that spans the observation space X, rather than replicating training examples. To enforce minimal sufficiency, we explicitly separate domain-level

and sample-level information by introducing sample-specific representations 𝑺(𝑖) = [𝒔1(𝑖), · · · , 𝒔ℓ(𝑖)], one for each individual example 𝑿(𝑖). These sample-level soft tokens encode unique aspects of each example 𝑿(𝑖). We optimize 𝑫∗ and 𝑺(𝑖) via a contrastive loss:

###### ∑𝑛

1

𝑝(𝑿(𝑖)|𝑫∗,𝑺(𝑖)) ∑𝑗≠𝑖 𝑝(𝑿(𝑗)|𝑫∗,𝑺(𝑖))

. (3)

log

L2 = −

𝑛

𝑖=1

The numerator guarantees that 𝑫∗ and 𝑺(𝑖) jointly reconstruct 𝑿(𝑖). On the other hand, the denominator penalizes 𝑫∗ for encoding information that could facilitate 𝑺(𝑖) to reconstruct unrelated examples 𝑿(𝑗) for 𝑗 ≠ 𝑖. This design embodies the principle that 𝑫∗ should capture shared aspects across all samples in the domain, while 𝑺(𝑖) focuses on the unique aspects of the specific sample 𝑿(𝑖). By forcing the LLM to rely on 𝑺(𝑖) to distinguish 𝑿(𝑖) from other samples 𝑿(𝑗), we prevent 𝑫∗ from overfitting to the specifics of individual examples. This, in turn, promotes a more generalizable, minimal sufficient domain representation (i.e., an inductive abstraction). With this loss, we can prove:

- Proposition 1. Given 𝑿(𝑖) ∈ 𝑿(1:𝑛), L2 directly maximizes 𝐼(𝑺(𝑖), 𝑿(𝑖)|𝑫∗), ensuring that 𝑺(𝑖) captures unique information about 𝑿(𝑖).
- Proposition 2. L2 directly minimizes 𝐼(𝑺(𝑖); 𝑫∗), promoting disentanglement between 𝑺(𝑖) and 𝑫∗.

The detailed proofs of Proposition 1 and Proposition 2 are listed in Appendix A.1.1 and A.1.2. Finally, the overall loss function for learning the minimal sufficient representation 𝑫∗ for the domain

- D is a combination of the domain-level likelihood and the contrastive loss:

L = L1 + 𝜆L2, (4) where 𝜆 is a weighting hyperparameter that controls the trade-off between maximizing data likelihood and promoting domain representation disentanglement. A higher 𝜆 encourages the LLM to learn a more minimal, generalizable domain representation by preventing overfitting to individual samples. Further, we theoretically prove the connection between optimization objective L and the information-theoretic principles of minimal sufficiency (Proposition 3, proof in Appendix A.1.3).

- Proposition 3. The optimization objective L serves as a tractable proxy for learning a representation 𝑫∗ that approximates the information-theoretic properties of a minimal sufficient statistic. Specifically, minimizing L jointly encourages: 1. Sufficiency: The maximization of mutual information 𝐼(𝑫∗; 𝑿) , ensuring 𝑫∗ captures relevant domain information. 2. Minimality: The minimization of mutual information 𝐼(𝑫∗;𝑺) , ensuring 𝑫∗ is disentangled from sample-specific information.

###### 3.4. Data Synthesis

In practice, the minimal sufficient domain representation 𝑫∗ and the sample-level prompt 𝑺(𝑖) are jointly optimized during training using the objective L in equation 4 (on the left side of Figure 2). Once training is complete, only the learned 𝑫∗ is used to guide the LLM during the data synthesis phase (i.e., sampled from distribution 𝑝(𝑿|𝑫∗)), as shown on the right side of Figure 2. Notably, the LLM is fixed throughout both the training and synthesis stages. The key intuition is that by disentangling 𝑫∗ from sample-specifics, we prevent it from merely memorizing the training examples and instead encourage the generation of more diverse outputs. Proposition 4 provides the formal theoretical guarantee for this. It proves that, under quantifiable conditions, the distribution guided by our minimal sufficient representation, 𝑝(𝑿|𝑫∗), has a strictly larger effective support (the set of high-probability outcomes) than the distribution 𝑝𝐷 guided by a standard, non-minimal prompt.

###### Minimal Sufficient Domain Representation Learning Synthetic Processing

###### Training Inference

###### Reference Sample

###### Synthetic Sample

𝑡 … 𝑡

𝑡 ( ) 𝑡 ( ) …

𝑡 ( )

… [MASK]

… … …

[MASK]

### LLM

### LLM

[Figure 4]

[Figure 5]

…

𝑑 ∗ … 𝑑 ∗

𝑠 ( ) … 𝑠 ( )

𝑒 (𝑡 ( )) 𝑒 (𝑡 ( )) 𝑑 ∗ 𝑑 ∗

𝑒 (𝑡 ( )) …

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Domain Soft Tokens

Sample Soft Tokens

Reference Sample

Domain Soft Tokens

- Figure 2 | The left side represents the minimal sufficient domain representation learning process,

where 𝑡1, · · · , 𝑡𝑇 denote the raw natural language tokens of a reference sample, 𝑒𝑚 denotes the corresponding embedding. Domain-level soft tokens 𝑫∗ and sample-level soft tokens 𝑺(𝑖) are jointly optimized. The right side represents using only optimized 𝑫∗ to synthesize domain-specific samples.

- Proposition 4. Let 𝑝𝐷 and 𝑝𝐷∗ denote the distributions 𝑝(𝑿|𝑫) (from vanilla prompt tuning) and 𝑝(𝑿|𝑫∗) (from minimal sufficient, disentangled prompt tuning via L1 + 𝜆L2), respectively. For any 𝜖 ∈ (0, 1/𝑒), define the 𝜖-support of 𝑝, supp𝜖(𝑝) = {𝑥 : 𝑝(𝑥) > 𝜖}, with cardinality 𝑆𝜖𝑝 = |supp𝜖(𝑝)|. Define the uniformity gap for 𝑝 as 𝛿𝑝 = log𝑆𝜖𝑝 − 𝐻(𝑝), where 𝐻(𝑝) is the entropy of 𝑝. If

1

(5) then 𝑆𝜖𝑝

𝐻(𝑝𝐷∗) − 𝐻(𝑝𝐷) > 𝛿𝑝𝐷 + 𝜖log

𝜖

𝐷∗ > 𝑆𝜖𝑝𝐷; that is, the 𝜖-support of 𝑝(𝑿|𝑫∗) is strictly larger than that of 𝑝(𝑿|𝑫). Proof. See Appendix A.1.4. □

###### 4. Experiments

###### 4.1. Experimental Setup

Domain and Benchmark Selection. To effectively evaluate DOMINO, we require benchmarks that embody our core problem definition: domains where characteristics are implicit, evolving, and difficult to articulate textually. The recently proposed dynamic benchmark LiveCodeBench (Jain et al., 2024) provides a perfect testbed. In competitive programming, the “domain” is not a static category like “code generation”, but the constantly evolving style, complexity, and set of algorithmic patterns characteristic of platforms during a specific time frame. It is notoriously difficult to write a single prompt that captures “the essence of a typical LeetCode problem from late 2024”, yet it is easy to collect examples from that period. Following these principles, we select two distinct domains from the recently proposed dynamic benchmark LiveCodeBench (Jain et al., 2024): Live Code Generation and Live Code Execution. We define a temporal cutoff point for each domain: domain samples collected before this point are used as reference data (input-output pairs), while domain samples collected afterward serve as the test set. This setup directly tests the model’s ability to induce generalizable principles rather than memorizing past examples. The details are in Appendix A.2.1.

LLM Backbone Selection. We select OPENCODER-7B (Huang et al., 2025a), QWEN2.5-CODER-

- 7B (Hui et al., 2024), and QWEN2.5-CODER-14B (Hui et al., 2024) as the LLM backbones, as they are among the top-performing LLMs in general code tasks. We use these LLMs (Instruction version) to synthesize domain-specific data that aligns with the distributions of the two target domains.

Baselines. We detail the baseline methods; all corresponding prompts are provided in Appendix A.2.7.

- • Direct Prompt: We directly prompt the LLM backbone (both the Base and Instruction versions) to assess the LLM backbone’s basic capabilities on the two target domains.
- • Reference SFT: We use the reference data from the target domain to perform supervised finetuning (SFT) on the corresponding LLM Base version, in order to evaluate its domain adaptation capability.
- • MAGPIE-Human: We manually inspect the reference data and attempt to summarize the domain characteristics in natural language, which are then used as the MAGPIE system prompt to guide the LLM in synthesizing domain-specific samples.
- • MAGPIE-Few Shot: We randomly select three reference samples and prompt the LLM to summarize the underlying domain characteristics in natural language. This self-generated domain description is then used as the MAGPIE system prompt in synthesizing domain-specific samples.
- • DOMINO-Direct Domain: We directly use the L1 objective in Section 3.2 to obtain the implicit domain representation, which is then used to guide the LLM in synthesizing domain samples.

DOMINO Train Details. For both DOMINO-Direct Domain and DOMINO, we use input sequences from domain reference samples for representation learning. We set the hyperparameter 𝜆 to 1, with 256 domain-level soft tokens in both settings, and 256 sample soft tokens additionally used in DOMINO. Note that the representation training is conducted using Instruction version of the LLM.

Synthesis Process and Post-processing. For MAGPIE-{Human, Few Shot}, DOMINO-Direct Domain, and DOMINO, we first synthesize domain-specific input sequences using vLLM (Kwon et al., 2023) with the Instruction version of LLMs, setting temperature T to 0.8. For fair comparison, we generate 80K input sequences for Live Code Generation domain and 40K for Live Code Execution domain across all four methods. A unified post-processing pipeline is then applied to filter out lowquality inputs (using the same LLM; details in Appendix A.2.8). After filtering, the same LLM is used to generate output sequences, forming input-output pairs for subsequent supervised fine-tuning.

Domain Adaptation. For all synthesized samples across methods generated using the Instruction version LLMs, we perform supervised fine-tuning on the corresponding Base versions using LLAMAFACTORY (Zheng et al., 2024). Detailed training configurations are provided in Appendix A.2.9.

###### 4.2. Main Results

The main results are presented in Table 1, with each section block corresponding to one LLM backbone. We can find that: (i) Directly prompting the Instruction version of an LLM consistently outperforms its Base version across both domains, which is expected since instruction-tuned LLMs are explicitly aligned to follow system instructions. (ii) When fine-tuned directly on the domain reference data (Reference SFT), the LLM demonstrates a certain level of domain adaptation in the Live Code Generation domain. However, in the Live Code Execution domain, we observe a performance drop, suggesting that direct fine-tuning is prone to overfitting, especially in domains where the LLM already performs well out of the box. (iii) Both MAGPIE-Human and MAGPIE-Few Shot yield performance gains. Interestingly, MAGPIE-Few Shot slightly outperforms MAGPIE-Human, indicating that allowing the LLM to infer domain characteristics from a few reference samples is more effec-

- Table 1 | Main Results of DOMINO and the compared methods on the two target domains. DOMINODirect Domain is an ablation of our method using only L1 without L2.

Live Code Generation Live Code Execution Methods Pass@1 Pass@5 Pass@10 Pass@1 Using OPENCODER-8B-Instruct Huang et al. (2024b) as Backbone

OPENCODER-8B-Base 7.96 14.56 17.36 25.93 OPENCODER-8B-Instruct 10.00 17.23 19.76 37.96

Reference SFT 9.34 14.13 16.88 27.78 MAGPIE-Human 8.77 13.83 15.88 32.41

MAGPIE-Few Shot 9.81 15.32 17.49 34.26

DOMINO-Direct Domain 10.15 16.24 19.39 38.88 DOMINO 12.63 16.81 20.44 42.59 Using QWEN2.5-CODER-7B-Instruct Hui et al. (2024) as Backbone

QWEN2.5-CODER-7B-Base 9.04 15.35 17.96 45.37 QWEN2.5-CODER-7B-Instruct 16.88 21.73 23.35 55.55

Reference SFT 13.47 21.59 23.95 36.11 MAGPIE-Human 12.45 20.85 22.72 42.59

MAGPIE-Few Shot 14.36 20.98 24.23 48.15 DOMINO-Direct Domain 15.74 24.66 28.14 50.92

DOMINO 17.31 26.81 29.94 56.48 Using QWEN2.5-CODER-14B-Instruct Hui et al. (2024) as Backbone

QWEN2.5-CODER-14B-Base 19.46 26.42 27.4 45.37 QWEN2.5-CODER-14B-Instruct 23.35 30.13 31.73 59.26

Reference SFT 19.64 24.86 26.33 38.88 MAGPIE-Human 20.78 27.17 28.74 57.41

MAGPIE-Few Shot 20.12 25.27 27.45 57.41 DOMINO-Direct Domain 21.62 28.97 31.73 60.19

###### DOMINO 24.75 30.45 33.24 62.04

tive than relying on manually written system prompts in unfamiliar domains. This highlights the LLM’s capacity to capture subtle and latent domain patterns that may be overlooked by humans. (iv) Finally, DOMINO-Direct Domain and DOMINO achieve the best overall performance. Notably, DOMINO shows that synthetic domain-specific samples generated by the Instruction version of an LLM can be used to fine-tune its Base version, resulting in performance that even surpasses the original Instruction version. This finding suggests that the domain adaptation capabilities of LLMs can be further enhanced through self-generated, domain-aligned data.

4.3. Deeper Analysis

- 4.3.1. Visualizing Synthetic Sample Distribution.

To better illustrate the relationship between synthesized and reference data distributions, we compare samples generated by DOMINO and MAGPIE-Few Shot. We use CODET5-EMBEDDING (Wang

- et al., 2023b) to obtain semantic embeddings and visualize them via t-SNE (van der Maaten and Hinton, 2008). As shown in Figure 3 and Figure 4, DOMINO’s synthesized samples are more widely dispersed in the embedding space and exhibit greater consistency with the distribution of reference samples. In contrast, the samples synthesized by MAGPIE-Few Shot are cluster tightly and show weaker alignment with reference samples. These results highlight DOMINO’s ability to generate diverse, domain-consistent data that better captures underlying domain traits.

[Figure 15]

- Figure 3 | t-SNE of latent embeddings for synthetic and reference samples in Live Code Generation domain.

[Figure 16]

Figure 4 | t-SNE of latent embeddings for synthetic and reference samples in Live Code Execution domain.

- Table 2 | Qualitative Analysis: From Sample Mimicry to Domain Abstraction.

Source Example Problem

k-avoiding array: Given n and k, find the minimum sum of a k-avoiding array of length n where no two distinct elements sum to k.

Reference Sample

Synthetic from 𝑫 (Overfitted)

k-subsequence: Given a string n of length k, find the number of k-subsequences.

1. Graph Path: Find the distance between two airports in a tree-like network. 2. Arithmetic Subsequence: Find the longest arithmetic subsequence with a given difference. 3. Palindrome Operations: Find the minimum operations to make a string of digits a palindrome.

Synthetic from 𝑫∗ (Diverse but In-Domain)

###### 4.3.2. Interpretability of 𝐷∗.

To better interpret the specific domain patterns captured by 𝑫∗, we present a case study using samples that exemplify the clear distinction between the standard synthetic set (𝑫) and the proposed diverse set (𝑫∗). Critically, the synthetic samples from 𝑫∗ were specifically selected from regions of the tSNE embedding space that are distant from the dense cluster representing 𝑫, visually confirming their diversity. The specific samples are provided in the Appendix A.2.2. The qualitative comparison (concluded in Table 2) provides concrete evidence: Even with a relatively large reference set (713 samples for Live Code Generation), the model trained with only a sufficiency objective (𝑫, from L1) still overfits to superficial details. The addition of our contrastive disentanglement objective (L2) is crucial for pushing the model beyond mere memorization. It forces the model to learn a minimal representation (𝑫∗) that successfully performs the inductive step of abstracting core domain principles, leading to the generation of truly diverse and novel samples.

###### 4.3.3. Generalization of DOMINO to More Task Domains.

To demonstrate DOMINO’s effectiveness across a more diverse set of tasks, we have extended our experiments to the critical and well-defined Instruction Following domain. The detailed synthesis procedure and experimental results are provided in Appendix A.2.3 and Table 3. As shown, DOMINO

- Table 3 | Results of DOMINO and the compared methods on the NLP tasks.

Methods IF Average

Instruct Following (IF) Paraphrase Simplify Story Gen. Summarize Qwen2.5-7B-Base 44.50 41.23 48.27 53.84 34.65

Qwen2.5-7B-Instruct 51.91 50.62 60.83 54.58 41.61

Reference SFT 45.78 44.07 49.42 52.29 37.36 MAGPIE Few Shot 51.19 47.34 54.78 57.91 44.73

DOMINO-Direct Domain 53.78 52.49 58.37 57.91 46.38 DOMINO 55.39 54.28 61.96 58.23 47.12

consistently outperforms all baselines across the four subtasks, boosting the average score by 3.48 points over the strong instruction-tuned backbone and by 1.61 points over DOMINO-Direct Domain. These results reinforce the original findings and demonstrate that DOMINO’s ability to learn and generalize from implicit domain characteristics is not limited to coding but extends to a diverse range of NLP tasks.

- 4.3.4. Impact of Temperature T.

In LLM-driven data synthesis, the temperature parameter controls the LLM’s output diversity. To assess its impact on DOMINO, we vary T from 0.2 to 1.0. As shown in Table 4, DOMINO achieves relatively stable performance across different T settings. This robustness is attributed to the strong guidance of the domain representation 𝑫∗, which effectively constrains generation to preserve domain characteristics as much as possible.

Table 4 | Results of varying T on synthetic process in the Live Code Generation using QWEN2.5CODER-7B Instruction as the LLM backbone.

Temperature

Live Code Generation Pass@1 Pass@5 Pass@10

T = 0.2 16.35 22.88 24.55 T = 0.4 16.77 24.67 27.54 T = 0.6 15.45 25.17 29.34

- T = 0.8 17.31 26.81 29.94

- T = 1.0 16.53 25.95 29.34

| | | |17.3|1%<br><br>17.8|5|
|---|---|---|---|---|---|
| | |16.3|5%| | |
| | | | | | |
| |.5|6%| | | |
| | | | | | |
| | | | | | |

64 128 256 512

Domain Soft Token Count

10

12

14

16

18

20

Pass@1(%)

11

%

Figure 5 | Results of the impact of domain soft token counts in Live Code Generation using QWEN2.5-CODER-7B Instruction as backbone.

- 4.3.5. Impact of Domain-level Soft Token Counts.

𝑘] are shared across all reference samples within the target domain and are designed to capture the domain’s underlying characteristics. The number of domain soft tokens, 𝑘, is a tunable hyperparameter that determines the capacity of this representation. A larger 𝑘 enables the encoding of more domain-specific features across a broader embedding space. To assess the impact of 𝑘, we vary it over the set {64, 128, 256, 512}. As shown in Figure 5, performance consistently improves with larger values of 𝑘, suggesting that higher-capacity domain representations better capture fine-grained domain characteristics and provide stronger guidance for synthetic data generation.

In DOMINO, the domain-level soft tokens [𝒅1∗, · · · , 𝒅∗

Table 5 | Results of varying 𝜆 in DOMINO in the Live Code Generation domain using QWEN2.5CODER-7B Instruction as LLM backbone.

30

Pass@1 Pass@5 Pass@10

PassRate(%)

25

Live Code Generation Pass@1 Pass@5 Pass@10 𝜆 = 0.25 15.38 22.44 24.38

20

Hyperparameter 𝜆

15

100% 80% 60% 40% 20%

Reference Samples Percentage

𝜆 = 0.5 15.95 22.88 24.77

- 𝜆 = 1 17.31 26.81 29.94

- 𝜆 = 2 17.58 25.75 28.58 𝜆 = 4 16.58 26.38 28.89

- Figure 6 | Impact of reference sample percentage in Live Code Generation domain using QWEN2.5CODER-7B Instruction as the backbone.

- 4.3.6. Impact of Reference Data Percentage.

In DOMINO, the domain characteristics are inserted into domain soft tokens by leveraging reference data. To evaluate the impact of reference data quantity, we vary the proportion of the original reference set from 100% to 20%. As shown in Figure 6, reducing the proportion of reference data consistently degrades performance. This aligns with the intuition: more reference samples provide a more comprehensive view of the domain, enabling more effective representation learning. Furthermore, we also evaluate the impact of reference data quantity on the Live Code Execution task; the corresponding results and analysis are presented in Appendix A.2.5.

- 4.3.7. Impact of Hyperparameter 𝜆.

In the original DOMINO optimization objective, the hyperparameter 𝜆 controls the trade-off between maximizing data likelihood and promoting domain representation disentanglement. To evaluate its impact on performance, we vary 𝜆 across the set {0.25, 0.5, 1, 2, 4}. As shown in Table 5, setting 𝜆 ≥ 1 leads to a noticeable performance improvement compared to 𝜆 < 1. This indicates that assigning greater weight to disentanglement helps the LLM learn a more minimal and generalizable domain representation by mitigating overfitting to individual samples.

- 4.3.8. Further Analysis.

1. Case studies provide an intuitive representation of domain-specific synthetic samples in Appendix A.2.10; 2. A comparison of synthetic against reference samples, using equal numbers, shown in Appendix A.2.4; 3. More samples for MAGPIE-Few Shot shown in Appendix A.2.6.

###### 5. Conclusion

In this work, we propose DOMINO, a method for domain-specific data synthesis under implicit supervision. By learning minimal sufficient domain representations from reference data, DOMINO enables scalable domain-specific data synthesis without manual prompt engineering, making it possible to generate a virtually unlimited number of domain-aligned samples. Theoretical and empirical results show that DOMINO can capture essential characteristics of the domain, generate diverse, domainaligned samples even without explicit domain definitions or any prior knowledge.

###### 6. Acknowledgements

This work was supported by the NGICS Next-Generation Industrial Control System Technology Research, and the Fundamental Research Funds for the Central Universities (Zhejiang University NGICS Platform). This research was also supported by Ant Group.

###### References

T. Chu, Y. Zhai, J. Yang, S. Tong, S. Xie, D. Schuurmans, Q. V. Le, S. Levine, and Y. Ma. SFT memorizes, RL generalizes: A comparative study of foundation model post-training. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=dY ur3yabMj.

- K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton,

- R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021.

DeepSeek-AI. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024. URL https://github.com/deepseek-ai/DeepSeek-LLM.

N. Ding, Y. Chen, B. Xu, Y. Qin, S. Hu, Z. Liu, M. Sun, and B. Zhou. Enhancing chat language models by scaling high-quality instructional conversations. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3029–3051, Singapore, Dec. 2023. Association for Computational Linguistics. doi: 10.18653/v1/ 2023.emnlp-main.183. URL https://aclanthology.org/2023.emnlp-main.183/.

Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. Technical report, Google, 2024. URL https://storage.googleapis.com/deepmind-media/gemini/ge mini_v1_5_report.pdf.

E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Lowrank adaptation of large language models. In International Conference on Learning Representations,

2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

- S. Huang, T. Cheng, J. K. Liu, J. Hao, L. Song, Y. Xu, J. Yang, J. Liu, C. Zhang, L. Chai, R. Yuan, Z. Zhang, J. Fu, Q. Liu, G. Zhang, Z. Wang, Y. Qi, Y. Xu, and W. Chu. Opencoder: The open cookbook for top-tier code large language models, 2025a. URL https://arxiv.org/abs/2411.04905.

Y. Huang, X. Lin, Z. Liu, Q. Cao, H. Xin, H. Wang, Z. Li, L. Song, and X. Liang. MUSTARD: Mastering uniform synthesis of theorem and proof data. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview.net/forum?id=8xliOUg9EW.

Y. Huang, X. Liu, Y. Gong, Z. Gou, Y. Shen, N. Duan, and W. Chen. Key-point-driven data synthesis with its enhancement on mathematical reasoning, 2024b. URL https://arxiv.org/abs/2403.02333.

Y. Huang, S. Wu, C. Gao, D. Chen, Q. Zhang, Y. Wan, T. Zhou, C. Xiao, J. Gao, L. Sun, and X. Zhang. Datagen: Unified synthetic dataset generation via large language models. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://openreview.net/forum?id= F5R0lG74Tu.

- B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu, L. Zhang, T. Liu, J. Zhang, B. Yu, K. Dang, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.

N. Jain, K. Han, A. Gu, W. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. CoRR, abs/2403.07974, 2024.

W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

- H. Li, Q. Dong, Z. Tang, C. Wang, X. Zhang, H. Huang, S. Huang, X. Huang, Z. Huang, D. Zhang, Y. Gu, X. Cheng, X. Wang, S.-Q. Chen, L. Dong, W. Lu, Z. Sui, B. Wang, W. Lam, and F. Wei. Synthetic data (almost) from scratch: Generalized instruction tuning for language models, 2024. URL https://arxiv.org/abs/2402.13064.

L. Long, R. Wang, R. Xiao, J. Zhao, X. Ding, G. Chen, and H. Wang. On LLMs-driven synthetic data generation, curation, and evaluation: A survey. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 11065–11082, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.finding s-acl.658. URL https://aclanthology.org/2024.findings-acl.658/.

- I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

Y. Nie, Y. Kong, X. Dong, J. M. Mulvey, H. V. Poor, Q. Wen, and S. Zohren. A survey of large language models for financial applications: Progress, prospects and challenges, 2024. URL https://arxiv.or g/abs/2406.11903.

OpenAI. Introducing ChatGPT, 2022. URL https://openai.com/index/chatgpt/. A. Plaat, A. Wong, S. Verberne, J. Broekens, N. van Stein, and T. Back. Reasoning with large language

models, a survey, 2024. URL https://arxiv.org/abs/2407.11511.

Qwen, :, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Q. Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github.io /blog/qwen2.5/.

L. van der Maaten and G. Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9(86):2579–2605, 2008. URL http://jmlr.org/papers/v9/vandermaaten08a.html.

Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13484–13508, Toronto, Canada, July 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.754. URL https: //aclanthology.org/2023.acl-long.754/.

- Y. Wang, H. Le, A. Gotmare, N. Bui, J. Li, and S. Hoi. CodeT5+: Open code large language models for code understanding and generation. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 1069–1088, Singapore, Dec. 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.emn lp-main.68. URL https://aclanthology.org/2023.emnlp-main.68/.

- C. White, S. Dooley, M. Roberts, A. Pal, B. Feuer, S. Jain, R. Shwartz-Ziv, N. Jain, K. Saifullah, S. Dey, Shubh-Agrawal, S. S. Sandha, S. V. Naidu, C. Hegde, Y. LeCun, T. Goldstein, W. Neiswanger, and M. Goldblum. Livebench: A challenging, contamination-free LLM benchmark. In The Thirteenth International Conference on Learning Representations, 2025.

- J. Wu, H. Yu, B. Liu, W. Yang, P. Di, J. Li, and Y. Zhang. Lamdas: Llm as an implicit classifier for domain-specific data selection, 2025. URL https://arxiv.org/abs/2509.06524.

C. Xu, Q. Sun, K. Zheng, X. Geng, P. Zhao, J. Feng, C. Tao, Q. Lin, and D. Jiang. WizardLM: Empowering large pre-trained language models to follow complex instructions. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=CfXh93 NDgH.

- Z. Xu, F. Jiang, L. Niu, Y. Deng, R. Poovendran, Y. Choi, and B. Y. Lin. Magpie: Alignment data synthesis from scratch by prompting aligned LLMs with nothing. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=Pnk7vMbz nK.

Y. Ye, Z. Huang, Y. Xiao, E. Chern, S. Xia, and P. Liu. Limo: Less is more for reasoning, 2025. URL https://arxiv.org/abs/2502.03387.

Q. Zhang, K. Ding, T. Lyv, X. Wang, Q. Yin, Y. Zhang, J. Yu, Y. Wang, X. Li, Z. Xiang, K. Feng, X. Zhuang, Z. Wang, M. Qin, M. Zhang, J. Zhang, J. Cui, T. Huang, P. Yan, R. Xu, H. Chen, X. Li, X. Fan, H. Xing, and H. Chen. Scientific large language models: A survey on biological & chemical domains, 2024a. URL https://arxiv.org/abs/2401.14656.

- S. Zhang, L. Dong, X. Li, S. Zhang, X. Sun, S. Wang, J. Li, R. Hu, T. Zhang, F. Wu, and G. Wang. Instruction tuning for large language models: A survey, 2024b. URL https://arxiv.org/abs/2308

.10792.

Y. Zheng, R. Zhang, J. Zhang, Y. Ye, Z. Luo, Z. Feng, and Y. Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.

- T. Y. Zhuo, V. M. Chien, J. Chim, H. Hu, W. Yu, R. Widyasari, I. N. B. Yusuf, H. Zhan, J. He, I. Paul, S. Brunner, C. GONG, J. Hoang, A. R. Zebaze, X. Hong, W.-D. Li, J. Kaddour, M. Xu, Z. Zhang, P. Yadav, N. Jain, A. Gu, Z. Cheng, J. Liu, Q. Liu, Z. Wang, D. Lo, B. Hui, N. Muennighoff, D. Fried, X. Du, H. de Vries, and L. V. Werra. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=YrycTjllL0.

###### A. Appendix

- A.1. Theoretical Guarantee

- A.1.1. The Proof of Proposition 1.

Proposition: Given 𝑿(𝑖) ∈ 𝑿(1:𝑛), L2 directly maximizes 𝐼(𝑺(𝑖), 𝑿(𝑖)|𝑫∗), ensuring that 𝑺(𝑖) captures unique information about 𝑿(𝑖).

Proof. By definition:

𝐼(𝑺(𝑖); 𝑿(𝑖)|𝑫∗) = 𝔼𝑝(𝑫∗,𝑺(𝑖),𝑿(𝑖))

[ log

𝑝(𝑺(𝑖)|𝑫∗, 𝑿(𝑖)) 𝑝(𝑺(𝑖)|𝑫∗)

], (6)

where 𝔼𝑝(𝑫∗,𝑺(𝑖),𝑿(𝑖)) represents the expectation over the distribution 𝑝(𝑫∗,𝑺(𝑖), 𝑿(𝑖)). Applying Bayes’ theorem, we can express the numerator as:

𝑝(𝑺(𝑖)|𝑫∗, 𝑿(𝑖)) =

𝑝(𝑿(𝑖)|𝑫∗,𝑺(𝑖))𝑝(𝑺(𝑖)|𝑫∗) 𝑝(𝑿(𝑖)|𝑫∗)

. (7) On the other hand, the denominator can be expanded as:

𝑝(𝑺(𝑖)|𝑫∗) =

∑𝑛

𝑗=1

𝑝(𝑿(𝑗),𝑺(𝑖)|𝑫∗) =

∑𝑛

𝑗=1

𝑝(𝑿(𝑗)|𝑫∗,𝑺(𝑖))𝑝(𝑺(𝑖)|𝑫∗). (8)

𝐼(𝑺(𝑖); 𝑿(𝑖)|𝑫∗) = 𝔼𝑝(𝑫∗,𝑺(𝑖),𝑿(𝑖))

[ log

𝑝(𝑿(𝑖)|𝑫∗,𝑺(𝑖)) 𝑝(𝑿(𝑖)|𝑫∗) ∑𝑛𝑗=1 𝑝(𝑿(𝑗)|𝑫∗,𝑺(𝑖))

],

≥ 𝔼𝑝(𝑫∗,𝑺(𝑖),𝑿(𝑖))

[ log

𝑝(𝑿(𝑖)|𝑫∗,𝑺(𝑖)) ∑𝑛𝑗=1 𝑝(𝑿(𝑗)|𝑫∗,𝑺(𝑖))

],

= 𝔼𝑝(𝑫∗,𝑺(𝑖),𝑿(𝑖))

[ log

𝑝(𝑿(𝑖)|𝑫∗,𝑺(𝑖)) 𝑝(𝑿(𝑖)|𝑫∗,𝑺(𝑖)) + ∑𝑗≠𝑖 𝑝(𝑿(𝑗)|𝑫∗,𝑺(𝑖))

],

≥ −L2 (9) The inequality holds because 𝑝(𝑿(𝑖)|𝑫∗) is a categorical distribution (i.e. 𝑝(𝑿(𝑖)|𝑫∗) ≤ 1) and 𝑝(𝑿(𝑖)|𝑫∗,𝑺(𝑖)) is always non-negative. So, minimizing L2 directly maximizes 𝐼(𝑺(𝑖), 𝑿(𝑖)|𝑫∗). □

- A.1.2. The Proof of Proposition 2.

Proposition: L2 directly minimizes 𝐼(𝑺(𝑖); 𝑫∗), promoting the disentanglement between 𝑺(𝑖) and 𝑫∗. Proof. Given the limited capacity (or size) of 𝑺(𝑖), the total mutual information 𝐼(𝑺(𝑖); 𝑿(𝑖)) is bounded. In practice, the combined length of the domain-level prompt 𝑫∗ and the sample-level prompt 𝑺(𝑖) is generally smaller than the length of the observed data 𝑿(𝑖), thus, this assumption is valid. By applying the chain rule of mutual information, we have:

𝐼(𝑺(𝑖); 𝑿(𝑖)) = 𝐼(𝑺(𝑖); 𝑫∗) + 𝐼(𝑺(𝑖); 𝑿(𝑖)|𝑫∗). (10)

Since L2 maximizes 𝐼(𝑺(𝑖); 𝑿(𝑖)|𝑫∗) as proven in Proposition 1 and the total 𝐼(𝑺(𝑖); 𝑿(𝑖)) is bounded, it necessitates the minimization of 𝐼(𝑺(𝑖); 𝑫∗) to maintain the equation’s integrity. □

- A.1.3. The Proof of Proposition 3.

Proposition: The optimization objective L = L1 + 𝜆L2 serves as a tractable proxy for learning a representation 𝑫∗ that approximates the information-theoretic properties of a minimal sufficient statistic. Specifically, minimizing L jointly encourages:

- 1. Sufficiency: The maximization of mutual information 𝐼(𝑫∗; 𝑿) , ensuring 𝑫∗ captures relevant

domain information.

- 2. Minimality: The minimization of mutual information 𝐼(𝑫∗;𝑺) , ensuring 𝑫∗ is disentangled

from sample-specific information.

Proof. The proof analyzes the two components of the loss function and assume that the empirical loss, calculated over the reference samples, is a reasonable estimator of the expected loss over the true data distribution.

Part 1: Minimizing L1 Encourages Sufficiency. The first loss term, L1, is the negative log-likelihood of the data 𝑿 given the domain represen-

tation 𝑫∗. In expectation, minimizing this empirical loss corresponds to minimizing the conditional entropy 𝐻(𝑿|𝑫∗):

min(L1) =⇒ min 𝐻(𝑿|𝑫∗) (11)

The mutual information 𝐼(𝑫∗; 𝑿) is defined as 𝐼(𝑫∗; 𝑿) = 𝐻(𝑿) − 𝐻(𝑿|𝑫∗). Since the data entropy 𝐻(𝑿) is a constant with respect to the model parameters, minimizing the conditional entropy 𝐻(𝑿|𝑫∗) is equivalent to maximizing the mutual information 𝐼(𝑫∗; 𝑿).

min 𝐻(𝑿|𝑫∗) ⇐⇒ max 𝐼(𝑫∗; 𝑿) (12) Thus, minimizing L1 drives the representation 𝑫∗ to capture as much information as possible

about the data 𝑿, fulfilling the sufficiency criterion. Part 2: Minimizing L2 Encourages Minimality. The second loss term, L2, is a contrastive loss. As established in Proposition 1, minimizing

L2 maximizes the conditional mutual information 𝐼(𝑺; 𝑿|𝑫∗). To understand how this promotes minimality, we use the chain rule of mutual information:

𝐼(𝑺; 𝑫∗) = 𝐼(𝑺; 𝑿) − 𝐼(𝑺; 𝑿|𝑫∗) (13)

The sample-specific representation 𝑺 has a fixed information capacity, bounded by its entropy 𝐻(𝑺), which is determined by the model’s architecture. This means 𝐼(𝑺; 𝑿) ≤ 𝐻(𝑺). This fixed capacity creates an “information budget” for what 𝑺 can learn about 𝑿. This budget is partitioned between information that is also in 𝑫∗ (the shared information, 𝐼(𝑺; 𝑫∗)) and information that is unique to 𝑺 given 𝑫∗ (the unique information, 𝐼(𝑺; 𝑿|𝑫∗)). Since the total budget 𝐼(𝑺; 𝑿) is bounded by the constant 𝐻(𝑺), and the objective of minimizing L2 is to maximize 𝐼(𝑺; 𝑿|𝑫∗), the optimization is strongly incentivized to allocate as much of the budget as possible to 𝐼(𝑺; 𝑿|𝑫∗). This necessarily encourages the minimization of 𝐼(𝑺; 𝑫∗). By the symmetry of mutual information, minimizing 𝐼(𝑺; 𝑫∗) is equivalent to minimizing 𝐼(𝑫∗;𝑺). This drives the domain representation 𝑫∗ to be uninformative about the sample-specific details 𝑺, thus satisfying the minimality criterion. □

###### A.1.4. The Proof of Proposition 4.

Proposition: Let 𝑝𝐷 and 𝑝𝐷∗ denote the distributions 𝑝(𝑿|𝑫) (from vanilla prompt tuning) and 𝑝(𝑿|𝑫∗) (from minimal sufficient, disentangled prompt tuning via L1 + 𝜆L2), respectively. For any 𝜖 ∈ (0, 1/𝑒), define the 𝜖-support of 𝑝, supp𝜖(𝑝) = {𝑥 : 𝑝(𝑥) > 𝜖}, with cardinality 𝑆𝜖𝑝 = |supp𝜖(𝑝)|. Define the uniformity gap for 𝑝 as 𝛿𝑝 = log𝑆𝜖𝑝 − 𝐻(𝑝), where 𝐻(𝑝) is the entropy of 𝑝. If

1

(14) then 𝑆𝜖𝑝

𝐻(𝑝𝐷∗) − 𝐻(𝑝𝐷) > 𝛿𝑝𝐷 + 𝜖log

𝜖

𝐷∗ > 𝑆𝜖𝑝𝐷; that is, the 𝜖-support of 𝑝(𝑿|𝑫∗) is strictly larger than that of 𝑝(𝑿|𝑫).

Proof. By definition, log𝑆𝜖𝑝 = 𝐻(𝑝) +𝛿𝑝 and, by a standard entropy-support bound, for any 𝑝, 𝐻(𝑝) ≤ log𝑆𝜖𝑝 + 𝜖log 1𝜖 =⇒ −𝛿𝑝 ≤ 𝜖log 1𝜖. Thus,

𝐷∗ − log𝑆𝜖𝑝𝐷 = [𝐻(𝑝𝐷∗) − 𝐻(𝑝𝐷)] + [𝛿𝑝𝐷∗ − 𝛿𝑝𝐷] (15) > [𝛿𝑝𝐷 + 𝜖log

log𝑆𝜖𝑝

1 𝜖] + [𝛿𝑝𝐷∗ − 𝛿𝑝𝐷] (16)

1

1 𝜖 + 𝜖log

1

= 0, (17) since,

= 𝛿𝑝𝐷∗ + 𝜖log

> −𝜖log

𝜖

𝜖

1

1

- (18)
- (19)

−𝛿𝑝𝐷∗ ≤ 𝜖log

=⇒ 𝛿𝑝𝐷∗ ≥ −𝜖log

𝜖

𝜖

Therefore,

𝐷∗ > 𝑆𝜖𝑝𝐷 (20) This closes the proof. □

log𝑆𝜖𝑝

𝐷∗ > log𝑆𝜖𝑝𝐷, 𝑖.𝑒., 𝑆𝜖𝑝

- A.2. Experimental Details

- A.2.1. Target Domain Data Statistics.

We select two distinct domains from the recently proposed dynamic benchmark LiveCodeBench (Jain et al., 2024): Live Code Generation and Live Code Execution. To ensure a fair and realistic evaluation that mimics a real-world “live” setting, we apply a consistent temporal cutoff: specifically, we use all domain samples released before the final update’s timestamp as reference data (input–output pairs), while every sample that appears in that final update constitutes the test set.

For the Live Code Generation domain, the data collected before September 2024 as reference data and data collected after that as the test set.

For the Live Code Execution domain, the data collected before October 2023 as reference data (after deduplication) and data collected after that as the test set.

- Table 6 | Statistics of the two target domains: Live Code Generation and Live Code Execution.

Reference Samples Test Set Temporal Cutoff Point Live Code Generation 713 167 September 2024 Live Code Execution 66 108 October 2023

Target Domain Instances. We present detailed examples of the reference data for the two target domains in Figure 9 and Figure 10.

Evaluation: For evaluation, we use the official evaluation scripts from LiveCodeBench and report the corresponding Pass@k metrics. The inference prompts used for the two target domains are shown in Figure 19 and Figure 20.

- A.2.2. Case Study: Interpretability of 𝐷∗.

To better interpret the specific domain patterns captured by 𝑫∗, we present a case study using samples that exemplify the clear distinction between the standard synthetic set (𝑫) and our proposed diverse

set (𝑫∗). Critically, the synthetic samples from 𝑫∗ are specifically selected from regions of the t-SNE embedding space that are distant from the dense cluster representing 𝑫, visually confirming their diversity. The specific samples, provided in Figure 7, reveal two key findings:

- • Overfitting in 𝑫: The sample from 𝑫 demonstrates sample-level mimicry. It not only adopts the high-level structure (problem statement → examples → constraints) but also copies superficial details like variable names (n, k) and problem naming conventions (k-xxx), indicating it has overfitted to specific instances.
- • Domain Abstraction in 𝑫∗: In contrast, samples from 𝑫∗ capture the abstract domain structure of a competitive programming problem while introducing genuine topical diversity. They explore entirely new problem types—spanning graph theory, dynamic programming, and string algorithms—that are distinct from the reference sample.

##### Reference Sample Synthetic Samples From D Synthetic Samples From D*

You are given a string n of length k. A string is called k-subsequence of n if n can be written as t1t2...tk, where every ti is a non-empty substring of n. Return the number of k-subsequences of n, given that k is a divisor of n. Since the answer may be very large, return the answer modulo 109 + 7. A substring is a contiguous sequence of characters within a string.

You are given an integer array arr and an integer difference. Return the length of the longest subsequence in arr which is an arithmetic sequence such that the difference between adjacent elements in the subsequence is difference. A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

You are given two integers, n and k. An array of distinct positive integers is called a kavoiding array if there does not exist any pair of distinct elements that sum to k. Return the minimum possible sum of a k-avoiding array of length n.

Example 1: Input: n = "ababa", k = 3 Output: 7 Explanation: The 7 3-subsequences of n are: "aba", "aba", "baba", "aba", "baba", "aba", "aba". Note that some subsequences are repeated, but we only need to count distinct subsequences.

Example 1: Input: n = 5, k = 4 Output: 18 Explanation: Consider the kavoiding array [1,2,4,5,6], which has a sum of 18. It can be proven that there is no kavoiding array with a sum less than 18.

Example 1: Input: arr = [1,2,3,4], difference = 1 Output: 4 Explanation: The longest arithmetic subsequence is [1,2,3,4].

Constraints: 1 <= arr.length <= 10^5

-10^4 <= arr[i], difference <= 10^4

Constraints: 1 <= n.length <= 10^4 n consists of lowercase English letters. 1 <= k <= 10

Constraints: 1 <= n, k <= 50

- Figure 7 | Synthetic Samples from 𝑫 and 𝑫∗.

###### A.2.3. Generalization of DOMINO to More Task Domains.

To demonstrate DOMINO’s effectiveness across a more diverse set of tasks, we extended our experiments to the critical and well-defined Instruction Following domain. We chose this domain for two key reasons:

- • Diverse Subtasks: It comprises a variety of core language tasks (paraphrasing, summarization, simplification, story generation), allowing us to test DOMINO’s versatility.
- • Challenging, Unseen Data: We use LiveBench (White et al., 2025), a dynamic benchmark whose data is continuously updated. This ensures our test set is free from contamination and that even SOTA models do not achieve perfect scores, providing a meaningful test of generalization. Proving that DOMINO’s synthetic data can improve performance on such a challenging, live benchmark is a strong testament to its value.

Following the protocol from main experiment of Live Code Generation and Live Code Execution, we use the temporal cutoff (June 30, 2024) to create reference and test sets. The reference set and test set each contain 200 samples. We then use DOMINO with a Qwen2.5-7B-Instruct (Team, 2024) backbone to synthesize 40K samples for fine-tuning. The results, evaluated using the official LiveBench suite, are presented in Table 3. As shown, DOMINO consistently outperforms all baselines across the four subtasks, boosting the average score by 3.48 points over the strong instruction-tuned backbone and by 1.61 points over the standard prompt-tuning approach (DOMINO-Direct Domain). These results reinforce the original findings and demonstrate that DOMINO’s ability to learn and generalize from implicit domain characteristics is not limited to coding but extends to a diverse range of NLP tasks.

###### A.2.4. Comparison of Synthetic Samples vs. Reference Samples (Equal Size).

To further compare the performance gap when using the same number of DOMINO’s synthetic samples versus original reference samples, we conducted an SFT experiment comparing the two. As shown in Table 7, the DOMINO’s synthetic data achieves performance very close to the original data. This small gap is expected, as the original reference samples represent the ground-truth “gold” data for the domain. This result validates the high per-sample quality of DOMINO’s generated sample.

- Table 7 | Performance Comparison of Reference vs. DOMINO’s Synthetic Samples (Equal Size).

Task Metric Reference Samples DOMINO’s Synthetic Samples

Pass@1 13.47 12.69 Pass@5 21.59 20.73

Live Code Generation

Pass@10 23.95 23.35 Live Code Execution Pass@1 36.11 34.26

###### A.2.5. Impact of Reference Data Percentage on Live Code Execution.

To evaluate the impact of reference data quantity on the Live Code Execution domain, we mirror the methodology of Figure 6 (Live Code Generation) using the QWEN2.5-CODER-7B Instruct backbone.

We vary the proportion of the reference set from 100% down to 20% and the results are shown in Figure 8. Figure 8 confirms the conclusion: reducing the amount of in-domain reference data leads to a corresponding drop in performance. The results also reveal a performance plateau, with almost no improvement from 80% to 100% of the reference data. This reinforces the central claim by suggesting that while a sufficient quantity of reference data is crucial for high performance, gains diminish once a point of saturation is reached. Determining this optimal quantity across different domains remains a compelling question for future work.

###### A.2.6. Impact of more reference samples for MAGPIE.

In our comparative method, MAGPIE-Few shot, we originally randomly selected three reference samples to include in the prompt for domain representation summarization using the LLM. This setup considered that each reference sample typically contains several hundred tokens. To further investigate whether providing more reference samples to MAGPIE method would lead to performance gains, we conducte the corresponding experiment, with results presented in Table 8.

As shown in Table 8, the MAGPIE performance remains comparable when using 3 or 5 few-shot samples. However, increasing the count to 10 severely challenges the LLM’s ability to extract domain-

Pass@1

55.0

PassRate(%)

52.5

50.0

47.5

45.0

100% 80% 60% 40% 20%

Reference Samples Percentage

- Figure 8 | Result of the impact of reference sample percentage in Live Code Execution domain using QWEN2.5-CODER-7B Instruction as backbone.

specific information from the extended context. Our empirical evidence indicates that this larger few-shot count actually compromises performance. These results confirm that additional reference samples do not benefit MAGPIE; rather, they have a detrimental effect.

- Table 8 | Results of more reference samples for MAGPIE-Few shot.

###### Live Code Generation Pass@1 Pass@5 Pass@10

MAGPIE Few-Shot = 3 14.36 20.98 24.23 MAGPIE Few-Shot = 5 13.89 21.21 23.35 MAGPIE Few-Shot = 10 11.81 16.82 18.39

###### A.2.7. Baseline Prompts.

We present the system prompts used by MAGPIE-Human and MAGPIE-Few Shot for the two main target domains in the Figure 13, Figure 14, Figure 15 and Figure 16.

###### A.2.8. Post-processing.

For the input sequences synthesized by DOMINO and other methods, we apply a unified filtering mechanism to remove low-quality samples. The filtering prompt is shown in Figure 17 and Figure 18. We retain only synthetic input sequences of “Excellent” quality. It is important to note that the same LLM backbone used for synthesizing the domain-specific sample inputs is also used for both filtering and generating the corresponding output sequences.

###### A.2.9. Domain Adaptation Configurations.

We fine-tuned the OPENCODER-8B Base, QWEN2.5-CODER-7B Base, and QWEN2.5-CODER-14B Base models on the synthesized domain-specific samples using LLAMA-FACTORY (Zheng et al., 2024), on a server equipped with 8×NVIDIA A100 80GB GPUs. During fine-tuning, we applied LoRA (Hu et al., 2022) with lora_rank=8 and lora_target=all. All experiments used the AdamW (Loshchilov and Hutter, 2019) optimizer with an initial learning rate of 5e-5. Detailed training parameters are shown in Table 9.

- Table 9 | The hyper-parameters for supervised fine-tuning.

Hyper-parameter Value

Learning Rate 5 × 10−5 Number of Epochs 2 Number of Devices 8 Per-device Batch Size 3 Gradient Accumulation Steps 1 Effective Batch Size 24 Optimizer Adamw with 𝛽𝑠 = (0.9, 0.999) and 𝜖 = 10−8 Learning Rate Scheduler linear Warmup Steps 0 Max Sequence Length 4096

###### A.2.10. A Case Study of Domain-Specific Synthetic Samples.

From Figure 11 and Figure 12, it can be seen that the domain-specific samples synthesized by DOMINO are very similar to the domain reference samples. Although it is difficult to precisely describe this domain similarity using natural language, they exhibit consistency in certain features, such as structural characteristics.

###### A.3. Limitations

Our proposed method, DOMINO, encodes domain-specific patterns as minimal sufficient representations based on reference samples. To be effective, this process relies on the assumption that the reference samples are representative of the target domain. If the reference data is not representative of the domain, the resulting domain representation 𝑫∗ will fail to capture the true characteristics of the domain, ultimately limiting the quality and generalizability of the synthesized data.

###### A.4. Future Work

In the methodology, while the sample-specific representations 𝑺(𝑖) are discarded during synthesis, they could potentially serve tasks such as retrieving the most similar reference sample for a given query, which we leave for future work. We also plan to investigate DOMINO’s robustness on mixeddomain reference sets. We hypothesize that the contrastive disentanglement mechanism may naturally encourage the domain representation 𝑫∗ to focus on the most prevalent shared patterns, implicitly filtering out outlier noise, though this requires further study. Furthermore, the learned domain representation 𝑫∗ captures domain-level characteristics and can serve as an implicit classifier for data selection, analogous to LAMDAS (Wu et al., 2025), which we leave for future investigation.

###### Reference Samples in Live Code Generation Domain

You are given two integers, n and k. An array of distinct positive integers is called a k-avoiding array if there does not exist any pair of distinct elements that sum to k. Return the minimum possible sum of a k-avoiding array of length n.

You are given a 0-indexed integer array nums. You have to find the maximum sum of a pair of numbers from nums such that the maximum digit in both numbers are equal. Return the maximum sum or -1 if no such pair exists.

Given a positive integer num represented as a string, return the integer num without trailing zeros as a string.

- Example 1:

Input: num = "51230100" Output: "512301" Explanation: Integer "51230100" has 2 trailing zeros, we remove them and return integer "512301".

- Example 2:

- Example 1: Input: nums = [51,71,17,24,42] Output: 88 Explanation: For i = 1 and j = 2, nums[i] and nums[j] have equal maximum digits with a pair sum of 71 + 17

= 88. For i = 3 and j = 4, nums[i] and nums[j] have equal maximum digits with a pair sum of 24 + 42

= 66. It can be shown that there are no other pairs with equal maximum digits, so the answer is 88.

- Example 2:

- Example 1: Input: n = 5, k = 4 Output: 18 Explanation: Consider the kavoiding array [1,2,4,5,6], which has a sum of 18. It can be proven that there is no k-avoiding array with a sum less than 18.
- Example 2: Input: n = 2, k = 6 Output: 3 Explanation: We can construct the array [1,2], which has a sum of 3. It can be proven that there is no kavoiding array with a sum less than 3.

Input: num = "123" Output: "123" Explanation: Integer "123" has no trailing zeros, we return integer "123".

Input: nums = [1,2,3,4] Output: -1 Explanation: No pair exists in nums with equal maximum digits.

Constraints: 1 <= num.length <= 1000 num consists of only digits. num doesn't have any leading zeros.

Constraints: 2 <= nums.length <= 100 1 <= nums[i] <= 10^4

Constraints: 1 <= n, k <= 50

- Figure 9 | Detailed examples of the reference data in Live Code Generation domain.

def smallestString(s: str) -> str: if s == "a" * len(s):

return "a"*(len(s) - 1)+"z" r = "" p = 0 for i in s:

if p == 1:

if i == "a": p = 2 r += i

else:

r += chr(ord(i) - 1) elif p == 0:

if i == "a": r += i

else: p = 1 r += chr(ord(i) - 1)

else: r += i

return r assert smallestString(s = 'acbbc’)

== ?

Reference Samples in Live Code Execution Domain

def minimumBeautifulSubstrings(s: str) -> int:

good = [] num = 1 n = len(s) while True:

b = bin(num)[2:] if len(b) > n:

break good.append(b) num *= 5

dp = [int(1e9)] * (n + 1) dp[0] = 0 gs = set(good) for i in range(n):

for j in range(i + 1): if s[j:i + 1] in gs:

dp[i + 1] = min(dp[i + 1], dp[j] + 1)

return -1 if dp[n] == int(1e9) else dp[n] assert minimumBeautifulSubstrings(s = '0’)

== ?

- Figure 10 | Detailed examples of the reference data in Live Code Execution domain.

You are given an array A of length N. For each i (1 \leq i \leq N), compute the minimum value of the expression A[i] - A[j] + A[k] for all possible combinations of i, j, and k such that i \leq j \leq k. Input The input is given from Standard Input in the following format: N A_1 A_2 ... A_N Output Print the minimum value of the expression A[i] - A[j] + A[k] for all possible combinations of i, j, and k such that i \leq j \leq k.

You are given an integer array arr and an integer difference. Return the length of the longest subsequence in arr which is an arithmetic sequence such that the difference between adjacent elements in the subsequence is difference. A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

Constraints

- Example 1:

- Input: arr = [1,2,3,4], difference = 1 Output: 4 Explanation: The longest arithmetic subsequence

- is [1,2,3,4].

Example 2: Input: arr = [1,3,5,7], difference = 2 Output: 2 Explanation: The longest arithmetic subsequence

- is [1,3,5,7].

- * 3 \leq N \leq 10^5
- * 1 \leq A_i \leq 10^9
- * All values in the input are integers.

Example Input 5 2 1 3 1 4 Output 0 For example, choosing i = 2, j = 2, and k = 3 yields the minimum value of the expression.

Input 4

Constraints: 1 <= arr.length <= 10^5

4 1 5 2 Output -4

-10^4 <= arr[i], difference <= 10^4

For example, choosing i = 1, j = 2, and k = 3 yields the minimum value of the expression

- Figure 11 | Samples synthesized by DOMINO in the Live Code Generation domain.

def maximumPopulation(years: List[int]) -> int:

n = len(years) cnt = [0] * 2003 for y in years:

- cnt[y+1] += 1
- cnt[y+2] -= 1

ans = -1 m = 0 for i in range(2003):

cnt[i] += cnt[i-1] if cnt[i] > m:

ans = i m = cnt[i]

return ans

assert maximumPopulation(years = [1950, 1960, 1960, 1970, 1970, 1970])

== ?

def minimumTimeToType(word: str) -> int: ans = len(word) prev = "a" for w in word:

diff = abs(ord(w) - ord(prev)) ans += min(diff, 26 - diff) prev = w

return ans assert minimumTimeToType(word = ‘a’)== ?

- Figure 12 | Samples synthesized by DOMINO in the Live Code Execution domain.

Prompt for MAGPIE-Few Shot in the Live Code Execution Domain

### Characteristics of the current coding task:

- 1. The python function involves array manipulation and logical operations. The functions typically take arrays or lists as input and perform operations to derive or validate certain properties of these arrays.
- 2. The function input also showcase the use of assertions to validate the correctness of the functions.

- Figure 13 | Prompt for MAGPIE-Few Shot in the Live Code Execution domain.

Prompt for MAGPIE-Few Shot in the Live Code Generation Domain

### Characteristics of the current coding task:

- 1. **Problem Description**: a few sentences about the problem.
- 2. **Input Constraints**: Each test case has specific constraints on the size of the input, such as the number of elements in an array, the length of a string, or the range of values in an array.
- 3. **Output Format**: The output should match the number of test cases, with each result corresponding to the input test case.
- 4. **Constraints on Operations**: There may be constraints on the number of operations that can be performed, such as performing an operation at most once or within a specific range.

- Figure 14 | Prompt for MAGPIE-Few Shot in the Live Code Generation domain.

Prompt for MAGPIE-Human in the Live Code Execution Domain

### Characteristics of the current coding task: It begins with a Python function that performs the required operations, followed by an example input provided after the function. The input example typically starts with an assert statement.

- Figure 15 | Prompt for MAGPIE-Human in the Live Code Execution domain.

Prompt for MAGPIE-Human in the Live Code Generation Domain

### Characteristics of the current coding task: It begins with a task description, followed by a relevant example (including both input and output) with necessary explanatory information, and ends with a set of constraints.

- Figure 16 | Prompt for MAGPIE-Human in the Live Code Generation domain.

Unified Filtering Prompt for Live Code Generation

# Instruction You need to assess the quality of the given coding question based on its clarity, specificity, completeness, and challenge.

The rating scale is as follows:

- - very poor: The question is unclear, vague, or disorganized and the examples are wrong. It lacks essential information and context, contains many unreadable characters for humans or has large scale repetitive parts.
- - poor: The question is somewhat unclear or lacks important details, requiring significant clarification, or contains irrelevant content and repetition.
- - average: The question has moderate clarity and specificity. The question is basically complete but may require some additional information for full understanding, and the diﬀiculty is relatively easy.
- - good: The question is clear and specific, and it is generally well-articulated. The question is complete, providing suﬀicient context to understand its intent, with little to no irrelevant content or noise, and has a moderate level of diﬀiculty.
- - excellent: The question is very clear, specific, and well-articulated. It contains all the necessary information and context for providing a comprehensive response and also has a certain level of challenge, requiring multi-step reasoning, complex algorithms or data structures.

## Coding Question ``` {input} ```

## Output Format Given the coding question, you first need to give an assessment, highlighting the strengths and/or weaknesses of the coding question. Then, you need to output a rating from very poor to excellent by filling in the placeholders in [...]: ``` {{

''explanation'': ''[...]'', ''input_quality'': ''[very poor/poor/average/good/excellent]''

}} ```

- Figure 17 | Unified filtering prompt for Live Code Generation domain.

Unified Filtering Prompt for Live Code Execution

# Instruction You need to assess the quality of the given python function based on its clarity, specificity, completeness, and challenge.

The rating scale is as follows:

- - very poor: The function is unclear, vague, or disorganized. It contains many unreadable characters for humans or has large scale repetitive parts.
- - poor: The function is unclear or lacks important details, contains irrelevant content and repetition.
- - average: The function has moderate clarity and specificity. The diﬀiculty is relatively easy.
- - good: The function is clear and specific, and it is generally well-articulated.
- - excellent: The function is very clear, specific and well-articulated, and poses a certain level of diﬀiculty.

## Function ``` {input} ```

## Output Format Given the function, you first need to give an assessment, highlighting the strengths and/or weaknesses of the function. Then, you need to output a rating from very poor to excellent by filling in the placeholders in [...]: ``` {{

''explanation'': ''[...]'', ''input_quality'': ''[very poor/poor/average/good/excellent]''

}} ```

- Figure 18 | Unified filtering prompt for Live Code Execution domain.

Inference Prompt for Live Code Generation

You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

### Question:\n{question.question_content} { if question.starter_code } ### Format: {PromptConstants.FORMATTING_MESSAGE}

```python {question.starter_code} ``` { else } ### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_MESSAGE}

```python # YOUR CODE HERE ``` { endif }

### Answer: (use the provided format with backticks) ```

- Figure 19 | Inference prompt for Live Code Generation domain.

Inference Prompt for Live Code Execution

You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON] def performOperation(s):

s = s + s return ''b'' + s + ''a''

assert performOperation(s = ''hi'') == ?? [/PYTHON] [THOUGHT] Let's execute the code step by step:

- 1. The function performOperation is defined, which takes a single argument s.
- 2. The function is called with the argument ''hi'', so within the function, s is initially ''hi''.
- 3. Inside the function, s is concatenated with itself, so s becomes ''hihi''.
- 4. The function then returns a new string that starts with ''b'', followed by the value of s (which is now ''hihi''), and ends with ''a''.
- 5. The return value of the function is therefore ''bhihia''. [/THOUGHT] [ANSWER] assert performOperation(s = ''hi'') == ''bhihia'' [/ANSWER]

[PYTHON] {code} assert {input} == ?? [/PYTHON] [THOUGHT]

- Figure 20 | Inference prompt for Live Code Execution domain.

