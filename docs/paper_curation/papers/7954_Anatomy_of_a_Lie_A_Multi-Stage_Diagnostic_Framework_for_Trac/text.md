# arXiv:2603.15557v1[cs.CV]16Mar2026

## Anatomy of a Lie: A Multi-Stage Diagnostic Framework for Tracing Hallucinations in Vision-Language Models

Lexiang Xiong1,∗ Qi Li1,∗ Jingwen Ye2, Xinchao Wang1† 1National University of Singapore 2Monash University {lexiang, liqi}@u.nus.edu, xinchao@nus.edu.sg

https://github.com/Lexiang-Xiong/CAD

### Abstract

Vision-Language Models (VLMs) frequently ‘hallucinate’—generate plausible yet factually incorrect statements—posing a critical barrier to their trustworthy deployment. In this work, we propose a new paradigm for diagnosing hallucinations, recasting them from static output errors into dynamic pathologies of a model’s computational cognition. Our framework is grounded in a normative principle of computational rationality, allowing us to model a VLM’s generation as a dynamic cognitive trajectory. We design a suite of information-theoretic probes that project this trajectory onto an interpretable, low-dimensional Cognitive State Space. Our central discovery is a governing principle we term the geometric-information duality: a cognitive trajectory’s geometric abnormality within this space is fundamentally equivalent to its high information-theoretic surprisal. Hallucination detection is thus elegantly re-framed as a geometric anomaly detection problem. Evaluated across diverse settings—from rigorous binary QA (POPE) and comprehensive reasoning (MME) to unconstrained open-ended captioning (MS-COCO)—our framework achieves state-of-the-art performance. Crucially, it operates with high efficiency under weak supervision and remains highly robust even when calibration data is heavily contaminated. This approach enables a causal attribution of failures, mapping observable errors to distinct pathological states: perceptual instability (measured by Perceptual Entropy, HEvi), logical-causal failure (measured by Inferential Conflict, SConf), and decisional ambiguity (measured by Decision Entropy, HAns). Ultimately, this opens a path toward building AI systems whose reasoning is transparent, auditable, and diagnosable by design.

### 1 Introduction

Consider a striking paradox in Vision-Language Models (VLMs) Yu et al. (2025); Li et al. (2025b); Team (2024); Li et al. (2025a; 2023): when asked “Is there a motorcycle in the image?”, a model might confidently hallucinate evidence—“In the image, there is a motorcycle parked”—yet inexplicably conclude, “Therefore, the final answer is No.” We term this cascade of errors computational cognitive dissonance, as illustrated in Figure 1. What makes this specific case so dramatic is that a double failure (perceiving a non-existent object, then logically contradicting that very perception) leads to a coincidentally correct final answer.

This phenomenon exposes a critical insight that severely limits VLM deployment in highstakes domains Ji et al. (2023); Li & Wang (2026); Bai et al. (2024); Li et al. (2024); Wang et al. (2025b): hallucinations are rarely monolithic errors that can be diagnosed by a single metric like “accuracy” or “self-consistency.” Instead, they are often complex, multi-stage pathologies where distinct failures—such as perceptual drift and logical bypass—compound and

∗Equal contribution. †Corresponding author.

(a) The Scene: Computational Cognitive Dissonance

(b) Macroscopic Diagnosis: Geometric Anomaly

[Figure 1]

100

Geometric Anomaly

InferentialConflict()SConf

10 1

0

10 1

Nominal Manifold (Density)

Other Hallucinations

This Anomalous Case

0 10 2 10 1

User : Is there a motorcycle in the image?

Perceptual Instability (HEvi)

VLM Evidence : In the image, there is a motorcycle parked

next to a bus.The motorcycle is not in motion, and it is stationary... VLM Conclusion : Therefore, the final answer is No.

###### (c) Stage 1 Diagnosis: Perceptual Failure

(d) Stage 2 Diagnosis: Logical-Causal Failure

|Nominal|Threshold|H|allucinated: "parked"| | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |Density of Nominal In|xtreme<br><br>SConf = 1.28<br><br>ferences|
|---|---|---|
| | | |
| |E| |

ProbabilityDensity(SymlogScale)

1.0

Token-levelUncertainty

- 100
- 101

0.8

0.6

0.4

Conflict

0.2

0

0.0

10 1 0 10 1 100

0 10 20 30 40 50

Token Generation Step (Evidence Chain)

Inferential Conflict Score (SConf)

Figure 1: An example of computational cognitive dissonance in Idefics2, where a cascade of failures leads to a coincidentally correct answer. (1) Perceptual Failure: The model hallucinates a ‘motorcycle’ in the evidence chain, an object not present in the image (a cyclist is visible). Our framework captures this as high Perceptual Instability (see panel (c)). (2) Logical Failure: The model then contradicts its own faulty evidence, concluding the final answer is ‘No’. This breakdown of self-consistency is diagnosed as extremely high Inferential Conflict (see panel (d)). This case study demonstrates the limitation of accuracy-only evaluations and highlights our framework’s ability to perform a stage-bystage differential diagnosis of a VLM’s cognitive process, identifying complex, multi-stage failure trajectories.

interact within a single cognitive trajectory. Current approaches to hallucination detection generally treat the generation process as an indivisible, monolithic event. They either evaluate the semantic consistency of final outputs via multiple sampling Manakul et al. (2023); Farquhar et al. (2024) or probe for a binary ‘truthfulness’ representation within internal states Azaria & Mitchell (2023); Chen et al. (2024b). While foundational, these reductionist views conflate fundamentally different failure modes. They struggle to distinguish whether a hallucination stems from an initial failure to ground concepts in the image (perceptual drift) or from an illogical jump that bypasses extracted facts (inferential bypass). Our central thesis is that hallucination is a process-level failure that must be diagnosed within a structured model of cognition.

To address this, we introduce a normative principle of computational rationality Gershman et al. (2015); Oulasvirta et al. (2022) for VLMs, formalized as a Markovian information

flow: Image (I) −−−−−→Perception Textual Evidence (Tevi) −−−−−→Inference Final Answer (A). This principle asserts that for a rational agent, the final answer A is conditionally independent of the image

I given the evidence Tevi, implying the conditional mutual information I(A; I|Tevi) must be zero. Critics might argue that requiring an explicit evidence chain limits the applicability of such a framework. However, we employ Chain-of-Thought (CoT) not as a strict operational constraint, but as a crucial diagnostic probe in explainable AI (XAI)—akin to a medical contrast agent. By forcing the model to externalize its latent reasoning, we make the implicit cognitive trajectory observable and mathematically diagnosable.

To diagnose this cognitive process, we design a suite of probes. While SConf directly measures violations of our core principle, Perceptual Entropy (HEvi) and Decision Entropy

(HAns) quantify the stability of the process’s initial and final stages, providing a complete diagnostic picture. These probes act as natural coordinates to project the high-dimensional trajectory onto an interpretable 3D Cognitive State Space:

- • Perceptual Instability (HEvi): Measured via Perceptual Entropy, this probes the uncertainty at the perception stage (I → Tevi).
- • Logical-Causal Failure (SConf): Measured via Inferential Conflict, this directly quantifies the information leakage that violates our core principle.
- • Decisional Ambiguity (HAns): Measured via Decision Entropy, this probes the final uncertainty at the trajectory’s terminal stage.

Collectively, these probes summarize each cognitive trajectory as a Cognitive State Vector within this space.

This perspective reveals a powerful geometric-information duality: a trajectory’s geometric abnormality within this space is fundamentally an expression of its high informationtheoretic surprisal. Our experimental results (see Figure 3) provide strong empirical evidence for this duality. Normative cognitive trajectories consistently evolve towards stable, low-energy basins of attraction, forming a dense submanifold. Hallucinations, conversely, are high-energy deviations that are perturbed off this manifold, appearing as geometric anomalies. This duality serves as the theoretical bridge that translates the semantic problem of hallucination into a rigorous geometric anomaly detection task Stolz et al. (2020).

This novel reframing achieves state-of-the-art detection performance while offering significant practical advantages. Unlike multi-sample methods, our approach requires only a single generation pass plus a highly efficient non-autoregressive replay (detailed in Section 3). Furthermore, it operates under weak supervision (requiring only ground-truth answers, not fine-grained hallucination labels) and remains highly resilient even when calibration data is contaminated with up to 30% noise. To demonstrate its versatility, we rigorously validate our framework across a spectrum of tasks: from the controlled adversarial subset of POPE Li et al. (2023), to the comprehensive reasoning categories of MME Fu et al., and finally to unconstrained open-ended captioning on MS-COCO Lin et al. (2014).

In summary, our primary contribution is not merely a new state-of-the-art hallucination detector, but a principled diagnostic framework that provides a new lens through which to understand VLM failures. Our contributions are:

- • We propose a diagnostic framework that reframes hallucination from a static flaw to a dynamic analysis of a VLM’s cognitive trajectory, grounded in a normative principle of computational rationality.
- • We design a suite of information-theoretic probes that act as natural coordinates to project the generative process onto an interpretable Cognitive State Space, enabling a stage-by-stage differential diagnosis.
- • Grounded in a powerful geometric-information duality, we introduce a novel detection method based on geometric anomaly detection, which operates under weak supervision and achieves state-of-the-art performance.
- • We deliver a novel mechanistic categorization of VLM failure modes by analyzing the topology of their cognitive manifolds, diagnosing complex errors like ‘computational cognitive dissonance’.

### 2 Related Work

Our research is positioned at the confluence of VLM hallucination evaluation, mitigation, and internal state analysis Bai et al. (2024); Ji et al. (2023). With these foundations, we introduce a novel diagnostic framework that moves beyond static error detection to model the VLM’s dynamic internal cognitive process.

VLM Hallucination Benchmarking. A significant body of work has focused on quantifying VLM hallucinations from final outputs. The pioneering CHAIR metric Rohrbach et al. (2018) measured hallucinated objects in captions. To address its instability, works like POPE Li et al. (2023) and ROPE Chen et al. (2024c) established a stable polling-based evaluation

paradigm. To capture more complex failure modes, recent benchmarks have expanded to evaluate real-world instruction following (VisIT-Bench Bitton et al. (2023), MMHal-Bench Sun et al. (2023), HallusionBench Guan et al. (2024)), fine-grained visual-text alignment (LM2-Bench Peyrard et al. (2024), WYSWIR Wang et al. (2023)), and advanced commonsense reasoning (Visual Riddles Yarom et al. (2024)). Adversarial benchmarks such as MAD-Bench Qian et al. (2024) further probe robustness against deceptive prompts. While these static evaluation suites are invaluable for assessing what errors a model makes across diverse scenarios, they predominantly treat the VLM as a black box. Our work provides a crucial complement: diagnosing the dynamic generative process itself to explain how and why these errors occur.

Inference-Time Hallucination Detection and Mitigation. Recent efforts have focused on inference-time strategies. One prominent line of work is contrastive decoding, which penalizes outputs driven primarily by language priors rather than visual evidence Wang et al. (2025a); Vu et al. (2025). State-of-the-art methods like Hallucination-Induced Optimization (HIO) Chen et al. (2024a) refine this by training a dedicated ‘evil’ model to provide a targeted contrastive signal. Parallel efforts in automated evaluation (auto-eval) employ strong LLMs (e.g., Clair Tsun et al. (2024)) or contrastive grounding techniques (e.g., Contrastive Region Guidance Wang et al. (2024)) to assess output quality without manual labels. Another direction analyzes internal signals, such as VADE Prabhakaran et al. (2025), which models attention map sequences. While these methods excel at scoring or correcting the final output, our framework is distinctly focused on diagnosing the mechanistic failure. Our Inferential Conflict metric (Section 3) directly isolates the illicit vision-language information flow, offering a causal interpretation that auto-evals typically lack.

Internal State Analysis for Hallucination. A nascent line of inquiry explores VLM internal states, inspired by seminal research in LLMs suggesting truthfulness is encoded in hidden activations Azaria & Mitchell (2023); Chen et al. (2024b); Orgad et al. (2024); Ferrando et al. (2024). While methods like VADE Prabhakaran et al. (2025) analyze internal patterns, they focus on attention mechanisms. Other approaches, often adapted from the text-only domain, may treat the VLM’s internal state as a monolithic representation Du et al. (2024); Park et al. (2025). This simplification is ill-equipped to distinguish between a failure in initial perception versus a breakdown in subsequent reasoning. Distinct from all prior work, our research introduces a multi-faceted diagnostic framework that models a VLM’s reasoning not as a static state, but as a measurable cognitive trajectory through distinct, macroscopic stages. This process-oriented view enables a mechanistic, differential diagnosis of where a breakdown originates.

Practical Advantages of Our Framework. Beyond its theoretical grounding, our framework offers significant practical advantages. It operates under weak supervision—requiring only ground-truth final answers rather than expensive, often ambiguous token-level annotations required by fully supervised detectors. Once calibrated, our method is highly efficient, requiring only the initial generation and a single non-autoregressive forward pass through the language decoder. This makes it significantly faster than multi-sample consistency methods Manakul et al. (2023); Farquhar et al. (2024) and highly scalable for real-world deployment.

### 3 Methodology: An Information-Geometric Framework for Diagnosing Hallucination

We reconceptualize VLM hallucination not as a simple output error, but as a symptom of a breakdown within the model’s internal information processing. We first formalize the ideal, logically self-consistent cognitive process through an axiomatic probabilistic graphical

model (PGM) that defines the normative flow of information: I −−−−−→Perception Tevi −−−−−→Inference A, where I is the visual input, Tevi is the explicitly generated evidence chain, and A is the final answer to a given query Q.

This model embodies a critical axiom from information theory: the generated evidence Tevi serves as a sufficient statistic for the final answer A with respect to the image I. Mathemati-

cally, this defines the ground truth of a rational process as one where the conditional mutual information is zero: I(A; I|Tevi) = 0. Deviations from this axiom signify a logical failure. While our conceptual framework applies to general generation, we anchor our mathematical formalization and primary diagnosis in structured Visual Question Answering (VQA) tasks, as their constrained action spaces allow for rigorous quantification of these latent failure modes.

An Information-Geometric View of Cognition. We define a VLM’s generation of a token trajectory τ as a probabilistic event drawn from a distribution P(τ|I, Q). The informational content of any specific trajectory is its self-information, or surprisal, defined as I(τ) = − log P(τ|I, Q). This allows for a rigorous, first-principles definition of hallucination: a nominal cognitive process is a low-surprisal event, corresponding to a high-probability trajectory that aligns with the model’s learned world model. Conversely, we define hallucination as a high-surprisal cognitive event—a rare, low-probability trajectory that deviates unexpectedly from this nominal behavior.

To diagnose such events, we project the high-dimensional internal state of the generation process into a low-dimensional, 3D Observable Information Manifold. Each generation is represented by a Cognitive State Vector v = [HEvi, SConf, HAns] on this manifold. We posit that nominal processes correspond to points residing in high-density regions, or ‘attractors’ on this manifold. Our three diagnostic probes are thus reinterpreted as direct, quantitative measures of the information flow’s properties at different stages:

- • Perceptual Entropy (HEvi) measures the initial state’s information entropy, quantifying the uncertainty in the evidence formulation stage.
- • Inferential Conflict (SConf) directly measures information leakage across cognitive stages, quantifying the violation of our core axiom.
- • Decision Entropy (HAns) quantifies the terminal state’s residual entropy, measuring the final decision uncertainty.

#### 3.1 Probing Perceptual Uncertainty (HEvi)

To quantify the initial state’s information entropy, we measure the uncertainty of the evidence formulation stage (I → Tevi) with Perceptual Entropy (HEvi).1 A high HEvi signifies an unstable starting point for the cognitive trajectory. We model the model’s choice at each token step i as a Bernoulli trial Ci ∈ {Factual,Uncertain} by defining two disjoint token subsets: a factual set Vf and an uncertainty set Vu. For the logits li of each token, we project the softmax distribution onto this semantic axis:

∑t∈Vf softmax(li)t ∑t∈Vf∪Vu softmax(li)t

p(Ci = F) =

, p(Ci = U) = 1 − p(Ci = F) (1)

The token-level entropy is the Shannon entropy Hi = H(Ci). The final metric is the pathaveraged entropy: HEvi = |T1

evi| ∑i|T=evi1 | Hi.

#### 3.2 Probing Inferential Conflict (SConf)

To operationalize our idealized causal graph (I → Tevi → A), we introduce Inferential Conflict (SConf). This probe estimates the Conditional Pointwise Mutual Information (CPMI) to quantify the strength of the illicit direct causal path from I to A. It is a pointwise metric for a specific outcome Atoken, making it highly suitable for diagnosing single, concrete instances of generation. It measures the information gain from the visual modality on the generated answer token Atoken2, conditioned on the textual evidence Tevi. This quantity is computed

- 1The complete word lists, adapted from prior work on language model uncertainty Ji et al. (2025); Yona et al. (2024), along with a sensitivity analysis, are in Appendix A.1.
- 2For multi-token answers, Atoken is defined as the first token corresponding to the primary decision keyword (e.g., ’Yes’ or ’No’). This localization is made reliable by the structured prompts used in our experimental setup.

Algorithm 1 Cognitive Anomaly Detection Framework (CAD)

Function DIAGNOSEHALLUCINATION (I, Q, M,CalibratedComponents)

- 1: (MGMM, µ, σ) ← CalibratedComponents
- 2:
- 3: ▷ 1. Generate cognitive trajectory and extract metrics
- 4: (Tevi, Amodel,scores) ← M.generate(I, Q,output_scores=True)
- 5: HEvi ← CalcPerceptualEntropy(scoresevi)
- 6: SConf ← CalcInferentialConflict(I, Q, Tevi, M)
- 7: HAns ← CalcDecisionEntropy(scoresans)
- 8:
- 9: ▷ 2. Compute anomaly score in the cognitive space
- 10: vnew ← [HEvi, SConf, HAns]
- 11: Standardize vnew using µ, σ.
- 12: Shall ← − log p(vnew|MGMM)
- 13: return Shall

// Note: CalibratedComponents are pre-computed offline by fitting a GMM on cognitive state vectors from a purified set of non-hallucinatory samples.

as the log-probability difference:

SConf = log pv(Atoken|I, Tevi) − log pt(Atoken|∅I, Tevi) (2) = CPMI(Atoken; I | Tevi) (3)

where pv is the probability with visual context and pt is the counterfactual probability without it. A large positive SConf indicates strong, positive point-wise information flowing directly from the visual input to the final answer, unmediated by the evidence, thus measuring the violation of d-separation. To obtain pt, we perform a causal intervention by replaying the generation process with the visual input ablated3 Pearl (2009). A practical boundary condition is that the VLM architecture must allow for such a causal intervention.

#### 3.3 Probing Decision Uncertainty (HAns)

Finally, to quantify the terminal state’s residual entropy, we measure the final uncertainty with Decision Entropy (HAns). A high entropy indicates the system has failed to converge to a stable, determined state.

HAns = − ∑

p(a) log2 p(a) (4)

a∈{Yes,No}

#### 3.4 Diagnosis via Geometric Anomaly Detection in the Cognitive State Space

Our framework performs diagnosis at inference-time on single instances after a one-time, hallucination-label-free calibration. This phase learns the geometric structure of the ‘nominal cognitive state space’ Snominal.

- Phase 1: Learning the Geometry of the Nominal Cognitive State Space. We represent each VLM generation by its 3D Cognitive State Vector v = [HEvi, SConf, HAns]. This calibration is

fitted on a calibration set Dcal. This process requires only ground-truth final answers (e.g., ‘Yes’/‘No’), a form of weak supervision that is vastly more accessible and scalable than obtaining fine-grained, token-level hallucination labels. We hypothesize that the landscape of nominal states is multi-modal, as different types of valid cognitive processes (e.g., simple object recognition versus complex relational reasoning) may form distinct, dense clusters in the state space. We therefore employ a Gaussian Mixture Model (GMM), which is naturally suited to capturing such underlying structures, to model the probability density

3In our implementation with Idefics2, this is achieved by providing ‘images=None’ to the processor during the text-only forward pass.

- Table 1: Main AUC results on the POPE benchmark (Adversarial). Best performance is in bold, second best is underlined. Our single-pass, weakly supervised CAD significantly outperforms all baselines.

Method Cost Llava-v1.6 Idefics2 Qwen2-VL DeepSeek-VL Average

| | | | |
|---|---|---|---|
|Token Entropy Neg Log Prob Supervised Probe Semantic Entropy Ours (CAD)|1x 1x 1x 10x 1x|0.603 0.806 0.409 0.732<br><br>0.604 0.832 0.428 0.710 0.787 0.898 0.762 0.715 0.711 0.751 0.673 0.702 0.910 0.947 0.776 0.798<br><br><br>|0.638 0.644 0.791 0.709 0.858<br><br>|

- Table 2: Macro-averaged AUC results on the MME benchmark. Our CAD framework shows strong generalization across diverse multimodal reasoning and perceptual tasks.

Method Llava-v1.6 Idefics2 Qwen2-VL DeepSeek-VL Average

Token Entropy 0.5106 0.5486 0.5291 0.6687 0.5643 Neg Log Prob 0.5061 0.5449 0.5136 0.6578 0.5556 Supervised Probe 0.7620 0.7905 0.7408 0.7104 0.7509 Semantic Entropy 0.6811 0.7081 0.6616 0.6199 0.6677 Ours (CAD) 0.8514 0.8411 0.7233 0.7680 0.7960

p(v|nominal). This set undergoes a Coherence Filter step: we first select for correct final answers and then apply automated heuristics to exclude ‘lucky guesses’ (e.g., cases where the model answers ‘Yes’ while its generated evidence explicitly states ‘There is no such object in the image’). This purification ensures our GMM learns a less biased estimate of the true density on Snominal. Prior to fitting, we standardize each dimension and determine the optimal number of GMM components via the Bayesian Information Criterion (BIC).

- Phase 2: Hallucination Diagnosis as a High-Surprisal Cognitive Event. From an information geometry perspective, a hallucination is a cognitive process whose state vector v is geometrically distant from the learned high-density regions (attractors). Its Hallucination Score is therefore the self-information content, or surprisal, of observing this atypical state vector:

Shall(v) = I(v) = − log p(v|MGMM) (5) This score quantifies the ‘unexpectedness’ of the observed cognitive trajectory. Nominal processes are common, predictable, low-information events, whereas hallucinations are rare, high-information deviations. The workflow is summarized in Algorithm 1.

### 4 Experiments

#### 4.1 Experimental Setup

Datasets and Multi-Dimensional Evaluation Protocol. Our work’s core philosophy is that object-level hallucination is merely the final, observable symptom of a broader cognitive failure. To thoroughly evaluate our framework across diverse settings and address the limitations of narrow benchmark testing, we design a multi-dimensional evaluation protocol:

- • Diagnostic Deep-Dive (POPE Li et al. (2023)): We transform the POPE benchmark into a rich diagnostic playground using Chain-of-Thought (CoT) prompts to external-

ize reasoning (Tevi). Following Li et al. (2023), we strictly focus on the ‘adversarial‘ subset to diagnose genuine, hard-to-detect hallucinations, which serves as our primary testbed for mechanistic analysis.

- • Comprehensive Generalization (MME Fu et al.): To ensure our method generalizes beyond specific task formats, we evaluate on the expansive MME benchmark, reporting macro-averaged results across its diverse perceptual and reasoning categories.

- • Open-Ended Validation (MS-COCO Lin et al. (2014)): As a targeted ablation, we validate our perceptual probe on open-ended image captioning, using the CHAIR Rohrbach et al. (2018) metric to confirm its independent generalizability.

Evaluated Models and Baselines. To ensure a comprehensive analysis across architectural families, we evaluate four state-of-the-art VLMs: Llava-v1.6-Mistral-7B Liu et al. (2024), Idefics2-8b Laurençon et al. (2023), Qwen2-VL Team (2024), and DeepSeek-VL2Small DeepSeek-AI (2024). We compare our proposed Cognitive Anomaly Detection (CAD) against a comprehensive suite of baselines: 1) Token Entropy and Neg Log Probability (token-level uncertainty); 2) a Supervised Probe Chen et al. (2024b) (a linear classifier trained on final hidden states using balanced hallucination labels); and 3) Semantic Entropy Farquhar et al. (2024) (a strong, sampling-based consistency baseline requiring 10× inference cost). Detailed metric calculations, GMM hyperparameter optimization (via BIC), and prompt templates are provided in the Appendix.

#### 4.2 State-of-the-Art Detection Across Diverse Benchmarks

We find that reframing hallucination as a diagnosable cognitive process anomaly leads to a state-of-the-art framework that is both effective and efficient.

Performance on POPE. As shown in Table 1 and Figure 2, our CAD framework—which models the distribution of only non-hallucinatory examples— achieves superior overall performance. Critically, the log-log ROC curves (Figure 2b) highlight that CAD maintains high true positive rates even at extremely low false positive rates (FPR < 10−2), a regime where baseline methods often fail. This confirms that modeling the geometric properties of the cognitive process offers superior reliability for real-world deployment compared to simple uncertainty thresholds.

Generalization on MME. Moving beyond simple object existence questions, Table 2 demonstrates CAD’s strong generalization capabilities on the comprehensive MME benchmark. MME encompasses diverse multimodal tasks including spatial reasoning, OCR, and commonsense logic. In these complex scenarios, the natural variance in generated text increases significantly. While the Supervised Probe, benefiting from in-domain training labels, shows competitive performance, our weaklysupervised CAD framework achieves comparable or even superior results (e.g., 0.851 on Llava-v1.6) without requiring any hallucination labels for calibration. This highlights CAD’s robustness and adaptability across a wide array of diverse task types.

(a) ROC (Linear Scale)

(b) ROC (Log-Log Scale)

100

| | |deepseek (0.798) Qwen2-VL (0.776|)|
|---|---|---|---|
| | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

1.0

TruePositiveRate(Log)

0.8

TruePositiveRate

0.6

10 1

0.4

Idefics2 (0.947) Llava-v1.6-Mistral-7B (0.910)

0.2

10 2

0.0

10 4 10 3 10 2 10 1 100

0.0 0.2 0.4 0.6 0.8 1.0

False Positive Rate (Linear)

False Positive Rate (Log)

Figure 2: ROC curves of our Cognitive Anomaly Detection (CAD) framework. (a) Linear-scale ROC curves show superior overall performance across all architectures. (b) log-log curves highlight CAD’s dominance in the critical low-FPR regime (FPR < 10−2), which is essential for reliable realworld deployment.

#### 4.3 Mechanistic Diagnosis: Unveiling Cognitive Fingerprints

The success of our CAD framework over the supervised ‘Supervised Probe‘ is a crucial finding, suggesting that analyzing the entire cognitive trajectory for anomalous patterns provides a richer signal than a localized biopsy of a single token’s state. While the SOTA results are compelling, a deeper question arises: what underlying mechanistic differences cause performance to vary so dramatically across models? Our framework’s primary strength is its diagnostic capability. By projecting the generative process into the 3D Cognitive State Space, we can visualize the ‘cognitive manifolds’ of nominal and hallucinatory behavior. As Figure 3 reveals, different VLM architectures follow strikingly different pathological pathways.

###### (a) Llava-v1.6-Mistral-7B

###### (b) Qwen2-VL

###### Perception vs. Inference

###### Perception vs. Decision

###### Inference vs. Decision

###### Perception vs. Inference

###### Perception vs. Decision

###### Inference vs. Decision

1.75

1.75

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 0

- 1

- 2

- 3

1.50

1.50

- 0
- 1
- 2
- 3
- 4

1.50

1.50

1.25

1.25

1.25

1.25

###### Non-Hallu.

###### Non-Hallu.

1.00

1.00

1.00

1.00

HAns

HAns

HAns

HAns

SCon

SCon

0.75

0.75

0.75

0.75

0.50

0.50

0.50

0.50

1

0.25

0.25

0.25

0.25

2

0.00

0.00

0.00

0.00

1

3

0.00 0.05 0.10 0.15

0.00 0.05 0.10 0.15

1 0 1 2 3

0.00 0.05 0.10

0.00 0.05 0.10

2 0 2 4

HEvi

HEvi

HEvi

HEvi

SConf

SConf

1.75

1.75

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 0

- 1

- 2

- 3

1.50

1.50

- 0
- 1
- 2
- 3
- 4

1.50

1.50

1.25

1.25

1.25

1.25

1.00

1.00

1.00

1.00

###### Hallu.

###### Hallu.

HAns

HAns

HAns

HAns

SCon

SCon

0.75

0.75

0.75

0.75

0.50

0.50

0.50

0.50

1

0.25

0.25

0.25

0.25

2

0.00

0.00

0.00

0.00

1

3

0.00 0.05 0.10 0.15

0.00 0.05 0.10 0.15

1 0 1 2 3

0.00 0.05 0.10

0.00 0.05 0.10

2 0 2 4

HEvi

HEvi

HEvi

HEvi

SConf

SConf

###### (c) Idefics2

###### (d) Deepseek-VL2-Small

###### Perception vs. Inference

###### Perception vs. Decision

###### Inference vs. Decision

###### Perception vs. Inference

###### Perception vs. Decision

###### Inference vs. Decision

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

2.5

1.2

1.2

1.2

1.2

- 0
- 1
- 2
- 3

2.0

1.0

1.0

1.0

1.0

1.5

###### Non-Hallu.

###### Non-Hallu.

0.8

0.8

0.8

0.8

1.0

HAns

HAns

HAns

HAns

0.6

0.6

SCon

SCon

0.6

0.6

0.5

0.4

0.4

0.4

0.4

0.0

0.2

0.2

0.2

0.2

0.5

0.0

0.0

0.0

0.0

1

0.00 0.05 0.10 0.15

0.00 0.05 0.10 0.15

0 1 2

0.00 0.05 0.10

0.00 0.05 0.10

0 2

HEvi

HEvi

HEvi

HEvi

SConf

SConf

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

2.5

1.2

1.2

1.2

1.2

- 0
- 1
- 2
- 3

2.0

1.0

1.0

1.0

1.0

1.5

0.8

0.8

0.8

0.8

###### Hallu.

###### Hallu.

1.0

HAns

HAns

HAns

HAns

0.6

0.6

SCon

SCon

0.6

0.6

0.5

0.4

0.4

0.4

0.4

0.0

0.2

0.2

0.2

0.2

0.5

0.0

0.0

0.0

0.0

1

0.00 0.05 0.10 0.15

0.00 0.05 0.10 0.15

0 1 2

0.00 0.05 0.10

0.00 0.05 0.10

0 2

HEvi

HEvi

HEvi

HEvi

SConf

SConf

Figure 3: Visualizing the ‘Cognitive Fingerprints’ of Hallucination. Density projections of the 3D Cognitive State Space, separated into non-hallucinatory (blue, top row of each pair) and hallucinatory (red, bottom row of each pair) processes. These manifolds reveal unique failure signatures for each model.

Idefics2’s ‘Structural Disorder’ Pattern. Idefics2 presents a stark signature. Its nominal processes form an extremely compact, low-variance manifold—a tight blue cluster representing a rigid, stable cognitive workflow. Hallucinations are characterized as anomalous deviations from this stable state. This suggests a structural failure mechanism, explaining why our density-based anomaly detector is exceptionally effective for this model.

Llava’s ‘Transparent Struggle’ Pattern. Llava-v1.6 exhibits a cognitively transparent pattern. Its hallucinatory manifold (red) occupies a region largely separable from the nominal one (blue), characterized by high Inferential Conflict and Decisional Ambiguity. Its internal cognitive struggle is explicitly manifested through our metrics, making its failure mode highly transparent.

Qwen2-VL & DeepSeek’s ‘Entangled States’ Pattern. These models exhibit the most insidious pattern: ‘confident lies.’ The ‘Entangled States’ pattern provides a direct, geometric explanation for their lower AUC scores. For these models, the hallucinatory manifold (red) forms its own dense, confident clusters that deeply intertwine with the ’healthy’ manifold (blue). This is not a failure of our method, but a profound diagnostic finding. It reveals that for certain architectures, hallucination is not merely a process anomaly but can be a content error originating from a seemingly normal process, highlighting a key challenge for future research.

#### 4.4 Ablation Study and Real-World Robustness

To validate our framework’s multi-component design and its practical viability, we conducted comprehensive ablation studies and stress-testing.

p = 2.8e 08 p = 5.6e 07 p = 5.1e 05 p = 3.2e 05

HEv

- Figure 5: Generalization of Perceptual Instability (HEvi) to Open-Ended Captioning (MSCOCO, N = 1000). Without any task-specific tuning, our perceptual probe consistently assigns significantly higher entropy to hallucinatory captions (red) compared to factual ones (blue) across all four architectures. Statistical Significance: The distinction is profound, with

Welch’s t-test yielding p ≪ 0.001 in all cases, validating that HEvi captures a fundamental cognitive signature of hallucination beyond VQA formats. Cross-Model Insight: The shared Y-axis highlights that models like Qwen2-VL and DeepSeek-VL2 (bottom row) exhibit higher baseline entropy in their factual generations compared to Llava and Idefics2.

The Necessity of a Holistic Diagnosis. The significant ‘Synergy Gain’ revealed in Figure 4(a) is a testament to the multi-dimensional nature of cognitive failure. For a model like Idefics2, no single probe provides a sufficient signal (individual AUCs ∼0.75-0.78). Only by viewing the cognitive trajectory as a point in our 3D state space can its ‘anomalous deviations’ be reliably detected, as evidenced by the jump to 0.947 AUC, quantitatively proving that a holistic diagnosis is necessary.

###### (a) Synergy Gain

###### (b) Component Impact

1.2

0.4

1.0

0.95

0.91

AbsoluteAUC

0.3

0.78 0.80

0.8

AUCDrop

0.6

0.2

0.4

0.1

0.2

0.0

0.0

Llava Idefics2 Qwen2 DeepSeek

Llava Idefics2 Qwen2 DeepSeek

HEvi (Perceptual) SConf (Inferential)

HAns (Decisional) Synergy Gain

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 4: Ablation Study. (a) Standalone metrics vs. synergistic gain. The gray hatched area represents the Synergy Gain. (b) Impact of component removal (∆AUC), revealing model-specific ‘diagnostic fingerprints’.

The Adaptive Diagnostician. Figure 4(b) reveals the unique ‘diagnostic fingerprint’ of each model. Our GMMbased detector acts as an adaptive diagnostician: for Llava, it learns to be highly sensitive to deviations along the Inferential Conflict axis; for DeepSeek-VL, it identifies Perceptual Instability as the key symptom. The GMM’s ability to learn where to focus—and what to ignore (e.g., the confounding noise from HEvi for Idefics2)—demonstrates its power to adapt to each model’s unique cognitive fingerprint. Generalization of Individual Probes: HEvi on Open-Ended Tasks. While our holistic framework is powerful, it is crucial to validate the independent efficacy and generalizability of its constituent probes. We conducted a targeted ablation on our Perceptual Instability (HEvi) probe by evaluating it on the challenging open-ended task of MS-COCO image captioning (N = 1000). Since unconstrained captioning lacks a binary decision boundary (e.g., Yes/No) to anchor the Inferential Conflict (SConf) and Decision Entropy (HAns) metrics, this setting allows us to isolate perceptual drift as a primary driver of hallucination in freeform generation. As shown in Figure 5, HEvi scores are significantly higher for hallucinated captions (validated via CHAIR Rohrbach et al. (2018)) across all models, with profound statistical significance (p ≪ 0.001). This result not only demonstrates the standalone power of our perceptual probe but also proves its ability to generalize far beyond structured VQA tasks.

Robustness to Calibration Contamination. CAD’s reliance on weak supervision—calibrating solely on samples with correct final answers—is a significant practical advantage, as perfectly clean data is often unavailable in real-world deployments. To evaluate its limits, we stress-test the framework by intentionally contaminating the calibration

###### (a) Llava-v1.6

###### (b) Idefics2

###### (c) Qwen2-VL

###### (d) DeepSeek-VL2-Small

1.00

0.944

0.95

0.922 0.919

0.906

DetectionAUC

- 0.90

0.862

0.85

0.806

0.797

0.779

0.80

0.772 0.769

0.752 0.747

0.75

CAD (Ours)

0.70

0% 5% 10% 15% 20% 25% 30%

0% 5% 10% 15% 20% 25% 30%

0% 5% 10% 15% 20% 25% 30%

0% 5% 10% 15% 20% 25% 30%

Noise Ratio

Noise Ratio

Noise Ratio

Noise Ratio

- Figure 6: Robustness to Calibration Contamination across Architectures. We evaluate detection performance (AUC) as the calibration set is increasingly contaminated with hallucinatory samples (0%–30%). (a) Sensitivity of Transparent Struggle: Llava-v1.6 exhibits a noticeable drop, as mistakenly treating its highly conflicted, high-variance hallucinatory states as nominal forces the GMM to blur the anomaly boundary. (b) Resilience of Structural Disorder: Idefics2 maintains high performance (> 0.91) with negligible degradation, as its compact nominal core remains robust against geometrically distinct outliers. (c, d) Stability of Entangled States: Qwen2-VL and DeepSeek-VL2 show flatter trajectories; their performance is constrained by the intrinsic geometric overlap of their states rather than calibration noise. The gray dashed line shows a strict 2% performance drop threshold relative to the clean baseline.

set with 0%–30% undetected hallucinations. As illustrated in Figure 6, the resilience of our GMM-based detector aligns elegantly with the underlying ‘cognitive fingerprints’ of each architecture. Idefics2 remains remarkably robust, maintaining an AUC above 0.91 even under 30% contamination. This confirms its Structural Disorder profile: the nominal cognitive core is so geometrically compact that the GMM effectively treats injected hallucinations as negligible outliers, leaving the learned density of the manifold intact.

In contrast, Llava-v1.6’s noise-sensitivity reflects its Transparent Struggle: including highvariance hallucinations forces GMM variance dilation, blurring thresholds. As its hallucinations are characterized by conspicuously high-variance and extreme inferential conflict, mistakenly including them as “nominal” forces the GMM to artificially stretch its variance parameters to encompass these anomalies. Conversely, Qwen2-VL and DeepSeek-VL2 remain stable as noise merely reinforces their inherent Entangled States overlap. Such resilience under severe contamination underscores CAD’s utility for robust real-world auditing.

### 5 Conclusion

In this paper, we introduce a diagnostic framework that reframes VLM hallucination from a static output error to a dynamic failure within the model’s cognitive process. By modeling generation as a three-stage cognitive trajectory and projecting it into an interpretable state space, we develop a state-of-the-art, single-pass, and weakly supervised anomaly detector. Crucially, our robust framework provides the first mechanistic categorization of VLM hallucinations, uncovering distinct ‘cognitive fingerprints’ for different architectures—from transparent failures to deeply entangled errors. This approach advances beyond mere detection, enabling the stage-by-stage diagnosis of complex, multi-modal failure modes, ultimately providing a transparent new lens for auditing and building reliable VisionLanguage Models.

### References

Amos Azaria and Tom M. Mitchell. The internal state of an LLM knows when its lying. In Conference on Empirical Methods in Natural Language Processing, 2023. doi: 10.18653/v1/2023.findings-emnlp.68.

Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey, 2024.

Yonatan Bitton, Hritik Bansal, et al. VisIT-Bench: A benchmark for vision-language instruction following inspired by real-world use. In Neural Information Processing Systems Track on Datasets and Benchmarks, 2023.

Beitao Chen, Xinyu Lyu, Lianli Gao, Jingkuan Song, and Hengtao Shen. Alleviating hallucinations in large vision-language models through hallucination-induced optimization. In Neural Information Processing Systems, 2024a.

Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. INSIDE: LLMs’ internal states retain the power of hallucination detection. In International Conference on Learning Representations, 2024b.

Tiejin Chen, Xiaoou Liu, Longchao Da, Vagelis Papalexakis, and Hua Wei. Uncertainty quantification of large language models through multi-dimensional responses, 2025.

Xuweiyi Chen, Ziqiao Ma, Xuejun Zhang, Sihan Xu, Shengyi Qian, Jianing Yang, David F. Fouhey, and Joyce Chai. Multi-object hallucination in vision-language models. In Neural Information Processing Systems, 2024c.

Jeremy R. Cole, Michael J.Q. Zhang, Daniel Gillick, Julian Martin Eisenschlos, Bhuwan Dhingra, and Jacob Eisenstein. Selectively answering ambiguous questions. In Conference on Empirical Methods in Natural Language Processing, 2023.

DeepSeek-AI. Deepseek-vl: Towards real-world vision-language understanding. https://deepseek. com/technologies/deepseek-vl, 2024.

Xuefeng Du, Chaowei Xiao, and Yixuan Li. HaloScope: Harnessing unlabeled LLM generations for hallucination detection. In Neural Information Processing Systems, 2024.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630:625–630, 2024. doi: 10.1038/s41586-024-07421-0.

Javier Ferrando, Oscar Obeso, Senthooran Rajamanoharan, and Neel Nanda. Do I know this entity? knowledge awareness and hallucinations in language models. In International Conference on Learning Representations, 2024.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, and Jonathan Herzig. Does fine-tuning LLMs on new knowledge encourage hallucinations? In Conference on Empirical Methods in Natural Language Processing, 2024.

Samuel J Gershman, Eric J Horvitz, and Joshua B Tenenbaum. Computational rationality: A converging paradigm for intelligence in brains, minds, and machines. Science, 349(6245):273–278, 2015.

Tianrui Guan, Fuxiao Liu, et al. HallusionBench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Yuheng Huang, Jiayang Song, Zhijie Wang, Shengming Zhao, Huaming Chen, Felix Juefei-Xu, and Lei Ma. Look before you leap: An exploratory study of uncertainty analysis for large language models. IEEE Transactions on Software Engineering, 51:413–429, 2023. doi: 10.1109/TSE.2024.3519464.

Hugging Face. Idefics2-8b-chatty model card. https://huggingface.co/HuggingFaceM4/ idefics2-8b-chatty, 2024.

Andrew Jesson, Nicolas Beltran-Velez, Quentin Chu, Sweta Karlekar, Jannik Kossen, Yarin Gal, John P. Cunningham, and David M. Blei. Estimating the hallucination rate of generative AI. In Neural Information Processing Systems, 2024.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023.

Ziwei Ji, Lei Yu, Yeskendir Koishekenov, Yejin Bang, Anthony Hartshorn, Alan Schelten, Cheng Zhang, Pascale Fung, and Nicola Cancedda. Calibrating verbal uncertainty as a linear feature to reduce hallucinations, 2025.

Jannik Kossen, Jiatong Han, Muhammed Razzak, Lorenz Schut, Shreshth A. Malik, and Yarin Gal. Semantic entropy probes: Robust and cheap hallucination detection in LLMs, 2024.

Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. Obelics: An open web-scale filtered dataset of interleaved image-text documents, 2023.

Qi Li and Xinchao Wang. Sponge tool attack: Stealthy denial-of-efficiency against tool-augmented agentic reasoning. arXiv preprint arXiv:2601.17566, 2026.

Qi Li, Cheng-Long Wang, Yinzhi Cao, and Di Wang. Data lineage inference: Uncovering privacy vulnerabilities of dataset pruning. arXiv preprint arXiv:2411.15796, 2024.

Qi Li, Runpeng Yu, Haiquan Lu, and Xinchao Wang. Every step counts: Decoding trajectories as authorship fingerprints of dllms. arXiv preprint arXiv:2510.05148, 2025a.

Qi Li, Runpeng Yu, and Xinchao Wang. Vid-sme: Membership inference attacks against large video understanding models. arXiv preprint arXiv:2506.03179, 2025b.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji rong Wen. Evaluating object hallucination in large vision-language models. In Conference on Empirical Methods in Natural Language Processing, 2023.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pp. 740–755. Springer, 2014.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-1.6: Improved reasoning, ocr, and world knowledge, 2024.

Potsawee Manakul, Adian Liusie, and Mark Gales. SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In Conference on Empirical Methods in Natural Language Processing, 2023.

Alexander Nikitin, Jannik Kossen, Yarin Gal, and Pekka Marttinen. Kernel language entropy: Finegrained uncertainty quantification for LLMs from semantic similarities. In Neural Information Processing Systems, 2024.

Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, and Yonatan Belinkov. LLMs know more than they show: On the intrinsic representation of LLM hallucinations. In International Conference on Learning Representations, 2024.

Antti Oulasvirta, Jussi PP Jokinen, and Andrew Howes. Computational rationality as a theory of interaction. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems, pp. 1–14, 2022.

Seongheon Park, Xuefeng Du, Min-Hsuan Yeh, Haobo Wang, and Yixuan Li. Steer LLM latents for

hallucination detection, 2025. Judea Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, 2009. Maxime Peyrard et al. LM2-Bench: A closer look at how well vlms implicitly link explicit matching

visual cues. In European Conference on Computer Vision, 2024.

Vishnu Prabhakaran, Purav Aggarwal, Vinay Kumar Verma, Gokul Swamy, and Anoop Saladi. VADE: Visual attention guided hallucination detection and elimination. In Annual Meeting of the Association for Computational Linguistics, 2025.

Yusu Qian, Haotian Zhang, Yinfei Yang, and Zhe Gan. How easy is it to fool your multimodal LLMs? an empirical analysis on deceptive prompts, 2024.

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. In Proceedings of the 2018 conference on empirical methods in natural language processing, pp. 4035–4045, 2018.

Bernadette J Stolz, Jared Tanner, Heather A Harrington, and Vidit Nanda. Geometric anomaly detection in data. Proceedings of the national academy of sciences, 117(33):19664–19669, 2020.

Zhiqing Sun, Sheng Shen, et al. Aligning large multimodal models with factually augmented rlhf (mmhal-bench), 2023.

Qwen Team. Qwen2-vl technical report. https://qwenlm.github.io/blog/qwen2-vl/, 2024.

- Chan Tsun et al. Clair: Evaluating image captions with large language models. In Conference on Empirical Methods in Natural Language Processing, 2024.

Minh Vu, Brian K. Tran, Syed A. Shah, Geigh Zollicoffer, N. Hoang-Xuan, and Manish Bhattarai. HalluField: Detecting LLM hallucinations via field-theoretic modeling, 2025.

- Chao Wang, Weiwei Fu, and Yang Zhou. TPC: Cross-temporal prediction connection for visionlanguage model hallucination reduction, 2025a.

Cheng-Long Wang, Qi Li, Zihang Xiang, Yinzhi Cao, and Di Wang. Towards lifecycle unlearning commitment management: Measuring sample-level unlearning completeness. In 34th USENIX Security Symposium (USENIX Security 25), pp. 6481–6500, 2025b.

David Wang et al. Contrastive region guidance: Improving grounding in vision-language models without training. In European Conference on Computer Vision, 2024.

Junyang Wang et al. What you see is what you read? improving text-image alignment evaluation. In Neural Information Processing Systems, 2023.

Michal Yarom et al. Visual riddles: a commonsense and world knowledge challenge for large vision and language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Gal Yona, Roee Aharoni, and Mor Geva. Can large language models faithfully express their intrinsic uncertainty in words? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 7752–7764, 2024.

Runpeng Yu, Qi Li, and Xinchao Wang. Discrete diffusion in large language and multimodal models: A survey. arXiv preprint arXiv:2506.13759, 2025.

### A Appendix

#### A.1 Implementation and Reproducibility Details

Prompt Template. For all experiments, we used a consistent Chain-of-Thought prompt to elicit the three-stage cognitive process. The template enforces a structured output format to facilitate parsing of the evidence chain (Tevi) and the final answer (A).

User: <image > You are an expert image analyst. Follow this output format strictly: First , provide a brief explanation of what you see in the image. Then , conclude with 'Therefore , the final answer is Yes.' or 'Therefore , the final answer is No.' Question: {question}

Assistant:

- Figure 7: The Chain-of-Thought prompt template used in our experiments. The explicit structural constraints ensure reliable separation of evidence and decision tokens.

Metric Calculation Details. The three core metrics are calculated as follows. Let V be the vocabulary and I the image.

- 1. Uncertainty Word Set (VU). The complete set of uncertainty-related words used to construct VU was adapted from prior work on linguistic uncertainty Ji et al. (2025); Yona et al. (2024): {"probably", "likely", "possibly", "might", "may", "seems", "appears", "perhaps", "suggests", "could", "believe", "guess", "assume", "unlikely", "not sure", "could be"}, including their space-prefixed variations (e.g., " probably").
- 2. Probe Calculations.

- • HEvi (Perceptual Instability): For each token i in the evidence chain Tevi with probability distribution pi, let the total probability of uncertainty tokens be Pi(U) = ∑t∈VU pi(t). The semantic binary entropy is HB(pi) = −Pi(U)log2 Pi(U) − (1 − Pi(U)) log2(1 − Pi(U)). HEvi is the mean of HB(pi) over all tokens in Tevi.
- • SConf (Inferential Conflict): Let tans be the final answer token (e.g., "Yes" or "No") and Cevi be the full context preceding it. This is calculated as the Conditional Pointwise Mutual Information (CPMI):

SConf = log P(tans|I, Cevi) − log P(tans|∅I, Cevi) (6) The second term is obtained by a teacher-forcing replay of the generated token sequence using only the textual prompt as input, with visual features entirely excluded (passed as zero tensors or ‘None‘). This ratio quantifies the specific information gain from the visual modality at the decision step.

- • HAns (Decisional Ambiguity): This is the standard Shannon entropy of the probability distribution P over the entire vocabulary V at the position of the final answer token: HAns = − ∑w∈V P(w) log P(w).

Supervised Probe Implementation. For the ‘Supervised Probe‘ baseline, we used the code provided by Chen et al. (2024b) to implement a linear probe with ‘LogisticRegression‘ from ‘scikit-learn‘. The probe was trained on hidden states extracted from the final token of the generated evidence chain. This location was determined by identifying the sequence marker "Therefore, the final answer is" and taking the hidden state of the token immediately preceding it. Training was performed on a balanced dataset using 5-fold stratified crossvalidation to ensure fair comparison.

#### A.2 GMM Calibration and Hyperparameters

Model Selection via BIC. A critical hyperparameter for our Gaussian Mixture Model (GMM) is the number of components K. We assume the "manifold of nominal cognition" is not necessarily unimodal (e.g., different reasoning types may form different clusters). To determine the optimal K, we employed the Bayesian Information Criterion (BIC) on the calibration set. We swept K ∈ [1,10].

As shown in Figure 8, distinct minima were observed for each model, reflecting their varying internal latent structures. The optimal components selected were: 5 for Llava-v1.6, 7 for Idefics2, 8 for Qwen2-VL, and 7 for DeepSeek-VL2.

Calibration Data Purification (Coherence Filter). To ensure our GMM models the density of truly nominal cognitive processes, we applied a "Coherence Filter" to the calibration set (which consists of samples where the final answer is correct). This step removes "lucky guesses"—cases where the model answers correctly despite hallucinating in the evidence chain. The heuristic rules are:

- 1. Negation Consistency: If the final answer is "Yes", the evidence chain must not contain strong negation phrases (e.g., "no [object]", "not present") associated with the query object.
- 2. Object Existence: If the final answer is "Yes", the object name must appear in the evidence chain without being preceded by negative modifiers.

Samples failing these checks were excluded from the calibration set Dcal, ensuring the learned manifold Snominal represents consistent reasoning trajectories.

###### (a) Llava-v1.6

###### (b) Idefics2

###### (c) Qwen2-VL

###### (d) DeepSeek-VL

25000

25000

22500

21000

20000

20000

20000

20000

17500

15000

15000

BICScore

19000

15000

12500

10000

10000

18000

10000

5000

5000

17000

7500

5000

0

0

16000

K=5

K=7

K=8

K=7

2500

1 2 3 4 5 6 7 8 9 10

1 2 3 4 5 6 7 8 9 10

1 2 3 4 5 6 7 8 9 10

1 2 3 4 5 6 7 8 9 10

Number of Components (K)

Number of Components (K)

Number of Components (K)

Number of Components (K)

- Figure 8: GMM Model Selection. Bayesian Information Criterion (BIC) scores for varying numbers of Gaussian components. The red dashed line indicates the selected optimal K for each model, minimizing the BIC score.

### B Future Work

The diagnostic nature of our framework opens several exciting avenues for future work. The ability to pinpoint the stage of cognitive failure paves the way for targeted, stage-specific mitigation strategies. For instance, a high HEvi could trigger a visual re-evaluation mechanism (e.g., zooming or re-sampling), while a high SConf could activate a more rigorous logical verification step (e.g., self-consistency checks). Furthermore, extending this framework to mitigate the most deceptive ‘deep cognitive errors’—where models are confidently wrong (low entropy, high distance)—remains a critical direction for building trustworthy VLM systems.

