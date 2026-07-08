# arXiv:2603.18472v2[cs.AI]9Apr2026

## Cognitive Mismatch in Multimodal Large Language Models for Discrete Symbol Understanding

#### Yinghui Li1, Jiayi Kuang2, Peng Xing1, Daixian Liu1, Yongheng Zhang4, Junnan Dong3, Shu-Yu Guo1, Yangning Li1, Qingyu Zhou4, Wenhao Jiang5, Hai-Tao Zheng1, Ying Shen2, Liang Lin2, and Philip S. Yu6

1Tsinghua University, China 2Sun Yat-sen University, China 3The Hong Kong Polytechnic University, China 4Independent Researcher 5Guangdong Laboratory of AI and Digital Economy (SZ), China 6University of Illinois Chicago, USA

*E-mail: liyinghuihhh@gmail.com

#### ABSTRACT

Multimodal large language models (MLLMs) perform strongly on natural images, yet their ability to understand discrete visual symbols remains unclear. We present a multi-domain benchmark spanning language, culture, mathematics, physics and chemistry, organized into three cognitive levels: perception and recognition, combination and reasoning, and association and critical thinking. Across leading MLLMs, we observe a consistent cognitive mismatch. Models frequently underperform on elementary symbol recognition while appearing relatively competent on more complex reasoning tasks. This recognitionreasoning inversion indicates that current systems often compensate with linguistic priors, template retrieval or procedural reasoning instead of robust visual grounding. The pattern is especially clear for sparse, low-redundancy symbols such as handwritten characters, formula graphs, circuit diagrams and chemical structures. These results show that symbolic understanding remains a major bottleneck for multimodal intelligence and motivate training and evaluation schemes that prioritize grounded perception in discrete semantic spaces.

#### Introduction

Since the advent of the Large Language Models era in Artificial Intelligence, Multimodal Large Language Models (MLLMs) have consistently remained one of the most cutting-edge and prominent research topics1–3. Beyond textual media, MLLMs aim to endow artificial systems with the ability to see, perceive and reason about the physical world, thereby moving toward a more comprehensive form of intelligence. In recent years, the rapid rise of embodied intelligence has further elevated the significance of MLLMs4–6. The fundamental reason behind this trend is that the cognitive paradigm represented by MLLM is closer to human intelligence in understanding the world and represents an essential step towards achieving Artificial General Intelligence (AGI)7–9. Consequently, understanding and emulating fundamental mechanisms of human cognition in the real world has become crucial for advancing MLLMs.

Symbols have been the indispensable cornerstone of human cognition and the evolution of intelligence since the dawn of the human species10. From prehistoric cave paintings encoding survival knowledge to the structured syntax of natural language, symbolic systems allow humans to transcend the limits of individual sensory experience and accumulate abstract knowledge across generations11–13. As cognitive science represented by semiotic theories has elucidated, human intelligence is inherently symbolic, relying on the creation, manipulation and communication of symbols to support reasoning and collective understanding14,15. For multimodal intelligence, the open question is therefore not simply whether a model can produce a plausible answer, but whether it can truly perceive discrete symbols and ground its reasoning in the correct symbolic evidence.

The distinction is especially sharp between continuous and discrete semantic spaces. Mainstream MLLMs are primarily trained on natural scenes and image-text pairs in which visual inputs map to relatively holistic semantic narratives, as in captioning, visual question answering and grounding16–18. By contrast, the images we focus on are composed of semantically independent symbolic units whose meaning depends on exact recognition and structured composition. In a handwritten Chinese sentence, a function graph, a circuit diagram or a chemical structure, a single misread stroke, bond or relation can overturn the whole interpretation. This representational gap is illustrated in Fig. 1, which contrasts continuous semantic spaces with the discrete symbolic processing route that is far more natural to human cognition.

Across prior research on MLLMs, investigations into discrete symbols remain notably scarce. Existing evaluation suites cover broad multimodal understanding, OCR-heavy perception, expert reasoning and scientific problem solving19–27.

a. Continuous and Discrete Semantic Spaces b. The Human Eye’s Sensitivity to Visual Symbols

[Figure 1]

[Figure 2]

[Figure 3]

c. The Hierarchical Neurocognitive Pathway for Discrete Symbolic Semiosis

[Figure 4]

###### L3: ASSOCIATION & CRITICAL THINKING

Associates the overall symbolic structure for high-level linguistic semantics, detects and automatically corrects errors.

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

###### L2: COMBINATION & REASONING (Compositional Symbol Reasoning)

L1: RECOGNITION & PERCEPTION (Basic Symbol Recognition)

Combines multiple recognized symbols to infer discrete semantics.

Processes raw visual data into recognizable symbols.

- Figure 1. Continuous and discrete semantic spaces differ fundamentally in how visual information is organized. Discrete symbol understanding requires precise recognition of individual symbolic units and their structured relations.

Specialized methods have also begun to address symbolic domains such as documents, music, art, geometry, chemistry and other structured scientific inputs28–40. However, these approaches remain fragmented across specific domains and rarely disentangle whether models truly recognize the symbols they manipulate.

To address this problem, we construct a unified benchmark spanning five symbolic domains, each organized into three cognitive levels: perception and recognition, combination and reasoning, and association and critical thinking. This hierarchy follows the progressive route by which humans process symbols, from recognizing local visual units to integrating them compositionally and finally judging their semantic consistency in context. Across these domains, we find a recurrent and counterintuitive pattern: models often fail at foundational symbol recognition yet produce apparently competent higher-level reasoning. This recognition-reasoning inversion is the core empirical signature of the cognitive mismatch we study.

Our benchmark covers language symbols, cultural symbols, mathematical symbols, physical symbols and chemical symbols, thereby spanning both the social-scientific and natural-scientific symbolic systems that structure human knowledge. The task design is intentionally hierarchical. At the first level, models must identify the symbolic semantics of individual units, such

- as malformed handwritten characters, graph elements, circuit primitives or atoms and bonds in a molecular structure. At the second level, they must integrate multiple symbols and reason over their joint semantics, for example by combining local evidence in a graph or diagram into a higher-order conclusion. At the third level, they must move beyond straightforward recognition and perform correction, consistency judgment or culturally mediated interpretation. This progression allows us to ask not only whether a model can answer a question, but also what cognitive route is likely supporting that answer.

Using this design, we curate a large-scale benchmark containing more than 13,000 image-question-answer instances and 38 sub-tasks. The central motivation is diagnostic rather than competitive. In natural images, an approximately correct semantic summary is often sufficient; in discrete symbolic images, a minor local deviation can cause a total semantic shift. A missing stroke can turn a correct character into an invalid one, an omitted bond can alter a molecule, and a misread axis relation can invalidate a mathematical conclusion. This property makes discrete semantic spaces especially suitable for testing whether MLLMs truly ground their outputs in visual evidence, or instead rely on linguistic priors, memorized templates and shortcuts.

This perspective also explains why we place equal emphasis on social-symbolic and scientific-symbolic domains. Language and cultural symbols probe whether models can handle irregularity, convention and context-sensitive meaning in settings close to everyday cognition. Mathematics, physics and chemistry, by contrast, stress formal structure, low tolerance for local error and the need to preserve symbolic consistency over longer reasoning chains. Bringing these domains together lets us examine whether current MLLMs possess a general capacity for symbolic understanding, or whether their successes remain local, domain-dependent and heavily contingent on the kinds of patterns most represented in pre-training data.

[Figure 11]

###### Figure 2. Performance and representative cases for language-symbol understanding across three cognitive levels. Models are evaluated on faked-character detection, misspelled-character detection and visual-semantic correction.

#### Results

###### Language symbols

Language symbols expose the most immediate gap between visual perception and language-driven generation. As shown in Figure 2, performance remains weak across all three levels. In the faked-character detection task, the overall performance of most models is extremely poor, often failing to distinguish an invalid handwritten character from its nearest legal counterpart. The dominant failure mode is forced normalization: the model silently repairs an anomalous glyph into the closest familiar character and thereby erases the very evidence it is asked to detect. A second failure mode is unstable localization, in which models follow the instruction to mark errors but confuse legal and illegal characters because they lack fine-grained structural discrimination. Both behaviors indicate that the symbolic image is being collapsed into a coarse linguistic prior before a reliable character-space representation is formed.

This weakness further propagates into the more difficult tasks of misspelled-character detection and visual-semantic correction. Only a few models achieve usable exact-match accuracy, and even then, many corrections are only directionally relevant rather than fully correct. Once the first visual decision is wrong, later correction becomes semantically unstable: a misperceived character can trigger an entirely unrelated lexical continuation. The language domain therefore, provides a particularly clean and compelling illustration of our central finding. Higher-level outputs may appear fluent, but this outward fluency often masks a fundamental failure to anchor the answer rigorously in the correct symbolic evidence. A fuller breakdown of the language results is provided in Supplementary Sec. B.1.

Qualitative cases reinforce this interpretation. Some models do not treat malformed characters as anomalies at all, but instead automatically replace them with the nearest valid glyph during generation, effectively overwriting the perceptual evidence at the earliest stage. Others attempt to follow the task instruction but still fail to separate structurally incorrect forms from merely difficult handwriting, producing a high number of predicted error positions with poor precision. In the correction setting, the same instability can lead to severe semantic drift: once a character is misread, the model may continue with a semantically coherent but irrelevant phrase. This shows that the major obstacle is not only linguistic correction itself, but the absence of a stable mechanism for maintaining semantic consistency from visual symbol recognition to textual output.

The language tasks, therefore, illuminate a broader issue in multimodal learning. Because large language models are optimized to produce well-formed text, they appear inclined to normalize uncertainty into familiar lexical forms. That tendency is useful for open-ended generation, but detrimental when the task requires preserving anomaly, ambiguity, or fine-grained deviation. In this sense, the weakness of handwritten symbols is not a marginal OCR problem. It is a structural sign that current MLLMs still struggle to keep visual uncertainty alive long enough for higher-level reasoning to operate on it faithfully.

###### Cultural symbols

Compared with handwritten characters, emoji-based tasks produce stronger low-level performance, especially for common English expressions. As summarized in Figure 3, models often recover the literal semantics of individual emojis and, in simple cases, compose them into familiar words, suggesting that high-frequency and socially standardized visual symbols are easier to align with language than sparse handwritten forms. Even so, the apparent strength is fragile. Once a sequence requires the model to preserve a negative operator, a discourse constraint or a culturally conventional reading, many systems fall back on the most statistically accessible phrase rather than the meaning actually licensed by the whole symbolic configuration in context.

The limitation becomes more evident in Chinese idiom tasks, where correct interpretation depends not only on object semantics but also on homophony, cultural convention and cross-symbol associative reasoning. Here, character-level overlap remains substantially higher than exact idiom recovery, showing that models can often map isolated emojis to local meanings while failing to complete the intended symbolic leap. In other words, the bottleneck is no longer simple recognition of the visual token itself, but the controlled integration of multiple symbolic cues without hallucinating a more familiar expression.

This domain is especially revealing because it sits between literal recognition and culturally grounded semiosis. In English emoji tasks, some models already show competent low-level composition, indicating that they can align familiar pictographic symbols with lexical units learned from web-scale corpora. However, once the task requires suppressing an overly salient local meaning, performance drops rapidly. In Chinese idiom tasks the challenge is even sharper: correct answers may depend on object semantics, homophonic correspondence and shared cultural convention at the same time. The resulting gap between partial character recovery and full idiom recovery indicates that many models can access fragments of the intended semantics, yet remain unable to consolidate them into the culturally appropriate symbolic interpretation.

This pattern matters because cultural symbols are often treated as comparatively easy for multimodal models. Our results suggest a more nuanced picture. Emoji understanding is indeed easier when the task can be reduced to lexical substitution, but it remains difficult when symbolic interpretation must be constrained by context or by a social convention that cannot be read directly from the icon itself. The cultural domain thus shows that even highly familiar visual symbols can expose the limits of multimodal grounding once meaning becomes relational, conventional and compositionally constrained. Detailed analyses for both English emoji composition and Chinese idiom interpretation are given in Supplementary Sec. B.2.

[Figure 12]

###### Figure 3. Performance and representative cases for cultural-symbol understanding. The tasks progress from lexical grounding of emojis to idiomatic composition and culturally mediated interpretation.

Mathematical symbols Mathematical symbols reveal a sharp inversion between low-level perception and apparently stronger higher-level reasoning.

- Figure 4 illustrates this contrast. In Level 1 tasks, models frequently miss basic visual distinctions such as function type, graph shape or local geometric attributes, because these tasks demand precise localization and have extremely low error tolerance. Yet they sometimes perform better on Level 2 and 3 tasks, where linguistic rules and familiar mathematical templates can partially compensate for visual uncertainty. Models may fail to read a curve faithfully while still producing a plausible explanation by invoking properties of logarithmic, exponential or quadratic functions from prior knowledge.

Qualitative cases show that even when the answer is correct, the path is often language-dominant rather than image-dominant. Models tend to bypass direct visual judgment and instead reconstruct the problem through symbolic calculation, textual cues or memorized proof patterns. This behavior explains why stronger reasoning scores should not be read as evidence of stronger perception. Rather, the mathematical domain shows very clearly how procedural inference can conceal deficits in the visual grounding of graphs, shapes and spatial relations.

The inversion is informative as it suggests poor Level 1 performance reflects not total incapacity, but a language-dominant preference. When questions contain enough textual cues or resemble familiar mathematical patterns, the model often mobilizes internal rule knowledge to compensate for incomplete perception. However, this compensation is selective and unstable. It works best when a problem is reconstructed from stereotyped functional or geometric schemas, yet fails when success depends on faithfully reading local visual structure. Thus, the mathematical domain reveals a core MLLM contradiction: stronger symbolic reasoning coexists with weaker symbolic seeing, because the former frequently bypasses the latter.

Meanwhile, the mathematical results caution against interpreting reasoning traces at face value. A long and apparently rigorous derivation can coexist with a weak visual parse of the graph or geometric figure. In several representative cases, the model’s explanation sounds mathematically competent because it is anchored in general rule knowledge, not because it has securely extracted the local visual details required by the task. This is why mathematical symbolic evaluation must distinguish between procedural fluency and grounded diagram understanding rather than treating them as the same capability in practice. Further mathematical analyses are reported in Supplementary Sec. B.3.

Physical symbols Physics tasks show that sparse scientific symbols remain difficult even when the underlying verbal knowledge is familiar. As

- Figure 5 indicates, most models perform poorly on basic recognition of formulas, graphs and circuit relations, indicating that they struggle to convert visual physical symbols into stable mathematical or conceptual representations. Many responses can correctly recite the relevant law in words but still fail at the symbolic mapping step that the question actually requires. A model may invoke Ohm’s law, for example, yet still confuse the functional form of the quantity being plotted or substitute an incorrect parameter while otherwise following a plausible reasoning chain.

This gap between verbal knowledge and symbolic execution widens with multi-step reasoning. In several cases, the reasoning trace remains locally coherent but is derailed by an early visual error, such as a misread constant, a misidentified graph or an omitted structural relation in the diagram. The physics domain therefore highlights that scientific reasoning in multimodal systems depends not only on correct laws, but also on preserving symbolic consistency from the first visual parse.

The physics results also show that correct verbal knowledge alone is not sufficient for grounded multimodal competence. Some models can reproduce textbook definitions or standard physical principles with impressive fluency, yet once asked to translate a concrete visual configuration into the appropriate symbolic relation, their reasoning becomes unstable. In this sense, physical symbols expose a separation between knowing a law and reading the specific instantiation of that law from a diagram. The more sparsely encoded and low-redundancy the visual representation becomes, the more visible this weakness is, especially in tasks involving circuits, mechanics trajectories and graph-based quantity mapping.

From the perspective of scientific use, this distinction is critical. Physics problems often appear visually simple while actually depending on exact symbolic correspondences among quantities, directions, circuit components and geometric constraints. A system that can narrate the correct principle but cannot preserve those correspondences is not merely making a small perceptual mistake; it is failing at the core interface between visual evidence and formal reasoning. The physical-symbol domain therefore highlights why robust multimodal science requires more than verbal scientific literacy. Supplementary Sec. B.4 provides the extended physics analyses and representative error patterns.

###### Chemical symbols

Among all domains, chemistry makes the structural nature of complex symbolic understanding especially explicit. Figure 6 shows that in Level 1 tasks, many models can identify salient individual atom labels or obvious bond types, but they often ignore the implicit topological rules of skeletal formulas. Carbon and hydrogen counts are therefore routinely omitted or inferred incorrectly, aromatic structures are flattened into simpler bond categories, and local visual fragments are combined through memorized templates rather than performing true structural parsing. These errors fundamentally show that current visual encoders still struggle with sparse lines, small overlaps and convention-dependent notation today.

[Figure 13]

###### Figure 4. Performance and representative cases for mathematical-symbol understanding, covering function graphs and geometric figures across recognition, reasoning, and verification tasks.

[Figure 14]

###### Figure 5. Performance and representative cases for physical-symbol understanding across mechanics and electromagnetism. Sparse physical symbols remain difficult for most models, especially in visually grounded mapping tasks.

[Figure 15]

###### Figure 6. Performance and representative cases for chemical-symbol understanding. The benchmark examines molecular structure recognition, reaction reasoning and higher-level correction or prediction.

When the task shifts to balancing reactions, identifying conditions or predicting products, performance becomes more dependent on rule-based integration. Some models can detect that a reaction is unbalanced or that a low-temperature reagent is implicated, but their internal reasoning often remains pattern-based rather than causally grounded. Correct answers are possible, yet unstable: once an earlier bond, substituent or coefficient is misread, the downstream chemical chain collapses. Chemistry therefore provides one of the clearest demonstrations that symbolic reasoning without robust symbolic perception is inherently brittle and difficult to generalize.

At the same time, chemistry shows that stronger reasoning-oriented models can sometimes outperform visually stronger models in intermediate tasks that rely heavily on rule integration. This divergence suggests that visual recognition ability and symbolic reasoning ability are not advancing in lockstep. A model may detect that a reaction needs balancing or invoke plausible chemical heuristics, yet still fail to ground those operations in the exact molecular or reaction structure shown in the image. Conversely, a model with better local parsing may still underperform if it cannot organize those symbols into a stable reasoning chain. The domain therefore makes visible the two-sided nature of the cognitive mismatch: current models are weak both at seeing fine-grained chemical symbols and at preserving structural consistency when reasoning over them.

Chemistry is especially valuable here because symbolic compression is unusually high: tiny visual marks encode valency, substitution, reaction conditions and transformation pathways. As a result, a model cannot rely on broad scene semantics to recover the missing meaning. It must preserve exact local structure. The persistent failures on skeletal formulas and reaction diagrams therefore provide strong evidence that the current multimodal pipeline still lacks an effective mechanism for converting sparse visual notation into a stable symbolic substrate on which domain reasoning can reliably operate. Expanded chemical results are presented in Supplementary Sec. B.5.

#### Discussion

Across five domains, the same pattern recurs. Current MLLMs do not fail merely because symbolic tasks are difficult in a generic sense; they fail because discrete symbol understanding demands a kind of visual precision that can no longer be absorbed into broad language plausibility. Human participants follow a more canonical cognitive trajectory, with high performance on foundational perceptual tasks and gradually lower performance as reasoning becomes more demanding. Many models, by contrast, exhibit the reverse tendency: they underperform on the perceptual baseline and then appear to recover on tasks where linguistic priors, procedural templates or contextual inference can compensate for missing visual evidence. This recognition-reasoning inversion is the operational form of the cognitive mismatch identified in this work.

The implication is not that current MLLMs lack reasoning ability. On the contrary, several models display strong semantic flexibility and can sometimes repair early perceptual errors through downstream inference. The problem is that this compensatory mechanism makes end-task success an unreliable indicator of genuine visual understanding. A model may arrive

- at the correct answer for the wrong reason, or may fail catastrophically when symbolic precision cannot be approximated by linguistic expectation. This helps explain why sparse handwritten characters, formula graphs, circuit diagrams and chemical structures remain persistent failure points despite impressive progress on natural-image benchmarks.

More broadly, our findings suggest that the dominant training paradigm still privileges jump mapping from vision to language concepts over the formation of stable symbolic visual primitives. Advancing multimodal intelligence in scientific and abstract domains will therefore require stronger supervision on discrete symbols, tighter coupling between perceptual evidence and reasoning, and evaluation protocols that explicitly separate grounded recognition from compensatory inference. Extended related work, fuller per-domain analyses, human-baseline details and additional case studies are provided in Supplementary Secs. A.1, B, B.6 and E.

The human baseline is important in this regard. Human participants do not show the same inversion pattern. Instead, they follow a more intuitive cognitive trajectory: perceptually grounded tasks are solved well, while performance gradually declines as association, correction and multi-step reasoning become more demanding. This contrast suggests that the current MLLM pipeline does not simply constitute a weaker version of human symbolic cognition; rather, it follows a qualitatively different route, one in which language prior often dominates over visual grounding. That difference matters because it changes how errors emerge. Human symbolic failures often occur after a symbol has been correctly perceived, whereas model failures frequently originate in the earliest perceptual step and are only later concealed by fluent reasoning.

These observations also have implications for model design. The issue is unlikely to be explained by data scarcity alone. Current visual encoders are optimized for continuous natural scenes, in which coarse semantic summaries are often sufficient, but discrete symbols demand preservation of topological and local structural information at a much finer granularity. Future progress will therefore likely require not only more data, but also training objectives and architectural biases that explicitly encourage the formation of stable symbolic visual primitives and force reasoning modules to remain anchored to visual evidence. Without such changes, models may continue to improve on benchmark-style reasoning while remaining fragile on the symbol systems that underpin human science, language and culture.

For evaluation, the same lesson applies. If benchmarks emphasize only final-answer correctness, then compensatory inference can easily be mistaken for grounded understanding. A model may appear strong because it reaches the right answer through prior knowledge, semantic plausibility or pattern completion, even when the underlying visual parse is wrong. Our benchmark is designed to make that discrepancy visible by separating recognition, reasoning and critical symbolic judgment across multiple domains. In doing so, it aims not only to compare current models, but also to provide a clearer target for future multimodal systems that must genuinely perceive and manipulate the discrete symbolic structures through which humans organize knowledge more faithfully.

#### Methods

###### Benchmark design

We construct a comprehensive and multidisciplinary benchmark for discrete semantic spaces across five foundational domains: language, culture, mathematics, physics and chemistry. The design follows a rigorous three-level cognitive hierarchy inspired by established principles of human symbolic processing. Level 1, Perception and Recognition, tests whether a model can identify the meaning of individual symbolic units from their visual form. Level 2, Combination and Reasoning, examines whether multiple symbols can be integrated into a coherent structured interpretation. Level 3, Association and Critical Thinking, evaluates whether the model can detect anomalies, correct errors or infer unconventional meanings from symbolic context.

This hierarchy lets us distinguish the nuanced mechanics of visual grounding from downstream reasoning. In continuous natural images, approximate semantic alignment is often sufficient. In discrete symbolic images, however, a small local mistake can alter the global answer entirely, leading to a complete failure of the reasoning chain. Our task design therefore emphasizes precision, compositional structure and cross-level diagnosis rather than focusing only on superficial end-to-end success. The complete task framework is detailed in Supplementary Sec. C.1.

###### Task construction across domains

For language symbols, we evaluate faked-character detection, misspelled-character detection and visual-semantic correction, focusing on handwritten Chinese characters with strong visual compositionality and structural complexity. For cultural symbols, we use emoji sequences to test lexical grounding, idiomatic composition in English and culturally mediated idiom inference in Chinese. For mathematical symbols, we cover function graphs and geometric figures, progressing from entity recognition to property reasoning and consistency checking. For physical symbols, we design tasks in mechanics and electromagnetism that require interpretation of intricate graphs, formulas, schematics and their interdependent relations. For chemical symbols, we evaluate molecular structural formulas and reaction equations, from atom- and bond-level parsing to balancing, condition inference and product prediction.

Across these diverse domains, the benchmark contains more than 13,000 meticulously curated instances. Some datasets are curated and reannotated from existing resources, while others are newly constructed specifically for our symbolic objectives. The goal is not merely to measure domain knowledge, but to test whether models can preserve symbolic fidelity as cognitive demands gradually increase throughout the levels. Additional dataset construction details are given in Supplementary Sec. C.2.

###### Models, prompting and human baseline

We evaluate a mixture of closed-source and open-source MLLMs, including GPT-4o, Claude-sonnet-4, Gemini-2.5-pro, o3, Qwen-max, Qwen2.5-VL, InternVL3-8B, Deepseek-vl2-tiny and LLaMA3-LLaVA-Next-8B, representing the current state-of-the-art in multimodal intelligence. Each model is queried with task-specific prompts aligned to the required output format of the benchmark. We keep prompts as direct as possible so that performance reflects genuine symbolic understanding rather than the effects of prompt engineering.

To contextualize the machine results, we also build a robust human baseline on a stratified sample of 1,000 instances from the full benchmark. Five highly educated bilingual annotators complete the tasks using the same prompts as the MLLMs, providing a gold-standard reference for our comparative analysis. This comparison is used to examine whether model behavior follows the same difficulty trajectory as human symbolic cognition when facing increased complexity. Further implementation details, together with the extended human-baseline analysis, are reported in Supplementary Secs. C.3 and B.6.

###### Evaluation

Evaluation is tailored to the intrinsic symbolic properties of each domain. For language symbols, we report precision, recall and F1 for error detection, along with exact match and edit distance for correction. For cultural symbols, we use exact-match style metrics at the word and sentence levels, character-overlap measures for partial idiom recovery, and an additional semanticsimilarity score to capture cases in which outputs are conceptually related but not lexically identical to the ground truth. For mathematical, physical and chemical symbols, we use accuracy on each task because the target outputs are structurally constrained and logically discrete.

These metrics are intended to separate partial recognition, approximate semantic recovery and exact symbolic correctness within discrete spaces. In the symbolic domains studied here, this distinction is essential: a response can be fluent, plausible and still fundamentally wrong in the precise way that matters most scientifically and structurally. The full metric definitions are provided in Supplementary Sec. C.3.

#### Data availability

Datasets introduced in this study are freely available at https://huggingface.co/datasets/Eternity-gaga/SymbolBench.

#### Code availability

The source code of this study is publicly available on Github at https://github.com/THUKElab/SymbolBench.

#### References

- 1. Caffagni, D. et al. The revolution of multimodal large language models: A survey. In Findings of the Association for Computational Linguistics: ACL 2024, 13590–13618 (2024).
- 2. Song, S. et al. How to bridge the gap between modalities: Survey on multimodal large language model. IEEE Transactions on Knowl. Data Eng. (2025).
- 3. Zhang, D. et al. Mm-llms: Recent advances in multimodal large language models. In Findings of the Association for Computational Linguistics ACL 2024, 12401–12430 (2024).
- 4. Xu, J., Sun, Q., Han, Q.-L. & Tang, Y. When embodied ai meets industry 5.0: Human-centered smart manufacturing. IEEE/CAA J. Autom. Sinica 12, 485–501 (2025).
- 5. Turgunbaev, R. et al. From perception to action with integrated vla systems. Tech. Sci. Integr. Res. 1, 11–17 (2025).
- 6. Szot, A. et al. From multimodal llms to generalist embodied agents: Methods and lessons. In Proceedings of the Computer Vision and Pattern Recognition Conference, 10644–10655 (2025).
- 7. Du, C. et al. Human-like object concept representations emerge naturally in multimodal large language models. Nat. Mach. Intell. 1–16 (2025).
- 8. Fei, H. et al. From multimodal llm to human-level ai: Modality, instruction, reasoning, efficiency and beyond. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024): Tutorial Summaries, 1–8 (2024).
- 9. Shen, H. et al. Zoomeye: Enhancing multimodal llms with human-like zooming capabilities through tree-based image exploration. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 6613–6629

(2025).

- 10. Perlovsky, L. I. Symbols: Integrated cognition and language. In Semiotics and intelligent systems development, 121–151 (IGI Global Scientific Publishing, 2007).
- 11. Taniguchi, T. et al. Symbol emergence in cognitive developmental systems: a survey. IEEE transactions on Cogn. Dev. Syst. 11, 494–516 (2018).
- 12. Bickerton, D. Symbol and structure: a comprehensive framework for language evolution. Stud. Evol. Lang. 3, 77–93

(2003).

- 13. Miyagawa, S., Lesure, C. & Nóbrega, V. A. Cross-modality information transfer: A hypothesis about the relationship among prehistoric cave paintings, symbolic thinking, and the emergence of language. Front. Psychol. 9, 115 (2018).
- 14. Rapoport, A. The role of symbols in human behavior. ETC: A Rev. Gen. Semant. 180–188 (1955).
- 15. Ahn, T., Janssen, M. A. & Ostrom, E. Signals, symbols, and human cooperation. In The origins and nature of sociality, 122–139 (Routledge, 2017).
- 16. Abdulgalil, H. D. & Basir, O. A. Next-generation image captioning: A survey of methodologies and emerging challenges from transformers to multimodal large language models. Nat. Lang. Process. J. 100159 (2025).
- 17. Kuang, J. et al. Natural language understanding and inference with mllm in visual question answering: A survey. ACM Comput. Surv. 57, 1–36 (2025).
- 18. Xiao, L., Yang, X., Lan, X., Wang, Y. & Xu, C. Towards visual grounding: A survey. IEEE Transactions on Pattern Analysis Mach. Intell. 1–20, DOI: 10.1109/TPAMI.2025.3630635 (2025).

- 19. Liu, Y. et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, 216–233 (Springer, 2024).
- 20. Li, B. et al. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13299–13308 (2024).
- 21. Fu, C. et al. MME: A comprehensive evaluation benchmark for multimodal large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2025).
- 22. Yue, X. et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9556–9567 (2024).
- 23. Li, Z. et al. A survey of state of the art large vision language models: Benchmark evaluations and challenges. In Proceedings of the Computer Vision and Pattern Recognition Conference, 1587–1606 (2025).
- 24. Xu, P. et al. Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models. IEEE Transactions on Pattern Analysis Mach. Intell. (2024).
- 25. Fu, L. et al. OCRBench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2025).
- 26. Lu, P. et al. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations (2024).
- 27. Zhang, R. et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, 169–186 (Springer, 2024).
- 28. Liu, Y. et al. Textmonkey: An ocr-free large multimodal model for understanding document. arXiv preprint arXiv:2403.04473 (2024).
- 29. Yu, Y.-Q. et al. Texthawk: Exploring efficient fine-grained perception of multimodal large language models. arXiv preprint arXiv:2404.09204 (2024).
- 30. Nacson, M. S. et al. Docvlm: Make your vlm an efficient reader. In Proceedings of the Computer Vision and Pattern Recognition Conference, 29005–29015 (2025).
- 31. Tang, M. et al. NOTA: Multimodal music notation understanding for visual large language model. In Chiruzzo, L., Ritter, A. & Wang, L. (eds.) Findings of the Association for Computational Linguistics: NAACL 2025, 7160–7173, DOI: 10.18653/v1/2025.findings-naacl.399 (Association for Computational Linguistics, Albuquerque, New Mexico, 2025).
- 32. Jiang, R. & Chen, C. W. Multimodal llms can reason about aesthetics in zero-shot. In Proceedings of the 33rd ACM International Conference on Multimedia, 6634–6643 (2025).
- 33. Fanelli, N., Vessio, G. & Castellano, G. Artseek: Deep artwork understanding via multimodal in-context reasoning and late interaction retrieval. arXiv preprint arXiv:2507.21917 (2025).
- 34. Gao, J. et al. G-LLaVA: Solving geometric problem with multi-modal large language model. In The Thirteenth International Conference on Learning Representations (2025).
- 35. Shi, W. et al. Math-LLaVA: Bootstrapping mathematical reasoning for multimodal large language models. In Al-Onaizan, Y., Bansal, M. & Chen, Y.-N. (eds.) Findings of the Association for Computational Linguistics: EMNLP 2024, 4663–4680, DOI: 10.18653/v1/2024.findings-emnlp.268 (Association for Computational Linguistics, Miami, Florida, USA, 2024).
- 36. Pan, Y. et al. Enhancing the geometric problem-solving ability of multimodal llms via symbolic-neural integration. In Proceedings of the 33rd ACM International Conference on Multimedia, 5394–5403 (2025).
- 37. Shi, W. et al. Multimodal mathematical reasoning with diverse solving perspective. arXiv preprint arXiv:2507.02804

(2025).

- 38. Tan, Q. et al. Chemmllm: Chemical multimodal large language model. arXiv preprint arXiv:2505.16326 (2025).
- 39. Li, J. et al. Chemvlm: Exploring the power of multimodal large language models in chemistry area. In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, 415–423 (2025).
- 40. Zhao, Z. et al. Chemdfm-x: towards large multimodal model for chemistry. Sci. China Inf. Sci. 67, 220109 (2024).
- 41. Bloom, B. S. et al. Taxonomy of. Educ. Object. 250 (1956).
- 42. Eco, U. A theory of semiotics, vol. 217 (Indiana University Press, 1979).

- 43. Hiippala, T. et al. Ai2d-rst: a multimodal corpus of 1000 primary school science diagrams. Lang. Resour. Eval. 55, 661–688 (2021).
- 44. Qin, Y. et al. InFoBench: Evaluating instruction following ability in large language models. In Ku, L.-W., Martins, A. & Srikumar, V. (eds.) Findings of the Association for Computational Linguistics: ACL 2024, DOI: 10.18653/v1/2024. findings-acl.772 (Association for Computational Linguistics, Bangkok, Thailand, 2024).
- 45. Tang, J. et al. Mtvqa: Benchmarking multilingual text-centric visual question answering. In Findings of the Association for Computational Linguistics: ACL 2025, 7748–7763 (2025).
- 46. Wang, Y. et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Adv. Neural Inf. Process. Syst. 37, 95266–95290 (2024).
- 47. Fu, X. et al. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, 148–166 (Springer, 2024).
- 48. Ying, K. et al. MMT-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask AGI. In Salakhutdinov, R. et al. (eds.) Proceedings of the 41st International Conference on Machine Learning, vol. 235 of Proceedings of Machine Learning Research, 57116–57198 (PMLR, 2024).
- 49. Meng, F. et al. MMIU: Multimodal multi-image understanding for evaluating large vision-language models. In The Thirteenth International Conference on Learning Representations (2025).
- 50. Jiang, D. et al. Mantis: Interleaved multi-image instruction tuning. Transactions on Mach. Learn. Res. (2024). Featured Certification, Outstanding Certification.
- 51. Chen, J. et al. MEGA-bench: Scaling multimodal evaluation to over 500 real-world tasks. In The Thirteenth International Conference on Learning Representations (2025).
- 52. Wang, F. et al. Muirbench: A comprehensive benchmark for robust multi-image understanding. In The Thirteenth International Conference on Learning Representations (2025).
- 53. Wang, S. et al. MFC-bench: Benchmarking multimodal fact-checking with large vision-language models. In Workshop on Reasoning and Planning for Large Language Models (2025).
- 54. Kazemzadeh, S., Ordonez, V., Matten, M. & Berg, T. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), 787–798 (2014).
- 55. Mao, J. et al. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, 11–20 (2016).
- 56. Li, C. et al. Elevater: A benchmark and toolkit for evaluating language-augmented visual models. Adv. Neural Inf. Process. Syst. 35, 9287–9301 (2022).
- 57. Li, Y. et al. Evaluating object hallucination in large vision-language models. In Bouamor, H., Pino, J. & Bali, K. (eds.) Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 292–305, DOI: 10.18653/v1/2023.emnlp-main.20 (Association for Computational Linguistics, Singapore, 2023).
- 58. Guan, T. et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14375–14385 (2024).
- 59. Lin, S., Hilton, J. & Evans, O. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: long papers), 3214–3252 (2022).
- 60. Li, J., Cheng, X., Zhao, X., Nie, J.-Y. & Wen, J.-R. Halueval: A large-scale hallucination evaluation benchmark for large language models. In The 2023 Conference on Empirical Methods in Natural Language Processing (2023).
- 61. Zeng, Z. et al. MR-ben: A meta-reasoning benchmark for evaluating system-2 thinking in LLMs. In The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024).
- 62. Rein, D. et al. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling (2024).
- 63. Hao, Y. et al. Can MLLMs reason in multimodality? EMMA: An enhanced multimodal reasoning benchmark. In Forty-second International Conference on Machine Learning (2025).
- 64. Mialon, G., Fourrier, C., Wolf, T., LeCun, Y. & Scialom, T. GAIA: a benchmark for general AI assistants. In The Twelfth International Conference on Learning Representations (2024).

- 65. Xu, F. F., Alon, U., Neubig, G. & Hellendoorn, V. J. A systematic evaluation of large language models of code. In Proceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming, MAPS 2022, 1–10, DOI: 10.1145/3520312.3534862 (Association for Computing Machinery, New York, NY, USA, 2022).
- 66. Uniyal, M., Singh, M., Verbruggen, G., Gulwani, S. & Le, V. One-to-many testing for code generation from (just) natural language. In Al-Onaizan, Y., Bansal, M. & Chen, Y.-N. (eds.) Findings of the Association for Computational Linguistics: EMNLP 2024, 15397–15402, DOI: 10.18653/v1/2024.findings-emnlp.902 (Association for Computational Linguistics, Miami, Florida, USA, 2024).
- 67. Liu, J., Xia, C. S., Wang, Y. & Zhang, L. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Adv. Neural Inf. Process. Syst. 36, 21558–21572 (2023).
- 68. Jain, N. et al. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations (2025).
- 69. Shaham, U., Ivgi, M., Efrat, A., Berant, J. & Levy, O. ZeroSCROLLS: A zero-shot benchmark for long text understanding. In The 2023 Conference on Empirical Methods in Natural Language Processing (2023).
- 70. Jin, B., Yoon, J., Han, J. & Arik, S. O. Long-context LLMs meet RAG: Overcoming challenges for long inputs in RAG. In The Thirteenth International Conference on Learning Representations (2025).
- 71. Zhang, X. et al. ∞Bench: Extending long context evaluation beyond 100K tokens. In Ku, L.-W., Martins, A. & Srikumar, V. (eds.) Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 15262–15277, DOI: 10.18653/v1/2024.acl-long.814 (Association for Computational Linguistics, Bangkok, Thailand, 2024).
- 72. Wang, H. et al. Multimodal needle in a haystack: Benchmarking long-context capability of multimodal large language models. In Chiruzzo, L., Ritter, A. & Wang, L. (eds.) Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 3221–3241, DOI: 10.18653/v1/2025.naacl-long.166 (Association for Computational Linguistics, Albuquerque, New Mexico, 2025).
- 73. Li, K. et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22195–22206 (2024).
- 74. Fang, X. et al. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. Adv. Neural Inf. Process. Syst. 37, 89098–89124 (2024).
- 75. Zhou, J. et al. Mlvu: Benchmarking multi-task long video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, 13691–13701 (2025).
- 76. Wu, H., Li, D., Chen, B. & Li, J. Longvideobench: A benchmark for long-context interleaved video-language understanding. Adv. Neural Inf. Process. Syst. 37, 28828–28857 (2024).
- 77. Wang, W. et al. Lvbench: An extreme long video understanding benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 22958–22967 (2025).
- 78. Liu, Y. et al. TempCompass: Do video LLMs really understand videos? In Ku, L.-W., Martins, A. & Srikumar, V. (eds.) Findings of the Association for Computational Linguistics: ACL 2024, 8731–8772, DOI: 10.18653/v1/2024.findings-acl. 517 (Association for Computational Linguistics, Bangkok, Thailand, 2024).
- 79. Fu, C. et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, 24108–24118 (2025).
- 80. Cheng, K. et al. Seeclick: Harnessing GUI grounding for advanced visual GUI agents. In ICLR 2024 Workshop on Large Language Model (LLM) Agents (2024).
- 81. Qiu, Z. et al. Can large language models understand symbolic graphics programs? In The Thirteenth International Conference on Learning Representations (2025).
- 82. Rawles, C. et al. Androidworld: A dynamic benchmarking environment for autonomous agents. In The Thirteenth International Conference on Learning Representations (2025).
- 83. Li, W. et al. On the effects of data scale on UI control agents. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2024).
- 84. Li, K. et al. Screenspot-pro: GUI grounding for professional high-resolution computer use. In Workshop on Reasoning and Planning for Large Language Models (2025).

- 85. Xie, T. et al. Scaling computer-use grounding via user interface decomposition and synthesis. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2025).
- 86. De Saussure, F. The linguistic sign. Semiot. An introductory anthology 35 (1985).
- 87. Key, L. & Noble, B. P. An analysis of Ferdinand de Saussure’s Course in general linguistics (Macat Library, 2017).
- 88. Valdez, J. L. C. et al. Semiotics and artificial intelligence (ai): An analysis of symbolic communication in the age of technology. In Future of Information and Communication Conference, 481–494 (Springer, 2024).
- 89. Danesi, M. Ai, popular culture, semiotics. In AI-Generated Popular Culture: A Semiotic Perspective, 1–22 (Springer, 2024).
- 90. Hatt, M. & Klonk, C. Semiotics. In Art history, 200–222 (Manchester University Press, 2025).
- 91. Iskanderova, T. What is semiotics? In Unveiling Semiotic Codes of Fake News and Misinformation: Contemporary Theories and Practices for Media Professionals, 5–9 (Springer, 2024).
- 92. Yu, H. et al. Benchmarking vision-language models on chinese ancient documents: From ocr to knowledge reasoning

(2025). 2509.09731.

- 93. Li, Y. et al. Towards real-world writing assistance: A Chinese character checking benchmark with faked and misspelled characters. In Ku, L.-W., Martins, A. & Srikumar, V. (eds.) Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 8656–8668, DOI: 10.18653/v1/2024.acl-long.469 (Association for Computational Linguistics, Bangkok, Thailand, 2024).
- 94. Chen, Z., tingzhu chen, Zhang, W. & Zhai, G. OBI-bench: Can LMMs aid in study of ancient script on oracle bones? In The Thirteenth International Conference on Learning Representations (2025).
- 95. Fuentes-Ferrer, R., Duque-Domingo, J. & Herrera, P. J. Recognition of egyptian hieroglyphic texts through focused generic segmentation and cross-validation voting. Appl. Soft Comput. 171, 112793 (2025).
- 96. Liu, X., Han, X., Chen, S., Dai, W. & Ruan, Q. Ancient yi script handwriting sample repository. Sci. Data 11, 1183

(2024).

- 97. Liu, C. et al. Improving sentiment analysis accuracy with emoji embedding. J. Saf. Sci. Resil. 2, 246–252 (2021).
- 98. Zheng, Y., Lyu, H. & Luo, J. Irony in emojis: A comparative study of human and llm interpretation. arXiv preprint arXiv:2501.11241 (2025).
- 99. Kuang, J. et al. Express what you see: Can multimodal llms decode visual ciphers with intuitive semiosis comprehension? In Findings of the Association for Computational Linguistics: ACL 2025, 12743–12774 (2025).
- 100. Nayak, S. et al. Benchmarking vision language models for cultural understanding. In Al-Onaizan, Y., Bansal, M. & Chen, Y.-N. (eds.) Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 5769–5790, DOI: 10.18653/v1/2024.emnlp-main.329 (Association for Computational Linguistics, Miami, Florida, USA, 2024).
- 101. Zheng, C. et al. Artmentor: Ai-assisted evaluation of artworks to explore multimodal large language models capabilities. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, 1–18 (2025).
- 102. Mundada, G. et al. WildScore: Benchmarking MLLMs in-the-wild symbolic music reasoning. In Christodoulopoulos, C., Chakraborty, T., Rose, C. & Peng, V. (eds.) Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 16858–16874, DOI: 10.18653/v1/2025.emnlp-main.853 (Association for Computational Linguistics, Suzhou, China, 2025).
- 103. Wang, K. et al. Measuring multimodal mathematical reasoning with math-vision dataset. Adv. Neural Inf. Process. Syst. 37, 95095–95169 (2024).
- 104. Zou, C. et al. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. In The Thirteenth International Conference on Learning Representations (2025).
- 105. Nath, O., Bathina, H., Khan, M. S. U. R. & Khapra, M. M. Can vision-language models evaluate handwritten math? In Che, W., Nabende, J., Shutova, E. & Pilehvar, M. T. (eds.) Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 14784–14814, DOI: 10.18653/v1/2025.acl-long.720 (Association for Computational Linguistics, Vienna, Austria, 2025).
- 106. Shi, Y. et al. Amsbench: A comprehensive benchmark for evaluating mllm capabilities in ams circuits. arXiv preprint arXiv:2505.24138 (2025).
- 107. Shen, H. et al. Phyx: Does your model have the" wits" for physical reasoning? arXiv preprint arXiv:2505.15929 (2025).

- 108. Dai, S. et al. PhysicsArena: The first multimodal physics reasoning benchmark exploring variable, process, and solution dimensions. In Christodoulopoulos, C., Chakraborty, T., Rose, C. & Peng, V. (eds.) Findings of the Association for Computational Linguistics: EMNLP 2025, 17290–17316, DOI: 10.18653/v1/2025.findings-emnlp.937 (Association for Computational Linguistics, Suzhou, China, 2025).
- 109. Wang, L. et al. Physunibench: An undergraduate-level physics reasoning benchmark for multimodal models. arXiv preprint arXiv:2506.17667 (2025).
- 110. Xiang, K. et al. Seephys: Does seeing help thinking? – benchmarking vision-based physics reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2025).
- 111. Zhai, Z. et al. Chemtables: a dataset for semantic classification on tables in chemical patents. J. Cheminformatics 13, 97

(2021).

- 112. Huang, Y. et al. Chemeval: a comprehensive multi-level chemical evaluation for large language models. arXiv preprint arXiv:2409.13989 (2024).
- 113. Guo, K. et al. Can llms solve molecule puzzles? a multimodal benchmark for molecular structure elucidation. Adv. Neural Inf. Process. Syst. 37, 134721–134746 (2024).
- 114. Yin, S. et al. A survey on multimodal large language models. Natl. Sci. Rev. 11, nwae403, DOI: 10.1093/nsr/nwae403

(2024). https://academic.oup.com/nsr/article-pdf/11/12/nwae403/61201557/nwae403.pdf.

- 115. Chu, W. et al. Scaling llama 3 training with efficient parallelism strategies. In Proceedings of the 52nd Annual International Symposium on Computer Architecture, ISCA ’25, 1703–1716, DOI: 10.1145/3695053.3731410 (Association for Computing Machinery, New York, NY, USA, 2025).
- 116. McKinzie, B. et al. Mm1: methods, analysis and insights from multimodal llm pre-training. In European Conference on Computer Vision, 304–323 (Springer, 2024).
- 117. Wu, J. et al. Visionllm v2: An end-to-end generalist multimodal large language model for hundreds of vision-language tasks. Adv. Neural Inf. Process. Syst. 37, 69925–69975 (2024).
- 118. Radford, A. et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763 (PmLR, 2021).
- 119. Jia, C. et al. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, 4904–4916 (PMLR, 2021).
- 120. Li, J., Li, D., Savarese, S. & Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, 19730–19742 (PMLR, 2023).
- 121. Liu, H., Li, C., Wu, Q. & Lee, Y. J. Visual instruction tuning. Adv. neural information processing systems 36, 34892–34916

(2023).

- 122. Lin, B. et al. Video-LLaVA: Learning united visual representation by alignment before projection. In Al-Onaizan, Y., Bansal, M. & Chen, Y.-N. (eds.) Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 5971–5984, DOI: 10.18653/v1/2024.emnlp-main.342 (Association for Computational Linguistics, Miami, Florida, USA, 2024).
- 123. Liu, H., Li, C., Li, Y. & Lee, Y. J. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 26296–26306 (2024).
- 124. Li, B. et al. LLaVA-onevision: Easy visual task transfer. Transactions on Mach. Learn. Res. (2025).
- 125. Bai, S. et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 (2025).
- 126. Yang, A. et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).
- 127. Wang, W. et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265 (2025).
- 128. Yang, L. et al. MMaDA: Multimodal large diffusion language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025).
- 129. Fu, C. et al. VITA-1.5: Towards GPT-4o level real-time vision and speech interaction. In The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025).
- 130. Zhu, E. et al. MAPS: Advancing multi-modal reasoning in expert-level physical science. In The Thirteenth International Conference on Learning Representations (2025).

- 131. Wiesner, F., Wessling, M. & Baek, S. Towards a physics foundation model. arXiv preprint arXiv:2509.13805 (2025).
- 132. Edwards, C. et al. Translation between molecules and natural language. In Goldberg, Y., Kozareva, Z. & Zhang, Y. (eds.) Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 375–413, DOI: 10.18653/v1/2022.emnlp-main.26 (Association for Computational Linguistics, Abu Dhabi, United Arab Emirates, 2022).
- 133. Peng, S. et al. Multimath: Bridging visual and mathematical reasoning for large language models. arXiv preprint arXiv:2409.00147 (2024).
- 134. Zhang, D. et al. Chemllm: A chemical large language model. arXiv preprint arXiv:2402.06852 (2024).
- 135. Ding, L., Zhao, M., Yin, F., Zeng, S. & Liu, C.-L. A large-scale database for chemical structure recognition and preliminary evaluation. In 2022 26th International Conference on Pattern Recognition (ICPR), 1464–1470 (IEEE, 2022).
- 136. He, C. et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 3828–3850 (2024).
- 137. Zhang, X. et al. Evaluating the performance of large language models on gaokao benchmark. arXiv preprint arXiv:2305.12474 (2023).
- 138. Wu, Z. et al. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302 (2024).
- 139. Grattafiori, A. et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).
- 140. Zhu, J. et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479 (2025).
- 141. Hurst, A. et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).
- 142. Anthropic. Claude 4. https://www.anthropic.com/news/claude-4 (2024). Accessed: 2025-10-13.
- 143. Team, Q. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115 (2024).
- 144. OpenAI. Introducing o3 and o4-mini. https://openai.com/zh-Hans-CN/index/introducing-o3-and-o4-mini/ (2025). Accessed: 2025-10-13.
- 145. Comanici, G. et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025).
- 146. Papineni, K., Roukos, S., Ward, T. & Zhu, W.-J. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, 311–318 (2002).

#### Contents

Supplementary 21

- A Extended Introduction 21 A.1 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- A.1.1 General Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- A.1.2 Symbolic Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- A.1.3 Architectures for Symbolic Domain . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- B Extended Results 25

- B.1 Language Symbols . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- B.1.1 Weak Recognition Ability for Faked Characters . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- B.1.2 Insufficient Recognition of Misused Characters . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- B.1.3 Lack of a Mechanism for Maintaining Semantic Consistency . . . . . . . . . . . . . . . . . . . . 26

- B.2 Cultural Symbols . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- B.2.1 Models perform well at the low level but still face challenges . . . . . . . . . . . . . . . . . . . . 26
- B.2.2 Models show limited performance in Chinese idiom tasks . . . . . . . . . . . . . . . . . . . . . 27
- B.2.3 Semantic Similarity Analysis Reveals Random Guessing Patterns in Association and Critical Thinking 27

- B.3 Mathematical Symbols . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- B.3.1 Visual perception ability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- B.3.2 Model performance comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- B.3.3 Fine-grained task analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- B.4 Physical Symbols . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- B.4.1 Limited recognition of sparse physical symbols across most models . . . . . . . . . . . . . . . . . 28
- B.4.2 Improved performance through symbolic reasoning yet with marked divergence . . . . . . . . . . . 29
- B.4.3 Global correction and prediction remain challenging for most models . . . . . . . . . . . . . . . . 29

- B.5 Chemical Symbols . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- B.5.1 Most models show limited capability in basic chemical symbol recognition . . . . . . . . . . . . . 29
- B.5.2 Rule-based symbolic integration drives reasoning yet exposes structural gaps . . . . . . . . . . . . 30
- B.5.3 Only a few models succeed in global chemical correction and prediction . . . . . . . . . . . . . . 30

- B.6 Human Performance Baseline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- B.7 Comprehensive analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

- B.7.1 Models generally lack the ability to recognize complex symbols . . . . . . . . . . . . . . . . . . 31
- B.7.2 The models exhibit rote memorization rather than genuine understanding . . . . . . . . . . . . . . 31
- B.7.3 Reasoning ability sometimes exists as a compensatory mechanism . . . . . . . . . . . . . . . . . 32
- B.7.4 There is no model that excels in all domains . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- B.7.5 The model structure may be the culprit for insufficient understanding of discrete symbolic. . . . . . . 34

- C Extended Methods 34

- C.1 Task Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34

- C.1.1 Language Symbols . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- C.1.2 Cultural Symbols . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- C.1.3 Mathematical Symbols . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- C.1.4 Chemical Symbols . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- C.1.5 Physical Symbols . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37

- C.2 Dataset Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38

- C.2.1 Data Sources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- C.2.2 Data Annotation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- C.2.3 Data Validation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- C.2.4 Data Statistics and Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39

- C.3 Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40

- C.3.1 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
- C.3.2 Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41

- D Extended Conclusion and Future Perspectives 43
- E Additional Case Studies 43

Supplementary

- A Extended Introduction

Since the advent of the Large Language Models era in Artificial Intelligence, Multimodal Large Language Models (MLLMs) have consistently remained one of the most cutting-edge and prominent research topics1–3. Beyond textual media, MLLMs aim to endow artificial systems with the ability to see, perceive, and reason about the physical world, thereby moving toward a more comprehensive form of intelligence. In recent years, the rapid rise of Embodied Intelligence has further elevated the significance of MLLMs4–6. The fundamental reason behind this trend is that the cognitive paradigm represented by MLLM is closer to human intelligence in understanding the world and represents an essential step towards achieving Artificial General Intelligence (AGI)7–9. Consequently, understanding and emulating fundamental mechanisms of human cognition in the real world has become crucial for advancing MLLMs, which is also our core objective: to promote MLLMs to reason in a manner more aligned with human thought, thereby fostering more human-like artificial intelligence.

Symbols have been the indispensable cornerstone of human cognition and the evolution of intelligence since the dawn of the human species10. From prehistoric cave paintings encoding survival knowledge to the structured syntax of natural language, symbolic systems allow humans to transcend the limits of individual sensory experience and accumulate abstract knowledge across generations11–13. As cognitive science represented by semiotic theories has elucidated, human intelligence is inherently symbolic, relying on the creation, manipulation, and communication of symbols to support reasoning and collective understanding14,15. Crucially, as illustrated in Figure 1(b), the human visual system exhibits unique sensitivity to visual symbols, enabling perceptual inputs to be rapidly encoded into discrete symbolic representations such as characters, gestures, and schematic patterns. This symbolic capacity is not merely a tool for communication but constitutes the very fabric of human thinking—underpinning our ability to conceptualize abstract ideas, solve complex problems, and construct shared realities.

However, this intrinsic alignment between human cognition and discrete symbols stands in stark contrast to the dominant training paradigms of current MLLMs. As depicted in Figure 1(a), continuous semantic spaces and discrete semantic spaces differ fundamentally in their representational structure. Most existing MLLMs are optimized to process continuous visual signals—such as natural images of scenes—mapping them to coherent semantic narratives through tasks such as image captioning16, Visual Question Answering17, and visual grounding18. In contrast, discrete semantic spaces consist of semantically independent symbolic units, where meaning emerges from precise identification and combinatorial relations among symbols. For example, in symbolic images such as mathematical equations or chemical structure diagrams, correct interpretation requires the model to recognize each symbol as a discrete semantic entity and reason over its structured composition. This representational gap poses a fundamental challenge for current MLLMs.

Across prior research on MLLMs, investigations into discrete symbols remain notably scarce. It is precisely this gap in understanding the mechanisms by which MLLMs process discrete symbolic visual information that motivates the present work. To bridge this, we draw inspiration from hierarchical neural and cognitive pathways underlying human symbolic processing, as illustrated in Figure 1(c). Cognitive neuroscience suggests that humans do not process symbols in a flat, end-to-end manner. Instead, symbolic cognition unfolds along a progressive pipeline, which begins with recognition and perception, where raw visual inputs are parsed into recognizable symbolic units, followed by combination and reasoning, where symbols are syntactically combined to infer compositional meaning. At the highest level, association and critical thinking monitor logical consistency, detect errors, and resolve ambiguities. We argue that true mastery of discrete semantic spaces by MLLMs requires competence across the entire cognitive spectrum, rather than relying solely on statistical correlations or linguistic priors. Thus, by investigating the visual semiotic behaviors of MLLMs in the discrete semantic space we define, we believe that our research not only fills a key gap in current MLLM research but also lays the foundation for developing intelligent systems that are more interpretable and more closely aligned with human symbolic cognition.

To make this benchmark framing more concrete, Figure 7 in Supplementary Sec. A visualizes the three-level hierarchy together with representative task examples across the five domains.

Operationalizing the above hierarchical cognitive mode, we introduce a comprehensive benchmark designed to systematically evaluate the visual symbolic capabilities of MLLMs in discrete semantic spaces. Unlike prior benchmarks that primarily emphasize natural image understanding or open-ended visual question answering, our framework focuses on structured, abstract, and highly symbolic visual representations that explicitly encode meaning. Importantly, these symbols cannot be trivially recognized through low-level visual capabilities (e.g., OCR). Drawing on insights from human visual neuroscience and cognitive psychology, our framework aligns with these cognitive stages and spans five distinct symbolic domains that mirror the evolution of human knowledge: Language (e.g., handwritten and faked Chinese characters), Culture (e.g., emojis and idioms), Mathematics (e.g., function graphs and geometry), Physics (e.g., circuit diagrams and mechanics), and Chemistry (e.g., molecular structures). To rigorously assess the depth of symbolic understanding, we structure our benchmark across a three-level cognitive hierarchy inspired by Bloom’s taxonomy and semiotic theory41,42, as shown in Figure 7. The first level assesses recognition and perception, evaluating whether models can reliably identify basic symbolic primitives such as

[Figure 16]

###### Figure 7. Overview of the benchmark task design framework and illustrative examples. (a) The instantiation of the three-level symbolic understanding hierarchy across five distinct domains: Language, Cultural, Mathematical, Physical, and Chemical symbols. (b) Representative examples of the tasks designed for our benchmark.

handwritten characters, schematic elements in function plots, molecular components, or physical diagram symbols. The second level targets compositional reasoning, where symbols must be integrated and interpreted according to domain knowledge, such

- as inferring functional properties from graphs or analyzing force interactions in mechanics. The third level probes associative and critical cognition, requiring models to detect inconsistencies, correct malformed symbols, and interpret non-literal or context-dependent meanings. We collect large-scale raw data from existing public datasets and a large volume of handwritten symbol data from human annotation experts. Based on a strong base of MLLMs, we perform domain classification annotation and question generation, resulting in 38 different sub-tasks and 13k question-image-answer pairs. After strict automated quality verification and manual validation, we obtain a complete evaluation dataset with a corresponding evaluation suite.

We analyze domain performance, cognitive difficulty, and inter-domain correlations (Figure 8). Most models exhibit unbalanced symbolic understanding, with proprietary models showing broader coverage across all domains than open-source counterparts. A critical finding is that language symbols represent the most challenging domain for all tested MLLMs. In contrast, models perform significantly better on natural science symbols, particularly in mathematics and chemistry, suggesting current architectures are more proficient at processing structured molecular and mathematical notations than identifying nuanced anomalies in linguistic characters. We investigate performance across a three-level hierarchy comprising Level 1 (perception), Level 2 (reasoning), and Level 3 (critical thinking). Figure 8 shows a non-linear pattern where average Level 2 scores are frequently higher than or comparable to Level 1 across many models. This counterintuitive “recognition-reasoning inversion” suggests that MLLMs rely on robust linguistic and structural priors to infer compositional meanings even when fine-grained visual perception of individual symbols is imperfect. To explore mutual influences, we analyze the correlation between social science symbols (language and culture) and natural science symbols (mathematics, physics, and chemistry). A strong positive correlation exists within the natural sciences; models excelling in mathematics typically demonstrate superior performance in other formalized, rule-based scientific fields. Conversely, the relationship between language and cultural symbols appears more fragmented. While top-tier models lead in both areas, others exhibit specialized capabilities in specific pockets. This divergence indicates that cultural understanding requires a distinct set of semantic knowledge that does not fully overlap with pure linguistic parsing, reflecting the unique difficulty of interpreting non-formalized symbols in discrete spaces.

These extensive evaluations across state-of-the-art MLLMs of varying scales reveal several findings. We observe a counterintuitive recognition–reasoning inversion: models often perform better on higher-level reasoning tasks than on foundational perceptual recognition tasks. This suggests that current MLLMs frequently bypass robust visual symbol grounding, instead relying on linguistic priors or memorized patterns. In several domains, particularly chemistry and mathematics, models exhibit procedural imitation, successfully reproducing solution patterns without a genuine understanding of the underlying symbols. Moreover, strong language reasoning capabilities can partially compensate for deficient visual perception, thereby masking perceptual failures through contextual inference. Finally, no single model demonstrates consistent performance across symbolic domains, indicating that current strengths remain largely domain-dependent and data-driven rather than systematic.

We provide the broader benchmark-level summary and the inter-domain comparison in Figure 8 in Supplementary Sec. B, where the imbalance between domains and the recognition-reasoning inversion can be seen more globally.

Taken together, these findings expose a fundamental cognitive mismatch in contemporary MLLMs and underscore the necessity for benchmarks that explicitly disentangle perception, reasoning, and critical symbolic understanding. Thinking further, the observed limitations are rooted in the fundamental divergence between the continuous representational bias of current visual encoders (e.g., CLIP-based ViTs) and the compositional rigor required by discrete semiotics. While MLLMs excel at directing visual signals to high-level linguistic concepts, this process often bypasses the intermediate structural parsing essential for symbolic semiosis. Unlike natural images, where semantic “gist” is preserved through spatial redundancy, discrete symbols exhibit high information density where a single stroke deletion (e.g., in faked characters or chemical bonds) triggers a total semantic shift. This “Cognitive Mismatch” suggests that current architectures lack a structural bottleneck capable of preserving the topological integrity of symbols, representing a foundational barrier to achieving Human-aligned Artificial General Intelligence. The main contributions of this paper are summarized as follows:

- • A symbolic perspective on MLLM evaluation: We introduce the first framework dedicated to assessing MLLMs in discrete semantic spaces, shifting the focus from continuous perception to structured symbolic interpretation.
- • A hierarchical, multi-domain benchmark: We construct a large-scale, high-quality benchmark spanning five symbolic domains and three cognitive levels, enabling fine-grained diagnosis of model capabilities.
- • Insights into fundamental cognitive limitations: Our comprehensive analysis reveals systematic deficiencies in finegrained visual symbol grounding and highlights the persistent reliance of current MLLMs on heuristic linguistic shortcuts, offering valuable new directions for advancing embodied and symbolic intelligence.

[Figure 17]

###### Figure 8. Cross-domain summary of benchmark results. (a) Radar charts illustrate the fine-grained performance of models across General, Language, Culture, Math, Physics and Chemistry domains. (b) Global performance aggregated by difficulty, with accuracy averaged across five symbolic domains. (c) Scatter plots exploring interrelationships between domains.

###### A.1 Related Work

- A.1.1 General Benchmarks The evaluation landscape for Multimodal Large Language Models (MLLMs) has evolved into a multifaceted ecosystem, transitioning from foundational general capabilities to complex cognitive and interactive intelligence. Comprehensive benchmarks43–46 typically employ meticulously crafted multiple-choice or open-ended questions to assess dimensions such as vision–language understanding, world knowledge, and multi-step reasoning. Complementing these, fine-grained perception benchmarks19,21,22,47–53 evaluate not only object and scene recognition but also the ability to infer semantic relations, action intentions, and underlying logical connections. Visual grounding tasks54–56 further assess precise localization of target regions based on textual descriptions, while hallucination and safety suites57–60 aim to quantify faithfulness and mitigate ungrounded generation. To probe advanced intelligence, researchers have introduced challenges in abstract reasoning61–64, code synthesis65–68, and long-context processing69–72. More recently, the frontier has shifted toward dynamic and interactive capabilities, incorporating video understanding benchmarks73–79 to assess the comprehension of temporal information, alongside autonomous decision-making in agentic GUI environments80–85. Despite this expansive breadth, existing benchmarks primarily focus on naturalistic scenes, often overlooking the structured, abstract symbolic systems that underpin human civilization.
- A.1.2 Symbolic Benchmarks Semiotics is the study of how symbols carry and convey meaning86–89. In semiotic theory, a sign is not the object itself but consists of two components: the signifier, referring to the form of the symbol, and the signified, denoting the concept or meaning it represents90,91. In the social sciences, evaluation has moved from modern OCR25,92,93 to deciphering complex ancient scripts like Oracle Bone Inscriptions94, Egyptian Hieroglyphs95, and Ancient Yi96. Cultural assessment has transitioned from emoji-based sentiment analysis97,98 to sophisticated semantic generation99, geo-diverse VQA100, and interactive art critique101. Recent work like WildScore102 further probes the structural reasoning of musical scores. In the natural sciences, mathematical benchmarks have evolved from static formula parsing26,27,103 to dynamic program-based synthesis104 and fine-grained error correction105. Physics evaluation now encompasses circuit analysis106, grounded reasoning107,108, and university-level problem solving that resists textual shortcuts109,110. Similarly, chemistry suites focus on table structure extraction111, versatile real-world scenarios112, and molecular elucidation via spectral data113. Despite this proliferation of datasets, most existing work evaluates reasoning in a terminal fashion, focusing on the final answer. Our work addresses this critical gap by mirroring a human-like cognitive progression, providing a diagnostic hierarchy from discrete symbol identification to compositional logic and emergent semantic inference across five foundational domains.
- A.1.3 Architectures for Symbolic Domain In recent years, Multimodal Large Language Models (MLLMs)114–117 have developed rapidly. Early models such as CLIP118 and ALIGN119 laid the foundation through large-scale image-text contrastive learning, followed by BLIP-2120 and the LLaVA series121–124, which further advanced the performance of MLLMs in image understanding, visual question answering, and open-domain dialogue. More recent studies have shifted their focus to architectural efficiency and native multimodal integration. Key innovations include M-ROPE for temporal-spatial alignment125,126, Cascade Reinforcement Learning for scientific reasoning127, and unified understanding-generation architectures128,129. To handle symbolic data, specialized paradigms have emerged. For text-intensive perception, models utilize window attention28,29 and layout-compressed query embeddings30. In cultural reasoning, approaches like NotaGPT31 align 2D symbols with text sequences, while ArtCoT32 and ArtSeek33 apply evidence-based Chain-of-Thought (CoT) to minimize hallucinations. For scientific symbols, methodologies emphasize structural rigor through geometric element alignment34,35, symbolic verification mechanisms36,37, and external simulator integration130,131. Molecular modeling has similarly shifted from string translation132 to discrete token-level fusion38 and high-resolution image compression39,40. However, these approaches remain fragmented across specific domains; our benchmark provides a unified framework to drive the development of models capable of integrated, multi-level symbolic reasoning.

B Extended Results

###### B.1 Language Symbols

- B.1.1 Weak Recognition Ability for Faked Characters

- In task 1 (faked character detection), the overall performance of most models was extremely poor, with F1 scores universally below 2. Only Gemini-2.5-pro, o3, and GPT-4o achieved slightly higher results. In contrast, most open-source models performed particularly poorly; LLaMA3-llava-next-8b, for instance, often defaulted to outputting a templated “cannot analyze the image” prompt without attempting to identify or correct the faked characters. This phenomenon reflects a deficiency in the underlying visual encoding capability of current MLLMs for sparse character structures, especially in abnormal cases involving missing strokes or faint handwriting, where they fail to establish a stable character-space representation.

Qualitative analysis reveals two primary failure modes. First, some models did not recognize the faked characters as errors but instead automatically replaced them with the most similar legal glyphs in their outputs, as seen with the characters “推”

(push) and “荐” (recommend) in Case 1 of Figure 2. This demonstrates a typical forced normalization behavior, whereby the model repairs anomalous strokes at the visual stage into a symbol that can be mapped to its linguistic vocabulary, thereby erasing the anomalous features at the perceptual level. Second, while some models could follow the instruction to mark errors, they lacked precise symbol discrimination ability, often mistaking normal characters for anomalous ones and thus producing incorrect localizations and redundant annotations. For example, in Case 1 in Supplementary Sec. E, a model misidentified the correct character “违” (violate) as a faked character. This behavior shows an inability to distinguish between poorly written yet correct characters and structurally incorrect faked characters, leading to an imbalanced detection result characterized by a high X_count_pred but a low F1 score.

- B.1.2 Insufficient Recognition of Misused Characters

- In task 2 (contextual character misuse identification), models must not only recognize individual characters but also integrate visual recognition with contextual semantics to identify word or sentence-level misspellings. The results show that while Gemini-2.5-pro and o3 maintained their lead, the F1 scores of GPT-4o and the Qwen series were clustered at the low level of around 5. This indicates that current models still struggle to integrate character recognition with syntax and semantics into a coherent comprehension structure when faced with semantically dependent symbolic tasks. Notably, both Deepseek-vl2-tiny and LLaMA3-llava-next-8b scored 0, but for different reasons. LLaMA3-llava-next-8b failed to correctly perceive the text in the image, often hallucinating entirely irrelevant characters and sentences and attempting to analyze this fabricated content. In contrast, Deepseek-vl2-tiny, as shown in Case 2 in Supplementary Sec. E, could accurately recognize the entire sentence via OCR but treated it as an indivisible whole, inserting it directly into the JSON output, thus demonstrating a lack of a character-level decomposition and comparison mechanism.

Interestingly, the 7B parameter Qwen2.5-vl achieved a score comparable to the much larger GPT-4o and close to its sibling Qwen-max. This suggests that in such highly specific symbolic tasks, performance is not solely dependent on model scale but may be influenced by the training data composition. The inclusion of fine-grained textual and handwritten document data during the Qwen series’ training might explain its relative advantage on this task.

From a qualitative perspective, models exhibit shortcomings when required to combine semantic information to judge errors. In Case 2 of Figure 2, when the task involved distinguishing between the homophones “的” (of) and “地” (adverbial particle), models often visually identified “的” (of) as a legitimate character but could not determine whether its grammatical position was correct based on the context. In practice, “的” (of) typically precedes nouns, while “地” (adverbial particle) precedes verbs; however, the models failed to understand this grammatical dependency and instead randomly generated unrelated replacement characters (e.g., incorrectly swapping “室” (room) for “试” (try)).

B.1.3 Lack of a Mechanism for Maintaining Semantic Consistency

- In task 3 (visual-semantic character correction), under the exact match metric, only Gemini-2.5-pro and InternVL3-8B achieved double-digit scores, whereas LLaMA3-llava-next-8b scored zero, indicating it never once successfully completed a correction. The edit distance metric reveals the reason for this performance disparity. Gemini-2.5-pro and InternVL3-8B had a very low edit distance, suggesting that while their corrections were not perfectly accurate, they were directionally relevant, generating semantically similar characters. Some of their outputs even exhibited an “analysis-correction” chain-of-thought process. This indicates they have established a tighter synergistic mechanism between visual perception and semantic reasoning, allowing the models not only to “see” the error but also to understand and correct it accordingly.

In contrast, the high edit distance of models like Claude-sonnet-4, Deepseek-vl2-tiny, and Qwen2.5-vl indicates that their outputs deviated severely from the correct answers at both the character and semantic levels. These models often lack stable capabilities for error localization and controlled correction. For example, in Case 3 of Figure 2, a model misidentified an incorrect character as the visually similar character “烧” (burn), leading it to generate the completely unrelated phrase “烧烤板” (grill plate). This is a case where erroneous visual perception dominated semantic generation, leading to severe semantic drift.

The most extreme case was LLaMA3-llava-next-8b, whose edit distance reached an astonishing 179.7. This was not merely a failure of correction; rather, the model, unable to recognize the image, experienced severe hallucinations, generating a large volume of irrelevant text. It even repeatedly used templated excuses in its output, such as “due to the low image quality,” to directly admit its failure to recognize valid characters. Drawing on the results from Level 1 (perception and recognition) and Level 2 (combination and reasoning), it can be inferred that most models failed to correctly perform error localization in the earlier stages, which naturally precluded them from making targeted corrections in Level 3 (association and critical thinking).

###### B.2 Cultural Symbols

- B.2.1 Models perform well at the low level but still face challenges Compared to the two Chinese tasks, MLLMs performed better on English. As shown in Figure 3 (a) and (b), GPT-4o achieved impressive F1 scores of 55.8 and 35.2 on the word-level and sentence-level tasks for English word and idiom recognition, respectively. This is likely because the model encountered more similar English text during training, making it more adept

- at reasoning with English words. In Case 4 of Figure 3, the model accurately identified pine on the left and apple on the right, successfully integrated their semantics, and inferred the compound word pineapple, demonstrating strong capabilities in semantic composition and association.

However, MLLMs consistently struggle with hallucinations when decoding emojis as visual codes. As shown in Case 5 of Figure 3, the model recognized the smiling face and lightbulb emojis but ignored the critical semantic constraint of the third “prohibition” symbol. It proceeded to hallucinate the positive idiom “bright idea,” which is semantically opposite to the correct answer, “Not the brightest bulb.” This reveals that once the models capture the linguistic meaning of an individual emoji, they tend to immediately retrieve related words or idioms from their internal knowledge while ignoring the crucial context provided by the surrounding emojis.

- B.2.2 Models show limited performance in Chinese idiom tasks We evaluate four-character and multi-character idioms. MLLMs perform poorly: GPT-4o achieves accuracy scores of 3.3 and 5.0 on these tasks. However, even the strongest open source model Qwen2.5-VL lags behind GPT-4o; the differing strengths of the models remain apparent on our more challenging benchmark. In Case 6 of Figure 3, the model interpreted and reasoned about the first two emojis along the dimension of “color,” generating a four-character idiom that was unrelated to the latter two emojis. In reality, the key to this image lies in reasoning from object semantics and homophonic relationships—for example, recognizing that the “bucket” (桶, tˇong) shown in the image is a homophone for the character meaning “same” (同, tóng). Tasks of this nature require not only symbol recognition but also cross-modal divergent thinking and the ability to perform phonetic-semantic associative reasoning. Accuracy at the Chr-1 level is significantly higher, indicating that MLLMs can perform basic text translations corresponding to individual emojis, but they have limited visual intuitive semiosis capabilities to further infer the corresponding linguistic meanings based on the relevant emoji context, particularly for the reasoning.
- B.2.3 Semantic Similarity Analysis Reveals Random Guessing Patterns in Association and Critical Thinking We further compute the semantic similarity between the responses and the ground truth, applying LLM to score from 1 to 5. As shown in Figure 3 (c), the average scores are low, with the English task significantly higher than those on the Chinese task. When carefully observing the distribution, we observe that 1)for the Chinese task, most of the scores are concentrated in 1 and 2, indicating the poor performance of MLLMs; 2)while for the English task, most of the scores are concentrated in 1 and 5, demonstrating that the MLLMs can either predict the answer correctly, or get irrelevant answers.

In the Chinese multi-character idiom task of Case 7 of Figure 3, the model successfully recognized the first two emojis, “一” (one) and “日” (day), accurately capturing the initial semantic cue. However, it failed to establish a logical connection among the subsequent symbols. Notably, although there were a total of 8 character elements in the image, the model ultimately output only a 4-character idiom. This indicates that when MLLMs are faced with long-sequence, composite symbol combinations, their visual attention tends to focus on the beginning of the input, while the subsequent content is either ignored or overridden by early semantic hypotheses.

- B.3 Mathematical Symbols The overall experimental results reveal significant hierarchical disparities and contradictions in model performance across various task tiers. These disparities not only reflect the inherent difficulty of the tasks but also expose a deeper imbalance in how multimodal models develop their visual perception and language reasoning capabilities. Notably, most models perform worse on basic recognition tasks than on complex reasoning tasks, contradicting intuitive cognitive expectations.

- B.3.1 Visual perception ability Firstly, the fundamental difference in task design is one of the core reasons for this phenomenon. Level 1 (perception and recognition) heavily relies on precise visual localization and symbolic semantic mapping, with extremely high data sensitivity—any pixel-level deviation can lead to failure—and very low error tolerance, where answering incorrectly means failing completely. For example, in task 2 (function type classification), the model must accurately identify curves like ex as “exponential functions.” However, minor rendering differences such as jagged edges or axis compression often cause misclassification. Even advanced models like Qwen2.5 achieve only 37.3 points on this task. In contrast, Level 2 (combination and reasoning) and Level 3 (association and critical thinking) tasks focus more on multimodal logical reasoning and rule generalization, allowing partial compensation through linguistic logic. Take task 15 (geometric definition consistency check) as an example: even if the model fails to precisely locate corners or edges, it can still detect errors via logical rules such as “the sum of internal angles in a triangle is not equal to 180,” enabling Llama3 to score 66.7. These structural differences expose the visual limitations of models when facing Level 1 (perception and recognition) tasks.

In Case 8 of Figure 4, the model was asked to identify the type of function shown in the graph. Although its pixel perception abilities were limited, the model was able to perform reasoning by elimination using its existing knowledge base of function definitions. It first analyzed the characteristics of exponential and logarithmic functions, then, based on the curve in the image exhibiting an upward-opening, U-shaped trajectory, it ultimately inferred that the graph corresponds to a quadratic function.

- B.3.2 Model performance comparison Secondly, the adaptability of models significantly influences their performance across different levels of tasks. Smaller models like Deepseek-vl2-tiny excel in clearly defined reasoning tasks (e.g., task6 function monotonicity reasoning), as they tend to bypass complex visual details and instead rely on language pattern matching. Larger models like GPT-4o demonstrate stronger adaptability in open-ended error detection tasks (Level 3), compensating for visual localization errors through joint vision-language representations. However, this divergence also leads to anomalous cases. For instance, Qwen2.5 performs exceptionally well on geometric computation tasks (e.g., area calculation), likely due to its use of built-in geometric knowledge templates such as the Pythagorean theorem to circumvent visual shortcomings. Yet, it scores only 8.3 points on task 3 (geometric shape classification), revealing a severe lack of visual generalization capability. This indicates that while large models possess strong language reasoning abilities, their foundational visual encoding capabilities still require improvement.

As shown in Case 9 of Figure 4, this task could have been completed quickly through direct visual judgment. The function in the graph is a monotonically decreasing straight line, and the required range for x could have been obtained simply by reading the two intersection points with the coordinate axes. However, the model did not directly utilize this explicit visual information. Instead, it engaged in a three-step symbolic computation process: first identifying the coordinates of the intersection points with the x and y axes, then substituting these into the equation to solve for the slope k, and finally deriving the corresponding x range via algebraic operations. This indicates that even when provided with sufficient visual clues, the model still tends to bypass the image information and instead relies on linguistic logic to perform lengthy symbolic reasoning.

- B.3.3 Fine-grained task analysis Additionally, the misalignment in mainstream pre-training objectives contributes to poor performance on Level 1 (perception and recognition) tasks. Most current multimodal models are trained to encourage “jump mapping” from vision to language concepts rather than fine-grained visual localization. For example, VQA and captioning tasks emphasize generating coherent language descriptions rather than identifying every detail in images. This training approach leaves models ill-equipped to handle tasks requiring precise visual analysis. GPT-4o scores only 26.3 on task-4 (Geometric Element Attribution) but achieves 77.7 on task-14 (Function Definition Validation), indicating its strength lies in semantic-based plausibility judgment rather than visual element enumeration. In Case 10 of Figure 4, when asked to determine the number of zeros of the function, the model

directly read the printed text “y = log2(x)” from the image. It thereby bypassed the need to identify the function type from the curve’s shape, avoiding a more fine-grained visual task.

Ultimately, we arrive at an important conclusion: the weakness of current models on Level 1 (perception and recognition) tasks is not entirely due to insufficient capability but rather to their selective reliance on language reasoning mechanisms, which suppress the development of the visual modality. This phenomenon reveals a key contradiction in multimodal learning—when the language modality becomes too dominant, the growth potential of the visual modality is compressed, preventing the model from truly “understanding the world visually”. This further confirms the necessity of constructing high-quality multimodal symbolic datasets. Only through such datasets can we break the current “language-dominant, vision-passive” paradigm and push multimodal models toward higher-level cognitive abilities.

- B.4 Physical Symbols

- B.4.1 Limited recognition of sparse physical symbols across most models As shown in Figure 5, most models performed poorly on the Level 1 (perception and recognition) task for physics symbols, with mean accuracies below 30%. For example, GPT-4o achieved an accuracy of 14.1%, Qwen2.5-vl 16.9%, and LLaMA3-llavanext-8b was almost entirely non-functional, reaching 1.8%. This widespread low performance indicates that current MLLMs have a deficiency in their ability to translate visual symbols into expressions of physical quantities and formulaic structures.

It is evident that models are generally able to recognize and recite the textual descriptions of physical laws but fail to correctly perform the mathematical mapping at the symbolic level. For instance, in Case 11 of Figure 5, a model could correctly state the definition of Ohm’s law relevant to the problem but, during symbolization, incorrectly wrote the quadratic relationship between power and current as a linear proportional one, and misidentified the straight-line graph in option B as a parabola. Such errors reflect that the model’s understanding of formula structures remains at a linguistic level, lacking both geometric intuition for the functional relationships between physical quantities and an awareness of symbolic consistency.

Furthermore, in Case 3 in Supplementary Sec. E, although a model could identify the required principle of conservation of energy, it made a critical parameter error during substitution: the original problem stated H = 3mgk , but the model incorrectly wrote it as H = mgk . Although its reasoning path was logically sound for the most part, the deviation in the numerical stage led to an incorrect final conclusion. This indicates that when faced with a complex physics symbol system involving multiple steps and intertwined concepts, the stability and accuracy of its reasoning chain are easily disrupted.

The reasoning process of Deepseek-vl2-tiny was opaque and unstable. On some problems, such as Case 3 in Supplementary Sec. E, it was able to directly generate the correct answer without providing any intermediate reasoning, whereas on others (like Case 11 of Figure 5), it offered an incorrect explanation that “power increases linearly with current.” This inconsistency

suggests that its correct answers may originate from template retrieval or pattern matching of its training corpora, rather than from systematic deduction based on physical principles. In contrast, Gemini-2.5-pro and o3 demonstrated more stable performance. Notably, Gemini-2.5-pro achieved a 60.2% accuracy on the electrical symbol recognition task, showing that its visual encoder possesses higher parsing precision when handling formula symbols with low pixel density and partial overlaps.

- B.4.2 Improved performance through symbolic reasoning yet with marked divergence

- In the Level 2 (combination and reasoning) task, the score distribution among models follows a similar pattern to Level 1 (perception and recognition), with Gemini-2.5-pro and o3 still significantly ahead, while GPT-4o and the Qwen series generally hover around the low level of 15 points. As seen in Case 4 in Supplementary Sec. E, models are capable of listing numerous general formulas related to the motion of charged particles, indicating they possess a certain reserve of physics knowledge. However, when required to combine these formulas with specific conditions from the problem, such as the electric field width, the models often show a disconnect. They lack systematic task-planning ability and are unable to select the correct chain of formulas for simultaneous solving. As a result, the reasoning process remains at the level of formula stacking and superficial pattern matching, ultimately halting after listing several equations or producing a seemingly plausible yet physically incorrect answer through the model’s guessing mechanism. This phenomenon indicates that while current multimodal models may know which physical laws to apply on a knowledge level, they have a significant deficiency in dynamically integrating symbols, parameters, and spatial constraints, lacking the ability to establish a continuous logical pathway from visual input to formulaic deduction.

B.4.3 Global correction and prediction remain challenging for most models

- In the Level 3 (association and critical thinking) task, task 5 (mechanical diagram consistency correction) emerged as a common bottleneck for all models. With the exception of Gemini-2.5-pro, o3, and Claude-sonnet-4, which achieved relatively acceptable scores, the remaining models almost universally failed, with some even scoring zero. This result highlights a significant deficiency in the capabilities of current Multimodal Large Language Models for symbolic error correction and physical logic consistency reasoning within complex visual scenarios.

As shown in Case 12 of Figure 5, although the model is able to accurately identify multiple basic components in the circuit diagram, this recognition often relies on the semantic assistance of letter markings in the diagram, such as ’A’ for Ammeter and ’V’ for Voltmeter, rather than on a genuine visual analysis of the symbols’ morphology. However, the model still lacks the ability to integrate visual symbol semantics with the physical context. It mistakes the slider P of the sliding rheostat for the symbol P representing power, and proceeds to make incorrect logical deductions on this basis.

- B.5 Chemical Symbols To investigate the cognitive limits of MLLMs in the highly symbolic domain of chemistry, we designed and implemented a benchmark with three difficulty levels. This benchmark simulates the cognitive progression from basic chemical knowledge recognition to advanced chemical reasoning and error correction, encompassing the comprehension of core chemical notations, such as structural formulas and reaction equations.

- B.5.1 Most models show limited capability in basic chemical symbol recognition We can observe the overall results for the Level 1 (perception and recognition) tasks in Figure 6. Although most models exhibit some preliminary recognition capabilities at a basic level, their overall accuracy remains low. Gemini-2.5-pro was the most prominent performer, achieving accuracies of 46.1% and 26.7% in task 1 (element identification in structural diagrams) and task 2 (chemical bond recognition), respectively, with a mean score of 39.4% that was superior to other models. In contrast, mainstream multimodal models, including GPT-4o and Qwen2.5-vl, generally scored lower on task 2 than on task 1, with some even dropping to zero.

From a qualitative analysis, most models can identify explicitly rendered chemical symbols in an image, such as the clearly labeled N and O atoms in Case 13 of Figure 6. However, they universally ignore the implicit rules of chemical skeletal formulas, wherein each vertex and endpoint in the skeletal structure represents a carbon atom, and the number of attached hydrogen atoms must be inferred based on valency. Because the models fail to understand this, both C and H atoms are omitted, leading to incorrect parsing of the molecular structure. While some models can identify high-level structures like “benzene ring” and “nitro group,” demonstrating an ability to capture local features, their atom-counting process relies on memorized templates rather than actual structural analysis. For example, they directly combine the formulas for a “benzene ring” and a “nitro group,” mechanically adding them to get 6 hydrogen atoms, but fail to infer from the image that one hydrogen on the ring had been substituted by the nitro group; the correct result should be 5 hydrogen atoms.

In Case 14 of Figure 6, most models are able to identify single and double bonds, but their counts are significantly lower than in the actual structure. This issue reflects the difficulty current MLLM visual encoders have in processing sparse lines, intersecting connections, and structural abbreviations, making them unable to effectively distinguish the types and quantities of bonds. Furthermore, models fail to identify the aromatic ring structure present in the diagram, which is neither a pure single

nor a pure double bond. This indicates that current models lack the ability to jointly model symbolic hierarchy and chemical semantics, and thus still struggle to truly understand the chemical meaning behind the structural symbols.

- B.5.2 Rule-based symbolic integration drives reasoning yet exposes structural gaps When the task shifted from static symbol recognition to dynamic reasoning requiring the application of chemical laws, the performance gap among models widened. The results show that o3 performed best at Level 2 (combination and reasoning), followed by Claude-sonnet-4 and Gemini-2.5-pro. Notably, although Gemini-2.5-pro exhibited the strongest symbol recognition ability, its overall score in Level 2 (combination and reasoning) was lower than o3’s. This suggests that o3 possesses stronger capabilities for reasoning chain integration and chemical knowledge transfer; despite being slightly weaker in visual parsing, it holds an advantage in tasks that require quantitative balancing and reaction type judgment. In contrast, the performance of LLaMA3-llava-next-8b and Qwen-max lagged, reflecting their deficiencies in constructing stable reasoning chains and integrating information from both the symbol and rule levels.

From a qualitative perspective, in Case 15 of Figure 6, some models were able to correctly identify chemical symbols in an equation and detect coefficient anomalies, demonstrating a preliminary capacity for anomaly detection. However, their proposed corrections were often incorrect, indicating that the models rely primarily on pattern-matching reasoning—for example, simply identifying that “the equation needs balancing”—rather than a true understanding of the principle of atom conservation. More typically, even when models detected an imbalance, their reasoning chains became disordered during the balancing attempt. There was a lack of a causal link between adjusting formulas and verifying counts, leading them to cyclically output the same step or get stuck in non-convergent, repetitive calculations.

- B.5.3 Only a few models succeed in global chemical correction and prediction In the Level 3 (Association and Critical Thinking) tasks, models perform multi-level reasoning across complex chemical reaction diagrams and textual descriptions, including reaction type discrimination, product prediction, and condition analysis. Such tasks present a comprehensive challenge to a model’s capabilities in symbol understanding, rule transfer, and causal generation. Overall, Gemini-2.5-pro again performed best, achieving a high score of 86.7 in task 7 (reaction product prediction), which demonstrates its capacity to generate results grounded in chemical knowledge structures. Conversely, although o3 and Claude-sonnet-4 performed adequately on the Level 2 (Combination and Reasoning) reasoning tasks, they were significantly weaker on the multi-dimensional prediction tasks in Level 3 (Association and Critical Thinking). This is likely due to their weaker symbol recognition capability, which limits their capacity to integrate information globally. Interestingly, Qwen-max, which consistently lagged in Level 1 and Level 2, jumped to third place in the Level 3 average score. This suggests that its stronger global understanding may have compensated for its deficiencies in symbol parsing, enabling it to complete higher-level chemistry tasks in a leapfrogging manner. This could also be related to the inclusion of chemistry topics in its training set.

Case 5 in Supplementary Sec. E shows that some models can identify key reactants in a chemical equation, such as an alkyl halide and a lithium reagent, and proactively invoke internal knowledge during their analysis, such as “lithium reagent reactions typically need to be conducted at extremely low temperatures.” They can then use this knowledge to filter options and ultimately generate the correct answer. This indicates that models are beginning to exhibit a principle-based heuristic reasoning capability. However, this ability is not yet stable. Errors in some models often stem from early-stage symbol recognition deviations, such as misreading bond types or omitting substituents, which causes the entire subsequent logical chain to fail.

- B.6 Human Performance Baseline To further benchmark the cognitive alignment of Multimodal Large Language Models (MLLMs), we establish a human performance baseline to evaluate the processing of complex abstract visual symbols in real-world scenarios. By comparing human performance across diverse domains and hierarchical levels, we aimed to discern whether model reasoning stems from true visual perception or a reliance on linguistic priors.

We conduct a stratified sampling of 1,000 instances from our comprehensive dataset of over 13,000 samples, ensuring a 95% confidence level. The sample was proportionally distributed across the cognitive hierarchy: 253 samples for Level 1 (Perception and Recognition), 439 for Level 2 (Combination and Reasoning), and 308 for Level 3 (Association and Critical Thinking). Five professional volunteers, all holding Master’s degrees or higher and possessing high bilingual proficiency in Chinese and English, participated in the experiment. They were instructed to complete the tasks using the identical prompts provided to the MLLMs to ensure a rigorous and fair comparison. As shown in Figure 9, the experimental results from the human-baseline comparison demonstrate that human performance follows a canonical cognitive trajectory, where accuracy steadily declines as the hierarchical level and task complexity increase. This trend empirically validates our three-level framework as a true reflection of escalating cognitive demand. In stark contrast, even the most advanced MLLMs exhibit a profound performance gap at the foundational level. In the Language and Chemistry domains specifically, humans achieved near-ceiling performance in Level 1 tasks, confirming these as intuitive, perceptual baselines for the human visual system. Conversely, MLLMs frequently exhibit the "Recognition–Reasoning Inversion" phenomenon, where they perform significantly better on complex reasoning

###### Performance Comparison between Human and Gemini

L1_Human L1_Gemini L2_Human L2_Gemini L3_Human L3_Gemini

100

| |
|---|

| |
|---|

| |
|---|

| |
|---|

80

60

Score

40

20

0

Language Math Physics Chemistry Emoji

Figure 9. Human performance on the benchmark compared with the best-performing model Gemini.

tasks than on basic symbol recognition. This disparity confirms that current models do not adhere to human-like visual-cognitive logic. Instead, they appear to rely on learned linguistic probabilities and procedural imitation rather than robust visual grounding. Consequently, the models’ “visual understanding” is often governed by what they statistically expect to see rather than the actual visual input, a cognitive mismatch that inherently leads to biases and hallucinations.

###### B.7 Comprehensive analysis

- B.7.1 Models generally lack the ability to recognize complex symbols Contrary to the common intuition that recognition is simpler than reasoning, our results indicate that foundational Level 1 (perception and recognition) remains a significant weakness for nearly all models, a pattern illustrated by the representative failures in Figure 10. This phenomenon was particularly prominent for language, chemistry, and physics symbols, where almost all models, including top-tier ones like Gemini 2.5 Pro and GPT-4o, exhibited high error rates. This suggests that while the visual systems of current MLLMs excel at processing the holistic features of natural images, such as texture, shape, and semantics, they lack the capacity for systematic and compositional parsing when faced with formalized, abstract symbolic systems. In one representative comprehensive case, the model demonstrates good descriptive capabilities in its output. It identifies the large triangle, the red areas, and the intersecting line segments, and attempts a hierarchical count based on geometric intuition. However, it ultimately arrives at only 8 triangles, omitting several composite triangles formed by the combination of local units. It fails to treat adjacent basic shapes as a whole to perceive and construct new symbolic entities formed by the combination of these parts and superimposed upon the original figure.
- B.7.2 The models exhibit rote memorization rather than genuine understanding The success of models on certain Level 2 (combination and reasoning) procedural tasks stands in contrast to their failures in Level 1 (perception and recognition), suggesting that their success stems not from genuine understanding but from a form of procedural imitation. This phenomenon is most pronounced in the chemistry domain. For example, Qwen-max scored 0 in Level 1 yet achieved a certain score in Level 2. This discrepancy indicates that its success does not follow a logically coherent process of first recognizing, then understanding, and finally solving the problem. Instead, it is more likely relying on powerful pattern-matching capabilities, effectively “memorizing” the overall visual paradigms of numerous chemical equations in its training data. A similar phenomenon can be observed in mathematics, where the excellent performance of the Qwen series and Claude-sonnet-4 is related to the large amount of mathematical content included in their training corpora.

In another representative comprehensive case, although the model could correctly identify the reactants and products in the image, it chose to determine the reaction type by comparing whether the types and quantities of elements changed before and after the reaction. However, in a correctly balanced chemical equation, the types and quantities of elements are inherently conserved, rendering this process an invalid reasoning step. Ultimately, the model concluded that it was a “neutralization reaction” and could correctly provide the definition of one, but this definition was irrelevant to the given problem and could not be genuinely applied to the current reaction. In other words, the model was merely reciting memorized chemical knowledge at a semantic level but failed to apply it correctly to the specific context.

[Figure 18]

Figure 10. Case studies from the comprehensive analysis, highlighting model failures in recognizing and integrating symbols.

- B.7.3 Reasoning ability sometimes exists as a compensatory mechanism A model’s language reasoning ability can, to some extent, compensate for its deficiencies in visual perception. For instance, InternVL-2.6-8B performed better on the Level 3 (association and critical thinking) in the linguistic symbols domain than on the foundational tasks in Level 1 (perception and recognition) and Level 2 (combination and reasoning). Similarly, Gemini-2.5-pro’s performance on the comprehensive physics reasoning task surpassed its results in the symbol recognition stage. This suggests that the vision and language modules in these models are not yet deeply integrated, functioning more as complementary components. When visual recognition is inaccurate, the powerful language reasoning module often infers an answer based on contextual patterns or common logical paths, thereby masking or even compensating for the shortcomings of the visual system.

In a representative geometric reasoning task, the model did not explicitly identify which edges or angles were collinear from a visual standpoint. However, by leveraging its logical grasp of an “isosceles triangle” from the textual condition “AB = AC,” it automatically deduced that “∠ABC = ∠ACB” and proceeded to calculate the correct answer based on the knowledge chain that “adjacent angles of a parallelogram are supplementary.” In this process, the role of the image was weakened to that of scenario confirmation, while the model’s strong language reasoning ability bypassed the need for a precise perception of geometric relationships from the image.

Similarly, in a representative physics scenario, even if the model failed to recognize structural features like the track and object positions during the image analysis phase, it could still reconstruct the physical scenario from the text of the problem description. By identifying phrases such as “launch point height h = 0.45 m” and “B is at rest at the end of the horizontal section of the track,” it automatically invoked kinematics knowledge to deduce the time. This behavior indicates that the model’s reasoning relies more on the dispatch of linguistic knowledge and logical chains than on directly extracting quantitative information from the visual input.

- B.7.4 There is no model that excels in all domains None of the models exhibit robust and consistent performance across all five symbolic domains. As summarized in Figure 11, strong performance was typically concentrated in domains where pre-training data is likely more abundant and diverse, such as cultural symbols and mathematics. For instance, models like o3 and Gemini-2.5-pro achieved their highest scores on Level 1 (perception and recognition) tasks in the cultural symbols domain, likely benefiting from the vast corpora of related content in web data. Similarly, models with a focus on reasoning showed strong capabilities in mathematics, particularly on programmatic

[Figure 19]

[Figure 20]

Perception & Recognition Combination & Reasoning Association & Critical Thinking

CultureMathLanguagePhysicsChemistry

Perception & Recognition Combination & Reasoning Association & Critical Thinking

Perception & Recognition Combination & Reasoning Association & Critical Thinking

Perception & Recognition Combination & Reasoning Association & Critical Thinking

Perception & Recognition Combination & Reasoning Association & Critical Thinking

Figure 11. Overall performances of different models across the five symbolic domains.

##### a b

|Language C|Cultural Symbol| |Math Symbol|
|---|---|---|---|
|Level 1 (526) Faked Character Detection|Level 1 (843) Emoji Semantic Grounding|Func typ|Level 1 (680)<br><br>Geometric Shape,139 ction Keypoint, 114<br><br>Geometric Element Attribution, pes of function,74<br><br>170|
|Level 2 (488) Contextual Character<br><br>Misuse Identification|Level 2 (783) Emoji-Based Idiomatic<br><br>Reasoning|Fun<br><br>Global Functio|Level 2 (1811)<br><br>perimeter and area of ction Monotonicity<br><br>geometric figures, 573<br><br>Reasoning, 93<br><br>Geometric Relation, 212 Function Property,18<br><br>Solid Geometry, 141 n Domain and Value,63<br><br>Combined Mathematical, 711|
|Level 3 (824) Visual-Semantic Character Correction|Level 3 (1,549) Cross-Cultural Emoji-toIdiom Mapping (4-word & Multi-word)|Functio The fu comply|The geometric shape does not Level 3 (444)<br><br>comply with the definition. 113 n Plot Consistency, 45<br><br>The correct three views of the nction graph does not<br><br>solid shape. 118 with the definition., 61<br><br>High-level Critical, 107|
|Chemi<br><br>Level 1 (681) Types and numbers of elements in chemical<br><br>structural formulas 48|istry Symbol<br><br>f Key functional grou<br><br>chemical structural fo 1 200|ups in ormulas|Physical Symbol<br><br>Level 1 (658)<br><br>electrical symbol Mechanical symbol<br><br>recognition, 63 recognition 595|
|Level 2 (2000)<br><br>Types of reactions in chem equations 1000<br><br>Level 3 (993)<br><br>Molecular Structure Err Correction 191|ical Balancing coefficie chemical equations<br><br>Reaction Conditio or<br><br>Product & Yeld, 8|nts in 1000<br><br>on & 802|Level 2 (766)<br><br>analysis of circuit analysis of force conditions,<br><br>operating conditions, 701<br><br>65<br><br>Level 3 (291)<br><br>Electrical Diagram Error Mechanical Diagram<br><br>Correction，176 Consistency, 115|

mechanical （1411）

emoji express

idiom, word, emotion

Electrical （304）

Cultural （3175）

Physical symbol (1715)

symbol

(3175)

language symbol (1838)

Chemistry

Visual Semiosis13K

chemistry structure （1672）

symbol (3674)

Math symbol (2935)

misspelled

and faked

character checking （1838）

reflection equation

（2002）

geometry

structure （1821）

function image （1114）

Figure 12. Overview of the benchmark task design, including language symbols, cultural symbols, math symbols, chemistry symbols and physical symbols.

tasks. In stark contrast, these same top-performing models often exhibited a significant performance decline in domains less represented in pre-training data, such as chemistry and physics. This indicates that the strengths of current MLLMs are more associative and empirical, adept at leveraging massive corpora from common domains, rather than deductive and systematic.

- B.7.5 The model structure may be the culprit for insufficient understanding of discrete symbolic. Our analysis suggests that the failure of MLLMs in sparse symbol recognition (e.g., skeletal formulas and circuit diagrams) is not merely a consequence of data scarcity, but an inherent artifact of the Vision Transformer (ViT) architecture. Standard ViTs partition images into fixed-size patches, processed via global self-attention with coarse-grained positional encodings. This mechanism is optimized for capturing low-frequency semantic features in natural scenes but effectively acts as a spatial low-pass filter for discrete symbols. In sparse structures like chemical bonds (Case 14) or electrical junctions (Case 12), the precise coordinates and connectivity are often "blurred" or aliased within the attention maps, leading to the observed inability to maintain symbolic consistency. The model’s reliance on linguistic priors thus becomes a compensatory survival strategy for its deficient perceptual resolution at the token-grid level.

#### C Extended Methods

To evaluate the visual and intuitive understanding of complex symbols by multimodal large language models (MLLMs), we construct a comprehensive benchmark spanning multiple domains, including language, art, mathematics, physics, and chemistry. The symbols involved include incorrect or substituted characters, emojis, musical notations, function symbols, geometric shapes, mechanical and electrical symbols, as well as chemical structural diagrams. These symbols represent a broad array of conventional forms commonly encountered across various facets of human culture.

We propose a three-level symbolic understanding task framework to systematically assess MLLMs across different modalities. The tasks range from basic recognition of individual symbols to more sophisticated associative and inferential reasoning. Each domain-specific task set is designed to reflect this progression, allowing us to investigate how models respond to different types of symbols under increasing cognitive demands. We curated datasets from as many fields as possible, with new annotations tailored to our task objectives, and conducted in-depth evaluations and analyses of multiple state-of-the-art multimodal models. The full task-organization schema is shown in Figure 12.

- C.1 Task Design According to the hierarchy of human symbolic cognition, our benchmark defines three levels of symbolic understanding tasks.

At the first level, “Perception and Recognition”, the model is required to understand the symbolic semantics represented by individual basic symbols. For the understanding of a single complex symbol, it is necessary to first extract its visual features (e.g., lines, shapes, colors, etc.), learn the visual semantics of the visual input itself, and then convert it into a meaningful symbolic semantic unit. For example, seeing a “red light” can identify that it has the symbolic semantics of “danger”. After

continuous training through daily life and social culture, the human brain can directly convert visual symbolic input into corresponding semantics. We aim to observe the model’s recognition ability at this level, thereby exploring whether multimodal large language models can learn such human-like ability to directly understand the abstract textual semantics corresponding to complex visual symbols under the training of large-scale image-text data currently available.

At the second level, “Combination and Reasoning”, the model needs to further perform more global understanding and reasoning in the discrete semantic space of images. It is required to combine multiple symbols, understand the joint semantics expressed by these symbols in the discrete semantic space, and even conduct further reasoning. For example, at Level 1, it is only necessary to understand the meaning of individual electrical symbols in a circuit diagram, while at Level 2, it is necessary to combine multiple electrical symbols and their connecting wires to infer the connection mode and working state of the entire circuit. This necessitates the model to possess semantic knowledge of symbols and their interrelationships on the premise of symbol comprehension, and also to examine the model’s reasoning ability.

Finally, building on the reasoning ability for symbol combination, the third level focuses on high-level associative and critical thinking abilities, “Association and Critical Thinking”. At the first two levels, almost all symbols presented to the model are correct, and most of the information they convey is the inherent meaning of the symbols themselves. However, at the third level, the model is required to break through this limitation. When encountering incorrect symbols or symbols that convey unconventional symbolic semantics, it must perform in-depth semantic judgment and reasoning based on the context. Among

- them, associative ability is the support of critical thinking. The model first associates through the contextual context, based on knowledge reserves such as the conventional semantics of symbols, relevant cultural backgrounds, and usage patterns. It
- then engages in critical thinking based on this associative information to complete the prediction of the rationality of symbols, correction of errors, and interpretation of special semantics. Specifically, in real-world scenarios, humans construct critical cognition of symbols precisely through association based on contextual cues: on the one hand, when facing deviated or defective symbols (e.g., the typo “chioce”), the human brain employs an autonomous correction mechanism, which can automatically associate it with the correct symbol (“choice”) according to the textual or situational context in which it is located, without disrupting the understanding of the overall context. This process requires not only associative ability to guide the direction of correction but also critical thinking to verify the rationality of the correction. On the other hand, for symbol combinations with special usages (e.g., homophonic emoji expressions), it is necessary to first associate information such as the inherent meaning of symbols, their connection with the context, and homophonic conventions in social culture. Critical thinking is then used to overcome the interference of the inherent meaning of symbols and to interpret the unconventional semantics that differ from these inherent meanings. Therefore, this level primarily examines the model’s ability to accurately capture subtle differences in symbolic semantics, as well as the high-level semantic understanding and reasoning ability, guided by association and centered on critical thinking within contextual scenarios.

Overall, targeting the special discrete semantic visual information of visual complex symbols, we have constructed a multi-level benchmark covering individual symbol understanding, combined symbol reasoning, and high-level association and critical thinking based on the special understanding mechanism of the human brain. Corresponding tasks have been designed and datasets established in multiple fields to achieve comprehensive evaluation and exploration of the symbol understanding mechanism of multimodal large language models.

- C.1.1 Language Symbols Language constitutes one of the most fundamental symbolic systems in human cognition and serves as the primary modality processed by large language models. In this benchmark, we focus on Chinese characters as a representative linguistic symbol system with strong visual compositionality. Unlike alphabetic writing systems, Chinese characters encode semantic and phonetic information through structured visual components and strokes, making them an ideal testbed for evaluating visually grounded symbolic understanding.

At the first level, the benchmark evaluates the model’s ability to perceive and recognize invalid or non-existent characters, referred to as faked characters. These symbols arise from stroke-level or component-level perturbations and do not belong to any standardized character set. Since such characters cannot be resolved through lexicon lookup or OCR-based transcription, successful identification requires fine-grained visual perception and structural analysis of character morphology.

Building upon basic recognition, the second level focuses on contextual semantic reasoning over valid but incorrectly used characters. Misused characters are visually or phonetically similar to the intended ones and are grammatically valid in isolation, yet semantically incompatible with the surrounding context. This level examines whether the model can integrate visual recognition with linguistic context to detect semantic inconsistencies and identify the source of misuse.

At the third level, the benchmark targets cross-modal correction capability. The model is required to actively correct faked characters or misused characters by jointly leveraging visual structure, phonetic similarity, and contextual semantics. This level reflects a higher-order symbolic competence, assessing whether the model can construct stable mappings between visual perception and linguistic reasoning to perform autonomous error correction.

- C.1.2 Cultural Symbols With the proliferation of online communication and social media, emojis have evolved into a globally shared yet culturally nuanced symbolic system. Although emojis are standardized in appearance, their meanings are not fixed; they emerge through collective usage, cultural conventions, and contextual adaptation. As a result, emojis function not only as visual icons but also as culturally grounded symbols that convey emotions, intentions, and idiomatic meanings beyond their literal depiction.

At the first level, the benchmark evaluates the model’s ability to ground individual emojis in their commonly accepted semantic representations. This task examines whether the model can map visual emoji symbols to corresponding lexical meanings, reflecting basic visual-semantic alignment.

The second level advances to compositional reasoning, where sequences of emojis jointly express idiomatic meanings in English. In these tasks, emojis act as substitutes for words or phrases, requiring the model to infer sentence-level semantics from their symbolic composition. By restricting target interpretations to widely recognized English expressions, this level emphasizes symbolic combination and syntactic reasoning while minimizing ambiguity.

The third level introduces cross-cultural and cross-linguistic symbolic reasoning. Given an emoji sequence, the model must infer its corresponding Chinese idiom, which often relies on phonetic resemblance, metaphorical association, or culturally specific conventions. This level explicitly examines the model’s ability to transcend surface visual semantics and capture culturally mediated meanings, highlighting emojis as a form of cross-cultural symbolic language shaped by shared practices.

- C.1.3 Mathematical Symbols Mathematical reasoning represents a high-level form of symbolic cognition, characterized by abstraction, formal structure, and strict logical dependencies. In human cognition, mathematical understanding is not limited to linear symbolic expressions but heavily relies on visual representations such as function graphs and geometric diagrams, which provide spatial intuition

for abstract concepts. In contrast to prior benchmarks that focus primarily on text-based formulas encoded in LATEX, this benchmark targets visually grounded mathematical symbols and evaluates whether models can establish stable semantic and logical relationships directly from images.

To reflect the internal structure of visual mathematics, our tasks are organized around two major sub-fields: function graphs and geometric figures. Across both sub-fields, we design a three-level hierarchy that progressively evaluates perception, reasoning, and critical verification.

At the first level (Perception and Recognition), the model is required to identify fundamental mathematical entities and structural components from images. In the function sub-field, this includes recognizing key point sets (e.g., zeros, extrema, and inflection points) and classifying the functional type (such as polynomial, exponential, or trigonometric). In the geometric subfield, tasks involve identifying the class of planar figures and attributing special geometric elements, including notable lines and angles. This level assesses whether the model can reliably extract symbolic primitives and local structures that serve as the basis for higher-level reasoning.

At the second level (Combination and Reasoning), the benchmark shifts from isolated symbol recognition to integrated reasoning over complete mathematical structures. For function graphs, the model must infer global properties such as monotonicity, parity, periodicity, domain, and range, as well as perform value-based reasoning at specific points. For geometry, tasks require quantitative computation (e.g., length, area, or volume), reasoning about congruence and similarity, and understanding three-dimensional configurations through net diagrams and spatial relationships. These tasks examine whether the model can align local visual symbols with global mathematical constraints and axioms.

At the third level (Association and Critical Thinking), the benchmark evaluates the model’s ability to verify, diagnose, and correct holistic mathematical representations. Tasks include detecting inconsistencies in function plots, judging whether a curve satisfies the formal definition of a function, validating whether geometric figures meet their claimed definitions, and identifying correct orthographic projections of solid objects. This level emphasizes structural consistency checking, error localization, and corrective reasoning, reflecting advanced mathematical understanding beyond direct computation.

- C.1.4 Chemical Symbols Chemical knowledge is encoded through a highly structured symbolic system that includes molecular structural formulas and chemical reaction equations. Structural diagrams represent atoms, bonds, and functional groups, while reaction equations describe transformation processes governed by conservation laws and energetic constraints. Together, these representations form a precise symbolic language that supports both static description and dynamic reasoning.

To reflect this internal structure, the chemical symbol tasks are organized into two sub-fields: molecular structural formulas and chemical reaction equations, evaluated across three hierarchical levels.

At the first level (Perception and Recognition), the benchmark focuses on visual parsing of molecular structures. The model must identify element types, count atomic occurrences, and recognize different bond types within structural formulas. This level assesses whether the model can translate pixel-level information into a structured symbolic representation that faithfully captures molecular composition.

[Figure 21]

Figure 13. The data construction pipeline of our benchmark.

At the second level (Combination and Reasoning), the tasks move beyond static structures to symbolic reasoning under chemical laws. The model is required to classify reaction types and balance chemical equations by enforcing element conservation. These tasks evaluate whether the model can reason over symbolic representations in accordance with fundamental chemical principles.

At the third level (Association and Critical Thinking), the benchmark examines advanced chemical reasoning and correction abilities. Tasks include detecting and correcting errors in molecular formulas or reaction equations, inferring missing reaction conditions such as temperature, predicting reaction products, and estimating reaction yields. This level requires the integration of symbolic reasoning with domain-specific chemical knowledge, reflecting a mature understanding of chemical processes.

- C.1.5 Physical Symbols Similar to mathematics, physics relies heavily on symbolic and graphical representations to support understanding and reasoning. Especially in core fields like mechanics and electromagnetism, visual symbolic systems such as schematics, force diagrams, and circuit diagrams are widely used to abstractly depict complex physical processes. These diagrams encode rich information within a compact two-dimensional space, including force directions, field distributions, and structural connection relationships. Accurately interpreting these symbolic diagrams requires not only precise visual perception but also multi-level logical reasoning in conjunction with physical laws. Whereas existing evaluations primarily focus on text-based physics problems, which mainly assess the ability to extract parameters from text and perform numerical calculations, our benchmark instead proposes a more foundational and challenging paradigm: it emphasizes the understanding of visual symbolic content and the modeling of its physical meaning. It examines whether models can extract physical laws from symbolic graphics and achieve the cross-modal leap from visual perception to theoretical reasoning. Accordingly, we organize the physical symbol tasks into two sub-fields: mechanics and electricity, each evaluated through a three-level hierarchical framework.

In Perception and Recognition, the benchmark evaluates whether the model can correctly parse fundamental physical symbols from diagrams. In mechanics, this involves recognizing force symbols, their directions, magnitudes, and points of application, as well as associating them with the corresponding objects. In electricity, tasks require identifying electronic components such as power sources, resistors, and wires, along with their connection patterns. This level focuses on the extraction of symbolic primitives and their local semantic roles.

Then, at Combination and Reasoning, the model must integrate multiple symbols and infer system-level physical behavior. For mechanics, this includes reasoning about net forces, equilibrium conditions, acceleration directions, and motion tendencies under given constraints. For electrical systems, the model must infer current directions and component states based on circuit topology and source polarity. These tasks assess whether the model can map symbolic compositions to the governing laws.

At the third level (Association and Critical Thinking), the benchmark targets high-order validation and correction capabilities. The model is required to detect and correct inconsistencies in mechanical or electrical diagrams, such as contradictory force

directions or invalid circuit connections. Additionally, it must perform comprehensive reasoning in coupled systems, predicting system stability, energy flow, or behavioral changes under modified conditions. This level emphasizes holistic consistency checking and physically grounded critical reasoning.

- C.2 Dataset Construction After establishing the task domains and definitions for our hierarchical benchmark on complex symbol understanding, we proceed with the construction of the dataset. As illustrated in Figure 13, the process begins with domain-specific data collection, followed by task-type and difficulty-level classification based on the design principles described in the previous section. Each data sample is annotated with a corresponding question-answer pair. Following the initial annotation, both automated and manual validation steps are performed to ensure data quality. Once the dataset distribution is statistically analyzed, we supplement additional samples where necessary to maintain a balanced representation across all task types.

- C.2.1 Data Sources To build our benchmark, we draw on a wide range of existing open-source multimodal benchmarks, selectively extracting data relevant to our task domains. For linguistic symbols, we utilize the VisualC3 dataset93, a high-quality collection of handwritten Chinese character errors and grammatical mistakes extracted from real-world student essays. This dataset provides authentic samples of miswritten and misused characters in handwritten form. For emoji symbols, data are sourced from eWe-bench99, which collects idioms and phrases expressed using emoji from internet communication. The data have undergone multiple rounds of automatic filtering and human verification to ensure both expression quality and representativeness.

For mathematical symbols, we extract function and geometry-related samples from several multimodal mathematics benchmarks, including MultiMath-300K133 and MathVista26. Chemistry-related data come from ChemBench-4K134, MiniCASIA-CSDB135, and annotated middle and high school chemistry materials, which provide molecular structural diagrams, chemical equations, and their associated prediction problems. Physical symbol data are obtained from multi-disciplinary benchmarks such as OlympiadBench136, MMMU-Pro22, and Gaokao-Bench137. From these, we extract physics-related tasks involving mechanics and electrical circuits, covering a range of difficulties from university entrance exams to higher education-level problems.

- C.2.2 Data Annotation During annotation, we perform filtering, classification, and task-specific labeling of the raw data. The process begins with a coarse filtering step to extract data samples relevant to our task design. For example, from over 300K mathematical examples, we select items involving function plots and geometric figures. When existing domain tags are available, we retain and reclassify the samples accordingly—for instance, categorizing chemical structure images from Mini-CASIA-CSDB into the chemical structure domain. For unlabeled data, we use keyword matching or large language models (LLMs) to identify relevant content. The keywords and prompts used for each domain are detailed in Supplementary Sec. C.

After preliminary classification, we align the samples with our task framework and re-annotate each with an appropriate question and answer. For some datasets, this process is relatively straightforward. In VisualC3, for instance, we reorganize the image-question-answer triplets according to our three task levels, assigning the corresponding task labels. For chemistry tasks, we convert SMILES strings into rendered molecule images, replace them with placeholders in the original text, and redesign the questions and answers to match our framework.

Mathematics tasks require more extensive re-annotation, as most existing data do not include fine-grained task-level labels. We first explain the 17 task types defined across three difficulty levels, provide concrete examples, and use LLMs to generate initial annotations for domain type (function, geometry, or other), task level (Level 1–3), and task type (Task 1–17). These annotations are subsequently verified and corrected by human experts. Physics tasks follow a similar annotation pipeline.

In cases where data could not be sourced from existing datasets, we recruited trained annotators to construct examples manually. For error detection and correction tasks in chemistry and physics, which require uncommon miswritten symbols, domain experts with bachelor’s degrees or higher were invited to hand-draw erroneous diagrams. Each annotator was provided with a correct reference image and instructed to introduce intentional symbolic mistakes while recording the error types.

For chemical structural symbols, error types include atom omission (EA), extra atoms (WB), bond type errors (CHG), charge errors (CHG), and stereochemistry errors (SC). For chemical equations, we identify element imbalance (UB), incorrect conditions (WC), invalid reaction arrows (WA), incorrect states (MS), and catalyst annotation errors (CAT). For mechanical diagrams, we define force direction errors (FD), missing forces (FM), extra forces (FE), and incorrect analysis objects (DA). For electrical circuit diagrams, the error types include incorrect component symbols (CE), short circuits (SP), open circuits (OL), labeling errors (CD), and misconnected meters (MP). To maintain balanced datasets, annotators were instructed to produce an approximately equal number of examples for each error category.

Table 1. Distribution of tasks across domains, sub-fields, and hierarchical levels.

Field Sub-field Task ID Task Name Level Instances Level Total

- Character T1 Faked Character Detection Perception and Recognition 526 526
- Character T2 Contextual Character Misuse Identification Combination and Reasoning 488 488
- Character T3 Visual-Semantic Character Correction Association and Critical Thinking 824 824

Language

- Emoji T1 Emoji Semantic Grounding Perception and Recognition 843 843
- Emoji T2 Emoji-Based Idiomatic Reasoning Combination and Reasoning 783 783
- Emoji T3 Cross-Cultural Emoji-to-Idiom Mapping (4-word) Association and Critical Thinking 1,360 1,549
- Emoji T4 Cross-Cultural Emoji-to-Idiom Mapping (Multi-word) Association and Critical Thinking 189

Culture

- Function T1 Function Keypoint Recognition Perception and Recognition 114 680
- Function T2 Function Type Classification Perception and Recognition 74

- Geometry T3 Geometric Shape Classification Perception and Recognition 139
- Geometry T4 Geometric Element Attribution Perception and Recognition 170 General T5 Basic Mathematical Symbol Recognition Perception and Recognition 183

- Function T6 Function Monotonicity Reasoning Combination and Reasoning 93

1811

- Function T7 Global Function Property Inference Combination and Reasoning 18
- Function T8 Function Domain and Value Reasoning Combination and Reasoning 63

Mathematics

- Geometry T9 Geometric Quantitative Reasoning Combination and Reasoning 573
- Geometry T10 Geometric Relation Inference Combination and Reasoning 212
- Geometry T11 Solid Geometry Structure Reasoning Combination and Reasoning 141 General T12 Combined Mathematical Reasoning Combination and Reasoning 711

- Function T13 Function Plot Consistency Verification Association and Critical Thinking 45 444
- Function T14 Function Definition Validation Association and Critical Thinking 61

- Geometry T15 Geometric Definition Consistency Check Association and Critical Thinking 113
- Geometry T16 Orthographic Projection Identification Association and Critical Thinking 118 General T17 High-level Mathematical Analysis Association and Critical Thinking 107

Mechanics T1 Mechanical Symbol Recognition Perception and Recognition 595

658

Electricity T2 Electrical Component Recognition Perception and Recognition 63 Mechanics T3 Force–Motion Reasoning Combination and Reasoning 701

Physics

766

Electricity T4 Circuit Operation Reasoning Combination and Reasoning 65 Mechanics T5 Mechanical Diagram Consistency Correction Association and Critical Thinking 115

291 Electricity T6 Electrical Diagram Error Correction Association and Critical Thinking 176

- Structure T1 Element Identification in Structural Diagrams Perception and Recognition 481 681
- Structure T2 Chemical Bond Recognition Perception and Recognition 200

- Reaction T3 Reaction Type Classification Combination and Reasoning 1,000 2,000
- Reaction T4 Chemical Equation Balancing Combination and Reasoning 1,000 Structure T5 Molecular Structure Error Correction Association and Critical Thinking 191

Chemistry

- Reaction T6 Reaction Condition Inference Association and Critical Thinking 300
- Reaction T7 Reaction Product Prediction Association and Critical Thinking 202
- Reaction T8 Reaction Yield Estimation Association and Critical Thinking 300

993

- C.2.3 Data Validation To ensure the high quality of the benchmark, we implemented a two-stage validation process combining automated and manual review. Automated validation is used to detect duplicate or missing entries and to verify the integrity and readability of image files. Manual validation is more nuanced and domain-specific. For linguistic and emoji symbol tasks, human experts evaluate whether the expressions align with natural usage and filter out samples containing nonstandard or inappropriate content. This is especially important for emoji data sourced from online platforms, where we remove samples involving discrimination, violence, or profanity to ensure ethical and safe data use.

For mathematics, physics, and chemistry tasks, human reviewers confirm the completeness and correctness of problem statements and answers. In mathematics tasks, particular attention is given to verifying task type and difficulty assignments. For chemistry and physics error-detection tasks, reviewers inspect whether annotated error types accurately describe the errors depicted in the image. After this validation phase, we retained approximately 96.7% of the data, demonstrating the effectiveness of our annotation protocol and the overall reliability of the constructed dataset.

- C.2.4 Data Statistics and Analysis Before finalizing the dataset, we conducted a detailed statistical analysis to examine the sample distribution across task types and ensure that no subtask was severely underrepresented. For subtasks with insufficient data, we revisited the data collection and annotation pipeline to augment them. This involved identifying additional data sources or applying self-instruct methods, in which large language models are guided to generate new samples based on example images, task descriptions, and annotated templates. Through this iterative augmentation strategy, we ensured a balanced and comprehensive benchmark.

The final benchmark dataset consists of 13,148 samples, covering five symbolic domains and three hierarchical cognitive

[Figure 22]

Figure 14. Dataset distribution. (a) Overall data distribution across five domains. (b) Detailed task distribution specifically for Mathematical symbols.

levels. Table 1 provides an overview of the dataset composition across domains, levels, and sub-fields. From a hierarchical perspective, the dataset is relatively balanced across the three cognitive levels, with 3,388 samples at the Perception and Recognition level, 5,848 samples at the Combination and Reasoning level, and 3,912 samples at the Association and Critical Thinking level. This distribution reflects our design intent: while higher-level reasoning tasks are essential for evaluating symbolic intelligence, they are grounded in a substantial volume of lower-level perceptual and compositional tasks.

Across domains, the dataset spans a diverse range of symbolic systems, including Language (1,838 samples), Culture (2,986 samples), Mathematics (2,935 samples), Physics (1,715 samples), and Chemistry (3,674 samples). Each domain is internally structured according to its own symbolic characteristics and sub-fields, while adhering to the same three-level cognitive hierarchy.

In the Language Symbols domain, samples are distributed across all three levels, progressing from visual recognition of faked characters, to contextual identification of character misuse, and finally to visual-semantic character correction. Similarly, the Cultural Symbols domain centers on emojis as culturally grounded symbols, ranging from individual emoji semantic grounding to cross-cultural emoji-to-idiom mapping at the highest level. The Mathematical Symbols domain is organized around two sub-fields—function graphs and geometric figures—and exhibits a clear hierarchical progression. Lower-level tasks emphasize the recognition of fundamental visual elements, mid-level tasks focus on property inference and quantitative reasoning, and higher-level tasks target structural verification and error diagnosis in mathematical representations. For the Physical Symbols domain, tasks are divided into mechanics and electricity. The dataset moves from identifying core symbolic components in diagrams, to reasoning about system behavior under physical laws, and finally to detecting and correcting inconsistencies in coupled mechanical and electrical systems. Finally, the Chemical Symbols domain covers both molecular structural formulas and chemical reaction equations. Its task design reflects the transition from static visual parsing of molecular structures, through conservation-based reasoning in reactions, to high-order correction and prediction tasks that integrate symbolic reasoning with chemical domain knowledge. Detailed numerical statistics for each domain, sub-field, and cognitive level are summarized in Table 1, while additional visual breakdowns are provided in Figure 14.

###### C.3 Experiments

- C.3.1 Baselines To comprehensively evaluate the proposed benchmark, this study selected a series of representative Multimodal Large Language Models (MLLMs) as baselines, covering both open-source and closed-source models of various parameter scales. Specifically, the open-source models include DeepSeek-VL2-Tiny (3B)138, Qwen2.5-VL (7B)125, LLaMA3-LLaVA-Next-8B139, and InternVL3-8B140, which represent recent advances in visual–language alignment and instruction tuning for image–text understanding. For the closed-source commercial models, we evaluated GPT-4o141, Claude-Sonnet-4142, Qwen-Max143, o3-2025-0416144, and Gemini-2.5-Pro145, all of which are leading proprietary MLLMs with strong cross-modal reasoning and generation capabilities. The detailed descriptions of each model are provided below:

- • DeepSeek-VL2-Tiny (3B)138: A lightweight, open-source multimodal model developed by the DeepSeek team. It employs an efficient vision-language alignment strategy and demonstrates excellent performance on text-rich image

- understanding and fine-grained visual question answering tasks.
- • Qwen2.5-VL (7B)125: An open-source multimodal model from Alibaba’s Qwen series, featuring powerful visual understanding and cross-modal reasoning capabilities. The model supports high-resolution image input and excels in text generation, OCR scenarios, and complex image-text reasoning tasks.
- • LLaMA3-LLaVA-Next-8B139: Based on Meta’s LLaMA3 language model and combined with the LLaVA-Next framework for Visual Instruction Tuning. This model is capable of handling diverse image-text tasks, including visual question answering, image description, and multi-turn reasoning, demonstrating powerful open-ended multimodal understanding.
- • InternVL3-8B140: Adopts a Unified Multimodal Encoder architecture. Through multi-image scene training and multistage pre-training strategies, it exhibits excellent performance in complex visual tasks such as table understanding and image retrieval.
- • GPT-4o141: A natively multimodal large model from OpenAI, capable of processing text, image, audio, and video inputs within a single architecture. The model achieves high-precision cross-modal understanding and generation, showing exceptional performance in visual reasoning, contextual understanding, and real-time voice interaction.
- • Claude-Sonnet-4142: A high-performance multimodal model from Anthropic that demonstrates robust performance in complex text reasoning and image understanding. Through an optimized architecture, it achieves high-quality generation under high-throughput conditions and represents a balance between intelligence and stability.
- • Qwen-Max143: A closed-source multimodal model from Alibaba that leverages a Mixture-of-Experts architecture for improved scalability and efficiency. It employs Supervised Fine-Tuning (SFT) to enhance task-specific performance and is further aligned with user preferences via Reinforcement Learning from Human Feedback (RLHF), enabling the model to better understand and meet user needs.
- • o3-2025-0416144: A reasoning model from OpenAI. The core feature of this model is its significantly enhanced capability for complex multi-step reasoning and its logical rigor, and it demonstrates high reliability on difficult instructions.
- • Gemini-2.5-Pro145: A highly-capable, natively multimodal reasoning model from Google. It can comprehend vast datasets and challenging problems across diverse information sources, including text, audio, images, video, and even entire code repositories. It ranks among the top-performing commercial MLLMs currently available.

- C.3.2 Metrics To systematically evaluate the performance of MLLMs on complex symbolic understanding tasks, we designed a hierarchical evaluation metric system covering five distinct symbolic domains, including language, cultural, mathematical, physical, and chemical symbols, across three levels of task difficulty.

For language symbols, we adopt different metrics across three difficulty levels. At Level 1, for faked character detection tasks, we employ the F1-score and X_count_pred, where F1 measures the harmonic balance between precision and recall in detecting faked characters, and X_count_pred denotes the ratio of correctly predicted faked characters to the total number of faked characters. Formally,

TP TP+FP

TP TP+FN

2×Precision×Recall Precision+Recall

, (1) where TP, FP, and FN denote the number of true positives, false positives, and false negatives.

Precision =

, Recall =

, F1 =

The proportion of correctly identified faked characters is defined as:

Xcount_pred =

Nfaked_correct Nfaked_total

, (2)

where Nfaked_correct refers to the number of correctly detected faked characters.

At Level 2, for misspelled character detection tasks, we employ F1-score and id_count_pred, where id_count_pred denotes the proportion of correctly identified misspelled characters relative to the total number of such errors:

Nmisspelled_correct Nmisspelled_total

idcount_pred =

, (3)

where Nmisspelled_correct refers to the number of correctly detected misspelled characters.

At Level 3, for character correction tasks, evaluation is based on exact_match and edit_distance. Exact_match measures the proportion of perfectly corrected outputs among all predictions:

exact_match =

Nexact_match Ntotal

, (4)

where Nexact_matchis the number of outputs that are entirely identical to the ground truth. In addition, edit_distance quantifies the overall deviation between the predicted and reference sequences, serving as a measure of how far the model’s generation diverges from the correct symbolic form. To facilitate comparison across sequences of varying lengths, we report the normalized edit distance, defined as:

Ntotal

1 Ntotal

edit_distance(yi,yˆi) max |yi|,|yˆi|

### ∑

, (5)

NormEditDist =

i=1

where yi and yˆi are the reference and predicted sequences for the i-th example, respectively. This normalization ensures the metric remains bounded between 0 and 1, with lower values indicating better alignment with the ground truth.

For cultural symbols, we design three hierarchical evaluation settings corresponding to different levels of abstraction, each intended to capture a different facet of multimodal understanding ability. At Level 1, we compute Precision, Recall, and F1-score by comparing predicted words or characters against those in the ground truth. At this level, the task primarily reflects an MLLM’s ability to perceive and recognize the surface forms of symbols—when precision and recall are both high, it indicates that the model can correctly identify symbolic elements even before engaging in higher-level semantic reasoning.

At Level 2, evaluation is performed at both the sentence level and the word level. We calculate Precision, Recall, and F1-score to assess idiomatic accuracy, and further include BLEU-1 and BLEU-2146 to measure the semantic similarity and fluency between generated and reference sentences. The BLEU-n score is defined as:

BLEU-n = BP·exp

n

### ∑

wk log pk , (6)

k=1

where pk denotes the k-gram precision, wk is the weight for each k-gram, and BP is the brevity penalty:

1, if c > r, e1−r/c, if c ≤ r,

BP =

(7)

with c and r representing the lengths of the candidate and reference sentences, respectively. Here, sentence-level metrics directly measure whether an MLLM can recover the complete semantic unit (such as an idiom) from visual information. A perfect match at the sentence or word level implies that the model not only perceives individual symbols but also successfully composes them into coherent linguistic meaning. BLEU-1 captures unigram correctness, while BLEU-2 further reflects the model’s ability to maintain short-range structure, providing insight into its capability for symbolic-to-semantic generation.

At Level 3, evaluation is performed at both the word and character levels. At the word level, we calculate Word-level Accuracy, defined as the ratio of exactly matched words to the total number of words. At the character level, we again compute Precision, Recall, and F1-score to quantify symbol-level correctness and semantic consistency. To further examine the model’s image-to-language comprehension and reasoning ability, we introduce Chr-1 and Chr-2, representing the proportions of words that contain at least one and at least two correctly predicted characters, respectively:

- Chr-1 =

Nwords with ci≥1 Nword

, (8)

- Chr-2 =

Nwords with ci≥2 Nword

. (9)

These two indicators help distinguish partial comprehension from complete failure, as higher Chr-1 or Chr-2 values indicate that the model at least recognizes key symbolic components even if it fails to generate the full expression.

After analyzing the character/word-level metrics, we observed that some MLLMs can correctly infer the meanings of individual emojis but still fail to produce the correct idiom, often generating expressions that are semantically related yet lexically or structurally different, and in some cases even exhibiting semantic drift or hallucination. To capture this phenomenon,

we introduce an additional metric for evaluating semantic similarity between the model output and the reference. A large language model (GPT-4o) serves as an expert scorer, assigning a similarity score on a 1–5 scale, where 1 denotes complete dissimilarity and 5 denotes complete semantic equivalence. This metric complements the character- and word-level measures by emphasizing semantic coherence rather than surface form, offering a more holistic assessment of an MLLM’s symbolic comprehension.

For mathematical, physical, and chemical symbols, all tasks are evaluated using Accuracy (Acc), which measures the proportion of correctly predicted symbols or symbolic relations among all test samples. This metric reflects the model’s ability

- to recognize structured symbolic elements and maintain logical consistency in scientific reasoning:

Accuracy =

Ncorrect Ntotal

, (10)

here, Ncorrect denotes the number of correctly predicted samples, and Ntotal denotes the total number of samples.

#### D Extended Conclusion and Future Perspectives

In this work, we systematically investigate the capacity of multimodal large language models (MLLMs) to perceive, reason about, and critically associate visual symbols in discrete semantic spaces. Motivated by fundamental principles of human symbolic cognition, we introduce a hierarchical, multi-domain benchmark that disentangles perceptual recognition, compositional reasoning, and critical symbolic cognition. Through extensive evaluations of state-of-the-art MLLMs, our study reveals a pronounced cognitive mismatch: despite impressive reasoning capabilities, current models frequently fail at foundational visual symbol grounding, relying instead on linguistic priors, procedural imitation, or memorized patterns. Our findings challenge a prevailing assumption in multimodal intelligence that visual recognition is inherently simpler than reasoning. Instead, we observe a consistent recognition-reasoning inversion phenomenon, where higher-level reasoning performance often masks deficiencies in low-level symbolic perception. This phenomenon underscores a key limitation of existing training paradigms: while models excel at leveraging large-scale continual natural images, they struggle to construct stable, compositional visual representations of abstract, discrete symbols. As a result, apparent success on symbolic tasks may reflect compensatory language-driven inference rather than genuine visual understanding.

Looking forward, several research directions emerge. First, future MLLMs may benefit from training objectives that explicitly emphasize discrete visual symbol formation, such as supervision on symbolic primitives or structured perceptual bottlenecks that prevent premature reliance on language priors. Second, tighter integration between vision and reasoning modules, potentially through iterative perception–reasoning loops, may help align visual evidence with logical inference, reducing compensatory shortcuts. Third, extending symbolic benchmarks to interactive or embodied settings could further illuminate their agentic abilities, showing how perception, action, and symbolic reasoning co-evolve in dynamic environments. Finally, insights from cognitive science and neuroscience offer a promising avenue for designing architectures and curricula that more closely mirror human symbolic learning trajectories.

In summary, this work highlights that symbolic understanding remains a critical, underexplored frontier for multimodal intelligence. By exposing fundamental limitations and providing a structured evaluation framework, we hope to catalyze future research toward MLLMs that not only reason fluently, but also perceive, interpret, and critique the symbolic structures that underpin human knowledge.

#### E Additional Case Studies

To facilitate a more granular and comprehensive examination of the visual semiotic behaviors of Multimodal Large Language Models (MLLMs) within discrete semantic spaces, this Supplementary provides a suite of representative case studies (Cases 1–5). Encompassing diverse cognitive dimensions from foundational perception to high-level logical reasoning, these cases elucidate the intricate behavioral patterns and cognitive nuances inherent in symbolic processing.

Please read the text in the image and output the complete sentence exactly as it appears, marking any incorrect characters with a capital letter X.

[Figure 23]

Answer Correct: 我回到家，给家人看了后，家人露出了久违的笑容，那天，爷爷笑了，笑得很灿烂，虽然没有说什么，但我能 感觉出来，爷X很开心。 Trans.: I went home and showed it to my family. They smiled, a smile I hadn’t seen in a long time. That day, my grandpa smiled, smiled brightly. Although he didn’t say anything, I could feel that grandpa was very X.

###### Model Output

Model: 我回到家，给家人看了后，家人露出了久X的笑容。那天，爷爷笑了，笑得很灿烂，虽然没有说什么，但我能感 觉出来，爷爷很开心。 Trans.: I went home and showed it to my family. They smiled, revealing a smile that had been missing for a long X. That day, grandpa smiled, smiled brightly. Although he didn’t say anything, I could feel that grandpa was very happy.

###### Analysis

Lacking precise symbolic discrimination, the model misclassified correct characters as anomalies.

Case 1. Case of Language: Faked Character Detection.

###### Question

Task 2: Misspelled-character detection. Ignore punctuation and the wrong-character marker X, and also ignore the unknown character U. Identify misspelled characters by comparing the source and the correct sentence.

[Figure 24]

Answer Correct: {“n”: 1, “diffs”: [{“idx”: 1, “src”: “射(shoot)”, “tgt”: “躺(lie down)”}]} Model Output

Model: {“n”:1,“diffs”:[{“idx”:1,“src”:“射在沙发上看。这有许多种书。如果想”}]} Trans.: {“n”:1,“diffs”:[{“idx”:1,“src”:“Shooting on the sofa and reading. There are many kinds of books here. If you want.”}]}

###### Analysis

Lacking a mechanism for character-level decomposition and comparison.

Case 2. Case of Language: Misused-Character Detection.

As shown in the figure, a light spring with a stiffness coefficient k is placed inside a smooth vertical tube. The lower end of the spring is fixed to the ground, and its upper end is connected to a small ball A of mass m. When the system is at rest, ball A is at position O. Another small ball B, also of mass m, is released from rest at point P, located a distance H above A. After colliding instantaneously with A, the two balls move downward together, performing simple harmonic motion. Both balls can be regarded as point masses, and during motion, the deformation of the spring remains within its elastic limit. When the spring is stretched or compressed by a distance x, its elastic potential energy is Ep = 12kx2. Given that H = 3mgk and the acceleration due to gravity is g, find the maximum distance that ball A moves downward from point O after the collision.

[Figure 25]

###### Answer 3mg/k

Model Output According to the law of conservation of energy, the gravitational potential energy of ball B is converted into kinetic energy: mgH = 12mv2B. Substituting H = mgk and simplifying gives vB = 2mgk .

###### Analysis

Originally H = 3mgk , but substituting H = mgk led to an incorrect final result.

Case 3. Case of Physics: Mechanical Symbol Recognition.

Question Within the rectangular region ABCD, there exist multiple adjacent layers of uniform electric fields. Each layer has a height of d, and the electric field strength in each layer has a magnitude of E, with its direction alternating vertically between layers. The length of side AB is 12d, and that of side BC is 8d. A beam of particles, each with mass m and charge +q, enters the field from the midpoint of the left boundary of the device. The particles have an initial kinetic energy of Ek and an incident angle θ, moving within the plane of the paper. The effects of gravity and inter-particle interactions are neglected. When Ek = 83qEd and particles are uniformly injected into the electric field with incident angles θ in the range −π2 ∼ π2 , find the ratio of the number of particles exiting through side CD to the total number of incident particles, denoted as N : N0.

[Figure 26]

Answer [’1/2’]

Model Output

8

3. To solve this problem, we need to analyze the motion of the particle in each layer of the electric field and then calculate the number of particles that will exit the device from the CD edge. The calculation of the number of particles that exit the device from the CD

edge involves integrating the particle’s velocity vector over the entire length of the device. This is a complex integral that requires knowledge of calculus and numerical methods to solve. However, the final answer is given as: N = 83.

###### Analysis

The reasoning process remains at the level of formula stacking, ultimately halting after listing several equations and randomly producing a seemingly plausible but physically incorrect answer.

Case 4. Case of Physics: Circuit Operation Reasoning.

###### Question

What are the customary durations and temperature gradients necessary for the execution of this chemical reaction?

- [A] Brc1ccc2c(c1)OCC2 and CN(C)C=O undergo a reaction, culminating in the production of O=Cc1ccc2c(c1)OCC2. -14.0◦C is commonly recognized as the preferred temperature for this chemical reaction. In this chemical reaction, 14.8 hours is usually required.
- [B] Brc1ccc2c(c1)OCC2 and CN(C)C=O undergo a reaction, culminating in the production of O=Cc1ccc2c(c1)OCC2. -78.0◦C is commonly recognized as the preferred temperature for this chemical reaction. In this chemical reaction, 0.75 hours is usually required.
- [C] Brc1ccc2c(c1)OCC2 and CN(C)C=O undergo a reaction, culminating in the production of O=Cc1ccc2c(c1)OCC2. 148.5◦C is commonly recognized as the preferred temperature for this chemical reaction. In this chemical reaction, 18.7 hours is usually required.
- [D] Brc1ccc2c(c1)OCC2 and CN(C)C=O undergo a reaction, culminating in the production of O=Cc1ccc2c(c1)OCC2. -43.0◦C is commonly recognized as the preferred temperature for this chemical reaction. In this chemical reaction, 17.2 hours is usually required.

[Figure 27]

Answer B

Model Output

B. Here are the key points to consider: Temperature: Lithium reagents are typically used in low temperatures to prevent side reactions and maintain reactivity. Common temperatures for such reactions range from -40°C to -80°C. Analysis of Options: Option A: Temperature: -14.0°C Time: 14.8 hours. This temperature is too low for typical lithium reagent reactions, which usually require much lower temperatures. Option B: Temperature: -88.0°C. Time: 0.75 hours. This temperature is very low, which is typical for lithium reagent reactions. The short time of 0.75 hours is also plausible for reactions that proceed quickly under such conditions.

###### Analysis

Able to identify key reactants in the equation and actively invoke its internal knowledge for analysis.

Case 5. Case of Chemistry: Reaction Condition Inference.

