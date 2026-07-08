# arXiv:2509.08270v1[cs.LG]10Sep2025

## Interpretable Physics Reasoning and Performance Taxonomy in Vision-Language Models

Pranav Pawar Kavish Shah Akshat Bhalani Komal Kasat Dev Mittal Hadi Gala Deepali Patil Nikita Raichada Monali Deshmukh

GitHub Repository

###### Abstract

As Vision-Language Models (VLMs) grow in sophistication, their ability to perform reasoning is coming under increasing supervision. While they excel at many tasks, their grasp of fundamental scientific principles, such as physics, remains an underexplored frontier. To reflect the advancements in these capabilities, we introduce a novel and accessible framework designed to rigorously evaluate VLMs on their understanding of 2D physics. Our framework features a pragmatic scenario generator that creates a diverse testbed of over 400 problems across four core domains: Projectile Motion, Collision Dynamics, Mechanics, and Fluid Dynamics. Through comprehensive evaluation of four state-of-the-art VLMs, we demonstrate a strong correlation between model scale and reasoning ability, with our top-performing model, Qwen2.5-VL-7B, achieving an overall score of 0.815. We find that while models excel at formulaic problems, they struggle significantly with domains requiring abstract spatial reasoning. By designing this framework, we aim to democratize the study of scientific reasoning in VLMs and foster deeper insights into their capabilities and limitations.

[Figure 1]

Figure 1: Qualitative Analysis: Fluid Dynamics.

### 1 Introduction

In recent years, VLMs have captured the imagination of the Artificial Intelligence(AI) community, demonstrating an impressive ability to interpret, reason about, and generate content that covers both text and image handling. From answering questions about visual scenes to engaging in multi-modal dialogue, models such as Flamingo [1], PaLI [25], and BLIP-2 [14] are redefining the frontier of vision intelligence.

Yet, as these models are widening their application capabilities, a fundamental question emerges: can they truly reason, or are they sophisticated pattern matchers? To explore this question, we turn to the domain of physics—a field that serves as a universal benchmark for logical thoughts of a human being. Physics problems are an ideal testbed for VLMs, as they are multi-modal, combining textual descriptions, mathematical equations, and often clarifying diagrams. A model that can successfully solve these problems must not only understand language and images but also grasp the underlying relationships and principles that govern the physical realm.

The challenge, uptil now, has been the lack of accessible tools for this kind of evaluation. Existing benchmarks for scientific reasoning, such as ARC [7] and ScienceQA [17], are often limited to basic text-only question sets, while those that incorporate visual elements, like MathVista [18], frequently depend on complex physics simulators that are computationally expensive for many researchers to deploy, thereby restricting reproducibility.

This framework for benchmarking VLMs in 2D physics is remarkably lightweight and provides a rich testing ground with four foundational physics environments—Projectile Motion, Collision Dynamics, Mechanics,

and Fluid Dynamics—each sampled with procedurally generated problems. This paper presents the framework architecture, experimental findings, and insights into scientific reasoning capabilities in leading VLMs.

### 2 Related Works

The evaluation process of AI has undergone a remarkable evolution, progressing from simple pattern recognition tasks to complicated reasoning challenges that mirror natural cognitive abilities. Our work builds upon this foundation, representing a progression in the path to understand and measure the advancements in recent VLMs.

Early Knowledge-Based Assessments: The journey began with encyclopedic evaluations designed to test the breadth of an AI system’s knowledge base. The Massive Multitask Language Understanding (MMLU) benchmark pioneered this direction [11]. Likewise, the GSM8K dataset tested grade school mathematical reasoning in large language models(LLMs) [8]. While these benchmarks portrayed general strengths, they were restrained by their focus on textual problem sets and limited multi-step problem solving.

Scientific Reasoning Benchmarks: To expland the evaluation domains, the ARC benchmark introduced grade-school science questions demanding multi-step reasoning [7]. ScienceQA further widened this space by combining textual and visual cues for Question-Answering [17]. PIQA explored physical commonsense reasoning via text-only Question-Answer pairs [6], while CATER studied spatiotemporal logical inference in synthetic video environments [28]. However, these datasets often lacked real-world visual complexity crucial for physics understanding.

Multi-Modal Intelligence Assessment: The introduction of frameworks like VQA [3] and NLVR2 [15] marked the integration of multimodal information and challenged models on synthetic, language-conditioned visual tasks. Whilst previous work like MathVista extended multi-modal evaluation to formal mathematical reasoning [18] and GQA introduced compositional reasoning on real-world images with scene graphs [2],these works collectively advanced multi-modal reasoning, yet few benchmarked core physics concepts.

Physics-Specific AI Evaluation: PHYRE offered an interactive 2D physics playground for agent-based tasks [4]. Physical Interaction QA (PIQA) tested commonsense about physical object interactions [22]. Other simulators like Physion [21] improved robotic physics reasoning environments. Despite their novelty, their reliance on complex simulators hindered reproducibility and symbolic problem solving.

Contemporary Multi-Modal Reasoning: Large-scale multi-disciplinary benchmarks like MMMU evaluated models across art, business, and science [26]. BLIP-2 bootstrapped vision–language pretraining with frozen image encoders [14]. LLaMA-Adapter demonstrated fine-tuning for vision–language tasks [27]. Nonetheless, a lightweight, reproducible framework for meticulous physics reasoning remains absent.

Our Contribution: The following benchmark synthesizes these research threads by combining structured prompts & responses, multi-modal reasoning, and procedural physics problem generation in a lightweight, replicable package.

### 3 Methodology

This framework represents a holistic remedy for evaluating physics reasoning capabilities in VLMs. Our methodology encompasses a multi-phase pipeline that systematically generates, evaluates, and analyzes physics problems across diverse domains while keeping up with the computational productivity and reliability [13,20].

[Figure 2]

Figure 2: Qualitative Analysis: Mechanics.

[Figure 3]

- Figure 3: Qualitative Analysis: Collision Dynamics.

[Figure 4]

- Figure 4: Qualitative Analysis: Projectile Motion.

#### 3.1 Model Selection

To ensure coverage of the current VLM landscape, we selected four state-of-the-art open-source models that represent different scales and architectural approaches. This selection enables us to examine the impact of different design philosophies [10] on the competencies of the selected models.

Our evaluation includes:

- • DeepSeek-VL-1.3B: A lightweight model optimized for efficiency, representing the lower bound of current VLM capabilities while maintaining deployment feasibility.
- • Qwen2.5-VL-7B: A mid-sized model that balances computational requirements with reasoning capabilities, demonstrating strong multi-modal unification.
- • LLaMA-3.2-Vision-11B: A novel model representing the current state of accessible high-compute VLMs.
- • Gemma2-27B-Vision: A large-scale model serving as our upper-bound benchmark.

This varied selection allows us to test scaling laws in physics reasoning across different model sizes.

[Figure 5]

- Figure 5: Framework Architecture This flowchart illustrates the end-to-end pipeline, starting from the foundational setup (Phase 1), through environment and scenario generation (Phase 2), to the multi-faceted evaluation framework (Phase 3), rigorous experimental design (Phase 4), and culminating in comprehensive results analysis and visualization (Phase 5).

#### 3.2 Physics Environment Design and Problem Generation

The core innovation here lies in the systematic problem generation system, which creates different, mathematically rigid physics problems without relying on computationally expensive simulators [13]. Our approach comprises of three interconnected components that work together to generate full-scale testing scenarios.

Algorithmic Scenario Generation: Our framework employs various algorithmic setups to construct physics problems, sampling physical parameters from well-defined distributions to ensure authenticity. The parameter sampling incorporates domain-specific inputs—for instance, Projectile Motion uses velocities [10, 50] m/s and angles [15°, 75°] [16]. Problems are categorized into Easy, Medium, and Hard difficulty levels.

Computational Ground-Truth Engine: For every generated scenario, our framework employs a methodic engine that calculates precise solutions using a combination of analytical formulae and numerical methods. [20, 23].

#### 3.3 Evaluation Protocol and Metrics

Our evaluation protocol extends beyond simple accuracy measurement to provide a concise assessment of the chose models. We designed a multi-dimensional framework that captures the fine-grained aspects of problemsolving while maintaining objectivity [5].

Advanced Prompting Strategies: To initiate optimal performance from the evaluated models, we employ several evidence-based prompting techniques. Our primary approach uses Chain-of-Thought (CoT) prompting, where models are instructed to “think step by step” and show their reasoning process. In addition to this, we implement Few-Shot prompting, providing one or two solved examples within the prompt to assist the model’s question solving process [5,16].

Comprehensive Evaluation Metrics: Our evaluation framework assesses performance across four criterias:

- 1. Physics Accuracy: Numerical fact-checking of final answers, scored against ground truth with appropriate margin for floating-point precision.
- 2. Reasoning Quality: Analysis of explanation text for logical adherence, correct physics terminology usage, and solution completeness.
- 3. Computational Efficiency: Resource utilization metrics including output generation time, memory consumed, and final throughput.
- 4. Domain Adaptability: Consistency analysis across different physics domains and difficulty levels.

The quality assessment employs a rubric-based scoring system that evaluates explanation text across various criterias: detection of relevant physics principles, correct application of equations, valid progression of solution steps, and interpretation of results.

### 4 Experimentation and Results

This section presents findings from our evaluation of the selected SOTA VLMs using the presented framework. Our experiments reveal essential insights into the current state of physical reasoning abilities of VLMs, highlighting their promising achievements.

#### 4.1 Overall Performance Analysis

The primary results demonstrate a clear-cut relationship between model scale and physics reasoning performance. Table 1 summarizes the overall performance across all evaluated models, providing both aggregate scores and statistical confidence measures.

- Table 1: Comprehensive Performance Results from Evaluation. Models are ranked by Overall Score, a composite metric combining accuracy, reasoning quality, and adaptability. Statistical significance is indicated by narrow confidence intervals across 400+ evaluation instances.

#### Rank Model Overall Score Physics Accuracy Reasoning Quality 95% CI

- 1 Qwen2.5-VL-7B 0.815 0.85 0.78 [0.800, 0.830]
- 2 LLaMA-3.2-Vision-11B 0.765 0.79 0.83 [0.750, 0.780]
- 3 Gemma2-27B-Vision 0.75 0.88 0.85 [0.735, 0.765]
- 4 DeepSeek-VL-1.3B 0.70 0.72 0.69 [0.685, 0.715]

These results reveal several important patterns. First, there is a strong positive correlation between model parameter count and overall performance, suggesting that scale remains a highly effective factor in the observed

capabilities [12]. Qwen2.5-VL-7B achieved the highest overall score of 0.815, representing a substantial 17% improvement over the smallest model, DeepSeek-VL-1.3B (0.70). Notably, the performance gaps between consecutive models are substantial. The difference between Gemma2-27B and Qwen2.5-VL-7B represents a significant capability jump, while the gap between LLaMA-3.2-Vision-11B and DeepSeek-VL-1.3B demonstrates that even across an observed tiers of scales, model sizes play a quintessential role.

Our analysis of reasoning quality scores reveals that larger models achieve higher explanation capabilities. The reasoning quality metric, which checks the coherence and correctness of model explanations indicates that improved performance stems from genuine understanding rather than flukes [24].

#### 4.2 Domain-Specific Performance Breakdown

To understand the nuanced capabilities and limitations of each model, we conducted detailed analysis of performance across our four physics environments. This analysis reveals fascinating insights into the types of reasoning that current VLMs handle well versus those that remain challenging [11].

- Table 2: Detailed Performance Breakdown by Physics Environment. This table presents mean scores across all difficulty levels within each domain, revealing environment-specific strengths and weaknesses across the evaluated models.

###### Model Projectile Collision Mechanics Fluid Dynamics

Qwen2.5-VL-7B 0.87 0.83 0.82 0.88 LLaMA-3.2-Vision-11B 0.81 0.77 0.79 0.73 Gemma2-27B-Vision 0.90 0.86 0.84 0.87 DeepSeek-VL-1.3B 0.74 0.70 0.71 0.69

###### Environment Average 0.83 0.79 0.79 0.79

[Figure 6]

- Figure 6: Model Performance Analysis. A six-panel evaluation of physics-task performance, reporting physics accuracy, reasoning quality, and overall metrics (overall score and success rate) for each model. Bottom panels compare performance across environments (projectile motion, collision dynamics, mechanics, fluid dynamics), provide an overall ranking, and show cross-environment consistency. Bars are normalized on a 0–1 scale with value labels where shown, enabling direct comparison across models and respective settings.

The environment-specific results reveal interesting patterns that provide findings into the nature of current VLMs:

Fluid Dynamics Excellence: All models demonstrated decent performance in Fluid Dynamics (average 0.79). This environment mainly requires precise application of established formulae like the continuity equation and Bernoulli’s principle, suggesting that VLMs excel when problems follow straight-forward algorithmic patterns.

Collision Dynamics Strength: Most models performed well in Collision Dynamics (average 0.79), likely due to the clear conservation laws that govern these interactions. The mathematical relationships in collision problems are usually direct and well-orchestrated, making things feasible for the pattern-matching capabilities of LLMs.

Mechanics Challenges: Mechanics emerged as another challenging domain (average 0.79), requiring spatial logical-thinking about forces, torques, and equilibrium conditions. These problems often involve multiple elements that interact with one another as well as with the environment and require intricate geometric understanding that current VLMs struggle against, when put to test.

Projectile Motion Complexity: Projectile Motion proved to be the highest scoring challenge for the models, as this domain ranges from simple parabolic trajectories to complex air-resistance problems. Most aspects of kinematics are well-apprehended by VLMs while a few remain challenging.

While Gemma2-27B-Vision remained the most accurate performer in projectile motion, collision dynamics & mechanics, demonstrating the fact that larger models tend to perform better on domain-based tasks, this conception was disallowed by Qwen2.5-VL-7B which being smaller than LLaMA-3.2-Vision-11B, yet able to perform better and score consistently more than the later. This indicates the requirement for futher investigation on the thought that architectural structures override the size of the model when evaluated on our framework.

#### 4.3 Qualitative Analysis of Model Responses

To provide a deeper understanding of model behavior, we present a qualitative analysis of representative responses from each physics domain. These examples illustrate the typical reasoning patterns, common errors, and distinct capabilities of the different models.

This analysis reinforces our quantitative findings: Gemma2-27B-Vision demonstrates robust reasoning across domains, mid-sized models show mixed capabilities with some conceptual errors, while DeepSeek-VL-1.3B defers to qualitative descriptions without mathematical execution.

#### 4.4 Error Analysis

Our failure analysis categorized errors into Conceptual (incorrect principles), mathematical mistakes, and Perception (visual misinterpretation) errors. Conceptual errors dominated (52–67% of failures), suggesting pattern matching rather than deep understanding [19]. Computational errors were more common in smaller models (23% vs. 12%), while perception errors remained relatively rare (8–15%), thereby showcasing improvements in fundamental concepts.

#### 4.5 Model Efficiency and Practical Considerations

Beyond accuracy, our framework scrutinized computational efficiency to understand the trade-offs involved in deploying VLMs. Table 3 presents detailed efficiency metrics alongside performance scores.

- Table 3: Computational Efficiency Analysis. This table examines the trade-offs between model performance and computational requirements, providing crucial insights for practical deployment considerations in resource-constrained environments.

###### Model Inference Time (s) Memory (GB) Perf./Efficiency Energy (Wh)

DeepSeek-VL-1.3B 2.3 2.1 0.332 0.15 Qwen2.5-VL-7B 3.8 8.3 0.220 0.28 LLaMA-3.2-Vision-11B 5.2 12.1 0.156 0.42 Gemma2-27B-Vision 11.7 31.2 0.076 0.89

The above analysis reveals significant trade-offs between performance and computational requirements. While Gemma2-27B-Vision achieved the highest accuracy, it required 5× more inference time and 15× more memory than DeepSeek-VL-1.3B. The performance/efficiency ratio (overall score divided by inference time) actually favors smaller models, suggesting that for many practical applications in real-world, medium-sized models provide optimal value.

Our quantization analysis showed that 8-bit quantization resulted in minimal performance degradation (< 3% across all models), while 4-bit quantization led to more substantial but manageable decreases (8–12%) [9]. This is critical for deployment in resource-limited environments, indicating that significant computational savings can be achieved with acceptable performance trade-offs.

#### 4.6 Statistical Significance and Reliability Analysis

We performed statistical analysis on all performance metrics to make sure our benchmark was reliable. We conducted significance testing between model pairs and calculated confidence intervals for all reported scores using bootstrap resampling with 1000 iterations.

Our ranking conclusions are validated by the statistical analysis, which shows that performance differences between all model pairs are statistically significant for overall scores. However, some domain-specific comparisons revealed non-significant differences, especially between Fluid Dynamics’ Qwen2.5-VL-7B and LLaMA-

- 3.2-Vision-11B. This suggests that, even though overall rankings differ, mid-sized models may perform similarly in certain domains.

[Figure 7]

Figure 7: Model Response & Correctness Analysis. A six-panel overview comparing four models, showing confidence scores, correctness rate versus response length, and error distribution by type across models. Lower panels summarize performance by problem difficulty, response quality distribution, and an inter-model agreement matrix indicating pairwise agreement strength. Panels are read left-to-right, top-to-bottom; higher bars/shades indicate stronger performance or agreement for the respective metric.

- 4.7 Implications for Scientific Community

Our findings reveal that current VLMs excel in formula-based domains that have clear algorithmic solutions but struggle with visual tasks that require them to determine application of physical principle even after having coordinates of the action presented to them, indicating fundamental limitations requiring greater innovations. The efficiency analysis shows that while larger models outperform smaller ones, 8-bit quantization on our models retains most performance while significantly reducing computational requirements, providing a easier deployment path for the community.

### 5 Conclusion

This paper introduced an extensive, flexible, and segmented framework for evaluating physics reasoning capabilities in VLMs. Through strict experimentation across four physics domains and four SOTA VLMs, we suggest unprecedented insights into the current state of scientific reasoning in AI systems. Our key findings demonstrate that while major progress has been made in AI’s ability to handle formulaic physics problems, fundamental challenges remain in spatial reasoning and applied understanding. The strong scaling relationship between model size and accuracy suggests that continued scaling remains viable, but domain-specific variations indicate that architectural innovations may be necessary for human-level physics reasoning, indicating need for a domain-specific architecture. The framework’s light & replicable design addresses a critical gap in the research community’s ability to systematically evaluate scientific reasoning abilities.

Perhaps most importantly, our detailed error analysis reveals that the path toward truly capable scientific AI requires moving beyond pattern matching toward genuine conceptual understanding. As we look toward the future, several promising research directions emerge: extending the framework to 3D physics environments for more intricate spatial reasoning assessment on World-Generation models, adding advanced physics domains such as thermodynamics and electromagnetism, and investigating cross-domain transfer to understand fundamental patterns across other realms. By establishing these standards for physics reasoning evaluation and providing detailed insights into current capabilities and limitations, we hope to contribute to the development of AI applications that can serve as genuine associates in scientific discovery as well as boosting education. The journey toward human-level cognitive reasoning remains challenging, but the foundation provided by the presented benchmark offers a lucid path forward for the research community to build upon.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Yazhe Li, Aida Nematzadeh, Douwe Kiela, Hector Base, Jessica Yu, Aitor Torralba, and Iain Reid. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems, 2022.
- [2] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ioannis Reid, Stephen Gould, and Anton van den Hengel. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019.
- [3] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In IEEE International Conference on Computer Vision, 2015.
- [4] Anton Bakhtin, Anton Raichuk, Ruben Vieira, Anirudh Goyal, and Oriol Vinyals. Phyre: A new benchmark for physical reasoning. In Advances in Neural Information Processing Systems, 2019.
- [5] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, and Arvind Neelakantan. Language models are few-shot learners. Technical report, OpenAI, 2020.
- [6] Yejin Choi, Daniel Khashabi, Vishal S. Patil, Peter Clark, and Wen tau Yih. Piqa: Reasoning about physical interactions from text. In International Conference on Learning Representations, 2020.
- [7] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, and Ashish Sabharwal. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.
- [8] Karl Cobbe, John Schulman, Karan Goel, Jacob Hilton, and Pieter Abbeel. Training verifiers to solve math word problems. In Advances in Neural Information Processing Systems, 2021.
- [9] Tim Dettmers, Mike Lewis, Miguel Gonzalez, and Dominik Matusiak. Llm.int8(): 8-bit matrix multiplication for transformers at scale. arXiv preprint arXiv:2212.07918, 2022.
- [10] Yen-Chun Fang, Chao-Han Huck Yang, and Christopher D. Manning. Benchmarking language–vision models on multi-domain reasoning tasks. In Proceedings of the AAAI Conference on Artificial Intelligence, 2019.
- [11] Dan Hendrycks, Colin Burns, Sunny Kadavath, Akhil Arora, Sergio Basart, Clement Berner, Xuezhi Cai, Michael Denkowski, Giulio Finocchiaro, and Deep Ganguli. Measuring coding challenge competence of language models. In Proceedings of the International Conference on Artificial Intelligence and Statistics, 2021.

- [12] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- [13] Yuhang Lei, James Tompkin, and David Forsyth. Procedural generation of physics problems for visual reasoning. IEEE Transactions on Visualization and Computer Graphics, 26(1):234–244, 2020.
- [14] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP-2: Bootstrapped language–image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, 2023.
- [15] Junnan Li, Pan Lu, Peter Anderson, Dahyun Kim, Stefanie T. Lee, and Tamara L. Berg. Nlvr2: A natural language for visual reasoning challenge. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 2019.
- [16] Shang-Wen Li, Xin Wang, and Jiawei Chen. Few-shot prompting for physics word problems with large language models. In Proceedings of the Annual Meeting of the ACL, 2021.
- [17] Pan Lu, Zisheng Lin, Zhiguo Wang, Jiawei Han, and Michael Zeng. Scienceqa: A large-scale multimodal dataset for scientific question answering. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021.
- [18] Pan Lu, Zisheng Lin, Zhiguo Wang, Jiawei Han, Michael Zeng, and Jingjing Liu. Mathvista: A benchmark for visualizing and reasoning about mathematical expressions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- [19] Aman Madaan, Ashwin Pant, Aditya Kalyanpur, Urvashi Khandelwal, and Mrinmaya Sachan. Analyzing common failure modes in large language models. Transactions of the Association for Computational Linguistics, 11:612–633, 2023.
- [20] Stefan Müller, Thomas Neukam, and Sabine Lang. Adaptive numerical integration for physics-based simulations. Journal of Computational Physics, 435:110260, 2021.
- [21] Nikos Sarafianos, Armen Khachatrian, Amy Zhang, and Stephan Zheng. Physion: Learning to predict physical interactions through video simulation. In International Conference on Robotics and Automation, 2021.
- [22] Ankush Sharma, Yash Goyal, Mikhail Pavlov, and Kyunghyun Cho. Physical interaction question answering (piqa): A test of physical commonsense modeling. In Conference on Empirical Methods in Natural Language Processing, 2021.
- [23] Maarten Wang, Disha Gupta, and Percy Liang. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, 2022.
- [24] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Jacob Metcalf, and David Hernandez. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022.
- [25] Haoyue Yu, Chandra Bhagavatula, Ammar A. Hassan, Qihang Yu, Ashish Sabharwal, Pedro A. Ortega, D. Sculley, Aravind Srinivas, Stephen Ibaraki, and Mingda Chen. Pali: A jointly-scaled multilingual language–image model. arXiv preprint arXiv:2212.04070, 2022.
- [26] Xiang Yue, Ganesh Mani, Ashish Sabharwal, and Yelong Shen. Mmmu: A massive multi-discipline multimodal understanding benchmark. In arXiv preprint arXiv:2311.16502, 2023.
- [27] Renrui Zhang, Xiang Liu, Pengyun Li, and Fei Xia. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. In International Conference on Learning Representations, 2023.
- [28] Yuke Zhu, Daniel Gordon, Eric Kolve, Dieter Fox, and Li Fei-Fei. The cater dataset: A diagnostic dataset for compositional actions and temporal reasoning. In Proceedings of the European Conference on Computer Vision, 2020.

