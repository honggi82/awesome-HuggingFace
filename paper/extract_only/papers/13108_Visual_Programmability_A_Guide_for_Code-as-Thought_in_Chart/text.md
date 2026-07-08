# arXiv:2509.09286v1[cs.CV]11Sep2025

## Visual Programmability: A Guide for Code-as-Thought in Chart Understanding

Bohao Tang1,2 Yan Ma3 Fei Zhang1,2 Jiadi Su2,3 Ethan Chern1,2

Zhulin Hu1 Zhixin Wang2 Pengfei Liu1,2∗ Ya Zhang1∗

1 Shanghai Jiao Tong University 2 Shanghai Innovation Institute 3 Fudan University

### Abstract

Chart understanding presents a critical test to the reasoning capabilities of VisionLanguage Models (VLMs). Prior approaches face critical limitations: some rely on external tools, making them brittle and constrained by a predefined toolkit, while others fine-tune specialist models that often adopt a single reasoning strategy, such as text-based chain-of-thought (CoT). The intermediate steps of text-based reasoning are difficult to verify, which complicates the use of reinforcement-learning signals that reward factual accuracy. To address this, we propose a Code-asThought (CaT) approach to represent the visual information of a chart in a verifiable, symbolic format. Our key insight is that this strategy must be adaptive: a fixed, code-only implementation consistently fails on complex charts where symbolic representation is unsuitable. This finding leads us to introduce Visual Programmability: a learnable property that determines if a chart–question pair is better solved with code or direct visual analysis. We implement this concept in an adaptive framework where a VLM learns to choose between the CaT pathway and a direct visual reasoning pathway. The selection policy of the model is trained with reinforcement learning using a novel dual-reward system. This system combines a data-accuracy reward to ground the model in facts and prevent numerical hallucination, with a decision reward that teaches the model when to use each strategy, preventing it from defaulting to a single reasoning mode. Experiments demonstrate strong and robust performance across diverse chart-understanding benchmarks. Our work shows that VLMs can be taught not only to reason but also how to reason, dynamically selecting the optimal reasoning pathway for each task.

### 1 Introduction

The capabilities of Vision-Language Models (VLMs), built upon powerful Large Language Models [5, 49], have rapidly advanced multimodal understanding (e.g., [40, 26, 1, 8, 3]). Among the many applications, chart understanding stands out as a critical benchmark [15], testing an AI’s ability to connect low-level visual perception [21] with high-level logical inference. Despite significant progress with specialized models [7, 28, 32], a fundamental generalization problem remains: even state-of-the-art VLMs show a stark performance decline on the complex, "in-the-wild" charts found in real-world contexts [18, 52].

Prevailing efforts to overcome this generalization challenge have largely followed two dominant strategies, each with distinct drawbacks. The first approach treats the VLM as a controller for external tools and APIs [17, 13, 48] (see Figure 1a). While powerful, their reliance on a predefined toolkit

∗Co-corresponding author. †Open-source implementation are available at github.com/Aphelios-Tang/Code-as-Thought.

Preprint. Under review.

[Figure 1]

- Figure 1: Adaptive Reasoning vs. Fixed Strategies for Chart Understanding. Prevailing approaches are limited by their rigid strategies. (a) Tool-Use Models are constrained by a predefined toolkit and fail on novel tasks. (b) Specialized Models employ a single reasoning pattern (e.g., text-only or code-only), which limits their generalization. In contrast, our (c) Adaptive Framework first assesses a task’s Visual Programmability. It then dynamically selects the precise Code-as-Thought pathway for programmable tasks or the robust Direct Visual Reasoning pathway for complex ones, achieving superior performance across all chart types.

makes them brittle when encountering charts that require capabilities beyond their predefined functions [43, 62, 39, 38]. The second strategy involves fine-tuning specialized models on chart-specific data [7, 28, 32] (see Figure 1b). These models typically rely on a monolithic reasoning pattern—that is, they exclusively use a single mode of thought, such as text-based Chain-of-Thought or code-based reasoning. This lack of flexibility hinders their ability to generalize to out-of-distribution (OOD) visualizations, as no single reasoning style is optimal for all chart types [52, 58].

The limitations of predefined toolkits highlight the appeal of a more universal and flexible tool: code. Unlike a fixed API, code can be dynamically generated to create novel tools tailored to the specific visual complexities of any chart, a concept explored in recent agentic vision systems [66]. However, the shared failure of rigid approaches motivates our core insight: the optimal reasoning strategy depends on the task itself. Some charts are easily broken down into programmable elements [10], while others require a holistic visual analysis that code cannot capture. This requires moving beyond refining a single reasoning chain [53] to mastering strategy selection—a shift that reflects a broader trend in AI towards deliberate problem-solving [51, 45, 61] and adaptive computation [12]. This principle is also central to the design of frontier models like GPT-5 [35], which aim to integrate similar adaptive capabilities.

To address these challenges, we propose the concept of Visual Programmability: a learnable, task-dependent property that indicates whether a given chart-question pair is best solved through programmatic reasoning or direct visual analysis. We implement this concept in an adaptive framework that enables a VLM to autonomously choose its reasoning pathway. The model’s decision-making policy is trained via reinforcement learning (RL)—specifically, using the Group Relative Policy Optimization (GRPO) algorithm—guided by a novel dual-reward system. This system is carefully designed to foster adaptive behavior: a data-accuracy reward ensures the generated code is factually grounded to the chart’s content, thereby preventing numerical hallucination. In parallel, a dedicated decision reward explicitly teaches the model the boundaries of programmability, preventing the policy from collapsing into a single, suboptimal mode.

Our experiments, conducted on the Qwen2.5-VL model [3] across a diverse suite of benchmarks, validate our approach. The resulting adaptive model consistently outperforms both pure visual

baselines and rigid code-based methods. It achieves this by dynamically modulating its strategy, heavily employing code-based reasoning (>60%) on benchmarks where it is advantageous, while minimizing its use (<10%) where it is detrimental. Ablation studies confirm that our dual-reward system is essential for preventing mode collapse and fostering strategic diversity. Our contributions are threefold:

- • We introduce Visual Programmability, a novel concept to determine if a chart task is suitable for code-based reasoning, serving as the foundation for adaptive strategy selection.
- • Building on this concept, we develop an adaptive framework that learns to choose the optimal reasoning path (code or vision). This framework is trained with a specialized dual-reward RL system that promotes both factual accuracy and strategic flexibility.
- • Our adaptive model demonstrates outstanding performance and generalization, consistently outperforming rigid strategies across diverse benchmarks by intelligently switching between reasoning modes.

### 2 Related Work

#### 2.1 Programmatic Reasoning for Chart Understanding

The field of chart understanding has evolved rapidly, driven by more capable models and challenging benchmarks. Early research established foundational datasets and tasks [33, 29]. The arrival of powerful Vision-Language Models (VLMs) [23, 22, 9] and frontier systems [1, 8, 3] led to strong performance on these benchmarks, prompting the development of specialized, fine-tuned models [25, 7, 28, 14, 32]. However, their success was often misleading, as they tended to learn benchmarkspecific shortcuts rather than generalizable reasoning skills. This weakness was exposed by a new wave of diverse and complex benchmarks [55, 52, 58, 60], where even state-of-the-art models showed a significant performance drop [18, 15]. We argue that this generalization gap stems not from a lack of model capability, but from strategic rigidity.

To overcome this, many have turned to programmatic reasoning. This paradigm draws inspiration from Large Language Models (LLMs) that function as controllers for external tools and APIs [11, 43, 62, 39, 38]. This concept was quickly adapted for VLMs, enabling them to leverage external modules for visual tasks [57, 46]. A related but distinct strategy involves VLMs generating code not to call a pre-defined tool, but as a symbolic reasoning step for direct execution [48, 47, 6, 2]. Other approaches have explored novel architectures, such as Mixture-of-Experts (MoE) models that route chart-related tasks to specialized modules, which may include code generation capabilities [59, 16]. A separate line of work seeks to integrate symbolic reasoning more deeply by visually grounding the thought process, for instance by inserting coordinate pointers [34], generating sketches [17], or enabling other forms of programmatic self-reflection [65]. While improving code generation skills [67] and creating richer datasets [63] are beneficial, these approaches still rely on a fixed strategy. Our work departs from this by proposing that VLMs possess inherent symbolic capabilities. We thus reframe the challenge: instead of augmenting VLMs with external tools, we teach them to recognize when to deploy their own code-like reasoning, shifting the focus from tool use to strategic selection.

#### 2.2 Adaptive Learning to Strategic Cognition

The idea of a system dynamically adjusting its strategy based on the input, known as adaptive computation, is a long-standing concept in machine learning [4] and finds parallels in theories of human cognition [20]. In modern AI, this often appears as MoE layers, dynamic routing, or adaptive input fusion [42, 19, 56, 37, 50]. These methods typically adapt how computation is performed by selecting different model parameters or pathways. Our work applies this principle at a higher level of abstraction: we teach a model to adapt what reasoning process it uses, making a strategic choice between holistic visual analysis and formal, code-based reasoning. This moves beyond computational efficiency to what we term strategic cognition.

Reinforcement learning (RL) is a natural fit for teaching a model to make such strategic choices without explicit labels. While often used for preference alignment [36, 41], our work uses RL to optimize for verifiable task correctness, a paradigm that has proven effective in other complex reasoning tasks [31, 46]. We use policy-gradient methods [44, 54] in an efficient training setup [68],

which is well-suited for learning from the binary correct/incorrect feedback common in our task [24]. However, a naive accuracy-based reward can cause the model to default to a single, safer strategy—a phenomenon known as mode collapse. To address this, our key contribution is a specialized dualreward system. This system combines the standard accuracy signal with a "decision reward" that explicitly encourages strategic diversity. In doing so, we use RL not just to solve the task, but to teach the model how to effectively manage its own cognitive toolkit.

### 3 Exploring Code-as-Thought as a Universal Strategy

The limitations of the fixed strategies discussed previously motivate us to explore whether a more powerful, formal paradigm could serve as a universal solution for chart understanding. This line of inquiry leads us to investigate Code-as-Thought (CaT) and to pose a foundational question:

Is Code-as-Thought a "silver bullet" for chart understanding?

To answer this, we first investigated the efficacy of a single, fixed CaT strategy. We trained a specialist model on structured data and evaluated its generalization across four diverse benchmarks. We discovered a core limitation that motivates our adaptive framework. Figure 2 visualizes the results on two of these benchmarks—the highly structured ChartX and the complex, "in-the-wild" CharXiv—which most clearly illustrate the performance trade-offs. A detailed description of the setup and full results across all four benchmarks are provided in Appendix A.

[Figure 2]

- Figure 2: Performance of Fixed Strategies Highlights a Critical Trade-off. While the Code-asThought (CaT) strategy excels on structured charts (ChartX), its performance collapses on complex, ’in-the-wild’ charts (CharXiv). All values are accuracy (%).

The results reveal a sharp dichotomy in generalization performance. As shown in Figure 2, the CaT specialist (achieving 71.6% with SFT) excels on the structured ChartX data, confirming its power in high-programmability scenarios. However, this rigid strategy proves brittle. On the complex charts from CharXiv, its accuracy collapses to a mere 18.4%. This failure is often driven by numerical hallucination—where the model generates code from a flawed perception of the chart, then reasons faithfully from this incorrect foundation. A case of this phenomenon is detailed in Appendix B.

Furthermore, we found that enhanced skill and policy optimization are not a panacea. The right side of the figure illustrates that even after applying reinforcement learning (RL), the model’s performance on CharXiv remains critically low, failing to resolve the core conflict. Results with extensive pre-training (CPT+RL) exhibit the same trend and are provided in Appendix A. The conclusion is clear: the issue is not the model’s competence (how well it codes) but determining the strategy’s applicability (whether it should code at all). These experiments confirm the potential of Code-as-Thought but reveal that the optimal strategy is task-dependent, motivating our core thesis: an intelligent system must learn when to use its tools, not just how.

### 4 Adaptive Code-Based Reasoning Framework

[Figure 3]

- Figure 3: Overview of our adaptive reasoning framework. (Top) We introduce the concept of Visual Programmability and use it to guide data annotation. (Middle) At inference, our adaptive VLM selects a reasoning pathway based on the perceived Visual Programmability (VP) of the task. (Bottom) The model’s selection policy is trained using reinforcement learning with a multi-component reward function and the GRPO algorithm.

Our framework enables a Vision-Language Model (VLM) to dynamically select the optimal reasoning strategy for a given chart. As illustrated in Figure 3, it consists of three core parts: an adaptive inference system, a training process based on reinforcement learning, and the underlying concept of Visual Programmability that guides the model’s learning.

- 4.1 Visual Programmability: Understanding the Boundaries of Code

Not all charts are equally well-suited to analysis using Code-as-Thought. To address this, we introduce the concept of Visual Programmability: a learnable, task-dependent property that serves as the foundation for our adaptive reasoning system. It gauges whether a chart-question pair can be faithfully reasoned about using code. This property is not a binary yes-or-no question; rather, it represents a range of suitability influenced by a chart’s structural clarity, its visual complexity, and the nature of the query itself.

- Figure 4 provides several cases that illustrate this concept.

High vs. Low Programmability. The suitability of code-based reasoning varies widely. Some charts exhibit high programmability. These are typically standard bar, line, or scatter plots with clean layouts, where the underlying data can be programmatically extracted with high fidelity. Figure 4 (a) shows a clear example: a standard line chart with explicitly marked data points, making it ideal

[Figure 4]

Figure 4: Cases of Visual Programmability for different charts and tasks.

for precise computational analysis. In contrast, other charts have low programmability. As seen in

- Figure 4 (b), these often include complex scientific visualizations where meaning is conveyed through holistic patterns, such as data contours and distributions. For these charts, essential information is often lost or distorted during symbolic translation.

The Critical Role of Task Dependency. Crucially, Visual Programmability is not an intrinsic chart property alone; it is fundamentally dependent on the user’s query. This is demonstrated by the case in

- Figure 4 (c). For a simple counting task like, "How many distinct data series are plotted?", the chart has high programmability, as the task only requires identifying discrete visual elements. However, for a value-extraction task like, "What is the approximate value of the orange line (h/a = 1000) when d = 7?", the same chart exhibits low programmability. The logarithmic scale makes precise data extraction extremely difficult and error-prone. In this scenario, a Code-as-Thought approach would likely yield a confidently incorrect answer, making direct visual reasoning a more reliable strategy.

This dependency on both the chart and the question necessitates a dynamic reasoning system. An intelligent agent cannot rely on a fixed strategy; it must learn to assess Visual Programmability on the fly to select the most appropriate reasoning path. To enable this, we developed a framework to annotate data for this property, providing the necessary signal for learning this adaptive skill (see

- Appendix D).

#### 4.2 Adaptive Reasoning Mechanism

We formulate the chart-understanding task as a policy learning problem. Given a chart image I and a question Q, our model learns a policy πθ that generates a complete response y. This process is explicitly factorized to first select a strategy token s ∈ {<CODE>,<DIRECT>}, then generate the corresponding reasoning and answer:

P(y|I,Q) = P(s|I,Q) · P(y|I,Q,s). (1)

This factorization is realized by building upon a powerful base model (Qwen2.5-VL-7B) and teaching it to first commit to a strategy by generating a special token, which then dictates the subsequent generation path:

- • Code-based Path (<CODE>): The model generates a Code-as-Thought (CaT) pathway. It writes code to parse the chart into a structured format (e.g., a DataFrame) and then performs computations to find the answer. This path is ideal for charts with high Visual Programmability.
- • Direct Path (<DIRECT>): The model generates a natural language CoT, performing reasoning based on its holistic visual perception. This path is essential for charts with low Visual Programmability where symbolic decomposition would lose critical information.

For automated evaluation, the final answer from both paths must be enclosed in \boxed{}.

#### 4.3 Training via Reinforcement Learning

The crucial challenge is the absence of ground-truth labels for strategy selection. We overcome this by formulating the training as a reinforcement learning problem, allowing the model to learn the optimal policy from outcome-based reward signals.

#### 4.3.1 GRPO Policy Update

We employ Group Relative Policy Optimization (GRPO) [44], a policy-gradient algorithm particularly effective for tasks with verifiable, sparse rewards. GRPO is an efficient algorithm that does not require an explicit reward or value model. For each training instance, we sample a group of G responses from a previous version of the policy, πold, and evaluate them using our comprehensive reward function. The general optimization objective is formulated as:

πθ(ri|x) πold(ri|x)

πθ(ri|x) πold(ri|x)

Ai,clip

JGRPO(θ) = E

,1 − ϵ,1 + ϵ Ai − βDKL(πθ||πref) ,

min

i

(2) where Ai = R(r

i,ϕi)−mean(R(ξ,ϕ))

std(R(ξ,ϕ)) represents the normalized advantage of the i-th response within the group. The parameter ϵ controls the clipping threshold. The final term is a KL divergence penalty against a reference policy, πref, which is regulated by the coefficient β. In our final training configuration, we set β = 0 to focus the optimization entirely on the group-relative reward signal, effectively removing this regularization term.

#### 4.3.2 Comprehensive Reward Function

A naive reward function focused solely on answer accuracy would be insufficient and could lead to mode collapse—where the model defaults to a single, suboptimal strategy. To prevent this and guide the model toward true adaptive behavior, we designed a comprehensive reward function R as a weighted sum of four specialized components:

R = waccracc + wdecisionrdecision + wdatardata + wformatrformat. (3) The components are:

- 1. Accuracy Reward (racc): The primary reward, providing a binary signal (1.0 or 0.0) based on the correctness of the final answer.
- 2. Decision Reward (rdecision): Our key innovation to prevent mode collapse. This reward explicitly incentivizes selecting the correct strategy based on the chart’s pre-annotated Visual Programmability. It gives a full reward for a correct answer via the correct strategy, a partial reward for a wrong answer but using the correct strategy (to encourage exploration), and zero reward for using the wrong strategy. This component is essential for teaching the model to learn the decision boundary.
- 3. Data Accuracy Reward (rdata): Applied only to the <CODE> path, this reward tackles the issue of code "hallucination." It programmatically compares the DataFrame generated by the model’s code to a ground-truth data table, evaluating the fidelity of the extracted data. This ensures the model generates code that is not just syntactically valid, but semantically faithful to the chart. The calculation process is visualized in Figure 5.
- 4. Format Reward (rformat): A small reward to enforce correct output structure (i.e., using \boxed{}), ensuring reliable parsing.

This multi-faceted reward design creates a nuanced optimization landscape that simultaneously pushes the model toward accuracy and strategic intelligence. The detailed implementation of the Data Accuracy Reward is provided in Appendix C.

### 5 Experiments

#### 5.1 Experimental Setup

Training Data. Our training is based on the ChartMimic [60] dataset, which contains 4,800 diverse chart-code pairs without QA. To support our adaptive learning goal, we expanded this dataset

[Figure 5]

Figure 5: Illustration of the Data Accuracy Reward calculation.

by generating new question-answer pairs with Gemini-2.5-Flash [8], using the prompts found in

- Appendix E. This process resulted in a balanced training set that includes charts well-suited for code-based reasoning as well as those demanding direct visual interpretation.

Evaluation Benchmarks. We evaluate our models on four benchmarks chosen to represent a wide spectrum of Visual Programmability. This allows for a comprehensive assessment of our model’s ability to adapt its reasoning strategy.

- • ChartX [55]: Represents the high-programmability end of the spectrum. Its 1,152 structured charts are ideal for testing the effectiveness of code-based reasoning.
- • ChartBench [58]: Focuses on numerical reasoning where data points are not explicitly labeled, forcing the model to perform visual interpolation. This makes it a strong test for programmatic data extraction from visual cues. We use 2,000 samples from its NQA task.
- • ChartQA [29]: Features 2,396 real-world charts with both human-generated and templatebased questions, testing a broad spectrum of complexities from basic data retrieval to multi-step reasoning.
- • CharXiv [52]: Represents the low-programmability, "in-the-wild" end of the spectrum. Its 1,323 charts from scientific papers are complex and stylistically diverse, testing the model’s robustness and holistic understanding when code is not feasible.

Training Details. We initialized the Qwen2.5-VL-7B model and trained it using the EasyR1 [68] framework with GRPO. The training is guided by the multi-component reward function defined in Equation 3. After tuning on a validation set, we set the weights for the final training run as follows: answer accuracy (wacc) at 0.8, decision appropriateness (wdecision) at 0.3, data fidelity (wdata) at 0.15, and format compliance (wformat) at 0.05. All prompts used are shown in Appendix E. A complete list of all other hyperparameters and implementation specifics is provided in Appendix H.

#### 5.2 Comparison with Fixed-Strategy Baselines

- Table 1: Comparison with fixed-strategy baselines on four chart understanding benchmarks. Our adaptive RL model achieves the highest average accuracy by dynamically selecting its reasoning strategy. All values are accuracy (%).

Model Type Reasoning Strategy ChartX ChartBench ChartQA CharXiv Average

Standard CoT 59.2 50.1 84.9 38.4 58.2 Code CoT (Fixed) 59.8 53.4 79.4 28.8 55.4 Adaptive 57.8 51.4 84.4 22.8 54.1

Base Models (No RL)

Standard CoT 61.5 52.8 86.6 43.8 61.2 Code CoT (Fixed) 64.0 54.0 86.7 41.9 61.7 Adaptive (Ours) 65.6 54.8 86.4 44.3 62.8

RL Models

As shown in Table 1, our adaptive framework achieves the highest average accuracy (62.8%), outperforming all fixed-strategy baselines. This advantage stems from its learned ability to dynamically select the optimal reasoning path.

- Table 2 reveals this strategic behavior. On high-programmability benchmarks like ChartX and ChartBench, our model favors the code-based path (76.0% and 66.6% usage) to leverage its precision. On the complex CharXiv benchmark, it astutely reduces code usage to just 10.1%, avoiding the pitfalls of a rigid code-only approach and achieving the highest accuracy (44.3%). The results on ChartQA further suggest that our Data Accuracy Reward improves not only when the model uses code, but also how reliably it does so.

- Table 2: Code usage percentage across benchmarks for our adaptive model versus fixed strategies. The model learns to apply code frequently on high-programmability charts and sparingly on lowprogrammability ones. All values are percentages (%).

Model Type Reasoning Strategy ChartX ChartBench ChartQA CharXiv

Base Models (No RL)

Standard CoT 0.0 0.0 0.0 0.0 Code CoT (Fixed) 98.9 100.0 98.3 99.5 Adaptive 99.7 99.6 98.8 92.9

RL Models

Standard CoT 0.0 0.0 0.0 0.0 Code CoT (Fixed) 100.0 100.0 100.0 100.0 Adaptive (Ours) 76.0 66.6 98.3 10.1

5.3 Comparison with State-of-the-Art Models

To contextualize our results, we compare our adaptive framework against several state-of-the-art (SOTA) models. All models, unless noted, were evaluated under our stringent protocol to ensure a fair comparison. As shown in Table 3, our model achieves the highest average accuracy (62.8%), significantly outperforming other SOTA models. This performance gap, especially on diverse benchmarks like ChartX and CharXiv, underscores the advantage of our adaptive reasoning approach.

- Table 3: Comparison with state-of-the-art models on four key generalization benchmarks. Our model demonstrates outstanding performance, achieving the highest average accuracy. All values are percentages (%).

Model Parameters ChartX ChartBench ChartQA CharXiv Average

ChartVLM-Large [55] 8.3B 35.0 28.8 66.7 14.7 36.3 ChartGemma [30] 3B 28.7 27.5 69.0 20.3 36.4 ChartMoE [59] 8B 33.6 29.5 74.2 28.3 41.4 Orsta-7B [27] 7B 60.3 52.0 84.6 41.5 59.6

Point-RFT [34] 7B - - 90.04† 36.02* Thyme-VL [64] 7B - - 86.1* - -

###### Ours (Adaptive) 7B 65.6 54.8 86.4 44.3 62.8

*Results are taken directly from the original paper. †In-domain evaluation result taken from the original paper.

#### 5.4 Analysis on Different Model Scales

We assess our approach on models of varying scales (3B and 32B) to test for scalability. The results, presented in Table 4. On the larger 32B model, our adaptive framework scales effectively, achieving the highest average accuracy (61.0%) and top performance on the challenging ChartX and CharXiv benchmarks.

The results from the 3B model present a more nuanced picture. While the fixed ‘Code CoT‘ strategy yields the best average performance (56.5%), we hypothesize that the adaptive strategy’s performance is constrained by the smaller model’s limitations in handling longer contexts. The adaptive prompt, which requires the model to first decide on a strategy and then execute it, is more cognitively demanding than a direct instruction. Nonetheless, it is striking that after RL, the ‘Standard

- Table 4: Performance comparison on 3B and 32B models. Our adaptive framework scales effectively to larger models, achieving the best overall performance on the 32B scale. The best results in each RL-trained category are highlighted in bold. All values are accuracy (%).

Model Size Training Reasoning Strategy ChartX ChartBench ChartQA CharXiv Average

3B

Base Model (No RL)

Standard CoT 48.0 39.2 13.8 26.7 31.9 Code CoT (Fixed) 51.3 42.0 28.0 29.3 37.7 Adaptive 1.0 0.7 0.3 10.6 3.2

RL-Trained

Standard CoT 9.3 9.3 41.8 21.3 20.4 Code CoT (Fixed) 58.5 48.5 82.3 36.7 56.5 Adaptive (Ours) 55.6 43.5 73.6 33.6 51.6

32B

Base Model (No RL)

Standard CoT 53.7 47.2 83.4 36.3 55.2 Code CoT (Fixed) 56.3 49.6 84.8 39.9 57.7 Adaptive 56.6 45.7 84.4 37.7 56.1

RL-Trained

Standard CoT 54.7 47.9 84.6 35.9 55.8 Code CoT (Fixed) 59.6 49.5 87.9 44.5 60.4 Adaptive (Ours) 60.2 48.4 87.7 47.5 61.0

CoT‘ model’s performance collapses (from 31.9% to 20.4%), while both code-based strategies see substantial gains. This indicates that our structured, code-centric reward system provides a far more stable and effective learning signal than a simple accuracy reward on free-form text.

5.5 Ablation Studies

To validate our framework’s design, we conducted ablation studies on the components of our reward function.

- 5.5.1 Dissecting the Reward Function

- Table 5: Ablation study on reward components. The full reward function is essential for achieving the highest accuracy. All values are accuracy (%).

Reward Configuration ChartX ChartBench ChartQA CharXiv Average

racc + rformat (Baseline) 62.2 52.2 86.5 43.6 61.1 + rdata (w/o rdecision) 64.3 53.5 86.4 39.4 60.9 + rdecision (w/o rdata) 63.6 52.4 86.3 43.3 61.4

Full Reward (Ours) 65.6 54.8 86.4 44.3 62.8

The results in Table 5 and 6 demonstrate the synergy between our reward components. The Decision Reward (rdecision) is essential for preventing mode collapse. Without it, the model defaults to a single strategy—either 0% or 100% code usage—leading to poor performance on certain benchmarks (e.g., a 4.2 point drop on CharXiv).

While rdecision teaches the model when to use a tool, the Data Accuracy Reward (rdata) teaches it how to use it well. Without rdata, the model becomes overly cautious on programmable charts (e.g., only 50.4% code usage on ChartX). The full reward function encourages a balanced and confident policy, leading to the best overall performance.

- Table 6: Code usage percentage in the reward ablation study. The decision reward (rdecision) is critical for preventing mode collapse and enabling adaptive behavior. All values are percentages (%).

Reward Configuration ChartX ChartBench ChartQA CharXiv racc + rformat (Baseline) 0.0 0.0 0.0 0.0

+ rdata (w/o rdecision) 100.0 100.0 100.0 100.0

#### + rdecision (w/o rdata) 50.4 11.0 87.4 0.7 Full Reward (Ours) 76.0 66.6 98.3 10.1

#### 5.5.2 The Critical Role of Numerical Fidelity

- Table 7: The stark correlation on the ChartX benchmark between the accuracy of extracted numerical data and final answer correctness. High-fidelity data extraction is demonstrably a prerequisite for success.

#### Numerical Accuracy Score (rdata) Final Answer Accuracy (%)

< 0.6 (Low Fidelity) 48.4 0.6 - 0.8 (Medium Fidelity) 60.5 > 0.8 (High Fidelity) 85.6

This analysis confirms the importance of our data accuracy reward. As shown in Table 7, there is a direct and stark correlation between the fidelity of extracted data and the final answer accuracy. High-fidelity extraction leads to an impressive 85.6% accuracy, demonstrating that correct data extraction is a prerequisite for successful reasoning on programmable charts.

Figure 6 shows that our reward function actively teaches this principle. During training, the model improves on tasks where it can extract data accurately, while it "unlearns" guessing on tasks where its data extraction is poor. This confirms that rdata is effective at grounding the model’s reasoning in factual data from the chart.

[Figure 6]

- Figure 6: Training dynamics on ChartX, illustrating the effect of the Data Accuracy Reward (rdata). (Left) Overall task accuracy increases. (Middle) Accuracy on problems with high data fidelity

(rdata > 0.6) rises sharply. (Right) Accuracy on problems with low data fidelity (rdata < 0.6) trends downward, as the model unlearns to guess.

#### 5.6 Qualitative Analysis: Knowing When to Code

To illustrate our framework’s practical intelligence, we present two contrasting cases (see Appendix G) that highlight its ability to select the optimal reasoning strategy.

- Case 1: Success on High-Programmability Tasks. On a standard stacked area chart from ChartX that required precise calculation, our adaptive model correctly chose the <CODE> path, extracting exact data and computing the correct answer. In contrast, a fixed ‘Standard CoT‘ model relied on visual estimation and failed. This shows the model’s ability to leverage code for precision.
- Case 2: Success on Low-Programmability Tasks. Faced with a complex scientific plot from CharXiv requiring qualitative comparison, a fixed code-based model failed by hallucinating a data table. Our adaptive model, however, correctly identified the task’s low programmability and chose the <DIRECT> path. It performed a robust visual comparison, leading to the correct answer and demonstrating its critical skill in avoiding tools when they are unsuitable.

### 6 Discussion and Conclusion

Our work confronts a central paradox in chart understanding: why do powerful Code-as-Thought methods that excel on structured charts often fail catastrophically on complex, "in-the-wild" visual-

izations? Our findings indicate the answer lies in a property we term Visual Programmability—the degree to which a chart’s essential information can be faithfully translated into a programmatic format. Code-as-Thought provides a decisive advantage when a chart exhibits structural transparency and the task demands high numerical precision. Conversely, it becomes actively harmful on charts with low programmability, such as scientific plots where meaning is conveyed through holistic patterns that resist symbolic decomposition. The core success of our framework is teaching a model to recognize this critical boundary. For a detailed exploration of limitations, future research directions, and the broader implications of this work, please see Appendix F.

We challenged the prevailing one-size-fits-all paradigm in visual reasoning and pivoted from seeking a single best method to developing a model that intelligently chooses the right one. By introducing Visual Programmability and training a model with a novel dual-reward system, we cultivated its ability to dynamically select between Code-as-Thought and direct visual reasoning. Our model learns to autonomously deploy Code-as-Thought for structured tasks while strategically relying on visual intuition for ambiguous ones. The key insight is that robust, general-purpose reasoning emerges not from a superior monolithic strategy, but from the meta-cognitive skill of knowing one’s own strengths and limitations. This work provides a concrete blueprint for building more flexible AI systems—systems that don’t just follow procedures, but strategically decide how to think.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Syeda Nahida Akter, Aman Madaan, Sangwu Lee, Yiming Yang, and Eric Nyberg. Self-imagine: Effective unimodal reasoning with multimodal models using self-imagination. arXiv preprint arXiv:2401.08025, 2024.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [4] Emmanuel Bengio, Pierre-Luc Bacon, Joelle Pineau, and Doina Precup. Conditional computation in neural networks for faster models. arXiv preprint arXiv:1511.06297, 2015.
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [6] Zhenfang Chen, Rui Sun, Wenjun Liu, Yining Hong, and Chuang Gan. Genome: generative neuro-symbolic visual reasoning by growing and reusing modules. arXiv preprint arXiv:2311.04901, 2023.
- [7] Zhi-Qi Cheng, Qi Dai, and Alexander G Hauptmann. Chartreader: A unified framework for chart derendering and comprehension without heuristic rules. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22202–22213, 2023.
- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [9] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in neural information processing systems, 36:49250–49267, 2023.
- [10] Yue Dai, Soyeon Caren Han, and Wei Liu. Msg-chart: Multimodal scene graph for chartqa. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pages 3709–3713, 2024.

- [11] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pages 10764–10799. PMLR, 2023.
- [12] Alex Graves. Adaptive computation time for recurrent neural networks. arXiv preprint arXiv:1603.08983, 2016.
- [13] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14953–14962, 2023.
- [14] Yucheng Han, Chi Zhang, Xin Chen, Xu Yang, Zhibin Wang, Gang Yu, Bin Fu, and Hanwang Zhang. Chartllama: A multimodal llm for chart understanding and generation. arXiv preprint arXiv:2311.16483, 2023.
- [15] Kung-Hsiang Huang, Hou Pong Chan, Yi R Fung, Haoyi Qiu, Mingyang Zhou, Shafiq Joty, Shih-Fu Chang, and Heng Ji. From pixels to insights: A survey on automatic chart understanding in the era of large foundation models. IEEE Transactions on Knowledge and Data Engineering, 2024.
- [16] Muye Huang, Lingling Zhang, Han Lai, Wenjun Wu, Xinyu Zhang, and Jun Liu. Vprochart: Answering chart question through visual perception alignment agent and programmatic solution reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 3689–3696, 2025.
- [17] Muye Huang, Lingling Zhang, Jie Ma, Han Lai, Fangzhi Xu, Yifei Li, Wenjun Wu, Yaqiang Wu, and Jun Liu. Chartsketcher: Reasoning with multimodal feedback and reflection for chart understanding. arXiv preprint arXiv:2505.19076, 2025.
- [18] Mohammed Saidul Islam, Raian Rahman, Ahmed Masry, Md Tahmid Rahman Laskar, Mir Tafseer Nayeem, and Enamul Hoque. Are large vision language models up to the challenge of chart comprehension and reasoning? an extensive investigation into the capabilities and limitations of lvlms. arXiv preprint arXiv:2406.00257, 2024.
- [19] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.
- [20] Daniel Kahneman. Thinking, fast and slow. macmillan, 2011.
- [21] Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In International Conference on Machine Learning, pages 18893–18912. PMLR, 2023.
- [22] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.
- [23] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping languageimage pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022.
- [24] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.
- [25] Fangyu Liu, Julian Martin Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, and Yasemin Altun. Deplot: One-shot visual language reasoning by plot-to-table translation. arXiv preprint arXiv:2212.10505, 2022.
- [26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

- [27] Yan Ma, Linge Du, Xuyang Shen, Shaoxiang Chen, Pengfei Li, Qibing Ren, Lizhuang Ma, Yuchao Dai, Pengfei Liu, and Junjie Yan. One rl to see them all: Visual triple unified reinforcement learning. arXiv preprint arXiv:2505.18129, 2025.
- [28] Ahmed Masry, Parsa Kavehzadeh, Xuan Long Do, Enamul Hoque, and Shafiq Joty. Unichart: A universal vision-language pretrained model for chart comprehension and reasoning. arXiv preprint arXiv:2305.14761, 2023.
- [29] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.
- [30] Ahmed Masry, Megh Thakkar, Aayush Bajaj, Aaryaman Kartha, Enamul Hoque, and Shafiq Joty. Chartgemma: Visual instruction-tuning for chart reasoning in the wild. arXiv preprint arXiv:2407.04172, 2024.
- [31] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.
- [32] Fanqing Meng, Wenqi Shao, Quanfeng Lu, Peng Gao, Kaipeng Zhang, Yu Qiao, and Ping Luo. Chartassisstant: A universal chart multimodal language model via chart-to-table pre-training and multitask instruction tuning. arXiv preprint arXiv:2401.02384, 2024.
- [33] Nitesh Methani, Pritha Ganguly, Mitesh M Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In Proceedings of the ieee/cvf winter conference on applications of computer vision, pages 1527–1536, 2020.
- [34] Minheng Ni, Zhengyuan Yang, Linjie Li, Chung-Ching Lin, Kevin Lin, Wangmeng Zuo, and Lijuan Wang. Point-rft: Improving multimodal reasoning with visually grounded reinforcement finetuning. arXiv preprint arXiv:2505.19702, 2025.
- [35] OpenAI. Gpt-5 system card. Technical report, OpenAI, 2025. PDF available; Accessed: 2025-08-11.
- [36] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [37] Rameswar Panda, Chun-Fu Richard Chen, Quanfu Fan, Ximeng Sun, Kate Saenko, Aude Oliva, and Rogerio Feris. Adamml: Adaptive multi-modal learning for efficient video recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7576–7585, 2021.
- [38] Aaron Parisi, Yao Zhao, and Noah Fiedel. Talm: Tool augmented language models. arXiv preprint arXiv:2205.12255, 2022.
- [39] Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. Gorilla: Large language model connected with massive apis. Advances in Neural Information Processing Systems, 37:126544–126565, 2024.
- [40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [41] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.
- [42] Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. Dynamic routing between capsules. Advances in neural information processing systems, 30, 2017.

- [43] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539– 68551, 2023.
- [44] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [45] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.
- [46] Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, et al. Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617, 2025.
- [47] Sanjay Subramanian, Medhini Narasimhan, Kushal Khangaonkar, Kevin Yang, Arsha Nagrani, Cordelia Schmid, Andy Zeng, Trevor Darrell, and Dan Klein. Modular visual question answering via code generation. arXiv preprint arXiv:2306.05392, 2023.
- [48] Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11888–11898, 2023.
- [49] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [50] Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J Zico Kolter, Louis-Philippe Morency, and Ruslan Salakhutdinov. Multimodal transformer for unaligned multimodal language sequences. In Proceedings of the conference. Association for computational linguistics. Meeting, volume 2019, page 6558, 2019.
- [51] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.
- [52] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697, 2024.
- [53] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [54] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.
- [55] Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Peng Ye, Min Dou, Botian Shi, et al. Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning. arXiv preprint arXiv:2402.12185, 2024.
- [56] Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy Lin. Deebert: Dynamic early exiting for accelerating bert inference. arXiv preprint arXiv:2004.12993, 2020.
- [57] Yichang Xu, Gaowen Liu, Ramana Rao Kompella, Sihao Hu, Tiansheng Huang, Fatih Ilhan, Selim Furkan Tekin, Zachary Yahn, and Ling Liu. Language-vision planner and executor for text-to-visual reasoning. arXiv preprint arXiv:2506.07778, 2025.
- [58] Zhengzhuo Xu, Sinan Du, Yiyan Qi, Chengjin Xu, Chun Yuan, and Jian Guo. Chartbench: A benchmark for complex visual reasoning in charts. arXiv preprint arXiv:2312.15915, 2023.

- [59] Zhengzhuo Xu, Bowen Qu, Yiyan Qi, Sinan Du, Chengjin Xu, Chun Yuan, and Jian Guo. Chartmoe: Mixture of diversely aligned expert connector for chart understanding. arXiv preprint arXiv:2409.03277, 2024.
- [60] Cheng Yang, Chufan Shi, Yaxin Liu, Bo Shui, Junjie Wang, Mohan Jing, Linran Xu, Xinyu Zhu, Siheng Li, Yuxiang Zhang, et al. Chartmimic: Evaluating lmm’s cross-modal reasoning capability via chart-to-code generation. arXiv preprint arXiv:2406.09961, 2024.
- [61] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.
- [62] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.
- [63] Fatemeh Pesaran Zadeh, Juyeon Kim, Jin-Hwa Kim, and Gunhee Kim. Text2chart31: Instruction tuning for chart generation with automatic feedback. arXiv preprint arXiv:2410.04064, 2024.
- [64] Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, Haonan Fan, Kaibing Chen, Jiankang Chen, Haojie Ding, Kaiyu Tang, Zhang Zhang, Liang Wang, Fan Yang, Tingting Gao, and Guorui Zhou. Thyme: Think beyond images, 2025.
- [65] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023.
- [66] Shitian Zhao, Haoquan Zhang, Shaoheng Lin, Ming Li, Qilong Wu, Kaipeng Zhang, and Chen Wei. Pyvision: Agentic vision with dynamic tooling. arXiv preprint arXiv:2507.07998, 2025.
- [67] Xuanle Zhao, Xianzhen Luo, Qi Shi, Chi Chen, Shuo Wang, Zhiyuan Liu, and Maosong Sun. Chartcoder: Advancing multimodal large language model for chart-to-code generation. arXiv preprint arXiv:2501.06598, 2025.
- [68] Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, and Yuwen Xiong. Easyr1: An efficient, scalable, multi-modality rl training framework. https://github. com/hiyouga/EasyR1, 2025. GitHub repository.

### A Detailed Analysis of Fixed-Strategy Experiments

Experimental Setting. To create our specialist model, we fine-tuned Qwen2.5-VL-7B using a Supervised Fine-Tuning (SFT) approach on the ChartX validation set [55]. This dataset consists of approximately 4,800 highly structured charts well-suited for programmatic analysis. We then evaluated this specialized model’s generalization ability across four diverse test suites, each containing 500 samples designed to span a spectrum of difficulty and style:

- • In-Domain (ChartX [55]): A stratified sample from the official test set, ensuring equal representation of chart types (e.g., bar, line, pie). This measures performance on data from the same distribution as the training set.
- • Near-Domain (ChartBench [58]): A similarly stratified sample from ChartBench. This benchmark, while out-of-domain (OOD), shares structural and stylistic similarities with ChartX, testing for near-transfer capabilities.
- • Far-Domain (ChartQA [29]): A random sample from the human-annotated portion of the test set. These examples often require deeper, qualitative reasoning, posing a rigorous challenge to purely quantitative methods.
- • Far-Domain (CharXiv [52]): A random sample from CharXiv, which contains "in-thewild" scientific charts with significant visual complexity and stylistic diversity. This serves as a stress test for generalization.

This multi-faceted evaluation was designed to reveal how a strategy optimized for clean, structured data would perform when confronted with the ambiguities and complexities of real-world visualizations.

Table 8: Detailed performance of "One-Size-Fits-All" Strategies. This table provides the full numerical data visualized in Figure 2 in the main text. All models are fine-tuned (SFT or RL) on the ChartX validation set. The CPT model first undergoes continued pre-training on Chart2Code160k [67] to enhance its core chart-to-code ability. Despite optimization, no single strategy excels across all benchmarks, revealing a fundamental performance trade-off.

Prompt Strategy Training Method ChartX ChartBench ChartQA CharXiv Average

Base Model 59.8 51.6 80.4 38.2 58.7 SFT on ChartX 69.8 56.2 72.0 37.2 58.8 RL on ChartX 63.0 53.0 81.6 39.4 59.3

Standard CoT

Base Model 62.6 53.4 74.8 29.8 55.2 SFT on ChartX 71.6 56.8 68.2 18.4 53.8 RL on ChartX 66.6 55.8 78.0 37.0 59.4 CPT + RL on ChartX 69.2 54.0 68.6 32.0 56.0

Code-as-Thought

Detailed Analysis. The results in Table 8 reveal a sharp dichotomy in generalization performance. The code-based specialist (SFT, Code-based CoT) excelled on structured data, achieving an impressive 71.6% on ChartX. However, this rigid strategy proved brittle when generalized, with accuracy plummeting on complex charts like CharXiv to just 18.4%. This shows how reasoning patterns effective for simple charts become detrimental when misapplied. Furthermore, this failure is not a simple matter of competence that can be fixed with more training. Optimizing the policy with reinforcement learning (RL) or maximizing coding skill on a vast dataset (CPT + RL) failed to resolve this core conflict.

### B Case Study: Failure due to Numerical Hallucination

As discussed in Section 3, a critical failure mode for rigid, code-based strategies is numerical hallucination. This occurs when the model incorrectly perceives the visual information in a chart and generates flawed code based on this misperception. The model then proceeds to execute its own flawed logic, leading to an answer that is logically consistent with its internal (wrong) representation but factually incorrect.

[Figure 7]

- Figure 7: Failure of a Rigid Code-Based Strategy on a CharXiv Example. The model is tasked with analyzing the original chart (a) from the CharXiv dataset. It generates Python code (indicated in the red box) to extract the data, but this code hallucinates an incorrect data structure. Chart (b) is the visualization produced by executing the model’s flawed code. The model then faithfully reasons over its own erroneous chart (b) to arrive at the answer ’8’, a stark deviation from the ground truth of 26. This case exemplifies how a rigid code-based approach can fail by building logical conclusions on a foundation of numerical hallucination.

### C Data Accuracy Reward Implementation

The Data Accuracy Reward (rdata) is a critical component for ensuring that the model’s generated code is not only syntactically correct but also faithfully extracts the data from the chart. This reward is calculated by comparing the DataFrame generated by the model’s code against a ground-truth CSV. The full process is detailed in Algorithm 1.

The COMPAREVALUES function is designed to be robust. For numerical values, it uses a relative tolerance of 10−2 to handle minor extraction or floating-point discrepancies. For textual values, it performs case-insensitive, normalized string matching. It also correctly handles NaN values, returning true only if both values are NaN.

### D Annotation Framework for Visual Programmability

To train a model capable of recognizing Visual Programmability, we developed a rigorous annotation framework grounded in expert human judgment. We chose this approach because the boundary between visual and symbolic representation is fundamentally cognitive; it involves nuanced, tacit knowledge that is difficult to capture with purely algorithmic rules.

#### D.1 Guiding Principle

We built our methodology around a single, functional question for annotators: "Does a code-based representation preserve the essential information required to correctly answer this question?" This principle ensures that every label is context-aware, reflecting how the task depends on both the chart’s properties and the user’s specific query.

#### D.2 Assessment Criteria

Annotators evaluated each chart-question pair using a two-step assessment designed to mirror the decision-making process we want our model to learn.

• Primary Assessment: Information Preservation. The core question was whether the chart’s essential information could be faithfully translated into code. Annotators considered if the underlying data could be reliably extracted from visual elements (e.g., bar heights, point positions) and if this programmatic format would retain everything needed to answer the question. If critical information was lost in this translation—such as the meaning conveyed

Algorithm 1 Data Accuracy Reward Computation Require: Generated code cpred, Ground truth CSV csvgt Ensure: Data accuracy reward rdata

- 1: Extract DataFrame construction code from cpred using AST parsing
- 2: DFpred ← CONSTRUCTDATAFRAME(extracted_data)
- 3: DFgt ← PARSECSV(csvgt)
- 4: if DFpred is None or DFgt is None then
- 5: return 0.0
- 6: end if
- 7: ▷ Column Completeness Score
- 8: matched_cols ← 0
- 9: for each column cref in DFgt do
- 10: cnormref ← NORMALIZE(cref) ▷ Remove spaces, lowercase
- 11: best_match ← FUZZYMATCH(cnormref , DFpred.columns)
- 12: if match_score > 50 then
- 13: matched_cols ← matched_cols + 1
- 14: end if
- 15: end for
- 16: rcol ← matched_cols / len(DFgt.columns)
- 17: ▷ Row Completeness Score
- 18: rrow ← ⊮[len(DFpred) = len(DFgt)]
- 19: ▷ Value Accuracy Score
- 20: total_accuracy ← 0
- 21: compared_cols ← 0
- 22: for each matched column pair (cpred,cgt) do
- 23: correct_values ← 0
- 24: for each row i in min(len(DFpred), len(DFgt)) do
- 25: if COMPAREVALUES(DFpred[cpred][i], DFgt[cgt][i]) then
- 26: correct_values ← correct_values + 1
- 27: end if
- 28: end for
- 29: col_accuracy ← correct_values / num_comparisons
- 30: total_accuracy ← total_accuracy + col_accuracy
- 31: compared_cols ← compared_cols + 1
- 32: end for
- 33: rvalues ← total_accuracy / compared_cols
- 34: ▷ Combined Data Accuracy Score
- 35: rdata ← 0.2 · rcol + 0.1 · rrow + 0.7 · rvalues
- 36: return rdata

by complex annotations, visual metaphors, or specific color gradients—the instance was marked as having low programmability for that task.

• Secondary Assessment: Reconstruction Feasibility. As a practical test, annotators performed a "mental compilation." They envisioned how the chart might be programmatically recreated using a standard plotting library like Matplotlib. If key visual elements or context could not be captured in this hypothetical reconstruction, it served as a strong signal for low programmability.

- D.3 Annotation Process and Quality Control To ensure the quality and consistency of our dataset, we followed a structured process.

- • Binary Categorization. For practical model training, we classified each instance into one of two categories: high programmability (suitable for code-based reasoning) or low programmability (requires direct visual reasoning). This binary choice frames the model’s learning objective as a clear, decisive action.

- • Systematic Guidelines. All annotations were guided by a detailed rulebook. In ambiguous or boundary cases, annotators were instructed to be conservative, prioritizing the integrity of the visual information over forcing a programmatic representation.
- • Quality Assurance. We regularly reviewed batches of annotated samples to ensure adherence to our guidelines. This iterative validation process helped maintain high levels of consistency and quality throughout the dataset.

By grounding our dataset in this human-centric process, we provide our model with a supervisory signal that reflects the nuances of human cognition. This enables it to learn a flexible, adaptive policy for chart understanding that moves beyond the limitations of rigid, rule-based systems.

### E Detailed Prompt Specifications

This section details the key prompts used for data generation and model training. The prompt for generating synthetic question-answer pairs is presented in Prompt E. The baseline prompts for direct Chain-of-Thought and mandatory code-based reasoning are shown in Prompt E and Prompt E, respectively. Finally, the master prompt that guides our adaptive model to learn strategy selection is detailed in Prompt E.

#### Prompt for Synthetic Question-Answer Pair Generation

You are a specialized generator of chart comprehension questions. Using (i) a chart graphic and (ii) the Python code that creates it, formulate **one** question with its correct answer. ### Guidelines

- 1. Answers must come from chart observation and code understanding
- 2. Provide exactly one brief, precise response with **no additional details**
- 3. Avoid multiple choice, yes/no, or lengthy descriptive formats
- 4. Emphasize questions requiring data interpretation expertise
- 5. Keep answers short (numbers, percentages, names, dates, or brief terms) ### Question Categories #### **Numerical Operations**

- - **Counting Tasks**: Enumerate items, groups, or elements with properties
- - **Basic Mathematics**: Addition, subtraction, multiplication, division
- - **Descriptive Statistics**: Average, median, mode, range, maximum, minimum
- - **Ratio Analysis**: Proportional relationships between categories
- - **Conditional Analysis**: Elements meeting specific requirements
- - **Multi-step Problems**: Combined computational operations #### **Object Recognition**
- - **Ranking Identification**: Highest or lowest performing entities
- - **Peak Value Location**: Items with extreme measurements
- - **Group Classification**: Category membership identification
- - **Time-based Analysis**: Performance identification across periods
- - **Benchmark Comparison**: Items relative to specific standards #### **Comparison Tasks**
- - **Head-to-head Analysis**: Direct comparison between entities
- - **Position Ranking**: Order determination in sequences
- - **Variation Analysis**: Largest differences between items #### **Temporal Analysis**
- - **Trend Identification**: Increase/decrease periods
- - **Change Detection**: Significant transition moments
- - **Pattern Analysis**: Cyclical or seasonal behaviors ### Answer Types

- - **Numeric**: ‘92‘, ‘4.2‘, ‘17%‘
- - **Monetary**: ‘$2,100‘, ‘£1,400‘
- - **Names**: ‘Samsung‘, ‘India‘, ‘2022‘
- - **Categories**: ‘Transportation‘, ‘Media‘
- - **Time**: ‘August‘, ‘Q4‘, ‘2018‘
- - **Ratios**: ‘3:5‘, ‘1.7‘ ### Output Format ‘‘‘json {{"question": "Question text here", "answer": "Short answer"}} ‘‘‘ ### Task

**Chart Image**: <image>

**Python Code**: {python_files} Develop one JSON question-answer pair.

Baseline Chain-of-Thought (CoT) Prompt

Carefully examine this chart. Based on your observations, answer the question. Let’s reason step by step, then put your final answer under format \\boxed{}.

Code-based Chain-of-Thought (Code-CoT) Prompt

You must carefully examine the chart and the question. First redraw the image using Python code. This code should aim to focus on data accuracy and basic chart type representation. The code must be runnable. Before any plotting, import pandas and construct one ‘pandas.DataFrame‘ named ‘chart_data‘ that contains all raw numerical data you will use. The DataFrame must include appropriate column names and keep the original row order. Then describe your step-by-step thought process and answer the question using a single word or phrase and put it under format \\boxed{}.

Master Prompt for the Adaptive Reasoning Framework

You are an expert at analyzing charts and answering questions about them. You have two powerful approaches, with code-based analysis being your preferred method when applicable. ## Core Principle Code-based analysis is highly effective and should be your first choice when charts contain extractable data. Code provides precision, reproducibility, and often superior accuracy compared to visual estimation.

## Approach Selection Examples

- **Example 1:** Bar chart with clear axis labels and readable values

- - Question: "What’s the average value across all bars?"
- - Best Choice: <CODE> (Perfect for extracting values and calculating precisely)

- **Example 2:** Complex 3D visualization or heavily artistic infographic

- - Question: "What trend does this show?"
- - Best Choice: <DIRECT> (Data extraction would be unreliable here)

## When to Use Each Approach ### Use <CODE> When Charts Are Analyzable:

- - **Any Standard Chart**: Bar, line, pie, scatter, histogram - even if slightly messy
- - **Readable Data Points**: If you can see numbers or estimate from gridlines - **use code!**
- - **Clear Structure**: Regular patterns, axes, legends - perfect for code extraction
- - **Questions Needing Precision**: Calculations, comparisons, trends - code gives exact answers
- - **Moderate Complexity**: Don’t avoid code just because extraction takes effort - be brave! ### Use <DIRECT> Only When Code Is Truly Impractical:
- - **Extremely Artistic/Stylized**: Heavy design elements completely obscure data structure
- - **No Readable Scale**: Completely missing or unintelligible axes
- - **Pure Qualitative**: Questions only about general patterns, not specific values
- - **Severely Distorted**: 3D effects or perspectives that make extraction impossible ## **Decision Framework**

- **Step 1: Code Preference Check**

- - Can I see any numerical data or gridlines? → **TRY <CODE>**
- - Are there clear bars, lines, or data points? → **TRY <CODE>**
- - Would precise calculations help answer this question? →

**TRY <CODE>**

- **Step 2: Only if Step 1 fails**

- - Is the chart purely artistic with no extractable structure?

→ Use <DIRECT>

- - Is the question purely qualitative? → Use <DIRECT>

## Response Format

**First, make your choice with confidence:**

- - For code-assisted analysis: output <CODE>
- - For direct analysis: output <DIRECT>

### If Using Code-Assisted Analysis (<CODE>):

**Start with**: <CODE> Then proceed with your analysis using code as helpful. Before any coding, import pandas and construct one ‘pandas.DataFrame‘ named ‘chart_data‘ that contains all raw numerical data you will use. The DataFrame must include appropriate column names and keep the original row order. You may:

- - Redraw/recreate the chart data for comprehensive analysis
- - Use code for calculations, comparisons, or data processing
- - Combine visual observations with computational analysis
- - Focus on the most relevant chart elements for the question

- *Note: Choose the code approach that best fits the chart and question - full redrawing, partial extraction, or targeted calculations.* ### If Using Direct Analysis (<DIRECT>):
- **Start with**: <DIRECT> Then provide your reasoning and analysis in the most effective way for the question. Consider:

- - Key observations and findings from the chart
- - Your reasoning process and logical steps
- - Relevant patterns or trends you identify

## Final Answer Format Every response MUST end with \\boxed{your_answer}

Now analyze the given chart and question. Choose your approach based on the chart’s extractability and the question’s requirements.

### F Broader Implications and Future Directions

Our work on adaptive chart reasoning, while focused on a specific domain, offers insights into a broader challenge in artificial intelligence: developing systems that can flexibly navigate between different problem-solving strategies. Just as humans alternate between rapid, intuitive pattern recognition and slower, deliberate symbolic reasoning [20], future AI systems must master not only individual skills but also the meta-level ability to select the right tool for the job.

From Modality Fusion to Method Fusion. Much of the research in multimodal AI has centered on modality fusion—the effective combination of information from different sensory channels. Our framework points towards a complementary and perhaps equally important paradigm: method fusion. This refers to the ability to select and combine different reasoning strategies (e.g., visual-perceptual vs. symbolic-programmatic) even when operating within a single modality. The challenge is not only to perceive the world through multiple senses but to think about it through multiple "lenses," fluidly shifting between holistic, pattern-based analysis and precise, step-by-step decomposition as the problem demands.

Competence Awareness as a Foundational Capability. A key takeaway from our research is that models can be trained to recognize the boundaries of their own competence with respect to specific methods. This nascent form of meta-cognitive awareness—knowing not just how to solve a problem, but knowing which of its available methods is most likely to succeed—is a fundamental prerequisite for robust and reliable AI. We foresee that future general-purpose systems will need to develop richer internal models of their own capabilities, enabling them to make more dependable strategy selections when faced with novel tasks.

Limitations and Key Future Directions. While our adaptive framework represents a significant step, its current limitations highlight critical areas for future research that build directly upon our findings.

- • Granular and Hybrid Reasoning: The current decision-making process is a binary choice between "code" and "direct" reasoning. This could be extended to a more granular hybrid model, where code is used for reliable data extraction while visual reasoning concurrently interprets qualitative patterns from the same chart. Furthermore, assessing programmability at the chart-level is coarse; future models could learn to perform region-based assessment, applying different strategies to different parts of a single complex figure.
- • Expanding the Vocabulary of Formal Reasoning: Our model’s "Code as Thought" process is currently centered on data analysis logic. A natural evolution is to expand the scope of this native formal reasoning. Instead of orchestrating external tools, future work could enrich the model’s internal symbolic language to encompass other formalisms, such as the logic of signal processing for time-series charts or graph-theoretic principles for network diagrams. This would extend the reach of the model’s innate symbolic capabilities, allowing it to tackle a wider range of problems programmatically without breaking the "native reasoning" paradigm.
- • Self-Supervised Policy Learning: A key challenge is reducing the reliance on annotated training data for programmability. A promising direction is developing self-supervised methods where the model learns the decision boundary by correlating its choice of strategy

with final task success. This would effectively teach the model to recognize the reliable application range of its own internal methods without requiring explicit human-provided labels.

Towards Dynamic Strategy Composition. Looking further ahead, our current framework makes a discrete selection between predefined strategies. A significant extension would be for future systems to dynamically compose novel strategies from a set of primitive cognitive operations. For instance, when analyzing a complex visualization, an advanced system might synthesize a hybrid approach on the fly: invoking its internal graph-based logic for structural analysis while using its time-series forecasting logic for temporal patterns. This compositional flexibility, guided by a learned meta-policy, would represent a significant leap towards more human-like adaptability.

The Path Forward. The journey from narrow tools to general intelligence will likely require architectural innovations that foster cognitive flexibility. Our adaptive framework, though applied to chart understanding, provides a concrete instantiation of these principles. By teaching a model to recognize when formal reasoning is a powerful asset versus a brittle liability, we take a tangible step toward systems that reason not just powerfully, but appropriately. The ultimate goal is not to build models that always default to their most complex methods, but ones that can gracefully match their computational effort and reasoning style to the structure of the problem at hand—a hallmark of true intelligence.

### G Detailed Case Studies

This section provides the full visualizations and detailed model outputs for the qualitative analysis presented in Section 5.6. Each case includes the figure, the task details, and the verbatim model outputs from both a baseline and our adaptive model.

[Figure 8]

##### Figure 8: Case Study 1: High-Programmability Chart from ChartX.

[Figure 9]

##### Figure 9: Case Study 2: Low-Programmability Chart from CharXiv.

### H Implementation and Hyperparameter Details

Our model was trained using the configuration and hyperparameters summarized in Table 9. We used the EasyR1 [68] framework for our reinforcement learning implementation. The base model, Qwen2.5-VL-7B, was trained for 200 episodes. The vision tower of the model remained frozen during training to preserve its pre-trained perceptual capabilities.

Table 9: Training Configuration Details

Configuration Value Model Configuration Base Model Qwen2.5-VL-7B Vision Tower Frozen Precision BFloat16 Max Prompt Length 5,120 tokens Max Response Length 3,072 tokens Data Configuration Seed 42 Shuffle True Filter Overlong Prompts True Training Hyperparameters Algorithm GRPO (without KL penalty) Learning Rate 1.0 × 10−6 Optimizer AdamW (BF16 variant) Global Batch Size 64 Rollout Batch Size 256 Micro Batch Size (Update) 4 Micro Batch Size (Experience) 16 Training Episodes 4 Gradient Clipping 1.0 Rollout Configuration Number of Rollouts (n) 5 Temperature 1.0 Top-p 0.99 Infrastructure GPUs 8 × NVIDIA H800 Tensor Parallelism 1 FSDP Enabled CPU Offloading Disabled Gradient Checkpointing Enabled Validation Validation Batch Size 512 Validation Frequency Every 5 episodes Validation before Training Yes

