arXiv:2506.22992v1[cs.AI]28Jun2025

# MARBLE: A Hard Benchmark for Multimodal Spatial Reasoning and Planning

Yulun Jiang1,2 Yekun Chai2 Maria Brbić1 Michael Moor2

1 EPFL 2 ETH Zurich

https://marble-benchmark.github.io July 1, 2025

##### Abstract

The ability to process information from multiple modalities and to reason through it step-by-step remains a critical challenge in advancing artificial intelligence. However, existing reasoning benchmarks focus on text-only reasoning, or employ multimodal questions that can be answered by directly retrieving information from a non-text modality. Thus, complex reasoning remains poorly understood in multimodal domains. Here, we present MARBLE, a challenging multimodal reasoning benchmark that is designed to scrutinize multimodal language models (MLLMs) in their ability to carefully reason step-by-step through complex multimodal problems and environments. MARBLE is composed of two highly challenging tasks, M-Portal and M-Cube, that require the crafting and understanding of multistep plans under spatial, visual, and physical constraints. We find that current MLLMs perform poorly on MARBLE—all the 12 advanced models obtain near-random performance on M-Portal and 0% accuracy on M-Cube. Only in simplified subtasks some models outperform the random baseline, indicating that complex reasoning is still a challenge for existing MLLMs. Moreover, we show that perception remains a bottleneck, where MLLMs occasionally fail to extract information from the visual inputs. By shedding a light on the limitations of MLLMs, we hope that MARBLE will spur the development of the next generation of models with the ability to reason and plan across many, multimodal reasoning steps.

## 1 Introduction

Human reasoning is inherently multimodal and sequential—integrating modalities such as language or vision as context to draw conclusions through structured, step-by-step thought. While LLMs have made significant strides in step-by-step reasoning [27, 11, 8, 18], the multimodal reasoning abilities of Multimodal LLMs (MLLMs) are still in their infancy and not yet well understood. Achieving complex, multi-step, multimodally grounded reasoning is critical for building intelligent systems that can generalize across domains and interact adaptively with complex environments.

Recent benchmarks – such as ScienceQA [14], MathVista [16], and MMMU [30] – have shown that MLLMs can solve tasks involving both visual and linguistic understanding. However, these benchmarks often emphasize relatively shallow forms of reasoning, such as single-step question answering or factual retrieval. They frequently conflate perception (e.g., interpreting an image or diagram) with reasoning (e.g., drawing logical inferences, comparing evidence, or crafting a multi-step plan), reducing complex reasoning to pattern matching and multimodal integration. As a result, current evaluations underexplore and undermeasure an MLLM’s capacity for deep, structured reasoning. Moreover, the recent literature has focused heavily on abstract reasoning in domains such as advanced mathematics or code generation, where multimodal embodiment plays a limited role. In contrast, interacting with and planning in spatially and physically constrained environments

Table 1: Conceptual overview of the MARBLE benchmark.

Dataset Description Modality Subtasks # Samples Metrics

M-Portal Solving complex multimodal spatial reasoning and planning problems.

Text, Image Plan correctness, Fill-the-blanks

F1-Score, Accuracy

512 512

M-Cube Assembling 3D Cube from six jigsaw pieces.

Text, Image CUBE, CUBE-easy

1,000 1,000

Accuracy

is a fundamental dimension of human intelligence but it is largely missing from today’s MLLM evaluations. While a recent effort introduced an escape room-inspired benchmark [26], frontier models were not sufficiently challenged by its task complexity, achieving up to 100% escape rate. Thus, hard benchmarks that stress multi-step planning and spatial reasoning under physical constraints remain an open need. Analogous to how difficult challenges have historically driven progress, we believe that an ARC-like test [6] for multimodal reasoning could spark foundational advances in MLLM capabilities.

In this work, we present MARBLE (MultimodAl Reasoning Benchmark for Language modEls), a highly challenging multimodal reasoning benchmark specifically designed to evaluate step-bystep, multimodally grounded reasoning in MLLMs. Our benchmark introduces tasks that are cognitively demanding, requiring models to decompose complex multimodal prompts into interpretable intermediate steps, align information across inputs, and to carefully craft a multi-step plan to solve complex problems under diverse spatial and physical constraints. Unlike prior datasets that overemphasize final-answer accuracy, our benchmark emphasizes reasoning trajectories and plans, providing both gold-standard rationales and mechanisms for evaluating intermediate step fidelity. MARBLE consists of two main tasks, M-Portal which tests complex spatial reasoning and planning abilities inspired by the puzzle video game Portal 2, and M-Cube, which tests the ability to understand and assemble 3D jigsaw pieces into a target cube shape, inspired by the Happy Cube puzzle. Each dataset also contains two subtasks at different difficulty levels, as shown in Table 1.

We conduct an extensive evaluation of MARBLE across 12 state-of-the-art MLLMs and reasoning models. Intriguingly, all the prominent models obtain near-random performance on M-Portal and 0% accuracy on M-Cube. Even in simplified configurations, only about half of the models are able to outperform the random baseline. Notably, GPT-o3 [18] is the only model demonstrating reasonable performance on easier tasks, achieving 17.6% on the simpler M-Portal subtask and 72.0% accuracy on CUBE-easy. These results indicate that complex multimodal reasoning remains a significant challenge for current MLLMs. Our further analysis shows that perception is still a bottleneck for multimodal reasoning: all the advanced MLLMs completely fail to understand and extract structured information from the visual inputs. Additionally, we present an interactive setup for M-Cube where the model iteratively refines its response based on the feedback from a solution validator tool, reflecting the real-world problem-solving processes. We hope that MARBLE will serve as a probing benchmark to reveal the limitations of current MLLMs and drive the development of next-generation models with stronger capabilities in multi-step multimodal reasoning and planning.

## 2 Related work

Chain-of-Thought and multimodal reasoning paradigms. The Chain-of-Thought (CoT) prompting paradigm has significantly advanced reasoning in language models by enabling stepwise decomposition of complex problems [27]. The Multimodal Chain-of-Thought (MCoT), its extension to the multimodal domain, represents a natural progression, encouraging models to articulate intermediate reasoning steps while integrating multiple modalities such as images, text, and diagrams. Recent works like [25] highlight prompt-based, plan-based, and learning-based MCoT strategies, yet

also underscore the lack of robust, diagnostic benchmarks tailored to multimodal reasoning.

Recent multimodal instruction tuning approaches fine-tune LLMs augmented with visual encoders to follow multimodal prompts [13, 33]. While these models can generate fluent outputs, their reasoning often lacks depth or consistency, particularly on tasks involving spatial, numerical, or abstract visual patterns [30, 5].

Multimodal reasoning benchmarks. Several datasets have been proposed to evaluate multimodal reasoning, such as ScienceQA [14], MMMU [30], MathVista [15], EMMA [9] and MEGABench [4]. These benchmarks span academic knowledge domains and require integrating visual and textual information. However, they often prioritize answer accuracy over the evaluation of the full reasoning trace, making it difficult to diagnose model errors. Others, like PuzzleVQA [5] and NLVR [28], introduce abstract reasoning challenges but are limited in modality diversity and stepwise supervision. Recent works like Critic-V [31] and MMIR [29] introduced frameworks for multimodal inconsistency detection or critic-guided refinement, which improved performance but was limited to rather shallow reasoning paths.

There are few previous benchmarking approaches that leveraged multimodal tasks inspired by video game puzzle environments [32, 19, 24]. Most recently and closely related, [26] proposed MM-Escape, an escape-room like environment where MLLMs have to navigate and leverage the surroundings (e.g., retrieving a hidden key) in order to escape a room. While this benchmark shares some similarity with the M-Portal task in MARBLE, M-Portal introduces a novel and much harder, multi-step problem solving challenge. To illustrate this, consider GPT-4o model which solved 70−100% of the maps in MM-Escape, but performed very poorly on M-Portal (e.g., 4.1% accuracy on fill-the-blanks).

## 3 MARBLE: a benchmark for multimodal spatial reasoning and planning

We present MARBLE, a challenging game-inspired multimodal reasoning benchmark designed to evaluate the complex reasoning abilities of multimodal LLMs (MLLMs). In contrast to prior reasoning benchmarks that evaluate only the final answer independent of the reasoning trace, MARBLE focuses on assessing the correctness of the reasoning process itself. MARBLE consists of two tasks, MPortal and M-Cube, both of which require complex, multi-step and multimodal reasoning skills to forge an appropriate plan that accounts for complex spatial and physical problem constraints. The M-Portal task challenges MLLMs to solve problems derived from Portal 2 videogame with multi-step reasoning and planning. The M-Cube evaluates MLLMs in their ability to solve Happy Cube puzzles, i.e., rotate complex shapes to arrange them into 3D cubes under physical constraints.

### 3.1 M-Portal

The M-Portal task is a multimodal reasoning task that involves planning, spatial reasoning, as well as multimodal integration. M-Portal is inspired by the game Portal 2, a first-person perspective puzzle videogame released by Valve in 2011. Portal challenges players to overcome obstacles and to pass through rooms by means of placing two portals through which players can teleport. A key mechanic in Portal is the conservation of momentum: when a player enters one portal with a given velocity, they exit the second portal with the same relative momentum. This enables creative traversal strategies, such as jumping across large gaps or over obstacles, by combining gravity-driven falls with portal placement. Various additional features (e.g., buttons, lasers, tractor beams, liquids) add further complexity to the puzzle environments. The ultimate trial will be for MLLMs to interactively navigate and solve the game. However, to enable broad accessibility and usability of this benchmark, we abstract a given map into a set of visual question-answering tasks that require the MLLM to

Portal 2: Complex multi-step problem solving Solution:

[Figure 1]

- Step 1: Place portals in positions a, b and jump down into b to get ejected from a to press the button c.

- Step 2: Button c releases a cube to land on button d which activates the bridge e.

- Step 3: Place portals in positions f, g to walk across the bridge towards the cube at location d.

- Step 4: Pick up the cube and step on button d which also activates the downwards pushing tractor beam at location h.

- Step 5: Throw the cube down to the device at i that catapults it over to the target area.

- Step 6: The tractor beam intercepts the cube and pushes it on the slot j which opens the (blue) exit door and elevates a platform at location k.

- Step 7: Place portals in positions l, a, walk through l, walk across k to reach the exit.

- b
- c

h

- j
- k

a

- d
- e

g

f

l

i

[Figure 2]

[Figure 3]

Step 2 Step 6

- Figure 1: Overview of the M-Portal Dataset of the MARBLE Benchmark. Illustrated is a rather basic level Portal 2 problem, which only requires seven steps to solve. For comparison, the advanced problems introduced in this benchmark may involve several dozens of steps. Also, steps are not always decomposed into their most atomic form to keep enough complexity within a step to make mistaken steps harder to detect. Appendix A provides more examples.

integrate several depictions of the map, a textual instruction to the map, in order to examine partial or complete chain of thought (CoT) solution plans that may consist of dozens of steps. Figure 2 gives an introductory overview of how a basic portal map could look like, displaying a scene overview (top left), the step-by-step solution, and a few in-game screenshots.

M-Portal consists of 1024 problems that comprise two types of evaluations, plan correctness and fill-the-blanks, each contributing to 512 problems.

Problem statement. Given an input X = (I,T), where I is a set of multimodal inputs (e.g., screenshots of a Portal map or textual contextualization of the environment) and T is a task instruction, the objective is to generate a Chain-of-Thought (CoT) plan P = (s1,s2,...,sn) consisting of interpretable, physically sound reasoning steps that, if executed, would successfully solve the problem. The reward of a plan R(P) is 1 if the exit door is passed, and 0 otherwise. Then the objective is to evaluate the ability of models to implement the mapping F∗ that maximizes the reward, i.e.,

F∗ = arg max

EX∼D [R(F(X))], where (1) F : X  → P = (s1,s2,...,sn). (2)

F

###### Portal 2 map Solution

Mistakes

Screenshots

[Figure 4]

- Step 1

- Step 2

- Step 3

- Step 4

- Step 5

- Step 6

[Figure 5]

- a
- b
- c

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Step 2

h

- j
- k

- d
- e

- Step 4

- Step 5

g

Dataset

f

l

i

generation

Map instruction

|You enter a room that has four platforms<br><br>separated by water.<br><br>The exit is on the right.<br><br>[…]|
|---|

…

…

Step n

Human annotation

Plan correctness

Candidate steps

Fill-the-blanks

- A: Step 2
- B: Step 4

- C: Step 5

- Step 4 Step 2
- D:
- E:

For 5 mistakes, create 2^5 = 32

Step 1

candidate CoTs. For each of candidate:

Masked solution

…

Evaluation

- Step 3

- Step 4

- Step 5

- Step 6

|Is this candidate the correct solution? (yes / no)<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>+ Map<br><br>instruction|
|---|

Step 1

|Which steps need to be filled?<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>+ Map<br><br>instruction|
|---|

Step 3

…

Step n

Correct solution: no

Step n

Correct solution: A, D, C

- Figure 2: Data generation and evaluation pipeline for the M-Portal task. The top row illustrates how a given Portal 2 map (sourced from the community test chambers) was analyzed with human annotation to produce a set of illustrative screenshots that fully depict the map, textual map instructions, a ground-truth solution chain of thought (CoT), as well as a set of five mistaken steps. The steps are designed to operate independently so that mistakes and correct steps can be easily combined. The bottom row indicates two evaluation types of M-Portal: first, plan correctness, a binary evaluation where candidate solutions have to be rated as correct or wrong. Second, a fill-the-blanks evaluation, where multiple steps of the ground truth CoT solution are masked, and multiple options are available to fill in at the right place.

Data collection. For data collection, a human annotator with advanced Portal 2 experience browsed through top-rated maps from the Portal 2 community test chambers. We focused on the community test chambers, as they were often self-contained, well-defined problems in a single room. The annotator selected 16 high-quality maps that received top user-rating, while being compactly shaped such that they would be amenable to capture within a few screenshots. Figure 2 gives an overview of how the M-Portal dataset was created in the top row, whereas the bottom row indicates the evaluation strategies employed in the M-Portal task.

Evaluation subtasks. Since direct execution and success validation in the Portal environment would depend on a closed-source game environment and could involve a brittle interfacing and limited accessibility, we focus on evaluating the ability of a model to reason about the correctness of candidate plans or the missing steps in incomplete plans. For this, we consider two types of closed-ended evaluations: plan correctness and fill-the-blanks tasks.

- i) Plan correctness: Is the provided candidate plan correct?

Plan correctness is the binary classification task and requires answering yes/no questions. It is a harder task compared to fill-the-blanks because models have to carefully review lengthy candidate plans that may be dozens of steps long and involve various spatial and physical constraints and dependencies. These candidates may contain no mistake at all up until five mistaken steps. This

task has a significant class imbalance, as one Portal map with five available mistaken steps allows the creation of 25 = 32 candidates that leverage individual mistakes, whereas only one out of 32 candidates is correct.

- ii) Fill-the-blanks: Can the model accurately identify several missing steps given surrounding context and a few candidate options?

On the easier fill-the-blanks task, models receive a partial plan to solve the Portal map whereas several steps are masked. To fill the missing steps, the model needs to choose five correct options from five mistake or distracting options in a correct order. Even though this task is hard for a naive random baseline, for a model that is able to interpret the multimodal inputs X as well as the partial solution, it should be easier to identify the correct missing steps especially since mistaken steps also appear in their correct version as highly similar options. Furthermore, fill-the-blanks can also be seen as a simplification as it helps the model focus its attention on a few relevant steps out of a large sequence, whereas in the binary evaluation any step could be potentially mistaken.

### 3.2 M-Cube

Problem statement. The M-Cube task is a 3D spatial puzzle inspired by the Happy Cube, a mechanical puzzle originally invented by Dirk Laureyssens in 1986. In this task, one is presented with 6 jigsaw-style pieces taken from the faces of a 5 × 5 × 5 Cube. Each piece is featured by the bump and gap pattern on its edges. The goal is to assemble the pieces into a valid cube where the edges are aligned seamlessly without gap or overlap. To solve the M-Cube task, an MLLM needs to assign each piece into a cube face with proper orientation, i.e., to rotate and/or flip the piece accordingly to align with other pieces. For each problem, an MLLM must account for 6! possible piece-to-face assignments (modulo rotational symmetries), and for each piece, 8 discrete states of rotations and flips, resulting in a combinatorial explosion of candidate solutions. Among the vast search space, only very few solutions are valid given the geometric constrains imposed by the interlocking bump and gap patterns. András et al. [1] reported that most commercially available cubes have only one solution (up to rotational equivalences), making this a challenging reasoning problem.

Data generation. While the M-Cube tasks are inspired by the Happy Cube puzzle, we generate all samples synthetically. Figure 3 gives an overview of the workflow. Specifically, the data generation pipeline starts with a 5 × 5 × 5 cube and disassembles the surface into 6 interlocking pieces. Each piece can be regarded as a 5×5 grid, where the center 3×3 region is always preserved. For remaining cells located on the edges, we randomly assign each cell to one of the adjacent faces of the big 5×5×5 cube, to create the bump and gap patterns along the boundary. After that, the obtained pieces are shuffled and rendered from a random 3D viewpoint as the input to an MLLM. We interactively selected viewpoint ranges such that the shape was clearly discernible. Concretely, we render the objects by sampling a camera elevation in the range of –155° to –115° and an azimuth in the range of –150° to –90°, relative to the canonical front view. The base view corresponds to an elevation of –135° and an azimuth of –120°, with uniformly random perturbations of ±20° and ±30°, respectively. h

Solution validator. The model is required to find the correct piece-to-face mapping and the orientation of 6 pieces. However, for each problem, there is no unique solution since a cube contains 24 rotational symmetries. Therefore, instead of directly comparing the answer to ground-truth, we provide a solution validator by testing whether the solution from MLLM could successfully assemble the pieces into a perfect cube. Beside binary evaluation, the solution validator could also identify the conflicts in a given configuration, such as mismatched edges. This diagnostic feedback can be used by an MLLM to iteratively refine its solution. See Appendix A for example.

Data object MLLM

Sampling of pieces

Cube Rendered problem

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

{‘left’: [[…]] , ’top’: [[…]] , … }

[Figure 28]

[Figure 29]

[Figure 30]

Assembly plan

Solution validator

[Figure 31]

[Figure 32]

[Figure 33]

- Figure 3: Overview of the M-Cube workflow including data generation, problem rendering, as well as solution validation. Appendix A provides more dataset examples.

Evaluation subtasks. To measure the performance of MLLMs with controlled difficulty level, we create two subtasks called CUBE and CUBE-easy. Each subtask contains 1000 examples. CUBE-easy is a simplified version of CUBE along three axes: i) the input pieces are represented as 2D arrays instead of the rendered image to reduce the perception error of MLLM (see the discussion in Section 3.5 for more details); ii) each puzzle is specially designed such that the solution does not require flipping of any pieces; iii) a partial solution with the arrangement of 4 pieces is provided in the prompt, leaving only 2 missing pieces to be placed. Consequently, ii) and iii) significantly reduce the size of search space. In comparison, CUBE retains the full complexity of the task, where the MLLM needs to understand the input images, and explore over all the possible arrangements of the 6 pieces.

### 3.3 Evaluated models

We evaluate performance on the MARBLE benchmark using eight state-of-the-art MLLMs, including both open-source and closed-source models with advanced multimodal reasoning capabilities. Specifically, we assess three representative open-source MLLMs: Qwen2.5-VL-72B [3], InternVL3-78B [34] and Llama-4-Scout [17], alongside six closed-weight models: GPT-4o [10], GPT-o3 [18], GPT-o4mini [18], Claude-3.7-Sonnet [2] Gemini-2.5-pro [7], and Seed1.5-VL [22]. In addition, we also include three text-only models DeepSeek-R1 [8], DeepSeek-R1-0528 and Qwen3 [3] in the evaluation. We remove or manually convert the input images into textual descriptions to evaluate the models that only takes text inputs. All the experiment configurations, prompts and hyperparameters are detailed in the Appendix B. Experiments are conducted on a single node server with 8 Nvidia H200 GPUs.

### 3.4 Results on M-Portal

Overall performance. We evaluate state-of-the-art MLLMs on the plan correctness and fill-the-blanks tasks of the M-Portal, as reported in Table 2. On the plan correctness task, all investigated models (MLLMs as well as text-only LLMs) performed very poorly with a minority class F1 score of around 6%, similar to the random baseline. In the easier fill-the-blanks task, 8 out of 12 models outperform the random baseline. In particular, the performance gap compared to the random baseline is substantial (≥ 5%) for DeepSeek-R1, Claude-3.7-Sonnet, DeepSeek-R1-0528, Gemini-2.5-pro and GPT-o3 that significantly outperforms all other models. Still, even the best performing model, GPT-o3, manages to correctly solve only 17.6% of the problems. Note that although the fill-the-blanks task results in random baseline scores, it is expected to be easier than the plan correctness task for models capable of interpreting the multimodal inputs and leveraging the partial solution.

Token usage. On both tasks, the number of output tokens of reasoning models is significantly larger than those of open-source models, e.g., Gemini-2.5-pro spends on average 9.2 thousand tokens

- Table 2: Performance of state-of-the-art MLLMs on the M-Portal tasks. Models are evaluated on two types of closed-ended tasks: the plan correctness and the fill-the-blanks tasks. We report F1-score for plan correctness and accuracy for fill-the-blanks. We also report the average output token for each model. Standard deviation is reported in Appendix B. *All the visual inputs are removed for text-only LLMs.

Plan correctness Fill-the-blanks Models F1 (%) Tokens (k) Acc (%) Tokens (k) Random 6.1 - 3e-3 Qwen3-235B-A22B* 0.0 11.7 0.0 9.5 InternVL3-78B 6.4 0.1 0.0 0.1 Qwen2.5-VL-72B 6.6 0.2 0.2 0.1 Llama-4-Scout 6.5 0.3 0.2 0.5 GPT-4o 6.5 0.2 0.4 0.1 GPT-o4-mini 0.0 0.2 3.1 1.5 Seed1.5-VL 7.6 0.6 3.5 1.6 DeepSeek-R1* 6.1 2.5 5.5 7.6 Claude-3.7-Sonnet 6.3 1.1 6.8 1.9 DeepSeek-R1-0528* 0.0 4.1 8.4 11.3 Gemini-2.5-pro 4.7 5.3 16.1 9.2 GPT-o3 6.6 0.8 17.6 3.6

for one question of fill-the-blanks. Moreover, the model tends to think more on fill-the-blanks tasks compared to the plan correctness task. We hypothesize that this is related to the question format.

Influence of blanks. In the fill-the-blanks task on M-Portal, each question contains multiple steps in the complete solution, and part of them are masked. To systematically understand the impact of missing information, we construct a series of questions where the model is asked to fill n blanks from 2n candidate options. We evaluate the performance of Qwen2.5-VL-72B and the result is shown in Figure 4. Notably, the model obtains around 70% accuracy when only a single blank is present. However, the performance declines rapidly as the number of blanks increases, dropping to less than 1% when n ≥ 4, which indicates the challenges of the subtask under the conditions of extensive missing information.

Qwen2.5-VL-72B

60

Accuracy(%)

40

20

0

1 2 3 4 5 6

# Blanks

Figure 4: The influence of number of blanks to M-Portal.

Results with hint images. To better understand the M-Portal benchmark, we also collect one or more hint images for 14 out of 16 maps in the dataset. These images illustrate key insight to solve each map (see Appendix A for example). We then evaluate the MLLM’s performance when provided with the hint images. For the experiment, we use Seed1.5-VL, as it obtains the highest F1 score on plan-correctness. The result shows a slight improvement on the plan-correctness, increasing the F1 score from 7.6% to 8.6%, while the performance of fill-the-blanks remains unchanged. These results suggest that even with an additional visual context, M-Portal continues to pose a significant challenge for multimodal reasoning models.

### 3.5 Results on M-Cube

In this section, we first evaluate state-of-the-art MLLMs on the CUBE and CUBE-easy tasks of the M-Cube. After that, we disentangle M-Cube into two factors, perception and reasoning, and conduct comprehensive experiments to understand the challenges of M-Cube. Perception denotes the process of understanding visual inputs while reasoning refers to the process of searching the valid solution from the huge search space. Our results show that both are bottleneck of the current MLLMs. Finally, we show that MLLMs could also use the solution validator as a tool to gather feedback and refine its response for solving the complex reasoning problems.

Overall performance. The results on the CUBE and CUBE-easy tasks of M-Cube are shown in

- Table 3. Intriguingly, all the advanced MLLMs completely fail on the harder subtask CUBE and obtain 0% accuracy despite more than 10,000 tokens spent on thinking the problems. The results highlight the complex multimodal reasoning process involved in CUBE, where the model has to iterate over verification and backtracking through a long reasoning chain to make a final answer. In comparison, on the simplified CUBE-easy task, 6 out 12 frontier models are able to perform better than random guess. Among them, GPT-o3 achieves a remarkable performance of 72.0% accuracy, substantially outperforming the second best models GPT-o4-mini, which only reaches 16%. Despite being simplified, the number of reasoning tokens spent on CUBE-easy is still the same or a bit higher than that of CUBE, suggesting that CUBE-easy is already a challenging task for most existing MLLMs. Interestingly, for some models (GPT-4o, GPT-o3 and GPT-o4-mini), the token usage of CUBE is significantly lower than CUBE-easy. We hypothesize this might due to the visual inputs of CUBE resulting in less in-depth reasoning for these models.

- Table 3: Model performance on the M-Cube. The tasks are evaluated in an open-ended fashion: a given problem may have multiple valid solutions, thus we create a solution validator to test the validity of assembly plans that are proposed by the model. We report accuracy in percentage points for both tasks. Random guess is estimated as the ratio of valid solutions to the total search space. Standard deviation is reported in Appendix B. *Result is obtained by converting visual input into 2D arrays in text.

CUBE CUBE-easy Models Acc (%) Tokens (k) Acc (%) Tokens (k) Random 1e-5 - 3.1 Qwen3-235B-A22B* 0.0 7.2 0.3 16.1 Llama-4-Scout 0.0 0.7 1.6 1.2 Qwen2.5-VL-72B 0.0 0.3 2.0 0.8 GPT-4o 0.0 0.2 2.0 0.6 Seed1.5-VL 0.0 3.9 2.0 16.6 InternVL3-78B 0.0 0.1 2.8 1.0 Claude-3.7-Sonnet 0.0 13.2 7.4 13.2 DeepSeek-R1-0528* 0.0 21.3 8.0 21.3 Gemini-2.5-pro 0.0 27.8 11.0 28.4 Deepseek-R1* 0.0 16.3 14.0 17.3 GPT-o4-mini 0.0 1.6 16.0 11.0 GPT-o3 0.0 1.9 72.0 21.0

[Figure 34]

[Figure 35]

Single Jigsaw-style piece

Question: Convert the image into a 5x5 array, where 0 = gap and 1 = bump …

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

###### MLLM response:

[Figure 40]

[[1, 1, 1, 0, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 0, 1, 1, 1]]

[Figure 41]

[Figure 42]

[Figure 43]

- Figure 5: Perception remains a bottleneck for M-Cube. Left: A perception task designed to test MLLM’s ability on retrieve structured information from visual input (full prompt in Appendix A) and example response of an MLLM. Right: Performance of 8 MLLMs on this perception task based on 200 test examples. Accuracy is measured both at individual cells and for the entire 5 × 5 piece. All the MLLMs perform poorly and completely fail on the full-piece accuracy.

Error on perception. To solve the M-Cube puzzle, the first step is to understand the visual input and retrieve the relevant information, which serves as the basis of the reasoning steps afterwards. Thus, we design a perception task to measure whether the MLLMs could correctly extract information from the input image: given a jigsaw-style piece in a 3D viewpoint, the model is asked to convert the piece into a 5 × 5 array, as shown in Figure 5. We evaluate all the 8 MLLMs on this perception task with 200 test examples, and report the accuracy on cells and accuracy of the whole piece also on Figure 5. Surprisingly, we found all the models could only achieve around 70% accuracy per cell. The best perception performance, is 76% accuracy from Gemini-2.5-pro, meaning that the model could still occasionally make mistakes. As a result, all the models achieve 0% accuracy on the whole piece. These results highlight that even advanced MLLMs struggle with this seemingly simple perception task, posing a potential bottleneck for multimodal reasoning in complex scenarios like CUBE. Though there have been a few works discussing the shortfalls of visual capabilities of MLLMs, such as [20] and [23], it’s the first time that MLLMs have been reported to perform poorly on such simple structured perception tasks, to the best of our knowledge.

Error on reasoning. Apart from the perception errors, M-Cube still remains a highly challenging problem due to the vast search space from the combination of all possible arrangements and orientations of 6 pieces. Figure 6 illustrates the size of search space of M-Cube as a function of both the number of missing pieces and whether a solution requires flipping the pieces. In particular, CUBE comprises 6! ∗ 86 = 188,743,680 possible solutions. In comparison, CUBE-easy only contains 32 possible solutions, a 5,000,000 fold reduction of the hypothesis space. To isolate the reasoning challenge from perceptual limitation, we manually convert the visual inputs into corresponding text arrays. We then compare the performance of DeepSeek-R1 in different search space configurations, as shown in Figure 7. The model obtains 57% accuracy in the simplest setting with only one missing piece. However, the performance drops drastically as the search space expands, falling to 0% when more than 3 pieces are missing. The substantial decline underscores the difficulty of reasoning among expanding combinatorial search space, a major bottleneck for existing reasoning models. In summary, besides perception error, reasoning among the vast search space is also a challenge, making M-Cube an especially difficult task for state-of-the-art MLLMs.

Search Space of M-Cube

CUBE

108

Flip needed Flip ignored

#Possiblesolutions

106

104

102

CUBE-easy

1 2 3 4 5 6

# Missing pieces

- Figure 6: Search space of the M-Cube dataset under different configurations.

Performance Comparison

Flip needed Flip ignored

50

Accuracy(%)

40

30

20

10

0

1 2 3 4 5 6

# Missing pieces

Figure 7: Performance of DeepSeek-R1 across varying levels of task difficulty of the M-Cube dataset.

Results with solution validator. The ability to use tools or perform function calls has emerged as a crucial feature in latest MLLMs [21]. In case of M-Cube, the solution validator could serve as an auxiliary tool to assist MLLMs in tackling complex reasoning tasks. In each round, the model proposes a candidate solution and evaluates it with the solution validator. Based on the validator’s feedback, the model could iteratively refine its response towards a better solution in the next round. Specifically, we design two types of feedback: (i) Binary feedback, which simply indicates whether a solution is correct or not in a black box manner, (ii) Detailed feedback, which not only verifies the correctness of the solution but also provides diagnostic information such as which edges of the cube are in conflict. Figure 8 shows the performance of GPT-o4mini under both types of feedback. On CUBE-easy, the performance increases significantly for both binary and detailed feedback and detailed feedback consistently outperforms binary feedback, increasing the performance from 10% to up to 28% accuracy after 5 rounds of interactions, which indicates the value of diagnostic information. However, on more challenging CUBE dataset,

the performance using the solution validator tool remains 0% regardless of the feedback type, highlighting the limitation of current MLLMs in solving harder multimodal reasoning problems.

CUBE-easy with Solution Validator

Feedback Type

Binary

25

Detailed

In summary, we introduce a multi-step setup within M-Cube that enables iterative refinement through the feedback from a solution validator. This setup closely mirrors how humans tackles real-world problem-by making initial attempts, gathering feedback from the environment, and refining their strategies accordingly. However, many current reasoning models would not retain and build upon previous reasoning steps, often discarding the reasoning in earlier context1, resulting in less effective reasoning in multi-round setup. Therefore, future models capable of interleaved thinking and tool use would benefit more from such validator-assisted setup.

Accuracy(%)

20

15

10

1 2 3 4 5 Round of Conversation

Figure 8: Performance of GPT-o4-mini on CUBE-easy with binary or detailed feedback from solution validator. On CUBE, the performance will remain 0%.

1Check this OpenAI API document for example.

## 4 Discussion

This paper introduces MARBLE, a hard multimodal reasoning benchmark for MLLMs. MARBLE provides a focused testbed for evaluating MLLMs on complex spatial reasoning and planning tasks that are underlying heterogenous physical constraints. Our tasks are designed such that an MLLM must first understand the physical constraints imposed by the multimodal input, and then formulate a coherent, multi-step plan that draws from a vast search space in order to solve the problem. MARBLE fills the gap of multimodal reasoning evaluation by shifting the focus from outcome accuracy to process-oriented, multi-steps reasoning that requires coherent multimodal understanding. By contributing a challenging benchmark for multi-step, multimodal reasoning amidst spatial and physical constraints, MARBLE aspires to elicit more progress and innovation in MLLM development that will unlock unprecedented abilities in reasoning and planning amidst complex and multimodal environments—capabilities that are essential for real-world, embodied, and general-purpose intelligence.

Our empirical evaluation reveals that state-of-the-art MLLMs struggle significantly with MARBLE. They can only outperform random baselines in simplified ablations and fail even on structured perception tasks, underscoring limitations in both reasoning and visual understanding.

Limitations and future work. For ease of use, we do not explore real-time interactive settings, nor do we fine-tune or adapt models at test time. Future work should investigate interactive and adaptive approaches, enabling models to reason with and through different modalities—such as “thinking with images”—in a more compositional way.

Broader impact. As with any benchmark, there is a risk of overfitting to dataset-specific patterns. However, our setting involves abstract puzzle domains, which do not raise direct societal risks. Advancing multimodal reasoning has strong potential for positive impact in domains like healthcare, accessibility, and education. Rigorous benchmarks like MARBLE can help ensure that future systems are robust and beneficial ahead of deployment.

## References

- [1] Szilárd András, Kinga Sipos, and Anna Soós. Which is harder?-Classification of Happy Cube puzzles. 2013.
- [2] Anthropic. Anthropic 3.7 Sonnet and Claude Code, February 2025. URL https://www. anthropic.com/news/claude-3-7-sonnet.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.
- [4] Jiacheng Chen, Tianhao Liang, Sherman Siu, Zhengqing Wang, Kai Wang, Yubo Wang, Yuansheng Ni, Wang Zhu, Ziyan Jiang, Bohan Lyu, et al. MEGA-Bench: Scaling multimodal evaluation to over 500 real-world tasks. arXiv preprint arXiv:2410.10563, 2024.
- [5] Yew Ken Chia, Vernon Toh Yan Han, Deepanway Ghosal, Lidong Bing, and Soujanya Poria. PuzzleVQA: Diagnosing multimodal reasoning challenges of language models with abstract visual patterns. arXiv preprint arXiv:2403.13315, 2024.
- [6] Francois Chollet, Mike Knoop, Gregory Kamradt, and Bryan Landers. Arc prize 2024: Technical report. arXiv preprint arXiv:2412.04604, 2024.

- [7] Google DeepMind. Gemini 2.5: Our most intelligent ai model, March 2025. URL https://blog. google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/ #gemini-2-5-thinking.
- [8] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-R1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [9] Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can MLLMs reason in multimodality? EMMA: an enhanced multimodal reasoning benchmark. arXiv preprint arXiv:2501.05444, 2025.
- [10] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [11] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, and Ilge Akkaya. Openai o1 system card. CoRR, abs/2412.16720, 2024. doi: 10.48550/ARXIV.2412.16720. URL https://doi.org/10.48550/arXiv.2412.16720.
- [12] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
- [13] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [14] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 2022.
- [15] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

- [16] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations, 2023.
- [17] Meta. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation, April

2025. URL https://ai.meta.com/blog/llama-4-multimodal-intelligence/.

- [18] OpenAI. Introducing OpenAI o3 and o4-mini, April 2025. URL https://openai.com/index/ introducing-o3-and-o4-mini/.
- [19] Davide Paglieri, Bartłomiej Cupiał, Samuel Coward, Ulyana Piterbarg, Maciej Wolczyk, Akbir Khan, Eduardo Pignatelli, Łukasz Kuciński, Lerrel Pinto, Rob Fergus, et al. Balrog: Benchmarking agentic llm and vlm reasoning on games. arXiv preprint arXiv:2411.13543, 2024.
- [20] Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind. In Proceedings of the Asian Conference on Computer Vision, 2024.
- [21] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36: 68539–68551, 2023.
- [22] ByteDance Seed Team. Seed1.5-VL technical report. arXiv preprint arXiv:2505.07062, 2025.
- [23] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? Exploring the visual shortcomings of multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [24] Oguzhan Topsakal, Colby Jacob Edell, and Jackson Bailey Harper. Evaluating large language models with grid-based game competitions: an extensible llm benchmark and leaderboard. arXiv preprint arXiv:2407.07796, 2024.
- [25] Yaoting Wang, Shengqiong Wu, Yuecheng Zhang, Shuicheng Yan, Ziwei Liu, Jiebo Luo, and Hao Fei. Multimodal chain-of-thought reasoning: A comprehensive survey. arXiv preprint arXiv:2503.12605, 2025.
- [26] Ziyue Wang, Yurui Dong, Fuwen Luo, Minyuan Ruan, Zhili Cheng, Chi Chen, Peng Li, and Yang Liu. How do multimodal large language models handle complex multimodal reasoning? placing them in an extensible escape game. arXiv preprint arXiv:2503.10042, 2025.
- [27] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 2022.
- [28] Anne Wu, Kianté Brantley, and Yoav Artzi. A surprising failure? multimodal llms and the NLVR challenge. arXiv preprint arXiv:2402.17793, 2024.
- [29] Qianqi Yan, Yue Fan, Hongquan Li, Shan Jiang, Yang Zhao, Xinze Guan, Ching-Chen Kuo, and Xin Eric Wang. Multimodal inconsistency reasoning (mmir): A new benchmark for multimodal reasoning models. arXiv preprint arXiv:2502.16033, 2025.
- [30] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

- [31] Di Zhang, Jingdi Lei, Junxian Li, Xunzhi Wang, Yujie Liu, Zonglin Yang, Jiatong Li, Weida Wang, Suorong Yang, Jianbo Wu, et al. Critic-V: VLM critics help catch vlm errors in multimodal reasoning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9050–9061, 2025.
- [32] Xiangxi Zheng, Linjie Li, Zhengyuan Yang, Ping Yu, Alex Jinpeng Wang, Rui Yan, Yuan Yao, and Lijuan Wang. V-mage: A game evaluation framework for assessing visual-centric capabilities in multimodal large language models. arXiv preprint arXiv:2504.06148, 2025.
- [33] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In The Twelfth International Conference on Learning Representations.
- [34] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

## A Illustration of Example Problems

### A.1 M-Portal

Figure 9 gives an extended overview of the M-Portal problem. It introduces a simple example problem, created for illustrative purposes and does not cover the full complexity the benchmark. Each map in M-Portal requires a sequence of actions to solve, making it a complex multimodal reasoning problem.

Figure 10 shows a challenging example problem of the M-Portal task of MARBLE. Figure 10 shows input images and instruction text that describe the problem. A manually curated solution is shown on the right side, together with five mistaken steps, below. A hint image depicts the crucial insight that allows to solve the map.

Portal 2: Complex multi-step problem solving Solution:

[Figure 44]

Step 1: Place portals in positions a, b and

jump down into b to get ejected from a to

c

press the button c.

Step 2: Button c releases a cube to land on

button d which activates the bridge e.

h

Step 3: Place portals in positions f, g to walk across the bridge towards the cube at

location d.

- j
- k

a

Step 4: Pick up the cube and step on button

- d
- e

d which also activates the downwards

g

pushing tractor beam at location h.

Step 5: Throw the cube down to the device

f

l

at i that catapults it over to the target area.

i

Step 6: The tractor beam intercepts the cube

and pushes it on the slot j which opens the (blue) exit door and elevates a platform at

location k.

Step 7: Place portals in positions l, a, walk

b

through l, walk across k to reach the exit.

[Figure 45]

[Figure 46]

Step 2 Step 3

[Figure 47]

[Figure 48]

Step 5

Step 6

[Figure 49]

[Figure 50]

Step 7 Step 7

#### Figure 9: Overview of the Portal-2 Dataset of the MARBLE-Benchmark. Illustrated is a rather basic level Portal 2 problem, which only requires seven steps to solve. For comparison, the advanced problems introduced in this benchmark may involve several dozens of steps. Also, steps are not always decomposed into their most atomic form to keep enough complexity within a step to make mistaken steps harder to detect.

Problem description

###### Problem images (excerpt)

"You enter room 1, which is connected to room 2 on the right, separated by a shield wall. Room 1 contains a button on the floor that activates a stair leading up to a platform. On this platform, there is a switch that controls a mirror cube machine located in room 2. Room 2 features a laser source that hits the wall and a laser teleportation machine. When activated by a button press, this teleportation machine sends any object placed on it (such as a cube) to the endpoint of the laser ray, wherever the laser is directed. This allows cubes to travel through shield walls that would otherwise block movement. However, teleportation does not work through solid walls. Room 2 also has a button that activates a cube machine located next to the teleportation device. Room 3 is separated from room 1 by a shield wall and contains a button that opens the door to room 4. Room 4 is a small area with only a button on the floor, which opens the exit door."

[Figure 51]

[Figure 52]

Solution

- "Step 1: Go to room 2 (on the right) and press the switch to drop a cube.",
- "Step 2: Shoot a blue portal where the laser hits the wall and one on the wall that points to the central room (room 1).",
- "Step 3: Place the cube on the laser teleportation machine and press the switch to send the cube via laser to room 1.",
- "Step 4: Go to room 1 and place the cube on the button.",
- "Step 5: Walk up the stairs to press the little button, which drops a mirror cube in room 1.",
- "Step 6: Pick up the mirror cube and place it in front of the laser source such that the laser points towards room 3.",
- "Step 7: Create a new cube by pressing the little button in room 2.",
- "Step 8: Place the new cube on the laser teleportation machine and press the button to send the cube.",
- "Step 9: Pick up the mirror cube and place it on the teleportation device.",
- "Step 10: Shoot an orange portal where the laser source hits the wall and a blue portal at the wall next to the teleportation device to direct the laser to the mirror cube which needs to point to room 3.",
- "Step 11: Activate the teleportation machine by pressing the button next to the machine.",
- "Step 12: Go to room 3, pick one cube, and place it on the button to open the door to room 4. Take the other cube and bring it to room 4, placing it on the button on the floor to open the exit door.",
- "Step 13: Go through the exit door. Problem solved."

[Figure 53]

[Figure 54]

[Figure 55]

Mistakes

"Step 2: Shoot a blue portal where the laser hits the wall and an orange portal on the same wall close to the boundary to room 1 such that the cube gets sent to room 1.",

Hint image

[Figure 56]

- "Step 5: Go to room 2 and collect the mirror cube who dropped due to the button press in room 1.",
- "Step 6: Pick up the mirror cube and place it in front of the laser source such that the laser points towards room 2.",

"Step 10: Shoot an orange portal where the laser source hits the wall and a blue portal at the wall of the entrance in room 1, such that the laser points to room 3.",

"Step 12: Go to room 3, pick one cube, and place it on the button of room 4 to open the door in room 4. Take the other cube and placing it on the button of room 3, now both doors are open."

- Figure 10: Illustration of an example problem of the M-Portal dataset (problem 5), composed of a problem description, images, solution steps, mistakes, and optional hint images.

### A.2 M-Cube

#### Figure 11 presents a complete example question of M-Cube task, and the solution to the instance with the corresponding 2D and 3D visualization. Figure 12 shows the prompt of the perception task.

You are a spatial-reasoning assistant that solves Happy Cube puzzles. INPUT

- • Six jigsaw-style pieces, labelled A–F.
- • Each piece is described by the bump/gap pattern on its four edges (Edge1, Edge2, Edge3 and Edge4) as shown in the image, the center squares are always filled.

TASK Build a 5 × 5 × 5 cube that uses every piece exactly once.

- • Assign face (Top, Bottom, Front, Back, Left, Right) with proper orientation (rotation and/or flip).
- • The finished cube closed perfectly—no overlaps, no gaps. The first and last 'square' of the edge contribute to the corners of the cube, make sure:
- • When two cube faces touch, the non-corner part of their edges must be complementary bit-for-bit.
- • At cube corners (where three faces meet), all connecting edges each piece to one cube must align seamlessly.

NOTE For each piece, report the orientation via the following mental exercise:

- 1. Hold the finished cube in both hands.
- 2. Turn the whole cube until the face you are about to name is now facing you like the front of a box.
- 3. In this view, write down the edge numbers located on left and top.

• Format: (Cube-Face, Left-Edge#, Top-Edge#) Note that left-edge and top-edge must be adjacent. OUTPUT Answer the question with the arrangement of 6 pieces. Your response should end with: The final solution is ```

- A: (Cube-Face, Left-Edge#, Top-Edge#)
- B: (Cube-Face, Left-Edge#, Top-Edge#)
- C: (Cube-Face, Left-Edge#, Top-Edge#)
- D: (Cube-Face, Left-Edge#, Top-Edge#)
- E: (Cube-Face, Left-Edge#, Top-Edge#)
- F: (Cube-Face, Left-Edge#, Top-Edge#) ```

2D Visualization 3D Visualization

[Thinking…] SOLUTION

| | | | | | | | | | |[Figure 57]| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |

- A: (Back, 2, 1),
- B: (Bottom, 1, 2),
- C: (Top, 1, 2),
- D: (Right, 3, 2),
- E: (Left, 3, 2),
- F: (Front, 4, 3)

Figure 11: Illustration of M-Cube Problem. Top: Example input image and prompt of the problem. Bottom: Example solution to the problem (left) and corresponding 2D and 3D visualization (right). The visualization is not part of the inputs or outputs of the benchmark.

You are given an image of a 5 x 5 grid. In the grid, each cell on the edges is randomly preserved or dropped, while the center 3 x 3 region is always preserved. Now convert the input image into a 2D array, where 0 = gap and 1 = bump, and ensure edge1 = left, edge2

= top, edge3 = right, edge4 = bottom in the 2D array. You should answer with "Here is the converted 2D array: [array]" where [array] is a 2D array in the format of Python list of lists.

Figure 12: Prompt for evaluating the perception ability of MLLMs on M-Cube.

The solution validator of M-Cube can serve as an auxiliary tool to assist MLLM in solving the reasoning problems. Given a candidate solution, the solution validator could determine whether the solution is correct or not (binary feedback). In addition, it can also provide diagnostic information such as edge conflicts (detailed feedback). Figure 13 illustrates an example where the MLLM leverages feedback from the validator to iteratively refine its solution.

Question

[Instruction]...

[Thinking]... The final solution is: A: (Back, 2, 1) B: (Left, 1, 4) C: (Top, 2, 1) D: (Right, 3, 2) E: (Bottom, 3, 4) F: (Front, 4, 3)

MLLM

The solution is incorrect! Edge conflicts (5):

- • Edge at top-back: Top face (piece C – edge 2) is not compatible with Back face (piece A - edge 2)
- • Edge at top-front: Top face (piece C – edge 4) is not compatible with Front face (piece F - edge 2)
- • Edge at top-left: Top face (piece C – edge 1) is not compatible with Left face (piece B - edge 2)
- • Edge at top-right: Top face (piece C – edge 3) is not compatible with Right face (piece D - edge 2)
- • Edge at bottom-back: Bottom face (piece E – edge 4) is not compatible with Back face (piece A - edge 4)

Solution validator

[Thinking]... The final solution is:

- A: (Back, 2, 1)
- B: (Bottom, 1, 2)
- C: (Top, 1, 2)
- D: (Right, 3, 2)
- E: (Left, 3, 2)
- F: (Front, 4, 3

MLLM

The solution is correct!

Solution validator

#### Figure 13: Example of MLLM using solution validator as a tool to gather feedback and iteratively refine its response on the M-Cube dataset.

## B Experiment Details.

- Table 4 provides a comprehensive list of all the models evaluated oin this paper, along with the hyperparameters. We use the same hyperparameters for evaluating both the M-Portal and M-Cube tasks. For open-source models such as Qwen2.5-VL-72B, InternVL3-78B and Llama-4-Scout, we use vLLM [12] for efficient inference, with a setting of temperature of 0 and maximum output token length of 16,000 for all the models. The open-source models are evaluated on the whole evaluation suite of M-Cube and M-Portal.

In contrast, close-source models such as GPT-4o, Claude-3.7-Sonnet, Gemini-2.5-pro, GPT-o3 and GPT-4o-mini are evaluated with their respective APIs. The "reasoning effort" parameter, which controls the allowed length of reasoning chain, is set to "medium" for GPT-4o-mini and Gemini-2.5Pro, and 12,000 for Claude-3.7 Sonnet. Due to the limit of budget, we choose 200 representative examples on M-Cube and the whole set of M-Portal for evaluating close-source models.

The prediction of a reasoning model can vary significantly on different random seed. Due to the budget constraints, we do not re-run each experiment multiple times to directly measure the variance. Instead, we report standard deviation estimated by bootstrapping, as shown in Table 5.

- Table 4: MLLMs and corresponding hyperparameters for evaluating MARBLE benchmark. “Reasoning effort” represents the budget of reasoning tokens to generate before the final response. * For reasoning models, max tokens denote the sum of tokens generated for reasoning and final response.

Model Date Temperature Reasoning Effort Max Tokens*

Qwen2.5-VL-72B 2025.02.19 0.0 - 16,000 InternVL3-78B 2025.04.11 0.0 - 16,000 Llama-4-Scout 2025.04.05 0.0 - 16,000 Qwen3-235B-A22B 2025.04.29 0.6 - 16,000 GPT-4o 2024.08.06 0.0 - 16,000 DeepSeek-R1 2025.01.22 - - 16,000 DeepSeek-R1-0528 2025.05.28 - - 16,000 Seed-1.5-VL 2025.04.28 - - 16,000 Claude-3.7-Sonnet 2025.02.19 - 12,000 16,000 Gemini-2.5-pro 2025.05.06 - medium 25,000 GPT-o4-mini 2025.04.16 - medium 25,000 GPT-o3 2025.04.16 - medium 40,000

- Table 5: Results of M-Portal and M-Cube datasets, reported with standard deviation (± STD) estimated via bootstrapping.

Plan-correctness Fill-the-blanks CUBE CUBE-easy Models F1(%) ± STD Acc(%) ± STD Acc(%)± STD Acc(%)± STD

Qwen3-235B-A22B 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.3 ± 0.2 Llama-4-Scout 6.5 ± 1.7 0.2 ± 0.2 0.0 ± 0.0 1.6 ± 0.4 Qwen2.5-VL-72B 6.6 ± 1.6 0.2 ± 0.2 0.0 ± 0.0 2.0 ± 0.4 GPT-4o 6.5 ± 1.5 0.4 ± 0.3 0.0 ± 0.0 2.0 ± 1.4 Seed1.5-VL 7.6 ± 5.4 3.5 ± 0.8 0.0 ± 0.0 2.0 ± 1.4 InternVL3-78B 6.4 ± 1.7 0.0 ± 0.0 0.0 ± 0.0 2.8 ± 0.5 Claude-3.7-Sonnet 6.3 ± 1.6 6.8 ± 1.1 0.0 ± 0.0 7.4 ± 2.7 DeepSeek-R1-0528 0.0 ± 0.0 8.4 ± 1.2 0.0 ± 0.0 8.0 ± 2.7 Gemini-2.5-pro 4.7 ± 4.4 16.1 ± 1.7 0.0 ± 0.0 11.0 ± 3.1 Deepseek-R1 6.1 ± 4.1 5.5 ± 1.0 0.0 ± 0.0 14.0 ± 3.4 GPT-o4-mini 0.0 ± 0.0 3.1 ± 0.8 0.0 ± 0.0 16.0 ± 3.7 GPT-o3 6.6 ± 3.1 17.6 ± 1.7 0.0 ± 0.0 72.0 ± 4.5

