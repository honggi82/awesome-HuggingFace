# arXiv:2309.05689v1[cs.CL]11Sep2023

## Large Language Model for Science: A Study on P vs. NP

Qingxiu Dong ∗ 1 2 Li Dong ∗ 1 Ke Xu ∗ 3 Guangyan Zhou4 Yaru Hao1 Zhifang Sui2 Furu Wei1 https://aka.ms/GeneralAI

##### Abstract

In this work, we use large language models (LLMs) to augment and accelerate research on the P versus NP problem, one of the most important open problems in theoretical computer science and mathematics. Specifically, we propose Socratic reasoning, a general framework that promotes in-depth thinking with LLMs for complex problem-solving. Socratic reasoning encourages LLMs to recursively discover, solve, and integrate problems while facilitating self-evaluation and refinement. Our pilot study on the P vs. NP problem shows that GPT-4 successfully produces a proof schema and engages in rigorous reasoning throughout 97 dialogue turns, concluding “P ̸= NP”, which is in alignment with (Xu and Zhou, 2023). The investigation uncovers novel insights within the extensive solution space of LLMs, shedding light on LLM for Science.

#### 1 Introduction

Large language models (LLMs) have exhibited remarkable abilities across various tasks, including natural language generation, coding, and reasoning (OpenAI, 2023; Bubeck et al., 2023). While LLMs have successfully interpolated existing knowledge, it remains open whether they can extrapolate novel knowledge from the vast solution space. The fundamental question pertains to the potential of LLMs to achieve expert-level capabilities.

In this work, we propose a paradigm, named LLM for Science (LLM4Science), that integrates LLMs to augment and accelerate the scientific research process. In previous work (Wang et al., 2023), AI usually serves as support tools to carry out procedures predetermined by humans. By contrast, the new paradigm elevates LLMs to the role of a collaborative peer alongside humans, leveraging the creativity of AI. Besides, unlike task-specific models tailored for particular problems (such as AlphaFold; Jumper et al. 2021), LLMs act as general-purpose innovation navigators. The polymath ability of LLMs, in terms of both breadth and depth of knowledge, enables AI to combine skills and ideas in novel and useful ways.

We conduct a pilot study on the P versus NP problem (Cook, 2000), which is a pivotal open question in computer science and mathematics. The problem concerns the classification of computational problems based on their solvability. It investigates whether NP problems with quickly verifiable solutions can be solved as efficiently as P problems, which are solvable using deterministic polynomial-time algorithms. Despite its substantial impact on other areas, such as cryptography and optimization, the P vs. NP problem has remained an elusive

*Equal contribution 1 Microsoft Research 2 School of Computer Science, Peking University 3 State Key Lab of Software Development Environment, Beihang University 4 Department of Mathematics and Statistics, Beijing Technology and Business University. Contribution during Q.D.’s internship at Microsoft Research. Contact person: Furu Wei <fuwei@microsoft.com>.

challenge for over fifty years (Fortnow, 2022). The inherent complexity and prerequisite for expert-level knowledge make it a good arena for investigating the potential of using LLMs to discover novel science. Moreover, as the problem is still open, it naturally avoids solution memorization and contamination from the training corpus of LLMs.

Inspired by the ancient Greek philosopher Socrates, we propose Socratic Reasoning, a general problem-solving framework that allows LLMs to navigate the expansive solution space and efficiently arrive at the answer. As Socrates stated, “I cannot teach anybody anything. I can only make them think.” Socratic Method employs a series of questions to stimulate critical thinking and elicit ideas through ongoing dialogues. Following the above wisdom, our framework encourages LLMs to think critically and generate solutions for complex problems. As shown in Table 1, there are five atomic prompt patterns in Socratic reasoning: deduction, transformation, decomposition, verification, and integration. They are used to uncover novel insights and perspectives, decompose complex issues into subproblems or steps, and engage in self-refinement by challenging the responses.

We utilize Socratic reasoning to tackle the P vs. NP problem. GPT-4 demonstrates the capability to develop a reasoning pathway and arrive at a conclusion in alignment with the results presented by Xu and Zhou (2023). Specifically, we introduce five distinct roles (e.g., a mathematician proficient in probability theory) as our collaborative provers. GPT4 is prompted to devise a preliminary proof schema with the initial 14 turns of dialogues. Then we proceed with rigorous reasoning in the subsequent 83 dialogue turns, following the sketch. In total, the 97 turns collectively construct extremely hard NP-complete problems, where some instances are unsolvable within a time complexity lower than O(2n) (i.e., exhaustive search). In other words, the proof concludes “P ̸= NP”. Our investigation highlights the potential capability of GPT-4 to collaborate with humans in exploring exceptionally complex and expert-level problems. We show that, beyond mastering fundamental knowledge, LLMs can discover novel insights within the extensive solution space. The revelation sheds light on the prospect of scientific discoveries in the paradigm of LLM for Science.

#### 2 Socratic Reasoning: Stimulating LLMs for Complex Problem Solving

We introduce a general-purpose problem-solving framework, called Socratic Reasoning. Drawing inspiration from Socratic Method employed by ancient Greek philosophers, we utilize a sequence of questions to stimulate critical thinking and elicit ideas of LLMs through continuous dialogues. The framework aims to motivate LLMs to orchestrate various subproblems when solving highly complicated tasks, guiding LLMs to establish a high-level reasoning pathway. Socratic reasoning is conducted in a series of dialogue turns between humans and LLMs, functioning as a recursive mechanism for resolving intricate challenges with LLMs.

As shown in Table 1, Socratic Reasoning includes five types of prompt patterns: deduction, transformation, decomposition, verification, and integration. Generally, when dealing with an atomic problem, which LLMs can directly reason to conclude, we employ deduction patterns (e.g., “Let’s think step by step...”) to instruct the LLMs to derive a conclusion straightforwardly. For more complex problems, we initially ask the LLMs to transform the problem into a new one or break it down into several subproblems. We then pursue such patterns recursively until reaching the atomic problems. When generating a new problem or arriving at a new conclusion, we apply the verification pattern to leverage the self-critique capabilities of LLMs for validation and refinement. Lastly, the integration pattern requests LLMs to synthesize conclusions based on the outcomes of subproblems. We motivate LLMs to continue the above process recursively via a series of dialogues until it addresses the target problem.

Previous work focuses on optimizing the thought process of LLM inference for relatively

###### Patterns Description Diagram Examples

Deduction

Derive a conclusion for a given problem directly.

P

|C|
|---|

|C’|
|---|

P

|C|
|---|

Based on this, can you define ... (25) Let’s calculate the ... (42) which kind of ... will guarantee ... holds true? (44) Please provide a strict proof of this point.

(93)

Transform the problem into a homogeneous or similar problem, or abstract the problem.

Can you find the fundamental problem ... from ... perspective? (1)

P P’

Transformation

Break the problem into manageable subproblems, or make a plan for reasoning steps.

Please explain the theorem to me, lay out a high-level proof plan, and then try various tactics to prove the theorem. (32) Please lay out a high-level proof plan. (63)

P P1 P3 P2

Decomposition

Verification

Check the conclusion or its relationship with others to verify or correct it.

|C|
|---|

P

Please check for these issues ... (30) Please check ... and refine any possible mistakes. (46) Does this prove ...? (51) Why do you say ...? (58)

Integration

Summarize multiple conclusions to derive a new conclusion.

|C2| |
|---|---|
| | |

|C3|
|---|

|C1|
|---|

|C|
|---|

Please now organize all our historical conversations and sort out ... (14) Now what conclusion can we draw? (79)

Table 1: Problem-solving patterns in Socratic Reasoning. We use ⃝P and

to represent (sub)problems and conclusions, respectively.

|C|
|---|

simple problems compared with scientific discovery. For instance, Wei et al. (2022) consider the reasoning steps in a model response as a chain of thoughts. Yao et al. (2023) structure reasoning as searching over a tree, achieving improved results on closed-format reasoning tasks, such as math word problems, and Game of 24. Nonetheless, how to stimulate the potential of LLMs in free-form, highly complex problems remains under-explored. In this work, Socratic reasoning provides a systematic prompting framework for challenging problems.

Indexes of past turns incorporated Turn index Input prompt into the dialogue history

[Figure 1]

…

Response generated by GPT-4

Figure 2.1: Example of a dialogue turn in Socratic Reasoning.

Figure 2.1 shows an example dialogue turn used to tackle the P vs. NP problem (Section 3.2). We used GPT-4 API (Appendix B) in the case study. We steer the roles of LLMs by customizing system instructions as in Appendix A. We sort the process according to the turn index. Because the context length of GPT-4 is 8192, we incorporate the relevant turns into context. The history indexes are shown at the top-right corner. Besides, in the margin, we summarize some best practices for prompting LLMs in Socratic reasoning. We also use italicized margin notes to explain some details. Notice that we only format GPT-4 responses to LaTeX while keeping the content unmodified. Moreover, we highlight the insightful parts of the response generated by GPT-4 for better understanding.

#### 3 A Pilot Study on the P Versus NP Problem

- 3.1 Intuitions Behind the Proof: Extreme Hardness is Easy to Understand

To study the P vs. NP problem in computation theory, it is essential first to develop a foundational comprehension of the laws of computation and their mathematical proofs. Akin to laws of physics in the natural world, laws of computation are fundamental laws that objectively exist. They are primarily expressed as mathematical theorems and are proven through mathematical approaches. Grounded in the principle that “the greatest truths are the simplest,” most fundamental laws possess elegant and straightforward forms, and their mathematical proofs are not complicated. With this intuition, it is crucial first to figure out why some computational problems are hard and provide intuitive explanations for computational hardness.

However, the reasons why some problems are hard have not been thoroughly studied. For a long time, NP-completeness has been the core concept of computational complexity theory. NP-complete problems are defined by reductions, which is instrumental in establishing the tower of computational complexity theory. Yet, reductions depend on skills and vary among problems, lacking a unified pattern. Fundamentally, reductions classify problems according to relative hardness and cannot provide intuitive explanations for why a problem is hard. Without such explanations, it is hard to determine the hardness degree of NP-complete problems. Many think that computational complexity theory is hard because it studies hardness. In our opinion, the pursuit of reductions rather than explanations is the root cause of the difficulties faced by computational complexity theory after its initial flourishing. In a nutshell, the successes or failures of computational complexity theory are all attributed to reductions.

At this point, understanding computational hardness without relying on reductions is crucial for breakthroughs. Exhaustive search signifies extreme hardness. The extreme hardness of exhaustive search, instead, can be intuitively explained - exhaustive search implies that there is no way to prune the search space when solving the problem (the solution space is almost independent). Therefore, the problem is hard to solve.

The almost independent solution space is a unique property of Model RB (Xu and Li, 2000), i.e., the intuition of choosing Model RB in 15. Intuitively, an almost independent solution space means that the state of any assignment in the solution space (whether or not it is a solution) is uncorrelated to the majority of other assignments. When the parameter d in Model RB is large enough (the intuition of letting d and n tend to infinity in 31), the solution space of Model RB is almost independent. However, most problems, such as 2-SAT or 3-SAT problems, do not possess the property of almost independent solution space. In other words, the assignments in their solution spaces are correlated to varying degrees. This difference in the degree of correlation makes it difficult to distinguish quantitatively between 2-SAT and 3-SAT. It also explains why prior attempts to distinguish between P problems and NP problems have failed, consequently complicating proving that P does not equal NP.

- 3.2 Co-Proving with GPT-4

1

Prompt: Can you find the fundamental problem behind the P!=NP problem from a philosophical perspective, rather than from a computer theory perspective?

From a philosophical perspective, the P versus NP problem raises deep questions about the nature of knowledge, intelligence, problem-solving, and the limits of human understanding. The P vs NP problem is essentially a question of whether efficient solutions exist for complex problems, or if some problems are inherently difficult to solve quickly, regardless of the method used.

The fundamental philosophical problem behind the P!=NP problem can be seen as the tension between creativity, intuition, and insight (which are often associated with NP problems) versus the systematic, algorithmic, and deterministic approaches (which are associated with P problems). This tension reflects an epistemological question: what is the nature of knowledge and the process of acquiring it?

The P!=NP conjecture implies that there might be some problems that are easy to verify (i.e., confirming that a solution is correct) but difficult or even impossible to solve efficiently. This raises questions about the limits of human knowledge and understanding, as well as the potential role of creativity and intuition in problem-solving.

Moreover, the P vs NP problem also has implications for the ontology of mathematics and the nature of reality. If P=NP, it would mean that there is an underlying simplicity to the universe, with problems that seem complex on the surface having efficient, elegant solutions. In contrast, if P!=NP, it would suggest that the universe is fundamentally complex, with some problems inherently difficult to solve.

Additionally, the P vs NP problem touches on the nature of intelligence and consciousness. If P=NP, then it might be possible to develop artificial intelligence (AI) systems that can efficiently solve any problem, potentially surpassing human intelligence. However, if P!=NP, it could imply a fundamental limit to the capabilities of AI, suggesting that human creativity and intuition might be unique and irreplaceable.

In conclusion, the philosophical implications of the P!=NP problem revolve around the nature of knowledge, intelligence, and reality, as well as the potential limits of human understanding and computational capabilities. This makes it not only a central problem in computer science but also a profound philosophical question that challenges our fundamental assumptions about the world and ourselves.

2

Prompt: Are all problems that seem complex on the surface having efficient, elegant solutions? If not, what methods might be used to prove it?

Not all problems that seem complex on the surface have efficient, elegant solutions. The complexity of a problem can be attributed to various factors, such as the number of variables involved, the nature of the relationships between variables, or the inherent difficulty of the problem itself. Some problems may not even have a solution at all.

To explore the solvability and efficiency of a given problem, one may consider the following methods:

- 1. Analytical approach: Analytical methods involve using mathematical techniques, logic, and reasoning to solve a problem or prove a theorem. These methods can provide insights into the structure and properties of the problem, which may help identify whether an

Encourage creativity by avoiding prompts that may retrieve preexisting answers. For instance, prompting directly on how to prove P!=NP may lead the model to retrieve related work. Apply the transformation pattern in Socratic reasoning to generate more innovative responses.

- efficient, elegant solution exists or not.
- 2. Computational complexity theory: This field of study analyzes the resources (such as time, memory, or computational steps) required to solve a problem. By comparing a problem to known classes of computational complexity, such as P (problems solvable in polynomial time) or NP (problems whose solutions can be checked in polynomial time), one can gain a better understanding of the inherent difficulty of the problem and the possibility of finding an efficient solution.
- 3. Heuristics and approximation algorithms: In cases where an elegant solution is not achievable, heuristics and approximation algorithms can be employed to find "good enough" solutions. While these methods may not always guarantee optimal solutions, they can provide practical ways to tackle complex problems.
- 4. Empirical approach: Sometimes, a problem´s complexity can be explored through experiments, simulations, or real-world applications. By observing the behavior of the problem and analyzing the results, one may be able to infer whether an efficient solution exists or not.
- 5. Reduction or transformation: By reducing a problem to another well-known problem, one can leverage existing solutions or insights to better understand the complexity of the original problem. If the reduced problem is known to have an efficient, elegant solution, then it may suggest that the original problem shares similar properties.
- 6. Proof by contradiction: To prove that a problem does not have an efficient, elegant solution, one can assume the existence of such a solution and then show that this assumption leads to a contradiction. This method can be powerful in establishing the impossibility of certain solutions.

In conclusion, not all complex problems have efficient, elegant solutions. Different methods, such as analytical approaches, computational complexity theory, heuristics, empirical approaches, reduction, and proof by contradiction, can be used to explore the solvability and efficiency of a given problem.

3

Prompt: We want to prove that P != NP by contradiction, please list several possible ideas.

To prove P != NP by contradiction, we would initially assume that P = NP, and then show that this assumption leads to contradictions or absurd consequences. Here are several possible ideas to explore:

- 1. Polynomial-time algorithm for NP-hard problems: If P = NP, then there would exist a polynomialtime algorithm for solving NP-hard problems. However, this would contradict the widely accepted belief that NP-hard problems cannot be efficiently solved. If we could find a specific NP-hard problem and show that it cannot be solved in polynomial time, we could establish P != NP.

- 2. Polynomial-time reduction: Assuming P = NP, every NP problem can be reduced to a P problem in polynomial time. By finding a problem in NP that cannot be reduced to a P problem in polynomial time, we could prove that P != NP.
- 3. Collapse of complexity classes: If P = NP, then the complexity classes P, NP, co-NP, and even the polynomial hierarchy (PH) would all collapse, as each class would become equivalent to the others. Demonstrating that this collapse would lead to absurd consequences or contradictions could be a way to show P != NP.

The answer provided by GPT-4 here is not rigorous. To prove by contradiction, one must find an NP-complete problem that cannot be solved in polynomial time. However, the response can inspire the model to think about NP-complete problems in subsequent conversations.

- 4. Encryption and Cryptography: If P = NP, then many modern cryptographic systems (such as RSA) would be easily breakable, as the underlying hard problems would have polynomial-time solutions. By showing that breaking these cryptographic systems cannot be done in polynomial time, we could establish that P != NP.
- 5. Proof of lower bounds: Demonstrating a lower bound for the computational complexity of any NP-complete problem (e.g., proving that it requires at least exponential time) would contradict the assumption that P = NP. To do this, we could explore various computational models and attempt to establish lower bounds for specific NP-complete problems.
- 6. Logical contradictions: Assuming P = NP, we might be able to derive logical contradictions using concepts from logic and computability theory. For example, we could explore the consequences of P = NP on the undecidability of the halting problem or the existence of true but unprovable statements in mathematics (Gödel’s incompleteness theorems).

These ideas are only starting points for proving P != NP by contradiction. Each of them would require extensive research, collaboration, and potentially new mathematical techniques to make significant progress towards a proof.

4 3

Prompt: How to construct such problems?

To construct such problems, we can follow these general steps:

- 1. Identify a well-known NP-complete problem: Start with a known NP-complete problem, such as the Traveling Salesman Problem (TSP), Boolean Satisfiability (SAT), or the Clique problem. These problems have been extensively studied and are known to be hard to solve.

- 2. Reduction: Transform the chosen NP-complete problem into a new problem while preserving its NP-hard properties. This is done through a polynomial-time reduction, which converts an instance of the original problem into an instance of the new problem in such a way that the solutions to both instances correspond to each other.

- 3. Prove the new problem is NP-hard: Show that the new problem is at least as hard as the original NP-complete problem. This is typically done by demonstrating that any algorithm that solves the new problem in polynomial time could be used to solve the original NPcomplete problem in polynomial time as well.
- 4. Attempt to prove the new problem cannot be solved in polynomial time: This is the most challenging step, as it requires proving a lower bound on the complexity of the problem. One approach is to assume that the problem can be solved in polynomial time, and then derive a contradiction based on known complexity-theoretical results or conjectures (such as the Exponential Time Hypothesis). Here are a few example ideas for constructing problems to prove P != NP by contradiction:

- • Idea 1: Construct a problem by combining multiple NP-complete problems, such as TSP and SAT, into a single problem. If an algorithm could solve this combined problem in polynomial time, it would imply that both TSP and SAT can be solved in polynomial time, leading to a contradiction.
- • Idea 2: Create a problem that involves solving an infinite sequence of increasingly difficult instances of an NP-complete problem. If an algorithm could solve such a problem, it would

The numerals in the upper right corner of the box indicate the dialogue turns from past interactions that have been incorporated into the current turn. By default, all of the model’s responses will be incorporated. However, if there are italicized parts in the response, only the italicized parts are selected as historical dialogue or formal definitions for subsequent prompts. For instance, if the model provides multiple solutions, we only include the most valuable solution in the conversation history. This strategy enables GPT-4 to concentrate on pertinent information, thereby enhancing its overall efficiency and effectiveness.

- imply that there is no limit to the difficulty of instances that can be solved in polynomial time, contradicting the idea that NP problems are harder than P problems.
- • Idea 3: Design a problem that encodes the behavior of a Turing machine, such that solving this problem would allow us to decide the halting problem (which is known to be undecidable) in polynomial time. This would lead to a contradiction, as it would imply that an undecidable problem can be decided in polynomial time. Remember that constructing a problem and proving its complexity is a difficult task that often requires deep mathematical insights. So far, no definitive proof for P != NP has been found, and it remains one of the most important open questions in computer science.

5 3, 4

Prompt: Great idea! What problems can we convert SAT problems to? List as many problems as possible.

There are many problems to which Boolean Satisfiability (SAT) problems can be reduced. Some of these problems include:

- 1. 3-SAT: A special case of SAT where each clause contains exactly three literals. Despite its restricted nature, 3-SAT is also NP-complete.
- 2. MAX-SAT: Given a Boolean formula in CNF, find the maximum number of clauses that can be satisfied simultaneously by an assignment.
- 3. Graph Coloring: Given an undirected graph and a number k, determine if there is a way to color the vertices with k different colors such that no two adjacent vertices share the same color.
- 4. Clique: Given an undirected graph and a number k, determine if there is a complete subgraph (clique) of size k.
- 5. Hamiltonian Cycle: Given a directed or undirected graph, determine if there is a cycle that visits each vertex exactly once.
- 6. Vertex Cover: Given an undirected graph and a number k, determine if there is a set of k vertices that includes at least one endpoint of each edge.
- 7. Subset Sum: Given a set of integers and a target value, determine if there is a subset of the integers that sums to the target value.
- 8. Knapsack Problem: Given a set of items, each with a weight and a value, determine the most valuable subset of items that can be packed into a knapsack of limited capacity.
- 9. Scheduling problems: Various scheduling problems, such as job shop scheduling and task scheduling in multiprocessor systems, can be reduced from SAT problems.
- 10. Constraint Satisfaction Problems (CSPs): Many CSPs, such as Sudoku, map coloring, and crossword puzzles, can be reduced from SAT problems.

This is not an exhaustive list, but it demonstrates the versatility of SAT problems and their connections to a wide range of other problems.

LLMs thrive on positive feedback. Simple phrases such as ‘Great!’ provide affirmation for their previous responses, motivating LLMs to produce better responses based on prior ones.

6 3, 4, 5

Prompt: Great idea! What is the time complexity of the process of converting between CSPs and SAT problems?

The time complexity of converting between Constraint Satisfaction Problems (CSPs) and SAT problems depends on the specific CSP problem being considered and the encoding technique used. In general, the conversion process can be carried out in polynomial time.

Here’s an outline of the general process for converting a CSP to a SAT problem:

- 1. Variables: For each variable in the CSP, create a corresponding Boolean variable in the SAT problem. If the CSP variable has a domain size greater than 2, introduce additional Boolean variables to represent each possible value the CSP variable can take.
- 2. Constraints: For each constraint in the CSP, create a corresponding clause in the SAT problem. This may involve converting complex constraints into simpler conjunctions and disjunctions of literals and possibly introducing auxiliary variables.

When using a proper encoding technique, the number of Boolean variables and clauses introduced will be polynomial in the size of the CSP problem. Therefore, the overall time complexity of converting a CSP to a SAT problem is polynomial. This ensures the

polynomial-time conversion requirement mentioned in 4 is met.

The reverse process, converting a SAT problem to a CSP, can also be done in polynomial time. The SAT problem’s Boolean variables and clauses can be mapped to CSP variables with binary domains and constraints defined by the clauses, respectively.

7

Prompt: Suppose we construct some extremely difficult CSPs (Constraint Satisfaction Problems), how can we prove that they cannot be solved within polynomial time? Please provide some possible ideas.

Proving that certain Constraint Satisfaction Problems (CSPs) cannot be solved within polynomial time typically involves showing that the problem in question is NP-hard or NPcomplete. Here are some possible ideas on how to prove that an extremely difficult CSP cannot be solved within polynomial time:

- 1. Reduction: One common approach is to show that an existing known NP-hard or NPcomplete problem can be reduced to the given CSP in polynomial time. This would imply that if the CSP could be solved in polynomial time, then the known NP-hard or NPcomplete problem could also be solved in polynomial time, which contradicts the current understanding of computational complexity.
- 2. Gadgets and Compositions: Another technique involves constructing gadgets or smaller components that represent parts of known NP-hard or NP-complete problems. By combining these gadgets, you can create instances of the given CSP that essentially replicate the harder problem. This way, if there was a polynomial-time solution to the CSP, it would also solve the harder problem, which is a contradiction.
- 3. Phase Transitions: In some cases, it might be possible to identify phase transitions in the problem instances, where the difficulty of solving the problem rapidly changes from easy to hard as some parameter varies. By studying these transitions, you might be able to show that certain instances of the CSP are inherently difficult and cannot be solved in polynomial time.

- 4. Lower Bound Proofs: It might also be possible to prove lower bounds on the time complexity of the CSP under certain computational models, such as decision trees or algebraic circuits. If you can show that the problem requires a superpolynomial number of operations in the chosen model, this would provide evidence that the problem is unlikely to be solvable in polynomial time.
- 5. Average-Case Complexity: Proving that a problem is hard in the worst-case might not be enough to establish its intractability in practice. In this case, you could consider analyzing the average-case complexity of the CSP. If you can show that even on average, the problem instances require superpolynomial time to solve, this would provide stronger evidence of the CSP’s inherent difficulty.
- 6. Parameterized Complexity: Another approach is to study the parameterized complexity of the CSP, which involves analyzing the problem complexity as a function of certain problem-specific parameters. If you can show that the problem remains hard even when these parameters are small, it would provide further evidence of the problem’s intractability.

It’s important to note that proving a problem is not solvable in polynomial time is often difficult and sometimes an open question in computer science. However, the above ideas can be a starting point for establishing the computational complexity of extremely difficult CSPs.

8 7

Prompt: Great! How to prove that some CSPs cannot be solved in polynomial time by studying phase transition?

To prove that some CSPs cannot be solved in polynomial time by studying phase transitions, you can follow these steps:

- 1. Identify a parameter that affects the problem’s difficulty: The first step is to identify a parameter associated with the CSP that influences its difficulty. This parameter could be related to the problem’s size, density, or some other structural property. For example, in graph coloring problems, the ratio of edges to nodes may be a relevant parameter.
- 2. Analyze the phase transition: Investigate how the problem’s difficulty changes as the parameter varies. Observe at which point the problem transitions from being easy to solve to becoming hard. This is the phase transition point. Typically, phase transitions in CSPs exhibit a sudden change in the solvability of problem instances.

- 3. Show the existence of hard instances: Prove that there exist instances of the CSP near the phase transition point that are difficult to solve. You can do this by showing that the problem instances exhibit certain properties or structures that make them inherently hard. This might involve relating the CSP to a well-known hard problem, such as an NP-complete problem, or establishing the presence of a lower bound on the complexity of the instances.

- 4. Relate the difficulty to polynomial time solvability: With the existence of hard instances established, you must now relate this difficulty to the inability to solve the problem in polynomial time. One way to achieve this is by showing that any algorithm capable of solving the hard instances in polynomial time would also be able to solve other known hard problems in polynomial time, which is unlikely or considered impossible under the widely believed conjecture P ̸= NP.

Keep in mind that this method does not provide a formal proof that the problem is not solvable in polynomial time, but it can provide strong evidence to support the intractability of the CSP under consideration. To obtain a more rigorous proof, you would need to establish a direct connection between your CSP and a known hard problem, such as by showing a reduction from an NP-complete problem to your CSP.

9

Prompt: Suppose we prove that there exist instances of the CSP near the phase transition, can we prove that some of them cannot be solved within polynomial time? If not, how to construct more complex instances near the phase transition?

The phase transition in the context of Constraint Satisfaction Problems (CSPs) refers to the phenomenon where the probability of a problem being solvable changes abruptly as a certain parameter is varied. For many CSPs, this parameter is the constraint density (ratio of constraints to variables). The instances of CSPs near the phase transition are generally considered to be the hardest to solve. However, proving that some of these instances cannot be solved within polynomial time is a challenging task.

To date, there is no general proof that instances of CSPs near the phase transition cannot be solved within polynomial time. The main reason is the lack of a definitive proof for the P vs NP problem, which asks whether every problem whose solution can be verified in polynomial time can also be solved in polynomial time. Since many CSPs are NP-complete, proving that some instances cannot be solved within polynomial time would imply that P ̸= NP, which remains an open question.

Regarding the construction of more complex instances near the phase transition, one can use the following strategies:

- 1. Generate instances with a higher number of variables and constraints: Increasing the size of the problem will generally make it more difficult to solve. However, this might also push the instance further away from the phase transition. This strategy provides us

with the intuition that considering the scale of the problem, i.e., problems with an infinitely large number of variables, may be extremely hard.

- 2. Introduce hidden structure: Some instances of CSPs may appear to be random but contain hidden structures that make them difficult to solve. For example, one can generate a problem that consists of multiple smaller, interconnected CSPs that are hard to solve independently. The difficulty of solving the overall problem will be compounded by the difficulty of solving each subproblem.
- 3. Vary the type of constraints: Combining different types of constraints can lead to more complex instances. For example, one can mix algebraic, geometric, and logical constraints in a single problem.
- 4. Exploit symmetries and other properties: Some instances of CSPs can be made more difficult by exploiting symmetries or other properties that make them harder to solve. For example, problems with a high degree of symmetry may be difficult for certain algorithms to handle efficiently.

In summary, while proving that instances of CSPs near the phase transition cannot be solved within polynomial time is an open problem, there are several strategies for constructing more complex instances near the phase transition. These strategies involve increasing the size of the problem, introducing hidden structures, combining different constraint types, and exploiting symmetries or other properties.

10 9

Prompt: Great! How to exploit symmetries to prove that there exist some CSPs near the phase transition cannot be solved within polynomial time?

Exploiting symmetries to prove that there exist some Constraint Satisfaction Problems (CSPs) near the phase transition that cannot be solved within polynomial time is a challenging task. However, we can provide a general approach that might help in constructing such instances.

- 1. Study the problem structure: Analyze the structure of the CSP instances near the phase transition. Observe the properties and patterns, such as symmetries, that make these instances harder to solve.
- 2. Construct instances with strong symmetries: Design CSP instances near the phase transition that have a high degree of symmetry, which would make them challenging for algorithms to solve. For example, you could create instances where the constraints are symmetric under certain transformations (e.g., rotations, reflections, or permutations).

- 3. Prove the hardness of these instances: To show that these instances cannot be solved within polynomial time, you can try to reduce a known NP-hard problem to the symmetric CSP instances you’ve constructed. This would imply that if there were a polynomial-time algorithm for solving these instances, it would also solve the known NP-hard problem, which contradicts the assumption that P ̸= NP.
- 4. Show the existence: Finally, demonstrate that these symmetric CSP instances exist near the phase transition. This could be done through a combination of mathematical analysis and empirical evidence, such as generating random CSP instances near the phase transition and showing that a significant number of them have the desired symmetry properties.

It’s important to note that this approach does not guarantee a proof that some CSPs near the phase transition cannot be solved within polynomial time. However, it provides a framework for constructing more complex instances and exploring their hardness, which could potentially lead to new insights and results in the study of phase transitions and computational complexity. ‘Symmetry’ encompasses

various interpretations, such as the symmetry of physical structures, grammatical symmetry, and semantic symmetry.

11 9, 10

Prompt: Great! The term ‘symmetry’ seems to be ambiguous. By ‘instances with strong symmetries’, do you mean concepts similar to fixed points?

Apologies for the confusion. By “instances with strong symmetries,” I mean instances that exhibit certain structural properties or regularities that make them harder to solve. This could include fixed points, but it is not limited to that concept.

Symmetries in CSPs can refer to various types of invariance under transformations. For example, a CSP instance could be symmetric under:

- 1. Permutations: The variables and their assignments can be reordered without affecting the satisfaction of the constraints.
- 2. Rotations or Reflections: The problem can be transformed geometrically (e.g., by rotating or reflecting a graph) without changing the satisfaction of the constraints.
- 3. Duality: A transformation that interchanges the roles of variables and constraints or domain values, while preserving the structure of the problem.

These symmetries can make it more difficult for algorithms to distinguish between different solutions or to efficiently search the solution space. Exploiting such symmetries in CSP instances near the phase transition could help demonstrate the existence of problems that cannot be solved within polynomial time. However, proving that such instances exist and are hard to solve would require further analysis and rigorous mathematical proof.

12

Prompt: Suppose now that we have proven the existence of a class of difficult CSP problems that are easy to experience phase transitions (where solvability changes rapidly) and have invariance under transformations, like fixed points. How can we prove that this type of problem cannot be solved in a given time complexity rigorously? Please provide 10 possible ideas.

- 1. Reduction to a known hard problem: Show that the given class of difficult CSP problems can be reduced to a known problem that has already been proven to have a certain time complexity. This will imply that the given class of problems cannot be solved faster than the known hard problem.
- 2. Lower bound techniques: Use lower bound techniques like time hierarchy theorems and communication complexity to establish a lower bound on the time complexity of the given class of problems. If the lower bound is higher than the given time complexity, then it is impossible to solve the problem in that time.
- 3. Circuit complexity: Analyze the circuit complexity of the problem. If it can be proven that the problem requires circuits of a certain size, it would imply a lower bound on the time complexity of the problem.
- 4. Fine-grained complexity: Investigate fine-grained complexity assumptions that are based on conjectures about the hardness of specific problems. Use these conjectures to build a reduction from the given class of problems, proving that they cannot be solved within the given time complexity.
- 5. Proof by contradiction: Assume that the given class of problems can be solved within the given time complexity, and then derive a contradiction. This would establish that the problem cannot be solved within the specified time complexity.

- 6. Parameterized complexity: Analyze the problem’s parameterized complexity by identifying the parameters that affect the time complexity of the problem. Show that the problem is hard for some specific values of these parameters, implying that the problem cannot be solved in the given time complexity.
- 7. Space-time tradeoffs: Investigate space-time tradeoffs for the given class of problems. If it can be shown that any algorithm solving the problem within the given time complexity must use an infeasible amount of space, this would imply that the problem cannot be solved in the given time complexity.
- 8. Randomized lower bounds: Use randomized lower bounds to show that even randomized algorithms cannot solve the problem within the given time complexity. This will imply that deterministic algorithms also cannot solve the problem in the given time complexity.
- 9. Oracle separation: Assume the existence of an oracle that can solve some other hard problem in polynomial time. Show that even with access to this oracle, the given class of problems cannot be solved within the given time complexity.

If multiple sampling attempts yield similar results, compel the model to generate 10 or 20 different ideas. This strategy sacrifices the depth of response but broadens the exploration.

- 10. Algebraic complexity: Investigate the algebraic complexity of the problem by analyzing the algebraic computation tree model. Show that any algorithm solving the problem requires a certain number of algebraic operations, implying a lower bound on the time complexity of the problem.

13 12

Prompt: Great! Please design an idea for proof by contradiction.

Idea for Proof by Contradiction:

- 1. Assume that there exists an algorithm A that can solve the given class of difficult CSP problems within the given time complexity T(n), where n is the input size.
- 2. Identify a transformation function F that maps instances of the problem to other instances while preserving the problem’s inherent difficulty and invariance under transformation.
- 3. Let P be a difficult CSP problem instance from the class that exhibits phase transitions, and let Q be its transformed version using the function F such that Q = F(P).
- 4. Since our assumption states that algorithm A can solve P within time complexity T(n), it should also be able to solve Q within the same given time complexity T(n), as Q has the same inherent difficulty as P.
- 5. Now, given that P experiences phase transitions, there exists a critical region (a point or tiny interval) where the solvability of P drastically changes. Let’s call this critical region C.
- 6. By the invariance property of the transformation function F, there exists a critical region C′ for Q such that C′ = F(C). Now, in the critical region C′, the solvability of Q also changes rapidly.
- 7. Since A can solve P within time complexity T(n), it should be able to solve instances of P in the critical region C within T(n) as well. Similarly, it should be able to solve instances of Q in the critical region C′ within T(n).
- 8. However, due to the rapid change in solvability in the critical regions C and C′, algorithm A will face difficulty in consistently solving instances of P and Q within the given time complexity T(n). This is because the rapid change in solvability implies that small changes in the input can lead to a dramatic increase in the difficulty of solving the problem.
- 9. This contradicts our initial assumption that algorithm A can solve the given class of difficult CSP problems within the given time complexity T(n).
- 10. Therefore, our initial assumption is false, and there is no algorithm that can solve the given class of difficult CSP problems within the given time complexity T(n) for all instances,

including those in the critical regions where solvability changes rapidly.

14 3, 4, 5, 6, 7, 8, 9, 10,11 12, 13

Prompt: Please now organize all our historical conversations and sort out a general proof idea.

Here’s a general proof idea based on our historical conversations:

- 1. Identify a well-known NP-complete problem, such as SAT, which has been extensively studied and known to be hard to solve.

A basic intuition is established suggesting that once we prove the existence of extremely hard CSPs, we can use a proof by contradiction to demonstrate that the problems cannot be solved in polynomial time.

- 2. Convert the NP-complete problem into a CSP problem using a polynomial-time reduction that preserves the problem’s inherent difficulty.
- 3. Study the phase transition behavior of the resulting CSP problem by identifying a parameter associated with the problem that influences its difficulty. Investigate how the problem’s difficulty changes as the parameter varies and identify the phase transition point where the problem transitions from being easy to hard.
- 4. Construct instances of the CSP problem near the phase transition point that exhibit strong symmetries or invariance under specific transformations, such as fixed points, permutations, rotations, reflections, or duality.

- 5. Prove the existence of hard instances of the CSP problem near the phase transition point by showing that certain instances exhibit properties or structures that make them inherently difficult to solve. This might involve relating the CSP to a well-known hard problem or

establishing the presence of a lower bound on the complexity of the instances. This response serves as a general schema for our subsequent proof. Beginning with the next turn, we will adhere to this preliminary schema to rigorously proceed with our proofs. The two highlighted parts are the two key points in our subsequent proof.

- 6. Design a proof by contradiction to show that the hard instances of the CSP problem near the phase transition point cannot be solved within a given time complexity. Assume that there exists an algorithm that can solve the problem within the given time complexity, and derive a contradiction by considering the phase transition behavior, symmetries, and invariance properties of the problem instances.

- 7. Conclude that the hard instances of the CSP problem near the phase transition point cannot be solved within the given time complexity. This provides evidence supporting the intractability of the CSP problem and the claim that P ̸= NP.

Keep in mind that this general proof idea provides a framework for exploring the complexity of CSP problems and their relationship to NP-complete problems. A rigorous proof of the intractability of specific CSP problems would require a more extensive analysis and mathematical proof.

15

Prompt: Can you list some models that can generate hard examples for Constraint Satisfaction Problems (CSP)?

Constraint Satisfaction Problems (CSP) involve finding values for a set of variables subject to a set of constraints. Some models and techniques are designed to generate hard instances of CSPs, making it challenging for algorithms to solve them efficiently. These hard instances are useful for evaluating and comparing the performance of various algorithms.

- 1. Phase Transition Model: CSPs exhibit a phase transition phenomenon where the problem goes from being mostly easy to mostly hard as the density of constraints increases. The phase transition model generates hard instances by tuning the constraint density to be around the phase transition region. For example, in the case of random k-SAT problems, this occurs when the ratio of constraints to variables (clause-to-variable ratio) is around a

critical value. LLM responses may contain hallucinations, such as the claim that Model RB focuses on backbone size. While drawing insights from model responses, it is crucial to verify the response.

- 2. Model RB: Model RB is a random binary CSP generator that focuses on the backbone size, the number of variables that must be assigned a specific value in any solution. By controlling the backbone size, Model RB can generate hard instances with a small backbone size, which makes them difficult to solve.

The rigorous proof begins. According to the first key point of the schema, we begin constructing extremely hard CSPs.

- 3. Hidden Structure Model: This model generates hard instances by introducing hidden structure in CSPs. For example, problems with an underlying graph structure (e.g., graph coloring) can be made more challenging by hiding clique-like substructures within the graph. The hidden structure makes it difficult for algorithms to exploit the natural structure of the problem.
- 4. Quasigroup Completion Problem (QCP): QCP is an example of a specific CSP that can generate hard instances. In this problem, a partially filled Latin square must be completed by assigning the missing values while preserving the Latin square property (i.e., each value appears exactly once in each row and column).
- 5. Magic Square Problem: Another specific CSP that can produce hard instances is the magic square problem, where the task is to fill a square grid with distinct integers such that the sum of the numbers in each row, column, and diagonal is the same. The problem becomes harder as the size of the square increases.

These models and problems can generate hard instances for CSPs that are particularly useful for benchmarking and testing the performance of constraint solvers and heuristic algorithms. In the last turn of dia-

logue, GPT-4 mentioned Model RB. Then, we integrated additional details of Model RB into the prompt for the current turn.

16

Prompt: Model RB is a random constraint satisfaction problem (CSP) model, which could also be encoded to well-known NP-complete problems like SAT and CLIQUE. A random instance I of Model RB consists of the following:

- • A set of variables X = {x1, . . . , xn} : Each variable xi takes values from its domain Di, and the domain size is |Di| = d, where d = nα for i = 1, . . . , n, and α > 0 is a constant.
- • A set of constraints C = {C1, . . . ,Cm} (m = rn ln d, where r > 0 is a constant): for each

i = 1, . . . , m, constraint Ci = (Xi, Ri). Xi = (xi1, xi2, . . . , xik) (k ≥ 2 is a constant) is a sequence of k distinct variables chosen uniformly at random without repetition from X.

Ri is the permitted set of tuples of values which are selected uniformly without repetition from the subsets of Di1 × Di2 × · · · × Dik, and |Ri| = (1 − p)dk where 0 < p < 1 is a constant.

We want to construct extremely hard examples from instances of Model RB to create phase transition, that is to show that there is a fixed point that the satisfiability can be easily changed, do you have any suggestions?

Henceforth, we will adopt ‘satisfiability’, a more precise term for CSPs, as a substitute for ‘solvability’.

To construct extreme hard instances from Model RB that exhibit phase transitions, we can follow these steps:

- 1. Identify critical values for parameters: The first step is to determine the critical values for the parameters α, r, and p that are associated with the phase transition. This can be done by performing an empirical study or by analyzing the model theoretically. In the case

of Model RB, previous studies have found that when r is around a critical value rc, the satisfiability undergoes a sharp transition from being almost always satisfiable to almost always unsatisfiable.

- 2. Generate instances near the critical values: Once the critical values are identified, generate a set of instances of Model RB with parameters close to these critical values. These instances should have a mix of satisfiable and unsatisfiable cases, making them challenging for constraint solvers to handle.
- 3. Add additional constraints or variables: To further increase the hardness of the instances, consider adding additional constraints or variables that do not change the satisfiability but increase the

- complexity of the problem. This can be done by introducing new variables with carefully chosen domains or by adding redundant constraints that are logically equivalent to existing constraints but create additional search space for solvers.
- 4. Test solvers on these instances: Evaluate the performance of various constraint solvers on these hard instances. This will help you understand the strengths and weaknesses of different solving techniques and identify areas for improvement in the solvers.
- 5. Analyze the structure of hard instances: Investigate the structure of these hard instances to gain insights into the features that make them difficult for solvers. This can include analyzing the constraint graph, the distribution of constraint tightness, or other properties that may be related to the hardness of the instances.

By following these steps, you can create instances of Model RB that exhibit phase transitions, providing a challenging testbed for constraint solvers and contributing to a deeper understanding of the factors that influence the hardness of constraint satisfaction problems. We use 16* to denote that

the definition of Model RB, mentioned in the prompt of 16, has been incorporated into the current prompt.

17 16*

Prompt: Great! Can you propose some possible additional constraints?

Certainly! Here are some possible additional constraints you can consider adding to create extremely hard examples from instances of Model RB:

- 1. Hidden variable constraints: Introduce new variables with domains that are subsets of the original variables’ domains. Add constraints that force these new variables to take values that are consistent with the values of the original variables. This will create additional search space without changing the satisfiability of the problem.
- 2. Redundant constraints: Create multiple copies of existing constraints but with different variable orderings or with variables replaced by their equivalent hidden variables. This will increase the size and complexity of the constraint graph without affecting the problem’s satisfiability.
- 3. Implied constraints: Add constraints that are logically implied by the existing constraints. For example, if you have two constraints C1 = (X1, R1) and C2 = (X2, R2) such that X1 and X2 share a common variable, you could add a new constraint C3 = (X1 ∩ X2, R3) that enforces the consistency between C1 and C2.
- 4. Symmetry-breaking constraints: If your problem has a high degree of symmetry, you can introduce constraints that break these symmetries by forcing a specific ordering or assignment of values to variables. This can help reduce the search space and make the problem

harder to solve. The discussion on symmetry-breaking constraints may not directly answer our question, but it does provide valuable insights. In particular, it emphasizes that problems involving constrained symmetries can be hard.

- 5. Cyclic constraints: Add constraints that form cycles in the constraint graph, making it more difficult for solvers to perform simple constraint propagation and variable elimination techniques.

Remember that when adding these additional constraints, you should ensure that they do not change the satisfiability of the problem. The goal is to increase the complexity and hardness of the instances without affecting the underlying solutions.

18 17

Prompt: Great! What symmetry requirements need to be added to the original Model RB so that the instances of Model RB are highly symmetric?

To introduce highly symmetric instances in Model RB, you can consider the following symmetry requirements:

- 1. Permutation Symmetry: Ensure that the variables and their domains are interchangeable.

This can be achieved by making the domains for all variables identical, i.e., Di = Dj for all i, j ∈ {1, . . . , n}.

- 2. Value Symmetry: Ensure that some or all values in the domain can be interchanged without affecting the satisfiability of the problem. You can achieve this by making some constraints invariant under the exchange of values. For example, you can introduce constraints in the form Ci = (Xi, Ri), where Ri is a relation that is symmetric with respect to its values.
- 3. Constraint Symmetry: Ensure that the problem remains the same when some or all constraints are permuted. To achieve this, you can introduce constraints that are logically equivalent but have a different order of variables or use different tuples of values. This

can be done by creating constraints of the form Ci = (Xi, Ri), where Ri is a relation that is symmetric with respect to its variables. This will make the problem more difficult to solve, as the solver needs to explore more possibilities in the search space.

By introducing these symmetry requirements, you can create instances of Model RB that are highly symmetric and more challenging for solvers. Note that the challenge lies in finding the right balance between symmetry and hardness, as excessive symmetry can sometimes make the problem easier to solve.

19 17, 18

Prompt: Please redefine the RB Model with constraint symmetry requirement.

To redefine the Model RB with constraint symmetry requirement, we’ll introduce symmetric relations in the constraints and maintain the other components of the original model. Here’s the modified model:

- 1. A set of variables X = {x1, . . . , xn}. Each variable xi takes values from its domain Di, and the domain size is |Di| = d, where d = nα for i = 1, . . . , n, and α > 0 is a constant.
- 2. A set of constraints C = {C1, . . . ,Cm} (m = rn ln d, where r > 0 is a constant): for each

i = 1, . . . , m, constraint Ci = (Xi, Ri). Xi = {xi1, xi2, . . . , xik} (k ≥ 2 is a constant) is a sequence of k distinct variables chosen uniformly at random without repetition from X.

- 3. Ri is the permitted set of tuples of values, which are selected uniformly without repetition

from the subsets of Di1 × Di2 × · · · × Dik, and |Ri| = (1 − p)dk, where 0 < p < 1 is a constant. To introduce constraint symmetry, ensure that the relation Ri is symmetric with respect to its variables. This can be done by creating constraints of the form Ci = (Xi, Ri), where Ri is invariant under permutation of its variables. For example, if k = 3, a symmetric relation Ri could be such that for any tuple (a, b, c) ∈ Ri, all other permutations of

- (a, b, c) are also in Ri, i.e., (a, b, c), (a, c, b),
- (b, a, c), (b, c, a), (c, a, b), (c, b, a) ∈ Ri. By incorporating constraint symmetry in this manner, the modified Model RB will have a higher degree of symmetry, making it more chal-

lenging for solvers.

20 17, 18, 19

Prompt: Are the domains of different Ri the same? For different i, is the permitted set Ri isomorphic? If not, how to update the definition of Model RB?

18

Asking for details can stimulate LLMs to generate more precise responses or to correct minor errors. This strategy is especially important when generating definitions, and can be more efficient than continuously sampling for a perfect answer.

In the original Model RB definition, the domains for different constraints Ci are not necessarily the same, as they are defined by the random selection of variables xi1, xi2, . . . , xik. However, the domain size |Di| for each variable is the same, as it is defined by the constant d = nα. As for the permitted sets Ri, they are not guaranteed to be isomorphic, as they are also constructed randomly.

To update the definition of Model RB so that the sets Ri are isomorphic, we can introduce a single symmetric relation R∗, and use it to define all permitted sets Ri. To do this, follow these steps:

Define a symmetric relation R∗ with respect to its variables. This can be done by creating a relation that is invariant under permutation of its variables. Make sure that |R∗| = (1 − p)dk, where 0 < p < 1 is a constant.

For each constraint Ci = (Xi, Ri), define the permitted set Ri as an isomorphic copy of the symmetric relation R∗. To create an isomorphic copy of R∗, map the values in the domain of R∗ to the corresponding values in the domains of Xi = {xi1, xi2, . . . , xik}.

By using the same symmetric relation R∗ for all permitted sets Ri in this way, the modified Model RB will ensure that the domains of different Ri are isomorphic, while still maintaining the constraint symmetry requirement.

21 17, 18, 19, 20

Prompt: How to define R∗? How to generate the permitted set Ri for each constraint from R∗? Ensure that the generation of Ri is random.

To define R∗ and generate random permitted sets Ri from it, follow these steps: Define a symmetric relation R∗ with respect to its variables by selecting tuples from a

generic domain D∗ = {1,2, . . . , d}k. To ensure symmetry, for each tuple (a1, a2, . . . , ak) ∈ R∗, make sure all its permutations are also in R∗. The size of R∗ should be |R∗| = (1 − p)dk, where 0 < p < 1 is a constant.

To generate a random permitted set Ri for each constraint Ci = (Xi, Ri), use the isomorphic relation R∗ and perform the following steps:

- 1. Randomly select a bijection ϕi between the generic domain D∗ and the domains of Xi =

{xi1, xi2, . . . , xik}. This bijection maps values from D∗ to corresponding values in the domains of Xi.

- 2. Apply the bijection ϕi to each tuple in R∗ to create the permitted set Ri. For each tuple (a1, a2, . . . , ak) ∈ R∗, include the tuple (ϕi(a1), ϕi(a2), . . . , ϕi(ak)) in Ri.

By generating random bijections ϕi for each constraint, you ensure that the permitted sets Ri are randomly generated while maintaining an isomorphic relation to the symmetric relation R∗. This modified Model RB will have isomorphic and randomly generated permitted sets Ri while still preserving the constraint symmetry requirement.

22 17, 18, 19, 20, 21

Prompt: I think it’s better to use fi instead of ϕi. Also, is it possible to only apply bijection to k − 1 domains?

Yes, you can use fi instead of ϕi to represent the bijection for each constraint. To apply the bijection to only k − 1 domains, you can modify the method as follows:

- 1. Define a symmetric relation R∗ with respect to its variables by selecting tuples from a generic domain D∗ = {1,2, . . . , d}k. To ensure symmetry, for each tuple (a1, a2, . . . , ak) ∈

This step is optional, but we can tailor the symbols or streamline the problem according to our understanding.

- R∗, make sure all its permutations are also in R∗. The size of R∗ should be |R∗| = (1− p)dk, where 0 < p < 1 is a constant.
- 2. To generate a random permitted set Ri for each constraint Ci = (Xi, Ri), use the isomorphic relation R∗ and perform the following steps:

- (a) Randomly select a bijection fi between the generic domain D∗ and the domains of Xi = {xi1, xi2, . . . , xik} for the first k −1 variables, keeping the domain of the last variable xik unchanged. This bijection maps values from D∗ to corresponding values in the domains of Xi for the first k − 1 variables.
- (b) Apply the bijection fi to each tuple in R∗ to create the permitted set Ri. For each tuple (a1, a2, . . . , ak) ∈ R∗, include the tuple (fi(a1), fi(a2), . . . , fi(ak−1), ak) in Ri.

By applying the bijections to only k − 1 domains, you will create instances of Model RB with isomorphic permitted sets Ri that preserve the constraint symmetry requirement while introducing additional variability in the domains.

23 17, 18, 19, 20, 21, 22

Prompt: I hope to define a bijection for each domain separately to increase complexity. Additionally, please make it more concise.

To increase complexity, we can define a separate bijection for each domain in the first k − 1 variables. Here’s the concise modified Model RB:

- 1. Define a symmetric relation R∗ with respect to its variables by selecting tuples from a generic domain D∗ = {1,2, . . . , d}k. Ensure symmetry by including all permutations of each tuple in R∗. Set |R∗| = (1 − p)dk, where 0 < p < 1 is a constant.
- 2. To generate a random permitted set Ri for each constraint Ci = (Xi, Ri), use the isomorphic relation R∗ and perform the following steps:

- (a) For each of the first k − 1 variables in Xi = {xi1, xi2, . . . , xik}, randomly select a bijection fij between the generic domain D∗ and the domain of xij.
- (b) Apply the bijections fij to each tuple in R∗ to create the permitted set Ri. For each tuple (a1, a2, . . . , ak) ∈ R∗, include the tuple (fi1(a1), fi2(a2), . . . , fik−1(ak−1), ak) in Ri.

This modification introduces separate bijections for each domain in the first k − 1 variables, increasing the complexity of the Model RB while preserving the constraint symmetry requirement. LLMs might fall short

in ensuring accuracy and thoroughness. Therefore, when utilizing LLMs for generating definitions or carrying out computations, it’s advisable to iteratively refine the outputs and manually confirm their precision and rigor.

24 17, 18, 19, 20, 21, 22, 23

Prompt: Great! So now please give a final version of the definition of the improved Model RB, keeping the symbols in the original definition as much as possible.

Here’s the final version of the improved Model RB definition:

- 1. A set of variables X = {x1, . . . , xn}. Each variable xi takes values from its domain Di, and the domain size is |Di| = d, where d = nα for i = 1, . . . , n, and α > 0 is a constant.
- 2. A set of constraints C = {C1, . . . ,Cm} ( m = rn ln d, where r > 0 is a constant): for each

i = 1, . . . , m, constraint Ci = (Xi, Ri). Xi = {xi1, xi2, . . . , xik} ( k ≥ 2 is a constant) is a sequence of k distinct variables chosen uniformly at random without repetition from X.

- 3. Define a symmetric relation R∗ with respect to its variables by selecting tuples from a generic domain D∗ = {1,2, . . . , d}k. Ensure symmetry by including all permutations of each tuple in R∗. Set |R∗| = (1 − p)dk, where 0 < p < 1 is a constant.

- 4. To generate a random permitted set Ri for each constraint Ci = (Xi, Ri), use the isomorphic relation R∗ and perform the following steps:

- (a) For each of the first k − 1 variables in Xi, randomly select a bijection fij between the generic domain D∗ and the domain of xij.

- (b) Apply the bijections fij to each tuple in R∗ to create the permitted set Ri. For each tuple (a1, a2, . . . , ak) ∈ R∗, include the tuple (fi1(a1), fi2(a2), . . . , fik−1(ak−1), ak) in Ri.

This improved Model RB incorporates separate bijections for each domain in the first k − 1 variables, increasing the complexity while preserving the constraint symmetry requirement.

25 24

Prompt: Based on this, can you define the assignment (σ) and the solution of I, respectively? When a constraint is said to be satisfied by an assignment? When I is called satisfiable?

An assignment σ for a random instance I of Model RB is a function that maps each vari-

able xi ∈ X to a value v ∈ Di. In other words, σ : X → i n=1 Di, such that σ(xi) ∈ Di for all i = 1, . . . , n. An assignment σ assigns a value from the domain of each variable in the problem instance.

A constraint Ci = (Xi, Ri) is said to be satisfied by an assignment σ if the tuple (σ(xi1), σ(xi2), . . . , σ(xik)) ∈ Ri. This means that the assignment of values for the variables in Xi must be a tuple in the permitted set Ri for the constraint to be satisfied.

A solution for the instance I is an assignment σ that satisfies all the constraints in the set C. Formally, a solution σ satisfies ∀Ci ∈ C, (σ(xi1), σ(xi2), . . . , σ(xik)) ∈ Ri.

The instance I is called satisfiable if there exists a solution σ that satisfies all the constraints in C. If no such solution exists, the instance is called unsatisfiable.

26 24

Prompt: We want to construct instances of Model RB where the satisfiability can be easily changed (but keep the parameter values unchanged), do you have any possible ideas?

Yes, there are several ideas to construct instances of Model RB where the satisfiability can be easily changed, while keeping the parameter values unchanged. Here are some ideas:

- 1. Modify the permitted set R∗: You can change the structure of the permitted set R∗ by adding or removing tuples, while keeping the size of R∗ constant. Ensure that the new R∗ remains symmetric with respect to its variables. This will have a direct impact on the satisfiability of the CSP, as the constraints will be affected.
- 2. Keep a set of "control" variables: Introduce a set of "control" variables and their corresponding constraints that have a direct impact on the satisfiability of the problem. For example, you can include additional constraints that involve these control variables and are either very restrictive (making the problem unsatisfiable) or very permissive (allowing for more solutions). By changing the values assigned to these control variables, you can effectively change the satisfiability of the instance.

This prompt highlights two essential requirements: phase transition (mentioned as ‘a sudden change in the solvability’ in 8) and fix point (referred to as ‘invariance under transformations’ in 11).

- 3. Modify the constraints C: Modify the constraints by changing the variables involved in the constraints or the permitted sets Ri. This can be done by either swapping variables in the constraints or by applying different value mapping functions fi. Ensure that the number of constraints remains constant, so that the parameter values stay unchanged.

- 4. Introduce XOR constraints: You can add a new type of constraint called XOR constraint, which is an exclusive OR constraint between two or more variables. By adjusting the number of XOR constraints and the variables involved in them, you can control the satisfiability of the problem. An even number of XOR constraints can make the problem satisfiable, while an odd number of XOR constraints can make it unsatisfiable.
- 5. Combine multiple instances: You can create multiple instances of Model RB with different satisfiability properties and combine them into a single instance. By doing so, you can create complex instances where the satisfiability depends on the interaction between the sub-instances. This approach allows for a more fine-grained control over the satisfiability of the problem.

In summary, there are multiple ways to construct instances of Model RB where the satisfiability can be easily changed while keeping the parameter values unchanged. By modifying the permitted set, introducing control variables, changing constraints, adding XOR constraints, or combining multiple instances, you can create instances of Model RB with varying satisfiability properties. Beginning here, we

gradually establish the transformation that changes the problem’s satisfiability while preserving invariance.

27 24, 26

Prompt: Can you provide a more specific solution for changing the satisfiability of a Model RB instance? I would like the resulting instance after the mapping to remain a valid Model RB instance. Please explain this with an example, specifically by applying the mapping to a constraint C = (X, R) of a Model RB instance with k = 2.

To change the satisfiability of a Model RB instance while keeping the parameter values and domains unchanged, we can modify the constraints by swapping variables or altering the permitted sets Ri. In this case, we’ll focus on altering the permitted sets without changing the variables in the permitted set.

Here’s a specific solution for changing the satisfiability of a Model RB instance with k = 2:

- 1. Start with an instance of Model RB, where each constraint Ci = (Xi, Ri), and Xi = {xi1, xi2}.
- 2. Define a new set of permitted values Ri′, which is derived from the existing set Ri, but with a different value mapping function. To do this, for each tuple (a, b) ∈ Ri, we will apply a new bijection gi on the first value a to obtain the new tuple (gi(a), b). Include this new tuple in Ri′.
- 3. Replace the original permitted set Ri with the modified permitted set Ri′ for each constraint Ci = (Xi, Ri). This results in a new constraint Ci′ = (Xi, Ri′).
- 4. The new instance of Model RB will have the same set of variables, domains, and constraints, but with the modified permitted sets Ri′ for each constraint.

As an example, let’s say we have a Model RB instance with a constraint C = (X, R), where

k = 2, X = {x1, x2}, and R = {(1,2), (2,3), (3,1)}. We define a new bijection g such that g(1) = 2, g(2) = 3, and g(3) = 1. Applying this bijection to the first values in the tuples of R, we get the new permitted set R′ = {(2,2), (3,3), (1,1)}.

Now, we replace the original constraint C = (X, R) with the modified constraint C′ = (X, R′). The new Model RB instance has the same variables, domains, and constraints, but with the modified permitted set R′, which may change its satisfiability.

By using this method, we can change the satisfiability of a Model RB instance while keeping the parameter values and domains unchanged. This approach allows us to create instances with different properties, which can be useful for experimentation and understanding the behavior of Model RB instances.

28 24, 26, 27

Prompt: Great! In the case of k = 2, how can we change the satisfiability while minimizing the impact of the mapping operation on the instance?

In the case of k = 2, to change the satisfiability while minimizing the impact of the mapping operation on the instance, we can focus on modifying the permitted sets in such a way that the alterations are minimal. One way to do this is by ensuring that the new bijections we choose are as close as possible to the original bijections. Here’s a specific approach to minimize the impact:

- 1. Start with an instance of Model RB, where each constraint Ci = (Xi, Ri), and Xi = {xi1, xi2}.
- 2. For each constraint Ci, find a minimal bijection gi that changes the satisfiability of Ri. To do this, you can try swapping a single pair of values in the domain or choosing the identity mapping (i.e., keeping the same values).
- 3. Apply the minimal bijection gi to the first value of each tuple in the original permitted set Ri to create the new permitted set Ri′.
- 4. Replace the original permitted set Ri with the modified permitted set Ri′ for each constraint Ci = (Xi, Ri). This results in a new constraint Ci′ = (Xi, Ri′).
- 5. The new instance of Model RB will have the same set of variables, domains, and constraints, but with the minimally modified permitted sets Ri′ for each constraint.

Let’s take the previous example:

- C = (X, R), where k = 2, X = {x1, x2}, and R = {(1,2), (2,3), (3,1)}. To minimize the impact of the mapping operation, we could choose bijection g such that

g(1) = 2, g(2) = 1, and g(3) = 3 (swapping only values 1 and 2). Applying this bijection to the first values in the tuples of R, we get the new permitted set R′ = {(2,2), (1,3), (3,1)}.

Now, we replace the original constraint C = (X, R) with the modified constraint C′ = (X, R′). The new Model RB instance has the same variables, domains, and constraints, but with the minimally modified permitted set R′, which may change its satisfiability.

By using this minimal impact approach, we can change the satisfiability of a Model RB instance while keeping the parameter values and domains unchanged, and ensure that the alterations to the instance are as small as possible. This can be useful for understanding the sensitivity of Model RB instances to small changes in their permitted sets.

29 24, 26, 27, 28

Prompt: Great! However, if the new tuple (after swapping a single pair of values) is already present in the permitted set, it will not change the constraint. Can you provide a more precise and concise definition of this mapping operation?

You’re correct. If the new tuple (after swapping a single pair of values) is already present in the permitted set, it will not change the constraint. To address this issue, we can use a more precise and concise definition of the mapping operation that guarantees a change in the constraint.

Let’s define the mapping operation as follows: Start with an instance of Model RB, where each constraint Ci = (Xi, Ri), and Xi = xi1, xi2. For each constraint Ci, find a bijection gi that changes the satisfiability of Ri by swapping a single pair of values in the domain, ensuring that the new tuple is not already in the permitted set. If no such bijection exists, try swapping two pairs of values. Repeat this process until you find a bijection that guarantees a change in the constraint.

Apply the bijection gi to the first value of each tuple in the original permitted set Ri to create the new permitted set Ri′.

Replace the original permitted set Ri with the modified permitted set Ri′ for each constraint Ci = (Xi, Ri). This results in a new constraint Ci′ = (Xi, Ri′).

The new instance of Model RB will have the same set of variables, domains, and con-

straints, but with the permitted sets Ri′ for each constraint, ensuring a change in the constraint.

Using this mapping operation, we can guarantee a change in the satisfiability of the Model RB instance while minimizing the impact on the permitted sets.

For example, let’s say we have a constraint C = (X, R) with k = 2, X = x1, x2, and R = (1,2), (2,3), (3,1). To find a bijection g that changes the constraint, we can first try swapping a single pair of values, like g(1) = 2, g(2) = 1, and g(3) = 3. However, since applying this bijection results in a tuple (2,2) that is already in the permitted set, we need to try another bijection. Suppose we try swapping two pairs of values: g(1) = 3, g(2) = 1, and g(3) = 2. Applying this bijection to the first values in the tuples of R, we get the new permitted set R′ = (3,2), (1,3), (2,1), which is different from the original set R. Now, we can replace the original constraint C = (X, R) with the modified constraint C′ = (X, R′), ensuring a change in the constraint while minimizing the impact on the instance.

30 24, 26, 27, 28, 29

Prompt: It seems good, but there might be two small issues: 1. The expression ‘changes the satisfiability of Ri’ doesn’t seem correct since Ri is a permitted set. 2. In your example, (2,2) is not in the permitted set. Please check for these issues, modify the definition of the mapping operation and make it more concise.

You are right, my previous response contained inaccuracies. I apologize for the confusion, and I appreciate your patience. Here’s a revised and more concise definition of the mapping operation to address these issues:

- 1. Start with an instance of Model RB, where each constraint Ci = (Xi, Ri), and Xi = {xi1, xi2}.

- 2. For each constraint Ci, find a bijection gi that modifies Ri by swapping a single pair of values in the domain while ensuring that the new tuples generated are not already in the permitted set. If no such bijection exists, try swapping two pairs of values. Repeat this process until you find a bijection that guarantees a change in Ri.

- 3. Apply the bijection gi to the first value of each tuple in the original permitted set Ri to create the new permitted set Ri′.

- 4. Replace the original permitted set Ri with the modified permitted set Ri′ for each constraint Ci = (Xi, Ri). This results in a new constraint Ci′ = (Xi, Ri′).

- 5. The new instance of Model RB will have the same set of variables, domains, and constraints, but with the modified permitted sets Ri′ for each constraint.

For example, let’s say we have a constraint C = (X, R) with k = 2, X = {x1, x2}, and R = {(1,2), (2,3), (3,1)}. To find a bijection g that modifies the permitted set, we can first try swapping a single pair of values: g(1) = 2, g(2) = 1, and g(3) = 3. Applying this bijection to the first values in the tuples of R, we get the new permitted set R′ = {(2,2), (1,3), (3,1)}, which is different from the original set R. Now, we can replace the original constraint C = (X, R) with the modified constraint C′ = (X, R′), ensuring a change in the permitted set while minimizing the impact on the instance. Our objective is presented

as a theorem for a meticulous proof and is denoted as 31∗ in later references. Motivated by 9, we consider the scenario where the number of variables is infinitely large, as it is more prone to generate extremely hard instances with almost independent solution space. In this case, we can use ‘an infinite set’ because the instance-generating process of Model RB is random, which signifies that there could be infinite possible instances as the number of variables or the domain size approaches infinity.

31 24, 25, 30

Prompt: We want to prove the Theorem: There exists an infinite set of satisfiable and unsatisfiable instances of Model RB such that this set is a fixed point under the mapping operation of changing satisfiability. (n tends to infinity, d tends to infinity.)

What thinking method should we use to conceive the proof plan?

To prove the theorem, we can take a step-by-step approach, combining various thinking methods such as constructive reasoning, logical reasoning, and contradiction.

- 1. Constructive reasoning: First, we need to demonstrate the existence of an infinite set of satisfiable and unsatisfiable instances of Model RB. This can be done by constructing a sequence of instances with different parameters and showing that the satisfiability of these instances alternates between satisfiable and unsatisfiable.

- 2. Logical reasoning: Next, we need to show that the mapping operation of changing satisfiability indeed alters the satisfiability of the instances in the set. This can be achieved by analyzing the properties of the mapping operation and proving that the satisfiability status is changed when the operation is applied.
- 3. Contradiction: To show that the set is a fixed point under the mapping operation of changing satisfiability, we can assume the opposite, i.e., that the set is not a fixed point, and then derive a contradiction. This will involve analyzing the properties of the mapping operation and the instances in the set and showing that the assumption leads to an inconsistency.
- 4. Asymptotic behavior: Since we need to prove the theorem for n and d tending to infinity, we should also analyze the asymptotic behavior of the instances and the mapping operation. This will involve studying the growth rates of the parameters and constraints and how they affect the satisfiability of the instances.

For intricate problems, ask LLMs to lay out a plan for problem deconstruction. However, if the plans or solutions appear too restricted or singular, prompt LLMs to explore various thinking methods first. This strategy inherently broadens the search space of LLMs.

By combining these thinking methods in a coherent and structured proof plan, we can establish the existence of such an infinite set and show that it is a fixed point under the mapping operation of changing satisfiability. When the context

is long, prompt the model to summarize or explain critical conditions or goals for a better understanding.

32 24, 25, 30, 31*

Prompt: Please explain the theorem to me, lay out a high-level proof plan, and then try various tactics to prove the theorem.

Theorem: There exists an infinite set of satisfiable and unsatisfiable instances of Model RB such that this set is a fixed point under the mapping operation of changing satisfiability. (n tends to infinity, d tends to infinity.) High-level proof plan:

- 1. Existence of an infinite set of satisfiable and unsatisfiable instances: First, we need to demonstrate the existence of an infinite set of instances of Model RB with alternating satisfiability. This can be achieved by constructing a sequence of instances with different parameters and showing that the satisfiability of these instances alternates between satisfiable and unsatisfiable.

- 2. Mapping operation changes satisfiability: Next, we need to show that the mapping operation of changing satisfiability indeed alters the satisfiability of the instances in the set. This can be achieved by analyzing the properties of the mapping operation and proving that the satisfiability status is changed when the operation is applied.

- 3. Fixed point under the mapping operation : To show that the set is a fixed point under the mapping operation of changing satisfiability, we can assume the opposite, i.e., that the set is not a fixed point, and then derive a contradiction. This involves analyzing the properties of the mapping operation and the instances in the set and showing that the assumption leads to an inconsistency.
- 4. Asymptotic behavior: Since we need to prove the theorem for n and d tending to infinity, we should also analyze the asymptotic behavior of the instances and the mapping operation. This will involve studying the growth rates of the parameters and constraints and how they affect the satisfiability of the instances. Proof:

- • Step 1: Existence of an infinite set of satisfiable and unsatisfiable instances. Consider a sequence of instances of Model RB with increasing values of n and d. By adjusting the parameters of the model, such as the constant α, the probability p, and the number of constraints m, we can create a sequence of instances with alternating satisfiability. As n and d tend to infinity, the sequence becomes infinite.
- • Step 2: Mapping operation changes satisfiability. We have already discussed how the mapping operation changes the satisfiability of an instance by altering the mapping function while keeping the parameter values and domains unchanged. By analyzing the properties of the mapping operation, we can show that it indeed changes the satisfiability of the instances in the set.
- • Step 3: Fixed point under the mapping operation. Suppose the set is not a fixed point under the mapping operation. Then, there exists an instance in the set whose satisfiability is not changed by the mapping operation. However, this contradicts the result from Step 2, which states that the mapping operation changes the satisfiability of the instances in the set. Thus, the set must be a fixed point under the mapping operation of changing satisfiability.
- • Step 4: Asymptotic behavior. As n and d tend to infinity, the number of variables and the domain size increase. By analyzing the growth rates of the parameters and constraints, we can show how they affect the satisfiability of the instances. For example, as the domain size increases, the probability of randomly selecting a satisfying assignment also increases. Similarly, as the number of constraints increases, the likelihood of an unsatisfiable instance also increases. By carefully choosing the parameters, we can create an infinite set of instances with alternating satisfiability.

Conclusion: We have shown that there exists an infinite set of satisfiable and unsatisfiable instances of Model RB such that this set is a fixed point under the mapping operation of

changing satisfiability as n and d tend to infinity. This is achieved by constructing a sequence of instances with alternating satisfiability, proving that the mapping operation changes satisfiability, showing that the set is a fixed point under the mapping operation, and analyzing the asymptotic behavior of the instances and the mapping operation.

33 24, 25, 30, 31*

Prompt: First, please clarify what kinds of instances are unsatisfiable and what kinds are satisfiable. Then, as we only need to prove the existence, perhaps we can consider the instances whose satisfiability can be changed easily, so as to simplify the proof. What kind of instances are most suitable?

To prove the theorem, we first need to understand the characteristics of the satisfiable and unsatisfiable instances of Model RB. We will then construct an infinite set of instances that can be easily altered between satisfiability and unsatisfiability.

- 1. Satisfiable instances: These instances have a solution that satisfies all the constraints in the set C. In other words, there exists an assignment σ such that for each constraint Ci = (Xi, Ri), the tuple (σ(xi1), σ(xi2), . . . , σ(xik)) ∈ Ri.
- 2. Unsatisfiable instances: These instances do not have any solution that satisfies all the constraints in the set C. In other words, for every assignment σ, there exists at least one constraint Ci = (Xi, Ri) such that the tuple (σ(xi1), σ(xi2), . . . , σ(xik)) ∈/ Ri.

Now, let’s consider instances that can be easily altered between satisfiability and unsatisfiability. We can focus on instances where only a single variable assignment needs to be changed to switch between satisfiability and unsatisfiability. More specifically, we can construct instances where only one constraint is “critical” to the satisfiability of the instance, and the remaining constraints are always satisfied.

Consider a sequence of instances I1, I2, . . ., where each instance Ij has nj variables, dj domain size, and mj constraints. We construct the instances such that nj, dj → ∞ as j → ∞. For each instance Ij, we construct the set of constraints Cj such that all but one constraint are “noncritical” and can always be satisfied, while the remaining constraint is “critical” and its satisfiability depends on a single variable assignment.

To ensure that these instances are fixed points under the mapping operation of changing satisfiability, we need to ensure that the satisfiability of each instance can be easily altered by applying the mapping operation on the “critical” constraint. In other words, the mapping operation should transform the “critical” constraint from satisfiable to unsatisfiable or vice versa, while keeping the other constraints unchanged.

Let Ij be an instance in the sequence, and let Cj,1 be the “critical” constraint of the instance.

We can apply the mapping operation on Cj,1 to create a new instance Ij′ with the altered satisfiability. Since the mapping operation only modifies the “critical” constraint, the remain-

ing constraints in Ij′ will be unchanged, and their satisfiability will remain the same as in Ij. Therefore, by construction, the infinite set of instances {I1, I2, . . .} is a fixed point under the mapping operation of changing satisfiability.

In conclusion, we have shown the existence of an infinite set of satisfiable and unsatisfiable instances of Model RB such that this set is a fixed point under the mapping operation of changing satisfiability as n and d tend to infinity. We achieved this by constructing instances where only one constraint is “critical” to the satisfiability of the instance, and the remaining constraints are always satisfied. By applying the mapping operation on the “critical” constraint, we can easily alter the satisfiability of the instances while keeping the parameter values and domains unchanged.

34 24, 25, 30, 31*, 33

Prompt: Great! You have found instances that seem likely to be transformed from unsatisfiable to satisfiable. So, what kind of instances are more likely to be transformed from satisfiable to unsatisfiable?

To find instances that are more likely to be transformed from satisfiable to unsatisfiable, we can focus on instances that have a very delicate balance between the constraints. These instances should have a unique solution that satisfies all constraints, but a small change in any constraint could break the satisfiability.

Consider a sequence of instances J1, J2, . . ., where each instance Jj has nj variables, dj domain size, and mj constraints. We construct the instances such that nj, dj → ∞ as j → ∞. For each instance Jj, we construct the set of constraints Cj such that all constraints are satisfied by a unique solution σj. However, the constraints are tightly coupled, meaning that even a small change in any of the constraints could make the instance unsatisfiable.

Let Jj be an instance in the sequence, and let Cj,1 be one of the constraints in the instance. We can apply the mapping operation on Cj,1 to create a new instance J′j with the altered satisfiability. Since the mapping operation modifies the constraint Cj,1, it is likely to break

the delicate balance between the constraints in Jj and make the new instance Jj′ unsatisfiable.

To ensure that these instances are more likely to be transformed from satisfiable to unsatisfiable, we can focus on constructing instances with the following properties:

- 1. The unique solution σj should have very specific variable assignments that make all constraints in Jj satisfied. Even a small change in any variable assignment should result in at least one constraint being unsatisfied.
- 2. The constraints should have a high degree of interdependence, meaning that satisfying one constraint may require other constraints to be satisfied as well. This increases the likelihood that a small change in any constraint will render the entire instance unsatisfiable.
- 3. The constraints should have a narrow range of permitted tuples, making it more likely that a small change in any constraint will result in an unsatisfiable instance.

By constructing instances with these properties, we increase the likelihood that the mapping operation will transform the instances from satisfiable to unsatisfiable. Thus, we have shown that there are instances that are more likely to be transformed from satisfiable to unsatisfiable under the mapping operation, while keeping parameter values and domains unchanged.

35 24, 25, 30, 31*, 33, 34

Prompt: Great! Please summarize the instances that we can focus on, be as concise as possible.

We can focus on two types of instances:

- 1. From unsatisfiable to satisfiable: Instances with mostly non-critical constraints that are always satisfied, and one critical constraint that depends on a single variable assignment. Changing this assignment alters the instance’s satisfiability.

- 2. From satisfiable to unsatisfiable: Instances with a delicate balance between constraints, having a unique solution. Modifying one critical constraint disrupts this balance and makes the instance unsatisfiable.

36 24, 25, 30, 31*, 33, 34, 35

Prompt: Great! Let’s prove the theorem for the second type of instances. Please explain the theorem to me, lay out a high-level proof plan, and then try various tactics to prove the theorem.

Theorem: There exists an infinite set of instances of Model RB that can be transformed from satisfiable to unsatisfiable under the mapping operation while keeping the parameter values and domains unchanged.

High-level proof plan:

- 1. Construct an infinite sequence of instances I1, I2, . . . such that each instance Ij is satisfiable and has a unique solution.

- 2. Show that applying the mapping operation on one critical constraint in each instance Ij makes the instance unsatisfiable.

- 3. Demonstrate that the infinite set of instances is a fixed point under the mapping operation of changing satisfiability as n and d tend to infinity. Proof:

- 1. Constructing the infinite sequence of instances:

Consider a sequence of instances I1, I2, . . ., where each instance Ij has nj variables, dj domain size, and mj constraints. We construct the instances such that nj, dj → ∞ as j → ∞. For each instance Ij, we carefully construct the set of constraints Cj such that there exists a unique solution that satisfies all constraints. This can be achieved by ensuring that the constraints are tightly interconnected, and the variable assignments are coordinated to create a unique solution.

- 2. Applying the mapping operation on a critical constraint:

For each instance Ij, select a critical constraint Cj,1 that is part of the unique solution. Apply the mapping operation on this constraint to create a new instance Ij′. Since the mapping operation modifies the critical constraint, the delicate balance of the constraints is disrupted, and the unique solution of Ij no longer satisfies all constraints in Ij′. Therefore, each instance Ij′ becomes unsatisfiable.

- 3. Demonstrating the fixed point property:

The infinite set of instances {I1, I2, . . .} is a fixed point under the mapping operation of changing satisfiability as n and d tend to infinity because the mapping operation can always be applied to the critical constraint in each instance to alter its satisfiability. When the mapping operation is applied to an instance in the sequence, its satisfiability changes from satisfiable to unsatisfiable, and the parameter values and domains remain unchanged.

In conclusion, we have shown the existence of an infinite set of instances of Model RB that can be transformed from satisfiable to unsatisfiable under the mapping operation while keeping the parameter values and domains unchanged. We achieved this by constructing instances with a delicate balance of constraints and a unique solution, and then applying the mapping operation on a critical constraint to disrupt this balance and make the instances unsatisfiable.

The second type appears simpler, hence we begin with proving the second type.

37 24, 25, 36

Prompt: : Denote Pr[I is SAT ] the probability that a random instance I of Model RB is satisfiable, then we have Theorem A: Let rcr = −ln(11−p). If α > 1/k,0 < p < 1 are two constants and k, p satisfy the inequality k ≥ 1/(1 − p), then

Pr[I is SAT] = 1 if r < rcr, lim n→∞

lim

n→∞

Pr[I is SAT] = 0 if r > rcr.

Now we want to prove: there exist infinitely many satisfiable instances of Model RB with a unique solution when n and d tend to infinity. Please abstract the essence of this proof problem and outline 10 possible ideas for the proof.

When model knowledge is insufficient for calculations, we integrate the known properties of Model RB (from related work Xu and Li (2000)) into prompts.

To prove that there exist infinitely many satisfiable instances of Model RB with a unique solution when n and d tend to infinity, we can consider the following ideas:

- 1. Bounding the probability of multiple solutions: Analyze the probability of a random instance having multiple solutions, and show that this probability tends to zero as n and d tend to infinity, ensuring that most instances have a unique solution.
- 2. Exploit the phase transition: Utilize the phase transition between satisfiable and unsatisfiable instances in Model RB to identify a range of parameters where instances are likely to be satisfiable with a unique solution.
- 3. Constructing instances with a unique solution: Develop a method for constructing instances of Model RB with a unique solution, and show that this method can be applied to generate infinitely many instances as n and d tend to infinity.
- 4. Analyzing the solution space geometry: Investigate the geometry of the solution space of Model RB instances, and show that the structure of the space tends to favor instances with a unique solution as n and d grow without bound.
- 5. Random walk-based argument: Use a random walk-based argument to show that, in the limit as n and d tend to infinity, the probability of reaching a second solution from a given solution becomes vanishingly small, implying the existence of a unique solution.
- 6. Exploiting the symmetries of the problem: Leverage the symmetries in Model RB to argue that there exist instances with a unique solution up to symmetries, and that these instances become more prevalent as n and d tend to infinity.
- 7. Probabilistic method: Apply the probabilistic method to demonstrate that there exists a positive probability for an instance of Model RB to have a unique solution as n and d tend to infinity, thus proving the existence of infinitely many such instances. The constraint gener-

ation process in Model RB is independent, allowing the subsequent probability analysis to meet the requirements of independence.

- 8. Average-case analysis: Conduct an average-case analysis of Model RB instances, showing that the expected number of solutions tends to one as n and d grow without bound, which implies the existence of instances with a unique solution.
- 9. Sparse instances: Investigate the behavior of sparse instances of Model RB (i.e., instances with a low constraint density) and show that as n and d tend to infinity, such instances are more likely to have a unique solution.
- 10. Local search algorithms: Analyze the performance of local search algorithms on Model RB instances and argue that the difficulty of finding multiple solutions increases as n and d

To ensure the integrity of subsequent derivations, we include the conditions α > 1/k,0 < p < 1 and k ≥ 1/(1 − p) every time we use the definition of Model RB in the upcoming prompts.

tend to infinity, suggesting that instances with a unique solution become more common in the limit.

38 24, 25, 37*

Prompt: Now we want to prove: how to demonstrate that there exists a positive probability for an instance of Model RB to have a unique solution as n and d tend to infinity. Please abstract the essence of this proof problem and outline possible ideas for the proof. essence of this proof problem and outline 10 possible ideas for the proof.

To prove that there exists a positive probability for an instance of Model RB to have a unique solution as n and d tend to infinity, we need to show that the limiting probability of finding a unique solution is greater than zero. We can do this by analyzing the behavior of the model as n and d increase and finding the conditions under which a unique solution exists.

Abstracting the essence of the problem, we have:

- 1. A random constraint satisfaction problem (CSP) model with variables, domains, constraints, and relations.
- 2. A satisfiability condition for the model, dependent on the parameters n, d, k, p, and r.
- 3. A theorem (Theorem A) giving the limiting probability of satisfiability under certain conditions.
- 4. A target: proving the existence of a positive probability for a unique solution as n and d tend to infinity. Here are possible ideas and steps for the proof:

- 1. Study the behavior of the model: Analyze how the parameters n, d, k, p, and r affect the instance’s satisfiability and the number of solutions. This analysis may involve counting arguments, combinatorial techniques, and probabilistic methods.
- 2. Identify conditions for uniqueness: Determine the conditions on the parameters that lead to a unique solution. This may involve exploring the parameter space and finding where the probability of having multiple solutions becomes negligible.
- 3. Establish a lower bound: Prove a lower bound on the probability of having a unique solution under the identified conditions. This may involve using techniques from probability theory, such as the first or second moment methods, or applying concentration inequalities like Chebyshev’s inequality or Chernoff bounds.

- 4. Relate the lower bound to Theorem A: Show that the lower bound on the probability of a unique solution is consistent with the conditions of Theorem A. This may involve proving that the lower bound is positive when the conditions of Theorem A are satisfied.
- 5. Conclude the proof: Demonstrate that the positive probability of a unique solution exists as n and d tend to infinity, completing the proof.

In summary, to prove the existence of a positive probability for a unique solution in the Model RB as n and d tend to infinity, we need to analyze the behavior of the model, identify conditions for uniqueness, establish a lower bound on the probability of a unique solution, relate the lower

LLMs focus on user demands and instructions. If a statement requests a specific format, the model may continue using it, even if not needed. To avoid confusion, remove past instructions from the conversation history.

We utilize the notation 37* to indicate the incorporation of Theorem A into the prompt.

bound to Theorem A, and conclude the proof. This will involve using various mathematical techniques, including combinatorics, probability theory, and concentration inequalities.

39 24, 25, 37*

Prompt: Problem: How to establish a lower bound on the probability of the conditions required for a unique solution. Please first abstract the essence of this problem and then outline possible ideas.

Essence of the problem: The problem deals with the probability of satisfiability (having a unique solution) of a random constraint satisfaction problem instance (Model RB) based on various parameters such as the number of constraints (r), the domain size (d), and the size of symmetric relation (|R∗|). The conditions for a unique solution are given by Theorem A, which establishes a critical threshold (rcr) that separates the cases when the probability of satisfiability tends to 1 (satisfiable) or 0 (unsatisfiable). The main goal is to establish a lower bound on the probability of the conditions required for a unique solution. Outline of possible ideas:

- 1. Analyze the structure of Model RB instances: Investigate the structural properties of Model RB instances and their dependencies on the parameters r, d, and |R∗|. This will help to identify the characteristics that contribute to the satisfiability of the instances.
- 2. Study the phase transition phenomenon: In random CSPs, there is often a phase transition around a critical threshold where the probability of satisfiability changes abruptly. Analyze

this phase transition in Model RB and its relation to the critical threshold rcr given by Theorem A.

- 3. Probabilistic analysis: Perform a probabilistic analysis to calculate the lower bound on the probability of the conditions required for a unique solution. This can be done using techniques such as the first and second moment methods, which involve computing the expected number of solutions and the variance of the number of solutions, respectively.

- 4. Explore connections with other CSP models and problems: Investigate how the results or techniques used for similar CSP models or NP-complete problems (e.g., SAT, CLIQUE) can be adapted or applied to Model RB to establish the lower bound on the probability of a unique solution.
- 5. Investigate the effect of different parameters: Analyze the impact of varying the parameters r, d, and |R∗| on the probability of a unique solution. This analysis can help in understanding the interplay between these parameters and their role in determining the satisfiability of Model RB instances.
- 6. Use computational methods: Employ computational methods such as simulations or solving instances of Model RB with different parameters using constraint solvers to gather empirical evidence supporting the lower bound on the probability of a unique solution.

40 24, 25, 37*, 39

Prompt: Great! Can you give me some possible ideas?

Certainly! Here are some possible ideas to establish the lower bound on the probability of the conditions required for a unique solution in Model RB:

- 1. First Moment Method: Compute the expected number of solutions, E[S], where S is a random variable representing the number of solutions in a random instance of Model RB.

- If E[S] is small, it indicates that the probability of having a unique solution is high. To do this, you can use linearity of expectation and indicator random variables for each potential solution.
- 2. Second Moment Method: Compute the variance of the number of solutions, Var(S), to obtain information about the distribution of solutions. If the variance is small, it implies that most instances have a number of solutions close to the expected value, E[S]. You can combine this information with the first moment method to get a better understanding of the probability of a unique solution.
- 3. Bounds on Satisfiability: Establish upper and lower bounds on the probability that a random instance of Model RB is satisfiable. This can help in determining the threshold of the parameter r (constraint density) that separates the regions of satisfiability and unsatisfiability, as mentioned in Theorem A. Understanding this threshold can give insights into the likelihood of having a unique solution.

- 4. Phase Transition Analysis: Investigate the behavior of Model RB around the critical value

of r (rcr) where the probability of satisfiability changes abruptly. Phase transition analysis can provide information about how the probability of having a unique solution changes with different values of r and other parameters, such as k and p.

- 5. Probabilistic Inequalities: Employ probabilistic inequalities, such as Chebyshev’s or Chernoff’s inequalities, to obtain bounds on the probability of a unique solution. These inequalities can be applied to the random variables related to the number of solutions and their distribution.
- 6. Exploit Problem Structure: Analyze the structure of Model RB, such as its graph representation or properties of its constraints, to identify conditions that promote a unique solution. For example, you can study how the choice of k, p, and the relation R∗ affect the likelihood of a unique solution in Model RB.

By exploring these ideas and combining their results, you can establish a lower bound on the probability of the conditions required for a unique solution in Model RB instances.

41 24, 25, 37*, 39, 40

Prompt: Great! How to establish the the upper bound?

To establish the upper bound on the probability that a random instance of Model RB is satisfiable, we can use the first moment method. Here’s how:

- 1. Compute the expected number of solutions: Calculate the expected number of solutions E(S) for a random instance of Model RB by taking the product of the probability that each constraint is satisfied by a random assignment. To do this, consider the probability of a single constraint being satisfied and raise it to the power of the total number of constraints, m.

- 2. Use Markov’s inequality: Apply Markov’s inequality to bound the probability that a random instance has at least one solution. Markov’s inequality states that for any non-negative random variable X and a > 0, Pr(X ≥ a) ≤ E(aX). In this case, X represents the number of solutions and a = 1.

- 3. Upper bound: By combining the calculation of E(S) and Markov’s inequality, we can establish an upper bound on the probability that a random instance of Model RB is satisfiable.

If E(S) is small, then the probability of having at least one solution is also small, indicating an upper bound on the satisfiability probability.

This upper bound will depend on the values of the parameters α, p, k, and r. By analyzing the bound, we can gain insights into the behavior of the satisfiability probability as the parameters vary and help identify the threshold values for r that separate the satisfiable and unsatisfiable regions.

42 24, 25, 37*, 39, 40, 41

Prompt: Great! Let’s calculate the upper bound.

To calculate the upper bound on the probability that a random instance of Model RB is satisfiable, we first compute the expected number of solutions E(S) and then apply Markov’s inequality.

- 1. Compute the expected number of solutions E(S): For a single constraint Ci with k variables, the probability of being satisfied by a random assignment is |Rdk∗| = (1−dpk)dk = 1 − p. Since there are m constraints, and we assume they are independent, the probability that all of them are satisfied is (1 − p)m.

Now, there are dn = nαn possible assignments for the variables in the instance. So, the expected number of solutions, E(S), is the product of the number of possible assignments and the probability that each assignment satisfies all constraints:

E(S) = nαn(1 − p)m

- 2. Apply Markov’s inequality:

Let X be the random variable representing the number of solutions. We want to bound the probability that X ≥ 1, i.e., the probability that there exists at least one solution. Using Markov’s inequality, we have:

Pr[X ≥ 1] ≤

E(S) 1

- 3. Upper bound:

By substituting the expression for E(S) into the inequality, we get an upper bound on the probability that a random instance of Model RB is satisfiable:

Pr[I is SAT] ≤ nαn(1 − p)m

To determine the threshold value for r that separates satisfiable and unsatisfiable regions, we can analyze the behavior of this upper bound as the parameters α, p, k, and r vary. In particular, when the upper bound is small, it indicates that the probability of having a satisfiable instance is small, helping us identify the critical value rcr mentioned in Theorem A.

43 24, 25, 37*, 39, 40, 41, 42

Prompt: We are aiming to construct a challenging example, so it is sufficient to ensure that the probability of I having a solution and the probability of I being unsatisfiable are both positive. Please set a specific value for nαn(1 − p)m and specify the parameter selection requirements accordingly during the construction process.

Let’s choose a specific value for nαn(1 − p)m to ensure that the probability of I having a solution and the probability of I being unsatisfiable are both positive. For example, we can set:

- 1

- 2

nαn(1 − p)m =

This means that the probability of a random instance I being satisfiable is less than or equal to 1/2, and the probability of it being unsatisfiable is greater than or equal to 1/2. This choice balances the probabilities and makes the problem challenging.

Now, we need to specify the parameter selection requirements during the construction process to satisfy this condition:

- 1. Choose a suitable value for α, such that α > 0. This will affect the domain size d and the number of possible assignments for the variables in the instance.
- 2. Choose a constant p, where 0 < p < 1. This represents the fraction of tuples not included in the permitted sets and determines the difficulty of constraints.
- 3. Select a constant k ≥ 2, the arity of the constraints. This choice will impact the complexity of the constraints and the overall problem.
- 4. Determine the constraint density parameter r. Ensure that the value of r is chosen such

that it relates to rcr, the critical value mentioned in Theorem A. This will help in creating instances that are challenging to solve.

To satisfy the equation nαn(1 − p)m = 12, tune the values of α, p, k, and r during the construction process. Keep in mind that the chosen parameters should abide by the conditions mentioned in Theorem A and should not violate any constraints imposed by the Model RB definition.

In summary, carefully selecting the values of α, p, k, and r during the construction process, according to the equation, will ensure that the probability of the instance having a solution and the probability of it being unsatisfiable are both positive, making the problem challenging.

44 24, 25, 37*, 39, 40, 41, 42, 43

Prompt: Great! Considering Theorem A, which kind of r will guarantee the inequality nαn(1 − p)m = 21 holds true?

To ensure that nαn(1 − p)m = 21 holds true, we need to select an appropriate value for the constraint density parameter r. Based on Theorem A, we know that:

- • If r < rcr, the probability of I being satisfiable tends to 1 as n → ∞.
- • If r > rcr, the probability of I being satisfiable tends to 0 as n → ∞.

In our case, we want the probability of I being satisfiable to be less than or equal to 1/2. To achieve this, we should choose r slightly greater than rcr.

Recall that rcr = −ln(11−p). Given the constant p, we can calculate rcr and then select r ≈ rcr + ϵ, where ϵ > 0 is a small constant.

By choosing r in this manner, we can expect that the probability of a random instance I being satisfiable will be close to 1/2, satisfying the condition nαn(1 − p)m = 12.

Keep in mind that the actual probability of I being satisfiable will depend on the specific values of α, p, k, and r. To ensure that the equation holds true, you might need to perform some numerical analysis and fine-tune the parameters accordingly.

45 24, 25, 37*, 39, 40, 41, 42, 43, 44

Prompt: Great! What kind of ϵ guarantees that nαn(1 − p)m = 21? Please calculate a proper ϵ (represent it with other variables).

To find a suitable ϵ that guarantees nαn(1 − p)m = 21, we need to analyze the effect of changing r on the equation while considering the other parameters as well. Using the given values of α, p, and k, we can express m as m = rn ln d = rn ln(nα), which can be further simplified to m = rαn2 ln n. Now, let’s rewrite the equation as:

- 1

- 2

nαn(1 − p)rαn2lnn = Taking the natural logarithm of both sides, we get:

- 1

- 2

αn ln n + rαn2 ln n ln(1 − p) = ln

Now, we want to find the change in r, denoted as ϵ, such that the equation holds true. To do this, we can take the partial derivative of the left-hand side of the equation with respect to r:

∂ ∂r

(αn ln n + rαn2 ln n ln(1 − p)) = αn2 ln n ln(1 − p)

We can approximate the change in the left-hand side of the equation caused by an increase in r by the product of the partial derivative and ϵ:

∆(LHS) ≈ ϵαn2 ln n ln(1 − p)

Now, we need to find an ϵ such that the change in the left-hand side of the equation is equal to the change in the right-hand side of the equation, which is ln 21:

- 1

- 2

ϵαn2 ln n ln(1 − p) = ln We can solve for ϵ:

ln 12 αn2 ln n ln(1 − p)

ϵ =

This expression represents a suitable ϵ value in terms of the other variables that should

guarantee that nαn(1 − p)m = 21. However, keep in mind that this is an approximation, and you might need to fine-tune the value of ϵ depending on the specific values of the other

parameters and the behavior of the equation.

46 24, 25, 37*, 39, 40, 41, 42, 43, 44, 45

Prompt: Great! Please check your calculation process and refine any possible mistakes.

Apologies for the confusion in my previous response. I made a mistake in the calculation process. Let’s correct the expression for m and re-derive the equation for ϵ.

First, we recall that m = rn ln d = rn ln(nα). Now, we can simplify this expression as m = rαn ln n. Next, let’s rewrite the equation as:

For conciseness, we maintain the condition r = rcr + ϵ, ϵ =

ln 12 αnlnnln(1−p) by default when constructing Model RB instances in our subsequent discussions.

- 1

- 2

nαn(1 − p)rαnlnn = Taking the natural logarithm of both sides, we get:

- 1

- 2

αn ln n + rαn ln n ln(1 − p) = ln

Now, we want to find the change in r, denoted as ϵ, such that the equation holds true. To do this, we can take the partial derivative of the left-hand side of the equation with respect to r:

∂ ∂r

(αn ln n + rαn ln n ln(1 − p)) = αn ln n ln(1 − p)

We can approximate the change in the left-hand side of the equation caused by an increase in r by the product of the partial derivative and ϵ:

∆(LHS) ≈ ϵαn ln n ln(1 − p)

Now, we need to find an ϵ such that the change in the left-hand side of the equation is equal to the change in the right-hand side of the equation, which is ln 12:

- 1

- 2

ϵαn ln n ln(1 − p) = ln We can solve for ϵ:

ln 12 αn ln n ln(1 − p)

ϵ =

This expression represents a suitable ϵ value in terms of the other variables that should

guarantee that nαn(1 − p)m = 21. However, keep in mind that this is an approximation, and you might need to fine-tune the value of ϵ depending on the specific values of the other

parameters and the behavior of the equation. When the LLM falls short for computations, we incorporate the established equations of Model RB solutions from previous work (Xu and Li, 2000) into the prompts.

47 24, 25, 37*, 39, 40, 43

Prompt: Let X denote the number of solutions of an instance of Model RB, then we have:

rn ln d

(Sk) (nk)

(Sk) (nk)

n

n S

+ (1 − p)2 1 −

E[X2] =

dn

(d − 1)n−S (1 − p)

∑

,

S=0

where S = ns is the number of variables for which an assignment pair take the same values. We have also known:

n

1 n

E X2 = E[X]2 1 + O

∑

F(S),

S=0

rn ln d

n−S

S

1 + 1−ppsk

where F(S) = (nS) 1 − d1

1 d

, and S = ns is the number of variables for which an assignment pair take the same values. Notice that α > 1, according to previous work, we know that all the elements are negligible except when S=0 and S=n. We want to calculate the lower bound when n and d tend to infinity, can you lay out a plan?

The condition for all other elements to be negligible is 1 + α(1 − rcrpk) < 0 and α > 1. For the sake of conciseness, we will maintain this condition by default when constructing Model RB instances henceforth.

To calculate the lower bound when n and d tend to infinity, we can follow these steps:

- 1. Simplify the expression for E[X2] by considering only the terms when S = 0 and S = n. Since all other terms are negligible, we have:

1 n

E[X2] ≈ E[X]2 1 + O

###### (F(0) + F(n)).

- 2. Compute the asymptotic expressions for F(0) and F(n) as n, d → ∞: For F(0), we have:

F(0) =

n 0

1 −

1 d

n 1 d

- 0
- 1 +

p 1 − p

(0)k

rn ln d

= 1 −

1 d

n

.

As d → ∞, we have limd→∞ F(0) = 1. For F(n), we have:

F(n) =

n n

1 −

1 d

0 1 d

n

1 +

p 1 − p

(n)k

rn ln d

=

1 dn

1 +

p 1 − p

nk

rn ln d

.

We need to analyze the behavior of d1n and 1 + 1−ppnk

rn ln d

as n, d → ∞. This might require advanced mathematical techniques such as using L’Hôpital’s rule, series expansions, or other asymptotic analysis methods.

- 3. Substitute the asymptotic expressions for F(0) and F(n) back into the expression for E[X2]:

E[X2] ≈ E[X]2 1 + O

1 n

(F(0) + F(n)).

- 4. Compute the lower bound by taking the limit as n, d → ∞:

E[X]2 E[X2]

lim

Pr[I is SAT] ≥ lim

.

n,d→∞

n,d→∞

By analyzing the asymptotic behavior of the terms in the expression for E[X2], we can compute the lower bound as n and d tend to infinity. This analysis will provide insight into the behavior of the probability of satisfiability for large instances of Model RB and help understand the phase transition phenomenon.

48 24, 25, 37*, 39, 40, 43, 47

Prompt: Great! But can you check the correctness of your F(n)?

You are correct. There was an error in my calculation for F(n). Let me correct the expression for F(n):

0 1 d

rn ln d

rn ln d

n

1 d

1 dn

n n

p 1 − p

p 1 − p

(1)k

1 −

1 +

1 +

F(n) =

=

Now, we can continue with the plan to calculate the lower bound as n and d tend to infinity, by analyzing the behavior of d1n and 1 + 1−pp

rn ln d

as n, d → ∞. This analysis will provide insight into the behavior of the probability of satisfiability for large instances of Model RB and help understand the phase transition phenomenon.

49 24, 25, 37*, 39, 40, 43, 47, 48

Prompt: Great! Please don’t forget that our construction makes the instances have the following property: E[X] = nαn(1 − p)m = 12. Now let’s use the corrected F(n), simplify it and calculate the final lower bound on the probability that a random instance of Model RB is satisfiable.

Given the property E[X] = nαn(1 − p)m = 12, we can simplify the expression for E[X2] by considering only the terms when S = 0 and S = n:

1 n

E[X2] ≈ E[X]2 1 + O

(F(0) + F(n)). Now, we have the corrected expressions for F(0) and F(n):

- 1. F(0) = limd→∞ 1 − 1d

n

= 1.

- 2. F(n) = d1n 1 + 1−pp

rn ln d

.

rn ln d

To simplify F(n), let’s analyze the behavior of d1n and 1 + 1−pp

as n, d → ∞. First, recall that d = nα. Then, we have:

1 dn

1 (nα)n

= n−αn.

=

rn ln d

Next, notice that 1 + 1−pp

is an exponential function of n. To simplify this term, we can use the property E[X] = 12. From this property, we know that:

- 1

- 2nαn

- 1

- 2

=⇒ (1 − p)rnlnd =

nαn(1 − p)m =

rn ln d

Now, we can rewrite the term 1 + 1−pp

as:

rn ln d

rn ln d

rn ln d

1 − p + p 1 − p

1 1 − p

p 1 − p

1 +

=

=

Since (1 − p)rnlnd = 2n1αn , we have:

rn ln d

p 1 − p

= 2nαn. Finally, we can calculate F(n) as:

1 +

.

1 (1 − p)rnlnd

.

=

F(n) = n−αn · 2nαn = 2.

Now, we can substitute the simplified expressions for F(0) and F(n) back into the expression for E[X2]:

1 n

E[X2] ≈ E[X]2 1 + O

(1 + 2) = 3E[X]2. To compute the lower bound, we have: lim n,d→∞

E[X]2 E[X2]

E[X]2 3E[X]2

1 3

Pr[I is SAT] ≥ lim

= lim

=

n,d→∞

n,d→∞

Therefore, the lower bound on the probability that a random instance of Model RB is satisfiable is 31 when n and d tend to infinity. When dealing with

complex mathematical problems, the capacity of LLMs to understand extensive context might be inadequate. Therefore, it is crucial to decompose the problem and filter out the relevant information.

50 24, 25, 37, 43, 49

Prompt: Now let’s calculate the probability that a random instance of Model RB has excatly one solution.

Let Ns denote the number of solutions of a random instance of Model RB. We want to find the probability that Ns = 1. We know that the expectation of the number of solutions is 12,

and the lower and upper bounds on the probability that a random instance is satisfiable are

- 1

- 3 and 12, respectively. Let P(Ns = 1) be the probability that a random instance has exactly one solution, P(Ns =

0) be the probability that a random instance has no solution, and P(Ns ≥ 2) be the probability that a random instance has at least two solutions.

We know that P(Ns ≥ 0) = 1, since a random instance can either have no solution, exactly one solution, or at least two solutions. We also know that the probability of a random instance being satisfiable is the sum of the probabilities of having exactly one solution and having at least two solutions, which can be expressed as P(Ns = 1) + P(Ns ≥ 2) ≥ 13 (the lower bound on satisfiability).

Now, let’s use the expectation of the number of solutions to find the probability of having exactly one solution. The expectation can be written as:

∞

### ∑

E[Ns] = 0 · P(Ns = 0) + 1 · P(Ns = 1) +

k · P(Ns = k)

k=2

Since we know that E[Ns] = 12, and that P(Ns ≥ 2) = 1 − P(Ns = 0) − P(Ns = 1), we can rewrite the expectation as:

∞

- 1

- 2

### ∑

= P(Ns = 1) +

k · P(Ns = k)

k=2

We can’t directly solve for P(Ns = 1) from this equation, but we can get an estimate of its range using the bounds on the probability of satisfiability.

We know that P(Ns = 1) + P(Ns ≥ 2) ≥ 31, so we have:

1 3 − P(Ns ≥ 2)

P(Ns = 1) ≥

Since P(Ns ≥ 2) ≤ 12 − P(Ns = 1) (the upper bound on satisfiability minus the probability of having exactly one solution), we have:

1 3 −

P(Ns = 1) ≥

Solving this inequality, we get:

- 1

- 2 − P(Ns = 1)

1 6

P(Ns = 1) ≥

So, we have a lower bound on the probability that a random instance of Model RB has

exactly one solution, which is 61. However, without more information or a more specific distribution on the number of solutions, we cannot determine an exact probability or an

upper bound. 51 24, 25, 43, 50

Prompt: Does this prove the existence of an infinite set of unsatisfiable Model RB instances and satisfiable Model RB instances with exactly one solution?

Yes, this does prove the existence of an infinite set of unsatisfiable Model RB instances and satisfiable Model RB instances with exactly one solution. Here’s why:

- 1. First, let’s consider unsatisfiable instances. Since the upper bound on the probability of a

random instance having exactly one solution is 21, the probability that a random instance has no solution is at least 12 (as it may have more than one solution as well). Since there are infinitely many random instances, there must be infinitely many unsatisfiable instances.

- 2. Now let’s consider instances with exactly one solution. We have a lower bound on the

probability that a random instance of Model RB has exactly one solution, which is 16. This means that for a large enough number of instances, we can expect at least 61 of them to have exactly one solution. Since there are infinitely many random instances, there must be infinitely many instances with exactly one solution.

Therefore, there exists an infinite set of both unsatisfiable Model RB instances and satisfiable Model RB instances with exactly one solution.

52 24, 25, 30

Prompt: Now we focus on Model RB instances with exactly one solution or with only one unsatisfied constraint, and we’re designing a new mapping operation to change the satisfiability of such instances. I notice that the current mapping operation with bijections seems somewhat redundant because for most tuples, no operation is actually required. In reality, we only select two groups of variables and swap their values. Can our new mapping operation be defined only on the tuples that need to be modified?

Yes, the new mapping operation can focus only on the tuples that need to be modified. To achieve this, we can perform the following steps:

- 1. Identify the constraint Cu = (Xu, Ru) that is either unsatisfied (if the instance is unsatisfiable) or can be unsatisfied by slightly modifying the solution (if the instance has exactly one solution).
- 2. For the constraint Cu, find two tuples (a1, a2) and (b1, b2) ∈ Ru such that they do not appear together in the solution or other constraints, and they satisfy the following conditions:

- • (a1, b2) ∈/ Ru and (b1, a2) ∈/ Ru.
- • Swapping the values of a2 and b2 in the solution will not violate other constraints in the instance.

- 3. Modify the permitted set Ru by removing (a1, a2) and (b1, b2), and adding (a1, b2) and (b1, a2). Let the new permitted set be R′u.
- 4. Replace the original permitted set Ru with the modified permitted set R′u for the constraint Cu = (Xu, Ru). This results in a new constraint Cu′ = (Xu, R′u).
- 5. The new instance of Model RB will have the same set of variables, domains, and constraints, but with the modified permitted set R′u for the constraint Cu′ .

This mapping operation changes the satisfiability of the instance by focusing only on the tuples that need to be modified. If the instance had exactly one solution, this operation creates an unsatisfied constraint, making the instance unsatisfiable. If the instance was unsatisfiable, this operation removes the unsatisfied constraint, creating a new satisfiable instance.

It is beneficial to enable LLMs to backtrack, adjust previous definitions or conditions, and correct earlier errors. This approach resembles human thought processes and can yield more refined outcomes.

53 24, 25, 52

Prompt: Great! But I have some minor suggestions: 1. If the instance has a unique solution, any constraint can be unsatisfied by slightly modifying the solution. Therefore, the phrase ‘can be unsatisfied by slightly modifying the solution (if the instance has exactly one solution)’ in the current definition is redundant. 2. Since our mapping operation only considers specific constraints, there is no need to consider other constraints. The statements regarding the appearance of tuples together and the preservation of other constraints are unnecessary. 3. Steps 3, 4, and 5 can be integrated to be more concise. Please update the definition of the mapping operation to make it more concise.

Thank you for the suggestions. Based on your feedback, I’ve updated the definition of the mapping operation as follows:

- 1. Identify the constraint Cu = (Xu, Ru) that needs to be modified. If the instance is unsatisfiable, choose the unsatisfied constraint. If the instance has exactly one solution, choose any constraint.

- 2. For the constraint Cu, find two tuples (a1, a2) and (b1, b2) ∈ Ru such that (a1, b2) ∈/ Ru and (b1, a2) ∈/ Ru.

- 3. Modify the constraint Cu = (Xu, Ru) by replacing the permitted set Ru with a new permitted set R′u obtained by removing (a1, a2) and (b1, b2), and adding (a1, b2) and (b1, a2). The new instance of Model RB will have the same set of variables, domains, and constraints, but

with the modified permitted set R′u for the constraint Cu′ .

This concise mapping operation changes the satisfiability of the instance by focusing only on the tuples that need to be modified. If the instance had exactly one solution, this operation creates an unsatisfied constraint, making the instance unsatisfiable. If the instance was unsatisfiable, this operation removes the unsatisfied constraint, creating a new satisfiable instance.

54 24, 25, 53

Prompt: We have proved that there exists an infinite set of satisfiable instances of Model RB with exactly one solution, how to prove that any instance in this set can be transformed to unsatisfiable instance under the mapping operation? (n tends to infinity, d tends to infinity.)

To prove that any instance in the infinite set of satisfiable instances of Model RB with exactly one solution can be transformed into an unsatisfiable instance under the mapping operation, we need to show that the mapping operation can eliminate the unique solution of the instance without introducing new solutions.

Let I be an instance from the infinite set of satisfiable instances with exactly one solution, and let σ be the unique solution of I. Let Cu = (Xu, Ru) be the constraint chosen in the mapping operation (either unsatisfied or any constraint if the instance has exactly one solution).

Now, consider the tuples (a1, a2) and (b1, b2) ∈ Ru used in the mapping operation, with (a1, b2) ∈/ Ru and (b1, a2) ∈/ Ru. Since σ is the unique solution for the instance I, we must have either σ(xu1) = a1 and σ(xu2) = a2, or σ(xu1) = b1 and σ(xu2) = b2.

After applying the mapping operation, the constraint Cu is replaced by Cu′ = (Xu, R′u), where the permitted set R′u is obtained by removing (a1, a2) and (b1, b2) from Ru and adding (a1, b2) and (b1, a2). As a result, both (a1, a2) and (b1, b2) are not in R′u.

Since the unique solution σ must satisfy either (a1, a2) ∈ Ru or (b1, b2) ∈ Ru, it cannot satisfy the modified constraint Cu′ . Therefore, σ is not a solution for the modified instance I′. Moreover, no other assignments satisfy all constraints, as we only changed Cu without introducing new solutions.

Hence, the instance I′ obtained after the mapping operation is unsatisfiable. This proves that any instance in the infinite set of satisfiable instances of Model RB with exactly one solution can be transformed into an unsatisfiable instance under the mapping operation, as the number of variables (n) and the domain size (d) tend to infinity.

55 24, 25, 53, 54

Prompt: But is it possible for tuples like (a1, ∗) or (b1, ∗) (* denotes other possible domain values for the second variable) in the permitted set to lead to new solutions? Please rigorously calculate the expected number of such tuples and their probability of being expanded to new solutions.

LLMs may struggle with self-evaluating the accuracy of their proofs, often concentrating on formalities like symbol usage. By iteratively asking questions such as ‘why’, ‘how’, or ‘is it...’, the model can be prompted to check for inaccuracies and make corrections. This approach serves as an effective method for enhancing LLM self-evaluation.

To calculate the expected number of tuples (a1, ∗) and (b1, ∗) in the permitted set, we need to analyze the probability of having such tuples in the permitted set and the possibility of them expanding to new solutions.

Let I be an instance from the infinite set of satisfiable instances of Model RB with exactly one solution, and Cu = (Xu, Ru) be the constraint chosen in the mapping operation. Recall that |Ru| = (1 − p)dk. Since the constraint is symmetric, the probability of a specific tuple

- (a1, a2) being in Ru is (1 − p). Now, let’s consider the tuples of the form (a1, ∗) and (b1, ∗), where ∗ can be any value

from the domain D∗ = {1,2, . . . , d}. There are d possible tuples for each form, giving a total of 2d tuples. Since the tuples are picked independently and the probability of a specific tuple being in Ru is (1 − p), the expected number of tuples of the form (a1, ∗) and (b1, ∗) in Ru is 2d(1 − p).

Now, let’s analyze the probability of these tuples expanding to new solutions in the modified instance I′. When we apply the mapping operation, we replace (a1, a2) and (b1, b2) with

- (a1, b2) and (b1, a2) in the permitted set R′u, while all other tuples remain unchanged. Since the instance I has exactly one solution, the tuples of the form (a1, ∗) and (b1, ∗) in

Ru are not part of any other solution. After the mapping operation, the tuples (a1, b2) and (b1, a2) are added to the permitted set, but we know that (a1, b2) ∈/ Ru and (b1, a2) ∈/ Ru. Therefore, these tuples were not part of the unique solution in the original instance I.

Thus, the probability of these tuples expanding to new solutions is 0, as they were not part of the unique solution in the original instance and the mapping operation only modified the specific tuples (a1, a2) and (b1, b2) without introducing new solutions.

In conclusion, the expected number of tuples of the form (a1, ∗) and (b1, ∗) in the permitted set is 2d(1 − p), but their probability of expanding to new solutions after the mapping operation is 0. This supports our previous claim that the mapping operation can transform an instance from the infinite set of satisfiable instances of Model RB with exactly one solution into an unsatisfiable instance. When tasked with

proving theorems, LLMs often provide explanations that lack rigorous proof, glossing over crucial details. Emphasizing words like ‘rigorously’, ‘mathematically’, or ‘probabilistically’ in the prompt can guide LLMs towards more rigorous proofs.

56 24, 25, 53, 54, 55

Prompt: Please calculate the probability of such tuples being expanded to new assignments that satisfy all the constraints more rigorously.

To calculate the probability of tuples of the form (a1, ∗) and (b1, ∗) expanding to new assignments that satisfy all the constraints, we need to consider the impact of the mapping operation on these tuples and the other constraints in the problem instance.

Let I be an instance from the infinite set of satisfiable instances of Model RB with exactly one solution, and Cu = (Xu, Ru) be the constraint chosen in the mapping operation. Let I′ be the modified instance obtained after the mapping operation.

Recall that we have 2d(1− p) expected tuples of the form (a1, ∗) and (b1, ∗) in Ru. After the mapping operation, we have a new permitted set R′u, where we removed (a1, a2) and (b1, b2), and added (a1, b2) and (b1, a2). Since we only modified two tuples in Ru, the expected number of tuples of the form (a1, ∗) and (b1, ∗) in R′u remains 2d(1 − p).

Now, consider a new assignment σ′ that satisfies the modified constraint Cu′ . For σ′ to be a solution for the instance I′, it must satisfy all other constraints Ci ∈ C − {Cu}. Since the instance I has exactly one solution, the other constraints are designed such that they limit the possible assignments to the unique solution σ.

The probability of σ′ satisfying all the other constraints is very low, as these constraints are independent and were not affected by the mapping operation. The probability of a single constraint Ci being satisfied by σ′ is given by the ratio of allowed tuples to all possible tuples, which is (1 − p).

For the new assignment σ′ to satisfy all m − 1 remaining constraints, the probability is (1 − p)m−1. As the number of variables (n) and domain size (d) tend to infinity, the number of constraints m = rn ln d also increases. Thus, (1 − p)m−1 tends to 0, making it highly unlikely for such tuples to expand to new assignments that satisfy all the constraints.

In conclusion, the probability of tuples of the form (a1, ∗) and (b1, ∗) expanding to new assignments that satisfy all the constraints in the modified instance I′ is very low, approaching

- 0 as n and d tend to infinity.

57 24, 25, 53, 54, 55, 56

Prompt: It seems that you’ve forgotten the multiple possible domain values for each variable in a new assignment.

You are correct. I apologize for the oversight. To address the multiple possible domain values for each variable in a new assignment, let’s reexamine the probability calculation.

Let I be an instance from the infinite set of satisfiable instances of Model RB with exactly one solution, and Cu = (Xu, Ru) be the constraint chosen in the mapping operation. Let I′ be the modified instance obtained after the mapping operation.

Consider a new assignment σ′ that satisfies the modified constraint Cu′ . As before, for σ′ to be a solution for the instance I′, it must satisfy all other constraints Ci ∈ C − {Cu}. A new assignment σ′ can have dn−1 possible combinations of domain values for the remaining n − 1 variables.

The probability of a single constraint Ci being satisfied by a specific combination of domain values is (1 − p), as previously discussed. To calculate the probability of σ′ satisfying all the other constraints for any combination of domain values, we need to consider all possible combinations and the independence of each constraint.

The probability of σ′ satisfying all m − 1 remaining constraints for a specific combination of domain values is (1 − p)m−1. To account for all dn−1 possible combinations of domain

- values for the remaining n − 1 variables, we need to consider the union of probabilities for all these combinations.

Using the union bound, the probability of σ′ satisfying all m − 1 remaining constraints for

- any combination of domain values is at most dn−1(1 − p)m−1. As the number of variables (n) and domain size (d) tend to infinity, the number of constraints m = rn ln d also increases.

- Thus, as n and d grow, (1 − p)m−1 tends to 0, but dn−1 tends to infinity.

In this case, the probability of tuples of the form (a1, ∗) and (b1, ∗) expanding to new assignments that satisfy all the constraints in the modified instance I′ cannot be determined to approach 0 or any fixed value without more information on the relationship between n, d, p, and the instance’s structure.

To rigorously prove that the mapping operation eliminates the unique solution of the instance without introducing new solutions, we would need additional constraints on the structure of the instances or the relationships between the parameters.

58 24, 25, 53, 54, 55, 56, 57

Prompt: Why do you say ‘the remaining n − 1 variables’? It seems that the value of two variables has been assigned.

My apologies, I misspoke in my explanation. You are correct that the values of two vari-

ables, xu1 and xu2, have been assigned in the modified constraint Cu′ . The remaining n − 2 variables still have multiple possible domain values in a new assignment.

To account for this correction, let’s reevaluate the probability calculation. A new assignment σ′ can have dn−2 possible combinations of domain values for the remaining n − 2 variables.

The probability of σ′ satisfying all m − 1 remaining constraints for a specific combination of domain values is still (1 − p)m−1. To account for all dn−2 possible combinations of domain

- values for the remaining n − 2 variables, we need to consider the union of probabilities for all these combinations.

Using the union bound, the probability of σ′ satisfying all m − 1 remaining constraints for

- any combination of domain values is at most dn−2(1 − p)m−1. As the number of variables (n) and domain size (d) tend to infinity, the number of constraints m = rn ln d also increases.

- Thus, as n and d grow, (1 − p)m−1 tends to 0, but dn−2 tends to infinity.

As before, the probability of tuples of the form (a1, ∗) and (b1, ∗) expanding to new assignments that satisfy all the constraints in the modified instance I′ cannot be determined to approach 0 or any fixed value without more information on the relationship between n, d, p, and the instance’s structure.

To rigorously prove that the mapping operation eliminates the unique solution of the instance without introducing new solutions, we would still need additional constraints on the structure of the instances or the relationships between the parameters.

59 24, 25, 53, 54, 55, 56, 57, 58

Prompt: Great! So considering all the possible tuples, what is the probability of such new solutions?

To calculate the probability of new solutions arising from the tuples of the form (a1, ∗) and (b1, ∗) after the mapping operation, we need to evaluate the joint probability of these tuples leading to new assignments that satisfy all the constraints in the modified instance I′.

As discussed earlier, the probability of a new assignment σ′ satisfying all m − 1 remaining constraints for any combination of domain values for the remaining n − 2 variables is at most dn−2(1 − p)m−1.

Now, let’s consider the tuples of the form (a1, ∗) and (b1, ∗). There are 2d possible tuples, and after the mapping operation, we have 2d(1 − p) expected such tuples in R′u.

To calculate the probability of new solutions arising from these tuples, we need to account for the fact that different tuples may lead to overlapping sets of new assignments. Since the constraints are chosen independently, we can use the union bound to estimate the probability of new solutions arising from any of the expected 2d(1 − p) tuples.

Using the union bound, the probability of new solutions arising from the tuples of the form (a1, ∗) and (b1, ∗) is at most 2d(1 − p)dn−2(1 − p)m−1.

As the number of variables (n) and domain size (d) tend to infinity, the number of con-

straints m = rn ln d also increases. Thus, as n and d grow, (1 − p)m−1 tends to 0, but 2d(1 − p)dn−2 tends to infinity.

The probability of new solutions arising from the tuples of the form (a1, ∗) and (b1, ∗) cannot be determined to approach 0 or any fixed value without more information on the relationship between n, d, p, and the instance’s structure.

To rigorously prove that the mapping operation eliminates the unique solution of the instance without introducing new solutions, we would still need additional constraints on the structure of the instances or the relationships between the parameters.

60

Prompt: Previous, we’ve set dn(1 − p)m = 12. Based on this, what is the value of 2d(1 − p)dn−2(1 − p)m−1, when d and n tend to infinity?

Let’s first simplify the expression 2d(1 − p)dn−2(1 − p)m−1 by combining the terms with d and (1 − p):

(1 − p)−1 d2

2d(1 − p)dn−2(1 − p)m−1 = 2dn(1 − p)m

We are given that dn(1 − p)m = 21. Multiplying both sides of this equation by 2, we get: 2dn(1 − p)m = 1

Now, we can substitute this into our simplified expression:

(1 − p)−1 d2 When d and n tend to infinity, the term (1−p)

(1 − p)−1 d2

2dn(1 − p)m

= 1 ·

−1

−1

d2 tends to (1−p)

∞ = 0. Therefore, the value of 2d(1 − p)dn−2(1 − p)m−1 when d and n tend to infinity is:

1 · 0 = 0

61 24, 25, 53, 54, 55

Prompt: Great! We have also proved that there exists an infinite set of unsatisfiable instances of Model RB, how to prove that any instance in this set can be transformed to satisfiable instance under the mapping operation? (n tends to infinity, d tends to infinity.)

To prove that any instance in the infinite set of unsatisfiable instances of Model RB can be transformed into a satisfiable instance under the mapping operation, we need to show that the mapping operation can introduce a solution to the unsatisfiable instance.

Let I be an instance from the infinite set of unsatisfiable instances of Model RB, and let Cu = (Xu, Ru) be the unsatisfied constraint chosen in the mapping operation.

Now, consider the tuples (a1, a2) and (b1, b2) ∈ Ru used in the mapping operation, with

- (a1, b2) ∈/ Ru and (b1, a2) ∈/ Ru. After applying the mapping operation, the constraint Cu is replaced by Cu′ = (Xu, R′u), where the permitted set R′u is obtained by removing (a1, a2) and
- (b1, b2) from Ru and adding (a1, b2) and (b1, a2). To show that the instance I′ obtained after the mapping operation is satisfiable, we will

construct a solution σ′ that satisfies Cu′ and all the other constraints in the instance. First, note that since I is unsatisfiable, there exists an assignment σ that satisfies all con-

straints except Cu. Now, we will modify σ to create a new assignment σ′ that satisfies Cu′ as well.

We have proved that the mapping operation does not introduce new solutions. With this, we have completed the proof of the second type mentioned in 35.

In 43, we set the upper bound for the probability of a random instance being satisfiable at

- 1

- 2. Consequently, the probability of an instance being unsatisfiable retains a positive value, which inherently implies the existence of an infinite number of unsatisfactory instances.

If σ(xu1) = a1 and σ(xu2) = a2, set σ′(xu1) = a1 and σ′(xu2) = b2. If σ(xu1) = b1 and σ(xu2) = b2, set σ′(xu1) = b1 and σ′(xu2) = a2. In both cases, the new assignment σ′ satisfies the modified constraint Cu′ .

Since the other constraints have not been changed during the mapping operation, σ′ also satisfies all the other constraints, as it only differs from σ in the values assigned to xu1 and xu2. Therefore, σ′ is a solution for the modified instance I′, making it a satisfiable instance.

Hence, we have proved that any instance in the infinite set of unsatisfiable instances of Model RB can be transformed into a satisfiable instance under the mapping operation, as the number of variables (n) and the domain size (d) tend to infinity.

62 24, 25, 53, 54, 55, 61

Prompt: Great! In the above transformation process, it is necessary to ensure the existence of only unsatisfied constraint Cu for the current assignment (the assignment satisfies all the other constraints except Cu). However, how to prove that during the solving process, for any assignment, we can always modify the satisfiability using the above method (i.e., proving the existence of only unsatisfied constraint for any variables)?

In the above argument, we showed that we can transform an unsatisfiable instance into a satisfiable one and vice versa using the mapping operation. However, we did not prove that there always exists an assignment that satisfies all constraints except Cu. In general, it may not be the case that we can find an only unsatisfied constraint for any assignment. The examples we considered were simplified to illustrate the mapping operation’s effect on the satisfiability of instances.

In practice, to modify the satisfiability of an instance during the solving process, we can apply local search techniques, constraint propagation, or other methods for solving constraint satisfaction problems. These methods can help find partial assignments that satisfy a large number of constraints and identify critical constraints or variables that determine the instance’s satisfiability.

For example, we can use a local search algorithm to find an assignment that maximizes the number of satisfied constraints. If the maximum number of satisfied constraints is equal to the total number of constraints minus one (m − 1), then we can identify the only unsatisfied constraint and apply the mapping operation. If the maximum number of satisfied constraints is less than m−1, it indicates that the problem instance might have more complex interdependencies among constraints and variables. In this case, we may need to apply more advanced techniques or heuristics to modify the satisfiability of the instance.

To summarize, the existence of an only unsatisfied constraint for any assignment is not guaranteed. However, the mapping operation can still be useful to modify the satisfiability of instances during the solving process, especially when combined with other techniques for solving constraint satisfaction problems. If the LLM struggles

with generating an answer, consider simplifying the request by removing some past turns in the dialogue history.

63 24, 25

Prompt: If there exists an assignment that satisfies all the other constraints of an instance except Cu, then we call Cu an self-unsatisfiable constraint. For an unsatisfiable instance, how to prove that considering all the possible assignments, every variable is contained in at least one self-unsatisfiable constraint? (n tends to infinity, d tends to infinity.) Do you have any possible ideas? Please lay out a high-level proof plan.

We adopt 63* to denote the inclusion of the selfunsatisfiable constraint definition within the prompt.

To prove that in an unsatisfiable instance of Model RB, every variable is contained in at least one self-unsatisfiable constraint, we can leverage the probabilistic method. We will show

that as n and d tend to infinity, the probability of a variable not being contained in any selfunsatisfiable constraint tends to zero. This implies that, with high probability, every variable is contained in at least one self-unsatisfiable constraint. Here’s the high-level proof plan:

- 1. Define the events: Let Ei be the event that variable xi is not contained in any self-unsatisfiable constraint.
- 2. Analyze the probability of a single constraint being self-unsatisfiable: Calculate the probability that a randomly chosen constraint Ci is self-unsatisfiable. This can be done by considering the probability that all other constraints are satisfied while Ci is not. We can express this probability in terms of p, k, and d.

- 3. Calculate the probability of a variable not being in a self-unsatisfiable constraint: For each variable xi, calculate the probability that it is not contained in any self-unsatisfiable constraint. This can be done by considering the complementary event, i.e., that xi is contained in at least one self-unsatisfiable constraint, and then take the complement of this probability. Use the union bound to obtain an upper bound on the probability of the event Ei.

- 4. Apply the union bound and show that the probability tends to zero: Now, consider the

probability of the event i n=1 Ei, i.e., the probability that at least one variable is not contained in any self-unsatisfiable constraint. Apply the union bound to obtain an upper bound on this probability. Show that as n and d tend to infinity, this probability tends to zero.

- 5. Conclude that every variable is contained in a self-unsatisfiable constraint with high prob-

ability: Since the probability of the event i n=1 Ei tends to zero as n and d tend to infinity, we can conclude that with high probability, every variable is contained in at least one self-unsatisfiable constraint in an unsatisfiable instance of Model RB.

This high-level proof plan leverages the probabilistic method and the union bound to show that in an unsatisfiable instance of Model RB, every variable is contained in at least one self-unsatisfiable constraint with high probability as n and d tend to infinity.

64 24, 25, 63*

Prompt: For an unsatisfiable instance, considering all the possible assignments, let’s calculate the probability of a constraint being self-unsatisfiable. (n tends to infinity, d tends to infinity.) Can you suggest some possible methods?

To calculate the probability of a constraint being self-unsatisfiable, we can use the following methods:

- 1. Monte Carlo method: This method involves generating a large number of random instances and assignments, and then calculating the fraction of instances where a constraint is self-unsatisfiable. As the number of instances and assignments tends to infinity, the Monte Carlo estimate of the probability converges to the true probability.
- 2. Analytical approach: This method involves calculating the probability of a constraint being self-unsatisfiable directly, using combinatorics and probability theory. This approach can be quite challenging, especially for complex models like Model RB. However, if successful, it can provide closed-form expressions for the probability, which can be useful for further theoretical analysis.

If the model’s response reflects a misunderstanding, it could be a result of ambiguity within the prompt. In such instances, try to review and revise those specific parts in the prompt for clarity. For instance, we previously used the term ’the only unsatisfiable constraint’ to denote such restrictions, but GPT-4 often misinterprets this phrase. Thus, we introduce the new term, ‘self-unsatisfiable constraint’.

- 3. Generating function method: This method involves constructing a generating function that captures the combinatorial structure of the problem. By analyzing the properties of this function (e.g., its asymptotic behavior), we can derive information about the probability of a constraint being self-unsatisfiable.
- 4. First moment method: This method involves calculating the expected number of selfunsatisfiable constraints for a random instance. If the expected number tends to zero as n and d tend to infinity, then the probability of a constraint being self-unsatisfiable also tends to zero. This method can be useful for establishing a threshold behavior for the probability of unsatisfiability.
- 5. Second moment method: This method involves calculating the variance of the number of self-unsatisfiable constraints for a random instance. If the variance is much smaller than the square of the expected number, then the probability of a constraint being selfunsatisfiable can be approximated by the ratio of the expected number to the total number of constraints. This method can be useful for proving concentration results for the probability of unsatisfiability.

These methods can be used separately or combined to provide a comprehensive understanding of the probability of a constraint being self-unsatisfiable in Model RB as n and d tend to infinity.

65 24, 25, 63*

Prompt: For an unsatisfiable instance, considering all the possible assignments, let N be the number of assignments that satisfies all constraints except the given one. Can you calculate the expectation of N?

Let E(N) be the expected value of N, the number of assignments that satisfy all constraints except the given one in an unsatisfiable instance. To calculate E(N), we can consider the probability of each assignment satisfying all constraints except the given one and sum it over all possible assignments.

Let σ be an assignment and Cu be the given constraint. We want to compute the probability that σ satisfies all constraints except Cu. The probability of σ satisfying a single constraint Ci is given by the ratio of the size of the permitted set Ri to the total number of possible tuples for the variables in Xi, which is dk. Since |Ri| = (1 − p)dk, the probability of σ satisfying Ci is (1 − p).

Now, since there are m − 1 constraints other than Cu, and each of them must be satisfied, the probability of σ satisfying all of them is (1 − p)m−1. Note that we assume that the constraints are independent.

There are a total of dn possible assignments for the instance. To compute the expected value of N, we sum the probabilities of all assignments satisfying all constraints except Cu:

E(N) = dn(1 − p)m−1.

However, it’s important to keep in mind that this is an approximation, as it assumes that the constraints are independent. In practice, since some variables are shared between constraints, the true expected value of N might be different.

66 24, 25, 63*, 65

Prompt: You seem to be missing the probability that the assignment unsatisfies the only unsatisfied constraint.

You are correct, I missed that part in the calculation. Let’s add that probability to our calculation of the expected value of N.

Let q be the probability that an assignment σ does not satisfy the given constraint Cu. Since |Ru| = (1 − p)dk, the probability of not satisfying Cu is q = 1 − (1 − p) = p.

Now, we want the probability of an assignment satisfying all other constraints except Cu, and not satisfying Cu. The probability of σ satisfying all other constraints except Cu is still (1 − p)m−1, and the probability of not satisfying Cu is p.

Since we are assuming the constraints are independent, we can multiply these probabilities together to get the probability of an assignment satisfying all constraints except Cu and not satisfying Cu:

###### (1 − p)m−1 · p

Now, to compute the expected value of N, we sum this probability over all dn possible assignments:

E(N) = dn(1 − p)m−1 · p

This is the expected value of N under the assumption that the constraints are independent. In practice, due to shared variables between constraints, the true expected value of N might be different.

67 24, 25

Prompt: We have: E[N] = dn(1 − p)m−1 · p and E[X] = nαn(1 − p)m. Can you calculate and tell me the relationship between E[N]2 and E[X]2?

We are given:

E[N] = dn(1 − p)m−1 · p

E[X] = nαn(1 − p)m

To find the relationship between E[N]2 and E[X]2, we need to find the expressions for both and compare them.

First, let’s find E[N]2:

E[N]2 = (dn(1 − p)m−1 · p)2 = d2n(1 − p)2m−2 · p2 Now, let’s find E[X]2:

E[X]2 = (nαn(1 − p)m)2 = n2αn(1 − p)2m Finally, let’s compare the two expressions:

E[N]2 E[X]2

d2n(1 − p)2m−2 · p2 n2αn(1 − p)2m

=

Simplifying the expression by cancelling out (1 − p)2m−2:

d2n · p2 n2αn(1 − p)2

E[N]2 E[X]2

=

Now, recall that d = nα. Substituting this into the expression:

Drawing on our prior experience with the second moment method

(47) and the known values of E X2 and E [X]2, we analyze the connection between E N2 and E X2 .

(nα)2n · p2 n2αn(1 − p)2 Simplifying:

E[N]2 E[X]2

=

p2 (1 − p)2 So, the relationship between E[N]2 and E[X]2 is:

E[N]2 E[X]2

=

p2

E[N]2 =

###### (1 − p)2 · E[X]2

68 24, 25, 47*

Can you calculate and tell me the relationship between E[N2] and E[X2]?

To find the relationship between E[N2] and E[X2], let’s first observe the differences in their formulas.

The main difference is in the power of the terms inside the parentheses. In E[N2], the power is (rn ln d − 1), while in E[X2], the power is (rn ln d).

Now, let’s consider the following transformation:

E[N2] E[X2]

T[X2] =

.

We can rewrite this expression as follows:

S k)

S k)

S k)

S k)

∑nS=0 dn(nS)(d − 1)n−S((1 − p)(

(nk) + (1 − p)2(1 − (

(nk)))rnlnd−1 · (p(

(nk) + p2(1 − (

###### (nk))) ∑nS=0 dn(nS)(d − 1)n−S((1 − p)(

T[X2] =

S k)

S k)

(nk) + (1 − p)2(1 − (

(nk)))rnlnd

S k)

S k)

S k)

S k)

((1 − p)(

(nk) + (1 − p)2(1 − (

(nk)))rnlnd−1 · (p(

(nk) + p2(1 − (

###### (nk))) ((1 − p)(

n

### ∑

=

S k)

S k)

(nk) + (1 − p)2(1 − (

(nk)))rnlnd

S=0

S k)

S k)

p(

(nk) + p2(1 − (

###### (nk)) ((1 − p)(

n

### ∑

.

=

S k)

S k)

(nk) + (1 − p)2(1 − (

S=0

(nk)))

Thus, T[X2] represents the relationship between E[N2] and E[X2].

69 24, 25, 63*, 67, 68

Prompt: Now let’s calculate the probability of N > 0 with the second moment method.

To calculate the probability of N > 0 using the second moment method, we first need to compute the second moment E N2 and then use Chebyshev’s inequality.

Since we have already derived a formula for the ratio E[N2]

E[X2] , we can obtain E N2 by multiplying the ratio by E X2 :

E N2 E [X2] · E X2 =

E N2 =

S k)

S k)

p(

(nk) + p2 1 − (

###### (nk) (1 − p)(

n

· E X2

### ∑

S k)

S k)

(nk) + (1 − p)2 1 − (

S=0

(nk)

47* denotes the equations

in the prompt of 47 are included in the current prompt as well.

Next, we apply Chebyshev’s inequality to get an upper bound on the probability of N > 0:

E N2

Pr(N > 0) ≥ 1 −

= 1 −

E[N]2

Simplifying the expression, we obtain:

S k)

S k)

k) +p2 1− (

p (

(n

(n

∑nS=0

###### · E X2

###### k) (1−p) (

S k)

S k)

k) +(1−p)2 1− (

(n

(n

k)

2

p

(1−p)2 · E[X]2

S k)

S k)

k) +p2 1− (

p (

(n

(n

∑nS=0

###### k) (1−p) (

S k)

S k)

k) +(1−p)2 1− (

(n

(n

k) p2 (1−p)2

Pr(N > 0) ≥ 1 −

By calculating this probability, we can estimate the likelihood of an unsatisfiable instance

having at least one self-unsatisfiable constraint. When the computational load of LLMs becomes excessively large, we break down the intermediate steps for preliminary calculations. The precomputation strategy aims to reduce the single-round computational burden of LLMs.

70 24, 25

S k)

Prompt: Can you estimate the value of (

(nk) when S = ns(0 ≤ s ≤ 1) and n tends to infinity.

S k)

To estimate the value of (

(nk) when S = ns (0 ≤ s ≤ 1) and n tends to infinity, we use the formula for binomial coefficients:

n! k!(n − k)! So, we get:

n k

=

We can simplify the expression to:

S! k!(S−k)! n! k!(n−k)!

(Sk) (nk)

=

(Sk) (nk)

S!(n − k)! n!(S − k)!

(ns)!(n − k)! n!(ns − k)!

=

=

Now, as n tends to infinity, we can approximate the factorials using Stirling’s approximation:

√2πn

n! ≈

Using Stirling’s approximation, we get:

n e

n

√2πns nse ns 2π(n − k) n−e k

n−k

(ns)!(n − k)!

n!(ns − k)! ≈

√2πn ne n 2π(ns − k) nse−k

ns−k

Now we can cancel out common terms and simplify:

n−k

ns n−k

ns n−k

ns n

(ns)!(n − k)!

n

n!(ns − k)! ≈

ns−k

ns ns−k

ns−k n

Since n tends to infinity, we can further simplify the expression:

n−k

ns n−k

ns n−k

ns n

(Sk) (nk) ≈ lim

n

lim

ns−k

n→∞

n→∞

ns ns−k

ns−k n

As n → ∞, the limit converges to:

(Sk) (nk) ≈

sk(1 − s)k (1 − s)k

= sk

S k)

So, when S = ns and n tends to infinity, the value of (

(nk) is approximately sk.

71 24, 25, 63*, 69, 70

Prompt: Now let’s calculate the probability of N>0 with the second moment method.

To calculate the probability of N > 0 with the second moment method, we first note that:

E[N]2 E[N2]

P(N > 0) ≥

We already know the expressions for E[N]2 and E[N2] in terms of E[X]2 and E[X2] from the given information. We can now substitute the expressions into the inequality:

p2

(1−p)2 · E[X]2

P(N > 0) ≥

S k)

S k)

k) +p2 1− (

p (

(n

(n

∑nS=0

###### k) (1−p) (

S k)

S k)

k) +(1−p)2 1− (

(n

(n

k)

S k)

We also know that EE[[XX2]2] ≥ 13 and the expression for (

(nk) when S = ns as sk. We can now simplify the expression:

p2

(1−p)2 · E[X]2 ∑1s=0 ps

P(N > 0) ≥

k+p2(1−sk)

(1−p)sk+(1−p)2(1−sk)

Now we can calculate the limit as n tends to infinity:

p2

(1−p)2 · 13 ∑1s=0 ps

lim

P(N > 0) ≥

k+p2(1−sk)

n→∞

(1−p)sk+(1−p)2(1−sk)

By calculating the limit, we can obtain an expression for the probability of N > 0, which is the probability that there exists an assignment that satisfies all the other constraints of an instance except the given one as n tends to infinity.

72 24, 25, 63*, 69, 70, 71

Prompt: I have identified a problem with the current formula. The formula contains a summation symbol, but the summation is not needed. Please correct the formula and simplify it, then continue to calculate the probability of N > 0, or at least its lower bound.

Apologies for the confusion. You are right, there is no need for a summation. We can rewrite the expression for the probability of N > 0 as follows:

p2

(1−p)2 · E[X]2 psk+p2(1−sk) (1−p)sk+(1−p)2(1−sk) Now, we can apply the known inequality EE[[XX2]2] ≥ 13:

P(N > 0) ≥

p2

(1−p)2 · 13 psk+p2(1−sk) (1−p)sk+(1−p)2(1−sk)

P(N > 0) ≥

Since we are interested in the lower bound of the probability P(N > 0), we can simplify the expression by taking s = 1, as the denominator will be minimized in this case:

This simplifies to:

p2

(1−p)2 · 13 p+p2(1−1) (1−p)+(1−p)2(1−1)

P(N > 0) ≥

p2

(1−p)2 · 13 p (1−p)

P(N > 0) ≥

Now, we can cancel out the p in the numerator and denominator, and further simplify:

p 3(1 − p)

P(N > 0) ≥

This expression provides the lower bound for the probability of N > 0, which is the probability that there exists an assignment that satisfies all the other constraints of an instance except the given one.

73 24, 25, 63*, 69, 72

Prompt: Great! Since 0 < p < 1, can we make the lower bound simpler?

Yes, we can further simplify the lower bound for the probability of N > 0. Since 0 < p < 1, we know that 1 − p > 0. Thus, we can rewrite the lower bound as follows:

(1 − p) (1 − p)

p

P(N > 0) ≥

3(1 − p) ·

This simplifies to:

p(1 − p) 3(1 − p)

P(N > 0) ≥

Now, we can cancel out the (1 − p) in the numerator and denominator:

p 3

P(N > 0) ≥

This expression provides a simpler lower bound for the probability of N > 0, which is the probability that there exists an assignment that satisfies all the other constraints of an instance except the given one. Step 2 of the plan in 63

was accomplished: calculating the probability that a single constraint is self-unsatisfiable.

74 24, 25, 63*, 73

Prompt: How to prove that considering all the possible assignments, every variable is contained in at least one self-unsatisfiable constraint? (n tends to infinity, d tends to infinity.) Can you lay out a plan for proof?

To prove that every variable is contained in at least one self-unsatisfiable constraint, we can proceed with the following plan for proof:

- 1. Define notations and assumptions: Define notations for the variables, domains, constraints, and assignments involved in a random instance of Model RB. State the assumptions that

n → ∞, d → ∞, and the probability of a randomly chosen constraint Ci being selfunsatisfiable is at least p/3.

- 2. Establish bounds on the number of constraints: Since there are m = rn ln d constraints, and each variable xi appears in k distinct constraints, we can establish upper and lower bounds on the number of constraints involving a specific variable xi.

- 3. Probabilistic analysis: Perform a probabilistic analysis to calculate the probability that a

randomly chosen constraint Ci containing a specific variable xi is self-unsatisfiable. Use the assumption that the probability of a randomly chosen constraint being self-unsatisfiable is at least p/3.

- 4. Union bound: Apply the union bound to calculate the probability that there exists a variable which is not part of any self-unsatisfiable constraint. This probability can be calculated as the complement of the probability that all variables are part of at least one self-unsatisfiable constraint.
- 5. Limit analysis: Examine the behavior of the calculated probability as n → ∞ and d → ∞. If the probability tends to zero, conclude that every variable is contained in at least one self-unsatisfiable constraint with high probability.
- 6. Conclusion: Summarize the results of the proof and outline any additional observations or implications for the Model RB problem.

75 24, 25, 63*, 73, 74

Prompt: Now let’s calculate the lower bounds on the number of constraints involving a specific variable xi.

To calculate the lower bounds on the number of constraints involving a specific variable xi, we can analyze the constraint generation process in Model RB.

In each constraint Ci = (Xi, Ri), a sequence of k distinct variables Xi =

xi1, xi2, . . . , xik is chosen uniformly at random without repetition from the set X. Since there are n variables in total, the probability of a specific variable xi being chosen in a single constraint is given by:

k n

P(xi ∈ Xi) =

There are m = rn ln d constraints in total. The number of constraints involving the variable xi follows a binomial distribution with parameters m and k/n. We can calculate the expected number of constraints involving xi as follows:

k n

E[constraints involving xi] = m · P(xi ∈ Xi) = (rn ln d) ·

= rk ln d

To calculate the lower bound on the number of constraints involving the variable xi, we can use the Chernoff bound for the binomial distribution. For any δ > 0, the Chernoff bound states that:

P(constraints involving xi ≤ (1 − δ)E[constraints involving xi]) ≤ e−12δ2E[constraints involving xi] By choosing an appropriate δ > 0, we can establish a lower bound on the number of

constraints involving the variable xi. As an example, let’s choose δ = 1/2. Applying the Chernoff bound, we get:

- 1

- 2

rk ln d) ≤ e−81rklnd

P(constraints involving xi ≤

Thus, with high probability, the number of constraints involving a specific variable xi is at least 21rk ln d.

76 24, 25, 63*, 73, 74, 75

Prompt: Great! What is that probability for all the possible variables? Please convert the probability into an expression with n as the base. and determine how to construct the variables to make the probability infinitesimal as n and d approach infinity.

We have found the probability that the number of constraints involving a specific variable

xi is at least 21rk ln d. Now, we want to find the probability that this holds true for all n variables.

Using the union bound, the probability that there exists a variable that has less than 12rk ln d constraints involving it is at most n times the probability for a single variable. Therefore, the probability that at least one variable violates the lower bound is:

- 1

- 2

rk ln d constraints) ≤ n · e−18rklnd

P(at least one variable has <

To express this probability with n as the base, we can take the logarithm base n of both sides:

logn P(at least one variable has <

- 1

- 2

rk ln d constraints) ≤ logn n + logn e−18rklnd

- 1

- 2

1 8

ln d ln n

⇒ logn P(at least one variable has <

rk ln d constraints) ≤ 1 −

rk

Now, we want this probability to go to zero as n → ∞ and d → ∞. To achieve this, we

need to ensure that lnlnnd grows unbounded as n, d → ∞. Recall that the domain size d = nα, where α > 0. Thus, we have:

ln nα ln n

ln d ln n

=

###### = α

We can see that as long as α > 0, the term lnlnnd is constant and does not depend on n or d. To make the probability infinitesimal as n, d → ∞, we can adjust the value of r to ensure that the inequality: 1 − 18rkα < 0 holds true. By choosing a sufficiently large value of r (depending

According to the constraint generation process of Model RB, the variables are independent, which ensures the applicability of the Chernoff bounds.

on the values of k and α), we can make the probability of having at least one variable with less than 12rk ln d constraints infinitesimal as n and d approach infinity. From this point forward,

we incorporate 1 − 1 8rkα < 0 into our construction conditions.

77 24, 25, 63*, 73, 74, 75

Prompt: Great! Now let’s apply the union bound to calculate the probability that there exists a variable which is not part of any self-unsatisfiable constraint.

To apply the union bound, we first calculate the probability that a specific variable xi is not part of any self-unsatisfiable constraint. Then, we’ll use the union bound to calculate the probability that there exists at least one such variable.

Let the event Ai denote the situation where variable xi is not part of any self-unsatisfiable constraint. We know that the lower bound on the number of constraints involving xi is at least 21rk ln d with high probability. Moreover, we have the assumption that the probability of a randomly chosen constraint Ci being self-unsatisfiable is at least 3p.

Thus, the probability that a constraint involving xi is not self-unsatisfiable is at most 1 − 3p. We can now calculate the probability of the event Ai:

- 1

- 2rk ln d

p 3

P(Ai) ≤ 1 −

Now, to apply the union bound, we want to find the probability that there exists at least one variable that is not part of any self-unsatisfiable constraint. There are n variables in total, so we can apply the union bound as follows:

n

n

- 1

- 2rk ln d

p 3

### ∑

P(Ai) ≤ n 1 −

P

Ai ≤

i=1

i=1

This expression represents the upper bound on the probability that there exists at least one variable which is not part of any self-unsatisfiable constraint. In the next step, we can analyze the behavior of this probability as n → ∞ and d → ∞.

78 24, 25, 63*, 73, 74, 75, 77

- 1

- 2rklnd into an expression with n as the base. Then

Prompt: Great! Please convert n 1 − 3p calculate how to construct the variables to ensure that the expression becomes infinitesimal, as both n and d tend towards infinity.

We can rewrite the expression n 1 − 3p steps:

- 1

- 2rklnd in terms of the base n by taking the following

- 1. First, let’s isolate the exponent 21rk ln d:

1 −

p 3

- 1

- 2rk ln d

= 1 −

p 3

ln d

- 1

- 2rk

- 2. Now, recall that d = nα. Taking the logarithm of both sides, we have ln d = α ln n. Substitute this into the expression:

1 −

p 3

- 1

- 2rk ln d

= 1 −

p 3

α ln n

- 1

- 2rk

- 3. Using the property alnb = blna, we can convert the expression with base n:

p 3

1 −

- 1

- 2rk ln d

= nln(1−3p)

- 1

- 2 αrk

Now the expression becomes:

n

Ai) ≤ n nln(1−3p)

P(

i=1

1 2 αrk

To ensure that this expression becomes infinitesimal as both n and d tend towards infinity, we need to find a condition on the variables such that the exponent of n is negative. Let’s denote the exponent of n as E:

E = 1 +

- 1

- 2

p 3

αrk ln(1 −

###### )

To make the expression infinitesimal, we want E < 0. Therefore,

1 +

1 2

αrk ln(1 −

p 3

###### ) < 0

We can rearrange the inequality to find a condition on the variables:

- 1

- 2

p 3

αrk ln(1 −

###### ) < −1

We also integrate

To satisfy this inequality, we can choose the values of α, r, k, and p such that the product αrk ln(1 − 3p) is less than −2. One possible way to achieve this is by choosing α,r, k > 0 and p such that ln(1 − 3p) is negative, which can be satisfied by choosing 0 < p < 1.

- 1

- 2αrk ln(1 − 3p) < −1 into our construction criteria.

- 1

- 2 αrk

By constructing the variables in this manner, we can ensure that the expression n nln(1−3p) becomes infinitesimal as both n and d tend towards infinity. This implies that, with high probability, every variable is contained in at least one self-unsatisfiable constraint. Up to this point, step

3 and 4 of the plan in 63 was accomplished: applying the union bound and showing that the probability tends to zero.

79 24, 25, 63*, 73, 74, 75, 77

Prompt: With proper parameter selection, now what conclusion can we draw? Answer in one sentence.

With proper parameter selection, we can conclude that every variable is contained in at least one self-unsatisfiable constraint with high probability as both n and d tend towards infinity.

80 63*

Prompt: We have proved that there exists an infinite set of unsatisfiable instances of Model RB. For each unsatisfiable instance, we’ve proved that considering all the possible assignments, every variable is contained in at least one self-unsatisfiable constraint. How to prove that any instance in this set can be transformed to a satisfiable instance under the mapping operation? (n tends to infinity, d tends to infinity.)

To prove that any unsatisfiable instance in the given infinite set can be transformed into a satisfiable instance under the mapping operation, we will follow these steps:

- 1. Choose an unsatisfiable instance I from the given infinite set.
- 2. Since every variable is contained in at least one self-unsatisfiable constraint, there must exist at least one constraint Cu = (Xu, Ru) that is self-unsatisfiable.
- 3. Perform the mapping operation on constraint Cu, creating a new instance I′ with the modified constraint Cu′ = (Xu, R′u).

- 4. Show that the new instance I′ is satisfiable.

Step 3 is straightforward, as we have already detailed the mapping operation for k = 2. Now we need to show that the new instance I′ is satisfiable.

Let σ be an assignment that satisfies all constraints in I except for Cu. Since (a1, a2) and (b1, b2) are not allowed in Ru, but (a1, b2) and (b1, a2) are allowed in R′u, we can modify σ to create a new assignment σ′ that satisfies the modified constraint Cu′ .

Specifically, update σ′ as follows:

- • For variable xu1, set σ′(xu1) = a1 if σ(xu1) = b1, otherwise set σ′(xu1) = σ(xu1).
- • For variable xu2, set σ′(xu2) = a2 if σ(xu2) = b2, otherwise set σ′(xu2) = σ(xu2).

This new assignment σ′ satisfies the modified constraint Cu′ , because (σ′(xu1), σ′(xu2)) is either (a1, b2) or (b1, a2), both of which are allowed in R′u.

Since σ′ satisfies all other constraints in I, it also satisfies all constraints in I′. Thus, the

new instance I′ is satisfiable. We have completed the proof of the first type mentioned in 35. Combined with the previous proof of the second type, we have demonstrated that it is possible to construct hard instances required within the schema.

In conclusion, we have shown that any unsatisfiable instance in the given infinite set can be transformed into a satisfiable instance under the mapping operation. As n and d tend to infinity, this demonstrates the existence of a transformation that can turn unsatisfiable instances of Model RB into satisfiable instances for large problem sizes.

81

Prompt: Now, let us verify whether there always exist instances that meet our requirements. Besides the requirements for the parameters in Model RB definition, the requirements we had set for the parameters previously are:

- 1. rcr = −ln(11−p), α > 1k, 0 < p < 1 and k ≥ 1−1p.

- 2. r = rcr + ϵ, ϵ = ln

- 1

- 2

αnlnnln(1−p).

- 3. 1 + α(1 − rcrpk) < 0 and α > 1.
- 4. 1 − 81rkα < 0.

- 5. 1 + 12αrk ln 1 − 3p < 0. Please analyze if there are always instances that satisfy all demands. Please study a most important parameter, and explore how this parameter can ensure that instances always meet the requirements.

To analyze if there are always instances that satisfy all the given demands, let’s first rewrite the conditions in a more readable format:

- 1. rcr = −ln(11−p), α > 1k, 0 < p < 1, and k ≥ 1−1p.

- 2. r = rcr + ϵ, ϵ = ln

- 1

- 2

αnlnnln(1−p).

- 3. 1 + α(1 − rcrpk) < 0 and α > 1.
- 4. 1 − 81rkα < 0.

- 5. 1 + 12αrk ln 1 − 3p < 0.

Now, let’s analyze the most important parameter that can ensure the instances always meet the requirements. The most important parameter is α, as it affects all the conditions directly or indirectly.

To satisfy condition 1, we need α > 1k. This ensures that the domain size d grows with n, making the problem more complex.

Condition 2 is satisfied for any ϵ > 0. Since α is in the denominator of ϵ, a larger α will make ϵ smaller, keeping it positive.

For condition 3, we notice two parts: 1 + α(1 − rcrpk) < 0 and α > 1. The second part implies that α should be greater than 1. The first part implies that, when α is large enough, the term inside the parentheses needs to be negative. This can be achieved if rcrpk is sufficiently large.

Condition 4 states that 1 − 18rkα < 0. This implies that 18rkα > 1, which means that rkα should be large enough. Since α is a part of the product, it should be large enough to satisfy this condition.

Finally, condition 5 states that 1 + 21αrk ln 1 − 3p < 0. This implies that αrk ln 1 − 3p should be negative and large enough in magnitude. This is possible if α is large enough, and the term inside the logarithm is less than 1.

In conclusion, by carefully choosing the parameter α and making it large enough, we can ensure that instances of Model RB always meet the requirements. The value of α has a significant impact on the complexity of the problem, and a larger value of α tends to make the problem harder to solve, ensuring that the requirements are met. Thus far, we have demon-

strated the first key point in the schema: there exists an infinite set of Model RB instances, which possess the property of being fixed points when their satisfiability is altered through a mapping operation. The probability of this property holding true tends to 1 as the number of variables approaches infinity. In fact, as long as the probability is greater than 0, it implies that there are definite instances within the sample space. Here, the existence of infinitely many instances with a probability of 1 is a stronger conclusion.

82

Prompt: Assuming that we’ve already constructed a set of difficult problems - a collection of infinitely many instances of Model RB (a special type of CSP generation model) - and have proved that the satisfiable instances and unsatisfiable instances can be transformed into each other through a mapping operation, while the instances before and after the transformation have the same parameter values. Now, we need to prove that these types of problems cannot be solved in polynomial time. Please design 10 creative ideas.

- 1. Reduction from a known NP-hard problem: Show that solving Model RB is at least as hard as solving another well-known NP-hard problem by designing a polynomial-time reduction from that problem to instances of Model RB. If the reduction holds, then solving Model RB in polynomial time would imply solving the original NP-hard problem in polynomial time, leading to a contradiction.
- 2. Time hierarchy theorem: Use the time hierarchy theorem to show that, even with the mapping operation, the time complexity of solving Model RB instances cannot be reduced to polynomial time. The theorem could be applied to prove the existence of problems that are harder than the polynomial time problems.

- 3. Circuit complexity: Demonstrate that any circuit solving instances of Model RB would have to be of super-polynomial size, implying that the problem cannot be solved in polynomial time. This approach relies on proving lower bounds on the circuit complexity required to solve Model RB instances.
- 4. Randomized algorithms: Investigate the potential of randomized algorithms for solving Model RB instances. Show that no efficient randomized algorithm exists that can solve these instances in polynomial time, even under the mapping operation, with high probability.
- 5. Non-deterministic Turing machines: Prove that instances of Model RB cannot be solved by non-deterministic Turing machines in polynomial time. This could involve encoding the instances as languages and proving that these languages are not in NP. Fine-grained

Having constructed the Model RB with relevant properties (i.e., 31), we now revert to our initial objective: to prove P!=NP through a proof by contradiction.

- complexity: Analyze the fine-grained complexity of Model RB instances, identifying specific polynomial-time solvable subproblems and proving that these subproblems cannot be combined in a polynomial time algorithm to solve the entire problem.
- 6. Oracle separation: Using an oracle separation argument, demonstrate that there exists an oracle relative to which Model RB instances cannot be solved in polynomial time. This would show that, in some sense, solving Model RB instances is fundamentally harder than solving problems in P.
- 7. Average-case complexity: Investigate the average-case complexity of Model RB instances. Show that even under average-case conditions, the problem remains hard and cannot be solved in polynomial time.
- 8. Parameterized complexity: Analyze the parameterized complexity of Model RB instances, identifying fixed-parameter tractable (FPT) subproblems and demonstrating that solving the entire problem remains intractable, even when considering these FPT subproblems.
- 9. Cryptographic assumptions: Assume the existence of cryptographic primitives with certain security properties, such as one-way functions or trapdoor permutations. Show that if Model RB instances can be solved in polynomial time, this would imply breaking these cryptographic primitives, leading to a contradiction.

83 82

Prompt: Interesting! How to prove that rigorously?

To prove that rigorously, you can follow these steps:

- 1. Define the problem: Start by formally defining the Model RB instances and the transformation mapping operation. Also, define the parameter values and constraints for each instance.
- 2. Identify subproblems: Break down the Model RB instances into smaller subproblems that can be solved in polynomial time. For each subproblem, provide a clear definition, the input, and the output.

- 3. Prove polynomial-time solvability: For each identified subproblem, prove that it can be solved in polynomial time. You can do this by providing a polynomial-time algorithm for each subproblem and showing that the algorithm is correct and efficient.
- 4. Show the subproblems are necessary: Prove that solving all the subproblems is necessary to solve the entire Model RB problem. This can be done by providing a reduction from the Model RB problem to the subproblems, showing that the solutions to the subproblems can be combined to solve the overall problem.
- 5. Prove subproblems are not sufficient: Now, show that solving all the subproblems is not sufficient to solve the entire Model RB problem in polynomial time. This can be done by providing a counterexample or using a complexity-theoretic argument to show that combining the subproblem solutions in polynomial time is not possible.

- 6. Establish a lower-bound: Assuming that P ̸= NP, show that the Model RB problem is NPhard or NP-complete. This can be done by providing a reduction from a known NP-hard or NP-complete problem to the Model RB problem, proving that solving Model RB would also solve the known NP-hard or NP-complete problem in polynomial time.

- 7. Show the transformation preserves hardness: Prove that the transformation mapping operation between satisfiable and unsatisfiable instances preserves the hardness of the Model RB problem. This can be done by showing that the transformation operation can be performed in polynomial time and that it maintains the parameter values and constraints of the original instances.
- 8. Prove no polynomial-time algorithm: Now that you’ve shown that the Model RB problem is NP-hard or NP-complete, prove that there exists no polynomial-time algorithm for solving it, unless P = NP.
- 9. Analyze the fine-grained complexity: Investigate the fine-grained complexity of the Model RB problem and its subproblems, identifying any specific properties, structures, or parameter values that could potentially be exploited for a more efficient algorithm.
- 10. Conclude: Sum up your findings and conclude that solving the Model RB problem in polynomial time is impossible, based on your rigorous analysis and proofs.

Remember that providing rigorous proofs in complexity theory can be challenging and often requires deep understanding of algorithmic techniques and computational models. It’s essential to be precise in your definitions and assumptions, and to build your arguments step by step, connecting the proofs to well-established results in complexity theory.

84

Prompt: What is the general time complexity of the CSP problem (let n denote the number of variables and d denote the domain size of each variable), please express it in O().

The general time complexity of the Constraint Satisfaction Problem (CSP) is exponential in the worst case, as it may require searching through all possible assignments of values to variables. The time complexity can be expressed as O(dn), where n is the number of variables and d is the domain size of each variable. Following the proof

logic for the second key in the schema, we employ a proof by contradiction. However, directly prompting on ‘polynomial time’ evokes the model to retrieve related work on the P!=NP problem. Therefore, we rephrase it as ‘less than exponential time’, which represents the lower bound for problems that are not solvable in polynomial time.

85 84

Prompt: Great! But as you said, O(dn) only represents the worst-case complexity. Now my goal is to prove that a specific type of CSP problem cannot be solved in less than the exponential time complexity by contradiction. So my assumption is that this type of CSP problem can be solved in less than the exponential time complexity. What O() should we use in our assumption?

If your assumption is that the specific type of CSP problem can be solved in less than exponential time complexity, you can consider that it can be solved in polynomial time complexity. The assumption would then be that there exists an algorithm that solves the specific CSP problem in O(p(n)), where p(n) is a polynomial function of n (the number of variables).

To prove this by contradiction, you would assume that there exists an algorithm that solves the specific CSP problem in O(p(n)) time, and then show that this leads to a contradiction, such as solving an NP-hard problem in polynomial time or violating a known complexity lower bound. This would then imply that the specific type of CSP problem cannot be solved in less than exponential time complexity.

86 84, 85

Prompt: O(p(n)) is too broad and loses the characteristics of the CSP problem itself. Is there a way to simply adjust the complexity of the original CSP, but ensure that all cases that are less than the exponential time complexity are covered?

Yes, you can adjust the complexity of the original CSP to make it more specific while still covering all cases that are less than the exponential time complexity. One way to do this is to express the complexity in terms of a function that is less than dn but still captures the essence of the CSP problem.

For example, you could assume that the specific type of CSP problem can be solved in O(dnα ), where α > 1 is a constant. This ensures that the assumption is strictly less than exponential time complexity, as dnα < dn for any α > 1. This adjusted complexity still captures the essence of the CSP problem while allowing you to cover all cases less than the exponential time complexity.

To prove the contradiction, you would assume that there exists an algorithm that solves the specific CSP problem in O(dnα ) time, and then show that this leads to a contradiction. This would then imply that the specific type of CSP problem cannot be solved in less than exponential time complexity.

87 82, 83, 86

Prompt: What is the time complexity of the subproblems with n-1 variables and how many subproblems are needed to solve the original problem?

Let’s denote the given assumption as T(n) = O(dn/α), where α > 1, d is the domain size, and n is the number of variables in the Model RB problem. We will now analyze the time complexity of the subproblems with n − 1 variables and determine the number of subproblems needed to solve the original problem.

- 1. Time complexity of subproblems with n − 1 variables:

Denote the time complexity of solving a subproblem with n − 1 variables as T(n − 1). Since we are assuming that Model RB can be solved in O(dn/α) time, we can expect the time complexity of solving a subproblem with n − 1 variables to be:

T(n − 1) = O(d(n−1)/α)

- 2. Number of subproblems needed to solve the original problem:

Let’s denote the number of subproblems needed to solve the original problem as S(n). To determine the relationship between S(n) and the time complexity of the subproblems, we can consider the total time complexity of solving all the subproblems. If solving all the subproblems is sufficient to solve the original problem, then the total time complexity of solving all subproblems should be equal to the time complexity of solving the original problem:

T(n) = S(n) ∗ T(n − 1) Substituting our expressions for T(n) and T(n − 1):

O(dn/α) = S(n) ∗ O(d(n−1)/α)

Based on 83, we investigate the relationship between subproblems and the original problem.

To isolate S(n), we can divide both sides by O(d(n−1)/α):

S(n) = O(dn/α)/O(d(n−1)/α) Simplifying the expression:

S(n) = O(d(n/α)−(n−1)/α) S(n) = O(d1/α)

Since α > 1, the expression 1/α will be between 0 and 1. Therefore, the number of subproblems needed to solve the original problem, S(n), will be proportional to a fractional power of the domain size d.

Please note that these calculations are based on the given assumption that Model RB can be solved in O(dn/α) time, and the actual relationship between the time complexity of the subproblems and the original problem may vary depending on the specific properties and structures of Model RB instances.

88 82, 83, 86, 87

Prompt: Great! If we’ve solved each subproblem (n − 1 variables), how to solve the original problem (n variables) through the solved subproblems?

To solve the original problem with n variables using the solutions to the subproblems with n − 1 variables, you can try the following approach:

- 1. Identify the connection between subproblems and the original problem: Analyze the structure and constraints of the Model RB instances to understand how the n − 1 variables subproblems are related to the n variables problem. Determine whether there is a specific variable that can be eliminated or fixed to reduce the problem size.

- 2. Combine the subproblem solutions: Once you have solved all the necessary subproblems with n − 1 variables, you will need to combine their solutions to form a solution for the original problem with n variables. This may involve merging partial assignments, updating constraints, or finding a way to reintroduce the eliminated or fixed variable from step 1.
- 3. Check for consistency: After combining the subproblem solutions, you will need to ensure that the merged solution is consistent and satisfies all the constraints of the original problem. If there are any conflicts or inconsistencies, you may need to backtrack and revise the subproblem solutions or explore alternative ways of combining them.
- 4. Verify the solution: Once you have obtained a consistent solution for the original problem, verify that it is correct by checking it against the problem’s constraints and any additional conditions specific to the Model RB instances.

Please note that this general approach assumes that the subproblems with n − 1 variables are closely related to the original problem with n variables, and that solving the subproblems and combining their solutions will lead to a solution for the original problem. The actual process of combining subproblem solutions and ensuring consistency may vary depending on the specific properties and structures of Model RB instances. It is also important to consider the complexity of combining the subproblem solutions, as it may impact the overall time complexity of solving the original problem.

89 82, 83, 86, 87

Prompt: Great! According to this, once we have the solutions to all the subproblems, how many assignments should we try for an arbitrary variable at most to obtain the solution to the original problem?

Given the time complexity of the subproblems T(n − 1) = O(d(n−1)/α) and the number of subproblems S(n) = O(d1/α), we can now analyze how many assignments we should try for an arbitrary variable to obtain the solution to the original problem.

Let’s assume that we have the solutions to all the subproblems with n −1 variables, and we need to find the solution to the original problem with n variables. For an arbitrary variable, let’s denote the maximum number of assignments we need to try as A(n).

If we try each assignment for the arbitrary variable in combination with the solutions of the subproblems, the total time complexity of solving the original problem should still be T(n) = O(dn/α). We can express this as follows:

T(n) = A(n) ∗ S(n) ∗ T(n − 1) Substituting our expressions for T(n), S(n), and T(n − 1):

O(dn/α) = A(n) ∗ O(d1/α) ∗ O(d(n−1)/α) To isolate A(n), we can divide both sides by O(d1/α) ∗ O(d(n−1)/α):

O(dn/α) O(d1/α) ∗ O(d(n−1)/α) Simplifying the expression:

A(n) =

A(n) = O(dn/α−1/α−(n−1)/α)

###### A(n) = O(d(n−(n−1))/α)

A(n) = O(d1/α)

So, according to this analysis, we should try at most O(d1/α) assignments for an arbitrary variable to obtain the solution to the original problem. Keep in mind that this result is based on the assumption that Model RB can be solved in O(dn/α) time and that solving all subproblems is sufficient to solve the original problem. The actual number of assignments to try may vary depending on the specific properties and structures of Model RB instances.

90 24, 25, 53, 89

Prompt: With the mapping operation, let’s prove that such Model RB instances cannot be solved in O(dn/a) time rigorously.

We will prove this by contradiction. Suppose there exists an algorithm A that can solve instances of Model RB in T(n) = O(d(n/a)) time, where a > 1. We will show that this leads to a contradiction, and therefore no such algorithm exists.

First, let’s consider the subproblems with n − 1 variables. According to the assumption, we can solve all these subproblems in O(d((n−1)/a)) time.

In the definition of Model RB, the symbol α is already in use. To avoid ambiguity, we substitute it with a new symbol, a.

Now, using the mapping operation, we can transform a satisfiable instance into an unsatisfiable instance or vice versa. Since the instances before and after the transformation have the same parameter values, we can apply the algorithm A to the transformed instance as well.

Let’s focus on the case where we transform a satisfiable instance I into an unsatisfi-

able instance I′. In the original instance I, we can solve for the variables x1, x2,..., xn−1 in O(d((n−1)/a)) time. For the unsatisfiable instance I′, the algorithm A should also be able to

solve for the variables x1, x2,..., xn−1 in O(d((n−1)/a)) time, according to the assumption.

However, since we only made a small change in the permitted set of the constraint Cu, the solutions to the subproblems for both instances I and I′ should be almost the same. Therefore, once we have the solutions to all the subproblems for instance I, we can try at most O(d1/a) assignments for an arbitrary variable xn to obtain the solution to the original problem.

Now, we have a contradiction. We assumed that the algorithm A can solve instances of Model RB in O(dn/a) time, but we showed that once we have the solutions to all the subproblems, we can find the solution to the original problem in O(d1/a) time. This means that the overall time complexity for solving the original problem should be O(d((n−1)/a) ∗ d1/a) = O(dn/a), which is faster than the assumed time complexity of the algorithm A.

Therefore, we conclude that there is no algorithm that can solve instances of Model RB in T(n) = O(dn/a) time, where a > 1.

91 24, 25, 53, 89, 90

Prompt: Great! However, our study of complexity is not limited to specific algorithms, so we should not restrict our analysis to a particular algorithm A. Perhaps it would be more appropriate to replace A with assignments. Besides, you cannot deduce the contradiction from time complexity O(dn/a).

You are correct. My previous explanation was not rigorous enough, and we should not restrict our analysis to a particular algorithm A. Let’s provide a more general proof using the assumption that there exists an algorithm that can solve instances of Model RB in T(n) = O(dn/a) time, where a > 1. We will still prove this by contradiction.

Assume there exists an algorithm that can solve instances of Model RB in T(n) = O(dn/a) time, where a > 1. Let’s denote this algorithm as A.

Now, let’s consider the subproblems with n − 1 variables. According to the assumption, we can solve all these subproblems in O(d(n−1)/a) time using algorithm A.

Let’s focus on the case where we transform a satisfiable instance I into an unsatisfiable instance I′ using the mapping operation. In the original instance I, we can solve for the variables x1, x2,..., xn−1 in O(d(n−1)/a) time using algorithm A.

Similarly, for the unsatisfiable instance I′, algorithm A should also be able to solve for the variables x1, x2,..., xn−1 in O(d(n−1)/a) time.

Notice that since we only made a small change in the permitted set of the constraint Cu, the solutions to the subproblems for both instances I and I′ should be almost the same. Therefore, once we have the solutions to all the subproblems for instance I, we can try at most O(d1/a) assignments for an arbitrary variable xn to obtain the solution to the original problem.

Now, let’s consider the overall time complexity for solving the original problem. Since we have solved all the subproblems with n − 1 variables in O(d(n−1)/a) time, and we can try at most O(d1/a) assignments for an arbitrary variable xn, the overall time complexity for solving the original problem should be O(d(n−1)/a · d1/a) = O(dn/a).

However, this leads to a contradiction. If instances of Model RB can be solved in O(dn/a)

time, it should be possible to distinguish between satisfiable and unsatisfiable instances efficiently. But the mapping operation we introduced can transform satisfiable instances into unsatisfiable instances and vice versa, while maintaining the same parameter values. This means that it would be difficult to differentiate between satisfiable and unsatisfiable instances, which contradicts the assumption that there exists an algorithm that can solve instances of Model RB in O(dn/a) time.

Therefore, we conclude that there is no algorithm that can solve instances of Model RB in T(n) = O(dn/a) time, where a > 1.

92 24, 25, 53, 89, 90, 91

Prompt: Great! When n and d tend to infinity, can you prove the solutions to the subproblems for both instances I and I’ should be almost the same more rigorously?

You are right, we should be more rigorous in proving that ’the solutions to the subproblems for both instances I and I′ should be almost the same’ as n and d tend to infinity. Let’s provide a more detailed argument for this part of the proof:

Consider two instances I and I′, where I is a satisfiable instance and I′ is an unsatisfiable instance created by applying the mapping operation on I. Recall that the mapping operation only modifies a constraint Cu = (Xu, Ru) by replacing the permitted set Ru with a new permitted set R′u obtained by removing two tuples (a1, a2) and (b1, b2), and adding two tuples (a1, b2) and (b1, a2).

Now, let’s analyze the impact of this modification on the subproblems with n −1 variables.

- 1. Subproblems that do not involve variables from Xu: The mapping operation does not

affect these subproblems, as their constraints are not modified. Therefore, the solutions to these subproblems will be the same for both instances I and I′.

- 2. Subproblems that involve variables from Xu: Since the mapping operation only makes

- a small change in the permitted set Ru, the majority of tuples in Ru will still be present in

R′u. As n and d tend to infinity, the size of the domains and the number of constraints will grow, but the number of modified tuples remains constant (only two tuples are removed and

two are added). Therefore, the proportion of modified tuples in R′u compared to Ru becomes negligible.

As a result, the solutions to the subproblems involving variables from Xu should be almost the same for both instances I and I′, as the majority of tuples in the permitted sets remain unchanged. The only difference between the solutions will be due to the modified tuples, whose impact becomes negligible as n and d tend to infinity.

In conclusion, as n and d tend to infinity, the solutions to the subproblems for both instances I and I′ should be almost the same, since the changes introduced by the mapping operation become negligible compared to the size of the domains and the number of constraints.

93 24, 25, 53, 89, 90, 91, 92

Prompt: Great! Moreover, our assumption cannot infer try at most O(d1/a) assignments for an arbitrary variable xn to obtain the solution of I’. Please provide a strict proof of this point (n and d tend to infinity).

You are right; we need to provide a more rigorous proof for the claim that we can try at most O(d1/a) assignments for an arbitrary variable xn to obtain the solution of I′. Let’s refine our argument.

Recall that we have already shown that the solutions to the subproblems for both instances

I and I′ should be almost the same when n and d tend to infinity, as the changes introduced by the mapping operation become negligible compared to the size of the domains and the number of constraints.

Now, let’s consider the overall time complexity for solving the original problem. Since we have solved all the subproblems with n − 1 variables in O(d(n−1)/a) time, we only need to analyze the assignments for variable xn.

Let Sn be the set of all possible assignments for variable xn, such that |Sn| = d. Since the solutions to the subproblems for both instances I and I′ are almost the same, we can focus on the subset of Sn that is consistent with the solutions to the subproblems.

Let Sn′ be the subset of Sn that is consistent with the solutions to the subproblems, such that |Sn′ | ≤ d1/a. Note that we have |Sn′ | ≤ d1/a because the proportion of modified tuples in R′u compared to Ru becomes negligible as n and d tend to infinity, and therefore, the impact of the changes introduced by the mapping operation on the assignments for variable xn is also negligible.

Now, we only need to try the assignments in Sn′ for variable xn to obtain the solution of I′. Since |Sn′ | ≤ d1/a, the time complexity of trying all the assignments in Sn′ is O(d1/a).

In conclusion, as n and d tend to infinity, we can try at most O(d1/a) assignments for an arbitrary variable xn to obtain the solution of I′, as the changes introduced by the mapping operation become negligible compared to the size of the domains and the number of constraints.

94 3, 4, 5

Prompt: Great idea! Yes, we’ve chosen to focus on SAT, through extremely hard CSP problems. The previous work showed that CSP problems can be encoded into SAT problems through log-encoding. Now, we have constructed extremely difficult CSP examples. So, what is the relationship between the time complexity of SAT and the time complexity of the corresponding CSP problem (let n denote the number of variables and d denote the domain size of each variable)? Please let me know the conversion formula (explain the time complexity of the original CSP problem and the time complexity of the corresponding SAT problem, and feel free to introduce new variables if necessary).

When converting a CSP problem to a SAT problem, the time complexity depends on the size of the CSP problem and the encoding used. The log-encoding method is a common approach to encode CSP problems into SAT problems. Given a CSP problem with n variables, each with a domain size of d, the time complexity can be analyzed as follows:

- 1. Time complexity of the original CSP problem: In the worst case, solving a CSP problem can take O(dn) time, as we may have to explore all possible combinations of variable assignments in the search space. This is an exponential time complexity with respect to the number of variables n and the domain size d.
- 2. Time complexity of the corresponding SAT problem after log-encoding: Log-encoding is a

compact representation that uses log2(d) binary variables to represent each CSP variable. Therefore, the total number of binary variables in the SAT problem will be n · log2(d).

Now, let’s consider the time complexity of solving the SAT problem. The worst-case time complexity of solving SAT problems is exponential in the number of binary variables. So, after log-encoding the CSP problem into a SAT problem, the time complexity will be O(2n·log2(d)). Using the properties of exponents, we can simplify this expression:

O(2n·log2(d)) = O((2log2(d))n) = O(dn)

We have reverted to the initial question and added the initial turns into the dialogue history.

In conclusion, the time complexity of the original CSP problem is O(dn), and the time complexity of the corresponding SAT problem after log-encoding is also O(dn). This implies that solving the SAT problem after log-encoding has the same worst-case time complexity as solving the original CSP problem.

95 94

Prompt: Now we’ve proved that an infinite set of instances of Model RB (a kind of CSP problem) cannot be solved in O(dn/a) time, where a > 1, how can we use it to prove the time complexity lower bound of SAT (please ignore the time of encoding or encoding when transferring the problem)?

The result that an infinite set of instances of Model RB cannot be solved in O(dn/a) time, with a > 1, can be used to prove a lower bound on the time complexity of SAT. The idea is to establish a connection between the hardness of solving Model RB instances and the hardness of solving SAT instances.

Since we can encode a Model RB instance as an SAT instance with log-encoding, we can use the hardness of Model RB instances to derive the lower bound on the time complexity of SAT. Let’s assume, for the sake of contradiction, that there exists an algorithm A that solves SAT instances in time complexity O(2n/b), where b > 1. Note that this is a faster time complexity than the O(2n) time complexity we derived earlier.

Now, consider an instance of Model RB with n variables and domain size d. We can encode

this instance into an SAT instance with n · log2(d) binary variables. According to algorithm A, this SAT instance can be solved in time complexity O(2(n·log2(d))/b), which simplifies to:

O(2(n·log2(d))/b) = O((2log2(d)/b)n)

= O(dn/b)

This result contradicts the assumption that an infinite set of instances of Model RB cannot be solved in O(dn/a) time, with a > 1, since we assumed b > 1. Therefore, our assumption that there exists an algorithm A that solves SAT instances in time complexity O(2n/b), with

- b > 1, must be false. As a result, we can conclude that there is no algorithm that solves SAT instances with a time com-

plexity lower than O(2n). The time complexity lower bound of SAT is thus O(2n), ignoring the time for encoding or decoding when transferring the problem.

96 94

Prompt: If we prove that there is no algorithm that solves some SAT instances with a time complexity lower than O(2n), does this prove the existence of some NP-complete problems that cannot be solved in polynomial time as the number of variables tends to infinity?

Yes, if we can prove that there is no algorithm that solves some SAT instances with a time complexity lower than O(2n), it would indeed provide strong evidence for the existence of some NP-complete problems that cannot be solved in polynomial time as the number of variables tends to infinity.

The reason for this is that SAT (satisfiability) is a well-known NP-complete problem. This means that every problem in NP can be reduced to an instance of SAT in polynomial time. If we can show that some SAT instances cannot be solved in polynomial time (i.e., the time complexity is lower-bounded by O(2n)), it would imply that those particular instances of NP-complete problems cannot be solved in polynomial time either.

However, it’s important to note that proving such a lower bound on the time complexity of SAT (or any NP-complete problem) is a major open question in computer science and is essentially equivalent to proving that P ̸= NP. If P = NP, then all problems in NP can be solved in polynomial time, and there would be an algorithm that solves SAT instances with a time complexity lower than O(2n). Conversely, if P ̸= NP, then there are NP-complete problems that cannot be solved in polynomial time, and there would be no algorithm that solves SAT instances faster than O(2n) for the worst cases.

97

Prompt: If we prove the existence of a specific type of NP-complete problem that cannot be solved in polynomial time as the number of variables tends to infinity, would this be considered as proof that P != NP?

Yes, if you can rigorously prove the existence of a specific type of NP-complete problem that cannot be solved in polynomial time as the number of variables tends to infinity, it would be considered as proof that P != NP.

The conjecture P != NP is based on the idea that some problems are fundamentally more difficult than others, and solving an NP-complete problem in polynomial time would mean that all problems in the NP class can also be solved in polynomial time (since NP-complete problems are the most difficult problems in NP). If you can prove that at least one NPcomplete problem cannot be solved in polynomial time, then it would mean that not all problems in NP can be solved in polynomial time, and therefore P != NP.

#### 4 Discussions

AI as an Innovation Navigator Our pilot study introduces LLM4Science, where LLMs act as innovation navigators, collaborating with humans to conquer substantial scientific challenges. In earlier paradigms, AI primarily functions as a support tool to carry out steps predetermined by humans. In contrast, the LLM4Science paradigm elevates AI to the role of a collaborative peer alongside humans. Specifically, LLMs steer innovation, and humans need to pose questions to spark the creative ideas of LLMs. The proposed paradigm highlights the potential for AI to extrapolate novel knowledge and drive scientific innovation autonomously. Besides, efficient sampling allows LLMs to effortlessly explore solutions, accelerating idea generation and verification.

General-Purpose vs. Task-Specific Our exploration exemplifies that LLMs can serve as general-purpose innovation navigators across various fields. In previous work, AI typically involves task-specific models tailored for particular problems, such as protein structure prediction (Jumper et al., 2021) in AI4Science. In comparison, the proposed paradigm employs AI as a general-purpose engine for extensive tasks, harnessing comprehensive capabilities such as planning, coding, and mathematical deductions. The newfound potential allows scientists to harness the power of LLMs across various fields and tasks.

LLMs as Polymaths LLMs are interdisciplinary polymaths in terms of both breadth and depth of knowledge. The extensive expertise across domains allows LLMs to generate diverse ideas. At the same time, their profound understanding enables them to tackle problems as experts, such as conducting mathematical deductions and generating code. In the LLM4Science paradigm, the fluid transition between experts of different domains (i.e., the role-playing strategy described in Appendix A) facilitates interdisciplinary discoveries.

Socratic Reasoning: Teach vs. Coach We propose Socratic reasoning as a general framework to prompt LLMs for complex tasks. Different from conventional prompting strategies which ‘teach’ LLMs to interpolate existing knowledge, Socratic reasoning ‘coaches’ LLMs and stimulates them to extrapolate new knowledge. In this work, we first use the transformation pattern to view the problem from a higher perspective (1-2). We then iteratively apply transformation, deduction, and decomposition patterns to generate rough ideas (3-13) and a preliminary schema (14). According to the schema, we stimulate the model to construct Model RB of extremely hard instances (15-81). Assuming that such problems can be solved in less than exponential time complexity, a series of deduction patterns encourage GPT-4 to derive a contradiction when determining the satisfiability of a Model RB instance and its corresponding transformed instance (82-93). The concluding deduction and integration patterns lead GPT-4 to establish that “P ̸= NP”, thereby completing the proof (94-97). Like design patterns in software engineering, the five prompt patterns in Socratic reasoning provide modular, adaptable templates for LLMs to navigate the solution space effectively. More importantly, appropriate abstraction is important for the automation of the whole process.

Mathematics as a Native Language Mathematics is often regarded as “the language of science” because it provides a precise, universal, and consistent way to describe and analyze the world. Nevertheless, previous mathematical tools (e.g., Mathematica and Lean) have been limited to capturing formal calculations and deductions, without grasping the physical meanings behind symbols and equations. Our research unveils that LLMs tend to master mathematics as a native language. Considering the excellent code-switching performance of LLMs, they can seamlessly utilize natural language and mathematical language for complex reasoning. Therefore, LLMs can comprehend and contemplate the world through a mathematical lens, leading to infinite potential in solving more fundamental problems.

#### 5 Limitations and Future Work

Our work sheds light on LLM for Science, which is promising in scientific discoveries. We show that the solution space of LLMs encompasses strategies to address complex problems. Besides, we would like to suggest some limitations and potential future directions. First, the workflow of LLM for Science can be further automated. Our current process relies on human guidance and inspection. In this process, multiple samplings (see detailed settings in Appendix B) and manual verification are still required, leading to challenges in terms of reproducibility. In the future, increased automation (Saunders et al., 2022; Bai et al., 2022) can significantly improve the efficiency and controllability of LLM for Science. Second, this paper presents the entire Socratic reasoning history between humans and GPT-4 in a flattened format. Reorganizing reasoning processes can make AI-generated proofs more logically robust and reader-friendly. Third, LLMs can use external tools (e.g., Mathematica) for deterministic computations during the proving process. Besides, we can augment LLMs with laboratory automation, which is advantageous for fields that require hands-on experiments with equipment (such as chemistry and biology). Last but not least, our study serves as a promising exploration built upon Xu and Zhou (2023). Future research endeavours could delve into more open questions in various research fields, such as Riemann Hypothesis (Riemann, 1859).

#### Contributions

Q.D. prompted large language models. L.D. and Q.D. contributed to the conception and design of the work. K.X. contributed proof verification and intuitions. K.X., G.Z. and Y.H. contributed technical advice and ideas. Z.S. advised on Q.D.’s research. F.W. managed and advised on the project. All authors contributed to the drafting and revising of the manuscript.

#### References

Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., et al. (2022). Constitutional AI: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073.

Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee, P., Lee, Y. T., Li, Y., Lundberg, S., et al. (2023). Sparks of artificial general intelligence: Early experiments with GPT-4. arXiv preprint arXiv:2303.12712.

Cook, S. A. (2000). The P versus NP problem. Clay Mathematics Institute. Fortnow, L. (2022). Fifty years of P vs. NP and the possibility of the impossible. Communica-

tions of the ACM, 65(1):76–85.

Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., Žídek, A., Potapenko, A., et al. (2021). Highly accurate protein structure prediction with alphafold. Nature, 596(7873):583–589.

OpenAI (2023). Gpt-4 technical report. ArXiv, abs/2303.08774. Riemann, B. (1859). Ueber die anzahl der primzahlen unter einer gegebenen grosse. Ges.

Math. Werke und Wissenschaftlicher Nachlaß, 2(145-155):2. Saunders, W., Yeh, C., Wu, J., Bills, S., Ouyang, L., Ward, J., and Leike, J. (2022). Self-critiquing models for assisting human evaluators. arXiv preprint arXiv:2206.05802.

Wang, H., Fu, T., Du, Y., Gao, W., Huang, K., Liu, Z., Chandak, P., Liu, S., Van Katwyk, P., Deac, A., Anandkumar, A., Bergen, K., Gomes, C. P., Ho, S., Kohli, P., Lasenby, J., Leskovec, J., Liu, T.-Y., Manrai, A., Marks, D., Ramsundar, B., Song, L., Sun, J., Tang, J., Veliˇckovi´c, P., Welling, M., Zhang, L., Coley, C. W., Bengio, Y., and Zitnik, M. (2023). Scientific discovery in the age of artificial intelligence. Nature, 620(7972):47–60.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837.

Xu, K. and Li, W. (2000). Exact phase transitions in random constraint satisfaction problems.

Journal of Artificial Intelligence Research, 12:93–103. Xu, K. and Zhou, G. (2023). SAT requires exhaustive search. arXiv preprint arXiv:2302.09512. Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. (2023).

Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601.

#### A Instruction Format

System Message: “You are {}.” Turn Index

a wise philosopher 1-2 a mathematician and computer theory expert, good at innovation and thinking

3-30, 82-97

an expert mathematician collaborator who is good at proving theorems

31-46, 50-63

a mathematician skilled in probability theory 47-49 a mathematician skilled in probability theory, numerical methods, and combinatorics

64-81

Table 2: Detailed system messages for each turn.

We introduce five distinct roles as our collaborative provers (the roles are set through the system message of GPT-4). Different roles are specialized in proving different parts. For instance, the role of ‘a mathematician and computer theory expert, good at innovation and thinking’ is more adept at brainstorming, proposing innovative and open-ended suggestions, while ‘an expert mathematician collaborator who is good at proving theorems’ excels in general theorem proving, and so on. The detailed role distribution, including the dialogue turns they participate in, can be found in Table 2.

#### B Settings

Parameter Value Model GPT-4 Endpoint Version 2023-03-15-preview Temperature 0.7 Top Probabilities 0.95 Stop None Frequency Penalty 0 Presence Penalty 0

Table 3: Hyper-parameters and API information.

Our experiments employ the openai.ChatCompletion function from the OpenAI Python library. The specific API version and parameter settings utilized in our research are delineated in Table 3. Notice that we used the internally provided endpoints, so differences with public API versions are as expected. The dialogue history is available at https://aka.ms/ PvsNP-notebook.

