Attention Heads of Large Language Models: A Survey Zifan Zheng1,4, Yezhaohui Wang1,4, Yuxin Huang2,4, Shichao Song1, Mingchuan Yang3, Bo Tang1, Feiyu Xiong1, and Zhiyu Li1,*

- 1Institute for Advanced Algorithms Research (IAAR), Shanghai, China
- 2Institute for AI Industry Research (AIR), Tsinghua University, Beijing, China 3Research Institute of China Telecom, Beijing, China 4These authors contributed equally.

*Correspondence: lizy@iaar.ac.cn

###### Summary

### arXiv:2409.03752v3[cs.CL]23Dec2024

Since the advent of ChatGPT, Large Language Models (LLMs) have excelled in various tasks but remain as black-box systems. Understanding the reasoning bottlenecks of LLMs has become a critical challenge, as these limitations are deeply tied to their internal architecture. Among these, attention heads have emerged as a focal point for investigating the underlying mechanics of LLMs. In this survey, we aim to demystify the internal reasoning processes of LLMs by systematically exploring the roles and mechanisms of attention heads. We first introduce a novel four-stage framework inspired by the human thought process: Knowledge Recalling, In-Context Identification, Latent Reasoning, and Expression Preparation. Using this framework, we comprehensively review existing research to identify and categorize the functions of specific attention heads. Additionally, we analyze the experimental methodologies used to discover these special heads, dividing them into two categories: Modeling-Free and Modeling-Required methods. We further summarize relevant evaluation methods and benchmarks. Finally, we discuss the limitations of current research and propose several potential future directions.

###### Keywords

Attention Head, Mechanistic Interpretability, Large Language Model (LLMs), Cognitive Neuroscience

###### 1 Introduction

The Transformer architecture1 has demonstrated outstanding performance across various tasks, such as Natural Language Inference and Natural Language Generation. However, it still retains the black-box nature inherent to Deep Neural Networks (DNNs).2,3 As a result, many researchers have dedicated efforts to understanding the internal reasoning processes within these models, aiming to uncover the underlying mechanisms.4 This line of research provides a theoretical foundation for models like BERT5 and GPT6 to perform well in downstream applications. Additionally, in the current era where Large Language Models (LLMs) are widely applied, interpretability mechanisms can guide researchers in intervening in specific stages of LLM inference, thereby enhancing their problem-solving capabilities.7–9 Among the components of LLMs, attention heads play a crucial role in the reasoning process. Particularly in recent years, attention heads within LLMs have garnered significant attention, as illustrated in Figure 1. Numerous studies have explored attention heads with specific functions. This paper consolidates these research efforts, organizing and analyzing the potential mechanisms of different types of attention heads. Additionally, we summarize the methodologies employed in these investigations.

[Figure 1]

GoogleTrendsPopularity

ChatGPT Released

Figure 1: The global Google Trends Popularity of the keywords “Attention Head” and “Model Interpretability”. The data retrieval date is December 4th, 2024.

The logical structure and classification method of this paper are illustrated in Figure 2. We begin with the background of the problem in Background, where we present a simplified representation of the LLMs’ structures (Mathematical representation of LLMs) and explain the related key terms (Glossary of key terms). In Overview of special attention heads, we first summarize the four stages of human thought processes from a cognitive neuroscience perspective and apply this framework to analyze the reasoning mechanisms of LLMs. Using this as our classification criterion, we categorize existing work on attention heads, identifying commonalities among heads that contribute to similar reasoning processes (from Knowledge Recalling (KR) to Expression Preparation (EP)) and exploring the collaborative mechanisms of heads functioning at different stages (How do attention heads work together?).

Investigating the internal mechanisms of models often requires extensive experiments to validate hypotheses. To provide a comprehensive understanding of these methods, we summarize the current experimental methodologies used to explore special attention heads in Unveiling the discovery of attention heads. We divide these methodologies into two main categories based on whether they require additional modeling: Modeling-Free (Modeling-Free) and Modeling-Required (Modeling-Required).

In addition to the core sections shown in Figure 2, we summarize the evaluation tasks and benchmarks used in relevant studies in Evaluation. Furthermore, in Additional topics, we compile research on the mechanisms of Feed-Forward Networks (FFNs) and Mechanical Interpretability to help deepen our understanding of LLM structures from multiple perspectives. Finally, in Discussion, we offer our insights on the current state of research in this field and outline several potential directions for future research.

In summary, the strengths of our work are:

- • Focus on the latest research. Although earlier researchers explored the mechanisms of attention heads in models like BERT, many of these conclusions are now outdated. This paper primarily focuses on highly popular LLMs, such as LLaMA and GPT, consolidating the latest research findings.
- • An innovative four-stage framework for LLM reasoning. We have distilled key stages of human thought processes by integrating knowledge from cognitive neuroscience, psychology, and related fields. Furthermore, we have applied these stages as an analogy for LLM reasoning.

Attention Heads of Large Language Models: A Survey

Introduction & Background

Black-box LLM White-box Model

# … + N +

FF

.

Mechanism? ..

| | |
|---|---|
|Special Attention Heads| |

###### Collaborative Mechanism

Relation between 4 stages and model layers

[Figure 2]

EP LR ICI KR

Knowledge Recalling (KR)

In-Context Identification (lCl)

Shallow Middle Deep

Human

LLM

MCQA Syllogism IOI …

Latent Reasoning (LR)

Application Scene

Expression Preparation (EP)

Experimental Method

Discussion

Evaluation

Limitations

Mechanism Exploration

Modeling-Free

Generalizability Multi-collaboration Theoretical support

Modification-Based

Common Evaluation

Replacement-Based

Add. Topics

Future Directions

Modeling-Required

Complex Task Unified Framework Align with Human

FFN Interpretability

Training-Free

Machine Psychology

Training-Required

Figure 2: The framework of this survey.

- • Detailed categorization of attention heads. Based on the proposed four-stage framework, we classify different attention heads according to their functions within these stages, and we explain how heads operating at different stages collaborate to achieve alignment between humans and LLMs.
- • Clear summarization of experimental methods. We provide a detailed categorization of the current methods used to explore attention head functions from the perspective of model dependency, laying a foundation for the improvement and innovation of experimental methods in future research.

###### 2 Out-of-scope topics

First, we need to clarify the boundaries of the topic reviewed in this paper. In other words, some works fall outside the scope of our focus.

As the latest research on attention head interpretability is primarily based on LLMs, this paper focuses on the heads within current mainstream LLM architectures, specifically those with a decoder-only structure. As such, we do not discuss early studies related to the Transformer, such as those focusing on attention heads in BERT-based models.10–12

Some studies on mechanistic interpretability propose holistic operational principles that encompass embeddings, attention heads, and MLPs. However, this paper focuses exclusively on attention heads. Consequently, we do not cover the roles of other components within the Transformer architecture from Overview of special attention heads to Unveiling the discovery of attention heads; these are only briefly summarized in Additional topics.

###### 3 Background

###### 3.1 Mathematical representation of LLMs

As mentioned in Out-of-scope topics, to facilitate the discussion in the subsequent sections, we first define the mathematical notations for the transformer layer of an LLM. Note that there are two main layer normalization methods in LLMs: Pre-Norm and Post-Norm.13,14 However, since these are not the focus of this paper, we will omit Layer Normalization in this section.

As shown in Figure 3, a model M consists of an embedding layer, L identical transformer layers, and an unembedding layer. The input to M are one-hot sentence tokens, with a shape of {0,1}N×|V|, where N is the length of the token sequence and |V| represents the vocabulary size.

After passing through the embedding layer, which applies semantic embedding WE ∈ R|V|×d and positional encoding PE (e.g., RoPE15), the one-hot matrix is transformed into the input X0,0 ∈ RN×d for the first transformer layer, where d represents the dimension of the token embedding (or latent vector).

###### Transformer Layer × 𝑳𝑳

Χℓ+1,0

Χℓ,1

Χℓ,0

|𝑉𝑉|

|𝑉𝑉|

d

d

|V|

| |
|---|
| |
| |
| |

## +

+ FFN

𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝑡𝑡0(Χℓ,0)

Χℓ,1

Χ𝐿𝐿+1,0

𝐹𝐹𝐹𝐹𝐹𝐹ℓ(Χℓ,1)

Χ0,0

...

N

𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴1𝑡𝑡(Χℓ,0)

Qℎℓ Κℓℎ Vℓℎ

…

Unembedding Layer

One-hot Vectors

Next Token

###### Embedding

| | |
|---|---|
| | |

× Oℓℎ

Layer

𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝑡𝑡𝐻𝐻(Χℓ,0)

Attention Head

Figure 3: The overall structure of decoder-only LLMs.

In the ℓ-th ℓ(1 ≤ ℓ ≤ L) transformer layer, there are two residual blocks. The first residual

block takes the input matrix Xℓ,0 ∈ RN×d and combines it with the output Xattnℓ —produced by a multi-head attention mechanism with H attention heads—to compute Xℓ,1 (as shown

in Equation 1). Subsequently, Xℓ,1 serves as the input for the second residual block. Here, Attnhℓ (·) (1 ≤ ℓ ≤ L,1 ≤ h ≤ H) represents the computation function of the h-th attention head in the ℓ-th layer (see Equation 3), where 1 ≤ h ≤ H.

H

Xattnℓ =

Attnhℓ (Xℓ,0)

(1)

h=1

Xℓ,1 = Xℓ,0 + Xattnℓ

Similarly, as shown in Equation 2, the second residual block combines Xℓ,1 with the output Xffnℓ

obtained after passing through the FFN, yielding the final output Xℓ+1,0 of the ℓ-th decoder block. This output also serves as the input for the ℓ+1-th decoder block. Here, FFNℓ (·) consists of linear layers (and activation functions) such as GLU (Gated Linear Units), SwiGLU,16 or MoEs.17,18

Xffnℓ = FFNℓ (Xℓ,1) Xℓ+1,0 = Xℓ,1 + Xffnℓ

(2)

Here, we will concentrate on the details of Attnhℓ (·). This function can be expressed using matrix operations. Specifically, each layer’s Attnhℓ (·) function corresponds to four low-rank matrices: WQhℓ,WKhℓ,WVhℓ ∈ Rd×

d

d

H×d. By multiplying Xℓ,0 with WQhℓ, the query matrix Qhℓ ∈ RN×

H ,Ohℓ ∈ R

d

H is obtained. Similarly, the key matrix Khℓ and the value matrix Vℓh can be derived. The function Attnhℓ (·) can then be expressed as Equation 3.1

⊤

###### ℓ · Vℓh · Ohℓ (3)

Attnhℓ (Xℓ,0) = softmax Qhℓ · Kh

| |
|---|
| |
| |
| |

#### +

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

| |
|---|
| |
| |
| |

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

+ + + + + + + + + + + +

[Figure 13]

| |
|---|
| |
| |
| |

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

+ + + + + + + + + + + +

[Figure 20]

| |
|---|
| |
| |
| |

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

+ + + + + + + + + + + +

[Figure 27]

| |
|---|
| |
| |
| |

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

+ + + + + + + + + + + +

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Figure 4: The diagram of residual streams. From the perspective of residual streams, the inference process of LLMs can be understood at a micro-level where attention heads access latent state matrices from several residual streams, as indicated by the gray arrows across layers in the diagram. At a macro-level, different residual streams control the information flow through attention heads, as shown by the gray wavy lines in the diagram.

###### 3.2 Glossary of key terms

This paper mentions several specialized terms that are fundamental to understanding and analyzing the reasoning mechanisms of LLMs. These terms are organized into two categories: conceptual frameworks, which provide theoretical abstractions for modeling LLM reasoning, and empirical analysis methods, which offer practical tools for experimentally probing and validating these frameworks. Below, we provide explanations for these key terms. For additional definitions related to model interpretability, please refer to the works of Nanda19.

###### 3.2.1 Conceptual frameworks

Circuits Circuits are abstractions of the reasoning logic in deep models. The model M is viewed as a computational graph. There are two main approaches to modeling circuits. One approach treats the features in the latent space of M as nodes and the transitions between features as edges.20,21 The other approach views different components of M, such as attention heads and neurons, as nodes; and the interactions between these components, such as residual connections, as edges.22 A circuit is a subgraph of M. Researchers have discovered many important circuits, such as the Bias Circuit,23 Knowledge Circuit,24 and so on.

Residual Stream As shown in Figure 4, each row in the figure can be viewed as a residual stream. The residual stream after layer ℓ is the sum of the embedding and the outputs of all layers up to layer ℓ, serving as the input to layer ℓ+1. Elhage et al.25 conceptualized the residual stream as a shared bandwidth through which information can flow. Different layers (or tokens) utilize this shared bandwidth, with lower layers (or previous tokens) writing information and higher layers (or subsequent tokens) reading it.

QK Matrix & OV Matrix We expand Equation 3 into Equation 4. According to the study by Elhage et al.25, WQhℓWKh

⊤

ℓ is referred to as the QK matrix (QK circuit), while WVhℓOhℓ is referred to as the OV matrix (OV circuit). Specifically, the QK matrix enables the computation of attention scores between the N tokens in Xℓ,0, thereby facilitating the reading of information from certain residual streams. Meanwhile, the OV matrix is responsible for writing the processed information back into the corresponding residual streams.

Query Vectors’ Matrix Key Vectors’ Matrix Value Vectors’ Matrix

⊤

Attnhℓ (Xℓ,0) = softmax Xℓ,0 · WQhℓ · Xℓ,0 · WKhℓ

· Xℓ,0 · WVhℓ · Ohℓ

(4)

⊤

ℓ · X⊤ℓ,0 · Xℓ,0 · WVhℓOhℓ

= softmax Xℓ,0 · WQhℓWKh

QK Matrix OV Matrix

###### 3.2.2 Empirical analysis methods

Activation Patching Activation patching is aimed to analyze the impact of the modifications on the model’s final decisions. It involves substituting activation values in specific layers of a model with alternatives—such as activations from different inputs, baseline values, or perturbed versions. Specifically, three types of effects are considered: direct effect, indirect effect, and total effect, as illustrated in Figure 5.

##### + =

Outputs Outputs Outputs

Outputs

Original Direct Effect Indirect Effect Total Effect

Figure 5: Three different types of calculating effects.

Ablation study Ablation study and activation patching are conceptually related but differ in their methods of operation. Instead of replacing activations, it involves removing specific components of the LLM to observe how the output is affected.26 The key distinction between the two methods lies in their mechanism: activation patching modifies activations to simulate the logical replacement of a component, whereas ablation study physically removes the component entirely.

Logit lens When calculating effects like those shown in Figure 5, logit lens can quantify this effect. It is often used in conjunction with activation patching or ablation studies. Specifically, it uses the unembedding layer to map an intermediate representation vector to the logits values of the vocabulary, allowing for the comparison of logits differences or other metrics. More details are in the Colab notebook.

###### 3.3 Existing related surveys

To the best of our knowledge, there is no survey focused on the mechanisms of LLMs’ attention heads. Specifically, Rauker¨ et al.27 mainly discussed non-Transformer architectures, with little focus on attention heads. The surveys by Gon¸calves et al.28, Santana and Colombini29, Chaudhari et al.30, Brauwers and Frasincar31 cover older content, primarily focusing on the various attention computation methods that emerged during the early development of the Transformer. However, current LLMs still use the original scaled-dot product attention, indicating that many of the derived attention forms have become outdated. Although Luo and Specia32 focused on the internal structure of LLMs, they only summarized experimental methodologies and overlooked research findings related to operational mechanisms.

Table 1: Summary of the relationship between LLMs and human behaviors explored in existing studies.

Research Paper Viewpoints on the Relationship Between LLMs and Human (Brains) Liang et al.33

“Self-Feedback” mechanism in LLMs mirrors human metacognition34 by enabling models to evaluate and refine their own reasoning.

Dasgupta et al.35 The language model can exhibit many of the varied, context-sensitive patterns of human reasoning behavior. Li et al.36

Different attention heads in LLMs exhibit specialized roles, analogous to the modular organization of human brain regions.

Janik37 LLMs exhibit some human-like memory characteristics, such as primacy and recency effects. Schrimpf et al.38

Representations in Transformers show significant similarity to human brain neural activities during language tasks, particularly in terms of predictive processing (Errors flows bottom-up to adjust the model).

The attention distributions of LLMs for implicit semantic relations in language closely align with human response patterns in perceptual tasks.

Marjieh et al.39

Mischler et al.40 The attention mechanism may partially reflect the brain’s predictive coding theory41.

###### 4 Overview of special attention heads

Previous research has shown that the decoder-only architecture described in Background follows the Scaling Law, and it exhibits emergent abilities once the number of parameters reaches a certain threshold.42,43 Many LLMs that have emerged subsequently demonstrate outstanding performance in numerous tasks, even close to humans. However, researchers still do not fully understand why these models are able to achieve such remarkable results. To address this question, recent studies have begun to delve into the internal mechanisms of LLMs, focusing on their fundamental structure—a neural network composed of multi-attention heads and FFNs.

We have observed that many studies concentrate on the functions of attention heads, attempting to explain their reasoning processes. Additionally, several researchers have drawn parallels of reasoning methods between LLMs and human, as illustrated in Table 1. These findings suggest that certain research insights from studies of the human brain may be transferable to the study of attention heads. Therefore, in this section, we first summarize a four-stage framework inspired by human cognitive paradigms and use it as a guiding method to classify the functions of different attention heads.

###### 4.1 How does the human brain & attention head think?

As shown in Table 1, the role of an attention head, as its name suggests, is quite analogous to the functions of the human brain. In some representative earlier works, the OAR model abstracts human brain knowledge and information into a graph composed of objects, attributes, and relations.44 Based on this abstraction, Wang and Chiew45 proposed a mathematical model of problem solving. Specifically, the solver’s brain first utilizes its own OAR model to identify the content of the problem, distinguishing the objects and attributes within it, and constructs a sub-OAR model accordingly. Then, the solver combines their knowledge to search for potential solution goals and solution paths, evaluating these candidate solutions. If the evaluation results are unsatisfactory, the solver iteratively explores and evaluates solutions until suitable ones are found. Ultimately, the result of problem solving is represented as a part of the relations in the sub-OAR model.

Similarly, the ACT-R model, which consists of five modules—Perception (P), Working Memory (WM), Procedural Memory (PM), Declarative Memory (DM), and Motor (M)—highlights the interaction between various modules in human cognition.46 The P module receives environmental inputs (e.g., visual or auditory information) and transmits them to the WM module. WM retrieves condition-action rules stored in PM in an if-then format to generate the next action. If additional knowledge is required, WM retrieves it from DM. Finally, the action is executed through the M module.47,48

In summary, these studies center on how humans retrieve knowledge, perceive and understand problems or environments, and conceive and execute actions. Inspired by these works, we propose a more universally applicable four-stage framework for describing the process by which the human brain solves specific problems: Knowledge Recalling (KR), In-Context Identification (ICI), Latent Reasoning (LR), and Expression Preparation (EP). These four stages can interact with and transition between one another, as illustrated in Figure 6.

When solving a problem, humans first need to recall the knowledge they have learned that is relevant to the issue at hand. This process is known as Knowledge Recalling (KR). During this stage, the hippocampus integrates memories into the brain’s network49 and activates different types of memories as needed with the help of dynamic associations.50,51 Confronted with the specific text of the problem, humans need to perform In-Context Identification (ICI). This means that the brain not only focuses on the overall structural content of the text52 but also parses the

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

- Figure 6: The four-stage framework of human thinking and LLM reasoning. The relationship between these four stages is not a linear progression but rather a graph-like transformation. Both humans and LLMs iteratively retrieve internal knowledge, observe the problem, and reason to arrive at the final answer.

syntactic53 and semantic54 information embedded within it.

Once the brain has acquired the aforementioned textual and memory information, it attempts to integrate this information to derive conclusions, a process known as Latent Reasoning (LR). This stage primarily includes arithmetic operations55 and logical inference56. Finally, the brain needs to translate the reasoning results into natural language, forming an answer that can be expressed verbally. This is the Expression Preparation (EP) stage. At this point, the brain bridges the gap between “knowing” and “saying”.57

As indicated by the arrows in Figure 6, these four stages are not executed in a strictly onedirection fashion when humans solve problems; rather, they can jump and switch between each other. For example, the brain may “cycle” through the identification of contextual content (the ICI stage) and then retrieve relevant knowledge based on the current context (the KR stage). Similarly, if latent reasoning cannot proceed further due to missing information, the brain may return to the Knowledge Recalling and In-Context Identification stages to gather more information.

We will now draw an analogy between these four steps and the mechanisms of attention heads, as depicted in Figure 7. Previous research has shown that LLMs possess strong contextual learning abilities and have many practical applications.58 As a result, much of the work on interpretability has focused on the ability of LLMs to capture and reason about contextual information. Consequently, the functions of currently known special attention heads are primarily concentrated in the ICI and LR stages, while fewer attention heads operate in the KR and EP stages.

###### 4.2 Knowledge Recalling (KR)

For LLMs, most knowledge is learned during the training or fine-tuning phases, which is embedded in the model’s parameters. This form of knowledge is often referred to as LLMs’ “parametric knowledge”. Similar to humans, certain attention heads in LLMs recall this internally stored knowledge—such as common sense or domain-specific expertise—to be used in subsequent reasoning. These heads typically retrieve knowledge by making initial guesses or associating based on specific content within the context, injecting the memory information into the residual stream as initial data or supplementary information. A brief summary of their functionalities is shown in Table 2.

General Tasks Associative Memories59,60, Memory Head61

[Figure 61]

[Figure 62]

§Knowledge Recalling (KR)

Constant Head (MCQA)62, Single Letter Head (MCQA)62, Negative Head (BDT)63

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Specific Tasks

Previous Head64,65, Positional Head66–68, Rare Words Head67, Duplicate Head22, Retrieval head69, Global Retrieval head70

[Figure 68]

Overall Structure

[Figure 69]

[Figure 70]

[Figure 71]

Subword Merge Head66,71, Syntactic Head67,72, Negative Name Mover Head22, Mover Head24, Name Mover Head73, Backup Name Mover Head73, Letter Mover Head74

§In-Context Identification (ICI)

Syntactic Information

SpecialAttentionHeads

Context Head61, Content Gatherer Head62,75, Sentiment Summarizer76, Semantic Induction Head77, Subject Head & Relation Head24,78

[Figure 72]

Semantic Information

[Figure 73]

[Figure 74]

[Figure 75]

In-context Learning Summary Reader76, Function Vector79, Induction Head64,80–85

[Figure 76]

[Figure 77]

[Figure 78]

Truthfulness Head86,87, Accuracy Head88,89, Consistency Head90, Vulnerable Head91

[Figure 79]

Effective Reasoning

§Latent Reasoning (LR)

Correct Letter Head (MCQA)62, Iteration Head (Sequence)92, Successor Head (Ordinal)93, Inhibition Head (Term)22,94

[Figure 80]

[Figure 81]

Task-Specific Reas.

Information Aggregation Mixed Head78

[Figure 82]

Signal Amplification Amplification Head62, Correct Head95

§Expression Preparation (EP)

[Figure 83]

Instruction Alignment Coherence Head88, Faithfulness Head96

- Figure 7: Taxonomy of special attention heads in language models. The icons before each head indicate the specific LLM architectures where the head was discovered. : InternLM series. : Yi series. : Gemma series. : Mistral series. : Llama series. : Qwen series. : Pythia series. : GPT series. : Toy models, such as two-layer decoder-only transformers.

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

In general tasks, Bietti et al.59 identified that certain attention heads can give rise to associative memories, progressively storing and retrieving knowledge during the model’s training phase. The weight matrices of these heads can be viewed as a weighted sum of the outer products of various vectors (e.g., input-output vectors or key-value vectors). Through their processing, these heads filter out noise from a superposed activation state while preserving essential features. Furthermore, as the embedding dimension d increases, they become more adept at refining relevant information and linking it to useful memories.60 The so-called Memory Head is capable of retrieving content related to the current problem from the model’s parametric knowledge.61 This retrieved content could be knowledge learned during pre-training or experience accumulated during previous reasoning processes. Specifically, shallow FFNs enrich the semantics of entities present in the problems. Based on this enriched information, the Memory Head recalls attributes associated with these entities and writes them back into the residual stream.

In specific task scenarios, such as when LLMs tackle Multiple Choice Question Answering (MCQA) problems, the answer is typically an option letter (e.g., A/B/C/D) rather than a short text. In these cases, they may initially use Constant Head to evenly distribute attention scores across all options. Alternatively, they might use Single Letter Head to assign a higher attention score to one option while giving lower scores to others, thereby capturing all potential answers.62 In addition, in the context of Binary Decision Tasks (BDT), which are problems where the solution space is discrete and contains only two options, such as yes-no questions or answer verification, Yu et al.63 found that LLMs often exhibit a negative bias when handling such tasks. This could be because the model has learned a significant amount of negative expressions related to similar tasks from prior knowledge during training. Consequently, when the model identifies a given text

as a binary task, a Negative Head may “preemptively” choose the negative answer due to this prior bias.

Table 2: Key Attention Heads in Knowledge Recalling (KR).

Head Name Input Feature Output Feature Layer Distribution

Memory Head User context & Intermediate results Relevant parametric knowledge injected Shallow / Middle Constant Head All options in multiple-choice tasks Uniformly distributed attention scores Middle Single Letter Head Answer options Focused attention on a single candidate Middle Negative Head Binary decision task context Bias attention scores toward negative expressions Middle

###### 4.3 In-Context Identification (ICI)

Understanding the in-context nature of a problem is one of the most critical processes to effectively address it. Just as humans read a problem statement and quickly pick up on various key pieces of information, some attention heads in LLMs also focus on these elements. Specifically, attention heads that operate during the ICI stage use their QK matrices to focus on and identify overall structural, syntactic, and semantic information within the in-context. This information is then written into the residual stream via OV matrices.

###### 4.3.1 Overall Structural Information Identification

Identifying the overall structural information within a context mainly involves LLMs attending to content in special positions or with unique occurrences in the text. Previous Head 64,65 and Positional Head 66–68 attend to the positional relationships within the token sequence. They capture the embedding information of the current token and the previous token. Rare Words Head focuses on tokens that appear with the lowest frequency, emphasizing rare or unique tokens.67 Duplicate Head excels at capturing repeated content within the context, giving more attention to tokens that appear multiple times.22

Besides, as LLMs can gradually handle long texts, this is also related to the “Needle-in-aHaystack” capability of attention heads. (Global) Retrieval Head can accurately locate specific tokens in long texts.69,70,97 These heads enable LLMs to achieve excellent reading and in-context retrieval capabilities.

###### 4.3.2 Syntactic Information Identification

For syntactic information identification, sentences primarily consist of subjects, predicates, objects, and clauses. Syntactic Head can distinctly identify and label nominal subjects, direct objects, adjectival modifiers, and adverbial modifiers. Some words in the original sentence may get split into multiple subwords because of the tokenizer (e.g., “happiness” might be split into “happi” and “ness”). The Subword Merge Head focuses on these subwords and merges them into one complete word.66,71

Additionally, Yao et al.24 proposed the Mover Head cluster, which can be considered as “argument parsers”. These heads often copy or transfer a sentence’s important information (such as the subject’s position) to the [END] position. The [END] position refers to the last token’s position in the sentence being decoded by the LLM, and many studies indicate that summarizing contextual information at this position facilitates subsequent reasoning and next-token prediction. Name Mover Head and Backup Name Mover Head move the names in the text to the [END] position. Letter Mover Head extracts the first letters of certain words in the context and aggregates

them at the [END] position.74 Conversely, Negative Name Mover Head prevents name information from being transferred to the [END] position.22,73

###### 4.3.3 Semantic Information Identification

As for semantic information identification, Context Head extracts information from the context that is related to the current task.61 Further, Content Gatherer Head “moves” tokens related to the correct answer to the [END] position, preparing to convert them into the corresponding option letter for output.62,75 The Sentiment Summarizer proposed by Tigges et al.76 can summarize adjectives and verbs that express sentiment in the context near the [SUM] position. The [SUM] position is located directly before the [END] position and enables subsequent heads to effectively read and reason.

Capturing the message about relationships is also important for future reasoning. Semantic Induction Head captures semantic relationships within sentences, such as part-whole, usage, and category-instance relationships.77 Subject Head and Relation Head focus on subject attributes and relation attributes, respectively, and then inject these attributes into the residual stream.78

###### 4.4 Latent Reasoning (LR)

The KR and ICI stages focus on gathering information, while Latent Reasoning (LR) is where all the collected information is synthesized and logical reasoning occurs. Whether in humans or LLMs, the LR stage is the core of problem-solving. Specifically, QK matrices of a head perform implicit reasoning based on information read from the residual stream, and then the reasoning results or signals are written back into the residual stream through OV matrices.

###### 4.4.1 In-context Learning

In-context Learning is one of the most widely discussed areas. Building on the work of Pan98, it primarily includes two types: Task Recognition (TR) and Task Learning (TL). Both aim to infer the solution based on the context; however, they differ fundamentally in their reliance on pre-trained priors. TR leverages the prior knowledge of LLMs to interpret demonstrations. For instance, sentiment classification tasks often involve labels with clear semantic meanings, such as “positive” and “negative”, which LLMs are likely to have internalized during pre-training. In contrast, TL requires the model to learn a novel mapping function between input-output pairs, where the examples and labels lack an inherent semantic connection.

For Task Recognition: Summary Reader can read the information summarized at the [SUM] position during the ICI stage and use this information to infer the corresponding sentiment label.76 Todd et al.79 proposed that the output of certain mid-layer attention heads can combine into a Function Vector. These heads abstract the core features and logical relationships of a task, based on the semantic information identified during ICI, and thereby trigger task execution.

For Task Learning, the essence of solving these problems is enabling LLMs to inductively find patterns. Induction Heads are among the most widely studied attention heads.64,80,81,83 They capture patterns such as “... [A][B] ... [A]” where token [B] follows token [A], and predict that the next token of this sequence should be [B]. Specifically, the Induction Head in the residual stream of the second [A] can access information from that of all preceding tokens. This mainly includes information about “what the previous token is” for each token, which is provided by the Previous Head. The Induction Head then matches this information with the information in the current residual stream, i.e., it matches the second [A] with the [A] preceding [B], to perform further reasoning.

Induction Head tends to strictly follow a pattern once identified and completes fill-in-the-blank reasoning. However, in most cases, the real problem will not be identical to the examples—just as a student’s exam paper will not be exactly the same as their homework. To address this, Yu and Ananiadou99 identified the In-context Head, whose QK matrix calculates the similarity between information at the [END] position and each label. The OV matrix then extracts label features and weights them according to the similarity scores to determine the final answer (take all labels into consideration rather than only one label).

###### 4.4.2 Effective Reasoning

Some studies have identified heads related to reasoning effectiveness. Truthfulness Head 86,87 and Accuracy Head 88,89 are heads highly correlated with the truthfulness and accuracy of answers. They help the model infer truthful and correct results in QA tasks, and modifying the model along their activation directions can enhance LLMs’ reasoning abilities. Similarly, the Consistency Head ensures the internal consistency of LLMs when asked the same question in different ways.90

However, not all heads positively impact reasoning. For example, Vulnerable Head is overly sensitive to certain specific input forms, making it susceptible to irrelevant information and leading to incorrect results.91 During reasoning, it is advisable to minimize the influence of such heads.

###### 4.4.3 Task Specific Reasoning

Finally, some heads are specialized for specific tasks. In MCQA tasks, Correct Letter Head can complete the matching between the answer text and option letters in order to determine the final answer choice.62 When dealing with tasks related to sequential data, Iteration Head can iteratively infer the next intermediate state based on the current state and input.92 For arithmetic problems, Successor Head can perform increment operations on ordinal numbers.93

In tasks such as syllogistic reasoning and information extraction, the Inhibition Head (also referred to as the Suppression Head) can aggregate outputs from other heads and suppress certain information. For example, it can suppress a subject or a middle term in order to reduce their associated logit values after unembedding.22,94

These examples illustrate how various attention heads specialize in different aspects of reasoning, contributing to the overall problem-solving capabilities of LLMs.

###### 4.5 Expression Preparation (EP)

Table 3: Key Attention Heads in Expression Preparation (EP).

Head Name Input Feature Output Feature Layer Distribution

Mixed Head Outputs of Subject & Relation Heads Integrated and concise final representation Deep Amplification Head Correct answer signals Amplified attention on correct tokens Deep Correct Head Hidden states of different options Focused attentions on final output tokens Deep Coherence Head Contextualized reasoning outputs Fluent and coherent text’s tokens Middle / deep Faithfulness Head Reasoning results and instructions Selected faithful contexts Deep

During the Expression Preparation (EP) stage, LLMs need to align their reasoning results with the content that needs to be expressed verbally. As shown in Table 3, EP heads may first aggregate information from various stages. Chughtai et al.78 proposed the Mixed Head, which can linearly combine and aggregate information passed along by heads from the ICI and LR stages (such as Subject Heads, Relation Heads, Induction Heads, etc.). The aggregated results

are then written back into the residual stream and ultimately mapped onto the vocabulary logits via the unembedding layer.

Some EP heads have a signal amplification function. Specifically, they read information about the context or reasoning results from the residual stream, then enhance the information that needs to be expressed as output, and write it back into the stream. Amplification Head 62 and Correct Head 95 amplify the signal of the correct choice letter in MCQA problems near the [END] position. This amplification ensures that after passing through the Unembedding layer and softmax calculation, the correct choice letter has the highest probability.

In addition to information aggregation and signal amplification, some EP heads are used to align the model’s reasoning results with the user’s instructions. In multilingual tasks, the model may sometimes fail to respond in the target language desired by the user. Coherence Head ensures linguistic consistency in the generated content.88 They help LLMs maintain consistency between the output language and the language of user’s query when dealing with multilingual inputs. Faithfulness Head is strongly associated with the faithfulness of Chain-of-Thought (CoT), which refers to whether the model’s generated response accurately reflects its internal reasoning process and behavior, i.e., the consistency between output and internal reasoning.96 Enhancing the activation of these heads allows LLMs to better align their internal reasoning with the output, making the CoT results more robust and consistent.

However, for some simple tasks, LLMs might not require special EP heads to refine language expression. In this situation, the information written back into the residual stream during the ICI and LR stages may be directly suitable for output, i.e., skip the EP stage and select the token with the highest probability.

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

- Figure 8: Diagram of the relationship between the stages where heads act and the layers they are in, as described from Knowledge Recalling (KR) to Expression Preparation (EP).

###### 4.6 How do attention heads work together?

As illustrated in Figure 8, if we divide the layers of a LLM (e.g., GPT-2 Small) into three segments based on their order—shallow (e.g., layers 1-4), middle (e.g., layers 5-8), and deep (e.g., layers 9-12)—we can map the relationship between the stages where heads act and the layers they are in, according to the content above. However, when combined with Figure 6, this pattern reflects

only the majority of cases; there are instances where LLMs return to the KR or ICI stage at deeper layers—for example, in the MCQA and IOI cases discussed below.

To gain a enhanced understanding of the relationships between these heads, researchers have investigated the potential semantic meanings embedded in the query vector qhℓ,j = Qhℓ[:,j] and key vector khℓ,j = Khℓ[:,j].62,75 For example, when solving a MCQA problem, the model first infers the correct answer in text form. It then needs to map this text to the corresponding option letter based on the list of choices. At this point, during the ICI stage, the Content Gatherer Head moves the tokens of the inferred answer text to the [END] position. Then, in the LR stage, the Correct Letter Head uses the information passed by the Content Gatherer Head to identify the correct option. The query vector in this context effectively asks, “Are you the correct label?” while recalling the gathered correct answer text. The key vector represents, “I’m choice [A/B/C/D], with the corresponding text [...]”. After matching the right key vector to the query vector, we can get the correct answer choice.

Consider the Parity Problem, which involves determining whether the sum of an input sequence

a1:t , consisting of only 0s and 1s, is odd or even. Let parity state sequence si denote the parity (odd or even) of the sum of the first i digits in the sequence, as defined in Equation 5. For example, given the input sequence a1:6 = 001011 , the corresponding parity state sequence is

s1:6 = eeooeo . When querying the LLM with the prompt “ a1:t [EOI] s1:t−1 [END]”, where [EOI] represents the End-Of-Input token, the expected response is the final parity state st.

Under these settings, during the ICI stage, a Mover Head transmits information from the [EOI] position to the [END] position. In the LR stage, an Iteration Head first reads the [EOI]’s position index from [END] and uses its query vector to ask, “Are you position t?” The key vector for each token responds, “I’m position t′.” This querying process identifies the last digit in the input sequence, which, combined with st−1, allows the model to calculate st.

 

i

ak mod 2 = 0,

e, if

k=1

(5)

si =

i

o, if

ak mod 2 = 1.



k=1

Further research has explored integrating multiple special attention heads into a cohesive working mechanism.22,75,94,100 Wang et al.22, Merullo et al.75, and Kim et al.94 have independently identified the collaborative mechanisms of attention heads, such as mover heads, induction heads, and inhibition heads, in different task scenarios, namely Object Identification and Syllogistic Reasoning. Their studies, all conducted on the GPT-2 model,6 have yielded remarkably similar conclusions regarding the information transfer patterns among several key attention heads. Here we take the IOI (Indirect Object Identification) task, which tests the model’s ability to deduce the indirect object in a sentence, as an example. Figure 9 outlines the main collaboration process.

- 1. In the KR stage, the Subject Head and the Relation Head focus on “Mary” and “bought flowers for”, respectively, triggering the model to recall that the answer should be a human name.78
- 2. Then in the ICI stage, the Duplicate Head identifies that “John” appears multiple times, while the Name Mover Head focuses on both “John” and “Mary” and moves them to the [END] position.
- 3. During the iterative stages of ICI and LR, the Previous Head and the Induction Head work together to attend to “John”. All this information is passed to the Inhibition Head, thereby suppressing the logits value of “John”.

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

- Figure 9: Schematic diagram of the collaborative mechanism of different attention heads in IOI task.22 Each oval represents a specific attention head, and the color indicates the depth of the layer where the head is located. These colors are aligned with those in Figure 4 and Figure 8.

- 4. Finally in the stage of EP, the Amplification Head boosts the logits value for “Mary”.

In summary, attention heads in LLMs work collaboratively across stages like KR, ICI, LR, and EP. This structured cooperation enables the model to solve complex tasks by effectively aligning and propagating relevant information through layers, further reflecting similarities between the working mechanisms of attention heads and the human brain.

- 5 Unveiling the discovery of attention heads

How can we uncover the specific functions of those special heads mentioned in Overview of special attention heads? In this section, we will unveil the discovery methods. Current research primarily employs experimental methods to validate the working mechanisms of those heads. We categorize the mainstream experimental approaches into two types based on whether they require the construction of new models: Modeling-Free and Modeling-Required. The classification scheme and method examples are shown in Figure 10.

###### 5.1 Modeling-Free

Table 4: Brief summarization of Modeling-Free methods.

Type Specific Method Core Operation Representative Works Modification-Based

Directional Addition Adding extra information to a specific component’s latent state Tigges et al.76, Yu et al.63, Turner et al.101 Directional Subtraction Subtracting part of information from a specific component’s latent state Tigges et al.76, Geiger et al.102

Wang et al.22, Yu and Ananiadou99, Jin et al.61, Yao et al.24, Mohebbi et al.103

Zero Ablation The component’s latent state is replace with zero vectors

McDougall et al.73, Wang et al.22, Kim et al.94, Hanna et al.104

The component’s latent state is replace with the mean state across all samples passing through it

Mean Ablation

Replacement-Based

Merullo et al.75, Todd et al.79, Wang et al.22, Lieberum et al.62, Wiegreffe et al.95

The component’s activation is replaced with corresponding activation run by a corrupted prompt

Na¨ıve Activation Patching

Modeling-Free methods do not require setting up new models, making them widely applicable in interpretability research. These methods typically involve altering a latent state computed during the LLMs’ reasoning process and then using Logit Lens to map the intermediate results to token

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

###### +/-

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

###### + +

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

+

[Figure 214]

[Figure 215]

⟹

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

| | |
|---|---|
| | |

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

| | |
|---|---|
| | |

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

|𝒟𝑟𝑖𝑔ℎ𝑡⋂𝒟𝑎𝑙𝑙| |𝒟𝑎𝑙𝑙|

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

RetrievalScoreℓℎ =

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

NASℓℎ = ෍

𝐴𝑡𝑡𝑛ℓℎ 𝑖,𝑡𝑌𝑒𝑠 + 𝐴𝑡𝑡𝑛ℓℎ 𝑖,𝑡𝑁𝑜

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

𝑖

+

[Figure 342]

[Figure 343]

𝐴𝑡𝑡𝑛ℓℎ 𝑖,𝑡𝑁𝑜 𝐴𝑡𝑡𝑛ℓℎ 𝑖,𝑡𝑌𝑒𝑠

[Figure 344]

[Figure 345]

[Figure 346]

⋅ 𝑙𝑜𝑔

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

……

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

- Figure 10: Pie chart of methods for exploring special attention heads and diagram of various methods.

logits or probabilities. By calculating the logit (or probability) difference, researchers can infer the impact of the change. Modeling-Free methods primarily include Activation Patching and Ablation Study. However, due to the frequent interchange of these terms in the literature, a new perspective is required to distinguish them. This paper further divides these methods into Modification-Based and Replacement-Based Methods based on how the latent state representation is altered, as summarized in Table 4.

Modification-Based methods involve altering the values of a specific latent state while retaining some of the original information, under the hypothesis that concepts are encoded as linear directions in the representation space.105 Directional Addition retains part of the information in the original state and then directionally adds some additional information.

For instance, Tigges et al.76 input texts containing positive and negative sentiments into LLMs, obtaining positive and negative representations from the latent state. The difference between these two representations can be seen as a sentiment direction in the latent space. By adding this sentiment direction vector to the activation of the attention head, the effect on the output can be analyzed to determine whether the head has the ability to summarize sentiment. Similarly, Ortu et al.100 explored the competitive relationships between different mechanisms. They directionally amplified the attention score of one token towards another, allowing the latent representation to include more information about that token.

Conversely, Directional Subtraction retains part of the original state information while directionally removing some of it.106 This method can be used to investigate whether removing specific information from a latent state affects the model’s output in a significant way, thereby revealing whether certain attention heads can back up or fix the deleted information.

Replacement-Based methods, in contrast to Modification-Based methods, discard all information in a specific latent state and replace it with other values. Zero Ablation and Mean Ablation replace the original latent state with zero values or the mean value of latent states across all samples from a dataset, respectively. This can logically “eliminate” the head or cause it to lose its special function, allowing researchers to assess its importance.

Naive Activation Patching is the traditional patching method. It involves using a latent state obtained from a corrupted prompt to replace the original latent state at the corresponding position. For example, consider the original prompt “John and Mary went to the store.” Replacing “Mary”

with “Alice” results in a corrupted prompt. By systematically replacing the latent state obtained under the original prompt with the one obtained under the corrupted prompt across each head, researchers can preliminarily determine which head has the ability to focus on names based on the magnitude of the impact.26,107 Alternatively, we can also replace the latent state obtained from the corrupted run with the original one. By doing so, we can observe how the head’s behavior shifts back towards the performance on the original prompt.

###### 5.2 Modeling-Required

Table 5: Brief summarization of Modeling-Required methods.

Type Specific Method Core Operation Representative Works

Li et al.86, Hoscilowicz et al.87, Gould et al.93, Guo et al.88, Yang et al.90, Jin et al.108

Train a classifier to distinguish heads with different functions using activation values

Probing

Training-Required

Edelman et al.80, Cabannes et al.92, Reddy84, Elhage et al.25

Train an approximate simplified model (e.g., a two-layer Transformer or an attention-only model)

Simplified Model Training

Jin et al.61, Wu et al.69, Crosbie83, Yu et al.63, Ji-An et al.82

Calculate the score that reflects the relationship between the component’s attributes and LLM features

Scoring

Training-Free

Others New methods that have not yet been widely adopted Ferrando and Voita66, Conmy et al.109

Modeling-Required methods involve explicitly constructing models to delve deeper into the functions of specific heads. Based on whether the newly constructed models require training, we further categorize Modeling-Required methods into Training-Required and Training-Free methods, as summarized in Table 5.

Training-Required methods necessitate training the newly established models to explore mechanisms. Probing is a common training-based method. This approach extracts activation values from different heads as features and categorizes heads into different classes as labels. A classifier is then trained on this data to learn the relationship between the activation patterns and the head’s function. Subsequently, the trained classifier can serve as a probe to detect which heads within the LLMs possess which functions.86,90

Another approach involves training a simplified transformer model on a clean dataset for a specific task. Researchers investigate whether the heads in this simplified model exhibit certain functionalities, which can then be extrapolated whether similar heads in the original model possess the same capabilities. This method reduces computational costs during training and analysis, while the constructed model remains simple and highly controllable.92

Training-Free methods primarily involve designing scores that reflect specific phenomena. These scores can be viewed as mathematical models that construct an intrinsic relationship between the attributes of components and certain model characteristics or behaviors. For instance, when investigating Retrieval Heads, Wu et al.69 defined a Retrieval Score. This score represents the frequency with which a head assigns the highest attention score to the token it aims to retrieve across a sample set, as shown in Equation 6. A high Retrieval Score indicates that the head possesses a strong “Needle in a Haystack” ability.

Similarly, when exploring Negative Heads, Yu et al.63 introduced the Negative Attention Score (NAS), as shown in Equation 7. Here, i denotes the i-th token in the input prompt, and tYes and tNo represent the positions of “Yes” and “No” in the prompt, respectively. A high NAS suggests that the head focuses more on negative tokens during decision-making, making it prone to generating negative signals.

RetrievalScorehℓ = |Dright ∩ Dall|

(6)

|Dall|

Attnhℓ [i,tNo] Attnhℓ [i,tYes]

NAShℓ =

(7)

Attnhℓ [i,tYes] + Attnhℓ [i,tNo] · log

i

In addition to scoring, researchers have proposed other novel training-free modeling methods. Ferrando and Voita66 introduced the concept of an Information Flow Graph (IFG), where nodes represent tokens and edges represent information transfer between tokens via attention heads or FFNs. By calculating and filtering the importance of each edge to the node it points to, key edges can be selected to form a subgraph. This subgraph can then be viewed as the primary internal mechanism through which LLMs perform reasoning.

###### 6 Evaluation

This section summarizes the benchmarks and datasets used in the interpretability research of attention heads. Based on the different evaluation goals during the mechanism exploration process, we categorize them into two types: Mechanism Exploration Evaluation and Common Evaluation. The former is designed to evaluate the working mechanisms of specific attention heads, while the latter assesses whether enhancing or suppressing the functions of certain special heads can improve the overall performance of LLMs.

Table 6: Selected benchmarks for mechanism exploration evaluation.

Benchmark Type Main Task Source Release Date

LRE110 Knowledge recalling Infer object entities given subject-entity prompts MIT 2023.09 ToyMovieReview76 Sentiment analysis Infer positive/negative sentiment EleutherAI 2023.10 ToyMoodStory76 Sentiment analysis Infer positive/negative sentiment EleutherAI 2023.10 FV-Caplitalize79 Token-level reasoning Infer the capital letter given some words NEU 2023.10 ICL-MC80 Token-level reasoning Infer next state based on in-context Harvard 2024.02 Succession93 Arithmetic reasoning Infer next number in a incremental sequence Cambridge 2023.12 Iteration-Synthetic92 Arithmetic reasoning Infer the next state of an iterative process Meta 2024.06 Omniglot111 Word-level reasoning Infer label from few samples NYU 2015.12 IOI22 Word-level reasoning Infer the indirect object UCB 2022.11 Colored Object75 Word-level reasoning Infer the correct color of a material Brown U 2023.10 World-Capital61 Word-level reasoning Infer the capital city given a country UCAS 2024.02

###### 6.1 Mechanism exploration evaluation

To delve deeper into the internal reasoning paths of LLMs, many researchers have synthesized new datasets based on existing benchmarks. The primary feature of these datasets is the simplification of problem difficulty, with elements unrelated to interpretability, such as problem length and query format, being standardized. As shown in Table 6, these datasets essentially evaluate the model’s knowledge reasoning and knowledge recalling capabilities, but they simplify the answers from a paragraph-level to a token-level.

Take exploring sentiment-related heads as an example, Tigges et al.76 created the ToyMovieReview and ToyMoodStory datasets, with specific prompt templates illustrated in Figure 11. Using these datasets, researchers employed sampling methods to calculate the activation differences of each head for positive and negative sentiments. This allowed them to recognize heads with significant differences as potential candidates for the role of Sentiment Summarizers.

###### 6.2 Common evaluation

The exploration of attention head mechanisms is ultimately aimed at improving the comprehensive capabilities of LLMs. Many researchers, upon identifying a head with a specific function, have attempted to modify that type of head—such as by enhancing or diminishing its activation—to

ToyMovieReview: I thought this movie was ADJECTIVE, I VERBed it. Conclusion: This movie is .

###### ToyMoodStory:

- NAME1 VERB1 parties, and VERB2 them whenever possible.
- NAME2 VERB3 parties, and VERB4 them whenever possible. One day, they were invited to a grand gala. [NAME1 or NAME2] feels very .

- Figure 11: Prompt template for ToyMovieReview and ToyMoodStory dataset. For example, ADJECTIVE could be “fantastic” / “horrible”, VERB could be “like” / “dislike”.

Table 7: Selected benchmarks for common evaluation.

Benchmark Type Main Task Source Release Date

MMLU112 Knowledge reasoning Solve problems with widespread knowledge UCB 2020.09 TruthfulQA113 Knowledge reasoning Answer questions that span 38 categories Oxford 2021.09 LogiQA114 Logic resoning Deduce the answer of logical problems FDU 2020.07 MQuAKE115 Logic resoning Deduce the answer via multi-hop reasoning Princeton 2023.05 SST/SST2116 Sentiment analysis Infer positive/negative sentiment Standford 2013.10 ETHOS117 Sentiment analysis Detect hate speech in online comments AUT 2020.06 Needle-in-a-Haystack Long context retrieval Retrieve content from long context Github 2023.11 AG News118 Text comprehension Infer the category of news NYU 2015.02 TriviaQA119 Text comprehension Answer questions based on documents UoW 2017.05 AGENDA120 Text comprehension Generate the abstract of a passage UoW 2019.04

observe whether the LLMs’ responses become more accurate and useful. We classify these Common Evaluation Benchmarks based on their evaluation focus, as shown in Table 7. The special attention heads discussed in this paper are closely related to improving LLMs’ abilities in five key areas: knowledge reasoning, logic reasoning, sentiment analysis, long context retrieval, and text comprehension.

###### 7 Additional topics

In this section, we summarize various works related to the LLMs interpretability. Although these works may not recognize new special heads as discussed in Overview of special attention heads, they delve into the underlying mechanisms of LLMs from other perspectives. We will elaborate on these studies under two categories: FFN Interpretability and Machine Psychology.

###### 7.1 FFN interpretability

As discussed in Background, apart from attention heads, FFNs also plays a significant role in the LLMs reasoning process. This section primarily summarizes research focused on the mechanisms of FFNs and the collaborative interactions between attention heads and FFNs.

One of the primary functions of FFNs is to store knowledge acquired during the pre-training phase. Dai et al.121 proposed that factual knowledge stored within the model is often concentrated in a few neurons of the MLP, reflecting the sparsity of the model.122 Geva et al.123 observed that the neurons in the FFN of GPT models can be likened to key-value pairs, where specific keys can retrieve corresponding values, i.e., knowledge. Lv et al.124 discovered a hierarchical storage of

knowledge within the model’s FFN, with lower layers storing syntactic and semantic information, and higher layers storing more concrete factual content.

FFNs effectively complement the capabilities of attention heads across the four stages described in Overview of special attention heads. The collaboration between FFNs and attention heads enhances the overall capabilities of LLMs. Geva et al.125 proposed that attention heads and FFNs can work together to enrich the representation of a subject and then extract its related attributes, thus facilitating factual information retrieval during the Knowledge Recall (KR) stage. Stolfo et al.126 found that, unlike attention heads, which focus on global information and perform aggregation, FFNs focus only on a single representation and perform local updates. This complementary functionality allows them to explore textual information both in breadth (attention heads) and depth (FFNs).

In summary, each component of LLMs plays a crucial role in the reasoning process. The individual contributions of these components, combined with their interactions, accomplish the entire process from Knowledge Recalling to Expression.

###### 7.2 Machine Psychology

Current research on the LLMs interpretability often draws parallels between the reasoning processes of these models and human thinking. This suggests the need for a more unified framework that connects LLMs with human cognition. The concept of Machine Psychology has emerged to fill this gap,127 exploring the cognitive activities of AI through psychological paradigms.

Recently, Hagendorff128 and Johansson et al.129 have proposed different approaches to studying machine psychology. Hagendorff’s work focuses on using psychological methods to identify new abilities in LLMs, such as heuristics and biases, social interactions, language understanding, and learning. His research suggests that LLMs display human-like cognitive patterns, which can be analyzed to improve AI interpretability and performance.

Johansson’s framework integrates principles of operant conditioning130 with AI systems, emphasizing adaptability and learning from environmental interactions. This approach aims to bridge gaps in AGI research by combining insights from psychology, cognitive science, and neuroscience.

Overall, Machine Psychology provides a new perspective for analyzing LLMs. Psychological experiments and behavioral analyses may lead to new discoveries about these models. As LLMs are increasingly applied across various domains of society, understanding their behavior through a psychological lens becomes increasingly important, which offers valuable insights for developing more intelligent AI systems.

###### 8 Discussion

###### 8.1 Limitations in existing studies

Although substantial progress has been made in uncovering the internal mechanisms of LLMs, several key limitations persist in existing research. These can be summarized as follows:

- • Lack of task generalizability. Current research primarily explores simple application scenarios that are limited to specific types of tasks. For example, Wang et al.22 and Merullo et al.75 have identified reasoning circuits in LLMs through tasks such as the IOI task and the Color Object Task. However, these circuits have not been validated across other tasks, making it challenging to determine whether these mechanisms are universally applicable.

- • Lack of Mechanism Transferability. As shown in Figure 7, many discovered special heads have only been explored within a few specific LLMs, or even on custom-built toy models. This raises a critical question: does a specialized head identified in one LLM exhibit the same functionality in another LLM? However, current research lacks investigations into the transferability of such mechanisms across different model series.
- • Limited focus on multi-head collaboration. Most studies investigate the mechanisms of individual attention heads, with only a few researchers studying the collaborative relationships among multiple heads. Consequently, existing work lacks a comprehensive framework for understanding the coordinated functioning of all attention heads in LLMs and analogizing the human brains.
- • Absence of theoretical supports. Many studies propose hypotheses about circuits based on observed phenomena and validate these hypotheses through experiments. However, this approach cannot establish the theoretical soundness of the mechanisms, nor can it determine whether the observed mechanisms are merely coincidental.

###### 8.2 Future directions and challenges

Building on the limitations discussed above and the content presented earlier, this paper outlines several potential research directions for the future:

- • Exploring mechanisms in more complex tasks. Investigate whether certain attention heads possess special functions in more complex tasks, such as open-ended question answering,131,132 math problems133,134 and tool-using tasks135.
- • Mechanism’s robustness against prompts. Research has shown that current LLMs are highly sensitive to prompts, with slight changes potentially leading to opposite responses.136 Future work could analyze this phenomenon through the lens of attention head mechanisms and propose solutions to mitigate this issue.
- • Developing new experimental methods. Explore new experimental approaches, such as designing experiments to verify whether a particular mechanism is indivisible or whether it has universal applicability.
- • Building a comprehensive interpretability framework. This framework should encompass both the independent and collaborative functioning mechanisms of most attention heads and other components.
- • Integrating Machine Psychology. Incorporate insights from Machine Psychology to construct an internal mechanism framework for LLMs from an anthropomorphic perspective, understanding the gaps between current LLMs and human cognition and guiding targeted improvements.

###### 9 Limitation

Current research on the interpretability of LLMs’ attention heads is relatively scattered, primarily focusing on the functions of individual heads, while lacking a rigorous definition of the overall framework. As a result, the categorization of attention head functions from the perspective of human cognitive behavior in this paper may not be perfectly orthogonal, potentially leading to some overlap between different stages.

###### Experimental procedures

###### Resource availability

###### Lead contact

Additional information, questions, and requests should be directed to the lead contact, Dr. Zhiyu Li (lizy@iaar.ac.cn).

Materials availability Not applicable, as no new unique reagents were generated.

###### Data and code availability

Our reference list is available at GitHub (https://github.com/IAAR-Shanghai/Awesome-AttentionHeads).

###### Author contributions

Conceptualization, Z.Z., Y.W. and S.S.; planning, Z.Z.; investigation: Z.Z. and Y.W.; original draft, Z.Z. and Y.H.; visualization, Z.Z. and Y.H.; review & editing, all authors; project administration, M.Y., B.T., F.X. and Z.L.; supervision, Z.L.

###### Declaration of interests

The authors declare no competing interests.

###### References

- 1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems 30.
- 2. Gilpin, L. H., Bau, D., Yuan, B. Z., Bajwa, A., Specter, M., and Kagal, L. (2018). Explaining explanations: An overview of interpretability of machine learning. In: 2018 IEEE 5th International Conference on data science and advanced analytics (DSAA). IEEE ( 80–89).
- 3. Lipton, Z. C. (2018). The mythos of model interpretability: In machine learning, the concept of interpretability is both important and slippery. Queue 16, 31–57.
- 4. Montavon, G., Samek, W., and M¨uller, K.-R. (2018). Methods for interpreting and understanding deep neural networks. Digital signal processing 73, 1–15.
- 5. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. Preprint at arXiv. https://doi.org/10.48550/arXiv.1810.04805.

- 6. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. et al. (2019). Language models are unsupervised multitask learners. OpenAI blog 1, 9.
- 7. Chuang, Y.-S., Qiu, L., Hsieh, C.-Y., Krishna, R., Kim, Y., and Glass, J. (2024). Lookback lens: Detecting and mitigating contextual hallucinations in large language models using only attention maps. Preprint at arXiv. https://doi.org/10.48550/arXiv.2407.07071.
- 8. Li, H., Chi, H., Liu, M., and Yang, W. (2024). Look within, why llms hallucinate: A causal perspective. Preprint at arXiv. https://doi.org/10.48550/arXiv.2407.10153.
- 9. Ji, Z., Chen, D., Ishii, E., Cahyawijaya, S., Bang, Y., Wilie, B., and Fung, P. (2024). Llm internal states reveal hallucination risk faced with a query. Preprint at arXiv. https://doi.org/10.48550/arXiv.2407.03282.
- 10. Kovaleva, O., Romanov, A., Rogers, A., and Rumshisky, A. (2019). Revealing the dark secrets of BERT. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics ( 4365–4374).
- 11. Wang, A., and Cho, K. (2019). BERT has a mouth, and it must speak: BERT as a Markov random field language model. In: Proceedings of the Workshop on Methods for Optimizing and Evaluating Neural Language Generation. Association for Computational Linguistics ( 30–36).
- 12. Pande, M., Budhraja, A., Nema, P., Kumar, P., and Khapra, M. M. (2021). The heads hypothesis: A unifying statistical approach towards understanding multi-headed attention in bert. In: Proceedings of the AAAI conference on artificial intelligence vol. 35. ( 13613– 13621).
- 13. Liu, L., Liu, X., Gao, J., Chen, W., and Han, J. (2020). Understanding the difficulty of training transformers. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics ( 5747–5763).
- 14. Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T. (2020). On layer normalization in the transformer architecture. In: International Conference on Machine Learning. PMLR ( 10524–10533).
- 15. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. (2024). Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063.
- 16. Shazeer, N. (2020). Glu variants improve transformer. Preprint at arXiv. https://doi.org/10.48550/arXiv.2002.05202.
- 17. Fedus, W., Zoph, B., and Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research 23, 1–39.
- 18. Cai, W., Jiang, J., Wang, F., Tang, J., Kim, S., and Huang, J. (2024). A survey on mixture of experts. Preprint at arXiv. https://doi.org/10.48550/arXiv.2407.06204.
- 19. Nanda, N. (2022). A comprehensive mechanistic interpretability explainer & glossary. Neel Nanda’s Blog 1.
- 20. Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., and Carter, S. (2020). Zoom in: An introduction to circuits. Distill 5, e00024–001.

- 21. Geiger, A., Lu, H., Icard, T., and Potts, C. (2021). Causal abstractions of neural networks. In: Advances in Neural Information Processing Systems vol. 34. Curran Associates, Inc. ( 9574–9586).
- 22. Wang, K. R., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J. (2023). Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In: The Eleventh International Conference on Learning Representations.
- 23. Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., and Shieber, S. (2020). Investigating gender bias in language models using causal mediation analysis. Advances in neural information processing systems 33, 12388–12401.
- 24. Yao, Y., Zhang, N., Xi, Z., Wang, M., Xu, Z., Deng, S., and Chen, H. (2024). Knowledge circuits in pretrained transformers. Preprint at arXiv. https://doi.org/10.48550/arXiv.2405.17969.
- 25. Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T. et al. (2021). A mathematical framework for transformer circuits. Transformer Circuits Thread. https://transformer-circuits.pub/2021/framework/index.html.
- 26. Heimersheim, S., and Nanda, N. (2024). How to use and interpret activation patching. Preprint at arXiv. https://doi.org/10.48550/arXiv.2404.15255.
- 27. Rauker,¨ T., Ho, A., Casper, S., and Hadfield-Menell, D. (2023). Toward transparent ai: A survey on interpreting the inner structures of deep neural networks. In: 2023 ieee conference on secure and trustworthy machine learning (satml). IEEE ( 464–483).
- 28. Gonc¸alves, T., Rio-Torto, I., Teixeira, L. F., and Cardoso, J. S. (2022). A survey on attention mechanisms for medical applications: are we moving toward better algorithms? IEEE Access 10, 98909–98935.
- 29. Santana, A., and Colombini, E. (2021). Neural attention models in deep learning: Survey and taxonomy. Preprint at arXiv. https://doi.org/10.48550/arXiv.2112.05909.
- 30. Chaudhari, S., Mithal, V., Polatkan, G., and Ramanath, R. (2021). An attentive survey of attention models. ACM Transactions on Intelligent Systems and Technology (TIST) 12, 1–32.
- 31. Brauwers, G., and Frasincar, F. (2021). A general survey on attention mechanisms in deep learning. IEEE Transactions on Knowledge and Data Engineering 35, 3279–3298.
- 32. Luo, H., and Specia, L. (2024). From understanding to utilization: A survey on explainability for large language models. Preprint at arXiv. https://doi.org/10.48550/arXiv.2401.12874.
- 33. Liang, X., Song, S., Zheng, Z., Wang, H., Yu, Q., Li, X., Li, R.-H., Xiong, F., and Li, Z. (2024). Internal consistency and self-feedback in large language models: A survey. Preprint at arXiv. https://doi.org/10.48550/arXiv.2407.14507.
- 34. Rouault, M., McWilliams, A., Allen, M. G., and Fleming, S. M. (2018). Human metacognition across domains: insights from individual differences and neuroimaging. Personality neuroscience 1, e17.
- 35. Dasgupta, I., Lampinen, A. K., Chan, S. C., Creswell, A., Kumaran, D., McClelland, J. L., and Hill, F. (2022). Language models show human-like content effects on reasoning. arXiv preprint arXiv:2207.07051 2.

- 36. Li, Y., Michaud, E. J., Baek, D. D., Engels, J., Sun, X., and Tegmark, M. (2024). The geometry of concepts: Sparse autoencoder feature structure. arXiv preprint arXiv:2410.19750.
- 37. Janik, R. A. (2023). Aspects of human memory and large language models. arXiv preprint arXiv:2311.03839.
- 38. Schrimpf, M., Blank, I. A., Tuckute, G., Kauf, C., Hosseini, E. A., Kanwisher, N., Tenenbaum, J. B., and Fedorenko, E. (2021). The neural architecture of language: Integrative modeling converges on predictive processing. Proceedings of the National Academy of Sciences 118, e2105646118.
- 39. Marjieh, R., Sucholutsky, I., van Rijn, P., Jacoby, N., and Griffiths, T. L. (2024). Large language models predict human sensory judgments across six modalities. Scientific Reports 14, 21445.
- 40. Mischler, G., Li, Y. A., Bickel, S., Mehta, A. D., and Mesgarani, N. (2024). Contextual feature extraction hierarchies converge in large language models and the brain. arXiv preprint arXiv:2401.17671.
- 41. Millidge, B., Seth, A., and Buckley, C. L. (2021). Predictive coding: a theoretical and experimental review. arXiv preprint arXiv:2107.12979.
- 42. Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. (2020). Scaling laws for neural language models. Preprint at arXiv. https://doi.org/10.48550/arXiv.2001.08361.
- 43. Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E. et al. (2021). On the opportunities and risks of foundation models. Preprint at arXiv. https://doi.org/10.48550/arXiv.2108.07258.
- 44. Wang, Y. (2007). The oar model of neural informatics for internal knowledge representation in the brain. International Journal of Cognitive Informatics and Natural Intelligence (IJCINI) 1, 66–77.
- 45. Wang, Y., and Chiew, V. (2010). On the cognitive process of human problem solving. Cognitive systems research 11, 81–92.
- 46. Anderson, J. R. Rules of the mind. Psychology Press (2014).
- 47. Whitehill, J. (2013). Understanding act-r-an outsider’s perspective. arXiv preprint arXiv:1306.0125.
- 48. Laird, J. E. (2022). An analysis and comparison of act-r and soar. arXiv preprint arXiv:2201.09305.
- 49. Squire, L. R. (1992). Memory and the hippocampus: a synthesis from findings with rats, monkeys, and humans. Psychological review 99, 195.
- 50. Tulving, E. (1972). “episodic and semantic memory,” in organization of memory. (No Title) ( 381).
- 51. Sartori, G., and Orr`u, G. (2023). Language models and psychological sciences. Frontiers in Psychology 14, 1279317.

- 52. Kintsch, W. (1988). The role of knowledge in discourse comprehension: a constructionintegration model. Psychological review 95, 163.
- 53. Chomsky, N. Aspects of the Theory of Syntax. 11 MIT press (2014).
- 54. Jackendoff, R. S. Semantic structures vol. 18. MIT press (1992).
- 55. Dehaene, S. The number sense: How the mind creates mathematics. OUP USA (2011).
- 56. Johnson-Laird, P. Mental models. Towards a cognitive science of language, inference, and consciousness. Harvard University Press (1983).
- 57. Levelt, W. J. (1999). Models of word production. Trends in cognitive sciences 3, 223–232.
- 58. Winata, G. I., Madotto, A., Lin, Z., Liu, R., Yosinski, J., and Fung, P. (2021). Language models are few-shot multilingual learners. In: Proceedings of the 1st Workshop on Multilingual Representation Learning. Association for Computational Linguistics ( 1–15).
- 59. Bietti, A., Cabannes, V., Bouchacourt, D., Jegou, H., and Bottou, L. (2024). Birth of a transformer: A memory viewpoint. Advances in Neural Information Processing Systems 36.
- 60. Dana, L., Pydi, M. S., and Chevaleyre, Y. (2024). Memorization in attention-only transformers. arXiv preprint arXiv:2411.10115.
- 61. Jin, Z., Cao, P., Yuan, H., Chen, Y., Xu, J., Li, H., Jiang, X., Liu, K., and Zhao, J. (2024). Cutting off the head ends the conflict: A mechanism for interpreting and mitigating knowledge conflicts in language models. In: Findings of the Association for Computational Linguistics ACL 2024. Association for Computational Linguistics ( 1193–1215).
- 62. Lieberum, T., Rahtz, M., Kramar,´ J., Irving, G., Shah, R., and Mikulik, V. (2023). Does circuit analysis interpretability scale? evidence from multiple choice capabilities in chinchilla. Preprint at arXiv. https://doi.org/10.48550/arXiv.2307.09458.
- 63. Yu, S., Song, J., Hwang, B., Kang, H., Cho, S., Choi, J., Joe, S., Lee, T., Gwon, Y. L., and Yoon, S. (2024). Correcting negative bias in large language models through negative attention score alignment. Preprint at arXiv. https://doi.org/10.48550/arXiv.2408.00137.
- 64. Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A. et al. (2022). In-context learning and induction heads. Preprint at arXiv. https://doi.org/10.48550/arXiv.2209.11895.
- 65. Nanda, N., Rajamanoharan, S., Kramar,´ J., and Shah, R. (2023). Fact finding: Attempting to reverse-engineer factual recall on the neuron level. In: AI Alignment Forum, 2023c. URL https://www. alignmentforum. org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-toreverse-engineer-factual-recall. ( 19).
- 66. Ferrando, J., and Voita, E. (2024). Information flow routes: Automatically interpreting language models at scale. Preprint at arXiv. https://doi.org/10.48550/arXiv.2403.00824.
- 67. Voita, E., Talbot, D., Moiseev, F., Sennrich, R., and Titov, I. (2019). Analyzing multi-head selfattention: Specialized heads do the heavy lifting, the rest can be pruned. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics ( 5797–5808).

- 68. Raganato, A., and Tiedemann, J. (2018). An analysis of encoder representations in transformer-based machine translation. In: Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP. Association for Computational Linguistics ( 287–297).
- 69. Wu, W., Wang, Y., Xiao, G., Peng, H., and Fu, Y. (2024). Retrieval head mechanistically explains long-context factuality. Preprint at arXiv. https://doi.org/10.48550/arXiv.2404.15574.
- 70. Tang, H., Lin, Y., Lin, J., Han, Q., Hong, S., Yao, Y., and Wang, G. (2024). Razorattention: Efficient kv cache compression through retrieval heads. Preprint at arXiv. https://doi.org/10.48550/arXiv.2407.15891.
- 71. Correia, G. M., Niculae, V., and Martins, A. F. T. (2019). Adaptively sparse transformers. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics ( 2174–2184).
- 72. Chen, A., Shwartz-Ziv, R., Cho, K., Leavitt, M. L., and Saphra, N. (2024). Sudden drops in the loss: Syntax acquisition, phase transitions, and simplicity bias in MLMs. In: The Twelfth International Conference on Learning Representations.
- 73. McDougall, C., Conmy, A., Rushing, C., McGrath, T., and Nanda, N. (2023). Copy suppression: Comprehensively understanding an attention head. Preprint at arXiv. https://doi.org/10.48550/arXiv.2310.04625.
- 74. Garc´ıa-Carrasco, J., Mate,´ A., and Trujillo, J. C. (2024). How does gpt-2 predict acronyms? extracting and understanding a circuit via mechanistic interpretability. In: International Conference on Artificial Intelligence and Statistics. PMLR ( 3322–3330).
- 75. Merullo, J., Eickhoff, C., and Pavlick, E. (2024). Circuit component reuse across tasks in transformer language models. In: The Twelfth International Conference on Learning Representations.
- 76. Tigges, C., Hollinsworth, O. J., Geiger, A., and Nanda, N. (2023). Linear representations of sentiment in large language models. Preprint at arXiv. https://doi.org/10.48550/arXiv.2310.15154.
- 77. Ren, J., Guo, Q., Yan, H., Liu, D., Zhang, Q., Qiu, X., and Lin, D. (2024). Identifying semantic induction heads to understand in-context learning. In: Findings of the Association for Computational Linguistics ACL 2024. Association for Computational Linguistics ( 6916– 6932).
- 78. Chughtai, B., Cooney, A., and Nanda, N. (2024). Summing up the facts: Additive mechanisms behind factual recall in llms. Preprint at arXiv. https://doi.org/10.48550/arXiv.2402.07321.
- 79. Todd, E., Li, M., Sharma, A. S., Mueller, A., Wallace, B. C., and Bau, D. (2024). Function vectors in large language models. In: The Twelfth International Conference on Learning Representations.
- 80. Edelman, B. L., Edelman, E., Goel, S., Malach, E., and Tsilivis, N. (2024). The evolution of statistical induction heads: In-context learning markov chains. Preprint at arXiv. https://doi.org/10.48550/arXiv.2402.11004.

- 81. Singh, A. K., Moskovitz, T., Hill, F., Chan, S. C., and Saxe, A. M. (2024). What needs to go right for an induction head? a mechanistic study of in-context learning circuits and their formation. In: Forty-first International Conference on Machine Learning.
- 82. Ji-An, L., Zhou, C. Y., Benna, M. K., and Mattar, M. G. (2024). Linking incontext learning in transformers to human episodic memory. Preprint at arXiv. https://doi.org/10.48550/arXiv.2405.14992.
- 83. Crosbie, J. (2024). Induction heads as an essential mechanism for pattern matching in in-context learning. Preprint at arXiv. https://doi.org/10.48550/arXiv.2407.07011.
- 84. Reddy, G. (2024). The mechanistic basis of data dependence and abrupt learning in an in-context classification task. In: The Twelfth International Conference on Learning Representations.
- 85. Aky¨urek, E., Wang, B., Kim, Y., and Andreas, J. (2024). In-context language learning: Architectures and algorithms. In: Forty-first International Conference on Machine Learning.
- 86. Li, K., Patel, O., Viegas,´ F., Pfister, H., and Wattenberg, M. (2024). Inference-time intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems 36.
- 87. Hoscilowicz, J., Wiacek, A., Chojnacki, J., Cieslak, A., Michon, L., Urbanevych, V., and Janicki, A. (2024). Nl-iti: Optimizing probing and intervention for improvement of iti method. Preprint at arXiv. https://doi.org/10.48550/arXiv.2403.18680.
- 88. Guo, P., Ren, Y., Hu, Y., Cao, Y., Li, Y., and Huang, H. (2024). Steering large language models for cross-lingual information retrieval. In: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. ( 585–596).
- 89. Yin, F., Ye, X., and Durrett, G. (Preprint at arXiv). Lofit: Localized fine-tuning on llm representations. arXiv preprint arXiv:2406.01563. https://doi.org/10.48550/arXiv.2406.01563.
- 90. Yang, J., Chen, D., Sun, Y., Li, R., Feng, Z., and Peng, W. (2024). Enhancing semantic consistency of large language models through model editing: An interpretability-oriented approach. In: Findings of the Association for Computational Linguistics ACL 2024. Association for Computational Linguistics ( 3343–3353).
- 91. Garc´ıa-Carrasco, J., Mate,´ A., and Trujillo, J. (2024). Detecting and understanding vulnerabilities in language models via mechanistic interpretability. In: Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24. International Joint Conferences on Artificial Intelligence Organization ( 385–393).
- 92. Cabannes, V., Arnal, C., Bouaziz, W., Yang, X. A., Charton, F., and Kempe, J. (2024). Iteration head: A mechanistic study of chain-of-thought. In: ICML 2024 Workshop on Mechanistic Interpretability.
- 93. Gould, R., Ong, E., Ogden, G., and Conmy, A. (2024). Successor heads: Recurring, interpretable attention heads in the wild. In: The Twelfth International Conference on Learning Representations.
- 94. Kim, G., Valentino, M., and Freitas, A. (2024). A mechanistic interpretation of syllogistic reasoning in auto-regressive language models. Preprint at arXiv. https://doi.org/10.48550/arXiv.2408.08590.

- 95. Wiegreffe, S., Tafjord, O., Belinkov, Y., Hajishirzi, H., and Sabharwal, A. (2024). Answer, assemble, ace: Understanding how transformers answer multiple choice questions. Preprint at arXiv. https://doi.org/10.48550/arXiv.2407.15018.
- 96. Tanneru, S. H., Ley, D., Agarwal, C., and Lakkaraju, H. (2024). On the difficulty of faithful chain-of-thought reasoning in large language models. In: Trustworthy Multi-modal Foundation Models and AI Agents (TiFA).
- 97. Fu, T., Huang, H., Ning, X., Zhang, G., Chen, B., Wu, T., Wang, H., Huang, Z., Li, S., Yan, S. et al. (2024). Moa: Mixture of sparse attention for automatic large language model compression. Preprint at arXiv. https://doi.org/10.48550/arXiv.2406.14909.
- 98. Pan, J. What in-context learning “learns” in-context: Disentangling task recognition and task learning. Master’s thesis Princeton University (2023).
- 99. Yu, Z., and Ananiadou, S. (2024). How do large language models learn in-context? query and key matrices of in-context heads are two towers for metric learning. Preprint at arXiv. https://doi.org/10.48550/arXiv.2402.02872.
- 100. Ortu, F., Jin, Z., Doimo, D., Sachan, M., Cazzaniga, A., and Scholkopf,¨ B. (2024). Competition of mechanisms: Tracing how language models handle facts and counterfactuals. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics ( 8420–8436).
- 101. Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., and MacDiarmid, M. (2023). Activation addition: Steering language models without optimization. Preprint at arXiv. https://doi.org/10.48550/arXiv.2308.10248.
- 102. Geiger, A., Wu, Z., Potts, C., Icard, T., and Goodman, N. (2024). Finding alignments between interpretable causal variables and distributed neural representations. In: Causal Learning and Reasoning. PMLR ( 160–187).
- 103. Mohebbi, H., Zuidema, W., Chrupała, G., and Alishahi, A. (2023). Quantifying context mixing in transformers. In: Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics. Association for Computational Linguistics ( 3378–3400).
- 104. Hanna, M., Liu, O., and Variengien, A. (2023). How does gpt-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model. In: Advances in Neural Information Processing Systems vol. 36. Curran Associates, Inc. ( 76033–76060).
- 105. Park, K., Choe, Y. J., and Veitch, V. (2024). The linear representation hypothesis and the geometry of large language models. In: Proceedings of the 41st International Conference on Machine Learning vol. 235 of Proceedings of Machine Learning Research. PMLR ( 39643–39666).
- 106. Liang, X., Wang, H., Wang, Y., Song, S., Yang, J., Niu, S., Hu, J., Liu, D., Yao, S., Xiong, F. et al. (2024). Controllable text generation for large language models: A survey. Preprint at arXiv. https://doi.org/10.48550/arXiv.2408.12599.
- 107. Zhang, F., and Nanda, N. (2024). Towards best practices of activation patching in language models: Metrics and methods. In: The Twelfth International Conference on Learning Representations.

- 108. Jin, M., Yu, Q., Huang, J., Zeng, Q., Wang, Z., Hua, W., Zhao, H., Mei, K., Meng, Y., Ding, K. et al. (2024). Exploring concept depth: How large language models acquire knowledge at different layers? Preprint at arXiv. https://doi.org/10.48550/arXiv.2404.07066.
- 109. Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., and Garriga-Alonso, A. (2023). Towards automated circuit discovery for mechanistic interpretability. In: Advances in Neural Information Processing Systems vol. 36. Curran Associates, Inc. ( 16318–16352).
- 110. Hernandez, E., Sharma, A. S., Haklay, T., Meng, K., Wattenberg, M., Andreas, J., Belinkov, Y., and Bau, D. (2024). Linearity of relation decoding in transformer language models. In: Proceedings of the 2024 International Conference on Learning Representations.
- 111. Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction. Science 350, 1332–1338.
- 112. Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. (2021). Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR).
- 113. Lin, S., Hilton, J., and Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics ( 3214– 3252).
- 114. Liu, J., Cui, L., Liu, H., Huang, D., Wang, Y., and Zhang, Y. (2020). Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. In: Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20. International Joint Conferences on Artificial Intelligence Organization ( 3622–3628).
- 115. Zhong, Z., Wu, Z., Manning, C., Potts, C., and Chen, D. (2023). MQuAKE: Assessing knowledge editing in language models via multi-hop questions. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics ( 15686–15702).
- 116. Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., and Potts, C.

(2013). Recursive deep models for semantic compositionality over a sentiment treebank. In: Proceedings of the 2013 conference on empirical methods in natural language processing. ( 1631–1642).

- 117. Mollas, I., Chrysopoulou, Z., Karlos, S., and Tsoumakas, G. (2020). Ethos: an online hate speech detection dataset. Preprint at arXiv. https://doi.org/10.1007/s40747-021-00608-2.
- 118. Zhang, X., Zhao, J., and LeCun, Y. (2015). Character-level convolutional networks for text classification. Advances in neural information processing systems 28.
- 119. Joshi, M., Choi, E., Weld, D., and Zettlemoyer, L. (2017). TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In: Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics ( 1601–1611).
- 120. Koncel-Kedziorski, R., Bekal, D., Luan, Y., Lapata, M., and Hajishirzi, H. (2019). Text Generation from Knowledge Graphs with Graph Transformers. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). Association for Computational Linguistics ( 2284–2293).

- 121. Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., and Wei, F. (2022). Knowledge neurons in pretrained transformers. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics ( 8493–8502).
- 122. Voita, E., Ferrando, J., and Nalmpantis, C. (2024). Neurons in large language models: Dead, n-gram, positional. In: Findings of the Association for Computational Linguistics ACL

2024. Association for Computational Linguistics ( 1288–1301).

- 123. Geva, M., Schuster, R., Berant, J., and Levy, O. (2021). Transformer feed-forward layers are key-value memories. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics ( 5484–5495).
- 124. Lv, A., Zhang, K., Chen, Y., Wang, Y., Liu, L., Wen, J.-R., Xie, J., and Yan, R. (2024). Interpreting key mechanisms of factual recall in transformer-based language models. Preprint at arXiv. https://doi.org/10.48550/arXiv.2403.19521.
- 125. Geva, M., Bastings, J., Filippova, K., and Globerson, A. (2023). Dissecting recall of factual associations in auto-regressive language models. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics ( 12216–12235).
- 126. Stolfo, A., Belinkov, Y., and Sachan, M. (2023). A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics ( 7035–7052).
- 127. Krichmar, J. L., and Edelman, G. M. (2002). Machine psychology: autonomous behavior, perceptual categorization and conditioning in a brain-based device. Cerebral Cortex 12, 818–830.
- 128. Hagendorff, T. (2023). Machine psychology: Investigating emergent capabilities and behavior in large language models using psychological methods. Preprint at arXiv. https://doi.org/10.48550/arXiv.2303.13988.
- 129. Johansson, R., Hammer, P., and Lofthouse, T. (2024). Functional equivalence with nars. Preprint at arXiv. https://doi.org/10.48550/arXiv.2405.03340.
- 130. Staddon, J. E., and Cerutti, D. T. (2003). Operant conditioning. Annual review of psychology 54, 115–144.
- 131. Narayan, S., Cohen, S. B., and Lapata, M. (2018). Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. arXiv preprint arXiv:1808.08745.
- 132. Li, M., Chen, M.-B., Tang, B., ShengbinHou, S., Wang, P., Deng, H., Li, Z., Xiong, F., Mao,

- K., Peng, C., and Luo, Y. (2024). NewsBench: A systematic evaluation framework for assessing editorial capabilities of large language models in Chinese journalism. In: Ku,
- L.-W., Martins, A., and Srikumar, V., eds. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Bangkok, Thailand: Association for Computational Linguistics ( 9993–10014). URL: https://aclanthology. org/2024.acl-long.538. doi:doi:10.18653/v1/2024.acl-long.538.

- 133. Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. (2021). Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.
- 134. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R. et al. (2021). Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.
- 135. Chen, Z., Du, W., Zhang, W., Liu, K., Liu, J., Zheng, M., Zhuo, J., Zhang, S., Lin, D., Chen, K., and Zhao, F. (2024). T-eval: Evaluating the tool utilization capability of large language models step by step. In: Ku, L.-W., Martins, A., and Srikumar, V., eds. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Bangkok, Thailand: Association for Computational Linguistics ( 9510–9529). URL: https://aclanthology.org/2024.acl-long.515. doi:doi:10.18653/v1/2024.acl-long.515.
- 136. Yu, Q., Zheng, Z., Song, S., Li, Z., Xiong, F., Tang, B., and Chen, D. (2024). xfinder: Robust and pinpoint answer extraction for large language models. Preprint at arXiv. https://doi.org/10.48550/arXiv.2405.11874.

