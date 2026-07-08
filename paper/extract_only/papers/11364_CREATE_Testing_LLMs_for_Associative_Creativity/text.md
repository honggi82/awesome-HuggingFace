# arXiv:2603.09970v2[cs.CL]11May2026

[Figure 1]

## CREATE: Testing LLMs for Associative Creativity

#### Manya Wadhwa∗

New York University

Tiasa Singha Roy New York University

Harvey Lederman The University of Texas at Austin

Junyi Jessy Li The University of Texas at Austin

Greg Durrett New York University

[Figure 2]

Project Page Repository Dataset

[Figure 3]

### Abstract

A key component of creativity is associative reasoning: the ability to draw novel yet meaningful connections between concepts. We introduce CREATE, a benchmark designed to evaluate models’ capacity for creative associative reasoning. CREATE requires models to generate sets of paths connecting concepts in a model’s parametric knowledge. Paths should have high specificity (distinctiveness and closeness of the concept connection) and high diversity (dissimilarity from other paths), and models are scored more highly if they produce a larger set of strong, diverse paths. This task shares demands of real creativity tasks like hypothesis generation, including an extremely large search space, but enables collection of a sizable benchmark with objective answer grading. Evaluation of frontier models shows that the strongest models achieve higher creative utility than others, but high multiplicity of answers and complexity of the search makes benchmark saturation difficult to achieve. Furthermore, our results illustrate that thinking models are not always more effective on our task, even with high token budgets. Recent approaches for “creative prompting” give some but limited additional improvement. CREATE provides a sandbox for developing new methods to improve models’ capacity for associative creativity.

### 1 Introduction

Creativity, as one of the three pillars in Sternberg’s Triarchic theory of intelligence [37] and the highest cognitive skill in Bloom’s Taxonomy [3], is central to scientific discovery, writing, and creative problem solving. A flurry of recent work aims to develop AI agents for these tasks; for example, for hypothesis generation [19], research idea generation [10], and the entire scientific process [20, 22, 24, 1, 35]. But how do we know if our models are creative enough? Real-world, complex queries are challenging and subjective to evaluate [5, 41], while symbolic benchmarks using abstract tasks [25, 33] may not reflect how the models are used in reality.

This work presents a new benchmark for associative creativity, striking a balance between real-world applicability and verifiability. We take motivation from combinatorial creativity [4], which requires the formation of new thoughts by linking, recombining, or drawing analogies across familiar concepts. This process closely aligns with what we term associative creativity and plays a foundational role in creative reasoning. Prior research, from Koestler’s bisociation [13] to Gentner’s structure-mapping

∗Corresponding Author: mw4141@nyu.edu

Preprint.

She starred alongside Chris Evans in The Materialists, and Chris Evans is in Captain America. But also, she’s Antonio Banderas’s stepdaughter, and Antonio Banderas starred in Shrek.

###### CREATE

The Material starred ists

Chris Evans starred

Captain America

| | |
|---|---|
| | |

Question with many possible answers to test associate reasoning:

starred in

- Agent 1
- Agent 2

Judge

in

in

Dakota Johnson

| | |
|---|---|
| | |

strong connections

Creativeutility

Antonio

| | |
|---|---|
| | |

Shrek 2 of different types

step Banderas daughter

starred How is Dakota in Johnson connected to people who starred in fantasy/sci-ﬁ movies?

Agent 1 Agent 2

She is a US citizen and Elijah Wood who starred in The Lord Of the Rings is also a US citizen.

Lord of the Rings

United States

Elijah Wood starred

Dakota Johnson citizen

citizen

in

weak connection

Figure 1: Motivating example of brainstorming paths in knowledge graphs. In CREATE, only the question is given; reasoning over the graph is implicit in the model’s parameters and thinking trace, similar to drawing connections for scientific research. Agents are rewarded in our creative utility metric for finding more strong, distinct paths.

theory [8], demonstrates that many creative insights originate in the associative leaps characteristic of combinatorial creativity. Our work focuses on associations between concepts: being able to surface nonobvious but striking connections between known concepts is crucial for new discoveries.

We present CREATE, a benchmark designed to evaluate LLMs’ ability to form open-ended associations among real-world entities. Figure 1 shows an example of a query designed to test this kind of reasoning. CREATE is practical, as it relies on world knowledge about concepts and relations rather than synthetic scenarios; it also sits at the right level of complexity: it is simple enough to evaluate yet still requires multi-hop reasoning about real concepts.

The queries in CREATE are constructed to span different conceptual domains: people in movies, people in positions of power, genes associated with certain diseases etc. By formulating questions in a sufficiently broad manner, we can test whether LLMs can enumerate the creative, “trivia-like” connections that a human domain expert might be aware of as salient and interesting. Figure 1’s possible answers show diverse and interesting possible connections. Our task involves reasoning about a more open-ended answer space compared to past work on multi-hop reasoning like HotpotQA [44] and similar datasets [40, 11, 6]. However, CREATE has both objective and hard-to-find answers, unlike more “brainstorming”-oriented tasks [46]. The ability to enumerate large numbers of distinct, strong connections directly ties to the ability to do strong brainstorming in other ideation settings.

Our central goal is for LLMs to generate connections that (a) reflect close but non-typical associations between concepts and (b) are diverse rather than simple variations on a single theme. These criteria align with standard desiderata for creative products: high quality coupled with meaningful novelty. Following prior work [46], we capture both aspects with a unified creative utility metric that integrates quality and diversity.

Our results show that frontier models achieve higher creativity utility scores compared to weaker models or open-source models. However, when we zoom in on particular cases, we find that no single model adequately covers the space of strong answer paths. Our analysis also shows that spending more reasoning tokens does not necessarily lead to higher scores on our benchmark, raising questions about how current AI systems search conceptual spaces and what is required for them to genuinely support or participate in creative processes.

### 2 Background

Tests for human creativity and diversity Human creativity is often assessed using tasks such as the Alternative Uses Task [9], Divergent Association Task [27], and Remote Associates Test [23]. While these are demanding for humans, taxing working memory and requiring the coordination of multiple concepts under cognitive constraints, they are comparatively easy for LLMs [42], which can exploit large context windows, retrieval tools, and parallel generation. In addition, standard test instances are often seen during pre-training, unlike for human participants. We show the saturation of these benchmarks more in Appendix K. By contrast, our benchmark requires forming novel connections among real-world entities (Sebastian Vettel, DSG3 ) rather than familiar everyday objects (e.g., book, bottle, brick) that are commonly used in the Alternative Uses Task.

Tests of model creativity Prior work proposes benchmarks [32, 21, 30, 7, 10, 19] focusing on specific creative abilities like creative writing, scientific ideation and code generation. However, evaluation in these specialized domains is challenging where the use of LLM-as-a-judge for studies at scale can conflate quality, diversity and usefulness [34]. It is unclear how to scale human evaluation: even AI-created papers undergoing real review processes are later found to have major flaws [18]. In contrast, Nagarajan et al. [25] and Schapiro et al. [33] introduce symbolic, highly controlled testbeds for probing combinatorial creativity and training models on abstract tasks. However, these settings are far removed from the complex, open-ended ways LLMs are used in real applications. Our work achieves the best of both worlds: we propose a realistic, knowledge-grounded task involving real-world entities and concepts. By leveraging knowledge graphs and familiar domains, we expect LLM-as-a-judge evaluations to be comparatively reliable.

Models are also criticized for returning homogeneous outputs [47, 12]. We do not directly engage with “distributional pluralism” [36, 16, 43] in this work; however; it indicates that LLMs may face a performance ceiling on our benchmark, even if ensembled together, and demonstrates why current LLMs may not be creative enough in practice [2, 29].

### 3 Conceptual Framework

#### 3.1 Defining and measuring associative creativity

Define U(x) as a set of possible outputs for some task x. For instance, if x is a mathematical theorem, u ∈ U(x) is a proof, if x is a writing prompt, then u is a story, etc.

We assume an LLM can produce a set U ⊂ U(x) through some sampling procedure, which we denote via U ∼ πLLM(U | p(x)) for a prompt p conditioned on the task. A key feature of our tasks is that |U| is likely to be of moderate size at most (10-100) with standard use of LLMs; an LLM must spend significant compute to identify high-quality items u, so enumerating large numbers of possible u is challenging.

We evaluate on two dimensions. First, a quality measure f : u ∈ U → R assigns a score to each item u. Second, a distance measure d : U × U → R≥0 defines distances between elements of U.2 We can combine these objectives into a single metric following NoveltyBench [46]. We define the creative utility of a set U as

s(U) = max

τ

|U|

γi−1f uτ(i) min j<i

d uτ(i),uτ(j) , (1)

i=1

where τ denotes an ordering over U. Items are scored via their incremental marginal utility under this ordering, which is a function of how many items have already been selected (γ), their distance to previously selected items (d), and their quality f. The first item is assumed to have a distance of 1 from the empty set.

γ defines a user “patience.” Values close to 1 favor larger sets, which we believe reflects how LLMs would be used for challenging creative tasks in practice: a user could scan through a moderately-sized list of ideas.3

We assume that LLMs return unordered sets U (e.g., through repeated sampling), but s(U) requires an ordering to be computed. We compute a greedily optimal τi by finding the element uk that maximizes

) (2)

f(uk)min j<i

d(uk,uτ

j

Associative creativity on graphs This work focuses on associative creativity, which instantiates U as paths in graphs. At a high level, many creative insights come from linking concepts through structured relationships. For example, a drug and a disease may be connected through shared

- 2We note that these criteria are explored in the theory of submodular functions [17] and in techniques like determinantal point processes [15]
- 3This objective differs from that of NoveltyBench in two key ways. First, their notion of item redundancy is discrete 0-1

rather than continuous, which is not appropriate for our setting. Second, we eliminate their normalization term 1−1−γ|γU| ; γ already discounts additional items and this terms tends to strongly penalize returning larger sets.

biological pathways or intermediate processes, rather than by a single direct link. This abstraction provides a simple way to study the associative part of combinatorial creativity. Concepts or entities in our graphs are nodes and relationships between them are edges. From this view, generating a creative association means finding a high-quality (strong) connection, and ideally one that is non-obvious.

One distinguishing feature of this setting is that there are potentially large numbers of valid paths. However, quality is right-tailed: that is, there are elements of U that are substantially more valuable than others, but which are hard to find. We can think of these as works of great literature, or great creative ideas, amidst a large number of more conventional (but still well-executed) concepts. This differs from NoveltyBench, where many responses to a prompt (What’s the best car to get in 2023?) may be equivalent in quality.

- 3.2 CREATE Formalization How do we connect Kareem Abdul-Jabbar to someone who is a member of the American Academy of Arts and Sciences (AAAS)?

We now concretize our framework in the setting of reasoning about graph connections. Let G = (E,R) be a knowledge graph with entities E and relations R. Paths u are represented by sequences of triples u =

min

people educated at UCLA

(~1M,speciﬁcity1) chancellors(~11,speciﬁcityofUCLA4) Speciﬁcity: 1

educated at

chancellor

member

Path2Path1

Kareem AbdulJabbar

Gene D. Block

AAAS

UCLA

Distance: 0.7

min

(e1,r1,e2),...,(en,rn,en+1) where each triple (ei,ri,ei+1) consists of two entities ei,ei+1 ∈ E connected by a relation ri ∈ R. We say that a path is structurally valid if consecutive triples share entities, i.e., the second entity of (ei,ri,ei+1) equals the first entity of (ei+1,ri+1,ei+2) (Note that LLMs may produce structurally invalid paths in their generations). We say that a path is factual if the triples represent true relations.

Medal of Freedom awardees (~600, speciﬁcity 2)

Medal of Freedom awardees

(~600,speciﬁcity2) Speciﬁcity: 2

award received Presidenti

awarded to Toni

member

Kareem AbdulJabbar

al Medal Freedom

AAAS

Morrison

Figure 2: Scoring generated paths. For specificity, we prompt an LLM as a judge to give an estimated class size, which is mapped to a number between 1-5. Triples belong to a large class size get a low score. The specificity of a path is the specificity of its weakest triple. Paths are embedded and compared to compute distance

The universe U(x) consists of all structurally valid paths in G connecting entities ei,ek ∈ E and obeying the constraints of the prompt x. We evaluate factuality separately.

Our prompts x go beyond simple queries and instead ask about sets, as shown in Figure 1. We only consider structurally valid paths that satisfy the prompt given by x; in Figure 1, whether a path successfully connects Dakota Johnson with a fantasy or sci-fi movie.

Quality of Paths We say that paths are high-quality if they are factual and if they are specific.

Specificity measures how many entities could plausibly participate in a given connection. Informally, we would think of Dakota Johnson as more strongly connected to Antonio Banderas than she is to other actors (Fig. 1); she has fewer step-fathers than co-stars. Being connected by being citizens of the same country is a weak or a less specific connection, unless it were a particularly small country.

This intuition suggests that what governs the specificity of a relationship is the number of entities that typically participate in it on a given side. More precisely, since paths are composed of triples, we define the quality of a path as the specificity of its weakest triple.

For an individual triple (ei,ri,ei+1), we define two predicate-induced classes:

CA(ei,ri,ei+1) = {x | (x,ri,ei+1) is true},CB(ei,ri,ei+1) = {y | (ei,ri,y) is true}. (3) The specificity of a triple is then defined as a function of the larger of these two classes:

σ(ei, ri, ei+1) = g max |CA(ei, ri, ei+1)|, |CB(ei, ri, ei+1)| . (4)

where g(·) is a monotonically decreasing function, so that triples participating in large, non-selective classes receive lower specificity scores. For example, in Figure 2, specificity scores for the triples in the first path are [1,4] and thus the score for the path is 1.

Factuality ensures that relations generated in u exist in the graph. Let q(u) denote a factuality function for path u, where q(u) = 1 if all triples in u represent true relations.

The quality f(u) of a path u (from Section 3) is defined as

σ(ei,ri,ei+1). (5)

f(u) = I[q(u) = 1] min

(ei,ri,ei+1)∈u

Intuitively, a path is only as strong as its weakest relation. This formulation favors paths composed of selective, informative relations and penalizes those with highly generic edges.

Both specificity and factuality are implemented with LLM judges described in Section 4. These judges are evaluated for quality in Section 6.

Distance function We define the following distance function to be used in the measures in Section 3.1. Let a path u be represented by its string form str(u). We define the distance between two paths as a transformation function of the cosine distance between their string representations.

d(ui,uj) = g(1 − cos(str(ui),str(uj))), (6) where, g(x) is a cosine-annealed cosine distance

- 1

- 2 1 − cos π(0x.7)2 0 ≤ x ≤ 0.7,

(7)

g(x) =

1 0.7 < x ≤ 1.

We use this transformation because we observed that paths with distances > 0.7 differed very substantially, with incidental overlap due to a few shared tokens or concepts within the domain. Paths with distances < 0.4 were almost always varying on very similar connections; we did not perceive any marginal value from these. Our function rescales the distance interval to account for these judgments; see plot and examples in Appendix C.

### 4 Constructing the CREATE Benchmark

Dataset Curation: We use Wikidata to construct queries in CREATE. The data generation process is described in Algorithm 1 in Appendix B. We begin by manually selecting a set of relation–category pairs (r,c), each of which defines a class Cr,c = {x | (x,r,c) ∈ G}. These pairs are chosen to ensure that the resulting classes are compact and semantically coherent, with members linked by a specific role, condition, or function, for example, one (r,c) pair in our dataset is (member of sports team, Scuderia Toro Rosso). Triples belonging to the corresponding class Cr,c, sampled from G, include [(Sebastian Vettel, member of sports team, Scuderia Toro Rosso), (Daniel Ricciardo, member of sports team, Scuderia Toro Rosso), (Alex Albon, member of sports team, Scuderia Toro Rosso) , ..., (Max Verstappen, member of sports team, Scuderia Toro Rosso)].

In the next step, we form unordered pairs from entities in Cr,c within the class. For each pair, we randomly select one of the two triples and sample informative outgoing edges from G, yielding an additional one-hop relation. For example, we create a pair: [(Sebastian Vettel, member of sports team, Scuderia Toro Rosso), (Max Verstappen, member of sports team, Scuderia Toro Rosso)], and expand on the second triple to find the link: (Max Verstappen, unmarried partner, Kelly Piquet). We then create a source path that links the three together in a chain: [(Sebastian Vettel, member of sports team, Scuderia Toro Rosso), (Max Verstappen, member of sports team, Scuderia Toro Rosso), (Max Verstappen, unmarried partner, Kelly Piquet)]. The resulting query is formed using the head of the first triple and the tail entity of the last triple. We use gpt-4o-mini-2024-07-18 to re-write the structured triple into a natural language query: “What is the connection between Sebastian Vettel and someone who has been/is a partner of Kelly Piquet?” We show additional examples in Table 2.

Note, we use this source path only as a way to establish that there exists at least one strong path between the entities. We do not evaluate against it as a ground truth reference; we instead evaluate a model on its ability to generate multiple distinct and strong paths.

Benchmark Statistics: We select eleven relations (r) that span diverse domains. The starting entity is one of the types: people, genes, or chemical compounds, spanning both encyclopedic knowledge, classic “trivia” domains like movies, and scientific knowledge. In total we have 931 natural language queries, with at least one verified connection between the entities. Table 3 lists the relations, number

of queries per relation and number of unique starting entities. Examples of source paths and generated queries are given in Table 2; examples are also shown in Figure 3. We describe the filtering process in Appendix B.

Quality Metric f(u): We estimate triple specificity using a single prompt in which a large language model (gpt-oss-120b) jointly assesses the sizes of both predicate-induced classes and directly identifies the larger of the two. This estimated maximum class size is mapped to a discrete score on a five-point scale. Appendix D provides more details. We evaluate factuality using an LLM-as-a-judge (gpt-oss-120b) on the relation level. The prompt for evaluating factuality is given in Appendix K.2.

Distance Metric d(ui,uj): The pairwise distance between paths, by getting their embeddings using all-MiniLM-L6-v2 and using the transformed cosine distance as the function.

### 5 Experimental Setup

Models: We evaluate both non-thinking and thinking large language models on the proposed task, covering a range of architectures, sizes, and training paradigms. Non-thinking models include GPT4.1-mini, GPT-4.1, Qwen2.5-32B-Instruct [38], Qwen3-30B-A3B-Instruct-2507[39], and OLMo-3.132B-Instruct [26]. Thinking models include GPT-5.5, GPT-5.4, GPT-5, GPT-5-mini, Gemini-3.1-Pro, Gemini-3-Pro, Claude-Opus-4.7, Claude-Sonnet-4.6, Claude-Haiku-4.5, Qwen3-32B [39], OLMo3.1-32B-Think [26] and gpt-oss-120b [28]. Appendix I gives model and inference details.

Base prompt: All models are evaluated using a shared base prompt (Appendix K.3) (referred to as ‘original’ in the results) designed to elicit diverse, high-quality relational connections. We prompt the model to generate multiple, possible candidates, so we end up with a set of connections as the output. The model is instructed to output these multiple paths in the form a JSON enclosed by <answer> tags. We then parse out the text into a structured JSON to get all model generated paths. To ensure comparability across models, we use a temperature of 0.7 and a maximum generation length of 4096 tokens for non-thinking models. For thinking models we found this max length was too small, so we vary the token budget/available reasoning budget. For the frontier models that are API-based, we test with the default reasoning budget (unless stated otherwise), with the exception of GPT-5-mini, where we vary the reasoning effort parameter across the supported settings (low, medium, and high). For the open-source models, we explicitly vary the reasoning budget (16k and 32k). Finally, we filter all paths to be structurally valid paths (Section 3) before computing results.

Prompt Variations: One potential approach to mitigating lack of creativity is the use of alternative prompting strategies. Prompt variations are commonly employed to steer model behavior and elicit alternative responses, for example, by paraphrasing the prompt or explicitly requesting a different answer when the initial output is unsatisfactory [46]. In our experiments, we examine whether such prompt-based interventions are effective in eliciting more diverse and higher-quality relational paths from large language models.

- (1) Be creative. We augment the base prompt with the instruction: “Be creative in the type of relationships explored and generated.” to encourage broader exploration of relation types.
- (2) Verbalized Sampling. We adopt verbalized sampling [45] to reduce mode collapse by having the model explicitly express a probability distribution over its outputs. Detailed prompt is given in Appendix K.4.
- (3) In-context regeneration/iterative: After the initial generation, we query the model once again and explicitly ask the model to provide a different answer while keeping all previous answers in the conversation context. This allows the model to see its previous responses and deliberately generate something new. The prompt for this is given in Appendix K.5.
- (4) Resampling: We obtain multiple independent generations from the model using the base prompt (Appendix K.3) with a temperature set to 0.7. Each generation is independent and the model has no knowledge of previous generations.

### 6 Metric Validation

First, we ensure that our evaluation procedure appropriately measure specificity, factuality, and creativity of our paths. We conduct a series of human validation experiments along these axes.

What are different ways of connecting Jean-Pierre Jeunet, the French ﬁlm director, and someone who was nominated for the European Film Award for Best Animated Feature Film?

What are different ways of connecting Ella Jenkins, a notable ﬁgure in children's music, and someone who is associated with the progressive rock genre?

What are different ways of connecting Mick Thomson, the guitarist known for his work in heavy metal, and someone who is a percussionist by occupation?

Mick Thomson performed on “The End, So Far” “The End, So Far” has performer Michael Pfaff, Michael Pfaff occupation Percussionist

Jean-Pierre Jeunet directed Micmacs, Micmacs cast member Yolande Moreau, Yolande Moreau voice actor in Mia and the Migoo Mia and the Migoo nominated for European Film Award for Best Animated Feature Film

Ella Kenkins birthplace St. Louis, St. Louis origin of Pavlov's Dog (band), Pavlov's Dog (band) genre progressive rock

f(u) = 4

. The population connects to Michael Pfaff,

f(u) = 3

but more weakly. Mick Thomson was just a guest performer on “The End, So Far”, so this is a more obscure “trivia” connection

. Other paths include connections through St. Louis, but other entities with weaker connections (e.g., David Sanborn, who collaborated with David Bowie)

f(u) = 4

. Other paths use Micmacs but don’t connect through Yolande Moreau

- Figure 3: Examples of model-generated paths along with quality scores. Path scores are intuitively sensible, and these paths are unique to a single model despite seeming interesting. Table 10 shows the corresponding closest population paths along with the distance numbers.

Specificity: First, we validate that the LLM judge’s specificity scores align with human judgments. The LLM judge produces a class-size estimate mapped to a 1–5 score. Two authors independently annotated 75 paths, assigning scores on the same 1–5 scale while keeping the underlying class-size mapping in mind. The annotators agree substantially, achieving a Krippendorff’s alpha [14] of 0.68. The Pearson correlation between the average of the two annotator ratings and the gpt-oss-120b ratings is 0.67, indicating agreement between the judge and the annotators as well. Some of the main disagreements are because of differences in knowledge domains between annotators and their familiarity with the entities.

Factuality: To validate the judge for factuality, one of the authors annotated a set of 346 modelgenerated triples. The annotator used web search to try to substantiate each relation and produce high-quality labels. On this human-labeled set, the factuality judge achieves an overall balanced accuracy of 85.9%. For incorrect relations, the evaluator achieves high recall (0.94) but lower precision (0.52), while for correct relations it achieves high precision (0.98) and recall (0.77). The most common sources of error are entity/relation misinterpretation and knowledge of long-tail entities.

Robustness to judge model: Recall that both factuality and specificity are evaluated using gpt-oss-120b. Tables 6 and 8 show that our choice of LLM-as-a-judge is comparable to stronger models, and Table 9 confirms that our main findings are robust to this choice. Taken together, these results demonstrate that LLM judges for specificity and factuality (1) correlate substantially with human ratings; (2) are robust enough for model rankings to remain unchanged when the judge model is changed.

Creative Utility: Having established that specificity and factuality are appropriately measured, we now evaluate our metric end-to-end. Two authors independently scored model-generated outputs for 105 CREATE queries on the basis of overall creativity. These judgments were anchored to a 1-5 scale with rubric guidance, but ultimately rest on a subjective notion of creativity similar to the analysis in Figure 3: does the connection seem salient and interesting as a piece of trivia? This evaluation is inherently less quantitatively grounded than the other metrics in this paper, and it relies on the annotators’ domain knowledge, which is not uniform across the represented queries.

The annotators achieved a Krippendorff’s alpha of 0.62, indicating moderate agreement in their relative rankings. The Pearson correlation between each annotator and the creative utility metric was 0.51 and 0.48 respectively, with the correlation between annotator averages and the metric was 0.55.

Crucially, end-to-end scoring with a judge is not a viable basis for our benchmark. We show in Appendix F that our proposed metric produces stable rankings across evaluators, compared to directly prompting an LLM with a holistic scoring rubric, supporting the value of decomposition over end-to-end LLM judgment for creativity.

Taken together, these results show that (1) our evaluations of specificity and factuality appropriately measure these quantities; (2) the overall creativity utility metric correlates well with human judgments; (3) the creative utility metric under different evaluators yields a stable ranking compared to end-to-end LLM scoring. More details and examples of the validation study are provided in Appendix E.

Qualitative Evaluation: To complement the quantitative validation, we examine specific outputs in detail, illustrating what high scoring paths look like in practice. Figure 3 shows examples that have quality at least 3, and which have high minimum distance to the population of paths from that query over all models. From the resulting scored pairs, we chose 3 representative samples.

[Figure 4]

- Figure 4: Model performance on CREATE. We report the creative utility vs. patience values for (a) different frontier models (b) different reasoning effort for GPT-5-mini and gpt-oss-120b (c) different prompt variations for GPT-5-mini. GPT-5.5 and Claude-Sonnet-4.6 achieve the best scores; we explain the low performance of some frontier models in the text.

First, the paths that are high-scoring under our metrics represent appropriately-scored relations. The first and last are classic strong connections through acting. The second is weaker, although St. Louis is a less common origin for bands than many other cities. Second, these connections are novel, where each of these instances were found by only a single model. We want to note that different models surface different interesting connections, especially for tail entities, and no single model captures most of them.

7 Results

Frontier models achieve the highest creative utility. Figure 4 reports creative utility of our different target LLMs on our benchmark at a range of patience values. With patience 1 (no decay in contribution for subsequent paths, except as given by the distance function), LLMs score as high as 18, indicating that at least 18 paths are found on average. GPT-5.5, Claude-Sonnet-4.6, and GPT-5 are ranked first, second, and third across patience values. Moreover, differences between models become more pronounced as patience increases, highlighting how models sustain quality and diversity over longer sets of generations. At patience 1, the gaps between the top two models (GPT-5.5/Claude-Sonnet-4.6) and the rest are statistically significant (p < 0.05); see Appendix J.1 for details.

Certain models are conservative on this task. A few models notably underperform on the task. Gemini 3.1-Pro performs the worst, partially because it is hesitant to generate answers that could be non-factual, despite being asked to in the prompt. Claude-Opus-4.7 also performs poorly even at high reasoning effort. We attribute this to the adaptive thinking capability of this model; we do not see that it thinks long enough on these problems to produce a substantial number of paths.

Reasoning has a mixed impact. In the second subfigure of Figure 4, we see that GPT-5-mini benefits substantially from a larger reasoning budget, while gpt-oss-120b shows little to no improvement. This suggests that the returns to additional reasoning are model-specific rather than universal, and may depend on how effectively a model uses its reasoning trace to plan diverse, high-utility paths.

In Table 16 we report average maximum quality of model generated paths, average pairwise distance within a set, number of factual paths generated, and length of paths generated. These values additionally allow us to conclude that while utility increases with number of paths, some models do well with low path counts: for instance, Gemini-3-Pro obtains similar utility to GPT-5-mini (high reasoning) despite producing fewer paths, by having higher quality and more distinct paths. Appendix J shows examples of raw predictions from GPT-5, Gemini-3-Pro, and Claude-Haiku-4.5, along with examples of paths with high and low quality quality scores (before factuality filtering).

Prompt-based interventions weakly and inconsistently affect creativity across models. In Figure 4 (right side), we show how the utility values change with prompting variations for GPT-5-mini. Iterative prompting is the most effective in increasing the utility, which is expected since these approaches introduce additional generation paths that directly contribute to the metric.

- Figure 5 illustrates how utility changes for different models. For prompt variations, ‘creative’ and ‘original’ prompts perform very similarly across models; they also tend to produce similar number of paths post validity and factuality filtering. ‘verbalized’ prompting produces very low number of valid paths and factual paths. Despite the strength of iterative prompting, resampling performs the

Table 1: Examples of snippets from reasoning traces of open-source models. LLMs show different problem solving behavior employing different strategies to brainstorm connections

Strategy Excerpt Goal Setting Qwen3-32B: First, I need to start with David Koechner, the actor. Target is someone

whose place of birth is Newport Beach, paths need to end with that triple.

Feature Enumeration

Qwen3-32B: First, the obvious paths might be through his acting roles. He’s in ‘Anchorman,’ ‘The Hangover,’ etc. But maybe someone he worked with was born there. Like, maybe a co-star in one of his movies is from Newport Beach. ... For example, in

‘Anchorman,’ the main cast includes Will Ferrell, Steve Carell, etc. Backward Chaining OLMo-3.1-32B-Think : I think I have to make do with what I can think of. Let me try to outline possible paths: 1. David Koechner is a co-star with [X], who was born in Newport Beach. (Need to find X)

Bidirectional Enumeration

gpt-oss-120b: Let’s think of people born in Newport Beach, CA: Notable: Michael Clarke Duncan? No, he was born in Chicago. Let’s list: Nicole Kidman? No. Some celebrities: Beau Bridges? Born in Los Angeles.

best; however, both of these techniques leverage a much larger inference budget on the task. Detailed results for this experiment are given in Table 19.

Models produce similar paths overall. We measure how a generated path differs from the population of paths generated by all models for the same query. We focus on high-quality responses (quality > 3) across 300 queries. For every path that satisfies the criteria above, we calculate each path’s distance from a population of responses for the same query. Table 21 reports the average distance from each path to its closest neighbor in the population. Values are largely similar across models and prompt variations, with “iterate” standing out as the most effective strategy, suggesting that explicitly instructing a model to diverge from its prior outputs yields more distinct responses than resampling alone. Despite the small average distances, there is a small fraction of paths with high distance values from the population that are creatively interesting, as we show in Section 6. See Appendix J.2 for full methodology and the results per model.

### 8 Reasoning Trace Analysis

- Table 1 presents excerpts from the reasoning traces of several open-source models as they attempt to solve CREATE queries. These excerpts illustrate a handful of problem-solving strategies we manually identified. Among the strategies shown, we observe goal setting, in which the model establishes a direction and lists the task constraints, and feature enumeration, in which the model systematically iterates through known properties of the head entity. We also find evidence of backward chaining where the model first sketches outlines of candidate solution paths and then progressively fills in the missing details. Finally, we observe bidirectional enumeration, where the model tends to brainstorm connections from both ends of the query, reasoning from the head entity while simultaneously enumerating from the tail entity. Figure 9 shows a reasoning trace with strategies highlighted, including enumeration (yellow and orange text) and backward chaining (green text). One thing to note from the reasoning chain is that models tend to revisit the same nodes leading to some repeated information, which undermines the effectiveness of the search process. Characterizing these strategies and their effectiveness reveals where reasoning helps versus where it hurts. This points to concrete targets for improving search efficiency and informs the design of training signals and inference-time interventions for better LLM creativity.

[Figure 5]

[Figure 6]

Figure 5: Alternative prompting methods can lead to improvements depending on the model. Iterate and Resample interventions lead to the highest creative utility scores.

### 9 Conclusion

This paper presents a benchmark for testing LLMs at associative creativity. Queries require models to find a set of strong, distinct connections between real concepts, mimicking the kind of associative creativity required more broadly, but maintaining evaluatability. Our results highlight strengths of current models, but also show that higher thinking effort is not necessarily a silver bullet in these settings. Further work is needed to fully leverage LLMs for tasks requiring associative creativity.

### Acknowledgments

This work was partially supported by NSF CAREER Awards IIS-2145280 and IIS-2145479, NSF grants IIS-2433071 and IIS-2107524, NIH grant 1R01LM01460001, the NSF AI Institute for Foundations of Machine Learning (IFML), and the NSF-Simons AI Institute for Cosmic Origins (CosmicAI) under NSF Cooperative Agreement 2421782 and Simons Foundation MPS-AI-00010515. This work is also partially supported by the Sloan Foundation and grants from Amazon and Open Philanthropy. This research has been supported by computing support from the Torch cluster at NYU as well as a compute grant from NVIDIA.

### References

- [1] Dhruv Agarwal, Bodhisattwa Prasad Majumder, Reece Adamson, Megha Chakravorty, Satvika Reddy Gavireddy, Aditya Parashar, Harshit Surana, Bhavana Dalvi Mishra, Andrew McCallum, Ashish Sabharwal, and Peter Clark. AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- [2] Barrett R Anderson, Jash Hemant Shah, and Max Kreminski. Homogenization Effects of Large Language Models on Human Creative Ideation. In Proceedings of the 16th conference on creativity & cognition, pages 413–425, 2024.
- [3] Benjamin Samuel Bloom, Max D Engelhart, Edward J Furst, Walker H Hill, and David R Krathwohl. Taxonomy of educational objectives, volume 2. Longmans, Green New York, 1964.
- [4] Boden, Margaret A. The creative mind: myths and mechanisms. Basic Books, Inc., USA, 1991.
- [5] Tuhin Chakrabarty, Philippe Laban, Divyansh Agarwal, Smaranda Muresan, and Chien-Sheng Wu. Art or Artifice? Large Language Models and the False Promise of Creativity. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, pages 1–34, 2024.
- [6] Wenxuan Ding, Shangbin Feng, Yuhan Liu, Zhaoxuan Tan, Vidhisha Balachandran, Tianxing He, and Yulia Tsvetkov. Knowledge Crosswords: Geometric Knowledge Reasoning with Large Language Models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 2609–2636, 2024.
- [7] Daniel Fein, Sebastian Russo, Violet Xiang, Kabir Jolly, Rafael Rafailov, and Nick Haber. LitBench: A Benchmark and Dataset for Reliable Evaluation of Creative Writing. arXiv preprint arXiv:2507.00769, 2025.
- [8] Dedre Gentner. Structure-mapping: A theoretical framework for analogy. Cognitive science, 7(2):155–170, 1983.
- [9] Joy Paul Guilford, Paul R Christensen, Philip R Merrifield, and Robert C Wilson. Alternate uses. 1978.
- [10] Sikun Guo, Amir Hassan Shariatmadari, Guangzhi Xiong, Albert Huang, Myles Kim, Corey M Williams, Stefan Bekiranov, and Aidong Zhang. IdeaBench: Benchmarking Large Language Models for Research Idea Generation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 5888–5899, 2025.
- [11] Jie He, Nan Hu, Wanqiu Long, Jiaoyan Chen, and Jeff Z Pan. MINTQA: A Multi-Hop Question Answering Benchmark for Evaluating LLMs on New and Tail Knowledge. arXiv preprint arXiv:2412.17032, 2024.

- [12] Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, Alon Albalak, and Yejin Choi. Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond). In Neural Information Processing Systems, 2025.
- [13] Arthur Koestler. The Act of Creation. 1964.
- [14] Klaus Krippendorff. Reliability in content analysis. Human Communication Research, 30(3):411–433, 2004.
- [15] Alex Kulesza and Ben Taskar. Determinantal Point Processes for Machine Learning. Found. Trends Mach. Learn., 5:123–286, 2012.
- [16] Thom Lake, Eunsol Choi, and Greg Durrett. "From Distributional to Overton Pluralism: Investigating Large Language Model Alignment". In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6794–6814, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics.
- [17] Hui Lin and Jeff Bilmes. A Class of Submodular Functions for Document Summarization. In Dekang Lin, Yuji Matsumoto, and Rada Mihalcea, editors, Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 510–520, Portland, Oregon, USA, June 2011. Association for Computational Linguistics.
- [18] Emmy Liu. The Promises and Pitfalls of AI Scientists. Online, 2025.
- [19] Haokun Liu, Sicong Huang, Jingyu Hu, Yangqiaoyu Zhou, and Chenhao Tan. HypoBench: Towards Systematic and Principled Benchmarking for Hypothesis Generation. arXiv preprint arXiv:2504.11524, 2025.
- [20] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Nicolaus Foerster, Jeff Clune, and David Ha. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. ArXiv, abs/2408.06292, 2024.
- [21] Yining Lu, Dixuan Wang, Tianjian Li, Dongwei Jiang, Sanjeev Khudanpur, Meng Jiang, and Daniel Khashabi. Benchmarking Language Model Creativity: A Case Study on Code Generation. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2776–2794, 2025.
- [22] Bodhisattwa Prasad Majumder, Harshit Surana, Dhruv Agarwal, Bhavana Dalvi Mishra, Abhijeetsingh Meena, Aryan Prakhar, Tirth Vora, Tushar Khot, Ashish Sabharwal, and Peter Clark. DiscoveryBench: Towards Data-Driven Discovery with Large Language Models. In Proceedings of ICLR, 2024.
- [23] Martha T Mednick and Sharon Halpern. Remote associates test. Psychological Review, 1968.
- [24] Ludovico Mitchener, Angela Yiu, Benjamin Chang, Mathieu Bourdenx, Tyler Nadolski, Arvis Sulovari, Eric C. Landsness, Dániel L. Barabási, Siddharth Narayanan, Nicky Evans, Shriya Reddy, Martha S. Foiani, Aizad Kamal, Leah P. Shriver, Fang Cao, Asmamaw T. Wassie, Jon M. Laurent, Edwin Melville-Green, Mayk Caldas Ramos, Albert Bou, Kaleigh F. Roberts, Sladjana Zagorac, Timothy C. Orr, Miranda E. Orr, Kevin J. Zwezdaryk, Ali E. Ghareeb, Laurie McCoy, Bruna Gomes, Euan A Ashley, Karen E. Duff, Tonio Buonassisi, Tom Rainforth, Randall J. Bateman, Michael Skarlinski, Samuel G. Rodriques, Michaela M. Hinks, and Andrew D. White. Kosmos: An AI Scientist for Autonomous Discovery. ArXiv, abs/2511.02824, 2025.
- [25] Vaishnavh Nagarajan, Chen Henry Wu, Charles Ding, and Aditi Raghunathan. Roll the dice & look before you leap: Going beyond the creative limits of next-token prediction. In Forty-second International Conference on Machine Learning, 2025.
- [26] Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan

- Lambert, Pete Walsh, Pradeep Dasigi, Robert Berry, Saumya Malik, Saurabh Shah, Scott Geng, Shane Arora, Shashank Gupta, Taira Anderson, Teng Xiao, Tyler Murray, Tyler Romero, Victoria Graf, Akari Asai, Akshita Bhagia, Alexander Wettig, Alisa Liu, Aman Rangapur, Chloe Anastasiades, Costa Huang, Dustin Schwenk, Harsh Trivedi, Ian Magnusson, Jaron Lochner, Jiacheng Liu, Lester James V. Miranda, Maarten Sap, Malia Morgan, Michael Schmitz, Michal Guerquin, Michael Wilson, Regan Huff, Ronan Le Bras, Rui Xin, Rulin Shao, Sam Skjonsberg, Shannon Zejiang Shen, Shuyue Stella Li, Tucker Wilde, Valentina Pyatkin, Will Merrill, Yapei Chang, Yuling Gu, Zhiyuan Zeng, Ashish Sabharwal, Luke Zettlemoyer, Pang Wei Koh, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. Olmo 3, 2025.
- [27] Jay A Olson, Johnny Nahas, Denis Chmoulevitch, Simon J Cropper, and Margaret E Webb. Naming unrelated words predicts creativity. Proceedings of the National Academy of Sciences, 118(25):e2022340118, 2021.
- [28] OpenAI. gpt-oss-120b and gpt-oss-20b model card, 2025.
- [29] Vishakh Padmakumar and He He. Does Writing with Language Models Reduce Content Diversity? In Proceedings of the International Conference on Learning Representations, 2024.
- [30] Vishakh Padmakumar, Chen Yueh-Han, Jane Pan, Valerie Chen, and He He. Beyond Memorization: Mapping the Originality-Quality Frontier of Language Models. arXiv preprint arXiv:2504.09389, 2025.
- [31] Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 1532–1543, 2014.
- [32] Kai Ruan, Xuan Wang, Jixiang Hong, Peng Wang, Yang Liu, and Hao Sun. LiveIdeaBench: Evaluating LLMs’ Divergent Thinking for Scientific Idea Generation with Minimal Context. arXiv preprint arXiv:2412.17596, 2024.
- [33] Samuel Schapiro, Sumuk Shashidhar, Alexi Gladstone, Jonah Black, Royce Moon, Dilek Hakkani-Tur, and Lav R Varshney. Combinatorial Creativity: A New Frontier in Generalization Abilities. arXiv preprint arXiv:2509.21043, 2025.
- [34] Chenglei Si, Tatsunori Hashimoto, and Diyi Yang. The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas. arXiv preprint arXiv:2506.20803, 2025.
- [35] Chenglei Si, Zitong Yang, Yejin Choi, Emmanuel Candès, Diyi Yang, and Tatsunori Hashimoto. Towards Execution-Grounded Automated AI Research. arXiv eprint 2601.14525, 2026.
- [36] Taylor Sorensen, Jared Moore, Jillian Fisher, Mitchell Gordon, Niloofar Mireshghallah, Christopher Michael Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, Tim Althoff, and Yejin Choi. A Roadmap to Pluralistic Alignment. arXiv eprint 2402.05070, 2024.
- [37] RJ Sternberg. Beyond IQ: A Triarchic Theory of Human Intelligence. CUP Archive, 1985.
- [38] Qwen Team. Qwen2.5: A Party of Foundation Models, September 2024.
- [39] Qwen Team. Qwen3 Technical Report, 2025.
- [40] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop Questions via Single-hop Question Composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.
- [41] Manya Wadhwa, Zayne Sprague, Chaitanya Malaviya, Philippe Laban, Junyi Jessy Li, and Greg Durrett. EvalAgent: EvalAgent: Discovering Implicit Evaluation Criteria from the Web. In Proceedings of the Conference on Language Modeling, 2025.
- [42] E Wenger and Y Kenett. We’re Different, We’re the Same: Creative Homogeneity Across LLMs. arXiv, 2025.
- [43] Peter West and Christopher Potts. Base Models Beat Aligned Models at Randomness and Creativity. URL https://arxiv. org/abs/2505.00047.

- [44] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380, 2018.
- [45] Jiayi Zhang, Simon Yu, Derek Chong, Anthony Sicilia, Michael R Tomz, Christopher D Manning, and Weiyan Shi. Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity. arXiv preprint arXiv:2510.01171, 2025.
- [46] Yiming Zhang, Harshita Diddee, Susan Holm, Hanchen Liu, Xinyue Liu, Vinay Samuel, Barry Wang, and Daphne Ippolito. NoveltyBench: Evaluating Creativity and Diversity in Language Models. In Second Conference on Language Modeling, 2025.
- [47] Yiming Zhang, Avi Schwarzschild, Nicholas Carlini, Zico Kolter, and Daphne Ippolito. Forcing Diffuse Distributions out of Language Models. arXiv preprint arXiv:2404.10859, 2024.

### A Limitations

This dataset is limited to the English language and is based on facts available on Wikidata, although not restricted to those facts at evaluation time. It reflects a curated set of domains (people, genes and medical drugs) meant to be representative of trivia connections and ideation, but without perfect coverage of all real-world applications. As a result, this dataset may not perfectly reflect associative creativity in other languages or other domains; however, the metric that we propose is generally applicable, and the approach to constructing the dataset is transferable to other domains as well.

### B Benchmark

We describe the data generation process in Algorithm 1. To ensure quality and non-trivial connections, we remove any links where the source entity also satisfies the tail connection. We also remove any relations (r,c) where r induces a class that has a very large membership (e.g. native language, instance of, date of birth, gender, described at URL etc). We show examples of queries generated by our system in Table 2. Statistics about the relations used in our benchmark along with the number of entities and queries are given in Table 3. A source path generated according to our benchmark is about three triples long.

Algorithm 1 Generate Natural Language Queries from a Knowledge Graph

Require: Knowledge graph G = (E, R); category set C = {(r, c) | r ∈ R, c ∈ E}; text model M(·) Ensure: Query list Q

- 1: Q ← ∅ {collected natural-language queries}
- 2: for all (r, c) ∈ C do
- 3: Cr,c ← {x ∈ E | (x, r, c) ∈ G} {entities in class defined by (r, c)}
- 4: P ← {{x1, x2} ⊆ Cr,c | x1 ̸= x2} {unordered entity pairs within the class}
- 5: for all {x1, x2} ∈ P do
- 6: sample uniformly x ∈ {x1, x2} {choose one endpoint to expand by one hop}
- 7: O(x) ← {(x, r′, y) ∈ G | r′ ∈ R, y ∈ E} {outgoing edges from x}
- 8: sample uniformly (x, r′, y) ∈ O(x) {pick an informative 1-hop fact}
- 9: s ← (x1, r, c), (x2, r, c), (x, r′, y) {source path: shared class + extra hop}
- 10: q ← M prompt [x1, _, _], [_, r′, y] {convert s into NL query}
- 11: Q ← Q ∪ {q} {store query for evaluation}
- 12: end for
- 13: end for
- 14: return Q

Benchmark License Our benchmark is licensed under CC BY-SA 4.0.

### C Distance Function

In Section 3.2, we introduce the distance function used for calculating the creative utility metric. Table 4 shows examples of pairs of paths, along with the raw cosine distance and the transformed

- Table 2: Examples of instances of (relation, category) in our dataset, along with pairs of triples retrieved that satisfy this category, expanded triple 2, and examples of the constructed query. (Relation, Category) Triples retrieved Triple 2 expanded Example Query (cast member, Goodfellas)

- Triple 1: (Goodfellas, cast member, Robbie Vinton)

(Vincent Gallo, occupation, painter)

What are different ways of connecting Robbie Vinton, an American Actor, and someone who is a painter?

- Triple 2: (Goodfellas, cast member, Vincent Gallo)

(member of sports team, Scuderia Toro Rosso)

- Triple 1: (Sebastian Vettel, member of sports team, Scuderia Toro Rosso)

(Max Verstappen, unmarried partner, Kelly Piquet)

What are different ways of connecting Sebastian Vettel, the German racing driver, and someone who has been/is a unmarried partner of Kelly Piquet?

- Triple 2: (Max Verstappen, member of sports team, Scuderia Toro Rosso)

- Triple 1: (prednisolone, medical condition treated, ulcerative colitis)

(methotrexate, subject has role, folic acid antagonists)

What are different ways of connecting prednisolone, a medication used to treat various conditions, and a substance that plays a role as a folic acid antagonist?

- Triple 2: (methotrexate, medical condition treated, ulcerative colitis)

(medical condition treated, ulcerative colitis)

[Figure 7]

- Figure 6: Pre-transformation, transformation function, and post-transformation cosine distance scores for GPT-5 (medium) paths. Many pairs being scored at a distance of 1 lined up with our intuitive assessment that they represented completely different relationships.

values. Paths with values cosine distance >0.7 are substantially different, focusing on different domains and properties of the entities (for example, familial connections vs professional service), where as those under 0.4 have similarities in domains of properties traversed, entities and sometimes are paraphrases of each other (for example, connections via position in office or both paths focusing on familial connections).

### D Evaluation Prompts

Table 4: Examples of paths, raw distance values, and distance values post transformation

cosine distance g(x)

Path 1 Path 2

((‘george h. w. bush’, ‘successor as president to’, ‘ronald reagan’), (‘ronald reagan’, ‘spouse’, ‘nancy reagan’), (‘nancy reagan’, ‘position held’, ‘first lady of the united states’))

((‘george h. w. bush’, ‘vice president to’, ‘ronald reagan’), (‘ronald reagan’, ‘spouse’, ‘nancy reagan’), (‘nancy reagan’, ‘position held’, ‘first lady of the united states’))

0.13 0.003

((‘lee jong wook’, ‘was director-general of’, ‘world health organization’), (‘world health organization’, ‘has employee educated at’, ‘london school of hygiene & tropical medicine’))

((‘lee jong wook’, ‘worked at’, ‘world health organization’), (‘world health organization’, ‘employed’, ‘david nabarro’), (‘david nabarro’, ‘educated at’, ‘london school of hygiene & tropical medicine’))

((‘josefa edralin marcos’, ‘relative’, ‘michael marcos keon’), (‘michael marcos keon’, ‘occupation’, ‘lawyer’))

((‘josefa edralin marcos’, ‘spouse’, ‘ferdinand marcos’), (‘ferdinand marcos’, ‘legal mentor of’, ‘benjamin abalos jr.’), (‘benjamin abalos jr.’, ‘occupation’, ‘lawyer’))

0.33 0.12

0.40 0.24

((‘christopher nolan’, ‘sibling’, ‘jonathan nolan’), (‘jonathan nolan’, ‘spouse’, ‘lisa joy’))

((‘christopher nolan’, ‘directed’, ‘the dark knight trilogy’), (‘the dark knight trilogy’, ‘screenplay by’, ‘jonathan nolan’), (‘jonathan nolan’, ‘spouse’, ‘lisa joy’))

0.58 0.77

((‘joe spinell’, ‘worked with director’, ‘martin scorsese’), (‘martin scorsese’, ‘place of birth’, ‘new york city’), (‘new york city’, ‘includes borough’, ‘brooklyn’))

((‘joe spinell’, ‘acted in tv movie’, ‘she waits’), (‘she waits’, ‘produced by’, ‘abc’), (‘abc’, ‘headquarters in’, ‘new york city’), (‘new york city’, ‘includes borough’, ‘brooklyn’))

0.71 1.0

((‘george h. w. bush’, ‘associated with’, ‘bush family’), (‘bush family’, ‘includes’, ‘barbara bush’), (‘barbara bush’, ‘position held’, ‘first lady of the united states’))

((‘george h. w. bush’, ‘served as’, ‘director of central intelligence’), (‘director of central intelligence’, ‘advised’, ‘hillary clinton’), (‘hillary clinton’, ‘position held’, ‘first lady of the united states’))

0.8 1.0

Prompts K.1 and K.2 are used for evaluating specificity and factuality.

For the specificity prompt, we elicit class sizes per triple from the model. Once we have the class sizes we use the following mapping to convert the size to a 1-5 scale.

q(class_size) = 

5, if class_size ≤ 10, 4, if 10 < class_size < 100, 3, if 100 ≤ class_size < 500, 2, if 500 ≤ class_size < 5000, 1, otherwise

.



### E Metric Validation

Table 3: List of source relations, number of unique entities, number of queries along with the average # of source paths per relation.

ID Label Unique e1 # Queries P166 award received 99 99

- P39 position held 93 94
- P40 child 86 88 P463 member of 88 89 P54 member of sports team 99 99

- P57 director 30 47
- P58 screenwriter 48 60

- P161 cast member 39 96
- P162 producer 43 65 P2175 medical condition treated 96 100 P2293 genetic association 94 94

#### E.1 Factuality

In this section, we report the detailed performance metrics for gpt-oss-120b as the evaluator for factuality. We annotated 346 relations

Table 7: Factuality Judgments and Explanations for Knowledge Triples. The judgments highlighted are incorrect because of entities and the relation being too long tail.

Triples LLM Explanation LLM Judgments (rick mcphail, associated musical artist, thees uhlmann) (thees uhlmann,country of citizenship,germany)

hallucinated, not hallucinated

The first triple links ‘rick mcphail’ to ‘thees uhlmann’ as an associated musical artist. While Thees Uhlmann is a real German musician, there is no verifiable record of a musician named Rick McPhail, nor any known collaboration or association between them. The lack of evidence for the existence of Rick McPhail as a musical artist makes this triple hallucinated. The second triple states that Thees Uhlmann’s country of citizenship is Germany. This matches publicly available biographical information about Uhlmann, confirming the claim as accurate and not hallucinated.

(robert metcalfe, educated at, harvard university) (harvard university, employer, edmund m. clarke] (edmund m. clarke, doctoral student, sergio vale aguiar campos)

The first triple is correct: Robert Metcalfe earned his Ph.D. from Harvard University, so both entities exist and the ’educated at’ relation is factual. The second triple claims Harvard University employs Edmund M. Clarke; Clarke was a long-time professor at Carnegie Mellon University and has no known employment by Harvard, making this claim unsupported. The third triple asserts that Edmund M. Clarke supervised a doctoral student named Sergio Vale Aguiar Campos; there is no public record of such a student under Clarke, suggesting the entity or relationship is fabricated. Therefore, only the first triple is not hallucinated.

not hallucinated, hallucinated, hallucinated

(triples) and then validated the evaluator against our judgments. We report the precision/recall for the binary classification in Table 5. The balanced accuracy for the evaluator is 0.86.

The most common sources of errors for this factuality evaluator are, firstly, the model incorrectly identifies and assesses generations involving niche and long tail entities, which points to a knowledge gap. Secondly, the model often misinterprets relations and entities leading to false judgments. Often if the entity is too general, it is evaluated strictly, even though the relation holds true between the general entities. Table 7 shows two examples of LLM judgment on triples generated by one of models.

Table 5: LLM-as-a-judge performance for evaluating factuality of a generated path.

Class Precision Recall F1-Score Support

- 0 0.52 0.94 0.67 72
- 1 0.98 0.77 0.87 274

#### E.2 Comparing Different LLM-as-a-judge models

To evaluate the choice of model for the LLM-as-a-judge, we compare different LLMs against our human annotations. We test GPT-5-mini, GPT-4o-mini and GPT-4.1mini. From Tables 6 and 8 we see that our choice of the model i.e. gpt-oss-120b does fairly well given the cost and the effectiveness.

#### E.3 Human annotations on creativity

We give the following instructions to human annotators for evaluating the creativity/interestingness of the model generated outputs. The annotations are on a scale of 1-5.

Table 6: Performance of different models as the factuality evaluator. We report the binary accuracy, the F1 for the positive i.e. factual class (pos) and the negative i.e. hallucinated class (neg).

Model Binary Acc F1(pos) F1 (neg) gpt-oss-120b 0.86 0.87 0.67 GPT-5-mini 0.82 0.87 0.67 GPT-4o-mini 0.69 0.83 0.53 GPT-4.1-mini 0.68 0.85 0.50

- Table 9: This table shows (a) the stability of our proposed metric under different evaluators (b) how our proposed creative utility metric is more stable compared to simply prompting an LLM for creativity judgments.

Proposed Creative Utility model GPT-4.1-mini rank GPT-4o-mini rank gpt-oss-120b rank

GPT-5-mini (low) 9.94 5 7.86 5 6.81 5 GPT-5-mini (medium) 11.30 4 8.99 4 8.58 4 GPT-5-mini (high) 12.58 3 10.23 3 10.19 3 GPT-5 13.41 1 11.33 1 12.34 1 Claude-Haiku-4.5 7.48 6 6.79 6 5.54 6 Gemini-3-Pro 13.07 2 10.91 2 10.68 2

Single prompt w/ LLM as a judge

GPT-5-mini (low) 3.63 3 4.29 5 3.07 5 GPT-5-mini (medium) 3.58 4 4.33 3 3.13 4 GPT-5-mini (high) 3.65 2 4.41 2 3.25 2 GPT-5 3.51 5 4.32 4 3.16 3 Claude-Haiku-4.5 3.05 6 3.84 6 2.63 6 Gemini-3-Pro 3.99 1 4.53 1 3.58 1

- 1: Not creative: Nearly all paths pass through the same intermediate concept(s) or domain. Variations are superficial, different relation labels but the same nodes. The set feels like one idea restated.
- 2: Slightly creative: Most paths cluster around one dominant intermediate concept or domain, with one or two outliers. The outliers are either weak (obscure intermediate nodes) or structurally similar to the rest. Minimal semantic spread.
- 3: Moderately creative: Paths span distinct intermediate domains. Some clustering remains but there is genuine topical breadth. Intermediate nodes are reasonably well-known. Few paths are meaningfully different from each other.
- 4: Creative: Paths traverse meaningfully distinct intermediate domains via salient, well-known entities. Topical clustering is limited. The core set of paths is diverse — some paths add new angles rather than pad existing ones.
- 5: Highly creative: Paths spread across a wide semantic space. Almost no two paths share a central intermediate concept. Intermediate entities are salient and important in their respective domains. The set feels like a genuine, broad-coverage map of how the two entities can be meaningfully linked.

### F Stability of Creative Utility vs. Stability of End-to-End Evaluation

We compare two approaches for evaluating outputs on CREATE: creative utility and end-to-end evaluation of creativity with LLM-as-a-judge. We show in this section that creative utility is more stable across LLMs: due to being grounded in objective quantities like class sizes, results vary less as the judge model is changed.

Table 8: Pearson’s correlation of specificity judgments between humans and different models

Ann gpt-oss-120b GPT-4o-mini gpt-41-mini GPT-5-mini

- A 0.64 0.78 0.56 0.81

- B 0.58 0.63 0.47 0.75 avg 0.67 0.78 0.56 0.85

To evaluate this, we re-ran our evaluations using different LLMs. We sampled a subset of 165 instances (15 per relation type) with outputs from the following models: GPT-5, Gemini-3-pro, Claude-Haiku-4.5-med, and GPT-5-mini (at high, medium, and low reasoning levels). In addition to our standard gpt-oss-120b evaluator, we explored two additional models: GPT-4o-mini and GPT-4.1-mini.

- Table 10: Examples of model-generated paths u, the population path u is closest to, along with quality scores and minimum distance values.

Path σ population path d

(jean-pierre jeunet, directed, micmacs), (micmacs, cast member, yolande moreau), (yolande moreau, voice actor in, mia and the migoo), (mia and the migoo, nominated for, european film award for best animated feature film)

4 (jean-pierre jeunet, directed, micmacs), (micmacs, featured animation by, annecy international animated film festival winners), (annecy international animated film festival winners, nominated for, european film award for best animated feature film)

0.40

(jean-pierre jeunet, directed, delicatessen), (delicatessen, cast member, karin viard), (karin viard, voice actor in, the suicide shop), (the suicide shop, nominated for, european film award for best animated feature film)

4 (jean-pierre jeunet, directed, delicatessen), (delicatessen, nominated for, european film award for best animated feature film)

0.18

(jean-pierre jeunet, directed, amélie), (amélie, cast member, jamel debbouze), (jamel debbouze, voice actor in, why i did (not) eat my father), (why i did (not) eat my father, nominated for, european film award for best animated feature film)

4 (jean-pierre jeunet, directed, amélie), (amélie, features, jamel debbouze), (jamel debbouze, featured in, a monster in paris), (a monster in paris, nominated for, european film award for best animated feature film)

0.04

4 (jean-pierre jeunet, directed, amélie), (amélie, starred, mathieu kassovitz), (mathieu kassovitz, voice actor in, april and the extraordinary world), (april and the extraordinary world, nominated for, european film award for best animated feature film)

0.00

(jean-pierre jeunet, directed, amélie), (amélie, cast member, mathieu kassovitz), (mathieu kassovitz, voice actor in, april and the extraordinary world), (april and the extraordinary world, nominated for, european film award for best animated feature film)

As shown in Table 9, our creative utility metric remains stable across different LLMs used for factuality and specificity evaluation. It is notably more stable than prompting an LLM directly for creative utility judgments, where scores vary by over an entire point on a 1-5 scale. To quantify this further, we computed Krippendorff’s alpha between the three evaluators across all six models at the instance level ( 900 instances total). On our creative utility metric, our LLM annotators achieve an alpha of 0.76, compared to just 0.31 for direct prompting. This substantial gap indicates that our decomposed metric yields far more consistent rankings across evaluator models, whereas direct rubric-based prompting is highly sensitive to the choice of LLM judge.

### G Qualitative Validation

In Table 10 we show all valid, factually correct paths with quality ≥3 for the query: ‘What are different ways of connecting Jean-Pierre Jeunet, the French film director, and someone who was nominated for the European Film Award for Best Animated Feature Film?’. For our analysis in Section 6, from a set of predictions for this query, we pick the path with the highest minimum distance from the population. This path is the most distinctive in the set of paths generated compared to a population. If we look at some paths with a lower distance score, we see that there is more overlap in the entities and relations explored, thereby showing that our metric is able to capture distinctive paths generated by the model.

### H Inference Prompts and Variations

Our inference pipeline has one base prompt and we also evaluate responses to different prompt variations to see if prompting has any influence on the creativity of responses. The base prompt is Prompt K.3, the verbalized sampling prompt is Prompt K.4. For the ‘creative’ prompt variation, we simply append the instruction ‘- Be creative in the type of relationships explored and generated’ to the base prompt. For the iterative prompt variation, the prompt is Prompt K.5.

Table 11: LLM checkpoints

Model Checkpoint reasoning budget

Olmo-3.1-32B-Instruct allenai/Olmo-3.1-32B-Instruct Olmo-3.1-32B-Think allenai/Olmo-3.1-32B-Think 16k / 32k Qwen3-30B-Instruct Qwen/Qwen3-30B-A3B-Instruct-2507 Qwen3-32B Qwen/Qwen3-32B 16k/32k gpt-oss-120b openai/gpt-oss-120b low/med/high Claude-Haiku-4.5 claude-haiku-4-5-20251001 (default) Claude-Sonnet-4.6 claude-sonnet-4-6 (default) Claude-Opus-4.7 claude-opus-4-7 (default-high / xhigh)

- GPT-4.1 gpt-4.1-2025-04-14 -

- GPT-4.1-mini gpt-4.1-mini-2025-04-14 GPT-5-mini gpt-5-mini-2025-08-07 low/med/high GPT-5 gpt-5-2025-08-07 default-med

- GPT-5.4 gpt-5.4-2026-03-05 default - med

- GPT-5.5 gpt-5.5-2026-04-23 default - med Gemini-3-Pro gemini-3-pro-preview medium Gemini-3.1-Pro gemini-3.1-pro-preview medium

[Figure 8]

Figure 7: Distribution of creative utility scores for GPT-5.5 and GPT-5.

### I Model Details

All inference was performed through LiteLLM, with proprietary models accessed via their respective APIs and open-source models served through self-hosted vLLM endpoints. For open-source models, we run the model inference using NVIDIA H200s GPUs.

For open-source models we match the compute where possible, running inference at 16k and 32k tokens. For gpt-oss-120b we run inference at low/medium/high (as mentioned in the model specification). For thinking models, we stick to default or medium reasoning, with two exceptions: (a) Claude Opus, where we had to use high/xhigh because medium reasoning did not trigger the thinking since this model uses adaptive thinking; and (b) GPT-5-mini, where we experiment with the range of reasoning efforts available i.e. low/med/high.

### J Results

Tables 12, 13 and 14 show examples of raw model outputs for a query. Claude tends to be very conservative when attempting our task, focusing more on verification of the information and hence giving fewer connections. Table 15 shows paths along with the corresponding specificity scores. Relations that are stronger, like, being a part of the same sports team, or being featured in certain movies, which have have less number of members, score higher on specificity. General relations, like educated at a university, score lower on quality since the class size of people being educated at a university is pretty large.

- Figure 7 shows the distribution of creative utility scores for two models, GPT-5.5 and GPT-5.

In Table 20 we report the creative utility values when we keep all valid model generated paths and do not filter for factuality. We discuss this in detail in Appendix J.3.

Tables 17 and 18 show examples of raw outputs from two prompt variations, ‘creative’ and ‘verbalized sampling’ respectively for the same query as the tables above by GPT-5-mini. We see that creative prompting doesn’t change the distribution of relations as much and verbalized sampling really reduces the number of paths generated, probably because of the added complexity of verbalizing a probability with each path.

#### J.1 Significance Testing

For the 6 models in Figure 4(a), we test all 62 = 15 pairwise differences in mean creative utility at patience=1.0. For each pair, we compute per-query utility differences (intersecting on the query

indices both models scored) and run a paired bootstrap with 10,000 resamples to obtain a 95% percentile CI on the mean difference and a two-sided p-value. We apply Bonferroni correction across the 15 pairs. 13 of 15 pairs are significant at α = 0.05. The two exceptions are Claude-Sonnet-4.6 vs. GPT-5.5 and Claude-Opus-4.7 (xhigh) vs. Gemini-3-pro.

#### J.2 Quality and Distinctiveness

Following our initial analysis in Section 6, we perform an automated experiment to discern the distinctiveness of the generations. Using ν, we analyze the frontier of high-quality responses produced by each model.

[Figure 9]

We subsample 300 queries from each model and, for each setting, filter out invalid paths, factual inconsistent paths, and paths with quality ≤ 3. For the remaining paths for each query, we calculate each paths’ distance from a population of responses for the same query. Avg ν reflects the average distance to the closest path in the population, and max ν reflects the best produced instance for each query.

Figure 8: The graph shows how the creative utility (patience=0.9) of a model changes when we include factuality in the objective. Models trade off factuality for utility.

Table 21 shows that distinctiveness values are similar across models. We particularly focus on prompt variations, and see that on average they do not produce very distinctive responses compared to the base prompt. Interestingly, “iterate” has the strongest positive impact here. This metric captures something different than utility: we see that resampling is less useful here, but knowing what a model has generated before and explicitly instructing it to generate different things seems to find more distinct items.

#### J.3 Quality vs Factuality

Table 20, we report creative utility without any factuality component of quality, along with the average number of factually correct triples generated by a model. We observe a tradeoff between utility and factuality: GPT-5 achieves lower utility compared to Claude-Sonnet-4.6 but has higher factuality.

To quantify the utility-factuality tradeoff, we define a factuality filter. We first define t as the fraction of triples in a path that are factually correct. We then vary t and modify Equation 1 to compute a factuality-adjusted utility:

|U|

γi−1 I q uτ(i) > t ×σ uτ(i) min j<i

s(U, t) = max

d uτ(i), uτ(j) ,

τ

i=1

Note, instead of having a unified quality metric as defined in Equation 5, we consider factuality to be another dimension for this analysis and use σ as the quality measure.

- Figure 8 shows a consistent drop in creative utility the factuality cutoff increases, as expected. At the most lenient, when we consider all valid paths generate by the models, we observe Sonnet-4.6 achieving the highest creative utility, followed by GPT-5.5, then gpt-oss-120b close behind. But, at

the strictest setting, only retaining paths that are entirely factual, we observe GPT-5.5 and Sonnet-4.6 performing the best, while open-source models have a substantial drop, revealing a large gap. Note that because many paths have 2 or 3 relations, the steepest drop-off is from a threshold of 0.5 to 0.7.

#### J.4 Search Trace

We use Prompt K.6 to extract entities and relations from a search trace using gpt-4.1-mini-2025-0414. Table 22 summarizes the number of entities, relations and triples explored by the model in its reasoning trace.

- Figure 9 shows an example of a reasoning trace. We see various strategies the model uses to brainstorm connections, like enumeration, cross domain connections and template based search. We also see a lot of repetitions of entities and search paths, indicating not optimal use of tokens for this task and thus identifying a gap.

### K Tests for human creativity and diversity

Human creativity is often assessed using tasks like Remote Associations Test (RAT) [23], Alternate Uses Task (AUT) [9], and Diversity Association Task (DAT) [27]. In this section, we show that these tests are saturated for frontier models and new benchmarks are needed.

We describe the tests along with the evaluation criteria:

Remote Associations Test (RAT) This test consists of three common stimulus words that appear to be unrelated. The subject must think of a fourth word that is somehow related to each of the first three words.

Table 22: Number of entities (e) and relations (r) explored by thinking models in their reasoning traces.

Model r e

Olmo-3.1-32B / 16k 12.04 18.83 Olmo-3.1-32B / 32k 12.22 20.05 Qwen3-32B / 16k 15.12 27.33 Qwen3-32B / 32k 14.81 26.18 gpt-oss-120b / low 24.23 28.97 gpt-oss-120b /high 24.73 27.99

#### Alternate Uses Task (AUT) This test mea-

sures creativity of a subject by asking them to list as many unique uses for a common object (e.g. a brick or a paperclip) as possible. It evaluates four key areas generally, fluency, originality, flexibility and elaboration. In this work we only focus on originality [42].

Diversity Association Task (DAT) This task involves naming ten nouns that differ as much as possible from each other [42].

For RAT, we calculate the accuracy. For AUT and DAT we follow the originality calculations according to prior work [42]. For AUT, originality is how novel the response is compared to the prompt. For DAT, originality is how different a pair of words are in the response. For both of these tests, we use glove embeddings [31] and calculate cosine distance as measure of novelty. Since AUT only has 5 query words, we run each of these 10 times, to get an evaluation set of 50 instances. Similarly for DAT, we sample 50 responses from the model.

Table 24 shows examples of instances for each of the tests along with model responses and the quantitative scores. In Table 23 we summarize the performance for GPT-4.1-mini and GPT-5-mini on these three benchmarks. We see that flagship reasoning models achieve very high accuracy on RAT. For the diversity benchmarks AUT and DAT, we see both models scoring high on originality.

|Okay, let me try to figure out how to connect David Koechner to someone born in Newport Beach. First, I need to start with David Koechner, the actor. The target is someone whose place of birth is Newport Beach. So the paths need to end with that triple. First, the obvious connection: David Koechner was born in Denver, Colorado, so he's not from Newport Beach. So he's not directly connected. Therefore, the paths need to go through intermediaries. Let me think about his filmography. He's in "Anchorman," "Office Space," etc. Maybe someone he worked with was born in Newport Beach. For example, if he worked with an actor who was born there. But I don't know off the top of my head. Alternatively, maybe he's part of a production company that's based in Newport Beach? Not sure. Another angle: maybe he's associated with a charity or event in Newport Beach. For example, if he participated in a charity event there, then the event's organizer was born in Newport Beach. But that's a stretch. What about family connections? If he has a relative or a spouse who was born in Newport Beach. But I don't think his wife is from there. He married Rachel Dratch, a comedian. I don't know her birthplace, but if she's from there, that's a path. But I need to verify that. […..] Wait, maybe he has a co-star who was born in Newport Beach. Let's see. For example, in "Anchorman," the main cast includes Will Ferrell, Steve Carell, etc. I don't know their birthplaces. Maybe someone else in his filmography. For instance, in "The InLaws," he worked with Michael Douglas. Michael Douglas was born in Minneapolis. Not helpful. Maybe someone else.<br><br>Wait, Newport Beach is a city in California. Maybe someone he's worked with in a movie set in Newport Beach? For example, if a movie he was in was filmed there, and someone involved in the production was born there. But that's a bit indirect.<br><br>Let me think of specific people born in Newport Beach. For example, the actor Chris Pine was born in Los Angeles but maybe someone else. The singer Gwen Stefani was born in Alabama but raised in Orange County, which is near Newport Beach. Not exactly born there. Maybe the actor Steve McQueen? No, he was from Illinois. The surfer Laird Hamilton was born in California, but I think he's from Oxnard. Hmm.<br><br>Wait, maybe a director or producer he's worked with was born in Newport Beach. For example, Jay Roach directed "Anchorman," but he's from New York. Judd Apatow? He's from Massachusetts. Maybe someone else. […..]<br><br>Wait, maybe someone who directed a movie he was in was born in Newport Beach. For example, in "Meet the Parents," he was in that. Jay Roach directed, but he's from New York. The director of "Office Space" was Mike Judge, who was born in Dallas. Not helpful.<br><br>Alternatively, maybe a co-star in a lesser-known project. For example, in "The Legend of Bagger Vance," he had a role. The cast included Will Smith, Matt Damon, etc. But their birthplaces?<br><br>This is getting tricky. Maybe I need to consider less obvious connections. For example, if he's a fan of a local sports team in Newport Beach, and a player on that team was born there. For instance, the Newport Beach Yacht Club. If he's a member, and a member of the yacht club was born there. But again, no specific data. Another thought: maybe he has a friend who is a real estate agent in Newport Beach, and that agent was born there. But without specific info, this is too speculative.<br><br>I think I have to make do with what I can think of. Let me try to outline possible paths:<br><br>1. David Koechner is a co-star with [X], who was born in Newport Beach. (Need to find X)<br>2. David Koechner is part of a production company that has a connection to someone born in Newport Beach.<br>3. David Koechner attended an event where a participant was born in Newport Beach.<br>4. David Koechner is a friend of someone who was born in Newport Beach.<br>5. David Koechner's work in a movie/TV show was filmed in Newport Beach, and someone involved in that production was born there.<br>6. David Koechner is part of a charity organization that has a connection to Newport Beach, leading to someone born there.<br><br><br>But since I don't have specific data on X, these are all speculative. However, for the purpose of the task, I need to generate as many as possible, even if they are weaker.<br><br>Let me try to think of any possible X. For example, if David Koechner worked with the actor and comedian, maybe someone like... Let's see, in "Anchorman," there's David Spade, but he was born in New York. The actor Seth Rogen? No, he's from Canada. Maybe someone else in his filmography. In "The In-Laws," he worked with Albert Brooks, born in New York. Hmm.<br><br>if it's weak. Let's try to list possible paths:<br><br>1. David Koechner - co-star - [Unknown Actor X] - place of birth - Newport Beach.<br>2. David Koechner - worked on - [Movie/Show Y] - filmed in - Newport Beach - location scout - [Person Z] - place of birth Newport Beach.<br>3. David Koechner - friend - [Person A] - place of birth - Newport Beach.<br>4. David Koechner - member of - [Organization B] - member - [Person C] - place of birth - Newport Beach.<br>5. David Koechner - participated in - [Event D] - held in - Newport Beach - attendee - [Person E] - place of birth - Newport Beach.<br><br><br>But since I don't have concrete entities for X, Y, Z, A, B, C, D, E, these are placeholders. However, the task requires concrete entities. Therefore, I need to find real people or entities.|
|---|

##### Figure 9: Example of a reasoning chain for a query ‘What are possible connections between David Koechner and someone who was born in Newport Beach?’

###### Prompt K.1: Strength Prompt

Goal: Given a knowledge graph path consisting of multiple triples of the form (subject, predicate, object), estimate — FOR EACH TRIPLE — the number of members in the LARGER of two possible classes defined by the predicate. Formal Definitions (apply to each triple (s, p, o)): Define TWO classes.

- Class A (Subject-variation class):

- {{ x | (x, p, o) is true }} This is the set of all possible SUBJECTS x that could replace s while keeping (p, o) fixed. The given subject s counts as ONE MEMBER of this class.

Class B (Object-variation class):

- {{ y | (s, p, y) is true }} This is the set of all possible OBJECTS y that could replace o while keeping (s, p) fixed. The given object o counts as ONE MEMBER of this class. Directional Discipline (CRITICAL):

- - You MUST evaluate both Class A and Class B independently for each triple. - NEVER collapse or reinterpret the relation.
- - For directional predicates (e.g., ’influenced’, ’taught’, ’founded’, ’won’, ’received’):

- * Class A asks: who/what stands in relation (predicate) TO the object?
- * Class B asks: who/what the subject stands in relation (predicate) TO?

- - Always respect the original direction of the predicate. Procedure (apply to EACH triple in the path independently):

- 1. Identify Class A from (p, o).
- 2. Identify Class B from (s, p).
- 3. Use factual knowledge and reasonable estimation to estimate the size of EACH class.
- 4. Select the LARGER of the two estimated class sizes.
- 5. Explain your reasoning step by step for both classes. Output instructions:

- - Return valid JSON only.
- - The output must be a JSON list, with one object per triple, in the same order as the path.
- - Each object must contain exactly two keys:
- - “explanation": a natural-language explanation for that triple’s Class A and Class B reasoning.
- - “judgment": a single integer representing the estimated size of the LARGER class.
- - Do not include any text outside the JSON list.
- - Do not include markdown or extra keys.
- - Each judgment must be a single integer (no commas, no extra text). The output must exactly match this schema: {{ “explanation": “string", “judgment": integer }}

==== INPUT ==== Path: { path }

the American Academy of Arts and Sciences?’ by GPT-5 (med).

<answer>{

- “1”: [ [“Kareem Abdul-Jabbar”, “educated at”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “employs”, “Terence Tao”], [“Terence Tao”, “member of”, “American Academy of Arts and Sciences”] ],
- “2”: [ [“Kareem Abdul-Jabbar”, “educated at”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “employs”, “Andrea Ghez”], [“Andrea Ghez”, “member of”, “American Academy of Arts and Sciences”] ],
- “3”: [ [“Kareem Abdul-Jabbar”, “educated at”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “employs”, “Judea Pearl”], [“Judea Pearl”, “member of”, “American Academy of Arts and Sciences”] ],
- “4”: [ [“Kareem Abdul-Jabbar”, “educated at”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “employs”, “Leonard Kleinrock”], [“Leonard Kleinrock”, “member of”, “American Academy of Arts and Sciences”] ],
- “5”: [ [“Kareem Abdul-Jabbar”, “educated at”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “employs”, “Barbara Kruger”], [“Barbara Kruger”, “member of”, “American Academy of Arts and Sciences”] ],
- “6”: [ [“Kareem Abdul-Jabbar”, “educated at”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “employs”, “Catherine Opie”], [“Catherine Opie”, “member of”, “American Academy of Arts and Sciences”] ],
- “7”: [ [“Kareem Abdul-Jabbar”, “educated at”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “employs”, “Peter Sellars”], [“Peter Sellars”, “member of”, “American Academy of Arts and Sciences”] ],
- “8”: [ [“Kareem Abdul-Jabbar”, “educated at”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “employs”, “David Jewitt”], [“David Jewitt”, “member of”, “American Academy of Arts and Sciences”] ],
- “9”: [ [“Kareem Abdul-Jabbar”, “educated at”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “employs”, “Jared Diamond”], [“Jared Diamond”, “member of”, “American Academy of Arts and Sciences”] ],
- “10”: [ [“Kareem Abdul-Jabbar”, “educated at”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “chancellor”, “Gene D. Block”], [“Gene D. Block”, “member of”, “American Academy of Arts and Sciences”] ],
- “11”: [ [“Kareem Abdul-Jabbar”, “educated at”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “employs”, “Victoria Sork”], [“Victoria Sork”, “member of”, “American Academy of Arts and Sciences”] ],
- “12”: [ [“Kareem Abdul-Jabbar”, “educated at”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “named building”, “Eli and Edythe Broad Art Center”], [“Eli and Edythe Broad Art Center”, “named after”, “Eli Broad”], [“Eli Broad”, “member of”, “American Academy of Arts and Sciences”] ],
- “13”: [ [“Kareem Abdul-Jabbar”, “educated at”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “governed by”, “Regents of the University of California”], [“Regents of the University of California”, “member”, “Sherry Lansing”], [“Sherry Lansing”, “member of”, “American Academy of Arts and Sciences”] ],
- “14”: [ [“Kareem Abdul-Jabbar”, “award received”, “Presidential Medal of Freedom”], [“Presidential Medal of Freedom”, “awarded to”, “Toni Morrison”], [“Toni Morrison”, “member of”, “American Academy of Arts and Sciences”] ],
- “15”: [ [“Kareem Abdul-Jabbar”, “award received”, “Presidential Medal of Freedom”], [“Presidential Medal of Freedom”, “awarded to”, “Stephen Hawking”], [“Stephen Hawking”, “member of”, “American Academy of Arts and Sciences”] ],
- “16”: [ [“Kareem Abdul-Jabbar”, “award received”, “Presidential Medal of Freedom”], [“Presidential Medal of Freedom”, “awarded to”, “Sandra Day O’Connor”], [“Sandra Day O’Connor”, “member of”, “American Academy of Arts and Sciences”] ],
- “17”: [ [“Kareem Abdul-Jabbar”, “award received”, “Presidential Medal of Freedom”], [“Presidential Medal of Freedom”, “awarded to”, “Tom Hanks”], [“Tom Hanks”, “member of”, “American Academy of Arts and Sciences”] ],
- “18”: [ [“Kareem Abdul-Jabbar”, “award received”, “Presidential Medal of Freedom”], [“Presidential Medal of Freedom”, “awarded to”, “Bill Gates”], [“Bill Gates”, “member of”, “American Academy of Arts and Sciences”] ],
- “19”: [ [“Kareem Abdul-Jabbar”, “award received”, “Presidential Medal of Freedom”], [“Presidential Medal of Freedom”, “awarded to”, “Melinda French Gates”], [“Melinda French Gates”, “member of”, “American Academy of Arts and Sciences”] ], }< /answer>

connecting Kareem Abdul−Jabbar, the legendary basketball player, and someone who is a member of the American Academy of Arts and Sciences?’ by Claude-Haiku-4.5 (med). The output has the summarized reasoning trace from the API along with the final output.

<think> I need to find connection paths from Kareem Abdul−Jabbar to someone who is a member of the American Academy of Arts and Sciences. Let me think about what I know:

1. Kareem Abdul−Jabbar is a legendary basketball player 2. I need to find paths that connect him to people who are members of the American Academy of Arts and Sciences (AAAS) Some people I know who are members of AAAS: - Many academics, artists, scientists - Possibly some celebrities who have also become academically distinguished Let me think about Kareem Abdul−Jabbar’s connections: - He’s been in movies and TV - He wrote books - He was interviewed by many people - He’s been involved in various charitable and cultural activities - He’s coached basketball - He was in the NBA Some notable connections: 1. Kareem appeared in movies - could he have appeared with someone who’s in AAAS? 2. Kareem wrote books - could he have a publisher or editor in AAAS? 3. Kareem could have worked with coaches or teammates 4. Kareem could have connections through cultural/civil rights work Let me think of specific people in AAAS: - Spike Lee (filmmaker) - might be in AAAS? I should verify. Actually, I’m not certain about this. - Bill Cosby appeared with Kareem - but Bill Cosby is controversial - Other athletes who also became scholars? Actually, let me think more carefully: - Kareem wrote a book called “Giant Steps“ - who might have worked on this with him who’s in AAAS? - Kareem appeared in “Airplane¡‘ - Robert Hays, Julie Hagerty - not sure if any are in AAAS One strong connection I can think of: - Kareem Abdul-Jabbar worked with/appeared in films - Some filmmakers/actors might be in AAAS Let me think about less obvious connections:

- His involvement in civil rights and cultural initiatives could connect him to prominent scholars and intellectuals who are AAAS members. He may have received honorary degrees from universities, which would put him in contact with academic leadership. University presidents and prominent faculty are often AAAS members, so these institutional connections could provide pathways. Looking at specific connections, I’m fairly confident about some of Kareem’s documented activities—his film appearances like Airplane!, his published books, his NBA coaching role, and his media interviews. However, I’m uncertain about direct AAAS membership among people he’s worked with without being able to verify those details. Some potential connections exist through cultural figures like Spike Lee if they collaborated, or through the various academics and journalists he may have interacted with, but I can’t confirm these with certainty. The most reliable paths would likely run through his university affiliations and the prominent institutional leaders he’s encountered through his honorary degrees and educational involvement. Been in a Spike Lee film or interviewed by him, though I’m uncertain about the specifics. Let me consider academic routes instead—Kareem wrote extensively on Islam, history, and culture, so he likely connected with scholars who could be AAAS members. I should stick to what I can verify: his appearance in Airplane!, his published books, and his cultural influence. The strongest path would trace through someone whose AAAS membership I’m confident about rather than speculating on uncertain connections. Potential collaboration routes exist through athletes, filmmakers, authors, and journalists he’s worked with, but without verified information I need to be careful. If Spike Lee is an AAAS member, a connection through film work could work—Kareem to a Spike Lee project to Lee to the Academy. Alternatively, through his writing: Kareem’s books could connect him to publishers or editors with Academy ties. < /think> <answer> “‘json { “1”: [ (“Kareem Abdul-Jabbar”, “appeared in”, “Airplane!”), (“Airplane!”, “directed by”, “Jim Abrahams”), (“Jim Abrahams”, “member of”, “American Academy of Arts and Sciences”) ],

- “2”: [ (“Kareem Abdul-Jabbar”, “appeared in”, “Airplane”), (“Airplane”, “directed by”, “David Zucker”), (“David Zucker”, “member of”, “American Academy of Arts and Sciences”) ],
- “3”: [ (“Kareem Abdul-Jabbar”, “authored”, “Giant Steps”), (“Giant Steps”, “published by”, “Bantam Books”), (“Bantam Books”, “parent company”, “Penguin Random House”), (“Penguin Random House”, “employs”, “Distinguished Editors”), (“Salman Rushdie”, “published by”, “Penguin Random House”), (“Salman Rushdie”, “member of”, “American Academy of Arts and Sciences”) ],
- “4”: [ (“Kareem Abdul-Jabbar”, “interviewed by”, “Charlie Rose”), (“Charlie Rose”, “member of”, “American Academy of Arts and Sciences”) ],
- “5”: [ (“Kareem Abdul-Jabbar”, “appeared in”, “Mycroft Holmes”), (“Mycroft Holmes”, “television network”, “PBS”), (“PBS”, “broadcasts”, “American Masters”), (“American Masters”, “features”, “Cultural figures”), (“Frank McCourt”, “featured in”, “American Masters”), (“Frank McCourt”, “member of”, “American Academy of Arts and Sciences”) ],
- “6”: [ (“Kareem Abdul-Jabbar”, “wrote foreword for”, “Books on history and culture”), (“Books on history and culture”, “reviewed by”, “Academic scholars”), (“Henry Louis Gates Jr.”, “scholar”, “Harvard University”), (“Henry Louis Gates Jr.”, “member of”, “American Academy of Arts and Sciences”) ] } ”’ < /answer>

the American Academy of Arts and Sciences?’ by Gemini-3-Pro.

<answer> { “1”: [ [“Kareem Abdul-Jabbar”, “recipient of”, “Presidential Medal of Freedom”], [“Presidential Medal of Freedom”, “award received by”, “Barack Obama”], [“Barack Obama”, “member of”, “American Academy of Arts and Sciences”] ],

- “2”: [ [“Kareem Abdul-Jabbar”, “cast member of”, “Airplane!”], [“Airplane!”, “cast member”, “Leslie Nielsen”], [“Leslie Nielsen”, “brother of”, “Erik Nielsen”], [“Erik Nielsen”, “colleague of”, “Brian Mulroney”], [“Brian Mulroney”, “member of”, “American Academy of Arts and Sciences”] ],
- “3”: [ [“Kareem Abdul-Jabbar”, “played for”, “Los Angeles Lakers”], [“Los Angeles Lakers”, “owned by”, “Jerry Buss”], [“Jerry Buss”, “father of”, “Jeanie Buss”], [“Jeanie Buss”, “partner of”, “Phil Jackson”], [“Phil Jackson”, “coached”, “Michael Jordan”], [“Michael Jordan”, “recipient of”, “Presidential Medal of Freedom”], [“Presidential Medal of Freedom”, “award received by”, “Bill Gates”], [“Bill Gates”, “member of”, “American Academy of Arts and Sciences”] ],
- “4”: [ [“Kareem Abdul-Jabbar”, “author of”, “Writings on the Wall”], [“Writings on the Wall”, “publisher”, “Time”], [“Time”, “founded by”, “Henry Luce”], [“Henry Luce”, “member of”, “American Academy of Arts and Sciences”] ],
- “5”: [ [“Kareem Abdul-Jabbar”, “educated at”, “UCLA”], [“UCLA”, “faculty member”, “Jared Diamond”], [“Jared Diamond”, “member of”, “American Academy of Arts and Sciences”] ],
- “6”: [ [“Kareem Abdul-Jabbar”, “co-star of”, “Game of Death”], [“Game of Death”, “starring”, “Bruce Lee”], [“Bruce Lee”, “student of”, “Ip Man”], [“Ip Man”, “portrayed by”, “Tony Leung Chiu-wai”], [“Tony Leung Chiu-wai”, “cast member of”, “Shang-Chi and the Legend of the Ten Rings”], [“Shang-Chi and the Legend of the Ten Rings”, “cast member”, “Michelle Yeoh”], [“Michelle Yeoh”, “member of”, “American Academy of Arts and Sciences”] ],
- “7”: [ [“Kareem Abdul-Jabbar”, “played for”, “Milwaukee Bucks”], [“Milwaukee Bucks”, “owned by”, “Herb Kohl”], [“Herb Kohl”, “served in”, “United States Senate”], [“United States Senate”, “member”, “Hillary Clinton”], [“Hillary Clinton”, “member of”, “American Academy of Arts and Sciences”] ],
- “8”: [ [“Kareem Abdul-Jabbar”, “appointee of”, “Hillary Clinton”], [“Hillary Clinton”, “member of”, “American Academy of Arts and Sciences”] ],
- “9”: [ [“Kareem Abdul-Jabbar”, “writer for”, “Veronica Mars”], [“Veronica Mars”, “created by”, “Rob Thomas”], [“Rob Thomas”, “collaborated with”, “Stephen King”], [“Stephen King”, “member of”, “American Academy of Arts and Sciences”] ],
- “10”: [ [“Kareem Abdul-Jabbar”, “member of”, “Naismith Memorial Basketball Hall of Fame”], [“Naismith Memorial Basketball Hall of Fame”, “inductee”, “Bill Bradley”], [“Bill Bradley”, “member of”, “American Academy of Arts and Sciences”] ],
- “11”: [ [“Kareem Abdul-Jabbar”, “competed against”, “Larry Bird”], [“Larry Bird”, “teammate of”, “Bill Walton”], [“Bill Walton”, “played for”, “Boston Celtics”], [“Boston Celtics”, “fan of”, “Ben Affleck”], [“Ben Affleck”, “directed”, “Argo”], [“Argo”, “produced by”, “George Clooney”], [“George Clooney”, “member of”, “American Academy of Arts and Sciences”] ],
- “12”: [ [“Kareem Abdul-Jabbar”, “subject of”, “Kareem: Minority of One”], [“Kareem: Minority of One”, “narrator”, “Liev Schreiber”], [“Liev Schreiber”, “cast member of”, “Spotlight”], [“Spotlight”, “cast member”, “Mark Ruffalo”], [“Mark Ruffalo”, “cast member of”, “The Avengers”], [“The Avengers”, “cast member”, “Robert Downey Jr.”], [“Robert Downey Jr.”, “collaborated with”, “Mel Gibson”], [“Mel Gibson”, “directed”, “Hacksaw Ridge”], [“Hacksaw Ridge”, “cast member”, “Andrew Garfield”], [“Andrew Garfield”, “cast member of”, “The Social Network”], [“The Social Network”, “screenwriter”, “Aaron Sorkin”], [“Aaron Sorkin”, “member of”, “American Academy of Arts and Sciences”] ],
- “13”: [ [“Kareem Abdul-Jabbar”, “coach of”, “Los Angeles Lakers”], [“Los Angeles Lakers”, “player”, “Kobe Bryant”], [“Kobe Bryant”, “collaborated with”, “John Williams”], [“John Williams”, “member of”, “American Academy of Arts and Sciences”] ],
- “14”: [ [“Kareem Abdul-Jabbar”, “guest star on”, “The Big Bang Theory”], [“The Big Bang Theory”, “created by”, “Chuck Lorre”], [“Chuck Lorre”, “collaborated with”, “Alan Alda”], [“Alan Alda”, “member of”, “American Academy of Arts and Sciences”] ],
- “15”: [ [“Kareem Abdul-Jabbar”, “born in”, “New York City”], [“New York City”, “mayor”, “Michael Bloomberg”], [“Michael Bloomberg”, “member of”, “American Academy of Arts and Sciences”] ],
- “16”: [ [“Kareem Abdul-Jabbar”, “recipient of”, “Double Helix Medal”], [“Double Helix Medal”, “awarded by”, “Cold Spring Harbor Laboratory”], [“Cold Spring Harbor Laboratory”, “chancellor emeritus”, “James Watson”], [“James Watson”, “member of”, “American Academy of Arts and Sciences”] ],
- “17”: [ [“Kareem Abdul-Jabbar”, “collaborated with”, “Raymond Obstfeld”], [“Raymond Obstfeld”, “faculty at”, “Orange Coast College”], [“Orange Coast College”, “located in”, “Costa Mesa”], [“Costa Mesa”, “headquarters of”, “South Coast Repertory”], [“South Coast Repertory”, “commissioned work by”, “Amy Freed”], [“Amy Freed”, “colleague of”, “Tony Kushner”], [“Tony Kushner”, “member of”, “American Academy of Arts and Sciences”] ],
- “18”: [ [“Kareem Abdul-Jabbar”, “drafted by”, “Harlem Globetrotters”], [“Harlem Globetrotters”, “honorary member”, “Henry Kissinger”], [“Henry Kissinger”, “member of”, “American Academy of Arts and Sciences”] ], } < /answer>

- Table 15: Examples of model-generated paths u along with the strength scores, for the query: ‘What are different ways of connecting Kareem Abdul-Jabbar, the legendary basketball player, and someone who is a member of the American Academy of Arts and Sciences?’. All models are prompted with the default reasoning effort (medium).

Model Path σ GPT-5 (kareem abdul-jabbar, member of, national basketball players association),

(national basketball players association, led by, michele roberts), (michele roberts, member of, american academy of arts and sciences)

4

(kareem abdul-jabbar, educated at, university of california, los angeles), (university of california, los angeles, employs, terence tao), (terence tao, member of, american academy of arts and sciences)

1

Gemini-3-Pro (kareem abdul-jabbar, featured in, black patriotism), (black patriotism, author, cornel west), (cornel west, member of, american academy of arts and sciences)

4

(kareem abdul-jabbar, educated at, ucla), (ucla, faculty member, jared diamond), (jared diamond, member of, american academy of arts and sciences)

1

Claude-Haiku-4.5 (kareem abdul-jabbar, appeared in, airplane!), (airplane!, directed by, jim abrahams), (jim abrahams, member of, american academy of arts and sciences)

4

(kareem abdul-jabbar, wrote foreword for, books on history and culture), (books on history and culture, reviewed by, academic scholars), (henry louis gates jr., scholar, harvard university), (henry louis gates jr., member of, american academy of arts and sciences)

1

- Table 16: Model performance on CREATE. We report creative utility scores (s) at two patience levels

= 0.7 and 0.9. The value in round brackets next to a model name indicates the reasoning effort/budget, for models where applicable. We also report the number of factual paths (|U|(factual)), average maximum quality, average pairwise distance, average path length i.e. number of triples per path and average number of tokens in the output generated.

Model s0.7 s0.9 |U| (factual) σ d path length num_tokens Olmo-3.1-32B-Instruct 3.77(3.58) 4.13(4.34) 2.27(3.01) 2.26(1.15) 0.50(0.39) 3.10(0.89) 863(387) Olmo-3.1-32B-Think (16k) 4.78(3.96) 5.25(4.95) 2.16(3.22) 2.81(1.25) 0.45(0.38) 2.40(0.82) 10370(3594) Olmo-3.1-32B-Think (32k) 4.97(4.24) 5.52(5.35) 2.32(3.41) 2.89(1.24) 0.44(0.37) 2.48(0.89) 10334(3634) Qwen3-30B-Instruct 5.20(4.60) 6.27(6.42) 5.12(6.99) 2.43(0.99) 0.53(0.33) 3.32(0.99) 1938(646) Qwen_Qwen3-32B (16k) 4.69(3.88) 5.08(4.64) 2.11(2.38) 2.77(1.06) 0.53(0.40) 2.84(0.69) 3207(1292) Qwen_Qwen3-32B (32k) 4.71(3.77) 5.11(4.56) 2.18(2.42) 2.80(1.06) 0.55(0.40) 2.82(0.70) 3225(1562)

gpt-oss-120b (high) 7.63(4.50) 9.05(6.27) 5.79(4.26) 2.66(0.93) 0.67(0.30) 3.06(0.59) 4484(1321) gpt-oss-120b (med) 7.59(4.57) 9.04(6.39) 5.78(4.31) 2.64(0.95) 0.66(0.31) 3.04(0.59) 4480(1307) gpt-oss-120b (low) 7.63(4.64) 9.12(6.55) 5.63(4.18) 2.68(0.95) 0.65(0.32) 3.08(0.61) 4287(1226)

Claude-Haiku-4.5 (med) 4.84(3.87) 5.30(4.67) 2.84(3.01) 2.61(1.16) 0.48(0.36) 3.19(0.99) 1599(511) Claude-Sonnet-4.6 (high) 9.71(5.60) 13.41(9.69) 11.52(9.59) 2.88(0.92) 0.64(0.27) 3.54(0.84) 12063(12955) Claude-Opus-4.7 (high) 7.34(4.99) 8.74(7.00) 5.07(4.49) 3.09(1.12) 0.57(0.33) 2.86(0.63) 2074(1710) Claude-Opus-4.7 (xhigh) 8.12(5.36) 10.01(7.87) 6.39(5.21) 3.08(1.13) 0.58(0.31) 2.92(0.64) 4163(3534)

- GPT-4.1 7.49(5.25) 9.39(8.01) 6.04(5.28) 2.73(0.97) 0.65(0.33) 3.16(0.77) 1076(430)

- GPT-4.1-mini 6.15(5.08) 7.16(6.81) 3.56(3.72) 2.89(1.01) 0.61(0.37) 2.87(0.71) 795(260) GPT-5-mini (high) 7.83(4.95) 10.16(7.85) 15.34(10.70) 2.46(1.09) 0.51(0.29) 3.60(0.92) 23370(5683) GPT-5-mini (med) 7.09(4.61) 8.54(6.56) 7.87(5.54) 2.68(1.11) 0.53(0.32) 3.34(0.80) 6351(1746) GPT-5-mini (low) 6.21(4.19) 7.03(5.40) 4.85(3.78) 2.73(1.12) 0.51(0.34) 3.24(0.77) 1910(483) GPT-5 (med) 8.98(5.11) 12.03(8.67) 17.79(14.02) 2.59(1.07) 0.54(0.27) 4.06(1.17) 18245(5822)

- GPT-5.4 (med) 8.98(5.30) 12.09(8.86) 21.19(14.45) 2.65(1.07) 0.52(0.26) 4.10(1.15) 14721(4166)

- GPT-5.5 (med) 9.71(5.41) 13.48(9.44) 22.62(16.03) 2.67(1.04) 0.56(0.26) 4.05(1.16) 13557(4948)

Gemini-3-Pro (med) 8.29(5.19) 10.41(7.95) 5.85(4.96) 3.01(1.00) 0.67(0.32) 4.09(1.52) 1764(1069) Gemini-3.1-Pro (med) 7.37(4.83) 8.73(6.67) 4.61(3.55) 2.92(1.09) 0.66(0.33) 3.42(1.18) 3200(1669)

- Table 17: Prompt variation: ‘creative’. Example of raw model-generated output for the query: ‘What are different ways of connecting Kareem Abdul-Jabbar, the legendary basketball player, and someone who is a member of the American Academy of Arts and Sciences?’ by GPT-5-mini.

<answer> { 1: [ [“Kareem Abdul-Jabbar”, “played for”, “UCLA Bruins men’s basketball team”], [“UCLA Bruins men’s basketball team”, “coached by”, “John Wooden”], [“John Wooden”, “member of”, “American Academy of Arts and Sciences”] ],

- 2: [ [“Kareem Abdul-Jabbar”, “played for”, “Los Angeles Lakers”], [“Los Angeles Lakers”, “teammate”, “Magic Johnson (basketball)”], [“Magic Johnson (basketball)”, “member of”, “American Academy of Arts and Sciences”] ],
- 3: [ [“Kareem Abdul-Jabbar”, “acted in”, “Airplane! (film)”], [“Airplane! (film)”, “co-star”, “Peter Graves”], [“Peter Graves”, “member of”, “American Academy of Arts and Sciences”] ],
- 4: [ [“Kareem Abdul-Jabbar”, “authored”, “Becoming Kareem: Growing Up On and Off the Court”], [“Becoming Kareem: Growing Up On and Off the Court”, “co-authored by”, “Raymond Obstfeld”], [“Raymond Obstfeld”, “member of”, “American Academy of Arts and Sciences”] ],
- 5: [ [“Kareem Abdul-Jabbar”, “wrote for”, “Time (magazine)”], [“Time (magazine)”, “employed”, “Henry Louis Gates Jr.”], [“Henry Louis Gates Jr.”, “member of”, “American Academy of Arts and Sciences”] ],
- 6: [ [“Kareem Abdul-Jabbar”, “appeared on”, “The Oprah Winfrey Show”], [“The Oprah Winfrey Show”, “hosted by”, “Oprah Winfrey”], [“Oprah Winfrey”, “member of”, “American Academy of Arts and Sciences”] ],
- 7: [ [“Kareem Abdul-Jabbar”, “appeared on”, “60 Minutes”], [“60 Minutes”, “interviewed by”, “Morley Safer”], [“Morley Safer”, “member of”, “American Academy of Arts and Sciences”] ],
- 8: [ [“Kareem Abdul-Jabbar”, “participated in”, “NBA All-Star Game”], [“NBA All-Star Game”, “featured”, “Bill Russell”], [“Bill Russell”, “member of”, “American Academy of Arts and Sciences”] ],
- 9: [ [“Kareem Abdul-Jabbar”, “wrote foreword for”, “a book by”], [“a book by”, “authored by”, “Toni Morrison”], [“Toni Morrison”, “member of”, “American Academy of Arts and Sciences”] ], 10: [ [“Kareem Abdul-Jabbar”, “appeared in”, “documentary film”], [“documentary film”, “directed by”, “Spike Lee”], [“Spike Lee”, “member of”, “American Academy of Arts and Sciences”] ],

- 11: [ [“Kareem Abdul-Jabbar”, “worked with”, “United States Department of State (cultural diplomacy)”], [“United States Department of State”, “partnered with”, “Yo-Yo Ma”], [“Yo-Yo Ma”, “member of”, “American Academy of Arts and Sciences”] ],
- 12: [ [“Kareem Abdul-Jabbar”, “interviewed by”, “Henry Louis Gates Jr.”], [“Henry Louis Gates Jr.”, “colleague of”, “Toni Morrison”], [“Toni Morrison”, “member of”, “American Academy of Arts and Sciences”] ],
- 13: [ [“Kareem Abdul-Jabbar”, “served on board of”, “A non-profit organization”], [“A non-profit organization”, “board member”, “Quincy Jones”], [“Quincy Jones”, “member of”, “American Academy of Arts and Sciences”] ],
- 14: [ [“Kareem Abdul-Jabbar”, “appeared at”, “Harvard University event”], [“Harvard University event”, “hosted by”, “Henry Louis Gates Jr.”], [“Henry Louis Gates Jr.”, “member of”, “American Academy of Arts and Sciences”] ],
- 15: [ [“Kareem Abdul-Jabbar”, “wrote essays for”, “The New York Times”], [“The New York Times”, “employed”, “Tom Hanks (as contributor)”], [“Tom Hanks”, “member of”, “American Academy of Arts and Sciences”] ],
- 16: [ [“Kareem Abdul-Jabbar”, “appeared on stage with”, “civil rights leaders”], [“civil rights leaders”, “included”, “Maya Angelou”], [“Maya Angelou”, “member of”, “American Academy of Arts and Sciences”] ],
- 17: [ [“Kareem Abdul-Jabbar”, “participated in”, “literary festival”], [“literary festival”, “featured”, “Margaret Atwood”], [“Margaret Atwood”, “member of”, “American Academy of Arts and Sciences”] ],
- 18: [ [“Kareem Abdul-Jabbar”, “collaborated with”, “sports historians”], [“sports historians”, “included”, “David McCullough”], [“David McCullough”, “member of”, “American Academy of Arts and Sciences”] ],
- 19: [ [“Kareem Abdul-Jabbar”, “appeared on”, “public radio program”], [“public radio program”, “hosted by”, “Ira Glass”], [“Ira Glass”, “member of”, “American Academy of Arts and Sciences”] ],
- 20: [ [“Kareem Abdul-Jabbar”, “supported”, “arts education initiatives”], [“arts education initiatives”, “partnered with”, “Meryl Streep”], [“Meryl Streep”, “member of”, “American Academy of Arts and Sciences”] ] } < /answer>

- Table 18: Prompt variation: ‘verbalized sampling’. Example of raw model-generated output for the query: ‘What are different ways of connecting Kareem Abdul-Jabbar, the legendary basketball player, and someone who is a member of the American Academy of Arts and Sciences?’ by GPT-5-mini.

<answer> { “1”: { “path_probability”: 0.34, “path”: [ [“Kareem Abdul-Jabbar”, “member of”, “American Academy of Arts and Sciences”] ] },

- “2”: { “path_probability”: 0.24, “path”: [ [“Kareem Abdul-Jabbar”, “attended”, “University of California, Los Angeles”], [“University of California, Los Angeles”, “employs”, “Terence Tao”], [“Terence Tao”, “member of”, “American Academy of Arts and Sciences”] ] },
- “3”:{ “path_probability”: 0.16, “path”: [ [“Kareem Abdul-Jabbar”, “coached by”, “John Wooden”], [“John Wooden”, “member of”, “American Academy of Arts and Sciences”] ] },
- “4”: { “path_probability”: 0.15, “path”: [ [“Kareem Abdul-Jabbar”, “received award from”, “Barack Obama”], [“Barack Obama”, “member of”, “American Academy of Arts and Sciences”] ] },
- “5”: { “path_probability”: 0.11, “path”: [ [“Kareem Abdul-Jabbar”, “interviewed by”, “Henry Louis Gates Jr.”], [“Henry Louis Gates Jr.”, “member of”, “American Academy of Arts and Sciences”] ] } } < /answer>

- Table 19: Utility (s), max σ, average pairwise distance d, number of valid and factual paths, and average number of tokens for models outputs based on the prompt variations. Model ID variation s0.7 s0.9 |U| σ d path length num tokens

Claude-haiku-4-5 (med)

original 4.84(3.87) 5.30(4.67) 2.84(3.01) 2.61(1.16) 0.48(0.36) 3.19(0.99) 1599(511) creative 4.84(3.91) 5.32(4.79) 2.83(3.13) 2.64(1.14) 0.49(0.36) 3.11(0.95) 1633(533) verbalized 4.42(3.48) 4.72(4.06) 2.33(2.24) 2.58(1.17) 0.47(0.37) 3.05(0.94) 1677(489) iterate 7.57(5.18) 9.84(8.11) 9.37(9.43) 2.45(0.97) 0.60(0.31) 3.33(0.93) 1599(511)

- GPT-4.1 original 7.49(5.25) 9.39(8.01) 6.04(5.28) 2.73(0.97) 0.65(0.33) 3.16(0.77) 1076(430) creative 7.84(5.40) 9.94(8.23) 6.23(5.48) 2.72(0.94) 0.66(0.33) 3.18(0.79) 1110(416) verbalized 5.63(4.32) 6.26(5.37) 3.02(2.64) 2.77(1.05) 0.59(0.39) 3.08(0.83) 632(191) iterate 9.67(5.73) 13.84(10.38) 11.29(9.18) 2.69(0.87) 0.70(0.29) 3.24(0.77) 989(355) resample 11.02(5.48) 16.04(10.42) 18.58(14.87) 2.68(0.85) 0.69(0.26) 3.22(1.10) 3269(1104)

- GPT-4.1-mini original 6.15(5.08) 7.16(6.81) 3.56(3.72) 2.89(1.01) 0.61(0.37) 2.87(0.71) 795(260) creative 6.16(5.23) 7.26(7.20) 3.59(3.97) 2.95(1.05) 0.59(0.38) 2.89(0.72) 845(266) verbalized 4.35(3.94) 4.62(4.50) 1.83(2.01) 2.99(1.15) 0.52(0.41) 2.89(0.76) 535(138) iterate 8.18(5.93) 10.69(9.50) 6.75(6.77) 2.87(0.93) 0.65(0.34) 2.90(0.66) 726(232) resample 9.48(5.76) 12.76(9.80) 10.75(10.05) 2.80(0.92) 0.65(0.31) 2.86(0.64) 2367(687)

GPT-5-mini (med) original 7.09(4.61) 8.54(6.56) 7.87(5.54) 2.68(1.11) 0.53(0.32) 3.34(0.80) 6351(1746) creative 7.11(4.54) 8.56(6.43) 8.16(5.72) 2.66(1.10) 0.53(0.31) 3.36(0.80) 6418(1802) verbalized 6.55(4.33) 7.52(5.66) 5.71(3.94) 2.68(1.11) 0.54(0.33) 3.19(0.79) 6249(1561) iterate 10.04(5.13) 14.20(9.18) 16.35(10.33) 2.56(0.90) 0.66(0.28) 3.35(0.71) 6351(1746)

Olmo-3.1-32B Instruct

original 3.77(3.58) 4.13(4.34) 2.27(3.01) 2.26(1.15) 0.50(0.39) 3.10(0.89) 863(387) creative 3.67(3.60) 4.04(4.41) 2.28(3.14) 2.21(1.11) 0.49(0.39) 3.08(0.86) 882(365) verbalized 2.70(2.61) 2.79(2.85) 1.21(1.55) 2.21(1.23) 0.40(0.41) 3.05(0.95) 604(435) iterate 5.07(4.30) 5.98(5.95) 4.74(5.65) 2.15(1.02) 0.56(0.35) 3.21(0.85) 797(572) resample 6.11(4.76) 7.41(6.76) 6.74(7.48) 2.25(1.04) 0.58(0.33) 3.16(0.88) 2539(662)

Olmo-3.1-32B Think (16k)

resample 6.71(5.22) 8.14(7.45) 5.99(7.84) 2.80(1.14) 0.53(0.34) 2.47(0.88) 30939(8503) creative 4.79(4.07) 5.25(4.97) 2.08(3.17) 2.83(1.22) 0.46(0.37) 2.45(0.80) 10747(3506) verbalized 4.53(3.53) 4.77(4.04) 1.46(2.05) 2.90(1.18) 0.46(0.39) 2.31(0.82) 10002(3344) iterate 6.08(4.72) 7.11(6.44) 3.84(5.21) 2.86(1.19) 0.49(0.35) 2.53(0.79) 8435(3326)

- original 4.78(3.96) 5.25(4.95) 2.16(3.22) 2.81(1.25) 0.45(0.38) 2.40(0.82) 10370(3594)

Qwen3-30B Instruct

- original 5.20(4.60) 6.27(6.42) 5.12(6.99) 2.43(0.99) 0.53(0.33) 3.32(0.99) 1938(646)

creative 5.34(4.56) 6.45(6.47) 3.87(6.26) 2.35(0.92) 0.52(0.32) 3.47(1.05) 2563(1074) verbalized 3.66(3.46) 3.96(4.08) 1.77(2.70) 2.43(1.08) 0.47(0.38) 3.41(0.85) 1508(1172) iterate 6.24(5.16) 8.12(7.99) 9.84(13.31) 2.47(0.99) 0.52(0.32) 3.31(0.96) 1933(672) resample 7.10(5.24) 9.34(8.34) 11.00(13.71) 2.43(0.93) 0.56(0.31) 3.40(0.94) 7834(2209)

Qwen3-32B (16k) original 4.69(3.88) 5.08(4.64) 2.11(2.38) 2.77(1.06) 0.53(0.40) 2.84(0.69) 3207(1292) creative 4.73(3.92) 5.14(4.66) 2.14(2.38) 2.75(1.03) 0.54(0.41) 2.81(0.68) 3257(1409) verbalized 4.03(3.53) 4.26(4.05) 1.53(1.85) 2.85(1.13) 0.51(0.43) 2.64(0.73) 3041(1261) iterate 6.38(4.71) 7.52(6.48) 4.10(4.34) 2.79(0.98) 0.58(0.36) 2.91(0.66) 2166(1221) resample 2.99(2.96) 3.10(3.20) 0.11(0.66) 2.40(1.03) 0.47(0.41) 2.95(0.90) 9646(3168)

- Table 20: Performance on CREATE if we do not include factuality in the quality objective. There is a trade off between factuality and creative utiilty.

s0.9 |U| σ d path length factuality num tokens

Claude-Opus-4.7 (high) 13.27(7.35) 9.02(4.61) 3.11(1.06) 0.65(0.26) 2.90(0.62) 0.77(0.19) 2074(1710) Claude-Opus-4.7 (xhigh) 15.12(8.07) 11.39(5.17) 3.15(1.04) 0.65(0.24) 2.95(0.59) 0.77(0.18) 4163(3534) Claude-Sonnet-4.6 (high) 19.30(9.15) 20.43(9.82) 2.88(0.81) 0.69(0.22) 3.61(0.83) 0.81(0.15) 12063(12955) GPT-5 (med) 15.40(9.00) 28.52(15.12) 2.61(1.04) 0.56(0.24) 4.08(1.16) 0.86(0.13) 18245(5822) GPT-5-mini (high) 14.54(10.29) 24.07(10.91) 2.50(1.05) 0.54(0.28) 3.60(0.90) 0.86(0.19) 23370(5683)

- GPT-5.4 (high) 16.26(9.22) 37.28(13.11) 2.67(1.04) 0.56(0.23) 4.19(1.24) 0.84(0.14) 14721(4166)

- GPT-5.5 (high) 18.02(9.84) 40.17(16.05) 2.70(1.01) 0.60(0.23) 4.15(1.25) 0.83(0.14) 13557(4948) Gemini-3-pro 18.70(8.63) 14.10(5.82) 3.02(0.87) 0.73(0.22) 4.56(2.24) 0.76(0.15) 1764(1069) Gemini-3.1-Pro 13.89(7.11) 8.65(3.31) 2.93(1.00) 0.73(0.24) 3.59(1.25) 0.80(0.16) 3200(1669) Olmo-3.1-32B-Think (32k) 9.55(6.45) 4.90(5.18) 2.99(1.16) 0.51(0.33) 2.48(0.91) 0.66(0.30) 10334(3634) Qwen_Qwen3-32B (32k) 13.15(5.82) 6.89(3.29) 2.88(0.84) 0.73(0.27) 3.00(0.73) 0.60(0.22) 3225(1562) gpt-oss-120b (high) 16.92(6.60) 12.11(3.47) 2.78(0.69) 0.75(0.21) 3.11(0.54) 0.73(0.19) 4484(1321)

- Table 21: We compare how much a model generated path differs from the population of paths produced by all other models. We report the average of the distance between a path and the closest path in the population along with the average strength and then number of paths considered.

Model Variation avg distance avg σ num Gemini-3-pro original 0.04 3.77 223 GPT-5 original 0.02 3.73 224 GPT-4.1 original 0.03 3.76 215

creative 0.028 3.793 216 verbalized 0.020 3.820 184 iterate 0.053 3.600 205 resample 0.026 3.754 219

GPT-5-mini original 0.02 3.76 208 creative 0.03 3.75 216 verbalized 0.02 3.76 214 iterate 0.06 3.60 225

- Table 23: Human creativity test, example, number of instances, and LLM performance on these tests. We only look at average accuracy and average originality for RAT and AUT/DAT respectively.

Test Num points Metric GPT-4.1-mini GPT-5-mini RAT 212 accuracy 0.48 (0.5) 0.86 (0.35)

5 words / 10 samples each

AUT

originality 0.81 (0.06) 0.80 (0.09) DAT 50 samples originality 0.84 (0.01) 0.82 (0.03)

- Table 24: Qualitative examples for GPT-5-mini (gpt-5-mini-2025-08-07). AUT and DAT scores are originality; RAT score is accuracy.

Test Instruction (full prompt) Model response Score

RAT Find a fourth word that is related to all three of the following words: cottage / swiss / cake Respond with only the single word answer, nothing else.

cheese 1.0

AUT Imagine that someone gives you a book. Write down as many uses as you can think of for this object, up to 10 uses. Please list the uses as words or phrases (single word answers are ok), separated by semicolons. Do not write anything besides your proposed uses.

Reading; Doorstop; Paperweight; Coaster; Gift; Kindling; Art canvas; Pressing flowers; Prop for photos; Shelf divider

0.83

DAT Instructions: Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words. Rules: 1. Only single words in English. 2. Only nouns (e.g., things, objects, concepts). 3. No proper nouns (e.g., no specific people or places). 4. No specialised vocabulary (e.g., no technical terms). 5. Think of the words on your own (e.g., do not just look at objects in your surroundings). 6. Complete this task in less than four minutes. 7. Return just the list of words, separated by commas, and do not include any other content.

mountain, idea, apple, justice, elephant, ship, song, brick, sand, cloud

0.88

- Prompt K.2: Factuality Prompt You are a fact-checking expert evaluating a multi-hop factual path for factual accuracy and logical validity. The path consists of an ordered list of triples. Each triple has the form: (subject, relation, object)

Your task is to evaluate EACH triple in the given path and determine whether it is hallucinated or not. Use the path only for entity disambiguation and contextual grounding. Do not infer unsupported facts.

Evaluation criteria (apply independently to each triple):

- 1. Entity existence and disambiguation:

- - Verify that both the subject and object are real, identifiable entities (people, places, organizations, events, or concepts).
- - Use the path context only when necessary to disambiguate entities.
- - If either entity is fabricated or incorrectly identified, mark the triple as hallucinated.

- 2. Relationship validity:

- - Verify that the stated relation describes a real, historically valid, currently valid, or plausibly valid relationship between the subject and object.
- - If the relationship is plausible and could reasonably have occurred, even if not directly verifiable, mark it as not hallucinated.
- - If it is niche information about the subject and only trivia heads would know compared to common knowledge then do not mark it as hallucinated.

- 3. Hallucination detection:

- - Mark the triple as hallucinated if:
- - An entity is fabricated or does not exist
- - The relation is fabricated, nonsensical, or incorrect
- - The asserted connection clearly contradicts well-known facts
- - The triple introduces made-up properties or roles

- 4. Directionality handling: - If the relation is ambiguous or commonly bidirectional, evaluate both directions. - If either direction corresponds to a true or plausible relationship, mark the triple as not hallucinated. Output instructions: - Return valid JSON only.

- - Include exactly two keys: “explanation" and “judgments".
- - Under “explanation", provide a natural-language explanation evaluating the triples in the path.
- - Under “judgments", provide the final hallucination judgment for each triple as a list, in the same order as the path.
- - Do not include any text outside the JSON object.
- - Do not include markdown or extra keys.

The output must exactly match this schema: {{ “explanation": “string", “judgments": [“hallucinated" | “not hallucinated", ...] }} Input:

Path: { path }

###### Prompt K.3: Base prompt used in CREATE Query: { query }

Task: Identify how two real-world entities are connected by producing MANY connection paths. A connection path is a sequence of factual triples (head, relationship, tail) forming a continuous chain that begins with one entity and ends with a required target condition.

You MUST generate as many distinct valid paths as possible. Within each individual path, prefer STRONG connections (highly exclusive, specific relationships). Across the full set of paths, maintain DIVERSITY: include both popular/well-known connections and less well-known “trivia” connections, and avoid over-concentrating on the most obvious domain (e.g., for a celebrity, do not only use their main profession—add distinct non-professional connections when available).

Path definition: - Every path MUST start with the head entity: ‘[entity a]’

- - Every path MUST end with a triple whose relationship is ’[rel b]’ and whose tail entity is ’[entity b]’
- - Paths may be direct or indirect and may include one or more intermediate entities Rules and quality constraints:
- - Entities must be concrete, real-world entities only (people, organizations, works, places, genes, diseases, species, etc.). No abstract concepts.
- - Do not ask follow-up questions; respond using the best available factual knowledge.
- - Temporal connections are allowed (relationships may span different historical periods).
- - Disambiguation is required: use canonical names and qualifiers where necessary (e.g., ’Michael Jordan (basketball)’).
- - If multiple canonical entities share the same name, explore ALL of them explicitly where relevant. Deduplication:
- - Do not repeat the same path.
- - Do not repeat the same triple within a single path.
- - Prefer paths that are meaningfully different (different intermediate nodes and/or different relationships), not trivial rephrasings. Coverage and diversity:
- - Generate as many distinct valid paths as you can.
- - Explore a broad range of relationship types for ’[entity a]’.
- - Include BOTH:

- (a) strong/obvious connections (the first things most people would think of), AND
- (b) less well-known but still factual connections (“trivia”) that are distinct from the popular ones.

- - After you have produced several paths in a dominant domain (e.g., movies/acting for an actor), actively search for other distinct domains (e.g. philanthropy) when possible. Relationship quality guidance:
- - Prefer strong, specific, and distinctive relationships.
- - Strong = highly exclusive (e.g., parent/child, founder-of, spouse, authored, CEO-of, member-of a small group).
- - Weaker = shared broad attributes (e.g., “attended”, “lives in”, “worked on” in very large productions).
- - In each individual path, prioritize strong links early in the chain when possible.
- - Across paths, start with strong + distinctive paths, then include progressively more general/weaker but still valid paths to maximize coverage. Output requirements (strict):
- - Return ONLY a JSON object wrapped in answer tags. Do not include any explanatory text.
- - The JSON object must use integer keys starting from 1.
- - Each triple must be of the form: (head entity, relationship, tail entity).
- - Relationship strings must be 1–3 words.
- - If no valid path exists, return an empty JSON object. Enumerate all distinct valid connection paths that satisfy the above constraints.

###### Prompt K.4: Verbalized Sampling Prompt Query: { query }

Task: Identify how two real-world entities are connected by producing MANY connection paths. A connection path is a sequence of factual triples (head, relationship, tail) forming a continuous chain that begins with one entity and ends with a required target condition.

You MUST generate as many distinct valid paths as possible. Within each individual path, prefer STRONG connections (highly exclusive, specific relationships). Across the full set of paths, maintain DIVERSITY: include both popular/well-known connections and less well-known “trivia” connections, and avoid over-concentrating on the most obvious domain (e.g., for a celebrity, do not only use their main profession—add distinct non-professional connections when available).

Path definition: - Every path MUST start with the head entity: ’Morton Gould’ - Every path MUST end with a triple whose relationship is ’discography’ and whose tail entity is ’Benny Carter discography’ - Paths may be direct or indirect and may include one or more intermediate entities

Rules and quality constraints: - Entities must be concrete, real-world entities only (people, organizations, works, places, genes, diseases, species, etc.). No abstract concepts. - Do not ask follow-up questions; respond using the best available factual knowledge. - Temporal connections are allowed (relationships may span different historical periods). Disambiguation is required: use canonical names and qualifiers where necessary (e.g., ’Michael Jordan (basketball)’). - If multiple canonical entities share the same name, explore ALL of them explicitly where relevant.

Deduplication: - Do not repeat the same path. - Do not repeat the same triple within a single path. - Prefer paths that are meaningfully different (different intermediate nodes and/or different relationships), not trivial rephrasings.

Coverage and diversity: - Generate as many distinct valid paths as you can. - Explore a broad range of relationship types for ’Morton Gould’. - Include BOTH: (a) strong/obvious connections (the first things most people would think of), AND (b) less well-known but still factual connections (“trivia”) that are distinct from the popular ones. - After you have produced several paths in a dominant domain (e.g., movies/acting for an actor), actively search for other distinct domains (e.g. philanthropy) when possible.

Relationship quality guidance: - Prefer strong, specific, and distinctive relationships. - Strong = highly exclusive (e.g., parent/child, founder-of, spouse, authored, CEO-of, member-of a small group). - Weaker = shared broad attributes (e.g., “attended”, “lives in”, “worked on” in very large productions). - In each individual path, prioritize strong links early in the chain when possible. - Across paths, start with strong + distinctive paths, then include progressively more general/weaker but still valid paths to maximize coverage.

- - For each path, assign a normalized confidence score in [0.0,1.0] representing the relative likelihood that a knowledgeable person would recognize or know this connection. Higher scores should correspond to more direct, typical, or well-known relationships, while lower scores should correspond to more indirect, obscure, or atypical relationships. The confidence scores across all generated paths must sum to exactly 1.0, and the paths should be ordered from highest to lowest confidence. Output requirements (strict): - Return ONLY a JSON object wrapped in <answer> tags. Do not include any explanatory text.
- - The JSON object must use integer keys starting from 1.
- - Each integer key maps to an object with: “path_probability”: a float in the range [0.0, 1.0], rounded to two decimal places, representing the normalized likelihood of the path relative to the other paths, such that the probabilities across all paths sum to 1.0. “path”: a list of triples of the form (head entity, relationship, tail entity). - Each triple must be of the form: (head entity, relationship, tail entity).
- - Relationship strings must be 1–3 words.
- - If no valid path exists, return an empty JSON object. Enumerate all distinct valid connection paths that satisfy the above constraints.

###### Prompt K.5: One round of Iterative prompting

{ “content”: base prompt, “role”: “user”}, { “content": [answer from previous iteration], “role": “assistant" }, { “content": “I am restating the query: { query } Give me more/different associations than the answers you gave in the previous response. If your previous response is empty, then try again. Output requirements (strict):

- - Return ONLY a JSON object wrapped in answer tags. Do not include any explanatory text.
- - The JSON object must use integer keys starting from 1.
- - Each triple must be of the form: (head entity, relationship, tail entity).
- - Relationship strings must be 1–3 words.
- - If no valid path exists, return an empty JSON object. Enumerate all distinct valid connection paths that satisfy the above constraints.", “role": “user" }

###### Prompt K.6: Prompt for extracting entities and relations from a reasoning trace

You are given a search reasoning trace: a long, free-form explanation of how an answer was derived, including candidate facts, entities, and relationships. Your task is to extract a structured JSON object with two keys: “relation" and “entities".

- ======================== ### 1. Output Format ======================== Return **only** a single JSON object with this exact top-level structure: {{ “relation": (“relation1",“relation2",...), “entities": (“entity1",“entity2",...) }} No extra keys, no comments, no explanations, no text outside the JSON, and no trailing commas.
- ======================== ### 2. Key Definitions ======================== #### “relation"

- - An array of **strings**.
- - Each string is a **relation / predicate phrase** that appears explicitly or is clearly implied in the trace, describing how two entities can be linked.
- - Include:
- - Verb/verb-phrase relations (e.g., “worked at", “served in", “attended (alma mater)").
- - Noun-phrase relations that behave as connectors (e.g., “commander-in-chief").
- - **Do not** include:
- - Full sentences.
- - Dates, numbers, or descriptions unrelated to relations.
- - Entity names.

#### “entities"

- - An array of **strings**.
- - Each string is an **entity or concept** mentioned or implied in the reasoning trace.
- - Include:
- - People (e.g., “Joe Biden", “John Adams", “Abigail Adams", “George Washington").
- - Institutions and places (e.g., “White House", “U.S. Senate", “National Archives").
- - Roles or collective bodies (e.g., “U.S. Presidency", “First Lady of the United States").
- - Abstract but named items (e.g., “List of U.S. Presidents", “Media portrayals (e.g., John Adams miniseries)").
- - **Do not** include:
- - Relations (those go into “relation").
- - Dates, numbers, or purely descriptive adjectives.
- - Full sentences.

- ======================== ### 3. General Instructions ========================

- 1. **Scan the entire reasoning trace** and extract:

- - All relation phrases → “relation"
- - All entity names or conceptual entities → “entities"

- 2. **Do not deduplicate** items within each list. Generate all entities and relations that appear in the trace as they appear in the trace. Do not remove duplicates.
- 3. Normalize entity and relation names to lowercase, remove extra spaces, and remove punctuation.

- 3. **Do not infer facts** that are not mentioned or clearly implied.
- 4. **Return only valid JSON**, with:

- - Double quotes around strings
- - Arrays for both keys
- - No other output

- ======================== ### 4. Input to process ======================== { trace }

### NeurIPS Paper Checklist

#### 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes] Justification: The main claims are in the abstract and Section 1. Guidelines:

- • The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.
- • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.
- • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

##### 2. Limitations Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Limitations are discussed in the Appendix. Guidelines:

- • The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.
- • The authors are encouraged to create a separate “Limitations” section in their paper.
- • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

#### 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: Our work does not contain theoretical results Guidelines:

- • The answer [N/A] means that the paper does not include theoretical results.
- • All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- • All assumptions should be clearly stated or referenced in the statement of any theorems.
- • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- • Theorems and Lemmas that the proof relies upon should be properly referenced.

#### 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We will release the benchmark and the evaluation code. All evaluation prompts are given in Appendix D. All inference prompts are given in H. The inference details are given in the experimental setup in Section 5 and Appendix I.

Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- • While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

#### 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes] Justification: We will release the benchmark and the evaluation code with proper documentation. Guidelines:

- • The answer [N/A] means that paper does not include experiments requiring code.
- • Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.
- • While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.
- • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: All evaluation prompts are given in Appendix D. All inference prompts are given in H. The inference details are given in the experimental setup in Section 5 and Appendix I. We also discuss the data construction process in Section 4 and Appendix B and the human evaluation process in Section 6 and Appendix E.

Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- • The full details can be provided either with the code, in appendix, or as supplemental material.

#### 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We use paired bootstrap tests to check if the difference in performance between the models is significant or not. We report the results in Appendix J.1. We also mark the confidence intervals on the results for prompt variations in Figure 5.

Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- • The assumptions made should be given (e.g., Normally distributed errors).
- • It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).
- • If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes] Justification: We include details about the compute resources in Appendix I. Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

#### 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes] Justification: The research conducted in the paper conform in every respect with the NeurIPS Code of Ethics. Guidelines:

- • The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.
- • If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.
- • The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

#### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [N/A]

Justification: This work presents a dataset for benchmarking LLMs in their ability to do creative ideation. As a benchmark, its purpose is primarily to understand these capabilities, and we do not foresee any major risks.

Guidelines:

- • The answer [N/A] means that there is no societal impact of the work performed.
- • If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.
- • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A] Justification: We do not believe our data has a high risk of misuse, and we do not release any models. Guidelines:

- • The answer [N/A] means that the paper poses no such risks.
- • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes] Justification: We mention the license in Appendix B. Guidelines:

- • The answer [N/A] means that the paper does not use existing assets.
- • The authors should cite the original paper that produced the code package or dataset.
- • The authors should state which version of the asset is used and, if possible, include a URL.
- • The name of the license (e.g., CC-BY 4.0) should be included for each asset.

- • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- • If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

#### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes] Justification: We will upload the benchmark and code with documentation. Guidelines:

- • The answer [N/A] means that the paper does not release new assets.
- • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- • The paper should discuss whether and how consent was obtained from people whose asset is used.
- • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A] Justification: This work does not involve crowdsourcing nor research with human subjects. Guidelines:

- • The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.
- • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A] Justification: The paper does not involve any human subjects research. Guidelines:

- • The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

- • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

#### 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A] Justification: The core method developed in this research does not involve LLMs in any important, original or non-standard components. Guidelines:

- • The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- • Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.

