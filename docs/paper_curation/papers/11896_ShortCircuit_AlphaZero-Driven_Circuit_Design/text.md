arXiv:2408.09858v2[cs.LG]2Oct2024

SHORTCIRCUIT: ALPHAZERO-DRIVEN GENERATIVE CIRCUIT DESIGN

Dimitrios Tsaras1,2, Antoine Grosnit1,3, Lei Chen1, Zhiyao Xie2, Haitham Bou-Ammar∗1,4, Mingxuan Yuan∗1 Noah’s Ark Lab, Huawei1, HKUST2, TU Darmstadt3, University College London4

ABSTRACT

Chip design relies heavily on generating Boolean circuits, such as AND-Inverter Graphs (AIGs), from functional descriptions like truth tables. This generation operation is a key process in logic synthesis, a primary chip design stage. While recent advances in deep learning have aimed to accelerate circuit design, these efforts have mostly focused on tasks other than synthesis, and traditional heuristic methods have plateaued. In this paper, we introduce ShortCircuit, a novel transformer-based architecture that leverages the structural properties of AIGs and performs efﬁcient space exploration. Contrary to prior approaches attempting endto-end generation of logic circuits using deep networks, ShortCircuit employs a two-phase process combining supervised with reinforcement learning to enhance generalization to unseen truth tables. We also propose an AlphaZero variant to handle the double exponentially large state space and the reward sparsity, enabling the discovery of near-optimal designs. To evaluate the generative performance of our model , we extract 500 truth tables from a set of 20 real-world circuits. ShortCircuit successfully generates AIGs for 98% of the 8-input test truth tables, and outperforms the state-of-the-art logic synthesis tool, ABC, by 18.62% in terms of circuits size.

- 1 INTRODUCTION

The rapid proliferation of AI has triggered an unprecedented surge in computational demands, exceeding the capabilities of existing hardware and thereby becoming a major bottleneck to AI’s continued growth. Chip design plays a pivotal role in enabling the next-generation of computing systems. However, traditional methodologies struggle to keep pace with the accelerating demands, underscoring the need for innovative chip design approaches to accelerate the design process and discover novel architectures. At its core, a chip is the physical embodiment of a Boolean function, transforming binary inputs into desired outputs. Creating these embodiments is facilitated by logic synthesis, a crucial step in chip design that converts functional descriptions into directed graphs comprising logic gates. The resulting circuit must balance competing objectives, including power efﬁciency, performance, and silicon area (PPA), presenting a formidable optimization challenge. In this paper, we investigate the use of Machine Learning (ML) to generate optimized digital circuits directly from Boolean logic speciﬁcations, offering a fresh perspective on the chip design process.

Truth tables provide a complete and unambiguous way of representing a Boolean function by exhaustively enumerating the output values for all possible binary input combinations. As such, we use truth tables as the input Boolean logic description for our problem. The output of our approach are directed graphs (DAGs) in the form of an AND-Inverter Graphs (AIGs), a widely used data structure in Electronic Design Automation (EDA) (Mishchenko and Brayton, 2006; Wolf et al., 2013). AIGs consist of 2-input AND-gates as nodes, connected with normal or inverted wires, offering a simple, scalable, and universal intermediate representation for various EDA applications. Their popularity in the industry stems from their ability to efﬁciently represent complex Boolean functions, making them an ideal choice for our ML-based circuit generation approach.

[Figure 2]

∗Equal supervision

Recent efforts attempt to accelerate chip production by leveraging the development of ML methods at different steps of EDA (Huang et al., 2021; Gubbi et al., 2022) , notably for placement (Ward et al., 2012), routing (Alawieh et al., 2020), and for logic synthesis (Tu et al., 2024). Rather than directly tackling the graph generation problem, most ML methods for logic synthesis focus on the optimization of synthesis recipes, which are sequences of operators acting on a logic graph to modify its structure while preserving the associated boolean function. More recently, deep-generative methods emerged aiming to generate logic graphs rather than sequences of graph operators d’Ascoli et al. (2024); Li et al. (2024a); Dong et al. (2023). The generative approach offers higher potential as it offers more ﬂexibility than working with a ﬁxed set of operators; on the other hand, this approach involves exploring a much larger space, due to the double exponential growth of the search space with the number of Boolean function inputs.

Such a vast search space renders traditional ML methods ineffective for optimal AIG generation. However, recent advances have demonstrated that, with tailored model architectures and explorationexploitation-aware training protocols, remarkable performance can be achieved even in tasks involving such large spaces. Indeed, we can attribute the success of methods like AlphaGO Silver et al. (2016), AlphaZero Silver et al. (2017), AlphaFold Jumper et al. (2021) and even the emergence of large language models Devlin et al. (2019); Kaplan et al. (2020) to the development of custom model architectures capturing structural properties of board games, proteins, or language. Moreover, the training of these models either leverages naturally abundant data or employs speciﬁc dataaugmentation and exploration-exploitation strategies to improve their performance.

In this work, we propose ShortCircuit, a new transformer-based architecture structurally adapted for generating AND-Inverter graphs. Our transformer takes logic nodes represented as truth tables as input, and each forward pass predicts the next AND node to create in order to realize a target truth table. Moreover, we utilize an AlphaZero policy variant to effectively navigate the large state space and discover more compact designs.

We summarize our contributions as follows. i) We formally deﬁne the challenging problem of generating AIGs from target truth tables, characterized by a doubly exponential state space and a quadratically expanding action space. ii) We introduce ShortCircuit, a novel AIG-aware transformer-based architecture, enabling effective exploration of this vast search space. iii) We propose a two-step training approach, combining supervised learning and reinforcement learning, to efﬁciently learn generalizable patterns and prune the search space, improving solution quality and scalability. Finally, iv) we empirically demonstrate the effectiveness of ShortCircuit by producing circuits 18.62% smaller compared to the state-of-the-art logic synthesis, tool ABC (Mishchenko et al., 2007), and showcase the potential of ML methods to revitalize the ﬁeld of logic synthesis with a fresh perspective.

We organize the rest of the paper by introducing the necessary concepts and related works in section 2, which provides a grounding for the formal problem formulation that we present in section 3. We then detail our proposed approach, notably our model architecture 4 and our tailored training procedure in section 5. We ﬁnally present in section 6 an empirical evaluation of our method’s performance.

- 2 BACKGROUND

A digital circuit cascades logic gates to realize a Boolean function f : {0,1}n → {0,1}m, mapping a Boolean vector of size n to a Boolean vector of size m. An And-InverterGraph (AIG) is a Directed Acyclic Graph (DAG) that is commonly used to represent a Boolean function at the early stage of the chip design process, due to its simplicity and ubiquity.

- 2.1 AND-INVERTER GRAPHS

An AIG is composed of three types of nodes, (1) primary inputs, which we also refer to as inputs, (2) primary outputs, which we call the outputs, and (3) 2-input AND-nodes representing the logic gate AND. In this work, we focus on the generation of single output AIGs that represent Boolean functions of the form f : {0,1}n → {0,1} as they play an important role in logic synthesis. Fig. 1 illustrates the structure of an AIG with n = 3 inputs, where {Ik}1≤k≤3 represent input nodes, ∧4,∧5 are AND gates, and O is the output. Edge orientation indicates the direction of the Boolean signal propagation from one node (called fanin) to another (called fanout). Moreover, the two types

|O Output Node<br><br>∧ AND Node<br><br>I Input Node<br><br>Normal Edge<br><br>Negated Edge|
|---|

1 1

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

|O<br><br>[Figure 11]<br><br>1 1 0 0<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>Output:| |
|---|---|
|[Figure 19]<br><br>[Figure 20]<br><br>0<br><br>[Figure 21]<br><br>[Figure 22]<br><br>1<br><br><br>[Figure 23]| |

0

[Figure 24]

[Figure 25]

- 0

[Figure 26]

[Figure 27]

[Figure 28]

- 0

[Figure 29]

[Figure 30]

- 0

[Figure 31]

[Figure 32]

- 1

[Figure 33]

[Figure 34]

[Figure 35]

- 1

[Figure 36]

[Figure 37]

- 1 0

[Figure 38]

[Figure 39]

0 0

[Figure 40]

[Figure 41]

[Figure 42]

∧5

[Figure 43]

[Figure 44]

[Figure 45]

- 0

[Figure 46]

[Figure 47]

- 1 0 0

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

∧4

- 0

[Figure 59]

[Figure 60]

- 1

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

- 0

[Figure 65]

[Figure 66]

- 1

0

- 0

[Figure 67]

[Figure 68]

- 0

[Figure 69]

[Figure 70]

[Figure 71]

- 1

[Figure 72]

[Figure 73]

- 1 1

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

- 0

[Figure 80]

[Figure 81]

[Figure 82]

- 0

[Figure 83]

[Figure 84]

- 1

[Figure 85]

[Figure 86]

- 1

[Figure 87]

[Figure 88]

[Figure 89]

|I1 I2 I3<br><br>[Figure 90]<br><br>[Figure 91]<br><br>0<br><br>[Figure 92]<br><br>[Figure 93]<br><br>1 0<br><br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>1 1 0<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>0<br><br>[Figure 106]<br><br>[Figure 107]<br><br>0<br><br>[Figure 108]<br><br>[Figure 109]<br><br>1<br><br><br>[Figure 110]<br><br>Inputs:|
|---|

1

[Figure 111]

[Figure 112]

- 0

[Figure 113]

[Figure 114]

- 1

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

- Figure 1: Representation of an AIG, showing the truth table associated to each node.

I3 I2 I1 I1 I2 I3 ∧4 ∧5 O

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

- 0 0 0 0 0 0 0 0 1

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

- 0 0 1 1 0 0 0 0 1

[Figure 133]

[Figure 134]

- 0 1 0 0 1 0 0 0 1

[Figure 135]

- 0 1 1 1 1 0 1 0 1

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

- 1 0 0 0 0 1 0 1 0

[Figure 140]

- 1 0 1 1 0 1 0 1 0

[Figure 141]

- 1 1 0 0 1 1 0 1 0

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

- 1 1 1 1 1 1 1 0 1

[Figure 146]

.

[Figure 147]

Table 1: Truth table of each node appearing in the AIG from Fig. 1

of edges, plain and dashed, in Fig. 1 indicate that the Boolean signal can be inverted when going from a fanin to a fanout. The primary output is always connected to a single AND node with a direct or an inverter link. As AIGs only contain AND operations and Boolean inversions, we can map them to a canonical form (CNF). For instance, the CNF of the AIG in Fig. 1 naturally derived from its topology is O = ¬(¬(I1 ∧ I2) ∧ I3), and we can conversely easily go from a CNF to an AIG. Moreover, applying equivalence-preserving operations to the CNF produces new CNFs that still encode the same Boolean function. Similarly, topologically distinct AIGs can realize the same function, and to compare the quality of two AIGs a primary criterion is to compare their sizes, measured by the number of gates they contain (Mishchenko and Brayton, 2006). Smaller AIGs are generally preferred as they simplify subsequent tasks such as placement and routing, and lead to more efﬁcient circuits.

- 2.2 TRUTH TABLES

As for any other logical graphs, we can capture the behavior of an AIG by propagating Boolean values from its primary inputs to its primary outputs and applying the logical operations encountered on the directed paths. Considering again the exemplar AIG from Fig. 1, if I1 = 1, I2 = 1, and I3 = 0, propagatingthe Boolean signals we can verify that the AIG output at O is 1. By enumerating all possible input combinations and recording the corresponding output values for each gate, we can build the AIG’s full truth table, as shown in Table 1. Each row corresponds to the values of the AIG nodes for a speciﬁc set of entries (I1,I2,I3) = (i1,i2,i3) ∈ {0,1}3 displayed on the left part of the table. We can extract from this representation a binary vector of size 2n for each AIG node. For instance, the "truth table" vector representation of node ∧4 is (0 0 0 1 0 0 0 1)⊤, and the primary output one is (1 1 1 1 0 0 0 1)⊤. After discussing related works in more details, we will explain in the next section how our method utilizes this rich vector representation to perform AIG generation.

- 2.3 RELATED WORKS

We discuss heuristics for AIG generation and ML methods for logic operator sequence optimization in Appendix C, and focus here on the deep learning approaches tailored to logic graph generation.

Learning to Generate One Circuit at a Time A ﬁrst approach to generate Boolean networks with deep neural networks consists of substituting the gates and wires of a logic circuit by learnable nodes and connections to form a neural network (Belcak and Wattenhofer, 2022; Zimmer et al., 2023; Hillier et al., 2023). The network parameters are learnt by minimizing the error made by forward passes compared to the target binary output. On the one hand, this method allows to cope with larger number of primary inputs as, instead of learning an entire family of logic graphs (e.g., the 8-input AIGs), it is specialized on one particular target truth table. On the other hand, it requires to train a new neural network for each target truth table, representing a signiﬁcant runtime bottleneck. Therefore, several works inspired by the development of foundational generative models have

followed another direction, consisting in learning the synthesis process itself with a deep neural network.

Learning Circuit Synthesis using Deep Learning While Roy et al. (2021) use a CNN backbone to generate preﬁx circuits by adding or deleting nodes in a N × N grid, Li et al. (2024b) employ an auto-regressive diffusion model to generate DAG for high-level synthesis stage , and Dong et al. (2023) design a two-level GNN architecture to synthesize analog circuits that work with non-binary signals. Closer to our work are Boolformer (d’Ascoli et al., 2024) and Circuit Transformer (CT) (Li et al., 2024a), both tackling digital network synthesis with an auto-regressive transformer-based architecture. They train their policies with a supervised training phase, predicting the next element of a logic graph given a target truth table, and doing inference through beam-search or MCTS simulations. Contrary to our work, Boolformer and CT use a symbolic representation of the Boolean formula associated to a logic graph that is encoded via depth-ﬁrst search. Therefore instead of representing already built nodes by their truth tables and predicting the next node to add to the graph, they tokenize the symbols and let their model generate the next symbol of the logic formula (a logical operation, an input or an output), which requires more forward passes than for our method to produce a similar AIG. Besides, given a target truth table T⋆, Boolformer directly takes as input the boolean representation of T⋆ as we do, while Li et al. (2024a) pass a (non-optimal) AIG realizing T⋆ to the CT to generate an improved logic network.

- 3 PROBLEM DEFINITION

Truth tables encompass the results for all possible input values, but do not provide sufﬁcient structural information to derive an AIG with the corresponding output representation. Heuristics that generate a Boolean expression, or equivalently an AIG, from a truth table generally lead to exponentially large solutions that need further reﬁnement. Consequently, solving this AIG generation problem would signiﬁcantly impact digital circuit design.

Problem Deﬁnition: Given a target truth table T⋆ ∈ {0,1}n, construct an AIG with the minimum number of nodes, such that its output node O has a truth table TO that matches T⋆. Note that the size of a truth table associated with an n-input AIG is 2n, thus, the set containing all truth tables of size 2n has a cardinality of 22

, which corresponds to a magnitude of 1019 for the space of truth tables that a 6-input AIG can represent. Exploring such a large space efﬁciently requires developing speciﬁc models and training techniques. These must exploit the structural properties of the problem while managing the exploration-exploitationtrade-off inherent in such scenarios.

n

3.1 STATE REPRESENTATION & NOTATIONS

We formally deﬁne an AIG with n inputs as a graph G = (V,E), where V and E represent the node and edge sets. To capture the dynamic nature of AIG construction, we introduce a temporal parameter, t, which simultaneously represents the current time step and the number of AND-nodes in the graph, denoted as Gt = (Vt,Et). This allows us to model the evolution of the AIG over time, with the graph growing as new AND-nodes are added. We assign a unique integer ID to each node in the graph, regardless of its type; thus, at time t the node set is Vt = {I1,...,In,∧n+1,...,∧n+t}, or with a node-type agnostic notation, Vt = {v1,...,vn+t}. Following this notation, the graph G0 represents an AIG containing only input nodes, i.e., with V0 = {I1,I2,...,In}.

In our generative process, we encode the state corresponding to AIG Gt as a 3-tuple st = (Tt,T⋆,At). Here, Tt = {T1,T2,...,Tn+t} is the set of truth tables associated with the current nodes, T⋆ is the target truth table, and At = {an+1,an+2,...,an+t} is the ordered set of actions performed so far. Each action generates a new AND-node by connecting two existing nodes in one of four possible conﬁgurations: (vi,vj), (vi,¬vj), (¬vi,vj), and (¬vi,¬vj). Our goal is to perform a series of N actions, transforming G0 into a terminal AIG GN such that the truth table of the last generated AND-node, Tn+N, matches or is the negation of the target truth table T⋆. Note that our environment is stateful, as its history inﬂuences future decisions. Furthermore, our environment poses an additional challenge due to its action space expanding quadratically with each new node we add.

Truth Tables of Built Nodes

Target T⋆

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

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Inputs:

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

...

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

[Figure 214]

[Figure 215]

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

### Value Module details

Policy Module details

Node PE Target PE

S× Transformer Layers

P× Transformer Layers P×

H×

Exclude Target Embedding

Transformer Layers

Input Sequence: Node Hidden Embeddings

Target Sequence: Target Hidden Embedding

Self-Attention

Value Module

Policy Module

Cross-Attention

Self-Attention Logits:

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

1.2 0.0 9.3 10.8

9.

Linear

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

0.3 −3.6 −17

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Softmax Tanh

8.3 2.2

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

−6.2

State Value Logit

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

Action Outputs: Probabilities State Value

- Figure 2: ShortCircuit model takes as inputs a target truth table T⋆ and the truth tables of the already built nodes. It ﬁrst appends a type-dependent positional encoding before going through several transformer layers. Then, the model is split into two heads respectively outputting a probability distribution over the next possible actions (policy module on the left), and a value reﬂecting the quality of the current inputs (value module on the right).

- 4 MODEL ARCHITECTURE

We propose an iterative approach to AIG construction, where we gradually build the circuit, starting with G0 and letting our model decide at each step which AND-node to add, aiming at realizing a given target truth table T⋆. To generate the next gate, the model takes the set of existing nodes

- as input, and it outputs a probability distribution over the set of AND-nodes that can be built by combining any pair of already existing nodes, taking edge types into account. Formally, let |Vt| denote the number of nodes in the current state of the graph, then the action space is a 4 × |Vt| × |Vt| tensor, where each |Vt| × |Vt| slice corresponds to a connection type. Speciﬁcally, the cell (i,j) in a given slice indicates the probability of connecting node vi with vj, and the slice index ǫ ∈ {1,2,3,4}, corresponds to a combination of speciﬁc edge types: (vi,vj), (¬vi,vj), (vi,¬vj), or (¬vi,¬vj). Therefore, we can sample a triplet (ǫ,i,j) following the distribution given by this

- 4 × |Vt| × |Vt| tensor and add the corresponding node to the graph. The process ends when the truth table of the sampled node matches either the target one T⋆ or its full negation ¬T⋆, or after reaching a maximum number of steps Nmax.

To effectively explore and prune the vast state space, our model comprises a policy and a value module to assess intermediate states and strategically get closer to the desired target. As shown on Fig. 2, our architecture consists of a shared core embedding the truth tables applying position encodings and H transformer encoder layers. Then, the hidden embeddings are passed as input to the 4 stacked policy modules and to the value module. The policy modules combined with a softmax produce a distribution over next actions, and the value module ending with tanh predicts an expected reward in [−1,1].

ǫ Build ∧i from (ǫ,i,j)

[Figure 289]

[Figure 290]

- 1 ∧i ∧ ∧j

[Figure 291]

[Figure 292]

- 2 ¬ ∧i ∧ ∧j

[Figure 293]

[Figure 294]

[Figure 295]

- 3 ∧i ∧ ¬ ∧j

[Figure 296]

[Figure 297]

[Figure 298]

- 4 ¬ ∧i ∧ ¬ ∧j

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

Table 2: Connector types for each ǫ.

Positional Encoding (PE) In natural language processing (NLP) tasks, PE allows transformers to capture a sequential relation in the inputs. In our setting, we do not need to hint the model about the graph structure graph, as the truth tables already convey all the necessary information. Moreover,

the self-attention operation should treat every node equally, as there is no limitation regarding which two nodes can be combined to form a new node. Nevertheless, the model should be able to treat differently the built nodes and the target truth table, leading us to introduce two learnable positional encoders to distinguish them (“Node PE” on Fig. 2).

Policy Module Transformers are the state-of-the-art architecture to handle sequential data. These models particularly shine when trained to predict the next token in a sequence by outputting a ﬁxedsize tensor representing a sampling probability over a token glossary. In our case though, the set of nodes that we can build grows at each step, as more pairs of nodes can be combined to produce the next node. Inspired by NLP tasks, for which attention scores among related tokens are high, we use attention to guide next node generation. Thus, we directly use the ﬁnal self-attention map of four parallel policy modules to get the probability to connect any two nodes with speciﬁc edge types. Each policy module has P transformer encoder layers and outputs a ﬁnal self-attention layer. In the last self-attention layer, we exclude the entry corresponding to the target node and returns the self-attention scores based on the existing AIG nodes embeddings. We aggregate the scores from the four policy modules and mask the ones associated to already built nodes and their negate versions. We ﬁnally apply a softmax to the remaining scores to produce a single probability distribution.

Value Module The value module is a critical component to assess how favorable a state is and therefore prevent expanding unpromising states. Our value module consists of S many transformer encoder layers. As the quality of a state not only depends on the nodes that are present in the graph

- at that stage, but also on the target truth table that should be realized, the value module also uses the two learnable type-based positional encoders introduced above. After performing the embedding, we compute the cross-attention between the graph nodes and the target truth table, which yields a new vector representation of the target. Finally, we feed this vector to a linear layer producing a single value, which should reﬂect the quality of the current state.

- 5 TRAINING SHORTCIRCUIT

The sparse nature of the problem makes it practically impossible to discover functionally correct graphs when exploring the double exponential state space uniformly. This challenge necessitates a more effective training approach for our ShortCircuit. To address this, we propose a two-stage training regimen consisting of a supervised pre-training stage to initialize the policy module, followed by an AlphaZero-style ﬁne-tuning phase to improve the policy module and train the value module. Although pre-training the policy module provides a good prior to predict the most useful next actions, simply following it does not guarantee that an AIG matching T⋆ will be constructed due to the inherent difﬁculty of the problem. This limitation highlights the importance of the ﬁne-tuning stage, which aims to reﬁne the policy module and leverage the value module for improved performance.

- 5.1 PRE-TRAINING

Just as large language models in NLP are pre-trained using next-token prediction, we can pre-train our transformer model on a next-node prediction task. This approach requires a corpus of singleoutput AIGs that our model can learn to regenerate node by node, starting from the truth tables of the primary inputs and a target T⋆. Since the AIG generation problem has not yet been addressed by the ML community, there is no large, well-structured corpus of publicly available (truth table, AIG) pairs to perform this pre-training. Therefore, we ﬁrst curate a dataset of single-output AIGs for the pre-training of ShortCircuit by leveraging existing open-source collections of digital circuits.

Data Extraction To generate an AIG dataset with the desired input and output sizes, we utilize the EPFL benchmarks (Amarú et al., 2015), which contain a collection of 20 real circuits realizing arithmetic and control functions. On average, the arithmetic and control circuits have 175 inputs and 137 outputs with 22520 AND-nodes, as detailed on Table 4 and 5. Since these circuits have more inputs than the AIGs we aim to generate, we extract subgraphs, or cuts, from them. A cut refers to a connected subset of nodes in the AIG that divides the graph into two disjoint parts. The root of a cut is the node to which all directed paths within the cut converge to and a leaf node is a node in the cut that have at least one fanin outside of the cut. By design, a cut forms a single-output AIG

A = { a5 = (3,1,2), a6 = (1,3,5), a7 = (2,3,4), a8 = (3,6,7) } A = (A1,A2,

O

O

|A3| |,A4|
|---|---|---|
| | | |

)

Submatrix: A3,1:2,1:2





|[Figure 305]<br><br>[Figure 306]<br><br>1<br><br>[Figure 307]|
|---|

0 0 0 0 0 0 0 0 0 0

I1 I2 I3 I4

∧8

Edge type marking

[Figure 308]

[Figure 309]

[Figure 310]

0 0 0 0 0 0 0

∧6

Node ordering

[Figure 311]

[Figure 312]

# 0

[Figure 313]

[Figure 314]

0 0 1

- ∧5
- ∧6
- ∧7

∧5 ∧7

[Figure 315]

[Figure 316]

 

 

[Figure 317]

[Figure 318]

I1 I2 I3 I4

I1 I2 I3 I4 ∧5 ∧6 ∧7

- Figure 3: We start data pre-processing by sorting the AIG nodes in topological order. Then, we identify the action types ǫ ∈ {1,2,3,4} based on the edges. Next, we build the sequence of actions A and generate the global action tensor A = (A1,A2,A3,A4). We highlight the structure of A3, which contains a 1 at entries (1,2) and (6,7) for the generation of ∧5 and ∧8 (actions a5 and a8).

with a number of inputs corresponding to the number of leaves. We defer to Appendix B.2 the full description of the cut extraction method we develop to build a dataset of single-output AIGs.

Data Preparation Since our policy should predict the next action, i.e. the next node to add to a partial AIG, we need to convert the AIGs we load from our training dataset into a sequence of ground-truth actions. As different series of actions can lead to the same graph with N nodes, we ﬁrst sort the nodes of the training AIG we load into a topological order {I1,...,In,∧n+1,...,∧N} (or {v1,...,vN} with node-type agnostic notation), where ∧N is connected to the output O. We also convert the AIG nodes into truth tables, as described in section 2, and use the truth table of O as the target T⋆. From the topological sequence of nodes, we build the sequence of actions that our policy should learn to perform when its goal is to generate T⋆. As mentioned in section 4, creating node vk = (¬)vi ∧ (¬)vj, with 1 ≤ i < j < k, corresponds to action ak = (ǫ,i,j), whose ﬁrst component ǫ ∈ {1,2,3,4}, indicates the types of the edges connecting vk to its parents, as detailed in Table 2. This procedure leaves us with the sequence of actions A = {an+1,an+2,...,aN}, starting with index n+1 since the n primary inputs are given at the beginning of the AIG generation process.

To efﬁciently generate target action distributions, we aggregate all the actions into a sparse 3dimensional tensor A = (A1,A2,A3,A4) where each element Aǫ is a N ×N matrix representing the actions with connection type ǫ. The value of the entry (i,j) of Aǫ is set to 1 if (ǫ,i,j) belongs to A and to zero otherwise. Thus, if all the nodes up to vk are already built, considering each submatrix Aǫ,1:k,1:k taking the ﬁrst k rows and k columns of Aǫ allows to easily identify which nodes with connection ǫ we could build next. Taking the submatrices for all values of ǫ, we obtain the target action distribution by setting the entries corresponding to already performed actions at 0, and normalizing the resulting tensor. Fig. 3 illustrates this action tensor building procedure.

Data Augmentation The ﬁrst data augmentation we employ consists of using the same AIG for both targets T⋆ and ¬T⋆. This is valid because we can generate one target or the other by connecting the ﬁnal node ∧N with the output O using a regular or an inverter edge. Our second data augmentation leverages the fact that any order of truth table rows is valid, provided that the same order is used for all the nodes. Since our ShortCircuit’s inputs are truth tables, it is desirable for the model to be invariant to row permutations. Formally, the model should generate the same next-action prediction

whether it receives the truth tables T1,...,TN and T⋆, where Ti = (t(1)i ,...,t(2

n)

i ) ∈ {0,1}2

, or when it gets the permuted truth tables σ(T1),...,σ(TN),σ(T⋆) where σ is a permutation in S2n and σ(Ti) = t(iσ(1)),...,t(σ(2

n

n))

i . As structurally encoding this invariance into our policy architecture would be computationally too expensive, we apply random permutations to the inputs of our model during the training, which does not impact the other metadata introduced in the previous section.

Pre-Training Flow With the prepared augmented data, we can proceed to train our policy module to match the ground-truth next-action distributions of our training set. For training loss, we experimented with KL divergence and cross-entropy, both of which measure the distance between two probability distributions. In practice, KL divergence loss yielded better results. Besides, the backbone of our model being a transformer, we implement a custom masking strategy during training. to maintain causality in the auto-regressive generation process. Since the primary inputs and the target truth table are available from the start, and as there is no causality for their existence, we allow full attention for their embeddings, and only apply a causal mask for the rest of the nodes.

- 5.2 FINE-TUNING

Fine-tuningaims to align the value and policy module so they operate effectivelytogether. Unlike the policy module, we cannot properly initialize the value module during pre-training as the generated dataset only contains successful examples, which would mislead the value module to consider that all states are “good”. Skipping pre-training, though, would lead to a random exploration of the vast search space of truth tables, which would likely result in encountering only “bad” states, preventing the model from learning what a “good” state is. Therefore, the most viable option to train our value module is through experience, by performing searches with a pre-trained policy module. We utilize AlphaZero as the orchestration framework to reﬁne the policy and value modules.

AlphaZero AlphaZero has demonstrated remarkable success in board games with enormous state spaces, such as chess (1044) and Go (10170). Since truth tables features similar state space problem, we adapt AlphaZero’s effective search and pruning capabilities to navigate AIG generation. By combining a policy module to propose actions and a value module to evaluate state viability, AlphaZero strikes a balance between exploitation and exploration. We adapt and modify the selection strategy, predictor upper conﬁdence bound applied to trees (PUCT) used by AlphaZero, as follows:

[Figure 320]

N(s,a) N(s,a) + 1

PUCT(s,a) = Q(s,a) + b Q(s,a;θ) + c P(s,a;θ) a

[Figure 321]

where, Q(s,a) represents the propagated discounted discovered reward, while Q(s,a;θ) represents the predicted expected Q-value, P(s,a;θ) is the policy module’s probability distribution, N(s,a) tracks state visitations, and b and c are parameters balancing exploration and exploitation. Computing Q(st,a;θ) for every action is too expensive, so we initialize Q(st,a) = Q(st,a;θ) = 0, perform the action that maximizes PUCT(st,a), and only compute the value of the state Q(st+1) once we visit it. The term Q(s,a) + bQ(s,a;θ) represents the exploitation in PUCT, as if during search our method discovers a "good" state or a terminal state, we exploit it and focus the search locally to discover more compact designs. The term P(s,a;θ) suggests actions to perform, but the term √

[Figure 322]

a N(s,a)/(N(s,a)+1) promotes exploration.

AlphaZero stores intermediate results and metadata, such as Q(s,a), Q(s,a;θ), P(s,a;θ), and N(s,a), in the nodes visited during MCTS. These nodes are associated with states and form a tree, where edges indicate the actions performed to reach each node-state pair. When simulation starts, we mark the initial state as the root node, compute the action distribution, and inject Dirichlet noise. During simulation, AlphaZero follows PUCT to choose actions and continues until meeting one of the three following stopping conditions: encountering a state s that is not expanded, reaching a maximum number of steps, or arriving at a terminal state. If the state is not expanded, we need to compute Q(s;θ) and P(s,a;θ) for that state and we back-propagate Q(s;θ) to the previous MCTS nodes, and increment N(s,a). Once we complete the given number of simulations, AlphaZero applies the most visited action, argmaxa∈A N(s,a). In our case, we rather follow the observed discounted reward argmaxa∈A Q(s,a) as we ﬁnd the visitation count signal too noisy given our simulation budget.

Fine-Tuning Flow Generating trajectories for millions of truth tables is computationally challenging. Thus, to best exploit our resources, our ﬁne-tuning regimen consists of data collection and model training processes. These data collectors generate trajectories and add their ﬁndings to a ﬁxed length replay buffer. Under the hood, the data collectors store the metadata, including truth tables and discovered reward Q(s), of the MCTS root node for each step in the trajectory in the replay buffer. Successful trajectories receive a reward of 1, while failed ones receive

− min(hd(TN,T⋆),hd(TN,¬T⋆)), where hd is the Hamming distance and TN is the last generated truth table.

The trainer process randomly samples data from the replay buffer and uses the truth tables as input for the model. Since the value module aims to predict the Q-value of a state, the goal is to minimize the mean squared error (MSE) between the predicted value and the retrieved Q(s). The target distribution for the policy module is the normalized number of visitations N(s,a)/

a N(s,a). Similar to pre-training, we minimize the KL-divergence between the output of the policy module and the target probability distribution. Finally, the trainer process broadcasts the updated weights asynchronously to the data collectors after a given number of training steps.

- 6 EXPERIMENTAL EVALUATION

We introduce the implementation details such as model and search hyperparameters, datasets and baseline methods in section 6.1 and 6.2. We compare the effectiveness of ShortCircuit against several baselines in 6.3. Finally, section 6.4 presents a study on the impact of number of simulations.

- 6.1 EXPERIMENTAL SETUP

We train and evaluate ShortCircuit on the 8-input truth tables that we randomly extract from the EPFL benchmark, as described in section 5.1. We speciﬁcally choose to test on these circuits, since they correspond to real-world Boolean functions that have more practical interest than uniformly random truth-tables. In total, we extract 1.8 million AIGs with an average number of AND-nodes of 10.08. We pre-train ShortCircuit with a batch size of 1024 for 250 epochs, and ﬁnetune the model until it converges. Our model architecture is as depicted on Fig. 2 and uses transformer blocks following Llama 3 (Meta Llama team, 2024) structure. We use are H = 4 and P = S = 3 transformer blocks for the different parts with 16 heads and an intermediate embedding size of 4096, summing to 51.6 million parameters. During pre-training, we apply a uniformly sampled permutationtransform to the training sequencesand a target negationtransformboth with probability 50%.

- 6.2 BASELINES

We derive each truth table in our test from the primary output of an extracted cut. Therefore, we can use that cut as a baseline, denoted as Cut, as it is an AIG realizing the target truth table. Moreover, we leverage a popular logic optimization ﬂow, resyn2, that applies multiple operators to optimize the Cut AIGs. While resyn2 cannot guarantee the optimality of the resulting AIGs, it provides a reliable optimality proxy for the current graph sizes, which we represent as a horizontal line with the tick label O, when applicable. Additionally, we compare ShortCircuit against the stateof-the-art open-source logic synthesis tool ABC. This library applies a series of Boolean algebra transformations to generate an AIG from a truth table. The sequence of commands we use in ABC is as follows: read_truth -x [truth table]; collapse; sop; strash; write [outfile]. We also apply resyn2 to the obtained AIGs, which we denote as ABC+resyn2. Finally, we compare ShortCircuit against Boolformer with default beam size of 10. This learned method produces an optimized Boolean expression given a truth table, and we convert this expression into an AIG without introducing any additional logic redundancy.

- 6.3 GENERATION QUALITY EXPERIMENTS

We evaluate ShortCircuit against our baselines on 500 truth tables associated with randomly sampled AIGs from the EPFL benchmarks. We allow ShortCircuit to attempt to generate a circuit with up to 30 AND-nodes. ShortCircuit performs 8 MCTS simulations and generates up to 20 AND-nodes in each simulation, before performing an action. The success rate of ShortCircuit on this test set is 98%.

- Fig. 4a compares the average size of the successfully generated circuits compared to Cut, ABC, and ABC+resyn2. ShortCircuit generates compact AIGs, near-optimal as indicated by the horizontal line at the tick O, with an average time per generation of 1.83s. Speciﬁcally, ShortCircuit produces circuits with 9.44 AND-nodes on average, which is 0.76% larger than the optimal ones that have

- 8
- 9
- 10
- 11
- 12

Cut ABC ABC+resyn2 ShortCircuit

#AND-nodes

O

Algorithm

(a) ShortCircuit vs. traditional methods

Boolformer Cut ABC ABC+resyn2 ShortCircuit

- 8
- 9
- 10

#AND-nodes

O

Algorithm

(b) ShortCircuit vs. traditional and learned methods

- Figure 4: Average number of AND-nodes for the successfully generated AIGs across several baselines.

9.37 AND-nodes on average. Furthermore, the AIGs generated by ShortCircuit are signiﬁcantly smaller than the ones from the baselines Cut and ABC, achieving a size reduction of 5.77% and 18.62%, respectively. Finally, ShortCircuit produces slightly smaller AIGs than ABC+resyn2 by 0.26%. While these results demonstrate the effectiveness of ShortCircuit, changes of parameters, such as increasing the AND-node limits, can further improve the success rate, albeit causing extra runtime.

- Fig. 4b compares the AIG sizes against the same baselines as above with the addition of Boolformer. Boolformer successfully generated 85% of the given truth tables, so we report the results only on the successfully generated by both learned methods. ShortCircuit still maintains similarly good performance on this test subset as on the previous set shown on Fig. 4a. Boolformer produces larger AIGs and fails in particular to generate the larger AIGs, but it produces a Boolean expression in about 0.75s. ShortCircuit’s AIGs are 13.14% smaller while the running time, 1.2s, is comparable.

- 6.4 IMPACT OF NUMBER OF SIMULATIONS

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

20 21 22 23 24 25 26 27 28

90

92

94

96

98

100

# MCTS Simulations

Success(%)

- Figure 5: ShortCircuit’s success rate vs. number of MCTS simulations per action.

- 9.8
- 10

| | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | |2|4| |8| |1|6| | | |32 6|4|1|28| |25|6| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |

#AND-nodes

9.6

9.4

9.2

9

10−1 100 101 102

Average runtime per generation (s)

Figure 6: Average #AND-nodesper AIG vs. generation time for ShortCircuit with varying MCTS simulations marked on top of each data point.

To better understand the role of MCTS simulations in the performance of ShortCircuit, we investigate their impact on the success rate, the circuit size, and the execution time. Using the same test set of 500 truth tables as in section 6.3, we evaluate our method when it solely relies on its policy module. Fig. 5 illustrates how the success rate evolves as the number of MCTS simulations increases from 1 to 256. For clarity, we append an integer i next to our method’s name (ShortCircuit[i]) to indicate the number of simulations we perform.

When using only 1 simulation, ShortCircuit[1] performs a greedy search, where the policy selects the most likely action, a = argmaxa∈A P(s,a;θ). This greedy strategy yields a low success rate of 92.2% but beneﬁts from a very short generation time of just 0.11s. On the other end, ShortCircuit[256] achieves a signiﬁcantly higher success rate of 98.6%, albeit with a much longer running time of 106.45s. Increasing the number of simulations enables the model to explore a larger – but still limited – portion of the solution space, resulting in higher success rates and the discovery of more compact designs. For example, ShortCircuit[256] generated as “optimal” AIGs as O. Fig. 6

highlights this trade-off by revealing a Pareto front, suggesting that we can adjust the number of MCTS simulations to achieve the desired balance between success rate, design quality, and running time.

- 7 CONCLUSION

In this paper, we introduced ShortCircuit, a novel transformer-based architecture for generating AIGs from a target truth table. Our approach combines a structurally aware transformer model with an AlphaZero-inspired policy variant, enabling efﬁcient navigation through the doubly exponential state space associated with truth tables. In our experiments, we demonstrated the effectiveness of ShortCircuit in producing high-quality AIGs that are signiﬁcantly smaller than those generated by one of the state-of-the-art logic synthesis tools ABC and the trained Boolformer. Speciﬁcally, our method achieved a relative size reduction of 18.62%, and 15.13% respectively.

This work contributes to the expanding ﬁeld of ML applications in chip design, showcasing the potential of deep learning to revitalize the ﬁeld with new perspectives. We demonstrated that it is possible to generate high-quality AIGs from truth tables, paving the way for future research in this area. Future work will focus on extending ShortCircuit to handle AIGs with multiple outputs, integrating our approach with existing logic synthesis tools, and exploring its application in industrial settings. Finally, our goal is to enable the creation of more efﬁcient, scalable, and innovative computing systems, and we believe that ShortCircuit is an important step towards realizing this vision.

REFERENCES

Alan Mishchenko and Robert K. Brayton. Scalable logic synthesis using a simple circuit structure.

2006. URL https://api.semanticscholar.org/CorpusID:8597391. Clifford Wolf, Johann Glaser, and Johannes Kepler. Yosys-a free verilog synthesis suite. 2013. URL https://api.semanticscholar.org/CorpusID:202611483.

Guyue Huang, Jingbo Hu, Yifan He, Jialong Liu, Mingyuan Ma, Zhaoyang Shen, Juejian Wu, Yuanfan Xu, Hengrui Zhang, Kai Zhong, Xuefei Ning, Yuzhe Ma, Haoyu Yang, Bei Yu, Huazhong Yang, and Yu Wang. Machine learning for electronic design automation: A survey. ACM Trans. Design Autom. Electr. Syst., 26:40:1–40:46, 2021. URL https://api.semanticscholar.org/CorpusID:231839647.

Kevin Immanuel Gubbi, Sayed Aresh Beheshti-Shirazi, Tyler Sheaves, Soheil Salehi, Sai Manoj PD, Setareh Rafatirad, Avesta Sasan, and Houman Homayoun. Survey of machine learning for electronic design automation. In Proceedings of the Great Lakes Symposium on VLSI 2022, GLSVLSI ’22, page 513–518, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450393225. doi: 10.1145/3526241.3530834. URL https://doi.org/10.1145/3526241.3530834.

Samuel I. Ward, Myung-Chul Kim, Natarajan Viswanathan, Zhuo Li, Charles J. Alpert, Earl E. Swartzlander, and David Z. Pan. Keep it straight: teaching placement how to better handle designs with datapaths. In ACM International Symposium on Physical Design, 2012. URL https://api.semanticscholar.org/CorpusID:8570378.

Mohamed Baker Alawieh, Wuxi Li, Yibo Lin, Love Singhal, Mahesh A. Iyer, and David Z. Pan. High-deﬁnition routing congestion prediction for large-scale fpgas. In 2020 25th Asia and South Paciﬁc Design Automation Conference (ASP-DAC), pages 26–31, 2020. doi: 10.1109/ ASP-DAC47756.2020.9045178.

Kaihui Tu, Xifan Tang, Cunxi Yu, Lana Josipovic,´ and Zhufei Chu. Logic Synthesis, pages 135–

164. Springer Nature Singapore, Singapore, 2024. ISBN 978-981-99-7755-0. doi: 10.1007/ 978-981-99-7755-0_9. URL https://doi.org/10.1007/978-981-99-7755-0_9.

Stéphane d’Ascoli, Samy Bengio, Joshua M. Susskind, and Emmanuel Abbe. Boolformer: Symbolic regression of logic functions with transformers, 2024. URL https://openreview.net/forum?id=wmzFZ9lJrD.

Xihan Li, Xing Li, Lei Chen, Xing Zhang, Mingxuan Yuan, and Jun Wang. Circuit transformer: End-to-end circuit design by predicting the next gate. ArXiv, abs/2403.13838, 2024a. URL https://api.semanticscholar.org/CorpusID:268553512.

Zehao Dong, Weidong Cao, Muhan Zhang, Dacheng Tao, Yixin Chen, and Xuan Zhang. Cktgnn: Circuit graph neural network for electronic design automation. arXiv preprint arXiv:2308.16406, 2023.

David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Vedavyas Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy P. Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of go with deep neural networks and tree search. Nat., 529(7587):484–489, 2016. doi: 10.1038/NATURE16961. URL https://doi.org/10.1038/nature16961.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy P. Lillicrap, Karen Simonyan, and Demis Hassabis. Mastering chess and shogi by self-play with a general reinforcement learning algorithm. CoRR, abs/1712.01815, 2017. URL http://arxiv.org/abs/1712.01815.

John M. Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A A Kohl, Andy Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, and Demis Hassabis. Highly accurate protein structure prediction with alphafold. Nature, 596:583 – 589, 2021. URL https://api.semanticscholar.org/CorpusID:235959867.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pretraining of deep bidirectional transformers for language understanding. In North American Chapter of the Association for Computational Linguistics, 2019. URL https://api.semanticscholar.org/CorpusID:52967399.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeff Wu, and Dario Amodei. Scaling laws for neural language models. ArXiv, abs/2001.08361, 2020. URL https://api.semanticscholar.org/CorpusID:210861095.

Alan Mishchenko et al. Abc: A system for sequential synthesis and veriﬁcation. URL http://www. eecs. berkeley. edu/alanmi/abc, 17, 2007.

Peter Belcak and Roger Wattenhofer. Neural combinatorial logic circuit synthesis from inputoutput examples. CoRR, abs/2210.16606, 2022. doi: 10.48550/ARXIV.2210.16606. URL https://doi.org/10.48550/arXiv.2210.16606.

Matthieu Zimmer, Xuening Feng, Claire Glanois, Zhaohui JIANG, Jianyi Zhang, Paul Weng, Dong Li, Jianye HAO, and Wulong Liu. Differentiable logic machines. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=mXfkKtu5JA.

Adam Hillier, Ngân (NV) Vu,˜ Daniel J. Mankowitz, Daniele Calandriello, Edouard Leurent, Georges Rotival, Ivan Lobov, Kshiteej Mahajan, Marco Gelmi, and Natasha Antropova. Learning to design efﬁcient logic circuits, 2023. URL https://cassyni.com/events/S2LPTWZeMh9TGcLJe5jpqK. NANDA Workshop 2023.

Rajarshi Roy, Jonathan Raiman, Neel Kant, Ilyas Elkin, Robert Kirby, Michael Siu, Stuart Oberman, Saad Godil, and Bryan Catanzaro. Preﬁxrl: Optimization of parallel preﬁx circuits using deep reinforcement learning. In 2021 58th ACM/IEEE Design Automation Conference (DAC), pages 853–858, 2021. doi: 10.1109/DAC18074.2021.9586094.

Mufei Li, Viraj Shitole, Eli Chien, Changhai Man, Zhaodong Wang, Srinivas, Ying Zhang, Tushar Krishna, and Pan Li. LayerDAG: A layerwise autoregressive diffusion model of directed acyclic graphs for system. In Machine Learning for Computer Architecture and Systems 2024, 2024b. URL https://openreview.net/forum?id=IsarrieeQA.

Luca Amarú, Pierre-EmmanuelGaillardon, and GiovanniDe Micheli. The epﬂ combinationalbenchmark suite. In Proceedings of the 24th International Workshop on Logic & Synthesis (IWLS), number CONF, 2015.

Meta Llama team. Introducing meta llama 3: The most capable openly available llm to date. Meta AI Blog, 2024. URL https://ai.meta.com/blog/meta-llama-3/.

Maurice Karnaugh. The map method for synthesis of combinational logic circuits. Transactions of the American Institute of Electrical Engineers, Part I: Communication and Electronics, 72: 593–599, 1953. URL https://api.semanticscholar.org/CorpusID:51636736.

W. V. Quine. The problem of simplifying truth functions. The American Mathematical Monthly, 59(8):521–531, 1952. ISSN 00029890, 19300972. URL http://www.jstor.org/stable/2308219.

W. V. Quine. A way to simplify truth functions. The American Mathematical Monthly, 62(9):627–631, 1955. ISSN 00029890, 19300972. URL http://www.jstor.org/stable/2307285.

E. J. McCluskey Jr. Minimization of boolean functions. Bell System Technical Journal, 35(6):1417–1444, 1956. doi: https://doi.org/10.1002/j.1538-7305.1956.tb03835.x. URL https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1956.tb03835.x.

R.L. Rudell and A. Sangiovanni-Vincentelli. Multiple-valued minimization for pla optimization. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 6(5):727–750,

1987. doi: 10.1109/TCAD.1987.1270318.

Alan Mishchenko, Robert Brayton, Stephen Jang, and Victor Kravets. Delay optimization using sop balancing. In 2011 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), pages 375–382, 2011. doi: 10.1109/ICCAD.2011.6105357.

John A. Darringer, William H. Joyner, C. Leonard Berman, and Louise Trevillyan. Logic synthesis through local transformations. IBM Journal of Research and Development, 25(4):272–280, 1981. doi: 10.1147/rd.254.0272.

Alan Mishchenko, Satrajit Chatterjee, and Robert Brayton. Dag-aware aig rewriting a fresh look at combinational logic synthesis. In Proceedings of the 43rd Annual Design Automation Conference, DAC ’06, page 532–535, New York, NY, USA, 2006. Association for Computing Machinery. ISBN 1595933816. doi: 10.1145/1146909.1147048. URL https://doi.org/10.1145/1146909.1147048.

Heinz Riener, Eleonora Testa, Winston Haaswijk, Alan Mishchenko, Luca Amarù, Giovanni De Micheli, and Mathias Soeken. Scalable generic logic synthesis: One approach to rule them all. In 2019 56th ACM/IEEE Design Automation Conference (DAC), pages 1–6, 2019.

Antoine Grosnit, Cedric Malherbe, Rasul Tutunov, Xingchen Wan, Jun Wang, and Haitham Bou Ammar. Boils: Bayesian optimisation for logic synthesis. In 2022 Design, Automation & Test in Europe Conference & Exhibition (DATE), pages 1193–1196. IEEE, 2022.

Chang Feng, Wenlong Lyu, Zhitang Chen, Junjie Ye, Min jie Yuan, and Jianye Hao. Batch sequential black-box optimization with embedding alignment cells for logic synthesis. 2022 IEEE/ACM International Conference On Computer Aided Design (ICCAD), pages 1–9, 2022. URL https://api.semanticscholar.org/CorpusID:254927570.

Abdelrahman Hosny, Soheil Hashemi, Mohamed Shalan, and Sherief Reda. DRiLLS: Deep Reinforcement Learning for Logic Synthesis. 2020 25th Asia and South Paciﬁc Design Automation Conference (ASP-DAC), pages 581–586, September 2020. doi: 10.1109/ASP-DAC47756.2020. 9045559.

Guanglei Zhou and Jason H. Anderson. Area-driven fpga logic synthesis using reinforcement learning. In 2023 28th Asia and South Paciﬁc Design Automation Conference (ASP-DAC), pages 159–165, 2023.

Yu Qian, Xuegong Zhou, Hao Zhou, and Lingli Wang. An efﬁcient reinforcement learning based framework for exploring logic synthesis. ACM Trans. Des. Autom. Electron. Syst., 29(2), jan 2024. ISSN 1084-4309. doi: 10.1145/3632174. URL https://doi.org/10.1145/3632174.

Winston Haaswijk, Edo Collins, Benoit Seguin, Mathias Soeken, Frédéric Kaplan, Sabine Süsstrunk, and Giovanni De Micheli. Deep learning for logic optimization algorithms. In 2018 IEEE International Symposium on Circuits and Systems (ISCAS), pages 1–4, 2018. doi: 10.1109/ISCAS.

- 2018.8351885.

Yasasvi V. Peruvemba, Shubham Rai, Kapil Ahuja, and Akash Kumar. Rlguided runtime-constrained heuristic exploration for logic synthesis. In 2021 IEEE/ACM International Conference On Computer Aided Design (ICCAD), page 1–9. IEEE Press, 2021. doi: 10.1109/ICCAD51958.2021.9643530. URL https://doi.org/10.1109/ICCAD51958.2021.9643530.

Keren Zhu, Mingjie Liu, Hao Chen, Zheng Zhao, and David Z. Pan. Exploring logic optimizations with reinforcement learning and graph convolutional network. In 2020 ACM/IEEE 2nd Workshop on Machine Learning for CAD (MLCAD), pages 145–150, 2020. doi: 10.1145/3380446.3430622.

Animesh Basak Chowdhury, Benjamin Tan, Ryan Carey, Tushit Jain, Ramesh Karri, and Siddharth Garg. Bulls-eye: Active few-shot learning guided logic synthesis. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 42(8):2580–2590, 2023. doi: 10.1109/TCAD.2022.3226668.

Cunxi Yu, Houping Xiao, and Giovanni De Micheli. Developing synthesis ﬂows without human knowledge. In Proceedings of the 55th Annual Design Automation Conference, DAC ’18, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450357005. doi: 10. 1145/3195970.3196026. URL https://doi.org/10.1145/3195970.3196026.

Cunxi Yu and Wang Zhou. Decision making in synthesis cross technologies using lstms and transfer learning. In 2020 ACM/IEEE 2nd Workshop on Machine Learning for CAD (MLCAD), pages 55–60, 2020. doi: 10.1145/3380446.3430638.

Nan Wu, Jiwon Lee, Yuan Xie, and Cong Hao. Lostin: Logic optimization via spatio-temporal information with hybrid graph models. In 2022 IEEE 33rd International Conference on Application-speciﬁc Systems, Architectures and Processors (ASAP), pages 11–18, 2022. doi: 10.1109/ASAP54787.2022.00013.

Albert Bou, Matteo Bettini, Sebastian Dittert, Vikash Kumar, Shagun Sodhani, Xiaomeng Yang, Gianni De Fabritiis, and Vincent Moens. Torchrl: A data-driven decision-making library for pytorch, 2023.

- A NOTATIONS

Table 3 contains a condensed summary of the notation introduced throughout the paper.

Symbol Meaning n Number of inputs in the AIG Ij Input node j in the AIG O Output node for the AIG ∧i AND-node i for the AIG T⋆ Target truth table

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

s State a Action T Set of truth tables in the AIG A Set of actions N Current number of nodes in the AIG

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

Nmax Max number of nodes allowed in the AIG ShortCircuit generates

[Figure 342]

ǫ Building type of an AND-node with respect to its two fanins (Table 2) A Sparse 3-dimensional tensor accumulating all the target actions Sk Set of permutations of {1..k}

[Figure 343]

[Figure 344]

[Figure 345]

σ(·) Random row permutation function

[Figure 346]

Q(s,a) Discovered Q-value for a state s and action a Q(s,a;θ) Predicted expected Q-value for a state s and action a P(s,a;θ) Predicted action probability distribution

N(s,a) Visit count of action a and state s b,c Parameter balancing exploration and exploitation in PUCT

[Figure 347]

Table 3: List of symbols and notations used in the paper.

- B DATA COLLECTION

- B.1 EPFL BENCHMARKS

Tables 4 and 5, contain detailed information about the arithmetic and random control circuits in the EPFL benchmarks (Amarú et al., 2015), respectively. The circuits have been mapped from behavioral descriptionsinto logic gates and are intentionally suboptimalfor scientiﬁc purposes. Arithmetic circuits, as their name hints, are combinatorial AIGs representing an arithmetic operation such as square root, logarithm, etc., while the set of random control circuits consists of controller circuits.

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

Circuit Name # Inputs # Outputs # AND-nodes Levels

[Figure 355]

Adder 256 129 1020 255 Barrel Shifter 135 128 3336 12

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

Divisor 128 128 44762 4470 Hypotenuse 256 128 214335 24801

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

[Figure 377]

Log2 32 32 32060 444 Max 512 130 2865 287

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

Multiplier 128 128 27062 274 Sine 24 25 5416 225

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

Square-root 128 64 24618 5058

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

Square 64 128 18484 250 Average: 166 102 37396 3608

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

- Table 4: Arithmetic circuits in the EPFL benchmark suite and their statistics

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

Circuit Name # Inputs # Outputs # AND-nodes Levels Round-Robin Arbiter 256 129 11839 87

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

Alu Control Unit 7 26 174 10 Coding-Cavlc 19 11 693 16

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

Decoder 8 128 304 3

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

i2c Controller 147 142 1342 20 Int to Float Converter 11 7 260 16

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

Memory Controller 1204 1231 46836 114

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

Priority Encoder 128 8 978 250 Lookahead XY Router 60 30 257 54

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

Voter 1001 1 13758 70 Average: 284 171 7644 64

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

- Table 5: Random/Control circuits in the EPFL benchmark suite and their statistics

- B.2 CUT EXTRACTION

To extract a cut from a given node with a target number of inputs n, we designate the given node as the root and of the cut and its two parents as the initial leaf set. We then iteratively remove a random node from the leaf set and add its parents to the leaf set while maintaining the leaf property. This process continues until the leaf set contains n nodes. Finally, we create an AIG out of the visited nodes within the cut where we mark the leaf set as the inputs and the root node as the output. We provide the outline of the procedure in Algorithm 1.

[Figure 498]

Algorithm 1 Cut Extraction Require: Root node: r, number of cut inputs: n

[Figure 499]

- 1: leaf_set = {left_parent(r), right_parent(r)}
- 2: node_set = {r}
- 3: while size(leaf_set) < n do
- 4: node = leaf_set.random_pop()
- 5: node_set.insert(node)
- 6: leaf_set.insert(left_parent(node)) & leaf_set.insert(right_parent(node))
- 7: Ensure leaf property in leaf_set
- 8: Construct AIG from leaf_set and node_set

[Figure 500]

We can modify this algorithm to extract additional cuts per node by repeating the process until we ﬁnd a cut with n − 1 leaf nodes. For this cut, instead of randomly expanding a leaf node, we create n − 1 copies of the cut and expand each leaf node individually, storing the resulting cuts. In practice, we actually employ Algorithm 2 to extract AIGs from the EPFL circuits. Although

- Algorithm 1 conveys the core idea of AIG extraction, the following algorithm is more effective from an engineering standpoint, as it allows to extract more cuts from the same node.

The revised algorithm takes two inputs: n, the desired number of input nodes, and the root node. It initializes the leaf set by adding the parents of the root node. It then iteratively removes a random node from the leaf set and expands it using Algorithm 3. This process continues until the cut contains n − 1 leaf nodes, at which point we create n − 1 copies of the current cut state and expand each leaf node to generate n unique cuts. For brevity, we omitted the details of ensuring the leaf property after node expansion in the main text, which is addressed in Algorithm 4. This algorithm ensures that no node in the leaf set has a parent also in this set, which would violate the leaf property. If a parent is already in the leaf set, we can remove the node from the set and add the other parent that is not yet in the leaf set.

[Figure 502]

- Algorithm 2 Multi-Cut Extraction Require: Root node: r, Number of cut inputs: n

[Figure 503]

- 1: leaf_set = {left_parent(r), right_parent(r)}
- 2: node_set = {r}
- 3: while size(leaf_set) < n − 1 do
- 4: node = leaf_set.random_pop()
- 5: Cut_Expansion(node, leaf_set, node_set)
- 6: for leaf in leaf_set do
- 7: copy_leaf_set = leaf_set.copy() & copy_node_set = node_set.copy()
- 8: copy_leaf_set.delete(leaf)
- 9: Cut_Expansion(leaf, copy_leaf_set, copy_node_set)
- 10: Construct AIG from copy_leaf_set and copy_node_set

[Figure 504]

[Figure 505]

- Algorithm 3 Cut Expansion Require: Node to expand: node, Current leaf nodes: leaf_set, Current nodes in cut: node_set

[Figure 506]

- 1: node_set.insert(node)
- 2: leaf_set.insert(left_parent(node)) & leaf_set.insert(right_parent(node))
- 3: preserve_leaf_property(leaf_set, node_set)

[Figure 507]

[Figure 508]

- Algorithm 4 Preserve Leaf Property Require: Current leaf nodes: leaf_set, Current nodes in cut: node_set

[Figure 509]

- 1: for leaf in leaf_set do
- 2: if left_parent(leaf) in leaf_set then
- 3: leaf_set.delete(leaf) & node_set.insert(leaf)
- 4: leaf_set.insert(right_parent(node))
- 5: else if right_parent(leaf) in leaf_set then
- 6: leaf_set.delete(leaf) & node_set.insert(leaf)
- 7: leaf_set.insert(left_parent(node))

[Figure 510]

- C ADDITIONAL RELATED WORKS

- C.1 HEURISTICS FOR AIG GENERATION AND OPTIMIZATION

As the inference of CNFs using exact SAT solvers often lead to exponentially large expressions, various heuristics such as Karnaugh maps (Karnaugh, 1953), or Quine-McCluskey methods (Quine, 1952; 1955; McCluskey Jr., 1956)), and algorithms (Rudell and Sangiovanni-Vincentelli, 1987) have been designed to obtain more compact expressions or circuits. Further efforts accompanying the rise in chip demand led to the development of widely used logic synthesis libraries that implement equivalence-preserving Boolean network operators. The open-source library ABC (Mishchenko et al., 2007)) notably comprises dozens of logic graph operators aiming at reducing a network size or depth (Mishchenko et al., 2011; Mishchenko and Brayton, 2006). Interestingly, some important operators such as resub or rewrite (Darringer et al., 1981; Mishchenko et al., 2006) acts on the subject graph through a series of local modiﬁcations involving small single-output AIGs. Besides, applying a single operator on a logic network is suboptimal compared to applying several operators sequentially, though ﬁnding the best sequence is also a hard problem (Riener et al., 2019).

- C.2 MACHINE LEARNING FOR LOGIC SYNTHESIS

Many ML approaches have been explored to tackle the operator ﬂow optimization progress. Some stateless optimization methods, such as Bayesian optimization (Grosnit et al., 2022; Feng et al., 2022), search for the best ﬂow without considering the subject graph speciﬁcities. Alternatively, state-based methods formulate the operator sequence optimization as a Reinforcement learning problem, and train policies on selected features of the logic network. While some works use high-level statistics of the subject graph (e.g., its number of nodes) (Hosny et al., 2020; Zhou and Anderson,

2023; Qian et al., 2024), others rely on tailored graph convolutional networks (GCN) to extract richer features at the cost of a longer training time (Haaswijk et al., 2018; Peruvemba et al., 2021; Zhu et al., 2020; Basak Chowdhury et al., 2023). Similarly, standard deep network architectures, such as CNNs (Yu et al., 2018), LSTMs (Yu and Zhou, 2020), or GCNs (Wu et al., 2022) have been trained to predict the quality of a logic synthesis ﬂow in a supervised way. Contrary to these works, we target AIG generation itself and not operator optimization.

- D TRAINING PARAMETERS AND IMPLEMENTATION

We implement ShortCircuit with PyTorch and TorchRL (Bou et al., 2023). During pre-training, We use a cosine annealing with warm restarts learning rate scheduler with a starting learning rate of 1 × 10−3 and a batch size of 1024 for 250 epochs. During ﬁne-tuning we use a batch size of 128, a replay buffer with capacity of 1M and sync the parameters every 500 training steps.

