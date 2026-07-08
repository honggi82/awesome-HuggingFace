# arXiv:2602.14699v1[cs.DB]16Feb2026

## Qute: Towards Quantum-Native Database

Xuanhe Zhou∗

Muzhi Chen

Wei Zhou

Surui Tang

Bangrui Xu

Shanghai Jiao Tong University inefable@sjtu.edu.cn

Shanghai Jiao Tong University

Shanghai Jiao Tong University wzdb@sjtu.edu.cn

Shanghai Jiao Tong University tangsurui@sjtu.edu.cn

Shanghai Jiao Tong University

zhouxuanhe@sjtu.edu.cn

dreameter@sjtu.edu.cn

Guoliang Li

Tsinghua University liguoliang@ tsinghua.edu.cn

Bingsheng He

National University of Singapore dcsheb@nus.edu.sg

Yitong Song

Yeye He

Hong Kong Baptist University ytsong@hkbu.edu.hk

Microsoft Corporation

yeyehe@microsoft.com

Fan Wu

Shanghai Jiao Tong University fwu@cs.sjtu.edu.cn

### Abstract

This paper envisions a quantum database (Qute) that treats quantum computation as a first-class execution option. Unlike prior simulation-based methods that either run quantum algorithms on classical machines or adapt existing databases for quantum simulation, Qute instead (i) compiles an extended form of SQL into gate-efficient quantum circuits, (ii) employs a hybrid optimizer to dynamically select between quantum and classical execution plans, (iii) introduces selective quantum indexing, and (iv) designs fidelity-preserving storage to mitigate current qubit constraints. We also present a three-stage evolution roadmap toward quantumnative database. Finally, by deploying Qute on a real quantum processor (origin_wukong), we show that it outperforms a classical baseline at scale, and we release an open-source prototype at https://github.com/weAIDB/Qute.

### 1 Introduction

Quantum databases aim to leverage quantum computers to accelerate workloads that are increasingly infeasible for classical machines. For example, analytical query processing in columnar OLAP systems often depends on linear scans over unindexed tables [46]. By leveraging entangled qubit superpositions and quantum algorithms such as Grover’s search [45], quantum databases can substantially accelerate these queries. Recent research follows two main directions. First, quantum computation for database applies quantum algorithms to classical databases for tasks like query optimization [26, 38], index tuning [17], approximate query processing [43]. However, they remain simulation-based and lack full-stack quantum integration. Second, database for quantum computation leverages classical database techniques to manage quantum workloads, such as encoding quantum circuits in tabular form, but does not consider optimization on the database side [18, 23].

To address these limitations, we pursue an end-to-end quantum-native database, which treats quantum computation not as an isolated accelerator but as a first-class execution substrate integrated throughout query parsing, planning, and execution. However, there are four main challenges. First, SQL operators must be reformulated into representations that map naturally to quantum primitives (e.g., superposition, interference). Second, the optimizer must plan across noise-sensitive quantum operators whose latency, success probability, and approximation error differ fundamentally from classical optimizers. Third, the limited qubit budget makes it necessary to selectively probe only a small fraction of the dataset

∗Xuanhe Zhou is the corresponding author.

Vector Retrieval Interactive BI

Multimodal Analytics Quantum Machine Learning

Ad-Hoc Search Pattern Discovery

Error Detection & Correction Roll-back

End-to-End Quantum Procesor

Quantum-Aware Optimizer

Small-Scale Filtered Data

[Figure 1]

[Figure 2]

Prefix Routing

Fidelity-Aware Cost Model

Groverbased Filter

Error Budget

[Figure 3]

[Figure 4]

[Figure 5]

Quantum Storage

Classical Main Storage

[Figure 6]

Classical Storage

Hybrid Index

Unified Quantum Memory

S3: QuantumNa ve Database

S1: QuantumAssisted Database

S2: QuantumCentric Database

#### Figure 1: Potential Evolution of Quantum Databases.

during quantum execution. Finally, quantum-active attributes must be stored in fidelity-preserving formats to maintain high usability.

In this paper, we introduce Qute, an integrated quantum database architecture that addresses the above challenges. First, Qute compiles extended SQL queries into quantum-gate efficient circuits via a three-tier intermediate representation. Second, it employs a hybrid optimizer that combines quantum and classical operators. Third, it offloads costly operators (e.g., similarity joins) onto quantum circuits that encode computation into probability amplitudes. Fourth, to address scale limitations like qubit numbers, Qute introduces selective quantum indexing based on hybrid tree structures enhanced. Fifth, it adopts a fidelity-aware data format and storage engine supporting compressed tensor networks.

As shown in Figure 1, we envision the development of quantumnative databases across three stages. In the first stage, quantum computing serves as a co-processor for classical systems, where data remains in classical memory and only small subsets are loaded into quantum registers for basic tasks like data filtering. This stage lacks index acceleration and relies on rollback mechanisms to maintain correctness under noise. In the second stage, more data can be stored in quantum memory, enabling hybrid classical–quantum indexing and quantum operators such as joins and aggregations. In the third stage, with scalable fault-tolerant hardware, all data (including multimodal formats like video) can reside in quantum memory, enabling fully quantum-native storage, indexing, processing.

### 2 Preliminary

We presentquantumconcepts supportingQute,including:(1) qubits,

(2) quantum gates, and (3) quantum circuits.

Qubit. Classical computers encode information as deterministic bits in {0, 1}, and solve computation tasks by explicitly enumerating and evaluating candidate solutions one by one. In contrast, quantum computers encode information using qubits, whose states are not restricted to a single deterministic value but can exist as superpositions of the classical basis states |0⟩ and |1⟩. Upon measurement, a qubit collapses to a definite value of 0 or 1 [32]. Formally, a qubit state is |𝜓⟩ = 𝛼 |0⟩ + 𝛽 |1⟩, where amplitudes 𝛼 and 𝛽 encode both magnitude and phase, and measurement yields 0 or 1 with probabilities |𝛼|2 and |𝛽|2 satisfying |𝛼|2 + |𝛽|2 = 1. With qubits, numerous candidate solutions can be represented implicitly [2], and appropriate quantum gates and circuits can amplify the probability of desired solutions, thereby reducing search complexity [15].

Quantum Gate. A quantum gate is an operation that changes qubit states in a controlled way, enabling amplifying correct possibilities while suppressing incorrect ones. Quantum gates are commonly classified into two types: single-qubit gates and multi-qubit gates.

- • (1) Single-qubit gate manipulates the amplitude (e.g., 𝑋-axis and 𝑌-axis rotations and flips) and phase (e.g., 𝑍-axis rotations) of an individual qubit to achieve operations essential for data representation and quantum computation. For state preparation, (𝑖) the Hadamard (H) gate offers a uniform initial representation for quantum computation by mapping a basis state into an equal superposition of |0⟩ and |1⟩; (𝑖𝑖) X-axis and Y-axis rotations (i.e., 𝑅𝑋 (𝜃) and 𝑅𝑌 (𝜃)) encode row values by adjusting the amplitude contributions of a qubit between |0⟩ and |1⟩; (𝑖𝑖𝑖) the Pauli-X flip operates a qubit between |0⟩ and |1⟩, which is used to mark specific basis states. For phase adjustment, Z-axis rotation gate (i.e., 𝑅𝑍 (𝜃)) modifies the relative phase between |0⟩ and |1⟩, which determines whether different possibilities reinforce or cancel each other (thereby amplifying the probability of target rows). For value encoding, parameterized

rotations 𝑅𝛼 (𝜃), where 𝛼 ∈ {𝑋,𝑌,𝑍}, adjust both magnitudes and phases, allowing numerical information to be embedded [30, 32].

- • (2) Multi-qubit gate performs conditional operations and create entanglement among qubits, thereby establishing their correlations. Such correlations are essential for quantum parallelism and computational speedups that cannot be achieved on classical computers. Multi-qubit gates can be broadly classified into three categories: (𝑖) Controlled gates act on target qubits only when the control qubits are in specific states; (𝑖𝑖) Entangling gates bind multiple qubits together to create non-classical correlations; (𝑖𝑖𝑖) Permutation gates exchange the states of qubits to rearrange data. For example, Controlled-SWAP (CSWAP) gate, which combines controlled and permutation operations, swaps the states of two target qubits only when the control qubit is in a specific state [29].

Quantum Circuit. A quantum circuit organizes a sequence of quantum gates to implement quantum algorithms and produce the desired results. We use a representative quantum circuit (i.e., Grover’s Algorithm [16]) to illustrate its potential for accelerating data filtering. Given a boolean predicate 𝑓 : X → {0, 1} (in the SQL WHERE clause) over a dataset of size 𝑁 = |X|, data filtering identifies all records 𝑥 ∈ X such that 𝑓 (𝑥) = 1. The circuit begins by uniformly encoding the full dataset via the H gate, and then organizes a sequence of gates with the following two operators.

- • (1) Oracle (𝑶𝒇 ): The oracle acts as a predicate evaluator, implemented as a quantum operator that encodes the query condition

#### Table 1: Classical vs. Quantum Database Components.

Component Classical Database Quantum Database Compiler Generate efficient machine code for

Generate circuits executable directly on the quantum backend.

classic execution (CPUs, GPUs, ...).

Operator Perform tuple-at-a-time or vector-

Replace tuple scans with amplitude and interference operations.

ized processing (CPU cost ↓).

Plan Optimizer Select the lowest-cost deterministic execution path.

Dynamically balance quantum speedup vs. noise / success rate.

Shrink the candidate set that must be loaded into quantum computer.

Data Indexing Accelerate lookup operations.

Data Format Byte-addressable and easy to parse. Encode data in qubit states. Storage Engine Ensure durability and efficient I/O. Maintain reconstructable large-scale

data with compression techniques.

(e.g., 𝑓 (𝑥) = 1). Applied to a superposition |𝑥⟩, it marks records satisfying the predicate by a phase inversion, i.e., 𝑂𝑓 |𝑥⟩ = (−1)𝑓 (𝑥) |𝑥⟩. • (2) Diffusion Operator (𝑫): This operator amplifies success probability by reflecting probability amplitudes about their mean, increasing the amplitudes of marked states and suppressing others. It is implemented by applying H gates to all qubits, performing a multi-qubit phase inversion on |0 · · · 0⟩ state, and then reversing the Hadamard transformation, i.e., 𝐷 = 2 |𝑠⟩ ⟨𝑠| − 𝐼.

√︁𝑁/𝑀 steps (where 𝑀 is the matched record count), measurement yields the matched record with high probability. Unlike a classical linear scan requiring 𝑂(𝑁) predicate evaluations, the circuit finds matching records using 𝑂(

By iteratively applying the operator 𝐺 = 𝐷𝑂𝑓 for 𝑘 ≈ 𝜋4

√

𝑁) evaluations (i.e., the worst case 𝑀 = 1), highlighting its advantage for filtering over unindexed data.

### 3 System Overview

We present differences between classical and quantum databases (Table 1) and we outline the key components of Qute (Figure 2).

Design Principles. Quantum databases follow several design principles that differ from classical ones. (1) Minimize quantum data loading by aggressively shrinking the quantum-active footprint (e.g., via selective indexing, pre-filtering, on-demand encoding). (2) Leverage amplitude-level parallelism instead of tuple-by-tuple processing (e.g., reformulating as quantum operators). (3) Adopt fidelity-aware optimization strategies (e.g., potential noise in quantum operators). (4) Maintain hybrid executability by keeping both quantum and classical paths viable, thereby resolving hardware noise and resource constraints.

Quantum SQL Compiler. Qute integrates a quantum-native compilation stack: (1) We handle standard relational operators (e.g., filtering, aggregation) alongside quantum-friendly functions such as sampling-based estimation and probabilistic evaluation, which can be realized using primitives like amplitude estimation [36]. Note we can automatically generate gate-efficient circuits for each operator via a modular circuit generator [33]; (2) We employ a threetier LLVM-style intermediate representation (logical IR, quantumextended IR, and physical circuit IR), supporting stepwise lowering from declarative SQLs to executable quantum circuits [37]; (3) We maintain a dynamic catalog of rewrite rules that uncovers useful transformations for complex joins, and multi-predicate filter queries via specific quantum features like superposition-enabled probabilistic branching and interference-based correlation detection.

Quantum Plan Optimizer. Qute models each quantum operator 𝑞 using a compact profile (𝑇𝑞,𝑃𝑞,𝜀𝑞) that summarizes its latency,

###### SQL Query

##### Query Engine

Execution Engine

Quantum Operators

###### Quantum-Aware Query Parser

Grover

Quantum Filter

exact operators

statistics operators

approximate operators

Grover+ CSWAP

Quantum Join

Amplitude Estimation

Quantum Agg

Operator Compiler

QuantumAware Rewriter

Three-Level LLVM IR

###### ...

[Figure 7]

[Figure 8]

[Figure 9]

Quantum-Aware Indexing

quantum circuits

Query Optimizer

[Figure 10]

Hybrid Plan Generation

Fidelity-Aware Cost Model

Runtime Optimization

success probability

calibration overhead

latency

Storage Engine

Statistical Estimates

LOAD(TN_ID, ε)

fidelity F TN Version

SAMPLE(TN_ID, k)

Entropy Draim

REFRESH(TN_ID)

ancilla sub-registers quantum registers amplitude loading

Table Data

core / vector / auxiliary attributes

#### Figure 2: Example Quantum Database System Design.

success probability, and approximation error, allowing the optimizer to compare it directly with classical operators. Using these profiles, execution plans embed both quantum and classical realizations of eligible operators, thereby keeping domain choices flexible. At runtime, Qute monitors quality and stability, adapting to noise by increasing shots, switching to more robust circuit variants, or falling back to classical paths when quantum performance degrades.

Quantum-Accelerable Operators. Qute offloads costly operations to quantum subroutines that encode computation into probability amplitudes rather than row-wise execution [10, 41]. For instance, for similarity joins, Qute bypasses dimension-wise traversal by encoding vectors into quantum states and estimating their inner products through interference-based overlap.

Quantum-Aware Indexing. To address the quantum memory bottleneck, we develop index structures that enable selective quantum probing, meaning that only index entries relevant to the query are probed before allocating quantum resources. We begin with lightweight quantum scans over one-dimensional B+-trees to quickly narrow the candidate set. When this yields only a small number of candidates, the remaining work is completed via classical filtering. However, if many candidates persist, we transition to a quantum KD-tree that jointly indexes multiple dimensions, enabling more aggressive pruning without exhaustively scanning all rows.

Quantum Data Format. (1) Basis Encoding: Core row identifiers like 𝐼𝐷𝑟𝑜𝑤 are mapped to computational basis states, enabling uniform superposition and parallel oracle processing. (2) Amplitude Encoding: Data vectors (e.g., embeddings) are encoded into state amplitudes via rotation gates, compressing values into probability distributions aligned with row indices. (3) Control Encoding: Metadata and flags are stored in ancilla qubits that act as control lines, enabling conditional gate execution based on row-level attributes.

Quantum Storage Engine. Unlike page-based designs [24], Qute adopts a state-centric approach. Quantum-active attributes are stored as compressed tensor networks (TNs), acting as logical pages that support bounded reconstruction and sampling under fidelity constraints. The engine provides fidelity-aware primitives to enable controlled approximation while tracking entropy and noise: LOAD(TN_ID, 𝜖) for approximate access, SAMPLE(TN_ID, 𝑘) for bounded sampling, and REFRESH(TN_ID) to restore fidelity when degradation exceeds threshold. Classical metadata is stored separately to ensure recoverability.

### 4 Quantum-Accelerable Operators

Building on the basic circuits introduced in Section 2, we design quantum circuits tailored to core database operations. Each operator is restructured to exploit quantum parallelism while preserving relational semantics. Table 2 summarizes the quantum-accelerable operators and we illustrate three representative types below.

### 4.1 Quantum Data Filtering

Large-scale data filtering is a basic database operation, yet it incurs high latency on classical systems when processing sparse, unindexed data because of exhaustive full-table scans. To overcome this bottleneck, Qute adapts Grover’s algorithm to a quantum data filtering operator. We design a custom quantum circuit that implements a sequence of logic and diffusion gates, replacing the classical linear scan with an accelerated quantum search.

Circuit Design. Specifically, we map the row identifier (𝐼𝐷𝑟𝑜𝑤) domain as the search space and compile 𝜙 into a quantum oracle 𝑂𝜙 that marks satisfying rows (as 𝑂𝑓 in Section 2). Compound SQL predicates are transformed into phase-marking logic via ancillaassisted encoding. For example, a predicate such as age > 30 AND city = ‘NY’ is implemented by first evaluating each subcondition into ancilla qubits and then combining them using multi-controlled operations to mark the final target states. After Grover iterations amplify the amplitude of matched 𝐼𝐷𝑟𝑜𝑤, Qute performs classical reconciliation, i.e., deduplicating matched 𝐼𝐷𝑟𝑜𝑤 and rechecking them against 𝜙 to guarantee correctness.

### 4.2 Quantum Similarity Join

Similarity joins are fundamental in databases and multi-modal data retrieval [3, 8, 39, 44]. They aim to retrieve all pairs of vectors from two datasets whose distances are within a given threshold. The core operation of a similarity join is distance computation (e.g., inner product), which determines the set of vectors that are similar (i.e., with small distance) to each input vector. Classical machines typically incur high costs to compute inner products between vectors (e.g., due to repeated dimension-wise multiplication and distance evaluations across numerous vector pairs). Instead, Qute encodes vectors into quantum states and computes the inner product without explicitly traversing all dimensions or candidate vector pairs.

Circuit Design. As illustrated in Figure 3, given two normalized vectors 𝑥, 𝑦 ∈ R𝑑, Qute first encodes them into quantum states |𝑥⟩ and |𝑦⟩. Each vector is transformed into a quantum state using parameterized single-qubit rotations (e.g., 𝑅𝑌 (𝜃)), following the vector-to-state mapping described in Section 3. This encoding ensures that the inner product between 𝑥 and 𝑦 is reflected in the

#### Table 2: Quantum-Accelerable Relational Operators (𝑀 and 𝑁 are total and qualifying row numbers; 𝑑 is the vector dimension; 𝜖 is precision; 𝑛 and 𝑞∗ are qubit counts for data and precision registers; 𝑎∗ is ancilla; 𝑏 is bandwidth; 𝐷 is the gate depth).

Complexity Cost Model Quantum Classical

Type Operator (SQL)

Typical Scenario Candidate Quantum Algorithm

Qubit Num Gate Depth (per-circuit call)

(without index)

𝑁 ) 𝑂(𝑁 ) 𝑛 + 𝑎orc 𝑂 (𝐷orc + 𝐷diff ) · √︁𝑁/𝑀

√

Equality Filter WHERE column = value Grover [16] (Search) 𝑂(

𝑁 ) 𝑂(𝑁 ) 𝑛 +𝑏 + 𝑎cmp 𝑂 (𝐷cmp(𝑏) + 𝐷diff ) · √︁𝑁/𝑀

√

Range Filter WHERE column > value Grover [16] (Threshold Oracle)

𝑂(

Filter

𝑁 ) 𝑂(𝑁 ) 𝑛 +𝑏 + 𝑎pref 𝑂 (𝐷pref (𝑏) + 𝐷diff ) · √︁𝑁/𝑀

√

LIKE WHERE column LIKE ‘val%’ Grover [16] (Prefix-Match Oracle)

𝑂(

√

√

EXISTS Existence check Grover [16] (Quantum Counting)

𝑁 ) 𝑂(𝑁 ) 𝑛 + 𝑎orc 𝑂 (𝐷orc + 𝐷diff ) ·

𝑂(

𝑁

√𝑁2

√𝑁2) / probe 𝑂(𝑁1 · 𝑁2) 𝑛2 +𝑏key + 𝑎orc 𝑂 (𝐷idx + 𝐷key + 𝐷diff (𝑛2)) ·

Equi-join column1 = column2 Grover [16] (Index Probing)

𝑂(

/probe

Join

√𝑁2

O(√𝑁2) / probe 𝑂(𝑁1 · 𝑁2) 𝑛2 +𝑏key +𝑏 +

Non-equi join column1 ≤ column2 Grover [16]

𝑂 (𝐷cmp + 𝐷diff (𝑛2)) ·

(Comparison Oracle)

𝑎cmp

/probe Similarity Join Similarity search Grover [16]

√

𝑁 ) 𝑂(𝑑 · 𝑁 ) 2𝑞 + 𝑎prep + 1 𝑂 2 · 𝐷prep(𝑑) + 𝑂(𝑞)

𝑂(

(SWAP Test)

√

√

MIN Global minimum search Dürr–Høyer Minimum Finding [9]

𝑁 ) 𝑂(𝑁 ) 𝑛 +𝑏 + 𝑎cmp 𝑂 (𝐷cmp + 𝐷diff ) ·

𝑂(

𝑁 COUNT Cardinality estimation Amplitude Estimation [6] 𝑂(1/𝜖) 𝑂(𝑁 ) 𝑛 + 𝑎orc + 𝑞𝜖 𝑂((𝐷orc + 𝐷diff ) · (1/𝜖))

Aggregation

Normalization +

Amplitude Estimation [6] 𝑂(1/𝜖) 𝑂(𝑁 ) 𝑛+𝑏𝑣 +𝑎rot +𝑞𝜖 𝑂((𝐷load + 𝐷rot + 𝐷diff ) · (1/𝜖)) AVG Mean value calculation

SUM Total value calculation

Normalization +

Amplitude Estimation [6] 𝑂(1/𝜖) 𝑂(𝑁 ) 𝑛+𝑏𝑣 +𝑎rot +𝑞𝜖 𝑂((𝐷load + 𝐷rot + 𝐷diff ) · (1/𝜖)) Other Sampling Approximate query Amplitude Estimation [6] 𝑂(

𝑁 ) 𝑂(𝑁 ) 𝑛 + 𝑎orc 𝑂 (𝐷orc + 𝐷diff ) · √︁𝑁/𝑀

√

overlap between the corresponding quantum states. To estimate this overlap, Qute employs the SWAP Test [10], a standard quantum subroutine for inner product estimation. Specifically, an ancilla qubit is initialized into a uniform superposition using an H gate, after which a CSWAP operation is applied between the two data registers conditioned on the ancilla. A second H gate is then applied to the ancilla qubit. Upon measuring the ancilla qubit, the probability of observing outcome |0⟩ is given by Pr[ancilla = 0] = (1+|⟨𝑥|𝑦⟩|2)/2. By repeatedly executing the circuit and measuring the ancilla qubit, Qute estimates |⟨𝑥|𝑦⟩|2 from the empirical measurement frequency. Classical post-processing converts measurement outcomes into approximate inner products for identifying vector pairs.

where𝑉max is a known upper bound on the aggregated values. Intuitively, rows with larger values contribute more strongly to the probability of observing the Good flag. The overall probability of measuring the Good qubit as |1⟩ corresponds to the average of the normal-

ized values across all rows, i.e., 𝑃𝑟 [Good = 1] = 𝑁1 𝑥 𝑁=−01 𝑉𝑣max(𝑥) . By estimating using amplitude estimation and scaling it by 𝑁 and𝑉max, Qute recovers the desired result: SUM(𝑣) = 𝑁 ·𝑉max ·Pr[Good = 1].

### 5 Quantum-Aware Index Structure

Due to limited qubit capacity and circuit depth, existing quantum computers can encode only a small fraction of data in superposition [18], making selective data loading essential. Existing approaches load only relevant B+-tree leaf nodes into the quantum processor, but this strategy generalizes poorly to multi-dimensional queries and suffers significant performance degradation. We therefore propose a quantum-accelerated indexing strategy that narrows the query scope early via lightweight probing, supporting both oneand multi-dimensional workloads.

### 4.3 Quantum Aggregation

Classical aggregations (e.g., SUM) compute results by iterating through all qualifying rows and accumulating their values. It incurs high latency when the qualifying set is large, as it scales linearly with the row volumes. Quantum systems encode information as probability amplitudes, making it infeasible to replicate a classical accumulator directly. To address this, Qute reformulates aggregation as a probability estimation task so that I/O does not dominate end-to-end cost with coherent vector access (e.g. via QRAM). By mapping numeric values to the observation probability of a target qubit, it utilizes amplitude estimation [6] to compute the global sum efficiently without iterating through individual rows.

Our strategy integrates quantum probing on selected dimensions with classical post-filtering. For multi-column predicates (Figure 4), we first probe several one-dimensional quantum B+-trees constructed on selected dimensions. This probing identifies nodes that are fully contained in, or partially overlap with, the query range [27]. For example, for the query 𝑞1 : 𝑑1 ∈ [5, 8], a quantum probe returns 𝑘1 = 4 tuples satisfying the predicate on 𝑑1. Among the result sets returned by all probed B+-trees, we select the smallest one as the candidate set, and denote its size by 𝑘𝑠. Based on 𝑘𝑠, we dynamically choose between two strategies.

Circuit Design. Qute first prepares a uniform superposition over all 𝑁 candidate row identifiers. For each row identifier |𝑥⟩, the corresponding value 𝑣(𝑥) is accessed and used to control a dedicated flag qubit, referred to as the Good qubit. Specifically, a controlled rotation is applied so that the probability of the Good qubit being measured as |1⟩ is proportional to the normalized value 𝑣(𝑥)/𝑉max,

• (1) Classical Post-Filtering. If the number of candidates is small (i.e., 𝑘𝑠 = 𝑂(log 𝑁)), we perform classical verification on the remaining 𝑑 − 1 dimensions. For query 𝑞1, we further filter the 4 retrieved rows using the secondary predicate 𝑑2 ∈ [1, 6]. It incurs a cost of 𝑂(𝑘𝑠 · (𝑑 − 1)), efficient due to the small candidate size.

QSELECT INNER_PRODUCT(v1, v2) AS ip FROM vectors v1, vectors v2 USING SWAP_TEST_KERNEL ENCODING amplitude(dimensions = [1, 2]) SHOTS 2048

Z

q[2]

(AncillaQubit)

- q[0] ( )
- q[1] ( )

0

Result

Amplify Encoding H & CSWAP Measurement

[Figure 11]

[Figure 12]

[Figure 13]

Pr[ancilla = 0] Introducing Ancilla Computing for state overleap estimation

2. Compu ng Similarity via State Interference

1. Encoding Vectors as Rota on Angles

3. Reading Similarity from Z-axis Projec on

#### Figure 3: CSWAP-Gate Based Inner Product Calculation.

Quantum B+Tree (d1)

…

q₁: d₁∈[5,8] and d₂∈[1,6] q2: d₁∈[1,9] and d₂∈[2,10]

intermediate tree node

[Figure 14]

Quantum Filtering

| |3.45| |5.34| |6.67| |
|---|---|---|---|---|---|---|

|Leaf1|d1<br><br>…|d2<br><br>…|
|---|---|---|
| | | |
| |…|…|
| |…|…|

|Leaf2|d1|d2|
|---|---|---|
| |3.45|7.81|
| |5.01|4.56|
| |…|…|

|Leaf3|d1|d2|
|---|---|---|
| |5.34|4.56|
| |5.67|3.21|
| |…|…|

|Leaf4|d1|d2|
|---|---|---|
| |6.67|2.34|
| |8.45|6.78|
| |…|…|

Small ks(4) Large ks(8)

|Classical Post-Filtering<br><br>Data<br><br>|Query by d1|
|---|
<br><br>[Figure 15]<br><br>k0 rows<br><br>[Figure 16]<br><br>[Figure 17]<br><br>classical computer|
|---|

|Quantum Multi-Divided KD-Tree<br><br>Main Tree (d1 index) 6.67<br><br>…<br><br>3.21<br><br>d1<br><br>d2<br><br>2.34<br><br>d2<br><br>5.67<br><br>d1<br><br>3.21<br><br>d1<br><br>6.67<br><br>d1<br><br>2.34<br><br>d1<br><br>… … QuantumFiltering<br><br>[Figure 18]|
|---|

#### Figure 4: Quantum-Aware Multi-Dimensional Index.

• (2) Quantum Multi-Divided KD-Tree Search. If 𝑘𝑠 is large, classical filtering becomes less effective. We escalate to a quantum multi-divided KD-tree that jointly indexes multiple dimensions. By recursively partitioning the data along different dimensions and applying quantum-assisted probing at each level, this structure significantly reduces the candidate set before verification, making it more suitable for large 𝑘𝑠 scenarios.

Additionally, for queries involving disjunctive conditions (e.g., 𝑑1 ∈ [5, 8] ∨𝑑2 ∈ [2, 5]), we independently probe each dimension and combine the resulting superpositions.

### 6 Quantum-Aware Optimizer

Classical query optimizers rely on deterministic assumptions that fail to capture the stochasticity and noise inherent in quantum hardware. Qute addresses this limitation by treating quantum operators as probabilistic, accuracy-bounded primitives, thereby shifting the optimization goal from pure latency minimization to a robust tradeoff between speed and reliability.

Estimation Model. The cost model extends classical optimization by treating a quantum operator as a stochastic, accuracy-bounded primitive rather than a deterministic function. The optimizer evaluates each operator using a compact tuple (𝑇𝑞,𝑃𝑞,𝜀𝑞) that captures its latency, success probability, and approximation error, which are quantified directly from circuit structure and hardware characteristics. Each factor is explicitly incorporated into the objective function, allowing the optimizer to trade off execution speed against expected reliability under real hardware behavior.

- • (1) Time Estimation (𝑻𝒒). Execution time is derived from the circuit’s layerized schedule after topology-aware routing. For layers

𝐿 = {𝐿1, . . .,𝐿𝐾}, 𝑇𝑞 = 𝑘 𝐾=1 max𝑔∈𝐿𝑘 𝑡𝑔 + 𝑡ctrl , where 𝑡𝑔 is the calibrated duration of each gate and 𝑡ctrl accounts for control and synchronization overheads between layers. It involves two latency contributors: (i) serialization imposed by limited qubit connectivity and (ii) SWAP-induced depth inflation, both of which are used to assess whether a quantum operator outperforms the classical ones.

- • (2) Noise and Success Probability (𝑷𝒒). Quantum success probability is estimated as a circuit-level composition of gate errors

, with 𝜖𝑔 the hardware-reported gate error rates and𝑇2eff an effective

and decoherence. For each layer, 𝑝𝑘 = 1 − 𝑔∈𝐿𝑘 𝜖𝑔 · exp −𝑇𝑡𝑘eff

2

coherence bound. The operator-level success probability is approximated by 𝑃𝑞 = 𝑘 𝐾=1 𝑝𝑘. It ties plan quality directly to circuit depth and layout: deeper or poorly routed circuits incur multiplicative degradation, guiding the optimizer to prune overly fragile plans.

• (3) Hybrid Execution Error (𝜺𝒒). Each quantum operator is paired with a deterministic classical fallback. The optimizer evalu-

ates its expected runtime as E[𝑇𝑞] = 𝑃𝑞 ·𝑇𝑞quantum+(1−𝑃𝑞)·𝑇𝑞classical, ensuring that incorrect or failed executions do not compromise

result correctness. The approximation error 𝜀𝑞, derived from the operator’s amplitude, estimation precision, or sampling budget, is incorporated as a constraint or penalty depending on the query’s accuracy requirements. Parameters are updated using runtime observations, enabling the optimizer to refine plan choices adaptively.

By grounding each component of (𝑇𝑞,𝑃𝑞,𝜀𝑞) in explicit circuit and hardware features, the model provides a concrete and actionable basis for quantum-classical plan cost modeling.

Hybrid Plan Generation. In Qute, each plan encodes both quantum and classical implementations of selected operators, preserving semantic equivalence while exposing explicit alternatives for cost and quality trade-offs. Importantly, cross-domain coordination costs (e.g., data movement between classical and quantum runtimes) are explicitly incorporated into the optimizer’s objective function, preventing overly optimistic cost estimates. Furthermore, Qute supports deferred binding of execution-domain choices, allowing the runtime dispatcher to select the most suitable realization based on real-time conditions (e.g., backend calibration state, queue delays). Adaptive Runtime Optimization. To ensure robust execution in the presence of quantum noise and stochastic failures, Qute integrates adaptive runtime optimization as a first-class mechanism. When a quantum path fails to meet declared quality or latency thresholds, the system dynamically reallocates effort by increasing the number of sampling shots, switching to more robust circuit variants, or falling back to classical computation. Critically, all such adaptations operate under bounded-error semantics: fallback paths are validated for correctness in advance, and all runtime decisions are constrained by the optimizer’s declared error and latency budgets. This architecture ensures that query results remain trustworthy even when quantum hardware fails unpredictably, allowing Qute to balance optimism with reliability in real deployments.

### 7 Preliminary Results

Prototype Implementation. We implement a minimal prototype of Qute that supports database filtering operations in [1]. The classical execution engine handles SQL parsing, query planning, and result reconciliation, while data filtering is offloaded to a quantum computing backend. The prototype is implemented using the QPanda3 library (v1.0) and developed on the origin_wukong platform. All quantum circuits are executed on a real QPU with 72 qubits and noise. Each quantum experiment is executed with 2000 measurement shots by default, and the results are averaged.

Dataset. To evaluate the effectiveness of quantum computing over large-scale data, we construct a synthetic dataset with 𝑁 = 240 tuples using a random generation method. We consider filter queries of the form SELECT RID FROM T WHERE 𝜙(𝑥), where 𝜙 consists of single-attribute or conjunctive predicates over the generated attributes. Predicate parameters are chosen to control query selectivity, and the selectivity of each predicate is bounded by 2%, with multiple selectivity levels generated within this bound.

Evaluation Methods. Given that classical database runtime is highly sensitive to hardware and system optimizations, we use a cost-model-based evaluation for a fair comparison of algorithmic complexity. Specifically, we evaluate two approaches: (1) a classical database baseline, whose performance is estimated using a traditional cost model [28]; and (2) Qute, whose performance is measured using actual execution when the data size permits and otherwise estimated using our cost model (Section 6).

Results. Figure 5 compares the processing time of Qute and a classical database under increasing data scale. For Qute, the measured runtime on small datasets (up to 210) is shown as discrete sample points with a Grover error of ±8.0%, while the runtime estimated by our cost model is shown as a continuous curve. The close agreement between the measured and estimated results indicates that the cost model accurately captures the performance of Qute and can be reliably used for large-scale extrapolation. The estimated performance of the classical database is also reported for comparison. The results reveal a clear crossover point at approximately 𝑁 ≈ 230. For smaller datasets, the classical database is faster due to lower constant overhead, whereas for larger datasets, Qute consistently outperforms the classical approach. It highlights the scalability advantage of Qute for extreme-scale, unindexed filtering workloads.

### 8 Evolution and Further Directions

We classify the evolution of quantum databases into three stages (S1-S3) based on data scale and computational capability (Figure 1). ❶ Quantum-Assisted Database. Quantum computing functions as a co-processor to classical systems. Due to limited qubits, short coherence times, and high data-loading costs, quantum resources are used only for selective, compute-intensive tasks such as fulltable scans using Grover’s search. Data remains in classical memory, with quantum execution triggered in real time. There is no index acceleration, and fallback mechanisms ensure correctness when quantum runs fail or exceed noise thresholds. The primary bottleneck lies in the classical-to-quantum transfer bandwidth.

❷ Quantum-Centric Database. With improved quantum memory and loading methods, quantum computing becomes the main

Classical (estimated, with penalty)

1014

Qute (estimated by our cost model)

1012

Qute (k=1..10) ±8.0%

Crossover

1010

Time(ms)

108

crossover Time * = 219s

106

104

0 5 10 15 20 25 30 35 40

k (N = 2^k)

Figure 5: Performance Comparison (Qute vs. Classic). execution engine. Hybrid quantum-classical indexes emerge, enabling selective access to superposition states. A fidelity-aware optimizer plans across quantum and classical paths using cost models that account for decoherence, gate latency, and qubit limits. Routing mechanisms dispatch subqueries accordingly. Core operations such as joins and aggregations are implemented using quantum primitives, including SWAP tests and Amplitude Estimation, enabling efficient hybrid execution.

❸ Quantum-Native Database. At this stage, fault-tolerant quantum hardware supports more powerful storage and computation. All data types (including structured, unstructured, multimodal) reside in quantum memory. Advanced data encodings (e.g., vector quantization algorithms) compress and manipulate large datasets directly. Classical components are minimized, and the system achieves fully quantum-native storage, indexing, and processing.

### 9 Related Work

Quantum Computation for Databases. Prior works include surveys and efforts in query processing [7, 19], query optimization [11, 12, 26, 31, 35, 38], index selection [17, 22], data integration [13], and transaction scheduling [4, 5, 14]. Other efforts explore quantum-assisted approximate query processing [43] and data structures [25]. While promising, they largely retain classical data models and storage layers, using quantum computation as an isolated acceleration step without cross-layer integration.

Databases for Quantum Computation. Several works explore how classical database abstractions can support quantum computation. This includes storing and manipulating relational data on quantum hardware [24, 34], compiling SQL queries to quantum circuits [23]. Quantum machine learning has also been applied to relational and graph data [20, 21, 40, 42], though these focus on model training rather than database execution.

### 10 Conclusion

This paper envisions quantum-native databases as a foundational shift in data management, in which quantum computation is integrated across the full stack, from query parsing to execution. We proposed Qute, a system that addresses key challenges in operator compilation, hybrid optimization, selective indexing, and fidelity-aware storage. Unlike prior simulation-based approaches, Qute demonstrates a unified design that supports practical quantum query execution. We outlined a three-stage roadmap highlighting both architectural milestones and research opportunities for the future of quantum data systems.

### References

- [1] Qute. (Quantum Database Prototype). https://github.com/weAIDB/Qute
- [2] Pablo Arrighi and Gilles Dowek. 2012. The Physical Church-Turing Thesis and the Principles of Quantum Theory. Int. J. Found. Comput. Sci. 23, 5 (2012), 1131–1146.
- [3] Martin Aumüller and Matteo Ceccarello. 2022. Implementing Distributed Approximate Similarity Joins using Locality Sensitive Hashing. In EDBT: International Conference on Extending Database Technology.
- [4] Tim Bittner and Sven Groppe. 2020. Avoiding Blocking by Scheduling Transactions Using Quantum Annealing. In Proceedings of the 24th Symposium on International Database Engineering & Applications (IDEAS). ACM, 1–10.
- [5] Tim Bittner and Sven Groppe. 2020. Hardware Accelerating the Optimization of Transaction Schedules via Quantum Annealing by Avoiding Blocking. Open Journal of Cloud Computing 7, 1 (2020), 1–21.
- [6] Gilles Brassard, Peter Høyer, Michele Mosca, and Alain Tapp. 2002. Quantum Amplitude Amplification and Estimation. Contemp. Math. 305 (2002), 53–74.
- [7] Umut Çalikyilmaz, Sven Groppe, Jinghua Groppe, Tobias Winker, Stefan Prestel, Farida Shagieva, Daanish Arya, Florian Preis, and Le Gruenwald. 2023. Opportunities for Quantum Acceleration of Databases: Optimization of Queries and Transaction Schedules. Proc. VLDB Endow. 16, 9 (2023), 2344–2353.
- [8] Yanqi Chen, Xiao Yan, Alexandra Meliou, and Eric Lo. 2025. DiskJoin: Large-scale Vector Similarity Join with SSD. ACM SIGMOD 3, 6 (2025), 1–27.
- [9] Christoph Dürr and Peter Høyer. 1996. A Quantum Algorithm for Finding the Minimum. In Proceedings of the 5th International Conference on Quantum Computing and Quantum Communications (QCQC). Springer, Berlin, Heidelberg, 362–371.
- [10] Wang Fang and Qisheng Wang. 2025. Optimal Quantum Algorithm for Estimating Fidelity to a Pure State. In ESA (LIPIcs, Vol. 351). Schloss Dagstuhl - LeibnizZentrum für Informatik, 4:1–4:12.
- [11] Tobias Fankhauser, Marc E. Solèr, Rudolf M. Füchslin, and Kurt Stockinger. 2021. Multiple Query Optimization Using a Hybrid Approach of Classical and Quantum Computing. arXiv preprint arXiv:2107.10508 (2021). https://arxiv.org/abs/2107. 10508
- [12] Tobias Fankhauser, Marc E. Soler, Rudolf M. Füchslin, and Kurt Stockinger. 2023. Multiple Query Optimization Using a Gate-Based Quantum Computer. IEEE Access 11 (2023), 23066–23081. doi:10.1109/ACCESS.2023.3254637
- [13] Kai Fritsch and Stefanie Scherzinger. 2023. Solving Hard Variants of Database Schema Matching on Quantum Computers. Proceedings of the VLDB Endowment 16, 12 (2023), 3990–3993.
- [14] Sven Groppe and Jinghua Groppe. 2021. Optimizing Transaction Schedules on Universal Quantum Computers via Code Generation for Grover’s Search Algorithm. In Proceedings of the 25th International Database Engineering & Applications Symposium (IDEAS). ACM, 149–156.
- [15] Lov K. Grover. 1996. A Fast Quantum Mechanical Algorithm for Database Search. In STOC. ACM, 212–219.
- [16] Lov K. Grover. 1996. A fast quantum mechanical algorithm for database search. In Proceedings of the 28th Annual ACM Symposium on Theory of Computing (STOC). 212–219.
- [17] Le Gruenwald, Rui Zhang, and Jialu Zhou. 2023. Index Tuning with Machine Learning on Quantum Computers for Large-Scale Database Applications. In Proceedings of the VLDB 2023 Workshops. VLDB Endowment.
- [18] Rihan Hai, Shih-Han Hung, Tim Coopmans, Tim Littau, and Floris Geerts. 2025. Quantum Data Management in the NISQ Era. Proc. VLDB Endow. 18, 6 (2025), 1720–1729.
- [19] Rihan Hai, Shih-Han Hung, and Sebastian Feld. 2024. Quantum Data Management: From Theory to Opportunities. In ICDE. IEEE, 5376–5381.
- [20] Mohsen Heidari, Ananth Y. Grama, and Wojciech Szpankowski. 2022. Toward Physically Realizable Quantum Neural Networks. In Association for the Advancement of Artificial Intelligence (AAAI). https://www.cs.purdue.edu/homes/spa/ papers/AAAI21.pdf Implementation and training scheme for quantum neural networks.
- [21] Mohsen Heidari, Arun Padakandla, and Wojciech Szpankowski. 2021. A Theoretical Framework for Learning from Quantum Data. In IEEE International Symposium on Information Theory (ISIT). https://arxiv.org/abs/2107.06406 Theoretical foundations for learning from quantum data.
- [22] Manish Kesarwani and Jayant R. Haritsa. 2024. Index Advisors on Quantum Platforms. Proceedings of the VLDB Endowment 17, 12 (2024).
- [23] Manish Kesarwani and Jayant R. Haritsa. 2024. Is Quantum-Based SQL Query Execution Viable?. In VLDB Workshops. VLDB.org.
- [24] Tuodu Li, Gongsheng Yuan, Chang Yao, Meng Shi, Ziyue Wang, Ling Qian, and Jiaheng Lu. 2024. Quantum Storage Design for Tables in RDBMS. In VLDB

- Workshops. VLDB.org.
- [25] Tim Littau, Ziyu Li, and Rihan Hai. [n.d.]. Towards Quantum Data Structures for Enhanced Database Performance. VLDB 2024 Workshop 2150 ([n.d.]), 8097.
- [26] Hanwen Liu, Federico M. Spedalieri, and Ibrahim Sabek. 2025. A Demonstration of Q2O: Quantum-augmented Query Optimizer. Proceedings of the VLDB Endowment 18, 12 (2025), 5439–5443. doi:10.14778/3750601.3750691
- [27] Hao Liu, Xiaotian You, and Raymond Chi-Wing Wong. 2024. First Tree-like Quantum Data Structure: Quantum B+ Tree. CoRR abs/2405.20416 (2024).
- [28] Stefan Manegold, Peter Boncz, and Martin L Kersten. 2002. Generic database cost models for hierarchical memory systems. In VLDB’02: Proceedings of the 28th International Conference on Very Large Databases. Elsevier, 191–202.
- [29] D. Michael Miller, Robert Wille, and Zahra Sasanian. 2011. Elementary Quantum Gate Realizations for Multiple-Control Toffoli Gates. In ISMVL. IEEE Computer Society, 288–293.
- [30] Mikko Möttönen, Juha J. Vartiainen, Ville Bergholm, and Martti M. Salomaa.

2005. Transformation of quantum states using uniformly controlled rotations. Quantum Inf. Comput. 5, 6 (2005), 467–473.

- [31] Nitin Nayak, Jan Rehfeld, Tobias Winker, Benjamin Warnke, Umut Çalıkyılmaz, and Sven Groppe. 2023. Constructing Optimal Bushy Join Trees by Solving QUBO Problems on Quantum Hardware and Simulators. In Proceedings of the International Workshop on Big Data in Emergent Distributed Environments (BiDEDE). 1–7. doi:10.1145/3579142.3594298
- [32] Michael A. Nielsen and Isaac L. Chuang. 2016. Quantum Computation and Quantum Information (10th Anniversary edition). Cambridge University Press.
- [33] Jessica Pointing, Oded Padon, Zhihao Jia, Henry Ma, Auguste Hirth, Jens Palsberg, and Alex Aiken. 2021. Quanto: Optimizing Quantum Circuits with Automatic Generation of Circuit Identities. CoRR abs/2111.11387 (2021).
- [34] Carla Rieger, Michele Grossi, Gian Giacomo Guerreschi, Sofia Vallecorsa, and Martin Werner. 2024. Operational Framework for a Quantum Database. arXiv preprint arXiv:2405.14947 (2024). https://arxiv.org/abs/2405.14947 Defines operations and implementation of quantum data storage and manipulation in a quantum database framework.
- [35] Manuel Schönberger, Stefanie Scherzinger, and Wolfgang Mauerer. 2023. Ready to Leap (by Co-Design)? Join Order Optimisation on Quantum Hardware. Proceedings of the ACM on Management of Data 1, 1 (2023), 1–27. doi:10.1145/3581830
- [36] Prasanth Shyamsundar. 2023. Non-Boolean quantum amplitude amplification and quantum mean estimation. Quantum Inf. Process. 22, 12 (2023), 423.
- [37] Yannick Stade, Lukas Burgholzer, and Robert Wille. 2025. Towards Supporting QIR: Steps for Adopting the Quantum Intermediate Representation. In SC Workshops. ACM, 1907–1915.
- [38] Immanuel Trummer and Christoph Koch. 2016. Multiple Query Optimization on the D-Wave 2X Adiabatic Quantum Computer. Proceedings of the VLDB Endowment 9, 9 (2016), 684–695. doi:10.14778/2947618.2947622
- [39] Nimish Ukey, Guangjian Zhang, Zhengyi Yang, Binghao Li, Wei Li, and Wenjie Zhang. 2023. Efficient continuous kNN join over dynamic high-dimensional data. World Wide Web 26, 6 (2023), 3759–3794.
- [40] Martin Vogrin, Rok Vogrin, Sven Groppe, and Jinghua Groppe. 2024. Supervised Learning on Relational Databases with Quantum Graph Neural Networks. In Proceedings of the Second International Workshop on Quantum Data Science and Management (QDSM’24). https://www.vldb.org/workshops/2024/proceedings/ QDSM/QDSM.5.pdf
- [41] Nathan Wiebe, Ashish Kapoor, and Krysta M. Svore. 2015. Quantum algorithms for nearest-neighbor methods for supervised and unsupervised learning. Quantum Inf. Comput. 15, 3&4 (2015), 316–356.
- [42] Tobias Winker, Sven Groppe, Valter Uotila, Zhengtong Yan, Jiaheng Lu, Maja Franz, and Wolfgang Mauerer. 2023. Quantum Machine Learning: Foundation, New Techniques, and Opportunities for Database Research. In Companion of the 2023 International Conference on Management of Data (SIGMOD/PODS). Association for Computing Machinery, 45–52. https://doi.org/10.1145/3555041.3589404
- [43] Sai Wu, Meng Shi, Dongxiang Zhang, Junbo Zhao, Gongsheng Yuan, and Gang Chen. 2024. When Quantum Computing Meets Database: A Hybrid Sampling Framework for Approximate Query Processing. IEEE Trans. Knowl. Data Eng. 36, 12 (2024), 9532–9546.
- [44] Jiadong Xie, Jeffrey Xu Yu, and Yingfan Liu. 2025. Fast Approximate Similarity Join in Vector Databases. ACM SIGMOD 3, 3, Article 158 (June 2025), 26 pages. doi:10.1145/3725403
- [45] Gongsheng Yuan, Yuxing Chen, Jiaheng Lu, Sai Wu, Zhiwei Ye, Ling Qian, and Gang Chen. 2024. Quantum Computing for Databases: Overview and Challenges. CoRR abs/2405.12511 (2024).
- [46] Xinyu Zeng, Yulong Hui, Jiahong Shen, Andrew Pavlo, Wes McKinney, and Huanchen Zhang. 2023. An Empirical Evaluation of Columnar Storage Formats. Proc. VLDB Endow. 17, 2 (2023), 148–161.

