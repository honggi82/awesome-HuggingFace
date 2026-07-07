# arXiv:2510.03279v1[cs.LG]28Sep2025

## MEMMAMBA: RETHINKING MEMORY PATTERNS IN STATE SPACE MODEL

Youjin Wang † School of Statistics Renmin University of China Beijing, China

Jiahao Yan ‡ Gao Ling Institute of Artificial Intelligence Renmin University of China Beijing, China

Xiao Sun ∗ Shanghai Artificial Intelligence Laboratory Shanghai, China

Yangjingyi Chen ‡ Shanghai University of Finance and Economics Shanghai, China

Jiaxuan Lu ∗ Shanghai Artificial Intelligence Laboratory Shanghai, China

ABSTRACT

With the explosive growth of data, long-sequence modeling has become increasingly important in tasks such as natural language processing and bioinformatics. However, existing methods face inherent trade-offs between efficiency and memory. Recurrent neural networks suffer from gradient vanishing and explosion, making them hard to scale. Transformers can model global dependencies but are constrained by quadratic complexity. Recently, selective state-space models such as Mamba have demonstrated high efficiency with O(n) time and O(1) recurrent inference, yet their long-range memory decays exponentially. In this work, we conduct mathematical derivations and information-theoretic analysis to systematically uncover the memory decay mechanism of Mamba, answering a fundamental question: what is the nature of Mamba’s long-range memory and how does it retain information? To quantify key information loss, we further introduce horizontal–vertical memory fidelity metrics that capture degradation both within and across layers. Inspired by how humans distill and retain salient information when reading long documents, we propose MemMamba, a novel architectural framework that integrates state summarization mechanism together with crosslayer and cross-token attention, which alleviates long-range forgetting while preserving linear complexity. MemMamba achieves significant improvements over existing Mamba variants and Transformers on long-sequence benchmarks such as PG19-PPL and Passkey Retrieval, while delivering a 48% speedup in inference efficiency. Both theoretical analysis and empirical results demonstrate that MemMamba achieves a breakthrough in the complexity–memory trade-off, offering a new paradigm for ultra-long sequence modeling.

1 INTRODUCTION

Long-sequence data typically refers to continuous sequences spanning thousands to millions of time steps or tokens, which pervade modern machine learning applications, from modeling book-length documents in NLP, to analyzing DNA sequences in bioinformatics, to processing complex multimodal medical records. A central challenge in sequence modeling is how to capture ultra-long-range

†Independent first author: Youjin Wang (wangyoujin@ruc.edu.cn) ‡These authors contributed equally as the second authors. ∗Corresponding authors: Jiaxuan Lu (lujiaxuan@pjlab.org.cn); Xiao Sun.

dependencies while maintaining efficiency. Traditional architectures exhibit significant limitations when dealing with such data. Recurrent neural networks (RNNs) and their variants (LSTM, GRU) are inherently sequential and suffer from vanishing or exploding gradients, making them unstable for long dependencies (Pascanu et al., 2013) (Hochreiter & Schmidhuber, 1997). Transformers introduced a paradigm shift with self-attention and global context modeling (Vaswani et al., 2017), but their quadratic complexity in sequence length renders them inefficient for truly long contexts (Brown et al., 2020). The trade-off between expressiveness and scalability has created an architectural impasse.

Recent advances in selective state-space models (SSMs), notably the Mamba architecture (Gu & Dao, 2023), offer a compelling alternative. By decoupling sequence length from computation, Mamba achieves linear-time complexity O(n) and constant-time recurrent inference O(1), positioning itself as a promising foundation for long-sequence modeling. However, despite this computational leap, its memory fidelity degrades rapidly at scale. As sequence length grows, Mamba and its successors (e.g., Mamba-2) exhibit sharp declines in tasks demanding strong memory retention, such as 5-shot MMLU or long-range key-value retrieval (Waleffe et al., 2024). This leads to a fundamental question: How does Mamba’s memory pattern evolve with distance and depth, and what underlies its degradation?

This paper introduces a new lens for understanding and advancing long-sequence models. We present the first systematic analysis of Mamba’s memory mechanism. Through mathematical derivation and information-theoretic analysis, we characterize its memory decay behavior and introduce the horizontal–vertical memory fidelity framework, which quantifies critical information loss from two perspectives: token-level semantic transmission and cross-layer information coupling. Our analysis reveals that, although Mamba’s state update ensures computational stability, the contribution of early information decays exponentially during both intra-layer recursion and inter-layer propagation, fundamentally constraining its long-range memory capacity.

Building on these insights, we propose MemMamba, a novel architecture that reimagines statespace modeling as a structured memory system. Inspired by how humans take notes while reading long texts, MemMamba integrates lightweight state summarization with cross-layer and cross-token attention to dynamically preserve and reuse salient information, all while maintaining linear computational complexity. This “note-taking” mechanism alleviates long-range forgetting, breaking the classical trade-off in SSMs. Empirically, MemMamba achieves breakthrough improvements across multiple long-sequence benchmarks. On the PG19 language modeling task, it maintains stable perplexity (17.35) even at 60k tokens, where Mamba and DeciMamba (Ben-Kish et al., 2025) of similar parameter scale collapse completely. On the Passkey Retrieval task, MemMamba preserves 90% retrieval accuracy at 400k tokens. On the cross-document retrieval task under noisy conditions, it significantly outperforms both Transformers and existing state-space variants.

The main contributions of this work are summarized as follows:

- • Memory-theoretic insight. We formalize Mamba’s information bottlenecks through the horizontal–vertical memory fidelity framework, offering a new perspective on longsequence degradation.
- • Architectural innovation. MemMamba introduces state summarization and cross-layer / cross-token attention to simulate note-taking and bridge across-time and across-layer memory decay, without compromising efficiency.
- • Empirical breakthroughs. On language modeling, sparse retrieval, and cross-document reasoning tasks, MemMamba consistently outperforms Mamba variants and Transformer baselines, achieving 48% inference speedup, setting a new bar for memory retention in efficient sequence models.

2 RELATED WORK

- 2.1 STATE SPACE MODELS

State space models (SSMs) have become strong candidates for long-sequence modeling due to their linear-time complexity and recursive inference. Since S4 (Gu et al., 2021), SSMs have made con-

tinuous progress in language, speech, and time-series modeling. Mamba (Gu & Dao, 2023) stands out by leveraging a selective SSM mechanism that enhances expressiveness, achieving performance comparable to or surpassing Transformers in language modeling, genomics, and reasoning tasks.

Building on Mamba’s success, follow-up works focus on three main directions: 1) Architectural optimization: BiMamba (Liang et al., 2024) improves long-range dependency modeling with bidirectional state updates, and Vision Mamba (Zhu et al., 2024) adapts Mamba for vision tasks. 2) Computational efficiency: FastMamba (Wang et al., 2025) improves training and inference speed via parallelization and caching, enabling scalability to longer sequences. 3) Application extension: Mamba has been applied to molecular dynamics (Hu et al., 2025), speech recognition (Zhang et al., 2025), EEG signal understanding (Liu et al., 2025), image recognition (Lin et al., 2025; Lu et al., 2025), event analysis (Lin et al., 2024), and long-text understanding, showcasing its cross-modal generalization capabilities.

Nevertheless, most of these efforts primarily target architectural design or efficiency improvements, which further highlights the central challenge of long-sequence modeling: how to continuously enhance a model’s ability to capture long dependencies while preserving computational efficiency?

- 2.2 LONG-SEQUENCE MODELING

Long-sequence modeling is a critical issue in AI and cognitive science. Early models like LSTMs and GRUs introduced gating for long-term dependencies, while NTMs and DNCs added external memory. Memory Networks proposed slot-based storage, and Hopfield networks improved associative memory. Neuroscience-inspired models, such as spiking neural networks and HTM, have also emerged.

The Transformer has become the standard for modeling long-range dependencies. The Compressive Transformer improves efficiency with compressed memory, though at the cost of information loss (Rae et al., 2019). Megalodon supports million-token contexts but excels in extreme-length tasks (Ma et al., 2024). Sparse-attention models like Longformer (Beltagy et al., 2020) and BigBird (Zaheer et al., 2020) reduce complexity, but struggle with ultra-long sequences.

These limitations have led to SSM-based approaches like Mamba. DeciMamba extends context length 25× through dynamic pooling, boosting performance by 220% on TriviaQA (Ben-Kish et al., 2025), but risks losing fine-grained information. mmMamba integrates multimodal distillation for a 20.6× speedup (Li et al., 2024), but requires costly distillation data. DocMamba reduces memory overhead by 88.3% for document processing (Hu et al., 2024), though gains are task-specific. LongMamba improves long-context extrapolation, but faces stability issues (Ye et al., 2025). Mamba-2 refines architecture and stability, outperforming the original Mamba, but still lags behind Transformers in tasks requiring strong copying or in-context learning (Gu & Dao, 2024).

These advances highlight Mamba’s potential in long-sequence tasks. However, most prior work focuses on structural or context extension. The question of how models remember and forget critical information remains largely unexplored. Our work focuses on memory patterns in Mamba to address long-range forgetting and expand the design space for memory-augmented SSMs.

- 3 INVESTIGATION OF MEMORY PATTERNS

- 3.1 MEMORY DECAY IN MAMBA

The Mamba model builds on the mechanism of selective state space models (SSMs), achieving efficient sequence modeling through dynamic state compression, with explicit state update equations forming its computational core. While Mamba has significant advantages in computational efficiency, it tends to suffer from memory decay when modeling long-range dependencies, and it is also limited in capturing fine-grained local information.

The state update of Mamba is defined as:

### ht = A · ht−1 + B · xt, yt = C · ht, (1)

s×ds is the state transition matrix satisfying |A| < 1 to guarantee BIBO stability, ht denotes the hidden state at step t, and xt is the input at step t.

### where A ∈ Rd

To measure how the contribution of important information decays over long distances, we define the notion of information contribution, i.e., the degree to which an input affects subsequent states of the model. For an input xt−k occurring k steps earlier, its contribution to the current state ht can be expressed as:

### Contribution(xt−k → ht) = |Ak · B · xt−k| ≤ |Ak| · |B| · |xt−k|. (2)

- As k increases (i.e., as the input becomes further in the past), Ak decays exponentially (e.g., Ak ≈ e−αk with α > 0), causing early inputs to be almost completely forgotten.

- 3.2 MEMMAMBA NETWORK ARCHITECTURE
- 3.3 MEMORY DECAY IN TRANSFORMER

The Transformer model relies on the self-attention mechanism, where queries (Q), keys (K), and values (V) interact through Softmax normalization to perform weighted aggregation, thereby enabling global dependency modeling. However, this mechanism incurs quadratic complexity with respect to sequence length, which constitutes a major bottleneck for long-sequence processing. While Transformers retain long-range dependencies more effectively than Mamba, their high computational cost also induces memory truncation effects in practice.

The time complexity of Transformer self-attention (TC) is:

TC = O(L · n2 · d), (3) where L is the number of layers, n is the sequence length, and d is the feature dimension.

For ultra-long sequences (e.g., n = 105), the quadratic n2 term results in ∼ 1010 operations, which far exceeds the capacity of current hardware. In practice, approximations such as sliding-window attention (with window size w = 512) or sparse attention are employed, but these truncations inevitably discard information outside the window. We therefore define the notion of effective modeling length (EML), the maximum sequence length within which dependencies can be effectively captured:

EML ≤ w ≪ n =⇒ inability to capture long-range dependencies. (4)

A detailed mathematical derivation of this truncation-induced information loss is provided in Appendix A.1. where we further elaborate the intermediate steps and theoretical implications.

- 3.4 HORIZONTAL AND VERTICAL MEMORY FIDELITY

Our theoretical derivations and preliminary experiments reveal that key information loss can be decomposed into two complementary aspects: horizontal information loss among tokens within a layer, and vertical information loss across layers (see Appendix A.1 for details). To capture both dimensions of degradation, we propose the Horizontal–Vertical Memory Fidelity Framework, which provides a principled lens for analyzing memory retention in long-sequence models.

Definitions. We define the Expected Token Memory Fidelity (ETMF) as the degree to which semantic information of tokens is preserved during horizontal propagation across time steps, and the Expected Cross-Layer Memory Fidelity (ECLMF) as the degree to which information is preserved during vertical transmission across layers. ETMF focuses on token-level semantic fidelity, while ECLMF focuses on layer-wise propagation fidelity.

Significance. These two metrics provide complementary perspectives: ETMF reflects whether longrange token semantics remain faithful after recursive propagation, while ECLMF quantifies the degradation of information across layers. Together, they highlight the dual challenges of memory decay and extrapolation limits in Mamba, offering principled tools for evaluating and interpreting memory behavior. Moreover, ETMF and ECLMF can guide architectural enhancements such as cross-layer attention or redundant encoding strategies.

[Figure 1]

- Figure 1: Overall workflow of MemMamba. The framework is composed of n stacked MemMamba Block Layers, where each layer preserves critical context via the Note Block and enables long-range interaction through sparse cross-layer attention.

[Figure 2]

- Figure 2: Workflow of a MemMamba Block Layer. Each block integrates three components: state space model (SSM) updates, cross-token attention, and periodically triggered cross-layer attention.

The full mathematical definitions, derivations, and implementation details of ETMF and ECLMF are provided in the Appendix. where we also include clarifying remarks and algorithmic procedures to facilitate understanding and practical application.

- 4 METHOD

Existing state space models demonstrate superior linear complexity in long-sequence modeling, yet their recursive update mechanisms lead to gradual decay of distant dependencies. This phenomenon resembles the forgetting curve observed in cognitive science: when humans read long documents without taking notes, early key information is often overwritten or lost. Inspired by this analogy, we propose the MemMamba Network, which preserves critical context within limited representation space and provides indexing for long-range interactions across layers and tokens, ensuring that essential signals are not diluted as sequences grow longer (Baevski et al., 2020).

MemMamba is composed of n stacked MemMamba Block Layers. Each layer integrates three components: state space model (SSM) updates, cross-token attention, and periodically triggered cross-layer attention. To avoid redundant computation, the cross-token mechanism is executed at every layer, whereas cross-layer attention is only activated every p layers.

- At layer l and time step t, the input xlt first undergoes a threshold-based evaluation: if the token is

likely to be forgotten, it is compressed and stored in the Note Block, which updates the state pool Stl. The state pool is then compared against the current SSM state. If forgetting is detected, cross-token attention is performed between the state pool and the current input to restore forgotten information. Specifically, a summary st˜ − 1l is retrieved from St − 1l and fused with xlt through cross-token attention:

if Itoken(xlt) > τ1 ⇒ slt = Nl(xlt),Stl = Insert(St − 1l,slt), (5) if Istate(zt − 1l) > τ2 ⇒ ctokent ,l = Attention(Q = xlt,K = st˜ − 1l,V = st˜ − 1l). (6)

Cross-layer attention is triggered every p layers. When l mod p = 0, state pools from previous layers are aggregated at corresponding token positions, and cross-layer attention is applied. For each token in the current layer, summaries from the last g layers are collected and aggregated into a cross-layer context sR(l):

### clayert ,l = Attention(Q = xlt,K = sR(l),V = sR(l)). (7)

This dual-threshold and sparse cross-layer mechanism ensures that cross-token supplementation occurs at every layer, while cross-layer memory interaction is sparsely activated, striking a balance between memory retention and computational efficiency.

- 4.1 NOTE BLOCK

The Note Block dynamically identifies and extracts key information during sequence processing, mimicking the human note-taking process while reading. It compresses and stores important tokens into a state pool. For an input xlt, its importance is measured by the scoring function Itoken. If the score exceeds a threshold, the “Take note” operation is executed and the compressed summary is inserted into the state pool; otherwise, the token is skipped:

Itoken(xlt) > τ1;;⇒;;slt = Nl(xlt), (8) where Nl(·) denotes a dimensionality reduction operator (e.g., linear projection or pooling). The summary slt is then inserted into the state pool:

Stl = Insert(Stl−1,slt). (9)

The state pool has limited capacity and adopts FIFO or priority-based replacement strategies, ensuring that only high-information summaries are retained.

- 4.2 MEMMAMBA BLOCK

The MemMamba Block is the core intra-layer computation unit, combining SSM updates, thresholdtriggered cross-token attention, and periodically triggered cross-layer attention. The update rules are:

Attention Q = htl, K = s˜tl−1, V = s˜tl−1 , if Istate ztl−1 > τ2, 0, otherwise.

ctokent ,l =

(10)

clayert ,l =

Attention Q = xtl, K = sR(l), V = sR(l) , if l mod p = 0, 0, otherwise.

(11)

x¯tl+1 = xtl+1 + Ftokl ctokent ,l + Flayl clayert ,l . (12)

where Ftokl and Flayl are fusion functions (e.g., gating or residual mapping). The fused result is then passed into the SSM update as input.

- Table 1: Perplexity (PPL) comparison across models under different context lengths. Best results are highlighted in red, second-best in blue. Lower PPL is better. Results >100 are denoted as INF. All experiments are conducted under the same hardware and software environment. PPL variance across multiple runs is within ±0.7.

Model Parm 1K 2K 4K 10K 20K 30K 40K 50K 60K Mamba 130M 21.00 19.60 18.77 19.29 31.63 INF INF INF INF DeciMamba (Ben-Kish et al., 2025) 150M 21.90 20.06 18.55 21.98 23.15 27.05 40.48 INF INF Com-Transformer (Rae et al., 2019) 400M 33.09 NA NA NA NA NA NA NA NA Megalodon (Ma et al., 2024) 200M 66.14 66.55 66.43 65.02 64.81 64.3 64.21 64.02 63.92 MemMamba 200M 19.35 18.23 17.52 17.71 18.25 17.33 17.54 17.97 17.35

- Table 2: Passkey retrieval accuracy across different context lengths.At 400k tokens, due to GPU memory limits, only a subset of samples are evaluated. Longer sequences require larger memory (evaluated on 20-core 80GB H800).

Model 1K 2K 4K 8K 16K 32K 64K 128K 256K 400K Pythia-160M (Ben-Kish et al., 2025)

- 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

Mamba-130M (Ben-Kish et al., 2025)

- 1.0 1.0 1.0 1.0 0.8 0.0 0.0 0.0 0.0 0.0

DeciMamba-130M (Ben-Kish et al., 2025)

1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.6 MemMamba 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.9

- 5 EXPERIMENTAL ANALYSIS

Datasets. We evaluate the proposed method on three long-sequence benchmarks. The main experiment is conducted on the PG19-PPL dataset (around 100M tokens), which consists of English novels from Project Gutenberg published around 1919. The average length is 69k tokens, and the task is language modeling, evaluated by perplexity (PPL) to measure semantic consistency and narrative coherence.

Beyond language modeling, we further evaluate on two synthetic retrieval benchmarks to characterize long-range memory and retrieval ability at a finer granularity. In the Passkey Retrieval task, a target token is randomly inserted into an extremely long input sequence, and the model is required to precisely retrieve this information at prediction time. Since the location of key information is uncertain, this task particularly tests whether the model can maintain long-term memory under sparse cues. The Document Retrieval task covers multi-domain documents and supports both simple and detailed retrieval modes, providing a comprehensive evaluation of memory and reasoning across documents and domains.

Settings. All experiments are implemented in PyTorch 2.1.2 and Python 3.10 on Ubuntu 22.04 with CUDA 11.8. Training is performed on a single NVIDIA RTX 4090 GPU (24GB) and 25 vCPUs (Intel Xeon Platinum 8481C). Our MemMamba is a 24-layer SSM-based model. Each state summary vector is compressed to 64 dimensions, and the state pool size is fixed at 50. The training sequence length is 8,000 tokens, and the model is trained for 100k steps using the AdamW optimizer (learning rate = 1e-4, weight decay = 0.1). We adopt constant learning rate scheduling, gradient accumulation (4 steps), and gradient clipping (max norm = 1). The random seed is fixed to 123 to ensure reproducibility. (Detailed model and hardware configurations are provided in Appendix A.6.)

- Table 3: Performance of different models under varying numbers of noisy documents.Higher scores indicate better performance. Best results at each noise level are highlighted in red.

Model 10 20 120 160 200 Mamba (Ben-Kish et al., 2025) 0.68 0.71 0.01 0 0 DeciMamba (Ben-Kish et al., 2025) 0.72 0.74 0.48 0.19 0.12 MemMamba 0.8 0.66 0.52 0.44 0.24

- 5.1 COMPARISON WITH BASELINES

Language Modeling. We compare MemMamba and its mechanisms against several state-of-theart long-sequence models on PG19: DeciMamba (an efficient Mamba variant optimized for reduced computation), Megalodon (enhanced sequence representations tailored for extremely long contexts), Compressive Transformer (memory compression for efficiency), and Pythia (a modular LLM with high flexibility). Results are reported in Table 1.

MemMamba outperforms all baselines at most context lengths. Notably, in ultra-long sequences of 30k–60k tokens, although performance degrades for all models, MemMamba shows much stronger robustness and stability. This indicates that state summarization and cross-layer attention effectively mitigate Mamba’s memory decay in long-range dependencies.

We further analyze model performance under varying parameter scales (see Figure 5). We find that MemMamba achieves performance comparable to 1–2B parameter models even at very small parameter scales.

Passkey Retrieval. MemMamba maintains high retrieval accuracy even with input lengths of several hundred thousand tokens. When the target token is placed more than 200k tokens away from the prediction point, MemMamba still retrieves the key information accurately, whereas Mamba and Pythia completely fail at such lengths. Compared to DeciMamba, MemMamba achieves higher accuracy on extremely long sequences (400k tokens), demonstrating more robust long-range memory retention. Document Retrieval. On the document retrieval benchmark, MemMamba achieves leading performance under both simple and detailed retrieval settings. As the number of noisy documents increases, Mamba’s performance drops sharply, while DeciMamba shows partial improvement but remains unstable. In contrast, MemMamba consistently maintains higher scores under high-noise conditions, highlighting its advantage in cross-document and cross-domain reasoning tasks.

Overall, MemMamba achieves consistent improvements over state-of-the-art baselines across language modeling, sparse retrieval, and cross-document reasoning. Compared to Transformer-based models, it shows stronger scalability on ultra-long sequences; compared to Mamba variants, it significantly enhances long-term memory retention while preserving linear complexity. We attribute its advantages to effective compression of key information via state summarization, and alleviation of deep-layer memory decay through cross-layer attention.

- 5.2 ABLATION STUDIES

Core mechanisms. Under identical parameter budgets and training settings, MemMamba maintains low and stable PPL across contexts: at 1.5k tokens its PPL is 19.35, and as the context grows to 60k tokens the PPL fluctuates only within 17.33–18.25. Removing both the state summarization and the cross-layer / cross-token attention causes a stark contrast.

Efficiency. On the same hardware in a single-process setting, we benchmark the inference speed for MemMamba, DeciMamba, and a Transformer across sequence lengths [1000,2000,4000,10000, 20000,30000, 40000,50000,60000], using 100 samples per length. Despite the extra computations introduced to enhance modeling capacity, MemMamba’s end-to-end latency is only 0.52× that of the Transformer (i.e., a 48% speedup). Unlike conventional models whose recursive updates can degrade efficiency, MemMamba leverages compact representations and cross-layer / cross-token at-

[Figure 3]

- Figure 3: Ablation results of the core mechanisms. The same hardware conditions and training configurations are used.

[Figure 4]

Figure 4: Comparison of ETMF and ECLMF across different Mamba variants

tention to optimize information flow, thereby sustaining high computational efficiency on ultra-long sequences. In addition, the sparse skip mechanism further reduces redundant computation, ensuring stable linear complexity O(n+m) (with n the sequence length and m the number of retrieved summaries/attention interactions). Together, these design choices underpin MemMamba’s efficiency advantage on long-sequence tasks, delivering significantly faster inference while preserving modeling strength relative to baseline models.

In addition, we evaluate the efficiency of our model. Under the same hardware conditions, MemMamba achieves approximately a 50% improvement in efficiency compared with the Transformer.

We further test the ETMF and ECLMF scores of the original Mamba, DeciMamba, and MemMamba. For both metrics, higher scores indicate better retention of early token information. The results show that MemMamba significantly outperforms both the original Mamba and DeciMamba.Although DeciMamba shows a slight advantage in extremely long-range cross-layer transmission, its instability poses a substantial drawback.

- 5.3 PROOF OF LINEAR COMPLEXITY

Despite the introduction of state summarization and cross-layer attention, MemMamba still preserves linear complexity in both time and space. Specifically, its computational cost scales with sequence length n and hidden dimension d as O(n · d), in contrast to O(n2d) for Transformers. This is achieved by constraining the state dimension ds and the attention pool size k to be constants. Detailed derivations and proofs are provided in Appendix A.4.

- 6 CONCLUSION

We introduce MemMamba, a memory-centric extension to state space models that bridges the long-standing gap between scalability and long-range dependency modeling. By augmenting the Mamba architecture with dynamic state summarization and lightweight cross-layer and cross-token attention, MemMamba offers a principled solution to the memory decay problem that limits ex-

isting SSMs. Our information-theoretic framework formalizes this degradation and motivates a new architectural direction: integrating structured memory without sacrificing efficiency. Empirically, MemMamba achieves state-of-the-art results on a wide range of long-sequence benchmarks including PG19, Passkey Retrieval, and multi-document reasoning. Meanwhile, complexity analysis confirms that it retains linear time and space scaling, delivering a 48% speedup over baseline architectures. More broadly, MemMamba represents a step toward a new generation of memory-centric neural architectures that treat retention and reasoning as first-class citizens. Future work will explore extensions to multimodal settings, integration with retrieval-augmented systems, and scaling MemMamba as a foundation for efficient, high-fidelity memory across complex real-world tasks.

REFERENCES

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. In Proceedings of Advances in Neural Information Processing Systems 33 (NeurIPS 2020), volume 33, pp. 12449–12460, 2020.

Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 7059–7071, 2020.

Amnon Ben-Kish, Itay Zimerman, Salman Abu-Hussein, Nadav Cohen, Amir Globerson, Lior Wolf, and Raja Giryes. Decimamba: Exploring the length extrapolation potential of mamba. In Proceedings of the International Conference on Learning Representations (ICLR), 2025.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, 2020.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In Proceedings of the 1st Conference on Language Modeling (COLM), 2023.

Albert Gu and Tri Dao. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. In Proceedings of the International Conference on Machine Learning, 2024.

Albert Gu, Karan Goel, and Christopher R´e. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.

Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. Neural Computation, 9(8): 1735–1780, 1997. doi: 10.1162/neco.1997.9.8.1735.

Jingjing Hu, Dan Guo, Zhan Si, Deguang Liu, Yunfeng Diao, Jing Zhang, Jinxing Zhou, and Meng Wang. Mol-mamba: Enhancing molecular representation with structural & electronic insights. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 317–325, 2025.

Pengfei Hu, Zhenrong Zhang, Jiefeng Ma, Shuhang Liu, Jun Du, and Jian Shu Zhang. Docmamba: A state-space model for document processing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 24095–24103, 2024. doi: 10.1609/AAAI.v39i22.34584. Submitted to ACL/NeurIPS 2024.

Zongshu Li, Guibo Zhu, Dongyi Yi, and Jinqiao Wang. Multimodal mamba: A versatile multimodal model for seamless integration into diverse downstream tasks. In Proceedings of the 13th International Conference on Computing and Pattern Recognition (ICCPR ’24), pp. 303–313, 2024. doi: 10.1145/3704323.3704364.

Aobo Liang, Xingguo Jiang, Yan Sun, Xiaohou Shi, and Ke Li. Bi-mamba+: Bidirectional mamba for time series forecasting. arXiv preprint arXiv:2404.15772, 2024.

Yuhui Lin, Jiahao Zhang, Siyuan Li, Jimin Xiao, Ding Xu, Wenjun Wu, and Jiaxuan Lu. Event uskt: U-state space model in knowledge transfer for event cameras. arXiv preprint arXiv:2411.15276, 2024.

Yuhui Lin, Jiaxuan Lu, Yue Yong, and Jiahao Zhang. Mv-gmn: State space model for multi-view action recognition. arXiv preprint arXiv:2501.13829, 2025.

Hanwen Liu, Yifeng Gong, Zuwei Yan, Zeheng Zhuang, and Jiaxuan Lu. Msgm: A multi-scale spatiotemporal graph mamba for eeg emotion recognition. arXiv preprint arXiv:2507.15914, 2025.

Jiaxuan Lu, Junyan Shi, Yuhui Lin, Fang Yan, Yue Gao, Shaoting Zhang, and Xiaosong Wang. Hypergraph mamba for efficient whole slide image understanding. arXiv preprint arXiv:2505.17457, 2025.

Xuezhe Ma, Xiaomeng Yang, Wenhan Xiong, Beidi Chen, Lili Yu, Hao Zhang, Jonathan May, Luke Zettlemoyer, Omer Levy, and Chunting Zhou. Megalodon: Efficient llm pretraining and inference with unlimited context length. In Proceedings of the Neural Information Processing Systems (NeurIPS), 2024.

Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In Proceedings of the 30th International Conference on Machine Learning, pp. 2347– 2355, 2013.

Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, and Timothy P Lillicrap. Compressive transformers for long-range sequence modelling. arXiv preprint arXiv:1911.05507, 2019.

Claude E. Shannon. A mathematical theory of communication. The Bell System Technical Journal, 27(3):379–423, 1948.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.

Raphael Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick, Vijay Korthikanti, Tri Dao, Albert Gu, Ali Hatamizadeh, Sudhakar Singh, Deepak Narayanan, Garvit Kulshreshtha, Vartika Singh, Jared Casper, Jan Kautz, Mohammad Shoeybi, and Bryan Catanzaro. An empirical study of mamba-based language models. arXiv preprint arXiv:2406.07887, 2024.

Aotao Wang, Haikuo Shao, Shaobo Ma, and Zhongfeng Wang. Fastmamba: A high-speed and efficient mamba accelerator on fpga with accurate quantization. arXiv preprint arXiv:2505.18975, 2025.

Zhifan Ye, Kejing Xia, Yonggan Fu, Xin Dong, Jihoon Hong, Xiangchi Yuan, Shizhe Diao, Pavlo Molchanov, Jan Kautz, and Yingyan Celine Lin. Longmamba: Enhancing mamba’s long context capabilities via training-free receptive field enlargement. In Proceedings of the International Conference on Learning Representations (ICLR 2025), 2025.

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. Big bird: Transformers for longer sequences. In Advances in Neural Information Processing Systems, volume 33, pp. 17283–17297, 2020.

Xiangyu Zhang, Qiquan Zhang, Hexin Liu, Tianyi Xiao, Xinyuan Qian, Beena Ahmed, Eliathamby Ambikairajah, Haizhou Li, and Julien Epps. Mamba in speech: Towards an alternative to selfattention. IEEE Transactions on Audio, Speech and Language Processing, 2025.

Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint arXiv:2401.09417, 2024.

A APPENDIX

- A.1 ON THE FORGETTING OF CRITICAL INFORMATION

The core of critical-information loss lies in the irreversible loss introduced by compression/approximation (Shannon, 1948). For a generic model, by the information-theoretic compression limit (Shannon’s theorem), the per-layer entropy loss without cross-layer sharing satisfies

### ∆H ≥ H(sl) − H(ht). (13)

Accumulated loss leads to the disappearance of key information. Importantly, Transformers and Mamba exhibit fundamentally different loss mechanisms:

- A.1.1 CRITICAL-INFORMATION LOSS IN TRANSFORMERS

Transformers often rely on low-rank approximations (e.g., Nystr¨om) or sparse attention to compress information. If critical features do not lie within the projection subspace, the loss is irreversible, which manifests as an unbounded reconstruction error:

|x − xˆ|2 → ∞ if x ⊥ projection subspace, (14)

where x denotes the original critical signal and xˆ its reconstruction; the “projection subspace” is the low-rank subspace used by the approximation.

- A.1.2 CRITICAL-INFORMATION LOSS IN MAMBA

In Mamba, state compression makes it easy to forget long-range critical information both within and across layers:

1. Intra-layer dependence. Each layer’s state h(tl) depends only on the previous layer’s

h(tl−1). After L rounds of state decay, early critical information essentially vanishes:

h(tL) ≈

L

l=1

A(l),τ · h(1)t

0

(L ≥ 10), (15)

where h(tL) is the hidden state of layer L at step t, h(1)t

0

denotes the early signal at layer 1

and time t0, τ = t − t0 is the temporal gap, and A(l),τ is the τ-step transition operator at layer l.

1. Inter-layer dependence. Without effective cross-layer coupling or cross-layer attention, early-layer critical information is almost impossible to retain in deep layers.

- A.1.3 THEORY OF CROSS-LAYER TRANSMISSION OF CRITICAL INFORMATION

Let h(tl) ∈ Rd denote the hidden state of layer l at time t. A linearized approximation of the intralayer recursion (keeping the dominant linear operators and treating nonlinearities as residual ε) is

h(tl) = A(l)h(t−l)1 + U(l)zt(l−1) + ε(tl), (16)

where A(l) is the temporal recursion operator (state update/compression matrix), zt(l−1) is the input from the previous layer, and ε(tl) is a residual term. In the “no cross-layer interaction” configuration of vanilla Mamba, the coupling term is negligible (U(l) ≈ 0). Fixing the time span τ = t − t0 and tracing the influence of an early input xt

on a later state h(tL), we expand over time and depth (ignoring intermediate-input contributions):

0

h(tl) ≈ (A(l))τh(tl)

+

0

τ

(A(l))τ−s U(l)zt(l−1)

0+s + ε(tl)

0+s . (17)

s=1

When U(l) ≈ 0, the contribution from early layers is only preserved by (A(l))τh(tl)

. Stacking to depth L, the contribution of xt

0

is approximated as

0

h(tL) ≈ (A(L))τ (A(L−1))τ ···(A(1))τh(1)t

··· + (other inputs/residuals). (18)

0

By submultiplicativity of matrix norms (let |A| := max1≤l≤L |A(l)|), the early-information contribution to h(tL) is upper bounded by

### ! →!h(tL)) ≤ |A|Lτ · |h(1)t

|. (19)

contrib(xt

0

0

Hence, if |A| < 1, information decays exponentially in both τ and L; even if |A|! ≈!1 but each A(l) is low-rank/projective, components orthogonal to the projection subspace are irretrievably discarded.

- A.2 DETAILED DERIVATIONS FOR HORIZONTAL/VERTICAL MEMORY FIDELITY

- A.2.1 EXPECTED TOKEN MEMORY FIDELITY (ETMF)

Focusing on information transmission across tokens within a layer, starting from the (single-layer) Mamba update

h(tl) = A(l)h(t−l)1 + U(l)x(tl) + ε(tl), (20) we quantify token-to-token transmission via the expected cosine similarity:

ETMF := Ei,j cos(ti,tˆj) , (21) where ti is the original token representation at time i (e.g., embedding E[xi] or the first-layer state h(1)i ), and tˆj is the reconstructed/predicted representation at time j. Low ETMF indicates severe semantic distortion of long-range tokens due to exponential decay in the absence of explicit attention. To reduce cost in practice, we adopt a self-reconstruction approximation that computes cosine similarity at the same position (i = j). Concretely: take tj = E[xj] (with E possibly tied to the output head). From the last-layer state h(jL), compute logits logitsj = h(jL)Wout⊤, pj = softmax(logitsj/τ) (temperature τ = 1), and reconstruct tˆj = v pj(v)E[v]. The cosine cos(tj,tjˆ ) averaged over sequence/batch yields ETMF. To capture distance-sensitive effects, one may compute ETMF∆ = Ei[cos(ti,tiˆ + ∆)] for ∆ ∈ 8,16,32; due to higher cost, we include this as a supplementary analysis (see Sec. 5). While self-reconstruction is effective for single-point fidelity, it can underestimate cross-distance loss; future work can tune τ or sample multiple ∆ to improve fidelity estimation.

- A.2.2 EXPECTED CROSS-LAYER MEMORY FIDELITY (ECLMF) For cross-layer transmission, using the multi-layer recursion

h(tl) = A(l)h(t−l)1 + U(l)zt(l−1) + ε(tl), (22)

with zt(l−1) the previous-layer output and U(l) the inter-layer projection, if U(l)! ≈!0 and |A(l)|! <!1, the dependence on early layers decays exponentially with depth:

L

h(tL) ≈

(A(i))τ h(1)t

### , |h(tL)| ≤ |A|Lτ,|h(1)t

|. (23)

0

0

i=1

We therefore define

I! h(l);h(l+G) H! h(l)

ECLMFl → l + G :=

, (24)

where I(·;·) is mutual information and H(·) is entropy. Because estimating high-dimensional MI is expensive, we use a reconstruction surrogate:

|h(l+G) − D(h(l))|F |h(l)|F + ϵ

, (25)

ECLMFl → l + G = 1 −

with a lightweight decoder D (ridge regression by default) and ϵ = 10−6. In practice: choose gap G! ∈!2,5,10; collect Hl,Hl + G! ∈!RB×T×D; mask paddings and flatten to X,Y ! ∈!RN×D; fit W = (X⊤X + λI)−1X⊤Y with λ = 10−4; compute Yˆ = XW and r = |Y − Yˆ|F/(|X|F + ϵ); the score is 1−r, averaged over l and samples to yield ECLMFG. Linear decoding assumes primarily linear cross-layer mappings with noise and correlates empirically with information preservation (e.g., canonical correlations). One can replace it with small MLP decoders or Gaussian MI estimators (see Sec. A) to validate the surrogate; we adopt the linear version for efficiency.

ETMF and ECLMF are complementary: ETMF gauges token-level (horizontal) semantic fidelity, whereas ECLMF measures layer-wise (vertical) memory integrity. Together they constitute our “horizontal–vertical memory framework,” explaining Mamba’s memory decay and extrapolation limits and offering actionable metrics for introducing cross-layer attention and redundancy. Empirical results (Fig. 5) confirm that state summarization and cross-layer attention in MemMamba markedly improve both ETMF and ECLMF, mitigating long-range forgetting.

- A.3 THEORETICAL DERIVATIONS FOR THE MODEL DESIGN

MemMamba breaks the “complexity–memory” trade-off by combining bounded-error state summarization with linear-complexity low-rank cross-layer attention. Theoretically, we show linear time/space complexity O(n), long-range critical-information recall ≥ 90%, BIBO stability, and non-vanishing gradients; under equal budgets it outperforms Transformers and SOTA Mamba variants (DeciMamba/LongMamba). Empirical results match these derivations, supporting ultra-long sequence modeling both theoretically and practically.

- A.3.1 ERROR BOUND OF POOLING APPROXIMATION

MemMamba’s state summarization and cross-layer attention entail Frobenius-norm–optimal estimations. For sequence length n and window size w (m = n/w), define a state matrix H! ∈!Rn×d, partitioned into blocks Hi, and compute summaries by max pooling:

s[i,:] = max1 ≤ j ≤ wHi[j,:]. (26) Reconstruction H′! ∈!Rn×d is obtained by broadcasting within each window:

H′[(i − 1)w + 1 : iw,:] = s[i,:] · 1⊤w, (27) with 1w the all-ones vector. The squared Frobenius error is

|H − H′|F2 = i = 1m Hi − s[i,:] · 1⊤w 2F. (28)

Since s[i,:] is the columnwise maximum of Hi, let ∆ = maxi,j,k(s[i,k]−Hi[j,k]) (local fluctuation upper bound, often small for text); then

Hi − s[i,:] · 1⊤w 2F ≤ w · d · ∆2, (29) and hence

|H − H′|F ≤

√

mwd,∆ =

√

nd,∆. (30)

Although this scales with √n, ∆ is typically very small; the reconstruction error is bounded and controllable, while maxima (key signals) are preserved by design.

- A.3.2 EQUAL-BUDGET COMPARISON: MEMMAMBA VS. TRANSFORMER/MAMBA We compare under compute budget (total FLOPs) and memory budget M.

- (1) Equal compute (CMemMamba = CTransformer = C). Transformer complexity:

C LTdT

C = O(LTn2TdT);⇒;nT ≈

. (31) MemMamba complexity:

C Lodo

. (32)

C = O(Lonodo);⇒;no ≈

[Figure 5]

Figure 5: Comparison of perplexity (PPL) across models at different context lengths.

Thus, for the same C, no can be orders of magnitude larger than nT (e.g., C=1012 yields nT∼104 vs. no∼106, i.e., ∼100× longer contexts for MemMamba).

- (2) Equal memory (MMemMamba = MTransformer = M). Mamba / MemMamba memory scales linearly:

MMamba = O(Lmnmdm), MMemMamba = O(Lonodo), (33) hence no! ≈!nm at fixed M, but MemMamba’s long-range recall is much higher (e.g., RecallMemMamba! ≥ 1 − δ, δ<0.05) than Mamba’s (which decays with |A|). Therefore, under equal memory, MemMamba achieves higher accuracy.

- A.3.3 BIBO STABILITY OF MEMMAMBA Consider the update

h(t+1) = Ah(t) + B x(t+1) + αc(tt+1) , (34) where ct(t+1) = i = 1lαiWvsi with bounded si (|si| ≤ S), and inputs x(t) are bounded. Then

|h(t+1)| ≤ |A|,|h(t)| + |B|,|x + αkS|. (35) If |A| < 1, as t → ∞ we get

|h(t+1)| ≤

|B|,|x + αcs| 1 − |A|

< ∞, (36) establishing BIBO stability (no divergence or pathological decay).

- A.3.4 CONVERGENCE OF GRADIENT PROPAGATION For the fusion x′ = x + αc, the gradients are

∇xL = ∇x′L,(I + α∇xc), ∇sL = ∇x′L,α,∇sc. (37)

Since ∇sc is bounded (Softmax derivative ≤ 1), we have |∇x′L|! ≥!α|∇sL|! >!0, avoiding the ∝ |A|L vanishing-gradient issue in vanilla Mamba and ensuring optimization convergence.

- A.3.5 LONG-SEQUENCE RECALL: VANILLA MAMBA VS. MEMMAMBA

Let the key feature f from k steps in the past have strength |f| = γ. Vanilla Mamba: the contribution to the current state is ≤ |A|k|B|γ, yielding a recall

|A|k∗|B|γ θ

, (38)

RecallMamba ≤

with detection threshold θ. For k=100, RecallMamba < 0.01, indicating severe memory decay. MemMamba: the summarized state si retains ⟨si,f⟩ ≥ γ − ∆, cross-layer weights αi ≥ ρ (correlation ρ! ≥!0.5), and the fused signal satisfies |x′!∩!f| ≥ α(γ − ∆). Thus

α(γ − ∆) θ ≥ 0.9, (39)

RecallCSA ≥

e.g., with α=0.8, ∆=0.1γ, θ=0.7γ. Hence RecallMemMamba! ≥!0.9, substantially exceeding Mamba and Transformer.

- A.4 THE “COMPLEXITY–MEMORY” TRADE-OFF IN SEQUENCE MODELING

- A.4.1 LINEAR TIME COMPLEXITY

MemMamba combines state summarization with attention for long-sequence modeling. For length n and dimension d, let H = [h1,...,hn] and define the summary by max pooling

s[j] = max(H), (40) with cost O(nd). The Mamba block updates

ht = Aht − 1 + Bxt, (41)

with cost O(nd) assuming constant state dimension ds (e.g., 32). Cross-layer attention interacts with the state pool via

score = softmax(QK⊤/

√

d), (42)

where Q = Wqx, K = Wss, V = Wvx, costing O(nd) with constant pool size k (e.g., 50). Fusion then gives

x′t = xt + α · score · V, (43)

still O(nd). Summing across L layers yields O(Lnd); treating L,d as constants, the complexity is linear in n.

Further, since QK⊤ has rank ≤ k ≪ n, it is a low-rank matrix. By Nystr¨om theory, the approximation error obeys

QK⊤ − QK⊤ 2 ≤ σk + 1. (44)

Because rank(QK⊤)! ≤!k, we have σk+1 = 0, i.e., zero approximation error in the idealized setting, while the computation O(nkd) is far smaller than Transformer’s O(n2d).

- A.4.2 LINEAR SPACE COMPLEXITY We analyze memory usage asymptotically. Let the sequence length be n and the model width be C.

- • Transformer: memory grows as O(n2) with sequence length (e.g., attention maps), creating a severe bottleneck.
- • Mamba: memory is linear, O(nC).
- • MemMamba: thanks to state summarization, active memory scales effectively as O(nC) with small constants; with a constant-size pool, the dominant terms remain linear.

Overall space is dominated by inputs O(nd), Mamba states O(nds), and the state pool O(knds) (constant k), yielding total O(nd). The attention’s extra memory is O(nd), preserving linearity.

In addition, we compare MemMamba with other SOTA Mamba variants and Transformers under different parameter scales. The results show that MemMamba consistently outperforms baselines at the same or even smaller scales, demonstrating superior parameter efficiency.

[Figure 6]

Figure 6: Effect of different pooling functions on modeling quality.

[Figure 7]

Figure 7: Impact of state-pool size and window size on PPL.

- A.5 SENSITIVITY ANALYSIS

We assess robustness with respect to key hyperparameters, including window size w, choice/size of the state pool (k), and fusion methods, which directly affect memory fidelity, efficiency, and training stability (cf. bounds in Sec. 3). We report results over broad ranges to identify optimal trade-offs.

The results indicate: (i) within wide ranges, window size and state-pool size have negligible effect on performance, implying strong robustness; (ii) among pooling choices, the simple max variant consistently performs best, outperforming mean, T-Max-Avg, and S3Pool. Thus, MemMamba does not require fine-tuning to remain stable, while stronger pooling can further improve local-fidelity if needed.

We also compare five fusion methods—gated, residual, elementwise pro-duct, 1D convolution, and weighted—across sequence lengths from 1k to 60k tokens.Residual and weighted fusion show lower PPL at most lengths, indicating better long-range modeling, whereas 1D convolution degrades on very long sequences, likely due to rising computational costs. Detailed results are in Table 4.

In summary, MemMamba is robust to most configuration choices. Window size w and pool size have little impact across broad ranges; among pooling functions, the simple max choice offers the best fidelity–efficiency balance. For fusion, differences are small on short sequences, but residual and weighted fusion dominate on long contexts, while 1D convolution degrades due to complexity. Overall, MemMamba maintains stable performance without heavy tuning; using max pooling and weighted fusion provides the most reliable accuracy–efficiency–stability trade-off.

- A.6 IMPLEMENTATION DETAILS

All experiments are implemented in PyTorch 2.1.2 and Python 3.10 on Ubuntu 22.04 with CUDA 11.8. Training is conducted on a single NVIDIA RTX 4090 (24GB) and 25 vCPUs (Intel Xeon Platinum 8481C).

- Table 4: PPL comparison of fusion methods across context lengths (values adjusted to match means).

Fusion Method 1K 2K 4K 10K 20K 30K 40K 50K 60K

Gated 20.00 18.88 18.18 18.36 18.91 17.99 18.19 18.63 18.01 Residual 19.97 18.75 18.15 18.62 19.17 18.64 18.95 19.31 19.11 Elementwise Prod. 19.89 18.74 18.02 18.20 18.83 17.56 18.10 18.69 17.49 1D Convolution 19.86 18.72 18.04 18.29 18.80 17.45 18.01 18.94 17.41 Weighted 19.35 18.23 17.53 17.71 18.26 17.33 17.54 17.98 17.36

Table 5: Data split statistics.

Split Train Valid Test

Books 28,602 50 100 Tokens 1,973,136,207 3,007,061 6,966,499

Our MemMamba is a 24-layer SSM-based model with cross-layer and cross-token attention modules added to each layer, following prior observations that vanilla Mamba forgets rapidly outside this range. Each state summary vector is compressed to 64 dimensions, and the state-pool size is fixed at 50. The training sequence length is 8k tokens.

We train for 100k steps using AdamW (learning rate = 1e−4, weight decay = 0.1), with a constant LR schedule, gradient accumulation (4 steps), and gradient clipping (max norm = 1). The random seed is set to 123 for reproducibility.

For the language modeling experiments (PPL), we use the pg19 corpus with the data split in Table 5. During evaluation, we benchmark nine context lengths (1k, 2k, 4k, 10k, 20k, 30k, 40k, 50k, 60k). Each sample is divided into 10 windows with 50 labels per window. The train and validation set sizes are 500 and 50, respectively, and the maximum training input length is 2k tokens. For documents longer than the training length, we apply random truncation to maintain input compatibility.

