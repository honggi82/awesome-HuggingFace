# arXiv:2505.09738v1[cs.CL]14May2025

## Achieving Tokenizer Flexibility in Language Models through Heuristic Adaptation and Supertoken Learning

Shaurya Sharthak1, Vinayak Pahalwan1, Adithya Kamath2, and Adarsh Shirawalmath3 1(shaurya.sharthak,vinayak)@tinycompany.in

- 2adi_kmt@proton.me
- 3adarsh@tensoic.com

May 16, 2025

Abstract

Pretrained language models (LLMs) are often constrained by their fixed tokenization schemes, leading to inefficiencies and performance limitations, particularly for multilingual or specialized applications. This tokenizer lock-in presents significant challenges; standard methods to overcome this often require prohibitive computational resources. Although tokenizer replacement with heuristic initialization aims to reduce this burden, existing methods often require exhaustive residual fine-tuning and still may not fully preserve semantic nuances or adequately address the underlying compression inefficiencies. Our framework introduces two innovations, first, Tokenadapt a model-agnostic tokenizer transplantation, and second, novel pre-tokenization learning multi-word supertokens to enhance compression and reduce fragmentation. Tokenadapt initializes new unique token’s embeddings via a hybrid heuristic. Our hybrid approach combines two methods: a local heuristic based on subword decomposition using old tokenizer, and a global heuristic utilizing the top-k semantically similar tokens from original vocabulary. This methodology aims to preserve semantics while significantly minimizing retraining requirements. Empirical investigations validate both contributions: the transplantation heuristic successfully initializes unique tokens, markedly outperforming conventional baselines and sophisticated methods including Transtokenizer and ReTok, while the supertokens approach achieves notable compression gains. Our zero-shot perplexity results demonstrate that the TokenAdapt hybrid initialization consistently yields lower perplexity ratios compared to both ReTok and TransTokenizer baselines across different base models and newly trained target tokenizers. TokenAdapt typically reduced the overall perplexity ratio significantly compared to ReTok, achieving up to approximately a 2-fold improvement in these aggregate scores.

### 1 Introduction

The remarkable success of large language models (LLMs) across diverse natural language processing tasks is often constrained by their tight coupling to the specific tokenization schemes employed during their pre-training phase. This inherent binding between the learned representations of the model and its initial tokenizer presents a misalignment of the vocabulary that compromises semantic fidelity during adaptation. Suboptimal tokenization can cause processing inefficiencies, primarily through token fragmentation, which breaks down meaningful semantic units into excessive subtokens. This issue is particularly pronounced when encountering out-of-vocabulary text common to multilingual or specialized corpora (Minixhofer et al. [2023b]; Dobler and de Melo [2024]). The resulting increase in computational demands and inference latency (Chen et al. [2024]; Ahia et al. [2023]) can directly hinder downstream task performance, an effect especially pronounced in multilingual applications or specialized domains hampered by poor vocabulary coverage and representation (Balachandran and M [2023]; Uygun et al. [2024]; Aribandi et al. [2024]). Overcoming this tokenizer lock-in without sacrificing the immense knowledge encoded in pre-trained models and avoiding prohibitive retraining costs is therefore a crucial research challenge. Existing strategies to bridge this gap vary significantly in approach and cost. A prevalent strategy involves extending the original vocabulary with target-specific tokens, followed by substantial Continued Pre-Training (CPT) or Language Adaptive Pre-Training (LAPT) in relevant data (Balachandran and M [2023]; Uygun et al. [2024]; Gala et al. [2024]).

Tokenadapt Framework is available at : Tinycompany-AI/TokenAdapt SuperTokenizer Training code is available at : Tinycompany-AI/SuperTokenizer

Although effective, particularly for under-represented languages, this approach requires significant computing resources and large target language corpora, which are often costly or unavailable (Aribandi et al. [2024]). Efforts to optimize CPT through data selection underscore the inherent expense of this training-centric paradigm.

Crucially, however, vocabulary extension alone often fails to rectify inefficiencies stemming from the original tokenizer’s merge operations, which may be suboptimal for the target data or specialized domain. Newly added tokens exist alongside these potentially inefficient merge patterns learned on the source data, meaning the overall tokenization efficiency and compression may not improve significantly compared to using a tokenizer specifically trained on the target corpus. This limitation highlights a key advantage of tokenizer transplantation: replacing the tokenizer entirely allows adopting potentially superior merge strategies inherent in a target trained tokenizer, leading to better compression and sequence representation. Consequently, when the number of required new tokens becomes substantial (e.g., exceeding several hundred), transplantation is often the recommended approach as it addresses both the vocabulary gap and fundamentally improves tokenization efficiency. When combined with effective initialization techniques (as explored in this work), transplantation offers a path to both improved tokenization efficiency and faster model convergence compared to mere vocabulary extension. This suggests that simply adding a limited number of tokens might be a less holistic or efficient long-term solution than adopting a fully transplanted, well-initialized tokenizer, especially for significant domain or language shifts.

Recognizing these pitfalls, another major line of research focuses on replacing the tokenizer completely and developing effective initialization strategies for the new embedding layers to bootstrap the adaptation process. Early work demonstrated feasibility by retraining only the embedding layer, sometimes from random initialization, while freezing the core model de Vries et al. [2021]. Recent approaches introduce more targeted initialization methods: ReTok employs simple averaging of constituent sub-token embeddings (derived via the old tokenizer) before efficient peripheral training of input/output layers Chen et al. [2024]. Other techniques leverage auxiliary resources: FOCUS uses similarities between new and overlapping tokens calculated in an auxiliary fastText embedding space Minixhofer et al. [2023a]; WECHSEL finds n-nearest neighbor source subwords using static multilingual embeddings Minixhofer et al. [2021]; and CLP-Transfer combines overlapping source embeddings using similarities derived from a smaller target-language model Minixhofer et al. [2023b]. While these methods improve upon random initialization and accelerate adaptation (often requiring subsequent LAPT), they may face limitations: simple averaging can lack semantic precision Chen et al. [2024]; reliance on overlap or neighbors may not generalize well to highly divergent vocabularies; and dependence on auxiliary or static resources introduces potential alignment issues and computational costs associated with these external dependencies. Crucially, most still necessitate a non-trivial, albeit reduced, training phase to achieve optimal performance. More advanced techniques aiming for true zero-shot transfer exist, such as using hypernetworks Minixhofer et al. [2024] or SMT-based alignment with parallel data Gee and Manning [2024], but these often introduce significant upfront training complexity or specific data requirements (e.g., parallel corpora).

Addressing this, we introduce a novel framework for tokenizer transplantation, designed for broad applicability across common transformer Vaswani et al. [2023] architectures (handling both tied and untied embedding configurations). The core of our transplantation method is a hybrid heuristic initialization strategy for newly introduced unique vocabulary tokens. This strategy synergistically combines two distinct estimates:

- 1. A local, compositional estimate: New tokens are first decomposed using the original tokenizer. A highquality external text embedding model then assesses semantic similarities between the full token string and its constituent sub-token strings. These similarities provide weights for combining the original embeddings of the identified sub-tokens. The sub-token similarities are weighted using the length normalization as empirical results demonstrate its effectiveness.
- 2. A global similarity estimate: Using the same external embedding space and efficient vector search, we identify the nearest neighbors for the new token within the entire original vocabulary based on semantic similarity. The original embeddings of these neighbors are subsequently weighted according to their similarity scores.

The final initialization assigned to a new token is a weighted combination of these local and global estimates. This approach is carefully designed to accurately project new tokens into the original model’s embedding space from the very beginning, thus preserving crucial semantic relationships. By prioritizing such high-fidelity initialization, along with our proposed supertoken learning methodology, we aim to achieve tokenizer flexibility in LLMs.

[Figure 1]

- Figure 1: Core logic of the Local Heuristic.

[Figure 2]

- Figure 2: Core logic of the Global Heuristic.

### 2 Background and Related Work

Our research on enabling tokenizer flexibility intersects with several established areas, including language model adaptation techniques, cross-lingual transfer methods, advancements in tokenization algorithms, and methods for manipulating embedding spaces. This section situates our contributions specifically, the TokenAdapt framework featuring a novel hybrid heuristic for model-agnostic transplantation, and the exploration of learned ’supertokens’ within this landscape.

#### 2.1 The Interplay of Tokenization and Model Adaptation

The choice of tokenization scheme profoundly impacts Large Language Model (LLM) performance and computational efficiency. Standard subword algorithms like BPE Sennrich et al. [2016], WordPiece, and Unigram aim for a balance between vocabulary size and sequence length, yet their effectiveness often diminishes outside their primary training data’s domain or language Dobler and de Melo [2024]. Pre-training primarily on high-resource languages frequently leads to over-fragmentation when processing morphologically complex languages, low-resource languages, or specialized text such as source code Aribandi et al. [2024]. This fragmentation not only inflates sequence lengths, thereby increasing computational costs during training and inference Chen et al. [2024], but can also degrade performance by splitting coherent semantic units Aribandi et al. [2024]. Moreover, the inherent coupling between a model’s learned representations and its specific tokenizer creates barriers to interoperability Minixhofer et al. [2024].

Addressing these limitations necessitates adapting pre-trained models to new vocabularies or domains, often involving significant computational effort. A common paradigm is vocabulary extension coupled with Continued Pre-Training (CPT) or Language-Adaptive Pre-Training (LAPT) on target-specific data Gururangan et al. [2020], Balachandran and M [2023], Uygun et al. [2024], Gala et al. [2024]. While effective for specialization, the high compute and data demands can be prohibitive Aribandi et al. [2024]. Furthermore, simply adding tokens may not fix underlying inefficiencies if the original tokenizer’s merge strategy is suboptimal for the target data. More resource-efficient approaches involve peripheral fine-tuning, where only outer layers like embeddings are retrained, sometimes after basic initialization de Vries et al. [2021], Chen et al. [2024]. However, these methods generally still require a non-trivial training phase. This motivates the exploration of entirely replacing the tokenizer and developing sophisticated initialization techniques to minimize or eliminate post-transplantation training, the primary focus of the next subsection and our work.

#### 2.2 Embedding Initialization Strategies for Tokenizer Transplantation

When replacing a tokenizer entirely, initializing the new embedding matrix effectively is crucial for preserving the base model’s capabilities and minimizing retraining. Various strategies have been proposed, ranging in complexity and reliance on external resources.

Simple heuristics often leverage the old tokenizer’s segmentation directly. ReTok Chen et al. [2024], for example, initializes a new token’s embedding by averaging the original embeddings of the sub-tokens produced by the old tokenizer. While computationally cheap, averaging can obscure semantic nuances, especially with heavy fragmentation.

Other methods utilize auxiliary semantic spaces to inform initialization. Overlap-based techniques like FOCUS

- Minixhofer et al. [2023a] (using fastText) and CLP-Transfer Minixhofer et al. [2023b] (using smaller LMs) weight combinations of existing embeddings based on similarities calculated in these external spaces. Neighbor-based approaches like WECHSEL Minixhofer et al. [2021] identify semantically similar source tokens using static multilingual embeddings and combine their vectors. These auxiliary methods incorporate richer semantic information but often rely on finding suitable overlapping or neighboring tokens in the source vocabulary and may still serve as precursors to further fine-tuning. Our TokenAdapt hybrid approach incorporates aspects of both compositional analysis (local heuristic) and neighbor-based averaging (global heuristic), distinctively weighting the local component based on internal structure (length normalization) alongside external semantic guidance, aiming for higher zero-shot fidelity.

More complex mapping techniques promise deeper alignment but come with significant overhead. ZeTT

- Minixhofer et al. [2024] trains model-specific hypernetworks, requiring substantial upfront computation. TransTokenizer Gee and Manning [2024] uses SMT alignment from large parallel corpora, limiting its applicability where such data is unavailable. These represent heavier-weight solutions compared to the more data-lean, heuristic-based initialization targeted by TokenAdapt.

- 2.3 TokenAdapt

Our work, TokenAdapt, contributes to this landscape by offering an efficient and broadly applicable framework for tokenizer replacement. Its core innovation is the hybrid initialization strategy, which uniquely balances an internal compositional analysis (local heuristic) with a global semantic context check (global heuristic), both informed by auxiliary embeddings. Unlike many prior methods requiring significant retraining or complex prerequisites (like parallel data or hypernetworks), TokenAdapt is designed for high zero-shot semantic preservation, minimizing adaptation costs. Furthermore, our investigation into ’supertokens’ explores synergistic improvements at the tokenization layer itself, pursuing enhanced representational efficiency complementary to the transplantation process, similar in spirit but distinct in mechanism from approaches like SuperBPE Liu et al. [2025].

### 3 Methodology: Semantic Grafting via TokenAdapt

The core of the method lies in initializing embeddings for unique tokens (Vunique = Vnew \ Vold). For each unique token tnew, its embedding enew is synthesized by combining two estimates: a local compositional estimate (detailed in Section 3.1) and a global neighborhood estimate (detailed in Section 3.2).

#### 3.1 Local Heuristic: Compositional Semantic Reconstruction

This heuristic approximates a unique token’s embedding based on its constituent parts as defined by the original tokenizer, weighted by semantic similarity and relative length. Let snew be the string of tnew and ϕ¯new its normalized auxiliary embedding.

##### Process:

- 1. Decomposition: Tokenize snew using Told to obtain a sequence of valid original sub-token IDs Sold = (id1,...,idm). Let sj = Told.decode(idj).
- 2. Weight Calculation: Compute contribution weights wjlocal:

- (a) Calculate semantic similarity: αjsem = ϕ¯new ·ϕ¯j (using normalized auxiliary embeddings ϕ¯j of sub-tokens).
- (b) Calculate initial semantic weights: w˜jsem = Softmax({αpsem}p)j.
- (c) Calculate length normalization score: λj = len(sj)/max(1,len(snew)).
- (d) Compute combined intermediate score: cj = ( ˜wjsem + λj)/2.
- (e) Compute final weights using temperature τ:

exp(cj/τ) m p=1 exp(cp/τ)

wjlocal = Softmax({cp/τ}p)j =

(1)

- 3. Embedding Synthesis: Construct the local estimate elocal by summing the original embeddings Eold weighted by wjlocal:

m

wjlocal · Eold[idj,:] (2)

elocal =

j=1

This calculation yields einlocal and eoutlocal (if untied) using the corresponding original embedding matrices.

#### 3.2 Global Heuristic and Hybrid Integration

This component estimates the embedding based on semantically similar tokens from the entire original vocabulary, identified via the auxiliary space.

##### Global Heuristic Process:

###### 1. Neighbor Discovery: Query the auxiliary kNN index Iaux with ϕ¯new (the new token’s auxiliary embedding)

to find the k nearest neighbors from Vold. Let the valid results be N = {(idneigh1 ,α1glob),...,(idneighk′ ,αkglob′ )} where k′ ≤ k.

###### 2. Similarity Weighting: Calculate weights wlglob from the similarities αlglob using softmax with temperature τ:

exp(αlglob/τ) k′ p=1 exp(αpglob/τ)

wlglob = Softmax({αpglob/τ}p)l =

(3)

###### 3. Embedding Synthesis: Compute the global estimate eglob as the weighted average of the original embeddings of the neighbors:

- k′
- l=1

wlglob · Eold[idneighl ,:] (4)

eglob =

This yields einglob and eoutglob (if untied).

Hybrid Integration: The final embedding enew synergistically blends the local (elocal) and global (eglob) estimates using the hyperparameter wglob ∈ [0,1]:

 

(1 − wglob) · elocal + wglob · eglob if both valid elocal if only local valid eglob if only global valid Initial Random Vector if neither valid

(5)

enew :=



This combined embedding enew is assigned to the new embedding matrix Enew. The complete workflow, including model finalization (updating layers, handling weight tying), is shown in Algorithm 1.

An intuitive refinement explored during development was to enforce a minimum cosine similarity threshold, θ,

within the Global Heuristic (Section 3.2). The goal was pragmatic: filter out low-similarity neighbors (αlglob < θ) from the weighted average (Eq. 4), thereby pruning potential noise and theoretically enhancing the fidelity of the

synthesized embedding. However, the empirical results defied this intuition. Applying such thresholds (e.g., θ = 0.5) consistently yielded a paradoxical increase in model perplexity relative to the standard, unfiltered heuristic. This degradation occurred even when ample high-similarity neighbors remained above the threshold. The intricate dynamics of the embedding space evidently resist naive similarity filtering; the optimal contribution blend appears to be more complex. Consequently, the definitive TokenAdapt procedure forgoes this thresholding, embracing the empirically superior performance of the unfiltered Global Heuristic. This finding underscores the subtle, non-linear nature of learned semantic representations.

- Algorithm 1 TokenAdapt: Transplantation Workflow

Require: Mold, Told, Tnew, Φaux, Hyperparameters (τ,k,wglob) Ensure: Updated Model Mnew adapted to Tnew.

- 1: Initialize Enew based on Vnew size and Mold’s embedding dimension.
- 2: Build auxiliary kNN index Iaux over Vold using Φaux. ▷ Foundation for Global Heuristic
- 3: ▷ Phase 1: Inherit Shared Knowledge
- 4: for all shared token t ∈ Vshared do
- 5: Copy embedding eold(t) to Enew at position for t. ▷ Direct transfer
- 6: end for
- 7: ▷ Phase 2: Synthesize Unique Representations
- 8: for all unique token tnew ∈ Vunique do
- 9: Compute Local Estimate elocal (via sub-token composition, length norm).
- 10: Compute Global Estimate eglob (via kNN neighbors in Iaux).
- 11: Combine estimates: enew ← HybridCombine(elocal,eglob,wglob).
- 12: Assign enew to Enew at position for tnew.
- 13: end for ▷ Handle input/output layers appropriately if untied
- 14: ▷ Phase 3: Finalize Model
- 15: Update Mold’s embedding layers with the completed Enew.
- 16: Re-apply weight tying if applicable for Mold.
- 17: return Final adapted model Mnew.

The TokenAdapt workflow (Alg. 1) has 3 phases: (1) Shared token transfer, (2) New token synthesis via local+global embedding fusion, (3) Model finalization with weight tying.

[Figure 3]

- Figure 3: Core logic of the Local and Global Heuristics respectively. This diagram illustrates the two main pathways (Local and Global) for generating components of a new token’s embedding, which are then combined via Hybrid Integration.

### 4 Experimental Setup & Results

The primary base models for transplantation were meta-llama/Llama-3.2-3B (Meta-Llama) and Qwen/Qwen2.5-3B (Qwen). We evaluated adaptation to two target tokenizers: the custom standard fhai50032/QTK-81K and our custom supertoken tokenizer, tinycompany/Adi-Bun-128K. The adaptation algorithm 1 operates in three phases: (1) Direct transfer of shared token embeddings from Mold, (2) Hybrid synthesis of new tokens combining local subword features and global kNN-based embeddings, and (3) Model integration with updated embedding layer and weight tying when applicable. Performance was primarily assessed using zero-shot perplexity on language/domain subsets (English, Hindi, Code, Math, Hinglish) of the tinycompany/ppl dataset.

We compared TokenAdapt against Random Initialization, Mean Initialization, and ReTok [Chen et al., 2024] initialization (simple sub-token averaging). Comparisons with other methods like Transtokenizer [Gee and Manning, 2024] are discussed contextually based on published results. For TokenAdapt, we used the auxiliary embedding function (Φaux) derived from models trained on a diverse corpus including data similar to tinycompany/ppl.

Table 1: Zero-Shot Perplexity Ratio Comparison - Lower is Better

Experiment Context Initialization Method Overall English Code Math Hinglish Hindi

Mean Ratio Mean Ratio Mean Ratio Mean Ratio Mean Ratio Mean Ratio Base: Llama-3.2-3B → Target: QTK-81K

TokenAdapt (Local Only) 64.4 1.21 1.40 2.02 14.3 1682.4 TokenAdapt (Hybrid) 48.2 1.29 1.35 2.16 13.6 1108.4 TokenAdapt (Hybrid+Thr.) 48.2 1.28 1.34 1.53 15.97 1105.1 ReTok Baseline 71.1 1.29 1.42 1.54 14.7 1875.6 TransTokenizer Baseline 145.9 94.1 15.4 73.8 125.1 1052.2

Base: Qwen2.5-3B → Target: QTK-81K

TokenAdapt (Local Only) 109.3 1.44 2.27 1.64 12.6 6149.4 TokenAdapt (Hybrid) 85.8 1.39 1.86 1.37 13.4 4615.9 TokenAdapt (Hybrid+Thr.) 85.04 1.38 1.91 2.00 13.44 775.42 ReTok Baseline 139.1 1.43 2.28 1.64 12.8 8007.9 TransTokenizer 101.6 103.0 23.0 100.4 83.0 1302.1

Base: Llama-3.2-3B → Target: Adi-Bun-128K (ST)

TokenAdapt (Local Only) 1371.3 733.1 87.2 71.0 236.2 41.46 TokenAdapt (Hybrid) 577.5 376.1 93.6 78.2 105.6 13.12 TokenAdapt (Hybrid+Thr.) 589.03 256.23 46.06 194.98 550.37 6997.74 ReTok Baseline 1174.0 1749.8 134.0 806.6 9249.5 9279.7 TransTokenizer Baseline 4173.6 22005.2 1284.6 6110.5 32117.8 10818.8

PPL Ratio = (Transplanted Mean PPL / Original Base Model Mean PPL). Lower ratios indicate less degradation. See Sec. 4 for base PPL values and specific configurations. Parallel data used for Transtokenization experiments were: allenai/nllb and open_subtitles. Bolder datapoint is best result. Underlined is the 2nd best result.

Various heuristic parameters were tested, including the global weight wglob and temperature τ, as specified below. Our zero-shot perplexity evaluations reveal a clear advantage for the TokenAdapt initialization method, particularly the hybrid variant. Across the different experimental contexts, TokenAdapt consistently yielded the lowest overall perplexity ratios, indicating significantly better preservation of the original model’s capabilities immediately following the tokenizer swap compared to alternatives. Both the ReTok and TransTokenizer baselines exhibited substantially higher perplexity ratios, signifying greater degradation in zero-shot performance, as shown in Table 1. Lower ratios indicate better performance preservation. Specific configurations tested, corresponding to the markers in Table 1, include: TokenAdapt (Local Only, ) used wg = 0.0,τ = 0.6; TokenAdapt (Hybrid, ) used wg = 0.3,τ = 0.6; TokenAdapt (Hybrid+Threshold), ) used Hybrid settings with a similarity threshold Thr = 0.45.

### 5 Conclusion

This work confronted the prevalent challenge of tokenizer lock-in inherent in many pre-trained language models. This constraint often hinders efficiency and performance, especially in specialized or multilingual contexts, while demanding significant computational resources for adaptation. Addressing this, we introduced TokenAdapt, a model-agnostic framework designed to facilitate seamless tokenizer transplantation with minimal retraining overhead.

Our core contribution lies in a novel hybrid heuristic for initializing unique token embeddings. By synergistically combining a local estimate—rooted in semantic compositional reconstruction using the original tokenizer’s sub-words and crucially informed by length normalization—with a global estimate derived from semantic neighborhood averaging in an auxiliary embedding space, TokenAdapt effectively grafts semantic understanding onto the new vocabulary structure. Furthermore, we proposed the concept of learned multi-word supertokens as a complementary strategy to enhance sequence compression and mitigate token fragmentation

Our empirical investigations validate the efficacy of this approach. The hybrid transplantation heuristic demonstrated robust zero-shot initialization capabilities, markedly outperforming established baselines and sophisticated methods including Transtokeniser, and ReTok in preserving model performance post-transplantation. Preliminary exploration of supertokens also confirmed their potential for significant compression gains. The counter-intuitive finding—that similarity thresholding in the global heuristic paradoxically degrades performance—underscores the complex nature of embedding space interactions and informed the refinement of our core algorithm.

Ultimately, TokenAdapt offers a practical and computationally efficient pathway for adapting powerful LLMs to new tokenization schemes. This lowers the barrier for tailoring models to specific domains, languages, or efficiency requirements. This work opens avenues for future research, including exploring more adaptive heuristic weighting, investigating alternative auxiliary semantic spaces, developing integrated strategies for co-optimizing supertokens and transplantation, and further dissecting the observed thresholding phenomenon. By providing effective tools to overcome tokenizer limitations, we aim to enhance the versatility and applicability of large language models across a broader spectrum of tasks and resource constraints.

### Acknowledgments

The authors acknowledge the support and resources provided. We would like to thank Tensoic and Google TRC program which enabled this research. The authors also thank our colleagues for insightful discussions and feedback throughout the project.

### References

Lasha Ahia, Alexander Poli, Christopher Ré, Matei Zaharia, and Stefano Ermon. MegaByte: Predicting million-byte sequences with multiscale transformers. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 218–238, Honolulu, Hawaii, USA, July 2023. PMLR. URL https://proceedings.mlr.press/v202/ahia23a.html.

Vamshi Krishna Aribandi, Arpan Mandal, and Nikesh Garera. Cost effective continual pre-training for bridging the llm performance gap for indic languages, 2024. URL https://arxiv.org/abs/2402.10244.

Abhinand Balachandran and Arun Raj M. Tamil-llama: A new tamil language model based on llama 2, 2023. URL https://arxiv.org/abs/2311.05845.

Zhen Chen, Jianing Wang, Qiushi Sun, Xiubo Geng, Nuo Xu, Wenji Mao, and Daxin Jiang. Retok: Replacing tokenizer

to enhance representation efficiency in large language model, 2024. URL https://arxiv.org/abs/2410.04335. codeparrot. github-code. URL https://huggingface.co/datasets/codeparrot/github-code.

Wietse de Vries, Andreas van Cranenburgh, and Malvina Nissim. As good as new. how to successfully recycle English GPT-2 to make models for other languages. In Findings of the Association for Computational Linguistics: ACLIJCNLP 2021, pages 880–890, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/ 2021.findings-acl.74. URL https://aclanthology.org/2021.findings-acl.74.

Miriam Dobler and Gerard de Melo. An empirical study on cross-lingual vocabulary adaptation for efficient language model inference, 2024. URL https://arxiv.org/abs/2402.10712.

Jesse Dodge, Maarten Sap, Ana Marasovi, William Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, and Matt Gardner. Documenting large webtext corpora: A case study on the colossal clean crawled corpus, 2021. URL https://arxiv.org/pdf/2104.08758.

Jay Gala, Thanmay Jayakumar, Jaavid Aktar Husain, Aswanth Kumar M, Mohammed Safi Ur Rahman Khan, Diptesh Kanojia, Ratish Puduppully, Mitesh M. Khapra, Raj Dabre, Rudra Murthy, and Anoop Kunchukuttan. Airavata: Introducing Hindi Instruction-tuned LLM, 2024. URL https://arxiv.org/abs/2401.15006.

Andrew R. Gee and Christopher D. Manning. Trans-tokenization and cross-lingual vocabulary transfers: Language adaptation of llms for low-resource nlp, 2024. URL https://arxiv.org/abs/2408.04303.

Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. Don’t stop pretraining: Adapt language models to domains and tasks, 2020. URL https://arxiv.org/abs/2004. 10964.

Alisa Liu, Sang Michael Xie, Michail Papauschek, Graham Neubig, Jonathan Frankle, Volodymyr Kuleshov, and

Ankit Singh. SuperBPE: Space Travel for Language Models, 2025. URL https://arxiv.org/abs/2503.13423. Meta-Llama. meta-llama/llama-3.2-3b. URL https://huggingface.co/meta-llama/Llama-3.2-3B.

Oscar Minixhofer, Fabian Paischer, and Navid Rekabsaz. Wechsel: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models, 2021. URL https://arxiv.org/abs/2112.06598.

Oscar Minixhofer, Gábor Berend, and Jie Yang. Focus: Effective embedding initialization for language adaptation of large language models, 2023a. URL https://arxiv.org/abs/2305.14481.

Oscar Minixhofer, Fabian Paischer, and Navid Rekabsaz. Efficient language model training through cross-lingual and progressive transfer learning, 2023b. URL https://arxiv.org/abs/2301.09626.

Oscar Minixhofer, Marcelo Orenes-Vera, and Ivan Vulić. Zero-shot tokenizer transfer, 2024. URL https://arxiv. org/abs/2402.08966.

Qwen. Qwen/qwen2.5-3b. URL https://huggingface.co/Qwen/Qwen2.5-3B.

Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, Berlin, Germany, August 2016. Association for Computational Linguistics. doi: 10.18653/v1/ P16-1162. URL https://aclanthology.org/P16-1162.

Emincan Uygun, Erion Çano, and İzzet Emre Kiciman. Llamaturk: Adapting open-source generative large language models for low-resource language, 2024. URL https://arxiv.org/abs/2401.03608.

JAshish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2023. URL https://arxiv.org/abs/1706.03762.

Yifan Zhang, Yifan Luo, Yang Yuan, and Andrew Chi-Chih Yao. Automathtext: Autonomous data selection with language models for mathematical texts, 2024. URL https://arxiv.org/pdf/2402.07625.

zicsx. mc4-hindi-cleaned-3.0. URL https://huggingface.co/datasets/zicsx/mC4-Hindi-Cleaned-3.0.

### A Experimental Details

#### A.1 SuperTokenizer Training Pseudocode

Algorithm 2 outlines the supertokenizer training, explicitly showing the stochastic chunking that precedes BPE training. Algorithm 2 Supertoken Tokenizer Training (Core Logic v2)

- 1: function AugmentedIterator(C,Pdist,sepstr)
- 2: for all text t ∈ C do
- 3: lengths ← GenerateChunkLengths(|t|,Pdist) ▷ Prob. lengths based on Pdist
- 4: taug ← ””; pos ← 0
- 5: for all l ∈ lengths do
- 6: chunk ← t[pos : pos + l]
- 7: taug ← taug + chunk
- 8: if not last chunk then taug ← taug + sepstr
- 9: end if
- 10: pos ← pos + l
- 11: end for
- 12: yield taug
- 13: end for
- 14: end function
- 15:
- 16: Initialize Tsuper with models.BPE().
- 17: Set Normalizer, Pre-tokenizer (Split(sepstr,’removed’) then ByteLevel), Post-processor, Decoder.
- 18: Configure trainer ← BpeTrainer(V,Stok,...).
- 19: iteratoraug ← AugmentedIterator(C,Pdist,sepstr).
- 20: Train Tsuper.train_from_iterator(iteratoraug,trainer).
- 21: Wrap Tsuper with PreTrainedTokenizerFast, add necessary tokens.
- 22: return Tsuper.

#### A.2 Supertoken Training Explanation

Algorithm 2 details the supertoken training. The key component is the AugmentedIterator. For each text t from the corpus C, it first calls GenerateChunkLengths which determines a sequence of chunk lengths based on the probability distribution Pdist. It then reconstructs the text as taug by concatenating these chunks and inserting the separator string sepstr between them. The main tokenizer Tsuper is configured such that its pre-tokenizer first splits taug using sepstr (and removes it), before applying standard byte-level processing. The BPE trainer then operates on the resulting stream of variable-length chunks generated by the iterator. This process encourages BPE merges to occur primarily *within* these probabilistically defined chunks, fostering the creation of longer "supertokens". The tokenizer is finalized by wrapping with Hugging Face’s tools and adding custom tokens.

#### A.3 Datasets Used

For tokenizer training, we curated a diverse collection of text spanning multiple domains and languages to support robust and generalizable tokenization. Data sources included AutoMathText Zhang et al. [2024], Allen AI C4 English Dodge et al. [2021], Codeparrot GitHub codeparrot, and MC4 Hindi Cleaned zicsx. These datasets were selected to ensure coverage across technical writing, natural language, code, and multilingual content.

To maintain consistency and efficiency during training, we sampled a representative subset from these corpora. Special care was taken to address potential issues arising from document duplication and distributional imbalance. In particular, we applied standard preprocessing and filtering techniques to reduce the influence of outlier sequences and duplicated content. This helped avoid skewed token statistics that could impact the learned vocabulary.

#### A.4 Domain-Specific Utilization of Multi-Word Tokens

We conducted a comparative analysis to assess the real-world token granularity resulting from our probabilistic pretokenization strategy. By tokenizing a few thousands samples from each target domain (English, Hindi, Math, Code)

with both our Supertoken and the baseline tokenizer, we could directly compare the prevalence of multi-word tokens actually used. This analysis empirically confirms our hypothesis: the Supertoken tokenizer consistently employs a higher proportion of multi-word units than the baseline across these domains, translating the theoretical design into practical effectiveness. The compression ratio observed across multiple domains when using the Supertokenizer support the growing body of evidence suggesting limitations in purely subword-based approaches. Similar positive results were reported by Goyal et al. Liu et al. [2025], whose SuperBPE method achieved notable downstream gains by enabling tokens to span whitespace. Our work, using a different technique, further validates the potential performance benefits of incorporating multi-word units into the tokenizer’s vocabulary.

[Figure 4]

- Figure 4: Comparison of word count distributions for unique token types utilized by the Supertoken (ST, Red) and Baseline (Base, Blue) tokenizers across different domain corpora (10k samples each). Subplots show results for (a) English, (b) Hindi, (c) Math, and (d) Code. The Y-axis represents the log count of unique token types observed containing the specified number of words (X-axis).

Table 2: Tokenizer Compression Comparison Across Domains (Total Tokens Used - Lower is Better)

Tokenizer English Hindi Python Java C++ Math Adi-Bun (128k)a 283,657 9,174 4,765 1,821 11,464 3,887 DeepSeek-R1 (128k)b 338,873 22,854 5,270 2,223 13,081 5,375 Krutrim-Ins (131k)c 343,067 15,477 5,246 2,223 13,007 5,361 Gemma-3-27b(262k)d 329,308 10,833 5,998 2,590 16,175 5,867 a tinycompany/Adi-Bun-128K b deepseek-ai/DeepSeek-R1 (128k variant) c krutrim-ai-labs/Krutrim-2-Instruct d unsloth/Gemma-3-27b-it-unsloth-bnb-4bit

#### A.5 A Look at Tokenization Granularity

Our proposed Supertokens extend the concept of standard tokenization by learning multi-word units through a probabilistic pre-tokenization strategy, aiming to enhance compression and reduce fragmentation compared to conventional methods. A comparative look at how these diverse tokenizers segment identical text passages reveals these inherent differences.

Table 3: Comparison of text segmentation by different tokenization strategies.

##### Tokenizer Tokenization

[Figure 5]

GPT4o

(203 Bytes, 43 Tokens: 4.721 Bytes/token)

Llama3

(203 Bytes, 43 Tokens: 4.721 Bytes/token)

[Figure 6]

SuperBPE 200K

(203 Bytes, 30 Tokens: 6.767 Bytes/token)

[Figure 7]

[Figure 8]

Adibun 128K (Ours)

(203 Bytes, 32 Tokens: 6.344 Bytes/token)

