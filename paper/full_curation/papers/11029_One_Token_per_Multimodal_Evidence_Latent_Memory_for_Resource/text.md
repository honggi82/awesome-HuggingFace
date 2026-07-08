# arXiv:2606.10572v1[cs.AI]9Jun2026

## One Token per Multimodal Evidence: Latent Memory for Resource-Constrained QA

Zhi Zheng∗ Ziqiao Meng Hao Luan Wei Liu Wee Sun Lee School of Computing, National University of Singapore

### Abstract

External memory effectively grounds large language models (LLMs) and visionlanguage models (VLMs)-based question answering (QA) in relevant multimodal evidence. However, existing memory paradigms represent each memory item in raw text and image forms, so retrieval-based systems must pass the retrieved text or images to the generation LLMs/VLMs, resulting in high token consumption and storage pressure, making it unaffordable for resource-constrained applications. We propose Latent Memory, a latent-space memory paradigm that replaces each raw text or image evidence item with a single high-dimensional latent token produced by a small compressor LLM/VLM. Rather than retrieving raw evidence for generation, Latent Memory operates in a unified latent representation space: the query is embedded into this space to retrieve relevant latent tokens, and the retrieved latent tokens are directly prompted to a pretrained LLM or VLM for answer generation. To make each latent token simultaneously informative for reconstruction, retrieval, and generation, we train the compressor with reconstruction, contrastive, and distillation objectives in a unified end-to-end manner. Latent Memory is evaluated on seven text-only QA benchmarks (e.g., HotpotQA) and multimodal QA benchmarks, where it achieves competitive QA performance compared to advanced RAG baselines while consuming 3× to 10× fewer generator tokens. It can also deliver the strongest image-grounded QA performance on WebQA.2

### 1 Introduction

Large language models (LLMs) and vision-language models (VLMs) have demonstrated remarkable capabilities in complex reasoning and knowledge-intensive tasks [1, 2, 3, 4], especially when equipped with an external memory [5] and then retrieving the relevant memory items [6] for faithful and reliable generation. This memory usually contains external knowledge evidence or dialogue history in a multimodal form, allowing models to ground their outputs in long-context tasks or multi-turn dialogues, thus becoming a core component of a wide range of applications, including long-context question answering (QA) [7, 4], coding agents [8, 9], and agentic AI assistants [10, 11].

Although external memory can improve answer generation by providing relevant evidence, the prevailing memory paradigm remains computationally expensive. The main bottleneck lies in generation, where the evidence in the memory is prompted to the LLM/VLM generator in uncompressed form, incurring substantial token costs and latency overhead. This problem is further amplified in handling the multimodal evidence in the memory, where each image may require megabytes of storage, and will expand into hundreds of visual tokens during generation [12, 13]. Together, these costs limit the scalability of external memory for long-context and multi-turn interactions, and make it difficult to deploy memory-augmented systems under tight storage or latency constraints, such as on-device assistants and edge-AI applications [14, 15, 16].

∗Email: zhi.zheng@u.nus.edu 2Code is available at https://github.com/zz1358m/Latent-Memory-Master

Preprint.

QA with Explicit Contextual Memory (Existing Paradigm)

QA with Explicit Contextual Memory (Existing Paradigm)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

: Textual Evidence : Image Evidence

: Textual Evidence : Image Evidence

Contextual Memory C

Contextual Memory C

RAG Context1:......

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

RAG Context1:......

Annie Morton: Terry Richardson:

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Annie Morton: Terry Richardson:

- Context2:......
- Context3: <image> Title:......

[Figure 13]

[Figure 14]

- Context2:......
- Context3: <image> Title:......

[Figure 15]

[Figure 16]

###### Output Answer A

###### Output Answer A

[Figure 17]

[Figure 18]

- Image1: with Title:
- Image2: with Title: High Storage Cost

[Figure 19]

[Figure 20]

- Image1: with Title:
- Image2: with Title: High Storage Cost

Frozen Larger-Size LLM/VLM for Generation

Frozen Larger-Size LLM/VLM for Generation

[Figure 21]

[Figure 22]

Terry Richardson

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Terry Richardson

[Figure 27]

[Figure 28]

Input

Input

Thousands of Tokens

Thousands of Tokens

High Token Cost and Slow Processing

[Figure 29]

High Token Cost and Slow Processing

[Figure 30]

Query Q Who is older, Annie Morton or Terry Richardson?

Query Q Who is older, Annie Morton or Terry Richardson?

(a) The conventional RAG-based contextual generation

- ① Transforming Context C to Latent Memory (Ours)
- ② Efficient QA with Retrieved Latent Memory (Ours)

- ① Transforming Context C to Latent Memory (Ours)
- ② Efficient QA with Retrieved Latent Memory (Ours)

1

1

2

Contextual Memory C

Latent Memory M

2

Contextual Memory C

Latent Memory M

- Latent 1: 1
- Latent 2: 2
- Latent 3: 3 ∈ 

- Latent 1: 1
- Latent 2: 2
- Latent 3: 3 ∈ 

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Annie Morton: Terry Richardson:

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

3

Annie Morton: Terry Richardson:

LLM/VLM with Trainable LoRA for Compression

3

[Figure 39]

LLM/VLM with Trainable LoRA for Compression

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Compressing each multimodal evidence unit to One-Token

- Image1: with Title:
- Image2: with Title:

[Figure 47]

[Figure 48]

Compressing each multimodal evidence unit to One-Token

- Image1: with Title:
- Image2: with Title:

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Unified Latent Representation Space for Retrieval and Generation

[Figure 54]

Unified Latent Representation Space for Retrieval and Generation

Low Storage Cost

Low Storage Cost

1

Compress query to ∈  for retrieval

1

Compress query to ∈  for retrieval

2

- Latent Context1: Output Answer A

- Latent Context2: 1

2

- Latent Context1: Output Answer A

- Latent Context2: 1

Frozen Larger-Size LLM/VLM for Generation

Frozen Larger-Size LLM/VLM for Generation

Terry Richardson

[Figure 55]

[Figure 56]

3

[Figure 57]

Top Similarity

Terry Richardson

[Figure 58]

[Figure 59]

3

[Figure 60]

Top Similarity

One-Token Each

Input

One-Token Each

Input

[Figure 61]

Better Efficiency-Performance Trade-off

Query Q Who is older, Annie Morton or Terry Richardson?

[Figure 62]

Better Efficiency-Performance Trade-off

Query Q Who is older, Annie Morton or Terry Richardson?

(b) Generation with retrievable Latent Memory (Ours)

- Figure 1: (a) shows the existing pipeline for memory-based generation. To improve storage efficiency and token efficiency, our Latent Memory (b) can compress each multimodal evidence into one latent token, which achieves better retrieval ability and competitive generation performances.

In pursuit of an efficient memory representation paradigm for applications with strict latency or storage constraints, this paper investigates whether multimodal contextual memory can be represented as highly compressed one-token latent representations capable of replacing raw evidence during generation. To this end, we propose Latent Memory, a framework in which each unit of evidence is compressed into a compact, retrievable, high-dimensional latent token that can be directly utilized by a frozen LLM/VLM generator without fine-tuning. As illustrated in Figure 1(b), Latent Memory uses a small compression LLM/VLM to encode each text or image context into a single latent token, forming a unified multimodal latent representation space. At query time, the question is embedded into the same space to retrieve relevant latent tokens through similarity search. These latent tokens are then projected into the hidden dimension of a frozen LLM or VLM for answer generation, replacing the conventional token-based context. This design enables:

- (1) Token Efficiency and Storage Efficiency: The proposed Latent Memory leads to fewer token consumptions and lower storage pressure to handle contextual multimodal evidence.
- (2) Unified Representation and Retrieval Space: The latent tokens are used for both contextual representation and retrieval, making it easier to obtain better low-budget embeddings.

To obtain better latent tokens for multimodal evidence, we fine-tune a small LLM/VLM for tokens that preserve both the raw information and retrieval capability. So, the training loss combines: (i) a reconstruction loss to ensure latent tokens carry sufficient information of the original evidence; (ii) a contrastive loss to align queries with only their labeled supporting evidence in the latent space; and (iii) a distillation loss to align the generation behavior prompted on latent tokens with that of the behavior generator prompted with raw positive evidence. We evaluate Latent Memory on both text-only and multimodal benchmarks. On 2WikiMultihopQA and MuSiQue, one-token Latent Memory trained on HotpotQA attains competitive EM/F1 compared to raw-evidence-level retrieval augmented generation (RAG) baselines using only 3× less LLaMA-8B or Mistral-7B generator tokens. On multimodal QA benchmarks, Latent Memory yields 10× efficiency gains and the outstanding image-grounded QA results under the LLaVA-13B and Gemma3-12B generators. Our contributions are as follows:

- • We propose Latent Memory, an efficient multimodal memory paradigm for resource-constrained scenarios, which compiles each evidence item into a single high-dimensional latent token.

- • We show that one-token Latent Memory can support both retrieval and answer generation through a training objective that combines reconstruction, contrastive learning, and distillation.
- • Empirically, Latent Memory substantially improves the efficiency-performance trade-off across eight text-only and multimodal contextual QA benchmarks, and four generator LLM/VLMs.

### 2 Related Work

External Memory and RAG. For faithful QA, LLM/VLM usually takes the provided multimodal context as memory and cooperates with the external memory via full-context prompting [17] or RAG [18, 19, 20]. Dense Retrieval [21], BM25 Retrieval [22], and other recent RAG methods [23, 24] encode the feature of queries and evidence into a shared space for the nearest-neighbor search. In multimodal settings, recent multimodal embedding methods [25, 26, 27] retrieve images based on embedding models and prompt image tokens (576 in LLaVA VLMs [12] and 256 in Gemma VLMs [13]) to VLMs. However, all of these methods represent memory at the raw-evidence level; as the multimodal memory scales up, this representation becomes a storage and efficiency bottleneck. Latent Memory stores multimodal evidence as compact latent tokens for less storage space, and injects only latent vectors rather than raw evidence into the generation LLM for token efficiency.

Evidence Compression. Besides the RAG method, there are also Evidence Compression methods proposed to address the over-lengthened large memory. LLMLingua [28] learns to drop unnecessary tokens, AutoCompressor [29], ICAE [30], and LCC [31] learn to compress text documents into a small number of continuous tokens for efficient LLM-based QA generation, while xRAG [32], and the concurrent work, CLaRa [33] take the same latent for retrieval. However, these methods focus only on text-based situations, making them incompatible with recent multimodal memory settings. Moreover, as detailed in Appendix A.2, considering that high-dimensional tokens are actually larger than the raw text evidence units, these algorithms cannot provide the storage saving as Latent Memory does in multimodal settings (discussed in Appendix B.5).

Latent-Space Reasoning. Latent reasoning methods [34, 35, 36, 37, 33, 31, 38, 39, 40] show that LLMs can reason and communicate through continuous latent states rather than discrete token sequences, while [41, 42, 43, 44] tries to represent multimodal comprehension and generation in a unified embedding space. Latent Memory applies the idea principle to design a new memory paradigm, where multimodal evidence is encoded, retrieved, and prompted in a unified latent representation.

### 3 Methodology: Latent Memory

To achieve efficient external memory, this paper proposes the Latent Memory paradigm. In this section, we first present how the one-token Latent Memory is built and used for QA generation. Then, we will describe how the compression LLM/VLM is trained to produce the Latent Memory.

##### 3.1 Definition: Latent Memory for QA with Contextual Memory

A QA problem seeks the answer A = (a1,a2,...,a|A|) of question Q = (q1,q2,...,q|Q|) with N external contexts C = {xi}Ni=1. Prompting all the contexts C for generation will improve perplexity and may exceed the pre-trained context window in some cases. So RAG systems are usually employed to retrieve a subset of raw evidence R(q,C) and prompt them to a LLM/VLM generator as follows:

###### |A|

P(at | a<t,Q,R(Q,C),ϕ). (1)

P(A | Q,C,ϕ) =

t=1

As shown in Figure 1(b), Latent Memory changes this interface by storing and passing raw evidence to a collection of compact latent tokens. Latent tokens serve as a retrieval representation for selecting relevant evidence, and then the retrieved tokens are directly prompted to the generator as a continuous evidence token. The full inference-time pipeline consists of four components.

Memory Compression. For each evidence item xi ∈ C, we use a small compressor LLM/VLM θ to produce a single Latent Memory token. Concretely, xi is appended with a learnable embedding (noted [MEM]), and we take the final hidden state of this token as the latent token:

zi = θ(xi,[MEM]) ∈ Rd

, (2)

θ

Reconstruction Loss Recon for Context

Amanda Lepore:

Supervision

LLM with Trainable LoRA for Text Reconstruction

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

e text = CE(text) Recon img = img MSE(image)

Input

Latent Token  ∈ 

LLM/VLM with Trainable LoRA for

[Figure 67]

Amanda Lepore:

Output

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Input

Input MLP

[Figure 73]

Input Compression

[Figure 74]

[Figure 75]

Embeddings of

Supervision

Contrastive Loss Contrast for Latent Memory Retrieval

 ∈ +

Positive Contexts +

 ∈ −

Constructing Latent Memory +

Output

Input

[Figure 76]

[Figure 77]

[Figure 78]

Annie Morton: Terry Richardson:

[Figure 79]

[Figure 80]

LLM/VLM with Trainable LoRA for Compression

Negative Contexts −

Contrast = −1 | +|

Constructing Latent Memory −

Latent Representation Space

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Output

exp( T / )

Amanda Lepore: Madonna (book):

Input

log

[Figure 85]

[Figure 86]

[Figure 87]

∈ + − exp( T / )

[Figure 88]

Latent Query Token  ∈ 

[Figure 89]

Query Q

Output

∈ +

Input

Who is older, Annie Morton or Terry Richardson?

Distillation Loss Distill for Latent Token Representations

Prompt with +

Context 1: ; Context 2: Who is older, Annie Morton or Terry Richardson?

Ref Answer A’

Terry Richardson

Distill = KL( ’,  )

Input

Output

[Figure 90]

[Figure 91]

Latent Prompt with +

Frozen Larger-Size Generation LLM/VLM

Latent context 1: 1 ; Latent context 2: 2 Who is older, Annie Morton or Terry Richardson?

[Figure 92]

[Figure 93]

Answer A

Output

Terry Richardson

Input

- Figure 2: The training process of the compressor and decoder consists of three losses. Reconstruction Loss LRecon aims at recovering the raw text and image in a teacher-forcing way. Contrastive Loss LConstrast aligns the query embedding to positive latent tokens. Distillation Loss LDistill aligns the generator output between prompting raw evidence and latent tokens.

where dθ is the hidden dimension of the model θ. The Latent Memory M = {zi}Ni=1 is then formed by collecting all these latent tokens while discarding the original raw evidence.

Retrieving Latent Memory. Prompting generators with compressed contexts may be enough for faithful generation within the context window. However, with a large amount of context, the corresponding latent tokens still pose a significant processing difficulty. So, we design to make the Latent Memory token retrievable, projecting zi into a dr dimensional retrieval space with an MLP projection, noted zi = MLPr(zi), MLPr : Rd

θ. In retrieving the most relevant latent context for the query q, we compress the query into a query representation q ∈ Rd

→ Rd

r

r in this retrieval space. We then retrieve the top-k memories by inner product:

q⊤ zi. (3)

I = arg top-k

i

Latent-Conditioned Generation. Retrieved latent tokens {zi : i ∈ I} are mapped into the higherdimensional hidden space of the frozen LLM/VLM ϕ for generation. Let zj denote the projected latent token corresponding to the j-th retrieved memory. We construct the input embeddings of the generator with another projector Wg ∈ Rd

ϕ×dθ and prompt them to ϕ as follows:

###### |A|

P(at | a<t,Q, z1,..., zk,ϕ), where zi = Wgzi. (4)

P(A | Q,C,ϕ,θ) =

t=1

Reconstruction (Optional). To preserve some interpretability, Latent Memory can be roughly reconstructed back to the raw multimodal context. Through a fine-tuned decoder π, we can recover the raw-text for text-evidence or the caption of an image evidence in an autoregressive manner as follows:

|xi|

P(xi,t | xi,<t,zi,π). (5)

P(xi | zi) =

t=1

Latent Memory also allows image reconstruction for the latent tokens that are compressed from an image. We train a multi-layer perceptron (MLP) to predict the CLIP embedding of the image [45]. Then, the image can be roughly recovered with a pre-trained diffusion-based image generator unCLIP [46], conditioning on the recovered CLIP embedding.

##### 3.2 Training the Compressor for Latent Memory

Latent Memory carries retrieval, information-providing, and optional reconstruction roles. In this subsection, we will introduce the algorithm to fine-tune a powerful compressor θ so that one latent token can support all these roles simultaneously.

As illustrated in Figure 2, the training procedure optimizes the compressor with three complementary signals. (1) A reconstruction objective encourages each compressed latent token zi to preserve the content of its original evidence. (2) A contrastive objective shapes the retrieval space by pulling queries close only to their supporting evidence. (3) A distillation objective aligns the behavior of the frozen generator conditioned on latent memories with the behavior of the same generator conditioned on raw evidence. To avoid catastrophic forgetting, the large generator ϕ is kept frozen throughout training. We only optimize the LoRAs of compressor LLM/ LLM θ, reconstruction decoder LLM π, and retrieve and generate projections Wr and Wg. The training process only requires supervision from positive samples, and does not need other supervision signals, e.g., labeled answers.

Multimodal Reconstruction Loss. A one-token memory should not collapse into a purely discriminative retrieval identifier. It should still preserve recoverable information about the evidence item it represents. Therefore, for a text evidence item xi (or the caption of an image), we fine-tune an LLM decoder π to reconstruct xi in its Latent Memory zi compressed with θ and a trainable LoRA form Eq. (2) in a teach-forcing way as follows:

|xi|

LtextRecon = −

log Pπ (xi,t | xi,<t,zi). (6)

t=1

i

For image evidence, we do not reconstruct raw pixels. Instead, following the idea of unCLIP-style reconstruction [46], we reconstruct the CLIP image embedding of the original image. Given an image

evidence item ximgi and its latent token zimgi . We train a lightweight MLP to predict a CLIP-space image embedding vi with the loss function as follows:

∥vi − MLP(zi)∥22 . (7)

LimgRecon =

i

In multimodal training, the reconstruction term combines the available text-side and image-side reconstruction signals as LRecon = LtextRecon + λimgLimgRecon.

Contrastive Retrieval Loss. The retrieval projection in Section 3.1 maps each Latent Memory zi into a retrieval vector zi. To make this space useful for evidence selection, we train another LoRA of compressor for query representation q to be close to the latent memories of supporting evidence and far from irrelevant memories. For each question Q, let M+ and M− denote its positive and sampled negative latent evidence, respectively. We use a multi-positive contrastive objective, where each positive contributes one InfoNCE [47] term:

exp q⊤ zj/τ

1 |M+i | z

, (8)

LContrast =

−log

zk∈M+∪M− exp q⊤ zk/τ

j∈M+

where τ is the temperature. Since both text and image evidence are projected into the same retrieval space, the same loss supports unified retrieval over mixed multimodal memory.

Generation distillation loss. To ensure latent tokens have similar roles compared to raw evidence, we add a distillation objective. For each training example, the teacher distribution is obtained by autoregressively running the frozen generator ϕ with the raw supporting context C+. This produces

a teacher sequence Atea = (atea1 ,...,atea|Atea|). The student uses the same frozen generator ϕ, but replaces the raw context with the projected latent memories z1,..., zk. We then minimize the token-level KL divergence along the teacher-generated trajectory:

|Atea|

KL P · | atea<t ,Q,C+),ϕ .∥P · | atea<t ,Q, z1,..., zk,ϕ . (9)

LDistill =

t=1

This term teaches the compressor and generator projection to produce latent tokens that the frozen generator can interpret as evidence. In this way, distillation loss connects the latent retrieval interface to the final QA objective without fine-tuning the large LLM/VLM generator.

- Table 1: Text-based QA results using Meta-Llama-3-8B-Instruct as the generation LLM. All methods use a frozen Meta-Llama-3-8B-Instruct generator. The Average columns report the out-of-domain average over 2WikiMultihopQA and MuSiQue. Bold indicates the best metric in each column, and underlining indicates the second-best one. R@k = Recall@k.

Generation LLM (fixed): Meta-Llama-3-8B-Instruct

In-Domain Out-of-Domain Dataset HotpotQA 2WikiMultihopQA MuSiQue Average Method EM F1 R@k #Tok EM F1 R@k #Tok EM F1 R@k #Tok EM F1 R@k #Tok Full Context 42.0 57.8 – 1462 17.7 39.2 – 1074 6.0 17.1 – 2580 11.9 28.2 – 1827 Evidence Compression Baselines

LLMLingua (20%) 31.8 44.8 – 283 17.0 30.4 – 199 11.6 21.8 – 492 14.3 26.1 – 346 LLMLingua (10%) 25.0 36.1 – 154 14.8 24.7 – 108 6.9 14.9 – 259 10.9 19.8 – 184 LLMLingua (5%) 20.9 30.2 – 87 15.0 22.2 – 63 4.3 10.6 – 137 9.7 16.4 – 100

RAG Baselines

- BM25 Retrieval (k=1) 32.5 44.8 30.2 68 17.2 27.2 18.3 60 6.2 14.2 7.0 69 11.7 20.7 12.7 65

- BM25 Retrieval (k=2) 36.8 49.9 45.5 106 16.3 28.3 29.2 94 8.0 16.7 12.2 106 12.2 22.5 20.7 100 BM25 Retrieval (k=5) 41.3 55.3 65.5 224 16.3 32.7 46.9 196 9.5 19.6 22.4 221 12.9 26.2 34.7 209

- Dense Retrieval (k=1) 29.4 41.0 27.9 67 19.0 30.0 23.3 61 7.8 17.1 9.6 70 13.4 23.6 16.5 66

- Dense Retrieval (k=2) 32.6 45.4 42.2 104 18.1 31.7 37.6 95 9.9 19.9 17.0 106 14.0 25.8 27.3 101 Dense Retrieval (k=5) 37.0 50.8 62.4 214 19.1 37.1 58.0 198 12.8 23.4 31.3 218 16.0 30.3 44.7 208

- Qwen3-Emb-0.6B (k=1) 30.9 43.6 33.2 70 18.1 30.9 32.5 66 8.1 18.2 10.3 73 13.1 24.6 21.4 70

- Qwen3-Emb-0.6B (k=2) 35.6 49.0 50.1 109 17.5 33.5 47.7 102 9.8 20.4 18.5 112 13.7 27.0 33.1 107 Qwen3-Emb-0.6B (k=5) 40.0 54.5 70.1 224 19.1 38.6 64.3 208 13.7 24.7 34.8 230 16.4 31.7 49.6 219 Ours: Latent Memory

- Latent Memory (k=1) 27.4 39.4 34.6 36 19.8 29.2 28.4 33 5.8 14.6 8.7 37 12.8 21.9 18.6 35

- Latent Memory (k=2) 31.6 45.2 62.6 45 21.5 33.5 49.5 42 7.3 16.9 15.5 46 14.4 25.2 32.5 44 Latent Memory (k=5) 34.8 48.9 87.1 72 24.3 36.7 74.2 69 8.7 19.2 30.1 73 16.5 28.0 52.2 71

Finally, the overall training objective is as follows, where λRecon, λContrast, and λDistill control the relative weights of reconstruction, retrieval, and distillation, respectively:

L = λReconLRecon + λContrastLContrast + λDistillLDistill, (10)

### 4 Experiments

We evaluate Latent Memory in two settings: (1) text-only contextual QA and (2) multimodal contextual QA. In the main text-only setting, we use a frozen Meta-Llama-3-8B-Instruct [48] generator ϕ and fine-tune LLaMA-3.2-1B-Instruct [48] with LoRA adapter as the compressor θ. To compare against text-only latent-context baselines that are built around Mistral, we additionally report a frozen Mistral-7B-Instruct [49] generator setting, where both the Latent Memory compressor/encoder and the reconstruction decoder are LLaMA-3.2-1B-Instruct. In the multimodal setting, we use a frozen LLaVA-1.5-13B [12] generator and fine-tune LLaVA-1.5-7B [12] with LoRA adapters as the compressor. We also include another setting with Gemma-3-12B-Instruct [13] generator and Gemma-3-4B [13] compressor for multimodal QA in Appendix C.6. For all reported settings, the reconstruction decoder π is fine-tuned from LLaMA-3.2-1B-Instruct. Experiments are done on a Nvidia H200 141GB GPU. Full training and evaluation details are deferred to Appendix B.

Dataset. In the text-only setting, the Latent Memory is trained on HotpotQA training dataset and evaluated on the validation set of in-domain HotpotQA and out-of-domain 2WikiMultihopQA, MuSiQue. To investigate a larger transfer, Appendix C.2 adds a generalization-on-more-domains suite that spans open-domain factoid QA and scientific-document QA. WebQA [50] is used for multimodal training and evaluation. In testing on its validation set, we report image-grounded (n=2,511) and textgrounded (n=2,455) subsets separately, while retrieval itself remains unified over a mixed text-image candidate pool. We also consider multimodal domain transfer on SlideQA in Appendix C.2. For the Text-only setting, all evidence is processed in the “Title: Sentence“ form. We process evidence as “Title: Evidence“ and “Caption: Image“ forms for the multimodal setting.

Baselines. We include two categories of baselines. (1) Context-based baselines, including generation with full-context and evidence-compression baselines LLMLingua [28], xRAG [32], and CLaRa [33]. (2) RAG baselines, including BM25 Retrieval [22], Dense Retrieval [21], Retrieval with practical off-the-shelf Qwen-3-Embedding [51] (and its variant fine-tuned on in-domain), Nemo Retriever [52], and Qwen-3-VL-Embedding [25]. All baselines and the proposed Latent Memory-based QA use the same pre-trained generator. Baseline details are in Appendix E.

- Table 2: Text-based QA results using Mistral-7B-Instruct as the generation LLM. The Latent Memory rows use LLaMA-3.2-1B-Instruct as both compressor/encoder and reconstruction decoder. xRAG and CLaRa use their pretrained Mistral-based checkpoints. The Average columns report the out-ofdomain average over 2WikiMultihopQA and MuSiQue.

Generation LLM (fixed): Mistral-7B-Instruct

In-Domain Out-of-Domain Dataset HotpotQA 2WikiMultihopQA MuSiQue Average Method EM F1 R@k #Tok EM F1 R@k #Tok EM F1 R@k #Tok EM F1 R@k #Tok Full Context 21.0 44.9 – 1701 3.8 27.1 – 1244 2.8 14.9 – 3012 3.3 21.0 – 2128 Evidence Compression Baselines LLMLingua (20%) 8.9 27.6 – 406 1.4 18.4 – 286 1.3 10.3 – 708 1.3 14.4 – 497 LLMLingua (10%) 6.8 22.6 – 217 0.6 15.1 – 149 0.9 7.4 – 371 0.8 11.2 – 260 LLMLingua (5%) 4.7 17.9 – 117 0.4 13.0 – 81 0.5 5.5 – 191 0.5 9.3 – 136 xRAG 12.1 25.3 – 42 3.5 17.6 – 38 1.4 9.5 – 42 2.5 13.6 – 40 CLaRa-16x (k=5) 5.2 15.3 55.8 119 5.1 16.2 26.5 115 0.7 6.0 22.3 119 2.9 11.1 24.4 117 RAG and Latent-context Baselines

- BM25 Retrieval (k=1) 11.0 29.6 30.2 76 0.4 14.4 18.3 66 0.9 8.1 6.9 75 0.6 11.3 12.6 71

- BM25 Retrieval (k=2) 13.8 33.8 45.5 120 0.7 15.8 29.2 104 1.7 9.5 12.1 119 1.2 12.6 20.7 111 BM25 Retrieval (k=5) 15.9 37.8 65.5 255 1.0 18.6 46.9 221 1.7 10.7 22.4 250 1.3 14.7 34.6 236

- Dense Retrieval (k=1) 9.0 26.4 27.9 75 0.6 15.8 23.3 67 1.0 10.2 9.5 77 0.8 13.0 16.4 72

- Dense Retrieval (k=2) 11.3 30.2 42.2 117 1.0 17.9 37.6 107 1.2 11.1 16.9 119 1.1 14.5 27.3 113 Dense Retrieval (k=5) 13.9 34.0 62.4 244 2.7 22.5 58.0 225 1.6 12.8 31.3 247 2.2 17.6 44.7 236

- Qwen3-Emb-0.6B (k=1) 10.3 28.4 33.2 79 0.7 17.1 32.5 74 1.0 10.6 10.3 81 0.9 13.8 21.4 77

- Qwen3-Emb-0.6B (k=2) 12.6 32.7 50.1 124 1.2 19.8 47.7 115 1.5 11.2 18.5 126 1.3 15.5 33.1 121 Qwen3-Emb-0.6B (k=5) 15.1 36.6 70.1 256 3.0 24.0 64.3 236 1.6 13.4 34.8 262 2.3 18.7 49.6 249

- Qwen3-Emb-0.6B-ft (k=1) 12.3 32.2 37.9 80 0.9 17.0 35.5 74 1.3 11.4 9.8 83 1.1 14.2 22.6 79

- Qwen3-Emb-0.6B-ft (k=2) 16.3 38.4 59.7 127 1.7 21.5 54.4 119 1.4 12.1 16.5 133 1.6 16.8 35.4 126 Qwen3-Emb-0.6B-ft (k=5) 18.6 42.5 80.5 267 3.8 25.9 71.3 247 2.0 14.3 30.8 280 2.9 20.1 51.0 264 Ours: Latent Memory

- Latent Memory (k=1) 12.3 30.4 34.8 36 1.8 20.6 27.6 32 2.3 11.8 8.4 36 2.1 16.2 18.0 34

- Latent Memory (k=2) 14.8 34.8 62.2 46 2.0 22.8 49.4 42 2.2 12.8 14.9 46 2.1 17.8 32.2 44 Latent Memory (k=5) 17.5 37.8 86.6 76 2.9 24.1 75.9 72 3.6 14.7 28.8 76 3.2 19.4 52.3 74

| | |
|---|---|
| | |
| | |
| | |
| | |
|Fu|ll|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

k=5

32.5

30.0

k=5

k=2

27.5

20%

- k=1

- k=2

TokenF1

k=1

25.0

k=5

22.5

k=2

10%

20.0

k=1

LLMLingua

BM25 Retrieval

17.5

5%

Dense Retrieval

Qwen3-Emb-0.6B

Latent Memory (1B, Ours)

15.0

50 100 200

Avg. Input Tokens to Generator 1827

(a) LLaMA-8B

22

| | |
|---|---|
| | |
|Fu|ll|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

xRAG

CLaRa-16x LLMLingua BM25 Retrieval

k=5 k=5

20

Dense Retrieval

- k=1

- k=2

Qwen3-Emb-0.6B

k=5

OODAverageF1

18

Qwen3-Emb-0.6B-ft

k=2

Latent Memory (1B, Ours)

16

k=2

20%

k=1

1 tok

14

k=1 k=5

12

10%

k=2

k=1

k=5

10

5%

40 100 250 500

2128

Avg. Input Tokens to Generator

(b) Mistral-7B-Instruct

- Figure 3: Text-only QA trade-off curves on the out-of-domain average F1 over 2WikiMultihopQA and MuSiQue.

Metrics. For text-only QA, we report Exact Match (EM), Token F1, Recall@k, and average generator input tokens (#Tok). For WebQA, we report F1, answer accuracy (Acc), Recall@k, and #Tok under the same unified retrieval setting. Acc follows the official WebQA evaluation protocol [50]. In all tables, #Tok reports the task-relevant generator-side prompt budget (including the context length and the query length) and excludes fixed chat-template scaffolding; in multimodal settings, it still reflects the effective tokenized cost after visual expansion.

##### 4.1 Text-only Setting

- Table 1 presents metrics of the proposed Latent Memory, context-based, and RAG baselines under a fixed 8B LLaMA generator. Qwen3-Emb-0.6B means using the Qwen3-Emedding-0.6B [51] for retrieval, which is a size similar to our 1B compressor. One-token Latent Memory shows the strongest out-of-domain average Recall@k among the reported text-only retrieval methods. As shown in Figure 3, Latent Memory achieves competitive EM/F1 performance with much fewer tokens. At k=5, the 1B model uses only 71 tokens on the out-of-domain average, about a third versus 209 for BM25

- Table 3: Multimodal QA (WebQA) with unified per-sample retrieval and unified generation over both images and text facts using a frozen LLaVA-1.5-13B generator.

Generation VLM (fixed): LLaVA-1.5-13B

Method WebQA-Image WebQA-Text Average

F1 Acc R@k #Tok F1 Acc R@k #Tok F1 Acc R@k #Tok Full Context 0.0 0.0 - 11655 6.0 10.0 - 8371 3.0 5.0 – 10013 RAG Baselines: Text-only Retrieval Baselines

- BM25 Retrieval (k=1) 14.5 21.5 21.8 295 39.9 42.7 31.8 133 27.2 32.1 26.8 214

- BM25 Retrieval (k=2) 19.2 23.4 28.3 476 43.9 48.4 51.0 234 31.6 35.9 39.6 355 BM25 Retrieval (k=5) 32.6 29.5 37.9 932 46.7 54.3 73.0 552 39.6 41.9 55.5 742

- Dense Retrieval (k=1) 18.9 24.5 39.7 476 34.9 40.4 25.5 264 26.9 32.4 32.6 370

- Dense Retrieval (k=2) 30.0 30.1 53.2 814 38.5 49.3 40.1 538 34.2 39.7 46.6 676 Dense Retrieval (k=5) 49.8 36.6 67.1 1645 37.6 59.5 62.0 1396 43.7 48.1 64.6 1520 RAG Baselines: Multimodal Retrieval Baselines

- Nemo-Emb-1B (k=1) 19.9 25.2 48.7 507 41.1 44.0 40.4 130 30.5 34.6 44.6 319

- Nemo-Emb-1B (k=2) 32.9 30.6 64.8 892 46.8 50.9 66.6 233 39.9 40.8 65.7 563 Nemo-Emb-1B (k=5) 53.0 37.2 81.4 1885 48.6 57.9 87.1 629 50.8 47.6 84.3 1257

- Qwen3-VL-Emb-8B (k=1) 15.1 22.0 24.1 284 40.8 43.6 40.7 131 28.0 32.8 32.4 208

- Qwen3-VL-Emb-8B (k=2) 20.1 24.5 32.9 465 46.1 50.2 66.4 235 33.1 37.3 49.6 350 Qwen3-VL-Emb-8B (k=5) 34.1 30.3 49.7 957 47.8 57.2 87.2 612 41.0 43.8 68.5 784 Ours: Generation with retrieving Latent Memory

- Latent Memory (k=1) 32.0 28.7 56.6 42 28.8 30.8 24.0 44 30.4 29.7 40.3 43

- Latent Memory (k=2) 56.5 39.5 74.7 52 30.0 32.8 41.1 54 43.2 36.2 57.9 53 Latent Memory (k=5) 69.4 44.2 91.2 82 30.7 34.3 70.5 84 50.0 39.2 80.8 83

and 208 for Dense. We discuss the time complexity in Appendix B.4. These results indicate the superiority of using the same latent representation space for both retrieval and generation evidence.

- Table 2 further evaluates the same text-only setting under a frozen Mistral-7B-Instruct generator, which allows direct comparison with pretrained Mistral-based latent-context baselines such as xRAG and CLaRa. In this setting, our Latent Memory still uses the LLaMA-3.2-1B encoder/compressor and decoder. As shown in Figure 3, Ours-1B at k=5 reaches the strongest out-of-domain Recall@k and competitive F1 while using 74 generator tokens, compared with 264 tokens for Qwen3-Emb-0.6B fine-tuned on HotpotQA at k=5.

Importantly, the retrieval behavior is not only an in-domain effect: the same HotpotQA-trained Latent Memory is evaluated without additional tuning on out-of-domain 2WikiMultihopQA and MuSiQue. Thus, the comparison tests the transfer of the learned latent interface rather than dataset-specific memorization. In Appendix C.2, we generalize the Latent Memory with a compressor trained on HotpotQA to four more datasets, where Latent Memory can still demonstrate a strong performanceefficiency trade-off. This competitive performance may be partly explained by its stronger retrieval behavior. Latent Memory reaches an out-of-domain average Recall@k of 52.2 at k = 5.

4.2 Multimodal Setting

|k=5| |k=5| |
|---|---|---|---|
| | | | |
| | | | |
|k=2|k=5|k=5| |
| || |
|---|
<br><br>k=2| | |
| | | | |
| |k=2| | |
| || |
|---|
<br><br>k=2| | |
|k=1|k=1 k=1<br><br>|BM25 Retrieval<br><br>Dense Retrieval| |
| | |NemoRetriever-1B| |
| |k=1<br><br>|Qwen3-VL-Emb-8B Latent Memory (7B, Ours)| |
| | | | |

102 103

25

30

35

40

45

50

AverageF1

|Full-cont token consu|ext mptio|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

104

n

Avg. Input Tokens to Generator

Figure 4: LLaVA-based multimodal WebQA trade-off curves across k∈{1,2,5}.

We report the Latent Memory and baselines on the WebQA benchmark, which requires unified retrieval and multi-hop reasoning over a multimodal candidate pool. For BM25 and Dense retrieval, we retrieve only the caption for the image evidence.

- Table 3 shows the image-grounded (n=2,511) and text-grounded (n=2,455) WebQA subsets separately. One-token Latent Memory is strongest on the image-grounded subset while using far fewer generator tokens than raw-evidence retrieval baselines: at k=5, it reaches 69.4 image F1 with only 82 tokens, compared with 53.0 for Nemo-Emb at 1885 tokens. On the full benchmark, Latent Memory gives a competitive average F1 with a much smaller token budget, while Nemo-Emb gives the best average F1/Acc at substantially higher cost.

Similar to the text-only setting, this behavior can be attributed to the fact that the unified representation of Latent Memory leads to a better Recall@K (10%+ higher compared to Dense Retrieval) on textgrounded and especially image-grounded questions. Moreover, as another reason, Raw-image prompting may exceed the pretrained context window for the generator, leading to poor-quality output (full-context often outputs meaningless content or blanks). To intuitively reflect the generation process augmented with Latent Memory, we provide a case study in Section 5.

### 5 Discussion

Ablations and Analysis. Table 4 conducts an ablation study over the core reconstruction loss LRecon, where the results are averaged over HotpotQA, 2WikiMultihopQA, and MuSiQue. Removing reconstruction lowers both answer quality and retrieval accuracy, with a larger drop in EM/F1 than in Recall@k. Removing negative evidence from reconstruction also hurts retrieval and generation, supporting the view that negative evidence helps anchor the unified latent representation space. The full per-dataset breakdown and more ablation variants are reported in Appendix C.3 to C.5.

- Table 4: Ablation on reconstruction. Colored subscripts indicate the gap to the default model.

k

Original Latent Memory w/o Reconstruction Loss LRecon w/o LRecon on Negative Evidence M− EM F1 R@k EM F1 R@k EM F1 R@k

- k=1 17.7 27.7 23.9 16.5−1.1 27.2−0.5 23.8−0.1 16.3−1.3 28.1+0.3 23.2−0.6
- k=2 20.1 31.9 42.6 19.9−0.2 31.0−0.9 41.8−0.7 18.4−1.7 32.0+0.2 39.6−3.0 k=5 22.6 34.9 63.8 20.9−1.7 32.7−2.2 62.9−0.9 20.8−1.9 33.8−1.2 62.3−1.5

Better Latent Memory Capability with more Token Budget. One-token Latent Memory already gives a strong efficiency–quality trade-off, and allocating more latent tokens per evidence item can improve the quality. As summarized in Table 5, on the out-of-domain average over 2WikiMultihopQA and MuSiQue, upgrading to 8-token Latent Memory improves EM/F1 enough to surpass the strongest text-only RAG baseline (i.e., Qwen-3-Emb-0.6B) at each k, while still using fewer generator tokens. The gain mainly comes from stronger generation rather than retrieval, since Recall@k changes only modestly. Full in-domain and out-of-domain results are in Appendix C.1.

- Table 5: Token-count ablation summary. We show the average Out-of-domain (2WikiMultihopQA and MuSiQue results). RAG* denotes the strongest RAG baseline at the same k.

RAG* 1-token Latent Memory 8-token Latent Memory ∆ (8-token − 1-token) EM F1 R@k #Tok EM F1 R@k #Tok EM F1 R@k #Tok EM F1 R@k #Tok

k

- k=1 13.1 24.6 21.4 70 12.8−0.3 21.9−2.7 18.6 35−35 14.4+1.3 24.7+0.1 19.9 42−28 +1.6 +2.8 +1.3 +7

- k=2 13.7 27.0 33.1 107 14.4+0.7 25.2−1.8 32.5 44−63 17.3+3.6 29.0+2.0 34.5 58−49 +2.9 +3.8 +2.0 +14 k=5 16.4 31.7 49.6 219 16.5+0.1 28.0−3.7 52.2 71−148 19.9+3.5 32.5+0.9 53.7 106−113 +3.4 +4.6 +1.5 +35

A Representative Case Study. Figure 5 illustrates how Latent Memory behaves on an imagegrounded WebQA example. The retrieved latent evidence supports the correct answer and preserves the counting information required by the question. The optional reconstruction further shows that the latent token retains interpretable semantic content for both text and image evidence, rather than acting only as an opaque retrieval identifier.

Limitation on Current Modality Coverage and Future Directions. The current design assumes that evidence can be decomposed into atomic text or image units, and each unit can be compressed and retrieved independently. This assumption is reasonable for WebQA-style mixed evidence, where answers are often grounded in a small number of facts or images. However, it becomes limiting when the input meaning depends on the global structure. Complex tables require row-column relations and layout information; long videos require temporal ordering; and document pages may require spatial relations between captions, figures, and surrounding text. Compressing such inputs into isolated latent tokens may preserve local semantics but lose these structural dependencies. A natural next step is therefore to augment Latent Memory with structural axes such as position, layout, and time, so that retrieval and generation can operate over both local evidence semantics and global organization.

Multimodal Context Memory

[Figure 94]

[Figure 95]

Image:

Text: Window shopping: London's Burlington Arcade, which opened in 1819, positioned itself as an elegant and exclusive ven...

###### Image:

Text: Burlington Arcade: The present tenants include a range of clothing, footwear and accessory shops, art and antique dealers, and the jewellers and dea...

.....

Caption: View of the current Heming store at London's Burlington Arcade

Caption: Craven Park, Barrow

Text: .....

[Figure 96]

Input

###### Retrievable Latent Memory

Output

① Latent Memory Construction

LLM/VLM with Trainable LoRA for Compression

[Figure 97]

[Figure 98]

Output

[Figure 99]

Question: "How many display windows can be found on the front of the Heming store at London's Burlington Arcade?"

Frozen LargerSize Generation

Three

[Figure 100]

Input

[Figure 101]

② Retrieve based on Query

LLM/VLM

Latent context 1: 1

③ Generation with Latent Tokens

Reconstructed Text Context in −

Reconstructed Context (Negative)

Reconstructed Context (Positive)

[Figure 102]

[Figure 103]

[Figure 104]

Image:

###### Image:

Input

Text: Window shopping: London...... Text: Burlington Arcade: tenants..... Text: Heming (company): H..... (CE = 0.6)

LLM with Trainable LoRA/ MLP for Decoding

[Figure 105]

Output

[Figure 106]

④ Latent Token Reconstruction (Optional)

Title: View of.. (CE = 0.7)

Title: Craven.. (CE = 0.8)

Figure 5: LLaVA-1.5-based Latent Memory on an image-grounded WebQA question. The figure consists of four parts. ① Compressing multimodal evidence forms a unified Latent Memory. ② The retrieval process aligns the query embedding with the latent token with the positive image, while irrelevant candidates are pushed away in the unified latent space. ③ The Latent Memory grounded QA preserves the counting ability. ④ The optional reconstruction for both image and text evidence.

### 6 Conclusion

We introduced Latent Memory, a novel memory paradigm that compiles each evidence item into one latent token, retrieves these latent memories by compressing the query, and feeds them directly to a frozen generator for QA. On text QA, it achieves competitive EM/F1 performance compared to RAG baselines with only 3× fewer tokens. On multimodal WebQA, it is especially effective for image-grounded questions and delivers the strongest average-F1 trade-off while sharply reducing generator cost by up to 10×. Latent Memory improves the Recall@k by employing a unified latent representation and prevents the context from exceeding the generator’s context window length.

Future Work Overall, Latent Memory provides an efficient alternative to the current token-level memory paradigm. It is promising for use in scenarios that require fast response and low storage pressure, such as edge devices and other resource-intensive scenarios. In the future, as discussed in Section 5, we will extend the Latent Memory to more modalities, including complex tables and complex videos. This paper focuses on external evidence, so we do not consider Agentic Memory [10, 53, 54], which are generated by models themselves. Future works also include extending the Latent Memory to resource-contained agents scenarios for better Agentic Memory comprehension.

### References

- [1] Murong Yue. A survey of large language model agents for question answering. arXiv preprint arXiv:2503.19213, 2025.
- [2] Yu-Hsuan Lin, Qian-Hui Chen, Yi-Jie Cheng, Jia-Ren Zhang, Yi-Hung Liu, Liang-Yu Hsia, and Yun-Nung Chen. Llm inference enhanced by external knowledge: A survey. arXiv preprint arXiv:2505.24377, 2025.
- [3] Alphaeus Dmonte, Roland Oruche, Marcos Zampieri, Prasad Calyam, and Isabelle Augenstein. Claim verification in the age of large language models: A survey. arXiv preprint arXiv:2408.14317, 2024.
- [4] Zhi Zheng and Wee Sun Lee. Reasoning-cv: Fine-tuning powerful reasoning llms for knowledge-assisted claim verification. arXiv preprint arXiv:2505.12348, 2025.

- [5] Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng Guo, Ruiming Tang, and Yong Liu. From human memory to ai memory: A survey on memory mechanisms in the era of llms. arXiv preprint arXiv:2504.15965, 2025.
- [6] Shangyu Wu, Ying Xiong, Yufei Cui, Haolun Wu, Can Chen, Ye Yuan, Lianming Huang, Xue Liu, Tei-Wei Kuo, Nan Guan, et al. Retrieval-augmented generation for natural language processing: A survey. arXiv preprint arXiv:2407.13193, 2024.
- [7] Xavier Daull, Patrice Bellot, Emmanuel Bruno, Vincent Martin, and Elisabeth Murisasco. Complex qa and language models hybrid architectures, survey. arXiv preprint arXiv:2302.09051, 2023.
- [8] Xiaofei Dong, Xueqiang Zhang, Weixin Bu, Dan Zhang, and Feng Cao. A survey of llm-based agents: Theories, technologies, applications and suggestions. In 2024 3rd International Conference on Artificial Intelligence, Internet of Things and Cloud Computing Technology (AIoTC), pages 407–413. IEEE, 2024.
- [9] Zhi Zheng, Zhuoliang Xie, Zhenkun Wang, and Bryan Hooi. Monte carlo tree search for comprehensive exploration in llm-based automatic heuristic design. arXiv preprint arXiv:2501.08603, 2025.
- [10] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.
- [11] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? In International Conference on Learning Representations, volume 2024, pages 54107–54157, 2024.
- [12] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [13] Gemma Team. Gemma 3 technical report. ArXiv, abs/2503.19786, 2025.
- [14] Taehwan Park, Geonho Lee, and Min-Soo Kim. Mobilerag: A fast, memory-efficient, and energy-efficient method for on-device rag. arXiv preprint arXiv:2507.01079, 2025.
- [15] Onur Mutlu, Ataberk Olgun, and ˙Ismail Emir Yüksel. Memory-centric computing: solving computing’s memory problem. In 2025 IEEE International Memory Workshop (IMW), pages 1–4. IEEE, 2025.
- [16] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.
- [17] Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun Wang, Chenchen Zhang, Ge Zhang, Jiebin Zhang, et al. A comprehensive survey on long context language modeling. arXiv preprint arXiv:2503.17407, 2025.
- [18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.
- [19] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In International conference on machine learning, pages 3929–3938. PMLR, 2020.
- [20] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane DwivediYu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented language models. Journal of Machine Learning Research, 24(251):1–43, 2023.
- [21] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 6769–6781, 2020.
- [22] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond, volume 4. Now Publishers Inc, 2009.
- [23] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrievalaugmented generation. 2024.
- [24] Muhammad Arslan, Hussam Ghanem, Saba Munawar, and Christophe Cruz. A survey on rag with llms. Procedia computer science, 246:3781–3790, 2024.

- [25] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. Qwen3-vl-embedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026.
- [26] Haonan Jiang, Yuji Wang, Yongjie Zhu, Xin Lu, Wenyu Qin, Meng Wang, Pengfei Wan, and Yansong Tang. Embed-rl: Reinforcement learning for reasoning-driven multimodal embeddings. arXiv preprint arXiv:2602.13823, 2026.
- [27] Chenwei He, Xiangzhao Hao, Tianyu Yang, Yuxiang Ma, Yuheng Jia, Lingxiang Wu, Chaoyang Zhao, Haiyun Guo, and Jinqiao Wang. Plume: Latent reasoning based universal multimodal embedding. arXiv preprint arXiv:2604.02073, 2026.
- [28] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Llmlingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 13358–13376, 2023.
- [29] Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. Adapting language models to compress contexts. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3829–3846, 2023.
- [30] Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. In-context autoencoder for context compression in a large language model. arXiv preprint arXiv:2307.06945, 2023.
- [31] Zeju Li, Yizhou Zhou, and Qiang Xu. Latent context compilation: Distilling long context into compact portable memory. arXiv preprint arXiv:2602.21221, 2026.
- [32] Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, and Dongyan Zhao. xrag: Extreme context compression for retrieval-augmented generation with one token. Advances in Neural Information Processing Systems, 37:109487–109516, 2024.
- [33] Jie He, Richard He Bai, Sinead Williamson, Jeff Z Pan, Navdeep Jaitly, and Yizhe Zhang. Clara: Bridging retrieval and generation with continuous latent reasoning. arXiv preprint arXiv:2511.18659, 2025.
- [34] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.
- [35] Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. Codi: Compressing chain-of-thought into continuous space via self-distillation. arXiv preprint arXiv:2502.21074, 2025.
- [36] Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, Xipeng Qiu, and Dahua Lin. Sim-cot: Supervised implicit chain-of-thought. arXiv preprint arXiv:2509.20317, 2025.
- [37] Zhi Zheng and Wee Sun Lee. Beyond imitation: Reinforcement learning for active latent planning. arXiv preprint arXiv:2601.21598, 2026.
- [38] Muxin Fu, Xiangyuan Xue, Yafu Li, Zefeng He, Siyuan Huang, Xiaoye Qu, Yu Cheng, and Yang Yang. Latentmem: Customizing latent memory for multi-agent systems. arXiv preprint arXiv:2602.03036, 2026.
- [39] Guibin Zhang, Muxin Fu, and Shuicheng Yan. Memgen: Weaving generative latent memory for selfevolving agents. arXiv preprint arXiv:2509.24704, 2025.
- [40] Xinlei Yu, Chengming Xu, Guibin Zhang, Zhangquan Chen, Yudong Zhang, Yongbo He, Peng-Tao Jiang, Jiangning Zhang, Xiaobin Hu, and Shuicheng Yan. Vismem: Latent vision memory unlocks potential of vision-language models. arXiv preprint arXiv:2511.11007, 2025.
- [41] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15619–15629, 2023.
- [42] Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, and Pascale Fung. Vl-jepa: Joint embedding predictive architecture for vision-language. arXiv preprint arXiv:2512.10942, 2025.
- [43] Heejeong Nam, Quentin Le Lidec, Lucas Maes, Yann LeCun, and Randall Balestriero. Causal-jepa: Learning world models through object-level latent interventions. arXiv preprint arXiv:2602.11389, 2026.

- [44] Jingwen Sun, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, and Zhibo Chen. Vla-jepa: Enhancing vision-language-action model with latent world model. arXiv preprint arXiv:2602.10098, 2026.
- [45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [46] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- [47] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.
- [48] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [49] Yihang Jiang, Xiaoyang Li, Guangxu Zhu, Hang Li, Jing Deng, Kaifeng Han, Chao Shen, Qingjiang Shi, and Rui Zhang. 6g non-terrestrial networks enabled low-altitude economy: Opportunities and challenges. arXiv preprint arXiv:2311.09047, 2023.
- [50] Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. Webqa: Multihop and multimodal qa. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16495–16504, 2022.
- [51] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.
- [52] Mengyao Xu, Gabriel Moreira, Ronay Ak, Radek Osmulski, Yauhen Babakhin, Zhiding Yu, Benedikt Schifferer, and Even Oldridge. Llama nemoretriever colembed: Top-performing text-image retrieval model. arXiv preprint arXiv:2507.05513, 2025.
- [53] Yu Wang and Xi Chen. Mirix: Multi-agent memory system for llm-based agents. arXiv preprint arXiv:2507.07957, 2025.
- [54] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.
- [55] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. arXiv preprint arXiv:2410.10594, 2024.
- [56] Maxime Louis, Hervé Déjean, and Stéphane Clinchant. Pisco: Pretty simple compression for retrievalaugmented generation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 15506–15521, 2025.
- [57] Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and Jinsong Su. Ume-r1: Exploring reasoning-driven generative multimodal embeddings. arXiv preprint arXiv:2511.00405, 2026.
- [58] Rui-Jie Zhu, Tianhao Peng, Tianhao Cheng, Xingwei Qu, Jinfa Huang, Dawei Zhu, Hao Wang, Kaiwen Xue, Xuanliang Zhang, Yong Shan, et al. A survey on latent reasoning. arXiv preprint arXiv:2507.06203, 2025.
- [59] Xinghao Chen, Anhao Zhao, Heming Xia, Xuan Lu, Hanlin Wang, Yanjun Chen, Wei Zhang, Jian Wang, Wenjie Li, and Xiaoyu Shen. Reasoning beyond language: A comprehensive survey on latent chain-of-thought reasoning. arXiv preprint arXiv:2505.16782, 2025.
- [60] Yue Liu, Jiaying Wu, Yufei He, Ruihan Gong, Jun Xia, Liang Li, Hongcheng Gao, Hongyu Chen, Baolong Bi, Jiaheng Zhang, et al. Efficient inference for large reasoning models: A survey. arXiv preprint arXiv:2503.23077, 2025.
- [61] Xinlei Yu, Zhangquan Chen, Yongbo He, Tianyu Fu, Cheng Yang, Chengming Xu, Yue Ma, Xiaobin Hu, Zhe Cao, Jie Xu, et al. The latent space: Foundation, evolution, mechanism, ability, and outlook. arXiv preprint arXiv:2604.02029, 2026.

- [62] Qixun Wang, Yang Shi, Yifei Wang, Yuanxing Zhang, Pengfei Wan, Kun Gai, Xianghua Ying, and Yisen Wang. Monet: Reasoning in latent visual space beyond images and language. arXiv preprint arXiv:2511.21395, 2025.
- [63] Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chenyang Zhao, Shuohang Wang, Yelong Shen, and Xin Eric Wang. Soft thinking: Unlocking the reasoning potential of llms in continuous concept space. arXiv preprint arXiv:2505.15778, 2025.
- [64] Natasha Butt, Ariel Kwiatkowski, Ismail Labiad, Julia Kempe, and Yann Ollivier. Soft tokens, hard truths. arXiv preprint arXiv:2509.19170, 2025.
- [65] Chünhung Wu, Jinliang Lu, Zixuan Ren, Gangqiang Hu, Zhi Wu, Dai Dai, and Hua Wu. Llms are single-threaded reasoners: Demystifying the working mechanism of soft thinking. arXiv preprint arXiv:2508.03440, 2025.
- [66] Zhi Zheng and Wee Sun Lee. Soft-grpo: Surpassing discrete-token llm reinforcement learning via gumbel-reparameterized soft-thinking policy optimization. arXiv preprint arXiv:2511.06411, 2025.
- [67] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380, 2018.
- [68] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625, 2020.
- [69] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.
- [70] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, 2019.
- [71] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, 2017.
- [72] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4599–4610, 2021.
- [73] Ryo Kamoi, Tanya Goyal, Juan Diego Rodriguez, and Greg Durrett. Wice: Real-world entailment for claims in wikipedia. arXiv preprint arXiv:2303.01432, 2023.
- [74] Ziyan Jiang, Xueguang Ma, and Wenhu Chen. Longrag: Enhancing retrieval-augmented generation with long-context llms. arXiv preprint arXiv:2406.15319, 2024.

### Appendix Contents

- 1. Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- (a) RAG and Embedding Retrieval . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- (b) Evidence Compression for Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- (c) Retrieval-Compression Interaction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- (d) Latent-Space Modeling for LLMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .17

- 2. Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- (a) Prompt Templates . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .18
- (b) Detailed Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .19
- (c) Hyperparameter Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .20
- (d) Time Complexity Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- (e) Space Complexity Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- 3. Additional Experiments and Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- (a) Token Count Ablation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- (b) Generalization Ability on More Domains . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .26
- (c) Ablation on Core Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .27
- (d) Ablation on Stronger Text Compressors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- (e) Direct Transfer to Similar Generator . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- (f) Multimodal Results with Gemma . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- (g) Latent Tokens as Retrievers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30

- 4. Case Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

- (a) Reconstruction Quality of Latent Tokens . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- (b) Text-only Case Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- (c) More Multimodal QA Case Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34

- 5. Baselines, Datasets, and Licenses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35

### A Related Work

Table 6: Capability comparison with representative related work. Multimodal (Text & Image) discusses whether the method can adapt to multimodal documents or evidence. Efficient generation means whether this method aims at reducing the generator token consumption. Unified generation & retrieval means that the same compressed representation is used both for retrieval and as the object consumed by the generator (✓ making it unnecessary to work separately on compression and retrieval, xRAG (△) using the same retrieved embeddings as generator input at inference, but relies on a separately trained bridge rather than unified retrieval-generation training). No need to fine-tune generator LLM/VLM means the memory item can be directly fed into the generator without requiring it to comprehend. (✓ making it easier to deploy and avoid catastrophic forgetting)

Unified generation & retrieval

Multimodal (Text & Image)

Efficient generation

No need to fine-tune generator LLM/VLM

Representative work

- (1) RAG baselines: Raw-evidence RAG BM25 / Dense RAG /

Qwen3-Embedding [22, 21, 51]

✗ ✗ ✗ ✓

VisRAG / Qwen3-VL-Embedding / Nemo Retriever [55, 25, 52]

✓ ✗ ✗ ✓

- (2) Compression-based baselines: Evidence compression for generation LLMLingua / ICAE [28, 30] ✗ ✓ ✗ ✓ AutoCompressor [29] ✗ ✓ ✗ ✗ LCC / PISCO [31, 56] ✗ ✓ ✗ ✗

(1) & (2) Interaction baselines: Embedding-based retrieval then latent-context generation

xRAG [32] ✗ ✓ △ ✓ CLaRa [33] ✗ ✓ ✓ ✗

Ours: Compiling Latent Memory for retrieval & generation Latent Memory ✓ ✓ ✓ ✓

In this section, I will discuss works that are related to the proposed Latent Memory paradigm, as well as a broad range of works that have a similar latent-representation idea. Table 6 presents a comprehensive collection of existing related works.

##### A.1 RAG and Embedding Retrieval

BM25-based sparse RAG. Retrieval-augmented generation selects a small subset of evidence and then lets a pretrained generator answer from the retrieved content [18, 20]. BM25 is the classical sparse retrieval baseline, relying on lexical matching and term statistics rather than learned semantic representations [22]. It is simple and directly compatible with frozen LLMs, but the generator still consumes raw retrieved text, so the retrieval step does not reduce the per-evidence token cost of generation.

Text-only Dense retrieval and embedding models. Dense retrieval learns a neural text-embedding space for query-evidence matching, as in DPR-style retrieval [21]. As the most recent embedding model for dense retrieval, Qwen3-Embedding strengthens this retrieval front end with a modern text embedding and reranking model [51]. These methods improve semantic retrieval over textual evidence, but the representation is still used primarily as a retrieval key; after retrieval, the generator receives raw text rather than the embedding itself. Moreover, these works cannot generalize well to multimodal settings, especially on images with details that cannot be accurately captioned.

VisRAG and multimodal embedding retrieval. Multimodal RAG extends retrieval from text-only corpora to mixed text-image evidence. VisRAG is representative of raw-evidence multimodal RAG: it retrieves relevant visual documents but still hands visual inputs to the generator [55]. As pretrained embedding model for multimodal RAG, Qwen3-VL-Embedding provides a unified vision-language embedding and reranking framework for text-image retrieval [25]. Nemo Retriever is another strong text-image retrieval model used as a retrieval front end [52].

Recent reasoning-aware multimodal embedding methods further refine what the retrieval vector should encode: Embed-RL optimizes multimodal embeddings through reinforcement learning signals [26], UME-R1 explores reasoning-driven generative multimodal embeddings [57], and PLUME uses latent reasoning for universal multimodal embedding [27]. As summarized in Table 6, these

methods support multimodal retrieval and can be used with pretrained VLM generators, but they remain raw-evidence RAG: retrieved images or text are still passed to the generator in their native input form. Thus, a retrieved image can still expand into many visual tokens, keeping generation expensive even when retrieval is accurate.

##### A.2 Evidence Compression for Generation

Discrete prompt compression. Evidence-compression methods reduce the context consumed by the generator after the evidence has already been chosen. LLMLingua prunes or rewrites discrete prompt tokens, preserving compatibility with pretrained LLMs while shortening the input [28]. This improves generator-side efficiency, but it does not build a retrievable memory: the shortened context is produced for the current prompt rather than stored as a corpus item for future retrieval.

Latent or compact context compression. ICAE compresses context into learned memory-like latent states for language modeling [30]. AutoCompressor trains models to summarize previous context into compact summary vectors that can condition later generation [29]. LCC studies latent context compression for reducing long-context generation cost [31]. PISCO similarly targets compact softcontext representations for efficient generation [56]. These methods correspond to the second block of Table 6: they focus on efficient generation, but they generally do not retrieve over the compressed evidence representation. So, because there is no retrieval system, these methods can exceed the generation context window when the irrelevant context is very long, resulting in meaningless output. Several methods also require the generator side to be trained or adapted to understand the compressed tokens, so compression and retrieval remain separate problems.

##### A.3 Retrieval-Compression Interaction

These are also works that combine the idea of retrieval and compression.

xRAG. xRAG first presents the idea of using a unified representation for generation & retrieval. However, as we mark it with △ for unified generation and retrieval in Table 6. xRAG relies on training a bridge based on pre-trained retrieval embeddings; noting that there are representations being important for generation but meaningless for retrieval (e.g., some details), the two-stage process makes it unable to find the optimal and unified generation & retrieval representation.

CLaRa. The most recent work, CLaRa, is closer to unified latent-context retrieval and generation because its latent context can be retrieved and then used for generation [33]. Both CLaRa and xRAG have the same limitation: it is text-only. Moreover, CLaRa requires a large amount of pre-training effort, and the latent representation in CLaRa is processed to fine-tune LLM/VLM Latent Memory, which might lead to catastrophic forgetting.

Latent Memory vesus xRAG & CLaRa.

Compared to the two methods mentioned above, the Latent Memory paradigm extends the text-only problem to multimodal scenarios. The success of Latent Memory in multimodal scenarios not only improves generation token efficiency but also reduces storage pressure, which cannot be achieved by pure text-only Latent Memory. We provide a more detailed description in Appendix B.5, showing that saving text-based data does not significantly increase storage pressure compared to Latent Memory; on the contrary, high-dimensional vectors are often more difficult to store. However, for images, Latent Memory in LLaVA scenarios (4096-dimensional bf16 latent token) is more efficient than an uncompressed RGB image once the image is larger than roughly 53 × 53 pixels.

##### A.4 Latent-Space Modeling for LLMs

Latent reasoning and hidden-state computation. Latent Memory is inspired by a broader direction of latent-space representation [58, 59, 60, 61]. Previous work shows that continuous hidden states can carry useful information in a wide collection of applications, including latent chain-of-thought math reasoning [34, 35, 36, 37, 62, 40], soft-thinking reasoning [63, 64, 65, 66], and agentic communication in multi-turn / multi-agent [38, 39]. It is worth noting that although some works also use a similar name, Memory [38, 39, 40], we are the first to use latent tokens to save storage and generation token consumption in multimodal contextual QA. Besides, some studies seek a unified latent representation space for multimodal understanding and generation [42, 44, 43]. These studies motivate the idea that hidden-state vectors have the ability to carry unified representations for contextual memory.

### B Implementation Details

##### B.1 Prompt Templates

Teacher and student use the same system-user-assistant scaffold. Across text-only and LLaVA pipelines, the instruction is: “You are a helpful assistant. Answer the question concisely (a few words or a short phrase) based on the provided context.” The only difference is the evidence representation: the teacher sees raw evidence, while the student receives one latent token for each retrieved evidence item in the same retrieved order.

Text-only teacher prompt

[system] You are a helpful assistant. Answer the question concisely (a few words or a short phrase) based on the provided context. [user]

- Context 1: <retrieved fact 1>
- Context 2: <retrieved fact 2>

... Context k: <retrieved fact k> Question: <question> [assistant] Answer:

Text-only student prompt

[system] You are a helpful assistant. Answer the question concisely (a few words or a short phrase) based on the provided context. [user]

- Latent context 1: [LATENT]
- Latent context 2: [LATENT]

... Latent context k: [LATENT] Question: <question> [assistant] Answer:

Multimodal teacher prompt

[system] You are a helpful assistant. Answer the question concisely (a few words or a short phrase) based on the provided context. [user]

- Context 1: <retrieved text fact 1>
- Context 2: <image> Title: <retrieved image title/caption 2>

... Question: <question> [assistant] Answer:

Multimodal student prompt

[system] You are a helpful assistant. Answer the question concisely (a few words or a short phrase) based on the provided context. [user]

- Latent context 1: [LATENT]
- Latent context 2: [LATENT]

... Latent context k: [LATENT] Question: <question> [assistant] Answer:

##### B.2 Detailed Pipeline

Retrieval projection MLP. The retrieval head maps the compressor hidden state into a 512dimensional retrieval space. In all reported text-only and multimodal runs, it is implemented as

retrieval_proj(z) = ℓ2 (LayerNorm(Linear(z))),

where the linear layer maps dθ → dr and dr = 512 in all reported runs. The ℓ2 normalization is applied before FAISS storage and query-time similarity search, so inner product retrieval is equivalent to cosine similarity.

Generator projection MLP. The generator-side projector converts a retrieved Latent Memory vector into the frozen generator hidden dimension. It is a two-layer MLP with LayerNorm:

cross_proj(z) = LayerNorm(W2 GELU(W1z)).

For the text-only setting, it maps the Llama-3.2-1B hidden size dθ = 2048 to the Meta-Llama-3-8B generator hidden size dϕ = 4096, with intermediate dimension (2048 + 4096)/2 = 3072. For multimodal LLaVA, it maps the LLaVA-7B compressor hidden size dθ = 4096 to the LLaVA-13B generator hidden size dϕ = 5120, with intermediate dimension (4096 + 5120)/2 = 4608.

Decoder and image-reconstruction MLPs. For textual reconstruction, the projected latent token is inserted into the decoding prompt from Appendix B.1, and the decoder π is trained to autoregressively recover the original text evidence or image caption. The decoder projector maps compressor hidden states into the lightweight decoder hidden space with Linear(dθ,dπ) → LayerNorm, where dπ denotes the hidden dimension of the decoder π. For image evidence, we do not reconstruct raw pixels. Instead, the image embedding reconstruction head predicts the frozen CLIP CLS hidden state. This MLP is

img_embed_decode_proj(z) = W2 LayerNorm(GELU(W1z)),

where W1 maps dθ to (dθ + dv)/2 and W2 maps to dv = 1024 for the CLIP target used in our runs. For LLaVA-7B, this midpoint is (4096 + 1024)/2 = 2560.

Online question answering. At inference time, the query is encoded with the query adapter and projected into the same normalized retrieval space. FAISS returns the top-k memory entries. Their full latent vectors are ordered by retrieval score, projected with cross_proj, and inserted into the frozen generator through inputs_embeds. Ordinary prompt tokens are embedded with the generator embedding layer; projected latent memories are spliced between the prompt prefix and suffix. No generator weights are updated during training or evaluation.

Trainable components. The text model uses separate LoRA adapters for compress, query, decode, and query_decode. The multimodal model additionally uses image_decode. Encoderside adapters target q_proj, k_proj, v_proj, and o_proj. Decoder-side adapters also include gate_proj and up_proj. The trainable non-LoRA components are the [MEM] token, retrieval projector, generator projector, decoder projector, and image embedding reconstruction projector.

Training-Loss Distillation. The teacher is the frozen generator prompted with raw positive evidence, while the student uses the same frozen generator prompted with projected latent memories. For text-side distillation, the student latent context is randomly augmented with 0–3 sampled hardnegative latent memories during training, while the teacher prompt remains positive-only. This augmentation exposes the student generator to small retrieval noise without changing the teacher target distribution; the KL loss is still computed on the first 16 generated answer tokens.

##### B.3 Hyperparameter Settings

Table 7: Main hyperparameters and LoRA configuration used in the reported runs.

Component Text-only Multimodal Backbone / generator Llama-3.2-1B-Instruct compressor →

LLaVA-1.5-7B → frozen LLaVA-1.513B

frozen Meta-Llama-3-8B-Instruct generator

Compression / latent dimension dθ

2048 for Llama-3.2-1B; one latent token per evidence item in the main setting

4096 for LLaVA-1.5-7B; one latent token per text or image evidence item

Retrieval dimension dr 512-dimensional normalized retrieval vector for FAISS inner-product search

512-dimensional normalized retrieval vector shared by text and image evidence

Generation dimension dϕ 4096 for the frozen Meta-Llama-3-8B generator; cross_proj: 2048 → 3072 → 4096

5120 for frozen LLaVA-1.5-13B with cross_proj: 4096 → 4608 → 5120

Text reconstruction decoder Llama-3.2-1B-Instruct Llama-3.2-1B-Instruct for reported WebQA runs

Decoder / reconstruction dimension dπ

2048 decoder hidden size; decode_proj: 2048 → 2048

2048 LLaMA decoder hidden size for reported WebQA runs; LLaVA decode_proj: 4096 → 2048; image embedding target dv=1024

Encoder-side LoRA targets q_proj, k_proj, v_proj, o_proj q_proj, k_proj, v_proj, o_proj Decoder-side LoRA targets q_proj, k_proj, v_proj, o_proj,

q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj

gate_proj, up_proj

LoRA rank / alpha / dropout r=64, α=128, 0.05 r=64, α=128, 0.05 Optimizer and LR AdamW, peak LR 1×10−4 AdamW, peak LR 1×10−4 Training batching batch size 8, gradient accumulation 4 batch size 8, gradient accumulation 2 Training epochs 3 epochs (about 20 hours) 2 epochs (about 30 hours) Text loss weights λrecon=0.5, λcontrast=0.2, λdistill=1.0 λrecon=0.5, λcontrast=0.2, λdistill=1.0 Image loss weights – λimage-contrast=0.2, λimage-distill=2.0 Embedding reconstruction – λembed-recon=5.0 Hard negatives per sample up to 8 text negatives up to 4 text + 4 image negatives Distillation supervision first 16 answer tokens first 16 answer tokens

The hyperparameter of loss is selected basically based on the scale.

Additional implementation choices are shared across settings. During training, retrieval scores are computed in memory without building a FAISS index. During offline evaluation, we build an FAISS inner-product index over normalized retrieval vectors. Negative reconstruction is enabled for text evidence and image captions.

Query reconstruction is disabled in the main runs. The token count considers from “Context 1:“ to the end, so introducing one more latent token may consume more than one generator token budget (usually 7 to 10 based on LLM/VLM tokenizer). For WebQA token accounting, raw retrieved images are counted after the generator’s native visual frontend; Latent Memory bypasses this raw visual expansion and inserts one projected latent token per retrieved item.

##### B.4 Time Complexity Analysis

This section compares the dominant time complexity under two deployment settings: the evidence index or memory bank has already been compiled, or it must be compiled for the current corpus. Let

- C = {xi}Ni=1 be the evidence pool and let L¯ be the average number of raw text or visual tokens per evidence item. We denote the retrieval depth by k, the number of latent tokens per evidence item by T, the question length by |Q|, the frozen generator hidden size by dϕ, the retrieval embedding model hidden size by de, and the Latent Memory compressor hidden size by dθ. The table omits retrieval/search overhead and keeps only the dominant evidence-encoding and generator-prefill terms.

- Table 8: Dominant time complexity comparison under precompiled and non-precompiled deployment. Retrieval/search overhead is omitted; the key term is the frozen generator prefill length.

Method With precompiled index / memory Without precompilation Full Context O (|Q| + NL¯)2dϕ O (|Q| + NL¯)2dϕ Raw-evidence RAG O (|Q| + kL¯)2dϕ O NL¯2de + (|Q| + kL¯)2dϕ Latent Memory O (|Q| + kT)2dϕ O NL¯2dθ + (|Q| + kT)2dϕ

These formulas follow directly from Transformer prefill complexity: forwarding a sequence of length S through the frozen generator costs O(S2dϕ). Full Context uses S = |Q| + NL¯, RAG uses S = |Q| + kL¯ after selecting k raw evidence items, and Latent Memory uses S = |Q| + kT because each retrieved evidence item is represented by T latent tokens.

• Compared with Full Context, Latent Memory avoids sending the whole evidence pool to the generator. Full Context has a generator-prefill term O((|Q|+NL¯)2dϕ), which grows quadratically with the number of evidence items N when L¯ is fixed. With precompilation, Latent Memory instead uses O((|Q| + kT)2dϕ), which no longer depends on N in the generator. Without precompilation, Latent Memory adds the evidence compilation term O(NL¯2dθ), which is linear in N and can be highly parallelized across evidence items. As shown in the figure below, we plot the relationship of time consumption with the number of evidence items. Latent Memory shows linear complexity and leads to significantly less time complexity, even with compilation.

Full Context Latent Compile Latent Prefill

| |6.39<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| |2.52 2.69| | | | |
| |0.86<br><br>1.36| | | | |
| |0.44 0.33<br><br>0.57 0.03 0.03 0.03 0.06| | | | |
| | | | | | |

- 0
- 1
- 2
- 3
- 4
- 5
- 6

Time(s)

100 200 500 1000

Number of evidence items

- Figure 6: Time analysis on HotpotQA, full-context vs Latent Memory. Compile means pre-compile, and prefill means the prefill process before answer-generation. We run 30 times for variance.

• Compared with raw-evidence RAG, Latent Memory has a similar non-precompiled setup structure: RAG embeds all evidence items, while Latent Memory compiles them into latent memories. The key difference is again the generator input after retrieval. RAG still sends k raw evidence items to the generator, giving |Q| + kL¯, whereas Latent Memory sends kT latent tokens, giving |Q| + kT. Since T ≪ L¯ and usually k ≪ N, Latent Memory has the smallest online generatorprefill complexity among the three methods. Moreover, fewer tokens lead to less probability of out-of-memory, and a square-root less space complexity.

##### B.5 Storage Complexity Analysis

- Table 9 reports an average per-evidence storage comparison on WebQA. The text side uses the official WebQA text snippets in WebQA_train_val.json. The image side uses the official WebQA image files after extraction, which occupy about 75GB in total (the compressed package is about 50GB), averaged over the 350,777 unique image IDs appearing in the official train/validation annotations. We count only the stored evidence representation and do not include small metadata fields.

- Table 9: Average per-item storage on WebQA. Raw text and raw image storage follow the official WebQA annotations and extracted image files. Latent text uses the text-only 1-token memory size (2048 bf16 values), while latent image uses the LLaVA 1-token memory size (4096 bf16 values).

#### Evidence type Stored representation Avg. storage / item Storage effect

Text Official raw text snippet 0.23 KB Reference Text Latent Memory token 4.00 KB 17.6× larger

Image Official extracted image file 209 KB Reference Image Latent Memory token 8.00 KB 26.1× smaller

This comparison clarifies where the storage advantage comes from. On the pure language side, a short raw snippet is often smaller than a high-dimensional latent vector, so text-only Latent Memory should mainly be understood as reducing generator-side token usage rather than persistent corpus storage. On the image side, however, the raw evidence item is much larger; replacing each extracted WebQA image with a single LLaVA latent token gives a clear storage reduction while also avoiding raw visual-token expansion during generation.

The same conclusion also follows from a simple analytical threshold. A one-token LLaVA Latent Memory stores 4096 bf16 values, i.e.,

Slatent/image = 2 × 4096 = 8192 bytes. (11) For an uncompressed RGB image with height H and width W, raw storage is

Simage = 3HW bytes. (12)

Thus, Latent Memory is smaller whenever 3HW > 8192, or H = W > 8192/3 ≈ 52.3 for square images. In other words, a 4096-dimensional bf16 latent token is more storage efficient than an uncompressed RGB image once the image is larger than roughly 53 × 53 pixels. If a separate 512-dimensional fp32 retrieval key is also stored, the per-item latent storage becomes 8192 + 4 × 512 = 10240 bytes, shifting the square-image threshold only to about 59 × 59 pixels.

The online activation footprint follows the same pattern. Raw multimodal RAG must instantiate visual embeddings for each retrieved image inside the VLM, giving an evidence activation size of O(kLd¯ ϕ). Latent Memory instead instantiates only projected latent tokens, giving O(kTdϕ). Thus the online generation path reduces both token consumption and activation memory when L¯ ≫ T, which is typical for image evidence after visual-token expansion.

### C Additional Experiments and Discussion

The experiments in the main text answer the three questions as follows:

- • 1-token Latent Memory can show a token-performance trade-off in multimodal settings, replacing the raw retrieved context.
- • Replacing raw evidence with 8-token Latent Memory results in a better trade-off, surpassing the best RAG baseline on out-of-domain performance.

The appendix then studies where this behavior comes from and how broadly it holds. We organize the additional evidence around the following questions:

- • RQ-A1: How much token each latent evidence are needed for text-only and multimodal QA? Appendix C.1 varies the number of latent tokens per evidence item.
- • RQ-A2: Does the Latent Memory perform well across evidence domains, compressors, and generators? Appendices C.2, C.5, C.4, and C.6 test broader text and image QA benchmarks, compressors, and generator settings.
- • RQ-A3: Which training objectives support the unified representation space? Appendix C.3 ablates reconstruction, negative reconstruction, query reconstruction, and augmentation settings.
- • RQ-A5: How effective are compression and retrieval, respectively? Appendix C.7 excludes R@k improvements, how effective is compression? We considered a variant using latent tokens to provide the generator with raw evidence after retrieval to observe compression effectiveness.

##### C.1 Token Count Ablation

This section varies the number of latent tokens allocated to each evidence item. The default model uses one token; we also evaluate 2-, 4-, and 8-token variants while keeping the evidence unit and retrieval granularity unchanged. For an evidence item xi, the multi-token variant emits

Zi = [zi,1,zi,2,...,zi,T], zi,t ∈ Rd

. (13) Retrieval still uses one key per evidence item by pooling these tokens before the retrieval projection:

θ

T

1 T

zi,t, zi = LayerNorm(Wrz¯i). (14)

z¯i =

t=1

Thus, the ablation tests whether extra latent capacity improves generation enough to justify the larger token budget.

- Table 10 reports the text-QA results, and Figures 7–9 summarize the accuracy, token, and recall trends. Analysis. Three conclusions are consistent across text-only setting (Table 10 and Figures 7–9).

- 1. Increasing the latent-token budget consistently improves EM and F1, and the larger-token variants surpass the main retrieval baselines in answer quality.
- 2. By contrast, increasing the latent-token budget does not lead to a comparable improvement in Recall@k at fixed k.
- 3. These quality gains are achieved while preserving the overall trade-off: larger latent budgets do increase generator tokens, but they remain substantially more efficient than Full Context and still compare favorably with the raw retrieval baselines.

- Table 10: Token-count ablation on text QA over HotpotQA, 2WikiMultihopQA, and MuSiQue. The Average columns report the out-of-domain average over 2WikiMultihopQA and MuSiQue. The

- 1-token setting is the main Latent Memory configuration; 2/4/8-token variants allocate more latent tokens per evidence item. #Tok reports the task-relevant generator budget.

In-Domain Out-of-Domain Dataset HotpotQA 2WikiMultihopQA MuSiQue Average Method EM F1 R@k #Tok EM F1 R@k #Tok EM F1 R@k #Tok EM F1 R@k #Tok Full Context 42.0 57.5 – 1462 17.7 39.7 – 1074 6.0 15.5 – 2580 11.9 27.6 – 1827 LLMLingua (20%) 31.8 44.8 – 283 17.0 30.4 – 199 11.6 21.8 – 492 14.3 26.1 – 346 LLMLingua (10%) 25.0 36.1 – 154 14.8 24.7 – 108 6.9 14.9 – 259 10.9 19.8 – 184 LLMLingua (5%) 20.9 30.2 – 87 15.0 22.2 – 63 4.3 10.6 – 137 9.7 16.4 – 100 Retrieval Augmented Generation Methods

- BM25 Retrieval (k=1) 32.5 44.7 30.2 68 17.2 27.2 18.3 60 6.2 14.2 7.0 69 11.7 20.7 12.7 65

- BM25 Retrieval (k=2) 36.8 49.9 45.5 106 16.2 28.4 29.2 94 8.0 16.7 12.2 106 12.1 22.6 20.7 100 BM25 Retrieval (k=5) 41.3 55.3 65.5 224 16.2 32.8 46.9 196 9.5 19.6 22.4 221 12.9 26.2 34.7 209

- Dense Retrieval (k=1) 29.4 41.0 27.9 67 19.0 29.9 23.3 61 7.8 17.1 9.6 70 13.4 23.5 16.5 66

- Dense Retrieval (k=2) 32.6 45.3 42.2 104 18.1 31.7 37.6 95 9.9 19.9 17.0 106 14.0 25.8 27.3 101 Dense Retrieval (k=5) 37.0 50.7 62.4 214 19.1 37.4 58.0 198 12.8 23.4 31.3 218 16.0 30.4 44.7 208

- Qwen3-Emb-0.6B (k=1) 30.9 43.6 33.1 70 18.1 30.9 32.5 66 8.1 18.1 10.3 73 13.1 24.5 21.4 70

- Qwen3-Emb-0.6B (k=2) 35.6 49.0 50.0 109 17.4 33.5 47.7 102 9.8 20.4 18.5 112 13.6 27.0 33.1 107 Qwen3-Emb-0.6B (k=5) 40.0 54.5 70.1 224 19.1 38.6 64.3 208 13.7 24.6 34.8 230 16.4 31.6 49.6 219 Retrieval Augmented Generation for Latent Tokens in the Latent Memory (Ours)

- 1-token (k=1) 27.4 39.4 34.6 36 19.8 29.2 28.4 33 5.8 14.6 8.7 37 12.8 21.9 18.6 35

- 1-token (k=2) 31.6 45.2 62.6 45 21.5 33.5 49.5 42 7.3 16.9 15.5 46 14.4 25.2 32.5 44

- 1-token (k=5) 34.8 48.9 87.1 72 24.3 36.7 74.2 69 8.7 19.2 30.1 73 16.5 28.0 52.2 71

- 2-token (k=1) 27.3 39.5 34.6 37 20.2 29.9 28.0 34 5.7 15.2 8.7 38 13.0 22.6 18.4 36

- 2-token (k=2) 31.4 45.2 62.6 47 23.0 34.7 49.0 44 7.3 17.6 15.2 48 15.2 26.2 32.1 46 2-token (k=5) 35.2 49.3 87.1 77 26.6 38.0 73.8 74 8.5 19.8 29.4 78 17.6 28.9 51.6 76

- 4-token (k=1) 28.6 40.4 35.2 39 21.9 31.4 30.9 36 5.8 15.8 9.0 40 13.9 23.6 20.0 38

- 4-token (k=2) 33.0 46.9 63.2 51 25.1 36.2 53.5 48 7.6 18.1 16.0 52 16.4 27.2 34.8 50 4-token (k=5) 35.8 49.9 88.0 87 28.9 39.1 76.7 84 8.8 19.9 31.0 88 18.9 29.5 53.9 86

- 8-token (k=1) 31.6 44.7 35.1 43 21.2 31.6 30.5 40 7.5 17.7 9.2 44 14.4 24.7 19.9 42

- 8-token (k=2) 38.9 53.2 63.4 59 24.0 36.4 52.8 56 10.6 21.5 16.2 60 17.3 29.0 34.5 58 8-token (k=5) 42.8 58.5 88.0 107 26.8 40.5 76.2 104 13.0 24.5 31.1 108 19.9 32.5 53.7 106

###### OOD EM vs. Token Budget

20

k=5

k=5

18

| |
|---|

k=5

k=2

k=5

k=5

OODAverageEM

k=5

16

k=2

| |
|---|

k=2

k=2

k=2

14

k=2

k=1

k=1

k=1

k=1

k=5

| |
|---|

k=2

k=1

k=1

12

Full Context

BM25 Retrieval

Latent (2-token) Latent (4-token) Latent (8-token)

LLMLingua (20%) LLMLingua (10%) LLMLingua (5%)

Dense Retrieval

10

Qwen3-Emb

Latent (1-token)

102 103 Average Input Tokens to Generator

- Figure 7: OOD average text-QA EM as a function of average generator tokens. The average is computed over 2WikiMultihopQA and MuSiQue.

102 103 Average Input Tokens to Generator

16

18

20

22

24

26

28

30

32

OODAverageF1

k=1

k=2

k=5

k=1

k=2

k=5

k=1

k=2

k=5

k=1

k=2

k=5

k=1

k=2

k=5

k=1

k=2

k=5

k=1

k=2

k=5

OOD F1 vs. Token Budget

| |
|---|

| |
|---|

Full Context

LLMLingua (20%) LLMLingua (10%) LLMLingua (5%)

BM25 Retrieval

Dense Retrieval

Qwen3-Emb

Latent (1-token)

Latent (2-token) Latent (4-token) Latent (8-token)

- Figure 8: OOD average text-QA F1 as a function of average generator tokens. The average is computed over 2WikiMultihopQA and MuSiQue.

1 2 5 k

20

30

40

50

OODAverageRecall@k

OOD Recall@k vs. Retrieval Cutoff

| |
|---|

| |
|---|

| |
|---|

BM25 Retrieval

Dense Retrieval

Qwen3-Emb

- Latent (1-token)

- Latent (2-token)

Latent (4-token) Latent (8-token)

- Figure 9: OOD average Recall@k as a function of k. The average is computed over 2WikiMultihopQA and MuSiQue.

##### C.2 Generalization Ability on More Domains

The experiments in the main body of this paper demonstrate that Latent Memory remains effective on the OOD dataset, but these results can be considered generalizations to similar corpora. In this section, we consider whether this effectiveness can be extended to more different situations and datasets that are very different from HotpotQA. Table 11 reports generalization to four additional text QA benchmarks: NQ (open-domain factoid), TriviaQA (trivia-style factoid), Qasper (scientific document QA), and WICE (claim verification). Unlike the three datasets in the main text, these datasets do not have a retrieval label design, so we did not calculate R@k. No fine-tuning is performed on these datasets; the same checkpoint trained on HotpotQA is evaluated directly. This tests whether the compressed latent representations generalize across domains and task types.

- Table 11: Generalization ability on more text domains under the same frozen Meta-Llama-3-8BInstruct generator. No fine-tuning is performed on these datasets. #Tok reports the task-relevant generator budget. Unlike the three datasets in the main text, these datasets do not have a retrieval label design, so we did not calculate R@k.

NQ TriviaQA Qasper WiCE Average Method EM F1 #Tok EM F1 #Tok EM F1 #Tok EM F1 #Tok EM F1 #Tok Full Context 0.0 1.1 23588 18.9 24.7 10828 2.5 16.5 4701 51.1 51.1 2311 18.1 23.3 10357 Raw-evidence retrieval

- BM25 (k=1) 25.3 36.6 54 71.0 77.7 66 6.0 17.1 50 43.6 43.6 135 36.5 43.8 77

- BM25 (k=2) 26.7 38.6 90 70.7 77.8 106 8.0 20.5 82 44.1 44.1 210 37.4 45.3 122 BM25 (k=5) 28.1 40.4 197 71.0 78.3 226 9.5 23.6 187 51.7 51.7 417 40.1 48.5 257

- Dense (k=1) 28.2 40.5 58 70.2 77.1 57 5.5 16.6 46 39.9 39.9 116 36.0 43.5 69

- Dense (k=2) 30.6 43.5 96 71.2 78.3 85 8.5 21.1 76 43.6 43.6 174 38.5 46.6 108 Dense (k=5) 33.1 46.5 210 71.7 79.2 171 11.5 26.8 168 46.4 46.4 351 40.7 49.7 225

- Qwen3-Emb-0.6B (k=1) 30.8 43.0 60 71.9 79.0 70 8.5 19.8 47 42.2 42.2 126 38.4 46.0 76

- Qwen3-Emb-0.6B (k=2) 32.0 45.1 99 72.1 79.8 110 10.0 26.6 80 42.5 42.5 201 39.1 48.5 122 Qwen3-Emb-0.6B (k=5) 34.5 48.2 218 72.8 80.9 221 8.5 30.2 171 47.5 47.5 389 40.8 51.7 250 Latent Memory

- Latent Memory (k=1) 22.2 33.7 26 66.5 72.9 34 3.0 10.9 25 45.5 45.5 59 34.3 40.8 36

- Latent Memory (k=2) 22.7 34.6 35 66.4 73.0 44 4.5 12.2 34 52.5 52.5 68 36.5 43.1 45 Latent Memory (k=5) 23.4 35.0 62 65.1 71.7 70 6.0 12.3 61 55.9 55.9 94 37.6 43.7 72

- 8-token Latent Memory (k=1) 25.5 37.2 33 68.9 75.4 42 5.0 14.5 32 52.8 52.8 66 38.1 45.0 43

- 8-token Latent Memory (k=2) 24.9 36.8 49 68.1 74.8 58 5.0 14.3 48 57.0 57.0 82 38.7 45.7 59 8-token Latent Memory (k=5) 26.7 38.5 97 67.7 74.4 106 5.0 13.2 96 60.3 60.3 129 39.9 46.6 107

On these datasets, the RAG method based on Qwen-3-Embedding performs best, but we note that the 1-token and 8-token versions of Latent Memory exhibit good trade-offs. The 8-token Latent Memory method achieves similar performance to BM25 while requiring 2.5 times fewer tokens.

Multimodal generalization. Table 12 further evaluates multimodal generalization on a 20-image dataset SlideQA (Consider getting detailed answers from 1-2 correct slides.) with the same frozen LLaVA-1.5-13B generator. Latent Memory improves retrieval coverage and competitive EM value with far fewer generator tokens, but Nemo remains stronger on EM/F1.

- Table 12: Generalization ability on the multimodal SlideQA domain under a frozen LLaVA-1.5-13B generator. Bold marks the best result in each metric column, and underlining marks the second best one. #Tok reports the generator-side input budget.

Dataset: SlideQA Generation VLM (fixed): LLaVA-1.5-13B Method EM F1 R@k #Tok Full-context baseline Full Context 0.0 0.0 – 11871 Raw-evidence multimodal retrieval

- Nemo (k=1) 17.1 26.2 7.2 621
- Nemo (k=2) 8.3 17.7 14.1 1212 Nemo (k=5) 4.7 13.2 30.5 2887 Latent Memory

- Latent Memory (k=1) 11.1 15.7 21.5 37

- Latent Memory (k=2) 7.0 12.4 33.6 47 Latent Memory (k=5) 5.5 10.3 56.0 77

##### C.3 Core Training Ablation

- Table 13 ablates the reconstruction and distillation-negative designs used to train the text-only 1B compressor. The retrieval and generation settings are fixed; only the training targets or distillation context augmentation change. The variants are:

- • Ours-1B: Full model that reconstructs both positive evidence and sampled negative evidence, with query reconstruction disabled.
- • w/o Constructing: Evidence reconstruction is removed entirely, so neither positive nor negative evidence is reconstructed.
- • w/o Constructing Negative Examples: Only positive evidence is reconstructed; sampled negative evidence is not used as a reconstruction target.
- • w/ Query Reconstruction: Adds query reconstruction to the default evidence-reconstruction setup.
- • w/ Distillation Negative Augmentation: During distillation, the student latent context is augmented with 0-3 randomly sampled irrelevant negative memories, while the teacher still sees only positive.

- Table 13: Generator-side reconstruction ablation on text QA. All variants use the same frozen MetaLlama-3-8B-Instruct generator and matched latent-token budget.

HotpotQA 2WikiMultihopQA MuSiQue Average Variant EM F1 R@k EM F1 R@k EM F1 R@k EM F1 R@k Default model: reconstruct positive and negative evidence

- Ours-1B (k=1) 27.4 39.4 34.6 19.8 29.2 28.4 5.8 14.6 8.7 17.7 27.7 23.9

- Ours-1B (k=2) 31.6 45.2 62.6 21.5 33.5 49.5 7.3 16.9 15.5 20.1 31.9 42.6 Ours-1B (k=5) 34.8 48.9 87.1 24.3 36.7 74.2 8.7 19.2 30.1 22.6 34.9 63.8 Reconstruction Loss

- w/o Constructing (k=1) 23.4 37.8 33.4 20.3 28.8 29.8 5.9 15.1 8.3 16.5 27.2 23.8

- w/o Constructing (k=2) 27.6 42.7 62.7 24.2 32.4 48.4 7.9 17.8 14.4 19.9 31.0 41.8 w/o Constructing (k=5) 30.6 45.2 85.2 22.7 33.8 73.3 9.4 19.2 30.4 20.9 32.7 62.9

- w/o Constructing Negative (k=1) 24.3 39.3 34.1 19.0 30.1 28.0 5.7 14.8 7.6 16.3 28.1 23.2

- w/o Constructing Negative (k=2) 29.1 44.9 58.0 19.8 33.9 47.0 6.3 17.2 13.8 18.4 32.0 39.6 w/o Constructing Negative (k=5) 31.7 46.2 86.2 22.3 36.1 72.2 8.3 18.9 28.7 20.8 33.8 62.3

- w/ Query Reconstruction (k=1) 22.3 35.9 34.2 19.3 28.4 25.8 4.8 13.6 7.8 15.5 26.0 22.6

- w/ Query Reconstruction (k=2) 26.5 41.7 60.7 22.6 31.8 45.4 6.3 15.4 13.7 18.5 29.6 39.9 w/ Query Reconstruction (k=5) 30.1 45.7 84.9 26.3 35.4 72.6 7.7 18.0 26.3 21.3 33.0 61.3 Distillation Loss

- w/o Negative Augmentation (k=1) 26.2 38.2 33.4 21.0 30.9 27.6 5.5 14.6 8.1 17.6 27.9 23.0

- w/o Negative Augmentation (k=2) 31.3 44.4 59.0 22.8 34.8 48.4 6.6 16.2 13.8 20.2 31.8 40.4 w/o Negative Augmentation (k=5) 31.6 44.8 83.0 24.3 35.8 73.6 7.2 17.5 25.5 21.0 32.7 60.7

##### Analysis.

- 1. Ours-1B is the strongest default overall, but the main takeaway is that the reconstruction objective stabilizes both retrieval and generation.
- 2. Removing evidence reconstruction entirely (w/o Constructing) consistently hurts EM/F1 and also lowers Recall@k on average. This joint degradation supports the view that the same latent token is serving retrieval and generation in a unified representation space.
- 3. Only reconstructing positive evidence (w/o Constructing Negative Examples) is less stable and degrades retrieval more clearly. This supports the role of negative evidence as a geometric anchor for the latent memory space.
- 4. Adding query reconstruction (w/ Query Reconstruction) is harmful in this setting. It forces the query adapter to preserve surface reconstruction information, which conflicts with its role as a retrieval query representation.
- 5. Adding 0–3 irrelevant negative memories during distillation (w/ Distillation Negative Augmentation) does not recover the full model, especially when k = 5.

Across the datasets, removing reconstruction weakens both retrieval and generation rather than trading one for the other. Negative evidence is useful when it is reconstructed as part of the memory representation, but query reconstruction and direct irrelevant-memory augmentation perturb the intended roles of the query encoder and generator-side conditioning target.

##### C.4 Stronger Text Compressors

This section asks whether a larger or a different-family compressor improves one-token Latent Memory’s quality. The generator (frozen Meta-Llama-3-8B-Instruct), token budget, and training recipe are held fixed; only the compression backbone changes. Table 14 compares LLaMA-1B (the default), LLaMA-3B, and Qwen-1.5B, averaged over HotpotQA, 2WikiMultihopQA, and MuSiQue.

- Table 14: Text-compressor ablation on text QA. The frozen generator and training recipe are fixed; only the compression backbone changes.

HotpotQA 2WikiMultihopQA MuSiQue Average Encoder EM F1 R@k EM F1 R@k EM F1 R@k EM F1 R@k LLaMA-family compressors

LLaMA-1B (k=1) 27.4 39.4 34.6 19.8 29.2 28.4 5.8 14.6 8.7 17.7 27.7 23.9 LLaMA-1B (k=2) 31.6 45.2 62.6 21.5 33.5 49.5 7.3 16.9 15.5 20.1 31.9 42.6 LLaMA-1B (k=5) 34.8 48.9 87.1 24.3 36.7 74.2 8.7 19.2 30.1 22.6 34.9 63.8

- LLaMA-3B (k=1) 29.6 42.5 35.7 20.1 30.5 29.1 6.3 15.8 9.1 18.7 29.6 24.6

- LLaMA-3B (k=2) 34.4 48.9 64.2 23.0 35.9 51.9 8.9 19.4 16.0 22.1 34.7 44.0 LLaMA-3B (k=5) 35.4 49.7 88.6 25.1 37.3 77.6 9.9 19.9 30.1 23.5 35.6 65.4 Qwen-family compressor

- Qwen-1.5B (k=1) 25.5 37.2 34.8 20.6 30.3 30.0 5.0 13.2 8.8 17.0 26.9 24.5

- Qwen-1.5B (k=2) 29.2 41.8 62.1 24.0 35.4 52.1 6.0 14.6 15.5 19.8 30.6 43.2 Qwen-1.5B (k=5) 30.7 43.4 86.4 27.4 37.4 76.5 6.2 16.3 30.1 21.4 32.3 64.3

##### Analysis.

- 1. LLaMA-3B is consistently the strongest encoder in this comparison, improving both EM/F1 over LLaMA-1B and Qwen-1.5B at all reported k values.
- 2. The gap is more pronounced in EM/F1 than in Recall@k, which suggests that encoder choice influences downstream answer quality more strongly than retrieval coverage alone.
- 3. Since the token budget is matched across all three variants, the advantage of LLaMA-3B is best interpreted as a representational benefit rather than an efficiency effect.

The dataset-level breakdown further shows that the stronger compressor helps both in-domain and OOD evaluation. On HotpotQA, LLaMA-3B improves the default LLaMA-1B model from 34.8/48.9 EM/F1 to 35.4/49.7 at k=5, while maintaining similar high Recall@k. On 2WikiMultihopQA, the improvement is more visible: LLaMA-3B reaches 25.1 EM and 37.3 F1 at k=5, compared with 24.3/36.7 for LLaMA-1B, and also improves Recall@k from 74.2 to 77.6. MuSiQue remains the most difficult OOD dataset; LLaMA-3B improves EM/F1 modestly, but Recall@k is nearly unchanged at k=5, indicating that the encoder upgrade mainly improves the quality of the latent representation consumed by the generator rather than solving all retrieval coverage limitations. Qwen-1.5B obtains competitive Recall@k on 2Wiki and MuSiQue, but its lower EM/F1 suggests that high retrieval scores do not necessarily imply better latent-conditioned generation.

##### C.5 Direct Transfer to Similar Generator

This experiment studies whether Latent Memory trained with the Meta-Llama-3-8B-Instruct generator can be directly reused by another similar LLaMA generator. We keep the compressor, latent memory bank, retrieval procedure, and projection interface fixed, and replace only the frozen answer generator with LLaMA-3.1-8B-Instruct. No additional compressor training or latent-token adaptation is performed. Table 15 reports the full text-QA results under this transferred generator.

- Table 15: Direct generator transfer on text QA. Latent tokens are trained under Meta-Llama-3-8BInstruct and evaluated directly with a frozen LLaMA-3.1-8B-Instruct generator. #Tok reports the generator-side input budget.

Generation LLM (fixed): LLaMA-3.1-8B-Instruct

HotpotQA 2WikiMultihopQA MuSiQue Average Method EM F1 R@k #Tok EM F1 R@k #Tok EM F1 R@k #Tok EM F1 R@k #Tok Full-context reference Full Context 50.2 65.9 – 1462 35.2 48.7 – 1074 18.8 30.7 – 2580 34.7 48.4 – 1706 Sparse raw-evidence RAG baseline BM25 Retrieval (k=1) 29.6 43.0 30.2 68 13.5 23.2 18.3 60 6.1 13.3 7.0 69 16.4 26.5 18.5 66 BM25 Retrieval (k=2) 34.6 48.4 45.5 106 16.7 25.4 29.2 94 7.8 15.2 12.2 106 19.7 29.7 29.0 102 BM25 Retrieval (k=5) 41.2 55.4 65.5 224 24.4 32.0 46.9 196 10.6 19.2 22.4 221 25.4 35.5 44.9 214 Dense raw-evidence RAG baselines Dense Retrieval (k=1) 26.5 39.3 27.9 67 15.0 25.5 23.3 61 6.6 15.1 9.6 70 16.0 26.6 20.3 66 Dense Retrieval (k=2) 30.6 44.1 42.2 104 18.9 29.0 37.6 95 8.3 17.5 17.0 106 19.3 30.2 32.3 102 Dense Retrieval (k=5) 36.9 50.9 62.4 214 27.8 37.0 58.0 198 12.6 22.7 31.3 218 25.7 36.8 50.6 210

- Qwen3-Emb (k=1) 30.9 43.6 33.1 70 18.1 30.9 32.5 66 8.1 18.2 10.3 73 19.0 30.9 25.3 70

- Qwen3-Emb (k=2) 35.6 49.0 50.0 109 17.4 33.5 47.7 102 9.8 20.4 18.5 112 21.0 34.3 38.8 108 Qwen3-Emb (k=5) 40.0 54.5 70.1 224 19.1 38.6 64.3 208 13.7 24.7 34.8 230 24.3 39.3 56.4 221 Direct transfer of Latent Memory

- 1-token Latent Memory (k=1) 23.7 36.8 34.6 36 21.4 29.0 28.3 33 5.1 13.3 8.7 37 16.7 26.4 23.9 36

- 1-token Latent Memory (k=2) 29.6 43.6 62.7 45 26.9 34.9 49.5 42 6.2 15.7 15.5 46 20.9 31.4 42.6 45

- 1-token Latent Memory (k=5) 34.8 48.3 87.0 72 33.0 39.9 74.2 69 8.1 18.4 30.1 73 25.3 35.5 63.8 72

- 2-token Latent Memory (k=1) 24.5 36.8 34.6 37 22.6 30.0 28.0 34 5.2 13.2 8.7 38 17.4 26.7 23.7 36 2-token Latent Memory (k=2) 29.9 43.6 62.5 47 27.3 34.9 49.1 44 7.4 16.9 15.3 48 21.5 31.8 42.3 46 2-token Latent Memory (k=5) 34.6 47.8 87.0 77 31.0 38.6 73.8 74 9.3 19.9 29.4 78 25.0 35.5 63.4 76

- 4-token Latent Memory (k=1) 25.5 38.3 35.2 39 23.4 30.8 30.9 36 5.4 14.0 9.2 40 18.1 27.7 25.1 38

- 4-token Latent Memory (k=2) 32.7 45.8 63.2 51 30.2 36.9 53.5 48 8.1 18.1 16.2 52 23.7 33.6 44.3 50 4-token Latent Memory (k=5) 36.4 49.4 88.0 87 33.7 40.0 76.7 84 9.3 19.6 31.0 88 26.5 36.3 65.2 86

- 8-token Latent Memory (k=1) 29.4 43.1 35.1 80 24.2 32.4 30.5 97 7.4 16.4 9.2 101 20.3 30.6 24.9 93

- 8-token Latent Memory (k=2) 38.2 52.7 63.4 96 29.5 38.1 52.7 113 11.1 21.1 16.3 117 26.2 37.3 44.1 109 8-token Latent Memory (k=5) 44.2 58.7 88.0 144 33.5 42.0 76.2 161 14.3 24.7 31.1 165 30.7 41.8 65.1 157

Analysis. The transferred latent tokens remain usable with the new generator without re-training. The one-token variant preserves the same token-efficiency pattern as in the main experiments, while larger latent-token budgets provide a clear capacity gain. In particular, the 8-token variant improves the average EM/F1 to 30.7/41.8 at k=5, exceeding the strongest raw-evidence embedding baseline at the same k while still using fewer generator tokens. This suggests that Latent Memory is not tightly bound to a single frozen LLaMA generator: once the latent tokens learn to act as compact evidence, another compatible LLaMA instruction model can consume them directly.

##### C.6 Multimodal Results with Gemma

- Table 16 reports WebQA results when the generator is switched to a frozen Gemma-3-12B-Instruct. This model-swap setting keeps the retrieval pool and evidence candidates fixed, and compares four groups: full-context prompting, text-only raw-evidence retrievers, multimodal raw-evidence retrievers, and direct generation from Latent Memory. We fine-tune Gemma-3-4B-PT as the compressor and use LLaMA-3.2-1B-Instruct as the reconstruction decoder.

- Table 16: Multimodal QA (WebQA) with unified retrieval and generation using a frozen Gemma-3-

- 12B-Instruct generator. Generation VLM (fixed): Gemma-3-12B-Instruct

Method WebQA-Image WebQA-Text Avg F1 Acc R@k #Tok F1 Acc R@k #Tok F1 Acc R@k #Tok

Full-context reference Full Context 13.8 18.0 – 5856 54.9 54.9 – 4337 34.3 36.4 – 5097

Raw-evidence text retrieval baselines

BM25 Retrieval (k=1) 8.2 9.2 21.8 167 38.6 40.4 31.8 106 23.4 24.8 26.8 137 BM25 Retrieval (k=2) 8.3 9.5 28.3 274 43.6 44.8 51.0 183 26.0 27.2 39.6 228 BM25 Retrieval (k=5) 8.8 10.5 37.9 559 49.5 50.3 73.0 415 29.2 30.4 55.5 487 Dense Retrieval (k=1) 10.6 15.8 39.7 241 36.2 37.0 25.5 158 23.4 26.4 32.6 200 Dense Retrieval (k=2) 10.3 14.6 53.1 411 39.0 40.0 40.1 303 24.6 27.3 46.6 357 Dense Retrieval (k=5) 10.6 14.7 67.1 848 45.2 45.9 62.0 750 27.9 30.2 64.5 799

Raw-evidence multimodal retrieval baselines

Qwen3-VL-8B (k=1) 8.6 11.6 24.0 164 40.7 42.5 40.8 108 24.7 27.1 32.4 136 Qwen3-VL-8B (k=2) 9.1 11.1 32.6 270 47.6 48.7 66.3 187 28.4 29.9 49.5 228 Qwen3-VL-8B (k=5) 9.5 12.4 49.6 570 52.3 53.5 87.5 442 30.9 32.9 68.5 506

- Nemo-Emb (k=1) 11.3 17.8 48.8 255 41.5 43.3 40.5 107 26.4 30.5 44.6 181

- Nemo-Emb (k=2) 11.1 17.0 64.7 444 48.9 50.1 66.6 186 30.0 33.5 65.7 315 Nemo-Emb (k=5) 11.2 17.3 81.5 948 52.1 53.0 87.1 450 31.7 35.1 84.3 698 Ours: direct generation from Latent Memory

- Latent Memory (k=1) 11.7 18.7 52.5 38 31.6 31.9 23.7 39 21.6 25.3 38.1 38

- Latent Memory (k=2) 11.3 18.0 70.1 47 31.9 32.3 41.3 48 21.6 25.1 55.7 48 Latent Memory (k=5) 11.3 17.5 89.0 74 31.3 31.7 70.5 75 21.3 24.6 79.8 74

Analysis. With Gemma-3-12B-Instruct, Latent Memory remains the most token-efficient option. Demonstrating the best WebQA-Image Accuracy and the second best F1 with 10× less tokens.

##### C.7 Latent Tokens as Retrievers

Latent Memory’s latent token plays two roles at once: it is a retrieval key used to rank evidence and a generation input consumed by the frozen generator. This section isolates these two roles with a hybrid as-RAG mode. In this mode, latent tokens are used only for retrieval; after the top-k evidence items are selected, generation is performed from the retrieved raw text or images rather than from latent tokens. We report text-only and multimodal variants separately because the raw evidence format differs across settings.

- • Ours-1B: full Latent Memory—latent tokens used for both retrieval and generation.
- • Ours-1B-RAG: latent tokens used for retrieval only; generator sees raw retrieved text sentences.
- • Latent-Token-based RAG: multimodal variant of the above—latent tokens rank the unified text-image pool; generator sees raw retrieved text and images.

Comparing these variants reveals how much of Latent Memory’s gain comes from better retrieval (latent key quality) versus better generation (latent token quality as generator input).

##### Analysis.

- 1. In the plain text setting, the "Latent Memory as RAG" method based on raw evidence effectively isolates retrieval and compression. We observed that with Latent Memory k=1, it outperforms Latent-Memory-as-RAG (k=5) using the same number of tokens. This demonstrates that even

- Table 17: Text-only-as-RAG results. Latent retrieval selects raw text evidence, and generation is performed from the retrieved raw text. #Tok excludes fixed prompt scaffolding.

HotpotQA 2WikiMultihopQA MuSiQue Avg Methods EM F1 EM F1 EM F1 EM F1 R@k #Tok Full-context reference Full Context 42.0 57.8 17.7 39.2 6.0 17.1 21.9 38.0 – 1706 Raw-evidence RAG baseline

- Qwen3-Emb (k=1) 30.9 43.6 18.1 30.9 8.1 18.2 19.0 30.9 25.3 70

- Qwen3-Emb (k=2) 35.6 49.0 17.4 33.5 9.8 20.4 21.0 34.3 38.8 108 Qwen3-Emb (k=5) 40.0 54.5 19.1 38.6 13.7 24.6 24.3 39.2 56.4 221 Direct generation from Latent Memory

- Latent Memory (k=1) 27.4 39.4 19.8 29.2 5.8 14.6 17.7 27.7 23.9 36

- Latent Memory (k=2) 31.6 45.2 21.5 33.5 7.3 16.9 20.1 31.9 42.6 45 Latent Memory (k=5) 34.8 48.9 24.3 36.7 8.7 19.2 22.6 34.9 63.8 72

Latent retrieval as raw-evidence RAG Latent Memory-as-RAG (k=1) 37.0 49.8 18.6 30.7 10.6 19.9 22.1 33.5 23.9 75 Latent Memory-as-RAG (k=2) 44.0 58.9 18.4 35.4 13.2 22.8 25.2 39.0 42.6 125 Latent Memory-as-RAG (k=5) 49.9 66.2 21.9 42.8 18.0 30.2 29.9 46.4 63.8 259

- Table 18: Multimodal-as-RAG results. Latent retrieval ranks the unified text-image pool, but generation is performed from retrieved raw text and raw images.

Method WebQA-Image WebQA-Text Avg

F1 Acc R@k #Tok F1 Acc R@k #Tok F1 Acc R@k #Tok Raw-evidence multimodal retrieval baselines

- Qwen3-VL-8B (k=1) 15.1 22.0 24.1 284 40.8 43.5 40.7 131 27.9 32.8 32.4 208

- Qwen3-VL-8B (k=2) 20.1 24.5 32.9 465 46.1 50.2 66.3 235 33.1 37.3 49.6 350 Qwen3-VL-8B (k=5) 34.1 30.3 49.7 957 47.8 57.2 87.2 612 41.0 43.8 68.5 784

- Nemo-Emb (k=1) 19.9 25.2 48.7 508 41.1 44.0 40.4 130 30.5 34.6 44.6 319

- Nemo-Emb (k=2) 32.9 30.6 64.8 892 46.8 50.9 66.6 233 39.9 40.8 65.7 563 Nemo-Emb (k=5) 53.0 37.2 81.4 1885 48.6 57.9 87.1 629 50.8 47.6 84.2 1257 Direct generation from Latent Memory

- Latent Memory (k=1) 32.0 28.7 56.6 42 28.8 30.8 24.0 44 30.4 29.7 40.3 43

- Latent Memory (k=2) 56.5 39.5 74.7 52 30.0 32.8 41.1 54 43.2 36.2 57.9 53 Latent Memory (k=5) 69.4 44.2 91.2 82 30.7 34.3 70.5 84 50.0 39.2 80.8 83 Latent retrieval as raw-evidence multimodal RAG

- Latent Memory-as-RAG (k=1) 21.4 25.8 56.6 638 34.9 37.6 24.0 122 28.2 31.7 40.3 380

- Latent Memory-as-RAG (k=2) 38.3 33.3 74.7 1233 40.3 43.9 41.1 207 39.3 38.6 57.9 720 Latent Memory-as-RAG (k=5) 54.7 37.0 91.2 2970 45.1 51.2 70.5 461 49.9 44.1 80.8 1715

disregarding the advantage of retrieval, the compression portion of Latent Memory still exhibits a trade-off, and the 8-token Latent Memory results show a better trade-off.

- 2. In multimodal WebQA, the conclusion is different. The "Latent Memory as RAG" method based on raw evidence outperforms in the text portion, but it lags behind using latent tokens in the image portion. This indicates that the performance improvement on the image side is related to obtaining a more efficient representation, which helps alleviate the context processing pressure on large models.
- 3. Furthermore, the improvement in recall@k itself may provide inspiration for the training of subsequent embedding models. This is an embedding design based on providing raw context, demonstrating the effectiveness of unified retrieval and representation space.

### D Case Study

The case studies are not additional leaderboard evidence; they diagnose how the compact representation behaves on individual examples. We use them to answer three qualitative questions:

- • RQ-C1: What does reconstruction reveal about the latent token? Appendix D.1 checks whether reconstruction preserves key evidence rather than copying the full sentence verbatim.
- • RQ-C2: Can text-only latent retrieval recover complete multi-hop chains? Appendix D.2 shows examples where the retrieved latent memories cover all required supporting facts.
- • RQ-C3: How does unified text-image retrieval behave on concrete multimodal questions? Appendix D.3 shows additional WebQA cases using the same unified text-image memory pool as the main experiments.

##### D.1 Reconstruction Quality of Latent Tokens

Reconstruction Loss vs. Latent Token Count

| |1 token 2 tokens 4 tokens 8 tokens<br><br>| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

1.75

1.50

ReconstructionCE

1.25

1.00

0.75

0.50

0.25

0.00

0 5k 10k 15k 20k 25k 30k 35k

Training step

- Figure 10: Training reconstruction loss for 1-, 2-, 4-, and 8-token Latent Memory variants. More latent tokens consistently reduce the reconstruction CE, indicating that additional latent capacity preserves more evidence information during compression.

- Table 19: Qualitative reconstruction examples on HotpotQA. CE is measured against the original evidence sentence under teacher forcing.

Original evidence 1-token reconstruction CE 8-token reconstruction CE

Shirley Temple: Shirley Temple Black (April 23, 1928–February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood’s number one box-office draw as a child actress from 1935 to 1938.

actress greater greater greater greater greater ...

0.226 Shirley Temple (April 23, 1928– February 10, 2014) was an actress, singer, dancewoman, and business actress who was Hollywood’s number one child ...

0.062

2014 S/S: 2014 S/S is the debut album of South Korean group WINNER.

S:#:#:#:#:#:#:#:#:#:# ... 0.486 2014 is the debut album of South Korean group WINNER ...

0.016

The table above provides examples for latent reconstruction. 1-tokens are okay for low CE but cannot fully reconstruct the evidence; improving the number of tokens per latent evidence is helpful for reconstruction as the reconstruction loss in Figure 10 will drop accordingly.

The ability to reconstruct the text not only provides a stronger and more faithful representation of the original text but also enhances the interpretability of latent memory.

##### D.2 Text-only Case Studies

- Table 20 presents two additional text-only cases. Both require multi-hop composition rather than simple lexical matching: the first links a screenwriter to a Nicolas Cage film through two supporting facts, while the second links a person to a company and then to the company’s headquarters. In both cases, the required evidence chain is ranked at the top, and the final answer is recovered from a compact retrieved set.

Table 20: Two text-only case studies. Both examples achieve exact-match generation, and the retrieved evidence chain is fully covered by the top-ranked latent retrieval results.

- Case 1 Question: What screenwriter with credits for “Evolution” co-wrote a film starring Nicolas Cage and Téa Leoni?

Generation Gold answer: David Weissman. Predicted answer: David Weissman. EM: 1.0 Retrieved positives: 3/3.

Latent Memory

Rank Pos? (gold)

Retrieval score

Corresponding text of retrieved latent token CE loss

- 1 Y 0.507 David Weissman: His film credits include “The Family Man” (2000), “Evolution” (2001), and “When in Rome” (2010).

0.260

- 2 Y 0.447 The Family Man: The Family Man is a 2000 American romantic comedy-drama film ... starring Nicolas Cage and Téa Leoni.

0.089

- 3 Y 0.394 David Weissman: David Weissman is a screenwriter and director.

0.010

- Case 2 Question: Where is the company that Sachin Warrier worked for as a software engineer headquartered?

Generation Gold answer: Mumbai. Predicted answer: Mumbai. EM: 1.0 Retrieved positives: 2/2.

Latent Memory

Corresponding text of retrieved latent token CE loss

Rank Pos? (gold)

Retrieval score

- 1 Y 0.623 Tata Consultancy Services: Tata Consultancy Services Limited (TCS) is an Indian multinational information technology service company headquartered in Mumbai, Maharashtra.

0.342

- 2 Y 0.557 Sachin Warrier: He was working as a software engineer in Tata Consultancy Services in Kochi.

0.083

##### D.3 More Multimodal QA Case Studies

- Figure 11 provides additional WebQA examples beyond the main-text case study, showing the counting and comparison ability in multimodal reasoning. These examples illustrate how Latent Memory retrieves from a unified text-image candidate pool and then conditions the frozen LLaVA-1.5-

- 13B generator on the retrieved latent tokens. We include both successful and challenging multimodal cases to show the qualitative behavior behind the aggregate results: image-grounded examples highlight whether visual evidence can be preserved after compression, while text-grounded examples show how textual facts are selected and used under the same retrieval interface.

###### Multimodal Context Memory

[Figure 107]

[Figure 108]

Text: Campbell County: Wyoming Campbell County is a county in the U.S. state of Wyoming. As of the 2010 United States Census, the population was 46,133, making it the third-most populous county in Wyoming.

###### Image:

Image:

.....

Text: Campbell County, Wyoming: As of the 2010 United States Census, the population was 46,133, making it the third-most ...

Wright Town Hall in Wright, Wyoming

Caption: Wright Town Hall in Wright, Wyoming

Caption: Campbell County Courthouse in Gillette, Wyoming

Text: .....

[Figure 109]

Input

###### Retrievable Latent Memory

Output

① Latent Memory Construction

LLM/VLM with Trainable LoRA for Compression

[Figure 110]

[Figure 111]

Output

[Figure 112]

Question: "How many flag poles are there near the entrance to the Campbell County Courthouse in Gillette, Wyoming??"

Frozen LargerSize Generation

two

[Figure 113]

Input

[Figure 114]

LLM/VLM

② Retrieve based on Query

Latent context 1: 1

③ Generation with Latent Tokens

Reconstructed Text Context in −

Reconstructed Context (Positive) Reconstructed Context (Negative)

[Figure 115]

[Figure 116]

Image:

###### Image:

Input

Text: Campbell County: Wyoming... (CE = 0.7) Text: Campbell County, Wyoming: A.... (CE = 0.9) Text: .....

LLM with Trainable LoRA/ MLP for Decoding

[Figure 117]

Output

[Figure 118]

④ Latent Token Reconstruction (Optional)

Title: Campbell.. (CE = 0.6)

Title: Wright Tow.. (CE = 0.3)

###### Multimodal Context Memory

[Figure 119]

[Figure 120]

Image:

###### Image:

Text: Bentley Mulsanne (2010): The Mulsanne W.O. Edition was presented at the 2018 Geneva Motor Show as a 100th anniversary celebration of the foundin...

..... Text: Bentley Mulsanne (1980–1992): The BMW Goldfisch V16 engine was tested in the Bentley Mulsanne a Text: .....

Caption: Bentley Mulsanne

Caption: 1967 Cadillac Series 75..

[Figure 121]

Input

###### Retrievable Latent Memory

Output

① Latent Memory Construction

LLM/VLM with Trainable LoRA for Compression

[Figure 122]

[Figure 123]

Output

[Figure 124]

Question: "Which car has rounder headlights, the 1967 Cadillac Series or the Bentley Mulsanne?"

Frozen LargerSize Generation

Bentley Mulsanne

[Figure 125]

Input

[Figure 126]

LLM/VLM

1

② Retrieve based on Query

Latent context:

③ Generation with Latent Tokens

Latent context:

Reconstructed Text Context in −

Reconstructed Positive Image Context

[Figure 127]

###### Image:

###### Image:

Input

[Figure 128]

Text: Bentley Mulsanne (2010): The Mulsanne W.O. Edition was presented at the 2018 Geneva Motor Show as .....(CE=1.8)

LLM with Trainable LoRA/ MLP for Decoding

[Figure 129]

Output

[Figure 130]

④ Latent Token Reconstruction (Optional)

Title: Bentley Mulsanne...

Title: 1967 Cadillac Se.....

Figure 11: Additional multimodal WebQA case studies with LLaVA-1.5-13B. Each example uses the same Latent Memory retrieval-and-generation pipeline as the main experiments.

### E Baselines, Datasets, and Licenses

Datasets. Table 21 summarizes all datasets used in the paper. For the main text-only setting, Latent Memory is trained on the HotpotQA training split and evaluated on HotpotQA validation, 2WikiMultihopQA, and MuSiQue without additional task-specific fine-tuning for the out-of-domain datasets. The generalization experiments further evaluate the same text checkpoint on NQ, TriviaQA, Qasper, and WICE to cover open-domain factoid QA, scientific-document QA, and claim verification.

For multimodal QA, WebQA is used for both training and evaluation under its public split; evaluation is reported separately for image-grounded and text-grounded questions, while retrieval is performed over the unified text-image candidate pool. SlideQA is a pure visual dataset covering slides and detail capture. We hope to find more multimodal datasets similar to WebQA that involve retrieval and reasoning, but unfortunately, this is almost the only one. The rest of the datasets often contain data outside the Latent Memory scope, such as tables and videos.

As shown in Table 20 and case study Figures, all evidence is processed in “Title: Sentence“ form for the Text-only setting, “Title: Evidence“ and “Caption: Image“ for the multimodal setting.

Table 21: Coverage of the datasets used in this work.

Dataset Knowledge / Task Domain Primary Source HotpotQA [67] Multi-hop entity-centric QA with

Encyclopedic factual knowledge

Wikipedia articles and supporting sentences

supporting-fact grounding

2WikiMultihopQA [68]

Cross-page factual reasoning and relation composition

Encyclopedic factual knowledge

Wikipedia-based evidence chains

Multi-paragraph textual QA collections

MuSiQue [69] More compositional multi-hop QA over multiple supporting paragraphs

Broad factual knowledge

WebQA [50] Unified text-image QA with visually grounded evidence

Web knowledge, mixed textual and visual evidence

Web text and web images

SlideQA Multimodal slide-domain QA with text and visual evidence

Slide / presentation understanding

Slide pages, associated visual regions, and textual metadata

NQ [70] Open-domain factoid QA, often short-answer retrieval

Open-domain factual knowledge

Search queries paired with Wikipedia evidence

TriviaQA [71] Trivia-style factoid QA with strong entity coverage

Open-domain factual knowledge

Trivia questions with web / Wikipedia evidence

Qasper [72] Document-grounded QA over scientific papers

Scientific document understanding

Research papers with caption, section, and paragraph structure

WICE [73] Evidence-centered claim verification and explanation

Fact verification / evidence reasoning

Claim-evidence collections built from textual evidence sources

Baseline settings. For text-only QA, the main table uses the same fixed Meta-Llama-3-8BInstruct generation model and the same question prompt format. Full Context concatenates all available context sentences in the sample and sends them directly to the generator. LLMLingua [28] compresses the raw text context first, and the compressed text is then inserted into the same generator prompt. BM25 Retrieval [22] ranks candidate sentences with sparse lexical matching and feeds the top-k raw sentences to the generator. Dense Retrieval [21] ranks the same candidate sentence pool with all-MiniLM-L6-v2 sentence embeddings and also feeds top-k raw sentences. Qwen3-Embedding [51] uses the Qwen3 text embedding model as a stronger dense retriever, with generation still performed from retrieved raw text. Since a series of embedding models, such as the Qwen-3-Embedding model, are usually pre-trained for our in-domain scenario (Wikipedia), fine-tuning on our chosen in-domain environment could disrupt the balance; so we believe that direct comparison with Latent Memory for retrieval is relatively fair.

Because xRAG [32] and CLaRa [33] are built around Mistral-style latent-context generation, we only compare to their pretrained models on the Mistral-7B-Instruct setting in Table 2. For xRAG, we use the pretrained Hannibal046/xrag-7b generator-side checkpoint together with the pretrained Salesforce/SFR-Embedding-Mistral retrieval encoder. For CLaRa, we use the released pretrained CLaRa checkpoints at the reported 16× compression settings (which are more suitable for our sentence-level evidence). Our Mistral-setting Latent Memory uses LLaMA-3.2-1B-Instruct as both the compressor/encoder and the reconstruction decoder, while the answer generator is frozen Mistral-7B-Instruct.

For multimodal QA, all baselines retrieve from the same unified WebQA candidate pool [50] containing both text facts and images. We report two frozen generator families: LLaVA-1.5-13B in the main WebQA setting and Gemma-3-12B-Instruct in Appendix C.6. In the implementation, the LLaVA setting is run through scripts/baselines_llava.py, while the Gemma setting is run through scripts/baselines_qwen.py, whose generator wrapper also supports model_type=gemma3. For Gemma, the same system instruction is prepended inside the user message because the Gemma chat template used by the processor does not take a separate system role in our code. Full Context passes all candidate text and images to the generator, which is often expensive because images expand through the model’s visual frontend. BM25 Retrieval [22] performs sparse retrieval over textual fields; if an image candidate is retrieved through its title/caption metadata, the raw image is still provided to the VLM for generation. Dense Retrieval [21] uses textual surrogates for retrieval: text facts are encoded directly, while images are represented by their titles/captions for scoring; after retrieval, the original raw image is sent to the generator. Qwen3-VL-Embedding [25], and Nemo Retriever [52] are multimodal retrieval baselines that rank text-image candidates with pretrained vision-language retrieval models, but the generator still consumes the retrieved raw text or image rather than the retrieval embedding. This keeps the retrieval candidate pool unified while making the raw-evidence baselines comparable to Latent Memory.

For all retrieval baselines, k denotes the number of retrieved evidence items and is matched to the corresponding Latent Memory setting when reported. Token counts always measure the generatorside input budget after constructing the final prompt. Thus, text retrieval baselines count the retrieved text tokens, multimodal baselines count the effective tokens consumed after visual processing, and Latent Memory counts the projected Latent Memory tokens inserted into the frozen generator.

Licenses. Table 22 summarises the main datasets, pretrained models, and baselines used in this work. We report the license or usage terms stated on the official upstream release pages that we could verify directly. When an official public release does not state a license, we mark it as not stated rather than inferring one. For benchmarks that redistribute web pages or images, downstream use must additionally comply with the rights of the original content providers.

Table 22: Summary of datasets, baselines, and upstream license provenance.

Item Type License / Terms Source

HotpotQA [67] Dataset CC BY-SA 4.0 Hugging Face: hotpotqa/hotpot_qa 2WikiMultihopQA [68] Dataset Apache License 2.0 Hugging Face: xanhho/2WikiMultihopQA MuSiQue [69] Dataset Available Online Hugging Face dataset cards:

bdsaglam/musique WebQA [50] Dataset Available Online WebQA official projects NQ [70, 74] Dataset GitHub: TIGER-Lab/LongRAG TriviaQA [71] Dataset Apache License 2.0 in the Hug-

Hugging Face: mandarjoshi/trivia_qa

ging Face release used by our data pipeline

Qasper [72] Dataset CC BY 4.0 Hugging Face: allenai/qasper WICE [73] Dataset Available Online WiCE project page by the authors and mirrored

dataset pages SlideQA Dataset Available Online Hugging Face:

NTT-hil-insight/SlideVQA

BM25 Retrieval [22] Baseline Available Online Standard sparse retrieval baseline Dense Retrieval [21] Baseline Apache License 2.0 Hugging Face:

sentence-transformers/all-MiniLM-L6-v2

Qwen3-Embedding [51]

Baseline Apache License 2.0 Hugging Face:

Qwen/Qwen3-Embedding-0.6B

Qwen3-VLEmbedding [25]

Baseline Apache License 2.0 Hugging Face:

Qwen/Qwen3-VL-Embedding-8B

Nemo Retriever [52] Baseline Apache License 2.0 Hugging Face:

nvidia/llama-nemotron-embed-vl-1b-v2 LLMLingua [28] Baseline Apache License 2.0 Microsoft LLMLingua repository xRAG [32] Baseline Available Online Pretrained checkpoint:

Hannibal046/xrag-7b; retrieval encoder: Salesforce/SFR-Embedding-Mistral

CLaRa [33] Baseline Not stated Released pretrained CLaRa checkpoints for 16× compression settings

Llama-3.2-1B-Instruct [48]

Model Llama 3.2 Community License Hugging Face / Meta model card:

meta-llama/Llama-3.2-1B-Instruct

Meta-Llama-3-8BInstruct [48]

Model Meta Llama 3 Community License

Hugging Face / Meta model card: meta-llama/Meta-Llama-3-8B-Instruct

Mistral-7B-Instruct [49] Model Apache License 2.0 Hugging Face / Mistral AI model release:

mistralai/Mistral-7B-Instruct-v0.3

LLaVA-1.5-7B [12] Model Llama 2 Community License Hugging Face: liuhaotian/llava-v1.5-7b; LLaVA website

LLaVA-1.5-13B [12] Model Llama 2 Community License Hugging Face: liuhaotian/llava-v1.5-13b; LLaVA website

Gemma-3-4B-PT [13] Model Gemma Terms of Use / Gemma license

Hugging Face: google/gemma-3-4b-pt; Google AI for Developers Gemma Terms

Gemma-3-12B-IT [13] Model Gemma Terms of Use / Gemma license

Hugging Face: google/gemma-3-12b-it; Google AI for Developers Gemma Terms

CLIP ViT-L/14-336 [45]

Model MIT for OpenAI CLIP code/release

GitHub: openai/CLIP; Hugging Face mirror: openai/clip-vit-large-patch14-336

