# arXiv:2410.12788v3[cs.CL]21May2025

## Meta-Chunking: Learning Text Segmentation and Semantic Completion via Logical Perception

Jihao Zhao1 Zhiyuan Ji1 Yuchen Feng2 Pengnian Qi2 Simin Niu1 Bo Tang2 Feiyu Xiong2 Zhiyu Li2∗

1School of Information, Renmin University of China, Beijing, China 2Institute for Advanced Algorithms Research, Shanghai, China

### Abstract

While Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for boosting large language models (LLMs) in knowledge-intensive tasks, it often overlooks the crucial aspect of text chunking within its workflow. This paper proposes the Meta-Chunking framework, which specifically enhances chunking quality through a dual strategy that identifies optimal segmentation points and preserves global information. Initially, breaking limitations of similaritybased chunking, we design two adaptive chunking techniques based on uncertainty, namely Perplexity Chunking and Margin Sampling Chunking, by utilizing the logical perception capabilities of LLMs. Given the inherent complexity across different texts, we integrate meta-chunk with dynamic merging, striking a balance between fine-grained and coarse-grained text chunking. Furthermore, we establish the global information compensation mechanism, encompassing a two-stage hierarchical summary generation process and a three-stage text chunk rewriting procedure focused on missing reflection, refinement, and completion. These components collectively strengthen the semantic integrity and contextual coherence of chunks. Extensive experiments demonstrate that Meta-Chunking effectively addresses challenges of the chunking task within the RAG system, providing LLMs with more logically coherent text chunks. Additionally, our methodology validates the feasibility of implementing high-quality chunking tasks with smaller-scale models, thereby eliminating the reliance on robust instruction-following capabilities. Our code is available at https://github.com/IAAR-Shanghai/Meta-Chunking.

### 1 Introduction

Retrieval-augmented generation (RAG), as a technical paradigm that integrates information retrieval with generative models, effectively mitigates inherent limitations of large language models (LLMs), such as data freshness [1], hallucinations [2–4], and the lack of domain-specific knowledge [5,6]. As the core architecture for knowledge-intensive tasks [7], its efficacy is fundamentally constrained by the optimization boundary of the synergistic retrieval-generation mechanism, because the quality of retrieved text chunks directly determines the performance ceiling [8–10], and the foundation of this process lies in the text chunking. Optimally segmenting documents into semantically complete and coherent chunks not only enhances the generation accuracy of LLM by concentrating information and reducing redundancy [11,12], but also significantly improves the processing efficiency of the system while reducing computational resource consumption [13].

As a preprocessing unit within the RAG system, this process is often overlooked and has consequently received insufficient in-depth investigation [14–16]. Current mainstream methods primarily rely

∗Corresponding author: lizy@iaar.ac.cn

Preprint. Under review.

on rules or semantic similarity [17–19]. Although these approaches are engineering-friendly, they typically fail to capture the nuanced logical dependencies between sentences. As illustrated in Figures 3 and 4, chunks with logical progression are often incorrectly segmented due to low cosine similarity, leading to retrieval results deviating from the core semantic unit. Recently proposed LumberChunker [20], while invoking LLM APIs to more accurately identify content divergence points, requires models with advanced instruction-following capabilities, thereby incurring substantial resource costs. This raises a practical question: How can we fully utilize the powerful reasoning capabilities of LLMs while efficiently accomplishing the text chunking task at a lower cost?

Inspired by these observations, this paper proposes the Meta-Chunking framework, which synergistically optimizes the logical perception capabilities of LLMs with information integrity constraints to specifically address the issue of logical discontinuities in text chunking. We design two uncertaintybased adaptive boundary detection algorithms: Perplexity (PPL) Chunking and Margin Sampling (MSP) Chunking. These algorithms leverage the implicit and explicit evaluation capabilities of LLMs for logical coherence, respectively, to identify chunk boundaries. Meanwhile, the resulting meta-chunks are treated as independent logical units, and a dynamic merging strategy is introduced to achieve a balance between fine-grained and coarse-grained segmentation. On the other hand, to further enhance the cognitive completeness of text chunks, we construct an information compensation pipeline: (1) Implementing a missing-aware rewriting mechanism during the post-chunking phase, which systematically repairs semantic discontinuities caused by segmentation through a three-stage optimization process of missing reflection, refinement, and completion. (2) Adopting a two-layer summarization technique for each text chunk, we extract core knowledge anchors from both the document-level macro themes and the paragraph-level micro semantics, thereby further improving the global recall rate of chunks. It is noteworthy that due to the scarcity of relevant datasets in the chunking domain, we carefully prepare training data for aforementioned methods and fine-tune small language models (SLMs) to achieve efficient application.

We summarize contributions of this work as follows:

- • Through lightweight chunking algorithm design, the logical analysis capability of LLMs is decoupled into computable the PPL features and MSP indicators, achieving identification of textual logical boundaries and dynamic balance of chunking granularity.
- • We establish a information compensation mechanism that collaboratively executes through a three-stage missing-aware rewriting process and a two-stage context-aware summary generation, repairing the semantic discontinuities in text chunks.
- • To verify the effectiveness of our proposed Meta-Chunking framework, we conduct multidimensional experiments and analyses using five datasets. The results indicate that this framework delivers more logically coherent text chunks to the RAG system, demonstrating the feasibility of achieving high-quality chunking tasks on SLMs.

### 2 Related Works

##### 2.1 Text Chunking in RAG

By expanding the input space of LLMs through introducing retrieved text chunks [21,22], RAG significantly improves the performance of knowledge-intensive tasks [23]. Text chunking plays a crucial role in RAG, as ineffective chunking strategies can lead to incomplete contexts or excessive irrelevant information, thereby hurting the performance of question answering (QA) systems [24]. Besides typical granularity levels like sentences or paragraphs [19,25], there are other advanced methods available. [26] introduced a novel retrieval granularity called Proposition, which is the smallest text unit that conveys a single fact. This method excels in fact-based texts like Wikipedia. However, it may not perform ideally when dealing with content that relies on flow and contextual continuity, such as narrative texts, leading to the loss of critical information. Meanwhile, LumberChunker [20] iteratively harnesses LLMs to identify potential segmentation points within a continuous sequence of textual content, showing some potential for LLMs chunking. However, this method demands a profound capability of LLMs to follow instructions and entails substantial consumption when employing the Gemini model.

- 2.2 Uncertainty Theory of LLMs

Quantifying uncertainty in LLMs is currently an active research direction in the field of artificial intelligence [27–29]. Information theory provides a solid theoretical foundation and a suite of mathematical tools to measure the inherent degree of uncertainty in probability distributions or signals. For instance, Entropy is employed to gauge the randomness of a model’s prediction for the next token [30]. Semantic Entropy further extends this concept to encompass clusters of semantically similar generated sequences [31]. Perplexity [32], a classic metric for evaluating LLMs, indirectly reflects the strength of logical relationships between sentences by measuring the model’s surprise regarding sequential data. Additionally, Mutual Information is capable of quantifying the amount of information shared between different random variables, making it useful for assessing cognitive uncertainty among various outputs of different models [33].

- 3 Methodology

- 3.1 Text Chunking of Meta-Chunking

Our approach is grounded in a core principle: allowing variability in chunk size to more effectively capture and maintain the logical integrity of content. This dynamic adjustment of granularity ensures that each segmented chunk contains a complete and independent expression of ideas, thereby avoiding breaks in the logical chain during the segmentation process. This not only enhances the relevance of document retrieval but also improves content clarity.

As illustrated in Figure 1, our method integrates the advantages of traditional text segmentation strategies, such as adhering to preset chunk length constraints and ensuring sentence structural integrity, while enhancing the ability to guarantee logical coherence during the segmentation process. We refer to each text chunk obtained through segmentation as a Meta-Chunk, which consists of a collection of sequentially arranged sentences within a paragraph. These sentences not only have semantic relevance but, more importantly, also contain profound linguistic logical connections, including but not limited to general-specific, parallel, sequential, and illustrative relationships, as shown in Figure 4. Through observation, it is found that there are often tight logical connections between consecutive sentences within a meta-chunk. However, these sentences exhibit low semantic similarity due to their divergent content representations. As mentioned in [34], semantic chunking has failed to demonstrate advantages across multiple experimental paradigms. We believe that this phenomenon is closely related to the original theoretical modeling intentions of semantic similarity algorithms. These methods essentially model the degree of semantic overlap between texts to quantify the correlation between two paragraphs or between a sentence and a paragraph. Nevertheless, sentences at the micro level that have logical associations but express different content limit their applicability. The detailed analysis is presented in Appendix B. In order to address the aforementioned issue, we implement the following strategies based on the uncertainty theory in LLMs.

##### 3.1.1 Perplexity Chunking

Given a text, the initial step involves segmenting it into a collection of sentences denoted as (x1,x2,...,xn), with the ultimate goal being to further partition these sentences into several chunks, forming a new set (X1,X2,...,Xk), where each chunk comprises a coherent grouping of the original sentences. We split the text into sentences and use the model to calculate the PPL of each sentence xi based on the preceding sentences:

K k=1PPLM(tik|ti<k,t<i)

PPLM(xi) =

(1)

K

where K represents the total number of tokens in xi, tik denotes the k-th token in xi, and t<i signifies all tokens that precede xi. To locate the key points of text segmentation, the algorithm further analyzes the distribution characteristics of PPLseq = (PPLM(x1),PPLM(x2),...,PPLM(xn)), particularly focusing on identifying minima.

Our primary focus is on two types of minimum points: when the PPL on both sides of a point are higher than at that point, and the difference on at least one side exceeds the preset threshold θ; or when the difference between the left point and the point is greater than θ and the right point equals the point value. These minima are regarded as potential chunk boundaries. If the text

[Figure 1]

- Figure 1: Overview of the entire process of Meta-Chunking. Each circle represents a complete sentence, and the sentence lengths are not consistent. The vertical lines indicate where to segment. Circles with the same background color represent a meta-chunk, which is dynamically combined to make the final chunk length meet user needs.

exceeds the processing range of LLMs or device, we strategically introduce a key-value (KV) caching mechanism. Specifically, the text is first divided into several parts according to tokens, forming multiple subsequences. As the PPL calculation progresses, when the GPU memory is about to exceed the server configuration or the maximum context length of LLMs, the algorithm appropriately removes KV pairs of previous partial text, thus not sacrificing too much contextual coherence.

##### 3.1.2 Margin Sampling Chunking

It is noteworthy that LumberChunker [20] encounters difficulties when applied to smaller models, primarily due to its requirement for generating text in a specified format and subsequent regular expression extraction. To address this limitation, we introduce the MSP strategy that analyzes the marginal probability distribution during model decision-making to determine whether chunking should be performed. The method can be formulated as:

′

′

MarginM(xi) = PM y = k1|Prompt(xi,X

) − PM y = k2|Prompt(xi,X

) (2)

where (k1,k2) indicates a binary decision between yes or no for a segmentation judgment. Prompt(xi,X

′

′

, regarding whether they should be merged, where X

) represents forming an instruction between xi ∈ {xl}nl=1 and X

′

encompasses either a single sentence or multiple sentences. Through the probability PM obtained by model M, we can derive the probability difference MarginM(xi) between the two options. Subsequently, by contrasting MarginM(xi) with the threshold θ, a conclusion can be drawn regarding whether the two sentences should be segmented. Moreover, setting a threshold for decision criteria is a common requirement across all strategies, and we bring in a dynamic threshold mechanism. Specifically, in the initialization phase of the θ, we assign it a starting value of 0. Subsequently, we fine-tune θ by keeping track of historical MarginM(xi) values and computing their mean, thereby enabling more flexible adjustment of chunking.

##### 3.1.3 Dynamic Merging

To address diverse chunking needs of users, merely adjusting the threshold to control chunk size sometimes leads to uneven chunking sizes as the threshold increases, as shown in Appendix F.

Therefore, we propose a strategy combining meta-Chunk with dynamic merging, aiming to flexibly respond to varied chunking requirements. Firstly, we employ either PPL Chunking or MSP chunking to partition the document into a series of meta-chunks, denoted as (c1,c2,...,cα). Traditional chunking methods treat sentences as independent logical units, whereas we adopt meta-chunks as independent logical units. Subsequently, according to the user-specified chunk length L, we iteratively merge adjacent meta-chunks until the total length satisfies or approximates the requirement. Specifically, if len(c1,c2,c3) = L or len(c1,c2,c3) < L while len(c1,c2,c3,c4) > L, then c1,c2,c3 are regarded as a complete chunk.

##### 3.2 Semantic Completion of Meta-Chunking

To address the semantic gap issue arising from the loss of contextual information in text chunking, we propose a globally enhanced text rewriting and summary generation mechanism. Specifically, we leverage a LLM as a discriminator to examine whether each chunk suffers from semantic deficiencies, and if so, initiate the rewriting process in Section 3.2.1. After handling these deficiencies, we perform the summary generation in Section 3.2.2 on all chunks to further improve recall, laying a solid foundation for ultimately enhancing QA performance. The detailed design scheme is elaborated in Appendix C.

##### 3.2.1 Globally Augmented Text Chunk Rewriting

Preprocessing (Optional) For extremely long documents that present challenges for full ingestion by LLM, an inter-chunk relevance analysis leveraging semantic embeddings is employed. This process involves generating vector representations for each chunk using a semantic similarity model and quantifying the strength of their semantic associations by calculating the cosine similarity between these vectors. Such an approach facilitates the identification of potential contextual information pertinent to the current chunk.

- Stage 1 (Missing Reflection) Utilizing an LLM, and incorporating the potentially relevant information identified during the preprocessing phase, each chunk undergoes an in-depth reflective analysis. The core task is to explicitly identify which premises, backgrounds, related facts, or conclusive statements are missing from the current chunk. The LLM should comprehensively list the areas where information is missing and specify the information that needs to be supplemented.
- Stage 2 (Missing Refinement) This phase is dedicated to score and filter the potentially missing information detected in the previous stage. We aim to prevent the introduction of irrelevant or erroneous supplementary content, thereby ensuring the precision of the augmentation process.
- Stage 3 (Missing Completion) Based on the refined omission loci and the requisite information confirmed in the preceding stage, the LLM is prompted to integrate these informational segments with the current text chunk. The goal is to generate a new chunk that is contextually seamless, semantically natural, and effectively achieves robust inter-chunk information fusion.

##### 3.2.2 Context-Aware Summary Generation

The primary objective of this part is to generate a concise summary, enriched with global information, for each text chunk, thereby further augmenting the contextual awareness of the chunk.

- (1) The model utilizes global information to generate a supplementary summary for the target text chunk. This process is designed to compensate for the discourse background and external relational information that the chunk may lack due to segmentation.
- (2) With respect to the content of the chunk itself, the model independently generates a local summary that encapsulates its core viewpoint. Subsequently, the aforementioned two summaries are fused and refined into an enhanced summary sentence that can articulate the content of the chunk from a global perspective.

To support the proposed rewriting and summary generation components, we construct 20,000 training data samples for each of them, adhering to the process described above. Meanwhile, we opt for full fine-tuning of the SLM. For an input sequence X and a target output sequence Y = (y1,y2,...,yT), the loss function is defined as:

T

1 N

log P(yt|y<t,X;θ) (3)

L(θ) = −

t=1

where P(yt|y<t,X;θ) represents the probability that the model predicts the true target token yt given the input X and the previously generated prefix y<t, θ denotes the model parameters, and N is the number of samples in a batch. Detailed information on the dataset construction and hyperparameter configurations for fine-tuning can be found in Appendix C.

- 3.3 Theoretical Analysis of PPL Chunking

LLMs are designed to learn a distribution Q that approximates the empirical distribution P from sample texts. To quantify the closeness between these two distributions, cross-entropy is typically employed as a metric. Under the discrete scenario, cross-entropy of Q relative to P is formally defined as follows:

H(P,Q) = Ep[−logQ] = −

x

P(x)log Q(x) = H(P) + DKL(P||Q) (4)

where H(P) represents the empirical entropy, and DKL(P||Q) is the Kullback-Leibler (KL) divergence between Q and P. The PPL of LLMs, mathematically speaking, is defined as:

PPL(P,Q) = 2H(P,Q) (5)

It is essential to notice that, since H(p) is unoptimizable and bounded as shown in Appendix A, what truly impacts the discrepancy in PPL calculations across different LLMs is the KL divergence, which serves as a metric to assess the difference between distributions. The greater the KL divergence is, the larger the disparity between two distributions signifies. Furthermore, high PPL indicates the cognitive hallucination of LLMs towards the real content, and such portions should not be segmented.

On the other hand, [35] approximates the entropy of any language through a function

GK = −

Tk

P(Tk)log2 P(tk|Tk−1)

= −

Tk

P(Tk)log2 P(Tk) +

Tk−1

P(Tk−1)log2 P(Tk−1) (6)

where Tk represents k consecutive tokens (t1,t2,...,tk) in a text sequence, entropy can then be expressed as

H(P) = lim

K→∞

GK (7) Then, based on the proof in Appendix A that GK+1 ≤ GK for all K ≥ 1, we can derive

G1 ≥ G2 ≥ ··· ≥ lim

K→∞

GK = H(P) (8)

By combining equation (4) and (8), we observe that for large-scale text processing tasks, increasing the context length tends to reduce the cross-entropy or PPL, a phenomenon that reflects the ability of LLMs to make more effective logical inferences and semantic understandings after capturing broader contextual information. Consequently, during PPL Chunking experiments, we maximize the input of longer text sequences to LLMs, anticipating more substantial performance gains.

- 4 Experiment

- 4.1 Datasets and Metrics

We conduct a comprehensive evaluation on five datasets, focusing on both Chinese and English languages, and covering multiple metrics. The LongBench benchmark [36] comprises various datasets, among which we exploit three English datasets and one Chinese dataset, covering both single-hop and multi-hop QA tasks, with evaluations conducted based on F1 and chunking time metrics. The CRUD [19] is a novel benchmark designed for evaluating RAG systems, employing BLEU series, ROUGE-L, and BERTScore metrics for assessment.

- Table 1: Main experimental results are presented in four QA datasets. The best result is in bold, and the second best result is underlined.

Dataset 2WikiMultihopQA Qasper MultiFieldQA-en MultiFieldQA-zh Chunking Method F1 Time F1 Time F1 Time F1 Time

Baselines with rule-based or similarity-based chunking Original 11.89 0.21 9.45 0.13 29.89 0.16 22.45 0.06 Llama_index 11.74 8.12 10.15 5.81 28.30 6.25 21.85 5.53 Similarity Chunking 12.00 416.45 9.93 307.05 29.19 318.41 22.39 134.80 Dense X Retrieval 5.49 57633.07 8.23 39762.54 29.72 41789.49 - LumberChunker (Qwen2.5-14B) 13.28 5244.91 9.44 3777.03 33.09 3832.04 24.35 7228.78

Perplexity Chunking and Margin Sampling Chunking PPL Chunking (Qwen2-0.5B) 13.56 140.54 9.62 65.45 31.02 79.72 23.52 64.02 PPL Chunking (Qwen2-7B) 13.41 736.69 9.79 486.48 32.35 523.74 23.20 424.96 PPL Chunking (Baichuan2-7B) 12.98 858.99 10.04 569.72 32.55 632.80 23.36 569.72 MSP Chunking (Qwen2-0.5B) 12.13 1471.31 10.15 1071.34 31.91 1103.51 23.69 888.19 MSP Chunking (Qwen2-7B) 13.20 8781.82 11.37 5755.79 33.56 6287.31 24.45 9746.76

##### 4.2 Baselines

We primarily compare Meta-Chunking with two types of methods, namely rule-based chunking and dynamic chunking, noting that the latter incorporates both semantic similarity models and LLMs. The original rule-based method simply divides long texts into fixed-length chunks, disregarding sentence boundaries. The Llama_index method [18] offers a more nuanced approach, balancing the maintenance of sentence boundaries while ensuring that token counts in each segment are close to a preset threshold. On the other hand, similarity chunking [37] utilizes sentence embedding models to segment text based on semantic similarity, effectively grouping highly related sentences together. Dense X Retrieval [26] introduces a new retrieval granularity called propositions, which condenses and segments text by training an information extraction model. Alternatively, LumberChunker [20] employs LLMs to predict optimal segmentation points within the text. These methods exhibit unique strengths in adapting to the context and structure of texts.

##### 4.3 Experimental Settings

We primarily use Qwen2-0.5B2, Qwen2-7B1 and Baichuan2-7B3 for Meta-Chunking, with Qwen2.53B1 being employed for fine-tuning. Without additional annotations, all language models used in this paper adopt chat or instruction versions. When chunking, the default parameter configurations of the models are adopted. For evaluation, Qwen2-7B is employed with the following settings: top_p =

- 0.9, top_k = 5, temperature = 0.1, and max_new_tokens = 1280. When conducting QA, the system necessitates dense retrievals from the vector database, with top_k set to 8 for CRUD, and 5 for LongBench. Text chunking is performed on the NVIDIA H800, while model training and evaluation are carried out on the NVIDIA A800. To control variables, we maintain consistent chunk lengths for various chunking methods across each dataset. Detailed experimental setup information can be found in Appendix D.

### 5 Results and Analysis

##### 5.1 Main Results

Comparison against Baselines. We systematically evaluate the performance of five baseline methods, with the results presented in Table 1. Compared with traditional rule-based and semantic chunking methods, as well as the state-of-the-art LumberChunker method which leverages Qwen2.514B, MSP Chunking exhibits improved and more stable performance. Meanwhile, PPL Chunking demonstrates advantages in balancing performance and processing time. Furthermore, our approach mitigates the current dilemma where text chunking heavily relies on strong instruction-following

- 2https://huggingface.co/Qwen
- 3https://huggingface.co/baichuan-inc

- Table 2: Performance of global information compensation mechanism via text chunk rewriting and summary generation based on chunking results.

Chunking Method BLEU-1 BLEU-2 BLEU-3 BLEU-4 BLEU-Avg ROUGE-L BERTScore

Original 0.3515 0.2788 0.2340 0.1997 0.2548 0.4213 0.8489 Llama_index 0.3620 0.2920 0.2480 0.2134 0.2682 0.4326 0.8521 Similarity Chunking 0.3382 0.2692 0.2257 0.1931 0.2462 0.4131 0.8442 LumberChunker 0.3456 0.2781 0.2343 0.2011 0.2542 0.4160 0.8514 Qwen2.5-14B 0.3650 0.2928 0.2469 0.2126 0.2679 0.4351 0.8549 Qwen2.5-72B 0.3722 0.2985 0.2526 0.2170 0.2743 0.4405 0.8550 PPL (Qwen2-7B) 0.3724 0.3012 0.2561 0.2206 0.2774 0.4445 0.8584 PPL (Baichuan2-7B) 0.3816 0.3096 0.2631 0.2267 0.2847 0.4520 0.8603 Meta-Chunking 0.3924 0.3207 0.2744 0.2378 0.2963 0.4614 0.8657

w/o Rewriting 0.3909 0.3179 0.2711 0.2349 0.2934 0.4590 0.8622 w/o Summary 0.3834 0.3122 0.2659 0.2301 0.2876 0.4562 0.8635

capabilities. It can even be integrated with a 0.5B SLM without incurring a significant performance decline. This implies that the full potential of SLMs in text chunking tasks has not yet been entirely harnessed. Their notable efficiency and commendable performance warrant further exploration, positioning them as truly practical tools for chunking.

Effectiveness of Semantic Completion. To validate the effectiveness of our proposed metachunking framework, experiments are conducted on the CRUD benchmark. During the dataset preparation phase, we meticulously structure 20,000 samples for each of the two components through a rigorous processing pipeline. This dataset is then utilized to fine-tune the Qwen2.5-3B model, with the obtained comparative results illustrated in Table 2. Building upon chunking performance, the meta-chunking framework yield further enhancements to overall system. You can find a more in-depth discussion in Section 5.4 and Appendix C.

##### 5.2 Demystifying the Effect of Instruction-Following Capability

The experimental results in Section 5.1 preliminarily suggest that our method imposes weaker requirements on a model’s instruction-following capabilities. However, as pointed out in [38–40], prompts influence both the output and reasoning performance of LLMs. Therefore, we conduct a more thorough analysis of the interaction between a model’s chunking ability and instructions. As shown in Table 3, by comparing the base model with the instruction model, we find that the PPL Chunking exhibits greater emphasis on a model’s reasoning ability, without imposing stringent requirements on the capability to follow specific instructions. The MSP Chunking, conversely, due to dependency on prompts, emerges a certain degree of need for this ability. Furthermore, we design two types of prompts for MSP Chunking: a regular one and a more precise one, as detailed in Tables 6 and 7. From Figure 2, it can be observed that smaller models can benefit from more precise prompts, whereas larger models may experience a decline in performance when subjected to them.

Table 3: Performance comparison of LLMs chunking utilizing two types of Qwen2-7B. base represents the basic model, while inst. denotes the model fine-tuned with instructions.

Dataset 2Wiki Qasper MQA-en MQA-zh Chunking Method F1 F1 F1 F1

MSP Chunkingbase 12.02 9.86 32.24 21.55 MSP Chunkinginst. 13.20 11.37 33.56 24.45 PPL Chunkingbase 14.15 10.11 31.35 23.63 PPL Chunkinginst. 13.41 9.39 32.35 23.20

##### 5.3 Impact of Overlapping Chunking Strategies

As demonstrated in Table 4, we investigate the performance of several methods that support overlapping chunks, with their specific implementation details described in Appendix D. The dynamic overlap strategy of PPL Chunking assigns sentences located at the minima of the PPL distribution to both the preceding and subsequent chunks, thereby more effectively bridging semantic connections

[Figure 2]

- Figure 2: Performance comparison of MSP Chunking using two types of prompts across LLMs of different sizes.

between text chunks. Specifically, apart from a 1% improvement on the BERTScore, PPL Chunking overlap method achieves performance gains of 2%-3% across the remaining metrics.

Table 4: Performance of different methods on the CRUD benchmark with overlapping chunks.

Chunking Method Overlap BLEU-1 BLEU-2 BLEU-3 BLEU-4 BLEU-Avg ROUGE-L BERTScore

Original Fixed 0.3330 0.2641 0.2214 0.1881 0.2410 0.4060 0.8425 Llama_index Dynamic 0.3326 0.2645 0.2214 0.1890 0.2413 0.4039 0.8439 PPL (Qwen2-7B) Dynamic 0.3582 0.2898 0.2450 0.2097 0.2657 0.4308 0.8548 PPL (Baichuan2-7B) Dynamic 0.3656 0.2952 0.2497 0.2143 0.2705 0.4393 0.8549

##### 5.4 Rationale for Performance Gains from Text Chunk Rewriting

This section aims to elucidate the mechanism by which globally augmented text chunk rewriting enhances system performance. As illustrated in Table 5, we compare different types of text chunks under two distinct semantic retrievers by calculating the average similarity scores for the Top-8 retrieved chunks in response to queries. The results indicate that rewritten chunks exhibit superior alignment with the query intent, thereby facilitating the acquisition of content that is highly relevant to the questions. Furthermore, as depicted by experiments in Figure 5, rewritten chunks consistently represent a lower PPL across different LLMs. This phenomenon provides evidence that the global augmentation mechanism enables LLMs to better comprehend retrieved texts by optimizing contextual coherence.

Table 5: Comparison of average similarity scores for different text chunk types across two retrievers.

Metric AVGSim@8 Chunking Method bge-base-zh-v1.5 bge-large-zh-v1.5

Original 0.8053 0.8049 Llama_index 0.8071 0.8032 Similarity Chunking 0.7751 0.7682 LumberChunker 0.8084 0.8034 Append Summaries 0.8246 0.8161 Rewrite Chunks 0.8410 0.8426

### 6 Conclusion

Addressing issues of logical discontinuity and semantic incompleteness in text chunking, this paper proposes the Meta-Chunking framework, which establishes a systematic solution through the dual constraints of logical perception and information integrity optimization by LLMs. Specifically, we engineer two uncertainty-based adaptive boundary detection algorithms and introduce a dynamic merging strategy to enhance the logical completeness of chunking results. Furthermore, a collaborative information compensation mechanism is developed. It repairs semantic discontinuities caused by segmentation through globally missing-aware rewriting and context-aware summary generation. By autonomously constructing high-quality training datasets and fine-tuning SLMs, this algorithm can be efficiently deployed. Experimental results also corroborate that our framework can achieve higher-quality text chunking while being adaptable to SLMs. We anticipate that our insights will inspire further researches into text chunking, ultimately fostering the development of RAG systems.

### References

- [1] H. He, H. Zhang, and D. Roth, “Rethinking with retrieval: Faithful large language model inference,” arXiv preprint arXiv:2301.00303, 2022.
- [2] Y. Chen, Q. Fu, Y. Yuan, Z. Wen, G. Fan, D. Liu, D. Zhang, Z. Li, and Y. Xiao, “Hallucination detection: Robustly discerning reliable answers in large language models,” in Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, 2023, pp. 245–255.
- [3] G. Zuccon, B. Koopman, and R. Shaik, “Chatgpt hallucinates when attributing answers,” in Proceedings of the Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, 2023, pp. 46–51.
- [4] X. Liang, S. Song, Z. Zheng, H. Wang, Q. Yu, X. Li, R.-H. Li, F. Xiong, and Z. Li, “Internal consistency and self-feedback in large language models: A survey,” arXiv preprint arXiv:2407.14507, 2024.
- [5] X. Li, S. Chan, X. Zhu, Y. Pei, Z. Ma, X. Liu, and S. Shah, “Are chatgpt and gpt-4 generalpurpose solvers for financial text analytics? a study on several typical tasks,” arXiv preprint arXiv:2305.05862, 2023.
- [6] X. Shen, Z. Chen, M. Backes, and Y. Zhang, “In chatgpt we trust? measuring and characterizing the reliability of chatgpt,” arXiv preprint arXiv:2304.08979, 2023.
- [7] A. Lazaridou, E. Gribovskaya, W. Stokowiec, and N. Grigorev, “Internet-augmented language models through few-shot prompting for open-domain question answering,” arXiv preprint arXiv:2203.05115, 2022.
- [8] H. Li, Y. Su, D. Cai, Y. Wang, and L. Liu, “A survey on retrieval-augmented text generation,” arXiv preprint arXiv:2202.01110, 2022.
- [9] C.-H. Tan, J.-C. Gu, C. Tao, Z.-H. Ling, C. Xu, H. Hu, X. Geng, and D. Jiang, “Tegtok: Augmenting text generation via task-specific and open-world knowledge,” arXiv preprint arXiv:2203.08517, 2022.
- [10] W. Lin, R. Blloshmi, B. Byrne, A. de Gispert, and G. Iglesias, “Li-rage: Late interaction retrieval augmented generation with explicit signals for open-domain table question answering,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 2023, pp. 1557–1566.
- [11] S. Xu, L. Pang, H. Shen, and X. Cheng, “Berm: Training the balanced and extractable representation for matching to improve generalization ability of dense retrieval,” arXiv preprint arXiv:2305.11052, 2023.
- [12] W. Su, Y. Tang, Q. Ai, Z. Wu, and Y. Liu, “Dragin: Dynamic retrieval augmented generation based on the real-time information needs of large language models,” arXiv preprint arXiv:2403.10081, 2024.
- [13] M. Besta, A. Kubicek, R. Niggli, R. Gerstenberger, L. Weitzendorf, M. Chi, P. Iff, J. Gajda, P. Nyczyk, J. Müller et al., “Multi-head rag: Solving multi-aspect problems with llms,” arXiv preprint arXiv:2406.05085, 2024.
- [14] G. Sidiropoulos and E. Kanoulas, “Analysing the robustness of dual encoders for dense retrieval against misspellings,” in Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022, pp. 2132–2136.
- [15] Z. Zhuang, Z. Zhang, S. Cheng, F. Yang, J. Liu, S. Huang, Q. Lin, S. Rajmohan, D. Zhang, and Q. Zhang, “Efficientrag: Efficient retriever for multi-hop question answering,” arXiv preprint arXiv:2408.04259, 2024.
- [16] Y. Kim, H. J. Kim, C. Park, C. Park, H. Cho, J. Kim, K. M. Yoo, S.-g. Lee, and T. Kim, “Adaptive contrastive decoding in retrieval-augmented generation for handling noisy contexts,” arXiv preprint arXiv:2408.01084, 2024.
- [17] Q. Zhang, Q. Chen, Y. Li, J. Liu, and W. Wang, “Sequence model with self-adaptive sliding window for efficient spoken document segmentation,” in 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2021, pp. 411–418.
- [18] Langchain, https://github.com/langchain-ai/langchain, 2023.

- [19] Y. Lyu, Z. Li, S. Niu, F. Xiong, B. Tang, W. Wang, H. Wu, H. Liu, T. Xu, and E. Chen, “Crudrag: A comprehensive chinese benchmark for retrieval-augmented generation of large language models,” arXiv preprint arXiv:2401.17043, 2024.
- [20] A. V. Duarte, J. Marques, M. Graça, M. Freire, L. Li, and A. L. Oliveira, “Lumberchunker: Long-form narrative document segmentation,” arXiv preprint arXiv:2406.17526, 2024.
- [21] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval augmented language model pre-training,” in International conference on machine learning. PMLR, 2020, pp. 3929–3938.
- [22] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel et al., “Retrieval-augmented generation for knowledge-intensive nlp tasks,” Advances in Neural Information Processing Systems, vol. 33, pp. 9459–9474, 2020.
- [23] O. Ram, Y. Levine, I. Dalmedigos, D. Muhlgay, A. Shashua, K. Leyton-Brown, and Y. Shoham, “In-context retrieval-augmented language models,” Transactions of the Association for Computational Linguistics, vol. 11, pp. 1316–1331, 2023.
- [24] W. Yu, H. Zhang, X. Pan, K. Ma, H. Wang, and D. Yu, “Chain-of-note: Enhancing robustness in retrieval-augmented language models,” arXiv preprint arXiv:2311.09210, 2023.
- [25] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, and H. Wang, “Retrievalaugmented generation for large language models: A survey,” arXiv preprint arXiv:2312.10997, 2023.
- [26] T. Chen, H. Wang, S. Chen, W. Yu, K. Ma, X. Zhao, D. Yu, and H. Zhang, “Dense x retrieval: What retrieval granularity should we use?” arXiv preprint arXiv:2312.06648, 2023.
- [27] R. Zhang, H. Zhang, and Z. Zheng, “Vl-uncertainty: Detecting hallucination in large visionlanguage model via uncertainty estimation,” arXiv preprint arXiv:2411.11919, 2024.
- [28] Y. Li, R. Qiang, L. Moukheiber, and C. Zhang, “Language model uncertainty quantification with attention chain,” arXiv preprint arXiv:2503.19168, 2025.
- [29] L. Da, X. Liu, J. Dai, L. Cheng, Y. Wang, and H. Wei, “Understanding the uncertainty of llm explanations: A perspective based on reasoning topology,” arXiv preprint arXiv:2502.17026, 2025.
- [30] Z. Atf, S. A. A. Safavi-Naini, P. R. Lewis, A. Mahjoubfar, N. Naderi, T. R. Savage, and A. Soroush, “The challenge of uncertainty quantification of large language models in medicine,” arXiv preprint arXiv:2504.05278, 2025.
- [31] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, “Detecting hallucinations in large language models using semantic entropy,” Nature, vol. 630, no. 8017, pp. 625–630, 2024.
- [32] X. Liu, T. Chen, L. Da, C. Chen, Z. Lin, and H. Wei, “Uncertainty quantification and confidence calibration in large language models: A survey,” arXiv preprint arXiv:2503.15850, 2025.
- [33] Y. Abbasi Yadkori, I. Kuzborskij, A. György, and C. Szepesvari, “To believe or not to believe your llm: Iterative prompting for estimating epistemic uncertainty,” Advances in Neural Information Processing Systems, vol. 37, pp. 58077–58117, 2024.
- [34] R. Qu, R. Tu, and F. Bao, “Is semantic chunking worth the computational cost?” arXiv preprint arXiv:2410.13070, 2024.
- [35] C. E. Shannon, “Prediction and entropy of printed english,” Bell system technical journal, vol. 30, no. 1, pp. 50–64, 1951.
- [36] Y. Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu, A. Zeng, L. Hou et al., “Longbench: A bilingual, multitask benchmark for long context understanding,” arXiv preprint arXiv:2308.14508, 2023.
- [37] S. Xiao, Z. Liu, P. Zhang, and N. Muennighof, “C-pack: packaged resources to advance general chinese embedding. 2023,” arXiv preprint arXiv:2309.07597, 2023.
- [38] J. He, M. Rungta, D. Koleczek, A. Sekhon, F. X. Wang, and S. Hasan, “Does prompt formatting have any impact on llm performance?” arXiv preprint arXiv:2411.10541, 2024.
- [39] Y.-C. Chang, M.-S. Huang, Y.-H. Huang, and Y.-H. Lin, “The influence of prompt engineering on large language models for protein–protein interaction identification in biomedical literature,” Scientific Reports, vol. 15, no. 1, p. 15493, 2025.

- [40] S. Srivastava and Z. Yao, “Revisiting prompt optimization with large reasoning models-a case study on event extraction,” arXiv preprint arXiv:2504.07357, 2025.
- [41] C. Huyen, “Evaluation metrics for language modeling,” The Gradient, vol. 40, 2019.
- [42] S. Dragomir and C. Goh, “Some bounds on entropy measures in information theory,” Applied Mathematics Letters, vol. 10, no. 3, pp. 23–28, 1997.
- [43] Y. Tang and Y. Yang, “Multihop-rag: Benchmarking retrieval-augmented generation for multihop queries (2024),” arXiv preprint arXiv:2401.15391.
- [44] H. Jiang, Q. Wu, X. Luo, D. Li, C.-Y. Lin, Y. Yang, and L. Qiu, “Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression,” arXiv preprint arXiv:2310.06839, 2023.

### A Theoretical Proof for PPL Chunking

Firstly, we illustrate the relationship between cross-entropy and two distributions P and Q in another way. Based on sequencing inequality

n

aibi ≥

i=1

n

n

aibj(i) ≥

aibn+1−i

i=1

i=1

where a1 ≥ a2 ≥ ··· ≥ an, b1 ≥ b2 ≥ ··· ≥ bn and (j(1),j(2),...,j(n)) is an arbitrary sorting of (1,2,...,n), it can be observed that the sum of products of larger numbers paired together is the maximum, while the sum of products of larger numbers paired with smaller numbers is the minimum. We desire the cross-entropy H(P,Q) to be as small as possible, which means that when P(x) is relatively large, −log Q(x) should be relatively small, thereby resulting in Q(x) also being relatively large. Therefore, a smaller cross-entropy indicates that the prediction is closer to the actual label.

Afterwards, inspired by insights provided in [41], a property of formula (8) is proved: GK+1 ≤ GK for all K ≥ 1.

Proof.

GK − GK+1

= −

P(Tk)loga P(tk|Tk−1) +

P(Tk+1)loga P(tk+1|Tk)

Tk

Tk+1



 

 tk,tk+1

P(Tk+1)loga P(tk+1|Tk) −

P(Tk)loga P(tk|Tk−1)

=

tk

Tk−1

 



 tk,tk+1

P(Tk)loga P(tk|Tk−1)

≥

P(Tk+1)loga P(tk+1|Tk−1) −

tk

Tk−1

 



 tk,tk+1

P(Tk−1,tk)loga P(tk|Tk−1)

P(Tk−1,tk,tk+1)loga P(tk+1|Tk−1) −

=

tk

Tk−1

 



 tk+1

P(Tk−1,tk)loga P(tk|Tk−1)

P(Tk−1,tk,tk+1) −

loga P(tk+1|Tk−1)

=

tk

tk

Tk−1

 



 tk+1

P(Tk−1,tk+1)loga P(tk+1|Tk−1) −

P(Tk−1,tk)loga P(tk|Tk−1)

=

tk

Tk−1

=0

The reason for the last equality is that tk+1 and tk belong to the same domain. Thus, the proof is complete.

| |
|---|

Eventually, we illustrate bounds of entropy, so as to demonstrate the positive correlation between H(P,Q) and DKL(P||Q) in formula (4).

Proof. Let P be a discrete random variable with a finite range of values denoted by W := {w1,w2,...,wl}. Set pi = P{P = wi} for i = 1,2,...,l, and assume that pi > 0 for all i ∈ {1,2,...,l}. According to Lemma 2 in [42], if

- θi

- θj ≤ φ(ε) := 1 + εlnc + εlnc(εlnc + 2)

γ := max

i,j

then

0 ≤ logc

l

l

pkθk −

pk logc θk ≤ ε

k=1

k=1

where θk ∈ (0,+∞), pk ≥ 0 with lk=1pk = 1 and c > 1. Given that θk = 1/pk, the aforementioned inequality can be transformed into

0 ≤ logc l − Hc(P) ≤ ε where ε > 0 satisfies the following conditions

- pi

- pj ≤ φ(ε)

max

i,j

Furthermore, we can derive bounds for entropy as logc l − ε ≤ Hc(P) ≤ logc l. The proof is concluded.

| |
|---|

[Figure 3]

- Figure 3: Overview of RAG pipeline, as well as examples based on rules, similarity, and PPL Chunking. The same background color represents being located in the same chunk.

### B Design Philosophy of Logical Chunking

Our approach to text segmentation, centered on logical chunking, distinguishes itself fundamentally from methods primarily reliant on semantic similarity by prioritizing the preservation of complete logical arguments and the integrity of idea expression within each chunk. To ensure logical integrity, our method allows for variable chunk sizes. This dynamic granulation produces chunks that are

complete ideational units, thereby preventing logical discontinuities during segmentation, which leads to enhanced document retrieval relevance and improved content clarity.

The key advantage of this logical approach is its ability to recognize and maintain coherence even when constituent sentences exhibit low semantic similarity due to discussing different facets or representations of a core idea. Semantic chunking can falter here, potentially fragmenting a coherent logical argument if the direct semantic overlap between consecutive, logically-linked sentences is not high. In contrast, our method ensures that each meta-chunk is a self-contained logical expression, thereby avoiding breaks in the logical chain.

As illustrated in Figure 4, among the four scenarios we enumerated, the sentences maintain logical relationships with each other. It can be observed that the PPL distribution based on LLMs exhibits a gradual declining trend, and our chunking method would group these sentences into a single text chunk. However, the semantic similarity between these sentences is relatively low, indicating a high probability of them being separated, which may consequently lead to logical fragmentation.

[Figure 4]

- Figure 4: Examples of PPL value variations and semantic similarity for sentences with different logical relationships, where x ⊃ y, x|y, x− > y, and x := y refer to general-specific, parallel, sequential, and illustrative relationships, respectively.

### C Detailed Procedure for Semantic Completion

When the original text is segmented into isolated text chunks, each chunk may lose cross-chunk contextual associations, global structural coherence, or implicit logical relationships, leading to the following issues:

- • Incomplete Information: Critical details are truncated or dispersed across multiple chunks.
- • Semantic Discontinuity: Logical relationships between chunks are fragmented, impairing the model’s comprehension of the overall semantics.
- • Noise Interference: Irrelevant content is erroneously included within chunks, degrading the accuracy of retrieval and generation tasks.

[Figure 5]

- Figure 5: Trends in PPL distribution variations between original and rewritten text chunks across different LLMs.

By employing globally enhanced rewriting and summary generation, we can supplement each text chunk with missing global information, bridge semantic gaps, and ultimately elevate the response quality of RAG systems.

During the construction of our training dataset, we initially employ the QwQ-32B4 model, leveraging its long-inference mode, to comprehensively identify informational gaps and the requisite supplementary content. Following this, the ERNIE-3.5-128K5 model is utilized to perform model-based scoring and filtration of this potentially missing information. These refined informational fragments are then fused with the content of the current text chunk, generating a text segment that is both contextually coherent and semantically more complete.

Simultaneously, we leverage the ERNIE-3.5-128K model to generate highly condensed summaries informed by global information. This process aims to enhance the overall contextual awareness of text chunks. Specifically, ERNIE-3.5-128K employs a two-stage strategy: it utilizes document-level global information to generate a supplementary summary for the target text chunk, and concurrently produces a local summary for the text chuk itself. Subsequently, the model meticulously fuses these two types of summaries, ultimately yielding an enhanced summary sentence that clearly articulates the text chunk from a global perspective.

Through this meticulously designed series of processes, we leverage a LLM-driven data distillation pipeline to obtain voluminous and diverse high-quality training samples. At present, we construct 20K data instances for each of the two modules, providing crucial guidance signals for the full fine-tuning of SLMs. This approach enables our framework to uniquely balance high performance with lightweight deployment.

### D Main Experimental Details

All language models utilized in this paper employ the chat or instruct versions where multiple versions exist, and are loaded in full precision (Float32). The vector database is constructed using Milvus, where the embedding model for English texts is bge-large-en-v1.56, and bge-base-zh-v1.57 for Chinese texts. When conducting QA, the system necessitates dense retrievals from the vector database, with top_k set to 8 for CRUD and 5 for LongBench. In experiments, we utilize a total of five baselines, and their specific configurations are detailed as follows:

##### (a) Rule-based Chunking Methods

- 4https://huggingface.co/Qwen/QwQ-32B
- 5https://console.bce.baidu.com/qianfan/overview
- 6https://huggingface.co/BAAI/bge-large-en-v1.5
- 7https://huggingface.co/BAAI/bge-base-zh-v1.5

- • Original: This method divides long texts into segments of a fixed length, such as two hundred Chinese characters or words, without considering sentence boundaries.
- • Llama_index [18]: This method considers both sentence completeness and token counts during segmentation. It prioritizes maintaining sentence boundaries while ensuring that the number of tokens in each chunk are close to a preset threshold. We use the SimpleNodeParser function from Llama_index, adjusting the chunk_size parameter to control segment length. Overlaps are handled by dynamically overlapping segments using the chunk_overlap parameter, ensuring sentence completeness during segmentation and overlapping.

##### (b) Dynamic Chunking Methods

- • Similarity Chunking [37]: Utilizes pre-trained sentence embedding models to calculate the cosine similarity between sentences. By setting a similarity threshold, sentences with lower similarity are selected as segmentation points, ensuring that sentences within each chunk are highly semantically related. This method employs the SemanticSplitterNodeParser from Llama_index. For English texts, we exploit the bge-large-en-v1.5 model, and for Chinese texts, the bge-base-zh-v1.5 model. The size of the text chunks is controlled by adjusting the similarity threshold.
- • LumberChunker [20]: Leverages the reasoning capabilities of LLMs to predict suitable segmentation points within the text. We utilize Qwen2.5 models with 14B parameters, set to full precision.
- • Dense X Retrieval [26]: Introduces a new retrieval granularity called propositions, which condenses and segments text by training an information extraction model.

In order to control variables during the experiment, we ensure that each dataset have approximately the same chunk size using different chunking methods. Our primary experiments are conducted on the following datasets: 2WikiMultihopQA, Qasper, MultiFieldQA-en, MultiFieldQA-zh, and CRUD, with chunk lengths set to 122, 120, 112, 178, and 178 characters, respectively.

In the Margin Sampling Chunking method, we also use prompt, which mainly consists of two parts: instructions for guiding LLMs to perform chunking and two segmentation schemes. The specific form is shown in Table 6.

Table 6: Prompt used in Margin Sampling Chunking.

#### Chunking Prompt

This is a text chunking task. You are a text analysis expert. Please choose one of the following two options based on the logical structure and semantic content of the provided sentence:

- 1. Split sentence1+sentence2 into sentence1 and sentence2 two parts;
- 2. Keep sentence1+sentence2 unsplit in its original form; Please answer 1 or 2.

Table 7: Prompt with more granular task descriptions for Margin Sampling Chunking.

#### Chunking Prompt

This is a text chunking task. You are a text analysis expert. Please group two related paragraphs together and separate unrelated paragraphs based on the logical structure and semantic content of the provided sentences. Choose one chunking method from the following two options according to the above requirements:

- 1. Split sentence1+sentence2 into sentence1 and sentence2 two parts;
- 2. Keep sentence1+sentence2 unsplit in its original form; Please answer 1 or 2.

As we delve deeper into the influence of text chunking strategies on the performance of complex QA tasks, we further investigate the performance of various chunking strategies when overlapping

chunks are employed. The original chunking overlap method uses a fixed number of characters from the end of one chunk to overlap with the start of the next. The Llama_index overlap approach builds upon this by additionally considering sentence integrity. The PPL Chunking overlap strategy, on the other hand, dynamically assigns sentences represented by minimal points of PPL to both the preceding and subsequent chunks, resulting in dynamic overlap. These approaches generally produce overlap lengths averaging around 50 Chinese characters. Specific experimental results are presented in Section 5.3.

### E Exploration of Chunking Approach for Performance of Re-ranking

To explore the impact of chunking strategies on the RAG system, we evaluate the combination of different chunking and re-ranking methods using the MultiHop-RAG benchmark [43]. Initially, a top-10 set of relevant texts is filtered exploiting a dense retriever. We then compare two re-ranking strategies: (1) the BgeRerank method, leveraging the bge-reranker-large model [37], and (2) the PPLRerank method with the Qwen2-1.5B model, utilizing the re-ranking method mentioned in the coarse-grained compression section in [44].

[Figure 6]

Figure 6: Performance of re-ranking strategies combined with different chunking methods. ppl represents direct PPL Chunking, with a threshold of 0.5. The base reveals not utilizing re-ranking strategy.

Experimental results in Figure 6 reveal that PPL Chunking and PPLRerank achieve the best overall performance across all metrics. Further analysis demonstrate that, compared to traditional chunking, PPL Chunking not only provide performance gains independently but also significantly enhance the effectiveness of the subsequent re-ranking. Notably, while traditional chunking and re-ranking strategies already deliver performance improvements, PPL Chunking resulted in even greater reranking gains. For instance, in the Hits@8 metric, PPLRerank under the original chunking yielded a 1.42% improvement, whereas PPLRerank under PPL Chunking achieved a 3.59% improvement. The specific numerical values depicted in the figure can be found in Table 8.

### F Comparative Analysis of Two PPL Chunking Strategies

As shown in Figure 7, we compare two PPL Chunking strategies: direct PPL Chunking and PPL Chunking with dynamic combination, both of which are effective across the CRUD benchmark. Through experimental analysis, we find that the latter demonstrates superior performance. This is primarily due to direct PPL Chunking, which may result in overly long chunks, whereas the PPL Chunking with dynamic combination method effectively maintains chunk length and logical consistency.

In addition, PPL Chunking achieves significant performance improvements compared to traditional segmentation methods on BLEU series metrics and ROUGE-L. This indicates that our methods enhance the accuracy and fluency of the generated text to the reference text. Furthermore, this experiment reveals the delicate balance between model size and performance. Specifically, the performance of Qwen2-1.5B and Baichuan2-7B under this evaluation framework is closely matched, often surpassing the Qwen2-7B model across multiple metrics. The precise numerical data illustrated in the figure are available in Table 9.

[Figure 7]

- Figure 7: Performance of different methods on the CRUD benchmark. ppl represents direct PPL Chunking, with a threshold of 0.5. comb. indicates PPL Chunking with dynamic combination, with a threshold of 0 when performing PPL Chunking.

- G Hyperparameter Selection for PPL Chunking

We conduct an in-depth exploration of chunking in four long-text QA datasets of LongBench, and carry out gradient experiments on the threshold of PPL Chunking, aiming to reveal the intrinsic relationship between PPL distribution and chunking effectiveness. As shown in Figure 8, when chunk length is small, the direct PPL Chunking brings greater benefits, whereas when the chunk length is longer, PPL Chunking with dynamic combination performs better. In addition, experimental results indicate that the optimal configuration of PPL Chunking relies on the PPL distribution of texts: when the PPL distribution is relatively stable, it is more appropriate to select a lower threshold (such as setting the threshold to 0 in HotpotQA, MuSiQue, and DuReader); whereas when the PPL distribution exhibits large fluctuations, choosing a higher threshold (such as setting the threshold to 0.4 in NarrativeQA) can effectively distinguish paragraphs with different information densities, improving the chunking effect. Therefore, when employing PPL for chunking, it is crucial to comprehensively consider the dual factors of chunk length and text PPL distribution to determine the relatively optimal configuration that maximizes performance.

[Figure 8]

Figure 8: Performance of different methods in four long-text QA datasets of LongBench is evaluated based on F1, F1, F1, and ROUGE-L. ppl represents direct PPL Chunking, and comb. indicates PPL Chunking with dynamic combination. Multi represents threshold values of the parallel method in four datasets, which are 0.5, 0.5, 1.34, and 0.5 respectively, resulting in chunk lengths of 87, 90, 71, and 262 in sequence.

- H Collection and Refinement of Training Data

##### H.1 Filtering of Corpora Related to QA Tasks

In this experiment, we select the QA dataset from the CRUD benchmark. Among them, the single-hop QA dataset consists of questions focused on extracting factual information from a single document. These questions typically require precise retrieval of specific details such as dates, individuals, or events from the provided text. Before the chunking phase, we collect original news articles used in all types of QA tasks in CRUD. Specifically, since CRUD provides evidence context snippets relied

on by each QA pair, as well as the original news library where the context snippets are extracted, we can obtain the original news articles containing the context snippets through sentence matching. Taking the two-hop QA as an example, CRUD provides two news snippets, news1 and news2, which are necessary to answer questions. We then save the matched original news articles matched_news1 and matched_news2 that contain news1 and news2. Finally, from the original news library of 80,000 articles, we recall all 10,000 news articles containing context snippets as the initial text for chunking.

##### H.2 Dataset Construction for Rewriting and Summary Generation

To ensure the impartiality and validity of our evaluation, 10K documents obtained through the previously described filtering process are designated as an independent test set. To rigorously prevent data leakage, the dataset used for training the text rewriting and summarization components is entirely sampled from the remaining document corpus, with no overlap with this test set. Specifically, we randomly select 20K long documents from the non-test documents and apply the PPL Chunking via the Baichuan2-7B model for preliminary segmentation. Subsequently, we strategically sample text chunks of varying lengths from each document. Finally, following the data generation pipeline detailed in Section 3.2, we prepare the training data for model fine-tuning. The final model performance is evaluated on the reserved independent test set described above.

### I Corresponding Numerical Values of Images

In data analysis, the intuitive nature of visual representations facilitates a rapid grasp of the overall landscape. By simultaneously presenting the corresponding numerical values, we provide quantitative foundations for in-depth analysis, enabling a more precise interpretation of experimental results and trends.

- Table 8: Performance of re-ranking strategies combined with different chunking methods in the MultiHop-RAG benchmark. ppl represents direct PPL Chunking, with a threshold of 0.5.

Chunking and Re-ranking Hits@8 Hits@6 Hits@4 Hits@2 MAP@10 MRR@10

Original 0.5627 0.5180 0.4523 0.3499 0.1512 0.3507 Original and BgeRerank 0.5818 0.5406 0.4741 0.3379 0.1486 0.3391 Original and PPLRerank 0.5769 0.5521 0.5055 0.4102 0.1849 0.4147 PPL (Qwen2-1.5B) 0.6838 0.6244 0.5503 0.4151 0.1954 0.4195 PPL (Qwen2-1.5B) and BgeRerank 0.6927 0.6435 0.5721 0.4381 0.2075 0.4413 PPL (Qwen2-1.5B) and PPLRerank 0.7197 0.6931 0.6568 0.5721 0.2590 0.5558

- Table 9: Performance of different methods on the CRUD benchmark. ppl represents direct PPL Chunking, with a threshold of 0.5. comb. indicates PPL Chunking with dynamic combination, with a threshold of 0 when performing PPL Chunking.

Chunking Method BLEU-1 BLEU-2 BLEU-3 BLEU-4 BLEU-Avg ROUGE-L BERTScore

Original 0.3515 0.2788 0.2340 0.1997 0.2548 0.4213 0.8489 Llama_index 0.3620 0.2920 0.2480 0.2134 0.2682 0.4326 0.8521 Qwen2-1.5Bppl 0.3714 0.3013 0.2569 0.2223 0.2778 0.4426 0.8563 Qwen2-7Bppl 0.3661 0.2935 0.2481 0.2127 0.2691 0.4379 0.8558 Baichuan2-7Bppl 0.3725 0.3011 0.2558 0.2207 0.2772 0.4429 0.8562 Qwen2-1.5Bcomb. 0.3760 0.3034 0.2577 0.2224 0.2797 0.4443 0.8586 Qwen2-7Bcomb. 0.3724 0.3012 0.2561 0.2206 0.2774 0.4445 0.8584 Baichuan2-7Bcomb. 0.3812 0.3091 0.2622 0.2259 0.2840 0.4494 0.8603

