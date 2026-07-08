# arXiv:2510.03561v1[cs.CL]3Oct2025

## Reactive Transformer (RxT) - Stateful Real-Time Processing for Event-Driven Reactive Language Models

Adam Filipek (adamfilipek@rxai.dev) Reactive AI (https://rxai.dev)

October 2025

Abstract

The Transformer architecture has become the de facto standard for Large Language Models (LLMs), demonstrating remarkable capabilities in language understanding and generation. However, its application in conversational AI is fundamentally constrained by its stateless nature and the quadratic computational complexity (O(L2)) with respect to sequence length L. Current models emulate memory by reprocessing an ever-expanding conversation history with each turn, leading to prohibitive costs and latency in long dialogues. This paper introduces the Reactive Transformer (RxT), a novel architecture designed to overcome these limitations by shifting from a data-driven to an event-driven paradigm. RxT processes each conversational turn as a discrete event in real-time, maintaining context in an integrated, fixed-size Short-Term Memory (STM) system. The architecture features a distinct operational cycle where a generator-decoder produces a response based on the current query and the previous memory state, after which a memory-encoder and a dedicated Memory Attention network asynchronously update the STM with a representation of the complete interaction. This design fundamentally alters the scaling dynamics, reducing the total user-facing cost of a conversation from quadratic (O(N2 · T)) to linear (O(N ·T)) with respect to the number of interactions N. By decoupling response generation from memory updates, RxT achieves low latency, enabling truly real-time, stateful, and economically viable long-form conversations. We validated our architecture with a series of proof-of-concept experiments on synthetic data, demonstrating superior performance and constant-time inference latency compared to a baseline stateless model of comparable size.

### 1 Introduction

The introduction of the Transformer architecture [1] marked a watershed moment in Natural Language Processing (NLP). Its self-attention mechanism proved exceptionally effective at capturing long-range dependencies, leading to the rise of Large Language Models (LLMs) like GPT and BERT, which now dominate the field. These models excel at a wide range of tasks by processing vast sequences of text in a single forward pass.

However, this operational paradigm reveals a critical flaw when applied to conversational tasks. Transformers are inherently stateless; each input sequence is processed in isolation. To maintain conversational context, models are forced to re-ingest and re-process the entire history of the dialogue with every new user message. This ”brute-force” approach to context management results in a computational cost that scales quadratically with the number of conversational turns, rendering long-running dialogues economically impractical and leading to significant latency issues. While ever-larger context windows (up to millions of tokens) are being explored, they do not solve the underlying scaling problem but merely postpone it at an escalating cost.

Moreover, that stateless history reprocessing is not only extremely inefficient, but also fundamentally wrong for the awareness and real AGI, and is not natural - that’s not how humans think. We don’t need to remind all day or week history in order to remember what we were doing 10 minutes ago. Awareness also require continuous processing, but with stateless LLMs it’s more expensive and slower on each step, and

is limited to model’s context, so they are not compatible. Our ”Reactivity Hypothesis” states that ”real awareness and AGI models require continuous, stateful, real-time processing” - LLMs are not fulfilling any from the requirements, while Memory-Augmented Neural Networks fulfill them only partially.

This paper argues that the path forward for conversational AI requires a fundamental paradigm shift from data-driven stateless processing to event-driven, stateful computation. We introduce the Reactive Transformer (RxT), an architecture designed from the ground up for stateful real-time dialogue. RxT treats each user query as an event, processes it against an internal memory state, generates a response, and then asynchronously updates its memory. This approach offers two primary contributions:

- 1. A Novel Event-Driven Architecture: RxT redefines the roles of the encoder and decoder within a cyclical, asynchronous workflow. This decouples the user-facing task of response generation from the internal task of memory consolidation, drastically reducing perceived latency.
- 2. Linear Computational Scaling: By maintaining a fixed-size internal memory and processing only the current interaction, RxT eliminates redundant computation. The total cost of a conversation scales linearly with the number of turns, making long, coherent dialogues computationally feasible.

We posit that this architecture represents a crucial step away from simply scaling existing models and towards creating systems that can maintain a persistent, evolving understanding of an interaction, akin to a true conversational partner.

### 2 Background - the evolution of conversation modeling

The challenge of maintaining state and context in sequence modeling is not new. Our work builds upon a rich history of research, positioning RxT as a synthesis of prior concepts and a novel solution to their limitations.

#### 2.1 From Recurrence to Attention

Early approaches to sequence modeling were dominated by Recurrent Neural Networks (RNNs), including Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks [2]. These models process sequences token-by-token, maintaining an internal hidden state that serves as a compressed memory of past information. While naturally stateful, RNNs suffer from the vanishing gradient problem and sequential processing bottlenecks, limiting their ability to capture very long-range dependencies and leverage parallel hardware [3]. The Transformer architecture supplanted RNNs by replacing recurrence with a self-attention mechanism, allowing for parallel processing and superior modeling of long-distance relationships within a single sequence [1]. However, this came at the cost of abandoning the inherent inter-sequence statefulness of RNNs.

#### 2.2 Augmenting Neural Networks with Memory

The concept of equipping neural networks with external, addressable memory was explored in early architectures like Neural Turing Machines (NTM) and Differentiable Neural Computers [4, 5]. These models coupled a neural network ”controller” with a memory matrix, learning to perform read and write operations via differentiable attention mechanisms. While powerful, these systems were often difficult to train and did not see widespread adoption.

#### 2.3 System-Level Memory for LLMs

With the dominance of stateless LLMs, memory has been reintroduced at the system level rather than at the architectural level. Retrieval-Augmented Generation (RAG) frameworks connect LLMs to external knowledge bases (e.g., vector databases) to provide factual grounding [6]. Agentic frameworks such as LangChain orchestrate interactions between LLMs and various tools, including text-based memory modules. Although effective, these approaches treat the LLM as a black-box reasoning engine. The memory remains external, explicit, and subject to the same context window and reprocessing limitations of the underlying model it is just a text, added to prompt before user’s query/chat history, that could be only summarized, so its

expressiveness and compression possibilities are weak. What’s most important, LLM agents still operate in inefficient and incorrect (for awareness) data-driven full history processing paradigm.

#### 2.4 Alternative Architectures for Long Sequences

To address the quadratic complexity of attention, alternative architectures have emerged. State Space Models (SSMs), such as S4 and its successor Mamba, exhibit linear or near-linear scaling with sequence length [7, 8]. SSMs are inspired by control theory and use a recurrent mechanism that is highly parallelizable during training. However, their state is designed to capture dependencies within a continuous sequence (intra-sequence), not to manage a persistent, addressable memory between discrete conversational interactions (inter-sequence). In dialogue applications, they still rely on processing the full, growing history.

#### 2.5 Stateful Memory-Augmented Transformers

More recently, efforts have been made to reintegrate memory directly into the Transformer architecture. The Stateful Memory-Augmented Transformer (MemBART) [9] and it’s predecessor Memformer [10] are a notable examples. They augment a pre-trained encoder-decoder model with a memory module. However, its operational cycle is synchronous: the encoder processes the input query, reads from, and writes to memory, and only then passes the result to the decoder. This means that all memory operations contribute directly to the user-perceived latency, making it less suitable for real-time applications. Furthermore, they update their memory based solely on the input sequence, so previous answer still have to be included in input with query, making their processing not real-time (that require processing only current query/interaction), and initial processing phase is even slower. Stateful MATs were designed as memory extension for encoder-decoder transformers, extending their standard training methods with additional step.

#### 2.6 Next generation of the stateful processing - Reactive Transformer

Those stateful architectures were steps towards correct direction, with natural kind of processing and better results for conversational task than their stateless LLM counterparts, even with the limited expressiveness of their memory systems. However, they were released in 2020 and 2022, before the release of ChatGPT in 2023, which made the fundamentally wrong stateless ”chat template” processing a default choice. Since then, mainstream research has been moving in this wrong direction, further extending the model contexts, leading to even greater inefficiencies and costs. The decision to choose and develop unnatural stateless LLMs, which have almost no advantages (only a smaller number of overall parameters) over the natural stateful processing, has led to an unnecessary waste of enormous amounts of money and energy, especially for inference. It also pushed the research into directions like deep KV-cache optimizations, that in fact aren’t even needed in stateful processing (however, Reactive Transformer is using KV-cache for self-attention and full pre-cache for memory cross-attention for even better performance). The argument, that access to all the history tokens is an advantage over fixed-size memory is weak - it’s rather one of the biggest disadvantage of stateless modeling and the main source of model’s hallucinations in long multi-turn dialogues. Even the biggest LLM models have very weak responses in long conversations, full of hallucinations that are mixing information from different time steps. On the opposite, fixed-size memory provides the most important information for the dialogue in each step, potentially limiting hallucinations.

The Reactive Transformer addresses the shortcomings of these prior works. Unlike LLMs, it is natively stateful. Unlike RNNs, it leverages the power of parallel attention within interactions. Unlike RAG, its memory is an integrated, implicit part of the model. And unlike synchronous MATs like MemBART, its asynchronous memory update is based on both query and answer, achieving unnoticeable latency, making it truly suitable for real-time processing - all the responses are generated instantly in almost the same time, no matter of number of previous messages. Reactive Transformer is not the extension for current architectures, but redefines processing paradigm and requires separately designed custom training stages, that are finally cheaper than LLM training, even when they are more advanced.

### 3 The Reactive Transformer Architecture

The Reactive Transformer is an encoder-decoder architecture designed around the Event-Driven AI paradigm. Instead of processing a monolithic data sequence (the entire conversation history), the model operates in a continuous loop, treating each query-response pair as a discrete interaction. Its core components are a Generator-Decoder, a Memory Encoder, and a Memory Attention network, which collectively manage a multi-layer Short-Term Memory (STM) state.

The operational flow of RxT fundamentally reverses the classic Transformer encoder-decoder pipeline and makes it cyclical:

- 1. Response Generation: At interaction step t, the Generator-Decoder receives the user query Xt and generates a response Yt. This process is conditioned on the memory state from the previous interaction, STMt−1, which is accessed via Memory Cross-Attention.
- 2. Asynchronous Memory Update: After the response Yt has been generated and streamed to the user, the Memory Encoder processes a concatenation of the full interaction, concat(X,Y ), to produce a rich semantic representation, the Encoded Data (EDt).
- 3. Memory Consolidation: The Memory Attention network takes the previous memory state STMt−1 and the Encoded Data EDt as input to compute the updated memory state, STMt. This new state is then carried over to the next interaction, t + 1.

This asynchronous cycle ensures that the computationally intensive memory update process does not block the generation of the response, minimizing user-perceived latency.

#### 3.1 Architectural Components

Generator-Decoder: The decoder is responsible for autoregressive text generation. Each of its layers follows a standard pre-norm structure but includes an additional Memory Cross-Attention sub-layer. The sequence of operations is: Masked Self-Attention, Memory Cross-Attention, and a Feed-Forward Network (FFN). To manage parameter counts effectively, the decoder’s FFNs are implemented as Mixture-of-Experts (MoE) layers, allowing the decoder to have a much larger capacity than the encoder while maintaining a similar number of layers and hidden dimensions.

Memory Encoder: The encoder’s sole purpose is to create a condensed representation of the completed interaction. It processes the concatenated query and response concat(X,Y ) through a series of standard encoder layers (Self-Attention and a dense FFN) to produce the hidden states that form the Encoded Data (EDt).

Memory Attention Network: Memory attention is responsible for memory updates, based on the results of encoder - Encoded Data (EDt). It has multiple variants with different attention layers configuration and could use Memory Self-Attention or Interlayer Memory Attention (or combination of both) to prepare STMt - 1 for the final updates, when STMt - 1 is combined with EDt to produce final STMt. Residual Gates after each memory attention layer additionally decide how much the current and previous data should be used in the final updated state.

Shared Embeddings: The encoder, decoder, and memory system operate in a unified vector space. To facilitate this and reduce parameters, all components share a single token embedding layer.

#### 3.2 The Attention-Based Memory System (ABMS)

The core innovation of RxT lies in its integrated memory system. The STM is not a sequence of past tokens but a collection of fixed-size, learnable vectors (memory slots), organized into layers corresponding to each layer of the encoder and decoder.

Memory Read (Memory Cross-Attention): During generation, the decoder needs to access the conversational context stored in the STM. This is achieved via Memory Cross-Attention. In this operation, the hidden states of the decoder’s input sequence act as the Queries (Q), while the memory slots from the corresponding STM layer act as the Keys (K) and Values (V ).

RetrievedContext = Attention(Q = Hdec,K = STMt−1,V = STMt−1) (1)

[Figure 1] -- Stateful Real-Time Processing for Event-Driven Reactive Lang_images/imageFile1.png>)

###### Figure 1: Simple Memory Attention variant

[Figure 2] -- Stateful Real-Time Processing for Event-Driven Reactive Lang_images/imageFile2.png>)

###### Figure 2: Memory Self-Attention variant

[Figure 3] -- Stateful Real-Time Processing for Event-Driven Reactive Lang_images/imageFile3.png>)

###### Figure 3: Interlayer Memory Attention variant

[Figure 4] -- Stateful Real-Time Processing for Event-Driven Reactive Lang_images/imageFile4.png>)

###### Figure 4: Gated Self/Interlayer Memory Attention variant

[Figure 5] -- Stateful Real-Time Processing for Event-Driven Reactive Lang_images/imageFile5.png>)

###### Figure 5: Attention-Based Memory System Architecture

Crucially, since the STM slots do not have an inherent sequential order, positional encodings (e.g., RoPE) are applied only to the queries (Hdec) and not to the keys and values from memory.

Memory Write (Memory Attention): The memory update process is conceptually the inverse of the read operation. Here, the memory slots from the previous state STMt−1 act as the Queries (Q), while the Encoded Data EDt from the Memory Encoder provides the Keys (K) and Values (V ). This allows each memory slot to actively seek out and integrate relevant information from the latest interaction.

Update = Attention(Q = STMt−1,K = EDt,V = EDt) (2) The final memory state is then computed via a residual connection, typically controlled by a gate.

#### 3.3 Memory Attention Variants

To provide flexibility in how memory is consolidated, we propose several variants for the Memory Attention network, as illustrated in Figures.

- • Simple Memory Attention: The most direct approach, where the STM queries the Encoded Data as described above.
- • Memory Self-Attention: An additional self-attention step is introduced where memory slots attend

to each other (Q,K,V all from STMt−1) before attending to the Encoded Data. This allows the model to reorganize and consolidate information within the memory itself.

- • Interlayer Memory Attention: To facilitate communication across different levels of abstraction, memory slots in a given layer can also attend to an aggregated representation (e.g., the mean) of all other STM layers. This can help reduce information redundancy.
- • Gated Variants: The flow of information from the self-attention or interlayer attention steps can be controlled by an additional Residual Gate, allowing the model to learn how much internal consolidation is needed.

#### 3.4 Residual Gates

To prevent catastrophic forgetting and control the plasticity of the memory, the residual update is managed by a Residual Gate. Instead of a simple addition (STMt = STMt−1 + Update), the gate computes a dynamic interpolation:

STMt = (1 − G) ⊙ STMt−1 + G ⊙ Update (3)

where G is a gating vector computed by a small network, typically using a sigmoid activation function. The sigmoid ensures that the update is a weighted average, which empirically prevents the magnitude of the memory vectors from exploding over many interactions and provides more stable training. The gate computation can be dynamic (dependent on the input states) or static (a learned constant). Alternatively, tanh activation could be used or the gating could be completely skipped, but it requires much stronger regularization of memory updates in training algorithms. The gate result with tanh activation is calculated with:

STMt = (1 − G) ⊙ STMt−1 + (1 + G) ⊙ Update (4)

#### 3.5 Comparison with Stateful Memory-Augmented Transformers

Before Reactive Transformer (RxT), there were the attempts to provide stateful conversation processing with linear cost scaling. Furthermore, those stateful models achieved better results than comparable stateless models, especially in long multi-turn conversations. It makes the decision to proceed with stateless LLMs even more irrational. However, there are notable differences between RxT and Stateful MAT, that made RxT fundamental paradigm shift instead of being just an evolution.

[Figure 6] -- Stateful Real-Time Processing for Event-Driven Reactive Lang_images/imageFile6.png>)

###### Figure 6: Memory Attention variants

[Figure 7] -- Stateful Real-Time Processing for Event-Driven Reactive Lang_images/imageFile7.png>)

###### Figure 7: Residual Gate variants

[Figure 8] -- Stateful Real-Time Processing for Event-Driven Reactive Lang_images/imageFile8.png>)

Figure 8: Diagrams of stateful and stateful real-time processing

###### 3.5.1 Compatibility with event-driven paradigm

Event-Driven AI paradigm made for stateful real-time processing is based on two types of events handled by the conversational language models: query/input event (current user’s query) and response/output event (model’s answer). Process of generating response/output event in reaction on query/input event is defined as interaction. Synchronous memory update in encoder-first stateful MATs requires previous model’s answer to be passed with current query as an input, to preserve all dialog history context. It breaks the fundamental Event-Driven AI rule, by processing events from two conversation steps at once (and leads to higher encoding cost and latency). In the opposite, RxT’s input is always only current user’s query. That difference has also practical implications:

- • Stateful MAT: Memory is accessed and updated (in encoder), basing on previous answer and current query.
- • RxT: Memory is accessed (in decoder) and updated (by encoder and memory attention), basing on current interaction: query and answer. That’s far more natural and expressive, because model knows for which query the answer were generated.

###### 3.5.2 Single Responsibility Principle

In encoder-first stateful MATs, the encoder is responsible for memory access and update, while the decoder is using combined memory and input in its cross-attention. In Reactive Transformer, components responsibility is separated, leading to easier tasks for each component and better expressiveness in result:

- • Decoder’s self-attention is responsible for current interaction inter-sequence dependencies - both query and generated answer are treated as single interaction input to self-attention. It’s closer to LLMs text completion, but without the long conversation history.
- • Decoder’s memory cross-attention is responsible for memory access for processed query and generated answer. It contains only the information from previous steps, not polluted by input data (that’s handled by self-attention).
- • Encoder has only self-attention and responsibility to transform the current interaction into memory vector spaces. That’s a simple, full sequence processing task, without any memory access and update that’s why encoder could be a lot smaller (dense vs. MoE decoder) to balance the overall parameters count
- • Memory Attention network is responsible for memory updates, based on current interaction and previous memory state. Due to the separation from the encoder, it could have multiple expressive variants.

###### 3.5.3 Design choices for architecture and training pipelines

Stateful MATs and Reactive Transformer were designed with different goals in mind and started from different baselines.

Stateful MATs were created to extend the encoder-decoder transformer with memory and are made for stateful sequence to sequence processing. They are using the same decoder as baseline seq-to-seq models and modifying only the encoder. The advantage of this approach is only slightly modified training pipeline.

On the other hand, Reactive Transformer was designed to extend the decoder-only LLMs with stateful processing and memory system, and replace them in conversational tasks. It changes the data-driven sequence completion into event-driven interaction completion, instead of using sequence to sequence processing. RxT was created from scratch, without any baseline and require completely re-designed training pipeline to handle its asynchronous nature. It’s natively trained for conversational tasks and rather cannot be used for single-step generative tasks.

### 4 Inference and Computational Analysis

The architecture of Reactive Transformer (RxT) leads to a highly efficient inference process and fundamentally different scaling properties compared to traditional LLMs, and asynchronous memory update eliminates Stateful MAT latency overhead.

- 4.1 Inference Process A single interaction in RxT consists of two phases:

###### 1. Prompt Processing & Autoregressive Generation:

- • The input query Xt is processed in a single forward pass through the decoder to populate a self-attention KV cache. This is the ”prompt phase”.
- • The decoder then generates the response Yt token by token. For each token, it attends to the self-attention KV cache (containing the query and previously generated response tokens) and the memory cross-attention KV cache.
- • Optimization: Since the memory state STMt−1 is static throughout the generation of a single response, its Key and Value projections for the Memory Cross-Attention mechanism can be fully pre-cached before the prompt phase. This eliminates redundant projections for every generated token, significantly accelerating the generation process.

###### 2. Asynchronous Memory Update:

• Once generation is complete, the concatenated interaction concat(X,Y ) is processed by the Memory Encoder and Memory Attention network to compute STMt. This happens ”in the background” and does not add to the user-perceived latency.

#### 4.2 Computational Cost Analysis

To analyze the computational cost, let N be the number of interactions in the conversation history, Tquery be the length of the current query, Tanswer be the length of the generated answer, and T be the average length of a full interaction (T ≈ Tquery + Tanswer).

Standard Transformer LLM: At interaction N + 1, the model must first process a prompt of length Lprompt ≈ (N · T) + Tquery. Subsequently, it generates Tanswer tokens autoregressively.

###### • Single Interaction Cost (Computational): The total cost is a sum of two phases:

- 1. Prompt Processing: This phase is parallelizable but has a cost quadratic to the input length: O(L2prompt) ≈ O((N · T + Tquery)2).
- 2. Token Generation: This phase is sequential. To generate each of the Tanswer tokens, the model must perform an attention operation over a KV cache that contains the entire preceding conversation history. The cost to generate a single token is therefore linear with the cache length, which is approximately Lprompt. The total generation cost is thus O(Lprompt + Tanswer) ≈ O((N + 1) · T).

The combined cost is dominated by these two terms: O((N · T + Tquery)2 + (N + 1) · T). As N grows, both prompt processing and per-token generation become prohibitively expensive.

###### • Total Conversation Cost (User-Facing): From a user-cost perspective (e.g., API pricing based on token count), the total number of tokens processed across N interactions scales quadratically. The total number of tokens is approximately Nk=1(k · T), which results in a cost of O(N2 · T).

Reactive Transformer (RxT): At any interaction, the model’s decoder processes a query of length Tquery and generates a response of length Tanswer. The memory update cost is handled asynchronously.

###### • Single Interaction Cost (Computational):

- 1. Prompt Processing: The cost is quadratic only to the current, short query: O(Tquery2 + Tquery · Smem).
- 2. Token Generation: In stark contrast to standard LLMs, the KV cache for self-attention only contains tokens from the current interaction, and the memory cross-attention cache is of a fixed size. Therefore, the computational cost to generate each subsequent token is not only low but, critically, constant with respect to the overall conversation length N. The total generation cost is approximately O(Tquery + Tanswer + Smem).
- 3. Asynchronous memory update: RxT has additional memory update step after all response

token generation. It scales quadratically with short-term memory size O(Smem2 ), but is not influencing the user’s latency.

The combined user-perceived cost is O(Tquery2 + Tquery · Smem + Tquery + Tanswer + Smem), and additional update cost O(Smem2 ), the values independent of N.

###### • Total Conversation Cost (User-Facing): The model processes a roughly constant number of tokens, T, for each of the N interactions. Therefore, the total user-facing cost scales linearly: O(N · T).

This analysis reveals a dual advantage for RxT: not only is the initial prompt processing exponentially cheaper for long conversations, but the per-token generation speed remains constant, whereas for standard LLMs, it degrades linearly with the length of the dialogue. Table 2 summarizes this comparison.

- * Smem > Tquery, so in Synchronous MAT prompt phase Smem2 is the dominant factor, while in RxT Tquery · Smem dominates.
- ** Thanks to the asynchronous update phase, RxT includes answers in memory, so it has full dialog context without delays for the user.
- *** With persistent KV-cache

Table 1: Asymptotic comparison of computational costs for a single message of T tokens in conversation of N turns.

Model Architecture Prompt phase cost Generate phase Asynchronous phase

(Computational, turn N) (Computational, turn N) (Computational, slots S) LLM O((N · T)2) O((N · T) + Tanswer) -

O(T · (N · T))***

LLM with RAG O((Tmem + N · T)2) O((Tmem + N · T) + Tanswer) SSM / Linear Attn O(N · T) O((N · T) + Tanswer) Synchronous MAT O(T2 + Smem2 ) O(Tanswer + Smem) -

O(Smem2 )*

###### RxLM (RxT) O(Tquery2 + Tquery · Smem) O(Tquery + Tanswer + Smem) O(Smem2 )**

O(Tquery · Smem)*

Table 2: Comparison of end user costs for a conversation of N turns.

###### Model Architecture Single Interaction Cost Total User-Facing Cost Scaling vs. (Tokens) (Tokens) Interactions

LLM O(N · T) O(N2 · T) Quadratic SSM / Linear Attn O(N · T) O(N2 · T) Quadratic Synchronous MAT O(T) O(N · T) Linear Reactive Transformer (RxT) O(T) O(N · T) Linear

### 5 Supervised Training Curriculum

The architectural design of the Reactive Transformer, with its asynchronous, multi-component nature, presents unique training challenges that cannot be addressed by a naive, end-to-end optimization strategy. Such an approach is prone to instability and convergence failure due to the complex interplay between the generator, encoder, and the non-interpretable memory state. To overcome these hurdles, a carefully designed, multi-stage supervised training curriculum is proposed. This curriculum acts as a scaffolding process, systematically pre-training and integrating each component to solve predictable failure modes before unifying the entire system. The objective is to establish a shared semantic space, pre-condition the memory network for its abstract task, and finally, teach the full, memory-dependent interaction cycle.

#### 5.1 Overview of the Training Pipeline

The supervised training pipeline consists of four distinct stages, designed to incrementally build the model’s capabilities. The overarching strategy is to first establish a robust foundation for the language components, then independently prepare the memory system, and finally, integrate them to learn the complete, stateful conversational flow. This structured approach is essential for solving the ”cold start” problem inherent in two key areas: the interaction between the decoder and the memory system, and the memory update mechanism itself. By addressing these challenges systematically, the curriculum prepares the model for the subsequent, more advanced reinforcement learning phases. The four supervised stages are:

- 1. Joint Language Model Pre-Training: Co-trains the Generator-Decoder and Memory Encoder on a large text corpus to learn fundamental language representations and align their vector spaces.
- 2. Joint Interaction Supervised Fine-Tuning (SFT): Adapts the pre-trained language components to the specific format of conversational interactions using a dataset of query-response pairs.
- 3. Self-Supervised Memory Attention Pre-Training: Trains the Memory Attention network on a proxy task to produce semantically coherent outputs, addressing the lack of direct ground-truth labels for memory states.

- 4. Supervised Memory-Aware Training: Unifies all pre-trained components and trains the full model on multi-step dialogues, teaching the decoder to leverage the accumulated memory state for maintaining context.
- 5.2 Stages 1 & 2: Joint Language Model Pre-Training and Fine-Tuning

The initial stages are focused on co-training the Generator-Decoder and Memory Encoder to establish a shared semantic foundation. The model is treated analogously to a standard encoder-decoder transformer, employing a dual-objective function.

The training algorithm proceeds as follows: An input sequence S is duplicated. One copy is processed autoregressively by the decoder, while the other is randomly masked to create Smask for the encoder. The Memory Encoder processes Smask, and its final hidden states are passed to a dedicated Masked Language Modeling (MLM) head to compute the MLM loss, LMLM.

Concurrently, the hidden states from each layer of the encoder, ED = {ed1,ed2,...,edL}, are detached from the computation graph. This crucial step prevents gradients from the decoder from flowing back into the encoder, effectively treating the encoder’s output as a fixed target for the decoder. To improve generalization and prevent the decoder from becoming overly reliant on this perfect context, a small amount of random noise, ϵ, is added: ED′ = ED + ϵ. These noisy states serve as the Key and Value inputs for the decoder’s Memory Cross-Attention layers. The decoder then processes the original, unmasked sequence S autoregressively, conditioned on the context from ED′, and the standard autoregressive cross-entropy loss, LAR, is computed. The total loss for the joint training is a weighted sum:

LJoint = αLAR + βLMLM (5)

where α and β are hyperparameters balancing the two objectives. This ”teacher forcing” approach rapidly bootstraps the connection between the encoder and decoder. The addition of noise acts as a vital regularization technique, mitigating the primary risk of this stage: the decoder developing a weak self-attention mechanism by depending too heavily on the ”cheated” context provided by the encoder.

The Supervised Fine-Tuning (SFT) stage follows the exact same algorithm but shifts the data distribution from a general text corpus to a dataset of structured conversational turns, typically formatted with special tokens (e.g., ‘[Query]... [Answer]‘). This adapts the model to the specific turn-taking format of dialogue.

#### 5.3 Stage 3: Self-Supervised Memory Attention Pre-Training

The central challenge in training the Memory Attention network is that its target output—the updated memory state STMt—is a high-dimensional, non-interpretable tensor for which no human-generated labels exist. To circumvent this, a self-supervised proxy task is employed. The objective is to train the network to produce a plausible combination of the previous memory state and the new information from the current interaction.

The algorithm is initialized with a previous memory state STMt−1 (initially random noise) and the Encoded Data EDt from a pre-trained encoder. A pseudo-label, STMtarget, is generated via a dynamic weighted average:

STMtarget = (1 − wt) · STMt−1 + wt · EDt (6)

The weighting factor wt is annealed over a sequence of interactions within a curriculum. For the first interaction, wt is high (e.g., 0.9) to prioritize incorporating the new information. For subsequent interactions, wt is progressively decreased, encouraging retention and integration. The Memory Attention network then computes the actual updated state, STMt = MemAttn(STMt−1,EDt). The loss function is the negative cosine similarity between the predicted and target states, which encourages semantic alignment without enforcing an exact match:

LMem = −cosine similarity(STMt,STMtarget) (7)

This training stage is a critical piece of the curriculum’s scaffolding. A randomly initialized Memory Attention network would output vectors that are effectively noise. If this noisy output were fed directly to the decoder in the next stage, it would act as a powerful distractor, corrupting the learning signal. The decoder would likely learn to ignore its memory cross-attention layers entirely, defeating the architecture’s purpose. Empirical

evidence confirms this failure mode, with initial reinforcement learning rewards dropping to near-zero without this pre-training step. Therefore, the primary function of this stage is not to create a perfect memory system, but to pre-condition the network to produce outputs that are semantically coherent and reside within the same vector space as the other components, thus solving the cold start problem and enabling subsequent training stages to succeed.

#### 5.4 Stage 4: Supervised Memory-Aware Training

The final supervised stage unifies all pre-trained components to train the model on its intended, event-driven operational cycle. This is the first point at which the decoder learns to rely on a meaningful, accumulated memory state from genuinely past interactions, rather than the ”cheated” context from the joint training stages.

The algorithm uses a curriculum of multi-step dialogues, {I1,I2,...,IN}.

- 1. The memory state STM0 is initialized with random noise (e.g., from a normal distribution). The model processes the first interaction I1 = (X1,Y1), with the decoder conditioned on STM0. The autoregressive loss L1 is computed. This step explicitly trains the model to handle the beginning of a conversation from a blank state.
- 2. For each subsequent step t from 1 to N − 1:

- (a) The completed interaction It is encoded using the Memory Encoder to produce EDt.
- (b) The memory state is updated using the pre-trained Memory Attention network: STMt = MemAttn(STMt−1,EDt).
- (c) The decoder processes the next interaction’s query Xt+1 and generates the response Yt+1, conditioned on the newly computed memory state STMt.
- (d) The autoregressive loss Lt+1 is computed for this interaction.

- 3. The total loss is the sum of losses from all steps, and backpropagation is performed. To stabilize training, the parameters of the encoder and memory attention network may be initially frozen and then gradually unfrozen.

This stage directly optimizes the model’s ability to maintain conversational coherence by forcing the decoder to extract relevant context from the dynamically updated STM. Upon completion, the model possesses a partially functional memory system and is fully prepared for refinement via reinforcement learning.

6 Experiments and Results

To validate the architectural claims and the effectiveness of the training curriculum, a series of experiments were conducted. The primary research questions were: (1) Does the RxT architecture outperform a standard decoder-only LLM of comparable size on multi-turn conversational tasks? (2) Do the performance benefits of RxT scale with model size? (3) Does the proposed training curriculum successfully enable the memory system to function effectively?

##### 6.1 Experimental Setup Models: Four RxT variants of increasing scale were trained, alongside a baseline model for comparison.

- • RxT Variants: RxT-Alpha Nano (12M parameters), RxT-Alpha Micro (26M), RxT-Alpha Mini (100M), and RxT-Alpha Synthetic (160M). The Nano model used Interlayer Memory Attention, while the larger variants used the more expressive Gated Self/Interlayer Memory Attention. Interlayer variants achieved better results than simple/self memory attention in supervised memory-aware training experiments, so we are using them as a default choice.
- • Baseline: A 22M parameter decoder-only Transformer LLM was trained on the same data to provide a direct and fair comparison point for the smaller RxT models.

Datasets: All models were trained using the TinyStories [12] dataset for general language pre-training [13]. Subsequently, custom multi-turn interaction datasets derived from TinyStories were used for fine-tuning and evaluation. These datasets, referred to as MRL Curriculum Datasets, consist of series of interconnected interactions designed to test context retention.

Models architectures: All RxT models, as well as reference stateless decoder-only model are using our Sparse Query Attention (SQA) [11] for all attention layers. It’s computational efficiency is a perfect match for RxT Encoder and Memory Attention networks, because they are based on full sequence processing. All the models are also using Mixture-of-Experts in decoders. Baseline LLM is using exactly the same configuration for the fair comparison. We could compare the models with similar size GPT-1 [14] and GPT-2 [15] models or stateful MemBART [9], but those models are old and have outdated architectures, so we decided, that it will be unfair comparison - it’s not hard to outperform models from previous generations. While stateful predecessors like MemBART offer important context, our primary goal is to demonstrate the advantages of RxT’s event-driven paradigm over the dominant stateless approach used in today’s state-of-the-art systems

Evaluation Metrics: Performance was assessed using three metrics:

- • Perplexity (PPL): A standard measure of a language model’s ability to predict a sequence of text. Lower values indicate better fluency.
- • Accuracy: Standard next-token prediction accuracy.
- • MRL Reward Score: A custom, composite metric designed to serve as a proxy for conversational quality, scaled to a 0-10 range. It is a weighted sum of BLEU score (for fluency), cosine similarity between the generated response and the ground-truth response (for immediate relevance), and previous interaction content, with cosine similarity between the generated response and the history of preceding ground-truth interactions (for long-term coherence). This reward score methodology will be deeply described in one of our next research papers for memory benchmark.

#### 6.2 Performance on Memory-Aware Language Modeling

The results from the final Supervised Memory-Aware Training stage provide a clear quantitative measure of the models’ language modeling capabilities. Table 3 summarizes the performance of all models on the multi-turn dialogue test set.

Table 3: Memory-Aware Training Performance on Multi-Step Dialogues.

Model Parameters Perplexity (PPL) Accuracy (%) LLM Baseline 22M 4.37 55

RxT-Alpha Nano 12M 2.74 ∼81 RxT-Alpha Micro 26M 2.56 ∼82 RxT-Alpha Micro (updated) 26M 2.31 ∼82 RxT-Alpha Mini 100M 2.50 ∼80 RxT-Alpha Synthetic 160M 2.18 ∼82

The results demonstrate a stark performance gap. Even the smallest RxT model (Nano, 12M) significantly outperforms the larger 22M LLM baseline, achieving a perplexity of 2.74 compared to the baseline’s 4.37. This trend holds across all scales, with every RxT variant showing superior fluency and predictive accuracy. The data also confirms that the RxT architecture benefits from increased capacity, as perplexity generally decreases with more parameters, dropping to 2.18 for the 160M Synthetic model. Furthermore, the inclusion of the ”RxT-Alpha Micro (updated)” model, which was trained with improvements to the pipeline, isolates the impact of the training methodology itself. Its notable performance gain (PPL dropping from 2.56 to 2.31) underscores that methodological refinements are as crucial as architectural design.

[Figure 9] -- Stateful Real-Time Processing for Event-Driven Reactive Lang_images/imageFile9.png>)

Figure 9: Supervised Dialog Perplexity of different scale models

[Figure 10] -- Stateful Real-Time Processing for Event-Driven Reactive Lang_images/imageFile10.png>)

- Figure 10: Supervised Dialog Perplexity compared to the same size stateless LLM

[Figure 11] -- Stateful Real-Time Processing for Event-Driven Reactive Lang_images/imageFile11.png>)

- Figure 11: Supervised Dialog Accuracy compared to the same size stateless LLM

#### 6.3 Conversational Coherence Evaluation

While perplexity measures fluency, the MRL Reward Score provides a more holistic assessment of a model’s ability to maintain a coherent, context-aware conversation. Table 4 presents the benchmark results over an 8+1 step interaction sequence.

Table 4: MRL Conversational Coherence Benchmark Results (8+1 Interaction Steps).

###### Model Parameters Mean Reward Max Reward Min Reward

LLM Baseline 22M 2.4 4.0 1.4 RxT-Alpha Nano 12M 3.1 4.4 1.4 RxT-Alpha Micro (updated) 26M 3.4 5.2 1.8 RxT-Alpha Mini 100M 3.7 6.2 1.8 RxT-Alpha Synthetic 160M 3.8 6.8 1.9

The MRL reward data reinforces the findings from the perplexity analysis. The mean reward, which serves as the best single indicator of overall conversational quality, shows a clear and consistent trend: all RxT models outperform the baseline, and performance scales with model size, rising from 3.1 for Nano to 3.8 for Synthetic. This provides strong evidence that the memory system is not only functional but also increasingly effective at larger scales.

The minimum and maximum reward scores offer further insight. The higher minimum reward for the RxT models (e.g., 1.8-1.9 for the larger variants vs. 1.4 for the LLM) suggests greater robustness. The dedicated memory system appears to provide a more stable contextual foundation, making the models less prone to catastrophic failures where they completely lose the thread of the conversation. The maximum reward indicates the model’s peak performance, showing that larger RxT models are capable of producing significantly higher-quality, more coherent responses.

#### 6.4 Prompt Phase Latency in Memory-Based Dialogue Benchmark

We benchmarked the prompt phase latency of our RxT model against a stateless reference LLM. The benchmark was conducted in the dialogue memory setting with up to 8 conversational steps. As shown in Figure 13, the reference LLM exhibits a steady increase in latency from 0.09s at step 1 to over 0.22s at step 8, due to the quadratic dependence on context length inherent to decoder-only architectures. In contrast, RxT

[Figure 12] -- Stateful Real-Time Processing for Event-Driven Reactive Lang_images/imageFile12.png>)

###### Figure 12: Memory and Dialog Quality Benchmark scores

[Figure 13] -- Stateful Real-Time Processing for Event-Driven Reactive Lang_images/imageFile13.png>)

Figure 13: Prompt phase latency in dialog for RxT and reference stateless LLM

maintains nearly constant latency across all steps (∼0.06s), independent of dialogue depth, thanks to its fixed-size memory mechanism.

These results highlight a key efficiency advantage of RxT: unlike decoder-only LLMs, whose prompt phase grows more expensive as the dialogue context expands, RxT achieves predictable and stable inference time, which is critical for interactive applications requiring long-term consistency and low-latency responses.

#### 6.5 Analysis and Discussion

The combined experimental results provide compelling evidence for the principle of architectural specialization. The 22M parameter LLM baseline must solve the problem of dialogue context using only its general-purpose self-attention mechanism over a long, undifferentiated sequence of tokens. This is an inefficient, brute-force approach that requires the model to re-discover the dialogue’s structure and identify relevant historical information from scratch at every single turn.

In contrast, the 26M RxT-Alpha Micro model, with a comparable parameter count, dramatically outperforms the baseline. This is not a result of more parameters, but of a more intelligent organization of those parameters for the specific task of dialogue. The RxT architecture embodies a ”division of labor”: the Memory Encoder’s role is to summarize the immediate past, the Memory Attention network’s role is to integrate that summary into a persistent state, and the Generator-Decoder’s role is to generate a response conditioned on this curated context. This design offloads the cognitive burden of long-term context management from the decoder into specialized, purpose-built components.

The benchmarks confirmed the results achieved with MemBART [9], where memory-augmented transformer significantly outperformed similar stateless alternatives.

The implication is that for complex, structured tasks like maintaining conversational state, simply increasing the size of a generic, monolithic model may be a fundamentally suboptimal path. Designing architectures that reflect the inherent structure of the problem can lead to superior performance with greater parameter and computational efficiency. The linear-time inference cost of RxT is not merely an optimization; it is a symptom of a more appropriate and effective architectural design for stateful interaction. The success of these small-scale models serves as a strong proof-of-concept, motivating future work in scaling the Reactive Transformer to larger models and more complex, real-world datasets.

### 7 Conclusion and Future Work

The Reactive Transformer presents a new architectural paradigm for conversational AI. By embracing an event-driven, stateful approach, it directly addresses the critical bottlenecks of computational complexity and latency that plague current Large Language Models in dialogue applications. Its asynchronous operational cycle and integrated attention-based memory lead to linear scaling of costs and enable genuine real-time interaction.

This work lays the foundation for a new class of Reactive Language Models (RxLMs). Future work will detail the multi-stage training curriculum required to effectively train this architecture, including supervised and reinforcement learning phases. Furthermore, the current Short-Term Memory system is a stepping stone towards more advanced models incorporating a persistent Long-Term Memory (LTM), enabling true live learning and infinite context retention. We believe this direction is essential for moving beyond language modeling and toward developing more capable, aware, and truly interactive AI systems.

Supervised training of Reactive Transformer is just a first stage of bigger curriculum, that’s extended by follow-up Reinforcement Learning stages (Memory Reinforcement Learning and Reinforcement Learning from Human Feedback for Reactive Models), that will be deeply described in our future work.

Future work will focus on scaling the Reactive Transformer to larger parameter counts and benchmarking on complex, real-world datasets. A key priority will be a direct comparison against other leading efficient architectures, such as State Space Models like Mamba , to comprehensively map the landscape of nextgeneration sequence models.

### License and Patent Notice

Patent pending for the Reactive Transformer architecture (#P.453260). Commercial usage is regulated by the Reactive AI Models & Architecture License, that require reactive models to be trained using our RxNN/RxLM framework (https://github.com/RxAI-dev/rxlm) or third party libraries licensed by Reactive AI.

### References

- [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,  Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017.
- [2] Sepp Hochreiter and Ju¨rgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.
- [3] Yoshua Bengio, Patrice Simard, and Paolo Frasconi. Learning long-term dependencies with gradient descent is difficult. IEEE transactions on neural networks, 5(2):157–166, 1994.
- [4] Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines. arXiv preprint arXiv:1410.5401, 2014.
- [5] Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwinska, Sergio Go´mez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, et al. Hybrid computing using a neural network with dynamic external memory. Nature, 538(7626):471–476, 2016.
- [6] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Ku¨ttler, Mike Lewis, Wen-tau Yih, Tim Rockta¨schel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.
- [7] Albert Gu, Karan Goel, and Christopher R´e. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.
- [8] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

- [9] Qingyang Wu and Zhou Yu. Stateful memory-augmented transformers for efficient dialogue modeling. arXiv preprint arXiv:2209.07634, 2022.
- [10] Qingyang Wu et al. Memformer: A Memory-Augmented Transformer for Sequence Modeling. arXiv preprint arXiv:2010.06891, 2020.
- [11] Adam Filipek. Sparse Query Attention (SQA): A Computationally Efficient Attention Mechanism with Query Heads Reduction. arXiv preprint arXiv:2510.01817, 2025.
- [12] Ronen Eldan, Yuanzhi Li. TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv preprint arXiv:2305.07759, 2023.
- [13] Nirvan Patil et al. Regional Tiny Stories: Using Small Models to Compare Language Learning and Tokenizer Performance. arXiv preprint arXiv:2504.07989, 2025.
- [14] Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training. OpenAI Technical Report.
- [15] Brown, T., et al. (2020). Language Models are Few-Shot Learners. Advances in Neural Information Processing Systems, 33.

