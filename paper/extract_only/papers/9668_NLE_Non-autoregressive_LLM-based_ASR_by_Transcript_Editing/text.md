## NLE: Non-autoregressive LLM-based ASR by Transcript Editing

Avihu Dekel, Samuel Thomas, Takashi Fukada, George Saon IBM Research

avihu.dekel@ibm.com

# arXiv:2603.08397v1[eess.AS]9Mar2026

### Abstract

While autoregressive (AR) LLM-based ASR systems achieve strong accuracy, their sequential decoding limits parallelism and incurs high latency. We propose NLE, a non-autoregressive (NAR) approach that formulates speech recognition as conditional transcript editing, enabling fully parallel prediction. NLE extracts acoustic embeddings and an initial hypothesis from a pretrained speech encoder, then refines the hypothesis using a bidirectional LLM editor trained with a latent alignment objective. An interleaved padding strategy exploits the identity mapping bias of Transformers, allowing the model to focus on corrections rather than full reconstruction. On the Open ASR leaderboard, NLE++ achieves 5.67% average WER with an RTFx (inverse real-time factor) of 1630. In single-utterance scenarios, NLE achieves 27x speedup over the AR baseline, making it suitable for real-time applications.

Index Terms: Speech Recognition, Text Editing, NonAutoregressive

### 1. Introduction

LLM-based ASR systems [1, 2] improve transcription accuracy by pairing a pretrained speech encoder with a pretrained LLM, typically via a learned projector that maps acoustic representations into the LLM embedding space. In most existing systems, the LLM serves as an autoregressive decoder, generating text one token at a time. While this design yields strong accuracy, the sequential nature of autoregressive decoding limits parallelism, incurs substantial end-to-end latency, and results in slower inference (lower RTFx, where RTFx measures the ratio of audio duration to processing time). This limitation becomes particularly severe in real-time conversational settings where batch processing is not feasible, as the inability to parallelize token generation directly translates to high per-utterance latency. Moreover, these systems discard the initial hypothesis often produced by the speech encoder, despite it frequently providing a reasonable draft that could be refined rather than regenerated from scratch. This work focuses on addressing these limitations and enables parallelizable LLM-based inference.

Connectionist Temporal Classification (CTC) [3] provides an efficient alternative mechanism for mapping long acoustic sequences to shorter token sequences. A CTC encoder produces frame-level token posteriors, and inference collapses repeated tokens and removes blanks using an efficient, fully parallel decoding procedure. However, CTC decoding is constrained by conditional independence and monotonic alignment assumptions. Moreover, CTC models have limited language modeling capabilities and lack the broad linguistic priors of large pretrained language models, limiting their ability to recover plausible content when acoustic evidence is weak. In practice, CTC

NLE (Ours)

CTC Encoder

Other

Pareto Frontier

6.4

6.2

AvgWER

Phi-4 Multimodal

Parakeet 0.6B

Granite Speech 2B

6.0

NLE (Ours)

Controlled AR Qwen-3-ASR baseline

1.7B

5.8

Canary-Qwen Granite 2.5B

NLE++ (Ours)

Speech 8B

5.6

200 300 500 1000 1750 2500 4250

RTFx (log scale)

Figure 1: Open ASR leaderboard WER-RTFx tradeoff comparing NLE and NLE++ against top-6 models (as of Feb 2026). Both NLE variants lie on the Pareto frontier (no other model achieves both lower WER and higher RTFx), achieving competitive accuracy with superior inference speed.

outputs often exhibit local errors such as phonetic substitutions or structural errors such as word deletions, especially in noisy or ambiguous conditions. Many of these errors are systematic and locally correctable, making CTC hypotheses well-suited as drafts for downstream editing rather than full regeneration.

We introduce NLE, a Non-autoregressive LLM-based Editing ASR system. NLE reframes LLM-based speech recognition as conditional transcript editing: rather than decoding tokens autoregressively, NLE edits a hypothesis extracted from a pretrained speech encoder, guided by acoustic context from the same encoder. This editing formulation enables fully parallel prediction, thereby achieving fast inference.

Our approach adapts a pretrained LLM to the editing task, in order to leverage its linguistic knowledge. We modify the attention mechanism in the LLM from causal to bidirectional [4, 5], enabling non-autoregressive editing with full context. The model is trained using a CTC-style objective with latent alignments [6], which naturally handles the variablelength mapping between input and target transcripts. We use lightweight LoRA adapters [7] to adapt the model to the bidirectional attention and editing objective. The LoRA adapters can be disabled and the attention mechanism reconfigured back to causal mode, restoring the original LLM functionality. This design allows the LLM weights to be shared across both ASR and downstream tasks such as question answering on transcribed text – a key use case for speech-enabled LLMs [2].

We use an interleaved token layout with explicit insertion

slots that enables local insertions with minimal token movement. This design exploits the identity mapping bias of Transformers – the tendency to copy input tokens unchanged through residual connections and tied embeddings (see Section 3.3) – allowing the model to focus on corrections rather than full reconstruction.

On the Open ASR leaderboard [8], a challenging benchmark containing 39 strong ASR models, NLE++ achieves 5.67% average WER with an RTFx of 1630. As shown in Figure 1, both NLE and NLE++ are on the Pareto frontier in the WER-RTFx space, offering a strong accuracy-speed tradeoff compared to leading models. In single-utterance inference, NLE achieves 27x speedup over the autoregressive baseline, demonstrating the substantial practical advantages of nonautoregressive decoding for latency-critical applications.

### 2. Related Work

##### 2.1. LLM-based ASR

Recent work has explored integrating large language models into ASR systems by conditioning pretrained LLMs on speech representations via learned projection layers [1, 9, 10, 2]. Similar to encoder-decoder models like Whisper [11], these approaches pair a speech encoder with a text decoder, but leverage pretrained LLMs to benefit from their linguistic knowledge. By exploiting the strong linguistic priors of LLMs, they improve transcription accuracy, particularly in challenging acoustic conditions or when handling rare words and proper nouns. However, most LLM-based ASR systems rely on autoregressive decoding, where tokens are generated sequentially, inherently limiting parallelism and resulting in high inference latency and lower throughput – a critical bottleneck for real-time applications. Moreover, autoregressive models can hallucinate plausible but incorrect content when acoustic evidence is weak or ambiguous, and they discard the initial hypothesis often produced by the speech encoder despite it frequently providing a reasonable draft.

Alternative architectures such as RNN-Transducers (RNNT) [12] and Token-and-Duration Transducers (TDT) [13, 14] offer faster inference through streaming capabilities, but they still generate tokens sequentially and lack access to the broad linguistic knowledge encoded in pretrained LLMs. Our work addresses these limitations by using an LLM as a nonautoregressive editor that refines an initial CTC hypothesis in parallel, combining the speed of NAR decoding with the linguistic knowledge of pretrained LLMs.

##### 2.2. NAR ASR

Non-autoregressive (NAR) ASR methods aim to reduce inference latency by predicting tokens in parallel. Connectionist Temporal Classification (CTC) [3] is the most widely adopted NAR approach, marginalizing over alignments using dynamic programming and enabling efficient parallel decoding. However, CTC models typically lack strong language modeling capabilities and are constrained by conditional independence assumptions, limiting their ability to leverage linguistic context for disambiguation [15, 16]. Several NAR refinement approaches have been proposed to improve upon CTC. SoftCorrect [17] applies constrained CTC loss for transcript correction. Mask-predict methods [18, 19, 20] and iterative refinement methods [21, 22, 23] perform multiple passes, addressing CTC’s conditional independence by conditioning on partial predictions.

Despite offering substantial speed advantages over autoregressive models, NAR methods often struggle to perform insertions (i.e., correcting deletion errors) and to maintain long-range linguistic consistency. Many rely on fixed-length predictions or masking strategies that make insertions difficult or require multiple refinement iterations. Our approach builds on CTC’s efficiency while addressing these limitations through two key ideas: an insertion-aware interleaved representation that enables local insertions without sequence-wide shifts, and bidirectional LLM-based editing that leverages pretrained linguistic knowledge for improved contextual reasoning.

##### 2.3. ASR Correction

Post-processing methods that correct first-pass ASR outputs have been extensively studied. Traditional approaches include N-best list rescoring and lattice-based rescoring [24, 25, 26], which reevaluate hypotheses using refined acoustic or language models. More recently, LLM-based correction methods have emerged, where ASR transcripts are fed to external LLMs for zero-shot or few-shot correction [27, 28], leveraging their linguistic knowledge. Supervised error-correction approaches train encoder-decoder models on pairs of erroneous transcripts and references [29], enabling learned correction patterns. Unlike these post-processing methods that operate on finalized ASR outputs, our approach integrates correction into the decoding process itself through non-autoregressive editing conditioned on acoustic embeddings, enabling joint acousticlinguistic refinement rather than text-only correction.

##### 2.4. NAR Text Editing and Translation

Non-autoregressive methods for machine translation and text editing share conceptual similarities with ASR, as both involve mapping between closely related input-output sequences. Early work [30] introduced NAR translation using fertility-based parallel generation, while the Levenshtein Transformer [31] models editing through explicit operations like KEEP, DELETE, and INSERT. Most relevant to our work, latent alignment models with CTC have been applied to NAR translation [6] and extended to text editing with an explicit COPY operation [32]. We adapt these latent alignment techniques to speech recognition by conditioning on acoustic embeddings rather than text alone, and by leveraging a pretrained causal LLM adapted to bidirectional attention instead of training from scratch.

### 3. Method

NLE takes an input utterance and processes it through a pretrained speech encoder to produce acoustic embeddings and a CTC transcript hypothesis. An LLM-based NAR editor then predicts an edited transcript using a CTC-style objective over a token sequence interleaved with insertion slots. Figure 2 illustrates the complete pipeline.

##### 3.1. Extracting Hypothesis and Embeddings

We use a pretrained CTC-based speech encoder which we freeze during training. We freeze the encoder to preserve its well-trained acoustic modeling capabilities, as the quality of its hypothesis directly affects the entire training process; we leave joint fine-tuning for future work. The encoder processes the input audio and outputs frame-level embeddings H ∈ RT×d, where T is the number of frames and d is the embedding dimension. It also produces frame-level CTC logits over a

“We need some help.”

ε _We ε _need _some_help ε

#### Re-editing

Pretrained CTC Encoder

Projector + Downsampler

Bidirectional pretrained LLM

[Figure 1]

[Figure 2]

[Figure 3]

LoRA

ε _We ε _need ε _help ε

Speech Embeddings

Proposed transcript

“We need help”

Tokenization + Interleaved Padding

Figure 2: Overview of NLE architecture. The frozen pretrained CTC encoder produces acoustic embeddings and an initial CTC hypothesis. The hypothesis is tokenized and interleaved with insertion slots (ϵ), then concatenated with the projected speech embeddings. The LoRA-adapted bidirectional LLM editor predicts the edited transcript using a CTC objective. The output can be iteratively re-edited (see Section 5.5).

character-based vocabulary, which are converted to a characterlevel string hypothesis using greedy decoding (argmax followed by blank removal and deduplication). This hypothesis serves as the initial draft for the editing model, while the embeddings H provide acoustic context to guide the refinement process.

##### 3.2. Retokenization and Interleaved Insertion Slots

The character-level CTC hypothesis is re-tokenized using the LLM’s subword tokenizer to align with the LLM’s vocabulary. Since the LLM tokenizer is trained on well-formed text, misspelled words in the CTC hypothesis (e.g., ”philossophy”) may not have a corresponding token and will be split into multiple subword units. We denote the resulting token sequence as x = (x1, . . . , xN).

To efficiently handle insertion edits, we construct an interleaved sequence with explicit insertion slots:

###### x˜ = (ϵ, x1, ϵ, x2, . . . , ϵ, xN, ϵ). (1)

where ϵ denotes a blank symbol (we reuse the LLM’s EOS token). This interleaved representation creates N + 1 insertion slots (one before each token, and one after the last token), enabling local insertions without requiring the model to shift the entire downstream sequence. Single-token insertions can be handled by filling one insertion slot. Multi-token insertions require shifting only the neighboring tokens within a local block to accommodate the new content, while the rest of the sequence remains unaffected.

To illustrate this, consider adding the tokens (a, b, c) between xk and xk+1. Given the original sequence:

###### (. . . ϵ, xk, ϵ, xk+1, ϵ, . . . )

the model can produce this insertion by making the following per-position predictions: moving xk/xk+1 to the left/right ϵ spot, and using the 3 freed-up middle positions to add (a, b, c):

###### (. . . , xk, a, b, c, xk+1, . . . )

This can be performed within a single forward pass, keeping the rest of the tokens intact. An insertion of K tokens requires

changing 2K−1 tokens from the original sequence, while keeping the remaining tokens unchanged. The maximum number of tokens that can be inserted is N + 1, which was never required in our training data1.

##### 3.3. Identity Mapping Bias in Transformers

Our editing approach exploits the identity mapping bias inherent to Transformer architectures [33, 34]. This bias arises from two key architectural properties:

- • Residual connections preserve input representations across layers, allowing information to flow unchanged through the network.
- • Tied input-output embeddings are used in most LLMs (including Granite 4.0 1B Base we used in NLE) and create a strong copying bias: the dot product between a token’s embedding and itself in the output projection matrix tends to be high, making the model naturally inclined to predict the input token.

When the interleaved sequence is decoded without edits, it naturally recovers the original hypothesis. This property is beneficial for our task, as most tokens in the CTC hypothesis are correct and should be preserved. The model can focus its learning capacity on identifying and correcting errors rather than reconstructing the entire transcript, which is crucial for maintaining strong performance while enabling fast parallel decoding. We further reinforce this bias through a copying regularization objective (Section 3.6). In Figure 2, for example, the model is only required to change x5 = ϵ to ” some”, while copying the rest of the input tokens, with no token displacements needed due to the interleaved insertion slots.

##### 3.4. Bidirectional LLM-based Editor

We compute acoustic embeddings using a learned projector Pθ that downsamples the frame-level speech embeddings H and maps them into the LLM’s embedding space. The interleaved hypothesis x˜ is embedded using the LLM’s token

1We set the minimum number of input tokens to 8 to allow sufficient insertion capacity for short utterances.

embedding table E. These representations are concatenated along the sequence dimension to form the input to the LLM: Z = [Pθ(H); E(˜x)] ∈ R(T′+2N+1)×dLLM, where T′ is the downsampled acoustic sequence length and dLLM is the LLM’s hidden dimension.

Starting from a pretrained causal LLM, we modify its attention mechanism to bidirectional by removing the causal attention mask, allowing each position to attend to all other positions, while keeping the positional embeddings unchanged. This bidirectional context is essential for effective editing, as corrections often require information from both past and future tokens. We adapt the LLM using LoRA (Low-Rank Adaptation), which enables efficient fine-tuning while preserving the model’s pretrained linguistic knowledge. The LoRA adapters can be disabled to restore the original causal LLM functionality, allowing the same model weights to be shared across ASR and other downstream tasks.

##### 3.5. CTC-based Editing Objective

The LLM editor outputs token logits L ∈ R(2N+1)×|V| over the vocabulary V for each position in the interleaved sequence x˜. We apply the standard CTC loss LCTC(L, x⋆) that marginalizes over all valid alignments between the predicted logits and the reference transcript x⋆ using dynamic programming. The CTC objective naturally handles the variable-length mapping between the interleaved input sequence and the target transcript, allowing the model to learn which positions to keep, delete, or use for insertions. This design choice avoids the need to prealign the reference and hypothesis, as alignment happens implicitly during the loss calculation through CTC’s dynamic programming.

Edit operations are performed as follows:

- • Copy: The identity bias enables copying tokens from the input hypothesis by default.
- • Replace: Substitutions are performed by predicting a different token at a position.
- • Delete: Deletions are performed by predicting the blank symbol ϵ at a token position.
- • Insert: Single-token insertions use the explicit insertion slots. Multi-token insertions require shifting neighboring tokens, which the model learns through the CTC loss that allows multiple valid alignments.

##### 3.6. Copying Regularization Objective

While the CTC objective allows the model to learn editing operations, it does not explicitly enforce the identity mapping bias. The Transformer layers mix features across positions, and the CTC loss permits multiple alignments that produce the same output text, including redundant edits (changes in the CTC lattice that collapse to the same decoded string). When adapting a pretrained LLM trained with next-token prediction, this bias may be overridden by the model’s learned next-token behavior. To encourage the model to preserve correct tokens and make edits more interpretable, we add an auxiliary copying regularization (CR) loss:

2N+1

log P(˜xi|Li), (2)

LCR = −

i=1

where Li are the logits at position i and x˜i are the input tokens from x˜. This cross-entropy loss encourages the model to predict

the input tokens at each position, reinforcing the copying bias. The total training objective becomes:

L = LCTC + λLCR, (3)

where λ controls the regularization strength. Since the CR loss actively encourages predicting the input sequence exactly, it is important to keep λ much smaller than the CTC loss to avoid suppressing necessary edits.

##### 3.7. Inference

At inference, we extract acoustic embeddings H and the CTC hypothesis from the frozen speech encoder. We retokenize the hypothesis and interleave it with blank symbols to form x˜, which is then concatenated with the projected acoustic embeddings and fed to the bidirectional LLM editor. The editor produces output logits L for each position in the interleaved sequence. We apply CTC greedy decoding (argmax followed by blank removal and deduplication) to obtain the final edited transcript xˆ.

Crucially, all predictions are made in parallel across the sequence, enabling significantly faster inference compared to autoregressive decoding. The computational cost is dominated by the single forward pass through the LLM, which processes all positions simultaneously. Optionally, the editing process can be iterated by feeding xˆ back as input for additional refinement steps, though this comes at the cost of additional forward passes (see Sec. 5.5 for analysis).

4. Experiments

##### 4.1. Training

Model Architecture. The CTC encoder contains 440M parameters and is based on a 16-layer Conformer architecture [35] with block attention using a block size of 200 frames (corresponding to 4 seconds at a 50 Hz frame rate). The encoder operates on 16 kHz audio with stacked log-mel features (80 mel bins, 2-frame stacking) and self-conditioning at layer 8 [2]. The CTC vocabulary consists of 384 character-based units. We extract hidden layer representations from four intermediate layers of the CTC encoder (layers 4, 8, 12, and 16), which are concatenated along the feature dimension to provide multi-scale acoustic representations.

To map the acoustic representations, we train a 1-layer QFormer projector [36] that downsamples the concatenated hidden layers by 5x (from 15-frame windows to 3 queries). Each query consists of the mean-pooled representation of its corresponding 5-frame segment and cross-attends to the full 15frame window with learnable positional embeddings. The QFormer has hidden dimension 1024, feedforward dimension 2048, and 16 attention heads with head dimension 64. Our base language model is the lightweight Granite 4.0 1B base [37], which we adapt using LoRA with rank 128, applied to both attention and MLP layers. The total number of trainable parameters (projector + LoRA) is only 14M. For the CTC blank symbol ϵ, we reuse the EOS token from the LLM vocabulary to avoid modifying the vocabulary (any special token would suffice).

Training Procedure. Models are trained for 3 epochs (180K steps total) using the AdamW optimizer, with a peak learning rate of 3e-5 and a cosine schedule with 5% warmup and minimum learning rate of 1% of peak. We adopt the balanced sampling strategy of [2] where the α ∈ [0, 1] hyperparameter controls the dataset balancing level and set α = 0.65

(compared to α = 0.6 in [2]) for all experiments. The copying regularization weight is set to λ = 0.02 (a small value to avoid dominating the CTC loss). We use a global batch size of approximately 320 utterances, spread across 8 H100 GPUs. Training takes approximately 30 hours for 3 epochs on 8 H100 GPUs, compared to 24 hours for the AR baseline (1.25x longer due to the interleaved padding increasing sequence length).

##### 4.2. Datasets

Our training datasets include AMI (IHM+SDM) [38], VoxPopuli [39], YODAS (9K hours subset) [40], CommonVoice 15 [41], MLS [42], Earnings22 [43], Fisher [44], CallHome [45], and SwitchBoard [46]. We train on five languages: English, Spanish, French, German, and Portuguese. The combined training corpus comprises approximately 70K hours of speech across five languages. For evaluation, we use the official validation and test sets of the training datasets, as well as three additional test-only datasets: GigaSpeech [47], TED-LIUM [48], and SPGISpeech [49].

During training, we filter out utterances longer than 80 seconds and examples where the character-to-frame ratio exceeds 0.66. We apply SpecAugment (time and frequency masking) and noise augmentation with 25% probability using MUSAN and FreeSound backgrounds at SNR levels between -5 and 20 dB.

##### 4.3. Evaluation Protocol

We evaluate NLE in terms of transcription accuracy and inference efficiency, focusing on the WER-RTFx trade-off. All RTFx measurements are obtained using offline batched inference on a single H100 GPU with bf16 precision. For our AR and NAR models, we use a batch size of 96 utterances. To assess performance in latency-critical conversational settings, we additionally measure RTFx with batch size 1, where the inability to parallelize computations in autoregressive models becomes more severe. For other models on the Open ASR leaderboard, we use the open-source optimized implementations from the leaderboard repository, measured on the same H100 hardware to ensure fair comparison. We use CTC greedy decoding (argmax+collapse) for CTC-based models, as beam search yields marginal improvements at significantly higher inference cost (we observed marginal WER improvement with 510x slower RTFx). For autoregressive models, we use greedy autoregressive decoding. WER is computed in lowercase, applying Whisper normalization and using the jiwer library. We report several aggregated results: Open ASR leaderboard [8] is the average WER across the datasets included in the leaderboard; the CV and MLS metrics are averages across all reported CommonVoice 15 and Multilingual LibriSpeech subsets, respectively.

5. Results

##### 5.1. Open ASR Leaderboard Comparison

We compare NLE against the top-6 models on the Open ASR leaderboard at the time of submission: Canary-Qwen 2.5B [50], Granite Speech 2B and 8B [2], Phi-4 Multimodal [51], Qwen3ASR 1.7B [52], and Parakeet 0.6B [50]. Figure 1 shows that NLE is on the Pareto frontier, ranking 4th in average WER (5.79%) while achieving 1722 RTFx. Among the top-6 models, only Parakeet (4264 RTFx) is faster, but at the cost of higher WER (6.05%). NLE demonstrates a superior accuracy-

Table 1: WER (%, lower is better) and RTFx (higher is better) comparison. NLE competes with AR accuracy while being 4×/27× faster in batched/single-utterance inference.

Dataset NLE AR CTC

AMI-IHM 8.3 8.6 9.4 AMI-SDM 21.4 23.8 24.4 CV15-DE 5.6 4.7 6.3 CV15-EN 7.3 7.1 9.5 CV15-ES 5.0 4.1 5.5 CV15-FR 8.2 7.2 10.8 CV15-PT 3.0 2.7 3.4 EARNINGS 10.0 10.1 11.5 GIGASPEECH 10.1 10.0 10.6 LS-CLEAN 1.4 1.5 1.7 LS-OTHER 3.1 3.1 3.7 MLS-DE 4.7 4.5 4.9 MLS-EN 4.8 4.7 5.7 MLS-ES 3.5 3.1 3.7 MLS-FR 4.6 4.5 5.6 MLS-PT 10.0 10.1 8.5 SPGI 3.5 3.5 4.5 TED-LIUM 3.9 3.7 3.9 VOX 6.2 6.2 7.1

Aggregate Metrics

Average (All 19) 6.54 6.48 7.40 Open ASR Average 5.79 5.82 6.55 CV Average 5.79 5.18 7.10 MLS Average 5.51 5.39 5.66

Speed Metrics RTFx: Batch size 96 1722 430 2584 RTFx: Batch size 1 322 12 760

speed tradeoff compared to other LLM-based systems, achieving competitive accuracy with 2-10x faster inference than models like Canary-Qwen, Granite Speech, Qwen3-ASR, and Phi4 Multimodal. Notably, NLE is the only model on the Pareto frontier on the Open ASR leaderboard that is multilingual, supporting five languages (English, Spanish, French, German, and Portuguese), while Canary-Qwen and Parakeet are English-only models.

5.1.1. NLE++ (Enhanced Training)

We additionally trained NLE++, a variant with the same architecture and data but with several training improvements: a larger projector (2 layers, 2× hidden and feedforward dimensions) with input/output dropout of 0.1, LoRA rank increased from 128 to 160, a 2× higher peak learning rate (6e-5), 5 training epochs (vs. 3), a 2× larger global batch size (640 utterances across 16 H100 GPUs on 2 nodes), and maximum audio length extended to 120 seconds (vs. 80 seconds). The larger projector and LoRA rank scale the total number of trainable parameters to 280M (160M projector + 120M LoRA), compared to 14M in NLE. As NLE++ differs in training budget and configuration from the controlled setup in Section 5, we report it separately in Table 2. NLE++ achieves 5.67% Open ASR WER – a 0.12% absolute improvement over NLE and improves the all-19 average from 6.54% to 6.44%. RTFx decreases modestly to 1630 (vs. 1722) due to the larger projector, but NLE++ remains on the Pareto frontier (Figure 1). These results suggest

- Table 2: Aggregate WER (%) and RTFx comparison of NLE vs. NLE++.

NLE NLE++ Aggregate Metrics

Average (All 19) 6.54 6.44 Open ASR Average 5.79 5.67 CV Average 5.79 5.41 MLS Average 5.51 5.76

Speed RTFx: Batch size 96 1722 1630 RTFx: Batch size 1 322 310

that NLE benefits from scaling up training compute and model capacity, pointing toward further improvements with additional resources.

##### 5.2. Controlled Evaluation

We conduct a controlled comparison against two baselines using the same training setup:

- • CTC encoder only: Greedy decoding from the pretrained speech encoder.
- • Controlled AR baseline: To enable a fair and controlled comparison between AR and NAR approaches, we train an AR LLM-based ASR system using the same encoder, projector, LLM backbone, datasets, balanced sampling strategy, and optimization setup described above. The systems differ only in the decoding strategy: autoregressive next-token prediction versus non-autoregressive editing. The AR baseline projects acoustic embeddings using the same projector, then concatenates them with the ground-truth transcript tokens wrapped with BOS and EOS tokens from the LLM vocabulary. The model is trained using standard next-token prediction with cross-entropy loss in causal attention mode. At inference, the model generates transcripts autoregressively using greedy decoding.

Table 1 presents results from our controlled evaluation across 19 test datasets covering 5 languages. NLE consistently outperforms the CTC encoder baseline, reducing average WER from 7.40% to 6.54% while maintaining high inference speed. NLE consistently improves over CTC, outperforming it on 17 out of 19 test sets. Compared to the controlled AR baseline, NLE achieves comparable accuracy with significantly faster inference: 4x speedup in batched scenarios (1722 vs 430 RTFx) and 27x speedup in single-utterance settings (322 vs 12 RTFx). The latency advantage is particularly pronounced in real-time conversational applications where batching is not possible, as the autoregressive model’s inability to parallelize becomes a critical bottleneck. We note that NLE underperforms the AR baseline on CommonVoice subsets, showing slightly worse performance in multilingual settings, possibly because the CTC encoder’s training data is predominantly English, resulting in weaker non-English hypotheses that NLE must then correct, compounded by the LLM tokenizer’s English-centric BPE vocabulary. These results confirm the efficacy of our proposed approach.

0.50

| |NLE<br><br>NoCR<br><br>EndPadding<br><br>NoAudioEmb<br><br>NoCTCHyp NoBidirect NoLora<br><br>| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

0.45

0.40

Valloss

0.35

0.30

0.25

0.20

20k 40k 60k 80k 100k 120k 140k 160k

Iterations

Figure 3: Validation loss over training steps for ablation study (see Section 5.3). NLE (full model) achieves the lowest validation loss, confirming that each design choice contributes positively to overall performance.

##### 5.3. Ablation Study

Figure 3 analyzes the impact of key design choices on validation loss, which consistently correlates with better WER on test sets.

Copying regularization (CR) loss. Removing the CR loss (NoCR) degrades validation loss, even though the reported loss includes the non-negative CR term, meaning the full model still reaches a lower loss despite the added penalty. This demonstrates that CR improves both training stability and final performance by explicitly encouraging the identity mapping bias.

Bidirectional attention. Restricting the LLM to causal attention (NoBidirect) limits its ability to leverage future context for editing. Notably, enabling bidirectional attention consistently improves convergence and validation loss, despite the LLM being pretrained with causal attention, confirming that future context is critical for effective non-autoregressive editing.

Padding strategy. Interleaved padding significantly outperforms end-of-sequence padding (EndPadding). End padding causes large token displacements during insertions, while interleaved padding preserves token locality, aligning with the Transformer’s locality bias.

Audio conditioning. Removing acoustic embeddings (NoAudioEmb; multiplying them by 0) substantially degrades performance, highlighting the necessity of acoustic grounding for accurate hypothesis refinement.

Hypothesis Conditioning. Removing the input hypothesis as conditioning (NoCTCHyp; replacing it with a sequence of blanks with the same length) also degrades performance, suggesting the model struggles with predicting the entire hypothesis from scratch.

LoRA adaptation. Keeping the LLM frozen (NoLoRA) degrades validation loss, highlighting that LoRA adaptation is important for optimal performance.

##### 5.4. Blank Density

We investigate the impact of varying the density of insertion slots in the interleaved sequence. Instead of inserting a blank between every token (Every 1), we experiment with inserting blanks every 2/3 tokens. Table 3 shows that reducing blank density degrades accuracy while providing minimal speedup, as the sequence length is dominated by acoustic tokens rather than text tokens. Inserting blanks between every token (Every

- Table 3: Impact of blank density (adding an insertion slot every K tokens) on the average WER and RTFx. Inserting a blank between every token (Every 1) achieves superior WER with negligible RTFx drop compared to sparser insertion strategies.

Blank Density Average WER (%) RTFx

- Every 1 (NLE) 6.54 1722
- Every 2 6.80 1750
- Every 3 6.91 1770

- Table 4: Average WER and RTFx for multi-step editing. Step

- 0 corresponds to the CTC baseline (no editing). A 2-step edit yields negligible accuracy gain at the cost of reduced RTFx, while a 3-step edit degrades performance below the single-step result.

Editing Steps Average WER (%) RTFx

- 0 (CTC only) 7.40 2584
- 1 6.54 1722
- 2 6.53 1259
- 3 6.59 1082

- 1) provides the best accuracy-speed tradeoff.

##### 5.5. Multi-Step Editing

We investigate whether iteratively applying the editor can further improve transcription accuracy. In multi-step editing, the editor’s output logits are decoded using CTC (argmax + collapse) and re-interleaved with insertion slots before being fed back as input to the editor for another round of refinement, while keeping the acoustic embeddings fixed. This process can be repeated multiple times, with each step potentially correcting errors introduced or missed in previous iterations. Table 4 suggests that applying a second editing step yields modest improvements, at the cost of reduced inference speed. However, further iterations show diminishing returns, with the third step slightly degrading performance. This degradation can be attributed to distribution mismatch: the editor is trained on CTC hypotheses, but at inference iteratively processes its own outputs, which have different error characteristics. Text augmentation strategies during training could potentially address this mismatch and improve multi-step refinement. For most applications, singlestep editing provides the best accuracy-speed tradeoff.

##### 5.6. Error Analysis

We analyze the types of errors made by different models to understand their behavior. Figure 4 shows the decomposition of WER into insertions, deletions, and substitutions for CTC, NLE, and AR models, averaged across all datasets as well as for two test-cases: AMI-SDM and MLS-PT. On average, the AR model exhibits the highest insertion rate, suggesting potential hallucinations when acoustic evidence is weak. The challenging AMI-SDM dataset illustrates this pattern more dramatically, with AR producing significantly more insertions. In contrast, NLE shows the highest deletion rate and lowest insertion rate, reflecting a more conservative editing strategy that prefers deletions over insertions.

Interestingly, on MLS-PT, both NLE and AR fail to improve over CTC, with NLE showing particularly high substitu-

Insertion

Deletion

Substitution

| |
|---|

| |
|---|

Average

AMI-SDM

MLS-PT

8

25

10

20

6

8

10.9

12.3

7.6

4.7

WER(%)

3.9

7.7

3.8

15

7.7

6

4

6.3

10

6.7

4

7.5

11.8

2

1.5

1.6

1.9

5

2

1.4

1.5

1.4

6.2

4.5

1.2

1.1

0.8

1.0

0.8

2.0

0.8

0

0

0

CTC NLE AR

CTC NLE AR

CTC NLE AR

- Figure 4: Insertion, deletion and substitution rates (%) for three conditions: average across all datasets, AMI-SDM, and MLSPT.

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

0 10 20 30 40 50 60 70

Time (%)

Decode

LLM

Projector

Retokenization

Encoder

1.4

30.1

1.5

1.1

65.9

- Figure 5: Inference time breakdown across the different stages of NLE. The encoder dominates at 66% of total time, with the LLM contributing ∼30%, and all remaining stages under 4%.

tion errors. The MLS-PT training set is very small, making this dataset particularly challenging and potentially leading to overfitting or insufficient adaptation to Portuguese. This highlights that when the CTC hypothesis quality is relatively good, the editing approach may introduce errors rather than corrections, emphasizing the importance of the initial hypothesis quality.

##### 5.7. Qualitative Analysis

To illustrate the types of corrections performed by NLE, Table 5 presents representative examples comparing CTC outputs with NLE predictions. The examples demonstrate how NLE corrects various error types: incorrect writing of several words (Example 1), substitution corrections (Example 2), and completing plausible content from severely corrupted input (Example 3). Examples 4-5 showcase multilingual capabilities, with NLE successfully correcting German and French transcripts. Notably, in the French example, NLE removes a long preamble (book title, chapter, and narrator information) that was spoken but should not be transcribed according to the ground truth labels, demonstrating that the model has learned to filter out metadata content based on the training data conventions. These examples highlight the model’s ability to leverage both acoustic embeddings and linguistic priors from the pretrained LLM to produce more accurate transcripts across multiple languages.

##### 5.8. Inference Time Breakdown

Figure 5 shows the distribution of inference time across different stages of NLE. The encoder forward pass dominates at 66%

- Table 5: Editing examples showing CTC outputs and NLE corrections. Errors are highlighted in red, corrections in blue, deletions of words in the reference are colored in orange.

- Example 1 REF: this thing we are going to design is a new remote control CTC: thes ting weare ona design ish a new remote control NLE: this thing we are going to design a new remote control

- Example 2 REF: often prevented from going to school they are forced to work eighteen to twenty hours per day CTC: often prevented from going to school they are forced set to work eighteen o twenty hours per day NLE: often prevented from going to school they are forced to work eighteen to twenty hours per day

- Example 3 REF: both are extremely important but innovation needs to get an extra push CTC: i ar h extremely important but innovation needs to get an extra push NLE: these are extremely important but innovation needs to get an extra push

- Example 4 REF: er wurde auf anderen kontinenten ubrigens bereits als wichtiger trendsetter gefeiert CTC: er wurde auf anderen kontinenten ubrigens bereitas wichtiger trensitte gefeiert NLE: er wurde auf anderen kontinenten ubrigens bereits als wichtige trente gefeiert

- Example 5 REF: oreille quand no´e eut rassembl´e les animaux devant l’arche il se dit CTC: chapitre xvi te ab c petit comte d jules maitre enregistrer pour l’ybrevoxe poir orge oreille quand no´ee eut rassembl´e les animaux devant l’arche il se dit NLE: oreilles quand no´e eut rassembl´e les animaux devant l’arche il se dit

of total time, accounting for roughly 2× more compute than the LLM. We note that the LLM operates on a 5× shorter sequence relative to the encoder output (10Hz after downsampling vs. 50Hz), which explains this result. The remaining stages – retokenization, projection, and output decoding – together account for only 4% of inference time. In contrast, an autoregressive system would see the LLM dominate due to sequential token generation, whereas NLE’s parallel decoding keeps the LLM contribution modest.

### 6. Discussion

We introduced NLE, a non-autoregressive LLM-based ASR system that reframes speech recognition as conditional transcript editing. NLE leverages a pretrained CTC encoder to produce both acoustic embeddings and an initial hypothesis, then applies a bidirectional LLM editor with an interleaved insertion slot representation. NLE achieves accuracy comparable to autoregressive baselines while delivering up to 4x faster inference in batched scenarios, with speedup advantages increasing further in single-utterance latency-critical conversational settings. Our approach demonstrates that pretrained LLMs can

be effectively adapted for non-autoregressive editing through lightweight LoRA adapters and attention mask modifications. The interleaved insertion slot scheme enables efficient handling of insertions while preserving the identity mapping bias of Transformer architectures.

##### 6.1. Limitations

NLE is less flexible than autoregressive models in handling tasks where the output significantly diverges from the input hypothesis. While NLE excels at correcting local errors, it is less likely to generalize to tasks requiring substantial changes to the hypothesis, such as spoken question answering, where the expected response is vastly different from the transcript. Moreover, when the CTC encoder and the LLM use different tokenizers, NLE requires transferring the hypothesis from GPU to CPU for retokenization and back, adding minor latency overhead. While using the LLM’s tokenizer for CTC training would avoid this overhead, it would significantly increase encoder training costs.

##### 6.2. Future Work

Several promising directions could further improve NLE’s capabilities. Text augmentation strategies during training could address the distribution mismatch observed in multi-step editing (Section 5.5), potentially enabling more effective iterative refinement. Such augmentations could include synthetic errors, paraphrasing, or using the model’s own predictions as training inputs to better match inference-time conditions. Combining the editing approach with mask-predict strategies could further improve transcription accuracy by allowing the model to iteratively refine uncertain predictions while maintaining parallel inference. This hybrid approach could leverage the strengths of both paradigms: the efficiency of single-pass editing for most tokens and the refinement capability of masking for challenging regions.

An additional promising direction is to restructure the LLM architecture to process audio and text in separate layers with cross-attention between modalities, reducing computational complexity from quadratic to linear with respect to audio length. This would be particularly beneficial for long-form audio processing.

Other avenues for future research include:

- • Investigating the use of the underlying LLM as a language model for CTC beam decoding to provide stronger linguistic priors during initial hypothesis generation.
- • Joint fine-tuning of the encoder and editor for end-to-end optimization (though this would require careful training strategies to maintain acoustic modeling performance).
- • Extending the approach to streaming scenarios through mechanisms that handle partial hypotheses and incomplete acoustic context or chunk-based processing strategies that maintain parallel inference advantages within each chunk.

### 7. References

- [1] C. Tang, W. Yu, G. Sun, X. Chen, T. Tan, W. Li, L. Lu, Z. MA, and C. Zhang, “Salmonn: Towards generic hearing abilities for large language models,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=14rn7HpKVk
- [2] G. Saon, A. Dekel, A. Brooks, T. Nagano, A. Daniels, A. Satt, A. Mittal, B. Kingsbury, D. Haws, E. Morais et al., “Granite-

- speech: open-source speech-aware llms with strong english asr capabilities,” arXiv preprint arXiv:2505.08699, 2025.
- [3] A. Graves, S. Fern´andez, F. Gomez, and J. Schmidhuber, “Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks,” in Proceedings of the 23rd international conference on Machine learning, 2006, pp. 369–376.
- [4] S. Gong, S. Agarwal, Y. Zhang, J. Ye, L. Zheng, M. Li, C. An, P. Zhao, W. Bi, J. Han, H. Peng, and L. Kong, “Scaling diffusion language models via adaptation from autoregressive models,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview. net/forum?id=j1tSLYKwg8
- [5] P. BehnamGhader, V. Adlakha, M. Mosbach, D. Bahdanau, N. Chapados, and S. Reddy, “Llm2vec: Large language models are secretly powerful text encoders,” in First Conference on Language Modeling, 2024.
- [6] C. Saharia, W. Chan, S. Saxena, and M. Norouzi, “Nonautoregressive machine translation with latent alignments,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 1098–1108.
- [7] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” ICLR, vol. 1, no. 2, p. 3, 2022.
- [8] V. Srivastav, S. Zheng, E. Bezzam, E. L. Bihan, A. Moumen, and S. Gandhi, “Open asr leaderboard: Towards reproducible and transparent multilingual speech recognition evaluation,” arXiv preprint arXiv:2510.06961, 2025.
- [9] Z. Ma, G. Yang, Y. Yang, Z. Gao, J. Wang, Z. Du, F. Yu, Q. Chen, S. Zheng, S. Zhang et al., “An embarrassingly simple approach for llm with strong asr capacity,” arXiv preprint arXiv:2402.08846, 2024.
- [10] Y. Chu, J. Xu, Q. Yang, H. Wei, X. Wei, Z. Guo, Y. Leng, Y. Lv, J. He, J. Lin et al., “Qwen2-audio technical report,” arXiv preprint arXiv:2407.10759, 2024.
- [11] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and

I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International conference on machine learning. PMLR, 2023, pp. 28492–28518.

- [12] A. Graves, “Sequence transduction with recurrent neural networks,” arXiv preprint arXiv:1211.3711, 2012.
- [13] H. Xu, F. Jia, S. Majumdar, H. Huang, S. Watanabe, and B. Ginsburg, “Efficient sequence transduction by jointly predicting tokens and durations,” in International Conference on Machine Learning. PMLR, 2023, pp. 38462–38484.
- [14] D. Rekesh, N. R. Koluguri, S. Kriman, S. Majumdar, V. Noroozi, H. Huang, O. Hrinchuk, K. Puvvada, A. Kumar, J. Balam et al., “Fast conformer with linearly scalable attention for efficient speech recognition,” in 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2023, pp. 1–8.
- [15] A. Hannun, “Sequence modeling with ctc,” Distill, 2017, https://distill.pub/2017/ctc.
- [16] W. Chan, N. Jaitly, Q. V. Le, and O. Vinyals, “Listen, attend and spell,” arXiv preprint arXiv:1508.01211, 2015.
- [17] Y. Leng, X. Tan, W. Liu, K. Song, R. Wang, X.-Y. Li, T. Qin, E. Lin, and T.-Y. Liu, “Softcorrect: Error correction with soft detection for automatic speech recognition,” in proceedings of the AAAI conference on artificial intelligence, vol. 37, no. 11, 2023, pp. 13034–13042.
- [18] Y. Higuchi, S. Watanabe, N. Chen, T. Ogawa, and T. Kobayashi, “Mask ctc: Non-autoregressive end-to-end asr with ctc and mask predict,” in Proc. Interspeech 2020, 2020, pp. 3655–3659.
- [19] Z. Fang, R. Zhang, Z. He, H. Wu, and Y. Cao, “Nonautoregressive chinese asr error correction with phonological training,” in Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2022, pp. 5907–5917.

- [20] M. Ghazvininejad, O. Levy, Y. Liu, and L. Zettlemoyer, “Maskpredict: Parallel decoding of conditional masked language models,” in Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), 2019, pp. 6112–6121.
- [21] W. Chan, C. Saharia, G. Hinton, M. Norouzi, and N. Jaitly, “Imputer: Sequence modelling via imputation and dynamic programming,” in International Conference on Machine Learning. PMLR, 2020, pp. 1403–1413.
- [22] Y. Leng, X. Tan, L. Zhu, J. Xu, R. Luo, L. Liu, T. Qin, X. Li, E. Lin, and T.-Y. Liu, “Fastcorrect: Fast error correction with edit alignment for automatic speech recognition,” Advances in Neural Information Processing Systems, vol. 34, pp. 21708–21719, 2021.
- [23] E. A. Chi, J. Salazar, and K. Kirchhoff, “Align-refine: Nonautoregressive speech recognition via iterative realignment,” in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2021, pp. 1920–1927.
- [24] L. Mangu, E. Brill, and A. Stolcke, “Finding consensus in speech recognition: word error minimization and other applications of confusion networks,” Computer Speech & Language, vol. 14, no. 4, pp. 373–400, 2000.
- [25] H. Sak, M. Saraclar, and T. G¨ung¨or, “On-the-fly lattice rescoring for real-time automatic speech recognition.” in Interspeech, 2010, pp. 2450–2453.
- [26] K. Hu, T. N. Sainath, R. Pang, and R. Prabhavalkar, “Deliberation model based two-pass end-to-end speech recognition,” in ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020, pp. 7799–7803.
- [27] R. Ma, M. Qian, P. Manakul, M. Gales, and K. Knill, “Can generative large language models perform asr error correction?” arXiv preprint arXiv:2307.04172, 2023.
- [28] R. Ma, M. Qian, M. Gales, and K. Knill, “Asr error correction using large language models,” IEEE Transactions on Audio, Speech and Language Processing, 2025.
- [29] R. Ma, M. J. Gales, K. M. Knill, and M. Qian, “N-best t5: Robust asr error correction using multiple input hypotheses and constrained decoding space,” arXiv preprint arXiv:2303.00456, 2023.
- [30] J. Gu, J. Bradbury, C. Xiong, V. O. Li, and R. Socher, “Nonautoregressive neural machine translation,” in International Conference on Learning Representations, 2018.
- [31] J. Gu, C. Wang, and J. Zhao, “Levenshtein transformer,” Advances in neural information processing systems, vol. 32, 2019.
- [32] Y. Zhang, Y. Zhang, L. Cui, and G. Fu, “Non-autoregressive text editing with copy-aware latent alignments,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 7075–7085.
- [33] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.
- [34] ——, “Identity mappings in deep residual networks,” in European conference on computer vision. Springer, 2016, pp. 630–645.
- [35] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu et al., “Conformer: Convolutionaugmented transformer for speech recognition,” in Proc. Interspeech 2020, 2020, pp. 5036–5040.
- [36] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19730–19742.
- [37] IBM Research, “Granite 4.0 language models,” https://github. com/ibm-granite/granite-4.0-language-models, 2025, accessed: 2025-10-01.

- [38] W. Kraaij, T. Hain, M. Lincoln, and W. Post, “The ami meeting corpus,” in Proc. International Conference on Methods and Techniques in Behavioral Research, 2005, pp. 1–4.
- [39] C. Wang, M. Riviere, A. Lee, A. Wu, C. Talnikar, D. Haziza, M. Williamson, J. Pino, and E. Dupoux, “Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2021, pp. 993–1003.
- [40] X. Li, S. Takamichi, T. Saeki, W. Chen, S. Shiota, and S. Watanabe, “Yodas: Youtube-oriented dataset for audio and speech,” in 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2023, pp. 1–8.
- [41] R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. Tyers, and G. Weber, “Common voice: A massively-multilingual speech corpus,” in Proceedings of the twelfth language resources and evaluation conference, 2020, pp. 4218–4222.
- [42] V. Pratap, Q. Xu, A. Sriram, G. Synnaeve, and R. Collobert, “Mls: A large-scale multilingual dataset for speech research,” in Proc. Interspeech 2020, 2020, pp. 2757–2761.
- [43] M. Del Rio, P. Ha, Q. McNamara, C. Miller, and S. Chandra, “Earnings-22: A practical benchmark for accents in the wild,” arXiv preprint arXiv:2203.15591, 2022.
- [44] C. Cieri, D. Miller, and K. Walker, “The Fisher corpus: A resource for the next generations of speech-to-text,” in Proceedings of the 4th International Conference on Language Resources and Evaluation (LREC), 2004, pp. 69–71.
- [45] A. Canavan, D. Graff, and G. Zipperlen, “Callhome american english speech,” Web Download, Philadelphia, 1997, lDC97S42.
- [46] J. J. Godfrey, E. C. Holliman, and J. McDaniel, “Switchboard: Telephone speech corpus for research and development,” in [Proceedings] ICASSP-92: 1992 IEEE International Conference on Acoustics, Speech, and Signal Processing, vol. 1. IEEE, 1992, pp. 517–520.
- [47] G. Chen, S. Chai, G.-B. Wang, J. Du, W.-Q. Zhang, C. Weng, D. Su, D. Povey, J. Trmal, J. Zhang et al., “Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio,” in Proc. Interspeech 2021, 2021, pp. 3670–3674.
- [48] A. Rousseau, P. Del´eglise, and Y. Esteve, “Ted-lium: an automatic speech recognition dedicated corpus.” in LREC, 2012, pp. 125– 129.
- [49] P. K. O’Neill, V. Lavrukhin, S. Majumdar, V. Noroozi, Y. Zhang, O. Kuchaiev, J. Balam, Y. Dovzhenko, K. Freyberg, M. D. Shulman et al., “Spgispeech: 5,000 hours of transcribed financial audio for fully formatted end-to-end speech recognition,” in Proc. Interspeech 2021, 2021, pp. 1434–1438.
- [50] M. Sekoyan, N. R. Koluguri, N. Tadevosyan, P. Zelasko, T. Bartley, N. Karpov, J. Balam, and B. Ginsburg, “Canary-1b-v2 & parakeet-tdt-0.6 b-v3: Efficient and high-performance models for multilingual asr and ast,” arXiv preprint arXiv:2509.14128, 2025.
- [51] A. Abouelenin, A. Ashfaq, A. Atkinson, H. Awadalla, N. Bach, J. Bao, A. Benhaim, M. Cai, V. Chaudhary, C. Chen et al., “Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras,” arXiv preprint arXiv:2503.01743, 2025.
- [52] X. Shi, X. Wang, Z. Guo, Y. Wang, P. Zhang, X. Zhang, Z. Guo, H. Hao, Y. Xi, B. Yang et al., “Qwen3-asr technical report,” arXiv preprint arXiv:2601.21337, 2026.

