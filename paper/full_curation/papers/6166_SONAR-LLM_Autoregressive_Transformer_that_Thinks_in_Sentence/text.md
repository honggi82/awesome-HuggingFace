# arXiv:2508.05305v2[cs.CL]25May2026

## SONAR-LLM: Autoregressive Transformer that Thinks in Sentence Embeddings and Speaks in Tokens

### Nikita Dragunov1, Temurbek Rahmatullaev1, Elizaveta Goncharova1, Nikita Kurdiukov2, Aysel Mirzoeva3, Anna Borisiuk4, Andrey Kuznetsov1, Anton Razzhigaev1 1FusionBrain Lab, AXXX 2T-Tech 3MSU 4DS-NLP Group, AXXX

Correspondence: nikitadragunovjob@gmail.com

### Abstract

The recently proposed Large Concept Model (LCM) generates text by predicting a sequence of sentence-level embeddings and training with either mean–squared error or diffusion objectives. We present SONAR-LLM, a decoderonly transformer that thinks in the same continuous SONAR embedding space yet is supervised through token-level cross-entropy propagated via the frozen SONAR decoder. This hybrid objective retains the semantic abstraction of LCM while eliminating its diffusion sampler and restoring a likelihood-based training signal. Across model sizes from 39 M to 1.3 B parameters, SONAR-LLM attains competitive generation quality. We report scaling trends, ablations, benchmark results and a theoretical analysis of inference FLOPs for context sizes up to 1 million tokens. We release the complete training code and pretrained checkpoints to foster reproducibility and future research.

### 1 Introduction

Most autoregressive language models learn tokenby-token: they minimize cross-entropy over a discrete vocabulary and emit one token per forward step (Brown et al., 2020; Raffel et al., 2020). This fine-grained decoding is simple to train and evaluate but becomes a throughput bottleneck for long sequences. Meta’s recently introduced Large Concept Model (LCM) (Barrault et al., 2024) addresses the latency issue by predicting a much shorter trajectory of sentence-level embeddings trained with diffusion or MSE objective. Yet removing tokenlevel likelihoods makes optimization less stable.

We present SONAR-LLM, an autoregressive decoder-only transformer that keeps LCM’s “think in sentence embeddings” idea while leveraging the advantages of cross-entropy learning. The model predicts SONAR (Duquenne et al., 2023) sentence embeddings but propagates loss through the frozen SONAR decoder down to individual tokens, cou-

- - Trainable parameters

- - Frozen parameters

- - Predicted output

- - Given input

tok1 tok2 tok3

SONAR DECODER

- Sentence 2

SONAR ENCODER

- Sentence 3 embedding

<BOS> tok1 tok2

SONAR-LLM

SONAR ENCODER

Sentence 1

Figure 1: Architecture of SONAR-LLM. The model autoregressively predicts the next sentence embedding given a prefix of embeddings and decodes it via the frozen SONAR decoder.

pling continuous reasoning with discrete supervision. This yields a one-shot sentence generator that is efficient, diffusion-free, likelihood-consistent, and fast at inference time.

Our contributions are:

- 1. Token-Aware Embedding Objective. We introduce a training objective that backpropagates token-level cross-entropy through a frozen SONAR decoder, aligning continuous predictions with discrete targets.
- 2. Scaling Laws Analysis. We provide a detailed scaling law fit for validation losses across model sizes, quantifying the scaling exponents for LLM, LCMs, and SONAR-LLM architectures.

- 3. Text Quality Evaluation. We evaluate the quality of generated texts and primarily compare SONAR-LLM against LCM-based sentence-level models using standard NLG metrics (BLEU, ROUGE-L, METEOR) and a GPT-4o-based model-as-a-judge evaluation, demonstrating consistently higher text quality.
- 4. Summarization Evaluation. We compare models on summarization tasks using XSum and CNN/DM benchmarks, showing that SONAR-LLM matches or exceeds the performance of other sentence-level approaches.
- 5. Inference Efficiency Analysis. We provide a theoretical analysis of inference FLOPs for embedding-based sentence-level models, showing that operating on sentence segments yields favorable scaling on long sequences compared to token-level decoding.
- 6. Reproducible Open-Source Release. All training code, evaluation scripts, and model checkpoints are publicly released to facilitate follow-up research.1

### 2 Related Work

#### 2.1 Token-level autoregressive models.

Large language models are trained by next-token prediction with cross-entropy over a discrete vocabulary (Brown et al., 2020), inheriting the Transformer architecture (Vaswani et al., 2017). Recent research has explored alternatives to self-attention for faster long-sequence processing; for example, MAMBA replaces attention with selective statespace updates and achieves linear-time generation while matching Transformer quality (Gu and Dao, 2023), with MAMBA-2 further improving efficiency through structured state-space duality (Dao and Gu, 2024). Google’s TITANS architecture introduces a neural long-term memory module that learns to memorize at test time, combining attention-based short-term memory with adaptive long-term storage to scale beyond 2 M context tokens (Behrouz et al., 2025).

#### 2.2 Latent-variable text generators.

Continuous and discrete VAEs generate sentences from latent codes (Bowman et al., 2016). VectorQuantised VAE (VQ-VAE) models compress sentences into a short sequence of discrete indices and

1https://github.com/FusionBrainLab/SONAR-LLM

decode them with an autoregressive prior (van den Oord et al., 2017). The SONAR encoder–decoder extends this idea to a language-agnostic, multimodal sentence embedding space covering 200 languages (Duquenne et al., 2023). Meta’s Large Concept Model (LCM) builds an autoregressive prior over SONAR embeddings and investigates MSE, quantization and diffusion losses in that space (Barrault et al., 2024). Our SONAR-LLM also operates in SONAR space but reinstates token-level crossentropy by back-propagating through the frozen decoder.

- 2.3 Diffusion and discrete denoising models for text.

Diffusion-LM denoises continuous wordembedding sequences to enable controllable generation without left-to-right constraints (Li et al., 2022). Discrete Denoising Diffusion Probabilistic Models (D3PMs) corrupt token sequences and learn to reverse the process in discrete space (Austin et al., 2021). Recent work improves training with a score-entropy objective, narrowing the perplexity gap to autoregressive baselines (Lou et al., 2024). LLADA demonstrates that masked diffusion models can match autoregressive LLMs at scale: an 8B-parameter model trained from scratch achieves performance comparable to LLaMA3 8B across diverse benchmarks (Nie et al., 2025).

- 2.4 Flow and ODE-based generators.

Flow Matching trains continuous normalizing flows without expensive simulation and subsumes diffusion as a special case (Lipman et al., 2023). Applying flow matching to text, FLOWSEQ generates high-quality sentences in a handful of ODE steps, greatly accelerating sampling (Hu et al., 2024).

In summary, research has progressed from token-wise decoding to latent concept prediction (LCM), diffusion and flow-based models. SONARLLM bridges these by learning an autoregressive prior in sentence embedding space while retaining likelihood-based supervision.

### 3 SONAR-LLM

The proposed SONAR-LLM is an autoregressive decoder–only Transformer that operates directly in the SONAR sentence-embedding space while being supervised with token-level cross-entropy. The

overall architecture of our approach is illustrated in Fig. 1.

- 3.1 Pre-processing and Sentence Segmentation

We segment text into small units using the Punkt unsupervised sentence tokenizer implemented in NLTK (Kiss and Strunk, 2006). Each sentence st is encoded with the frozen multilingual SONAR encoder (Duquenne et al., 2023), yielding a fixedlength vector et ∈ Rd (d=1024 in all experiments). Given a prefix of sentence embeddings (e1,...,et−1), the model predicts the embedding eˆt of the next sentence. This predicted vector is then decoded using the frozen SONAR decoder, and the generated sentence is compared to the true next sentence st, which serves as the training target.

- 3.2 Model Architecture

SONAR-LLM is a decoder-only Transformer with the same layer pattern as Llama 3 (Llama Team, 2024) but an embedding vocabulary of size one: the model predicts a continuous vector rather than a discrete token at each step. Formally, given prefix e<t = (e1,...,et−1), the network outputs eˆt = fθ(e<t) ∈ Rd. We train variants from 39 M to 900 M parameters by scaling width and depth; all use rotary position encodings and RMS-norm.

- 3.3 Cross-Entropy Through the Frozen Decoder

To avoid MSE or diffusion objectives, yet keep likelihood-based training, we decode eˆt back to token logits with the frozen SONAR decoder D using teacher-forcing approach, i.e., generate the next token logits based on predicted sentence embedding eˆt and previous ground-truth tokens st,<i:

zt,i = D(eˆt,st,<i) ∈ R|V|.

We minimize standard cross-entropy between zt and the ground-truth token sequence of sentence st:

T

L = −

log pθ(st | e<t)

t=1

(1)

|st|

T

= −

log softmax(zt,i)st,i .

t=1

i=1

Back-propagation flows through D keeping SONAR frozen and reducing memory overhead. Teacher–forcing supplies the ground-truth embedding et at the next time step.

- Example 1 Prefix: Once upon a time, in a small yard, there was a small daisy. The daisy had a name. Generated: Her name was Daisy. Daisy lived in a garden full of flowers. She loved to play with her friends. One day, Daisy saw a big butterfly. The butterfly was very pretty. Daisy wanted to be friends with the butterfly. She said, "Hi, butterfly! Do you want to play with me?" The butterfly said, "Yes, I want to play with you!" Daisy and the butterfly played all day. They had so much fun. At the end of the day, Daisy was happy to go home. She knew she would have a new friend to play with.
- Example 2 Prefix: Once upon a time, in a small house, there lived a kind and honest girl named Lily. She loved to bake cakes for her friends and family. Generated: One day, Lily decided to make a big cake for her friend. Lily went to the store to mix cake, flour, and sugar. She mixed and mixed all the ingredients until it was time to bake. Then, she put the cake in the oven and waited for it to bake. When the cake was done, Lily took it out of the house and shared it with her friends. They all loved the cake and said, "Thank you, Lily!" Lily felt happy and proud that she could make her friends happy.

Figure 2: Examples of texts generated by SONAR-LLM 900 M.

#### 3.4 End of sequence

We append a special literal sentence "End of sequence." to every document and encode it once with the SONAR encoder to obtain eeot. At inference, generation halts when the cosine similarity between the latest predicted embedding and eeot exceeds τstop=0.98, or when Tmax = 32 sentences are produced.

### 4 Results

We trained large language models (LLMs) of five different scales (39 M, 100 M, 300 M, 600 M, and 900 M parameters) for four epochs each, using the Llama 3 architecture on the TINYSTORIES dataset (Eldan and Li, 2023). Each run was conducted on a server equipped with up to 8 NVIDIA A100 GPUs (80 GB). When reporting model sizes for LLMs, we included the embedding matrices in the parameter list, as these were fully trained. We also trained SONAR-LLM, MSE-based LCM, and diffusion-based LCM. For SONAR-LLM and MSE-based LCM models, we used the same archi-

Epoch 1 Epoch 2 Epoch 3 Epoch 4

###### LLM

###### SONAR-LLM (ours)

1.69

2.07

Cross-EntropyLoss

Cross-EntropyLoss

1.27

1.83

39M 100M 300M 600M 900M

11M 34M 170M 450M 700M

Model Size

Model Size

###### MSE LCM (Meta)

Diffusion LCM (Meta)

215

122

DiffusionLoss

MSELoss

205

99

11M 34M 170M 450M 700M

39M 100M 300M 600M 900M

Model Size

Model Size

Figure 3: Scaling laws: validation loss dynamics vs. number of trainable parameters.

tecture configurations as their LLM counterparts, but excluded the embedding and decoder parameters from training. As a result, these models contain fewer trainable parameters: 11 M, 34 M, 170 M, 450 M, and 700 M, respectively, having the same depth and width. For consistency, we refer to model sizes (39 M – 900 M) based on the full LLM configurations, even when the number of trainable parameters is smaller.

For the diffusion-based LCM, we employed the two-tower architecture from the original paper. Both LCM versions were trained using the official implementation provided by the authors (Barrault et al., 2024).

For each model configuration, we performed three independent training runs with three different random seeds, and report results averaged across these runs.

All models were trained using a cosine learning rate scheduler. We experimented with two learning rates: 5 × 10−4 and 1 × 10−3. Based on validation loss performance, we found 1×10−3 to be optimal for SONAR-LLM, while the other models (LLM, MSE-based LCM, and diffusion-based LCM) performed better with a learning rate of 5 × 10−4.

Examples of generated texts can be found in Figure 2. 4.1 Scaling laws

The empirical scaling properties of the evaluated architectures, illustrated in Fig. 3, offer insights into their efficiency in leveraging increased model

Table 1: Fitted scaling law parameters for each model at epoch 4.

Model a α b

LLM 1.44 × 105 0.724 1.04 SONAR-LLM (ours) 2.86 × 103 0.596 1.70 MSE LCM (Meta) 3.71 × 104 0.525 199 Diffusion LCM (Meta) 2.70 × 105 0.515 84.1

parameters and training compute. This analysis focuses on the implications of these observed validation loss dynamics for each model type.

We fitted the classical scaling law

##### L(N) = aN−α + b

to the validation losses of all models at epoch 4. For each architecture, the fit was performed jointly over all model sizes and three independent random seeds, resulting in 15 data points per model. The results (Table 1) confirm that SONAR-LLM achieves a strong scaling exponent (α ≈ 0.596), comparable to other embedding-based models. For all models, the scaling laws exhibit a good fit to the data, with an R2-score exceeding 0.97. This demonstrates that SONAR-LLM can efficiently leverage increased model capacity, benefiting from both semantic abstraction and effective scaling behavior.

#### 4.2 Automatic Evaluation with GPT-4o

We evaluated the performance of all four model types on a dataset consisting of 512 generated stories, assessing grammatical correctness, cre-

LLM-greedy LLM-beam SONAR-LLM MSE LCM Diffusion LCM

| | | | | | | |
|---|---|---|---|---|---|---|
| |11<br><br>34<br><br>170 450 700| | | | | |
| | | | | | | |
| |11 34<br><br>170<br><br>450 700| | | | | |
| | | | | | | |
| | | | | | | |

8

6

Grammar

4

2

0

39M 100M 300M 600M 900M

| | | | | | | |
|---|---|---|---|---|---|---|
| |11 34 170 450<br><br>700| | | | | |
| |11<br><br>34 170<br><br>450<br><br>700| | | | | |
| | | | | | | |
| | | | | | | |

6

4

Creativity

2

0

39M 100M 300M 600M 900M

8

170

450 700

34

6

11

Consistency

4

2

700

11 34

450

170

0

39M 100M 300M 600M 900M

6

170 450 700

34

11

4

Plot

2

700

11 34

170 450

0

39M 100M 300M 600M 900M

Figure 4: GPT-4o-based evaluation scores (grammar, creativity, consistency, plot) by model and size. Trainable parameter counts are shown above bars for SONAR-LLM and MSE LCM.

ativity, coherence, and plot consistency, following the methodology proposed by (Eldan and Li, 2023). To initiate story generation, we used the first two sentences from validation set stories as prompts. During evaluation, GPT-4o was shown the full story–including the prompt and the generated continuation–but was explicitly instructed to assess only the continuation starting from the third sentence. All models were evaluated after four epochs of training. For the LLM, we experimented with both greedy decoding and beam sampling with four beams.

As illustrated in Fig. 4, error bars correspond to three standard deviations (3σ), providing 99.7% confidence intervals for the mean scores across random seeds. For SONAR-LLM and MSE-based LCM models, the numbers shown above the bars indicate the corresponding counts of trainable parameters.

The classic token-level LLM demonstrates the best performance across all metrics. With beam sampling, the largest 900 M model achieves strong grammatical correctness (9.1), coherence (8.1), and plot consistency (7.0), with creativity scores around 6.2. Greedy decoding shows comparable results with slightly lower coherence.

Among the concept-based models, our proposed SONAR-LLM achieves the highest story genera-

tion quality, significantly outperforming both the diffusion-based and MSE-based LCM variants across all four metrics. The 900 M SONAR-LLM reaches grammatical correctness of 7.1, creativity of 5.2, coherence of 6.0, and plot consistency of 5.2. In contrast, Diffusion LCM shows moderate performance (grammar 3.9, creativity 5.0, coherence 2.9, plot 2.8), while MSE LCM exhibits substantially lower scores across all metrics (grammar 2.1, creativity 3.0, coherence 1.3, plot 1.2), suggesting that direct MSE regression in embedding space struggles to maintain narrative structure.

#### 4.3 NLG Metrics

To assess how effectively models capture the distribution of the original data, we evaluated standard NLG metrics, including BLEU, ROUGE-L, and METEOR. Specifically, we selected 512 stories from the validation set and used the first two sentences from each story as a context (short prefix) to generate the third sentence. We then measured similarity between the generated sentence and the corresponding reference sentence from the validation set using the aforementioned metrics. Additionally, we performed the same evaluation using half of each story in terms of sentence count as a context (long prefix), to investigate model performance under varying context lengths. Results are

LLM-greedy

LLM-beam SONAR-LLM MSE LCM Diffusion LCM

| | | |
|---|---|---|
| | | |

###### BLEU (short prefix)

###### BLEU (long prefix)

15.0

15.0

12.5

12.5

10.0

Score

10.0

7.5

7.5

5.0

5.0

2.5

2.5

101 102 103 Trainable parameters (M)

101 102 103 Trainable parameters (M)

###### ROUGE (short prefix)

###### ROUGE (long prefix)

40

35

35

30

Score

25

30

20

25

15

20

101 102 103 Trainable parameters (M)

101 102 103 Trainable parameters (M)

###### METEOR (short prefix)

METEOR (long prefix)

35

30

30

25

20

25

Score

15

20

10

15

101 102 103 Trainable parameters (M)

101 102 103 Trainable parameters (M)

Figure 5: NLG scores by model and size.

provided in Fig. 5, where shaded regions represent one standard deviation across random seeds.

The NLG evaluation demonstrates that SONARLLM approaches the performance of standard autoregressive LLMs across most metrics at the largest evaluated scale. For the 900 M-parameter models (with approximately 700 M trainable parameters for SONAR-LLM), SONAR-LLM matches LLM performance on ROUGE-L with short prefixes (39.4 vs. 39.4) and achieves competitive results on METEOR (32.9 vs. 34.2). With longer context, similar patterns hold: SONARLLM attains comparable ROUGE-L (35.2 vs. 35.7). In contrast, both MSE-based and diffusion-based LCMs exhibit substantially lower performance across all metrics at this scale, with MSE LCM achieving 26.1 ROUGE-L and Diffusion LCM reaching 31.2 ROUGE-L on short prefixes.

#### 4.4 Summarization Evaluation

Summarization is a vital benchmark for sentencelevel language models, as it directly assesses their capability to capture semantic content and produce coherent, structured text. Prior works on sentencelevel LLMs, including the original LCM paper, emphasized summarization as a crucial test of their abstraction and compression abilities. Motivated

by this, we evaluated SONAR-LLM and relevant baselines on standard abstractive summarization benchmarks.

We pretrained 1.3 B-parameter models (1.1 B trainable parameters for SONAR-LLM and MSE LCM, excluding embedding matrix) on a diverse mixture of datasets, including TINYTEXTBOOKS, TINYORCATEXTBOOKS, TINYSTRANGETEXTBOOKS, TEXTBOOKSAREALLYOUNEED (Jain et al., 2023), WIKITEXT-103-DETOKENIZED (Merity et al., 2017), XSUM (Narayan et al., 2018), CNNDAILYMAIL (Hermann et al., 2015). We then evaluated summarization performance on test examples from the XSUM and CNNDAILYMAIL datasets, generating the same number of sentences as in the reference summaries (typically one sentence for XSum and three sentences for CNN/DM). Results were measured using ROUGE-L and METEOR metrics.

The results in Table 2 indicate that SONARLLM substantially outperforms existing sentencelevel baselines (MSE LCM and Diffusion LCM) on both datasets, confirming its effectiveness for summarization tasks. Compared to token-level LLMs, SONAR-LLM achieves comparable performance on the more abstractive XSum dataset but remains behind on CNN/DM, which tends to favor more

Table 2: Summarization results on XSum and CNN/DM. Mean (std) across seeds.

Model XSum CNN/DM

R-L MET R-L MET

LLM-greedy 20.4(1.1) 16.5(1.1) 19.3(0.6) 15.2(1.1) LLM-beam 20.2(1.1) 16.9(1.1) 18.4(0.1) 16.9(0.4) SONAR-LLM (ours) 18.5(0.7) 14.5(0.5) 15.5(0.7) 10.0(0.4) Diffusion LCM (Meta) 12.0(0.2) 8.8(0.4) 10.1(0.1) 4.8(0.2) MSE LCM (Meta) 11.6(0.5) 8.5(0.2) 7.8(0.2) 3.8(0.1)

extractive approaches. These observations indicate that SONAR-LLM can be a promising approach for sentence-level tasks involving abstraction and semantic compression.

#### 4.5 Unfreezing the SONAR Decoder

In all main experiments, we keep the SONAR encoder and decoder frozen and train only the autoregressive prior in the sentence-embedding space, primarily for computational efficiency.

We additionally explored unfreezing the SONAR decoder when training 1.3 B-parameter models on the mixed pretraining corpus described in Section 4.4. Unfreezing the decoder consistently reduced validation cross-entropy by more than 10% across runs. However, this improvement did not translate into statistically significant gains in downstream text quality: summarization performance on XSum and CNN/DailyMail remained comparable to those obtained with a frozen decoder.

We further evaluated this setting on the RULER benchmark (Hsieh et al., 2024), focusing on the NIAH (needle-in-a-haystack) task that requires exact retrieval of numerical values from long contexts. For this experiment, we fine-tuned the pretrained 1.3 B-parameter models from Section 4.4 on the RULER dataset. In this regime, unfreezing the decoder substantially improved exact-match accuracy (from 21.6% to 41%), indicating a clear benefit for symbol-heavy tasks.

Overall, these results suggest that decoder unfreezing is unnecessary for standard naturallanguage generation, where the SONAR embedding space is already well aligned with textual data. In contrast, for tasks that require precise manipulation or reproduction of numerical or symbolic content–such as long-context retrieval, mathematics, or code–adapting the embedding-to-token interface by unfreezing the decoder can be beneficial.

- 102

- 103

- 104

- 105

- 106

- 107

- 108

LLM

InferenceFLOPs(GFLOPs)

SONAR-LLM

| |
|---|

102 103 104 105 106 Sequence length (tokens)

Figure 6: Theoretical inference FLOPs for autoregressive LLM and SONAR-LLM as a function of sequence length (log–log scale).

#### 4.6 Inference Efficiency

We compared the theoretical inference complexity in FLOPs of SONAR-LLM and a standard LLM depending on the input sequence length. The comparison was performed for models with identical architectures configured at 600 M parameters. In the case of SONAR-LLM, we assumed an average sentence length of 60 tokens and, in addition to the complexity of the main SONAR-LLM model, we also included the FLOPs of the SONAR encoder and decoder. The inference setup of SONAR-LLM follows the same structural principles as the MSEbased LCM proposed by (Barrault et al., 2024), suggesting that both models exhibit similar inference efficiency due to similar design.

The results presented in Fig. 6 indicate that, for shorter sequences, standard token-level LLMs maintain a computational advantage due to their optimized token-wise autoregressive decoding. However, as the input length increases, this advantage diminishes: starting from approximately 4096 tokens, SONAR-LLM surpasses the standard LLM in inference efficiency. This is attributable to SONARLLM’s design, which processes entire sentences as atomic units, thereby reducing the number of required decoding steps relative to token-based models. While the theoretical computational complexity remains quadratic for both approaches, the effective cost for SONAR-LLM grows much more slowly with sequence length because it operates on a compressed sequence of sentence embeddings. In practice, this yields an almost linear growth in FLOPs up to 1 million tokens, as the quadratic term is scaled by the inverse square of the average sentence length.

### 5 Conclusion

We presented SONAR-LLM, a decoder-only Transformer that predicts sentence embeddings and is supervised via token-level cross-entropy propagated through a frozen SONAR decoder. This approach retains the semantic abstraction of concept-based models like LCM while restoring a likelihood-based training signal.

As a proof of concept, we trained SONARLLM on the TINYSTORIES dataset. Across model sizes, SONAR-LLM exhibits stable optimization behavior and competitive scaling trends, matching or improving upon those observed for both MSE-based and diffusion-based LCMs. In GPT-

- 4o evaluations, SONAR-LLM outperformed both LCM variants in grammar, coherence, creativity, and plot consistency. On standard NLG metrics, SONAR-LLM demonstrated strong performance, consistently matching the standard tokenlevel LLM. It also outperformed both the MSEbased and diffusion-based LCMs across all prefix lengths, establishing it as a promising alternative for sentence-level generation tasks.

To broaden the evaluation scope, we pretrained all models on a diverse mixture of instructional and open-domain corpora. This enabled us to assess summarization capabilities on standard datasets such as XSum and CNN/DM. SONARLLM achieved consistently stronger performance than prior sentence-level baselines and demonstrated its ability to handle summarization tasks with competitive quality, further validating the effectiveness of our proposed objective in more realistic settings.

Finally, we examined the effect of unfreezing the frozen SONAR decoder. While decoder unfreezing consistently improves cross-entropy during pretraining, it does not lead to measurable gains on standard natural-language benchmarks such as summarization. However, on symbol-heavy longcontext retrieval tasks (RULER/NIAH), unfreezing the decoder yields a substantial improvement in exact-match accuracy, nearly doubling performance. These results indicate that decoder adaptation is unnecessary for semantically grounded text, but becomes important when precise numeric or alphanumeric reproduction is required.

Our theoretical FLOPs analysis further demonstrates that SONAR-LLM achieves superior inference efficiency for long contexts compared to token-level LLMs: beyond 4096 tokens, its total

computational cost grows almost linearly with sequence length up to 1 million tokens. Importantly, this effect results from operating on sentence-level segments, but the underlying complexity is still quadratic. This property enables such embeddingbased architectures to serve as a practical and scalable option for long-context generation.

### Limitations

While our study reveals clear trends among the evaluated model architectures, several limitations remain.

First, our evaluation of generation quality combines standard automatic metrics (BLEU, ROUGEL, METEOR) with GPT-4o-based assessments of grammar, coherence, creativity, and plot consistency. While the latter offers a stronger proxy for human judgment, it is still limited by the behavior and biases of the underlying model. A more complete evaluation would benefit from direct human annotation or broader qualitative analysis.

Second, our experiments are limited to Englishlanguage data, and the reported results may not directly generalize to other languages. At the same time, the underlying SONAR encoder–decoder is trained as a multilingual model and produces language-agnostic sentence embeddings, which suggests that the observed trends may extend beyond English. A thorough multilingual evaluation is left for future work.

### Impact Statement

This paper introduces a sentence-level approach to language generation that operates in a continuous embedding space while retaining token-level likelihood training. By predicting and processing text at the granularity of sentences rather than individual tokens, the proposed method enables more efficient handling of long contexts and offers an alternative abstraction for language modeling. The primary goal of this work is to advance research in efficient, scalable language generation and representation learning.

While the model reasons in sentence embeddings instead of tokens, it does not introduce fundamentally new generative capabilities compared to existing large language models. The types of outputs it can produce, and the potential misuse scenarios (e.g., generation of misleading or low-quality content), remain largely the same as those associated with standard LLMs. At the same time, operat-

ing at a higher level of abstraction may amplify or suppress certain properties of text (such as verbosity or structural coherence), which could affect downstream applications in subtle ways.

Overall, we do not identify novel ethical risks specific to this approach beyond those already present in current language models. We expect that the primary impact of this work will be to support research on efficient long-context modeling, abstraction in language generation, and architectures that reason over larger semantic units than tokens.

### References

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. 2021. Structured denoising diffusion models in discrete state spaces. arXiv preprint arXiv:2107.03006.

Loïc Barrault and 1 others. 2024. Large concept models: Language modeling in a sentence representation space. arXiv preprint arXiv:2412.08821.

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. 2025. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663.

Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafal Jozefowicz, and Samy Bengio. 2016. Generating sentences from a continuous space. In Proceedings of the Conference on Computational Natural Language Learning (CoNLL), pages 10–21.

Tom B. Brown and 1 others. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901.

Tri Dao and Albert Gu. 2024. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. In Proceedings of the 41st International Conference on Machine Learning.

Paul-Ambroise Duquenne, Holger Schwenk, and Benoît Sagot. 2023. Sonar: Sentence-level multimodal and language-agnostic representations. In Findings of the Association for Computational Linguistics: ACL, pages 4969–4983.

Ronen Eldan and Yuanzhi Li. 2023. Tinystories: How small can language models be and still speak coherent english? arXiv preprint arXiv:2305.07759.

Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.

Karl Moritz Hermann, Tomáš Koˇciský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. 2015. Teaching machines to read and comprehend. In Advances in Neural Information Processing Systems, volume 28, pages 1693–1701.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Yang Zhang. 2024. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654.

Vu T. Hu, Danyang Wu, Yuki M. Asano, Pascal Mettes, Basura Fernando, Björn Ommer, and Cees Snoek. 2024. Flow matching for conditional text generation in a few sampling steps. In Proceedings of the European Chapter of the Association for Computational Linguistics (EACL), Short Papers, pages 380–392.

Saurabh Jain, Shuo Sun, Xueliang Yu, and Colin Raffel.

2023. Textbooks are all you need. arXiv preprint arXiv:2306.11644.

Tibor Kiss and Jan Strunk. 2006. Unsupervised multilingual sentence boundary detection. Computational Linguistics, 32(4):485–525.

Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori Hashimoto. 2022. Diffusion-lm improves controllable text generation. In Advances in Neural Information Processing Systems.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. 2023. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747.

Llama Team. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Aaron Lou, Chenlin Meng, and Stefano Ermon. 2024. Discrete diffusion modeling by estimating the ratios of the data distribution. In Proceedings of the International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 32819–32848.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017. Pointer sentinel mixture models. In Proceedings of the International Conference on Learning Representations (ICLR).

Shashi Narayan, Shay B. Cohen, and Mirella Lapata. 2018. Don’t give me the details, just the summary! topic-agnostic text summarization. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1797–1807.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. 2025. Large language diffusion models. arXiv preprint arXiv:2502.09992.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67.

Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. 2017. Neural discrete representation learning. In Advances in Neural Information Processing Systems, pages 6306–6315.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008.

