# arXiv:2512.10411v5[cs.CL]26Mar2026

## SWAA: Sliding Window Attention Adaptation for Efficient and Quality Preserving Long Context Processing

Yijiong Yua, Jiale Liub,d, Qingyun Wub,d, Huazheng Wanga,d, and Ji Peic aOregon State University, {yuyiji, huazheng.wang}@oregonstate.edu bPenn State University, {jiale.liu, qingyun.wu}@psu.edu cDeepSolution, research@deepsolution.chat dAG2ai, Inc., jiale, qingyun, huazheng@ag2.ai

### Abstract

The quadratic complexity of self attention in Transformer based LLMs renders long context inference prohibitively expensive. While Sliding Window Attention (SWA), the simplest sparse attention pattern, offers a linear complexity alternative, it suffers from catastrophic long context performance collapse, which stems from two fundamental factors: the training inference mismatch when naively applying SWA to models pretrained with Full Attention (FA), and the inherent structural inability to access distant information when applying SWA to every module at all times. To address these dual challenges, we propose Sliding Window Attention Adaptation (SWAA), a plug and play toolkit of recipes that adapts FA models to SWA without costly pretraining. SWAA systematically combines four core strategies to tackle these distinct issues: (1) Full Attention (FA) Decode and (2) Interleaving FA and SWA layers, which mitigate structural defects by selectively allowing access to distant information; alongside (3) preserving “sink” tokens and (4) lightweight fine tuning, which mitigate the training inference mismatch. Our experiments reveal that while isolated strategies are insufficient, specific synergistic combinations effectively recover long context performance. Despite varying computational overheads, our performance efficiency trade off analysis identifies optimal SWAA configurations for diverse scenarios, achieving 30% to 100% speedups for long context inference with acceptable quality retention. Our code, data and model weights are available at github.

### 1 Introduction

Transformer based Large Language Models (LLMs) (Vaswani et al., 2017) demonstrate remarkable capabilities, but their self attention scales quadratically with the input sequence length, making long context processing inefficient. Sliding Window Attention (SWA), the most straightforward and widely adopted sparse attention pattern, restricts each token’s attention to a fixed size local window. This reduces computational complexity to linearity, along with several other benefits (see Appendix A).

Despite its simplicity and ease of use, it causes catastrophic long context performance collapse due to two factors: the training inference mismatch, specifically the model’s parameters cannot adapt to the drift of attention distribution, if directly applying SWA to models pretrained with Full Attention (FA); and the inherent structural difficulty of accessing the information of distant tokens (Xiao, 2025), if SWA is applied to every module at all times, which structurally cuts off all direct information pathways to tokens beyond the local window.

Although various methods that use SWA for efficiency have attempted to address these dual challenges, current solutions struggle to balance flexibility and quality. For instance, training-

free methods like streaming attention (Xiao et al., 2024) offer high flexibility, enabling the rapid, plug and play application of SWA on arbitrary FA pretrained models by retaining “sink tokens”. However, they struggle to guarantee long-context generation quality because the structural inaccessibility to distant information still exists. Conversely, models like Gemma2 (Team, 2024a) successfully mitigate this structural limitation by interleaving FA and SWA layers. Yet, this approach necessitates pretraining from scratch with this rigidly defined architecture, which is prohibitively costly, lacks flexibility, and restricts users’ choices of base models.

This dilemma provides our core motivation: Is it possible to leverage the efficiency of SWA to accelerate arbitrary LLMs, while simultaneously resolving these dual challenges at a low cost to maintain acceptable generation quality?

We answer Yes to this question by proposing Sliding Window Attention Adaptation (SWAA), a plug and play toolkit of recipes for adapting FA pretrained models to SWA, which requires neither costly pretraining nor new modules beyond the standard Transformer architecture. Specifically, SWAA systematically combines four practical and composable strategies to tackle these distinct issues.

To mitigate SWA’s structural inability to access distant information, we employ:

- 1. Full Attention (FA) Decode: applying SWA selectively in time (only during the prefilling stage) while switching back to full attention for decoding, allowing the model to retrieve distant context when generating text. This strategy exhibits significant synergy with the use of Chain of Thought (CoT).
- 2. Interleaving FA/SWA layers: applying SWA selectively in space by mixing full attention and SWA layers (e.g., applying SWA to half of the layers), ensuring distant information can still propagate through those FA layers.

To mitigate the training inference mismatch caused by naively introducing SWA, we utilize:

- 3. Keep First k Tokens: explicitly preserving attention to the first k “sink” tokens, to keep the attention distribution stable.
- 4. Fine tuning with SWA: lightweight, SWA aware supervised fine tuning on long context data.

While FA Decode is a novel method we introduce, the other strategies have been proven individually effective in various contexts (Xiao et al., 2024; Team, 2024a; Zhang et al., 2024). However, how combine them for better, cheaper SWA adaptation remains unexplored.

In our experiments, we evaluate SWAA on Qwen3 (Team, 2025b) and Llama3.1 (Team, 2024b) across several long context benchmarks. First, we find that although each strategy is helpful, no single ingredient suffices to make SWA competitive with full attention in answer quality. Second, we show that specific synergistic combinations of these methods can effectively recover long context performance. Third, despite the varying degrees of computational overhead introduced by these methods, our analysis of performance-efficiency trade-offs identifies recommended SWAA configurations for diverse scenarios. For instance, SWAA can achieve nearly 100% speedup with 90% accuracy retention in efficiency first scenarios, or a 30% speedup with nearly 100% accuracy retention in quality first scenarios, making SWAA a flexible toolkit rather than a fixed, rigid strategy.

Our key contributions are:

- • We perform the first systematic study on adapting FA pretrained LLMs to SWA with multiple methods, specifically addressing both the training inference mismatch and the structural defects of SWA, providing a foundation for future research.
- • We propose SWAA, a flexible toolkit of practical SWA adaptation recipes that offer a robust performance efficiency balance for various use cases, accelerating LLM inference from the bottom level.
- • We implement our methods with Flash-Attention (Dao, 2024) and vLLM (Kwon et al., 2023), making SWAA plug and play and user friendly for practical deployment.

### 2 Related Works

The O(N2) complexity of self attention in Transformers (Vaswani et al., 2017) has spurred a wide field of research about more efficient language model architectures. Among the two most popular technological routes are sparse attention and linear attention.

#### 2.1 Sparse Attention

Our work falls in this category. Sliding Window Attention (SWA) represents the most basic form of local sparse attention, yet its performance is inherently limited. Therefore, model architectures such as Longformer (Beltagy et al., 2020), BigBird (Zaheer et al., 2020), and RATTENTION (Wang et al., 2025) combine local SWA on most tokens with special global attention on specific tokens to create a more powerful, albeit still sparse, pattern. Popular LLMs like Gemma2 (Team, 2024a) adopt SWA in half of their layers to balance the efficiency of SWA and performance of FA. Sliding Window Attention Training (SWAT) (Fu et al., 2025b) introduces architectural changes, such as sigmoid activation and balanced position embeddings, to stabilize SWA performance. More advanced methods like Native sparse attention (Yuan et al., 2025), although achieving excellent quality, involve more complicated implementation and optimization due to semantic aware attention operations (e.g., selecting the most important tokens based on attention weights).

Nevertheless, nearly all aforementioned methods require pretraining with a specific sparse pattern, which is costly and fails to leverage the advantages of existing pretrained models. Some works explore transforming a full attention trained models to sparse attention through post training or fine tuning. Deepseek sparse attention (DeepSeek-AI, 2025) introduces a lightning indexer and a fine grained token selection mechanism to sparsely process tokens in long context, based on full attention trained Deepseek-V3.1, but still requires nearly 1T tokens for continued pre training. LightTransfer (Zhang et al., 2024) attempts to adapt existing models to SWA without pretraining, which has the same motivation as ours. But it may generalize poorly across model families (see Appendix F).

#### 2.2 Linear Attention

An alternative approach involves reformulating the attention mechanism entirely to achieve linear, O(N), complexity. This includes methods such as RNN like linear attention transformers (Katharopoulos et al., 2020; Peng et al., 2023; Sun et al., 2023) and structured state space models (SSMs) like Mamba (Gu & Dao, 2023). Many works such as Jamba and Nemotron-Flash (Lieber et al., 2024; Linsong Chu et al., 2024; Team et al., 2025; Fu et al., 2025a) interleave linear attention layers with traditional attention layers to create hybrid model structures. While promising, they represent a fundamental architectural departure from the standard Transformer, and thus must be trained from scratch. Meanwhile, their performance is generally weaker than traditional Transformer based LLMs, and may harm the math and reasoning ability.

### 3 Method

We incorporate four core adaptation methods into our SWAA framework to address the structural limitations and the training inference mismatch. Their specific operations and motivations are introduced below.

- 3.1 Mitigating Structural Limitations

- 3.1.1 Full Attention (FA) Decode (Temporal Selection)

This is a simple yet novel operation, which applies SWA only to the prefilling stage. During the decoding (auto regressive generation) stage, each token reverts to full attention, allowing the model to access all previous tokens across the entire long context. The resulting attention mask is depicted in Figure 1a.

[Figure 1]

[Figure 2]

(a) FA Decode (b) Keep First

- Figure 1: (a) Attention mask for FA Decode. SWA is used for prompt tokens (prefill), and full attention is used for generated tokens (decode). (b) Attention mask for SWA combined with Keep First k Tokens.

This approach is inspired by human reading comprehension: humans typically scan a passage casually (prefilling) before thinking deeply to formulate an answer (decoding) for a specific problem. We term this strategy “reading casually, thinking carefully.” In our design, the SWA constrained prefilling stage corresponds to casual reading, while the full attention decoding stage enables careful thinking and unrestricted information retrieval.

Interestingly, this analogy implies that Chain of Thought (CoT) may naturally complements FA Decode, because enforcing an explicit “thinking” process naturally extends the generation length, thus significantly increases the chances and temporal pathways for the model to retrieve or reconstruct distant information that was structurally blocked during the prefilling stage.

#### 3.1.2 Interleaving FA/SWA Layers (Spatial Selection)

This method retains full attention on a subset of layers while applying SWA to the remainder, providing a spatial mechanism to propagate distant information through the network’s depth while balancing the efficiency of pure SWA. A common strategy involves designating one in every n layers to use full attention. For example, Gemma-2 (Team, 2024a) uses SWA only for layers [1, 3, 5, ...], and Gemma-3 (Team, 2025a) uses SWA only for layers [5, 11, 17, ...].

However, for an FA pretrained model, layers may inherently have distinct behaviors and functions, indicating this method may be sensitive to layer selection strategy. While methods like LightTransfer (Zhang et al., 2024) have explored fine-grained layer selection, we find it is not consistently superior in practice when adapting existing FA models (see Appendix F). Therefore, although the strategy better than fixed-interval selection exists in theory, there is currently no good method to ensure finding it. So, for simplicity and generalizability, we still adopt fixed-interval layer selections in our experiments, such as [0, 2, 4, ...] or [1, 3, 5, ...].

- 3.2 Mitigating Training Inference Mismatch

- 3.2.1 Keep First k Tokens

Streaming attention (Xiao et al., 2024) demonstrates that models pretrained with full attention allocate a disproportionate amount of attention to the initial tokens (the “attention sink”). Naively applying SWA cuts off access to these sinks, causing severe mismatch. By permanently retaining attention to these initial k tokens while using SWA, the stability of the attention distribution and the model’s output is successfully maintained. As shown in Figure 1b, any subsequent token can attend to its local window and the initial k tokens.

Crucially, we extend prior approaches. While streaming attention operates only at the KV cache level (retaining the cache of sink tokens during decoding without accelerating prefilling), we directly customize the Flash-Attention-2 (Dao, 2024) kernel to implement this

specific attention mask, so as to further accelerate the prefilling stage via SWA and eliminate the need for cumbersome KV cache modifications.

#### 3.2.2 Fine tuning

Fine tuning is a highly direct approach to mitigating the training inference mismatch by updating the model’s weights to adapt to sparse attention patterns. Crucially, fine tuning in our framework does not strictly imply training with naive SWA; rather, it can selectively incorporate any combination of the aforementioned methods. Integrating structural modifications, such as FA Decode, Interleaving Layers, or Keep First k Tokens during the fine tuning phase is relatively straightforward.

However, enabling CoT (i.e., fine tuning a “thinking” model) presents a unique data challenge. Most available long context datasets provide only brief ground truth answers, which can be used to fine tune a standard instruct model, but is not applicable to a thinking model since it would severely degrade its reasoning style and capabilities. Since our goal is to restore the model’s original capabilities under SWA configurations rather than teach it new ones, we adopt a self distillation approach (Yang et al., 2024) when fine tuning thinking models. Specifically, we utilize the original full attention thinking model to generate new answers for the dataset’s questions, thereby perfectly preserving the CoT style if it is a thinking model. These generated answers are then filtered for correctness using GPT-5Mini (OpenAI, 2025) to construct our training dataset. For each question, we sample 4 answers with a temperature of 1.0 to enhance diversity, which we empirically find yields slightly better adaptation than single answer generation.

### 4 Experiment

Our experiments are structured to first comprehensively analyze the accuracy of a wide variety of adaptation recipes, covering almost all possibly effective combinations of the components of SWAA, followed by a evaluation of their efficiency, rather than simply evaluating a single configuration where all proposed methods are activated simultaneously. This granular approach is driven by two primary considerations:

First, each structural method introduces varying degrees of computational overhead to LLM inference (as discussed in Appendix A); indiscriminately adopting all of them may get suboptimal efficiency. Instead, we should evaluate multiple configurations to identify the optimal cost-effectiveness.

Second, a systematic ablation by isolating and combining different elements allows us to rigorously verify the necessity and complex dynamics of every component, ultimately providing more comprehensive and useful insights for future research.

- 4.1 Experiment Setup

- 4.1.1 Models

Our primary experiments use Qwen3-4B-Thinking and Qwen3-4B-Instruct (Team, 2025b). The Thinking variant enforces chain of thought (CoT) style reasoning, whereas the Instruct variant usually just answers briefly, so we can clearly see the impact of CoT on overcoming SWA’s structural defects. To ensure generality, we additionally evaluate Qwen3-30B-A3BThinking, Qwen3-30B-A3B-Instruct (Team, 2025b), and Llama3.1-8B-Instruct (Touvron et al., 2023), as shown in Appendix C. All models are served with vLLM in float16 precision using a batch size of 64, with our customized Flash-attention-2 kernel to support Keep First and FA Decoding. We use greedy decoding (temperature = 0) for all evaluations. In preliminary experiments, we observed that vLLM yields slightly lower (about 1% to 5%) scores than HuggingFace Transformers due to precision related discrepancies.

#### 4.1.2 Evaluation Dataset

SWA is identical to full attention when the context length is within the window size. Even if the model is fine tuned, we can simply disable the LoRA adapters for short prompts to get the same response as the original model. Therefore, our evaluations focus on long context benchmarks, as re-evaluating models on standard short context benchmarks (e.g., MMLU (Hendrycks et al., 2021), GPQA (Rein et al., 2023)) is completely unnecessary.

We select LongMemEval (Wu et al., 2024), a benchmark consisting of various types of long context QA tasks with moderate difficulty, as our main benchmark, although it is originally designed for agent memory system evaluation. Its context length is controllable by selecting a specific number of chat sessions to concatenate as the context from a pool of hundreds of sessions (a session contains the chat history between user and assistant within a day). To create a moderately difficult and discriminative evaluation, we construct LongMemEval 24k by sampling 10 sessions, resulting in 500 samples ranging from 16k to 32k with an average context length of 24k.

We also experiment on LongBench-V2 (Bai et al., 2024b) and Ruler (Hsieh et al., 2024), which are most commonly used long context benchmarks, for additional validation of generalizability. For LongBench V2, we retain only the samples whose context length is under 128k due to GPU memory limitations; thus, 311 of 500 samples are kept. For Ruler, we choose the Multi-Query task (the most challenging needle-in-a-haystack task), which contains 500 samples, and control the context length to 128k (counted by Qwen3 tokenizer). To judge the correctness for accuracy, we use LLM as judge with GPT-5-Mini (OpenAI, 2025) for LongMemEval and exact match for LongBench-V2 and Ruler. However, we find other LongBench-V2 is too difficult for 4B level models, and Ruler’s task types are not diverse enough (see details in Appendix B), so our analysis will more focus on the clearer results of LongMemEval.

#### 4.1.3 Training Details

For the fine tuning dataset, we initially considered LongAlign (Bai et al., 2024a), a widely used long context fine tuning dataset for adapt a regular length model to long context tasks. However, since its sample count (∼10,000) is insufficient, we incorporate an additional 6,000 samples from Fusang-v1-long (Pan, 2024), a more comprehensive corpus of over 40,000 long context samples that includes LongAlign as a subset. We perform SWA aware fine tuning using LoRA (Hu et al., 2022). Unless otherwise noted, we use rank r = 16 and α = 128, and apply LoRA only to the query, key, and value projection modules. We adopt this parameter efficient setting because full parameter fine tuning often leads to overfitting and degradation of the model’s original capabilities in our preliminary experiments. We use a learning rate of 1e-4 with a cosine decay schedule. Models are fine tuned for a single epoch on the sampled long context dataset since we observe no meaningful gains from additional epochs (see Appendix E). Training of each SWAA configuration takes approximately 12 hours on an 8*H20 GPU server for Qwen3-4B and 30 hours for Qwen3-30B-A3B.

- 4.2 Experiment Results

- 4.2.1 Accuracy Analysis

A macroscopic analysis of our experimental results of accuracy (Table 1) reveals a fundamental principle: the catastrophic degradation caused by SWA cannot be resolved by addressing the dual bottlenecks, training inference distribution drift and structural severing of distant semantic flow, in isolation. Applying any single method, such as Keep First, FA Decode, Interleaving Layers, or naive SFT, yields negligible accuracy improvements over pure SWA (Table 1, rows 3 to 5 and 25). Conversely, when we pair methods targeting distinct challenges (e.g., Fine tuning and FA Decode, Fine tuning and Interleaving layers, or Keep First and FA Decode), we observe significant accuracy recoveries.

FA Decode and Temporal Retrieval Synergy. Applying SWA strictly during prefilling while reverting to FA during decoding provides a temporal selection mechanism that temporarily reopens direct pathways to distant tokens. We find this structural mitigation exhibits a

- Table 1: The accuracy of Qwen3-4B on LongMemEval, LongBench V2, and Ruler with different SWAA recipes.

Window

Keep FA LongMem LB V2 Ruler

No. SFT

FA layers

size first decode Think Non T Think Non T Think Non T

- 0 False Full / / / 73.0 62.0 34.6 35.2 85.6 92.8

- 1 False 2k [] 0 False 3.2 11.0 9.4 25.8 0.0 0.0

- 2 False 8k [] 0 False 13.2 19.8 15.1 22.1 0.0 0.0

- 3 False 2k [] 10 False 16.0 15.6 7.7 25.8 0.0 0.0

- 4 False 2k [] 0 True 11.8 14.2 26.2 25.2 0.2 0.0

- 5 False 2k [1, 3, 5, ...] 0 False 13.4 18.4 12.1 23.5 0.0 0.0

- 6 False 8k [] 0 True 26.2 25.0 22.8 25.5 0.0 0.0

- 7 False 2k [] 10 True 38.2 20.6 25.8 25.2 1.0 0.0

- 8 False 2k [] 100 True 50.0 17.8 24.2 26.5 3.2 0.4

- 9 False 2k [] 1000 True 50.0 20.2 23.8 25.2 3.2 0.2

- 10 False 2k [0, 2, 4, ...] 10 False 17.0 14.8 19.8 29.2 0.0 0.0

- 11 False 2k [0, 2, 4, ...] 0 True 32.2 26.0 23.8 29.9 1.0 2.0

- 12 False 2k [1, 3, 5, ...] 10 False 25.8 36.4 21.1 29.9 0.0 0.0

- 13 False 2k [1, 3, 5, ...] 0 True 59.2 34.8 28.5 26.8 22.8 32.8

- 14 False 4k [] 10 True 38.0 24.4 27.9 27.5 1.6 2.2

- 15 False 8k [] 10 True 49.2 35.2 36.7 30.2 3.2 6.2

- 16 False 2k [0, 2, 4, ...] 10 True 36.0 17.2 30.5 30.2 10.0 3.0

- 17 False 2k [1, 3, 5, ...] 10 True 65.0 53.6 35.4 28.6 59.2 75.8

- 18 False 2k [1, 3, 5, ...] 100 True 68.8 50.6 35.6 31.8 58.0 78.8

- 19 False 2k [1, 5, 9, ...] 10 True 53.2 31.4 29.6 26.4 9.8 26.4

- 20 False 2k [1, 9, 17, ...] 10 True 36.4 18.8 28.6 25.4 0.6 0.2

- 21 False 2k [3, 7, 11, ...] 10 True 54.2 34.6 27.7 26.0 1.6 4.2

- 22 False 4k [1, 3, 5, ...] 100 True 73.0 54.2 33.4 31.2 65.4 83.6

- 23 False 8k [1, 3, 5, ...] 100 True 71.6 56.6 35.4 31.5 67.2 86.0

- 24 True Full / / / 74.6 63.4 37.9 34.9 88.2 90.6

- 25 True 2k [] 0 False 18.8 23.8 7.4 30.9 0.0 0.0

- 26 True 2k [] 100 False 15.6 24.0 6.0 30.2 0.0 0.0

- 27 True 2k [] 0 True 57.9 42.0 29.2 30.2 0.4 0.8

- 28 True 2k [1, 3, 5, ...] 0 False 63.6 54.6 29.5 31.9 59.0 62.2

- 29 True 2k [] 10 True 59.0 41.4 28.9 29.2 9.2 14.4

- 30 True 2k [] 100 True 62.2 42.6 29.2 30.5 13.4 20.4

- 31 True 2k [1, 3, 5, ...] 0 True 73.2 58.8 38.3 34.6 74.0 85.8

- 32 True 2k [0, 2, 4, ...] 0 True 66.0 \ 31.5 \ 8.0 \

- 33 True 2k [1, 5, 9, ...] 0 True 68.8 47.0 32.0 32.2 7.6 28.0

- 34 True 2k [1, 3, 5, ...] 100 True 73.2 61.4 37.2 33.9 72.8 89.2

profound synergy with the use of CoT, i.e. thinking. Because CoT naturally extends the auto regressive generation phase, it provides the model with more sequential steps to actively retrieve and integrate the broad context that was structurally under processed during the prefilling stage. This temporal retrieval synergy is validated by the superior performance of the thinking model over the standard instruct variant under FA Decode configurations (rows 13 and 18).

Interleaving FA and SWA Layers: Spatial Information Highways. While FA Decode addresses temporal bottlenecks, interleaving FA layers establishes vital spatial pathways, allowing distant information to propagate through the network’s depth. This is a highly potent structural mitigation, but zero shot adaptation is sensitive to layer selection. For instance, configuring odd numbered layers with FA vastly outperforms even numbered ones for Qwen3-4B (row 13 vs. row 11). This asymmetry suggests that specific layers in

pretrained LLMs inherently function as global routing hubs, and misaligning SWA with these hubs disrupts the semantic flow.

Synergy of Structural Methods. Interestingly, combining two purely structural interventions, FA Decode and Interleaving Layers, also delivers substantial gains (row 13). This occurs because structural improvements selectively enable SWA, naturally nudging the overall attention pattern closer to its original full attention distribution. Consequently, this implicitly alleviates the training and inference mismatch alongside resolving structural deficits.

Keep First k Tokens: Training free Attention stabilizer. Models pretrained with Full Attention (FA) develop a strong reliance on initial “sink” tokens. Prior to fine tuning, naively applying SWA cuts off access to these sinks, causing severe attention distribution drift. Preserving attention to these first k tokens acts as a crucial distributional anchor, preventing complete generation collapse (rows 7 and 9). However, its effect is obviously weaker than, and will be covered by SFT, which intrinsically reshape the attention weights to not relying heavily on these sinks, rendering the Keep First strategy largely optional (row 27 vs. 29).

Fine Tuning: Aligning Model Weights with Sparse Attention Patterns. While training free methods provide partial recovery, SFT acts as the ultimate unifier, parametrically aligning the model’s internal representations with the modified sparse attention structures. Combining structural methods (FA Decode or Interleaving Layers) with lightweight fine tuning virtually eliminates the performance gap, restoring accuracy to perfectly match the full attention baseline (rows 27, 28, and 31).

The above findings are mainly based on the results of LongMemEval and LongBench V2, which are more general. While for Ruler, as it involves only the needle in a haystack task, it exhibits a more extreme pattern: the accuracy can hardly be improved unless we enable Interleaving Layers, coupled with at least one another method (row 13 and 28). This indicates the retrieval behavior may be concentrated in certain specific layers, therefore these layers must retain full attention.

70

60

Accuracy(%)

50

40

30

20

: w/ SFT | : w/o SFT

0.5 1.0 1.5 2.0 2.5 3.0 3.5

Time (s)

Color Window FA Layers FA Decode

Full 0 False 2k 0 False 2k 0 True 2k 1/4 True 2k 1/2 False 2k 1/2 True

(a) Qwen3-4B-Thinking

60

50

Accuracy(%)

40

30

20

: w/ SFT | : w/o SFT

0.5 1.0 1.5 2.0 2.5 3.0 3.5

Time (s)

Color Window FA Layers FA Decode

Full 0 False 2k 0 False 2k 0 True 2k 1/4 True 2k 1/2 False 2k 1/2 True

(b) Qwen3-4B-Instruct

- Figure 2: Accuracy and inference time of each configuration of Qwen3-4B on LongMemEval

#### 4.2.2 Efficiency Analysis and Recommended Recipes

Although every method is helpful to accuracy recovery, Interleaving Layers and FA Decode introduce varying degrees of computation overhead in inference compared to pure SWA, indicating that the performance-efficiency trade-off of each SWAA configurations must also be evaluated, to find the most cost-effective one. So, we evaluate the average running

- Table 2: Recommended SWA adaptation recipes for different needs and scenarios: whether you want to train, use a thinking model, or prefer efficiency or quality. means optional.

Train Thinking Priority FA Decode Inter-leaving Keep First

No No Any No Yes Efficiency No Yes Quality

Yes Any Efficiency Yes Any Quality

time per request of Qwen3-4B-Thinking on a single H100 GPU using vLLM’s bench serve utility (Kwon et al., 2023) with random input data and a total of 100 requests. The prompt length and output length are set to 128k and 512 tokens, respectively, representing a typical long context QA setting. More specific metrics such as time to first token (TTFT), time per output token (TPOT) and total throughput are shown in Appendix D.

To visualize the performance efficiency trade off, Figure 2 plots each configuration’s accuracy on LongMemEval 24k (Wu et al., 2024) against its average running time, while detailed TTFT, TPOT, and throughput statistics for each configuration are provided in Appendix D. We draw a line between the full attention point and the naive SWA point as a baseline curve: configurations above this line offer a better accuracy latency balance intuitively. For configurations with nearly identical time costs, we display only the one with the highest accuracy. Since Keep First k has negligible impact on runtime (Appendix D), all plotted configurations fix k = 10.

From Figure 2, we observe that many configurations achieve a clearly better performance efficiency ratio than baselines. For example, after SFT, (1) enabling either FA decoding or Interleaving Layers can enhance the inference speed by almost 100%, while maintaining nearly 90% of the original accuracy; (2) while combining them can achieve nearly 100% original accuracy maintenance and over 30% acceleration; (3) By controlling the number of layers using FA, efficiency and quality can be achieved between that of (1) and (2). Furthermore, for the thinking model, more points lie above the baseline curve compared to the non thinking model, indicating that CoT generally has a positive effect on improving the performance efficiency ratio of SWAA.

Thus, we conclude that many SWAA configurations reach excellent performance efficiency trade offs, but there is no single metric to quantify such trade offs to decide the globally optimal one. We therefore summarize recommended SWA adaptation recipes tailored to various deployment scenarios in Table 2. We must note that specific parameters should be flexibly set to meet application specific requirements, without the need to follow our experimental parameters (e.g., a 2k window, k = 10). For example, users can increase the window size to 4k or k to 128 for higher accuracy and acceptable additional overhead.

### 5 Conclusion

In this work, we validate the feasibility of adapting full attention pretrained LLMs to Sliding Window Attention (SWA) for better efficiency, offering a cost effective alternative, SWAA, that avoids training sparse attention models from scratch. By systematically deconstructing the adaptation process, we identify that the catastrophic degradation observed in naive implementations can be effectively mitigated through synergistic combinations of auxiliary methods. Our extensive experiments across Qwen and Llama families demonstrate that while trade offs between computational overhead and model performance are inevitable, optimized configurations can achieve excellent performance efficiency balance to effectively accelerate LLM inference.

### References

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench: A bilingual, multitask benchmark for long context understanding, 2023. URL https: //arxiv.org/abs/2308.14508.

Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. Longalign: A recipe for long context alignment of large language models, 2024a. URL https://arxiv.org/abs/2401.18058.

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks, 2024b. URL https: //arxiv.org/abs/2412.15204.

Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer, 2020. URL https://arxiv.org/abs/2004.05150.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id= mZn2Xyh9Ec.

DeepSeek-AI. Deepseek-v3.2: Pushing the frontier of open large language models, 2025. URL https://arxiv.org/abs/2512.02556.

Yonggan Fu, Xin Dong, Shizhe Diao, Hanrong Ye, Wonmin Byeon, Yashaswi Karnati, Lucas Liebenwein, Maksim Khadkevich, Alexander Keller, Jan Kautz, et al. Nemotron-flash: Towards latency-optimal hybrid small language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a.

Zichuan Fu, Wentao Song, Yejing Wang, Xian Wu, Yefeng Zheng, Yingying Zhang, Derong Xu, Xuetao Wei, Tong Xu, and Xiangyu Zhao. Sliding window attention training for efficient large language models, 2025b. URL https://arxiv.org/abs/2502.18845.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces,

2023. URL https://arxiv.org/abs/2312.00752.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=d7KBjmI3GmQ.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models?, 2024. URL https://arxiv.org/abs/2404.06654.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 2529, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Fran¸cois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pp. 5156–5165. PMLR, 2020. URL http://proceedings.mlr.press/v119/katharopoulos20a.html.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention, 2023. URL https://arxiv.org/abs/2309. 06180.

Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, Omri Abend, Raz Alon, Tomer Asida, Amir Bergman, Roman Glozman, Michael Gokhman, Avashalom Manevich, Nir Ratner, Noam Rozen, Erez Shwartz, Mor Zusman, and Yoav Shoham. Jamba: A hybrid transformer-mamba language model, 2024. URL https://arxiv.org/abs/2403.19887.

Tri Dao Linsong Chu, Divya Kumari et al. Bamba: Inference-efficient hybrid mamba2 model,

2024. URL https://huggingface.co/blog/bamba. OpenAI. ChatGPT, 2025. URL https://chatgpt.com/. Wenbo Pan. Fusang-v1: A large curation of instruction-tuning datasets for better bilin-

gual and long-range llms, 2024. URL https://huggingface.co/datasets/wenbopan/ Fusang-v1.

Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski, Xingjian Du, Matteo Grella, Kranthi Gv, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartłomiej Koptyra, Hayden Lau, Jiaju Lin, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Johan Wind, Stanisław Wo´zniak, Zhenyuan Zhang, Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. RWKV: Reinventing RNNs for the transformer era. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 14048–14077, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.936. URL https://aclanthology.org/2023.findings-emnlp.936.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level googleproof q&a benchmark, 2023. URL https://arxiv.org/abs/2311.12022.

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models,

2023. URL https://arxiv.org/abs/2307.08621.

- Gemma Team. Gemma 2: Improving open language models at a practical size, 2024a. URL https://arxiv.org/abs/2408.00118.
- Gemma Team. Gemma 3 technical report, 2025a. URL https://arxiv.org/abs/2503.19786.

Ling Team, Bin Han, Caizhi Tang, Chen Liang, Donghao Zhang, Fan Yuan, Feng Zhu, Jie Gao, Jingyu Hu, Longfei Li, Meng Li, Mingyang Zhang, Peijie Jiang, Peng Jiao, Qian Zhao, Qingyuan Yang, Wenbo Shen, Xinxing Yang, Yalin Zhang, Yankun Ren, Yao Zhao, Yibo Cao, Yixuan Sun, Yue Zhang, Yuchen Fang, Zibin Lin, Zixuan Cheng, and Jun Zhou. Every attention matters: An efficient hybrid architecture for long-context reasoning, 2025. URL https://arxiv.org/abs/2510.19338.

Llama Team. The llama 3 herd of models, 2024b. URL https://arxiv.org/abs/2407.21783. Qwen3 Team. Qwen3 technical report, 2025b. URL https://arxiv.org/abs/2505.09388.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. URL https://arxiv.org/abs/2302.13971.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 5998–6008, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

Bailin Wang, Chang Lan, Chong Wang, and Ruoming Pang. Rattention: Towards the minimal sliding window size in local-global attention models, 2025. URL https://arxiv. org/abs/2506.15545.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval: Benchmarking chat assistants on long-term interactive memory, 2024. URL https://arxiv.org/abs/2410.10813.

Guangxuan Xiao. Why stacking sliding windows can’t see very far. https://guangxuanx. com/blog/stacking-swa.html, 2025.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=NG7sS51zVF.

Zhaorui Yang, Tianyu Pang, Haozhe Feng, Han Wang, Wei Chen, Minfeng Zhu, and Qian Liu. Self-distillation bridges distribution gap in language model fine-tuning, 2024. URL https://arxiv.org/abs/2402.13669.

Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 23078–23097, Vienna, Austria, 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/ v1/2025.acl-long.1126. URL https://aclanthology.org/2025.acl-long.1126/.

Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontan´˜on, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. Big bird: Transformers for longer sequences. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/ paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html.

Xuan Zhang, Fengzhuo Zhang, Cunxiao Du, Chao Du, Tianyu Pang, Wei Gao, and Min Lin. Lighttransfer: Your long-context llm is secretly a hybrid model with effortless adaptation,

2024. URL https://arxiv.org/abs/2410.13846.

### A SWA’s Benefits and Each Method’s Drawbacks

SWA reduces the computational complexity to O(N · W), where W is the window size. The benefits are threefold: (1) SWA reduces the computational load, (2) conserves GPU memory by limiting the required Key-Value (KV) cache, and (3) enhances KV cache reusability beyond traditional prefix caching, since a token’s state is independent of tokens outside its local window. However, there is no free lunch: to preserve long-context capability, we have to use the methods of SWAA, each of which has drawbacks, impairing the aforementioned benefits to varying degrees.

FA Decode presents two primary drawbacks: (1) the benefits apply only to prefilling, while decoding speed is not accelerated as it utilizes full attention, and (2) the GPU memory required for the KV cache is not reduced, as the KV cache for the full context must be retained for decoding. Using CoT further increases the decoding time, reducing the speedup ratio. In practice, however, many distributed LLM services have to recompute the KV cache of the entire chat history because storing and loading the KV cache complicates engineering systems, making prefilling occurs more frequently than expected, thereby amplifying the advantage of this method.

Interleaving Layers introduces the most significant overhead, as only a subset of layers benefits from the computational savings of SWA. Furthermore, the GPU memory required for the KV cache is not reduced for the full-attention layers. Additionally, this method negates the KV cache reusability advantage of SWA, as the existence of full-attention layers violates the independence of the KV cache beyond the local window.

Keep First introduces very minor computational overhead, but it complicates efficient KV cache reuse. Due to positional encoding, a token’s KV state depends on its position relative to the initial k tokens, hindering simple cache reuse across different requests. A position encoding separation or offsetting mechanism may be needed.

### B Some Problems of Other Long-context Benchmarks

We find existing long-context benchmarks problematic for our specific needs. For example:

- 1. LongBench (Bai et al., 2023) is classic and widely used, but its average context length (most are under 16k) is relatively short for modern models, i.e., it is already too easy. And its data source is too old, leading to a risk of test data leakage. So we choose not to use it.
- 2. Ruler (Hsieh et al., 2024) has controllable context length, but its tasks are almost all synthetic and most of them are needle-retrieval tasks, thus failing to reflect the model’s overall long-context capability in real-world scenarios.
- 3. LongBench-V2 (Bai et al., 2024b) is well-designed to necessitate deep understanding and reasoning over very long context. But it is too challenging for 4B-level models (e.g., Qwen3-4B-Thinking only gets 35% accuracy, which is too close to the random guessing baseline of 25%), making the improvement of different methods less distinguishable. Moreover, since it is in a multiple-choice question format, the results may not be sufficiently reliable because the model has a 25% chance of guessing the correct option. However, despite these drawbacks, they have been widely used in long-context model benchmarking. Thus we still elect to conduct our experiments on LongBench V2 (Bai et al., 2024b) and Ruler (Hsieh et al., 2024) to verify the generalizability of our conclusions, as shown in Appendix C.

### C Experiment Results of Other Models

We show the accuracy results of Qwen3-30B-A3B in Table 3, and the results of Llama3.1-8BInstruct in Table 4. Due to the time-intensive nature of training, we only test a selective set of configurations with fine-tuning. For Llama3.1-8B-Instruct on Ruler, the context length is controlled to 32k due to its weaker long-context capability. From the results, we observe that for any benchmark, the trend of accuracy change is mostly consistent with our previous conclusions, demonstrating their generalizability.

### D Inference Efficiency

The TTFT, TPOT and total throughput when using vLLM on a single H100 are shown in Table 5. Since inference speed is highly dependent on hardware, implementation details, and workload characteristics, these numbers should be interpreted as reference values. From the results, we can still conclude that:

- 1. Interleaving Layers and FA Decode significantly slow down the speed compared to pure SWA.
- 2. Keep First k Tokens has a negligible impact on efficiency.
- 3. Increasing the window size slightly increases inference time. For example, increasing from 2k to 4k decreases throughput by only 10%, but a 4k window generally achieves higher accuracy based on previous experiments. Therefore, in practice, a 4k window is a more common choice.

In theory, FA Decode should yield a decoding speed identical to that of full attention. Yet, in this table, we observe acceleration on TPOT. This is because vLLM-v1 typically mixes different requests’ prefilling and decoding tokens in one sequence to improve GPU utilization. Thus, the speeds of prefilling and decoding may affect each other. If processing only a single request, the situation differs. For example, when the generation length is set to 2000, we find decoding takes over 95% of the total time, rendering the acceleration of the prefilling stage negligible—i.e., SWA with FA Decode is almost unable to improve efficiency in such extreme cases.

### E Influence of Training Epochs

As shown in Table 6, training for more than 1 epoch yields no improvement. Therefore, we choose to train for only 1 epoch.

### F Results of LightTransfer

LightTransfer (Zhang et al., 2024) represents a promising attempt at SWA adaptation on full-attention models without pretraining. It proposes a layer selection method for SWA adaptation that calculates a ”lazy ratio,” represented by the ratio of attention from tokens at the end of the sequence (from a calibration dataset) to recent tokens versus global tokens. Layers with a higher ”lazy ratio” are selected to apply SWA, while the rest retain full attention. This method is intuitive and theoretically sound, but our experiments reveal some negative results.

Since the complete code of LightTransfer is not open-source, we reproduce this method using LongAlign (Bai et al., 2024a) as the calibration dataset for lazy layer detection, where the number of last tokens is set to 64, and the recent token window is set to 1024. From our experimental results shown in Table 7, we find that:

- 1. For Qwen3-4B, LightTransfer even yields a counterproductive effect; allowing lazy layers to use FA yields higher scores, while following the original method (letting non-lazy layers use FA) results in significantly lower scores.
- 2. For Qwen3-30B, it provides nearly no improvement over fixed-interval selection.
- 3. Only for Llama3.1-8B does LightTransfer show advantages.

Therefore, we conclude that LightTransfer does not yield stable performance across various models. Although fine-grained layer selection methods are theoretically superior, we believe they require further investigation before integration into our SWAA recipes.

- Table 3: The accuracy of Qwen3-30B-A3B-Thinking and Qwen3-30B-A3B-Instruct on LongMemEval, LongBench-V2, and Ruler

No. SFT

Window

FA layers

Keep FA LongMem LB-V2 Ruler

size first decode Think Non-T Think Non-T Think Non-T

- 0 False Full / / / 79.2 71.6 49.7 42.6 95.6 99.4

- 1 False 2k [] 0 False 0.0 0.4 0.0 0.0 0.0 0.0

- 2 False 8k [] 0 False 0.0 0.2 0.0 0.0 0.0 0.0

- 3 False 2k [] 10 False 0.0 2.8 9.1 32.2 0.0 0.0

- 4 False 2k [] 0 True 0.2 0.2 0.0 0.0 0.0 0.0

- 5 False 2k [0, 2, 4, ...] 0 False 21.0 28.4 20.1 25.8 1.4 17.0

- 6 False 2k [] 10 True 43.8 23.6 9.1 32.2 0.0 0.0

- 7 False 2k [] 100 True 58.6 22.2 10.4 28.2 2.4 3.0

- 8 False 2k [] 1k True 59.0 25.4 11.7 29.5 3.4 5.8

- 9 False 4k [] 10 True 49.8 26.6 26.8 30.9 3.0 1.2

- 10 False 2k [0, 2, 4, ...] 10 True 74.8 63.0 22.1 33.6 85.4 98.8

- 11 False 2k [0, 4, 8, ...] 10 True 48.8 23.8 12.4 29.5 1.6 1.8

- 12 False 2k [1, 3, 5, ...] 10 True 51.6 24.0 30.2 28.9 17.8 64.2

- 13 False 2k [2, 6, 10, ...] 10 True 64.8 44.2 21.1 35.6 35.8 50.2

- 14 False 4k [0, 2, 4, ...] 100 True 74.6 64.4 29.5 35.9 89.4 99.0

- 15 True Full / / / 79.6 72.0 43.6 42.0 97.6 99.4

- 16 True 2k [] 0 True 62.2 51.0 35.9 33.9 4.6 22.4

- 17 True 2k [] 100 True 65.6 50.8 36.6 32.9 10.8 57.8

- 18 True 2k [0, 2, 4, ...] 0 True 72.6 \ 41.3 \ 91.0 \

- 19 True 2k [0, 2, 4, ...] 100 True 77.8 68.0 48.0 37.9 91.4 99.6

- Table 4: The accuracy of Llama3.1-8B-Instruct on LongMemEval, LongBench-V2, and Ruler

Window

Keep FA LongMem LB-V2 Ruler

No. SFT

FA layers

size first decode Non-Think Non-Think Non-Think

- 0 False Full / / / 61.0 33.2 82.4

- 1 False 2k [] 0 False 0.6 0.0 0.0

- 2 False 8k [] 0 False 1.2 0.0 0.0

- 3 False 2k [] 10 False 1.8 28.9 0.0

- 4 False 2k [] 0 True 0.0 0.0 0.0

- 5 False 2k [0, 2, 4, ...] 0 False 3.0 0.0 0.0

- 6 False 2k [] 10 True 16.8 28.9 1.0

- 7 False 2k [] 100 True 20.0 30.2 10.0

- 8 False 2k [] 1k True 24.2 30.2 18.4

- 9 False 4k [] 10 True 23.8 27.5 2.4

- 10 False 2k [0, 2, 4, ...] 10 True 42.6 28.2 22.8

- 11 False 2k [0, 4, 8, ...] 10 True 17.8 32.6 4.4

- 12 False 2k [1, 3, 5, ...] 10 True 21.0 26.5 23.4

- 13 False 2k [2, 6, 10, ...] 10 True 24.4 26.8 1.0

- 14 False 4k [0, 2, 4, ...] 100 True 44.0 30.9 52.4

- Table 5: Efficiency metrics of different SWAA configurations on vLLM. ”FA layers = 1/4” means one fourth of total layers use full attention while the others use SWA.

window keep first FA decode FA layers TTFT (s) TPOT (s) Throughput (k tks/s)

Full 0 False None 1681.44 0.16 3.74

2k 0 False None 203.20 0.02 30.72 2k 100 False None 207.74 0.02 30.65 2k 0 False 1/2 938.00 0.09 6.70 2k 0 True None 963.39 0.11 6.39 2k 0 True 1/2 1321.39 0.14 4.72 2k 0 True 1/4 1141.66 0.12 5.43

4k 0 False None 233.07 0.02 27.03 4k 100 False None 237.87 0.02 26.74 4k 0 False 1/2 949.02 0.09 6.64 4k 0 True None 990.00 0.11 6.23 4k 0 True 1/2 1340.91 0.14 4.64 4k 0 True 1/4 1166.69 0.13 5.32

- Table 6: Results of different training epochs of Qwen3-4B-Thinking on LongMemEval SFT (epochs) window size FA layers keep first FA decode Acc

- 1 2k [] 0 True 58.0

- 2 2k [] 0 True 57.6

- 3 2k [] 0 True 56.0

- Table 7: Results of LightTransfer on LongMemEval. ”lazy” represents the half layers with higher lazy ratio, i.e. those which should apply SWA in theory. ”non-lazy” represents the other part, i.e. those which should keep full attention.

SFT window size FA layers keep first FA decode Acc think Acc non-think Model Group: Qwen3-4B

- False 2k [0, 2, 4, ...] 100 True 48.8 18.4

- False 2k [1, 3, 5, ...] 100 True 70.8 50.4 False 2k lazy 100 True 70.2 47.8 False 2k non-lazy 100 True 54.0 19.6 Model Group: Qwen3-30B-A3B

- False 2k [0, 2, 4, ...] 100 True 75.8 64.2

- False 2k [1, 3, 5, ...] 100 True 60.2 25.8 False 2k lazy 100 True 61.8 25.2 False 2k non-lazy 100 True 74.8 59.2 Model Group: Llama3.1-8B-Instruct

- False 2k [0, 2, 4, ...] 100 True \ 39.8

- False 2k [1, 3, 5, ...] 100 True \ 24.2 False 2k lazy 100 True \ 20.2 False 2k non-lazy 100 True \ 49.8

