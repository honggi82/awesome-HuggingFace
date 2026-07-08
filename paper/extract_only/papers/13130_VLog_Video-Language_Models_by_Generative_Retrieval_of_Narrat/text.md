# VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary

Kevin Qinghong Lin, Mike Zheng Shou Show Lab, National University of Singapore

arXiv:2503.09402v2[cs.CV]9Jun2025

## Abstract

Cut a potato Cut a potato ? [16] [8968, 257, 21219]

Human daily activities can be concisely narrated as sequences of routine events (e.g., turning off an alarm) in video streams, forming an event vocabulary. Motivated by this, we introduce VLog, a novel video understanding framework that define video narrations as vocabulary, going beyond the typical subword vocabularies in existing generative video-language models. Built on the lightweight language model GPT-2, VLog feature three key innovations: (i) A generative retrieval model, marrying language model’s complex reasoning capabilities with contrastive retrieval’s flexible upgrading over narration vocabulary. (ii) A hierarchical vocabulary derived from large-scale video narrations using our narration pair encoding algorithm, enabling efficient indexing of specific events (e.g., cutting a tomato) by identifying broader scenarios (e.g., kitchen) with expressive postfixes (e.g., by the left hand). (iii) A vocabulary update strategy leveraging generative models to extend the vocabulary for novel events encountered during inference. To validate our approach, we introduce VidCapEval, a development set requiring concise narrations with reasoning relationships (e.g., before and after). Experiments on EgoSchema, COIN, and HiREST further demonstrate the effectiveness of VLog, highlighting its ability to generate concise, contextually accurate, and efficient narrations, offering a novel perspective on video understanding. Codes are released at https://github.com/ showlab/VLog.

##### LLM’s Vocab

##### VLog’s Vocab

[257] a [416] by

[16] Cut a potato …

Video LMs

... [1364] left [8968] cut

- [35] by right hand
- [36] by le  hand … [45] using a knife

[21219] potato

[Figure 1]

Vocab. Size:

[Figure 2]

32K v.s. 1K Token-level:

[Figure 3]

[Figure 4]

Subword v.s. Narra on Vocab. Upgrade:

Fixed v.s. Flexible

Figure 1. Key Idea of VLog.. In contrast to existing videolanguage models that rely on token-by-token generation on language’s subword vocabulary, VLog introduces a novel generative retrieval method based on a narration vocabulary, achieving a significant speedup (10×) when processing long videos.

ularies [42] and transformers with pretrained weights, then adapt them through multi-modal instruction tuning [24,30]. As a result, these models can generate video-conditioned textual outputs via next-token prediction, as illustrated on the left side of Fig. 1.

Despite these advancements, original LLM designs are not inherently suited for video understanding. For example, LLM subword vocabularies are typically large (e.g., LLama-3’s 128K vocabulary size [34]) to capture broad linguistic information, but incomplete subwords (e.g., ‘happ’) often lack visual interpretability. Moreover, token-wise generation during inference introduces a bottleneck, limiting the model’s ability to process video in real time.

## 1. Introduction

.

“Life is a succession of moments.” – Corita Kent

Recent advances in Large Language Models (LLMs) [12, 34] have inspired several research into transferring textual knowledge to multi-modal domains [30, 61], leading to the development of Video Large Language Models (VideoLLM) [1, 32]. These models mainly leverage foundational elements of LLMs, such as subword vocab-

In practical applications, models are not always required to provide exhaustive details. Instead, we often prefer a video model that delivers concise, contextual responses in real time—such as an AR glasses assistant customized for personal needs i.e., prioritize task-specific efficiency over generalist models. This raises the question: How can we

: Corresponding Author.

tailor a video model to meet our specific requirements? To address this, we draw inspiration from how humans naturally organize experiences. When reflecting on our day, we often recall it as a series of narrative events—such as washing dishes or reading a book—that form the ‘narration vocabulary’ of our daily lives.

Motivated by this, we propose VLog, a novel efficient video understanding framework. Unlike generative VideoLLM that rely on subword vocabularies, VLog represents video narrations as minimal token units and builds the vocabulary. As shown on the right side of Fig.1, this narration vocabulary results significantly reduces decoding times.

To leverage the narration vocabulary in LLMs, it need to train the embedding layers to accommodate newly added vocabulary units. In contrast, CLIP [39]-based retrieval models can directly extract embeddings for novel vocabulary without additional training, enabling efficient similarity matching. However, retrieval models often lack the reasoning capabilities of generative models, failing to handle advanced queries like “What’s the next action?” with a reference video. To address this issue, VLog introduces a novel generative retrieval architecture that repurposes the language model’s reasoning capabilities with a retrieval token, embedding both visual and query information for reasoning-oriented retrieval.

To construct the narration vocabulary, we develop a Narrative Pair Encoding method applied to exisiting video narration datasets [15], generating prefix sets (e.g., Cut a potato) and postfix sets (e.g., by the left hand). Additionally, to enable efficient indexing across a large vocabulary, we organize the vocabulary hierarchically instead of Bruteforce search. This design allows rapid indexing of prefix vocabulary subset (e.g., ‘Cut a tomato’) by first identifying scenarios (e.g.,‘kitchen’) and then refining the search with the postfix, as illustrated in Fig.1.

Recognizing that the initial vocabulary may not cover novel, out-of-vocabulary events, we devised a vocabulary upgrade strategy using an agentic workflow. When low similarity scores are detected for vocabulary entries, they are treated as novel events and processed by generative models like LMM [22] to generate scene descriptions. LLM [47] is then prompted to extend the relevant vocabulary entries.

Overall, our contributions are three folds:

- • We introduce the first narration vocabulary for VideoLLMs, which is constructed by our Narration Pair Encoding method and a hierarchical strategy to enable efficient indexing, resulting xxx speedup;
- • We introduce a novel generative retrieval strategy, which supports flexible vocabulary upgrading without training embedding for new vocabulary.
- • We demonstrate the effectiveness and efficiency of VLog on the new development set Vidcab-Eval and

public datasets, including COIN, EgoSchema, and HiREST, highlighting its strengths in efficient and accurate video understanding.

## 2. Related Works

#### 2.1. Video Captioning

Video captioning provides a natural way to interpret and describe video content, with a key focus on understanding human activity [7, 9, 11]. This includes various downstream tasks, such as dense captioning [53] and step captioning [46]. Some video captioning datasets emphasize long, descriptive paragraphs that aim to capture every detail [6]. While in real-world applications, brief, contextual sentences are often sufficient to convey what happens in a video—essentially, video narration. First highlighted in [15] as an efficient alternative for documenting untrimmed video streams, video narration functions like a “minute”, capturing key events or changes with sparse yet informative content over time. With a large and diverse set of video narrations, as in [15, 16], we can reasonably assume that most common daily activities, such as “turning off an alarm upon waking”, can be reused across contexts, much like human experience. In our work, we treat these narrations as a vocabulary of human behavior, using them to interpret new, incoming video streams.

#### 2.2. Video-Language Models

Retrieval Models. Early vision-language models [4,27,35, 39, 50] primarily relied on alignment approaches, leveraging contrastive objectives for scalable performance. These methods excelled in retrieval and classification tasks within pre-defined labeled spaces, offering high efficiency by narration-level dot-product similarities calculation [25, 28, 38, 55]. But this fashion lacks the ability to model complex reasoning relationships between media inputs; for example, they fail to retrieve a video clip based on a query like, “What occurred after?” which conditioned on a reference video clip.

Generative Models. Recently, focus has shifted to generative LMMs, with several works [1,10,23,24,30,52,61] developing large multimodal models by projecting visual inputs into textual embeddings and aligning them with LLMs through visual instruction tuning. However, such straightforward adaptations are often unsuitable for video understanding, as large subword-level vocabularies in LLMs, while broadly inclusive, lack visual interpretability. Moreover, slow decoding during testing hinders real-time applications. Therefore, recent studies have focused on improving the efficiency of VideoLLM for (long) video modeling [8,26,43,44,51,57].

Generative Retrieval. By examining the strengths and limitations of retrieval and generative models, a natural ap-

[Figure 5]

[Figure 6]

Visual Token Query Token ❄Frozen 🔥Fine-tuned

Vocab. Token Retrieval Token

Wash the potato

[Figure 7]

[Figure 8]

Language Models 🔥 Language Models🔥

[Figure 9]

Merger🔥

What's the next ac on?

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Vis. Enc. ❄

Vis. Enc. ❄ Text Enc. ❄

Vis. Enc. ❄ What's the

Text Enc. ❄

next ac on?

[Figure 15]

[Figure 16]

[Figure 17]

Narra on Vocab Cut a potato Wash the potato

Narra on Vocab Cut a potato Wash the potato

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

... Walk around

... Walk around

###### b. Retrieval models

###### c. Genera ve Retrieval – VLog

###### a. Genera ve models

Narra on, Fast No Flexible, without training

Narra on, Fast Yes Flexible, without training

Vocab, Speed: Subword, Slow Complex Reason: Yes Vocab. Upgrade: Training required

Figure 2. Comparison between different Video-Language model architectures: (a) Generative Models: These model with complex reasoning but are slow, generating subword tokens one by one, Vocabulary upgrading requires retraining. (b) Retrieval Models: These enable fast narration vocabulary via similarity matching and support vocabulary upgrading without training, but lack reasoning, useful only for simple alignment tasks. (c) Generative Retrieval (VLog): This approach combines fast narration-vocabulary and flexible vocabulary upgrading with complex reasoning by using a retrieval token, merging the advantages of both methods.

.

vocabulary O, conditioned on the visual inputs V, query Q, and previously generated tokens x<n:

proach is to unify them, leveraging reasoning alongside efficient retrieval. In the realm of LLMs, efforts [2,20,45,48] have been made to improve text embeddings that capture complex relationships, showing promise for practical applications. However, these advancements are less explored in multimodal [17,19,31], particularly in video understanding.

Pr(xn = oi | x<n,V,Q) for oi ∈ O, (1)

where Pr is parameterized by language model weights using cross-entropy. Normally, the next token is predicted by maximum likelihood:

In this work, we pursue this generative retrieval direction by introducing a novel method that marrying lightweight GPT-2’s reasoning [40] with the contrastive vision-text model SigLIP [54]. Moreover, we departing from traditional subword vocabularies, VLog redefines the narration vocabulary, incorporating retrieval-based search for flexible vocabulary upgrading and interpretable reasoning.

Pr(xn = oi | x<n,V,Q). (2)

x˜n = arg max

i

This autoregressive process enables the model to capture complex dependencies between the visual inputs and the query, as demonstrated in Fig.2(a). However, for dense narrations over a long video, this approach incurs high inference costs due to token-by-token decoding [13,49,56]. Despite the efficiency gains brought by adopting the narrationlevel vocabulary, adding new vocabulary units to the embedding layer of LLMs typically requires training, making the approach less flexible.

## 3. VLog

Task definitions: Given a video V = {vi}, where each vi represents a frame, and a conditional query Q, our objective is to generate an accurate narration X of the video. To ensure consistency across all variants, we avoid specialized frame sampling strategies, focusing instead on the distinct strengths of generative and retrieval modeling.

b. Retrieval with Vocabularies. Inspired by how humans retrieve past experiences to interpret new coming events, we propose reframing the token generation process as a retrieval task with a predefined narration set O˜ = {cut a potato,··· ,walk round}, serving as a behavior vocabulary as depicted in Fig.2(b). We employ a visiontext contrastive model, SigLIP [54] as it similarity matching can support flexible vocabulary upgrading. The model maps video frames and vocabulary tokens into a shared embedding space, yielding vocabulary embeddings o˜i ← SigLIP(oi) and frame embeddings vj ← SigLIP(vj). To

#### 3.1. Architecture

a. Generative Models. In most generative approaches, i.e., large multimodal models, a subword vocabulary O = {oi} from language models is typically used, where L = |O| denotes the vocabulary size (each oi represents a subword, such as ‘the’). Given visual inputs and a query encoded into a token sequence, the language model estimates the probability of the next token xn being the i-th token oi over the

model temporal relationships among frames, we use an additional module— a 2-layers transformer layers- to produce a compact clip representation v˜ ← {vj} from the sequence of frame embeddings.

To sample prediction by leveraging the narration vocabulary, we define the probability between a narration vocabulary o˜i ∈ O˜ and the video clip v˜ based on their cosine similarity over their projected embeddings:

Pr(X = o˜i | V) = v˜To˜i for o˜i ∈ O, (3) the prediction is then determined by X˜ = arg maxo

v ˜To˜i

i

This approach offers two benefits: first, it enables narration-level retrieval via dot product similarity calculation (i.e., Eq. 3); second, the vocabulary is independent of LLM embedding weights, making it practical for upgrades when needed. However, despite leveraging the strengths of contrastive models, this method struggle to capture the complex relationships in queries Q, posing challenges for retrieving with a queries e.g., ‘What happened next.’

c. Generative Retrieval (Ours). To leverage the reasoning modeling of generative models while harnessing the efficiency of narration-vocabulary and the flexiblity of vocabulary upgrading, we propose a novel generative retrieval model. As shown in Fig.2(c), we introduce a retrieval token with embedding t to bridge the generative language model and retrieval model. The retrieval token is positioned as the last input in the language model sequence, allowing it to attend to both the front visual and query inputs. After passing through the language model, the output embedding ˜t is assumed to encode both visual and query information, enabling retrieval while preserving reasoning ability:

Pr(X = o˜i | V,Q) = ˜tTo˜i for o˜i ∈ O˜, (4) the prediction is then yield by X˜ = arg ma˜xo

˜tTo˜i .

i

Moreover, our model has the following considerations:

- 1. Retrieval tokens initialization: The retrieval token, appearing at the end of the sequence, can be initialized as a learnable token, an EOS token, or using meanpooled visual inputs to encode overall visual information—examined in our experiments.
- 2. Asymmetric structure: Unlike visual and query inputs, vocabulary embeddings are not projected by the language model and remain fixed after initial computation, reducing forward computation costs with large vocabularies.

In this way, our architecture effectively addresses both reasoning and efficiency issues. Training objectives. Our generative retrieval model is trained with a standard contrastive objective:

exp(˜tTi o˜i/τ) j∈B exp(˜tTi o˜j/τ)

1 |B| i∈B

, (5)

L =

log

Narra on Pairing Encoding

Narration

###### VLog’s Vocabulary

Cut a potato Cut a potato by left hand Cut a potato with a knife Cut a potato by right hand

Prefix: Cut a potato Postfix: by left hand

with a knife by right hand

- (a) Construct Vocabulary by NPE.
- (b) Hierarchical Indexing.

###### 3-th Postfix

###### 1-th Scene

###### 2-th Preﬁx

Kitchen Gardening Shopping

Cut a potato Wash a dish Wipe the table

by left hand with a knife by right hand

Figure 3. Illustration of our Vocabulary. Construction and Indexing. Upper side: Given the narrations, we process them using our NPE method, breaking down each narration into prefix and postfix parts. Lower side: For efficient indexing, we organize the vocabulary hierarchically, where first-level scenes help navigate subsets of prefix narrations. Next, we append the prefix and continue retrieving the postfix.

where B is the batch size and τ = 0.05 is the temperature. This objective updates the language model weights to encourage retrieval tokens ˜t derived from visual and textual inputs to align with the target vocabulary o˜i.

#### 3.2. Vocabulary Construction

The narration vocabulary is key to our model. We source it from existing video datasets [15], which contain extensive narrations across various domains. We then clean these narrations and remove duplicates (see details in Supp).

Narration Pair Encoding. We observe that many narrations are often subjective and inconsistently formatted, as shown in Fig. 3 upper side, they share common prefixes (e.g. “cut a potato”) but differ in their postfixes, which add context. In natural language processing, Byte Pair Encoding (BPE) [42] addresses this issue by tokenizing text corpora into subword units. However, in our setting, we aim to build a narration-level vocabulary where BPE is not directly applicable. To address this, we introduce Narration Pair Encoding (NPE): We treat each narration as a potential prefix and search for longer narrations that start with it, collecting their postfixes—the additional words beyond the prefix. This approach yields two sets—a prefix set of narrations with non-empty postfixes and a shared postfix set. We detailed NPE algorithm in our Supplement.

To enable model for prefix and postfix retrieval, we first use the retrieval token to retrieve the prefix, concatenate it with the visual and query sequence, then use the retrieval token to retrieve a postfix from the postfix vocabulary.

Hierarchical Indexing. After completing the NPE process, we obtain both prefix and postfix narration sets. However, the large scale of narrations (e.g., millions) poses a challenge for efficient retrieval, making brute-force search im-

practical. Considering that human activity recordings in videos often align with specific scenarios, such as “cut a potato” occurring in a “kitchen” scene, we develop a hierarchical indexing strategy by associate narrations to its videos belonged scenario.

[Figure 24]

[Figure 25]

Score is low

Clean the dishwasher (Score = 0.23)

###### LMM

[Figure 26]

LLaVA-OV-0.5B

Describe this scene brieﬂy.

Outdoor Barbecuing

### VLog

###### Agentic workflow for OOV

What might take place during {X}?

The full retrieval chain is displayed in the lower part of Fig.3. Given a video, the model first identifies the video scenario and then retrieves the associated prefix narrations subset. This is efficient by reducing the search space. Then we continue retrieval the postfix from the postfix vocabulary. This helps to improve the narration expression.

|Upgrade|New Vocab<br><br>Set up the grill Turn on the grill<br><br>... Remove from heat|
|---|---|
| | |

[Figure 27]

###### Initial Vocab

[Figure 28]

###### LLM

Cut a potato Clean the dishwasher

[Figure 29]

Qwen2.5-0.5B

... Walk around

Figure 5. Agentic workflow of VLog Vocabulary upgrade. When a low retrieval score is detected, the visual inputs are sent to the LMM (LLaVA-OV-0.5B [22]) to generate scene descriptions. These descriptions are then processed by the LLM (Qwen2.50.5B [47]), which expands and updates the existing vocabulary.

#### 3.3. Instruction-Tuning Data

Once we developed such a generative retrieval model, the next issue we faced was the lack of training data with complex reasoning relationships, as most video-text retrieval data are paired solely for alignment. Fortunately, untrimmed video streams offer a natural solution by inherently modeling temporal relationships between narrations.

#### 3.4. Vocabulary Upgrading

Despite the diversity and scale of initial vocabulary, novel Out-of-Vocabulary (OOV) events may still occur, requiring: (i) detection of novel events, and (ii) expansion of the vocabulary with new entries.

Detect Novel Events. Our model uses the dot product in Eq. 3 as a relevance metric between the query and candidate vocabulary, akin to logits as a confidence measure in LLMs. Empirically, we set a threshold of 0.4: if the top1 vocabulary match falls below this threshold for a given visual input, we classify the event as OOV.

Open tap water

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

… …

What’s the next ac on? What’s the current ac on? What’s the previous ac on?

Seeking Help from Generative Models. To handle out-ofvocabulary narrations, we leverage both LMMs and LLMs, which offer complementary strengths: LMMs generate brief, visually grounded captions but may hallucinate on fine-grained actions, while LLMs, with broad knowledge and strong instruction-following, generate diverse candidate narrations. We integrate these models in an agentic workflow. As illustrated in Fig.5, where LMMs first provide concise visual scene descriptions, and LLMs then use this context to infer potential events that might occur in the scene, which are parsed as new vocabulary narrations. In practice, we use two models comparable in size to our GPT-2: the LMM – LLaVA-OV-0.5B [22], and the LLM – Qwen2.50.5B [47] guided by an in-context template.

Figure 4. Create video-text pairs that requires complex reasoning from untrimmed videos based on their temporal relationship.

As illustrated in Fig. 4, we develop three types of training data representing ‘before’, ‘next’, and ‘current’ relationships. For ‘next’ as an example, given a video clip Vi with its reference narration Xi, we trim a preceding segment V<i and append a prompt such as “What’s the next action?” Based on these temporal relationship, we create an 200K instruction-tuning training data from [15], namely VidcabTrain (i.e., Video vocabulary). Moreover, as we lack of such retrieval evaluation consider complex reasoning, we apply the same strategy but create a generative retrieval development set with 4.6K samples named as Vidcab-Eval. Notably, we carefully curate the instruction training data and development set using our NPE method. For instance, in Vidcab-Eval, the model must accurately retrieve both the prefix and postfix to achieve a higher score. We detail how to construct them in Supp.

This workflow differs from existing RetrievalAugmented Generation [14], serving instead as a new Generative-Augmented Retrieval approach, where generative models actively expand the vocabulary for improved retrieval accuracy.

Moreover, we incorporate scenario information from videos, prompting the model to encode the entire sequence of a video with queries like “What’s the overall activity in this video?” and answers such as “Cooking.” This organization supports hierarchical vocabulary indexing, which we will discuss in the next section.

## 4. Experiments

In this section, we structure our experiments to answer the following questions:

- Q1: What key advantages does VLog offer?
- Q2: How to adapt VLog to different tasks (e.g., Reasoning

Models FT? Navie Retrieval Casual Retrieval Decode time Visual Enc. Post process Vocabulary CIDEr R@1 CIDEr R@1 sec

Generative SigLIP-L GPT2-M GPT2-32K ✓ 64.8 7.9 53.7 3.1 0.362 Retrieval SigLIP-L MeanPool Eval-4.6K ✗ 63.6 4.6 N/A N/A 0.001 Retrieval SigLIP-L Adapter Eval-4.6K ✓ 95.8 11.8 48.9 2.1 0.016 VLog SigLIP-L GPT2-M Eval-4.6K ✓ 96.9 12.4 87.3 5.0 0.018 (20×) VLog-prefix SigLIP-L GPT2-M Ego4D-0.8M ✓ 91.3 10.9 83.9 3.7 0.035 (10×) VLog-prefix&postfix SigLIP-L GPT2-M Ego4D-0.8M ✓ 94.9 11.9 86.9 4.8 0.053 (6×)

- Table 1. Key ablation studies on Vidcab-Eval for naive and casual retrieval. Under same conditions, VLog provides accurate narration with significant speed improvements.

. QA beyond retrieval)?

#### 4.2. Key Advantages by VLog

- Q3: How effective is the vocabulary upgrading strategy (address out-of-vocabulary problem)?
- Q4: What are the key design choices in VLog (e.g., NPE and Vocabulary indexing)?

In Tab. 1, we evaluate key variants in VLog-Eval, which aims to produce accurate narrations based on both visual and textual queries. This includes two tracks: a na¨ıve track (without query, i.e. normal video-text alignment) and a causal track (with query, e.g., before / after / current). To enable both generative and retrieval models to report scores, we introduce two metrics to assess narration quality: CIDEr (common used by generative models) and Recall@1 (common used by retrieval models) alongside generation speed per clip for a comprehensive evaluation.

#### 4.1. Datasets and Settings

Vidcab-Eval is constructed by Sec. 3.3, we selected Ego4D [15] to build our initial vocabulary and training data due to its large scale and diversity, with millions of manually curated narrations. We ensure no overlap with downstream tasks such as EgoSchema. To evaluate our proposed generative retrieval setting, we evaluate on Vidcab-Eval development set.

Baselines: To ensure a fair comparison, all variants use the same visual encoder, and both the generative and VLog models are based on the same GPT2 model trained on the same data. For the retrieval baseline, we provide both zero-shot and fine-tuned versions, using VLog-Eval4.6K as the restricted vocabulary, ensuring that the correct term is within the vocabulary. To adapt the retrieval model for casual retrieval, we pool the visual and textual query inputs to obtain a unified embedding. For VLog, we provide variants using either Eval-4.6K or our full constructed vocabulary, with or without postfix retrieval. We have the following observation:

COIN is an instructional video dataset comprising 11,827 videos across 180 tasks in 12 domains related to daily life. We evaluate our model on three common COIN benchmarks: step recognition, step forecasting, and task summarization. We use this benchmark to study VLog’s ability for fine-grained activity recognition.

EgoSchema is a long-range video question-answering benchmark with 5K multiple-choice pairs across 250 hours of video, covering a wide range of human activities. Unlike prior benchmarks, correctly answering a question here requires at least 100 seconds of video viewing—well beyond existing standards. We use this benchmark to study VLog’s high-level reasoning abilities.

- (i) Overall, the scores in the na¨ıve setting are higher than in the causal setting, indicating that causal retrieval is more challenging than standard alignment, as it requires a finegrained understanding of temporal relationships.
- (ii) In the na¨ıve setting, VLog achieves comparable CIDEr and Recall@1 scores to retrieval models. In a more challenging causal setting, VLog significantly outperforms both generative and retrieval models, the latter of which pools both visual and query inputs but fails to model relationships effectively. This demonstrates that our generative retrieval approach effectively leverages the language model’s causal modeling within a flexible retrieval framework.
- (iii) Using our constructed vocabulary (Ego4D-0.8M, where the ground truth may not be included), the prefix’s performance is slightly lower than the baseline (Eval-4.6K), as the full vocabulary may not contain exact matches from the development set. However, when equipped with our postfix, performance improves, demonstrating the high expressive-

HiREST is a new benchmark for hierarchical procedural information. It includes videos from novel domains that not appeared in Ego4D and COIN, with numerous and highquality step captions by human annotators. Thus, we use its step-captioning task to study the effectiveness of VLog’s vocabulary upgrading strategy.

Implementation Details. VLog builds on GPT2-medium [40] with SigLIP [54], extracting video clips at 2 frames per second. For long videos, we uniformly sample long video into multiple fix length clips (1s) and process them in a streaming fashion. We pre-extract the visual features and store the textual embedding of narration vocabulary, which storage occupies 3.9 GB in total. During training, we finetune the models fully.

Top-1 Acc Step Task Next

Method PT by? Time (s)

ClipBERT [21] COCO+VG – 30.8 65.4 TimeSformer [5] HT100M – 46.5 85.3 34.0 Paprika [60] HT100M – 51.0 85.8 43.2 DistantSup [29] HT100M – 54.1 90.0 39.4 VideoTF [37] HT100M – 56.5 91.0 42.4 ProcedureVRL [59] HT100M – 56.9 90.8 46.8 VideoTaskGraph [3] HT100M – 57.2 90.5 40.2 VideoLLM-online-7B [8] N/A – 59.8 92.1 48.1

GPT2-medium N/A 0.21 44.6 82.4 32.1 VLog N/A 0.01 56.1 93.0 46.0 VLog Ego4D 0.05 57.4 94.4 48.4

###### Table 2. Activity perception results on COIN benchmarks: step

recognition, task recognition, and next-step forecasting.

.

ness of our full narration vocabulary and postfix.

(iv) In terms of decoding speed, VLog is comparable to the fastest retrieval model, achieving a 20x speedup over generative models. Although indexing slows slightly with a large vocabulary (0.8M), it remains significantly faster (10x) than generative models, highlighting its efficiency advantage with the narration vocabulary.

These ablation studies demonstrate the great potential of VLog, a video narrator that provide concise narrations with flexible query conditions and fast decoding speed.

#### 4.3. Adapting VLog to Different Tasks

Beyond casual retrieval tasks, we next demonstrate how VLog can be adapted to other mainstream tasks: Finegrained action perception and High-level reasoning QA.

Fine-grained Action Perception: COIN. The COIN tasks consist of closed-set action categories, enabling our models to adapt seamlessly by substituting the vocabulary with COIN’s predefined category set. Additionally, different setting i.e. steps, tasks, or next actions are matched in VLog by using varied queries as conditions. We develop a generative GPT2 baseline for comparison, which directly outputs the ground-truth string. All methods are fine-tuned on the COIN dataset, with results provided both with and without Ego4D vocabulary pretraining.

As demonstrated in Tab. 2, VLog outperforms the GPT2 baseline by a significant margin as well as process speed per clip (20×). With a lightweight model size (124M), VLog achieves performance comparable to the state-ofthe-art LMM [8], built on LLama-2-7B. Additionally, we demonstrate that generative retrieval pretraining from Ego4D successfully transfers to the COIN dataset.

High-level Reasoning QA: EgoSchema. In Tab.3, we present evaluation results on the Egoschema MCQ task. Our focus is not on surpassing state-of-the-art methods—most baselines [13, 49] rely on closed-source APIs (e.g., GPT-4o) or large-scale pretraining. Instead, we aim to validate the transferability of VLog’s reasoning capabilities. To enable QA, we use VLog as a narrator to densely

caption each long video, creating an informative document that LLMs can reference to answer questions accurately. An accurate narration should assist the LLM to correctly identify the answer. We list highly relevant baselines, which has narrator and (open-source LLMs) answers. We also develop generative baselines and compare their accuracy and runtime per 180 sec video. LLoVi [56], which uses the same LLama3-8B models as the answerer and a GPT2 narrator [58] pretrained on [15], being a comparable baseline.

Our VLog demonstrates an improvement margin (+3.4% accuracy on the subset and +4.3% on the full set), indicating more accurate narrations. Additionally, it achieves faster processing times per long video compared to generative baselines.

#### 4.4. Vocabularies Upgrading for OOV

In Tab. 4, we study the vocabulary upgrading strategy to address OOV problem. To design the experiments, we use HiREST’s step captioning as benchmark, which includes several novel event (e.g., make a diet coke and mentos rocket) not seen in our vocabulary. We include various baseline models for comprehensive comparison, selecting LLaVA-OV-0.5B [22] – the LMM in our agentic framework, conditioned on LLM’s [47] vocabulary. For VLog models, we develop several variants with different vocabulary sizes and sources, including Ego4D, and oracle HiREST vocabularies, we compare produce narrations with groundtruth narrations by captioning metrics.

In the experiment, we first observe that LLaVA-OV0.5B does not achieve a high score with video inputs compared with image inputs, likely suffering hallucinations in small model size. When come to VLog, it is notable that the large Ego4D vocabulary underperforms compared to COIN, likely due to differences between egocentric and web-instruction videos. However, augmenting with 445 vocabulary items tailored for HiREST improves performance, validating this approach. Adding the oracle task name as a condition results in a slight performance boost, and using the oracle HiREST vocabulary achieves the best results. This demonstrates that vocabulary selection is crucial and that vocabulary upgrade strategy supports adaptation to un-

Methods Narrator Answerer Ego? Subset Full Time (s)

|MVU [41] OWL-ViT [36] LLaVA-13B ✗ LangRepo [18] LaViLa [58] Mixtral-8x7B ✓ LLoVi [56] LaViLa [58] LLama3-8B ✓|60.3 37.6 –<br><br>66.2 41.2 –<br><br>67.0 38.8 –<br><br><br>|
|---|---|
|Generative SigLIP&GPT-2 LLama3-8B ✓ Ours VLog LLama3-8B ✓<br><br>|66.5 37.4 48.2 70.4 43.1 2.3|

Table 3. Video multiple-choice question-answering on Egoschema. Each baseline consists of a narrator (to provide video information as reference) and an answerer (to respond to the question). We report accuracy on both the subset and fullset, along with processing time per 180-second video.

Models Vocab size METEOR CIDEr SPICE

Method Casual Retrieval COIN-Step COIN-Task

LLaVA-OV (Vid) Qwen2 (152K) 1.2 1.0 0.1 LLaVA-OV (Img) Qwen2 (152K) 2.3 4.2 2.5

EOS 4.3 51.3 93.1 Learnable 4.2 50.9 92.5 Pooling 4.7 54.0 94.0

VLog Ego4D (0.8M) 2.6 5.4 2.6 VLog COIN (778) 3.0 6.9 2.3 VLog COIN +Upgrade (1223) 4.2 12.6 3.0 VLog HiREST’s task+Upgrade 4.8 14.7 3.2 VLog HiREST 5.8 21.2 4.2

Table 5. Ablation of different retrieval tokens initialization.

We found that the pooling strategy outperforms the EOS method, especially on COIN-step. However, for COINtask, which requires longer input durations, the scores are closer. This suggests that for fine-grained action recognition, effective initialization is crucial.

Table 4. Key studies on Vocabulary upgrade strategies for the HiREST Step Captioning Task across different vocabularies.

seen novel narrations.

Can LMs knowledge help Generative Retrieval? In Tab.6, we examine whether the pretrained knowledge of LLMs can enhance generative retrieval performance. For this, we develop several variants: GPT2 models with and without pretrained weights and models of different sizes to assess the strengths of LM’s knowledge.

#### 4.5. Ablation of Key Design Chocies

Hierarchical Indexing. In Fig. 6, we study the hierarchical indexing using the Egoschema val. set. ‘BF’ denotes Brutal search over the full vocabulary (0.8M), while ‘Hier.’ represents our proposed hierarchical indexing method. Compared to a brutal search over a large space, our strategy is 15× faster, only averaging 2.3 seconds per video, and yields comparable accuracy.

Method Size Casual Retrieval COIN-Step COIN-Task

GPT2 124M 4.7 54.0 94.0 GPT2 (Random init.) 124M 3.8 51.5 91.0 GPT2-Medium 355M 4.6 55.4 94.8 GPT2-Large 774M 4.9 56.4 93.2

Accuracy (%) Time (s)

80

80

40

Retrieve by Scene

Random Sample

| |70.6| |70.4| |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

70

34.5

Table 6. Ablation of LM’s pretrained weights and sizes.

60

30

Accuracy

60

In general, pretrained and larger GPT2 models are beneficial, achieving higher scores on COIN-Step, which indicates that increased model size and richer textual knowledge enhance casual relationship for generative retrieval.

40

20

50

40

20

10

30

1 10 100 1000

2.3

0

0

Num of Refer Video for Vocab

BF Ours

BF Ours

#### 4.6. Visualization

Figure 6. Ablation of Vocabulary Indexing Strategy

Figure 7. Ablation of Reference Video Number

Below, we provide an example to illustrate how VLog operates in a full decoding process. Given a video clip, VLog begins by identifying the scenario—‘doing yardwork’—then proceeds to the corresponding prefix vocabulary, resulting in “move a lawn mower machine” and finally completes the postfix as ‘on grass.’ Additional examples, illustrations on handling OOV cases and failure cases are available in the Supp. quantitative analysis Sec.

Figure 8. Ablation studies of Vocabulary indexing on Egoschema QA. Left: our hierarchical indexing (by scene) is 15× faster than Brutal search over full vocabulary while keeping the accuracy. Right: Using the same number of reference video for constructing narration vocabulary, selected by scene yields more relevant vocabularies.

In Fig. 7, we compare VLog scene-based indexing with random indexing, adjusting the vocabulary size by varying the number of reference videos. We find that scene-based retrieval consistently yields a more reliable narration vocabulary and outperforms random sampling for the same vocabulary size. Accuracy improves with larger vocabulary sizes and additional reference videos, ensuring matched narrations appear more frequently.

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

What is the next action in the video?

3-th Postfix In the room On the grass … With the left hand With the right hand

1-th Scene Doing yardwork Gardening … Eating in a canteen Fixing something

2-th Prefix Pull a grass mower Move a lawn mower … Spread the silicone Place the small plank

Prediction: Move a lawn mower.

Prediction: Move a lawn mower on the grass

Definition of Retrieval Token. In Tab.5, we study how to define the memory token within VLog, which could be the following variants: (i) EOS, (ii) A learnable token, (iii) Pooling by all visual features.

Figure 9. Illustration of VLog’s full decoding process.

## 5. Conclusion and Limitations

[Figure 40]

[Figure 41]

Language Models🔥

Language Models🔥

We present VLog, a novel framework for video streaming with a narration vocabulary. Built on a lightweight GPT-2 model, VLog introduces an innovative generative retrieval approach, combining causal modeling with retrieval efficiency. Additionally, VLog incorporates hierarchical vocabulary indexing and a vocabulary update mechanism. Experiments across several datasets demonstrate that VLog enables concise, contextually accurate, and efficient narrations, highlighting its potential for real-time video processing. We acknowledge that VLog’s design is constrained by its predefined vocabulary. Future work will explore how to explore VLog to more diverse domains with a generalized query support.

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Vis. Enc. ❄ What's the

Text Enc. ❄

Vis. Enc. ❄ What's the

Text Enc. ❄

next ac on?

next ac on?

[Figure 46]

[Figure 47]

Preﬁx Set Cut a potato Wash the potato

Pos ix Set By Le  hand By right hand

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

... Walk around

... Using the knife

Figure 10. Illustration of VLog’s progressively decode prefix and postfix vocabulary respectively.

Algorithm 1 Narration Pairing Encoding Require: List of narrations N Ensure: Prefix list P, Postfix list S

Acknowledgements. This research is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No: AISG3-RP-2022030).

- 1: Step 1: Build Prefix Dictionary
- 2: Initialize prefix dictionary D ← empty
- 3: for all narration n ∈ N do
- 4: Split n into words [w1,w2,...,wk]
- 5: for i = 1 to k do
- 6: Prefix p ← w1 w2 ... wi
- 7: Add n to D[p]
- 8: end for
- 9: end for
- 10: Step 2: Extract Prefixes and Postfixes
- 11: Initialize prefix list P ← ∅, postfix list S ← ∅
- 12: for all narration n ∈ N do
- 13: if D[n] contains only n then
- 14: Add n to P
- 15: else
- 16: for all other narration s ∈ D[n] do
- 17: if s ̸= n then
- 18: Get suffix t ← Remove prefix n from s
- 19: Add t to S
- 20: end if
- 21: end for
- 22: end if
- 23: end for

## A. Appendix

#### A.1. VidCab construction

We begin by sourcing video clips from EgoClip [27], excluding any videos associated with downstream tasks such as Egoschema [33]. Next, we clean the narrations by removing special tags like ‘#C C’ and perform deduplication within each video, resulting in approximately 0.8M narrations. Using our Narration Pairing Encoding method, we generate a prefix set containing 0.6M entries and a postfix set with 5K entries, where the postfix is shared across all narrations and deduplicated. Finally, we create a training and evaluation split at a 10:1 ratio, referred to as VidCabTrain and VidCab-Eval, respectively.

#### A.2. Narration Pairing Encoding

In the above algorithm, we display the process of our Narration Pairing Encoding algorithm, which mainly includes two parts: (i) Build Prefix Dictionary: This step exhaustively enumerates all possible word combinations for each phrase to build a map between any prefix and the corresponding postfix narrations. (ii) Extract Prefixes and Postfixes: For each narration, we determine whether other narrations share its full prefix. If not, we add it to the prefix list. If they do, we extract and collect the differing postfixes from the narrations that share its prefix.

#### A.3. Vocabulary Update Templates

In Fig. 10, we display how VLog progressively decode the prefix and postfix respectively. It first use the memory token to retrieve the prefix narration, and next it append the prefix narration and use the memory token to retrieve the postfix for a full narration.

Below, we attach the prompts for Qwen2.5 [47], which is used for produce narrations directly.

List possible short actions that could take place in the scene. Write each action as a short narration (a verb with a noun). Separate by ’;’ The following is examples.

scene: In the heart of the kitchen, a man skillfully slices into a ripe mango, its golden flesh gleaming under the light.

narration: Slice mango; Hold knife; Cut mango; Place seed; Wipe counter; Drop pieces; Grip mango; Rest knife; Smell mango; Gather chunks.

scene: A woman sits by the fireplace, knitting a scarf as the flames crackle warmly in the background.

narration: Knit scarf; Hold needles; Loop yarn; Adjust thread; Pull stitch; Rest hands; Drop yarn; Smell smoke; Listen flames; Rub hands; Fold scarf; Gather wool; Stare fire; Sit still; Tap needle.

scene: {scene} narration:

The {scene} is output by LLaVA-OV-0.5B [22] with prompt: “What is the overall activity in the scene? Answer briefly in one sentence.”

#### A.4. Experimental Settings

Our SigLIP model [54] is based on google/siglip-so400m-patch14-384. During training, we fully fine-tune the GPT-2 model with a batch size of 32, a learning rate of 3e-4, and a sampling rate of 8 frames per short video clip. For long videos, such as those in the EgoSchema dataset [33], we do not compress the entire video into a single embedding. Instead, we uniformly sample long videos into multiple fixed-length clips (1 second each) and process them in a streaming fashion.

#### A.5. Complexities Analysis

Let us clarify each term when generating N narrations: (i) Encoding: We embed the entire vocabulary once and then reuse it – O(1). (ii) Decoding: This should be O(αN), where α is the speed per decoding step. (iii) Upgrading (optionally): O(C), where C is the upgrade times (C ≪ N). For a large N, the overall complexity O(1)+O(αN)+O(C) → O(αN) remains efficient as the encoding and upgrading costs become negligible. Below is the timing analysis on 4.6K VidCap-Eval:

|Models Vocab. size<br><br>|Encoding(s) Decoding(s) Upgrading(s)|R@1 Total (s)<br><br>|
|---|---|---|
|GPT-2 32K VLog 4.6K VLog 4.6K (+486)|– 207.8 –<br><br>3.6 10.4 –<br><br>3.6 10.5 38.4<br><br><br>|7.9 207.8<br><br>12.4 14.0<br>13.7 52.4<br>|

#### A.6. Subwords v.s. Narration Vocabulary on Easy v.s. Complex tasks?

Our VLog is prioritize task-specific efficiency over generalist models. We compare the two in the below Table.

| |Domain Vocabulary Backbone Decoding Highlights|
|---|---|
|VideoLLMs VLog<br><br>|General Subwords LLMs (2B+) Token Gen. Multi-Purpose Specific Narrations GPT-2 (345M) Retrieval Efficiency|

Whether ‘Subwords-’ or ‘Narration-’ Vocabulary is depends on how tasks define minimal semantic units for videos. Subwords capture every detail but may be redundant for long videos, while narrations offer event contexts quickly but may miss finer details. To balance expressive granularity and efficiency, an idea is to cooperate two fashions like our vocab. upgrading or retrieve narration first and then generate minimal subwords as needed. We are interested in further exploring the latter.

- A.7. Improvment by Stronger LLM

We chose GPT-2 because its simplicity and lightweight nature make it a representative baseline. To demonstrate scaling, we upgraded GPT-2 to Qwen2-7B, resulting in significant performance gains, and beat its comparable baseline Qwen2-VL-7B.

|Models<br><br>|Size EgoSchema QA val. Decoding Time (s)|
|---|---|
|Qwen2-VL VLog (GPT-2) VLog (Qwen2)|7B 72.8 79.4<br><br>345M 70.4 2.3 7B 74.8 6.4<br><br>|

- B. Qualitative Examples

- B.1. VLog for Reasoning Retrieval

- In Fig.12, we illustrate how VLog retrieves the vocabu-

lary (blue indicating prefixes and green indicating postfixes) conditioned on different queries. For instance, in example (b), the query “What is the next activity in the video?” retrieves “Grab a bag of chips using the left hand” as a result, while the query “What is the previous activity in the video?” retrieves “Adjust the steering wheel using the hand” as a result, demonstrating VLog ’s capability to infer relationships between sequential events.

B.2. How does Vocabulary Updating work?

- In Fig.13, we demonstrate how VLog’s vocabulary up-

dating process effectively expands its descriptive range. Given the first frame of a video clip, LLaVA-OV [22] generates an initial brief description, which is then passed to Qwen2.5 [47] to imagine and expand possible vocabulary terms. For instance, in (a), LLaVA-OV identifies a simple construction project involving multiple yellow pencils, and Qwen2.5 extends this by generating potential actions such as “Arrange pencils” and “Hold pencils,” which collectively capture most events in the video.

However, limitations still exist with the models. For example, in (c), while the activity “Make Pineapple Fritters” is identified, the model struggles to detect the specific ingredient “pineapple,” making it challenging for the expanded vocabulary to recognize or describe the desired object accurately. These challenges highlight areas for improvement in object-specific vocabulary generation.

#### B.3. Limtation by VLog.

[Figure 52]

Broad Descriptive Range A mother helping her child with homework.

[Figure 53]

###### Character Information

Hermione told Harry a piece of news that surprised Harry.

- Figure 11. VLog still fail to capture the video with broad descriptive range or high-level information e.g. characters.

We acknowledge that VLog still has limitations, as illustrated in 11. For example, when videos have a broad expressive range, such as those involving multiple individuals or focusing on different aspects depending on subjective interpretation, it becomes challenging to rely on a narrationwise closed-set vocabulary. Additionally, in more complex scenarios, such as movies, where character information and dialogues play a central role, the current approach struggles. These cases may require a return to a generalist model capable of handling subword tokens for richer representations.

## References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.
- [2] Akari Asai, Timo Schick, Patrick Lewis, Xilun Chen, Gautier Izacard, Sebastian Riedel, Hannaneh Hajishirzi, and Wen-tau Yih. Task-aware retrieval with instructions. arXiv preprint arXiv:2211.09260, 2022.
- [3] Kumar Ashutosh, Santhosh Kumar Ramakrishnan, Triantafyllos Afouras, and Kristen Grauman. Video-mined task graphs for keystep recognition in instructional videos. In NeurIPS, 2023.
- [4] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738, 2021.
- [5] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In ICML, 2021.
- [6] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015.

- [7] Dibyadip Chatterjee, Fadime Sener, Shugao Ma, and Angela Yao. Opening the vocabulary of egocentric actions. Advances in Neural Information Processing Systems, 36, 2024.
- [8] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18407–18418, 2024.
- [9] Tianyi Cheng, Dandan Shan, Ayda Hassen, Richard Higgins, and David Fouhey. Towards a richer 2d understanding of hands at scale. Advances in Neural Information Processing Systems, 36:30453–30465, 2023.
- [10] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023.
- [11] Ahmad Darkhalil, Dandan Shan, Bin Zhu, Jian Ma, Amlan Kar, Richard Higgins, Sanja Fidler, David Fouhey, and Dima Damen. Epic-kitchens visor benchmark: Video segmentations and object relations. Advances in Neural Information Processing Systems, 35:13745–13758, 2022.
- [12] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [13] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A memory-augmented multimodal agent for video understanding. arXiv preprint arXiv:2403.11481, 2024.
- [14] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023.
- [15] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995–19012, 2022.
- [16] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. arXiv preprint arXiv:2311.18259, 2023.
- [17] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec: Training vision-language models for massive multimodal embedding tasks. arXiv preprint arXiv:2410.05160, 2024.
- [18] Kumara Kahatapitiya, Kanchana Ranasinghe, Jongwoo Park, and Michael S Ryoo. Language repository for long video understanding. arXiv preprint arXiv:2403.14622, 2024.
- [19] Jing Yu Koh, Ruslan Salakhutdinov, and Daniel Fried. Grounding language models to images for multimodal in-

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

What is the activity in the video? Making Bricks What is the activity in the video? (Hit the bricks with mallet) (using the right hand) What is the next activity in the video? (Arrange brick) (using the left hand) What is the previous activity in the video? (Pour out the soil) (with both hands)

- (a)
- (b)
- (c)
- (d)

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

What is the activity in the video? Grocery shopping indoors What is the activity in the video? (Look at snacks) (on a shelf) What is the next activity in the video? (Grab a bag of chips) (using the left hand) What is the previous activity in the video? (Adjust the steering wheel) (using the hand)

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

What is the activity in the video? Bike mechanic What is the activity in the video? (Adjust a wheel) (on a table) What is the next activity in the video? (Place down a tire) (on a table) What is the previous activity in the video? (walks around) (a garage)

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

What is the activity in the video? Indoor Navigation What is the activity in the video? (Adjust a wheel) (on a table) What is the next activity in the video? (Place down a tire) (on a table) What is the previous activity in the video? (walks around) (a garage)

- Figure 12. VLog enables retrieval through reasoning, conditioned on different queries. Blue represents prefixes, while green represents postfixes.

puts and outputs. In International Conference on Machine Learning, pages 17283–17300. PMLR, 2023.

- [20] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nvembed: Improved techniques for training llms as generalist embedding models. arXiv preprint arXiv:2405.17428, 2024.
- [21] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L. Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In CVPR, pages 7331–7341, June 2021.
- [22] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [23] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.
- [24] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.
- [25] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu

- Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10965–10975, 2022.
- [26] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. arXiv preprint arXiv:2311.17043, 2023.
- [27] Kevin Qinghong Lin, Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Z XU, Difei Gao, Rong-Cheng Tu, Wenzhe Zhao, Weijie Kong, et al. Egocentric video-language pretraining. Advances in Neural Information Processing Systems, 35:7575–7586, 2022.
- [28] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. Univtg: Towards unified videolanguage temporal grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2794–2804, 2023.
- [29] Xudong Lin, Fabio Petroni, Gedas Bertasius, Marcus Rohrbach, Shih-Fu Chang, and Lorenzo Torresani. Learning to recognize procedural activities with distant supervision. In CVPR, pages 13843–13853, 2022.
- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

[Figure 70]

LLaVA-OV-0.5B: The overall activity in the scene appears to be a simple construction project involving multiple yellow pencils. The image shows the pencils arranged in a triangular formation, with one pencil on top of another, suggesting a structure or object that might be crafted or assembled using these pencils.

Qwen2.5-0.5B: 'Arrange pencils; Hold pencils; Position pencils; Arrange vertically; Rotate pencils; Use pencil sharpener; Apply pressure; Press and hold pencil; Move pencil up and down; Press and hold pencil again; Push pencil back down; Check if pencils are aligned correctly; Continue until complete; Clean up workspace; Repeat process.'

(a): Build a Trebuchet

[Figure 71]

LLaVA-OV-0.5B: The overall activity in the scene is a person holding and working on a paper flower from a paper bird origami kit

Qwen2.5-0.5B: 'Hold paper flower; Unfold paper; Shape flower petals; Arrange petals; Pinch stem; Secure flowers; Review design; Repeat assembly; Take final step; Gently unfold.'

(b): Fold an Origami Parrot

[Figure 72]

LLaVA-OV-0.5B: The overall activity in the scene is cooking, specifically preparing batter for frying or baking.

Qwen2.5-0.5B: 'Mix dough; Pour flour into bowl; Stir until smooth; Add water; Knead dough; Place dough in bowl; Cover and knead; Roll out dough; Heat oil in pan; Place dough in hot oil; Fry or bake; Serve; Repeat process if needed; Cook thoroughly; Allow to cool; Serve immediately.'

(c): Make Pineapple Fritters

[Figure 73]

LLaVA-OV-0.5B: The overall activity in the scene involves creating a little twiisesie texture on a doll by using bantu knots.

Qwen2.5-0.5B: 'Knit bantu knot; Hold doll; Wrap loop; Insert loop; Twist loops; Pinch fingers; Pull tight; Gently pull; Rest doll; Repeat; Create twiisesie pattern; Gently stretch; Stretch doll; Adjust tension; Press firmly; Rest doll.'

(d): Cut a Doll's Hair

- Figure 13. Illustration of VLog’s vocabulary updating process. Given the first frame of a video clip, LLaVA-OV [22] provides a brief initial description, which is then passed to Qwen2.5 [47] to generate and expand the possible vocabulary.

- 2023.

- [31] Feipeng Ma, Hongwei Xue, Guangting Wang, Yizhou Zhou, Fengyun Rao, Shilin Yan, Yueyi Zhang, Siying Wu, Mike Zheng Shou, and Xiaoyan Sun. Multi-modal generative embedding model. arXiv preprint arXiv:2405.19333,

2024.

- [32] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023.
- [33] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. arXiv preprint

arXiv:2308.09126, 2023.

- [34] Meta. Build the future of ai with meta llama 3. https: //llama.meta.com/llama3/, 2024.
- [35] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2630–2640, 2019.
- [36] M Minderer, A Gritsenko, A Stone, M Neumann, D Weissenborn, A Dosovitskiy, A Mahendran, A Arnab, M Dehghani, Z Shen, et al. Simple open-vocabulary object detection with vision transformers. arxiv 2022. arXiv preprint

- arXiv:2205.06230, 2, 2022.
- [37] Medhini Narasimhan, Licheng Yu, Sean Bell, Ning Zhang, and Trevor Darrell. Learning and verification of task structure in instructional videos. arXiv:2303.13519, 2023.
- [38] Shraman Pramanick, Yale Song, Sayan Nag, Kevin Qinghong Lin, Hardik Shah, Mike Zheng Shou, Rama Chellappa, and Pengchuan Zhang. Egovlpv2: Egocentric video-language pre-training with fusion in the backbone. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5285–5297, 2023.
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang, editors, ICML, pages 8748–8763, 2021.
- [40] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, page 9, 2019.
- [41] Kanchana Ranasinghe, Xiang Li, Kumara Kahatapitiya, and Michael S Ryoo. Understanding long videos in one multimodal language model pass. arXiv preprint arXiv:2403.16998, 2024.
- [42] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.
- [43] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. Longvu: Spatiotemporal adaptive compression for long video-language understanding, 2024.
- [44] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Xun Guo, Tian Ye, Yan Lu, Jenq-Neng Hwang, et al. Moviechat: From dense token to sparse memory for long video understanding. arXiv preprint arXiv:2307.16449, 2023.
- [45] Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A Smith, Luke Zettlemoyer, and Tao Yu. One embedder, any task: Instruction-finetuned text embeddings. arXiv preprint arXiv:2212.09741, 2022.
- [46] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1207– 1216, 2019.
- [47] Qwen Team. Qwen2.5: A party of foundation models, September 2024.
- [48] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pretraining. arXiv preprint arXiv:2212.03533, 2022.
- [49] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understand-

- ing with large language model as agent. arXiv preprint arXiv:2403.10517, 2024.
- [50] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022.
- [51] Shiwei Wu, Joya Chen, Kevin Qinghong Lin, Qimeng Wang, Yan Gao, Qianli Xu, Tong Xu, Yao Hu, Enhong Chen, and Mike Zheng Shou. Videollm-mod: Efficient videolanguage streaming with mixture-of-depths vision computation. Advances in Neural Information Processing Systems, 37:109922–109947, 2024.
- [52] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.
- [53] Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vidchapters-7m: Video chapters at scale. arXiv preprint arXiv:2309.13952, 2023.
- [54] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training, 2023.
- [55] Chuhan Zhang, Ankush Gupta, and Andrew Zisserman. Helping hands: An object-aware ego-centric video recognition model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13901–13912, 2023.
- [56] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple llm framework for long-range video question-answering. arXiv preprint arXiv:2312.17235, 2023.
- [57] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.
- [58] Yue Zhao, Ishan Misra, Philipp Kr¨ahenb¨uhl, and Rohit Girdhar. Learning video representations from large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6586– 6597, 2023.
- [59] Yiwu Zhong, Licheng Yu, Yang Bai, Shangwen Li, Xueting Yan, and Yin Li. Learning procedure-aware video representation from instructional videos and their narrations. In CVPR, pages 14825–14835, 2023.
- [60] Honglu Zhou, Roberto Mart´ın-Mart´ın, Mubbasir Kapadia, Silvio Savarese, and Juan Carlos Niebles. Procedure-aware pretraining for instructional video understanding. In CVPR, pages 10727–10738, 2023.
- [61] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

