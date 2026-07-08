# arXiv:2512.15560v2[cs.CV]28Dec2025

[Figure 1]

## GRAN-TED: Generating Robust, Aligned, and Nuanced Text Embedding for Diffusion Models

Bozhou Li1,2*† Sihan Yang1,3* Yushuo Guan2* Ruichuan An1 Xinlong Chen4 Yang Shi1 Pengfei Wan2 Wentao Zhang1 Yuanxing Zhang2 1Peking University 2Kling Team, Kuaishou Technology 3Xi’an Jiaotong University 4School of Artificial Intelligence, UCAS

#### Abstract

The text encoder is a critical component of text-to-image and text-to-video diffusion models, fundamentally determining the semantic fidelity of the generated content. However, its development has been hindered by two major challenges: the lack of an efficient evaluation framework that reliably predicts downstream generation performance, and the difficulty of effectively adapting pretrained language models for visual synthesis. To address these issues, we introduce GRAN-TED, a paradigm to Generate Robust, Aligned, and Nuanced Text Embeddings for Diffusion models. Our contribution is twofold. First, we propose TED-6K, a novel textonly benchmark that enables efficient and robust assessment of an encoder’s representational quality without requiring costly end-to-end model training. We demonstrate that performance on TED-6K, standardized via a lightweight, unified adapter, strongly correlates with an encoder’s effectiveness in downstream generation tasks. Notably, under our experimental setup, compared with training a diffusion model from scratch, evaluating with TED-6K is about 750× faster. Second, guided by this validated framework, we develop a superior text encoder using a novel two-stage training paradigm. This process involves an initial fine-tuning stage on a Multimodal Large Language Model for better visual representation, followed by a layer-wise weighting method to extract more nuanced and potent text features. Our experiments show that the resulting GRAN-TED encoder not only achieves state-of-the-art performance on TED-6K but also leads to demonstrable performance gains in text-to-image and text-to-video generation. Our TED6K dataset and evaluation code are available at the following link: https://anonymous.4open.science/ r/GRAN-TED-4FCC/.

*Equal contribution †Work done during an internship at Kling Team

#### 1. Introduction

Diffusion models have recently emerged as the dominant paradigm for text-to-image (T2I) [3, 14, 22, 40, 43, 44] and text-to-video (T2V) [13, 20, 35, 47, 52] generation. Central to these models is the text encoder, which transforms textual prompts into semantic representations that guide the visual synthesis process. An effective text encoder must accurately represent a wide range of semantic elements, including subject features, static attributes, spatial relationships, and temporal events. This capability is crucial for enabling the generative model to achieve compositional generalization—synthesizing novel scenes from combinations of concepts not explicitly seen during training. Despite its critical role, the importance of the text encoder was largely overlooked in the early stages of diffusion model development, with most research focusing on the diffusion mechanism itself. This oversight is consequential, as the quality of the text encoder fundamentally determines the semantic fidelity between the prompt and the generated visual content.

This trend has begun to reverse, with a clear architectural evolution in text encoders. The community has progressed from using early CLIP-based encoders [44] to more powerful models like T5 [22], and has now converged on integrating Large Language Models (LLMs) as the state-ofthe-art backbone [14, 48]. The rationale is clear: the advanced semantic understanding of LLMs promises to directly translate into higher-fidelity visual generation, improving the alignment between text and image.

However, despite the adoption of these powerful LLMs, generative models still exhibit significant challenges in prompt fidelity. Common failures observed in practical applications and on benchmark evaluations include incorrect object counts, misinterpretation of relational or referential phrases, attribute binding errors, and a failure to adhere to negative constraints. As the crucial module between textual descriptions and visual synthesis, a more accurate and robust text representation is paramount to addressing these

shortcomings. Developing a superior text encoder tailored for generative models presents two primary challenges:

- • Evaluating Representation Quality for Generation: A significant hurdle is the lack of a suitable evaluation framework. Standard NLP benchmarks for LLMs do not directly correlate with performance in visual generation. Similarly, metrics from information retrieval fail to capture critical dimensions such as aesthetic quality, compositional complexity, or nuanced world knowledge. Conversely, evaluating an encoder by training a full generative model end-to-end is computationally prohibitive and inefficient for rapid iteration.
- • Adapting LLM Representations for Visual Synthesis: The second challenge lies in effectively adapting the rich features of a pre-trained LLM for the specific demands of visual generation. It is non-trivial to ensure that the semantic information encoded in the LLM’s latent space is represented in a manner that is precise, unambiguous, and fully interpretable by the diffusion model, thereby translating improved textual understanding into tangible gains in generation fidelity.

To address these two challenges, we propose GRANTED to Generate Robust, Aligned, and Nuanced Text Embedding for Diffusion models. We first introduce a robust and unified evaluation framework centered on a novel textonly benchmark named TED-6K. The TED-6K benchmark, composed of visual captions and verified true/false statements, allows for efficient and robust assessment of an encoder’s representational quality. We demonstrate that an encoder’s performance on TED-6K, which is measured by representation similarity and standardized via a lightweight adapter, strongly correlates with its effectiveness in downstream text-to-image and text-to-video generation. In our experimental setup , training a T2I diffusion model from scratch for evaluation takes about 50 hours, whereas evaluating an encoder with TED-6K takes only about 4 minutes, enabling 720× faster iteration.

Leveraging this validated framework, we then explore a two-stage paradigm to develop a superior text encoder GRAN-TED. In the first stage, we finetune Qwen3-VL-8BInstruct [46] for better aligned latent visual representation towards generation. In the second stage, we implement a novel layer-wise weighting method, which extracts more nuanced text features from the layers of the trained model in the first stage. Our experiments confirm that this two-stage paradigm yields a new robust text encoder on TED-6K, simultaneously leading to demonstrable performance gains in T2I and T2V models.

#### 2. Related Works

##### 2.1. Text Encoder for Diffusion Models.

In diffusion models, the selection of text encoders is diverse. Early diffusion models commonly employ the text encoder from CLIP [41] or the encoder module from T5 [42], since their pre-training objectives are inherently designed to efficiently encode text into semantic representations [39, 44]. As the potential of decoder-only LLMs [33, 55] for text encoding tasks is gradually unveiled [5, 17], some researchs begin to select LLMs or MLLMs [2, 28, 46] as text encoders [25, 29, 34, 47] for diffusion models.

Besides leveraging the single feature layer from a single text model, some approaches employ the feature fusion in the text encoder to enhance semantic capabilities. Seedream [14] integrates LLMs and a glyph-aligned model as the text encoder. While most contemporary diffusion models tend to use the hidden states from a single layer of the text encoder as the text condition (typically the penultimate layer), the work by Wang et al. [48] points to a superior feature aggregation strategy. They demonstrate that normalizing and averaging the hidden states from all layers (NormAvg) yields a more potent text representation than using any single-layer features or simple averaging (Avg):

1 N

ctext =

N

LayerNorm(hi) (1)

i=1

However, these methods may not fully explore the connection between the different representational capabilities of each layer of the language model and downstream tasks. Further studies also explore post-training strategy of the text encoder after the end-to-end training of diffusion models [6, 27], requiring heavy training costs.

##### 2.2.Text-alignment BenchmarksforText Encoders.

Prevailing text-alignment benchmarks focus on assessing the correspondence between the final visual output and the input text prompt. For instance, GenAI-Bench [30] leverages a Vision-Language Model (VLM) to judge the alignment of generated images in terms of various attributes. Similarly, VIEScore [21] and GPT4-Eval [56] employ GPT4V [38] to directly produce an image-text alignment score based on the visual output. These methods are costy and inefficient, since they assess the diffusion model as a whole, rather than isolating the text encoder’s specific contribution. This necessitate the training of a distinct diffusion model for each candidate encoder—a prohibitively costly endeavor.

Some other methods leverage existing retrieval benchmarks and LLM benchmarks as proxy evaluations. Retrieval-based benchmarks like MTEB [36] are commonly-used proxy tasks. The text embeddings are encoded into a single pooled vector in these benchmarks,

which has a granularity mismatch with the usage in diffusion models, since the text embeddings in diffusion models leverage the full sequence of hidden states for conditioning [13, 14, 20, 35, 40, 47]. LLM benchmarks like MMLU [16] and AIME [1] are neither good proxy tasks, since their metrics focus on the perception and reasoning abilities, rather than the text alignment ability in visual generation.

#### 3. Method

##### 3.1. Overview

As illustrated in Figure 1, we have developed a robust pipeline designed to identify the optimal text encoder for diffusion models from a vast pool of candidates. First, we introduce TED-6K, a benchmark specifically tailored to the domain of visual generation. Unlike traditional methods [15, 21, 30], TED-6K serves as a text-only evaluation protocol, enabling a multidimensional assessment of an encoder’s representational capabilities without the prohibitive cost of training a full diffusion model end-to-end. Next, we develop a unified context aggregator to ensure a fair comparison across diverse architectures, including CLIP, T5, LLMs and MLLMs, which allows us to equitably evaluate the representational quality of various models on the TED-6K benchmark. Guided by this rigorous screening process, we identify Qwen3-VL-8B-Instruct [46] as the superior backbone. Building upon this backbone, we finetune it on the captioning dataset to develop an enhanced text encoder, termed GRAN-TED. Furthermore, we investigate strategies to efficiently fuse distinct feature layers within GRAN-TED, thereby maximizing the quality of the resulting text embeddings. The specifics of the benchmark formulation, the context aggregator architecture, our developed encoder, and the feature fusion strategy are detailed in Sections 3.2 - 3.5, respectively.

##### 3.2. Text-only Benchmark for Evaluation

We first build a text-only benchmark TED-6K illustrated in Figure 2. Our goal is to systematically create a dataset capable of probing a text encoder’s representational capabilities across key semantic dimensions. To this end, we design and execute the following multi-stage data pipeline:

- • Raw Data Curation and Filtering: We begin by collecting raw image and video materials from several highquality open-source datasets, including DREAM1K [49], CAPability [31], V-STaR [9], and CapArena [8]. These sources are selected for their diversity, high visual fidelity, and semantic complexity.
- • Generation of Fine-Grained Base Captions: For each filtered visual sample, we utilize the advanced capabilities of Gemini 2.5 Pro [11] to generate a fine-grained descriptive caption. We guide the model to produce descriptions of the utmost detail, while ensuring maximum coverage

- of our predefined semantic categories.
- • Construction of Multi-Dimensional Semantic Evaluation Pairs: To probe specific semantic capabilities, we construct evaluation data for each base caption from eight predefined semantic perspectives: action, spatial relationship, temporal relationship, coreference, adjectives, adverbs, quantity, OCR, and basic event. First, if the original base caption contains information that could be articulated from a given perspective, we prompt the model to generate one positive statement consistent with the base caption’s content. Subsequently, to construct high-quality negative samples, we instruct the model to perform three distinct types of targeted semantic modifications on this positive statement, such as attribute swapping or relation reversal. A critical constraint is that each modification has to create an explicit semantic contradiction with the original base caption, aiming to generate “hard negatives” that are plausible yet factually incorrect.
- • Rigorous Human Verification: Finally, all statements undergo rigorous manual verifications to ensure the consistency of positive statements with the original caption, and the validity and confusability of negative statements.

This pipeline culminates in a dataset of 6,641 evaluation instances, each comprising a source caption, a corresponding positive statement, and a set of semantically contradictory negative statements. The specific distribution of evaluation instances across the different semantic categories is illustrated in Figure 2.

##### 3.3. Sentence-level Context Aggregator

It is difficult to fairly evaluate the representation capability of different kinds of text encoders with traditional metics, like single-vector retrieval or Question-Answer (QA) accuracy. Most text encoders like LLM or MLLMs have no single-vector retrieval capability. Meanwhile, evaluating text encoders for diffusion models via question-answering (QA) accuracy is misleading. This approach is problematic because it cannot be applied to encoders like CLIP that lack QA functionality. More importantly, it conflates two distinct mechanisms within LLMs. QA relies on causal decoding using prefilling features from all model layers to infer an answer, whereas a Diffusion Transformer (DiT) typically uses a static embedding from a single layer as its conditioning. Therefore, an LLM’s reasoning prowess in QA is not a reliable proxy for its ability to generate high-quality representations for visual generation.

To fairly assess the representation capability of different text encoders when used in DiTs, we design a unified context aggregator that mimics how DiTs consume text embeddings, and evaluate them at the representation level on TED-6K. The context aggregator, Acontext, consists of two self-attention Transformer layers and a learnable context token, ccontext. It functions by aggregating the sentence-level

[Figure 2]

[Figure 3]

[Figure 4]

Benchmarking by TED-6K

Specializing MLLM

Layer-wise Feature Weighting

[Figure 5]

Qwen3-VL Training

Semantic Perspective of T2I and T2V

Two-step Training Strategy

Video / Image Q & A

[Figure 6]

Question: What is the main subject of this video?

| | |
|---|---|
| | |
| | |

DiT Backbone

DiT Backbone

[Figure 7]

[Figure 8]

Base Model: Qwen3-VL 🏅 > MiMo-VL🥈 > ...

Answer: A ﬁrst-person view of a skier racing down a steep, snowy mountain slope in an ...

| | |
|---|---|
|Weig|hting<br><br>[Figure 9]|
| | |

| | |
|---|---|
|Weig|hting<br><br>[Figure 10]|
| | |

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Caption: The video begins from a ﬁrst-person perspective atop a towering snowy ridge, with a wide yet exceptionally steep snow valley ahead and distant mountain ranges clearly visible under a clear blue sky. ...

Context Aggregator Training

Step 1 Step 2

. . .

[Figure 17]

Decoder-only LLM

Encoder-only

MLLM

[Figure 18]

GRAN-TED w/o Weighting

UMT5-XXL

Qwen3 Qwen3-VL

AltCLIP

InternLM MiMo-VL

Caption Dataset

T2I/T2V Dataset

. . . . . . . . .

DenseFusion MiraData . . . VideoUFO

ScaleCap Koala-36M . . . MiraData

(a) (b) (c)

- Figure 1. An overview of our complete framework, integrating our evaluation and development pipelines. Figure (a) illustrates the TED Evaluation Framework, consisting of the TED-6K benchmark and a context aggregator to assess the representational capabilities of text encoders. Figure (b) shows the construction of our TED Encoder, where Qwen3-VL-8B-Instruct is fine-tuned on a curated VQA and captioning dataset to specialize the MLLM. Figure (c) depicts our final GRAN-TED solution, which incorperates a learnable layer-wise weighting module to generate GRAN-TED for diffusion models.

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

winter or early spring day, with no movement present in the scene.

background, a window showing bare tree branches suggests a

atmosphere of the cozy indoor setting. Through the clear

visible jar with colorful spoons adding to the warm, inviting

is stationary, with a nearby cup labeled 'V ANILLA Latte' and a

revealing a ﬂuffy texture and inviting appearance. The entire scene

topped with whipped cream and a powdered sugar-dusted biscuit,

detail, showcasing each pastry's golden-brown crust, which is

a slightly elevated angle focuses on texture and

a balanced perspective, the close-up shot from

the foreground with a standard lens providing

ﬂ oral-patterned plate. Captured prominently in

together in a center composition on a decorative,

three cream-ﬁlled pastries are arranged closely

deﬁned by soft lighting and pastel colors,

In a bright, realistic, and cozy composition

is stationary, with a nearby cup labeled 'VANILLA Latte' ...

revealing a ﬂuffy texture and inviting appearance. The entire scene

topped with whipped cream and a powdered sugar-dusted biscuit,

elevated angle focuses on texture and detail, show..., which is

providing a balanced perspective, the close-up shot from a slightly

plate. Captured prominently in the foreground with a standard lens

together in a center composition on a decorative, ﬂoral-patterned

and pastel colors, three cream-ﬁlled pastries are arranged closely

In a bright, realistic, and cozy composition deﬁned by soft lighting

C. The cup labeled 'VAMILLA Late ' D. The cup labeled 'Vamille Lette'

A. The cup labeled 'VANILLA Latte' B. The cup labeled 'Vanilla Latte'

Final Evaluation Case

Human Correcting

Incorrect Statement Generating

VANILLA Latte Vanilla Latte VAMILLA Latte Vamilla Latte

VisualSemanticExtractingbyCategories

Action

Coreference

Relationship

Temporal

Relationship

Spatial

Adjective

Adverb

Quantity

Basic Event

OCR

DREAM-1K V-STaR CapArena CAPability

[Figure 27]

[Figure 28]

TED-6K

OCR

Action [755]

Temporal [608]

[680]

Spatial

[828]

Adverb Basic Event

[1025]

Coreference [762]

Adjective [1021]

[748] [214]

Quantity

Visually-rich Image&Video Data Collecting

Dense Caption Generating

- Figure 2. Left. The data construction pipeline for TED-6K, consisting of four stages: (1) Data Curation and Filtering; (2) Base Caption Generation; (3) Semantic Pair Construction; (4) Human Verification. Right. The data Composition of the TED-6K dataset.

semantics from an input text representation, ctext, into the context token. This information flow can be represented as:

c′context ←− Acontext([ctext,ccontext]) (2)

where c′context is the updated context token, now infused with the information from ctext. The detailed architecture is provided in the Appendix A.

Figure 3 illustrates the overall workflow of our evaluation framework. To construct the positive pairs for training, we employed the following strategy for each image: using

the Qwen3-VL-235A22B-Instruct, we generated two semantically similar but textually diverse captions by prompting it with two distinct prompts at a temperature of 1.0. Overalll, we collect 500k caption pairs from DenseFusion [26], ScaleCap [54], MiraData [18], Koala-36M [50], and Video-UFO [51]. Notably, the data in the TED-6K benchmark is distinct from our fine-tuning dataset, ensuring a fair evaluation of the model’s generalization capabilities.

To enable the context aggregator Acontext to effectively aggregate core sentence-level semantics from the sequence of hidden states, we train it using a contrastive loss [37].

[Figure 29]

[Figure 30]

[Figure 31]

Captioner A

Caption A

EmbeddingGenerator

ContextAggregator

ContrastiveLoss

TextEncoder

[Figure 32]

[Figure 33]

[Figure 34]

Caption

- A

Captioner

- B

- B

Caption

- C

[Figure 35]

[Figure 36]

[Figure 37]

Captioner

Text Encoder Embedding Generator Context Aggregator

Last Layer

Encoder-only

TransformerLayer

TransformerLayer

Penultimate Layer

Decoder-only LLM

Norm Layer + Avg.

MLLM

Layer-Wise Feature Weighting

[Figure 38]

[Figure 39]

(a)

Caption: In a bright, realistic, and cozy composition defined by soft lighting and pastel colors, three cream-filled pastries are arranged closely together in ... the close-up shot from a slightly elevated angle focuses on texture and detail, showcasing each pastry's golden-brown crust, which is topped with whipped cream and a powdered sugar-dusted biscuit, revealing a fluffy texture and inviting appearance. The entire scene is stationary, with a nearby cup labeled 'VANILLA Latte' ...

EmbeddingGenerator

ContextAggregator

TextEncoder

Cosine Similarity

- Statement A: The cup labeled 'VANILLA Latte'

- Statement B: The cup labeled 'VAMILLA Latte'

- Statement C: The cup labeled 'Vanilla Latte'

- Statement D: The cup labeled 'Vamilla Latte'

(b)

- Figure 3. The context aggregator architecture and its training&inference process. (a) training process of the context aggregator. (b) the inference process during evaluation on TED-6K.

Section 4.1. This superiority is largely attributable to their multimodal training, which forces the text encoder to learn a semantic space that inherently aligns with visual concepts.

However, the training objectives of most existing MLLMs are geared towards developing general-purpose multimodal reasoning capabilities, with their training data often incorporating tasks like visual grounding and multimodal reasoning. These tasks primarily enhance a model’s reasoning abilities, while its core function as a text encoder for generative tasks—the capacity to provide high-quality textual representations—is not explicitly optimized.

These analyses raise a natural question: can we develop an even more potent model by fine-tuning a strong MLLM backbone on tasks specifically tailored for visual generation? Motivated by this hypothesis, we introduce GRAN-TED, a text encoder specialized for visual generation, which we developed by building upon the Qwen3-VL8B-Instruct model. Specifically, we adopt a data-centric approach [4] to enhance the model’s capability on text embeddings. We have meticulously collected a large-scale dataset of images/videos and their corresponding semantically rich captions. Based on this collection, we construct a massive set of Visual Question Answering (VQA) pairs from multiple perspectives directly relevant to visual generation, such as object attributes, spatial relationships, and temporal order. By fine-tuning the Qwen3-VL-8B-Instruct model on this highly targeted VQA dataset, we develop GRAN-TED, a text encoder enhanced for diffusion models.

First, both captions from a positive pair are passed through a frozen text encoder (using various feature extraction configurations like Last Layer or Norm-Avg) to obtain their respective textual representations, cAtext and cBtext. Then, the aggregator, Acontext, processes each of these representations to produce their corresponding context tokens, c′context, A and c′context, B. Finally, we apply the contrastive loss, which is trained to pull together the context tokens from the same image while pushing apart those from different images.

During the evaluation on TED-6K, a source caption and its candidate statements are passed through a frozen text encoder, feature extractor and the context aggregator that were specifically trained for it, to obtain their respective context tokens. An evaluation instance is marked as correct if the similarity score between the source caption and the positive statement is the highest among all candidates. The final accuracy is the proportion of correctly identified instances.

##### 3.4. Specializing MLLMs as Text Encoders for Visual Generation

Upon the context aggregator and TED-6K, we find that MLLMs exhibit a significant advantage over other models when used as text encoders for generative tasks, detailed in

##### 3.5. Layer-wise Feature Weighting

Empirical results from Wang et al. [48] show that normalizing and averaging hidden states (Norm-Avg) significantly outperforms simple averaging (Avg). In pre-norm LLMs, where the norm of hidden states tends to increase with layer depth [19, 24], the Norm-Avg operation not only serves to stabilize the numerical scale of the features but also functions as an implicit, inverse-norm weighting scheme.

This insight inspires us to hypothesize that if a fixed, implicit weighting is beneficial, an explicit and learnable weighting mechanism could achieve superior performance. To this end, we propose a learnable, layer-wise feature weighting module. The module first applies layer normalization to the hidden state hi of each layer. It then assigns a set of learnable scalar weights wi to these normalized representations, which are converted into attention scores αi via a softmax function:

exp(wi)

αi =

L j=1 exp(wj)

(3)

The final fused representation ctext is then computed as

the weighted sum of these normalized representations:

ctext =

L

αi · LayerNorm(hi) (4)

i=1

This module allows the model to autonomously learn the importance of features from different hierarchical levels as required by the task, thereby constructing a more informative textual representation.

However, the training objective of a diffusion model evolves: early stages focus on denoising low-frequency global structures, while later stages refine high-frequency details. Given that different layers of an LLM capture distinct types of semantic information [12, 45], the optimal weighting of textual features for these two stages would likely differ. Consequently, allowing the layer weights to update continuously would cause them to drift with the evolving training objective, creating a non-stationary text condition that poses a significant challenge to the stable convergence of the diffusion model (e.g., DiT).

To mitigate this instability, we introduce a two-step training strategy. We first train the layer weights jointly with the diffusion model for an initial number of steps, allowing them to converge to a generally effective scheme. After that, we freeze these learned weights, thereby providing a stable, high-quality text representation for the remainder of the model’s training. We provide detailed discussion on the two-step training strategy in the Appendix B.

#### 4. Experiments

We evaluate our TED-6K benchmark across several mainstream text encoders. To validate the benchmark’s effectiveness—whether its scores can predict real-world performance in downstream generative tasks—we select a representative subset of these encoders and train corresponding T2I and T2V models for a formal correlation analysis. Both the TED-6K and T2I&T2V benchmarks shows that GRANTED has a superior text representation capability. The detailed experimental setup can be seen in Appendix C.

##### 4.1. TED-6K Results

- Table 1 presents the evaluation results of various text encoder types on our text-only benchmark. Fine-tuned on our curated dataset, GRAN-TED not only outperforms all peer models but also remarkably closes the gap with 32B Model.

A detailed analysis of the results in Table 1 reveals several clear and insightful trends:

• Potential of Decoder-only Architectures: The benchmark results on TED-6K unequivocally reveal that decoder-only LLMs exhibit significantly superior representational capabilities as text encoders compared to traditional encoder-only models. Notably, this performance gap does not stem solely from model size. For

instance, the 5B-parameter UMT5-XXL encoder still performs considerably worse than the smaller 4B Qwen3VL-Instruct.

- • Value of Multimodal Training: A key and noteworthy finding is that MLLMs consistently and significantly outperform their LLM backbones on our text-only benchmark, despite the evaluation itself being entirely devoid of visual inputs. We believe this advantage stems from the multimodal training process, which compels the text encoder to learn how to more effectively encode visuallygrounded concepts within its hidden states, thereby enabling the language model component to better leverage these representations for downstream tasks like VQA.
- • Ambiguous Effect of Instruction Tuning: In contrast to the clear trends observed in multimodal settings, instruction tuning does not exhibit a consistent impact on the representational capabilities on TED-6K.
- • Impact of Thinking model Interestingly, the thinking models exhibit a paradoxical effect: while it enhances the representational quality of single-layer features, it conversely degrades performance when multi-layer features are aggregated via the Norm-Avg strategy.
- • Effective Scaling Trends: The scaling law on the model size differs significantly depending on the feature extraction strategy. When relying solely on single-layer features, we observed no clear scaling trend. However, when using the Norm-Avg strategy, the model’s score on TED6K exhibits a much clearer positive correlation with its parameter size.
- • Importance of Feature Aggregation: In our comparison of single-layer features, we do not observe that the penultimate layer offers a consistent and universal performance advantage over the final layer. However, across all evaluated models, aggregating multi-layer features via the Norm-Avg strategy uniformly outperforms using features from any single layer on TED-6K. As different layers capture distinct granularities of information, from syntactic to semantic, aggregating features across them yields a more semantically complete and richer textual representation, leading to superior benchmark performance.

Table 2 presents a detailed performance breakdown of the models on the various sub-tasks of TED-6K. GRANTED outperforms the baseline model across all semantic dimensions except for Adjective and Adverb. Further analysis indicates that the model excels at capturing macro-level events (e.g., “Action”) but still has room for improvement in dimensions requiring fine-grained detail (e.g., “Spatial Relationship”, “OCR”), highlighting a clear direction for future work.

##### 4.2. Correlation with Generation Performance

To validate our text-only benchmark and investigate the consistency of its scores with downstream generative per-

Table 1. Performance comparison of different text encoders and feature extraction strategies on our text-only benchmark.

Model Type Model Name Last Layer Penultimate Layer Norm-Avg Encoder-only

UMT5-XXL [10] 44.60 46.89 51.41 AltCLIP [7] 47.15 46.59 51.86

Qwen3-8B-Instruct 53.62 53.00 55.77 Qwen3-8B-Base 53.37 54.19 55.94 Qwen3-4B-Instruct 53.64 53.58 55.25 Qwen3-4B-Base 52.84 53.06 54.98 Qwen3-4B-Thinking 52.90 54.10 54.86 Qwen3-32B-Instruct 53.23 54.42 56.98

Decoder-only LLM

Qwen3-VL-8B-Instruct 55.37 54.25 56.81 Qwen3-VL-8B-Thinking 55.67 55.40 56.51 Qwen3-VL-4B-Instruct 54.03 54.92 55.20 Qwen3-VL-32B-Instruct 54.60 55.91 57.24 MiMo-VL-7B [53] 54.54 56.29 56.48 Ovis-2.5-9B [32] 54.66 54.92 56.26

MLLM

GRAN-TED w/o weighting 55.87 55.13 57.22 GRAN-TED - - 57.42

- Table 2. Performance breakdown of Qwen3-VL versus GRANTED on the sub-tasks of TED-6K, using the Norm-Avg strategy for both models

Semantic Category Qwen3-VL GRAN-TED Quantity 52.01 52.87 (+0.86) Adjective (Adj) 46.91 45.74 (-1.17) Coreference 55.77 57.16 (+1.39) Basic Event 76.98 77.86 (+0.88) Adverb (Adv) 47.71 45.73 (-1.98) Spatial Relationship 46.32 47.11 (+0.79) OCR 35.98 37.12 (+1.14) Temporal Relationship 63.32 65.99 (+2.67) Action 68.87 71.19 (+2.32)

formance, we conduct a quantitative analysis by calculating the Pearson correlation coefficient between our benchmark scores and the evaluation results from both T2I and T2V tasks. The results for the T2I and T2V tasks are presented in Table 3 and Table 4, respectively. We observe that the scores from our text-only benchmark exhibit a statistically significant and strong positive correlation with the models’ performance metrics on downstream generative tasks.

##### 4.3. Ablation Study

To validate the effectiveness of the proposed method, we further select Qwen3-VL-8B-Instruct as the base model for our experiments. The results can be seen in Table 5. As shown in the table, our two-step training strategy proposed in Section 3.5 significantly improves performance over the vanilla Norm-Avg, yet it only introduces a small number of

- Table 3. Correlation between our benchmark scores and T2I generation performance. All evaluated encoders are instruction models, except for UMT5-XXL.

Text Encoder TED-6K GenAI

Umt5XXL(last-layer) 44.60 54.67 Qwen2.5-VL-7B (last-layer) 53.05 71.15 Qwen3-8B (last-layer) 53.62 71.15 InternVL3.5-8B-MPO (last-layer) 55.31 74.46 Qwen2.5-VL-7B(norm-avg) 55.99 75.51 Qwen3-VL-8B(norm-avg) 56.81 76.17

Pearson Correlation: r = 0.9914,p = 1.09e − 4

- Table 4. Correlation between our benchmark scores and T2V generation performance.

###### Text Encoder TED-6K GenAI

Qwen3-8B (last-layer) 53.62 65.13 Qwen3-VL-8B (last-layer) 55.37 70.59 InternVL3.5-8B-MPO (last-layer) 55.31 68.70 Qwen3-VL-8B (norm-avg) 56.81 77.94

Pearson Correlation: r = 0.9587,p = 0.04129

learnable parameters equal to the number of Transformer layers in the LLM.

Notably, when the two-step strategy is omitted and only learnable weights are introduced with continuous training, performance slightly degrades. This observation aligns with the argument in Section 3.5 that a non-stationary text condition can interfere with the training process. We visualize

the weights assigned to each layer at different training steps during the training progress. As illustrated in Figure 4, the evolution of the layer weights reveals two key phenomena.

First, the weights of different layers exhibit nonmonotonic dynamics. For instance, in the shallower regions of the model (e.g., layers 5 through 8), the weights for these layers collectively first increase and then decrease over the course of training. Second, the overall weight distribution displays persistent instability, failing to converge to an optimal solution. Even after 200k training steps, some deeper layers, such as the penultimate layer (-2), are still undergoing a substantial and consistent increase, while the weights of many other layers remain in flux. This observation provides a direct visual confirmation of our hypothesis that the text condition is indeed non-stationary under a continuous training paradigm, thereby underscoring the necessity of our second-stage weight-freezing strategy.

step-12000 step-36000 step-60000 step-84000

- 0.10

0.08

step-108000 step-132000 step-156000 step-180000 step-204000

SoftmaxWeight

0.06

0.04

0.02

0.00

0 5 10 15 20 25 30 35 Layer Index

Figure 4. Dynamics of the learnable layer weights over the course of continuous training (i.e., without the two-step strategy). The weight values shown are normalized via Softmax.

Table 5. Ablation study over Qwen3-VL-8B-Instruct model. “Learnable Weights”: trainable layer weights with continuous training.“Two-Step Training”: the two-step training strategy to the learnable weights.

Method GenAI-Bench

Norm-Avg (Fixed Weights) 76.17 + Learnable Weights 75.94 + Two-Step Training 77.01

Finally, we train diffusion models integrated with GRAN-TED, enhanced by both Norm-Avg and two-step layer-wise feature weighting. We then compare these models against a strong baseline trained on the Norm-Avg representations from the original Qwen3-VL-8B-Instruct. The results demonstrate that models conditioned on our full methodology achieve superior generation performance on both T2I and T2V tasks, as shown in Table 6.

Table 6. Downstream generation performance of GRAN-TED.

Method T2I T2V

Norm-Avg 76.17 77.94 GRAN-TED 77.41(+1.24) 80.33(+2.39)

##### 4.4. Further Analysis

Performance Discrimination from TED-6K. The models achieve relatively similar scores as shown in Tab. 1. The reason is that part of the caption-statement pairs are simple for a comprehensive assessment, resulting in a low representational challenge. However, on the more demanding downstream tasks (Tab. 3-4), which concentrate on difficult prompts, the performance gaps between models become significantly more pronounced. This validates that TED-6K is effectively discriminative.

Robustness of TED-6K. To ascertain the contextual dependency of the questions in TED-6K, we decouple the captions from their corresponding statements and make shuffle. The Qwen3VL-8B-Instruct model is then tasked with direct question-answering on this shuffled data. Across five experimental runs, the model achieves an accuracy between 27.05% and 28.63%. This performance is approximately at the level of random guessing (25% for a four-option task), which demonstrates that the questions in TED-6K are sufficiently challenging and require the correct contextual caption to be answered accurately.

Aggregator v.s. Single-vector Representation. A comparison between using a trained aggregator and applying mean pooling over the final layer’s outputs reveals divergent results across architectures. According to Tab. 7, while mean pooling improves representation quality for Encoder-only model, it still falls short of the best performance on TED6K and exhibits a trend inconsistent with downstream tasks. Conversely, for Decoder-only LLMs, this method markedly impairs representational power. This justifies the necessity of our approach, which involves training an aggregator for effective model validation.

Defectiveness of QA-style Assessment. We compare the proposed similarity-based method with a direct QA method. For the QA task, we prompt the Qwen2.5-72B model to first convert each sample’s statements into a question and then ask the text encoders to provide an open-ended answer. As detailed in Tab. 7, the model’s powerful reasoning lead to uniformly accurate answers, which diminishes the method’s ability to distinguish between test cases. This demonstrates the suitability of the similarity-based approach as a more discriminative evaluation metric.

Stability of TED-6K. To account for the variability from model training in our Aggregator approach, we train the aggregators for UMT5-XXL, Qwen3-8B-Instruct, and Qwen3VL-8B-Instruct five times each. The maximum

score variation across these runs is only 0.02 for any given model. This level of variance would not alter the assessed document order relations in the TED-6K benchmark, which confirms the stability of our evaluation methodology.

Table 7. Performance comparison of our aggregator-based method against mean-pooling approach and QA-style evaluations.

Model Aggregator Mean-pooling Q& A UMT5-XXL 44.65 50.64 97.81 Qwen3-8B-Instruct 53.62 39.18 97.90 Qwen3-VL-8B-Instruct 55.37 42.59 97.94

#### 5. Conclusion

In this work, we address the critical yet often overlooked role of the text encoder in T2I and T2V diffusion models. First, we establish a robust and efficient evaluation framework centered on our novel text-only benchmark, TED6K, for rapid and reliable encoder assessment. Second, we propose a two-stage paradigm to develop the GRANTED encoder, involving targeted finetuning for visualsemantic alignment and a layer-wise weighting approach. Experiments validate that the GRAN-TED encoder not only achieves state-of-the-art performance on the TED-6K benchmark but also translates these gains into demonstrable improvements in semantic accuracy and compositional coherence for both T2I and T2V models.

[Figure 40]

## GRAN-TED: Generating Robust, Aligned, and Nuanced Text Embedding

### for Diffusion Models Supplementary Material

#### A. Aggregator Architecture

We implement the sentence-level context aggregator Acontext as a lightweight attention-based pooling module. Given token-level hidden states from a frozen text encoder, Acontext produces a single sentence-level context embedding c′context that mimics the text conditioning consumed by a DiT.

Inputs. Let ctext ∈ RB×N×D denote a sequence of N token embeddings of dimension D for a batch of size B. The aggregator supports two kinds of inputs: (i) single-layer features ctext ∈ RB×N×D and (ii) stacked multi-layer features {hi}Li=1 with hi ∈ RB×N×D from L encoder layers.

Layer-wise fusion. When multi-layer representations are available, Acontext first fuses them into a single token sequence before applying attention. Each layer output is normalized with a parameter-free LayerNorm, yielding {h˜i}Li=1, and the normalized features are either (i) averaged across layers, or (ii) combined using a learnable softmaxweighted sum. Concretely,

 

1 L

L i=1

h˜i, (average),

ctext =

exp(ωi)

L i=1 αih˜i, αi =

, (learnable),



L j=1 exp(ωj)

(5)

where ω = {ωi}Li=1 are scalar layer weights (optionally trainable). For single-layer inputs this fusion step is skipped

and ctext is simply the encoder hidden states.

context token and attention stack. Given the fused sequence ctext ∈ RB×N×D, Acontext prepends a learnable context token ccontext ∈ R1×1×D (shared across the batch):

x0 = concat(ccontext, ctext) ∈ RB×(1+N)×D. (6)

This context token serves as a global context carrier. The resulting sequence is processed by a stack of Lblk preLayerNorm Transformer blocks, each consisting of multihead self-attention followed by a two-layer MLP with GELU activation and residual connections. A padding mask (when available) is applied in a bidirectional way: padded positions are ignored in attention, while all valid tokens and the context token can freely attend to each other. This twoblock self-attention stack implements the core of the aggregator architecture.

Output. After the attention stack, we obtain xenc ∈ RB×(1+N)×D. We apply a final LayerNorm and, if needed, a linear projection to match the target dimension Dout:

xproj = Proj(LayerNorm(xenc)) ∈ RB×(1+N)×D

. (7)

out

The aggregated context embedding is read out from the projected context token:

c′context = xproj[:,0,:] ∈ RB×D

. (8)

out

In summary, the aggregator instantiates the abstract mapping

c′context ← Acontext [ctext, ccontext] , (9)

where ctext denotes token-level text embeddings and ccontext is represented by the learnable context token.

#### B. Theoretical Justification for the Two-Step Training Strategy

We present a formal argument for our two-step training strategy, grounded in the principles of optimization dynamics and the spectral bias of neural networks. The argument proceeds in a clear logical chain: we first establish the asymmetric roles of the model parameters θ and the layer weights ω. We then introduce spectral bias as the mechanism driving an evolving optimization objective. This leads to the deduction of a perpetually moving target for ω, which results in a non-stationary conditioning distribution. Finally, we present the two-step strategy as a practical resolution to this optimization pathology.

##### B.1. Premise 1: Asymmetric Roles in Joint Optimization

Let θ be the parameters of the denoising network ϵθ, and ω be the learnable layer weights of our text fusion module fω that produces the context vector ccontext. While jointly optimized, their roles are fundamentally asymmetric. The goal is to minimize the loss L(θ,ω):

L(θ,ω)

min

θ,ω

0,ϵ,t ϵ − ϵθ(xt,t,ccontext) 2 , where ccontext = fω(text embeddings).

Ec

context∼Pccontext(·;ω) Ex

= min

θ,ω

(10) Here, Pc

(·;ω) is the distribution of the conditioning vector ccontext produced by fω.

context

- • The role of θ: The parameters θ define a function approximator. Their optimization answers the question: “Given

a fixed input distribution Pc

context

(·;ω), how should the function ϵθ change to best predict the noise?”

- • The role of ω: The parameters ω = {ωi} are distributional control parameters. Their optimization answers the question: “For the current function ϵθ, how should

the input distribution Pc

be shaped (via the layer-wise weights ωi in fω) to minimize the overall loss?”

context

This asymmetry is the foundational premise of our argument.

##### B.2. Premise 2: Evolving Objective via Spectral Bias

The training objective for θ is not static due to the well-established principle of spectral bias in deep neural networks: networks learn low-frequency functions more rapidly than high-frequency ones.

In the context of diffusion models, this manifests as a natural curriculum:

- 1. Initial training: The network ϵθ first learns to denoise low-frequency components of the signal. This corresponds to capturing the coarse, global structure of the image, which is the dominant source of error at large timesteps t (the low-SNR regime).
- 2. Later training: Once low-frequency components are learned, the residual error is concentrated in the highfrequency components. The optimization then shifts focus to learning these, which corresponds to refining local details and textures, the primary task at small timesteps t (the high-SNR regime). Therefore, the denoising task itself evolves, progressing

from a low-frequency to a high-frequency focus. This implies that the type of conditioning information required by θ also evolves continuously throughout training. In particular, the effective configuration of the layer weights ωi in fω that produce ccontext must adapt over time.

##### B.3. Deduction: A Continuously Drifting Optimum for the Layer Weights

Combining our two premises leads to a critical deduction. For any fixed state of the model θk, there exists an instantaneous optimal configuration ωk∗ that shapes the most effective conditioning distribution for that specific state:

ωk∗ = arg min

L(θk,ω). (11)

ω

From Premise 2, we know that the task for θk is continuously evolving due to spectral bias. From Premise 1, we know that the role of ω is to adapt the input distribution to best suit the current θk. It logically follows that the instantaneous optimum ωk∗ must also be in a state of perpetual drift as θk learns:

ωk∗ ̸= ωk∗+1. (12)

This creates a “moving target” problem, where the layer weights ωk are forced to perpetually chase an optimum that never settles.

##### B.4. Consequence: Non-Stationary Distribution and Instability

The perpetual chase of a drifting optimum ensures that the layer weights remain in constant flux (ωk+1 ̸= ωk). This directly implies that the conditioning distribution is nonstationary:

(·;ωk). (13)

ωk+1 ̸= ωk =⇒ Pc

(·;ωk+1) ̸= Pc

context

context

This non-stationarity is a form of covariate shift internal to the training loop, a known cause of optimization pathology. The gradient ∇θL is computed based on a “stale” distribution Pc

(·;ωk) that is immediately invalidated by the update to ωk+1. This mismatch between the environment used for gradient calculation and the one the updated model faces leads to inefficient and potentially unstable training.

context

##### B.5. Resolution: The Two-Step Strategy

The two-step strategy is designed to arrest the continuous drift of the conditioning distribution.

- 1. Step 1 (joint optimization). We perform joint optimization over (θ,ω) to find a robust ω∗: it represents an effective compromise across the continuum of objectives encountered during the initial, dynamic phase of training, as the spectral bias drives θ from low-frequency to high-frequency behavior.
- 2. Step 2 (freezing the layer weights). We then freeze the layer weights at this compromise point, i.e., set ω = ω∗ and keep it fixed for the remainder of training. This halts the evolution of ω and forces the conditioning distribution to become stationary:

(·;ω∗). (14) By stabilizing the conditioning distribution Pc

###### (·;ωk+1) = Pc

###### (·;ωk) = Pc

###### Pc

context

context

context

, we eliminate the internal covariate shift. The optimization problem for θ is transformed into a well-posed task on a fixed loss landscape, which facilitates more reliable and efficient convergence. In practice, this matches the empirical observation in the main paper: once the layer-wise fusion weights ωi have converged to a stable pattern, further updating them brings marginal benefit while introducing additional noise into the training dynamics of θ.

context

#### C. Experimental Details

##### C.1. Experimental Setup

• Context Aggegator Setup: For our text-only evaluation, the adapter module consists of a two-layer attention network with a hidden dimension of 1024. It is trained for

one epoch on our 500k-sample dataset with a learning rate of 1e-5.

- • Text-to-Image Training: For the text-to-image task, we employ the Lumina-Image-2.0-2B [40] architecture and train the models from scratch. For each text encoder under evaluation, we substitute it into the architecture while keeping all other components fixed. The models are trained at a 256x256 resolution on a private dataset, using a batch size of 512 for a total of 144k steps. In our two-step training strategy, we first train the layer weights ωi jointly with the main model for 96k steps, after which we freeze ωi for the remainder of the training.
- • Text-to-Video Training: In contrast, for the text-tovideo task, we fine-tune the pre-trained Wan2.1-T2V1.3B model [47]. For each text encoder, we replace the original one and re-initialize the text embedding layers within the DiT module. To accelerate semantic alignment, we employ an interleaved training strategy where the ratio of training steps for image data (batch size 512) to video data (batch size 128) is 7 : 3. The fine-tuning process consisted of a total of 20k training steps. In our two-step training strategy, we first train the layer weights ωi jointly with the main model for 12k steps, after which we freeze ωi for the remainder of the training.

In both the T2I and T2V training setups, the main backbone of the diffusion model remains fully trainable. The learning rate is set to 1e-4. For evaluation, we use GenAI-Bench [23]. During inference, all visual contents are generated using 50 denoising steps and the default negative prompt from GenAI-Bench. The CFG scale is set to 4.0 for the T2I task and 5.0 for the T2V task.

##### C.2. Training Data Example

This section presents concrete examples of the two key types of training data used in our experiments: (1) the selfsupervised contrastive learning data for training the Context Aggregator.and (2) the highly targeted VQA data for finetuning the GRAN-TED encoder.

###### C.2.1. Aggretator Training Data

Our aggregator is trained using a contrastive learning paradigm, with the data construction process illustrated in Figure 6. For any given source caption, which serves as the Anchor, we generate a positive sample by prompting the model to re-caption the same visual content (image/video) using a different prompt. This yields a semantically similar but textually distinct caption. Concurrently, captions derived from entirely different visual content are treated as negative samples.

###### C.2.2. GRAN-TED Training Data

Specific examples of the Visual Question Answering (VQA) data used for our fine-tuning, illustrating the videobased and image-based VQA pairs, are shown in Figure 5.

Video

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Q & A

Question: What is the style of the video? Elaborate on it.

Answer: The video is realistic, with a style resembling a movie or TV series, utilizing natural lighting to capture scenes as they would appear in real life, and featuring a natural and muted color palette, which collectively creates an effect of realism and authenticity.

Image

|[Figure 47]|
|---|

Q & A

Question: Describe this image through background. Answer: The background is a dense forest canopy with a continuous sea of green, stretching extensively without any signs of urban development. Sunlight highlights various shades of green, creating a mosaic effect. The serene atmosphere is enhanced by soft shadows, suggesting an early morning or late afternoon. With no notable landmarks or text, the focus remains on the natural environment.

Figure 5. Examples of visual question answering (VQA) training samples, covering both image-based and video-based settings.

##### C.3. Prompt

In this section, we provide the detailed prompts used throughout our experiments. Specifically, we include:

- • prompts for generating the base captions for images and videos (Figs. 7, 9);
- • prompts for constructing the positive and negative statements for the TED-6K benchmark (Figs. 11, 12, together with the statement type definitions in Fig. 13);
- • prompts for generating the positive pairs required for contrastive learning (Figs. 7, 8, 9, and 10);

###### Example Captions Used for Positive/Negative Statement Tasks Caption (from Image/Video 1).

In a serene outdoor setting with bamboo and dim lighting, a middle-aged Asian man in traditional attire sits at a low wooden table. Wearing a white robe, a light brown vest, and a white headband, he is surrounded by a dimly lit courtyard or garden with bamboo trees and greenery. The scene, set at night, features a small wooden structure with a lantern and candles on the table. The man rises from his seat, turning his head with a serious and alert expression, hinting at concern. The realistic scene, possibly from a TV series or movie, emphasizes character portrayal with dramatic lighting, mixing natural and artificial sources to create a moody atmosphere. The muted color palette with earthy tones suggests a historical setting, while the camera captures the moment with a medium close-up shot, panning to the upper left at a normal speed. The lower angle positioning and shallow depth of field blur the distant background, transitioning from a view of the man’s back to a profile.

###### Positive Caption (from Image/Video 1, different prompt).

A man of East Asian descent, likely in his 20s or 30s, with dark hair pulled back in a topknot and a white cloth draped over it, wears a light beige, loose-fitting robe with long sleeves and a white undergarment. He has a clean-shaven face, dark eyes, and an average build. In a serene outdoor setting at night, surrounded by bamboo and lanterns, he sits contemplatively before standing up with a serious expression. The scene, capturing a traditional or historical context, features soft, warm light from lanterns contrasting with dark greenery. Filmed with natural lighting, a muted color palette, and shallow depth of field, the medium close-up shot tilts upward, framing the man slightly to the right, evoking a somber, contemplative mood.

###### Negative Caption (from Image/Video 2).

A motorcyclist rides down a wide, empty road lined with trees and greenery on a sunny day. The motorcyclist, dressed in dark clothing, is positioned in the center of the frame, moving away from the camera. The road features clear lane lines, sidewalks, and is flanked by greenery and trees. The motorcycle is not clearly visible due to the distance. The background includes a grassy hillside on the right, with streetlights and trees along the road. The sky is clear and blue, creating a realistic, documentary-style scene with a natural, unedited feel. The motorcyclist continues straight, gradually becoming smaller in the frame before disappearing from view. The scene is characterized by medium saturation, moderate contrast, and brightness, with a normal color temperature. Captured with a deep depth of field, the camera pans left, maintaining a height roughly equal to the motorcyclist’s line of sight, framing the subject slightly to the left of center.

Figure 6. An example of the contrastive learning data used to train the context aggregator. For a given caption, a Positive Sample is generated by re-captioning the same visual content (image/video) with a different prompt. Captions from entirely different visual content are used as Negative Samples.

Your task is to generate a single, highly detailed (fine-grained), and coherent caption for the given image. Core Principles

- 1. Fidelity: Your caption must be loyal to the visual facts of the image. Do not invent details, assumptions, or information that is not clearly visible.
- 2. Coherence: Your final output must be a single, flowing paragraph. The description should be seamless and natural, prioritizing readability and flow.

Path to a Detailed Caption Start by identifying the “big picture” (the main event or subject). Then, mentally review the following dimensions to see if you can add more detail naturally, without disrupting the narrative flow:

- • The Scene (Basic Event): What is the overall setting or event?
- • The “Doing” (Actions): What are the key subjects (people, animals) doing?
- • The “Placing” (Spatial Relationships): Where are things located in relation to each other?
- • The “Attributes” (Adjectives, Adverbs & Quantity):

- – What are the qualities of subjects/objects? (e.g., color, texture, shape)
- – How are actions being performed? (e.g., walking slowly)
- – What is the degree of a quality? (e.g., very tall, incredibly bright)
- – How many are there?

- • The “Clarity” (Coreference): Clearly link actions and descriptions to their correct subjects, ensuring there is no ambiguity about who is doing what.
- • The “Text” (OCR): Is there any clearly readable text (on signs, shirts)? Important. This list is a tool for consideration, not a checklist to be completed. You do not need to force every one of these points into your caption. The final, synthesized caption is all that matters. It must be loyal, fine-grained, and—above all—coherent.

Figure 7. Prompt used for Image Caption 1 generation.

Your task is to generate one high-quality, fine-grained caption for the provided image. Core Objectives

- 1. Fidelity (Be Factual): Describe only what is visually present in the image. Do not invent or assume details, emotions, or relationships that are not explicitly shown.
- 2. Coherence (Be Natural): The caption must be a single, well-written paragraph. It should be fluid and easy to read, not just a list of facts.

A Guide to Rich Detail To ensure your caption is thorough, consider the following elements. This is a guide for inspiration, not a strict checklist; your primary goal is a natural, coherent narrative. Only include these details if they fit seamlessly.

- • 1. The Scene (Basic Event): What is the setting? (e.g., a park, a kitchen, a concert)
- • 2. Key Actions: What are the subjects doing?
- • 3. Visual Properties (Adjectives, Adverbs, Quantity):

- – What do things look like? (e.g., red, wooden, small)
- – How are actions/qualities modified? (e.g., running quickly, very bright)
- – How many key items are there?

- • 4. Layout (Spatial Relationships): Where are things located in relation to each other? (e.g., on the table, next to the window)
- • 5. Clarity (Coreference): Ensure it is obvious which description belongs to which subject.
- • 6. Visible Text (OCR): Is there any text on signs, clothing, or objects that is clear enough to read? Your final output is the caption itself. It must be a single, coherent paragraph that is both detailed and strictly faithful to the image.

Figure 8. Prompt used for Image Caption 2 generation.

- Prompt Used for Video Caption 1 Generation Your task is to generate a single, highly detailed (fine-grained), and coherent caption for the given video.

###### Core Principles

- 1. Fidelity: Your caption must be loyal to the visual facts of the entire video clip. Do not invent details, assumptions, or information that is not clearly visible.
- 2. Coherence: Your final output must be a single, flowing paragraph. The description should be seamless and natural,

prioritizing readability and flow. Path to a Detailed Caption Start by identifying the big picture (the main event or subject). Then, mentally review the following dimensions to see if you can add more detail naturally, without disrupting the narrative flow:

- • The Scene (Basic Event): What is the overall setting or event?
- • The “Doing” (Actions): What are the key subjects (people, animals) doing over time? (e.g., running, talking, opening)
- • The “Timing” (Temporal Relationship): How do events unfold or relate in time?
- • The “Placing” (Spatial Relationships): Where are things located in relation to each other, and do these relationships change?
- • The “Attributes” (Adjectives, Adverbs & Quantity):

- – What are the qualities of subjects/objects? (e.g., color, texture, shape)
- – How are actions being performed? (e.g., walking slowly)
- – What is the degree of a quality? (e.g., very tall)
- – How many are there?

- • The “Clarity” (Coreference): Clearly link actions and descriptions to their correct subjects.
- • The “Text” (OCR): Is there any clearly readable text (on signs, shirts) that appears long enough to be read? This list is a tool for consideration, not a checklist to be completed. The final, synthesized caption must be loyal, fine-grained, and coherent.

Figure 9. Prompt used for Video Caption 1 generation.

- Prompt Used for Video Caption 2 Generation Your task is to generate one high-quality, fine-grained caption for the provided video.

###### Core Objectives

- 1. Fidelity (Be Factual): Describe only what is visually present across the duration of the video. Do not invent or assume details, emotions, or relationships that are not explicitly shown.
- 2. Coherence (Be Natural): The caption must be a single, well-written paragraph. It should be fluid and easy to

read, not just a list of facts. A Guide to Rich Detail To ensure your caption is thorough, consider the following elements. This is a guide for inspiration, not a strict checklist; only include these details if they fit seamlessly.

- • 1. The Scene (Basic Event): What is the setting? (e.g., a park, a kitchen, a concert)
- • 2. Key Actions: What are the subjects doing? (e.g., running, talking, changing expression)
- • 3. Event Flow (Temporal Relationship): What is the sequence or timing of events?
- • 4. Visual Properties (Adjectives, Adverbs, Quantity):

- – What do things look like? (e.g., red, wooden, small)
- – How are actions/qualities modified? (e.g., running quickly, very bright)
- – How many key items are there?

- • 5. Layout (Spatial Relationships): Where are things located in relation to each other?
- • 6. Clarity (Coreference): Ensure it is obvious which description belongs to which subject.
- • 7. Visible Text (OCR): Is there any text on signs, clothing, or objects that is clear enough to read? Your final output is the caption itself. It must be a single, coherent paragraph that is both detailed and strictly faithful to the video.

Figure 10. Prompt used for Video Caption 2 generation.

###### Prompt Used for Positive Statement Generation

You are provided with a long, detailed caption and a “Statement Type”. Your task is to generate a clear, coherent statement that addresses the specific Statement Type, based only on the information given in the caption.

###### Instructions

- 1. Read the specified Statement Type (e.g., Basic Event, Action, Quantity, Spatial Relationship).
- 2. Carefully read the long, detailed caption to find the specific information that corresponds to this Statement Type.
- 3. If the information is clearly present, generate a declarative statement that only describes that Statement Type using the caption’s information, without adding irrelevant details.
- 4. If the caption does not contain explicit information for this perspective, the "statement" value in the JSON output must be null.

Important notes: generate a statement only if the information is clearly present; do not make assumptions. The Statement Type dictates the perspective. Use null only when there is a mismatch between the caption’s content and the requested Statement Type.

Input Statement Type: {statement type placeholder} Caption: {caption placeholder}

###### Output Format

{

"statement": "Your generated statement here.", "reasoning": "Brief explanation of how this statement was generated

from the specified ’Statement Type’ perspective, OR why it could not be."

}

Figure 11. Prompt used for positive statement generation.

###### Prompt Used for Negative Statement Generation

You are provided with a long, detailed caption, a “Statement Type”, and a “Positive Statement” that is factually true according to the caption. Your task is to generate exactly three Negative Statements that are plausible but factually incorrect according to the caption.

###### Inputs

- • Caption: {caption placeholder}

- • Statement Type: {statement type placeholder}

- • Positive Statement: {positive statement placeholder} Requirements for Negative Statements

- 1. Maintain Type: Each negative statement must be of the same Statement Type as the positive.
- 2. Contradict Caption: Each negative statement must be demonstrably false based on the information in the caption.
- 3. Be Plausible & Confusing: Negatives should be minimally different from the positive or be plausible alternatives that could have been true but are not, making them hard to distinguish from the positive statement without carefully reading the caption.

Output Format Please output only a JSON object containing exactly three strings:

{

"negative_statements": [ "Your first negative statement here.", "Your second negative statement here.", "Your third negative statement here."

] }

Figure 12. Prompt used for negative statement generation.

###### Statement Types

- 1. Basic Event Statement Description: A high-level summary of the main activity, occurrence, or overall situation depicted in the content. It captures the core event without going into fine-grained details. Examples: “The content shows a family having a picnic in the park.”; “A person is giving a presentation in a conference room.”; “Two cats are playing in a living room.”
- 2. Action Recognition Statement Description: Identifies and describes a specific, concrete action performed by a subject (person or animal). More granular than a Basic Event statement, focusing on a single action rather than the entire scene. Examples: “A woman is chopping vegetables.”; “The cat is stretching.”; “A man opened a book.”
- 3. Temporal Statement Description: Describes the temporal order or sequence of events within the content, using clear temporal connectors (e.g., “first”, “then”, “after”, “while”). Examples: “First, the man opened the door, then he walked inside.”; “The audience started clapping after the woman finished her speech.”; “The dog was barking while the car was driving by.”
- 4. Spatial Relationship Statement Description: Describes the spatial position and layout relationships between two or more objects or entities. Can be static (fixed positions) or dynamic (positions anchored to a specific time or event). Examples: “The lamp is on the desk.”; “The cat was under the chair when the man entered the room.”; “The ball is to the left of the goal after it was kicked.”
- 5. Quantity Statement Description: Explicitly identifies the exact number or count of specific objects, people, or entities. Examples: “There are three people standing on the sidewalk.”; “The content shows two red cars.”; “One person is wearing a hat.”
- 6. Adjective Statement Description: Describes a characteristic, quality, or state of being of a noun (object, person, etc.). The focus is the adjective itself. Examples: “The car is red.”; “A tall woman is visible.”; “The cat is fluffy.”
- 7. Adverb Statement Description: Describes the manner, place, time, frequency, or degree of an action or quality, with the adverb as the core. Examples: “The man is walking slowly.”; “The building is very tall.”; “The light is flashing repeatedly.”
- 8. Coreference Statement Description: Resolves an ambiguous reference by explicitly connecting it to its antecedent (e.g., linking a pronoun or descriptive noun phrase to the entity it refers to). Examples: “The pronoun ’it’ in the phrase ’...a dog. It is barking.’ refers to the dog.”; “The phrase ’the main character’ refers to the young white woman wearing a blue tank top.”
- 9. OCR Statement Description: Transcribes clearly visible text content (OCR) within the content, directly quoting the text. Examples: “The text on the sign reads ’STOP’.”; “The person’s shirt has the word ’University’ on it.”; “A bus with the destination ’Downtown’ is visible.”

Figure 13. Definitions of the statement types used in our positive and negative statement generation tasks.

#### References

- [1] AIME. Aime problems and solutions. https:// artofproblemsolving.com/wiki/index.php/ AIME_Problems_and_Solutions, 2025. Art of Problem Solving Wiki. Accessed: 2025-11-12. 3
- [2] Ruichuan An, Sihan Yang, Ming Lu, Renrui Zhang, Kai Zeng, Yulin Luo, Jiajun Cao, Hao Liang, Ying Chen, Qi She, et al. Mc-llava: Multi-concept personalized vision-language model. arXiv preprint arXiv:2411.11706, 2024. 2
- [3] Ruichuan An, Sihan Yang, Renrui Zhang, Zijun Shen, Ming Lu, Gaole Dai, Hao Liang, Ziyu Guo, Shilin Yan, Yulin Luo, et al. Unictokens: Boosting personalized understanding and generation via unified concept tokens. arXiv preprint arXiv:2505.14671, 2025. 1
- [4] Tianyi Bai, Hao Liang, Binwang Wan, Yanran Xu, Xi Li, Shiyu Li, Ling Yang, Bozhou Li, Yifan Wang, Bin Cui, et al. A survey of multimodal large language model from a datacentric perspective. arXiv preprint arXiv:2405.16640, 2024. 5
- [5] Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. Llm2vec: Large language models are secretly powerful text encoders. arXiv preprint arXiv:2404.05961, 2024. 2
- [6] Chaofeng Chen, Annan Wang, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, and Weisi Lin. Enhancing diffusion models with text-encoder reinforcement learning. In European Conference on Computer Vision, pages 182–198. Springer, 2024. 2
- [7] Zhongzhi Chen, Guang Liu, Bo-Wen Zhang, Qinghong Yang, and Ledell Wu. Altclip: Altering the language encoder in clip for extended language capabilities. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8666–8682, 2023. 7
- [8] Kanzhi Cheng, Wenpo Song, Jiaxin Fan, Zheng Ma, Qiushi Sun, Fangzhi Xu, Chenyang Yan, Nuo Chen, Jianbing Zhang, and Jiajun Chen. Caparena: Benchmarking and analyzing detailed image captioning in the llm era. arXiv preprint arXiv:2503.12329, 2025. 3
- [9] Zixu Cheng, Jian Hu, Ziquan Liu, Chenyang Si, Wei Li, and Shaogang Gong. V-star: Benchmarking videollms on video spatio-temporal reasoning. arXiv preprint arXiv:2503.11495, 2025. 3
- [10] Hyung Won Chung, Noah Constant, Xavier Garcia, Adam Roberts, Yi Tay, Sharan Narang, and Orhan Firat. Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining. arXiv preprint arXiv:2304.09151,

2023. 7

- [11] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 3
- [12] Guy Dar, Mor Geva, Ankit Gupta, and Jonathan Berant. Analyzing transformers in embedding space. arXiv preprint arXiv:2209.02535, 2022. 6

- [13] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113,

2025. 1, 3

- [14] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025. 1, 2, 3
- [15] Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, and Pheng-Ann Heng. Are video models ready as zero-shot reasoners? an empirical study with the mme-cof benchmark. arXiv preprint arXiv:2510.26802, 2025. 3
- [16] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020. 3
- [17] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec: Training vision-language models for massive multimodal embedding tasks. arXiv preprint arXiv:2410.05160, 2024. 2
- [18] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions, 2024. 4
- [19] Jeonghoon Kim, Byeongchan Lee, Cheonbok Park, Yeontaek Oh, Beomjun Kim, Taehwan Yoo, Seongjin Shin, Dongyoon Han, Jinwoo Shin, and Kang Min Yoo. Peri-ln: Revisiting normalization layer in the transformer architecture. arXiv preprint arXiv:2502.02732, 2025. 5
- [20] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 1, 3
- [21] Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. Viescore: Towards explainable metrics for conditional image synthesis evaluation. arXiv preprint arXiv:2312.14867, 2023. 2, 3
- [22] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space,

2025. 1

- [23] Baiqi Li, Zhiqiu Lin, Deepak Pathak, Jiayao Li, Yixin Fei, Kewen Wu, Tiffany Ling, Xide Xia, Pengchuan Zhang, Graham Neubig, et al. Genai-bench: Evaluating and improving compositional text-to-visual generation. arXiv preprint arXiv:2406.13743, 2024. 12
- [24] Bozhou Li, Xinda Xue, Sihan Yang, Yang Shi, Xinlong Chen, Yushuo Guan, Yuanxing Zhang, and Wentao Zhang. The unseen bias: How norm discrepancy in pre-norm

- mllms leads to visual information loss. arXiv preprint arXiv:2512.08374, 2025. 5
- [25] Pengzhi Li, Pengfei Yu, Zide Liu, Wei He, Xuhao Pan, Xudong Rao, Tao Wei, and Wei Chen. Ldgen: Enhancing text-to-image synthesis via large language model-driven language representation. arXiv preprint arXiv:2502.18302,

2025. 2

- [26] Xiaotong Li, Fan Zhang, Haiwen Diao, Yueze Wang, Xinlong Wang, and Lingyu Duan. Densefusion-1m: Merging vision experts for comprehensive multimodal perception. Advances in Neural Information Processing Systems, 37:18535–18556, 2024. 4
- [27] Yanyu Li, Xian Liu, Anil Kag, Ju Hu, Yerlan Idelbayev, Dhritiman Sagar, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Textcraftor: Your text encoder can be image quality controller. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7985– 7995, 2024. 2
- [28] Weifeng Lin, Xinyu Wei, Ruichuan An, Peng Gao, Bocheng Zou, Yulin Luo, Siyuan Huang, Shanghang Zhang, and Hongsheng Li. Draw-and-understand: Leveraging visual prompts to enable mllms to comprehend what you want. arXiv preprint arXiv:2403.20271, 2024. 2
- [29] Weifeng Lin, Xinyu Wei, Ruichuan An, Tianhe Ren, Tingwei Chen, Renrui Zhang, Ziyu Guo, Wentao Zhang, Lei Zhang, and Hongsheng Li. Perceive anything: Recognize, explain, caption, and segment anything in images and videos, 2025. 2
- [30] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision, pages 366–384. Springer, 2024. 2, 3
- [31] Zhihang Liu, Chen-Wei Xie, Bin Wen, Feiwu Yu, Jixuan Chen, Boqiang Zhang, Nianzu Yang, Pandeng Li, Yinglu Li, Zuan Gao, et al. What is a good caption? a comprehensive visual caption benchmark for evaluating both correctness and thoroughness. arXiv preprint arXiv:2502.14914, 2025. 3
- [32] Shiyin Lu, Yang Li, Yu Xia, Yuwei Hu, Shanshan Zhao, Yanqing Ma, Zhichao Wei, Yinglun Li, Lunhao Duan, Jianshan Zhao, et al. Ovis2. 5 technical report. arXiv preprint arXiv:2508.11737, 2025. 7
- [33] Yulin Luo, Ruichuan An, Bocheng Zou, Yiming Tang, Jiaming Liu, and Shanghang Zhang. Llm as dataset analyst: Subpopulation structure discovery with large language model. In European Conference on Computer Vision, pages 235–252. Springer, 2024. 2
- [34] Bingqi Ma, Zhuofan Zong, Guanglu Song, Hongsheng Li, and Yu Liu. Exploring the role of large language models in prompt encoding for diffusion models. arXiv preprint arXiv:2406.11831, 2024. 2
- [35] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025. 1, 3
- [36] Niklas Muennighoff, Nouamane Tazi, Lo¨ıc Magne, and Nils

- Reimers. Mteb: Massive text embedding benchmark. arXiv preprint arXiv:2210.07316, 2022. 2
- [37] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 4
- [38] OpenAI. Gpt-4v(ision) system card, 2024. 2
- [39] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2
- [40] Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Jiakang Yuan, Xinyue Li, Dongyang Liu, et al. Luminaimage 2.0: A unified and efficient image generative framework. arXiv preprint arXiv:2503.21758, 2025. 1, 3, 12
- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 2
- [42] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 2
- [43] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021. 1
- [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2
- [45] Oscar Skean, Md Rifat Arefin, Yann LeCun, and Ravid Shwartz-Ziv. Does representation matter? exploring intermediate layers in large language models. arXiv preprint arXiv:2412.09563, 2024. 6
- [46] Qwen Team. Qwen3-vl: Sharper vision, deeper thought, broader action. https://qwen.ai/blog?id= 99f0335c4ad9ff6153e517418d48535ab6d8afef,

2025. Accessed: 2025-09-23. 2, 3

- [47] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 2, 3, 12
- [48] Andrew Z Wang, Songwei Ge, Tero Karras, Ming-Yu Liu, and Yogesh Balaji. A comprehensive study of decoder-only llms for text-to-image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28575–28585, 2025. 1, 2, 5
- [49] Jiawei Wang, Liping Yuan, Yuchen Zhang, and Haomiao Sun. Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634, 2024. 3

- [50] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8428–8437, 2025. 4
- [51] Wenhao Wang and Yi Yang. Videoufo: A million-scale userfocused dataset for text-to-video generation. arXiv preprint arXiv:2503.01739, 2025. 4
- [52] Thadd¨aus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 1
- [53] LCT Xiaomi and Core Team. Mimo-vl technical report. arXiv preprint arXiv:2506.03569, 2025. 7
- [54] Long Xing, Qidong Huang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Jinsong Li, Shuangrui Ding, Weiming Zhang, Nenghai Yu, et al. Scalecap: Inferencetime scalable image captioning via dual-modality debiasing. arXiv preprint arXiv:2506.19848, 2025. 4
- [55] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 2
- [56] Xinlu Zhang, Yujie Lu, Weizhi Wang, An Yan, Jun Yan, Lianke Qin, Heng Wang, Xifeng Yan, William Yang Wang, and Linda Ruth Petzold. Gpt-4v (ision) as a generalist evaluator for vision-language tasks. arXiv preprint arXiv:2311.01361, 2023. 2

