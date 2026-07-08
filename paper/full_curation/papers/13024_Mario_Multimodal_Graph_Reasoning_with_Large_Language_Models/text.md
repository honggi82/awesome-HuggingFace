# arXiv:2603.05181v2[cs.CV]26Mar2026

Image-only Text+Image

## Mario: Multimodal Graph Reasoning with Large Language Models

Exclusive

###### 2.25%

Overlapping

6.98% 7.71%

Yuanfu Sun1,2*, Kang Li3*, Pengkang Guo4, Jiajin Liu1,2, Qiaoyu Tan1† 1New York University Shanghai, 2New York University, 3Tsinghua University, 4EPFL

Average Modality Preference

###### 2.65%

7.40%

Text-only

Image-only

union

{yuanfu.sun, qiaoyu.tan}@nyu.edu, lik24@mails.tsinghua.edu.cn

Union(∪)

### Abstract

[Figure 1]

[Figure 2]

Text+Image 2.25%

Recent advances in large language models (LLMs) have opened new avenues for multimodal reasoning. Yet, most existing methods still rely on pretrained vision–language models (VLMs) to encode image–text pairs in isolation, ignoring the relational structure that real-world multimodal data naturally form. This motivates reasoning on multimodal graphs (MMGs), where each node has textual and visual attributes and edges provide structural cues. Enabling LLM-based reasoning on such heterogeneous multimodal signals while preserving graph topology introduces two key challenges: resolving weak cross-modal consistency and handling heterogeneous modality preference. To address this, we propose Mario, a unified framework that simultaneously resolves the two above challenges and enables effective LLM-based reasoning over MMGs. Mario consists of two innovative stages. Firstly, a graph-conditioned VLM design that jointly refines textual and visual features through fine-grained cross-modal contrastive learning guided by graph topology. Secondly, a modality-adaptive graph instruction tuning mechanism that organizes aligned multimodal features into graph-aware instruction views and employs a learnable router to surface, for each node and its neighborhood, the most informative modality configuration to the LLM. Extensive experiments across diverse MMG benchmarks demonstrate that Mario consistently outperforms state-of-the-art graph models in both supervised and zero-shot scenarios for node classification and link prediction. The code will be made available at Mario.

6.98% 7.71%

70.96%

Text-Only 2.65% 2.05% Image-Only

7.40%

(a) Modality Consistency Bar Chart (b) Modality Preference Venn Diagram

Figure 1. (a) Cosine similarity between text and image embeddings across three models on four datasets. (b) Venn diagram over three prompt templates with different modality inputs: Text-Only, Image-Only, and Text+Image. Each colored circle corresponds to one template. Among nodes correctly classified by at least one template, each region shows the proportion exclusively covered by the corresponding template combination (where overlapping regions blend the colors). Results are averaged over four datasets.

as isolated image–text pairs. This view sits uneasily with how multimodal data actually appear in the wild. Rather than isolated image–text pairs, these entities are interlinked and are more faithfully modeled as structured collections of multimodal nodes [39]. Treating such data as independent, unrelated pairs leaves a substantial portion of the multimodal signal unused, in particular the information carried by the relations among multimodal entities.

This mismatch has prompted a recent line of work to make the relational nature of multimodal data explicit by casting them as multimodal graphs (MMGs) [25, 39, 46]: each node carries textual and visual attributes while edges supply additional structural priors. A seemingly natural recipe [25] then suggests itself: reuse powerful vision–language models (VLMs) (e.g., CLIP [32]) to encode the node-level modalities, and hand these embeddings to a graph-capable backbone like GNNs [18, 22, 29, 38] for propagation and reasoning. This strategy is appealing because it preserves the representational strength of pretrained VLMs and delegates only the structural part to the graph model. Nonetheless, it quietly rests on a strong premise: that the textual and visual views of each node are already semantically well synchronized, and that injecting them into the graph will not magnify any cross-modal mismatch.

### 1. Introduction

Large language models (LLMs) [10, 23, 40] have evolved from pure text processors into general-purpose multimodal reasoners that can follow instructions, ground entities, and integrate visual signals [31]. Yet most current pipelines [25, 30, 37, 42] continue to assume and operate on a simplified input form, in which multimodal data are handled

*Equal contribution †Corresponding author

However, in realistic MMGs the image attached to a

node is not always a clean visual rendering of the text, and the text is not always a faithful caption of the image; in isolation, either side can be short, noisy, or semantically underspecified. This gives rise to our first challenge in multimodal graph reasoning: C1: Weak cross-modal consistency. In such cases, neighboring nodes often provide critical cues that disambiguate modality semantics or reinforce missing information, especially when modalities only partially overlap. As shown in Fig. 1’s bar chart, by incorporating graph topology into the alignment process, our model Mario achieves significant cross-modal consistency, yielding a 68% average gain over frozen CLIP and an additional 6% improvement over node-wise fine-tuning. This empirically supports our claim that the design of a structureaware vision–language model is a necessary prerequisite for LLMs to perform reliable reasoning on multimodal graphs.

When such aligned multimodal features are in place, a second difficulty surfaces—C2: heterogeneous modality preference. In conventional LLM-based Graph Model (GraphLLM) settings [27, 34], where node inputs are unimodal, it is reasonable to employ a shared instruction template for all nodes. However, this assumption fails in MMGs, where the informativeness of each modality can vary significantly across nodes, and the above alignment mechanisms largely focus on common cross-modal patterns, whereas the aligned visual and textual features still retain their individual parts [43]. Some nodes are richly described and thus text-salient, others have noisy text and must rely on distinctive visual cues, and many actually require complementary evidence from both modalities. Moreover, when reasoning over local subgraphs, the “effective” modality for the anchor node can be perturbed by its neighbors’ noisy, incomplete, or redundant modalities. As the Venn diagram in Fig. 1 shows, among nodes correctly classified by at least one template, nearly 30% are covered by only one or two of the three modality-specific templates rather than all three. This suggests that a one-size-fitsall prompting strategy underutilizes available multimodal supervision, highlighting the need for adaptive and nodespecific prompting strategies. This leads us to ask:

(Q) Can we exploit graph structure to enforce reliable cross-modal alignment, and, on top of that, use the aligned representations to drive a dynamic prompting policy that adaptively chooses the most informative modality for each node and its local context?

To answer this question, we introduce Mario—a dual-stage Modality-Adaptive Reasoning over multimodal graphs framework that tightly couples structure-aware cross-modal alignment with instruction-tuned LLMs. Stage 1 serves as a graph-conditioned vision–language model: it employs a dual-tower encoder augmented with a topologyaware multimodal mixer inspired by GNN-nested designs

[41] to align text and image features under multi-hop structural guidance, yielding structure-aware, cross-modally coherent node representations. Stage 2 then organizes these aligned features into a family of multimodal instruction views and jointly trains a Modality-Adaptive Prompt Router with the LLM, so that for each node Mario can use it to decide which modality configuration to surface to the LLM and route the instance to the most predictive view during inference. Our key contributions are summarized as follows:

- • We undertake the study of employing LLM reasoning on MMGs, identifying two previously underexplored obstacles, cross-modal inconsistency and heterogeneous modality preference, and introduce a novel frameworkMario that simultaneously addresses both challenges.
- • We introduce a graph-conditioned vision–language model, a new VLM paradigm that aligns image and text under topology to yield symmetric, structure-aware node representations jointly grounded in both modalities.
- • We break the prevailing reliance of GraphLLMs on a fixed-modality template by introducing modalityadaptive graph instruction tuning, a new tuning scheme which routes each node to the most informative modality.
- • We conduct extensive evaluations across diverse MMG benchmarks, demonstrating Mario’s state-of-the-art performance in node classification and link prediction. Notably, Mario consistently outperforms leading baselines, achieving up to 1.6× gains in zero-shot transfer settings.

### 2. Related Work

Large Language Models for Graph Reasoning. Studies [6, 9, 11, 12, 45] have demonstrated the potential of LLMs in augmenting graph representations and generating contextualized textual information, thereby enhancing semantic expressiveness and improving downstream graph learning performance. Another line of work utilizes graph or language models to obtain graph tokens, which are then embedded into prompt templates for instruction tuning [5, 27, 34, 44] or directly provide LLMs with more text information related to the graph structure through in-context learning (ICL) to directly infer without training [16, 33]. When applying graph instruction tuning, it enables LLMs to understand these structure-informed tokens and perform appropriate reasoning, leading to better generalization in certain graph-related tasks. However, most of these studies operate only on textual modality (text-attributed graphs) and generalize poorly to multimodal graph reasoning. Moreover, their instruction tuning typically relies on a single template with fixed modality inputs, overlooking that different nodes may favor different modality information.

Multimodal Graph Learning. Multimodal graph learning extends conventional graph representation learning by integrating multiple modalities, often text and images, to enhance node/edge representations. While recent multimodal

Given a target node..., the image feature is <GI>. It has the followfollowing following neighbors at hop 1: N1: <G >, <label >It has the following neighbors at hop 2, .....Based on the information provided, please......

ResponseResponseA B

Response C

LargeLanguageModel

 

Given a target node..., the text feature is <GT>. It has the following neighbors at hop 1: N1: <GT >, <label >It has the following neighbors at hop 2, .....Based on the information provided, please......

풎풎

Given a target node..., the text feature is <GT> and the image features is <GI>.... It has the following neighbors at hop 1: N1: <GT >, <G >, <label >It has the following neighbors at hop 2, .....Based on the information provided, please......

softmax

###### Posterior

Stage2 Loss

low high

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Silvercell Turtles Brooch Pins Light Blue Austrian Rhinestone Crystal...

[Figure 17]

풊 

New Charming Jewelry Women's Vintage Noble Dragonfly Crystal...

###### Structure-aware Image-Text Alignment TextImage[CLS][CLS]EmbeddingEmbedding

Layer N

[Figure 18]

.

[Figure 19]

[Figure 20]

Anchor

Add & Norm

EVER FAITH Women's Austrian Crystal Enamel Colorful Bouquet ...

[Figure 21]

Add & Norm

Layer 1

[Figure 22]

###### ...

Add & Norm

...

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Add & Norm

[Figure 33]

Layer N

[Figure 34]

...

- 1-hop
- 2-hop

MLP

.

MLP

…

…

…

[Figure 35]

Add & Norm

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Add & Norm

Image Encoder

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

MLP

Layer 1

MLP

###### ...

[Figure 56]

...

Add & Norm

InfoNCELoss

θ

Add & Norm

θ

Add & Norm

Add & Norm

...

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

MLP

###### Mixer

MLP

θ

θ

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Multi-head Attention

Silvercell Turtles Brooch Pins Light l...

Add & Norm

Add & Norm

positive

ç

Reinjection

Reinjection

MLP

MLP

Multi-head Attention

Multi-head Attention

Multi-head Attention

    

θ

θ

Add & Norm

[Figure 79]

Add & Norm

Multi-head Attention

[Figure 80]

    

Multi-head Attention

[Figure 81]

Anchor

θ

풊풎    negative

θ

 , 

Multi-head Attention

Add & Norm

Add & Norm

⊕

ç

⊕

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Multi-head Attention

MHA with Graph Position Bias

Multi-head Attention

[Figure 92]

[Figure 93]

⊕

[Figure 94]

- 1-hop
- 2-hop

⊕

New Charming Jewelry

[Figure 95]

Multi-head Attention (MHA)

Multi-head Attention (MHA)

Text Encoder

= [ ]

= [ ]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

...

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

⊕

[Figure 116]

⊕

= [ ]

= [ ]

... Women's...

... Multimodal Graph

[Figure 117]

[Figure 118]

,  ⊕

⊕

Aggregate Aggregate

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

, 

= [ ]

…

…

…

= [ ]

 , 

,  = ,  [ ] , 

,  = ,  [ ] , 

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

(a)

[Figure 143]

Text

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

###### ...

###### Anchor A Anchor A

[Figure 151]

[Figure 152]

Structure-aware Aligned

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Anchor

softmax = [0.7, 0.2, 0.1]

softmax = [0.7, 0.2, 0.1]

Step 1

Modality-Adaptive Prompt Router

Step N

Image

Given a target node... It has the following neighbors at hop 1: N1: <GT >, <G >... and it has the following neighbors at...

Given a target node... It has the following neighbors at hop 1: N1: <GT >, <G >... and it has the following neighbors at...

Given a target node... It has the following neighbors at hop 1: N1: <GT >, <G >... and it has the following neighbors at...

Given a target node... It has the following neighbors at hop 1: N1: <GT >, <G ... and it has the following neighbors at...

Given a target node... It has the following neighbors at hop 1: N1: <G >... and it has the following neighbors at hop 2...

Given a target node... It has the following neighbors at hop 1: N1: <GT > <G > ... and it has the following neighbors...

Degr e

[Figure 157]

[Figure 158]

KL

KL

concat

concat

[Figure 159]

Anchor B

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Anchor A

[Figure 166]

[Figure 167]

Graph Info

≈

Modality Info

Response

Response

Text

Text

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

###### 2hop

###### 1hop

[Figure 183]

[Figure 184]

softmax = [0.6, 0.2, 0.2]

softmax = [0.6, 0.3, 0.1]

concat

[Figure 185]

Pooling

Pooling

Weighted Sum

Weighted Sum

[Figure 186]

[Figure 187]

Image

Image

[Figure 188]

###### Answer:..... (b)

[Figure 189]

Anchor B

[Figure 190]

Given a target node... It has the following neighbors at hop 1: N1: <G >... and it has the ...

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

Node-wise Modality Preference

softmax = [0.6, 0.3, 0.1]

[Figure 197]

[Figure 198]

Multiple Domains Preference Aversion Template Bank Forward Backward

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

Tunable Frozen

- Figure 2. Overview of the proposed Mario framework. Given a MMG, Stage 1 uses a graph-conditioned vision–language model to perform structure-aware image–text alignment: images and texts are initially encoded, symmetrically refined by a transformer-embedded mixer that injects graph structure into token embeddings, and then aligned via contrastive learning. Stage 2 builds on these aligned features with modality-adaptive graph instruction tuning, where a lightweight router, trained under LLM supervision (a), infers each node’s modality preference and selects the most suitable modality-specific template for effective multimodal graph reasoning (b).

graph datasets and benchmarks [25, 39, 46], including some multimodal KG works [17, 26], have facilitated research in this area, the development of effective multimodal graph models remains an open challenge. A common approach [13, 35, 37] leverages multimodal representation alignment, primarily relying on VLMs [21, 32] to generate multimodal node embeddings and do downstream graph tasks. More recently, MLaGA [8] uses a graph-guided multimodal aligner with instruction tuning to provide LLMs graph-aware multimodal tokens, but it aligns only after fusing text and image into a shared query representation (not per-modality). As a result, it implicitly assumes equal modality utility and fails to address node-level modality inconsistency and heterogeneous modality preference in MMGs. Graph4MM [28] introduces hop-diffused attention to inject multi-hop structural information directly into self-attention, but it primarily targets MMGs with missing modalities, overlooking the prevalent fully observed setting where each node has both text and image. Consequently, existing approaches still miss two key realities in MMGs: (i) text–image consistency can be weak, making na¨ıve fusion/alignment unreliable, and (ii) modality preference varies across nodes.

(vi,vj) ∈ E and 0 otherwise. X is the set of multimodal node features, where each element xv ∈ X is a structured pair (xtextv ,ximagev ), with xtextv ∈ Rd

t and ximagev ∈ Rd

i denoting the textual and visual features of node v.

Large Language Models and Instruction Tuning. Large Language Models can be adapted to downstream tasks via instruction tuning, which enables parameter-efficient or full updates to model parameters θ. Unlike conventional finetuning, instruction tuning enhances LLMs through structured prompts that combine task-specific instructions with optional learnable soft tokens. Given an input sequence of textual tokens X = x1,x2,...,xp, an instruction function T produces a structured prompt T (X). This may be concatenated with a learnable soft token set S = s1,s2,...,sq to form the augmented prompt P = [T (X);S]. The model then generates an output sequence Y conditioned on P and X. Training jointly optimizes S and θ to improve task adaptation while reducing the need for full model retraining.

### 4. Methodology

In this section, we introduce the detailed architecture of Mario, as illustrated in Figure 2. It consists of two novel stages: a graph-conditioned vision–language model (Sec. 4.1) that enhances cross-modal consistency in multimodal graphs (C1 ✓), and a modality-adaptive graph instruction tuning scheme (Sec. 4.2) tailored to heterogeneous modality preferences (C2 ✓). We conclude with an overview of the model’s training and inference procedures, along with a discussion of runtime complexity (Sec. 4.3).

### 3. Preliminary

Multimodal Graph. A multimodal graph is a structured data format where each node is associated with multiple modalities. In our case, each node has a textual description and an associated image. Formally, the graph is denoted as G = (V,E,A,X), where V is the node set, E the edge set, and A ∈ RN×N the adjacency matrix with Aij = 1 if

#### 4.1. Graph-conditioned Vision-Language Model

Objective. Given a multimodal graph G where every node v owns a text sequence Tv and an image patch sequence Iv, Stage 1 learns a latent space in which (i) text and vision cues of the same node are metrically close, (ii) structure-aware alignment is performed to preserve fine-grained modality information and (iii) embeddings respect neighborhood dependencies to improve modality consistency. We introduce a graph-conditioned vision–language model (GVLM) that employs a dual-tower architecture, where each encoder is equipped with a Topology-Aware Multimodal Mixer so that its [CLS] summary attends over all tokens and patches under graph guidance. Bidirectional InfoNCE [36] is then applied to these graph-conditioned [CLS] representations, aligning the two streams in a structure-aware manner.

Modality Encoding. We employ two separate L-layer Transformers, one for each modality. For a given node v, the hidden representation at layer l is denoted as Hlv,M ∈ Rt

M×d, where M ∈ t,i indicates the modality (text or image), tt = m is the number of text tokens, and ti = n is the number of image patches. The initial Layer-0 embeddings are derived from pretrained vision-language representations like CLIP, enriched with position embeddings to support subsequent transformer layers. The [CLS] token embedding, hlv,M = Hlv,M[0], serves as the node representation used in the following topology-aware multimodal mixer.

Topology-Aware Multimodal Mixer. For modality M ∈ {t,i} and layer l, we first gather the [CLS] summaries of all graph nodes to build a node-by-feature matrix:

HlG,M = hu,Ml u∈V ∈ R|V|×d. (1)

Each attention head h=1,...,H projects the node representations into queries, keys, and values via:

QM,h, KM,h, VM,h = HlG,M · WM,hQ , WM,hK , WM,hV , (2)

where each WM,h· ∈ Rd×(d/H) is a trainable projection matrix for head h. Scaled dot-product attention, enriched

with a graph-aware position bias, is then applied:

⊤ M,h

HGl,M = Hh=1softmax QM,hK

###### + Bh VM,h. (3)

√

d/H

The learned position bias Bh encodes graph structural roles, distinguishing relationships between the nodes, serving as a form of relative positional information, and is implemented as head-specific learnable scalars indexed by shortest-path-distance buckets. Concatenating all heads

yields HGl,M ∈ R|V|×d; where the v-th row hv,Ml encodes the structure-aware representation for node v

Reinjection for Multimodal Context Integration. To propagate the structure-aware representation hv,Ml back

into the token stream, we append it to the front of the modality-specific sequence to replace the previous [CLS] token embedding, thereby keeping the sequence length and embedding dimension unchanged. This augmented sequence is then processed by the next Transformer block:

Hv,Ml = hv,Ml ∥ Hv,Ml [1 :] . (4)

A new [CLS] token is produced at each layer, allowing the model to iteratively refine its node-level representation by blending freshly aggregated graph context with the original token features. Repeating this mixer–reinjection operation for L layers yields final, structure-aware embeddings

htextv = HLv,t[0], himagev = HLv,i[0], (5)

which capture both modality-specific nuances and topology-aware signals, and will later serve as GVLM prototypes for cross-modal contrastive alignment.

Cross-Modal Contrastive Learning. To further tighten the gap between modalities, we perform a bidirectional contrastive objective on the structure-aware modality embeddings obtained above to train our GVLM: for each node v in a batch B, its text–image pair htextv ,himagev is the sole positive, while all cross-node combinations serve as negatives sampled all in batch. Since these embeddings already fold in topology, the neighbor signals—absent in plain text corpora—become the key to narrowing the cross-modal gap, forcing the model to learn representations that are simultaneously modality-aligned and structure-aware. We minimize the symmetric, temperature-scaled InfoNCE loss

1 |B| v∈B

s(v,v)/τ

s(v,v)/τ

log e

u∈B es(v,u)/τ + log e

LS1 = −

u∈B es(u,v)/τ ,

(6) where s(u,v) is the cosine similarity between htextu and himagev and τ controls sharpness. This step delivers modality-consistent, topology-aware representations that feed directly into Stage 2’s adaptive instruction tuning and empirically boost downstream cross-modal coherence.

#### 4.2. Modality-Adaptive Graph Instruction Tuning

Objective. Building on the features obtained in Stage 1, Stage 2 endows the LLM with node-level modality adaptivity. (i) For each node, we construct prompts with multimodal graph signals under three modality views (text, image, multimodal) by blending its features with top 1/2hop neighbors. (ii) A Modality–Adaptive Prompt Router (MAPR), trained with a probability-weighted LLM loss plus a KL term, reallocates probability toward the view that yields the lowest loss, so the LLM can exploit informative modalities while down-weighting noisy ones.

Multimodal Graph-Contextual Signals. For every node v, we expose the LLM to both its intrinsic multimodal features and the most relevant structural context. We introduce

two special tokens, ⟨GTv⟩ and ⟨GIv⟩, which are placed in the prompt to provide the LLM with text and image information about graph node v. Their embeddings are obtained by applying a learnable shared projector P to the Stage-1 features htextv and himagev , mapped into the LLM token embedding space. To enrich the prompt with neighborhood evidence, we examine 1-hop and 2-hop neighbors N1(v),N2(v) from the training set and select the Top-K nodes per hop that are relatively important to v based on the cosine similarity between the concatenated embeddings [htextu ∥himageu ] and [htextv ∥himagev ]. For each chosen neighbor u we create and optionally inject ⟨GTu⟩,⟨GIu⟩ based on the anchor node v’s modality preference, and attach its label ℓu following the ICL paradigm for LLMs (during inference, this entire procedure is restricted to training nodes only; validation/test nodes are never used as in-context exemplars), thereby forming different modality-specific templates.

Prompt Template Bank. A node’s modality preference is shaped jointly by its own modalities and its local subgraph. To expose complementary MMG evidence under different preferences, we form three modality-specific special-token groups for node v:

Svtxt = ⟨GTv⟩ ; ⟨GTu1⟩, . . . , ⟨GTuK⟩ , Svvis = ⟨GIv⟩ ; ⟨GIu1⟩, . . . , ⟨GIuK⟩ ,

Svmm = ⟨GTv⟩, ⟨GIv⟩ , . . . , ⟨GTuK⟩, ⟨GIuK⟩ ,

where ⟨GT·⟩ and ⟨GI·⟩ are the graph-text and graph-image tokens defined earlier. We then define the prompt for node v as the concatenation of three parts—I (task instruction), rv (anchor node raw text), and Sv(k) (modality-specific special tokens)—i.e., Pv(k) = I ⊕ rv ⊕ Sv(k), where k ∈ {txt,vis,mm}. Here, I is a concise instruction (e.g., “Predict the node category.”) that specifies the task; rv passes the anchor node’s raw textual content to preserve finegrained semantics; and Sv(k) supplies the modality-specific graph signals from v and its top 1/2-hop neighbors, which seamlessly weaves together node-specific multimodal cues and the most informative structural signals before being routed to the LLM to unlock the in-context learning potential of LLM in the context of multimodal graphs.

Modality–Adaptive Prompt Router. However, how to determine each node’s modality preference while being minimally affected by poorer modalities and make the LLM update selectively according to its advantageous modality remains open. To this end, we introduce a lightweight yet expressive Modality–Adaptive Prompt Router (MAPR) placed before the LLM. For each node v we concatenate its

- Stage 1 multimodal embedding, the mean-pooled 1-/2-hop context, and a logarithmic degree term to serve as input:

zv = htextv ; himagev ; ϕ(1)(v) ; ϕ(2)(v) ; log dv ∈ R4d+1, (7)

1 2|Nh(v)|

htextu + himageu , (8)

ϕ(h)(v) =

u∈Nh(v)

For node pairs in the link prediction task, we first pool the two nodes and treat them as a single pseudo-node, then construct the input in the same manner as for regular nodes. Afterwards, we select their common neighbors in the template and embed the corresponding tokens. During training, we expose all templates in the bank to the LLM so that it learns to rank prompts containing different modality information. The lightweight MLP router fuses diverse information into one feature and obtains modality-selection logits sv ∈ R3 that are normalised to routing probabilities pv = softmax(sv) = [p(vtxt),p(vvis),p(vmm)]⊤. For each template, the LLM produces a negative causal language modeling loss through function:

|Yv|

ℓ(vk) = −

i=1

log pθ yi | y<i,Pv(k) , (9)

which we convert into a performance posterior qv = softmax −[ℓ(vtxt),ℓ(vvis),ℓ(vmm)] and use it to guide MAPR updates. Training minimizes the composite loss:

1 |B| v∈B k

qv(k) ℓ(vk) + λKL qv ∥pv . (10)

LS2 =

where the first term is a performance-weighted objective that routes gradient to each template in proportion to the posterior qv inferred from its loss, while the KL term regularizes the router by matching its predictive distribution pv to qv. This teacher–student coupling shifts probability mass toward the lower-loss templates, encouraging the LLM to rely on informative modalities and to down-weight noisy ones. MAPR stabilizes optimization and improves generalization by tailoring supervisory signal to modality preference, attenuating gradients from mismatched modalities.

#### 4.3. Discussion

Training & Inference Strategy. Mario adopts a sequential routine. (i) Stage1 pre-training. We first optimize the GVLM with the cross-modal InfoNCE loss LS1 until convergence, obtaining Θ⋆S1. (ii) Stage 2 instruction tuning. Keeping Θ⋆S1 fixed, we then fine-tune the LLM using LoRA [15] together with MAPR using the composite loss LS2. The datasets used in the two training stages are kept identical. At inference time, the MAPR switches from the soft routing used in training to a hard policy, selecting the template k⋆ = arg maxp(vk), and feeds only the corresponding prompt into the LLM, thus incurring no extra compute compared with a single-template pipeline.

Complexity Analysis. (i) Stage 1. Each multimodal mixer layer attends over all nodes’ [CLS] tokens with graph

Table 1. Single-Focus performance comparison on four MMG datasets. Red ↑ number denotes the absolute accuracy gain over each baseline. Since LLaVA 1.5 does not support multiple image inputs, we concatenate images of node pairs and their neighbors into a single canvas before feeding the model to ensure a fair comparison. We fine-tune Qwen2.5-VL by feeding it the anchor node and its neighbors’ texts+images. The best baseline in each modality setting is highlighted in bold, and Mario (ours) is shown in the last row with underline.

|Methods| |Node Classification Accuracy (%)| |Link Prediction Accuracy (%)|
|---|---|---|---|---|
|Datasets| |Movies Reddit CDs Arts<br><br>| |Movies Reddit CDs Arts<br><br>|

Text-only

GCN 43.77 ↑9.86 84.29 ↑11.01 51.44 ↑11.99 76.92 ↑15.21 70.20 ↑23.70 74.23 ↑17.07 71.63 ↑21.07 66.27 ↑23.69 GATv2 48.71 ↑4.92 85.57 ↑9.73 54.67 ↑8.76 80.39 ↑11.74 72.63 ↑21.27 70.63 ↑20.67 73.27 ↑19.43 69.47 ↑20.49 SAGE 43.17 ↑10.46 85.64 ↑9.66 52.16 ↑11.27 85.26 ↑6.87 65.47 ↑28.43 71.53 ↑19.77 68.73 ↑23.97 65.50 ↑24.46

LLaMA3-8B 15.50 ↑38.13 77.23 ↑18.07 33.52 ↑29.91 72.42 ↑19.71 63.10 ↑30.80 79.30 ↑12.00 74.30 ↑18.40 66.40 ↑23.56 GraphGPT 23.85 ↑29.78 22.99 ↑72.31 23.85 ↑39.58 57.35 ↑34.78 53.50 ↑40.40 42.90 ↑48.40 68.95 ↑23.75 65.46 ↑24.50 LLaGA 49.57 ↑4.06 92.14 ↑3.16 54.74 ↑8.69 89.32 ↑2.81 75.90 ↑18.00 88.45 ↑2.85 84.90 ↑7.80 82.30 ↑7.66

GraphPrompter 46.35 ↑7.28 91.16 ↑4.14 46.27 ↑17.16 84.41 ↑7.72 67.90 ↑26.00 87.73 ↑3.57 78.70 ↑14.00 71.80 ↑18.16 Image-only

GCN 45.24 ↑8.39 88.63 ↑6.67 51.61 ↑11.82 76.24 ↑15.89 69.53 ↑24.37 75.07 ↑16.23 71.43 ↑21.27 67.47 ↑22.49 SAGE 44.37 ↑9.26 89.83 ↑5.47 54.94 ↑8.49 80.29 ↑11.84 73.50 ↑20.40 74.00 ↑17.30 67.67 ↑25.03 62.93 ↑27.03 GATv2 50.02 ↑3.61 89.87 ↑5.43 55.82 ↑7.61 78.46 ↑13.67 74.67 ↑19.23 74.50 ↑16.80 73.40 ↑19.30 69.37 ↑20.59

LLaVA1.5-13B 17.78 ↑35.85 65.78 ↑29.52 30.13 ↑33.30 52.01 ↑40.12 45.50 ↑48.40 49.80 ↑41.50 50.13 ↑42.57 51.10 ↑38.86 Text+Image

|GCN GATv2 SAGE LLaVA1.5-13B Qwen2.5-VL UniGraph2 GraphGPT-A LLaGA-A GraphPrompter-A Graph4MM MLaGA<br><br>| |46.96 ↑6.67 88.09 ↑7.21 52.67 ↑10.76 76.76 ↑15.37 49.29 ↑4.34 89.80 ↑5.50 56.44 ↑6.99 81.19 ↑10.94<br><br>44.07 ↑9.56 90.21 ↑5.09 54.74 ↑8.69 85.35 ↑6.78 18.89 ↑34.74 72.33 ↑22.97 40.10 ↑23.33 57.63 ↑34.50<br><br>49.86 ↑3.77 70.11 ↑25.19 53.52 ↑9.91 88.99 ↑3.14<br><br>45.91 ↑7.72 92.65 ↑2.65 52.13 ↑11.30 78.81 ↑13.32 18.81 ↑34.82 21.98 ↑73.32 29.56 ↑33.87 58.73 ↑33.40<br><br>50.61 ↑3.02 92.94 ↑2.36 56.29 ↑7.14 88.83 ↑3.30 45.54 ↑8.09 92.85 ↑2.45 52.06 ↑11.37 83.86 ↑8.27<br><br>51.07 ↑2.56 92.89 ↑2.41 55.53 ↑7.90 89.32 ↑2.81 49.42 ↑4.21 89.79 ↑5.51 56.45 ↑6.98 89.82 ↑2.31<br><br><br><br><br>| |69.93 ↑23.97 74.37 ↑16.93 71.07 ↑21.63 67.53 ↑22.43 72.73 ↑21.17 72.67 ↑18.63 73.20 ↑19.50 70.03 ↑19.93<br><br>70.27 ↑23.63 72.00 ↑19.30 68.37 ↑24.33 66.27 ↑23.69 65.80 ↑28.10 74.63 ↑16.67 56.90 ↑35.80 67.90 ↑23.86<br><br><br>88.10 ↑5.80 88.20 ↑3.10 80.90 ↑11.80 80.40 ↑9.56 64.60 ↑29.30 80.40 ↑10.90 81.00 ↑11.70 67.80 ↑22.16 49.55 ↑44.35 51.55 ↑39.75 72.37 ↑20.33 63.46 ↑26.50 77.90 ↑16.00 88.90 ↑2.40 84.15 ↑8.55 82.05 ↑7.91 70.90 ↑23.00 87.82 ↑3.48 79.83 ↑12.87 80.11 ↑9.85 90.24 ↑3.66 90.80 ↑0.50 83.21 ↑9.49 81.60 ↑8.36<br><br>89.96 ↑3.94 90.35 ↑0.95 72.87 ↑19.83 82.97 ↑6.99<br><br><br>|
|---|---|---|---|---|
|Mario-8B (ours)<br><br>| |53.63 95.30 63.43 92.13<br><br>| |93.90 91.30 92.70 89.96<br><br>|

bias, yielding a per-layer cost O(|Vs|2d), where Vs denotes the sampled node set and typically |Vs| ≪ |V|. However, only 1–2 layers are sufficient in practice to reach alignment convergence, so the overall training time remains moderate compared with a deeper vanilla Transformer stack. (ii)

- Stage 2. For every training sample we execute three forward–backward passes—one per template—yielding a cost

of O 3fLLM . Empirically, the router allows the model to converge in roughly half the epochs needed by a single template baseline with lower losses, offset by the extra per-step computation (see the empirical training curve in Figure 3).

### 5. Experiments

We conducted extensive experiments, primarily aimed at addressing the following research questions (RQs): RQ1: How does Mario perform on standard multimodal graph reasoning tasks compared to leading baselines that take different modalities as input? RQ2: How well does Mario generalize in zero-shot settings when evaluated on entirely unseen MMGs? RQ3: To what extent does the graphconditioned vision–language model enhance representation

learning and contribute to the multimodal instruction tuning process? RQ4: How does Modality–Adaptive Graph Instruction Tuning improve predictive performance over single-template baselines and how efficient is it?

#### 5.1. Experiment Configurations

Datasets. Our experiments span a diverse range of domains for MMGs, including E-commerce: AmazonArts&Crafts [14], Amazon-CDs&Vinyl [14], Amazon-Toys [39], Amazon-Movies [39], Social networks: Reddit-S [39], and Literature: Goodreads (Books) [46]. In these graphs, nodes represent items or posts, and edges indicate copurchase or co-comment relationships. Among them, four datasets are used in the Single-Focus (Table 1) and MixTraining (Table 2) settings, while the remaining two are reserved for transfer experiments (Table 3), covering two primary tasks: node classification (NC) and link prediction (LP). Appendix shows the details of datasets and splits.

Baselines. To ensure fair and comprehensive comparison, we categorize baselines by input modality: (1) Textonly: Text-based GNNs (GCN [18], GraphSAGE [20], GATv2 [3]), LLM (LLaMA3-8B with LoRA), and text-

Table 2. Mix-Training performance comparison on four multimodal graph datasets. The table design is consistent with Table 1.

|Methods| |Node Classification Accuracy (%)| |Link Prediction Accuracy (%)|
|---|---|---|---|---|
|Datasets| |Movies Reddit CDs Arts<br><br>| |Movies Reddit CDs Arts<br><br>|

Text-only

GCN 47.15 ↑3.83 86.17 ↑7.03 50.76 ↑9.17 79.03 ↑12.17 70.30 ↑22.15 74.50 ↑16.23 72.93 ↑20.57 62.33 ↑30.27 SAGE 46.85 ↑4.13 89.96 ↑3.24 53.24 ↑6.69 87.46 ↑3.74 63.30 ↑29.15 72.03 ↑18.70 62.03 ↑31.47 63.70 ↑28.90

LLaGA 47.80 ↑3.18 91.14 ↑2.06 51.33 ↑8.60 74.02 ↑17.18 87.28 ↑5.17 88.95 ↑1.78 90.32 ↑3.18 87.06 ↑5.54 GraphPrompter 45.21 ↑5.77 90.36 ↑2.84 44.70 ↑15.23 83.91 ↑7.29 70.27 ↑22.18 86.70 ↑4.03 83.20 ↑10.30 88.30 ↑4.30 LLaMA3-8B 33.27 ↑17.71 59.63 ↑33.57 35.76 ↑24.17 65.21 ↑25.99 63.10 ↑29.35 69.51 ↑21.22 74.10 ↑19.40 67.40 ↑25.20

Image-only

GCN 39.20 ↑11.78 89.30 ↑3.90 51.20 ↑8.73 75.03 ↑16.17 66.17 ↑26.28 70.83 ↑19.90 67.07 ↑26.43 62.00 ↑30.60 SAGE 38.74 ↑12.24 90.25 ↑2.95 53.60 ↑6.33 79.40 ↑11.80 51.70 ↑40.75 68.20 ↑22.53 62.93 ↑30.57 59.37 ↑33.23

Text+Image

|GCN SAGE MLaGA<br><br>| |47.76 ↑3.22 89.87 ↑3.33 52.28 ↑7.65 78.59 ↑12.61 47.75 ↑3.23 90.39 ↑2.81 56.51 ↑3.42 88.09 ↑3.11 50.08 ↑0.90 91.45 ↑1.75 57.46 ↑2.47 86.32 ↑4.88<br><br>| |71.70 ↑20.75 74.30 ↑16.43 68.60 ↑24.90 66.87 ↑25.73 64.30 ↑28.15 71.23 ↑19.50 68.43 ↑25.07 61.33 ↑31.27 90.90 ↑1.55 90.56 ↑0.17 86.53 ↑6.97 83.57 ↑9.03<br><br>|
|---|---|---|---|---|
|Mario-8B (ours)| |50.98 93.20 59.93 91.20<br><br>| |92.45 90.73 93.50 92.60<br><br>|

centric GraphLLMs (LLaGA [4], GraphPrompter [27], GraphGPT [34]). (2) Image-only: GNNs over image embeddings and LVLMs such as LLaVA v1.5-13B [24] limited to visual inputs. (3) Text+Image: GNNs on fused text–image features (Graph-MLLM Style [25]), MLaGA [8], Graph4MM [28] and UniGraph2 [13]; LVLMs (Tuned Qwen2.5-VL [2]) with dual-modality inputs; and augmented text-based GraphLLMs (the “-A” variant) that simulate multimodal reasoning by appending image captions to textual inputs to help them do multimodal reasoning.

Implementation details. Our default backbone LLM is LLaMA3.1-8B. As shown in our backbone ablation results (Appendix), Mario is largely robust to the choice of LLM backbone. We by default adopt CLIP encoders to extract initial node embeddings from textual and visual modalities, as it yields satisfactory performance in our preliminary evaluations without further fine-tuning. In addition to all experiments presented below, we also provide in the Appendix more details on: hyperparameter settings, LLM backbone ablations, variance analysis, t-SNE analysis of our GVLM alignment, comparisons between frozen and LoRA-tuned Mario, template design, GPU resources, as well as additional main experiment results and sensitivity analyses. All results are averaged over three runs.

#### 5.2. Overall Performance Comparison (RQ1)

We benchmark Mario on NC and LP, training and testing on each dataset separately. We refer to this setting as the Single-Focus regime. Table 1 reports results against baselines categorized by their underlying modality.

Observation 1: Mario delivers the highest accuracy across all datasets and both tasks under the singlefocus setting. For instance, it lifts NC performance on CDs from 56.45% (best baseline) to 63.43%, and LP accuracy improves by an average of 4.73% across four datasets. These gains stem from the synergy between Stage

1’s structure-aware image–text alignment and Stage 2’s modality-adaptive graph instruction tuning, which together encode product semantics and relations more faithfully.

##### Observation 2: Enabling LLMs to directly interpret

aligned multimodal features with structural information is more effective than augmenting input via image-to-text conversion. Augmented GraphLLMs (e.g. LLaGA, GraphPrompter, GraphGPT) relatively trail Mario by an average 5.48%, 11.00% and 135.9%, respectively, on NC. Retaining modality-specific vectors—augmented by graph context—preserves fine-grained semantics and avoids the noise and redundancy introduced by text-only surrogates.

5.3. Generalization & Transferability (RQ2)

We first train Mario on an equal four-way mixture of datasets and test on each domain individually (Table 2) and refer to this as data generalization. Next, we evaluate zero-shot transfer: the model is trained on one (or several) source graphs and assessed on an unseen graph (Table 3).

##### Observation 3: Although the model’s performance de-

clines to some extent under the Mix-Training setting, Mario manages to maintain a relatively small performance drop and even works better while still preserving a significant lead over the baselines. In Table 2, Mario achieves an average relative improvement of 2.88% in NC and 2.57% in LP over the best baseline, further proving Mario’s strong generalization across multiple datasets in joint training, maintaining superior performance while adapting to diverse graph structures and modalities.

##### Observation 4: Mario achieves robust zero-shot rea-

soning in multimodal graph inference, outperforming baselines by a notable margin. Mario achieves 1.64× higher NC accuracy than the best baseline on Toys → Movies, 1.48× on Toys+Movies → CDs, and 1.25× higher LP accuracy on Toys → Movies. This can be attributed to its GVLM, which preserves graph-invariant semantics across

Table 3. Zero-Shot Results (Accuracy).

|Model Tasks<br><br>|Toys → Movies|Toys+Movies → CDs|Toys → Books|
|---|---|---|---|
| |NC LP<br><br>|NC LP<br><br>|NC LP<br><br>|

GCN 5.29 62.23 5.64 55.70 11.47 45.43 GATv2 6.58 62.03 6.58 63.00 14.50 49.00 SAGE 3.61 61.80 5.87 66.13 7.73 48.40

GraphPrompter 11.15 72.70 36.66 51.00 23.00 62.50 LLaGA 8.63 62.00 9.69 72.85 8.85 52.50 MLaGA 24.95 79.87 16.20 52.82 44.85 57.85

Mario-8B (Ours) 41.00 86.60 54.32 82.50 47.30 78.30

[Figure 204]

[Figure 205]

(a) Movies (b) Reddit

- Figure 3. Training curves of Mario vs. the text-only template (Fixed) on two datasets, with early-stopping epochs in the end.

modalities, and the modality-adaptive router, which induces transferable inductive bias by dynamically selecting the most informative prompt per node—even in unseen graph topologies. These results highlight Mario’s strong zero-shot reasoning ability across diverse unseen domains.

#### 5.4. Ablation Study (RQ3)

To address RQ3, we replace Stage 1’s GVLM with other architectures that capture graph structural information, such as GNNs. We evaluate these models on NC across three datasets to validate the superiority of our GVLM design.

Observation 5: Fine-grained alignment of structured image-text features leads to stronger LLM reasoning than global or structure-agnostic alignment. GNNs and MLP often overlook token-level interactions, losing fine visual details. As shown in Table 4, GVLM surpasses GNNs and MLP across all datasets, especially achieving up to +5.15% relative average gain on Movies. While vanilla Transformer-based models are more complex, GVLM adopts a shallow design and converges quickly. Its runtime is only 1.5× that of GNNs/MLPs—an acceptable trade-off for significantly better performance.

Table 4. Ablation Study of Stage 1’s Model.

Model Arts Reddit Movies

Acc(%) s/epoch Acc(%) s/epoch Acc(%) s/epoch

|GCN SAGE GATv2 MLP| |90.32 ↑2.00 114 90.75 ↑1.52 117 90.03 ↑2.33 129 89.95 ↑2.42 109<br><br>|93.30 ↑2.14 85 92.10 ↑3.47 89 92.90 ↑2.58 93 92.70 ↑2.80 79<br><br>|50.90 ↑5.36 78<br>51.10 ↑4.95 81 51.30 ↑4.54 85 50.70 ↑5.78 75<br>|
|---|---|---|---|---|
|GVLM| |92.13 174<br><br>|95.30 135<br><br>|53.63 122<br><br>|

#### 5.5. Efficiency Study and Visualization (RQ4)

In this section, we delve into the effectiveness and efficiency of Modality-Adaptive Graph Instruction Tuning.

Observation 6: With Modality-Adaptive Graph Instruction Tuning, LLMs exhibit notably faster convergence and consistently outperform all single-template counterparts

[Figure 206]

[Figure 207]

(a) Arts&Reddit (b) CDs&Movies

- Figure 4. Comparison of Mario with three fixed prompt templates containing different modality information across the four datasets.

by a large margin. Fig. 3 compares the training losses of Mario against a single-template variant (w/o MAPR). As shown, Mario achieves significantly faster convergence on both Movies (2.3×) and Reddit (1.3×), while also attaining lower final losses after convergence. Although each epoch of Mario takes approximately 1.5-2× longer than the singletemplate variant observed in our experiments, its accelerated convergence enables it to complete training in a comparable overall time. Fig. 4 provides a detailed comparison between Mario and different single-template variants. Benefiting from its adaptive tuning mechanism, Mario consistently outperforms all variants by a large margin—for example, on the CDs dataset, it relatively surpasses the average performance of variants by 3.4%. This indirectly demonstrates the importance of respecting nodes’ modality preferences.

text

image

text+image

(a) Movies

text

image

text+image

(b) Arts

- Figure 5. Visualization of Router Selections across two MMGs. Observation 7: Modality preferences in MMGs largely

follow a homophily pattern. To make this concrete, Fig. 5 visualizes MAPR’s modality choice for each node within two larger subregions sampled from two MMGs. Within each region, the distribution of modality preferences is clearly non-uniform: nodes with the same color (preferred modality) often appear in small clusters, and certain areas are dominated by a single modality (e.g., green text+image nodes in Arts in the middle). This suggests that neighboring nodes, which are connected because users tend to co-view or co-purchase the corresponding items, often share similar semantic attributes and thus benefit from similar “best” modalities. We observe analogous locally coherent patterns in many other parts of the above two graphs as well.

- 6. Conclusion

In this paper, we highlight two underexplored challenges in MMG reasoning: cross-modal inconsistency and heterogeneous modality preference. We propose Mario, a

novel unified two-stage framework that performs structureaware image–text alignment with a graph-conditioned vision–language model, then applies modality-adaptive graph instruction tuning via a lightweight router that learns nodespecific routing to satisfy the nodes’ modality preferences. Extensive experiments on multiple MMG benchmarks show that Mario consistently outperforms strong baselines and enables more reliable multimodal graph reasoning. We hope our work paves the way for future advances in LLM-based multimodal graph reasoning.

### References

- [1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 1(2):3, 2023. 11
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. 7
- [3] Shaked Brody, Uri Alon, and Eran Yahav. How attentive are graph attention networks? arXiv preprint arXiv:2105.14491,

2021. 6

- [4] Runjin Chen, Tong Zhao, Ajay Jaiswal, Neil Shah, and Zhangyang Wang. Llaga: Large language and graph assistant. arXiv preprint arXiv:2402.08170, 2024. 7
- [5] Runjin Chen, Tong Zhao, AJAY KUMAR JAISWAL, Neil Shah, and Zhangyang Wang. Llaga: Large language and graph assistant. In Forty-first International Conference on Machine Learning, 2024. 2, 11
- [6] Zhikai Chen, Haitao Mao, Hang Li, Wei Jin, Hongzhi Wen, Xiaochi Wei, Shuaiqiang Wang, Dawei Yin, Wenqi Fan, Hui Liu, et al. Exploring the potential of large language models (llms) in learning on graphs. ACM SIGKDD Explorations Newsletter, 25(2):42–61, 2024. 2
- [7] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, march

2023. URL https://lmsys. org/blog/2023-03-30-vicuna, 3(5),

2023. 11

- [8] Dongzhe Fan, Yi Fang, Jiajin Liu, Djellel Difallah, and Qiaoyu Tan. Mlaga: Multimodal large language and graph assistant. arXiv preprint arXiv:2506.02568, 2025. 3, 7, 12
- [9] Yi Fang, Dongzhe Fan, Daochen Zha, and Qiaoyu Tan. Gaugllm: Improving graph contrastive learning for textattributed graphs with large language models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 747–758, 2024. 2
- [10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning

- capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1
- [11] Jiayan Guo, Lun Du, Hengyu Liu, Mengyu Zhou, Xinyi He, and Shi Han. Gpt4graph: Can large language models understand graph structured data? an empirical evaluation and benchmarking. arXiv preprint arXiv:2305.15066, 2023. 2
- [12] Xiaoxin He, Xavier Bresson, Thomas Laurent, Adam Perold, Yann LeCun, and Bryan Hooi. Harnessing explanations: Llm-to-lm interpreter for enhanced text-attributed graph representation learning. In The Twelfth International Conference on Learning Representations, 2023. 2
- [13] Yufei He, Yuan Sui, Xiaoxin He, Yue Liu, Yifei Sun, and Bryan Hooi. Unigraph2: Learning a unified embedding space to bind multimodal graphs. In Proceedings of the ACM on Web Conference 2025, pages 1759–1770, 2025. 3, 7
- [14] Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. Bridging language and items for retrieval and recommendation. arXiv preprint arXiv:2403.03952, 2024. 6
- [15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 5
- [16] Jin Huang, Xingjian Zhang, Qiaozhu Mei, and Jiaqi Ma. Can llms effectively leverage graph structural information: when and why. arXiv preprint arXiv:2309.16595, 2023. 2
- [17] Wei Huang, Peining Li, Meiyu Liang, Xu Hou, Junping Du, Yingxia Shao, Guanhua Ye, Wu Liu, Kangkang Lu, and Yang Yu. Elmm: Efficient lightweight multimodal large language models for multimodal knowledge graph completion, 2025. 3
- [18] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016. 1, 6
- [19] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023. 11
- [20] Guohao Li, Matthias M¨uller, Bernard Ghanem, and Vladlen Koltun. Training graph neural networks with 1000 layers. In International conference on machine learning, pages 6437–

6449. PMLR, 2021. 6

- [21] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022. 3
- [22] Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semi-supervised learning. In Proceedings of the AAAI conference on artificial intelligence, 2018. 1
- [23] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 1

- [24] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 7
- [25] Jiajin Liu, Dongzhe Fan, Jiacheng Shen, Chuanhao Ji, Daochen Zha, and Qiaoyu Tan. Graph-mllm: Harnessing multimodal large language models for multimodal graph learning. arXiv preprint arXiv:2506.10282, 2025. 1, 3, 7
- [26] Junming Liu, Siyuan Meng, Yanting Gao, Song Mao, Pinlong Cai, Guohang Yan, Yirong Chen, Zilin Bian, Ding Wang, and Botian Shi. Aligning vision to language: Annotation-free multimodal knowledge graph construction for enhanced llms reasoning, 2025. 3
- [27] Zheyuan Liu, Xiaoxin He, Yijun Tian, and Nitesh V Chawla. Can we soft prompt llms for graph learning tasks? In Companion Proceedings of the ACM on Web Conference 2024, pages 481–484, 2024. 2, 7, 11
- [28] Xuying Ning, Dongqi Fu, Tianxin Wei, Wujiang Xu, and Jingrui He. Graph4mm: Weaving multimodal learning with structural information. In International Conference on Machine Learning, pages 46448–46472. PMLR, 2025. 3, 7
- [29] Kenta Oono and Taiji Suzuki. Graph neural networks exponentially lose expressive power for node classification. arXiv preprint arXiv:1905.10947, 2019. 1
- [30] Xuran Pan, Tianzhu Ye, Dongchen Han, Shiji Song, and Gao Huang. Contrastive language-image pre-training with knowledge graphs. Advances in Neural Information Processing Systems, 35:22895–22910, 2022. 1
- [31] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 1
- [32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 3
- [33] Yuanfu Sun, Zhengnan Ma, Yi Fang, Jing Ma, and Qiaoyu Tan. Graphicl: Unlocking graph learning potential in llms through structured prompt design. arXiv preprint arXiv:2501.15755, 2025. 2
- [34] Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. Graphgpt: Graph instruction tuning for large language models. arXiv preprint arXiv:2310.13023, 2023. 2, 7
- [35] Zhulin Tao, Yinwei Wei, Xiang Wang, Xiangnan He, Xianglin Huang, and Tat-Seng Chua. Mgat: Multimodal graph attention network for recommendation. Information Processing & Management, 57(5):102277, 2020. 3, 12
- [36] A¨aron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 4
- [37] Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, Richang Hong, and Tat-Seng Chua. Mmgcn: Multi-modal graph convolution network for personalized recommendation of micro-video. In Proceedings of the 27th ACM international conference on multimedia, pages 1437–1445, 2019. 1, 3, 12

- [38] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? arXiv preprint arXiv:1810.00826, 2018. 1
- [39] Hao Yan, Chaozhuo Li, Zhigang Yu, Jun Yin, Ruochen Liu, Peiyan Zhang, Weihao Han, Mingzheng Li, Zhengxin Zeng, Hao Sun, et al. When graph meets multimodal: Benchmarking on multimodal attributed graphs learning. arXiv preprint arXiv:2410.09132, 2024. 1, 3, 6
- [40] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. 1
- [41] Junhan Yang, Zheng Liu, Shitao Xiao, Chaozhuo Li, Defu Lian, Sanjay Agrawal, Amit Singh, Guangzhong Sun, and Xing Xie. Graphformers: Gnn-nested transformers for representation learning on textual graph. Advances in Neural Information Processing Systems, 34:28798–28810, 2021. 2
- [42] Minji Yoon, Jing Yu Koh, Bryan Hooi, and Ruslan Salakhutdinov. Multimodal graph learning for generative tasks. arXiv preprint arXiv:2310.07478, 2023. 1
- [43] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? arXiv preprint arXiv:2210.01936, 2022. 2
- [44] Mengmei Zhang, Mingwei Sun, Peng Wang, Shen Fan, Yanhu Mo, Xiaoxiao Xu, Hong Liu, Cheng Yang, and Chuan Shi. Graphtranslator: Aligning graph model to large language model for open-ended tasks. In Proceedings of the ACM on Web Conference 2024, pages 1003–1014, 2024. 2
- [45] Jianan Zhao, Le Zhuo, Yikang Shen, Meng Qu, Kai Liu, Michael Bronstein, Zhaocheng Zhu, and Jian Tang. Graphtext: Graph reasoning in text space. arXiv preprint arXiv:2310.01089, 2023. 2
- [46] Jing Zhu, Yuhang Zhou, Shengyi Qian, Zhongmou He, Tong Zhao, Neil Shah, and Danai Koutra. Multimodal graph benchmark. arXiv preprint arXiv:2406.16321, 2024. 1, 3, 6

## Mario: Multimodal Graph Reasoning with Large Language Models Supplementary Material

### 7. Appendix

#### 7.1. Dataset Details

Statistics and Introduction. The detailed statistics of the datasets we used are shown in Table 5. In these datasets, nodes represent individual products or posts, and edges denote relationships such as co-purchase or co-comment interactions between products or posts. Each node is assigned a label corresponding to its category. Every node is enriched with two modalities: textual attributes, such as product titles, descriptions, or post content, and visual attributes extracted from associated product or post images. Unlike conventional image–text benchmarks where captions are written to explicitly describe the visual content, here the two modalities are only loosely coupled and often contain complementary or even disjoint information. For example, a clothing item may have a textual description that focuses on material and fit (e.g., “soft cotton hoodie with relaxed, oversized silhouette, ideal for fall weather”) while its image emphasizes color, style, and brand logos that are never mentioned in the text. Conversely, the text may include attributes such as size range, discount information, or usertargeted marketing slogans that are not visually observable. Data Splits. For the node classification task, we adopt a standardized 6:2:2 split into training, validation, and testing for Mario and all the baselines. For the link prediction task, the training, validation, and test sets consist of 3,000, 2,000, and 1,000 edges, respectively, for training and evaluation.

Table 5. Dataset statistics across multiple MMG datasets.

Dataset Domain # Nodes # Edges # Classes

Movies E-commerce 16,672 109,195 20 Toys E-commerce 20,694 63,443 18 CDs E-commerce 36,272 844,878 15 Arts E-commerce 28,195 197,428 7

Reddit(S) Social Media 15,894 566,160 20 Goodreads Literature 685,294 7,235,084 11

#### 7.2. Experiment Details

In this section, we provide additional explanations for experiment details not covered in the paper.

Image to caption conversion. Since current GraphLLM baselines do not support processing image features, we convert images into captions using VLMs in the text+vision experiments to enhance textual modality with auxiliary information. This facilitates multimodal graph reasoning. The model used for this purpose is Qwen-VL-Chat [1].

L(V)LMs-Based Baseline Experiment Execution. In the experiments, we frequently mention using LLaMA and LLaVA. All these experiments were conducted with the assistance of vLLM [19]. vLLM is a high-performance library for efficient LLM inference and serving. It provides stateof-the-art serving throughput with optimizations such as PagedAttention, continuous batching, CUDA acceleration, FlashAttention, and speculative decoding, ensuring lowlatency execution. vLLM seamlessly integrates with Hugging Face models, supports various decoding strategies, and enables tensor/pipeline parallelism across diverse hardware platforms. In our experiments, we utilized vLLM to efficiently serve LLaMA and LLaVA, enabling scalable inference for text-based and MMG reasoning tasks while ensuring computational efficiency and high throughput.

Hyper-Parameter Settings. We provide a detailed discussion of the hyper-parameter settings used in our experiments. For Stage 1, we usually employ one layer (up to two) of GraphTransformer for structure-aware text-image alignment and we sample ∼10 nodes (Vs) to feed into the GVLM. For Stage 2, we typically select 10-15 neighbors to provide neighbor context and conduct 10 epochs of instruction tuning using LLaMA3.1-8B with early stop strategy. We use a four-layer MLP as the MAPR, and set λ = 0.01. For link prediction experiments, we only provide the neighbor context of the first node in the prompt, but these are common neighbors with the other node. The projection layer consists of two layers. For GraphPrompter [27], we use LLaMA3.1-8B as the final LLM for inference. For LLaGA [5], we follow the original paper and adopt the same setting, where Vicuna [7] serves as the primary foundational large language model. We truncate the final tokens input length to 512. All experiments involving LLM deployment were conducted on two A100-SXM4-80-GB GPUs. For GraphLLM-based baselines, we did not evaluate the vision-only setting. This is because such frameworks are inherently text-centric by design, and we followed their original modeling philosophy without extending them to visiononly scenarios. Additionally, we experimented with using image captions alone to support inference within these models, but the performance was significantly worse compared to text-only or image+text settings. Therefore, we did not include the vision-only results in the paper.

#### 7.3. t-SNE Visualization of GVLM Alignment

To further illustrate the qualitative differences between the three models in Fig. 1, we visualize their aligned text and image features using t-SNE on two multimodal graphs, Movies and Reddit. For each dataset, we randomly sample a

| |
|---|

(a) Movies – Frozen CLIP

| |
|---|

(b) Movies – Tuned CLIP

| |
|---|

(c) Movies – Mario’s GVLM (ours)

| |
|---|

(d) Reddit – Frozen CLIP

| |
|---|

(e) Reddit – Tuned CLIP

| |
|---|

(f) Reddit – Mario’s GVLM (ours)

Figure 6. t-SNE visualizations of aligned multimodal features on Movies (top) and Reddit (bottom) for the three models in Fig. 1. For each dataset, we project a randomly sampled subset of nodes from the full graph, using their aligned text and image representations as input to t-SNE. Comparing the six panels reveals how different alignment strategies affect the relative organization of text and image features in the shared latent space.

subset of nodes from the full graph and project their aligned text/image representations to 2D. This subsampling allows us to focus more closely on the structural differences between models while still capturing representative patterns. The six panels in Fig. 6 show the resulting distributions for the three models on Movies (top row) and Reddit (bottom row), respectively. Beyond the overall layout, we observe consistent qualitative trends across the six panels in Fig. 1 in the paper. On both Movies and Reddit, the frozen CLIP features form two loosely overlapping clouds, indicating a sizeable gap between text and image representations. Fine-tuning CLIP shrinks this gap and slightly tight-

- ens the clusters, but the two modalities still remain partially misaligned. In contrast, Mario produces a much more intertwined manifold where text and image features are almost co-located and organized along smoother global structures, suggesting that our graph-conditioned alignment achieved by Mario’s GVLM substantially improves cross-modal consistency while preserving meaningful semantic variation.

#### 7.4. Comparison with MMGCN and MGAT

In the main paper, MMGCN [37] and MGAT [35] are excluded, as they focus primarily on recommendation-style tasks and showed weak performance in our setting through initial experiments. For completeness, we provide here a small-scale comparison to substantiate this design choice. Table 6 reports their node classification accuracy on Movies and Arts, alongside representative ”text+image” GNN baselines under the same experimental protocol.

Overall, MMGCN and MGAT do not show clear advantages over standard GNNs. On Movies, MMGCN is essentially on par with GCN and still below GATv2, while

Table 6. Node classification accuracy (%) on Movies and Arts for additional multimodal baselines (MMGCN, MGAT) and representative unimodal GNNs (text+image settings).

Model Movies Arts SAGE 44.07 85.35 GATv2 49.29 81.19 GCN 46.96 76.76 MMGCN 46.79 86.63 MGAT 40.17 87.25

MGAT performs worse than all three GNN baselines. On Arts, MMGCN and MGAT slightly outperform some GNNs, but the gains are modest and all these methods remain far from the strong multimodal models and Mario reported in the main tables. Since SAGE, GATv2, and GCN are already treated as weak baselines in our core comparison, adding MMGCN and MGAT there would not change the conclusions; we therefore only include them in this appendix section for completeness. A similar conclusion can also be drawn from MLaGA [8].

#### 7.5. Additional GNNs Zero-Shot Results

In our zero-shot experiments in the paper, we assess the transferability of graph neural networks (GNNs) to new datasets, without re-training their core parameters. Specifically, when transitioning between datasets, we retain the trained GNN model, including its network architecture and learned parameters, and only replace the classifier layer corresponding to the new dataset. This approach ensures that the underlying graph feature extractor remains unchanged, allowing us to evaluate the generalization capacity of differ-

Table 7. Frozen Mario versus LoRA-Tuned Mario (Accuracy %).

Model Trainable Params Movies Reddit CDs Arts Node Classification (Trainable parameters are from Stage 2)

|Frozen Mario|18,886,656 (0.2346%)<br><br>|50.85<br><br>|93.60<br><br>|60.45|89.69|
|---|---|---|---|---|---|
|Mario + LoRA|22,294,528 (0.2768%)|53.63|95.30<br><br>|63.43|92.13|

Link Prediction (Trainable parameters are from Stage 2)

|Frozen Mario<br><br>|18,886,656 (0.2346%)<br><br>|90.90<br><br>|89.00|88.60<br><br>|86.30|
|---|---|---|---|---|---|
|Mario + LoRA<br><br>|22,294,528 (0.2768%)|93.90<br><br>|91.30<br><br>|92.70|89.96|

- ent models under domain shifts. Table 8 presents the additional zero-shot transfer results

across different models. This result serves as a supplement to Table 3 in the paper (where GraphLLMs adopt the textonly setting, and the other baselines adopt the text+vision setting). For the results below without explicit modality specification, the text-only modality is used (different from the setting in the paper). We evaluate the same two transfer settings: (1) Toys → Movies, where models trained on the Toys dataset are directly applied to the Movies dataset, and (2) Toys+Movies → CDs, where models trained on both the Toys and Movies datasets are tested on the CDs dataset. The evaluation is conducted under two tasks: NC (Node Classification Accuracy) and LP (Link Prediction Accuracy).

Across both transfer settings, our Mario significantly outperforms all baselines, demonstrating strong zero-shot adaptation capabilities. In contrast, traditional GNNs such as GCN, GATv2, and SAGE struggle to generalize, exhibiting considerably lower performance. For instance, in the Toys → Movies setting, GCN achieves an NC score of only 3.29, while Mario achieves 41.00, more than 10 times higher. A similar trend is observed in Toys+Movies → CDs, where Mario attains an NC score of 54.32, substantially outperforming all baselines.

Furthermore, while MLP-based models (both text-only and vision-only versions) show moderate performance in link prediction, they underperform in node classification due to their inability to leverage structural dependencies effectively. These results underscore the limitations of conventional GNNs in zero-shot scenarios and highlight the advantages of our Mario model in learning transferable multimodal representations.

#### 7.6. Frozen vs. LoRA-Tuned Mario

We also find that LoRA-tuned Mario outperforms its frozen counterpart, and both exceed all baselines by a large margin. As shown in Table 7, LoRA tuning yields consistent gains of about 1.7–3.0 points in node classification accuracy across all four datasets (e.g., from 50.85 to 53.63 on Movies and from 89.69 to 92.13 on Arts), and similarly improves link prediction by roughly 2–4 points. These improvements come with only a tiny increase in the number of trainable

Table 8. Zero-Shot Results (Accuracy %).

|Toys → Movies|Toys+Movies → CDs|
|---|---|
|NC LP|NC LP|

Model

|MLP GCN GATv2 SAGE MLP(Vision Only)|6.12 52.60<br><br>3.29 62.13<br><br>4.32 64.47<br><br><br>3.11 55.83<br><br>4.61 52.13<br><br><br>|7.04 50.20 10.01 64.17<br>8.13 67.97 6.14 59.63<br>9.06 46.09<br>|
|---|---|---|
|Mario-8B (Ours)<br><br>|41.00 86.60<br><br>|54.32 82.50|

parameters, from 18.9M (0.2346%) to 22.3M (0.2768%) of the full LLM, indicating that Mario is already strong in a frozen-LLM regime while a lightweight LoRA adapter can further boost performance without sacrificing parameter efficiency.

#### 7.7. Ablation Study of LLM Backbone

To assess the robustness of Mario across different LLMs, we conduct an ablation study using a range of LLM backbones, including both LLaMA-based and non-LLaMA families. As summarized in Table 9, Mario consistently delivers strong performance regardless of the specific LLM used, highlighting the generalizability of our framework.

Within the LLaMA2 family, increasing model size from 7B to 13B results in negligible improvement: on Arts, accuracy rises slightly from 91.06% to 91.23%, while performance on Toys slightly drops from 81.20% to 80.93%. Similarly, when switching from LLaMA2 to Vicuna-v1.5 (also LLaMA2-based), results remain largely consistent, indicating that mere scaling or minor tuning of the base LLM does not significantly alter performance in our multimodal graph reasoning tasks.

More importantly, Mario remains effective even when paired with structurally different LLMs. Using FLAN-T5XXL, a T5-style encoder-decoder model, Mario achieves 92.08% on Arts and 81.63% on Toys, outperforming all LLaMA2 variants. Furthermore, Mario-8B (LLaMA3), our best-performing configuration, achieves 92.13% and 82.58% on Arts and Toys respectively, demonstrating

[Figure 208]

Figure 7. Sensitivity analysis of the projection layer in Arts and Movies

[Figure 209]

Figure 8. Sensitivity analysis of the number of neighbors per hop in Arts and Toys

stronger capability than its LLaMA2 predecessors.

These observations collectively indicate that Mario’s architectural design—rather than the choice of LLM backbone—is the key contributor to its strong performance. Whether applied to decoder-only (LLaMA), instructiontuned (Vicuna), or encoder-decoder (FLAN-T5) models, Mario exhibits consistent gains, underscoring its backboneagnostic robustness in multimodal graph reasoning.

Table 9. Ablation Study of Different LLMs. (Accuracy %)

##### Different Size Arts Toys

Mario-7B (LLaMA2) 91.06 81.20 Mario-13B (LLaMA2) 91.23 80.93

- Mario-7B (Vicuna-v1.5) 91.09 81.07 Mario (FLAN-T5-XXL) 92.08 81.63

- Mario-8B (LLaMA3) 92.13 82.58

#### 7.8. Sensitivity Analysis

To assess the effectiveness of our designed instruction templates that incorporate multimodal node features, we conducted a sensitivity analysis on two critical components: the number of projection layers and the length of the neighbor context. These factors directly influence how effectively multimodal information is aligned and delivered to the LLM for reasoning. As shown in Figure 7, introducing a projection layer consistently improves performance over the baseline without projection. Notably, employing two layers yields the best or near-best results across both Arts and Movies datasets. This suggests that a lightweight projection module facilitates better multimodal alignment without incurring excessive complexity, enhancing the model’s ability to interpret visual-textual signals.

Similarly, Figure 8 illustrates that incorporating a limited number of neighbors per hop significantly boosts performance compared to the zero-neighbor setting. For instance, in the Toys dataset, adding neighbor context improves ac-

Table 10. Heterophily Ratios of Benchmark Datasets

Dataset Movies Toys Arts CDs Goodreads Reddit Heterophily Ratio 0.53 0.26 0.34 0.69 0.33 0.04

curacy by over 10%. However, further increasing the number of neighbors yields marginal or unstable gains, indicating that a moderate amount of structural context is optimal. These results highlight the importance of integrating a controlled amount of structural information into the prompt, allowing the LLM to better contextualize the target node during reasoning.

#### 7.9. Variance Analysis

Following prior GraphLLM studies, we initially omitted variance reporting. However, our experiments reveal that the variance of our method is relatively small—typically around ±0.07 or ±0.14 across three random runs. For reference, Table 11 summarizes partial variance scores on representative datasets and tasks (Metric: Accuracy).

Table 11. Partial variance results of Mario across datasets and tasks over 3 runs.

Method Movies (NC) Arts (LP)

Mario (Single Focus) 53.63 ± 0.07 89.96 ± 0.14 Mario (Mix Training) 50.98 ± 0.08 92.60 ± 0.12

#### 7.10. Quantitative Analysis of Modality Preference

This subsection explains the statistics in the Venn diagram of Figure 1. Specifically, the six numbers in Figure 1(b) can be grouped into three categories: (i) the proportion of nodes that can be correctly classified only by the template corresponding to a single modality; (ii) the proportion of nodes that can be correctly classified only when two templates are both correct (rather than counting a node as correct if either template is correct); and (iii) the proportion of nodes that can be correctly classified by all three template types. The proportions in the first category are 2.65%, 2.25%, and 2.05%; those in the second category are 7.71%, 7.40%, and

- 6.98%; and the third category accounts for 70.96%. These numbers sum to 100%. Therefore, the statement that “about 30% of nodes cannot be correctly classified by all templates jointly” is computed as 100%−70.96% = 29.04%, which is approximately 30%. All percentages are normalized within the set of nodes correctly classified by at least one template.
- 7.11. Robustness against Varying Heterophily

To reduce overfitting to locally uniform neighborhoods and to expose the model to richer semantic context, we adopt

a multi-hop neighbor selection strategy. Expanding the receptive field beyond immediate neighbors allows Mario to retrieve distant yet relevant nodes, so the router is not forced to rely solely on short-range label similarity when forming prompts. To quantify structural diversity in our benchmarks, we compute each graph’s heterophily ratio, defined as the fraction of edges linking nodes with different labels.

Despite the heterophily ratios varying widely across datasets—from near-homophilic graphs such as Reddit (0.04) to strongly heterophilic ones like CDs (0.69) and Movies (0.53)—Mario consistently maintains strong performance. This suggests that our Stage-1 feature–based similar neighbor selection remains reliable across different structural regimes, confirming that it generalizes well even when local neighborhoods are not label-coherent. Importantly, while this finding is complementary to Observation 7 in the paper, it addresses a different question: here we show robustness of the selection strategy under varying heterophily, rather than characterizing the spatial pattern of modality preferences within the graph.

#### 7.12. Training Compute Analysis

We compare Mario’s Stage 1/2 with all baselines on identical data under a compute-matched budget (Table 12), which reports the training cost and resulting performance on our main datasets. To ensure that every method had sufficient opportunity to converge, we initially capped each Stage-1 run at 2 GPU-hours on A100 SXM4 80GB GPUs and terminated runs that remained unconverged at the cap. In practice, however, we observed that the GVLM and all graphbased baselines consistently converged within 1 GPU-hour, with no notable gap in training overhead across methods, indicating that our comparisons are conducted under a largely fair and compute-balanced setting. We further include a new Tuned CLIP baseline for a stronger reference; since its optimization is typically more compute-intensive than GVLM/GNN-style training, we set its compute cap to 1 GPU-hour as well to maintain fairness given that the other methods already converge within this budget. As shown in Table 12 (the lower part), MAPR converges in roughly half the epochs of the baselines (stage2), with an average total runtime only 0.25 (rather than 3×) GPU-hours higher. Finally, the resulting average accuracies under these compute caps closely match those reported in Table 4 and Figure 4, with no noticeable discrepancies.

Table 12. Detailed Cost Breakdown (Stage 1 & 2).

Method Arts Reddit Movies #Epoch Avg Acc(%) Tuned CLIP/Other Baselines vs. GVLM (stage1) (Columns 2–4: GPU-hours)

|GCN SAGE GATv2 MLP| |0.95 0.91 0.93 0.90|0.94 0.90 0.91 0.87<br><br>|0.88 0.88 0.89 0.85|36.9 34.3 33.0 37.0<br><br>|78.17 ↑2.79<br><br>77.98 ↑3.04<br><br>78.08 ↑2.92<br><br><br>77.78 ↑3.30|
|---|---|---|---|---|---|---|
|Tuned CLIP| |0.99<br><br>|0.98|0.99<br><br>|22.3<br><br>|78.01 ↑3.00|
|GVLM| |0.92|0.91|0.88|23.0|80.35|

Single-template Variants vs. MAPR (Stage2) (Columns 2–4: GPU-hours)

|Text-only Image-only Text+Image| |5.65 5.65 5.77|4.02 4.58 4.09<br><br>|4.11 4.11 4.23<br><br>|6.3 6.6 6.3|78.56 ↑2.28 78.93 ↑1.80 78.50 ↑2.36<br><br>|
|---|---|---|---|---|---|---|
|Mario (MAPR)| |5.82|4.15|4.35|3.0|80.35|

#### 7.13. Prompt Template

The prompt templates used for adaptive multimodal graph instruction tuning in the two multimodal graph reasoning tasks, node classification and link prediction, are shown in Table 13. Since the templates for different modalities are broadly similar, differing mainly in which modality-specific features of the anchor node and its neighbors are embedded—we present the template for the text+image case as an illustrative example.

#### 7.14. Case Study

To better understand how Mario behaves on multimodal graphs, we conduct a qualitative case study on both tasks. We compare Mario-8B against three strong closed/API-based L(V)LMs—ChatGPT-5.1-Thinking, Gemini-3Pro, and Qwen3-Max—on several representative nodes and node pairs drawn from the Movies, Toys, and CDs graphs (Figs. 9–20). Because these models are accessed only through high-level APIs, we cannot inject special feature tokens when prompting as we do for Mario. Instead, we adopt a uniform and conservative prompting protocol: for each case, we present the anchor node (or node pair) together with its neighbors using the same high-level templates as in Table 13, and we input each neighbor’s raw text and image jointly to ensure a fair comparison.

For the node classification case shown in Figs. 11–12, the anchor node’s text describes the content of a lecture series, whereas the associated image focuses almost entirely on after-sales information (lifetime warranty and replacement policy) and provides very little semantic signal about the lecture itself. Mario’s MAPR, conditioned on both the anchor’s multimodal features and its local subgraph, correctly infers that the image is not the preferred modality for this classification task and routes the node through a text-centric template. This decision matches the underlying graph semantics and illustrates that the router is able to down-weight visually salient but task-irrelevant informa-

tion.

Across all the illustrated cases, Mario’s behavior is consistently competitive with, and sometimes superior to, strong closed-source L(V)LMs. In several examples, all closed models converge to the same intuitive but graphinconsistent label, while Mario is the only method that predicts the correct class—for instance, in the case of Figs. 9–10, where Mario is the only model that assigns the ground-truth category and all other systems fail. These qualitative results further corroborate that Mario is an effective and reliable framework for multimodal graph reasoning.

Table 13. Prompt Templates for Node Classification and Link Prediction Tasks. Note that this template is designed to include both text and image features of the node. If the input is text-only or image-only, simply retain the corresponding single modality feature.

##### Task Prompt Template

Node Classification I’m starting a node classification task in the <dataset>. Each node represents a <product> with text and image features, and edges indicate <relationship>. Given a target node, the raw text is ..., the text feature is <text feature> and the image feature is <image feature>. The neighbors are described in the following template: <text feature>, <image feature>, and <label>.

- It has the following neighbors at hop 1:

- N1: <1-hop neighbor 1 text feature>, <1-hop neighbor 1 image feature>, <1-hop neighbor 1 label>
- N2: <1-hop neighbor 2 text feature>, <1-hop neighbor 2 image feature>, <1-hop neighbor 2 label>
- N3: ...........

- It has the following neighbors at hop 2:

- N1: <2-hop neighbor 1 text feature>, <2-hop neighbor 1 image feature>, <2-hop neighbor 1 label>
- N2: <2-hop neighbor 2 text feature>, <2-hop neighbor 2 image feature>, <2-hop neighbor 2 label>

...... Based on the information provided, please classify the target node into one of the following categories: {all categories}.

Link Prediction I’m starting a link prediction task in the <dataset>. Each node represents a <product> with text and image features, and edges indicate <relationship>. Given the two nodes:

- Node 1: The raw text is ... the text feature is <text feature>, and the image feature is <image feature>.
- Node 2: The raw text is ... the text feature is <text feature>, and the image feature is <image feature>.

The neighbors of node 1 (common neighbors with node 2) are described in the following template: <text feature>, <image feature>.

- It has the following neighbors at hop 1 (Directly connected):

- N1: <1-hop neighbor 1 text feature>, <1-hop neighbor 1 image feature>
- N2: <1-hop neighbor 2 text feature>, <1-hop neighbor 2 image feature>
- N3: ...........

- It has the following neighbors at hop 2 (Indirectly connected by shared neighbors):

- N1: <2-hop neighbor 1 text feature>, <2-hop neighbor 1 image feature>
- N2: <2-hop neighbor 2 text feature>, <2-hop neighbor 2 image feature>

...... Based on the information provided, please determine whether a link exists between the two nodes. Answer ”yes” if a link exists or ”no” if it does not.

Figure 9. A case from the Movies dataset in the NC task that Mario identifies as preferring Text+Image modality information.

|Case Study: Node Classification-Text+Image-Movies|
|---|

Anchor node raw text: In a strange dark age based on Celtic myths, the Divine Empire’s path of conquest seems unstoppable... until a savage priest makes a critical mistake while attempting to resurrect a Demon Lord! Now the scales of fate tip in the other direction.

###### Label list:

’A&E Home Video’ ’Art House & International’ ’BBC’ ’Blu-ray’ ’Boxed Sets’ ’Classics’ ’Criterion Collection’ ’Fully Loaded DVDs’ ’Genre for Featured Categories’ ’HBO’ ’Holidays & Seasonal’ ’Independently Distributed’ ’Movies’ ’Music Artists’ ’Musicals & Performing Arts’ ’Paramount Home Entertainment’ ’Science Fiction & Fantasy’ ’Studio Specials’ ’TV’ ’Walt Disney Studios Home Entertainment’

###### ChatGPT-5.1-Thinking:

Science Fiction & Fantasy ✗ Gemini-3-Pro:

Science Fiction & Fantasy ✗ Qwen3-Max:

Science Fiction & Fantasy ✗ Mario-8B:

Genre for Featured Categories ✓

[Figure 210]

Figure 10. Anchor node’s image in Figure 9.

Figure 11. A case from the Movies dataset in the NC task that Mario identifies as preferring Text-only modality information.

|Case Study: Node Classification-Text-only-Movies|
|---|

Anchor node raw text: Roots of Human Behavior; Description: While human history is usually studied from the perspective of a few hundred years, anthropologists consider deeper causes for the ways we act. Now, in these 12 engrossing lectures, you’ll join an expert anthropologist as she opens an enormous window of understanding for you into the thrilling legacy left by our primate past.

###### Label list:

’A&E Home Video’ ’Art House & International’ ’BBC’ ’Blu-ray’ ’Boxed Sets’ ’Classics’ ’Criterion Collection’ ’Fully Loaded DVDs’ ’Genre for Featured Categories’ ’HBO’ ’Holidays & Seasonal’ ’Independently Distributed’ ’Movies’ ’Music Artists’ ’Musicals & Performing Arts’ ’Paramount Home Entertainment’ ’Science Fiction & Fantasy’ ’Studio Specials’ ’TV’ ’Walt Disney Studios Home Entertainment’

ChatGPT-5.1-Thinking:

TV ✗

###### Gemini-3-Pro:

Genre for Featured Categories ✓ Qwen3-Max: Movies ✗

###### Mario-8B:

Genre for Featured Categories ✓

[Figure 211]

Figure 12. Anchor node’s image in Figure 11

Figure 13. A case from the Movies dataset in the NC task that Mario identifies as preferring Image-only modality information.

|Case Study: Node Classification-Image-only-Movies|
|---|

Anchor node raw text: Castle: Season 6; Description: After Castle’s stunning romantic proposal to Beckett, what happens next? TV’s magnetic crime-fighting couple faces a whole new set of challenges as they juggle wedding plans and their most intriguing cases yet in ABC’s CASTLE: THE COMPLETE SIXTH SEASON. Beckett’s new job with the Justice Department takes her away from the wisecracking love of her life. But Castle’s devotion to his new fiancee -- and her fascinating line of work -- jeopardizes her career and creates a chain of events that might separate them forever. Back on the home front, Castle is none too pleased to discover his daughter has seemingly been captivated by, and now living with, her new, free-spirited boyfriend.

###### Label list:

’A&E Home Video’ ’Art House & International’ ’BBC’ ’Blu-ray’ ’Boxed Sets’ ’Classics’ ’Criterion Collection’ ’Fully Loaded DVDs’ ’Genre for Featured Categories’ ’HBO’ ’Holidays & Seasonal’ ’Independently Distributed’ ’Movies’ ’Music Artists’ ’Musicals & Performing Arts’ ’Paramount Home Entertainment’ ’Science Fiction & Fantasy’ ’Studio Specials’ ’TV’ ’Walt Disney Studios Home Entertainment’

ChatGPT-5.1-Thinking:

TV ✗

Gemini-3-Pro:

TV ✗

Qwen3-Max:

TV ✗

###### Mario-8B:

Boxed Sets ✓

[Figure 212]

Figure 14. Anchor node’s image in Figure 13

Figure 15. A case from the Toys dataset in the LP task that Mario identifies as preferring Text+Image modality information.

|Case Study: Link Prediction-Text+Image-Toys|
|---|

###### Node pair raw text:

- Node 1: The Crazy Scientist series, a collection of science tricks, was created by a joint venture of 2 crazy scientists and the Purple Cow. A combination bound to create an excellent and yet crazy experience! The Crazy Scientist Young Researches is a set of science tricks for kids to try out and discover 20 fun and fascinating facts about the world around you. Create excellent and crazy experiences that can be enjoyed by the entire family! Each science trick comes with a simple yet clever scientific explanation. A perfect STEAM gift! Challenge your brainpower and make intriguing discoveries about the world around us. Have fun experimenting and learning with the Young Researches amazing activities. Provide children the opportunity to become real researchers and follow easy instructions of science experiments that can be conducted using common household materials. Whats included? The box contains 20 activity cards with detailed instructions. Recommended for children ages 6 and up. Some science tricks may require adult supervision as indicated.
- Node 2: Learn the scientific principles behind optical illusions with the 4M Illusion Science kit. Experiment with 20 classic optical illusions included in this kit. The kit includes illusion trick cards, spinning top with illusion cards, 3D picture cards, markers, 3D glasses, and more. A 20-page instruction book is included, describing the science of optical illusions and how to create a wide range of illusory effects. Perfect for young scientists with an interest in optics. Recommended for ages 7 years and up.

Label list: ’yes’ The two toys are co-purchased. ’no’ The two toys are not co-purchased.

###### ChatGPT-5.1-Thinking:

No ✗

###### Gemini-3-Pro:

Yes ✓

###### Qwen3-Max:

Yes ✓

###### Mario-8B:

Yes ✓

[Figure 213]

[Figure 214]

(a) Node 1. (b) Node 2.

Figure 16. Node Pair’s images in Figure 15.

Figure 17. A case from the CDs dataset in the LP task that Mario identifies as preferring Text-only modality information.

|Case Study: Link Prediction-Text-only-CDs|
|---|

###### Node pair raw text:

- Node 1: You Can Do It Yoga for MS Volume 2 DVD; Description: This DVD contains 2 complete classes. The first is a beginner/gentle yoga class. It includes some floor poses and some standing poses along with a guided meditation. Runtime: 54 minutes The second class is a beginner/intermediate yoga class. It includes some floor poses and some standing and balancing poses along with a guided meditation. Runtime: 50 minutes,This DVD contains 2 complete classes.
- Node 2: Thoughts Become Things; Description: You create your own reality and by changing your thoughts, words, and actions in the simplest of ways, you can create fantastic change. - Mike Dooley".

Label list: ’yes’ The two CDs are co-purchased. ’no’ The two CDs are not co-purchased.

###### ChatGPT-5.1-Thinking:

No ✗

###### Gemini-3-Pro:

Yes ✓

###### Qwen3-Max:

No ✗

###### Mario-8B:

Yes ✓

[Figure 215]

[Figure 216]

(a) Node 1. (b) Node 2.

Figure 18. Node Pair’s images in Figure 17.

Figure 19. A case from the CDs dataset in the LP task that Mario identifies as preferring Image-only modality information.

|Case Study: Link Prediction-Image-only-CDs|
|---|

###### Node pair raw text:

- Node 1: Howard Lovecraft And The Frozen Kingdom; Description: After visiting his father in Arkham Sanitarium, young Howard Lovecraft ignores his father&#146s ominous warning and uses the legendary Necronomicon to open a portal to a strange, frozen world filled with horrifying creatures and grave danger. Alone and scared, Howard befriends a hideous creature he names Spot who becomes his companion throughout their treacherous journey across the Frozen Kingdom.
- Node 2: A Serbian Film (Uncut) by Srdjan Todorovic; Description: Milos, a retired adult film star, leads a normal family life with his wife Maria and six-year old son Petar in tumultuous Serbia, trying to make ends meet. A sudden call from his former colleague Layla will change everything. Aware of his financial problems, Layla introduces Milos to Vukmir - a mysterious, menacing and politically powerful figure in the adult film business. A leading role in Vukmir’s production will provide financial support to Milos and his family for the rest of their lives.

Label list: ’yes’ The two CDs are co-purchased. ’no’ The two CDs are not co-purchased.

###### ChatGPT-5.1-Thinking:

No ✓

###### Gemini-3-Pro:

No ✓

###### Qwen3-Max:

Yes ✗

###### Mario-8B:

No ✓

[Figure 217]

[Figure 218]

(a) Node 1. (b) Node 2.

Figure 20. Node Pair’s images in Figure 19

