# arXiv:2509.03405v1[cs.CL]3Sep2025

## LMEnt: A Suite for Analyzing Knowledge in Language Models from Pretraining Data to Representations

### Daniela Gottesmanτ Alon Gilae-Dotanτ Ido Cohenτ Yoav Gur-Ariehτ Marius Mosbachω Ori Yoranτ Mor Gevaτ τ Tel Aviv University ω Mila – Quebec AI Institute & McGill University

gottesman3@mail.tau.ac.il

### Abstract

Language models (LMs) increasingly drive real-world applications that require world knowledge. However, the internal processes through which models turn data into representations of knowledge and beliefs about the world, are poorly understood. Insights into these processes could pave the way for developing LMs with knowledge representations that are more consistent, robust, and complete. To facilitate studying these questions, we present LMEnt, a suite for analyzing knowledge acquisition in LMs during pretraining. LMEnt introduces: (1) a knowledge-rich pretraining corpus, fully annotated with entity mentions, based on Wikipedia, (2) an entity-based retrieval method over pretraining data that outperforms previous approaches by as much as 80.4%, and (3) 12 pretrained models with up to 1B parameters and 4K intermediate checkpoints, with comparable performance to popular open-sourced models on knowledge benchmarks. Together, these resources provide a controlled environment for analyzing connections between entity mentions in pretraining and downstream performance, and the effects of causal interventions in pretraining data. We show the utility of LMEnt by studying knowledge acquisition across checkpoints, finding that fact frequency is key, but does not fully explain learning trends. We release LMEnt to support studies of knowledge in LMs, including knowledge representations, plasticity, editing, attribution, and learning dynamics.

huggingface.co/LMEnt github.com/LMEnt

[Figure 1]

[Figure 2]

### 1 Introduction

Language models (LMs) capture substantial amounts of knowledge and beliefs about the world from their training data (Petroni et al., 2019;

Roberts et al., 2020; AlKhamissi et al., 2022; Andreas, 2022). A fundamental, yet underexplored, question is how such knowledge representations are formed and shaped during pretraining. Namely, what is the interplay between data composition, training dynamics, and the internal knowledge mechanisms in LMs. An understanding of these processes could provide better control over the model’s knowledge, potentially improving its factuality and reasoning.

A key prerequisite for studying the interplay between data and knowledge representations is the ability to locate exactly where specific knowledge appears in the pretraining data. However, such metadata is rarely available for pretraining corpora, and existing tools for locating it rely on post-hoc string-based search (Elazar et al., 2024; Liu et al., 2024a, 2025), which lack robustness against variability in phrasing of semantically equivalent information. For example, consider retrieving information about the American football team the “Buffalo Bills” (Fig. 2). Stringbased approaches are less likely to retrieve relevant documents where the team is referred to as “Buffalo”, “The Bills”, or even “the team”. Further, when expanding queries with name aliases, existing tools often suffer from noisy retrieval, as we observe in §5.2. This noise is due in part to alias ambiguity, failing to differentiate between the team “Buffalo” and the city “Buffalo, NY”.

In this work, we tackle this gap by introducing LMEnt (Language Models with annotated Entity mentions), a suite of pretrained models with full transparency about the locations in training where a specific entity was mentioned. The focus on entity mentions stems from mounting evidence that entity names are central to the construction of knowledge representations in LMs (rather than individual tokens; Li et al., 2021; Meng et al., 2022; Geva et al., 2023; Gottesman and Geva, 2024). LMEnt has three components: (1) 7.3M

Pretraining Document Corpus

|Joshua Patrick Allen (born May 21,<br><br>1996) is an American professional football quarterback for the Buffalo Bills of the National Football League (NFL)...he led the Bills to their first division title and playoff victory since 1995 en route to an AFC Championship Game appearance. … company founded and based in Buffalo, New York. Allen is a spokesperson for the John R. Oishei Children's Hospital in Buffalo…<br><br>|
|---|

LMEnt Entity Mentions

Extracting Entity Mentions

- 1

- 2

|Mention: “Buffalo Bills”<br><br>| |Q221626 Buffalo Bills|
|---|---|
<br><br>📌1.0 🔗0.9 ⬅0 📋0<br><br>Candidate entities:|
|---|

- 3

- 4

|Mention: “he”<br><br>📌0 🔗0.99 ⬅0.99 📋1<br><br>| |Q40435 Josh Allen|
|---|---|
<br><br>Candidate entities:|
|---|

#### LMEnt

An open-source suite for analyzing knowledge in language models

Hyperlink Extraction

Knowledge-rich pretraining data annotated with fine-grained entity mentions

###### Entity-based retrieval from pretraining data

###### Pretrained models with full visibility into knowledge at each training step

- ● 12 models: 3 scales X 4 epochs
- ● 4K intermediate checkpoints
- ● Find all training steps where the model observed information about Buffalo Bills.

- ● Discover all entities at a certain step.

Entity Linking 🔗

Find all data chunks about Buffalo Bills

Joshua Patrick Allen (born May 21, 1996) is an American professional football quarterback for the Buffalo Bills of the National Football League (NFL) He played college football… and was selected seventh overall by the Bills

Entity-based query String-based query

|Mention: “true buffalo”<br><br>📌0 🔗0.99 ⬅0.99 📋1<br><br>| |Q40435 Bubalina|
|---|---|
<br><br>Candidate entities:|
|---|

| |Q221626 Buffalo Bills|
|---|---|

“Buffalo Bills”

|Mention: “Buffalo”<br><br>📌0 🔗0.99 ⬅0.99 📋1<br><br>| |Q40435 Buffalo, NY|
|---|---|
<br><br>Candidate entities:|
|---|

|Of the two surviving species, the American bison, B. bison, found only in North America, is the more numerous. Although colloquially referred to as a buffalo in the United States and Canada,[2] it is only distantly related to the true buffalo.<br><br>|
|---|

📋Coreference Resolution⬅

[Figure 4]

| |Q221626 Buffalo Bills|
|---|---|

[Figure 5]

| |Q40435 Josh Allen|
|---|---|

Q221626 Buffalo Bills

Q1215884 National Football League

Training steps

Figure 1: The LMEnt suite is composed of three components (left) fine-grained entity mentions for every document in the pretraining corpus, (middle) an index that retrieves by the entity QID and outperforms string-based retrieval methods, (right) 12 models trained on 1, 2, 4, and 6 epochs where each step can be mapped to the entities it mentions, and each entity can be traced to the steps that introduce it.

###### LMEnt vs String Retrieval

Pretraining Chunks Training

for as many as 80% of entities, while maintaining > 97% precision as the limit on the number of chunks retrieved increases. Conversely, string-based tools exhibit sharp precision drops, to as low as 27%. Last, we demonstrate the utility of LMEnt by analyzing knowledge acquisition across checkpoints (§6). We find that while learning correlates with fact frequency, the rates of both learning and forgetting increase with frequencya phenomenon not yet fully understood.

fine-grained entity annotations across all documents in a pretraining corpus built on English Wikipedia (Fig. 1, left, §2), (2) an Elasticsearch index with 10.5M chunks which enables retrieval of all the chunks that mention a specific entity by their unique Wikidata identifier (Fig. 1, center, §3), and (3) a collection of 12 LMs with 170M, 600M, and 1B parameters, pretrained on the annotated data with 110 intermediate checkpoints per epoch (Fig. 1, right, §4).

LMEnt retrieves using the entity QID, whereas String Retrieval retrieves by matching the entity name string in text.

|Joshua Patrick Allen (born May 21,<br><br>1996) is an American professional football quarterback for the Buffalo Bills of the National Football League (NFL).<br><br>|
|---|

When did my model see information about the Buffalo Bills?

1

What chunks does my model see about the Buffalo Bills?

|company founded and based in Buffalo, New York. Allen is a spokesperson for the John R. Oishei Children's Hospital in Buffalo<br><br>2|
|---|

1 3

Overall, LMEnt provides novel capabilities for precise tracking of information during pretraining and lays the groundwork for studying the interplay between pretraining data and knowledge representations in LMs. We release LMEnt to the community as an open-source suite on HuggingFace, including the pretraining data, fine-grained entity mentions, pretrained models, and intermediate checkpoints.

To build LMEnt, we annotate each document in the English Wikipedia with fine-grained entity mentions extracted from three complementary sources: Wikipedia hyperlinks, entity-linking and coreference resolution (Fig. 2). We then train LMEnt models on this annotated corpus, which has 0.03%–4.7% of the tokens typically used to train LMs of similar sizes.

3

|he led the Bills to their first division<br><br>title and playoff victory since 1995 en route to an AFC Championship Game appearance.|
|---|

What chunks does my model see about buffalo?

|referred to as a buffalo in the United States and Canada, it is only distantly related to the true buffalo<br><br>4|
|---|

Steps

Experiments show that LMEnt is an effective testbed for studying knowledge acquisition in LMs (§5). On factual question answering (Mallen et al.,

### 2 Labeling the English Wikipedia with Fine-Grained Entity Mentions

- 2023), LMEnt models reach comparable performance (7.4% on all entities, 66% on popular entities) to Pythia-1.4B (Biderman et al., 2023) (8.7%, 67%) and OLMo-1B (Groeneveld et al.,
- 2024a) (10.4%, 66%), with noticeably lower performance than OLMo-2-1B (OLMo et al., 2025) (13.9%, 74%) and SmolLM-1.7B (Allal et al.,
- 2025) (14.9%, 73%) mostly due to recall failures on rare facts. Moreover, LMEnt entitybased retrieval shows clear benefits over stringbased search methods similar to WIMBD (Elazar et al., 2024) and Infinigram (Liu et al., 2024a); LMEnt returns more relevant document chunks

To support analyzing knowledge evolution over training, we select a corpus rich in factual content and annotate its documents with fine-grained entity mentions (Fig. 1, left). These annotations enable tracking exactly which entities a model observes at each training step when training on this data. Next, we explain our choice of Wikipedia for pretraining data, and the annotation process.1

1The annotation pipeline used 8 H100 GPUs (80GB VRAM each). Running coreference resolution and entity linking took 5 days will full GPU utilization.

Document

###### Chunk 1

| |Q221626 Buffalo Bills|
|---|---|

| |Q93565992 John R. Oishei Children’s Hospital|
|---|---|

| |Q40435 Buffalo, NY|
|---|---|

1

Josh Allen Joshua Patrick Allen (born May 21, 1996) is an American professional football quarterback for the Buffalo Bills of the National Football League (NFL). He played college football for the Wyoming Cowboys and was selected seventh overall by the Bills in the 2018 NFL draft. During his Buffalo tenure, he has led the team to a total of six playoff appearances, five consecutive division titles, seven postseason victories, and two conference championship game appearances. Allen had a breakout season in 2020 when he led the Bills to their first division title and playoff victory since 1995 en route to an AFC Championship Game appearance.

⬅ 🔗 📌 📋

2

3

###### Chunk 2

4

|Buffalo|
|---|

company founded and based in Buffalo, New York. Allen is a spokesperson for the John R. Oishei Children's Hospital in

As

📌 H: 0 🔗 EL: 0.99 ⬅ C: 0.99 📋 CC: 1

part of a deal with the hospital, Allen makes appearances, visits patients, and is in commercials to support fundraising efforts. During the 2019 season, Allen donated $200 to the hospital for each of his touchdowns.

4

3

- 1

2

|Mention: “Buffalo”<br><br>📌0 🔗0.98 ⬅1.0<br><br>| |Q40435 Buffalo, NY|
|---|---|
<br><br>Candidate entities:|
|---|

|Mention: “Buffalo Bills”<br><br>| |Q221626 Buffalo Bills|
|---|---|
<br><br>📌1.0 🔗0.9 ⬅1.0<br><br>Candidate entities:|
|---|

|Mention: “the team”<br><br>📌0 🔗0.0 ⬅1.0<br><br>Candidate entities:<br><br>| |Q221626 Buffalo Bills|
|---|---|
|
|---|

|Mention: “Buffalo”<br><br>| |Q221626 Buffalo Bills|
|---|---|
<br><br>📌0.0 🔗0.0 ⬅1.0<br><br>Candidate entities:|
|---|

📌 0 🔗 0 ⬅ 0.25 📋 1

Buffalo Bills

5

|Mention: “the hospital”<br><br>| |Q93565992 John R. Oishei Children’s Hospital|
|---|---|
<br><br>| |Q40435 Buffalo, NY|
|---|---|
<br><br>📌0 🔗0 📋0.96<br><br>📌0 🔗0 📋0.04<br><br>Candidate entities:|
|---|

- Figure 2: The document for the entity “Josh Allen” is split into two chunks, which are processed independently during pretraining. In the first chunk, the “Buffalo Bills” (Q221626) is mentioned explicitly—identified through hyperlinks and entity linking—and implicitly, through coreference resolution. Although both Q221626 (Buffalo Bills) and Q40435 (the city of Buffalo) share the surface form “Buffalo”, LMEnt disambiguates them as Mention

Text mention: 'Buffalo Bills' Candidates: [{'qid': 'Q221626'

'scores_by_source': {H: 1.0, EL: 0.9, C: 0, CC: 0}

(2) is linked to Q221626 with 1.0 confidence by coreference resolution, while Mention (4) is linked to Q40435 with 0.98 confidence using entity linking, and the two mentions are placed in separate coreference clusters. Mention (3) “the team” is linked to Q221626 since it is in the same coreference cluster as “Buffalo Bills”.

Pretraining data Wikipedia is a natural choice for the source of pretraining data because it is a knowledge base that is structured around entities, such as people, locations, and world events. Specifically, related entity pages are connected through hyperlinks, which enables us to easily disambiguate entities and map them to their unique identifiers, called a QIDs. Moreover, it provides a snapshot of world knowledge at a given time, minimizing contradicting information. In addition, Wikipedia is a common resource for knowledgefocused benchmarks (e.g., Petroni et al., 2021; Sciavolino et al., 2021; Lewis et al., 2021; Mallen et al., 2023), making it well-suited for tracking how changes in pretraining affect learning.

3. Providing coverage of all mentions across the document, as LMs train on each chunk in isolation and may lack the full document context.

We meet these conditions by leveraging three complementary sources for annotating entity mentions. The first by extracting Wikipedia hyperlinks and applying entity linking; and the second through coreference resolution. To satisfy the third condition, we structure the entity mentions so that each mention’s character span maps to its entity QID. We detail each of these steps next.

##### 2.1 Entity Mention Sources

Hyperlinks To extract hyperlinks, we parse the raw XML Wikipedia dump and extract the url attributes from href tags. Each url links to the Wikipedia article of the corresponding entity. We perform entity disambiguation by mapping the url to its QID using Wikidata.

Annotation objectives Our goal is to be able to query which entities a model saw at any given training step. To achieve this, we design an entitymention annotation process that satisfies the following criteria (see Fig. 2 for illustration):

Entity linking Since hyperlinks are manually added by human editors, they are the most reliable source, but they offer poor coverage within a document.2 To identify additional mentions, we use ReFinED (Ayoola et al., 2022), a state-of-theart modular entity linking system that performs

- 1. Disambiguation between entities with similar names and aliases, e.g., “Buffalo” for the Buffalo Bills team and “Buffalo” for the city of Buffalo, by mapping mentions to Wikidata QIDs (Vrandeˇci´c and Krötzsch, 2014).
- 2. Capturing entities referenced indirectly through descriptive phrases and pronouns, e.g., “the team”, “their” and “the hospital”.

2Wikipedia style guidelines recommend linking only the first mention of an entity in a document: https://en.wikipedia.org/wiki/Wikipedia: Manual_of_Style/Linking.

mention detection, fine-grained typing, and entity disambiguation. ReFinED supports zero-shot entity linking by encoding the mention context and candidate entity descriptions using RoBERTa (Liu et al., 2019), and selecting the entity whose description embedding best aligns with the mention representation. By replacing the underlying base of entity descriptions from our Wikipedia dump, we can identify mentions of new entities and map them to their QIDs without retraining ReFinED.

Coreference resolution The above two sources are used to extract explicit entity mentions, however, entities may still be referred to implicitly, e.g., “the team” in Fig. 2. To fill in this gap, we use the Maverick coreference resolution model (Martinelli et al., 2024), which achieves state-ofthe-art performance on WikiCoref (Ghaddar and Langlais, 2016). To run Maverick at scale, we reduce its memory footprint by replacing the model backbone, allowing us to parallelize inference on documents (see Appendix B.1 for details). Maverick outputs clusters of mentions that refer to the same entity. However, these mentions must still be mapped to their QID, which we discuss below.

##### 2.2 Scoring Entity Mentions in a Document

For every document in the corpus, each of the sources yields a set of entity mentions defined by a character span m = (cstart,cend) (the same mention can be extracted from one or more sources). Here, our goal is to map all mentions in a document to their QIDs, providing the flexibility to chunk the document arbitrarily while being able to identify which chunks contain a given entity.

To create mention-QID mappings, we define three scoring procedures based on the different sources that indicate how confidently a mention can be linked to its corresponding entity (Fig. 2):

- • Hyperlinks (H) identify the first occurrence of a direct entity mention like “Buffalo Bills”. This mention links to the Wikipedia article about the football team and is mapped to QID Q221626 using Wikidata. Since hyperlinks are the most reliable source, we define H(m,Q221626) = 1 if there is a hyperlink in the span of m directing to the Wikipedia page of the entity with QID Q221626, and H(m,Q221626) = 0 otherwise.

[Figure 7]

- • Entity linking (EL) are used to increase coverage of direct mentions like, e.g., “Buf-

[Figure 8]

falo”. The ReFinED model already performs entity disambiguation and provides a score reflecting its confidence that a mention links to a QID, e.g., M(m,Q40435) = 0.98. Therefore, we simply use this score such that EL(m,Q40435) = M(m,Q40435).

• Coref (C) connects indirect mentions, like pronouns, aliases, and generic descriptors, to their entity QIDs. Suppose we are trying to resolve the QID of “the team”. We run Maverick on the document, and it identifies that “the team” is related to the cluster of mentions containing “Buffalo Bills”, “the Buffalo”, “the Bills”, and “their”. How can we leverage the cluster to map “the team” to its QID? We compute a distribution of scores over all QIDs already mapped to some mention in the cluster. In this case, Q221626 is the only QID supported by this cluster so C(“the team”,Q221626) = 1.0. Since this score is computed using the entire cluster, the C score is the same across all mentions in the cluster. Occasionally, a coreference mention encapsulates multiple entities, as in “John R. Oishei Children’s Hospital Buffalo”, which leads to a score distribution over several entities. In this case of mapping ambiguity, we rely on textual similarity of mentions in the cluster to promote one entity over another. §B.2 provides the formal definition of this score, and a detailed explanation of how we resolve mapping ambiguity.

[Figure 9]

The final entity mention structure maps each mention m to a list of candidate entity QIDs with up to three scores per candidate. Keeping three scores per mention affords us flexibility to filter chunk retrieval based on source and confidence thresholds (§3.2). This list structure also accommodates cases where a span encompasses multiple entities or belongs to multiple coreference clusters. Also, by mapping character spans to QIDs, we can chunk the document arbitrarily and identify which entities are referenced by each chunk. In §B.3, we include a qualitative analysis of error patterns over a random sample of 112 mentions to justify the design of the scores.

### 3 Entity-based Retrieval from Pretraining Data

Now that the documents are annotated with entity mentions (Fig. 1, left), we need to process them

into chunks of information for pretraining. During this processing, we propagate the relevant annotations, allowing us to infer the entities mentioned at the particular training step when a chunk is introduced (Fig. 1, right). In §3.1, we describe how documents are processed into chunks, and in §3.2, we discuss our index for efficient retrieval of all chunks (and consequently, training steps) mentioning an entity (Fig. 1, center).

##### 3.1 Data Processing for Training

Tokenization Each document is tokenized using the dolma2-tokenizer (Soldaini et al., 2024). This results in a dataset with a total of 3.6 billion unique tokens, which is less than 0.09% of the tokens used to train OLMo-2 (OLMo et al., 2025).

Chunking A chunk is a sequence of tokens that the model independently processes per step. Unlike existing chunking strategies, like concat-andchunk used in Liu et al. (2019); Ott et al. (2019); Touvron et al. (2023a,b), we ensure that each chunk only contains content from a single document using the Variable Sequence Length Curriculum (Pouransari et al., 2024). This prevents the model from learning spurious correlations between unrelated documents.

Chunking curricula split documents at some sequence length, which can result in entity mentions being split across chunk boundaries. To prevent splitting an entity mention, we modify the chunking logic to terminate just before a mention and pad the remaining tokens to the desired length. These design choices aim to reduce artifacts when training for knowledge recall. The entity mentions for a chunk are defined as those fully contained within its boundaries. Practically, we extend the OLMo-Core framework (OLMo et al., 2025) to retrieve chunks with their entity mentions and adjust mention span indices to be relative to chunk boundaries.

##### 3.2 Chunk Retrieval

To support retrieval of chunks, we build an Elasticsearch index (Banon, 2010) of all chunks in the corpus. Each entry in the index includes the chunk ID, the chunk text, and associated entity mentions. The chunk ID is traceable to the specific training step at which it was introduced, allowing us to retrieve all chunks seen by an intermediate model checkpoint. We can also match an entity mention on its QID, source, and three scores, allowing us

to retrieve all chunks that mention certain entities, and the flexibility to tune thresholds for the scores.

Retrieval algorithms typically apply an ordering scheme to the items they return (Robertson et al., 1995). To accommodate this, we assign a score to each chunk based on the entity mentions that match the retrieval query. For every matching mention, we compute a weighted average of its three source scores defined in §2.2. The score used to rank the chunk is the highest of these aggregated scores among its matching mentions.

### 4 The LMEnt Suite

With the procedures for annotating entity mentions and processing pretraining data established, we now provide an overview of the resulting pretraining data index and LMEnt models.

Pretraining data index Our pretraining data includes 3.6B tokens over 10.5M chunks, and features more than 7M different entities. The 400M mentions extracted are composed of 115M mentions from hyperlinks, 203M from entity linking, and 310M from coreference resolution. Additional statistics are provided in §A.1, Tab. 2.

Pretrained models We introduce a collection of models trained on the annotated dataset, which serve as a testbed for analyzing knowledge evolution over training. We train models of three different sizes, 170M, 600M, and 1B parameters, based on the OLMo-2 architecture (OLMo et al., 2025). Notably, the 170M model has the computeoptimal size for 3.6B tokens (Hoffmann et al., 2022). Each model is trained for 1, 2, 4, and 6 epochs (3.6B, 7.2B, 14.4B, 21.6B tokens, respectively), resulting in four variants per size. Each epoch initializes a different batching and ordering of the chunks. We also release intermediate checkpoints every 1,000 training steps, yielding 110 checkpoints per epoch and proportionally more for longer training schedules. Additional details on model training are in §B.4.

### 5 Experiments

In the previous sections, we described the construction and composition of the LMEnt suite. Here, we establish the credibility of the framework by showing that LMEnt models match the knowledge recall performance of existing open-source models (§5.1), and that entity-based chunk re-

PopQA (1K+ Subject Answer Chunks Shared)

0.8

2-1B

1.4B 1B

1.7B

0.7

1B-6E

1B

360M

600M-6E

0.6

135M

410M

Accuracy

0.5

1B-20K-84B

Model Family

0.4

LMEnt

170M-6E

0.3

OLMo

1B-10KS

Pythia

160M

0.2

SmolLM2

0.1

Baselines

1B-0S

0.0

10 3 10 2 10 1 100 101 102 103 FLOPs (trillions, log scale)

- Figure 3: Accuracy on popular PopQA entities as a function of compute budget. LMEnt models achieve comparable performance to other models with better compute efficiency.

trieval using LMEnt mentions outperforms stringbased methods (§5.2).

##### 5.1 Model Performance

Experimental setup To evaluate the LMEnt models, we use benchmarks that test for knowledge in Wikipedia. We choose two widely adopted benchmarks: PopQA (Mallen et al., 2023) and PAQ (Lewis et al., 2021). PopQA contains questions about 11K entities with varying popularities, and PAQ has 65M questions generated from Wikipedia passages. To keep computational costs manageable we subsample PAQ and only consider samples containing the same subject entities found in PopQA. This results in an evaluation set of approximately 70K questions.3 Results on PAQ are found in §A.2. In addition to knowledgecentric tasks, we report results on commonsense reasoning, multiple choice, and reading comprehension tasks in §A.4. Since LMEnt models are not instruction-tuned, we convert all samples into cloze-style prompts, such that the answer is the next expected phrase. For example, the original QA-style query “what tv network does funny or die presents air on” translates to “Funny or Die Presents airs on the TV network”. Full details of this conversion process are in §B.6.

Baselines We evaluate the performance of LMEnt against leading open-source models of comparable sizes, including Pythia (160M, 410M, 1B, 1.4B) (Biderman et al., 2023), OLMo (1B, 120K-84B) (Groeneveld et al., 2024a), OLMo-21B (OLMo et al., 2025), and SmolLM2 (135M, 360M, 1.7B) (Allal et al., 2025). We include

3Evaluating a 1B parameter model on all PAQ questions using one H-100 80GB GPU would take 200 days.

0.7

LMEnt-1B

0.6

MeanAccuracy

LMEnt-600M LMEnt-170M

0.5

0.4

0.3

0.2

0.1

0.0

1 10 10 100 100 1K 1K+ Subj+Answer Chunks Bins

Figure 4: Mean accuracy on PopQA questions binned according to the number of chunks that the subject and answer entities co-occur in. Increasing model size helps learning associations between entities that appear more frequently in the same chunk.

OLMo-1-20K-84B—an intermediate checkpoint of OLMo-1B trained for 20K steps on 84B unique tokens—because it is the most comparable baseline to LMEnt models in terms of training token count.4 We also include two intermediate LMEnt baselines: LMEnt-1B-0E (randomly initialized) and LMEnt-1B-0.1E (trained for 10K steps). These serve to illustrate how more training data improves knowledge recall capabilities.

Knowledge of LMEnt models is on par with models trained on up to 3K times more data Fig. 3 shows the performance of LMEnt models and the baselines as a function of compute budget, measured in FLOPs. LMEnt achieves comparable performance on popular entities in PopQA to both the Pythia and OLMo-1B models, despite being trained with two orders-of-magnitude less compute. We also present results for all popularity levels in the §A.2 which show similar trends. Though, since PopQA is dominated by tail entities, overall accuracy is lower across all models.

Notably, the LMEnt 600M model surpasses the OLMo-1B intermediate checkpoint and the Pythia-1B checkpoint, which reaffirms that pretraining on Wikipedia enables models to acquire substantial factual knowledge (Devlin et al., 2019) and achieve better compute-performance tradeoffs. This suggests that LMEnt models provide a useful testbed for studying knowledge in LMs, with implications for models trained on substantially larger amounts of data.

Measuring the effect of model scale on knowledge encoding Since all LMEnt models are pre-

4Notably, OLMo-1-20K-84B is trained on 4 times more data than our largest LMEnt-1B model, and it may not observed all the facts we evaluate on.

Retrieval Match Examples

QID + Scores Buffalo, New York; Buffalo; the Queen City; this city; the City’s; the City of Buffalo; the city Buffalo; a town on the banks of Lake Erie; his hometown

LMEnt

Canonical Buffalo, New York Extended

CS-SS

Buffalo, New York; Buffalo, NY; Buffalo, N. Y.; City of Light; The Queen City; The Nickel City

Canonical buffalo, new york Extended

CI-SS

buffalo, new york; buffalo, ny; buffalo, n. y.; city of light; the queen city; the nickel city

Table 1: Example matched mentions for the entity “Buffalo, New York” (Q40435) across different methods. For LMEnt, a mention is considered a match if one of its candidate entities has the QID Q221626 and satisfies at least one of the score thresholds. The Canonical method matches only the entity’s canonical name, whereas the Extended method matches any of the entity’s aliases.

trained on the same data, we can isolate the effect of model size on knowledge acquisition (Roberts et al., 2025). Fig. 4 shows that increasing model size improves learning popular of facts (subject and answer co-occur in ≥ 100 chunks), while having little effect on tail facts (1–100 chunks). We leave exploring how scaling model size further can enhance knowledge learning for future work.

##### 5.2 Chunk Retrieval Performance

A key use-case of the LMEnt entity annotations is retrieving all the training data chunks that mention a specific entity. Here, we compare the quality of entity-based retrieval using LMEnt mentions to existing string-based retrieval methods.

Experimental setup We evaluate on a test set of 1K entities, chosen via stratified sampling by their number of hyperlinks. We use the LMEnt index to retrieve their corresponding chunks, matching both on the entity’s QID and on at least one of the score thresholds: H = 1, EL ≥ 0.6, C ≥ 0.6. These thresholds were empirically determined using a dev set of 60 entities, described in §A.5. We then compare the set of retrieved chunks to those retrieved by popular string-based methods. To evaluate retrieval precision, we use Gemini 2.5 Flash (Comanici et al., 2025) as a judge (Gu et al., 2024; Li et al., 2024) which predicts “Yes” or “No” based on whether the mention directly relates to the target entity. The prompt used and the statistical test justifying our use in an LM-

LMEnt wins

Tie

Baseline wins

| |
|---|

| |
|---|

| |
|---|

CI-SS Canonical CI-SS Expanded

|66.7| | | |12.6| |20.7| | |
|---|---|---|---|---|---|---|---|---|
|80.4| | | | | |9.9| |9.7|
|66.3| | | |12.3| |21.4| | |
|79.1| | | | | |9.8|11.1| |
|27.3|43.9| | | |28.9| | | |
|43.4| |23.3| |33.3| | | | |
|52.1| | |18.6| |29.3| | | |

CS-SS Canonical CS-SS Expanded

LMEnt -C LMEnt -EL -C

LMEnt -H -C

Figure 5: Pairwise wins rates for LMEnt with multiple string-based methods and ablated LMEnt variations. LMEnt outperforms string-based methods by 66.7%– 80.4%. Ablations (bottom three rows) show that hyperlinks and entity linking are the most crucial components of LMEnt.

judge are provided in §B.5. Since some entities are mentioned very frequently (Fig. 6), we randomly sample 100 chunks from the set retrieved per method and entity, and measure the absolute number of chunks judged as “Yes” for each entity.

Baselines We compare to string-based baselines that retrieve chunks using either casesensitive (CS-SS) or case-insensitive (CI-SS) exact matches of entity names within the chunk text. The Canonical variant matches only on the entity’s canonical name, and the Extended variant additionally matches against the entity’s Wikidata aliases which is a common strategy used in prior work to improve recall (e.g., Cohen et al., 2024; Yang et al., 2024).

The CS-SS Canonical baseline approximates the Infinigram (Liu et al., 2024a) tool which matches based on case-sensitive n-grams, and CI-SS Canonical baseline approximates WIMBD (Elazar et al., 2024) which retrieves on exact case-insensitive string matches. Although CS-SS Canonical relies on exact string matches rather than n-gram matches, we show in §5.2 that shorter references (e.g., the ngram “Buffalo”) introduce noise as they retrieve chunks referencing the city, animal, and football team together. Tab. 1 summarizes the retrieval strategies used by LMEnt and the baselines. For clarity, we also include example mentions that triggered the retrieval of a chunk for the entity “Buffalo, NY” (Q40435).

Entity-based retrieval largely outperforms the baselines Fig. 5 presents pairwise win rates between retrieval methods, where a win is defined as one method retrieving more “Yes”-judged chunks than another for a given entity. LMEnt con-

NumberofChunksRetrieved

Method LMEnt CI-SS Canonical CS-SS Canonical CS-SS Expanded CI-SS Expanded

- 100

- 101

- 102

- 103

- 104

- 105

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Tail (H <= 10)

Torso (10 < H <= 1K)

Head (H > 1K)

Figure 6: Distribution of the number of chunks retrieved by LMEnt, and both Canonical and Expanded variants of CI-SS and CS-SS. Retrieval was performed for 1K PopQA entities, selected via stratified sampling based on hyperlink counts. LMEnt retrieves more chunks than the Canonical variants for torso and tail entities, which together account for 99.7% of all entities in Wikipedia. The higher number of chunks returned for head entities by Expanded variants is likely bloated by noisy retrieval (Fig. 7).

sistently outperforms all other methods, retrieving more correct mentions for 66.3%–80.4% of the entities. Surprisingly, the Expanded variants of both CI-SS and CS-SS perform worse than their canonical counterparts by 40%, suggesting that additional noise introduced by alias expansion outweighs any gains in recall (§A.6, Fig. 13). Additionally, we observe that LMEnt is superior to string-based methods across all entity frequency levels—and often by a substantial number of chunks (§A.7, Fig. 14).

Ablating entity mention sources reduces retrieval quality We also ablate components of LMEnt and examine the impact on win rate (Fig. 5). Hyperlinks are the most valuable component as relying solely on them (LMEnt -EL -C) wins on 33.3% of entities. In contrast, using only entity linking (LMEnt -H -C) lowers wins to 29.3%. This indicates that while hyperlinks play a crucial role, entity linking alone remains reasonably effective—highlighting the potential of extending LMEnt to pretraining corpora beyond Wikipedia, where hyperlinks are unavailable. Removing coreference resolution has little effect on LMEnt’s performance, because judging such chunks in isolation is difficult. For example, in the chunk referring to Donald Trump solely as “his Mexico City Policy”, it is difficult to identify that “his” implicitly refers to Trump without context from earlier in the document which mentions him explicitly.

1.0

0.8

Precision@k

0.6

k

0.4

1 5 10 100 1K

10K 25K 50K 100K

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.2

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.0

LMEnt CS-SS Canonical

CI-SS Expanded

Figure 7: Precision at various retrieval depths (k), the top-k chunks retrieved per entity. As the depth increases, LMEnt maintains above 97% precision, while CS-SS Canonical and CI-SS Expanded consistently decrease to 84% and 27%.

LMEnt provides better coverage for rare entities Fig. 6 shows the distribution over the total number of retrieved chunks per entity. We observe that LMEnt offers greater coverage for tail and torso entities compared to Canonical stringbased variants. This is especially significant since tail and torso entities together make up 99.7% of the total entities in the corpus (§A.1, Tab. 2). While it appears that Expanded string-based variants retrieve more chunks, this is likely noise due to alias ambiguity. There is also a noticeable difference in variance—specifically string-based methods show egregious retrieval failures, sometimes retrieving less than 10 documents for head entities and no documents for torso entities.

LMEnt maintains high precision as the number of retrieved chunks increases We also analyze the precision of retrieved chunks at varying retrieval depths, k. For each method, we consider only the top-k retrieved chunks per entity, and randomly sample 100 chunks if k ≥ 100 for feasibility. An LLM then judges the mention, and precision is computed as the fraction of "Yes" responses out of k. As shown in Fig. 7, LMEnt maintains high precision ≥ 97% as k increases, while precision substantially declines for all other methods, reaching 27% and 84% for CI-SS and CS-SS, respectively. This indicates that as more chunks are retrieved, string-based methods introduce noise while LMEnt maintains high quality.

Entity co-occurence better aligns with model performance than popularity The number of Wikipedia pageviews for a subject entity, denoted as Subj Pop, has been the standard way to proxy real-world popularity and is known to correlate with a model’s ability to recall facts about it

1K+

100K+

10K 100K

PopularityBins

100 1K

ChunkBins

1K 10K

Subj Chunks

10 100

Answer Chunks

100 1K

Subj+Answer Chunks

Subj Pop

Answer Pop 1 10

1 10

0.0 0.2 0.4 0.6 Mean Accuracy

Figure 8: Accuracy for LMEnt-1B-6E model on PopQA. Subj+Answer Chunks counts chunks that mention both the subject and answer entities of a question. Answer Chunks and Subject Chunks count chunks that mention the answer and subject entities individually. Subj Pop and Answer Pop are pageview popularities from Mallen et al. (2023). Subj+Answer Chunks correlates best with model behavior.

(Mallen et al., 2023). In Fig. 8, we compare model performance against several potential indicators. In PopQA, each question includes both a subject entity and an answer entity. Leveraging the LMEnt index, which can retrieve chunks containing many co-occurring entities, we introduce a new indicator: Subject-Answer Chunks. This is defined as the number of chunks in which the subject and answer entities co-occur. Intuitively, this co-occurence shows the strongest correlation with model performance. Interestingly, this trend is also consistent for models trained on pretraining data that extend beyond Wikipedia (§A.3, Fig. 12).

### 6 Knowledge Acquisition in Pretraining

We demonstrate the utility of LMEnt for knowledge analysis in LMs, focusing on the question of when during training do models learn knowledge best and how this relates to fact frequency.

Experimental setup To study this, we use the facts in PopQA and evaluate the LMEnt-1B model every 20K steps. Since entity mentions do not specify the relations linking entities in a chunk, we define a fact by a (Subject, Answer) entity pair. A fact is “learned” if the model correctly answers at least one question that links the subject to the answer. Between training steps (X,X + 20K), we consider all the facts seen in the training interval and measure (a) fact frequency: the number of chunks in this interval where the subject and answer co-occurred (Subject+Answer Chunks), (b) % of learned facts: facts that were not learned at

Figure 9: % of facts learned (top) and forgotten (bottom) by fact frequency between intermediate checkpoints of LMEnt-1B-6E. While fact frequency correlates with gains in knowledge, it is unclear why rates of both learning and forgetting increase with frequency.

step X and then learned at step X + 20K, and (c) % of forgotten facts: facts that were learned at step X and then not learned, i.e. “forgotten”, at step X + 20K.

Findings Fig. 9 shows that in the first 20K steps of training, the model acquires a substantial amount of knowledge, and continues to learn and forget facts as training proceeds. Interestingly, the rates of both learning and forgetting increase with fact frequency, leading to higher net gains in knowledge (§A.8, Fig. 15). The model seems to learn facts across all frequencies in intermediate steps. Moreover, knowledge acquisition is not correlated with learning rate, which decreases after 1K warmup steps, as acquisition is highest in the last training steps. Overall, this suggests that while knowledge acquisition is correlated with fact frequency, the interplay between internal mechanisms and training dynamics is yet to be fully understood.

### 7 Related Work

Factual knowledge in language models Focusing on injecting factual knowledge into language models, several previous works combined learning from explicit knowledge graphs and textual data (Ahn et al., 2016; Yang et al., 2017; Logan et al., 2019; Peters et al., 2019; Xiong et al., 2020; Zouhar et al., 2021; Zhao et al., 2025, inter alia). Similar to us, Logan et al. (2019) released a dataset, Linked WikiText-2, linking 2M words from Wikipedia to the Wikidata knowledge graph. While their focus is on improving language modeling performance by accessing the knowl-

edge graph, our focus is on creating a resource that allows to study the acquisition of knowledge purely from language modeling. Another body of related work focuses on evaluating how much factual knowledge LMs possess and how they acquire it during training (Petroni et al., 2019; Elazar et al.,

- 2021; Jiang et al., 2020; Liu et al., 2021; Li et al.,
- 2022; Chang et al., 2024; Allen-Zhu and Li, 2024; Kim et al., 2025, inter alia). Our work is particularly relevant to work on knowledge acquisition and will facilitate future work (see §8.1) on the training dynamics of knowledge acquisition.

Annotating and searching training data Existing tools for studying the effect of training data on model behavior and representations use efficient string-based retrieval that relies on exact matches (Elazar et al., 2024; Liu et al., 2025) or n-gram matches (Liu et al., 2024a). LMEnt uses fine-grained entity annotations for retrieval, allowing a more precise analysis of the role of pretraining data for factuality in language models (§5.2).

Open-source LM toolkits Similar to LMEnt, Pythia (Biderman et al., 2023) OLMo (Groeneveld

- et al., 2024a; OLMo et al., 2025), LLM360 (Liu et al., 2024b), SmolLM (Allal et al., 2025; Bakouch et al., 2025), and NVIDIA (Kuchaiev et al., 2019; NVIDIA, 2025) released a suite of models as well as their training data and training framework to facilitate the study of scaling laws and training dynamics in relation to the pre-training data. LMEnt differs from these previous works in focusing particularly on providing a toolkit to study questions related to (factual) knowledge in language models. 8 Conclusion and Discussion

We introduce LMEnt, a suite of 12 LMs, matching pretraining data annotated with entity mentions, and an entity-based retrieval index that facilitates studying the connection between knowledge acquisition and pretraining data. We show that LMEnt models are capable of knowledge recall tasks, and that our entity-based retrieval outperforms string-based methods on 66.3%–80.4% of entities. Overall, LMEnt serves as a flexible and extensible testbed for investigating a broad set of questions regarding knowledge representations in LMs. In the following sections, we list future applications of LMEnt and limitations.

##### 8.1 Future Applications of LMEnt

Knowledge plasticity and editing LMEnt can be used to investigate the plasticity of knowledge in language models. That is, identifying steps during pretraining when models are more receptive to acquiring new knowledge—following previous works that show that pretrained models struggle to learn new facts (Lyle et al., 2022), or that excessive pretraining can make models resistant to fine-tuning (Springer et al., 2025). Since LMEnt pinpoints where entities appear during pretraining, it can be used to examine a model’s ability to internalize new facts during training.

Improving factuality of LMs LMEnt provides entity annotations for each chunk in the pretraining corpus. A possible extension is using the annotated corpus to improve the model’s factuality by experimenting with different methods for data ordering, or using the annotations to edit the pretraining data, for example by replacing an implicit mention (pronouns, descriptors) with it’s explicit entity name.

Effect of other data sources In this work, we train models on a relatively small and knowledgerich corpus, which results in models that are capable in knowledge tasks, yet perform poorly on out-of-distribution tasks, such as commonsense reasoning (§A.4). Modern LMs are trained on a much larger corpora, often surpassing 10T tokens (Allal et al., 2025), derived from sources that are knowledge-poor, e.g. coding (Rozière et al., 2024; Hui et al., 2024), synthetic stories (Eldan and Li, 2023), and formal languages (Hu et al., 2025). One can easily extend the LMEnt pretraining dataset with such sources to analyze if the addition of these tokens improves factuality. Our approach can also be applied to other knowledgerich sources; while most sources are not equipped with hyperlinks, even relying just on entity linking is still effective (LMEnt -H -C, Fig. 5).

Mechanistic interpretability The enhanced visibility that LMEnt provides into the training process facilitates a controlled, yet natural, setup for studying the formation of latent knowledge representations and circuits in LMs.

##### 8.2 Limitations

Pretraining data, model size and architecture LMEnt presents a relatively small pretraining corpus and collection of compute-efficient models.

Such experimental settings have been successful in recent years, enabling academic labs to develop procedures that were later scaled effectively, e.g., (Rafailov et al., 2023). As mentioned in §8.1, an interesting future direction is to scale LMEnt annotations to other knowledge-rich or knowledgepoor sources—thereby increasing the size of the pretraining corpus and facilitating explorations into how different types of data affect knowledge acquisition. Also, LMEnt can be easily extended to architectures beyond dense transformers like mixture-of-experts architectures (Dai et al., 2024; Muennighoff et al., 2025; Cai et al., 2025), which is already supported by the OLMo framework (Groeneveld et al., 2024b; Muennighoff et al., 2025).

Beyond pretraining Since models see the vast majority of tokens during pretraining, this stage is critical for understanding how they acquire and represent knowledge. While our study focuses on pretraining, it is typically followed by mid- and post-training phases. In these stages, models are presented with additional high-quality next-token prediction data and fine-tuning examples (OLMo

- et al., 2025; Kumar et al., 2025). Future work could extend LMEnt by adding annotations to the data used for mid- and post-training, and further train LMEnt models on this data.

Dense Retrievers In this work, we do not compare LMEnt retrieval against dense retrievers such as DPR (Karpukhin et al., 2020) and ColBERT (Khattab and Zaharia, 2020), since sparse retrieval methods like WIMBD (Elazar et al., 2024) and Infinigram (Liu et al., 2024a) are the predominant approaches for retrieving relevant information in pretraining corpora.

### Acknowledgments

We are grateful to Maor Ivgi, Alon Mendelson, Yanai Elazar, and Ohav Barbi for their valuable discussions and feedback. We also thank Noam Steinmetz, Clara Suslik, and Asaf Avrahamy for their participation in the LM-judge evaluation. This work was supported in part by AMD’s AI & HPC Fund, the Mila P2v5 grant, the MilaSamsung grant, the Google PhD Fellowship program, the Alon scholarship, and the Israel Science Foundation grant 1083/24.

### References

Sungjin Ahn, Heeyoul Choi, Tanel Pärnamaa et al.

2016. A Neural Knowledge Language Model.

Badr AlKhamissi, Millicent Li, Asli Celikyilmaz et al. 2022. A review on language models as knowledge bases.

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch et al. 2025. Smollm2: When smol goes big – data-centric training of a small language model.

Zeyuan Allen-Zhu and Yuanzhi Li. 2024. Physics of language models: part 3.1, knowledge storage and extraction. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org.

Jacob Andreas. 2022. Language models as agent models. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 5769–5779, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Tom Ayoola, Shubhi Tyagi, Joseph Fisher et al. 2022. ReFinED: An efficient zero-shot-capable approach to end-to-end entity linking. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Industry Track, pages 209–220, Hybrid: Seattle, Washington + Online. Association for Computational Linguistics.

Elie Bakouch, Loubna Ben Allal, Anton Lozhkov et al. 2025. SmolLM3: smol, multilingual, long-context reasoner. https:// huggingface.co/blog/smollm3.

Shay Banon. 2010. Elasticsearch: A distributed, RESTful search and analytics engine.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony et al. 2023. Pythia: A suite for analyzing large language models across training and scaling. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 2397–2430. PMLR.

Weilin Cai, Juyong Jiang, Fan Wang et al. 2025. A survey on mixture of experts in large language models. IEEE Transactions on Knowledge and Data Engineering, page 1–20.

Nitay Calderon, Roi Reichart and Rotem Dror. 2025. The alternative annotator test for LLMas-a-judge: How to statistically justify replacing human annotators with LLMs. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16051–16081, Vienna, Austria. Association for Computational Linguistics.

Hoyeon Chang, Jinho Park, Seonghyeon Ye et al. 2024. How do large language models acquire factual knowledge during pretraining? In Advances in Neural Information Processing Systems, volume 37, pages 60626–60668. Curran Associates, Inc.

Roi Cohen, Eden Biran, Ori Yoran et al. 2024. Evaluating the ripple effects of knowledge editing in language models. Transactions of the Association for Computational Linguistics, 12:283–298.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities.

Damai Dai, Chengqi Deng, Chenggang Zhao et al. 2024. DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1280–1297, Bangkok, Thailand. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee et al. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Yanai Elazar, Akshita Bhagia, Ian Helgi Magnusson et al. 2024. What’s in my big data? In The Twelfth International Conference on Learning Representations.

Yanai Elazar, Nora Kassner, Shauli Ravfogel et al. 2021. Measuring and improving consistency in pretrained language models. Transactions of the Association for Computational Linguistics, 9:1012–1031.

Ronen Eldan and Yuanzhi Li. 2023. Tinystories: How small can language models be and still speak coherent english? arXiv preprint arXiv:2305.07759.

Mor Geva, Jasmijn Bastings, Katja Filippova et al. 2023. Dissecting recall of factual associations in auto-regressive language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12216–12235, Singapore. Association for Computational Linguistics.

Abbas Ghaddar and Phillippe Langlais. 2016. WikiCoref: An English coreference-annotated corpus of Wikipedia articles. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), pages 136–142, Portorož, Slovenia. European Language Resources Association (ELRA).

Daniela Gottesman and Mor Geva. 2024. Estimating knowledge in large language models without generating a single token. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 3994– 4019, Miami, Florida, USA. Association for Computational Linguistics.

Dirk Groeneveld, Iz Beltagy, Evan Walsh et al. 2024a. OLMo: Accelerating the science of language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15789–15809, Bangkok, Thailand. Association for Computational Linguistics.

Dirk Groeneveld, Iz Beltagy, Evan Walsh et al. 2024b. OLMo: Accelerating the science of language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15789–15809, Bangkok, Thailand. Association for Computational Linguistics.

Jiawei Gu, Xuhui Jiang, Zhichao Shi et al. 2024. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594.

Pengcheng He, Xiaodong Liu, Jianfeng Gao et al. 2021. {DEBERTA}: {DECODING}{enhanced} {bert} {with} {disentangled} {attention}. In International Conference on Learning Representations.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch et al. 2022. Training compute-optimal large language models. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA. Curran Associates Inc.

Michael Y. Hu, Jackson Petty, Chuan Shi et al. 2025. Between circuits and Chomsky: Prepretraining on formal languages imparts linguistic biases. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9691–9709, Vienna, Austria. Association for Computational Linguistics.

Binyuan Hui, Jian Yang, Zeyu Cui et al. 2024. Qwen2.5-coder technical report.

Zhengbao Jiang, Frank F. Xu, Jun Araki et al. 2020. How can we know what language models know? Transactions of the Association for Computational Linguistics, 8:423–438.

Vladimir Karpukhin, Barlas Oguz, Sewon Min et al. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, Online. Association for Computational Linguistics.

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’20, page 39–48, New York, NY, USA. Association for Computing Machinery.

Jiyeon Kim, Hyunji Lee, Hyowon Cho et al. 2025. Knowledge entropy decay during language model pretraining hinders new knowledge acquisition. In The Thirteenth International Conference on Learning Representations.

Knowledgator. 2025. Flashdeberta. https: //github.com/Knowledgator/ FlashDeBERTa. Accessed: 2025-08-01.

Oleksii Kuchaiev, Jason Li, Daniil Shleifer et al. 2019. Nemo: a toolkit for building ai applications using neural modules. arXiv preprint arXiv:1909.09577.

Komal Kumar, Tajamul Ashraf, Omkar Thawakar et al. 2025. Llm post-training: A deep dive into reasoning large language models.

Patrick Lewis, Yuxiang Wu, Linqing Liu et al. 2021. PAQ: 65 million probably-asked questions and what you can do with them. Transactions of the Association for Computational Linguistics, 9:1098–1115.

Belinda Z. Li, Maxwell Nye and Jacob Andreas. 2021. Implicit representations of meaning in neural language models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1813–1827, Online. Association for Computational Linguistics.

D Li, B Jiang, L Huang et al. 2024. From generation to judgment: Opportunities and challenges of llm-as-a-judge. arxiv.

Shaobo Li, Xiaoguang Li, Lifeng Shang et al. 2022. How pre-trained language models capture factual knowledge? a causal-inspired analysis. In Findings of the Association for Computational Linguistics: ACL 2022, pages 1720– 1732, Dublin, Ireland. Association for Computational Linguistics.

Jiacheng Liu, Taylor Blanton, Yanai Elazar et al. 2025. OLMoTrace: Tracing language model outputs back to trillions of training tokens. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 178– 188, Vienna, Austria. Association for Computational Linguistics.

Jiacheng Liu, Sewon Min, Luke Zettlemoyer et al. 2024a. Infini-gram: Scaling unbounded n-gram language models to a trillion tokens. In First Conference on Language Modeling.

Yinhan Liu, Myle Ott, Naman Goyal et al. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Zeyu Liu, Yizhong Wang, Jungo Kasai et al. 2021. Probing across time: What does RoBERTa know and when? In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 820–842, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Zhengzhong Liu, Aurick Qiao, Willie Neiswanger et al. 2024b. LLM360: Towards fully transparent open-source LLMs. In First Conference on Language Modeling.

Robert Logan, Nelson F. Liu, Matthew E. Peters et al. 2019. Barack’s wife hillary: Using knowledge graphs for fact-aware language modeling. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 5962–5971, Florence, Italy. Association for Computational Linguistics.

Clare Lyle, Mark Rowland and Will Dabney. 2022. Understanding and preventing capacity loss in reinforcement learning. In International Conference on Learning Representations.

Alex Mallen, Akari Asai, Victor Zhong et al. 2023. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9802–9822, Toronto, Canada. Association for Computational Linguistics.

Giuliano Martinelli, Edoardo Barba and Roberto Navigli. 2024. Maverick: Efficient and accurate coreference resolution defying recent trends. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13380–13394, Bangkok, Thailand. Association for Computational Linguistics.

Kevin Meng, David Bau, Alex J Andonian et al. 2022. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems.

Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld et al. 2025. OLMoe: Open mixture-ofexperts language models. In The Thirteenth International Conference on Learning Representations.

NVIDIA. 2025. Nvidia nemotron nano 2: An accurate and efficient hybrid mamba-transformer reasoning model.

Team OLMo, Pete Walsh, Luca Soldaini et al. 2025. 2 olmo 2 furious. CoRR, abs/2501.00656.

Myle Ott, Sergey Edunov, Alexei Baevski et al. 2019. fairseq: A fast, extensible toolkit for sequence modeling. In Proceedings of NAACLHLT 2019: Demonstrations.

Matthew E. Peters, Mark Neumann, Robert Logan et al. 2019. Knowledge enhanced contextual word representations. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 43–54, Hong Kong, China. Association for Computational Linguistics.

Fabio Petroni, Aleksandra Piktus, Angela Fan et al. 2021. KILT: a benchmark for knowledge intensive language tasks. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2523–2544, Online. Association for Computational Linguistics.

Fabio Petroni, Tim Rocktäschel, Sebastian Riedel et al. 2019. Language models as knowledge bases? In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2463–2473, Hong Kong, China. Association for Computational Linguistics.

Hadi Pouransari, Chun-Liang Li, Jen-Hao Rick Chang et al. 2024. Dataset decomposition: Faster LLM training with variable sequence length curriculum. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Rafael Rafailov, Archit Sharma, Eric Mitchell et al. 2023. Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems.

Adam Roberts, Colin Raffel and Noam Shazeer. 2020. How much knowledge can you pack into the parameters of a language model? In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5418–5426, Online.

Nicholas Roberts, Niladri S. Chatterji, Sharan Narang et al. 2025. Compute optimal scaling of skills: Knowledge vs reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 13295–13316, Vienna, Austria. Association for Computational Linguistics.

Stephen E Robertson, Steve Walker, Susan Jones et al. 1995. et almbox. 1995. okapi at trec-3. Nist Special Publication Sp, 109:109.

Baptiste Rozière, Jonas Gehring, Fabian Gloeckle et al. 2024. Code llama: Open foundation models for code.

Christopher Sciavolino, Zexuan Zhong, Jinhyuk Lee et al. 2021. Simple entity-centric questions challenge dense retrievers. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6138– 6148, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Luca Soldaini, Rodney Kinney, Akshita Bhagia et al. 2024. Dolma: an open corpus of three trillion tokens for language model pretraining research. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15725–15788, Bangkok, Thailand. Association for Computational Linguistics.

Jacob Mitchell Springer, Sachin Goyal, Kaiyue Wen et al. 2025. Overtrained language models are harder to fine-tune. In I Can’t Believe It’s Not Better: Challenges in Applied Deep Learning.

Hugo Touvron, Thibaut Lavril, Gautier Izacard et al. 2023a. Llama: Open and efficient

foundation language models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Denny Vrandeˇci´c and Markus Krötzsch. 2014. Wikidata: a free collaborative knowledgebase. Commun. ACM, 57(10):78–85.

Wenhan Xiong, Jingfei Du, William Yang Wang et al. 2020. Pretrained encyclopedia: Weakly supervised knowledge-pretrained language model. In International Conference on Learning Representations.

Sohee Yang, Elena Gribovskaya, Nora Kassner et al. 2024. Do large language models latently perform multi-hop reasoning? In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10210–10229, Bangkok, Thailand. Association for Computational Linguistics.

Zichao Yang, Phil Blunsom, Chris Dyer et al. 2017. Reference-aware language models. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 1850–1859, Copenhagen, Denmark. Association for Computational Linguistics.

Linxi Zhao, Sofian Zalouk, Christian K. Belardi et al. 2025. Pre-training large memory language models with internal and external knowledge.

Vilém Zouhar, Marius Mosbach, Debanjali Biswas et al. 2021. Artefact retrieval: Overview of NLP models with knowledge base access. In Workshop on Commonsense Reasoning and Knowledge Bases.

### A Appendix: Supplementary Results

- A.1 Pretraining Data Statistics

Tab. 2 summarize statistics over the LMEnt pretraining corpus. These results support the overview provided in §4.

# tokens 3.6B # chunks 10.5M # entities 7.3M # batches (steps) 109K # total mentions 400M # hyperlink mentions (H) 115M # entities H = 1 993K # entities 1 < H ≤ 10 2.2M # entities 10 < H ≤ 100 967K # entities 100 < H ≤ 1K 141K # entities H > 1K 13K # entity linking mentions 203M # of coreference mentions 151M # of coreference cluster mentions 310M

Table 2: Statistics of the LMEnt pretraining corpus.

- A.2 PopQA and PAQ Performance

In this section we present additional results for PopQA and PAQ accuracies as functions of compute budget. Fig. 10 shows the results for all entities and moderately frequently co-occuring subject and answer entities in PopQA, and Fig. 11 shows the results for all entities and frequently co-occuring subject and answer entities in PAQ. These results support §5.1.

10 3 10 2 10 1 100 101 102 103 FLOPs (trillions, log scale)

0.02

0.04

0.06

0.08

0.10

0.12

0.14

Accuracy

PopQA (All)

1B

1B-20K-84B

2-1B

170M-6E

600M-6E

1B-6E

1.4B 1B

160M

410M

1.7B

135M

360M

1B-10KS 1B-0S

Model Family

LMEnt

OLMo

Pythia

SmolLM2

Baselines

10 3 10 2 10 1 100 101 102 103 FLOPs (trillions, log scale)

0.0

0.1

0.2

0.3

0.4

0.5

0.6

Accuracy

PopQA (100-1K Subject Answer Chunks Shared)

1B

1B-20K-84B

2-1B

170M-6E

600M-6E

1B-6E 1.4B 1B

160M

410M

1.7B

135M

360M

1B-10KS 1B-0S

Model Family

LMEnt OLMo

| |
|---|

Pythia

SmolLM2

Baselines

- Figure 10: Accuracy on PopQA as a function of compute budget: (left) all entities and (right) questions for which the subject and answer entities appear together in 100-1K chunks.

- A.3 Popularity and Fact Frequency Indicators for Model Behavior on Additional Models

In this section, we present additional results for PopQA accuracies sliced by popularity and fact frequency indicators. Fig. 10 shows the results for OLMo-2-1B (OLMo et al., 2025), Pythia-1.4B (Biderman et al., 2023), and SmolLM-2-1.7B (Allal et al., 2025). These results support §5.2.

- A.4 Results on Additional Tasks We report results on non-knowledge tasks in Tab. 3, which complement §5.1.
- A.5 Empirical Experiment for Choosing LMEnt Score Thresholds

In this section, we described the empirical experiment that determined the score thresholds for LMEnt retrieval (§5.2). To determine thresholds for the hyperlink score H, entity-linking score EL, and coreference scores C and CC, we sample 60 entities from PopQA based on hyperlinks counts. For each

PAQ (All)

PAQ (1K+ Subject Answer Chunks Shared)

2-1B

2-1B

0.40

Model Family

Model Family

0.06

LMEnt OLMo

LMEnt OLMo

1.7B

1.7B

0.35

0.05

| |
|---|

| |
|---|

0.30

Pythia

Pythia

SmolLM2

SmolLM2

1B

0.04

0.25

Accuracy

Accuracy

Baselines

Baselines

1.4B

360M

1B

0.20

0.03

1B-6E

1.4B

0.15

360M

1B

0.02

1B-6E

600M-6E

0.10

135M

600M-6E

1B

170M-6E

410M

1B-20K-84B

0.01

170M-6E

135M

0.05

410M

1B-20K-84B

1B-10KS 1B-0S

160M

1B-10KS 1B-0S

160M

0.00

0.00

10 3 10 2 10 1 100 101 102 103 FLOPs (trillions, log scale)

10 3 10 2 10 1 100 101 102 103 FLOPs (trillions, log scale)

- Figure 11: Accuracy on PAQ as a function of compute budget: (left) all entities and (right) questions for which the subject and answer entities appear together in 1K+ chunks.

0.0 0.2 0.4 0.6 Mean Accuracy

1 10

10 100

100 1K

1K+

ChunkBins

Subj Chunks

Answer Chunks

Subj+Answer Chunks

Subj Pop

Answer Pop 1 10

100 1K

1K 10K

10K 100K

100K+

PopularityBins

OLMo-2-1B

0.0 0.2 0.4 0.6 Mean Accuracy

1 10

10 100

100 1K

1K+

ChunkBins

Subj Chunks

Answer Chunks

Subj+Answer Chunks

Subj Pop

Answer Pop 1 10

100 1K

1K 10K

10K 100K

100K+

PopularityBins

Pythia-1.4B

0.0 0.2 0.4 0.6 Mean Accuracy

1 10

10 100

100 1K

1K+

ChunkBins

Subj Chunks

Answer Chunks

Subj+Answer Chunks

Subj Pop

Answer Pop 1 10

100 1K

1K 10K

10K 100K

100K+

PopularityBins

SmolLM-1.7B

- Figure 12: Accuracy of OLMo-2-1B (OLMo et al., 2025), Pythia-1.4B (Biderman et al., 2023), and SmolLM-21.7B (Allal et al., 2025) on PopQA, sliced by various indicators of popularity and fact frequency on PopQA.

entity, we retrieve their scores using a fixed hyperlink threshold of H = 1, while varying EL, C, and CC across the set {0.4,0.5,0.6,0.7,0.8}, and evaluate precision over chunks retrieved at different depths k. If k > 100, we randomly sample 100 of the k returned chunks and submit them to Gemini 2.5 Flash Preview 6-17 using the prompt shown in Fig. 17 to compute precision at k. Results are reported in Tab. 4.

A.6 Win Rates

In Fig. 13, we extended the experiments in (§5.2, Fig. 5) to compare win rates between the best string based method, CS-SS Canonical, and all other string-based variants (left), and well as CI-SS Canonical versus CI-SS Expanded (right). These results complement Fig. 5.

|23.5|56.4| | |20.1|
|---|---|---|---|---|
|43.5| | |37.4|19.1|
|40.2| |41.8| |18.0|

CI-SS Canonical CI-SS Expanded

CS-SS Expanded

| |
|---|

CS-SS Canonical wins

| |
|---|

Tie

| |
|---|

Other wins

|41.8|39.8|18.5|
|---|---|---|

CI-SS Expanded

| |
|---|

CI-SS Canonical wins

| |
|---|

Tie

| |
|---|

Baseline wins

- Figure 13: Pairwise win rates between CS-SS Canonical and other string-based methods (left). Pairwise win rates between CI-SS Canonical and CI-SS Expanded (right).

- Table 3: Performance of LMEnt models and baseline models on commonsense reasoning, multiple choice, and reading comprehension tasks.

Model arc_challenge arc_easy boolq copa coqa drop gsm8k hellaswag jeopardy naturalqs_open openbookqa piqa sciq squad triviaqa truthfulqa winogrande

- LMEnt-170M-1E 0.217 0.370 0.574 0.570 0.072 0.022 0.020 0.271 0.006 0.002 0.304 0.541 0.633 0.038 0.000 0.471 0.493
- LMEnt-170M-2E 0.247 0.368 0.620 0.610 0.081 0.024 0.025 0.274 0.006 0.008 0.320 0.560 0.697 0.043 0.002 0.471 0.505 LMEnt-170M-4E 0.254 0.282 0.367 0.510 0.000 0.000 0.000 0.247 0.000 0.000 0.264 0.487 0.237 0.000 0.000 0.488 0.488 LMEnt-170M-6E 0.271 0.389 0.400 0.640 0.082 0.034 0.025 0.280 0.006 0.007 0.356 0.541 0.711 0.056 0.004 0.471 0.501

- LMEnt-600M-1E 0.271 0.409 0.604 0.650 0.085 0.012 0.025 0.285 0.008 0.001 0.340 0.555 0.722 0.044 0.001 0.472 0.490
- LMEnt-600M-2E 0.301 0.405 0.620 0.630 0.113 0.046 0.055 0.296 0.009 0.020 0.350 0.554 0.744 0.085 0.005 0.468 0.516 LMEnt-600M-4E 0.254 0.389 0.565 0.610 0.150 0.055 0.015 0.306 0.034 0.010 0.338 0.559 0.735 0.092 0.016 0.470 0.515 LMEnt-600M-6E 0.274 0.409 0.625 0.630 0.167 0.075 0.050 0.318 0.023 0.029 0.374 0.569 0.762 0.119 0.045 0.444 0.510

- LMEnt-1B-1E 0.254 0.421 0.603 0.650 0.109 0.063 0.035 0.295 0.009 0.019 0.358 0.557 0.714 0.084 0.006 0.469 0.506
- LMEnt-1B-2E 0.244 0.377 0.626 0.630 0.132 0.068 0.045 0.309 0.020 0.018 0.360 0.557 0.765 0.089 0.047 0.448 0.523 LMEnt-1B-4E 0.278 0.391 0.545 0.670 0.160 0.067 0.010 0.322 0.021 0.024 0.374 0.557 0.763 0.092 0.008 0.455 0.517 LMEnt-1B-6E 0.264 0.446 0.611 0.680 0.157 0.070 0.035 0.328 0.043 0.031 0.390 0.555 0.770 0.107 0.025 0.439 0.529 OLMo-1B 0.338 0.565 0.617 0.770 0.574 0.218 0.030 0.627 0.342 0.123 0.434 0.729 0.879 0.624 0.324 0.329 0.594

- OLMo-1B-20K-84B 0.288 0.458 0.640 0.760 0.269 0.079 0.000 0.433 0.059 0.051 0.406 0.676 0.790 0.251 0.085 0.394 0.531
- OLMo-2-1B 0.435 0.635 0.646 0.800 0.694 0.353 0.395 0.688 0.641 0.191 0.474 0.746 0.953 0.818 0.512 0.369 0.654

Pythia-1.4B 0.348 0.539 0.582 0.760 0.566 0.206 0.020 0.540 0.273 0.098 0.434 0.707 0.874 0.565 0.250 0.386 0.566 Pythia-1B 0.331 0.525 0.623 0.740 0.519 0.185 0.025 0.478 0.195 0.077 0.378 0.683 0.890 0.534 0.188 0.389 0.533 Pythia-410M 0.301 0.467 0.588 0.700 0.428 0.166 0.040 0.393 0.069 0.058 0.360 0.663 0.833 0.438 0.104 0.410 0.532 Pythia-160M 0.311 0.423 0.498 0.640 0.218 0.136 0.020 0.300 0.019 0.031 0.324 0.606 0.746 0.176 0.042 0.442 0.502

SmolLM2-1.7B 0.468 0.704 0.733 0.840 0.720 0.259 0.315 0.695 0.706 0.194 0.496 0.760 0.940 0.771 0.545 0.367 0.671 SmolLM2-360M 0.448 0.649 0.620 0.760 0.596 0.190 0.045 0.551 0.516 0.107 0.444 0.710 0.905 0.656 0.305 0.334 0.591 SmolLM2-135M 0.365 0.558 0.617 0.660 0.429 0.135 0.025 0.410 0.244 0.079 0.422 0.670 0.842 0.467 0.175 0.388 0.530

- Table 4: Empirical evaluation of LMEnt retrieval precision across multiple thresholds and retrieval depths k on the development set of 60 PopQA entities.

Score Thresholds 1 5 10 100 1K 10K 100K

- H=1, EL = C = CC = 0.4 0.95 0.95 0.96 0.98 0.99 0.98 0.94
- H=1, EL = C = CC = 0.5 0.95 0.95 0.96 0.98 0.99 0.99 0.91
- H=1, EL = C = CC = 0.6 0.95 0.96 0.96 0.98 0.99 0.98 0.94
- H=1, EL = C = CC = 0.7 0.95 0.96 0.96 0.98 0.99 0.98 0.91
- H=1, EL = C = CC = 0.8 0.95 0.96 0.96 0.98 0.97 0.88 0.96

##### A.7 LMEnt wins across all entity popularity levels, often by large margins

- Fig. 14 shows “Yes” chunk count differences between methods and the cumulative percentage of entities where one method outperforms the other, across entities of different frequencies. For tail entities (right), LMEnt outperforms by a few chunks, but this is meaningful given the scarcity of mentions. For torso entities middle), LMEnt wins by a substantial margin (≥ 20 additional correct chunks) in 40% of cases. For head entities (left), the margin is typically smaller (1–25 chunks), suggesting that string-based search performs more competitively in these cases – though, LMEnt still outperforms it on ≥ 60% of entities.

20 0 20 40 60 80 100

0%

20%

40%

60%

80%

Win%

Head (H > 1K)

20 0 20 40 60 80 100 # Better chunks (out of 100)

Torso (10 < H <= 1K)

20 0 20 40 60 80 100

Tail (H <= 10)

String-based wins Tie Entity-based wins

Figure 14: Comparison between LMEnt retrieval and the best performing string-based approach (CS-SS Canonical). Results are grouped by popularity; tail entities (right, under 10 hyperlinks), torso entities (center, between 10 and 1K chunks) center, and head entities (left, at least 1K chunks). LMEnt outperforms string-based retrieval on all popularity levels, and often wins by a large amount of better chunks for popular entities.

A.8 Knowledge Acquisition during Pretraining

- Fig. 15 shows the net gains in facts learned between LMEnt-1B-6E checkpoints across frequency bins of subject and answer entity co-occurence. The split of % of facts learned and forgotten is in §6, Fig. 9.

Net % of Facts Learned

+0.10

+0.08

+0.06

+0.04

+0.02

0

1 10 10 100 100 1K 1K+ Fact Frequency (per Interval)

Training Step Intervals

(0, 20K) (20K, 40K)

(40K, 60K) (60K, 80K)

(80K, 100K) Sum

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 15: Percentage of net gains in facts learned between intermediate checkpoints of LMEnt-1B-6E. We analyze these results in detail in §6.

### B Appendix: Additional Implementation Details

##### B.1 Running Maverick At Scale

The backbone of Maverick (Martinelli et al., 2024) is DeBERTa (He et al., 2021) which has quadratic attention complexity with respect to sequence length. Therefore, running Maverick on long documents is impractical, e.g. 15K token long documents take 40-50GB VRAM. Therefore, we reduce the memory footprint, and accelerate and parallelize inference across multiple documents by replacing the DeBERTa backbone with a FlashAttention-based implementation (Knowledgator, 2025). Then, we apply Maverick to non-overlapping windows of 50K tokens, which may result in multiple distinct coreference clusters for the same entity. In these cases, we rely on the coverage of hyperlinks and the entity linker (which runs on the entire document) to map each cluster to the target entity’s QID.

##### B.2 Coreference scoring

In this section we detail how we compute the score derived from coreference resolution, and how we resolve ambiguity when multiple entities are associated with a single coreference cluster (§2.2).

Coref (C) is designed to associate shortened references, not previously found by hyperlinks or entity linking, with their canonical names, e.g. “the hospital” with “John R. Oishei Children’s Hospital”. The coreference cluster of “the hospital” contains “the hospital” and “John R. Oishei Children’s Hospital in Buffalo”. There are two entities related to this cluster, with QIDs: “Q93565992” and “Q40435”. We can reduce ambiguity by leveraging the shared substring of “(H,h)ospital” to promote “the hospital” being mapped to “Q93565992” more strongly. We represent this promotion by a score denoted by C, which we formally define below.

Let the coreference cluster associated with m be C, and define the sets of hyperlink and entity linking mentions that overlap with C as CH and CEL, respectively. To calculate C(m,e), for each mention m′ ∈ CH ∪ CEL, we compute the longest common substring (LCS) with m. Let sim(m,m′) be the harmonic mean of the LCS overlap ratios between mention m and m′, defined as:

sim(m,m′) =

0.0 if |m| = 0 or |m′| = 0 2·a·b a+b otherwise

LCS(m,m′) |m|

LCS(m,m′) |m′|

where a =

, b =

We weight either the hyperlink or entity linking score of m′ for e by its textual similarity to m:

support(m,m′,e) = sim(m,m′) ·

H(m′,e) if m′ ∈ CH EL(m′,e) if m′ ∈ CEL

(1)

(2)

Finally, the C score for mention m and entity e is given by the maximum support score over all other mentions m′ ∈ C:

support(m,m′,e) (3)

C(m,e) = max

m′∈CH∪CEL

Coref-Cluster (CC) connects generic indirect mentions (that don’t share textual similarity with the entity name) to their canonical entity names, e.g. “their” with “Buffalo Bills”. In this case, we compute a distribution of scores over all the entities linked to some mention in the cluster. The score CC(e) is defined per entity per cluster, and is shared across all mentions m ∈ C.

For each mention m ∈ C and entity e, we look at the existing source scores and compute a msupport score. We don’t aggregate the existing source scores (e.g. compute an average), because we found that this disproportionately rewarded single mentions identified by multiple sources and outweighed multiple mentions found by single sources, leading to incorrect cluster to entity associations. The Csupport for entity e in cluster C is simply the sum over all msupport scores.

 

H(m,e) if m ∈ CH EL(m,e) if m ∈ CEL C(m,e) otherwise

msupport(m,e) =

msupport(m,e) (4)

Csupport(e) =



m∈C

To compute a distribution of scores over all entities mentioned in the cluster, we compute a softmax over CC(C,e′) scores for the set of entities e′ supported by at least one mention in C, denoted by EC.

exp Csupport(e) e′∈EC exp Csupport(e′)

CC(C,e) =

(5)

##### B.3 Error Analysis of LMEnt Annotations

This section describes a qualitative error analysis in which we sampled 112 mentions retrieved using LMEnt, and manually analyzed the annotation errors observed and their frequency. We saw three errors total (2.7%) that are displayed in Fig. 16. The errors were results of entity linking failures and one coreference resolution failure.

[Figure 10]

Figure 16: Error analysis of LMEnt entity mentions.

##### B.4 Pretrained Models

Details of the LMEnt models are described here and support the overview in §4. The 170M, 600M, and 1B models use (layers,hidden dimension) configurations of (10,768), (16,1344), and (16,2048), respectively. A hyperparameter search was conducted over the following ranges: global batch size

{16,384, 32,768, 65,536, 131,072, 262,144}, peak learning rate {3×10−4, 6×10−4, 8×10−4, 1.2× 10−3, 3×10−3, 5×10−3}, and weight decay {0.005, 0.05, 0.1}. Hyperparameters were selected based on the minimal final training perplexity. All LMEnt models were trained using the AdamW optimizer with a global batch size of 32,768, rank batch size of 8,192, peak learning rate of 5×10−3, weight decay of 0.05, and 1,000 warmup steps.

##### B.5 LM Judge for Chunk Precision

To automatically evaluate the retrieval quality of both LMEnt (our entity-based method) and string-based baselines (see § 5.2), we use Gemini 2.5 Flash Preview 6-17. For each of the 1K entities in the test set, we apply LMEnt, its ablations, and string-based methods CS-SS-(Canonical, Expanded) and CI-SS-(Canonical, Expanded), to retrieve sets of relevant chunks. For a retrieved set containing more than 100 chunks, we randomly sample 100 chunks from it for evaluation.

To determine whether a chunk mentions the entity, we prompt Gemini using the instruction shown in Fig. 17. To provide context, we include the entity’s description from ReFinED (Ayoola et al., 2022), which consists of the Wikipedia article title and the first sentence in the article. For each chunk, we identify matching mentions of the entity to create a short context window for Gemini. For LMEnt, mentions are selected based on QID and score thresholds. For string-based methods, we use the highlight block in the Elasticsearch (Banon, 2010) query to extract character spans that match the entity name. Around each mention, we select a 130-character window to give Gemini some context of the mention in the chunk text. We evaluate up to three mentions per chunk. If Gemini returns “Yes” for any, the chunk is marked as mentioning the entity. If all are judged “No”, the chunk is considered not to mention it. For LMEnt, mentions are prioritized by a weighted sum of their scores, and for string-based methods, mentions are evaluated in the order they appear in the document.

Justifying use of LLM-as-a-Judge To support our use of an LLM-as-a-Judge for evaluating modelgenerated responses, we use the alternative annotator test introduced by Calderon et al. (2025). This test determines whether the LLM’s performance is comparable or better than that of a randomly chosen human annotator. In line with their methodology, we employed three human annotators (graduate students) and used a dataset of 100 entity mentions randomly sampled from ‘Yes” and “No” judged chunks. For each mention, the annotators received the same inputs as the LLM judge: the entity name, entity description, and 130-character context window around the mention, and were asked to evaluate whether the chunk named the entity. Instructions are found in Fig. 18. Following Calderon et al. (2025), we set ϵ = 0.1, which yielded a winning rate of ω = 1.00 with a p-values of 0.001,0.001,and 8.28e−5 for all three annotators, indicating that the LLM can be relied on.

##### B.6 Converting Queries to Cloze-Style Prompts

Since LMEnt models are not instruction-tuned, we evaluate them on PAQ by converting each question into a cloze-style prompt, where the expected answer is the next predicted phrase. To perform this transformation, we use Gemini 2.5 Flash Preview 6-17 with the prompt shown in Fig. 19.

Goal: Determine if the provided Text directly mentions or discusses the specific Entity. Process: First, carefully read the Text and identify any specific names, terms, or phrases that appear and might relate to the Entity. Second, compare these identified text mentions (if any) to the provided Entity. Rules for "Directly Mentions or Discusses":

- 1. The Text must contain the exact name of the Entity or a clearly identifiable and standard part of the Entity's name (e.g., "Einstein" for "Albert Einstein") that contextually refers directly to the Entity.
- 2. The mention must refer to the Entity itself. Merely using the Entity's name as a descriptive modifier for something else associated with the Entity (e.g., "the Wahhabi school of Saudi Arabia" referring to the school, not the country) does not count as directly mentioning or discussing the Entity, unless the descriptive phrase is the Entity (e.g., "University of London" for the entity "University of London").
- 3. Exclusions (Do NOT count as direct mention/discussion): Mentions of individuals or things related to the Entity but not the Entity itself (e.g., family members, associates, parts or products of the entity unless the mention clearly refers back to the entity).
- 4. Inferences based on context, related events, locations, or concepts if the Entity's name is not used to refer to the Entity itself according to Rule 3.
- 5. Distinguishing Similar Names: If the Text contains a name that is similar to the Entity's name but refers to a different person, place, or thing (e.g., a different historical figure, a different organization with a similar name, or differing spellings like "Elisabeth" vs. "Elizabeth" referring to different individuals), it does not count as a direct mention of the target Entity. The context must clearly indicate the reference is to the specific Entity provided.
- 6. Possessives, actions, or attributes must be logically consistent with the entity’s type. Implausible associations, like assigning a teacher to Ennis, a town in County Clare, Ireland (e.g., “Ennis’s teacher”), are invalid. Consider each rule individually and assign an intermediate judgment of Pass or Fail. If any rule receives a Fail, the final decision must be No. Output: Provide only the answer 'Yes' or 'No' based on the above rules. Do not include any explanation or other text.

###### Entity: {entity_description} Text: {text} Answer:

Figure 17: Prompt given to Gemini to automatically judge whether a chunk mentions an entity directly.

[Figure 11]

Figure 18: Instructions given to annotators for evaluating LLM-as-a-judge.

Your goal is to rephrase the input Question as a Cloze Statement so that the Answer is the next word that would logically complete the sentence.

Instructions:

- 1. Do not lose any context from the question. The cloze statement must preserve the full meaning and specificity of the original question. For example, Question: What TV network does Funny or Die Presents air on? Answer: HBO Cloze Statement: Funny or Die Presents airs on the TV network
- 2. Rephrase the question into a natural, grammatically correct sentence. If the question is incomplete or unclear, you may add minimal language to clarify the intended answer. For example, Question: stevie cameron worked as a confidential informant for the rcmp during which controversial Cloze Statement: Stevie Cameron worked as a confidential informant for the rcmp during the controversial
- 3. End the statement right before the answer.
- 4. Use any specific labels or terms from the question. For example, if the question specifies "capital city" or "TV network," include that exact phrase in the cloze. Output ONLY the Cloze Statement at the end of thinking.

Question: {question} Answer: {answer} Cloze Statement:

Figure 19: Prompt given to Gemini 2.5 Flash Preview 6-17 to convert PAQ questions to cloze-style prompts.

