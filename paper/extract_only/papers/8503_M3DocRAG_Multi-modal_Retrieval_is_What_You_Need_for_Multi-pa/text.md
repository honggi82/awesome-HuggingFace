# arXiv:2411.04952v1[cs.CV]7Nov2024

## M3DOCRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding

Jaemin Cho1* Debanjan Mahata2 Ozan ˙Irsoy2 Yujie He2 Mohit Bansal1 1UNC Chapel Hill 2Bloomberg {jmincho,mbansal}@cs.unc.edu {dmahata,oirsoy,yhe247}@bloomberg.net

### Abstract

Document visual question answering (DocVQA) pipelines that answer questions from documents have broad applications. Existing methods focus on handling single-page documents with multi-modal language models (MLMs), or rely on text-based retrieval-augmented generation (RAG) that uses text extraction tools such as optical character recognition (OCR). However, there are difficulties in applying these methods in real-world scenarios: (a) questions often require information across different pages or documents, where MLMs cannot handle many long documents; (b) documents often have important information in visual elements such as figures, but text extraction tools ignore them. We introduce M3DOCRAG, a novel multi-modal RAG framework that flexibly accommodates various document contexts (closed-domain and open-domain), question hops (singlehop and multi-hop), and evidence modalities (text, chart, figure, etc.). M3DOCRAG finds relevant documents and answers questions using a multi-modal retriever and an MLM, so that it can efficiently handle single or many documents while preserving visual information. Since previous DocVQA datasets ask questions in the context of a specific document, we also present M3DOCVQA, a new benchmark for evaluating open-domain DocVQA over 3,000+ PDF documents with 40,000+ pages. In three benchmarks (M3DOCVQA/MMLongBench-Doc/MP-DocVQA), empirical results show that M3DOCRAG with ColPali and Qwen2-VL 7B achieves superior performance than many strong baselines, including state-of-the-art performance in MP-DocVQA. We provide comprehensive analyses of different indexing, MLMs, and retrieval models. Lastly, we qualitatively show that M3DOCRAG can successfully handle various scenarios, such as when relevant information exists across multiple pages and when answer evidence only exists in images.

*Work done during an internship at Bloomberg as a recipient of the Bloomberg Data Science Ph.D. Fellowship.

### 1. Introduction and Background

Document visual question answering (DocVQA) [14, 31, 40, 42, 57] is a multi-modal task that answers textual questions by interpreting information contained within document images. Existing methods on DocVQA either focus on visual question answering (VQA) on a single-page document (Fig. 1 (a)) or extract text from documents (e.g., via optical character recognition (OCR) [43, 53] or PDF text extraction [18, 49]) and use retrieval-augmented generation (RAG) [35], where a retrieval model finds relevant paragraphs and a language model answers questions given the paragraphs (Fig. 1 (b)). However, there are difficulties in applying these methods in real-world document understanding scenarios: (a) questions often require information across different pages or documents, where existing VQA methods cannot handle many long documents; (b) some documents feature complex visual formats such as tables, charts, and mixed layouts, but text extraction methods such as OCR ignore these nuances, leading to incomplete or inaccurate document interpretations. Accurately and efficiently answering questions across numerous, lengthy documents with intricate layouts would greatly benefit many domains such as finance, healthcare, and law, where document AI assistants can streamline the daily processing of large volumes of documents, improving productivity and enabling faster, more informed decision-making.

To overcome these limitations of existing DocVQA approaches, we introduce M3DOCRAG (Multi-modal Multipage Multi-Document Retrieval-Augmented Generation; Sec. 2), a novel multi-modal RAG framework that flexibly accommodates various document contexts (closed-domain and open-domain), question hops (single-hop and multihop), and evidence modalities (text, chart, figure, etc.). As illustrated in Fig. 1 (c), the M3DOCRAG framework retrieves relevant document pages using a multi-modal retrieval model, such as ColPali [17], and generates answers to questions from the retrieved pages using a multimodal language model (MLM), such as Qwen2-VL [59]. M3DOCRAG operates in three stages: In (1) document

###### (a) Single-page DocVQA

- Can’t handle many/long documents 😥

[Figure 1]

|Text Query|
|---|

|[Figure 2]|
|---|

|Answer|
|---|

Multi-modal LM

Single-page Document

- (b) Text-based RAG
- (c) M3DocRAG (Ours)

- - Ignore visual information 😥

[Figure 3]

- - Can handle many/long documents
- - Preserve visual information 🤗

|Text Query|
|---|

[Figure 4]

[Figure 5]

Text Extraction (e.g., OCR)

Text Retrieval

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

|Answer|
|---|

…

LM

Many Long Documents

Extracted Text

Few Relevant Paragraphs

[Figure 11]

|Text Query|
|---|

[Figure 12]

|[Figure 13]<br><br>[Figure 14]|
|---|

[Figure 15]

Multi-modal LM

Multi-modal Retrieval

[Figure 16]

|Answer|
|---|

Many Long Documents

Few Relevant Pages

- Figure 1. Comparison of multi-modal document understanding pipelines. Previous works focus on (a) Single-page DocVQA that cannot handle many long documents or (b) Text-based RAG that ignores visual information. Our (c) M3DOCRAG framework retrieves relevant documents and answers questions using multi-modal retrieval and MLM components, so that it can efficiently handle many long documents while preserving visual information.

Existing DocVQA datasets: Closed-domain M3DocVQA (Ours): Open-domain

“What was the gross profit in the year 2009?”

Context: Single PDF

Model Answer

|[Figure 17]|
|---|

Context-specific question

“Which B.Piazza title came earlier: the movie S. Stallone’s son starred in or the movie with half of a lady’s face on the poster?

Context: 3K PDFs

Model Answer

Open-domain question

[Figure 18]

[Figure 19]

[Figure 20]

- Figure 2. Comparison of existing DocVQA datasets (left; e.g., DocVQA [42]) and our M3DOCVQA dataset (right). In contrast to previous DocVQA datasets that have questions that are specific to a single provided PDF (e.g., “What was the gross profit in the year 2009?”), M3DOCVQA has information-seeking questions that benchmark open-domain question answering capabilities across more than 3,000 PDF documents (i.e., 40,000+ pages).

embedding (Sec. 2.1), we convert all document pages into RGB images and extract visual embeddings (e.g., via ColPali) from the page images. In (2) page retrieval (Sec. 2.2), we retrieve the top-K pages of high similarity with text queries (e.g., MaxSim operator for ColPali). For the opendomain setting, we create approximate page indices, such as inverted file index (IVF) [52, 66], for faster search. In (3) question answering (Sec. 2.3), we conduct visual question answering with MLM to obtain the final answer. Please also see Fig. 3 for the detailed illustration of the framework. M3DOCRAG can flexibly handle DocVQA in both closed domain (i.e., a single document) and open-domain (i.e., a large corpus of documents) settings.

While M3DOCRAG framework supports DocVQA in an open-domain setting, the existing DocVQA datasets are not adequate for this setting, since their questions are in the context of a specific document, such as “What was the gross profit in the year 2009?” [14, 40, 42, 57], as illustrated in Fig. 2 (left). Hence, we also introduce M3DOCVQA (Multi-modal Multi-page Multi-Document Visual Question Answering), an open-domain dataset that significantly raises the challenge of DocVQA to answering questions from a large document corpus (Sec. 3). By extending the MultimodalQA dataset’s [54] closed-domain context to an open-domain setting, M3DOCVQA introduces 2,441 multi-hop questions spanning 3,368 PDF doc-

##### 1) Document Embedding

###### Visual embeddings of all pages 𝑃

[𝑁,𝑛 ,𝑑] [𝑛 ,𝑑] [𝑛 ,𝑑] [𝑛 ,𝑑]

##### Corpus 𝐶

| | | | | |
|---|---|---|---|---|

Page embeddings of 1st doc ...

...

[Figure 21]

[Figure 22]

| | | | | |
|---|---|---|---|---|

| |Convert to| |Visual Encoder| |
|---|---|---|---|---|
| |Images| |(ColPali)| |

[Figure 23]

...

Page embeddings of ith doc

| | | | | |
|---|---|---|---|---|

...

[𝑛 ,𝑑]

| | | | | |
|---|---|---|---|---|

𝑀 documents (with 𝑁 total pages)

...

...

Page embeddings of Mth doc

[𝑛 ,𝑑]

| | | | | |
|---|---|---|---|---|

##### 2) Page Retrieval

##### 3) Question Answering

|Text Query 𝑞|
|---|

| |(in open-domain setting) Faster search with approximate indexing|Text Encoder (ColPali)<br><br>MaxSim|
|---|---|---|
| |[Figure 24]| |

Visual embeddings of all pages

er

| | | | | |
|---|---|---|---|---|

|[Figure 25]<br><br>[Figure 26]|
|---|

| | | | | |
|---|---|---|---|---|

|Answer 𝑎|
|---|

Multimodal LM

| | | | | |
|---|---|---|---|---|

...

Top-𝐾 Pages (𝑃 )

| | | | | |
|---|---|---|---|---|

- Figure 3. Our M3DOCRAG framework (Sec. 2) consists of three stages: (1) document embedding (Sec. 2.1), (2) page retrieval (Sec. 2.2), and (3) question answering (Sec. 2.3). In (1) document embedding, we extract visual embedding (with ColPali) to represent each page from all PDF documents. In (2) page retrieval, we retrieve the top-K pages of high relevance (MaxSim scores) with text queries. In an open-domain setting, we create approximate page indices for faster search. In (3) question answering, we conduct visual question answering with multi-modal LM (e.g. Qwen2-VL) to obtain the final answer.

contexts (closed-domain and open-domain), question hops (single-hop and multi-hop), and evidence modalities (text, chart, figure, etc.). As illustrated in Fig. 3, M3DOCRAG operates in three stages: (1) encoding document images into visual embeddings (Sec. 2.1), (2) retrieving relevant document pages (Sec. 2.2), and (3) generating answers to questions based on the retrieved pages (Sec. 2.3). Below, we explain the problem definition and the details of each stage.

uments, which collectively contain over 41,005 pages of diverse multi-modal content, including text, images, and tables. This dataset presents real-world challenges by requiring models to navigate complex reasoning paths across pages and within various types of document elements, better reflecting the intricacies of document understanding.

To demonstrate the effectiveness of M3DOCRAG, we compare M3DOCRAG with state-of-the-art baselines in three benchmarks: M3DOCVQA, MMLongBenchDoc [40], and MP-DocVQA [57], which cover both opendomain (Sec. 5.1) and closed-domain (Sec. 5.2) DocVQA settings. Experiment results show that M3DOCRAG with ColPali and Qwen2-VL 8B achieves superior performance than many strong baselines, including the state-of-the-art performance in MP-DocVQA. We also provide a comprehensive analysis (Sec. 5.3) about different indexing, MLMs, and retrieval components. Finally, we show qualitative examples (Sec. 5.4) where M3DOCRAG can successfully handle various scenarios, such as when the relevant information exists across multiple pages and when answer evidence only exists in images. Overall, M3DOCRAG is an effective, efficient, and flexible framework for answering questions from multi-modal documents in various settings.

Problem definition. We define a corpus of documents as C = {D1,D2,...,DM}, where M is the total number of documents, and each document Di consists of a set of pages, Pi, represented as RGB images. From the documents in C, we construct a global set of page images P =

M i=1 Pi = {p1,p2,...,pN}, where each pj represents an individual page image, and N is the total number of page images across all documents in C (i.e., N = Mi=1 |Pi|). The objective of M3DOCRAG is to accurately answer a given question q using the multi-modal information available in the corpus of documents C. First, we identify PKq , the top K (≪ N) pages that are most relevant to answering the query q from the global page set P. Then, we obtain the final answer with a question answering model that takes retrieved page images PKq and query q as inputs. The problem of question answering can be categorized into two settings with different document context sizes:

### 2. M3DOCRAG: A Unified Framework for Multi-modal, Multi-page, Multi-document Understanding

Closed-domain question answering – The query q should be answerable from a given single document Di. The retrieval model outputs the top K relevant page images

We propose M3DOCRAG, a novel multi-modal RAG framework that flexibly accommodates various document

PKq , from the page images Pi of the document Di.

Open-domain question answering – The query q may require information from single or multiple documents within the entire document corpus C. The retrieval model outputs the top K relevant page images PKq from the entire set of page images P.

#### 2.1. Document Embedding

In M3DOCRAG, both textual query q and page images P are projected into a shared multi-modal embedding space using ColPali [17]. ColPali is a multi-modal retrieval model based on a late interaction mechanism, which encodes the text and image inputs into unified vector representations and retrieves the top K most relevant images. ColPali adopts both training objective and similarity scoring from ColBERT [29, 50], which utilizes a shared architecture to encode either textual or visual inputs. In our framework, each page p ⊆ Pi of a document Di is treated as a single image with fixed dimensions (width × height).

From an image of a page, we extract a dense visual embedding Ep ∈ Rn

v×d, where nv represents the number of visual tokens per page (which remains constant across all pages), and d denotes the embedding dimension (e.g., 128). For a textual query q, we similarly obtain an embedding Eq ∈ Rn

q×d, where nq is the number of text tokens. For efficiency, we treat each page of a document independently. This allows us to flatten all pages in the document corpus C into a single page-level embedding tensor: EC ∈ RN×n

v×d, where N represents the total number of pages in the entire document corpus, nv is the number of visual tokens per page, and d is the embedding dimension. M3DOCRAG can flexibly adapt to different retrieval settings, such as a single-page document (N = 1), a single document with multiple pages (e.g. N = 100), and a large corpus of multi-page documents (e.g. N > 1,000).

#### 2.2. Page Retrieval

The relevance between the query q and the page p is computed using the MaxSim score s(q,p):

nq

Ei,q· · Ej,p·

s(q,p) =

max

j∈[nv]

i=1

where · denotes the dot product, and Ei,· ∈ Rd denotes the i-th row (vector) of the embedding matrix E ∈ Rn×d. We

then identify PKq , the top K (≪ N) pages that are most relevant to answering the query q; i.e. we search K pages

scoring highest s(q,p). That is,

PKq = {pq1,pq2,...,pqK} = argtop-kp∈P s(q,p)

Approximate indexing for open-domain page retrieval. Searching pages over in a large document corpus can be

time-consuming and computationally expensive. When a faster search is desired, we create page indices offline by applying approximate nearest neighborhood search, based on Faiss [16, 26]. We use exact search for closed-domain page retrieval and employ inverted file index (IVF) [52, 66] (IVFFlat in Faiss) for an open-domain setting, which could reduce page retrieval latency from 20s/query to less than 2s/query when searching across 40K pages. See Sec. 5.3 for a detailed comparison of speed-accuracy tradeoffs across different indexing methods.

#### 2.3. Question Answering

We run visual question answering by giving the text query q and retrieved page images PKq to a multi-modal language model to obtain the final answer. For this, we employ multimodal language models (e.g. Qwen2-VL [59]) that consist of a visual encoder EncVis and a language model LM. The visual encoder takes K-retrieved page images PKq as inputs and outputs visual embeddings (different from ColPali encoder’s outputs). The language model takes the visual embeddings and text embeddings of query q as inputs and outputs the final answer a in the autoregressive manner:

a = LM(EncVis(PKq ),q).

### 3. M3DOCVQA: A New Benchmark for Opendomain Document Understanding

We present M3DOCVQA (Multi-modal Multi-page MultiDocument Visual Question Answering), a new opendomain DocVQA benchmark designed to evaluate the ability to answer questions using multi-modal information from a large corpus of documents.

As illustrated in Fig. 2, existing DocVQA datasets [31, 40, 42, 57] primarily focus on evaluating question answering within the context of a single document (i.e., closeddomain). These datasets are not well-suited for benchmarking open-domain visual question answering, where relevant information, often in multiple modalities such as text, images, and tables, must be retrieved from multiple documents. This limitation stems from their questions being designed around specific content on certain pages within a single document. In real-world scenarios, users often seek answers that span across multiple documents and modalities, making open-domain settings critical. However, the questions in the existing DocVQA datasets are not applicable in such an open-domain setting. For example, a question from MP-DocVQA, such as “What was the gross profit in the year 2009?” assumes that the model already has access to specific information within the document.

M3DOCVQA challenges models in an open-domain DocVQA setting, where they must navigate a large ‘haystack’ of multi-modal documents and retrieve relevant

MultimodalQA (Talmor et al., 2021) Our PDFs in M3DocVQA

|2. ^ On the back of shirt.<br>3. ^ Barcelona makes a donation to UNICEF in order to display the charity's logo on the back of the club's kit.<br>4. ^ On the shorts.<br>5. ^ Málaga makes a donation to UNESCO in order to display the charity's logo on the club's kit.<br>6. ^ On the left sleeve.<br><br><br>Managerial changes<br><br>Team<br><br>Outgoing manager<br><br>Manner of departure<br><br>Date of vacancy<br><br>Replaced by<br><br>Date of appointment<br><br>Position in table<br><br>Barcelona Pep Guardiola End of contract 30 June 2012 Tito Vilanova 13 June 2012 Pre-Season<br><br>[Figure 27]<br><br>[Figure 28]<br><br>Valencia<br><br>|[Figure 29]|
|---|
<br><br>Unai Emery End of contract 30 June 2012<br><br>|[Figure 30]|
|---|
<br><br>Mauricio Pellegrino<br><br>4 June 2012 Pre-Season<br><br>Rayo Vallecano<br><br>|[Figure 31]|
|---|
<br><br>José Ramón Sandoval<br><br>End of contract 30 June 2012<br><br>|[Figure 32]|
|---|
<br><br>Paco Jémez 14 June 2012 Pre-Season<br><br>Granada<br><br>|[Figure 33]|
|---|
<br><br>Abel Resino End of contract 30 June 2012<br><br>|[Figure 34]|
|---|
<br><br>Juan Antonio Anquela<br><br>18 June 2012 Pre-Season<br><br>Espanyol<br><br>|[Figure 35]|
|---|
<br><br>Mauricio Pochettino<br><br>Mutual consent<br><br>26 November 2012<br><br>|[Figure 36]|
|---|
<br><br>Javier Aguirre<br><br>28 November 2012<br><br>20th<br><br>Valencia<br><br>|[Figure 37]|
|---|
<br><br>Mauricio Pellegrino<br><br>Sacked<br><br>1 December 2012<br><br>|[Figure 38]|
|---|
<br><br>Voro (caretaker)<br><br>1 December 2012<br><br>12th<br><br>Valencia<br><br>|[Figure 39]|
|---|
<br><br>Voro (caretaker)<br><br>End of tenure as caretaker<br><br>5 December 2012<br><br>|[Figure 40]|
|---|
<br><br>Ernesto Valverde<br><br>3 December 2012<br><br>12th<br><br>Deportivo La Coruña<br><br>|[Figure 41]|
|---|
<br><br>José Luis Oltra Sacked<br><br>30 December 2012<br><br>|[Figure 42]|
|---|
<br><br>Domingos Paciência<br><br>31 December 2012<br><br>20th<br><br>Sevilla<br><br>|[Figure 43]|
|---|
<br><br>Míchel Sacked<br><br>14 January 2013<br><br>|[Figure 44]|
|---|
<br><br>Unai Emery<br><br>14 January 2013<br><br>12th<br><br>Granada<br><br>|[Figure 45]|
|---|
<br><br>Juan Antonio Anquela<br><br>Sacked<br><br>30 January 2013<br><br>|[Figure 46]|
|---|
<br><br>Lucas Alcaraz<br><br>30 January 2013<br><br>17th<br><br>Mallorca<br><br>|[Figure 47]|
|---|
<br><br>Joaquín Caparrós<br><br>Sacked<br><br>4 February 2013<br><br>|[Figure 48]|
|---|
<br><br>Gregorio Manzano<br><br>5 February 2013 19th<br><br>Deportivo La Coruña<br><br>|[Figure 49]|
|---|
<br><br>Domingos Paciência<br><br>Mutual consent<br><br>11 February 2013<br><br>|[Figure 50]|
|---|
<br><br>Fernando Vázquez<br><br>11 February 2013<br><br>20th<br><br>Celta de Vigo<br><br>|[Figure 51]|
|---|
<br><br>Paco Herrera Sacked<br><br>18 February 2013<br><br>|[Figure 52]|
|---|
<br><br>Abel Resino<br><br>18 February 2013<br><br>18th<br><br>League table<br><br>|Málaga 38 16 9 13|53 50 +3 57|
|---|---|
<br><br>Pos Team Pld W D L GF GA GD Pts Qualiﬁcation or relegation<br><br>1 Barcelona (C) 38 32 4 2 115 40 +75 100<br>2 Real Madrid 38 26 7 5 103 42 +61 85 Qualiﬁcation for the Champions League group stage<br>3 Atlético Madrid 38 23 7 8 65 31 +34 76<br>4 Real Sociedad 38 18 12 8 70 49 +21 66 Qualiﬁcation for the Champions League play-off round<br>5 Valencia 38 19 8 11 67 54 +13 65 Qualiﬁcation for the Europa League group stage<br>6<br>|
|---|

Average attendance

29,430

[Figure 53]

Osasuna Pamplona El Sadar 19,553 Rayo Vallecano Madrid Campo de Vallecas 15,489 Real Madrid Madrid Santiago Bernabéu 85,454 Real Sociedad San Sebastián Anoeta 32,076

Create account Log in

2011–12 2013–14

“Question”: “…”, “Answer”: “…” “Supporting Contexts”: [

|Location of teams in 2012–13 La Liga<br><br>|[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]|
|---|
|
|---|

2012–13 La Liga 38languages

Ramón Sánchez Pizjuán

Article Talk Read Edit View history Tools

Sevilla Seville

45,500

From Wikipedia, the free encyclopedia

Valencia Valencia Mestalla 55,000 Valladolid Valladolid José Zorrilla 26,512 Zaragoza Zaragoza La Romareda 34,596

The 2012–13 La Liga season (known as the Liga BBVA for sponsorship reasons) was the 82nd since its establishment. The campaign began on 18 August 2012, and ended on 1 June 2013. Barcelona won the league for a 22nd time, after leading the league the entire season and amassing 100 points, equalling Real Madrid's points record from the previous season. As in previous years, Nike provided the ofﬁcial ball for all matches, with a new Nike Maxim Liga BBVA model to be used throughout the season for all matches.

- 1. Obtain URLs of supporting contexts
- 2. Render in a web browser
- 3. Create PDFs

La Liga

Season 2012–13 Dates 18 August 2012 – 1 June

2013 Champions Barcelona

{

Relegated Mallorca Deportivo La Coruña Zaragoza

Teams

“text”: “…”, “title”: “2012-13 La Liga”, “url”:

A total of 20 teams contested the league, including 17 sides from the 2011–12 season and three promoted from the 2011–12 Segunda División. This included the two top teams from the Segunda División, and the victorious team of the play-offs.

Champions League

Barcelona Real Madrid Atlético Madrid Real Sociedad

…

###### Personnel and sponsorship

Villarreal CF, Sporting de Gijón and Racing de Santander were relegated to 2012–13 Segunda División the previous season: Villarreal were relegated after twelve years in La Liga, Sporting de Gijón returned to Segunda División after a four-year tenure in La Liga, while Racing de Santander ended ten consecutive seasons in La Liga, the longest period in its history.

Europa League Valencia Real Betis Sevilla

|Petronor|
|---|
|Azerbaijan, Huawei and Kyocera|
|Foundation, UNICEF and TV3|
|and Andalucía|
|and Estrella Galicia|
|Galicia|
| |
|Confremar and IG Markets|
|Granada|
|Comunitat Valenciana|
|UNESCO|
|Maya|
|Lacturale and Nevir|
|Adquisiciones Empresariales and|
| |
|and Kutxa|
|Interwetten|
|Solar|
|de Castilla|
|and Canal+|

Kit manufacturer

Team Head Coach Captain

Shirt sponsor

Matches played

380

Athletic Bilbao Marcelo Bielsa Carlos Gurpegui Umbro Atlético Madrid Diego Simeone Gabi Nike Barcelona Tito Vilanova Carles Puyol Nike Qatar Betis Pepe Mel Juanma Macron Cirsa Celta de Vigo Paco Herrera Borja Oubiña Li-Ning Citroën Deportivo La Coruña Fernando Vázquez Manuel Pablo Lotto Estrella Espanyol Javier Aguirre Cristian Álvarez Puma Cancún Getafe Luis García Plaza Jaime Gavilán Joma Granada Lucas Alcaraz Manuel Lucena Luanvi Caja Levante Juan Ignacio Martínez Sergio Ballesteros Kelme Málaga Manuel Pellegrini Jesús Gámez Nike Mallorca Gregorio Manzano José Nunes Macron Riviera Osasuna José Luis Mendilibar Patxi Puñal Astore Rayo Vallecano Paco Jémez Piti Erreà AE — Nevir Real Madrid José Mourinho Iker Casillas Adidas BWIN Real Sociedad Philippe Montanier Xabi Prieto Nike Canal+ Sevilla Unai Emery Andrés Palop Umbro Valencia Ernesto Valverde David Albelda Joma JinKO Valladolid Miroslav Đukić Javier Baraja Kappa El Norte Zaragoza Manolo Jiménez Javier Paredes Mercury Proniño

[Figure 75]

[Figure 76]

Goals scored 1,091 (2.87 per match) Top goalscorer Lionel Messi

https://en.wikipedia.org/wiki/2012

[Figure 77]

[Figure 78]

The three teams that were relegated were replaced by three 2011–12 Segunda División sides: Deportivo de La Coruña made an immediate return to the top level as Segunda División champion. The second-placing team Celta de Vigo was also promoted to La Liga after a ﬁve-year absence. The third promoted team was decided in the promotion play-offs where Real Valladolid returned to La Liga after two seasons in Segunda División.

(46 goals)

[Figure 79]

[Figure 80]

Best goalkeeper

Thibaut Courtois (0.78 goals/match)

[Figure 81]

[Figure 82]

-13_La_Liga ... },

[Figure 83]

[Figure 84]

Biggest home win

Atlético Madrid 6–0 Deportivo La Coruña (9 December 2012)

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

###### Stadia and locations

Biggest away win

Rayo Vallecano 0–5 Barcelona

[Figure 89]

[Figure 90]

|Athletic Bilbao|Bilbao|San Mamés|
|---|---|---|
|Atlético Madrid|Madrid|Vicente Calderón|
|Barcelona|Barcelona|Camp Nou|
|Betis|Seville|Benito Villamarín|
|Celta Vigo|Vigo|Balaídos|
|Deportivo La Coruña|A Coruña|Riazor|
|Espanyol|Barcelona|Cornellà-El Prat|
|Getafe|Getafe|Coliseum Alfonso Pérez|
|Granada|Granada|Nuevo Los Cármenes|
|Levante|Valencia|Ciutat de València|
|Málaga|Málaga|La Rosaleda|
|Mallorca|Palma|Iberostar Stadium|

- (27 October 2012) Mallorca 0–5 Real Madrid
- (28 October 2012) Valencia 0–5 Real Madrid (20 January 2013)

Team Location of stadium Stadium Capacity

...

[Figure 91]

[Figure 92]

- 39,750 54,851 99,354 52,745 31,800 34,600
- 40,500 17,700

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Highest scoring

Deportivo La Coruña 4–5 Barcelona (20 October 2012)

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Longest winning run

12 matches Barcelona

[Figure 101]

[Figure 102]

Longest unbeaten run

19 matches Barcelona

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Longest winless run

15 matches Zaragoza

[Figure 107]

[Figure 108]

- 22,524 25,534 28,963
- 23,142

Longest losing run

6 matches Deportivo La Coruña Mallorca

[Figure 109]

[Figure 110]

“Question”: “…”,

[Figure 111]

[Figure 112]

Highest attendance

96,589 Barcelona 2–2 Real Madrid

[Figure 113]

[Figure 114]

1. ^ Huawei is the sponsor for select matches.

- Figure 4. Illustration of PDF collections in M3DOCVQA. We first collect the URLs of all supporting contexts (Wikipedia documents) of individual questions of MultimodalQA [54]. Then, we create PDF versions from their URLs by rendering them in a web browser.

information to generate the final answer. The dataset consists of 2,441 multi-hop questions spread across 3,368 PDF documents, totaling 41,005 pages. Each question is supported by evidence found in one or more documents, spanning multiple modalities such as text, images, and tables, capturing the complexity and diversity typical of real-world documents. Additionally, we provide the training split, consisting of 24,162 Wikipedia PDFs. Although the documents in the training split were not utilized in our experiments, they offer future researchers the opportunity to explore even larger-scale retrieval tasks or use the documents for training models, further expanding the potential applications of M3DOCVQA.

from a Wikipedia document) that directly contains the information needed to answer the questions, simplifying the task to reasoning within the provided material. In contrast, M3DOCVQA presents an open-domain setting, where models must retrieve information from a diverse collection of 3,368 PDF documents before attempting to answer any question. This not only requires handling largescale document retrieval but also dealing with multi-modal content–text, images, and tables–distributed across multiple documents. This key distinction highlights M3DOCVQA’s ability to simulate real-world challenges, where the relevant data is often spread across multiple sources. Consequently, M3DOCVQA serves as a robust benchmark for retrieval-augmented generation tasks in document understanding, pushing the boundaries of models to deal with large-scale, multi-modal, and multi-document settings.

To create M3DOCVQA, we extend the question-answer pairs from a short-context VQA dataset to a more complex setting that includes 1) PDF documents and 2) open-domain contexts. Specifically, we use the question-answer pairs from the development split1 of MultimodalQA [54], where models answer multi-hop questions based on short multimodal contexts (e.g., short text passages, 1-2 images, a table) sourced from Wikipedia. We retrieved the URLs of all Wikipedia documents used as context in any of the MultimodalQA development split questions. Then we generated PDF versions of the Wikipedia pages by rendering them in a Chromium web browser [56], using the Playwright Python package [45]. These PDFs retain all vector graphics and metadata, ensuring zoom-in functionality and maintaining operational hyperlinks. In addition, no objects are split between different pages in the resulting PDFs.

### 4. Experiment Setup

Datasets. We benchmark M3DOCRAG on three PDF document understanding datasets that represent different scenarios: (1) M3DOCVQA (Open-domain DocVQA);

- (2) MMLongBench-Doc [40] (Closed-domain DocVQA);
- (3) MP-DocVQA [57] (Closed-domain DocVQA). In M3DOCVQA, M3DOCRAG processes over 3,000 PDFs, totaling more than 40,000 pages. For MP-DocVQA, models handle a single PDF with up to 20 pages for each question. For MMLongBench-Doc, models handle a single PDF with up to 120 pages for each question.

While both M3DOCVQA and MultimodalQA [54] share the goal of evaluating question answering given multimodal context, M3DOCVQA introduces a more demanding scenario by requiring models to retrieve relevant information from a large set of documents, as opposed to being provided with a short context. In MultimodalQA, models are given short, curated context (e.g., a paragraph

Evaluation Metrics. For M3DOCVQA, we follow the evaluation setup of MultimodalQA [54]. For MMLongBench-Doc [40] and MP-DocVQA [57], we follow their official evaluation setups. For M3DOCVQA, we evaluate answer accuracy with exact match (EM) and F1. For MMLongBench-Doc, we extract short answers with GPT4o [46] from the model outputs and report answer accuracy with generalized accuracy (based on a rule-based

1The test split of MultimodalQA [54] is unavailable, and previous works have used the development split for comparison.

- Table 1. Open-domain DocVQA evaluation results on M3DOCVQA. The scores are based on F1, unless otherwise noted. Index: FlatIP

+ IVFFlat.

Evidence Modalities Question Hops Overall Image Table Text Single-hop Multi-hop EM F1

Method # Pages

Text RAG (w/ ColBERT v2)

- Llama 3.1 8B 1 8.3 15.7 29.6 25.3 12.3 15.4 20.0
- Llama 3.1 8B 2 7.7 16.8 31.7 27.4 12.1 15.8 21.2 Llama 3.1 8B 4 7.8 21.0 34.1 29.4 15.2 17.8 23.7 M3DOCRAG (w/ ColPali)

- Qwen2-VL 7B (Ours) 1 25.1 27.8 39.6 37.2 25.0 27.9 32.3

- Qwen2-VL 7B (Ours) 2 26.8 30.4 42.1 41.0 25.2 29.9 34.6 Qwen2-VL 7B (Ours) 4 24.7 30.4 41.2 43.2 26.6 31.4 36.5

evaluation script covering different answer types) and F1 score. For MP-DocVQA, we report answer accuracy with ANLS [8] and page retrieval with accuracy (same as recall@1, as there is a single page annotation for each question) by submitting the generation results to the test server.2

Models. We mainly experiment with the ColPali v1 [17]3 retrieval model and various recent open source multi-modal LMs with <10B parameters, including Idefics 2 [33], Idefics 3 [32], InternVL 2 [12], and Qwen2-VL [59]. We also experiment with a text-based RAG pipeline by combining recent widely used text retrieval and language models: ColBERT v2 [50] and Llama 3.1 [37]. We also compare ColPali v1 with ColQwen v0.1 [17],4 another recent multi-modal retrieval model that was trained with same objective/dataset as ColPali but initialized with Qwen2-VL 2B [59] backbone. For reproducible evaluation, we use deterministic greedy decoding for answer generation. We compare these multi-modal and text-based RAG pipelines with recent top entries with comparable parameters (<10B) reported on the leaderboards.

Other implementation details. We use PyTorch [47, 48], Transformers [60], and FlashAttention-2 [13] libraries for running models. We use Tesseract [53] for OCR in text RAG baselines, following Ma et al. [40]. We use Faiss [16, 26] for document indexing. We use the pdf2image [6] library to convert each PDF page into an RGB image with a resolution of DPI=144. While all PDF pages in M3DOCVQA have the same size – 8.5 (width) × 11 (height) in inches (i.e. US letter size) and 1224 (width) × 1584 (height) in pixels, in MP-DocVQA and MMLongBench-Doc datasets, pages have slightly different sizes. To handle this, we resize page images to the most common image size within the dataset – 1700 (width) ×

- 2https://rrc.cvc.uab.es/?ch=17&com=tasks
- 3https://huggingface.co/vidore/colpali
- 4https://huggingface.co/vidore/colqwen2-v0.1

2200 (height) for MP-DocVQA, and to the most common image size within each PDF document for MMLongBenchDoc. All experiments are conducted with a single H100 80GB GPU. We provide up to 4 pages as visual inputs to our multi-modal LMs, the maximum number of images we could fit in the single GPU.

### 5. Results and Key Findings

In the following, we describe experiment results of M3DOCRAG and baselines in both open-domain (Sec. 5.1) and closed-domain settings (Sec. 5.2). Next, we provide ablation studies (Sec. 5.3) about different page indexing strategies and different multi-modal LMs and retrieval models. Lastly, we show qualitative examples (Sec. 5.4) where M3DOCRAG can tackle M3DOCVQA questions whose answer source exists in various modalities.

#### 5.1. Open-domain DocVQA

Multi-modal RAG outperforms text RAG, especially on non-text evidence sources. Table 1 shows the evaluation results on M3DOCVQA. As a model needs to find relevant documents from 3,000+ PDFs for each question, we focus solely on RAG pipelines. We observe that our M3DOCRAG (ColPali + Qwen2-VL 7B) significantly outperforms text RAG (ColBERT v2 + Llama 3.1 8B), across all different evidence modalities / question hops / # pages. The performance gap is especially big when the evidence involves images, underscoring that M3DOCRAG addresses the information loss over non-textual content by text-only pipelines. We also notice that providing more retrieved pages as context generally increases the performance of both text RAG and M3DOCRAG (using the top 4 pages gives higher performance than the top 1 and 2 pages).

#### 5.2. Closed-domain DocVQA

Multi-modal RAG boosts long document understanding of MLMs. In MMLongBench-Doc, the models must handle a long PDF document (up to 120 pages) for each ques-

- Table 2. Closed-domain DocVQA evaluation results on MMLongBench-Doc. We report the generalized accuracy (ACC) across five evidence source modalities: text (TXT), layout (LAY), chart (CHA), table (TAB), and image (IMG), and three evidence locations: singlepage (SIN), cross-page (MUL), and unanswerable (UNA). The scores from non-RAG methods are from Ma et al. [40].

Evidence Modalities Evidence Locations Overall

Method # Pages

TXT LAY CHA TAB IMG SIN MUL UNA ACC F1

Text Pipeline LMs

ChatGLM-128k [5] up to 120 23.4 12.7 9.7 10.2 12.2 18.8 11.5 18.1 16.3 14.9 Mistral-Instruct-v0.2 [25] up to 120 19.9 13.4 10.2 10.1 11.0 16.9 11.3 24.1 16.4 13.8 Text RAG

ColBERT v2 + Llama 3.1 1 20.1 14.8 12.7 17.4 7.4 21.8 7.8 41.3 21.0 16.1 ColBERT v2 + Llama 3.1 4 23.7 17.7 14.9 24.0 11.9 25.7 12.2 38.1 23.5 19.7

Multi-modal Pipeline Multi-modal LMs

DeepSeek-VL-Chat [38] up to 120 7.2 6.5 1.6 5.2 7.6 5.2 7.0 12.8 7.4 5.4 Idefics2 [33] up to 120 9.0 10.6 4.8 4.1 8.7 7.7 7.2 5.0 7.0 6.8 MiniCPM-Llama3-V2.5 [61, 64] up to 120 11.9 10.8 5.1 5.9 12.2 9.5 9.5 4.5 8.5 8.6 InternLM-XC2-4KHD [15] up to 120 9.9 14.3 7.7 6.3 13.0 12.6 7.6 9.6 10.3 9.8 mPLUG-DocOwl 1.5 [22] up to 120 8.2 8.4 2.0 3.4 9.9 7.4 6.4 6.2 6.9 6.3 Qwen-VL-Chat [4] up to 120 5.5 9.0 5.4 2.2 6.9 5.2 7.1 6.2 6.1 5.4 Monkey-Chat [36] up to 120 6.8 7.2 3.6 6.7 9.4 6.6 6.2 6.2 6.2 5.6 M3DOCRAG

ColPali + Idefics2 (Ours) 1 10.9 11.1 6.0 7.7 15.7 15.4 7.2 8.1 11.2 11.0 ColPali + Qwen2-VL 7B (Ours) 1 25.7 21.0 18.5 16.4 19.7 30.4 10.6 5.8 18.8 20.1 ColPali + Qwen2-VL 7B (Ours) 4 30.0 23.5 18.9 20.1 20.8 32.4 14.8 5.8 21.0 22.6

tion. Since many multi-modal LMs have limited context length, Ma et al. [40] employed a concatenation strategy that combines all screenshot pages into either 1 or 5 images and inputs these concatenated images to multi-modal LMs. Table 2 shows that ColPali + Idefics2 surpass Idefics2 without RAG, as well as all previous multi-modal entries. In addition, ColPali + Qwen2-VL 7B achieves the best scores in overall F1 and most evidence modality/page settings. This demonstrates the effectiveness of multi-modal retrieval over handling many pages by concatenating low-resolution images. As observed in M3DOCVQA experiments, we also notice that providing more retrieved pages as context generally increases the performance of both text RAG and M3DOCRAG (using the top 4 pages gives higher performance than the top 1 page).

M3DOCRAG achieves the state-of-the-art performance in MP-DocVQA. In MP-DocVQA, the models must handle a PDF document of up to 20 pages for each question. Table 3 presents the top-performing entries in the MP-DocVQA test split leaderboard, comparing text-based and multi-modal RAG pipelines. While the text RAG (ColBERT v2 + Llama 3.1) falls short compared to existing approaches, all multi-modal RAG pipelines outperform their text-based counterpart. Notably, the M3DOCRAG pipeline (ColPali + Qwen2-VL 7B) delivers the state-of-the-art results on MP-DocVQA. It is interesting that while the existing entries were fine-tuned specifically for MP-DocVQA, the components used in M3DOCRAG (ColPali or Qwen2-

Table 3. Closed-domain DocVQA evaluation results on MPDocVQA. The RAG methods retrieve a single page to the downstream QA models.

Answer Accuracy Page Retrieval

Method

ANLS R@1 Multimodal LMs

Arctic-TILT 0.8B [10] 0.8122 50.79 GRAM [9] 0.8032 19.98 GRAM C-Former [9] 0.7812 19.98 ScreenAI 5B [3] 0.7711 77.88

Text RAG ColBERT v2 + Llama 3.1 8B 0.5603 75.33 M3DOCRAG ColPali + Qwen2-VL 7B (Ours) 0.8444 81.05

VL 7B) were not tailored to this dataset – although Qwen2VL 7B might have been trained on DocVQA [42], which shares some images with MP-DocVQA.

#### 5.3. Additional analysis

Different page indexing: speed and accuracy. In Table 4, we analyze the speed and accuracy of ColPali+Qwen2-VL 7B pipeline with different document embedding indexing methods. While the naive indexing with exact search (FlatIP) is slow (21s per query), we find that using approximate indexing such as inverted file [52, 66] (IVFFlat) and product quantization [27] (IVFPQ) can retain most of the accuracy, while making the search significantly faster (< 2s per query). We use

- Table 4. Speed-accuracy tradeoff with different indexing strategies on M3DOCVQA. Method: ColPali + Qwen2-VL 7B.

# Pages Indexing

Latency (s) (↓) Accuracy (↑) Retrieval VQA EM F1

1 FlatIP 21.0 1.1 28.9 33.7

- 1 FlatIP + IVFFlat 1.8 1.1 27.9 32.3

- 1 FlatIP + IVFPQ 0.2 1.1 25.9 30.3

- 2 FlatIP + IVFFlat 1.8 2.4 29.9 34.6

- 2 FlatIP + IVFPQ 0.2 2.4 29.0 33.5

4 FlatIP + IVFFlat 1.8 4.8 31.4 36.5 4 FlatIP + IVFPQ 0.2 4.8 29.9 34.7

FlatIP+IVFFlat indexing by default, and users can choose appropriate indexing methods depending on their deployment requirements.

- Table 5. Comparison of different multimodal LMs within M3DOCRAG, evaluated across different document understanding benchmarks. For retrieval, we use the top-1 page from ColPali for all datasets. We use FlatIP+IVFFlat indexing for M3DOCVQA.

Multimodal LMs

M3DOCVQA MMLongBench-Doc MP-DocVQA

F1 (↑) Acc (↑) ANLS (↑)

- Idefics2 8B 27.8 10.8 0.56
- Idefics3 8B 31.8 16.4 0.77 InternVL2 8B 30.9 17.3 0.81 Qwen2-VL 7B 32.3 18.8 0.84

Different multi-modal LMs. In Table 5, we compare four different multi-modal LMs in the M3DOCRAG framework: Idefics2 8B [33], Idefics3 8B [32], InternVL2 8B [12], and Qwen2-VL 7B [59]. The Qwen2-VL 7B model outperforms other MLMs in all three benchmarks. Thus, we use the model as our default MLM component.

- Table 6. Comparison of different multi-modal retrieval models within M3DOCRAG framework, evaluated across different document understanding benchmarks. We provide Qwen2-VL 7B with top-4 pages for MMLongBench-Doc/M3DOCVQA and top1 page for MP-DocVQA from the retrieval models. We use FlatIP+IVFFlat indexing for M3DOCVQA.

M3DOCVQA MMLongBench-Doc MP-DocVQA

Ret. Models

F1 (↑) Acc (↑) ANLS (↑)

ColPali v1 36.5 21.0 0.84 ColQwen v0.1 32.1 21.5 0.86

Different multi-modal retrieval models. In Table 6, we compare two different multi-modal retrival models in M3DOCRAG framework: ColPali v1 and ColQwen v0.1

(see Sec. 4 for details). Both models are trained with the same training objectives but are initialized with different MLM architectures: PaliGemma 2B [7] and Qwen2VL 2B [59], respectively. We find that ColPali achieves significantly better performance in M3DOCVQA, while ColQwen achieves slightly better performance in MPDocVQA and MMLongBench-Doc. Thus, we use ColPali as our default retrieval model.

#### 5.4. Qualitative Examples

In Fig. 5, Fig. 6, and Fig. 7, we provide qualitative examples of M3DOCRAG (ColPali + Qwen2-VL 7B)’s question answering results on several M3DOCVQA examples. In Fig. 5, the answer information is only visually stored within the game logo (‘man is leaning on a motorcycle’), and M3DOCRAG could find the information. In Fig. 6, the question requires multi-hop reasoning across different pages/documents, and M3DOCRAG could combine information from multiple retrieved pages. In Fig. 7, although ColPali did not retrieve the page that contains information about a team whose logo features a bat, Qwen-2 VL leverages its own knowledge ‘Valencia CF has a logo featuring a bat’, and could provide the final answer. Overall, the qualitative examples showcase that M3DOCRAG can successfully tackle different questions whose answer sources exist in various modalities.

### 6. Related Work

Document visual question answering. Mathew et al. [42] proposed document visual question answering (DocVQA) task, where a model extracts information from documents by treating them as images, like in generic visual question answering [1]. Most research on DocVQA focuses on handling a single-page document [22, 23, 30, 34, 41, 42, 55, 58, 63], and it has been now a common practice to include the single-page DocVQA [42] as a part of the image understanding evaluation suite among recent MLMs [7, 12, 20, 32, 46, 59]. Several recent works study applying MLMs for DocVQA on multi-page documents [31, 40, 57]. However, all previous works on DocVQA have focused on handling questions in the context of a specific document, such as “What was the gross profit in the year 2009?” [14, 40, 42, 57]. While this is probably due to the limited context length of the backbone multi-modal LMs, this does not reflect real-world scenarios, where users often ask questions that require information across different pages/documents. We address the limitation and propose M3DOCRAG framework and M3DOCVQA dataset for effective, efficient, and flexible document understanding under various document contexts (closed-domain and open-domain), question hops (singlehop and multi-hop), and evidence modalities (text, chart, figure, etc.).

|Question: “SIE Bend Studio's 2019 game cover has man leaning on what?”<br><br>ColPali + Qwen2-VL 7B: “motorcycle”|
|---|

###### Top 2 pages retrieved by ColPali

|Bend Studio 18languages<br><br>Article Talk Read Edit View history Tools<br><br>Bend Studio<br><br>|[Figure 115]<br><br>Formerly Blank, Berlyn & Co., Inc. (1992–1995) Eidetic, Inc. (1995–2000)<br><br>Company type Subsidiary Industry Video games Founded 1992; 32 years ago Founders Marc Blank<br><br>Michael Berlyn Headquarters Bend, Oregon, US Key people Christopher Reese (studio<br><br>director)<br><br>Products Bubsy 3D Syphon Filter Days Gone<br><br>Number of employees<br><br>150+[1] (2022)<br><br>Parent PlayStation Studios<br><br>(2000–present)<br><br>Website bendstudio.com<br><br>|
|---|
<br><br>|[Figure 116]|
|---|
<br><br>From Wikipedia, the free encyclopedia (Redirected from SIE Bend Studio)<br><br>Bend Studio (formerly Blank, Berlyn & Co., Inc. and Eidetic, Inc.) is an American video game developer based in Bend, Oregon. Founded in 1992, the studio is best known for developing Bubsy 3D, the Syphon Filter series, and Days Gone. Since 2000, Bend Studio is a first-party developer for PlayStation Studios.<br><br>History [edit]<br><br>Marc Blank and Michael Berlyn founded Bend Studio as Blank, Berlyn & Co. in 1992.[2][3] Blank had been a founder and the product development director for Infocom, while Berlyn, an author of adventure games, had previously worked at Infocom before moving to Accolade.[2] Blank was approached by a California company after an employee had used Cornerstone, a software package by Infocom, and remembered that the company also developed games. That company was looking to release a "sound-oriented game machine for cars", for which Blank suggested a series of sports games that would sound like radio broadcasts. The project never went into production and Blank repurposed the idea for an American football video game with an ambiance resembling a TV broadcast. In 1992, he pitched the idea to Berlyn, wondering whether Accolade would be interested in such a title.[2]<br><br>A few months after the 1993 release of Bubsy in Claws Encounters of the Furred<br><br>Kind, when Berlyn was on hiatus at Accolade, they began developing games under the Blank, Berlyn & Co. name. Blank became the president of the new company.[2] The company's first games were the puzzle video games Columbo's<br><br>Mystery Capers and Dell Crossword Puzzles for the Apple Newton. Both were released in November 1993 by StarCore, Apple's publishing label for the Newton.[4][5] Two further such games, Dell Crossword Puzzles and Other Word<br><br>[Figure 117]<br><br>Create account Log in<br><br>|Days Gone 21languages<br><br>Article Talk Read Edit View history Tools<br><br>||Days Gone<br><br>[Figure 118]|
|---|
<br><br>Developer(s) Bend Studio Publisher(s) Sony Interactive<br><br>Entertainment Director(s) John Garvin<br><br>Jeff Ross Producer(s) Darren Yager Designer(s) Ron Allen Programmer(s) John Hoffman Artist(s) Donald Yatomi Writer(s) John Garvin Composer(s) Nathan Whitehead Engine Unreal Engine 4 Platform(s) PlayStation 4<br><br>Windows<br><br>Release PlayStation 4 April 26, 2019 Windows May 18, 2021<br><br>Genre(s) Action-adventure Mode(s) Single-player|
|---|
<br><br>From Wikipedia, the free encyclopedia<br><br>Days Gone is a<br><br>|2019|
|---|
<br><br>action-adventure video game developed by Bend Studio and published by Sony Interactive Entertainment. The game was released for the PlayStation 4 in April 2019. A Windows port was released in May 2021.<br><br>Days Gone is set in post-apocalyptic Oregon two years after the start of a pandemic that turned a portion of humanity into vicious zombie-like creatures. Former outlaw-turned-drifter Deacon St. John discovers his wife Sarah, having been assumed dead, may still be alive and goes on a quest to find her. The game is played from a third-person perspective in which the player can explore an open world environment. Players can use firearms, melee weapons, and improvised weapons, and can use stealth to defend themselves against hostile humans and cannibalistic creatures known as Freakers. A major game mechanic is Deacon's motorcycle, which is used as the player character's main mode of transportation.<br><br>Days Gone was Bend Studio's first open-world project, its first original property since Syphon Filter (1999), and its first development project for home consoles after spending decades working on spinoff games for handheld consoles. The game's development took approximately six years; Bend Studio expanded nearly three-fold to support it. Major sources of inspiration for Days Gone were World War Z, The Walking Dead and Sons of Anarchy. The game was unveiled at E3 2016; its release was originally planned for 2018 but was delayed several times.<br><br>Upon release, Days Gone received mixed reviews from critics, who criticized the game's mission design and technical issues but praised the graphics, artificial intelligence, and Sam Witwer's performance as Deacon, while the story<br><br>[Figure 119]<br><br>[Figure 120]<br><br>Create account Log in|
|---|---|

- Figure 5. Qualitative example of ColPali + Qwen2-VL 7B on M3DOCVQA. Image regions relevant to the question/answer are highlighted with orange boxes. The answer information is only stored visually within the game logo, where a man is leaning on a motorcycle.

|1st|San Felipe Stakes|One and One-Sixteenth Miles|Santa Anita Park|Fast|
|---|---|---|---|---|
|4th|El Camino Real Derby|One and One-Sixteenth Miles|Bay Meadows|Fast|
|4th|Hollywood Futurity|One and One-Sixteenth Miles|Hollywood Park Racetrack|Fast|
|1st|Allowance|One Mile|Oak Tree at Santa Anita Park|Fast|
|2nd<br><br>6th|Norfolk Stakes<br><br>Del Mar Futurity|One and One-Sixteenth Miles<br><br>Seven Furlongs|Oak Tree at Santa Anita Park Del Mar Racetrack|Fast<br><br>Fast|
|1st|Maiden Special Weight|Seven Furlongs|Hollywood Park Racetrack|Fast|
|Privacy policy About Wikipedia Disclaimers Contact Wikipedia Code of Conduct Developers Statistics Cookie statement<br><br>Mobile view<br><br>This page was last edited on 7 October 2023, at 07:43 (UTC). Text is available under the Creative Commons Attribution-ShareAlike License 4.0; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.<br><br>References [edit]<br><br>1. ^ a P Warrior Horse Pedigree<br><br>2. ^ A.P. Warrior sold, retired - NTRA Archived November 25, 2006, at the Wayback Machine<br><br>3. ^ A. P. Warrior : Interactive Stallion Directory<br><br><br>|Categories: 2003 racehorse births Racehorses bred in Kentucky Racehorses trained in the United States Thoroughbred family 4-j<br><br>|
|---|
| | | | |

|A.P. Warrior Addlanguages<br><br>Article Talk Read Edit View history Tools<br><br>A.P. Warrior<br><br>|Sire A.P. Indy Grandsire Seattle Slew Dam Warrior Queen Damsire Quiet American<br><br>Stallion Foaled 2003 Country United States Colour Dark Bay Breeder Jim Fleming Owner Stanley E. Fulton Trainer John Shirreffs Record 13: 4-2-3 Earnings $548,595<br><br>Major wins San Felipe Stakes (2003) La Jolla Handicap (2003)|
|---|
<br><br>Sex<br><br>From Wikipedia, the free encyclopedia<br><br>A.P. Warrior (foaled February 24, 2003 in Kentucky) is a thoroughbred race horse who was a Kentucky Derby contender in 2006 and Grade II winner on both dirt and turf.[1]<br><br>Racing career [edit]<br><br>A son of A.P. Indy and Warrior Queen, he was owned and raced by Stanley E. Fulton, owner of Sunland Park Racetrack. Initially trained by Eoin Harty, John Shirreffs conditioned the horse for his last seven starts. He won four times in twelve starts.<br><br>In 2007, A.P. Warrior was retired and sold[2] to Stonewall Farm in Versailles, Kentucky where he stands for $15,000.[3] In 2011 he moved to Stonewall Farms Ocala Location in Ocala, FL. He died there in early 2016 of a heart attack.<br><br>Races [edit]<br><br>|Finish|Race|Distance|Track|Condition|
|---|---|---|---|---|
|3rd|Oak Tree Derby|One and One-Eighth Miles (Turf)|Oak Tree at Santa Anita Park|Firm|
|1st|La Jolla Handicap|One and One-Sixteenth Miles (Turf)|Del Mar Racetrack|Firm|
|3rd|Swaps Breeders' Cup Stakes|One and One-Eighth Miles|Hollywood Park Racetrack|Fast|
|2nd|Affirmed Handicap|One and One-Sixteenth Miles|Hollywood Park Racetrack|Fast|
|18th|Kentucky Derby|One and One-Quarter Miles|Churchill Downs|Fast|
|3rd|Santa Anita Derby|One and One-Eighth Miles|Santa Anita Park|Fast|
<br><br>[Figure 121]<br><br>Create account Log in|
|---|

|Question: “What distance was the AP Warrior fast race at the Del Mar Racetrack?”<br><br>ColPali + Qwen2-VL 7B: “Seven Furlongs”|
|---|

Top 2 pages retrieved by ColPali

- Figure 6. Qualitative example of ColPali + Qwen2-VL 7B on M3DOCVQA. Image regions relevant to the question/answer are highlighted with orange boxes. The question requires multi-page/document reasoning.

|Question: “What date was a player transferred in to Lorca FC in the 2017-18 season from the club with a logo featuring a bat?”<br><br>ColPali + Qwen2-VL 7B: “11 July 2017”|
|---|

###### Top 1 page retrieved by ColPali

|No. Pos. Nation Player<br><br>1 GK<br><br>|[Figure 122]|
|---|
<br><br>ESP Jaume Valens<br><br>2 DF<br><br>|[Figure 123]|
|---|
<br><br>ESP<br><br>Juan Pedro Pina (Captain)<br><br>3 DF<br><br>|[Figure 124]|
|---|
<br><br>ESP Carlos Pomares<br><br>4 DF<br><br>|[Figure 125]|
|---|
<br><br>ESP Fran Cruz<br><br>5 MF<br><br>|[Figure 126]|
|---|
<br><br>ESP<br><br>Eugeni (on loan from Valencia)<br><br>6 MF<br><br>|[Figure 127]|
|---|
<br><br>ESP Haritz Albisua<br><br>7 MF<br><br>|[Figure 128]|
|---|
<br><br>ESP Carlos Martínez<br><br>8 MF<br><br>|[Figure 129]|
|---|
<br><br>ESP<br><br>Tropi (on loan from Valencia)<br><br>9 FW<br><br>|[Figure 130]|
|---|
<br><br>ESP Chumbi<br><br>10 MF<br><br>|[Figure 131]|
|---|
<br><br>ESP Alberto Noguera<br><br>11 FW<br><br>|[Figure 132]|
|---|
<br><br>ESP Manuel Onwu<br><br>12 MF<br><br>|[Figure 133]|
|---|
<br><br>ESP<br><br>Nando (on loan from Alavés)<br><br>13 GK<br><br>|[Figure 134]|
|---|
<br><br>ESP Francisco Dorronsoro<br><br>14 MF<br><br><br>|[Figure 135]|
|---|
<br><br>ESP Cristian Bustos<br><br>No. Pos. Nation Player<br><br>15 DF<br><br>|[Figure 136]|
|---|
<br><br>SWE Markus Holgersson<br><br>16 MF<br><br>|[Figure 137]|
|---|
<br><br>ESP Sito (on loan from Valencia)<br><br>17 FW<br><br>|[Figure 138]|
|---|
<br><br>ESP<br><br>Manel Martínez (on loan from Girona)<br><br>18 DF<br><br>|[Figure 139]|
|---|
<br><br>ESP Molo<br><br>19 FW<br><br>|[Figure 140]|
|---|
<br><br>URU<br><br>Miguel Merentiel (on loan from Peñarol)<br><br>20 DF<br><br>|[Figure 141]|
|---|
<br><br>ESP Carlos Peña<br><br>21 FW<br><br>|[Figure 142]|
|---|
<br><br>ESP Dani Ojeda<br><br>22 MF<br><br>|[Figure 143]|
|---|
<br><br>ESP Adán Gurdiel<br><br>23 MF<br><br>|[Figure 144]|
|---|
<br><br>ESP Abel Gómez<br><br>24 MF<br><br>|[Figure 145]|
|---|
<br><br>ESP<br><br>Javi Muñoz (on loan from Real Madrid)<br><br>25 GK<br><br>|[Figure 146]|
|---|
<br><br>URU Franco Torgnascioli<br><br>26 DF<br><br><br>|[Figure 147]|
|---|
<br><br>ESP<br><br>José Carlos (on loan from Betis)<br><br>28 FW<br><br>|[Figure 148]|
|---|
<br><br>NGA Manu Apeh<br><br>— DF<br><br>|[Figure 149]|
|---|
<br><br>ESP Antonio López<br><br>|Transfers [edit]<br><br>List of Spanish football transfers summer 2017#Lorca FC|
|---|
<br><br>In [ edit ]<br><br>|Date|Player|From|Type|Fee|Ref|
|---|---|---|---|---|---|
|30 June 2017||[Figure 150]|
|---|
<br><br>Samu Martínez||[Figure 151]|
|---|
<br><br>Hospitalet|Loan return|Free| |
|30 June 2017||[Figure 152]|
|---|
<br><br>Haritz Albisua||[Figure 153]|
|---|
<br><br>Lleida Esportiu|Loan return|Free| |
|30 June 2017||[Figure 154]|
|---|
<br><br>Mikel Fernández||[Figure 155]|
|---|
<br><br>Lleida Esportiu|Loan return|Free| |
|6 July 2017||[Figure 156]|
|---|
<br><br>Jaume Valens||[Figure 157]|
|---|
<br><br>Mallorca B|Transfer|Free|[1]|
|11 July 2017||[Figure 158]|
|---|
<br><br>Tropi||[Figure 159]|
|---|
<br><br>Valencia|Loan|Free|[2]|
|
|---|

- Figure 7. Qualitative example of ColPali + Qwen2-VL 7B on M3DOCVQA. Image regions relevant to the question/answer are highlighted with orange boxes. The VQA component could combine both the retrieved knowledge (Tropi was transferred on 11 July 2017) and its own knowledge (Valencia CF has a logo with a bat) to provide the final answer.

Retrieval-augmented generation. Retrieval-augmented generation (RAG) [35] has emerged as a hybrid approach combining retrieval systems with generative models to improve the quality and relevance of generated content [19]. RAG has been widely studied for open-domain question answering [2, 21, 24, 28, 39, 65], where the community has well-established practices for text-based pipelines. A line of work in VQA studies RAG on visual questions that require world knowledge [11, 44, 51, 62], but their retrieval context is usually generic images and/or short text snippets and does not cover DocVQA settings. To the best of our knowledge, no prior work has explored RAG setting for multi-modal document understanding only with multi-modal models (instead of using OCR methods). Our framework tackles opendomain question answering over documents with complex

multi-modal contexts, including textual, tabular, and visual information across different pages and documents.

### 7. Conclusion

We introduce M3DOCRAG, a novel multi-modal RAG framework that flexibly accommodates various document contexts (closed-domain and open-domain), question hops (single-hop and multi-hop), and evidence modalities (text, chart, figure, etc.). In M3DOCRAG, a multi-modal retrieval model identifies relevant pages from single or multiple documents, which are then processed by a multimodal language model, where all documents are represented as pixels. Next, we introduce M3DOCVQA, the first benchmark that evaluates open-domain multi-modal document understanding capabilities. M3DOCVQA consists of 2,000+ questions and 3,000+ PDF documents, and the questions need to be answered with various modalities such as images, text, and tables. Our experiments in three datasets (M3DOCVQA, MP-DocVQA, and MMLongBench-Doc) demonstrate significant advantages of M3DOCRAG over existing methods, including the state-of-the-art performance in MP-DocVQA. We also provide analysis comparing different indexing strategies, multi-modal LMs, and multi-modal retrieval models. Finally, we show qualitative examples where M3DOCRAG can successfully tackle different questions whose answer sources exist in various modalities. We hope that our work encourages future advancements in multi-modal frameworks for document understanding, paving the way for more robust, scalable, and practical solutions in real-world applications.

### Ethical Considerations

Limitations. Since our multimodal retrieval models and multimodal LMs were trained with English-heavy datasets, they might not understand prompts or documents written in non-English. While our M3DOCRAG framework can benefit many document understanding applications, the model components could present false or biased information. Thus, the framework should be used with human supervision in real-world applications. Note that M3DOCRAG is designed with flexibility so that users can update or replace components as more accurate solutions for each element of the framework become available in the future.

Data collection. We do not involve human subjects during data collection. We do not claim ownership/rights of the Wikipedia documents, and we attribute the source Wikipedia document URLs to all pages.

### References

[1] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi

Parikh. VQA: Visual question answering. In ICCV, 2015. 8

- [2] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection, 2023. 10
- [3] Gilles Baechler, Srinivas Sunkara, Maria Wang, Fedir Zubach, Hassan Mansoor, Vincent Etter, Victor C˘arbune, Jason Lin, Jindong Chen, and Abhanshu Sharma. ScreenAI: A Vision-Language Model for UI and Infographics Understanding, 2024. 7
- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A frontier large vision-language model with versatile abilities. arXiv preprint, abs/2308.12966,

2023. 7

- [5] Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. LongAlign: A recipe for long context alignment of large language models. arXiv preprint, abs/2401.18058, 2024. 7
- [6] Edouard Belval. pdf2image, 2017. 6
- [7] Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Boˇsnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. PaliGemma: A versatile 3B VLM for transfer, 2024. 8
- [8] Ali Furkan Biten, Andres Mafla, Lluis Gomez, Valveny C V Jawahar, and Dimosthenis Karatzas. Scene Text Visual Question Answering. In ICCV, 2019. 6
- [9] Tsachi Blau, Sharon Fogel, Roi Ronen, Alona Golts, Roy Ganz, Elad Ben Avraham, Aviad Aberdam, Shahar Tsiper, and Ron Litman. Gram: Global reasoning for multi-page vqa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15598–15607,

2024. 7

- [10] Łukasz Borchmann, Michał Pietruszka, Wojciech Ja´skowski, Dawid Jurkiewicz, Piotr Halama, Paweł J´oziak, Łukasz Garncarek, Paweł Liskowski, Karolina Szyndler, Andrzej Gretkowski, Julita Ołtusek, Gabriela Nowakowska, Artur Zawłocki, Łukasz Duhr, Paweł Dyda, and Michał Turski. Arctic-TILT. Business Document Understanding at SubBillion Scale, 2024. 7
- [11] Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W Cohen. Murag: Multimodal retrieval-augmented generator for open question answering over images and text. arXiv preprint arXiv:2210.02928, 2022. 10
- [12] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 6, 8

- [13] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024. 6
- [14] Yihao Ding, Siwen Luo, Hyunsuk Chung, and Soyeon Caren Han. Pdfvqa: A new dataset for real-world vqa on pdf documents. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 585–601. Springer, 2023. 1, 2, 8
- [15] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, et al. Internlm-Xcomposer24KHD: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. arXiv preprint, abs/2404.06512, 2024. 7
- [16] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazar´e, Maria Lomeli, Lucas Hosseini, and Herv´e J´egou. The faiss library,

2024. 4, 6

- [17] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, C´eline Hudelot, and Pierre Colombo. ColPali: Efficient Document Retrieval with Vision Language Models, 2024. 1, 4, 6
- [18] Mathieu Fenniak and PyPDF2 Contributors. The PyPDF2 library, version 2, 2022. 1
- [19] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023. 10
- [20] Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. 8
- [21] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM: Retrieval-Augmented Language Model Pre-Training. In ICML, 2020. 10
- [22] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding, 2024. 7, 8
- [23] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In Proceedings of the 30th ACM International Conference on Multimedia, page 4083–4091, New York, NY, USA, 2022. Association for Computing Machinery. 8
- [24] Gautier Izacard and Edouard Grave. Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. In EACL, 2021. 10
- [25] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b, 2023. 7
- [26] Jeff Johnson, Matthijs Douze, and Herv´e J´egou. Billionscale similarity search with gpus. IEEE Transactions on Big Data, 7(3):535–547, 2021. 4, 6
- [27] Herve J´egou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. IEEE Transactions

- on Pattern Analysis and Machine Intelligence, 33(1):117– 128, 2011. 7
- [28] Vladimir Karpukhin, O Barlas, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense Passage Retrieval for Open-Domain Question Answering. In EMNLP, pages 6769–6781, 2020. 10
- [29] Omar Khattab and Matei Zaharia. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR 2020 - Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 39–48, 2020. 4
- [30] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In European Conference on Computer Vision (ECCV), 2022. 8
- [31] Jordy Van Landeghem, Rafał Powalski, Rub`en Tito, Dawid Jurkiewicz, Matthew Blaschko, Łukasz Borchmann, Micka¨el Coustaty, Sien Moens, Michał Pietruszka, Bertrand Ackaert, Tomasz Stanisławek, Paweł J´oziak, and Ernest Valveny. Document Understanding Dataset and Evaluation (DUDE). In ICCV, 2023. 1, 4, 8
- [32] Hugo Lauren¸con, Andr´es Marafioti, Victor Sanh, and L´eo Tronchon. Building and better understanding visionlanguage models: insights and future directions, 2024. 6, 8
- [33] Hugo Lauren¸con, L´eo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models?,

2024. 6, 7, 8

- [34] Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: screenshot parsing as pretraining for visual language understanding. In Proceedings of the 40th International Conference on Machine Learning. JMLR.org, 2023. 8
- [35] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen Tau Yih, Tim Rockt¨aschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In NeurIPS, 2020. 1, 10
- [36] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. arXiv preprint, abs/2311.06607,

2023. 7

- [37] Llama Team. The llama 3 herd of models, 2024. 6
- [38] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. DeepSeek-VL: towards real-world vision-language understanding. arXiv preprint, abs/2403.05525, 2024. 7
- [39] Hongyin Luo, Yung-Sung Chuang, Yuan Gong, Tianhua Zhang, Yoon Kim, Xixin Wu, Danny Fox, Helen Meng, and James Glass. Sail: Search-augmented instruction learning,

2023. 10

- [40] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong,

- et al. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. arXiv preprint arXiv:2407.01523, 2024. 1, 2, 3, 4, 5, 6, 7, 8
- [41] Minesh Mathew, Viraj Bagal, Rub`en P´erez Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V. Jawahar. Infographicvqa. 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2582–2591, 2021. 8
- [42] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 1, 2, 4, 7, 8
- [43] Jamshed Memon, Maira Sami, Rizwan Ahmed Khan, and Mueen Uddin. Handwritten optical character recognition (ocr): A comprehensive systematic literature review (slr). IEEE Access, 8:142642–142668, 2020. 1
- [44] Thomas Mensink, Jasper Uijlings, Lluis Castrejon, Arushi Goel, Felipe Cadar, Howard Zhou, Fei Sha, Andr´e Araujo, and Vittorio Ferrari. Encyclopedic VQA: Visual questions about detailed properties of fine-grained categories. In Proceedings of the IEEE International Conference on Computer Vision, pages 3090–3101, 2023. 10
- [45] Microsoft. Playwright for python, 2021. 5
- [46] OpenAI. Hello gpt-4o, 2024. 5, 8
- [47] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chana, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in PyTorch. In NIPS Workshop, 2017. 6
- [48] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas K¨opf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32 (NeurIPS), 2019. 6
- [49] pdfminer. pdfminer.six, 2019. 1
- [50] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. NAACL 2022 - 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Proceedings of the Conference, pages 3715–3734, 2022. 4, 6
- [51] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge, 2022. 10
- [52] Sivic and Zisserman. Video google: a text retrieval approach to object matching in videos. In Proceedings Ninth IEEE International Conference on Computer Vision, pages 1470– 1477 vol.2, 2003. 2, 4, 7
- [53] Ray Smith. An overview of the tesseract ocr engine. In ICDAR, 2007. 1, 6
- [54] Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant. Multimodalqa: Complex question

- answering over text, tables and images. arXiv preprint arXiv:2104.06039, 2021. 2, 5
- [55] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. Unifying vision, text, and layout for universal document processing, 2023. 8
- [56] The Chromium Project Authors. The chromium projects,

2024. 5

- [57] Rub`en Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multipage docvqa. Pattern Recognition, 144:109834, 2023. 1, 2, 3, 4, 5, 8
- [58] Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, and Xiaomo Liu. DocLLM: A layout-aware generative language model for multimodal document understanding, 2023. 8
- [59] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 4, 6, 8
- [60] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, and Jamie Brew. HuggingFace’s Transformers: State-of-the-art Natural Language Processing. In EMNLP, 2020. 6
- [61] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. LLaVA-UHD: An LMM perceiving any aspect ratio and high-resolution images. arXiv preprint, abs/2403.11703,

2024. 7

- [62] Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, Jure Leskovec, Percy Liang, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. Retrieval-Augmented Multimodal Language Modeling. In ICML, 2023. 10
- [63] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Mingshi Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, Qin Jin, Liang He, Xin Lin, and Feiyan Huang. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. In Conference on Empirical Methods in Natural Language Processing,

2023. 8

- [64] Tianyu Yu, Haoye Zhang, Yuan Yao, Yunkai Dang, Da Chen, Xiaoman Lu, Ganqu Cui, Taiwen He, Zhiyuan Liu, Tat-Seng Chua, and Maosong Sun. RLAIF-V: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness. arXiv preprint, abs/2405.17220, 2024. 7
- [65] Fengbin Zhu, Wenqiang Lei, Chao Wang, Jianming Zheng, Soujanya Poria, and Tat-Seng Chua. Retrieving and reading: A comprehensive survey on open-domain question answering. arXiv preprint arXiv:2101.00774, 2021. 10
- [66] Justin Zobel and Alistair Moffat. Inverted files for text search engines. ACM Comput. Surv., 38(2):6–es, 2006. 2, 4, 7

