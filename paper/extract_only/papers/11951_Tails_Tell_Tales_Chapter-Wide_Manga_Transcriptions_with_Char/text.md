## Tails Tell Tales: Chapter-Wide Manga Transcriptions with Character Names

##### Ragav Sachdeva, Gyungin Shin⋆, and Andrew Zisserman

Visual Geometry Group, Dept. of Engineering Science, University of Oxford

# arXiv:2408.00298v1[cs.CV]1Aug2024

Abstract. Enabling engagement of manga by visually impaired individuals presents a significant challenge due to its inherently visual nature. With the goal of fostering accessibility, this paper aims to generate a dialogue transcript of a complete manga chapter, entirely automatically, with a particular emphasis on ensuring narrative consistency. This entails identifying (i) what is being said, i.e., detecting the texts on each page and classifying them into essential vs non-essential, and (ii) who is saying it, i.e., attributing each dialogue to its speaker, while ensuring the same characters are named consistently throughout the chapter.

To this end, we introduce: (i) Magiv2, a model that is capable of generating high-quality chapter-wide manga transcripts with named characters and significantly higher precision in speaker diarisation over prior works; (ii) an extension of the PopManga evaluation dataset, which now includes annotations for speech-bubble tail boxes, associations of text to corresponding tails, classifications of text as essential or non-essential, and the identity for each character box; and (iii) a new character bank dataset, which comprises over 11K characters from 76 manga series, featuring 11.5K exemplar character images in total, as well as a list of chapters in which they appear. The code, trained model, and both datasets can be found at: https://github.com/ragavsachdeva/magi

|Magi|
|---|

|Magiv2|
|---|

Output Input Output

[Figure 1]

|[Figure 2]<br><br>Input|
|---|

[Figure 3]

[Figure 4]

[Figure 5]

- <1>: Read this way

- <1>: A pistol, eh? Is that so…
- <2>: Are you doubting me?
- <3>: Calm, down Luffy!
- <4>: Let's just have a good time!
- <5>: Yeah! Pirates always have a good time!
- <6>: The sea is vast and there's lots of islands to explore!
- <7>: And best of all, pirates have freedom!

- <2>: Wow! <1>: You guys stop filling his head with crazy ideas. <5>: But it's true! <4>: Right!?

Shanks: A pistol, eh? Is that so… Luffy: Are you doubting me? Other: Calm, down Luffy! Yasopp: Let's just have a good time! Lucky Roux: Yeah! Pirates always have a good time!

... Lucky Roux: But it's true! Yasopp: Right!?

... Makino: Ha Ha Ha! We'll celebrate together when you return. Luffy: Hee hee

[Figure 6]

|- Single Page<br>- Numerical labels instead of character names<br>- Lower precision for speaker diarisation<br>|
|---|

|- Multi-Page<br>- Principal characters are named consistently across pages<br>- Higher precision for speaker diarisation<br>|
|---|

[Figure 7]

Fig. 1: (Left) Magi [46] generates a page-level transcript, with non-essential texts and without character names. (Right) Magiv2 (ours) generates chapter-wide transcripts with principal characters consistently named across pages, higher precision for speaker diarisation and only dialogue-essential texts.

⋆ Core contribution

### 1 Introduction

Manga, a Japanese form of comic art, is celebrated globally for its rich narratives and distinctive graphical style. It engages millions of readers through its compelling visuals and complex character development. However, this visually dependent medium poses significant accessibility challenges for people with visual impairments (PVI). Recent advances in computer vision and machine learning present an opportunity to bridge this gap.

Despite the potential, there has been limited research in the field of improving manga accessibility for visually impaired readers. One notable recent work by Sachdeva and Zisserman [46] addresses the problem of automatically generating transcriptions for manga images. Their contributions include the development of Magi, a model capable of processing a high-resolution manga page to detect characters, texts, and panels, as well as predict character clusters and associate dialogues to their respective speakers. Additionally, they introduced two new datasets, Mangadex-1.5M and PopManga, for training and evaluation purposes.

While the Magi model represents a promising initial step, it remains far from being practically usable due to several critical limitations. First, a major shortcoming is its failure to incorporate character names within the generated transcripts, instead denoting different characters with numerical labels such as 1,

##### 2, 3, etc. As the transcripts are generated on a page-by-page basis, this approach inevitably leads to inconsistent character numbering across different pages. To improve the readability, it is essential to generate chapter-wide transcripts with consistent character names, since numerical labels are non-intuitive and make the transcripts difficult to follow. Second, Magi struggles to reliably associate text with the appropriate speaker, often attributing dialogues to the wrong characters. This misattribution disrupts the flow of conversation, leading to a disjointed and confusing narrative. Improving the association of text with the correct speaker is crucial to maintain the coherence of the dialogue and prevent reader confusion. Third, it is inept at distinguishing between essential and non-essential texts for the dialogue. Non-essential text, such as scene descriptions (e.g., street signs, graffiti, product labels) and sound effects (e.g., “Thud,” “Whoosh”), should not be attributed to any character and, if improperly included as dialogues in the transcript, can disrupt the narrative flow.

To address these three limitations, we propose Magiv2, a robust and enhanced model capable of generating chapter-wide manga transcripts with consistent character names. Fig. 1 shows the comparison of our model with Magi [46]. Our approach is built upon several key insights. First, recognising the challenges of character naming in manga transcripts—both providing names and ensuring their consistency—Magiv2 leverages a character bank featuring names and images of principal characters and utilises a training-free, constraint optimisation method to consistently name all characters across the entire chapter, significantly outperforming traditional clustering-based methods. Second, we note that speech-bubble tails are crucial visual cues, intended by manga artists to indicate who is speaking. By making our model tail-aware, we significantly enhance text-to-speaker association performance. Third, we observe that distinguishing

between essential and non-essential texts can largely be accomplished visually due to differences in font styles and the placement of texts. We leverage this prior and introduce a lightweight text-classification head on top of our visual backbone that can effectively differentiate dialogues from non-dialogue texts.

In summary, we make the following contributions: (i) We introduce a state-ofthe-art model, Magiv2, capable of generating comprehensive manga transcripts across entire chapters, complete with character names and enhanced speaker associations. (ii) We extend the PopManga evaluation dataset by incorporating annotations for character names, speech bubble tails, text-to-tail associations, and text classification. This dataset is used to evaluate the performance of the new capabilities of Magiv2, such as character recognition. (iii) We release a new, meticulously curated character bank dataset for 76 manga series, encompassing more than 11K principal characters, with 11.5K exemplar character images in total. Additionally, this dataset includes detailed metadata such as the series names and specific chapters where each character appears.

With these contributions, over 10,000 published manga chapters (from series in PopManga) can now be transcribed directly using our model, with the potential for more as the character bank dataset grows.

### 2 Related Work

Comic understanding. Using computer vision to analyse and understand comic books has been extensively explored. Several datasets have been contributed to facilitate this research including Manga109 [1,3,23], DCM [35], eBDtheque [11], PopManga [46] etc. There are several existing works that propose solutions for panel detection [13, 36–38, 43, 46, 52], text/speech balloon detection [3, 36, 37, 39,46], depth-estimation [5], character detection [17,18,37,46,50], character reidentification, [40,46,48,51,54], speaker identification, [43,46], captioning [41], and transcript generation [46].

Person identification using a character bank. Identifying and naming people in images or videos has been a long studied research problem. Often times this either requires complex reasoning, e.g. inferring the name of a person based on how other people address them, or prior context and memory, e.g. a person may have been introduced previously and this information needs to be remembered. Given the complexity of this task, a common approach is to rely on an external character bank which can be used for matching query character images with a gallery [4,6,15,16,25,34,53].

Comic Accessibility for PVI users. Several efforts have been made to understand the challenges faced by PVI when accessing comics [21, 42, 47] and solutions have been proposed in the form of tactile books [32], textured images [8], audiobooks [49] etc. Recent works have also explored the use of computer vision and machine learning to automatically caption simple comic strips [41] and generate dialogue transcripts of more complex ones [46].

|1. Detection and Association|
|---|

|2. Chapter-Wide Character Naming|
|---|

|3. Transcript Generation|
|---|

Character Bank

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

|[Figure 16]|
|---|

[Figure 17]

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]|
|---|

|[Figure 21]|
|---|

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

[Figure 27]

[Figure 28]

[Figure 29]

Kie: Tanjiro! Kie: Your face is pitch-black. Kie: Come here. Kie: You don’t have to go. Kie: It’s snowing and it’s dangerous. Tanjiro: I want to us to have a great new year’s feast, so I’ll go sell as much charcoal as I can… Tanjiro: …even if it’s just a little. Kie: Thank you. … Nezuko: I was putting Rokuta to

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Giyu Tomioka

Nezuko Kamado

Tanjiro Kamado

Takeo Kamado

Shigeru Kamado

Sakonji Urokodaki

Hanako Kamado

Rokuta Kamado

Kie Kamado

Saburo

[Figure 36]

Tanjiro

Chapter-Wide Crops

Assigned Names

|Must Link<br><br>|[Figure 37]|
|---|
<br><br>|[Figure 38]|
|---|
<br><br>|[Figure 39]|
|---|
<br><br>|[Figure 40]|
|---|
<br><br>|[Figure 41]|
|---|
<br><br>|[Figure 42]|
|---|
<br><br>|[Figure 43]|
|---|
<br><br>|[Figure 44]|
|---|
|Cannot Link<br><br>|[Figure 45]|
|---|
<br><br>|[Figure 46]|
|---|
<br><br>|[Figure 47]|
|---|
<br><br>|[Figure 48]|
|---|
<br><br>|[Figure 49]|
|---|
<br><br>|[Figure 50]|
|---|
<br><br>|[Figure 51]|
|---|
<br><br>|[Figure 52]|
|---|
<br><br>|
|---|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

Tanjiro Kamado

[Figure 59]

|[Figure 60]|
|---|

|[Figure 61]|
|---|

Kie

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

[Figure 66]

Kie Kamado

Tanjiro Kie

[Figure 67]

Constraint Optimisation

|[Figure 68]|
|---|

|[Figure 69]|
|---|

...

|[Figure 70]|
|---|

|[Figure 71]|
|---|

Hanako Kamado

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

sleep. …

|[Figure 75]|
|---|

[Figure 76]

|[Figure 77]|
|---|

Shigeru Kamado

Kie

[Figure 78]

Tanjiro

- Fig. 2: Inference pipeline. Given a manga chapter, along with a character bank: (1) each page is processed independently to detect various elements and their relationships, such as character and text boxes, and their association. Next, (2) using the character bank, names are assigned to detected character crops across all pages using a constraint optimisation approach. Finally, (3) the transcript is generated by performing OCR, ordering all the texts and removing non-essential texts.

### 3 Overview of the Chapter-Wide Inference

Given a manga chapter, our goal is to generate a transcript of all pages while ensuring narrative consistency. However, processing all pages simultaneously to generate a dialogue transcript in a single forward pass is computationally prohibitive (a typical manga chapter comprises 15 to 30 pages), necessitating a segmented approach. To mitigate the computational burden and efficiently generate a chapter-wide transcript with accurate speaker attribution, we employ the following three-step process. The complete inference pipeline is shown in Fig. 2.

- 1. Detection and Association. The initial step involves processing each manga page independently, framed as a graph generation problem. This step is similar to [46], but with modifications to incorporate additional elements. In our graph, “nodes” represent bounding boxes of detected characters, texts, panels, and notably, speech bubble tails. The “edges” represent pairwise relationships between character-character, text-character, and text-tail. More details regarding the architecture and training strategy are described in Sec. 4.
- 2. Chapter-Wide Character Naming. Given the crops of detected characters from all pages of a manga chapter, along with a character bank comprising images and names of principal characters, the goal is to assign each character crop to the correct principal character in the character bank, if one exists, otherwise assign it to the “other” class. This step simultaneously allows naming of speakers in the final transcripts, and consistent character identification across pages, if the assignments are correct. We formulate this as a constraint optimisation approach and provide more details in Sec. 5.
- 3. Transcript Generation. Finally, the gathered information is compiled to generate the chapter-wide transcript. This is a four step process: First, the detected text boxes are filtered such that the texts that are classified as non-essential are removed; Second, the remaining text boxes are organised in their reading order

|Node Prediction|
|---|

|Edge Prediction|
|---|

|[Figure 79]<br><br>[Figure 80]|
|---|

|[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]|
|---|

[Figure 85]

✔ ✔ ✔

Character Text Panel

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

Char-Char + Text-Char + Text-Tail

|[Figure 89]|
|---|

|[Figure 90]|
|---|

[Figure 91]

|[Figure 92]|
|---|

Box + Class

|[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

[Figure 99]

Transformer

|[Figure 100]<br><br>[Figure 101]|
|---|

✘

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

Tail

- Fig. 3: Simplified Detection and Association Architecture. The input to the model is an RGB image of a manga page. The transformer decoder outputs several feature vectors which are used to predict bounding boxes for characters, texts, panels and tails (“nodes”). These features are further processed in pairs to predict charactercharacter, text-character and text-tail associations (“edges”).

(by first sorting the pages, then sorting panels on each page [46] and finally sorting texts within each panel [14]); Third, Optical Character Recognition (OCR) is used to extract texts from the manga pages; Finally, the transcript is generated by utilising the text-character associations predicted in Sec. 4 and character names predicted in Sec. 5. We provide more details in supp. mat.

- 4 Detection and Association

Given a manga page, the objective of this section is to detect the various components that constitute a manga page—particularly the panels, characters, text blocks and tails (i.e., localise where they are on the page), and also to associate them: character-character association (i.e., character clustering), text-character association (i.e. speaker diarisation), and text-tail association. We cast this as generating a graph, as described in step 1 of the inference process of Sec. 3. The model architecture for this detection and association is illustrated in Fig. 3.

Briefly, the model ingests a high resolution manga page as input, which is first processed by a CNN backbone, followed by a transformer encoder-decoder resulting in N object feature vectors. These feature vectors are processed by the detection head to regress a bounding box and classify it into character, text, panel, tail or background. This completes the nodes part of the graph. To generate text-character edges, the features corresponding to detected text boxes and character boxes are processed in pairs by a speaker association head to make a binary prediction (whether the edge exists or not). Similarly, a tail association head processes pairs of text and tail feature vectors resulting in text-tail edges. Character-character edges are obtained by processing pairs of detected character feature vectors along with their respective crop embedding feature vectors (obtained by a separate crop embedding module). Finally, the feature vectors corresponding to detected texts are processed by a linear layer to classify them into essential vs non-essential. Further details regarding the model architecture and implementation are provided in Sec. 4.2.

#### 4.1 Semi-Supervised Training

A significant challenge in training the graph generation model is the quality and completeness of the training data. We utilise two datasets: (i) Mangadex-1.5M, which is unlabelled, and (ii) PopManga (Dev), which is partially labelled1.

To address the challenge of partially annotated training data, we approach the training of our graph generation model through semi-supervised learning. We begin by curating a small subset of the PopManga (Dev) set, and endow it with both tail-related annotations and text classification labels. Additionally, we extract partial labels for the Mangadex-1.5M dataset using Magi [46], which of course lack tail-related and text classification annotations. Given this combination of data—small subset with complete annotations and larger subsets with partial pseudo-annotations—we adopt the following training strategy.

Initially, we warm up our model by training it on the large-scale partiallylabeled pseudo-annotations. We then fine-tune this model on the smaller, fully annotated dataset. Subsequently, this model is used to mine complete, but possibly noisy, annotations for the large-scale data. We then re-train our model from scratch on this newly annotated large-scale data, which remains pseudoannotated but now more comprehensively annotated, followed by additional finetuning on the small, completely annotated data. This cycle is repeated multiple times to refine our model. Detailed training recipe is provided in the supp. mat.

This training approach ensures robust detection and association capabilities despite the incomplete initial annotations. This style of training paradigm is common in noisy label learning literature, where training the model on noisy but large scale data (in our case pseudo-annotations), followed by fine-tuning on clean data, and then re-mining the pseudo-annotations, results in improved training data, which in turn can be used to train a better model [2, 7, 22, 44, 45]. A crucial aspect of this methodology is the re-training of the model from scratch after each round of mining pseudo-annotations, which is essential to avoid confirmation bias and prevent the model from overfitting on its own predictions.

#### 4.2 Implementation

The graph generation model architecture consists of a ResNet50 [12] backbone, followed by a encoder-decoder transformer with 6 layers each, hidden dimension of 256, 8 attention heads and conditional cross-attention [31]. The crop-embedding module is an encoder-only transformer with 12 layers, hidden dimension of 768, and 12 attention heads. The text-character, text-tail and character-character edge prediction heads are all 3-layered MLPs, and the textclassification head is a simple linear layer. Our training objective for box prediction is the same as in [31]. We further apply Binary Cross Entropy loss to the outputs of our edge-prediction as well as text-classification heads. Additionally, we apply Supervised Contrastive Loss [19] to the per-page embeddings

1 All pages contain character boxes, text boxes, and character clusters, but only a subset of pages include text-to-character associations. Furthermore, none of the pages have annotations for speech bubble tails or labels for text classification.

from the crop-embedding module. We trained our model, on 2×A40 GPUs using AdamW [29] optimiser with both learning rate and weight decay of 0.0001, and batch size of 16.

### 5 Chapter-Wide Character Naming

The objective in this section to assign each detected character crop in the chapter to one of the characters in the character bank (introduced in Sec. 6), unless they are “other”. This is the second, chapter-wide, step of the inference process.

The question is, how to optimise this assignment objective? Naively, this can be accomplished greedily by computing the similarity of each crop with each character in the character bank and taking the argmax. However, we can do better, by leveraging additional constraints (must-link and cannot-link) from per-page associations. Specifically, the graph computed in Sec. 4, provides us with character-character edges which can be transformed into must-link constraints, i.e. these crops are of the same characters and must be assigned the same identity, and cannot-link constraints, i.e. these crops are of different characters and must be assigned to different identities. These per-page must-link and cannot-link constraints provide a stronger signal than simple crop-based similarity as they factor in surrounding visual cues, e.g. two characters in the same panel are likely to be different characters, regardless of the visual similarity. This assignment problem can be formulated as a Mixed Integer Linear Programming problem, for which there are several existing solvers e.g. COIN-OR Branch and Cut Solver (CBC) [10,30].

Problem Definition. Formally, suppose there are n character crops in a particular chapter and k characters in the character bank. Additionally, suppose that we have a set of must-link constraints M, representing pairs of crops that must be assigned to the same character, and a set of cannot-link constraints C, representing pairs of crops that must not be assigned to the same character. We further define a (k+1)th character, which is a dummy character to capture “other” when the crop is of a character that is not in the character bank.

Variable. Let xij be a binary variable that equals 1 if character crop i is assigned to character j in the character bank, and 0 otherwise.

Objective Function. The objective is to compute the optimal assignment of crops to characters in the character bank, i.e. computing xij, which is achieved by

n

min

x

i=1

k+1

dijxij, where dij =

j=1

η if j = k + 1, ∥ei − ej∥ otherwise

(1)

and ei,ej are embeddings for crop i and character j, respectively, and η is a fixed outlier-threshold hyperparameter (in practice, η = 0.75).

Constraints. The objective function above is minimised subject to the following constraints:

k+1

xij = 1, ∀i ∈ {1,...,n} (2)

j=1

xu,j − xv,j = 0, ∀(u,v) ∈ M, ∀j ∈ {1,...,k + 1} (3)

xu,j + xv,j ≤ 1, ∀(u,v) ∈ C, ∀j ∈ {1,...,k} (4)

where Eq. (2) ensures that each crop is assigned to exactly one character, Eq. (3) enforces the must-link constraints, and Eq. (4) enforces the cannotlink constraints.

Note that in Eq. (4), j ̸= k + 1. This is because there may be two different characters (hence must not be linked) that are not in the character bank (hence must be linked to “other”). In other words, a cannot link constraint between crops u,v is applied such that these crops must not be assigned to the same character, unless they are assigned to “other” i.e. (k + 1)th character. In Sec. 7, we compare the proposed constraint optimisation approach with traditional clustering based approaches and demonstrate that our method significantly outperforms the baselines.

### 6 Datasets: PopCharacters and PopManga-X

The recently introduced PopManga [46] dataset provides annotations for character boxes, text boxes, per-page character clusters and speaker associations. It is divided into three splits: Dev, Test-S and Test-U, with around 2000 images of manga pages in Test, and S & U meaning that other chapters from the series are Seen or Unseen during training.

In this section we detail two data related contributions: (a) We compile a character bank of principal manga characters in PopManga. Please see Fig. 4 for some examples and dataset statistics; (b) We extend the annotations of the PopManga test set to facilitate the evaluation of the new Magiv2 model capabilities, such as character labelling. Please see Fig. 5 for an overview of the extended test dataset along with statistics on various types of available annotations.

PopCharacters. We introduce a new character bank dataset, called PopCharacters, comprising principal characters2 in PopManga. For each principal character in PopCharacters, we provide (i) the character’s name, (ii) a set of webscraped thumbnail images of the character, and (iii) the series it belongs and a list of manga chapters the character appears in. Additionally, for a subset of the characters in PopCharacters, which appear far more frequently than others, we also provide a set of exemplar images queried from within the manga chapters

2 We define principal characters as those who play crucial roles in the main story of a series. Please refer to the supplementary material for more details.

[Figure 105]

- Fig. 4: PopCharacters—the proposed character bank dataset. (Top) We show exemplars for eight different characters from PopCharacters, along with their name (in red), followed by the name of the series and the list of chapters they appear in. (Bottom) We display histograms showing the number of characters (left), the number of frequently-occurring characters with additional exemplar images (middle), and the average number of exemplars per frequently-occurring character (right) in each series for PopCharacters.

and verified by human-in-the-loop. The purpose of this dataset is two-fold: (i) it enables Magiv2 to transcribe hundreds of thousands of manga pages (that are commercially available), with names for principal characters; and (ii) it provides a valuable resource for training models on tasks such as character recognition and character clustering. Further details on the dataset curation process is provided in the supp. mat.

PopManga-X. In the PopManga test splits (i.e., Test-S and Test-U), we provide new annotations for speech-bubble tail bounding boxes, associations of text boxes to tail boxes, and text categories (essential vs non-essential), providing the test-bed for tail-related predictions and text classification (see Fig. 5). Moreover, we label each character box in the test splits with the name of the character (consistent with PopCharacters), thereby offering global character clusters across the series and permitting the evaluation of chapter-wide character cluster predictions. To differentiate this extended dataset with more types of annotations from the original, we call this PopManga-X. Further details are provided in supp. mat.

|PopManga-X|
|---|

|PopManga|
|---|

|[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]|
|---|

|[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]|
|---|

Tail boxes

[Figure 116]

Character boxes

[Figure 117]

Character Names

[Figure 118]

[Figure 119]

Text boxes

Text – Character Associations

Non-essential texts

[Figure 120]

[Figure 121]

[Figure 122]

Per-Page character IDs

Text - Tail Associations

|Stats|
|---|

Essential Non-Essential

65% Principal Characters

35% Non-Principal Characters

[Figure 123]

[Figure 124]

|[Figure 125]|
|---|

w/otailw/tail

[Figure 126]

|[Figure 127]|
|---|

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

|[Figure 136]|
|---|

|[Figure 137]|
|---|

40% 01% 35% 24%

|[Figure 138]|
|---|

|[Figure 139]|
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

18k+ character boxes 7k+ tail boxes

20k+ text boxes

- Fig. 5: PopManga-X. (Top-Left) Ground-truth annotations in PopManga. (TopRight) Ground-truth annotations added to test splits of PopManga, now referred to as PopManga-X. (Bottom) Statistics on various elements of PopManga-X test splits.

### 7 Results

To achieve high-quality chapter-wide manga transcriptions, we identify three core tasks in Sec. 3: (i) per-page detection and association, (ii) chapter-wide character naming, and (iii) transcript generation. In the following, we report our model’s performance on the first two tasks and note that their quality directly determines the quality of the third. In other words, to evaluate the quality of the generated transcript, it is enough to evaluate the first two tasks only, as they reflect the correctness of the transcript. Qualitative results are shown in Fig. 6.

#### 7.1 Detection and Association/Graph Generation

Given a single manga page as input, here we evaluate the performance of the graph generation model in terms of (i) predicting panels, texts, characters, and tails, (ii) predicting text-character edges, text-tail edges, character-character edges, and (iii) classifying predicted texts into essential vs non-essential.

Nodes: To measure the quality of predicted “nodes” i.e. predicted bounding boxes, we use the standard object detection evaluation measures. In Tab. 1,

|Detection and Association|
|---|

|[Figure 152]|[Figure 153]|[Figure 154]|[Figure 155]|
|---|---|---|---|

|Character Naming|
|---|

|[Figure 156]|[Figure 157]|[Figure 158]|
|---|---|---|

|[Figure 159]<br><br>|[Figure 160]<br><br>Giant Fish|
|---|
<br><br>|[Figure 161]<br><br>Pterodactyle|
|---|
<br><br>|[Figure 162]<br><br>Shenron|
|---|
<br><br>|[Figure 163]<br><br>Son Goku|
|---|
<br><br>Character Bank<br><br>Bulma|
|---|

| |[Figure 164]|[Figure 165]|[Figure 166]|
|---|---|---|---|
|[Figure 167]|[Figure 168]|[Figure 169]|[Figure 170]|

Character Bank

[Figure 171]

[Figure 172]

Ging Freecs

[Figure 173]

[Figure 174]

Gon Freecs

Ging’s Grandma

[Figure 175]

Noko

|[Figure 176]|
|---|

[Figure 177]

Kon

[Figure 178]

Mito Freecs

Kite

Master of the swamp

|[Figure 179]|[Figure 180]|
|---|---|
| | |

|[Figure 181]|[Figure 182]|
|---|---|

|Transcripts|
|---|

[Figure 183]

[Figure 184]

<Yūma Kuga>: I'm hungry. Let's get something to eat. <Osamu Mikumo>: Something to eat? Do you even have Japanese money? <Yūma Kuga>: Sure I do. I haven't even used any yet. UMM...

<Taiju Oki>: Listen up, Senku!! There's no stopping me! It's gotta be today!! After five long years of having feelings for Yuzurika... I'm finally gonna confess my love!! <Senku Ishigami>: Hmm... inter- eating...very inter- eating. I'll be cheering for you so hard from here in the science lab that my vocal cords <Taiju Oki>: Oh yeah? Thanks, Senku! <Senku Ishigami>: silence. I won't cheer even one millimeter for you, you big oaf <Taiju Oki>: Wait, so which is it?! <Senku Ishigami>: A fool who takes five whole years to say anything is the epitome of absurdity. <Taiju Oki>: Allow me to provide a method so rational it'll kill you. <Senku Ishigami>: This will send your pheromone production

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

<Osamu Mikumo>: See? ...?! <Yūma Kuga>: "I only have paper money... But I ought to be able to get something with 100 of these. <Osamu Mikumo>: Y- You idiot! Put that away! Come on! <Yūma Kuga>: ??! <Osamu Mikumo>: I can't believe it...! Even if it is his first time in Japan, how ignorant can you get? <Yūma Kuga>: Now we've got a totally different problem on our hands! <Other>: Listen, Kuga. <Osamu Mikumo>: Don't wave money around in public. It'll cause trouble.

into overdrive. Basically, it's a love option. Your success is ten billion percent assured if you drink this! <Taiju Oki>: Thanks, Senki, but... ...No thanks! I can't go and cheat my way into her heart. That's right. Like the honest and upright dude I am... ...I told Yuzurtha to meet me under the Camphor Tree. That's where I'll confess!! <Other>: Was that really a love option, Senki...? <Senku Ishigami>: Of course not, it's just ordinary gasoline. I produced it from plastic bottle caps. Just think about the atomic structure of poly- ethylene, you fools! It's merely long gasoline molecules chopped up by a few hydrocarbons. Simple enough to <unsure>: I Dun Geddit. <Other>: So Taiju would've died if he drank it? Huh? <Senku Ishigami>: Heh heh... I was ten billion percent sure he wouldn't drink it. He's an honest fool

<Yūma Kuga>: Hm...? is that so? <Osamu Mikumo>: You saw those people whispering around us... !??

<Yūma Kuga>: Oops, sorry... <Other>: Oww, that hurt! Dude, I think that broke my leg. Oh yeah, it's totally broken. <Osamu Mikumo>: ...?! <Other>: What'd you do that for, squirt? You think you can

walk off scot- free? I'd guess 50,000... No, 100,000 should cover it. <Osamu Mikumo>: These punks... They must've seen the wad of cash... <Yūma Kuga>: You want money...? to go to the doctor? <Osamu Mikumo>: don't listen to him! There's no way he got hurt!

- Fig. 6: Qualitative Predictions. (Top) We show the predictions from our graph generation model—panels (in green), character (in blue), texts classified as essential (in red), and speech bubble tails (in purple). The text-character edges (dashed red lines), text-tail edges (dashed purple lines) and character-character edges (unique colour for each connected component) are also shown. (Middle) We show the prediction for character names across multiple pages of two different manga series, demonstrating character naming consistency. (Bottom) We show the final, generated, multi-page transcripts using our method.

we report the results for the average precision metric [24] for detecting character boxes, text boxes, tail boxes and panel boxes. We compare our results against, Magi [46], DASS [50] and zero-shot results from GroundingDino [27], on PopManga-X and Manga109 [1].

Edges: Our model is trained to predict three different kinds of edges: (i) charactercharacter, (ii) text-character and, (iii) text-tail. To evaluate the character-character edges, we treat it as a per-page clustering problem and report the same metrics as [46], namely AMI, NMI, P@1, R-P, MRR and MAP@R, on PopManga-X and Manga109 [1], in Tab. 2. For text-character and text-tail edges, we treat it as a binary classification problem (whether the predicted edge is correct or not) and report the average precision metric on PopManga-X in Tab. 3. We compare our results against [46] and provide more details on the edge evaluation procedure in the supp. mat.

- Table 1: Detection Results. We report the average precision results, which have an upper bound of 1.0.

|PopManga-X (Test-S)| | | |PopManga-X (Test-U)<br><br>| | |Manga109| |
|---|---|---|---|---|---|---|---|---|
|method<br><br>|Char|Text<br><br>|Tail<br><br>|Char<br><br>|Text|Tail<br><br>|Body<br><br>|Panel|
|DASS [50] Grounding-DINO [27] Magi [46] Magiv2 (Ours)<br><br>|0.8410 0.7250 0.8485 0.8544|0.7922 0.9227 0.9372<br><br>|0.8766<br><br>|0.8580 0.7420 0.8615 0.8720<br><br>|0.8301 0.9208 0.9353<br><br>|0.8737<br><br>|0.9251 0.7985 0.9015 0.9046|0.5131 0.9357 0.9405<br><br>|

Text Classification: Finally, we evaluate the performance of our model on categorising the detected texts into essential vs non-essential. In this work, we classify a text as essential, if it is a spoken dialogue, interjection or an internal thought by a character, or context added by the narrator. Everything else is non-essential e.g. sound-effects, editorial footnotes, scene-texts etc. We report the average precision results on PopManga-X in Tab. 3. We use the confidence scores from Magi [46] as a baseline, while acknowledging that it was not explicitly trained for this task.

Discussion. When compared with prior works, our model achieves (i) better per-page character clustering results, particularly in the crop-only setting, which we attribute to the semi-supervised learning training scheme, where mining better pseudo labels in turn improves the model’s performance; (ii) significant improvement in speaker diarisation (i.e. text-character matching) results, which is largely attributed to the introduction of speech-bubble tails; (iii) comparable bounding box detection results. Furthermore, our work unlocks new functionality, not supported by prior works, including (i) detecting tail boxes, text-tail associations, and (ii) text classification into essential vs non-essential, which can be used to improve the quality of the generated transcripts.

- Table 2: Per-Page Character Clustering Results. We report results using several metrics. They all have an upper bound of 1.0.

method AMI NMI MRR MAP@R P@1 R-P PopManga-X (Test-S)

|Magi (crop only) [46] Magiv2 (crop only) (Ours)<br><br>|0.4892 0.5826|0.7178 0.8120<br><br>|0.9008 0.9275|0.7840 0.8401<br><br>|0.8423 0.8831<br><br>|0.8008 0.8526<br><br>|
|---|---|---|---|---|---|---|
|Magi [46] Magiv2 (Ours)<br><br>|0.6574 0.6745|0.8501 0.8610<br><br>|0.9312 0.9431<br><br>|0.8439 0.8669<br><br>|0.8884 0.9066|0.8555 0.8770<br><br>|

PopManga-X (Test-U)

|Magi (crop only) [46] Magiv2 (crop only) (Ours)|0.4862 0.5711<br><br>|0.7326 0.8108|0.9061 0.9321<br><br>|0.7926 0.8491|0.8477 0.8898<br><br>|0.8076 0.8598|
|---|---|---|---|---|---|---|
|Magi [46] Magiv2 (Ours)|0.6527 0.6650<br><br>|0.8503 0.8579<br><br>|0.9347 0.9508<br><br>|0.8557 0.8818<br><br>|0.8936 0.9202<br><br>|0.8656 0.8898|

Manga109 (Body)

|Magi (crop only) [46] Magiv2 (crop only) (Ours)|0.5690 0.6204<br><br>|0.7694 0.8152<br><br>|0.9237 0.9400<br><br>|0.8259 0.8646|0.8721 0.9002<br><br>|0.8389 0.8737<br><br>|
|---|---|---|---|---|---|---|
|Magi [46] Magiv2 (Ours)|0.6345 0.6456<br><br>|0.8202 0.8336<br><br>|0.9383 0.9514|0.8567 0.8812<br><br>|0.8966 0.9179<br><br>|0.8667 0.8895|

- Table 3: Text-Related Results. We report the average precision results, which have an upper bound of 1.0.

|method<br><br>|Text - Character Association<br><br>| |Text - Tail Association<br><br>| |Text Classification| |
|---|---|---|---|---|---|---|
| |PopManga-X (Test-S)|PopManga-X (Test-U)|PopManga-X (Test-S)<br><br>|PopManga-X (Test-U)|PopManga-X (Test-S)|PopManga-X (Test-U)|
|Magi [46]<br><br>|0.5248|0.5632<br><br>|-<br><br>|-|0.9617<br><br>|0.9692|
|Magiv2 (Ours)|0.7499|0.7512|0.9838<br><br>|0.9830<br><br>|0.9897<br><br>|0.9914|

#### 7.2 Chapter-Wide Character Naming/Character Identification

Here we evaluate the efficacy of our method in forming chapter-wide character clusters and evaluate whether the same characters across pages are assigned the same name. For evaluation, we utilise test splits of PopManga-X where the input this time is an entire manga chapter. There are 50 chapters in the two test sets in total. For each chapter, we curate a chapter-specific character bank from the PopCharacters dataset, comprising principal characters that appear in this chapter. This chapter-specific character bank consists of names of principal characters along with exactly 1 exemplar image per character. Furthermore, all non-principal characters in the chapter, i.e. characters for which we do not have a name and exemplar image, are grouped into a single “other” category. Given a manga chapter and chapter-specific character bank, we report the accuracy of character naming, in Tab. 4.

Naive Baseline. A straightforward solution to the chapter-wide character naming problem is to formulate it as a clustering problem. Assuming that the number of characters in the character bank, k, is equal to the number of ground truth clusters in the chapter (which may not be true in practice), a simple approach is to compute embeddings for each character crop in the chapter and then cluster them into k clusters.

We take two approaches to implement clustering based baselines: (i) simple K-means clustering [28], with k + 1 clusters (an extra cluster for “other” char-

acters, not in the character bank); and (ii) first filter out all “other” characters using outlier/anomaly detection [26] and then perform simple K-means clustering with k clusters. Once the clusters have been computed they are assigned to character names by using Hungarian matching [20] between the embeddings for cluster centres and exemplar images in the character bank.

Discussion. When compared with traditional clustering based solutions, we show that our method performs significantly better. In particular, there are two critical shortcomings with the clustering-based baselines: i) similar looking, but distinct, characters are very likely to be grouped into a single cluster. This further impacts the cluster assignment of other crops, given that the number of clusters is fixed; (ii) such methods do not leverage spatial cues to assist in clustering, e.g. two characters in the same panel are likely to be different characters, regardless of their visual similarity. Our proposed method is more robust to such shortcomings as evident by the superior performance. We also observe that using ground truth must-link and cannot-link constraints for each page, significantly improves the quality of the results. This finding has a very significant implication—in the future, it is sufficient to improve the per-page model in order to improve the chapter-wide results. This is of great value because training a per-page model is much more tractable. In the supplementary, we show how the choice of reference exemplar image in the character bank can further impact the performance. A key limitation of this approach, however, is that it groups all non-principal characters into a single “other” category. It is not designed to disambiguate ‘unnamed person 1’ from ‘unnamed person 2’. We leave this as future work.

- Table 4: Character Naming Results. We report the accuracy results, which have an upper bound of 1.0.

|embedding model|method|notes|PopManga-X (Test-S)|PopManga-X (Test-U)|
|---|---|---|---|---|
|Magi<br><br>|K-means [28]<br><br>|nclusters= k + 1<br><br>|0.3351|0.3820|
|Magiv2|K-means [28]<br><br>|nclusters= k + 1<br><br>|0.3801|0.4223|
|Magi<br><br>|iForest [26] + K-means [28]<br><br>|nclusters= k|0.4549<br><br>|0.4646|
|Magiv2|iForest [26] + K-means [28]<br><br>|nclusters= k|0.5101<br><br>|0.4942|
|Magi<br><br>|Constraint Optimisation (Ours)|Predicted per-page constraints<br><br>|0.6637|0.7058|
|Magiv2<br><br>|Constraint Optimisation (Ours)|Predicted per-page constraints<br><br>|0.7273|0.7530|
|Magi|Constraint Optimisation (Ours)|GT per-page constraints|0.7445|0.7975|
|Magiv2|Constraint Optimisation (Ours)|GT per-page constraints|0.7987|0.8526|

### 8 Conclusion

In this work, we present a solution for generating chapter-wide manga transcriptions with consistent character names and clearer narrative. We contribute a new SOTA model, a training-free constraint optimisation approach to chapterwide character naming, and new datasets to facilitate further research and comparisons. With these contributions it is now possible to transcribe over 10,000 manga chapters that are currently available commercially, complete with character names, allowing for a much richer reading experience for the visually impaired audience.

Acknowledgements: This research is supported by EPSRC Programme Grant VisualAI EP/T028572/1 and a Royal Society Research Professorship RP\R1\191132. This work was partially supported using resources provided by the Cambridge Service for Data Driven Discovery (CSD3) operated by the University of Cambridge Research Computing Service (www.csd3.cam.ac.uk), provided by Dell EMC and Intel using Tier-2 funding from the Engineering and Physical Sciences Research Council (capital grant EP/T022159/1), and DiRAC funding from the Science and Technology Facilities Council (www.dirac.ac.uk). Gyungin Shin would like to thank Zheng Fang for the enormous support.

### References

- 1. Aizawa, K., Fujimoto, A., Otsubo, A., Ogawa, T., Matsui, Y., Tsubota, K., Ikuta, H.: Building a manga dataset “manga109” with annotations for multimedia applications. IEEE MultiMedia 27(2), 8–18 (2020). https://doi.org/10.1109/mmul. 2020.2987895
- 2. Arazo, E., Ortego, D., Albert, P., O’Connor, N., McGuinness, K.: Unsupervised label noise modeling and loss correction. In: International conference on machine learning. pp. 312–321. PMLR (2019)
- 3. Baek, J., Matsui, Y., Aizawa, K.: Coo: Comic onomatopoeia dataset for recognizing arbitrary or truncated texts. In: European Conference on Computer Vision. pp. 267–283. Springer (2022)
- 4. Bain, M., Nagrani, A., Brown, A., Zisserman, A.: Condensed movies: Story based retrieval with contextual embeddings. In: Proceedings of the Asian Conference on Computer Vision (2020)
- 5. Bhattacharjee, D., Everaert, M., Salzmann, M., Süsstrunk, S.: Estimating image depth in the comics domain. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 2070–2079 (2022)
- 6. Chen, Z., Feng, B., Ngo, C.W., Jia, C., Huang, X.: Improving automatic name-face association using celebrity images on the web. In: Proceedings of the 5th ACM on International Conference on Multimedia Retrieval. pp. 623–626 (2015)
- 7. Cordeiro, F.R., Sachdeva, R., Belagiannis, V., Reid, I., Carneiro, G.: Longremix: Robust learning with high confidence samples in a noisy label environment. Pattern recognition 133, 109013 (2023)
- 8. Des livres à voir et à toucher. https://www.lavillebraille.fr/des-livres-avoir-et-a-toucher/
- 9. Fandom. https://www.fandom.com/
- 10. Forrest, J., Lougee-Heimer, R.: Cbc (coin-or branch and cut). https://github. com/coin-or/Cbc, computational Infrastructure for Operations Research (COINOR)
- 11. Guérin, C., Rigaud, C., Mercier, A., Ammar-Boudjelal, F., Bertet, K., Bouju, A., Burie, J.C., Louis, G., Ogier, J.M., Revel, A.: ebdtheque: a representative database of comics. In: 2013 12th International Conference on Document Analysis and Recognition. pp. 1145–1149. IEEE (2013)
- 12. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

- 13. He, Z., Zhou, Y., Wang, Y., Wang, S., Lu, X., Tang, Z., Cai, L.: An end-to-end quadrilateral regression network for comic panel extraction. In: Proceedings of the 26th ACM international conference on Multimedia. pp. 887–895 (2018)
- 14. Hinami, R., Ishiwatari, S., Yasuda, K., Matsui, Y.: Towards fully automated manga translation. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 35, pp. 12998–13008 (2021)
- 15. Hong, X., Sayeed, A., Mehra, K., Demberg, V., Schiele, B.: Visual writing prompts: Character-grounded story generation with curated image sequences. Transactions of the Association for Computational Linguistics 11, 565–581 (2023)
- 16. Huang, Q., Xiong, Y., Rao, A., Wang, J., Lin, D.: Movienet: A holistic dataset for movie understanding. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16. pp. 709–727. Springer (2020)
- 17. Inoue, N., Furuta, R., Yamasaki, T., Aizawa, K.: Cross-domain weakly-supervised object detection through progressive domain adaptation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5001–5009 (2018)
- 18. Jiang, J., Chen, B., Wang, J., Long, M.: Decoupled adaptation for cross-domain object detection. In: International Conference on Learning Representations (2022), https://openreview.net/forum?id=VNqaB1g9393
- 19. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., Krishnan, D.: Supervised contrastive learning. Advances in neural information processing systems 33, 18661–18673 (2020)
- 20. Kuhn, H.W.: The Hungarian Method for the Assignment Problem. Naval Research Logistics Quarterly (1955)
- 21. Lee, Y., Joh, H., Yoo, S., Oh, U.: Accesscomics: an accessible digital comic book reader for people with visual impairments. In: Proceedings of the 18th International Web for All Conference. pp. 1–11 (2021)
- 22. Li, J., Socher, R., Hoi, S.C.: Dividemix: Learning with noisy labels as semisupervised learning. arXiv preprint arXiv:2002.07394 (2020)
- 23. Li, Y., Aizawa, K., Matsui, Y.: Manga109dialog a large-scale dialogue dataset for comics speaker detection. arXiv preprint arXiv:2306.17469 (2023)
- 24. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. pp. 740–755. Springer (2014)
- 25. Liu, D., Keller, F.: Detecting and grounding important characters in visual stories. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 13210–13218 (2023)
- 26. Liu, F.T., Ting, K.M., Zhou, Z.H.: Isolation-based anomaly detection. ACM Transactions on Knowledge Discovery from Data (TKDD) 6(1), 1–39 (2012)
- 27. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499 (2023)
- 28. Lloyd, S.P.: Least squares quantization in pcm. IEEE Transactions on Information Theory 28(2), 129–137 (1982)
- 29. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)
- 30. Lougee-Heimer, R.: The common optimization interface for operations research: Promoting open-source software in the operations research community. IBM Journal of Research and Development 47(1), 57–66 (2003). https://doi.org/10.1147/ rd.471.0057

- 31. Meng, D., Chen, X., Fan, Z., Zeng, G., Li, H., Yuan, Y., Sun, L., Wang, J.: Conditional detr for fast training convergence. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3651–3660 (2021)
- 32. Meyer, P.: Life - a tactical comic for the blind people (2013)
- 33. Musgrave, K., Belongie, S.J., Lim, S.N.: Pytorch metric learning. ArXiv abs/2008.09164 (2020)
- 34. Nagrani, A., Zisserman, A.: From benedict cumberbatch to sherlock holmes: Character identification in tv series without a script. In: BMVC (2017)
- 35. Nguyen, N.V., Rigaud, C., Burie, J.C.: Digital comics image indexing based on deep learning. Journal of Imaging 4(7), 89 (2018)
- 36. Nguyen, N.V., Rigaud, C., Burie, J.C.: Comic mtl: optimized multi-task learning for comic book image analysis. International Journal on Document Analysis and Recognition (IJDAR) 22, 265–284 (2019)
- 37. Ogawa, T., Otsubo, A., Narita, R., Matsui, Y., Yamasaki, T., Aizawa, K.: Object detection for comics using manga109 annotations. arXiv preprint arXiv:1803.08670

(2018)

- 38. Pang, X., Cao, Y., Lau, R.W., Chan, A.B.: A robust panel extraction method for manga. In: Proceedings of the 22nd ACM international conference on Multimedia. pp. 1125–1128 (2014)
- 39. Piriyothinkul, B., Pasupa, K., Sugimoto, M.: Detecting text in manga using stroke width transform. In: 2019 11th International Conference on Knowledge and Smart Technology (KST). pp. 142–147. IEEE (2019)
- 40. Qin, X., Zhou, Y., Li, Y., Wang, S., Wang, Y., Tang, Z.: Progressive deep feature learning for manga character recognition via unlabeled training data. In: Proceedings of the ACM Turing Celebration Conference-China. pp. 1–6 (2019)
- 41. Ramaprasad, R.: Comics for everyone: Generating accessible text descriptions for comic strips. arXiv preprint arXiv:2310.00698 (2023)
- 42. Rayar, F.: Accessible comics for visually impaired people: Challenges and opportunities. In: 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR). vol. 3, pp. 9–14. IEEE (2017)
- 43. Rigaud, C., Le Thanh, N., Burie, J.C., Ogier, J.M., Iwata, M., Imazu, E., Kise, K.: Speech balloon and speaker association for comics and manga understanding. In: 2015 13th International Conference on Document Analysis and Recognition (ICDAR). pp. 351–355. IEEE (2015)
- 44. Sachdeva, R., Cordeiro, F.R., Belagiannis, V., Reid, I., Carneiro, G.: Evidentialmix: Learning with combined open-set and closed-set noisy labels. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 3607–3615

(2021)

- 45. Sachdeva, R., Cordeiro, F.R., Belagiannis, V., Reid, I., Carneiro, G.: Scanmix: learning from severe label noise via semantic clustering and semi-supervised learning. Pattern recognition 134, 109121 (2023)
- 46. Sachdeva, R., Zisserman, A.: The manga whisperer: Automatically generating transcriptions for comics. In: CVPR (2024)
- 47. Samarawickrama, C., Lenadora, D., Ranathunge, R., De Silva, Y., Perera, I., Welivita, K.: Comic based learning for students with visual impairments. International Journal of Disability, Development and Education 70(5), 769–787 (2023)
- 48. Soykan, G., Yuret, D., Sezgin, T.M.: Identity-aware semi-supervised learning for comic character re-identification. arXiv preprint arXiv:2308.09096 (2023)
- 49. Star wars audio comics. https://www.youtube.com/@StarWarsAudioComics/
- 50. Topal, B.B., Yuret, D., Sezgin, T.M.: Domain-adaptive self-supervised pre-training for face & body detection in drawings. arXiv preprint arXiv:2211.10641 (2022)

- 51. Tsubota, K., Ogawa, T., Yamasaki, T., Aizawa, K.: Adaptation of manga face representation for accurate clustering. In: SIGGRAPH Asia 2018 Posters. pp. 1–2

(2018)

- 52. Wang, Y., Zhou, Y., Tang, Z.: Comic frame extraction via line segments combination. In: 2015 13th International Conference on Document Analysis and Recognition (ICDAR). pp. 856–860. IEEE (2015)
- 53. Xu, M., Yuan, X., Shen, J., Yan, S.: Cast2face: Character identification in movie with actor-character correspondence. In: Proceedings of the 18th ACM international conference on Multimedia. pp. 831–834 (2010)
- 54. Zhang, Z., Wang, Z., Hu, W.: Unsupervised manga character re-identification via face-body and spatial-temporal associated clustering. arXiv preprint arXiv:2204.04621 (2022)

## Tails Tell Tales: Chapter-Wide Manga Transcriptions with Character Names — Supplementary Material

##### Ragav Sachdeva, Gyungin Shin, and Andrew Zisserman

Visual Geometry Group, Dept. of Engineering Science, University of Oxford

### 1 More on datasets

In this section we provide more details regarding the two dataset contributionsPopCharacters and PopManga-X.

#### 1.1 PopCharacters

PopCharacters is a character bank dataset comprising principal characters from PopManga dataset, and containing information such as the characters’ names, the manga series each character belongs to, a list of chapters that each character appears in, and a set of exemplar images for each character. In the following we provide details on the data curation process.

Web-scraping. Curating a character bank of principal characters for any arbitrary manga is a very tedious process. The only solution today is to read all chapters of the manga in question, manually keep track of all the characters that have been introduced and store this information in a dataset. This, of course, is a very expensive endeavour. Luckily, a lot of this heavy lifting has already been done by fans of most mangas in the PopManga dataset (see Fig. 1). Therefore, to compile the PopCharacters dataset, we semi-automatically scrape Fandom [9], a website for fans to catalogue details regarding their favourite manga. This results in 11K+ principal characters, across 76 series, with 16K+ thumbnail images, which forms the core the PopCharacters dataset.

Analysis. We make a few observations about the data scraped from Fandom. First, not all 84 manga series in PopManga have Fandom webpages with character information suitable for our purposes. Second, the downloaded thumbnails for characters are often not from the manga but instead from the anime adaptation of the series, and sometimes also from the live adaptation (see Fig. 4). These images have a significant distribution shift in terms of the appearance of the character and are not a good representation of the manga character. Third, out of the 11K+ characters scraped, around half do not have information on which chapters they appear in.

[Figure 191]

[Figure 192]

[Figure 193]

- Fig. 1: Web-scraping from Fandom. For a given series, available on Fandom, we can often scrape the list of chapters, principal characters, character appearances, and thumbnail images for the characters.

Given that the thumbnails scraped from Fandom are often not a good representation of the manga characters, we manually add a few ‘exemplar’ images for each character using crops from manga chapters. However, this is too expensive to do for each of the 11k characters. For instance, in manga series like One Piece, there are more than 1.2K characters that have been catalogued by fans. While each of them might play a significant role in the story, a handful of them appear far more frequently than others (see Fig. 2a). Since the purpose of this dataset is to help name the detected characters during inference, we limit the scope of ‘exemplar mining’ to characters that appear very frequently.

Identifying frequently appearing characters. We identify characters that occur frequently using the list of chapters where a given character appears. In particular, we consider the ‘character appearance frequency’ (which is the proportion of chapters the character appears in, defined as the number of chapters in which a character appears divided by the total number of chapters; e.g., if a certain character appears in Chapters 1, 2, 4, and 8 when there are 16 chapters for the series, its frequency is 0.25) as a quantitative measure. However, there is a notable challenge when using this statistic—as the number of chapters and characters are all different for different series, it is not obvious how to find a good character frequency threshold which well divides characters into the two groups and can be robustly used across the different series. For example, as shown in Fig. 2, two series with high and low number of characters reveal stark contrast in distribution in appearance frequency across different characters.

To solve this challenge, we take the following two-step approach. First, given a series, we classify whether it has a small number of characters (<= 30). If it does, we regard all of the characters as high-frequency characters. Then, for each series with many characters (> 30), we sort all the characters in decreasing

[Figure 194]

[Figure 195]

(a) One Piece (b) Vagabond

- Fig. 2: Proportion of chapters for each character for two series: One Piece (a) and Vagabond (b). The proportion of chapters (y-axis) is defined as the number of chapters in which a character appears divided by the total number of chapters.

order of their appearance frequency and select characters that account for up to 80% of the entire distribution.

[Figure 196]

[Figure 197]

Exemplars

[Figure 198]

[Figure 199]

Query

[Figure 200]

|[Figure 201]|
|---|

|[Figure 202]|
|---|

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|

|[Figure 208]|
|---|

|[Figure 209]|
|---|

|[Figure 210]|
|---|

. . .

. . .

Retrieved from PopManga

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Manual filtering

- Fig. 3: Exemplar mining. Given the thumbnail images scraped from Fandom, we retrieve matching character crops from the PopManga dataset. Manual filtering is then performed to only keep high confidence matches.

Exemplar mining. After identifying the set of high-frequency characters, we use their scraped thumbnails as query images to retrieve high confident matches from the set of all character crops in PopManga. In the interest of diversity of appearance, we randomly select up to 20 retrieved candidate images (instead of considering top-20 most similar matches), which are then filtered manually (see Fig. 3). Our filtering criteria is: (i) remove false positives, (ii) remove low quality images (e.g. with significant occlusion by speech bubbles or other characters). In some cases there were a significant number of false positives, which

[Figure 221]

- Fig. 4: Distribution of PopCharacters thumbnails. Three different versions of thumbnails (i.e., manga, anime, and live-action adaptation) for four characters are shown.

does not reflect poorly on the embedding module but rather on the distribution shift of scraped thumbnails, as noted above in Fig. 4.

Shortcomings. Despite being the first of its kind in the research community, the PopCharacters dataset has a few shortcomings. First, unlike in movies, manga characters are much more likely to undergo radical appearance changes due to aging, magical abilities that involve transformations, or simply changes in style as shown in Fig. 5. In this version of the dataset, although we include images of characters undergoing appearance changes, we do not classify differences between images of the same character, which may limit its applicability. Second, the dataset has not been manually verified in its entirety and may contain some noise, incomplete or even incorrect information as a consequence of web-scraping. Having said that, the subset of the data that is used in evaluation (as character bank in ‘chapter-wide character naming’) has been been through human quality assurance to ensure robust and fair benchmarking.

#### 1.2 PopManga-X

PopManga-X is the extended version of the PopManga dataset [46], wherein the test images now contain annotations for (i) speech-bubble tail bounding boxes, (ii) text-box to corresponding tail-box association, (iii) the name (identity) of each character box, and (iv) sub-classification of text boxes. In the following we provide details on the data annotation process.

Speech bubble tails. In manga, speech bubbles (see Fig. 6) are often used to enclose dialogues, narrations etc. These speech bubbles often, not always, have tails indicating who the speaker is, or even who the speaker is not (in case of “negative tails” or tails pointing away from a character and towards the edge of

[Figure 222]

- Fig. 5: Edge cases of PopCharacters. Examples of characters undergoing nonnegligible changes in their appearance. Typical cases of aging, style changes, and physical transformations are shown on the left, middle, and right, respectively.

the panel). We manually annotate these tail boxes by drawing tight bounding boxes around them. Additionally, we annotate the text-tail associations, which is a many-to-many relationship—for instance, a multi-part speech balloon may only have 1 tail but multiple text boxes, and, a speech bubble may have multiple tails indicating that it is being simultaneously said by multiple characters.

Character Name Annotation. Previously in PopManga, character boxes had a per-page cluster ID indicating which character boxes on the page belong to the same character (i.e., have the same identity). These cluster IDs were not globally unique across the entire chapter or the series. Towards the goal of character name aware transcript generation, we label each character box with a globally unique ID (name). This is done manually by a human by considering the context of the story and using reference images from PopCharacters (where available).

Text Category Annotation. Manga pages have all sorts of texts for the reader to enjoy. However, not all of it is essential to generate a transcript and in fact can actually be a nuisance, if inappropriately included in the transcript. We manually classify the text boxes in PopManga-X to record this information. Specifically, we identified the following 9 initial text categories. Fig. 8 shows a histogram for these text categories. For visual examples of the text categories, see Fig. 7.

– Action/sound word: onomatopoeia or verbs describing action (e.g., “bang!”

or “Slam!”)

[Figure 223]

- Fig. 6: Examples of speech bubbles and their intentions. We note that this list is not complete and also not universal. Manga artists typically have their own unique conventions (e.g., it is common to also have speech bubbles with no tails as ‘normal speech’ and not ‘thinking’). Image taken from animeoutline.com.

- Table 1: Text categories. The terms “bkg info” and “conv. text” refer to background information and conversational text, respectively.

| |text category<br><br>|
|---|---|
|Non-essential Essential|action/sound word, editorial note, scene text, others bkg info, conv. text, interjections (explicit and implicit), internal thought<br><br>|

- – Background information: narration or context
- – Conversational text: conversations between characters
- – Internal thought: texts for internal thoughts of characters
- – Explicit interjection: interjections that are meant to be shown to other characters in the same scene
- – Implicit interjection: interjections that are not supposed to be noticed by other characters or interjections in an internal thought
- – Editorial note: book/chapter names, page number, or meta-level information that is not relevant to the story
- – Scene text: texts that are part of objects such as signs
- – Others.

We then group these 9 categories into two: essential or non-essential for dialogue as shown in Tab. 1. As a result, there are 13k+ and 7k+ texts for dialogues and non-dialogues, respectively.

[Figure 224]

- Fig. 7: Visual examples of the text categories in PopManga-X. For each text box, its text category is shown in red. A/S, BI, CT, EN, IJ (E), and IT denote action/sound word, background information, conversational text, editorial note, interjection (explicit), and internal thought, respectively. Same text box colours within each page indicate the same text category.

[Figure 225]

Fig. 8: Histogram of the text labels.

### 2 More on semi-supervised training

The training datasets used to train the detection and association model is a mixed bag of unlabelled (most), partially labelled (some) and comprehensively labelled (few) images. We utilise Algorithm 1 to train our model in a semisupervised way.

Algorithm 1 Model Training Procedure

- 1: Input: Dataset Dl labeled subset, Du unlabeled subset
- 2: Initialise: Model parameters, θinit
- 3: Phase 0: Mine partial pseudo-labels (no tails etc.)
- 4: for each xi in Du do
- 5: yˆi ← predict(xi) using Magi [46]
- 6: end for
- 7: Phase 1: Warm-up
- 8: for each xi in Du do
- 9: Train model, using partial pseudo-labels yˆi
- 10: end for
- 11: Update model parameters θinit → θinterim
- 12: Phase 2: SSL training
- 13: repeat
- 14: Phase 2a: Train on labelled data
- 15: for each epoch do
- 16: Train model on Dl
- 17: end for
- 18: Update model parameters θinterim → θtuned
- 19: Phase 2b: Mine complete pseudo-labels
- 20: for each xi in Du do
- 21: yˆi ← predict(xi) using model θtuned
- 22: end for
- 23: Phase 2c: Re-train on pseudo labels
- 24: Initialise model with θinit
- 25: for each epoch do
- 26: Train on Du using complete pseudo labels yˆ
- 27: end for
- 28: Update model parameters: θinit → θinterim
- 29: until fixed number of cycles
- 30: Phase 3: Fine-tuning
- 31: for each epoch do
- 32: Train model on Dl
- 33: end for
- 34: Update model parameters θinterim → θtuned

### 3 More on edge prediction evaluation

The detection and association model is designed to output three kinds of edges: (i) character to character, (ii) text to character, and (iii) text to tail. In this section we provide more details regarding how our model’s edge predictions are evaluated and the design decisions.

Character-character edges. The important thing to note about charactercharacter edges is that they are transitive in nature, see Fig. 9. Therefore, the evaluation setting is that of cluster prediction and the metrics used—AMI, NMI,

##### R-P, P@1, MAP@R, MRR—are the ones commonly used in clustering literature. Their implementation is taken from [33]. These metrics better reflect the task at hand than directly measuring the edge prediction quality, as is done below for other two types of edges.

[Figure 226]

[Figure 227]

[Figure 228]

(a) (b) (c)

Fig. 9: Demonstrating transitive nature of character-character edges. Even though there are significantly more number of edges in (a) than (b), they both have the same prediction for character clusters (connected components). On the contrary, (b) and (c) only differ by 1 edge (marked in dashed), yet it completely changes the predicted character clusters.

Text-tail edges. The text-tail edges represent the relationship between text boxes and tail boxes, i.e., whether a given tail corresponds to a given text box. This can be a many-to-many relationship, i.e., 1 or more text boxes can have a 0 or more tails. For instance, a multi-part speech bubble has 2+ text boxes which may only have a single tail box, or a single text box may have many tails indicating the case where multiple speakers simultaneously say something. To evaluate all these cases in a unified fashion, we treat it as a binary classification problem, and compute the average precision metric, as shown in Fig. 10.

Tail 1 Tail 2 Tail 3 Tail 4

Tail 1 Tail 2 Tail 3 Tail 4

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

- Text 1
- Text 2
- Text 3
- Text 4
- Text 5

- Text 1
- Text 2
- Text 3
- Text 4
- Text 5

Page1Page2

flatten and concatenate

compute average precision

0.9

Tail 1 Tail 2 Tail 3

Tail 1 Tail 2 Tail 3

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

- Text 1
- Text 2
- Text 3

- Text 1
- Text 2
- Text 3

Ground truth across all pages

Predicted scores across all pages

Predicted association scores Ground truth associations

Fig. 10: Text-tail (and text-character edge) evaluation process.

Text-character edges. The text-character edges embed the speaker diarisation information, i.e., which text box is said by which character. The important thing here is to note the distinction between (i) text box to character box association, and (ii) text box to character identity association. While the model outputs the former, we actually care about the latter in terms of generating the transcript. In other words, ‘text box 1 is associated to character box 1’ is no different than ‘text box 1 is associated to character box 2’, if character box 1 and 2 belong to the same character. This text to character identity evaluation setting (instead of text to character box) is even more important when the speaker of a text box is not present in the same panel as the text, but is present in preceding and subsequent panels. In such cases, it is almost arbitrary as to which particular character box is the right association; however, there is only a single correct character identity that must be associated with this text.

To evaluate the text-character identity predictions, we consider the text box to character box predictions by the model, and max-pool the scores for character boxes that have the same identity. Afterwards, the evaluation process is the same as for text-tail edges, as denoted in Fig. 10.

### 4 More on character naming evaluation

As mentioned in the paper, the purpose of this evaluation is to measure the efficacy of the method in forming chapter-wide character clusters. When evaluating a specific chapter, we sample exactly 1 exemplar image (from PopCharacters dataset) for each character that appears in this chapter. This exemplar is used as a reference image when assigning/matching crops to this character, and it was arbitrarily chosen by a human beforehand for each character in the test set and fixed for the purposes of evaluation.

The advantage of using a single exemplar image per character is that it keeps the evaluation fair as some characters have more exemplars than others. However, this introduces another variable – “the choice of exemplar image” which can significantly impact the performance results (some exemplars are better representations of the character than others). To demonstrate just how significant the gap can be, we report the character naming results in Tab. 2 when “optimal exemplars” are used. Specifically, instead of arbitrarily choosing and fixing an exemplar image from PopCharacters dataset, for a given character, we consider all possible crops of this character in the chapter being evaluated (using ground truth information) and use the crop which has the highest average similarity to all the other crops of this character, as the “optimal exemplar”. For the sake of evaluation, this has a few benefits – (i) it eliminates the “choice of exemplar” variable from the evaluation process, (ii) sampling the exemplar from within the chapter increases the likelihood of the exemplar being visually representative of the character, thus reducing the effect of edge case noted in Fig. 5, and (iii) it makes the evaluation setting agnostic to the external character bank which is desirable for future benchmarking and comparison.

As evident from Tab. 2, the character naming results are significantly better when “optimal exemplars” are used. Of course during inference we would never have such knowledge about optimal exemplars, and therefore these results are difficult to achieve in practice. An ideal case scenario during inference is that several diverse exemplars are available for each character and their average embedding is used as the representation of the character. To keep things simple, we have not deeply investigated this.

- Table 2: Character Naming Results. We report the accuracy results, which have an upper bound of 1.0.

|embedding model|method|notes|exemplars|PopManga-X (Test-S)|PopManga-X (Test-U)|
|---|---|---|---|---|---|
|Magi|K-means [28]|nclusters= k + 1<br><br>|random, fixed<br><br>|0.3800<br><br>|0.3993|
|Magiv2<br><br>|K-means [28]|nclusters= k + 1<br><br>|random, fixed|0.4126<br><br>|0.4221|
|Magi<br><br>|iForest [26] + K-means [28]<br><br>|nclusters= k<br><br>|random, fixed|0.4710<br><br>|0.4859|
|Magiv2<br><br>|iForest [26] + K-means [28]|nclusters= k<br><br>|random, fixed|0.5298|0.5096|
|Magi<br><br>|Constraint Optimisation (Ours)<br><br>|Predicted per-page constraints|random, fixed|0.6637<br><br>|0.7058|
|Magiv2<br><br>|Constraint Optimisation (Ours)<br><br>|Predicted per-page constraints<br><br>|random, fixed|0.7273<br><br>|0.7530|
|Magi<br><br>|Constraint Optimisation (Ours)<br><br>|Predicted per-page constraints|optimal|0.8164<br><br>|0.8375|
|Magiv2<br><br>|Constraint Optimisation (Ours)|Predicted per-page constraints|optimal<br><br>|0.8735<br><br>|0.8770|
|Magi|Constraint Optimisation (Ours)|GT per-page constraints|random, fixed|0.7445|0.7975|
|Magiv2|Constraint Optimisation (Ours)|GT per-page constraints|random, fixed|0.7987|0.8786|
|Magi|Constraint Optimisation (Ours)|GT per-page constraints|optimal|0.8579|0.8786|
|Magiv2|Constraint Optimisation (Ours)|GT per-page constraints|optimal|0.9219|0.9302|

### 5 More on transcript generation

After detection (characters, texts, panels and tails) and association (charactercharacter, text-character, text-tail), and chapter-wide character naming, generating the transcript is relatively straightforward. As mentioned in the paper, this is a four-step process: (i) filtering non-essential texts, (ii) text ordering, (iii) OCR, and (iv) generating the transcript using the predicted text-character associations and character names. The implementation for the ordering algorithm

and the OCR model have been directly taken from [46] as the purpose of this work is not to improve on these.

Beyond that we highlight a few design decisions that can be made while generating the transcripts to make them more robust to model’s mistakes and ensure narrative consistency. First, low-confidence speaker predictions for essential texts can be rendered as ‘<unsure>’ in the transcript, rather than confusing the reader. Second, given that we have detected tail boxes, and matched them to their corresponding text boxes, with our method it is possible to indicate the speakers in the transcript only for the texts that have tails. In other words, for texts without tails, it might be reasonable to just include them in the series of dialogues without indicating the speaker and let the reader infer the speakers from the context. This has two benefits: (i) it is in-line with the manga artists’ intention (the fact that they chose to not draw an explicit tail for some texts), and (ii) texts without tails are usually where the model makes more mistakes.

On using LLMs to enhance the transcripts. In this work we also investigated using LLMs to enhance the quality of the generated transcripts. While the LLMs do a very good job at fixing OCR mistakes and can be employed successfully for that purpose as a post-processing step, we were mainly interested in exploring if LLMs can leverage conversational history and context to fix speaker prediction mistakes. We discovered that text-based speaker diarisation is a very challenging problem where the ambiguity in predicting who the speaker is increases drastically as the number of speakers increases. Often in two-person conversations, it is possible to deduce a change in speaker based on the conversation pattern; however, with three or more potential speakers, many of whom can possibly be ‘other’, the problem becomes very challenging, and we did not have much success with using LLMs. We also investigated training a vision-language model that leverages both vision and language cues for this task, but had limited success which we attribute to two reasons: (i) lack of large-scale ground truth annotations (our training was largely done on pseudo-annotations mined for large-scale data which is quite noisy), and (ii) the inherent imbalance in the data (most texts in fact can be attributed to the correct speaker simply by picking the nearest one, therefore the signal for language during training is rather weak).

