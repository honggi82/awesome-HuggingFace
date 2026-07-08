## Relational Visual Similarity

# arXiv:2512.07833v2[cs.CV]9Apr2026

Thao Nguyen1, Sicheng Mo2, Krishna Kumar Singh3, Yilin Wang3, Jing Shi3, Nicholas Kolkin3 Eli Shechtman3, Yong Jae Lee1,3,†, Yuheng Li3,† 1University of Wisconsin-Madison, 2University of California, Los Angeles, 3Adobe Research

https://thaoshibe.github.io/relsim

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

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

[Figure 33]

[Figure 34]

[Figure 35]

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

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Group A Reference Group B

Figure 1. Would you say images in Group A are similar to the Reference Image? Current state-of-the-art image similarity models (e.g., LPIPS [1], CLIP [2]) would answer no. These models would say only Group B are similar to the reference image, as they equate similarity with a high degree of shared perceptual attribute features (i.e., color, shape, semantic class). However, as humans, we would confidently say yes—images in both groups are similar to the reference. While Group B is similar in perceptual attributes, Group A is similar in a more abstract, relational sense (e.g., “transformation of {subject} through time”, first row). In this paper, we propose to model this missing dimension of visual similarity, or called relational visual similarity, capturing human-like reasoning over relational structures.

### Abstract

Humans do not just see attribute similarity—we also see relational similarity. An apple is like a peach because both are reddish fruit, but the Earth is also like a peach: its crust, mantle, and core correspond to the peach’s skin, flesh, and pit. This ability to perceive and recognize relational similarity, is arguable by cognitive scientist to be what distinguishes humans from other species. Yet, all widely used visual similarity metrics today (e.g., LPIPS, CLIP, DINO) focus solely on perceptual attribute similarity and fail to capture the rich, often surprising relational similarities that humans perceive. How can we go beyond the visible content of an image to capture its relational properties? How

† denotes equal advising

can we bring images with the same relational logic closer together in representation space? To answer these questions, we first formulate relational image similarity as a measurable problem: two images are relationally similar when their internal relations or functions among visual elements correspond, even if their visual attributes differ. We then curate 114k image–caption dataset in which the captions are anonymized—describing the underlying relational logic of the scene rather than its surface content. Using this dataset, we finetune a Vision–Language model to measure the relational similarity between images. This model serves as the first step toward connecting images by their underlying relational structure rather than their visible appearance. Our study shows that while relational similarity has a lot of realworld applications, existing image similarity models fail to capture it—revealing a critical gap in visual computing.

### 1. Introduction

The ability to perceive and recognize visual similarity is arguably the most fundamental sense for any visual creature, including humans, to interact and make sense of the world [3, 4]. We process visual attributes to guide decisions: recognizing that a peach is red might signal that it is edible. We also notice similarities across different objects (e.g., shape, color, texture) to categorize, remember, and abstract them: an apple and a peach are both red and round, so they are likely both fruits. Beyond this, we can see relational similarity as well: we abstract familiar patterns to understand more complex or unseen phenomena. For example, we can anticipate the Earth is like a peach, as its layers—crust, mantle, and core—roughly correspond to the peach’s skin, flesh, and pit, even though no one has directly observed it. In cognitive science, attribute similarity and relational similarity are often considered the two central pillars when it comes to understanding human perception of similarity [5, 6]. Attribute similarity underlies everyday activities (e.g., recognition [7], classification [8], memorization [9]), while relational similarity fuels reasoning and creativity (e.g., analogies [10], abstract thought [11]). Some researchers argue that relational similarity is even more central to human cognition, as it drives analogical learning and creativity—the traits that set humans apart from other intelligent species [12–14].

Unfortunately, current state-of-the-art visual similarity frameworks focus almost exclusively on attribute-level similarity. Traditionally, image similarity in computer vision has been framed as the task of comparing two images and deciding whether they are visually similar, typically at the pixel or feature level using handcrafted descriptors [15, 16]. In recent years, large-scale hierarchical datasets (e.g., ImageNet [17]) and cross-modal datasets (e.g., LAION-2B [18]) have enabled deep learning models to move beyond lowlevel visual details. Modern approaches (e.g., [2, 19–23]) can recognize different images of the same semantic class or images that match a rough textual description—for example, “a photo of matchsticks”—even if they differ in shape, color, or other low- to mid-level details (Fig. 1, Group B, first row).

However, by focusing primarily on surface-level features, these models struggle to capture relational similarity (see [24, 25], Sec. 4.2). For instance, they cannot easily recognize that the burning stages of a match resemble the ripening stages of a banana (Fig. 1, Group A, first row). Capturing this type of similarity requires a shift in perspective: instead of relying solely on visual features, we must reason about how different visual elements, interact, abstracting the underlying relationships. For example, both the match and the banana undergo a gradual transformation over time. The similarity lies not in their specific appearance but in the logic of change. This raises questions: which attributes should be preserved or ignored during comparison? How can we identify which relational patterns are relevant or useful?

Dataset Data Format Example BAPPS

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

image triplet (low-level perceptual)

LPIPS

[1]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

NIGHTS [22]

image triplet (mid-level perceptual)

Dream

Sim

ImageNet

- [17]

semantic class (attribute-based)

Image Net

“Bernese mountain dog”

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

LAION-2B

- [18]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

“Cats of Torcello, Italy”

“An annoyed cat”

{image, caption} (attribute-based)

LAION

[Figure 78]

[Figure 79]

{image, anonymous caption} (relational-based)

[Figure 80]

[Figure 81]

Ours “Aout{object}against graystands

Ours (relsim)

background”

Table 1. Survey of prominent datasets used for training visual similarity metrics. All are organized based on attribute similarity, whereas ours focuses on relational similarity.

Insights from cognitive science, encouragingly, offer a spark for these questions. Works [10, 26] showed that humans process attribute similarity perceptually, but relational similarity requires conceptual abstraction, often supported by language or prior knowledge. This suggests that recognizing relational similarity first requires understanding the image, drawing on knowledge, and abstracting its underlying structure. Take the example of a photo of burning matches: we first observe how each match relates to the others—they burn sequentially from left to right. With prior knowledge, we understand that burning is a temporal transformation, a process that can occur in many other objects (e.g., a leaf aging, a banana ripening). If asked to write a caption capturing this logic rather than the specific objects, one might write “transformation of {subject} over time”. We call such captions anonymous captions—they do not describe any particular visible object but instead capture the relational logic conveyed by the image. These captions act as the glue connecting images with similar underlying logic. In other words, a successful relational visual similarity model must understand, abstract, and use anonymous captions to bring logically similar images together.

To model relational similarity, we follow a path inspired by insights from cognitive science. Since no existing dataset captures relational visual similarity (see Tab. 1), we first filter a large image corpus, LAION-2B [18], to extract 114k images likely to contain transferable relational structures. This step improves dataset quality by removing low-quality, mislabeled, or relationally uninformative images, which are common in LAION-2B [27, 28]. We then train an anonymous captioning model to generate captions for these images, creating a set of {image, anonymous caption} pairs. Finally, we train a relational visual similarity model, relsim, on this dataset, optimizing it to bring together images whose captions encode similar relational abstractions. We demonstrate the utility of relsim for tasks such as relational image retrieval and analogical image generation.

In short, our contributions are as follows:

- • A new notion of image similarity, relational visual similarity, which complements traditional attribute similarity.
- • A novel relational dataset, consisting of 114k {imageanonymous captions} designed to capture the abstraction and logic in each image.
- • A new tuned metric, relsim, that captures the relational visual similarity between two images.
- • Analysis of the relationship between relational and attribute similarity, along with experiments demonstrating the limitations of current image similarity models.
- • Demonstration of downstream applications in image retrieval and image generation.

### 2. Related Works

Similarity in Cognitive Science. The question of what makes two subjects similar has always been considered one of the most significant questions in cognitive science [3, 9, 29–31]. Similarity is fundamental to human cognition, as it affects how the mind organizes, categorizes, and reasons about the world. For decades, Tversky’s theory of similarity [9], also called the contrast model, has been widely adopted and has inspired multiple domains [1, 22, 32]. Tversky frames similarity as a psychological comparison of matching individual properties or characteristics of objects (e.g., size, shape, color). For example, an apple and a banana are similar because they are both fruits. While powerful, Tversky’s theory cannot account for similarities such as the one Stephen Hawking made when he said, “I regard the brain as a computer” [33]. There are no obvious visual features shared between a human brain and a computer. This kind of similarity, which cannot be fully accounted for by Tversky’s model, was later formalized as relational similarity, alongside its counterpart, now called attribute similarity. These concepts emerged from Gentner’s research on analogy, often referred as Structure-Mapping theory [10]. Relational similarity is a comparison based on the relationships between objects. Returning to the previous example, Stephen Hawking was making a relational comparison: he viewed the brain as a biological machine and the process of death as analogous to a computer breaking down. Substantial research shows that while both type of similarity are important, relational similarity (often associated with analogical reasoning) plays a distinct and often deeper role in human cognition (i.e., analogical learning and reasoning [3, 12–14]).

Image Similarity. Comparing similarity between two visual signals is a core concept in computer vision, as it underpins many tasks (e.g., object recognition, image retrieval, image matching). Before the deep learning era, most image similarities were computed directly via pixellevel metrics (e.g., L1, L2, MSE, RMSE, PSNR) or handcrafted features (e.g., SSIM [34], FSIM [35], SIFT [15]). With the rise of deep learning and neural networks (e.g.,

VGG [21], ResNet [23]), deep-feature-based image similarity metrics better align with human perceptual judgment (e.g., LPIPS [1], PieAPP [36], DISTS [37]). More recently, with the aid of Vision Transformers (ViT) [38] and SelfSupervised Learning (SSL), modern vision encoders (e.g., DINO [20], CLIP [2], dreamsim [22], SigLIP [39]) not only provide robust visual embeddings for image similarity, but also enable semantic comparisons that go beyond pixel-level matching. However, all of these approaches rely on the assumption that image similarity is based solely on attribute similarity, and thus cannot capture relational similarity, as we demonstrate in our experiments (Sec. 4.2). Here, we, for the first time, propose to consider relational visual similarity.

Mutimodal Large Language Models. Research on multimodal models (e.g., [40–48]) has become an increasingly attractive topic in recent years. In particular, progress in developing unified models that can both understand and generate visual and textual inputs/outputs has transformed how we interpret and interact with visual information. While traditional vision encoders (e.g., CLIP [2]) can mostly only “see” what is explicitly shown in an image (e.g., “a photo of a mother hugging a child”), integrating them with MLLMs allows us to capture what is not directly depicted (e.g., “the image representing a sense of parental care”). Since relational similarity often requires a deeper understanding of images that goes beyond mere perception, we choose to leverage MLLMs, particularly Vision Language Models (VLMs), as the backbone for image feature extraction.

### 3. Relational Visual Similarity

We formalize the problem of measuring the relational visual similarity as follows. Given two input images I1 and I2, we aim to train a visual feature extractor fV such that the resulting features capture the relational similarity between the two images. Our core assumption is that if two images exhibit high relational similarity, then their corresponding anonymous captions, A1 and A2, should also be similar. Specifically, we define the relational similarity score s12 between the two images as:

s12 = fV (I1) · fV (I2) ≈ fT(A1) · fT(A2),

where “·” denotes the cosine similarity between the feature embeddings. Here, fT represents a textual encoder that produces embeddings for the corresponding captions.

In Sec. 3.1, we describe how to construct the relational dataset, including how to sample image {Ii}Ni=1 and generate their corresponding anonymous captions {Ai}Ni=1. Then, in Sec. 3.2, we detail the training procedure for fV .

#### 3.1. Creating a Relational Dataset

Filtering interesting images {Ii}Ni=1. Not all images are equally informative with deep logic for learning relational

Group 1

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

LAION-2B

[Figure 87]

[Figure 88]

creatively

[Figure 89]

|[Figure 90]<br><br>[Figure 91]<br><br>Anonymous<br><br>Caption 1|
|---|

creatively uses {Food} to resemble a {Object}

[Figure 92]

[Figure 93]

[Figure 94]

creatively

uses {Food}

[Figure 95]

[Figure 96]

[Figure 97]

|𝑇1|
|---|

|𝑇2|
|---|

|𝑇3|
|---|

|𝑇4|
|---|

uses {Food}

to resemble a

Text Embedding

|| |
|---|
<br><br>[Figure 98]<br><br>Anonymous Caption|
|---|

[Figure 99]

… …

to resemble a

{Object}

Manually Verify

| | |
|---|---|
|[Figure 100]<br><br>Ima<br><br>Filt|ge<br><br>er|
| | |

[Figure 101]

|[Figure 102]<br><br>Anonymous<br><br>Caption|
|---|

{Object}

Group M

|𝐼1|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 103]

|[Figure 104]<br><br>[Figure 105]<br><br>Anonymous Caption M|
|---|

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

|𝐼2|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 113]

|relsim|
|---|

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Training Example

re

[Figure 119]

|𝐼3|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 120]<br><br>[Figure 121]<br><br>Transform a {Fruit} into a {Animal} by carving.|
|---|

[Figure 122]

[Figure 123]

|[Figure 124]<br><br>[Figure 125]|
|---|

|[Figure 126]<br><br>[Figure 127]|
|---|

|[Figure 128]<br><br>[Figure 129]|
|---|

[Figure 130]

[Figure 131]

VLM

|𝐼4|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Instruction

(a) Image Filtering

(b) Anonymous Captioning (c) Relational Visual Similarity (relsim) Training

- Figure 2. Overall pipeline. (a) We train an image filtering model to select high-quality relational images from LAION-2B [18]. (b) Anonymous captioning model is trained on groups of images that share the same underlying logic, pairing all images in each group with the same anonymous caption. (c) Training relational visual similarity (relsim) model involves a contrastive loss between image features and their corresponding anonymous captions.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

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

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Random LAION Increasing “interestingness”

[Figure 168]

[Figure 169]

- Figure 3. Examples of relationally interesting vs. ordinary images.

A dynamic sequence of Monarch butterflies…

Single Image

|[Figure 170]<br><br>[Figure 171]|
|---|

Series of images illustrating the {Movement

Type} of a group of {Insect Name}…

…

A dynamic sequence of nearly identical {Figure} traces an unfolding curvilinear path…

Image Group

|[Figure 172]<br><br>[Figure 173]|
|---|

|[Figure 174]<br><br>[Figure 175]|
|---|

A visual representation of {Subject} performing

|[Figure 176]|[Figure 177]<br><br>|
|---|---|
| | |

{Type of Motion} in progressive stages, arranged horizontally to demonstrate the flow of

movement…

|[Figure 178]<br><br>[Figure 179]|
|---|

Figure 4. Writing an anonymous caption is hard from a single image, but easier with an image group where the pattern is clear.

structures. For instance, an image of a single sofa merely conveys surface-level object appearance, offering limited deep cues about relational organization. In contrast, a photo of “strawberry heart” expresses creatively compositional relations that can be abstracted and transferred to new visual content (e.g., “walnut brain”, Fig. 3, second row).

Motivated by this observation, we manually curate M = 532 groups of images, where all images within a group exhibit the same underlying relational logic or pattern. Each group has Ng images (a minimum of 2 and a maximum of 10 images). We present each full group to an frozen VLM and prompt it to produce a single anonymous caption Ag—a relational description that avoids object-specific terms by replacing them with placeholders (e.g., {subject}). This caption is then human-verified and paired with every image in the group, yielding an anonymous training dataset (Fig. 2b):

Given the vast nature of LAION-2B, we first perform a filtering step to identify images potentially containing higherorder relational cues (which we refer to as interesting images). We fine-tune Qwen2.5-VL-7B-Instruct [41] to classify whether an image is relationally interesting, using 1.3k positive and 11k negative human-labeled examples (Fig. 2a). Annotators were instructed: “Can you see any relational pattern, logic, or structure in this image that could be useful for creating or linking to another image?”. The fine-tuned model achieves 93% agreement with human judgments, and when applied to LAION-2B, it yields N = 114k images identified as relationally interesting. Details of the prompt and model configuration are provided in the Supp.

{(Iig,Ag) | i = 1,...,Ng}Mg=1 This procedure encourages the model to assign similar anonymous captions to images expressing the same relational pattern. We use Qwen2.5-VL-7B-Instruct [41] to train this captioning model. After training, we apply it to all “interesting” images identified in the previous step, yielding a dataset consisting of images annotated with anonymous relational captions, {Ii,Ai}Ni=1, where N = 114,881 to be exact.

Generating anonymous captions {Ai}Ni=1. Writing a shared relational attribute from a single image is inherently challenging. For example, given only a sequence depicting a butterfly’s flight stages (Fig. 4, first row), it is unclear which visual details are irrelevant and which constitute the underlying relational pattern. In contrast, when this image is shown alongside others expressing the same logic (Fig. 4, second row), the shared relational structure becomes immediately apparent, making it easy to articulate a caption that abstracts away object specifics.

#### 3.2. Modeling Relational Visual Similarity

Objective. Given the collection of relationally interesting images with their corresponding anonymous captions {(Ii,Ai)}Ni=1, we train a visual extractor fV with a frozen text encoder fT to produce normalized embeddings:

fV (Ii) ∥fV (Ii)∥

fT(Ai) ∥fT(Ai)∥

vi =

, ti =

.

We compute the similarity between an image and its anonymous caption using a dot product scaled by a learnable temperature parameter τ > 0:

vi⊤tj τ

sij =

.

For a batch of size B, we use the InfoNCE training loss [2]:

L = B1 Bi=1 − log exp(s

ii)

B j=1 exp(sij)

This training paradigm encourages the visual extractor to capture relationally meaningful features that align with the abstract concepts represented in the anonymous captions.

Model Selection. Traditional visual similarity methods rely on pure vision encoders (e.g., [2, 20, 22]), which derive representations solely from attribute-level features. We find these vision-only encoders insufficient for capturing relational similarity, even after tuned, as relational reasoning goes beyond mere visual recognition (See 4.2).

To address this, we leverage Vision Language Models (VLMs) for two reasons: (1) vision encoders emphasize visual attributes or semantics, which can conflict with relational understanding; and (2) relational reasoning often requires higher-level semantic knowledge—which can be found nowhere better than in a Large Language Model, where it was already trained with world knowledge. Accordingly, we employ a VLM as our visual extractor fV (Fig. 2c). Optionally, the task–instruction can be paired with the image as a fixed, steering prompt (e.g., “Carefully analyze image to understand its underlying logic...”).

### 4. Experiments

We now discuss our experimental settings, baselines, and evaluation protocol, followed by additional analyses.

#### 4.1. Settings

Implementation. We adopt Qwen2.5-VL-7B-Instruct [41] as our visual feature extractor fV . Specifically, we append a learnable query token to the end of the image as instruction token, and feed them together into the LLM. We use the query token’s feature from the LLM’s last layer as our visual relational feature. For the text embedding model fT, we use all-MiniLM-L6-v2, a widely used and efficient pre-trained model from the Sentence-Transformers library [49]. We train Qwen2.5-VL-7B-Instruct with LoRA [50] for 15k iterations on a single node with 8×A100 GPUs and a batch size of 64.

Data. To ensure complete separation between training and evaluation, we randomly split the dataset of 114k images into 100k for training and 14k for evaluation. For evaluation, we consider the image retrieval setting. Specifically, given a query image, we retrieve the most similar image from the database (excluding the query itself); ideally, the

retrieved image should be relationally similar to the query. The database consists of the 14k images from the test set, combined with another 14k new images randomly sampled from LAION-2B [18] to better approximate a real-world database. From this database, 1000 images are randomly chosen from 14k test set to serve as query images.

Evaluation protocol. We employ GPT-4o [42] as an automated judge to evaluate retrieval results. For each query image and retrieved image pair, GPT-4o is prompted to assign a relational similarity score on a scale from 0 to 10, where 10 indicates highly relationally similar and 0 indicates no similarity (See Supp. for full prompt). Along with this automatic evaluation, we conduct a user study to capture human preferences. Participants are shown a query image along with two retrieved images: one from ours and one from a baseline method (randomly named as A or B)—and are asked to select which retrieved image is relationally more similar to the query (A, B, or Same). For each baseline, we randomly constructed 300 triplets, and each triplet was independently evaluated by at least three users, resulting in approximately 900 responses per baseline. This study allows us to quantify the proportion of cases in which users prefer our retrieval results over the baselines.

Baselines. We compare our approach with prominent image similarity metrics, including LPIPS [1], DINO [20], dreamsim [22], and CLIP-I [2] (image-to-image). These models can directly output similarity scores for a pair of images. We also consider baselines that operate via captions. In these settings, we first prompt Qwen [41] to generate an anonymous or abstract caption for each image, and then perform retrieval using this caption as the query feature. We evaluate two variants: (1) Apply CLIP-based text-to-image retrieval denoted as CLIP-T; and (2) Text-to-text retrieval denoted as Qwen-T. Note that in both of these caption-based baselines, we use the original Qwen model rather than our finetuned version. This allows us to show the performance of prompting a VLM to produce the anonymous caption from a single image (see Fig. 4) whereas finetuned model is our method which benefits from a group of images.

#### 4.2. Evaluations

Can existing metrics capture relational similarity? Results are presented in Fig. 6, where higher values indicate better performance. As shown, LPIPS [1], which focuses purely on perceptual similarity, achieves the lowest score (4.56). DINO [20] performs only slightly better (5.14), likely because it is trained solely in a self-supervised manner on image data. CLIP-I [2] yields the strongest results among the baselines (5.91), presumably because some abstraction is sometimes present in image captions. However, CLIP-I still underperforms relative to our method, as achieving a better score may require the ability to reach even higherlevel abstractions, such as those in anonymous captions. Our

A cat's head with human-like features and a pearl earring, blending animal and human elements in a surreal, artistic composition. A woman stands on a circular platform, framed by a giant, menacing gorilla's hand and face, creating a surreal, scale-defying illusion.

A white form emerges from a narrow gap in weathered wooden planks, creating a stark contrast between light and shadow.

A giraffe stands atop an ironing board, its neck reaching toward a distant light source, creating a surreal juxtaposition of nature and domesticity.

LPIPS dreamsim DINO CLIP-I Qwen-T

Ours

LPIPS dreamsim DINO CLIP-I Qwen-T Ours

|[Figure 180]<br><br>[Figure 181]|
|---|

|[Figure 182]<br><br>[Figure 183]|
|---|

|[Figure 184]<br><br>[Figure 185]|
|---|

|[Figure 186]<br><br>[Figure 187]|
|---|

|[Figure 188]<br><br>[Figure 189]|
|---|

|[Figure 190]<br><br>[Figure 191]|
|---|

|[Figure 192]<br><br>[Figure 193]|
|---|

|[Figure 194]<br><br>[Figure 195]|
|---|

|[Figure 196]<br><br>[Figure 197]|
|---|

|[Figure 198]<br><br>[Figure 199]|
|---|

|[Figure 200]<br><br>[Figure 201]|
|---|

|[Figure 202]<br><br>[Figure 203]|
|---|

NearestNeighborsQuery

NearestNeighborsNearestNeighborsQueryQuery

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

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

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

LPIPS dreamsim DINO CLIP-I Qwen-T Ours LPIPS dreamsim DINO CLIP-I Qwen-T Ours

|[Figure 276]<br><br>[Figure 277]|
|---|

|[Figure 278]<br><br>[Figure 279]|
|---|

|[Figure 280]<br><br>[Figure 281]|
|---|

|[Figure 282]<br><br>[Figure 283]|
|---|

|[Figure 284]<br><br>[Figure 285]|
|---|

|[Figure 286]<br><br>[Figure 287]|
|---|

|[Figure 288]<br><br>[Figure 289]|
|---|

|[Figure 290]<br><br>[Figure 291]|
|---|

|[Figure 292]<br><br>[Figure 293]|
|---|

|[Figure 294]<br><br>[Figure 295]|
|---|

|[Figure 296]<br><br>[Figure 297]|
|---|

|[Figure 298]<br><br>[Figure 299]|
|---|

Query

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

NearestNeighbors

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

- Figure 5. Attributes vs. Relational Visual Image Retrieval. Visualization of nearest neighbor using different visual similarity metrics. As can be seen, only ours understands and can detect the relational similarity.

LPIPSDINOdreamsimCLIP-ICLIP-TQwen-TTunedDINOTunedCLIP Ours

- 5

- 6

- 7

GPTScore

4.56

5.14

5.76 5.91

5.33

4.86

5.62

6.02

6.77

- Figure 6. Relational visual similarity performance. All existing image similarity metrics fail to capture relational similarity, even after being tuned. Our final model (relsim) which leverages knowledge from VLMs, achieves the highest score (6.77).

model is hard to prompt and often leaks semantic or attribute information, causing retrieval to overly focus on semantics rather than relational similarity, thus yielding poor results (i.e., 5.33 and 4.86, compared with ours, 6.77).

Knowledge is essential for capturing relational similarity. Our argument is that relational similarity requires more than visual perception—it demands a deeper form of image understanding. Such knowledge is largely absent in visionencoder-only models. To test this hypothesis, we conduct an ablation study in which we finetune pure vision encoders (CLIP [2] and DINO [20]) using the same anonymous captions training data and the same loss. The results (denoted as Tuned CLIP/DINO), shown in the right panel of Fig. 6, indicate that finetuning with anonymous captions does improve these models’ ability to capture structural relationships. However, their performance still falls short of our model, which is equipped with a VLM. This gap is likely because VLMs, which integrate visual features with language-based world knowledge, are inherently necessary to understand and encode relational similarity.

vision encoder, being equipped with LLM knowledge and anonymous captions, yields the highest score (6.77).

Why generate anonymous captions from a group? As described in the approach section, our anonymous captions are generated from manually selected groups of similar images. Using a group makes it easier to identify the shared relational structure required for a high-quality anonymous caption. The CLIP-T and Qwen-T baselines further illustrate this point (Fig. 6): in both cases, anonymous captions are produced from a single image using the original Qwen2.5VL-7B-Instruct [41]. We find that, under this setting, the

Do humans agree with ours? The result of our user study, shown in Fig. 8, indicates that users consistently prefer

|[Figure 372]|
|---|

|[Figure 373]|
|---|

||[Figure 374]<br><br>[Figure 375]|
|---|
<br><br>|[Figure 376]<br><br>[Figure 377]|
|---|
<br><br>|[Figure 378]<br><br>[Figure 379]|
|---|
<br><br>|[Figure 380]<br><br>[Figure 381]|
|---|
<br><br>|[Figure 382]<br><br>[Figure 383]|
|---|
<br><br>|[Figure 384]|
|---|
|
|---|

||[Figure 385]<br><br>[Figure 386]|
|---|
<br><br>|[Figure 387]<br><br>[Figure 388]|
|---|
<br><br>|[Figure 389]<br><br>[Figure 390]|
|---|
|
|---|

[Figure 391]

relationalsimilarity(relsim)

|same|
|---|

[Figure 392]

|same logic, look different|
|---|

nt

logic,

||[Figure 393]<br><br>[Figure 394]|
|---|
<br><br>|[Figure 395]<br><br>[Figure 396]|
|---|
<br><br>|[Figure 397]<br><br>[Figure 398]|
|---|
|
|---|

same appearance

Query

[Figure 399]

[Figure 400]

|random images|
|---|

attribute similarity (CLIP)

[Figure 401]

Figure 7. Similarity space showing different kinds of visual similarity in terms of degree of relational vs. attribute similarity.

our method across all baseline comparisons, with preference rates ranging from 42.5-60.7%. The gray bars indicate the tie rate. This is highly encouraging, as it demonstrates not only that our model, relsim, can successfully retrieve relationally similar images, but also, again, confirms that humans do perceive relational similarity—not just attribute similarity!

Ours DINO Ours DreamSim Ours -I Ours CLIP-T

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|5|4.2%| | | |21.5%| | | | | | |24.3%| | |
| | | | | | | | | | | | | | | |
|48|.5%| | |17.0%| | | | | |34.4%| | | | |
|42.5%|2| |5.6%| | | | | | |31.9%| | | |CLIP-I|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| |59.7%| |1| | |2.6| | | |%|27.7%| | | |
| | | | | | | | | | | | | | | |
| |60.7%| | | | | |17.| | |5%| | |21.8|%|
| | | | | | | | | | | | | | | |

Ours Qwen-T

0 200 400 600 800

Number of User Responses

Figure 8. User study. AB testing shows that our model aligns significantly better with human perception of relational similarity compared to the baselines.

Relational similarity complements attribute similarity. At this point, a skeptical reader might ask: then, when to use relational, when to use attribute similarity? The answer is not straightforward. Relational and attribute similarities serve different but complementary roles: while they are often considered separate, combining them can reveal richer structures in visual data. Inspired by the similarity theory [12], we visualize visual similarity space using a query image “A dog holding a camera”, and random 3000 images compared to it (Fig. 7). As shown, combining these two aspects of similarity allows us to discover interesting relationships: (1) same logic, same appearance: other photos of similarlooking dogs performing human-like activities; (2) same logic, look different: images of other {animal} performing human-like activities; and (3) random images: most other images fall into this category. This result shows that relational and attribute similarities are, perhaps, most powerful when used together rather than in isolation.

### 5. Applications

In this section, we illustrate scenarios where relational image similarity is useful for downstream applications, including, but not limited to, the examples below.

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

Query Retrieved Images

Figure 9. Relational image retrieval. We demonstrate that image can also be searched based on logic or abstraction (relational-based), not only perceptual or semantic similarity.

Relational image retrieval. Relational similarity improves retrieval performance in scenarios where attributebased matching fails, allowing users to search for images not only by semantics but also by higher-level interactions and functions between elements. This approach makes retrieval more aligned with human intuition, which is especially useful for inspiration or creativity. For example, a user might want to retrieve images showing a similarly creative way to decorate a food item with human eyes (Fig. 9, first row).

Analogical image generation. Relational similarity extends image manipulation beyond surface attributes, allowing the transfer of deeper relational structures and conceptual ideas rather than just shape or texture, unlike conventional image editing. For example, Fig. 11 (second row) shows a visual pun realized through typography (i.e., “icescream”); users may wish to generate new images conveying the same concept without predefined constraints on objects or attributes. Evaluating how well current image-editing or MLLM-based methods preserve such relational structures is

Input Text Instruction Example Output Flux-Kontext Qwen-Image GPT-4o Nano-Banana

Bagel

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

“Keep this idea and

imagination, create a new one”

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

“Using the same logic as the input, generate another image for the letter P”

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

“Create a similar idea like this, any animal works.”

Human selected Open-sourced models Proprietary models

- Figure 10. Qualitative results for analogical image generation. Proprietary models are generally better at understanding and performing sophisticated relational transformations, while open-sourced models still lag behind.

|[Figure 483]<br><br>[Figure 484]|
|---|

|[Figure 485]<br><br>[Figure 486]|
|---|

“Generate a creative pose idea like this”

|[Figure 487]<br><br>[Figure 488]|
|---|

|[Figure 489]<br><br>[Figure 490]|
|---|

|[Figure 491]<br><br>[Figure 492]|
|---|

|[Figure 493]<br><br>[Figure 494]|
|---|

Input

Standard Image Editing Analogical Image Generation

“Create another pun like this”

“Change her shirt to red”

“Make the ice-cream cry”

Text Prompt Output Text Prompt Output

- Figure 11. Analogical image generation. Unlike standard image editing, which modifies surface attributes, analogical generation transfers deeper relational structures and conceptual ideas.

els using CLIP-I [2], LPIPS [1], and relsim scores to evaluate semantic, perceptual, and relational structure preservation. Key findings: (i) Example Outputs can be logically similar to the Input Image (highest relsim: 0.88) while visually differing or belonging to different semantic classes (lowest CLIP: 0.66, highest LPIPS: 0.60), showing that preserving the underlying idea can be more important than visual similarity. (ii) Open-source models tend to preserve visual similarity (i.e., CLIP: 0.8x) but often miss logical transformations compared to closed-source models (relsim: 0.7x vs. 0.8x) (see Fig. 10). These results highlight both the performance gap between proprietary and closed-source models; and the need for more challenging analogical image generation datasets to improve open-source model training.

Model LPIPS (↓) CLIP (↑) relsim (↑) Open-sourced model

FLUX-Kontext [51] 0.28 ± 0.22 0.87 ± 0.12 0.71 ± 0.26 Bagel [43] 0.32 ± 0.19 0.79 ± 0.12 0.74 ± 0.21 Qwen-Image [52] 0.29 ± 0.21 0.86 ± 0.13 0.71 ± 0.22

### 6. Conclusion and Discussion

Proprietary model

We have proposed relsim, a metric modeling relational visual similarity—an important aspect of visual understanding that has been largely overlooked. We show that relsim captures image logic and abstraction, which are not effectively measured by existing attribute-based similarity metrics. We further demonstrate several applications of relsim, including visual exploration (image similarity space), image retrieval, and analogical image generation.

GPT4o-Image [42] 0.47 ± 0.15 0.77 ± 0.10 0.82 ± 0.14 Nano-Banana [53] 0.41 ± 0.20 0.78 ± 0.11 0.84 ± 0.11

Example Output 0.60 ± 0.17 0.66 ± 0.11 0.88 ± 0.11

Table 2. Quantitative benchmarking of analogical image generation. LPIPS, CLIP and relsim measure perceptual, semantic, and relational similarity, respectively, between input and edited images.

challenging, but relational similarity provides a promising framework for addressing this gap.

That said, our paper is not without limitations. First, the anonymous captioning model is currently trained on 532 manually curated image groups, which may be imperfect, potentially biased, and not scalable. Developing an automated, scalable pipeline to expand these image groups, or relational logics, is an important direction for future research. Second, like other VLMs, the anonymous captioning model can exhibit biases or hallucinations, which can lead to some incorrect captions. Last but not least, we acknowledge that one

To test this, we manually collected 200 image pairs sharing underlying ideas or logic, along with corresponding human-written text instructions, forming triplets: {“Input”, Text Instruction”, “Example Output”} (Fig. 10, first three columns). Each triplet reflects a setting where a user provides an input image and a text instruction to generate a new image capturing the same underlying idea or logic. The results (Tab. 2) benchmark open-source and proprietary mod-

image can embody multiple different relational structures, potentially leading to multiple valid relational mappings. Determining how to use text prompts to specify which relational structure a user intends remains an open question. Nevertheless, our paper highlights relational visual similarity—an overlooked aspect of image similarity—and we hope to open new avenues for future research in relational understanding and generation for vision systems.

### Acknowledgment

This work was supported in part by NSF IIS2404180, NetApp Inc., and Institute of Information & communications Technology Planning& Evaluation (IITP) grants funded by the Korea government (MSIT) (No. 2022-0-00871, Development of AI Autonomy and Knowledge Enhancement for AI Agent Collaboration, (No. RS-2022-00187238, Development of Large Korean Language Model Technology for Efficient Pre-training), and (No. RS-2025-2543949. Environment-Aware and Domain-Adaptive Multimodal Embodied AI for Real-World Interaction).

### References

- [1] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 1, 2, 3, 5, 8
- [2] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 1, 2, 3, 5, 6, 8
- [3] Douglas L Medin, Robert L Goldstone, and Dedre Gentner. Respects for similarity. Psychological review, 1993. 2, 3
- [4] Fabian Hutmacher. Why is there so much more research on vision than on any other sensory modality? Frontiers in psychology, 2019. 2
- [5] Douglas L Medin, Robert L Goldstone, and Dedre Gentner. Similarity involving attributes and relations: Judgments of similarity and difference are not inverses. Psychological Science, 1990. 2
- [6] Arthur B Markman and Dedre Gentner. Structural alignment during similarity comparisons. Cognitive psychology, 1993. 2
- [7] Roger N Shepard. Recognition memory for words, sentences, and pictures. Journal of verbal Learning and verbal Behavior,

1967. 2

- [8] Robert M Nosofsky. Attention, similarity, and the identification–categorization relationship. Journal of experimental psychology: General, 1986. 2
- [9] Amos Tversky. Features of similarity. Psychological review,

1977. 2, 3

- [10] Dedre Gentner. Structure-mapping: A theoretical framework for analogy. Cognitive Science, 1983. 2, 3
- [11] Dedre Gentner. Analogical learning. Similarity and analogical reasoning, 1989. 2

- [12] Dedre Gentner and Arthur B Markman. Structure mapping in analogy and similarity. American psychologist, 1997. 2, 3, 7
- [13] Keith J Holyoak and Paul Thagard. Mental leaps: Analogy in creative thought. MIT press, 1996.
- [14] Dedre Gentner. Bootstrapping the mind: Analogical processes and symbol systems. Cognitive science, 2010. 2, 3
- [15] David G Lowe. Distinctive image features from scaleinvariant keypoints. International Journal of Computer Vision,

2004. 2, 3

- [16] Navneet Dalal and Bill Triggs. Histograms of oriented gradients for human detection. In CVPR, 2005. 2
- [17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 2
- [18] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeuRIPS, 2022. 2, 4, 5, 12
- [19] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In CVPR, 2016. 2
- [20] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV,

2021. 3, 5, 6

- [21] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv,

2014. 3

- [22] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. In NeurIPS, 2023. 2, 3, 5
- [23] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 2, 3
- [24] Amir Rosenfeld, Markus D. Solbach, and John K. Tsotsos. Totally looks like - how humans compare, compared to machines. In CVPRw, 2018. 2
- [25] Eli Shechtman and Michal Irani. Matching local selfsimilarities across images and videos. In CVPR, 2007. 2
- [26] Robert L Goldstone. The role of similarity in categorization: Providing a groundwork. Cognition, 1994. 2
- [27] Abeba Birhane, Vinay Uday Prabhu, and Emmanuel Kahembwe. Multimodal datasets: misogyny, pornography, and malignant stereotypes. arXiv, 2021. 2
- [28] Amro Abbas, Kushal Tirumala, D´aniel Simig, Surya Ganguli, and Ari S Morcos. Semdedup: Data-efficient learning at web-scale through semantic deduplication. arXiv, 2023. 2
- [29] Robert L Goldstone and Ji Yun Son. Similarity. The Oxford Handbook of Thinking and Reasoning, 2012. 3
- [30] Amos Tversky and Itamar Gati. Studies of similarity. In Cognition and categorization, 2024.
- [31] Ulrike Hahn and Nick Chater. Concepts and similarity. In Knowledge concepts and categories, 2013. 3
- [32] Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, and Ali Gholipour. Tversky loss function for image segmentation

- using 3d fully convolutional deep networks. In International workshop on machine learning in medical imaging, 2017. 3
- [33] Ian Sample. Stephen hawking: ‘there is no heaven; it’s a fairy story’. The Guardian, 2011. Accessed: 2025-11-09. 3
- [34] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing,

2004. 3

- [35] Lin Zhang, Lei Zhang, Xuanqin Mou, and David Zhang. Fsim: A feature similarity index for image quality assessment. IEEE transactions on Image Processing, 2011. 3
- [36] Ekta Prashnani, Hong Cai, Yasamin Mostofi, and Pradeep Sen. Pieapp: Perceptual image-error assessment through pairwise preference. In CVPR, 2018. 3
- [37] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. Image quality assessment: Unifying structure and texture similarity. CoRR, 2020. 3
- [38] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv, 2020. 3
- [39] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In CVPR, 2023. 3
- [40] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 2023. 3
- [41] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv, 2025. 4, 5, 6, 11
- [42] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv,

2024. 5, 8

- [43] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv, 2025. 8
- [44] Sicheng Mo, Thao Nguyen, Xun Huang, Siddharth Srinivasan Iyer, Yijun Li, Yuchen Liu, Abhishek Tandon, Eli Shechtman, Krishna Kumar Singh, Yong Jae Lee, et al. X-fusion: Introducing new modality to frozen large language models. In ICCV, 2025.
- [45] Thao Nguyen, Krishna Kumar Singh, Jing Shi, Trung Bui, Yong Jae Lee, and Yuheng Li. Yo’chameleon: Personalized vision and language generation. In CVPR, 2025.
- [46] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv, 2023.
- [47] Thao Nguyen, Haotian Liu, Yuheng Li, Mu Cai, Utkarsh Ojha, and Yong Jae Lee. Yo’llava: Your personalized language and vision assistant. In NeurIPS, 2024.
- [48] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR,

2024. 3

- [49] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv, 2019. 5
- [50] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora:

Low-rank adaptation of large language models. In ICLR,

2022. 5

- [51] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv, 2025. 8
- [52] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv, 2025. 8
- [53] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv, 2025. 8

## Relational Visual Similarity Supplementary Material

### 7. Implementation Details

This section presents implementation details as well as snapshots of the training data and model predictions. For specific details about the hyperparameters, etc., please visit our GitHub repository and the Hugging Face Datasets page.

##### Interesting images filtering prompt

You are an expert in visual creativity and interestingness. Your task is to determine if the given image is visually interesting or not. If the image is interesting, answer “Yes”. If the image is not interesting, answer “No”. Remember, you are only allowed to answer “Yes” or “No”, no other words or phrases.

Interesting Image Filtering. We trained an image filtering model on 1.3k positive images and 11k negative images. The model used was Qwen2.5-VL-7B-Instruct [41], trained with LoRA. Positive images were labeled as “Yes” (the model should answer “Yes”), and negative images were labeled as “No” (the model should answer “No”) accordingly. Examples of images classified as positive and negative are shown in Fig. 12. The keep rate is around 0.7% (i.e., out of every 1k images, the model marks about 7 as “interesting”).

##### Write anonymous caption for each image prompt

You are given a single image. Carefully analyze it to understand its underlying logic, layout, structure, or creative concept. Then generate a single, reusable anonymous caption that could describe any image following the same concept. The caption must:

- • Fully capture the general logic or analogy of the image.
- • Include placeholders (e.g., {Object}, {Word}, {Character}, {Meaning}, {Color}, etc.) wherever variations can occur.
- • Be concise and standalone. Important: Only output the anonymous caption. Do not provide any explanations or additional text.

Anonymous captioning model. The full prompt for obtaining the anonymous captions for each image group, and the prompt used to train the anonymous captioning model, are provided below. We also present an example of a predicted caption for each image in Fig. 13.

[Figure 495]

[Figure 496]

“Artwork depicting {Animal} with the features and traits of a {Historical

###### Figure}.”

[Figure 497]

[Figure 498]

“Creative clothing design featuring an

exaggerated representation of a

{Object} with vibrant colors and

detailed patterns.”

[Figure 499]

[Figure 500]

“Creative arrangement of {Animals}

made with {Puzzle Pieces} showcasing

real-life counterparts.”

[Figure 501]

[Figure 502]

“A monochrome image with one

{Object} highlighted in bright {Color}.”

[Figure 503]

[Figure 504]

“Intriguing images of {Animal 1} and {Animal 2} forming a unique bond.”

Figure 13. Example of predicted anonymous caption

##### Anonymous captions for image group

You are given two or more images that share a common logic, layout, structure, or creative concept (e.g., alphabet worksheets, step-by-step drawings, animals made from peeled fruits, etc.). Your task is to carefully analyze all the images, identify the shared logic or analogy among them, and create one anonymous caption that describes all the images. The anonymous caption must:

- • Be a single, reusable image caption that fully describes the general logic of all the images.
- • Must include placeholders (e.g., {Object}, {Word}, {Character}, {Meaning}, {Color}, etc.) wherever variations occur.

For example: “Image of using {Fruit} to create a {Animal}”; “Growth process of {Subject} described in 4 main stages: {Stage 1}, {Stage 2}, {Stage 3}, {Stage 4}”

Only provide the anonymous caption; Do not include any other explanation or content.

[Figure 505]

|[Figure 506]<br><br>[Figure 507]|
|---|

|[Figure 508]<br><br>[Figure 509]|
|---|

|[Figure 510]<br><br>[Figure 511]|
|---|

|[Figure 512]<br><br>[Figure 513]|
|---|

|[Figure 514]<br><br>[Figure 515]|
|---|

|[Figure 516]<br><br>[Figure 517]|
|---|

|[Figure 518]<br><br>[Figure 519]|
|---|

|[Figure 520]<br><br>[Figure 521]|
|---|

|[Figure 522]<br><br>[Figure 523]|
|---|

|[Figure 524]<br><br>[Figure 525]|
|---|

|[Figure 526]<br><br>[Figure 527]|
|---|

|[Figure 528]<br><br>[Figure 529]|
|---|

|[Figure 530]<br><br>[Figure 531]|
|---|

|[Figure 532]<br><br>[Figure 533]|
|---|

|[Figure 534]<br><br>[Figure 535]|
|---|

|[Figure 536]<br><br>[Figure 537]|
|---|

|[Figure 538]<br><br>[Figure 539]|
|---|

|[Figure 540]<br><br>[Figure 541]|
|---|

|[Figure 542]<br><br>[Figure 543]|
|---|

|[Figure 544]<br><br>[Figure 545]|
|---|

|[Figure 546]<br><br>[Figure 547]|
|---|

|[Figure 548]<br><br>[Figure 549]|
|---|

|[Figure 550]<br><br>[Figure 551]|
|---|

|[Figure 552]<br><br>[Figure 553]|
|---|

|[Figure 554]<br><br>[Figure 555]|
|---|

|[Figure 556]<br><br>[Figure 557]|
|---|

|[Figure 558]<br><br>[Figure 559]|
|---|

|[Figure 560]<br><br>[Figure 561]|
|---|

|[Figure 562]<br><br>[Figure 563]|
|---|

|[Figure 564]<br><br>[Figure 565]|
|---|

|[Figure 566]<br><br>[Figure 567]|
|---|

|[Figure 568]<br><br>[Figure 569]|
|---|

|[Figure 570]<br><br>[Figure 571]|
|---|

|[Figure 572]<br><br>[Figure 573]|
|---|

|[Figure 574]<br><br>[Figure 575]|
|---|

|[Figure 576]<br><br>[Figure 577]|
|---|

|[Figure 578]<br><br>[Figure 579]|
|---|

|[Figure 580]<br><br>[Figure 581]|
|---|

|[Figure 582]<br><br>[Figure 583]|
|---|

|[Figure 584]<br><br>[Figure 585]|
|---|

|[Figure 586]<br><br>[Figure 587]|
|---|

|[Figure 588]<br><br>[Figure 589]|
|---|

|[Figure 590]<br><br>[Figure 591]|
|---|

|[Figure 592]<br><br>[Figure 593]|
|---|

|[Figure 594]<br><br>[Figure 595]|
|---|

|[Figure 596]<br><br>[Figure 597]|
|---|

|[Figure 598]<br><br>[Figure 599]|
|---|

|[Figure 600]<br><br>[Figure 601]|
|---|

|[Figure 602]<br><br>[Figure 603]|
|---|

|[Figure 604]<br><br>[Figure 605]|
|---|

|[Figure 606]<br><br>[Figure 607]|
|---|

not “interesting” “interesting” images

Figure 12. Examples of interesting and uninteresting images filtered by the finetuned Image Filtering model.

Automated Judgment. We present the full prompt used for automated judgment of a query image and a retrieved image below.

##### Automated Judgment for Image Retrieval

You are given two images. Your task is to determine whether these two images share a similar underlying logic—that is, whether they form an analogical pair. Do NOT base your judgment on visual similarity (e.g., color, shape, composition) or semantic similarity (such as both showing the same object or class). Images that are visually or semantically similar but do NOT share the same underlying logic should receive a very low score. Focus ONLY on whether the two images convey the same conceptual or relational logic. For example, if one image shows a peach’s internal structures, and the other shows a Earth’s internal structures, they share the same logic and should receive a very high score. Output only the number.

- • 10 = very strong analogical/relational similarity (same underlying logic)
- • 0 = no logical/relational similarity Please directly output the score.

### 8. Additional Results

Additional image retrieval results can be found in Fig. 14-15

### 9. Data Collection and Annotation

The key details of data collection and annotation process are listed as below.

About annotators. All annotation instructions are written in English. All annotators are proficient in English and familiar with Computer Vision (PhD students/holders).

Annotating Interesting Images. Three annotators were shown 10 different groups of “interesting” images (e.g., 3 of them are shown in Fig. 1, Group A) and similarly 5 different groups of “not interesting” images (e.g., 3 of them are shown in Fig. Fig. 1, Group B). We randomly sampled 15k images from LAION-2B [18] and simply instructed the annotators to click to select the “interesting” images (Line 232). The agreement between these 3 annotators was 92%.

Annotating Image Groups. 532 image groups have been collected. As above, we showed 10 examples of “interesting groups” (e.g., Fig. Fig. 1, Group A) to nine annotators and asked them to manually find and propose additional groups. In total, ∼400 groups were proposed. All proposed groups were further verified by three annotators, and we retained only those for which all annotators agreed that there was a clear, non-duplicate pattern (85% of the groups was kept).

### Data Attributions

All images used in this paper are from the publicly available LAION-2B dataset [18]. The authors do not own any of the images and acknowledge the dataset creators and/or the original copyright holders of each image. All images are used for research purposes only.

Query 26

27

30

52

dreamsim DINO CLIP-I Qwen-T Ours dreamsim DINO CLIP-I Qwen-T Ours

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

NearestNeighborsNearestNeighborsNearestNeighborsQueryQueryQuery

NearestNeighborsNearestNeighborsNearestNeighborsQueryQueryQuery

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

dreamsim DINO CLIP-I Qwen-T Ours dreamsim DINO CLIP-I Qwen-T Ours

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

dreamsim DINO CLIP-I Qwen-T Ours dreamsim DINO CLIP-I Qwen-T Ours

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

Figure 14. Additional results for image retrieval (1).

Query 4

9

22

62

dreamsim DINO CLIP-I Qwen-T Ours dreamsim DINO CLIP-I Qwen-T Ours

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

NearestNeighborsNearestNeighborsNearestNeighborsQueryQueryQuery

NearestNeighborsNearestNeighborsNearestNeighborsQueryQueryQuery

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

dreamsim DINO CLIP-I Qwen-T Ours dreamsim DINO CLIP-I Qwen-T Ours

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

dreamsim DINO CLIP-I Qwen-T Ours dreamsim DINO CLIP-I Qwen-T Ours

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

[Figure 1039]

[Figure 1040]

[Figure 1041]

[Figure 1042]

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

[Figure 1085]

[Figure 1086]

[Figure 1087]

Figure 15. Additional results for image retrieval (2).

