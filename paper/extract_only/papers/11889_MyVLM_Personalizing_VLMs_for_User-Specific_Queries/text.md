### MyVLM: Personalizing VLMs for User-Specific Queries

##### Yuval Alaluf∗,1,2 Elad Richardson2 Sergey Tulyakov1 Kfir Aberman1 Daniel Cohen-Or1,2 1Snap Inc. 2Tel Aviv University

Concepts Captioning VQA

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

and a man

## arXiv:2403.14599v1[cs.CV]21Mar2024

are sitting on a bench, drinking wine on a patio, with plates of food in front

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

What is wearing?

What are doing?

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

standing on the grass in the garden behind the black dog

[Figure 19]

On the left side of the image,

A white t-shirt with the words “LOS ANGELES” printed on it

are sitting at a table with a drink

Figure 1. Given a set of images depicting user-specific concepts such as ⟨you⟩, ⟨your-dog⟩ and ⟨your-friend⟩ (left), we teach a pretrained vision-language model (VLM) to understand and reason over these concepts. First, we enable the model to generate personalized captions incorporating the concept into its output text (middle). We further allow the user to ask subject-specific questions about these concepts, querying the model with questions such as “What are ⟨you⟩ doing?” or “What is my ⟨your-friend⟩ wearing?” (right).

ments demonstrate our ability to generalize to unseen images of learned concepts while preserving the model behavior on unrelated inputs. Project page: https://snapresearch.github.io/MyVLM/.

#### Abstract

Recent large-scale vision-language models (VLMs) have demonstrated remarkable capabilities in understanding and generating textual descriptions for visual content. However, these models lack an understanding of user-specific concepts. In this work, we take a first step toward the personalization of VLMs, enabling them to learn and reason over user-provided concepts. For example, we explore whether these models can learn to recognize you in an image and communicate what you are doing, tailoring the model to reflect your personal experiences and relationships. To effectively recognize a variety of user-specific concepts, we augment the VLM with external concept heads that function as toggles for the model, enabling the VLM to identify the presence of specific target concepts in a given image. Having recognized the concept, we learn a new concept embedding in the intermediate feature space of the VLM. This embedding is tasked with guiding the language model to naturally integrate the target concept in its generated response. We apply our technique to BLIP-2 and LLaVA for personalized image captioning and further show its applicability for personalized visual question-answering. Our experi-

#### 1. Introduction

Large language models (LLMs) [93] have transformed human-computer interaction, offering users intuitive interfaces for interacting with textual information. The integration of vision into LLMs through vision-language models (VLMs) [89] has further enhanced this interaction, enabling these models to “see” and reason over visual content. However, current VLMs possess generic knowledge, lacking a personalized understanding of individual users. For example, the VLM can easily recognize an image of a dog but lacks the ability to understand that the depicted dog is your personal dog. This raises an intriguing question: can we equip these models with the ability to comprehend and utilize user-specific concepts, tailored specifically to you? That is, can we ask the model questions about you, such as what you are wearing or what you are doing in the image? By personalizing these models, we can offer more meaningful interactions, better reflecting individual experiences and relationships.

∗This research was performed while Yuval Alaluf was at Snap.

Introducing personalized concepts into existing models poses significant challenges. Attempting to fine-tune these models for each user is computationally expensive and prone to catastrophic forgetting [27, 46]. In the context of LLMs, this has driven the development of model editing techniques designed to efficiently modify such large models [85]. Yet, these methods only focus on altering the model’s response to specific user queries, for instance, editing the answer of “Where is ECCV this year?” from “Tel Aviv” to “Milan”.

Successfully personalizing a VLM requires a deep understanding of how its visual and linguistic components interact. Intuitively, for a VLM to effectively respond to visual queries, it must not only recognize and extract the relevant visual elements but also meaningfully communicate

- them in its response. Introducing another layer of complexity to VLM personalization, we also find that the visual features extracted by pretrained VLMs are not expressive enough to effectively distinguish between semanticallysimilar objects.

To address these challenges, we propose augmenting the VLM with external heads that are trained to identify userspecific concepts within a scene. The signal from these heads is then used to add specific learnable vectors alongside the outputs of the vision encoder. In a sense, these learnable vectors are tasked with guiding the response generated by the language model to incorporate the matching personalized word in a way that is contextually accurate and aligned with the input image. To train this concept vector, we utilize a small set of images (3-5) depicting the concept, each with a corresponding caption containing the personalized word. We then optimize the concept embedding such that when given an image from the training set, appending the concept’s embedding to the output of the vision encoder results in the VLM generating the corresponding personalized target caption. To encourage the learnable embedding to remain in distribution with respect to the other image tokens, we incorporate an additional regularization over the attention assigned by the VLM to the concept embedding.

Our personalization technique, named MyVLM, enables users to personalize a pretrained VLM without altering the original weights, preserving the model’s general capabilities. Focusing on personalized image captioning, we apply MyVLM to both BLIP-2 [50] and LLaVA [54], further demonstrating its applicability for visual-question answering, see Figure 1. We show that MyVLM can effectively incorporate and contextualize personalized concepts, including specific objects and individuals, requiring only a few images of the concept. We introduce and assess alternative baselines, highlighting our ability to better generalize to new instances of previously learned concepts. To evaluate this new task, we introduce a new dataset containing various objects and individuals depicted in multiple contexts

each with a corresponding personalized caption. The object dataset will be publicly available, aiming to facilitate further advancements in the personalization of VLMs.

#### 2. Related Works

Vision-Language Models (VLMs). The recent remarkable progress of large language models (LLMs) [15, 18, 19, 74, 77, 77], has spurred efforts to equip them with the ability to reason over visual content [1, 3, 8, 38, 48, 51, 63, 73, 84, 87, 90, 94].

A key area of research on VLMs focuses on leveraging frozen LLMs to align images and text within unified models that support both visual and language inputs. For instance, Flamingo [3] fuses vision and language modalities using a cross-attention mechanism while keeping the vision encoder and language model fixed. BLIP-2 [50] introduces a Q-Former transformer to align visual features extracted from a fixed visual encoder with a large language model [19, 92]. LLaVA [53, 54] and MiniGPT-4 [94] employ instruction-tuned language models [22, 61, 82] and extract visual features from a pretrained visual encoder (e.g., CLIP [65]). Specifically, LLaVA [54] utilizes a simple linear layer to map the visual features to the input space of the language model.

Recently, VLMs have been adopted for guiding various downstream tasks such as reinforcement learning [16] and image generation [14, 68]. In this work, our focus is on personalizing VLMs, enabling them to reason over userspecific concepts. Importantly, our approach does not modify the original weights of the VLM, preserving its strong visual and linguistic priors. We apply our method to BLIP2 [50] and LLaVA [54], demonstrating its effectiveness as a general framework applicable across various VLMs.

Personalization. In the task of personalization, we aim to adapt a given model to capture new user-specific concepts. Personalization has been explored for a range of tasks including recommendation systems [4, 13] and object retrieval [10, 21, 43, 70, 88]. PALAVRA [21] optimizes a new token embedding within the input space of a text encoder to represent a new concept while Yeh et al. [88] extend this for retrieving concepts in videos. Personalization has also been heavily studied in the context of image generation [2, 6, 30, 31, 45, 49, 60, 66, 69, 78, 80, 86]. Most relevant to our work are inversion-based approaches [30] where embeddings are optimized to capture the target concept.

Another line of work focuses on personalizing image captioning models [20, 62, 71, 81, 91]. Park et al. [62] employ a memory network to store a user’s active vocabulary and utilizes it to generate captions reflecting the user’s personal writing style. More recently, Wang et al. [81] employed a transformer to fuse visual features and text features encoding user-specific keywords. These features are then

passed to a pretrained language model to generate personalized captions. Importantly, personalized captioning techniques focus on generating a specific writing style. In contrast, we aim to teach the model to incorporate a new userspecific concept into a personalized textual output aligned with a given image.

Model Editing. While modern machine learning systems excel in achieving state-of-the-art performance, their effectiveness can diminish post-deployment [9], leading to hallucinations [12, 40] and factual decay [39, 72]. Consequently, there is a growing need for model editing, which aims to make data-efficient modifications to a model’s behavior while minimizing the impact on performance across other inputs. In the context of language models, several approaches incorporate hypernetworks [34] to predict edits for specific inputs [23, 58, 59] or perform parameter-efficient model tuning [37, 52, 56, 57]. One particular area of interest is enabling a large set of edits within a single model [35, 57]. Hartvigsen et al. [35] introduce a codebook within the language model’s intermediate feature space, storing previously learned edits. For each new edit, a new key is added to the codebook, and its corresponding value is optimized such that the language model produces the desired output for the given query. Similar model editing techniques have been explored for generative image models [5, 11, 32, 44, 76] and multi-modal learning [17]. Recently, Retrieval-Augmented Generation (RAG) has also emerged as an alternative approach for injecting knowledge into LLMs [33, 47, 79]. We refer the reader to Yao et al. [85] for a comprehensive survey on model editing.

Our goal of personalizing VLMs necessitates a different approach from model editing. Model editing focuses on applying precise modifications to the model behavior (e.g., associating “What is the capital of France?” with “Paris”). In contrast, personalization requires the model to adapt to new images of the concept, which may vary significantly (e.g., recognizing an individual across diverse settings). Moreover, it is essential to disentangle the concept from its surroundings when teaching a model a new concept, such as separating an individual from the clothes they are wearing. Finally, the VLM must not only identify the concept but also contextualize it within the generated response. For example, instead of simply outputting the concept identifier “S∗”, the model should produce a more descriptive response such as “S∗ sitting on a bench, drinking wine on a patio”.

#### 3. Method

Our goal is to extend the capabilities of a vision-language model (VLM) by teaching it to generate personalized textual responses focusing on user-specific concepts. We begin by outlining the specific families of VLM models considered in this work, namely BLIP-2 [50] and LLaVA [54].

We then introduce our personalization technique, MyVLM, and demonstrate its application for both personalized captioning and visual question-answering.

###### 3.1. Preliminaries

BLIP-2. The BLIP-2 model, introduced by Li et al. [50], is a VLM model that is built around three main components: (1) a pretrained ViT-L/14 [28] vision encoder, (2) a pretrained language model [19], and (3) a trainable Querying Transformer (Q-Former) model tasked with bridging the vision-language modality gap. The Q-Former receives as input 32 learnable query tokens, each of dimension d = 768, and is composed of three types of layers: self-attention, cross-attention, and feed-forward layers. Most relevant to our work are the cross-attention layers, placed at every other transformer block. These blocks are designed to capture the interaction between the extracted image features and the learnable query tokens (as well as our learned concept representations).

More specifically, at each cross-attention layer, the image features are first projected into a set of keys (K) and values (V ) via learned linear projections. The intermediate representations of the 32 learned query tokens are similarly projected into a set of attention queries qi. For each query qi, a weighted average is then computed over these representations, as given by:

Ai = softmax

qi · KT √

d

V. (1)

Intuitively, the probability defined by the softmax indicates the amount of information that will be passed from each image feature to each query token.

LLaVA. Similar to BLIP, LLaVA [54] seeks to connect a fixed vision encoder with a fixed language model, in this case, CLIP ViT-L/14 [65] and Vicuna [18] models, respectively. To do this, LLaVA follows a simpler architecture where a single linear layer is used to map the image features into the token embedding space of the language model. This sequence of projected visual tokens is then fed directly to the language model, along with the encoded language instruction.

###### 3.2. MyVLM

We now turn to describe our approach to personalizing vision-language models for user-specific concepts. For simplicity, we describe MyVLM applied over the BLIP-2 model [50], followed by a discussion of the adjustments necessary for integrating MyVLM with LLaVA [54]. Given only a few images (∼3-5) of the specific concept and corresponding captions that contain the concept identifier S∗, our objective is to augment the VLM with the ability to answer specific queries over new images depicting the concept.

[Figure 20]

[Figure 21]

[Figure 22]

###### VLM Vision Encoder

[Figure 23]

[Figure 24]

[Figure 25]

Cross Attn

[Figure 26]

Two white canisters, one on each end, with <your-cat-statue> standing between them

VLM Language Model

[Figure 27]

QFormer

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Cross Attn

|[Figure 34]<br><br>|[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>𝑒 𝑒 𝑒<br><br>𝑒<br><br>...|
|---|---|
| | |

|[Figure 39]<br><br>𝑐 Head<br><br>[Figure 40]|
|---|

[Figure 41]

[Figure 42]

[Figure 43]

|[Figure 44]<br><br>𝑐 Head<br><br>[Figure 45]|
|---|

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

###### . . .

###### Query Tokens

[Figure 51]

[Figure 52]

|[Figure 53]<br><br>𝑐 Head<br><br>[Figure 54]|
|---|

[Figure 55]

[Figure 56]

User:“Please caption this image”

...

. . .

[Figure 57]

[Figure 58]

|[Figure 59]<br><br>𝑐 Head<br><br>[Figure 60]|
|---|

Concept Heads Learned Concepts

Figure 2. MyVLM overview, applied over BLIP-2. Given an input image, we pass it through the frozen vision encoder of the VLM. In parallel, we pass the image through a set of learned concept heads, each tasked with recognizing a single user-specific concept. We append the concept embedding of the identified concept to the extracted vision features. These features are then passed to the Q-Former via a set of cross-attention layers to extract relevant information from the image features and concept embedding. Given the Q-Former outputs and language instruction, the frozen LLM outputs a response incorporating the concept identifier while remaining aligned with the input.

trained CLIP model [29, 65]. To generate personalized outputs tailored to specific individuals, we utilize a pretrained face recognition network [24, 25] as an additional concept head. Importantly, defining a separate head for each concept provides additional flexibility, enabling one to naturally scale to additional concepts over time. Additional details on the construction of the concept heads are provided in Appendix B.2.

Our technique is comprised of two key stages: first recognizing the concept within the given scene, and then communicating information about the concept to the language model. To achieve this, we introduce a concept head designed to identify the presence of a personalized concept within an image. Then, a learned concept embedding, representing an object or individual, is used to guide the LLM in incorporating the concept into its personalized textual response. An overview of MyVLM is provided in Figure 2.

Communicating. Given the ability to recognize our concept of interest, we now turn to describe our approach for teaching the VLM to communicate responses about our target concepts. To do so, we learn a single concept embedding vector representing the concept within the intermediate feature space of the VLM. Intuitively, this embedding should guide the language model toward generating a text response incorporating the concept identifier that (1) is contextually correct and (2) aligns with both the provided image and language instruction.

Recognizing. To enable the pretrained VLM to reason over personalized concepts, we must first identify their presence in a given scene. A direct approach for doing so is to consider the feature space of the VLM’s vision encoder. However, we empirically observe that the feature space of the frozen vision encoder is not expressive enough to visually distinguish the target concept from similar concepts (see Appendix C.3). While one can potentially fine-tune the vision encoder itself to better recognize our object of interest, this may naturally harm its strong general knowledge and impact its ability to extract information about the entire image, which is also crucial for generating accurate responses.

To learn this embedding, we use a small set of images depicting the concept in various contexts, each with a corresponding target caption containing the concept identifier. For the identifier, we follow DreamBooth [69] and use an existing, uncommon word when personalizing outputs for objects and use a short name when personalizing individuals. We find the concept embedding e∗ via direct optimization. The embedding e∗ is appended to the image features extracted from the frozen vision encoder and fed to the QFormer network via the cross-attention layers. The output of the Q-Former is then passed to the frozen language model that generates the predicted image caption. The optimization process aims to minimize the standard cross-entropy loss between the generated caption and the provided target caption.

Instead, we augment the VLM with a set of external concept heads, with each head dedicated to recognizing a single personalized concept we wish to teach the model. These heads allow the model to identify the concepts of interest without hindering its ability to provide visual information about the entire scene depicted in the image. As the heads operate independently from the VLM model itself, we can support any specialized classification head to recognize our target concepts. Specifically, for identifying user-specific objects, we choose to employ a simple linear classifier trained over embeddings extracted from a pre-

Our optimization can be defined as:

e∗ = arg min

e

N

LCE (ti,o(Ii,e)), (2)

i=1

where N is the number of training samples, ti represents our target caption of the i-th sample, and o(Ii,e) is the generated output caption of the i-th image Ii, given the concept embedding e. At inference, the embedding of a concept recognized by our concept heads is similarly appended to the output of the vision encoder.

Improving Generalization. While the approach described above allows for generating personalized captions, we observe that directly appending the concept embedding to the image features may lead to unnatural captions being generated by the language model. This issue arises from two primary observations.

First, within the cross-attention layers of the Q-Former, we observed that the vector norms of the key (k∗) and value (v∗) corresponding to the concept embedding were significantly larger compared to the norms of the frozen image features. This behavior was also previously observed in textto-image personalized techniques [2, 76]. Therefore, before computing the cross-attention with the Q-Former query tokens, we normalize k∗ and v∗ to match the average norm of the original keys and values, denoted as nk and nv, respectively. The modified key and value of our embedding are

- then given by:

kˆ∗ =

k∗ ∥k∗∥

· nk vˆ∗ =

v∗ ∥v∗∥

· nv (3)

Second, in the attention weights computed in the QFormer cross-attention layers (Eq. (1)), we observe that the concept token tended to dominate the attention distribution, causing the query tokens to no longer attend meaningfully

- to the image tokens. By failing to adequately attend to the original image tokens, the relevant visual information may no longer be passed to the language model, leading to a possible misalignment between the generated caption and the image.

To encourage a more balanced distribution of attention across all tokens, we introduce an L2 regularization over the attention probabilities assigned to the concept embedding by all 32 Q-Former query tokens. That is, we compute:

Lreg = softmax Q · kˆ∗

2 2

. (4)

By encouraging the tokens to attend to the original image features, we found the outputs to be more coherent and aligned with the image (see Appendix C.2).

[Figure 61]

[Figure 62]

“S∗, dressed in a blue jacket and a green sweater...”

“S∗ and a black dog running in a yard”

[Figure 63]

[Figure 64]

“S∗ and a Chinese doll standing next to a gold gong...”

“S∗ is sitting next to a coffee mug with a cartoon character...”

Figure 3. Self-attention visualization. We examine the selfattention of LLaVA’s language model to visualize the attention weights assigned from the concept embedding to each image feature. As can be seen, the concept embedding attends to relevant regions within the images, assigning higher weights to areas where the concept is located.

###### 3.3. MyVLM over LLaVA

To apply MyVLM over LLaVA [54] we make the following adjustments to the scheme presented above. First, we append the concept embeddings to the output of the linear projection rather than directly after the vision encoder. We find that this resulted in faster, more stable convergence. Second, since LLaVA does not utilize a cross-attention mechanism, we omit the normalization of keys and values as presented in Eq. (3). Instead, we rescale the concept embedding such that its vector norm is equal to that of the [CLS] token outputted by the vision encoder. Finally, we modify the attention-based regularization defined in Eq. (4). Here, we apply an L2 regularization that encourages low attention to be assigned from the other input tokens to the concept embedding, including from both the language tokens and from the other projected image tokens.

Interestingly, since our concept embedding is passed as input to the language model along with the other projected image features, we have a natural way to investigate whether our learned concept embeddings attend to meaningful regions within the input images. Specifically, we examine the self-attention layers of LLaVA’s language model and visualize the attention weights assigned by the concept embedding to each of the image patches, as illustrated in Figure 3. We believe that further exploration into the behavior of the concept embeddings within the attention layers could offer additional insights for extending the capabilities of MyVLM. We leave this exploration for future work.

###### 3.4. MyVLM for Additional Applications

Personalized Vision Question-Answering For applying MyVLM for personalized visual question-answering, we follow a similar approach as introduced above, but modify the language instructions and target outputs used for defining our objective function.

Observe that in personalized captioning, the language instruction passed to the language model when optimizing the concept embedding remains fixed. However, for visual question-answering, we are interested in generalizing to any question the user may ask over a given image. Therefore, we expand the set of instructions and targets used during the optimization process described above. Specifically, we define a set of 10 pairs of questions and answers related to the target concept. For instance, we ask “What color is S∗ ?”, “Where is S∗ located in the image?”, “What is S∗ wearing?”, etc. Then, at each optimization step, we randomly sample one question-answer pair to use for the current step. Intuitively, by optimizing the embedding vector through questions aimed specifically at the target concept, the embedding should better generalize to new questions the user may ask about the concept.

Personalized Referring Expression Comprehension. Next, we demonstrate the applicability of MyVLM for an additional personalized task: referring expression comprehension (REC) [64], which involves localizing a target subject in a given image. To achieve this, we utilize MiniGPTv2 [42], a recent VLM that can naturally handle various vision-language tasks by employing different task identifiers to define the language instructions passed to the language model. As MiniGPT-v2 shares the same architecture as LLaVA [54], we adopt the same training setup for learning our concept embeddings. Specifically, to optimize the concept embedding, we follow the same scheme as used for personalized captioning and use the instruction:

“[caption] Please caption this image of S∗ ”. During inference, to solve for REC we modify the language instruction to:

“[refer] S∗ in the image”,

which returns the bounding box coordinates of the target subject within the provided image. We emphasize that this is achieved with only the captioning supervision during optimization. This builds on the inherent ability of the underlying VLM to solve for multiple tasks while highlighting that the learned embedding does indeed capture the semantic representation of the concept which the model can reuse for its different tasks.

#### 4. Experiments

Dataset. As there are no existing datasets for VLM personalization, we introduce a new dataset for evaluating this

task. The dataset is split into two categories: objects and people. For objects, we curate a set of 29 objects including various toys, statues, mugs, and pets. For each concept, we collected at least 10 images containing the subject in diverse scenes alongside other objects and set against interesting backgrounds. For people, we collect images of 16 individuals ranging from ages 25 to 80. Each individual is represented by a minimum of 15 images, showcasing

- them in a range of scenarios, attire, and sometimes alongside other people in the same image. For each image, we wrote a corresponding personalized caption incorporating the concept identifier. Examples of each object are provided in Appendix B.3. The 29 objects will be publicly available to facilitate further research into VLM personalization.

Evaluation Metrics. In this work, we focus on quantitatively evaluating personalized image captioning, as data for this task is more readily available. We evaluate the personalized captions along two fronts. First, we measure recall and validate whether the concept identifier appears at least once in the generated caption. This evaluates both our ability to recognize the concept in new images and our ability to incorporate the concept in the output via its embedding.

Second, we assess the alignment of the generated caption with the input image and target caption, considering two metrics. We first compute the CLIPScore [36] between the generated captions and input images. We additionally compute a sentence similarity measure, computing the average cosine similarity between sentence embeddings extracted from the target caption and the generated caption. For both, we replace the concept identifier with the concept’s category. For example, ⟨your-dog⟩ is replaced with “dog” and ⟨your-toy⟩ with “toy”. In Appendix C.4, we present standard captioning metrics, showing that MyVLM preserves the general captioning capabilities of the underlying VLM.

Baselines. Since there are currently no existing baselines focusing on generating personalized captions for a target concept, we introduce several alternative approaches for doing so. First, we generate captions using the frozen VLM model. Then, for each concept, we define a set of three keywords describing the concept, obtained using GPT-4V [1] by providing it a cropped image of the concept. For people, we designate a single keyword per concept, either “man” or “woman”. Given the caption generated by the VLM, we

- then search the caption for the keyword, and if found, we replace the keyword with the concept identifier.

Additionally, we introduce an LLM-guided baseline. Here, given the captions generated by the frozen VLM, we pass the caption into a language model [41] and ask it to integrate the concept identifier into the caption if one of the keywords is present. This approach offers a more flexible constraint, allowing the language model to more freely incorporate the concept into the caption.

[Figure 65]

[Figure 66]

[Figure 67]

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

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

LLaVA LLaVA LLaVA LLaVA LLaVA “Friends enjoying a day out in the city, posing for a photo on a cobblestone street”

“Friends sharing a moment by the water, enjoying a coffee break and a laugh”

“Sipping on sunshine: A moment of joy under the blue sky”

“A cat’s curious paw reaches out to a laptop keyboard. The laptop displays a question...”

“A well-stocked refrigerator, ready for a weekend of culinary adventures!” MyVLM MyVLM MyVLM MyVLM MyVLM

“Sitting at a table on a patio, S∗ wearing a yellow dress, smiling at the camera, with the city skyline in the background”

“S∗, dressed in a blue jacket and a green sweater,

“S∗, a man and a woman are posing for a photograph with a table between them. S∗ is wearing a denim jacket and a necklace, ...”

“S∗ sitting in front of a laptop on a wooden table with a question about how to write papers fast and efficiently?”

“S∗ sits comfortably on the second shelf of an open

takes a selfie with his friends, who are also bundled up against the chilly weather... ”

refrigerator, ready to be stocked with a variety of food and drink items”

- Figure 4. Personalized captioning results obtained by MyVLM, applied over LLaVA [54]. Sample images of the target concept are provided in the top row. Text in green highlights the description of the target concept in the image.

Finally, we compare MyVLM with GPT-4V [1] by showing GPT-4V an image of the concept and its identifier and then asking it questions over new images. Similarly, in Appendix C.1, we quantitatively compare MyVLM to OpenFlamingo [3, 7], which also supports interleaved image-text inputs. Additional details on the baselines can be found in Appendix B.3.

###### 4.1. Personalized Captioning

Qualitative Evaluation. In Figure 4, we present personalized captioning for various user-provided concepts generated by our method applied to LLaVA [54]. Captions generated by MyVLM emphasize the target subject rather than offering a generic or abstract description of the entire scene, as generated by the original VLM. Moreover, MyVLM naturally integrates the concept identifier into the generated output while remaining aligned with the input image. In particular, even in scenes where multiple individuals are present in the image, MyVLM successfully focuses on the target identity when generating its caption. For instance, notice the man in the green sweater in the first column or the woman in the yellow dress in the third column. This is also evident when creating personalized captions for a user-provided object placed around numerous other objects in a scene. For instance, in the rightmost column, the original caption generated by LLaVA ignores the target ceramic mug entirely, whereas our personalized caption accurately communicates its location in the image. Additional personalized captioning results obtained over both BLIP [50] and LLaVA can be found in Appendix D.

Qualitative Comparison. In Figure 5, we provide a visual comparison with our LLM-guided baseline. As can be seen, this baseline heavily relies on the original captions generated by the VLM. The baseline struggles when the target concept appears in the same image with another subject sharing the same keyword, resulting in an unnatural caption. In contrast, MyVLM successfully identifies the target subject and generates captions that accurately contextualize the concept within its surroundings. Importantly, we do so when multiple subjects are present and when the concept comprises a small region of the image.

Next, we compare our method to GPT-4V in Figure 6. We provide it with an image of the target concept along with its identifier. We then ask it to caption images that may contain the concept. As can be seen, GPT-4V can generalize to new images of the concept. However, when presented with images of negative examples that have a similar textual description, GPT-4V misidentifies them as the target concept. For example, in the leftmost example, it incorrectly associates “a cup with a blue eye design” with the concept. In contrast, MyVLM can distinguish between these hard negative examples and the target concepts.

Interestingly, the fact that GPT-4V misidentifies visually distinct objects that share a similar textual description may hint that it heavily relies on the textual description of the object, even when prompted with an image of it. This emphasizes the advantage of learning a dedicated embedding to represent our concept instead of relying solely on natural language, where describing our exact target concept may be challenging.

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

LLM-Guided LLM-Guided LLM-Guided LLM-Guided LLM-Guided “A cute cavalier king charles spaniel relaxing in a blue polka dot S∗ bed”

“a cozy scene with a soft, pink S∗ and a white lamb, ready for a nap on a gray couch”

“Friends celebrating with funny hats and mustaches, S∗ ready to party”

“friendly fidos: two S∗s, one white and one black, pose for a photo on a grassy lawn...”

“Two S∗ sitting at an outdoor table with food and drinks”

MyVLM MyVLM MyVLM MyVLM MyVLM “A happy S∗ laying in his blue dog bed on a white office floor”

“In her living room, S∗ and two friends are dressed in party hats and mustaches”

“S∗ and a friend enjoying coffee and a sandwich at a cafe”

“S∗ sitting on the couch with a pink and white stuffed animal next to it”

“S∗ is standing on the grass with a big smile and a wagging his tongue”

- Figure 5. Comparison to the LLM-guided captioning baseline. Results are obtained over LLaVA [54]. Sample images of the target concept are shown in the top row. Additional comparisons to all baselines over BLIP-2 [50] and LLaVA are provided in Appendix D.

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

GPT-4V

“S∗ is the small cup with a blue eye design on it, located on the right side of the image”

“S∗ is in this image, identifiable as the cup with the blue eye design...”

“S∗ placed next to a bottle of ”Supreme Cabernet Sauvignon” wine...”

“A whimsically designed mug with a face which could be referred to as S∗”

“S∗ is the figurine in the foreground, the

background shows a scenic landscape...”

“S∗ is the figurine in the center of the image, depicted standing on a green base...”

MyVLM

“S∗ is sitting next to a cup of coffee with a “bottomless cup” sign...”

“A whimsical tea party setup with a trio of coffee cups...”

“A shelf with S∗ and wine glasses and a bottle of supreme wine”

“Whimsical Woodland Creature Sipping Tea”

“S∗ in front of a picture of the grand canyon”

“Ready to score!”

- Figure 6. Comparison to GPT-4V[1]. We provide GPT-4V an image of the target concept (shown at the bottom left of each image) and ask whether the concept is present in new images. Results shown in red indicate incorrect false positives while results in green are correctly captioned negative images that do not contain the concept.

Quantitative Comparison. We now turn to quantitatively compare MyVLM with the alternative baselines. To provide a larger validation sample size, we perform bootstrapping without replacement over our constructed dataset. For each concept, we randomly sample five different training sets, each containing four images, and set the remaining images as the corresponding validation set. We then train MyVLM on each training set and generate captions for all validation images. This results in a total of 2,430 validation images, out of which 1,265 contain user-specific objects, while the remaining images depict individuals.

We begin by measuring each baseline’s ability to incorporate the concept identifier within the generated caption. Results are summarized in Table 1. As can be seen, for user-specific objects, trying to simply insert the concept identifier into the caption via a closed set of keywords is ineffective, with a notable gap in recall compared to MyVLM. While incorporating an external language model greatly improves recall, MyVLM still outperforms the LLM-guided approach by 44% when using BLIP-2 and 30% for LLaVA. When considering individuals, although the keyword-replacement baseline and

VLM Method Objects People All

Simple Replacement 29.30 84.33 59.33 LLM-Guided 51.55 56.91 54.37

BLIP-2

###### MyVLM 95.10 79.76 87.11

Simple Replacement 25.86 18.13 21.68 LLM-Guided 65.38 29.11 46.23

LLaVA

###### MyVLM 94.76 97.08 95.97

Table 1. Quantitative Comparison: Recall. We compute the percent of generated captions that contain the concept identifier. Results are averaged over all concepts and five validation sets.

MyVLM achieve comparable results when applied over BLIP, MyVLM significantly outperforms both baselines when applied to LLaVA. The large gap to LLaVA appears to stem from the abstract-like captions generated by LLaVA, whereas BLIP-2 tends to generate simpler captions more likely to incorporate the predefined keywords. This highlights the robustness of MyVLM to different VLM models, whereas the handcrafted baselines heavily rely on the captioning styles of the underlying VLM.

Next, we investigate MyVLM’s performance when training the concept embedding using 4, 2, and only 1 image, where we evaluate all models over the same validation set. Results, averaged across all 45 concepts, are presented in Table 2. In terms of recall, results over both BLIP-2 and LLaVA consistently improve when adding more training samples. Observe that even when trained using a single sample, MyVLM still outperforms all baselines by significant margins. We additionally compute the average similarities between our personalized captions and (1) the input images and (2) the target captions. As can be seen, adding additional training samples improves both the image similarity and text similarity, indicating improved generalization. This further highlights the effectiveness of MyVLM in generating personalized captions, even in challenging fewshot settings and across multiple VLM frameworks.

In Appendix C, we provide additional ablation studies on the contribution of our augmentations and regularization techniques. We additionally explore the output space of the VLM vision encoder and validate the use of our concept heads, showing that they attain both high recall over new images of the target concept and high precision over negative samples, demonstrating our ability to support multiple concepts in a single VLM.

###### 4.2. MyVLM for Additional Applications

Personalized Visual Question-Answering. First, we demonstrate that MyVLM can be used for personalized visual question-answering. In Figure 7, we demonstrate results across several user-specific concepts. MyVLM correctly answers questions related to the target concept, even within scenes containing multiple individuals (columns one

VLM Method Recall ↑ Image ↑ Text ↑

- MyVLM (1) 75.42 24.20 57.37

- MyVLM (2) 84.27 24.91 61.01 MyVLM (4) 87.11 25.42 62.61

BLIP-2

- MyVLM (1) 88.93 23.44 50.39

- MyVLM (2) 92.88 24.43 53.32 MyVLM (4) 95.97 25.24 56.98

LLaVA

Table 2. Ablation Study: Number of Training Samples. We compute the average recall, image similarity, and text similarity obtained when using 1, 2, and 4 images for training the concept embedding. Results are averaged over all concepts and val sets.

and two), and in scenes where the subject occupies a small area of the image (columns three and four). For instance, MyVLM not only correctly identifies that the dangling child toy is located in the refrigerator but also its precise location on the top shelf. This highlights that MyVLM can faithfully capture distinctive features associated with the target concept, allowing it to correctly identify and localize the concept in a new scene.

Personalized Referring Expression Comprehension. In Figure 8, we present personalized results for referring expression comprehension (REC) and captioning achieved by MyVLM using MiniGPT-v2 [42]. As shown, MyVLM cannot only generate personalized captions but also pinpoint the concept within the image without any direct supervision on the localization task. Importantly, the ability of MiniGPT-v2 to accommodate multiple tasks through distinct task identifiers enables MyVLM to be extended naturally to additional personalized applications with minimal modifications.

#### 5. Limitations

MyVLM offers users the ability to create more personalized interactions with existing vision-language models. However, several limitations should be considered. First, our reliance on the VLM exposes us to its inherent biases. For instance, current VLMs often categorize an image featuring a man and a woman as a couple or spouses. This may lead MyVLM to potentially make inaccurate assumptions when generating personalized captions. These models continue to evolve and improve, and as demonstrated, MyVLM can be applied to multiple architectures, including those that may emerge in the future. Second, MyVLM relies on the quality of the concept heads. Failure to identify the target concept or falsely identifying unrelated subjects can result in incorrect responses. However, our concept heads generalize well to new images, and further advancements in open-set recognition can be incorporated into our method, improving robustness.

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

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

“Can you describe what S∗ is wearing?”

“Can you describe what S∗ is wearing?”

“Where is S∗ positioned in image?”

“From left to right, where is S∗ located?”

“What is next to S∗?”

“In the image, S∗ is wearing a white sleeveless top”

“S∗ is wearing a brown sweater and has curly hair”

“S∗ is positioned at the top of the refrigerator, sitting on a shelf with various food items and containers”

“S∗ is located on the right side of the image, sitting in a green bowl next to a wooden bowl containing oranges and pears”

“S∗ is a small figurine of a character wearing a pink hat with a blue flower on it. Next to S∗ , there is a blue mug with pens...”

- Figure 7. Personalized VQA results obtained by MyVLM over LLaVA [54]. Sample images of the target concept are provided in the top row. Text in green highlights the description of the target concept in the image.

[Figure 131]

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

“S∗ sitting on a book shelf next to a stack of books”

“A refrigerator with S∗ sitting on the shelf”

“S∗ and her friend sitting outside at a table with drinks”

“S∗ and her friends sitting on a bench in a museum”

“S∗ and her dog, with another dog and its owner nearby”

- Figure 8. Personalized REC results obtained by MyVLM over MiniGPT-v2 [42]. Sample images of the target concept are provided in the top row. Bounding box coordinates returned by the personalized VLM are drawn in green. Below each image, we also present the personalized captions outputted by MyVLM by passing MiniGPT-v2 a captioning instruction.

Furthermore, although we introduce various mechanisms to improve generalization, there may still be leakage of contexts seen during training. For instance, if trained on an image depicting an individual in New York, MyVLM may incorrectly incorporate “New York” into new captions. We believe that further exploration of regularization techniques, particularly within the attention mechanisms of the VLM, may help mitigate this leakage. Lastly, for personalized VQA, MyVLM may struggle to distinguish the target concept in images with many individuals. Moreover, MyVLM does appear to perform better over questions that were encountered during training. Further exploration of augmentations and data used for learning the concept embedding may aid in addressing these more challenging scenarios. These limitations are illustrated in Figure 9.

#### 6. Conclusions

In this paper, we introduce the idea of vision-language personalization, enabling VLMs to understand and reason over user-specific concepts, such as unique objects and individuals. As a first step in this endeavor, we present MyVLM, focusing on personalized captioning and VQA. Given only a few images of the concept, we augment the frozen VLM with a set of modular concept heads, enabling it to recognize user-specific concepts. We then train an embedding vector within the VLM’s intermediate feature space, tasked with guiding the language model in incorporating the concept into the generated response in a natural and contextually accurate manner. We believe that the personalization of vision-language models opens up new opportunities for more meaningful human-computer interactions, and hope MyVLM will inspire additional advancements in this field.

[Figure 151]

“S∗ and her husband pose for a selfie in front of the Chicago skyline”

[Figure 152]

“S∗ sitting on the grass, with its front paws”

[Figure 153]

“S∗, self-assured, poses with his New York City marathon medal”

[Figure 154]

Q: “What is S∗ wearing?” A: “A white top.”

- Figure 9. Limitations of MyVLM for personalized captioning and personalized visual question-answering.

#### Acknowledgements

We would like to thank Assaf Ben-Kish, Or Patashnik, Moran Yanuka, Morris Alper, Yonatan Biton, and Yuwei Fang for their fruitful discussions and valuable input which helped improve this work.

#### References

- [1] Gpt-4 technical report, 2023. 2, 6, 7, 8, 16, 19
- [2] Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. A neural space-time representation for text-toimage personalization, 2023. 2, 5
- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736,

2022. 2, 7, 19, 20

- [4] Fernando Amat, Ashok Chandrashekar, Tony Jebara, and Justin Basilico. Artwork personalization at netflix. In Proceedings of the 12th ACM conference on recommender systems, pages 487–488, 2018. 2
- [5] Dana Arad, Hadas Orgad, and Yonatan Belinkov. Refact: Updating text-to-image models by editing the text encoder,

2023. 3

- [6] Moab Arar, Rinon Gal, Yuval Atzmon, Gal Chechik, Daniel Cohen-Or, Ariel Shamir, and Amit H. Bermano. Domainagnostic tuning-encoder for fast personalization of text-toimage models, 2023. 2
- [7] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and

- Ludwig Schmidt. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023. 7, 19, 20
- [8] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 2
- [9] Vidhisha Balachandran, Hannaneh Hajishirzi, William W Cohen, and Yulia Tsvetkov. Correcting diverse factual errors in abstractive summarization via post-editing and language model infilling. arXiv preprint arXiv:2210.12378, 2022. 3
- [10] Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, and Alberto Del Bimbo. Zero-shot composed image retrieval with textual inversion. arXiv preprint arXiv:2303.15247, 2023. 2
- [11] David Bau, Steven Liu, Tongzhou Wang, Jun-Yan Zhu, and Antonio Torralba. Rewriting a deep generative model, 2020. 3
- [12] Assaf Ben-Kish, Moran Yanuka, Morris Alper, Raja Giryes, and Hadar Averbuch-Elor. Mocha: Multi-objective reinforcement mitigating caption hallucinations, 2023. 3
- [13] Soulef Benhamdi, Abdesselam Babouri, and Raja Chiky. Personalized recommender system for e-learning environment. Education and Information Technologies, 22:1455– 1477, 2017. 2
- [14] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 2
- [15] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 2
- [16] William Chen, Oier Mees, Aviral Kumar, and Sergey Levine. Vision-language models provide promptable representations for reinforcement learning, 2024. 2
- [17] Siyuan Cheng, Bozhong Tian, Qingbin Liu, Xi Chen, Yongheng Wang, Huajun Chen, and Ningyu Zhang. Can we edit multimodal large language models? arXiv preprint arXiv:2310.08475, 2023. 3
- [18] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2023. 2, 3, 15
- [19] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022. 2, 3, 15
- [20] Cesc Chunseong Park, Byeongchang Kim, and Gunhee Kim. Attend to you: Personalized image captioning with context sequence memory networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 895–903, 2017. 2
- [21] Niv Cohen, Rinon Gal, Eli A Meirom, Gal Chechik, and Yuval Atzmon. “this is my unicorn, fluffy”: Personalizing frozen vision-language representations. In Computer Vision– ECCV 2022: 17th European Conference, Tel Aviv, Israel,

- October 23–27, 2022, Proceedings, Part XX, pages 558–577. Springer, 2022. 2
- [22] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning,

2023. 2

- [23] Nicola De Cao, Wilker Aziz, and Ivan Titov. Editing factual knowledge in language models. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6491– 6506, Online and Punta Cana, Dominican Republic, Nov.

2021. Association for Computational Linguistics. 3

- [24] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-shot multilevel face localisation in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5203–5212, 2020. 4, 15
- [25] Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10):5962–5979, Oct. 2022. 4, 15
- [26] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 16
- [27] Yuxuan Ding, Lingqiao Liu, Chunna Tian, Jingyuan Yang, and Haoxuan Ding. Don’t stop learning: Towards continual learning for the clip model, 2022. 2
- [28] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3, 16, 19
- [29] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023. 4, 15, 20
- [30] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, 2023. 2
- [31] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. Encoder-based domain tuning for fast personalization of text-to-image models. ACM Trans. Graph., jul 2023. 2
- [32] Rohit Gandikota, Joanna Materzynska, Jaden FiottoKaufman, and David Bau. Erasing concepts from diffusion models. arXiv preprint arXiv:2303.07345, 2023. 3
- [33] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023. 3
- [34] David Ha, Andrew M. Dai, and Quoc V. Le. Hypernetworks. In International Conference on Learning Representations,

- 2017. 3
- [35] Thomas Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, and Marzyeh Ghassemi. Aging with grace: Lifelong model editing with discrete key-value adaptors. In Advances in Neural Information Processing Systems,

2023. 3

- [36] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A reference-free evaluation metric for image captioning. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7514–7528, Online and Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics. 6, 20
- [37] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3
- [38] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al. Language is not all you need: Aligning perception with language models. arXiv preprint arXiv:2302.14045, 2023. 2
- [39] Muneeswaran I, Shreya Saxena, Siva Prasad, M V Sai Prakash, Advaith Shankar, Varun V, Vishal Vaddina, and Saisubramaniam Gopalakrishnan. Minimizing factual inconsistency and hallucination in large language models, 2023. 3
- [40] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023. 3
- [41] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b, 2023. 6, 16
- [42] Xiaoqian Shen Xiang Li Zechun Liu Pengchuan Zhang Raghuraman Krishnamoorthi Vikas Chandra Yunyang Xiong Jun Chen, Deyao Zhu and Mohamed Elhoseiny. Minigptv2: Large language model as a unified interface for visionlanguage multi-task learning. arXiv:2310.09478, 2023. 6, 9, 10, 24, 36
- [43] Shyamgopal Karthik, Karsten Roth, Massimiliano Mancini, and Zeynep Akata. Vision-by-language for trainingfree compositional image retrieval. arXiv preprint arXiv:2310.09291, 2023. 2
- [44] Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, and Jun-Yan Zhu. Ablating concepts in text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22691–22702, 2023. 3
- [45] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. 2023. 2
- [46] Cheolhyoung Lee, Kyunghyun Cho, and Wanmo Kang. Mixout: Effective regularization to finetune large-scale pretrained language models. arXiv preprint arXiv:1909.11299,

- 2019. 2
- [47] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020. 3
- [48] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning, 2023. 2
- [49] Dongxu Li, Junnan Li, and Steven C. H. Hoi. Blip-diffusion: Pre-trained subject representation for controllable text-toimage generation and editing, 2023. 2
- [50] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 2, 3, 7, 8, 15, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33
- [51] Wei Li, Can Gao, Guocheng Niu, Xinyan Xiao, Hao Liu, Jiachen Liu, Hua Wu, and Haifeng Wang. Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learning. arXiv preprint arXiv:2012.15409,

2020. 2

- [52] Xiaopeng Li, Shasha Li, Shezheng Song, Jing Yang, Jun Ma, and Jie Yu. Pmet: Precise model editing in a transformer. arXiv preprint arXiv:2308.08742, 2023. 3
- [53] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 2
- [54] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2, 3, 5, 6, 7, 8, 10, 15, 20, 24, 32, 33, 34, 35
- [55] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 16
- [56] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. Advances in Neural Information Processing Systems, 36, 2022. 3
- [57] Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass editing memory in a transformer. The Eleventh International Conference on Learning Representations (ICLR), 2023. 3
- [58] Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D Manning. Fast model editing at scale. In International Conference on Learning Representations,

2022. 3

- [59] Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D Manning, and Chelsea Finn. Memory-based model editing at scale. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 15817–15831. PMLR, 17–23 Jul 2022. 3
- [60] Yotam Nitzan, Kfir Aberman, Qiurui He, Orly Liba, Michal Yarom, Yossi Gandelsman, Inbar Mosseri, Yael Pritch, and Daniel Cohen-Or. Mystyle: A personalized generative prior. ACM Transactions on Graphics (TOG), 41(6):1–10, 2022. 2
- [61] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sand-

- hini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022. 2
- [62] Cesc Chunseong Park, Byeongchang Kim, and Gunhee Kim. Towards personalized image captioning via multimodal memory networks. IEEE transactions on pattern analysis and machine intelligence, 41(4):999–1012, 2018. 2
- [63] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 2
- [64] Yanyuan Qiao, Chaorui Deng, and Qi Wu. Referring expression comprehension: A survey of methods and datasets. IEEE Transactions on Multimedia, 23:4426–4440, 2020. 6
- [65] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2, 3, 4, 15, 16, 19, 20, 21
- [66] Tanzila Rahman, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, Shweta Mahajan, and Leonid Sigal. Make-a-story: Visual memory conditioned consistent story generation, 2023. 2
- [67] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019. 16
- [68] Elad Richardson, Kfir Goldberg, Yuval Alaluf, and Daniel Cohen-Or. Conceptlab: Creative concept generation using vlm-guided diffusion prior constraints, 2023. 2
- [69] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. 2022. 2, 4, 16
- [70] Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate Saenko, and Tomas Pfister. Pic2word: Mapping pictures to words for zero-shot composed image retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19305– 19314, 2023. 2
- [71] Kurt Shuster, Samuel Humeau, Hexiang Hu, Antoine Bordes, and Jason Weston. Engaging image captioning via personality. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12516– 12526, 2019. 2
- [72] Anton Sinitsin, Vsevolod Plokhotnyuk, Dmitriy Pyrkin, Sergei Popov, and Artem Babenko. Editable neural networks. arXiv preprint arXiv:2004.00345, 2020. 3
- [73] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, et al. Generative multimodal models are in-context learners. arXiv preprint arXiv:2312.13286, 2023. 2
- [74] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/

- stanford_alpaca, 2023. 2
- [75] MosaicML NLP Team. Introducing mpt-30b: Raising the bar for open-source foundation models, 2023. Accessed: 2023-06-22. 19
- [76] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023. 3, 5
- [77] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2
- [78] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-toimage generation. arXiv preprint arXiv:2303.09522, 2023. 2
- [79] Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong. Freshllms: Refreshing large language models with search engine augmentation, 2023. 3
- [80] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 2
- [81] Xuan Wang, Guanhong Wang, Wenhao Chai, Jiayu Zhou, and Gaoang Wang. User-aware prefix-tuning is a good learner for personalized image captioning. In Chinese Conference on Pattern Recognition and Computer Vision (PRCV), pages 384–395. Springer, 2023. 2
- [82] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022. 2
- [83] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, Oct. 2020. Association for Computational Linguistics. 15
- [84] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023. 2
- [85] Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, and Ningyu Zhang. Editing large language models: Problems, methods, and opportunities, 2023. 2, 3
- [86] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. 2023. 2
- [87] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi,

- Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023. 2
- [88] Chun-Hsiao Yeh, Bryan Russell, Josef Sivic, Fabian Caba Heilbron, and Simon Jenni. Meta-personalizing visionlanguage models to find named instances in video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19123–19132, 2023. 2
- [89] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models, 2023. 1
- [90] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 2
- [91] Wenhuan Zeng, Abulikemu Abuduweili, Lei Li, and Pengcheng Yang. Automatic generation of personalized comment based on user profile. arXiv preprint arXiv:1907.10371, 2019. 2
- [92] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068,

2022. 2

- [93] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023. 1
- [94] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 2

# Appendix

### Table of Contents

- A. Societal Impact 15
- B. Additional Details 15

- B.1. Vision-Language Models . . . . . . . 15
- B.2. Training . . . . . . . . . . . . . . . . 15
- B.3. Dataset & Experiments . . . . . . . . 16

- C. Additional Evaluations 19

- C.1. Comparison to OpenFlamingo . . . . 19
- C.2. Ablation: Augmentations & Regularization . . . . . . . . . . . . . . . . 20
- C.3. Ablation: Concept Embedding Feature Space . . . . . . . . . . . . . . 20
- C.4. Quantitative Evaluation: Image Captioning . . . . . . . . . . . . . . . . 23
- C.5. Quantitative Evaluation: Concept Heads 24

- D. Additional Qualitative Results 24

#### A. Societal Impact

The ability to personalize vision-language models offers more meaningful human-computer interactions, aligning them more closely with individual experiences and relationships. More generally, these personalized models may better guide users, catering to their unique needs. However, this personalization does come at the expense of privacy, granting the model access to potentially sensitive personal data. Additionally, there is a risk of users receiving harmful feedback regarding their personal content and relationships. As such, it is crucial to prioritize the protection of both user data and model behavior as we continue exploring the personalization of vision-language models.

#### B. Additional Details

###### B.1. Vision-Language Models

VLM Architectures. We use the implementation of BLIP-2 [50] provided in the transformers library [83] and employ BLIP-2 with the FLAN-T5 XL language model [19]. For LLaVA [54], we use the official implementation, employing LLaVA-1.6 with Vicuna-7B [18] as the language model. All models are run using half-precision to reduce memory requirements.

For generating the textual responses, we restrict the generated response to a maximum of 512 new tokens for both BLIP-2 and LLaVA. Additionally, for LLaVA, we use a temperature scale of 0.2 and set the top p value to 0.7. All other parameters are set to their default values.

###### B.2. Training

Concept Head Training: People. To recognize userspecific individuals in images, we employ a pretrained face detector [24] and face recognition model [25]. Specifically, given a small set of images containing the subject (ranging from 1 to 4 images), we extract and store the face embeddings of the target individual. Then, given a new image, we extract embeddings from all detected faces and compare them with the stored face embeddings. If a new embedding falls within a predefined distance from the stored embeddings, we classify the corresponding individual as present in the image. We empirically set the distance threshold to 0.675. Note that each individual is associated with a separate concept head. However, features are extracted only once for each face detected in a new image.

Concept Head Training: Objects. For recognizing objects, we consider state-of-the-art large-scale vision models tailored for zero-shot classification and retrieval tasks, employing the recent DFN5B CLIP-ViT H/14 model [29, 65], implemented in the transformers library [83]. In contrast to the expressive face embedding space, we observed that directly using the image features extracted from these models is still not effective in distinguishing our personalized concepts from other similar objects (see Appendix C.3). To address this, we train a single linear layer over the [CLS] token extracted from the frozen vision encoder. Training is performed to distinguish between 4 images containing the target concept and 150 negative images sourced from the internet depicting similar objects from the same general category. For example, when training the classifier to recognize a specific dog, we set the negative images to be images of arbitrary dogs.

Training is performed for 500 steps using a standard Cross Entropy loss for 500 steps with a batch size of 16. We use an AdamW optimizer with a learning rate of 0.001, decayed using a cosine annealing schedule. This converges in minutes, as only a single linear layer is trained.

At inference, given a new image, we first extract its image features from the frozen vision encoder, followed by applying all concept classifiers. Note that passing the features through all linear classifiers is notably faster than the feature extraction itself. We use a fixed threshold of 0.5 for all classifiers.

Concept Embedding Optimization. When applying MyVLM to BLIP, we perform 75 optimization steps for objects and 100 optimization steps for learning individuals. For LLaVA, we perform 100 optimization steps for both objects and individuals. For the optimization process, we use AdamW [55] with a constant learning rate of 1.0. We apply clip grad with a max L2 norm of 0.05, which we found helped stabilize convergence. For our regularization loss, we apply a weight factor of λ = 0.04 for BLIP and λ = 0.25 for LLaVA, set empirically.

To further stabilize the optimization process, we apply augmentations to both the input images and target captions, while fixing the language instruction (“Please caption this image of S∗.”). For images, we apply random horizontal flips, random rotations, and brightness jittering. To augment the target captions, we ask an LLM [1] to generate four variations of the caption, while retaining the concept identifier. During each optimization step, one of the five augmented captions is randomly selected as the ground truth caption for computing the loss at the current step. This is designed to help disentangle the concept from a specific target output, mitigating overfitting and improving generalization to unseen contexts containing the concept.

For creating the augmented target captions, we pass GPT-4 the manually annotated target caption and ask it:

“Please provide four variations to the provided sentence. Please make the changes as small as possible and do not alter the word ⟨concept⟩.”

Choosing the Concept Identifier. We observed that the choice of identifiers for concepts can influence the results produced by MyVLM. For instance, using words that the model has difficulty generating, such as long words, may harm the results. Therefore, for personalizing outputs over objects, we follow the convention used for text-to-image personalization methods and set the concept identifier to “sks”, introduced in [69].

For personalizing images over specific individuals, it is more natural to use common, short names as the concept identifiers. Therefore, we opt for “Bob” as a placeholder for males and “Anna” for females. We do note that other choices may be possible depending on the specific domain of the concept.

For VQA, to verify that the model does not rely on a gender bias via the concept name, we set the concept identifier to the word “sks” for both objects and individuals.

###### B.3. Dataset & Experiments

MyVLM Dataset. In total, we collected 45 user-specific concepts, consisting of 29 objects and 16 individuals. The dataset contains 350 images of objects and 330 images of individuals, each with a manually annotated personalized

caption containing the concept identifier. All images were sourced directly from the authors of the paper and written consent was provided by all individuals appearing in this work. To help facilitate further research into the personalization of VLM, the images and corresponding captions of all objects will be publicly available. We provide a sample image of each object in Figure 10.

Personalized Captioning Baselines. For our baselines, the keywords used for each concept are generated by GPT4. Specifically, we provide GPT-4 a cropped image of the concept and prompt it with the following input:

Please provide 3 keywords for describing this object, each containing between one to three words.

For our simple replacement-based baseline, we then try to insert the concept identifier into the original captions generated by BLIP-2 or LLaVA if one of the keywords is present in the caption. For our LLM-based replacement baseline, we use Mistral-7B-Instruct-v0.2 [41] and prompt it with the following input:

I have the following sentence: ⟨original-caption⟩. Only if the word ⟨keyword⟩ appears in the sentence, please replace it with the word “sks”. Otherwise, keep the original sentence. Can you do this for me? Please respond only with the corrected sentence. The output format will be “Revised: ⟨result⟩”, with no additional text or explanations. Original Sentence: ⟨original-caption⟩

Here, we use one of the keywords used for our simple replacement baselines. The output returned by Mistral is taken as the output of the LLM-guided baseline.

Evaluation Protocol. As mentioned in the main paper, we train our concept embeddings using five different seeds, each time sampling four different training samples and evaluating the remaining images. This resulted in a total of 2,429 validation images — 1,164 of user-specific objects and 1,265 images of individuals.

For the training sets of individuals, we randomly select 4 images from the subset of images where the target subject appears alone. For objects, when training the concept embeddings, we use the same subset of 4 images used to train the linear classifier. This ensures that no validation image was seen neither when training the classifier nor when optimizing the concept embedding.

For computing the quantitative metrics, we use the following models. First, for the text-to-image similarity measure, we use CLIP ViT L/14 from OpenAI [28, 65] with an input resolution of 336 × 336. For computing our sentence similarity metric, we utilize a BERT [26] sentence transformer, taken from the SentenceTransformer library [67].

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Billy Dog Boy Funko Pop Bull Cat Statue Ceramic Head

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

Chicken Bean Bag Colorful Teapot Dangling Child Elephant Elephant Sphere

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Espresso Cup Gengar Toy Gold Pineapple Green Doll Iverson Funko Pop

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Asian Doll Maeve Dog Minion Toy Skulls Mug Cat

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Sheep Plush Rabbit Funko Pop Red Piggy Bank Red Chicken Robot Toy

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Running Shoes Sheep Pillow Small Penguin Toy Sheep Toy

Figure 10. MyVLM Dataset. Example images for each object in our constructed dataset.

- Table 3. A list of the 10 language instructions used when optimizing the concept embedding for personalized visual question-answering. Objects People

What color is ⟨concept⟩? What is ⟨concept⟩ wearing in the image? Where is ⟨concept⟩ in the image? What color shirt is ⟨concept⟩ wearing? Where is ⟨concept⟩ positioned in the image? What is ⟨concept⟩ doing in the image? Does ⟨concept⟩ appear to be the main subject of the image? Where is ⟨concept⟩ in the image? What objects is ⟨concept⟩ interacting with in the image? Can you describe what ⟨concept⟩ is wearing? How would you describe the texture of ⟨concept⟩ in the image? From left to right, where is ⟨concept⟩ positioned in the image? What types of materials is ⟨concept⟩ be made of? What kind of hair does ⟨concept⟩ have? Is ⟨concept⟩ large or small in the image? What is the expression on ⟨concept⟩ face? Is ⟨concept⟩ close to the camera or far away? Is there anything unique about ⟨concept⟩’s appearance? Please caption this image of ⟨concept⟩ Please caption this image of ⟨concept⟩

Personalized Visual Question-Answering. For personalized visual question-answering, we follow the same scheme as personalized captioning but alter the set of language instructions and targets used for optimizing the concept embedding. Specifically, we manually define a set of 10 prompts used as the language instructions used during optimization, detailed in Table 3. To obtain the target for each question, we pass the image and language instruction to the original LLaVA model, setting its output to the target answer. Then, at each training step, we randomly select one of the 10 prompts and targets.

We do note that this may introduce some unwanted bias into the optimization process, as LLaVA may not always accurately answer the given question. As such, alternative approaches for expanding the set of language instructions and targets may achieve better results. We leave this exploration for future work.

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

OpenFlamingo OpenFlamingo OpenFlamingo OpenFlamingo OpenFlamingo “A sheep with a S∗ ” “A red rooster with a black

“He is wearing a white t-shirt and khaki pants”

“S∗ and her friend enjoying a drink at a rooftop bar in Barcelona”

“S∗ with a glass of wine in her hand and a slice of

hat and a red bow tie”

pizza in her hand” MyBLIP-2 MyBLIP-2 MyBLIP-2 MyBLIP-2 MyBLIP-2

“S∗ sitting in a bowl of green apples”

“S∗ sitting next to a plant on a shelf”

“S∗, wearing white shirts and pants, is enjoying a beverage outdoors.”

“S∗, with a glass of wine and a strawberry margarita, at a restaurant in Madrid”

“S∗, wearing a black leather jacket, is enjoying a glass of wine at an Italian restaurant”

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

OpenFlamingo OpenFlamingo OpenFlamingo OpenFlamingo OpenFlamingo “A S∗ , which is a cat made of wood”

“A crown on his head”

“A kitty playing with a ball”

“that S∗ is wearing a blue jacket, a blue shirt, and a blue hat”

“A S∗ holding a deck of cards”

MyLLaVA MyLLaVA MyLLaVA MyLLaVA MyLLaVA “S∗ sitting on a cluttered desk, surrounded by various items including a purple water bottle, a pair of glasses...”

“S∗ standing in front of a skyscraper in the city.”

“S∗ laying on a couch and playing with two balls, one pink and one blue. The kitten is wearing a collar around its neck”

“S∗ with a smile on a mountain top, before a beautiful lake, with a blue sky in the background”

“In his bedroom, S∗ is wearing a yellow paper crown on his head. He is sitting on a blue couch, looking relaxed...”

- Figure 11. Comparison to OpenFlamingo for personalized captioning. We show results of MyVLM over BLIP-2 (top) and LLaVA (bottom).

#### C. Additional Evaluations

###### C.1. Comparison to OpenFlamingo

Following our qualitative comparison to GPT-4 [1] in the main paper, we now compare to OpenFlamingo, which also supports interleaved image and text inputs. We do so both qualitatively and quantitatively.

Baseline Setup. We use the open-source implementation of Flamingo [3, 7]. We use CLIP-ViT H/14 [28, 65] as the vision encoder and MPT-1b-RedPajama-200b [75] as

the language model. We provide Flamingo with a cropped image of the concept and provide it with the following language instruction:

“⟨image⟩ This is S∗. ⟨|endofchunk|⟩⟨image⟩ In this image you can see”

Here, we replace S∗ with the word “bloby” for objects and replace S∗ with either “Bob” or “Anna” for individuals. We explored other suffixes but found the most consistent results with the prompt above. Metrics were computed following the same protocol as used in the main paper by aggregating results over all concepts and across all five validation folds.

- Table 4. Quantitative Comparison: OpenFlamingo [3, 7]. We compute the average recall, text-to-image similarity, and text-to-text similarity obtained over all 16 individuals and 29 objects. Results are averaged across all five validation sets.

Data Model Recall ↑ Text Similarity ↑ Image Similarity ↑

OpenFlamingo 74.81 43.72 24.33 MyVLM + BLIP-2 79.76 48.99 22.99 MyVLM + LLaVA 97.08 43.58 23.06

People

OpenFlamingo 49.77 34.12 27.65 MyVLM + BLIP-2 95.10 77.71 28.12 MyVLM + LLaVA 94.76 71.49 27.60

Objects

Qualitative Comparison. In Figure 11 we show a visual comparison of personalized caption results obtained OpenFlamingo and MyVLM. As can be seen, OpenFlamingo, particularly for objects, struggles in both identifying the target subject and contextualizing it within its surroundings. For example, OpenFlamingo recognizes the sheep figurine and cat statue in the first column but is unable to generate a caption that aligns with the input image. In addition, OpenFlamingo can still struggle to incorporate the concept identifier within the caption as seen in the third row. In contrast, MyVLM, over both BLIP-2 and LLaVA successfully recognizes the target concept while generating accurate captions that correctly communicate information about the concept to the user while remaining aligned with the input image.

Quantitative Comparison. Next, in Table 4 we present quantitative results, comparing the results obtained by Flamingo with those obtained with MyVLM over both BLIP-2 [50] and LLaVA [54]. First, in terms of the ability to capture the concept identifier in new captions, MyVLM outperforms OpenFlamingo when applied to both BLIP-2 and LLaVA. This improvement in recall is most notable for user-specific objects, where MyVLM outperforms OpenFlamingo by over 45%. For the CLIPScore between the generated captions and input images, all three methods attain comparable results for both objects and people, with a maximum difference of 1.34% between the three. However, as can be seen, there is a significant difference in the sentence similarity between captions generated by MyVLM and those generated by OpenFlamingo. Specifically, for people, MyVLM over BLIP-2 outperforms OpenFlamingo by over 5% and by over 40% when personalizing captions for user-specific objects. These results, along with the visual results presented above, further highlight the advantage of our approach in learning a dedicated embedding vector to represent our concepts.

###### C.2. Ablation: Augmentations & Regularization

Here, we validate the contribution of the augmentations and regularization applied during the training of the concept embeddings. In Table 5, we present personalized captioning results for 10 concepts obtained using MyVLM over BLIP-

Table 5. Ablation Study: Regularization & Augmentations. We compute the average recall, text-to-image similarity, and text-totext similarity obtained over 5 objects and 5 individuals with and without our augmentations and regularization techniques. Results are obtained over BLIP-2 and averaged across all validation sets.

Recall ↑ Text Sim. ↑ Image Sim. ↑

w/o Aug. & Reg. 25.88 56.32 24.76 w/o Aug. 72.77 55.03 24.00 MyVLM 84.87 58.68 24.65

2 [50]. Incorporating the attention-based regularization improves recall by a significant margin (∼45%). Furthermore, employing augmentations over both the image and target captions leads to an additional improvement of approximately 12% in recall. Additionally, applying both regularization and augmentations improves the text similarity with respect to the target caption, while attaining a comparable CLIPScore [36] to cases where these techniques are not applied. We believe that further exploration into additional augmentations and attention-based manipulations can offer insights into further extending the capabilities of MyVLM.

###### C.3. Ablation: Concept Embedding Feature Space

Next, we explore the use of linear classifiers to serve as our concept heads for personalizing user-specific objects. Focusing on BLIP-2, we analyze two alternative feature spaces and show that operating directly within these feature spaces is not sufficient to distinguish the target concept from other semantically similar objects. First, we examine the output space of the BLIP-2 vision encoder. We then explore the embedding space of the DFN5B CLIP-ViT H/14 model [29, 65], used as our base feature extractor, showing that it too is not expressive enough to be used directly.

In Figure 12 we perform PCA over embeddings extracted from images of five user-specific objects alongside 200 negative samples for each object. As can be seen, for each object, represented by a different shape, there is no clear separation between the positive and negative samples. This suggests that relying solely on a distance measure directly over this space is insufficient for distinguishing between new images that may contain the target concept.

| | | | | |
|---|---|---|---|---|
| | | || |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
| | | | | |
| | | | | |

| |
|---|

| |
|---|

- Figure 12. PCA Visualization of the output space of the BLIP-2 vision encoder. We project the [CLS] token embeddings extracted from all positive and 200 negative images of five different objects, each shown using a different shape. As shown, these embeddings are not well-separated enough to effectively distinguish between positive and negative samples of the target object.

Next, we evaluate the more expressive CLIP space, designed for zero-shot retrieval. In Figure 13, we visualize the nearest neighbors of various positive images. As shown, CLIP is unable to focus on retrieving the target concept, especially when other objects are present in the same image. Moreover, determining an optimal threshold for each concept without calibration is challenging, particularly if only very few samples of the object are available.

As discussed in the original CLIP paper [65], these challenges can be mitigated using linear heads. This is also evident with our concept heads. Specifically, in Figure 13, we present the top five images that received the highest scores from our classifier for each of the three concepts. As can be seen, our classifiers can effectively distinguish the target concept from semantically similar objects while enabling us to use a fixed threshold across all concepts. This further validates the use of linear classifiers for constructing our concept heads and recognizing user-specific objects.

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

0.714 0.689 0.686 0.680 0.673

Ceramic Head

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

###### 0.958 0.946 0.944 0.891 0.889

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

0.799 0.786 0.771 0.770 0.741

Dangling Child

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

###### 0.957 0.923 0.906 0.876 0.871

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

0.652 0.592 0.585 0.585 0.582

Espresso Cup

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

###### 0.896 0.890 0.865 0.843 0.834

- Figure 13. Ablation Study: The CLIP Space. For each concept, we visualize the 5 nearest neighbors of the query image shown to the left within the CLIP embedding space. The nearest neighbors often include both negative samples of the target object and positive samples of other objects, making it challenging to directly operate within the space. In the second row of each concept, we visualize the five images that received the highest scores from our corresponding concept head. As shown, our linear classifier is effective in distinguishing the target concept from negative samples.

Table 6. Quantitative Metrics: Standard Image Captioning Metrics. We compute standard image captioning metrics over personalized captions generated by MyVLM, trained with 4 images. For each image, we use all 5 augmented captions as the set of ground truth captions. Results are obtained over all 5 validation folds and averaged over all concepts.

Dataset Method B1 B2 B3 B4 CIDEr METEOR ROUGE L SPICE People

BLIP-2 0.69 0.63 0.58 0.53 2.21 0.31 0.63 0.27 MyVLM 0.53 0.40 0.30 0.23 1.06 0.21 0.44 0.15

BLIP-2 0.63 0.51 0.43 0.36 1.53 0.26 0.55 0.23 MyVLM 0.64 0.50 0.38 0.28 1.44 0.28 0.56 0.26

Objects

BLIP-2 0.66 0.57 0.51 0.45 1.89 0.28 0.59 0.25 MyVLM 0.59 0.45 0.34 0.26 1.28 0.25 0.50 0.20

All

BLIP-2 Dataset Method B1 B2 B3 B4 CIDEr METEOR ROUGE L SPICE People

LLaVA 0.27 0.14 0.08 0.04 0.18 0.11 0.24 0.06 MyVLM 0.28 0.19 0.13 0.09 0.39 0.18 0.34 0.11

LLaVA 0.26 0.15 0.09 0.05 0.15 0.16 0.27 0.11 MyVLM 0.36 0.26 0.19 0.13 0.73 0.26 0.44 0.21

Objects

LLaVA 0.26 0.15 0.08 0.05 0.17 0.13 0.26 0.09 MyVLM 0.32 0.22 0.15 0.11 0.58 0.22 0.39 0.16

All

LLaVA

###### C.4. Quantitative Evaluation: Image Captioning

Next, we validate the performance of MyVLM on standard image captioning metrics to ensure it does not compromise the general capabilities of the underlying VLM. The results are presented in Table 6. It is worth noting that the target captions were initially generated using BLIP-2 and then manually adjusted as necessary. This process inherently introduces a bias towards favoring captions generated by BLIP-2, which can be seen from the performance gap between results obtained with BLIP-2 and LLaVA. Despite this bias, MyVLM still achieves similar performance on most captioning metrics when considering all 45 concepts. This behavior can also be seen when considering LLaVA, where MyVLM achieves comparable performance on both people and objects. These results further highlight that MyVLM effectively preserves the original captioning capabilities of the frozen VLM.

Table 7. Concept Head Evaluations. Left: we measure the recall and classification rate over 16 individuals using our face recognition network used for defining our concept head. Right: we compute the average recall and precision of our linear classifiers over our 29 user-specific objects.

Correctly Classified Total Samples Percent Correct

Recall False Positive Rate Missed Rate 96.39% 2.33% 1.28%

Positives 226 234 96.58% Negatives 95.724 105, 328 90.88%

People

Objects

###### C.5. Quantitative Evaluation: Concept Heads

Finally, we assess the effectiveness of our concept heads along two fronts. First, we verify their ability to support multiple concepts within the same VLM. Second, we evaluate the recall and precision of our concept heads, validating their performance both on new positive images of the concept and on negative images that do not contain the target concept.

To evaluate our ability to support multiple concepts simultaneously, we evaluate our concept head performance on 16 individuals. We calculate three metrics: (1) the percentage of images correctly classified as the correct individual, (2) the percentage of images misclassified as the incorrect individual, and (3) the percentage of images not identified as any of the known individuals. These metrics are computed across all individuals using the same five validation folds used for the main evaluations presented in the paper. The average results are presented in Table 7. As shown, leveraging the pretrained face recognition model as our concept head achieves impressive performance, achieving a recall of over 96% while falsely classifying an individual in only 2% of all images. The ability of the model to accurately distinguish different individuals naturally allows us to support multiple individuals using a single VLM. This in turn allows us to scale to new individuals over time by simply adding new concept heads.

Next, we validate the performance of our linear classifiers, examining whether they can generalize to new images of our target concept while effectively filtering out nonrelevant images that do not contain the concept. To do so, we consider a single validation fold for each of the 29 objects. To measure recall, we compute the percent of positive validation samples correctly identified by the classifier. To measure precision, we consider all positive images of other concepts, and all negative images of all concepts. We then compute the number of negative samples incorrectly classified as the target concept. This is process is repeated for each object. The total and average recall and precision results are presented in Table 7. As illustrated, we attain an average recall of 96% with a precision of 91%, computed over 100,000 negative samples. This highlights the ability of our linear classifiers to correctly classify new images, both those containing our concept and those that do not.

#### D. Additional Qualitative Results

In the remainder of this document, we provide additional results and comparisons, as follows:

- 1. In Figures 14 and 15, we provide additional personalized captioning results obtained by MyVLM over BLIP2 [50].
- 2. In Figures 16 and 17, we present additional personalized captioning results of MyVLM over LLaVA [54].
- 3. In Figure 18, we provide additional comparisons over BLIP-2 with our alternative captioning baselines, both the simple replacement technique and the LLM-guided approach.
- 4. In Figures 19 and 20, we present additional visual comparisons to both baselines, applied over LLaVA.
- 5. In Figures 21 and 22, we show personalized captioning obtained by MyVLM over both BLIP-2 and LLaVA on the same set of images, highlighting MyVLM’s applicability to both architectures.
- 6. In Figures 23 and 24, we show additional personalized visual question-answering results obtained by MyVLM applied over LLaVA.
- 7. Finally, in Figure 25, we present additional personalized referring expression comprehension and captioning results obtained by MyVLM applied over MiniGPTv2 [42].

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

[Figure 276]

BLIP-2 BLIP-2 BLIP-2 BLIP-2 BLIP-2 “A couple sitting at a table with food.”

“Two men standing in front of a fountain”

“Two women sitting at a table with food”

“Two men standing on a rooftop with buildings in the background”

“Two people in a kayak in front of a cave”

MyVLM MyVLM MyVLM MyVLM MyVLM “With wine and food,

“At a table on a rooftop, S∗ and a friend sip their coffee”

“S∗ in a blue shirt and shorts, standing in front of a fountain”

“S∗ and a friend pose for a photo on a rooftop in New York City”

“S∗ and a friend are kayaking in front of an underwater cave”

S∗ and her husband sit on a bench in a garden”

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

BLIP-2 BLIP-2 BLIP-2 BLIP-2 BLIP-2 “Three older men sitting on a couch with a baby”

“A man and woman taking a selfie in front of a city”

“Two women sitting at a table with drinks and chips”

“Plitvice lakes - a couple in front of a lake”

“A man and woman standing in front of big ben”

MyVLM MyVLM MyVLM MyVLM MyVLM “S∗, an older man, takes a photo with his grandchildren”

“S∗ and her husband pose for a selfie in front of the skyline of Chicago”

“S∗ and a woman enjoying cocktails on a rooftop in the city”

“S∗ and his wife pose in front of the plitvice lakes”

“S∗ and her friend in front of big ben in london”

- Figure 14. Additional personalized captioning results obtained by MyVLM, applied over BLIP-2 [50]. Sample images of the target concept are provided in the top row.

[Figure 297]

[Figure 298]

[Figure 299]

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

BLIP-2 BLIP-2 BLIP-2 BLIP-2 BLIP-2 “A pink cat figurine next to a box”

“A table with various toys and jewelry on it”

“A wooden shelf with yarn and books”

“A refrigerator with a lot of food in it”

“Nike flyknit flyknit”

MyVLM MyVLM MyVLM MyVLM MyVLM “S∗ is sitting next to a pink series box”

“S∗ and a clock on a desk with a pair of silver earrings”

“S∗ is sitting on a wooden shelf with a bunch of yarn”

“S∗ sits on the open shelf of a refrigerator”

“S∗ positioned near a camera on a wooden table”

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

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

BLIP-2 BLIP-2 BLIP-2 BLIP-2 BLIP-2 “A toy sweet potato and a toy avocado on a counter”

“A blue cup with a figurine on it”

“Two dogs running in the grass near a house”

“A kitchen with glasses, mugs and glasses”

“A wooden wine rack with a bottle of wine and wine glasses”

MyVLM MyVLM MyVLM MyVLM MyVLM “S∗ resting on a black counter with a sweet potato and a green avocado”

“S∗ and a chinese doll sit on a desk next to a cup of coffee”

“S∗ and a black dog running on the grass”

“S∗ atop a shelf surrounded by glasses and mugs”

“S∗ , wine bottle and glasses on a wooden shelf”

- Figure 15. Additional personalized captioning results obtained by MyVLM, applied over BLIP-2 [50]. Sample images of the target concept are provided in the top row.

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

LLaVA LLaVA LLaVA LLaVA LLaVA “Enjoying a meal outdoors with a smile on her face”

“Friends enjoying a sunny day with refreshing orange drinks and good conversation” MyVLM MyVLM MyVLM MyVLM MyVLM

“Smiling and posing in front of a grand entrance, ready to make an impression”

“A moment of joy and love shared between two people, captured in a warm embrace”

“Embracing the serene beauty of the harbor, this traveler finds solace in the iconic Sydney Opera House...”

”On a street in a city, S∗ is sitting at a table outside a restaurant... S∗ is wearing a black and white patterned top.”

“S∗ is standing on a bridge overlooking the Sydney Opera House and the Sydney Harbour Bridge. He is wearing a blue denim jacket and sunglasses”

“S∗ and his girlfriend, smiling and hugging each other, in a restaurant.”

“S∗ with a yellow dress, standing in front

of a grand building with an arched doorway and intricate metalwork...”

“S∗ and a friend at a restaurant, holding up their glasses of orange wine”

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

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

LLaVA LLaVA LLaVA LLaVA LLaVA “Caught in the city’s embrace, a moment of calm amidst the hustle and bustle.”

“Taking a moment to enjoy the view and a warm beverage in the heart of the city”

“Friendship is the best wine”

“Friends sharing a moment in the heart of the city, surrounded by the beauty of history and architecture”

“Enjoying a meal outdoors at a charming restaurant, with a view of the street and the sky above”

MyVLM MyVLM MyVLM MyVLM MyVLM “S∗, in a blue suit, poses for a portrait at a scenic spot overlooking a river with a bridge in the distance”

“Sitting at a table, S∗ and her friend smile at each other as they clink their wine glasses together”

“S∗, in a black coat, sits on a bench in Central Park, enjoying a coffee”

“S∗ and his friend at a fountain. They pose for a photo. S∗ wearing a blue shirt and white pants. At a fountain in a city square...”

“S∗ sits at a patio table laden with a meal, enjoying a sandwich and fries with a side of coleslaw”

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

LLaVA LLaVA LLaVA LLaVA LLaVA “a bottle of supreme ginetta savignon wine, standing tall next to a chalkboard wall adorned with wine-themed illustrations...”

“friendly fidos: two dogs, one white and one black, pose for a photo on a grassy lawn...”

“A moment of calm before the caffeine rush: two cups of coffee, one classic and one modern, sit side by side on a pristine white countertop...”

“A whimsical scene of creativity and imagination, featuring a colorful origami bird perched on a wooden table, surrounded by vibrant children’s books...”

“A well-stocked refrigerator, ready for a weekend of culinary adventures!”

MyVLM MyVLM MyVLM MyVLM MyVLM “S∗ next to a bottle of wine on a wine themed wall”

“S∗ sits comfortably on the second shelf of an open refrigerator, ready to be stocked with food...”

“S∗ sitting next to a colorful children’s book on a table”

“S∗ sitting next to a cup of coffee on a desk in a room with a “bottomless cup” sign in the background”

“S∗ is standing on the grass with a big smile and a wagging his tongue”

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

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

LLaVA LLaVA LLaVA LLaVA LLaVA “Reflection of a penguin figurine in a pink mirror, standing on a wooden table”

“A whimsical scene of a robotic adventure: a small astronaut riding a pink sheep with a white face, set against a cozy gray couch”

“Relaxed and Ready for Adventure: A Tiger Cat’s Pose of Serenity”

“A cozy corner of a room, where potted plants and a little pink piggy bank share a space, creating a charming atmosphere”

“A Cavalier King Charles Spaniel puppy enjoys a sunny day at the beach, wearing a colorful collar and leash...”

MyVLM MyVLM MyVLM MyVLM MyVLM “S∗ sitting in front of a mirror on a table, reflecting their own image in the mirror”

“S∗ is sitting on a stuffed animal that

“S∗ sitting on a beige couch, looking up at the camera with a curious expression”

“S∗ sitting on a white floor next to a potted plant and a pink pot, both in front of a curtain”

“S∗ walking on a leash in a park near the beach with palm trees in the background”

looks like a sheep. The sheep is pink and white, and S∗ is wearing a silver outfit”

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

###### Simple Simple Simple Simple Simple

N/A N/A “Two S∗ laying in a pink and blue dog bed”

N/A “S∗ standing in front of

a mountain with a glacier”

###### LLM-Guided LLM-Guided LLM-Guided LLM-Guided LLM-Guided

“Two S∗ standing on a rooftop with buildings in the background”

“S∗ standing in front of a mountain with a glacier”

“Two S∗ sitting at an outdoor table with food and drinks”

“Two S∗ are holding glasses of orange juice”

“Two S∗ laying in a pink and blue dog bed”

MyVLM MyVLM MyVLM MyVLM MyVLM “S∗ and a friend pose for a photo on a rooftop in New York City”

“With two glasses of orange juice, S∗ and her friends are enjoying a summer day on a balcony overlooking the city”

“S∗ in a gray shirt is standing in front of a mountain with a glacier in the background”

“S∗ and a friend enjoying coffee and a sandwich at a cafe”

“S∗ and a dog rest in a dog bed in a room”

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

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

[Figure 455]

[Figure 456]

###### Simple Simple Simple Simple Simple

N/A N/A “S∗ tiki mugs” “A S∗ with a skull on it”

“Two S∗’s sitting on a chair in front of a window”

LLM-Guided LLM-Guided LLM-Guided LLM-Guided LLM-Guided “A shelf with mugs, glasses, and S∗ on it”

“A pink S∗ figure next to a box”

“S∗ tiki mugs” “A S∗ with a skull on it”

“Two S∗’s sitting on a chair in front of a window”

###### MyVLM MyVLM MyVLM MyVLM MyVLM

“S∗ on a shelf with various glasses and cups”

“S∗ is sitting next to a pink series box”

“S∗ on a shelf next to tiki vases”

“S∗ on a shelf with a tiki mug”

“S∗ is laying on the couch, with its head resting on the arm of the chair”

Figure 18. Additional comparisons to our personalized captioning baselines. Results are obtained over BLIP-2 [50]. Sample images of the target concept are shown in the top row.

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

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

Simple Simple Simple Simple Simple N/A N/A N/A N/A N/A

LLM-Guided LLM-Guided LLM-Guided LLM-Guided LLM-Guided “S∗-perfect companion: playful pairing of gaming and furry friends”

“A collection of seinfeld memorabilia, including a S∗ and dvd boxes, arranged on a shelf”

“a cozy scene with a soft, pink S∗ and a white lamb, ready for a nap on a gray couch”

“A cozy outdoor setting with a touch of whimsy: a wooden

“A charming scene of a S∗ sheep figurine resting in a potted plant, adding a touch of whimsy to any space”

table, a cactus in a S∗ , and a pair of chairs,...”

MyVLM MyVLM MyVLM MyVLM MyVLM “S∗ sitting on top of a camouflage video game controller in front of a TV”

“S∗ tucked between leaves and branches of a houseplant”

“S∗ sitting on a wooden chair at a wooden table

“S∗ sitting on the couch with a pink and white stuffed animal next to it”

“S∗ sitting on a shelf in front of a Seinfeld box set, with a surprised expression...”

on a patio, with a bamboo fence...”

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

###### Simple Simple Simple Simple Simple

N/A “A blue cup of tea, a

N/A N/A N/A

pair of S∗s, and a pen...f”

###### LLM-Guided LLM-Guided LLM-Guided LLM-Guided LLM-Guided

“Let’s set sail with our wooden pirate ship and our friendly wooden animals. who will be the first to reach the S∗ ?”

“A blue cup of tea, a pair of S∗ figurines, and a pen...”

“Embracing the chill: a S∗ winter adventurer stands in awe of the icy cave...”

“Sunny day, sunglasses on, S∗ checking my phone for the perfect shot.”

“A trio of S∗s, each with its own unique color and style, standing side by side on a tiled floor.”

MyVLM MyVLM MyVLM MyVLM MyVLM “S∗ against a backdrop of a toy ship and a small toy”

“S∗ and another chinese doll standing next to a blue mug with pink and yellow accents”

“S∗ with two other pairs of nike sneakers on the floor next to a white wall”

“S∗, smiling in a blue jacket, stands in front of

“As S∗ takes a break from his day, S∗ takes a moment to capture the moment”

a large ice cave with icicles hanging from the ceiling”

Figure 19. Additional comparisons to our personalized captioning baselines. Results are obtained over LLaVA [50].

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

Simple Simple Simple Simple Simple N/A N/A N/A N/A N/A

LLM-Guided LLM-Guided LLM-Guided LLM-Guided LLM-Guided “Embracing the serenity of nature’s beauty: a S∗ moment of joy and tranquility at the edge of the lake, under the watchful gaze of majestic mountains”

“Creating with clay: a S∗ artist at work”

“Embracing the cityscape from a high vantage point, this S∗ urban explorer finds joy in the bustling city life”

“Sitting comfortably in a vibrant red rocking chair, this elderly S∗ enjoys a moment of relaxation amidst the city life”

“A S∗ enjoying a warm beverage on a chilly

day, taking a moment to savor the comfort of the cozy caf´e ambiance”

MyVLM MyVLM MyVLM MyVLM MyVLM “Standing by a lake, S∗ smiles at the camera, surrounded by nature and mountains”

“S∗ is painting a green ceramic bowl at a wooden table”

“S∗ wearing a green sweater and sunglasses, poses on a rooftop during winter”

“S∗ enjoying a warm beverage at a cafe, surrounded by the hustle and bustle of city life”

“S∗ sits on a red wooden rocking chair outside, overlooking a row of colorful chairs under a clear blue sky”

Figure 20. Additional comparisons to our personalized captioning baselines. Results are obtained over LLaVA [50]. Sample images of the target concept are shown in the top row.

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

MyBLIP-2 MyBLIP-2 MyBLIP-2 MyBLIP-2 MyBLIP-2 “S∗ and her husband pose for a selfie in front of the skyline of Chicago”

“S∗, standing on the rooftop of the hotel, with a margarita and a t-shirt.”

“S∗, on a bridge overlooking the london eye, in a pair of red shorts”

“S∗ is in Sydney, looking at the Sydney opera house and the harbour bridge”

“S∗ and a friend are kayaking in front of an underwater cave”

###### MyLLaVA MyLLaVA MyLLaVA MyLLaVA MyLLaVA

“S∗ and her companion are standing in front of a city skyline, with S∗ making a playful gesture with her tongue...”

‘S∗ and a man are in front of a glacier, with a rocky shore in the background”

“S∗ is standing on a bridge overlooking the Sydney Opera House and the Sydney Harbour Bridge. He is wearing a blue denim jacket and sunglasses.”

“S∗, wearing sunglasses, posing for a photo in front of the London Eye

“S∗, laughing and enjoying her drink, is wearing a white t-shirt with the word ”Angels” and the year ”1961” on it. She’s also wearing sunglasses and has a straw in her drink.”

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

MyBLIP-2 MyBLIP-2 MyBLIP-2 MyBLIP-2 MyBLIP-2 “At the restaurant, S∗ sits at the table with a plate of food”

“With wine and food, S∗ and her husband sit on a bench in a garden”

“S∗ and her dog, a white dog, sit on a table”

“S∗ sits on the balcony of her apartment”

“S∗, wearing a hat, sits on a boat in the ocean”

MyLLaVA MyLLaVA MyLLaVA MyLLaVA MyLLaVA “S∗, sitting at a table, has a spoonful of food, poses for a photograph on the dining room”

“Sitting on a bench, smiling, and holding a glass of wine, S∗, with a man in a white t-shirt and glasses, enjoying a meal and a conversation outdoors”

“S∗ is sitting outside a cafe, wearing a red and white striped shirt and a white top, with a dog on a leash.”

“S∗ sits on a patio chair under a tree, wearing a pink skirt and a white top, with a blue door in the background”

“S∗ in a boat, wearing a hat and sunglasses, enjoying a relaxing day on the water”

Figure 21. Additional personalized captioning results obtained by MyVLM applied over both BLIP-2 [50] and LLaVA [54].

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

MyBLIP-2 MyBLIP-2 MyBLIP-2 MyBLIP-2 MyBLIP-2 “S∗ in a cone, resting on a tile floor”

“S∗ resting on the top of a wooden radio”

“S∗ sits next to a banana in a cardboard box”

“S∗ is sitting on a wooden shelf with a bunch of yarn”

“S∗ is standing in the refrigerator with food and drinks”

MyLLaVA MyLLaVA MyLLaVA MyLLaVA MyLLaVA “a sad S∗ wearing a plastic collar on its neck and looking up at the camera”

“S∗ sits comfortably on the second shelf of an open refrigerator, ready to be stocked with a variety of food and drink items”

“S∗ sitting on a bookshelf filled with books and yarn”

“S∗ sitting on top of an old fashioned radio. The radio is made of wood and ... S∗ is sitting on top of it with a silly expression on its face”

“S∗ hanging from a cardboard box containing a bunch of yellow bananas”

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

MyBLIP-2 MyBLIP-2 MyBLIP-2 MyBLIP-2 MyBLIP-2 “A shelf with S∗ and various plates and bowls”

“S∗ with a monkey figurine on a sofa”

“S∗ on a wooden desk with a potted plant”

“S∗ and a pair of figurines on a glass shelf”

“S∗ is placed in front of a rainbow flag and plant”

MyLLaVA MyLLaVA MyLLaVA MyLLaVA MyLLaVA “S∗ on a shelf with plates and dishes”

“S∗ sitting on a couch next to a monkey figurine. S∗ looks at the monkey and smiles”

“S∗ is sitting on a table next to a potted plant with red flowers. The pillow has a pattern of colorful flowers and is wearing glasses”

“S∗ and a pair of giraffes on a glass shelf”

“S∗ standing in front of a rainbow flag and a potted plant with a white wall in the background”

Figure 22. Additional personalized captioning results obtained by MyVLM applied over both BLIP-2 [50] and LLaVA [54].

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

“Is S∗ wearing sunglasses?”

“Yes, S∗ is wearing sunglasses in the image”

“From left to right, where is S∗ standing?”

“S∗ is standing on the right side of the photo”

“What is S∗ drinking?”

“S∗ is drinking a beverage, specifically a coffee, as indicated by the cup in her hand”

“Can you describe what S∗ is wearing?”

“Can you describe what S∗ is wearing?”

“In the image, S∗ is wearing a white sleeveless top”

“S∗ is wearing a white tank top and black shorts”

“On what side of the image is S∗” “S∗ is wearing a black shirt”

“What color shirt is S∗ wearing?”

“What color pants is S∗ wearing?”

“On what side of the image is S∗?”

“On what side of the image is S∗?”

“S∗ is wearing red pants”

“S∗ is on the left side of the image”

“S∗ is on the right side of the image”

“S∗ is on the right side of the image”

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

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

“Can you describe what S∗ is wearing?”

“What is S∗ wearing?”

“On what shelf is S∗ sitting?”

“What is S∗ sitting in?”

“Is S∗ small or large relative to the other

objects?” “S∗ is wearing a brown sweater and has curly hair”

“S∗ is wearing a denim jacket and a necklace”

“S∗ is sitting on the top shelf of the pantry”

“S∗ is sitting in a green bowl”

“S∗ is small relative to the other objects in the

image” “From left to right, where is S∗ positioned in the image?”

“Where is S∗ in the image?”

“What color is S∗?” “From left to right, where is S∗ located?”

“Where is S∗ positioned in image?”

“In the image, S∗ is positioned on the right side of the table”

“In the image, S∗ is sitting at a table with the other two people, likely enjoying a meal or a drink together”

“The color S∗ is white” “S∗ is located on the

right side of the image, sitting in a green bowl next to a wooden bowl containing oranges and pears”

“In the image, S∗ is positioned on top of the refrigerator, sitting on a shelf with various food items and containers”

Figure 23. Additional personalized visual question-answering results obtained by MyVLM, applied over LLaVA [54].

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

“What is S∗ sitting between?”

“What is standing next to S∗?”

“What does S∗ look like?”

“From left to right, where is S∗ located?”

“From left to right, where is S∗?”

“S∗ is sitting between the two white elephant statues on the top shelf of the bookshelf”

“S∗ is a small figurine of a character wearing a

pink hat with a blue flower on it. Next to S∗ , there is a blue mug with pens and pencils inside it”

“S∗ looks like a cute, cartoonish bunny rabbit with a blue hoodie, glasses, and a smiling face...”

“S∗ is located in the middle of the three shoes”

“S∗ is sitting on the left side of the shelf, while alexandria is sitting on the right side”

“Is S∗ small or large relative to the other objects in the image?”

“S∗ is small relative to the other objects in the image”

“From left to right, where is S∗ located?”

“S∗ is located on the left side of the window sill”

“From left to right, where is S∗ located?”

“S∗ is located on the right side of the image”

“What color is S∗?” “What color is S∗?”

“S∗ is pink.” “S∗ is yellow”

Figure 24. Additional personalized visual question-answering results obtained by MyVLM, applied over LLaVA [54].

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

“S∗ and a black dog walking towards each other in a garden”

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

“S∗ next to a cup of coffee that says coffee on it”

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

“A S∗ is inside of a washing machine.”

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

“S∗ and his friend are standing on the balcony of their apartment in New York City.”

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

“S∗ is sitting at a table with a man, and they are both looking at each other.”

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

“S∗ sitting on a shelf next to a bunch of pencils”

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

“A bowl full of oranges with S∗ sitting on top of them”

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

“A toy S∗ sitting on a white surface next to three white figurines of monkeys”

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

“S∗ and a man are in a kayak, with a cave in the background”

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

“S∗ and her boyfriend sitting on an airplane”

Figure 25. Additional personalized REC results obtained by MyVLM over MiniGPT-v2 [42]. Sample images of the target concept are provided in the top row. Bounding box coordinates returned by the personalized VLM are drawn in green. Below each image, we also present the personalized captions outputted by MyVLM by passing MiniGPT-v2 a captioning instruction.

