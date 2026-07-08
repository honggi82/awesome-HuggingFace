# arXiv:2410.13360v3[cs.CV]28Mar2025

## RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models

Haoran Hao1,2∗, Jiaming Han1∗, Changsheng Li3, Yu-Feng Li2, Xiangyu Yue1,4† 1MMLab, The Chinese University of Hong Kong 2National Key Laboratory for Novel Software Technology, Nanjing University 3Beijing Institute of Technology 4SHIAE, CUHK

This is <K>, Personalized Captioning She lives in Korea.

[Figure 1]

[Figure 2]

user database

[Figure 3]

[Figure 4]

[Figure 5]

<K> and <J> enjoying a relaxing afternoon at a trendy café.

remember

[Figure 6]

[Figure 7]

[Figure 8]

- K J

[Figure 9]

[Figure 10]

- L A

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

This is <J>, He is <K>’s boyfriend.

###### Personalized Conversation

[Figure 18]

[Figure 19]

[Figure 20]

What is <K> doing?

[Figure 21]

<J> <K>

[Figure 22]

[Figure 23]

retrieve

<K> is sitting at a table in a café, wearing a blue polka-dot dress. She is holding a glass of a pinkish drink with a straw and appears to be sipping from it.

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

generate What’s the relationship between them?

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

<J> is <K>’s boyfriend, based on the image, they

RAP-MLLM

[Figure 33]

appear to be enjoying each other's company at a café, their relaxed body language and the casual setting imply a close and comfortable relationship.

Figure 1. Introduce some user-specific concepts to our RAP-MLLM, it can remember them and achieve excellent performance in a variety of personalized multimodal generation tasks.

#### Abstract

The development of large language models (LLMs) has significantly enhanced the capabilities of multimodal LLMs (MLLMs) as general assistants. However, lack of userspecific knowledge still restricts their application in human’s daily life. In this paper, we introduce the Retrieval Augmented Personalization (RAP) framework for MLLMs’ personalization. Starting from a general MLLM, we turn it into a personalized assistant in three steps. (a) Remember: We design a key-value database to store userrelated information, e.g., user’s name, avatar and other attributes. (b) Retrieve: When the user initiates a conversation, RAP will retrieve relevant information from the database using a multimodal retriever. (c) Generate: The input query and retrieved concepts’ information are fed into MLLMs to generate personalized, knowledge-augmented responses. Unlike previous methods, RAP allows real-time

∗ Equal contribution † Corresponding author

concept editing via updating the external database. To further improve generation quality and alignment with userspecific information, we design a pipeline for data collection and create a specialized dataset for personalized training of MLLMs. Based on the dataset, we train a series of MLLMs as personalized multimodal assistants. By pretraining on large-scale dataset, RAP-MLLMs can generalize to infinite visual concepts without additional finetuning. Our models demonstrate outstanding flexibility and generation quality across a variety of tasks, such as personalized image captioning, question answering and visual recognition. The code, data and models are available at https://hoar012.github.io/RAP-Project/.

#### 1. Introduction

Recently, the development of large language models (LLMs) has significantly enhanced their language processing and generating capabilities [60]. Building on this foundation, the integration of visual and textual ability

- Table 1. Comparison of Different Personalization Methods. RAP needs only 1 image with its personalized description, showing outstanding convenience and flexibility in practical applications.

Number of Image Data Requirements for Personalization Support Method Positive Negative Caption Description Question-Answer Recognition Real-time edit Text-only QA

Fine-tuning n - Yes Yes No No ✗ ✓ MyVLM [2] n 150 Yes No Yes Yes ✗ ✗ Yo’LLaVA [32] n 200 No No Yes Yes ✗ ✓ RAP(Ours) 1 - No Yes No No ✓ ✓

through vision-language alignment brings powerful multimodal LLMs (MLLMs) [12, 15, 29, 33, 45, 51, 56]. MLLMs have shown significant improvement in various tasks, such as image description and question answering, highlighting their potential as humans’ assistants. However, their lack of user-specific knowledge continues to limit their effectiveness as personalized assistants in daily life.

A qualified personalized assistant first should be able to recognize and remember user-related concepts, such as the dog named 〈Lala〉 adopted by the user. Although existing MLLMs have been trained on large-scale datasets and possess strong recognition and classification capabilities, directly transferring this knowledge to a user’s personal concepts remains challenging. For instance, current leading MLLMs cannot remember your dog’s name, even if you have mentioned it before, and they lack awareness of your identity and preferences. Furthermore, the assistant should generate responses tailored to the user’s preferences and requirements. However, collecting extensive personal data to train a unique assistant for each user is impractical.

To address this issue, the personalization of MLLMs has gained increasing attention, with several approaches already being proposed. MyVLM [2] utilizes external classification heads to recognize specific concepts, and learns an embedding for each concept to personalize the outputs of vision language models (VLMs) [24, 29]. Another concurrent work, Yo’LLaVA [32], learns a few special tokens to represent each concept. However, both approaches require continuous learning and model updates as new concepts emerge. As shown in Table 1, they require multiple labeled images of the target concept along with a large number of negative images, making data collection a significant challenge. Alternatively, fine-tuning the model for each new concept also incurs substantial computational costs. This presents a challenge in dynamic, ever-changing real-world scenarios, where the computing power of users’ personal devices is often limited, and all data must be stored locally for privacy concerns.

To address these challenges, we propose the Retrieval Augmented Personalization (RAP) framework, designed to allow MLLMs to update their supported concepts without additional training. Specifically, RAP works in three key

steps. (a) Remember: RAP includes a designed database to help remember each concept via storing its image and basic information, e.g., name, avatar and other attributes. (b) Retrieve: When a user initiates a conversation, RAP will retrieve relevant information from the database using a multimodal retriever. (c) Generate: The input query and retrieved concepts information are incorporated into the MLLM’s input for personalized, knowledge-augmented generation. As shown in Table 1, our RAP requires only one image per concept with its related information for personalization. At the same time, it allows users to make real-time adjustments to the model’s outputs by modifying their personal databases, eliminating the need for retraining. Examples of real-time concept editing are presented in Table 12.

Another significant challenge is the lack of large-scale datasets for training MLLMs’ personalized generation capabilities. To address this, we design a pipeline to collect extensive training data and create a comprehensive dataset, which enables to train MLLMs to effectively understand and utilize user-related information for generation. Based on this dataset, we train LLaVA [29] and Phi3-V [36] as novel personalized assistants and evaluate their performance across various tasks, including personalized image captioning, question answering, and visual recognition. Experimental results demonstrate that our RAP-MLLMs excel in a wide range of personalized generation tasks, showcasing excellent generation quality and flexibility.

Our contributions are summarized as follows:

- • We propose the RAP framework for MLLMs’ personalization, allowing models pre-trained on our dataset to adapt to diverse users and infinite new concepts without further training.
- • We develop a pipeline for collecting large-scale data and create a dataset specifically designed for the personalized training and evaluation of MLLMs. This dataset enables us to train a series of MLLMs to function as personalized assistants.
- • Our models demonstrate exceptional performance across various personalized multimodal generation tasks, including image captioning and question answering. Additionally, they exhibit a strong capability to recognize personal concepts within images.

#### 2. Related Work

Multimodal Large Language Models. Recently, numerous advanced large language models (LLMs) [1, 8, 43, 44, 57] have been proposed, showing remarkable performance in addressing a wide range of tasks. The rapid development of LLMs has led to the emergence of multimodal LLMs (MLLMs) [12, 15, 29, 33, 45, 56, 61], which excel in general visual understanding and complex reasoning tasks. For instance, LLaVA [28, 29] and MiniGPT-4 [61] align visual and language modalities through visual instruction tuning, showcasing impressive capabilities in multimodal conversations. GPT4RoI [58] and RegionGPT [13] enhance finegrained understanding and reasoning for specific regions by training on region-level instruction datasets. Despite these advancements in tasks such as image captioning and question answering, the lack of user-specific knowledge restricts the generation of personalized content. In this work, we focus on the personalization of MLLMs, enabling them to remember and understand user-specific concepts, and generate personalized content tailored to user preferences.

Personalization of MLLMs. In the realm of artificial intelligence, personalization typically refers to the process of customizing a system, application, or model to meet individual needs and preferences [42, 46, 47, 50]. Substantial efforts have been made to generate images of a user’s personal objects or within certain contexts [10, 14, 21, 23, 38, 41, 49]. For example, Dreambooth [38] employs transfer learning in text-to-image diffusion models via finetuning all parameters for new concepts. In this paper, we mainly aim at enabling MLLMs to remember and understand user-specific concepts, and generate personalized language outputs. Several studies have focused on the personalization of MLLMs, among which the most relevant works are MyVLM [2] and Yo’LLaVA [32]. MyVLM introduces the task of personalizing VLMs. It utilizes external classification heads to recognize specific concepts, and learns an embedding for each concept to personalize the outputs of VLMs. Yo’LLaVA personalizes LLaVA by extending its vocabulary and learning specific tokens for each concept. However, both approaches require continuous model updates as new concepts emerge, which presents challenges in dynamic real-world applications. In this work, we propose the RAP framework for the personalization of MLLMs, enabling models pre-trained on our dataset to continuously update supported concepts without additional fine-tuning.

Retrieval Augmented Generation. Retrieval-based methods for incorporating external knowledge have been effective in improving generation across various knowledgeintensive tasks [3, 11, 27, 48, 52, 59]. DPR [19] introduces Dense Passage Retrieval, marking a shift from sparse to dense retrieval techniques. Later, MuRAG [5] proposes to use multimodal knowledge to augment language generation. Self-Rag [3] introduces special tokens to make re-

trieval adaptive and controllable. ERAGent [42] presents a comprehensive system for retrieval-augmented language models. With the advancements in MLLMs, RAG has been widely applied to multimodal generative tasks. For instance, FLMR [26] employs multi-dimensional embeddings to capture finer-grained relevance between queries and documents, achieving significant improvement in the RA-VQA setting. While existing methods primarily enhance models’ performance by retrieving from external knowledge bases, few of them consider the personalization task. Although RAG has been applied to image generation [4, 55] and image captioning [25, 35], there is currently no existing work that focuses on personalizing MLLMs through RAG, to the best of our knowledge.

#### 3. Retrieval Augmented Personalization

Existing MLLMs typically align other modalities with language. For instance, LLaVA [29] projects visual tokens into text space, and then generates subsequent tokens using an LLM. While these MLLMs perform well in various tasks, the lack of memory and comprehension of personal concepts hinders effective user-specific responses. In this work, we mainly focus on personalizing MLLMs to generate tailored language responses, such as creating personalized captions for user’s images and answering questions about personal concepts. In this section, we detail the implementation steps of our proposed Retrieval-Augmented Personalization (RAP) framework. Unlike previous approaches that usually necessitate additional data collection and further training to learn new concepts, after pre-training on our dataset, RAP-MLLMs can adapt to diverse users and infinite new concepts without further training. In section 3.1, we present the RAP framework that is applicable to various types of MLLMs, and then in section 3.2, we provide details of the proposed dataset.

##### 3.1. RAP Framework

RAP works in three steps: Remember, Retrieve and Generate. An overview of the framework is shown in Figure 2.

Remember. The premise of personalization is that the model can remember personal concepts and relevant information, such as the dog named 〈Lala〉 adopted by 〈A〉. To facilitate this, we construct a database M to store personal concepts, which comprises an avatar Ij, a name along with a brief description Tj for each concept. The key ki for each concept in the database is its visual feature, obtained by feeding its image Ij into a pre-trained image encoder E(·). Examples of our database are presented in Figure 2. When a user initiates a conversation, the input can be represented as Q = (Xv,Xq), which may include both image Xv and some textual instructions Xq. The first step involves identifying possible concepts within the input image that have been previously stored in the database. Previous meth-

[Figure 34]

###### Crop region of interest

[Figure 35]

Name: <A> A young woman who loves dog. Age: 22.

|[Figure 36]|
|---|

[Figure 37]

| | | |
|---|---|---|
| |Retriever| |
| | | |

[Figure 38]

Name: <Lala> A German Shepherd dog.

User database

###### … …

[Figure 39]

Open world detector

User input

[Figure 40]

Name: <H> A man from America. Hobby: Traveling.

###### Real-time update

[Figure 41]

User-related information

[Figure 42]

[Figure 43]

Instruction: Please give a caption of this image.

Name: <A> A young woman who loves dog. Age: 22

Name: <Lala> A German Shepherd dog.

Proj. Proj. Proj.

[Figure 44]

[Figure 45]

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

[Figure 56]

[Figure 57]

[Figure 58]

Multimodal Large Language Model

Response: <A> enjoys a sunny grooming session with her dog <Lala> in the countryside.

[Figure 59]

[Figure 60]

- Figure 2. Retrieval-Augmented Personalization Framework. Region-of-interest detected by an open world detector are used to retrieve concepts from the database. The images and information of the retrieved concepts are then integrated into the input for the MLLM.

ods [2] typically need to learn an external classifier to determine whether a concept appears in the input image, which requires a substantial amount of training data and can only apply to specific concept. To enhance the generalizability of the recognition process, we do not construct specific modules for each concept. Instead, we employ a universal detection model, such as YOLO [37] and YOLO-World [7], as recognition model R(·). Given the predefined setting P that specifies which categories should be remembered, the region-of-interest can be acquired via Xu = R(Xv,Xq|P). Retrieve. Identified region-of-interest will be used as query to retrieve from the database. For each recognized component Xiu, we feed the image crop into the image encoder E(·) to get its visual feature vi = E(Xiu), which is a ndimensional vector. Then we calculate the euclidean distance between the visual feature and each key kj ∈ M, which is calculated as Dist(vi,kj) = ∥vi − kj∥. The TopK image-text pairs {(I1,T1),(I2,T2),···(IK,TK)} with the lowest distances are selected from the database. We also introduce retrieval using concept names, such as ⟨sks⟩ for a unique concept. When the user mentions the name of an object documented in the database, our model retrieves its related information from the database. This also enables our model to respond to text-only queries effectively.

Generate. Each pair Mj = (Ij,Tj) provides related information about a user’s personal concept and will be incorporated into the input of the MLLM. Take LLaVA [29] as an example. The image Ij is first encoded by a pre-trained vision encoder, such as CLIP [34], to obtain their visual tokens Zj. These image tokens are then projected by a projector into language tokens Hvj, which could be understood by the language model. Simultaneously, corresponding text

information Tj are transformed into text tokens Hqj. This process is also applied to both Xv and Xq. All these tokens are incorporated into the MLLM’s input to generate language response. During training, we keep parameters of both the detector and retriever frozen, just train the MLLM’s parameters θ. Given the length L of the output sequence, the probability of the target answer Xa is computed as:

L

pθ(Xa,i|Xv,Xq,M1,···MK,Xa,<i). (1)

i=1

##### 3.2. Personalization Dataset

Most existing MLLMs struggle to generate personalized outputs even if additional concept information is provided, and there is currently no large-scale dataset for personalized training of MLLMs. To this end, we design a pipeline for data creation and curate a novel dataset specifically for the personalized training of MLLMs. We use Gemini-1.5 [12] to generate annotations for our dataset. An overview of our pipeline and dataset is presented in Figure 3.

The first component of our dataset is dedicated to visual grounding. In this task, an MLLM is trained to determine whether a specific concept is present in an image, particularly identifying if the person or object in a reference image appears in the given image. When a positive match is detected, we also require the model to provide the bounding box for the identified concept. For single-concept grounding, we primarily use the RefCOCO dataset [20]. Based on RefCOCO’s annotations, we crop target concepts from the images and assign names to them, which serve as references for specific concepts. We then query Gemini to generate concise descriptions about properties of the concepts in these cropped regions, by which we construct a

Rotate Flip Novel view

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Augment

Crop

3D model

[Figure 68]

This is <D>, give a description of him.

• A man wearing a bright orange T-shirt…

###### Concept Information Background

[Figure 69]

- Instruction type 1: Visual Grounding Instruction: Specify the rectangular boundaries of <D> in the image. Answer: [0.12, 0.15, 0.44, 0.94].
- Instruction type 2: Image Captioning & Description Instruction: Please give a caption of the image. Answer: <D> is playing ultimate frisbee, he's getting ready to pick up the frisbee.

The game is played on a field of grass.

- Instruction type 3: Question Answering: Instruction: What color is <D>’s shorts? Answer: Black.

<N>

Noise Concept

- Figure 3. Our Pipeline for Data Collection. We first crop the target concept from the image based on the dataset annotations and then query Gemini [12] to generate its personalized description. We also apply data augmentation to diversify these cropped images. Then we combine them with the original image to derive a series of instructions and answers from Gemini. When noise concepts are included in the additional information, the answer remains unchanged, helping to train the MLLMs’ ability to filter out irrelevant concepts.

large-scale database including numerous different concepts. The training data pairs images and these descriptions as queries and the corresponding bounding boxes as outputs. However, data generated in this way is insufficient to simulate the complexity of real-world recognition, especially when the target concept in the reference and input image is captured from different perspectives. To address this, we incorporate the ILSVRC2015-VID video object detection dataset [39], TAO [9] and CustomConcept101 [21] to enrich our dataset. For multi-object grounding, we use the Object365 dataset [40] to construct our training data.

The second component of our dataset is designed for instruction following. This section includes training data for tasks such as image captioning, image description, and question answering. For the image captioning and description data, we provide cropped images of target concepts, accompanied by their names and related information from the large-scale database, then query Gemini to generate a caption or description that reflects the concepts depicted in the entire image. For question answering, we first design a set of seed questions to serve as examples. These examples are used to prompt the annotator, Gemini, to generate new questions and corresponding answers. This iterative process facilitates the creation of a rich and diverse collection of conversations that MLLMs can learn from. We construct such data using RefCOCO [20], Object365 [40], TAO [9] and CustomConcept101 [21] dataset.

To enhance alignment with real-world scenarios, it is essential to collect data featuring the same identity in various environments. Thus, we also include multiple images about the same individual from the CelebA dataset [30] and

produce question answering data about the individual. To further diversify the dataset, we apply image editing techniques for data augmentation. This includes performing random rotations and flips on the cropped images, as well as generating novel views of the concepts by diffusion models. Specifically, we use Inpaint-Anything [53] to separate the foreground from the background, and use Wonder3D [31] and SiTH [16] to synthesize novel views of foreground object or person, respectively. Finally, we combine these elements to generate images of the target concept from different perspectives.

To support multi-concept personalized generation, it is necessary to retrieve multiple potential concepts from the database. In the generation step, the MLLM must prioritize accurate and contextually relevant information. Considering that retrieval results can be inaccurate, potentially leading to unreasonable answers, we construct negative samples by adding noise concepts to the input while preserving the original output. This approach trains the model’s discrimination ability. By exposing the MLLM to both relevant and irrelevant information during training, it learns to discern and filter out noise concepts, thereby enhancing its robustness during inference. Additionally, we include a subset of the LLaVA-Instruct-665k visual instruction dataset [28] to retain general knowledge from the original MLLM. Further details about our dataset can be found in Appendix D.

#### 4. Experiment

Implementation Details. We conduct experiments on LLaVA-1.5-13B [28] and Phi3-V-3.8B [36], resulting in

- Table 2. Qualitative Comparison on Image Captioning. Image examples of target concepts are shown in the left and captions generated are shown in the right. We use green text to denote correct target concepts.

Image Caption

LLaVA: A man is sitting at a table with a dog, and there are wine glasses and a fork on the table. LLaVA-LoRA: 〈collie dog〉 looking pleased as she shares a meal with her owner. MyVLM: 〈my dog〉 positioned on a chair by a black table, holding a wine glass in her hand. A white dog sits on the floor beside her... RAP-LLaVA(Ours): 〈my dog〉 is a very good boy, and he loves to sit at a table with his owner. They are enjoying a meal.

[Figure 70]

[Figure 71]

[Figure 72]

my dog

LLaVA: A man and a woman are standing in a kitchen, preparing food together. The woman is cutting lettuce on a cutting board, while the man watches her. There are several tomatoes ...

[Figure 73]

[Figure 74]

H

LLaVA-LoRA: 〈H〉 and 〈K〉 are preparing a meal together. MyVLM: 〈T〉 and her friend 〈H〉 are looking very serious as they take in the scenery. RAP-LLaVA(Ours): 〈H〉 is helping 〈T〉 prepare a salad in the kitchen.

[Figure 75]

T

[Figure 76]

[Figure 77]

Phi3-V: A group of stuffed animals, including a blue one, are sitting on a black surface. LLaVA-LoRA: 〈B〉, 〈G〉 and 〈W〉 are happily exploring the grassland. MyVLM: 〈G〉 and his crew are always ready to jump into a new adventure. RAP-Phi3-V(Ours): 〈W〉 is hanging out with 〈G〉 and 〈B〉 on the lawn. They are having a great time playing!

B

[Figure 78]

G

[Figure 79]

W

two personalized MLLMs, RAP-LLaVA and RAP-Phi3V. We select YOLO-Worldv2 [7] as the detector and construct a multimodal retriever using Facebook AI Similarity Search (FAISS) [18], employing a pre-trained CLIP ViTL/14-336 [34] as the visual encoder. Due to the context length limitation of the backbone language model, for RAPLLaVA and RAP-Phi3-V, we retrieve the 2 and 3 different concepts with the highest similarity, respectively. More implementation details can be found in Appendix C.

Training. In the training phase, we skip the recognition and retrieval procedures, instead perform instruction tuning to train the MLLMs. We adhere to most settings from the original experiment of LLaVA [28], except for using a maximum learning rate of 1e-4 and training for 1 epoch. We employ low-rank adapters [17] to reduce the number of trainable parameters, and train our models on 8 A100 GPUs with a valid batch size of 64.

##### 4.1. Personalized Image Captioning

In this section, we evaluate our models on generating personalized image captions with user-specific concepts. We extend the dataset introduced by MyVLM [2] via adding 16 new concepts, including both objects and humans, forming

8 concept pairs that appear together. For each pair, there are 8-13 images used for testing. This multi-concept setting presents additional challenges for personalization.

Settings. We compare our models with MyVLM [2] and fine-tuning based method LLaVA-LoRA [17]. For LLaVALoRA and MyVLM, the training dataset contains 1 image accompanied by 5 captions for each concept. For LLaVALoRA, we train it with captions of the training images for 3 epochs, applying low-rank adapters [17] and the same hyperparameters as our models. For MyVLM, following their training process, we first train the classification head with the positive and 150 negative images, then train the corresponding concept embedding with the provided captions for each concept. For our models, we construct a database where each concept is represented by a cropped image and a text description. Details of our database could be found in Appendix H. All remaining images are used as test samples. This evaluation process is repeated three times with different seeds, resulting in a total of 1,182 images used for evaluation, and we report the average results.

Qualitative Comparison. In Table 2, we present image captions generated by different methods to make a comparison. While LLaVA [28] and Phi3-V [36] generally pro-

- Table 3. Quantitative Evaluation on Image Captioning. We report Recall, Precision and F1-score in the table, the best result in each metric is bold and the second is underlined.

Method LLM Recall Precision F1-score LLaVA [28] + Retriever Vicuna-13B 1.260 48.76 2.450 LLaVA-LoRA [17] Vicuna-13B 82.97 93.28 87.82 MyVLM-LLaVA [2] Vicuna-13B 84.65 86.37 85.50 RAP-LLaVA Vicuna-13B 93.51 96.47 94.97 RAP-Phi3-V Phi3-V-3.8B 88.14 95.10 91.49

Figure 4. Performance under varying number of personalized concepts.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

50 100 150 200 250 300 Number of Concepts

85

90

95

F1-score

MyVLM

LLaVA-LoRA

RAP-Phi3-V RAP-LLaVA

- Table 4. Quantitative Evaluation on Question Answering and Visual Recognition. The best result in each setting is bold and the second is underlined. Evaluation results of GPT-4V [33] are also provided as reference. Weighted results are computed as arithmetic means.

Question Answering Visual Recognition

Method LLM Train #Image

Visual Text-only Weighted Positive Negative Weighted

GPT-4V [33] + Prompt GPT-4V ✗ 1 0.866 0.982 0.924 0.809 0.992 0.901 GPT-4V [33] + Prompt GPT-4V ✗ 5 0.887 0.987 0.937 0.851 0.998 0.925

LLaVA [28] Vicuna-13B ✗ - 0.899 0.659 0.779 0.000 1.000 0.500 LLaVA [28] + Retriever Vicuna-13B ✗ 1 0.912 0.863 0.887 1.000 0.025 0.513

LLaVA-LoRA [17] Vicuna-13B ✓ 1 0.900 0.583 0.741 0.988 0.662 0.825 LLaVA-LoRA [17] Vicuna-13B ✓ 5 0.935 0.615 0.775 0.997 0.444 0.721 MyVLM-LLaVA [2] Vicuna-13B ✓ 5 0.912 - - 0.994 0.845 0.919 Yo’LLaVA [32] Vicuna-13B ✓ 5 0.929 0.883 0.906 0.949 0.898 0.924

RAP-LLaVA(Ours) Vicuna-13B ✗ 1 0.935 0.938 0.936 0.979 0.982 0.980 RAP-Phi3-V(Ours) Phi3-V-3.8B ✗ 1 0.941 0.850 0.896 0.922 0.988 0.955

vide brief and clear captions for most test images, their lack of understanding of the user-specific concepts restricts them from generating a more personalized caption. LLaVALoRA and MyVLM can generate personalized captions, however, the limited training data often results in imprecise outputs, particularly noticeable when multiple concepts are present in the same image. In contrast, our models generate clear and accurate captions based on the database content, which also ensures the reliability of the outputs. Additional examples of personalized captions generated by the models could be found in Appendix E.

Quantitative Evaluation. We employ recall, precision and the comprehensive metric F1-score as our evaluation metrics. Recall is calculated as the percentage of correct occurrences of target concepts, while precision is the ratio of correct concept names to the total number of concept names presented. The experimental results are shown in Table 3. Notably, the classification heads of MyVLM exhibit higher error rates when the number of positive images is limited, leading to weaker performance. Even with additional concept information provided through retrieval, the vanilla LLaVA [28] still fails to effectively accomplish the personalized generation task, underscoring the necessity of

the proposed dataset. Our models demonstrate superior performance in both recall and precision metrics, highlighting the advantages of our RAP-MLLMs in data efficiency.

##### 4.2. Personalized Question Answering

Settings. In this section, we evaluate different methods on the benchmark of personalized question answering introduced by Yo’LLaVA [32], which contains both visual and text-only questions about user concepts. For each concept, we generate a description to serve as its information in our database. For LLaVA-LoRA, we feed these descriptions and corresponding images to train the model to describe the concept’s properties. Additionally, we incorporate text-only queries and answers to enhance the model’s textual understanding. The training dataset for Yo’LLaVA and MyVLM consists of 5 positive images with question answering pairs and 200 negative images per concept. For GPT-4V [33], images and related concept information are provided as a supplementary prompt. The evaluation metric is accuracy. Additional details are provided in Appendix C.

Results and Analysis. The experimental results are provided in Table 4. LLaVA and LLaVA-LoRA both perform well in visual based question answering, because substan-

- Table 5. We evaluate model’s performance with perfect retrieval, and test contributions of text information and dataset components.

Setting Recall Precision F1-score RAP-LLaVA 93.51 96.47 94.97 Skip retrieval 96.16 (+2.7) 100.0 (+3.5) 98.04 (+3.1)

- - Text information 94.91 (+1.4) 88.66 (-7.8) 91.68 (-3.3)

- - Data augmentation 89.25 (-4.3) 98.01 (+1.5) 93.42 (-1.6)

- - Negative samples 95.74 (+2.2) 58.21 (-38.3) 72.40 (-22.6)

tial information of the target concept can be obtained from the images. However, their performance is quite poor when images of the target concept mentioned in the question are not available. MyVLM performs well in visual question answering but does not support text-only question answering. Yo’LLaVA excels in text-only question answering, but its performance is still limited by the insufficient information provided by the learned tokens of a concept. In contrast, our models demonstrate balanced performance in both visual and text-only question answering. By providing a single image, our RAP-LLaVA surpasses baseline methods and achieves performance comparable to that of GPT-4V.

Visual Recognition. We also evaluate the models’ recognition abilities for a more comprehensive comparison. MLLMs are required to determine whether a personal concept is present in an image. We query them with ”Is 〈sks〉 in the image? Answer with a single word.”, where 〈sks〉 is replaced by corresponding concept name. For positive images, the desired response is ”Yes” and ”No” for negative. Results show that without knowledge of personal concepts, the vanilla LLaVA consistently produces negative responses. After training on target concepts, LLaVA-LoRA, MyVLM and YoLLaVA tend to give positive responses, but struggle to differentiate between similar concepts, resulting in weaker performance on negative images. Our models demonstrate exceptional performance in both positive and negative scenarios, achieving the best overall results.

##### 4.3. Ablation Study

Influence of Number of Learned Concepts. In real-world scenario, users’ personal databases typically expand over time. Next, we evaluate the performance of various methods with varying numbers of learned concepts. We extend the database with hundreds of new concepts selected from the RefCOCO dataset [20], ensuring no overlap with the test dataset. For LLaVA-LoRA and MyVLM, we provide images containing the target concepts along with their captions as training data, and assess models’ performance on the original test dataset. The results are presented in Fig-

- ure 4. More learned concepts result in increased recognition errors, leading to a decline in performance for each model. Our RAP-MLLMs maintain the highest performance under different settings.

Generation Ability of MLLM. We skip the recognition and retrieval processes, providing the MLLM with relevant information of each concept present in the image to evaluate the generation capability of the trained MLLM. The results, shown in Table 5, indicate that when relevant concept information is supplied, our RAP-LLaVA achieves superior generation performance, obtaining 100% precision without outputting irrelevant concepts as well as a higher recall rate. Text Information. We remove additional text information to examine its impact on personalization. The results are shown in Table 5 and 7 (Appendix). This additional text provides extra information that helps the model understand specific concepts, effectively improving the accuracy of the answers and benefiting personalized generation. Furthermore, it is especially helpful for question answering, as it can provide information not directly available from the image, such as a user’s age and hobbies, or a pet’s preferred food, as shown in the example in Table 12 of the Appendix. Dataset Composition. We conduct experiments to assess the contribution of each component in our dataset. First, we remove data generated through data augmentation and train the model. The results displayed in Table 5 indicate an obvious decrease in the recall metric for image captioning, resulting in a lower overall performance. We further exclude constructed negative samples from the dataset and retrain the model, then we find that it performs poorly on precision metric. This suggests a diminished ability to discriminate against noise concepts not present in the image.

Additional Ablation. We conduct ablation studies on retriever’s performance, and examine the impact of retrieving different numbers of concepts. We also evaluate models’ performance on several multimodal benchmarks, and the results demonstrate that RAP-LLaVA retains most general knowledge of the original LLaVA, while achieving superior performance in knowledge intensive tasks. Due to space limit, we put the results in Appendix Section B.1.

#### 5. Conclusion

In this paper, we introduce the RAP framework for personalizing MLLMs. This framework enables MLLMs to understand infinite user-specific concepts, generate personalized captions and respond to user-related queries. To enhance the quality of the generated content and better align outputs with user configuration, we curate a large-scale dataset for personalized training of MLLMs and train a series of MLLMs to function as personalized assistants. Experimental results show that RAP-MLLMs achieve exceptional performance in various personalized generation tasks, while allowing real-time adjustments to generation settings.

Acknowledgements. This work is partially supported by the National Natural Science Foundation of China (Grant No. 62306261), and The Shun Hing Institute of Advanced Engineering (SHIAE) Grant (No. 8115074).

#### References

- [1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024. 3
- [2] Yuval Alaluf, Elad Richardson, Sergey Tulyakov, Kfir Aberman, and Daniel Cohen-Or. Myvlm: Personalizing vlms for user-specific queries. arXiv preprint arXiv:2403.14599,

2024. 2, 3, 4, 6, 7, 14

- [3] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. arXiv preprint arXiv:2310.11511, 2023. 3
- [4] Andreas Blattmann, Robin Rombach, Kaan Oktay, Jonas M¨uller, and Bj¨orn Ommer. Retrieval-augmented diffusion models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. 3
- [5] Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W Cohen. Murag: Multimodal retrieval-augmented generator for open question answering over images and text. arXiv preprint arXiv:2210.02928, 2022. 3
- [6] Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-Wei Chang. Can pre-trained vision and language models answer visual information-seeking questions? arXiv preprint arXiv:2302.11713, 2023. 12
- [7] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. Yolo-world: Real-time open-vocabulary object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16901–16911, 2024. 4, 6, 13
- [8] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6,

2023. 3

- [9] Achal Dave, Tarasha Khurana, Pavel Tokmakov, Cordelia Schmid, and Deva Ramanan. Tao: A large-scale benchmark for tracking any object. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part V 16, pages 436–454. Springer,

2020. 5, 15

- [10] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3
- [11] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023. 3
- [12] Gemini-Team. Gemini 1.5: Unlocking multimodal under-

- standing across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 2, 3, 4, 5, 14
- [13] Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei Liu. Regiongpt: Towards region understanding vision language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13796– 13806, 2024. 3
- [14] Cusuh Ham, Matthew Fisher, James Hays, Nicholas Kolkin, Yuchen Liu, Richard Zhang, and Tobias Hinz. Personalized residuals for concept-driven text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8186–8195, 2024. 3
- [15] Jiaming Han, Kaixiong Gong, Yiyuan Zhang, Jiaqi Wang, Kaipeng Zhang, Dahua Lin, Yu Qiao, Peng Gao, and Xiangyu Yue. Onellm: One framework to align all modalities with language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26584– 26595, 2024. 2, 3
- [16] I Ho, Jie Song, Otmar Hilliges, et al. Sith: Single-view textured human reconstruction with image-conditioned diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 538–549, 2024. 5
- [17] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. 6, 7, 12
- [18] Jeff Johnson, Matthijs Douze, and Herv´e J´egou. Billionscale similarity search with gpus. IEEE Trans. Big Data, 7(3):535–547, 2021. 6, 13
- [19] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wentau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 6769–6781. Association for Computational Linguistics, 2020. 3
- [20] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara L. Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, EMNLP 2014, October 25-29, 2014, Doha, Qatar, A meeting of SIGDAT, a Special Interest Group of the ACL, pages 787–798. ACL, 2014. 4, 5, 8, 15
- [21] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 3, 5, 15
- [22] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. CoRR, abs/2408.03326, 2024. 12
- [23] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image

- generation and editing. Advances in Neural Information Processing Systems, 36, 2024. 3
- [24] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 2

- [25] Jiaxuan Li, Duc Minh Vo, Akihiro Sugimoto, and Hideki Nakayama. Evcap: Retrieval-augmented image captioning with external visual-name memory for open-world comprehension. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13733– 13742, 2024. 3
- [26] Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, and Bill Byrne. Fine-grained late-interaction multi-modal retrieval for retrieval augmented visual question answering. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10

- 16, 2023, 2023. 3

- [27] Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, et al. Ra-dit: Retrieval-augmented dual instruction tuning. arXiv preprint arXiv:2310.01352, 2023. 3
- [28] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 3, 5, 6, 7, 12, 13, 14, 15
- [29] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 2, 3, 4
- [30] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), 2015. 5, 15
- [31] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9970–9980, 2024. 5
- [32] Thao Nguyen, Haotian Liu, Yuheng Li, Mu Cai, Utkarsh Ojha, and Yong Jae Lee. Yo’llava: Your personalized language and vision assistant. arXiv preprint arXiv:2406.09400,

2024. 2, 3, 7, 13

- [33] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 2, 3, 7
- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, pages 8748–

8763. PMLR, 2021. 4, 6, 13

- [35] Rita Ramos, Desmond Elliott, and Bruno Martins. Retrieval-augmented image captioning. arXiv preprint arXiv:2302.08268, 2023. 3

- [36] Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad S. Khan. Llava++: Extending visual capabilities with llama-3 and phi-3, 2024. 2, 5, 6
- [37] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 779–788, 2016. 4
- [38] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 22500–22510. IEEE,

2023. 3

- [39] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211–252, 2015. 5, 15
- [40] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019, pages 8429–8438. IEEE, 2019. 5, 15
- [41] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8543–8552, 2024. 3
- [42] Yunxiao Shi, Xing Zi, Zijing Shi, Haimin Zhang, Qiang Wu, and Min Xu. Eragent: Enhancing retrieval-augmented language models with improved accuracy, efficiency, and personalization. arXiv preprint arXiv:2405.06683, 2024. 3
- [43] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Stanford alpaca: An instruction-following llama model, 2023. 3
- [44] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3
- [45] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. arXiv preprint arXiv:2403.15377, 2024. 2, 3
- [46] Stanisław Wo´zniak, Bartłomiej Koptyra, Arkadiusz Janz, Przemysław Kazienko, and Jan Koco´n. Personalized large language models. arXiv preprint arXiv:2402.09269, 2024. 3
- [47] Yihan Wu, Ruihua Song, Xu Chen, Hao Jiang, Zhao Cao, and Jin Yu. Understanding human preferences: Towards more personalized video to text generation. In Proceedings of the ACM on Web Conference 2024, pages 3952–3963,

2024. 3

- [48] Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhtu-

- rina, Mohammad Shoeybi, and Bryan Catanzaro. Retrieval meets long context large language models. arXiv preprint arXiv:2310.03025, 2023. 3
- [49] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3

- [50] Chun-Hsiao Yeh, Bryan Russell, Josef Sivic, Fabian Caba Heilbron, and Simon Jenni. Meta-personalizing visionlanguage models to find named instances in video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19123–19132, 2023. 3
- [51] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549, 2023. 2
- [52] Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. Making retrieval-augmented language models robust to irrelevant context. arXiv preprint arXiv:2310.01558, 2023. 3
- [53] Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin, Wenjun Zeng, and Zhibo Chen. Inpaint anything: Segment anything meets image inpainting. arXiv preprint arXiv:2304.06790, 2023. 5
- [54] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 12
- [55] Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, and Ziwei Liu. Remodiffuse: Retrieval-augmented motion diffusion model. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 364–

373. IEEE, 2023. 3

- [56] Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, et al. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. arXiv preprint arXiv:2407.03320, 2024. 2, 3
- [57] Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023. 3
- [58] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Yu Liu, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on regionof-interest. arXiv preprint arXiv:2307.03601, 2023. 3
- [59] Ruochen Zhao, Hailin Chen, Weishi Wang, Fangkai Jiao, Xuan Long Do, Chengwei Qin, Bosheng Ding, Xiaobao Guo, Minzhi Li, Xingxuan Li, et al. Retrieving multimodal information for augmented generation: A survey. arXiv preprint arXiv:2303.10868, 2023. 3
- [60] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie

Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023. 1

[61] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 3

## RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models

### Supplementary Material

#### A. Appendix Overview

- • Section B: Additional evaluations of our models.
- • Section C: More experimental details.
- • Section D: More details of the RAP dataset.
- • Section E: Additional demonstrations.
- • Section F: Analysis of limitations of our work.
- • Section G: Analysis of potential ethics issues.
- • Section H: Examples of the personalized database.

#### B. Additional Evaluation Results

##### B.1. Ablation Study

Dataset Composition. We conduct ablation experiments on the question answering and recognition benchmark, experimental results are present in Table 7. The results further demonstrate that our data augmentation and the constructed negative samples also contribute to the model’s performance.

N=45

N=300 N=500

N=150

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

0.9

0.8

0.7

1 2 3 4 5

K

- Figure 5. Retriever’s Top-K Recall under varying database size N.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

1 2 3 4 5

K

0.2

0.4

0.6

0.8

N=45

N=150

N=300 N=500

Figure 6. Retriever’s Top-K Precision under varying database size N.

Retriever. The retriever’s performance is crucial for a RAG system. We assess the retriever’s performance on the personalized captioning dataset. We use the detection model to identify potential concepts and retrieve the K concepts with the highest similarity from the database. Figure 5 and

- Figure 6 show the Top-K recall and precision for different values of K and database sizes N. The results indicate that as the database size increases, the retriever’s performance declines. While a larger K generally enhances recall, it also introduces more irrelevant concepts, leading to a drop in precision. Notably, even with 500 personal concepts to remember, the Top-5 recall rate can still exceed 90%, which guarantees the effectiveness of our RAP framework.

0.9

0.8

Recall

Precision

F1-score

0.7

1 2 3 4 5

K

Figure 7. Captioning Performance with varying number of retrieved concepts.

Impact of Retrieving Different Numbers of Concepts. We train a new model, RAP-LLaVA-OneVision (0.5B) [22], to analyze the impact of the number of retrieved concepts in detail. As shown in Figure 7, retrieving more concepts from a database of 300 concepts improves the recall metric of image captioning, but also introduces more noise, leading to a drop in precision. This reflects the trade-off between supporting a larger set of personalized concepts and maintaining generation precision.

Table 6. Evaluation on Knowledge-intensive Multimodal Benchmarks. KB: Knowledge Base.

Method MMMU [54] InfoSeek [6]

LLaVA [28] 0.364 0.205 LLaVA-LoRA [17] 0.359 0.205 RAP-LLaVA 0.361 0.218 RAP-LLaVA(With KB) 0.369 0.344

Multimodal Benchmark. We also evaluate our model’s performance on several traditional multimodal benchmarks, including MMMU [54] and InfoSeek [6]. We assess our models’ performance both with and without external knowledge base. For MMMU [54], we use 30K images paired with corresponding captions from Wikipedia as the external knowledge base. During testing, we retrieve the three most similar images based on the question’s image and incorporate only the textual knowledge to the input. For InfoSeek [6], we randomly sample 5K questions from the validation set and construct a knowledge base containing 50K entities from Wikipedia database provided by the authors, which includes all relevant entities associated with the questions. For each question, we retrieve the most similar entity and add only the textual knowledge to the input.

We evaluate on the validation set of MMMU, and 5K questions sampled from the validation set of InfoSeek. We use the official scripts to get the results, which are pre-

Table 7. Ablation studies on Question Answering and Visual Recognition. Weighted results are computed as arithmetic means.

Question Answering Visual Recognition

Method

Visual Text Weighted Positive Negative Weighted RAP-LLaVA 0.935 0.938 0.936 0.979 0.982 0.980

- - Text information 0.935 0.908 (-0.030) 0.921 (-0.015) 0.988 (+0.009) 0.930 (-0.052) 0.959 (-0.021)

- - Data augmentation 0.924 (-0.011) 0.918 (-0.020) 0.921 (-0.015) 0.943 (-0.036) 0.988 (+0.006) 0.965 (-0.015)

- - Negative samples 0.918 (-0.017) 0.933 (-0.005) 0.925 (-0.011) 0.958 (-0.021) 0.985 (+0.003) 0.971 (-0.009)

sented in Table 6. From the results, our RAP-LLaVA retains most general knowledge of the original LLaVA [28]. It also equips the MLLM with the ability to retrieve information from an external knowledge base, demonstrating superior performance in knowledge intensive tasks.

##### B.2. Standard Image Captioning Metrics

We further evaluate using standard image captioning metrics. The results are shown in Table 8. Despite the additional concept information is offered, LLaVA’s performance declines due to its inability to effectively utilize the information. Although our RAP-LLaVA is not trained on these concepts, it still achieves comparable performance on general image captioning tasks while enabling personalization.

##### B.3. Robustness to Retrieval Errors

As described in Section 3.2, we construct negative samples to enhance models’ robustness to retrieval errors. Fig-

- ure 5 and 6 show the retriever’s recall and precision under different database sizes. As the database size grows, the retriever’s recall and precision decrease, which means a higher likelihood of retrieving irrelevant concepts. Despite this, our models maintain outstanding performance, as shown in Figure 4, which demonstrates their robustness to retrieval errors. Additionally, results in Table 5 show that our models perform even better when the retrieval process is error-free. In Table 11, we provide examples to show model’s robustness to retrieval errors. Thanks to negative samples in the training dataset, RAP-MLLMs can distinguish irrelevant concepts and generate accurate responses.

MyVLM

LLaVA-LoRA

Yo’LLaVA

RAP-LLaVA

- 101

- 102

- 103

- 104

- 105

TimeCost(second)

| |
|---|

100 200 300 400 500

Number of Concepts

Figure 8. Time Cost of Personalization. We conduct experiment with 2 A800 GPUs.

##### B.4. Time Cost of Personalization

We also evaluate the time cost associated with different methods for learning a set of user’s concepts. The results are presented in Figure 8. MyVLM has to train an external recognition model for each concept and learn an embedding to adjust the model’s outputs. Similarly, Yo’LLaVA needs to learn new tokens for each concept. During the optimization process, both approaches necessitate multiple forward and backward pass of the MLLM, resulting in significant time consumption. In contrast, our RAP only requires time for encoding the image and adding its embedding to the database, which can be accomplished in just a few seconds. This significantly enhances the convenience and practicality of our models in practical applications.

#### C. More Experimental Details

Implementation Details. We utilize YOLO-Worldv2-X [7] as the detection model, setting detection classes to include all categories stored in the database to reduce the interventions from unrelated objects. We construct a multimodal retriever using Facebook AI Similarity Search (FAISS) [18], employing a pre-trained CLIP ViT-L/14-336 [34] as the visual encoder. Each key in the database is generated by inputting a concept’s image into the CLIP visual encoder, resulting in a 768-dimensional vector. Considering the restriction of context length of the backbone language model, we retrieve the 2 most similar images from the database for each region of interest. Then we select 2 and 3 different concepts with the highest similarity among all as supplementary inputs for RAP-LLaVA and RAP-Phi3-V, respectively.

Baselines. For MyVLM, we find that when the training data is very limited, it is quite hard for the classification head to work effectively. Therefore, we use data augmentation to help improve its performance. Specifically, we crop the single image into several pieces containing the target concept to improve the accuracy of classification heads. To distinguish between multiple different concepts that may appear in the image, we use 〈sks1〉, 〈sks2〉... as concept identifiers. For YoLLaVA, we present its experimental results reported in the original paper [32]. For GPT-4V, reference images and descriptions are provided as additional prompts, thus it can generate responses about the target concept.

Table 8. Quantitative Evaluation with Standard Image Captioning Metrics. We compute standard image captioning metrics for personalized captions generated by RAP-LLaVA. For each image, we treat all 5 augmented captions as the set of ground truth captions. The results are averaged across all three evaluations and all concepts.

Model B1 B2 B3 B4 METEOR ROUGE L CIDEr SPICE LLaVA [28] 0.177 0.100 0.057 0.032 0.138 0.240 0.428 0.136 LLaVA [28] + Retriever 0.087 0.038 0.014 0.005 0.074 0.132 0.025 0.020 MyVLM-LLaVA [2] 0.195 0.103 0.061 0.036 0.213 0.263 0.435 0.145 RAP-LLaVA 0.178 0.094 0.058 0.037 0.215 0.189 0.388 0.108

Multi-concept Data Collection. We collect videos from YouTube and sample frames from them. For each video, we detect multiple concepts and use Gemini [12] to find frames that contain both concepts in a given pair. For each pair, there are 8 to 13 images used for testing. Examples of these concept pairs are shown in Table 2. We generate five captions for each image to enhance the concept learning of baseline methods.

#### D. Details of Dataset

- D.1. Dataset Composition

- • We provide a summary of the composition of our dataset in Figure 9, which visually represents the distribution of different components.
- • Table 9 presents detailed numerical data for each part.
- • In Table 10, we specify the sources for each component of our dataset.

- D.2. Instructions

In this section, we present the instruction templates used to create our dataset:

- • Table 22 contains instructions for visual grounding and recognition.
- • Table 23 includes example instructions for image captioning.
- • Table 24 presents example instructions for image description.
- • Table 25 presents example questions used for question answering synthesis.

#### E. Additional Demonstrations

In this section, we provide more qualitative results obtained by various models.

- • In Table 12, we demonstrate how our models achieve real-time editing of concepts by modifying the database.
- • In Table 13, we demonstrate the real-time addition of new concepts by updating the database.
- • In Table 14, we present qualitative results on personalized conversation of RAP-LLaVA.

- • In Table 15, we present qualitative results on personalized conversation of RAP-Phi3-V.
- • In Table 16, we present additional image captions generated by RAP-LLaVA and other methods.
- • In Table 17, we present additional image captions generated by RAP-Phi3-V and other methods.
- • In Table 18, we provide demonstrations of image description generated by RAP-LLaVA and LLaVA.
- • In Table 19, we provide demonstrations of image description generated by RAP-Phi3-V and Phi3-V.
- • In Table 20 and 21, we provide results on visual recognition of RAP-LLaVA. It also has the ability to give precise bounding box of specific concept in the image.

#### F. Limitation

Our proposed RAP framework is a retrieval-based method. The limitations of RAP mainly concern the additional computational cost of generation and the precision of the retriever. While incorporating external information effectively generates more specific answers, it inevitably increases the context length for MLLMs, leading to additional computational overhead during the generation process. We will further explore ways to mitigate this computational burden. Another limitation is that the personalization performance of our RAP-MLLMs depends on the retriever’s capability. This proposes a need for a robust multimodal retriever that can discern intricate features to enhance retrieval precision. Despite these limitations, RAP offers a timely solution for MLLM personalization. By retrieving from a user’s specific database, RAP facilitates reliable and flexible personalized generation, which is valuable in practical applications.

#### G. Potential Ethics Issues

In our RAP framework, users can deploy and manage their own databases and personalized MLLMs locally. This design gives them greater flexibility to customize the model to fit their specific needs while ensuring privacy and security. However, improper use of personalized MLLMs can result in bias or other ethical concerns. For example, the model may unintentionally generate biased information or spread

Figure 9. Composition of our dataset.

Image Description Question Answering

Image Captioning LLaVA-Instruction

Recognition Visual Grounding

Table 9. Statistics of our dataset.

Type Size Visual Grounding 100K Recognition 40K Caption & Description 37K Question Answering 16K LLaVA-Instruction 67K Total 260K

Table 10. Data source. Type Source Dataset Visual Grounding

RefCOCO [20], TAO [9] ILSVRC2015-VID [39], Object365 [40] Recognition CustomConcept101 [21], CelebA [30] Caption & Description

RefCOCO [20], TAO [9] Object365 [40], CustomConcept101 [21]

RefCOCO [20], TAO [9] Object365 [40], CustomConcept101 [21] CelebA [30]

Question Answering

LLaVA-Instruction LLaVA-Instruct-665K [28]

#### H. Examples of the Personalized Database

We give some visualized examples of our database in Table 26. For each concept in the database, users need to provide an image with its name and optional text information. During inference, the images, names and other information of retrieved concepts are integrated into the input for the MLLM. Users have the flexibility to define the name and personalized description based on their preferences, and our RAP-MLLMs will generate answers according to the provided information.

- Table 11. Examples of Robustness to Retrieval Errors. Our RAP-MLLMs can distinguish noisy concepts and generate accurate answers.

Image Retrieved Concepts

[Figure 80]

[Figure 81]

[Figure 82]

my dog dog*

[Figure 83]

dog1

Question: Give a caption of this image. RAP-Phi3-V: 〈my dog〉 is looking out the window. Question: Can you see 〈dog*〉 in this image? RAP-Phi3-V: No, 〈dog*〉 is not visible in this picture.

misleading or harmful content. To mitigate these risks, it is crucial to implement effective mechanisms for bias detection, fairness evaluation, and responsible model usage.

- Table 12. Examples of Concept Editing. Based on the information recorded in the database, our RAP-LLaVA can provide reliable and accurate answers.

Image Concept Information in the Database

[Figure 84]

my dog

- 1. A white and gray dog with long fur. He has black eyes. His favorite food is chicken.

- 2. A white and gray dog with long fur. He has black eyes. His favorite food is beef.

[Figure 85]

Question: What is 〈my dog〉’s favorite food?

- 1. RAP-LLaVA: Chicken.

- 2. RAP-LLaVA: His favorite food is beef.

- Table 13. Examples of Concept Updating. The first caption is generated when toy2 not yet stored in the database. Once the new concept is added, RAP-LLaVA can recognize both toy1 and toy2.

Image Concept Information in the Database

[Figure 86]

- toy1 A plush toy. It is orange with a yellow belly and a brown nose.

[Figure 87]

- toy2

This is a plush toy of the bluey character. It is a light blue color with a purple patch on its head, and its ears are yellow.

[Figure 88]

Question: Give a caption of this image.

- 1. RAP-LLaVA: 〈toy1〉 is ready for bed! He’s snuggled up with his friend, a blue and yellow dog plushie. They’re both looking so cozy and ready for a good night’s sleep!

(Add new concept 〈toy2〉 to the database)

- 2. RAP-LLaVA: 〈toy1〉 and 〈toy2〉 are ready for bed! They are both snuggled under the covers with their favorite stuffed animals.

Table 14. Examples of personalized conversations obtained by RAP-LLaVA.

Image Concept Information in the Database

[Figure 89]

A tabby cat with its eyes closed is laying on a light grey surface. The cat’s fur is a mix of light and dark brown stripes.

my cat

Question: Where is 〈my cat〉? RAP-LLaVA: 〈my cat〉 is on a staircase. Question: What is the type of 〈my cat〉? RAP-LLaVA: 〈my cat〉 is a tabby cat. Question: What is the color of 〈my cat〉? RAP-LLaVA: 〈my cat〉 is a tabby cat with a mix of light and dark brown stripes.

[Figure 90]

Image Concept Information in the Database

[Figure 91]

〈A〉 is a smiling Asian man with short black hair, wearing glasses and a grey sweater.

A

Question: Can you see 〈A〉 in this photo? RAP-LLaVA: Yes, 〈A〉 is captured in this image. Question: What color is 〈A〉’s shirt?

[Figure 92]

RAP-LLaVA: 〈A〉 is wearing a purple shirt. Question: What is 〈A〉’s right hand holding? RAP-LLaVA: Chopsticks.

Image Concept Information in the Database

[Figure 93]

〈dug〉 is a golden retriever from the movie Up. He has a brown coat and a big, goofy smile, wearing a collar with some round objects.

dug

Question: Can you describe 〈dug〉 in detail?

RAP-LLaVA: 〈dug〉 is a golden retriever from the movie Up. He has a brown coat and a big, goofy smile. He is wearing a collar with some round objects. He looks very happy and playful.

(Text-only conversation)

Image Concept Information in the Database

[Figure 94]

P 〈P〉 is a bald man with a red beard.

(Text-only conversation)

Question: Can you describe 〈P〉’s appearence in detail?

RAP-LLaVA: 〈P〉 is a bald man with a red beard. He is wearing orange shorts and a blue shirt. He is sitting in a black director’s chair with his legs crossed.

Table 15. Examples of personalized conversations obtained by RAP-Phi3-V.

Image Concept Information in the Database

[Figure 95]

〈bo〉 is a well-groomed, medium-sized Shiba Inu with a thick, cinnamoncolored coat, cream accents, alert eyes, and a black collar.

bo

Question: What is 〈bo〉 doing? RAP-Phi3-V: 〈bo〉 is lying on its side on a fluffy, gray dog bed. It is resting its head on a white stuffed toy. 〈bo〉 has a thick, cinnamon-colored coat with cream accents, alert eyes, and a black collar. 〈bo〉 looks very relaxed and content. Question: Where do you think he likes to play? RAP-Phi3-V: It’s hard to say for sure, but it’s possible that 〈bo〉 enjoys going to the park. Many dogs like to go to the park to play and get exercise.

[Figure 96]

Image Concept Information in the Database

[Figure 97]

Anya A young girl with pink hair and big green eyes.

[Figure 98]

Bond A white dog wearing a black bow tie.

Question: What are they doing, describe in detail.

[Figure 99]

RAP-Phi3-V: 〈Anya〉 is feeding 〈Bond〉, a white dog wearing a black bow tie. 〈Bond〉 is eating out of a bowl. 〈Anya〉 is kneeling down and watching 〈Bond〉 eat. 〈Bond〉 is standing on a tiled floor and looking at the bowl. 〈Anya〉 is wearing a pink sweater and has pink hair. 〈Bond〉 is wearing a black bow tie. The scene is indoors, and there is a TV in the background.

Question: What is 〈Anya〉 wearing? RAP-Phi3-V: 〈Anya〉 is wearing a black sweater.

Image Concept Information in the Database

[Figure 100]

〈C〉 is a young woman with long blonde hair, wearing a white sleeveless shirt. She has bright red lipstick and light makeup.

C

(Text-only conversation)

Question: What color is 〈C〉’s hair? RAP-Phi3-V: 〈C〉’s hair is blonde. Question: What color of clothes does she like? RAP-Phi3-V: She likes white clothes.

Table 16. Additional qualitative comparison on image captioning between RAP-LLaVA and other methods.

Image Caption

LLaVA: A blue coffee cup sits next to two dolls on a window sill. LLaVA-LoRA: 〈doll*〉 positioned on a windowsill with a blue mug containing pencils MyVLM: 〈doll*〉 is found sitting beside a window ledge with a cup and toothbrushes nearby. In the image, 〈doll*〉 appears to be peacefully resting, possibly enjoying the view or taking a break. RAP-LLaVA(Ours): 〈doll*〉 and her friend are ready for a fun day at the office! They’re ready to take on any challenge that comes their way.

[Figure 101]

[Figure 102]

[Figure 103]

doll*

LLaVA: A wooden shelf holds several potted plants, including a large clay pot and a small vase. The shelf is positioned near a wall, and the plants are arranged in a visually appealing manner. LLaVA-LoRA: 〈mug*〉 on a wooden shelf next to a plant and two potted plants. MyVLM: 〈mug*〉 on a a blue and white tiled floor next to indoor plants, a shelf with pots and a vase. RAP-LLaVA(Ours): A brown ceramic tiki mug with the face of a tiki head, 〈mug*〉, sits on a wooden shelf with two potted plants.

[Figure 104]

[Figure 105]

[Figure 106]

mug*

[Figure 107]

[Figure 108]

LLaVA: A cat is sitting on a ledge near a staircase. LLaVA-LoRA: 〈my cat〉 sitting on a staircase, looking down. MyVLM: 〈my cat〉 in a typical pose, gripping the banister of a staircase. He is wearing a collar. RAP-LLaVA(Ours): 〈my cat〉 is relaxing on the stairs. Look at those eyes! What a beautiful cat.

[Figure 109]

my cat

[Figure 110]

[Figure 111]

LLaVA: A cat is laying on a blanket on a couch, next to a colorful teapot. LLaVA-LoRA: 〈my cat〉 reclining on a chair with a 〈teapot*〉 beside MyVLM: 〈teapot*〉 on the couch near an orange, yellow, and blue teapot. The cat is laying on a blanket, and the teapot is placed on a table. RAP-LLaVA(Ours): 〈my cat〉 is taking a well-deserved nap next to 〈teapot*〉.

my cat

[Figure 112]

teapot*

[Figure 113]

[Figure 114]

LLaVA: A man and a woman are walking down a street, with the man taking a selfie using his cell phone. They are both smiling as they walk, and the man is holding his phone ... LLaVA-LoRA: 〈H〉 and 〈K〉 are looking happy as they walk down the street together. MyVLM: 〈H〉 and 〈T〉 are looking very serious as they take in the scenery. RAP-LLaVA(Ours): 〈T〉 is walking down the street with her friend 〈H〉.

H

[Figure 115]

T

LLaVA: Two colorful parrots are perched on a palm tree, sitting next to each other. LLaVA-LoRA: 〈parrot1〉 and 〈parrot2〉 are looking very serious as they perch on a palm frond. MyVLM: 〈parrot1〉 and 〈parrot2〉 look so cute together.

[Figure 116]

[Figure 117]

- parrot1
- parrot2

[Figure 118]

RAP-LLaVA(Ours): 〈parrot1〉 and 〈parrot2〉 are both beautiful birds. 〈parrot1〉 has a bright red beak and 〈parrot2〉 has a light blue head and pink feet. They are both perched on a branch.

Table 17. Additional qualitative comparison on image captioning between RAP-Phi3-V and other methods.

Image Caption

[Figure 119]

[Figure 120]

Phi3-V: A small white sheep figurine is sitting on a wooden table next to a plant. LLaVA-LoRA: 〈sheep*〉 positioned next to a potted plant on a wooden table. MyVLM: 〈sheep*〉 positioned within a square shape within a vase made of metal positioned on a table made of wood. RAP-Phi3-V(Ours): 〈sheep*〉 is ready to play on the shelf with the plant!

[Figure 121]

sheep*

[Figure 122]

[Figure 123]

Phi3-V: A brown and white dog is sitting on the grass with a leash attached to it. LLaVA-LoRA: 〈dog1〉 sitting on the grass with a look of determination. MyVLM: on a cheerful note, 〈dog1〉 restfully resides, surrounded by the soothing sounds of nature. RAP-Phi3-V(Ours): This is a picture of 〈dog1〉, a cute dog with light brown and white fur. He looks happy and playful.

[Figure 124]

dog1

[Figure 125]

[Figure 126]

Phi3-V: A bottle of wine is displayed next to a wooden cow statue. LLaVA-LoRA: 〈bull*〉 positioned next to a bottle of supreme cabernet sauvignon. MyVLM: 〈gold pineapple〉 pars the bottle of wine. A bottle of wine sits next to a bottle of wine. RAP-Phi3-V(Ours): 〈bull*〉 stands guard over a bottle of SUPREME wine.

[Figure 127]

bull*

[Figure 128]

[Figure 129]

Phi3-V: A small figurine of a basketball player is placed on top of a box. LLaVA-LoRA: 〈funko pop*〉 sitting on a desk next to a box and a monitor. MyVLM: 〈funko pop*〉 stands out in a photo set against a dark background. RAP-Phi3-V(Ours): 〈funko pop*〉 is ready to dominate the court!

[Figure 130]

funko pop*

[Figure 131]

[Figure 132]

Phi3-V: A man and woman are sitting at a table in a restaurant, having a conversation and enjoying their time together. LLaVA-LoRA: 〈J〉 and 〈K〉 enjoying a meal together. MyVLM: 〈J〉 and 〈K〉 enjoying their time together. RAP-Phi3-V(Ours): 〈J〉 and 〈K〉 are enjoying a drink at a bar.

- J
- K

[Figure 133]

[Figure 134]

[Figure 135]

Phi3-V: A small statue of a man wearing glasses is sitting next to a cup on a table. LLaVA-LoRA: A table hosts 〈cup*〉 and a sculpture of a head wearing glasses. MyVLM: A table with 〈cup*〉, an espresso cup, and a ceramic mug depicting a face wearing glasses sits on a wooden surface. RAP-Phi3-V(Ours): 〈head*〉 and 〈cup*〉 are ready for their morning coffee.

cup*

[Figure 136]

head*

Table 18. Qualitative results of personalized image description obtained by RAP-LLaVA.

Image Description

LLaVA: The image features a large, fluffy cat sitting on a countertop, possibly in a kitchen. The cat is positioned near a sink and appears to be resting or sleeping. The countertop is covered with a towel, providing a comfortable surface for the cat to lay on ...

[Figure 137]

[Figure 138]

[Figure 139]

RAP-LLaVA(Ours): The image shows 〈my cat〉, a tabby cat with its eyes closed, laying on a light grey surface. The cat’s fur is a mix of light and dark brown stripes. The cat is relaxed and comfortable.

my cat

LLaVA: The image features a small metal robot figure, possibly a toy, standing on a table or counter. The robot is positioned in the center of the scene, surrounded by a variety of leaves scattered around it. The leaves are of different sizes and ...

[Figure 140]

[Figure 141]

[Figure 142]

RAP-LLaVA(Ours): The image shows a toy figurine of Iron Man, a superhero from Marvel comics. He is wearing a silver and red suit of armor and is standing on a surface covered in leaves. The toy is called 〈toy3〉.

toy3

LLaVA: The image features a chair with a pillow on it, which has a picture of a sheep on it. The chair is positioned in front of a desk with a laptop and a monitor on it. The laptop is placed on the left side of the desk, while the monitor is on the right side ...

[Figure 143]

[Figure 144]

[Figure 145]

RAP-LLaVA(Ours): The image shows a chair with a 〈pillow*〉 on it. The 〈pillow*〉 is a pattern of flowers, leaves, and a sheep wearing glasses with a flower crown. There is also a laptop and a computer monitor in the background.

pillow*

[Figure 146]

[Figure 147]

LLaVA: The image features a bookshelf filled with various books. On the shelf, there are two Rubik’s cubes, one green and one white, sitting next to each other. The bookshelf is well-stocked with books, occupying most of ...

RAP-LLaVA(Ours): The image shows a bookshelf with books on it. There are two toys on the shelf. One is a 〈toy4〉, which looks like a cat with orange fuzzy ears and a brown cord attached. The other is a Rubik’s cube.

[Figure 148]

toy4

LLaVA: The image features a bed with two stuffed animals, a dog and a cat, lying on it. The dog is positioned on the left side of the bed, while the cat is on the right side. The stuffed animals are placed close to each other, giving the impression of a cozy and ...

[Figure 149]

[Figure 150]

toy1

[Figure 151]

RAP-LLaVA(Ours): The image shows two plush toys, 〈toy1〉 and 〈toy2〉, lying in bed together. 〈toy2〉 is a light blue plush with a purple patch on its head and yellow ears. 〈toy1〉 is an orange plush with a yellow belly and a brown nose. They are both looking at the camera.

toy2

LLaVA: The image features a young girl hugging a large white dog in a room. The girl is wearing a black dress and appears to be enjoying her time with the dog. The dog is positioned close to the girl, and they seem to be sharing a tender moment together ...

[Figure 152]

[Figure 153]

Anya

RAP-LLaVA(Ours): 〈Anya〉 is a young girl with pink hair and big green eyes. She is wearing a black dress and black shoes. She is standing in a room with wooden floors and is hugging a white dog named 〈Bond〉. 〈Bond〉 is wearing a black bow tie. The dog is standing on its hind legs and is looking at the camera. The room is decorated with a blue rug and a wooden cabinet ...

[Figure 154]

Bond

Table 19. Qualitative results of personalized image description obtained by RAP-Phi3-V.

Image Description

Phi3-V: The image features a display of various animal figurines on a shelf. There are three giraffes, two of which are standing and one sitting, placed next to each other. A sheep figurine is also present on the shelf, located between the giraffes ...

[Figure 155]

[Figure 156]

[Figure 157]

RAP-Phi3-V(Ours): The image shows a white 〈sheep*〉 sitting on a shelf next to a giraffe figurine. The 〈sheep*〉 is made of felt and has black eyes and a black nose. The giraffe figurine is made of metal and has a colorful necklace around its neck. The shelf is white ...

sheep*

Phi3-V: The image features a wooden shelf with two wine bottles and two wine glasses. One of the wine bottles is positioned towards the left side of the shelf, while the other is located more towards the center. The wine glasses are placed on the right side of the ...

[Figure 158]

[Figure 159]

RAP-Phi3-V(Ours): The image shows a wooden shelf with three glassware items on it. The top glassware item is a 〈mug*〉 with a tiki head design. The other glassware items are wine glasses. There is a sign on the wine bottle that says ”SUPREME”. There is also a plant in a pot in the background.

[Figure 160]

mug*

Phi3-V: The image features two dogs playing together in a grassy yard. One dog is located on the left side of the yard, while the other dog is on the right side. They are both running and enjoying their time outside. In the background, there are several chairs ...

[Figure 161]

[Figure 162]

[Figure 163]

RAP-Phi3-V(Ours): The image shows two dogs playing in a grassy backyard. The dog on the left is a white and gray dog with long fur. The dog on the right is a black and brown dog with long fur. The dogs are both running and playing on the grass. The dog on the left is named 〈my dog〉.

my dog

Phi3-V: The image features a white shelf with several decorative items on it. There are three chicken-shaped decorations hanging from the shelf, each with a different design. Additionally, there are three more chicken decorations placed on the shelf, making ...

[Figure 164]

[Figure 165]

RAP-Phi3-V(Ours): The image shows three handmade stuffed toys. The toys are all shaped like chickens and are hanging on a white shelf. The toys are all different colors and have different designs on them. The toy on the left is called 〈toy4〉.

[Figure 166]

toy4

Phi3-V: The image features a group of three cartoon characters, two of which are wearing hats. They are standing together on a sidewalk, with one character positioned towards the left side of the frame, another in the center, and ...

[Figure 167]

RAP-Phi3-V(Ours): The image shows 〈character1〉 and 〈character2〉 standing in front of a brick wall. 〈character1〉 is a cartoon fox wearing a yellow button-down shirt and a purple tie. He has big eyes, fluffy fur, and a sly grin on his face. 〈character2〉 is a cartoon rabbit with large ears, a police officer’s hat, and a red and yellow safety vest. She has a sweet, innocent look on ...

- character1

[Figure 168]

- character2

[Figure 169]

[Figure 170]

[Figure 171]

Phi3-V: The image features a baby lying on a bed, with a dog resting on the floor nearby. The baby appears to be playing with a toy, possibly a puzzle, while the dog watches attentively ...

baby*

RAP-Phi3-V(Ours): The image shows a baby, 〈baby*〉, sleeping in a carrier. The baby is wrapped in a colorful blanket and is secured by a black strap. A french bulldog, 〈dog2〉, is sleeping next to the baby. The baby and dog are both sleeping peacefully.

[Figure 172]

dog2

[Figure 173]

[Figure 174]

Question: Is 〈head*〉 in this image? RAP-LLaVA: Yes, 〈head*〉 is in the photo.

[Figure 175]

Question: Give 〈head*〉’s bounding box in the image. RAP-LLaVA: [0.57, 0.67, 0.68, 0.84]

head*

[Figure 176]

[Figure 177]

Question: Is 〈head*〉 in this image? RAP-LLaVA: No, 〈head*〉 is not part of this photograph.

[Figure 178]

head*

[Figure 179]

[Figure 180]

Question: Can you identify if 〈toy*〉 is captured in this picture? RAP-LLaVA: Yes, 〈toy*〉 is visible in this picture.

[Figure 181]

Question: Please provide 〈toy*〉’s bounding coordinates in the image. RAP-LLaVA: [0.42, 0.19, 0.54, 0.39]

toy*

[Figure 182]

[Figure 183]

Question: Can you identify if 〈toy*〉 is captured in this picture? RAP-LLaVA: 〈toy*〉 is not captured in this image.

[Figure 184]

toy*

[Figure 185]

[Figure 186]

Question: Can you confirm if 〈A〉 appears in this photo? RAP-LLaVA: Yes, 〈A〉 is in this photo.

[Figure 187]

Question: Describe 〈A〉’s position in the image. RAP-LLaVA: [0.24, 0.12, 0.72, 0.98]

A

[Figure 188]

[Figure 189]

Question: Can you confirm if 〈A〉 appears in this photo? RAP-LLaVA: 〈A〉 is not present in this photograph.

[Figure 190]

A

[Figure 191]

[Figure 192]

Question: Is 〈statue1〉 included in this shot? RAP-LLaVA: Yes, 〈statue1〉 is included in this shot.

[Figure 193]

Question: Give 〈statue1〉’s bounding box in the image. RAP-LLaVA: [0.36, 0.16, 0.59, 0.41]

statue1

[Figure 194]

[Figure 195]

Question: Is 〈statue1〉 included in this shot? RAP-LLaVA: 〈statue1〉 is not captured in this picture.

[Figure 196]

statue1

[Figure 197]

[Figure 198]

Question: Is 〈my dog〉 captured in this image? RAP-LLaVA: Yes, I recognize 〈my dog〉 in this picture.

[Figure 199]

Question: Specify 〈my dog〉’s bounding box in the photograph. RAP-LLaVA: [0.33, 0.23, 1.0, 0.69]

my dog

[Figure 200]

[Figure 201]

Question: Is 〈my dog〉 captured in this image? RAP-LLaVA: No, 〈my dog〉 is not included in this shot.

[Figure 202]

my dog

[Figure 203]

[Figure 204]

Question: Is 〈teapot*〉 visible in this image? RAP-LLaVA: Yes, 〈teapot*〉 is visible in this picture.

[Figure 205]

Question: Describe 〈teapot*〉’s position in the image. RAP-LLaVA: [0.52, 0.69, 0.73, 0.91]

teapot*

[Figure 206]

[Figure 207]

Question: Is 〈teapot*〉 visible in this image? RAP-LLaVA: No, 〈teapot*〉 is not visible in this picture.

[Figure 208]

teapot*

Table 22. Instructions for visual grounding and recognition.

Visual recognition:

- • Is ⟨V⟩ visible in this picture?
- • Is ⟨V⟩ in this image?
- • Do you see ⟨V⟩ in the photo?
- • Is ⟨V⟩ present in this photograph?
- • Can you identify if ⟨V⟩ is captured in this picture?
- • Is ⟨V⟩ depicted in this image?
- • Does the picture feature ⟨V⟩?
- • Can you confirm if ⟨V⟩ appears in this photo?
- • Is ⟨V⟩ included in this shot?
- • Is ⟨V⟩ shown in this image?
- • Can you tell if ⟨V⟩ is part of this photograph?
- • Is there any sign of ⟨V⟩ in this picture?
- • Can you detect ⟨V⟩ in the photo?
- • Is ⟨V⟩ captured in this image?
- • Do you recognize ⟨V⟩ in this picture? Visual grounding:
- • Give 〈V〉’s bounding box in the image.
- • Describe 〈V〉’s position in the image.
- • Please provide the coordinates of the bounding box for 〈V〉 in the given image.
- • Specify the rectangular boundaries of 〈V〉 in the image.
- • Give 〈V〉’s position in the following image.
- • Please provide 〈V〉’s bounding coordinates in the image.
- • Indicate the bounding box for 〈V〉 in the image.
- • Show the bounding box for 〈V〉 in the picture.
- • Specify 〈V〉’s bounding box in the photograph.
- • Mark 〈V〉’s bounding box within the image.

Table 23. Instructions for image captioning.

Image caption:

- • Give a caption of the image.
- • Give a personalized caption of this image.
- • Provide a brief caption of the image.
- • Summarize the visual content of the image.
- • Create a short caption of the image.
- • Offer a short and clear interpretation of the image.
- • Describe the image concisely.
- • Render a concise summary of the photo.
- • Provide a caption of the given image.
- • Can you provide a personalized caption of this photo?
- • Could you describe this image concisely?

Table 24. Instructions for image description.

Image description:

- • Describe the image.
- • Give a description of the image.
- • Give a description of the image in detail.
- • Give a short description of the image.
- • Describe the image in detail.
- • Please provide a description of the image.
- • Can you give me details about the image?
- • Could you explain what’s shown in the image?

Table 25. Seed questions used for question answering synthesis.

Person:

- • What is 〈H〉’s hair color?
- • What is 〈H〉’s height (estimated)?
- • What is 〈H〉’s skin tone?
- • What is 〈H〉’s eye color?
- • What style of clothing is 〈H〉 wearing?
- • Does 〈H〉 have any visible tattoos?
- • Does 〈H〉 wear glasses or contact lenses?
- • Does 〈H〉 have any facial hair?
- • What is 〈H〉’s approximate age?
- • What is 〈H〉’s build or body type?
- • What is 〈H〉 doing? Object:
- • What color is 〈O〉?
- • What pattern is on 〈O〉?
- • What shape does 〈O〉 have?
- • What size is 〈O〉?
- • What is the texture of 〈O〉?
- • Is 〈O〉 shiny or matte?
- • What material is 〈O〉 made of?
- • Does 〈O〉 have any patterns or designs on it?
- • Is 〈O〉 new or worn?
- • Does 〈O〉 have any visible brand or logo?
- • Is 〈O〉 functional or decorative? Multi-concept question:
- • What do 〈C1〉 and 〈C2〉 have in common?
- • What activity are 〈C1〉 and 〈C2〉 engaged in?
- • Where could 〈C1〉 and 〈C2〉 be located?
- • What is the most noticeable difference between 〈C1〉 and 〈C2〉?
- • What are they doing?

Table 26. Examples of our database. A concept should be provided with an image and its personalized description.

Image Concept Information

[Figure 209]

Anya A young girl with pink hair and big green eyes.

[Figure 210]

This is a cute figurine of a girl wearing a pink and blue dress, holding a white bubble.

doll*

[Figure 211]

- toy1 A plush toy. It is orange with a yellow belly and a brown nose.

[Figure 212]

- toy2

This is a plush toy of the bluey character. It is a light blue color with a purple patch on its head, and its ears are yellow.

[Figure 213]

This is a figurine of a cat. The cat has a blue body with yellow, red, and green stripes and a long tail that is also striped.

statue*

[Figure 214]

cat* A small ginger kitten with bright blue eyes looks up at the camera.

[Figure 215]

H A young man is wearing a plain tan t-shirt. His hair is short and curly.

[Figure 216]

my dog A white and gray dog with long fur. He has black eyes.

[Figure 217]

A young woman with blonde hair is wearing a white tank top and blue jeans.

T

