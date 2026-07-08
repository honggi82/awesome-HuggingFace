# arXiv:2307.08581v1[cs.CV]17Jul2023

[Figure 1]

## BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs

Yang Zhao∗, Zhijie Lin∗, Daquan Zhou, Zilong Huang, Jiashi Feng, Bingyi Kang† {zhaoyang98, linzhijie11, daquanzhou, zilonghuang, jshfeng, bingyikang} @bytedance.com ∗ Equal Contribution, † Project Lead

### Abstract

LLMs have demonstrated remarkable abilities at interacting with humans through language, especially with the usage of instruction-following data. Recent advancements in LLMs, such as MiniGPT-4, LLaVA, and X-LLM, further enlarge their abilities by incorporating multi-modal inputs, including image, video, and speech. Despite their effectiveness at generating precise and detailed language understanding of the given modality signal, these LLMs give up the ability to ground specific parts of inputs, thus only constructing a coarse-grained mapping. However, explicit and informative correspondence between text and other modalities will not only improve the user experience but also help to expand the application scenario of multi-modal LLMs. Therefore, we propose BuboGPT, a multi-modal LLM with visual grounding that can perform cross-modal interaction between vision, audio and language, providing fine-grained understanding of visual objects and other given modalities. As a result, BuboGPT is able to point out the specific location of an object in the image, when it is generating response or description for that object. Our contributions are two-fold: 1) An off-the-shelf visual grounding module based on SAM that extracts entities in a sentence and find corresponding masks in the image. 2) A two-stage training scheme and instruction dataset to endow joint text-image-audio understanding. Our experiments show that BuboGPT achieves impressive multi-modality understanding and visual grounding abilities during the interaction with human. It performs consistently well when provided by arbitrary modality combinations (either aligned or unaligned). Our code, model and dataset are available at https://bubo-gpt.github.io.

### 1 Introduction

The large language models (LLMs) have made significant progress and demonstrated promising abilities in few-shot and zero-shot learning by leveraging instruct tuning [1] on carefully curated datasets. To harness the potential of LLMs beyond just language, some recent studies [2, 3, 4, 5, 6, 7, 8, 9, 10] successfully connect LLMs with more input signals (e.g., image, video, speech and audio), and build powerful multi-modal chatbots. However, these models often perform understanding without digging into the fine-grained relation between the visual objects and other given modalities. For example, when an illustrative figure is given, a visually-enhanced LLM will generate a highquality description with rich details, but in a black-box manner. Instead, an instructive teacher-bot is going to show its audience which part of the figure it is referring to and what is happening there. Such visual grounding abilities are intriguing to LLMs but previously under-explored in the literature.

In this paper, we propose BuboGPT, the first attempt to incorporate visual grounding into LLMs by relating visual objects with other modalities. Moreover, it is able to perform joint multi-modal understanding and chatting for text, vision and audio, which is achieved by learning a shared representation space that aligns well with pre-trained LLMs.

[Figure 2]

Vicuna GroundingVisual

Linear

Linear

Q-Former Find the sounding source.

Q-Former

Based on the image of a boy riding a bicycle, it appears that the source of the audio is the sound of a dog barking … creates a vivid and lively scene in the image.

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Figure 1: The overall framework of BuboGPT.

To this end, we first build an off-the-shelf visual grounding pipeline based on SAM [11] to explore the fine-grained relation between different visual objects and modalities. The pipeline is composed of three modules, namely, a tagging module, a grounding module and a entity-matching module. The tagging module is a pre-trained modal [12] that can generate multiple text tags/labels that are relevant to the input image. The SAM-based [11] grounding module [13] further localize the semantic mask or box on the image for each tag/label. Then, the entity-matching module leverages the reasoning capabilities of LLMs to retrieve matched entities from tags and image descriptions. In this way, we connect visual objects and other modalities by using language as a bridge.

Then, to unlock the multi-modal understanding ability for arbitrarily combined inputs, we employ a two-stage training scheme similar to Mini-GTP4 [2]. More specifically, we use ImageBind [14] as the audio encoder, BLIP-2 [15] as the vision encoder and Vicuna [16] as the LLM. In the first stage, we learn a Q-former to align vision or audio features with language on image or audio caption datasets respectively. In the second stage, we perform multi-modal instruct tuning on a high-quality instruction-following dataset. We observe that the construction of this dataset is crucial for the LLM to recognize whether a modality is provided and whether the input modalities are well matched with each other. Therefore, we devise a novel high-quality dataset, which is composed of four subsets: 1) vision instruction dataset; 2) audio instruction dataset; 3) sound localization dataset with positively paired image-audio examples; 4) image-audio captioning dataset with negative pairs. Note that by introducing the negative image-audio pairs for semantic reasoning, the BuboGPT can learn better multi-modal alignment and demonstrate stronger capabilities of joint understanding.

Our experiments show that BuboGPT achieves impressive visual grounding abilities during multimodal chat, even when arbitrary combinations of multi-modal inputs are provided, whether matched or unmatched. We summarize our key contributions as follows:

- • We build a multi-modal LLM, BuboGPT for multi-modal understanding including image, audio and text by learning a common semantic space and further explore the fine-grained relation between different visual objects and different modalities.
- • We construct a high-quality multi-modal instruction-tuning dataset including fine-grained audio descriptions and cross-modal sound localization, and introduce both positive and negative image-audio pairs for semantic matching to facilitate the cross-modal understanding.

### 2 Related Work

Pre-trained LLMs in Mutli-modal Learning. Due to the scaling up of training data and model size, large language models [17, 18, 19, 16] have demonstrated remarkable abilities across various linguistic tasks in a few-shot and zero-shot manner and also enabled conversational communication with humans. To leverage the powerful linguistic abilities of LLMs, some methods [20, 21] propose to connect different accessedation models for multi-modal tasks by using LLMs as a dispatch scheduler.

Based on high-quality multi-modal instruction-following data, recent end-to-end methods [2, 3, 4, 5, 6, 7, 8, 9, 10] have been introduced to extend LLMs for multi-modal learning as well. Some works such as Mini-GPT4 [2], X-LLM [3] and Video-ChatGPT [10] propose to align the input features of different modalities with pre-trained LLMs by learned visual encoder. Some works such

- as LLaMA-Adapter [5] and Otter [7] insert learnable cross-attention layers into the pre-trained LLMs

ii

to incorporate multi-modalities knowledge. These prior methods mainly focus on tackling visual inputs (e.g. videos and images) [2, 5, 6, 4, 9, 7] or ignoring the fine-grained relation between the visual objects and other given modalities [8, 3]. We further attempt to incorporate visual grounding into LLMs by relating visual objects with other modalities and propose to learn multi-modal alignment including image, audio and text in a common space.

Multi-modal Instruction Tuning Dataset. To explore instruction tuning for multi-modal learning, [22] first introduces a multi-modal instruction tuning benchmark that is composed of 62 diverse multi-modal tasks in a unified seq-to-seq format. Mini-GPT4 [2] curates an instruction following dataset by combining Conceptual Caption [23, 24], SBU [25] and LAION [26] with hand-designed prompt, while LLaVA [6] proposes to use GPT-4 [17] to generate more detailed captions to expand COCO dataset [27]. Otter [7] further builds a multi-modal in-context tuning dataset to facilitate the in-context learning capabilities of multi-modal LLMs. Further, we build a high-quality instruction tuning dataset including fine-grained audio description and introduce the negative image-audio pairs for semantic reasoning to enhance the reasoning capabilities of our model.

### 3 Methods

The overall framework of BuboGPT is presented in Figure 1. As the Figure shown, we perform joint multi-modal understanding and chatting for text, vision and audio, which is achieved by learning a shared representation space that aligns well with pre-trained Vicuna [16]. We also build an off-theshelf visual grounding pipeline to explore the fine-grained relation between different visual objects and modalities.

#### 3.1 Visual Grounding Pipeline

To explore the relation between different visual objects and input modalities, we further build the visual grounding pipeline, composed of a tagging module, a grounding module and a entity-matching module, as shown in Figure 2. Concretely, for a given image, we first use Recognize Anything Model (RAM) [12], a strong model based on Swin-transformer [28] for image tagging to generate relevant candidate tags, denoted as {t1,t2,...,tn

t}, where ti is the i-th semantic tag and nt is the number of detected tags. We then connect the tags with comma to form the prompt “t1,t2,...,tn

” and use the Grounding DINO [13], a open-set object detection model with referring textual queries to identify the visual entities and the corresponding boxes relevant to the tags. Followed by Segment Anything Model (SAM) [11], the boxes are taken as prompt to get fine-grained semantic masks.

t

With the tagging and grounding module, we then obtain all the visual entities and the corresponding grounding information, denoted as {(e1,g1),(e2,g2),...,(en

)}, where ei,gi are separately the i-th visual entities and grounding information (i.e. boxes and masks), ne is the number of entities. To model the relation between different visual entities and input modal-

,gn

e

e

ities, we employ the text output to of our multi-modal LLM as the bridge and build a entitymatching module based on GPT-4 to retrieve the matching pairs. To construct the prompt template

“<List>e1,e2,...,en

</List>,<Text>to</Text>”, we utilize the powerful LLM for reasoning and retrieve the matching pairs, which reflects the relation between visual entities and input modalities.

e

#### 3.2 Multi-Modal LLM Training

BuboGPT considers the interaction between three modalities, i.e., text, vision and audio. It aligns a vision encoder and an audio encoder with the LLM with a Q-former for each modality. More specifically, we utilize the visual encoder together with the pre-trained Q-Former in BLIP-2 [15] and audio encoder in ImageBind [14] for visual and audio perception. For joint understanding over multiple modalities, we employ Vicuna as the LLM. We use a linear projection layer to connect the modality Q-Former with the LLM. To effectively train such a model, we develop the following two-stage training scheme. The modality encoders and Vicuna model with be fixed throughout the training procedure.

- Stage 1: Single-modal Pre-training Similar to MiniGPT-4 [2], the first stage is designed to align the output of the linear projection layer to the word embedding space of the LLM. This is achieved by training the modality Q-Former and linear projection layer on a large number of modality-text paired data. For visual perception, we only train the projection layer for image captioning with the Q-Former from BLIP2 fixed. For audio understanding, we jointly train the Q-Former and the projection layer

iii

What is the animal?

[Figure 7]

Multi-modal LLM

Entity-matching Module

A brown dog

[Figure 8]

[Figure 9]

Tagging Module

Grounding Module

brown | chase | sheepdog | corgi | dog | field | … | grassy | park | run

['dog', 'sheepdog', …, 'field', 'corgi', 'grassy', 'brown dog', 'grass', 'park']

Figure 2: The pipeline of visual grounding that is composed of a tagging module, a grounding module and a entity-matching module.

###Human: <Vision><ModalityHere></Vision> What is the image? ###Assistant: ###Human: <Audio><ModalityHere></Audio> Pay attention to the audio and describe what you notice. ###Assistant: ###Human: <Vision><ModalityHere></Vision> <Audio><ModalityHere></Audio> Please find the source that emits the given sound in this image. ###Assistant: ###Human: <Vision><ModalityHere></Vision> <Audio><ModalityHere></Audio> Are the audio and image related to each other? What are they? ###Assistant:

Table 1: Instruction-following prompt examples for various input sources.

for audio captioning. There will not be any prompt used for both settings, the model just take the corresponding image or audio as input and predict the corresponding caption.

- Stage 2: Multi-Modal Instruct Tuning This stage aims to equip the multi-modal LLM with the ability to understand human instructions such that it can generate proper responses based on the given modality signal. To this end, we curate a high-quality multi-modal instruction-following dataset, which contains image-text, audio-text and image-audio-text pairs. To make the model adapt to arbitrary combination of input modalities, we design a general prompt as: ###Human:

<Vision><ModalityHere></Vision> <Audio><ModalityHere></Audio> <instruction> ###Assistant:. <Vision></Vision> and <Audio></Audio> are special identifiers for image and audio input.

<ModalityHere> is going to be replaced by a sequence of image or audio tokens before feeding into the LLM. <instruction> is the human instruction related to the input sigals for the LLM to assist on. We list a few examples for different combinations of input modalities in Tab. 1. We empirically accessed that when only positively paired image-audio data are included in this stage, the model always assumes the image and audio are related to each other even though random sampoles are used

- at test time. Therefore, we manually create some negative pairs and asking the LLM to tell what are they respectively. The experiments show that introducing such negative paired data is able to overcome this problem significantly. We leave the creation of datasets in the next section.

### 4 Datasets

#### 4.1 Pretraining Datasets

Following MiniGPT-4 [2], we use a combined dataset of CC3M [23], CC12M [24], SBU [29] and LAION [26] to train the visual projection layer, resulting in a total of 130 million image-text pairs. For audio, we mainly use the WaveCaps [30] dataset, which contains 403,050 audio clips with average duration of 67.59 seconds and average caption length of 7.8 words. It combines four

iv

datasets including FreeSound (262,300) [31], BBC Sound Effects (31,201)1, SoundBible (1,231)2 and AudioSet strongly-labelled subset (108,317) 3, and transform their raw-descriptions into captions with ChatGPT.

- 4.2 Instruction-Tuning Datasets

- 4.2.1 Image-Text Dataset

We employ two previously published datasets for visual instruct tuning. The first one is released by MiniGPT-4, which contains 3,439 high-quality text-image pairs. The second one provided by LLaVA [6] is curated from 158K samples based on the COCO dataset, including three types of instructions, i.e., converstaions (58K), detailed description (23K) and complex reasnoning (77K).

- 4.2.2 Audio-Text Dataset

When it comes to the field of audio understanding, we also need to conduct the instruction-tuning operation on the audio Q-former. However, unlike vision-language understanding, a severe need still exists for high-quality and well-organized instruction-tuning datasets in this field. To this end, we generate a series of expressive and descriptive data to facilitate this process.

Specifically, we first investigate different kinds of existing audio caption datasets and select Clotho [32] as the original dataset to make the description extension. The reason can be explained in two folds. On the one hand, it has a moderate and acceptable scale to act as an instruction-tuning dataset, and the semantic range of audio is large enough. On the other hand, every audio has five short captions from different annotators, covering various possible scenes related to the audio and increasing the diversity of descriptions.

After obtaining the original data, we need to rewrite the short captions into descriptive and imaginative paragraphs. Considering the extraordinary ability of GPT-4 in the field of few-shot learning, text generation, and complex reasoning, we utilize it to help us automatically assemble short captions into long descriptions to mitigate the reliance on human annotation. The final description is expected to cover all the related original captions. For example, given the series of captions [“A person is turning a map over and over.”, “A person is very carefully wrapping a gift for someone else.”, “A person is very carefully wrapping a gift for someone else.”, “He sighed as he turned the pages of the book, stopping to scan the information.”, “Papers are being turned, stopped, then turned again, and someone is breathing.”], the description paragraph is expected to be “A person is repeatedly flipping some papers. They might be reading a book, flipping through a map, or wrapping presents. Judging from the repeated flipping sounds, they are concentrating on repeating this action.”. We design a task-related prompt and construct some few-shot examples like this to promote the in-context reasoning process. As a result, we collect a novel dataset Clotho-Detail 4 for instruction-tuning in audio understanding, which contains 3938 items and the average length of descriptions is 52.70 words.

- 4.2.3 Audio-Image-Text Dataset

Positive Set In order to further empower our model with the comprehensive ability of multi-modal reasoning, we apply a group of audio-image pairs to help the model to understand the correspondence between the audio and its source. Among the existing audio-vision datasets, VGGSS [33] turns out to be a better choice in this process. It covers a wide range of sounding objects, and the audio only relates to a specific region in the corresponding image. Therefore, we retrieve all the data cases and use a group of fixed templates to wrap the corresponding class labels into natural sentence descriptions. As a result, we generate a total of 5,158 <audio, image, text> pairs to act as the triple-modality instruction tuning dataset 5.

- 1https://sound-effects.bbcrewind.co.uk/
- 2https://soundbible.com/
- 3https://research.google.com/audioset/download_strong.html
- 4https://huggingface.co/datasets/magicr/BuboGPT/blob/main/Clotho-detail-annotation.

json

- 5https://huggingface.co/datasets/magicr/BuboGPT/blob/main/vggss-instruction-tuning.

json

v

Negative Set As discussed in the method section (Sec. 3.2), relying solely on the above dataset causes the LLM fail to recognize irrelevant audio-image pairs. Therefore, we construct negative <audio, image, text> pairs such that <text> gives independent descriptions for audio and image inputs. The audio is randomly sampled from the audio-text dataset presented in Sec. 4.2.2, while the image is randomly sampled from the MiniGPT-4 dataset discussed in Sec. 4.2.2. The text is constructed by concatenating the two captions that starts with “The image” and “The audio”.

### 5 Experiment Results

In this section, we aim to answer the following two questions: 1) whether our BuboGPT is able to provide accurate and instructive visual grounding when the inputs contain images? 2) whether the modal is able to perceive arbitrary combinations of modalities and generate proper responses.

We first consider using a single image as input for fine-grained visual understanding with grounding. As shown in Fig. 3-7, the model can accurately associate textural words or phrases with image regions in various scenarios with different complexities. Then, when a single audio clip is provided for audio understanding, BuboGPT gives informative descriptions covering nearly all acoustic parts included, even when some audio fragments are too short for humans to notice, see Fig. 8-13 for details. Next, we show that the model can perform sound localization with a matched audio-image pair provided, which gives a perfect example for aligned audio-image understanding. As illustrated in Fig. 14-17, the model is going to generate an overall description for both input image and audio, then point out which object in the image emits the sound after reasoning. It is worth noting that our model can give correct predictions when we provide different audio and keep the image unchanged. This demonstrates that our model can understand both modalities comprehensively rather than generate answers with prior bias from a single modality. Moreover, we empirically accessed that if the model is only tuned with well-aligned image-audio data, it actually fails to discriminate when an irrelevant image and audio pair is provided, resulting in a non-factual response that is not consistent with the given image or audio (Fig. 20). After introducing the negatively paired dataset as discussed in Sec. 4.2.3, the model can tell whether the image and audio are relevant to each other and generate high-quality response for arbitrary audio-image understanding, as evidenced by Fig. 18-19.

### 6 Conclusion

In this report, we propose a multi-modal LLM, BuboGPT, which is capable of joint multi-modal understanding including image, audio and text, and perform more fine-grained understanding of multi-modal inputs by exploring the relation between different visual objects and modalities. We also build a high-quality instruction tuning dataset and the experiments show that BuboGPT achieves impressive visual grounding abilities during multi-modal chat, even when arbitrary combinations of multi-modal inputs are provided, whether matched or unmatched.

### 7 Limitations

Language hallucination. Following prior works, our method is based on the pre-trained Vicuna model, which inherits the limitations of LLMs including generating non-existent knowledge or non-factual information. The problem might be resolved by training with more high-quality data and developing trustworthy LLMs.

Inadequate capacities of Grounding QA. Since the connection between grounding results and different modalities is built upon text conversations without extra training, the capacities of QA on specific objects remain limited. The model can be improved by introducing fine-grained visual grounding datasets and considering the spatial location as extra input.

vi

[Figure 10]

[Figure 11]

[Figure 12]

Please describe the image.

[Figure 13]

The image shows a ① cat wearing ② sunglasses and sitting on an orange ③ float in a ④ swimming pool . The cat appears to be relaxed and enjoying its time in the water.

[Figure 14]

Is this image funny and why?

[Figure 15]

This image can be considered funny because it depicts a cat , a typically indoor pet, wearing sunglasses and laying in a swimming pool , which is a place that is typically associated with water activities for humans. It is an unexpected and unusual scene, which can make people chuckle. Additionally, the cat's expression and relaxed posture make it seem comfortable and content, adding to the humor of the image.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

① cat ② sunglasses ③ float ④ swimming pool

[Figure 24]

[Figure 25]

[Figure 26]

Feel free to chat with Bubo-GPT !

Figure 3: Example 1 of fine-grained visual understanding with grounding.

vii

[Figure 27]

[Figure 28]

[Figure 29]

what is the image?

[Figure 30]

The image shows ① two black ② swans swimming in the ③ water and forming a heart shape with their ④ beak. The swans appear to be engaging in a playful or affectionate gesture as they move around each other gracefully, displaying their elegant black feathers and white beaks. It captures a moment of harmony and beauty in the natural world, where these creatures appear to be enjoying each other's company.

[Figure 31]

what is the meaning of the behaviour?

[Figure 32]

The behaviour of the black swans in the image is an act of affection, usually between a mated pair or between friends . It is a visual display of bonding and mutual affection. Swans are social birds and engage in various social behaviours to strengthen their relationships. In this particular case, the two black swans are standing next to each other and creating a heart shape by joining their beaks together. This behaviour might have been observed during breeding season or as a way of communication and recognition among mated pairs. Regardless of the specific context, the heart shape represents the bond between the two swans, and their act of creation reflects the beauty and harmony that nature can bring to our lives.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

① two ② swans ③ water ④ beak

[Figure 41]

[Figure 42]

[Figure 43]

Feel free to chat with Bubo-GPT !

Figure 4: Example 2 of fine-grained visual understanding with grounding.

viii

[Figure 44]

[Figure 45]

[Figure 46]

Describe the image.

[Figure 47]

The image shows a ① small kitten in a white and brown color pattern laying in the bottom of a tan and brown leather purse or small ② tote bag on a ③ wooden table . The kitten appears to be curled up and comfortably nestled in the bag. The bag itself has a ④ strap on the side that allows the bag to be held and carried, likely making it an attractive and convenient option for the cat to nap and rest.

[Figure 48]

Can you tell me why the cat laying down inside the bag?

[Figure 49]

The kitten is laying down inside the purse or bag because it may feel comfortable or safe inside the bag. Cats often curl up and sleep in small spaces, like bags or boxes , as it provides a sense of security and comfort. The purse, with its zippered opening, may allow the cat to move around and get comfortable in the bag without being too confined or frightened by its size. Additionally, the soft material of the purse may mimic the texture of a cat's bedding , making it a desirable sleeping spot.

[Figure 50]

Any other possible reason?

[Figure 51]

Another possible reason why the kitten is laying down inside the bag is that it might have found a cozy and sheltered place to nap. The bag can act as an enclosed space that provides a sense of privacy for the cat, allowing it to rest without fear of being disturbed or startled. The soft interior of the bag might also be inviting for the kitten to rest and snuggle into. It is important to remember that cats love to curl up and hide , and a bag can offer them a comfortable space to do so. Overall, there are various reasons why a kitten might choose to sleep inside a purse or bag, and the comfort and privacy it provides can be a significant factor.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

① small kitten ② tote bag ③ wooden table ④ strap

[Figure 60]

[Figure 61]

[Figure 62]

Feel free to chat with Bubo-GPT !

Figure 5: Example 3 of of fine-grained visual understanding with grounding.

ix

[Figure 63]

[Figure 64]

[Figure 65]

what is the image?

The image shows a ④ woman with a black fur and white hair sitting on the ③ grass , holding the back of a ② chimpanzee. The woman is wearing a white shirt and is smiling. The chimpanzee is reaching its ① mouth towards the woman's mouth and the woman is making a kissing sound with her mouth open.

[Figure 66]

[Figure 67]

What does the image suggest?

[Figure 68]

The image suggests that humans and animals can have positive relationships and enjoy each other's company. It shows a woman playing with a chimpanzee , suggesting that they are engaging in a form of play or bonding activity. The image may suggest that it is possible for humans and animals to communicate and interact with each other in a friendly manner.

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

① mouth ② chimpanzee ③ grass ④ woman

[Figure 77]

[Figure 78]

[Figure 79]

Feel free to chat with Bubo-GPT !

Figure 6: Example 4 of of fine-grained visual understanding with grounding.

x

[Figure 80]

[Figure 81]

[Figure 82]

what is the image?

The image is a small ① frog holding a ② banjo , with its legs spread out on either side of the banjo. The frog is sitting on top of a ③ moss-covered tree stump.

[Figure 83]

[Figure 84]

Anything interesting?

[Figure 85]

Yes, there are several interesting features in the image. The first is the frog's intricate posing on the instrument. The frog appears to be playing the instrument with its legs , which adds an element of creativity and playfulness to the image. The second interesting aspect is the forest setting, which adds a sense of nature and serenity to the image. Finally, the mossy ground adds a touch of whimsy and makes the overall scene appear more rustic and charming.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

① frog ② banjo ③ moss-covered tree stump

[Figure 93]

[Figure 94]

[Figure 95]

Feel free to chat with Bubo-GPT !

Figure 7: Example 5 of of fine-grained visual understanding with grounding.

xi

[Figure 96]

[Figure 97]

[Figure 98]

Audio 1: chime, bird, breeze

[Figure 99]

Please describe this audio.

[Figure 100]

In this audio, a bell tower chimes in the background, creating a serene atmosphere, blending with the sounds of a breeze gently blowing through trees and birds singing in the distance . The combination of these elements provides a calming ambiance with a touch of natural harmony.

[Figure 101]

[Figure 102]

[Figure 103]

Feel free to chat with Bubo-GPT !

- Figure 8: Example 1 of audio understanding. Audio 1 can be accessed via this link.

[Figure 104]

[Figure 105]

Please describe this audio.

[Figure 106]

The audio features a clear, melodic tune that resonates within the scene. The performance sounds as if it is happening in a spacious area, with the musician playing their instrument and creating a soothing atmosphere. The melody is made up of a series of chords, with different notes being played in unison. This audio captures the essence of a person creating music in an unspecified setting.

[Figure 107]

Feel free to chat with Bubo-GPT !

[Figure 108]

[Figure 109]

[Figure 110]

Audio 2: music, box, melody

[Figure 111]

- Figure 9: Example 2 of audio understanding. Audio 2 can be accessed via this link.

xii

[Figure 112]

[Figure 113]

[Figure 114]

Audio 3: people talking

[Figure 115]

Please describe this audio.

[Figure 116]

The audio depicts a lively atmosphere in a restaurant or gathering, where a group of people engage in conversations. Amidst the chatter, one individual intermittently joins the conversation. The multitude of voices creates a bustling, dynamic atmosphere, capturing the essence of social interaction and camaraderie.

[Figure 117]

[Figure 118]

[Figure 119]

Feel free to chat with Bubo-GPT !

- Figure 10: Example 3 of audio understanding. Audio 3 can be accessed via this link.

[Figure 120]

[Figure 121]

Please describe this audio.

[Figure 122]

The audio captures the sound of someone typing rapidly and energetically on a keyboard, creating a rhythmic pattern of clicks and keyboard sounds. This intense activity could be associated with working on a computer or writing a long-form document.

[Figure 123]

Feel free to chat with Bubo-GPT !

[Figure 124]

[Figure 125]

[Figure 126]

Audio 4: keyboard typing

[Figure 127]

[Figure 128]

Could this audio come from a swimming pool?

[Figure 129]

No, the audio does not suggest the sound of swimming or water. The keyboard clicking and typing sounds dominate the scene, as they would in an office or work environment.

- Figure 11: Example 4 of audio understanding. Audio 4 can be accessed via this link.

xiii

[Figure 130]

[Figure 131]

[Figure 132]

Audio 5: heavy rain

[Figure 133]

Please describe this audio.

[Figure 134]

In the audio, rain is the primary focus, with heavy rain falling down and hitting the ground with force. This creates a vivid and immersive atmosphere of a stormy day, with rain pouring down and creating a dynamic soundscape. Additionally, there may be some distant sounds of traffic or vehicles passing by, adding further context to the ambiance.

[Figure 135]

If I’m in the scene corresponding to the audio, what should I do first?

[Figure 136]

Based on the audio, it appears to be a heavy rainstorm. If you’re in the scene, you might first take cover from the rain, potentially by seeking shelter under a roof or an overhang, or by running for cover. Alternatively, you might simply enjoy the sound of the rain and take in the experience.

[Figure 137]

[Figure 138]

[Figure 139]

Feel free to chat with Bubo-GPT !

- Figure 12: Example 5 of audio understanding. Audio 5 can be accessed via this link.

xiv

[Figure 140]

[Figure 141]

[Figure 142]

Audio 6: breaking glasses

[Figure 143]

Please describe this audio.

[Figure 144]

The audio features a cacophony of loud crashes and the sound of glass shattering, suggesting a situation involving multiple instances of breaking glass. The sound is consistent with what could be produced by breaking glass objects, such as breaking wine glasses, a car back window, or breaking glass bottles. It could also represent the impact of breaking a pane of glass when a large object strikes it, producing a loud, jarring noise.

[Figure 145]

Does this sound portend any potential danger?

[Figure 146]

The audio doesn’t explicitly convey any immediate danger, but the sound of breaking glass can be unsettling and potentially alarming. The shattering could be the result of an accident or intentional act, and the level of danger would depend on the situation. The sound of breaking glass can also be associated with certain types of events like explosions or breaking windows during a storm. In any case, it’s important to pay attention and take appropriate precautions.

[Figure 147]

[Figure 148]

[Figure 149]

Feel free to chat with Bubo-GPT !

- Figure 13: Example 6 of audio understanding. Audio 6 can be accessed via this link.

xv

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Audio 7: dog barking

[Figure 155]

Try to find the sounding source in the image according to the given audio and generate a detailed description about it.

[Figure 156]

Based on the provided image of a ① boy riding a ② bicycle , it appears that the source of the audio is the sound of a ③ dog barking. The boy appears to be on a ④ dirt road , likely traveling through a wooded area, as the tree branches are visible in the background. The dog, which is running alongside the boy's bicycle , is barking in an excited manner, suggesting that it may be chasing after the boy or simply running alongside him. This dynamic duo's adventurous journey through nature creates a vivid and lively scene in the image.

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

##### ① boy ② bicycle ③ dog ④ dirt road

[Figure 166]

[Figure 167]

[Figure 168]

Feel free to chat with Bubo-GPT !

- Figure 14: Example 1 of aligned audio-image understanding. Audio 7 can be accessed via this link.

xvi

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Audio 8: bicycle bell

[Figure 174]

Try to find the sounding source in the image according to the given audio and generate a detailed description about it.

The sound heard in the image appears to be a ① bicycle bell, which is the device located near the right rear wheel of the bike , and is associated with the action of riding the bicycle, particularly when the rider accelerates or slows down. The ② person on the bicycle is wearing a blue shirt and is either wearing ③ a hat or has one on their head.

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

##### ① bicycle ② person on the bicycle ③ a hat

[Figure 184]

[Figure 185]

[Figure 186]

Feel free to chat with Bubo-GPT !

- Figure 15: Example 2 of aligned audio-image understanding. Audio 8 can be accessed via this link.

xvii

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Audio 9: hair dryer

[Figure 192]

Try to find the sounding source in the image according to the given audio and generate a detailed description about it.

[Figure 193]

The sound in the image can be attributed to a ① hair dryer . The ② woman in the picture is holding a hair dryer in her ③ hand and using it to dry her ④ cat 's fur . The blowing air from the hair dryer can be heard, providing an auditory representation of the activity taking place in the image.

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

##### ① hair dryer ② woman ③ hand ④ cat

[Figure 203]

[Figure 204]

[Figure 205]

Feel free to chat with Bubo-GPT !

- Figure 16: Example 3 of aligned audio-image understanding. Audio 9 can be accessed via this link.

xviii

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

Audio 10: cat meowing

[Figure 211]

Try to find the sounding source in the image according to the given audio and generate a detailed description about it.

[Figure 212]

[Figure 213]

##### The audio in the image is produced by a ① cat hissing at the

② person , who is either holding or near the cat. This can be identified through the sound of the cat's loud , agitated hissing as well as the context of the person interacting with the cat.

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

##### ① cat ② person

[Figure 220]

[Figure 221]

[Figure 222]

Feel free to chat with Bubo-GPT !

- Figure 17: Example 4 of aligned audio-image understanding. Audio 10 can be accessed via this link.

xix

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Audio 11: continual bell chime

[Figure 228]

Please tell me a story based on the given audio and image.

[Figure 229]

[Figure 230]

In this audio, a clock strikes hour after hour, creating a haunting and mournful sound. The sound echoes through the forest, wafting through the trees and into the distance. The image depicts a ① mink perched on a mossy ② tree trunk. The mink stares intently into the distance, possibly listening to the toll of the clock, representing the passing of time. This atmosphere might suggest a sense of solitude, introspection, or reminiscence.

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

##### ① mink ② tree trunk

[Figure 236]

[Figure 237]

[Figure 238]

Feel free to chat with Bubo-GPT !

- Figure 18: Example 1 of arbitrary audio-image understanding. Audio 11 can be accessed via this link.

xx

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

Audio 12: children playing

[Figure 244]

Please tell me a story based on the given audio and image.

[Figure 245]

In this audio, there is a group of kids playing on the ① beach

during ② sunset , enjoying the beautiful view and the gentle breeze. They can be seen running, playing, and laughing together, possibly creating cheerful and light-hearted memories that could last a lifetime. The serene environment, characterized by soothing sounds of seagulls and the ③ gentle waves , provides a soothing backdrop to the lively and innocent activities.

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

##### ① beach ② sunset ③ gentle waves

[Figure 253]

[Figure 254]

[Figure 255]

Feel free to chat with Bubo-GPT !

- Figure 19: Example 2 of arbitrary audio-image understanding. Audio 12 can be accessed via this link.

xxi

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Audio 11: continual bell chime

[Figure 261]

Please tell me a story based on the given audio and image.

In the image, we see a small brown squirrel perched on a branch of a tree in the forest . The sound of wind rustling through the leaves provides the background ambiance. Suddenly, a bird starts chirping and singing a melody. The chirping gradually becomes louder and more persistent, creating a harmonious and peaceful atmosphere. As the bird sings, the squirrel starts moving around on the branch. Its ears perk up, and its tail is alert. Then, the squirrel suddenly stops and listens intently, its eyes focused on the bird's melody. The bird's song reaches its peak and then gradually fades away, leaving the forest silent. The squirrel remains on the branch, looking out into the woods, as if contemplating the beauty of nature and the solitude of its surroundings. The contrast between the peaceful bird song and the serene silence of the forest creates a captivating scene. This combination of the image and audio represents a moment of stillness and awe as the squirrel and bird share the beauty of their respective experiences in the forest.

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

Feel free to chat with Bubo-GPT !

- Figure 20: Failure case of arbitrary audio-image understanding without using negative audio-image pairs.

xxii

### References

- [1] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652,

2021. i

- [2] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. i, ii, iii, iv
- [3] Feilong Chen, Minglun Han, Haozhi Zhao, Qingyang Zhang, Jing Shi, Shuang Xu, and Bo Xu. X-llm: Bootstrapping advanced large language models by treating multi-modalities as foreign languages. arXiv preprint arXiv:2305.04160, 2023. i, ii, iii
- [4] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. i, ii, iii
- [5] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint

- arXiv:2303.16199, 2023. i, ii, iii

[6] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint

- arXiv:2304.08485, 2023. i, ii, iii, v

- [7] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023. i, ii, iii
- [8] Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instructionfollow them all. arXiv preprint arXiv:2305.16355, 2023. i, ii, iii
- [9] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. arXiv preprint arXiv:2306.07207,

2023. i, ii, iii

- [10] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424,

2023. i, ii

- [11] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. ii, iii
- [12] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. arXiv preprint arXiv:2306.03514, 2023. ii, iii
- [13] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. ii, iii
- [14] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15180–15190, 2023. ii, iii
- [15] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. ii, iii
- [16] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2023. ii, iii
- [17] OpenAI. Gpt-4 technical report, 2023. ii, iii
- [18] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. ii
- [19] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022. ii
- [20] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023. ii
- [21] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. arXiv preprint arXiv:2303.17580, 2023. ii

xxiii

- [22] Zhiyang Xu, Ying Shen, and Lifu Huang. Multiinstruct: Improving multi-modal zero-shot learning via instruction tuning. arXiv preprint arXiv:2212.10773, 2022. iii
- [23] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565,

2018. iii, iv

- [24] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3558–3568, 2021. iii, iv
- [25] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. Advances in neural information processing systems, 24, 2011. iii
- [26] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. iii, iv
- [27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. iii
- [28] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. iii
- [29] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022. iv
- [30] Xinhao Mei, Chutong Meng, Haohe Liu, Qiuqiang Kong, Tom Ko, Chengqi Zhao, Mark D Plumbley, Yuexian Zou, and Wenwu Wang. Wavcaps: A chatgpt-assisted weakly-labelled audio captioning dataset for audio-language multimodal research. arXiv preprint arXiv:2303.17395, 2023. iv
- [31] Frederic Font, Gerard Roma, and Xavier Serra. Freesound technical demo. In Proceedings of the 21st ACM international conference on Multimedia, pages 411–412, 2013. v
- [32] Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen. Clotho: an audio captioning dataset. ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 736–740, 2019. v
- [33] Honglie Chen, Weidi Xie, Triantafyllos Afouras, Arsha Nagrani, Andrea Vedaldi, and Andrew Zisserman. Localizing visual sounds the hard way. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16862–16871, 2021. v

xxiv

