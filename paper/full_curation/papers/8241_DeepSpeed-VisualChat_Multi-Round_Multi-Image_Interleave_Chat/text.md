# arXiv:2309.14327v3[cs.CV]29Nov2023

## DeepSpeed-VisualChat: Multi Round Multi Images Interleave Chat via Multi-Modal Casual Attention

Zhewei Yao, Xiaoxia Wu, Conglong Li, Minjia Zhang, Heyang Qin Olatunji Ruwase, Ammar Ahmad Awan, Samyam Rajbhandari, Yuxiong He

DeepSpeed of Microsoft

Abstract

Most of the existing multi-modal models, hindered by their incapacity to adeptly manage interleaved image-and-text inputs in multi-image, multi-round dialogues, face substantial constraints in resource allocation for training and data accessibility, impacting their adaptability and scalability across varied interaction realms. To address this, we present the DeepSpeed-VisualChat framework, designed to optimize Large Language Models (LLMs) by incorporating multi-modal capabilities, with a focus on enhancing the proficiency of Large Vision and Language Models in handling interleaved inputs. Our framework is notable for (1) its open-source support for multi-round and multi-image dialogues, (2) introducing an innovative multi-modal casual attention mechanism, and (3) utilizing data blending techniques on existing datasets to assure seamless interactions in multi-round, multi-image conversations. Compared to existing frameworks, DeepSpeed-VisualChat shows superior scalability up to 70B parameter language model size, representing a significant advancement in multi-modal language models and setting a solid foundation for future explorations.1

### 1 Introduction

State-of-the-art large language models (LLMs) like GPT [7, 28] have showcased exceptional prowess in myriad text generation and comprehension tasks, especially when subjected to zero-/few-shot learning. Once these models undergo supervised fine-tuning or reinforcement learning combined with human feedback, their proficiency in versatile interactive challenges—ranging from coding tasks [10] to quantitative reasoning [17], mathematical proofs [14, 43], and AI chatbot interactions [27, 3, 29, 44]—becomes comparable to human experts.

Seeking to transcend the bounds of text-only processing inherent to LLMs, numerous researchers have made strides in endowing these models with multi-modal capabilities. These advances span across various modalities such as images, audios, and videos, often achieved via feature alignment and model alterations [9, 11, 48, 23, 18, 5, 12]. Notably, among these multi-modal endeavors, large vision and language models (LVLMs) have garnered significant interest [48, 23], mainly owing to their potential in facilitating comprehensive visual-textual understanding.

Current frameworks and studies largely focus on either (1) tasks related to individual images, like visual question answering and captioning [23], or (2) handling multiple images but requiring concurrent input [18]. Neither approach adeptly manages interleaved image-and-text inputs. The QWen-VL framework [5], an extension of the LLaVA architecture [23], makes progress in this direction. However, its training costs prove prohibitive for many research labs, and it withholds its training data. In parallel, the SparklesChat model [12], annotated by GPT4, relies on continuous training with MiniGPT4 [48] due to its limited dataset. Both QWen-VL and SparklesChat adopt prevailing LVLMs designs without innovative architectural exploration.

1Code will be released soon as a part of https://github.com/microsoft/DeepSpeedExample

[Figure 1]

Figure 1: An example of DeepSpeed-VisualChat.

Additionally, in multi-image contexts, their performance is found lacking, even with significant training investments 2, as shown in our comparisons Figure 9.

While larger language models typically demonstrate superior generation abilities compared to their smaller counterparts, prevailing frameworks [18, 23, 48, 5] predominantly concentrate their efforts on LLMs with 7 or 13 billion parameters (e.g., LLaMa-2-7B or LLaMa-2-13B [39]). This focus restricts the exploration of the extensive capabilities inherent to larger LLMs.

To address the aforementioned challenges, we introduce the DeepSpeed Multi Round and Multi Images Chat framework (DeepSpeed-VisualChat), offering several key contributions:

- • Fully Open-Sourced Multi-round Multi-image Framework: DeepSpeed-VisualChat, one of the pioneering fully open-sourced frameworks, enables multi-round and multi-image dialogues, accommodating interleaved text-and-image inputs, as visualized in Figure 1.
- • Multi-Modal Casual Attention (MMCA): We devise a novel MMCA for multi-modal models that independently computes attention weights across various modalities. MMCA attains objectives analogous to conventional cross-attention mechanisms [18], yet offers enhanced casual attention interpretations for generative tasks, eliminating the need for additional modules or parameters, and it presents superior training data efficiency compared to standard casual attention [48, 23].
- • Data Blending for Interleaved Inputs: To facilitate conversations with interleaved modalities, DeepSpeed-VisualChat employs assorted data blending techniques on existing datasets, overcoming the shortage of interleaved text-and-image inputs in most available open-sourced datasets.
- • Unprecedented Scalability: We leverage the DeepSpeed framework [31] to amplify our training with a 2B visual encoder from [13] and a 70B language decoder from LLaMa-2 [39], illustrating the remarkable scalability of our framework.

These innovations demonstrate our commitment to progressing multi-modal conversational AI models, ensuring enhanced interoperability, attention mechanism sophistication, scalable solutions, and comprehensive dataset utilization.

2We count the pretraining cost of MiniGPT4 as a part of SparklesChat

### 2 Related Work

Multi-modal models, especially those focusing on vision-language integrations, typically fall into two distinct categories: dual-encoder-based models [38, 26, 36, 47, 30, 22, 16, 21, 42, 6, 40, 45, 41, 20, 8, 30, 15, 33, 32, 46, 35, 24, 34], and models comprising visual encoders and textual decoders [2, 19, 23, 18, 48, 11, 37, 5, 4, 12]. Our work is associated with the latter, often referred to as Large Visual Language Models (LVLMs), thus our discussion predominantly revolves around LVLMs.

Most implementations of LVLMs deploy one of two architecture styles: (1) The Flamingo design [2, 18, 4] incorporates cross-attention, introducing new parameters to LLMs to interlink visual and textual elements. (2) The Flamingo design [2, 18, 4] incorporates cross-attention, introducing new parameters to LLMs to interlink visual and textual elements. Although both designs effectively assimilate visual information and generate textual content, their advantages and drawbacks are manifold. The Flamingo design necessitates extensive training/inference memory and fewer data due to the introduction of numerous new parameters. Conversely, the MiniGPT4 design, while less memory-intensive, is more data-dependent to effectively align visual and textual features. Consequently, an emerging query is whether a novel architecture can harmonize the introduction of fewer new parameters with data efficiency.

Despite the substantial advancements achieved by existing LVLMs, certain aspects, particularly multiround multi-image conversation involving interleaved image and text input, remain unaddressed. Works like [23] predominantly concentrate on single image conversation, and [18] necessitate simultaneous input of all images, limiting their applicability in conventional conversational applications.

The paucity of data pertinent to these scenarios has led researchers to explore available data to facilitate new applications. A contemporary work, SparklesChat [12], exploits the GPT-4 [1] API to synthesize several thousands of multi-round multi-image data in a unique format. However, SparklesChat does not innovate any new architectural modifications and relies on the pre-trained model, MiniGPT4 [48], thereby incurring additional training costs and complexity. Moreover, SparklesChat does not exhibit any superior performance in unseen multi-image tasks compared to DeepSpeed-VisualChat, even without utilizing multi-image interleaved chat data. Refer to Figure 9 and A.1 for detailed examples.

### 3 Method

Our model architecture is built on the structure of MiniGPT4 [48, 23], as depicted in Figure 2. Specifically, we maintain the entirety of the visual encoder and the whole language model, with the exception of the embedding layer, in a frozen state. Thus, the only trainable parameters within our model are the visual feature projection layer (a linear layer) and the language model’s embedding. In total, our set of trainable parameters ranges around O(10M) to O(100M), primarily contingent on the size of the embedding layer.

Diverging from the previous MiniGPT4 architecture, we substitute the conventional casual attention mechanism with our proposed multi-modal casual attention mechanism (refer to Section 4.1). This modification solely alters the computation of casual attention and does not incorporate any new parameters.

We adopt a unified instruction tuning format for all experiments, and the template is shown in Figure 3. It is crucial to note that we do not employ any special tokens as our prefix; for example, “### Image i” is not a special token, and the tokenizer interprets it as a regular string.

[Figure 2]

Figure 2: Model Structure. A pre-trained vision encoder encodes an image which is then projected through a linear layer to align with the hidden dimension of the text embedding layer’s output. These different inputs are subsequently merged and forwarded to language models like LLaMa-2 powered by our new Multi-Modal Casual Attention (MMCA) mechanism. Here, both the vision encoder and the language model are frozen.

<System Insturction > % You are a powerful vision −language assistant .

- ### Image 1: <image> % some image , e . g . , cat −1.png ### Question : <question> % please describe the image . ### Answer : <answer> % It ’ s a cute black cat .
- ### Image 2: <image> % some image , e . g . , cat −2.png
- ### Image 3: <image> % some image , e . g . , cat −3.png ### Question : <question> % What’ s difference between three cats ? ### Answer : <answer> % The color of three cats are d i f f e r e n t .

. . .

- Figure 3: Here <System Instruction>, <question>, <answer> can be simply replaced by text, and <image> can be replaced by real image tokens. The content after “%” is an example.

[Figure 3]

- Figure 4: Different Attention Mechanisms: Examine the differing attention mechanisms using an input sentence "User: Please describe the image." coupled with three Image tokens (I-token1, I-token2, I-token3). On the left, we demonstrate standard causal attention [48, 5], treating image tokens as text. In the middle, we present cross attention applied to images, while maintaining standard causal attention for text tokens. On the right, we illustrate our innovative multi-modal attention proposal where image tokens only perform self-attention, and text tokens attend to text/image tokens independently, highlighted with an orange mask. This mechanism is defined by: softmax(QKT ⊙ M1) + softmax(QKT ⊙ M2) with Q and K as query and key, M1 = [M == 1], and M2 = [M == 2], with M ∈ R10×10 in this case.

In alignment with the recent trend of instruction fine-tuning, the final loss of our model is calculated solely on “<answer>”, as illustrated in Figure 3. If multiple conversations are present, we compute the loss for all corresponding “<answer>” instances.

Throughout the paper, unless specifically mentioned, we employ the LLaMa-2 family as our language and utilize the extracted (and frozen) visual encoder from QWen-VL [5] as our visual encoder, which accepts 448x448 images and produces 256 image tokens per image. The rationale for opting for QWen-VL ’s encoder over the typically utilized CLIP [30] is elaborated in Section 4.3. The sequence length for training LLaMa-2 is capped at 4096. When referring to our model as DeepSpeed-VisualChat-xB (e.g., DeepSpeed-VisualChat-13B), the size is exclusively contingent on the language model components (LLaMa-2-13B).

- 4 Multi-Round Single-Image Exploration

#### 4.1 Multi-Modal Casual Attention

There are two common attention mechanisms used to connect the visual and textual components in a multi-modal model: causal attention, as used in [48, 5], and cross attention, as used in [18, 2].

Causal Attention (CA): The CA-based method simply projects visual features (i.e., the features from the output of the final visual encoder layer) into textual features and combines them with the normal textual

features after the textual embedding layer to feed into LLMs. The benefit of CA is that it’s a natural extension of the original attention mechanism in LLMs, and as such, it doesn’t introduce any extra modules or parameters. However, this approach raises some intuitive problems:

- (1) For a visual token, it attends to previous visual and textual tokens, even though visual tokens are

already fully encoded in a bidirectional manner and don’t need further attention from other visual tokens or the beginning of textual tokens.

- (2) For a textual token, it needs to learn how to distribute its attention weights between its previous

textual and image tokens. Due to these issues, we found that the data efficiency of CA in LVLMs is often problematic. To address this, LLaVA and QWen-VL require visual-language pretraining to fully align visual features with textual features. We also test and compare it with our proposed MMCA in Section 4.2.

Cross Attention (CrA): The alternative, cross attention (CrA), along with CA, exhibits better data efficiency but also comes with a few drawbacks:

- (1) It introduces new parameters to the model. For example, Otter has more than 1.5 billion trained

parameters compared to the millions of trained parameters in LLaVA. This significantly increases the training cost and memory requirements.

- (2) It requires careful design if an image is introduced in the middle of a conversation during training, as

previous text tokens should not be able to attend to the image.

Multi-Modal Causal Attention Mechanism (MMCA): To overcome these issues, we propose a new multi-modal causal attention mechanism (MMCA). The overall idea is as follows:

- (1) For visual tokens, they only attend to themselves, as visual tokens are encoded by the visual encoder.
- (2) For textual tokens, they attend to all their previous tokens. However, they have two separate attention

weight matrices for their previous textual tokens and image tokens.

The intuition behind the second point of MMCA is that the attention weight for one modality may affect the other modality. For instance, a textual token may pay more attention to textual information than visual information. Therefore, if the attention weight matrix is normalized across both modalities, the attention score for visual tokens might be very small. Refer to Figure 4 for a visualization of the three attention mechanisms.

- 4.2 Result

- 4.2.1 Comparison between Different Attentions

Experimental Setting We employ the LLaMa-2-7B language model in conjunction with the QWen-VLvisual-encoder as our visual encoder. These two models are connected via a straightforward linear projection layer. Our model underwent training on two LLaVa datasets, as outlined in the initial two rows of Table 1.

During training, all models were run for 5 epochs with a training batch size of 128. Our primary evaluation focused on single-image captioning and single-image Visual Question Answering (VQA). The peak learning rate was set to 1e-3 for both the projection linear layer and the embedding layer, and we employed the AdamW optimizer [25] with first- and second-order coefficients set to (0.0, 0.95).

For dataset splitting, we divided the training and validation datasets in a 90/10 ratio across the entire dataset. Additionally, we incorporated 10% of the total training iterations as warm-up steps. Our training framework of choice was DeepSpeed [31], and we utilized FP16 training to expedite the training process.

Throughout this work, we mainly compare the generation capability of different models on certain examples without comprehensively testing models on existing benchmark. Please see more details in Section 6 for limitations of our work.

Demo results. We begin by showcasing various examples that highlight the capabilities of DeepSpeedVisualChat in single-image visual language conversations, employing different attention mechanisms. As demonstrated in Figure 5, Figure 6, and Figure 7, DeepSpeed-VisualChat, when coupled with MMCA, effectively discerns visual details in images and furnishes coherent responses to user queries.

Furthermore, DeepSpeed-VisualChat exhibits a more comprehensive and precise grasp of image details compared to alternative attention mechanisms, such as the use of combined masks from both causal attention

Table 1: Training datasets summary. Due to context length limitation, for otter_mimicit_sn, otter_mimicit_tvc, and otter_mimicit_vst datasets we only used the samples with ≤ 8 images.

Name Num. samples Description

- (1) llava 49924 The detail description and complex reasoning data used by the LLaVA model [23]. Randomly concatenate 1 to 3 samples into one sample. Details in Section 5.1.
- (2) llava_dial 37818 The conversation data used by the LLaVA model [23]. Randomly concatenate 1 to 2 samples into one sample. Details in Section 5.1.
- (3) otter_mimicit_cgd 70940 The COCO (General) data used by the Otter model [18].
- (4) otter_mimicit_sd 8006 The SD (Surveillance) data used by the Otter model [18].
- (5) otter_mimicit_sn 487 The SN (Indoor Ego.) data used by the Otter model [18].
- (6) otter_mimicit_tvc 2 The TVC (TV) data used by the Otter model [18].
- (7) otter_mimicit_vst 115 The VIST (Story) data used by the Otter model [18].
- (8) llava_otter_blend 48869 Data blended from llava, llava_dial, otter_mimicit_cgd. Details in Section 5.1.
- (9) sparkles_dialogue 6520 The SparklesDialogue data used by the SparklesChat model [12]. Total 222681

[Figure 4]

- Figure 5: Example visual and language inputs that demonstrate the output comparison between (1) the standard causal attention (CA) (2) the standard causal attention combined with cross-attention (CA + CrA) and (3) the special multi-modal causal attention (MMCA) in DeepSpeed-VisualChat-Single.

and cross attention. It is also evident that, in contrast to the combination of CrA and CA, as well as MMCA, CA alone may exhibit slightly more errors (Figure 5) and capture a lower degree of reasoning capability (Figure 7).

##### 4.2.2 Result of DeepSpeed-VisualChat-Single

Experimental Setting All settings remain consistent with those outlined in Section 4.2.1, with the exception of two modifications: an increase in the language model size from LLaMa-2-7B to LLaMa-2-13B and an extension of the training epoch count from 5 to 10.

Demo results Upon elevating the language model size from 7B to 13B, we observe a marked improvement in the model’s ability to recognize images and capture the logical connections between questions and images.

Additionally, referring to the upper example in Figure 8, aside from the issue of incorrect descriptions, it becomes apparent that DeepSpeed-VisualChat-Single-7B occasionally exhibits repetitive context. However, this problem is less frequent with DeepSpeed-VisualChat-Single-13B, owing to the enhanced generative capabilities of larger language models. This observation aligns with the superior performance typically associated with larger language models.

[Figure 5]

###### Figure 6: DeepSpeed-VisualChat-Single accurately identifies the squirrel and camera in the image, while the baseline model mistakenly includes “standing next to a tree”.

[Figure 6]

###### Figure 7: DeepSpeed-VisualChat-Single accurately identifies the scene as a beautiful lake and offers a set of plausible suggestions. In contrast, the baseline misinterprets the image as containing “dock with a boat ramp”.

[Figure 7]

[Figure 8]

###### Figure 8: The above two examples illustrate the difference between DeepSpeed-VisualChat-Single-13B and DeepSpeed-VisualChat-Single-7B.

#### 4.3 Other Learning

Throughout the training process of DeepSpeed-VisualChat-Single, we accumulated several additional lessons. It’s important to note that most of these observations lack sufficient evidence and require further exploration. We present them here to assist others, but they should not be considered final conclusions.

- • Better Visual Encoder: Commonly, the CLIP visual encoder is used in LVLMs. However, the CLIP encoder’s resolution is limited to 224x224, which restricts the level of detail in the images. In our testing, we discovered that using the newly released visual encoder from QWen-VL significantly improves the final model quality due to its higher input resolution (448x448) and larger encoder size (2B parameters).
- • Overfitting or Not: Typically, we select the best evaluation checkpoint or one close to it for final testing. However, during DeepSpeed-VisualChat-Single training, we found that the final checkpoint, even if it appears overfitted, often delivers better testing results compared to middle checkpoints. Does this imply that we should intentionally overfit our model? The answer is no. We experimented with 5, 10, and 20 epochs for DeepSpeed-VisualChat-Single-13B and observed that 10-epoch training typically yields superior final model quality.
- • Adding LoRA to Visual Encoder or Language Decoder: We attempted to introduce LoRA-based training to enhance model quality. However, applying LoRA to either module did not yield any significant benefits.
- • Lowering the Learning Rate for Pretrained Components: We experimented with a smaller learning rate for language embedding since it is already pretrained. However, our results indicated that there is no significant difference when using a separate lower learning rate.
- • Using Chat-/Non-Chat-Based Models: We explored both chat-based and non-chat-based LLama-2 models. Our findings suggest that when using the chat-based model, strict adherence to the chat-based model’s instruction tuning format is crucial. Failing to do so resulted in even worse model quality than the non-chat-based model.
- • Inserting New Special Tokens or Not: As illustrated in Figure 3, a few tokens can be replaced by new inserted special tokens, such as encoding "###Human: " as a new special token. However, our testing revealed that it is better not to incorporate them as special tokens. Introducing them as special tokens significantly worsened our generation performance compared to the previous approach.

### 5 Multi-Round Multi-Image Exploration

#### 5.1 Data Blending

One critical missing element for enabling multi-round and multi-image conversations is data. The sole source of multi-round multi-image data we located is the SparklesDialogue dataset [12], which contains a mere 6520 samples. To address this limitation, we employed two methods to synthesize multi-round multi-image data from existing single-image or single-round data: simple data concatenation and LLaVA-Otter data blending.

##### 5.1.1 Simple data concatenation

For the "llava" and "llava_dial" datasets utilized by the LLaVA model, each sample comprises single/multiround conversations for a single image. To simulate scenarios where a user sequentially asks questions about multiple images, we conducted straightforward data post-processing for these two datasets. Specifically, we randomly concatenated different numbers of samples into a single sample. In the case of "llava," we concatenated 1 to 3 samples, while for "llava_dial," we concatenated 1 to 2 samples (refer to Table 1).

##### 5.1.2 LLaVA-Otter data blending

We noticed that the llava and llava_dial datasets used by LLaVA model and the otter_mimicit_cgd dataset used by the Otter model all use the COCO train2017 images. For the llava and llava_dial datasets, each sample includes a single/multi-round conversations for a single image. For the otter_mimicit_cgd dataset, each sample includes a single-round conversation for a pair of images. This enables us to build a synthesized multi-round multi-image data llava_otter_blend as a more natural blending: for each sample in the otter_mimicit_cgd dataset, we look for llava and llava_dial samples that use the same image, and then build a new sample in a "llava/llava_dial conversations then otter_mimicit_cgd conversation" fashion (as shown in Table 1).

- 5.2 Results

- 5.2.1 Comparison with QWen-VL and SparklesChat

Experimental Setting We utilize the datasets (1) to (8) as illustrated in Table 1. We deliberately exclude the dialogue data from SparklesChat to thoroughly investigate whether our newly proposed data blending technique can facilitate the interweaving of multi-image multi-round chats without the incorporation of new data. LLaMa-2-13B is employed as our language model, and the model is trained over 6 epochs. All other settings remain consistent with those outlined in Section 4.2.

Demo Results We compare DeepSpeed-VisualChat-13B with QWen-VL and SparklesChat as illustrated in Figure 9 and Figure 10. The tasks presented in Figure 9 are unseen to all the trained models. Notably, DeepSpeed-VisualChat-13B outperforms in terms of answer quality when compared to the other models. Specifically, while QWen-VL excels at offering succinct and accurate descriptions of individual images, it struggles to recall the first or second images during subsequent questions. On the other hand, SparklesChat excels at discerning differences between images, yet occasionally provides imprecise descriptions of individual images.

The tasks in Figure 10 center around narratives. Narratives are the primary training focus of SparklesChat and might be a part of QWen-VL’s training data (as its data is proprietary), but they were not part of DeepSpeed-VisualChat’s training (i.e., datasets (1) to (8) as mentioned in Table 1). Despite this, DeepSpeedVisualChat continues to provide commendable descriptions of individual images and exhibits some narrative skills. In contrast, both QWen-VL and SparklesChat demonstrate superior narrative abilities. Nonetheless, each model has its own set of limitations for specific questions.

It is worth noting that the training expenditure for DeepSpeed-VisualChat is significantly lower than that for QWen-VL and SparklesChat, with the latter having utilized the pre-training checkpoint of MiniGPT4.

- 5.2.2 Result of DeepSpeed-VisualChat

Experimental Setting The setting remains the same as mentioned above, with the addition of 6.5K more examples from SparklesChat.

Demo result We perform comparisons between DeepSpeed-VisualChat-13B with and without incorporating data from SparklesChat. For clarity, we will refer to DeepSpeed-VisualChat-13B without (and with) SparklesChat’s data as DeepSpeed-VisualChat-13B-Set1 (-Set2). First and foremost, by integrating SparklesChat’s data, DeepSpeed-VisualChat demonstrates enhanced narrative capability. Similar to SparklesChat, the newly trained model also displays a reduced ability to concentrate on the details of individual images.

Beyond the aforementioned, the introduction of additional data yields another intriguing observation. DeepSpeed-VisualChat-13B-Set2 exhibits increased sensitivity to prompt tuning compared to its predecessor. Specifically, as shown in Figure 12, a slight alteration to the prompt (highlighted in red text) without changing the question’s meaning, leads DeepSpeed-VisualChat-13B-Set2 to provide disparate answers. Conversely, the

[Figure 9]

- Figure 9: The above example (the conversation is from left to right panel) illustrates the difference among DeepSpeed-VisualChat-13B, QWen-VL, and SparklesChat. QWen-VL provides considerable short and accurate answers on describing the individual image but fails to remember the first image at the last second question. While SparklesChat is good at interpreting the difference between images but provides inaccurate information when describing individual images.

original DeepSpeed-VisualChat-13B-Set1 tends to offer more congruent responses. We hypothesize that this heightened sensitivity results from an imbalance in question formats/templates introduced by SparklesChat’s dataset.

For cross-comparison between DeepSpeed-VisualChat and QWen-VL/SparklesChat, please refer to Figure A.1 and Figure A.2.

##### 5.2.3 DeepSpeed-VisualChat with LLaMa-2-70B

We have initiated training with LLaMa-2-70B, maintaining the same training settings as outlined in Section 5.2.2. However, the resulting model is not adequately trained. We conjecture that the hyper-parameters optimal for LLaMa-2-13B may not be suitable for LLaMa-2-70B; for instance, the learning rate might be excessive, and the number of training epochs insufficient. Perfecting the training of DeepSpeed-VisualChat-70B is earmarked for future work.

[Figure 10]

- Figure 10: The given example, displayed in a left-to-right panel conversation format, demonstrates the disparities among DeepSpeed-VisualChat-13B, QWen-VL, and SparklesChat. QWen-VL excels in delivering succinct and accurate responses when describing individual images. Conversely, SparklesChat elaborates on the first two images but inaccurately identifies the scene as being from "Winnie" although later correctly attributing it to "Zootopia". When it comes to narrative skills, both QWen-VL and SparklesChat exhibit proficiency in story-telling, a skill that our model lacks. This deficiency can be attributed to the absence of narrative-centric content within the training data utilized for this model. In Figure 11, we will compare the model trained with and without narrative-centric content.

[Figure 11]

###### Figure 11: The above example (the conversation is from left to right panel) illustrates the differences between DeepSpeed-VisualChat-13B-Set1 and DeepSpeed-VisualChat-13b-Set2. We see that Set1 is better in describing the individual images (for example, Set1 can recognize the fox and "next window") but do not have ability to tell a story. While Set2 lost some ability to focus on the details of the images but can tell a story based on the given images.

[Figure 12]

- Figure 12: The given example, displayed in a left-to-right panel conversation format, demonstrates the disparities between DeepSpeed-VisualChat-13B-Set1 and DeepSpeed-VisualChat-13B-Set2, given a slightly different prompt series.

##### 5.2.4 Other Learning

- • Exploration of Projection Layers: We experimented with two different projection layers to bridge visual encoders and LLMs: a single linear layer and a Vision Transformer layer. We did not observe any benefits from the Vision Transformer approach in the preliminary phase, so we decided not to pursue this route further.
- • Advanced Data Blending Techniques: We explored more intricate data blending methods, such as shuffling the image ID of the Otter and LLaVA datasets. For example, in the Otter dataset, the paired images were later referenced as the first and third images by inserting another image as the second one. However, our experiments led to deteriorated performance, characterized by incomplete sentences and incorrect references. Upon reviewing the data, we hypothesized that these issues were probably due to incorrect references in the training data during the data blending process.

### 6 Limitations and Conclusions

Limitations Given that the focus of this work is not on benchmarking evaluations, we abstained from incorporating any such results. This might have resulted in the demonstrations illustrated in the paper appearing biased and not being comprehensive. Additionally, we have observed that data is a pivotal

component to achieve high-quality LVLMs, but we were unable to provide such datasets due to constraints on resources. We acknowledge that larger language models can potentially offer superior model quality, but we encountered difficulties in training a model based on LLaMa-2-70B. Attempts were made to train with the LLaMa-2-70B model, but the end results were suboptimal. We also noted the hallucination issue with DeepSpeed-VisualChat, a problem inherited from the LLaMa-2 family.

Conclusions In conclusion, We propose DeepSpeed-VisualChat, the Multi Round and Multi Images Chat framework, a pioneering solution offering open-source support for multi-image dialogues and featuring an innovative multi-modal casual attention mechanism and efficient data blending techniques. Our approach surpasses contemporaneous models in scalability enabling enhanced adaptability in diverse interactive scenarios, without incurring additional training costs or complexity. This breakthrough facilitates unprecedented advancements in large vision and language models, laying a robust foundation for the seamless integration of visual and textual information in future multi-modal models.

### Contributions

ZY: Full engagement and project lead. XW: Training/evaluation pipeline development and writing. CL: Data support. MZ: Training pipeline support. QH: DeepSpeed ZeRO feature adaptation. OR: DeepSpeed ZeRO feature adaptation. AAA: Software support. SR: Consulting. YH: Team lead.

### Acknowledgment

We thank the entire DeepSpeed team for their support.

### References

- [1] Open AI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. CoRR, abs/2204.14198, 2022.
- [3] ChatLLaMa Authors. Chatllama. https://github.com/juncongmoo/chatllama, 2023.
- [4] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023.
- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.
- [6] Hangbo Bao, Wenhui Wang, Li Dong, Qiang Liu, Owais Khan Mohammed, Kriti Aggarwal, Subhojit Som, Songhao Piao, and Furu Wei. Vlmo: Unified vision-language pre-training with mixture-of-modalityexperts. In Advances in Neural Information Processing Systems, 2022.
- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

- [8] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022.
- [9] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In CVPR, 2023.
- [10] GitHub. Github copilot. https://github.com/features/copilot/, 2021.
- [11] Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790, 2023.
- [12] Yupan Huang, Zaiqiao Meng, Fangyu Liu, Yixuan Su, Nigel Collier, and Yutong Lu. Sparkles: Unlocking chats across multiple images for multimodal instruction-following models. arXiv preprint arXiv:2308.16463, 2023.
- [13] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. If you use this software, please cite it as below.
- [14] Shima Imani, Liang Du, and Harsh Shrivastava. MathPrompter: Mathematical reasoning using large language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 5: Industry Track), pages 37–42, Toronto, Canada, July 2023. Association for Computational Linguistics.
- [15] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 4904–4916. PMLR, 2021.
- [16] Wonjae Kim, Bokyung Son, and Ildoo Kim. ViLT: Vision-and-language transformer without convolution or region supervision. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 5583–5594. PMLR, 2021.
- [17] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022.
- [18] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Mimic-it: Multi-modal in-context instruction tuning. arXiv preprint arXiv:2306.05425, 2023.
- [19] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.
- [20] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, pages 12888–12900. PMLR, 2022.
- [21] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. In Advances in neural information processing systems, volume 34, pages 9694–9705, 2021.

- [22] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, Yejin Choi, and Jianfeng Gao. Oscar: Object-semantics aligned pre-training for vision-language tasks. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, editors, Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XXX, volume 12375 of Lecture Notes in Computer Science, pages 121–137. Springer, 2020.
- [23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.
- [24] Haotian Liu, Kilho Son, Jianwei Yang, Ce Liu, Jianfeng Gao, Yong Jae Lee, and Chunyuan Li. Learning customized visual models with retrieval-augmented knowledge. CVPR, 2023.
- [25] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [26] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 13–23, 2019.
- [27] OpenAI. Chatgpt. https://openai.com/blog/chatgpt, 2022.
- [28] OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023.
- [29] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 2021.
- [31] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3505–3506, 2020.
- [32] Sheng Shen, Chunyuan Li, Xiaowei Hu, Yujia Xie, Jianwei Yang, Pengchuan Zhang, Zhe Gan, Lijuan Wang, Lu Yuan, Ce Liu, et al. K-lite: Learning transferable visual models with external knowledge. In Advances in Neural Information Processing Systems, 2022.
- [33] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How much can clip benefit vision-and-language tasks? In ICLR, 2022.
- [34] Sheng Shen, Zhewei Yao, Chunyuan Li, Trevor Darrell, Kurt Keutzer, and Yuxiong He. Scaling vision-language models with sparse mixture of experts. arXiv preprint arXiv:2303.07226, 2023.
- [35] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. FLAVA: A foundational language and vision alignment model. CoRR, abs/2112.04482, 2021.
- [36] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. VL-BERT: pre-training of generic visual-linguistic representations. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020.

- [37] Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355, 2023.
- [38] Hao Tan and Mohit Bansal. LXMERT: Learning cross-modality encoder representations from transformers. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pages 5099–5110. Association for Computational Linguistics, 2019.
- [39] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [40] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. CoRR, abs/2202.03052, 2022.
- [41] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. arXiv preprint arXiv:2208.10442, 2022.
- [42] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. SimVLM: Simple visual language model pretraining with weak supervision. In ICLR, 2022.
- [43] Sean Welleck, Jiacheng Liu, Ximing Lu, Hannaneh Hajishirzi, and Yejin Choi. Naturalprover: Grounded mathematical proof generation with language models. Advances in Neural Information Processing Systems, 35:4913–4927, 2022.
- [44] Zhewei Yao, Reza Yazdani Aminabadi, Olatunji Ruwase, Samyam Rajbhandari, Xiaoxia Wu, Ammar Ahmad Awan, Jeff Rasley, Minjia Zhang, Conglong Li, Connor Holmes, Zhongzhu Zhou, Michael Wyatt, Molly Smith, Lev Kurilenko, Heyang Qin, Masahiro Tanaka, Shuai Che, Shuaiwen Leon Song, and Yuxiong He. DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales. arXiv preprint arXiv:2308.01320, 2023.
- [45] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. CoRR, abs/2205.01917, 2022.
- [46] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, Ce Liu, Mengchen Liu, Zicheng Liu, Yumao Lu, Yu Shi, Lijuan Wang, Jianfeng Wang, Bin Xiao, Zhen Xiao, Jianwei Yang, Michael Zeng, Luowei Zhou, and Pengchuan Zhang. Florence: A new foundation model for computer vision. CoRR, abs/2111.11432, 2021.
- [47] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. VinVL: Revisiting visual representations in vision-language models. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pages 5579–5588. Computer Vision Foundation / IEEE, 2021.
- [48] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

### A More examples

[Figure 13]

###### Figure A.1: The given example, displayed in a left-to-right panel conversation format, demonstrates the disparities among DeepSpeed-VisualChat-13B-Set1 and DeepSpeed-VisualChat-13B-Set2, QWen-VL, and SparklesChat.

[Figure 14]

###### Figure A.2: The given example, displayed in a left-to-right panel conversation format, demonstrates the disparities among DeepSpeed-VisualChat-13B-Set1 and DeepSpeed-VisualChat-13B-Set2, QWen-VL, and SparklesChat.

