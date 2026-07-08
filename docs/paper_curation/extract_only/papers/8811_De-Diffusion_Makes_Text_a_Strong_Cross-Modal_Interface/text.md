## De-Diffusion Makes Text a Strong Cross-Modal Interface

Chen Wei1,2 Chenxi Liu1 Siyuan Qiao1 Zhishuai Zhang1 Alan Yuille2 Jiahui Yu1 1Google DeepMind 2Johns Hopkins University

(i) Transferable Prompts for Text-to-Image Generation

(ii) Multi-Modal Dialogue

# arXiv:2311.00618v1[cs.CV]1Nov2023

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

What is unusual about this image?

It is unusual to see a corgi riding a bicycle, especially in a busy city like New York…

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Bard

Prompt

[Figure 10]

“a photograph closeup dog corgi wearing white hat corgi yellow glasses…”

[Figure 11]

[Figure 12]

[Figure 13]

Stable Diffusion Midjourney Imagen

De-Diffusion

[Figure 14]

[Figure 15]

“a photograph closeup dog corgi wearing white hat corgi yellow glasses atop on red handle bicycle street nyc corgi smiling wearing an orange fedora smile yellow sunglasses people car multiple screen buildings street…”

decoder (text2img diffusion)

encoder

(iii) Multi-Modal Few-Shot Learning with LLMs

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Q: What is the corgi dog wearing? A:

Yellow sunglasses. PaLM 2

Q: What is the tabby cat wearing? A: Red sweater.

“a photograph closeup dog corgi wearing white hat corgi yellow glasses…”

[Figure 20]

“a photograph a cat wearing reddish red wool sweater with glasses…”

[Figure 21]

[Figure 22]

Prompt

[Figure 23]

Figure 1. De-Diffusion is an autoencoder whose decoder is a pre-trained text-to-image diffusion model. It encodes an input image into a piece of information-rich text, which mixes comprehensive semantic concepts present in the image to be a “scrambled caption”. We group semantics by color for illustration. De-Diffusion text can act as a flexible interface between different modalities, for example, enabling diverse vision-language applications including: (i) providing transferable prompts for different text-to-image tools, (ii) enabling text-only chatbots, e.g., Bard [1], to engage in multi-modal dialogue, and (iii) injecting image context into off-the-shelf large language models (LLMs), e.g., PaLM 2 [5], to perform open-ended visual question answering by prompting the LLM with few-shot examples.

### Abstract

the fixed text-to-image diffusion decoder to reconstruct the original input – a process we term De-Diffusion. Experiments validate both the precision and comprehensiveness of De-Diffusion text representing images, such that it can be readily ingested by off-the-shelf text-to-image tools and LLMs for diverse multi-modal tasks. For example, a single De-Diffusion model can generalize to provide transferable prompts for different text-to-image tools, and also achieves a new state of the art on open-ended vision-language tasks by simply prompting large language models with few-shot examples. Project page: dediffusion.github.io.

We demonstrate text as a strong cross-modal interface. Rather than relying on deep embeddings to connect image and language as the interface representation, our approach represents an image as text, from which we enjoy the interpretability and flexibility inherent to natural language. We employ an autoencoder that uses a pre-trained text-toimage diffusion model for decoding. The encoder is trained to transform an input image into text, which is then fed into

### 1. Introduction

We have witnessed LLM-powered products such as ChatGPT taking over the world by storm. Nowadays many people are convinced of the benefits that LLMs can bring in understanding natural language conversations and assisting humans in creative tasks. However, what is the path forward? One clear direction and trend is towards multi-modality, allowing the model to understand additional modalities such as image, video, and audio. GPT-4 [60] is a multi-modal model with impressive image understanding capabilities, and has recently rolled out to the public together with audio-processing capabilities. Gemini is also “multi-modal from day one” [2]. Multi-modal models like these have a fundamental design choice to make, i.e., how different modalities should communicate and connect? In the context of this work, we rephrase the question as: what is the cross-modal interface?

We argue that a good cross-modal interface should at least possess the following two properties: (1) content preserving, i.e., signals from the original modality can be reconstructed from the interface representation to a high degree; (2) semantically meaningful, i.e., the interface representation contains useful abstractions of the raw signals, so that understanding and reasoning can be performed more easily. Balancing these two properties is challenging, and in fact they can often be in contention with each other. For example, the raw signals from the original modality satisfy content preserving perfectly, but are lacking on the semantically meaningful front.

Ever since the deep learning era [19,32,33,46], deep embeddings have been the go-to choice as cross-modal interface. They can be good at preserving image pixels if trained as an autoencoder [33], and can also be semantically meaningful, with the most recent exemplar being CLIP [63]. In this paper, we do not argue that deep embeddings are a bad cross-modal interface per se, but instead convey the idea that according to our experiments, text can be a strong alternative cross-modal interface.

If we consider the relationship between the speech and text for a quick second, text has always been so natural of a cross-modal interface that we do not typically think of it as such. Converting the speech audio to text well preserves the content such that we can reconstruct the speech audio with the mature text-to-speech technique. We are also confident that the transcribed text contains all the semantics information, in other words, semantically meaningful. By analogy, we can also “transcribe” an image into text, which has the more familiar name of image captioning. But when we compare typical image captions against the two properties of cross-modal interface, they do not preserve content well but only capture the most salient semantic concepts. In other words, image captions are more about precision than comprehensiveness [13,88], and it is hard to answer any and

all visual questions from the short captions (e.g., Fig. 6).

While image captions do not make an ideal interface representation, we argue that precise and comprehensive text, if attainable, remains a promising option, both intuitively and practically. Intuitively, humans rely on language to articulate our physical surroundings, engage in reasoning, and deliver solutions. In other words, we constantly “transcribe” information about the external world into language and use it as an interface for higher-level cognition [16,24]. Practically, text is the native input domain for LLMs. Using text as the interface can avoid the need for adaptive training often required with deep embeddings [4, 49]. Given that training and adapting top-performing LLMs can be prohibitively expensive [4, 5, 60], text provides a modular design that opens up more possibilities. The question is, how can we attain precise and comprehensive text of images?

We resort to the classic autoencoding for a solution [33]. Unlike common autoencoders, we utilize a pre-trained textto-image diffusion model as the decoder, and naturally, with text as the latent space. The encoder is trained to transform an input image into text, which is then fed into the textto-image diffusion model for decoding. To minimize the reconstruct error, the latent text, though often mixing semantic concepts together to be a “scrambled caption” of the input image, has to be both precise and comprehensive. No extra supervision is used other than images themselves.

Recent generative text-to-image models excel at converting arbitrary rich text of, e.g., tens of words, to highly detailed images that closely follow the prompts [59,65,68,71, 91]. This essentially suggests the remarkable capability of these generative models to process complex text into visually coherent outputs. By employing one of these generative text-to-image models as the decoder, the optimized encoder explores the wide latent space of text and unpacks the enormous visual-language knowledge encapsulated within the generative model, embodying a foundational paradigm known as Analysis by Synthesis [7,10,94].

We show De-Diffusion text extensively captures semantic concepts in images, and, when used as text prompts, enables diverse vision-language applications (Fig. 1). DeDiffusion text can generalize to be a transferable prompt for different text-to-image tools. Evaluated quantitatively by reconstruction FID [31], De-Diffusion text significantly outperforms human-annotated captions as prompts to a third-party text-to-image model [68]. De-Diffusion text also enables off-the-shelf LLMs to conduct open-ended visionlanguage tasks by simply prompting LLMs with few-shot task-specific examples. We highlight De-Diffusion outperforms Flamingo [4] on open-ended few-shot VQA [6] with 100× fewer learnable weights and without using interleaved image-text supervision. The results demonstrate DeDiffusion text effectively interconnects both human interpretations and various off-the-shelf models across domains.

### 2. Related Work

Autoencoding is a classical approach for learning representations [33,70]. It uses an encoder to map the input into a compressed, meaningful representation, and a decoder to reconstruct the input from this representation to be as close as possible to the original. This simple autoencoding concept underpins many unsupervised representation learning algorithms across domains [20, 30, 34, 45, 82]. By forcing the model to compress then reconstruct the input, autoencoders discover useful structural representations of the data. For example, Neural De-Rendering [87] is a generalized autoencoder that utilizes a deterministic rendering function as the decoder and maps images into structured and disentangled scene descriptions. Inspired by its name “de-rendering”, we name our approach “De-Diffusion”.

A specific type of autoencoder, VQ-VAE [67,79] or discrete VAE [66], is designed to learn discrete, structured representations in the latent space. This can be especially useful for modeling data with categorical or symbolic attributes. These methods are now widely adopted in multimodal models to tokenize images [22, 66, 68, 91]. However, VQ-VAE’s latent space is hidden and often entangled, requiring adaptive fine-tuning for downstream tasks. DeDiffusion also utilizes a discrete latent space. In contrast, we directly encode images into a sequence of text, which is directly interpretable.

SPAE [92] and LQAE [54] are two recent approaches that encode images into the vocabulary space of a fixed LLM. They jointly learn the encoder and decoder from scratch. Consequently, although the latent space is discrete text, it tends to act as a “cipher code” that only the co-trained decoder can interpret. This limits generalization to human understanding and off-the-shelf LLMs and textto-image models. In contrast, De-Diffusion utilizes a pretrained text-to-image diffusion model as the decoder, obtaining interpretable text as the latent representation.

How many words is an image worth? The adage “a picture is worth a thousand words” means that still images can convey complex and sometimes multiple ideas more effectively than a mere verbal description. Indeed, a single image can tell a story that would take many words to explain. The question, how many words is an image worth, is constantly explored by the computer vision community [23, 25, 26, 52]. For example, “An image is worth 16×16 words”, or ViT [21], proposes to take the image patches as tokens (words) and process these tokens by Transformers [80], which has become one of the standard vision backbones now. In this sense, our work can also been seen as “An image is worth 75 words”, for we encode input images into a sequence of 75 tokens.

Several prior works also explore to use text to represent images [9, 89] and combine with with LLMs. However,

these works rely on multiple captioning and classification models, whose outputs are concatenated to be the text representation. Their performance is heavily dependent on the captioning and classification models, and we demonstrate in § 4 that even human-annotation captions can lack the extensive details covered in De-Diffusion text.

Vision-language models. The breakthrough in NLP [11, 20, 37, 38, 60, 64, 85], especially their abilities to perform few-shot learning, has inspired a large body of visionlanguage work. A family of vision-language models is based on contrastive learning [29], where images and text are projected in to a same embedding space [41,43,50,61, 63, 90, 95]. De-Diffusion differs from contrastive models as we encode image as text, instead of deep embeddings. Another family of vision-language models fuses vision and language models by jointly training them with large-scale image-text data [4,14,47,53,58,66,90,93]. In contrast, DeDiffusion takes a modular design with text as the representation, bypassing the heavy cost image-text data collection and jointly training large-scale vision and language models.

### 3. Method

#### 3.1. De-Diffusion for Text Representation

Autoencoder. Autoencoding is one of the classical methods for representation learning [33, 70]. An autoencoder first encodes an input x into a latent representation z, then decodes z back to x˜ for reconstruction. Both the encoder and the decoder are optimized so that the reconstructed input x˜ is as similar as possible to the original input x. By doing so, the compressed representation z preserves the information in the input. Since no more supervision is required except the input itself, autoencoding is an unsupervised approach without the heavy burden of human annotation.

Text as the latent representation. While autoencoders can learn compressed representations z that preserve useful information, it is difficult to use the latent z for downstream tasks without any additional training, let alone direct human interpretation. In this work, we propose to encode the input image into text. Practically, the encoder compresses each image into a sequence of BPE-encoded text tokens [73], where each token can take on a discrete value from the vocabulary. To faithfully reconstruct the image from the latent text, the text must precisely and comprehensively capture the semantic concepts present in the image, making a interface representation, in contrast to image captions that only focus on the most visually salient information.

Text-to-image diffusion as the decoder. One potential concern is that the encoder might still encrypt the images into a cipher code that only the decoder can decipher, making human interpretation challenging. This is particularly likely when the encoder and the decoder are jointly trained.

pixel reconstruction

||attn. pooler<br><br>[Figure 24]|
|---|
<br><br>|image back bone<br><br>[Figure 25]<br><br>[Figure 26]<br><br>/|
|---|
|
|---|

||U-Net<br><br>[Figure 27]|
|---|
<br><br>|CLIP text model<br><br>[Figure 28]|
|---|
|
|---|

###### input text output

gumbel softmax

image-to-text encoder

text-to-image diffusion decoder

- Figure 2. Architecture of De-Diffusion. The overall structure is an autoencoder, with (i) a pre-trained text-to-image diffusion model as the decoder, (ii) text as the latent representation, and (iii) a image-to-text encoder consisting of a image backbone and an attentional pooler. Lock and unlock denote frozen and learnable weights, respectively. We use Gumbel-softmax [42,56] for discrete text tokens.

Optimization. Same as other autoencoders, the training objective of De-Diffusion is to minimize the reconstruction error between the input image and the reconstruction from the pre-trained diffusion model. Specifically, both the loss function and the noise variance schedule strictly follow those of the pre-trained diffusion model [35]. The training data of De-Diffusion only includes images, without human annotations or paired text descriptions.

To mitigate this concern [54], we introduce a pre-trained text-to-image diffusion model as the decoder, and dub our method as “De-Diffusion”.

Text-to-image diffusion models, as the name suggested, learn the relationship between text and images from a large dataset of image-text pairs and excel at converting texts into highly detailed images. They already establish the projection from descriptive text to image, and we unpack this encapsulated knowledge by employing a frozen text-to-image diffusion model as the decoder. As illustrated in Fig. 2, the text-to-image diffusion model consists of a CLIP text encoder [63] and a U-Net [69], and the codebook is then naturally the vocabulary of the CLIP text encoder.

Our model can be viewed as a special discrete autoencoder with discrete text tokens as the latent. Similar to other discrete autoencoders [66, 67, 79], we use Gumbelsoftmax [42, 56] as the continuous relaxation to backpropagate the gradients from the decoder through the discrete latent. The relaxation becomes tight as the temperature τ →0. We find that an annealing schedule of temperature τ is important for stable training.

When training De-Diffusion, we freeze the parameters of the text-to-image diffusion decoder. In each mini-batch, we expose the decoder with one randomly sampled noise level for each sample. This resembles the training procedure for diffusion models [35], except the parameters are fixed and the text conditions are outputs of the image-to-text encoder instead of the training data.

To increase the information density and readability, we exclude all the punctuation in the vocabulary, which accounts for around 6% of the original vocabulary of CLIP text encoder. As a result, only word tokens and number tokens are allowed. We ablation this design choice in Tab. 4b.

Image-to-text encoder. The encoder maps the input image into text. It starts with an image backbone that extracts image features, followed by an attentional pooler [39,90] that turns the features into output text tokens. The image backbone can be a pre-trained and frozen model that excels at image feature extraction. It can also be randomly initialized, supervised by the reconstruction objective during DeDiffusion training. We ablate the two choices in Tab. 4d.

#### 3.2. Implementation Details

Text-to-image diffusion model. The text-to-image diffusion model used for De-Diffusion training is based on Imagen [71]. The U-Net has 600M parameters with an embedding dimension of 256 and input resolution of 64×64. The text encoder is from OpenCLIP ViT-H/14 [15]. The training data is WebLI [14], an image-language dataset built from public web images and texts. We use v-prediction as the objective [72], a batch size of 2048, and train for 3M steps. For reference, this text-to-diffusion model achieves an FID of 5.37 on 30K 64×64 MS-COCO 2014 validation images. Image backbone and attentional pooler. We utilize a pretrained CoCa ViT-L model with input resolution 288×288 as the image backbone, and freeze it during De-Diffusion training [21, 90]. This CoCa model is pre-trained on JFT3B [76] and ALIGN datasets [43]. Our attentional pooler is equipped with 75 queries, in addition to the [SOS] and [EOS] tokens to fully utilize the 77 context length defined by CLIP text encoder [15,63]. The attention pooler has five Transformer blocks which are always randomly initialized.

The attentional pooler projects n learnable queries to n text tokens by a few Transformer blocks [80]. Each Transformer block consists of a self-attention layer over all the queries, a cross-attention layer to gather features from the image backbone, and an MLP layer. After the Transformer blocks, a linear layer projects the queries to discrete text tokens from the vocabulary of CLIP text encoder, in order to connect to the diffusion decoder. The n queries are positional sensitive, meaning that each query corresponds to a specific position in the CLIP text encoder. The n output text tokens, together with the special tokens [SOS] and [EOS], are then fed into the diffusion decoder. We ablate the effect of n, the number of text tokens, in Tab. 4a.

| | | | | |
|---|---|---|---|---|
| | | | | |

PaLI-X

18

BLIP-2

14

Human

FID

De-Diffusion

10

6

2 4 6 8 guidance scale

- Figure 3. Evaluating different captioning methods by text-toimage reconstruction. The text-to-image model is a pre-trained Stable Diffusion v2-base model [68]. We report FID (↓) on 30K MS-COCO (2014) validation split with 256×256 images. De-Diffusion obtains better FID than human-annotated captions, BLIP-2 [49] (fine-tuned on MS-COCO), and PaLI-X [12] (a multitask captioning model). Numerical results are provided in Tab. 6.

Training of De-Diffusion. The De-Diffusion training data also comes from WebLI [14], while only the images but not the text are used. The broad domain coverage of WebLI enables zero-shot and few-shot evaluations of De-Diffusion on downstream applications in the next section (§ 4). For memory efficiency, we use the Adafactor optimizer [74] with β1 =0.9, β2 =0.999 and a decoupled weight decay ratio of 0.01. We train with a batch size of 2048 for 500K steps, taking around 2.5 days on 64 TPUv4 chips. The learning rate starts at 3e-4 and is annealed to 3e-6 with cosine decay [55], along with a 10K step warmup [27]. The Gumbel-softmax temperature begins from 2.0 and is exponentially annealed to 0.3 through the entire schedule, which we find is sufficient to close the gap between the continuous relaxation during training and the discrete inference.

- 4. Experiments and Applications

In this section, we introduce several applications of DeDiffusion text, ranging from transferable prompts for textto-image tools and few-shot vision-language understanding. To demonstrate the versatility of De-Diffusion text across different tasks and domains – that is, its ability to serve as a strong cross-modal interface – all the applications use text from a single De-Diffusion model detailed in § 3.2.

#### 4.1. Transferable Text-to-Image Prompt

Since De-Diffusion encodes an input image into text and decode it by a text-to-image diffusion model, it is trivial for De-Diffusion text to serve as a prompt suggestion to reconstruct an image by this specific text-to-image diffusion decoder. Furthermore, we demonstrate that De-Diffusion

text is transferable to other unseen decoders, i.e., text-toimage tools, such as Imagen [71], Stable Diffusion [68] and Midjourney [3]. This suggests that De-Diffusion text is not over-fitted to a single text-to-image decoder but generalizable across different text-to-image frameworks, which is crucial to make a cross-model interface.

Quantitative evaluation. We quantitatively evaluate the ability of De-Diffusion text to transfer to other text-toimage diffusion models and compare with traditional captioning methods. To do this, we develop a benchmark that uses a third-party pre-trained text-to-image model to reconstruct an image from either De-Diffusion text or captions. Specifically, we first obtain De-Diffusion text and captions for a given image. Both are then input into the third-party text-to-image model to synthesize the corresponding image. We compare the synthesized image to the original. Text containing more precise and comprehensive descriptions allows the model to produce images more similar to the original. By evaluating the similarity between original and synthesized images, our benchmark quantifies the precision and comprehensiveness of different methods.

We use the pre-trained Stable Diffusion v2-base [68] as a generic text-to-image generator, whose weights and training data are oblivious to both De-Diffusion and captioning methods. We measure the similarity between original and synthesized 256×256 images using FID (Frechet Inception Distance) [31] on 30K images from MS-COCO 2014 validation split [13]. Image generation utilizes different classifier-free guidance [36] scales from 1.5 to 8.0, along with 50 steps of DDIM sampling [75].

We evaluate De-Diffusion, human captions and two state-of-the-art image captioning methods, plotted in Fig. 3:

- (i) Human-annotated captions from MS-COCO provide

a strong FID baseline of 8.08 at guidance scale 3.0. We synthesize new images using the longest of the five annotated captions, which we find works best. Other options to utilize human captions are discussed in Appendix A.

- (ii) BLIP-2 refers to its ViT-g OPT 2.7B variant [49],

which is fine-tuned on MS-COCO. As one of the state-ofthe-art captioning methods, BLIP-2’s FID curve is close to that of human-annotated captions.

- (iii) PaLI-X [12] performs fine-tuning on multiple cap-

tion datasets, instead of solely on MS-COCO. As a result, its FID curve is higher than that of BLIP-2.

- (iv) De-Diffusion is trained with solely web images, but

not MS-COCO images or any human-annotated captioning data. It has an indirect access to noisy web image-language pairs through the pre-trained diffusion model. However, De-Diffusion achieves the lowest FID of 6.43 at guidance 3.0, significantly better than the human-annotated captions.

These results indicate that De-Diffusion text precisely and comprehensively verbalizes image details, allowing it to effectively transfer to other text-to-image tools.

Qualitative evaluation. Our visualizations in Figs. 6 and 7 demonstrate that De-Diffusion text is more comprehensive than human-annotated captions. Images are from MSCOCO 2014 validation split and we test with three prominent text-to-image tools including Stable Diffusion XL [62], Midjourney [3], and Imagen [71].

The results show that De-Diffusion text covers finegrained semantic aspects ranging from objects and their positional relationships, human attributes, backgrounds, to action subcategories. In contrast, human-annotated captions often neglect fine-grained semantic details, leading to high variance in the generated images across text-to-image tools. While the descriptions in human captions are precise, DeDiffusion text much more comprehensively enumerates key objects, their attributes, their relationships and background contexts. This comprehension allows cross-tool text-toimage reconstruction with De-Diffusion.

Figs. 8 and 9 visualize text-to-image reconstruction with De-Diffusion text on synthetic images from other text-toimage tools Ideogram1 and Lexica2. We provide the synthetic links of these images in Appendix E. Fig. 8 shows DeDiffusion can provide fine-grained descriptions for complex and diverse synthetic images besides photographic images in MS-COCO (Figs. 6 and 7). The prompts also transfer across different text-to-image models. Fig. 9 further highlights the ability of De-Diffusion to articulate diverse image types and explicitly name the genre such as “cg wallpaper”, “watercolor painting”, “etching logo”, and a plain image of a black circle. These results suggest that De-Diffusion can be applied to provide cross-tool prompt inspiration for useruploaded images to explore new vocabulary and aesthetics.

#### 4.2. Multi-Modal Few-Shot Learner

We next show that De-Diffusion can convert an offthe-shelf LLM, which is never trained on vision-language data, to perform open-ended vision-language task by simply prompting the LLM with few-shot examples, and no adaptive training is required.

LLMs exhibit surprising generalization ability with fewshot learning, adapting to new tasks from just a few annotated task-specific examples without any further training [11]. However, these powerful models are limited to text. Since then, methods have emerged to enable multimodal capabilities by encoding images into the word embedding space [63,78] or training a new module to connect vision and language embeddings [4,49]. However, these approaches have downsides – not only would they introduce prohibitively heavy computational costs due to joint training with enormous language models like 540B PaLM [17], but the visual embeddings also bind to a specific language model such that changing the language model requires re-

- 1https://ideogram.ai
- 2https://lexica.art

trainable VQAv2OKVQACOCO

|methods LLM params. shot<br><br>|test-dev val test|
|---|---|
|BLIP-2 ViT-g [49] FlanT5XXL 108M 0 LENS [9] FlanT5XXL 0 0 AnyMAL ViT-G [58] Llama270B - 0 PICa-Full [89] GPT-3 0 16<br><br>|65.0† 45.9† 62.6 43.3 64.2 42.6 95.9 56.1 48.0 -|
|OpenFlamingo-9B [8]MPT7B - 0 OpenFlamingo-9B [8]MPT7B - 4 OpenFlamingo-9B [8]MPT7B - 32<br><br>|52.7 37.8 79.5 54.8 40.1 89.0<br>53.3 42.4 99.5<br>|
|IDEFICS-80B [47] Llama65B 14B 0 IDEFICS-80B [47] Llama65B 14B 4 IDEFICS-80B [47] Llama65B 14B 32|60.0 45.2 91.8<br><br>63.6 52.4 110.3 65.9 57.8 116.6<br><br>|
|Flamingo-9B [4] Chinchilla7B 2B 0 Flamingo-9B [4] Chinchilla7B 2B 4 Flamingo-9B [4] Chinchilla7B 2B 32|51.8 44.7 79.4 56.3 49.3 93.1 60.4 51.0 106.3<br><br>|
|Flamingo-80B [4] Chinchilla70B 10B 0 Flamingo-80B [4] Chinchilla70B 10B 4 Flamingo-80B [4] Chinchilla70B 10B 32|56.3 50.6 84.3<br><br>63.1 57.4 103.2 67.6 57.8 113.8<br><br>|
|De-Diffusion ViT-L PaLM 2-S 135M 0 De-Diffusion ViT-L PaLM 2-S 135M 4 De-Diffusion ViT-L PaLM 2-S 135M 32|63.9 51.4 63.4<br><br>64.0 53.5 87.1<br><br><br>63.1 53.3 92.0<br><br>|
|De-Diffusion ViT-L PaLM 2-L 135M 0 De-Diffusion ViT-L PaLM 2-L 135M 4 De-Diffusion ViT-L PaLM 2-L 135M 32|67.2 57.0 88.5<br><br>67.9 58.2 100.3<br><br>68.4 60.6 103.7<br>|

Table 1. Vision-language few-shot learning. We report VQA accuracy [6] for visual question answering on VQAv2 [28] and OKVQA [57] in the open-ended setting, and CIDEr [81] for MSCOCO image captioning [13]. The Bold denotes the top performance and the underlined denotes the second-best in each column. † in-domain COCO images are used for training.

training. This limits the flexibility of these multi-modal models to keep pace with rapid progress in LLMs.

Unlike previous methods based on deep embeddings, De-Diffusion encodes images into text that any language model can readily comprehend. This allows off-the-shelf language models to ground images by simply interleaving task instructions and De-Diffusion text in any order, as Fig. 1 shows. Using text as a cross-modal interface, De-Diffusion empowers off-the-shelf language models with multi-modal abilities. We next demonstrate that this modular approach achieves state-of-the-art performance on different multi-modal few-shot learning benchmarks, thanks to the comprehensive image context provided by De-Diffusion text, and seamless integration with advanced reasoning abilities provided by the LLMs.

Multi-modal few-shot learning. We follow the evaluation protocol of Flamingo [4] to assess few-shot learning on three vision-language tasks including VQAv2 [28],

[Figure 29]

- (a) reference image

[LLM prompt] Answer the question given the context. Image context: a colvonvscocam blog closeup of young rear man looking carrying head teddybear wearing a red sweater it in white barriers amidst between a a wooden polcoping a rails opposite blurry except a woman people sitting right off white cabinets lit white windows and sill approximately wearing hair hair burgundy and black tabletop backpack approximately wallets blush brown hair hair tallinn salzburg church church church backpack closeup closeup hair hair cubic Image question: What toy is this? Short answer: [LLM completion] teddy bear. [GT answers] stuffed animal, teddy bear

[Figure 30]

- (b) reference image

[LLM prompt] Answer the question given the context. Image context: a colcandidenverlanticcloseup former recent train train parked traditionenclosed metrotram in a red livery it on railroad platform containing wearing a a yellowpolsurround a knob beside platform near a under platform shelter right there and roof shadows and platform and tracks etc wore worn worn maumaroon brown white stripes markings contentworn yellow yellow stripes train pretoria namibia railway platform train operator worn brown windows platform platform Image question: What other big vehicle is often painted about the same shade as this vehicle? Short answer: [LLM completion] fire truck. [GT answers] firetruck, fire truck

- Figure 4. VQA with an off-the-shelf LLM, where De-Diffusion text of the reference image is inserted after “Image context” in the LLM prompt. The LLM then completes the prompt to answer the visual question. De-Diffusion text provides abundant visual details, e.g., teddy bear in (a) and red livery of the train in (b). We use PaLM 2-L [5] as the LLM. Samples are from OKVQA [57].

OKVQA [57] and MS-COCO caption [13]. De-Diffusion text for the support images is interleaved along with their questions, answers, and captions to form prompts for the LLMs. The LLM’s completion is considered a correct answer only if it exactly matches the ground truth. More details are in Appendix B. Results are shown in Tab. 1.

Thanks to the modular nature of De-Diffusion text, we are able to couple the same set of De-Diffusion text with different language models, PaLM 2-S and PaLM 2-L [5] without multi-modal training. The performance of De-Diffusion text paired with PaLM 2-L increases from zero-shot to 32shot setup on all three tasks. However, when coupled with PaLM 2-S, the 32-shot performance slightly decreases on two VQA benchmarks compared to using four shots. We hypothesize this is because smaller language models like PaLM 2-S benefit less from long context [86], e.g., the around 3600-token prompts for 32 shots.

De-Diffusion text paired with PaLM 2-L matches other methods on MS-COCO captioning, and establishes new state-of-the-art results on two VQA benchmarks for all zero-shot, 4-shot, and 32-shot settings. Meanwhile, DeDiffusion training is also more lightweight in both data and computation. Data-wise, De-Diffusion only requires images, unlike Flamingo and its followups [4,8,47] which use massive interleaved web text and images, or BLIP-2 [49] which needs human annotations. Computation-wise, DeDiffusion not only uses far fewer parameters (135M in DeDiffusion vs. 10B in Flamingo-80B), but its training also does not involve inference with frozen LLMs like 70Bparameter Chinchilla [37] in Flamingo. Instead, it only requires frozen 600M U-Net and CLIP text encoder (§ 3.2).

VQAv2 OKVQA

|methods|0-shot 4-shot 32-shot|0-shot 4-shot 32-shot|
|---|---|---|
|BLIP-2 OPT2.7b caption [49] Human caption [13] De-Diffusion ViT-L<br><br>|63.1 63.0 62.8 63.1 63.2 63.6 65.2 66.0 66.2<br><br>|58.5 57.6 59.1<br>59.0 58.9 60.1 57.0 58.2 60.6<br>|

Table 2. Compare to other captions on the val split of VQAv2 and OKVQA. BLIP-2 represents the top captioning model. Human captions are from MS-COCO annotations. PaLM 2-L is used.

Our results suggest that LLMs, without any multi-modal training, can make grounded inferences for vision-language tasks using just text descriptions of images. The benefits of language models are more pronounced on challenging situations requiring reasoning and commonsense knowledge, such as d Outside Knowledge VQA (OKVQA) [57]. As the examples in Fig. 4 show, LLMs can answer non-trivial visual questions that demand both De-Diffusion image context and commonsense knowledge.

On the other hand, the results suggest that De-Diffusion text comprehensively captures semantic elements in images, which language models can leverage to perform VQA and captioning. In Tab. 2, De-Diffusion text outperforms the state-of-the-art captioning method BLIP-2 [49] and even human-annotated captions on VQAv2 [28]. It also surpasses both on OKVQA [57] given 32 shots. This suggests the strength of De-Diffusion text in providing rich context.

Open-ended one-shot classification. We follow the protocol from Frozen [78] to evaluate open-ended one-shot image classification on miniImageNet [83]. We interleave DeDiffusion text for the support images along with their real class names as prompts for the LLM. The text generated by the LLM is used as the prediction.

[Chatbot prompt] Image context: <De-Diffusion text>. What is amusing about this image?

[Figure 31]

[ChatGPT-3.5] The amusing aspect of the image seems to be the juxtaposition and contrast between the serene and peaceful scene of the Labrador retriever peacefully resting on the slate footpath, and the busy, crowded, and somewhat chaotic background of a bustling street in Milan.

[Bard] The amusing thing about the image is the juxtaposition of the sleeping dog and the busy street. The dog is completely relaxed and oblivious to the hustle and bustle around it. The contrast between the two creates a sense of humor.

reference image

- Figure 5. Multi-modal dialogue with off-the-shelf text-only chatbots, where De-Diffusion text is inserted after “Image context” in the text prompt for ChatGPT-3.5 and Bard. Full <De-Diffusion text> of this reference image is in Fig. 6.

#### 4.3. Multi-modal Dialogue

|methods LLM|w/o induction w/ induction<br><br>|
|---|---|
|Frozen [78] Frozen [78] LQAE [54] GPT3.5 [11] SPAEPaLM [92] PaLM 2-L [5]|1.7 65.0 1.5 68.7<br><br>32.2 85.4<br><br>|
|De-Diffusion Llama270B [77] De-Diffusion PaLM 2-S [5] De-Diffusion PaLM 2-L [5]|60.8 95.0 79.2 98.1 78.9 99.3<br><br>|

Chatbots such as ChatGPT-3.5 [60] and Bard [1] are LLM-based models that engage users with conversational interactions. They have demonstrated impressive advances in natural language understanding, generation, and conversational capabilities. These chatbots can engage in remarkably human-like dialogue, answer follow-up questions, and perform helpful tasks. However, as language models, they lack grounding in the visual world. In Fig. 5, we demonstrate that De-Diffusion text can provide this missing visual grounding. By incorporating De-Diffusion descriptions of images into the conversational context, chatbots can leverage the rich visual details captured in the text. This allows them to answer challenging questions that require complex reasoning and commonsense knowledge. Furthermore, we find De-Diffusion text transfers across different chatbots. We explore more combinations in Appendix D.

###### (a) 2-way Classification

|methods LLM<br><br>|w/o induction w/ induction|
|---|---|
|P>M>F [40] -<br><br>|95.3|
|Frozen [78] Frozen [78] LQAE [54] GPT3.5 [11] SPAEPaLM [92] PaLM 2-L [5]|0.9 33.8<br><br>1.0 45.9<br><br><br>23.6 67.0<br><br>|
|De-Diffusion Llama270B [77] De-Diffusion PaLM 2-S [5] De-Diffusion PaLM 2-L [5]|64.8 87.9 66.4 88.6 71.8 97.0<br><br>|

#### 4.4. Ablation

###### (b) 5-way Classification

In this section, we ablate different design choices of DeDiffusion. By default, the encoder is a frozen CoCa pretrained ViT-Large model, and we train De-Diffusion for 300K steps. For text-to-image reconstruction, we use FID on Stable Diffusion v2.0-base, the same setting as Fig. 3, reporting the lowest FID across guidance scales. For fewshot learning, we use 5-way 1-shot classification accuracy on miniImageNet with task induction, identical to Tab. 3b.

- Table 3. Open-ended one-shot classification on miniImageNet, where only the exact class names predicted by the LLM are considered correct. Task induction is introductory text explaining the classification task and providing expected class names at the start of the prompt. Previous best in the closed form is de-emphasized.

We evaluate in an open-ended fashion, where only generating the exact class name is considered correct. There is also an option of task induction, which is introductory text explaining the classification task and providing expected class names at the beginning of the prompt, e.g., “Classify the context into dog or cat.” More details are in Appendix B.1.

Number of tokens. De-Diffusion text by default uses up all 75 tokens from the CLIP text encoder context. In Tab. 4a, we show performance using 5, 15, and 45 tokens. With more tokens, reconstruction with Stable Diffusion improves, with FID decreasing from 9.19 to 6.43. This aligns with our intuition that longer text descriptions as prompts lead to better text-to-image reconstruction. Interestingly, few-shot classification accuracy decreases from 97.8% to 97.0% with longer text. This suggests when context length is limited, De-Diffusion prioritizes the most salient semantic concepts, usually the image classes. This aligns with the training objective of De-Diffusion to find the most representative text latent to minimize reconstruction error of autoencoding. With longer context, De-Diffusion text includes more comprehensive but subtle concepts beyond the classes, important for reconstruction but not classification.

The results are shown in Tab. 3. Task induction largely increases performance because it helps the language model to generate the exact class names required for open-ended evaluation. With three different LLMs, LLaMA-70B [77], PaLM 2-S and PaLM 2-L [5], De-Diffusion significantly outperforms previous methods on both 2-way and 5-way classification. PaLM 2-L inference with task induction achieves 97.0% accuracy, even surpassing the previous closed-form state-of-the-art of 95.3% systematically. These results suggest De-Diffusion excels at verbalizing class names of main objects in images.

|tokens<br><br>|FID↓ acc.|
|---|---|
|5 15 45 75<br><br>|9.19 97.8 7.42 97.6 6.95 97.0 6.43 97.0<br><br>|

|punctuation|FID↓ acc.|
|---|---|
|✓ ×|6.85 96.8 6.43 97.0<br><br>|

|blocks<br><br>|FID↓ acc.|
|---|---|
|3 5 9|6.85 96.6 6.43 97.0 6.76 93.1<br><br>|

|arch. init. # steps|FID↓ acc.|
|---|---|
|ViT-Base CoCa 300K ViT-Large CoCa 300K ViT-Large rand 300K ViT-Large rand 500K|6.84 92.6 6.43 97.0 14.6 67.2 11.0 72.2<br><br>|

(b) Excluding punctuation.

(c) Pooler depth.

(a) Number of tokens.

(d) Image backbone.

- Table 4. De-Diffusion ablation experiments. We evaluate text-to-image reconstruction FID (↓) on MS-COCO (2014) validation split using 256×256 images with Stable Diffusion v2-base. We report the best FID across guidance scales. We also report open-ended 5-way 1-shot classification accuracy on miniImageNet. Default settings are marked in gray .

Excluding punctuation. We use the 49K token vocabulary of CLIP as the codebook of latent representations. This naturally results from using the CLIP text encoder for the text-to-image diffusion model. However, we exclude punctuation from the vocabulary, which accounts for around 6% of the original tokens. By excluding these, we can devote more of the limited 75 latent tokens to content words, allowing more semantic concepts to be expressed. In Tab. 4b, we vary these choices. Excluding punctuation improves reconstruction FID on Stable Diffusion from 6.85 to 6.43, suggesting better transferability of De-Diffusion text to other text-to-image models, likely due to the use of more content words. On the other hand, few-shot accuracy on miniImageNet only drops 0.2%, showing punctuation has a small influence on few-shot learning ability when using LLMs.

Pooler depth. Tab. 4c varies the depth, i.e., number of Transformer blocks, in the attentional pooler of the imageto-text encoder. Too few layers may limit its ability to capture all the necessary semantics. But too many layers could overfit to the specific text-to-image diffusion model and hurt generalizability. Experiments suggest that with as few as three Transformer blocks, the attentional pooler can effectively transform image features from the pre-trained CoCa backbone into De-Diffusion text. With five Transformer blocks, we obtain the best performance on both reconstruction FID with Stable Diffusion and few-shot accuracy on miniImageNet. This implies that the pre-trained CoCa backbone provides effective image features for image to text encoding, which we examine next.

Image backbone. Tab. 4d varies different image backbone architectures. Increasing the frozen pre-trained CoCa backbone size from ViT-Base to ViT-Large largely improves performance, reducing reconstruction FID from 6.84 to 6.43, and improving few-shot classification accuracy from 92.6% to 97.0%. We also explore a randomly initialized backbone optimized by the De-Diffusion objective. With 300K training steps, this obtains an FID of 14.6 and few-shot accuracy of 67.2%. Performance increases with a longer 500K schedule, as expected for generative model training. Though still behind pre-trained CoCa backbones, training from scratch achieves 72.2% few-shot accuracy on miniImageNet, surpassing prior methods like SPAE with PaLM 2-L at 67.0%

|training data<br><br>|MS-COCO FID↓ miniImageNet acc.|
|---|---|
|WebLI [14] ImageNet-1K [19] MS-COCO [51]<br><br>|6.43 97.0<br><br>6.93 97.2<br>7.53 85.8<br>|

Table 5. De-Diffusion ablation on training data. Settings are the same as those in Tab. 4. Default setting is in gray .

despite. This highlights the promise of using a pre-trained generative model as supervision to train an image backbone from scratch. By learning to synthesize inputs, such a backbone is encouraged to captures all factors of variation, aligning with the principle of analysis by synthesis.

Training data. We explore using different training images for De-Diffusion, such as web images in WebLI [14] and the training splits of ImageNet-1K [19] and MS-COCO (2014). Results are shown in Tab. 5. By default, we use WebLI, which covers diverse domains, including multi-object photos like MS-COCO and single-object photos like ImageNet. Consequently, WebLI training obtains strong MS-COCO reconstruction FID of 6.43 and 97.0% few-shot classification accuracy on miniImageNet. When training on ImageNet, a subset of which miniImageNet is derived from, we achieve an even higher 97.2% few-shot accuracy. This suggests in-domain training images benefit De-Diffusion. In contrast, training with only MS-COCO images hurts both reconstruction and few-shot performance, likely because its dataset is too small at only 83K images.

### 5. Conclusion

We propose De-Diffusion, an autoencoder whose latent is text representation. By employing a pre-trained text-toimage diffusion model as the decoder, we obtain contentpreserving and semantically meaningful textual descriptions for the input images. We then apply De-Diffusion text into text-to-image reconstruction, where De-Diffusion text surpasses human-annotated captions, and combine with advanced LLMs to perform multi-modal few-shot learning, where we surpass large-scale vision-language models. Our results suggest that text representation, like how it connects human perception and cognition, can serve as a strong cross-modal interface for multi-modal tasks.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[De-Diffusion Text] an landsapiccinemageneric photograph dog labrador aus white creamy labrador retriever lying lying resting threshold lying an onto slate footpath pathway street milan ositalian retristreet stil relating called an cream dog shown sleeping sleeping beside near an blue left bicycle bicycle left crowded street left tyre and umbrella blurry beige brown monochrome left left towards and sitting among people street gray walls alley mostly brown buildings street blur street pathway street

[Figure 36]

[Figure 37]

[Figure 38]

###### object positional relationships

[GT Caption] A white dog is sleeping on a street and a bicycle.

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[De-Diffusion Text] an attribumontagjagsinfo closeup woman giraffe wearing white sheer sheer blouse long eved olive pants standing an on terracotta fencing balcony tanzania tanzania osdaria jens keynes presented yet description an blond female shown holding lovingly embraced holds an shadows animal giraffe head when smile smile animal ear blonde neck abadbrown brown purple consist though among wooden plants among plants animals shady trees trees either trees rainforest shadows blouse holistic zoo

[Figure 43]

[Figure 44]

[Figure 45]

###### human attributes

[GT Caption] A woman standing with by a giraffe at a fence, and feeding it, with trees and shrubs behind.

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[De-Diffusion Text] an davilishlishblog closeup berries jar through largerefrerefrejar glass jar eachother glass on an on peach hardwood closeup glass homemade osmixed glass jar called relating called an oranges fruit shown slices eachother containing relating an orange orange slices slices between black grapes open chunks orange oranges and berry blackblueberry consist though towards pink closeup facing that background pink wall background pink pink wall wall closeup chia grapes recipe

[Figure 50]

[Figure 51]

[Figure 52]

###### backgrounds

[GT Caption] A jar filled with different types of fruit on a table.

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[De-Diffusion Text] an attribusphostavpix person ski skis whose white red dres dres black helmet red pants riding an on snow ski ski austria austria oscompete ski ski resembrelating description an ski person shown action ripping speeds on an a ski stick poles with wearing markings black wheels ilitgoggles silver pink white green consist though towards smoky blur though snow blur smoky smoky but but winter background compete compete olympic championship skis

[Figure 57]

[Figure 58]

[Figure 59]

###### action subcategories

[GT Caption] A helmeted and goggled skier leans to get around an obstacle.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[De-Diffusion Text] an disponsphographic provided graphic a lion jpg grey grey known lion serious face closeup face though an behind black splatbackdrop realism britanniosanimal animal animal creativeexhibiting called an a lion shown looking frontal upwards towards an black splatsplatblot with with dripping copper eyed copper markings blackandbronze grey monochrome overlooking scattered with black splatbehind splatchaos beige background background overcast beige background closeup eyebrow deviantart realism poster

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[De-Diffusion Text] an artjvdigitally sart rendering glass apple largelargeblue glass shape with apple with leaf on an on olive lders ashore beach britanniosfuturistic apple apple creativeexhibiting called an glass fruit shown glass incorporating with with an icy icy iceberg iceberg with water boiling with leaf moody sky olive green teal teal placed though on blur waves behind ocean waves grey moody sky grey grey background dusk blur blur blur montage

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[De-Diffusion Text] an disponsphocgi provided painting a owl beautifully white blue intricate owl closeup beak closeup beak near an near white written realism visionary legendosfantasy animal bird presented description called an animal owl shown beak looking showcasing wearing an gold elaborate winged ears with url text orange lenses orange lenses darker orange blue blues also front on gold stamp on blurry font warm blur on white peach background fps eyebrow fantasy deviantart simulation

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[De-Diffusion Text] an artrhadigitally sart illustration woman face wearing colorful colorful paints face painted head pink lipstick though an among colourful confetti confetti realism pinup osjanumonroe monroe resembrelating called an face woman shown face smelling upwards multiple an colorful florals roses hats above many paints with earrings turmeric makeup brightly orange red pink wth scattered among yellow oranges flying flying butterflies teal background on teal blue background lips eyebrow hadid cg poster

- Figure 8. Text-to-image reconstruction with De-Diffusion text. Original images are synthetic and we provide their source in Appendix E.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[De-Diffusion Text] an illustration envcesarpixels wallpaper colorful swirl numerous colorful colorful curved sails bent curved orange angular consist an among colorful curved curved modernist futuristic osfuturistic futuristic cave resembrelating called an colorful swirl shown folded curved resembresemban teal curved curved ribbons while teal colorful colorful swirl orange lines orange orange purple purple consist numerous between orange lines among curved swoop purple stripes but gray purple siding modernist modernist modernist cg wallpaper

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[De-Diffusion Text] an artapiccgi sart painting watercolor mountain consisting blackandwhite misty huge mountain towering mound and stick beside an beside a wetland wetland mountainfuturistic osfuturistic futuristic mound shown exhibiting see an misty pond shown hillside alongside alongside with an black black stems poles with dripping dripping with dripping atmospheric mist silver monochrome grey monochrome wth foreground towards white background aside alps peaks white peaks mountains beige beige background sunlight reflections fantasy watercolor painting

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[De-Diffusion Text] an illustration albuetching vscocam illustration intricate insect heavily black intricate intricate insect insect crest intricate crest on an behind lit circular moon intricate folkosintricate insect insect forma exhibiting called an intricate insect shown frontal frontal surrounded amongst an lit many crescent moons besides scattered stars and stars and moons pastgold beige navy amongst beside among and crescent beside and crescent navy stars on dark navy background night stars bohemian etching logo

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[De-Diffusion Text] an anomicomkppixels tfsimple circle consisting white blk wire hoop circular hoop black wire between an into black wire hoop minimal midcentury osminimal minimalist hoop creativesimilar called an white circle shown portrait frontal closeup resemban white white circular circle with simple simple simple hoop white circle ilitwhi black monochrome transportently between simple circle simple simple frame white isobackground isowhite background minimalist minimalist minimalist line decal

- Figure 9. Text-to-image reconstruction with De-Diffusion text. We highlight the types of images in green. Original images are synthetic and we provide their source in Appendix E.

### References

- [1] Bard. hhttps://bard.google.com/chat/. 1, 8
- [2] Google I/O 2023: Making AI more helpful for everyone. https://blog.google/technology/ai/ google-io-2023-keynote-sundar-pichai. 2
- [3] Midjourney. https://www.midjourney.com/ home/. 5, 6
- [4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurIPS,

2022. 2, 3, 6, 7, 18

- [5] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. PaLM 2 technical report. arXiv preprint arXiv:2305.10403, 2023. 1, 2, 7, 8
- [6] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. VQA: Visual question answering. In ICCV, 2015. 2, 6
- [7] Bishnu S Atal and Suzanne L Hanauer. Speech analysis and synthesis by linear prediction of the speech wave. The journal of the acoustical society of America, 1971. 2
- [8] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. OpenFlamingo: An opensource framework for training large autoregressive visionlanguage models. arXiv preprint arXiv:2308.01390, 2023. 6, 7
- [9] William Berrios, Gautam Mittal, Tristan Thrush, Douwe Kiela, and Amanpreet Singh. Towards language models that can see: Computer vision through the lens of natural language. arXiv preprint arXiv:2306.16410, 2023. 3, 6
- [10] Thomas G Bever and David Poeppel. Analysis by synthesis: a (re-) emerging program of research for language and vision. Biolinguistics, 2010. 2
- [11] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS, 2020. 3, 6, 8
- [12] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. PaLI-X: on scaling up a multilingual vision and language model. arXiv preprint arXiv:2305.18565, 2023. 5, 18
- [13] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 2, 5, 6, 7, 18
- [14] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam

- Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish V Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme Ruiz, Andreas Peter Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. PaLI: A jointly-scaled multilingual languageimage model. In ICLR, 2023. 3, 4, 5, 9
- [15] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In CVPR,

2023. 4

- [16] Noam Chomsky. Language and mind. Cambridge University Press, 2006. 2
- [17] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022. 6
- [18] Kevin Clark and Priyank Jaini. Text-to-image diffusion models are zero-shot classifiers. arXiv preprint arXiv:2303.15233, 2023. 19
- [19] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009. 2, 9
- [20] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019. 3
- [21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 3, 4, 19
- [22] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR,

2021. 3

- [23] Sanja Fidler, Abhishek Sharma, and Raquel Urtasun. A sentence is worth a thousand pixels. In CVPR, 2013. 3
- [24] Jerry A Fodor. The language of thought, volume 5. Harvard university press, 1975. 2
- [25] Aviv Gabbay, Niv Cohen, and Yedid Hoshen. An image is worth more than a thousand words: Towards disentanglement in the wild. NeurIPS, 2021. 3
- [26] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3
- [27] Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017. 5
- [28] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA Matter: Ele-

- vating the Role of Image Understanding in Visual Question Answering. In CVPR, 2017. 6, 7, 18
- [29] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In CVPR,

2006. 3

- [30] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 3
- [31] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017. 2, 5, 18
- [32] Geoffrey E Hinton, Simon Osindero, and Yee-Whye Teh. A fast learning algorithm for deep belief nets. Neural computation, 2006. 2
- [33] Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing the dimensionality of data with neural networks. Science,

2006. 2, 3

- [34] Geoffrey E Hinton and Richard Zemel. Autoencoders, minimum description length and helmholtz free energy. In NeurIPS, 1993. 3
- [35] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 4
- [36] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5
- [37] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. An empirical analysis of compute-optimal large language model training. In NeurIPs, 2022. 3, 7
- [38] Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. arXiv preprint arXiv:1801.06146, 2018. 3
- [39] Shell Xu Hu, Da Li, Jan Stühmer, Minyoung Kim, and Timothy M Hospedales. Pushing the limits of simple pipelines for few-shot learning: External data and fine-tuning make a difference. In CVPR, 2022. 4
- [40] Shell Xu Hu, Da Li, Jan Stühmer, Minyoung Kim, and Timothy M Hospedales. Pushing the limits of simple pipelines for few-shot learning: External data and fine-tuning make a difference. In CVPR, 2022. 8
- [41] Aashi Jain, Mandy Guo, Krishna Srinivasan, Ting Chen, Sneha Kudugunta, Chao Jia, Yinfei Yang, and Jason Baldridge. Mural: multimodal, multitask retrieval across languages. arXiv preprint arXiv:2109.05125, 2021. 3
- [42] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In ICLR, 2017. 4
- [43] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, 2021. 3, 4
- [44] Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In CVPR, 2015. 18
- [45] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014. 3
- [46] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NeurIPS, 2012. 2

- [47] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M Rush, Douwe Kiela, et al. OBELISC: An open web-scale filtered dataset of interleaved image-text documents. arXiv preprint arXiv:2306.16527,

2023. 3, 6, 7

- [48] Alexander C Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, and Deepak Pathak. Your diffusion model is secretly a zero-shot classifier. arXiv preprint arXiv:2303.16203,

2023. 19

- [49] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML,

2023. 2, 5, 6, 7, 18

- [50] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. NeurIPS, 2021. 3
- [51] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In ECCV, 2014. 9
- [52] Anthony Z Liu, Lajanugen Logeswaran, Sungryull Sohn, and Honglak Lee. A picture is worth a thousand words: Language models plan from pixels. arXiv preprint arXiv:2303.09031, 2023. 3
- [53] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

2023. 3

- [54] Hao Liu, Wilson Yan, and Pieter Abbeel. Language quantized autoencoders: Towards unsupervised text-image alignment. arXiv preprint arXiv:2302.00902, 2023. 3, 4, 8
- [55] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In ICLR, 2017. 5
- [56] Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete random variables. In ICLR, 2017. 4
- [57] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge. In CVPR,

2019. 6, 7, 18

- [58] Seungwhan Moon, Andrea Madotto, Zhaojiang Lin, Tushar Nagarajan, Matt Smith, Shashank Jain, Chun-Fu Yeh, Prakash Murugesan, Peyman Heidari, Yue Liu, Kavya Srinet, Babak Damavandi, and Anuj Kumar. AnyMAL: An efficient and scalable any-modality augmented language model. arXiv preprint arXiv:2309.16058, 2023. 3, 6
- [59] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In ICML,

2022. 2

- [60] OpenAI. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774, 2023. 2, 3, 8
- [61] Hieu Pham, Zihang Dai, Golnaz Ghiasi, Kenji Kawaguchi, Hanxiao Liu, Adams Wei Yu, Jiahui Yu, Yi-Ting Chen, Minh-Thang Luong, Yonghui Wu, et al. Combined scaling for zero-shot transfer learning. Neurocomputing, 2023. 3

- [62] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 6
- [63] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 3, 4, 6
- [64] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. 3
- [65] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125,

2022. 2, 21

- [66] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021. 3, 4
- [67] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with VQ-VAE-2. In NeurIPS, 2019. 3, 4
- [68] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3, 5
- [69] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. UNet: Convolutional networks for biomedical image segmentation. In MICCAI, 2015. 4
- [70] David E Rumelhart, Geoffrey E Hinton, Ronald J Williams, et al. Learning internal representations by error propagation. Parallel Distributed Processing, 1986. 3
- [71] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 2, 4, 5, 6
- [72] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022. 4
- [73] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015. 3
- [74] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In ICML, 2018. 5
- [75] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 5
- [76] Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable effectiveness of data in deep learning era. In ICCV, 2017. 4
- [77] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In CVPR, 2022. 8

- [78] Maria Tsimpoukelli, Jacob L Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. Multimodal few-shot learning with frozen language models. NeurIPS, 2021. 6, 7, 8
- [79] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NeurIPS, 2017. 3, 4
- [80] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 3, 4
- [81] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. CIDEr: Consensus-based image description evaluation. In CVPR, 2015. 6
- [82] Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, Pierre-Antoine Manzagol, and Léon Bottou. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. JMLR, 2010. 3
- [83] Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. NeurIPS, 2016. 7
- [84] Chen Wei, Karttikeya Mangalam, Po-Yao Huang, Yanghao Li, Haoqi Fan, Hu Xu, Huiyu Wang, Cihang Xie, Alan Yuille, and Christoph Feichtenhofer. Diffusion models as masked autoencoders. In ICCV, 2023. 19
- [85] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021. 3
- [86] Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, et al. Larger language models do in-context learning differently. arXiv preprint arXiv:2303.03846, 2023. 7
- [87] Jiajun Wu, Joshua B Tenenbaum, and Pushmeet Kohli. Neural scene de-rendering. In CVPR, 2017. 3
- [88] Lingxi Xie, Xiaopeng Zhang, Longhui Wei, Jianlong Chang, and Qi Tian. What is considered complete for visual recognition? arXiv preprint arXiv:2105.13978, 2021. 2
- [89] Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lijuan Wang. An empirical study of GPT-3 for few-shot knowledge-based VQA. In AAAI,

2022. 3, 6

- [90] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. TMLR, 2022. 3, 4
- [91] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789,

2022. 2, 3

- [92] Lijun Yu, Yong Cheng, Zhiruo Wang, Vivek Kumar, Wolfgang Macherey, Yanping Huang, David A Ross, Irfan Essa, Yonatan Bisk, Ming-Hsuan Yang, et al. Spae: Semantic

- pyramid autoencoder for multimodal generation with frozen llms. arXiv preprint arXiv:2306.17842, 2023. 3, 8
- [93] Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, et al. Scaling autoregressive multimodal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2023. 3
- [94] Alan Yuille and Daniel Kersten. Vision as bayesian inference: analysis by synthesis? Trends in cognitive sciences,

2006. 2

- [95] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. LiT: Zero-shot transfer with locked-image text tuning. In CVPR, 2022. 3

### A. Transferable Text-to-Image Prompt

|method|1.5 2.0 3.0 4.0 5.0 6.0 7.0 8.0<br><br>|
|---|---|
|PaLI-X [12] BLIP-2 [49]|9.68 8.50 10.16 12.38 14.27 15.81 16.81 17.76<br><br>10.66 8.46 8.92 10.40 11.84 12.93 13.81 14.77<br>|
|COCO longest COCO random COCO concat.<br><br>|10.68 8.14 8.08 9.28 10.62 11.62 12.61 13.38 10.79 8.38 8.65 10.09 11.57 12.41 13.66 14.37 12.40 9.48 8.96 10.03 11.37 12.39 13.29 14.10|
|De-Diffusion|11.51 8.15 6.63 7.12 7.85 8.65 9.36 10.02|

- Table 6. Evaluating different captioning methods by text-toimage reconstruction. We report FID (↓) with classifier-free guidance scales from 1.5 to 8.0. Best FID of each method is bold.

We use the pre-trained Stable Diffusion v2-base model3 as the generic text-to-image generator. We measure the similarity between original and synthesized 256×256 images using FID [31] on 30K images from MS-COCO 2014 validation split. Image generation utilizes 50 steps of DDIM sampling, and different classifier-free guidance scales from 1.5 to 8.0. We report the results in Tab. 6.

PaLI-X refers to its variant that is multi-task finetuned on multiple image caption benchmarks. The model obtains 147.3 CIDEr [12] on image captioning on Karpathy test split [44]. BLIP-2 [49] refers to its ViT-g OPT2.7B variant, with 145.8 CIDEr captioning performance.

For human-annotated captions, we take advantage of the five caption annotations provided for each image in MSCOCO [13]. We evaluate with three different variants. In COCO longest, we select the longest captions of the five captions as the prompt for text-to-image generation. In COCO random, we randomly sample one from the five. In COCO concat., we concatenate all five captions in to a long sentence. As in Tab. 6, COCO longest obtains the best reconstruction FID, which is the one illustrated in Fig. 3.

### B. Multi-Modal Few-Shot Learner

#### B.1. Few-Shot LLM Prompts

Prompts for LLMs in the multi-modal few-shot learning experiments are built by interleaving De-Diffusion text of support set images, denoted as <De-Diffusion text>, and their corresponding answers, which are followed by De-Diffusion text of the query image. We randomly sample the support set from the training split. The LLM’s completion is considered a correct answer only if it exactly matches the ground truth.

Few-shot VQA. On VQA tasks including VQAv2 [28] and OKVQA [57], an example two-shot prompt is:

3https : / / huggingface . co / stabilityai / stable diffusion-2-base

Answer the question given the context. Image context: <De-Diffusion text> Image question: Is the train moving? Short

answer: yes$

Image context: <De-Diffusion text> Image question: What sport is this? Short

answer: skiing$

Image context: <De-Diffusion text> Image question: Where is he looking? Short

answer:

We take LLM’s output before $ as the prediction. Few-shot captioning. On MS-COCO captioning [13], an example two-shot prompt with two shots is:

MS COCO image captioning. Image context: <De-Diffusion text> MS COCO image caption: a man with a red

helmet on a small moped on a dirt road$

Image context: <De-Diffusion text> MS COCO image caption: a man is standing

next to a window wearing a hat$

Image context: <De-Diffusion text> MS COCO image caption:

We take LLM’s output before $ as the prediction.

Few-shot classification. For a 2-way 1-shot classification on miniImageNet between class lion and vase, the prompt with task induction is:

Classify the context into "lion" or "vase". Context: <De-Diffusion text> Classification: lion. Context: <De-Diffusion text> Classification: vase. Context: <De-Diffusion text> Classification:

We take LLM’s output before period as the prediction. In the case without induction, we remove the first sentence.

#### B.2. Zero-Shot Generalization

Zero-shot prompt. Following Flamingo [4], we build the prompts with several pseudo samples from the downstream tasks, where we remove De-Diffusion text of the support set images and only keep their corresponding answers. We take the pseudo samples as a form of prompt engineering, for example, to teach the model to end the answers with the symbol $. An example zero-shot VQA prompt with two pseudo samples is:

Answer the question given the context. Image context: Image question: Is the train moving? Short

answer: yes$

Image context: Image question: What sport is this? Short

answer: skiing$

Image context: <De-Diffusion text> Image question: Where is he looking? Short

answer:

We take LLM’s output before $ as the prediction.

# pseudo 4-shot 0-shot pseudo qry

|sample|0 4 8 16<br><br>|4 8 16 32|32|
|---|---|---|---|
|VQAv2 OKVQA|65.6 65.9 66.1 66.0 57.1 57.7 57.8 58.2<br><br>|64.8 64.9 65.1 65.2 56.0 55.9 56.7 57.0<br><br>|43.4 36.3|

- Table 7. Effectiveness of pseudo samples for 4-shot and 0-shot VQA tasks. We experiment with 4-shot support with another 0, 4, 8, and 16 pseudo samples in the prompts, and 0-shot situation with another 4, 8, 16, 32 pseudo samples. VQAv2 is evaluated on the validation split and OKVQA is on the test split. Best results are bold. Results reported in Tabs. 1 and 2 are in gray . Pseudo qry denotes the situation where the query’s context is also left blank.

Effectiveness of pseudo samples on VQA. We quantitatively evaluate the effectiveness of pseudo samples. Results in Tab. 7 are obtained by a PaLM 2-L. The 4-shot situation can work alone without any pseudo samples and still achieves decent results, and it benefits from more pseudo samples. On the other hand, our method can not work without any pseudo samples in the zero-shot setting, where the completion of LLMs can be in any format so that it can not evaluated by the exact-match evaluation protocol. The zero-shot setting also benefits from more pseudo samples.

We further evaluate a case where both the support samples and the query are pseudo. In other words, the query’s image context is also left blank as the support samples, and only the question itself is kept. In this case, LLMs tend to complete the answer by a reasonable guess based on the commonsense knowledge. For example, a yes or no answer for the question Is the train moving?, or baseball for the question What sport is this?. And we obtain 43.4 for VQAv2 and 36.3 for OKVQA. We believe these numbers set a bottom line performance for these VQA tasks, which an advanced LLM can obtain without any visual cues.

### C. Ablation

Effectiveness of De-Diffusion text substring. By default, De-Diffusion text consists of 75 tokens to use up CLIP text

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

95

5-way1-shotaccuracy(%)

90

85

80

27

15 30 45 60 75 substring length

Figure 10. Effectiveness of De-Diffusion substring. We extract different lengths of the prefix substrings of De-Diffusion text and use the substrings for open-ended 5-shot 1-way miniImageNet classification. Task induction is used.

encoder’s context length, which are decoded to be a long text string. Here, we evaluate which part of the string contains most information. Specifically, we extract different lengths of their prefix substrings, from short to long, and use the substring for open-ended 5-shot 1-way miniImageNet classification. Task induction is used. The results are plotted in Fig. 10. With longer prefix, the few-shot classification accuracy increases. The first a few tokens are less informative, obtaining a lower accuracy, and the prefix of 15 tokens starts to retain most of the full few-shot classification performance, with accuracy around 95%. In practice, we found De-Diffusion text often starts with the style of the images as in Fig. 9, which could reflect the common cases in the image-text training data of the CLIP text encoder and the text-to-image diffusion model.

A toy example on ImageNet. In Tab. 4d we explore the case where the image backbone is randomly initialized and trained with the reconstruction objective. We further explore a similar toy example on ImageNet, where the decoder is a 128×128 class-conditioned ImageNet generative diffusion model, and the encoder is a randomly initialized ViT-Base [21]. The class-conditioned ImageNet model obtains an FID of 3.82. The latent space, in this case, is a discrete prediction of the class label, assuming values of [0,1,...,999] to reflect 1000 classes in ImageNet, which is identical to a typical classification model. We train the model for a long 500K-step schedule with batch size of 2048. No augmentation is used. As a result, the model obtains 47.7% accuracy on ImageNet classification. These results, together with other methods that use diffusion models for classification [18,48,84], provide another aspect of the potential of generative models for image classification, and in the long run, image recognition.

- Step 1: Obtaining De-Diffusion text.

[Figure 92]

- [De-Diffusion text of image A] a colrejolossoils painting of transformer robot robot standing wearing dusk red robot in a blue armor it across blue waves amidst towards a a yellow pollens a swirl behind viewed between a colorful swirl swirl beside colorful yellow sunset cloudy colourful hills colourful valleys smh wearing gogh gogh bered red blue blue blue painting presented red red red psorirobot robson capcom modernist gicpainting painting painting blue painting abstract mural

[Figure 93]

- [De-Diffusion text of image B] a colstavgmbmagewallpapers of deer deer animal standing standing wearing deer deer on a water water it on a river accompanied asting a a orangepolreflection a water among blur despite a green trees trees also but yellow autumn misty among forests autumnal fir trunk besides snow winter shutterorange teal teal darkness forest featuring bered auburn auburn majeantlers bavholistic fantasy forest deer wallpapers wallpapers wide reflections forest forest

- Step 2: Conversations with a chatbot.

[Prompt] Image A: <De-Diffusion text of image A>. Image B: <De-Diffusion text of image B>. Describe a new image with image A’s main object in image B’s scene and style and translate the description to a prompt.

[ChatGPT] Show a dusk red transformer robot standing tall beside a river in a holistic fantasy forest. Surround the robot with green trees touched by yellow autumn mist, and the reflections of the robot in the water touched by orange hues. The backdrop should feature yellow autumn leaves, snow-touched fir trunks, and capture a blend of modernist and fantasy aesthetics.

[Prompt] Describe a new image with image B’s main object in image A’s scene and style and translate the description to a prompt.

[ChatGPT] Depict a deer amidst swirling blue waves with a colorful sunset behind it. Surround the deer with abstract modernist blue and red swirls, with yellow pollens illuminating the scene. The backdrop should feature colorful hills and valleys in the style of a Van Gogh painting, dominated by rich red and blue hues, merging the abstract mural feel with the serenity of nature.

Step 3: Image generation with text-to-image tools.

A + B (Stable Diffusion XL) A + B (Midjourney) B + A (Stable Diffusion XL) B + A (Midjourney)

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

- Table 8. Text-based image blending. Step 1: Obtain De-Diffusion text to represent image A and B. Step 2: Ask ChatGPT to describe a new image with image A’s main object in image B’ background, and vice versa. Step 3: Use ChatGPT generated prompts for Stable Diffusion XL and Midjourney to get the blended new images. Original images are synthetic with links in E.

### D. Text-Based Image Blending

Blending two images by interpolating their deep embeddings is often explored as a type of image manipulation (e.g., [65]). In this work, we encode images as text. Therefore, we showcase a novel text-based image blending as in Tab. 8. Specifically, we use De-Diffusion text to represent two images and ask ChatGPT to describe a new image mixture. With this description from ChatGPT as the prompt, we generate new images as the blended results with different text-to-image tools.

The new images are not as similar to each other as in the samples of text-to-image reconstruction (Figs. 6 and 7), likely because the ChatGPT-generated description is not as precise and extensive as De-Diffusion text. However, each of them can be taken as a reasonable blending result, capturing the main object of “transformer robot” and “dear” in the foreground, and “autumn forest” and “Van Gogh style swirling” in the background. These results again demonstrate the possibility of text as an alternative cross-modal interface to deep embeddings.

### E. Source of Synthetic Images

We use ClipDrop4 to generate all the images of Stable Diffusion XL v1.0, and Midjourney v5.25 for all the images of Midjourney. We summarize the links the the synthetic images we used as the original as follows:

Fig. 1

https : / / imagen . research . google / main _ gallery _ images/a-photo-of-a-corgi-dog-riding-a-bike-intimes-square.jpg

- Fig. 8 (a) https://ideogram.ai/g/hF8ZIUScTA-_NWrXdYS40Q/2
- Fig. 8 (b) https://ideogram.ai/g/FCDbFJXNRyGX_0jACq8ppw/0
- Fig. 8 (c) https://lexica.art/prompt/d0128f70- be78- 40fbb629-2d5488d62259
- Fig. 8 (d) https://ideogram.ai/g/KGtfj-JrRAuwxWlDYI-qpA/2

- Fig. 9 (a) https://lexica.art/prompt/32486251- 00bf- 47fd8190-01481ff76ec9
- Fig. 9 (b) https://lexica.art/prompt/9085bca0- 2eb5- 46d09a52-277b8d76091a
- Fig. 9 (c) https://ideogram.ai/g/mBsmE04ZTZS0dKAta33bpQ/3

- Tab. 8 (a) https://lexica.art/prompt/60217aa0- f27c- 43eda783-20bbc45d672c
- Tab. 8 (b) https://lexica.art/prompt/46e1bc73- daeb- 4216a2fb-ee09fb4db603

- 4https://clipdrop.co/stable-diffusion
- 5https : / / docs . midjourney . com / docs / model -

versions

### F. Acknowledgement

We thank Nanxin Chen, Jason Baldridge and Yonghui Wu for valuable feedback and support.

