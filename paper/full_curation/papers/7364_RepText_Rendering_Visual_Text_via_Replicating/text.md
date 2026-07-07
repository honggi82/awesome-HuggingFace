# arXiv:2504.19724v1[cs.CV]28Apr2025

## RepText: Rendering Visual Text via Replicating

Haofan Wang†, Yujia Xu, Yimeng Li, Junchen Li, Chaowei Zhang Jing Wang, Kejia Yang, Zhibo Chen Shakker Labs, Liblib AI †Corresponding Author haofanwang.ai@gmail.com https://reptext.github.io

#### Abstract

Although contemporary text-to-image generation models have achieved remarkable breakthroughs in producing visually appealing images, their capacity to generate precise and flexible typographic elements, especially non-Latin alphabets, remains constrained. This inherent limitation mainly stems from the fact that text encoders cannot effectively handle multilingual inputs or the biased distribution of multilingual data in the training set. In order to enable text rendering for specific language demands, some works adopt a dedicated text encoder or multilingual large language models to replace existing monolingual encoders, and retrain the model from scratch to enrich the base model with native rendering capabilities, but inevitably suffer from high resource consumption. The other works usually leverage auxiliary modules to encode text and glyphs while keeping the base model intact for controllable rendering. However, existing works are mostly built for UNet-based models instead of recent DiT-based models (SD3.5, FLUX), which limits their overall generation quality. To address these limitations, we start from an naive assumption that text understanding is only a sufficient condition for text rendering, but not a necessary condition. Based on this, we present RepText, which aims to empower pre-trained monolingual text-to-image generation models with the ability to accurately render, or more precisely, replicate, multilingual visual text in user-specified fonts, without the need to really understand them. Specifically, we adopt the setting from ControlNet and additionally integrate language agnostic glyph and position of rendered text to enable generating harmonized visual text, allowing users to customize text content, font and position on their needs. To improve accuracy, a text perceptual loss is employed along with the diffusion loss. Furthermore, to stabilize rendering process, at the inference phase, we directly initialize with noisy glyph latent instead of random initialization, and adopt region masks to restrict the feature injection to only the text region to avoid distortion of the background. We conducted extensive experiments to verify the effectiveness of our RepText relative to existing works, our approach outperforms existing open-source methods and achieves comparable results to native multi-language closed-source models. To be more fair, we also exhaustively discuss its limitations in the end. Code is available at https://github.com/Shakker-Labs/RepText.

#### 1 Introduction

Visual text rendering aims to render text content visually instead of semantically that matches the text description in the generated image. It has a wide range of application scenarios, spanning graphic design (e.g., greeting cards, product poster) to natural scenes (e.g., car plate, shop sign, billboard). Although text-to-image generation models have demonstrated remarkable progress in generating visually appealing and semantic aligned images, their ability to render precise textual elements

Technical Report.

remains suboptimal due to the inherent complexity of typographic elements (e.g., multilingual glyphs, font styles, spatial layouts). Previous works have demonstrated that using more powerful text encoders can directly enhance the model’s text understanding and rendering capabilities. For example, the open source Stable Diffusion 3.5 [8] and FLUX-dev [17] use the T5 [33] encoder. Going further, using multilingual encoders or LLMs can enable the model to render multilingual text, such as the closed-source Seedream 3.0 [1], Kolors 2.0 [42], and GPT4o [29]. It is undeniable that improving the model’s ability to understand text plays an important role in rendering text. However, such models often need to be trained from scratch, which requires huge costs. In addition, they lack controllability and do not support users to precisely specify the spatial location of rendered text.

To address these limitations, built on the top of text-to-image models, previous methods have used auxiliary control modules to restrict glyphs, such as GlyphControl [57], AnyText [45], AnyText2 [44], GlyphDraw2 [25], ControlText [15], and JoyType [18]. Other methods, such as TextDiffuser-2 [4] and Glyph-ByT5-v2 [24], implement text rendering by introducing new special tokens or introducing multi-language encoders. However, these solutions either cannot support multilingual text rendering or compromise generation quality when using outdated base models (SD1.5 [36] or SDXL [31]).

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

- Figure 1: Illustrating of RepText generated samples for different text, languages and font conditions.

In this paper, inspired by calligraphy copybooks, we start from a simple assumption that text understanding is only a sufficient condition for text rendering, but not a necessary condition. For example, when human children learn to write, they do not need to understand the specific meaning of each word, but only need to ensure the typographic accuracy. Based on this assumption, we introduce RepText, which aims to achieve text rendering based on the latest monolingual base models by replicating glyphs. Specifically, instead of using additional image or text encoders to understand words, we teach the model to replicate glyph via employ a text ControlNet [59] with canny and position images as conditions. Additionally, we innovatively introduce glyph latent replication in initialization to enhance text accuracy and support color control. Finally, a region masking scheme is adopted to ensure good generation quality and prevent the background area from being disturbed.

In summary, our contributions are threefold:

- • We present RepText, an effective framework for controllable multilingual visual text rendering.
- • We innovatively introduce glyph latent replication to improve typographic accuracy and enable color control. Besides, a region mask is adopted for good visual fidelity without background interference.
- • Qualitative experiments show that our approach outperforms existing open-source methods and achieves comparable results to native multi-language closed-source models.

#### 2 Related Work

###### 2.1 Text-to-image Models.

Over the past two years or so, text-to-image generation, especially diffusion-based generation has made incredible progress. From the open source Stable Diffusion 1.4 [36] (2022.8), Stable Diffusion

- 1.5 [36] (2022.10), Stable Diffusion XL [31] (2023.7), Playground 2.5 [20] (2024.2), Stable Diffusion

3.0 [8] (2024.2), FLUX-dev [17] (2024.8), Stable Diffusion 3.5 [8] (2024.10), HiDream-L1 [10] (2025.4), and the closed source Playground 3.0 [23] (2024.9), Recraft V3 [34] (2024.10), Ideogram

- 3.0 [14] (2025.3), Reve Image [35] (2025.3), GPT-4o [29] (2025.3), Seedream 3.0 [1] (2025.4), Midjourney V7 [27] (2025.4), although not the latest released model is the best, there is no doubt that the iteration of image generation models is accelerating. In terms of visual model structure, from the early UNet to the current mainstream MMDiT [8], the emergence of GPT-4o has brought hope to the paradigm of autoregression combined with diffusion models. In order to enhance text understanding, text encoders have gradually migrated from CLIP [32] to T5 [33], and there is a trend of further replacement with LLM or MLLM. These upgrades allow the model to generate images with better prompt following, higher aesthetic quality, higher resolution, and even have native multi-language understanding and rendering capabilities, as well as the ability to reference multiple concepts.

- 2.2 Controllable Text-to-Image Generation.

Controllable image generation usually refers to personalized generation or customized generation, that is, adding control signals for specified purposes and allowing text-to-image generation models to generate according to coarse-grained or fine-grained constraints, including spatial control, color, style, face, subject, etc. For example, ControlNet [59] and T2I-Adapter [28] are the most representative works for precise spatial control, which are conditioned on canny, depth maps, posture maps, etc. For subject-driven generation, classic works include training-based LoRA [12], Textual Inversion [9], Dreambooth [37], and recent works like IC-LoRA [13], OminiControl [39], ACE++ [26], EasyControl [60], UNO [51], and tuning-free IP-Adapter [58], InstantID [49], InstantStyle [47, 48, 54], InstantCharacter [40], which focus on learning representations of specific style, identity, and subject from one or several images. In addition to plugin-controlled models, more and more one-for-all models, such as Show-O [53], OmniGen [52], Janus [50], and VisualCloze [22], have been proposed recently for unified image generation. There are also some works [2, 5] supporting regional prompts.

###### 2.3 Visual Text Rendering.

It is a consensus that text encoders play a vital role in the text understanding and rendering process. The current mainstream text rendering methods can be divided into two categories according to whether the text encoder can understand the text content to be rendered. Most of the current open source methods are based on Stable Diffusion 1.5 [36] or Stable Diffusion XL [31], which means that although the CLIP tokenizer [32] can encode English, it cannot handle multilingual input, and the accuracy of the rendered text is very poor. Therefore, in order to support accurate multilingual rendering, several works such as GlyphControl [57], AnyText [45], AnyText2 [44], GlyphDraw2 [25], ControlText [15], and JoyType [18] use a text ControlNet [59] to control the glyphs. Other works like TextDiffuser-2 [4] and Glyph-ByT5-v2 [24] modify the design of the text encoder, such as introducing special tokens to refer to the text to be rendered, or using a tokenizer-free encoder such as ByT5 [55], but the generation quality of these works is often poor. In contrast, directly using a more powerful text encoder or multi-language encoder can bring more obvious improvements. For example, Stable Diffusion 3.5 [8] and FLUX [17] already have good English text rendering capabilities. We found that in this case, only a simple design is needed to achieve better controllable text rendering. Recent closed-source models, such as Kolors 2.0 [42], Seedream 3.0 [1], and GPT4o [29], can render text more accurately and flexibly because the models have native multi-language understanding capabilities. However, due to the lack of open source and good multilingual base models, and the high cost of replacing text encoders and retraining, in this paper, our goal is to achieve text rendering based on the latest monolingual base models. Instead of modifying the text encoder side by making the model understand the text content to be rendered, we take another way to achieve text rendering by making the model reasonably replicate the text.

#### 3 Methods

###### 3.1 Motivations

We start from a simple philosophy, that is, whether understanding text is a necessary and sufficient condition for rendering text, especially text with simple strokes. We provide several toy examples. First, think back to how human children learn to write. Most children begin writing by scribbling and drawing, not really knowing what they are writing, but simply imitating what is already around them, then they begin to recognize words, and literacy and writing skills go hand in hand. A follow-up example is copybook, that contains examples of handwriting and blank space for learners to imitate. For some complex artistic fonts, especially non-Latin scripts such as Chinese calligraphy, imitating the glyphs may occur even earlier than recognizing the text. In short, although recognizing and understanding text is undoubtedly helpful for writing, we argue that writing can also start with imitation, or replicating, which should also be valid in rendering visual text in generative models.

Based on this naive assumption, we use the pre-trained ControlNet-Union [38] that trained with canny edges on natural images to render text as preliminaries. As shown in Appendix Fig 6, it can already show some degree of replication, although there are obvious problems with the accuracy of the text, and the resulting degradation of the image quality. This motivate us to develop a method on the top of it that can replicate multilingual, multi-font text using existing monolingual text encoders.

###### 3.2 RepText

Framework. As shown in Fig 2, the RepText is a ControlNet-like framework and mostly inspired by GlyphControl [57] and JoyTypes [18]. To incorporate fine-grained glyph information and enable multilingual rendering, instead of directly using rendered glyph images as GlyphControl [57] where they depend on text encoder to understand the semantic meaning of the words, we use stronger text hint, canny edge extracting from images, additionally, to provide location information, we also use auxiliary position image to assist text rendering. The canny and position images are processed by the VAE encoder separately and concatenated over channel dimension before feeding into ControlNet branch. The text contents to be rendered are not manually added into prompts.

Training

Text

“a gundam robot holds a sign”

Encoder ❄

[Figure 19]

[Figure 20]

[Figure 21]

VAE

FLUX ❄

VAE

❄

❄

[Figure 22]

[Figure 23]

[Figure 24]

| | |
|---|---|
| | |

Text ControlNet

OCR Model

❄

[Figure 25]

Diffusion Loss Reward Loss

- Figure 2: The training pipeline of RepText, where we use both fine-grained canny edge and position mask as conditions to train text ControlNet, and further adopt a text perceptual loss.

To improve the accuracy of text generation, we further adopt a text perceptual loss as AnyText [45]. Specifically, in the training phase, given the predicted noise ϵt, current timestep t and noisy latent

image zt, we can directly predict z0 as described [11]. Then, we use the VAE decoder to obtain the approximated x

′

0 in pixel space. As we already have ground-truth annotations of text lines, we can accurately localize text regions from x

′

0 and x0, and use the cropped text images as inputs for the OCR model. Following AnyText [45], we also employ the PP-OCRv3 model [19]. The text perceptual loss is expressed as

1 hw h,w ||mp − m

′

p||22 (1)

Lreward =

p

′

where mp, m

p ∈ Rh×w×c are feature maps before the last fully connected layer of OCR model that represent the textual information in x0 and x

′

0 at position p. This Mean Squared Error (MSE) loss is used to improve the recognizability of generated text. The overall objective for training is formulated as

L = Ldenoise + λ × Lreward (2)

where λ is the scaling factor that adjusts the weights of reward OCR loss and denoising loss, and is empirically set to a small value such as 0.10 or 0.05.

Inference Strategy. In the inference phase, we introduce several key techniques as shown in Fig 3 to stabilize and improve text rendering performance.

##### Inference

Text

“a gundam robot holds a sign”

Encoder ❄

Glyph Latent Replication

[Figure 26]

[Figure 27]

[Figure 28]

VAE

FLUX

VAE

❄ noise*

❄

❄

[Figure 29]

noise

[Figure 30]

Text ControlNet

Regional Mask

[Figure 31]

Canny+Position Condition

- Figure 3: The inference framework of RepText with highlighted strategies: (1) Replicating from noise-free glyph latent, which improves text accuracy and enables color control. (2) Adopt regional mask for text regions, which avoids interference with non-text areas and ensures overall quality.

Replicating from glyph latent. Inspired by copybook, we initialize from noise-free glyph latent instead of random Gaussian noise, in other words replicate, to provide glyph guidance information at the beginning of the denoising steps. Only the text regions of noise-free glyph latent will be replicated and pasted back into random noise. We find that such a simple step plays an important role in improving the accuracy of rendered text. Benefiting from this design, RepText further allows users to specify text color without implicitly encoding color information through learnable layers. In our implementation, we find that directly copying and pasting would lead to a significant degradation of image quality because the noise-free area is not a Gaussian noise. Therefore, we introduce a weight coefficient to control the influence of glyph latent. The initialized latent zT is defined as follows.

λ1 × N(0,1) + λ2 × z0∗ if (x,y) ∈ p N(0,1) if (x,y) ∈/ p

(3)

zT =

where the z0∗ is the noise-free glyph latent encoded by VAE, λ1 and λ2 are strength coefficients for random noise and glyph latent. Only text regions p will be replicated. Please note that z0∗ can also be obtained via inversion technique, but it shows small difference.

Regional mask for text regions. Traditional ControlNet usually uses global hints as conditions, that is, both canny and depth are calculated on the entire image, while in our case, the conditional image is sparse and only the text area is valid. Therefore, in order to avoid interference with non-text areas during the denoising process, we additionally use regional masks to truncate the output of ControlNet, where the regional masks are binary and the text regions denoted by the bounding box are set to 1.

Chinese English Japanese Korean Vietnamese Russian

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

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Figure 4: RepText can render multilingual texts by replicating glyph condition.

#### 4 Experiments

###### 4.1 Implementation Details

We implemented our method on top of a widely adopted open-sourced text-to-image model FLUXdev [17], the text ControlNet branch consists of 6 double blocks and 0 single block following ControlNet-Union-Pro-2.0 [38] and is initialized from FLUX-dev. We use Anytext-3M [45] as pre-training dataset (all images are in 512x512). The training resolution is set to 512, the AdamW optimizer is used with a learning rate of 2e-5 and a batch size of 256. The OCR loss scale is set to 0.05, and we set the text drop ratio to 0.3. Besides, we collect a high-quality image dataset containing 10K images for fine-tuning, all of which are natural images such as road signs, store signs, etc., rather than synthetic. When fine-tuning, we enable buckets to train on different aspect ratios, decrease learning rate to 5e-6, and increase OCR loss scale to 0.10 and text drop ratio to 0.4. In the training phase, we render position images based on annotations, the canny images are extracted from masked images where the non text regions are set to 0. In the inference phase, users are allowed to render

their text on a blank image and use this glyph image to generate canny and position conditions. We set λ1 and λ2 to 0.9 and 0.1 empirically.

[Figure 56]

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

- Figure 5: Illustrating of RepText’s compatibility to community LoRAs. From top to bottom, they are FilmPortrait, wool yarn art and sketched style respectively.

###### 4.2 Qualitative Results

Qualitative evaluations are performed for different scenes, including multiple languages especially non-Latin, multiple fonts, multiple colors, and multiple lines. The multilingual results are shown in Fig 4. The other results can be found in Appendix Fig 7, Fig 8, and Fig 9 respectively to save pages. Benifit from our glyph replication, RepText can render accurate and controllable text content. More generated samples can be found in Appedix Fig 10 and Fig 11

###### 4.3 Comparison to Previous Methods

Baseline methods. For a comprehensive comparison, we compare with existing methods with monolingual or multilingual text rendering ability, including available open source and closed source models. For open source models, we use their official codes for inference unless otherwise specified. For closed-source models, we use their products or APIs for inference.

Monolingual. We compare with the open source Stable Diffusion 3.5 large [8], FLUX-dev [17] and HiDream-I1-Dev [10], and the close source FLUX 1.1 Pro Ultra [16], Ideogram 3.0 [14], Reve Image (Halfmoon) [35] and Recraft V3 [34]. Besides, we also compare with available open source methods that focus on controllable visual text rendering, including TextDiffuser [3], TextDiffuser2 [4], and GlyphControl [57]. Specially, we re-implement GlyphControl on FLUX-dev. For Recraft V3 [34], we use its online Frame function which is based on TextDiffuser2 [4] for controllable rendering. The result is in Appendix Fig 12. For rendering Latin text, since the base model itself has excellent understanding and rendering capabilities, RepText acts more like a position guide and allows users to specify fonts. Explicitly adding the text content (English) to be rendered in the prompt can also help but is not used in our experiments.

Multilingual. We compare with the open source Kolors 1.0 [42] and Cogview4 [43], and the close source Kolors 1.5 [42], Gemini Flash 2.0 [7], Wan2.1 Pro [46], GPT-4o [29], Seedream 3.0 [1] and Kolors 2.0 [42]. Please note that although Hunyuan-DiT [21] uses mT5 [56] as text encoder, it doesn’t support multilingual text rendering. The result is in Appendix Fig 13. Compared with open source methods, we have significant advantages in text accuracy and image quality. Compared with closed-source models using multilingual text encoders, we have better controllability. However, it must be admitted that due to their native multilingual understanding capabilities, the current most advanced text-to-image models like GPT-4o, Seedream 3.0, and Kolors 2.0 are more flexible in rendering text content than we are.

###### 4.4 Compatibility to Existing Works

To demonstrate the compatibility and effectiveness of our approach, we equip RepText with common adopted plugin models including style LoRAs, other ControlNets and IP-Adapter.

LoRAs. We use three open source LoRAs available on HuggingFace. Specifically, we select FilmPortrait1 which provides film texture, FLUX.1-dev-LoRA-MiaoKa-Yarn-World2 which creates wool yarn art, and FLUX.1-dev-LoRA-Children-Simple-Sketch3 for sketched style. As shown in Fig

- 5, our work is fully compatible to community LoRAs for stylization.

Other ControlNets. We use ControlNet-Union-Pro-2.0 [38] and ControlNet-Inpainting [6] to achieve spatial control and text editing. The results are shown in Appendix Fig 14.

IP-Adapter. We use FLUX.1-dev-IP-Adapter [41] as example. As shown in Appendix Fig 15, our method can be used together with IP-Adapter.

###### 4.5 Ablation Studies

Choice of ControlNet condition. We conducted experiments to analyze the impact of different ControlNet conditions. In the case of position only condition, it only provides position guidance, and in the case of Canny only condition, we can render the corresponding text, but the accuracy and harmony are limited, while with the joint Canny and position conditions, we can accurately render harmonious text. The result is in Appendix Fig 16.

Effect of Glyph Latent Replication. As shown in Appedix Fig 17 (left), initializing from glyph latent improves typographic accuracy at no cost. Additionally, it allows the user to specify colors without relying on an additional color encoder as shown in Appedix Fig 17 (right).

Effect of Regional Mask. Different from other ControlNets where the control signals are usually global and dense, text is a local and sparse control. We find that introducing region masks in inference phase helps improve the quality of non-text background as shown in Appendix Fig 18.

###### 4.6 Limitations and Future Works

Typical Failure Cases. Although RepText shows good text rendering capabilities and compatibility, it still has some limitations that stem from its own lack of understanding of text. We discuss several common bad cases as below.

Disharmony with the scene. Although the training dataset contains a lot of text data from natural scenes, such as road signs, the text encoder (T5-XXL) itself does not understand the text content that needs to be rendered (even the text content is added into prompt), referring to non-Latin text, so sometimes the text is stiffly pasted on the generated image as a signature or watermark, resulting in disharmony with the scene as shown in Appendix Fig 19 (a).

Limited text accuracy. For texts with complex strokes such as Tibetan or small fonts, even with our framework, the rendering accuracy is still poor as shown in Appendix Fig 19 (b). One of the reasons is that the control conditions is not precise enough, and the current compression ratio of VAE also leads to poor rendering of small characters.

Render extra texts. We find that even with the regional mask, some extra text still appeare in the non-rendered text area as shown in Appendix Fig 19 (c), and these text are usually meaningless, unrecognizable or repeated.

Limited text diversity. Limited by the text encoder, we have to utilize extra conditions and cannot flexibly control text properties through prompt, including its position, color, material, etc.

Not support precise color control. While initializing from the glyph potential allows for coarse color control, it cannot strictly render fine-grained colors, limiting its application in real-world scenarios.

- 1https://huggingface.co/Shakker-Labs/FilmPortrait
- 2https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-MiaoKa-Yarn-World
- 3https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch

Lack of distortion and perspective. Since the text content is completely controlled by front-view glyphs, limited by front-end rendering, it is inconvenient and difficult to flexibly generate text with deformation and perspective, and it is also difficult to generate some stylized text with distortion.

Future Works. As we have stated in previous sections, we acknowledge that the most flexible and effective way to render text is to let the model understand the specific meaning of each word, that is, to use a multilingual text encoder or MLLM, so as to achieve text rendering in natural scenes or poster scenes. The main concern is, besides replacing the text encoder and retraining it from scratch, is there a low-cost way, using fewer training parameters and training data, to enable existing text-to-image models to recognize and render text in different languages without compromising the original generation capabilities? For example, MetaQuery [30] reveals that MLLM’s understanding and reasoning capabilities can be used to augment image generation when both the MLLM backbone and Diffusion backbone are kept frozen but only a lightweight connector is trained, such an approach may also be applicable to visual text rendering.

#### 5 Conclusion

In this work, motivated by calligraphy copybooks, we present a simple yet effective framework RepText for controllable multilingual visual text rendering. We enable pre-trained monolingual text-to-image models with comparable capacity to generate legible visual text in different languages, fonts, and colors. Specifically, instead of using additional image or text encoders to understand words, we teach the model to replicate glyph via employ a text ControlNet with canny and position images as conditions. Additionally, we innovatively introduce glyph latent replication to enhance text accuracy and support color control. Finally, a region masking scheme is adopted to ensure good generation quality without interference from text information. Our approach outperforms existing open-source methods and achieves comparable results to native multi-language closed-source models. Next, we will explore how to efficiently enable monolingual models to understand multiple languages, thereby further improving the flexibility and accuracy of text rendering.

#### References

- [1] ByteDance, J.: Seedream 3.0, https://jimeng.jianying.com/ai-tool/home/
- [2] Chen, A., Xu, J., Zheng, W., Dai, G., Wang, Y., Zhang, R., Wang, H., Zhang, S.: Training-free regional prompting for diffusion transformers. arXiv preprint arXiv:2411.02395 (2024)
- [3] Chen, J., Huang, Y., Lv, T., Cui, L., Chen, Q., Wei, F.: Textdiffuser: Diffusion models as text painters. Advances in Neural Information Processing Systems 36, 9353–9387 (2023)
- [4] Chen, J., Huang, Y., Lv, T., Cui, L., Chen, Q., Wei, F.: Textdiffuser-2: Unleashing the power of language models for text rendering. In: European Conference on Computer Vision. pp. 386–402. Springer (2024)
- [5] Chen, Z., Li, Y., Wang, H., Chen, Z., Jiang, Z., Li, J., Wang, Q., Yang, J., Tai, Y.: Region-aware text-to-image generation via hard binding and soft refinement. arXiv preprint arXiv:2411.06558

(2024)

- [6] Creative, A.: Flux.1-dev controlnet-inpainting-beta. https://huggingface.co/ alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta (2024)
- [7] DeepMind, G.: Gemini flash 2.0, https://aistudio.google.com/
- [8] Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first international conference on machine learning (2024)
- [9] Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-Or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022)
- [10] HiDream-ai: https://github.com/hidream-ai/hidream-i1 (2025)
- [11] Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

- [12] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. ICLR 1(2), 3 (2022)
- [13] Huang, L., Wang, W., Wu, Z.F., Shi, Y., Dou, H., Liang, C., Feng, Y., Liu, Y., Zhou, J.: In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775 (2024)
- [14] Ideogram: Ideogram 3.0, https://ideogram.ai/t/explore
- [15] Jiang, B., Yuan, Y., Bai, X., Hao, Z., Yin, A., Hu, Y., Liao, W., Ungar, L., Taylor, C.J.: Controltext: Unlocking controllable fonts in multilingual text rendering without font annotations. arXiv preprint arXiv:2502.10999 (2025)
- [16] Labs, B.F.: Flux 1.1 pro ultra, https://fal.ai/models/fal-ai/flux-pro/v1.1-ultra
- [17] Labs, B.F.: Flux. https://github.com/black-forest-labs/flux (2024)
- [18] Li, C., Jiang, C., Liu, X., Zhao, J., Wang, G.: Joytype: A robust design for multilingual visual text creation. arXiv preprint arXiv:2409.17524 (2024)
- [19] Li, C., Liu, W., Guo, R., Yin, X., Jiang, K., Du, Y., Du, Y., Zhu, L., Lai, B., Hu, X., et al.: Pp-ocrv3: More attempts for the improvement of ultra lightweight ocr system. arXiv preprint arXiv:2206.03001 (2022)
- [20] Li, D., Kamko, A., Akhgari, E., Sabet, A., Xu, L., Doshi, S.: Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245

(2024)

- [21] Li, Z., Zhang, J., Lin, Q., Xiong, J., Long, Y., Deng, X., Zhang, Y., Liu, X., Huang, M., Xiao, Z., Chen, D., He, J., Li, J., Li, W., Zhang, C., Quan, R., Lu, J., Huang, J., Yuan, X., Zheng, X., Li, Y., Zhang, J., Zhang, C., Chen, M., Liu, J., Fang, Z., Wang, W., Xue, J., Tao, Y., Zhu, J., Liu, K., Lin, S., Sun, Y., Li, Y., Wang, D., Chen, M., Hu, Z., Xiao, X., Chen, Y., Liu, Y., Liu, W., Wang, D., Yang, Y., Jiang, J., Lu, Q.: Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding (2024)
- [22] Li, Z.Y., Du, R., Yan, J., Zhuo, L., Li, Z., Gao, P., Ma, Z., Cheng, M.M.: Visualcloze: A universal image generation framework via visual in-context learning. arXiv preprint arXiv:2504.07960

(2025)

- [23] Liu, B., Akhgari, E., Visheratin, A., Kamko, A., Xu, L., Shrirao, S., Lambert, C., Souza, J., Doshi, S., Li, D.: Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695 (2024)
- [24] Liu, Z., Liang, W., Zhao, Y., Chen, B., Liang, L., Wang, L., Li, J., Yuan, Y.: Glyph-byt5v2: A strong aesthetic baseline for accurate multilingual visual text rendering. arXiv preprint arXiv:2406.10208 (2024)
- [25] Ma, J., Deng, Y., Chen, C., Du, N., Lu, H., Yang, Z.: Glyphdraw2: Automatic generation of complex glyph posters with diffusion models and large language models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 5955–5963 (2025)
- [26] Mao, C., Zhang, J., Pan, Y., Jiang, Z., Han, Z., Liu, Y., Zhou, J.: Ace++: Instruction-based image creation and editing via context-aware content filling. arXiv preprint arXiv:2501.02487

(2025)

- [27] Midjourney: V7. https://www.midjourney.com/updates/v7-alpha (2025)
- [28] Mou, C., Wang, X., Xie, L., Wu, Y., Zhang, J., Qi, Z., Shan, Y.: T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In: Proceedings of the AAAI conference on artificial intelligence. vol. 38, pp. 4296–4304 (2024)
- [29] OpenAI: Gpt-4o, https://chatgpt.com/
- [30] Pan, X., Shukla, S.N., Singh, A., Zhao, Z., Mishra, S.K., Wang, J., Xu, Z., Chen, J., Li, K., JuefeiXu, F., et al.: Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256

(2025)

- [31] Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)
- [32] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

- [33] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research 21(140), 1–67 (2020)
- [34] Recraft: Recraft v3 raw, https://www.recraft.ai/projects
- [35] Reve: Halfmoon, https://preview.reve.art/app/explore
- [36] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- [37] Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22500–22510 (2023)
- [38] Shakker-Labs: Controlnet-union. https://huggingface.co/Shakker-Labs/FLUX. 1-dev-ControlNet-Union-Pro-2.0 (2025)
- [39] Tan, Z., Liu, S., Yang, X., Xue, Q., Wang, X.: Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098 (2024)
- [40] Tao, J., Zhang, Y., Wang, Q., Cheng, Y., Wang, H., Bai, X., Zhou, Z., Li, R., Wang, L., Wang, C., et al.: Instantcharacter: Personalize any characters with a scalable diffusion transformer framework. arXiv preprint arXiv:2504.12395 (2025)
- [41] Team, I.: Flux.1-dev ip-adapter. https://huggingface.co/InstantX/FLUX. 1-dev-IP-Adapter (2024)
- [42] Team, K.: Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint (2024)
- [43] THUDM: Cogview4, https://github.com/THUDM/CogView4
- [44] Tuo, Y., Geng, Y., Bo, L.: Anytext2: Visual text generation and editing with customizable attributes. arXiv preprint arXiv:2411.15245 (2024)
- [45] Tuo, Y., Xiang, W., He, J.Y., Geng, Y., Xie, X.: Anytext: Multilingual visual text generation and editing. arXiv preprint arXiv:2311.03054 (2023)
- [46] Wan: https://tongyi.aliyun.com/wanxiang/creation (2025)
- [47] Wang, H., Spinelli, M., Wang, Q., Bai, X., Qin, Z., Chen, A.: Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733 (2024)
- [48] Wang, H., Xing, P., Huang, R., Ai, H., Wang, Q., Bai, X.: Instantstyle-plus: Style transfer with content-preserving in text-to-image generation. arXiv preprint arXiv:2407.00788 (2024)
- [49] Wang, Q., Bai, X., Wang, H., Qin, Z., Chen, A., Li, H., Tang, X., Hu, Y.: Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519 (2024)
- [50] Wu, C., Chen, X., Wu, Z., Ma, Y., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C., et al.: Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848 (2024)
- [51] Wu, S., Huang, M., Wu, W., Cheng, Y., Ding, F., He, Q.: Less-to-more generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160 (2025)
- [52] Xiao, S., Wang, Y., Zhou, J., Yuan, H., Xing, X., Yan, R., Li, C., Wang, S., Huang, T., Liu, Z.: Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340 (2024)
- [53] Xie, J., Mao, W., Bai, Z., Zhang, D.J., Wang, W., Lin, K.Q., Gu, Y., Chen, Z., Yang, Z., Shou, M.Z.: Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528 (2024)
- [54] Xing, P., Wang, H., Sun, Y., Wang, Q., Bai, X., Ai, H., Huang, R., Li, Z.: Csgo: Content-style composition in text-to-image generation. arXiv preprint arXiv:2408.16766 (2024)
- [55] Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M., Roberts, A., Raffel, C.: Byt5: Towards a token-free future with pre-trained byte-to-byte models. Transactions of the Association for Computational Linguistics 10, 291–306 (2022)
- [56] Xue, L., Constant, N., Roberts, A., Kale, M., Al-Rfou, R., Siddhant, A., Barua, A., Raffel, C.: mt5: A massively multilingual pre-trained text-to-text transformer. arXiv preprint arXiv:2010.11934 (2020)

- [57] Yang, Y., Gui, D., Yuan, Y., Liang, W., Ding, H., Hu, H., Chen, K.: Glyphcontrol: glyph conditional control for visual text generation. Advances in Neural Information Processing Systems 36, 44050–44066 (2023)
- [58] Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023)
- [59] Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 3836–3847

(2023)

- [60] Zhang, Y., Yuan, Y., Song, Y., Wang, H., Liu, J.: Easycontrol: Adding efficient and flexible control for diffusion transformer. arXiv preprint arXiv:2503.07027 (2025)

#### A Supplementary Details

###### A.1 Preliminary Results

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Figure 6: Although ControlNet (Canny) is not trained specifically on visual text datasets, it can still reasonably render text by replicating the given glyph edges to some extent.

###### A.2 Qualitative Results

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

[Figure 105]

[Figure 106]

[Figure 107]

- Figure 7: RepText can render texts with use-specified fonts by replicating glyph condition.

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

- Figure 8: RepText can render texts with use-specified colors by initializing from glyph latent.

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

[Figure 131]

[Figure 132]

[Figure 133]

Figure 9: RepText can render multi-lines texts.

###### A.3 More Qualitative Results

Fig 10 shows our ability to generate TV and product posters, and Fig 11 shows more supplementary samples in natural scenes generated by RepText.

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

[Figure 168]

[Figure 169]

Figure 10: RepText can also synthesize movie or product posters using multilingual rendering capabilities and the generative model’s native Latin rendering capabilities.

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

[Figure 183]

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

Figure 11: Supplementary samples generated by RepText.

###### A.4 Comparison with Other Works

For a comprehensive comparison, we compare with existing methods with monolingual (Fig 12) or multilingual (Fig 13) text rendering ability.

SD3.5 FLUX-dev Hidream-dev F 1.1 Pro Ultra Ideogram 3.0 Reve

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

TextDiffuser TextDiffuser2 GlyphControl Recraft V3 RepText

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

###### Figure 12: Comparison with open-sourced and close-sourced on monolingual rendering.

Kolors 1.5 Gemini Flash 2.0 Wan2.1 Pro GPT-4o Kolors 2.0 Seedream 3.0

[Figure 274]

[Figure 275]

[Figure 276]

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

[Figure 297]

Kolors 1.0 CogView4 AnyText AnyText2 JoyType RepText

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

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

###### Figure 13: Comparison with open-sourced and close-sourced on multilingual rendering.

###### A.5 Compatibility to Existing Works

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

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

Figure 14: RepText with ControlNet-Union and ControlNet Inpainting.

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

a cat holds a sign

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

a pokemon holds a sign

Figure 15: RepText with IP-Adapter.

###### A.6 Ablation Studies

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

position-only

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

canny-only

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

canny+position

- Figure 16: Ablation on the choice of control conditions.

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

[Figure 377]

[Figure 378]

wo w

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

- Figure 17: Ablation on the effect of glyph latent replication.

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

###### Figure 18: Ablation on the effect of regional masks. With regional mask (bottom), the generated images show better quality than direct injection (Top). Zoom in for better visualization.

- A.7 Bad Cases We show several typical failure cases of RepText in Fig 19.

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

### (a) (b) (c)

- Figure 19: Typical Failure cases of RepText. (a) Disharmony with the scene, the text is rendered as signature or watermark. (b) Limited text accuracy especially for complex word or small font. (c) Extra texts appear as artifacts around text regions. The defective regions are highlighted by red box.

