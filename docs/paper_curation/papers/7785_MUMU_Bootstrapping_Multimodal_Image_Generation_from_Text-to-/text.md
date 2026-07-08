# arXiv:2406.18790v2[cs.CV]11Sep2024

## MUMU: BOOTSTRAPPING MULTIMODAL IMAGE GENERATION FROM TEXT-TO-IMAGE DATA

TECHNICAL REPORT

William Berman Researcher in Residence Sutter Hill Ventures WLBberman@gmail.com

#### Alex Peysakhovich∗

Researcher in Residence Sutter Hill Ventures alex.peys@gmail.com

September 13, 2024

### ABSTRACT

We train a model to generate images from multimodal prompts of interleaved text and images such as “a <picture of a man> man and his <picture of a dog> dog in an <picture of a cartoon> animated style.” We bootstrap a multimodal dataset by extracting semantically meaningful image crops corresponding to words in the image captions of synthetically generated and publicly available text-image data. Our model, MUMU, is composed of a vision-language model encoder with a diffusion decoder and is trained on a single 8xH100 GPU node. Despite being only trained on crops from the same image, MUMU learns to compose inputs from different images into a coherent output. For example, an input of a realistic person and a cartoon will output the same person in the cartoon style, and an input of a standing subject and a scooter will output the subject riding the scooter. As a result, our model generalizes to tasks such as style transfer and character consistency. Our results show the promise of using multimodal models as general purpose controllers for image generation.

[Figure 1]

Figure 1: An example of a multimodal prompt, and a resulting generation from our MUMU-Idefics2-SDXL model. The model inputs multimodal conditioning and outputs images.

∗Corresponding author

All face images in this work are used with permission. All non-generated images images in this work are owned by their respective copyright holders. All images are used only for research purposes.

### 1 Introduction

Text-to-image generative AI produces detailed images from simple text prompts [1, 2, 3, 4, 5, 6, 7]. However, text is not always sufficient to describe user intent. One potential improvement is multimodal prompting which allows users to specify both text and reference images. We bootstrap a multimodal prompt dataset from existing text-image data, and we train our image generation model with multimodal understanding, MUMU, by replacing the text encoder, CLIP [8], of a diffusion model [9, 10, 11, 12, 13, 14], SDXL [6], with a vision-language model [15, 16, 17, 18, 19, 20, 21], Idefics2 [19].

We construct a multimodal training set bootstrapped from text-image data. We use open vocabulary object detection to extract image crops corresponding to words in the image captions [22]. The image crops are then placed before their corresponding words in the text prompt, see Figure 3 for an example. Our data is mostly synthetically generated from SDXL with some added high quality publicly available data.

SDXL conditions on text via cross-attention [23] on CLIP hidden states. We replace the CLIP hidden states with those of a minorly modified Idefics2. Idefics2 is composed of a vision transformer [24] for embedding image inputs, a perceiver transformer for pooling image embeddings to a fixed sequence length, and a large vision-language model transformer. For MUMU, we remove Idefics2’s perceiver transformer to use a larger number of tokens per image. We find that removing the perceiver and using more tokens improves image quality with image quality saturating at approximately 1,000 tokens per image. We also add a small non-causal “adapter” transformer on top of Idefics2’s hidden states [25]. Figure 2 shows the full architecture. MUMU is further trained from base SDXL and Idefics2 on a single 8xH100 GPU node for approximately 300,000 steps or 6 days.

MUMU can directly place conditioning images into the generated image. Additionally, despite being only trained on crops from the same input image, MUMU can also harmonize conditioning images from different inputs into a coherent output. E.g. an input of a realistic person and a cartoon will output the same person in the cartoon style, see Figure 1. As a result, the model generalizes to tasks such as style transfer and character consistency. Additionally, the model can be merged into style-based fine-tunes of SDXL to generate conditioning objects in the fine-tune’s style.

MUMU struggles with consistency of small details (e.g. fine details of faces or clothing) and has slightly worse pure text adherence than base SDXL. Additionally, canonical Stable Diffusion artifacts carry over to MUMU e.g. ‘bleeding’ of conditioning across multiple objects. Some of these issues can likely be solved by scaling or by more careful dataset construction while other issues may require specialized design decisions.

In this report, we discuss MUMU’s dataset construction, strengths, and weaknesses. We also discuss how MUMU-like architectures fit into the general problem of controllable image generation.

### 2 Related Work and Background

#### 2.1 Diffusion Models

Diffusion models generate images by iteratively applying a denoising procedure. There are many ways to construct diffusion models, and choices such as how noise is added, how noise is removed, prediction targets, and model architecture are all well studied [9, 10, 11, 12, 13, 14, 26, 1, 6, 4, 27].

For this work, we use SDXL for which we give a quick summary and point the interested reader to the original paper for more details [6]. During training, image caption pairs, (pi,ci) are respectively encoded with a variational auto-encoder [1, 28, 29, 30], xi0 = V (pi), and a text encoder. A noise schedule is parameterized by noise levels, t, with variances, βt, such that a noised xit is sampled as xit ∼ N(xi0√1 − βt,βtI) i.e. xit = x0√1 − βt + ϵi√βt, ϵi ∼ N(0,I). The SDXL UNet, m, predicts ϵˆi = m(xit,ci,t) and is trained with MSE over batches i.e. L = n1 i(ϵˆi − ϵi)2.

For inference on a new caption cj, an initial latent, xjT ∼ N(0,I), is incrementally denoised over a subset of all timesteps to predict an unnoised latent, xj0, which is decoded to pixel space by the VAE. For this paper, we use 50 step DDIM [11] and classifier free guidance [31, 32] for all predictions.

SDXL conditions on the text, c, via cross attention [23] on CLIP [8] embeddings. There is evidence in favor of replacing the CLIP checkpoints used by SD1.5 and SDXL with larger encoders [4, 25, 7, 33]. Our main contribution is replacing CLIP with a VLM and allowing c to be multimodal.

#### 2.2 Vision Language Models (VLMs)

Language models operate on hidden states from tokenized text while vision transformers operate on hidden states from directly projected [24] or tokenized [34] image patches. VLMs operate on the interleaved hidden states of both text and images [15, 16, 17, 18, 20, 19]. Much VLM research has focused on adding image inputs to the same next-token-prediction decoder. Results show that even relatively small language models augmented with image inputs and fine tuned on a relatively small dataset can become capable visual reasoners in tasks like visual question answering [18].

MUMU’s encoder uses a standard vision-language architecture, see Figure 2. Our main results complement the VLM literature by showing that not only can existing encoders be easily bolted onto a pre-trained language module but existing decoders can be as well.

#### 2.3 Natively Multimodal Models

Open VLMs usually hybridize two pre-trained backbones. Existing work has demonstrated the image generation capabilities of natively trained multimodal models [35, 36, 37]. However, these models are closed weights and cannot be directly built upon. Our work shows that a multimodal image generation model can be constructed from existing open weights models allowing for community research on this topic.

#### 2.4 Diffusion Model Image Conditioning

We are not the first to point out that text is an insufficient vehicle for user intent in image generation. ControlNet [38], T2I-Adapter [39], and IP-Adapter [40] are non-mutually exclusive [41] methods of adding auxiliary image conditioning to pre-trained diffusion models. For all three methods, the output of a conditioning specific encoder is injected through either residual connections (ControlNet, T2I-Adapter) or newly added cross attention modules (IP-Adapter). Depth map, canny edge, and pose are examples of common ControlNet and T2I-Adapter conditionings, and reference image, style, character, and clothing are common IP-Adapter conditionings [42, 43, 44].

Training separate encoders per auxiliary conditioning has been useful for experimenting with and cheaply using new conditionings on top of a common base model. However, auxiliary encoders trained separately on top of a frozen base model can require inference time parameter tuning [45], and many added encoders may compose poorly.

Prior work [46] has considered combining relatively small multimodal encoders with SD1.5 using a similar training idea to the one presented here (object detection and interleaving captions) along with an alignment and instruction training phase. We add to this work by showing that just the interleaved object pre-training phase along with scaling the encoder and decoder is enough to achieve high quality multimodal control. In addition, we argue that training the entire model end-to-end rather than freezing decoders yields better outcomes.

#### 2.5 Style Transfer

Image style transfer has been studied across machine learning [47, 48, 49, 44, 50]. Inversion [48, 49], embedding injection, and block specific embedding injection [44, 50, 51] have all been used for style transfer in diffusion models. In comparison, multimodal encoders can be directly prompted with style reference images and do not require specialized architectures.

### 3 MUMU Architecture

We start from SDXL’s pre-trained convolutional UNet with transformer blocks cross attending to CLIP hidden states. We then ablate SDXL’s auxiliary CLIP text encoder and replace SDXL’s primary CLIP text encoder with the hidden states of the VLM Idefics2 [19].

Idefics2 is composed of a vision transformer [24] initialized from SigLIP [52] for embedding image inputs, a perceiver transformer for pooling image embeddings to a fixed sequence length [53], and a large vision-language model transformer initialized from Mistral 7b [54]. We ablate the perceiver transformer because we find its small number of image tokens hurt detail preservation. The Idefics2 hidden states are passed to a 16 layer non-causal transformer and then into SDXL cross-attention. Figure 2 shows the architecture.

[Figure 2]

Figure 2: MUMU-Idefics2-SDXL architecture. Red indicates modules which are trained, blue indicates frozen, black indicates embedding. Output is actual output from MUMU-Idefics2-SDXL to the given prompt.

### 4 Constructing MUMU Captioned Datasets

OBJECT EXTRACTION We bootstrapped our multimodal prompt dataset from text-image data by using GroundingDINO’s open vocab object detection [22] to extract image crops corresponding to words in image captions. We filtered out crops with total area less than 150×150 pixels or larger than 75% of the image. At train time, we inserted at most 3 crops per prompt before their corresponding words. We biased our train time data to include images with people, and we also replaced 70% of person crops with a crop of the person’s head because we hypothesized face quality would require more higher resolution face training data than clothing or pose quality.

SYNTHETIC DATA We used approximately 3 million synthetic images generated with SDXL and filtered on a minimum PickScore [55]. Our prompts were composed of a content, e.g. "a man and a dog hiking in the wilderness during the day," concatenated with a style, e.g. "Abstract, geometric, modernist, Cubist influences, bold color blocks, Art Deco elements." To encourage the model to disentangle content and style, we paired each content with many different styles. We used a LLM to extract both contents and styles from DiffusionDB [56], and we manually prompted a LLM to produce additional contents and styles.

[Figure 3]

REALISTIC DATA As SDXL does not produce perfect, high quality, realistic images, we augmented our synthetic data with approximately 2 million high quality, realistic, publicly available images largely containing people. We filtered these images to be SFW, high resolution, nonwatermarked, and to contain 0 or 1 people. The images were best effort center cropped to people and captioned with Llava 1.6.

Figure 3: A stylized example (not from our dataset) of the multimodal caption for a text-image pair. The object detection bounding boxes are cropped and inserted into the multimodal prompt before their corresponding words.

### 5 Training Details

We trained MUMU in two stages on a single 8xH100 GPU node with PyTorch [57] FSDP [58, 59].

All images were padded to square resolutions with black pixels. Image crops were always resized up or down to meet their target resolution i.e. we never used less than the target tokens per image. For detected person crops, we replaced the crop of the full person with a crop of their head 70% of the time. Image augmentation of random crops, random flips, Gaussian noise, and grayscaling was used on 20% of the conditioning images.

- STAGE 1 We inserted up to four images per prompt with each individual image using 324 tokens. In each prompt, up to three objects detected in the input image were inserted. 30% of the time we additionally inserted an image of canny edge, depth, or sketch [60] of the full input image. We hypothesized this would lead to better ability to combine different image content at test time.
- STAGE 2 For each prompt, we inserted a single high resolution face or person crop corresponding to 1,296 tokens to see if more tokens per image improved face quality.

HYPERPARAMETERS We trained Idefics2 and SDXL with LoRA [61] of rank 8 as the models diverged when fully trained. We believe with larger batch sizes, we could fully train all models, and it would result in better model quality.

Table 1: Hyperparameters. The same hyperparameters were used for all training stages. Model Training Learning Rate

Idefics2 ViT Not trained Idefics2 ViT to VLM MLP Rank 8 LoRA 3e − 6 Idefics2 VLM Rank 8 LoRA 3e − 6 Transformer Connector Fully trained 1e − 5 SDXL UNet Rank 8 LoRA 5e − 6

### 6 Evaluation

[Figure 4]

- Figure 4: Multimodal prompts with direct inputs of conditioning into the diffusion model (MUMU) allows for much better detail preservation than ChatGPT+DALLE3 which uses images and text to construct a highly detailed text prompt for a text-to-image generator.

We now evaluate strengths and weaknesses of MUMU. As a very basic first test, we compare our trained MUMU model to ChatGPT4 + DALLE-3 [3]. ChatGPT4 is able to input images, but, as far as we know, only passes text prompts to DALLE-3. In Figure 4 we compare a generation from MUMU given the prompt “a monster wearing armor playing guitar” and a generation from DALLE-3 when ChatGPT is given the conditioning images and prompted with “Please generate an image of a monster wearing armor and playing guitar. Attached are images of the monster, armor, and guitar respectively.” We see that MUMU does a much better job of keeping details of the conditioning images. While MUMU’s detail preservation is better than ChatGPT4+DALLE-3, MUMU is also not perfect. One prominent example is that the iconic shape of the guitar in the conditioning image is transformed into a rounded shape in the generated image.

#### Finding 1: Long input contexts are important

Current VLMs achieve impressive performance with relatively small numbers of tokens per image, e.g. GPT4V uses 85 + 170 per 512 × 512 tile tokens per image. The baseline version of Idefics2 uses a perceiver to downsample SigLIP

[Figure 5]

- Figure 5: MUMU preserves more detail at higher tokens per image. At lower tokens per image, MUMU captures the gist of ‘black robe’. At higher tokens per image, details such as the gold inlaid belt are better preserved.

tokens to a fixed 64 tokens per image. In early experiments including Idefics2’s perceiver, we found that MUMU was good at preserving high level concepts but struggled with details. Detail preservation increases with removing the perceiver and increasing tokens per image up to approximately 1,000 tokens per image, see Figure 5.

Using more tokens per image increases training costs, but inference is more reasonable as the encoder is only run once compared to the diffusion model that is iteratively applied at each step.

#### Finding 2: MUMU harmonizes diverse conditioning images

MUMU is only trained on crops from the same input image. However, as seen in Figure 10, the model learns to harmonize conditioning images from different inputs into a coherent generation. E.g. an input of a realistic person and a cartoon will output the same person in the cartoon style. Additionally, MUMU harmonizes object affordances. An input of a standing subject2 and a scooter will output the subject riding the scooter.

#### Finding 3: MUMU can perform some style transfer

We also investigated pure style transfer rather than simple image harmonization. In style transfer, the reference object is meant to serve as a global style reference rather than be placed in the generation. This task is quite far outside of the model’s training distribution.

We found that MUMU can somewhat perform style transfer though it is not perfect. Figure 7 shows more successful examples of style transfer while Figure 8 shows less successful ones.

In general, we found that a major failure point of MUMU was an inability to translate human faces into abstract styles. We hypothesize this is because of oversampling human head/face conditioning at train time.

#### Finding 4: Community SDXL fine-tunes can be model merged with MUMU

SD1.5 and SDXL have vibrant communities that create base model fine-tunes. We explored whether MUMU’s LoRAs can be used with style specific fine-tunes without re-training. To do this, we merged the MUMU LoRA with community SDXL fine-tunes3 to generate the prompted images in the fine-tune style. We see in Figure 9 that the MUMU LoRA appears to compose with the stylistic fine-tunes though more work is needed to see whether any quality degradation occurs compared to the baseline.

#### Finding 5: More work is needed for detail consistency

Prior examples show that even at large numbers of image tokens, MUMU-Idefics2-SDXL does not achieve perfect consistency in preserving details.

We now specifically consider the problem of character consistency. Figure 10 shows several portraits generated by MUMU from conditioning images of the authors compared to InstantID [41].4 InstantID is an adapter and ControlNet

2Blue monster is used with permission from https://www.spacecatceramics.com/.

- 3https://huggingface.co/cagliostrolab/animagine-xl-3.1 https://huggingface.co/stablediffusionapi/

samaritan-3d-cartoon

- 4The InstantID examples are done via the official Huggingface Demo https://huggingface.co/spaces/InstantX/

InstantID with all default parameters, no ‘style’ setting, an a prompt of ‘a portrait of a person.’ More hyper parameter tuning could increase aesthetic quality of the InstantID output.

[Figure 6]

- Figure 6: We use a variety of different subjects (dog, race car driver, animated cowboy, blue ceramic monster) in the same prompt “<image> <man/dog/monster> riding a <image> scooter in <image> New York”. MUMU is able to harmonize conditioning image styles and object affordances. We also see that MUMU suffers from common Stable Diffusion artifacts such as concept bleed (e.g. the red on the scooter for the race car driver subject). Blue monster is used with permission from https://www.spacecatceramics.com/

based SDXL fine-tune specialized for face generation. InstantID uses face embeddings, face key points, and text all as inputs. We see that InstantID does a better job at preserving facial features than MUMU, but a worse job at preserving other details (e.g. hair). We believe that further improving the vision model in a MUMU-like architecture an is a key area for improvement of facial consistency.

#### Cherry Picking

We present MUMU with examples rather than quantitative evaluations claiming ‘SoTA’ on specific tasks. We want to be transparent about MUMU’s failure cases and how much cherry picking was involved in choosing examples. MUMU has generally coherent outputs with typical Stable Diffusion artifacts such as extra objects, extra limbs, missing limbs, and concept bleeding. We chose all figures from 1 − 5 generations with a guidance scale of 6 and no negative prompt. We did no finetuning on specific or out of distribution subjects or prompts.

#### Summary

MUMU-Idefics2-SDXL is not "production ready," but its ability to generalize outside the training task shows the promise of multimodal models as general controllers for image generation. We believe many failure points presented here can be remedied simply with scale: increasing the data set, using more detailed captions, training for longer with bigger batch sizes, full training instead of using LoRA, and unfreezing the vision tokenizer. We leave scaling to the GPU rich.

[Figure 7]

Figure 7: MUMU harmonization means the model is able to do some amount of style transfer.

[Figure 8]

Figure 8: Style transfer for human subjects combined with very abstract styles is noticeably worse in our model. This is likely due to our special emphasis on faces in training.

[Figure 9]

Figure 9: MUMU is compatible with existing SDXL community style fine-tunes.

[Figure 10]

- Figure 10: MUMU, using ∼ 1000 tokens per image with the prompt “a <image> portrait”, compared to InstantID. MUMU misses important details and has worse ability to capture the face detail. However, MUMU can better capture non-face detail like hair and glasses.

[Figure 11]

- Figure 11: We sample the prompt from Figure 1, “a <picture of a man> man and his <picture of a dog> dog in an <picture of a cartoon> animated style,” 6 times. MUMU has generally coherent outputs with typical Stable Diffusion artifacts.

### 7 Some Open Questions Beyond Scaling

There are many directions beyond ‘make it bigger’ that we did not get a chance to explore. We outline some of them here in hopes of stimulating ideas for further research.

- Question 1: Architecture choices

LORA TRAINING We hypothesize the largest increase in model quality would come from fully training all models without LoRAs. We found that at our batch sizes and learning rates fully training the models let to divergences, so we leave this scaling to further work.

IMAGE TOKENIZATION It is not clear that SigLIP’s text-image discrimination pre-training objective and Idefics2’s VQA finetuning objective are optimal for preserving fine image details for generation. We believe alternative image tokenization methods or directly training the image encoder could improve model quality.

TOKEN BASED DECODER We use a diffusion decoder because SDXL is the highest quality open source image model. However, MUMU is decoder agnostic and could be retrained with an autoregressive or other token-based image generation backbone [62, 63, 64, 65, 2, 66, 35, 67, 37]. Unfortunately, most of these backbones are closed source, and open source token based models [68, 28] generate lower quality images than SDXL.

- Question 2: Data

We view the dataset image composition and annotations as perhaps the most important direction for MUMU. Our dataset is more than half synthetic data with relatively short prompts, and our small VLM generates relatively short and coarse captions for our non-synthetic data. Longer quality captions improve text-to-image models [3, 7, 33], and so we believe that longer multimodal captions with more text, more objects, and other information that is readily extractable and tokenizable (for example, spatial location of each object) could also improve MUMU.

In addition, we include object crops directly in the multimodal prompt. At test time, we typically use segmented objects with the background removed because leaving in the background causes the model to use background details in the generation. Segmenting objects rather than simply cropping them at training time (e.g. by using [69]) is an interesting extension.

- Question 3: Evaluation As there are no widely agreed upon metrics for multimodal prompted image models, we used qualitative evaluations in this paper. We constructed a multimodal prompt CLIP score measuring multimodal prompt-image alignment, but it was not a useful guide for model quality.

Our multimodal prompt-image CLIP score computed average CLIP similarity between objects in the input prompt and corresponding objects in the output image. However, it did not help us make good experimental decisions. For example, removing data augmentation created models with strong multimodal CLIP scores, but they had a tendency to copy-paste inputs directly into the generation and were not as good at harmonizing diverse inputs, see Figure 12 for an example output.

We also measured face consistency with the face embedding from [41], but models with high scores on this metric similarly were not as good at harmonizing input images.

Most likely good harmonization comes at some amount of detail loss. Understanding how to measure and balance these two objectives is an important avenue for future work.

### 8 Conclusion

We have demonstrated a method for bootstrapping multimodal prompts from a text, image dataset, and we have trained a multimodal prompted image generation model with off-the-shelf encoders and decoders.

We view multimodal inputs as an important step for any application of generative AI, and we have discussed many interesting directions for future research. Multimodal inputs unlock the possibility for users to not simply guess at what text will generate. For example, a user could hand draw a single image of a character, and then generate the character in various poses and environments. Ultimately we hope that our work contributes to the important task of making a ‘creative copilot’.

### References

[1] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and

[Figure 12]

- Figure 12: Models which did well on multimodal prompt-image CLIP score and face embedding similarity did not harmonize diverse inputs as well. This example uses the same prompt, “a <picture of a man> man and his <picture of a dog> dog in an <picture of a cartoon> animated style,” from Figure 1.

pattern recognition, pages 10684–10695, 2022.

- [2] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.
- [3] J. Betker, G. Goh, L. Jing, T. Brooks, J. Wang, L. Li, and A. Ramesh. Improving image generation with better captions. 2023.
- [4] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.
- [5] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023.
- [6] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [7] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.
- [8] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [9] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics, 2015.
- [10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [11] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [12] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution, 2020.
- [13] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations, 2021.
- [14] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022.

- [15] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.
- [16] Jun Chen, Han Guo, Kai Yi, Boyang Li, and Mohamed Elhoseiny. Visualgpt: Data-efficient adaptation of pretrained language models for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18030–18040, 2022.
- [17] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.
- [18] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.
- [19] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? arXiv preprint arXiv:2405.02246, 2024.
- [20] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Advances in Neural Information Processing Systems, 36, 2024.
- [21] Florian Bordes, Richard Yuanzhe Pang, Anurag Ajay, Alexander C Li, Adrien Bardes, Suzanne Petryk, Oscar Mañas, Zhiqiu Lin, Anas Mahmoud, Bargav Jayaraman, et al. An introduction to vision-language modeling. arXiv preprint arXiv:2405.17247, 2024.
- [22] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.
- [23] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, volume 30, 2017.
- [24] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
- [25] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.
- [26] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022.
- [27] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis, 2023.
- [28] Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image synthesis, 2021.
- [29] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.
- [30] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019.
- [31] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [32] Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis, 2021.
- [33] J. Chen, C. Ge, E. Xie, Y. Wu, L. Yao, X. Ren, Z. Wang, P. Luo, H. Lu, and Z. Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024.
- [34] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning, 2018.
- [35] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [36] Hexiang Hu, Kelvin CK Chan, Yu-Chuan Su, Wenhu Chen, Yandong Li, Kihyuk Sohn, Yang Zhao, Xue Ben, Boqing Gong, William Cohen, et al. Instruct-imagen: Image generation with multi-modal instruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4754–4763, 2024.
- [37] OpenAI. Hello gpt-4o, 2024. Accessed: 2024-06-19.

- [38] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.
- [39] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models, 2023.
- [40] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.
- [41] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024.
- [42] Mehmet Saygin Seyfioglu, Karim Bouyarmane, Suren Kumar, Amir Tavanaei, and Ismail B Tutar. Diffuse to choose: Enriching image conditioned inpainting in latent diffusion models for virtual try-all. arXiv preprint arXiv:2401.13795, 2024.
- [43] Jiehui Huang, Xiao Dong, Wenhui Song, Hanhui Li, Jun Zhou, Yuhao Cheng, Shutao Liao, Long Chen, Yiqiang Yan, Shengcai Liao, et al. Consistentid: Portrait generation with multimodal fine-grained identity preserving. arXiv preprint arXiv:2404.16771, 2024.
- [44] Haofan Wang, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards stylepreserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024.
- [45] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.
- [46] Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. Kosmos-g: Generating images in context with multimodal large language models. arXiv preprint arXiv:2310.02992, 2023.
- [47] Y. Jing, Y. Yang, Z. Feng, J. Ye, Y. Yu, and M. Song. Neural style transfer: A review. IEEE Transactions on Visualization and Computer Graphics, 26(11):3365–3385, 2019.
- [48] Y. Deng, F. Tang, W. Dong, W. Sun, F. Huang, and C. Xu. Arbitrary style transfer via multi-adaptation network. In Proceedings of the 28th ACM International Conference on Multimedia, pages 2719–2727, October 2020.
- [49] J. An, S. Huang, Y. Song, D. Dou, W. Liu, and J. Luo. Artflow: Unbiased image style transfer via reversible neural flows. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 862–871, 2021.
- [50] Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. arXiv preprint arXiv:2306.00983, 2023.
- [51] L. Rout, Y. Chen, N. Ruiz, A. Kumar, C. Caramanis, S. Shakkottai, and W. S. Chu. Rb-modulation: Training-free personalization of diffusion models using stochastic optimal control. arXiv preprint arXiv:2405.17401, 2024.
- [52] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pretraining. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023.
- [53] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651–4664. PMLR, 2021.
- [54] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
- [55] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:36652–36663, 2023.
- [56] Zijie J Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. Diffusiondb: A large-scale prompt gallery dataset for text-to-image generative models. arXiv preprint arXiv:2210.14896, 2022.
- [57] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.
- [58] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models, 2020.

- [59] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.
- [60] Zhuo Su, Wenzhe Liu, Zitong Yu, Dewen Hu, Qing Liao, Qi Tian, Matti Pietikäinen, and Li Liu. Pixel difference networks for efficient edge detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5117–5127, 2021.
- [61] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.
- [62] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021.
- [63] L. Yu, J. Lezama, N. B. Gundavarapu, L. Versari, K. Sohn, D. Minnen, Y. Cheng, A. Gupta, X. Gu, A. G. Hauptmann, and B. Gong. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.
- [64] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer, 2022.
- [65] Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, Candace Ross, Adam Polyak, Russell Howes, Vasu Sharma, Puxin Xu, Hovhannes Tamoyan, Oron Ashual, Uriel Singer, Shang-Wen Li, Susan Zhang, Richard James, Gargi Ghosh, Yaniv Taigman, Maryam Fazel-Zarandi, Asli Celikyilmaz, Luke Zettlemoyer, and Armen Aghajanyan. Scaling autoregressive multi-modal models: Pretraining and instruction tuning, 2023.
- [66] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [67] Emanuele Aiello, Lili Yu, Yixin Nie, Armen Aghajanyan, and Barlas Oguz. Jointly training large autoregressive multimodal models. arXiv preprint arXiv:2309.15564, 2023.
- [68] Suraj Patil, William Berman, Robin Rombach, and Patrick von Platen. amused: An open muse reproduction, 2024.
- [69] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023.

