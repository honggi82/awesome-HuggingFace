# arXiv:2411.14521v2[cs.CV]13Aug2025

## MyTimeMachine: Personalized Facial Age Transformation

LUCHAO QI, University of North Carolina at Chapel Hill, USA JIAYE WU, University of Maryland, USA BANG GONG and ANNIE N. WANG, University of North Carolina at Chapel Hill, USA DAVID W. JACOBS, University of Maryland, USA RONI SENGUPTA, University of North Carolina at Chapel Hill, USA

SAM [Alaluf et al. 2021]

CUSP [Gomez-Trenado et al. 2022]

AgeTransGAN [Hsu et al. 2022]

FADING [Chen and Lathuilière 2023]

+ DreamBooth [Ruiz et al. 2023] MyTimeMachine

[Figure 1]

Fig. 1. We introduce MyTimeMachine (MyTM) to perform personalized age regression (top) and progression (bottom) by training a person-specific aging model from a few (10∼50) personal photos spanning over a 20-40 year range. Our method outperforms existing age transformation techniques to generate re-aged faces that closely resemble the characteristic facial appearance of the user at the target age.

Facial aging is a complex process, highly dependent on multiple factors like gender, ethnicity, lifestyle, etc., making it extremely challenging to learn a global aging prior to predict aging for any individual accurately. Existing techniques often produce realistic and plausible aging results, but the re-aged images often do not resemble the person’s appearance at the target age and thus need personalization. In many practical applications of virtual aging, e.g. VFX in movies and TV shows, access to a personal photo collection of the user depicting aging in a small time interval (20∼40 years) is often available. However, naive attempts to personalize global aging techniques on personal photo collections often fail. Thus, we propose MyTimeMachine (MyTM), a method that combines a global aging prior with a personalized photo collection (ranging from as few as 10 images, ideally 50) to learn individualized age transformations. We introduce a novel Adapter Network that combines personalized aging features with global

Authors’ Contact Information: Luchao Qi, University of North Carolina at Chapel Hill, Chapel Hill, North Carolina, USA, lqi@cs.unc.edu; Jiaye Wu, University of Maryland, College Park, Maryland, USA, jiayewu@cs.umd.edu; Bang Gong, gongbang@cs.unc.edu; Annie N. Wang, awang13@cs.unc.edu, University of North Carolina at Chapel Hill, Chapel Hill, North Carolina, USA; David W. Jacobs, University of Maryland, College Park, Maryland, USA, dwj@umd.edu; Roni Sengupta, University of North Carolina at Chapel Hill, Chapel Hill, North Carolina, USA, ronisen@cs.unc.edu.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM 1557-7368/2025/8-ART https://doi.org/10.1145/3731172

aging features and generates a re-aged image with StyleGAN2. We also introduce three loss functions to personalize the Adapter Network with personalized aging loss, extrapolation regularization, and adaptive w-norm regularization. Our approach can also be extended to videos, achieving highquality, identity-preserving, and temporally consistent aging effects that resemble actual appearances at target ages, demonstrating its superiority over state-of-the-art approaches.

CCS Concepts: • Computing methodologies → Image processing. Additional Key Words and Phrases: Age Transformations, Personalization ACM Reference Format:

Luchao Qi, Jiaye Wu, Bang Gong, Annie N. Wang, David W. Jacobs, and Roni Sengupta. 2025. MyTimeMachine: Personalized Facial Age Transformation. ACM Trans. Graph. 44, 4 (August 2025), 16 pages. https://doi.org/10.1145/ 3731172

1 Introduction

What makes face aging so challenging? Virtual age transformation algorithms aim to digitally simulate the physical aging process of an individual’s face. The goal of these methods [He et al. 2019; Hsu et al. 2022; Li et al. 2021; Makhmudkhujaev et al. 2021; Or-El et al. 2020; Xie et al. 2024; Zhang et al. 2017] is to modify the shape and texture of the face to create the desired re-aging effect, while preserving the individual’s unique identity, along with the pose, lighting, and style of the input image. However, facial aging is often highly dependent on several factors, such as ethnicity, gender, genetics, lifestyle, and health conditions [Mendelson and Wong 2012; Swift et al. 2020], which makes it challenging to model.

Existing age transformation algorithms [Alaluf et al. 2021; GomezTrenado et al. 2022; Hsu et al. 2022] learn a generative global prior, modeling how an average face ages, typically using datasets like FFHQ [Karras et al. 2021]. While these methods generate visually appealing results, they often fail to reflect how a specific individual actually ages. For example, given an image of Al Pacino at 68, state-of-the-art techniques [Alaluf et al. 2021; Chen and Lathuilière 2023; Gomez-Trenado et al. 2022; Hsu et al. 2022] produce a realistic but inaccurate version of his 30-year-old self (see Fig. 1). In applications like re-aging actors, accurate personalization is essential, as audiences are often familiar with a subject’s appearance over time. However, accurately predicting an individual’s re-aged appearance from a single image is highly ill-posed and challenging, since aging is person-specific [Despois et al. 2020].

In this paper, we show that accurate age synthesis can be achieved with as few as 10, ideally 50, photos of an individual across a 20∼40 year time range. Personal photo collections are often available in many practical applications of virtual aging, and utilizing them can significantly improve the result, see MyTimeMachine in Fig 1. For example, consider de-aging effects often used in movies where a particular actor at 60 years old is shooting a scene where they need to be rendered as 30 years old. We can easily access the past 20∼40 years of photos of the actor to learn an accurate aging model. Similarly, consider an individual interested in simulating how a photograph of their loved one at 40 years would appear at 60 years old or beyond. We can also easily access the past 10-20 years of photo collection of their loved ones to understand the aging process and more accurately simulate their future appearance at 60 years and beyond. We therefore create a personalized aging method that can transform an input image to any target age, both within and beyond the age range represented in the personal photo collection used for training.

Simply personalizing a generic global age transformation algorithm, e.g. FADING [Chen and Lathuilière 2023], with Dreambooth [Ruiz et al. 2023] is ineffective. Personal photo collections often cover a limited range of age, pose, lighting, and style variations compared to large-scale facial datasets like FFHQ. Consequently, naive fine-tuning typically results in overfitting, limiting the model’s ability to generalize to unseen ages, poses, styles, and lighting conditions, as shown by the extrapolation failure of FADING + Dreambooth in row 4 of Fig. 10. Additionally, FADING is built on diffusion, facing the typical inversion-editability trade-off problem [Hertz et al. 2023; Tov et al. 2021]. Specifically, re-aging requires both high fidelity to the input face at similar ages and high editability as the target age diverges. In contrast, such trade-offs have been more well explored in StyleGAN2’s well-trained latent space [Bhattad et al. 2023; Roich et al. 2022; Tov et al. 2021; Xia et al. 2023]. Therefore, we demonstrate an effective approach to personalized age transformation based on StyleGAN2.

Our proposed personalized age transformation network, MyTimeMachine (MyTM), introduces a novel adapter network that updates global facial aging features with personalized aging characteristics, trained on a personal photo collection using custom loss functions. Built on top of SAM [Alaluf et al. 2021], a global age transformation network capable of continuous aging without per-image optimization, MyTM enhances SAM’s global age encoder,

which projects an input image into StyleGAN2’s latent space with a specified target age. We design a personalized adapter network that learns to adjust the global aging features. To train this adapter, we introduce three loss functions: personalized aging loss, extrapolation regularization, and adaptive w-norm regularization. The personalized aging loss ensures that identity-preserving features of the reaged image closely resemble those in a reference image from the personal photo collection at a similar target age. Extrapolation regularization controls aging effects beyond the training age range using global priors, while adaptive w-norm regularization addresses StyleGAN’s inversion-editability trade-off, ensuring distinct shape and texture changes due to aging while preserving identity. We extend MyTM to video re-aging by utilizing face-swapping techniques to generate temporally consistent and identity-preserving results.

We curated a longitudinal aging dataset comprising high-quality images of 12 celebrities, captured under diverse conditions, including varying poses, expressions, and lighting. Inspired by real-world applications of personalized aging, we train our model on this dataset and establish two experimental frameworks to evaluate its performance: one for age regression, where a 70-year-old is rendered younger, and another for age progression, where a 40year-old is rendered older. Our method outperforms existing global age transformation and naive personalization techniques, delivering high-quality, identity-preserving aging effects in both images and videos that resemble individual’s appearance at the target age.

In summary, our contributions are as follows: (i) We demonstrate that with access to a few (∼50) personal images spanning a few decades (20∼40 years), we can achieve high-quality, identitypreserving facial age transformations. These transformations accurately reflect the person’s appearance at the target age while maintaining the style of the input image. (ii) We introduce several key technical advancements that integrate a global aging prior with a personal photo collection to enable personalized aging. Our approach trains an adapter network to adjust the global aging prior, utilizing three custom loss functions: personalized age loss, extrapolation regularization, and adaptive w-norm regularization. (iii) We show that MyTM can also be extended to perform temporally consistent and identity-preserving reaging in videos, which is important for many VFX applications.

2 Related Work

Traditional age transformation methods fall into two categories: prototype-based [Kemelmacher-Shlizerman et al. 2014; Tiddeman et al. 2001] and physical model-based approaches [Suo et al. 2010; Tazoe et al. 2012]. For a detailed overview, we refer readers to the survey by [Fu et al. 2010]. Recently, generative models have shown impressive results in synthesizing and editing high-resolution face images, inspiring their use in aging tasks.

2.1 Global Age Editing

Global age editing refers to age transformation without personal data. Prior works [Nitzan et al. 2022b; Shen et al. 2022] leverage StyleGAN2’s well-trained latent space, identifying and traversing a linear age editing direction within it. However, this assumption often fails with larger age changes, especially across a lifespan, and

can entangle other attributes (e.g., gender or glasses) [Härkönen et al. 2020]. To address this, recent methods [Makhmudkhujaev et al. 2021; Or-El et al. 2020; Yao et al. 2021] apply nonlinear latent age transformations by training dedicated age encoders.

In diffusion models, recent method [Wahid et al. 2024] combines diffusion with EG3D [Chan et al. 2022] for 3D age editing. Other methods [Baumann et al. 2024; Grimmer and Busch 2024; Kwon et al. 2023; Li et al. 2023] perform 2D age editing through latent manipulation guided by CLIP [Radford et al. 2021]. However, these methods continue to struggle with attribute entanglement in the latent space. FADING [Chen and Lathuilière 2023] improves disentanglement by projecting the input face into the diffusion model’s latent space using NTI [Mokady et al. 2023] and applying age editing through p2p [Hertz et al. 2023]. However, FADING focuses on textural changes to achieve re-aging, neglecting the broader facial shape changes that occur over a person’s lifespan [Gomez-Trenado et al. 2022]. This limitation arises because p2p identifies age-related pixels through attention control, resulting in localized edits rather than facial structural changes [Rout et al. 2024]. To address this, we build on StyleGAN2, leveraging its well-trained latent space for both fine-grained textural control and structural changes.

- 2.2 Personalization of Generative Models

Personalization involves tuning face models with personal images. PTI [Roich et al. 2022] fine-tunes a StyleGAN2 generator anchored by an inverted latent code. Other approaches [Nitzan et al. 2022a; Qi et al. 2023; Zeng et al. 2023] adapt the generator on a small set of personal images (50∼100) to create a personalized prior. In diffusion models, Dreambooth [Ruiz et al. 2023] optimizes the weights of the network to adapt to a specific subject through a prompt identifier [Banerjee et al. 2024].

In the context of lifespan age transformation, these personalization techniques often overfit to the few training images in a limited range (e.g., ages 50 to 70), making it challenging to extrapolate to ages beyond the training range (e.g., 20 years old). We demonstrate that MyTM produces personalized face aging results within the training age range and generalizes to ages beyond it.

- 2.3 Video Re-aging

A recent video re-aging approach, FRAN [Zoss et al. 2022], applies facial masks to predict age-related changes within masked regions per frame. However, similar to FADING, such method often overlooks structural changes in facial shape that naturally occur over a person’s lifetime, such as the widening of a previously narrow face due to bone growth and shifts in facial fat distribution with age [Gomez-Trenado et al. 2022]. It also suffers with temporal consistency since it is trained on static images. To address this, Muqeet et al. [Muqeet et al. 2023] propose a re-aging model trained on synthetic video data, generating re-aged keyframes and interpolating between them to enhance temporal consistency. However, neither of these approaches is open-sourced or supports personalized video re-aging. Recent work [iperov 2024] seeks to enhance temporal consistency in identity-specific face-swapping by personalizing models for individual users. However, this approach demands around 5,000 images of the person’s face captured under various conditions, all

at a similar age, limiting its effectiveness for lifespan aging transformations.

To address this, we follow face-swapping techniques [Chen et al. 2020; Xu et al. 2022c], using our personalized re-aged face as the source for swapping. This approach eliminates the need for training a dedicated model on a large number of personal images.

3 Method

In Section 3.1, we outline the preliminaries of global aging—SAM [Alaluf et al. 2021]. Section 3.2 introduces our personalized aging adapter, MyTM, and how it combines personal and global aging. Finally, in Section 3.3, we describe the training losses for MyTM, explaining personalized re-aging for both interpolation and extrapolation (target ages within or beyond the training range).

- 3.1 Preliminaries

Here we provide a brief overview of SAM [Alaluf et al. 2021], a global age transformation network that forms the building block of our proposed personalized network, MyTM. SAM trains an age encoder (E𝜃) that maps an input image (𝑥) into the latent space W+ of StyleGAN, aligning with the desired target age (𝑎tgt). The latent code is then processed through the pre-trained StyleGAN (D𝜃) to generate the age-transformed face (𝑦tgt). SAM is trained on the FFHQ dataset [Karras et al. 2020], where the training procedure in-

volves producing an age-transformed output 𝑦tgt = D𝜃 (E𝜃 (𝑥,𝑎tgt)) in a forward pass. This process is supervised by the loss Lforward, encouraging the re-aged image to be similar to the input image:

Lforward(𝑦tgt) =𝜆𝑙2L2(𝑦tgt) + 𝜆𝑙𝑝𝑖𝑝𝑠LLPIPS(𝑦tgt)

+ 𝜆𝑖𝑑LID(𝑦tgt) + 𝜆𝑎𝑔𝑒Lage(𝑦tgt) (1)

L2, L𝑙𝑝𝑖𝑝𝑠 and LID matches age-transformed image (𝑦tgt) to input image (𝑥) in pixel space, LPIPS feature space [Zhang et al. 2018] and

arcface [Deng et al. 2022] feature space respectively. Lage matches the predicted age by a face age detector, 𝐷𝐸𝑋 (·) [Rothe et al. 2015], with target age 𝑎tgt:

Lage 𝑦tgt = ||𝑎tgt − 𝐷𝐸𝑋 𝑦tgt ||2 (2) After the forward pass, SAM encourages the transformed image (𝑦tgt) to be re-transformed back to the input image. This process helps ensure cycle consistency [Zhu et al. 2017] and can be formally described as 𝑦cycle = D𝜃 (E𝜃 (𝑦tgt,𝑎𝑥)) with the same loss: Lcycle(𝑦cycle) = Lforward(𝑦cycle). The complete training loss is then given by:

Lsam = Lforward(𝑦tgt) + Lcycle(𝑦cycle) (3)

- 3.2 MyTM: Designing a Personalized Age Adapter

Training a personalized aging prior from scratch is suboptimal due to the limited availability of personal aging data. To address this, we introduce MyTM, which personalizes a pre-trained age encoder by combining two components: 1) SAM, a pre-trained age encoder that captures a shared global aging prior learned from a diverse set of identities, and 2) Age Adapter, a personalized age adapter network trained exclusively on an individual. Specifically, we assume the individual has a personal photo collection of 𝑁 RGB images, 𝑥𝑖,

[Figure 2]

Eq. 6 Eq. 7

Eq. 5

- Fig. 2. Given an input face of Oprah Winfrey at 70 years old, our adapter re-ages her face to resemble her appearance at 30, while preserving the style of the input image. To achieve personalized re-aging, we collect ∼50 images of an individual across different ages and train an adapter network that updates the latent code generated by the global age encoder SAM. Our adapter preserves identity during interpolation when the target age falls within the range of ages seen in the training data, while also extrapolating well to unseen ages.

each with an associated ground truth age, 𝑎𝑖, represented as D = {(𝑥𝑖,𝑎𝑖)}𝑖𝑁=1.

Our key idea is to update the age-transformed latent code Wtgt+ produced by the global age transformation network, SAM, using a personalized adapter network, 𝐴𝑁 (·). The adapter takes the latent vector Wtgt+ predicted by SAM and computes an offset latent vector ΔWtgt+ = AN(E𝜃 (𝑥,𝑎tgt),𝑎tgt). As shown in Fig. 2, we then combine this personalized latent adaptation ΔWtgt+ with the global latent code Wtgt+ and pass the result through the pre-trained StyleGAN2 decoder. Formally,

𝑦tgt𝑝 = D𝜃 ( E𝜃 (𝑥,𝑎tgt) + AN(E𝜃 (𝑥,𝑎tgt),𝑎tgt) ) (4)

global aging personal aging

By doing so, MyTM enables the integration of partially observed, personalized aging information of an individual—using only images of that person—into the global aging trajectory. Our adapter is based on an MLP architecture, with detailed implementation of the age adapter network available in the supplementary material.

- 3.3 MyTM: Loss Funtions Our adapter is trained on a personal photo collection, which is noted

- as D = {(𝑥𝑖,𝑎𝑖)}𝑖𝑁=1. Weintroducethreelossfunctions—personalized aging loss, extrapolation regularization, and adaptive w-norm regularization—to integrate global priors with personal data. Addition-

ally, we also use the loss function Lsam from Sec. 3.1, based on SAM, to mitigate the forgetting of global priors [Kirkpatrick et al. 2017; Nitzan et al. 2023].

- 3.3.1 Personalized Aging Loss. After examining the loss formulation of SAM [Alaluf et al. 2021], we notice that the primary source

of aging information is the aging loss Lage, which relies on a pretrained age classifier. The problem with this global aging loss is that the age classifier is not robust across different ethnicities, styles, and individual aging patterns [Lin et al. 2022]. It is often impossible to train a robust aging detector that works well for every individual.

Rather than relying on the power of a global age classifier, we propose a personalized aging loss that encourages facial features of the transformed face to be similar to reference images in a similar age range in the training dataset. This encourages the re-aged image to closely resemble how the person looked at that age, ignoring the pose, lighting, and style variations.

We denote the minimum and maximum age of the training dataset

D = {(𝑥𝑖,𝑎𝑖)}𝑖𝑁=1 as 𝑎𝑚𝑖𝑛 = min(𝑎1...𝑎𝑛) and 𝑎𝑚𝑎𝑥 = max(𝑎1...𝑎𝑛). During training, we randomly sample a target age 𝑎tgt between minimum and maximum age 𝑎tgt ∼ U(𝑎𝑚𝑖𝑛,𝑎𝑚𝑎𝑥). We create a

reference set, Dtgt = (𝑥𝑗,𝑎𝑗) 𝑀𝑗=1, which contains actual images of the individual near the target age (𝑎tgt ± 3-years). We then employ a facial recognition network, arcface [Deng et al. 2022], to extract identity features and compute the similarity between the age-transformed image 𝑦tgt and all images in the reference set Dtgt, and only consider the maximum similarity. Considering maximum similarity over the reference set ensures that the identity recognition networks are not significantly influenced by stylistic differences between images. Formally, we define the personalized aging loss:

𝑀 𝑗=1

Lpers-age = 1 −𝑚𝑎𝑥 𝑅(𝑦tgt𝑝 ),𝑅(𝑥𝑗)

(5)

𝑅(·) is a pretrained arcface [Deng et al. 2022] network for facial feature recognition, ⟨·, ·⟩ computes the cosine similarity between its argument [Patashnik et al. 2021], and 𝑀 is the number of images in a reference set Dtgt with faces near the target age (𝑎tgt ± 3-years).

3.3.2 Extrapolation Regularization. When training the adapter network with personalized age loss, we observe that the network’s performance can degrade when 𝑎tgt falls outside the training age range [𝑎𝑚𝑖𝑛,𝑎𝑚𝑎𝑥]. Specifically, this degradation manifests as the generated images continuing to resemble the appearance at the boundaries of the training age range (𝑎𝑚𝑖𝑛,𝑎𝑚𝑎𝑥), rather than appropriately aging or de-aging. For instance, as illustrated in row 4 of Fig. 10 (FADING + Dreambooth), when the training set covers faces aged 30 to 70, the model may overfit, generating faces that still resemble a 30-year-old when 𝑎tgt = 10.

To prevent this extrapolation failure, we enforce the preservation of the pre-trained SAM’s output during extrapolation. We apply experience replay [Nitzan et al. 2023; Ruiz et al. 2023], which en-

courages the output of our personalized age encoder (𝑦tgt𝑝 ) to be similar to that produced by the pre-trained SAM (𝑦tgt):

Lreg-extra = 𝜆𝑙2L2(𝑦tgt𝑝 ,𝑦tgt) + 𝜆LPIPSLLPIPS(𝑦tgt𝑝 ,𝑦tgt)

+𝜆IDLID(𝑦tgt𝑝 ,𝑦tgt) (6)

- 3.3.3 Adaptive w-norm regularization. During personalization, we observedthatSAMstruggles tocapture distinct facial feature changes across ages, as illustrated in row 2 of Fig. 10. We attribute this issue to the inversion-editability trade-off [Roich et al. 2022; Tov et al. 2021]. Specifically, the latent codes predicted by SAM are distant from the training distribution and the center of the latent space, W, reducing their editing capacity and making personalization challenging. This trade-off is particularly relevant in facial aging tasks. When the target age is close to the input age, we encourage the latent codes close to SAM’s pre-trained output, preserving inversion accuracy while staying distant from the average latent code W. As the target age diverges from the input age, greater deformations in face structures are needed, requiring latent codes nearer to W to facilitate editing. To address this, we propose adaptive W-norm regularization inspired by [Richardson et al. 2021], where Lreg = 𝜆reg∥Wtgt+ − W∥ is used to constrain the latent codes. We further enhance this by making 𝜆reg a cosine function, ⟨·, ·⟩, of the difference between input and target age Δage = |𝑎𝑖 − 𝑎tgt|:

Lreg = 𝜆reg(Δage)∥Wtgt+ − W∥ 𝜆reg(Δage) = 1 − ⟨𝜋 · Δage/100⟩ (7)

4 Experiments

In Sec. 4.1 we first introduce our experimental setup, including datasets, experimental framework, state-of-the-art aging algorithms, and evaluation metrics. In Sec. 4.2 we present comparisons with state-of-the-art baselines, followed by the application of MyTM for video re-aging in Sec. 4.3. Finally, we present ablation studies in 4.4.

- 4.1 Experimental Setup

- 4.1.1 Dataset. We curated a dataset of images featuring 12 celebrities spanning a wide age range, including 7 males and 5 females from diverse ethnic backgrounds such as Caucasian, African American, Hispanic, and Asian. For further details, please refer to Table 5 in the supplementary material. For each celebrity, we train MyTM on 50 images and evaluate its performance using test images of the same individual at either 40 or 70 years old, depending on the task defined in the experimental setup.

- 4.1.2 Experimental Framework. We consider the following two realworld scenarios where age transformation techniques are heavily used and demand high quality. 1) Age regression or de-aging renders images of an individual to go back in time and is heavily used in VFX for movies and TV shows [Travis 2019]. Motivated by this, we design an experimental setup where we personalize our aging model by training on images from either a 30∼70 or 50∼70 age range

- and then evaluate de-aging performance on an unseen image at 70 years old to a target age (𝑎tgt ≤ 70 years old). We sample the target age every 10 years where 𝑎tgt ∈ {0, 10, 20, 30, 40, 50, 60, 70}. 2) Age progression or aging renders images of an individual going forward in time and is used for forensic investigations, missing person searches, or as an emotional support tool to visualize departed loved ones. We design an experimental framework where we personalize our aging model by training on images from 20∼40 years old and evaluate on unseen faces at 40 years old to generate a target age (𝑎tgt ≥ 40), where 𝑎tgt ∈ {40, 50, 60, 70, 80, 90, 100}.
- 4.1.3 State-of-the-art Aging Algorithms. We compare our results with the following state-of-the-art aging methods: (i) SAM [Alaluf

- et al. 2021], which uses a pre-trained StyleGAN2 decoder and trains an age encoder on FFHQ [Karras et al. 2021]. It treats aging as a continuous process, enabling fine-grained control over transformations. (ii) CUSP [Gomez-Trenado et al. 2022], which jointly trains both an age encoder and decoder on FFHQ-Aging [Or-El et al. 2020]. While effective for age transformations, it lacks fine-grained control due to its reliance on predefined age group-based transformations, limiting editing capabilities and supporting resolutions only up to 256 × 256. (iii) AgeTransGAN [Hsu et al. 2022], an encoder-decoder architecture, also limited by age group-based transformations, similar to CUSP. (iv) FADING [Chen and Lathuilière 2023], which inverts images into the latent space of a face diffusion model using NTI [Mokady et al. 2023], then edits them through p2p [Hertz et al. 2023]. We also introduce additional personalization baselines: (v) SAM Pers. f.t., naively fine-tune SAM on personal images; (vi) FADING + Dreambooth, personalizes FADING by following the Dream-

booth approach [Ruiz et al. 2023] with the prompt "photo of a [𝑎tgt] year old [sks] man/woman".

4.1.4 Evaluation Metrics. Following the evaluation protocols in prior aging baselines [Alaluf et al. 2021; Chen and Lathuilière 2023; Gomez-Trenado et al. 2022; Hsu et al. 2022], we evaluate our personalized age transformation results in terms of age accuracy and identity preservation, using the following metrics to evaluate the re-aged results quantitatively:

- • Age Accuracy (Age𝑀𝐴𝐸). Following [Alaluf et al. 2021; GomezTrenado et al. 2022], we define age mean absolute error as Age𝑀𝐴𝐸 = |𝑎tgtˆ − 𝑎tgt|, where 𝑎tgtˆ is predicted by FP-Age [Lin et al. 2022].
- • Identity Preservation (ID𝑠𝑖𝑚). Previous works [Teng et al. 2024; Zoss

- et al. 2022] evaluate identity preservation by comparing the re-aged face to the input face. However, facial recognition systems, such as arcface [Deng et al. 2022], demonstrate a strong dependence on age and thus favor age consistency between the re-aged face and the input face [Alaluf et al. 2021], favoring algorithms that perform small changes. We address this problem by creating reference image sets of the individual near the target age which are not used in training, and then calculate the identity similarity to the reference images at the target age, in contrast to using the input image. Formally,

ID𝑠𝑖𝑚 𝑦tgt = 𝑚𝑎𝑥 𝑅(𝑦tgt),𝑅(𝑥𝑗) 𝑀𝑗=1 (8)

where 𝑅(·) is a pretrained arcface [Deng et al. 2022] network for facial feature recognition and 𝑥𝑗 belongs in reference image set near

[Figure 3]

- Fig. 3. Performance of age transformation techniques for age regression (first two rows) and age progression (last two rows). The first column shows the input image, and the second column provides a reference image of the same person at the target age. MyTM (Ours) is compared against other state-of-the-art methods including SAM [Alaluf et al. 2021], CUSP [Gomez-Trenado et al. 2022], AgeTransGAN [Hsu et al. 2022], and FADING [Chen and Lathuilière 2023].

- Table 1. Performance of age regression where a 70-year-old face is de-aged to a target age 𝑎𝑡𝑔𝑡 ≤ 70. We also evaluate MyTM (Ours) using 20-year (𝑎tgt ∈ 50 ∼ 70) and 40-year (𝑎tgt ∈ 30 ∼ 70) age ranges in the training data. Bold indicates the best results, while underlined denotes the second-best.

Method Age𝑀𝐴𝐸(↓)

ID𝑠𝑖𝑚(↑) 𝑎tgt ≤ 70 𝑎tgt ∈ 50 ∼ 70 𝑎tgt ∈ 30 ∼ 70 SAM [Alaluf et al. 2021] 8.1 0.49 0.58 0.53

+ Pers. f.t. (50∼70) 8.2 0.48 0.58 -

+ Pers. f.t. (30∼70) 9.2 0.49 - 0.53 CUSP [Gomez-Trenado et al. 2022] 11.0 0.39 0.44 0.42 AgeTransGAN [Hsu et al. 2022] 11.1 0.53 0.65 0.58 FADING [Chen and Lathuilière 2023] 8.9 0.60 0.72 0.66 + Dreambooth [Ruiz et al. 2023] (50∼70) 25.9 0.63 0.78 -

+ Dreambooth [Ruiz et al. 2023] (30∼70) 23.0 0.64 - 0.70 Ours (50∼70) 7.7 0.65 0.76 Ours (30∼70) 7.8 0.67 - 0.72

the target age (𝑎tgt ± 3-years). We report the average ID𝑠𝑖𝑚 across all sampled target ages.

- 4.2 Comparison with Age Transformation Methods

(ID𝑠𝑖𝑚), with an 11.7% improvement (0.67 vs. 0.60) in ID𝑠𝑖𝑚 over the best-performing method, FADING. This improvement is also maintained during interpolation (e.g., when 𝑎tgt ∈ 30 ∼ 70), producing a 9.0% increase in ID𝑠𝑖𝑚 (0.72 vs. 0.66) compared to FADING, even when FADING overfits to the input image via NTI [Mokady et al. 2023], favoring its ID𝑠𝑖𝑚 score for smaller age gaps.

Compared to other personalized methods, our approach achieves both high age accuracy (Age𝑀𝐴𝐸) and strong identity preservation. SAM + Pers. f.t. shows minimal improvement over SAM alone, underscoring the effectiveness of our proposed loss function in Sec. 3.3. While FADING + Dreambooth [Ruiz et al. 2023] (50∼70) records a slight improvement over ours in ID𝑠𝑖𝑚 (0.78 vs. 0.76), it fails to maintain age accuracy (Age𝑀𝐴𝐸 25.9 vs. 7.7) and overfits to the training age range, limiting its ability to generalize to unseen ages.

4.2.2 Age progression. We perform age progression with a 20-year (ages 20 ∼ 40) range of personal photos. Age progression specifically evaluates the extrapolation ability of our technique to ages not seen in training. Quantitative results are presented in Table 2, with visual comparisons to pre-trained methods shown in Fig. 3, and a full visual comparison provided in Fig. 9 and Fig. 10.

4.2.1 Age Regression. We use two age ranges of personal photos—40 years (ages 30∼70) and 20 years (ages 50∼70)—to examine the impact of training age span. Results are presented in Table 1, with visual examples of pre-trained methods shown in Fig. 3. For a detailed visual comparison across all ages (0∼100), please refer to Fig. 9. Additionally, visual results are compared against personalized methods in Fig. 10.

Our model outperforms pre-trained baselines, achieving the highest age accuracy (6.3) and best identity preservation (0.70 for 𝑎tgt ≥ 40 and 0.78 for 𝑎tgt ∈ 40 ∼ 60), due to the benefits of personalization. As shown in Fig. 9, FADING often produces poor results when the target age differs greatly from the input age, due to NTI + p2p

Compared to pre-trained baselines (SAM, CUSP, AgeTransGAN, and FADING), our method achieves superior identity preservation

- Table 2. Performance of age progression where a 40-year-old face is aged to a target age 𝑎𝑡𝑔𝑡 ≥ 40. We evaluate MyTM (Ours) using 20-year (𝑎tgt ∈ 40 ∼ 60) and 𝑎tgt ≥ 40 age ranges in the training data. Bold indicates the best results, while underlined denotes the second-best. Note that FADING

+ Dreambooth has the lowest aging accuracy, as measured by Age𝑀𝐴𝐸.

ID𝑠𝑖𝑚(↑) 𝑎tgt ≥ 40 𝑎tgt ∈ 40 ∼ 60

Method Age𝑀𝐴𝐸(↓)

SAM [Alaluf et al. 2021] 6.9 0.54 0.58 + Pers. f.t. (20∼40) 10.3 0.56 0.59 CUSP [Gomez-Trenado et al. 2022] 7.3 0.44 0.48 AgeTransGAN [Hsu et al. 2022] 8.5 0.61 0.65 FADING [Chen and Lathuilière 2023] 7.6 0.62 0.71 + Dreambooth [Ruiz et al. 2023] (20∼40) 20.2 0.72 0.77 Ours (20∼40) 6.3 0.70 0.78

[Figure 4]

Fig. 4. User study comparing our method with baselines—FADING, AgeTransGAN, and SAM—for age regression (𝑎tgt ≤ 70) and age progression (𝑎tgt ≥ 40). We present the percentage of user preference for our method over the baselines.

editing [Rout et al. 2024]. Compared to other personalized methods, FADING + Dreambooth achieves slightly better ID𝑠𝑖𝑚 than our model (0.72 vs. 0.70). However, it struggles to extrapolate to unseen ages, resulting in a high Age𝑀𝐴𝐸 of 20.2.

4.2.3 User Studies. We conduct user studies to qualitatively evaluate our method through pairwise human evaluations. In each pair, users see the original input image alongside two re-aged results

- at the target age 𝑎tgt—one generated by an existing method and the other by ours, presented in random order. Users also receive

reference images showing the person’s face near 𝑎tgt and are asked to select the result that best matches the reference images while preserving the style of the original input image.

We evaluate our method across two age regression tasks (30∼70 and 50∼70) and one age progression task (20∼40), totaling three tasks. We then sample one input and re-aged image pair per celebrity, resulting in 10 pairs for each age regression task and 8 pairs for the age progression task. For each pairwise comparison, we collected 24 responses for FADING, 25 for AgeTransGAN, and 29 for SAM. As shown in Fig. 4, our method is significantly preferred over the baselines across all re-aging tasks.

- 4.3 MyTM for Video Re-aging

Having established a 2D personalized aging prior, we now extend our approachtovideo-basedface re-aging. A straightforward method

[Figure 5]

- Fig. 5. Naive pasting of MyTM’s re-aged face onto a video frame of Al Pacino from The Irishman. Left: The original frame, with the aligned face shown in the bottom-left corner. Right: The re-aged face generated by MyTM is naively pasted back onto the original frame. Red arrows highlight visible artifacts. Zooming in is recommended for a clearer view.

[Figure 6]

- Fig. 6. Face-swapping for video re-aging on a clip of Jackie Chan from Bleeding Steel. Left: Keyframe re-aged using MyTM. Right: The re-aged face is transferred to the remaining frames via face-swapping, preserving temporal consistency across the video.

would involve applying MyTM to re-age each frame individually and then pasting the transformed face back onto the original frame. However, this naive approach is prone to two key issues. (a) Face Detection Errors: Face detection algorithms such as dlib [King 2009] may fail to detect faces in challenging frames or may produce inaccurate landmark estimates. These inconsistencies lead to misalignment during the warping process, resulting in noticeable artifacts. (b) NonRigid Transformations: The re-aging process inherently alters the facial structure and contours. Because these changes are non-rigid, aligning and pasting the modified face onto the original frame becomes theoretically infeasible without distortion. An example of this failure is illustrated in Fig. 5, where the re-aged face is naively pasted back onto the original frame using landmark-based warping. For clarity, the original frame’s head is shown with partial transparency, highlighting the misalignment caused by naive compositing.

To this end, we build upon face-swapping techniques by utilizing Inswapper1, a widely adopted black-box model [Wang et al. 2024] for video re-aging. Given a source video, we manually select a keyframe in a near-frontal pose with minimal occlusion and motion blur as the basis for re-aging. MyTM is then applied to this keyframe to transform the face image to the desired target age, generating a personalized re-aged face. Next, for each video frame, the face in the current frame and the re-aged face are input into the swapping model to generate the final re-aged result. This re-aged face is then pasted back onto the current frame using landmark-based warping. This framework requires only a single re-aged face for swapping, ensuring strong temporal consistency while preserving personalized facial identity. Our video re-aging pipeline is illustrated in Fig. 6. For further details on temporal consistency, please refer to the supplement.

1https://github.com/deepinsight/insightface

[Figure 7]

Metric SAM N = 10 N = 50 N = 100 Age𝑀𝐴𝐸(↓) 8.1 8.5 7.8 8.0

Dataset D Size Ablations

IDsim(↑) 0.49 0.58 0.67 0.67

Fig. 7. Effect of training dataset size D on personalization. MyTM is trained on ages 30∼70 and tested for 𝑎tgt ≤ 70. Visual examples of Robert De Niro are shown at the top, with quantitative results displayed below. MyTM achieves personalized re-aging with as few as 10 images, with 50 images providing optimal performance.

- Table 3. Performance of both age regression and progression using non-

celebrity subjects, where a 40-year-old face is re-aged to a target age 𝑎tgt. Bold values indicate the best performance.

Method Age𝑀𝐴𝐸(↓)

ID𝑠𝑖𝑚(↑) 𝑎tgt ∈ 20 ∼ 40 𝑎tgt ≥ 40

SAM [Alaluf et al. 2021] 12.1 0.62 0.53 AgeTransGAN [Hsu et al. 2022] 10.3 0.67 0.61 FADING [Chen and Lathuilière 2023] 13.1 0.72 0.67 + Dreambooth [Ruiz et al. 2023] (20∼40) 20.9 0.78 0.69 Ours (20∼40) 9.3 0.83 0.74

- 4.4 Ablation Study

- 4.4.1 Effect of Dataset Size. We investigate the impact of training dataset size on MyTM by sampling subsets of images for each celebrity, with sizes of 10, 50, and 100. We then assess MyTM’s performance on the age regression task (ages 30∼70), which demands

the largest training age range. We report the average ID𝑠𝑖𝑚 in Fig. 7. Results indicate a significant performance improvement from 10 to 50 images, with minimal gains from 50 to 100 images. Consequently, we use 50 images for personalization, unless otherwise noted.

- 4.4.2 Effect of MyTM on Non-celebrities. We collect YouTube videos of five non-celebrities, where individuals documented their age transformation with an average age span of 20 to 54 years. From these videos, we create a dataset of approximately 50 training images per person and train personalized models on ages 20-40, evaluating them using held-out test images at age 40. To assess performance, we measure identity preservation for interpolation (ages 20-40) and extrapolation (>40), as well as transformed age accuracy, as shown in Table 3. Our results demonstrate that MyTM achieves overall the best identity preservation and age accuracy for both celebrity and in-the-wild non-celebrity data, both quantitatively and qualitatively.
- 4.4.3 Effect of Proposed Loss Functions and Architecture. We analyze the effectiveness of our proposed network architecture and loss functions by conducting an ablation study in Fig. 8. We begin with SAM and progressively introduce each proposed component, including custom loss terms and the adapter network. For the age regression task, we train MyTM on ages spanning 30 to 70, testing with target ages 𝑎tgt ≤ 70. Our proposed Personalized Aging Loss yields the most improvement in ID𝑠𝑖𝑚.

[Figure 8]

Adapter Network

Extrapolation Reg

Personalized Aging Loss

Adaptive W-norm Reg

Model

ID𝑠𝑖𝑚(↑) Age𝑀𝐴𝐸(↓) Ours ✓ ✓ ✓ ✓ 0.67 10.5

- A) - ✓ ✓ ✓ 0.65 11.0
- B) - - ✓ ✓ 0.65 17.9
- C) - - - ✓ 0.55 12.9 SAM Pers. f.t. - - - - 0.46 11.2 SAM - - - - 0.45 11.1

Fig. 8. Contributions of our proposed loss functions and the adapter network for the age regression task, trained on ages 30∼70 and tested for 𝑎tgt ≤ 70 on Al Pacino.

Our results in Fig. 8 show that incorporating the extrapolation loss significantly improves AgeMAE (from 17.9 to 11.0) without affecting identity similarity (IDsim remains at 0.65). This suggests that a personalized prior alone is insufficient for extrapolation, due to its training on a limited temporal range. To address this, we incorporate a global aging prior by encouraging MyTM’s output to align with that of a pre-trained SAM model. This fusion of personalized and global priors enables more accurate age transformation while preserving identity, further supporting the observed improvement in AgeMAE with stable IDsim.

4.4.4 Direct Video Editing vs. Face-Swapping. To further validate the effectiveness of our face-swapping pipeline in Sec. 4.3, we compare it against direct video-based re-aging methods [Tzaban et al. 2022; Xu et al. 2022a]. Specifically, we benchmark our method against STIT [Tzaban et al. 2022] and VideoEditGAN [Xu et al. 2022b] using a set of 14 video clips. To assess visual quality, we conducted a user study using randomly ordered pairwise comparisons. Participants evaluated each pair based on temporal consistency and identity similarity. Responses from 28 participants were collected, and user preferences for our method over the baselines are summarized in Table 4.

Our results indicate that direct video editing methods exhibit lower temporal consistency and weaker identity preservation compared to our face-swapping approach. STIT [Tzaban et al. 2022] exhibits noticeable flickering, especially in videos with fast head movements, extreme poses, or challenging lighting conditions. VideoEditGAN [Xu et al. 2022b] demonstrates better temporal consistency than STIT but still lacks temporal consistency and personalized re-aging compared to ours. Moreover, these methods require perframe PTI [Roich et al. 2022] inversion and optimization, leading to runtimes >3 hours per video on a single A6000 GPU. In contrast, our pipeline completes the same task in <5 minutes on the same hardware, while achieving better temporal coherence and more personalized re-aging.

- Table 4. User study comparing our face-swapping results with direct video re-aging baselines, STIT and VideoEditGAN. We evaluate temporal consistency and identity preservation, reporting user preference percentages (User%) for ours over each baseline.

Method Temporal Consistency (User%) ID (User%)

Ours vs. STIT [Tzaban et al. 2022] 86% 93% Ours vs. VideoEditGAN [Xu et al. 2022b] 71% 79%

- 5 Conclusion

We present MyTimeMachine, a personalized facial age transformation method leveraging individual photo collections and global aging priors, that outperforms existing approaches.

- 5.1 Limitations

While our model performs effective age transformations (see Fig. 12, supp. pdf), it can struggle with accessories such as glasses, largely due to limitations in the e4e encoder [Tov et al. 2021]. Furthermore, the pre-trained SAM model has difficulty in 1) modifying hair color, particularly transitioning to or from white. For example, hair does not turn white when the target age ≥ 80 — a common challenge in other aging works [Li et al. 2021; Tang et al. 2018]. Future work could focus on improving global aging models or developing specialized post-editing techniques for hairstyles. 2) producing red-eye artifacts when generating older faces. Although our proposed w-norm regularization mitigates these issues, they are not fully resolved.

- 5.2 Ethical Considerations

Facial aging is a complex and inherently challenging problem, and even with personalization, our model may lack robustness across all underrepresented populations. Our approach also has the potential to produce manipulated images of real individuals, which poses a significant societal risk. Although such concerns are common to generative models, we explicitly call for the fair and responsible use of our method. It is intended for research and positive applications only. Misuse that infringes on privacy, spreads misinformation, or causes harm is strongly discouraged. Future work should include safeguards such as synthetic image detection and fairness evaluation.

Acknowledgments

This research was supported in part by Lenovo Research (Morrisville, NC). We gratefully acknowledge the invaluable support and assistance of the members of the Mobile Technology Innovations Lab. This work was also supported in part by the National Science Foundation under Grant No. 2213335.

[Figure 9]

#### Fig. 9. Performance of age transformation techniques for age regression (top) and age progression (bottom). For age regression, MyTM (Ours) is trained across a 40-year range (ages 30 to 70), while for age progression, it is trained over a 20-year range (ages 20 to 40). Personalized training data age ranges are marked in red. A reference image of the same person, taken within 3 years of the target age, is included for comparison.

[Figure 10]

[Figure 11]

- Fig. 10. Performance of personalized age transformation techniques for age regression (top) and age progression (bottom). The input test images match those in Fig. 10 (top) and Fig. 10 (bottom) for consistency. For age regression, MyTM (Ours) is trained across a 40-year range (ages 30 to 70), while for age progression, it is trained over a 20-year range (ages 20 to 40). The age ranges included in the personal training data are highlighted in red. We also provide an example image of the same person within 3 years of the target age as a reference.

References

Mehran Aghabozorgi, Shichong Peng, and Ke Li. 2023. Adaptive IMLE for Few-shot Pretraining-free Generative Modelling. In Proceedings of the 40th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 202), Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (Eds.). PMLR, 248–264. https://proceedings.mlr.press/v202/ aghabozorgi23a.html

Yuval Alaluf, Or Patashnik, and Daniel Cohen-Or. 2021. Only a matter of style: age transformation using a style-based regression model. ACM Transactions on Graphics 40, 4 (Aug. 2021), 1–12. doi:10.1145/3450626.3459805

Sudipta Banerjee, Govind Mittal, Ameya Joshi, Sai Pranaswi Mullangi, Chinmay Hegde, and Nasir Memon. 2024. Identity-Aware Facial Age Editing Using Latent Diffusion. IEEE Transactions on Biometrics, Behavior, and Identity Science 6, 4 (2024), 443–457. doi:10.1109/TBIOM.2024.3390570

David Bau, Hendrik Strobelt, William Peebles, Jonas Wulff, Bolei Zhou, Jun-Yan Zhu, and Antonio Torralba. 2019. Semantic photo manipulation with a generative image prior. ACM Transactions on Graphics 38, 4 (Aug. 2019), 1–11. doi:10.1145/3306346.3323023

Stefan Andreas Baumann, Felix Krause, Michael Neumayr, Nick Stracke, Vincent Tao Hu, and Björn Ommer. 2024. Continuous, Subject-Specific Attribute Control in T2I Models by Identifying Semantic Directions. arXiv:2403.17064 [cs.CV] https: //arxiv.org/abs/2403.17064

Anand Bhattad, Viraj Shah, Derek Hoiem, and D. A. Forsyth. 2023. Make It So: Steering StyleGAN for Any Image Inversion and Editing. doi:10.48550/arXiv.2304.14403 Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, and Gordon Wetzstein. 2022. Efficient Geometry-aware 3D Generative Adversarial Networks. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, New Orleans, LA, USA, 16102–16112. doi:10.1109/ CVPR52688.2022.01565

Renwang Chen, Xuanhong Chen, Bingbing Ni, and Yanhao Ge. 2020. SimSwap: An Efficient Framework For High Fidelity Face Swapping. In Proceedings of the 28th ACM International Conference on Multimedia (Seattle, WA, USA) (MM ’20). Association for Computing Machinery, New York, NY, USA, 2003–2011. doi:10.1145/3394171. 3413630

Xiangyi Chen and Stéphane Lathuilière. 2023. Face Aging via Diffusion-based Editing. In 34th British Machine Vision Conference 2023, BMVC 2023, Aberdeen, UK, November 20-24, 2023. BMVA. https://papers.bmvc2023.org/0595.pdf

Jun Myeong Choi, Max Christman, and Roni Sengupta. 2024. Personalized Video Relighting With an At-Home Light Stage. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part XL (Milan, Italy). Springer-Verlag, Berlin, Heidelberg, 394–410. doi:10.1007/978-3031-73661-2_22

Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, and Stefanos Zafeiriou. 2022. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 10 (Oct. 2022), 5962– 5979. doi:10.1109/TPAMI.2021.3087709

Julien Despois, Frédéric Flament, and Matthieu Perrot. 2020. AgingMapGAN (AMGAN): High-Resolution Controllable Face Aging with Spatially-Aware Conditional GANs. In Computer Vision – ECCV 2020 Workshops, Adrien Bartoli and Andrea Fusiello (Eds.). Springer International Publishing, Cham, 613–628.

Amil Dravid, Yossi Gandelsman, Kuan-Chieh Wang, Rameen Abdal, Gordon Wetzstein, Alexei A. Efros, and Kfir Aberman. 2024. Interpreting the Weight Space of Customized Diffusion Models. doi:10.48550/arXiv.2406.09413

Yun Fu, Guodong Guo, and Thomas S. Huang. 2010. Age Synthesis and Estimation via Faces: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence 32, 11 (2010), 1955–1976. doi:10.1109/TPAMI.2010.36

Gege Gao, Huaibo Huang, Chaoyou Fu, Zhaoyang Li, and Ran He. 2021. Information Bottleneck Disentanglement for Identity Swapping. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 3403–3412. doi:10.1109/CVPR46437.

- 2021.00341

Guillermo Gomez-Trenado, Stéphane Lathuilière, Pablo Mesejo, and Óscar Cordón.

- 2022. Custom Structure Preservation in Face Aging. In Computer Vision – ECCV 2022, Shai Avidan, Gabriel Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner (Eds.). Springer Nature Switzerland, Cham, 565–580.

Marcel Grimmer and Christoph Busch. 2024. AgeDiff: Latent Diffusion-based Face Age Editing with Dual Cross-Attention. In 2024 IEEE International Workshop on Information Forensics and Security (WIFS).1–6. doi:10.1109/WIFS61860.2024.10810706

Erik Härkönen, Aaron Hertzmann, Jaakko Lehtinen, and Sylvain Paris. 2020. GANSpace: discovering interpretable GAN controls. In Proceedings of the 34th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’20). Curran Associates Inc., Red Hook, NY, USA, Article 825, 10 pages.

Zhenliang He, Meina Kan, Shiguang Shan, and Xilin Chen. 2019. S2GAN: Share Aging Factors Across Ages and Share Aging Trends Among Individuals. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Seoul, Korea (South), 9439–9448. doi:10.1109/ICCV.2019.00953

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohenor. 2023. Prompt-to-Prompt Image Editing with Cross-Attention Control. In The Eleventh International Conference on Learning Representations. https://openreview. net/forum?id=_CDixzkzeyb

Gee-Sern Hsu, Rui-Cang Xie, Zhi-Ting Chen, and Yu-Hong Lin. 2022. AgeTransGAN for Facial Age Transformation with Rectified Performance Metrics. In Computer Vision – ECCV 2022, Shai Avidan, Gabriel Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner (Eds.). Springer Nature Switzerland, Cham, 580–595.

iperov. 2024. iperov/DeepFaceLive. https://github.com/iperov/DeepFaceLive Tero Karras, Samuli Laine, and Timo Aila. 2021. A Style-Based Generator Architecture for Generative Adversarial Networks . IEEE Transactions on Pattern Analysis & Machine Intelligence 43, 12 (Dec. 2021), 4217–4228. doi:10.1109/TPAMI.2020.2970919

Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. 2020. Analyzing and Improving the Image Quality of StyleGAN . In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 8107–8116. doi:10.1109/CVPR42600.2020.00813 Ira Kemelmacher-Shlizerman, Supasorn Suwajanakorn, and Steven M. Seitz. 2014. Illumination-Aware Age Progression. In 2014 IEEE Conference on Computer Vision and Pattern Recognition. 3334–3341. doi:10.1109/CVPR.2014.426

Davis E. King. 2009. Dlib-ml: A Machine Learning Toolkit. Journal of Machine Learning Research 10 (2009), 1755–1758.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka GrabskaBarwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences 114, 13 (March 2017), 3521–3526. doi:10.1073/pnas. 1611835114

Mingi Kwon, Jaeseok Jeong, and Youngjung Uh. 2023. Diffusion Models Already Have A Semantic Latent Space. In The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=pd1P2eUBVfq

Jason Lee, Kyunghyun Cho, and Douwe Kiela. 2019. Countering Language Drift via Visual Grounding. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (Eds.). Association for Computational Linguistics, Hong Kong, China, 4385–4395. doi:10.18653/v1/D19-1447

Peipei Li, Rui Wang, Huaibo Huang, Ran He, and Zhaofeng He. 2023. Pluralistic Aging Diffusion Autoencoder. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Paris, France, 22556–22566. doi:10.1109/ICCV51070.2023.02067

Zeqi Li, Ruowei Jiang, and Parham Aarabi. 2021. Continuous Face Aging via Selfestimated Residual Age Embedding. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Nashville, TN, USA, 15003–15012. doi:10.1109/CVPR46437.2021.01476

Yiming Lin, Jie Shen, Yujiang Wang, and Maja Pantic. 2022. FP-Age: Leveraging Face Parsing Attention for Facial Age Estimation in the Wild. IEEE Transactions on Image Processing (2022), 1–1. doi:10.1109/TIP.2022.3155944

Mingcong Liu, Qiang Li, Zekui Qin, Guoxin Zhang, Pengfei Wan, and Wen Zheng. 2021. BlendGAN: Implicitly GAN Blending for Arbitrary Stylized Face Generation. In Advances in Neural Information Processing Systems.

Yuchen Lu, Soumye Singhal, Florian Strub, Aaron Courville, and Olivier Pietquin. 2020. Countering Language Drift with Seeded Iterated Learning. In Proceedings of the 37th International Conference on Machine Learning. PMLR, 6437–6447. https: //proceedings.mlr.press/v119/lu20c.html

Farkhod Makhmudkhujaev, Sungeun Hong, and In Kyu Park. 2021. Re-Aging GAN: Toward Personalized Face Age Transformation. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Montreal, QC, Canada, 3888–3897. doi:10.1109/ICCV48922.2021.00388

Bryan Mendelson and Chin-Ho Wong. 2012. Changes in the Facial Skeleton With Aging: Implications and Clinical Applications in Facial Rejuvenation. Aesthetic Plastic Surgery 36, 4 (2012), 753–760. doi:10.1007/s00266-012-9904-3

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2023. Nulltext Inversion for Editing Real Images using Guided Diffusion Models. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 6038–6047. doi:10.1109/CVPR52729.2023.00585

Abdul Muqeet, Kyuchul Lee, Bumsoo Kim, Yohan Hong, Hyungrae Lee, Woonggon Kim, and KwangHee Lee. 2023. Video Face Re-Aging: Toward Temporally Consistent Face Re-Aging. doi:10.48550/arXiv.2311.11642

Yotam Nitzan, Kfir Aberman, Qiurui He, Orly Liba, Michal Yarom, Yossi Gandelsman, Inbar Mosseri, Yael Pritch, and Daniel Cohen-Or. 2022a. MyStyle: A Personalized Generative Prior. ACM Trans. Graph. 41, 6, Article 206 (Nov. 2022), 10 pages. doi:10. 1145/3550454.3555436

Yotam Nitzan, Rinon Gal, Ofir Brenner, and Daniel Cohen-Or. 2022b. LARGE: LatentBased Regression through GAN Semantics. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, New Orleans, LA, USA, 19217–19227. doi:10.1109/CVPR52688.2022.01864

Yotam Nitzan, Michael Gharbi, Richard Zhang, Taesung Park, Jun-Yan Zhu, Daniel Cohen-Or, and Eli Shechtman. 2023. Domain Expansion of Image Generators . In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 15933–15942. doi:10.1109/CVPR52729. 2023.01529

Roy Or-El, Soumyadip Sengupta, Ohad Fried, Eli Shechtman, and Ira KemelmacherShlizerman. 2020. Lifespan Age Transformation Synthesis. In Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VI (Glasgow, United Kingdom). Springer-Verlag, Berlin, Heidelberg, 739–755. doi:10.1007/978-3-030-58539-6_44

Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. 2021. StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV). 2065–2074. doi:10.1109/ICCV48922. 2021.00209

Luchao Qi, Jiaye Wu, Annie N. Wang, Shengze Wang, and Roni Sengupta. 2023. My3DGen: A Scalable Personalized 3D Generative Model. doi:10.48550/arXiv.2307. 05468

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 139), Marina Meila and Tong Zhang (Eds.). PMLR, 8748–8763. https://proceedings.mlr.press/ v139/radford21a.html

Xiaohang Ren, Xingyu Chen, Pengfei Yao, Heung-Yeung Shum, and Baoyuan Wang. 2023. Reinforced Disentanglement for Face Swapping without Skip Connection. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV). 20608–20618. doi:10.1109/ICCV51070.2023.01889

- K. Ricanek and T. Tesafaye. 2006. MORPH: a longitudinal image database of normal adult age-progression. In 7th International Conference on Automatic Face and Gesture Recognition (FGR06). 341–345. doi:10.1109/FGR.2006.78

Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. 2021. Encoding in Style: a StyleGAN Encoder for Image-toImage Translation. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Nashville, TN, USA, 2287–2296. doi:10.1109/CVPR46437. 2021.00232

Daniel Roich, Ron Mokady, Amit H. Bermano, and Daniel Cohen-Or. 2022. Pivotal Tuning for Latent-based Editing of Real Images. ACM Trans. Graph. 42, 1, Article 6 (Aug. 2022), 13 pages. doi:10.1145/3544777

Rasmus Rothe, Radu Timofte, and Luc Van Gool. 2015. DEX: Deep EXpectation of Apparent Age from a Single Image. In 2015 IEEE International Conference on Computer Vision Workshop (ICCVW). IEEE, Santiago, Chile, 252–257. doi:10.1109/ICCVW.2015. 41

- L Rout, Y Chen, N Ruiz, C Caramanis, S Shakkottai, and W Chu. 2025. Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations. In The Thirteenth International Conference on Learning Representations.

Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. 2024. Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations. http://arxiv.org/abs/2410.10792

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 22500–22510. doi:10.1109/CVPR52729.2023.02155

Yujun Shen, Ceyuan Yang, Xiaoou Tang, and Bolei Zhou. 2022. InterFaceGAN: Interpreting the Disentangled Face Representation Learned by GANs. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 4 (2022), 2004–2018. doi:10.1109/TPAMI.2020.3034267

Jinli Suo, Song-Chun Zhu, Shiguang Shan, and Xilin Chen. 2010. A Compositional and Dynamic Model for Face Aging. IEEE Transactions on Pattern Analysis and Machine Intelligence 32, 3 (March 2010), 385–401. doi:10.1109/TPAMI.2009.39

Arthur Swift, Steven Liew, Susan Weinkle, Julie K Garcia, and Michael B Silberberg.

2020. The Facial Aging Process From the “Inside Out”. Aesthetic Surgery Journal 41, 10 (Dec. 2020), 1107–1119. doi:10.1093/asj/sjaa339

Xu Tang, Zongwei Wang, Weixin Luo, and Shenghua Gao. 2018. Face Aging with Identity-Preserved Conditional Generative Adversarial Networks. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, Salt Lake City, UT, USA, 7939–7947. doi:10.1109/CVPR.2018.00828

Yusuke Tazoe, Hiroaki Gohara, Akinobu Maejima, and Shigeo Morishima. 2012. Facial aging simulator considering geometry and patch-tiled texture. In ACM SIGGRAPH 2012 Posters (SIGGRAPH ’12). Association for Computing Machinery, New York, NY, USA, 1. doi:10.1145/2342896.2343002

Qianrui Teng, Rui Wang, Xing Cui, Peipei Li, and Zhaofeng He. 2024. Exploring 3D-aware Lifespan Face Aging via Disentangled Shape-Texture Representations . In 2024 IEEE International Conference on Multimedia and Expo (ICME). IEEE Computer Society, Los Alamitos, CA, USA, 1–6. doi:10.1109/ICME57554.2024.10687595

B. Tiddeman, M. Burt, and D. Perrett. 2001. Prototyping and transforming facial textures for perception research. IEEE Computer Graphics and Applications 21, 5 (July 2001), 42–50. doi:10.1109/38.946630

Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. 2021. Designing an encoder for StyleGAN image manipulation. ACM Trans. Graph. 40, 4, Article 133 (July 2021), 14 pages. doi:10.1145/3450626.3459838

Ben Travis. 2019. The Irishman: Robert De Niro Recreated Iconic GoodFellas Scene To Test De-Ageing Tech – Exclusive. https://www.empireonline.com/movies/news/ irishman-martin-scorsese-robert-de-niro-goodfellas-test/

Rotem Tzaban, Ron Mokady, Rinon Gal, Amit Bermano, and Daniel Cohen-Or. 2022. Stitch it in Time: GAN-Based Facial Editing of Real Videos. In SIGGRAPH Asia 2022 Conference Papers (Daegu, Republic of Korea) (SA ’22). Association for Computing Machinery, New York, NY, USA, Article 29, 9 pages. doi:10.1145/3550469.3555382

Junaid Wahid, Fangneng Zhan, Pramod Rao, and Christian Theobalt. 2024. DiffAge3D: Diffusion-based 3D-aware Face Aging. arXiv preprint arXiv:2408.15922 (2024).

Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. 2024. InstantID: Zero-shot Identity-Preserving Generation in Seconds. http://arxiv.org/abs/2401. 07519

Xintao Wang, Yu Li, Honglun Zhang, and Ying Shan. 2021. Towards Real-World Blind Face Restoration with Generative Facial Prior . In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 9164–9174. doi:10.1109/CVPR46437.2021.00905

Weihao Xia, Yulun Zhang, Yujiu Yang, Jing-Hao Xue, Bolei Zhou, and Ming-Hsuan Yang. 2023. GAN Inversion: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 3 (2023), 3121–3138. doi:10.1109/TPAMI.2022.3181070

Jiu-Cheng Xie, Jun Yang, Wenqing Wang, Feng Xu, and Hao Gao. 2024. Diverse and Lifespan Facial Age Transformation Synthesis with Identity Variation Rationality Metric. doi:10.48550/arXiv.2401.14036

Chao Xu, Jiangning Zhang, Yue Han, Guanzhong Tian, Xianfang Zeng, Ying Tai, Yabiao Wang, Chengjie Wang, and Yong Liu. 2022d. Designing One Unified Framework for High-Fidelity Face Reenactment and Swapping. In Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XV (Tel Aviv, Israel). Springer-Verlag, Berlin, Heidelberg, 54–71. doi:10.1007/978-3-03119784-0_4

- Yiran Xu, Badour AlBahar, and Jia-Bin Huang. 2022a. Temporally Consistent Semantic Video Editing. In Computer Vision – ECCV 2022, Shai Avidan, Gabriel Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner (Eds.). Springer Nature Switzerland, Cham, 357–374.
- Yiran Xu, Badour AlBahar, and Jia-Bin Huang. 2022b. Temporally Consistent Semantic Video Editing. https://arxiv.org/abs/2206.10590v1

Zhiliang Xu, Zhibin Hong, Changxing Ding, Zhen Zhu, Junyu Han, Jingtuo Liu, and Errui Ding. 2022c. MobileFaceSwap: A Lightweight Framework for Video Face Swapping. Proceedings of the AAAI Conference on Artificial Intelligence 36, 3 (Jun. 2022), 2973–2981. doi:10.1609/aaai.v36i3.20203

Xu Yao, Gilles Puy, Alasdair Newson, Yann Gousseau, and Pierre Hellier. 2021. High Resolution Face Age Editing. In 2020 25th International Conference on Pattern Recognition (ICPR). 8624–8631. doi:10.1109/ICPR48806.2021.9412383

Libing Zeng, Lele Chen, Yi Xu, and Nima Khademi Kalantari. 2023. MyStyle++: A Controllable Personalized Generative Prior. In SIGGRAPH Asia 2023 Conference Papers (Sydney, NSW, Australia) (SA ’23). Association for Computing Machinery, New York, NY, USA, Article 70, 11 pages. doi:10.1145/3610548.3618171

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR.

Zhifei Zhang, Yang Song, and Hairong Qi. 2017. Age Progression/Regression by Conditional Adversarial Autoencoder . In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 4352–4360. doi:10.1109/CVPR.2017.463

Yan Zheng and Lemeng Wu. 2024. InverseMeetInsert: Robust Real Image Editing via Geometric Accumulation Inversion in Guided Diffusion Models. doi:10.48550/arXiv. 2409.11734

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. 2017. Unpaired Imageto-Image Translation Using Cycle-Consistent Adversarial Networks. In 2017 IEEE International Conference on Computer Vision (ICCV). 2242–2251. doi:10.1109/ICCV. 2017.244

Gaspard Zoss, Prashanth Chandran, Eftychios Sifakis, Markus Gross, Paulo Gotardo, and Derek Bradley. 2022. Production-Ready Face Re-Aging for Visual Effects. ACM Trans. Graph. 41, 6, Article 237 (Nov. 2022), 12 pages. doi:10.1145/3550454.3555520

### MyTimeMachine: Personalized Facial Age Transformation Appendix

Along with this appendix, we provide additional visual materials (e.g., images and videos) on our project page 2. We highly recommend viewing the accompanying videos for a more comprehensive look at the visual results.

- A Overview of Appendices Our appendices contain the following additional details:

- • Sec. Bprovidesanoverviewofourdata preprocessing pipeline, the curated dataset (summarized in Table 5), and additional details about the celebrities used in experiments Sec. 4.2.
- • Sec. C provides implementation details of our personalized adapter network, including hyperparameters and training configurations.
- • Sec. Dpresentsbenchmarkingresults against other pre-trained aging methods, with qualitative results shown in Fig. 9.
- • Sec. E includes benchmarking results against alternative naive personalization techniques, with both quantitative and qualitative results displayed in Fig. 11 and Fig. 10.
- • Sec. F explains our choice of using StyleGAN2’s aging encoder for personalization over encoder-decoder GAN models or diffusion models.
- • Sec. G discusses the design rationale behind our video reaging pipeline.

- B Dataset Curation

Existing in-the-wild aging datasets [Lin et al. 2022; Ricanek and Tesafaye 2006; Zhang et al.2017] lack longitudinal data for individual subjects, as they do not offer multiple high-quality images of the same person over several decades. To address this limitation, we followed previous personalization works [Nitzan et al. 2022a; Zeng et al. 2023] and collected a new celeb dataset as summarized in

- Table 5. To better illustrate the age distributions, we count the number of images within the age ranges 20–40, 40–60, and 60–80, and report these numbers in the table. These ranges differ from the training ranges of 20–40, 50–70, and 30–70. For each celebrity, we first gathered facial images, then enhanced older images to improve visual quality, compensating for the limitations of earlier camera technology and image processing methods. Following [Wang et al. 2021], we restored grayscale or low-quality images to ensure a more consistent and enhanced visual representation over time. Faces were then cropped and aligned according to the FFHQ [Karras et al.

- 2021] standard. While downloading publicly available images, we extracted metadata, such as the time of capture, to calculate each subject’s age. For re-aging tasks in Sec. 4.2, the available age distribution of the collected celebrities varies; for instance, some celebrities have fewer than 50 images in the 20 to 40 age range. Therefore, we conduct age regression tasks for the following 10 celebrities: Al Pacino, Charles III, Elizabeth II, Robert De Niro, Oprah Winfrey, Morgan Freeman, Jackie Chan, Chow Yun-fat, Elaine Chao, and Margaret Thatcher, and age

- 2 https://mytimemachine.github.io/

Table 5. A longitudinal facial aging dataset featuring images of 12 celebrities. The number of images for each celebrity is reported across different age ranges. Unless stated otherwise, 50 images are selected for training.

Celebrity Age range 20∼40 40∼60 60∼80

Al Pacino 21∼84 89 56 198 Charles III 01∼76 219 409 530 Elizabeth II 03∼96 65 116 539 Robert De Niro 27∼81 121 340 286 Jennifer Aniston 02∼55 375 322 Oprah Winfrey 24∼70 163 529 315 Morgan Freeman 20∼87 4 136 290 Jackie Chan 21∼70 31 444 201 Chow Yun-fat 20∼68 91 109 60 Elaine Chao 16∼71 14 117 123 Diego Maradona 17∼60 165 301 Margaret Thatcher 20∼87 70 270 268

progression tasks for the following 8 celebrities: Al Pacino, Charles III, Elizabeth II, Jennifer Aniston, Oprah Winfrey, Chow Yun-fat, Diego Maradona, and Margaret Thatcher. These results also correspond to the number of pairs used in user studies, as discussed in Sec. 4.2. For dataset size ablation studies in Sec. 4.4, we use the same celebrities selected in the regression task.

C Implementation Details

Personalized Age Adapter Network. Inspired by [Bau et al. 2019; Liu et al. 2021; Patashnik et al. 2021], our adapter network is built on a multi-layer perceptron (MLP) architecture that takes as input the latent vector Wtgt+ and the target age (𝑎tgt), and outputs the offset vector ΔWtgt+ . Specifically, the 18 × 512 dimensional latent code Wtgt+ is first processed through a Global MLP, which produces a down-sampled global representation Wglobal of dimension 18×32, flattened to 1 × 512. Next, we design an Aging MLP that takes the scalar target age as input and generates a 1 × 16 dimensional age feature, 𝑎tgt-feat. We then train 18 independent Style MLPs, each operating on one of the𝑘 ∈ [1, 18] styles in the W+ space, to produce an offset vector for each style, ΔWtgt+ (𝑘). Each Style MLP receives the 1 × 512 dimensional age-transformed latent code from SAM, Wtgt+ (𝑘), the 1 × 512 dimensional global representation Wglobal, and the target aging feature 𝑎tgt-feat, and then outputs the per-style offset code ΔWtgt+ (𝑘). Both the Global, Aging, and 18 Style MLPs are designed as 2-layer neural networks with ReLU activation. This architecture enables the network to subtly and effectively adjust the latent representation, preserving the individual’s identity while incorporating personalized aging characteristics.

For each celebrity, we train our adapter network on a GPU A6000 for 10,000 iterations, which takes approximately 4 hours. We inherit SAM’s hyperparameters, including its original loss weights. Additionally, we set 𝜆pers-age = 1 for Eq. 5, 𝜆reg-extra = 1 for Eq. 6, and 𝜆reg = 1 for Eq. 7.

2 • Qi et al.

- D Comparison with SOTA Methods w/o Personalization As discussed in Sec. 4.2, we benchmark our approach against all available open-sourced pre-trained baselines, including SAM, CUSP, AgeTransGAN, and FADING, as shown in Fig. 9. We exclude RAGAN [Makhmudkhujaev et al. 2021] and PADA [Li et al. 2023] from our comparisons as they are not open-sourced. For baseline methods like CUSP and AgeTransGAN, which utilize pre-defined age groups based on FFHQ-Aging [Or-El et al. 2020], we interpolate between these age groups to demonstrate continuous aging, following the approach used by SAM [Alaluf et al. 2021].

For benchmarking, we primarily focus on identity-preserving performance rather than inversion performance when the target age matches the input age. Consequently, reconstruction metrics such as PSNR are not included in our evaluation.

- E Comparison with Naive Personalization Techniques.

We perform additional ablation studies using alternative personalization approaches on data for Al Pacino aged 30 ∼ 70 years, with results shown in Fig. 11. SAM Pers. f.t. behaves similarly to the pretrained SAM, as the latent codes are far from the latent center, limiting its editing capabilities. This aligns with the inversion-editability trade-off discussed in Sec. 3.3. SAM Pers. ft. + MyStyle [Nitzan et al. 2022a] first personalizes the SAM encoder, then tunes the decoder following the PTI pipeline [Roich et al. 2022]. However, this introduces significant artifacts due to changes in the latent distribution, which diverges from the pre-trained StyleGAN2 distribution. In SAM, global aging knowledge is learned with a fixed StyleGAN2 decoder, and modifying decoder weights distorts the latent space distribution, compromising the aging knowledge and introducing decoding artifacts.

For naive personalization using diffusion models, FADING + Dreambooth (DB) [Ruiz et al. 2023] overfits the aging results to the input image, especially when the target age lies outside the training age range. Additionally, this approach neglects age-related facial shape transformations, such as a toddler’s rounder face or proportional changes in facial features over time, which are caused by NTI + p2p as discussed in Sec. 2. We further adopt FADING’s paradigm using a more contemporary pipeline: FLUX 3 combined with RF-Inversion [Rout et al. 2025] and DreamBooth (DB), replacing the original Stable Diffusion 1.5 + NTI setup in FADING + DB. However, this updated pipeline still fails to extrapolate effectively, exhibiting similar overfitting failures as FADING + DB.

- F Why Personalizing the Encoder SAM?

Finetuning encoder-decoder GAN with limited personal data often leads to overfitting, mode collapse [Aghabozorgi et al. 2023], and data drift [Lee et al. 2019; Lu et al. 2020], preventing the model from generalizing to unseen test images of an individual [Qi et al.

- 2023]. Therefore, encoder-decoder GAN structures, like AgeTransGAN, necessitate a substantial amount of paired data to achieve effective personalization in aging transformations. For instance, personalizing the appearance of a celebrity such as Al Pacino would necessitate images of him at both ages 20 and 70, with consistent

- 3https://huggingface.co/black-forest-labs/FLUX.1-dev

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

[Figure 12]

SAM Pers. ft. + MyStyle

FADING + DB

FLUX + DB + RF-Inversion

Experiment SAM SAM Pers. ft.

Ours (30∼70)

ID𝑠𝑖𝑚(↑) 0.45 0.49 0.60 0.64 0.64 0.66

Fig. 11. We compare MyTM (Ours) with naive personalization techniques: SAM Pers. ft., SAM Pers. ft. + MyStyle, FADING + Dreambooth (DB), and FLUX + Dreambooth (DB) + RF-Inversion, trained on ages 30∼70 and tested within the same age range for Al Pacino. While SAM Pers. ft. + MyStyle achieves a high ID𝑠𝑖𝑚 score, it suffers from poor visual quality, resulting in adversarial examples for arcface.

pose, lighting, and expression. However, acquiring such data in realworld conditions is extremely challenging, as it demands rare and specific longitudinal images that capture individuals across a wide age span under controlled settings. This limitation makes encoderdecoder GANs less practical for applications where personalized aging transformations are desired. For diffusion models, there are several limitations in re-aging tasks: (1) They lack the W latent space, which enables fine-grained continuous aging control and editing [Dravid et al. 2024]. (2) Models like FADING, which use NTI + p2p for age editing, often struggle with the trade-off between inversion accuracy and editability [Rout et al. 2024]. Additionally, FADING frequently produces unstable results, as shown in Fig. 13, which we attribute to the unstable NTI optimization [Rout et al. 2024; Zheng and Wu 2024]. New stable optimization-free methods could be explored for diffusion models in the future. (3) VQ auto-encoders, commonly used in diffusion to encode images, can introduce artifacts, particularly in the human face domain [Mokady et al. 2023]. These issues highlight the need for an alternative approach, such as utilizing StyleGAN2’s well-trained latent space and optimization-free e4e encoder [Tov et al. 2021], to achieve high-quality, artifact-free re-aging transformations.

[Figure 13]

- Fig. 12. Limitations of MyTM. Our method may struggle with accessories (e.g., glasses), as these elements are not consistently handled by the e4e encoder [Tov et al. 2021].

[Figure 14]

- Fig. 13. Visual results of FADING using identical input and inference code. The instability in age transformation arises from the optimization of NTI [Mokady et al. 2023], leading to inconsistencies.

- G Why Not Use a Reference Image for Face-Swapping?

Firstly, obtaining images of a person at any arbitrary age is often challenging, particularly high-quality images comparable to our synthesized faces at 1024x1024 resolution. Even if reference images at the target age are available, face-swapping techniques [Chen et al.

2020; Gao et al. 2021; Xu et al. 2022d] generally yields optimal results when the source and target faces share similar styles, such as pose, expression, and lighting. Significant style differences between the source and target faces can cause artifacts like flickering, particularly in real-world video scenarios [Choi et al. 2024; Ren et al. 2023].

