# arXiv:2408.09739v1[cs.CV]19Aug2024

## TraDiffusion: Trajectory-Based Training-Free Image Generation

### Mingrui Wu1*, Oucheng Huang1∗, Jiayi Ji1, Jiale Li1, Xinyue Cai1, Huafeng Kuang1, Jianzhuang Liu2, Xiaoshuai Sun1, Rongrong Ji1

1Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University, 361005, P.R. China 2Shenzhen Institute of Advanced Technology, University of Chinese Academy of Sciences

###### Abstract

In this work, we propose a training-free, trajectory-based controllable T2I approach, termed TraDiffusion. This novel method allows users to effortlessly guide image generation via mouse trajectories. To achieve precise control, we design a distance awareness energy function to effectively guide latent variables, ensuring that the focus of generation is within the areas defined by the trajectory. The energy function encompasses a control function to draw the generation closer to the specified trajectory and a movement function to diminish activity in areas distant from the trajectory. Through extensive experiments and qualitative assessments on the COCO dataset, the results reveal that TraDiffusion facilitates simpler, more natural image control. Moreover, it showcases the ability to manipulate salient regions, attributes, and relationships within the generated images, alongside visual input based on arbitrary or enhanced trajectories. The code: https://github.com/och-mac/TraDiffusion.

### Introduction

Over the past few years, the field of image generation has experienced remarkable progress, particularly with the development of models (Goodfellow et al. 2020; Ho, Jain, and Abbeel 2020; Rombach et al. 2022; Saharia et al. 2022; Ramesh et al. 2022) trained on large-scale datasets sourced from the web. These models, particularly those that are text conditioned, have shown impressive capabilities in creating high-quality images that align with the text descriptions provided (Dhariwal and Nichol 2021; Song, Meng, and Ermon 2020; Isola et al. 2017; Song et al. 2020). However, while text-based control has been beneficial, it often lacks the precision and intuitive manipulation needed for fine-grained adjustments in the generated images. As a result, there has been growing interest in exploring alternative conditioning methods (Li et al. 2023; Nichol et al. 2021; Zhang et al. 2020; Zhang, Rao, and Agrawala 2023), such as edges, normal maps, and semantic layouts, to offer more nuanced control over the generated outputs. These diverse conditioning techniques broaden the scope of applications for generative models, extending from design tasks to data generation, among others.

Traditional methods (Zhang, Rao, and Agrawala 2023; Kim et al. 2023) with conditions such as edges, normal

*Equal Contribution.

maps, and semantic layouts can achieve precise object shape control, while box-based methods enable coarse layout control. However, we find that trajectory-based control aligns more closely with actual human attention (Xu et al. 2023; Pont-Tuset et al. 2020), and provides a level of control granularity between the fine mask and the coarse box, as shown in Figure 1. Therefore, in parallel with these traditional layout control methods, this paper proposes a trajectory-based approach for text-to-image generation to fill this gap.

The central challenge we address is the utilization of trajectory to control image generation. Several studies (Hertz et al. 2022; Kim et al. 2023; Chen, Laina, and Vedaldi 2024) have successfully manipulated images by adjusting attention maps in the text-related cross-attention layers on the stable diffusion models (Rombach et al. 2022), achieving effective control without additional training—a notably convenient approach. A standout method (Chen, Laina, and Vedaldi 2024) among these, known as backward guidance, indirectly adjusts the attention by updating the latent variable. This technique, compared to direct attention map manipulation, yields images that are smoother and more accurately aligned with intended outcomes. It capitalizes on the straightforward nature of box-based conditioning, which effectively focuses attention within a specified bounding box region and minimizes it outside, enhancing the relevance of generated content. However, given the inherently sparse nature of trajectory-based control, applying backward guidance in this context poses significant challenges, requiring innovative adaptations to harness its potential effectively.

In this paper, we propose a novel training-free trajectoryconditioned image generation method. This technique enables users to guide the positions of image elements described in text prompts through trajectories, significantly enhancing the user experience by providing a straightforward way to control the appearance of generated images. To enable effective trajectory-based control, we introduce a distance awareness energy function. which updates latent variables, guiding the target to exhibit a stronger response in regions closer to the specified trajectory. The energy function comprises two main components: a control function, which directs the target towards the trajectory, and a movement function, which reduces the response in irrelevant areas distant from the trajectory.

Our trajectory-based approach offers a promising solution

Prompt: “A train is coming down the track.”

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

- (a) Mask-conditioned method
- (b) Box-conditioned method

[Figure 8]

[Figure 9]

(c) Ours

Figure 1: Comparing the mask-conditioned method (a), box-conidtioned method (b) and our trajectory-conditioned method (c). The mask-conditioned method tends to have precise object shape control with a fine mask, which needs to be obtained by a specialized tool. The box-conidtioned methods enable coarse layout control. However, our trajectory-conditioned method provides a level of control granularity between the fine mask and the coarse box, which is user-friendly.

for layout-controlled image generation. Via qualitative and quantitative evaluations, we demonstrate the superior control capabilities of our method, achieving remarkable improvements in both the quality and accuracy of generated images. Moreover, our method exhibits adaptability to arbitrary trajectory inputs, allowing for precise control over object attributes, relationships, and salient regions.

generations by providing more precise control over the output, thereby improving the alignment with the input prompt.

#### Controlling Image Generation with Layouts

Layout controlled image generation introduces spatial conditioning to guide the image generation process. A lot of methods (Feng et al. 2024; Gafni et al. 2022; Hertz et al. 2022; Isola et al. 2017; Li et al. 2023; Liu, Breuel, and Kautz 2017; Wang et al. 2018; Xu et al. 2018; Zhang, Rao, and Agrawala 2023; Zhang et al. 2021; Zhu et al. 2017; Chen, Laina, and Vedaldi 2024; Feng et al. 2022; Kim et al. 2023; Xie et al. 2023; Yang et al. 2023; Wang et al. 2024; BarTal et al. 2023; Avrahami et al. 2023; Huang et al. 2023, 2022; Johnson, Gupta, and Fei-Fei 2018; Park et al. 2019; Sun and Wu 2019; Sylvain et al. 2021; Yang et al. 2022; Zhao et al. 2019; Qu et al. 2023; Li, Zhang, and Wang 2021; Tan et al. 2023; Li et al. 2020; Wu et al. 2022; Qin et al.

### Related Work

#### Image Diffusion Models

Image diffusion models represent a pivotal advancement in the domain of text-to-image generation. These models (Ho, Jain, and Abbeel 2020; Sohl-Dickstein et al. 2015; Song et al. 2020; Avrahami, Lischinski, and Fried 2022; Liu et al. 2022; Ruiz et al. 2023; Huang et al. 2024) operate by learning the intricate process of transforming textual descriptions into coherent and visually appealing images. One prominent approach within this paradigm is the Stable Diffusion Model (SDM) (Rombach et al. 2022), which enhances the fidelity and stability of image generation. The SDM is distinguished by its iterative denoising process initiated from a random noise map. This method, often performing in the latent space of a Variational AutoEncoder (VAE) (Kingma and Welling 2013; Van Den Oord, Vinyals et al. 2017), enables the generation of images that faithfully captures the semantics conveyed in the input text. Notably, SDMs leverage pretrained language models (Radford et al. 2021) to encode textual inputs into latent feature vectors, facilitating efficient exploration of the image manifold. While image diffusion models excel in synthesizing images from textual prompts, accurately conveying all details of the image remains a challenge, particularly with longer prompts or atypical scenes. To address this issue, recent studies have explored the effectiveness of classifier-free guidance (Ho and Salimans 2022). This innovative approach enhances the faithfulness of image

- 2021; Ren et al. 2024; Zakraoui et al. 2021) offer different approaches to incorporate spatial controls for enhancing image synthesis. GLIGEN (Li et al. 2023) and ControlNet (Zhang, Rao, and Agrawala 2023) are notable examples that introduce finer-grained spatial control mechanisms. These methods leverage large pretrained diffusion models and allow users to specify spatial conditions such as Canny edges, Hough lines, user scribbles, human key points, segmentation maps, shape normals, depths, cartoon line drawings and bounding boxes to define desired image compositions. However, the advancement of spatially controlled image generation models have also brought significant training costs, stimulating the development of a range of trainingfree layout control and image editing methods (Hertz et al.
- 2022; Xie et al. 2023; Kim et al. 2023). These approaches leverage the inherent capabilities of cross-attention layers found in state-of-the-art diffusion models, which establish connections between word tokens and the spatial layouts of generated images. By exploiting this connection, these

methods enable effective spatial control over the image synthesis process without the need for specialized training procedures.

“A pikachu is playing a basketball.”

Text Encoder

Denoising UNet

### Preliminaries

K V

K V

#### Problem Definition

Q

Cross Attention

Cross Attention

We aim to improve layout control in image generation, which is formulated as I = f(p,{c1,··· ,cn}), where the prompt p and a set of layout conditions {c1,··· ,cn} are fed into the pretrained model f to generate target image I. Given the model f, we hope to generate an image which aligns with the extra layout without further training or finetuning.

𝒛𝒕 𝒛𝒕−𝟏

[Figure 10]

[Figure 11]

E ( , )

[Figure 12]

[Figure 13]

Latent Optimization

###### E ( , )

#### Stable Diffusion

Stable Diffusion (SD) (Rombach et al. 2022) is a modern text-to-image generator based on diffusion (Saharia et al. 2022). SD consists of several key components: an image encoder and decoder, a text encoder, and a denoising network operating within a latent space.

Figure 2: Overview of the distance awareness guidance. With the provided trajectories, we calculate distance matrices for each trajectory. Subsequently, we compute the distance awareness energy function between these distance matrices and the attention map of each object. Finally, during the inference process, we conduct backpropagation to optimize the latent code.

During inference, the text encoder transforms the input prompt p into a set of fixed-dimensional tokens y = {y1,··· ,ym}. Then the denoising network, usually an UNet (Ronneberger, Fischer, and Brox 2015) with crossattention layers, takes a random noised sample latent code zt as input and returns zt−1. This denoising process is iterated t times to obtain the final latent code z0. Finally, the latent code z0 is fed into the image decoder to get the generated image.

object, rather than limiting the object to a specified shape or size. So we introduce trajectories to guide the layout of the generated image. Specifically, we provide a trajectory for a specified word or phrase in the prompt. The problem can be formulated as I = f(p,{(w1,l1),··· ,(wn,ln)}), where p represents the global prompt, and a set of word-line pairs (wi,li) serving as layout conditions, which are fed into the pretrained model f to generate the target image I. Based on the trajectories, we guide the locations of instances, attributes, relationships and actions without further training or finetuning. And the user can easily draw trajectories for image generation through the mouse or pen.

In SD, the denoising network plays an important role in connecting the text condition and image information. Its core mechanism lies in the cross-attention layers. The crossattention takes the transformed latent code z(τ) in layer τ as query, and the transformed text conditions y(τ) as keys and values, and the attention map is obtained as follows,

z(τ) · (y(τ))T √dk

A(τ) = softmax(

), (1)

#### Distance Awareness Guidance

Inspired by (Chen, Laina, and Vedaldi 2024), we try to control the image generation based on trajectories with backward guidance. However, due to the sparsity of the trajectories, it is difficult to directly combine backward guidance. A natural idea is to get the prior structure of an object through the attention maps of cross-attention layers, rather than directly using the trajectories to achieve backward guidance.

where dk is a scale factor, and A(τ) consists of A(iτ), i ∈ {1,··· ,m}, representing the impact of the i-th token on the

output.

### Method

In this section, we introduce the trajectory-based controllable text-to-image generation method (as shown in Figure 2) using the pretrained diffusion model (Rombach et al.

Prior Structure Based Guidance. To get the prior structure of an object, we first perform denoising of the Tk steps on the Stable Diffusion model and apply a threshold on the attention map of the current step to obtain a binary mask. Then we move the mask to align the center of the trajectory. By this, we can use this mask to replace the box to compute the energy function proposed in (Chen, Laina, and Vedaldi 2024).

- 2022), and describe the distance awareness energy function that combines the trajectory to achieve training-free layout control.

#### Controlling Image Generation with Trajectory

Previous works (Kim et al. 2023; Xie et al. 2023; Chen, Laina, and Vedaldi 2024) are mainly based on masks or boxes to control the layout, but masks are fine-grained, which is not user-friendly, and boxes are too coarse to limit the object area. These methods directly affect the prior structure of the generated object in the image. In some cases, we only want to guide the approximate location and shape of the

However, we find that this approach has several unavoidable drawbacks, as shown in Figure 8 of Appendix. a) In order to get a good quality mask, we have to carefully select the appropriate threshold, as well as suitable denoising steps. Too many denoising steps would produce a fine mask

Input Trajectory Attention Map Generated Image

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

Prompt:“A bear is on the grass at spring.”

Prompt: “A small train is coming along the track in the snow.”

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

Prompt: “A uniquely shaped building.”

Prompt: “A cute dog standing in a forest at autumn, high quality, professional photo.”

Figure 4: Examples of controlling the object shapes with arbitrary trajectories. We can adjust the posture of the object (top) or specify the approximate shape of the object (bottom) by varying the given trajectory.

Figure 3: Examples of controlling the salient areas of the objects with trajectories. We can adjust the position of the local salient area of the object by enhancing the local trajectory.

small value used to avoid division by zero, and A(µiτ) is the attention map determining how strongly each location µ in

but at the same time introduce an excessive amount of additional computation and an overfitting object prior. b) Since the Stable Diffusion model does not always produce highquality images, it always produces some unusable masks in some cases. Taken together, prior structure based guidance cannot be a robust guidance strategy.

layer τ is associated with the i-th token wi. This function steers the object to approach the given trajectory.

However, this does not effectively inhibit the attention response of the object in irrelevant regions far from the trajectory. So, we add a movement function to suppress the attention response from irrelevant regions far from the trajectory of the object accordingly. The movement function is formulated as

Distance Awareness Energy Function. To overcome the above limitations of prior structure based guidance, we propose to use a distance awareness energy function for guidance, as shown in Figure 2. Specifically, we first apply a control function to guide the object to approach a given trajectory, which is formulated as

A(µiτ) µ DµiA(µiτ)

Em A(τ),li,wi = (1 − µ

)2. (3)

The final distance awareness energy function is the combination of Ec and Em:

(Dµi + ϵ)−1A(µiτ) µ A(µiτ)

Ec A(τ),li,wi = (1 − µ

)2, (2)

E = Ec + λEm, (4)

where Dµi is a distance matrix computed by the OpenCV (Bradski 2000) function “distanceTransform”, in which each value denotes the distance from each location µ of the attention map to the given trajectory li, ϵ is a very

where λ is an adjustable hyperparameter. By computing E as loss and backpropagation to update the latent zt, we encourage the response of the cross-attention map of the i-th token to obtain higher values in the area close to the trajec-

Stable Diffusion Ours

Visual Input Text Inversion + Ours

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

- (a)

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Prompt: “A man is touching a dog.”

- (b)

Prompt: “A man wearing a red shirt and blue pants. high quality. ”

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Prompt:“A man wearing a blue shirt and red pants. high quality.”

Prompt: “A cute <*> standing in a forest at autumn, high quality, professional photo.”

Figure 6: Examples of controlling visual input.

not appear in the caption, so the previous works (Bar-Tal et al. 2023; Chen, Laina, and Vedaldi 2024) pad the instance names into the caption. But this inevitably changes the effect of the prompt on generating images, so we prioritize sampling images with instances in the captions rather than padding the captions.

Figure 5: Examples of controlling the attribute and relationship of objects. Based on trajectories, we can overcome the attribute confusion issue of the pre-trained Stable Diffusion model, generating visual results consistent with the given prompt (a), and adjust the positions of interactions (b).

Evaluation Metrics. We measure the quality of the generated images with FID. However, the traditional metrics are not suitable for evaluating the layout control of trajectorybased image generation methods, so we propose a novel Distance To Line (DTL) metric, which is defined as

tory li, which can be formulated as

µ∈mask e−D

µi

1 n i∈N

, (6)

DTL =

µ∈mask 1

zt ← zt − σt2η∇zt

E A(τ),li,wi , (5)

τ∈Φ i∈N

where mask is obtained by applying the YOLOv8mSeg (Jocher, Chaurasia, and Qiu 2023; Redmon et al. 2016) on the generated image, and N = {1,··· ,n}. The larger the DTL, the closer the generated object is to the given trajectory. Therefore, DTL not only verifies whether the desired objects are generated but also examines the alignment of the layout. We report mean DTL on all generated images.

where η > 0 is a hyperparameter controlling the strength of the guidance, Φ is a set of layers in UNet (Ronneberger, Fischer, and Brox 2015), N = {1,··· ,n}, and σt =

(1 − αt)/αt, with αt being a pre-defined parameter of diffusion (Ho, Jain, and Abbeel 2020; Rombach et al. 2022; Song, Meng, and Ermon 2020).

### Experiments

#### Experimental Setup

Evaluation Benchmark. We evaluate our approach on COCO2014 (Lin et al. 2014). Following previous works (Bar-Tal et al. 2023; Chen, Laina, and Vedaldi 2024), we randomly select 1000 images from its validation set, and each image is paired with a caption and has up to 3 instances with masks that occupy more than 5% of the image. However, the instances that are randomly sampled may

Implementation Details. Following the setting of (Chen, Laina, and Vedaldi 2024), we utilize Stable-Diffusion (SD) V-1.5 (Rombach et al. 2022) as the default pre-trained diffusion model. We select the cross-attention maps of the same layers as (Chen, Laina, and Vedaldi 2024) for computing the energy function. And the backpropagation of the energy function is performed during the initial 10 steps of the diffusion process and repeated 5 times at each step. The hyperparameters λ = 10 and η = 30. We fix the random seeds to 450. The experiments are performed on a RTX-3090 GPU.

Prior Structure Trajectory Expanding Ours w/o Movement Ours

Distance Matrix

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Prompt: “A small train is coming down the track.”

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Prompt: “The man is walking the dog.”

- Figure 7: Qualitative analysis of the components in our proposed method, including prior structure based guidance (left), expanding the trajectory to obtain a mask (middle), and our method without and with the movement function (right). We show the input condition and generated image for each component, and an extra attention map for our method.

#### Applications

Method DTL(↑) FID(↓)

Controlling the Salient Areas of Objects. Typically, attention models exhibit higher responses in salient regions of objects (Xu et al. 2015; Oktay et al. 2018; Zhang et al. 2019; Zeiler and Fergus 2014; Hu, Shen, and Sun 2018). Hence, we investigate whether enhancing local trajectories can effectively control the positions of salient regions within objects. As illustrated in Figure 3, we showcase our method’s capability to guide attention maps by manipulating local trajectories, thereby exerting control over the positioning of specific elements such as the train’s head and the dog’s head.

Stable Diffusion (Rombach et al. 2022) 0.0043 68.33 Prior structure 0.0077 66.91 Trajectory expanding 0.0080 64.87

Ours w/o movement 0.0119 64.68 Ours 0.0156 68.53

Table 1: Ablation study on each component of our method. Compared to the prior structure based guidance method and the trajectory expanding method, our method demonstrates the strongest level of control, with a DTL score about twice as high as those of the two baselines.

Controlling Shapes with Arbitrary Trajectories. We analyze the adaptability of our method to incorporate trajectory inputs of arbitrary shapes to generate the desired object shapes. As illustrated in Figure 4, by varying the trajectory, we can adjust the posture of the object, such as guiding the posture of a ‘bear’ into various positions such as crawling, standing, and sitting (Figure 4 top). Additionally, we can specify the approximate shape of the object by the trajectory (Figure 4 bottom).

visivility of the input objects.

#### Ablation Study

We perform the ablation study to validate the effect of each component in our proposed method. We first evaluate the Stable Diffusion model (Rombach et al. 2022) for reference. We consider the prior structure based guidance as the baseline, and a method of expanding to the fixed size outwards along the trajectory to obtain a mask is also compared. Then we experiment with only the control function to validate the controllability, and further add the movement function to verify that the method is able to suppress the response of object at the irrelevant regions far from the trajectory.

Controlling Attributes and Relationship. We analyze whether our method can control the attributes of objects and the relationships between objects. As illustrated in Figure 5, attribute confusion exists in the SD model. Despite our efforts to generate the shirts and pants in varied colors, it persistently confuses the attributes, resulting in the wrong colors for both. By controlling the attributes of the object based on trajectories, we can largely overcome the attribute confusion issue in the pre-trained Stable Diffusion model, generating visual results consistent with the given prompt (Figure 5 a). Additionally, we can adjust the positions of interactions between objects by adjusting the trajectories (Figure 5 b).

The results are shown in Table 1. We can observe that the prior structure based guidance and the trajectory expanding methods exhibit similarly low DTL scores. However, our method shows an improvement of about 50% in DTL compared to the two baselines when the movement loss is not added. Through further augmentation by the movement loss, our method demonstrates a significant 100% enhancement in DTL. Although there is a slight decrease in FID performance after adding the movement loss, we believe that this minor difference can be negligible due to the complexity of the COCO image distribution.

Controlling Visual Input. We analyze whether our method can control the visual input. As shown in Figure 6, we can adjust the orientations of the visual input objects through trajectories. However, it is worth noting that finer adjustments pose challenges, which relies on the available

Method Type Quality(↑) Controllability(↑) User-Friendliness(↑)

BoxDiff Box 3.52 3.22 2.07 Backward Guidance Box 3.24 3.69 2.07 DenseDiffusion Mask 3.30 3.56 1.07

Ours Trajectory 3.72 4.04 2.87

Table 2: The user studies, including quality, controllability (score from 1 to 5), and user-friendliness (score from 1 to 3).

The qualitative analysis of the components in our proposed method is shown in Figure 7. We can observe that both of the trajectory expanding based method and the prior structure based guidance method fail to generate outputs that strictly adhere to the trajectory control, potentially resulting in similar issues encountered with the box-based and maskbased approaches. Additionally, mask-based methods may struggle to capture effective prior structures of the objects. In contrast, our approach, without introducing additional movement loss, is capable of generating objects that adhere to the trajectory (top). However, due to the lack of suppression of irrelevant positions in the attention far away from the given trajectory, extra object generations occur (bottom). This issue is alleviated by further adding the movement loss.

The effect of the hyperparameter λ is shown in Table 4 and Figure 12 of Appendix. It shows that when λ = 20, it yields the highest DTL results. However, we also notice a comparable performance when λ = 10, and increasing λ further leads to a significant decrease in FID. In addition, as shown in Figure 12 of Appendix, we observe that excessively large values lead to over-suppression of the entire image, while values in the range of [5,10] yield the best results. Therefore, the default λ is set to 10.

#### Comparison with Prior Work

We compare our method with previous layout text-to-image generation methods, including mask-conditioned DenseDiffusion (Kim et al. 2023) and ControlNet (Zhang, Rao, and Agrawala 2023), and box-conditioned BoxDiff (Xie et al.

- 2023) and Backward Guidance (Chen, Laina, and Vedaldi
- 2024), in which DenseDiffusion, BoxDiff, and Backward Guidance are all training-free. In our method, we sample the trajectories inside the boxes or masks. Typically, existing evaluation metrics, like YOLO-score and mIOU, are inevitably biased towards each type of layout control method due to the lack of a unified and feasible metric for comparison. To address this, we compare our method with previous training-free methods by providing user studies on the results’ quality, controllability, and user-friendliness, based on the average scores from 15 users, as shown in Table 2.

The visual examples of the comparisons are shown in Figure 11 of Appendix. Mask-based methods often introduce excessive manual priors by utilizing too detailed masks, leading to the overly controlled generation of distorted and unrealistic objects. For example, this can be observed in the generation of the distorted airplanes (c) and elephants (d). Conversely, box-based methods, with their too coarse control conditions, completely disregard prior information about the object, leading to the generation of de-

formed and unnatural images, such as the floating frisbee (a), oversized umbrella (b), and snowboard depicted at a unreasonable angle (e). In contrast, our trajectory-based approach does not excessively intervene in the prior structure of the object and, with user-friendly simple controls, is capable of generating natural images.

In addition, it is noteworthy that trained layout text-toimage generation methods often have limitations in accommodating diverse semantic categories and conditional domains. This often necessitates retraining to adapt to new conditions, incurring additional cost and time. However, our innovative training-free method can seamlessly adapts the model to any semantic input, offering unparalleled convenience and flexibility to users.

### Limitations

While we have demonstrated simple and natural layout control by trajectory, our method is subject to a few limitations. Firstly, same as other training-free layout control textto-image generation methods, the quality of images generated based on trajectory is limited by the pre-trained SD model. Adjustments to both the prompt and trajectory may be necessary to achieve desired outcome. Secondly, similar to (Chen, Laina, and Vedaldi 2024), we also incur twice the inference cost compared to the pre-trained SD model. Thirdly, although trajectories are less coarse than bounding boxes, achieving precise adjustments to the shapes of objects remains challenging. Fourthly, we have currently only explored a limited range of possibilities in trajectory-based image generation, and we look forward to further exploration of its diverse applications in future work.

### Conclusions

In this work, we propose a trajectory-based layout control method for text-to-image generation without additional training or fine-tuning. Combining with the proposed distance awareness energy function to optimize the latent code of the Stable Diffusion model, we achieve user-friendly layout control. In the energy function, the control function steers the object to approach the given trajectory, and the movement function inhibits the response of the object in irrelevant regions far from the trajectory. A set of experiments show that our method can generate images more simply and naturally. Moreover, it exhibits adaptability to arbitrary trajectory inputs, allowing for precise control over object attributes, relationships, and salient regions. We hope that our work can inspire the community to explore more user-friendly text-to-image techniques, as well as uncover more trajectory-based applications.

### Acknowledgments

This work was supported by National Science and Technology Major Project (No. 2022ZD0118201), the National Science Fund for Distinguished Young Scholars (No.62025603), the National Natural Science Foundation of China (No. U21B2037, No. U22B2051, No. 62072389, No. 62302411), China Postdoctoral Science Foundation (No. 2023M732948), the Natural Science Foundation of Fujian Province of China (No.2022J06001), and partially sponsored by CCF-NetEase ThunderFire Innovation Research Funding (NO. CCF-Netease 202301).

### References

Avrahami, O.; Hayes, T.; Gafni, O.; Gupta, S.; Taigman, Y.; Parikh, D.; Lischinski, D.; Fried, O.; and Yin, X. 2023. Spatext: Spatio-textual representation for controllable image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18370–18380.

Avrahami, O.; Lischinski, D.; and Fried, O. 2022. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18208–18218.

Bar-Tal, O.; Yariv, L.; Lipman, Y.; and Dekel, T. 2023. Multidiffusion: Fusing diffusion paths for controlled image generation.

Bradski, G. 2000. The opencv library. Dr. Dobb’s Journal: Software Tools for the Professional Programmer, 25(11): 120–123.

Chen, M.; Laina, I.; and Vedaldi, A. 2024. Training-free layout control with cross-attention guidance. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 5343–5353.

Dhariwal, P.; and Nichol, A. 2021. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34: 8780–8794.

Feng, W.; He, X.; Fu, T.-J.; Jampani, V.; Akula, A.; Narayana, P.; Basu, S.; Wang, X. E.; and Wang, W. Y. 2022. Training-free structured diffusion guidance for compositional text-to-image synthesis. arXiv preprint arXiv:2212.05032.

Feng, W.; Zhu, W.; Fu, T.-j.; Jampani, V.; Akula, A.; He, X.; Basu, S.; Wang, X. E.; and Wang, W. Y. 2024. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems, 36.

Gafni, O.; Polyak, A.; Ashual, O.; Sheynin, S.; Parikh, D.; and Taigman, Y. 2022. Make-a-scene: Scene-based text-toimage generation with human priors. In European Conference on Computer Vision, 89–106. Springer.

Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A.; and Bengio, Y.

- 2020. Generative adversarial networks. Communications of the ACM, 63(11): 139–144. Hertz, A.; Mokady, R.; Tenenbaum, J.; Aberman, K.; Pritch, Y.; and Cohen-Or, D. 2022. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33: 6840–6851.

Ho, J.; and Salimans, T. 2022. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598.

Hu, J.; Shen, L.; and Sun, G. 2018. Squeeze-and-excitation networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 7132–7141.

Huang, L.; Chen, D.; Liu, Y.; Shen, Y.; Zhao, D.; and Zhou, J. 2023. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778.

- Huang, X.; Mallya, A.; Wang, T.-C.; and Liu, M.-Y. 2022. Multimodal conditional image synthesis with product-ofexperts gans. In European Conference on Computer Vision, 91–109. Springer.
- Huang, Y.; Huang, J.; Liu, Y.; Yan, M.; Lv, J.; Liu, J.; Xiong, W.; Zhang, H.; Chen, S.; and Cao, L. 2024. Diffusion model-based image editing: A survey. arXiv preprint arXiv:2402.17525.

Isola, P.; Zhu, J.-Y.; Zhou, T.; and Efros, A. A. 2017. Imageto-image translation with conditional adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1125–1134.

Jocher, G.; Chaurasia, A.; and Qiu, J. 2023. Ultralytics YOLOv8.

Johnson, J.; Gupta, A.; and Fei-Fei, L. 2018. Image generation from scene graphs. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 1219– 1228.

Kim, Y.; Lee, J.; Kim, J.-H.; Ha, J.-W.; and Zhu, J.-Y. 2023. Dense text-to-image generation with attention modulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 7701–7711.

Kingma, D. P.; and Welling, M. 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

Li, C.; Zhang, P.; and Wang, C. 2021. Harmonious textual layout generation over natural images via deep aesthetics learning. IEEE Transactions on Multimedia, 24: 3416–3428. Li, Y.; Cheng, Y.; Gan, Z.; Yu, L.; Wang, L.; and Liu, J. 2020. Bachgan: High-resolution image synthesis from salient object layout. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8365–8374.

Li, Y.; Liu, H.; Wu, Q.; Mu, F.; Yang, J.; Gao, J.; Li, C.; and Lee, Y. J. 2023. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22511–22521.

Lin, T.-Y.; Maire, M.; Belongie, S.; Hays, J.; Perona, P.; Ramanan, D.; Doll´ar, P.; and Zitnick, C. L. 2014. Microsoft coco: Common objects in context. In Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, 740– 755. Springer.

Liu, M.-Y.; Breuel, T.; and Kautz, J. 2017. Unsupervised image-to-image translation networks. Advances in Neural Information Processing Systems, 30.

Liu, N.; Li, S.; Du, Y.; Torralba, A.; and Tenenbaum, J. B. 2022. Compositional visual generation with composable diffusion models. In European Conference on Computer Vision, 423–439. Springer.

Nichol, A.; Dhariwal, P.; Ramesh, A.; Shyam, P.; Mishkin,

- P.; McGrew, B.; Sutskever, I.; and Chen, M. 2021. Glide: Towards photorealistic image generation and editing with textguided diffusion models. arXiv preprint arXiv:2112.10741. Oktay, O.; Schlemper, J.; Folgoc, L. L.; Lee, M.; Heinrich,

- M.; Misawa, K.; Mori, K.; McDonagh, S.; Hammerla, N. Y.; Kainz, B.; et al. 2018. Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999.

Park, T.; Liu, M.-Y.; Wang, T.-C.; and Zhu, J.-Y. 2019. Semantic image synthesis with spatially-adaptive normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2337–2346.

Pont-Tuset, J.; Uijlings, J.; Changpinyo, S.; Soricut, R.; and Ferrari, V. 2020. Connecting vision and language with localized narratives. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16, 647–664. Springer.

Qin, Z.; Zhong, W.; Hu, F.; Yang, X.; Ye, L.; and Zhang,

- Q. 2021. Layout Structure Assisted Indoor Image Generation. In 2021 IEEE 4th International Conference on Multimedia Information Processing and Retrieval (MIPR), 323–

329. IEEE.

Qu, L.; Wu, S.; Fei, H.; Nie, L.; and Chua, T.-S. 2023. Layoutllm-t2i: Eliciting layout guidance from llm for textto-image generation. In Proceedings of the 31st ACM International Conference on Multimedia, 643–654.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 8748–8763. PMLR.

Ramesh, A.; Dhariwal, P.; Nichol, A.; Chu, C.; and Chen, M. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2): 3.

Redmon, J.; Divvala, S.; Girshick, R.; and Farhadi, A. 2016. You only look once: Unified, real-time object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 779–788.

Ren, J.; Xu, M.; Wu, J.-C.; Liu, Z.; Xiang, T.; and Toisoul, A. 2024. Move Anything with Layered Scene Diffusion. arXiv:2404.07178.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10684– 10695.

Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, 234–241. Springer.

Ruiz, N.; Li, Y.; Jampani, V.; Pritch, Y.; Rubinstein, M.; and Aberman, K. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22500–22510.

Saharia, C.; Chan, W.; Saxena, S.; Li, L.; Whang, J.; Denton, E. L.; Ghasemipour, K.; Gontijo Lopes, R.; Karagol Ayan, B.; Salimans, T.; et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35: 36479–36494.

Sohl-Dickstein, J.; Weiss, E.; Maheswaranathan, N.; and Ganguli, S. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, 2256–2265. PMLR.

Song, J.; Meng, C.; and Ermon, S. 2020. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502.

Song, Y.; Sohl-Dickstein, J.; Kingma, D. P.; Kumar, A.; Ermon, S.; and Poole, B. 2020. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456.

Sun, W.; and Wu, T. 2019. Image synthesis from reconfigurable layout and style. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 10531–10540. Sylvain, T.; Zhang, P.; Bengio, Y.; Hjelm, R. D.; and Sharma, S. 2021. Object-centric image generation from layouts. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, 2647–2655.

Tan, H.; Yin, B.; Wei, K.; Liu, X.; and Li, X. 2023. Alrgan: Adaptive layout refinement for text-to-image synthesis. IEEE Transactions on Multimedia.

Van Den Oord, A.; Vinyals, O.; et al. 2017. Neural discrete representation learning. Advances in Neural Information Processing Systems, 30.

Wang, T.-C.; Liu, M.-Y.; Zhu, J.-Y.; Tao, A.; Kautz, J.; and Catanzaro, B. 2018. High-resolution image synthesis and semantic manipulation with conditional gans. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 8798–8807.

Wang, X.; Darrell, T.; Rambhatla, S. S.; Girdhar, R.; and Misra, I. 2024. InstanceDiffusion: Instance-level Control for Image Generation. arXiv preprint arXiv:2402.03290.

Wu, S.; Tang, H.; Jing, X.-Y.; Zhao, H.; Qian, J.; Sebe, N.; and Yan, Y. 2022. Cross-view panorama image synthesis. IEEE Transactions on Multimedia.

Xie, J.; Li, Y.; Huang, Y.; Liu, H.; Zhang, W.; Zheng, Y.; and Shou, M. Z. 2023. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 7452–7461.

- Xu, J.; Zhou, X.; Yan, S.; Gu, X.; Arnab, A.; Sun, C.; Wang, X.; and Schmid, C. 2023. Pixel aligned language models. arXiv preprint arXiv:2312.09237.
- Xu, K.; Ba, J.; Kiros, R.; Cho, K.; Courville, A.; Salakhudinov, R.; Zemel, R.; and Bengio, Y. 2015. Show, attend and tell: Neural image caption generation with visual attention.

In International Conference on Machine Learning, 2048–

2057. PMLR.

Xu, T.; Zhang, P.; Huang, Q.; Zhang, H.; Gan, Z.; Huang, X.; and He, X. 2018. Attngan: Fine-grained text to image generation with attentional generative adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 1316–1324.

Yang, Z.; Liu, D.; Wang, C.; Yang, J.; and Tao, D. 2022. Modeling image composition for complex scene generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7764–7773.

Yang, Z.; Wang, J.; Gan, Z.; Li, L.; Lin, K.; Wu, C.; Duan,

- N.; Liu, Z.; Liu, C.; Zeng, M.; et al. 2023. Reco: Regioncontrolled text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14246–14255.

Zakraoui, J.; Saleh, M.; Al-Maadeed, S.; and Jaam, J. M.

- 2021. Improving text-to-image generation with object layout guidance. Multimedia Tools and Applications, 80(18): 27423–27443.

Zeiler, M. D.; and Fergus, R. 2014. Visualizing and understanding convolutional networks. In Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I 13, 818– 833. Springer.

Zhang, H.; Goodfellow, I.; Metaxas, D.; and Odena, A. 2019. Self-attention generative adversarial networks. In International Conference on Machine Learning, 7354–7363. PMLR.

Zhang, L.; Chen, Q.; Hu, B.; and Jiang, S. 2020. Text-guided neural image inpainting. In Proceedings of the 28th ACM International Conference on Multimedia, 1302–1310.

Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 3836–3847.

Zhang, Z.; Ma, J.; Zhou, C.; Men, R.; Li, Z.; Ding, M.; Tang, J.; Zhou, J.; and Yang, H. 2021. UFC-BERT: Unifying multi-modal controls for conditional image synthesis. Advances in Neural Information Processing Systems, 34: 27196–27208.

Zhao, B.; Meng, L.; Yin, W.; and Sigal, L. 2019. Image generation from layout. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8584–8593.

Zhu, J.-Y.; Park, T.; Isola, P.; and Efros, A. A. 2017. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE International Conference on Computer Vision, 2223–2232.

### Appendix

#### Comparison with Prior Work

We compare our method with previous text-to-image generation methods with layout control on traditional metrics, including mask-conditioned methods DenseDiffusion, and box-conditioned methods BoxDiff and Backward Guidance, in which DenseDiffusion, BoxDiff and Backward Guidance are all training-free methods.

Method FID(↓) CLIP-score(↑)

BoxDiff 71.73 30.03 Backward Guidance 69.04 30.76 DenseDiffusion 74.70 30.34

Ours 68.53 30.78

- Table 3: Comparison with prior works on traditional metrics.

The examples as shown in Figure 11. In our implementation, ControlNet does not support the categories “dog”, “frisbee”, “umbrella”, “elephant” and “snowboard”, so we employee the superclass “animal” to replace “dog” and “elephant”, and do not control the “frisbee”, “umbrella” and “snowboard”. In contrast, our training-free method can adapt to any semantic input. And more examples as shown in Figure 16. We remove ControlNet in Figure 16 due to it cannot support most of semantic categories.

λ → 0 1 5 10 20 100

DTL(↑) 0.0119 0.0124 0.0137 0.0156 0.0158 0.0096 FID(↓) 64.68 65.46 66.39 68.53 72.80 129.91

- Table 4: Ablation study on the effect of the hyperparameter λ. The best performance is achieved when λ is around 10.

#### The Effect of Additional Conditions

We compare our trajectory-based method with pretrained Stable Diffusion model, as shown in Figure 9, we observe that the Stable Diffusion model often struggles when generating multiple targets. However, by incorporating additional control conditions, our approach successfully achieves the intended targets. And the examples of failed cases as shown in Figure 10.

Attention map Generated image

- (a)
- (b)

[Figure 81]

[Figure 82]

[Figure 83]

Prompt: “A pikachu is playing a basketball on grass.”

Condition Stable Diffusion Ours

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Prompt: “A hello kitty toy is next to a purple ball.”

Extracted & Moved mask

with Control

Attention map Generated image

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Prompt: “A monkey stands on an airplane.”

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Prompt: “A pekingese is on the beach near a bottle.”

[Figure 101]

[Figure 102]

[Figure 103]

Extracted & Moved mask with Control

- Figure 8: Examples of images generated based on Prior Structure based Guidance. Example (a) shows that over fine mask leads to the generated “pikachu” with three ears; and (b) shows that unusable masks are obtained when the pre-trained stable diffusion model generates the poor image. In each example, the top line is the generated image from the pre-trained stable diffusion model with related attention maps, the bottom line is the result based on the trajectory-conditioned Prior Structure based Guidance and related masks through applying the threshold on the attention map and moving to the given trajectory.

Prompt: “The dog is walking on the city streets with an umbrella.”

- Figure 9: Comparing with pretrained Stable Diffusion model. Our method can guide Stable Diffusion model to generate multiple targets, despite the inherent limitations of the Stable Diffusion model in this regard.

Input Trajectory Stable Diffusion Ours

Prompt: “A person sitting on a bench next to the sea with boat.”

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Prompt: “A cat playing with a dog under chair and desk.”

- Figure 10: The examples of failed cases. Our approach fails in controlling more targets, which may be related to the intrinsic mechanism of the stable diffusion model.

#### Is the trajectory similar to scribble?

We compare our trajectory-based method with ControlNet Scribble, as shown in Figure 13, the ControlNet with scribble essentially remains a mask-based method, as it cannot be effectively controlled using overly simplistic scribbles.

We also compare the recently proposed InstanceDiffusion. InstanceDiffusion is essentially a point-based method, and we observe that its scribble input supports a maximum of 20 points. Therefore, we randomly sample 20 points along the trajectory to serve as its input. As shown in Figure 14, InstanceDiffusion generates targets that are not aligned with the given scribble points.

#### The Effect of Different Random Seeds

We validate the impact of different random seeds on the outcomes of our method, as shown in Figure 15, our method can reliably achieve control over the targets.

Dense Diffusion

Backward Guidance Ours

Input Trajectory

Input Mask Input Box BoxDiff

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

- (a)
- (b)
- (c)
- (d)

[Figure 117]

[Figure 118]

[Figure 119]

ControlNet

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Prompt: “a person is going down a hill on a snowboard.”

[Figure 127]

- (e)

Prompt: “A dog returns a blue frisbee that was thrown on a beach.”

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Prompt: “A person walking down the street with an umbrella over their head.”

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Prompt: “A airplane carrying a smaller aircraft flies in the air.”

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Prompt: “An elephant gather food from a tree.”

Figure 11: Qualitative comparison with prior mask-based and box-based layout control works. The controlled targets are colored with green and orange. The mask-based and box-based layout control methods generate the unnatural images due to the control conditions that are too fine or too coarse. However, our simple trajectory-based approach yields more natural results.

##### λ -> 0 1 5 10 20 100 200

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

Prompt: “A man is walking a dog.”

Figure 12: Qualitative analysis the effect of the different λ. The values in the range of 5-10 yielded the best results.

Ours ControlNet_Scribble

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Prompt: “A cute dog standing in a forest at autumn, high quality, professional photo.”

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Prompt: “a person is going down a hill on a snowboard.”

- Figure 13: Comparing with ControlNet Scribble(middle and right). We observe that ControlNet with scribble essentially remains a mask-based method, as it cannot be effectively controlled using overly simplistic scribbles.

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Prompt: “A bear is on the grass at spring.”

[Figure 182]

Prompt: “A cute dog standing in a forest at autumn, high quality, professional photo.”

Ours InstanceDiffusion

[Figure 183]

[Figure 184]

- Figure 14: Comparing with InstanceDiffusion Scribble (right). We observe that InstanceDiffusion with scribble essentially remains a point-based method, it fails to align the generated targets with the provided scribble points.

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Prompt: “There is a rabbit and a carrot.”

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Prompt: “There is a dog and a banana.”

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

Prompt: “There is a cat and a towel.”

Figure 15: Examples with different random seeds. Our method can reliably achieve control over the targets.

Backward Input Mask Input Box Guidance BoxDiff Ours

Dense Diffusion

Input Trajectory

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

- (a)
- (b)
- (c)
- (d)

Prompt: “A white dog laying on his side in a window.”

- (e)

Prompt: “Many people and some train at a station.”

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

Prompt: “Four woman stand on the beach with their umbrella.”

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Prompt: “A couple of zebra are standing behind a fence.”

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

Prompt: “A person holding a bear cub on a leash.”

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Figure 16: More examples of comparing with prior works.

