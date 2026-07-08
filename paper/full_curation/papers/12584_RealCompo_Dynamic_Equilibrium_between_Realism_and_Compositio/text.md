# arXiv:2402.12908v3[cs.CV]14Oct2024

## RealCompo: Balancing Realism and Compositionality Improves Text-to-Image Diffusion Models

Xinchen Zhang1∗ Ling Yang2∗† Yaqi Cai3 Zhaochen Yu2 Kai-Ni Wang4 Jiake Xie5 Ye Tian2 Minkai Xu6 Yong Tang5 Yujiu Yang1 Bin Cui2

1Tsinghua University 2 Peking University 3 University of Science and Technology of China 4 Southeast University 5 PicUp.AI 6 Stanford University https://github.com/YangLing0818/RealCompo

### Abstract

Diffusion models have achieved remarkable advancements in text-to-image generation. However, existing models still have many difficulties when faced with multiple-object compositional generation. In this paper, we propose RealCompo, a new training-free and transferred-friendly text-to-image generation framework, which aims to leverage the respective advantages of text-to-image models and spatial-aware image diffusion models (e.g., layout, keypoints and segmentation maps) to enhance both realism and compositionality of the generated images. An intuitive and novel balancer is proposed to dynamically balance the strengths of the two models in denoising process, allowing plug-and-play use of any model without extra training. Extensive experiments show that our RealCompo consistently outperforms state-of-the-art text-to-image models and spatial-aware image diffusion models in multiple-object compositional generation while keeping satisfactory realism and compositionality of the generated images. Notably, our RealCompo can be seamlessly extended with a wide range of spatial-aware image diffusion models and stylized diffusion models.

### 1 Introduction

The field of diffusion models has witnessed exciting developments and significant advancements recently[63, 45, 18, 44, 39]. Among various generative tasks, text-to-image (T2I) generation [32, 19, 62] has gained considerable interest within the community. T2I diffusion models such as Stable Diffusion [40], Imagen [41] and DALL-E 2/3 [38, 4] have exhibited powerful capabilities in generating images with high aesthetic quality and realism [4, 35]. However, they often struggle to align accurately with the compositional prompt when it involves multiple objects or complex relationships [27, 3, 33], which requires the model to have strong spatial-aware ability.

One potential solution to optimize the compositionality of generated images is providing a spatialaware condition to control diffusion models [11, 64, 56], such as layout/boxes [34, 13], keypoint/pose [69] and segmentation map [21]. These spatial-aware conditions are fundamentally similar in functioning, thus we mainly focus our analysis on layout-to-image (L2I) models for simplicity. With the control of layout, L2I models [26, 7, 57] improve compositionality by generating objects at specified locations. For instance, GLIGEN [26] designs trainable gated self-attention layers to incorporate layout input and controls the strength of its incorporation by changing parameter β. Although L2I models improve the weaknesses of compositional text-to-image generation, their generated images exhibit a significant decline in realism compared to T2I models [26, 73].

∗Contributed equally. †Corresponding authors: yangling0818@163.com, bin.cui@pku.edu.cn.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

###### Prompt: On a wooden table, there is a white vase with a pink flower inserted in it, and a cat is sitting on the table to the right of the vase.

| |[Figure 1]<br><br>|
|---|---|
| | |
| | |

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

flower window

cat

vase

- (a)
- (b)

|cup|
|---|

table

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

flower window

cat

vase

|cup|
|---|

table

(c)

- Figure 1: Motivations of RealCompo. (a) and (c) The realism and aesthetic quality of generated images become poor as more layout is incorporated. (b) Even if layout is incorporated only in the early denoising stages, the control of text alone still fails to alleviate the poor realism issue.

We conducted experiments to analyze why a significant decrease in image realism exists. We analyze the layout injection mechanism in GLIGEN [26] by controlling the density of layout through parameter β. As shown in Fig. 1 (a) and (c), our experiments indicate that the density of layout directly influences the realism of generated images. As the control of layout gradually increases, the generated images become less aesthetic and more unstable. This demonstrates that layout and text, as different control conditions, guide the model towards different generation directions, with the former emphasizing compositionality and the latter emphasizing realism. To alleviate this issue, some models [27, 26] leverage the early-stage localization capability of diffusion models [68, 48] and incorporate layouts only during the initial denoising phase. In the later denoising stage, only use text to balance image realism. However, we found this approach yielded minimal effectiveness. We assumed β = 1 in the first t denoising steps and β = 0 in the subsequent denoising steps. As shown in Fig. 1 (b), the object’s position is already determined around 20 steps. However, it is common that the generated images exhibit almost no difference between t = 20 and t = 50. This suggests that even when the injection of layout is stopped in the later denoising stages, the control of text alone still fails to alleviate the poor realism issue. The trade-off between realism and compositionality in T2I and L2I models is challenging yet necessary.

To this end, we introduce a general training-free and transferred-friendly text-to-image generation framework RealCompo, which utilizes a novel balancer to achieve dynamic equilibrium between realism and compositionality in generated images. We first utilize LLMs to generate scene layouts from text prompt through in-context learning [31]. Then we propose an innovative balancer to dynamically compose pre-trained fidelity-aware (T2I, stylized T2I) and spatial-aware (e.g., layout, keypoint, segmentation map) image diffusion models. This balancer automatically adjusts the coefficient of the predicted noise for each model by analyzing their cross-attention maps during the denoising stage. By combining the respective strengths of the two models, it achieves a trade-off between realism and compositionality. Finally, we extend RealCompo to various spatial-aware conditions through a general compositional denoising process. Moreover, by changing the T2I model to a stylized T2I model, Realcompo can seamlessly achieve compositional generation specified with a particular style. These dramatically demonstrate the great generalization ability of RealCompo. Although there exist methods [59, 2] for composing multiple diffusion models, their application lacks flexibility because they require additional training and cannot be generalized to other conditionss and models. Our method effectively composes two models in a training-free manner, allowing for a seamless transition between various models.

To the best of our knowledge, RealCompo effectively achieves a trade-off between realism and compositionality in text-to-image generation. Choosing one (stylized) T2I model and one spatial-aware (e.g., layout, keypoint, segmentation map) image diffusion model, RealCompo automatically balances their fidelity and spatial-awareness to realize a collaborative generation. We believe RealCompo opens up a new research perspective in controllable and compositional image generation.

Our main contributions are summarized as the following:

- • We introduce a new training-free and transferred-friendly text-to-image generation framework RealCompo, which enhances compositional text-to-image generation by balancing the realism and compositionality of generated images.
- • We design a novel balancer to dynamically combine the predict noise from T2I model and spatial-aware (e.g., layout, keypoint, segmentation map) image diffusion model.

- • RealCompo has strong flexibility, can be generalized to balance various (stylized) T2I models and spatial-aware image diffusion models and can achieve high-quality compositional stylized generation. It provides a fresh perspective for compositional image generation.
- • Extensive qualitative and quantitative comparisons with previous outstanding methods demonstrate that RealCompo has significantly improved the performance in generating multiple objects and complex relationships.

### 2 Related Work

Text-to-Image Generation In recent years, the field of text-to-image generation has made remarkable progress [46, 58, 35, 17, 10, 70, 61], largely attributed to breakthroughs in diffusion models. By training on large-scale image-text paired datasets, T2I models such as Stable Diffusion (SD) [40], DALL-E 2/3 [38, 4], MDM [16], and Pixart-α [6], have demonstrated remarkable generative capabilities. However, there is still significant room for improvement in compositional generation when text prompts include multiple objects and complex relationships [56]. Many studies have attempted to address this issue through controllable generation [69] by providing additional conditions such as segmentation map [21], scene graph [60], layout [72], etc., to constrain the model’s generative direction to ensure the accuracy of the number and position of objects in the generated images. However, due to the constraints of the additional conditions, image realism may decrease [26]. Furthermore, several works [36, 8, 66, 63, 29] have attempted to bridge the language understanding gap in models by pre-processing prompts with Large Language Models (LLMs) [1, 47]. It is challenging for T2I models to achieve trade-off between realism and compositionality [63] of generated images.

Compositional Text-to-Image Generation Recently, numerous methods have been introduced to improve compositional text-to-image generation [51, 73, 67, 53, 24, 28]. These methods enhance diffusion models in attribute binding, object relationship, numeracy, and complex prompts. Recent studies can generally be divided into two types [50]: one primarily uses cross-attention maps for compositional generation [30, 23, 71], while the other provides more conditions (e.g., layout, keypoint, segmentation map) to achieve controllable generation [15, 73]. The first methods delve into a detailed analysis of cross-attention maps, particularly emphasizing their correspondence with the text prompt. Attend-and-Excite [5] dynamically intervenes in the generation process to improve the model’s generation results in terms of attribute binding (such as color). Most of the second methods offer layout as a constraint, enabling the model to generate images that meet this condition. This approach directly defines the area where objects are located, making it more straightforward and observable compared to the first type of methods [26]. LMD [27] provides an additional layout as input with LLMs. Afterward, a controller is designed to predict the masked latent for each object’s bounding box and combine them in the denoising process. However, these algorithms are unsatisfactory in the realism of generated images. A recent powerful framework RPG [63] utilizes Multimodal LLMs to decompose complex generation tasks into simpler subtasks to obtain satisfactory realism and compositionality of generated images. Orthogonal to this work, we achieve dynamic equilibrium between realism and compositionality by combining T2I and spatial-aware image diffusion models.

### 3 Method

In this section, we introduce our method, RealCompo, which designs a novel balancer to achieve dynamic equilibrium between realism and compositionality of generated images. We initially focus on the layout-to-image models. In Section 3.1, we analyze the necessity of incorporating influence for the predictive noise of each model and provide a method for calculating coefficients. In Section 3.2, we provide a detailed explanation of the update rules employed by the balancer, which utilizes a training-free approach to update coefficients dynamically. In Section 3.3, we provide a universal formula and denoising procedure that enable the balance of T2I models with any spatial-aware image diffusion model, such as keypoint or segmentation-to-image models based on ControlNet [69]. We also extend RealCompo to stylized compositional generation by stylized T2I models.

#### 3.1 Combination of Fidelity and Spatial-Awareness

LLM-based Layout Generation. Since spatial-aware conditions are similar essentially, we first choose layout as the representative of spatial-aware condition for introduction. As shown in Fig.

Prompt: A woman with long golden hair is sitting on the sofa. In front of her is a round stone table, on which from left to right are a burning candle and an orange cat that has been sitting there.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Fidelity

Realism

Text

[Figure 19]

[Figure 20]

[Figure 21]

LLMs

(Stylized) Text-to-Image: SD v1.5, SD v2.1, SDXL …. Stylized Models…

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

| | |
|---|---|
| | |

a woman with long golden hair

Layout Keypoint -to-Image: Segmentation GLIGEN, LayGuide, ControlNet …

an orange cat

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

| | | | |
|---|---|---|---|
| | | | |

[Figure 35]

a sofa

a burning candle

[Figure 36]

a stone round table

Spatial-aware

Compositionality

Text

Layout

- Figure 2: An overview of RealCompo framework for text-to-image generation. We first use LLMs or transfer function to obtain the corresponding layout. Next, the balancer dynamically updates the influence of two models, which enhances realism by focusing on contours and colors in the fidelity branch, and improves compositionality by manipulating object positions in the spatial-aware branch.

2, we leverage the powerful in-context learning [55] capability of Large Language Models (LLMs) to analyze the input text prompt and generate an accurate layout to achieve "pre-binding" between objects and attributes. The layout is then used as input for the L2I model. In this paper, we choose GPT-4 for layout generation. Please refer to Appendix B.1 for detailed explanation.

Combination of Two Types of Noise. In diffusion models, the model’s predicted noise ϵt directly affects the direction of the generated images. In T2I models, ϵtextt exhibits more directive toward realism [40], whereas in L2I models, ϵlayoutt demonstrates more directive toward compositionality [26]. To achieve the trade-off between realism and compositionality, a feasible but untapped solution is to compose the predicted noise of two models. However, the predicted noise from different models has its own generative direction, contributing differently to the generated results at different timesteps and positions. Based on this, we design a novel balancer that achieves dynamic equilibrium between the two models’ strengths at every position i in the noise for timestep t. This is achieved by analyzing the influence of each model’s predicted noise. Specifically, we first set the same coefficient for the predicted noise of each model to represent their influence before the first denoising step:

#### CoetextT = CoelayoutT ∼ N(0,I) (1)

In order to regularize the influence of each model, we perform a softmax operation on the coefficients to get the final coefficients:

exp(Coect) exp(Coetextt ) + exp(Coelayoutt )

ξct =

where c ∈ {text,layout}. The balanced noise can be derived according to the coefficient of each model:

(2)

ϵt = ξtextt ⊙ ϵtextt + ξlayoutt ⊙ ϵlayoutt (3) where ⊙ denotes pixel-wise multiplication.

Once the predicted noise ϵct and the coefficient Coect of each model are provided, the balanced noise can be derived from Eq. 2 and Eq. 3. At timestep t, the balancer dynamically updates coefficients as described in Section 3.2.

#### 3.2 Influence Estimation with Dynamic Balancer

The alignment between the generated images and the input prompts is largely influenced by model’s cross-attention maps, which encapsulate a wealth of matching information between visual and textual

elements, such as location and shape. Specifically, given the intermediate feature φ(zt) and the text embeddings τθ(y), cross-attention maps can be derived in the following manner:

Qc(Kc)T dck

Ac = Softmax

,c ∈ {text,layout} (4)

Q = WQ · φ(zt), K = WK · τθ(y) (5) where Q and K are respectively the dot product results of the intermediate feature φ(zt), text embeddings τθ(y), and two learnable matrices WQ and WK. Aij defines the weight of the value of the j-th token on the i-th pixel. Here, j ∈ {1,2,...,N(τθ(y))}, and N(τθ(y)) denotes the number of tokens in τθ(y). The dimension of K is represented by dk.

Update Rule of Dynamic Balancer. We designed a novel balancer that dynamically balances two models according to their cross-attention maps at timestep t. Specifically, we represent layout as B = {b1,b2,...,bv}, which is composed of v bounding boxes b. Each bounding box b corresponds to a binary mask Mb, where the value inside the box is 1 and the value outside the box is 0. Given the predicted noise ϵct and the coefficient Coect of each model, the balanced noise ϵt and denoised latent zt−1 can be derived from Eq. 3 and Eq. 12. By feeding zt−1 into two models, we obtain the cross-attention maps Act−1 output by the two models at timestep t − 1, which indicates the denoising quality feedback after the noise ϵct of the model at time t is weighted by ξct. Based on Act−1, we define the loss function as follows:

Ac(ijb,t−1) ⊙ Mb i Ac(ij

L(Atextt−1,Alayoutt−1 ) =

1− i

(6)

b,t−1)

c b

where c ∈ {text,layout}, jb denotes the token corresponding to the object in bounding box b. Since two models are controlled by different conditions, averaging the predicted noise equally will lead to instability in the generated images. This is because the T2I model breaks the layout constraints of the L2I model, reducing the compositionality of the generated images, as we have demonstrated in experimrnts in Fig. 8. Therefore, we designed this loss function to measure the alignment between the cross-attention maps and layout for each model. A smaller loss indicates better compositionality. The following rule is used to update Coect:

L(Atextt−1,Alayoutt−1 ) (7) where ρt is the updating rate. This update rule continuously strengthens the constraints on both models by assessing the positional alignment of the layout within the cross-attention maps, ensuring the maintenance of the localization capability of L2I model while injecting fidelity information of T2I model. It is worth noting that previous methods [5, 57, 27] for parameter updates based on function gradients were primarily using energy functions to update latent zt. We are the first to update the influence of predicted noise based on the gradient of the loss function, which is a novel and stable method well-suited to our task. The complete denoising process is detailed in Appendix B.3.

#### Coect = Coect − ρt∇Coec

t

#### 3.3 Extend RealCompo to any Spatial-Aware Conditions in a General Form

Other spatial-aware text-to-image diffusion models are essentially similar to L2I models. Keypoint-to-image (K2I) models generate specified actions or poses within each group of keypoints region, and segmentation-to-image (S2I) models fill indicated objects within each segmented region. The concept of "region" is always present, which transforms T2I generation from a macro perspective to utilizing regionbased control for T2I generation from a micro perspective. This concept is also the core of enhancing image compositionality. Compared with layout-based T2I generation, the only difference is that keypoints and segmentation maps have stronger control over the model based on regions, requiring that the pose is maintained and the object is correct and unique.

Fidelity Spatial-aware

zero convolution

neural network block (locked)

neural network block (locked)

neural network copy (locked)

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

zero convolution

RealCompo

ControlNet

Figure 3: RealCompo constructed on ControlNet.

Keypoint SDXL ControlNet RealCompo(Ours)

Segmentation map SD v2.1 ControlNet RealCompo(Ours)

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

A girl with an angelic appearance dances in a white dress as dark clouds gather

A house with dark red walls surrounded by flowers and trees.

in the sky, creating a somber atmosphere.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

A group of four friends, two girls and two boys wearing casual clothing, laughing together as they walk along a flower-lined path beneath a cherry blossom tree in full bloom.

In the living room there are two white sofas and a white coffee table, and the living room has two oversized floor-to-ceiling windows.

Figure 4: Extend RealCompo to keypoint- and segmentation-based image generation.

General Form for Extension to Other Spatial-Aware Conditions We rethink Eq. 6, which is RealCompo’s core approach in combining T2I and L2I models, where the only layout-related variable is the binary masks M. Considering that spatial-aware controllable T2I generation inherently focus on the concept of "region control", we introduce a transfer function:

M = f(C) (8)

where C represents other spatial-aware conditions such as keypoint and segmentation map. f(·) represents the calculation of the minimum and maximum values of the horizontal and vertical coordinates occupied by each set of keypoints or a segmentation block within the entire image coordinate system, which can be transformed into a layout and a binary mask M. Therefore, for any T2I models with spatial-aware control, the general loss function of RealCompo is:

Ac(ijb,t−1) ⊙ fb(C) i Ac(ij

L(Atextt−1,Aspatialt−1 ) =

1− i

(9)

b,t−1)

c b

where c ∈ {text,spatial}. Similarly, Coect is dynamically updated using Eq. 7. ControlNet [69] enables controllable T2I generation based on various spatial-aware conditions. In this work, the spatial-aware branches besides layout are all based on ControlNet, which is illustrated in Fig. 3. The generated images of keypoint- and segmentation-based RealCompo are shown in Fig. 4.

Extend RealCompo to Stylized Image Generation As an essential indicator of fidelity, image style [49, 65] guides us to expand the application potential of RealCompo. Since RealCompo mainly leverages T2I models to enhance and guide the realism and aesthetic quality of generated images. By replacing the T2I model with various stylized T2I models and combining it with a spatial-aware image diffusion model, we can achieve outstanding compositional generation under this style. The experiments are shown in Fig 7.

### 4 Experiments

#### 4.1 Experimental Setup

Implementation Details Our RealCompo is a generic, scalable framework that can achieve the complementary advantages of the model with any chosen (stylized) T2I models and spatial-aware image diffusion models. We selected GPT-4 [1] as the layout generator in our experiments, the detailed rules are described in Appendix B.1. For layout-based RealCompo, we chose SD v1.5 [40] and GLIGEN [26] as the backbone. For keypoint-based RealCompo, we chose SDXL [4] and ControlNet [69] as the backbone. For segmentation-based RealCompo, we chose SD v2.1 [40] and ControlNet [69] as the backbone. For style-based RealCompo, we chose two stylized T2I models: Coloring Page Diffusion and CuteYukiMix as the backbone, and chose GLIGEN [26] as the backbone of L2I model. All of our experiments are conducted under 1 NVIDIA 80G-A100 GPU.

Baselines and Benchmark To evaluate compositionality, we compare our RealCompo with the outstanding T2I and L2I models on T2I-CompBench [20]. This benchmark test models across aspects

Table 1: Evaluation results about compositionality on T2I-CompBench [20]. RealCompo consistently demonstrates the best performance regarding attribute binding, object relationships, numeracy and complex compositions. We denote the best score in blue , and the second-best score in green . The baseline data is quoted from PixArt-α [6].

Attribute Binding Object Relationship

Model

Numeracy↑ Complex↑ Color ↑ Shape↑ Texture↑ Spatial↑ Non-Spatial↑

- Stable Diffusion v1.4 [40] 0.3765 0.3576 0.4156 0.1246 0.3079 0.4461 0.3080 Stable Diffusion v2 [40] 0.5065 0.4221 0.4922 0.1342 0.3096 0.4579 0.3386 Structured Diffusion [12] 0.4990 0.4218 0.4900 0.1386 0.3111 0.4550 0.3355 Attn-Exct v2 [5] 0.6400 0.4517 0.5963 0.1455 0.3109 0.4767 0.3401 DALL-E 2 [38] 0.5750 0.5464 0.6374 0.1283 0.3043 0.4873 0.3696 Stable Diffusion XL [4] 0.6369 0.5408 0.5637 0.2032 0.3110 0.4988 0.4091 PixArt-α [6] 0.6886 0.5582 0.7044 0.2082 0.3179 0.5058 0.4117 GLIGEN[26] 0.4288 0.3998 0.3904 0.2632 0.3036 0.4970 0.3420 LMD+[27] 0.4814 0.4865 0.5699 0.2537 0.2828 0.5762 0.3323 RealCompo (Ours) 0.7741 0.6032 0.7427 0.3173 0.3294 0.6592 0.4657

of attribute binding, object relationship, numeracy and complexity. To evaluate realism, we randomly select 3K text prompts from the COCO validation set , we utilize ViT-B-32 [9] to calculate the CLIP score and LAION aesthetic predictor to calculate aesthetic score, reflecting the degree of match between generated images and prompts as well as the aesthetic quality, respectively. In addition to objective evaluations, we conducted a user study to evaluate RealCompo and stylized RealCompo in terms of realism, compositionality, and comprehensive evaluation.

4.2 Main Results

Results of Compositionality: T2I-CompBench We conducted tests on T2I-CompBench [20] to evaluate the compositionality of RealCompo compared to the outstanding T2I and L2I models. As demonstrated in Table 1, RealCompo achieved state-of-the-art performance on all seven evaluation tasks. It is clear that RealCompo and L2I models GLIGEN [26] and LMD+ [27] show significant improvements in spatial-aware tasks such as spatial and numeracy. These improvements are largely attributed to the guidance provided by the additional conditions, which greatly enhances the model’s compositional performance. RealCompo employs a balancer for better control over positioning, boosting its advantages in these aspects. However, the L2I models exhibit a noticeable decline in performance on tasks like texture and non-spatial. This decline is due to the injection of layout embeddings, which dilute the density of text embeddings, leading to suboptimal semantic understanding by the model. By composing additional T2I models, RealCompo provides sufficient textual information during the denoising process and achieves outstanding results in tasks that reflect realism, such as texture, non-spatial and complex tasks. As shown in Fig. 5, compared with the current outstanding

mountains

trees trees

water

A man and his wife is holding a dog.

[Figure 57]

a man

a dog

wife

[Figure 58]

[Figure 59]

Landscape with trees growing on both sides of a small river, and many snowy mountains in the distance.

[Figure 60]

Layout SD v1.5 GLIGEN LMD+ RealCompo(Ours)

[Figure 61]

On a wooden table sits a yellow vase adorned with yellow, white, and

purple flowers.

|purple f| | | |
|---|---|---|---|
| |white<br><br>lowers| | |
|yellow| |fl|ower|

flowers

a wooden table

a yellow vase

pu

s

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

A rabbit wearing sunglasses is sunbathing on a beach by the lake.

bench

sunglasses a rabbit

a lake

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Three cars are parked in front of two houses.

[Figure 77]

[Figure 78]

[Figure 79]

A person is looking at a stained-glass window and marveling at its beauty.

[Figure 80]

|||person|
|---|
<br><br>a stainedglass window<br><br>a|
|---|
|
|---|

a house a house

a car a car a car

Layout SD v1.5 GLIGEN LMD+ RealCompo(Ours)

Figure 5: Qualitative comparison between our RealCompo and the outstanding text-to-image model

- Stable Diffusion v1.5 [40], as well as the layout-to-image models, GLIGEN [26] and LMD+ [27]. Colored text denotes the advantages of RealCompo in generated images.

|T2I model Spatial-aware model RealCompo<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

Stylized T2I model Stylized L2I model RealCompo

60.0%

| |
|---|

60.0%

| |
|---|

Percentage

Percentage

40.0%

40.0%

20.0%

20.0%

0.0%

0.0%

Realism Compositionality Comprehensive Evaluation

Realism (Style Preservation)

Compositionality Comprehensive Evaluation

Figure 6: Results of user study.

L2I models GLIGEN and LMD+, RealCompo achieves a high level of realism while keeping the attributes of the objects matched and the number of positions generated correctly.

Results of Realism: Quantitative Comparison and User Study As shown in Table 2, our model significantly outperforms existing outstanding T2I and L2I models in both CLIP score and aesthetic score. We attribute this to the dynamic balancer, which enhances image realism and aesthetic quality while maintaining high compositionality. In addition to objective evaluations, we designed a user study to subjectively assess the practical performance of various methods. We randomly selected 15 prompts, including 5 for stylization experiments. Comparative tests were conducted using T2I models, spatial-aware image diffusion models, and RealCompo. We invited 39 users from diverse backgrounds to vote on image realism, image compositionality, and comprehensive evaluation, resulting in a total of 1755 votes. As illustrated in Fig. 6, RealCompo received widespread user approval in terms of realism and compositionality.

Table 2: Evaluation results on image realism.

Model CLIP Score↑ Aesthetic Score↑

Stable Diffusion v1.4 [40] 0.307 5.326 TokenCompose v2.1 [52] 0.323 5.067 Stable Diffusion v2.1 [40] 0.321 5.458 Stable Diffusion XL [4] 0.322 5.531

Layout Guidance[7] 0.294 4.947 GLIGEN[26] 0.301 4.892 LMD+[27] 0.298 4.964

RealCompo (Ours) 0.334 5.742

Results of Extend Applications: More Spatial-Aware Conditions We extend RealCompo to more spatial-aware controlled image generation. As shown in Fig. 4, keypoint- and segmentationbased RealCompo achieves outstanding performance in both realism and compositionality. This promising result reveals that as spatial-aware conditions, layout, keypoint, and segmentation map are fundamentally similar, RealCompo focuses on these similarities and achieves a general generative paradigm for compositional generation.

Results of Extend Applications: Stylized Generation Image style is an essential indicator of fidelity. We experiment with generalizing RealCompo to various pre-trained stylized T2I models. We

InstantStyle LMD

###### RealCompo(Ours)

Layout

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

a house

Style: Coloring

Page Diffusion (SD v1.5)

a car

Coloring page of a car park in front of a house.

red moon

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

a vampire girl

Style: CuteYukiMix (SD v1.5)

black wing black coat

A vampire girl with black wing and black coat, red moon, castle-background.

Figure 7: Extend RealCompo to stylized compositional generation.

RealCompo v1

RealCompo v4

RealCompo v2

RealCompo

RealCompo v3

Layout

LMD+

SD v1.5+GLIGEN

TokenCompose+LayGuide

TokenCompose+GLIGEN

w/o Dynamic Balancer

SD v1.5+LayGuide

|a cute small corgi<br><br>a movie|a cute small corgi<br><br>theater|
|---|---|
|a popcorn|a popcorn|

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Two cute small corgi sitting in a movie theater with two popcorns in front of them.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

a lighthouse

a cruise ship

sea

A cruise ship at sea and a lighthouse on the right.

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

|a sunflower a sunflower<br><br>a blue flowerpot<br><br>a butterfly|
|---|

Two sunflowers are growing in a blue flowerpot, and a butterfly dances gracefully.

- Figure 8: Ablation study on the significance of the dynamic balancer and qualitative comparison of RealCompo’s generalization to different models. We demonstrate that dynamic balancer is important to compositional generation and RealCompo has strong generalization and generality to different models, achieving a remarkable level of both fidelity and precision in aligning with text prompts.

selected the Coloring Page Diffusion and Cutyukimix as the foundational stylized models, focusing on the coloring page style and adorable style, respectively. As shown in Fig. 7, RealCompo perfectly inherits the style of the T2I models and, with the help of L2I model, achieves powerful compositional generation under these styles, which is currently difficult for stylized diffusion models to accomplish. We found it difficult for LMD to strictly maintain the style by simply replacing the backbone with a stylized model, often leading to text leakage [12]. For example, terms like "crayon" frequently appear in the coloring page style, indicating that the layout control disrupts the style or text control, making it challenging for L2I models to achieve stylized compositional generation. In contrast, by maintaining image realism and style, RealCompo demonstrates strong compositionality while better preserving the style compared to currently outstanding stylized models like InstantStyle [49].

#### 4.3 Ablation Study

Importance of Dynamic Balancer As shown in Fig. 8, we conducted experiments on the importance of the dynamic balancer. It is clear that without the use of the dynamic balancer, the generated images do not align with the layout. This is because the predicted noise in T2I model is not constrained by the layout, leading to the model generating the object at any position, and the quantity is uncontrollable. Although the image realism is high, the predicted noise of T2I model disrupts the object distribution of the predicted noise of L2I model, leading to poor compositionality of the generated images and uncontrollable in the generation process.

Generalizing to Different Backbones To explore the generalizability of RealCompo for various models, we choose two T2I models, SD v1.5 [40] and TokenCompose [52], and two L2I models, GLIGEN [26] and LayGuide (Layout Guidance) [7]. We combine them two by two, yielding four versions of RealCompo v1-v4. The experimental results are shown in Fig. 8. The four versions of RealCompo all have a high degree of realism in generating images and achieving desirable results regarding instance composition. This is attributed to the dynamic balancer combining the strengths of T2I and L2I models, and it can seamlessly switch between models because it is simple and requires no training. We also found that RealCompo, when using GLIGEN as the L2I model, performs better than when using LayGuide in generating objects that match the layout. For instance, in the images generated by RealCompo v4 in the first and third rows, "popcorns" and "sunflowers" do not fill up the bounding box, which can be attributed to the superior performance of the base model GLIGEN compared to LayGuide. Therefore, when combined with more powerful T2I and L2I models, RealCompo is expected to yield more satisfactory results.

### 5 Conclusion

In this paper, to solve the challenge of complex or compositional text-to-image generation, we propose the SOTA training-free and transferred-friendly framework RealCompo. In RealCompo, we propose a novel balancer that dynamically combines the advantages of various (stylized) T2I and spatial-aware (e.g., layout, keypoint, segmentation map) image diffusion models to achieve the trade-off between realism and compositionality in generated images. In future work, we will continue to improve this framework by using a more powerful backbone and extend it to more realistic applications.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.
- [3] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. arXiv preprint arXiv:2302.08113, 2023.
- [4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2:3, 2023.
- [5] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attentionbased semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42

(4):1–10, 2023.

- [6] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [7] Minghao Chen, Iro Laina, and Andrea Vedaldi. Training-free layout control with cross-attention guidance. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 5343–5353, 2024.
- [8] Xiaohui Chen, Yongfei Liu, Yingxiang Yang, Jianbo Yuan, Quanzeng You, Li-Ping Liu, and Hongxia Yang. Reason out your layout: Evoking the layout master from large language models for text-to-image synthesis. arXiv preprint arXiv:2311.17126, 2023.
- [9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
- [10] Chengbin Du, Yanxi Li, Zhongwei Qiu, and Chang Xu. Stable diffusion is unstable. Advances in Neural Information Processing Systems, 36, 2024.
- [11] Wan-Cyuan Fan, Yen-Chun Chen, DongDong Chen, Yu Cheng, Lu Yuan, and Yu-Chiang Frank Wang. Frido: Feature pyramid diffusion for complex scene image synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 579–587, 2023.
- [12] Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Reddy Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. Training-free structured diffusion guidance for compositional text-to-image synthesis. In The Eleventh International Conference on Learning Representations, 2023.
- [13] Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems, 36, 2024.
- [14] Myles Foley, Ambrish Rawat, Taesung Lee, Yufang Hou, Gabriele Picco, and Giulio Zizzo. Matching pairs: Attributing fine-tuned models to their pre-trained large language models. arXiv preprint arXiv:2306.09308, 2023.

- [15] Hanan Gani, Shariq Farooq Bhat, Muzammal Naseer, Salman Khan, and Peter Wonka. Llm blueprint: Enabling text-to-image generation with complex and detailed prompts. arXiv preprint arXiv:2310.10640, 2023.
- [16] Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Josh Susskind, and Navdeep Jaitly. Matryoshka diffusion models. arXiv preprint arXiv:2310.15111, 2023.
- [17] Yaru Hao, Zewen Chi, Li Dong, and Furu Wei. Optimizing prompts for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024.
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [19] Hexiang Hu, Kelvin CK Chan, Yu-Chuan Su, Wenhu Chen, Yandong Li, Kihyuk Sohn, Yang Zhao, Xue Ben, Boqing Gong, William Cohen, et al. Instruct-imagen: Image generation with multi-modal instruction. arXiv preprint arXiv:2401.01952, 2024.
- [20] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. arXiv preprint arXiv:2307.06350, 2023.
- [21] Ziqi Huang, Kelvin CK Chan, Yuming Jiang, and Ziwei Liu. Collaborative diffusion for multi-modal face generation and editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6080–6090, 2023.
- [22] Mehran Kazemi, Najoung Kim, Deepti Bhatia, Xin Xu, and Deepak Ramachandran. Lambada: Backward chaining for automated reasoning in natural language. arXiv preprint arXiv:2212.13894, 2022.
- [23] Yunji Kim, Jiyoung Lee, Jin-Hwa Kim, Jung-Woo Ha, and Jun-Yan Zhu. Dense text-to-image generation with attention modulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7701–7711, 2023.
- [24] Sen Li, Ruochen Wang, Cho-Jui Hsieh, Minhao Cheng, and Tianyi Zhou. Mulan: Multimodal-llm agent for progressive multi-object diffusion. arXiv preprint arXiv:2402.12741, 2024.
- [25] Xiaonan Li, Kai Lv, Hang Yan, Tianyang Lin, Wei Zhu, Yuan Ni, Guotong Xie, Xiaoling Wang, and Xipeng Qiu. Unified demonstration retriever for in-context learning. arXiv preprint arXiv:2305.04320, 2023.
- [26] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22511–22521, 2023.
- [27] Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. Llm-grounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models. arXiv preprint arXiv:2305.13655, 2023.
- [28] Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones: Concept neurons in diffusion models for customized generation. arXiv preprint arXiv:2303.05125, 2023.
- [29] Yujie Lu, Xianjun Yang, Xiujun Li, Xin Eric Wang, and William Yang Wang. Llmscore: Unveiling the power of large language models in text-to-image synthesis evaluation. Advances in Neural Information Processing Systems, 36, 2024.
- [30] Tuna Han Salih Meral, Enis Simsar, Federico Tombari, and Pinar Yanardag. Conform: Contrast is all you need for high-fidelity text-to-image diffusion models. arXiv preprint arXiv:2312.06059, 2023.
- [31] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837, 2022.
- [32] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, pp. 16784–16804. PMLR, 2022.
- [33] Geon Yeong Park, Jeongsol Kim, Beomsu Kim, Sang Wan Lee, and Jong Chul Ye. Energy-based cross attention for bayesian context update in text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

- [34] Quynh Phung, Songwei Ge, and Jia-Bin Huang. Grounded text-to-image synthesis with attention refocusing. arXiv preprint arXiv:2306.05427, 2023.
- [35] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [36] Leigang Qu, Shengqiong Wu, Hao Fei, Liqiang Nie, and Tat-Seng Chua. Layoutllm-t2i: Eliciting layout guidance from llm for text-to-image generation. In Proceedings of the 31st ACM International Conference on Multimedia, pp. 643–654, 2023.
- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.
- [38] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- [39] Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, and Gal Chechik. Linguistic binding in diffusion models: Enhancing attribute correspondence through attention map alignment. Advances in Neural Information Processing Systems, 36, 2024.
- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.
- [41] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.
- [42] Chenglei Si, Dan Friedman, Nitish Joshi, Shi Feng, Danqi Chen, and He He. Measuring inductive biases of in-context learning with underspecified demonstrations. arXiv preprint arXiv:2305.13299, 2023.
- [43] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. PMLR, 2015.
- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint

- arXiv:2010.02502, 2020.

[45] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint

- arXiv:2011.13456, 2020.

- [46] Jiao Sun, Deqing Fu, Yushi Hu, Su Wang, Royi Rassin, Da-Cheng Juan, Dana Alon, Charles Herrmann, Sjoerd van Steenkiste, Ranjay Krishna, et al. Dreamsync: Aligning text-to-image generation with image understanding feedback. arXiv preprint arXiv:2311.17946, 2023.
- [47] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [48] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for textdriven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1921–1930, 2023.
- [49] Haofan Wang, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024.
- [50] Ruichen Wang, Zekang Chen, Chen Chen, Jian Ma, Haonan Lu, and Xiaodong Lin. Compositional text-to-image synthesis with attention map control of diffusion models. arXiv preprint arXiv:2305.13921, 2023.
- [51] Xudong Wang, Trevor Darrell, Sai Saketh Rambhatla, Rohit Girdhar, and Ishan Misra. Instancediffusion: Instance-level control for image generation. arXiv preprint arXiv:2402.03290, 2024.

- [52] Zirui Wang, Zhizhou Sha, Zheng Ding, Yilin Wang, and Zhuowen Tu. Tokencompose: Grounding diffusion with token-level supervision. arXiv preprint arXiv:2312.03626, 2023.
- [53] Song Wen, Guian Fang, Renrui Zhang, Peng Gao, Hao Dong, and Dimitris Metaxas. Improving compositional text-to-image generation with large vision-language models. arXiv preprint arXiv:2310.06311, 2023.
- [54] Haibin Wu, Kai-Wei Chang, Yuan-Kuei Wu, and Hung-yi Lee. Speechgen: Unlocking the generative power of speech language models with prompts. arXiv preprint arXiv:2306.02207, 2023.
- [55] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155, 2023.
- [56] Tsung-Han Wu, Long Lian, Joseph E Gonzalez, Boyi Li, and Trevor Darrell. Self-correcting llm-controlled diffusion models. arXiv preprint arXiv:2311.16090, 2023.
- [57] Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7452–7461, 2023.
- [58] Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-toimage generation via diffusion gans. arXiv preprint arXiv:2311.09257, 2023.
- [59] Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Text-to-image generation via large mixture of diffusion paths. arXiv preprint arXiv:2305.18295, 2023.
- [60] Ling Yang, Zhilin Huang, Yang Song, Shenda Hong, Guohao Li, Wentao Zhang, Bin Cui, Bernard Ghanem, and Ming-Hsuan Yang. Diffusion-based scene graph to image generation with masked contrastive pre-training. arXiv preprint arXiv:2211.11138, 2022.
- [61] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4):1–39, 2023.
- [62] Ling Yang, Jingwei Liu, Shenda Hong, Zhilong Zhang, Zhilin Huang, Zheming Cai, Wentao Zhang, and Bin Cui. Improving diffusion-based image synthesis with context prediction. Advances in Neural Information Processing Systems, 36, 2024.
- [63] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. arXiv preprint arXiv:2401.11708, 2024.
- [64] Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Linjie Li, Kevin Lin, Chenfei Wu, Nan Duan, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Reco: Region-controlled text-to-image generation. In CVPR, 2023.
- [65] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.
- [66] YuTeng Ye, Jiale Cai, Hang Zhou, Guanwen Li, Youjia Zhang, Zikai Song, Chenxing Gao, Junqing Yu, and Wei Yang. Progressive text-to-image diffusion with soft latent direction. arXiv preprint arXiv:2309.09466, 2023.
- [67] Chun-Hsiao Yeh, Ta-Ying Cheng, He-Yen Hsieh, Chuan-En Lin, Yi Ma, Andrew Markham, Niki Trigoni, HT Kung, and Yubei Chen. Gen4gen: Generative data pipeline for generative multi-concept composition. arXiv preprint arXiv:2402.15504, 2024.
- [68] Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. Freedom: Training-free energyguided conditional diffusion model. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
- [69] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3836–3847, 2023.
- [70] Yuechen Zhang, Jinbo Xing, Eric Lo, and Jiaya Jia. Real-world image variation by aligning diffusion inversion chain. Advances in Neural Information Processing Systems, 36, 2024.

- [71] Yibo Zhao, Liang Peng, Yang Yang, Zekai Luo, Hengjia Li, Yao Chen, Wei Zhao, Wei Liu, Boxi Wu, et al. Local conditional controlling for text-to-image diffusion models. arXiv preprint arXiv:2312.08768, 2023.
- [72] Guangcong Zheng, Xianpan Zhou, Xuewei Li, Zhongang Qi, Ying Shan, and Xi Li. Layoutdiffusion: Controllable diffusion model for layout-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22490–22499, 2023.
- [73] Dewei Zhou, You Li, Fan Ma, Zongxin Yang, and Yi Yang. Migc: Multi-instance generation controller for text-to-image synthesis. arXiv preprint arXiv:2402.05408, 2024.

This supplementary material is structured into several sections that provide additional details and analysis related to our work on RealCompo. Specifically, it will cover the following topics:

- • In Appendix A, we provide a preliminary about Stable Diffusion.
- • In Appendix B.1, we provide a detailed pipeline about how to get layout through in-context learning of LLMs.
- • In Appendix B.2, we provide a detailed proof of the existence of the gradient in Eq. 7.
- • In Appendix B.3, we provide the pseudocode for RealCompo to thoroughly demonstrate its denoising process.
- • In Appendix B.4, we conduct a detailed analysis of the gradient changes of the two models in Eq. 7 during the denoising process.
- • In Appendix B.5, we analysis the limitations and future work of RealCompo.
- • In Appendix B.6, we analysis the broader impact of RealCompo.
- • In Appendix C, we provide more additional visualized results.

### A Preliminary

Diffusion models [18, 43] are probabilistic generative models. They can perform multi-step denoising on random noise xT ∼ N(0,I) to generate clean images through training. Specifically, a gaussian noise ϵ is gradually added to the clean image x0 in the forward process:

xt = √α¯tx0 + √1 − α¯tϵ (10) where ϵ ∼ N(0,I) and αt is the noise schedule. Training is performed by minimizing the squared error loss: min θ

L = Ex,ϵ∼N(0,I),t ∥ϵ − ϵθ(xt,t)∥22 (11)

The parameters of the estimated noise ϵθ are updated step by step by calculating the loss between the real noise ϵ and the estimated noise ϵθ(xt,t).

The reverse process aims to start from the noise xT, and denoise it according to the predicted noise ϵθ(xt,t) at each step. DDIM [44] is a deterministic sampler with denoising steps:

√1 − α¯tϵθ (xt,t) √α¯t

xt−1 = √α¯t−1

xt −

+ 1 − α¯t−1ϵθ (xt,t) (12)

Stable Diffusion [40] is a significant advancement in this field, which conducts noise addition and removal in the latent space. Specifically, SD uses a pre-trained autoencoder that consists of an encoder E and a decoder D. Given an image x, the encoder E maps x to the latent space, and the decoder D can reconstruct this image, i.e., z = E(x), x˜ = D(z). Moreover, Stable Diffusion supports an additional text prompt y for conditional generation. y is transformed into text embeddings τθ(y) through the pre-trained CLIP [37] text encoder. ϵθ is trained via:

L=Ez∼E(x),ϵ∼N(0,I),t ∥ϵ−ϵθ(zt,t,τθ(y))∥22 (13)

min

θ

In the inference process, noise zT ∼ N (0,I) is sampled from the latent space. By applying Eq. 12, we perform step-by-step denoising to obtain a clean latent z0. The generative image is then reconstructed through the decoder D.

### B Additional Analysis

#### B.1 LLM-based Layout Generation

Large Language Models (LLMs) have witnessed remarkable advancements in recent years [47, 22]. Due to their robust language comprehension, induction, reasoning, and summarization capabilities, LLMs have made significant strides in the Natural Language Processing (NLP) tasks [14, 54]. In the context of multiple-object compositional generation, text-to-image diffusion models exhibit a relatively weaker understanding of language, as reflected in the poor compositionality of the generated images. Consequently, exploring ways to harness the inferential and imaginative capacities of LLMs to facilitate their collaboration with text-to-image diffusion models, thereby producing images that adhere to the prompt, offers substantial research potential.

In our task, we leverage LLMs to directly infer the layout of all objects based on the user’s input prompt through in-context learning (ICL) [25, 42]. This layout is used for the layout-to-image model of RealCompo, eliminating the need to manually provide a layout for each prompt and achieve pre-binding of multiple objects and attributes. Specifically, as shown in Fig. 9, we construct prompt templates, which include descriptions of task rules (instruction), in-context examples (demonstration), and the user’s input prompt (test). Through imitation reasoning based on the instruction, LLM generate layout for each object, where each layout represents the coordinates of the top-left and bottom-right corners of a respective box. We selected the highly capable GPT-4 [1] as layout generator.

[Figure 107]

A man and his wife is holding a dog.

Embed Prompt into Templated

Instructions

You are an intelligent bounding box generator. I will provide you with a caption for a photo, image, or painting. Your task is to generate the bounding boxes for the objects mentioned in the caption, along with a background prompt describing the scene. The images are of size 1x1. The top-left corner has coordinate [0, 0]. The bottom-right corner has coordinate [1, 1]. The bounding boxes should not overlap or go beyond the image boundaries …

Demonstration

Input: A teddy bear sits next to a bird Output:

[('a teddy bear', [0.0, 0.1, 0.3, 0.7]), ('a bird', [0.5, 0.1, 1.0, 0.8])] Input : A book on the right and a bowl on the left Output :

[('a book', [0.6, 0.4, 1.0, 0.8]), ('a bowl', [0.0, 0.4, 0.4, 0.8])]

…

Test

Input: A man and his wife is holding a dog.

Request

Parsing

###### Output:

[Figure 108]

wife

a man

[(‘a man', [0.1, 0.1, 0.6, 1.0]), (‘wife', [0.5, 0.2, 0.9, 1.0]),

GPT-4 (‘a dog', [0.4, 0.45, 0.8, 1.0]]

a dog

- Figure 9: Firstly, the user’s input text is embedded into the prompt template. The template is then parsed using GPT-4 with frozen parameters, which yields descriptions of the objects in the prompt as well as their corresponding layout.

#### B.2 Analysis of the Existence of Gradient in Eq. 7

Here we set:

L(Atextt−1,Alayoutt−1 ) =

=

b

b

Lb(Atextt−1,Alayoutt−1 )

Atext(ijb,t−1) ⊙ Mb i Atext(ij

1 − i

b,t−1)

Alayout(ij

b,t−1) ⊙ Mb i Alayout(ij

+ 1 − i

b,t−1)

If the loss function is given by Eq. 6, the gradient in Eq. 7 can be derived as follows:

(14)

∂L Atextt−1,Alayoutt−1 ∂Coect

∂ b Lb Atextt−1,Alayoutt−1 ∂Coect

=

∂Lb Atextt−1,Alayoutt−1 ∂Coect

=

b

 

∂Lb Atextt−1,Alayoutt−1 ∂Ac(j

∂Ac(jb,t−1) ∂zt−1

∂zt−1 ∂ϵt

=

b,t−1)

b

  

∂Lb Atextt−1,Alayoutt−1 ∂Ac(j

∂Ac(jb,t−1) ∂zt−1

∂zt−1 ∂ϵt

=

b,t−1)

b

  

∂Lb Atextt−1,Alayoutt−1 ∂Ac(j

∂Ac(jb,t−1) ∂zt−1

∂zt−1 ∂ϵt

=

b,t−1)

b

 

∂Lb Atextt−1,Alayoutt−1 ∂Ac(j

∂Ac(jb,t−1) ∂zt−1

=

b,t−1)

b

  

#### ϵct · exp Coetextt + Coelayoutt

×

2

exp Coetextt + exp Coelayoutt

 

∂ξct ∂Coect

∂ϵt ∂ξct

  

exp Coetextt + Coelayoutt exp Coetextt + exp Coelayoutt

∂ϵt ∂ξct

2

  

#### ϵct · exp Coetextt + Coelayoutt

2

exp Coetextt + exp Coelayoutt

√1 − α¯t √αt

1 − α¯t−1 − σ2 −

(15)

For any T2I and L2I models, we have the following:

∂Lb Atextt−1,Alayoutt−1 ∂Ac(j

b,t−1)

J i Ac(ijb,t−1) ⊙ Mb − Mb i Ac(ijb,t−1) i Ac(ij

2 (16)

=

b,t−1)

where J is a matrix with all elements equal to 1. All variables in Eq. 15 are known, indicating the existence of the gradient in Eq. 7.

When using the loss function given by Eq. 9 under any spatial-aware conditions, the gradient in Eq. 7 can be derived as follows:

∂L Atextt−1,Aspatialt−1 ∂Coect

 

 

∂Lb Atextt−1,Aspatialt−1 ∂Ac(j

∂Ac(jb,t−1) ∂zt−1

∂ξct ∂Coect

∂zt−1 ∂ϵt

∂ϵt ∂ξct

=

b,t−1)

b

 

(17)

√1 − α¯t √αt

∂Lb Atextt−1,Aspatialt−1 ∂Ac(j

∂Ac(jb,t−1) ∂zt−1

1 − α¯t−1 − σ2 −

=

b,t−1)

b

  

#### ϵct · exp Coetextt + Coespatialt

×

2

exp Coetextt + exp Coespatialt

∂Lb Atextt−1,Aspatialt−1 ∂Ac(j

J i Ac(ijb,t−1) ⊙ fb(C) − fb(C) i Ac(ijb,t−1) i Ac(ij

2 (18)

=

b,t−1)

b,t−1)

where c ∈ {text,spatial}. Therefore, the gradient in Eq. 7 exists for the selection of different loss functions.

#### B.3 Inference details

We provide a detailed compositional denoising process for RealCompo, which achieves a complementary balance between the advantages of the T2I model and the spatial-aware diffusion model by combining their predicted noise during the denoising stage. We provide the pseudocode for the compositional denoising process of the layout-based RealCompo as followed, we have highlighted the innovations of our method in blue.

Algorithm 1 Compositional denoising procedure of layout-based RealCompo Input: A text prompt P, a set of layout B, a pretrained T2I model and a pretrained L2I model Output: A clear latent z0

- 1: zT ∼ N(0,I)
- 2: CoetextT = CoelayoutT ∼ N(0,I)
- 3: for t = T,...,1 do
- 4: if t > t0 then
- 5: ϵt,_ = L2I(zt,P,B,t)
- 6: else
- 7: ϵtextt ,_ = T2I(zt,P,t)
- 8: ϵlayoutt ,_ = L2I(zt,P,B,t)
- 9: Get the balanced noise ϵt from Eq. 2 and Eq. 3
- 10: Get the denoised latent zt−1 from Eq. 12
- 11: ϵtextt−1,Atextt−1 = T2I(zt−1,P,t)
- 12: ϵlayoutt−1 ,Alayoutt−1 = L2I(zt−1,P,B,t)
- 13: Compute L(Atextt−1,Alayoutt−1 ) from Eq. 6
- 14: Update Coect according to Eq. 7
- 15: Get the balanced noise ϵt from Eq. 2 and Eq. 3
- 16: end if
- 17: Get the denoised latent zt−1 from Eq. 12
- 18: end for
- 19: return z0

#### B.4 Gradient Analysis

Gradient Analysis We selected RealCompo v3 and v4 to analyze the gradient changes in Eq. 7 across all denoising stages. As shown in Fig. 10, we use the same prompt and random seed to visualize the gradient magnitude changes corresponding to T2I and L2I for each model version. We observe that the gradient magnitude change of RealCompo v4 fluctuated more in the early denoising stages. We argue that TokenCompose, which enhances the composition capability of multiple-object generation by fine-tuning the model using segmentation masks, may overlap in functionality with the layout-based multiple-object generation, and TokenCompose’s positioning of objects may not consistently align with the bounding box. Therefore, RealCompo must focus on balancing the positioning of TokenCompose and layout in the early denoising stages, leading to less stable gradients compared to RealCompo v3. Additionally, due to LayGuide’s weaker positioning ability compared to GLIGEN, RealCompo v4 may occasionally generate objects with less coverage of the bounding box, as mentioned in the ablation experiment in Section 4.3.

1e 5

RealCompo v3 L2I RealCompo v3 T2I RealCompo v4 L2I RealCompo v4 T2I

1.0

0.8

GradientMagnitude

0.6

0.4

0.2

0.0

0 10 20 30 40 50 Denoise Step

Figure 10: Changes of gradient magnitude in Eq. 7 across all denoising process for the T2I and L2I models of RealCompo v3 and v4.

#### B.5 Limitations and Future Work

Limitations While our RealCompo enhances both realism and compositionality in a training-free manner, it should be noted that the computational cost of our method is slightly higher compared to that of a single T2I model or a single spatial-aware image diffusion model, due to the need to combine two models and compute loss and gradients. However, by adjusting the combination stage of RealCompo, we can keep the computational cost within an acceptable range.

Future Work In future work, we aim to explore more efficient computational methods to improve the calculation efficiency of RealCompo while maintaining high-quality results. Additionally, we plan to extend its application to more challenging tasks such as text-to-video and text-to-3D generation.

#### B.6 Broader Impact

Recent significant advancements in text-to-image diffusion models have opened up new possibilities for creative design, autonomous media, and various other sectors. However, the dual-use nature of this technology raises concerns about its social impact. Image diffusion models carry the risk of misuse, particularly in the realm of impersonating humans. For example, in today’s society, malicious applications such as "deepfakes" have been employed in inappropriate contexts to fabricate attacks on specific public figures. It is crucial to clarify that our algorithm is designed to enhance the quality of image generation, and we do not endorse or facilitate such malicious applications.

### C More Generation Results

Layout SD1.5 GLIGEN LMD+ RealCompo(Ours)

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

a brown teddy bear

a blue water cup

a sofa

A brown teddy bear holding a blue water cup sits on the sofa.

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

a blue bird

two yellow

a branch blossom

A blue bird standing on a branch with two yellow blossoms on the right.

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

a fireworks display

a group of people

A group of people are watching a fireworks display.

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

a cat

a cat

a cat

a plate of food

table

Three cats are sitting on the table, with a plate of food in front of them.

| |a child a book<br><br>a tree|
|---|---|

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

A child is reading a book under a tree.

||a person<br><br>a camera<br><br>the city|
|---|
|
|---|

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

A person is holding a camera and taking photos of the city.

##### Figure 11: More generation results about layout-based RealCompo.

Keypoint SDXL ControlNet RealCompo(Ours)

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Cinematic photo an action shot of Leonardo teenage mutant turtle ninja, with katana weapon, wet and dirty background

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

2 girl, Elsa and Anna, sparks of magic between them, princess dress, background with sparkles, black purple red color schemes.

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Two astronauts standing on the moon, behind them is a white planet amidst the vast universe.

Figure 12: More generation results about keypoint-based RealCompo.

Segmentation map SD v2.1 ControlNet RealCompo(Ours)

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

An airplane parked on the runway as the sun sets behind it.

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

Two cats sitting on the windowsill looking at each other.

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Five men stand together in a line, serious in expression.

Figure 13: More generation results about segmentation-based RealCompo.

