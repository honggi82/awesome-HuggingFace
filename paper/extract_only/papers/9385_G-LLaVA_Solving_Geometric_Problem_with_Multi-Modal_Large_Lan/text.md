# arXiv:2312.11370v2[cs.CL]20Aug2025

[Figure 1]

## G-LLaVA : Solving Geometric Problem with Multi-Modal Large Language Model

### Jiahui Gao1,2∗, Renjie Pi3∗, Jipeng Zhang3, Jiacheng Ye2, Wanjun Zhong1, Yufei Wang1, Lanqing Hong1, Jianhua Han1, Hang Xu1, Zhenguo Li1, Lingpeng Kong2 1Noah’s Ark Lab 2The University of Hong Kong 3The Hong Kong University of Science and Technology

sumiler@connect.hku.hk, rpi@connect.ust.hk

Large language models (LLMs) have shown remarkable proficiency in human-level reasoning and generation capabilities, which encourages extensive research on their application in mathematical problem solving. However, current work has been largely focused on text-based mathematical problems, with limited investigation in problems involving geometric information. Addressing this gap, we aim to enable LLMs to solve geometric problems by understanding image input. We first analyze the limitations of current Multimodal Large Language Models (MLLMs) in this area: they struggle to accurately comprehending basic geometric elements and their relationships. To overcome these challenges, we take advantage of the unique characteristics of geometric problems (such as unique geometric logical form, and geometric scalability) and the capacity of the textual LLMs to build an enriched multimodal geometry dataset based on existing data. The augmented dataset, Geo170K, contains more than 170K geometric image-caption and question-answer pairs. Utilizing our constructed Geo170K dataset, we develop G-LLaVA, which demonstrates exceptional performance in solving geometric problems, significantly outperforming GPT-4-V on the MathVista benchmark with only 7B parameters.

### 1 Introduction

Large language models (LLMs) exhibit humanlike proficiency in reasoning (Wei et al., 2022; Wang et al., 2022; Zhou et al., 2022) and generation (Ouyang et al., 2022; Touvron et al., 2023), which encourages extensive research on their application in mathematical problem solving (Fu et al., 2023; Gou et al., 2023; Yue et al., 2023b; Luo et al., 2023; Zhao et al., 2023a,b; Jiang et al., 2023). These problems often require highly sophisticated

∗Equal Contribution.

and symbolic reasoning capabilities, often considered impossible to solve before the era of LLMs.

It is an intuitive approach to use LLMs for mathematical reasoning problems presented in a textual form. Nevertheless, a substantial proportion of mathematical reasoning problems necessitate the comprehension of geometric information. Moreover, even when certain problems do not overtly pertain to geometric information on the surface, the integration of geometrical-based methods often holds significant practical implications (e.g., analytic number theory). With the advent of GPT-4V (OpenAI, 2023), Gemini1 (Google, 2023), and numerous multi-modal large language models (MLLMs) (Zhu et al., 2023; Liu et al., 2023; Dai et al., 2023; Li et al., 2023; Bai et al., 2023; Lai et al., 2023; Gao et al., 2023b; Pi et al., 2023b), recent work has progressively looking into employing MLLMs to tackle geometric reasoning problems in mathematics (Yang et al., 2023; Lu et al., 2023; Yue et al., 2023a).

However, we have observed that even with the most advanced MLLMs, current systems still exhibit limitations in addressing geometric problems due to challenges in accurately comprehending geometric figures. For instance, as demonstrated in Figure 1, GPT4-V often produces inaccurate descriptions for geometric figures. Specifically, the model struggles with understanding the relationships between fundamental elements like points and lines, and in accurately interpreting elements such as the degree of an angle. We presume that the underlying reason for this may be the fact that these MLLMs are typically trained with images and descriptions from the general domain, and the ability to understand such semantics differs significantly from that required for geometric reasoning.

1Gemini, a concurrent work, was released one week before our submission. Consequently, our work is primarily benchmarked against GPT4-V and other MLLMs.

###### GPT4-V LLaVA1.5

###### MiniGPT-4 G-LLaVA

Describe the figure in detail.

Describe the figure in detail.

Describe the figure in detail.

Describe the figure in detail.

The image features a white circle with a line drawn through its center. The line is labeled with the letter "d" and extends from the center of the circle to the edge. The circle is divided into four equal parts, with each part labeled with the letters "a," "b," "c," and "d." The circle is also labeled with the number "0" in the top left corner. The overall appearance of the image suggests a mathematical or geometric concept.

The image shows a geometric figure consisting of a semicircle and a rightangled triangle. The semicircle has its diameter on the horizontal line segment AC, with its center marked as point O. Point D is at the topmost part of the semicircle, indicating the midpoint of the arc. The right-angled triangle is formed by the diameter AC of the semicircle and a vertical line segment BD dropping from point D to point B on the diameter AC. The triangle's hypotenuse is the radius of the semicircle, and the right angle is at point B.

The diagram shows a semicircle with a diameter AB. Point D is located on the semicircle, and a line CD is drawn from point D to point C. Point O is the center of the semicircle.

The image shows a right triangle with a hypotenuse of length 10 and legs of length 5 and 12. The triangle is labeled with the letters a, b, and c, which represent the vertices of the triangle. The hypotenuse is labeled with the number 10, and the legs are labeled with the numbers 5 and 12. The triangle is drawn in black ink on a white background.

|<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>The image shows a geometric figure consisting of a semicircle and a rightangled triangle. The semicircle has its diameter on the horizontal line segment AC, with its center marked as point O. Point D is at the topmost part of the semicircle, indicating the midpoint of the arc. The right-angled triangle is formed by the diameter AC of the semicircle and a vertical line segment BD dropping from point D to point B on the diameter AC. The triangle's hypotenuse is the radius of the semicircle, and the right angle is at point B.<br><br>The image features a white circle with a line drawn through its center. The line is labeled with the letter "d" and extends from the center of the circle to the edge. The circle is divided into four equal parts, with each part labeled with the letters "a," "b," "c," and "d." The circle is also labeled with the number "0" in the top left corner. The overall appearance of the image suggests a mathematical or geometric concept.<br><br>The image shows a right triangle with a hypotenuse of length 10 and legs of length 5 and 12. The triangle is labeled with the letters a, b, and c, which represent the vertices of the triangle. The hypotenuse is labeled with the number 10, and the legs are labeled with the numbers 5 and 12. The triangle is drawn in black ink on a white background.<br><br>GPT4-V LLaVA1.5<br><br><br><br>The diagram shows a semicircle with a diameter AB. Point D is located on the semicircle, and a line CD is drawn from point D to point C. Point O is the center of the semicircle.<br><br>MiniGPT-4 G-LLaVA<br><br><br><br><br><br><br><br><br><br>Describe the figure in detail.<br><br><br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]|
|---|

|[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>The image shows a geometric figure consisting of a semicircle and a right-angled triangle. The semicircle has its diameter on the horizontal line segment AC, with its center marked as point O. Point D is at the topmost part of the semicircle, indicating the midpoint of the arc. The right-angled triangle is formed by the diameter AC of the semicircle and a vertical line segment BD dropping from point D to point B on the diameter AC. The triangle's hypotenuse is the radius of the semicircle, and the right angle is at point B.<br><br>The image features a white circle with a line drawn through its center. The line is labeled with the letter "d" and extends from the center of the circle to the edge. The circle is divided into four equal parts, with each part labeled with the letters "a," "b," "c," and "d." The circle is also labeled with the number "0" in the top left corner. The overall appearance of the image suggests a mathematical or geometric concept.<br><br>The image shows a right triangle with a hypotenuse of length 10 and legs of length 5 and 12. The triangle is labeled with the letters a, b, and c, which represent the vertices of the triangle. The hypotenuse is labeled with the number 10, and the legs are labeled with the numbers 5 and 12. The triangle is drawn in black ink on a white background.<br><br>GPT4-V LLaVA1.5<br><br>[Figure 38]<br><br>The diagram shows a semicircle with a diameter AB. Point D is located on the semicircle, and a line CD is drawn from point D to point C. Point O is the center of the semicircle.<br><br>MiniGPT-4 G-LLaVA<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>Describe the figure in detail.<br><br>[Figure 43]|
|---|

- Figure 1: State-of-the-art MLLMs suffer severe hallucination on geometric figures, which greatly hinders their abilities on solving geometric problems. On the other hand, our G-LLaVA’s ability to interpret geometric figure is boosted after the alignment phase with our curated dataset.

To address this issue, one of the most direct and effective approaches is to enhance current MLLMs by augmenting them with data containing highquality descriptions of geometric information (Ye et al., 2022a; Meng et al., 2022). However, a significant challenge arises from the limited size of the largest publicly available geometric problem dataset, which contains only a few thousand question-answer pairs. Additionally, the current datasets lack descriptions of geometric images and exhibit a limited range of problem-solving methods, which constrains the model’s ability to understand basic geometric elements and affect its problemsolving capabilities.

geometric problems, surpassing SOTA MLLMs by a large margin. Specifically, G-LLaVA-13B outperforms LLaVA-13B by 27.4 on GPS minitest split of MathVista (Lu et al., 2023). In addition, with only G-LLaVA-7B, it is able to surpass the powerful GPT4-V on the geometry problem solving questions. Code and data will be available at https://github.com/pipilurj/G-LLaVA.

### 2 Related Work

Multi-Modal Large Language Model. Recent years have witnessed transformative advancements in the development of large language models (LLMs), characterized by a series of pioneering studies (Brown et al., 2020; Scao et al., 2022; Chowdhery et al., 2022; Smith et al., 2022; Hoffmann et al., 2022; Ouyang et al., 2022; Touvron et al., 2023; Bai et al., 2022). These breakthroughs have significantly elevated the capabilities of language understanding and generation, showcasing near-human proficiency across diverse tasks. Concurrently, the success of LLMs has inspired explorations into vision-language interaction, leading to the emergence of multi-modal large language models (MLLMs) (Liu et al., 2023; Li et al., 2023; Dai et al., 2023; Zhu et al., 2023; Dai et al., 2023; OpenAI, 2023; Bai et al., 2023; Su et al., 2023; Gao et al., 2023b). These models have exhibited remarkable capabilities in synthesizing detailed descriptions and engaging in dialogue based on visual

In this paper, we propose to synthesize geometric visual-text data leveraging existing datasets via text-only LLMs (e.g., ChatGPT). More specifically, we utilize the geometry characteristic to construct a multi-modal geometry dataset, building upon existing datasets. The data generation process involves incorporating utilizing uniqueness of geometric logic form, geomertic representation uniqueness, geometric scalability, etc (as shown in Figure 2). We term our generated dataset Geo170K, which contains around 60,000 geometric imagecaption pairs and more than 110,000 questionanswer pairs. This dataset is 28 times larger than GeoQA+, greatly expanding the coverage of geometric problems. With our collected Geo170K, we derive G-LLaVA, a MLLM capable of solving

Geometric Description Generation via Information Recovery

Generate Contrastive QA Pairs for Basic Elements

[Figure 44]

[Figure 45]

Logic Form

###### QA Example1

Similar(Triangle(A,B,C),Triangle(C,B,D)) ...

Q: ⊙ O is the circumscribed circle of △ABC, … ∠ABO=30.0, then the size of ∠ACB is ?

Image Description

The diagram consists of a triangle ABC inscribed within a circle O. Points A, B, and C are the vertices of the triangle, and they lie on the circumference of the circle. The center of the circle is marked as point O.

Contrastive QA pairs

Info Items

A: In triangle AOB, OA=OB, ∠ABO=30°; Therefore, ∠AOB=180°- 2∠ABO=120°; Therefore, ∠ACB=1/2 ∠AOB=60°

1. Triangle ABC is similar to triangle CBD. …

Q: Is point D the lies on line BC? A: No, … , point D lies on the line segment AB.

Diagram Representation Uniqueness Logic Structure Uniqueness

Geometry Characteristic

Geometric Solution Generalizability

Geometric Scalability Conditional Uniqueness Reasoning Diversity

Variable Substitution and Equation Solving Value Scaling Re-Formulating Condition as Unknown Sentence Paraphrase

QA Example2

Q: Point B is in line AC, line AB is 1, line AC is 3, line BC is ? A: Line BC is 2.

QA Example2 QA Example2 QA Example2

Q: Given that Point B lies on line segment AC, with AB=1 unit and AC=3 units, what is the length of segment BC?

Q: Point B is in line AC, line AB is x (x=1), line AC is y (y=3), line BC is ?

Q: Point B is in line AC, line AB is u, line AC is 3, line BC is 2. What is the value of u? A: Denote AB as u, we can determine u by solving u+2=3. Therefore, u equals 1.

Q: Point B is in line AC, line AB is 2, line AC is 6, line BC is ? A: The length of BC is 4 units, obtained by subtracting AB (2 units) from AC (6 units).

A: BC can be determined by subtracting the length of segment AB from the length of segment AC. Since AB is 1 unit and AC is 3 units, BC = AC - AB, which equals 2 units.

A: Denote BC as z, we can build the equation x+z=y, therefore z=y-x, which simplifies to z=3-1=2. Thus, BC is 2.

: Generate data using text-only LLM : Multimodal Geometric Alignment Data : Multimodal Geometric Instruction Data

- Figure 2: Framework of our multi-modal geometric data generation using the characteristics of geometry problems.

Data Generation via LLM. Bootstrapping data from pretrained models has long been an active area of research. (Ye et al., 2022a; Meng et al., 2022) generates training data using pretrained language models such as GPT-2 for classification tasks. (Gao et al., 2023a) improves the quality of generated dataset via bi-level approach. (Ye et al., 2022b) utilizes influence function to select in-context examples to aid data generation. Recently, automatic data generation becomes more ubiquitous with the advent of powerful LLMs such as ChatGPT, a line of recent works utilize ChatGPT-generated data to perform instruction tuning (Wang et al., 2023; Peng et al., 2023; Taori et al., 2023; Liu et al., 2023; Zhu et al., 2023; Bai et al., 2023; Pi et al., 2023a; Su et al., 2023; Yu et al., 2023; Chen et al., 2023; Zhang et al., 2023).

inputs. However, we observe that even the state-ofthe-art MLLMs face challenges in resolving geometric problems using diagrams and figures.

Geometry Problem Solving. The Geometry problem reasoning is an challenging visual mathematical reasoning problem. Early efforts by Seo et al. (2015); Sachan et al. (2017); Alvin et al. (2017); Sachan and Xing (2017) focused on creating datasets through manual efforts. More recent approaches have introduced enhanced methods and datasets, including Geometry3K (Lu et al., 2021), GeoQA (Chen et al., 2021), GeoQA+ (Cao and Xiao, 2022), UniGeo (Chen et al., 2022), UniMath (Liang et al., 2023), and SCA-GPS (Ning et al., 2023), aiming to improve both performance and explainability. However, the scale of current datasets remains limited, and the performance of traditional models in this domain has not achieved the level observed in other areas of mathematical problem solving, particularly when compared to methods that utilize large language models for solving math word problems (Cobbe et al., 2021; Wei et al., 2022; Gou et al., 2023).

### 3 Observation

We observe that most state-of-the-art (SOTA) MLLMs, although being adept at understanding daily visual scenes, have difficulty in comprehending geometric figures, even if they are simple and

#### Geometric Description Generation via Information Recovery

QA Pair: Question: As shown in the figure, circle O is the circumscribed circle of triangle ABC, and it is known that angle ABO = 30.0, then the size of angle ACB is () Answer: In triangle AOB, OA=OB, angle ABO=30°; Therefore, angle AOB=180°- 2 angle ABO

[Figure 46]

=120°; Therefore, angle ACB=1/2angle AOB=60° Diagram Description:

The diagram consists of a triangle ABC inscribed within a circle, where the circle is denoted as circle O. Points A, B, and C are the vertices of the triangle, and they all lie on the circumference of the circle. The center of the circle is marked as point O.

- Table 1: Full geometric diagram description generation via inverse information recovery. The description is generated based on the textual QA pair. The upper section shows the QA pair employed to instruct text-only ChatGPT, while the lower section ( in blue ) shows the responses produced by ChatGPT.

straightforward for humans. In Figure 1, we demonstrate the descriptions generated by SOTA MLLMs for geometric figure. We observe that severe hallucination exists in all the generated descriptions.

More specifically, we find GPT4-V has difficulty understanding relationships between basic elements like points and lines, and also struggles with precisely interpreting these elements themselves (such as the angle B in Figure 1). Furthermore, smaller MLLMs like LLaVA1.5 and MiniGPT4 demonstrate even greater difficulty in accurately identifying the types of geometric shapes present in a figure.

This inadequacy in interpreting geometric diagrams may be one of the major causes for the failure in solving geometric problems. In contrast, actual geometric diagrams typically exhibit clear and well-defined relationships among their elements. This geometry characteristic can be utilized to develop datasets that help mitigate the above issues and mitigate hallucination.

### 4 Geometric Data Generation

While previous efforts have been made to address multi-modal geometry problems (Chen et al., 2021, 2022; Cao and Xiao, 2022), the availability of geometry datasets remains limited. The key limitations of existing datasets are threefold: (1) limited data volume (a few thousands for the largest dataset), (2) absence of detailed descriptions for geometric images, and (3) a lack of diversity in problem-solving methodologies and answer pathways. This limitation presents challenges for MLLMs in accurately understanding geometric elements and providing precise geometric solutions.

To address this issue, we utilize the geometry characteristic to construct a multi-modal geometry

dataset based upon existing dataset. This dataset includes two parts: an alignment dataset to provide MLLMs with fundamental geometric knowledge and an instruction-tuning dataset to improve the assistant’s ability to understand user instructions and generate accurate geometry solutions.

4.1 Geometric Cross-Modal Alignment Data

- 4.1.1 Geometric Image Caption Generation

Image-caption datasets play a significant role in training MLLMs for understanding the context of images, which is essential for aligning image and text modalities. In the field of geometry, there is a lack of such datasets that offer detailed descriptions of geometric diagrams. To address this issue, we propose the generation of image descriptions from labeled question-answer (QA) pairs, as illustrated in Table 1. In particular, we use text-only ChatGPT 3.5 to create image captions based on these human-labeled QA pairs, which can be considered as a type of inverse information recovery. This approach leverages the strong understanding ability of ChatGPT to produce descriptions for geometric diagrams.

- 4.1.2 Contrastive QA Pairs for Basic Elements

Our approach also involves generating QA pairs to facilitate the comprehension of geometric diagrams, focusing primarily on their basic elements. The process begins with the interpretation of humanlabeled logical forms on Geometry3k (Lu et al., 2021). We employ text-only ChatGPT to convert these logical forms into clear descriptions that cover various geometric elements such as shapes, lines, and points, and their relationships.

After creating these diagram descriptions, the model begins to produce contrastive QA pairs.

#### Contrastive QA Pairs for Basic Elements

Logic Form: Similar(Triangle(A,B,C),Triangle(C,B,D))

[Figure 47]

- Triangle(A,B,C)
- Triangle(A,C,B) Equals(LengthOf(Line(C, B)), 5) Equals(LengthOf(Line(C, D)), 3) Equals(LengthOf(Line(B, D)), 4) PointLiesOnLine(D, Line(A, B)) Perpendicular(Line(A, C), Line(C, B)) Perpendicular(Line(C, D), Line(A, D))

- Phase-1 Generation

Info Items: The geometric figure have the following information:

- Info 1: The figure represents a triangle with vertices A, B, and C.

- Info 2: Triangle ABC is similar to triangle CBD.

- Info 3: Point D lies on the line segment AB.

- Info 4: Line segment AC is perpendicular to line segment CB.

... Summary:

The figure depicts a triangle ABC and triangle CBD, which is similar to triangle ABC. Additionally, triangle ACB is also present in the figure. The lengths of the segments CB, CD, and BD are 5, 3, and 4 units respectively. Point D lies on the line segment AB, and perpendicularity is indicated between the line segments AC and CB, as well as between the line segments CD and AD.

- Phase-2 Generation

##### QA pairs:

- 1. Question: Does the diagram include a triangle formed by vertices A, B, and D? Answer: No, the diagram does not include a triangle formed by vertices A, B, and D. The triangles present are ABC and CBD, and the point D lies on the line segment AB.

- 2. Question: Does the diagram contain a triangle with vertices A, C, and B? Answer: Yes, the diagram contains triangle ACB, which is the same as triangle ABC.

- 3. Question: Is point D the lies on line BC? Answer: No, point D is not on lines BC. The point D lies on the line segment AB.

- 4. Question: Is line segment AC perpendicular to line segment CB? Answer: Yes, line segment AC is perpendicular to line segment CB.

- Table 2: Geometric image description and contrastive QA pairs for understanding basic elements. The generation process consists of two phases: 1) Translating the human-labelled logic form into detailed information items and a summary of the diagram description. 2) Generating contrastive QA pairs based on the provided information and summary. The sections in blue display the responses generated by ChatGPT. The detailed prompt will be included in the appendix.

These pairs are designed to examine different aspects of the diagrams. Questions may explore the presence of certain geometric elements (e.g., "Are there triangular shapes in the diagram?") or check the accuracy of the relationships described (e.g., "Is point D the lies on line BC?"). This method enables the model to comprehend geometric concepts and to analyze and interpret the details in geometric diagrams accurately. The generation example is shown on Table 2.

#### 4.2 Geometric Instruction Data

After performing alignment leveraging the constructed alignment data, the model is able to better interpret the geometric diagram (Figure 1). However, they are still limited at solving geometric prob-

lems. Therefore, we construct an instruction tuning dataset based on existing datasets with the help of powerful LLMs. Specifically, we design a series of strategies to expand the question-answer pairs in existing datasets. The resulting dataset contains more than 110k QA pairs, which is the largest public geometric QA dataset available. We will introduce the proposed strategies in detail below.

#### 4.2.1 Equation Solving (ES)

As shown in Table 5, we replace the specific values in the original QA pairs with unknown variables and prompt the LLM to construct the solution by solving equation. Such data is helpful for the MLLM to generalize its understanding of the problem, which enables it to apply the similar reasoning

#### Original Example

Question: As shown in the figure, the length of the short arm of the railing at the railway crossing is 1.0 and the length of the long arm is 16.0. When the end of the short arm drops by 0.5, the end of the long arm rises () Answer: By the property of similar triangles, the lengths of corresponding sides are proportional. Since the long arm is 16 times longer than the short arm, the height it rises is 16 times the height the short arm drops, which is 0.5 multiplied by 16, resulting in a rise of 8 meters.

[Figure 48]

Table 3: The original example.

#### Value Scaling

Question: At a railroad crossing, the short arm of the barrier is 2.0 unit long and the long arm is 32 units long. If the short arm drops 1 units, by how many units will the long arm rise? Answer: Denote the rise of the long arm as ‘h’. In similar triangles, the ratios of their corresponding sides are in proportion. h can be calculated using the equation 1/h = 2/32. Solving for ‘h’ gives h = 32 * 1 / 2, which equals 16 units. Therefore, the long arm rises by 16 units.

[Figure 49]

Table 4: Synthetic example via value scaling.

and solution steps to different scenarios. The abstraction of the problem by using variables and solving equation helps the LLM focus on the underlying mathematical concepts and relationships, rather than getting caught up in specific numerical values.

#### 4.2.2 Value Scaling (VS)

As shown in Table 4, we augment the data by scaling the length values in the QA pairs. Note that for the same diagram, the QA pair is still correct if all the lengths in a geometric problem are scaled simultaneously. However, note that it is not the case for quantities such as angles. When different scaling of values are applied, the LLM becomes more flexible in handling different numerical inputs. Involving a range of values that extends beyond the initial training dataset aids in refining the model’s computational and reasoning capabilities, thereby contributing to its generalizability.

#### 4.2.3 Re-Formulating Condition as Unknown (RCU)

Motivated by(Weng et al., 2023; Yu et al., 2023), we design new multi-modal QA pairs that ask questions backwards, as shown in Table 6. Specifically, we reformulate questions to ask for the values originally present in the condition, and retain the generated data with correct answer only. In this way, the LLM is repeatedly exposed to the relationships

between variables, equations, and their solutions. This reinforcement helps the model learn the dependencies and connections between different elements in a mathematical problem.

#### 4.2.4 Sentence Paraphrase (SP)

We also conduct paraphrasing for both the question and answer pairs, as shown in Table 7. This exposes the LLM to a broader range of phrasing and language variations. This helps the model become more robust in understanding and generating diverse sentence structures. Consequently, it can handle similar questions with different phrasings and provide accurate responses.

5 Model Architecture and Training

#### 5.1 Model Architecture

We utilize the LLAVA (Liu et al., 2023) architecture for our model. The model mainly consists of a large language model (LLM) such as LLAMA2 (Touvron et al., 2023), a pretrained vision transformer (Radford et al., 2021) (ViT) as image encoder. In addition, a projection layer is required to map the visual features from the image encoder to the same dimension as the LLM.

During inference, given an image and a textual instruction, the image encoder first extracts the visual tokens from the image, which are then mapped to the dimension of LLM’s embedding space via

#### Variable Substitution and Equation Solving

Question: Given a railway crossing barrier with two arms, where the shorter arm has a length of ‘a’ units (a = 1.0) and the longer arm has a length of ‘b’ units (b = 16.0), determine the elevation ‘y’ units of the longer arm’s end when the shorter arm descends by ‘d’ units (d = 0.5). Express your answer as a function of ‘a’, ‘b’, and ‘d’.

[Figure 50]

Answer: To find the elevation ‘y’ of the long arm’s end, we apply the property of similar triangles and establish the equation d y = ab . Substituting the given values, we get 0y.5 = 161..00. Solving for ’y’ leads to y = b×ad, which simplifies to y = 16.10.×00.5. Therefore, ’y’ equals 8.0 units, indicating that the long arm rises by 8.0 units when the short arm descends by 0.5 units.

##### Question:

In a railway crossing lever system, the lengths of the shorter and longer arms are ‘s’ units (s = 1.0) and ‘l’ units (l = 16.0) respectively. If the shorter arm lowers by ‘deltaS’ units (deltaS = 0.5), what is the increase ‘deltaL’ in the length of the longer arm?

##### Answer:

The increase ‘deltaL’ in the longer arm can be found using the proportionality relationship of similar triangles. The equation can be expressed as: deltaLdeltaS = sl . Inserting the given values, the equation becomes deltaL0.5 = 161..00. Solving for ‘deltaL’ yields deltaL = l×deltaSs , which simplifies to deltaL = 16.10.×00.5. Thus, deltaL is 8.0 units.

Question: At a railroad crossing, the short arm of the barrier is 1.0 unit long and the long arm is 16.0 units long. If the short arm drops 0.5 units, by how many units will the long arm rise? Represent the rise of the long arm as ‘h’.

##### Answer:

The triangles formed by the shorter and longer arms as they descend are similar. The rise of the long arm ‘h’ can be calculated using the property of similar triangles, the equation is 0.5/h = 1/16. Solving for ‘h’ gives h = 16 * 0.5 / 1, which equals 8.0 units.

Table 5: Synthetic example via replacing values to unknown variables and solving the equation.

#### Re-Formulating Condition as Unknown (Weng et al., 2023; Yu et al., 2023)

Question: At a railroad crossing, the short arm of the barrier is u unit long and the long arm is 16.0 units long. When the end of the short arm drops by 0.5, the end of the long arm rises 8 units. What is the value of unknown variable u?

[Figure 51]

Answer: Denote the short arm of the barrier as variable u. By the property of similar triangles, we can determine u by solving the equation 0.5/8 = u/16. Therefore, u equals 1.

Table 6: Synthetic example via re-formulating condition as unknown.

#### Sentence Paraphrase

Question: In the illustration, the railing at the railway crossing has a short arm measuring 1.0 unit in length and a long arm measuring 16.0 units. When the short arm drops by 0.5 units, what is the corresponding rise in the long arm?

[Figure 52]

Answer: The triangles are similar, and their corresponding sides are proportional. The long arm is 16 times longer than the short arm, resulting in an 8-meter rise when the short arm drops by 0.5 meters.

Table 7: Synthetic example via sentence paraphrase.

the projection layer. Then, the mapped image features are concatenated with text embeddings to serve as the input to the LLM. Subsequently, the LLM begins to perform next-token-generation.

#### 5.2 Model Training

We train our G-LLaVA in two phases, namely 1) geometric visual-language alignment, and 2) geometric instruction tuning. In both phases, we leverage the conventional language modeling loss, which can be formulated as follows:

L(Star, Sin, I) = −

L

log p Start |F(star(<t), Sin, I)

t=1

(1)

where F represents the model. I represents the geometric figure; Star and Sin represent the target and input sentences, respectively; Start denotes the tth token of target output, and L stands for length.

- 6 Experiments

#### 6.1 Setup

Dataset. We generate the alignment data and instruction data utilizing training set of GeoQA+ (Cao and Xiao, 2022) and Geometry3K (Lu et al., 2021). More specifically, the contrastive question-answer (QA) pairs in the alignment data are generated using Geometry3K, which features human-labeled logical forms. Note that GeoQA+ covers the training set of GeoQA (Chen et al., 2021), and share the same val/test set as GeoQA (Chen et al., 2021). More details of data split on GeoQA and GeoQA+ is listed in Table 9. Our approach results in 60K alignment data samples, and more than 110K instruction data samples.

We compare our model with other MLLMs on the geometry problems on the minitest split MathVista (Lu et al., 2023), and compare our model with traditional in-domain model on the test split of GeoQA following (Chen et al., 2022; Liang et al., 2023). The geometry problems in MathVista minitest set is collected from four source datasets Geometry3K (Lu et al., 2021), GeoQA+ (Cao and Xiao, 2022), GEOS (Seo et al., 2015) and UniGeo (Chen et al., 2022).

Implementation Details. We employ ChatGPT (gpt-3.5-turbo-0613) for data generation. A detailed description of our prompts will be provided in the appendix. We use LLaVA (Liu et al., 2023)

Model Input Accuracy (%) Heuristics Baseline

Random Chance - 21.6 Frequent Guess - 34.1 Human Q,I 48.4

Close Source Model

Text-Only LLMs

2-shot CoT Claude-2 Q 29.8 2-shot CoT ChatGPT Q 36.5 2-shot CoT GPT-4 Q 44.7 2-shot PoT ChatGPT Q 30.8 2-shot PoT GPT-4 Q 33.2

Visual-Augmented LLMs

2-shot CoT Claude-2 Q,Ic,It 31.7 2-shot CoT ChatGPT Q,Ic,It 29.3 2-shot CoT GPT-4 Q,Ic,It 31.7 2-shot PoT ChatGPT Q,Ic,It 26.4 2-shot PoT GPT-4 Q,Ic,It 39.4

Multimodal LLMs

Multimodal Bard Q,I 47.1

- Gemini Nano 1 Q,I 21.6

- Gemini Nano 2 Q,I 23.6 Gemini Pro Q,I 40.4 Gemini Ultra Q,I 56.3 GPT4-V Q,I 50.5

Open Source Model

IDEFICS (9B-Instruct) Q,I 21.1 mPLUG-Owl (LLaMA-7B) Q,I 23.6 miniGPT4 (LLaMA-2-7B) Q,I 26.0 LLaMA-Adapter-V2 (7B) Q,I 25.5 LLaVAR Q,I 25.0 InstructBLIP (Vicuna-7B) Q,I 20.7 LLaVA (LLaMA-2-13B) Q,I 29.3 G-LLaVA-7B Q,I 53.4 G-LLaVA-13B Q,I 56.7

Table 8: Comparison of model performance on the testmini set of MathVista benchmarks (Lu et al., 2023) on geometry problem solving (GPS) . For input, Q represents for question, I represents for image, Ic represents for image caption generated by Bard, and It represents fo OCR text detected in the image. Baseline results are obtained from Lu et al. (2023). Human performance and the results surpassing human performance are highlighted in grey. Our results are highlighted in blue .

as our backbone. More specifically, we utilize LLAMA-2 (Touvron et al., 2023) as the language model and employ the visual encoder of a pretrained vision transformer (Radford et al., 2021) (ViT). The resolution of the input image is 336 by 336. We conduct experiments with both 7B and 13B LLMs. In the cross-modal alignment process, only the projection linear layer is trainable. During the instruction tuning phase, both the projection linear layer and the language model are trainable.

For training data, as we found the minitest split

|Dataset<br><br>|Train<br><br>|Validation|Test<br><br>|
|---|---|---|---|
|GeoQA+ (Cao and Xiao, 2022) GeoQA (Chen et al., 2021)<br><br>|6027 3499|745 745<br><br>|754 754|

Table 9: Data Split of GeoQA and GeoQA+.

|Model<br><br>|Input|Accuracy (%)<br><br>|
|---|---|---|
|Random Chance Frequent Guess|-<br><br>|25.0 32.1|

Top-10 Accuracy

NGS (Chen et al., 2021) Q,I 56.9 DPE-GPS (Cao and Xiao, 2022) Q,I 62.7 SCA-GPS (Ning et al., 2023) Q,I 64.1

Top-1 Accuracy

Geoformer (Chen et al., 2022) Q,I 46.8 UniMath (Liang et al., 2023) Q,I 50.0 G-LLaVA-7B Q,I 64.2 G-LLaVA-13B Q,I 67.0

Table 10: Comparison of model performance with traditional methods on GeoQA.

of MathVista contains some examples of Mixtrain.pk of GeoQA+, we remove those samples that also appears in minitest split of MathVista. The learning rate is set to 3e−5. We expand the images into squares during training, where the extended background color is set to white. For image augmentation, we set the maximum translation distance to 0.25 of the length of longer side. If not otherwise specified, the models are trained for 1 epoch for cross-modal alignment and 2 epochs for instruction tuning, respectively. And the batch sizes are set to 6 per GPUs and 32 per GPUs, respectively.

Evaluation Metric. We use accuracy as the metric for evaluation. Note that several prior studies (Chen et al., 2021, 2022; Cao and Xiao, 2022) report results using Top-10 accuracy (generating 10 sequences and selecting the first sequence that successfully addresses the problem as the prediction). Our experimental results directly report Top-1 accuracy. During instruction tuning, we enable the model to output the choice in a fixed format. For evaluation, we directly use regular expression to extract the predicted choices from the generated answers. The answer is considered false if the regular expression fails to extract a valid answer.

#### 6.2 Main Experiment

We compared G-LLaVA with other MLLMs on minitest split of MathVista (Lu et al., 2023) benchmark on Table 8. The results shows that, geometric cross-modal alignment and instructing tuning on our dataset is effective in improve MLLMs’ geometric problem solving ability. Our specific in-

domain model G-LLaVA-7B can even surpass the strong GPT4-V on geometric problems.

#### 6.3 Comparison with Conventional Methods

We additionally compare our method with conventional SOTA methods in geometry problem solving domain. As illustrated in Table 10, our method demonstrates a notable improvement in Top-1 accuracy over the existing SOTA techniques. Moreover, our model’s top-1 accuracy outperforms the baselines’ top-10 accuracy, demonstrating a significant improvement in predictive precision.

#### 6.4 Performance Across Problem Difficulties

We compare G-LLaVA with the baselines models on problems with different difficulty levels, as

- shown in Table 11. Specifically, OP represents the number of “operations", or reasoning steps that needs to be taken for solving the problem. The results verify that our G-LLaVA consistently outperforms baseline models by a large margin across various difficulty levels.

|Model|OP=1(%)<br><br>|OP=2(%)|OP=3(%)<br><br>|OP>=4(%)<br><br>|Total(%)|
|---|---|---|---|---|---|
|LLaVA-7B LLaVA-13B G-LLaVA-7B G-LLaVA-13B<br><br>|16.8 19.1 77.5 79.0<br><br>|20.9 21.3 60.8 64.9<br><br>|15.5 18.5 54.8 55.5<br><br>|22.9 24.6 40.9 49.1<br><br>|18.7 20.3 64.2 67.0<br><br>|

- Table 11: Different difficulty problems on GeoQA.

6.5 Performance Across Different Types of Questions

We compare G-LLaVA with the baselines models on problems with different type of questions, as shown in Table 12. The results suggest that GLLaVA performs better than the baseline models in various geometric problems such as angle, length, and area problems.

|Model|Angel|Length<br><br>|Area|Other<br><br>|Total|
|---|---|---|---|---|---|
|LLaVA-7B LLaVA-13B G-LLaVA-7B G-LLaVA-13B<br><br>|16.1 17.5 70.7 71.5<br><br>|22.2 23.0 56.5 61.1<br><br>|17.0 25.5 55.3 63.8<br><br>|14.3 28.6 42.9 57.1<br><br>|18.7 20.3 64.2 67.0<br><br>|

- Table 12: Performance of different types of questions on GeoQA.

#### 6.6 Effectiveness of Cross-Modal Geometric Alignment

To evaluate the alignment phase’s effectiveness, we conducted the analysis of the model’s performance with and without alignment phase in Table 13. The results suggest that the alignment phase enhances the model’s ability to interpret images, which is also illustrated by the qualitative result in Figure 1.

[Figure 53]

[Figure 54]

[Figure 55]

GPT4-V

[Figure 56]

As shown in the figure, AB is tangent to circle O at point B, and the extended line of AO intersects circle O at point C. Connect BC, if angle A = 36.0, then angle C is equal to () Choices: A:36° B:54° C:60° D:27°

[Figure 57]

The image you've uploaded appears to represent a geometric problem involving a circle with a tangent line and intersecting lines forming angles. To solve for angle C given that angle A is 36 degrees, we can use the properties of circles and angles. Here's how we can do it:

- 1.In a circle, a radius that intersects a tangent line at the point of tangency forms a right angle. This means that angle OBA is 90 degrees.
- 2.Triangle OAB is a right triangle with angle OAB being 90 degrees and angle AOB being 36 degrees. Using the fact that the sum of angles in a triangle is 180 degrees, we can find angle ABO.
- 3.Angle ABO is equal to 180 degrees minus the sum of angles AOB and OAB, which is 180−90−36=54 degrees.
- 4.180−90−36=54
- 5.Angle ABC is equal to angle ABO because they are both formed by the tangent line AB and the radius OB. So angle ABC is also 54 degrees.
- 6.Angle ACB is in the same segment as angle AOB, and angles in the same segment are equal. Therefore, angle ACB is also 36 degrees. So, angle C, which is angle ACB, is 36 degrees. Therefore, the correct choice is: A: 36°

[Figure 58]

[Figure 59]

G-LLaVA

Since AB is tangent to circle O at point B, therefore, angle ABO = 90 °. Since angle A = 36 °, therefore, angle BOA = 54°. Therefore, according to the inscribed angle theorem, angle C = 1/2 angle BOA = 27 °. Therefore, the correct choice is D.

[Figure 60]

[Figure 61]

[Figure 62]

GPT4-V

[Figure 63]

[Figure 64]

As shown in the figure, AB is the diameter of the semicircle, angle ABC = 50.0, point D is the midpoint of arc AC, then angle DAB is equal to () Choices: A:40° B:50° C:65° D:70°

To solve this problem, we need to apply some geometry principles. Here's the step-bystep solution:

- 1. Given that AB is the diameter of the semicircle, ∠ACB is a right angle because the angle subtended by a diameter in a semicircle is always a right angle.
- 2. Since ∠ACB is a right angle and ∠ABC = 50° , we can find ∠BAC using the fact that the angles in a triangle sum to 180° .
- 3. ∠BAC = 180° - 90° - 50° = 40° .
- 4. Point D is the midpoint of arc AC , which means AD = DC and ∠DAB = ∠DAC because angles subtended by the same arc are equal.
- 5. In a semicircle, the angle at the circumference subtended by the diameter is 90° . Since D is the midpoint of arc AC , ∠ADC would be half of 90° , which is 45° .
- 6. Now, ∠DAC is half of ∠ADC , because AD = DC . So, ∠DAC = ∠DAB = 45° / 2 = 22.5°.
- 7. Since ∠BAC = 40° and ∠DAB is half of that, ∠DAB = 40° / 2 = 20° .

[Figure 65]

[Figure 66]

G-LLaVA

Connecting BD, as shown in the diagram. Because point D is the midpoint of segment AC, that is, CD equals AD. Therefore, angle ABD equals angle CBD. And angle ABC is 50 degrees, so angle ABD is half of 50 degrees, which is 25 degrees. Since AB is the diameter of the semicircle, angle ADB is 90 degrees. Therefore, angle DAB is 90 degrees minus 25 degrees, which is 65 degrees. Therefore, the answer is C.

Therefore, the correct answer is that ∠DAB is 20°. However, this option is not listed in your multiple-choice answers. There might be an error in the question or in the interpretation. Please review the question and the steps provided to ensure that all given information and the steps are accurate.

[Figure 67]

[Figure 68]

[Figure 69]

GPT4-V

[Figure 70]

In parallelogram ABCD, the diagonal AC and BD intersect at point O, angle DAC = 42.0, angle CBD = 23.0, then angle COD is (). Choices: A:61 B:63 C:65 D:67

[Figure 71]

Given:

- - ∠DAC = 42°
- - ∠CBD = 23° In parallelogram ABCD, diagonals bisect each other, so ∠AOD = ∠COD and ∠BOC = ∠COB. Since opposite angles in a parallelogram are equal, we have ∠DAB = ∠DAC = 42° and ∠CDB = ∠CBD = 23° . Using the fact that consecutive angles in a parallelogram sum up to 180° : ∠AOD = 180° - ∠DAB = 180° - 42° = 138° Now, ∠AOD and ∠BOC form a straight line at point O, so: ∠COD + ∠BOC = 180° Given that ∠BOC = 23° , we find ∠COD by subtracting from 180° : ∠COD = 180° - ∠BOC = 180° - 23° = 157° Therefore, ∠COD = 157° .

[Figure 72]

[Figure 73]

G-LLaVA

Since quadrilateral ABCD is a parallelogram, we have AD parallel to BC. Therefore, angle BCA is equal to angle DAC, which is 42 degrees. Angle COD can be calculated as the sum of angle CBD and angle BCA, which is 42 degrees plus 23 degrees, equal to 65 degrees. Therefore, the correct answer is C.

Figure 3: Demonstration of geometric problem solving by GPT-4-V and G-LLaVA.

|Model<br><br>|Input<br><br>|Accuracy (%)|
|---|---|---|
|Random Chance Frequent Guess|-<br><br>|25.0 32.1|
|Geoformer (Chen et al., 2022) UniMath (Liang et al., 2023) G-LLaVA-7B<br><br>w/o alignment phase|Q,I Q,I Q,I Q,I<br><br>|46.8 50.0 64.2 62.8<br><br>|

Table 13: Effectiveness of alignment in the pre-training phase. Top-1 accuracy is reported.

### 7 Conclusion

In this paper, we make the attempt to address the limitations of current MLLMs in solving geometric problems. We propose strategies to enrich the data by leveraging LLMs, resulting in our augmented dataset, Geo170K. With this dataset, our G-LLaVA outperforms GPT-4-V on the geometric split of MathVista benchmark, with as few as 7B parameters. We hope our work provides new insights on improving multimodal LLMs’ ability of solving geometric problems.

### 8 Acknowledgement

We would like to thank Zhenwen Liang for his valuable discussions and insightful feedback.

### References

Chris Alvin, Sumit Gulwani, Rupak Majumdar, and Supratik Mukhopadhyay. 2017. Synthesis of solutions for shaded area geometry problems. In The Thirtieth International Flairs Conference.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A versatile visionlanguage model for understanding, localization, text reading, and beyond.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Jie Cao and Jing Xiao. 2022. An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In Proceedings of the 29th International Conference on Computational Linguistics, pages 1511–1520.

Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. 2022. UniGeo: Unifying geometry logical reasoning via reformulating mathematical expression. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3313–3323.

Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric Xing, and Liang Lin. 2021. GeoQA: A geometric question answering benchmark towards multimodal numerical reasoning. In Findings of

the Association for Computational Linguistics: ACLIJCNLP 2021, pages 513–523, Online. Association for Computational Linguistics.

Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. 2023. Shikra: Unleashing multimodal llm’s referential dialogue magic.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An opensource chatbot impressing gpt-4 with 90%* chatgpt quality.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. 2023. Instructblip: Towards general-purpose vision-language models with instruction tuning.

Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. 2023. Specializing smaller language models towards multi-step reasoning.

Jiahui Gao, Renjie Pi, Yong Lin, Hang Xu, Jiacheng Ye, Zhiyong Wu, Weizhong Zhang, Xiaodan Liang, Zhenguo Li, and Lingpeng Kong. 2023a. Self-guided noise-free data generation for efficient zero-shot learning.

Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, Hongsheng Li, and Yu Qiao. 2023b. Llama-adapter v2: Parameter-efficient visual instruction model.

Google. 2023. Gemini: A family of highly capable multimodal models.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. Tora: A tool-integrated reasoning agent for mathematical problem solving.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556.

Albert Qiaochu Jiang, Sean Welleck, Jin Peng Zhou, Timothee Lacroix, Jiacheng Liu, Wenda Li, Mateja Jamnik, Guillaume Lample, and Yuhuai Wu. 2023. Draft, sketch, and prove: Guiding formal theorem provers with informal proofs. In The Eleventh International Conference on Learning Representations.

Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. 2023. Lisa: Reasoning segmentation via large language model. arXiv preprint arXiv:2308.00692.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models.

Zhenwen Liang, Tianyu Yang, Jipeng Zhang, and Xiangliang Zhang. 2023. Unimath: A foundational and multimodal mathematical reasoner. In EMNLP.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. 2023. Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. 2021. Inter-GPS: Interpretable geometry problem solving with formal language and symbolic reasoning. In The 59th Annual Meeting of the Association for Computational Linguistics (ACL).

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. 2023. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct.

Yu Meng, Jiaxin Huang, Yu Zhang, and Jiawei Han. 2022. Generating training data with language models: Towards zero-shot language understanding. arXiv preprint arXiv:2202.04538.

Maizhen Ning, Qiu-Feng Wang, Kaizhu Huang, and Xiaowei Huang. 2023. A symbolic characters aware model for solving geometry problems. In Proceedings of the 31st ACM International Conference on Multimedia, MM ’23, page 7767–7775, New York, NY, USA. Association for Computing Machinery.

OpenAI. 2023. Gpt-4 technical report.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. 2023. Instruction tuning with gpt-4.

Renjie Pi, Jiahui Gao, Shizhe Diao, Rui Pan, Hanze Dong, Jipeng Zhang, Lewei Yao, Jianhua Han, Hang Xu, Lingpeng Kong, and Tong Zhang. 2023a. Detgpt: Detect what you need via reasoning.

Renjie Pi, Lewei Yao, Jiahui Gao, Jipeng Zhang, and Tong Zhang. 2023b. Perceptiongpt: Effectively fusing visual perception into llm.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision.

Mrinmaya Sachan, Kumar Dubey, and Eric Xing. 2017. From textbooks to knowledge: A case study in harvesting axiomatic knowledge from textbooks to solve geometry problems. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 773–784.

Mrinmaya Sachan and Eric Xing. 2017. Learning to solve geometry problems from natural language demonstrations in textbooks. In Proceedings of the 6th Joint Conference on Lexical and Computational Semantics (* SEM 2017), pages 251–261.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman Castagne, Alexandra Sasha Luccioni, François Yvon, Matthias Galle, et al. 2022. Bloom: A 176bparameter open-access multilingual language model. arXiv preprint arXiv:2211.05100.

Minjoon Seo, Hannaneh Hajishirzi, Ali Farhadi, Oren Etzioni, and Clint Malcolm. 2015. Solving geometry problems: Combining text and diagram interpretation. In Proceedings of the 2015 conference on empirical methods in natural language processing, pages 1466– 1476.

Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, et al. 2022. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. arXiv preprint arXiv:2201.11990.

Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. 2023. Pandagpt: One model to instruction-follow them all.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https:// github.com/tatsu-lab/stanford_alpaca.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language models with self-generated instructions.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903.

Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Kang Liu, and Jun Zhao. 2023. Large language models are better reasoners with self-verification. CoRR, abs/2212.09561.

Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. 2023. The dawn of lmms: Preliminary explorations with gpt-4v(ision).

Jiacheng Ye, Jiahui Gao, Qintong Li, Hang Xu, Jiangtao Feng, Zhiyong Wu, Tao Yu, and Lingpeng Kong. 2022a. Zerogen: Efficient zero-shot learning via dataset generation. In Empirical Methods in Natural Language Processing.

Jiacheng Ye, Jiahui Gao, Zhiyong Wu, Jiangtao Feng, Tao Yu, and Lingpeng Kong. 2022b. ProGen: Progressive zero-shot dataset generation via in-context feedback. In Findings of the Association for Computational Linguistics: EMNLP 2022.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023. Metamath: Bootstrap your own mathematical questions for large language models.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023a. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023b. Mammoth: Building math generalist models through hybrid instruction tuning.

Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Yu Liu, Kai Chen, and Ping Luo. 2023. Gpt4roi: Instruction tuning large language model on region-of-interest.

Xueliang Zhao, Xinting Huang, Wei Bi, and Lingpeng Kong. 2023a. Sego: Sequential subgoal optimization for mathematical problem-solving. arXiv preprint arXiv:2310.12960.

Xueliang Zhao, Wenda Li, and Lingpeng Kong. 2023b. Decomposing the enigma: Subgoal-based demonstration learning for formal theorem proving. arXiv preprint arXiv:2305.16366.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, and Ed Chi. 2022. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models.

