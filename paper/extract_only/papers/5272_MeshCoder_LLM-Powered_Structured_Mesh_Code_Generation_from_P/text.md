# arXiv:2508.14879v2[cs.GR]22Aug2025

## MeshCoder: LLM-Powered Structured Mesh Code Generation from Point Clouds

Bingquan Dai∗1,2, Li Ray Luo∗1, Qihong Tang1,3, Jie Wang1,4, Xinyu Lian1, Hao Xu1,5, Minghan Qin2, Xudong Xu1, Bo Dai1, Haoqian Wang†2, Zhaoyang Lyu†1, Jiangmiao Pang1

1Shanghai Artificial Intelligence Laboratory, Shanghai, China 2Tsinghua University, Beijing, China 3Harbin Institute of Technology, Shenzhen, China 4Beijing Institute of Technology, Beijing, China 5AI Thrust, HKUST(GZ), Guangzhou, China

[Figure 1]

- (a)

[Figure 2]

- (b)

Figure 1: (a) MeshCoder can predict codes and reconstruct 41 categories of objects. (b) MeshCoder takes in point clouds and produce part-segmented meshes by executing the predicted code in Blender. For the dishwasher, we apply transparency to the foremost part to showcase the internal structure.

### Abstract

Reconstructing 3D objects into editable programs is pivotal for applications like reverse engineering and shape editing. However, existing methods often rely on limited domain-specific languages (DSLs) and small-scale datasets, restricting

∗Equal contribution. †Corresponding authors.

Preprint. Under review.

their ability to model complex geometries and structures. To address these challenges, we introduce MeshCoder, a novel framework that reconstructs complex 3D objects from point clouds into editable Blender Python scripts. We develop a comprehensive set of expressive Blender Python APIs capable of synthesizing intricate geometries. Leveraging these APIs, we construct a large-scale paired object-code dataset, where the code for each object is decomposed into distinct semantic parts. Subsequently, we train a multimodal large language model (LLM) that translates 3D point cloud into executable Blender Python scripts. Our approach not only achieves superior performance in shape-to-code reconstruction tasks but also facilitates intuitive geometric and topological editing through convenient code modifications. Furthermore, our code-based representation enhances the reasoning capabilities of LLMs in 3D shape understanding tasks. Together, these contributions establish MeshCoder as a powerful and flexible solution for programmatic 3D shape reconstruction and understanding. Project homepage: https://daibingquan.github.io/MeshCoder.

### 1 Introduction

Inferring shape programs from 3D observations is of great importance for reverse engineering, shape editing, and 3D structure understanding. Prior work [1, 2, 3] has explored this problem by defining Domain-Specific Languages (DSLs) to model geometric and structural properties of objects and training neural networks to map 3D observations to shape programs. However, existing methods struggle to generalize to objects with complex geometry and structure. Two key limitations underlie this gap. First, existing DSLs are constrained to modeling simple primitives (e.g., cubes, spheres, cylinders) and cannot represent real-world objects with intricate parts. Second, training shape-to-code inference models demands large-scale paired datasets of 3D objects and their corresponding code, while such datasets are scarce. Prior work often relies on datasets with limited categories, geometric complexity and part count.

To address these challenges, we introduce MeshCoder, a novel framework for generating Blender Python scripts that reconstruct complex 3D objects into their constituent parts. First, we design a set of expressive Blender Python APIs that are capable of synthesizing intricate geometries beyond simple primitives. For instance, our APIs can create complex shapes by translating a 2D section curve along a specified trajectory, bridging section curves of different shapes, adding bevels or applying Boolean operations on basic shapes, repeating a basic shape in one dimension or two dimensions. With these concise yet powerful Blender Python APIs, we can model highly complex shapes, addressing the limitations of prior DSLs.

Second, we present a novel pipeline to construct a large-scale paired object-code dataset. We begin by synthesizing diverse object parts using our APIs with parametrically sampled parameters, yielding a part-level dataset. A part-to-code inference model is then trained on this dataset to predict code for individual parts. Next, we employ this model to construct a holistic object-code dataset. We use Infinigen-Indoor [4] to generate a dataset of objects, and each object is decomposed into its constituent parts. We use the part-to-code inference model to predict code for each part of an object, and then carefully design rules to concatenate code of all parts to obtain code of the object. This process yields a dataset of approximately 1 million objects spanning 41 categories, with objects up to more than 100 parts. Finally, we train a multimodal large language model (LLM) on this dataset to infer code from 3D objects. We use point clouds as 3D shape representations due to their ease of acquisition, and use a triplane-based tokenizer to transform the input point cloud to a set of fixed-length tokens. These tokens are fed into the LLM to generate Blender Python scripts that replicate input geometries in distinct semantic parts.

We evaluate our approach against existing shape-to-code methods, with experimental results and quantitative metrics demonstrating that our framework significantly outperforms prior work. Furthermore, by representing shapes as executable code, our method facilitates intuitive geometric and topological editing through simple code modifications. This capability enables precise alterations to object geometry and mesh topology, enhancing flexibility in downstream applications. Additionally, we conduct experiments on shape structural and geometric understanding tasks, revealing that our code-based representation improves the reasoning capabilities of large language models (LLMs) when interpreting 3D shapes. In summary, our contributions are outlined as follows:

- • We have developed a comprehensive set of Blender Python APIs, facilitating the modeling of intricate geometries. This enhanced API suite empowers the procedural generation of complex 3D structures, effectively addressing the limitations of traditional domain-specific languages (DSLs) in representing detailed and varied shapes.
- • We propose a pipeline to construct a large-scale paired object-code dataset. Using the dataset we constructed, we can train an shape-to-code inference model.
- • We trained MeshCoder, an Object-to-Code inference framework that generates Blender Python scripts to reconstruct 3D meshes from point clouds in a structured and editable manner. Our model encodes 3D shapes into part-level code, simplifying mesh editing and enhancing LLMs’ understanding of 3D objects.

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Shape token

ShapeTokenizer

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Code token

[Figure 14]

[Figure 15]

Large Language Model

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

Point Cloud

from bpy_lib import * #…… (code ignored) # object name: chair

create_curve(name=‘seat_11’, control_points=[[-0.0,

-0.05, -0.22], …, [-0.0, -0.05, -0.22]],

# part_1: leg

handle_type=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,

create_primitive(name='leg_1', primitive_type='cylinder', location=[-0.19, -0.25, 0.21], scale=[0.03, 0.03, 0.25], rotation=[0.31, 0.23, -0.61, 0.7])

0, 1, 1, 0, 0]) fill_grid(name='seat_11', thickness=0.0796) bevel(name='seat_11', width=0.04, segments=1) #……

#……

# part_14: back decoration

# part_5: leg decoration create_primitive(name='leg decoration_5', primitive_type='cylinder', location=[-0.16, -0.22,

create_primitive(name='back decoration_14',

primitive_type='cube', location=[-0.0, 0.45, 0.25], scale=[0.15, 0.05, 0.03], rotation=[0.0, 0.0, -1.0, 0.06])

-0.0], scale=[0.03, 0.03, 0.21], rotation=[0.24, 0.07, -0.02, -0.97]) #…… # part_11: seat

bevel(name='back decoration_14', width=0.21, segments=2)

- Figure 2: Overview of MeshCoder. The input point cloud is first encoded into shape tokens via a shape tokenizer. These tokens are then fed into a large language model (LLM), which autoregressively generates executable code representing part-based 3D structures. The decoded code specifies object’s name, part identities and names, enabling interpretable and modular reconstruction.

- 2 Related Work Shape programs. Shape programs provide a structured and interpretable framework for representing
- 3D geometry by utilizing domain-specific languages to describe the generative processes of shapes. Early work such as ShapeAssembly [5] introduced explicit shape programs that capture the hierarchical and part-based organization of objects. Subsequent methods, including ShapeCoder [6], PLAD [2], and ShapeLib [7], progressively improved program abstraction, learning efficiency, and scalability with large language models. Other approaches, such as those proposed by Liang [3] and Tian et al. [1], incorporate differentiable rendering or neuro-symbolic reasoning to enhance program inference and execution. While these methods exhibit strong generalization capabilities in composing simple geometric elements like boxes and cylinders, they often struggle to model complex part geometries or generate artist-grade quad meshes, which restricts their application in high-fidelity asset creation. In addition, a range of methods in CAD program generation [8, 9, 10, 11, 12, 13] have explored synthesizing code representations for individual CAD parts. However, these approaches are limited to isolated component generation and lack the capability to model complete multi-part objects with coherent structural relationships.

Part-based Representation. Part-based representations have proven highly valuable in 3D shape analysis and synthesis. Some approaches [14, 15, 16, 17, 18, 19, 20, 21, 22, 23] take a generative approach, assembling objects by combining predefined or learned parts into complete 3D structures. Other methods [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] focus on segmenting 3D objects

into individual parts, enabling more modular and flexible manipulation of shapes. For instance, SAMPart3D [24] introduces a scalable zero-shot 3D part segmentation framework that segments any 3D object into semantic parts at multiple granularities without requiring predefined part label sets as text prompts. PartSLIP [28] explores low-shot part segmentation of 3D point clouds by leveraging a pretrained image-language model, GLIP, transferring rich knowledge from 2D to 3D through GLIP-based part detection on point cloud rendering and a novel 2D-to-3D label lifting algorithm. SATR [26] performs zero-shot 3D shape segmentation via text descriptions by using a zero-shot

- 2D object detector, inferring 3D segmentation from multi-view 2D bounding box predictions by exploiting the topological properties of the underlying surface. Despite these advancements in part segmentation and reconstruction, these methods do not translate segmented parts into executable code representations, limiting their integration into code-driven design workflows.
- 3 Methodology

As shown in Figure 2, we aim to train an object-to-code inference model that takes in a point cloud of an object, and then predict the Blender python scripts of each part of the object. When executing the python scripts in Blender, we can obtain the same object in separated parts. To train such an object-to-code inference model, we need a dataset of paired objects and the corresponding codes. To obtain such a dataset, we first train a part-to-code inference model that predicts code for a single part on our synthetic dataset of paired parts and the corresponding codes. Then, given a dataset of objects separated in different parts, we use the trained part-to-code inference model to predict code for every part of an object. Finally, we concatenate the codes of every part of the object and obtain the code of the object. Now, we have a dataset of paired objects and the corresponding codes, and are ready to train the object-to-code inference model.

We explain the key steps described above in details in the following sections. First, we explain how to synthesize a dataset of paired parts and the corresponding codes in Section 3.1. Then, we describe the training procedure of the part-to-code inference model in Section 3.2. Next, we use the part-to-code inference model to obtain the code of an entire object in Section 3.3. Finally, we train the object-to-code inference model in Section 3.4.

#### 3.1 Part Dataset

We aim to generate a dataset of paired part shapes and codes. To do so, we implement probabilistic programs to generate Blender Python scripts, and obtain the corresponding shape by executing the code in Blender. We carefully design these probabilistic programs and ensure that the shapes generated are within the range [−1,1]3. There are several types of shapes that we generate, as illustrated in Figure 3. We explain them in the following paragraphs.

Primitive. Primitives are a set of fundamental geometric shapes, consistent with those defined in Blender. Specifically, we consider five basic shapes: cube, cylinder, UV sphere, cone, and torus. Each primitive is parameterized by three attributes: location (location ∈ R3), rotation (rotation ∈ H), and scale (scale ∈ R3), where H denotes the space of unit quaternions. The location specifies the shape’s position in 3D space, rotation defines its orientation via quaternions, and scale determines the shape’s size along its local axes. Examples of Primitives can be found in the first row of Figure 3.

Translation. Translation is defined as the geometry obtained by sweeping a 2D cross-sectional shape along a 3D trajectory curve. As illustrated in the second row of Figure 3, during this translation process, the tangent direction of the 3D trajectory remains perpendicular to the 2D shape, and the size of the section shape can change along the 3D trajectory.For a more detailed explanation, please refer to A.1. To implement this, we first define a 2D shape using a set of control points (i.e., spatial coordinates), and then specify a 3D trajectory curve in a similar manner. Specifically, our experiments consider five types of cross-sectional shapes: rectangles, circles, circular arcs, polygons, and Bézier curves. For the trajectories, we define six forms: straight lines, polylines, circles, circular arcs, rectangles, and Bézier curves. Notably, this method also allows a 2D shape to rotate around an axis to form a solid of revolution, making it suitable for modeling objects such as bottles and plates.

Bridge loop. Although the Translation method is capable of generating certain complex objects, it remains constrained in several ways. For instance, in the Translation operation, the 2D cross-

[Figure 50]

- Figure 3: Visualization of basic geometric shape types and their corresponding code. For each shape category, the code shown corresponds to the first example.

sectional shape is always orthogonal to the tangent direction of the trajectory. Moreover, the section shape is allowed to change only in scale, without any deformation in its geometry. To address this limitation, we introduce an alternative method for constructing geometries, namely the Bridge Loop. This geometry is constructed by first generating a sequence of 2D shapes and then connecting their corresponding vertices to form a continuous 3D geometry. Some cases can be seen in the third row of Figure 3. The Bridge Loop approach enables the creation of more complex geometries compared to those achievable via Translation alone. For a more detailed explanation, please refer to A.1.

Boolean. Boolean geometries refer to geometries formed by applying Boolean operations—namely union, intersection, and difference—to two or more of the fundamental shape categories defined in Section 3.1. The union operation enables the construction of complex composite geometries, while the difference operation is used to generate geometries with holes or indentations. We can see some examples and their corresponding codes in the fourth row of Figure 3.

Array. When a particular type of primitive geometry appears repeatedly in a regular pattern, we do not invoke the construction function for each primitive individually, as this would result in lengthy code. Instead, we employ an Array method to construct the entire structure collectively. Specifically, we define two types of Arrays: 1D Arrays, where a geometry is repeatedly instantiated along a curve, and 2D Arrays, where repetition occurs across a plane. Cases of this type can be seen in the last row of Figure 3.

#### 3.2 Part-to-code Inference Model

After constructing the dataset of paired code y and mesh M, we sample a point cloud x ∈ RN×3 from each mesh M, where N is the number of points in the point cloud. We train a part-to-code inference model h that takes in a point cloud x and predict the corresponding code y. The inference model consists of two modules: The shape tokenizer model and a fintuned LLM. The tokenizer model takes in the point cloud x and outputs a set of fixed length tokens z ∈ RL×D, where L is the number of shape tokens and D is the dimension of each token. We set D to the same dimension as the word embeddings in the LLM. Thereafter, the LLM takes in the shape tokens z and then predict y, the code of the point cloud x. We train the shape tokenizer model and finetune the LLM at the same time using the cross-entropy loss for the prediction of the next token in the shape code y. We use Llama-3.2-1B as the base LLM and finetune it using LoRA.

[Figure 51]

- Figure 4: Architecture of the shape tokenizer. We first project the point cloud into the triplane and obtain triplane features. The triplane features are patchified and reshaped into a 1D sequence, and fed into transformer blocks to obtain triplane tokens. Finally, we use a set of learnable tokens to aggregate information from triplane tokens via cross-attention.

The shape tokenizer model. We explain the detailed structure of the shape tokenizer model. As shown in Figure 4, the shape tokenizer model transforms a point cloud x ∈ RN×3 to a set of fixed length tokens z ∈ RL×D. We first project the point cloud x to a triplane and obtain triplane feature u ∈ R3×H×W×D

1, where H,W are the height and width of the planes, and D1 is the dimension of the triplane feature. The coordinates of each point is fed to a shared MLP and a feature of dimension D1 is obtained. We project each point’s feature to the three perpendicular planes according to the point’s position. Features projected to the same pixel are aggregated by max-pooling. Pixels that do not correspond to any point are filled with zeros. After obtaining the triplane feature u, we patchify it and reshape it into a 1D sequence v ∈ R(3·H/f·W/f)×D

1, where f is the patch size. We then feed the sequence v to a set of transformer blocks and outputs v′ ∈ R(3·H/f·W/f)×D

1. Next, to compress the number of tokens fed into the LLM, we use a learnable set of tokens w ∈ RL×D

2 to aggregate information from v′ using cross attention:

CrossAttn(Transformer(w),v′,v′), (1)

where Transformer denotes a transformer block, CrossAttn(Q,K,V ) denotes a cross attention block, and Q,K,V are query, key, value, respectively. By feeding w to a set of these cross attention blocks, we obtain tokens w′ ∈ RL×D

2 that contain information about the point cloud x. Finally, we use an

MLP to transform the dimension of w′ from D2 to D and obtain shape tokens z ∈ RL×D, where D is the dimension of the word embeddings in the LLM. Now, the shape tokens z can be readily fed to the LLM and predict the code corresponding to the point cloud x.

#### 3.3 Assemble Parts to Objects

After training the part-to-code inference model h, we can use it to obtain the code of an object. Given a dataset of objects, in which each object O is separated into its constituent parts O = {qi|i = 1,2,··· ,M}, where qi is the i-th part of object O, and M is the number of parts of the object O. We also assume that each part qi has its semantic label. We can use the part-to-code inference model h to obtain the code of each part. Specifically, we first normalize each part qi to the cube [−1,1]3 according to its minimum bounding box and obtain the shape qi′. Then we use the part-to-code inference model h to obtain its code yi′ = h(qi′). We then implement algorithms to transform the relevant numerical parameters in the code yi′ to the original location, scale, and pose of qi′ and obtain the code yi of the original shape qi. Finally, we concatenate the codes of all parts of the object, add semantic information to the code for each part, and obtain the code of the object y = {yi|i = 1,2,··· ,M}. When concatenating the code, we sort each part based on its spatial position. Specifically, we assign an index to each part following a spatial order from bottom to top, left to right, and front to back. An overview of this pipeline is illustrated in Figure 5. During code inference, the part point cloud qi is first transformed into a canonical space using a rotation matrix R, translation matrix T, and scaling factor s, resulting in qi′. The trained part-to-code inference model h generates the code yi′ of qi′. yi′ is then transformed back to the original pose and scale using the inverse of R, T, and s, and we obtain the code yi of qi.

#### 3.4 Object-to-code Inference Model

After obtaining the code y of each object O in the dataset, we can use them to train an object-to-code inference model. Our object-to-code inference model has the same structure as the part-to-code

[Figure 52]

- Figure 5: Pipeline of object-level code dataset construction using the part-to-code inference model. For each part point cloud qi, the code inference module independently predicts its corresponding code yi. All part codes yi are then concatenated to form the complete object code. We also add meaningful semantic information to the object code following the template shown in the figure. The complete code of the example chair is shown in Figure 2.

inference model described in Section 3.2. We initialize the weights of the object-to-code inference model as the weights of the trained part-to-code inference model, and use the same training method in Section 3.2 to train the object-to-code inference model. It is worth noting semantic information in the ground-truth code of objects enables the object-to-code inference model to learn the semantic structure of objects, and facilitate 3D shape understanding.

### 4 Experiment

#### 4.1 Datasets

- 4.1.1 Synthetic Part Dataset

To facilitate the training of our part-to-inference model, we first constructed a synthetic part dataset. Specifically, we utilized functions from our basic shape code library, randomly sampling their parameters based on manually defined distributions to generate paired data of synthetic parts and corresponding code. This process yielded 1.5 million point cloud–code pairs for primitive shapes, 3 million for Translation-based parts, 1.5 million for Bridge Loop structures, 1.5 million for Boolean operations, and 2.4 million for Array-based constructions. In total, our constructed part dataset comprises around 10 million point cloud–code pairs. We partitioned the dataset into 70% for training, 15% for validation, and 15% for testing.

- 4.1.2 Object Dataset

We trained our model on the Infinigen Indoor [4] dataset. Infinigen Indoor is a procedural framework for generating synthetic 3D indoor objects, where each generated instance is automatically composed by its corresponding parts. We have made extensive modifications to the original Infinigen codebase to enable it to produce both individual components and their complete assemblies. Using this framework, we constructed a synthetic dataset comprising 41 common object categories, generating 1 million object-code pairs in total. We partitioned the dataset into training, validation, and test sets, following the same split strategy as the Synthetic Part Dataset. For more details, please refer to the A.1.

#### 4.2 Implementation Details

We conduct training and evaluation on the Infinigen Indoor datasets [36]. For the part-to-code reconstruction model, we adopt the AdamW optimizer and train it for 20 epochs on NVIDIA A100

GPUs with a batch size of 512, and a learning rate of 10−4. We evaluate the model at every epoch and select the checkpoint with the lowest L2 Chamfer Distance (CD) loss. Then we initialize the weights of the object-to-code reconstruction model with the weights of the trained part-to-code reconstruction model, and train the model on Infinigen Indoor dataset for 10 epochs, with a batch size of 256, and a learning rate of 10−4. The checkpoint with the lowest CD loss is selected. For additional training details and the parameter settings of the models, please refer to A.3 and A.2.

#### 4.3 Reconstruction Performance

For reconstruction performance, we compare our method with two representative shape-to-code baselines, Shape2Prog [1] and PLAD [2]. Figure 6 illustrates visualization comparisons of results. We adopt IoU and L2 CD as our evaluation metrics. Specifically, we voxelize the model’s predicted outputs into 323 grids and compute the IoU between the predicted and ground truth voxel grids. In parallel, we sample point clouds from both the predicted outputs and the ground truth, and calculate the Chamfer Distance between the two point clouds. Regarding the number of points and normalization, please refer to the appendix A.4. In Table 1, we present reconstruction metrics for some specific object categories as well as the overall performance across the entire dataset. It can be observed that our method consistently outperforms the baselines in both IoU and CD metrics. Complete results for all categories in each dataset are provided in A.4. We conducted a series of ablation studies to evaluate the impact of various components within our model. For comprehensive details on these experiments, please refer to A.4.

- Table 1: Quantitative comparison of reconstruction performance between MeshCoder and baselines.

CD(×10−2)↓ IoU (%) ↑ Lamp Chair Sofa TableDining Toilet All Lamp Chair Sofa TableDining Toilet All

Method

Shape2Prog 25.44 1.30 2.14 1.03 7.51 6.01 16.96 49.68 65.29 71.26 51.14 45.03 PLAD 1.40 2.26 1.52 5.52 2.30 1.87 69.58 40.93 81.33 58.43 62.61 67.62 MeshCoder 0.004 0.060 0.027 0.024 0.022 0.063 86.23 81.87 93.81 88.14 89.10 86.75

[Figure 53]

Figure 6: Qualitative comparison of reconstruction performance between MeshCoder and baselines. MeshCoder can accurately reconstruct objects with intricate parts and complex structures.

#### 4.4 Shape Editing

MeshCoder facilitates the transformation of 3D shapes into high-level, human-readable code representations, significantly enhancing the interpretability and editability of complex geometries. This capability enables intuitive and precise modifications through code-based interventions. Our shape editing encompasses two primary categories: geometric editing and topological editing. As illustrated in Figure 7, geometric editing can be performed by adjusting function calls or modifying specific parameters within the generated code. For instance, we can adjust the parameters of the code to convert a square tabletop into a larger circular one. Additionally, topological editing, which is illustrated in Figure 8 such as adjusting mesh resolution, can be achieved by modifying designated parameters within the code, allowing for control over the mesh’s complexity and surface detail. This

[Figure 54]

- Figure 7: Parameter modification in the code conveniently to alter the geometric shape. Left: Change tabletop from square to circular. Right: Make the bathtub shallower.

[Figure 55]

- Figure 8: Mesh resolution adjustment by modifying the resolution parameters in the code. The figure depicts results with progressively increasing resolution from left to right.

code-centric approach streamlines the process of modifying 3D models, making it more accessible and efficient for applications requiring iterative design and customization. Additionally, it empowers users to adjust the model resolution according to their desired balance between storage requirements and mesh quality. For additional results and details, please refer to A.5 in the appendix.

#### 4.5 Shape Undertanding

MeshCoder is capable of predicting object codes with rich semantic information. These codes effectively capture structural and geometric details, making them valuable for shape understanding. By inputting the predicted codes into GPT, we can assist it in comprehending object structures. We conduct experiments on shape understanding, with an example illustrated in Figure 9. Additional results and details are given in A.6 in the appendix.

### 5 Limitations

[Figure 56]

import bpy

…… # code ignored # object name: office chair # part_1: wheel

create_primitive(name='wheel_1', …

…… # code ignored # part_4: wheel create_primitive(name=‘wheel_4’, …

…… # code and prompt ignored

Question: How many wheels are there in the chair?

[Figure 57]

Answer: 4 GPT-4

Figure 9: The pipeline of conducting experiments on shape understanding.

Although our method achieves significant advancements in category diversity, geometric complexity, and reconstruction accuracy compared to existing approaches, it primarily targets human-made objects. The applicability of code-based representations to organic forms, such as animals and humans, remains underdeveloped. We reserve this as a direction for future research.

### 6 Conclusion

In this work, we present MeshCoder, a comprehensive framework that translates 3D point cloud data into editable Blender Python scripts, enabling detailed reconstruction and intuitive editing of complex 3D objects. By developing a robust set of Blender Python APIs, we facilitate the modeling of intricate geometries. Leveraging these APIs, we constructed a large-scale dataset pairing 3D objects with their corresponding code representations, decomposed into semantic parts. Subsequently, we trained a multimodal large language model (LLM) capable of generating executable Blender scripts from point cloud inputs. Our approach not only achieves superior performance in shape-to-code reconstruction tasks but also enhances the reasoning capabilities of LLMs in 3D shape understanding.

By representing shapes as structured code, MeshCoder offers a flexible and powerful solution for programmatic 3D shape reconstruction and editing, paving the way for advanced applications in reverse engineering, design, and analysis.

### References

- [1] Yonglong Tian, Andrew Luo, Xingyuan Sun, Kevin Ellis, William T. Freeman, Joshua B. Tenenbaum, and Jiajun Wu. Learning to infer and execute 3d shape programs. In International Conference on Learning Representations, 2019.
- [2] R Kenny Jones, Homer Walke, and Daniel Ritchie. Plad: Learning to infer shape programs with pseudo-labels and approximate distributions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9871–9880, 2022.
- [3] Yichao Liang. Learning to infer 3d shape programs with differentiable renderer. arXiv preprint arXiv:2206.12675, 2022.
- [4] Alexander Raistrick, Lingjie Mei, Karhan Kayan, David Yan, Yiming Zuo, Beining Han, Hongyu Wen, Meenal Parakh, Stamatis Alexandropoulos, Lahav Lipson, Zeyu Ma, and Jia Deng. Infinigen indoors: Photorealistic indoor scenes using procedural generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21783–21794, June 2024.
- [5] R Kenny Jones, Theresa Barton, Xianghao Xu, Kai Wang, Ellen Jiang, Paul Guerrero, Niloy J Mitra, and Daniel Ritchie. Shapeassembly: Learning to generate programs for 3d shape structure synthesis. ACM Transactions on Graphics (TOG), 39(6):1–20, 2020.
- [6] R Kenny Jones, Paul Guerrero, Niloy J Mitra, and Daniel Ritchie. Shapecoder: Discovering abstractions for visual programs from unstructured primitives. ACM Transactions on Graphics (TOG), 42(4):1–17, 2023.
- [7] R Kenny Jones, Paul Guerrero, Niloy J Mitra, and Daniel Ritchie. Shapelib: designing a library of procedural 3d shape abstractions with large language models. arXiv preprint arXiv:2502.08884, 2025.
- [8] Pradeep Kumar Jayaraman, J. Lambourne, Nishkrit Desai, Karl D. D. Willis, Aditya Sanghi, and Nigel Morris. Solidgen: An autoregressive model for direct b-rep synthesis. ArXiv, abs/2203.13944, 2022. URL https://api.semanticscholar.org/CorpusID: 247761924.
- [9] Mohammad Sadil Khan, Elona Dupont, Sk Aziz Ali, Kseniya Cherenkova, Anis Kacem, and Djamila Aouada. Cad-signet: Cad language inference from point clouds using layer-wise sketch instance guided attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4713–4722, June 2024.
- [10] Rundi Wu, Chang Xiao, and Changxi Zheng. Deepcad: A deep generative network for computeraided design models. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 6752–6762, 2021. doi: 10.1109/ICCV48922.2021.00670.
- [11] Jingwei Xu, Zibo Zhao, Chenyu Wang, Wen Liu, Yi Ma, and Shenghua Gao. Cad-mllm: Unifying multimodality-conditioned cad generation with mllm, 2024.
- [12] Xiang Xu, Pradeep Kumar Jayaraman, Joseph G Lambourne, Karl DD Willis, and Yasutaka Furukawa. Hierarchical neural coding for controllable cad model generation. In International Conference on Machine Learning, pages 38443–38461, 2023.
- [13] Jianyu Wu, Yizhou Wang, Xiangyu Yue, Xinzhu Ma, Jingyang Guo, Dongzhan Zhou, Wanli Ouyang, and Shixiang Tang. Cmt: A cascade mar with topology predictor for multimodal conditional cad generation, 2025.
- [14] Juil Koo, Seungwoo Yoo, Minh Hieu Nguyen, and Minhyuk Sung. Salad: Part-level latent diffusion for 3d shape generation and manipulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14441–14451, 2023.

- [15] Anran Liu, Cheng Lin, Yuan Liu, Xiaoxiao Long, Zhiyang Dou, Hao-Xiang Guo, Ping Luo, and Wenping Wang. Part123: part-aware 3d reconstruction from a single-view image. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024.
- [16] Minghao Chen, Roman Shapovalov, Iro Laina, Tom Monnier, Jianyuan Wang, David Novotny, and Andrea Vedaldi. Partgen: Part-level 3d generation and reconstruction with multi-view diffusion models. arXiv preprint arXiv:2412.18608, 2024.
- [17] Yuhang Huang, SHilong Zou, Xinwang Liu, and Kai Xu. Part-aware shape generation with latent 3d diffusion of neural voxel fields. arXiv preprint arXiv:2405.00998, 2024.
- [18] Lin Gao, Jie Yang, Tong Wu, Yu-Jie Yuan, Hongbo Fu, Yu-Kun Lai, and Hao Zhang. Sdm-net: Deep generative network for structured deformable mesh. ACM Transactions on Graphics (TOG), 38(6):1–15, 2019.
- [19] Zhijie Wu, Xiang Wang, Di Lin, Dani Lischinski, Daniel Cohen-Or, and Hui Huang. Sagnet: Structure-aware generative network for 3d-shape modeling. ACM Transactions on Graphics (TOG), 38(4):1–14, 2019.
- [20] Kaichun Mo, Paul Guerrero, Li Yi, Hao Su, Peter Wonka, Niloy Mitra, and Leonidas J Guibas. Structurenet: Hierarchical graph networks for 3d shape generation. arXiv preprint arXiv:1908.00575, 2019.
- [21] Rundi Wu, Yixin Zhuang, Kai Xu, Hao Zhang, and Baoquan Chen. Pq-net: A generative part seq2seq network for 3d shapes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 829–838, 2020.
- [22] George Kiyohiro Nakayama, Mikaela Angelina Uy, Jiahui Huang, Shi-Min Hu, Ke Li, and Leonidas Guibas. Difffacto: Controllable part-based 3d point cloud generation with cross diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14257–14267, 2023.
- [23] Dmitry Petrov, Matheus Gadelha, Radomír Mˇech, and Evangelos Kalogerakis. Anise: Assemblybased neural implicit surface reconstruction. IEEE Transactions on Visualization and Computer Graphics, 2023.
- [24] Yunhan Yang, Yukun Huang, Yuan-Chen Guo, Liangjun Lu, Xiaoyang Wu, Edmund Y Lam, Yan-Pei Cao, and Xihui Liu. Sampart3d: Segment any part in 3d objects. arXiv preprint arXiv:2411.07184, 2024.
- [25] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16259–16268, 2021.
- [26] Ahmed Abdelreheem, Ivan Skorokhodov, Maks Ovsjanikov, and Peter Wonka. Satr: Zero-shot semantic segmentation of 3d shapes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15166–15179, 2023.
- [27] Zongji Wang and Feng Lu. Voxsegnet: Volumetric cnns for semantic part segmentation of 3d shapes. IEEE transactions on visualization and computer graphics, 26(9):2919–2930, 2019.
- [28] Minghua Liu, Yinhao Zhu, Hong Cai, Shizhong Han, Zhan Ling, Fatih Porikli, and Hao Su. Partslip: Low-shot part segmentation for 3d point clouds via pretrained image-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21736–21746, 2023.
- [29] Yuchen Zhou, Jiayuan Gu, Xuanlin Li, Minghua Liu, Yunhao Fang, and Hao Su. Partslip++: Enhancing low-shot 3d part segmentation via multi-view instance segmentation and maximum likelihood estimation. arXiv preprint arXiv:2312.03015, 2023.
- [30] Yuheng Xue, Nenglun Chen, Jun Liu, and Wenyun Sun. Zerops: High-quality cross-modal knowledge transfer for zero-shot 3d part segmentation. arXiv preprint arXiv:2311.14262, 2023.

- [31] Ardian Umam, Cheng-Kun Yang, Min-Hung Chen, Jen-Hui Chuang, and Yen-Yu Lin. Partdistill: 3d shape part segmentation by vision-language model distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3470–3479, 2024.
- [32] George Tang, William Zhao, Logan Ford, David Benhaim, and Paul Zhang. Segment any mesh: Zero-shot mesh part segmentation via lifting segment anything 2 to 3d. arXiv preprint arXiv:2408.13679, 2024.
- [33] Anh Thai, Weiyao Wang, Hao Tang, Stefan Stojanov, James M Rehg, and Matt Feiszli. 3× 2: 3d object part segmentation by 2d semantic correspondences. In European Conference on Computer Vision, pages 149–166. Springer, 2024.
- [34] Ziming Zhong, Yanyu Xu, Jing Li, Jiale Xu, Zhengxin Li, Chaohui Yu, and Shenghua Gao. Meshsegmenter: Zero-shot mesh semantic segmentation via texture synthesis. In European Conference on Computer Vision, pages 182–199. Springer, 2024.
- [35] Xiangyang Zhu, Renrui Zhang, Bowei He, Ziyu Guo, Ziyao Zeng, Zipeng Qin, Shanghang Zhang, and Peng Gao. Pointclip v2: Prompting clip and gpt for powerful 3d open-world learning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2639–2650, 2023.
- [36] Kaichun Mo, Shilin Zhu, Angel X. Chang, Li Yi, Subarna Tripathi, Leonidas J. Guibas, and Hao Su. PartNet: A large-scale benchmark for fine-grained and hierarchical part-level 3D object understanding. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

### A Appendix of MeshCoder: LLM-Powered Structured Mesh Code Generation from Point Clouds

- A.1 Datasets

- A.1.1 The principles of Translation and Bridge Loop

[Figure 58]

Figure 10: A schematic illustration of the principles of Translation and Bridge Loop. In the Translation module, the wireframe of the resulting mesh is shown as a cross-sectional circle is translated along a yellow trajectory. In the Bridge Loop module, the wireframe of the mesh is constructed by connecting the vertices of two 2D shapes.

As illustrated in the figure 10, in the Translation operation, a 2D cross-sectional shape (a circle in this example) and a 3D trajectory curve must first be defined. The Translation process generates a mesh by sweeping the 2D shape along the 3D trajectory. During this sweep, the cross-section remains perpendicular to the tangent direction of the trajectory at all times, and only uniform scaling (either enlargement or reduction) of the cross-section is permitted.

In contrast, the Bridge Loop operation begins with two predefined 2D shapes. By connecting the corresponding vertices of these two shapes, a mesh can be constructed. This method places no constraints on the types of 2D shapes used—meaning the two shapes can differ, such as a circle and a irregular closed shape in this example. Moreover, it imposes no restrictions on the relative orientations of the shapes. As a result, Bridge Loop overcomes the limitations of Translation, which requires the cross-section to align with the trajectory’s tangent direction. This enables Bridge Loop to generate more complex geometries that Translation cannot produce.

- A.1.2 Part datasets

[Figure 59]

Figure 11: The Fill Grid type, Spoon type and Fork type in basic shape code library

For certain shapes that are difficult to represent using the method we defined in Section 3.1, we introduce three additional categories: the Fill Grid type, Spoon type and Fork type. As illustrated in the Figure 11. For the Fill Grid type, we first construct a closed 3D shape (as opposed to the

- 2D cross-sectional shape used in Translation), fill it to form a surface, and then extrude it along its

normal direction to generate the final mesh. For the Spoon and Fork type, we draw inspiration from the implementation in Infinigen Indoor [4] and design dedicated procedural functions tailored for their generation.

We present two core functions from our codebase: the complete implementation for creating primitives (Figure 22) and the complete implementation for creating curves (Figure 23). The full codebase can be found in the supplementary materials.

More examples of parts and their corresponding complete code implementations are provided in Figures 12, 13, 14, 15, and 16.

[Figure 60]

- Figure 12: Examples of Primitive and complete code. And the code corresponds to the first two objects shown in the figure.

[Figure 61]

- Figure 13: Examples of Translation and complete code. And the code corresponds to the first two objects shown in the figure.

[Figure 62]

- Figure 14: Examples of Boolean and complete code. And the code corresponds to the first two objects shown in the figure.

Taking the Primitive type as an example, we describe how to use functions from the basic shape code library to generate a synthetic part dataset. We begin by randomly selecting the type of primitive to generate (e.g., cube, cylinder, etc.). Next, for each axis, we independently uniform sample a value x from the range [−2,2], and then set the corresponding scale as 10x. To determine the orientation of the shape, we uniformly sample a direction from a unit sphere and a roll angle from a uniform distribution. Once the orientation is fixed, we scale the shape uniformly along all three axes based on the size of its bounding box. Specifically, we ensure that the longest edge of the bounding box lies within the range [1,2]. Finally, we assign the shape a random position within the

- 3D space such that the entire shape remains within the [−1,1] bounds. For other shape types beyond Primitive, we follow a similar approach by randomly assigning values to the relevant parameters.

[Figure 63]

- Figure 15: Examples of Bridge Loop and complete code. And the code corresponds to the first two objects shown in the figure.

[Figure 64]

- Figure 16: Examples of Array and complete code. And the code corresponds to the first two objects shown in the figure.

#### A.1.3 Object datasets

For assembling part codes into a complete program, we provide a full example containing the complete code, as shown in Figure 17. Regarding the ordering strategy used when assembling parts into a complete object, we adopt a consistent spatial heuristic to determine part sequence. Specifically, parts are arranged from bottom to top, left to right, and front to back. To implement this, we divide the 3D space into a 32 × 32 × 32 grid and assign each part a characteristic grid cell that serves as the basis for sorting. The characteristic grid cell of a part is defined as follows: among all grid cells that the part occupies, we first select the one with the smallest z-coordinate. If multiple candidates share the same z-value, we choose the one with the smallest x-coordinate. If a tie still exists, we select the one with the smallest y-coordinate. Parts are then sorted based on the lexicographic order of these characteristic grid cells, which determines their final sequence within the object.

It is important to note that for each object, the prerequisite for successfully constructing its corresponding code lies in the ability of our part-to-code inference model to accurately infer all of its individual parts. We consider a part to be successfully inferred if the Chamfer Distance (CD) between the predicted point cloud and the ground truth is below 5 × 10−3. Therefore, when constructing the object-code pairs dataset, we only include objects for which all constituent parts meet this criterion. Objects with any part failing to meet this standard are discarded. As a result, the number of successfully constructed object-code pairs is smaller than the total number of objects in the original Infinigen dataset. In fact, the original Infinigen dataset we use contains 1.57 million object instances, from which we successfully construct 1 million shape-code pairs. For training and evaluation, we split the full Infinigen dataset into 70% for training, 15% for testing, and 15% for validation. Accordingly, MESHCODER is trained only on the subset of the shape-code pairs that fall within the training portion of the Infinigen dataset. In contrast, the baseline models are trained on the full set of objects in the training split of the original Infinigen dataset. Importantly, all evaluation results for our method and the baselines are reported on the same test set, i.e., the testing split of the complete Infinigen dataset.

[Figure 65]

Figure 17: A complete code example of converting part codes into a full object program.

[Figure 66]

Figure 18: Detailed configuration of the shape tokenizer.

#### A.2 Model architecture

We explain the detailed structure of the shape tokenizer. As illustrated in the Figure 18, we first project the input point cloud of shape Rn×3 onto three orthogonal planes to obtain tri-plane features with shape R3×128×128×32. With a patch size set to 16 × 16, these tri-plane features are encoded into tokens and fed into Transformer blocks, where the resulting representation is mapped to v and used as the key and value (K, V ) inputs. Meanwhile, a set of learnable tokens with shape R128×1024 are used as queries in a self and cross attention module. After passing through 12 layers of self and cross attention, we obtain output tokens of shape R128×1024, which are then projected to the final representation of shape R128×2048 via an MLP.

#### A.3 More training details

For the part-to-code reconstruction model, we adopt the AdamW optimizer and train it for 20 epochs on 64 NVIDIA A100 GPUs for about a week with a batch size of 512, and a learning rate of 10−4. We evaluate the model at every epoch and select the checkpoint with the lowest L2 Chamfer Distance (CD) loss. Then we initialize the weights of the object-to-code reconstruction model with the weights of the trained part-to-code reconstruction model, and train the model on Infinigen Indoor dataset for 10 epochs, with a batch size of 256, and a learning rate of 10−4. It is trained on 64 NVIDIA A100 GPUs for about 2 days. The checkpoint with the lowest CD loss is selected.

To further enhance the robustness and generalization ability of the object-to-code inference model, we apply data augmentation techniques. Specifically, we perform random rotation and scaling on the objects. Additionally, during training, we randomly sample the number of points in each point cloud within the range of 4096 to 16384, and add Gaussian noise to further perturb the input. MeshCoder is trained and evaluated on a unified dataset that aggregates all object categories.

#### A.4 Complete experiment result of Shape Reconstruction

For MeshCoder, during inference, each object is represented by a point cloud containing 16,384 points. Given the input point cloud, the object-to-code inference model is able to predict the corresponding Blender Python script code. The resulting code is then executed to generate a corresponding mesh. We uniformly sample 100,000 points from the generated mesh and compute the Chamfer Distance (CD) to the input point cloud using the L2 norm.

Given two point sets P and Q, each of size 100,000, the L2 Chamfer Distance is defined as:

1 |P| x∈P

CD(P,Q) =

1 |Q| y∈Q

∥x − y∥22 +

∥y − x∥22.

min

min

y∈Q

x∈P

To evaluate IoU, we voxelize both the ground-truth mesh and the predicted mesh into grids of resolution 323, and compute the voxel-based Intersection-over-Union (IoU) as:

IoU = |Vpred ∩ Vgt| |Vpred ∪ Vgt|

,

where Vpred and Vgt denote the sets of occupied voxels in the predicted and ground-truth voxel grids, respectively.

For baseline methods, which take voxel grids as input and output voxel grids, we first voxelize the ground-truth mesh into a 323 grid and feed it into the baseline models. The predicted voxel grid is then compared to the input voxelized ground truth to compute IoU. Additionally, we extract a mesh from the predicted voxel grid using the Marching Cubes algorithm and uniformly sample 100,000 points from the resulting mesh surface. These sampled points, along with the ground-truth point cloud, are then both uniformly scaled to fit within the [−1,1]3 volume. Finally, the Chamfer Distance is computed between the two normalized point clouds using the L2 norm.

It’s noticed that for each object category, we independently train the baseline models, according to their official code, resulting in category-specific checkpoints. These models are then evaluated on the corresponding test sets for each category.

The quantitative comparison of reconstruction metrics between MeshCoder and baseline methods across all object categories is summarized in Table 2 and Table 3. Some additional examples of object reconstruction results and their complete code can be referred to Figure 24, 25, 26.

In addition to evaluating our object-to-code inference model, we also perform a quantitative assessment of our part-to-code inference model. Specifically, for each category described in Section 3.1, we construct a test set consisting of 10,000 samples. We evaluate the model’s performance using the CD and voxel IoU metrics on these test sets. The results, shown in Table 4, demonstrate strong performance across all categories, with low CD values and high IoU scores, indicating that our part-to-code inference model is highly effective in generating accurate code representations for individual parts.

- Table 2: Comparison of reconstruction metrics across all categories. Chamfer Distance (CD) and IoU is shown in percentage (%).

Category L2 CD(×10−2) Voxel IoU (%)

MeshCoder PLAD Shape2prog MeshCoder PLAD Shape2prog

ArmChair 0.04 2.31 4.44 94.33 78.79 62.74 BarChair 0.03 2.23 2.55 88.73 74.96 58.23 Bathtub 0.09 1.22 2.45 78.70 74.50 42.94 BeverageFridge 0.22 1.12 12.63 88.03 82.13 39.13 Bottle 0.01 1.08 6.34 88.65 65.58 40.24 Bowl 0.02 1.43 6.29 89.93 60.02 25.60 CeilingClassicLamp 0.02 1.98 3.94 96.13 76.01 59.07 CeilingLight 0.03 3.46 1.32 65.83 40.61 44.97 CellShelf 0.01 1.93 9.40 94.67 59.02 22.30 Chair 0.06 2.26 1.30 81.87 40.93 49.68 Chopsticks 0.03 1.38 21.06 82.24 55.68 11.25 Cup 0.06 1.40 7.35 85.96 62.03 29.47 DeskLamp 0.02 1.76 8.77 80.28 64.31 25.35 Dishwasher 0.13 1.44 3.01 88.37 84.44 46.69 FloorLamp 0.00 2.13 22.97 85.96 66.89 17.16 Fork 0.14 0.34 8.40 58.86 89.28 11.03 Hardware 0.01 0.62 8.45 89.87 83.96 23.56 Jar 0.03 0.76 1.39 79.12 69.67 41.51 Lamp 0.00 1.40 25.44 86.23 69.58 16.96 LargeShelf 0.02 0.82 5.15 88.08 60.70 16.81 Lid 0.05 1.83 2.39 73.22 63.47 50.11 LiteDoor 0.03 1.36 5.75 94.75 36.91 18.71 LouverDoor 0.07 1.40 16.17 89.46 37.43 20.94 Microwave 0.07 1.44 11.04 91.72 55.65 49.38 OfficeChair 0.03 1.44 2.63 78.41 55.65 46.91 PanelDoor 0.04 1.31 6.50 94.60 37.18 20.94 Plate 0.04 0.96 1.07 72.70 70.72 60.05 SidetableDesk 0.01 0.67 4.50 93.23 91.75 35.75 SimpleBookcase 0.03 1.78 2.89 92.14 65.14 33.79 SimpleDesk 0.01 2.12 25.39 88.68 93.80 45.79 Sofa 0.03 1.52 2.14 93.81 81.33 65.29 Spoon 0.67 0.37 4.09 74.00 87.04 18.92 TableCocktail 0.02 2.59 5.93 88.47 60.49 25.19 TableDining 0.02 5.52 1.03 88.14 58.43 71.26 Toilet 0.02 2.30 7.51 89.10 62.61 51.14 TriangleShelf 0.01 2.30 12.61 88.75 62.61 30.59 TV 0.04 1.53 3.41 87.80 72.69 34.14 TVStand 0.01 0.78 13.50 91.26 73.78 22.57 Vase 0.30 0.73 19.10 72.26 89.95 60.94 Window 0.14 0.59 3.73 87.36 84.21 64.64 Wineglass 0.06 0.98 6.83 88.36 73.96 28.56

All (Avg.) 0.06 1.87 6.00 86.75 67.62 45.03

#### A.5 Complete experiment result of Shape Editing

We additionally present two examples of shape editing along with their complete code implementations. In Figure 27, we modify the thickness of the chair legs and armrests by adjusting the scale parameter. In Figure 28, we change the mesh resolution of a plate by modifying the resolution parameter.

#### A.6 Complete experiment result of Shape Understanding

When presented with a 3D point cloud of an object as input, MeshCoder can infer the corresponding code for the object. Upon execution of this code in Blender, the geometry of the object can be

Table 3: Comparison of standard deviation of reconstruction metrics across all categories.

Category CD IoU

MeshCoder PLAD Shape2prog MeshCoder PLAD Shape2prog

ArmChair 1.51 × 10−3 9.35 × 10−3 1.51 × 10−2 4.62 × 10−2 6.48 × 10−2 5.28 × 10−2 BarChair 1.82 × 10−4 1.18 × 10−2 9.90 × 10−3 8.19 × 10−2 1.00 × 10−1 8.30 × 10−2 Bathtub 6.93 × 10−4 1.02 × 10−2 6.70 × 10−3 1.31 × 10−1 1.91 × 10−1 8.57 × 10−2 BeverageFridge 4.84 × 10−3 2.81 × 10−3 3.44 × 10−2 1.13 × 10−1 6.23 × 10−2 6.64 × 10−2 Bottle 7.56 × 10−5 6.80 × 10−3 6.00 × 10−2 1.13 × 10−1 7.05 × 10−2 6.90 × 10−2 Bowl 4.83 × 10−5 5.13 × 10−3 8.78 × 10−3 8.11 × 10−2 6.24 × 10−2 2.35 × 10−2 CeilingClassicLamp 7.33 × 10−7 7.66 × 10−4 9.86 × 10−4 3.39 × 10−5 2.96 × 10−3 3.82 × 10−2 CeilingLight 1.79 × 10−6 3.90 × 10−3 4.44 × 10−3 3.10 × 10−2 5.08 × 10−2 6.93 × 10−2 CellShelf 3.37 × 10−5 1.94 × 10−2 6.95 × 10−2 9.65 × 10−2 1.34 × 10−1 9.45 × 10−2 Lamp 2.20 × 10−5 9.05 × 10−3 2.74 × 10−1 1.56 × 10−1 6.87 × 10−2 1.18 × 10−1 Chair 1.09 × 10−3 1.04 × 10−2 4.52 × 10−3 1.05 × 10−1 9.17 × 10−2 6.72 × 10−2 Chopsticks 3.64 × 10−3 1.31 × 10−2 1.85 × 10−1 1.87 × 10−1 1.00 × 10−1 1.01 × 10−1 Cup 1.59 × 10−3 5.79 × 10−3 3.60 × 10−2 9.84 × 10−2 6.80 × 10−2 6.98 × 10−2 DeskLamp 7.62 × 10−4 8.60 × 10−3 4.55 × 10−2 1.30 × 10−1 7.21 × 10−2 6.01 × 10−2 Dishwasher 9.66 × 10−3 2.69 × 10−3 2.39 × 10−2 1.27 × 10−1 4.74 × 10−2 8.82 × 10−2 FloorLamp 1.23 × 10−4 2.09 × 10−2 2.54 × 10−1 1.68 × 10−1 4.92 × 10−2 1.12 × 10−1 Fork 8.81 × 10−3 2.14 × 10−3 8.57 × 10−2 2.14 × 10−1 1.25 × 10−1 6.55 × 10−2 Hardware 2.20 × 10−4 3.07 × 10−3 4.48 × 10−2 1.21 × 10−1 1.02 × 10−1 1.34 × 10−1 Jar 1.40 × 10−4 2.44 × 10−3 6.11 × 10−3 1.44 × 10−1 6.31 × 10−2 8.98 × 10−2 LargeShelf 1.79 × 10−4 4.65 × 10−3 5.12 × 10−2 1.53 × 10−1 8.67 × 10−2 7.09 × 10−2 Lid 8.89 × 10−4 1.09 × 10−2 1.95 × 10−2 1.55 × 10−1 1.22 × 10−1 1.23 × 10−1 LiteDoor 5.79 × 10−3 4.39 × 10−3 2.88 × 10−2 1.44 × 10−1 6.32 × 10−2 9.69 × 10−2 LouverDoor 4.67 × 10−3 4.84 × 10−3 9.23 × 10−2 1.65 × 10−1 6.82 × 10−2 1.53 × 10−1 Microwave 3.92 × 10−3 2.43 × 10−2 3.15 × 10−2 7.26 × 10−2 1.34 × 10−1 1.65 × 10−1 OfficeChair 1.72 × 10−4 7.35 × 10−3 2.95 × 10−2 8.97 × 10−2 1.06 × 10−1 1.05 × 10−1 PanelDoor 9.05 × 10−3 4.79 × 10−3 3.74 × 10−2 1.50 × 10−1 7.17 × 10−2 1.09 × 10−1 Plate 1.73 × 10−4 6.40 × 10−3 5.78 × 10−3 1.70 × 10−1 1.29 × 10−1 1.74 × 10−1 SidetableDesk 5.11 × 10−5 3.52 × 10−3 5.37 × 10−2 9.64 × 10−2 5.83 × 10−2 1.23 × 10−1 SimpleBookcase 3.66 × 10−3 6.54 × 10−3 7.01 × 10−3 1.08 × 10−1 9.62 × 10−2 6.06 × 10−2 SimpleDesk 4.29 × 10−5 9.90 × 10−2 1.72 × 10−1 1.68 × 10−1 6.43 × 10−2 8.00 × 10−2 Sofa 1.35 × 10−3 5.78 × 10−3 6.99 × 10−3 6.61 × 10−2 7.32 × 10−2 6.37 × 10−2 Spoon 5.64 × 10−2 1.63 × 10−3 4.59 × 10−2 2.26 × 10−1 8.26 × 10−2 9.81 × 10−2 TableCocktail 2.03 × 10−4 2.85 × 10−2 3.10 × 10−2 1.09 × 10−1 2.11 × 10−1 8.68 × 10−2 TableDining 3.31 × 10−3 7.18 × 10−2 6.16 × 10−3 1.55 × 10−1 1.64 × 10−1 9.41 × 10−2 Toilet 1.09 × 10−4 8.44 × 10−3 1.99 × 10−2 4.22 × 10−2 5.24 × 10−2 5.41 × 10−2 TriangleShelf 3.60 × 10−5 9.30 × 10−3 9.47 × 10−2 1.03 × 10−1 6.63 × 10−2 1.08 × 10−1 TV 6.25 × 10−4 1.74 × 10−3 1.02 × 10−2 1.66 × 10−1 2.84 × 10−2 6.68 × 10−2 TVStand 2.59 × 10−5 1.45 × 10−3 5.63 × 10−2 1.31 × 10−1 9.48 × 10−2 5.92 × 10−2 Vase 9.97 × 10−3 3.44 × 10−3 1.05 × 10−1 2.68 × 10−1 2.48 × 10−2 4.00 × 10−2 Window 9.57 × 10−3 3.51 × 10−3 6.61 × 10−2 1.81 × 10−1 1.14 × 10−1 1.88 × 10−1 Wineglass 1.40 × 10−2 3.12 × 10−3 3.86 × 10−2 1.04 × 10−1 5.39 × 10−2 6.99 × 10−2

All (Std.) 2.92 × 10−3 2.49 × 10−2 7.23 × 10−2 1.25 × 10−1 1.94 × 10−1 1.92 × 10−1

obtained. Notably, the comments within the code encompass a variety of semantically rich cues, such as the object’s identity and the specifics of each component. The primary aim of this experiment is to highlight that our model can assist existing large language models, like GPT - 4, in understanding the structure of 3D objects. We provide the inferred code to GPT - 4 and then inquire about the geometry or structure of the object, as showed in Figure 19, Figure 20 and Figure 21. GPT - 4 is able to generate relevant responses based on the code inferred by our model. This demonstrates that our model possesses capabilities in understanding the geometry and structure of 3D objects and can aid large - scale models such as GPT in addressing such questions. However, our model does have limitations. Currently, the code inferred by our model solely contains geometric information of the object and does not include color information. As a result, it is unable to answer questions pertaining to color.

Table 4: Quantitative evaluation of the part-to-code inference model across different part categories. CD is reported in 10−2, and IoU is reported in percentage.

#### Category CD (×10−2) IoU (%)

Primitive 0.18 94.81 Boolean 0.03 96.13 Array 0.70 78.90 Bridge Loop 0.14 89.16 Translation 0.17 83.45

[Figure 67]

- Figure 19: Experiments on how GPT-4o can understand shape through given dishwasher code.

[Figure 68]

##### Figure 20: Experiments on how GPT-4o can understand shape through given office chair code.

[Figure 69]

##### Figure 21: Experiments on how GPT-4o can understand shape through given cell shelf code.

[Figure 70]

##### Figure 22: Implementation of the function for creating primitives

[Figure 71]

##### Figure 23: Implementation of the function for creating curves

[Figure 72]

##### Figure 24: An example of sofa. The input is a point cloud of a sofa, and the figure shows the code inferred by the object-to-code inference model, as well as the resulting mesh generated by executing the inferred code.

[Figure 73]

##### Figure 25: An example of bathtub. The input is a point cloud of a bathtub, and the figure shows the code inferred by the object-to-code inference model, as well as the resulting mesh generated by executing the inferred code.

[Figure 74]

##### Figure 26: An example of toilet. The input is a point cloud of a toilet, and the figure shows the code inferred by the object-to-code inference model, as well as the resulting mesh generated by executing the inferred code.

[Figure 75]

##### Figure 27: By modifying the scale parameters of the leg and arm parts, we adjust their thickness. The highlighted sections indicate the changes made.

[Figure 76]

##### Figure 28: By modifying the resolution parameter, we change its resolution. The highlighted sections indicate the changes made.

