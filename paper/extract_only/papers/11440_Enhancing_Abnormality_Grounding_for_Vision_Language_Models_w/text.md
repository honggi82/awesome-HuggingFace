# arXiv:2503.03278v1[cs.CV]5Mar2025

## Enhancing Abnormality Grounding for Vision Language Models with Knowledge Descriptions

Jun Li1,2, Che Liu4, Wenjia Bai4, Rossella Arcucci4, Cosmin I. Bercea*1,3( ), and Julia A.Schnabel*1,2,3,5( )

1 Technical University of Munich, Germany

- 2 Munich Center for Machine Learning, Germany
- 3 Helmholtz AI and Helmholtz Munich, Germany
- 4 Imperial College London, UK
- 5 King’s College London, UK

{june.li,cosmin.bercea,julia.schnabel}@tum.de

Abstract. Visual Language Models (VLMs) have demonstrated impressive capabilities in visual grounding tasks. However, their effectiveness in the medical domain, particularly for abnormality detection and localization within medical images, remains underexplored. A major challenge is the complex and abstract nature of medical terminology, which makes it difficult to directly associate pathological anomaly terms with their corresponding visual features. In this work, we introduce a novel approach to enhance VLM performance in medical abnormality detection and localization by leveraging decomposed medical knowledge. Instead of directly prompting models to recognize specific abnormalities, we focus on breaking down medical concepts into fundamental attributes and common visual patterns. This strategy promotes a stronger alignment between textual descriptions and visual features, improving both the recognition and localization of abnormalities in medical images.We evaluate our method on the 0.23B Florence-2 base model and demonstrate that it achieves comparable performance in abnormality grounding to significantly larger 7B LLaVA-based medical VLMs, despite being trained on only 1.5% of the data used for such models. Experimental results also demonstrate the effectiveness of our approach in both known and previously unseen abnormalities, suggesting its strong generalization capabilities. The code and model are available here§.

Keywords: Visual Grounding · Large Language Models · Multimodality

### 1 Introduction

Vision-Language Models (VLMs) [1–4] have achieved remarkable success in a variety of visual understanding tasks, such as image captioning [5], visual question

⋆ Shared senior authors. § Link: https://lijunrio.github.io/AG-KD/

###### Benchmark Performance

###### Question: Detect infiltration in the image.

[Figure 1]

| | | | |
|---|---|---|---|
|21.93| |25.50| |
| | | | |
|5.14|7.45| | |
| | | | |
|8.18|10.81| | |
| | | | |
|53.98| | |54.38|
| | | | |
|81.36| | |80.09|
| | | | |
|39.47| | |41.41|
| | | | |
|60.15| | |56.92|

model size (B)

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

infiltration

𝒎𝑨𝑷𝟓𝟎

[Figure 7]

[Figure 8]

emphysema

𝒎𝑨𝑷𝟕𝟓 𝒎𝑨𝑷𝟓𝟎𝟗𝟓 𝑹𝒐𝑫𝒆𝑶𝒕𝒐𝒕𝒂𝒍 𝑹𝒐𝑫𝒆𝑶𝒄𝒍𝒔 𝑹𝒐𝑫𝒆𝑶𝒔𝒉𝒂𝒑𝒆

[Figure 9]

[Figure 10]

atelectasis

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

shape density

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

location

𝑹𝒐𝑫𝒆𝑶𝒍𝒐𝒄

| |
|---|

| |
|---|

RadVLM Ours

- Fig.1: Overview of our approach. We train a 0.23B model on just 16,087 samples (1.5% of the data) and achieve similar or better results than the 7B RadVLM, pre-trained on 1 million samples, by using text descriptions that highlight key visual features of abnormalities.

answering [6], and visual grounding [7]. By jointly modeling both visual and textual representations, these models excel at associating image content with natural language descriptions, enabling them to be highly adaptable across diverse domains. Recent advancements have extended VLMs to the medical imaging domain, where models such as RadVLM [8], MAIRA-2 [9], and ChexAgent [10] have demonstrated significant potential in tasks like radiology report generation, question answering, and abnormality grounding. These models use large-scale paired X-rays image-text datasets [11–14], allowing them to perform different medical tasks, thus providing valuable support to radiologists in diagnosis.

Despite these advancements, abnormality grounding remains a critical yet underexplored task in medical image analysis. Unlike report generation [15–17] and question answering [18–20], which has been extensively studied within the context of VLMs, abnormality grounding requires not only the ability to understand textual queries but also the precise localization of corresponding abnormalities within images. Existing medical VLMs, though powerful, are typically large-scale models that require substantial computational resources and extensive pretraining on diverse datasets. While these models demonstrate strong general performance, their general-purpose design may limit their ability to effectively address specialized tasks such as abnormality grounding on medical images. Moreover, a significant challenge in abnormality grounding stems from the complexity of medical terminology and its weak alignment with visual features. In general visual grounding tasks, objects like “cat” or “dog” possess well-defined and easily recognizable features, facilitating the formation of direct visual-language associations. However, medical abnormalities present more nuanced challenges. Terms like “lung opacity” or “interstitial lung disease” refer to combinations of textural, morphological, and contextual features, none of which map to a single, welldefined visual counterpart. Instead, these terms describe subtle, heterogeneous manifestations that can vary based on clinical context and imaging modalities.

To address these challenges, we introduce a novel approach for abnormality grounding by incorporating decomposed knowledge descriptions tied to visual

Cross-entropy loss for tokens

A. Decompose Knowledge Description

B. Training Process

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

###### Lung opacity:

Transformer Decoders

" Any abnormal […] (including but not limited to consolidation, cavity, fibrosis, nodule, mass,

Transformer Encoders

calcification, interstitial thickening, etc.). "

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

Large Language Model Description of Decomposed Visual Attributes

Shape Color Density Location

Text Tokenizer

Image Encoder

[Figure 45]

|Locate { lung opacity }, which means { "An area of increased density in the lung fields typically appearing as a white or grayish patch " }.|
|---|

[Figure 46]

[Figure 47]

[Figure 48]

Lung opacity: "An area of increased density in the lung fields typically appearing as a white or grayish patch."

- Fig.2: Overview of our method. (A) shows the pipeline for obtaining decomposed knowledge descriptions, (B) presents the model architecture and training process for the abnormality grounding task.

Bigger text version，Color

features as shown in Figure 1. Specifically, we leverage textual descriptions capturing key visual attributes of medical abnormalities, including shape, density, and location—essential for accurate abnormality interpretation in medical images. By explicitly encoding this domain-specific knowledge, we enhance the model’s ability to associate complex medical terms with their corresponding visual features. Our method achieves superior performance compared to state-ofthe-art approaches [8,9], despite utilizing a much smaller model framework with fewer parameters [2]. Additionally, our approach demonstrates that knowledgeenhanced prompts can considerably boost performance in zero-shot settings, even for previously unseen abnormalities. Our key contributions are as follows:

- • We propose a knowledge-enhanced approach for abnormality grounding in VLM training, by using fine-grained, attribute-based textual descriptions to improve visual grounding performance.
- • We show that a small-scale VLM (0.23B parameters) can match the abnormality grounding performance of large-scale models (7B parameters) using only 1.5% of the training data.
- • We show that our approach considerably improves zero-shot generalization, enabling the model to better detect unseen abnormalities, which is essential in low-data scenarios.

### 2 Methods

Problem Setup. Unlike traditional object detection methods [21] that regress bounding box coordinates, VLMs frame detection as an autoregressive sequence generation task. Given an image I ∈ RH×W, where H and W denote its dimensions, and a text prompt T describing the target object, the VLM generates a set of bounding boxes L = {L1,L2,...,LM}. Each bounding box Li = {ℓix

} consists of four discrete tokens representing its top-left and bottom-right corners. Coordinates are normalized to [0,1000] as ℓx = xpixelW ·1000,

,ℓiy

,ℓix

,ℓiy

0

0

1

1

ℓy = ypixelH · 1000 and quantized into a fixed vocabulary, as described in [2]. The bounding box count M equals the number of detected instances. This formula-

tion enables detection to be naturally integrated into language modeling.

Decomposed Knowledge Description We improve VLM performance in detecting and localizing medical abnormalities by breaking down complex medical concepts into key visual attributes, such as shape, location, and density, as shown in Figure 2A. We first retrieve the medical definitions of each abnormality [22], which often lack explicit references to their visual manifestations in medical images. For example, the medical definition of lung opacity is: “Any abnormal focal or generalized opacity or opacities in lung fields (including but not limited to consolidation, cavity, fibrosis, nodule, mass, calcification, interstitial thickening).” While informative, such definitions contain extraneous details that do not emphasize the core visual characteristics of abnormalities. To address this, we define a set of visual attributes commonly used in medical imaging, including {shape, location, density, color}, which are crucial for characterizing an abnormality’s visual appearance. Note that although pixel intensity is a more accurate term in medical imaging, we retain color to ensure compatibility with generalist visionlanguage models that may not recognize the relationship between tissue density and intensity values. With both the definitions and visual attributes at hand, we design a prompting strategy to instruct the language model [23] to generate decomposed knowledge descriptions focusing on the visual aspects of each abnormality. Specifically, we construct an input template that integrates medical definitions with defined visual features. The prompt follows this structure:

Here is the medical definition of [abnormality name]: "[medicaldefinition]." Based on this definition and focusing on the following visual attributes (e.g., shape, location, density, color), provide a brief description of the abnormality.

By prompting the large language model with this query, we obtain descriptions such as: “An area of increased density in the lung fields, typically appearing as a white or grayish patch.” for each abnormality term.

Architecture. As shown in Figure 2B, we use the Florence-2 base [2] as the backbone, which integrates a visual encoder and a multi-modal encoderdecoder. The visual encoder, based on DaViT [24], processes an input image I ∈ RH×W×3 to produce flattened visual token embeddings V ∈ RN×D, where N is the number of visual tokens and D is their dimensionality. Simultaneously, knowledge-decomposed prompts are tokenized into text embeddings. Both the visual and text embeddings are passed through the multi-modal transformer encoder-decoder [25] to generate the final answer. The model generates output through auto-regressive decoding, applying cross-entropy loss to all discrete localization tokens. Finally, the loss is defined as:

N

L = −

log p(yi|{V,T})

i=1

where yi is the target localization token at position i, and p(yi|{V,T}) is the predicted probability distribution over the vocabulary, conditioned on both the visual and textual embedding. We fine-tune the entire model in an end-to-end manner using our decomposed textual knowledge prompts, which break down complex medical concepts into key visual attributes, guiding the model to focus on the core visual characteristics of abnormalities.

### 3 Experiments

Dataset. We trained our method on the VinDr-CXR dataset [22], a large-scale chest X-ray dataset with bounding boxes annotated by radiologists for various abnormalities. To ensure annotation consistency, we applied weighted box fusion [26] to merge overlapping bounding boxes and converted them into localization tokens. Since our task focuses on abnormality grounding, we retained only images with at least one annotated abnormality, resulting in 18,195 imageabnormality pairs, with 16,087 for training and 2,108 for test. To assess the zero-shot generalization capabilities of our method, we conducted experiments on the PadChest-GR dataset [14]. We focused on two zero-shot scenarios: generalization to a new dataset and detection of previously unseen diseases. Following the predefined data split, we selected the test set and converted its bounding box annotations into text-box pairs, resulting in 1,285 image–bounding box pairs. To distinguish between these scenarios, we further divided the test set into two subsets. The first subset, PadChest-known (641 pairs), contains six diseases that overlap with the VinDr-CXR dataset. The second subset, PadChest-unknown (644 pairs), includes diseases not present in VinDr-CXR, enabling us to assess the model’s performance in detecting previously unseen abnormalities.

Benchmark Baselines. We consider two recent state-of-the-art medical VLMs, MAIRA-2 [9] and RadVLM [8], for comparison, both of which are significantly larger (with 30∼56 times more parameters) than our model and trained on extensive multi-source datasets. Table 1 shows that MAIRA-2 is a 13B parameter model trained on 501,825 training samples from a combination of MIMICCXR [11], PadChest [13], and USMix [27]. RadVLM, a 7B parameter model, is trained on an even broader set, including MIMIC-CXR, CheXpert-Plus [28], CheXpert [29], Vindr-CXR, MS-CXR [30], and PadChest-GR [14], with a total of 1,022,742 image-instruction pairs. Our proposed model is trained solely on Vindr-CXR with 16,086 training samples, and has only 0.23B parameters.

Experimental setup. Our method leverages Florence-2-base [2] as the backbone, a 0.23B parameter architecture. We fine-tune the model using its pretrained weights with the Adam optimizer [31], setting the learning rate to 5×10−6 and weight decay to 0.01. Training is conducted with a batch size of 16 and an input resolution of 512×512 on two NVIDIA A6000 GPUs. For comparison, we use the publicly available pre-trained weights of MAIRA-2 and RadVLM and evaluate them on the same test set. In our ablation studies, we first establish a

- Table 1: Comparison on the VinDr-CXR and PadChest-GR datasets. Our method achieves competitive results on both datasets, with the best performance on VinDr-CXR and competitive results in a zero-shot setting on PadChestGR. Best and second-best performances are coloured Green and Yellow . Rloc, Rshape, Rcls, and Rtotal are different aspects of the RoDeO metrics. Methods marked with indicate the dataset had not been seen during training.

[Figure 49]

Test Set Method Params Train. Samp. mAP9550 mAP50 mAP75 Rloc Rshape Rcls Rtotal

[Figure 50]

MAIRA-2 13B 501,825 1.22 4.94 0.32 25.65 17.23 80.13 24.08 RadVLM 7B 1,022,742 8.18 21.93 5.14 60.15 39.47 81.36 53.98 Ours 0.23B 16,087 10.81 25.5 7.45 56.92 41.41 80.92 54.38

VinDr-CXR

MAIRA-2 13B 501,825 8.36 19.17 5.81 33.05 29.68 81.92 37.14 RadVLM 7B 1,022,742 2.53 10.84 0.81 58.61 29.18 79.64 46.16 Ours 0.23B 16,087 2.68 11.07 0.56 57.34 32.48 83.00 49.13

PadCh. Know.

[Figure 51]

baseline model that uses only abnormality labels, without knowledge-enhanced prompts. The final model, in contrast, incorporates these prompts during training, which provide additional visual context and improve abnormality grounding.

Evaluation metrics. We evaluate all models using standard abnormality detection metrics [32], including mean average precision (mAP) at various Intersection over Union (IoU) thresholds: mAP50, mAP50:95, and mAP75. Besides, we use the Robust Detection Outcome (RoDeO) [33], a metric for pathology detection that evaluates classification, shape, and localization for bounding box quality.

### 4 Results

#### 4.1 Comparison with SOTA

Fewer Parameters, Competitive Performance. Table 1 compares the performance of our method with existing state-of-the-art models on the VinDrCXR and PadChest-Known datasets. Despite having only 0.23B parameters, our model consistently outperforms much larger counterparts across multiple key evaluation metrics on the VinDr-CXR dataset. Specifically, our method achieves the highest mAP50 of 25.5%, surpassing the second-best model by 3.57 points. It also attains a mAP75 of 7.45%, outperforming the second-best model by 2.31 points. For mAP50:95, our model reaches 10.81%, exceeding the closest competitor by 2.63 points. Additionally, our method achieves the highest overall RoDeO score of 54.38%. These results demonstrate that, despite having significantly fewer parameters, our knowledge-enhanced model achieves competitive, and in some cases, superior performance compared to much larger models trained on extensive multi-source datasets. Notably, we observe that MAIRA-2 underperforms significantly compared to RadVLM, which can be attributed to MAIRA-2 not being trained on the VinDr-CXR dataset, thus operating in a zero-shot setting. In Figure 3, we also evaluate our method’s performance across individual

100204060800

94.26

79.07

###### Vindr-CXR

###### PadChest-GR Known

1st Ours (14/21) 1st Ours (3/6)

82.62

77.15

63.41

71.95

71.27

68.53

64.28

63.57

60.99

60.94

58.12

56.58

49.78 42.53 41.61

55.19

51.34

50.69

49.76

45.93

45.4

41.82

34.61

36.98

32.66

MAIRA-2 RadVLM Ours

- Fig.3: Performance for each disease class, with the y-axis representing the RoDeo total metric. Our method achieves first place in 14 out of 21 diseases from the VinDr-CXR dataset and 3 out of 6 known diseases from the PadChest-GR dataset. The best performances are highlighted in the callout.

diseases. Our model ranks first in 14 out of 21 diseases on VinDr-CXR, highlighting its ability to effectively detect abnormalities across a wide range of conditions.

Comparable Zero-shot Performance. We evaluate our method on the PadChest-Known dataset, which contains 6 diseases overlapping with the VindrCXR dataset, in a zero-shot setting. Table 1 shows that our method achieves the best performance in RoDeO shape matching, classification, and overall scores, with values of 32.48%, 83.00%, and 49.13%, respectively. Besides, our model ranks second in mAP50:95 with a score of 2.68% and in mAP50 with 11.07%. Our method, though not trained on PadChest, shows competitive performance against MAIRA-2 and RadVLM. We also evaluate the performance for each disease, where our model ranks first in 3 out of 6 classes, demonstrating its good generalization capability, as shown in the right part of the Figure 3.

#### 4.2 Ablation Study

Knowledge-enhanced prompts boost VLM’s performance. Table 2 presents the results of an ablation study comparing the baseline to the knowledgeenhanced method on the Vindr-CXR and PadChest-GR datasets. Our proposed method demonstrates substantial improvements across all evaluation metrics on the Vindr-CXR test set, achieving a mAP50 of 25.5% (vs. 13.26%), mAP75 of 7.45% (vs. 2.76%), and an overall RoDeO score of 54.38% (vs. 45.22%). These results highlight that integrating disease-specific visual knowledge enhances the model’s ability to detect abnormalities effectively, outpacing the baseline by a large margin. We also evaluate both methods’ performance on the PadChestGR Known dataset. While mAP75 shows a slight decrease compared to the baseline, all other metrics exhibit considerable improvement, indicating that the

- Table 2: Ablation study on the effect of knowledge descriptions. Base refers to the Florence-2 model [2], while Ours incorporates knowledge descriptions (KD). We evaluate in-distribution performance on VinDr-CXR and assess zeroshot generalization to an unseen dataset (PadChest-Known, marked by ) and to both an unseen dataset and previously unseen disease classes (PadChestUnknown, marked by ). Best performances are highlighted in green .

[Figure 52]

[Figure 53]

[Figure 54]

Test Set Method mAP9550 mAP50 mAP75 Rloc Rshape Rcls Rtotal Vindr-CXR

Base 4.92 13.26 2.76 44.26 34.36 78.19 45.22 + KD (Ours) 10.81 25.50 7.45 56.92 41.41 80.92 54.38

Pad. Know. Base 1.92 8.34 0.65 48.11 31.19 81.25 44.59

+ KD (Ours) 2.68 11.07 0.56 57.34 32.48 83.00 49.13 Pad. Unkn. Base 0.37 1.48 0.03 38.14 20.90 78.69 32.05

[Figure 55]

###### + KD (Ours) 0.95 3.05 0.29 44.71 22.93 86.12 33.72

[Figure 56]

[Figure 57]

knowledge integration allows the model to generalize better to different datasets. Knowledge-enhanced prompts improve detection of unknown findings. In this section, we further evaluate the performance of our proposed method on the PadChest-Unknown dataset, which contains 18 diseases not present in the Vindr-CXR dataset. Table 2 shows that the knowledge-enhanced method outperforms the baseline across all metrics. Specifically, mAP50 increases from 1.48% to 3.05%, and mAP75 improves from 0.03% to 0.29%. These results suggest that integrating disease-specific knowledge enhances the model’s ability to transfer knowledge to unknown diseases, i.e., not encountered during training, thereby improving zero-shot performance.

### 5 Discussion and Conclusion

We introduce a novel knowledge-enhanced approach to VLMs for abnormality grounding. By integrating fine-grained, decomposed disease-specific visual descriptions, our method demonstrates that smaller, task-specific models can achieve performance comparable to much larger VLMs trained on extensive datasets. This approach substantially improves zero-shot generalization, particularly for previously unseen datasets, making VLMs more effective in low-data settings. Our results further demonstrate that knowledge-enhanced prompts not only boost in-domain performance but also help the model generalize to new diseases, highlighting their potential for real-world medical applications where labeled data is scarce.

While our results demonstrate the effectiveness of our knowledge-enhanced prompts method, several avenues for future research remain. First, expanding the knowledge base to include a broader range of diseases, along with the integration of multimodal data sources, could further enhance the model’s generalization capability. Additionally, integrating decomposed, knowledge-enhanced prompts

with larger, more complex VLMs could push performance boundaries. Larger models trained on more extensive datasets have the potential to benefit from the specialized disease-specific knowledge embedded in the prompts, improving their performance. Finally, exploring dynamic prompt adjustment for each disease could further optimize model performance. Recent studies have shown that prompt engineering, which focuses on emphasizing the most relevant cues for each disease, can significantly enhance VLMs performance.

### References

- 1. OpenAI: ChatGPT can now see, hear, and speak. https://openai.com/index/ chatgpt-can-now-see-hear-and-speak/ (2024), accessed: 2024-11-26
- 2. Xiao, B., Wu, H., Xu, W., et al.: Florence-2: Advancing a unified representation for a variety of vision tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4818–4829 (2024)
- 3. Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., Zhou, J.: Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966 (2023)
- 4. Lu, H., Liu, W., Zhang, B., Wang, B., Dong, K., Liu, B., Sun, J., Ren, T., Li, Z., Yang, H., et al.: DeepSeek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525 (2024)
- 5. Stefanini, M., Cornia, M., et al.: From show to tell: A survey on deep learning-based image captioning. IEEE transactions on pattern analysis and machine intelligence 45(1), 539–559 (2022)
- 6. Lin, Z., Zhang, D., et al.: Medical visual question answering: A survey. Artificial Intelligence in Medicine 143, 102611 (2023)
- 7. Xiao, L., Yang, X., Lan, X., et al.: Towards visual grounding: A survey. arXiv preprint arXiv:2412.20206 (2024)
- 8. Deperrois, N., Matsuo, H., Ruipérez-Campillo, S., Vandenhirtz, M., Laguna, S., Ryser, A., Fujimoto, K., Nishio, M., Sutter, T.M., Vogt, J.E., et al.: RadVLM: A multitask conversational vision-language model for radiology. arXiv preprint arXiv:2502.03333 (2025)
- 9. Bannur, S., Bouzid, K., Castro, D.C., Schwaighofer, A., Thieme, A., Bond-Taylor, S., Ilse, M., Pérez-García, F., Salvatelli, V., Sharma, H., et al.: Maira-2: Grounded radiology report generation. arXiv preprint arXiv:2406.04449 (2024)
- 10. Chen, Z., Varma, et al.: Chexagent: Towards a foundation model for chest X-ray interpretation. arXiv preprint arXiv:2401.12208 (2024)
- 11. Johnson, A.E., Pollard, T.J., Greenbaum, N.R., Lungren, M.P., Deng, C.y., Peng, Y., Lu, Z., Mark, R.G., Berkowitz, S.J., Horng, S.: MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs. arXiv preprint arXiv:1901.07042 (2019)
- 12. Wu, J.T., Agu, N.N., Lourentzou, I., Sharma, A., Paguio, J.A., Yao, J.S., Dee, E.C., Mitchell, W., Kashyap, S., Giovannini, A., et al.: Chest imagenome dataset for clinical reasoning. arXiv preprint arXiv:2108.00316 (2021)
- 13. Bustos, A., Pertusa, A., Salinas, J.M., De La Iglesia-Vaya, M.: Padchest: A large chest x-ray image dataset with multi-label annotated reports. Medical Image Analysis 66, 101797 (2020)

- 14. Castro, D.C., Bustos, A., Bannur, S., Hyland, S.L., Bouzid, K., Wetscherek, M.T., Sánchez-Valverde, M.D., Jaques-Pérez, L., Pérez-Rodríguez, L., Takeda, K., et al.: Padchest-gr: A bilingual chest X-ray dataset for grounded radiology report generation. arXiv preprint arXiv:2411.05085 (2024)
- 15. Li, J., Su, T., Zhao, B., Lv, F., Wang, Q., Navab, N., Hu, Y., Jiang, Z.: Ultrasound report generation with cross-modality feature alignment via unsupervised guidance. arXiv preprint arXiv:2406.00644 (2024)
- 16. Tanida, T., Müller, P., Kaissis, G., Rueckert, D.: Interactive and explainable regionguided radiology report generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7433–7442 (2023)
- 17. Li, J., Li, S., Hu, Y., Tao, H.: A self-guided framework for radiology report generation. In: International Conference on Medical Image Computing and ComputerAssisted Intervention. pp. 588–598. Springer (2022)
- 18. Singhal, K., Tu, T., Gottweis, J., Sayres, R., Wulczyn, E., Amin, M., Hou, L., Clark, K., Pfohl, S.R., Cole-Lewis, H., et al.: Toward expert-level medical question answering with large language models. Nature Medicine pp. 1–8 (2025)
- 19. Li, J., Kim, S.H., Müller, P., Felsner, L., Rueckert, D., Wiestler, B., Schnabel, J.A., Bercea, C.I.: Language models meet anomaly detection for better interpretability and generalizability. In: Medical Image Computing and Computer Assisted Intervention–MICCAI 2024 Workshops. Lecture Notes in Computer Science, vol. 15401, pp. 1–11. Springer Nature Switzerland AG (2025)
- 20. Zhang, X., Wu, C., Zhao, Z., et al.: Pmc-VQA: Visual instruction tuning for medical visual question answering. arXiv preprint arXiv:2305.10415 (2023)
- 21. Jiang, P., Ergu, D., et al.: A review of yolo algorithm developments. Procedia computer science 199, 1066–1073 (2022)
- 22. Nguyen, H.Q., Lam, K., Le, L.T., Pham, H.H., Tran, D.Q., Nguyen, D.B., Le, D.D., Pham, C.M., Tong, H.T., Dinh, D.H., et al.: Vindr-cxr: An open dataset of chest X-rays with radiologist’s annotations. Scientific Data 9(1), 429 (2022)
- 23. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: GPT-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- 24. Ding, M., Xiao, B., Codella, N., et al.: Davit: Dual attention vision transformers. In: European Conference on Computer Vision. pp. 74–92. Springer (2022)
- 25. Waswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A., Kaiser, L., Polosukhin, I.: Attention is all you need. In: NIPS (2017)
- 26. Müller, P., Kaissis, G., Rueckert, D.: Chex: Interactive localization and region description in chest X-ray. In: European Conference on Computer Vision. pp. 92–

111. Springer (2024)

- 27. Demner-Fushman, D., Kohli, M.D., Rosenman, M.B., Shooshan, S.E., Rodriguez, L., Antani, S., Thoma, G.R., McDonald, C.J.: Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association 23(2), 304–310 (2016)
- 28. Chambon, P., Delbrouck, J.B., Sounack, T., Huang, S.C., Chen, Z., Varma, M., Truong, S.Q., Chuong, C.T., Langlotz, C.P.: CheXpert Plus: Augmenting a large chest X-ray dataset with text radiology reports, patient demographics and additional image formats. arXiv preprint arXiv:2405.19538 (2024)
- 29. Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B., Ball, R., Shpanskaya, K., et al.: Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In: Proceedings of the AAAI conference on artificial intelligence. vol. 33, pp. 590–597 (2019)

- 30. Boecking, B., Usuyama, N., Bannur, S., Castro, D.C., Schwaighofer, A., Hyland, S., Wetscherek, M., Naumann, T., Nori, A., Alvarez-Valle, J., et al.: Making the most of text semantics to improve biomedical vision–language processing. In: European Conference on Computer Vision. pp. 1–21. Springer (2022)
- 31. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2018)
- 32. Padilla, R., Netto, S.L., Da Silva, E.A.: A survey on performance metrics for objectdetection algorithms. In: 2020 International Conference on Systems, Signals and Image Processing (IWSSIP). pp. 237–242. IEEE (2020)
- 33. Meissen, F., Müller, P., Kaissis, G., et al.: Robust detection outcome: A metric for pathology detection in medical images. In: Medical Imaging with Deep Learning

