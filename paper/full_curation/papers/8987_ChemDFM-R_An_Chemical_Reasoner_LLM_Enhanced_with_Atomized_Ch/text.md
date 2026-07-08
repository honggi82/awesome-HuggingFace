# arXiv:2507.21990v4[cs.CE]14Apr2026

## ChemDFM-R: A Chemical Reasoning LLM Enhanced with Atomized Chemical Knowledge

Zihan Zhao∗,1, Ziping Wan∗,3, Lu Chen1,2,3,4,†, Xuanze Lin5, Shiyang Yu1, Situo Zhang1, Da Ma1, Zichen Zhu1, Danyang Zhang1, Huayang Wang1, Zhongyang Dai3, Liyang Wen3, Bo Chen3†, Xin Chen3 and Kai Yu1,3,4

1X-LANCE Lab, School of Computer Science, Shanghai Jiao Tong University, Shanghai, China 2Shanghai Innovation Institution, Shanghai, China 3Suzhou Laboratory, Suzhou, China 4Jiangsu Key Lab of Language Computing, Suzhou, China 5School of Chemistry and Chemical Engineering, Shanghai Jiao Tong University, Shanghai, China ∗Equal contributions †Corresponding author chenlusz@sjtu.edu.cn; chenb@szlab.ac.cn

Atomized chemical knowledge, such as functional group information of molecules and reactions, plays a pivotal intermediate role in the reasoning process that connects molecular structures with their properties and reactivities. While large language models (LLMs) have achieved impressive progress, the absence of atomized chemical knowledge results in their superficial understanding of chemistry and limited chemical reasoning capabilities. In this work, to tackle this problem, we develop a Chemical Reasoning LLM, ChemDFM-R. We first construct a comprehensive dataset of atomized chemical knowledge, ChemFG, annotating the presence of functional groups in molecules and the changes of functional groups during chemical reactions, to enhance the model’s understanding of the fundamental principles and internal logic of chemistry. Then, we propose a mixed-source distillation method that initializes the model’s reasoning capability with limited distilled data, and develop a four-stage training pipeline to equip the model with atomized chemical knowledge and chemical reasoning logic. Experiments on diverse chemical benchmarks demonstrate that ChemDFM-R achieves cutting-edge performance while providing interpretable, rationale-driven outputs, surpassing both the general-domain LLMs and domain-specific chemical LLMs. Moreover, ChemDFM-R achieves comparable or superior performance compared with cutting-edge commercial LLMs, such as o4-mini. Further case studies illustrate how explicit reasoning chains significantly improve the model’s reliability, transparency, and practicality in real-world human-AI collaboration scenarios.

- 1 Introduction

In the chemical world, functional groups play a pivotal role. As fine-grained atomized chemical knowledge, the presence and transformations of functional groups serve as critical connectors between molecular structures and high-level chemical phenomena, such as reactions and physicochemical properties. For example, the phenol, which consists of a phenyl group bonded to a hydroxy group, is water-soluble, while the toluene, which substitutes the hydrophilic hydroxy group of the phenol with the hydrophobic methyl group, is water-insoluble. These functional groups constitute essential tools for learning chemical principles, understanding the chemical world, and predicting experimental outcomes.

Recently, with the remarkable capabilities and performance demonstrated by large language models (LLMs) [1,

- 2, 3], great success has been achieved in constructing reasoning LLMs in the general domain [4, 5, 6, 7]. Beyond enhancing overall model performance, the reasoning-before-answering pattern directly demonstrates how and why the LLM arrives at the answer, thereby markedly improving the reliability and interpretability of the LLM’s response. However, existing LLMs generally exhibit insufficient mastery of atomized chemical knowledge owing to the shortage of relevant training data. Even existing chemical LLMs [8, 9, 10, 11, 12] primarily use literature text and high-level phenomena to enhance their chemical knowledge, while failing to adequately decompose knowledge into underlying mechanisms and atomized chemical knowledge. As a

[Figure 1]

- Figure 1 | The overview of ChemFG and ChemDFM-R. a. The Overview of ChemFG. We construct a comprehensive functional group identification toolkit, ChemFG-Tool. The figure demonstrates how the atomized knowledge is annotated and the data composition of the resulting corpus, ChemFG. b. The Overview of Training Pipeline. Our training pipeline is composed of four stages: 1) domain pre-training with the atomized-knowledge-enhanced corpus, ChemFG; 2) instruction tuning on the diversified chemical instruction tuning dataset we construct; 3) Mixed-source distillation that could initialize the reasoning capability with limited distilled data; 4) reinforcement learning with DAPO. c. The Overview of Results. This figure highlights the superior performance of ChemDFM-R on molecule-centric and reaction-centric tasks compared with current chemical LLMs, general non-reasoning LLMs, and general reasoning LLMs, spanning from open-source models to cutting-edge commercial models.

result, their understanding of chemistry often remains superficial, leading to limited performance on chemical tasks. Moreover, since atomized knowledge serves as a critical intermediate node in chemical reasoning, current general reasoning LLMs struggle to achieve efficient and accurate reasoning for chemical problems. Consequently, their rationales cannot provide sufficient reliability and verifiability.

To better leverage atomized chemical knowledge for building a chemical reasoning LLM, two key challenges must be addressed. 1) Despite the availability of large-scale molecular and reaction databases, comprehensive repositories of atomized chemical knowledge are still missing. Existing datasets predominantly operate at the molecular level, directly presenting the properties or reactivities of molecules. They do not explicitly encode atomized chemical knowledge, such as the presence and transformations of functional groups, to describe these high-level phenomena. In addition, dedicated tools for constructing such atom-level knowledge remain underdeveloped. 2) Distillation data for chemical reasoning based on atomized chemical knowledge are difficult to obtain efficiently. As a common method to initialize the reasoning capability of LLMs, distillation usually involves gathering rationales from advanced reasoning LLMs, such as DeepSeek-R1 [5] and o3-mini [13], and training student models using supervised finetuning. However, owing to the limited understanding of atomized knowledge and chemical logic, even powerful general-domain reasoning models will highly probably fail to

generate accurate and in-depth reasoning for chemical problems.

In this work, we focus on addressing the two aforementioned challenges in the chemistry domain and develop a chemical reasoning LLM, ChemDFM-R. Specifically, to tackle the first challenge, we develop a comprehensive functional group identification toolkit and build a large-scale atomized chemical knowledge database with it; to tackle the second challenge, we enhance the generation of distilled data with atomized knowledge and propose a mixed-source distillation method that enables effective distillation training with limited distilled data. The overview of our proposed method is shown in Figure 1.

Firstly, we present ChemFG, a large-scale chemical corpus consisting of both text-based knowledge from chemical literature and atomized chemical knowledge focusing on the presence of functional groups in molecules and the changes of functional groups during reactions. We develop a toolkit, ChemFG-Tool, to comprehensively identify functional groups from molecules and reactions. ChemFG-Tool can effectively identify 241 types of functional groups while handling the collision (e.g., the carbonyl group inside a carbamate should not be identified as a ketone or an ordinary ester). The corpus contains over 101 billion tokens from 12 million literature, 30 million molecules, and 7 million reactions.

Secondly, we develop a comprehensive training pipeline to enhance atomized chemical knowledge and teach models how to reason with it. Specifically, the training pipeline consists of four stages: 1) large-scale domain pre-training on ChemFG, 2) instruction tuning on diverse chemical tasks with diverse instructions, 3) mixed-source distillation that could effectively initialize models’ reasoning capability with limited distilled data, and 4) reinforcement learning with Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) [14].

Thirdly, we introduce ChemDFM-R, an atomized-knowledge-enhanced chemical reasoning LLM, and evaluate it on SciKnowEval [15] and ChemEval [16]. Extensive experiments demonstrate that ChemDFM-R significantly outperforms both general-domain LLMs and domain-specific chemical LLMs of similar size. On the more difficult benchmark, ChemEval, ChemDFM-R even achieves higher performance (73.8%) than the cutting-edge commercial LLMs, GPT-4o (63.3%), DeepSeek-R1 (57.6%), and o4-mini (67.3%), highlighting the effectiveness of the atomized chemical knowledge and our proposed training pipeline. Moreover, we conduct extensive human evaluations to assess the quality of the generated rationales and the potential of ChemDFM-R to facilitate reliable human-AI collaborations. The results show that ChemDFM-R produces more rational and efficient explanations than existing reasoning LLMs. These findings demonstrate that, with the enhancement of atomized chemical knowledge and dedicated training for chemical reasoning, ChemDFM-R can generate high-quality rationales that facilitate deeper understanding and verification of its answers, thereby substantially enhancing its reliability and interpretability.

### 2 Results

#### 2.1 The Construction of ChemFG

In the field of chemistry, functional groups serve as the bridge between molecular structures, properties, and reactivities, making them one of the most critical intermediate reasoning steps in chemical reasoning. However, existing training corpora of LLMs often lack detailed information on molecular functional groups, preventing models from directly and precisely learning this atomized chemical knowledge. Therefore, we collect a functional-group-centered domain pretraining corpus, ChemFG, which consists of data from three sources: literature, molecules, and reactions. The functional groups identification pipeline and basic statistics of ChemFG are shown in Figure 1a with details provided in Section A.1 of the Supplementary Information.

Functional Group Identification. Despite the Internet-scale publicly available molecule and reaction corpora, there are no existing databases that describe the correspondence between functional groups and molecules or reactions. To tackle this issue, we develop a functional group identification toolkit, ChemFG-Tool, based on thermo library [17] by extending its embedded SMARTS [18] list from 83 types of functional groups to 241 and improving its algorithms. With the help of ChemFG-Tool, we annotate the functional groups of all our domain-pretraining molecule data. Further details are provided in Section A.2 of the Supplementary Information.

As for reactions, we annotate the changes of functional groups during reactions with the following process.

First, with the help of atom mapping annotations provided by the USPTO-FULL dataset, we identify the reaction centers as the atoms that are involved in bond changes during reactions. Based on these reaction centers and our functional group identification toolkit, ChemFG-Tool, we identify the functional groups of the reactants that directly participate in the reaction and those of the product that directly result from the reaction. Finally, the reaction can be described as a functional group transformation, where reacting functional groups are converted into product functional groups. Besides functional groups, there are other structural changes during reactions that are equally important, including ring breaking, ring forming, and bond changes outside functional groups. Therefore, we also construct tools to identify these changes in a similar manner.

Quality Control. To ensure the annotation quality of functional groups, we hire three graduate-level chemical experts to conduct manual inspections. Firstly, all the experts agree that the extended SMARTS list has already covered the most common functional groups. For molecules, our tool’s annotation accuracy of 100 random samples reaches 98%, with errors primarily due to corner cases such as rare functional groups or complex interactions between functional groups and aromatic rings. For the annotation of reactions, our tool achieves 89% accuracy when tested with 100 random samples. The errors mainly arise from invalid reactions or wrong atom mapping annotations. Examples and analyses of the error are demonstrated in Section A.3 of the Supplementary Information.

- 2.2 The Overview of ChemDFM-R

To develop ChemDFM-R, we built upon Qwen2.5-14B [19], an advanced general-purpose LLM. As outlined in Figure 1b, the training pipeline of ChemDFM-R can be divided into two parts: 1) Atomized Chemical Knowledge Enhancement, where the basic general LLM is trained with atomized chemical knowledge; 2) Chemical Rationale Learning, where the model’s chemical reasoning capability is enhanced.

Atomized chemical knowledge enhancement consists of two stages: Domain Pretraining and Instruction Tuning. During domain pretraining, the model is trained on the ChemFG corpus using the next-token-prediction loss, during which the model learns the basic chemical knowledge from the literature and functional-group breakdowns. For instruction tuning, we construct a dataset with 16 tasks spanning from name translation and literature question-answering (QA) to molecular property ordering and reaction completion. Moreover, we diversify the task instructions extensively to enhance the model’s generalizability. Based on Qwen2.5-14B, atomized chemical knowledge enhancement results in an intermediate model named ChemDFM-I.

Chemical rationale learning consists of two stages: Mixed-Source Distillation and Reinforcement Learning. To efficiently and effectively initialize the model’s reasoning ability, we propose a specialized distillation method called mixed-source distillation. Specifically, instead of using the distilled chain of thought (COT) alone during the training, we use 1) the data from the instruction tuning dataset to maintain the chemical knowledge and capabilities; 2) pseudo-reasoning data to enhance the model’s functional-group-breakdown capability; and

- 3) distilled COT to introduce general reasoning patterns and abilities to the model. During reinforcement learning, we use the DAPO [14] algorithm with a rule-based reward system to unify the capabilities learned from the three data resources and further enhance the model’s reasoning capability. Based on ChemDFM-I, chemical rationale learning further produces the final model, ChemDFM-R.

- 2.3 ChemDFM-R enhances performance on chemical benchmarks

We evaluate ChemDFM-R and the baseline models on two of the most popular and comprehensive benchmarks specifically designed for assessing the chemical capabilities of LLMs: SciKnowEval [15] and ChemEval [16]. Given the large number of tasks included in SciKnowEval (19 tasks) and ChemEval (35 tasks), to facilitate fair and clear comparison, we categorized the tasks into three groups: text-centric, molecule-centric, and reaction-centric tasks. Details of the task categorization are provided in Section F of the Supplementary Information.

First, we show the effectiveness of our training pipeline by comparing the performances of ChemDFM-R with those of 1) Qwen2.5-14B-Instruct [19], which is the general-domain instruction tuning model of Qwen2.5-14B, and 2) ChemDFM-I, which incorporates atomized chemical knowledge enhancement but precedes the stage of chemical rationale learning. The quantitative results are illustrated in Figure 2a, while examples of the

[Figure 2]

###### Figure 2 | Performance comparison between Qwen2.5-14B-Instruct, ChemDFM-I, and ChemDFM-R. a. The average performance on different types of tasks of Qwen2.5-14B-Instruct, ChemDFM-I, and ChemDFM-R. b. The performance differences between Qwen-2.5-14B-Instruct, ChemDFM-I, and ChemDFM-R on each individual task of SciKnowEval and ChemEval.

ChemDFM-R’s rationales are analyzed in Section E of the Supplementary Information.

As showcased in Figure 2a, ChemDFM-R consistently outperforms Qwen2.5-14B-Instruct on both SciKnowEval and ChemEval, demonstrating that our specialization pipeline has successfully improved the model’s chemical capabilities. Specifically, the performances on text-centric tasks remain almost intact, while those on moleculecentric and reaction-centric tasks increase significantly. This proves that our method manages to improve the chemical capabilities of LLM while largely maintaining its abilities in understanding natural language.

Moreover, we also evaluated ChemDFM-I to illustrate the contributions of the different stages in our training pipeline. Results show that the atomized chemical knowledge enhancement stage consistently improves performance across all task categories, while the chemical rationale learning stage further strengthens performance on molecule- and reaction-centric tasks. We attribute this phenomenon to two factors. First, the molecule- and reaction-centric tasks typically demand more domainspecific chemical reasoning, such as molecular property prediction or retrosynthesis analysis. In contrast, text-centric tasks rely more on natural language understanding, such as chemical named entity recognition and literature question-answering. As a result, learning chemical reasoning over molecules and reactions provides limited benefit to these text-focused tasks. Second, for the sake of answer verifiability, the reinforcement learning tasks do not include purely text-based tasks, which may adversely affect the model’s text reasoning ability. Incorporating text-related tasks into the RL stage through joint training might help preserve performance on text-based tasks.

Furthermore, Figure 2b illustrates the performance changes across individual tasks. The results clearly show that most tasks benefit from our training pipeline, especially the molecule-centric tasks and reaction-centric tasks. Moreover, the two training stages provide complementary gains across different tasks, enabling the final model to achieve superior results on a broader range of tasks. Notably, among the tasks where ChemDFM-R does not surpass Qwen2.5-14B-Instruct, a substantial proportion involves numerical prediction, such as Yield Extraction, Molecular Property Calculation, and Product Yield Prediction. In fact, almost all the moleculecentric and reaction-centric tasks where ChemDFM-R falls short of Qwen2.5-14B-Instruct are those involving numerical reasoning and prediction. This pattern suggests that the numerical calculation and prediction abilities of ChemDFM-R are relatively weak, highlighting a potential direction for further improvements.

#### 2.4 ChemDFM-R outperforms advanced LLMs

To further demonstrate the prowess of ChemDFM-R, we compare it with three sets of models: 1) existing chemical LLMs, including MolInst [20], ChemLLM [9], ChemDFM [8], and ether0 [21]; 2) advanced non reasoning LLMs in the general domain, including GPT-4o [22] and Qwen3-14B (no think) [23]; 3) advanced reasoning LLMs in the general domain, including DeepSeek-R1 [5], Qwen3-14B (think) [23], and o4-mini [24]. The experimental results are illustrated in Figure 3. For detailed performances of individual tasks, please refer to Section F of the Supplementary Information.

As shown in Figure 3, ChemDFM-R significantly outperforms both general-domain LLMs and domain-specific chemical LLMs of similar size, especially on molecule-centric and reaction-centric tasks. In particular, ChemDFM-R achieves an average score of 73.8 on ChemEval, considerably surpassing Qwen3-14B (55.7), the next-generation model in the same series as our base model Qwen2.5-14B. ChemDFM-R also achieves significantly higher performance on molecule-centric and reaction-centric tasks (51.1 & 93.8) compared to Qwen3-14B (40.8 & 88.5) on SciKnowEval. These results further validate the effectiveness of the training pipeline of ChemDFM-R and highlight its prowess on chemical tasks. Moreover, existing chemical LLMs perform poorly on these benchmarks, with the best results reaching only 63.9 (on SciKnowEval) and 49.2 (on ChemEval). ChemDFM-R consistently outperforms them across all the task categories on both benchmarks, strongly demonstrating the effectiveness and value of ChemDFM-R and its training pipeline.

When compared to cutting-edge LLMs, ChemDFM-R achieves better performance than GPT-4o and DeepSeekR1, while demonstrating competitive results relative to o4-mini on SciKnowEval. Considering the tiny size of our model, this result strongly demonstrates the prowess of ChemDFM-R and the effectiveness of our training pipeline.

[Figure 3]

- Figure 3 | Performance comparison of ChemDFM-R and ten baseline LLMs, including chemistry-specific LLMs, general non-reasoning LLMs, and general reasoning LLMs, on SciKnowEval and ChemEval. We use RPS [25] to balance the different scales of the scores on different tasks in the ChemEval benchmark.

#### 2.5 Atomized knowledge is complementary to and more efficient than text-based knowledge

Directly verifying the effect of atomized knowledge enhancement would be extremely costly, since it requires repeatedly performing computationally expensive domain pretraining. Therefore, instead of training a model of the same size as ChemDFM-R on the full dataset for comparison, we used Qwen2.5-1.5B as the base model and conducted the full training pipeline on a 10% subset of ChemDFM-R’s data. By varying the data composition in the subset of the domain-pretraining corpus, we trained different versions of models for comparison while keeping the computational cost manageable. The results are presented in Table 1.

Compared with the model without any domain pretraining (Row 1), models pretrained on either the textbased-knowledge corpus (Row 2) or the atomized-knowledge corpus (Row 3) show improvements on most tasks. This demonstrates the effectiveness and necessity of domain pretraining for strengthening domain knowledge, and indirectly supports our hypothesis that general domain LLMs generally possess insufficient advanced chemical knowledge. Furthermore, the model pretrained solely on the atomized-knowledge corpus outperforms the model pretrained on the traditional text-based-knowledge corpus on many tasks, despite its corpus being only 20% of the latter. This provides strong evidence that fine-grained atomized knowledge enables more efficient domain knowledge enhancement. Finally, the model pretrained on the combined text-based- and atomized-knowledge corpora (Row 4) achieves the best overall performance, reflecting the complementary strengths of the two corpora and validating the effectiveness of our atomized-knowledge–enhanced domain pretraining approach.

#### 2.6 Mixed-source distillation successfully initializes the model’s reasoning capability

To validate the effectiveness of our newly designed mixed-source distillation method, we conduct an ablation study by gradually simplifying the composition of the distillation dataset. The results are shown in Table 2. The results prove that the traditional distillation method (Row 2) struggles to achieve positive impacts on performance in the chemical domain. It even underperforms the “zero” method (Row 1) proposed by Deepseek-R1, where there is no distillation stage before reinforcement learning. With the help of data sampled from the instruction-tuning dataset to maintain chemical capabilities and knowledge (Row 3), the model’s performance gets boosted significantly. Moreover, the pseudo-reasoning data further help the model to achieve higher performance (Row 4, which corresponds to the final setting of our proposed mixed-source distillation

- Table 1 | Ablation study results on SciKnowEval and ChemEval. DP represents Domain Pretraining, and Know. represent Knowledge. The best performance for each task is indicated using boldface. * We use RPS [25] to balance the different scales of the scores on different tasks in the ChemEval benchmark.

DP Corpus Composition SciKnowEval ChemEval* Atomized Know. Text-based Know. text mol. react. all text mol. react. all

- ✗ ✗ 66.9 30.0 34.5 49.9 44.6 25.5 25.4 34.8

- ✗ ✔ 67.1 30.6 36.5 50.4 43.6 25.6 28.5 35.2

- ✔ ✗ 65.7 31.8 37.8 50.3 51.9 25.9 33.8 40.8

- ✔ ✔ 66.2 31.7 37.4 50.5 53.7 26.9 31.0 41.1

- Table 2 | Ablation study results on SciKnowEval and ChemEval. “No Thinking” denotes the data source of the instruction-tuning dataset, “Pseudo” denotes the pseudo-reasoning data source, and “Distilled” denotes the teacher’s rationale data source. The best performance for each task is indicated using boldface. * We use RPS [25] to balance the different scales of the scores on different tasks in the ChemEval benchmark.

Data Source for Distillation SciKnowEval ChemEval* No Thinking Pseudo Distilled text mol. react. all text mol. react. all

- ✗ ✗ ✗ 77.7 49.9 94.2 69.2 76.4 70.9 59.9 70.4

- ✗ ✗ ✔ 75.5 49.6 87.9 67.2 72.0 68.8 63.6 68.9

- ✔ ✗ ✔ 77.4 50.6 92.4 69.1 78.0 83.1 61.8 74.5

- ✔ ✔ ✔ 76.8 52.0 94.5 69.5 80.3 84.5 61.3 75.8

method) and achieve better performance than the model from the “Zero” method training. These results strongly demonstrate the effectiveness and necessity of our design of mixed-source distillation.

#### 2.7 ChemDFM-R generates more accurate, reliable, and user-friendly rationales

To assess the quality of rationales under practical situations, we constructed ten graduate-level questions based on recent publications from several influential chemistry journals. The questions cover different major subfields of chemistry, including organic chemistry, inorganic chemistry, materials chemistry, analytical chemistry, and polymer chemistry. Then, different reasoning LLMs are leveraged to solve these questions through human-AI interactions. Five graduate-level chemistry experts were hired to evaluate these interactions across five dimensions, each of which is scored on a 5-point scale:

- • Chemical Correctness: The correctness of chemical knowledge and logic demonstrated throughout the reasoning process.
- • Answer Accuracy: The accuracy of the final answer given by the model.
- • Analytical Coverage: The extent to which different plausible possibilities are explored during reasoning.
- • Reasoning Coherence: Whether the reasoning remains focused, coherent, and aligned with the problem.
- • Effective Information Density: The density of useful information in the reasoning chain, reflecting the friendliness and efficiency of interaction.

The evaluation results are demonstrated in Figure 4, while the questions are listed in Section G of the Supplementary Information. From these results, we draw three main conclusions:

First, our model outperforms all baselines, including DeepSeek-R1, in both chemical correctness and answer accuracy, indicating that it possesses a stronger grasp of chemical knowledge. By carefully analyzing and comparing the evaluation results across question types, we observe that our model shows a clearer advantage on SMILES-related tasks, demonstrating a more precise understanding of molecular structures.

Second, our model achieves significantly higher scores in effective information density. Unlike models such as Qwen3-14B and DeepSeek-R1, which often generate extremely lengthy reasoning chains exceeding 1,000

[Figure 4]

- Figure 4 | Human assessment of the quality of rationales from different reasoning LLMs. Each small radar chart on the left corresponds to the scores of a question during the human assessment, and the large radar chart on the right is the average scores of the models on all ten questions. Please refer to Section G of the Supplementary Information for the detailed questions used in this assessment. As the rationales of o4-mini are not accessible, we only use the answer provided by o4-mini. All metrics are higher-is-better.

tokens, our model produces concise yet informative reasoning. This considerably reduces the burden on human users when verifying correctness or identifying errors.

Finally, our model performs slightly lower than Qwen3-14B and DeepSeek-R1 on analytical coverage. This is partially because their extremely lengthy reasoning chains allow them to enumerate many possible considerations. It is also true that our model commonly focuses on one or a few major possibilities rather than adequately enumerating all potential factors. This highlights a productive direction for future enhancement of our model’s analytical breadth.

#### 2.8 ChemDFM-R facilitates more reliable human-AI collaborations

Another important advantage of ChemDFM-R’s reasoning capability is that it allows humans to verify the correctness of answers, identify and correct errors, and discover new insights or perspectives. This enables more practical, reliable, and flexible human-AI collaboration. We then demonstrate the prowess of ChemDFM-R in enabling reliable human–AI collaboration through illustrative examples. Specifically, an example is showcased in Figure 5, while more examples are illustrated in Section H of the Supplementary Information.

To make our example more realistic, we design our conversation topic according to a newly published chemical research paper [26]. It is worth noticing that this paper was published in 2025, so it is impossible for ChemDFM-R to encounter this paper during either the pretraining of the base model, Qwen2.5-14B, or our domain-pretraining process. To demonstrate the potential of ChemDFM-R, we assume the role of a researcher seeking a new research inspiration and engage the model in a dialogue about the subfield to which this paper belongs.

First, as a researcher seeking inspiration, we ask ChemDFM-R to introduce the oxygen evolution reaction (OER) briefly, including the definition and mechanism of it. Although the answer itself is a relatively rigid response to the question, the model’s rationale reveals interesting insights. In its rationale, ChemDFM-R mentions that OER requires a transition metal oxide catalyst, which often presents valuable opportunities for further research. Therefore, we further ask it to give more detailed recommendations for the catalysts. As a response, ChemDFM-R proposes multiple different oxides, which is a wide range for us to dig into. However, in its rationale, ChemDFM-R itself says that “iridium oxide (IrO2) is particularly notable for its ...”, which is very inspiring. Since it is well known that iridium-based compounds are often very expensive, a natural follow-up question arises: how can we optimize this catalyst to reduce its cost while maintaining its catalytic performance? Surprisingly, ChemDFM-R manages to propose the initial ideas that align closely with the ideas presented in Li et. al. [26], “forming mixed metal oxides or heterostructures” and “optimizing the morphology and surface area of the catalysts”. At this point, a broad research direction has taken shape.

It is worth noticing that nearly all the inspirations are drawn from the rationale of ChemDFM-R, which demonstrates the significance and value of ChemDFM-R’s ability to generate reasoning. This example shows

- Turn1: What is oxygen evolution reaction (OER)? Please propose a reasonable mechanism of heterogeneous OER under acidic conditions.

<think>

To derive the answer, one starts by defining the oxygen evolution reaction (OER) as the electrochemical process that converts water into molecular oxygen, which is a critical step in water splitting. Recognizing that OER can occur under both acidic and basic conditions, the focus is on the mechanism under acidic conditions, where the reaction involves protons (H+) and water molecules. The process is typically facilitated by a catalyst, often a transition metal oxide, which provides active sites for the reaction. The mechanism is then broken down into several steps: first, the adsorption of water molecules onto the catalyst surface, followed by the formation of an intermediate species such as an adsorbed hydroxyl group (OH–) or a water-derived species. Subsequent steps involve the transfer of protons and electrons, leading to the formation of an oxygen-oxygen bond, which is a key step in the reaction. Finally, the desorption of molecular oxygen (O2) from the catalyst completes the cycle. This sequence of

adsorption, intermediate formation, bond formation, and desorption is consistent with the known electrochemical steps in OER under acidic

conditions, and it is supported by experimental observations and theoretical models. </think> <answer> The oxygen evolution reaction (OER) is an electrochemical process that involves the conversion of water into molecular oxygen (O2) and protons (H+). In heterogeneous OER under acidic conditions, the reaction typically occurs on a solid catalyst surface. The mechanism can be described as follows:

- 1. **Water adsorption**: Water molecules (H2O) adsorb onto the catalyst surface.
- 2. **Proton transfer**: Protons (H+) are transferred from the adsorbed water molecules to the catalyst surface, forming hydroxyl (OH-) groups.
- 3. **Oxygen adsorption**: The hydroxyl groups further react to form adsorbed oxygen species (O*), which are intermediates in the reaction.
- 4. **Oxygen evolution**: The adsorbed oxygen species combine to form molecular oxygen (O2), which desorbs from the catalyst surface and enters the solution. The overall reaction can be represented as:

2H2O → O2 + 4H+ + 4e</answer>

- Turn2: Since you mentioned that some transition metal oxides possess catalytic properties, can you recommend some specific metal oxides in detail?

<think> To determine which transition metal oxides exhibit catalytic properties for the oxygen evolution reaction (OER), one considers the electronic structure and surface reactivity of these materials. The oxides of metals such as iridium (Ir), ruthenium (Ru), and cobalt (Co) are known to be effective catalysts due to their favorable electronic properties and ability to stabilize reaction intermediates. Iridium oxide (IrO2) is particularly notable for its high activity and stability under acidic conditions, making it a widely used catalyst in OER applications. Ruthenium oxide (RuO2) also demonstrates excellent catalytic performance, while cobalt oxide (Co3O4) has been explored for its cost-effectiveness and moderate activity. These metal oxides are incorporated into electrochemical cells, where they facilitate the adsorption of water molecules, the formation of oxygencontaining intermediates, and the subsequent bond formation leading to O2 evolution. The selection of these materials is supported by extensive experimental data and theoretical calculations that highlight their ability to enhance the efficiency of the OER process. </think> <answer> Some transition metal oxides that possess catalytic properties for the oxygen evolution reaction (OER) include iridium oxide (IrO2), ruthenium oxide (RuO2), and cobalt oxide (Co3O4). These metal oxides are commonly used as catalysts in electrochemical cells for water splitting and other applications. They exhibit high catalytic activity under acidic conditions, making them suitable for OER. The catalytic activity of these metal oxides is attributed to their electronic structure and ability to stabilize reaction intermediates during the OER process. </answer>

- Turn3: I noticed that you mentioned iridium oxide (IrO2) is particularly notable, but iridium, as a noble metal, is relatively expensive. Can you devise a plan to reduce the cost while still maintaining good catalytic activity?

<think> To address the high cost of noble metal catalysts like iridium oxide while maintaining catalytic activity, one can explore strategies that reduce their usage or enhance their efficiency. One approach involves incorporating these metals into composite materials, where they are combined with cheaper transition metal oxides such as cobalt or nickel. By forming mixed metal oxides or heterostructures, the expensive metals can be distributed more efficiently, leveraging their catalytic properties while minimizing their overall amount. Additionally, optimizing the morphology

and surface area of the catalysts, such as by creating nanostructures or thin films, can increase the active sites available for the reaction, thereby

enhancing performance without requiring larger quantities of the noble metals. Furthermore, doping or alloying these metals with other elements can modify their electronic properties, improving their catalytic activity and reducing the necessary loading. These strategies collectively aim to balance cost-effectiveness with catalytic efficiency in the oxygen evolution reaction. </think> <answer> To reduce the cost of iridium oxide (IrO2) while maintaining good catalytic activity, one approach is to incorporate it into a composite material with a less expensive metal oxide, such as cobalt oxide (Co3O4) or nickel oxide (NiO). This composite material can exhibit enhanced catalytic properties due to the synergistic effects between the two metal oxides. Another strategy is to optimize the morphology and surface area of the catalyst, which can improve its activity and reduce the amount of IrO2 required. Additionally, doping IrO2 with other metal ions, such as iron (Fe) or manganese (Mn), can modify its electronic properties and enhance its catalytic performance. These approaches can help reduce the cost of IrO2 while still achieving good catalytic activity for the oxygen evolution reaction (OER). </answer>

###### Figure 5 | An example of reliable human-AI collaboration using ChemDFM-R. This conversation is inspired by Li et. al. [26]. This example illustrates a process of research inspiration seeking with the help of ChemDFM-R.

that, with the enhanced chemical knowledge and strong chemical reasoning capabilities, ChemDFM-R has the potential to facilitate reliable human-AI collaboration, thereby advancing AI-driven research and applications. More examples involving error correction and answer improvement with the help of rationales are demonstrated in Section H of the Supplementary Information.

### 3 Discussion

In this paper, we identify a key factor limiting the performance of existing LLMs and present ChemDFM-R, a new chemical reasoning LLM that achieves advanced performance. Specifically, we argue that the insufficient mastery of atomized chemical knowledge is a key factor underlying the limited performance of current LLMs, both general domain LLMs and current chemical LLMs, on chemical tasks. To tackle this problem, we develop a comprehensive toolkit, ChemFG-Tool, for functional group identification and construct a large-scale atomized-knowledge–enhanced chemical corpus, ChemFG. Leveraging ChemFG-Tool and ChemFG, we develop

- a comprehensive training pipeline that 1) enhances the model’s understanding of atomized chemical knowledge with domain pre-training and instruction tuning, and 2) teaches the model the key reasoning logic and method in chemistry with mixed-source distillation and reinforcement learning.

The ChemDFM-R model achieves advanced results across multiple tasks in our comprehensive evaluations of different benchmarks. It consistently outperforms all models of comparable size and, despite having only 14B parameters, achieves performance that is competitive with or even surpasses state-of-the-art proprietary models. These results strongly illustrate both the effectiveness of our training pipeline and the prowess of the resulting model, ChemDFM-R.

To better validate the effectiveness and necessity of the key designs of our training pipeline, we 1) compare ChemDFM-R’s performance with ChemDFM-I and Qwen2.5-14B-Instruct, which directly demonstrate the impact of our training process, and 2) design comprehensive ablation studies, validating the necessity of our design. From these evaluations, two main conclusions are drawn. Firstly, atomized knowledge enhancement could effectively and efficiently improve performance across all task categories. It achieves similar or even larger performance gains with only 20% of the tokens of the text-based knowledge. The best performance is achieved by combining atomized knowledge and text-based knowledge, illustrating the complementary strength of the two. Secondly, chemical rationale learning strengthens the performance on molecule- and reaction-centric tasks. We also validate the effectiveness and necessity of the mixed-source distillation design for initializing models’ reasoning capabilities when the distilled data is limited. Although the chemical rationale learning effectively endows the model with reasoning capability while also improving its overall performance, it weakens the model’s general language abilities, especially numerical prediction abilities. This is mainly because our RL stage relies on rule-based methods to calculate rewards, which restricts the types of tasks we can use. Designing reward mechanisms better suited to chemical contexts and allowing a broader range of RL tasks will be an important direction for the improvement of ChemDFM-R.

Besides the quantitative evaluation, we also conduct human assessments and case studies to 1) validate that ChemDFM-R could provide more precise and focused chemical rationales, and 2) illustrate the value of ChemDFM-R’s rationales in reliable human-AI collaborations under real-world scenarios. The results show that ChemDFM-R’s rationales are not only more precise and accurate, but also more concise and informative. This eliminates the need for researchers to go through reasoning chains containing thousands of tokens to locate valid reasoning and supporting evidence. Consequently, ChemDFM-R provides more reliable and verifiable answers, enhancing both usability and efficiency. We demonstrate ChemDFM-R’s potential in real-world human–AI collaboration by case studies, showcasing the critical role of its rationales in insight discovery, answer verification, and error detection and correction. However, although our model produces precise and informative reasoning chains, this conciseness can compromise analytical coverage. Achieving a better balance between maintaining concise reasoning chains and providing sufficiently diverse analyses could further enhance ChemDFM-R’s practical value in interactive settings.

Looking ahead, this work reveals a new perspective and methodology for training domain-specific LLMs. Traditionally, efforts to improve the performance of domain-specific LLMs have focused on increasing the size of the training corpus or enhancing the quality of the knowledge used during training. In this work, we point out that the learning efficiency of domain models varies depending on the form and granularity of the provided

[Figure 5]

- Figure 6 | The details of ChemDFM-R’s training pipeline. The four-stage training pipeline can be divided into two parts: a. Atomized Chemical Knowledge Enhancement where atomized chemical knowledge is introduced into the model, including 1) domain pre-training with the atomized-knowledge-enhanced corpus, ChemFG; 2) instruction tuning on the diversified chemical instruction tuning dataset we construct, which contains diversified chemical tasks and diversified chemical instructions;

- b. Chemical Rationale Learning where the model learns to reason in chemistry with atomized knowledge, including 3) Mixed-source distillation that could initialize the reasoning capability with limited distilled data; 4) reinforcement learning using DAPO with the rule-based reward system.

knowledge. Consequently, improving the efficiency of knowledge acquisition represents an equally important avenue for enhancing domain models’ performance. In this study, we achieve this by modifying the format of the knowledge-bearing corpus and incorporating finer-grained atomized domain knowledge, which significantly improves the model’s learning efficiency and leads to stronger overall performance. Similar strategies can be applied to other domains. Identifying the optimal form and granularity of domain knowledge for LLMs can yield improvements comparable to or even more efficient than increasing corpus size. Such approaches can either complement larger corpora or serve as effective alternatives when computational resources are limited. These insights gained from this work may inform the specialization training of LLMs across diverse domains, providing alternative methods and tools for developing stronger domain-specific LLMs.

### 4 Methods

We present the major details of our four-stage training pipeline for ChemDFM-R (Figure 6), while more details about the data-leakage prevention, hyperparameter choosing, and system prompts are introduced in Section D of the Supplementary Information.

#### 4.1 Atomized Chemical Knowledge Enhancement

In this stage, our model mainly learns the atomized chemical knowledge to prepare itself with “ingredients” to “cook” the chemical rationales. Specifically, we achieve that through domain pretraining and instruction tuning.

Domain Pretraining. In domain pretraining, we leverage the 101 billion-token ChemFG corpus introduced in Section 2.1 to familiarize our model with the knowledge related to functional groups. We train our model from one of the most advanced general LLMs, Qwen2.5-14B [19]. Considering that general knowledge is also vital for Chemical LLMs, we also incorporate a substantial amount of general-domain pretraining data into our domain pretraining corpus to ensure that the model retains its general capabilities as much as possible.

Instruction Tuning. The primary goal of instruction tuning is to teach the model how to analyze the purpose and requirements of a given task and make proper use of the knowledge learned in the pretraining phase. However, existing instruction tuning datasets in the field of chemistry are typically derived from well-studied chemistry tasks and suffer from a severe lack of diversity of both task varieties and instruction expressions. Therefore, we construct a new instruction tuning dataset for ChemDFM-R based on the instruction tuning dataset of ChemDFM [8]. To improve the overall task and instruction diversity, we introduce numerous new chemistry-related tasks, such as scientific paper QA, chemical property ordering, and reaction step prediction, and perform instruction-rewriting to achieve an average instruction-entry ratio of 1:50. For detailed information on the construction and composition of the instruction tuning dataset, please refer to Section B of the Supplementary Information. To maintain the general capabilities of our model, we mixed our chemical instruction tuning data with general instruction tuning data in a 1:2 ratio. The Qwen2.5-14B model is finetuned for 2 epochs on this mixed dataset after domain pretraining, resulting in the ChemDFM-I model.

#### 4.2 Chemical Rationale Learning

The primary goal of this stage is to teach the model how to reason with the atomized knowledge it has acquired. Chemical reasoning requires a deep understanding of chemical principles and logic, as well as the capability to apply them for analysis. These capabilities can not be learned or induced from general-domain reasoning training. Therefore, we propose a chemical rationale learning pipeline to specifically enhance the chemical reasoning capabilities of LLMs based on distillation and reinforcement learning.

Mixed-Source Distillation. We leverage distillation to prevent the early unstable cold start phase of reinforcement learning. It could illustrate the reasoning patterns to the model and build up its basic reasoning capabilities.

Specifically, the entries in the distillation dataset come from three sources, each of which corresponds to part of the abilities required for chemical reasoning: 1) the instruction tuning dataset of ChemDFM-R (∼70%) to maintain basic chemical knowledge and prevent catastrophic forgetting; 2) pseudo-reasoning data describing the functional groups of involved molecules or reactions (∼22%) and highlighting vital intermediate reasoning steps and functional group analyses; 3) teachers’ rationales from DeepSeek-R1 and o3-mini (∼8%) which introduce general reasoning patterns to the model and initiate its reasoning capabilities.

To improve the quality and efficiency of the teacher’s rationales in the chemical domain, instead of asking teacher models to generate rationales from scratch, we provide them with rich additional information. Specifically, the teacher models are provided with the question instruction, the ground truth answer, and the functional group information of the molecules and reactions in the question. Comparison of the rationales generated by DeepSeek-R1 with and without the additional information is illustrated in Figure 7. The rationales generated with full additional information are significantly more valid and in-depth than the other two. More examples with detailed analysis of the rationale generation are given in Section C.1 of the Supplementary Information. Moreover, to comprehensively assess the quality of our distilled rationales, we sampled a small subset of them and conducted a quantitative human assessment. The results are illustrated in Section C.2 of the Supplementary Information.

Similar to instruction tuning, we mix our mixed-source distillation dataset with general data in a 1:2 ratio. The general data are also sampled from multiple sources, where ∼92% of the entries are sampled from the

Traditional Distillation

|Input: Question|
|---|
|Generated Rationale: Starting from the bromo‐substituted bicyclic heterocycle, the reaction with the α‐chloroketone (CC(=O)CCl) under basic conditions in the presence of auxiliary reagents (including the carbamoyl derivative CN(C=O)C, cyclic ether solvent C1COCC1 and the acetate salts) generates a nucleophilic species that displaces the bromine atom; this substitution leads to the installation of a CH2COCH3 (acetylmethyl) side chain at the aromatic position, and detailed mechanistic steps involving activation, nucleophilic attack and subsequent stabilization yield the final product whose structure is best represented by the SMILES string NC1=NC2(CO1)c1cc(CC(=O)CH3)ccc1Oc1ncc(O)cc12. Answer:<br><br>NC1=NC2(CO1)c1cc(CC(=O)CH3)ccc1Oc1ncc(O)cc12|

Question:

Based on the provided reactants and reagents, can you determine the chemical structure of the resulting product? Reactants: NC1=NC2(CO1)c1cc(Br)ccc1Oc1 ncc(O)cc12.CC(=O)CCl

Reagents:

CN(C=O)C.C1COCC1.O=C([O])[O-].[K+].[K+].O.[Cl-].[Na+].O

Traditional Distillation with Answer

|Input: Question + Answer|
|---|
|Generated Rationale:<br><br>First, the starting heterocyclic structure is analyzed to identify the reactive sites, particularly noting the free hydroxyl<br><br>group on the aromatic heterocycle which is a common nucleophilic center. Next, by considering the reactant containing the acetyl chloride fragment (indicated by CC(=O)CCl) along with other reagents that could activate the acyl group (such as CN(C=O)C and the acetate ion O=C([O-])[O-] with accompanying cations), we infer that an acyl transfer reaction is taking place. The free OH group likely attacks an activated acyl intermediate generated from CC(=O)CCl, with the assistance of base or nucleophilic catalysts present in the reagents, leading to the substitution of the hydroxyl hydrogen with an acylated chain. In this context, the acyl group is appended through an ethoxy linkage (OCC(C)=O), indicating that a two-carbon chain bearing an acetyl terminus is incorporated via nucleophilic substitution. Finally, integrating this transformation into the overall structure of the starting heterocycle preserves the ring systems while modifying the phenolic OH into the acyloxy ether linkage, consistent with the structure provided in the product.|

Product:

Answer:

NC1=NC2(CO1)c1cc(Br)ccc1Oc1n cc(OCC(C)=O)cc12

Traditional Distillation with Answer and Functional Group (Ours)

|Input: Question + Answer + Functional Group Information|
|---|
|Generated Rationale: The reaction involves a phenol group (from the first reactant, NC1=NC2(CO1)c1cc(Br)ccc1Oc1ncc(O)cc12) and an aliphatic chloro-ketone (CC(=O)CCl). The phenol's hydroxyl group acts as a nucleophile, attacking the electrophilic carbon adjacent to the carbonyl in CC(=O)CCl via an SN2 mechanism. The chloride leaves as a leaving group, forming a new O-C bond between the phenolic oxygen and the acetyl-bearing carbon. This results in the substitution of the Cl atom in CC(=O)CCl with the oxygen from the phenol, generating an ether linkage (OCC(C)=O) attached to the aromatic ring. The ketone group (C=O) remains intact, and the bromine substituent on the aromatic ring is unaffected.|

FG Information:

A hydroxy group with an aromatic

ring attached + a chloro group > a ether group with an aromatic ring attached.

- Figure 7 | Comparison of rationales generated by o3-mini with and without additional input information. We mark the correct analyses in the rationale as green, while the wrong ones as red. For more examples and detailed analyses, please refer to Section C.1 of the Supplementary Information.

general data for instruction tuning of ChemDFM-R and ∼8% are from AM-Deepseek-R1-Distill-1.4M [27]. The ChemDFM-I model is finetuned for one epoch on this mixed dataset.

Reinforcement Learning. After distillation, reinforcement learning (RL) is leveraged to further enhance the reasoning capabilities of our model. To construct the reinforcement learning dataset, a subset of tasks that have verifiable answers is selected from our instruction-tuning dataset. The composition of the RL dataset is illustrated in Figure 6. We sampled data from the same sources used to build the instruction-tuning dataset, while minimizing overlap between the specific instances contained in the two datasets. These tasks are mixed together and uniformly sampled during training.

We follow the method recommended by DeepSeek-R1 [5], where the reward system consists of format rewards Rformat and correctness rewards Rcorrect. The format rewards evaluate whether a response strictly follows the reasoning format, which is “<think>...</think>\n<answer>...</answer>”. The correctness rewards evaluate whether a response is correct by comparing the answer between <answer> and </answer> with the ground truth answer. Specifically, considering the redundancy of the casual SMILES notations (one molecule could correspond to multiple SMILES), we first canonicalize all the SMILES in the response before calculating the accuracy rewards. Formally, for a given instruction and the model’s output o and ground truth answer a, we define the reward as

R(o,a) = λRformat(o) + Rcorrect(o,a), where λ is a coefficient that weights the two parts of the reward and is set to 1 during our training. We use the Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) [14] algorithm for our

reinforcement learning. For each question q with ground truth answer a from the training dataset D, DAPO algorithm first samples a group of G outputs {oi}Gi=1 from the model and then trains the policy model π by optimizing the following objective:

  1

 

|oi|

G

min ri,t(θ)Aˆi,t, c · Aˆi,t

JDAPO(θ) = E(q,a)∼D,{o

i}Gi=1∼πold(·|q)

G i=1 |oi|

t=1

i=1

c = clip(ri,t(θ),1 − ϵlow,1 + ϵhigh) s.t. 0 < |{oi | is_equivalent(a,oi)}| < G,

where ϵlow and ϵhigh denote the clipping range for the importance sampling ratio ri,t(θ), Aˆi,t denote the advantage for the i-th response, and is_equivalent(a,oi) measures whether the predicted answer in output oi matches the ground truth answer a. Formally, ri,t(θ) and Aˆi,t are calculated as follows:

Ri − mean({Ri}Gi=1) std({Ri}Gi=1)

πθ(oi,t | q,oi,<t) πold(oi,t | q,oi,<t)

,Aˆi,t =

,

ri,t(θ) =

where Ri denotes the reward for the i-th output oi from {oi}Gi=1.

### 5 Materials Availability

The parameters of the model generated in this study, ChemDFM-R, have been deposited in Huggingface: https://huggingface.co/OpenDFM/ChemDFM-R-14B.

### 6 Data Availability

The functional group identification toolkit, ChemFG-Tool, used during the construction of ChemFG is available at GitHub: https://github.com/OpenDFM/ChemFG-Tool. The evaluation benchmarks are obtained from the original benchmark authors.

###### Acknowledgments

This work was supported by the National Science and Technology Major Project (2023ZD0120703), the China NSFC Projects (62576212, 92370206, U23B2057, 62120106006), and the Shanghai Municipal Science and Technology Project (25X010202846).

###### Author Contributions

Conceptualization, Z. Zhao and L.C.; methodology, Z. Zhao, B.C., and L.C.; software, Z. Zhao, B.C., and D.M.; validation, Z.W., X.L., and S.Y.; data curation, Z. Zhao, B.C., Z.W., S.Z., H.W., Z.D., and W.L.; writing – original draft, Z. Zhao, B.C., Z. Zhu, and D.Z; writing – review & editing, Z. Zhao, L.C., and D.Z.; supervision, L.C., X.C, and K.Y.

Competing Interests The authors declare no competing interests.

### References

- [1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman,

- Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [3] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [4] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [5] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [6] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [8] Zihan Zhao, Da Ma, Lu Chen, Liangtai Sun, Zihao Li, Yi Xia, Bo Chen, Hongshen Xu, Zichen Zhu, Su Zhu, et al. Developing chemdfm as a large language foundation model for chemistry. Cell Reports Physical Science, 6(4), 2025.
- [9] Di Zhang, Wei Liu, Qian Tan, Jingdan Chen, Hang Yan, Yuliang Yan, Jiatong Li, Weiran Huang, Xiangyu Yue, Wanli Ouyang, et al. Chemllm: A chemical large language model. arXiv preprint arXiv:2402.06852, 2024.
- [10] Zihan Zhao, Bo Chen, Jingpiao Li, Lu Chen, Liyang Wen, Pengyu Wang, Zichen Zhu, Danyang Zhang, Yansi Li, Zhongyang Dai, et al. Chemdfm-x: towards large multimodal model for chemistry. Science China Information Sciences, 67(12):220109, 2024.
- [11] Yu Zhang, Yang Han, Shuai Chen, Ruijie Yu, Xin Zhao, Xianbin Liu, Kaipeng Zeng, Mengdi Yu, Jidong Tian, Feng Zhu, et al. Large language models to accelerate organic chemistry synthesis. Nature Machine Intelligence, pages 1–13, 2025.
- [12] Qian Tan, Dongzhan Zhou, Peng Xia, Wanhao Liu, Wanli Ouyang, Lei Bai, Yuqiang Li, and Tianfan Fu. Chemmllm: Chemical multimodal large language model. arXiv preprint arXiv:2505.16326, 2025.
- [13] OpenAI. Openai o3-mini, jan 2025.
- [14] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [15] Kehua Feng, Keyan Ding, Weijie Wang, Xiang Zhuang, Zeyuan Wang, Ming Qin, Yu Zhao, Jianhua Yao, Qiang Zhang, and Huajun Chen. Sciknoweval: Evaluating multi-level scientific knowledge of large language models. arXiv preprint arXiv:2406.09098, 2024.
- [16] Yuqing Huang, Rongyang Zhang, Xuesong He, Xuyang Zhi, Hao Wang, Xin Li, Feiyang Xu, Deguang Liu, Huadong Liang, Yi Li, et al. Chemeval: a comprehensive multi-level chemical evaluation for large language models. arXiv preprint arXiv:2409.13989, 2024.
- [17] Thermo: Thermodynamics and Phase Equilibrium component of Chemical Engineering Design Library (ChEDL) &#x2014; Thermo 0.6.0 documentation — thermo.readthedocs.io. https://thermo.readthedocs.io/. [Accessed 12-03-2026].

- [18] Daylight Theory: SMARTS - A Language for Describing Molecular Patterns — daylight.com. https: //www.daylight.com/dayhtml/doc/theory/theory.smarts.html. [Accessed 12-03-2026].
- [19] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [20] Yin Fang, Xiaozhuan Liang, Ningyu Zhang, Kangwei Liu, Rui Huang, Zhuo Chen, Xiaohui Fan, and Huajun Chen. Mol-instructions: A large-scale biomolecular instruction dataset for large language models. In ICLR. OpenReview.net, 2024.
- [21] Siddharth M. Narayanan, James D. Braza, Ryan-Rhys Griffiths, Albert Bou, Geemi P. Wellawatte, Mayk Caldas Ramos, Ludovico Mitchener, Samuel G. Rodriques, and Andrew D. White. Training a scientific reasoning model for chemistry. arXiv preprint arXiv:2506.17238, 2025.
- [22] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [23] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [24] OpenAI. Introducing openai o3 and o4-mini, 2025.
- [25] Jing Peng, Yucheng Wang, Bohan Li, Yiwei Guo, Hankun Wang, Yangui Fang, Yu Xi, Haoyu Li, Xu Li, Ke Zhang, Shuai Wang, and Kai Yu. A survey on speech large language models for understanding, 2025.
- [26] Gengnan Li, Adyasa Priyadarsini, Zhenhua Xie, Sinwoo Kang, Yuzi Liu, Xiaobo Chen, Shyam Kattel, and Jingguang G Chen. Achieving higher activity of acidic oxygen evolution reaction using an atomically thin layer of iro x over co3o4. Journal of the American Chemical Society, 147(8):7008–7016, 2025.
- [27] Han Zhao, Haotian Wang, Yiping Peng, Sitong Zhao, Xiaoyu Tian, Shuaiting Chen, Yunjie Ji, and Xiangang Li. 1.4 million open-source distilled reasoning dataset to empower large language model training. arXiv preprint arXiv:2503.19633, 2025.
- [28] PubChem. PubChem — pubchem.ncbi.nlm.nih.gov. https://pubchem.ncbi.nlm.nih.gov/. [Accessed 12-03-2026].
- [29] Maho Nakata and Tomomi Shimazaki. Pubchemqc project: a large-scale first-principles electronic structure database for data-driven chemistry. Journal of chemical information and modeling, 57(6):1300–1308, 2017.
- [30] Hanjun Dai, Chengtao Li, Connor Coley, Bo Dai, and Le Song. Retrosynthesis prediction with conditional graph logic network. Advances in Neural Information Processing Systems, 32, 2019.
- [31] Wengong Jin, Connor Coley, Regina Barzilay, and Tommi Jaakkola. Predicting organic reaction outcomes with weisfeiler-lehman network. Advances in neural information processing systems, 30, 2017.
- [32] Nadine Schneider, Nikolaus Stiefl, and Gregory A Landrum. What’s what: The (nearly) definitive guide to reaction role assignment. Journal of chemical information and modeling, 56(12):2336–2346, 2016.
- [33] Zipeng Zhong, Jie Song, Zunlei Feng, Tiantao Liu, Lingxiang Jia, Shaolun Yao, Min Wu, Tingjun Hou, and Mingli Song. Root-aligned smiles: a tight representation for chemical reaction prediction. Chemical Science, 13(31):9023–9034, 2022.
- [34] Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S Pappu, Karl Leswing, and Vijay Pande. Moleculenet: a benchmark for molecular machine learning. Chemical science, 9(2):513–530, 2018.
- [35] Yuwei Wan, Yixuan Liu, Aswathy Ajith, Clara Grazian, Bram Hoex, Wenjie Zhang, Chunyu Kit, Tong Xie, and Ian Foster. Sciqag: A framework for auto-generated science question answering dataset with fine-grained evaluation, 2024.

- [36] Weidong Yao, Ziqi Liu, Hao Ling, Hongyu Wang, Hufeng Zheng, Shao-Hua Wang, Dao-Yong Zhu, Sheng-Yong Zhang, and Xiaoming Chen. Convergent total synthesis of (-)-calidoustene. Journal of the American Chemical Society, 147(19):15963–15969, 2025.
- [37] Jingya Zhou, Longhui Chen, Ziang Miao, Jinjin Li, Yu-Xin Luan, Li Chen, and Pingping Tang. Photocatalyst-regulated trifluoromethoxylation of aryl halides under silver promotion. Journal of the American Chemical Society, 2025.
- [38] Johannes Schwarzmann, Tamina Z Kirsch, Benedikt Narz, and Crispin Lichtenberg. The methylbismuth dication: Pentagonal pyramidal coordination and ligand-induced lewis superacidity. Angewandte Chemie International Edition, page e16140, 2025.
- [39] Ryo Mandai, Takanori Iwasaki, and Kyoko Nozaki. Stable yet strongly lewis-acidic anions enabling cooperative catalysis with cationic transition-metal complexes. Angewandte Chemie International Edition, 64(37):e202503322, 2025.
- [40] Ruoyang Liu, Dan Zhao, Sailun Ji, Haipei Shao, Yongzhi Chen, Minjun Feng, Tie Wang, Juan Li, Ming Lin, Tze Chien Sum, et al. Harvesting singlet and triplet excitation energies in covalent organic frameworks for highly efficient photocatalysis. Nature Materials, 24(8):1245–1257, 2025.
- [41] Yimin Wu, Shengqiu Zhai, Yayin Deng, Xingliang Wang, Zhengyang Bin, and Jingsong You. Engineering localized aromaticity in amine-embedded polycyclic aromatic hydrocarbons for narrowband fluorescence emitter. Angewandte Chemie International Edition, 64(46):e202518763, 2025.
- [42] Junfeng Guo, Geetha Bolla, Jinlong Zhu, Jie Liu, Zijun Zhang, Wenjing Chen, Qing Liao, Wenping Hu, and Yonggang Zhen. Acid-responsive two-photon absorption switch via cocrystal-to-salt-to-cocrystal conversion. Angewandte Chemie, 137(45):e202517598, 2025.
- [43] Qingsong Zhang, Zhipeng Pei, Ah-Young Song, Miao Qi, Rebecca Shu Hui Khoo, Chongqing Yang, Tao Xia, Chen Zhou, Haiyan Mao, Zhiyuan Huang, et al. Manipulating aromaticity to redirect topochemical polymerization pathways. Journal of the American Chemical Society, 147(17):14715–14724, 2025.
- [44] Haley J Rugh, Jaeyong Lee, Chenyue Sun, Emily E Abdo, Juliana N Bem, Nitash P Balsara, and Geoffrey W Coates. Polysilaketals: High-performance polyether-based electrolytes with tunable disubstituted silane linkers. Angewandte Chemie, 137(2):e202415069, 2025.

### A Details about ChemFG

#### A.1 Raw Data Collection

Literature. Literature, including papers and textbooks, contains not only the widely accepted chemical knowledge and principles, but also the cutting-edge research in the field of chemistry. Therefore, to take full advantage of chemical literature, we collected over 12 million literature from the open Internet dated prior to January 2022. After further cleaning and deduplication, 79B tokens are obtained from it.

Molecules. Molecules are the fundamental participants in various chemical processes. Therefore, it is crucial for Chemical LLMs to understand molecular structures and properties. We manage to acquire large-scale molecule datasets from PubChem [28], one of the biggest open accessible chemical databases with more than 100M compounds. We include 30 million molecules along with their notations, descriptions (if applicable), and properties. Besides PubChem, we also leverage the PubChemQC [29] dataset, which contains the quantum chemical calculation results of 86M molecules from PubChem, to supplement the quantum chemical properties of these molecules, such as dipole moment and orbital energy. To diversify the final data entry, we randomize the order of the properties and use three different formats: markdown list, markdown table, and JSON dictionary to formulate the molecule data.

Reactions. Reactions are the major process in the chemical world. In ChemFG, we use the reactions from USPTO-FULL [30], one of the most comprehensive open-sourced chemical reaction databases. To avoid data leakage, we exclude the test set of USPTO-FULL, USPTO-MIT [31], and USPTO-50K [32] according to the products of reactions. Moreover, to further enhance the data diversity, we leverage the SMILES (Simplified Molecular Input Line Entry System) augmentation method introduced in R-SMILES [33] and achieve a total of 10 times augmentation of data. Finally, a corpus of 7 million reactions is obtained.

#### A.2 Functional Groups Coverage

The functional groups that can be recognized by our toolkit, ChemFG-Tool, are categorized based on the heteroatoms and listed as follows:

- • Hydrocarbon Groups (7): alkene, alkyne, allene, cumulene, carbocation, carbanion, carbene.
- • Boron Groups (6): borane, boronic acid, boronic ester, borinic acid, borinic ester, borate ester.
- • Oxygen Groups (36): alcohol, alkoxide, ether, phenol, phenolate, enol, enolate, enol ether, alkynol, alkynolate, alkynol ether, ketone, ketene, aldehyde, hemiketal, hemiacetal, ketal, acetal, carboxylic acid, carboxylate, ester, organic acid anhydride, carboxylic anhydride, organic carbonate, organic hydroperoxide, organic peroxide, peroxyacid, ortho ester, orthocarbonate ester, methylenedioxy, ethylenedioxy, oxonium ion, oxocarbenium ion, carbonyl ylide, oxonium ylide, epoxy.
- • Nitrogen Groups (62): primary amine, secondary amine, tertiary amine, ammonium cation, quat, amine oxide, enamine, hydroxylamine, hemiaminal, hemiaminal ether, thioaminal, thioaminal ether, aminal, primary ketimine, secondary ketimine, primary aldimine, secondary aldimine, amidine, guanidine, ketoxime, aldoxime, hydrazone, organic amide, amidate anion, imide, carbamic acid, carbamate ester, carbamate anion, azide, azo, hydrazine, acylhydrazine, amidrazone, cyanate, isocyanate, nitrile, isonitrile, cyanamide, carbodiimide, nitrate ester, nitrite ester, nitro, nitroso, nitrosamine, iminium cation, nitrone, nitronic acid, imidic acid, imidate anion, imidate, imidocarbonate, imidocarbamate, urea, azoxy, Noxoammonium, hydroxamic acid, hydroxamate, azanide, azomethine ylide, nitrile ylide, isodiazene, nitronate.
- • Sulfur Groups (85): mercaptan, thiolate anion, organic sulfide, thioenol, enedithiol, thioenolate, thioenol ether, persulfide anion, organic disulfide, sulfenic acid, sulfenic ester, sulfenamide, sulfoxide, sulfone, sulfine, sulfene, sulfinylamine, sulfur diimide, sulfinic acid, sulfonic acid, sulfinate ester, sulfonate ester, sulfinate anion, sulfonate anion, thiosulfinate, thiosulfonate, thiosulfurous ester, dithiosulfurous ester, thiosulfuric ester, organosulfite, organosulfate, dialkylsulfates, sulfinamide, sulfonamide, sulfamic acid, sulfamate, sulfamide, thiocyanate, isothiocyanate, thioketone, thioketene, thial, thioamide, thiourea, hemithioketal, hemithioacetal, dithiohemiketal, dithiohemiacetal, monothioketal, monothioacetal, dithioketal, dithioacetal, carbothioic S-acid, carbothioic O-acid, thiol form thiocarboxylate, thione form thiocarboxylate, thiolester, thionoester, carbodithioic acid, carbodithioic anion, carbodithioic ester, monothiocarbonate, xanthic acid, xanthate, xanthate anion, dithiocarbonate, trithiocarbonate, O-thiocarbamic acid, S-thiocarbamic acid, O-thiocarbamate, S-thiocarbamate, O-thiocarbamate anion, S-thiocarbamate anion, thioimidic acid, thioimidate anion, thioimidate, dithiocarbamic acid, dithiocarbamate, dithiocarbamate anion, imidothiocarbonate, imidodithiocarbonate, imidothiocarbamate, sulfonium ion, sulfonium ylide, sulfoxonium ylide.
- • Silicon Groups (5): silane, siloxane, silyl ether, silyl enol ether, silyl alkynol ether.
- • Phosphorus Groups (17): phosphine, phosphonium, aminophosphine, phosphine oxide, phosphinic acid, phosphinate, phosphonic acid, phosphonate, phosphite ester, phosphinite, phosphonite, phosphodiester, phosphate mono-ester, phosphate tri-ester, phosphoramide, thiophosphate, phosphonium ylide.
- • Halogen Groups (14): fluoro, chloro, bromo, iodo, halamine, sulfenyl halide, sulfinyl halide, sulfonyl halide, halosulfate, phosphoryl halide, phosphorus halide, acyl halide, imidoyl halide, thioacyl halide.
- • Organometalic Groups (5): organolithium, organomagnesium, organoaluminium, organozinc, organomercury.
- • Aromatic (4): pyrrolic N, pyridinic N, aromatic O, aromatic S.

The occurrence of these functional groups in the domain-pretraining corpus is shown in Figure 8.

O-thiocarbamic acid

thioimidic acid

thione form thiocarboxylic acid

alkynolate

dithiosulfurous ester

nitrile ylide

carbonyl ylide

thioketene

persulfide anion

thiosulfurous ester

Nitrogen Groups

O-thiocarbamate anion

silyl alkynol ether

dithiohemiketal

oxonium ylide

thione form thiocarboxylate

sulfinylamine

Oxygen Groups

S-thiocarbamate anion

carbodithioic anion

carbodiimide

enedithiol

nitronate

azomethine ylide

alkynol

Aromatic

thioacyl halide

nitronic acid

thiosulfinate

sulfinyl halide

dithiohemiacetal

carbodithioic acid

Halogen Groups

thiol form thiocarboxylate

cumulene

sulfine

thioenolate

isodiazene

thioimidate anion

orthocarbonate ester

Sulfur Groups

trithiocarbonate

imidic acid

nitrite ester

sulfur diimide

thiosulfuric ester

ketene

Hydrocarbon Groups

hemithioketal

halosulfate

carbene

imidothiocarbonate

dithiocarbamate anion

S-thiocarbamic acid

imidocarbonate

Phosphorus Groups

thiosulfonate

dithiocarbamic acid

organozinc

sulfenic acid

peroxyacid

sulfene

Silicon Groups

organomagnesium

organomercury

thial

carbodithioic ester

amidate anion

cyanate

dithiocarbonate

Boron Groups

sulfinate ester

monothioketal

azoxy

sulfenyl halide

borinic acid

borinic ester

Organometalic Groups

organolithium

organosulfite

hemithioacetal

oxocarbenium ion

monothiocarbonate

phosphoryl halide

phosphorus halide

100 101 102 103 104 105 106 107 108

thiol form thiocarboxylic acid

organoaluminium

alkynol ether

thionoester

borate ester

allene

phosphonium ylide

nitrosamine

thiocyanate

sulfoxonium ylide

nitrone

Hydrocarbon Groups Boron Groups Oxygen Groups Nitrogen Groups Sulfur Groups Silicon Groups Phosphorus Groups Halogen Groups Organometalic Groups Aromatic

imidodithiocarbonate

thioaminal

sulfamic acid

carbocation

- N-oxoammonium

sulfinate anion

- O-thiocarbamate

| |
|---|

phosphinic acid

alkoxide

sulfinic acid

| |
|---|

isothiocyanate

silyl enol ether

thiolate anion

oxonium ion

thioketone

sulfonium ylide

| |
|---|

dithioketal

phosphinate

sulfamate

phosphinite

amine oxide

| |
|---|

nitrate ester

organosulfate

siloxane

thiophosphate

phosphonite

| |
|---|

carbamate anion

sulfonium ion

imidoyl halide

cyanamide

phosphonium

| |
|---|

thioamide

dialkylsulfates

sulfenic ester

aminophosphine

dithioacetal

| |
|---|

phosphate tri-ester

thioenol

organic hydroperoxide

imidocarbamate

isocyanate

| |
|---|

phosphine oxide

borane

imidate anion

ortho ester

phosphite ester

boronic acid

| |
|---|

monothioacetal

organic acid anhydride

primary aldimine

phosphonic acid

sulfinamide

carboxylic anhydride

halamine

phosphoramide

sulfonyl halide

acyl halide

hemiketal

nitroso

enolate

thiolester

organic disulfide

sulfonate anion

primary ketimine

aldoxime

thioimidate

carbanion

isonitrile

phosphate mono-ester

phenolate

organic peroxide

boronic ester

azanide

sulfenamide

organic carbonate

phosphonate

hydroxamic acid

hemiacetal

dithiocarbamate

imidate

S-thiocarbamate

hydroxamate

sulfamide

amidrazone

iminium cation

hemiaminal

sulfonate ester

carbamic acid

mercaptan

sulfonic acid

hydroxylamine

epoxy

silane

sulfoxide

quat

azo

phosphodiester

imidothiocarbamate

silyl ether

thioaminal ether

ketal

hemiaminal ether

ethylenedioxy

secondary aldimine

carboxylate

ketoxime

secondary ketimine

enol

methylenedioxy

aldehyde

thioenol ether

phosphine

iodo

aminal

azide

enol ether

thiourea

amidine

acylhydrazine

alkyne

hydrazine

guanidine

hydrazone

ammonium cation

sulfone

acetal

carbamate ester

phenol

nitro

enamine

imide

nitrile

bromo

urea

organic sulfide

sulfonamide

ketone

carboxylic acid

primary amine

alcohol

O

alkene

ester

secondary amine

chloro

S

fluoro

tertiary amine

pyrrolic N

ether

pyridinic N

organic amide

100 101 102 103 104 105 106 107

###### Figure 8 | The distribution of the functional groups in the domain-pretraining corpus.

###### Molecule Labeling Error 1:

###### Molecule Labeling Error 2:

[Figure 6]

[Figure 7]

Correct Label: alkene, organoborane, boron chloride

Correct Label: (Unstable structure) iodo, pyrrolic N, aromatic S, amide, amidine

Our Annotation: alkene

Our Annotation: iodo, pyrrolic N, aromatic S

###### Reaction Labeling Error 1:

[Figure 8]

Correct Label: an olefin with an aromatic ring attached; an ester; a hemiketal; a breakable aliphatic ring + inorganic NN >> an amide; an imine; a new aromatic ring

Our Annotation: an olefin with an aromatic ring attached; an

ester; a hemiketal; a breakable aliphatic ring + inorganic NN >> an aliphatic O atom; a new aromatic ring

###### Reaction Labeling Error 2: Correct Label: Error Reaction

[Figure 9]

Our Annotation: an aliphatic C atom with three O atoms

attached + an aliphatic C atom with an olefin group attached >> a carboxylic acid; a new C - C bond

- Figure 9 | Examples of the error annotations of our functional group identification toolkit, ChemFG-Tool.

#### A.3 Quality Control

To validate the correctness of our functional group identification toolkit, ChemFG-Tool, we hired three graduate-level chemical experts to conduct manual inspections. Specifically, we sampled 100 annotated molecules and reactions, respectively, and asked the experts to determine whether the annotations were correct. Results show that ChemFG-Tool achieves 98% accuracy rate on molecules and 89% on reactions. Examples of the errors are demonstrated in Figure 9.

### B Instruction Tuning Dataset

#### B.1 Raw Data Collection

Our instruction tuning dataset is constructed of three parts corresponding to the three main information carriers in chemistry: molecule-centric tasks, reaction-centric tasks, and knowledge-centric tasks. The distribution of instruction tuning data is shown in Figure 10.

###### B.1.1 Molecule-Centric Tasks

- • Name Translation: The name translation between SMILES, IUPAC name, and molecular formula. The data is constructed from PubChem [28].
- • Description Generation: The molecule description task is to describe the molecule given its SMILES. The data is constructed from PubChem. We only use the high-quality descriptions that contain more than two sentences.
- • Molecule Design: The molecule design task is the reverse task of molecule description. It requires the model to predict the SMILES given the molecule description. We use the same high-quality description data from PubChem to construct this task.
- • Property Classification: These tasks require models to predict the value of molecular properties from a list of candidates (usually yes and no). The data is constructed from 5 of the most popular property classification datasets in MoleculeNet [34], namely BACE, BBBP, ClinTox, HIV, and Tox21.
- • Property Regression: These tasks require the models to predict the value of molecular properties,

- which is a real number. Data are also from MoleculeNet, namely FreeSolv, Lipo, and QM9.
- • Property Ordering: Provided a list of molecules, models are asked to rank them in ascending or descending order of some specific property. Raw data comes from the same source as property regression.
- • Property Selection: Provided a list of molecules, models are asked to select the one with the highest or lowest value of some specific property. Raw data comes from the same source as property regression.

###### B.1.2 Reaction-Centric Tasks

- • Reaction Completion: Given an incomplete reaction, models need to complete the missing reactants, reagents, or products. Raw data comes from USPTO-Full [30], USPTO-MIT [31], and USPTO-50K [32].
- • Step Prediction: Given a reaction, models are required to predict the experimental procedure to conduct it in the laboratory. Raw data comes from USPTO [30].
- • Yield Prediction: In this task, models are required to predict the yield of the given reactions. The data is constructed from the USPTO dataset.
- • Temperature Prediction: In this task, models are required to predict the temperature that is suitable for the given reactions to conduct. The data is constructed from the USPTO dataset.
- • Reaction Component Selection: In this task, a series of reactants and reagents is given with a list of candidate molecules. Models need to pick from the candidates the molecules that could participate in the reaction and lead to the highest yield. The data is constructed from the USPTO dataset.

###### B.1.3 Knowledge-Centric Tasks

- • Exam Questions: This task is composed of questions from the exams in middle school and high school. Raw data comes from the Open Internet.
- • Literature QA: In this task, models are required to answer questions based on the given paragraph. The data is extracted from the long paragraph following the method in SciQAG [35]. The raw data comes from the articles in the domain-pertaining. The articles are split into sections and then truncated into paragraphs within 2k to 3k tokens. We ask GPT-4o-mini to extract 15 keywords from each paragraph, then generate 10 question-answer pairs according to them. We adopt another LLM, Qwen2.5-14BInstruct, to evaluate the quality of the QA pair in 4 dimensions: completeness, accuracy, reasonableness, and agnosticism. The LLM will score the QA pair from 1 to 5 using the designed prompts. QA pairs with any scores below 5 are discarded. If there are more than 1 QA pair left, the questions are asked in conversation turns.
- • Literature Summarization: In this task, models are required to give a summarization of the paragraph. The summarization is generated from GPT-4o-mini from the paragraph sample.
- • Literature Translation: In this task, models are required to translate the English paragraph into Chinese. The translation is generated from GPT-4o-mini from the paragraph sample. Since the source data consists of OCR text extracted from English articles, which is inherently noisy, we decided to discard the reverse task of translating Chinese paragraphs into English.

#### B.2 Instruction Generation

To acquire a higher generalization capability, we adopt a two-stage process to obtain as diverse a set of instructions as possible for each task. Specifically, based on the number of data entries for each task, we first manually write 5-20 seed task descriptions accordingly. Then, we ask three different models, Qwen2.572B-Instruct, Llama-3.1-70B-Instruct, and GPT-4o-mini, to diversify these task descriptions. Specifically, during each request, we sample 5 descriptions from all the generated descriptions and ask the model to generate 10 new descriptions using 5 different prompts one by one. Following this, we append to each of the descriptions the instructions that introduce the input of each data entry by a formatted string of Python and ask GPT-4o-mini to further diversify them. As a result of the above diversification process, we constructed a number of instructions equivalent to approximately 2% of its total data volume for each task. In other words, each instruction appears around 50 times in the instruction-tuning dataset.

Pro p e   y R e

n

Description

nslatio

2 0

0

0,0 0 0

0

8 %

Tra

0,0

13 0,224

Molecule

%

gressio n

Prope y Selection

Generation

8

me

0

5 %

2

Na

Generation

513%0,224

200,010

8%

Prope y Classi cation 140%1,753

Tasks 21

ntric

Prope y Ordering 200,010

- 1,16
- 2,2

e

%

C

- 4
- 5

ule-

c

8%

ole

Exam Questions 538,468 21%

M

Knowledge-Centric Tasks

Temperature Prediction 90,000

Reaction-Centric Tasks

3%

838,468 32%

Reaction Component Selection

2538%6,461

96,461 4%

Yield

Prediction 140%0,000

Step

R e a

Prediction

ctio

150,000

- m

arizatio

- n

- n C
- o

- m

p o

- n e nt

6 %

A

Literature Tra

Q

15 0,0 0 0

Literature

0 4 %

6 %

0 4 %

0

0

m

0,0

1 0

0,0

u

Pre

S

0,0 0 0

4 %

10

e

0

dictio n

eratur

1

nslatio n

Lit

- Figure 10 | The distribution of instruction tuning data.

### C Quality Control of the Distillation Rationale Generation

#### C.1 Comparison of Rationales Generated by Teacher Models

As shown in Figure 7, 11, and 12, the rationales generated using our method exhibit distinct advantages in terms of accuracy, completeness, and logicality.

The advantage in accuracy is mainly reflected in Figure 7. In Figure 7, o3-mini completely misunderstands the chemical reaction that occurs between the given molecules. This might result from its incorrect identification of the reagents and the structure of the complex reactants (these errors are also reflected in the rationales). When given the correct answer, o3-mini still mistakenly identified chloro-ketone as acyl chloride and consistently adhered to this error throughout the reasoning process, resulting in a series of related structural inaccuracies. On the contrary, when the functional group information is provided, o3-mini manages to generate a near-perfect rationale with zero error.

The advantage in completeness is mainly reflected in Figure 11. In Figure 11, given only the question, DeepSeek-R1 only generates a brief analysis on the second option while dismissing the other three options with a single sentence in total. This overly simple analysis leads to a wrong prediction. With the help of the ground truth answer, the generated rationale analyzes all options individually. However, due to its lack of chemical knowledge, the analysis still exhibits errors in functional group recognition or overlooks key influencing factors. After enhancing chemical knowledge with the functional group information, DeepSeek-R1 finally manages to generate a more comprehensive analysis with few errors.

The advantage in logicality is mainly reflected in Figure 12. In Figure 12, with only the question, o3mini can hardly generate any useful rational. The rational merely repeats the IUPAC components mentioned in the question before rushing to a highly inaccurate conclusion without substantive analysis. When given the ground truth answer, o3-mini can construct a reasonably good rationale with minimal factual error.

Traditional Distillation

|Input: Question|
|---|
|Generated Rationale: The compound’s IUPAC name specifies a tetracyclic structure with a 7,15‐diaza tetracyclo[7.7.1.0^2,7.0^10,15]heptadecan backbone, featuring a ketone at position 6 and hydroxyl groups at positions 5 and 12; recognition of these functional groups, along with the indicated stereochemical configurations (1S,2R,5S,9S,10S,12S), leads to its identification as the natural product Pearsonine isolated from Pearsonia cajanifolia, and by mapping the four fused rings, incorporating the two nitrogen atoms, the carbonyl group, and the two<br><br>hydroxyl substituents, a SMILES string was constructed with explicit stereochemistry to yield the final result.<br><br>Answer: O=C1[C@@H]2CN(C[C@H](O)1)[C@@H]3CCN(C2)[C@H](O)C3|

Question:

Can you identify a molecule that fits the following criteria? What is the most likely substance? Description: (1S,2R,5S,9S,10S,12S)-5,12dihydroxy-7,15diazatetracyclo[7.7.1.02,7.010,15] heptadecan-6-one is a natural

Traditional Distillation with Answer

|Input: Question + Answer|
|---|
|Generated Rationale: The process begins by interpreting the IUPAC description, where the molecule is defined with a tetracyclic diaza framework and specific stereochemistry noted as (1S,2R,5S,9S,10S,12S), indicating the configuration of each chiral center, and contains two hydroxyl groups and one ketone functionality, which must be accurately placed in the resulting structure. Next, by analyzing the tetracyclic core given by the descriptor diazatetracyclo[7.7.1.02,7.010,15]heptadecan, one identifies the ring connectivity and the positions where heteroatoms (nitrogens) and functional groups (OH and C=O) are introduced in the scaffold. Standard SMILES notation requires the use of chiral descriptors such as [C@@H] or [C@H] to indicate the absolute configuration at each carbon, and ring closure numbers to denote the cycles correctly. By systematically assigning the atoms, ensuring that the ketone is placed at position 6 (as indicated by '6-one'), and incorporating the two hydroxyl substitutions at positions corresponding to 5 and 12, the connectivity is mapped following the lengths of carbon chains and ring junctures. In addition, the presence of nitrogen atoms in the diaza structure requires correct attachment without disrupting the stereochemical centers. Finally, converting the complete 3D structural<br><br>information into a linear SMILES string involves careful tracking of ring closures and stereochemical markers, leading<br><br>to the final notation that aligns with all the given criteria.|

product found in Pearsonia

cajanifolia with data available. SMILES:

Answer:

O=C1[C@@H](O)CC[C@@H]2[C@

H]3C[C@@H](CN12)[C@@H]1C[C

@@H](O)CCN1C3

Traditional Distillation with Answer and Functional Group (Ours)

FG Information:

|Input: Question + Answer + Functional Group Information|
|---|
|Generated Rationale: To derive the SMILES notation, one begins by examining the molecule’s systematic name and recognizing it describes a complex fused tetracyclic structure with specific stereochemistry indicated by (1S,2R,5S,9S,10S,12S). The process involves deciphering the ring system described as tetracyclo[7.7.1.02,7.010,15]heptadecanone, which informs the arrangement of the four fused rings and the connectivity of the bridgehead atoms. Next, the presence and positions of functional groups such as the dihydroxy substituents (at carbons 5 and 12) and the ketone group (at carbon 6, denoted by O=C) are identified and mapped onto the core structure. Additionally, the molecule contains two nitrogen atoms (as indicated by diaza at positions 7 and 15), which are incorporated into the cyclic system in place of carbon atoms; proper placement of these heteroatoms is key. After delineating these groups and establishing the connectivity of the cyclic systems, one assigns stereochemical configurations using the chiral markers “@” and “@@” according to the IUPAC configuration rules to reflect the specified S and R configurations accurately. Finally, using<br><br>SMILES conventions, the structure is linearized into a string that preserves ring closures, functional groups,<br><br>heteroatoms, and stereochemistry, resulting in the complete SMILES notation for the complex natural product.|

Two hydroxy groups, an amide group, and a tertiary amine group

###### Figure 11 | Comparison of rationales generated by o3-mini with and without additional input information. We mark the correct analyses in the rationale as green, while the wrong ones as red.

Traditional Distillation

Question:

|Input: Question|
|---|
|Generated Rationale:<br><br>The molecule with the lowest octanol/water distribution coefficient (log D) is the most hydrophilic. The second SMILES string contains a phenol group (O-), which is highly polar due to its hydroxyl group capable of hydrogen bonding, and an amide group (C(=O)N) that enhances water solubility through hydrogen bonding. Additionally, the<br><br>presence of an ether group (CCOCC) contributes to polarity. The other options feature more hydrophobic substituents like methyl groups (Cc), thioether (SCC), and fewer hydrogen-bonding groups, increasing their log D values. The combination of phenol, amide, and ether in the second molecule maximizes hydrophilicity, leading to the lowest log D. cAnswer: Oc1nnc(Cc2cccc(c2)C(=O)N3CCOCC3)c4ccccc14c|

Please use your knowledge of molecular properties in chemistry to answer the following question. Identify the molecule with the

lowest octanol/water distribution

coefficient from the given SMILES strings based solely on their structures. Only submit the corresponding SMILES of that molecule, without including any other information. Options: CN1CCN(CC1)C(=O)c2cc3ccccc3n2 C,Oc1nnc(Cc2cccc(c2)C(=O)N3CC OCC3)c4ccccc14,Cc1ccc2cc(C)c3n nc(SCC(=O)N4CCN(CC4)C(=O)c5oc cc5)n3c2c1

Traditional Distillation with Answer

|Input: Question + Answer|
|---|
|Generated Rationale: To determine the molecule with the lowest octanol/water distribution coefficient (log D), the hydrophilicity and<br><br>structural features of each option are analyzed. The first SMILES (CN1CCN(CC1)C(=O)c2cc3ccccc3n2C) contains a<br><br>piperazine ring with a secondary amide group (C=O) capable of hydrogen bonding, enhancing water solubility. The fused aromatic system (c2cc3ccccc3n2C) introduces moderate hydrophobicity, but the amide's polarity dominates. The second SMILES (Oc1nnc(Cc2cccc(c2)C(=O)N3CCOCC3)c4ccccc14) includes a hydroxyl group (O-) and an amide, but its extended aromatic systems (c4ccccc14) and alkyl chains (Cc2...) increase hydrophobicity. The third SMILES incorporates thioether (S-) and multiple methyl groups (Cc...), which are highly hydrophobic, along with bulky aromatic systems. Comparing these, the first molecule's combination of a polar amide and limited hydrophobic regions results in the highest hydrophilicity, leading to the lowest log D.|

Answer:

Answer:

CN1CCN(CC1)C(=O)c2cc3ccccc 3n2C

Traditional Distillation with Answer and Functional Group (Ours)

|Input: Question + Answer + Functional Group Information|
|---|
|Generated Rationale: The octanol/water distribution coefficient (logD) reflects a molecule's hydrophilicity and hydrophobicity. Lower logD<br><br>values correspond to higher hydrophilicity. The first SMILES (CN1CCN(CC1)C(=O)c2cc3ccccc3n2C) contains a tertiary amine and carboxamide. The tertiary amine can become protonated at physiological pH, increasing water solubility. The carboxamide group is polar, further enhancing hydrophilicity. The second SMILES<br><br>(Oc1nnc(Cc2cccc(c2)C(=O)N3CCOCC3)c4ccccc14) includes a phenolic hydroxyl group (pKa ~10), which may partially ionize at pH 7.4, but its lower ionization extent compared to the tertiary amine reduces its hydrophilicity. The third SMILES contains thioether/sulfide groups and multiple hydrophobic methyl substituents, increasing hydrophobicity.<br><br>The first molecule’s combination of ionizable tertiary amine and polar carboxamide gives it the lowest logD.|

FG Information:

A tertiary amine, an amide, and a pyrrolic N. An ether, an amide, two pyridinic Ns, and a phenol. An aromatic O, two amides, an thioether, three pyridinic Ns.

###### Figure 12 | Comparison of rationales generated by DeepSeek-R1 with and without additional input information. We mark the correct analyses in the rationale as green, while the wrong ones as red.

Substantive Issue 10% (27)

Minor Flaw 23% (60)

High Quality 67% (177)

- Figure 13 | Human validation result of teachers’ rationales. “Minor Flaw” represents acceptable flaws, such as skipping reasoning steps or missing possibilities. “Substantive Issue” means severe logic errors or nonsense reasoning.

However, the rationale still contains non-negligible issues in terms of logical coherence. A sound reasoning process should follow the approach exemplified by the reasoning chain generated by o3-mini using our method: analyzing in the order of molecular skeleton, functional groups, heteroatoms, and chiral centers. This sequence reflects a step-by-step refinement from the fundamental molecular structure to more intricate structural details. However, with only the question and answer, the generated rationale mixes these analytical steps and lacks critical details, such as the precise position of the nitrogen atom, resulting in a disorganized and incomplete reasoning process.

#### C.2 Quality Control

To quantitatively validate the quality of the teachers’ rationales generated by our method, we hired three graduate-level chemical experts to perform manual assessments. Results in Figure 13 show that among the sampled 264 rationales, 177 of them (67%) exhibit sufficiently high quality, 60 of them (23%) have minor, acceptable flaws such as reasoning step skipping or missing possibilities, and 27 of them (10%) contain substantive issues such as logic errors or nonsense reasoning. Two examples of distilled rationales with minor flaws are shown in Figure 14, and two examples of distilled rationales with substantive issues are shown in Figure 15. Considering that reinforcement learning will be used after distillation to correct errors and improve performance, and that it is difficult to systematically distinguish these problems in the rationales, we used all the data during the distillation process.

### D Training Details of ChemDFM-R

#### D.1 Mitigating Data Leakage

Data leakage is a crucial problem for the evaluation of LLMs. During the training of ChemDFM-R, measures have been taken to reduce the risk of data leakage. For the domain pretraining corpus, we avoid using the same molecules and reactions as those presented in SciKnowEval [15] and ChemEval [16] benchmarks. For instruction tuning and mixed-source distillation, the same molecules and reactions are deliberately excluded from a task if they appear in relative tasks in SciKnowEval and ChemEval. We use canonicalized SMILES to determine whether two molecules are the same. Reactions are considered the same if they share the same product.

##### D.2 Statistics about Computation and Data We trained ChemDFM-R on NVIDIA A800 Tensor Core GPUs for a total of 30840 GPU hours. Specifically:

1). for domain pretraining, the model was trained on the 101 billion ChemFG corpus (Section A) for 24728

###### Example 1 of Distilled Rationales with Minor Flaws: Overly simplify description of the reaction process

Instruction: Predict the product of the reaction using the given reactants and reagents, and express it in SMILES notation. Given:

[Figure 10]

Reactants: Cc1cc2c3ccccc3ccc2c2ccccc12.O=C1CCC(=O)N1Br Reagents: ClC1C=CC=CC=1.CO.C(OOC(=O)C1C=CC=CC=1)(=O)C1C=CC=CC=1

Provide the resulting product and its SMILES notation.

Rationale:

The reactants are a methyl-substituted polycyclic aromatic compound (SMILES: Cc1cc2c3ccccc3ccc2c2ccccc12) and a brominated imide halamine (SMILES: O=C1CCC(=O)N1Br). The halamine contains an N-Br bond, which is reactive. The aliphatic methyl group (CH3) attached to the aromatic system undergoes bromination via radical substitution. The N-Br bond in the halamine cleaves homolytically, generating a bromine radical (Br·). This abstracts a hydrogen atom from the methyl group, forming a methyl radical (CH2·). The bromine radical then combines with the methyl radical, resulting in a bromomethyl group (CH2Br) and a new Br-C bond. The aromatic system remains intact, and the imide group (O=C1CCC(=O)N1) from the halamine loses Br to form a secondary amine (NH) as a byproduct.

###### Example 2 of Distilled Rationales with Minor Flaws: Inappropriate naming

Instruction: To maximize the reaction yield, choose a candidate from the list Nc1ccc(-c2ccccc2)cn1, Cc1ccc(N)nc1, COc1ccc2nc(N)sc2c1, Nc1ncc(c2ccccc2)cc1-c1ccccc1 Answer: that best completes the given reaction O=C(CBr)c1ccc(Br)cc1.O=C([O-])O.[Na+]>CCO>Unknown product, ensuring each substance is represented in SMILES notation and adhering to the standard format for chemical reaction equations.

[Figure 11]

###### Rationale:

Starting with the reaction details, we first note that the mechanism likely involves nucleophilic attack by an amine on an electrophilic center generated from the acyl bromide species. Since all options supply a primary amine functionality, attention turns to the aromatic framework that supports it. By examining the aromatic heteroatoms and substituent patterns, one assesses how each molecule’s electronic properties might stabilize the intermediate and facilitate the reaction. In this case, the extended aromatic system and the presence of pyridinic nitrogen—with its electronwithdrawing nature—helpfully modulate reactivity and lower steric hindrance. Additionally, comparing specific substituents such as ether or additional alkyl groups, one finds that the option offering a conjugated biphenyl structure with optimized resonance stabilization is predicted to provide the highest yield by enhancing nucleophilicity while accommodating the transition state effectively. Given these considerations, the reasoning supports selecting the molecule with the most extended conjugated system featuring a primary amine attached to a pyridinic framework, as it best meets the criteria for high-yield synthesis under the reaction conditions.

###### Figure 14 | Examples of the distilled rationales with minor flaws.

- Example 1 of Distilled Rationales with Substantive Issue: Incorrect identification of the key components

Instruction: Can you please rank these molecules, given in SMILES format, according to their internal energy at 0 K from highest to lowest? SMILES List:

[H]OC([H])([H])C([H])([H])[C@@]1([H])O[C@@]2([H])C([H])([H])[C@@]2([H])O1,

[H]C#CC([H])([H])C1[NH2+][C@@]2([H])C([H])([H])[C@@]2([H])O1, [H]OC([H])([H])C1[NH2+][C@@]2([H])C([H])([H])C([H])([H])[C@@]2([H])O1, [H]OC([H])([H])C([H])([H])C1[NH2+][C@@]2([H])C([H])([H])[C@@]2([H])O1 Order of internal energy at 0 K:

Rationale:

At 0 K, internal energy is determined by molecular potential energy, which depends on bond strength, ring strain, and functional group stability. The first SMILES contains a six-membered acetal ring (low ring strain) and hydroxy groups. The second SMILES has an alkyne (high bond energy due to sphybridized carbons) and a five-membered hemiaminal ether ring (moderate ring strain). The third SMILES includes a six-membered hemiaminal ether ring (lower strain than five-membered) with an alcohol group and a charged NH2+ group (higher electrostatic potential energy). The fourth SMILES contains a five-membered hemiaminal ether ring (higher strain than six-membered) with an alcohol group and NH2+ charge. Alkyne > charged groups > five-membered ring strain > six-membered ring stability dictates the ranking: third (six-membered charged),second (alkyne + fivemembered),fourth (five-membered charged),first (six-membered uncharged).

- Example 2 of Distilled Rationales with Substantive Issue: Multiple logic errors

Instruction: For chemical analysis, select the molecule with the highest dipole moment by comparing the SMILES of the given options: [H]/N=C/NC(=O)NC(=O)C,[H]/N=C/NC(=O)C(C)(C)C,[H]/N=C(/C(=O)/C(=N/[H])N)N,[H]/N=C(\C(=O)N/C=N/[H])N. Provide the SMILES of the chosen molecule.

Rationale:

To determine which molecule has the highest dipole moment, analyze the polarity and symmetry of the functional groups. Dipole moments increase with greater electronegativity differences and additive polar bond vectors. The first molecule contains urea (two amide groups) and imide (two carbonyl groups),which are polar, but symmetry may reduce the net dipole. The second molecule has an amidine and a single amide, but bulky tertbutyl substituents (C(C)(C)C) introduce steric hindrance, potentially reducing polarity. The third molecule features a ketone (C=O) and amidine, but conjugated resonance between the ketone and amidine groups (C(=O)/C(=N/[H])) may delocalize electron density, opposing dipole contributions. The fourth molecule includes a carboxamide (C(=O)N) and two amidine groups (N=C-N), arranged asymmetrically. The carboxamide's carbonyl (C=O) and N-H bonds create strong dipoles, while the amidine groups (with protonated N atoms and adjacent C=N) contribute additional polarity. The lack of symmetry in the fourth molecule allows these dipoles to align constructively, resulting in the highest net dipole moment.

###### Figure 15 | Examples of the distilled rationales with substantive issues.

- Table 3 | The training hyperparameters used during the training of ChemDFM-R.

Domain Instruction Mixed-source Reinforcement

Pretraining Tuning Distillation Learning Initial Learning Rate 1e-5 1e-5 1e-5 5e-7

Minimal Learning Rate 1e-6 0 0 0 Optimizer Adam(0.9, 0.95) Scheduler Cosine

Max Sequence Length 8192 8192 8192 Max Generation Length - - - 1024

Train Batch Size 624 512 512 128 Rollout Batch Size - - - 512

Epochs 1 2 1 1 DAPO Group Size - - - 8

DAPO Epsilon - - - (0.2, 0.3) Initial KL Coefficient - - - 1e-3

GPU hours;

- 2). for instruction tuning, the model was trained on over 7.5 million instructions (Section B), which consist of 2.5 million chemical instructions and 5 million general instructions, for 3785 GPU hours;
- 3). for mixed-sourced distillation, the model was trained on the mixed-sourced distillation dataset, which is of the same scale as the instruction-tuning dataset, for 2059 GPU hours;
- 4). for reinforcement learning, the model was trained on 121,811 samples for 268 GPU hours.

#### D.3 Hyperparameters and System Prompts

Training Hyperparameters. The hyperparameters used during the training of ChemDFM-R are listed in Table 3.

Inference Setting. During the inference of ChemDFM-R, we set the temperature to 0.6, topK to 1, and topP to 1 with no other penalties.

System Prompts. During the training of ChemDFM-R, three different system prompts are used. Specifically:

- • For all samples in the instruction-tuning dataset and the non-reasoning samples in the mixed-source distillation dataset, we use: "You are a helpful assistant."
- • For the pseudo-reasoning data in the mixed-source distillation dataset, we use: "You are a helpful assistant that is good at answer chemical questions. You will analyze the presence of functional groups in molecules and the changes in functional groups during reactions before giving response. These analyses will help you solve the problem better. The analyses and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.\ni.e.,\n<think>\nanalyses here\n</think>\n<answer>\nanswer here\n</answer>"
- • For the distilled data and the training of reinforcement learning, we use "You are a helpful assistant that is good at reasoning. You always reason thoroughly before giving a response. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.\ni.e.,\n<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>"

### E The Analysis of ChemDFM-R’s Rationale

In Figure 16, ChemDFM-R is asked to pick a correct product for the given reaction. Instead of wasting time (tokens) on analyzing the SMILES in great detail, which DeepSeek-R1 always does, ChemDFM-R1 goes directly into the key point of this question: the functional groups present in the reactants and the potential reactions

Question: Given a question and four options, please select the right answer. Your answer should be "A", "B", "C" or "D". Please directly give the answer without any explanation. Given the reactants and reagents (in SMILES format) which are split by "." as follows: CC(C)(C)OC(=O)NCCc1ccc(OCCC23CC4CC(CC(C4)C2)C3)cc1.CO.Cl Which of the following candidate SMILES represents the correct product of this reaction?

A. COc1ccc(COC(=N)C(Cl)(Cl)Cl)cc1 B. c1ccc(OCCOCCCCCCCCOC2CCCCO2)cc1

C. C#Cc1ccc(C2(OCC)CC2)c(C(C)C)c1 D. NCCc1ccc(OCCC23CC4CC(CC(C4)C2)C3)cc1 <think> To approach the problem, one first confirms that the starting material incorporates a carbamate ester group (defined as an ester of carbonic acid that links to an –NH- fragment, which is often used for protecting amino groups during synthesis) and then considers that converting this functional group into a primary amine requires a deprotection step. This analysis involves recognizing that such transformations typically occur via acid-catalyzed mechanisms, where the reagents provided (including a molecule containing CO and Cl, which can function as a source of activating agents like an HCl, a common reagent for removing carbamate protecting groups) promote the cleavage of the tert-butyl protecting group, leading to the formation of the free amine. By examining the set of options supplied, only the selection whose structure contains the primary amine functionality, identified by an -NH2 attached directly to an aromatic ring substituted with an appropriate side chain (which matches the context provided in the reagent transformations), is consistent with this deprotection process. This methodical structural evaluation, considering both the reaction mechanism and the changes between the protecting group and the final product, leads to the selection of the correct molecule that exhibits the primary amine substitution pattern under the reaction conditions. </think> <answer> D </answer>

###### Figure 16 | An example of the responses generated by ChemDFM-R on the reaction prediction task.

Question: You are an expert chemist. Given the molecule SMILES, your task is to provide the detailed description of the molecule using your experienced chemical knowledge. Please strictly follow the format, no other information can be provided.

The Molecule SMILES is as follows: CCCCCC/C=C/C(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N2C=N C3=C(N)N=CN=C32)[C@H](O)[C@@H]1OP(=O)(O)O

<think> To derive the molecular description, one first examines the SMILES notation to identify the structural components such as the alkene, hydroxy, ether, anhydride, primary amine, carboxamide, thiolester, and phosphate mono-ester groups, as well as the aromatic heteroatoms like pyrrolic N and pyridinic N.

Recognizing these functional groups and their connectivity, one then correlates the structure with known

biochemical molecules, particularly those involved in metabolic pathways. By noting the presence of a coenzyme A moiety and a deprotonated thiolester linkage, which are characteristic of acyl-CoA derivatives, one can infer that the molecule is an acyl-CoA compound. This structural analysis, combined with knowledge of metabolic roles in organisms such as Escherichia coli and mouse, leads to the conclusion that the molecule is a metabolite with a specific acyl group attached to coenzyme A.

</think> <answer> The molecule is a medium-chain fatty acyl-CoA that results from the formal condensation of the thiol group of coenzyme A with the carboxy group of trans-2-nonenioic acid. It is a trans-2-enoyl-CoA, a medium-chain fatty acyl-CoA and a monounsaturated fatty acyl-CoA. It is a conjugate acid of a trans-2-nonenoyl-CoA(4-). </answer>

###### Figure 17 | An example of the responses generated by ChemDFM-R on the molecule captioning task.

Question: Please give the IUPAC name of the molecule given by SMILES: Cc1ccc(C(C)NC(=O)CCCN (c2cccc(Cl)c2)S(C)(=O)=O)c(C)c1.

<think> To derive the IUPAC name, one first examines the SMILES notation to identify the molecular structure, noting the presence of an aromatic ring substituted with methyl groups, a secondary amine linkage, and a sulfonamide group attached to a benzene ring bearing a chloro substituent. The next step involves determining the connectivity and substituent positions on the aromatic system, ensuring that the substituents are numbered according to IUPAC rules to give the lowest possible locants. The molecule is then divided into its principal structural components, with one part being the substituted benzene ring and the other part being the acyl chain linked to the sulfonamide group. The acyl chain is further analyzed to determine its substituent pattern, including the N-substitution with the sulfonamide group. By applying systematic nomenclature rules, including the priority of functional groups and the use of prefixes and suffixes to denote substituents and connectivity, the complete IUPAC name is constructed. This process integrates the identification of functional groups, the determination of substituent positions, and the systematic assembly of the name according to IUPAC guidelines. </think>

<answer> 4-(3-chloro-N-methylsulfonylanilino)-N-[1-(2,4-dimethylphenyl)ethyl]butanamide

</answer>

###### Figure 18 | An example of the responses generated by ChemDFM-R on the SMILES to IUPAC task.

between them. Specifically, ChemDFM-R successfully identifies the key functional group, the carbamate ester group. By recalling that the carbamate ester group is typically used to protect amino groups, ChemDFM-R infers that the reaction taking place is likely a deprotection reaction. Then, ChemDFM-R confirms its assumption by examining the provided reagents. Finally, ChemDFM-R predicts the feature of the possible product and picks the option that matches it. This example demonstrates the precision of ChemDFM-R in finding the key point of chemical questions, and the effectiveness and efficiency of ChemDFM-R’s rationales. Moreover, it is also worth noticing that, instead of using the “elimination-shortcut” which is commonly adopted by other cutting-edge reasoning LLMs, ChemDFM-R directly reasoned out the reaction mechanism and the features of the correct answer, thereby selecting the correct option.

As illustrated in Figure 17, when asked to describe a molecule given by SMILES, ChemDFM-R first analyzes the functional groups present in the molecule, such as the alkene group, the phosphate mono-ester group, and the pyrrolic N group. Then, ChemDFM-R successfully correlates the composition and connectivity of these functional groups with metabolic pathways and further manages to identify the molecule as a coenzyme A derivative. After that, it recognizes the deprotonated thiolester linkage in the molecule and further narrows down the molecule to an acyl-CoA derivative. Finally, ChemDFM-R gives a relatively comprehensive description of the molecule. ChemDFM-R even provides the potential role of this molecule in metabolic processes in its rationale, further demonstrating its strong reasoning ability as well as the value of the rationale as a complement to the final answer.

- Figure 18 showcases an example response of ChemDFM-R when asked to generate the IUPAC name of the given molecule. The IUPAC name is the standard name for a molecule, assigned according to the rules established by the International Union of Pure and Applied Chemistry (IUPAC). It can effectively reflect the functional groups present in the molecule and their connectivity. Therefore, ChemDFM-R starts its reasoning with a comprehensive analysis of the functional groups of the given molecule. Then, it emphasizes the importance of correctly labeling the atoms, which is precisely an area where large language models are particularly prone to errors. After this, ChemDFM-R follows the rule of IUPAC naming and divides the molecule into its principal structural components. It also specifically points out the N-substitution with the sulfonamide group. Finally, a complicated and correct IUPAC name is predicted by ChemDFM-R.

### F Detail Results of Benchmark Evaluation

- F.1 ChemEval

We consider the L1 and L2 level tasks in ChemEval [16] as text-centric tasks, the L3 level tasks as moleculecentric tasks, and the L4 level tasks as reaction-centric tasks. Moreover, there are tasks that we can not achieve a feasible grading in ChemEval. We temporarily skip these tasks. The raw results are demonstrated in Table 4.

- As illustrated in Table 4, ChemDFM-R manages to achieve competitive performance in the text-centric tasks compared with the cutting-edge LLMs, while achieving SOTA performance across all the molecule-centric tasks and a large portion of the reaction-centric tasks. A detailed analysis of the task characteristics reveals that ChemDFM-R tends to perform less effectively on tasks involving numerical prediction, which will be a key focus of our future optimization efforts.

F.2 SciKnowEval

We group the tasks in SciKnowEval [15] based on their input and output. Specifically, the task is classified as a text-centric task if there is no SMILES appear in its input or output, as a reaction-centric task if there are reaction SMILES appear in its input or output, and as a molecule-centric task otherwise. Due to budget limit, we currently skip the tasks that require GPT-4o for grading. The raw results are demonstrated in Table 5.

- As illustrated in Table 5, ChemDFM-R achieves competitive performance on SciKnowEval compared to cutting-edge LLMs. It is worth noting that ChemDFM-R’s performance advantage is less pronounced on SciKnowEval than on ChemEval. This is primarily because most tasks in SciKnowEval are formulated as multiple-choice questions, which substantially reduce the burden on the model’s comprehension and generation processes, allowing it to arrive at correct answers through “shortcuts” such as option comparison

o4-miniChemDFM-R

73.3AddEF134.258.746.75.00.070.065.070.865.068.3

95.095.095.095.0TempEF170.075.020.225.010.075.090.0

###### 95.095.095.095.095.095.095.095.095.0F140.05.0TimeE

83.3MolNGBLEU3.645.575.934.646.759.943.19.941.768.6

73.0IUPAC2MFEM0.00.025.00.00.040.00.05.00.060.0

86.0SMILES2MFEM0.00.045.00.00.010.05.00.00.055.0

55.0IUPAC2SMILESEM0.00.010.00.030.00.00.05.00.015.0

32.0SMILES2IUPACEM0.00.00.05.00.00.00.00.00.00.0

72.5MolPCAcc49.952.861.09.40.064.154.557.158.965.7

57.8Mol2PCBLEU29.836.438.918.52.337.240.830.438.533.7

23.5SubRecAcc0.03.38.22.00.07.50.05.00.018.3

42.0LRecAcc0.00.020.10.00.010.05.010.035.00.0

- 34.0RRecAcc0.00.019.00.05.010.015.05.025.030.0

- 35.035.0SolvRecAcc0.00.05.00.00.02.025.030.031.0

90.0Acc40.035.050.00.010.075.075.080.065.064.0MCTask

95.0TFTaskAcc55.075.070.05.00.080.085.080.085.081.0

17.3SATaskBLEU0.612.210.60.415.09.34.39.210.311.2

69.6CNERF10.044.159.022.819.266.760.967.157.163.1

37.7F14.50.012.30.00.020.324.120.225.524.9CERC

82.5SolvEF158.370.025.05.075.073.379.054.072.376.3

79.0CharMEF120.767.026.249.116.873.871.060.776.061.6

100.0CatTEF180.090.055.030.00.095.085.085.090.067.0

95.0YieldEF10.055.020.030.020.080.075.075.075.057.0

70.4BLEU56.345.754.04.42.166.862.264.767.465.1AbsGen

23.6OLGenBLEU18.912.522.42.78.714.310.018.013.115.4

55.0TopCAcc40.050.040.020.00.050.040.050.050.045.0

35.0ReactTRF120.013.315.05.05.030.025.030.020.028.5

17.9TempRecRMSE76.527.171.522.221.723.434.622.231.8↓∞

6.6TimeRecRMSE20.518.717.89.48.09.510.79.618.8↓∞

73.3PPreAcc0.00.03.35.010.00.018.25.835.060.0

55.0YPredAcc20.020.020.00.00.014.925.025.045.023.0

31.0RatePredOverlap3.013.84.79.00.018.214.99.97.214.1

5.0FBTaskAcc0.00.00.00.00.00.00.00.00.01.0

5.0S2SEM0.00.00.00.00.00.00.00.00.00.0

10.0IMDerAcc0.00.00.00.05.00.05.00.05.00.0

ether0GPT-4o Qwen3-14BDeekSeekQwen3-14B

-20B-DPO-13B-v1.0-8B-v1.5(nothink)-R1(think)

Molecule-CentricTasks

Reaction-CentricTasks

Text-CentricTasks

TasksMetricMolInst ChemLLMChemDFMChemDFM

###### |32

- Table 4 | The detailed benchmark results of different models on ChemEval [16].

o4-miniChemDFM-R

97.2ChemRI92.895.994.695.560.096.997.195.396.396.3

27.4BalancingEq.0.616.312.321.70.010.57.711.27.2510.5

50.9MolCap.14.421.022.722.50.913.98.612.610.014.0

88.975.478.777.679.357.385.884.286.384.080.2ChemLiterQA

100.0ReactMechInfer.96.398.598.197.877.798.596.798.197.898.4

99.0CompIden.andProp.93.697.495.296.872.898.698.296.498.697.3

99.5ChemDU95.298.297.496.574.498.698.698.999.497.9

94.982.888.885.988.30.993.693.292.193.492.1ChemHV

92.2ChemCal.34.932.736.835.322.752.453.978.891.452.3

62.4MolTox.Pred53.754.344.345.23.937.451.851.355.143.0

85.2ChemSafeTest68.774.859.565.317.581.281.783.180.481.5

95.7MolNameConv.60.370.869.781.035.186.771.541.883.789.9

48.9MolProp.Pred31.833.938.148.044.037.941.241.648.144.6

62.8MolWeightCal.25.725.916.922.922.425.917.04.431.135.4

45.438.933.231.129.327.234.737.239.138.118.7MolProp.Cal.

57.4MolStru.Pred33.130.935.625.931.346.130.97.536.631.7

89.4MolGen.64.452.984.470.260.938.669.344.575.486.6

99.3ReactionPred39.292.891.695.955.848.090.582.593.798.4

95.244.383.478.577.876.241.270.083.483.289.4Retrosynthesis

ether0GPT-4o Qwen3-14BDeekSeekQwen3-14B

-20B-DPO-13B-v1.0-8B-v1.5(nothink)-R1(think)

Molecule-CentricTasks

Reaction-CentricTasks

Text-CentricTasks

TasksMolInst ChemLLMChemDFMChemDFM

- Table 5 | The detailed benchmark results of different models on SciKnowEval [15].

and elimination.

### G Details of the Human Assessments of Rationale Quality The ten original questions we used are listed as follows. Organic Chemistry:

- • [36] I have used m-CPBA to convert the carbon-carbon double bond within the [H][C@]12CC[C@@]3(CC [C@]1C)C(=O)C1=C[C@@]4(C)CC[C@@](C(C)C)[C@]4([H])C[C@]1([H])[C@]23C into an epoxide, and obtained chiral epoxy products with different ratios (d.r. = 5:1). Please propose possible reasons.
- • [37] Clc1ccccc1 is difficult to react with [O-]C(F)(F)F under normal conditions, but it can be converted into a free radical cation under photocatalytic conditions and can react with [O-]C(F)(F)F in the presence of [Ag+] to obtain the product in high yield. Please provide the structure of the product.

###### Inorganic Chemistry:

- • [38] C[Bi+2]([O]1CCCC1)([O]1CCCC1)([O]1CCCC1)([O]1CCCC1)[O]1CCCC1 is a newly reported strong Lewis acid. Please provide the oxidation state, ligand, and coordination number of the metal ion in C[Bi+2]([O]1CCCC1)([O]1CCCC1)([O]1CCCC1)([O]1CCCC1)[O]1CCCC1, and explain the reason why it has strong Lewis acidity.
- • [39] Fc1c(F)c([B-](c2c(F)c(F)c(B3Oc4ccccc4O3)c(F)c2F)(c2c(F)c(F)c(B3Oc4ccccc4O3)c(F)c2F)c2c(F)c (F)c(B3Oc4ccccc4O3)c(F)c2F)c(F)c(F)c1B1Oc2ccccc2O1 is a newly reported stable Lewis acidic anion, which breaks the previous understanding that anions are incompatible with Lewis acids. Please analyze its structure, explain why it can act as a Lewis acid, and indicate its binding sites with Lewis bases.

###### Materials Chemistry:

- • [26] What is oxygen evolution reaction (OER)? Please propose a reasonable mechanism of heterogeneous OER under acidic conditions.
- • [40] C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3 and c1ccc2nsnc2c1 can form covalent organic frameworks through covalent bonding under certain conditions, which can utilize the excitation energy of singlet and triplet states for photocatalysis. Please determine which is the electron donor and which is the electron acceptor during the electron transfer process through the structural analysis of C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3 and c1ccc2nsnc2c1.

###### Analytical Chemistry:

- • [41] CC(C)(C)c1cc2cc(C(C)(C)C)cc3c4cc(C(C)(C)C)cc5cc(C(C)(C)C)cc(c(c1)c23)c54 is a fluorescent material. Please explain the reason why it can emit light from the perspective of molecular structure.
- • [42] c1cc(-c2ccc3cc4cc(-c5ccncc5)ccc4cc3c2)ccn1 and N#Cc1cc(C#N)c(C#N)cc1C#N can be co-assembled into a eutectic and emit orange light under photoluminescence. After the addition of O=C(O)C(F)(F)F, the eutectic will undergo a transformation, and the luminescence will change from orange light to yellow light. Please explain the reason.

###### Polymer Chemistry:

- • [43] O=S(=O)(Oc1nc(=Cc2ccco2)c(OS(=O)(=O)C(F)(F)F)nc1=Cc1ccco1)C(F)(F)F and O=S(=O)(O c1nc(=Cc2ccccc2)c(OS(=O)(=O)C(F)(F)F)nc1=Cc1ccccc1)C(F)(F)F are monomers of two polymers that form polymers under solid-state heating conditions. In addition, it was found that O=S(=O)(Oc1nc (=Cc2ccco2)c(OS(=O)(=O)C(F)(F)F)nc1=Cc1ccco1)C(F)(F)F is relatively more prone to dimerization when heated in the solution. Please infer the reason by comparing their structures.
- • [44] Under certain conditions, C[Si]1(C)OCCOCCO1 can undergo anionic ring-opening polymerization to obtain polymers, which exhibit extremely high ionic conductivity in lithium battery electrolytes, surpassing poly (ethylene oxide). Please analyze the reasons from a structural perspective.

- Turn1: What is oxygen evolution reaction (OER)? Please propose a reasonable mechanism of heterogeneous OER under acidic conditions.

<think> To derive the answer, one starts by defining the oxygen evolution reaction (OER) as the electrochemical process that converts water into molecular oxygen, which is a critical step in water splitting. Recognizing that OER can occur under both acidic and basic conditions, the focus is on the mechanism under acidic conditions, where the reaction involves protons (H+) and water molecules. The process is typically facilitated by a catalyst, often a transition metal oxide, which provides active sites for the reaction. The mechanism is then broken down into several steps: first, the adsorption of water molecules onto the catalyst surface, followed by the formation of an intermediate species such as an adsorbed hydroxyl group (OH–) or a water-derived species. Subsequent steps involve the transfer of protons and electrons, leading to the formation of an oxygenoxygen bond, which is a key step in the reaction. Finally, the desorption of molecular oxygen (O2) from the catalyst completes the cycle. This sequence of adsorption, intermediate formation, bond formation, and desorption is consistent with the known electrochemical steps in OER under acidic conditions, and it is supported by experimental observations and theoretical models. </think> <answer> The oxygen evolution reaction (OER) is an electrochemical process that involves the conversion of water into molecular oxygen (O2) and protons (H+). In heterogeneous OER under acidic conditions, the reaction typically occurs on a solid catalyst surface. The mechanism can be described as follows:

- 1. **Water adsorption**: Water molecules (H2O) adsorb onto the catalyst surface.
- 2. **Proton transfer**: Protons (H+) are transferred from the adsorbed water molecules to the catalyst surface, forming hydroxyl (OH-) groups.
- 3. **Oxygen adsorption**: The hydroxyl groups further react to form adsorbed oxygen species (O*), which are intermediates in the reaction.
- 4. **Oxygen evolution**: The adsorbed oxygen species combine to form molecular oxygen (O2), which desorbs from the catalyst surface and enters the solution. The overall reaction can be represented as:

2H2O → O2 + 4H+ + 4e-

</answer>

- Turn2: Since you mentioned in your thinking that the formation of oxygen-oxygen bond is the key to the reaction, please give an intermediate that reflect the formation of oxygen-oxygen bond and improve the mechanism you provided.

<think> To incorporate the formation of the oxygen-oxygen bond into the mechanism, one starts by recognizing that this bond formation is a critical step in the oxygen evolution reaction (OER). The process involves the generation of an adsorbed oxygen species (O) on the catalyst surface, which then undergoes a transformation to form an intermediate that contains the oxygen-oxygen bond. This intermediate is typically represented as a peroxide-like species (OOH) or a superoxide-like species (O2−), depending on the specific reaction conditions and the electronic properties of the catalyst. The formation of this bond is facilitated by the transfer of protons and electrons, which are provided by the acidic environment and the catalyst, respectively. The stabilization of the intermediate is crucial for the subsequent steps, including the final desorption of molecular oxygen (O2) from the catalyst surface. By integrating these specific intermediates and their roles in the mechanism, one can clearly illustrate how the oxygen-oxygen bond is formed and how it contributes to the overall OER process under acidic conditions. </think> <answer> In the mechanism of the oxygen evolution reaction (OER) under acidic conditions, the formation of the oxygen-oxygen bond is a critical step. This bond formation occurs through the generation of an intermediate species that contains two oxygen atoms. The specific intermediate formed depends on the reaction conditions and the catalyst used.

One possible intermediate is the peroxide species (OOH), which is formed when two hydroxyl groups (OH-) undergo a coupling reaction. This coupling reaction involves the transfer of an electron and a proton, leading to the formation of the peroxide bond (O-O).

Another possible intermediate is the superoxide species (O2−), which is formed when an oxygen atom (O) undergoes a one-electron reduction. This reduction can occur through the transfer of an electron from the catalyst surface or from another oxygen species.

The formation of these intermediates is essential for the subsequent steps in the OER mechanism, including the final desorption of molecular oxygen (O2) from the catalyst surface. The stability and reactivity of these intermediates are influenced by factors such as the electronic properties of the catalyst and the reaction conditions (e.g., pH, temperature).

</answer>

- Figure 19 | An example of reliable human-AI collaboration using ChemDFM-R. We draw inspiration from this conversation from Li et. al. [26]. This example illustrates a process of error detection and correction with the help of rationales.

### H More Examples of Reliable Human-AI Collaboration

- Figure 19 illustrates a conversation starting from the same turn as that illustrated in the main text. In this conversation, we focus on fully understanding the mechanism of the oxygen evolution reaction (OER). Suppose, as a newbie, we are unable to determine the correctness of the answer. With the help of ChemDFM-R’s rationale, we could easily discover that the key step of the reaction mentioned in the rationale, which is “the formation of an oxygen-oxygen bond”, is absent in the answer. This could serve as a reminder that the answer could be incorrect, and drive us to further request the model to clarify this inconsistency. After this follow-up inquiry, the model provided a better answer.
- Figure 20 is a conversation about a reaction proposed in Yao et. al. [36]. We first ask ChemDFM-R to explain the reason for the different ratios of the chiral epoxy products. Although the model’s answer is relatively broad, ChemDFM-R thoroughly analyzes the influence factors in its rationale, including the bond’s geometry and the spatial arrangement of substituents around the double bond, which is not included in the answer. With this information, we can further pursue the follow-up question and obtain an improved answer.

- Turn1: I have used m-CPBA to convert the carbon-carbon double bond within the [H][C@]12CC[C@@]3(CC[C@]1C)C(=O)C1=C[C@@]4(C)CC[C@@](C(C)C) [C@]4([H])C[C@]1([H])[C@]23C into an epoxide, and obtained chiral epoxy products with different ratios (d.r. = 5:1). Please propose possible reasons.

<think> The reaction involves m-CPBA (meta-chloroperbenzoic acid) oxidizing a carbon–carbon double bond in a complex cyclic structure to form an epoxide. m-CPBA acts as an

electrophilic oxidizing agent, where the peracid's oxygen attacks the double bond, forming a cyclic transition state. The stereochemical outcome is influenced by the complex cyclic framework's rigidity and steric factors. The double bond's geometry and the spatial arrangement of substituents around the double bond determine which face of the alkene is more accessible to the electrophilic oxygen from m-CPBA. The complex cyclic system imposes constraints that favor one diastereomeric pathway over another, leading to a 5:1 diastereomeric ratio. The chiral centers in the complex cyclic structure further influence the transition state energies, with one pathway being thermodynamically favored due to reduced steric strain or optimal orbital overlap during the epoxidation process.

</think> <answer> The complex cyclic framework of the molecule and the steric factors around the double bond lead to a preferential attack by the oxidant on one face of the double bond, resulting in a diastereomeric ratio of 5:1. </answer>

- Turn2: Please explain in detail how double bond's geometry and spatial arrangement of substituents around the double bond determine which face of the alkene is more accessible.

<think> The spatial arrangement of substituents around the double bond in the complex cyclic structure creates steric hindrance, which influences the accessibility of the alkene face to the oxidizing agent. The complex cyclic framework imposes rigidity, restricting the approach of m-CPBA to one face of the double bond. Substituents such as methyl group create steric bulk, favoring the less hindered transition state during epoxidation. Additionally, electronic factors, including the electron-donating or electron-withdrawing nature of substituents, further modulate the reactivity of the alkene face. The combination of steric and electronic effects results in a preferential attack on the less congested face, leading to the observed diastereomeric ratio of 5:1 in the epoxide products. </think> <answer> The complex cyclic structure imposes steric constraints that influence the accessibility of the double bond to the oxidizing agent. The arrangement of substituents around the double bond creates a situation where one face is more sterically accessible than the other, leading to a preference for the oxidizing agent to attack from that side, which is reflected in the diastereomeric ratio of the resulting epoxide products. </answer>

###### Figure 20 | An example of reliable human-AI collaboration using ChemDFM-R. We draw inspiration from this conversation from Yao et. al. [36]. This example illustrates a process of information completion.

