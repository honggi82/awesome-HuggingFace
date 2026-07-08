# arXiv:2502.09042v2[cs.CL]27Mar2025

## TYPHOON T1: AN OPEN THAI REASONING MODEL

Pittawat Taveekitworachai, Potsawee Manakul, Kasima Tharnpipitchai, and Kunat Pipatanakul SCB 10X R&D SCBX Group Bangkok, Thailand {pittawat,potsawee,kasima,kunat}@scb10x.com

ABSTRACT

This paper introduces Typhoon T1, an open effort to develop an open Thai reasoning model. A reasoning model is a relatively new type of generative model built on top of large language models (LLMs). A reasoning model generates a long chain of thought before arriving at a final answer, an approach found to improve performance on complex tasks. However, details on developing such a model are limited, especially for reasoning models that can generate traces in a low-resource language. Typhoon T1 presents an open effort that dives into the details of developing a reasoning model in a more cost-effective way by leveraging supervised fine-tuning using open datasets, instead of reinforcement learning. This paper shares the details about synthetic data generation and training, as well as our dataset and model weights. Additionally, we provide insights gained from developing a reasoning model that generalizes across domains and is capable of generating reasoning traces in a low-resource language, using Thai as an example. We hope this open effort provides a foundation for further research in this field.1

1 INTRODUCTION

Recent advancements in the field of large language models (LLMs) have gained more attention from the paradigm of scaling at test time (Snell et al., 2024; DeepSeek-AI, 2025), which improves the performance of an LLM by allocating compute resources during test time–i.e., generating more tokens– to provide higher compute. While there exist techniques like chain-of-thought (CoT) prompting (Wei et al., 2022), self-consistency (Wang et al., 2023b), and best-of-N evaluation (Snell et al., 2024), which do not change the weights of the LM and focus more on prompting and sampling strategies, another approach is the reasoning model. A reasoning model, also known as a thinking LLM (Wu et al., 2024), is a relatively new type of LLM that is able to generate a long reasoning trace before providing a conclusive answer, leading to improved performance across tasks.

The long reasoning trace of a reasoning model often consists of behaviors such as breaking down a task into sub-tasks, reflecting on its own intermediate results, and self-correcting them (DeepSeekAI, 2025). Models that fall into this category include OpenAI’s o-series2, Qwen’s QwQ3, and DeepSeek R1 (DeepSeek-AI, 2025). Although some of these studies and/or models (Wu et al., 2024; Muennighoff et al., 2025; Guan et al., 2025) are open to a certain degree, most are limited to only open-weight availability and obscure details, making them difficult to replicate in an open environment and to further progress research effort.

In this paper, we provide a detailed report on our lessons learned from developing a Thai reasoning model. We aim to develop a model capable of reasoning across tasks and generating reasoning traces in Thai, a low-resource language. In addition to the insights provided in this paper, we also open our datasets, data pipeline, training configurations, and model weights to support future research.

1Research artifacts: https://huggingface.co/collections/scb10x/ typhoon-t1-iclr-2025-sci-fm-artifacts-67e399bba281c900d778cc3e

- 2https://openai.com/index/learning-to-reason-with-llms/
- 3https://qwenlm.github.io/blog/qwq-32b-preview/

Table 1: A comparison of openness among popular reasoning models, focusing on dataset availability, data processing transparency, training methodology, and model accessibility. P denotes partial details. Typhoon T1 is the only model providing full openness across all categories, including its data recipe.

Model Datasets Data Recipe Training Recipe Model Weights

OpenAI’s o-series ✗ ✗ ✗ ✗ Google’s Gemini 2.0 Flash Thinking ✗ ✗ ✗ ✗ Qwen’s QwQ ✗ ✗ ✗ ✓ DeepSeek R1 ✗ ✗ P ✓ Typhoon T1 ✓ ✓ ✓ ✓

We name our reasoning model Typhoon T1 to honor the open Thai LLM we selected as our initial model, Typhoon 2 3B Instruct (Pipatanakul et al., 2024), for developing the Thai reasoning model. We utilize a supervised fine-tuning (SFT) approach using the constructed long-thought data, presenting an alternative to large-scale reinforcement learning (RL), which was previously introduced in DeepSeek R1 (DeepSeek-AI, 2025) and Skywork o1 (o1 Team, 2024) as methods capable of eliciting long reasoning behaviors.

RL is often unstable (Rafailov et al., 2023), which is why we selected the SFT approach for experimentation in this paper. Given the relatively new field of reasoning models, we believe there is more than one approach to achieving a reasoning model, e.g., SFT (Muennighoff et al., 2025; Qin et al., 2024), RL (DeepSeek-AI, 2025; o1 Team, 2024), and knowledge distillation (Team, 2025; Labs, 2025; Huang et al., 2024). We also experiment with different reasoning (i.e., thinking) formats to investigate whether introducing additional auxiliary structural XML tags to the thoughts helps improve reasoning performance. Table 1 shows a comparison of openness between popular reasoning models available on the market.

Our contributions are as follows:

- • We totally open all aspects of our approach in developing Typhoon T1, a Thai reasoning model.
- • We conduct a comprehensive ablation study to evaluate impact of thinking formats, dataset sizes, data mixture, and language on Typhoon T1 and its variants.

2 METHODOLOGY

An overview of our methodology for data generation and experiments is illustrated in Figure 1. In the remainder of this section, we explain the rationale for our base model selection and details of structured thinking, a new thinking format we introduce in this paper. Additionally, we provide details on the selected benchmarks used for evaluation across our experiments.

To develop a reasoning model, we first select an initial LLM. To enhance Thai performance, we consider various open-weight Thai LLMs, including OpenThaiGPT 1.5 (Yuenyong et al., 2024), Pathumma 1.0.0 4, and Typhoon 2 (Pipatanakul et al., 2024). Due to resource limitations, we choose Typhoon 2 3B Instruct as our baseline, as it is the only available open-weight 3B model. Typhoon 2 3B is based on Llama 3.2 3B Instruct (Grattafiori et al., 2024), with additional training to improve its Thai performance (Pipatanakul et al., 2024).

We select the instruct variant of the model, unlike DeepSeek R1 Zero (DeepSeek-AI, 2025), which is built on DeepSeek V3 Base, as we do not utilize RL. Our rationale is to start with a model that already follows instructions and enhance its ability to produce correct answers with long reasoning traces. Choosing an instruct model is more effective in this case than a base model, which would require a larger dataset to teach basic instruction-following skills.

To SFT the LLM into a reasoning model, we begin by preparing our training dataset. Instead of performing knowledge distillation from other reasoning models (Team, 2025; Labs, 2025), we opt to construct long reasoning traces ourselves. We develop a transformation-and-refinement pipeline using few-shot prompting (Brown et al., 2020) to transform and refine a normal response into a long-

4https://huggingface.co/nectec/Pathumma-llm-text-1.0.0

[Figure 1]

- Figure 1: Top: The transformation-and-refinement pipeline used for long-thinking data generation described in Sections 2.2.1 and 2.2.2. Bottom-Left: The structured long-thinking (the best thinking format) training pipeline for Typhoon T, as described in Section 3.1. Bottom-Right: The bilingual English-Thai Typhoon T1 model training pipeline detailed in Section 3.4.

form response. This allows better control over thought quality across tasks and avoids dependence on the constraints of a teacher model.

We select open datasets and refine the provided answers into long thought formats using an LLMbased pipeline. Once we obtain long-thought data, we apply SFT to the model. However, one key question remains: what is the most effective thinking format?

We identify two main reasoning formats dominating the field: (1) unstructured thinking and (2) semi-structured thinking. In unstructured thinking, the model generates a long response without explicit separation between the chain of thought and the answer. Examples include Qwen’s QwQ and Skywork o1 (o1 Team, 2024). Most existing reasoning models (Labs, 2025; Team, 2025; DeepSeekAI, 2025; Wu et al., 2024) fall into the semi-structured thinking category, where explicit separators, e.g., <think></think>, <answer></answer>, and <R>, distinguish the thinking process from the final answer.

We observe some prompt engineering approaches (Nye et al., 2021; Ziqi & Lu, 2023) using auxiliary tokens such as XML tags or Markdown to guide model responses. Inspired by these techniques, we introduce structured thinking, a new thinking format where reasoning models use auxiliary tags to generate plans and reasoning traces in each step through a scratchpad. Differences between three thinking formats is illustrated in 2.

The following subsection provides further details on structured thinking. Our research question is: Among the three thinking formats, which best improves model performance? To reduce variables in our analysis, we begin with an English-only dataset.

- 2.1 STRUCTURED THINKING

Structured thinking is an approach mostly inspired by plan-and-solve prompting (Wang et al., 2023a) and the use of scratchpad for LLMs to show intermediate output in a specified region (Nye et al., 2021). We introduce the following XML tags to help guide LLMs on structuring their thoughts and responses as show in 2 (c). The <thoughts></thoughts> and <response></response> tags are the high-level tags and only exist once in each response. <plan></plan> only contains the <step></step> tags denoting each step, while the rest of the tags are utilized for each step.

[Figure 2]

- Figure 2: Differences between three thinking formats: (a) Unstructured thinking, where no XML structural tags are included; (b) Semi-structured thinking, which is similar to unstructured thinking but adds <thoughts> and <response> tags to separate thoughts and responses; (c) Structured thinking, which introduces additional XML tags for structural purposes in the thoughts section.

The intuition behind designing the thinking process in this way is that, at each step, we encourage the model to first generate a title summarizing what should be done, followed by details or intermediate results recorded in the scratchpad. Then, we instruct the model to summarize its progress and determine the next step based on the plan. This design aligns with the plan-and-solve prompting approach, where prompting an LLM to generate a plan first can enhance its performance (Wang et al., 2023a). A complete example of this structured thinking style is provided in Appendix A.4.

- 2.2 DATA PREPARATION

To prepare a training set, we select open datasets across five domains: 1) mathematics, 2) instruction following, 3) coding, 4) safety, and 5) finance. The rationale behind this mixture is that mathematics is known to improve STEM performance of LLMs (OLMo et al., 2025), instruction following ensures generalization of reasoning capabilities, and code serves as a strong anchor language for an LM to think through and is known to improve performance (Gao et al., 2023; Chen et al., 2023). Finally, safety is included to improve the safety of the model, which may degrade from the introduction of long thinking, and finance is included to provide a representative of domain-specific data, which we will utilize to experiment and observe its effect on overall performance.

- 2.2.1 DATA MIXTURE

The selected open datasets for each domain and their proportions are described in Table 5 in Appendix A.3. While the majority of the datasets provide a straightforward ground truth answer, PRM800K is different.

PRM800K is a dataset used to construct a process reward model, which is a different type of model from an LLM. Therefore, we create a simple script to convert choices of each step in each ground truth into one complete response. At each step, we randomly select the correct or incorrect step to be included. In the case of an incorrect step, the script adds an additional step reflecting that the step is incorrect and resamples another step. This way, we create a dataset containing behaviors of self-correction, one of the behaviors common in reasoning models (DeepSeek-AI, 2025), in our training set.

We also perform post-processing on UltraFeedback by selecting only high-quality responses, filtering for records with a rating of the chosen response higher than 4.25. Finally, we downsample some

large datasets through uniform sampling from their respective training splits. In total, the training set consists of 55,677 records.

- 2.2.2 PIPELINE FOR DATA TRANSFORMATION AND REFINEMENT

Nevertheless, the training at this point only consists of a typical train-split fine-tuning set and does not include long reasoning chains like those demonstrated by reasoning models. Therefore, we prepare an LLM-based transformation-and-refinement pipeline for generating long reasoning chains from the training set. The pipeline consists of two main stages: 1) few-shot structured thinking transformation and 2) response refinement. In the first stage, we use few-shot prompting (Brown et al., 2020) with GPT-4o mini (gpt-4o-mini-2024-07-18), along with ground truth associated with each query. There are a total of three few-shot exemplars (available in Appendix A.5), which are manually curated by combining and structuring multiple state-of-the-art LLM responses.

The reason for transforming each ground truth into a structured thinking format first is to allow conversion into semi-structured and unstructured thinking formats later through simple text replacements (i.e., removing some or all auxiliary XML tags). This ensures equivalent levels of information, which is crucial for evaluating the effectiveness of different thinking formats.

After transformation, we utilize Qwen2.5-32B-Instruct5 to refine the generated response, ensuring structural and informational correctness. We prompt the LLM to correct formatting issues, fill in missing content, and improve response quality based on generated thoughts. The same few-shot exemplars are used to guide this refinement process.

We derive semi-structured and unstructured thinking formats by systematically removing auxiliary XML tags. Specifically, we replace tags with words to maintain contextual consistency and strip away all structural elements except for <thoughts></thoughts> and <response></response> tags to create semi-structured data. In contrast, unstructured thinking data is obtained by removing or substituting all XML tags.

In total, we have approximately 67M tokens in our training set ready for SFT. The average number of tokens per instruction is 145.20, while the average number of tokens per output is 1,060.94, increased from the original 248.58 tokens. The maximum number of steps in the thinking part (excluding planning) is 24, with an average of 4.99 steps.

- 2.3 EVALUATION

We utilize the following six benchmarks from different domains to assess the model’s performance: 1) GSM8K (Cobbe et al., 2021) is a math word problem dataset used to evaluate mathematical performance; 2) HumanEval+ (Liu et al., 2023; 2024), which evaluates code generation, as the code domain is present in our training data; 3)IFEval (Zhou et al., 2023a) is selected to evaluate instruction-following performance; 4) GPQA (Rein et al., 2024) and 5) MMLU Pro (Wang et al., 2024b) are included as additional challenging QA benchmarks; and 6) ThaiExam (Pipatanakul et al., 2023) is a multiple-choice benchmark consisting of standard exams used to evaluate Thai students. We used this benchmark to evaluate the model’s performance in Thai, given that the original model was tuned for improved Thai language capabilities.

- 3 EXPERIMENTS

For all experiments in this section, we used the setup described in Appendix A.6. We conduct five experiments, each presented in its own subsection, to answer the following five research questions.

- Q1 Does the thinking format matter for reasoning models? If so, what is the best format?
- Q2 How much data is needed to train a reasoning model?
- Q3 In which domain of the dataset matters most when developing a reasoning model?
- Q4 How can a reasoning model be adapted to generate its reasoning trace in Thai?
- Q5 What are the differences when generating reasoning traces in Thai versus in English for a multilingual reasoning model? 5https://huggingface.co/Qwen/Qwen2.5-32B-Instruct

Table 2: Performance of models on each benchmark (higher is better). Bold indicates the best score in each column. Underlined scores denote improvements over the baseline, Typhoon 2 3B Instruct. We apply this convention across all results tables. Typhoon 2 refers to Typhoon 2 3B Instruct, and Typhoon T refers to its variant trained on a long-thinking dataset. SFT refers to supervised fine-tuning on the original datasets. All reasoning models show improvement over SFT on the original dataset.

Model GSM8K HumanEval+ IFEval GPQA MMLU Pro ThaiExam Typhoon 2

Zero-Shot 57.32 63.51 69.32 25.00 26.61 32.69 Zero-Shot CoT 53.83 0.00 68.95 25.45 23.36 33.27 SFT 20.62 46.24 17.74 16.74 13.96 15.65

##### Typhoon T

Unstructured 59.82 67.88 34.01 24.78 20.44 21.36 Semi-structured 57.24 72.87 55.27 27.68 19.46 21.92 Structured 62.02 69.76 53.60 27.23 23.56 22.84

- 3.1 STRUCTURED THINKING PROVIDES IMPROVEMENTS FOR MATHEMATICS AND CODING

To assess the impact of training a model on a long-thinking dataset and to evaluate the effects of various thinking formats, we designed six distinct scenarios. The results are presented in Table 2:

- • Typhoon 2 3B Instruct: These scenarios evaluate the base model.

- 1. Zero-shot: We prompt the model in a zero-shot manner to establish its baseline performance. Note that modern LLMs may have been exposed to CoT data during training; consequently, the model might exhibit chain-of-thought behavior even in zero-shot settings.
- 2. Zero-shot CoT: We prompt the LLM with the phrase “Let’s think step by step,” as suggested by (Kojima et al., 2022), to observe its performance when explicitly encouraged to reason before answering, without any additional training.
- 3. SFT: To compare standard supervised fine-tuning on a typical training set with training on a long-thinking dataset, we fine-tune the base model on a combined training split of non-longthinking data and then evaluate its performance using zero-shot prompting.

- • 4.-6. Typhoon T: This variant of Typhoon 2 3B Instruct is trained on a long-thinking dataset generated following the approach described in Section 2. We experiment with three model variants—Unstructured, Semi-structured, and Structured—each trained on its respective thinkingformatted dataset.

We observe an unexpected result when applying zero-shot CoT prompting with the base model: a slight performance decrease across most benchmarks, except for GPQA and ThaiExam, where marginal improvements are observed. This finding contradicts the common observations that zeroshot CoT prompting enhances model performance by eliciting reasoning. We hypothesize that this decline may be due to additional training steps on Thai data, which have affected LLM’s reasoning performance (Khade et al., 2025). Furthermore, zero-shot CoT prompting led the model to generate completely incorrect code, scoring 0.00, likely due to distractions from unhelpful reasoning traces.

Similarly, we observe a significant drop in performance when applying SFT with the non-longthinking datasets. This suggests symptoms of catastrophic forgetting (Luo et al., 2025), where the model shifts its distribution toward the fine-tuned data and loses generalization capabilities (Kotha et al., 2024). In contrast, SFT with long-thinking data does not exhibit the same behavior. Nevertheless, models fine-tuned on the long-thinking dataset show degraded performance in instructionfollowing tasks (IFEval). Additionally, ThaiExam scores decrease, which is expected given that the fine-tuning dataset primarily consists of English data, thereby reducing the model’s performance on Thai-language benchmarks.

We further analyzed the average response length on the MMLU Pro benchmark and found that Typhoon T models generally generate longer responses (more output tokens) than Typhoon 2 3B Instruct variants. This demonstrates the effectiveness of SFT on long-thinking datasets, except when prompting Typhoon 2 with zero-shot chain-of-thought reasoning, which generates long but unhelpful responses. Additional results and discussions are provided in Appendix A.7.

Overall, unstructured thinking appears to be suboptimal compared to other formats, while structured thinking yields improvements on a greater number of benchmarks. We attribute the superior performance of structured thinking to the presence of auxiliary separating tags, which help segment different reasoning components. Moreover, we observe that structured thinking results in more concise responses compared to semi-structured thinking, suggesting increased reasoning efficiency. This is likely due to the biases introduced by additional auxiliary structural tags.

Interestingly, we observe rare instances where the structured-thinking model, when prompted with a Thai query using non-zero temperature, generates its reasoning traces in Thai–despite not being trained on Thai long-thinking data. This suggests that the model may generalize its reasoning patterns across languages.

- 3.2 BALANCING DATA QUANTITY: THE KEY TO OPTIMAL REASONING MODEL PERFORMANCE

To evaluate the impact of data quantity on the performance of the reasoning model, we construct downsampled versions of the main training dataset. The baseline for this experiment is Typhoon T, trained on structured long-thinking data from the previous subsection. To create the downsampled datasets, we sample x% from each subdataset that comprises the full dataset, ensuring the overall data distribution is preserved. We consider a range of x values: {75,50,25,10,5}.

[Figure 3]

Figure 3: Increasing the proportion of the training set beyond 75% results in performance degradation for some datasets, while GSM8K generally shows a trend of performance improvement with the proportion.

We note that we fixed the number of epochs rather than the total training steps, meaning that models trained on smaller datasets undergo fewer optimization steps. This ensures that each subset is trained for a proportional duration relative to the full dataset without artificially extending training. If we had controlled for total training steps instead (e.g., by increasing epochs for smaller datasets), we might observe different trends, potentially increasing the risk of overfitting. The results of this experiment are presented in Figure 3. The full results in a table format is available in Appendix A.8.

The findings suggest that an excessive amount of data may not be optimal for achieving the best performance. However, reducing the dataset size too much also leads to a decline in instruction-following performance (IFEval). Similarly, MMLU Pro exhibits a downward trend in performance beyond a certain data threshold, 50% in this experiment.

In contrast, HumanEval+ benefits from an increase in training data. Among the tested configurations, training on 75% of the total dataset (41,755 records) yielded the most effective structured thinking reasoning model. We designate the model trained on this dataset as Typhoon T1-EN.

- 3.3 DON’T LEAVE SAFETY OUT WHEN TRAINING A REASONING MODEL

To assess the impact of each domain in the training set, we perform a leave-one-out experiment, where we train a model following the approach in Section 3.1 on a training dataset with one specific domain removed. We use the best-performing configuration from the previous subsection (75% dataset), which also serves as our baseline. We expect to observe a decrease in performance and measure the impact of each domain by the extent of this decrease. The full results and additional discussions are available in Appendix A.9.

Among all domains, the safety domain has the most impact on performance. Removing it results in substantial performance drops in GSM8K, HumanEval+, IFEval, and ThaiExam. This effect can be partly attributed to the role of the safety domain in aligning the model to be helpful (Wang et al., 2023c). Without this alignment, the model’s overall capabilities decline across multiple benchmarks. Interestingly, the model performs the same or better on challenging English multiple-choice benchmarks, such as GPQA and MMLU Pro.

Table 3: Performance of model variants on various benchmarks, evaluating the impact of additional training data. CSFT refers to continual SFT. Adding 1.5k samples improves IFEval and ThaiExam scores, while CSFT significantly reduces overall performance.

Model GSM8K HumanEval+ IFEval GPQA MMLU Pro ThaiExam

Typhoon T1-EN 62.09 70.60 49.54 30.80 27.39 21.71 + 1.5k, CSFT 41.39 65.79 33.83 23.66 4.30 21.20 + 1.5k 60.12 67.90 51.76 29.91 19.32 23.56 + 1k 61.94 66.77 50.09 24.55 23.48 21.57 + 0.5k 60.88 68.24 49.72 25.45 23.05 22.62

In summary, our findings emphasize the necessity of a diverse training set—including prompt diversity, response style diversity, and domain diversity. While instruction-following is useful, it may not be as crucial as previously assumed for training a model to generalize its reasoning behavior across prompts. Instead, safety-focused instruction-following datasets prove to be more beneficial.

- 3.4 TRAINING WITH THAI-TRANSLATED DATA IMPROVES THAI PERFORMANCE AT THE COST OF OTHERS

To equip the model with the ability to generate Thai reasoning traces–especially when the user’s prompt is in Thai or explicitly requests Thai output–we train a model with additional translated Thai content. To obtain this translated content, we construct a translation pipeline described in Appendix A.10.

In this subsection, we investigate the best approach for equipping the reasoning model with the ability to generate Thai reasoning traces. We experiment with two main approaches: (1) continual SFT, where we take Typhoon T1-EN and perform an additional SFT with the translated Thai dataset, and (2) SFT from scratch, where we train from the beginning using a mixture of the translated Thai dataset and the Typhoon T1-EN training data. We found that the continual SFT approach yielded subpar performance compared to SFT from scratch, as shown in Table 3.

To better understand the effect of incorporating Thai-translated content, we experimented with different proportions of the translated dataset: using 2/3 and 1/3 of the full translated dataset, approximately 1,000 and 500 records, respectively. The results are presented in Table 3.

We observe that all models trained with additional Thai-translated data exhibit improved instructionfollowing capabilities (IFEval), with the most improvement occurring when training on the full Thaitranslated dataset. Similarly, performance on ThaiExam follows this trend, showing improvements as more Thai training data is included. However, we also observe a decline in performance on other benchmarks, likely indicating a trade-off due to the inherent capacity limitations of the model’s 3B parameters (Li et al., 2024; Allen-Zhu & Li, 2025).

We observe that the models trained with Thai translated data are able to generate its thinking trace in Thai when prompted in Thai (see Appendix A.11), while maintaining performance similar to that of Typhoon T1-EN. The model that achieved the highest improvement on ThaiExam–trained with the full Thai-translated dataset–is named Typhoon T1.

In addition, we observe that on ThaiExam, the percentage of Thai characters in the thinking trace (i.e., the content between <thoughts> tags) increased from 16.67% when generated with Typhoon T1-EN to 95.49%. This increase in the number of Thai characters, along with improved performance on ThaiExam, demonstrates the effectiveness of training with only a small portion of Thai-translated data. We hope this approach sparks interest in further studies on equipping reasoning models with the ability to generate their reasoning traces in low-resource languages.

- 3.5 LET THE REASONING MODEL CHOOSES ITS OWN REASONING LANGUAGE

To better understand the effect of thinking language on the performance, we conduct an experiment using Typhoon T1 from the previous section. Specifically, we examine the performance implications of enforcing thought generation in either English or Thai through prompting. This differs from a zero-shot approach, used in the previous subsection, where we allow the model to choose its own reasoning language–typically English for English prompts and Thai for Thai prompts.

Table 4: EN denotes forced reasoning in English, and TH denotes forced reasoning in Thai. Constraining Typhoon T1 to reason in a specific language degrades overall accuracy. English reasoning is slightly more effective than Thai reasoning across most benchmarks. However, allowing the model to choose its own thinking language yields the best performance.

Model GSM8K HumanEval+ IFEval GPQA MMLU Pro ThaiExam

Typhoon T1 60.12 67.90 51.76 29.91 19.32 23.56 + EN 46.17 0.00 48.98 26.56 16.55 25.31 + TH 48.29 0.00 44.73 25.67 16.05 24.66

We employ a standardized user prompt, provided in Appendix A.12, to explicitly direct Typhoon T1 to generate its reasoning traces in either English or Thai before producing a final answer. The results, presented in Table 4, indicate that restricting the model’s reasoning language negatively impacts overall task performance across multiple benchmarks. Notably, while both forced English and Thai reasoning lead to performance degradation, English slightly outperforms Thai in most cases.

Our findings suggest that constraining the reasoning language of a multilingual model may disrupt its ability to leverage cross-linguistic representations effectively. The observed performance degradation when enforcing English or Thai reasoning highlights the model’s reliance on flexible multilingual processing for optimal results. This aligns with prior research indicating that multilingual models often synthesize knowledge across languages rather than strictly adhering to a single linguistic paradigm (Chen et al., 2024; Hua et al., 2024). One possible explanation for the superior performance of unconstrained reasoning is that Typhoon T1 dynamically selects the most effective linguistic structures for intermediate thought generation, drawing from both English and Thai as needed, similar to Huang et al. (2023; 2025). Furthermore, the zero accuracy observed in HumanEval+ for both forced-language settings suggests that forcing a language can lead to the generation of unhelpful reasoning traces, similar to what was observed in Section 3.1.

- 3.6 SUMMARY: TYPHOON T1-EN AND TYPHOON T1

We compare the final performance of each model against Typhoon T1 3B Instruct in Figure 4. Based on all experiments in this section, we found the most optimal configuration for training 3B-parameter reasoning models using the SFT approach as follows:

- • Typhoon T1-EN: Uses structured thinking with 41,755 records, trained using the standard settings described in Appendix A.6.
- • Typhoon T1: Extends Typhoon T1-EN’s approach by incorporating an additional training set of 1,565 Thai-translated structured long-thinking records, as described in Section 3.4. This enables the model to generate Thai reasoning traces while maintaining Thai performance.

[Figure 4]

Figure 4: Final performance comparison of Typhoon T1-EN and Typhoon T1 against the baseline Typhoon T1 3B Instruct model across six evaluation benchmarks.

- 4 CONCLUSION

We present an open recipe for developing the reasoning model Typhoon T1, which requires no distillation from other models, generalizes across domains, and produces reasoning traces in English or Thai. We also introduce structured thinking–using auxiliary XML tags to create efficient reasoning formats with fewer tokens. Although Typhoon T1 shows strong improvements on GSM8K, HumanEval+, and GPQA, our approach involves trade-offs in performance due to model size, such as reduced instruction following and Thai performance. We hope our insights on data size, domain, and reasoning language and artifacts accelerate research on reasoning models.

REFERENCES

Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=FxNNiUgtfa.

Edward Beeching, Lewis Tunstall, and Sasha Rush. Scaling test-time compute with open models, 2024. URL https://huggingface.co/spaces/HuggingFaceH4/ blogpost-scaling-test-time-compute.

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. Graph of Thoughts: Solving Elaborate Problems with Large Language Models. Proceedings of the AAAI Conference on Artificial Intelligence, 38(16):17682–17690, Mar. 2024. doi: 10.1609/aaai.v38i16.29720. URL https://ojs.aaai.org/index.php/AAAI/ article/view/29720.

Gaurang Bharti. Wealth Alpaca. https://huggingface.co/datasets/gbharti/ wealth-alpaca_lora, 2023.

Zhenni Bi, Kai Han, Chuanjian Liu, Yehui Tang, and Yunhe Wang. Forest-of-Thought: Scaling TestTime Compute for Enhancing LLM Reasoning, 2025. URL https://arxiv.org/abs/ 2412.09078.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language Models are Few-Shot Learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/ file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.

Nuo Chen, Zinan Zheng, Ning Wu, Ming Gong, Dongmei Zhang, and Jia Li. Breaking Language Barriers in Multilingual Mathematical Reasoning: Insights and Observations. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 7001–7016, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.411. URL https://aclanthology.org/2024.findings-emnlp.411/.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks, 2023. URL https://arxiv.org/abs/2211.12588.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training Verifiers to Solve Math Word Problems, 2021. URL https://arxiv. org/abs/2110.14168.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, and Maosong Sun. UltraFeedback: Boosting Language Models with Scaled AI Feedback, 2024. URL https://arxiv.org/abs/2310.01377.

Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, 2023. URL https://arxiv.org/abs/2307.08691.

DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, 2025. URL https://arxiv.org/abs/2501.12948.

Ruomeng Ding, Chaoyun Zhang, Lu Wang, Yong Xu, Minghua Ma, Wei Zhang, Si Qin, Saravan Rajmohan, Qingwei Lin, and Dongmei Zhang. Everything of Thoughts: Defying the Law of Penrose Triangle for Thought Generation. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 1638–1662, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/ v1/2024.findings-acl.95. URL https://aclanthology.org/2024.findings-acl. 95/.

Zhihua Duan and Jialin Wang. Prompt-Based Monte Carlo Tree Search for Mitigating Hallucinations in Large Models, 2025. URL https://arxiv.org/abs/2501.13942.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. PAL: Program-aided Language Models, 2023. URL https://arxiv.org/ abs/2211.10435.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzm´an, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur ¸Celebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, V´ıtor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel,

Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The Llama 3 Herd of Models, 2024. URL https://arxiv.org/abs/2407.21783.

Yuling Gu, Oyvind Tafjord, Bailey Kuehl, Dany Haddad, Jesse Dodge, and Hannaneh Hajishirzi. OLMES: A Standard for Language Model Evaluations, 2024. URL https://arxiv.org/ abs/2406.08446.

Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, and Mao Yang. rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking, 2025. URL https://arxiv.org/abs/2501.04519.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring Mathematical Problem Solving With the MATH Dataset. In

Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview.net/forum?id=7Bywt2mQsCe.

Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, and Yanning Chen. Liger Kernel: Efficient Triton Kernels for LLM Training, 2025. URL https://arxiv.org/abs/2410.10989.

Tianze Hua, Tian Yun, and Ellie Pavlick. mOthello: When Do Cross-Lingual Representation Alignment and Cross-Lingual Transfer Emerge in Multilingual Models? In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Findings of the Association for Computational Linguistics: NAACL 2024, pp. 1585–1598, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-naacl.103. URL https://aclanthology.org/ 2024.findings-naacl.103/.

Haoyang Huang, Tianyi Tang, Dongdong Zhang, Xin Zhao, Ting Song, Yan Xia, and Furu Wei. Not All Languages Are Created Equal in LLMs: Improving Multilingual Capability by Cross-LingualThought Prompting. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 12365–12394, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.826. URL https://aclanthology.org/2023.findings-emnlp.826/.

Xin Huang, Tarun Kumar Vangani, Zhengyuan Liu, Bowei Zou, and Ai Ti Aw. AdaCoT: Rethinking Cross-Lingual Factual Reasoning through Adaptive Chain-of-Thought, 2025. URL https:// arxiv.org/abs/2501.16154.

Zhen Huang, Haoyang Zou, Xuefeng Li, Yixiu Liu, Yuxiang Zheng, Ethan Chern, Shijie Xia, Yiwei Qin, Weizhe Yuan, and Pengfei Liu. O1 Replication Journey – Part 2: Surpassing O1-preview through Simple Distillation, Big Progress or Bitter Lesson?, 2024. URL https://arxiv. org/abs/2411.16489.

Omkar Khade, Shruti Jagdale, Abhishek Phaltankar, Gauri Takalikar, and Raviraj Joshi. Challenges in Adapting Multilingual LLMs to Low-Resource Languages using LoRA PEFT Tuning. In Kengatharaiyer Sarveswaran, Ashwini Vaidya, Bal Krishna Bal, Sana Shams, and Surendrabikram Thapa (eds.), Proceedings of the First Workshop on Challenges in Processing South Asian Languages (CHiPSAL 2025), pp. 217–222, Abu Dhabi, UAE, January 2025. International Committee on Computational Linguistics. URL https://aclanthology.org/2025.chipsal-1. 22/.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large Language Models are Zero-Shot Reasoners. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=e2TBb5y0yFf.

Suhas Kotha, Jacob Mitchell Springer, and Aditi Raghunathan. Understanding Catastrophic Forgetting in Language Models via Implicit Inference. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= VrHiF2hsrm.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Bespoke Labs. Bespoke-Stratos: The unreasonable effectiveness of reasoning distillation. www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoningdistillation, 2025. Accessed: 2025-01-22.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing Frontiers in Open Language Model Post-Training, 2025. URL https://arxiv.org/abs/2411.15124.

Xiang Li, Shizhu He, Fangyu Lei, JunYang JunYang, Tianhuang Su, Kang Liu, and Jun Zhao. Teaching Small Language Models to Reason for Knowledge-Intensive Multi-Hop Question Answering. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 7804–7816, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.464. URL https://aclanthology.org/2024.findings-acl.464/.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s Verify Step by Step. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=v8L0pN6EOi.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=1qvx610Cu7.

Jiawei Liu, Songrun Xie, Junhao Wang, Yuxiang Wei, Yifeng Ding, and Lingming Zhang. Evaluating Language Models for Efficient Code Generation. In First Conference on Language Modeling,

2024. URL https://openreview.net/forum?id=IBCBMeAhmC.

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning, 2025. URL https://arxiv.org/abs/2308.08747.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. WizardCoder: Empowering Code Large Language Models with Evol-Instruct, 2023.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Cand`es, and Tatsunori Hashimoto. s1: Simple test-time scaling, 2025. URL https://arxiv.org/abs/2501.19393.

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. Show Your Work: Scratchpads for Intermediate Computation with Language Models, 2021. URL https://arxiv.org/abs/2112.00114.

Skywork o1 Team. Skywork-o1 Open Series. https://huggingface.co/Skywork, November 2024. URL https://huggingface.co/Skywork.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, Michal Guerquin, Hamish Ivison, Pang Wei Koh, Jiacheng Liu, Saumya Malik, William Merrill, Lester James V. Miranda, Jacob Morrison, Tyler Murray, Crystal Nam, Valentina Pyatkin, Aman Rangapur, Michael Schmitz, Sam Skjonsberg, David Wadden, Christopher Wilhelm, Michael Wilson, Luke Zettlemoyer, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. 2 OLMo 2 Furious, 2025. URL https://arxiv.org/abs/2501.00656.

Kunat Pipatanakul, Phatrasek Jirabovonvisut, Potsawee Manakul, Sittipong Sripaisarnmongkol, Ruangsak Patomwong, Pathomporn Chokchainant, and Kasima Tharnpipitchai. Typhoon: Thai Large Language Models. arXiv preprint arXiv:2312.13951, 2023.

Kunat Pipatanakul, Potsawee Manakul, Natapong Nitarach, Warit Sirichotedumrong, Surapon Nonesung, Teetouch Jaknamon, Parinthapat Pengpun, Pittawat Taveekitworachai, Adisai NaThalang, Sittipong Sripaisarnmongkol, Krisanapong Jirayoot, and Kasima Tharnpipitchai. Typhoon 2: A Family of Open Text and Multimodal Thai Large Language Models, 2024. URL https://arxiv.org/abs/2412.13702.

Yiwei Qin, Xuefeng Li, Haoyang Zou, Yixiu Liu, Shijie Xia, Zhen Huang, Yixin Ye, Weizhe Yuan, Hector Liu, Yuanzhi Li, and Pengfei Liu. O1 Replication Journey: A Strategic Progress Report – Part 1, 2024. URL https://arxiv.org/abs/2410.18982.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 53728–53741. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/ file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf.

Nazneen Rajani, Lewis Tunstall, Edward Beeching, Nathan Lambert, Alexander M. Rush, and Thomas Wolf. No Robots. https://huggingface.co/datasets/HuggingFaceH4/ no_robots, 2023.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. ZeRO: Memory optimizations Toward Training Trillion Parameter Models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–16, 2020. doi: 10.1109/SC41405.2020.00024.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. In First Conference on Language Modeling, 2024. URL https://openreview. net/forum?id=Ti67584b98.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters, 2024. URL https://arxiv. org/abs/2408.03314.

NovaSky Team. Sky-T1: Train your own O1 preview model within $450. https://novaskyai.github.io/posts/sky-t1, 2025. Accessed: 2025-01-09.

Jun Wang, Meng Fang, Ziyu Wan, Muning Wen, Jiachen Zhu, Anjie Liu, Ziqin Gong, Yan Song, Lei Chen, Lionel M. Ni, Linyi Yang, Ying Wen, and Weinan Zhang. OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models, 2024a. URL https:// arxiv.org/abs/2410.09671.

Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2609–2634, Toronto, Canada, July 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.147. URL https://aclanthology.org/2023. acl-long.147/.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In The Eleventh International Conference on Learning Representations, 2023b. URL https://openreview.net/forum?id=1PL1NIMMrw.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024b. URL https://openreview. net/forum?id=y10DM6R2r3.

Zhilin Wang, Yi Dong, Jiaqi Zeng, Virginia Adams, Makesh Narsimhan Sreedhar, Daniel Egert, Olivier Delalleau, Jane Polak Scowcroft, Neel Kant, Aidan Swope, and Oleksii Kuchaiev. HelpSteer: Multi-attribute Helpfulness Dataset for SteerLM, 2023c. URL https://arxiv.org/ abs/2311.09528.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of Thought Prompting Elicits Reasoning in Large Language Models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/ forum?id=_VjQlMeSB_J.

Tianhao Wu, Janice Lan, Weizhe Yuan, Jiantao Jiao, Jason Weston, and Sainbayar Sukhbaatar. Thinking LLMs: General Instruction Following with Thought Generation, 2024. URL https: //arxiv.org/abs/2410.10630.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik R Narasimhan. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https:// openreview.net/forum?id=5Xc1ecxO1h.

Sumeth Yuenyong, Kobkrit Viriyayudhakorn, Apivadee Piyatumrong, and Jillaphat Jaroenkantasima. OpenThaiGPT 1.5: A Thai-Centric Open Source Large Language Model, 2024. URL https://arxiv.org/abs/2411.07238.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-Following Evaluation for Large Language Models, 2023a. URL https: //arxiv.org/abs/2311.07911.

Yucheng Zhou, Xiubo Geng, Tao Shen, Chongyang Tao, Guodong Long, Jian-Guang Lou, and Jianbing Shen. Thread of Thought Unraveling Chaotic Contexts, 2023b. URL https: //arxiv.org/abs/2311.08734.

Jin Ziqi and Wei Lu. Tab-CoT: Zero-shot Tabular Chain of Thought. In Anna Rogers, Jordan BoydGraber, and Naoaki Okazaki (eds.), Findings of the Association for Computational Linguistics: ACL 2023, pp. 10259–10277, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.651. URL https://aclanthology.org/ 2023.findings-acl.651/.

A APPENDIX

- A.1 LIMITATIONS AND FUTURE WORK

Due to computational constraints, we experiment with a relatively small model, consisting of 3B parameters. This introduces inherent limitations in the model’s reasoning capabilities and trade-offs in its performance (Li et al., 2024; Allen-Zhu & Li, 2025). However, our approach demonstrates promising improvements in reasoning ability through structured long-thinking approach. Additionally, we restrict our experiments to zero-shot prompting and do not incorporate test-time scaling techniques, such as those introduced in Snell et al. (2024), which could further enhance performance.

Several important properties remain unexplored in our study of small reasoning models, including the impact of the number of reasoning steps, variations in structured thinking pattern, and a deeper analysis on finer-grained reasoning behaviors such as self-correction and problem decomposition. In addition, we did not conduct a comparison with a reinforcement learning baseline, given obscure details and inconclusive standards on this approach at the time of writing. Future work should investigate these aspects to better understand their contributions to model reasoning capabilities.

- A.2 RELATED WORK

- A.2.1 CHAIN-OF-THOUGHT PROMPTING

Chain-of-Thought (CoT) prompting (Wei et al., 2022) is an approach used to elicit reasoning in LLMs through in-context reasoning exemplars (Brown et al., 2020). This approach improves the performance of LLMs by generating additional tokens–“thoughts”–preceding a final answer. Similarly, zero-shot CoT prompting (Kojima et al., 2022) elicits the generation of a reasoning chain

before the final answer without any in-context exemplars, relying instead on a simple phrase like “Let´s think step by step.”

Various extensions of CoT prompting (Yao et al., 2023; Besta et al., 2024; Zhou et al., 2023b) have been proposed to further extend the reasoning process, allowing LLMs additional time to explore alternatives, verify intermediate responses, and correct themselves before providing a final answer. However, the complexity of these approaches increases significantly as they involve additional components such as verifiers and memory mechanisms for maintaining state.

- A.2.2 TEST-TIME COMPUTE SCALING

Test-time compute scaling (Snell et al., 2024) refers to allocating a greater compute budget during inference, allowing LLMs to engage in more extensive reasoning before generating a final answer. Several methods exist to achieve this, most of which involve search-based techniques, such as Monte Carlo Tree Search (MCTS) (Ding et al., 2024; Duan & Wang, 2025), tree traversal (Yao et al., 2023; Besta et al., 2024; Bi et al., 2025), and other search strategies (Snell et al., 2024; Wang et al., 2024a). These methods are often paired with a reward model, either an outcome reward model (Beeching

- et al., 2024) or a process reward model (Snell et al., 2024). Some studies (Qin et al., 2024; Guan
- et al., 2025) suggest that reasoning models can be enhanced by fine-tuning LLMs on search traces. However, we find this approach computationally expensive, especially at scale. Our approach, in contrast, is simpler to implement and more cost-effective than the aforementioned methods.

- A.2.3 REASONING MODELS

Reasoning models, also referred to as “thinking LLMs” (Wu et al., 2024), represent a recent advancement in the field of large language models. These models have demonstrated effectiveness in solving complex benchmarks by generating extended reasoning chains, often scaling with problem difficulty (DeepSeek-AI, 2025). Reasoning models are typically developed by augmenting an LLM through additional supervised fine-tuning (SFT) (Muennighoff et al., 2025) or reinforcement learning (RL) (DeepSeek-AI, 2025).

Nevertheless, many works (Team, 2025; Labs, 2025; Huang et al., 2024) have also demonstrated the effectiveness of knowledge distillation from a reasoning model. However, this distillation-based approach provides only a shortcut and requires access to an existing reasoning model. To gain a deeper understanding of reasoning models, we argue that developing such models from scratch– without leveraging an existing reasoning model–is a more effective approach. This study adopts that perspective to gain better insights into reasoning models.

- A.3 DATA MIXTURE

Table 5 shows data mixture of the training set used in the experiments in Section 3. Figure 5 shows data distribution of each domain in percentage.

- A.4 AN EXAMPLE OF STRUCTURED THINKING

Given the following prompt: “A model of a park was built on a scale of 1.5 centimeters to 50 meters. If the distance between two trees in the park is 150 meters, what is this distance on the model?”, we can expect to see the following response (indentation is manually included for readability):

<thoughts> <plan>

- <step order="1">Understanding the scale relationship</step>

- <step order="2">Setting up the proportion</step>

- <step order="3">Cross-multiplying to solve for the model distance</step>

- <step order="4">Calculating the final result</step>

- <step order="5">Verifying the calculations</step>

</plan> <step>

<title>Understanding the scale relationship</title>

Table 5: Data mixture of the training set. Domain/Dataset #Records Mathematics 21,941

MATH (Hendrycks et al., 2021) 7,500 Tulu 3 SFT Personas Math Grade (Lambert et al., 2025) 7,497 PRM800K Phase 2 (Lightman et al., 2024) 5,809 PRM800K Phase 1 (Lightman et al., 2024) 808 O1 Journey (Qin et al., 2024) 327

Instruction Following 13,188 No Robots (Rajani et al., 2023) 9,500 UltraFeedback (Cui et al., 2024) 3,688

Coding 10,814 Evol codealpaca v1 (Luo et al., 2023) 5,564 Tulu 3 SFT Personas Code (Lambert et al., 2025) 5,250

Safety 5,300 HelpSteer (Wang et al., 2023c) 5,300

Finance 4,434 Wealth Alpaca (Bharti, 2023) 4,434

### Total 55,677

[Figure 5]

Figure 5: This figures show domain distribution of the training set for the experiments.

<scratch_pad> The scale given is 1.5 cm to 50 m. This means that every 1.5 cm on the

model corresponds to 50 meters in reality. </scratch_pad> <summary>I’m clarifying the scale relationship to ensure accurate

calculations.</summary>

<next_step>Setting up the proportion</next_step> </step> <step>

<title>Setting up the proportion</title> <scratch_pad> Let \( x \) be the distance on the model (in centimeters). The

proportion can be set up as follows:

\[ \frac{1.5 \text{ cm}}{50 \text{ m}} = \frac{x \text{ cm}}{150 \text{ m}} \] This represents the relationship between the model and the actual

distance. </scratch_pad> <summary>I’ve established the proportion based on the scale provided.</

summary>

<next_step>Cross-multiplying to solve for the model distance</next_step> </step> <step>

<title>Cross-multiplying to solve for the model distance</title> <scratch_pad> Cross-multiplying gives us: \[ 1.5 \cdot 150 = 50 \cdot x \] This simplifies to: \[ 225 = 50x \] </scratch_pad> <summary>I’ve cross-multiplied to form an equation that allows solving

for \( x \).</summary>

<next_step>Calculating the final result</next_step> </step> <step>

<title>Calculating the final result</title> <scratch_pad> Now, I will solve for \( x \): \[ x = \frac{225}{50} \] Calculating this gives: \[ x = 4.5 \text{ cm} \] </scratch_pad> <summary>I’ve calculated the model distance based on the proportion

derived earlier.</summary>

<next_step>Verifying the calculations</next_step> </step> <step>

<title>Verifying the calculations</title> <scratch_pad> To verify:

- - The original scale is 1.5 cm to 50 m, which means 1.5 cm on the model should equal 50 m in real life.

- - The calculation shows that 4.5 cm on the model corresponds to 150 m in reality, which checks out with the scale.

</scratch_pad> <summary>I’ve double-checked the calculations to ensure accuracy.</

summary>

<next_step></next_step> </step> </thoughts> <response> To find the distance between the two trees on the model, we use the scale

provided. ### **Given:**

- - Scale: 1.5 cm corresponds to 50 m.

- - Distance between the trees in reality: 150 m. ### **Step-by-step Calculation:**

- 1. **Set up the proportion:** \[ \frac{1.5 \text{ cm}}{50 \text{ m}} = \frac{x \text{ cm}}{150 \text{ m

}} \]

- 2. **Cross-multiply:** \[ 1.5 \cdot 150 = 50 \cdot x \] Simplifying gives: \[ 225 = 50x \]

- 3. **Solve for \( x \):** \[ x = \frac{225}{50} = 4.5 \text{ cm} \]

### **Final Answer:** The distance between the two trees on the model is **4.5 centimeters**. </response>

- A.5 FEW-SHOT EXEMPLARS FOR DATA TRANSFORMATION AND REFINEMENT

- • https://pastes.io/example-1
- • https://pastes.io/example-2
- • https://pastes.io/example-3

- A.6 EXPERIMENTAL SETUP

We employ the following setup and tools for all our experiments. We use vLLM (Kwon et al., 2023) for efficient inference in our data pipeline and LlamaFactory (Zheng et al., 2024) for efficient multi-GPU training. Models are trained mostly on 2xH100s.

We perform a standard SFT on the training set using the same training configurations for all experiments. We save and evaluate models every 200 steps, with 5% of the training set used as a validation set, and load the best model at the end. We train each model for two epochs with a learning rate of 2e-5 using a cosine scheduler without warmup steps. The batch size is set to 6 with 6 gradient accumulation steps. For efficient training, we leverage DeepSpeed ZeRO Stage 2 (Rajbhandari

et al., 2020), Flash Attention 2 (Dao, 2023), and Liger Kernel (Hsu et al., 2025) with bfloat16 precision. The training configuration is available in Appendix A.6.1.

For evaluation, we use the open-source evaluation platform olmes (Gu et al., 2024) for GSM8K, HumanEval+, IFEval, GPQA, and MMLU Pro. We use the default evaluation configuration for all experiments, except for modifying the maximum context length to approximately 16,384 tokens to allow reasoning models to think longer. For ThaiExam, we use our simple evaluation program that applies regular expressions to extract answers and evaluates using exact match criteria. The olmes’s evaluation configuration is available in Appendix A.6.2.

- A.6.1 TRAINING CONFIGURATION FOR LLAMAFACTORY

|### model model_name_or_path: scb10x/llama3.2-typhoon2-3b-instruct<br><br>### method stage: sft do_train: true finetuning_type: full deepspeed: examples/deepspeed/ds_z2_config.json<br><br>### dataset dataset: <dataset_name> template: llama3 overwrite_cache: true preprocessing_num_workers: 16<br><br>### output output_dir: outputs/<outupt_model_path> logging_steps: 10 save_steps: 200 plot_loss: true overwrite_output_dir: true<br><br>### train per_device_train_batch_size: 6 gradient_accumulation_steps: 6 learning_rate: 2.0e-5 num_train_epochs: 2.0 lr_scheduler_type: cosine warmup_ratio: 0.0 bf16: true ddp_timeout: 180000000 use_unsloth: false enable_liger_kernel: true load_best_model_at_end: true resume_from_checkpoint: false flash_attn: fa2<br><br>### eval val_size: 0.05 per_device_eval_batch_size: 1 eval_strategy: steps eval_steps: 200<br><br>|
|---|

- A.6.2 EVALUATION CONFIGURATION FOR OLMES

|olmes --model <model_name> --task gsm8k::olmes codex_humanevalplus::tulu ifeval::tulu gpqa::llama3 mmlu_pro:cot::none --model-type vllm -output-dir results/<model_name><br><br>|
|---|

Table 6: Average number of output tokens generated by each model on the benchmarks.

Model GSM8K GPQA MMLU Pro ThaiExam Typhoon 2

Zero-shot 104.61 384.78 130.41 21.90 Zero-shot CoT 741.97 1238.54 1697.96 149.19 SFT 72.22 479.55 91.25 587.95

##### Typhoon T

Unstructured 169.03 478.53 491.33 829.21 Semi-structured 170.20 795.38 487.39 900.90 Structured 102.96 466.21 293.23 995.04

- Table 7: Performance at different dataset sizes. Smaller dataset sizes can sometimes outperform the 100% baseline, particularly in GPQA and MMLU Pro.

### Dataset Size GSM8K HumanEval+ IFEval GPQA MMLU Pro ThaiExam

100% 62.02 69.76 53.60 27.23 23.56 22.84 75% 62.09 70.60 49.54 30.80 27.39 21.71 50% 61.87 64.59 48.80 29.46 27.63 20.36 25% 62.09 66.93 50.46 29.69 30.05 20.54 10% 60.20 65.51 50.65 29.24 29.03 21.07 5% 60.88 64.62 47.13 30.13 29.65 19.91

- A.7 AVERAGE OUTPUT TOKENS

This section presents the average number of output tokens generated by each model on the listed benchmarks. The results, obtained from the experiment described in Section 3.1.

The results in Table 6 reveal notable variations in the average output tokens across different model configurations and benchmarks. In particular, when prompted with zero-shot CoT phrases, the Typhoon 2 3B Instruct produces a substantially higher number of tokens—especially on GPQA and MMLU Pro—suggesting that the model is distracted by unhelpful chain-of-thought reasoning and achieves poor results.

In contrast, the supervised fine-tuned Typhoon 2 3B Instruct on non-long-thinking datasets generates more concise outputs on GSM8K and MMLU Pro, reflecting the style of the training set, while exhibiting a significant increase in output length on ThaiExam. The observation on ThaiExam is likely due to the training set not containing Thai records, making it out-of-training-distribution.

- A.8 DATA SIZE ABLATION STUDY RESULTS

- Table 7 shows a full result of the experiment in Section 3.2.

A.9 LEAVE-OF-OUT EXPERIMENT RESULTS

- Table 8 shows a full result of the experiment in Section 3.3.

Removing instruction-following results in a slight performance decrease across all benchmarks, except for ThaiExam. This outcome is expected, as instruction-following is a foundational skill for an LLM to perform well across tasks, since most tasks require a model to understand instructions to effectively utilize its domain knowledge. We note that the slight increase in ThaiExam performance may be due to the model being exposed to more English instruction-following data, thereby mitigating the impact of Typhoon fine-tuning on Thai data (Pipatanakul et al., 2024).

Excluding the mathematics domain from the training set leads to a decline in GSM8K scores, a mathematics-focused benchmark. However, we observe a notable increase in the mathematics subject of MMLU Pro, rising from 23.69% to 29.90%. A key difference between MMLU Pro and the other benchmarks is that MMLU Pro consists of multiple-choice questions.

- Table 8: Leave-one-out experiment results, assessing the impact of removing specific domains from training. red values highlight the largest performance drop in each column. The “-” symbol denotes the removal of the corresponding domain from training. Excluding mathematical reasoning strongly improves IFEval performance, while safety removal boosts MMLU Pro.

Model GSM8K HumanEval+ IFEval GPQA MMLU Pro ThaiExam Typhoon T1-EN 62.09 70.60 49.54 30.80 27.39 21.71

- - IF 59.59 69.57 46.58 29.02 26.34 22.64
- - Math 59.51 69.47 53.60 25.45 28.52 20.88

- - Code 56.94 64.24 41.96 27.68 27.65 19.57

- - Safety 56.71 64.35 41.59 30.13 29.38 17.19
- - Finance 61.94 67.06 50.65 27.90 20.45 18.68

One possible explanation for this phenomenon is the distribution shift. Since our training set primarily contains mathematics problems requiring numerical responses rather than multiple-choice answers, removing the mathematics domain may allow the model to better adapt to multiple-choice formats. This finding highlights the importance of data diversity–not only in topics but also in response formats.

Unexpectedly, removing the coding domain negatively affects performance across all datasets. This finding aligns with previous research suggesting that coding serves as a strong representation (Gao et al., 2023) and enhances reasoning capabilities in LLMs (Chen et al., 2023). We also observe a significant performance drop in IFEval, suggesting that logical instruction-following capabilities may be partially induced through exposure to coding datasets.

Finally, removing domain-specific datasets–finance-related data in this case–generally leads to slight performance drops across all benchmarks and a notable decrease in MMLU Pro. The decline is particularly evident in the economics subject of MMLU Pro, where accuracy drops from 40.05% to 25.24%. This result underscores the importance of domain-specific datasets when training a reasoning model using supervised fine-tuning.

- A.10 TRANSLATION PIPELINE

The translation pipeline consists of an LLM, a fine-tuned Llama 3.1 8B Instruct in our case, prompted to act as a translator while adhering to specific rules, such as preserving structural XML tags. The exact prompt used for this purpose is provided bekiw. We translated approximately 25,000 records from the original 100% training set of the Typhoon T structured long-thinking model referenced in Section 3.1. We applied strict post-processing criteria, including removing prefixes that were not part of the translated content and filtering out translated instructions containing XML tags. After this process, we retained 1,565 high-quality translated records.

|Translate the following content into Thai appropriately.<br><br>**Rules**:<br><br>- Do not translate the tags used for structural formatting. Keep all XML tags intact and include in the final response.<br><br>- Translate only the content in English. Do not translate any other languages.<br><br>- Consider the context and adjust accordingly. You may change the order or add additional content to ensure fluency and make the text sound natural.<br><br>- Generate only the translated content, without explanations or formatting tags.<br><br>- Translate all content, not just the response tags.<br><br>- Do not change to a new structure. Keep all XML tags: <thoughts>, <plan >, <title>, <summary>, <scratch_pad>, <response>, etc.<br><br><br>Ensure you strictly follow the rules. ###======###<br><br>|
|---|

- A.11 AN EXAMPLE OF THAI THINKING TRACE GENERATED BY TYPHOON T1

We prompted Typhoon T1-EN and Typhoon T1 using the following prompt at temperature 0: https://pastes.io/thai-prompt (basic mathematics problem in Thai), and obtained the following results, also shown in Figure 6:

- • Typhoon T1-EN: https://pastes.io/english-thinking
- • Typhoon T1: https://pastes.io/thai-thinking

We can now observe that the thinking trace—which previously required Typhoon T1-EN to think in English—is now in Thai, and both results are correct. We use the following functions to count number of English and Thai characters for Section 3.4.

|def count_thai_char(text): count = 0 for char in text:<br><br>if ord(char) >= 3584 and ord(char) <= 3711:<br><br>count += 1 return count<br><br>def count_eng_char(text): count = 0 for char in text:<br><br>if ord(char) >= 65 and ord(char) <= 122:<br><br>count += 1 return count<br><br>|
|---|

- A.12 FORCE THINKING IN LANGUAGE PROMPTS

We perform an experiment to force a reasoning model to think in a specific language in Section 3.5. The following prompt is used to elicit such model behavior. Note that {lang} is replaced with “Thai” or “English”.

|You always think in {lang} for the <thoughts>. <thoughts> only contains { lang} language. However, the final <response> should be in the same language as the following query.<br><br>|
|---|

[Figure 6]

#### Figure 6: This figure shows Typhoon T1’s Thai thinking trace and Typhoon T1-EN’s English thinking trace.

