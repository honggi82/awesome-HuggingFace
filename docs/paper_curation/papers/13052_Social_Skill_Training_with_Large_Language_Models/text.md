# arXiv:2404.04204v1[cs.CL]5Apr2024

## Social Skill Training with Large Language Models

[Figure 1]

[Figure 2]

[Figure 3]

#### Diyi Yang ⋆ Caleb Ziems ⋆ William Held ⋆ Omar Shaikh ⋆ Michael S. Bernstein John Mitchell

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Stanford University Georgia Institute of Technology diyiy@cs.stanford.edu cziems@stanford.edu wheld3@gatech.edu oshaikh@stanford.edu msb@cs.stanford.edu mitchell@cs.stanford.edu

#### Abstract

People rely on social skills like conflict resolution to communicate effectively and to thrive in both work and personal life. However, practice environments for social skills are typically out of reach for most people. How can we make social skill training more available, accessible, and inviting? Drawing upon interdisciplinary research from communication and psychology, this perspective paper identifies social skill barriers to enter specialized fields. Then we present a solution that leverages large language models for social skill training via a generic framework. Our AI Partner, AI Mentor framework merges experiential learning with realistic practice and tailored feedback. This work ultimately calls for cross-disciplinary innovation to address the broader implications for workforce development and social equality.

#### 1 Introduction

People need both general and domain-specific skills to succeed in home and work life (Dean, 2017). Specialized workers need not only technical competence, but also field-specific soft skills that extend broader social skill-sets. For example, mental health counselors use active listening (Nemec et al., 2017), a skill for building trust and empathy (DeVito, 2019; Ramsey and Sohi, 1997). Similarly, business managers employ conflict resolution (De Dreu et al., 2001) and strengthen team bonds (DeVito, 2019) with specialized strategies (Lipsky et al., 2003). Learning these skills may involve passive observation, trial-and-error, or explicit instruction, but ultimately, a learner will need deliberate practice (Giddens and Griffiths, 2006), as social skills are inherently interactive.

Learning environments for social skills can be inaccessible, especially when training is offered by experts in formal programs, which are expensive,

⋆Equal contribution.

time-consuming, and limited in availability. Existing mechanisms for practice and feedback largely rely on expert supervision, making training difficult to scale. In particular, there may be a shortage of professionally trained coaches (Hoffmann et al., 2023; Wiggan et al., 2021), and most coaches who can provide tailored feedback are not able to help the large number of people who need it.

Practicing with peers can be a viable alternative only if peers are experienced. Individuals may also find it challenging or unsafe to practice high-risk tasks. Many people, especially from underrepresented groups and disadvantaged populations, have limited opportunities and social capital to learn and practice their target field’s specialized skill frameworks, which can exacerbate social inequality (Ovink and Veazey, 2011). We argue that large language models can help make social skill training more accessible, safe, and inviting, with tailored feedback in realistic, virtual practice spaces.

In this position paper, we propose Social Skill Training via two complementary visions of AI assistance, and outline a roadmap for their implementation. The first vision is that of the AI Partner, which can provide a scalable solution to experiential learning through simulated practice. Already, research has shown that human role-play can effectively teach communication, cooperation, and leadership skills (Gjeraa et al., 2014; Deutsch et al., 2011). Compared to on-the-job training, simulations allow learners to assume fewer risks and opportunity costs. By making simulations accessible, the AI Partner will reduce the socioeconomic barrier to enter specialized fields. Our complementary vision is the AI Mentor, which will offer personalized feedback based on domain expertise and factual knowledge. Together, the AI Partner and AI Mentor (APAM) framework can merge experiential learning with realistic practice and tailored feedback. Our paper calls for cross-disciplinary innovation to address APAMs broad implications.

#### 2 LLMs for Characters and Simulation

Prior research has shown simulation-based learning to be a highly effective educational tool (Blair et al., 2007; Tambe et al., 1995; Chernikova et al., 2020). These studies worked with manually constructed simulation templates. In this work, we will focus on LLM-based simulations as a more flexible and scalable solution.

Prompted LLMs can effectively roleplay believable characters (Argyle et al., 2023; Park et al.,

- 2022), who operate in specific contexts with plausible behaviors (Park et al., 2023), realistic preferences (Horton, 2023) and decision-making strategies (Zhao et al., 2024), human-like negotiation tactics (Gandhi et al., 2023), and empirically-attested psychological responses (Aher et al., 2023). Agentbased simulations, powered by LLMs, have been used for understanding debates (Du et al., 2023), strategic communication (Xu et al., 2023), collaboration (Zhang et al., 2023), conflict (Hua et al.,
- 2023), online behavior (Ren et al., 2024), and even urban planning (Zhou et al., 2024b).

Towards AI Partner design, the set of methodologies is rapidly expanding. Prompting can further be used alongside reinforcement learning to update LLMs according to a set of high-level guiding principles called a constitution (Bai et al., 2022), which can be written in plain text for rapid prototyping (Petridis et al., 2023). Extensions of LLM-based dialogue models include architectures for character consistency (Touvron et al., 2023), and fine-tuning on datasets specific to character simulation and conversation (Thoppilan et al., 2022; Shuster et al., 2022b; Kwon et al., 2023).

The development of the AI Mentor requires special care. Unlike the AI Partner, which may simulate real-world misconceptions, the AI Mentor should stay grounded in recent expert knowledge, which may not be present in the model’s original training corpus. As a possible solution, Retrieval Augmented Generation (RAG; Lewis et al., 2020) can fetch relevant knowledge from external sources and dynamically update the prompt (Xu et al., 2021; Shuster et al., 2022a; Jiang et al., 2023; Khattab et al., 2022). The use of these approaches largely aids the process of leveraging knowledge from textbooks and other curated sources.

The APAM framework may employ LLMs as only one part of a larger system that integrates retrieval, tool use (Schick et al., 2024), constitutional decision making, and generation into a sin-

gle pipeline (Wang et al., 2024a; Wu et al., 2022; Yang et al., 2022) via planning modules (Shinn et al., 2023; Yao et al., 2022). Planning can rely on traditional search algorithms (Yao et al., 2024), or may separately prompt another LLM to evaluate, filter, and rank candidate plans(Shinn et al., 2023; Yao et al., 2022). Planning may even rely on LLM-generated code, which is then executed to filter viable candidates (Wang et al., 2023).

#### 3 The APAM Framework

We propose a generic framework for social skill training with an AI Partner and an AI Mentor (APAM). Both are critical. When a user wants to learn a new social skill, the AI Partner can help them practice a relevant scenario with simulated conversation. The AI Mentor can provide knowledge-grounded feedback at critical junctures of the simulation.

##### 3.1 AI Partner

Constructing and deploying an AI Partner is nontrivial for multiple reasons. First, it is difficult to maintain consistency in the stylistic, behavioral, and emotional characteristics of the simulated partner (Weston and Sukhbaatar, 2023). Second, faithful simulations require a high level of complexity and detail that align with the target domain. Third, simulations should follow an efficient curriculum that quickly and deeply educates the learner. Thus, the AI partner should exhibit a consistent, plausible, and instructive personality. Note that diversity is one key component in an instructive system, which requires the simulated AI partner to demonstrate diverse social and cultural attributes. Through these dimensions, LLMs offer an actionable path to realize the ideal AI Partner.

##### 3.2 AI Mentor

The development of AI Mentor heavily relies on domain expertise, context awareness, and feedback efficacy, in addition to consistency that AI partners exhibit. Domain expertise means the AI Mentor should cite and elaborate on theories or frameworks from the related literature to develop their feedback, such as psychotherapy, conflict resolution, or negotiation, rather than producing generic and broad responses. Context awareness means that the AI Mentor should ground its suggestions in both the current scenario and the learner’s knowledge state, rather than citing generic or random facts. This creates technical challenges surrounding the handling

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

practices with

facilitates learningfor

### Learner

AI Mentor AI Partner

Mode 1 Conversational Content

Mode 1 Rubber Ducks

Static Foundations

[Figure 15]

[Figure 16]

Mode 2 Theory-Grounded Suggestions

Mode 2 Peer Roleplay

Generative Capabilities

[Figure 17]

[Figure 18]

Mode 3 Standardized Partner

Mode 3 Structured Feedback

Generative Possibilities

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Figure 1: Modes of the APAM framework. As AI capabilities improve, the APAM framework develops from its basis in non-AI teaching practices towards the possibility of realistic simulated AI Partner learning scenarios augmented with AI Mentor feedback that can be personalized based on prior practice sessions between the User and the AI Partner. With LLMs, our prior work has shown that AI Mentors can effectively generate suggestions based on best practices (Hsu et al., 2023) and AI Partners can replicate many of the benefits of roleplay (Shaikh et al., 2023a).

of long context and heterogeneous input. Feedback efficacy means the mentor should personalize the communicative style, timing, specificity, and granularity of its feedback to most effectively empower the learner at a given moment. Feedback should also be empathetic, respectful, and appropriate to the cultural and social context.

APAM systems to refresh their knowledge. Even if AI partners have imperfections in simulation and AI mentors provide relatively rigid theoretical feedback, the APAM framework can provide benefits by structurally facilitating exploration and analytical self-reflection (e.g., rubber duck debugging) (Schon and DeSanctis, 1986; Ku and Ho, 2010). APAM focuses on empowering users to become more aware of where they struggle.

##### 3.3 Methodology

We now propose a generic methodology for Social Skill Training via LLMs in four steps: (i) understanding the social processes that underlie one’s desired skill (e.g., conflict resolution); (ii) designing an AI partner to simulate conversations that expose the learner to the target processes, allowing the learner to practice; (iii) creating an AI mentor to provide tailored feedback; (iv) integrating the two agents into a simulated environment for users to learn safely. These four steps ensure effective social skill training. It is only through simulation in Step ii that users can practice realistically, and domain knowledge in Step iii that the system can provide pedagogically effective feedback. Finally, we can determine the success of our system in Step iv when we run comparative user studies.

##### 3.4 Examples of APAM

There are many domains where APAM can improve learners’ skills. Table 1 samples five broad skill clusters (e.g. active listening) which translate into career-specific domains (mental health counseling) through field-specific frameworks (motivational interviewing). These broad skill examples come with highly-developed psychological tests and self-report inventories to assess learning objectives. Our framework is not limited only to such canonical examples; we will discuss how to evaluate APAM systems more broadly in §6.

Recent work already exemplifies the APAM framework. For instance, Sharma et al. (2023) developed Hailey, an AI Mentor that provides feedback to mental health supporters when writing empathetic messages. To promote better political en-

Beginners are the ideal audience for the APAM framework, but skilled workers could also use

###### Social Skill Clusters Active Listening

###### Conflict Avoidance

Conflict Resolution (Behfar et al., 2008)

Empathy (Smith, 2006)

Rhetoric (Aristotle, 1984)

(Rogers and Farson, 1957)

(Morris-Rothschild and Brassard, 2006)

The ability to prevent disagreements or differences of opinion.

The ability to understand another person’s experience

Listening to express understanding of the speaker’s intentions.

The ability to resolve disagreements or differences of opinion

The ability to present strong arguments for one’s beliefs

Description

Active Listening Attitude Scale (Mishima et al., 2000)

Dutch Test for Conflict Handling (Van der Vliert, 2013)

Dutch Test for Conflict Handling (Van der Vliert, 2013)

Facilitative Interpersonal Skills (Anderson et al., 2007) Application Domain

Jefferson Scale (Hojat et al., 2001)

Evaluation

Litigation (Singer, 1988) DomainSpecific Framework

Counseling (Nemec et al., 2017)

Classroom Management

Product Management (Lipsky et al., 2003)

Nursing

(Yu and Kirk, 2009)

Positive Behavioral Interventions and Supports (Bradshaw et al., 2012)

Motivational Interviewing (Moyers et al., 2014)

Alternative Dispute Resolution (Lipsky et al., 2003)

Person-Centered Nursing

CREAC Legal Writing Paradigm (Kraft, 2014)

(McCormack and McCance, 2006)

Learner Novice Therapist Teacher-in-Training Manager Nurse-in-Training Novice Litigator AI Partner Digitized Patient Virtual Student Simulated Dispute Digitized Patient Simulated Courtroom AI Mentor Expert Counselor Experienced Teacher Mediator Experienced Nurse Expert Lawyer

- Table 1: Different use cases of APAM framework. Therapists and other specialists depend on general skill clusters like active listening, which are formalized in domain-specific frameworks like motivational interviewing. In this example (left column), the AI partner might be a digitized patient, while the AI mentor is an expert counselor.

gagement online, Argyle et al. (2023) developed an AI Mentor system that can provide feedback on polite and validating discourse strategies. In the legal domain, Jiang et al. (2024) leveraged LLMs to help non-experts learn intricate legal concepts to make legal knowledge accessible for encouraging civic participation in democracy. Besides these examples, we now discuss three APAM applications in more detail: CARE for peer counseling, Rehearsal for conflict resolution, and GPTeach for education.

CARE (AI Mentor) Peer counseling platforms depend on effective volunteer counselors, but most volunteers do not have access to personalized learning resources. One of our recent works introduces CARE: an interactive AI Mentor that trains peer counselors with automatic suggestions (Hsu et al., 2023). During the practical training stage, CARE diagnoses which counseling strategies are most suitable in the given context and suggests tailored responses. Counselors can choose to select, modify, or ignore these suggestions. We find that this LLM-based system, trained on Motivational Interviewing strategies from counseling conversation data, significantly helps novice counselors respond to challenging situations.

Rehearsal (AI Partner) Conflict is an uncomfortable and unavoidable part of life, but people can learn conflict resolution through deliberate and

strategic practice. Since few people have access to the necessary training resources, we developed the Rehearsal (Shaikh et al., 2023a) system to provide these at scale. Rehearsal helps users practice conflicts with a believable simulated interlocutor (AI Partner), identify alternative conversational paths, and learn through feedback on how to apply specific conflict strategies (AI Mentor). With Rehearsal, users can practice predefined conflict scenarios, or define their own novel scenarios. Our between-subjects evaluation showed that Rehearsal significantly helps learners navigate later unaided conflict compared to control groups.

GPTeach (AI Partner) For a teaching assistant (TA), three important domain-specific skills are academic communication, classroom management, and pedagogy. Learning these skills on-the-job can be stressful (Eddy and Gaston-Gayles, 2013). However, novice TAs in particular rarely have the resources they need to develop these skills before entering the classroom. TA instruction is often limited to static guides written by expert TAs; the first time new TAs ever practice is with actual students. To reduce potential harms, TAs should have a space to practice their teaching skills beforehand. To this end, GPTeach (Markel et al., 2023) uses LLMs to simulate a programming course in which simulated students make mistakes and ask questions like real

students. This allows novice TAs to practice across a wide range of student behaviors.

#### 4 Vision for Safe Deployment

LLMs have strong potential as tools for social skill training because they can flexibly generate coherent and natural text without the need for extensive topic-specific engineering used by prior works (Biswas et al., 2005). However, this flexibility often comes with more limited controllability, making such deployment dangerous for high-risk scenarios like therapy or mental health crises.

Our APAM framework provides guidelines for how to safely use AI in social skill training by decomposing safe usage into a continuum. In this section, each safety recommendation is tailored to a specific level of system capabilities. The different modes below represent different capability clusters that one might foresee from AI Mentors and AI Partners. By selecting a mode dependent on current capabilities and limitations, one can safely deploy valuable LLMs without requiring solutions to difficult open technical safety questions.

##### 4.1 AI Partner Continuum

- Mode 1: Rubber Ducking Simulation-based learning is grounded in a wealth of crossdisciplinary research (Cherryholmes, 1966; Dorn, 1989; Randel et al., 1992; Kincaid et al., 2003; Brennan and Vos, 2013). Even simple, low-fidelity simulations can prove effective, and to demonstrate this, we will consider the least developed partner: a passive, inanimate object. The practice of explaining your ideas to a rubber duck is called “Rubber ducking.” Despite being a passive “partner,” the rubber duck helps learners identify mistakes through the power of social learning and explanation (Ku and Ho, 2010). While today’s LLMs are certainly more powerful than a rubber duck, this highlights how “partners” can be valuable and safe even without human-level capabilities.
- Mode 2: Peer Roleplay Current Partner technologies (e.g., Rehearsal) resemble untrained peers roleplaying unfamiliar situations. While these simulations share surface level characteristics with real social situations, they often lack nuance, especially for roles which peers may not have lived experience with (Matz and Ebner, 2010). Despite this shortcoming, roleplay has long been a valuable tool for curriculum design, since it can help move learners from abstract theories to real-world practice.

Mode 3: Standardized Partner In high-risk domains like therapy, AI Partners will need to maintain a higher standard of consistency and reproducibility than most LLM-based simulation systems. We call this higher standard the Standardized Partner, like the “Standardized Patients” from medical training (van der Vleuten and Swanson, 1990) who are professionals trained to reproducibly simulate a patient with specific personality traits and ailments. In Medicine, Standardized Patients can prepare students as effectively as expert practitioners which shows that expert-training AI may not require the development of expert AI. Achieving this requires addressing the stereotypes (Shaikh et al., 2022), caricature and tropes (Cheng et al., 2023a,b), as well as misinformation (Lin et al., 2021) produced by today’s LLMs.

##### 4.2 AI Mentor Continuum

- Mode 1: Conversational Content Where AI Partners help learners learn through experience, AI Mentors connect formal or theoretical knowledge to these experiences. Fundamentally, Mentors can also be grounded in non-AI teaching principles: when educational materials follow a conversational rather than formal style, learning outcomes improve consistently (Sorden, 2012). The simplest AI Mentors are systems that rephrase dense or distractingly formal academic texts into the most understandable register.
- Mode 2: Theory-Grounded Suggestions Instead of presenting theories in the abstract, systems can offer theory-grounded suggestions to remind learners of the expert theories more naturally. Importantly, the suggestion format does not require that the system has perfect knowledge, since learners can benefit from judging even imperfect suggestions, developing their own intuitions about when theory applies. CARE (Hsu et al., 2023) is one such work that tests the limits of these capabilities.
- Mode 3: Structured Feedback AI Mentors can be improved to provide structured, actionable, and personalized suggestions with a greater scope and time-scale than the local-level feedback of Mode

2. This would require reasoning over long, multiturn conversations to an extent not possible with the attention mechanisms and context length limitations of current LLMs (Liu et al., 2024). The technical requirements of such an AI Mentor may be greater than that of developing an AI Expert directly, as teaching a task can be more challenging

than performing a task. We believe this challenge is merited as AI Mentors can create lasting value even after the exposure to the AI Mentor system ends. This enhances rather than diminishes human expertise in critical areas and avoids creating an ongoing dependency on AI.

#### 5 Technical Challenges

To create effective avenues for social skill training, APAM-systems require work on concrete technical challenges. In this section, we outline a core subset, prioritizing long-term interaction, expertdriven design of APAM systems, personalization, and designing for end-user interaction.

##### 5.1 Optimizing Long-Term Interactions

Persona Consistency First, LLMs should remain consistent when simulating social skill training. When individuals practice with LLMs over multiple turns, an LLM should not "forget" aspects in the initial prompt, e.g. by providing feedback unrelated to the initial instruction or ignoring attributes of the roleplayed simulation. Like Weston and Sukhbaatar (2023), we suspect a limitation of the attention mechanism in LLMs. As the context window increases, models may place less attention on initial instructions. One avenue of future work involves designing modeling or prompting methods to enforce consistency. Already, Ghost Attention (Touvron et al., 2023) and System Two Attention (Weston and Sukhbaatar, 2023) offer technical solutions to maintaining instruction consistency. Beyond modeling solutions, benchmarking consistency across multi-turn skill training—either by collecting real-world interaction datasets or constructing a static benchmark—would highlight deficiencies, addressable by future work.

Conversational Grounding is a fundamental component of interpersonal interaction (Clark, 1996). We often take multiple turns to establish an utterance as common ground. Humans do not simply "follow instructions"—we ask clarification questions, follow up on underlying details, or repair assumptions made by other interlocutors (Clark and Schaefer, 1989). This is critical for social skill training: dialogue agents must build grounding with individuals before generating feedback. Current instruction-following LLMs, however, are trained on single-step interactions (Ouyang et al., 2022). Thus, contemporary LLMs generate text that assumes character-

istics about an individual (Shaikh et al., 2023b; Chiu et al., 2024). In educational or mental health contexts, assuming the source of struggles can result in irrelevant feedback (Graesser et al., 1995; Strumwasser et al., 1991; Wang et al., 2024b). Integrating multi-turn preference optimization into LLMs is one promising avenue for future work; Hong et al. (2023) and Andukuri et al. (2024), for example, explore RLHF and self-improvement methods to generate conversational grounding. However, identifying where and why humans take time to establish common ground across a diverse set of situations—and training LLMs to reflect this behavior—is still an open problem.

##### 5.2 Integrating Expert Frameworks

For training systems to be effective and safe (Demszky et al., 2023), they should be closely integrated with domain-specific expert skill frameworks like motivational interviewing (c.f., Table 1). With LLM agents, however, adherence to specific frameworks is not guaranteed. By default, LLMs generate text in an unconstrained fashion. Ensuring generation adherence to expert frameworks is a highly domain-specific process. For effective interaction, experts must first outline and demonstrate specific strategies and constraints (Agrawala et al., 2011). Learning how to integrate these constraints into an LLM—either by finetuning a new model, designing new constrained decoding processes (Keskar et al., 2019), or building prompting pipelines (Wu et al., 2022)—are important avenues for future work. Finally, APAM-based systems should allow a practitioner to reflect on theory in structured feedback or peer role-play (Schon and DeSanctis, 1986). For example, technical methods should enable users to explore counterfactual roleplays or feedback grounded in theory.

##### 5.3 Designing for End-User Control

While we focus on technical methods that improve social skill training for LLMs, we note that these methods should be amenable to user adjustments. If an individual decides to change the underlying framework used by an LLM for training, adjustments should be directly possible in the system itself. Systems like ConstitutionMaker (Petridis et al., 2023), for example, allow end-users to design and edit prompting principles through an interactive interface. Similarly, new technical methods should come with interactive complements that enable direct control (Shneiderman, 1983). Since

training systems are inherently user-facing, designing interactive interfaces that allow for accessible control—either by an expert or learner—will allow individuals to customize the type of training they receive, instead of depending on a researcher.

##### 5.4 Personalizing Skill Training

Personalization is a key challenge even for general tasks around LLMs (Mysore et al., 2023; Tan et al., 2024; Wu et al., 2021). This connects to the consistency and diversity attributes of AI Partner, as well as feedback efficacy of AI Mentor. Effective skill training tailors feedback and experiences to meet the needs of each learner. Such personalized training has been made possible via APAM as learners can select and design AI partners or mentors that are relevant to them. It is, however, not trivial to operationalize personalization (Flek, 2020; Dudy et al., 2021). Prior research has investigated various writing styles (e.g., formal vs informal, simplified vs sophisticated language usage) (Alhafni et al., 2024; Li et al., 2023), and learners’ expertise in certain topics (Bernacki et al., 2021). Building upon these prior studies and taking into account the appropriate set of personalizationrelated attributes—as well as learners’ knowledge or expertise backgrounds (Huang et al., 2012)becomes increasingly important.

#### 6 Evaluation

The evaluation of AI partners and AI mentors is a major challenge; tools based on APAM involve complex computational systems and interaction with users with varying desires and backgrounds. To develop these training tools as a field, evaluation measures need to move beyond metrics traditionally used in Natural Language Processing to protocols from multiple relevant fields and stakeholders. Including multidisciplinary perspectives will help evaluate the empirical performance of such systems, their usability from a user perspective, and their long-term impacts on both users and their communities.

At present, research on text generation focuses largely on intrinsic evaluations which assess the quality of outputs with predefined rubrics or interactions. In Table 2, we separate these into fully automated evaluations and user-driven evaluations. Reference-based metrics, such as perplexity or Kullback–Leibler divergence (Theis et al., 2016), are common automated assessments of system qual-

ity, as they are both simple and allow for a rich definition of desired behavior through demonstrations. While many works aim to optimize metric reliability (Hashimoto et al., 2019), they often lose statistical power as researchers implicitly optimize metrics on fixed datasets (Goyal et al., 2023).

In pursuit of more flexible assessments, practitioners often turn to human qualitative assessments of systems, either through Likert scale-based scoring or comparative ranking of systems. In these procedures, system developers develop a rubric of desirable qualities such as believability, helpfulness, and consistency. These metrics are a useful gauge of the quality of these interactive systems.

However, as systems become more generally coherent and rubrics become more fine-grained, these methods of human validation often raise reproducibility concerns (Karpinska et al., 2021). While LLMs themselves are increasingly used to replace the human annotators in these processes (Dubois et al., 2024), this raises separate concerns about the systemic judgment biases of the LLM as a judge (Zheng et al., 2024). As such, other studies have focused on more coarse, functional metrics from user studies such as the Recommender Score (Markel et al., 2023) or the rate at which users make use of system outputs (Hsu et al., 2023).

To develop effective evaluations of more powerful systems, we believe domain users need to be involved as collaborators, rather than just as annotators. Potential users are best placed to assess the intrinsic measures that make a system usable, confusing, or even harmful. In current procedures, users are assigned to predefined roles assessing systems along strictly defined rubrics created by the system designers which centers the process on the developer’s preconceptions (Birhane et al., 2022).

Resolving this is not a simple matter of involving different individuals for the above evaluation metrics. Instead, potential users should be involved as stakeholders in high-level design—before development begins—to center the process around recognizing end-user expertise. For example, involving experts in the design of a training tool may highlight pedagogical theory overlooked by researchers. Watching end-users interact with prototype APAM systems will highlight opportunities for new technical methods or interactions. Practices from participatory design in HCI can serve as helpful guidelines for APAM platforms (Muller and Kuhn, 1993; Schuler and Namioka, 1993).

Ultimately, however, extrinsic evaluation of how

Intrinsic Evaluation

Metric Type Description Examples Applicability Category Reference Based Evaluation

Metrics of the similarity and distinguishability between a systems interactions and a set of gold standard interactions.

(Hashimoto et al., 2019) APAM Automated

Assessment of relevance of topics covered compared to expectations.

Topic Analysis

(Cheng et al., 2023b) AP

Using trained classifiers to categorize the frequency of known effective and realistic behaviours.

Classifier Based Scoring

(Sharma et al., 2024) APAM

Prompting LLMs to act as automated judges which provides Likert scale scores for a simulation.

(Zhou et al., 2023; Dubois et al., 2024)

LLM Prompt Scoring

AP

Comparative metric where systems are ranked based on a rubric of evaluation.

(Park et al., 2023; Zhou et al., 2024a)

Human Ranking

APAM

Likert Scale ratings of the system along given a rubric of evaluation.

Human Scoring

(Thoppilan et al., 2022) APAM

Rate at which participants utilize suggestions provided by an AI mentor system.

Suggestion Usage

(Hsu et al., 2023) AM User

Rating of how likely a user would be to recommend the system to a friend.

Recommender Score

(Markel et al., 2023) APAM Extrinsic Evaluation

Metric Type Description Examples Applicability Category Behavioral Impacts

Changes in qualitatively coded participant behaviors before and after exposure to the system.

(Shaikh et al., 2023a; Markel et al., 2023)

APAM Short-Term

Changes in participants’ self-reported efficacy on the skills practiced before and after exposure.

Self-Efficacy Reports

(Shaikh et al., 2023a) APAM

Changes in participant scores on closed-ended assessments of knowledge about the skills practiced.

Standardized Evaluation

(Shaikh et al., 2023a) APAM Short-Term Economic Outcomes

Impacts of a training program on participants short-term wages and employment.

(Adhvaryu et al., 2018; Chioda et al., 2021)

APAM

(Oreopoulos and Salvanes, 2011; Heckman and Kautz, 2012)

Impacts on non-financial measures such as health, risk-taking behaviors, and levels of societal trust.

Non-Financial Benefits

APAM Long-Term

(Barrera-Osorio et al., 2023)

Long-Term Economic Outcomes

Impacts of a training program on long-term earnings, workplace stability, and economic mobility.

APAM

- Table 2: Intrinsic and Extrinsic Evaluation Procedures applicable to APAM systems from prior work. At present, Natural Language Processing practitioners primarily focus on intrinsic evaluations for their systems. Here, we stress the importance of evaluating APAM systems using established measures for educational outcomes.

interaction with a tool changes the behavior of the individuals who use it. In the case studies we cover, measurable changes in behavior, selfefficacy reports, and standardized test scores have been used to evaluate short-term impacts. Whether these shifts are sustained and whether they create beneficial outcomes for participants, however, requires moving beyond what is currently evaluated in NLP. Educators and economists, however, have deep experience designing studies to evaluate the long-term impacts of training tools and education. This is most commonly utilized for economic outcomes, since these are the most straightforwardly measured benefits of soft-skills training (Adhvaryu et al., 2018; Chioda et al., 2021; Barrera-Osorio et al., 2023). The benefits of soft-skills training have additionally been measured through social outcomes that correlate with general quality of life (Oreopoulos and Salvanes, 2011; Heckman and Kautz, 2012).

We believe NLP researchers will develop more impactful systems by taking both intrinsic and ex-

trinsic evaluation perspectives. As specific methods begin to show promise through intrinsic automated metrics, they can begin to be utilized as part of human-centered systems design processes. Once a particular system is deemed satisfactory and capable by initial users, it can begin real-world trials to assess impacts from long-term use. This form of development—where deployment is staged alongside increasingly broad evaluation criteria—is key to the safe rollout of research with both high impact and potential risk (Mohs and Greig, 2017). For APAM and other NLP systems where value is derived through direct user interaction, we benefit by learning best practices specific to each domain.

Finally, given the high stakes but relatively lowcost of access to these tools, providing algorithmic auditors transparent access to the systems throughout this process should be standard practice in the development of APAM systems (Raji et al., 2020). Risk factors and adverse events (e.g., simulation failures, hallucination, over-reliance) stemming from any of these evaluation procedures should

be released in detail, rather than reported in aggregate, in order to facilitate external analysis of possible trends such as via the use of best practices in medical trials (FDA, 2009).

- 7 Discussion 7.1 Societal Impact

The APAM framework can help in a variety of societal applications. Broadly, APAM can be designed to help increase soft skills like self-awareness, social awareness, self-regulation, and relationship building, all of which can lead to personal wellbeing and broader social good outcomes (Jagers et al., 2018). Take the soft skill of self-awareness as an example: self-awareness is an understanding of personal and cultural identity, which correlates with physical wellbeing (Taylor and Usborne, 2010; Schwartz et al., 2008), mental health (Bhugra and Becker, 2005), and beneficial learning outcomes (Altugan, 2015). Psychologically, self-awareness is a foundation for optimism, confidence, and a sense of agency. Relatedly, training self-regulation skills like stress management and intrinsic motivation via APAM can lead to healthy lifestyle choices (Antoni et al., 2006). Relationship building skills like conflict resolution and communication enhance group learning and cooperative work, but most importantly for the individual, these strong relationships are expected to provide critical social support (Seeman et al., 2001) and a higher quality of life (Cohen, 2004). Additionally, skills of empathy and perspective-taking form a broader social awareness and are the foundations of strong citizenry (Wray-Lake and Syvertsen, 2011). Collectively, our APAM framework is expected to provide both individual and societal benefits in a more equitable manner, as it is designed to provide social learning opportunities to everyone.

APAM is a curriculum design tool that could enable learners to co-construct their learning paths could empower people to actively discover, define, and fill new niches in the economy. Some concrete applications of APAM are achievable in the short term. However, the potential impact of this broader vision necessitates further exploration. This requires new experts to design, train, and maintain AI tooling (Wilson et al., 2017) for APAM, as well as curriculum design experts to assess when and where true practice and mentorship is irreplaceable.

Even in areas where current systems for social learning of domain-critical soft skills exist, they

are often inaccessible, especially to the socioeconomically disadvantaged. Beyond improving the quality of existing outcomes, APAM could make the same high-quality learning environments available to everyone. AI alone is unlikely to solve all educational and social inequity. However, by focusing on skills that are often learned informally within social groups, APAM can reduce structural inequities which often compound for the already disadvantaged across generations (Oded, 2011).

##### 7.2 Concerns and Mitigation

Despite APAM’s benefits, there are still a set of issues we need to be aware of when building systems that can train social skills:

Stereotypes. LLM simulations can output caricatures (Cheng et al., 2023b) when prompted with broad characteristics such as race and gender. Often, stereotypes arise from under-description in the prompt (e.g., stereotypically casting a "boss" as a white male). We recommend that system designers highlight and encourage users to specify attributes that are important to the simulation (e.g., gender), enabling users to make changes as needed in the interface. Existing metrics of caricature can be used to raise warnings to users when simulations drift towards stereotypes while giving users full control over the personas with which they practice.

Distributional Shifts. APAM is designed primarily as a safe training environment in which users can build, practice, and transfer their social skills to the real world. We recommend that system designers should identify and clearly communicate the limitations of the simulation and its deviations from reality. We also recommend that any system based on APAM should take a human-centered development process, observing and interviewing real users to understand the system’s behaviors and gather feedback for iteration. Such evaluation will help track the nature and extent of any discrepancies between the simulation and the real world. Finally, to guard against users who might rely on the system too heavily, we recommend that the system caution users against overuse.

Job Risks. APAM is not designed as a direct replacement for paid human labor. Instead, APAM is a curriculum design tool that allows learners to co-construct their learning paths, empowering people to actively discover, define, and fill niches in the economy. At the corporate level, social skill

training programs will still need professional supervision, and these professionals can use automated tools for training events, just as they might have used a static curriculum or textbook in the past. Some individuals may opt for a cheap or free standalone option if they are on the margin. As such, we weigh this risk against the potential benefit of a free-to-use tool, which can assist a broader user population, especially those without professional training or social capital. Professional experts will be able to focus on more tailored, challenging scenarios for skill training, while still maintaining their high level of expertise and uniqueness.

#### 8 Summary and Outlook

This perspective paper examines a widespread challenge: mastering essential social skills for both personal and professional success. Opportunities to practice these skills in a safe learning environment are limited, especially for underprivileged groups. We show how LLMs can help create environments where everyone can practice social skills via our proposed AI Partner and AI Mentor framework. Here, the AI Partner offers a risk-free practice environment, while the AI Mentor provides knowledgeable tailored advice.

Below we highlight a few take-aways that illustrate how this approach can reshape social skills learning moving forward. Firstly, utilizing LLMs on APAM requires addressing multiple technical challenges, such as enhancing the simulation of AI partners to exhibit a consistent, plausible and instructive personality, and building AI mentors to have context awareness, domain expertise and feedback efficiency. Secondly, deploying LLM based social skill training systems has the potential to amplify limitations such as hallucinations and biases, thus our APAM framework offers a roadmap for how to use LLMs for social skill training by breaking safe usage into a continuum dependent on current capabilities. That is, the safe deployment of APAM should emphasize a gradual, risk-aware approach, as controllability and consistency improve for LLMs. Additionally, training social skills via LLMs might suffer from stereotypes and biases in LLM based simulation, distribution shifts and user reliance, as well as potential risks around job replacement. We recommend system designers take a human-centered development process, together with formative evaluations that iterate on the system’s behaviors using feedback from observation

sessions with real users.

Overall, the success of APAM depends on fostering interdisciplinary collaborations and team science across diverse research fields and across both academic and professional communities. Such a balanced, intentional, and collaborative process is essential for using LLMs for social good, particularly in areas such as social skill training.

#### References

Achyuta Adhvaryu, Namrata Kala, and Anant Nyshadham. 2018. The skills to pay the bills: Returns to on-the-job soft skills training. Technical report, National Bureau of Economic Research.

Maneesh Agrawala, Wilmot Li, and Floraine Berthouzoz. 2011. Design principles for visual communication. Commun. ACM, 54(4):60–69.

Gati V Aher, Rosa I Arriaga, and Adam Tauman Kalai. 2023. Using large language models to simulate multiple humans and replicate human subject studies. In International Conference on Machine Learning, pages 337–371. PMLR.

Bashar Alhafni, Vivek Kulkarni, Dhruv Kumar, and Vipul Raheja. 2024. Personalized text generation with fine-grained linguistic control. ArXiv preprint, abs/2402.04914.

Arzu Sosyal Altugan. 2015. The relationship between cultural identity and learning. Procedia-Social and Behavioral Sciences, 186:1159–1162.

T Anderson, CL Patterson, and AC Weis. 2007. Facilitative interpersonal skills performance analysis rating method. Unpublished coding manual, Department of Psychology, Ohio University, Athens, OH.

Chinmaya Andukuri, Jan-Philipp Fränken, Tobias Gerstenberg, and Noah D Goodman. 2024. Star-gate: Teaching language models to ask clarifying questions. ArXiv preprint, abs/2403.19154.

Michael H Antoni, Suzanne C Lechner, Aisha Kazi, Sarah R Wimberly, Tammy Sifre, Kenya R Urcuyo, Kristin Phillips, Stefan Glück, and Charles S Carver. 2006. How stress management improves quality of life after treatment for breast cancer. Journal of consulting and clinical psychology, 74(6):1143.

Lisa P Argyle, Christopher A Bail, Ethan C Busby, Joshua R Gubler, Thomas Howe, Christopher Rytting, Taylor Sorensen, and David Wingate. 2023. Leveraging ai for democratic discourse: Chat interventions can improve online political conversations at scale. Proceedings of the National Academy of Sciences, 120(41):e2311627120.

Aristotle. 1984. Rhetoric. Modern Library, New York. Translated from the Greek.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. 2022. Constitutional ai: Harmlessness from ai feedback. ArXiv preprint, abs/2212.08073.

Felipe Barrera-Osorio, Adriana Kugler, and Mikko Silliman. 2023. Hard and soft skills in vocational training: Experimental evidence from colombia. The World Bank Economic Review, 37(3):409–436.

Kristin J Behfar, Randall S Peterson, Elizabeth A Mannix, and William MK Trochim. 2008. The critical role of conflict resolution in teams: A close look at the links between conflict type, conflict management strategies, and team outcomes. Journal of applied psychology, 93(1):170.

Matthew L Bernacki, Meghan J Greene, and Nikki G Lobczowski. 2021. A systematic review of research on personalized learning: Personalized by whom, to what, how, and for what purpose (s)? Educational Psychology Review, 33(4):1675–1715.

Dinesh Bhugra and Matthew A Becker. 2005. Migration, cultural bereavement and cultural identity. World psychiatry, 4(1):18.

Abeba Birhane, William Isaac, Vinodkumar Prabhakaran, Mark Diaz, Madeleine Clare Elish, Iason Gabriel, and Shakir Mohamed. 2022. Power to the people? opportunities and challenges for participatory ai. In Proceedings of the 2nd ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization, pages 1–8.

Gautam Biswas, Krittaya Leelawong, Daniel Schwartz, Nancy Vye, and The Teachable Agents Group at Vanderbilt. 2005. Learning by teaching: A new agent paradigm for educational software. Applied Artificial Intelligence, 19(3-4):363–392.

Kristen Blair, Daniel L Schwartz, Gautam Biswas, and Krittaya Leelawong. 2007. Pedagogical agents for learning by teaching: Teachable agents. Educational Technology, pages 56–61.

Catherine P Bradshaw, Tracy E Waasdorp, and Philip J Leaf. 2012. Effects of school-wide positive behavioral interventions and supports on child behavior problems. Pediatrics, 130(5):e1136–e1145.

Ross Brennan and Lynn Vos. 2013. Effects of participation in a simulation game on marketing students’ numeracy and financial skills. Journal of Marketing Education, 35(3):259–270.

Myra Cheng, Esin Durmus, and Dan Jurafsky. 2023a. Marked personas: Using natural language prompts to measure stereotypes in language models. ArXiv preprint, abs/2305.18189.

Myra Cheng, Tiziano Piccardi, and Diyi Yang. 2023b. Compost: Characterizing and evaluating caricature in llm simulations. ArXiv preprint, abs/2310.11501.

Olga Chernikova, Nicole Heitzmann, Matthias Stadler, Doris Holzberger, Tina Seidel, and Frank Fischer. 2020. Simulation-based learning in higher education: A meta-analysis. Review of Educational Research, 90(4):499–541.

Cleo H Cherryholmes. 1966. Some current research on effectiveness of educational simulations: Implications for alternative strategies. American Behavioral Scientist, 10(2):4–7.

Laura Chioda, David Contreras-Loya, Paul Gertler, and Dana Carney. 2021. Making entrepreneurs: Returns to training youth in hard versus soft business skills. Technical report, National Bureau of Economic Research.

Yu Ying Chiu, Ashish Sharma, Inna Wanyin Lin, and Tim Althoff. 2024. A computational framework for behavioral assessment of llm therapists. ArXiv preprint, abs/2401.00820.

Herbert H Clark. 1996. Using language. Cambridge university press.

Herbert H Clark and Edward F Schaefer. 1989. Contributing to discourse. Cognitive science, 13(2):259– 294.

Sheldon Cohen. 2004. Social relationships and health. American psychologist, 59(8):676.

Carsten KW De Dreu, Arne Evers, Bianca Beersma, Esther S Kluwer, and Aukje Nauta. 2001. A theorybased measure of conflict management strategies in the workplace. Journal of Organizational Behavior: The International Journal of Industrial, Occupational and Organizational Psychology and Behavior, 22(6):645–668.

Susan A Dean. 2017. Soft skills needed for the 21st century workforce. Walden University.

Dorottya Demszky, Diyi Yang, David S Yeager, Christopher J Bryan, Margarett Clapper, Susannah Chandhok, Johannes C Eichstaedt, Cameron Hecht, Jeremy Jamieson, Meghann Johnson, et al. 2023. Using large language models in psychology. Nature Reviews Psychology, 2(11):688–701.

Morton Deutsch, Peter T Coleman, and Eric C Marcus.

2011. The handbook of conflict resolution: Theory and practice. John Wiley & Sons.

Joseph A DeVito. 2019. The interpersonal communication book. Instructor, 1(18):521–532.

Dean S Dorn. 1989. Simulation games: One more tool on the pedagogical shelf. Teaching Sociology, pages 1–18.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. 2023. Improving factuality and reasoning in language models through multiagent debate. ArXiv preprint, abs/2305.14325.

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2024. Alpacafarm: A simulation framework for methods that learn from human feedback. Preprint, arXiv:2305.14387.

Shiran Dudy, Steven Bedrick, and Bonnie Webber. 2021. Refocusing on relevance: Personalization in nlg. In Proceedings of the Conference on Empirical Methods in Natural Language Processing. Conference on Empirical Methods in Natural Language Processing, volume 2021, page 5190. NIH Public Access.

Pamela L Eddy and Joy L Gaston-Gayles. 2013. New faculty on the block: Issues of stress and support. In Faculty stress, pages 89–106. Routledge.

FDA. 2009. Adverse event reporting to irbs improving human subject protection. Guidance Clinical Investigators, Sponsors, and IRBs.

Lucie Flek. 2020. Returning the N to NLP: Towards contextually personalized classification models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7828– 7838, Online. Association for Computational Linguistics.

Kanishk Gandhi, Dorsa Sadigh, and Noah D Goodman.

2023. Strategic reasoning with language models. ArXiv preprint, abs/2305.19165.

Anthony Giddens and Simon Griffiths. 2006. Sociology. Polity.

Kirsten Gjeraa, Thea Palsgaard Møller, and D Østergaard. 2014. Efficacy of simulation-based trauma team training of non-technical skills. a systematic review. Acta Anaesthesiologica Scandinavica, 58(7):775–787.

Tanya Goyal, Junyi Jessy Li, and Greg Durrett. 2023. News summarization and evaluation in the era of gpt-3. Preprint, arXiv:2209.12356.

Arthur C Graesser, Natalie K Person, and Joseph P Magliano. 1995. Collaborative dialogue patterns in naturalistic one-to-one tutoring. Applied cognitive psychology, 9(6):495–522.

Tatsunori B. Hashimoto, Hugh Zhang, and Percy Liang. 2019. Unifying human and statistical evaluation for natural language generation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1689–1701, Minneapolis, Minnesota. Association for Computational Linguistics.

James J Heckman and Tim Kautz. 2012. Hard evidence on soft skills. Labour economics, 19(4):451–464.

Jennifer A Hoffmann, Megan M Attridge, Michael S Carroll, Norma-Jean E Simon, Andrew F Beck, and Elizabeth R Alpern. 2023. Association of youth suicides and county-level mental health professional

shortage areas in the us. JAMA pediatrics, 177(1):71– 80.

Mohammadreza Hojat, Salvatore Mangione, Thomas J Nasca, Mitchell JM Cohen, Joseph S Gonnella, James B Erdmann, Jon Veloski, and Mike Magee. 2001. The jefferson scale of physician empathy: development and preliminary psychometric data. Educational and psychological measurement, 61(2):349– 365.

Joey Hong, Sergey Levine, and Anca Dragan. 2023. Zero-shot goal-directed dialogue via rl on imagined conversations. ArXiv preprint, abs/2311.05584.

John J Horton. 2023. Large language models as simulated economic agents: What can we learn from homo silicus? Technical report, National Bureau of Economic Research.

Shang-Ling Hsu, Raj Sanjay Shah, Prathik Senthil, Zahra Ashktorab, Casey Dugan, Werner Geyer, and Diyi Yang. 2023. Helping the helper: Supporting peer counselors via ai-empowered practice and feedback. ArXiv preprint, abs/2305.08982.

Wenyue Hua, Lizhou Fan, Lingyao Li, Kai Mei, Jianchao Ji, Yingqiang Ge, Libby Hemphill, and Yongfeng Zhang. 2023. War and peace (waragent): Large language model-based multi-agent simulation of world wars. ArXiv preprint, abs/2311.17227.

Yueh-Min Huang, Tsung-Ho Liang, Yen-Ning Su, and Nian-Shing Chen. 2012. Empowering personalized learning with an interactive e-book learning system for elementary school students. Educational technology research and development, 60:703–722.

Robert J Jagers, Deborah Rivas-Drake, and Teresa Borowski. 2018. Equity & social and emotional learning: A cultural analysis. CASEL Assessment Work Group Brief series.

Hang Jiang, Xiajie Zhang, Robert Mahari, Daniel Kessler, Eric Ma, Tal August, Irene Li, Alex’Sandy’ Pentland, Yoon Kim, Jad Kabbara, et al. 2024. Leveraging large language models for learning complex legal concepts through storytelling. ArXiv preprint, abs/2402.17019.

Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. ArXiv preprint, abs/2305.06983.

Marzena Karpinska, Nader Akoury, and Mohit Iyyer. 2021. The perils of using Mechanical Turk to evaluate open-ended text generation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 1265–1285, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, and Richard Socher. 2019. Ctrl: A conditional transformer language model for controllable generation. ArXiv preprint, abs/1909.05858.

Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia. 2022. Demonstrate-searchpredict: Composing retrieval and language models for knowledge-intensive nlp. ArXiv preprint, abs/2212.14024.

J Peter Kincaid, Roger Hamilton, Ronald W Tarr, and Harshal Sangani. 2003. Simulation in education and training. Applied system simulation: methodologies and applications, pages 437–456.

Diane B Kraft. 2014. Creac in the real world. Clev. St. L. Rev., 63:567.

Kelly YL Ku and Irene T Ho. 2010. Metacognitive strategies that enhance critical thinking. Metacognition and learning, 5:251–267.

Deuksin Kwon, Sunwoo Lee, Ki Hyun Kim, Seojin Lee, Taeyoon Kim, and Eric Davis. 2023. What, when, and how to ground: Designing user personaaware conversational agents for engaging dialogue. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 5: Industry Track), pages 707–719.

Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Cheng Li, Mingyang Zhang, Qiaozhu Mei, Weize Kong, and Michael Bendersky. 2023. Automatic prompt rewriting for personalized text generation. ArXiv preprint, abs/2310.00152.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2021. Truthfulqa: Measuring how models mimic human falsehoods. ArXiv preprint, abs/2109.07958.

David B Lipsky, Ronald Leroy Seeber, and Richard D Fincher. 2003. Emerging systems for managing workplace conflict: Lessons from American corporations for managers and dispute resolution professionals, volume 18. Jossey-Bass San Francisco.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173.

Julia M Markel, Steven G Opferman, James A Landay, and Chris Piech. 2023. Gpteach: Interactive ta training with gpt-based students. In Proceedings of

the tenth acm conference on learning@ scale, pages 226–236.

David Matz and Noam Ebner. 2010. Using role-play in online negotiation teaching. In Venturing beyond the classroom, pages 293–317. DRI Press.

Brendan McCormack and Tanya V McCance. 2006. Development of a framework for person-centred nursing. Journal of advanced nursing, 56(5):472–479.

Norio Mishima, Shinya Kubota, and Shoji Nagata. 2000. The development of a questionnaire to assess the attitude of active listening. Journal of Occupational Health, 42(3):111–118.

Richard C Mohs and Nigel H Greig. 2017. Drug discovery and development: Role of basic biological research. Alzheimer’s & Dementia: Translational Research & Clinical Interventions, 3(4):651–657.

Britta K Morris-Rothschild and Marla R Brassard. 2006. Teachers’ conflict management styles: The role of attachment styles and classroom management efficacy. Journal of school psychology, 44(2):105–121.

TB Moyers, JK Manuel, D Ernst, T Moyers, J Manuel, D Ernst, and C Fortini. 2014. Motivational interviewing treatment integrity coding manual 4.1 (miti 4.1). Unpublished manual.

Michael J. Muller and Sarah Kuhn. 1993. Participatory design. Commun. ACM, 36(6):24–28.

Sheshera Mysore, Zhuoran Lu, Mengting Wan, Longqi Yang, Steve Menezes, Tina Baghaee, Emmanuel Barajas Gonzalez, Jennifer Neville, and Tara Safavi. 2023. Pearl: Personalizing large language model writing assistants with generation-calibrated retrievers. ArXiv preprint, abs/2311.09180.

Patricia B Nemec, Amy Cottone Spagnolo, and Anne Sullivan Soydan. 2017. Can you hear me now? teaching listening skills. Psychiatric rehabilitation journal, 40(4):415.

Galor Oded. 2011. Inequality, human capital formation, and the process of development. In Handbook of the Economics of Education, volume 4, pages 441–493. Elsevier.

Philip Oreopoulos and Kjell G Salvanes. 2011. Priceless: The nonpecuniary benefits of schooling. Journal of Economic perspectives, 25(1):159–184.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Sarah M Ovink and Brian D Veazey. 2011. More than “getting us through:” a case study in cultural capital enrichment of underrepresented minority undergraduates. Research in higher education, 52:370–394.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, pages 1–22.

Joon Sung Park, Lindsay Popowski, Carrie Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2022. Social simulacra: Creating populated prototypes for social computing systems. In Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology, pages 1–18.

Savvas Petridis, Ben Wedin, James Wexler, Aaron Donsbach, Mahima Pushkarna, Nitesh Goyal, Carrie J Cai, and Michael Terry. 2023. Constitutionmaker: Interactively critiquing large language models by converting feedback into principles. ArXiv preprint, abs/2310.15428.

Inioluwa Deborah Raji, Andrew Smart, Rebecca N White, Margaret Mitchell, Timnit Gebru, Ben Hutchinson, Jamila Smith-Loud, Daniel Theron, and Parker Barnes. 2020. Closing the ai accountability gap: Defining an end-to-end framework for internal algorithmic auditing. In Proceedings of the 2020 conference on fairness, accountability, and transparency, pages 33–44.

Rosemary P Ramsey and Ravipreet S Sohi. 1997. Listening to your customers: The impact of perceived salesperson listening behavior on relationship outcomes. Journal of the Academy of marketing Science, 25:127–137.

Josephine M Randel, Barbara A Morris, C Douglas Wetzel, and Betty V Whitehill. 1992. The effectiveness of games for educational purposes: A review of recent research. Simulation & gaming, 23(3):261–276.

Ruiyang Ren, Peng Qiu, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Hua Wu, Ji-Rong Wen, and Haifeng Wang. 2024. Bases: Large-scale web search user simulation with large language model based agents. ArXiv preprint, abs/2402.17505.

Carl Ransom Rogers and Richard Evans Farson. 1957. Active listening. Industrial Relations Center, the University of Chicago.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2024. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36.

Donald A Schon and Vincent DeSanctis. 1986. The reflective practitioner: How professionals think in action.

Douglas Schuler and Aki Namioka. 1993. Participatory design: Principles and practices. CRC Press.

Seth J Schwartz, Byron L Zamboanga, and Robert S Weisskirch. 2008. Broadening the study of the self: Integrating the study of personal identity and cultural identity. Social and personality psychology compass, 2(2):635–651.

Teresa E Seeman, Tina M Lusignolo, Marilyn Albert, and Lisa Berkman. 2001. Social relationships, social support, and patterns of cognitive aging in healthy, high-functioning older adults: Macarthur studies of successful aging. Health psychology, 20(4):243.

Omar Shaikh, Valentino Chai, Michele J Gelfand, Diyi Yang, and Michael S Bernstein. 2023a. Rehearsal: Simulating conflict to teach conflict resolution. ArXiv preprint, abs/2309.12309.

Omar Shaikh, Kristina Gligori´c, Ashna Khetan, Matthias Gerstgrasser, Diyi Yang, and Dan Jurafsky. 2023b. Grounding or guesswork? large language models are presumptive grounders. ArXiv preprint, abs/2311.09144.

Omar Shaikh, Hongxin Zhang, William Held, Michael Bernstein, and Diyi Yang. 2022. On second thought, let’s not think step by step! bias and toxicity in zeroshot reasoning. ArXiv preprint, abs/2212.08061.

Ashish Sharma, Inna W Lin, Adam S Miner, David C Atkins, and Tim Althoff. 2023. Human–ai collaboration enables more empathic conversations in textbased peer-to-peer mental health support. Nature Machine Intelligence, 5(1):46–57.

Ashish Sharma, Sudha Rao, Chris Brockett, Akanksha Malhotra, Nebojsa Jojic, and William B Dolan. 2024. Investigating agency of llms in human-ai collaboration tasks. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1968–1987.

Noah Shinn, Beck Labash, and Ashwin Gopinath. 2023. Reflexion: an autonomous agent with dynamic memory and self-reflection. ArXiv preprint, abs/2303.11366.

Ben Shneiderman. 1983. Direct manipulation: A step beyond programming languages. Computer, 16(08):57–69.

Kurt Shuster, Mojtaba Komeili, Leonard Adolphs, Stephen Roller, Arthur Szlam, and Jason Weston. 2022a. Language models that seek for knowledge: Modular search & generation for dialogue and prompt completion. ArXiv preprint, abs/2203.13224.

Kurt Shuster, Jing Xu, Mojtaba Komeili, Da Ju, Eric Michael Smith, Stephen Roller, Megan Ung, Moya Chen, Kushal Arora, Joshua Lane, et al. 2022b. Blenderbot 3: a deployed conversational agent that continually learns to responsibly engage. ArXiv preprint, abs/2208.03188.

Joseph William Singer. 1988. Persuasion. Mich. L. Rev., 87:2442.

Adam Smith. 2006. Cognitive empathy and emotional empathy in human behavior and evolution. The Psychological Record, 56(1):3–21.

Stephen D Sorden. 2012. The cognitive theory of multimedia learning. Handbook of educational theories, 1(2012):1–22.

Ira Strumwasser, Nitin V Paranjpe, Marianne Udow, David Share, Mary Wisgerhof, David L Ronis, Charlotte Bartzack, and Ali N Saad. 1991. Appropriateness of psychiatric and substance abuse hospitalization: implications for payment and utilization management. Medical Care, pages AS77–AS90.

Milind Tambe, W Lewis Johnson, Randolph M Jones, Frank Koss, John E Laird, Paul S Rosenbloom, and Karl Schwamb. 1995. Intelligent agents for interactive simulation environments. AI magazine, 16(1):15– 15.

Zhaoxuan Tan, Qingkai Zeng, Yijun Tian, Zheyuan Liu, Bing Yin, and Meng Jiang. 2024. Democratizing large language models via personalized parameter-efficient fine-tuning. ArXiv preprint, abs/2402.04401.

Donald M Taylor and Esther Usborne. 2010. When i know who “we” are, i can be “me”: The primary role of cultural identity clarity for psychological wellbeing. Transcultural psychiatry, 47(1):93–111.

Lucas Theis, Aäron van den Oord, and Matthias Bethge. 2016. A note on the evaluation of generative models. In 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings.

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. 2022. Lamda: Language models for dialog applications. ArXiv preprint, abs/2201.08239.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. ArXiv preprint, abs/2307.09288.

Cees PM van der Vleuten and David B Swanson. 1990. Assessment of clinical skills with standardized patients: state of the art. Teaching and Learning in Medicine: An International Journal, 2(2):58–76.

Evert Van der Vliert. 2013. Complex interpersonal conflict behaviour: Theoretical frontiers. Psychology Press.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An open-ended embodied agent with large language models. ArXiv preprint, abs/2305.16291.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2024a. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):1–26.

Rose Wang, Pawan Wirawarn, Omar Khattab, Noah Goodman, and Dorottya Demszky. 2024b. Backtracing: Retrieving the cause of the query. In Findings of the Association for Computational Linguistics: EACL 2024, pages 722–735, St. Julian’s, Malta. Association for Computational Linguistics.

Jason Weston and Sainbayar Sukhbaatar. 2023. System 2 attention (is something you might need too). ArXiv preprint, abs/2311.11829.

Greg Wiggan, Delphia Smith, and Marcia J WatsonVandiver. 2021. The national teacher shortage, urban education and the cognitive sociology of labor. The Urban Review, 53:43–75.

H Wilson, Paul Daugherty, and Nicola Bianzino. 2017. The jobs that artificial intelligence will create. MIT Sloan Management Review Summer.

Laura Wray-Lake and Amy K Syvertsen. 2011. The developmental roots of social responsibility in childhood and adolescence. New directions for child and adolescent development, 2011(134):11–25.

Tongshuang Wu, Ellen Jiang, Aaron Donsbach, Jeff Gray, Alejandra Molina, Michael Terry, and Carrie J Cai. 2022. Promptchainer: Chaining large language model prompts through visual programming. In CHI Conference on Human Factors in Computing Systems Extended Abstracts, pages 1–10.

Yuwei Wu, Xuezhe Ma, and Diyi Yang. 2021. Personalized response generation via generative split memory network. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1956–1970, Online. Association for Computational Linguistics.

Jing Xu, Arthur Szlam, and Jason Weston. 2021. Beyond goldfish memory: Long-term open-domain conversation. ArXiv preprint, abs/2107.07567.

Yuzhuang Xu, Shuo Wang, Peng Li, Fuwen Luo, Xiaolong Wang, Weidong Liu, and Yang Liu. 2023. Exploring large language models for communication games: An empirical study on werewolf. ArXiv preprint, abs/2309.04658.

Kevin Yang, Yuandong Tian, Nanyun Peng, and Dan Klein. 2022. Re3: Generating longer stories with recursive reprompting and revision. ArXiv preprint, abs/2210.06774.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2024. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. ArXiv preprint, abs/2210.03629.

Juping Yu and Maggie Kirk. 2009. Evaluation of empathy measurement tools in nursing: systematic review. Journal of advanced nursing, 65(9):1790–1806.

Jintian Zhang, Xin Xu, and Shumin Deng. 2023. Exploring collaboration mechanisms for llm agents: A social psychology view. ArXiv preprint, abs/2310.02124.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2024. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36.

Xuhui Zhou, Zhe Su, Tiwalayo Eisape, Hyunwoo Kim, and Maarten Sap. 2024a. Is this the real life? is this just fantasy? the misleading success of simulating social interactions with llms. ArXiv preprint, abs/2403.05020.

Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, et al. 2023. Sotopia: Interactive evaluation for social intelligence in language agents. ArXiv preprint, abs/2310.11667.

Zhilun Zhou, Yuming Lin, Depeng Jin, and Yong Li. 2024b. Large language model for participatory urban planning. ArXiv preprint, abs/2402.17161.

