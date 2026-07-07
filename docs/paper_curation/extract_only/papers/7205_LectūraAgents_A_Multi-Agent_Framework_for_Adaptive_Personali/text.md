Lect¯uraAgents: A Multi-Agent Framework for Adaptive
Personalized AI-Assisted Learning and Embodied Teaching
Jaward Sesay
Beijing Institute of Technology
Yue Yu†
Beijing Institute of Technology
Siwei Dong*
Peking University
Guangyao Chen*
Cornell University
Börje F. Karlsson*
Beijing Academy of Artificial
Intelligence
Figure 1: Overview of Lect¯uraAgents: a hierarchical multi-agent framework for end-to-end adaptive personalized learning experiences. Given a
lecture prompt or learning materials and a learner profile, a ProfessorAgent leads a collaborative team of specialized agents through research,
planning, design, evaluation and embodied delivery of lecture and study contents that adapt to the individual learner. The framework provides
students with access to real-time adaptive, personalized teaching and study sessions.
Abstract
Effective personalized AI-assisted learning demands learning sys-
tems that can not only generate accurate learner-specific educa-
tional materials, but also dynamically adapt their instruction to
diverse learners. However, existing educational agent frameworks
have primarily focused on lecture content automation and simula-
tions, which often fall short of modelling multimodal and embodied
instructional methods tailored for the individual learner. To this
end, we propose Lect¯uraAgents—a multi-agent framework that en-
ables personalized learning through end-to-end adaptive embodied
teaching. At its core, Lect¯uraAgents mirrors a professor-student
relationship, in which the ProfessorAgent leads a collaborative team
of specialized subordinate agents through research, planning, re-
view, and embodied delivery of lecture contents that adapt to a
learner’s needs. The framework offers three main contributions: (1)
† Corresponding author: yuyueanny@hotmail.com
* Equal contributing authors
The dataset for this work is available at: https://huggingface.co/datasets/Jaward/
lectura-agents-data.
a hierarchical multi-agent architecture for end-to-end personalized
learning; (2) an adaptive embodied teaching mechanism, wherein
the ProfessorAgent executes visible and pedagogically motivated
teaching actions (e.g., handwrite, highlight, underline, etc.) over con-
tents in a teaching environment while speaking; and (3) a Teaching
Action-Speech Alignment (TASA) algorithm that employs salience-
based heuristics and temporal semantic segmentation to generate
coherent teaching action sequences aligned with learner profiles.
We evaluate Lect¯uraAgents on diverse courses at high school, under-
graduate, and graduate levels using sample-specific rubric-based
analysis; with generated lecture materials and teaching actions
assessed and validated by expert educators. Experimental results
show consistent gains in lecture content quality, embodied teaching
quality, assessment, and personalization over existing approaches,
positioning Lect¯uraAgents as a pedagogically well-grounded frame-
work for personalized learning at scale.
1
arXiv:2606.16428v2  [cs.CL]  23 Jun 2026

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
1
Introduction
Adaptive personalized AI-assisted learning has emerged as a promis-
ing approach for tailoring instructions to individual learners, with
studies reporting gains in motivation, engagement, and learning
outcomes, especially in online educational settings [1–3]. However,
contemporary personalized learning solutions and frameworks typ-
ically focus on adapting what is recommended, rather than how
instructional content is delivered to the learner [4]. Research on
embodied teaching shows that performing teaching actions (e.g.,
writing, pointing or gesturing) during a lecture can help guide atten-
tion, foster conceptual understanding, and enhance overall learning
outcomes [5–7]. These findings point to the need for personalized
learning solutions that well integrate adaptive learning contents
with embodied instructional delivery.
Recent frontier models demonstrate strong reasoning and agen-
tic capabilities that have enabled planning, tool-use, and multi-step
problem solving, opening new possibilities for applications in per-
sonalized learning [8–11]. This breakthrough has led to the explo-
ration of LLM-powered agent frameworks for education, where
specialized agents automate learning and teaching tasks to support
students and educators [12]. Moreover, recent efforts have further
demonstrated the potential of leveraging multiple agents to act as
personal tutors and learning companions that provide on-demand
teaching and learning support based on individual needs [13–16].
However, the predominant focus of most related frameworks has
been on simulations, where agents enact roles in virtual classrooms
[17–19] simulate teacher–student dialogues to evaluate teaching
behaviors and feedback strategies [20, 21], or coordinate agent
workflows for generating personalized learning materials [22, 23].
These are important proof of concepts, but their impact is limited to
controlled virtual environments that do not capture the myriad nu-
ances of adaptive personalized learning in real life scenarios. Other
works have explored single-agent or prompt-engineered LLM tu-
toring systems [24–26] that generate explanations, feedback, or
instructional contents, but without rigorous review or modeling of
how such contents should be contextualized and adapted to diverse
learning profiles. Few related works extend beyond these scopes to
adopt a broader personalized instructional perspective that is often
centered on automating course content generation [27, 28], which
is primarily delivered in text-only modality, with no account for
personalized embodied instructional delivery. Collectively, these
systems offer valuable contributions to AI-assisted learning but
remain fragmented in scope, lacking a unified model that connects
personalized content generation with adaptive embodied delivery.
Consequently, key pedagogical features, including coordinated les-
son planning, iterative content review, embodied teaching, and
alignment between teaching behaviour and learner needs, remain
insufficiently addressed.
To address these limitations, we propose Lect¯uraAgents, a hierar-
chical multi-agent framework for end-to-end personalized lecture
generation and embodied lecture delivery. Our framework moves
beyond simulations and static content generation, to managing the
entire life cycle of a lecture (i.e., from preparation to delivery, as
shown in Figure 1), while adapting to individual learning prefer-
ences. Lect¯uraAgents offers three primary contributions:
1. A hierarchical multi-agent architecture for end-to-
end personalized learning: we propose the first multi-
agent framework with end-to-end personalization for learn-
ing. It mirrors a professor–student relationship, where a
ProfessorAgent coordinates specialized assistant agents (at
different hierarchies) to plan, research, review, and create
lecture contents tailored for the individual learner.
2. A Teaching Action-Speech Alignment (TASA) algo-
rithm: a novel technique that uses LLM-based semantic
analysis, temporal content segmentation, and salient heuris-
tics to accurately align relevant teaching actions to regions
or contents in a teaching environment (e.g., over a slide).
3. An embodied lecture delivery mechanism: our frame-
work enables a ProfessorAgent to perform visible, inter-
pretable teaching actions (e.g., highlight, handwrite, under-
line, etc.) directly over contents in the teaching environ-
ment (in our case, lecture slides) with a clear pedagogical
rationale for each action taken.
Lect¯uraAgents decomposes personalized instruction into agents
operating at three hierarchies across two sessions: Lecture Prepa-
ration and Lecture Delivery. In the preparation session, the Pro-
fessorAgent leads a team of validator and executor agents through
planning, research, generation, and evaluation of lecture artifacts.
During the delivery or teaching session, the ProfessorAgent utilizes
these artifacts to enact an embodied teaching role, executing visible
and pedagogically motivated teaching actions on contents in the
learning environment.
We conducted extensive evaluations of the framework on di-
verse courses at high school, undergraduate, and graduate levels,
assessing lecture quality, teaching quality and personalization. Our
experiments show that Lect¯uraAgents can produce high quality
lecture artifacts, while effectively adapting personalized teaching
strategies to diverse learner profiles through coherent embodied
teaching action sequences.
2
Related Work
2.1
Adaptive Personalized AI-Assisted Learning
The idea of personalized learning predates LLMs and LLM agents.
Early theories of memory, such as Atkinson and Shiffrin’s model of
how information is encoded and rehearsed [39] and Cowan’s ac-
count of short-term and long-term memory capacities [40], helped
establish the cognitive foundations for adapting instruction to the
ways learners process and retain information. These insights in-
spired models of personalized learning that emphasized learner-
centered pathways, individualized pacing and tailored support. Be-
fore LLMs became widely adopted, deep learning models were used
in intelligent tutoring systems (ITS) to monitor learners’ perfor-
mances, adjust task difficulty, and deliver personalized feedback
[41–43]. Reviews show that AI-assisted personalized learning has
a positive impact on students’ engagement and learning outcome
across diverse learning settings [44–46]. More recent empirical
studies of AI-driven adaptive platforms in university and language-
learning contexts, report gains across performance, satisfaction,
and self-directed learning [47–49]. Collectively, these findings make
clear the significance of adaptive personalized learning, forming
the foundations upon which our framework is built.
2

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Table 1: Comparison of Lect¯uraAgents with existing multi-agent frameworks in this domain
Framework
Teaching Modality
Embodied
Agent(s)
Teaching Action
Alignment
Personalization
Multi-Agent
Collaboration
EduAgent [29]
Text
✗
✗
✗
✗
Agent4Edu [30]
Text (simulation)
✗
✗
✓
✓
EducationQ [31]
Text (simulation)
✗
✗
✗
✗
FACET [32]
Text
✗
✗
✓
✗
KELE [33]
Text
✗
✗
✗
✗
Instructional Agents [34]
Text
✗
✗
✗
✓
EduPlanner [35]
Text
✗
✗
✓
✓
GenMentor [36]
Text
✗
✗
✓
✓
SimClass [37]
Text (simulation)
✗
✗
✓
✓
WikiHowAgent [38]
Text
✗
✗
✗
✓
Lect¯uraAgents
Multimodal (Embodied)
✓
✓
✓
✓
2.2
LLM Agent Frameworks for Education
Early works on LLM agents demonstrated how language models can
plan, use tools, decompose tasks, and coordinate multi-step reason-
ing across multiple collaborating agents [50–53]. These capabilities
soon inspired educational multi-agent frameworks. For instance,
EduAgent [29] models diverse student personas using cognitive-
science priors, Agent4Edu [30] simulates learner responses with
memory-based generative agents, and EducationQ [31] stages multi-
agent teacher-student-evaluator interactions to assess teaching be-
haviors. Similarly, systems like SimClass [37] and WikiHowAgent
[38] extend simulation to classroom dynamics and procedural learn-
ing. Course-content automation then became a focus, with Instruc-
tional Agents [34] generating full course materials through role-
based collaboration, and EduPlanner [35] iteratively refining les-
son plans via evaluator–optimizer agent loops. More recent works
have also introduced personalization: FACET [32] creates learner-
adapted worksheets, KELE [33] provides concept-level enrichment
and feedback, and GenMentor [36] builds personalized learning
paths from learner goals. While these contributions demonstrate
how multi-agent systems can enhance learning, they lack relevant
capabilities (as summarized in Table 1) that integrates personalized
content generation with embodied instructional delivery.
2.3
Embodied Teaching Agents
Embodied teaching in digital settings refers to instructional meth-
ods that combine verbal instruction with spatial teaching actions
(e.g., writing, highlighting, underlining, or pointing) over learning
contents in a virtual teaching environment. These actions help
guide attention, reduce cognitive load, and support concept forma-
tion [54, 55]. Earlier models like AutoTutor and its variations [56, 57]
demonstrated the benefit of animated pedagogical agents capable of
conversational scaffolding. Recent systems have explored program-
matic video-based approaches, for example, Xu et al. [58] explored
how AI-generated lecture videos compare with real lectures, while
AutoLectures [59] converts slides into narrated videos with high-
light actions (using Levenshtein and LLM-based matching), and
PASS [60] automated slide and speech generation from word doc-
uments. These efforts emphasize the importance of action-based
instructional cues, but fall short of delivering a coherent end-to-
end personalized, adaptive, and pedagogically informed embodied
instruction.
3
Lect¯uraAgents
We designed Lect¯uraAgents to be both domain-specific and exten-
sible given the nature of the problem we are trying to solve. Its
framework integrates planning, research, and pedagogical embodi-
ment within a cohesive, end-to-end hierarchical architecture that
supports both personalization and continual learning in tandem.
Lect¯uraAgents consists of four interconnected modules:
• LLM – The LLM module provides agents with access to
frontier models (e.g., GPT-5, Gemini 3 pro, Claude Sonnet
4, Deepseek V3.2, Qwen 3, and Kokoro TTS [61]) through
their respective custom APIs. It serves as the brain behind
agents, handling text, image, and speech modalities.
• Agent – This module holds the core logic for each agent’s
role, capabilities, and tools. It also enables coordinated
multi-agent collaboration through dynamically invoked ac-
tions for assigned tasks. The framework adopts a three-tier
hierarchical collaborative mechanism with a lead coordina-
tor agent managing a validator agent, who in turn manages
executor agents for lecture content generation. To complete
tasks, agents execute a series of actions utilizing custom
tools.
• TASA – The Teaching Action-Speech Alignment (TASA)
module handles alignment between embodied teaching ac-
tions and their corresponding lecture speeches. It provides
logic for salient heuristic analysis and temporal semantic
segmentation, which help provide context when curating
relevant teaching action sequences.
• Memory – This module implements short-term, long-term,
and dynamic memories, that allow agents to preserve con-
text, track learner needs, and adapt their behaviour over
time.
3

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Figure 2: Lect¯uraAgents Architecture. The framework adopts a hierarchical multi-agent architecture, modeled after a professor–students’
relationship. One in which a coordinator agent (or ProfessorAgent) guides a collaborative team of validator and executor agents through
planning, research, design, and delivery of personalized lecture contents. Multi-agent collaboration is mediated through an orchestration
layer with group-chat communication that enables iterative planning, self-evaluation, and continuous refinement of generated materials. This
architecture is supported by four interconnected modules: Agents, LLM, TASA, and Adaptive Memory.
These four modules span across the framework’s two main stages:
Lecture Preparation Session and Lecture Delivery Session. Moreover,
as shown in Figure 3, the delivery session supports two modes:
Teach Mode, which generates a new personalized lecture based on
the learner’s profile and provided learning materials, and Study
Mode, which allows learners to upload existing materials, such
as notes, books or projects, and interact with the ProfessorAgent
through real-time Q&A.
3.1
Lecture Preparation Session
In this stage, the ProfessorAgent leads a collaborative team of spe-
cialized agents through planning, research, alignment, review, and
creation of personalized lecture artifacts (e.g., lecture plan, slides,
scripts, speech, teaching actions, notes, etc.). A quick overview of
the entire process can be found in Algorithm 1.
Lecture Prompts and Configs. Lecture preparation begins by
processing the learner’s prompts along with a range of configu-
ration choices that define the scope, style, and preferences of the
lecture. The prompt captures the lecture topic, its intended cover-
age, and the learner profile, so the framework can adapt content
depth and learning preferences, while optional syllabus or refer-
ence materials help anchor the lecture to a course context or source
material. Additional settings specify the instructor persona, target
academic level (high school, undergraduate, masters, or PhD), lan-
guage of instruction (which currently includes English, Chinese,
French, or Spanish), and the approximate number of slides to be
generated. Learners can also choose their preferred voice model,
handwriting mode (either Handwriting RNN Model or Preset Font
Handwriting), LLM model, and research method (using Wikipedia
or Google search). Together, these inputs provide the initial condi-
tions that guide downstream multi-agent collaboration, planning,
research, content generation, and embodied teaching. Our teach-
ing and learning environment can be accessed via a browser (as
shown in Figure 3) for easier entry of all inputs. Additional details
on lecture prompts and configurations can be found in Appendix B.
Multi-agent Collaboration. When a lecture is prompted, the
ProfessorAgent first initiates the preparation session, creating a col-
laborative group chat named Swarm-of-Ranks Group Chat (shown
in Figure 4) – where agents at different ranks collaborate to com-
plete assigned tasks. In this group chat we have a coordinator (Pro-
fessorAgent), a validator (LecturePlanner), and different executor
(ResearchAgent, SlideAgent, ScriptAgent, SpeechAgent, and Teaching
Action-Speech Alignment agent or TasaAgent) agents. The coor-
dinator agent (Rank 1) supervises the validator agent (Rank 2),
who in turn manages executor agents at Rank 3. This hierarchical
structure allows for efficient review and successful completion of
4

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Figure 3: An example of adaptive personalized learning experience with Lect¯uraAgents
assigned tasks. Agents communicate by sending messages in the
group chat through a communication layer. There are nine message
types: [Task], [TaskAcknowledged], [Progress], [TaskCompleted],
[Approval], [Revisal], [Handoff], [RevisalSucceeded], and [Revisal-
Failed]. Table 2 shows the message-types respective agents can
send in the group chat.
Planning. The lecture preparation process starts with planning,
wherein the ProfessorAgent instructs the LecturePlanner to draft a
lecture plan based on the requested lecture topic and learner profile.
The LecturePlanner first conducts preliminary research on the topic,
then writes a detailed plan, and submits it for review and approval
by the ProfessorAgent.
Table 2: Message types for respective agents at different ranks
Rank(s)
Message Types
Agents
1, 2
[Task], [Approval],
[Revisal]
ProfessorAgent and
LecturePlanner
1, 2, 3
[TaskAcknowledged],
[Progress],
[TaskCompleted]
All agents
3
[Handoff],
[RevisalSucceeded],
[RevisalFailed]
ResearchAgent, SlideAgent,
ScriptAgent, SpeechAgent,
TasaAgent
5

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
The plan contains lecture metadata, learner profile, and detailed
descriptions of tasks for each executor agent with respective crite-
ria for completing assigned tasks. Once the plan is approved, the
LecturePlanner then instructs and coordinates executor agents to
generate lecture contents based on the plan. Subsequent prepara-
tion stages will involve sequential execution of tasks by executor
agents and iterative validation by the LecturePlanner.
Generation. This stage starts with the SlideAgent, which is
tasked with designing each slide (in HTML format), using a cus-
tom slide builder tool, and generating respective contents based on
structural and pedagogical criteria from the lecture plan. Each slide
is designed to support contents in text, image, video, and speech
modalities, via structured content blocks. Slide images can be either
generated or sourced online via web search. Next, the ScriptAgent
utilizes the generated slide contents (along with lecture plan and
research findings) to create a personalized and pedagogically in-
formed script for each created slide. Scripts are conditioned to
capture the learner’s attention, level of understanding, and learning
preferences. Finally, the scripts are then passed on to the SpeechA-
gent which performs speech synthesis, converts scripts to speech
(in the learner’s desired instructor voice), and creates word-level
timestamps for each speech action using Whisper ASR [62]. These
artifacts will later be used during alignment and review.
Alignment. Given the generated speech timestamps, scripts,
slide contents, and learner profile, the TasaAgent first performs
a preliminary teaching action analysis using segmentation and
salient heuristic tools in the TASA module. This analysis starts
with the temporal semantic segmentation of slide contents and
scripts to identify segments that should receive teaching actions; it
then applies salience-based heuristics to provide rationale for each
teaching action application.
Currently, Lect¯uraAgents supports two kinds of teaching actions:
Rough Notation (RN), e.g., highlight, underline, circle, box, etc., and
Handwriting actions (HW), i.e., writing down key points in natural
human-like handwriting style, while speaking. This analysis results
are then added to the agent’s context when mapping pedagogical
teaching actions to contents in the slide teaching environment.
The ProfessorAgent will later utilize the resulting teaching action
sequences during embodied teaching in the lecture delivery session.
Self-reflection. In addition to the hierarchical review mecha-
nism present in multi-agent collaboration, we ensure each agent
self-reflects on any completed tasks to find and fix issues before
submitting results for review by the validator agent. They do this by
first reviewing completed tasks, then self-validating them against
required criteria detailed in the lecture plan.
Personalization. We ensure personalization across all gener-
ated lecture contents—slides, images, quizzes, lecture notes, scripts
and teaching actions—by conditioning generation on the learner’s
profile, learning preferences, and usage history in memory. For
example, slide contents, as shown in Figure 5, are adapted to the
learner’s interests by framing concepts around a favourite sport
or hobby, or can be tailored into an easier-to-follow learning path
(e.g., more scaffolding or simpler analogies) when the student pro-
file indicates lower prior knowledge. Slide images are generated
to match the same themes and difficulty level, while quizzes are
personalized in both content and phrasing to assess understanding
using familiar scenarios. The resulting notes, scripts, and teaching
Algorithm 1 Lecture Preparation Session
Input:
Lecture Prompt 𝐿𝑃, Learner Profile 𝑈
Parameters:
Coordinator/validator agents {𝐴𝑃,𝐴𝐿𝑃}; executor
agents 𝐸𝐴= {𝐴𝑅,𝐴𝑆,𝐴𝑆𝑐,𝐴𝑆𝑝,𝐴𝑇};
preparation plan 𝑃= {𝑃1, 𝑃2, . . . , 𝑃𝑛}; adaptive
memory 𝑀𝐴= {𝑀𝑠, 𝑀𝐿, 𝑀𝑑};
Swarm-of-Ranks group chat 𝐺chat
Output:
Lecture artifacts 𝐿𝐴= {Plan, Slides, Script, Speech,
TeachingActions, Notes, Assessments}
1: Initialize memory {𝑀𝑠, 𝑀𝐿, 𝑀𝑑} and agents {𝐴𝑃,𝐴𝐿𝑃}
2: 𝐴𝑃starts prep session and instantiates 𝐺chat
3: 𝐴𝑃debriefs 𝐴𝐿𝑃and requests lecture plan 𝑃
4: repeat
5:
𝐴𝐿𝑃drafts 𝑃from (𝐿𝑃,𝑈) →𝑀𝑑
6:
𝐴𝑃reviews 𝑃and gives feedback →𝑀𝑠
7:
𝐴𝐿𝑃updates 𝑃based on feedback →𝑀𝑑
8: until 𝐴𝑃approves 𝑃or max iterations reached
9: Initialize executor agents 𝐸𝐴
10: for each 𝑃𝑖in 𝑃do
11:
𝐴𝐿𝑃debriefs 𝐸𝐴on assigned tasks
12:
for each executor 𝐸𝑖in 𝐸𝐴do
13:
repeat
14:
𝐸𝑖plans and executes task
15:
𝐸𝑖self-reflects and submits
16:
𝐴𝐿𝑃reviews task and gives feedback
17:
until 𝐴𝐿𝑃approves task or max iterations reached
18:
end for
19: end for
20: 𝐴𝐿𝑃submits artifacts 𝐿𝐴for final review by 𝐴𝑃
21: 𝐴𝑃reviews and validates 𝐿𝐴
22: return 𝐿𝐴
Figure 4: Swarm-of-Ranks group chat
actions mirror these choices to ensure a coherent, learner-specific
narrative throughout the lecture.
6

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Figure 5: Screenshot of a personalized slide for an undergraduate
student whose favorite sport is tennis, with key concepts explained
using tennis-themed visuals and embodied teaching actions over
slide contents.
Review. Finally in this session, generated lecture artifacts are as-
sembled by the LecturePlanner and submitted to the ProfessorAgent
for final review. During review the ProfessorAgent again validates
lecture artifacts based on lecture content quality, teaching quality,
action alignment, and personalization. Once review is successful,
the ProfessorAgent agent then takes on the role of teacher in the
subsequent lecture delivery session.
3.2
Lecture Delivery Session
During this stage, the ProfessorAgent assumes the role of an em-
bodied instructor that executes pedagogical teaching actions in the
slide environment using lecture artifacts from the lecture prepa-
ration session. In this work, we define a Teaching Action as a
semantically bounded, visually interpretable and pedagogically mo-
tivated operation performed by the ProfessorAgent over contents in
the teaching environment, while speaking. Each action comes with
a rationale for why it was taken at a particular time. We experiment
with two types of teaching actions:
1. Rough Notations (RN): These are actions that involve
marking or emphasizing existing contents on the slide. Ex-
amples include highlighting key terms, underlining impor-
tant phrases, circling diagrams, or boxing critical points.
RN actions are used to draw the learner’s attention to spe-
cific areas of the slide that are relevant to the current topic
being discussed. For improved user experience, we make
use of a hand-drawn annotation library [63] that simulates
human-like rough notations for these actions.
2. Handwriting (HW): These actions involve writing new in-
formation directly onto the slide canvas in a natural, human-
like handwriting style. This can include jotting down defi-
nitions, drawing diagrams, or annotating existing content.
HW actions serve to reinforce learning by actively engag-
ing the learner with newly introduced concepts during the
lecture. We utilize both a handwriting recurrent neural net-
work model based on Graves [64] and a preset font-based
handwriting synthesis for this teaching action.
These actions undergo preliminary review, analysis, and alignment
using our proposed Teaching Action-Speech Alignment (TASA)
algorithm, summarized in Algorithm 2.
Teaching Action–Speech Alignment (TASA) Algorithm.
TASA uses a combination of LLM-based salience heuristics
analysis and temporal semantic segmentation to help guide
the TasaAgent with prospective relevant teaching action se-
quences. The agent’s objective is to emit an ordered list of ped-
agogically informed teaching action-speech sequences 𝐴𝑆seq =
{𝑆1[𝑎1,𝑎2, . . . ,𝑎𝑛], . . . ,𝑆𝑛[𝑎1,𝑎2, . . . ,𝑎𝑛]}, for each slide 𝑆𝑛, where
each action 𝑎𝑛is given by:
𝑎𝑛= {actiontype𝑛, start𝑛, end𝑛, cfg𝑛}
(1)
𝑎𝑐𝑡𝑖𝑜𝑛𝑡𝑦𝑝𝑒𝑛can be either RN or HW, (𝑠𝑡𝑎𝑟𝑡𝑛,𝑒𝑛𝑑𝑛) gives the dura-
tion for the action, and 𝑐𝑓𝑔𝑛holds additional metadata or configu-
ration specific to the action type, as illustrated in Figure 6.
Figure 6: Data structure for Rough Notation and Handwriting teach-
ing actions in json.
Temporal Semantic Segmentation. Before performing
salience heuristics analysis, we first segment slide contents and
speech semantically (see Figure 7), in order to augment our agent’s
context for better teaching action sequences. Segment labels include
Pedagogical, Personalized, Salient, Adaptive, and Assessment, each
of which helps provide insight into the kind of teaching actions
to apply. For each slide region 𝑟𝑒𝑔𝑖𝑜𝑛𝑛∈𝑅𝑠and corresponding
speech segment with label 𝑙𝑎𝑏𝑒𝑙𝑛, the TasaAgent creates a segment
𝑠𝑒𝑔𝑚𝑒𝑛𝑡𝑛given by:
segment𝑛= {label𝑛, region𝑛, speech_segment𝑛}
(2)
specifically, for each candidate segment 𝑠𝑒𝑔𝑚𝑒𝑛𝑡𝑛in a slide 𝑆𝑛,
TASA analyses the segment data and assigns a suitable teaching
action along with a rationale 𝑟𝑛in natural language, explaining
why this action is appropriate for that specific region. The final
heuristics analysis data for a given slide is recorded as:
H (𝑆𝑛) = {segment𝑛,𝑎𝑛,𝑟𝑛}
(3)
which provides the TasaAgent with a structured context when
generating the resulting teaching action-speech sequences ASseq.
Embodied Teaching. Given the generated teaching action-
speech sequences, the ProfessorAgent dynamically schedules and in-
vokes respective teaching action functions over regions in the slide
environment (in sequence), while speaking. Each action function is
tied to a specific speech segment (with word-level timestamps) and
7

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Figure 7: The left slide shows already taken RN and HW teaching actions on a slide, while the right slide shows temporal semantic segmentation
of slide contents with segment labels, action types, rationales, regions and their respective speech and script segments.
applies a targeted visual operation such as handwriting, highlight-
ing, circling, or underlining, directly on the corresponding slide
region, as illustrated in Figure 8.
To ensure accurate and realistic embodiment, the agent is pro-
vided with a discrete world view of the slide environment and its
contents, while using a 3D quill-holding hand to execute the embod-
ied teaching actions with precise spatial targeting of regions and
their corresponding action types. As a result, embodied teaching
actions like handwriting, highlighting, circling, etc., are executed in
a natural, interpretable, and pedagogically grounded manner that
closely mirrors human instructional behaviour.
Figure 8: Illustration of Embodied Teaching in Lect¯uraAgents
4
Experiments
We conducted extensive quantitative and qualitative evaluations of
Lect¯uraAgents through diverse experiments, assessing the frame-
work’s performance on the following pedagogical metrics: lecture
content quality, teaching quality, assessment, and personalization.
Our main goal is to provide answers to two fundamental research
questions:
Algorithm 2 Teaching Action-Speech Alignment (TASA) Algo-
rithm
Input:
Slides 𝑆= {𝑆1,𝑆2, . . . ,𝑆𝑛}, scripts
𝑆𝑐= {𝑆𝑐1,𝑆𝑐2, . . . ,𝑆𝑐𝑛}, word-level speech
timestamps 𝑇𝑑, learner profile 𝑈
Parameters:
TasaAgent, regions 𝑅, labels {Pedagogical,
Personalized, Salient, Adaptive, Assessment};
RN and HW action types, salience data H,
dynamic memory 𝑀𝑑
Output:
𝐴𝑆seq
1: Initialize 𝑆, 𝑆𝑐, 𝑇𝑑, and 𝑈
2: for each slide 𝑆𝑛in 𝑆do
3:
Parse slide contents and identify regions 𝑅in 𝑆𝑛
4:
Analyze script for current slide 𝑆𝑛
5:
for each region 𝑅𝑛in 𝑅do
6:
𝐿𝑛←assign segment label 𝐿to region 𝑅𝑛
7:
𝑆𝑐𝑛←add appropriate speech segment
8:
𝑠𝑒𝑔𝑚𝑒𝑛𝑡𝑛←write segment data to 𝑀𝑑
9:
end for
10:
for each 𝑠𝑒𝑔𝑚𝑒𝑛𝑡𝑛in slide 𝑆𝑛do
11:
Analyze 𝑠𝑒𝑔𝑚𝑒𝑛𝑡𝑛
12:
𝑎𝑛←assign suitable action (RN or HW)
13:
𝑟𝑛←give rationale for action
14:
H ←write salience heuristic data to 𝑀𝑑
15:
end for
16:
T ←save segmentation and analysis data to 𝑀𝐿
17: end for
18: TasaAgent utilizes T to generate 𝐴𝑆seq
19: return 𝐴𝑆seq
1. RQ1: How does leveraging an adaptive hierarchical multi-
agent architecture create high-quality personalized lecture
contents that align with diverse learning profiles?
2. RQ2: How can an embodied tutor agent utilize generated
materials to execute coherent, visual, and pedagogically
8

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
informed teaching actions in a teaching environment (e.g.,
lecture slides presentation)?
4.1
Experiment Setup
The experiments were designed to assess the framework from end-
to-end, evaluating both personalized lecture generation and em-
bodied teaching capabilities. We start by performing pedagogical
evaluation on 280 personalized lectures generated using the frame-
work under the seven frontier models reported in Table 4. For each
model, we generate 40 lectures, with 10 lectures for each academic
level, using the same prompts, learner profiles, and text-to-speech
model (Kokoro TTS [61]) to ensure a fair comparison. Details on
these lectures can be found in Appendix A.2.2. We worked with
five expert educators, including subject teachers and university
instructors with experience in curriculum design and instructional
assessment, to define pedagogical rubrics grounded in recognized
instructional quality standards [65], as summarized in Table 3. Ad-
ditional details on the recruitment of these experts can be found in
Appendix A.2.4. We then adopted the evaluation method in Tutor-
Bench [66], with scoring primarily done by the expert educators
in order to avoid induced bias from an LLM judge. Thus, for a j-th
lecture, the framework’s overall performance score for each session
under a given model or baseline framework, is computed as the
weighted average of all passed rubric criteria 𝐴𝐴𝑅𝑗
𝑤, given by:
𝐴𝐴𝑅𝑗
𝑤=
Í𝑁𝑗
𝑖=1 𝑤𝑗
𝑖· 1𝑟𝑗
𝑖
Í𝑁𝑗
𝑖=1 𝑤𝑗
𝑖· 1𝑤𝑗
𝑖>0
(4)
where 𝑁𝑗is the number of rubric criteria for the j-th lecture, 𝑤𝑗
𝑖∈
{-5, -3, -1, 0, +1, +3, +5}, is the weight assigned to the i-th criterion,
and 𝑟𝑗
𝑖∈{0,1} indicates whether criterion i is satisfied. When a
criterion is satisfied 𝑟𝑗
𝑖= 1, it contributes a positive reward of +5,
+3, or +1, corresponding to it being a highly desirable, desirable
and important, or nice-to-have behaviour, respectively. When a
criterion is not satisfied 𝑟𝑗
𝑖= 0, it is explicitly treated as a failure
state and contributes a non-positive score, spanning a 0, -1, -3, and
-5 range: 0 denotes the lowest-severity failure (no credit), -1 is used
for a minor failure, -3 for a moderate failure, and -5 represents a
critical failure (highly undesirable behaviour).
4.1.1
Lecture Generation Evaluation
Here, we evaluate Lect¯uraAgents as a personalized lecture con-
tent generator. For each model, we generated 40 personalized lec-
tures covering maths, science, engineering, art, and history, with
10 lectures each for high school, undergraduate, master’s, and PhD
learning profiles. Topics were randomly selected with emphasis on
science subjects. Each lecture targeted one individual learner profile,
covering learners aged 13–35, with profiles varying by academic
level, prior knowledge, learning goals, learning style, and expected
difficulty. The resulting output after generation contains the fol-
lowing lecture artifacts: a detailed lecture plan, a research report,
syllabus, learner profile, 15 slides with images, per-slide scripts, lec-
ture speeches, personalized lecture notes and study guide, teaching
actions, teaching action–speech alignment, and assessments.
Table 3: Evaluation metrics and their respective rubrics
Lecture Generation
Evaluation Metric
Rubrics
Lecture Content
Quality (LCQ)
Accuracy, Clarity, Coherence, Cognitive
Load, Syllabus Coverage,
Instruction-following
Personalization
Quality (PQ)
Adaptive Emphasis, Preference Alignment,
Engagement, Motivation, Tone/Style
Assessment Quality
(AQ)
Concept Coverage, Cognitive
Appropriateness, Answer Validity;
Rationale
Lecture Delivery
Teaching Action
Quality (TAQ)
Temporal Alignment, Accurate
Handwriting Action, Accurate Rough
Notation Action, Spatial Accuracy, Active
Learning, Embodied Teaching
Evaluation Metrics. Using expert-defined rubrics detailed in Ta-
ble 3, we assess the framework’s personalized lecture content gen-
eration capability across three main evaluation metrics: Lecture
Content Quality (LCQ), Personalization Quality (PQ), and Assess-
ment Quality (AQ). LCQ measures accuracy, clarity, coherence,
cognitive load, and instruction-following rubric dimensions. PQ
evaluates adaptation to a learner profile (adaptive emphasis) and
learning preferences (preference alignment), engagement, motiva-
tion, and instructor’s tone or style. AQ measures concept coverage,
cognitive appropriateness, answer accuracies, and rationale quality.
Each lecture’s metric score is computed using the weighted aver-
age of all passed rubrics and then averaged across all 40 lectures
generated under each model.
4.1.2
Lecture Delivery Evaluation
Next, we evaluate the embodied and multimodal teaching capability
of the framework. For each generated lecture, the ProfessorAgent
is tasked with teaching all 15 slides using lecture artifacts created
in the lecture generation session. This stage evaluates the agent’s
teaching action quality, independent of content generation, allow-
ing us to assess multimodal alignment and embodied instructional
delivery capabilities specifically.
Evaluation Metrics. Lecture delivery is evaluated using the Teach-
ing Action Quality (TAQ) metric, which has six rubric dimensions
(detailed in Table 3). These include temporal and spatial alignment
of teaching actions, accurate handwriting and rough notation ac-
tions, active learning, and overall embodied teaching experience.
TAQ assesses how well each model exploits the frameworks archi-
tecture to deliver accurate, coherent, and pedagogically informed
teaching action sequences. For each slide, script, and teaching action
sequence, an expert educator judges whether each rubric criterion
is satisfied, and the overall average TAQ score is computed using
Equation 4.
Results. TAQ results indicate that Lect¯uraAgents enables gener-
ally accurate and coherent teaching action sequences across mod-
els. As shown in Figure 9, models perform strongly on spatially
9

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Table 4: (RQ 1) Evaluation of Lect¯uraAgents across pedagogical metrics under frontier models
Rank
Model
LCQ (%)
PQ (%)
AQ (%)
TAQ (%)
AAR (%)
1
Gemini 3 Pro
80.2
83.3
81.6
76.5
80.4
2
GPT-5.1
76.1
80.5
82.3
76.2
78.8
3
Claude 4.5 Sonnet
72.4
78.6
76.2
80.4
76.9
4
Gemini 2.5 Pro
70.5
75.2
80.1
72.3
74.5
5
DeepSeek V3.2
68.9
73.1
75.2
77.8
73.5
6
GPT-4o
67.5
71.4
72.8
73.2
71.2
7
Qwen 3 Omni
65.4
70.3
56.5
64.3
64.1
Figure 9: (RQ1 and RQ2) Results across rubric dimensions for each evaluation metric under each frontier model.
grounded criteria, particularly spatial accuracy, handwriting ac-
tions, rough notation actions, and embodied teaching. This sug-
gests that the framework can reliably convert generated lecture
materials into visible instructional actions. Figure 10 further shows
that teaching-action-related scores are distributed across multiple
lecture artifacts, indicating that embodied delivery is maintained
across the broader lecture package rather than appearing only in
isolated outputs. A key factor behind this stability is the TASA mod-
ule, which provides the ProfessorAgent with a structured view of
slide regions and aligns teaching actions with corresponding speech
segments. While temporal alignment remains comparatively more
variable due to the difficulty of fine-grained action–speech synchro-
nization, Figure 11 shows that TAQ and personalization-related
10

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Figure 10: (RQ2) Average distribution of Personalization Quality and
Teaching Action Quality across diverse learning profiles at various
academic levels.
Figure 11: (RQ2) Overall average distribution of Lecture Content
Quality scores across generated Lecture Materials from all models.
performance remain broadly stable across high school, undergrad-
uate, master’s, and PhD learner profiles. This suggests that the
embodied teaching mechanism generalizes across academic levels,
while timing-sensitive action selection remains an area for improve-
ment.
4.1.3
Comparative Evaluation with Related Frameworks
We further assess Lect¯uraAgents against existing frameworks in
this domain. Due to varying capabilities between baselines, we only
compare performances on shared capabilities to ensure fairness.
We identify two closely related open-source frameworks and one
learning system with publicly available lecture data: Instructional
Agents [34], GenMentor [36], and Google’s Learn Your Way sys-
tem [3]. Our comparative evaluation assesses each framework or
system based on lecture content quality (LCQ), assessment quality
(AQ), and personalization (PQ) evaluation metrics, using the same
evaluation method described in Section 4.1. For InstructionalAgents
and GenMentor, we generated 20 lectures using their publicly re-
leased implementations. For Learn Your Way, we used the publicly
available lectures provided on its website. Additional details about
the lecture set and selection process are provided in Appendix B.
We then generated the same lectures with Lect¯uraAgents using
identical topics, prompts, and learner profiles, and evaluated all
outputs using the methodology described in Section 4.1.
Table 5: Performance comparison of Lect¯uraAgents with existing
related frameworks
N (number of lectures) = 20
Framework /
System
LCQ (%)
PQ (%)
AQ (%)
Overall (%)
Instructional
Agents [34]
52.1
53.2
51.4
52.2
GenMentor [36]
50.8
64.6
46.6
54.0
Learn Your
Way [3]
58.9
60.1
62.5
60.5
Lect¯uraAgents
70.3
73.5
71.2
71.6
Results. As shown in Table 5, Lect¯uraAgents obtains higher scores
than the baseline systems across LCQ, PQ, and AQ. The most notable
difference is observed in personalization quality, indicating that the
framework is better able to adapt generated materials to learner
profiles. Its performance in lecture content and assessment quality
further suggests that the framework supports not only learner-
specific adaptation, but also coherent instructional organization
and alignment between lecture materials and assessment tasks.
4.1.4
Efficacy Study with Students
The preceding evaluations assessed the pedagogical capabilities of
the framework across multiple topics, models, and personalization
settings. However, the impact of the framework is better exam-
ined when these capabilities are tested on real learners. Therefore,
we conducted a small-scale efficacy study with real students to
measure the holistic pedagogical value of Lect¯uraAgents in terms
of learning support and learner experience. To provide a broader
comparison, we included both Learn Your Way, representing a
modern AI-assisted learning system, and Adobe Acrobat Reader
v23.008.20555, representing a widely used traditional digital study
reading software without generative AI capabilities. The study in-
volved 45 students divided equally across the three learning systems,
with 15 participants per system. Each group comprised five stu-
dents from each educational level—high school, undergraduate, and
master’s—with ages ranging between 15 to 25 years. Students were
recruited through a short pre-study topic-familiarity screening and
provided informed consent prior to participation.
Result. Figure 12 compares students’ post-learning assess-
ment performance across learning systems. The results show
that Lect¯uraAgents achieved the strongest performance across all
learner groups, followed by Learn Your Way and Adobe Reader. Al-
though the improvement is modest, its consistency suggests that the
framework’s personalized and embodied teaching capabilities sup-
ported better short-term comprehension and content recall, rather
than merely improving students’ subjective learning experience.
Consistent with this pattern, Table 6 shows that students using
Lect¯uraAgents reported stronger perceived content understanding,
11

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Table 6: Student responses to a survey given after assessment
To what extent do you agree or disagree with the following statements?
% somewhat agree or strongly agree
Lect¯uraAgents
N = 15
Learn Your Way
N = 15
Adobe Reader
N=15
I felt adequately prepared to complete the assessment after using
today’s educational tool.
95%
80%
72%
I felt like today’s educational tool helped me gain a good
understanding of the topic.
100%
92%
65%
I would like to use today’s educational tool to support my learning
needs in the future.
87%
73%
63%
The educational tool I used today would make me more effective at
learning compared to other educational tools I currently use at home or
in school.
84%
67%
44%
Figure 12: Average scores from immediate assessment on topics
learned using Lect¯uraAgents, Learn Your Way, and Adobe reader.
assessment readiness, future learning support, and overall learning
experience than those using Learn Your Way or Adobe Reader.
5
Limitations and Future Work
We acknowledge several limitations that may inform future work.
First, while Lect¯uraAgents performs well on lecture content gener-
ation and embodied delivery, the current teaching action–speech
alignment module relies heavily on offline heuristics with a limited
set of supported teaching actions. This may constrain the richness
of embodied instruction and robustness across diverse slide layouts.
Second, the multi-agent orchestration can introduce latency and
compute overhead. Finally, the framework can sometimes inherit
common LLM failure modes such as factual errors, inconsistent
reasoning, and tool or prompt-sensitivity. Future work will (1) ex-
pand the teaching action taxonomy and improve action fidelity; (2)
transition from heuristic action–speech alignment to learned poli-
cies (e.g., training policies in a presentation slide environment with
preference optimization or reinforcement learning); (3) strengthen
grounding to reduce hallucinations; and (4) optimize orchestration
for efficiency while preserving pedagogical coherence and control-
ling compute costs.
6
Conclusion
In this paper we introduced Lect¯uraAgents, a hierarchical multi-
agent framework for end-to-end adaptive, personalized AI-assisted
learning experiences. The framework addresses two major issues
in personalized AI-assisted learning: (1) how can AI adaptively
personalize instructional contents to best meet the needs of diverse
learners? (2) how can such instructional contents be delivered in
embodied and pedagogically meaningful ways to ensure better
learning outcomes? In order to effectively address these issues,
Lect¯uraAgents is first modelled on a professor-student relationship
framing, wherein a ProfessorAgent leads a collaborative class of
specialized subordinate agents through research, planning, evalu-
ation, and embodied delivery of instructional contents that adapt
to diverse students. The framework’s personalized and embodied
capabilities (e.g., TASA algorithm) offer students enhanced learning
and study experiences. We evaluated Lect¯uraAgents through two
main experiments: a pedagogical evaluation under frontier models
across high school, undergraduate, and graduate-level topics, and
an efficacy study with real students. Experimental results show
substantial improvements over baseline frameworks in lecture con-
tent quality, personalization, assessment quality, and embodied
teaching performance. In addition, these findings are validated by
results from our efficacy study with students, which provide prelim-
inary evidence that the framework can improve learning outcomes
while enhancing learner experience. In conclusion, we position
Lect¯uraAgents offers as a pedagogically grounded framework for
personalized AI-assisted learning at scale.
References
[1] Ming Yang and Feng Wen. 2023. AI-powered personalized
learning journeys: Revolutionizing information management
for college students in online platforms. J. Inf. Syst. Eng.
Manag. 8, 1 (2023). Article 23196. doi:10.55267/iadt.07.14079
[2] Hadi Farhood, Matthew Nyden, Amir Beheshti, and Stephan
Müller. 2025. Artificial intelligence-based personalized learn-
ing in education: A systematic literature review. Discover Artif.
Intell. 5 (2025). Article 331. doi:10.1007/s44163-025-00598-x
[3] LearnLM Team, Alicia Martín, Amir Globerson, Amy Wang,
Anirudh Shekhawat, Anna Iurchenko, Anisha Choudhury,
Avinatan Hassidim, Ayça Çakmakli, Ayelet Shasha Evron,
12

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Charlie Yang, Courtney Heldreth, Diana Akrong, Gal Eli-
dan, Hairong Mu, Ian Li, Ido Cohen, Katherine Chou, Komal
Singh, Lev Borovoi, Lidan Hackmon, Lior Belinsky, Michael
Fink, Niv Efron, Preeti Singh, Rena Levitt, Shashank Agarwal,
Shay Sharon, Tracey Lee-Joe, Xiaohong Hao, Yael Gold-Zamir,
Yael Haramaty, Yishay Mor, Yoav Bar Sinai, and Yossi Ma-
tias. 2025. Towards an AI-augmented textbook. Preprint
at https://arxiv.org/abs/2509.13348. arXiv:2509.13348 doi:10.
48550/arXiv.2509.13348
[4] Jian Peng and Yu Li. 2025. Frontiers of artificial intelligence
for personalized learning in higher education: A systematic
review of leading articles. Appl. Sci. 15, 18 (2025). Article
10096. doi:10.3390/app151810096
[5] Meghan Novack and Susan Goldin-Meadow. 2015. Learning
from gesture: How our hands change our minds. Educ. Psychol.
Rev. 27, 3 (2015), 405–412. doi:10.1007/s10648-015-9325-3
[6] Hanne Marie Hegna and Trine Ørbæk. 2024. Traces of em-
bodied teaching and learning: A review of empirical studies
in higher education. Teach. High. Educ. 29, 2 (2024), 420–444.
doi:10.1080/13562517.2021.1989582
[7] Mina C. Johnson-Glenberg and Cathy Megowan-Romanowicz.
2017. Embodied science and mixed reality: How gesture and
motion capture affect physics education. Cogn. Res.: Princ.
Implic. 2 (2017). Article 24. doi:10.1186/s41235-017-0060-9
[8] Shuo Wang, Tianyi Xu, Haoliang Li, Chen Zhang, Jian Liang,
Jian Tang, Philip S. Yu, and Qingsong Wen. 2024. Large lan-
guage models for education: A survey and outlook. Preprint
at https://arxiv.org/abs/2403.18105. arXiv:2403.18105 doi:10.
48550/arXiv.2403.18105
[9] Lijia Chen, Peng Chen, and Zhijun Lin. 2020. Artificial intel-
ligence in education: A review. IEEE Access 8 (2020), 75264–
75278. doi:10.1109/ACCESS.2020.2988510
[10] Shubham Sharma, Piyush Mittal, Mohit Kumar, and Vineet
Bhardwaj. 2025. The role of large language models in person-
alized learning: A systematic review of educational impact.
Discover Sustain. 6 (2025). Article 243. doi:10.1007/s43621-025-
01094-z
[11] Emma Chen, Jae-Eun Lee, Jialin Lin, and Kenneth Koedinger.
2024. GPTutor: Great personalized tutor with large language
models for personalized learning content generation. In Pro-
ceedings of the 11th ACM Conference on Learning @ Scale (L@S
’24). ACM, Atlanta, GA, USA. doi:10.1145/3657604.3664718
[12] Songlin Xu, Xinyu Zhang, and Lianhui Qin. 2024. EduA-
gent: Generative student agents in learning.
Preprint at
https://arxiv.org/abs/2404.07963.
arXiv:2404.07963 doi:10.
48550/arXiv.2404.07963
[13] Tianfu Wang, Yi Zhan, Jianxun Lian, Zhengyu Hu,
Nicholas Jing Yuan, Qi Zhang, Xing Xie, and Hui Xiong. 2025.
LLM-powered multi-agent framework for goal-oriented learn-
ing in intelligent tutoring systems. In Companion Proceedings
of the ACM Web Conference 2025. ACM, New York, NY, USA,
510–519. doi:10.1145/3701716.3715244
[14] En Sun and Li Tai. 2025.
MultiTutor: Collaborative LLM
agents for multimodal student support. In Proceedings of the
Innovation and Responsibility in AI-Supported Education Work-
shop (Proc. Mach. Learn. Res., Vol. 273). 174–190. Available
at https://proceedings.mlr.press/v273/sun25a.html.
https:
//proceedings.mlr.press/v273/sun25a.html
[15] Michael Vaccaro, Matthew Friday, and Andrea Zaghi. 2025.
Multi-agentic LLMs for personalizing STEM texts. Appl. Sci.
15 (2025). Article 7579. doi:10.3390/app15137579
[16] Qiang Yang, Yifan Yang, Shiqi An, Tianhao Hao, and Guang
Xu. 2025. LLM-based collaborative agents with pedagogy-
guided interaction modeling for timely instructive feedback
generation. In Proceedings of the 34th International Joint
Conference on Artificial Intelligence (IJCAI ’25). 9972–9980.
doi:10.24963/ijcai.2025/1108
[17] Yifan Ma, Shuo Hu, Zhu Bo, Yifan Wang, Yuxuan Kang, Shiyu
Liu, and Kang Hao Cheong. 2025. EduVerse: A user-defined
multi-agent simulation space for education scenarios. Preprint
at https://arxiv.org/abs/2510.05650. arXiv:2510.05650 doi:10.
48550/arXiv.2510.05650
[18] Meng Yue, Weinan Lyu, Walid Mifdal, Jaemin Suh, Yiming
Zhang, and Ziyu Yao. 2024. MathVC: An LLM-simulated
multi-character virtual classroom for mathematics education.
Preprint at https://arxiv.org/abs/2404.06711. arXiv:2404.06711
doi:10.48550/arXiv.2404.06711
[19] Hao Li, Jifan Yu, Xinyu Cong, Yifan Dang, Daniel Zhang-Li,
Lin Mi, Yi Zhan, Huiqin Liu, and Zhiyuan Liu. 2025. Which
type of students can LLMs act? Investigating authentic sim-
ulation with graph-based human–AI collaborative systems.
Preprint at https://arxiv.org/abs/2502.11678. arXiv:2502.11678
doi:10.48550/arXiv.2502.11678
[20] Yao Shi, Rongkeng Liang, and Yong Xu. 2025. EducationQ:
Evaluating LLMs’ teaching capabilities through multi-agent
dialogue frameworks. In Proceedings of the 63rd Annual Meet-
ing of the Association for Computational Linguistics (ACL ’25).
32799–32828. doi:10.18653/v1/2025.acl-long.1576
[21] Haein Jin, Minji Yoo, Jihye Park, Yejin Lee, Xin Wang, and
Joonhwan Kim. 2025. TeachTune: Reviewing pedagogical
agents against diverse student profiles with simulated stu-
dents. In Proceedings of the CHI Conference on Human Factors
in Computing Systems (CHI ’25). ACM, New York, NY, USA.
Article 1073. doi:10.1145/3706598.3714054
[22] Rui Jia, Mingyuan Zhang, Feng Liu, Bin Jiang, Kun Kuang,
and Zhiwei Dai. 2025. EduAgentQG: A multi-agent workflow
framework for personalized question generation. Preprint
at https://arxiv.org/abs/2511.11635. arXiv:2511.11635 doi:10.
48550/arXiv.2511.11635
[23] Jiahuan Pei, Fanghua Ye, Xin Sun, Wentao Deng, Koen
Hindriks, and Junxiao Wang. 2025. Conversational educa-
tion at scale: A multi-LLM agent workflow for procedural
learning and pedagogic quality assessment.
Preprint at
https://arxiv.org/abs/2507.05528.
arXiv:2507.05528 doi:10.
48550/arXiv.2507.05528
[24] Jing Liu, Zhi Huang, Tian Xiao, Jun Sha, Jiarui Wu, Qi Liu,
Shuo Wang, and Enhong Chen. 2024. SocraticLM: Exploring
Socratic personalized teaching with large language models.
In Advances in Neural Information Processing Systems, Vol. 37.
[25] Sayan P. Chowdhury, Vojtěch Zouhar, and Mrinmaya Sachan.
2024. AutoTutor meets large language models: A language
model tutor with rich pedagogy and guardrails. In Proceedings
of the 11th ACM Conference on Learning @ Scale (L@S ’24).
ACM, New York, NY, USA, 5–15. doi:10.1145/3657604.3662041
13

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
[26] Xuejiang Wang, Chee-Poh Lee, and Bilge Mutlu. 2025. Learn-
Mate: Enhancing online education with LLM-powered per-
sonalized learning plans and support. In Extended Abstracts of
the CHI Conference on Human Factors in Computing Systems
(CHI EA ’25). ACM, New York, NY, USA. doi:10.1145/3706599.
3719857
[27] Jana
Gonnermann-Müller,
Jennifer
Haase,
Konstantin
Fackeldey,
and
Sebastian
Pokutta.
2025.
FACET:
Teacher-centred
LLM-based
multi-agent
systems—
Towards personalized educational worksheets.
Preprint
at
https://arxiv.org/abs/2508.11401.
arXiv:2508.11401
doi:10.48550/arXiv.2508.11401
[28] Kia Karbasi, Kevin Hong, Mohammad Amin Samadi, and Gre-
gory Pottie. 2025. Multi-agent collaborative framework for
math problem generation. In Proceedings of the 18th Inter-
national Conference on Educational Data Mining. 613–618.
doi:10.5281/zenodo.15870246
[29] Songlin Xu, Xinyu Zhang, and Lianhui Qin. 2024. EduA-
gent: Generative student agents in learning.
Preprint at
https://arxiv.org/abs/2404.07963.
arXiv:2404.07963 doi:10.
48550/arXiv.2404.07963
[30] Weibo Gao, Qi Liu, Linan Yue, Fangzhou Yao, Rui Lv, Zheng
Zhang, Hao Wang, and Zhenya Huang. 2025. Agent4Edu:
Generating learner response data by generative agents for
intelligent education systems. In Proceedings of the AAAI
Conference on Artificial Intelligence, Vol. 39. 23923–23932.
doi:10.1609/aaai.v39i22.34565
[31] Yao Shi, Rongkeng Liang, and Yong Xu. 2025. EducationQ:
Evaluating LLMs’ teaching capabilities through multi-agent
dialogue framework. In Proceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics. 32799–32828.
doi:10.18653/v1/2025.acl-long.1576
[32] Jana
Gonnermann-Müller,
Jennifer
Haase,
Konstantin
Fackeldey,
and
Sebastian
Pokutta.
2025.
FACET:
Teacher-centred
LLM-based
multi-agent
systems—
Towards personalized educational worksheets.
Preprint
at
https://arxiv.org/abs/2508.11401.
arXiv:2508.11401
doi:10.48550/arXiv.2508.11401
[33] Xian Peng, Pan Yuan, Dong Li, Junlong Cheng, Qin Fang, and
Zhi Liu. 2025. KELE: A multi-agent framework for structured
Socratic teaching with large language models. In Findings of
the Association for Computational Linguistics: EMNLP 2025.
Association for Computational Linguistics, Suzhou, China,
16342–16362. doi:10.18653/v1/2025.findings-emnlp.888
[34] Hao Yao, Wenjie Xu, Jacob Turnau, Nathan Kellam, and Hui
Wei. 2025. Instructional agents: LLM agents on automated
course material generation for teaching faculties. Preprint
at https://arxiv.org/abs/2508.19611. arXiv:2508.19611 doi:10.
48550/arXiv.2508.19611
[35] Xueqiao Zhang, Chao Zhang, Jianwen Sun, Jun Xiao, Yi Yang,
and Yawei Luo. 2025. EduPlanner: LLM-based multi-agent
systems for customized and intelligent instructional design.
IEEE Trans. Learn. Technol. 18 (2025), 416–427. doi:10.1109/
TLT.2025.3561332
[36] Tianfu Wang, Yi Zhan, Jianxun Lian, Zhengyu Hu,
Nicholas Jing Yuan, Qi Zhang, Xing Xie, and Hui Xiong. 2025.
LLM-powered multi-agent framework for goal-oriented learn-
ing in intelligent tutoring systems. In Companion Proceedings
of the ACM Web Conference 2025. ACM, New York, NY, USA,
510–519. doi:10.1145/3701716.3715244
[37] Zheyuan Zhang, Daniel Zhang-Li, Jifan Yu, Linlu Gong, Jin-
chang Zhou, Zhanxin Hao, Jianxiao Jiang, Jie Cao, Huiqin
Liu, Zhiyuan Liu, Lei Hou, and Juanzi Li. 2025. Simulating
classroom education with LLM-empowered agents. In Pro-
ceedings of NAACL-HLT 2025. 10364–10379. doi:10.18653/v1/
2025.naacl-long.520
[38] Jiahuan Pei, Fanghua Ye, Xin Sun, Wentao Deng, Koen
Hindriks, and Junxiao Wang. 2025. Conversational educa-
tion at scale: A multi-LLM agent workflow for procedural
learning and pedagogic quality assessment.
Preprint at
https://arxiv.org/abs/2507.05528.
arXiv:2507.05528 doi:10.
48550/arXiv.2507.05528
[39] Richard C. Atkinson and Richard M. Shiffrin. 1968. Human
memory: A proposed system and its control processes. In
Psychology of Learning and Motivation. Vol. 2. Academic Press,
New York, 89–195. doi:10.1016/S0079-7421(08)60422-3
[40] Nelson Cowan. 2008. What are the differences between long-
term, short-term, and working memory? In Prog. Brain Res.
Vol. 169. 323–338. doi:10.1016/S0079-6123(07)00020-9
[41] Yu Su, Qi Liu, Qi Liu, Zhenya Huang, Yuting Yin, Enhong Chen,
Chao Ding, Shuwei Wei, and Guoping Hu. 2018. Exercise-
enhanced sequential modeling for student performance pre-
diction. In Proceedings of the AAAI Conference on Artificial
Intelligence, Vol. 32. doi:10.1609/aaai.v32i1.11864
[42] Qi Liu, Shiwei Tong, Chuanren Liu, Hongke Zhao, Enhong
Chen, Haiping Ma, and Shijin Wang. 2019. Exploiting cog-
nitive structure for adaptive learning. In Proceedings of the
25th ACM SIGKDD International Conference on Knowledge Dis-
covery & Data Mining. ACM, New York, NY, USA, 627–635.
doi:10.1145/3292500.3330922
[43] Zhenya Huang, Qi Liu, Yuying Chen, Le Wu, Keli Xiao, En-
hong Chen, Haiping Ma, and Guoping Hu. 2020. Learning or
forgetting? A dynamic approach for tracking the knowledge
proficiency of students. ACM Trans. Inf. Syst. 38, 2 (2020).
Article 19. doi:10.1145/3379507
[44] Ilie Gligorea, Marius Cioca, Romana Oancea, Andra-Teodora
Gorski, Hortensia Gorski, and Paul Tudorache. 2023. Adaptive
learning using artificial intelligence in e-learning: A literature
review. Educ. Sci. 13, 12 (2023). Article 1216. doi:10.3390/
educsci13121216
[45] Le Ying Tan, Shiyu Hu, Darren J. Yeo, and Kang Hao Cheong.
2025. Artificial intelligence-enabled adaptive learning plat-
forms: A review. Comput. Educ.: Artif. Intell. 9 (2025). Article
100429. doi:10.1016/j.caeai.2025.100429
[46] Graham Hardaker and Louise E. Glenn. 2025.
Artificial
intelligence for personalized learning: A systematic litera-
ture review.
Int. J. Inf. Learn. Technol. 42, 1 (2025), 1–14.
doi:10.1108/IJILT-07-2024-0160
[47] Raffaella Folgieri, Marisa Gil, Miriam Bait, and Claudio Luc-
chiari. 2024. AI-powered personalised learning platforms for
EFL learning: Preliminary results. In Proceedings of the 16th
International Conference on Computer Supported Education,
Vol. 2. 255–261. doi:10.5220/0012672000003693
14

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
[48] María Fernanda Contrino, María Reyes-Millán, Paola Vázquez-
Villegas, and José Membrillo-Hernández. 2024. Using an adap-
tive learning tool to improve student performance and satis-
faction in online and face-to-face education for a more per-
sonalized approach. Smart Learn. Environ. 11 (2024). Article 6.
doi:10.1186/s40561-024-00292-y
[49] Yajun Chen. 2025. Evaluation of the impact of AI-driven
personalized learning platform on medical students’ learning
performance. Front. Med. 12 (2025). Article 1610012. doi:10.
3389/fmed.2025.1610012
[50] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta
Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer,
Nicola Cancedda, and Thomas Scialom. 2023.
Tool-
former: Language models can teach themselves to use
tools. In Advances in Neural Information Processing Sys-
tems.
https://proceedings.neurips.cc/paper_files/paper/
2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-
Conference.html
[51] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran,
Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing
reasoning and acting in language models. In Proceedings of the
International Conference on Learning Representations. https:
//openreview.net/forum?id=WE_vluYUL-X
[52] Noah Shinn, Federico Cassano, Edward Berman, Ashwin
Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Re-
flexion: Language agents with verbal reinforcement learning.
Preprint at https://arxiv.org/abs/2303.11366. arXiv:2303.11366
doi:10.48550/arXiv.2303.11366
[53] Zhendong Chu, Shen Wang, Jian Xie, Tinghui Zhu, Yibo
Yan, Jingheng Ye, Aoxiao Zhong, Xuming Hu, Jing Liang,
Philip S. Yu, and Qingsong Wen. 2025. LLM Agents for Edu-
cation: Advances and Applications. In Findings of the Associa-
tion for Computational Linguistics: EMNLP 2025. Association
for Computational Linguistics, Suzhou, China, 13782–13810.
doi:10.18653/v1/2025.findings-emnlp.743
[54] Feifei Wang, Wenqing Li, Richard E. Mayer, and Hui Liu.
2018.
Animated pedagogical agents as aids in multime-
dia learning: Effects on eye-fixations during learning and
learning outcomes. J. Educ. Psychol. 110, 2 (2018), 250–268.
doi:10.1037/edu0000221
[55] Wenqing Li, Feifei Wang, Richard E. Mayer, and Hui Liu. 2019.
Getting the point: Which kinds of gestures by pedagogical
agents improve multimedia learning? J. Educ. Psychol. 111, 8
(2019), 1382–1395. doi:10.1037/edu0000352
[56] Arthur C. Graesser, Katja Wiemer-Hastings, Peter M. Wiemer-
Hastings, and Roger J. Kreuz. 1999. AutoTutor: A simulation
of a human tutor. Cogn. Syst. Res. 1 (1999), 35–51. doi:10.1016/
S1389-0417(99)00005-4
[57] Benjamin D. Nye, Arthur C. Graesser, and Xiangen Hu. 2014.
AutoTutor and family: A review of 17 years of natural language
tutoring. Int. J. Artif. Intell. Educ. 24 (2014), 427–469. doi:10.
1007/s40593-014-0029-5
[58] Tao Xu, Yuan Liu, Yaru Jin, Yueyao Qu, Jie Bai, Wenlan Zhang,
and Yun Zhou. 2025.
From recorded to AI-generated in-
structional videos: A comparison of learning performance
and experience. Br. J. Educ. Technol. 56, 4 (2025), 1463–1487.
doi:10.1111/bjet.13530
[59] Alexander Holmberg. 2025.
Generating narrated lecture
videos from slides with synchronized highlights. Preprint
at https://arxiv.org/abs/2505.02966. arXiv:2505.02966 doi:10.
48550/arXiv.2505.02966
[60] Tushar Aggarwal and Aarohi Bhand. 2025. PASS: Presenta-
tion automation for slide generation and speech. Preprint
at https://arxiv.org/abs/2501.06497. arXiv:2501.06497 doi:10.
48550/arXiv.2501.06497
[61] hexgrad. 2025. Kokoro-82M. Hugging Face model repository.
Open-weight text-to-speech model. https://huggingface.co/
hexgrad/Kokoro-82M
[62] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Chris-
tine McLeavey, and Ilya Sutskever. 2023. Robust speech recog-
nition via large-scale weak supervision. In Proceedings of
the 40th International Conference on Machine Learning (Proc.
Mach. Learn. Res., Vol. 202). 28492–28518. arXiv:2212.04356
doi:10.48550/arXiv.2212.04356
[63] Rough Notation. 2020.
Rough notation library.
GitHub
repository at https://github.com/rough-stuff/rough-notation.
https://github.com/rough-stuff/rough-notation
[64] Alex Graves. 2013.
Generating sequences with recurrent
neural networks. Preprint at https://arxiv.org/abs/1308.0850.
arXiv:1308.0850 doi:10.48550/arXiv.1308.0850
[65] Mariska H. Knol, Conor V. Dolan, Gideon J. Mellenbergh,
and Han L. J. van der Maas. 2016. Measuring the quality
of university lectures: Development and validation of the
Instructional Skills Questionnaire (ISQ). PLOS ONE 11, 2
(2016). e0149163. doi:10.1371/journal.pone.0149163
[66] Rakshith S. Srinivasa, Zora Che, Chen Bo Calvin Zhang,
Diego Mares, Ernesto Hernandez, Jayeon Park, Dean Lee,
Guillermo Mangialardi, Charmaine Ng, Ed-Yeremai Hernan-
dez Cardona, Anisha Gunjal, Yunzhong He, Bing Liu, and
Chen Xing. 2025. TutorBench: A benchmark to assess tu-
toring capabilities of large language models.
Preprint at
https://arxiv.org/abs/2510.02663.
arXiv:2510.02663 doi:10.
48550/arXiv.2510.02663
15

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Appendix A
A.1 Lect¯uraAgents: Detailed Architecture
A.1.1 Core Modules and Components
The framework is organized into four core modules, each serving a distinct purpose in the lecture generation and lecture delivery stages.
These modules provide the infrastructure for agent coordination, LLM integration, teaching action alignment, memory management, and
content rendering. The modular design enables easy extension and maintenance of individual components.
Table A1: Lect¯uraAgents’ Core Modules and Components
Module
Location
Function
Key Classes
Agents
Lectura/LecturaAgents/module/agents
The Agents module implements the core
agent architecture with base interfaces, role
definitions (Coordinator, Executor,
Validator), and state management. This
module provides the hierarchical three-tier
agent system with collaboration
mechanisms (sequential and parallel) and
orchestration through SwarmOfRanks. It
handles agent lifecycle, task execution,
validation, and inter-agent communication.
Agent (base class),
ProfessorAgent,
LecturePlanner,
ResearchAgent,
SlideAgent,
ScriptAgent,
SpeechAgent,
TasaAgent
LLMs
Lectura/LecturaAgents/module/llms
Provides unified abstraction layer for
multiple LLM providers (OpenAI, Google,
Anthropic, DeepSeek, Qwen, Local)
enabling seamless model switching.
Handles authentication, API
communication, response formatting,
function calling, and streaming. It abstracts
provider-specific differences to provide
consistent interface for all agents.
LLMProvider (base class),
OpenAIProvider,
GoogleAIProvider,
AnthropicProvider,
DeepSeekProvider,
QwenProvider,
LocalLLMProvider
TASA
Lectura/LecturaAgents/tasa
Implements Teaching Action Salience
Analysis (TASA) for generating and
aligning synchronized teaching actions
(rough notation, handwriting) with speech.
This module processes speech scripts to
embed action markers, extracts word-level
timestamps from audio using Whisper
ASR [62], and creates temporal alignment
between visual actions and spoken content.
TASA
Adaptive Memory
Lectura/LecturaAgents/memory
Implements a three-layer adaptive memory
system: short-term memory for session
context, long-term memory for persistent
learner data, and dynamic memory for
adaptive learning patterns. This module
provides a unified AdaptiveMemory
interface that enables agents to access
learner context, preferences, and learning
history for personalization.
ShortTermMemory,
LongTermMemory,
DynamicMemory,
AdaptiveMemory
A.1.2 Agent Hierarchy and Roles
Agents are organized by rank and responsibility. Rank 1 agents (ProfessorAgent) serve as coordinators and validators at the highest level,
Rank 2 agents (LecturePlanner, ResearchAgent) coordinate execution and validate outputs, while Rank 3 agents (SlideAgent, ScriptAgent,
SpeechAgent, TasaAgent) execute specific tasks. Each agent has clearly defined responsibilities and access to appropriate tools and actions
for their role.
16

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Table A2: Agent Hierarchy and Roles
Rank
Agent
Role
Responsibilities
Tools / Actions
1
ProfessorAgent
Coordinator and Tutor
• Initiates lecture sessions
• Reviews and approves plans
• Validates final artifacts
• Reviews final lecture artifacts
• Delivers embodied lectures
research()
create_syllabus()
review_plan()
instantiate_groupchat()
create_lecture_notes()
create_study_guide()
create_assessments()
create_personalization()
review_artifacts()
embodied_teaching()
2
Lecture Planner
Validator
• Creates lecture plans.
• Manages and validates tasks
done by subordinate executor
agents.
• Assembles generated lecture
artifacts and submits to
ProfessorAgent for final review.
research()
create_plan()
validate_task()
assemble_artifacts()
3
Executor Agents
Executors
ResearchAgent: Conducts multi-turn
web searches on lecture topic, writes a
detailed research report and submits for
review by LecturePlanner.
web_search()
SlideAgent: Generates personalized
slide contents; Designs and build slides
with structured content blocks based on
learner’s preferences.
slide_builder()
research()
file_parser()
ScriptAgent: Creates engaging,
personalized narration scripts that aligns
with both slide contents and learner’s
preferences.
analyze_slide()
write_script()
SpeechAgent: Synthesizes and
generates speech audio from scripts
based on learner’s preferred instructor
voice. Uses TTS/ASR tool to create
word-level timestamps .
Whisper [62], Kokoro
TTS [61]
create_timestamps()
TasaAgent: Uses tools in TASA module
to segment and annotate slide contents
with heuristic based context for
prospective action-speech sequences. It
then processes speech timestamps and
slide contents into synchronized
embodied teaching action sequences
with embeded action markers (highlight,
underline, handwriting, etc.).
TASA Module
temporal_segmentation()
heuristic_analysis()
A.1.3 Agent States and Lifecycle
Agents transition through a well-defined state machine during task execution. The lifecycle begins with the IDLE state, progresses through
acknowledgment and execution phases, and concludes with completion, failure, or revision states. This state management ensures proper
task tracking, error handling, and enables agents to revise their work based on feedback from higher-ranking agents.
Table A3: Lect¯uraAgents’ States and Lifecycles
State
Description
Transition
IDLE
Agent is waiting for a task.
ACKNOWLEDGED
17

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
State
Description
Transition
ACKNOWLEDGED
Agent has received and
acknowledged given.
EXECUTING
EXECUTING
Agent is actively working on
assigned task.
COMPLETED, FAILED or REVISAL
COMPLETED
Agent has completed task
successfully.
IDLE (for next task)
FAILED
Task execution was unsuccessful.
REVISAL
REVISAL
Agent is revising work based on
feedback from self-reflection or
review.
EXECUTING
A.1.4 Multi-agent Collaboration
Agents within the same rank can collaborate using two primary mechanisms: sequential collaboration for dependent tasks and parallel
collaboration for independent tasks. The SwarmOfRanks mechanism enables hierarchical coordination across multiple ranks, allowing
complex workflows where agents at different levels coordinate their activities. These collaboration patterns are essential for orchestrating
the multi-stage lecture generation process.
Table A4: Collaboration Mechanisms
Type
Description
Use Case
Sequential Colab
Agents complete tasks one after
another, sharing responses.
When tasks depend on previous
outputs.
Parallel Colab
Agents complete tasks
simultaneously, while sharing
responses.
When tasks are independent.
Swarm of Ranks
Hierarchical coordination across
ranks
Multi-rank workflows
A.1.5 Tools and Capabilities
The framework provides a comprehensive set of tools that agents use to accomplish their tasks. These tools range from web search and
file parsing to text-to-speech synthesis and code execution. Each tool is designed to be modular and reusable, with clear interfaces that
agents can invoke during their execution. The tools abstract away complex operations like API interactions, file processing, and multimedia
generation.
Table A5: Tools and Capabilities
Tool
Purpose
Usage
Dependencies
Web Search
Multi-turn web research using
SerpAPI.
Used by ResearchAgent, ProfessorAgent,
LecturePlanner and SlideAgent
SerpAPI
Slide World
Dynamic slide environment with
canvas for teaching sessions.
Used by ProfessorAgent for embodied
lecture delivery.
HTML/CSS/JS/Python
Slide Builder
Custom slide design tool.
Used by SlideAgent for building and
rendering slides.
HTML/CSS/JS/Python
File Parser
Parses PDF, TXT, MD files.
Used by ProfessorAgent and SlideAgent to
extract content from additional materials.
PyPDF2, python-docx
Command line
For command execution to create
lecture artifacts.
Used by all agents to
read/write/edit/save/delete files.
Bash/Zsh
TASA Segmentor /
Aligner
Segments, annotates and aligns slide
contents with speech timestamps for
synchronized
Used by ProfessorAgent and TasaAgent
TASA Module
Research
A unified research tool that makes
use of web search plus an LLM to
perform deep research on topics.
Used by ResearchAgent, ProfessorAgent,
LecturePlanner and SlideAgent
SerpAPI + Underlying LLM
Whisper [62]
Extracts word-level timestamps from
audio.
Used by TasaAgent for action alignment.
Whisper ASR model
18

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Tool
Purpose
Usage
Dependencies
Kokoro TTS [61]
Generate speeches from scripts with
desired instructor voice.
Used by SpeechAgent for speech
synthesis.
Kokoro TTS
A.1.6 Adaptive Memory
Lect¯uraAgents utilizes a three-layer memory architecture to support adaptive and personalized learning experiences. Short-term memory
captures recent interactions within a session, long-term memory stores persistent learner-specific data across sessions, and dynamic memory
adapts to learning patterns and preferences. The adaptive memory module provides a unified interface that combines all three memory
types, enabling agents to access relevant context efficiently.
Table A6: Memory Types and Functionalities
Memory Type
Function
Storage
Update Frequency
Short-term Memory
Handles recent interactions and
context.
In-memory (session-based)
Per interaction
Long-term Memory
Manages persistent learner-specific
data.
File-based (JSON)
Per session
Dynamic Memory
Adaptive learning patterns and
preferences.
In-memory + file-based
Continuously updated
A.1.7 LLMs
We ensure the framework supports multiple frontier models from leading LLM providers through a unified API, allowing seamless switching
between different models based on task requirements, cost considerations, and performance needs. Each provider implementation handles
authentication, API communication, and response formatting, while the unified interface ensures that agents can work with any supported
model without code changes. This design enables flexibility in choosing the most appropriate model for each task.
Table A7: Supported LLM Providers and Models
LLM Provider
Supported Models
OpenAI
GPT-5.1, GPT-4o, o3-pro
Google AI
Gemini 3 Pro, Gemini 2.5 Pro, Gemini
Flash 2.5 Lite
Anthropic
Claude 4.5 Sonnet, Claude 4.1 Sonnet
DeepSeek
DeepSeek V3.2, DeepSeek-R1
A.1.8 Slide Content Block Types
To ensure accurate alignment and rubust slide contents, we ensure each slide can support multiple content block types that enable rich,
structured presentation of information. Each block type is designed for specific pedagogical purposes, from definitions and equations for
core concepts to examples, steps, and questions for engagement. The framework automatically renders these blocks with appropriate styling
and formatting, ensuring consistent visual presentation across all slides.
Table A8: Various Types of Slide Content Blocks
Block Type
Description
Rendering
Usage
Bullets
Brief, concise key points about
concepts and topics.
HTML list elements (<ul></ul>,
<ol></ol>, etc.)
Holds main contents for topic
Definition
Key term definitions.
HTML styled definition div
Core concepts
Example
Concrete examples
HTML highlighted example div
Examples
Equation
Mathematical equations
LaTeX rendering in a div
Formulas, proofs
Question
Interactive questions
HTML Question box div
Engagement
Link
External references
hyperlink / link element
Resources
Table
Structured data
HTML table element
Comparisons, data
Video
YouTube video embeds
HTML iframe element
Educational short videos
Image
Illustrative and educative images
with captions
HTML image element
Illustration
Steps
Step-by-step procedures
HTML numbered list
Algorithms, processes, etc.
19

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
A.2 More on Evaluation Methodology
A.2.1 Overview
Our evaluation adopts a rubric-based methodology for both pedagogical and comparative assessment, with generated learning and teaching
artifacts scored and validated by expert educators. The evaluation examines two core capabilities of the framework: its ability to generate
high-quality personalized lecture content for diverse learner profiles, and its ability to utilize these generated materials during embodied
teaching. Specifically, we evaluate Lect¯uraAgents using four main metrics: Lecture Content Quality (LCQ), Personalization Quality (PQ),
Assessment Quality (AQ), and Teaching Action Quality (TAQ). These metrics are applied across three evaluation settings: (1) Pedagogical
Evaluation under Frontier Models, which assesses personalized lecture generation and embodied lecture delivery across different frontier
models; (2) Comparative Evaluation with Related Frameworks, which compares Lect¯uraAgents with existing educational agent or personalized
learning frameworks, including InstructionalAgents, LearnYourWay, and GenMentor; and (3) Efficacy Study with Students, which examines
the framework’s practical learning support and learner experience using real student participants.
A.2.2 Lect¯uraAgents’ Pedagogical Evaluation Under Frontier Models
During this evaluation, we generated 40 lectures per model across seven models, resulting in 280 lectures in total. For each model, the lecture
set included 10 lectures per academic level, with 20 learner profiles in total (five profiles per level). The topics covered science, engineering,
history, art, and business. Details on these lectures can be found in the released dataset, available at HuggingFace1. The generated lecture
artifacts were assessed across four evaluation metrics: Lecture Content Quality (LCQ), Personalization Quality (PQ), Assessment Quality
(AQ), and Teaching Action Quality (TAQ). The evaluation followed a two-stage procedure. In Stage 1, an LLM analyst provided structured
rubric-based analysis for each lecture, identifying evidence relevant to the instructional criteria under each metric, as detailed in Table A9
and Table A10. In Stage 2, expert educators reviewed the LLM-generated analysis, validated the evidence, assigned the final rubric scores, and
made corrections where necessary. The verified scores were then aggregated to compute metric-level scores, overall averages, visualizations,
and comparative insights into model performance across academic levels and evaluation dimensions.
Table A9: Stages in Pedagogical Evaluations
Stage
Task
Command
Output
Stage 1
An LLM (GPT 5.2) gives detail
analysis of generated lecture
contents per academic level based
on rubrics or criteria in the
evaluation metrics.
python3 evaluate.py \
--model model_name \
--lecture lecture_name \
--level level_name \
--llm analysis_model
(JSON)
Detailed analysis for each generated
lecture at each academic level under a
model.
Stage 2
An expert educator validates,
scores and aggregate results for
respective rubrics.
python3 evaluate.py \
--aggregate \
--lecture lecture_name \
--level level_name
(JSON, Charts)
Comprehensive scores and results.
Table A10: Details on Evaluation Metrics, Rubrics, Descriptions and Their Input Files
Lecture Generation Evaluation
Evaluation Metric
Rubrics / Criteria
Description
Input Files
Lecture Content
Quality (LCQ)
Accuracy
Verifies factual correctness across all
generated materials.
All generated files
Clarity
Assesses clarity of explanation across
teaching materials.
lecture_plan.json, learner_profile.txt,
syllabus.json, scripts.json,
slides_content.json, slides/*.html,
lecture_notes_/*.md quiz.json and
exam.json
Coherence
Evaluates logical flow across all materials.
All generated files
Cognitive Load
Assesses lecture contents alignment with
learner’s background or level.
learner_profile.txt, syllabus.json,
scripts.json, slides_content.json,
slides/*.html, lecture_notes_/*.md
quiz.json and exam.json
1HuggingFace dataset: https://huggingface.co/datasets/Jaward/lectura-agents-data
20

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Evaluation Metric
Rubrics / Criteria
Description
Input Files
Syllabus Coverage
Verifies topic coverage.
syllabus.json, scripts.json,
slides_content.json, slides/*.html,
lecture_notes_/*.md quiz.json and
exam.json, study_guide.md
Instruction-following
Checks framework’s adherence to
instructions, tasks or prompts.
All generated files
Personalization
Quality (PQ)
Adaptive Emphasis
Assesses the framework’s ability to adapt
instructions to the learner’s learning
preferences or profile through.
learner_profile.txt, scripts.json,
slides_content.json, slides/*.html,
lecture_notes_/*.md quiz.json and
exam.json, study_guide.md
Preference Alignment
Checks content alignment with learning
preferences.
teaching_actions.json, scripts.json,
slides_content.json, slides/*.html,
lecture_notes_/*.md, quiz.json and
exam.json, study_guide.md
Engagement
Evaluates framework’s capability to
consistently engage the learner.
teaching_actions.json, scripts.json,
slides_content.json, slides/*.html,
lecture_notes_/*.md, quiz.json and
exam.json, study_guide.md
Motivation
Evaluate motivational elements across
learning materials.
teaching_actions.json, scripts.json,
slides_content.json, slides/*.html,
lecture_notes_/*.md, quiz.json and
exam.json, study_guide.md
Tone/Style
Evaluate language appropriateness
scripts.json, slides_content.json,
lecture_notes_/*.md, study_guide.md,
learner_profile.txt
Assessment Quality
(AQ)
Concept Coverage
Verifies whether assessments covered all
topics in the syllabus.
quiz.json, exam.json, syllabus.json,
slides_content.json
Cognitive Appropriateness
Evaluates assessment difficulty and its
alignment with the learner’s profile.
learner_profile.txt, quiz.json,
exam.json, syllabus.json,
slides_content.json
Answer Validity
Checks accuracy of solutions to
assessments.
quiz.json, exam.json, syllabus.json,
slides_content.json
Rationale
Evaluates the quality of explanation in
solutions.
quiz_solutions.json,
exam_solutions.json
Lecture Delivery Evaluation
Evaluation Metric
Rubrics / Criteria
Description
Input Files
Teaching Action
Quality (TAQ)
Temporal Alignment
Validates action-speech alignments.
action_speech_alignment.json,
scripts.json, speech_timestamps.json
Accurate Handwriting Action
Checks accuracy of handwriting actions,
i.e., whether words or phrases are written
clearly and correctly at the right time
frame.
slides/*.html (after applied actions),
action_speech_alignment.json
Accurate Rough Notation
Checks accuracy of rough notation
actions, i.e., whether notations like
highlight, underline, and circle actions
are applied correctly in the right region
and at the right time frame.
slides/*.html (after applied actions,
action_speech_alignment.json
Spatial Accuracy
Verifies annotation precision.
slides/*.html (after applied actions),
action_speech_alignment.json
Active Learning
Assesses the effect of teaching actions on
the learner’s engagement or focus during
teaching.
slides/*.html, quiz.json, exam.json,
action_speech_alignment.json
21

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Evaluation Metric
Rubrics / Criteria
Description
Input Files
Embodied Teaching
Evaluates overall embodied teaching
experience.
tasa_analysis.json,
teaching_actions.json, slides/*.html
(after applied actions),
action_speech_alignment.json,
scripts.json, speech_timestamps.json
A.2.3 Rating
Each rubric or criteria is evaluated as a boolean (satisfied or not), and these boolean scores are weighted and averaged to produce Average
Achieved Ratings (AARs) at the metric and overall levels. Thus, for the j-th lecture, the overall performance score for under a given model, is
computed as the weighted average of all passed rubric criteria 𝐴𝐴𝑅𝑗
𝑤, given by:
𝐴𝐴𝑅𝑗
𝑤=
Í𝑁𝑗
𝑖=1 𝑤𝑗
𝑖· 1𝑟𝑗
𝑖
Í𝑁𝑗
𝑖=1 𝑤𝑗
𝑖· 1𝑤𝑗
𝑖>0
where 𝑁𝑗is the number of rubric criteria for the j-th lecture, 𝑤𝑗
𝑖∈{-5, -3, -1, 0, +1, +3, +5}, is the weight assigned to the i-th criterion and
𝑟𝑗
𝑖∈{0,1}indicates whether criterion i is satisfied. When a criterion is satisfied 𝑟𝑗
𝑖= 1, it contributes a positive reward of +5, +3 or +1,
corresponding to a highly desirable, a desirable and important, or a nice-to-have behaviour, respectively. When a criterion is not satisfied
𝑟𝑗
𝑖= 0, it is explicitly treated as a failure state and contributes a non-positive score, spanning 0, -1, -3, -5: 0 denotes the lowest-severity failure
(no credit), -1 a minor failure, -3 a moderate failure, and -5 a critical failure (highly undesirable behaviour).
A.2.4 Expert Recruitment and Evaluation Procedure
Five expert educators were recruited through purposive sampling based on their experience in teaching, curriculum development, and
educational assessment. The panel consisted of secondary-school teachers and university instructors from STEM, social science, and
humanities disciplines, each with at least five years of teaching experience. Prior to the evaluation, the experts participated in an online
workshop, during which the evaluation dimensions, criteria, and weighting scheme were reviewed and refined to ensure pedagogical
relevance and consistency across educational levels and subject domains. During the evaluation, experts were assigned respective lecture
samples according to their areas of expertise; they reviewed the generated lecture artifacts and assigned final scores based on the agreed-upon
rubrics.
A.2.5 Comparative Evaluation of Lect¯uraAgents with Related Frameworks
Comparative analysis was done against two multi-agent frameworks (Instructional Agents and GenMentor) and one system (Google’s
Learn Your Way). For the frameworks, we generated 20 lectures (5 for each level spanning 10 profiles) using their released code and then
generated the same lectures with Lect¯uraAgents and compared performances. Table A11 summarizes generated lecture topics and profiles per
framework or system. For Google’s Learn Your Way system, given that no source code was released we instead utilized their already generated
sample lectures openly available on their website. We then generated these lectures with Lect¯uraAgents and compared performances as
well. Our comparative evaluation assesses each framework or system based on lecture content quality (LCQ), assessment quality (AQ) and
personalization (PQ) evaluation metrics using the same evaluation method described in Appendix A.2.3 and Appendix A.2.4.
Table A11: Generated Lectures for Comparative Analysis
Framework / System
Lecture and Learner Profile Details
Instructional Agents and
GenMentor
Lecture Title: Newton’s Laws of Motion
Learner Profile: 8th-grade high schooler interested in STEM, enjoys basketball, and prefers visual, hands-on
learning through diagrams, examples, and practical activities.
Lecture Title: Photosynthesis and Cellular Respiration
Learner Profile: 9th-grade high schooler interested in creative writing and music, enjoys sketching, and learns
biology best through story-like explanations, visuals, and everyday analogies.
Lecture Title: Quadratic Equations and Functions
Learner Profile: 10th-grade high schooler preparing for advanced mathematics, enjoys chess, and prefers
worked examples, graph-based explanations, and short practice problems.
Lecture Title: The Solar System and Planetary Motion
Learner Profile: 11th-grade high schooler interested in astronomy and planetary systems, enjoys tennis, and
prefers simulations, diagrams, and applied problem solving.
Lecture Title: World War II: Causes and Consequences
Learner Profile: 12th-grade high schooler interested in modern history and global conflict, enjoys soccer, and
prefers timeline-based explanations with cause-and-effect reasoning.
22

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Framework / System
Lecture and Learner Profile Details
Lecture Title: Intro to Large Language Models
Learner Profile: Undergraduate computer science student interested in artificial intelligence and language
technologies, enjoys basketball, and prefers intuitive explanations followed by coding examples.
Lecture Title: Machine Learning: Supervised vs Unsupervised
Learner Profile: Undergraduate data science student interested in machine learning methods and data patterns,
enjoys hiking, and prefers visual comparisons using real datasets.
Lecture Title: Molecular Biology: Gene Expression
Learner Profile: Undergraduate biology student interested in genetics and molecular regulation, enjoys
swimming, and prefers process diagrams with concept checks.
Lecture Title: Operating Systems: Process Scheduling
Learner Profile: Undergraduate learner interested in environmental science and sustainability, enjoys
photography, and learns systems concepts best through visual workflows, resource-allocation analogies, and
practical examples.
Lecture Title: Thermodynamics: Entropy and Free Energy
Learner Profile: Undergraduate chemistry student interested in thermodynamics and energy transformations,
enjoys cooking, and prefers equation walkthroughs connected to everyday examples.
Lecture Title: Advanced Machine Learning: Deep Neural Networks
Learner Profile: Master’s-level engineering student interested in deep learning and neural architectures, enjoys
tennis, and prefers model diagrams with optimization intuition.
Lecture Title: Advanced Operating Systems
Learner Profile: Master’s-level systems student interested in distributed computing and resource management,
enjoys cycling, and prefers architecture diagrams with performance trade-offs.
Lecture Title: Computational Biology: Sequence Analysis
Learner Profile: Master’s-level computational biology student interested in genomics and sequence alignment,
enjoys photography, and prefers algorithmic workflows with biological examples.
Lecture Title: Cryptography and Network Security
Learner Profile: Master’s-level learner interested in ancient history and ethics, enjoys debate, and learns
cryptography best through historical examples, trust scenarios, and clear protocol diagrams.
Lecture Title: Distributed Systems Architecture
Learner Profile: Master’s-level computer science student interested in scalable systems and fault tolerance,
enjoys tennis, and prefers system-design scenarios with failure cases..
Lecture Title: Advanced Quantum Field Theory
Learner Profile: PhD researcher interested in quantum fields and particle interactions, enjoys baseball, and
prefers formal derivations supported by physical intuition.
Lecture Title: Non-Equilibrium Statistical Mechanics
Learner Profile: PhD researcher interested in statistical physics and complex systems, enjoys tennis, and prefers
rigorous mathematical development with simulation examples.
Lecture Title: Synthetic Biology: Circuit Design
Learner Profile: PhD researcher interested in synthetic biology and programmable cellular circuits, enjoys
running, and prefers circuit schematics with lab-oriented examples.
Lecture Title: Topological Data Analysis in ML
Learner Profile: PhD researcher interested in topology and machine learning geometry, enjoys rock climbing,
and prefers visual abstractions grounded in data examples.
Learn Your Way
Lecture Title: Atoms and Molecules
Learner Profile: Middle schooler who likes reading.
Lecture Title: Carbon
Learner Profile: Undergrad who likes painting.
Lecture Title: Microeconomics and Macroeconomics
Learner Profile: Undergrad who likes food.
Lecture Title: Logical Statements
Learner Profile: Undergrad who likes writing.
Lecture Title: The Ancient Roman Economy
Learner Profile: Undergraduate who likes plants..
23

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Framework / System
Lecture and Learner Profile Details
Lecture Title: The 'Long-Haired' Comets
Learner Profile: Undergraduate who likes movies..
Lecture Title: Early Human Evolution and Migration
Learner Profile: Undergrad who like tennis..
Lecture Title: Intro to Data Structures and Algorithms
Learner Profile: High schooler who likes basketball.
Lecture Title: Critical Reading and Evidence-Based Response
Learner Profile: Middle schooler who likes soccer.
Lecture Title: Disruptions in the Immune System
Learner Profile: Middle schooler who likes food
Lecture Title: Earth and Sky
Learner Profile: Middle schooler who likes photography
Lecture Title: Theories of Slef-development
Learner Profile: Undergrad who likes cooking.
Lecture Title: What is Learning
Learner Profile: Undergrad who likes music.
Lecture Title: ”Reading” to Understand and respond
Learner Profile: Middle schooler who likes music.
Lecture Title: Micronomics and Macronomics
Learner Profile: Undergrad who likes cooking.
Lecture Title: An Overview of Economic Systems
Learner Profile: High schooler who likes movies.
Lecture Title: Early Human Evolution and Migration
Learner Profile: Undergrad who likes tennis
24

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Appendix B
B.1 Code and Data
The data supporting this study is currently available on our huggingface repository at: https://huggingface.co/datasets/Jaward/lectura-agents-
data. The code can be made available upon reasonable request from the corresponding author. Please follow the installation instructions
below or in the readme file to get started.
B.1.1 Installation and Usage
1.
Add all required api keys inside the .env file in the parent directory. You will need to provide two main api keys (1) for the LLM you
want to use (OpenAI, Anthropic, Gemini and Deepseek); (2) A SerpApi key for research, while this is optional, it highly recommended
to add one, as it helps reduce hallucination. Get key here: https://serpapi.com/manage-api-key
2.
Cd into the parent directory and install all required packages using this command:
pip3 install -r requirements.txt
3.
If you wish to use the frontend for lecture generation, start the app with this command:
python3 main.py
Figure 13: Frontend view (with no generated lecture)
This will open the teaching environment in your browser at: http://127.0.0.1:8080/. The page should look like Figure 13:
There will be a few already generated lectures in the right Lectures pane for you to quickly try or you can also generate new lectures through
either the chat pane or in the left prompt pane. Generated lecture materials will appear below the slide as they are generated.
25

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
Figure 14: Swarm-of-Ranks group chat view (during lecture generation)
During lecture generation you can follow the whole process unfolds in real-time in the group chat session, as shown in Figure 14.
Figure 15: Teaching and learning environment (with teaching actions on lecture or study contents)
After lecture generation is complete, the view will automatically update with the slide deck (as shown in Figure 15). Below the deck are
controls (Next, Play, Previous, Restart, Temporal Segmentation, and Chat).
4.
If you wish to use the terminal for lecture generation, run this command:
python3 lecture_prep.py \
--lecture_title "Your Lecture Title Goes Here" \
26

Lect¯uraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching
--lecture_desc "Describe the kind of lecture you want here" \
--learner_profile "Add details about yourself, your learning preferences, and your current understanding level here" \
--slides <enter desired number of core slides here> \
--level <enter academic level: highschool, undergrad, masters, or phd> \
--instructor_voice <choose desired instructor voice: professor_lectura, professor_sky, professor_isabella, etc.. > \
--llm <select desired model here: gpt-5.1, gpt-4o, o3-pro, gemini-3-pro, gemini-flash-2.5-lite, claude-4.5, claude-4.1> \
--research <enter research method: llm or google> \
--language <enter output language: english, chinese, french, or spanish> \
--speech_gen <choose speech backend: kokoro-tts, gemini-2.5-tts, or gpt-4o-mini-tts> \
--handwriting_gen <choose handwriting mode: handwriting_rnn_model or preset_font_handwriting> \
--slide_image <choose slide image mode: generate_only, generate_web_search, web_search_only, material_generate_alt, material_web_alt, or
material_only> \
--syllabus "Optional syllabus or curriculum text here" \
--additional_materials "Optional reference text or path(s) to .pdf, .txt, or .md files, separated by commas" \
--data_root <optional custom output directory>
5.
Example prompt:
python3 lecture_prep.py \
--lecture_title "Intro to Data Structures and Algorithms" \
--lecture_desc "A Computer Science lecture for a highschooler who likes basketball. Ensure covering these topics and more: 1. Introduction to
Data Types and Abstraction 2. Introduction to Algorithms 3. Algorithm Vs Program. Understanding Data Structures 4. Abstract Data Types:
(List, Set, Map, Priority Queue, Graph)" \
--learner_profile "Name: Taylor. Focus: Advanced Computer Science. Interests: Specialized algorithms, system design. Hobby: basketball.
Learning style: Deep dive into technical details." \
--slides 24 \
--level highschool \
--instructor_voice professor_sky \
--llm gpt-5.2 \
--research google
6.
To view the generated lecture in the teaching environment run this command:
python3 lecture_delivery.py --lecture <lecture folder name>
The folder could be, for example, intro-to-data-structures-and-algorithms.
27
