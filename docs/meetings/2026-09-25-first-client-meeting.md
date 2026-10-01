# H18JDONUT Meeting Minutes

## Meeting 1 – First Client Meeting

| Field | Details |
|---|---|
| **Date** | 25 September 2026 |
| **Time** | Approximately 12:00–12:42 pm (Sydney time) |
| **Time Keeper** | Not recorded |
| **Attendees** | Raymond Louie (Client), Zhaoyu Zhang, Gewen Luo, Shumin Ye, Xinli Wei, Yining Xiong |
| **Apologies** | Purui did not attend the client discussion |
| **Please bring/read** | COMP9900 project specification and meeting agenda |
| **Other resources** | Google Teachable Machine |
| **Meeting purpose** | Clarify the educational goals, student journey, technical scope and next steps for the Interactive AI for Healthcare Education Platform |
| **Next Meeting** | 1 October 2026, 1:30 pm (Sydney time) |

---

## Agenda

| Item | Agenda Item | Assignee | Notes |
|---:|---|---|---|
| 1 | Project aims and audience | Raymond | The platform should provide a browser-based healthcare AI workshop. High school students are the main audience. A 40–60 minute workshop and an approximately five-minute Open Day demonstration were discussed. |
| 2 | Student journey and content | Team / Raymond | Students interact with distinct patient cards. A possible example is 20 training cards and 10 test cards. The website should visually explain data input, model training, prediction and biomarker / feature interpretation. |
| 3 | Model and data choices | Team / Raymond | Simple models and a small sample are acceptable. Fast live training is preferred where practical. A pre-trained model may be used as a fallback if live training is impractical. Synthetic or suitable public datasets may be used. |
| 4 | Website scope and deployment | Team / Raymond | Prioritise the desktop/browser-based website. QR scanning may be used to open or add the matching patient card data. Mobile optimisation can be considered later. Participant login is not currently required. |
| 5 | Schedule and follow-up | Zhaoyu / Raymond | The team will prepare a preliminary design for the next client meeting. Raymond will investigate CSE hosting and backend constraints. |

---

## Open Actions

| Item | Action / Discussed | Owner | Deadline | Status |
|---:|---|---|---|---|
| 1 | Prepare Technical Design: system architecture and web/software components; coordinate next client meeting invitation | Zhaoyu | Before next client meeting | To Do |
| 2 | Prepare Technical Design: machine-learning component | Gewen | Before next client meeting | To Do |
| 3 | Prepare interface storyboarding for the web application | Shumin | Before next client meeting | To Do |
| 4 | Prepare Background, Design Justifications and Functionality Mapping | Xinli | Before next client meeting | To Do |
| 5 | Prepare User Stories and Sprints | Yining / Purui | Before next client meeting | In Progress |
| 6 | Develop an initial age-appropriate interaction and example scenario for client review | Team | Next client meeting | To Do |
| 7 | Investigate CSE web hosting access and backend support | Raymond | Not specified | To Do |

---

## Questions for Mentor

No questions for the mentor were recorded during this client meeting.

---

## Questions for Client

- What should the precise QR scan and shared-screen interaction be for a classroom workshop?
- Which synthetic or public dataset and age-appropriate scenario should be used in the first demonstration?
- What CSE hosting access and backend facilities will be available?

---

## Decisions Made

| Item | Decision | Reason / Notes |
|---:|---|---|
| 1 | Use a browser-based healthcare AI learning platform | Suitable for classroom and Open Day use without requiring software installation |
| 2 | Use patient cards as the main workshop interaction | Each card represents a patient case and supports the training/test learning activity |
| 3 | Prioritise simple, understandable AI models | The educational workflow is more important than model complexity |
| 4 | Prefer live training where practical | Allows students to observe how training examples are used |
| 5 | Allow a pre-trained model as fallback | Prevents workshop timing or technical constraints from blocking the activity |
| 6 | Use age-appropriate and non-alarming explanations | Serious medical outcomes should not be presented as personal diagnoses |

---

## Jira / Sprint Updates

| Jira Item | Update | Owner |
|---|---|---|
| Sprint 1 | User Stories and Sprint scope to be refined based on client feedback | Yining / Purui |
| Product Backlog | QR interaction, training/test flow, explanation levels and future workshop features to be added | Yining / Purui |

---

## GitLab / Development Updates

| Area | Update | Owner |
|---|---|---|
| Frontend | Preliminary interface / storyboard to be prepared | Shumin |
| Backend | Architecture planning in progress | Zhaoyu |
| AI / Data | Model and dataset options to be investigated | Gewen |
| Project Management | Jira and User Stories to be updated | Yining |

---

## Blockers / Risks

| Item | Blocker / Risk | Owner | Next Action |
|---:|---|---|---|
| 1 | CSE hosting and backend constraints are not yet confirmed | Raymond / Team | Confirm available infrastructure |
| 2 | Dataset and first demonstration scenario are not yet finalised | Team | Investigate suitable synthetic or public datasets |
| 3 | Live model training may be too slow for the workshop | Team | Keep a pre-trained model as fallback |

---

## Notes

- The project should prioritise an educational and understandable AI experience rather than medical diagnostic accuracy alone.
- Patient-card, training, prediction and explanation workflows should remain suitable for the target student audience.