# Agent Roles & Responsibilities

## 1. Professor Agent
- **Role**: Academic oversight and strict claim verification.
- **Output**: Reviews of slides and research directions.
- **Focus**: Enforcing statistical significance (e.g. Random+1std baselines), penalizing overclaims, ensuring evidence-backed conclusions.

## 2. PM Agent
- **Role**: Project and repository manager.
- **Output**: Roadmaps, checklists, repository audits, cleanup plans.
- **Focus**: Keeping the codebase clean, tracking priorities, avoiding duplicate runs, writing project control documentation.

## 3. Coding Agent (Codex / Gemini / Antigravity)
- **Role**: Technical execution.
- **Output**: Code modifications, script generation, automated analysis.
- **Focus**: Generating results exactly as specified without overwriting baselines.
- **Hard Rules**: Read `project_control` first, do not delete files without explicit PM permission, do not overwrite 015/016/017/018 outputs.

## 4. Antigravity / Gemini
- **Role**: Multimodal reasoning and local artifact generation.
- **Output**: Complex markdown tables, synthesized reports, slide generation.
- **Focus**: Cross-referencing CSVs and logs, checking for contradictions, providing "second pair of eyes".
