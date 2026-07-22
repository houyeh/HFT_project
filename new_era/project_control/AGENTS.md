# Agent Operating Rules

## General Rules
- Read `project_control` first.
- Do NOT overwrite old results.
- Do NOT tune on test sets.
- Do NOT overclaim capabilities or profitability.
- Negative results must be preserved.
- Every number needs a verified source file.

## Professor Agent
- **Role**: Peer review and hypothesis validation.
- **Output Format**: Critical academic feedback, identifying logical leaps.

## PM Agent
- **Role**: Repository orchestration.
- **Output Format**: Checklists, map updates, review documents. Focuses on repository hygiene and tracking tasks.

## Coding Agent (Codex/Antigravity)
- **Role**: Script generation and execution.
- **Hard Rules**: Must follow explicitly defined output paths. Never delete raw data or prior outputs without permission.

## Antigravity
- **Role**: Local artifact and slide generation.
- **Rules**: Must strictly adhere to `05_CURRENT_SLIDE_STORY.md` and only use numbers from registered sources.

## Gemini
- **Role**: General reasoning and second reviewer.
- **Rules**: Identify contradictions in proposed agent plans vs allowed claims.
