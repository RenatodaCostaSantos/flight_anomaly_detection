---
name: Decisão de design — voos with_emr_traj no treino
description: Voos engine_failure_with_emr_traj são mantidos no treino; usados como teste de detecção de anomalia
type: project
---

Voos com sufixo `with_emr_traj` (Emergency Responder Trajectory) são mantidos no conjunto de treino porque a parte inicial do voo ainda é normal (antes da falha de motor).

**Why:** O objetivo é testar se o modelo detecta a anomalia nesses voos — que têm três fases distintas: normal → falha → EMR ativo. É um teste natural de generalização do detector.

**How to apply:** Ao avaliar resultados do modelo nesses voos, esperar sinalização de anomalia no momento da falha de motor, não no início. A fase EMR pode ou não ser sinalizada como anômala — ambos os comportamentos são informativos.
