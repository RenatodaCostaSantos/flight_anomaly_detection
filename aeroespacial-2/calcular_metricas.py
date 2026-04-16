"""
Script de verificação das métricas dos modelos de detecção de falha.

Os valores são lidos dos JSONs gerados pelo pipeline Kedro
(src/.../model_training/nodes.py → evaluate_model → classification_report).

Reconstituição da matriz de confusão a partir de precision/recall/support
para explicar o recall aparentemente alto de Class 0 (Normal).
"""

import json
from pathlib import Path

# ── Caminhos dos JSONs ────────────────────────────────────────────────────────
BASE = Path(__file__).parent
METRICS = {
    "all_features": BASE / "data/07_model_output/metrics.json",
    "fft_features": BASE / "data/07_model_output/fft_metrics.json",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def reconstruir_confusao(report: dict, classe_normal="0.0", classe_falha="1.0"):
    """Reconstrói TN/FP/TP/FN a partir do classification_report.

    sklearn garante:
        recall_c    = acertos_c / support_c
        precision_c = acertos_c / total_predito_c
    """
    r0 = report[classe_normal]
    r1 = report[classe_falha]

    tn = round(r0["recall"] * r0["support"])   # normal corretamente classificado
    fp = round(r0["support"]) - tn             # normal erroneamente = anomalia
    tp = round(r1["recall"] * r1["support"])   # falha corretamente detectada
    fn = round(r1["support"]) - tp             # falha perdida

    return {"TN": tn, "FP": fp, "TP": tp, "FN": fn}


def imprimir_modelo(nome: str, path: Path):
    with open(path) as f:
        data = json.load(f)

    rep = data["classification_report"]
    cm  = reconstruir_confusao(rep)

    print(f"\n{'='*60}")
    print(f"  Modelo: {nome}")
    print(f"{'='*60}")

    print("\n  Classification Report (test set):")
    print(f"  {'Classe':<12} {'Precisão':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*52}")
    for cls, label in [("0.0", "Normal  (0)"), ("1.0", "Falha   (1)")]:
        m = rep[cls]
        print(f"  {label:<12} {m['precision']:>10.4f} {m['recall']:>10.4f}"
              f" {m['f1-score']:>10.4f} {int(m['support']):>10,}")
    print(f"  {'Acurácia':<12} {rep['accuracy']:>43.4f}")

    print("\n  Matriz de Confusão Reconstruída:")
    print(f"  {'':20} Previsto Normal   Previsto Falha")
    print(f"  {'Real Normal':20} TN = {cm['TN']:>8,}   FP = {cm['FP']:>6,}")
    print(f"  {'Real Falha':20} FN = {cm['FN']:>8,}   TP = {cm['TP']:>6,}")

    total_janelas = cm["TN"] + cm["FP"] + cm["TP"] + cm["FN"]
    flagged       = cm["TP"] + cm["FP"]
    frac_flagged  = flagged / total_janelas

    print(f"\n  Janelas totais no test set : {total_janelas:>10,}")
    print(f"  Janelas marcadas anomalia  : {flagged:>10,}  ({frac_flagged:.2%} do total)")
    print(f"  -> O parâmetro contamination=0.03 (3%) explica esse volume.")

    if data.get("detection_latency_s") is not None:
        print(f"\n  Latência de detecção (único voo no test set): {data['detection_latency_s']:.2f} s")

    # ── Explica o recall alto de Class 0 ─────────────────────────────────────
    fp_rate = cm["FP"] / (cm["TN"] + cm["FP"])
    print(f"\n  Taxa de falso-positivo por janela: {fp_rate:.4%}")
    print(f"  -> Recall 0 ≈ {rep['0.0']['recall']:.4f} = {cm['TN']:,} TN / {cm['TN']+cm['FP']:,} janelas normais")
    print( "     Alta porque Isolation Forest com baixa contamination raramente")
    print( "     alerta — mas esse conservadorismo custa recall na classe de falha.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for nome, path in METRICS.items():
        imprimir_modelo(nome, path)

    print("\n\nTabela resumida (para o relatório):")
    print(f"  {'Modelo':<14} {'Classe':<12} {'Precisão':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*58}")
    for nome, path in METRICS.items():
        with open(path) as f:
            rep = json.load(f)["classification_report"]
        for cls, label in [("0.0", "Normal (0)"), ("1.0", "Falha  (1)")]:
            m = rep[cls]
            print(f"  {nome:<14} {label:<12} {m['precision']:>10.4f}"
                  f" {m['recall']:>10.4f} {m['f1-score']:>10.4f}")
        print()
