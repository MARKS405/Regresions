from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable


def _kind_y(dep_var: str, target_var: str) -> str:
    if dep_var == f"ln_{target_var}":
        return "log_level"
    if dep_var == f"d_ln_{target_var}":
        return "log_diff"
    if dep_var == f"d_ln_{target_var}_12":
        return "log_diff_12"
    if dep_var == target_var:
        return "level"
    return "other"


def executive_interpretation(
    coef_df: pd.DataFrame,
    target_var: str,
    dep_var: str,
    dummy_vars: Iterable[str] = (),
    max_rows: int = 6,
) -> str:
    if coef_df is None or coef_df.empty:
        return "No se generó interpretación porque no hay tabla de coeficientes."
    dummy_vars = set(dummy_vars or [])
    df = coef_df[coef_df["variable"].str.lower() != "const"].copy()
    if df.empty:
        return "El modelo solo incluye constante; no hay regresoras para interpretar."
    df["abs_t"] = df["t_stat"].abs()
    df = df.sort_values(["p_value", "abs_t"], ascending=[True, False]).head(max_rows)
    ykind = _kind_y(dep_var, target_var)

    lines = []
    sig = (coef_df["p_value"] < 0.10).sum()
    lines.append(f"El modelo estima `{dep_var}` para explicar/reconstruir `{target_var}`. En la especificación actual, {sig} coeficientes son significativos al 10% o menos.")
    for _, r in df.iterrows():
        var = r["variable"]
        beta = r["coef"]
        p = r["p_value"]
        sign = "positivo" if beta > 0 else "negativo"
        signif = "significativo" if p < 0.10 else "no significativo"
        if var in dummy_vars:
            if ykind in ["log_level", "log_diff", "log_diff_12"]:
                eff = (np.exp(beta) - 1) * 100
                msg = f"`{var}` tiene signo {sign} y es {signif}; cuando toma valor 1, se asocia con un cambio aproximado de {eff:.2f}% en la escala de `{target_var}`/su crecimiento, según la transformación usada."
            else:
                msg = f"`{var}` tiene signo {sign} y es {signif}; cuando toma valor 1, `{dep_var}` cambia en {beta:.4f} unidades, manteniendo lo demás constante."
        elif var.startswith("ln_") and ykind == "log_level":
            msg = f"`{var}` tiene signo {sign} y es {signif}; puede leerse como una elasticidad aproximada de {beta:.3f}."
        elif ykind in ["log_diff", "log_diff_12"]:
            msg = f"`{var}` tiene signo {sign} y es {signif}; un aumento de una unidad en esta regresora se asocia con {beta*100:.2f} p.p. aprox. en el crecimiento logarítmico de `{target_var}`."
        else:
            msg = f"`{var}` tiene signo {sign} y es {signif}; el coeficiente estimado es {beta:.4f}."
        lines.append(msg)
    lines.append("Lectura ejecutiva: interpretar como asociación condicional, no como causalidad automática. Validar siempre sentido económico, diagnóstico de residuos y estabilidad de la especificación.")
    return "\n\n".join(lines)
