import numpy as np
import pandas as pd


def simulate_retention_campaign(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    retention_cost: float,
    average_clv: float,
    targeting_ratio: float,
) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "n_targeted": 0,
            "correctly_identified_churners": 0,
            "retention_cost_total": 0.0,
            "revenue_saved": 0.0,
            "net_benefit": 0.0,
        }

    ranked = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values(
        "y_prob", ascending=False
    )
    n_targeted = max(1, int(len(ranked) * targeting_ratio))
    targeted = ranked.head(n_targeted)

    correctly_identified_churners = int(targeted["y_true"].sum())
    retention_cost_total = float(n_targeted * retention_cost)
    revenue_saved = float(correctly_identified_churners * average_clv)
    net_benefit = float(revenue_saved - retention_cost_total)

    return {
        "n_targeted": n_targeted,
        "correctly_identified_churners": correctly_identified_churners,
        "retention_cost_total": retention_cost_total,
        "revenue_saved": revenue_saved,
        "net_benefit": net_benefit,
    }
