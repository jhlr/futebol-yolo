#!/usr/bin/env python3
"""
Validação dos passes detectados contra uma anotação manual (ground truth).

Assista ao trecho e anote os passes em um CSV (passes_anotados.csv):

    t_s,time,sucesso
    3.4,0,1
    7.9,0,0
    ...

t_s = instante do passe em segundos, contado a partir do INÍCIO DO CLIPE
processado (o mesmo relógio do vídeo rastreado.mp4); time = 0/1 conforme as
cores do rastreado.mp4 (T0 = amarelo, Brasil; T1 = vermelho, Marrocos: veja meta.json -> times_bgr); sucesso = 1 se completado.
(O auxiliar anotar_passes.py facilita a marcação.)

Um passe detectado "casa" com um anotado se estiver a até --tolerancia segundos
e for do mesmo time. Reporta precisão, revocação e F1 da detecção de passes.

Extensão deste projeto: se <saida>/quadros_validos.csv existe, só entram na conta
os passes (anotados e detectados) em quadros VÁLIDOS (ao vivo, plano aberto): o
detector não enxerga replays/close-ups, então cobrar revocação neles seria injusto.
Também grava validacao_passes.json e as listas de falsos positivos/negativos
(validacao_erros.csv) para a análise dos erros dominantes.

Uso
---
python validar_passes.py --saida ../saidas/jogo1 --anotado ../saidas/jogo1/passes_anotados.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def casar(det: pd.DataFrame, gt: pd.DataFrame, tol: float) -> list[tuple[int, int]]:
    """Casamento guloso por proximidade temporal (cada anotação casa no máx. uma vez)."""
    usados: set[int] = set()
    pares: list[tuple[int, int]] = []
    for i, d in det.sort_values("t_ini_s").iterrows():
        cand = gt[(gt["time"] == d["time"]) & ((gt["t_s"] - d["t_ini_s"]).abs() <= tol)]
        cand = cand[~cand.index.isin(usados)]
        if len(cand):
            j = int((cand["t_s"] - d["t_ini_s"]).abs().idxmin())
            usados.add(j)
            pares.append((int(i), j))
    return pares


def so_validos(df: pd.DataFrame, col_t: str, valido_por_s: pd.Series) -> pd.DataFrame:
    """Mantém linhas cujo instante cai num quadro válido."""
    idx = (df[col_t] * valido_por_s.attrs["fps"]).round().astype(int).clip(0, len(valido_por_s) - 1)
    return df[valido_por_s.to_numpy()[idx.to_numpy()]].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Precisão/revocação da detecção de passes")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--anotado", required=True)
    ap.add_argument("--tolerancia", type=float, default=1.0, help="Segundos")
    a = ap.parse_args()
    pasta = Path(a.saida)

    det = pd.read_csv(pasta / "passes.csv")
    gt = pd.read_csv(a.anotado)
    arq_val = pasta / "quadros_validos.csv"
    if arq_val.exists():
        meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
        valido = pd.read_csv(arq_val)["valido"].astype(bool)
        valido.attrs["fps"] = float(meta["fps"])
        n_det0, n_gt0 = len(det), len(gt)
        det = so_validos(det, "t_ini_s", valido).reset_index(drop=True)
        gt = so_validos(gt, "t_s", valido).reset_index(drop=True)
        print(f"Filtro de quadros válidos: detectados {n_det0}→{len(det)}, anotados {n_gt0}→{len(gt)}")

    pares = casar(det, gt, a.tolerancia)
    vp = len(pares)
    precisao = vp / len(det) if len(det) else 0.0
    revocacao = vp / len(gt) if len(gt) else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if vp else 0.0
    acordo = sum(det.loc[i, "sucesso"] == gt.loc[j, "sucesso"] for i, j in pares) / vp if vp else 0.0

    casados_det = {i for i, _ in pares}
    casados_gt = {j for _, j in pares}
    fp = det[~det.index.isin(casados_det)].assign(tipo="falso_positivo", t_s=lambda d: d["t_ini_s"])
    fn = gt[~gt.index.isin(casados_gt)].assign(tipo="falso_negativo")
    erros = pd.concat([fp[["tipo", "t_s", "time", "sucesso"]], fn[["tipo", "t_s", "time", "sucesso"]]])
    erros.sort_values("t_s").to_csv(pasta / "validacao_erros.csv", index=False)

    res = {"detectados": len(det), "anotados": len(gt), "casados": vp, "tolerancia_s": a.tolerancia,
           "precisao": round(precisao, 3), "revocacao": round(revocacao, 3), "f1": round(f1, 3),
           "acordo_certo_errado_nos_casados": round(acordo, 3),
           "falsos_positivos": len(fp), "falsos_negativos": len(fn)}
    (pasta / "validacao_passes.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Detectados: {len(det)} | Anotados: {len(gt)} | Casados: {vp}")
    print(f"Precisão  : {precisao:.2f}")
    print(f"Revocação : {revocacao:.2f}")
    print(f"F1        : {f1:.2f}")
    print(f"Acordo certo/errado nos casados: {acordo:.2f}")


if __name__ == "__main__":
    main()
