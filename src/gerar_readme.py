#!/usr/bin/env python3
"""
Gera README.md a partir de docs/README.template.md, trocando marcadores por valores
LIDOS dos arquivos gerados pelos scripts. É assim que se garante que nenhum número do
README difere dos arquivos de saída (penalidade de 20% no enunciado).

Marcadores
----------
{{ARQ:caminho.da.chave}}                  valor de uma chave de um JSON em saidas/<jogo>/ARQ
{{ARQ:coluna=valor:outra_coluna}}         célula de um CSV: linha onde coluna==valor
{{tabela:ARQ|col1,col2,...}}              CSV inteiro (ou colunas escolhidas) em tabela markdown
{{arredondar(N):...}}                     arredonda o valor do marcador (ex.: {{arredondar(1):ARQ:chave}})
{{dividir(N[,D]):ARQ1:chave1:ARQ2:chave2}} ARQ1.chave1 / ARQ2.chave2 / D com N casas (JSON)

Uso
---
python gerar_readme.py --saida ../saidas/jogo1 --modelo ../docs/README.template.md --destino ../README.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

PADRAO = re.compile(r"\{\{(.+?)\}\}")


def formatar(v: Any, casas: int | None = None) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer() and casas is None):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.{casas}f}" if casas is not None else f"{v:g}"
    return str(v)


def valor_json(arq: Path, caminho: str) -> Any:
    d: Any = json.loads(arq.read_text())
    partes = caminho.split(".")
    while partes:
        if isinstance(d, list):
            d, partes = d[int(partes[0])], partes[1:]
            continue
        for j in range(len(partes), 0, -1):          # chaves podem conter ponto (ex.: "0.0")
            chave = ".".join(partes[:j])
            if chave in d:
                d, partes = d[chave], partes[j:]
                break
        else:
            raise KeyError(f"{arq}: chave {caminho} não encontrada")
    return d


def valor_csv(arq: Path, cond: str, coluna: str) -> Any:
    df = pd.read_csv(arq)
    c, v = cond.split("=", 1)
    sel = df[df[c].astype(str) == v]
    if len(sel) != 1:
        raise KeyError(f"{arq}: condição {cond} deu {len(sel)} linhas")
    return sel[coluna].iloc[0]


def tabela(arq: Path, cols: list[str] | None) -> str:
    df = pd.read_csv(arq)
    if cols:
        df = df[cols]
    cab = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    def cel(x: Any) -> str:
        return x if isinstance(x, str) else formatar(round(float(x), 3) if isinstance(x, float) else x)

    linhas = ["| " + " | ".join(cel(x) for x in r) + " |"
              for r in df.itertuples(index=False)]
    return "\n".join([cab, sep, *linhas])


def resolver(marcador: str, base: Path) -> str:
    casas: int | None = None
    m = re.match(r"arredondar\((\d+)\):(.*)", marcador)
    if m:
        casas, marcador = int(m.group(1)), m.group(2)
    m = re.match(r"dividir\((\d+)(?:,(\d+))?\):([^:]+):([^:]+):([^:]+):([^:]+)", marcador)
    if m:
        extra = float(m.group(2)) if m.group(2) else 1.0
        num = float(valor_json(base / m.group(3), m.group(4)))
        den = float(valor_json(base / m.group(5), m.group(6)))
        return formatar(num / den / extra, int(m.group(1)))
    if marcador.startswith("tabela:"):
        arq, _, cols = marcador[len("tabela:"):].partition("|")
        return tabela(base / arq, [c for c in cols.split(",") if c] or None)
    partes = marcador.split(":")
    caminho = base / partes[0]
    if caminho.suffix == ".json":
        return formatar(valor_json(caminho, partes[1]), casas)
    if caminho.suffix == ".csv":
        return formatar(valor_csv(caminho, partes[1], partes[2]), casas)
    raise ValueError(f"Marcador desconhecido: {marcador}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera o README a partir do modelo e dos arquivos gerados")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--destino", required=True)
    a = ap.parse_args()
    base = Path(a.saida)
    texto = Path(a.modelo).read_text()
    n = 0

    def trocar(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return resolver(m.group(1).strip(), base)

    Path(a.destino).write_text(PADRAO.sub(trocar, texto))
    print(f"{n} marcadores resolvidos → {a.destino}")


if __name__ == "__main__":
    main()
