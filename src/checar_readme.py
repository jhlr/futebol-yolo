#!/usr/bin/env python3
"""
Confere que TODO número escrito no README aparece (arredondado igual) em algum
arquivo gerado (saidas/**.json|csv|txt|log, docs/observacoes_visuais.md) ou é uma
constante declarada no próprio enunciado/spec (lista PERMITIDOS).

Ignora blocos de código, nomes de questões (Q2.1), datas, versões, URLs e números
dentro de links. O objetivo é pegar número digitado à mão que não vem de arquivo.

Uso
---
python checar_readme.py --readme ../README.md --saidas ../saidas --extras ../docs/observacoes_visuais.md
Sai com código 1 se houver número sem origem.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w])")
# Constantes do enunciado/spec (não são resultados medidos)
PERMITIDOS = set(range(1, 18)) | {
    20, 25, 26, 30, 35, 50, 60, 68, 70, 80, 104, 105, 120, 179, 640, 720, 1280, 2025, 2026, 2329,
    0.622, 0.588, 0.31, 0.96, 0.138, 0.444, 0.938, 0.916, 0.735,
}


def numeros_de_arquivos(raiz: Path, extras: list[Path]) -> list[float]:
    pool: list[float] = []
    # Só arquivos-resumo: CSVs por quadro (tracks, bola, posse...) têm milhões de números e fariam
    # qualquer valor "casar" por acaso; logs têm barras de progresso.
    csvs_resumo = {"comparacao_modelos.csv", "sensibilidade_raio.csv", "resultados_xp.csv", "jogadores_xp.csv",
                   "calibracao_erros.csv", "qa_segmentos.csv", "resultados_xp_direcao_invertida.csv"}
    arqs = [p for p in raiz.rglob("*") if p.is_file() and (p.suffix == ".json" or p.name in csvs_resumo)
            and p.name != "camera.npy"]
    arqs += [p for p in extras if p.exists()]
    for p in arqs:
        for t in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", p.read_text(encoding="utf-8", errors="ignore")):
            try:
                pool.append(float(t))
            except ValueError:
                continue
    return pool


def casa(x: float, casas: int, pool: list[float]) -> bool:
    return any(round(v, casas) == x or abs(v - x) < 10 ** -(casas + 1) / 2 + 1e-12 for v in pool)


def limpar(texto: str) -> str:
    texto = re.sub(r"```.*?```", " ", texto, flags=re.S)         # blocos de código
    texto = re.sub(r"`[^`]*`", " ", texto)                        # código inline
    texto = re.sub(r"\]\([^)]*\)", "]", texto)                    # alvos de link
    texto = re.sub(r"https?://\S+", " ", texto)
    texto = re.sub(r"\bQ\d+\.\d+\b", " ", texto)                  # Q2.1
    texto = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", texto)          # datas
    texto = re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\b", " ", texto)     # horários / HH:MM:SS
    texto = re.sub(r"\bPython \d+\.\d+\b|\byolov8[nsmlx]\b|\bYOLOv\d+\w*\b|\bv\d+\.\d+\b", " ", texto, flags=re.I)
    texto = re.sub(r"\b\d+x\d+\b", " ", texto)                    # 1280x720
    texto = re.sub(r"^\s*\d+\.\s", " ", texto, flags=re.M)        # numeração de listas
    texto = re.sub(r"^#+\s.*$", " ", texto, flags=re.M)           # títulos
    return texto


def main() -> None:
    ap = argparse.ArgumentParser(description="Confere números do README contra os arquivos gerados")
    ap.add_argument("--readme", required=True)
    ap.add_argument("--saidas", required=True)
    ap.add_argument("--extras", nargs="*", default=[])
    ap.add_argument("--listar", action="store_true", help="Lista também os números aceitos")
    a = ap.parse_args()

    pool = numeros_de_arquivos(Path(a.saidas), [Path(e) for e in a.extras])
    texto = limpar(Path(a.readme).read_text(encoding="utf-8"))
    sem_origem: list[str] = []
    for m in NUM.finditer(texto):
        tok = m.group(1)
        x = float(tok.replace(",", "."))
        casas = len(tok.replace(",", ".").split(".")[1]) if re.search(r"[.,]", tok) else 0
        if casas == 0 and x in PERMITIDOS:
            continue
        if casas > 0 and x in PERMITIDOS:
            continue
        if casa(x, casas, pool):
            if a.listar:
                print(f"ok   {tok}")
            continue
        ctx = texto[max(m.start() - 30, 0):m.end() + 20].replace("\n", " ")
        sem_origem.append(f"{tok}  …{ctx}…")
    if sem_origem:
        print(f"{len(sem_origem)} número(s) do README sem origem em arquivo gerado:")
        for s in sem_origem:
            print("  -", s)
        sys.exit(1)
    print("OK: todos os números do README aparecem nos arquivos gerados (ou são constantes do enunciado).")


if __name__ == "__main__":
    main()
