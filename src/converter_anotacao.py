#!/usr/bin/env python3
"""
Converte a planilha de anotação manual (xlsx) para o formato do validador (Q5.3).

A planilha tem 2 colunas sem cabeçalho: horário e texto livre do evento
("brasileiro passa pra outro basileiro", "marrocos rouba bola", ...).

Horário: o Excel leu "20:19" como 20 h 19 min. Aqui isso significa minuto 20,
segundo 19 do VÍDEO (mm:ss). O clipe começa em 20:00 do vídeo (--inicio-video),
então t_s = (horas*60 + minutos) - 1200.

Regras (literais, sem inferir nada que o anotador não escreveu):
  - "passa pra outro X"                    -> passe do time citado, sucesso = 1
  - "... passa pra outro X Y intercepta"  -> passe do time citado, sucesso = 0
  - chute a gol, tiro de meta, roubo, interceptação, domínio, posse, falta,
    disputa: NÃO são passes. Ficam na tabela de eventos, fora do gabarito.
  - linhas idênticas (mesmo instante, time e texto) são colapsadas em uma.

Saídas em --saida:
  anotacao_eventos.csv   todos os eventos normalizados (para ler/conferir)
  passes_anotados.csv    t_s,time,sucesso (entrada de validar_passes.py)

Uso
---
python converter_anotacao.py --xlsx "~/Downloads/posse de bola.xlsx" --saida ../saidas/jogo1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TIME_BRASIL = 0
TIME_MARROCOS = 1
DURACAO_CLIPE_S = 360


def time_do_texto(texto: str) -> int | None:
    """Time do sujeito da frase: a primeira menção entre Brasil e Marrocos."""
    achados: list[tuple[int, int]] = []
    for time, raizes in ((TIME_BRASIL, ("brasil", "basileiro")), (TIME_MARROCOS, ("marroc", "morroc"))):
        for raiz in raizes:
            pos = texto.find(raiz)
            if pos >= 0:
                achados.append((pos, time))
    return min(achados)[1] if achados else None


def classificar(texto: str) -> tuple[str, int | None]:
    """Devolve (tipo, sucesso). sucesso só existe para tipo == 'passe'."""
    if "passa pra outro" in texto:
        pos_passe = texto.index("passa pra outro")
        interceptado = "intercepta" in texto[pos_passe:]
        return "passe", 0 if interceptado else 1
    if "chuta pra gol" in texto:
        return "chute_a_gol", None
    if "goleiro chuta" in texto:
        return "tiro_de_meta", None
    if "intercepta" in texto or "rouba" in texto:
        return "roubo_ou_interceptacao", None
    if "falta" in texto:
        return "falta", None
    if "brigam" in texto:
        return "disputa", None
    if "domina" in texto or "com bola" in texto or "posse" in texto:
        return "posse", None
    return "outro", None


def segundos_do_clipe(horario: pd.Timedelta, inicio_video_s: int) -> int:
    """Excel leu mm:ss como hh:mm; reinterpreta e passa para o relógio do clipe."""
    total_min = int(horario.total_seconds() // 60)  # horas*60 + minutos
    return total_min - inicio_video_s


def ler_planilha(xlsx: Path) -> pd.DataFrame:
    bruto = pd.read_excel(xlsx, header=None, usecols=[0, 1], names=["horario", "texto"])
    return bruto.dropna(subset=["horario", "texto"]).reset_index(drop=True)


def montar_eventos(bruto: pd.DataFrame, inicio_video_s: int) -> pd.DataFrame:
    linhas: list[dict[str, object]] = []
    for horario, texto_original in zip(bruto["horario"], bruto["texto"], strict=True):
        texto = str(texto_original).strip().lower()
        t_s = segundos_do_clipe(pd.Timedelta(horario), inicio_video_s)
        if not 0 <= t_s <= DURACAO_CLIPE_S:
            raise ValueError(f"horario fora do clipe: {horario!r} -> t_s={t_s}")
        tipo, sucesso = classificar(texto)
        time = time_do_texto(texto)
        if tipo == "passe" and time is None:
            raise ValueError(f"passe sem time identificavel: {texto_original!r}")
        linhas.append(
            {
                "t_video": f"{(t_s + inicio_video_s) // 60}:{(t_s + inicio_video_s) % 60:02d}",
                "t_s": t_s,
                "time": time,
                "tipo": tipo,
                "sucesso": sucesso,
                "texto_original": str(texto_original).strip(),
            }
        )
    eventos = pd.DataFrame(linhas)
    antes = len(eventos)
    eventos = eventos.drop_duplicates(subset=["t_s", "time", "texto_original"]).reset_index(drop=True)
    print(f"Linhas lidas: {antes}; idênticas colapsadas: {antes - len(eventos)}")
    return eventos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", required=True, type=Path)
    ap.add_argument("--saida", required=True, type=Path)
    ap.add_argument("--inicio-video", type=int, default=1200, help="início do clipe no vídeo, em segundos (padrão 1200 = 20:00)")
    a = ap.parse_args()

    eventos = montar_eventos(ler_planilha(a.xlsx.expanduser()), a.inicio_video)
    a.saida.mkdir(parents=True, exist_ok=True)
    eventos.to_csv(a.saida / "anotacao_eventos.csv", index=False)

    passes = eventos[eventos["tipo"] == "passe"][["t_s", "time", "sucesso"]].astype(int)
    passes.to_csv(a.saida / "passes_anotados.csv", index=False)

    print(eventos["tipo"].value_counts().to_string())
    por_time = passes.groupby("time")["sucesso"].agg(["count", "sum"])
    print("Passes anotados por time (count = passes, sum = completados):")
    print(por_time.to_string())


if __name__ == "__main__":
    main()
