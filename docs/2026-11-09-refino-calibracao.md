# Refino da calibração pelas linhas do campo (2026-11-09)

## Problema
H0 saía de 9 pixels lidos a olho (erro de marcos de ~0,85 m), mas o campo projetado sobre
o quadro ficava torto: a linha do meio saía ~20 graus fora da linha branca real. Marcos
só no centro do campo deixam a rotação e a escala mal determinadas, e a cadeia G_t
acumula erro a cada corte. O erro "0,85 m" media a consistência dos cliques, não o
encaixe do campo na imagem.

## O que mudou
- `src/refino_linhas.py` (novo): por segmento aprovado, encaixa o modelo completo do campo
  (`campo.pontos_das_linhas`) nas linhas brancas detectadas em ~10 a 27 quadros do segmento.
  Termo direto (mapa de calor em 5 escalas + distância à linha), termo inverso (linhas
  detectadas em metros caem perto do modelo, evita colapso) e âncora em A = I.
- Aceite por segmento: ajusta em metade dos quadros e mede na outra metade. Só vale com
  ganho >= 0,05 na fração de pontos a menos de 3 px da linha.
- `homografia.projecao_por_quadro`: dono único de "pixel do quadro t para metros".
  `projetar`, `qa_homografia` e `extras_taticas` passam a usá-la (antes combinavam H0 e G à mão).
- `calibracao.json["refino_linhas"]` guarda H e métricas por segmento. `calibrar` regrava o
  arquivo sem essa chave (refino precisa rodar de novo depois de recalibrar).
- `src/pipeline.sh`: passo 7b.

## Resultado (quadros fora do ajuste, fração de pontos do modelo a menos de 3 px da linha)
| segmento | quadros | antes | depois |
|---|---|---|---|
| 8 | 12 | 0,188 | 0,496 |
| 9 | 18 | 0,247 | 0,435 |
| 10 | 21 | 0,193 | 0,420 |
| 12 | 5 | 0,144 | 0,303 |
| 19 | 27 | 0,146 | 0,232 |
| 22 | 15 | 0,180 | 0,491 |

Figura: `figuras/refino_antes_depois.jpg` (amarelo antes, ciano depois).

## Limites
- Segmentos 12 e 19 têm pouca linha visível (ganho menor; 12 usa só 5 quadros).
- O círculo central ainda sai um pouco maior que o real em alguns quadros (a olho, ~3 a 5%).
  Não investiguei a causa (escala horizontal pouco restrita, ou deriva de G_t dentro do segmento).
- Os erros por marco (0,37 a 1,29 m) em `calibracao_erros.csv` são os da calibração manual
  e não medem o refino. A medida do refino é a fração de inliers acima.
- Efeito nas métricas de jogo foi pequeno (mediana de m/min 138,8 para 142,6; posse igual).
  Validação de passes (43 anotados em quadros válidos): F1 0,40 para 0,35 (72 passes
  detectados, 20 casados contra 73 e 23 antes). Melhorar a geometria não melhorou a
  detecção de passes; o gargalo está na bola e no dono da posse.
- README regenerado em 2026-11-09 (template com refino na Q4.2 e validação na Q5.3);
  `checar_readme.py` passa. Os slides e `docs/2026-11-03-o-que-foi-feito.html` ainda trazem os números antigos.
