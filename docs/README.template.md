# Prática 3: Futebol Analytics com YOLO

Visão Computacional, Cesar School. Brasil x Marrocos, Copa do Mundo 2026 (1ª rodada, fase de grupos).

> **Estado desta versão.** Tudo o que não depende de anotação humana está feito e rodou de ponta a ponta.
> **A validação manual dos passes (Q5.3) está PENDENTE**: nenhum número de precisão/revocação/F1 existe ainda e
> `passes_anotados.csv` ainda não existe. As seções marcadas **PENDENTE** serão preenchidas depois.
> Este README é gerado por `src/gerar_readme.py` a partir de `docs/README.template.md`: todo número vem de um arquivo
> de `saidas/jogo1/` (conferido por `src/checar_readme.py`).

## 1. Grupo

| Nome | Email |
|---|---|
| Angelita Dias | TODO@cesar.school (email não informado) |
| Carolline Mariz | TODO@cesar.school (email não informado) |
| Fernando Rangel | TODO@cesar.school (email não informado) |
| Guaraci Rios | TODO@cesar.school (email não informado) |
| Ivo Caetano | TODO@cesar.school (email não informado) |
| João Rietra | jhlr@cesar.school |
| Mikael Mulatinho | TODO@cesar.school (email não informado) |

Quem apresenta (1 ou 2 pessoas): **TODO**.

## 2. Declaração de uso de IA

O código foi escrito e adaptado com ajuda de IA (agente Claude Code), a partir do código de referência da disciplina
(`eron93br/vc-20262-p3`, copiado para `src/` e estendido). **Este rascunho do README e das análises também foi escrito com
ajuda de IA a partir dos arquivos gerados; o grupo deve ler, conferir e reescrever com as próprias palavras antes da entrega**
(o enunciado pede análise e README autorais). TODO: o grupo confirmar/ajustar esta declaração.

## 3. Vídeo, trecho e como reproduzir

- Vídeo: <https://www.youtube.com/watch?v=vC3fV_awcWE> ("JOGO COMPLETO: BRASIL X MARROCOS", arquivo local 720p, 1280x720, 30 fps, 4 h 04 min).
- **Trecho usado: 00:20:00 a 00:26:00 do arquivo (6 min).** O enunciado pede 3 a 5 min; **os 6 min foram uma decisão do grupo e
  são um desvio declarado da regra**. O relógio do placar mostra aproximadamente o minuto 6:34 ao 12:34 do 1º tempo (0 x 0).
- O vídeo **não está no repositório** (só URL e intervalo). TODO: link do Google Drive com `rastreado.mp4` e o trecho cortado.
- Ambiente: MacBook Apple M1, 16 GB, backend MPS, Python 3.11 em `.venv`.

Comandos executados (de dentro de `src/`; a lista completa e na ordem está em `src/pipeline.sh`):

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r src/requirements.txt
cd src
ffmpeg -y -ss 00:20:00 -i ../dados/videos/fonte_jogo_completo.mp4 -t 360 -c:v libx264 -preset fast -crf 20 -an ../dados/videos/brasil_marrocos_20-26min.mp4
../.venv/bin/python escanear_trecho.py --video ../dados/videos/brasil_marrocos_20-26min.mp4 --saida ../saidas/jogo1/escaneamento
../.venv/bin/python comparar_modelos.py --video ../dados/videos/brasil_marrocos_20-26min.mp4 --saida ../saidas/jogo1/comparacao --inicio 0.8333333333 --fim 1.3333333333
../.venv/bin/python detectar_rastrear.py --video ../dados/videos/brasil_marrocos_20-26min.mp4 --saida ../saidas/jogo1 --modelo ../modelos/yolov8m.pt
../.venv/bin/python homografia.py camera --saida ../saidas/jogo1
../.venv/bin/python reclassificar_times.py --saida ../saidas/jogo1
../.venv/bin/python homografia.py calibrar --saida ../saidas/jogo1 --sem-janela
../.venv/bin/python bola_kalman.py --saida ../saidas/jogo1
../.venv/bin/python exemplos_bola.py --saida ../saidas/jogo1
../.venv/bin/python homografia.py projetar --saida ../saidas/jogo1
../.venv/bin/python qa_homografia.py --saida ../saidas/jogo1
../.venv/bin/python renderizar_video.py --saida ../saidas/jogo1
../.venv/bin/python analise_tracking.py --saida ../saidas/jogo1
../.venv/bin/python exemplos_id.py --saida ../saidas/jogo1
../.venv/bin/python metricas.py --saida ../saidas/jogo1 --sensibilidade
../.venv/bin/python xpass.py --saida ../saidas/jogo1 --what-if
../.venv/bin/python extras_taticas.py --saida ../saidas/jogo1
../.venv/bin/python anotar_passes.py roteiro --saida ../saidas/jogo1      # PENDENTE (humano): ver docs/2026-11-02-pendencias-humano.md
../.venv/bin/python validar_passes.py --saida ../saidas/jogo1 --anotado ../saidas/jogo1/passes_anotados.csv   # PENDENTE
```

Ordem diferente do roteiro: `camera` vem antes de `calibrar` porque a calibração usa marcos de **vários quadros** levados ao
quadro de referência por `G_t` (explicado em Q4.2), e `reclassificar_times.py` existe porque a calibração de cores original
caiu num replay (Q3.2). Código de tipagem: `mypy src/*.py` passa sem erros (`mypy.ini`).

## 4. Parte 1: aquisição

**Q1.1. Por que este jogo e este trecho?** O jogo é o Brasil x Marrocos pelo contraste dos uniformes (Brasil amarelo e azul,
Marrocos vermelho e verde), que ajuda o K-Means de cor. A câmera é uma tática muito alta e aberta, com os jogadores
pequenos (altura mediana de {{tracking_resumo.json:altura_mediana_pessoa_px}} px em 720p) e sol forte, com sombras de
estrutura no gramado. O trecho (20:00 a 26:00) foi escolhido pelo grupo e **não é um trecho limpo**: a varredura a 1 fps
(`escanear_trecho.py`) achou {{escaneamento/resumo_escaneamento.json:n_cortes_histograma}} quedas de histograma (cortes ou
transições), {{escaneamento/resumo_escaneamento.json:segundos_com_replay}} s com o selo REPLAY e o clipe começa e termina
com close-ups e replays (folhas de contato em `saidas/jogo1/escaneamento/`). Só {{camera_info.json:pct_validos_apos_qa}}% dos quadros
(`{{camera_info.json:quadros_validos_apos_qa}}` de `{{camera_info.json:n_frames}}`) viraram dados usáveis depois de tirar replays, close-ups e segmentos que
não conseguimos ancorar no campo (Q4.3). Mesmo assim processamos o intervalo inteiro (não trocamos o trecho) e marcamos os quadros inválidos.

**Q1.2. Resolução, fps, quadros e custo.** {{meta.json:largura}}x{{meta.json:altura}} a {{meta.json:fps}} fps; {{meta.json:n_frames}} quadros
no clipe. Detecção, rastreamento e times com `yolov8m` a 1280 px no M1: {{arredondar(2):meta.json:fps_processamento}} fps, ou seja,
cerca de {{dividir(0,60):meta.json:n_frames:meta.json:fps_processamento}} min para os 6 min de vídeo (aprox. {{dividir(0):meta.json:fps:meta.json:fps_processamento}} vezes mais lento que o tempo real).
A etapa de câmera (ORB + RANSAC) roda mais rápido que o YOLO e as demais etapas levam segundos.

## 5. Parte 2: detecção

Subtrecho de 30 s (50 s a 80 s do clipe, jogo corrido e sem cortes), comparado com `comparar_modelos.py`
(`saidas/jogo1/comparacao/comparacao_modelos.csv`). "Jogadores/quadro" conta só pessoas com o pé no gramado
(`gramado.py`); sem esse filtro a torcida e o staff inflam a contagem (coluna `pessoas_por_quadro_sem_filtro`).

**Q2.1. Tabela.**

{{tabela:comparacao/comparacao_modelos.csv|tag,imgsz,conf_bola,fps_processamento,jogadores_por_quadro,pessoas_por_quadro_sem_filtro,pct_quadros_com_bola,bola_pct_cobertura_final}}

A coluna `bola_pct_cobertura_final` é a cobertura da bola depois do filtro de gramado e do Kalman (Q4.1), por configuração; `pct_quadros_com_bola`
é a fração de quadros em que o YOLO devolveu alguma candidata (sem filtro).

Leitura: o `yolov8n` roda a {{comparacao/comparacao_modelos.csv:tag=n_1280:fps_processamento}} fps e o `yolov8m` a
{{comparacao/comparacao_modelos.csv:tag=m_1280:fps_processamento}} fps (a 1280). Em jogadores por quadro os três ficam perto
({{comparacao/comparacao_modelos.csv:tag=n_1280:jogadores_por_quadro}}, {{comparacao/comparacao_modelos.csv:tag=s_1280:jogadores_por_quadro}} e
{{comparacao/comparacao_modelos.csv:tag=m_1280:jogadores_por_quadro}}), então o modelo grande **não** achou muito mais gente. A
porcentagem de quadros com bola do `n` ({{comparacao/comparacao_modelos.csv:tag=n_1280:pct_quadros_com_bola}}%) é a mais alta, mas ela é só "o YOLO
devolveu alguma coisa que ele chama de bola", não "achou a bola" (ver Q2.3 e Q4.1: não temos gabarito de bola). Usamos o `yolov8m` no
clipe inteiro porque o `n` e o `s` espalham mais ids e mais candidatos de bola; mas essa escolha **não foi validada contra um gabarito**, e o `m`
custa {{arredondar(1):meta.json:fps_processamento}} fps. As colunas `fps` dos runs `m_1280_cb005`, `m_1280_k3` e `m_640` foram medidas em outros momentos
(com a máquina mais ou menos carregada), então não são comparáveis em detalhe com as três primeiras.

**Q2.2. Por que a bola é mais difícil?** No nosso vídeo a bola tem mediana de {{bola_cobertura.json:bola_largura_mediana_px}} x
{{bola_cobertura.json:bola_altura_mediana_px}} px, contra {{tracking_resumo.json:altura_mediana_pessoa_px}} px de altura de um jogador. Com tão
poucos pixels, o borrão de movimento da bola chutada e a oclusão por pernas e chuteiras (que são do tamanho e da cor dela) fazem o detector confundir
chuteira, meia, marca de cal e objeto branco na placa de LED com bola. Em Ahmad Yousef et al. (Tabela 2) é o mesmo padrão: o F1 de jogador
chega a 0,938 e o da bola sem Kalman fica em 0,310 (e vai a 0,96 com Kalman). A diferença do nosso caso é que não temos gabarito de bola para calcular F1; só
temos contagens de detecções e um julgamento visual de amostras (`docs/2026-11-02-observacoes_visuais.md`).

**Q2.3. Efeito de `--imgsz` e de `--conf-bola`.**

- `imgsz` 640 x 1280 com o `yolov8m`: a 640 só {{comparacao/comparacao_modelos.csv:tag=m_640:pct_quadros_com_bola}}% dos quadros têm candidata de bola,
  contra {{comparacao/comparacao_modelos.csv:tag=m_1280:pct_quadros_com_bola}}% a 1280. Com uma bola de {{bola_cobertura.json:bola_largura_mediana_px}} px, reduzir a imagem à metade
  apaga a bola. Em compensação a 640 roda a {{comparacao/comparacao_modelos.csv:tag=m_640:fps_processamento}} fps e há menos ids
  ({{comparacao/comparacao_modelos.csv:tag=m_640:ids_unicos_pessoas}} contra {{comparacao/comparacao_modelos.csv:tag=m_1280:ids_unicos_pessoas}}).
- `conf-bola` 0,05 x 0,10: baixar o limiar sobe os quadros com bola de {{comparacao/comparacao_modelos.csv:tag=m_1280:pct_quadros_com_bola}}% para
  {{comparacao/comparacao_modelos.csv:tag=m_1280_cb005:pct_quadros_com_bola}}%. Olhando uma amostra de 16 candidatas entre 0,05 e 0,10 (contagem visual),
  só 2 eram a bola; 7 eram objetos brancos redondos na base da placa de LED (provável bola reserva ou reflexo) e 7 eram falsos positivos (chuteira,
  camisa amarela na arquibancada, gramado vazio). Ou seja, mais cobertura, quase toda de lixo.
- Falsos positivos: `figuras/bola_candidatas_aleatorias.png` (amostra de 16 candidatas do `m_1280`: contagem visual 3 certas, 2 ambíguas, 11 falsas, a maioria chuteiras) e
  `figuras/bola_falsos_positivos.png` (candidatas rejeitadas pelo gating do Kalman no clipe inteiro). Contramedida que adicionamos em `bola_kalman.py`:
  descartar candidatas cujo entorno não é gramado (arquibancada, placas).

**Q2.4 (opcional). NÃO FEITO.** A variável de ambiente `ROBOFLOW_API_KEY` não estava definida nesta execução, então não baixamos o dataset
(`fsdm-olss1/detection-player`, Roboflow Universe, 2329 imagens, licença CC BY 4.0) nem treinamos. A comparação com o modelo COCO fica para quando a chave
estiver disponível; o `comparar_modelos.py --ajustado <pesos>` já aceita o modelo ajustado como mais uma linha da tabela.

## 6. Parte 3: rastreamento e times

**Q3.1. Trocas de ID.** Não temos gabarito de IDs; o que medimos é um **proxy** (`analise_tracking.py`): um track termina e outro começa
a até {{tracking_resumo.json:janela_quadros}} quadros e {{tracking_resumo.json:raio_px}} px, no mesmo segmento de câmera e longe da borda.
Contando só tracks de pelo menos 1 s, dá {{tracking_resumo.json:reaparicoes_entre_tracks_de_1s_ou_mais}} eventos em
{{tracking_resumo.json:minutos_validos}} min de vídeo válido, ou seja, cerca de {{tracking_resumo.json:reaparicoes_longas_por_minuto_valido}} por minuto
(sem o corte de 1 s: {{tracking_resumo.json:reaparicoes_por_minuto_valido}} por minuto). O número bruto de ids
({{tracking_resumo.json:ids_unicos_pessoas}} para uns {{tracking_resumo.json:pessoas_por_quadro_media}} jogadores por quadro) engana: os tracks curtos
(menos de 1 s: {{tracking_resumo.json:tracks_curtos_menos_de_1s}} ids) têm confiança média {{tracking_resumo.json:conf_media_tracks_curtos}}, contra
{{tracking_resumo.json:conf_media_tracks_de_2s_ou_mais}} nos tracks de 2 s ou mais, que carregam {{tracking_resumo.json:pct_linhas_em_tracks_de_2s_ou_mais}}% das linhas. Ou seja, boa parte dos ids é
cintilação do detector em caixas de baixa confiança (e caixas duplicadas), e não troca de jogador. Olhando 8 eventos sorteados
(`figuras/id_switch_exemplos.png`, contagem visual): 7 eram a mesma pessoa com ID novo (poucos quadros depois, ou após oclusão por outro jogador) e 1 era ambíguo, num
contato entre dois jogadores onde o ID pode ter passado para o adversário. Situações típicas: oclusão em aglomeração, jogador cortado pela borda e o fato
de a câmera girar (deslocamento do fundo: mediana {{tracking_resumo.json:camera_desloc_mediana_px_por_quadro}} px por quadro, p90 {{tracking_resumo.json:camera_desloc_p90_px_por_quadro}} px; o máximo de {{tracking_resumo.json:camera_desloc_max_px_por_quadro}} px provavelmente é erro de estimativa de `G_t`), contra caixas de ~25 px de largura, o que quebra a associação por IoU do ByteTrack. Tentamos re-rastrear em coordenadas do campo
para compensar a câmera; **piorou** (mais ids) e descartamos o experimento. Os picos de nascimento de ids estão em `figuras/ids_novos_tempo.png`.

**Q3.2. O K-Means de cor funcionou?** Depois de corrigir a calibração, pelo que vimos nos quadros (sem gabarito de time) os rótulos separam o amarelo do vermelho; não medimos a taxa de acerto. O problema original: `detectar_rastrear.py`
calibra o K-Means nos 60 primeiros quadros, e neste clipe eles são um replay com close-ups (um jogador marroquino e um técnico de terno cinza), então os
centróides saíram errados (o "time 1" ficou cinza). `reclassificar_times.py` refaz a classificação calibrando só com quadros válidos (foi o que usamos na execução principal, para não repetir 98 min de YOLO).
A causa também foi corrigida em `detectar_rastrear.py` (agora só calibra em plano aberto: 8 pessoas ou mais e altura mediana abaixo de 110 px); teste num subtrecho de 42 s
com `yolov8n` (`robustez/meta_teste_calibracao_corrigida.json`) deu centróides ({{robustez/meta_teste_calibracao_corrigida.json:times_bgr.0.0}}, {{robustez/meta_teste_calibracao_corrigida.json:times_bgr.0.1}}, {{robustez/meta_teste_calibracao_corrigida.json:times_bgr.0.2}}) e
({{robustez/meta_teste_calibracao_corrigida.json:times_bgr.1.0}}, {{robustez/meta_teste_calibracao_corrigida.json:times_bgr.1.1}}, {{robustez/meta_teste_calibracao_corrigida.json:times_bgr.1.2}}), parecidos com os da reclassificação. Em execuções novas, `reclassificar_times.py` não é necessário. Centróides finais
(`meta.json -> times_bgr`, em BGR): time 0 = ({{meta.json:times_bgr.0.0}}, {{meta.json:times_bgr.0.1}}, {{meta.json:times_bgr.0.2}}), amarelado (Brasil);
time 1 = ({{meta.json:times_bgr.1.0}}, {{meta.json:times_bgr.1.1}}, {{meta.json:times_bgr.1.2}}), avermelhado (Marrocos); figura `figuras/cores_times.png`.
Resultado por track: {{tracking_resumo.json:times_por_track.0.0}} tracks no time 0, {{tracking_resumo.json:times_por_track.1.0}} no time 1 e
{{tracking_resumo.json:times_por_track.-1.0}} sem time. Erros esperados e **não medidos** (sem gabarito de time): goleiros (camisa diferente das dos times), árbitro (de preto, cai em algum dos
dois grupos) e jogadores na sombra. Por que (S cos H, S sin H, V) em vez de H puro: o matiz H do OpenCV é um ângulo (0 a 179) e o vermelho fica nas duas pontas
(perto de 0 e perto de 179). A média simples de H de uma camisa vermelha dá algo perto de 90, que é ciano. Com S cos H e S sin H o matiz vira um ponto
num círculo e a média continua vermelha. Com a camisa vermelha do Marrocos, usar só H misturaria os pixels de matiz 0 e 179 num centróide ciano.

**Q3.3. `--k 3`.** No subtrecho de 30 s (`m_1280_k3`) o terceiro grupo ficou com {{comparacao/comparacao_modelos.csv:tag=m_1280_k3:tracks_outros}} tracks
("outros"), contra {{comparacao/comparacao_modelos.csv:tag=m_1280_k3:tracks_time0}} e {{comparacao/comparacao_modelos.csv:tag=m_1280_k3:tracks_time1}} nos times 0 e 1. Um cluster
com {{comparacao/comparacao_modelos.csv:tag=m_1280_k3:tracks_outros}} tracks é grande demais para ser só o árbitro e os goleiros, então não podemos afirmar que o árbitro foi separado (não conferimos track a track).
Vantagem: tira goleiros e árbitro dos dois times. Risco: o cluster extra também puxa jogadores na sombra ou de cor intermediária, que passam a "sumir" das métricas dos times.

## 7. Parte 4: bola e homografia

**Q4.1. Cobertura da bola antes e depois do Kalman (clipe inteiro, só quadros válidos).** O YOLO devolveu candidata de bola em
{{bola_cobertura.json:pct_deteccao_yolo_sobre_validos}}% dos {{bola_cobertura.json:quadros_validos}} quadros válidos (depois do filtro de gramado). O gating aceitou a
detecção em {{bola_cobertura.json:quadros_detectada_apos_gating}} quadros e o Kalman **preencheu** {{bola_cobertura.json:quadros_preenchidos_kalman}}; a cobertura final é de
{{bola_cobertura.json:pct_cobertura_final_sobre_validos}}% (`bola_cobertura.json`). Essa cobertura é **só preenchimento**: não sabemos quantos desses quadros têm a bola de verdade. Amostra visual de 16
lacunas do Kalman no subtrecho (`figuras/bola_kalman_lacunas.png`, contagem visual): em 5 a bola estava dentro do círculo da posição predita, em 2 a menos de uns 20 px, 1 ambígua e **em 8 a
bola não estava no recorte** (o filtro "inventou" a bola). Os acertos da amostra estão em lacunas curtas (3 a 6 quadros). Os erros estão sobretudo nas lacunas longas (10 a 15 quadros; 15 é o limite `--max-lacuna`), em que a
bola real saiu da trajetória reta que o filtro supõe (ou nunca esteve ali). Antes de colocarmos o filtro de gramado, 6 das
16 lacunas amostradas estavam dentro da arquibancada. Limitação conhecida: o Kalman roda em pixels da imagem, e a câmera gira; um Kalman em coordenadas estabilizadas
seria melhor e não foi feito.

**Q4.2. Calibração e erro de reprojeção.** O quadro 0 do clipe é um replay, então calibramos com marcos de **3 quadros** (pixels lidos por nós a olho, 9 marcos no total) levados ao quadro de referência pela cadeia `G_t`, com ajuste por mínimos quadrados (RANSAC a 1 m escolhia um subconjunto arbitrário com tão
poucos pontos). Erro médio {{arredondar(2):calibracao.json:erro_medio_m}} m e máximo {{arredondar(2):calibracao.json:erro_max_m}} m:

{{tabela:calibracao_erros.csv}}

A grade de 5 m reprojetada está em `saidas/jogo1/calibracao_grade.jpg`, `calibracao_grade_f1650.jpg`, `calibracao_grade_f2100.jpg` e, em quadros longe dos de calibração,
em `figuras/qa_homografia/`. Os maiores erros são de `circulo_inf`, `ga_dir_sup`, `circulo_esq` e `ga_dir_sup_linha`. Nossa hipótese (não testada): o extremo horizontal da elipse do círculo não é a projeção do ponto do círculo
que usamos como marco, e os pixels foram lidos a olho. Os três quadros ficam no **mesmo segmento** de câmera (o 9); os demais segmentos só chegam ao campo pela re-ancoragem (Q4.3). Todos os marcos
ficam no lado de cima (longe da câmera) do campo ou no meio: **não há marcos perto da linha lateral de baixo**, então a projeção ali é extrapolada e
menos confiável. **Resolução real: 9 marcos para 8 graus de liberdade, erro de cerca de 1 m; o que se vê nas grades é compatível, mas não validamos posições por um gabarito de campo.**

**Q4.3. `H0 · G_t`, deriva e re-ancoragem.** `G_t` leva o quadro `t` ao quadro de referência (ORB + RANSAC só no gramado, sem jogadores e placar, casando com um
quadro-chave) e `H0` leva o quadro de referência a metros; a posição é `H0 · G_t · [u, v, 1]`. Como a câmera só gira e dá zoom, `G_t` é uma homografia pura entre quadros.
Deriva: cada troca de quadro-chave soma erro. Medimos a deriva de forma **indireta**: o marco `circulo_esq` foi lido em dois quadros distantes (1366 e 1650) e os dois erros de reprojeção da tabela acima (linhas `circulo_esq`) ficam em torno de 1 m, e as grades de
`figuras/qa_homografia/` seguem batendo com as linhas minutos depois (segmento 22, por volta de 5 min), mas não calculamos uma curva de deriva. **O que fizemos de re-ancoragem**: em cada corte (queda de correlação de histograma) começa um novo segmento; ele só vale se casar com quadros-chave de segmentos anteriores
(ORB na cena toda abaixo do placar, não só no gramado, porque o gramado é repetitivo e a cena parada dá muito mais pontos) com pelo menos 80 inliers e uma homografia plausível.
Dos {{camera_info.json:n_segmentos}} segmentos candidatos (trechos contínuos de plano aberto, vários bem curtos), {{camera_info.json:n_segmentos_ancorados}} ancoraram; depois de um QA (se menos de 70% dos pés projetados caem dentro do campo, o segmento
é descartado) {{camera_info.json:segmentos_reprovados_qa}} foi reprovado por projeção implausível. O resto (quadros sem âncora: {{camera_info.json:motivos_invalidos.sem_ancora}}, replay:
{{camera_info.json:motivos_invalidos.replay}}, close-ups: {{camera_info.json:motivos_invalidos.plano_fechado_gramado}}) ficou **sem projeção**: preferimos NaN a posição errada. Tabela do QA:

{{tabela:qa_segmentos.csv}}

Proposta para ancorar melhor: casar com marcações de linha do campo detectadas (ou um modelo de campo) a cada N quadros, em vez de depender só de quadros-chave anteriores; ou
calibrar manualmente um quadro por segmento não ancorado.

**Q4.4. Pé x centro da caixa; bola no ar.** A homografia vale para pontos **no plano do gramado**. O pé (centro da base da caixa) está nesse plano; o centro da caixa está
a meio corpo de altura, e como a câmera é alta e oblíqua esse deslocamento vertical na imagem vira vários metros de erro no campo. Para a bola, que está no
chão só parte do tempo, o ponto projetado vale para a posição **no gramado abaixo da bola**: num lançamento a posição projetada fica adiante da bola real (efeito de paralaxe), e distâncias bola–jogador
ficam erradas. Não corrigimos isso; só registramos a limitação (a posse e os passes usam a bola projetada como se estivesse no chão).

## 8. Parte 5: métricas de jogo

Todos os números desta parte usam o raio de posse de {{metricas_times.json:raio_posse}} m, só quadros válidos, e dependem de uma bola em que não podemos confiar muito (Q4.1).

**Q5.1. Posse, distância e velocidade.** Posse por quadro com bola: time 0 (Brasil, amarelo) {{metricas_times.json:time0.posse_pct}}% e time 1 (Marrocos) {{metricas_times.json:time1.posse_pct}}%
(`metricas_times.json`), sobre {{metricas_times.json:quadros_com_bola}} quadros com bola em quadros válidos. Distância e velocidade estão em `metricas_jogadores.csv`, **por track_id**, e os ids se fragmentam
(Q3.1), então "distância percorrida por jogador" não é uma medida de jogador: mesmo os tracks mais longos têm só dezenas de segundos. Entre os {{metricas_times.json:plausibilidade.tracks_ate_10s_ou_mais}} tracks de
10 s ou mais, a mediana é de {{metricas_times.json:plausibilidade.mediana_m_por_min}} m/min (p90 {{metricas_times.json:plausibilidade.p90_m_por_min}}), **acima** dos cerca de 120 m/min de referência: o ruído de alguns décimos de metro no pé de cada quadro,
somado quadro a quadro (e a extrapolação da homografia), infla o caminho. A velocidade máxima (mediana {{metricas_times.json:plausibilidade.mediana_vel_max_ms}} m/s, máximo {{metricas_times.json:plausibilidade.max_vel_max_ms}} m/s) fica abaixo do limite de 12 m/s, que o código
já usa para descartar saltos. Conclusão: velocidades plausíveis, distâncias infladas pelo ruído e pelos ids quebrados.

**Q5.2. Sensibilidade ao raio de posse** (`sensibilidade_raio.csv`, `figuras/sensibilidade_raio.png`):

{{tabela:sensibilidade_raio.csv}}

Subir o raio de 1 para 3 m **dobra** o número de passes detectados ({{sensibilidade_raio.csv:raio_posse_m=1.0:passes}} para {{sensibilidade_raio.csv:raio_posse_m=3.0:passes}}) e mexe pouco na posse (time 0 entre {{sensibilidade_raio.csv:raio_posse_m=2.0:posse_time0_pct}}% e
{{sensibilidade_raio.csv:raio_posse_m=1.0:posse_time0_pct}}%): a posse é robusta ao raio, o número de passes não. Com uma bola e uma projeção tão ruidosas, o raio de 1 m perde toques e o de 3 m cria passes entre jogadores que só estão perto da bola.

**Q5.3. Validação manual dos passes. PENDENTE.** Não temos `passes_anotados.csv`. O que a pessoa precisa fazer:
`python anotar_passes.py roteiro --saida ../saidas/jogo1` (lista os trechos ao vivo a anotar) e `python anotar_passes.py passear --saida ../saidas/jogo1` (player que marca o instante do passe com a tecla `p`,
o time e o resultado), **sem olhar `passes.csv` antes**. Formato: `t_s,time,sucesso` (`t_s` em segundos do clipe; `time` 0 = amarelo/Brasil, 1 = vermelho/Marrocos como no `rastreado.mp4`; `sucesso` 1 = completou, 0 = errou).
Depois: `python validar_passes.py --saida ../saidas/jogo1 --anotado ../saidas/jogo1/passes_anotados.csv`. Resultado esperado a registrar aqui: precisão, revocação, F1 (`validacao_passes.json`) e os erros dominantes
(`validacao_erros.csv`). O detector achou {{metricas_times.json:time0.passes}} + {{metricas_times.json:time1.passes}} passes, dos quais {{xp_resumo.json:pct_certos}}% como "certos"; uma taxa tão abaixo do que se espera de passes
completados em jogo ({{arredondar(0):xp_resumo.json:pct_certos}}% contra algo perto de 80%) já indica que o detector **marca perda de posse onde não houve** (ruído da bola e do dono mais próximo). Isso é uma hipótese a confirmar com a anotação.

**Q5.4. Direção de ataque, trocas de ID e jogadores fora do quadro.** O código estimou que o time 0 ataca para a **direita** e o time 1 para a esquerda (`metricas_times.json`), e **não conseguimos confirmar** isso nos quadros:
os dois goleiros que apareceram (um de azul-claro no gol da direita, um de escuro no da esquerda) não identificamos de que time são, e a posição média dos times (`tracks_campo.csv`)
dá o time 0 mais à esquerda que o time 1 nos 6 segmentos válidos, que é compatível com as duas leituras. Como teste de robustez, rodamos `xpass.py` com a direção invertida
(`robustez/resultados_xp_direcao_invertida.csv`): a AUC do logístico passou de {{arredondar(3):resultados_xp.csv:modelo=logistico:AUC-ROC (↑)}} para
{{arredondar(3):robustez/resultados_xp_direcao_invertida.csv:modelo=logistico:AUC-ROC (↑)}} (uma diferença pequena, então os resultados de eXP não dependem muito da direção). **Correção proposta:** conferir no vídeo, olhando o goleiro e o
sentido do primeiro ataque, e forçar com `--ataque-time0`. Trocas de ID e jogadores fora do quadro: a câmera mostra só parte do campo, então só aparecem jogadores visíveis; passes e posse envolvendo alguém fora
do quadro viram "passe errado" ou "dono errado" (o mais próximo visível). Um ID novo para o mesmo passador em geral não cria falso passe (o filtro de 3 m de deslocamento da bola evita na maioria dos casos), mas encurta os tracks e
estraga distância e PAx por jogador. Correção proposta: estabilizar IDs com reidentificação por aparência ou com a posição em metros (nossa tentativa só com ByteTrack em metros não ajudou), e restringir as métricas coletivas ao que está visível.

## 9. Parte 6: eXP

**Q6.1. Features e os 17 canais de Gomes et al. (2026, Tabela 1).**

| Feature (`xpass.py`) | O que mede | Canal(is) análogo(s) da Tabela 1 |
|---|---|---|
| `distancia` | comprimento do passe | 17 (distância ao início do evento) e 3 (distância à bola) |
| `adversarios_na_linha` | defensores no cone de interceptação | 11 (defensores entre bola e célula) |
| `pressao_passador` | adversários a menos de 3 m do passador | 2 (posição do time defensor) |
| `espaco_destino` | distância do adversário mais próximo ao destino | 2 (posição do time defensor) e 4 (distância ao gol), de forma indireta |
| `progressao_x` | ganho em x | 5, 6 e 7 (cos, sen e ângulo ao gol) de forma indireta |
| `dist_destino_gol` | distância do destino ao gol | 4 (distância ao gol) |

**Não conseguimos extrair do vídeo de transmissão:** as velocidades (canais 8 a 9 e 12 a 16: bola, portador, atacantes e defensores em x e y) porque a posição vem de uma projeção ruidosa e de ids quebrados, e
a diferenciação amplifica ruído; a posição **de todos** os jogadores (canais 1 e 2), porque a câmera só mostra uma parte do campo e jogadores fora da imagem não existem para o modelo; a contagem de atacantes
entre bola e célula (canal 10) tem o mesmo problema; e a velocidade da bola exige uma bola confiável, que não temos.

**Q6.2. Baseline x heurístico x logístico** (`resultados_xp.csv`; **{{xp_resumo.json:passes_total}} passes**, {{xp_resumo.json:pct_certos}}% certos, só deste trecho: não juntamos passes de outro grupo com `--extra`):

{{tabela:resultados_xp.csv|modelo,AUC-ROC (↑),Brier (↓),Log-Loss (↓)}}

Referência do artigo: SoccerMap AUC 0,622, Brier 0,138, Log-Loss 0,444. Os nossos números **não são comparáveis**: o artigo usa dados de rastreamento reais e muitos passes; aqui são {{xp_resumo.json:passes_total}} passes, de um detector
ruidoso, e a validação cruzada de 5 folds sobre {{xp_resumo.json:passes_errados}} erros e {{xp_resumo.json:passes_certos}} acertos tem muito pouca confiança (um passe a mais ou a menos muda a AUC na segunda casa). Duas ressalvas
importantes: (1) o logístico tem a AUC mais alta, mas o coeficiente dominante é `espaco_destino` ({{xp_coeficientes.json:espaco_destino}}, features padronizadas), e o rótulo "passe errado" vem justamente de um adversário estar perto
da bola no destino: há **vazamento do rótulo na feature** em boa parte (o "passe errado" é definido por um adversário ser o próximo a tocar na bola, e a feature mede quão perto um adversário está do destino, que é onde ele tocou), então a AUC está inflada. (2) O baseline de distância tem AUC **abaixo de 0,5**: nesta amostra a distância sozinha não prevê o sucesso
(e a validação cruzada com poucos dados pode inverter o sinal em cada fold).

**Q6.3. Calibração.** `figuras/calibracao_xp.png`. A média prevista x taxa observada: heurístico {{arredondar(3):resultados_xp.csv:modelo=heuristico:média prevista}} x {{arredondar(3):resultados_xp.csv:modelo=heuristico:taxa observada}} (superestima os acertos: seus coeficientes foram um chute
para jogo real, onde ~80% dos passes completam), logístico {{arredondar(3):resultados_xp.csv:modelo=logistico:média prevista}} x {{arredondar(3):resultados_xp.csv:modelo=logistico:taxa observada}} (bate por construção: a regressão logística reproduz a taxa média da amostra).
Com 5 faixas e {{xp_resumo.json:passes_total}} passes, cada ponto da curva tem poucos passes e oscila.

**Q6.4. Superfície de eXP de um lance** (`figuras/superficie_xp_f914.png`: passe do primeiro evento detectado, passador #1564, time 0 atacando para a direita). As regiões frias (azul) são círculos ao redor dos jogadores adversários visíveis, e isso faz
sentido: passar para cima de um defensor tem eXP baixo. As regiões quentes (vermelho) são todo o espaço longe dos adversários, inclusive o círculo central e as pontas do campo. O que não faz sentido tático:
a superfície mostra o campo **quase todo quente** porque só entram os jogadores que a câmera enxerga; e regiões atrás da linha do passador ou perto da linha de fundo ficam quentes sem que o modelo conheça goleiro, linha de impedimento nem a direção de pressão.
Também aparecem "raios" retos na superfície: é a feature de cone de interceptação (`adversarios_na_linha`) criando fronteiras duras.

**Q6.5. What-if e PAx.** O what-if desloca o destino dos {{what_if_resumo.json:passes_errados}} passes errados em 3 a 6 m e reavalia: o ganho médio de eXP foi de {{what_if_resumo.json:ganho_medio_exp}}, com {{what_if_resumo.json:pct_ganho_acima_010}}% dos
passes errados com ganho maior que 0,10 (`what_if.csv`). Esse resultado é **quase garantido pelo desenho**: como `espaco_destino` domina o modelo, qualquer deslocamento para longe de um adversário sobe o eXP; não é evidência de que reposicionar o receptor teria salvado o passe.
PAx por jogador (`jogadores_xp.csv`): o maior é o passador #{{jogadores_xp.csv:passador=21589:passador}} (time 0), com {{jogadores_xp.csv:passador=21589:certos}} acertos em {{jogadores_xp.csv:passador=21589:passes}} passes e eXP somado {{arredondar(2):jogadores_xp.csv:passador=21589:eXP}}.
**Não dá para concluir nada sobre jogadores:** os ids são fragmentos de uns segundos, quase todo "jogador" tem um único passe (PAx = 1 menos o eXP desse passe), e 6 min (dos quais {{camera_info.json:pct_validos_apos_qa}}% usáveis) é pouco demais para estimar desempenho individual.

## 10. Métricas extras (Voronoi, compacidade, pressão, rede de passes)

Todas calculadas só de `tracks_campo.csv`, `passes.csv` e `posse_por_quadro.csv` (`extras_taticas.py`; resumo em `extras_resumo.json`); amostram 1 quadro a cada {{extras_resumo.json:amostragem_quadros}}.

| Métrica | Resultado | Figura | Limite de validade |
|---|---|---|---|
| (a) Controle de campo (Voronoi) | time 0: {{extras_resumo.json:controle_campo.media_time0_pct}}%, time 1: {{extras_resumo.json:controle_campo.media_time1_pct}}% da área visível ({{extras_resumo.json:controle_campo.amostras}} amostras) | `controle_campo.png` | só conta a área que a câmera mostra e só jogadores visíveis; assume que o jogador mais próximo "controla" a célula (sem velocidade); jogadores não detectados viram espaço livre falso |
| (b) Compacidade | área do envoltório: time 0 {{extras_resumo.json:compacidade.time0.area_hull_media_m2}} m², time 1 {{extras_resumo.json:compacidade.time1.area_hull_media_m2}} m²; largura {{extras_resumo.json:compacidade.time0.largura_media_m}} e {{extras_resumo.json:compacidade.time1.largura_media_m}} m; profundidade {{extras_resumo.json:compacidade.time0.profundidade_media_m}} e {{extras_resumo.json:compacidade.time1.profundidade_media_m}} m | `compacidade.png` | subestima o time inteiro (só vê quem está no quadro; goleiro e jogadores cortados ficam de fora) e um erro de classificação de time estica o envoltório |
| (c) Pressão | adversário mais próximo do portador, mediana: time 0 com a bola {{extras_resumo.json:pressao.time0_com_bola.dist_adv_mediana_m}} m, time 1 com a bola {{extras_resumo.json:pressao.time1_com_bola.dist_adv_mediana_m}} m | `pressao.png` | depende de saber quem é o portador, que depende da bola (Q4.1) e do ID do dono; o portador pode ser um falso dono |
| (d) Rede de passes e progressivos | {{extras_resumo.json:passes_progressivos.progressivos}} passes progressivos (ganho de pelo menos {{extras_resumo.json:passes_progressivos.limiar_m}} m ao gol) em {{extras_resumo.json:passes_progressivos.passes}}, dos quais {{extras_resumo.json:passes_progressivos.progressivos_certos}} completados | `rede_passes.png` | passes e ids são os do detector (Q5.3 pendente), nós da rede são fragmentos de jogador, não jogadores |

Os mapas de calor por time, o mapa de passes e a linha do tempo de posse estão em `figuras/mapa_calor_times.png`, `mapa_passes.png` e `posse_linha_tempo.png`.

## 11. O que falta e o que é fraco

- **PENDENTE (humano):** anotação de passes e a validação (Q5.3), emails dos 6 integrantes, quem apresenta, criar o repositório e fazer push, link do Drive, slides (.pptx e Google Slides). Ver `docs/2026-11-02-pendencias-humano.md`.
- **Q2.4 não feita** (sem `ROBOFLOW_API_KEY`).
- **Fraco, dito sem enfeite:** {{camera_info.json:pct_validos_apos_qa}}% do clipe virou dado; a bola é pouco confiável (Q4.1); passes e eXP saem desse ruído; a direção de ataque não está confirmada; a projeção
  não tem marcos perto da lateral de baixo; ids fragmentados.
- Código de referência: `soccermap_mini.py` (desafio extra, fora do escopo) só recebeu anotações de tipo, não foi executado.
