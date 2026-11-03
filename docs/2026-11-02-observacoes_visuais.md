# Observacoes visuais (julgamento humano/agente sobre imagens; NAO sao saida de script)

Tudo aqui foi lido a olho em mosaicos gerados por `src/exemplos_bola.py` e em folhas de contato de
`src/escanear_trecho.py`. Sao amostras pequenas e subjetivas: servem para ilustrar, nao para medir.
Numeros desta pagina podem aparecer no README, marcados como "contagem visual".

## Relogio do jogo

No quadro em 22 s do clipe o relogio do placar mostra 06:56; em 70 s mostra 07:43; em 130 s mostra 08:44.
Ou seja, o clipe (20:00-26:00 do arquivo) cobre aproximadamente o minuto 6:34 ao 12:34 de jogo, 1o tempo,
com o placar em 0 x 0 (BRA x MAR).

## Camera

Camera tatica muito alta e aberta (jogadores com 50-100 px de altura em 1280x720), sol forte com sombras
de estrutura no gramado, uniformes: Brasil amarelo/azul, Marrocos vermelho/verde.
A camera varre o campo todo (do gol esquerdo ao direito) em segundos.

## Amostra de 16 candidatas de bola do YOLO (m_1280, subtrecho de 30 s, seed 7)

Contagem visual: 3 claramente a bola, 2 ambiguas (objeto branco redondo na base da placa de LED), 11 falsos positivos
(a maioria chuteiras rosa/vermelhas, uma camisa amarela na arquibancada, um borrao verde na arquibancada).

## Amostra de 16 candidatas com conf entre 0.05 e 0.10 (m_1280_cb005, seed 7)

Contagem visual: 2 claramente a bola, 7 objetos brancos redondos na placa de LED/linha lateral (provavel bola reserva
ou reflexo, nao a bola em jogo), 7 falsos positivos (chuteiras, camisa amarela na arquibancada, gramado vazio).
Baixar o limiar de 0.10 para 0.05 acrescenta quase so lixo.

## Lacunas preenchidas pelo Kalman (m_1280, com filtro de gramado, seed 7)

Quadro do meio de 16 lacunas, posicao predita no circulo: bola dentro do circulo em 5, bola a ate cerca de 20 px
do circulo em 2, ambiguo em 1, bola ausente do recorte em 8 (gramado vazio ou so jogador).
Antes do filtro de gramado das candidatas, 6 das 16 lacunas amostradas estavam dentro da arquibancada/placas.

## Eventos de "reaparicao provavel" de ID (clipe inteiro, seed 3)

Oito eventos sorteados de `reaparicoes_provaveis.csv` (mosaico `id_switch_exemplos.png`): em 7 e a mesma pessoa
com um ID novo (poucos quadros depois, ou depois de oclusao por outro jogador) e 1 e ambiguo, em contato entre dois jogadores
(o ID pode ter passado para o adversario). Nenhum dos 8 e claramente uma troca entre dois jogadores.

## Direcao de ataque

Nao foi possivel confirmar a olho. Os goleiros que aparecem: um de azul-claro no gol da direita (visto aos 110 s do clipe) e um de
escuro no gol da esquerda (aos 120 s e 130 s); nao identificamos de qual time sao.
