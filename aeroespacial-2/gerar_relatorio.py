from fpdf import FPDF

TITULO = "Detecção de Falha de Motor em VANTs com Aprendizado de Máquina"

PARAGRAFOS = [
    (
        "Ideia central",
        (
            "O projeto desenvolve um sistema de detecção automática de falha de motor em voo "
            "para uma aeronave não tripulada de asa fixa (CarbonZ). A motivação é crítica: em voo "
            "autônomo, identificar o momento exato da perda de tração com latência mínima é o que "
            "viabiliza uma resposta de emergência eficaz — seja um pouso de emergência ou um comando "
            "de autorrotação. O conjunto de dados conta com 33 voos registrados em formato ROS, sendo "
            "23 com falhas induzidas de motor e 10 voos normais, com dados de múltiplos sensores "
            "coletados a ~520 Hz."
        ),
    ),
    (
        "Como foi feito",
        (
            "A solução foi construída como um pipeline Kedro completo, cobrindo desde a ingestão "
            "dos tópicos ROS (com alinhamento temporal entre sensores via merge_asof) até o "
            "treinamento de modelos. Foram desenvolvidas duas trilhas paralelas de engenharia de "
            "features: a primeira (all_features) extrai grandezas físicas como energia específica, "
            "esforço de controle e razão de planeio, além de estatísticas em janelas deslizantes "
            "(média, desvio e inclinação linear); a segunda (fft_features) opera no domínio da "
            "frequência, extraindo potência de pico, entropia espectral e razão de alta frequência "
            "dos sinais do IMU e magnetômetro. A seleção das 20 melhores features foi feita pelo "
            "d de Cohen, que mede a separabilidade entre as distribuições em voo normal e em falha "
            "sem o viés de autocorrelação temporal que afetaria a importância de Random Forest. "
            "O modelo escolhido foi Isolation Forest, treinado com janelas deslizantes de 20 amostras "
            "por ponto, no regime não supervisionado — adequado à escassez de exemplos rotulados de falha."
        ),
    ),
    (
        "Resultados e trade-off entre os modelos",
        (
            "Os dois modelos têm perfis de erro opostos, o que revela que capturam fenômenos físicos "
            "distintos. O modelo all_features detectou 16 das 23 falhas (69,6%) com latência mediana "
            "de 3,15 s, reagindo às consequências da falha: queda de energia, divergência entre "
            "throttle e velocidade, acúmulo de erros de controle. Seu ponto fraco é a baixa revocação "
            "na classe de falha (18,4%) — ele é conservador e tende a silenciar em casos ambíguos. "
            "Já o modelo fft_features detectou 15 falhas (65,2%) com latência mediana de 2,64 s, mas "
            "12 dessas 15 detecções ocorreram antes do momento oficial da falha — o que sugere que "
            "ele captura a assinatura de vibração e caos espectral da degradação do motor como "
            "precursor, não como consequência. O custo é alto: 60% dos voos normais geraram ao menos "
            "um alarme falso no all_features, contra 30% no fft_features. Em termos de aplicação, o "
            "all_features é preferível quando a confiabilidade operacional importa mais — cada alarme "
            "dele tem alta probabilidade de ser real. Já o fft_features seria mais adequado em contexto "
            "de detecção precoce tolerante a falsos positivos — por exemplo, como primeiro estágio de "
            "um sistema em cascata, onde sua detecção antecipada dispararia o all_features como "
            "confirmatório. Um ensemble com votação ponderada pode combinar a sensibilidade espectral "
            "do FFT com a especificidade física do all_features."
        ),
    ),
]

# ── Métricas (janela a janela) ────────────────────────────────────────────────
MET_CABECALHO = ["Modelo", "Classe", "Precisão", "Recall", "F1"]
MET_LINHAS = [
    ["all_features", "Normal (0)", "0.874", "0.999", "0.932"],
    ["all_features", "Falha  (1)", "0.983", "0.184", "0.310"],
    ["fft_features", "Normal (0)", "0.856", "0.955", "0.903"],
    ["fft_features", "Falha  (1)", "0.402", "0.158", "0.227"],
]
MET_COL_W = [42, 30, 28, 24, 24]

# ── Dados por voo (notebooks 04) ──────────────────────────────────────────────
# Classificação dos voos COM falha
# Colunas: nome curto | t_falha(s) | t_detec(s) | latência(s) | resultado | FP antes
FALHA_ALL = [
    ("2018-07-18 #1", "115.3", "—",     "—",     "FN",           "0"),
    ("2018-07-18 #2", "72.4",  "—",     "—",     "FN",           "0"),
    ("2018-07-18 EMR","115.6", "117.6", "1.92",  "TP",           "0"),
    ("2018-07-18 EMR2","113.1","115.8", "2.72",  "TP",           "0"),
    ("2018-07-30 EMR","122.2", "133.1", "10.98", "TP",           "0"),
    ("2018-07-30 #1", "115.8", "—",     "—",     "FN",           "0"),
    ("2018-07-30 #2", "90.7",  "—",     "—",     "FP antes",     "126"),
    ("2018-07-30 EMR2","116.2","129.1", "12.91", "TP",           "0"),
    ("2018-07-30 EMR3","86.7", "94.0",  "7.34",  "TP",           "0"),
    ("2018-07-30 EMR4","132.5","135.4", "2.92",  "TP",           "0"),
    ("2018-07-30 EMR5","89.4", "100.5", "11.16", "TP",           "0"),
    ("2018-09-11 #1", "102.6", "103.3", "0.72",  "TP",           "0"),
    ("2018-09-11 #2", "103.9", "—",     "—",     "FN",           "0"),
    ("2018-09-11 #3", "48.9",  "59.5",  "10.62", "TP",           "0"),
    ("2018-10-05 #1", "48.2",  "64.2",  "16.00", "TP",           "0"),
    ("2018-10-05 #2", "99.1",  "101.7", "2.58",  "TP",           "0"),
    ("2018-10-05 #3", "75.2",  "91.2",  "15.99", "TP",           "0"),
    ("2018-10-18 #1", "103.2", "106.0", "2.76",  "TP",           "0"),
    ("2018-10-18 #2", "110.1", "113.1", "3.00",  "TP",           "0"),
    ("2018-10-18 #3", "99.4",  "—",     "—",     "FN",           "0"),
    ("2018-10-18 #4", "97.3",  "—",     "—",     "FN",           "0"),
    ("2018-10-18 #5", "100.3", "103.6", "3.29",  "TP",           "0"),
    ("2018-10-18 #6", "101.6", "104.2", "2.66",  "TP",           "53"),
]

FALHA_FFT = [
    ("2018-07-18 #1", "115.3", "—",     "—",     "FN",           "69"),
    ("2018-07-18 #2", "72.4",  "—",     "—",     "FN",           "264"),
    ("2018-07-18 EMR","115.6", "117.8", "2.17",  "TP",           "755"),
    ("2018-07-18 EMR2","113.1","115.9", "2.85",  "TP",           "0"),
    ("2018-07-30 EMR","122.2", "127.2", "5.07",  "TP",           "380"),
    ("2018-07-30 #1", "115.8", "130.4", "14.64", "TP",           "543"),
    ("2018-07-30 #2", "90.7",  "—",     "—",     "FN",           "220"),
    ("2018-07-30 EMR2","116.2","—",     "—",     "FN",           "0"),
    ("2018-07-30 EMR3","86.7", "89.5",  "2.80",  "TP",           "0"),
    ("2018-07-30 EMR4","132.5","154.5", "21.99", "TP",           "74"),
    ("2018-07-30 EMR5","89.4", "98.3",  "8.97",  "TP",           "258"),
    ("2018-09-11 #1", "102.6", "104.0", "1.41",  "TP",           "57"),
    ("2018-09-11 #2", "103.9", "—",     "—",     "FN",           "26"),
    ("2018-09-11 #3", "48.9",  "—",     "—",     "FN",           "35"),
    ("2018-10-05 #1", "48.2",  "48.2",  "0.04",  "TP",           "0"),
    ("2018-10-05 #2", "99.1",  "99.3",  "0.19",  "TP",           "26"),
    ("2018-10-05 #3", "75.2",  "—",     "—",     "FN",           "93"),
    ("2018-10-18 #1", "103.2", "105.9", "2.64",  "TP",           "320"),
    ("2018-10-18 #2", "110.1", "—",     "—",     "FN",           "579"),
    ("2018-10-18 #3", "99.4",  "99.7",  "0.27",  "TP",           "1455"),
    ("2018-10-18 #4", "97.3",  "100.1", "2.80",  "TP",           "75"),
    ("2018-10-18 #5", "100.3", "100.7", "0.39",  "TP",           "654"),
    ("2018-10-18 #6", "101.6", "103.8", "2.25",  "TP",           "693"),
]

# Classificação dos voos SEM falha
# Colunas: nome curto | FP (alarmes falsos) | resultado
NORMAL_ALL = [
    ("2018-07-18 sem-falha", "0",   "TN"),
    ("2018-07-30 sem-falha", "4",   "FP"),
    ("2018-09-11 #1 sem-falha", "72",  "FP"),
    ("2018-09-11 #2 sem-falha", "0",   "TN"),
    ("2018-09-11 #3 sem-falha", "162", "FP"),
    ("2018-10-05 #1 sem-falha", "525", "FP"),
    ("2018-10-05 #2 sem-falha", "64",  "FP"),
    ("2018-10-05 #3 sem-falha", "57",  "FP"),
    ("2018-10-05 #4 sem-falha", "0",   "TN"),
    ("2018-10-18 sem-falha",    "0",   "TN"),
]

NORMAL_FFT = [
    ("2018-07-18 sem-falha", "0",   "TN"),
    ("2018-07-30 sem-falha", "184", "FP"),
    ("2018-09-11 #1 sem-falha", "0",   "TN"),
    ("2018-09-11 #2 sem-falha", "278", "FP"),
    ("2018-09-11 #3 sem-falha", "0",   "TN"),
    ("2018-10-05 #1 sem-falha", "0",   "TN"),
    ("2018-10-05 #2 sem-falha", "0",   "TN"),
    ("2018-10-05 #3 sem-falha", "332", "FP"),
    ("2018-10-05 #4 sem-falha", "0",   "TN"),
    ("2018-10-18 sem-falha",    "0",   "TN"),
]

# ── Fontes ────────────────────────────────────────────────────────────────────
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_ITALIC  = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"

# Cores
AZUL       = (20, 60, 120)
CINZA_HD   = (80, 80, 80)
PRETO      = (30, 30, 30)
VERDE_BG   = (220, 240, 220)
VERDE_DARK = (34, 120, 34)
VERMELHO_BG= (255, 220, 220)
VERM_DARK  = (160, 30, 30)
AMARELO_BG = (255, 240, 200)
AMAR_DARK  = (140, 100, 10)
BRANCO     = (255, 255, 255)


class PDF(FPDF):
    def header(self):
        self.set_font("DejaVu", "B", 9)
        self.set_text_color(*CINZA_HD)
        self.cell(0, 7, TITULO, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def footer(self):
        self.set_y(-13)
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(150)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

    # ── helpers de estilo ──────────────────────────────────────────────────
    def titulo_secao(self, texto):
        self.set_font("DejaVu", "B", 12)
        self.set_text_color(*AZUL)
        self.cell(0, 7, texto, new_x="LMARGIN", new_y="NEXT")

    def subtitulo(self, texto):
        self.set_font("DejaVu", "B", 10)
        self.set_text_color(*CINZA_HD)
        self.cell(0, 6, texto, new_x="LMARGIN", new_y="NEXT")

    def cabecalho_tabela(self, cols, widths, h=7):
        self.set_font("DejaVu", "B", 8)
        self.set_fill_color(*AZUL)
        self.set_text_color(*BRANCO)
        for w, col in zip(widths, cols):
            self.cell(w, h, col, border=1, fill=True, align="C")
        self.ln()

    def linha_tabela(self, cells, widths, fill_color, text_color=None, h=6, aligns=None):
        self.set_font("DejaVu", "", 8)
        self.set_fill_color(*fill_color)
        self.set_text_color(*(text_color or PRETO))
        aligns = aligns or ["C"] * len(cells)
        for w, cel, al in zip(widths, cells, aligns):
            self.cell(w, h, str(cel), border=1, fill=True, align=al)
        self.ln()

    def matriz_2x2(self, tp, fn, fp, tn):
        """Desenha a matriz de confusão 2x2 com cores semânticas."""
        CW, CH = 42, 14   # cell width / height

        # Labels de eixo
        self.set_font("DejaVu", "B", 8)
        self.set_text_color(*CINZA_HD)
        x0 = self.get_x()
        y0 = self.get_y()

        # Cabeçalho de colunas
        self.set_xy(x0 + CW, y0)
        self.cell(CW, 6, "Alarme (positivo)", border=0, align="C")
        self.cell(CW, 6, "Silêncio (negativo)", border=0, align="C")
        self.ln()

        # Linha 1: Com falha
        y1 = self.get_y()
        self.set_xy(x0, y1)
        self.set_font("DejaVu", "B", 8)
        self.set_text_color(*CINZA_HD)
        self.cell(CW, CH, "Real: com falha", border=1, align="C")

        # TP
        self.set_fill_color(*VERDE_BG)
        self.set_text_color(*VERDE_DARK)
        self.set_font("DejaVu", "B", 10)
        self.cell(CW, CH, f"TP = {tp}", border=1, fill=True, align="C")

        # FN
        self.set_fill_color(*VERMELHO_BG)
        self.set_text_color(*VERM_DARK)
        self.cell(CW, CH, f"FN = {fn}", border=1, fill=True, align="C")
        self.ln()

        # Linha 2: Sem falha
        y2 = self.get_y()
        self.set_xy(x0, y2)
        self.set_font("DejaVu", "B", 8)
        self.set_text_color(*CINZA_HD)
        self.cell(CW, CH, "Real: sem falha", border=1, align="C")

        # FP
        self.set_fill_color(*VERMELHO_BG)
        self.set_text_color(*VERM_DARK)
        self.set_font("DejaVu", "B", 10)
        self.cell(CW, CH, f"FP = {fp}", border=1, fill=True, align="C")

        # TN
        self.set_fill_color(*VERDE_BG)
        self.set_text_color(*VERDE_DARK)
        self.cell(CW, CH, f"TN = {tn}", border=1, fill=True, align="C")
        self.ln()

        # Reset cor
        self.set_text_color(*PRETO)


# ═══════════════════════════════════════════════════════════════════════════════
pdf = PDF()
pdf.add_font("DejaVu", style="",  fname=FONT_REGULAR)
pdf.add_font("DejaVu", style="B", fname=FONT_BOLD)
pdf.add_font("DejaVu", style="I", fname=FONT_ITALIC)
pdf.set_margins(20, 20, 20)
pdf.add_page()

# ── Título ────────────────────────────────────────────────────────────────────
pdf.set_font("DejaVu", "B", 14)
pdf.set_text_color(*PRETO)
pdf.multi_cell(0, 9, TITULO, align="C")
pdf.ln(5)

# ── Parágrafos ────────────────────────────────────────────────────────────────
for secao, corpo in PARAGRAFOS:
    pdf.titulo_secao(secao)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(*PRETO)
    pdf.multi_cell(0, 6, corpo, align="J")
    pdf.ln(4)

# ── Tabela de métricas (janela a janela) ──────────────────────────────────────
pdf.ln(2)
pdf.titulo_secao("Métricas por modelo e classe (nível de janela)")
pdf.ln(2)
pdf.cabecalho_tabela(MET_CABECALHO, MET_COL_W, h=7)

cores_met = [(245, 249, 255), (230, 238, 252)]
for i, linha in enumerate(MET_LINHAS):
    pdf.linha_tabela(linha, MET_COL_W, fill_color=cores_met[i // 2], h=7)

# ══════════════════════════════════════════════════════════════════════════════
# MATRIZES DE CONFUSÃO (NÍVEL DE VOO)
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.set_font("DejaVu", "B", 13)
pdf.set_text_color(*AZUL)
pdf.cell(0, 9, "Matrizes de Confusão — nível de voo (notebooks 04)", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("DejaVu", "I", 9)
pdf.set_text_color(*CINZA_HD)
pdf.multi_cell(0, 5,
    "Cada voo é classificado como um único evento: se o modelo emitiu alarme durante "
    "o intervalo de falha, o voo é TP; se nunca emitiu (ou só emitiu antes da falha), é FN. "
    "Voos sem falha que geraram ao menos um alarme são FP; os demais, TN.",
    align="J")
pdf.ln(6)

# ─── Lado a lado: all_features e fft_features ────────────────────────────────
def secao_modelo(pdf, nome, tp, fn, fp, tn, lat_med, lat_median,
                 voos_falha, voos_normal):
    pdf.titulo_secao(f"Modelo: {nome}")
    pdf.ln(1)

    # Resumo numérico
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(*PRETO)
    pdf.cell(0, 5,
        f"Falhas detectadas: {tp}/23   |   Falhas perdidas: {fn}/23   |   "
        f"Lat. média: {lat_med} s   |   Lat. mediana: {lat_median} s",
        new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Matriz 2x2
    pdf.matriz_2x2(tp, fn, fp, tn)
    pdf.ln(6)

    # Tabela: voos com falha
    pdf.subtitulo("Voos COM falha (23 voos)")
    pdf.ln(1)
    col_f  = [38, 18, 20, 18, 26, 24]
    head_f = ["Voo (data/índice)", "Falha (s)", "Detect. (s)", "Lat. (s)",
              "Resultado", "FP antes"]
    pdf.cabecalho_tabela(head_f, col_f)

    cor_result = {
        "TP":       (VERDE_BG,    VERDE_DARK),
        "FN":       (VERMELHO_BG, VERM_DARK),
        "FP antes": (AMARELO_BG,  AMAR_DARK),
    }
    for row in voos_falha:
        nome_v, t_f, t_d, lat, res, fp_a = row
        bg, fg = cor_result.get(res, ((248, 248, 248), PRETO))
        aligns = ["L", "C", "C", "C", "C", "C"]
        pdf.linha_tabela([nome_v, t_f, t_d, lat, res, fp_a],
                         col_f, fill_color=bg, text_color=fg, aligns=aligns)
    pdf.ln(4)

    # Tabela: voos sem falha
    pdf.subtitulo("Voos SEM falha (10 voos)")
    pdf.ln(1)
    col_n  = [90, 40, 14]
    head_n = ["Voo (data/índice)", "Alarmes falsos (janelas)", "Resultado"]
    pdf.cabecalho_tabela(head_n, col_n)

    for row in voos_normal:
        nome_v, fp_n, res = row
        bg, fg = cor_result.get(res, (VERDE_BG, VERDE_DARK)) if res == "TN" else (VERMELHO_BG, VERM_DARK)
        pdf.linha_tabela([nome_v, fp_n, res], col_n,
                         fill_color=bg, text_color=fg,
                         aligns=["L", "C", "C"])
    pdf.ln(8)


secao_modelo(pdf,
    nome="all_features",
    tp=16, fn=7, fp=6, tn=4,
    lat_med="6.72", lat_median="3.15",
    voos_falha=FALHA_ALL, voos_normal=NORMAL_ALL,
)

pdf.add_page()

secao_modelo(pdf,
    nome="fft_features",
    tp=15, fn=8, fp=3, tn=7,
    lat_med="4.57", lat_median="2.64",
    voos_falha=FALHA_FFT, voos_normal=NORMAL_FFT,
)

# ── Salvar ────────────────────────────────────────────────────────────────────
saida = "aeroespacial-2/relatorio_resultados.pdf"
pdf.output(saida)
print(f"PDF gerado: {saida}")
