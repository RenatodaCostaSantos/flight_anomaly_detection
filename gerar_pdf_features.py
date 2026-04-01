"""Gera PDF explicando as features de engenharia do notebook 02."""
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "Feature Engineering - Deteccao de Falha de Motor (CarbonZ UAV)", align="C")
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")

    def titulo_secao(self, texto):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 80, 160)
        self.set_fill_color(235, 242, 255)
        self.cell(0, 9, texto, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def subtitulo(self, texto):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 7, texto, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def corpo(self, texto, indent=0):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(10 + indent)
        self.multi_cell(190 - indent, 6, texto)
        self.ln(1)

    def formula(self, texto):
        self.set_font("Courier", "", 10)
        self.set_fill_color(248, 248, 248)
        self.set_draw_color(200, 200, 200)
        self.set_x(20)
        self.multi_cell(170, 6, texto, border=1, fill=True)
        self.ln(2)

    def bullet(self, label, descricao, indent=15):
        self.set_font("Helvetica", "B", 10)
        self.set_x(10 + indent)
        # Calcula largura do label
        label_w = self.get_string_width(label + "  ")
        self.cell(label_w, 6, label)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(190 - indent - label_w, 6, descricao)
        self.ln(0.5)

    def tabela_features(self, rows):
        col_w = [55, 30, 105]
        headers = ["Feature", "Tipo", "Por que e util"]
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(50, 90, 170)
        self.set_text_color(255, 255, 255)
        for i, (h, w) in enumerate(zip(headers, col_w)):
            self.cell(w, 7, h, border=1, fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)
        for idx, (feat, tipo, motivo) in enumerate(rows):
            fill = idx % 2 == 0
            self.set_fill_color(245, 248, 255) if fill else self.set_fill_color(255, 255, 255)
            self.set_font("Courier", "", 8)
            self.cell(col_w[0], 6, feat, border=1, fill=fill)
            self.set_font("Helvetica", "", 8)
            self.cell(col_w[1], 6, tipo, border=1, fill=fill)
            self.multi_cell(col_w[2], 6, motivo, border=1, fill=fill)
        self.ln(3)

    def destaque(self, texto):
        self.set_font("Helvetica", "I", 10)
        self.set_fill_color(255, 252, 220)
        self.set_draw_color(220, 180, 0)
        self.set_x(15)
        self.multi_cell(180, 6, texto, border=1, fill=True)
        self.set_draw_color(0, 0, 0)
        self.ln(2)


# =============================================================================
pdf = PDF()
pdf.set_margins(10, 15, 10)
pdf.set_auto_page_break(auto=True, margin=18)
pdf.add_page()

# ---- TITULO PRINCIPAL ----
pdf.set_font("Helvetica", "B", 18)
pdf.set_text_color(20, 60, 140)
pdf.cell(0, 12, "Feature Engineering", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 8, "Deteccao de Falha de Motor em UAV (CarbonZ)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 7, "Notebook 02 - Pipeline feature_engineering (Kedro)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)

pdf.set_draw_color(50, 90, 170)
pdf.set_line_width(0.8)
pdf.line(10, pdf.get_y(), 200, pdf.get_y())
pdf.set_line_width(0.2)
pdf.set_draw_color(0, 0, 0)
pdf.ln(6)
pdf.set_text_color(0, 0, 0)

# ---- CONTEXTO ----
pdf.titulo_secao("Contexto e Objetivo")
pdf.corpo(
    "O projeto usa dados de telemetria de voo de um UAV de asa fixa (CarbonZ) coletados a ~100 Hz. "
    "O objetivo e detectar automaticamente a falha do motor a partir dos sensores de bordo, "
    "sem acesso direto ao estado do motor."
)
pdf.corpo(
    "A pipeline feature_engineering transforma os dados brutos preparados (prepared_flights) "
    "em um conjunto enriquecido de features fisicas e temporais (feature_engineered_flights), "
    "salvo em data/04_feature/. Estas features sao o input para o modelo de deteccao de anomalia "
    "(Isolation Forest)."
)

pdf.destaque(
    "Intuicao central: altitude e energia se comportam de formas complementares na falha.\n"
    "A altitude cai imediatamente; a energia especifica permanece estavel por 2-3 s enquanto "
    "o aviao troca altitude por velocidade - e depois cai de forma monotonica e sustentada, "
    "que e o sinal mais limpo para o modelo."
)

# ---- TABELA RESUMO ----
pdf.titulo_secao("Resumo das Features Criadas")
rows = [
    ("energy_specific",      "Fisica",       "Apos ~2-3 s da falha, cai de forma monotonica - sinal mais limpo que altitude"),
    ("energy_rate",          "Derivada",      "Ruidoso como sinal instantaneo; util apenas suavizado (ver rolling slope)"),
    ("speed_horizontal",     "Cinematica",    "Velocidade no plano horizontal; cai sem thrust para compensar o arrasto"),
    ("speed_total",          "Cinematica",    "Modulo da velocidade 3D; complementa altitude no calculo de energia"),
    ("glide_ratio",          "Aerodinamica",  "Converge para L/D do CarbonZ (~8) em voo de planeio puro apos falha"),
    ("control_effort",       "Controle",      "Soma dos quadrados dos erros de rastreamento; o autopilot luta mais na falha"),
    ("*_mean_W",             "Temporal",      "Media suavizada sobre janela W; filtra picos transitorios"),
    ("*_std_W",              "Temporal",      "Variabilidade local; picos indicam transicao de regimes"),
    ("*_slope_W",            "Temporal",      "Tendencia media [unidade/s]; substituto confiavel de derivadas instantaneas"),
    ("fft_peak_power_*_W",   "Espectral",     "Potencia do componente de frequencia dominante; detecta oscilacoes periodicas"),
    ("fft_entropy_*_W",      "Espectral",     "Entropia espectral de Shannon [0-1]; 0=senoide puro, 1=ruido branco"),
    ("fft_high_ratio_*_W",   "Espectral",     "Fracao de potencia em alta frequencia; falha injeta energia em altas frequencias"),
]
pdf.tabela_features(rows)

# ====== FEATURE 1 ======
pdf.add_page()
pdf.titulo_secao("Feature 1 - Energia Mecanica Especifica (energy_specific)")
pdf.formula("energy_specific = alt_global + (speed_total^2) / (2 * g)")

pdf.corpo(
    "Em voo propulsionado, o motor injeta energia para compensar o arrasto. Sem motor, "
    "o arrasto dissipa energia sem reposicao - o sistema perde energia total de forma sustentada."
)

pdf.subtitulo("Tres fases apos a falha:")
pdf.bullet("Fase 1 (imediata):",
    "A altitude (alt_global) comeca a cair imediatamente. O autopilot inclina o nariz "
    "para baixo para manter a velocidade de voo e evitar o stall. A altitude nao e um "
    "indicador atrasado - e contemporaneo a falha.")
pdf.bullet("Fase 2 (~0 a 2-3 s):",
    "A energy_specific permanece relativamente estavel. O aviao esta trocando energia "
    "potencial por energia cinetica (PE -> KE), mantendo a soma h + v^2/2g aproximadamente "
    "constante. A energia nao e dissipada ainda, apenas redistribuida entre as formas.")
pdf.bullet("Fase 3 (apos ~2-3 s):",
    "O arrasto acumula dissipacao suficiente para superar a troca PE<->KE. A energy_specific "
    "comeca a cair de forma quase monotonica. Este e o sinal mais limpo e sustentado, "
    "menos sujeito a oscilacoes por manobras normais do que a altitude isolada.")

pdf.subtitulo("energy_rate (derivada instantanea):")
pdf.corpo(
    "A taxa de variacao instantanea da energia e RUIDOSA: apresenta picos positivos e negativos "
    "tanto no voo normal quanto durante a falha. Individualmente nao e um bom discriminador. "
    "O sinal util emerge apenas quando suavizado em janelas temporais - ver Feature 4 "
    "(energy_specific_slope_W), que e o substituto confiavel."
)

pdf.destaque(
    "Por que preferir energy_specific sobre altitude pura?\n"
    "Altitude pode oscilar por manobras normais (curvas, subidas programadas). A energia "
    "especifica combina altitude e velocidade em um escalar fisico que reflete o estado "
    "energetico total da aeronave - nao pode ser mantido sem source de energia (motor)."
)

# ====== FEATURE 2 ======
pdf.titulo_secao("Feature 2 - Velocidades e Razao de Planeio")

pdf.subtitulo("speed_horizontal e speed_total:")
pdf.corpo(
    "As velocidades horizontal e total caem de forma consistente logo apos a falha. "
    "Sem thrust, o arrasto desacelera o aviao sem compensacao. Sao importantes tanto "
    "como features diretas quanto como componentes do calculo de energia especifica."
)

pdf.subtitulo("vel_z_meas (velocidade vertical):")
pdf.corpo(
    "Positivo e oscilante apos a falha. A convencao no dataset e NED (North-East-Down): "
    "positivo = descendo. Antes da falha, oscila em torno de zero (altitude mantida pelo "
    "autopilot). Apos a falha, fica persistentemente positivo: o aviao desce sem recuperacao."
)

pdf.subtitulo("glide_ratio (razao de planeio):")
pdf.formula("glide_ratio = speed_horizontal / |vel_z_meas|  (quando vel_z_meas > 0)")
pdf.corpo(
    "Representa quantos metros o aviao avanca horizontalmente para cada metro que desce. "
    "Para o CarbonZ em planeio puro, o valor teorico e ~8 (L/D aerodinamico)."
)
pdf.corpo(
    "Comportamento caracteristico: durante o voo normal, oscila bastante. Ha episodios "
    "curtos de constancia - momentos de voo nivelado em velocidade estavel, onde o motor "
    "e a aerodinamica estao em equilibrio temporario."
)
pdf.corpo(
    "O que muda apos a falha NAO e apenas a constancia, mas a PERMANENCIA dela: o aviao "
    "entra em planeio puro e o glide_ratio converge para ~8 e nunca mais oscila pelo "
    "restante do voo."
)

pdf.destaque(
    "O discriminador correto nao e 'esta constante agora?' mas sim 'esta constante de "
    "forma SUSTENTADA?'. Exatamente o que glide_ratio_std_W com janela longa captura: "
    "seu valor cai para proximo de zero apos a falha e permanece assim."
)

# ====== FEATURE 3 ======
pdf.add_page()
pdf.titulo_secao("Feature 3 - Esforco de Controle (control_effort)")
pdf.formula("control_effort = err_pitch^2 + err_roll^2 + err_vel_z^2")

pdf.corpo(
    "Quando o motor falha, o autopilot continua comandando valores normais enquanto o "
    "aviao diverge do setpoint. Todos os erros de rastreamento crescem simultaneamente. "
    "Somar seus quadrados cria um escalar composto que amplifica o sinal de falha."
)

pdf.subtitulo("Componentes:")
pdf.bullet("err_pitch:",
    "Erro entre pitch desejado (setpoint do autopilot) e pitch real medido. "
    "Cresce pois o aviao nao consegue mais manter atitude com a perda de thrust.")
pdf.bullet("err_roll:",
    "Erro de rolagem. Tambem cresce na falha, pois o controle lateral fica comprometido.")
pdf.bullet("err_vel_z:",
    "Erro na velocidade vertical. O aviao passa a descer quando o autopilot tenta "
    "manter altitude - o maior contribuidor para o sinal composto.")

pdf.corpo(
    "O uso de quadrados (ao inves de valores absolutos) tem dois efeitos: "
    "(1) penaliza erros grandes desproporcionalmente, amplificando o sinal de falha; "
    "(2) garante que o escalar seja sempre positivo, independente do sinal dos erros individuais."
)

pdf.destaque(
    "O control_effort reflete a perspectiva do sistema de controle: o autopilot 'sabe' "
    "que algo esta errado porque esta se esforçando muito mais para corrigir desvios. "
    "Esta e uma feature de segunda ordem - captura a resposta do sistema a falha, "
    "nao a fisica da falha diretamente."
)

# ====== FEATURE 4 ======
pdf.titulo_secao("Feature 4 - Estatisticas de Janela Temporal (Rolling Features)")
pdf.corpo(
    "Valores instantaneos capturam o estado agora. Estatisticas de janela capturam TENDENCIAS. "
    "Para cada feature relevante, tres estatisticas sao calculadas sobre janelas deslizantes "
    "de W amostras (W in {50, 100, 200}, equivalentes a ~0.5 s, ~1 s e ~2 s a 100 Hz):"
)

pdf.bullet("*_mean_W:",
    "Media movel sobre W amostras. Fornece nivel suavizado do sinal, filtrando picos "
    "de alta frequencia e ruido de medicao.")
pdf.bullet("*_std_W:",
    "Desvio padrao movel. Captura variabilidade local. Picos indicam transicao de "
    "regime; queda persistente para zero (como glide_ratio_std) indica regime estavel.")
pdf.bullet("*_slope_W:",
    "Taxa de variacao media [unidade/s] sobre a janela, estimada por regressao linear. "
    "Filtra ruido transitorios e revela TENDENCIAS SUSTENTADAS.")

pdf.subtitulo("Por que tres tamanhos de janela?")
pdf.corpo(
    "Janelas curtas (~0.5 s): mais responsivas, capturam mudancas rapidas, mas mais "
    "sujeitas a falsos positivos por transitórios normais.\n"
    "Janelas medias (~1 s): balanco entre responsividade e robustez.\n"
    "Janelas longas (~2 s): mais suaves, capturam apenas tendencias realmente sustentadas. "
    "Ideal para discriminar planeio continuo de manobras momentaneas."
)

pdf.subtitulo("energy_specific_slope_W - o sinal mais limpo:")
pdf.corpo(
    "O slope de energy_specific sobre uma janela e o substituto mais confiavel para "
    "energy_rate (derivada instantanea). Ao calcular a variacao sobre uma janela inteira "
    "em vez do diferencial instantaneo, ele filtra picos transitorios e revela a tendencia "
    "sustentada negativa que caracteriza a falha do motor - distinguindo-a de manobras "
    "normais ou rajadas de vento que afetam apenas alguns frames."
)

pdf.destaque(
    "Features rolling sao fundamentais para o Isolation Forest: o modelo opera no espaco "
    "de features em cada instante de tempo. Sem features temporais, o modelo veria apenas "
    "o estado instantaneo e nao conseguiria distinguir uma manobra brusca passageira de uma "
    "falha real. Com rolling features, o modelo 've' a historia recente do voo."
)

# ====== FEATURE 5 ======
pdf.add_page()
pdf.titulo_secao("Feature 5 - Features Espectrais via FFT (fft_*)")
pdf.corpo(
    "As features de tempo capturam O QUANTO um sinal muda. O dominio de frequencia captura "
    "COMO ele oscila - um tipo de informacao invisivel para o Isolation Forest operando "
    "apenas com features de magnitude."
)

pdf.subtitulo("Por que FFT ajuda na deteccao de falha de motor?")
pdf.corpo(
    "Uma falha de motor tem precursores espectrais antes de ser detectavel no tempo:"
)
pdf.bullet("-",
    "O motor degradando cria oscilacoes periodicas na velocidade do ar (aspd_meas) por "
    "ripple de thrust e desequilibrio da helice - visiveis no espectro de frequencias "
    "antes de aparecerem na magnitude.")
pdf.bullet("-",
    "A energia especifica, que em cruzeiro normal e quasi-constante (espectro de baixa "
    "frequencia dominante), comeca a oscilar de forma caotica antes da queda monotonica.")
pdf.bullet("-",
    "Manobras normais tem energia concentrada em baixas frequencias. Eventos de falha "
    "injetam energia em altas frequencias, alterando o perfil espectral.")

pdf.subtitulo("Tres descritores por sinal e janela:")
pdf.bullet("fft_peak_power_{sinal}_{W}:",
    "Magnitude do componente de frequencia dominante (excluindo DC). Um pico alto indica "
    "que ha uma oscilacao forte e periodica - tipica de motor com ripple ou helice "
    "desequilibrada.")
pdf.bullet("fft_entropy_{sinal}_{W}:",
    "Entropia espectral de Shannon normalizada [0-1]. Valor 0 = sinal puro senoide "
    "(toda potencia em uma frequencia). Valor 1 = ruido branco (potencia igual em todas "
    "as frequencias). Falha tende a aumentar a entropia (sinal mais caotico).")
pdf.bullet("fft_high_ratio_{sinal}_{W}:",
    "Fracao da potencia espectral na metade superior da banda. Em voo normal, predominam "
    "baixas frequencias. Na falha, alta frequencia aumenta por vibracao e comportamento "
    "caotico.")

pdf.subtitulo("Janelas FFT e sinais alvo:")
pdf.corpo(
    "As janelas FFT usadas sao {500, 1000, 2000} amostras (~5 s, ~10 s, ~20 s a 100 Hz). "
    "Janelas maiores dao melhor resolucao de frequencia mas mais latencia. Os sinais alvo "
    "sao aspd_meas (velocidade do ar) e energy_specific - os mais informativos."
)
pdf.corpo(
    "Implementacao: usa sliding_window_view + batch rfft vetorizado (NumPy), sem loop "
    "Python por amostra - eficiente para dados a 100 Hz."
)

pdf.destaque(
    "Features espectrais sao especialmente uteis como complemento a features temporais: "
    "capturam um 'espaco ortogonal' de informacao. Um modelo que combina ambas tem acesso "
    "a tanto a magnitude quanto a estrutura de frequencia dos sinais - tornando-o mais "
    "robusto a diferentes modos de falha."
)

# ====== JANELAS E PARAMETROS ======
pdf.add_page()
pdf.titulo_secao("Parametros da Pipeline")
pdf.corpo(
    "Os parametros de engenharia de features sao configurados em conf/base/parameters.yml:"
)
pdf.formula(
    "feature_engineering:\n"
    "  rolling_windows: [50, 100, 200]   # amostras: ~0.5 s, ~1 s, ~2 s a 100 Hz\n"
    "  fft_windows: [500, 1000, 2000]    # amostras: ~5 s, ~10 s, ~20 s a 100 Hz"
)
pdf.corpo(
    "Para dados com frequencia de amostragem diferente ou com comportamentos de falha "
    "mais lentos, aumentar as janelas pode capturar melhor as tendencias sustentadas."
)

# ====== CONTAGEM ======
pdf.titulo_secao("Volume de Features Geradas")
pdf.corpo(
    "A partir das features base, a pipeline gera um conjunto expandido:"
)
linhas = [
    ("energy_specific, energy_rate",               "2 features fisicas"),
    ("speed_horizontal, speed_total",               "2 features cinematicas"),
    ("glide_ratio",                                 "1 feature aerodinamica"),
    ("control_effort",                              "1 feature de controle"),
    ("Rolling (mean/std/slope) x 3 janelas",        "~30-45 features temporais"),
    ("FFT (peak_power/entropy/high_ratio) x 3 jan.","~18-27 features espectrais"),
]
for feat, desc in linhas:
    pdf.bullet("-", f"{feat}: {desc}")

pdf.ln(2)
pdf.corpo(
    "O shape exato depende das features alvo selecionadas para rolling e FFT, "
    "configuradas nos parametros ROLLING_TARGET_FEATURES e FFT_TARGET_FEATURES "
    "nos nodes da pipeline."
)

# ====== FLUXO KEDRO ======
pdf.titulo_secao("Fluxo na Pipeline Kedro")
pdf.formula(
    "prepared_flights\n"
    "       |\n"
    "       v  engineer_features_for_all_flights_node\n"
    "       |\n"
    "feature_engineered_flights  -->  data/04_feature/\n"
    "       |\n"
    "       v\n"
    "model_training"
)
pdf.corpo("Para executar apenas esta etapa:")
pdf.formula("kedro run --pipeline=feature_engineering")

# ====== PROXIMOS PASSOS ======
pdf.titulo_secao("Proximos Passos")
pdf.bullet("1.", "Avaliar importancia das features com permutation importance ou SHAP no modelo treinado.")
pdf.bullet("2.", "Experimentar janelas rolling mais longas (ex: 500 amostras = ~5 s) para capturar tendencias mais suaves.")
pdf.bullet("3.", "Adicionar features de correlacao cruzada entre pares de sinais (ex: altitude x velocidade).")
pdf.bullet("4.", "Avaliar reducao de dimensionalidade (PCA) antes do Isolation Forest se o numero de features crescer muito.")

# ====== SALVA ======
output_path = "/home/kunumi/projetos/capacitacao-avancada/features_explicacao.pdf"
pdf.output(output_path)
print(f"PDF salvo em: {output_path}")
