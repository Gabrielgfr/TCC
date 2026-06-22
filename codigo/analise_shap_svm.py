import pandas as pd
import numpy as np
import os
import json
import re
import shap
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SEMENTE_ALEATORIA = 42

CAMINHO_MODELO  = r"C:\Users\GabrielRodrigues\Desktop\TCC\codigo\melhor_svm.pkl"
CAMINHO_SHAP    = r"C:\Users\GabrielRodrigues\Desktop\TCC\codigo\shap_values.npy"
CAMINHO_AMOSTRA = r"C:\Users\GabrielRodrigues\Desktop\TCC\codigo\amostra_teste_shap.pkl"
CAMINHO_GRAFICO = r"C:\Users\GabrielRodrigues\Desktop\TCC\figuras\shap_top15_qes.png"
CAMINHO_JSON    = r"C:\Users\GabrielRodrigues\Desktop\TCC\analise_completa_shap.json"

DIMENSOES_ENADE = {
    'Organizacao_Didatico_Pedagogica': [27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,47,48,54,55,56,57,58],
    'Infraestrutura_e_Instalacoes':    [59,60,61,62,63,64,65,66,67,68],
    'Oportunidades_de_Formacao':       [43,44,45,46,49,50,51,52,53]
}

def extrair_numero_qe(nome_coluna):
    match = re.search(r'\d+', nome_coluna)
    return int(match.group()) if match else None

def construir_mapa_qe_dimensao():
    mapa = {}
    for dim, lista_q in DIMENSOES_ENADE.items():
        for numero in lista_q:
            mapa[numero] = dim
    return mapa

def gerar_grafico_shap(importancia_shap, caminho_saida, top_n=15):
    mapa_qe_dim = construir_mapa_qe_dimensao()
    top_qes     = importancia_shap.head(top_n)

    paleta = {
        'Organizacao_Didatico_Pedagogica': '#2c7bb6',
        'Oportunidades_de_Formacao':       '#d7191c',
        'Infraestrutura_e_Instalacoes':    '#1a9641',
        'desconhecida':                    '#969696'
    }
    rotulos_dimensao = {
        'Organizacao_Didatico_Pedagogica': 'Organização Didático-Pedagógica',
        'Oportunidades_de_Formacao':       'Oportunidades de Formação',
        'Infraestrutura_e_Instalacoes':    'Infraestrutura e Instalações',
        'desconhecida':                    'Não classificada'
    }

    cores = []
    for qe in top_qes.index:
        numero = extrair_numero_qe(qe)
        dim    = mapa_qe_dim.get(numero, 'desconhecida')
        cores.append(paleta[dim])

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos   = range(len(top_qes))

    ax.barh(list(y_pos), top_qes.values, color=cores,
            edgecolor='white', linewidth=0.5, height=0.65)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top_qes.index, fontsize=10)
    ax.invert_yaxis()

    for i, valor in enumerate(top_qes.values):
        ax.text(valor + 0.01, i, f'{valor:.4f}',
                va='center', ha='left', fontsize=8.5, color='#333333')

    ax.set_xlabel('Importância SHAP média absoluta (|SHAP|)', fontsize=11)
    ax.set_title(f'Top {top_n} variáveis por importância SHAP — SVM (kernel RBF)',
                 fontsize=12, pad=12)
    ax.set_xlim(0, top_qes.values.max() * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    dims_presentes = set()
    for qe in top_qes.index:
        numero = extrair_numero_qe(qe)
        dims_presentes.add(mapa_qe_dim.get(numero, 'desconhecida'))

    patches = [mpatches.Patch(color=paleta[d], label=rotulos_dimensao[d])
               for d in paleta if d in dims_presentes]
    ax.legend(handles=patches, loc='lower right', fontsize=9,
              framealpha=0.85, title='Dimensão ENADE', title_fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico SHAP salvo em: {caminho_saida}")


def executar_shap():

    print("Carregando pipeline SVM...")
    pipeline_svm = joblib.load(CAMINHO_MODELO)

    scaler    = pipeline_svm.named_steps['padronizacao'] 
    modelo_svm = pipeline_svm.named_steps['modelo']        # SVR já treinado

    if os.path.exists(CAMINHO_SHAP) and os.path.exists(CAMINHO_AMOSTRA):
        print("SHAP values encontrados em disco — carregando...")
        shap_values   = np.load(CAMINHO_SHAP)
        amostra_bruta = joblib.load(CAMINHO_AMOSTRA)  # DataFrame bruto, escala original
    else:
        print("amostra_teste_shap.pkl ou shap_values.npy não encontrados.")
        print("Execute analise_tap.py primeiro para gerar os arquivos necessários.")
        return

    amostra_transformada = scaler.transform(amostra_bruta)

    amostra_transformada_df = pd.DataFrame(
        amostra_transformada,
        columns=amostra_bruta.columns
    )
    # =========================================================================

    print("Calculando SHAP values...")

    background = shap.kmeans(amostra_transformada_df, 10)

    explainer   = shap.KernelExplainer(modelo_svm.predict, background)
    shap_values = explainer.shap_values(amostra_transformada_df, silent=True)

    np.save(CAMINHO_SHAP, shap_values)
    print("SHAP values recalculados e salvos em disco.")

    # IMPORTÂNCIA MÉDIA ABSOLUTA POR QE
    importancia_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=amostra_bruta.columns       # nomes originais das QEs
    ).sort_values(ascending=False)

    # AGRUPA POR DIMENSÃO
    importancia_shap_dimensoes = {dim: 0.0 for dim in DIMENSOES_ENADE}
    for coluna, valor in importancia_shap.items():
        numero_q = extrair_numero_qe(coluna)
        if numero_q is None:
            continue
        for dim, lista_q in DIMENSOES_ENADE.items():
            if numero_q in lista_q:
                importancia_shap_dimensoes[dim] += valor
                break

    gerar_grafico_shap(importancia_shap, CAMINHO_GRAFICO, top_n=15)

    # SALVA JSON
    if os.path.exists(CAMINHO_JSON):
        with open(CAMINHO_JSON, 'r') as f:
            saida = json.load(f)
    else:
        saida = {}

    saida['importancia_shap_qes_svm']       = importancia_shap.to_dict()
    saida['importancia_shap_dimensoes_svm'] = importancia_shap_dimensoes

    with open(CAMINHO_JSON, 'w') as f:
        json.dump(saida, f, indent=4)

    print("\n===== SHAP — TOP 10 QEs mais importantes (SVM) =====")
    print(importancia_shap.head(10).to_string())

    print("\n===== SHAP — Importância por Dimensão (SVM) =====")
    for dim, imp in importancia_shap_dimensoes.items():
        print(f"  {dim}: {imp:.4f}")

    print("\nAnálise SHAP concluída! Resultados adicionados ao JSON.")


if __name__ == "__main__":
    executar_shap()