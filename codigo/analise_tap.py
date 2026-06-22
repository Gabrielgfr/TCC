import pandas as pd
import numpy as np
import os
import json
import joblib
import re

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEMENTE_ALEATORIA = 42

CAMINHO_ARQUIVO = r"C:\Users\GabrielRodrigues\Desktop\TCC\dataset\dataset_COMP_VF.xlsx"

DIMENSOES_ENADE = {
    'Organizacao_Didatico_Pedagogica': [27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,47,48,54,55,56,57,58],
    'Infraestrutura_e_Instalacoes': [59,60,61,62,63,64,65,66,67,68],
    'Oportunidades_de_Formacao': [43,44,45,46,49,50,51,52,53]
}

def extrair_numero_qe(nome_coluna):
    match = re.search(r'\d+', nome_coluna)
    return int(match.group()) if match else None

def executar_analise():
    print(f"Carregando dados de: {CAMINHO_ARQUIVO}")

    dados = pd.read_excel(CAMINHO_ARQUIVO)

    # SELEÇÃO DE VARIÁVEIS
    colunas_qe = [col for col in dados.columns if col.startswith('QE_')]
    X = dados[colunas_qe].copy()
    y = dados['TAP']

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEMENTE_ALEATORIA
    )

    # TRATAMENTO DE DADOS AUSENTES
    moda_treino = X_treino.mode().iloc[0]
    X_treino = X_treino.fillna(moda_treino)
    X_teste = X_teste.fillna(moda_treino)

    modelos = {
        'SVM': {
            'pipeline': Pipeline([
                ('padronizacao', StandardScaler()),
                ('modelo', SVR())
            ]),
            'parametros': {
                'modelo__C': [1, 10, 50],
                'modelo__epsilon': [0.05, 0.1, 0.2],
                'modelo__kernel': ['rbf'],
                'modelo__gamma': ['scale', 'auto']
            }
        },
        'MLP': {
            'pipeline': Pipeline([
                ('padronizacao', StandardScaler()),
                ('modelo', MLPRegressor(
                    max_iter=1000,
                    early_stopping=True,
                    random_state=SEMENTE_ALEATORIA
                ))
            ]),
            'parametros': {
                'modelo__hidden_layer_sizes': [(50,), (100,), (100,50)],
                'modelo__alpha': [0.0001, 0.001, 0.01],
                'modelo__learning_rate_init': [0.001, 0.0005]
            }
        },
        'Ridge': {
            'pipeline': Pipeline([
                ('padronizacao', StandardScaler()),
                ('modelo', Ridge())
            ]),
            'parametros': {
                'modelo__alpha': [0.1, 1.0, 10.0, 50.0]
            }
        },
        'Random Forest': {
            'pipeline': Pipeline([
                ('modelo', RandomForestRegressor(
                    random_state=SEMENTE_ALEATORIA,
                    n_jobs=-1
                ))
            ]),
            'parametros': {
                'modelo__n_estimators': [100, 200],
                'modelo__max_depth': [10, 20, None],
                'modelo__min_samples_split': [2, 5],
                'modelo__min_samples_leaf': [1, 2]
            }
        },
        'Decision Tree': {
            'pipeline': Pipeline([
                ('modelo', DecisionTreeRegressor(
                    random_state=SEMENTE_ALEATORIA
                ))
            ]),
            'parametros': {
                'modelo__max_depth': [5, 10, 20],
                'modelo__min_samples_split': [2, 5, 10],
                'modelo__min_samples_leaf': [1, 2, 4]
            }
        }
    }

    # EXECUÇÃO DO GRID SEARCH
    resultados_modelos = {}

    for nome_modelo, config in modelos.items():

        print(f"\nExecutando GridSearch para {nome_modelo}...")

        grid = GridSearchCV(
            estimator=config['pipeline'],
            param_grid=config['parametros'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0,
            return_train_score=False
        )

        grid.fit(X_treino, y_treino)

        y_pred = grid.best_estimator_.predict(X_teste)

        print("\nComparação real vs previsto (primeiras 10 linhas):")
        print(pd.DataFrame({
            'TAP_real': y_teste.values,
            'TAP_previsto': y_pred
        }).head(10))

        # MÉTRICAS 
        mse_teste  = mean_squared_error(y_teste, y_pred)
        rmse_teste = np.sqrt(mse_teste)
        mae_teste  = mean_absolute_error(y_teste, y_pred)
        r2_teste   = r2_score(y_teste, y_pred)

        rmse_validacao = np.sqrt(-grid.best_score_)
        idx = grid.best_index_
        rmse_por_fold = np.array([
            np.sqrt(-grid.cv_results_[f'split{k}_test_score'][idx])
            for k in range(5)
        ])
        rmse_std_cv = rmse_por_fold.std()

        resultados_modelos[nome_modelo] = {
            'melhores_parametros': grid.best_params_,
            'rmse_validacao_cv':   rmse_validacao,
            'rmse_std_cv':         rmse_std_cv,
            # =================================================================
            'mse_teste':  mse_teste,
            'rmse_teste': rmse_teste,
            'mae_teste':  mae_teste,
            'r2_teste':   r2_teste,
            '_best_estimator': grid.best_estimator_
        }

    # IMPORTÂNCIA DAS DIMENSÕES (Random Forest)
    print("\nCalculando importância das dimensões (Random Forest)...")

    melhor_rf = resultados_modelos['Random Forest']['_best_estimator']
    importancias = melhor_rf.named_steps['modelo'].feature_importances_

    importancia_dimensoes = {dim: 0.0 for dim in DIMENSOES_ENADE}
    qes_sem_dimensao = []

    for i, coluna in enumerate(X_treino.columns):
        numero_q = extrair_numero_qe(coluna)

        if numero_q is None:
            continue

        encontrou = False
        for dim, lista_q in DIMENSOES_ENADE.items():
            if numero_q in lista_q:
                importancia_dimensoes[dim] += importancias[i]
                encontrou = True
                break

        if not encontrou:
            qes_sem_dimensao.append(coluna)

    if qes_sem_dimensao:
        print(f"\n QEs sem dimensão mapeada (importância ignorada): {qes_sem_dimensao}")

    caminho_modelo = r'C:\Users\GabrielRodrigues\Desktop\TCC\codigo\melhor_svm.pkl'
    joblib.dump(resultados_modelos['SVM']['_best_estimator'], caminho_modelo)
    print(f"\nModelo SVM salvo em: {caminho_modelo}")

    for nome_modelo in resultados_modelos:
        resultados_modelos[nome_modelo].pop('_best_estimator', None)

    # SALVAR RESULTADOS
    saida = {
        'modelos': resultados_modelos,
        'importancia_dimensoes_rf': importancia_dimensoes
    }

    with open('analise_completa_tap.json', 'w') as arquivo:
        json.dump(saida, arquivo, indent=4)

    # RESULTADOS FINAIS
    print("\n===== RESULTADOS FINAIS =====")
    for modelo, resultado in resultados_modelos.items():
        print(f"\n{modelo}:")
        print(f"  RMSE (validação CV): {resultado['rmse_validacao_cv']:.4f} ± {resultado['rmse_std_cv']:.4f}")
        print(f"  MSE  (teste):        {resultado['mse_teste']:.4f}")
        print(f"  RMSE (teste):        {resultado['rmse_teste']:.4f}")
        print(f"  MAE  (teste):        {resultado['mae_teste']:.4f}")
        print(f"  R²   (teste):        {resultado['r2_teste']:.4f}")
        print(f"  Melhores parâmetros: {resultado['melhores_parametros']}")

    print("\n===== IMPORTÂNCIA DAS DIMENSÕES (Random Forest) =====")
    for dim, imp in importancia_dimensoes.items():
        print(f"  {dim}: {imp:.4f}")

    print("\nAnálise concluída! Arquivo salvo como 'analise_completa_tap.json'.")


if __name__ == "__main__":
    executar_analise()