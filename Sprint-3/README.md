# Sprint 3 — Modelagem de Machine Learning

## Predição de Risco de Sinistro em Equipamentos Agrícolas

Este projeto corresponde ao terceiro entregável da disciplina de Machine Learning e tem como objetivo desenvolver e avaliar modelos capazes de classificar a ocorrência de sinistros em equipamentos agrícolas.

## Etapas desenvolvidas

- Preparação dos dados
- One-Hot Encoding das variáveis categóricas
- Padronização das variáveis numéricas com StandardScaler
- Divisão dos dados em treino e teste utilizando holdout
- Treinamento de Regressão Logística
- Treinamento de Random Forest
- Avaliação utilizando Accuracy, Precision, Recall, F1-score e AUC
- Ajuste de hiperparâmetros com GridSearchCV
- Comparação entre os modelos
- Análise da importância das variáveis

## Resultados

O modelo com melhor desempenho foi o **Random Forest otimizado**, que apresentou:

| Métrica | Resultado |
|---|---:|
| Accuracy | 94,75% |
| Precision | 95,86% |
| Recall | 92,05% |
| F1-score | 93,91% |
| AUC | 94,18% |

A análise de importância das variáveis indicou a **velocidade média** como principal variável preditiva no conjunto de dados utilizado.

> Os resultados foram obtidos a partir de um dataset simulado e devem ser interpretados dentro do contexto experimental do projeto.

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

## Arquivo principal

`SPRINT3_Machine_Learning.ipynb`
