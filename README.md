# MLP para previsão de preço de café arábica
Aplicação do modelo Multilayer Perceptron (MLP) para a previsão diária de preço (em reais) de café arábica no Brasil, considerando dados históricos do CEPEA no período de 01/01/2015 à 01/01/2025.

# Etapas do projeto
1. Processamento dos dados: Adequações gerais dos dados para treinamento e teste do modelo.
2. Engenharia de atributos: Aplicação técnicas de normalização das variáveis, divisão do conjunto de dados para treinamento (80%) e teste (20%) e remodelação dos dados para o formato 2D conforme exigido pelo MLP (`amostras`, `características`).
3. Desenvolvimento e treinamento do modelo: Construção de um modelo LSTM multicamadas, considerando `relu` para ativação, o otimizador `adam`, a função de perda `mean_squared_error` e a métrica de performance `mae`. 
4. Avaliação do modelo: Cálculo das métricas de performance MAE, MAPE e RMSE.
5. Visualização: Gráfico de linha para visualização dos preços de treinamento, validação e previstos para análise de desempenho.

# Ferramentas
1. Python (`pandas`, `numpy`, `seaborn`, `matplotlib`) para manipulação e visualização dos dados.
2. Keras/TensorFlow para construção do modelo MLP.
3. Scikit-learn para pré-processamento, incluindo escalonamento e divisão dos dados, e para cálculo de métricas de performance.
4. Google Colab para treinamento do modelo.

# Conjunto de dados

Compreende dez anos (01/01/2015 - 01/01/2025) de dados referentes ao preço do café arábica no Brasil. Os dados não são necessariamente diários, apresentando, em geral, uma nova observação a cada dois ou três dias. O conjunto de dados possui as seguintes colunas:
1. data: data de coleta da respectiva observação.
2. preco_reais: preço em reais (R$) no Brasil para a saca de 60kg de café arábica beneficiado.
3. preco_dolares: preço em doláres (US$) no Brasil para a saca de 60kg de café arábica beneficiado.

# Fluxo do projeto

1. **Importação de Bibliotecas**:
   - Importação de bibliotecas necessárias como `pandas`, `numpy`, `seaborn`, `matplotlib`, e ferramentas de aprendizado de máquina do Keras e Scikit-learn.

2. **Carregamento e Pré-processamento dos Dados**:
   - Leitura dos dados a partir de um arquivo CSV.
   - Conversão da coluna de datas para o formato datetime.
   - Filtragem dos dados com base no período 01/01/2015 - 01/01/2025 e ordenação cronológica.

3. **Visualização dos Dados**:
   - Criação e plotagem de gráfico de linha para visualizar o histórico dos preços do café em reais.

4. **Preparação dos Dados para o Modelo LSTM**:
   - Normalização dos dados para uma escala 0-1 utilizando o `MinMaxScaler`.
   - Divisão dos dados em conjuntos de treinamento (80%) e teste (20%).
   - Criação de sequências de dados para treinamento e teste do modelo LSTM com uma janela de tempo fixa (`window_size`) de 60 dias.

5. **Construção do Modelo MLP**:
   - Definição do modelo MLP com camadas sequenciais:
    - Uma camada densa (Dense) com 64 neurônios e função de ativação relu, recebendo como entrada um vetor de tamanho window_size.
    - Uma camada de Dropout com taxa de 20% para reduzir overfitting.
    - Uma segunda camada densa com 32 neurônios e ativação relu.
    - Outra camada de Dropout com taxa de 20%.
    - Uma camada densa final com 1 neurônio (fully conected) para produzir a saída da previsão.
  
6. **Definição de Early Stopping**
   - Utilizado para evitar overfitting durante o treinamento.
   - Monitora a perda na validação (`val_loss`) a cada época.
   - Interrompe o treinamento se não houver melhora após 5 épocas consecutivas (`patience=5`).
   - Garante que o modelo finalize com os melhores pesos já obtidos (`restore_best_weights=True`).

7. **Treinamento do Modelo**:
   - Treinamento do modelo com os dados (`X_train` e `y_train`).
   - Parâmetros utilizados no treinamento:
      - Épocas (`epochs=100`): define o número máximo de passagens completas pelos dados de treino (até 100 rodadas).
      - Batch size (`batch_size=32`): número de amostras processadas antes da atualização dos pesos — aqui, 32 exemplos por vez.
      - `validation_split=0.2`: separa 20% dos dados de treino para avaliar o desempenho do modelo a cada época.
      - `callbacks=[early_stop]`: interrompe o treinamento automaticamente se a perda de validação não melhorar após 5 épocas, e restaura os melhores pesos obtidos.
      - `verbose=1`: exibe o progresso do treinamento em tempo real, mostrando o desempenho em cada época.

8. **Predição e Avaliação do Modelo**:
   - O modelo treinado é utilizado para prever os valores no conjunto de teste (`X_test`).
   - Avaliação do desempenho utilizando métricas como:
     - **MAE (Erro Médio Absoluto)**: Mede o erro médio entre valores reais e previstos.
     - **MAPE (Erro Percentual Absoluto Médio)**: Mede o erro percentual médio.
     - **RMSE (Raiz do Erro Quadrático Médio)**: Mede a raiz quadrada da média dos erros ao quadrado.

9. **Visualização das Previsões**:
   - Criação de um gráfico comparativo para ilustrar:
     - Dados reais de preços.
     - Previsões feitas pelo modelo.
     - Divisão dos dados reais entre os conjuntos de treinamento e teste.

10. **Workflow Principal**:
   - Todo o fluxo de trabalho é organizado em um bloco `if __name__ == "__main__"` para facilitar a execução direta do script.
   - A sequência de etapas descrita acima é automatizada para execução contínua.

# Resultados

A avaliação do modelo aponta os seguintes resultados para as métricas de performance:

1. **MAE (Mean Absolute Error)**:
O MAE é de R$ 26.12, ou seja, em média, as previsões do modelo estão desviando dos dados reais por R$ 26.12. Esse valor é relativamente pequeno, sugerindo que o modelo está fazendo previsões próximas aos dados reais na maioria das vezes.
2. **MAPE (Mean Absolute Percentage Error)**:
O MAPE é de 2.20% e, portanto, indica um erro de 2.20% em média entre os dados reais e os dados previstos. Da mesma forma, essa métrica sugere que o modelo produz previsões precisas, com erros percentuais pequenos.
3. **RMSE (Root Mean Squared Error)**:
O RMSE é de R$ 37.97 e, uma vez que o RMSE é mais sensível a grandes erros do que o MAE, esse resultado mais alto mostra que, embora a média dos erros seja pequena, há algumas previsões em que o modelo erra significativamente, provavelmente em casos isolados ou extremos (outliers).

Podemos concluir que o modelo MLP conseguiu extrair padrões relevantes dos dados, apresentando desempenho competitivo, conforme indicado pelos baixos valores das métricas de performance no conjunto de validação, mesmo sem considerar explicitamente a estrutura temporal das séries. No entanto, o modelo [LSTM](https://github.com/julianacamilen/modelo-lstm-cafe-arabica) apresentou desempenho superior, evidenciando sua maior capacidade de capturar dependências temporais nos dados.

