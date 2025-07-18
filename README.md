# Solar-power-prediction-LSTM
Previsão de gereção de energia solar de uma planta fotovoltaica com capacidade de 8000W, utilizando Redes Neurais Recorrentes (RNN) do tipo Long Short-Term Memory (LSTM) a partir de um modelo treinado com dados históricos coletado.


  

## Tecnologias Usadas

  - Python
  - TensorFlow / Keras
  - Pandas
  - NumPy
  - Scikit-Learn
  - Matplotlib


## dados utilizados
 Foram concatenados dados de uma planta de energia de producao de energia solar de capacidade de 8000W coletados de 15 em 15 minutos , do dia **2022-08-06 18:36:34** até **2024-11-16 18:31:30**  que foram salvos em um arquivo .xlsx .Para a previsáo utilizaremos a coluna **Potência de saída CC(W)**







## modelo de previsoes - **LSTM**

O modelo de rede neural utilizado neste projeto é baseado em Long Short-Term Memory (**LSTM**) que é um tipo de rede neural recorrente (**RNN**) criada para lidar com série de dependenias temporais de longo prazo. Foi utilizado a técnica Min-Max Scaling para normalizarmmos a variável de previsao (**Potência de saída CC(W)**). Em seguida , foram geradas janelas deslizantes de dados com tamanho fixo de 50 amostras, ou seja utiliza as ultimams 50 amostras para a preisao. O modelo neural foi construído com duas camadas LSTM empilhadas, contendo 64 unidades cada, seguidas por uma camada densa intermediária com 32 neurônios e função de ativação ReLU, e uma camada de saída com um único neurônio para regressão contínua. A rede foi treinada com o otimizador Adam e função de perda do erro quadrático médio (**MSE**), utilizando 80% dos dados para treinamento e 20% para validação. Após o treinamento, o modelo foi utilizado para realizar previsões sobre o conjunto de teste, sendo os resultados transformados de volta à escala original para interpretação e análise do desempenho.




## resultados
| Métrica | Valor |
|--------|--------|
| MAE    | 299.77 |
| MSE    | 456542.74 |
| RMSE   | 675.68 |
| R²     | 0.8916 |
| MAPE (%) | 15.39 |
#### Graficos comparando valores previstos e reais, em azul os previsto e em laranja os reais.
![image](https://github.com/user-attachments/assets/f47f09f9-a5f2-40ea-967f-476881f4aeda)
![image](https://github.com/user-attachments/assets/64aeaec6-61c9-4abb-974a-d73924ca308f)
![image](https://github.com/user-attachments/assets/31a36955-d9a3-48d8-93c3-8386858e0c94)
![image](https://github.com/user-attachments/assets/9db64a44-31a6-40f6-82cf-ae266c37aa4d)



Como Executar
  
  1. Coloque seu arquivo `dados_concatenados.xlsx` na pasta `data/`.
  2. Execute o script Python:





