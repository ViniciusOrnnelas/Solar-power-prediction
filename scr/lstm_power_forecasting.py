import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_excel("../data/dados_concatenados.xlsx")  # Caminho ajustado para pasta 'data'
df['Data E Hora'] = pd.to_datetime(df['Data E Hora'])
df = df.sort_values('Data E Hora')

# Selecionar a variável de interesse
valores = df['Potência de saída CC(W)'].values

# Normalizar os dados
scaler = MinMaxScaler()
valores_normalizados = scaler.fit_transform(valores.reshape(-1, 1))

# Função para criar sequências
def criar_sequencias(dados, tamanho_seq):
    x, y = [], []
    for i in range(len(dados) - tamanho_seq):
        x.append(dados[i:i + tamanho_seq])
        y.append(dados[i + tamanho_seq])
    return np.array(x), np.array(y)

# Criar sequências
tamanho_seq = 50
X, y = criar_sequencias(valores_normalizados, tamanho_seq)

# Divisão dos dados em treino e teste
percentual_treino = 0.8
tamanho_treino = int(len(X) * percentual_treino)
X_treino, y_treino = X[:tamanho_treino], y[:tamanho_treino]
X_teste, y_teste = X[tamanho_treino:], y[tamanho_treino:]

# Preparar os dados para o modelo LSTM
X_treino = np.expand_dims(X_treino, axis=-1)
X_teste = np.expand_dims(X_teste, axis=-1)

# Construção do modelo LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(tamanho_seq, 1)),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compilação do modelo
model.compile(optimizer='adam', loss='mse')

# Treinamento
history = model.fit(
    X_treino, y_treino,
    validation_data=(X_teste, y_teste),
    epochs=12,
    batch_size=32,
    verbose=1
)

# Fazer previsões
y_previsoes = model.predict(X_teste)

# Reverter a normalização
y_previsoes_invertido = scaler.inverse_transform(y_previsoes)
y_teste_invertido = scaler.inverse_transform(y_teste.reshape(-1, 1))

# Criar o DataFrame de resultados
datas_teste = df['Data E Hora'].iloc[tamanho_treino + tamanho_seq:].reset_index(drop=True)
resultado_df = pd.DataFrame({
    'Data E Hora': datas_teste,
    'actual': y_teste_invertido.flatten(),
    'predicted': y_previsoes_invertido.flatten()
})
resultado_df['erro_abs'] = abs(resultado_df['actual'] - resultado_df['predicted'])

# Calcular métricas de avaliação
mae = mean_absolute_error(y_teste_invertido, y_previsoes_invertido)
mse = mean_squared_error(y_teste_invertido, y_previsoes_invertido)
rmse = np.sqrt(mse)
r2 = r2_score(y_teste_invertido, y_previsoes_invertido)
mask = y_teste_invertido != 0
mape = np.mean(np.abs((y_teste_invertido[mask] - y_previsoes_invertido[mask]) / y_teste_invertido[mask])) * 100

# Mostrar métricas no console
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Exportar resultados para Excel
with pd.ExcelWriter('../results/resultado_previsao_multivariada2.xlsx', engine='openpyxl', mode='w') as writer:
    resultado_df.to_excel(writer, sheet_name='Resultados', index=False)

    metricas_df = pd.DataFrame({
        'Métrica': ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE (%)'],
        'Valor': [mae, mse, rmse, r2, mape]
    })
    metricas_df.to_excel(writer, sheet_name='Métricas', index=False)

# Visualizar os resultados
plt.figure(figsize=(12, 6))
plt.plot(resultado_df['Data E Hora'], resultado_df['actual'], label='Real', color='blue')
plt.plot(resultado_df['Data E Hora'], resultado_df['predicted'], label='Previsto', color='orange')
plt.xlabel('Data E Hora')
plt.ylabel('Potência de saída CC (W)')
plt.legend()
plt.title('Valores Reais vs Previstos')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
