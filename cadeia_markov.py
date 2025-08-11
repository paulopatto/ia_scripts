import numpy as np

# Definição dos estados do clima
states = ["Ensolarado", "Nublado", "Chuvoso"]

# Matriz de transição de estados
# As probabilidades em cada linha somam 1
transition_matrix = [
    [0.8, 0.15, 0.05],  # Transições a partir de "Ensolarado"
    [0.2, 0.6, 0.2],    # Transições a partir de "Nublado"
    [0.25, 0.25, 0.5]   # Transições a partir de "Chuvoso"
]

# Estado inicial
initial_state = "Ensolarado"

# Número de dias a prever
num_days = 10

# Função para encontrar o índice de um estado
def get_state_index(state):
    return states.index(state)

# Função para prever o clima para os próximos dias
def predict_weather(initial_state, num_days):
    current_state = initial_state
    forecast = [current_state]

    for _ in range(num_days - 1):
        current_index = get_state_index(current_state)
        next_state = np.random.choice(
            states, 
            p=transition_matrix[current_index]
        )
        forecast.append(next_state)
        current_state = next_state

    return forecast

# Realizar a previsão
forecast = predict_weather(initial_state, num_days)

# Exibir a previsão
print(f"Estado inicial: {initial_state}")
print("Previsão para os próximos dias:")
for day, state in enumerate(forecast, start=1):
    print(f"Dia {day}: {state}")
