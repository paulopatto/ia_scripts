import numpy as np
from scipy.stats import norm
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# 1. Criando uma Rede Bayesiana para modelar o problema
model = BayesianNetwork([
    ('HistoricoCompras', 'Compra'),
    ('TempoNoSite', 'Compra'),
    ('ClicouEmPromocao', 'Compra')
])

# 2. Definindo as probabilidades condicionais (CPDs)
cpd_historico = TabularCPD(variable='HistoricoCompras', variable_card=2, values=[[0.7], [0.3]])
cpd_tempo = TabularCPD(variable='TempoNoSite', variable_card=2, values=[[0.6], [0.4]])
cpd_promocao = TabularCPD(variable='ClicouEmPromocao', variable_card=2, values=[[0.8], [0.2]])

# Compra é influenciada pelas outras variáveis
cpd_compra = TabularCPD(
    variable='Compra',
    variable_card=2,
    values=[
        [0.9, 0.7, 0.8, 0.4, 0.6, 0.2, 0.3, 0.1],  # Probabilidade de não comprar
        [0.1, 0.3, 0.2, 0.6, 0.4, 0.8, 0.7, 0.9]   # Probabilidade de comprar
    ],
    evidence=['HistoricoCompras', 'TempoNoSite', 'ClicouEmPromocao'],
    evidence_card=[2, 2, 2]
)

# 3. Adicionando os CPDs ao modelo
model.add_cpds(cpd_historico, cpd_tempo, cpd_promocao, cpd_compra)

# 4. Validando o modelo
assert model.check_model()

# 5. Prevendo a probabilidade de compra dado um cenário
from pgmpy.inference import VariableElimination

inference = VariableElimination(model)
result = inference.query(variables=['Compra'], evidence={
    'HistoricoCompras': 1,  # O cliente tem histórico de compras
    'TempoNoSite': 0,       # O cliente passou pouco tempo no site
    'ClicouEmPromocao': 1   # O cliente clicou em uma promoção
})

print("Probabilidades de Compra:")
print(result) # Resultado da probabilidade de comprar ou não comprar com base nas evidências

# 6. Explorando distribuições para modelar incerteza adicional
# Exemplo: tempo no site segue uma distribuição normal
media_tempo = 5  # minutos
desvio_padrao_tempo = 2
tempo_observado = 6
prob_tempo = norm.cdf(tempo_observado, loc=media_tempo, scale=desvio_padrao_tempo)

print(f"Probabilidade de o cliente passar menos de {tempo_observado} minutos no site: {prob_tempo:.2f}")
