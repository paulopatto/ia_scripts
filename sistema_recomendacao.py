import heapq

class Produto:
    def __init__(self, nome, categoria, conversao_probabilidade):
        self.nome = nome
        self.categoria = categoria
        self.conversao_probabilidade = conversao_probabilidade

    def __repr__(self):
        return f"{self.nome} ({self.categoria})"

class AStarRecommendation:
    def __init__(self, produtos, heuristica):
        self.produtos = produtos
        self.heuristica = heuristica  # Função heurística: probabilidade de conversão
        self.grafo = self.criar_grafo()

    def criar_grafo(self):
        # Criando um grafo simplificado de produtos, onde cada produto se conecta ao próximo
        grafo = {}
        for produto in self.produtos:
            grafo[produto] = [p for p in self.produtos if p != produto]
        return grafo

    def a_star(self, inicio, objetivo):
        fila_prioridade = []
        heapq.heappush(fila_prioridade, (0 + self.heuristica(inicio), 0, inicio))  # f = g + h
        visitados = set()
        caminhos = {}

        while fila_prioridade:
            _, g, atual = heapq.heappop(fila_prioridade)

            if atual in visitados:
                continue

            visitados.add(atual)
            if atual == objetivo:
                break

            for vizinho in self.grafo[atual]:
                if vizinho not in visitados:
                    h = self.heuristica(vizinho)
                    heapq.heappush(fila_prioridade, (g + 1 + h, g + 1, vizinho))  # g é o custo acumulado
                    caminhos[vizinho] = atual

        # Recuperando o caminho
        caminho = []
        produto = objetivo
        while produto in caminhos:
            caminho.insert(0, produto)
            produto = caminhos[produto]
        return caminho

def heuristica(produto):
    # Quanto maior a probabilidade de conversão, mais atraente é o produto
    return -produto.conversao_probabilidade  # Vamos minimizar a heurística (quanto menor, melhor)

# Exemplo de produtos
produtos = [
    Produto("Produto A", "Categoria 1", 0.9),
    Produto("Produto B", "Categoria 1", 0.8),
    Produto("Produto C", "Categoria 2", 0.7),
    Produto("Produto D", "Categoria 2", 0.6),
]

# Criando o sistema de recomendação
recomendador = AStarRecommendation(produtos, heuristica)

# Buscar o melhor caminho entre dois produtos
inicio = produtos[0]  # Produto A
objetivo = produtos[2]  # Produto C

caminho_recomendado = recomendador.a_star(inicio, objetivo)

# Exibindo a recomendação
print("Caminho recomendado:")
for p in caminho_recomendado:
    print(p)
