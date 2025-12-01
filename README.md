# Análise de Sucesso de Jogos Steam

## Visão Geral
Este projeto realiza uma análise preditiva do sucesso de jogos na plataforma Steam, comparando jogos independentes (Indies) e jogos de grande produção (AAA).  
O objetivo é identificar fatores determinantes do sucesso e prever resultados para novos lançamentos utilizando técnicas de aprendizado de máquina.

---

## Objetivos
- Analisar diferenças de desempenho entre jogos Indies e AAA  
- Identificar fatores mais relevantes para o sucesso  
- Criar modelos preditivos para ajudar desenvolvedores e publishers  
- Gerar insights aplicáveis para estratégias de lançamento  

---

## Tecnologias Utilizadas

### Python 3.8 ou superior
Linguagem principal utilizada no projeto.

---

## Dependências Principais

### pandas (>=1.5.0)
- Manipulação e análise de dados tabulares  
- Leitura e processamento do dataset  

### numpy (>=1.21.0)
- Computação numérica eficiente  
- Operações matemáticas e manipulação de arrays  

### matplotlib (>=3.5.0)
- Criação de visualizações gráficas  
- Geração de figuras para análise  

### seaborn (>=0.12.0)
- Visualizações estatísticas avançadas  
- Complemento ao matplotlib  

### scikit-learn (>=1.0.0)
- Algoritmos de machine learning  
- Random Forest, Naive Bayes  
- Pré-processamento e validação de modelos  

### kagglehub (>=0.1.0)
- Download automático do dataset da Steam  
- Acesso ao repositório público do Kaggle  

---

## Dataset

O projeto utiliza o dataset **Steam Games Dataset**, disponível em:

https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data?select=games.json

O dataset contém mais de 100.000 jogos com informações como:
- Preço e data de lançamento  
- Gêneros e categorias  
- Plataformas suportadas  
- Desenvolvedores e publishers  
- Características técnicas  

---

## Como Executar

### Pré-requisitos
Instale o Python 3.8 ou superior:  
https://www.python.org/downloads/

Instale as dependências:
```bash
pip install -r requirements.txt
```

---

## Execução do Projeto
Execute o script principal com:
```bash
python run_analysis_2025.py
```

---

## Etapas do Processo

### 1. Download dos Dados
- Baixa automaticamente o dataset do Kaggle

### 2. Pré-processamento
- Limpeza, ajustes e preparação dos dados

### 3. Balanceamento
- Ajusta a distribuição das classes de sucesso

### 4. Modelagem
- Treinamento dos modelos:  
  - Random Forest  
  - Naive Bayes  

### 5. Avaliação
- Compara o desempenho entre os modelos

### 6. Análise
- Examina diferenças entre jogos Indies e AAA

### 7. Visualização
- Gera gráficos e relatórios

---

## Métricas de Avaliação
- **Acurácia:** porcentagem de previsões corretas  
- **Precisão:** proporção de verdadeiros positivos entre os previstos como positivos  
- **Recall:** proporção de verdadeiros positivos detectados  
- **F1-Score:** média harmônica entre precisão e recall  

---

## Definições

### Indie
- Preço ≤ 30 dólares  
- Até 3 desenvolvedores  

### AAA
- Preço > 30 dólares  
- Ou mais de 3 desenvolvedores  

### Sucesso
Baseado em combinação de:
- Avaliações positivas  
- Número de jogadores  
- Métricas de mercado  

---

## Limitações
- Baseado em dados históricos  
- Critérios simplificados para definição de sucesso  
- Informações limitadas ao que a Steam disponibiliza  
- Mercado de jogos altamente dinâmico  

---

## Aplicações Práticas
Este projeto pode ser utilizado por:
- Desenvolvedores independentes  
- Publishers  
- Pesquisadores  
- Estudantes interessados em análise de dados e machine learning  

---

## Licença e Uso
Este projeto é destinado a fins educacionais e de pesquisa.  
O dataset utilizado é público e disponível no Kaggle.
