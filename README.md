Análise de Sucesso de Jogos Steam
Visão Geral
Este projeto realiza uma análise preditiva do sucesso de jogos na plataforma Steam, com foco específico na comparação entre jogos independentes (Indies) e jogos de grande produção (AAA). O sistema utiliza técnicas de aprendizado de máquina para identificar os fatores determinantes do sucesso e prever a probabilidade de sucesso de novos lançamentos.

Objetivos
Analisar as diferenças de desempenho entre jogos Indies e AAA

Identificar os fatores mais importantes para o sucesso de jogos

Criar modelos preditivos que possam auxiliar desenvolvedores e publishers

Gerar insights acionáveis sobre estratégias de lançamento

Tecnologias Utilizadas
Python 3.8 ou superior
Linguagem de programação principal

Requer instalação do interpretador Python

Dependências Principais:
pandas (>=1.5.0)

Manipulação e análise de dados tabulares

Leitura e processamento do dataset de jogos

numpy (>=1.21.0)

Computação numérica eficiente

Operações matemáticas e manipulação de arrays

matplotlib (>=3.5.0)

Criação de visualizações gráficas

Geração de gráficos e figuras para análise

seaborn (>=0.12.0)

Visualizações estatísticas avançadas

Complemento ao matplotlib para gráficos mais elaborados

scikit-learn (>=1.0.0)

Implementação de algoritmos de machine learning

Contém Random Forest, Naive Bayes e ferramentas de avaliação

Pré-processamento e validação de modelos

kagglehub (>=0.1.0)

Download automático do dataset Steam da plataforma Kaggle

Acesso à base de dados pública de jogos

Dataset
O projeto utiliza o dataset "Steam Games Dataset" disponível publicamente no Kaggle (https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data?select=games.json) , que contém informações detalhadas sobre mais de 100.000 jogos da plataforma Steam, incluindo:

Informações de preço e lançamento

Gêneros e categorias

Suporte a plataformas

Desenvolvedores e publishers

Características técnicas


Como Executar
Pré-requisitos
Instale o Python 3.8 ou versão superior

Disponível em: https://www.python.org/downloads/

Instale as dependências:

bash
pip install -r requirements.txt
Execução do Projeto
Execute o seguinte comando no terminal:

bash
python run_analysis_2025.py
Etapas do Processo
O script executará automaticamente as seguintes etapas:

Download dos Dados: Baixa o dataset Steam do Kaggle

Pré-processamento: Limpeza e preparação dos dados

Balanceamento: Ajusta a distribuição de classes

Modelagem: Treina modelos Random Forest e Naive Bayes

Avaliação: Compara o desempenho dos modelos

Análise: Examina diferenças entre Indies e AAA

Visualização: Gera gráficos e relatórioso


Métricas de Avaliação
Os modelos são avaliados utilizando as seguintes métricas:

Acurácia: Porcentagem de previsões corretas

Precisão: Proporção de verdadeiros positivos entre os previstos como positivos

Recall: Proporção de verdadeiros positivos detectados

F1-Score: Média harmônica entre precisão e recall

Definições
Indie: Jogos com preço ≤ $30 e ≤ 3 desenvolvedores

AAA: Jogos com preço > $30 ou > 3 desenvolvedores

Sucesso: Baseado em uma combinação de avaliações positivas, número de jogadores e métricas de mercado

Limitações
Dados Históricos: A análise é baseada em dados históricos

Definição de Sucesso: Baseada em critérios simplificados

Disponibilidade de Dados: Limitada às informações públicas do Steam

Mercado Dinâmico: O mercado de jogos evolui rapidamente

Aplicações Práticas
Este projeto pode ser utilizado por:

Desenvolvedores Independentes: Para entender fatores de sucesso

Publishers: Para avaliação de projetos e investimentos

Pesquisadores: Para estudos acadêmicos sobre o mercado de jogos

Estudantes: Como exemplo de análise de dados e machine learning


Licença e Uso
Este projeto é destinado para fins educacionais e de pesquisa. O dataset utilizado é público e disponível no Kaggle.