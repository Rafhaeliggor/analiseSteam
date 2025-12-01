Análise de Sucesso de Jogos Steam


Visão Geral
  Este projeto realiza uma análise preditiva do sucesso de jogos na plataforma Steam, com foco específico na comparação entre jogos independentes (Indies) e jogos de grande produção (AAA).
  Utiliza técnicas de aprendizado de máquina para identificar fatores determinantes do sucesso e prever a probabilidade de sucesso de novos lançamentos.

Objetivos
  Analisar as diferenças de desempenho entre jogos Indies e AAA
  Identificar os fatores mais importantes para o sucesso
  Criar modelos preditivos para auxiliar desenvolvedores e publishers
  Gerar insights aplicáveis sobre estratégias de lançamento

Tecnologias Utilizadas
  Python 3.8 ou superior
  Linguagem de programação principal.

Dependências Principais
  pandas (>=1.5.0)
    Manipulação e análise de dados tabulares
    Leitura e processamento do dataset
  numpy (>=1.21.0)
    Computação numérica eficiente
    Operações matemáticas e manipulação de arrays
  matplotlib (>=3.5.0)
    Criação de visualizações gráficas
    Geração de figuras para análise
  seaborn (>=0.12.0)
    Visualizações estatísticas avançadas
    Complemento ao matplotlib
  scikit-learn (>=1.0.0)
    Implementação de algoritmos de machine learning
    Random Forest, Naive Bayes, ferramentas de avaliação
    Pré-processamento e validação de modelos
  kagglehub (>=0.1.0)
    Download automático do dataset Steam
    Acesso direto ao repositório público do Kaggle
    Dataset

O projeto utiliza o dataset Steam Games Dataset, disponível publicamente no Kaggle:
  https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data?select=games.json
O dataset contém informações sobre mais de 100.000 jogos, incluindo:
  Preço e data de lançamento
  Gêneros e categorias
  Plataformas suportadas
  Desenvolvedores e publishers
  Características técnicas

Como Executar
  Pré-requisitos
  
  Instalar Python 3.8 ou superior
  Disponível em: https://www.python.org/downloads/
  
  Instalar dependências:
    pip install -r requirements.txt

Execução do Projeto
  Use o comando:
  python run_analysis_2025.py

Etapas do Processo
  O script executa automaticamente:
    Download dos Dados
    Obtém o dataset do Kaggle

  Pré-processamento
    Limpeza, ajustes e preparação dos dados
  
  Balanceamento
    Ajusta a distribuição das classes de sucesso
  
  Modelagem
    Treinamento de modelos Random Forest e Naive Bayes
  
  Avaliação
    Comparação de desempenho entre os modelos
  
  Análise
    Examina diferenças entre jogos Indies e AAA

  Visualização
    Geração de gráficos e relatórios
    Métricas de Avaliação
    Acurácia: porcentagem de previsões corretas
    Precisão: proporção de verdadeiros positivos entre positivos previstos
    Recall: proporção de verdadeiros positivos detectados
    F1-Score: média harmônica entre precisão e recall

Definições
  Indie
    Preço ≤ 30 dólares
    Até 3 desenvolvedores
  
  AAA
    Preço > 30 dólares
    Ou mais de 3 desenvolvedores

Baseado em combinação de:
  Avaliações positivas
  Número de jogadores
  Métricas de mercado

Limitações
  Baseado em dados históricos
  Critérios simplificados para definição de sucesso
  Informações limitadas ao que a Steam disponibiliza publicamente
  Mercado de jogos altamente dinâmico

Aplicações Práticas
  O projeto pode ser útil para:
    Desenvolvedores independentes
    Publishers
    Pesquisadores

Estudantes interessados em análise de dados e ML

Licença e Uso

Projeto destinado a fins educacionais e de pesquisa.
O dataset utilizado é público e disponível no Kaggle.
