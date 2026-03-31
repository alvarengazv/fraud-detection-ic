<a name="readme-topo"></a>

<h1 align='center'>
  Atividade Prática 01 - Metodologia Experimental
</h1>

<div align='center'>

[![SO][Ubuntu-badge]][Ubuntu-url]
[![IDE][vscode-badge]][vscode-url]
[![Python][Python-badge]][Python-url]

<b>
  Guilherme Alvarenga de Azevedo<br>
  Maria Eduarda Teixeira Souza<br>
</b>
  
<br>
Inteligência Computacional <br>
Engenharia de Computação <br>
CEFET-MG Campus V <br>
2026/1 


</div>

## Introdução

O presente projeto foi desenvolvido como parte da "Atividade Prática 01 - Metodologia Experimental" da disciplina de Inteligência Computacional. O foco principal deste trabalho é aplicar de forma rigorosa as etapas do pipeline de aprendizado de máquina, incluindo a análise e compreensão dos dados, o pré-processamento minucioso, e a definição de uma metodologia experimental correta para a comparação de algoritmos.

O cenário de aplicação escolhido para este estudo de caso é a detecção de fraudes financeiras, um problema crítico e de grande impacto no mundo real, no qual sistemas inteligentes devem aprender a diferenciar o comportamento normal de um usuário das ações de um fraudador. Ao longo deste projeto, lidamos com os desafios estatísticos inerentes à base de dados e avaliamos o desempenho de diferentes classificadores, como o Random Forest, XGBoost e a Regressão Logística, demonstrando como o tratamento prévio dos dados influencia no poder preditivo das técnicas de Inteligência Computacional.

## Base de Dados

O conjunto de dados escolhido é o Financial Fraud Detection Dataset — 1M Transactions, disponível publicamente na plataforma [Kaggle](https://www.kaggle.com/datasets/sergionefedov/fraud-detection-1m-transactions-7-fraud-types). Ela se trata de um "compreensível conjunto de dados sintéticos que abrange transações de cartão de crédito e pagamentos digitais de 2022–2024. Construído com um modelo probabilístico realista de geração de fraudes que replica os principais padrões observados em operações de fraude do mundo real".

Problema Associado: O problema consiste em tentar identificar transações financeiras fraudulentas (como roubo de identidade, lavagem de dinheiro, clonagem de cartão e fraudes de invasão de conta) a partir da análise dos padrões de comportamento do usuário, histórico da conta, valor financeiro e dados de rede do dispositivo no momento da operação.

Tipo de Tarefa: A atividade propõe uma tarefa de classificação binária, cuja variável alvo a ser predita é a coluna is_fraud (onde 0 representa uma transação legítima e 1 representa uma transação fraudulenta).

Caracterização Geral: Trata-se de uma base de dados multivariada volumosa, contendo exatas 1.000.000 de transações associadas a um universo de 50.000 contas. O conjunto é composto por uma mistura variada de atributos, sendo aproximadamente 26% binários, 26% discretos e 17% contínuos. Entre as principais variáveis explanatórias, destacam-se o valor transacionado (amount), o escore de risco de rede (ip_risk_score), a velocidade de transações na última hora (velocity_1h), categorias nominais dos lojistas (merchant_category) e o tempo desde a última operação.

Desafios e Qualidade dos Dados: A partir da Análise Exploratória de Dados (EDA), foram identificados características e problemas críticos que nortearam a fase de pré-processamento:

- Desbalanceamento Extremo: A variável alvo é severamente desbalanceada. Do volume total, 98,3% (982.857) das transações são legítimas e apenas 1,7% (17.143) são fraudes. Devido a este abismo na proporção, métricas tradicionais como a acurácia tornam-se ineficazes e ilusórias para avaliar o classificador, exigindo o uso de PR-AUC, Estatística KS e Precisão em Recall Fixo (80%). Também tornou mandatório o uso de amostragem estratificada para a divisão dos dados de Treino/Validação/Teste.

- Vazamento de Dados (Target Leakage): A análise do mapa de valores ausentes identificou que a variável fraud_pattern (que descreve o rótulo do tipo de fraude) só é preenchida caso a transação seja, de fato, uma fraude, permanecendo nula no restante. A sua manutenção na base criaria um vazamento de resposta (target leakage), sendo sua exclusão obrigatória.

- Por se tratar de um problema de detecção de fraudes financeiras, essas anomalias estatísticas representam justamente os picos de ação de fraudadores (anomalias de comportamento), de modo que os outliers não devem ser removidos ou podados na limpeza.

- Redundância e Dimensionalidade: O cruzamento das variáveis contínuas num mapa de calor (Correlação de Pearson) revelou uma redundância perfeita (0,80) entre as colunas amount (valor) e amount_vs_avg_ratio (valor comparado com a média). 

Ao mesmo tempo, a aplicação da técnica One-Hot Encoding sobre as variáveis nominais do dataset elevou as características de entrada para 43 dimensões numéricas prontas para treinamento.

## 📚 O Projeto

Neste repositório você encontrará o código fonte do projeto, bem como os dados utilizados para a análise. O projeto foi desenvolvido em Python. Este trabalho também tem a produção de um slide em PDF para relatar o trabalho, que está disponível em [`Slide`](slides/slide.pdf).

De uma forma compacta e organizada, os arquivos e diretórios estão dispostos da seguinte forma:

  ```.
fraud-detection-ic/ 
    ├── output/
    │   ├── eda_output/
    │   │   ├── boxplots_por_classe.png
    │   │   ├── correlacao_com_alvo.png
    │   │   └── ... (e outros .png)
    │   ├── experimenting_output/
    │   │   ├── curvas_roc_pr.png
    │   │   ├── kfold_pr_auc.png
    │   │   ├── kfold_resultados.csv
    │   │   ├── matrizes_confusao.png
    │   │   └── resultados_teste_final.csv
    │   |
    ├── src/
    │   ├── eda/
    │   │   ├── eda.py
    │   │   └── plots.py
    │   ├── experimenting.py
    │   ├── main.py
    │   └── preprocessing.py
    ├── .gitignore
    ├── README.md
    ├── relatorio.pdf
    └── requirements.txt
  ```

## Instalando
Para instalar o projeto, siga os passos abaixo:

<div align="justify">
  Com o ambiente preparado, os seguintes passos são para a instalação, compilação e execução do programa localmente:

  1. Clone o repositório no diretório desejado:
  ```console
  git clone https://github.com/alvarengazv/fraud-detection-ic.git
  cd fraud-detection-ic
  ```
  2. Crie e ative um ambiente virtual (recomendado) - garanta que já possui o [Python](https://www.python.org/downloads/), no mínimo na versão 3.11.9:
  ```console
  python3 -m venv venv
  source venv/bin/activate   # Linux/macOS
  venv\Scripts\activate      # Windows
  ```
  3. Instale as dependências com pip: 
  ```console
    pip install -r requirements.txt
  ```
</div>
<div align="justify">
  
  4. Execute o programa:
      - **Linux/macOS**
        ```console
          # Usando Python diretamente
          # PYTHONPATH='src' python3 -m main
        ```

      - **Windows**
        ```console
          # Usando Python diretamente
          # python3 src/main.py
        ```
</div> 

<div align="justify">
  
  ## Utilização

  Responda com 's' para executar uma etapa e 'n' para pular.

  ```console 
  ...
  Deseja executar a Análise Exploratória (EDA)? [s/N]: n
  ...
  Deseja executar o Pré-processamento? [S/n]: n
  ...
  Deseja usar o dataset pré-processado existente? [S/n]: s
  ...
  Deseja executar a validação cruzada (K-Fold)? [s/N]: n
  ```

</div>

<div align="justify">
  
  ## Dependências

  O projeto utiliza as seguintes bibliotecas:

  - pandas
  - numpy
  - matplotlib
  - seaborn
  - kagglehub
  - scikit-learn
  - xgboost

</div>

> [!NOTE]
> No arquivo [`requirements.txt`](requirements.txt) tem todas essas informações.

<p align="right">(<a href="#readme-topo">voltar ao topo</a>)</p>

## 🧪 Ambiente de Compilação e Execução

<div align="justify">

  O trabalho foi desenvolvido e testado em várias configurações de hardware. Podemos destacar algumas configurações de Sistema Operacional e Compilador, pois as demais configurações não influenciam diretamente no desempenho do programa.

</div>

<div align='center'>

[![SO][Ubuntu-badge]][Ubuntu-url]
[![IDE][vscode-badge]][vscode-url]
[![Python][Python-badge]][Python-url]

| *Hardware* | *Especificações* |
|:------------:|:-------------------:|
| *Laptop*   | Dell Inspiron 13 5330 |
| *Processador* | Intel Core i7-1360P |
| *Memória RAM* | 16 GB DDR5 |
| *Sistema Operacional* | Ubuntu 24.04 |
| *IDE* | Visual Studio Code |
| *Placa de Vídeo* | Intel Iris Xe Graphics |

</div>

> [!IMPORTANT] 
> Para que os testes tenham validade, considere as especificações
> do ambiente de compilação e execução do programa.

<p align="right">(<a href="#readme-topo">voltar ao topo</a>)</p>

## 📨 Contato

<div align="center">
  <br><br>
     <i>Guilherme Alvarenga de Azevedo - Graduando - 7º Período de Engenharia de Computação @ CEFET-MG</i>
  <br><br>
  
  [![Gmail][gmail-badge]][gmail-autor1]
  [![Linkedin][linkedin-badge]][linkedin-autor1]
  [![Telegram][telegram-badge]][telegram-autor1]
  
  
  <br><br>
     <i>Maria Eduarda Teixeira Souza - Graduando - 7º Período de Engenharia de Computação @ CEFET-MG</i>
  <br><br>
  
  [![Gmail][gmail-badge]][gmail-autor2]
  [![Linkedin][linkedin-badge]][linkedin-autor2]
  [![Telegram][telegram-badge]][telegram-autor2]

</div>

<p align="right">(<a href="#readme-topo">voltar ao topo</a>)</p>

<a name="referencias">📚 Referências</a>

1. AZEVEDO, Guilherme A. SOUZA, Maria E. T. **FRAUD-DETECTION-IC**: Atividade Prática 01 - Metodologia Experimental. 2026. Disponível em: [https://github.com/alvarengazv/fraud-detection-ic](https://github.com/alvarengazv/fraud-detection-ic) Acesso em: 31 mar. 2026.

2. SILVA, Alisson M. **Inteligência Computacional**: Dados. Slides de Aula. 2026.

3. SILVA, Alisson M. **Inteligência Computacional**: Metodologia dos Experimentos. Slides de Aula. 2026.

4. SILVA, Alisson M. **Inteligência Computacional**: Experimentos. Slides de Aula. 2026.


[vscode-badge]: https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white
[vscode-url]: https://code.visualstudio.com/docs/?dv=linux64_deb
[make-badge]: https://img.shields.io/badge/_-MAKEFILE-427819.svg?style=for-the-badge
[make-url]: https://www.gnu.org/software/make/manual/make.html
[cpp-badge]: https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white
[cpp-url]: https://en.cppreference.com/w/cpp
[trabalho-url]: https://drive.google.com/file/d/1-IHbGaA1BIC6_CMBydOC-NbV2bCERc8r/view?usp=sharing
[github-prof]: https://github.com/mpiress
[main-ref]: src/main.cpp
[branchAMM-url]: https://github.com/alvarengazv/trabalhosAEDS1/tree/AlgoritmosMinMax
[makefile]: ./makefile
[bash-url]: https://www.hostgator.com.br/blog/o-que-e-bash/
[lenovo-badge]: https://img.shields.io/badge/lenovo%20laptop-E2231A?style=for-the-badge&logo=lenovo&logoColor=white
[ubuntu-badge]: https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white
[Ubuntu-url]: https://ubuntu.com/
[ryzen5500-badge]: https://img.shields.io/badge/AMD%20Ryzen_5_5500U-ED1C24?style=for-the-badge&logo=amd&logoColor=white
[ryzen3500-badge]: https://img.shields.io/badge/AMD%20Ryzen_5_3500X-ED1C24?style=for-the-badge&logo=amd&logoColor=white
[windows-badge]: https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white
[gcc-badge]: https://img.shields.io/badge/GCC-5C6EB8?style=for-the-badge&logo=gnu&logoColor=white
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/


[linkedin-autor1]: https://www.linkedin.com/in/guilherme-alvarenga-de-azevedo-959474201/
[telegram-autor1]: https://t.me/alvarengazv
[gmail-autor1]: mailto:gui.alvarengas234@gmail.com

[linkedin-autor2]: https://www.linkedin.com/in/dudatsouza/
[telegram-autor2]: https://t.me/dudat_18
[gmail-autor2]: mailto:dudateixeirasouza@gmail.com

[linkedin-badge]: https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=Linkedin&logoColor=white
[telegram-badge]: https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white
[gmail-badge]: https://img.shields.io/badge/-Gmail-D14836?style=for-the-badge&logo=Gmail&logoColor=white
