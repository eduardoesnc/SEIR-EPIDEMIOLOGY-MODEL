# Modelo Híbrido SEIR e Autômato Celular com Análise de Entropia

## 📌 Descrição do Projeto
Este projeto implementa um modelo híbrido avançado para simular a dinâmica de doenças infecciosas. Ele combina a abordagem temporal do modelo **SEIR (Suscetíveis, Expostos, Infectados, Recuperados)** com a representação espacial de um **Autômato Celular (AC)**. O objetivo é modelar a curva epidêmica de doenças com período de latência e, ao mesmo tempo, analisar a evolução da **desordem espacial** do surto utilizando o conceito de **Entropia da Informação (Shannon)**.

## 📊 Sobre os Modelos Utilizados
-   **Modelo SEIR:** Uma extensão do modelo SIR que inclui a classe "Expostos" (E) para indivíduos que foram infectados mas ainda não são capazes de transmitir a doença. É mais realista para doenças como sarampo, catapora ou COVID-19.
-   **Autômato Celular (AC):** Modelo espacial discreto usado para simular a propagação da doença em uma grade 1D. O estado de cada célula (S, E, I ou R) é atualizado com base em regras locais e na prevalência geral de infecção do modelo SEIR.
-   **Entropia da Informação:** Um conceito da teoria da informação usado para medir a incerteza ou a desordem em um sistema. Aqui, é aplicada ao padrão espacial do AC para quantificar o quão misturado e imprevisível é o arranjo dos diferentes estados de saúde na população.

## 🛠 Tecnologias Utilizadas
-   Python 3
-   NumPy
-   SciPy
-   Matplotlib
-   Numba (para otimização de performance)

## 📁 Estrutura do Projeto
```
📂 epdemologia-automato-celular
│-- 📂 data_output_seir
│   └── ca_evolution_seir.png
│   └── entropy_evolution.png
│   └── seir_curve.png
│-- 📜 simulacao_seir_entropia.py  # Script único contendo toda a implementação
│-- 📜 requirements.txt            # Dependências do projeto
│-- 📜 README.md                   # Documentação do projeto
```

## 🔧 Como Executar o Projeto
1.  Clone este repositório:
    ```bash
    git clone [https://github.com/eduardoesnc/SEIR-EPIDEMIOLOGY-MODEL.git](https://github.com/eduardoesnc/SEIR-EPIDEMIOLOGY-MODEL.git)
    ```
2.  Acesse o diretório do projeto:
    ```bash
    cd epdemologia-automato-celular
    ```
3.  Instale as dependências necessárias:
    ```bash
    pip install -r requirements.txt
    ```
4.  Execute o script principal:
    ```bash
    python simulacao_seir_entropia.py
    ```

## 📈 Visualização dos Dados
O código gera três visualizações principais, salvas no diretório `data_output_seir`:
1.  **Curva Epidêmica SEIR:** Gráfico mostrando a evolução das proporções de Suscetíveis, Expostos, Infectados e Recuperados.
2.  **Evolução do Autômato Celular:** Imagem que mostra a propagação espacial da doença na grade do AC ao longo do tempo.
3.  **Evolução da Entropia Espacial:** Gráfico que plota a entropia em função do tempo, ilustrando como a desordem do padrão espacial muda durante a epidemia.

## 📑 Fontes Teóricas
-   Os conceitos do modelo SEIR foram baseados em "Modeling Infectious Diseases in Humans and Animals" de Keeling & Rohani (Capítulo 2).
-   Os conceitos de autômatos celulares e entropia da informação foram baseados em "Cellular Automata: A Discrete View of the World" de Joel L. Schiff (Capítulo 1).