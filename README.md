# Using LLMs to Automate the Discovery of Fundraising Opportunities for Research Institutions

Este repositório contém o artefato desenvolvido para o projeto de pesquisa **“Using LLMs to Automate the Discovery of Fundraising Opportunities for Research Institutions”**, submetido ao **Simpósio Brasileiro de Sistemas de Informação (SBSI 2026)**.

O sistema integra:

- um **agente autônomo de navegação** para coleta de editais em portais de agências de fomento,  
- um pipeline de **ingestão, extração de metadados e indexação vetorial**,  
- uma arquitetura **RAG (Retrieval-Augmented Generation)** para recuperação contextual,  
- e uma **interface Streamlit** para consultas em linguagem natural.

Seu objetivo é reduzir o esforço manual gasto na busca e análise de editais, permitindo que pesquisadores e gestores encontrem oportunidades relevantes de forma mais rápida, precisa e transparente.

---

## Arquitetura Geral

A arquitetura é dividida em dois fluxos principais:

### **1. Fluxo de Atualização Automática (Agente Autônomo)**
Responsável por:
- Navegação automática nos portais de fomento (CNPq, CAPES, FAPESP)  
- Extração de páginas HTML e PDFs  
- Processamento dos textos  
- Divisão em fragmentos (chunking)  
- Indexação no banco vetorial (ChromaDB)

### **2. Fluxo de Consulta (RAG + LLM)**
- Usuário realiza perguntas em linguagem natural via Streamlit  
- Sistema recupera trechos relevantes usando embeddings  
- LLM gera respostas fundamentadas exclusivamente nos documentos indexados  
- Evita respostas fora de contexto (mitigação de alucinações)

---

## Tecnologias Utilizadas

- **Python 3.11+**
- **Streamlit** – Interface web interativa  
- **ChromaDB** – Armazenamento vetorial  
- **sentence-transformers / HuggingFace** – Embeddings  
- **OpenRouter / OpenAI / Anthropic** – LLMs  
- **BeautifulSoup / PyPDF** – Extração de texto  
- **Browser automation** – Agente autônomo

---

## Como Executar

### 1. Clone o repositório
```bash
git clone <link do repositório>
cd ArticleUsingLLMstoDiscoveryofFundraisingOpportunities
```
### 2. Crie o ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
```
### 3. Instale as dependências
```bash
pip install -r requirements.txt
```
### 4. Configure o arquivo .env
```bash
cp .env.example .env
```
### 5. Preencha com sua chave
```bash
OPENROUTER_API_KEY=...
```
### 6. Execute a interface Streamlit
```bash
streamlit run app.py
```
---

## Prova de Conceito (PoC)

A PoC inclui cenários funcionais que validam o artefato:

- Identificação de elegibilidade
- Extração de datas e restrições
- Filtragem por áreas e requisitos
- Resumo de oportunidades
- Recuperação fundamentada (trechos exibidos ao usuário)
- Detecção e recusa de perguntas fora do escopo

---

## Metodologia

Este artefato foi desenvolvido seguindo as etapas da metodologia DSR:

1. Identificação do problema
2. Definição dos objetivos da solução
3. Projeto e construção do artefato
4. Demonstração (PoC funcional)
5. Comunicação (artigo para SBSI 2026)

---

## Limitações

- Dependência de layout estável dos portais (fragilidade do scraping)
- Atualização automática configurada em código
- Falta de testes com usuários finais (planejado)
- Ausência de persistência entre sessões
