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

## 📚 Tecnologias Utilizadas

- **Python 3.11+**
- **Streamlit** – Interface web interativa  
- **ChromaDB** – Armazenamento vetorial  
- **sentence-transformers / HuggingFace** – Embeddings  
- **OpenRouter / OpenAI / Anthropic** – LLMs  
- **BeautifulSoup / PyPDF** – Extração de texto  
- **Browser automation** – Agente autônomo

---

## 🚀 Como Executar

### 1. Clone o repositório
```bash
git clone <link do repositório>
cd ArticleUsingLLMstoDiscoveryofFundraisingOpportunities

### 2. Clone o repositório
