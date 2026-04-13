# 🚀 Local LLM API Server

I built this project to run large language models locally and expose them through a clean API — similar to how OpenAI APIs work, but fully on your own machine.

It’s powered by FastAPI and uses Ollama under the hood to handle the models.

---

## 💡 Why I built this

While working with LLMs, I wanted:

* A **local alternative** to paid APIs
* More **control over inference**
* A way to **experiment and build apps on top of local models**

So I created this API layer that sits between a client and a local LLM.

---

## ✨ What it does

* 🔐 API key-based authentication
* ⚡ Rate limiting (to prevent abuse)
* 🔄 Supports both streaming and non-streaming responses
* 🤖 OpenAI-style `/chat` endpoint
* 📄 Auto-generated API docs with Swagger

---

## 🛠 Tech Stack

* FastAPI
* Ollama
* Uvicorn
* HTTPX

---

## 🚀 Getting Started

### 1. Install Ollama

```bash
brew install ollama
ollama serve
ollama pull mistral
```

---

### 2. Setup Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Run the server

```bash
uvicorn main:app --reload
```

---

### 4. Open API docs

Go to:
http://127.0.0.1:8000/docs

---


---
## 📌 Example Request
curl -X POST http://127.0.0.1:8000/generate \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'

---






This project helped me understand how LLM APIs actually work behind the scenes — from request handling to streaming responses.

If you're exploring local AI setups or building your own tools on top of LLMs, this might be useful for you too.
