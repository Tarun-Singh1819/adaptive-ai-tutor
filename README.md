🎓 Adaptive AI Research Paper Tutor
A High-Performance Multi-Agent System for Personalized Learning

This is an advanced AI assistant built to help you master research papers. It uses a Multi-Agent architecture to ensure you get the most accurate answers with the lowest possible wait time.


🧠 The Intelligence: Router & Agents
The core of this system is the Router Node. Instead of sending every question to the heavy Research pipeline, the Router analyzes your Query and Memory (STM/LTM) to decide which specialist should handle it:

Router Node: The "Brain" that reads your question and checks your past history to pick the best agent.

Overview Agent: For general summaries and high-level concepts using LLM knowledge and Web Search.

Doubt Agent: For quick clarifications and simple questions without needing deep paper analysis.

Research Agent: The specialist for deep, grounded answers using an Agentic RAG pipeline for paper-specific facts.


💾 Decoupled Memory System (STM & LTM)
Unlike standard memory systems, I built this as two large, independent workflows. Because it is decoupled, this memory engine can be easily integrated into other AI projects.

Short-Term Memory (STM): Manages the immediate conversation flow and context.

Long-Term Memory (LTM): Stores permanent user preferences and learned facts with semantic deduplication , this is important to know user preferances.

Async Memory Fire: Uses a "Fire-and-Forget" node to update both memory workflows in the background, ensuring zero lag for the user.


⚡ Key Technical Features
Ultra-Fast Performance: Complete workflow execution in just 3–4 seconds.

Async Memory Fire: Uses a "Fire-and-Forget" logic to update your learning history in the background, so the chat never slows down.

Dual-Layer Memory: * Short-Term Memory (STM): Tracks the current session's flow.

Long-Term Memory (LTM): Stores your personal preferences and learned facts permanently.

Smart Retrieval: Uses Hybrid Search (BM25 + FAISS) with Parent Chunk Expansion for 100% grounded answers.

LRU Cache: Remembers the last 3 papers to avoid repeated database calls.



🛠️ Tech Stack
Framework: LangGraph (for complex state-machine workflows)

Orchestration: LangChain

Vector DB: FAISS

Language: Python
