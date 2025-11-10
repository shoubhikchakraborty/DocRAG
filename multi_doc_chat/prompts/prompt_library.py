from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualized_prompt_query= ChatPromptTemplate.from_messages([
    ("system", (
        """ Given the conversation history and the most recent user query, rewrite the query as a standalone question that amkes sense without relying on
        the previous content. Do not give the answer, only reformulate the question if necessary; otherwise, return it unchanged.
        """
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        """You are an assistant designed to answer questions based **only** on the provided CONTEXT.  
        - Use the context to answer; do NOT invent facts outside it.  
        - If the answer is not present in the CONTEXT, respond exactly: "Answer not present in documents".  
        - If you need clarification from the user, ask a clarifying question.  
        - Keep your answer concise (answer + brief reasoning) in fewer than three sentences."""
    )),
    # include chat history for conversational context if relevant
    MessagesPlaceholder("chat_history"),
    # inject the retrieved documents here so the LLM sees them
    ("system", "Context (use only this to answer):\n{context}"),
    # then the user's question
    ("human", "{input}"),
])


PROMPT_REGISTRY = {
    "contextualize_question": contextualized_prompt_query,
    "context_qa": context_qa_prompt,
}