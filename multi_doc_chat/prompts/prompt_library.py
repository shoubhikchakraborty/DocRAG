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


context_qa_prompt= ChatPromptTemplate.from_messages([
    ("system", (
        """ You are an assistant designed to answer question based on the provided context.Rely only on the context provided to you. If answer is not present in the
        context respond the answer by your own. If you are confused reagarding the user question, ask to clarify it. Keep your answer concise along with reasoning in less than three sentences.
        """
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


PROMPT_REGISTRY = {
    "contextualize_question": contextualized_prompt_query,
    "context_qa": context_qa_prompt,
}