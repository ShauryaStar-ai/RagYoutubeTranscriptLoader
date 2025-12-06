from langchain_openai import ChatOpenAI
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
OPENAIKEY = os.getenv("OPEN_AI_API_KEY")
save_path = r"C:\Users\sonuh\code\shaurya\broCodeVideoEmbeddingFAISS"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAIKEY)

# Just reload the saved vectorstore
vectorstore = FAISS.load_local(
    save_path,
    embeddings,
    allow_dangerous_deserialization=True
)
model =ChatOpenAI(model="gpt-4", openai_api_key=OPENAIKEY)
retriever = vectorstore.as_retriever(
    search_type="similarity",   # 'similarity', 'similarity_score_threshold', 'mmr'
    search_kwargs={"k": 3}
)


# now I can send the docs from the retirver and the query to the model

prompt = PromptTemplate(
    template = """You are a very helpful AI assistant, 
    Given the context {docs} and the query {query} give me the answers if
     unsure always say "I don't know" and never say "I am not sure" or """,
    input_variables = ["docs", "query"]
)
query = "What is are primitative dataTypes"
retrieved_docs = retriever.invoke(query)
all_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

final_prompt = prompt.invoke({"docs": all_text, "query": query})
answer = model.invoke(final_prompt)
print(answer.content)