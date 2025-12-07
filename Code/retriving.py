from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda
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


prompt = PromptTemplate(
    template = """You are a very helpful AI assistant, 
    Given the context {docs} and the query {query} give me the answers if
     unsure always say "I don't know" and never say "I am not sure" or """,
    input_variables = ["docs", "query"]
)

# Lambda to format docs
format_docs = RunnableLambda(lambda docs: "\n\n".join([doc.page_content for doc in docs]))

# Chain: retrieve -> format -> prompt -> LLM
retriever_chain = RunnableSequence(

        retriever,              # Step 1: get docs
        format_docs,            # Step 2: format docs
        prompt,                 # Step 3: fill prompt with formatted docs
        model                   # Step 4: call LLM
)

# Query
query = "What are primitive data types?"
answer = retriever_chain.invoke(query)
print(answer.content)