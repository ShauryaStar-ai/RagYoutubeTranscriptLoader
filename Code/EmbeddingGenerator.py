from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.document_loaders  import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
OPENAIKEY = os.getenv("OPEN_AI_API_KEY")  # the windows reteiver of the env variable

video_id = "_3ezSpJw2E8"
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=TGVLmr194DI&list=PLZPZq0r_RZOOj_NOZYq_R2PECIMglLemc&index=2")
transcript = loader.load()
transcript = transcript[0].page_content
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAIKEY
)

# splitting it into chunks to pass to the model
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Max characters per chunk
    chunk_overlap=100,    # Overlap to preserve context
    separators=[
        "\n\n",           # Prefer splitting by paragraphs
        "\n",             # Then by lines
        ".",              # Then by sentences
        " ",              # Then by words
        ""                # Fallback: split by characters
    ]
)
chunks_transcript = splitter.split_text(transcript)
#Setting up fiass and adding chuncks
docs = [Document(page_content=chunk) for chunk in chunks_transcript]
save_path = r"C:\Users\sonuh\code\shaurya\broCodeVideoEmbeddingFAISS"


# Now create the FAISS vectorstore
vectorstore = FAISS.from_documents(docs, embeddings) # makes the vector store

vectorstore.save_local(save_path) # saves the vector store
print(f"FAISS index saved to: {save_path}")
