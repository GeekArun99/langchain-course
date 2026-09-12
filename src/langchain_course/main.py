import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

#env_variables
load_dotenv()

print("initializing the components...")

#llm-Brain
llm = ChatGroq(model = "qwen/qwen3.6-27b", temperature = 0)

#embedding-model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

#vectorstore_connection
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings,
)

#retriever_instance_for_vector_store
retriever = vectorstore.as_retriever(search_kwargs ={"k":3})

#prompt_template
prompt_template = ChatPromptTemplate.from_template(
    """Answer the questions based only on the following context :
    
    {context}

    Question : {question}

    Provide a detailed answer."""
)

#format_the_retrieved_chunks
def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

print("initializing the prompt template...")

#fucnction_to_create_langchain_chain
def create_retrieval_chain_with_lcel():
    """
        Create retrieval chain using LCEL (Langchain Expression Language)
        Returns a chain that can be invoked with {"question" : "..."}
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question") | retriever | format_docs
            )
        | prompt_template 
        | llm
        | StrOutputParser()
    )

    return retrieval_chain
    

if __name__ == "__main__":
    print("Retrieving...")

    #query 
    query = "What is Pinecone in Machine Learing.."

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answer questions about the text."),
        ("user", "{input}")
    ])

    print("initializing the chain...")

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question":query})
    print("\nAnswer: ")
    print(result_with_lcel)