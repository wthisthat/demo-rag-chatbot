import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# load env var
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX")
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-1.5-flash"
RETRIEVER_TOP_K = 5

# LangSmith configuration for tracing & debugging
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY']=LANGCHAIN_API_KEY
os.environ['LANGCHAIN_PROJECT']="maistorage_assessment"

# pc = Pinecone(api_key=PINECONE_API_KEY)

def init_pinecone_llm_embedding():
    """ Initialize embedding model, vector store, llm """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        )
    
    return vector_store,llm

def vector_retrieve_and_format_prompt(query, vector_store):
    """ Retrieve top K similar documents from vector database & format prompt for generation LLM"""
    results = vector_store.similarity_search_with_score(query=query,k=RETRIEVER_TOP_K)
    sources_metadata = []
    context = ''
    for i, (result, score) in enumerate(results, 1):
        """
        result.page_content: text
        result.metadata: text metadata
        score: similarity score
        """
        # minimum similarity score filtering
        if score > 0.5:
            context += f'Context #{i}: \n{result.page_content}\n\n'

            page = int(result.metadata.get('page'))
            metadata = {
                'page': page if page != 0 else 1,
                'source': result.metadata.get('source').split('/')[1],
                'text': result.page_content
            }
            sources_metadata.append(metadata)

    prompt_template = """You are a knowledgeable and professional corporate assistant for Company ABC. Your role is to provide accurate, up-to-date information on company policies, procedures, and guidelines from the employee handbook. \
Ensure your answers are compliant with the company's official documents and legal requirements. You must remain neutral, avoiding opinions, and ensure that all responses align with the company's values and ethical standards.

If you encounter a question that is greeting or pleasantries, you may answer it accordingly without referencing the context.
If you encounter a question outside the scope of your knowledge or unrelated to company policy (e.g., questions about public figures, external companies, or trivia), politely inform the user that you are only able to assist with company-related inquiries. \
Example response to an irrelevant question: 'I'm here to assist with questions related to Company ABC policies and procedures. For other inquiries, please refer to appropriate sources.'

QUESTION: {question}
CONTEXT:
{context}

ANSWER:"""

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

    return prompt, context, sources_metadata

def answer_generation(query, vector_store, llm):
    """ Generate answer for the query based on the prompt using LLM"""
    llm_prompt, context, source_metadata = vector_retrieve_and_format_prompt(query, vector_store)
    chain = llm_prompt | llm
    streaming_result = chain.stream({'question': query, 'context': context})

    return streaming_result, source_metadata