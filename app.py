import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain # allows us to chat with the vector store

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.huggingface_hub import HuggingFaceHub

from htmlTemplates import css, bot_template, user_template
import os

def initialize_Pinecone():
  pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
  pc.create_index(
      name="new-pdfs-dimension",
      dimension=1536,  # Replace with your model dimensions
      metric="euclidean",  # Replace with your model metric
      spec=ServerlessSpec(
          cloud="aws",
          region="us-west-2"
      )
  )
  print('Pinecone index \'new-pdfs-dimension\' initialized successfully!')

def main():
  load_dotenv()
  #initialize_Pinecone()
  initialize_UI()

# embeddings - vector (/ number) representation of a text. It also contains info about the meaning
# of that text --> we can potentially find similar meaning text

def get_pdf_text(pdf_files):
  text = ""
  for pdf in pdf_files:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text

def convert_to_text_chunks(raw_text):
  text_splitter = CharacterTextSplitter(
      separator='\n',
      chunk_size=1000, # 1000 chars, the size of the chunk of text
      chunk_overlap=200, # to protect us from ending in an incorrect place - it will start before the previous chunk
      length_function=len
  )
  text_chunks = text_splitter.split_text(raw_text)
  return text_chunks

def get_vector_store(text_chunks):
  embeddings = OpenAIEmbeddings()
  #print('OpenAIEmbeddings initialized')

  index_name = "new-pdfs-dimension"

  #print('text_chunks: ' + str(type(text_chunks)))
  vector_store = PineconeVectorStore.from_texts(text_chunks, embeddings, index_name=index_name)
  return vector_store

  #query = "What is the chimp system?"
  #results = vector_store.similarity_search(query)
  #print(results[0].page_content)
  #print('results: ' + str(type(results)))
  #print(results)


def get_conversation_chain(vector_store):
  #llm = ChatOpenAI()
  llm = HuggingFaceHub(repo_id='openai-community/gpt2-xl', model_kwargs={'temperature': 0.5, 'max_length': 500})

  memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=vector_store.as_retriever(),
      memory=memory
  )
  return conversation_chain

def handle_user_question(user_question):
  response = st.session_state.conversation_chain({'question': user_question})
  # st.write(response) if we want to see the raw response
  st.session_state.chat_history = response['chat_history']

  for i, message in enumerate(st.session_state.chat_history):
    if i % 2 == 0:
      st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
      st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def initialize_UI():
    st.set_page_config(page_title="Chat with many PDFs :books:", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation_chain" not in st.session_state:
      st.session_state.conversation_chain = None # will reinitialize the conversation_chain
    if "chat_history" not in st.session_state:
      st.session_state.chat_history = None

    st.header("Chatting with many PDFs using AI")
    user_question = st.text_input("What do you want to know (from your PDFs)?")
    if user_question:
      handle_user_question(user_question)

    #st.write(user_template.replace("{{MSG}}", "Hello, AI robot!"), unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}", "Hello, human!"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("The used PDF documents")
        pdf_files = st.file_uploader("Upload your PDF files here & click on Process", accept_multiple_files=True)
        if st.button("Process"):
          with st.spinner("Processing..."):
            # get PDF text
            raw_text = get_pdf_text(pdf_files)
            #st.write(raw_text) if we want to see the loaded text

            # get the text chunks
            text_chunks = convert_to_text_chunks(raw_text)
            # st.write(text_chunks) if we want to see the chunks of text

            # create vector store embeddings
            vector_store = get_vector_store(text_chunks)

            # create conversation chain
            st.session_state.conversation_chain = get_conversation_chain(vector_store) # will allows us to generate the new messages of the conversation
            # will be available outside of the spinner and sidebar scope

if __name__ == '__main__':
   main()
