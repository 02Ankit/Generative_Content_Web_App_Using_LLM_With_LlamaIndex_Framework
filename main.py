from dotenv import load_dotenv
import os
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.chat_engine.types import ChatMode
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.postprocessor import SentenceEmbeddingOptimizer
from node_postprocessors.duplicate_postprocessing import (
    DuplicateRemoverNodePostprocessor,
)

from llama_index.vector_stores import PineconeVectorStore
import pinecone
import streamlit as st



load_dotenv()

print("****Streamlit llmaindex Documentation Helper****")

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)
@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )

    pinecone_index = pinecone.Index(index_name="llamaindex-documantation-helper")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)



    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )

    # Query = "what is llamaIndex query engine?"
    # Query_engine = index.as_query_engine()
    # response = Query_engine.query(Query)
    # print(response)


index = get_index()

if "chat_engine" not in st.session_state.keys():
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=service_context.embed_model,
        percentile_cutoff=0.5,
        threshold_cutoff=0.7)

    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        verbose=True,
        node_postprocessors=[postprocessor, DuplicateRemoverNodePostprocessor()],
    )

st.set_page_config(
    page_title="Chat With Ankit Chat GPT Docs, powered by llamaindex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Chat with LlamaIndex docs 🦙 🦜🔗")

if "messages" not in st.session_state.keys():
    st.session_state.messages= [
        {
            "role": "assistant",
            "content": "Ask me a Question about llamaIndex's open source python Library?"
        }
    ]
if prompt := st.chat_input("your Question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=prompt)
            st.write(response.response)
            nodes = [node for node in response.source_nodes]
            for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                with col:
                    st.header(f"Source Node {i+1}: score= {node.score}")
                    st.write(node.text)

            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
            # import time
            # time.sleep(5)
