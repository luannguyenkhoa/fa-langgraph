import traceback
import os
import uuid
if os.environ.get("ENV", "LOCAL") == "LOCAL":
    from dotenv import load_dotenv
    load_dotenv()

import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig
from app.assistant import invoke, build_workflow, clear_memory
from util.callback import FinalStreamingStdOutCallbackHandler
from util.utils import strip_final_answer

st.set_page_config(page_title="Financial Assistant", page_icon="🍻")
st.title("🍻 Financial Assistant")
st.write("The current user by auth token feature is not implemented yet. So, please add the prefix: `I'm Maureen Lee.` to questions related to user's data if didn't ask.\n\
         The database is just quite simple with 2 tables only, you can see fully [here](https://drive.google.com/drive/folders/11MHq8C_rAwlikRyFu4DW4Hc0F0ZUebHz?usp=sharing)")

session_id = "user_1" # uuid.uuid4().hex
workflow, history = build_workflow(session_id)

print(f"history: {history}")

# Set up memory
st.session_state.messages = [m.to_dict() for m in history]

if len(st.session_state.messages) == 0:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I'm a delightful assistant. How is it going?"}]

# For reset chat history
def on_reset():
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I'm a delightful assistant. How is it going?"}]
    clear_memory(session_id)

col1, col2 = st.columns([3, 1])
with col1:
    view_messages = st.expander("View the message contents in session state")

with col2:
    st.button("Reset chat history", on_click=on_reset)

# Render current messages from StreamlitChatMessageHistory
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input(placeholder="Hi, I'm Maureen Lee. How is my financial health currently?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Replace FinalStreamingStdOutCallbackHandler to StreamlitCallbackHandler if just want to display the final answer only
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False) # FinalStreamingStdOutCallbackHandler()
            cfg = RunnableConfig(callbacks=[], configurable={ "thread_id": session_id })
            try:
                response = invoke(workflow=workflow, question=prompt, cfg=cfg)
                st.markdown(response.get("output"))
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                response = { "content": "Something went wrong. Please ask me again." }
                st.markdown(response.get("output"))
    message = {"role": "assistant", "content": response.get("output")}
    st.session_state.messages.append(message)
# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.messages)
