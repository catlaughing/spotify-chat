from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from src.ChatPinecone import ChatPinecone
from langchain_core.runnables.history import RunnableWithMessageHistory


import streamlit as st

st.set_page_config(page_title="Chatify: Spotify AI Chatbot", page_icon="🦜")
st.title("Chatify: Spotify AI Chatbot")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")

msgs = StreamlitChatMessageHistory("langchain_messages")
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

view_messages = st.expander("View the message contents in session state")

avatars = {"human": "user", "ai": "assistant"}
# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)


if prompt := st.chat_input(placeholder="What can we do to improve our apps?"):
    st.chat_message("user").write(prompt)
    # st.session_state.messages.append({"role": "user", "content": prompt})

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    if not pinecone_api_key:
        st.info("Please add your Pinecone API key to continue.")
        st.stop()

    ChatbotObj = ChatPinecone(pinecone_api_key, openai_api_key)
    chain_with_history = RunnableWithMessageHistory(
        ChatbotObj.chain,
        lambda session_id: msgs,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    with st.chat_message("assistant"):
        config = {"configurable": {"session_id": "any"}}
        response = chain_with_history.invoke({"question": prompt}, config)
        st.write(response["output"])

        context_content = "\n\n".join(
            f"**{cont.page_content}** - {cont.metadata['review_rating']}"
            for cont in response["context"]
        )

        input_prompt = ChatbotObj.prompt.format(
            context=context_content,
            question=prompt,
        )

        scores = ChatbotObj.scoring_chain.evaluate_strings(
            prediction=response["output"],
            input=input_prompt,
        )

        container = st.container(border=True)
        container_text = "\n\n".join(
            f"**{cont.page_content}** - {cont.metadata['review_rating']}"
            for cont in response["context"][:5]
        )
        container.write("**Review Examples**: \n\n" + container_text)

        container_2 = st.container(border=True)
        container_2.write("**Answer Scores and Reasoning**: \n\n" + scores["reasoning"])
