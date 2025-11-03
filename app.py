import streamlit as st

from agent_runner import agent_runner

st.set_page_config(page_title="💻 AI Ops Assistant", page_icon="🤖")
st.title("💬 AI Ops Assistant 聊天助手")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入问题..."):
    print(f"prompt: {prompt}")
    message = {"role": "user", "content": prompt}
    st.session_state.messages.append(message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            result = agent_runner.invoke({"messages": [message]})
            # print(f"result: {result}")

            # 查找 ToolMessage 的 content
            response = "没有找到工具执行结果"

            if "messages" in result:
                for msg in result["messages"]:
                    # 检查是否是 ToolMessage
                    if hasattr(msg, "__class__") and "ToolMessage" in str(
                        msg.__class__
                    ):
                        response = msg.content
                        break
                    # 如果没有 ToolMessage，使用 AIMessage 的内容
                    elif (
                        hasattr(msg, "__class__")
                        and "AIMessage" in str(msg.__class__)
                        and hasattr(msg, "content")
                    ):
                        if msg.content:  # 只有当 AIMessage 有内容时才使用
                            response = msg.content

            print(f"extracted response: {response}")
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
