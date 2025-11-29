import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Locl LLM Chat")

st.sidebar.title("設定")
model = st.sidebar.text_input("モデル名", value="llama3.1:8b")
temperature = st.sidebar.slider("temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
system_prompt = st.sidebar.text_area(
  "System Prompt",
  "あなたは有能なアシスタントです。日本語で回答してください。",
)

# タイトル
st.title("ふみの相棒 〜Locl LLM Chat〜")

# 会話の履歴を保存するためのセッションステート
if "messages" not in st.session_state:
    st.session_state.messages = []

# 会話の履歴をリセットするボタン
if st.sidebar.button("会話をリセット"):
    st.session_state.messages = []

# 会話の履歴を表示する
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("質問をどうぞ！")

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:12000/v1"
)

if prompt:
  st.session_state.messages.append({"role": "user", "content": prompt})
  # ユーザーのプロンプト（入力）を表示する
  with st.chat_message("user"):
      st.write(prompt)

  # システムプロンプトを最初に追加する
  if system_prompt.strip():
      messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
  else:
      messages = st.session_state.messages

  response =client.chat.completions.create(
      model=model,
      messages=messages,
      temperature=temperature
  )

  # LLMの応答を表示する
  with st.chat_message("assistant"):
    st.write(response.choices[0].message.content)

  # 会話の履歴を追加する
  st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})