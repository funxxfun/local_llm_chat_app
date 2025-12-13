import streamlit as st
from openai import OpenAI
import chromadb
from docx import Document
import requests

# ChromaDBの設定
DB_DIR = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_DIR)

if "collection" not in st.session_state:
    st.session_state.collection = chroma_client.get_or_create_collection(
      name="local_docs"
    )

# Ollamaからインストールしたモデルを使ったベクトル化関数
def ollama_embed(text):
    r = requests.post(
        "http://localhost:12000/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    data = r.json()
    return data["embedding"]


# Wordファイルを読み込む関数
def load_word_document(file):
  return "\n".join(para.text for para in Document(file).paragraphs)

# テキスト分割関数
def split_text(text):
  chunk_size = 200
  overlap = 50
  chunks = []
  start = 0
  while start < len(text):
      end = start + chunk_size
      chunks.append(text[start:end])
      start += chunk_size - overlap
  return chunks

## サイドバーの設定##
st.set_page_config(page_title="Locl LLM Chat")

st.sidebar.title("設定")
model = st.sidebar.text_input("モデル名", value="llama3.1:8b")
temperature = st.sidebar.slider("temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
system_prompt = st.sidebar.text_area(
  "System Prompt",
  "あなたは有能なアシスタントです。日本語で回答してください。",
)

# ワードファイルのアップロード
uploaded_files = st.sidebar.file_uploader(
  "Wordファイルをアップロード(.docx)",
  type=["docx"],
  accept_multiple_files=True
)

if st.sidebar.button("インデックス作成"):
  for file in uploaded_files:
    text = load_word_document(file)
    chunks = split_text(text)
    for i, chunk in enumerate(chunks):
      embed = ollama_embed(chunk)
      st.session_state.collection.add(
        documents=[chunk],
        embeddings=[embed],
        ids=[f"{file.name}_{i}"]
      )
  st.sidebar.success("インデックス作成が完了しました！")

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
  # ユーザーのプロンプト（入力）を表示する
  with st.chat_message("user"):
      st.write(prompt)

  # RAG検索
  query_embed = ollama_embed(prompt)
  results = st.session_state.collection.query(
    query_embeddings=[query_embed],
    n_results=2
  )

  if results["documents"]:
    context_text = "\n".join(results["documents"][0])
    rag_prompt = f"""
    以下は関連ドキュメントの抜粋です。
    {context_text}
    これらの情報を参考にして、次の質問に答えてください。
    {prompt}
    """

    final_prompt = rag_prompt
  else:
    final_prompt = prompt

  st.session_state.messages.append({"role": "user", "content": final_prompt})

  # システムプロンプトを最初に追加する
  if system_prompt.strip():
      messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
  else:
      messages = st.session_state.messages


  # LLMの応答を表示する
  # ストリーミング応答のためのプレースホルダーを作成
  with st.chat_message("assistant"):
    placeholder = st.empty()
    stream_response = ""
    stream =client.chat.completions.create(
      model=model,
      messages=messages,
      temperature=temperature,
      stream=True
    )

    # ストリーミングで応答を受け取りながら逐次的に表示する
    for chunk in stream:
      stream_response += chunk.choices[0].delta.content
      placeholder.write(stream_response)

  # 会話の履歴を追加する
  st.session_state.messages.append({"role": "assistant", "content": stream_response})