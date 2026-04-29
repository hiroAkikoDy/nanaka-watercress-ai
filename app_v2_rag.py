import os
import sys
import time
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# .envファイルから環境変数を読み込む
load_dotenv()

print("=" * 80)
print("【Neo4j RAGシステムを初期化中...】")
print("=" * 80)

# Flask アプリケーションの設定
app = Flask(__name__)
app.secret_key = os.urandom(24)

# グローバル変数としてRAGコンポーネントを保持
db = None
retriever = None
rag_chain = None

# BtoC向けSystemプロンプト
SYSTEM_PROMPT_TEMPLATE = """あなたは「ナナカファームのクレソン料理アドバイザー」です。
熊本の清らかな水で育てたクレソンを手に取ったあなたに、
今夜の食卓で使いこなせるレシピをご提案します。

【得意なこと】
・家庭で作れる具体的なレシピと調理のポイント
・冷蔵庫にある食材とクレソンの組み合わせ提案
・余ったクレソンの翌日活用法
・クレソンの栄養・保存方法のアドバイス
・世界19ジャンル190品超のクレソン料理の知識

【回答スタイル】
・家庭料理レベルでわかりやすく説明する
・材料は身近なスーパーで手に入るものを使う
・クレソン料理に関係ない質問には
  「クレソンの使い方についてお気軽にどうぞ😊」と答える

【参考データ】
{context}

質問: {question}

回答:"""


def format_docs(docs):
    """Neo4jから取得したDocumentを整形する"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"{i}. {doc.page_content}")
        formatted.append(f"   - 地域: {doc.metadata.get('region', '不明')}")
        formatted.append(f"   - 季節: {doc.metadata.get('season', '不明')}")
        formatted.append(f"   - 用途: {doc.metadata.get('use_case', '不明')}")
    return "\n".join(formatted)


def initialize_rag_system():
    """アプリ起動時に1回だけ実行されるRAGシステムの初期化"""
    global db, retriever, rag_chain

    try:
        # Neo4jVectorストアに接続（Hybrid Search有効）
        print("Neo4jに接続中...")
        db = Neo4jVector.from_existing_index(
            OpenAIEmbeddings(),
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name="watercress_index",
            keyword_index_name="watercress_keyword_index",
            search_type="hybrid",
            database=os.getenv("NEO4J_USERNAME"),  # Aura Free特有の設定
        )
        print("✓ Neo4j接続成功")

        # Retrieverを作成
        retriever = db.as_retriever(search_kwargs={"k": 3})
        print("✓ Retriever作成完了")

        # LLM: Z.ai GLM-4.7
        print("LLMを初期化中...")
        llm = ChatOpenAI(
            model="glm-4.7",
            openai_api_key=os.getenv("ZAI_API_KEY"),
            openai_api_base="https://api.z.ai/api/paas/v4/",
            temperature=0.7,
            timeout=60.0,
            max_tokens=4096,  # GLM-4.7の推論モード対策
        )
        print("✓ LLM初期化完了")

        # プロンプトテンプレート
        prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)

        # RAGチェーン（LCEL）を構築
        print("RAGチェーンを構築中...")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("✓ RAGチェーン構築完了")

        print("=" * 80)
        print("【Neo4j RAGシステムの初期化が完了しました】")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"✗ RAGシステムの初期化に失敗しました: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False


# アプリ起動時にRAGシステムを初期化
if not initialize_rag_system():
    print("RAGシステムの初期化に失敗したため、アプリケーションを終了します。")
    sys.exit(1)


@app.route("/")
def index():
    """トップページ"""
    # セッション初期化
    session.clear()
    session["messages"] = []
    return render_template("index_v2.html")


@app.route("/chat", methods=["POST"])
def chat():
    """チャットエンドポイント（RAG対応）"""
    try:
        data = request.json
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "メッセージが空です"}), 400

        # セッションから会話履歴を取得
        if "messages" not in session:
            session["messages"] = []

        messages = session["messages"]

        # 会話履歴を最新10件に制限（処理時間短縮のため）
        if len(messages) > 10:
            messages = messages[-10:]

        # ユーザーメッセージを履歴に追加
        messages.append({"role": "user", "content": user_message})

        # RAGチェーンで質問に回答（簡易リトライ付き）
        max_retries = 2
        assistant_message = None
        source_docs = []

        for attempt in range(max_retries):
            try:
                # 検索されたドキュメントを取得（出典表示用）
                source_docs = retriever.invoke(user_message)

                # RAGチェーンで質問に回答
                response = rag_chain.invoke(user_message)

                # GLM-4.7の推論モード対策
                if hasattr(response, 'content'):
                    assistant_message = response.content or getattr(response, 'reasoning_content', None)
                else:
                    assistant_message = response

                if assistant_message:
                    break

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"エラーが発生、リトライ中... (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(1)
                else:
                    raise

        if not assistant_message:
            raise ValueError("AIからの応答を取得できませんでした")

        # アシスタントの返答を履歴に追加
        messages.append({"role": "assistant", "content": assistant_message})

        # セッションを更新
        session["messages"] = messages
        session.modified = True

        # 出典情報を整形
        sources = []
        for doc in source_docs:
            sources.append({
                "content": doc.page_content,
                "region": doc.metadata.get("region", "不明"),
                "season": doc.metadata.get("season", "不明"),
                "use_case": doc.metadata.get("use_case", "不明"),
            })

        return jsonify({
            "reply": assistant_message,
            "sources": sources,
            "message_count": len(messages)
        })

    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        print(f"Error in /chat: {error_message}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": error_message}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """セッションリセット"""
    session.clear()
    session["messages"] = []
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nFlaskアプリを起動します (ポート: {port})")
    app.run(host="0.0.0.0", port=port, debug=False)
