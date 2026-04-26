import os
import time
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI, RateLimitError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Z.ai GLM-4 クライアント設定
client = OpenAI(
    api_key=os.getenv("ZAI_API_KEY"),
    base_url="https://api.z.ai/api/paas/v4/",
    timeout=60.0  # 60秒のタイムアウト
)

# 固定Systemプロンプト
SYSTEM_PROMPT = """あなたは「ナナカファームのクレソン料理専門AI」です。
熊本県でクレソンを生産するナナカファームが調査した、
世界19ジャンル・190品超のクレソン料理の知識を持っています。
飲食店の料理人・仕入れ担当者に役立つ情報を提供し、
具体的なレシピと調理のポイントを必ず示してください。
クレソンの栄養・効能にも触れ、
世界の料理文化の文脈でクレソンを説明してください。
クレソン料理に関係ない質問には「クレソン料理についてお聞きください」と答えてください。"""

@app.route("/")
def index():
    # セッション初期化
    session.clear()
    session["messages"] = []
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
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

        # API呼び出し用にSystemプロンプトを先頭に追加
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        # Z.ai API呼び出し（簡易リトライ付き）
        max_retries = 2

        for attempt in range(max_retries):
            try:
                # Z.ai API 呼び出し
                # max_tokensを大きくして推論プロセス＋実際の回答に十分なトークンを確保
                response = client.chat.completions.create(
                    model="glm-4.7",  # 小文字を使用
                    messages=api_messages,
                    temperature=0.7,
                    max_tokens=4096  # 推論プロセス＋回答に十分なトークン数
                )
                break
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    print(f"Rate limit error, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)  # 1秒待機
                else:
                    raise RateLimitError(
                        "Z.ai API が一時的に利用できません。しばらく待ってから再度お試しください。",
                        response=e.response,
                        body=e.body
                    )

        # レスポンスの検証
        if not response or not hasattr(response, 'choices'):
            raise ValueError("Invalid API response: missing 'choices'")

        if not response.choices or len(response.choices) == 0:
            raise ValueError("Invalid API response: empty 'choices'")

        if not response.choices[0].message:
            raise ValueError("Invalid API response: missing 'message'")

        # GLM-4 は reasoning_content または content のどちらかにレスポンスを返す
        message = response.choices[0].message
        assistant_message = message.content or getattr(message, 'reasoning_content', None)

        if not assistant_message:
            raise ValueError("Invalid API response: empty message content")

        # アシスタントの返答を履歴に追加
        messages.append({"role": "assistant", "content": assistant_message})

        # セッションを更新
        session["messages"] = messages
        session.modified = True

        return jsonify({
            "reply": assistant_message,
            "message_count": len(messages)
        })

    except RateLimitError as e:
        error_message = "Z.ai API が一時的に利用できません。サービスが過負荷状態です。数分待ってから再度お試しください。"
        print(f"Rate limit error in /chat: {str(e)}")
        return jsonify({"error": error_message}), 429
    except APITimeoutError as e:
        error_message = "AI の応答に時間がかかりすぎています。質問を短くするか、もう一度お試しください。"
        print(f"Timeout error in /chat: {str(e)}")
        return jsonify({"error": error_message}), 504
    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        print(f"Error in /chat: {error_message}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": error_message}), 500

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    session["messages"] = []
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
