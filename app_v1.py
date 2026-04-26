import os
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Z.ai GLM-4.7-Flash クライアント設定
client = OpenAI(
    api_key=os.getenv("ZAI_API_KEY"),
    base_url="https://api.z.ai/api/paas/v4/"
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

        # ユーザーメッセージを履歴に追加
        messages.append({"role": "user", "content": user_message})

        # API呼び出し用にSystemプロンプトを先頭に追加
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        # Z.ai API呼び出し
        print(f"Calling Z.ai API with model: glm-4.7-flash")
        response = client.chat.completions.create(
            model="glm-4.7-flash",
            messages=api_messages,
            temperature=0.7,
            max_tokens=2000
        )

        print(f"API Response: {response}")

        # レスポンスの検証
        if not response or not hasattr(response, 'choices'):
            raise ValueError("Invalid API response: missing 'choices'")

        if not response.choices or len(response.choices) == 0:
            raise ValueError("Invalid API response: empty 'choices'")

        if not response.choices[0].message:
            raise ValueError("Invalid API response: missing 'message'")

        assistant_message = response.choices[0].message.content

        if not assistant_message:
            raise ValueError("Invalid API response: empty message content")

        # アシスタントの返答を履歴に追加
        messages.append({"role": "assistant", "content": assistant_message})

        # セッションを更新
        session["messages"] = messages
        session.modified = True

        return jsonify({
            "response": assistant_message,
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
    session.clear()
    session["messages"] = []
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
