# saiteki_case_bot
# 使い方
* 環境変数を設定する

```bash
# OpenAI
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Slack Bot
export SLACK_BOT_TOKEN=xoxb-0000000000000-0000000000000-000000000000000000000000
export SLACK_APP_TOKEN=xapp-0-00000000000-0000000000000-0000000000000000000000000000000000000000000000000000000000000000
```

* python 準備。version は 3.8 以上で。

* ライブラリインストール

```bash
pip install -r requirements.txt
```

* 最適ワークスのマニュアルを vector store に保管する

```bash
python store_to_vectordb.py
```

* bot を動かす

```bash
python bot.py
```
