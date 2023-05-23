# %%
import os
from typing import Any, List, Optional

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


class FaissWithScore(FAISS):
    """
    FaissWithScore は、 FAISS を継承したクラスです。FAISS には、検索結果のスコアを取得する機能がないため、
    このクラスを使用することで、検索結果のスコアを取得できるようになります。
    """

    def similarity_search(self, query: str, k: int = 5, **kwargs: Any) -> List[Document]:
        """
        指定したクエリに最も近いドキュメントを FAISS から取得します。
        このメソッドは、 FAISS の similarity_search メソッドをオーバーライドしています。
        """
        # 検索結果のスコアを取得する
        docs_with_score = super().similarity_search_with_score(query, k)

        # 検索結果のスコアをドキュメントに追加する
        docs = []
        for doc, score in docs_with_score:
            doc.metadata["score"] = score
            docs.append(doc)
        return docs


class Agent:
    """
    Agent は、質問応答 (QA) システムを実装したクラスです。これは、質問を受け取り、関連する情報を検索し、最適な回答を生成します。
    FAISS をベクトルストアとして使用し、環境変数から OpenAI の API キーと環境を取得します。

    主なメソッド:
    - initialize_chain: FAISS を初期化し、QA チェーンを作成します。
    - run: 質問に対する回答を取得します。必要に応じて QA チェーンを初期化します。

    使用例:
    qa_agent = Agent()
    result = qa_agent.run("質問のテキスト")
    """

    def __init__(
        self, vector_store_folder_path: str = "vector_store_faiss", vector_store_index_name: str = "index"
    ) -> None:
        """
        Agent インスタンスを初期化します。qa_chain は None に設定されます。
        """
        self.qa_chain = None
        self.vector_store_folder_path = vector_store_folder_path
        self.vector_store_index_name = vector_store_index_name

    def initialize_chain(self):
        """
        FAISS を初期化し、QA チェーンを作成します。環境変数から OpenAI の API キーと環境を取得し、
        それらを使用して FAISS を初期化します。その後、ベクトルストアを用いた検索機能を持つ QA チェーンを作成します。
        """
        # vectore store として FAISS を使用
        embeddings = OpenAIEmbeddings()

        # 環境変数から OpenAI の API キーと環境を取得
        try:
            self.openai_api_key = os.environ["OPENAI_API_KEY"]
        except:
            raise Exception("OPENAI_API_KEY を環境変数に設定してください")

        # vectore store を初期化
        vectorstore = FaissWithScore.load_local(
            folder_path=self.vector_store_folder_path,
            embeddings=embeddings,
            index_name=self.vector_store_index_name,
        )

        # qa chain を作成
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4", temperature=0),
            chain_type="stuff",
            retriever=retriever,
            verbose=True,
            return_source_documents=True,
        )
        self.qa_chain = qa_chain

    def run(self, prompt):
        """
        質問に対する回答を取得します。QA チェーンが初期化されていない場合、initialize_chain メソッドを使用して初期化します。
        その後、質問を QA チェーンに渡し、回答と関連する情報を含む結果を返します。

        Args:
            prompt (str): 質問のテキスト

        Returns:
            dict: 回答テキストと関連情報を含む辞書
        """
        # qa chain がなければ作成
        if self.qa_chain is None:
            self.initialize_chain()

        # 質問に対する回答を取得する
        result = {}
        answer = self.qa_chain(prompt)
        result["answer_text"] = answer["result"]
        result["source_documents"] = [
            {
                "title": source_doc.metadata["title"],
                "url": source_doc.metadata["source"],
                "score": source_doc.metadata["score"],
            }
            for source_doc in answer["source_documents"]
        ]
        return result


if __name__ == "__main__":
    agent = Agent()
    answer = agent.run("印刷関連事例")
    print(answer)

# %%
