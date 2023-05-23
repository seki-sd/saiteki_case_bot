# %%
import gc
import os
import shutil
import time
import urllib.parse

import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


class VectorStore:
    """
    VectorStore クラスは、 FAISS を使用してドキュメントのベクトル化とベクトルストアへの保存を処理します。

    主なメソッド:
    - store_to_vectoredb: ドキュメントをベクトル化し、ベクトルストアに保存します。
    """

    def __init__(self, folder_path: str = "vector_store_faiss", index_name: str = "index") -> None:
        """"""
        self.folder_path = folder_path
        self.index_name = index_name

    def delete_all_indexes(self):
        """
        すべてのインデックスを削除します。
        """
        if os.path.isdir(self.folder_path):
            shutil.rmtree(self.folder_path)

    def store_to_vectoredb(self, documents):
        """
        documents をベクトル化し、ベクトルストアに保存します。

        Args:
            documents (list): Document オブジェクトのリスト
        """
        # documents をベクトル化して vectore store に保存する
        embeddings = OpenAIEmbeddings()
        vectorestore = FAISS.from_documents(documents, embeddings)
        vectorestore.save_local(self.folder_path, self.index_name)
        del vectorestore
        gc.collect()
        print(f"Saved vectorstore to {self.folder_path}, with index name {self.index_name}.")


class SaitekiManualHandler:
    """
    SaitekiManualHandler クラスは、Saiteki サポートサイトから記事を取得し、ドキュメントを生成、分割する機能を提供します。

    主なメソッド:
    - generate_document: 与えられた URL からドキュメントを生成します。
    - split_documents: ドキュメントを分割します。
    - get_documents_from_urls: 与えられた URL からドキュメントを取得します。
    """

    def __init__(self) -> None:
        pass

    def _request(self, url):
        """
        url にアクセスし、取得した HTML を解析して BeautifulSoup オブジェクトを返します。

        Args:
            url (str): アクセスする URL

        Returns:
            BeautifulSoup: 解析された HTML の BeautifulSoup オブジェクト
        """
        # url にアクセスする
        # headers は zendesk が定めるものを使う
        print(f"Accessing {url} ...")
        res = requests.get(url=url, headers={"user-agent": "Zendesk/External-Content"})
        soup = BeautifulSoup(res.text, "html.parser")
        # DDos対策
        time.sleep(1)
        return soup

    def _get_page_urls(self, url):
        """
        与えられた URL から、関連するページの URL リストを取得します。

        Args:
            url (str): 基本となる URL

        Returns:
            list: 関連するページの URL リスト
        """

        # 与えられた url にアクセス
        soup = self._request(url)
        # その中で、食わせたいページの path を取得する
        paths = []
        for x in soup.find_all("a", class_="u-w-12 u-h-12"):
            paths.append(x.get("href"))
        # path から url を取得する
        url_list = paths
        return url_list

    def generate_document(self, url):
        """
        与えられた URL にアクセスし、ページタイトルと本文を取得して Document オブジェクトを生成します。

        Args:
            url (str): 記事の URL

        Returns:
            Document: 生成された Document オブジェクト
        """
        # 与えられた url にアクセス
        soup = self._request(url)
        # ページタイトルを取得（タイトル - ユーザ名）
        title = soup.find("h1", class_="p-news-singleTitle M:u-size30 u-size24 u-700 u-color-blue_2 u-lh14 u-mb25").text
        author = soup.find("ul", class_="c-list u-flex u-items-center u-wrap")
        username = author.find("span", class_="u-size14").text
        title = f"{title} - {username}"
        # soup から div.article-body を取得する
        body_text = soup.find("div", class_="c-editor u-mb75").text
        # ページ本文から document を作る
        metadata = {"source": url, "title": title}
        document = Document(page_content=body_text, metadata=metadata)
        return document

    def _generate_documents(self, page_urls):
        """
        与えられたページの URL リストから Document オブジェクトのリストを生成します。

        Args:
            page_urls (list): ページの URL リスト

        Returns:
            list: Document オブジェクトのリスト
        """
        print(f"Generating documents ...")
        documents = []
        for url in page_urls:
            documents.append(self.generate_document(url))
        return documents

    def split_documents(self, documents):
        """
        文書を分割します。

        Args:
            documents (list): Document オブジェクトのリスト

        Returns:
            list: 分割された Document オブジェクトのリスト
        """
        print(f"Splitting documents ...")
        # documents を分割する
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
        )
        splitted_documents = splitter.split_documents(documents)
        return splitted_documents

    def get_documents_from_urls(self, urls):
        """
        与えられた URL リストから、関連するページと記事の URL を取得し、それらのドキュメントを生成、分割して返します。

        Args:
            urls (list): URL のリスト

        Returns:
            list: Document オブジェクトのリスト
        """
        # urls それぞれから、食わせたいページの url を取得する
        content_urls = []
        for url in urls:
            content_urls += self._get_page_urls(url)

        # documents を作る
        documents = self._generate_documents(content_urls)

        # documents を分割する
        splitted_documents = self.split_documents(documents)

        return splitted_documents


if __name__ == "__main__":
    # 与えた url から順に辿って、食わせたいページの document を取得する
    # これらのページはサポートページトップの「最適ワークス」「サービスマネージャー」「よくある質問(FAQ)」のページ
    root_urls = [
        "https://saiteki.works/case_study/",
    ]
    handler = SaitekiManualHandler()
    documents = handler.get_documents_from_urls(root_urls)

    # document をベクトル化して vectore store に保存する
    store = VectorStore()
    store.delete_all_indexes()
    store.store_to_vectoredb(documents)
