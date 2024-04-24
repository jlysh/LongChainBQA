from langchain_community.document_loaders import PyPDFLoader,UnstructuredWordDocumentLoader,UnstructuredExcelLoader,UnstructuredMarkdownLoader,TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from config import Config
import time
import os
import gradio as gr
import shutil

# import torch

# 加载LLM模型
tokenizer = AutoTokenizer.from_pretrained(Config.llm_model_name, trust_remote_code=True)
model = (
    AutoModel.from_pretrained(Config.llm_model_name, trust_remote_code=True)
    .half()
    .cuda()
)

model_name = Config.embedding_model_name
embeddings = HuggingFaceEmbeddings(model_name=model_name)


class RunLLM:
    def __init__(self):
        __path = os.path.dirname(__file__)
        self.upload_dir = os.path.join(__path, Config.upload_path)

    # 拆分文档
    def split_docs(self, documents, chunk_size=500, chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.docs = text_splitter.split_documents(documents)
        
    def load_doc(self,doc_path):
        try:
            # 获取文件扩展名
            _, file_extension = os.path.splitext(doc_path)
        except:
            raise FileNotFoundError("请检查文件后缀名是否正确,不支持无后缀名的文件")
        # 去除点号
        file_extension = file_extension[1:].lower()
        # 根据扩展名判断文件类型
        if file_extension in ['pdf']:
            loader = PyPDFLoader(file_path=doc_path,extract_images=True)
        elif file_extension in ['doc', 'docx']:
            loader = UnstructuredWordDocumentLoader(doc_path)
        elif file_extension in ['xlsx','xls']:
            loader = UnstructuredExcelLoader(file_path=doc_path,mode="elements")
        elif file_extension in ['txt']:
            loader = TextLoader(file_path=doc_path,encoding='utf-8')
        elif file_extension in ['md']:
            loader = UnstructuredMarkdownLoader(doc_path)
        else:
            raise FileNotFoundError("文件类型{}不被支持".format(file_extension))
        documents = loader.load()
        return documents
        
    def llm_model(self, query, history):
        query_similarity = self.vectorstore.similarity_search(query)
        context = [doc.page_content for doc in query_similarity]
        # 构造Promp
        prompt = f"已知信息: \n{context}\n根据已知信息回答问题: \n{query}"
        # prompt = f"In order to... They did... Experiment. What's the result? What's the conclusion"
        # llm生成回答
        result = model.chat(tokenizer, prompt, history=[])
        # print("AI回复:", result[0])

        for i in range(len(result[0])):
            time.sleep(0.03)
            yield "GPT: " + result[0][: i + 1]

    def process_doc(self, file):
        #若目录不存在，则创建目录
        if not os.path.exists(self.upload_dir):
            os.mkdir(self.upload_dir)
        #清空临时文件
        files = os.listdir(self.upload_dir)
        # 遍历文件列表
        for f in files:
            # 构建完整的文件路径
            file_path = os.path.join(self.upload_dir, f)
            # 检查是否为文件，不是文件（即文件夹）则跳过
            if os.path.isfile(file_path):
                # 删除文件
                os.remove(file_path)

        if hasattr(file,'name') == False:
            return '系统出现问题，请重新上传文件'
             
        # 从临时文件复制到指定路径
        file_name = os.path.basename(file.name)
        doc_path = os.path.join(self.upload_dir, file_name)
        shutil.copy(file.name, doc_path)
        yield (
            "{file_name} 已保存到路径{upload_dir}".format(
                file_name=file_name, upload_dir=self.upload_dir
            )
        )
        time.sleep(0.03)
        """数据处理，加载文档"""
        documents = self.load_doc(doc_path)
        
        # docs = split_docs(documents)
        """存入faiss数据库中"""
        # 将向量添加到数据库中
        self.vectorstore = FAISS.from_documents(
            documents=documents, embedding=embeddings
        )
        yield ("{file_name} 已加载到faiss向量数据库中".format(file_name=file_name))
        return doc_path

    def run(self):

        inputs = gr.components.File(label="上传文件")
        with gr.Blocks() as demo:
            # 将PDF文件加载到向量数据库中
            with gr.Column():
                gr.Interface(
                    fn=self.process_doc,
                    inputs=inputs,
                    outputs=gr.Textbox(),
                    title="本地知识库问答",
                )
            # 调用LLM基于PDF进行知识问答
            with gr.Column():
                gr.ChatInterface(self.llm_model)

        demo.queue().launch(server_name=Config.server_name, share=True, server_port=Config.server_port)


if __name__ == "__main__":
    runLLM = RunLLM()
    runLLM.run()
