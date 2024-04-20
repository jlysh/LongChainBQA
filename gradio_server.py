from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
import time
import os
import gradio as gr
import shutil
# import torch

# 加载LLM模型
tokenizer = AutoTokenizer.from_pretrained("./chatglm2-6b-int4", trust_remote_code=True)
model = (
    AutoModel.from_pretrained("./chatglm2-6b-int4", trust_remote_code=True)
    .half()
    .cuda()
)

model_name = "./all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)


class RunLLM:

    def __init__(self):
        __path = os.path.dirname(__file__)
        self.upload_dir = os.path.join(__path, "upload")

    # 拆分文档
    def split_docs(self, documents, chunk_size=500, chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.docs = text_splitter.split_documents(documents)

    def llm_model(self, query, history):
        query_similarity = self.vectorstore.similarity_search(query)
        context = [doc.page_content for doc in query_similarity]
        # 构造Promp
        prompt = f"已知信息: \n{context}\n根据已知信息回答问题: \n{query}"
        # prompt = f"In order to... They did... Experiment. What's the result? What's the conclusion"
        # llm生成回答
        result = model.chat(tokenizer, prompt, history=[])
        print("AI回复:", result[0])

        for i in range(len(result[0])):
            time.sleep(0.03)
            yield "AI: " + result[0][: i + 1]

    def process_pdf(self, file):
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
        pdf_path = os.path.join(self.upload_dir, file_name)
        shutil.copy(file.name, pdf_path)
        yield (
            "{file_name} 已保存到路径{upload_dir}".format(
                file_name=file_name, upload_dir=self.upload_dir
            )
        )
        time.sleep(3)
        """数据处理"""

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        # docs = split_docs(documents)
        """存入faiss数据库中"""
        # 将向量添加到数据库中
        self.vectorstore = FAISS.from_documents(
            documents=documents, embedding=embeddings
        )
        yield ("{file_name} 已加载到faiss向量数据库中".format(file_name=file_name))
        return pdf_path

    def run(self):

        inputs = gr.components.File(label="上传文件")
        with gr.Blocks() as demo:
            # 将PDF文件加载到向量数据库中
            with gr.Column():
                gr.Interface(
                    fn=self.process_pdf,
                    inputs=inputs,
                    outputs=gr.Textbox(),
                    title="本地知识库问答",
                )
            # 调用LLM基于PDF进行知识问答
            with gr.Column():
                gr.ChatInterface(self.llm_model)

        demo.queue().launch(server_name="127.0.0.1", share=True, server_port=8080)


if __name__ == "__main__":
    runLLM = RunLLM()
    runLLM.run()
