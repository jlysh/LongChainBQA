class Config:
    llm_model_name = './chatglm2-6b-int4'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = './all-MiniLM-L6-v2'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = 'resource/faiss/'
    docs_path = 'resource/txt/'
    upload_path = 'upload' #临时上传文件存放目录
    server_name = '127.0.0.1' #ip
    server_port = 8088 #端口号
    