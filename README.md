# LongChainBQA
[LLM] LLM本地知识问答 V1.0.1 本地8G显存即可使用

因为上传大文件(lfs)需要付费，所以删除模型文件，chatglm2-6b-int4\pytorch_model.bin
该文件需要自己去huggingface官网下载:https://huggingface.co/THUDM/chatglm2-6b-int4/tree/main

代码配置：
--model:chatglm2-6b-int4
--code:chatglm3-6b
--vdb:faiss
--cuda:12.1
--python:3.8及以上

使用方法：
步骤一:根据上述地址下载pytorch_model.bin模型文件并放在目录下面: .\chatglm2-6b-int4\    linux系统下载：wget https://huggingface.co/THUDM/chatglm2-6b-int4/resolve/main/pytorch_model.bin?download=true
步骤二:安装相关依赖 pip install -r requirements.txt
步骤三:启动程序 gradio_server.py
步骤四:先上传LLM参考的本地文件(可以是任意文本类型,如:*.pdf、*.txt、*.doc等)
步骤五:向LLM提问问题
