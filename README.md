# 1、基本环境
    Python版本： Python 3.10.12
    cuda版本： 12.2
    cudnn： 8.3.1
    显卡： RTX-3090-24G
    安装依赖包：
        pip install -r requirements.txt

# 2、训练脚本
	# 需要下载对应的模型文件：
	# LLM模型下载链接：https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct
	# Qwen2-vl下载链接： https://hf-mirror.com/Qwen/Qwen2-VL-7B-Instruct
	# 下载文件到本执行目录下
	# 训练数据来源（arxivQA采样2k + glm4v随机采样伪标50条）
    启动训练脚本
        sh train.sh

# 3、若过程有疑问/问题，请联系：1506025911@qq.com