# 🎭 EmoGen - 实时情绪识别与响应系统

## 📋 前置要求
- 摄像头设备
- OpenAI API key
- Python 3.8+
- 人脸检测模型文件（需手动下载）

## 💻 安装
```bash
# 克隆仓库
git clone https://github.com/yourusername/emogen.git

# 创建conda环境
conda create --name emogen python=3.8
conda activate emogen

# 安装依赖
cd emogen
pip install -r requirements.txt

# 创建模型目录
mkdir -p model