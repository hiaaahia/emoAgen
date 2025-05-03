from .config import client  # 从本地config模块导入已配置的client

class SentimentAnalyzer:
    """
    精简情绪分析器，只返回情绪分析结果字符串
    """
    
    def __init__(self, model="gpt-3.5-turbo", temperature=0.3):
        """
        初始化情绪分析器
        
        参数:
            model: 使用的模型名称 (默认: "gpt-3.5-turbo")
            temperature: 生成结果的随机性 (默认: 0.3)
        """
        self.model = model
        self.temperature = temperature
        self.system_prompt = (
            "你是一个专业的情绪分析助手。请用最简短的单个词语回答用户输入的主要情绪"
            "（只输出以下之一：快乐、悲伤、愤怒、惊讶、恐惧、厌恶、中立）"
            "不要包含任何解释或额外文本"
        )
    
    def analyze(self, text):
        """
        分析给定文本的情绪
        
        参数:
            text: 要分析的文本
            
        返回:
            str: 情绪标签（快乐/悲伤/愤怒/惊讶/恐惧/厌恶/中立）或错误信息
        """
        if not text.strip():
            return "错误：输入文本不能为空"
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=self.temperature,
                max_tokens=10  # 限制输出长度
            )
            
            # 清理结果，只保留第一个有效情绪词
            result = response.choices[0].message.content.strip()
            emotions = ["快乐", "悲伤", "愤怒", "惊讶", "恐惧", "厌恶", "中立"]
            for emotion in emotions:
                if emotion in result:
                    return emotion
            
            return "中立"  # 默认返回中立
            
        except Exception as e:
            return f"错误：{str(e)}"


# 使用示例
if __name__ == "__main__":
    # 实例化分析器
    analyzer = SentimentAnalyzer()
    
    # 测试分析
    test_texts = [
        "我今天中彩票了！太开心了！",
        "我的钱包被偷了，气死我了",
        "刚刚看到一只大蜘蛛，吓死我了",
        "这是个普通的陈述句",
        ""  # 空输入测试
    ]
    
    for text in test_texts:
        print(f"文本: '{text}'")
        print(f"情绪: {analyzer.analyze(text)}\n")