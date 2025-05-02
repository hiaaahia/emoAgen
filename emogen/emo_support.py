from .config import client
import pyttsx3
import time

class EmotionSupportSystem:
    def __init__(self, 
                 voice_speed=150,
                 voice_volume=1.0):
        """
        情绪支持系统
        
        参数:
            voice_speed: 语音速度(默认150)
            voice_volume: 语音音量(0.0-1.0)
        """
        # 初始化OpenAI客户端
        self.client = client
        
        # 初始化语音引擎
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', voice_speed)
        self.engine.setProperty('volume', voice_volume)
        
        # 情绪提示模板
        self.emotion_prompts = {
            'happy': "用户看起来很开心，用温暖亲切的语气说一句简短鼓励的话，不超过15个单词",
            'sad': "用户看起来有些难过，用温柔安慰的语气说一句关心的话，不超过20个单词",
            'angry': "用户似乎生气了，用平静缓和的语气说一句帮助冷静的话，不超过20个单词",
            'surprise': "用户显得很惊讶，用好奇友好的语气问一个简短的问题，不超过15个单词",
            'fear': "用户表现出恐惧，用 reassuring 的语气说一句安慰的话，不超过20个单词",
            'disgust': "用户显得反感，用中立但关心的语气建议转移注意力，不超过20个单词",
            'neutral': "用户表情平静，用友好的语气说一句日常问候，不超过15个单词",
            'default': "用户情绪状态不明显，用友好的语气说一句通用的问候，不超过15个单词"
        }
        
        # GPT配置
        self.gpt_model = "gpt-4o"
        self.max_tokens = 50
    
    def generate_response(self, emotion):
        """
        根据情绪生成GPT回应
        
        参数:
            emotion: 检测到的情绪
            
        返回:
            GPT生成的回应文本或None(如果出错)
        """
        prompt = self.emotion_prompts.get(emotion.lower(), self.emotion_prompts['default'])
        
        try:
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": "你是一个古灵精怪的情绪支持助手，用简短、口语化的句子直接回应用户"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成回应时出错: {e}")
            return None
    
    def speak(self, text):
        """
        用语音输出文本
        
        参数:
            text: 要朗读的文本
        """
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"语音输出时出错: {e}")
    
    def process_emotion(self, emotion):
        """
        处理检测到的情绪(生成回应并语音输出)
        
        参数:
            emotion: 检测到的情绪
            
        返回:
            生成的回应文本或None
        """
        print(f"[情绪检测] 当前情绪: {emotion}")
        
        response = self.generate_response(emotion)
        if response:
            print(f"[AI回应] {response}")
            self.speak(response)
        
        return response
    
    def __del__(self):
        """清理资源"""
        try:
            self.engine.stop()
        except:
            pass


# 使用示例
if __name__ == "__main__":
    support = EmotionSupportSystem(voice_speed=160)
    
    # 模拟情绪输入
    test_emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral']
    
    for emotion in test_emotions:
        print(f"\n测试情绪: {emotion.upper()}")
        support.process_emotion(emotion)
        time.sleep(2)  # 间隔避免语音重叠