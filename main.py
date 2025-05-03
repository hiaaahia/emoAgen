import threading
import queue
import time
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv
import os

# 假设这些是你已有的分析器
from emogen.emo_text_detection import SentimentAnalyzer
from emogen.emo_detection import RealTimeEmotionRecognizer
from emogen.config import client
# 加载环境变量
load_dotenv()

class EmotionAIChatbot:
    def __init__(self):
        # 初始化分析器
        self.text_analyzer = SentimentAnalyzer()
        self.face_recognizer = RealTimeEmotionRecognizer(show_video=False)
        
        # OpenAI客户端
        self.client = client
        
        # 情绪数据存储
        self.emotion_queue = queue.Queue()
        self._emotion_thread_running = False
        self._lock = threading.Lock()
        
        # 对话历史
        self.conversation_history = [
            {"role": "system", "content": 
            """你是一个擅长捕捉用户情绪的聊天大师，能根据用户的文本情绪和面部表情调整回复。用积极乐观带点幽默的语气回答。用中文回答。"""}
        ]
        
        # 情绪映射
        self.emotion_map = {
            "happy": "开心",
            "sad": "悲伤",
            "angry": "愤怒",
            "surprise": "惊讶",
            "fear": "恐惧",
            "disgust": "厌恶",
            "neutral": "中立"
        }

    def start_emotion_capture(self):
        """启动情绪捕捉线程"""
        with self._lock:
            if self._emotion_thread_running:
                return
            
            self._emotion_thread_running = True
            self.emotion_thread = threading.Thread(
                target=self._capture_emotions,
                daemon=True
            )
            self.emotion_thread.start()
            # print("摄像头已开启，开始捕捉表情情绪...")

    def _capture_emotions(self):
        """持续捕捉情绪到队列"""
        try:
            for emotion in self.face_recognizer.start():
                if not self._emotion_thread_running:
                    break
                if emotion.lower() not in ("no face", "error"):
                    self.emotion_queue.put(emotion.lower())
        except Exception as e:
            print(f"情绪捕捉出错: {str(e)}")
        finally:
            self.face_recognizer.stop()

    def stop_emotion_capture(self):
        """停止情绪捕捉并返回主要情绪"""
        with self._lock:
            if not self._emotion_thread_running:
                return "neutral"
            
            self._emotion_thread_running = False
            if hasattr(self, 'emotion_thread'):
                self.emotion_thread.join(timeout=1.0)
            
            # 统计队列中的情绪
            emotions = []
            while not self.emotion_queue.empty():
                emotions.append(self.emotion_queue.get())
            
            if not emotions:
                return "neutral"
            
            # 返回出现最多的情绪
            most_common = Counter(emotions).most_common(1)
            return most_common[0][0] if most_common else "neutral"

    def analyze_text_sentiment(self, text):
        """分析文本情绪"""
        try:
            return self.text_analyzer.analyze(text)
        except Exception as e:
            print(f"文本分析出错: {str(e)}")
            return "中立"

    def generate_response(self, text, text_emotion, facial_emotion):
        """生成AI回复"""
        # 构建情绪提示
        prompt = (
            f"用户文本输入: {text}\n"
            f"文本情绪: {text_emotion}\n"
            f"面部表情: {self.emotion_map.get(facial_emotion, facial_emotion)}\n"
            "请根据以上信息，用你认为最合适的方式回复用户。"
        )
        
        # 添加到对话历史
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=1000
            )
            
            ai_reply = response.choices[0].message.content
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_reply
            })
            
            return ai_reply
        except Exception as e:
            print(f"AI回复生成错误: {str(e)}")
            return "抱歉，我暂时无法回应。请稍后再试。"

    def run(self):
        """主运行循环"""
        try:
            print("情绪树洞已启动(输入'quit'退出)")
            
            while True:
                # 开始输入前启动情绪捕捉
                self.start_emotion_capture()
                
                # 获取用户输入
                user_input = input("\n你说: ")
                if user_input.lower() == 'quit':
                    break
                
                # 输入结束，停止捕捉并获取主要情绪
                facial_emotion = self.stop_emotion_capture()
                text_emotion = self.analyze_text_sentiment(user_input)
                
                print(f"\n分析结果:")
                print(f"- 文本情绪: {text_emotion}")
                print(f"- 面部情绪: {self.emotion_map.get(facial_emotion, facial_emotion)}")
                
                # 生成并显示回复
                print("\n别急，我在思考...")
                response = self.generate_response(
                    user_input,
                    text_emotion,
                    facial_emotion
                )
                
                print(f"\n树洞: {response}")

        except KeyboardInterrupt:
            print("\n程序被中断")
        finally:
            self.stop_emotion_capture()
            print("程序已退出")

if __name__ == "__main__":
    chatbot = EmotionAIChatbot()
    chatbot.run()