from emogen import RealTimeEmotionRecognizer
from emogen import EmotionSupportSystem
import time

def main():
    # 初始化情绪支持系统
    support = EmotionSupportSystem(voice_speed=160)
    
    # 初始化实时情绪检测器（5秒分析窗口）
    recognizer = RealTimeEmotionRecognizer(analysis_window=5)
    
    print("情绪识别系统已启动，按Ctrl+C退出...")
    
    try:
        # 使用生成器获取窗口期主要情绪
        for dominant_emotion in recognizer.start():
            print(f"\n[系统检测] 主要情绪: {dominant_emotion}")
            
            # 处理情绪并生成语音反馈
            support.process_emotion(dominant_emotion)
            
            # 间隔避免语音重叠
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n正在停止系统...")
    finally:
        recognizer.stop()
        print("系统已安全关闭")

if __name__ == "__main__":
    main()