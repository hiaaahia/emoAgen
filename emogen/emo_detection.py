import cv2
from deepface import DeepFace
import time
from collections import defaultdict

class RealTimeEmotionRecognizer:
    def __init__(self, camera_index=0, analysis_window=5):
        """
        初始化实时情绪识别器
        
        参数:
            camera_index: 摄像头索引(默认为0)
            analysis_window: 分析窗口时间(秒)，默认为5秒
        """
        self.camera_index = camera_index
        self.analysis_window = analysis_window
        self.cap = None
        self.net = None
        self.running = False
        
        # 情绪统计相关变量
        self.emotion_counter = defaultdict(int)
        self.last_emotion = "Waiting..."
        self.last_window_output = None
        self.window_start_time = time.time()
        
        # 初始化DNN人脸检测器
        self._init_face_detector()
    
    def _init_face_detector(self):
        """初始化OpenCV的DNN人脸检测器"""
        try:
            self.net = cv2.dnn.readNetFromCaffe(
                "model/deploy.prototxt.txt", 
                "model/res10_300x300_ssd_iter_140000.caffemodel"
            )
        except Exception as e:
            raise RuntimeError(f"无法加载人脸检测器模型: {e}")
    
    def _analyze_frame(self, frame):
        """分析单帧图像的情绪"""
        try:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()
            
            if len(detections) > 0 and detections[0, 0, 0, 2] > 0.5:
                result = DeepFace.analyze(
                    frame, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                return result[0]['dominant_emotion']
            return "No face"
        except Exception as e:
            print(f"分析错误: {e}")
            return "Error"
    
    def _update_emotion_statistics(self, emotion):
        """更新情绪统计并返回窗口期内的主要情绪(如果有)"""
        current_time = time.time()
        output = None
        
        # 如果超过窗口时间，返回主要情绪并重置统计
        if current_time - self.window_start_time > self.analysis_window:
            if self.emotion_counter:
                output = max(self.emotion_counter.items(), key=lambda x: x[1])[0]
            else:
                output = "No data"
            
            self.emotion_counter = defaultdict(int)
            self.window_start_time = current_time
        
        # 更新当前情绪统计
        self.emotion_counter[emotion] += 1
        self.last_emotion = emotion
        
        return output
    
    def _process_frame(self, frame):
        """处理单帧图像并返回处理后的帧和主要情绪(如果有)"""
        current_time = time.time()
        dominant_emotion = None
        
        # 每秒检测一次
        if not hasattr(self, 'last_detection_time') or current_time - self.last_detection_time >= 1:
            emotion = self._analyze_frame(frame)
            dominant_emotion = self._update_emotion_statistics(emotion)
            self.last_detection_time = current_time
        
        # 添加文本信息
        cv2.putText(frame, f'Current: {self.last_emotion}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'Dominant (Last {self.analysis_window}s): {self.last_window_output}', 
                (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
        
        stats_text = "Statistics: " + ", ".join([f"{k}:{v}" for k, v in self.emotion_counter.items()])
        cv2.putText(frame, stats_text, (30, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        
        remaining_time = max(0, self.analysis_window - (current_time - self.window_start_time))
        cv2.putText(frame, f'Window closes in: {remaining_time:.1f}s', (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        return frame, dominant_emotion
    
    def start(self):
        """
        启动实时情绪识别
        
        返回:
            生成器，每次窗口期结束时yield主要情绪结果
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError("无法打开摄像头")
            
            self.running = True
            print(f"实时情绪识别已启动(每秒检测一次，{self.analysis_window}秒窗口统计)，按'q'键退出...")
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法获取帧，退出...")
                    break
                
                # 处理帧并获取可能的主要情绪结果
                processed_frame, dominant_emotion = self._process_frame(frame)
                
                # 显示结果
                cv2.imshow("Real-Time Emotion Recognition", processed_frame)
                
                # 如果有窗口期结果，yield出去
                if dominant_emotion is not None:
                    self.last_window_output = dominant_emotion
                    yield dominant_emotion
                
                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
        
        except Exception as e:
            print(f"运行时错误: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """停止实时情绪识别并释放资源"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("实时情绪识别已停止")


# 使用示例
if __name__ == "__main__":
    recognizer = RealTimeEmotionRecognizer(analysis_window=5)
    
    try:
        # 使用生成器获取窗口期结果
        recognizer.start()
    except KeyboardInterrupt:
        recognizer.stop()