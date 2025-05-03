import cv2
from deepface import DeepFace
import time
from collections import defaultdict

class RealTimeEmotionRecognizer:
    def __init__(self, camera_index=0, show_video=True):
        """
        初始化实时情绪识别器
        
        参数:
            camera_index: 摄像头索引(默认为0)
            show_video: 是否显示视频画面(默认False)
        """
        self.camera_index = camera_index
        self.show_video = show_video
        self.cap = None
        self.net = None
        self.running = False
        self.last_emotion = "Waiting..."
        
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
    
    def _process_frame(self, frame):
        """处理单帧图像并返回处理后的帧和当前情绪"""
        current_time = time.time()
        current_emotion = None
        
        # 每秒检测一次
        if not hasattr(self, 'last_detection_time') or current_time - self.last_detection_time >= 1:
            current_emotion = self._analyze_frame(frame)
            self.last_emotion = current_emotion
            self.last_detection_time = current_time
        
        if self.show_video:
            # 添加文本信息
            cv2.putText(frame, f'Current Emotion: {self.last_emotion}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame, current_emotion
    
    def start(self):
        """启动实时情绪识别"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cap.isOpened():
                raise RuntimeError("无法打开摄像头")
            
            self.running = True
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法获取帧，退出...")
                    break
                
                # 处理帧并获取当前情绪
                processed_frame, current_emotion = self._process_frame(frame)
                
                # 仅在需要时显示结果
                if self.show_video:
                    cv2.imshow("Real-Time Emotion Recognition", processed_frame)
                
                # 如果有检测结果，yield出去
                if current_emotion is not None:
                    yield current_emotion
                
                # 检查退出键
                if self.show_video and (cv2.waitKey(1) & 0xFF == ord('q')):
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


# 使用示例
if __name__ == "__main__":
    recognizer = RealTimeEmotionRecognizer(show_video=True)
    try:
        # 每秒获取一次情绪结果
        for emotion in recognizer.start():
            print(f"当前情绪: {emotion}")
    except KeyboardInterrupt:
        pass
    finally:
        recognizer.stop()