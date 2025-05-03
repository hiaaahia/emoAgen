from .emo_detection import RealTimeEmotionRecognizer
from .emo_support import EmotionSupportSystem
from .emo_text_detection import SentimentAnalyzer

# 包版本信息
__version__ = "1.0.0"
__all__ = ['RealTimeEmotionRecognizer', 'EmotionSupportSystem', 'SentimentAnalyzer']

# 初始化日志配置（可选）
# import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )

# logger = logging.getLogger(__name__)
# logger.info(f"emogen package {__version__} initialized")