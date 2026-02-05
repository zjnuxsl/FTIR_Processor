"""
FTIR光谱处理器核心模块

包含数据管理、平滑处理、基线校正和峰分析功能
"""

from .data_manager import DataManager
from .smoothing_processor import SmoothingProcessor
from .baseline_corrector import BaselineCorrector
from .peak_analyzer import PeakAnalyzer

__version__ = "2.0.0"
__author__ = "zjnuxsl"
__email__ = "sl-xiao@zjnu.cn"

__all__ = [
    'DataManager',
    'SmoothingProcessor',
    'BaselineCorrector',
    'PeakAnalyzer'
]

