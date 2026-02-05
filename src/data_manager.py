"""
数据管理模块

负责FTIR光谱数据的加载、存储、验证和导出。
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class DataManager:
    """
    数据管理类
    
    负责FTIR光谱数据的加载、存储、验证和导出功能。
    
    Attributes:
        x_data: X轴数据（波数，单位cm^-1）
        y_data: Y轴原始数据（吸光度/透射率）
        smoothed_data: 平滑处理后的数据
        corrected_data: 基线校正后的数据
    """
    
    def __init__(self):
        """初始化数据管理器"""
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.smoothed_data: Optional[np.ndarray] = None
        self.corrected_data: Optional[np.ndarray] = None
        logger.info("数据管理器初始化完成")
    
    def load_from_csv(self, file_path: str) -> Tuple[bool, str]:
        """
        从CSV文件加载光谱数据
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            Tuple[bool, str]: (是否成功, 错误消息或成功消息)
        """
        try:
            logger.info(f"正在加载文件: {file_path}")
            data = pd.read_csv(file_path)
            
            # 验证数据格式
            if len(data.columns) < 2:
                error_msg = f"数据文件列数不足: {len(data.columns)}，需要至少两列数据"
                logger.error(error_msg)
                return False, error_msg
            
            if len(data) == 0:
                error_msg = "数据文件为空"
                logger.error(error_msg)
                return False, error_msg
            
            # 提取数据
            self.x_data = data.iloc[:, 0].values
            self.y_data = data.iloc[:, 1].values
            
            # 验证数据有效性
            valid, error_msg = self._validate_data(self.x_data, self.y_data)
            if not valid:
                self.x_data = None
                self.y_data = None
                return False, error_msg
            
            # 重置处理后的数据
            self.smoothed_data = None
            self.corrected_data = None
            
            success_msg = f"数据加载成功，数据点数: {len(self.x_data)}"
            logger.info(success_msg)
            return True, success_msg
            
        except FileNotFoundError:
            error_msg = "文件不存在"
            logger.error(error_msg)
            return False, error_msg
        except pd.errors.EmptyDataError:
            error_msg = "文件为空或格式不正确"
            logger.error(error_msg)
            return False, error_msg
        except pd.errors.ParserError as e:
            error_msg = f"CSV文件解析错误：{str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except PermissionError:
            error_msg = "没有权限读取该文件"
            logger.error(error_msg)
            return False, error_msg
        except UnicodeDecodeError:
            error_msg = "文件编码错误！请确保文件是UTF-8或GBK编码"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"数据加载出错：{str(e)}"
            logger.exception(error_msg)
            return False, error_msg
    
    def _validate_data(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[bool, str]:
        """
        验证数据有效性
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误消息)
        """
        # 检查NaN值
        if np.any(np.isnan(x_data)) or np.any(np.isnan(y_data)):
            error_msg = "数据包含无效值（NaN）"
            logger.error(error_msg)
            return False, error_msg
        
        # 检查无穷大值
        if np.any(np.isinf(x_data)) or np.any(np.isinf(y_data)):
            error_msg = "数据包含无穷大值（Inf）"
            logger.error(error_msg)
            return False, error_msg
        
        return True, ""
    
    def check_data_loaded(self, data_type: str = 'original') -> Tuple[bool, str]:
        """
        检查所需数据是否已加载
        
        Args:
            data_type: 数据类型，可选值：'original', 'smoothed', 'corrected'
            
        Returns:
            Tuple[bool, str]: (是否已加载, 错误消息)
        """
        if self.x_data is None or self.y_data is None:
            error_msg = "请先加载数据！"
            logger.warning("尝试操作但数据未加载")
            return False, error_msg
        
        if data_type == 'smoothed' and self.smoothed_data is None:
            error_msg = "没有可用的平滑数据！请先进行平滑处理。"
            logger.warning("尝试使用平滑数据但未进行平滑处理")
            return False, error_msg
        
        if data_type == 'corrected' and self.corrected_data is None:
            error_msg = "没有可用的校正数据！请先进行基线校正。"
            logger.warning("尝试使用校正数据但未进行基线校正")
            return False, error_msg
        
        return True, ""
    
    def get_data(self, data_type: str = 'original') -> Optional[np.ndarray]:
        """
        获取指定类型的Y轴数据
        
        Args:
            data_type: 数据类型，可选值：'original', 'smoothed', 'corrected'
            
        Returns:
            对应的Y轴数据，如果不存在返回None
        """
        if data_type == 'smoothed':
            return self.smoothed_data
        elif data_type == 'corrected':
            return self.corrected_data
        else:
            return self.y_data
    
    def set_smoothed_data(self, data: np.ndarray):
        """
        设置平滑后的数据
        
        Args:
            data: 平滑后的数据
        """
        self.smoothed_data = data.copy()
        logger.info("平滑数据已更新")
    
    def set_corrected_data(self, data: np.ndarray):
        """
        设置基线校正后的数据
        
        Args:
            data: 校正后的数据
        """
        self.corrected_data = data.copy()
        logger.info("校正数据已更新")
    
    def export_to_csv(self, file_path: str, data_type: str = 'smoothed') -> Tuple[bool, str]:
        """
        导出数据到CSV文件
        
        Args:
            file_path: 导出文件路径
            data_type: 要导出的数据类型
            
        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        try:
            # 检查数据是否存在
            is_loaded, error_msg = self.check_data_loaded(data_type)
            if not is_loaded:
                return False, error_msg
            
            # 获取要导出的数据
            y_data = self.get_data(data_type)
            if y_data is None:
                return False, f"没有可用的{data_type}数据"
            
            # 创建DataFrame并导出
            df = pd.DataFrame({
                '波数 (cm^-1)': self.x_data,
                '吸光度': y_data
            })
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            success_msg = f"{data_type}数据导出成功"
            logger.info(f"数据导出到: {file_path}")
            return True, success_msg
            
        except PermissionError:
            error_msg = "没有权限写入该文件！请检查文件是否被其他程序占用。"
            logger.error(error_msg)
            return False, error_msg
        except OSError as e:
            error_msg = f"文件写入错误：{str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"导出出错：{str(e)}"
            logger.exception(error_msg)
            return False, error_msg
    
    def has_data(self) -> bool:
        """
        检查是否有已加载的数据
        
        Returns:
            bool: 是否有数据
        """
        return self.x_data is not None and self.y_data is not None
    
    def get_data_length(self) -> int:
        """
        获取数据长度
        
        Returns:
            int: 数据点数，如果没有数据返回0
        """
        if self.x_data is not None:
            return len(self.x_data)
        return 0
    
    def get_data_range(self) -> Tuple[float, float]:
        """
        获取X轴数据范围
        
        Returns:
            Tuple[float, float]: (最小值, 最大值)
        """
        if self.x_data is not None:
            return float(np.min(self.x_data)), float(np.max(self.x_data))
        return 0.0, 0.0

