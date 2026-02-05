"""
平滑处理模块

负责各种光谱数据平滑算法的实现。
"""

import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class SmoothingProcessor:
    """
    平滑处理类
    
    提供多种光谱数据平滑算法：
    - Savitzky-Golay滤波
    - LOWESS局部加权回归
    - 移动平均
    - 高斯滤波
    - 中值滤波
    """
    
    def __init__(self):
        """初始化平滑处理器"""
        logger.info("平滑处理器初始化完成")
    
    @staticmethod
    def validate_savgol_params(window_length: int, polyorder: int, data_length: int) -> Tuple[bool, str]:
        """
        验证Savitzky-Golay滤波参数
        
        Args:
            window_length: 窗口长度
            polyorder: 多项式阶数
            data_length: 数据长度
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误消息)
        """
        if window_length % 2 == 0:
            return False, f"窗口长度必须是奇数，当前值: {window_length}"
        
        if window_length > data_length:
            return False, f"窗口长度({window_length})不能大于数据长度({data_length})"
        
        if polyorder >= window_length:
            return False, f"多项式阶数({polyorder})必须小于窗口长度({window_length})"
        
        return True, ""
    
    @staticmethod
    def validate_positive_int(value: int, param_name: str, min_value: int = 1) -> Tuple[bool, str]:
        """
        验证正整数参数
        
        Args:
            value: 要验证的值
            param_name: 参数名称
            min_value: 最小值
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误消息)
        """
        if value < min_value:
            return False, f"{param_name}必须大于等于{min_value}，当前值: {value}"
        return True, ""
    
    @staticmethod
    def validate_positive_float(value: float, param_name: str, min_value: float = 0.0) -> Tuple[bool, str]:
        """
        验证正浮点数参数
        
        Args:
            value: 要验证的值
            param_name: 参数名称
            min_value: 最小值
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误消息)
        """
        if value <= min_value:
            return False, f"{param_name}必须大于{min_value}，当前值: {value}"
        return True, ""
    
    @staticmethod
    def validate_range(value: float, param_name: str, min_val: float, max_val: float) -> Tuple[bool, str]:
        """
        验证参数是否在指定范围内
        
        Args:
            value: 要验证的值
            param_name: 参数名称
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误消息)
        """
        if value < min_val or value > max_val:
            return False, f"{param_name}必须在{min_val}到{max_val}之间，当前值: {value}"
        return True, ""
    
    def smooth_savgol(self, x_data: np.ndarray, y_data: np.ndarray, 
                      window_length: int, polyorder: int) -> Tuple[bool, np.ndarray, str]:
        """
        Savitzky-Golay滤波平滑
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            window_length: 窗口长度（必须是奇数）
            polyorder: 多项式阶数
            
        Returns:
            Tuple[bool, np.ndarray, str]: (是否成功, 平滑后的数据, 错误消息)
        """
        try:
            # 验证参数
            valid, error_msg = self.validate_savgol_params(window_length, polyorder, len(y_data))
            if not valid:
                logger.error(f"Savgol参数验证失败: {error_msg}")
                return False, y_data, error_msg
            
            smoothed = savgol_filter(y_data, window_length, polyorder)
            logger.info(f"Savgol平滑完成: window={window_length}, poly={polyorder}")
            return True, smoothed, ""
            
        except Exception as e:
            error_msg = f"Savgol平滑失败: {str(e)}"
            logger.exception(error_msg)
            return False, y_data, error_msg
    
    def smooth_lowess(self, x_data: np.ndarray, y_data: np.ndarray,
                      frac: float, iterations: int) -> Tuple[bool, np.ndarray, str]:
        """
        LOWESS局部加权回归平滑
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            frac: 平滑分数（0-1之间）
            iterations: 迭代次数
            
        Returns:
            Tuple[bool, np.ndarray, str]: (是否成功, 平滑后的数据, 错误消息)
        """
        try:
            # 验证参数
            valid, error_msg = self.validate_range(frac, "分数", 0.0, 1.0)
            if not valid:
                logger.error(f"LOWESS参数验证失败: {error_msg}")
                return False, y_data, error_msg
            
            valid, error_msg = self.validate_positive_int(iterations, "迭代次数", 1)
            if not valid:
                logger.error(f"LOWESS参数验证失败: {error_msg}")
                return False, y_data, error_msg
            
            smoothed = lowess(y_data, x_data, frac=frac, it=iterations, return_sorted=False)
            logger.info(f"LOWESS平滑完成: frac={frac}, iterations={iterations}")
            return True, smoothed, ""
            
        except Exception as e:
            error_msg = f"LOWESS平滑失败: {str(e)}"
            logger.exception(error_msg)
            return False, y_data, error_msg
    
    def smooth_moving_average(self, x_data: np.ndarray, y_data: np.ndarray,
                              window_length: int) -> Tuple[bool, np.ndarray, str]:
        """
        移动平均平滑
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            window_length: 窗口长度
            
        Returns:
            Tuple[bool, np.ndarray, str]: (是否成功, 平滑后的数据, 错误消息)
        """
        try:
            # 验证参数
            valid, error_msg = self.validate_positive_int(window_length, "窗口长度", 1)
            if not valid:
                logger.error(f"移动平均参数验证失败: {error_msg}")
                return False, y_data, error_msg
            
            if window_length > len(y_data):
                error_msg = f"窗口长度({window_length})不能大于数据长度({len(y_data)})"
                logger.error(error_msg)
                return False, y_data, error_msg
            
            smoothed = np.convolve(y_data, np.ones(window_length) / window_length, mode='same')
            logger.info(f"移动平均平滑完成: window={window_length}")
            return True, smoothed, ""
            
        except Exception as e:
            error_msg = f"移动平均平滑失败: {str(e)}"
            logger.exception(error_msg)
            return False, y_data, error_msg
    
    def smooth_gaussian(self, x_data: np.ndarray, y_data: np.ndarray,
                        sigma: float) -> Tuple[bool, np.ndarray, str]:
        """
        高斯滤波平滑
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            sigma: 标准差
            
        Returns:
            Tuple[bool, np.ndarray, str]: (是否成功, 平滑后的数据, 错误消息)
        """
        try:
            # 验证参数
            valid, error_msg = self.validate_positive_float(sigma, "标准差", 0.0)
            if not valid:
                logger.error(f"高斯滤波参数验证失败: {error_msg}")
                return False, y_data, error_msg
            
            smoothed = gaussian_filter1d(y_data, sigma)
            logger.info(f"高斯滤波平滑完成: sigma={sigma}")
            return True, smoothed, ""
            
        except Exception as e:
            error_msg = f"高斯滤波平滑失败: {str(e)}"
            logger.exception(error_msg)
            return False, y_data, error_msg
    
    def smooth_median(self, x_data: np.ndarray, y_data: np.ndarray,
                      window_length: int) -> Tuple[bool, np.ndarray, str]:
        """
        中值滤波平滑
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            window_length: 窗口长度
            
        Returns:
            Tuple[bool, np.ndarray, str]: (是否成功, 平滑后的数据, 错误消息)
        """
        try:
            # 验证参数
            valid, error_msg = self.validate_positive_int(window_length, "窗口长度", 1)
            if not valid:
                logger.error(f"中值滤波参数验证失败: {error_msg}")
                return False, y_data, error_msg
            
            if window_length > len(y_data):
                error_msg = f"窗口长度({window_length})不能大于数据长度({len(y_data)})"
                logger.error(error_msg)
                return False, y_data, error_msg
            
            smoothed = medfilt(y_data, window_length)
            logger.info(f"中值滤波平滑完成: window={window_length}")
            return True, smoothed, ""
            
        except Exception as e:
            error_msg = f"中值滤波平滑失败: {str(e)}"
            logger.exception(error_msg)
            return False, y_data, error_msg
    
    def smooth_data_in_ranges(self, x_data: np.ndarray, y_data: np.ndarray,
                              ranges: List[Tuple[float, float]], method: str,
                              **params) -> Tuple[bool, np.ndarray, str]:
        """
        在指定范围内对数据进行平滑处理
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            ranges: 要处理的范围列表 [(start1, end1), (start2, end2), ...]
            method: 平滑方法名称
            **params: 平滑方法的参数
            
        Returns:
            Tuple[bool, np.ndarray, str]: (是否成功, 平滑后的数据, 错误消息)
        """
        try:
            # 如果没有指定范围，使用全部数据
            if not ranges:
                ranges = [(float(np.min(x_data)), float(np.max(x_data)))]
                logger.info("未指定范围，使用全部数据")
            else:
                logger.info(f"在{len(ranges)}个范围内进行平滑处理")
            
            # 初始化结果数据
            smoothed_data = y_data.copy()
            
            # 对每个范围进行处理
            for start, end in ranges:
                # 获取范围内的数据
                mask = (x_data >= start) & (x_data <= end)
                x_range = x_data[mask]
                y_range = y_data[mask]
                
                # 检查数据长度
                if len(y_range) == 0:
                    logger.warning(f"范围 [{start}, {end}] 内没有数据点")
                    continue
                
                # 根据方法选择平滑算法
                if method == "savgol":
                    success, smoothed_range, error_msg = self.smooth_savgol(
                        x_range, y_range, params['window_length'], params['polyorder'])
                elif method == "lowess":
                    success, smoothed_range, error_msg = self.smooth_lowess(
                        x_range, y_range, params['frac'], params['iterations'])
                elif method == "moving_average":
                    success, smoothed_range, error_msg = self.smooth_moving_average(
                        x_range, y_range, params['window_length'])
                elif method == "gaussian":
                    success, smoothed_range, error_msg = self.smooth_gaussian(
                        x_range, y_range, params['sigma'])
                elif method == "median":
                    success, smoothed_range, error_msg = self.smooth_median(
                        x_range, y_range, params['window_length'])
                else:
                    return False, y_data, f"未知的平滑方法: {method}"
                
                if not success:
                    return False, y_data, error_msg
                
                # 更新对应范围的数据
                smoothed_data[mask] = smoothed_range
            
            logger.info(f"平滑处理完成，方法: {method}")
            return True, smoothed_data, ""
            
        except Exception as e:
            error_msg = f"平滑处理失败: {str(e)}"
            logger.exception(error_msg)
            return False, y_data, error_msg

