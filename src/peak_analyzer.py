"""
峰分析模块

负责光谱峰的识别和定量分析。
"""

import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class PeakAnalyzer:
    """
    峰分析类
    
    提供光谱峰的自动识别和定量分析功能：
    - 自动寻峰
    - 峰高度计算
    - 峰面积计算
    - 基线校正峰参数
    """
    
    def __init__(self):
        """初始化峰分析器"""
        self.peaks = []  # 存储识别到的峰位置
        self.peak_properties = {}  # 存储峰的属性
        logger.info("峰分析器初始化完成")
    
    def find_peaks_auto(self, x_data: np.ndarray, y_data: np.ndarray,
                        threshold: float, min_distance: int) -> Tuple[bool, List[Tuple[float, float]], str]:
        """
        自动识别光谱峰
        
        Args:
            x_data: X轴数据（波数）
            y_data: Y轴数据（吸光度）
            threshold: 峰高度阈值
            min_distance: 峰之间的最小距离（数据点数）
            
        Returns:
            Tuple[bool, List[Tuple[float, float]], str]: (是否成功, 峰列表[(波数, 高度)], 错误消息)
        """
        try:
            if y_data is None or len(y_data) == 0:
                error_msg = "没有可用的数据"
                logger.warning(error_msg)
                return False, [], error_msg
            
            # 验证参数
            if threshold < 0:
                error_msg = f"阈值必须大于等于0，当前值: {threshold}"
                logger.error(error_msg)
                return False, [], error_msg
            
            if min_distance < 1:
                error_msg = f"最小距离必须大于等于1，当前值: {min_distance}"
                logger.error(error_msg)
                return False, [], error_msg
            
            # 寻找峰
            peak_indices, properties = find_peaks(
                y_data,
                height=threshold,
                distance=min_distance
            )
            
            # 存储峰信息
            self.peaks = peak_indices
            self.peak_properties = properties
            
            # 构建峰列表
            peak_list = []
            for idx in peak_indices:
                wavenumber = x_data[idx]
                height = y_data[idx]
                peak_list.append((float(wavenumber), float(height)))
            
            logger.info(f"找到 {len(peak_list)} 个峰")
            return True, peak_list, ""
            
        except ValueError as e:
            error_msg = f"参数格式不正确：{str(e)}"
            logger.error(error_msg)
            return False, [], error_msg
        except IndexError as e:
            error_msg = f"数据索引错误：{str(e)}"
            logger.error(error_msg)
            return False, [], error_msg
        except Exception as e:
            error_msg = f"峰分析出错：{str(e)}"
            logger.exception(error_msg)
            return False, [], error_msg
    
    def analyze_peak(self, x_data: np.ndarray, y_data: np.ndarray,
                     peak_wavenumber: float, lower_limit: float, upper_limit: float,
                     baseline_y_data: Optional[np.ndarray] = None) -> Tuple[bool, Dict[str, Any], str]:
        """
        分析指定峰的详细参数
        
        Args:
            x_data: X轴数据（波数）
            y_data: Y轴数据（吸光度）
            peak_wavenumber: 峰位置（波数）
            lower_limit: 分析范围下限
            upper_limit: 分析范围上限
            baseline_y_data: 基线数据（可选，用于校正计算）
            
        Returns:
            Tuple[bool, Dict[str, Any], str]: (是否成功, 峰参数字典, 错误消息)
        """
        try:
            logger.info(f"开始分析峰: {peak_wavenumber} cm^-1")
            
            # 验证范围
            if lower_limit >= upper_limit:
                error_msg = f"下限({lower_limit})必须小于上限({upper_limit})"
                logger.error(error_msg)
                return False, {}, error_msg
            
            # 获取分析范围内的数据
            mask = (x_data >= lower_limit) & (x_data <= upper_limit)
            x_range = x_data[mask]
            y_range = y_data[mask]
            
            if len(x_range) == 0:
                error_msg = f"范围 [{lower_limit}, {upper_limit}] 内没有数据点"
                logger.error(error_msg)
                return False, {}, error_msg

            # 【修复】检查是否只有一个数据点
            if len(x_range) == 1:
                error_msg = f"范围 [{lower_limit}, {upper_limit}] 内只有一个数据点，无法计算基线"
                logger.error(error_msg)
                return False, {}, error_msg

            # 找到峰位置的索引
            peak_idx = np.argmin(np.abs(x_range - peak_wavenumber))
            peak_x = x_range[peak_idx]
            peak_y = y_range[peak_idx]

            # 计算直线基线（连接范围两端点）
            # 【修复】防止除零错误
            x_diff = x_range[-1] - x_range[0]
            if abs(x_diff) < 1e-10:
                baseline_slope = 0
            else:
                baseline_slope = (y_range[-1] - y_range[0]) / x_diff
            baseline_intercept = y_range[0] - baseline_slope * x_range[0]
            y_baseline = baseline_slope * x_range + baseline_intercept
            
            # 计算基线在峰位置的高度
            baseline_at_peak = baseline_slope * peak_x + baseline_intercept
            
            # 计算峰高度
            uncorrected_height = peak_y
            corrected_height = peak_y - baseline_at_peak
            
            # 计算峰面积
            # 【修复】兼容 NumPy 旧版本，使用 trapz 而不是 trapezoid
            try:
                uncorrected_area = np.trapezoid(y_range, x_range)
                corrected_area = np.trapezoid(y_range - y_baseline, x_range)
            except AttributeError:
                uncorrected_area = np.trapz(y_range, x_range)
                corrected_area = np.trapz(y_range - y_baseline, x_range)

            # 如果提供了基线数据，也计算相对于基线的参数
            if baseline_y_data is not None:
                baseline_range = baseline_y_data[mask]
                baseline_at_peak_corrected = baseline_range[peak_idx]
                corrected_height_vs_baseline = peak_y - baseline_at_peak_corrected
                try:
                    corrected_area_vs_baseline = np.trapezoid(y_range - baseline_range, x_range)
                except AttributeError:
                    corrected_area_vs_baseline = np.trapz(y_range - baseline_range, x_range)
            else:
                corrected_height_vs_baseline = corrected_height
                corrected_area_vs_baseline = corrected_area
            
            # 整理结果（使用中文键名以兼容GUI）
            results = {
                '波数': float(peak_x),
                '未校正峰高': float(uncorrected_height),
                '校正峰高': float(corrected_height),
                '未校正峰面积': float(uncorrected_area),
                '校正峰面积': float(corrected_area),
                '区间下限': float(lower_limit),
                '区间上限': float(upper_limit)
            }

            logger.info(f"峰分析完成: 位置={peak_x:.2f}, 校正高度={corrected_height:.4f}, 区间={lower_limit:.2f}-{upper_limit:.2f}")
            return True, results, ""
            
        except ValueError as e:
            error_msg = f"参数格式不正确：{str(e)}"
            logger.error(error_msg)
            return False, {}, error_msg
        except IndexError as e:
            error_msg = f"数据索引错误：{str(e)}"
            logger.error(error_msg)
            return False, {}, error_msg
        except Exception as e:
            error_msg = f"峰分析出错：{str(e)}"
            logger.exception(error_msg)
            return False, {}, error_msg
    
    def export_peak_list(self, peak_list: List[Tuple[float, float]], 
                        file_path: str) -> Tuple[bool, str]:
        """
        导出峰列表到CSV文件
        
        Args:
            peak_list: 峰列表 [(波数, 高度), ...]
            file_path: 导出文件路径
            
        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        try:
            if not peak_list:
                error_msg = "没有可导出的峰数据"
                logger.warning(error_msg)
                return False, error_msg
            
            # 创建DataFrame
            df = pd.DataFrame(peak_list, columns=['波数 (cm^-1)', '峰高度'])
            df.insert(0, '峰编号', range(1, len(peak_list) + 1))
            
            # 导出到CSV
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            success_msg = f"峰列表导出成功，共 {len(peak_list)} 个峰"
            logger.info(f"峰列表导出到: {file_path}")
            return True, success_msg
            
        except ValueError as e:
            error_msg = f"峰数据格式错误：{str(e)}"
            logger.error(error_msg)
            return False, error_msg
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
    
    def get_peak_count(self) -> int:
        """
        获取识别到的峰数量
        
        Returns:
            int: 峰数量
        """
        return len(self.peaks)
    
    def clear_peaks(self):
        """清除所有峰数据"""
        self.peaks = []
        self.peak_properties = {}
        logger.info("峰数据已清除")

