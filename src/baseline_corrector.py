"""
基线校正模块

负责各种基线校正算法的实现。
"""

import numpy as np
from pybaselines import Baseline
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class BaselineCorrector:
    """
    基线校正类
    
    提供多种基线校正算法：
    - Rubberband（橡皮筋法）
    - Modified Polynomial（修正多项式）
    - Iterative Modified Polynomial（自适应迭代多项式）
    - Whittaker-ASLS（Whittaker平滑与非对称最小二乘）
    - Mixture Model（混合模型/平滑样条）
    """
    
    def __init__(self):
        """初始化基线校正器"""
        self.baseline_fitter = Baseline()
        logger.info("基线校正器初始化完成")
    
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
    
    def correct_rubberband(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, str]:
        """
        Rubberband基线校正
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            
        Returns:
            Tuple[bool, np.ndarray, np.ndarray, str]: (是否成功, 校正后的数据, 基线, 错误消息)
        """
        try:
            baseline, params = self.baseline_fitter.rubberband(y_data)
            corrected = y_data - baseline
            logger.info("Rubberband基线校正完成")
            return True, corrected, baseline, ""
        except Exception as e:
            error_msg = f"Rubberband基线校正失败: {str(e)}"
            logger.exception(error_msg)
            return False, y_data, np.zeros_like(y_data), error_msg
    
    def correct_modpoly(self, x_data: np.ndarray, y_data: np.ndarray, 
                        poly_order: int) -> Tuple[bool, np.ndarray, np.ndarray, str]:
        """
        Modified Polynomial基线校正
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            poly_order: 多项式阶数
            
        Returns:
            Tuple[bool, np.ndarray, np.ndarray, str]: (是否成功, 校正后的数据, 基线, 错误消息)
        """
        try:
            # 验证参数
            valid, error_msg = self.validate_positive_int(poly_order, "多项式阶数", 1)
            if not valid:
                logger.error(f"ModPoly参数验证失败: {error_msg}")
                return False, y_data, np.zeros_like(y_data), error_msg
            
            baseline, params = self.baseline_fitter.modpoly(y_data, poly_order)
            corrected = y_data - baseline
            logger.info(f"ModPoly基线校正完成: poly_order={poly_order}")
            return True, corrected, baseline, ""
        except Exception as e:
            error_msg = f"ModPoly基线校正失败: {str(e)}"
            logger.exception(error_msg)
            return False, y_data, np.zeros_like(y_data), error_msg
    
    def correct_imodpoly(self, x_data: np.ndarray, y_data: np.ndarray,
                         poly_order: int, max_iter: int = 50) -> Tuple[bool, np.ndarray, np.ndarray, str]:
        """
        Iterative Modified Polynomial基线校正
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            poly_order: 多项式阶数
            max_iter: 最大迭代次数
            
        Returns:
            Tuple[bool, np.ndarray, np.ndarray, str]: (是否成功, 校正后的数据, 基线, 错误消息)
        """
        try:
            # 验证参数
            valid, error_msg = self.validate_positive_int(poly_order, "多项式阶数", 1)
            if not valid:
                logger.error(f"IModPoly参数验证失败: {error_msg}")
                return False, y_data, np.zeros_like(y_data), error_msg
            
            valid, error_msg = self.validate_positive_int(max_iter, "最大迭代次数", 1)
            if not valid:
                logger.error(f"IModPoly参数验证失败: {error_msg}")
                return False, y_data, np.zeros_like(y_data), error_msg
            
            baseline, params = self.baseline_fitter.imodpoly(
                y_data, poly_order, max_iter=max_iter)
            corrected = y_data - baseline
            logger.info(f"IModPoly基线校正完成: poly_order={poly_order}, max_iter={max_iter}")
            return True, corrected, baseline, ""
        except Exception as e:
            error_msg = f"IModPoly基线校正失败: {str(e)}"
            logger.exception(error_msg)
            return False, y_data, np.zeros_like(y_data), error_msg
    
    def correct_asls(self, x_data: np.ndarray, y_data: np.ndarray,
                     lam: float, p: float) -> Tuple[bool, np.ndarray, np.ndarray, str]:
        """
        Whittaker-ASLS基线校正
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            lam: 平滑参数（lambda）
            p: 非对称参数
            
        Returns:
            Tuple[bool, np.ndarray, np.ndarray, str]: (是否成功, 校正后的数据, 基线, 错误消息)
        """
        try:
            # 验证参数
            valid, error_msg = self.validate_positive_float(lam, "平滑参数", 0.0)
            if not valid:
                logger.error(f"ASLS参数验证失败: {error_msg}")
                return False, y_data, np.zeros_like(y_data), error_msg
            
            if p <= 0 or p >= 1:
                error_msg = f"非对称参数必须在0到1之间，当前值: {p}"
                logger.error(error_msg)
                return False, y_data, np.zeros_like(y_data), error_msg
            
            baseline, params = self.baseline_fitter.asls(y_data, lam=lam, p=p)
            corrected = y_data - baseline
            logger.info(f"ASLS基线校正完成: lam={lam}, p={p}")
            return True, corrected, baseline, ""
        except Exception as e:
            error_msg = f"ASLS基线校正失败: {str(e)}"
            logger.exception(error_msg)
            return False, y_data, np.zeros_like(y_data), error_msg
    
    def correct_mixture_model(self, x_data: np.ndarray, y_data: np.ndarray,
                              num_knots: int = 10) -> Tuple[bool, np.ndarray, np.ndarray, str]:
        """
        Mixture Model基线校正（使用平滑样条）
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            num_knots: 样条节点数
            
        Returns:
            Tuple[bool, np.ndarray, np.ndarray, str]: (是否成功, 校正后的数据, 基线, 错误消息)
        """
        try:
            # 验证参数
            valid, error_msg = self.validate_positive_int(num_knots, "样条节点数", 2)
            if not valid:
                logger.error(f"Mixture Model参数验证失败: {error_msg}")
                return False, y_data, np.zeros_like(y_data), error_msg
            
            if num_knots > len(y_data) // 2:
                error_msg = f"样条节点数({num_knots})不能大于数据长度的一半({len(y_data)//2})"
                logger.error(error_msg)
                return False, y_data, np.zeros_like(y_data), error_msg
            
            baseline, params = self.baseline_fitter.mixture_model(
                y_data, num_knots=num_knots)
            corrected = y_data - baseline
            logger.info(f"Mixture Model基线校正完成: num_knots={num_knots}")
            return True, corrected, baseline, ""
        except Exception as e:
            error_msg = f"Mixture Model基线校正失败: {str(e)}"
            logger.exception(error_msg)
            return False, y_data, np.zeros_like(y_data), error_msg
    
    def correct_baseline(self, x_data: np.ndarray, y_data: np.ndarray,
                        method: str, **params) -> Tuple[bool, np.ndarray, np.ndarray, str]:
        """
        执行基线校正
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            method: 校正方法名称
            **params: 方法参数
            
        Returns:
            Tuple[bool, np.ndarray, np.ndarray, str]: (是否成功, 校正后的数据, 基线, 错误消息)
        """
        try:
            # 验证数据
            if len(y_data) == 0:
                error_msg = "数据为空"
                logger.error(error_msg)
                return False, y_data, np.zeros_like(y_data), error_msg
            
            if np.any(np.isnan(y_data)) or np.any(np.isinf(y_data)):
                error_msg = "数据包含无效值（NaN或Inf）"
                logger.error(error_msg)
                return False, y_data, np.zeros_like(y_data), error_msg
            
            logger.info(f"开始基线校正，方法: {method}")
            
            # 根据方法选择校正算法
            if method == "rubberband":
                return self.correct_rubberband(x_data, y_data)
            elif method == "modpoly":
                return self.correct_modpoly(x_data, y_data, params.get('poly_order', 2))
            elif method == "imodpoly":
                return self.correct_imodpoly(x_data, y_data, 
                                            params.get('poly_order', 2),
                                            params.get('max_iter', 50))
            elif method == "asls":
                return self.correct_asls(x_data, y_data,
                                        params.get('lam', 1e6),
                                        params.get('p', 0.01))
            elif method == "mixture_model":
                return self.correct_mixture_model(x_data, y_data,
                                                  params.get('num_knots', 10))
            else:
                error_msg = f"未知的基线校正方法: {method}"
                logger.error(error_msg)
                return False, y_data, np.zeros_like(y_data), error_msg
                
        except Exception as e:
            error_msg = f"基线校正失败: {str(e)}"
            logger.exception(error_msg)
            return False, y_data, np.zeros_like(y_data), error_msg

