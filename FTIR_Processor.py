import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import savgol_filter, medfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from pybaselines import Baseline, polynomial, whittaker, spline
import os
import threading

# 导入新模块
try:
    from config_manager import ConfigManager, ProcessingPipeline
    from batch_processor import BatchProcessor, ExportManager
    HAS_ENHANCED_FEATURES = True
except ImportError:
    HAS_ENHANCED_FEATURES = False
    print("警告: 增强功能模块未找到，部分功能将不可用")

class SpectralProcessorGUI:
    def __init__(self, root):
        self.root = root

        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 初始化数据
        self.x_data = None
        self.y_data = None
        self.smoothed_data = None
        self.corrected_data = None
        self.baseline_fitter = Baseline()

        # 初始化变量
        self.data_source_var = tk.StringVar(value="original")
        self.y_label_var = tk.StringVar(value="吸光度")

        # 初始化图形属性
        self.smooth_ax1 = None
        self.smooth_ax2 = None
        self.baseline_ax1 = None
        self.baseline_ax2 = None
        self.smooth_canvas = None
        self.baseline_canvas = None

        # 初始化增强功能
        if HAS_ENHANCED_FEATURES:
            self.config_manager = ConfigManager()
            self.export_manager = ExportManager()
        else:
            self.config_manager = None
            self.export_manager = None

        # 创建菜单栏
        self.create_menu()

        # 创建主框架
        self.create_main_frame()

    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="加载数据", command=self.load_data)
        file_menu.add_separator()

        if HAS_ENHANCED_FEATURES:
            file_menu.add_command(label="保存配置", command=self.save_config)
            file_menu.add_command(label="加载配置", command=self.load_config)
            file_menu.add_separator()
            file_menu.add_command(label="批量处理", command=self.open_batch_dialog)
            file_menu.add_separator()

        file_menu.add_command(label="退出", command=self.root.quit)

        # 导出菜单
        export_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="导出", menu=export_menu)
        export_menu.add_command(label="导出平滑数据(CSV)", command=self.export_smooth_data)
        export_menu.add_command(label="导出基线校正数据(CSV)", command=self.export_baseline_data)

        if HAS_ENHANCED_FEATURES:
            export_menu.add_separator()
            export_menu.add_command(label="导出完整报告(Excel)", command=self.export_full_report_excel)
            export_menu.add_command(label="导出数据(带元数据)", command=self.export_with_metadata)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="平滑方法说明", command=self.show_smoothing_help)
        help_menu.add_command(label="基线校正说明", command=self.show_baseline_help)
        help_menu.add_separator()
        help_menu.add_command(label="关于", command=self.show_about)

    def update_plots(self):
        """更新所有图形的 y 轴标题"""
        y_label = self.y_label_var.get()
        
        # 更新平滑处理页面的图形
        if hasattr(self, 'smooth_ax1') and self.smooth_ax1 is not None:
            self.smooth_ax1.set_ylabel(y_label)
        if hasattr(self, 'smooth_ax2') and self.smooth_ax2 is not None:
            self.smooth_ax2.set_ylabel(y_label)
        if hasattr(self, 'smooth_canvas'):
            self.smooth_canvas.draw()
        
        # 更新基线校正页面的图形
        if hasattr(self, 'baseline_ax1') and self.baseline_ax1 is not None:
            self.baseline_ax1.set_ylabel(y_label)
        if hasattr(self, 'baseline_ax2') and self.baseline_ax2 is not None:
            self.baseline_ax2.set_ylabel(y_label)
        if hasattr(self, 'baseline_canvas'):
            self.baseline_canvas.draw()
    
    def create_main_frame(self):
        # 创建帮助按钮框架
        help_frame = ttk.LabelFrame(self.root, text="帮助说明")
        help_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建帮助按钮置为靠左对齐
        btn_container = ttk.Frame(help_frame)
        btn_container.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 添加帮助按钮
        ttk.Button(
            btn_container, 
            text="平滑方法说明", 
            width=15,
            command=self.show_smoothing_help
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_container, 
            text="基线校正说明", 
            width=15,
            command=self.show_baseline_help
        ).pack(side=tk.LEFT, padx=5)
        
        # 创建标签页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建页面
        self.smooth_frame = ttk.Frame(self.notebook)
        self.baseline_frame = ttk.Frame(self.notebook)
        self.peak_analysis_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.smooth_frame, text="平滑处理")
        self.notebook.add(self.baseline_frame, text="基线校正")
        self.notebook.add(self.peak_analysis_frame, text="特征峰分析")
        
        # 创建各页面内容
        self.create_smooth_page()
        self.create_baseline_page()
        self.create_peak_analysis_page()
        
    def create_smooth_page(self):
        """创建平滑处理页面"""
        # 创建左侧控制面板
        control_frame = ttk.LabelFrame(self.smooth_frame, text="控制面板")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Y轴标题选择
        y_label_frame = ttk.LabelFrame(control_frame, text="Y轴标题")
        y_label_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 加载数据按钮
        ttk.Button(control_frame, text="加载数据", 
                  command=self.load_data).pack(pady=5)
        
        # 数据范围选择
        range_frame = ttk.LabelFrame(control_frame, text="数据范围选择")
        range_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 范列表
        self.ranges_listbox = tk.Listbox(range_frame, height=3)
        self.ranges_listbox.pack(fill=tk.X, padx=5, pady=5)
        
        # 范围输入
        range_input_frame = ttk.Frame(range_frame)
        range_input_frame.pack(fill=tk.X, padx=5)
        
        ttk.Label(range_input_frame, text="起始值:").pack(side=tk.LEFT)
        self.range_start_var = tk.StringVar()
        ttk.Entry(range_input_frame, textvariable=self.range_start_var, width=10).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(range_input_frame, text="终止值:").pack(side=tk.LEFT)
        self.range_end_var = tk.StringVar()
        ttk.Entry(range_input_frame, textvariable=self.range_end_var, width=10).pack(side=tk.LEFT, padx=2)
        
        # 范围操作按钮
        range_btn_frame = ttk.Frame(range_frame)
        range_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(range_btn_frame, text="添加范围", command=self.add_range).pack(side=tk.LEFT, padx=2)
        ttk.Button(range_btn_frame, text="删除范围", command=self.delete_range).pack(side=tk.LEFT, padx=2)
        ttk.Button(range_btn_frame, text="清空范围", command=self.clear_ranges).pack(side=tk.LEFT, padx=2)
        
        # 平滑方法选择
        method_frame = ttk.LabelFrame(control_frame, text="平滑法")
        method_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.smooth_method = tk.StringVar(value="savgol")
        methods = [
            ("Savitzky-Golay", "savgol"),
            ("LOWESS（局部平滑）", "lowess"),
            ("移动平均", "moving_average"),
            ("高斯滤波", "gaussian"),
            ("中值滤波", "median")
        ]
        
        # 创建单选按钮
        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, value=value, 
                           variable=self.smooth_method, 
                           command=self.update_param_frame).pack(anchor=tk.W)
        
        # 参数设置框架
        self.param_frame = ttk.LabelFrame(control_frame, text="参数设置")
        self.param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 初始化参数设置
        self.update_param_frame()
        
        # 执行和导出按钮
        ttk.Button(control_frame, text="执行平滑", command=self.smooth_data).pack(pady=5)
        ttk.Button(control_frame, text="导出数据", command=self.export_smooth_data).pack(pady=5)
        
        # 创建右侧图形区
        plot_frame = ttk.Frame(self.smooth_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.smooth_fig = plt.Figure(figsize=(10, 6))
        self.smooth_ax1 = self.smooth_fig.add_subplot(211)
        self.smooth_ax2 = self.smooth_fig.add_subplot(212)
        
        self.smooth_canvas = FigureCanvasTkAgg(self.smooth_fig, master=plot_frame)
        self.smooth_canvas.draw()
        self.smooth_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添工具栏
        toolbar = NavigationToolbar2Tk(self.smooth_canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 在控制面板最下信
        author_frame = ttk.Frame(control_frame)
        author_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # 添加作者信息标签
        author_label = ttk.Label(
            author_frame, 
            text="作者: zjnuxsl\n邮箱: sl-xiao@zjnu.cn",
            justify=tk.LEFT,
            font=('SimSun' if '作者' in text else 'Times New Roman', 9)
        )
        author_label.pack(side=tk.LEFT)
        
    def create_baseline_page(self):
        """创建基线校正页面"""
        # 创建左侧控制面板
        control_frame = ttk.LabelFrame(self.baseline_frame, text="控制面板")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 加载数据按钮
        load_btn = ttk.Button(control_frame, text="加载数据",
                             command=self.load_data)
        load_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 数据源选择部分
        data_frame = ttk.LabelFrame(control_frame, text="数据选择")
        data_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 数据源选择按钮
        ttk.Radiobutton(data_frame, text="原始数据", value="original", 
                        variable=self.data_source_var, command=self.update_baseline_plot).pack(anchor=tk.W)
        ttk.Radiobutton(data_frame, text="平滑后数", value="smoothed", 
                        variable=self.data_source_var, command=self.update_baseline_plot).pack(anchor=tk.W)
        ttk.Button(data_frame, text="加载新数据", command=self.load_data).pack(pady=5)
        
        # 基线校正方法选择
        method_frame = ttk.LabelFrame(control_frame, text="校正方法")
        method_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.baseline_method = tk.StringVar(value="rubberband")
        methods = [
            ("Rubberband(推荐)", "rubberband"),  # FTIR最常用
            ("修正多项式", "modpoly"),  # 简单基线
            ("自适应迭代多项式", "imodpoly"),  # 适合非线性基线
            ("Whittaker-ASLS", "asls"),  # 处理基线漂移
            ("平滑样条", "mixture_model"),  # 复杂基线
        ]
        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, value=value, 
                          variable=self.baseline_method,
                          command=self.update_baseline_params).pack(anchor=tk.W)
        
        # 创建参数设框架
        self.baseline_param_frame = ttk.LabelFrame(control_frame, text="参数设置")
        self.baseline_param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 初始化数设置
        self.update_baseline_params()
        
        # 执行和导出按钮
        ttk.Button(control_frame, text="执行校正", command=self.correct_baseline).pack(pady=5)
        ttk.Button(control_frame, text="导出数据", command=self.export_baseline_data).pack(pady=5)
        
        # 创建右侧图形区域
        plot_frame = ttk.Frame(self.baseline_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.baseline_fig = plt.Figure(figsize=(10, 6))
        self.baseline_ax1 = self.baseline_fig.add_subplot(211)
        self.baseline_ax2 = self.baseline_fig.add_subplot(212)
        
        self.baseline_canvas = FigureCanvasTkAgg(self.baseline_fig, master=plot_frame)
        self.baseline_canvas.draw()
        self.baseline_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        toolbar = NavigationToolbar2Tk(self.baseline_canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 在控制面板最下方添加作者信息
        author_frame = ttk.Frame(control_frame)
        author_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # 添加作者信息标签
        author_label = ttk.Label(
            author_frame, 
            text="作者: zjnuxsl\n邮箱: sl-xiao@zjnu.cn",
            justify=tk.LEFT,
            font=('SimHei', 9)
        )
        author_label.pack(side=tk.LEFT)

    def switch_data(self, data_type):
        """切换数据源"""
        if data_type == "smoothed" and self.smoothed_data is None:
            messagebox.showerror("错误", "没有可用的平滑数据！请先进行平滑处理。")
            return
            
        if self.x_data is None:
            messagebox.showerror("错误", "请先加载数据！")
            return
            
        self.data_source_var.set(data_type)
        self.update_baseline_plot()
        data_type_text = "原始数据" if data_type == "original" else "平滑后数据"
        messagebox.showinfo("成功", f"已切换到{data_type_text}")

    def load_data(self):
        """加载数据文件"""
        try:
            file_path = filedialog.askopenfilename(
                title="选择数据件",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # 载数据
                data = pd.read_csv(file_path)
                if len(data.columns) < 2:
                    messagebox.showerror("错误", "数据文格式不正！需要至少两列数据。")
                    return
                    
                self.x_data = data.iloc[:, 0].values
                self.y_data = data.iloc[:, 1].values
                self.smoothed_data = None
                self.corrected_data = None
                
                # 更新图形显示
                self.plot_data()
                
                messagebox.showinfo("成功", "数据加载成功！")
                
        except Exception as e:
            messagebox.showerror("错误", f"数据加载出错：{str(e)}")

    def update_param_frame(self):
        """根据选择的平滑方法新参数设置框架"""
        # 清除所有参数设置
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        method = self.smooth_method.get()
        
        if method == "savgol":
            ttk.Label(self.param_frame, text="窗口长度:").pack()
            self.window_length_var = tk.StringVar(value="11")
            ttk.Entry(self.param_frame, textvariable=self.window_length_var).pack()
            
            ttk.Label(self.param_frame, text="多项式阶数:").pack()
            self.polyorder_var = tk.StringVar(value="3")
            ttk.Entry(self.param_frame, textvariable=self.polyorder_var).pack()
            
        elif method == "moving_average":
            ttk.Label(self.param_frame, text="窗口长度:").pack()
            self.window_length_var = tk.StringVar(value="5")
            ttk.Entry(self.param_frame, textvariable=self.window_length_var).pack()
            
        elif method == "gaussian":
            ttk.Label(self.param_frame, text="标准差:").pack()
            self.sigma_var = tk.StringVar(value="1.0")
            ttk.Entry(self.param_frame, textvariable=self.sigma_var).pack()
            
        elif method == "median":
            ttk.Label(self.param_frame, text="窗口长度:").pack()
            self.window_length_var = tk.StringVar(value="5")
            ttk.Entry(self.param_frame, textvariable=self.window_length_var).pack()
            
        elif method == "lowess":
            ttk.Label(self.param_frame, text="分数:").pack()
            self.frac_var = tk.StringVar(value="0.2")
            ttk.Entry(self.param_frame, textvariable=self.frac_var).pack()
            
            ttk.Label(self.param_frame, text="迭代次数:").pack()
            self.it_var = tk.StringVar(value="3")
            ttk.Entry(self.param_frame, textvariable=self.it_var).pack()

    def add_range(self):
        """添数据处理范"""
        try:
            start = float(self.range_start_var.get())
            end = float(self.range_end_var.get())
            if start >= end:
                raise ValueError("始值必须小于终止值")
            range_str = f"{start:.2f} - {end:.2f}"
            self.ranges_listbox.insert(tk.END, range_str)
        except ValueError as e:
            messagebox.showerror("错误", str(e))

    def delete_range(self):
        """删除选中的范围"""
        selection = self.ranges_listbox.curselection()
        if selection:
            self.ranges_listbox.delete(selection)

    def clear_ranges(self):
        """清空所有范围"""
        self.ranges_listbox.delete(0, tk.END)

    def get_selected_ranges(self):
        """获取所有选择的范围"""
        ranges = []
        for i in range(self.ranges_listbox.size()):
            range_str = self.ranges_listbox.get(i)
            start, end = map(float, range_str.split(" - "))
            ranges.append((start, end))
        return ranges

    def smooth_data(self):
        """执行数平滑处理"""
        if self.x_data is None or self.y_data is None:
            messagebox.showerror("错误", "请先加载数据")
            return
            
        try:
            method = self.smooth_method.get()
            
            # 获选中的范围
            ranges = self.get_selected_ranges()
            
            # 如果没有选择范围，使用全部数据
            if not ranges:
                ranges = [(min(self.x_data), max(self.x_data))]
            
            # 初始化平滑后的数据
            if self.smoothed_data is None:
                self.smoothed_data = np.copy(self.y_data)
            
            # 对每个范围进行平滑处理
            for start, end in ranges:
                # 获取范围内的数据
                mask = (self.x_data >= start) & (self.x_data <= end)
                x_range = self.x_data[mask]
                y_range = self.y_data[mask]
                
                # 根据选择的方法进行平滑
                if method == "savgol":
                    window_length = int(self.window_length_var.get())
                    polyorder = int(self.polyorder_var.get())
                    smoothed_range = savgol_filter(y_range, window_length, polyorder)
                    
                elif method == "moving_average":
                    window_length = int(self.window_length_var.get())
                    kernel = np.ones(window_length) / window_length
                    smoothed_range = np.convolve(y_range, kernel, mode='same')
                    
                elif method == "gaussian":
                    sigma = float(self.sigma_var.get())
                    smoothed_range = gaussian_filter1d(y_range, sigma)
                    
                elif method == "median":
                    window_length = int(self.window_length_var.get())
                    smoothed_range = medfilt(y_range, window_length)
                    
                elif method == "lowess":
                    frac = float(self.frac_var.get())
                    it = int(self.it_var.get())
                    smoothed_range = lowess(y_range, x_range, 
                                          frac=frac, it=it, return_sorted=False)
                
                # 更新对应范围的数据
                self.smoothed_data[mask] = smoothed_range
            
            # 更新图形显示
            self.plot_smooth_result()
            messagebox.showinfo("成功", "平滑处理完成！")
            
        except Exception as e:
            messagebox.showerror("错误", f"平滑处理出错：{str(e)}")

    def update_baseline_params(self):
        """根据选的基线校正法更新参数设置"""
        # 清除现有参数设置
        for widget in self.baseline_param_frame.winfo_children():
            widget.destroy()
            
        method = self.baseline_method.get()
        
        if method == "rubberband":
            # Rubberband方法参数
            ttk.Label(self.baseline_param_frame, text="点数:").pack()
            self.num_points_var = tk.StringVar(value="100")
            ttk.Entry(self.baseline_param_frame, textvariable=self.num_points_var).pack()

        elif method == "modpoly":
            # 修正多项式参数
            ttk.Label(self.baseline_param_frame, text="多项式阶数:").pack()
            self.poly_order_var = tk.StringVar(value="2")
            ttk.Entry(self.baseline_param_frame, textvariable=self.poly_order_var).pack()

            
        elif method == "imodpoly":
            # 自适应迭代多项式参数
            ttk.Label(self.baseline_param_frame, text="多项式阶数:").pack()
            self.poly_order_var = tk.StringVar(value="3")
            ttk.Entry(self.baseline_param_frame, textvariable=self.poly_order_var).pack()
            
            ttk.Label(self.baseline_param_frame, text="迭代次数:").pack()
            self.num_iter_var = tk.StringVar(value="100")
            ttk.Entry(self.baseline_param_frame, textvariable=self.num_iter_var).pack()
            
        elif method == "asls":
            # Whittaker-ASLS参数
            ttk.Label(self.baseline_param_frame, text="平滑参数():").pack()
            self.lam_var = tk.StringVar(value="1e7")
            ttk.Entry(self.baseline_param_frame, textvariable=self.lam_var).pack()
            
            ttk.Label(self.baseline_param_frame, text="非对称参数(p):").pack()
            self.p_var = tk.StringVar(value="0.01")
            ttk.Entry(self.baseline_param_frame, textvariable=self.p_var).pack()
            
        elif method == "mixture_model":
            # 平滑样条参数
            ttk.Label(self.baseline_param_frame, text="样条节点数:").pack()
            self.num_knots_var = tk.StringVar(value="10")
            ttk.Entry(self.baseline_param_frame, textvariable=self.num_knots_var).pack()
            
    def correct_baseline(self):
        """执行基线校正"""
        if self.x_data is None:
            messagebox.showerror("错误", "请先加载数据！")
            return
            
        try:
            method = self.baseline_method.get()
            data_source = self.data_source_var.get()
            
            # 选择数据源
            if data_source == "smoothed" and self.smoothed_data is not None:
                y_data = self.smoothed_data
            else:
                y_data = self.y_data
            
            # 根据不同法执行基线校正
            if method == "rubberband":
                baseline = self.baseline_fitter.rubberband(
                    y_data, 
                    num_knots=int(self.num_points_var.get())
                )[0]
                
            elif method == "modpoly":
                baseline = polynomial.modpoly(
                    y_data,
                    poly_order=int(self.poly_order_var.get())
                )[0]
                
            elif method == "imodpoly":
                baseline = self.baseline_fitter.imodpoly(
                    y_data,
                    poly_order=int(self.poly_order_var.get()),
                    max_iter=int(self.num_iter_var.get())
                )[0]
                
            elif method == "asls":
                baseline = whittaker.asls(
                    y_data,
                    lam=float(self.lam_var.get()),
                    p=float(self.p_var.get())
                )[0]
                
            elif method == "mixture_model":
                baseline = spline.mixture_model(
                    y_data,
                    num_knots=int(self.num_knots_var.get())
                )[0]
            
            self.corrected_data = y_data - baseline
            
            # 图形显示
            self.plot_baseline_result(baseline, y_data, self.x_data)
            messagebox.showinfo("成功", "线校正完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"基线校正出错：{str(e)}")

    def plot_data(self):
        # 更新平滑处理页面的图形
        self.smooth_ax1.clear()
        self.smooth_ax2.clear()
        self.smooth_ax1.plot(self.x_data, self.y_data, 'b-', label='原始数据')
        self.smooth_ax1.set_title('原始数据')
        self.smooth_ax1.set_xlabel('波数 (cm^-1)')
        self.smooth_ax1.set_ylabel(self.y_label_var.get())
        self.smooth_ax1.legend()
        self.smooth_ax1.grid(True)
        
        # 如果存在平滑数据，则显示
        if self.smoothed_data is not None:
            self.smooth_ax2.plot(self.x_data, self.smoothed_data, 'r-', label='滑后数据')
            self.smooth_ax2.set_title('平滑后数据')
            self.smooth_ax2.set_xlabel('波数 (cm^-1)')
            self.smooth_ax2.set_ylabel(self.y_label_var.get())
            self.smooth_ax2.legend()
            self.smooth_ax2.grid(True)
        
        self.smooth_fig.tight_layout()
        self.smooth_canvas.draw()
        
        # 更新基线校正页面的图形
        self.baseline_ax1.clear()
        self.baseline_ax2.clear()
        
        # 据数据源选择显的数据
        if self.data_source_var.get() == "smoothed" and self.smoothed_data is not None:
            plot_data = self.smoothed_data
            data_label = '平滑后据'
        else:
            plot_data = self.y_data
            data_label = '原始数据'
        
        self.baseline_ax1.plot(self.x_data, plot_data, 'b-', label=data_label)
        self.baseline_ax1.set_title(data_label)
        self.baseline_ax1.set_xlabel('波数 (cm^-1)')
        self.baseline_ax1.set_ylabel(self.y_label_var.get())
        self.baseline_ax1.legend()
        self.baseline_ax1.grid(True)
        self.baseline_fig.tight_layout()
        self.baseline_canvas.draw()

    def plot_smooth_result(self):
        self.smooth_ax2.clear()
        self.smooth_ax2.plot(self.x_data, self.smoothed_data, 'r-', label='平滑后数据')
        self.smooth_ax2.set_title('平滑后数据')
        self.smooth_ax2.set_xlabel('波数 (cm^-1)')
        self.smooth_ax2.set_ylabel(self.y_label_var.get())
        self.smooth_ax2.legend()
        self.smooth_ax2.grid(True)
        self.smooth_fig.tight_layout()
        self.smooth_canvas.draw()

    def plot_baseline_result(self, baseline, plot_data, x_data):
        self.baseline_ax1.clear()
        self.baseline_ax2.clear()
        
        # 根据数据源选择显示正确的标签
        data_labels = {
            "original": "原始据",
            "smoothed": "平滑后数据",
            "new_data": "新加载数据"
        }
        data_label = data_labels.get(self.data_source_var.get(), "数据")
        
        # 绘制数据和基线
        self.baseline_ax1.plot(x_data, plot_data, 'b-', label=data_label)
        self.baseline_ax1.plot(x_data, baseline, 'r--', label='基线')
        self.baseline_ax1.set_title(f'{data_label}和基线')
        self.baseline_ax1.set_xlabel('波数 (cm^-1)')
        self.baseline_ax1.set_ylabel(self.y_label_var.get())
        self.baseline_ax1.legend()
        self.baseline_ax1.grid(True)
        
        # 绘制校正后的数据
        self.baseline_ax2.plot(x_data, self.corrected_data, 'g-', label='校正后数据')
        self.baseline_ax2.set_title('基线校后的数据')
        self.baseline_ax2.set_xlabel('波数 (cm^-1)')
        self.baseline_ax2.set_ylabel(self.y_label_var.get())
        self.baseline_ax2.legend()
        self.baseline_ax2.grid(True)
        
        self.baseline_fig.tight_layout()
        self.baseline_canvas.draw()

    def export_smooth_data(self):
        if self.smoothed_data is None:
            messagebox.showerror("错误", "没有可导出的平数据")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if file_path:
                results = pd.DataFrame({
                    'x': self.x_data,
                    'raw_y': self.y_data,
                    'smoothed_y': self.smoothed_data
                })
                results.to_csv(file_path, index=False)
                messagebox.showinfo("成功", "数据导出成功！")
        except Exception as e:
            messagebox.showerror("错误", f"数导出出错：{str(e)}")

    def export_baseline_data(self):
        if self.corrected_data is None:
            messagebox.showerror("错误", "没有可导出的校正数据！")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if file_path:
                results = pd.DataFrame({
                    'x': self.x_data,
                    'raw_y': self.y_data,
                    'corrected_y': self.corrected_data
                })
                results.to_csv(file_path, index=False)
                messagebox.showinfo("成功", "数据导出成功！")
        except Exception as e:
            messagebox.showerror("错误", f"数据导出出错：{str(e)}")

    def update_baseline_plot(self):
        """更新基线校正页面的图形显示"""
        if self.x_data is None:
            return
            
        self.baseline_ax1.clear()
        self.baseline_ax2.clear()
        
        # 根据数据源选择显示的数据
        if self.data_source_var.get() == "smoothed" and self.smoothed_data is not None:
            plot_data = self.smoothed_data
            data_label = '平滑后数据'
        else:
            plot_data = self.y_data
            data_label = '原始数据'
        
        self.baseline_ax1.plot(self.x_data, plot_data, 'b-', label=data_label)
        self.baseline_ax1.set_title(data_label)
        self.baseline_ax1.set_xlabel('波数 (cm^-1)')
        self.baseline_ax1.set_ylabel(self.y_label_var.get())
        self.baseline_ax1.legend()
        self.baseline_ax1.grid(True)
        
        self.baseline_fig.tight_layout()
        self.baseline_canvas.draw()

    def update_file_display(self, filename):
        """更新文件名显示"""
        self.current_file_var.set(filename)
        if hasattr(self, 'file_label'):
            self.file_label.config(text=filename)
        self.root.update_idletasks()

    def show_help_window(self, title, text):
        """显示帮助窗口"""
        help_window = tk.Toplevel(self.root)
        help_window.title(title)
        help_window.geometry("600x700")
        
        # 创建文本框和滚动条
        text_frame = ttk.Frame(help_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(
            text_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=35,
            font=('SimHei', 10),
            padx=10,
            pady=10
        )
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # 插入帮助文本
        text_widget.insert(tk.END, text)
        text_widget.config(state=tk.DISABLED)
        
        # 添加关闭按钮
        ttk.Button(
            help_window, 
            text="关闭", 
            width=10,
            command=help_window.destroy
        ).pack(pady=10)

    def show_smoothing_help(self):
        """显示平滑方法说明"""
        help_text = """
        平滑方法说明：
        
        1. Savitzky-Golay滤（最适合FTIR光谱分析的平滑法）
        - 最适FTIR光谱分析平滑方法
        - 通过局部多项式回归计算平滑值
        - 能很好地保持峰的形状和位置
        - 适用于需要保持光谱精细结构的情况
        
        2. LOWESS（局部平滑效果好）
        - 局部加权散点平滑法
        - 适用于非线性趋势数据
        - 计算较慢但效果好
        
        3. 移动平均（简单数据平滑）
        - 简单的数据平滑方法
        - 计算滑动窗口内数据点的平均值
        - 适用于噪声较大但峰形状不重要的情况
        
        4. 高斯滤波（对正态分布噪声效果较好）
        - 使用高斯函数作为权重的加权平均
        - 对正态分布噪声效果较好
        - 平滑效果温和，不会过度失真
        
        5. 中值滤波（对突变噪声效果好）
        - 取滑动窗口内的中位数
        - 对突变噪声（离群值）效果好
        - 可能会改变峰的形状
        """
        self.show_help_window("平滑方法说明", help_text)

    def show_baseline_help(self):
        """显示基线校正方法说明"""
        help_text = """
        基线校正方法说明：
        
        1. Rubberband（橡皮带法，最合FTIR线校正）
        - 橡皮带法，最适合FTIR基线校正
        - 在光谱下方创建凸包
        - 自动识别基线点
        - 适用基线漂移明显的谱
        
        2. 修正多项式（传统多项式拟合方法）
        - 传统多项式拟合方法
        - 简单直观，计算快速
        - 适于简单基线漂移
        
        3. 自适应迭代多项式（自动寻找最优多项式拟合）
        - 自动寻找最优多项式拟合
        - 迭代优化以找到最佳基线
        - 适用于复杂基线形状
        
        4. Whittaker-ASLS（结合平滑和基线校正）
        - 结合平滑和基线校正
        - 自适应权重迭代
        - 对噪声数据效果好
        
        5. 平滑样条（灵活性好，可调节平滑度）
        - 使用样条数拟合基线
        - 灵活性好，可调节平滑度
        - 适用于连续变化的基线
        """
        self.show_help_window("基线校正方法说明", help_text)

    def create_peak_analysis_page(self):
        """创建特征峰分析页面"""
        # 创建左侧控制面板和右侧图形
        control_frame = ttk.Frame(self.peak_analysis_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        plot_frame = ttk.Frame(self.peak_analysis_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 数据选择框架
        data_frame = ttk.LabelFrame(control_frame, text="数据选择")
        data_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 初始化数据选择变量并设置默认值
        self.peak_data_var = tk.StringVar(value="original")
        
        # 创建单选按钮
        ttk.Radiobutton(data_frame, text="原始数据", 
                       variable=self.peak_data_var, 
                       value="original",
                       command=self.update_peak_plot).pack(anchor=tk.W)
        ttk.Radiobutton(data_frame, text="平滑后数据", 
                       variable=self.peak_data_var, 
                       value="smoothed",
                       command=self.update_peak_plot).pack(anchor=tk.W)
        ttk.Radiobutton(data_frame, text="基线校正后数据", 
                       variable=self.peak_data_var, 
                       value="corrected",
                       command=self.update_peak_plot).pack(anchor=tk.W)
        
        # 峰检测设置框架
        peak_settings_frame = ttk.LabelFrame(control_frame, text="峰检测设置")
        peak_settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 阈值设置
        threshold_frame = ttk.Frame(peak_settings_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        ttk.Label(threshold_frame, text="阈值:").pack(side=tk.LEFT)
        self.peak_threshold_var = tk.StringVar(value="0.1")
        ttk.Entry(threshold_frame, textvariable=self.peak_threshold_var, width=10).pack(side=tk.LEFT)
        
        # 最小距离设置
        distance_frame = ttk.Frame(peak_settings_frame)
        distance_frame.pack(fill=tk.X, pady=2)
        ttk.Label(distance_frame, text="最小距离:").pack(side=tk.LEFT)
        self.peak_distance_var = tk.StringVar(value="10")
        ttk.Entry(distance_frame, textvariable=self.peak_distance_var, width=10).pack(side=tk.LEFT)
        
        # 寻峰按钮
        ttk.Button(peak_settings_frame, text="寻找峰", 
                   command=self.find_peaks).pack(fill=tk.X, pady=2)
        
        # 峰列表框架
        peaks_frame = ttk.LabelFrame(control_frame, text="峰列表")
        peaks_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 添加列表标题
        header_frame = ttk.Frame(peaks_frame)
        header_frame.pack(fill=tk.X)
        ttk.Label(header_frame, text="波数(cm⁻¹)    高度").pack()
        
        # 创建带滚动条的列表
        list_frame = ttk.Frame(peaks_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.peaks_listbox = tk.Listbox(list_frame, height=10, 
                                       yscrollcommand=scrollbar.set,
                                       font=('Courier', 10))
        self.peaks_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.peaks_listbox.yview)
        
        # 添加选择事件绑定
        self.peaks_listbox.bind('<<ListboxSelect>>', self.on_peak_select)
        
        # 取消选择按钮
        ttk.Button(peaks_frame, text="取消选择", 
                   command=self.clear_peak_selection).pack(fill=tk.X, pady=2)
        
        # 导出按钮
        ttk.Button(peaks_frame, text="导出峰列表", 
                   command=self.export_peak_list).pack(fill=tk.X, pady=2)
        
        # 峰分析设置
        analysis_frame = ttk.LabelFrame(control_frame, text="峰分析设置")
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 波数范围选择
        lower_frame = ttk.Frame(analysis_frame)
        lower_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lower_frame, text="下限:").pack(side=tk.LEFT)
        self.peak_lower_var = tk.StringVar()
        self.peak_lower_var.trace_add("write", self.on_range_change)  # 添加回调
        ttk.Entry(lower_frame, textvariable=self.peak_lower_var, width=10).pack(side=tk.LEFT, padx=2)
        
        upper_frame = ttk.Frame(analysis_frame)
        upper_frame.pack(fill=tk.X, pady=2)
        ttk.Label(upper_frame, text="上限:").pack(side=tk.LEFT)
        self.peak_upper_var = tk.StringVar()
        self.peak_upper_var.trace_add("write", self.on_range_change)  # 添加回调
        ttk.Entry(upper_frame, textvariable=self.peak_upper_var, width=10).pack(side=tk.LEFT, padx=2)
        
        # 分析按钮
        ttk.Button(analysis_frame, text="分析选中峰", 
                   command=self.analyze_selected_peak).pack(fill=tk.X, pady=2)
        
        # 结果显示框
        result_frame = ttk.LabelFrame(control_frame, text="分析结果")
        result_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.result_text = tk.Text(result_frame, height=6, width=30)
        self.result_text.pack(fill=tk.X, pady=2)
        
        # 复制结果按钮
        ttk.Button(result_frame, text="复制结果", 
                   command=self.copy_results).pack(fill=tk.X, pady=2)
        
        # 创建图形
        self.peak_fig, self.peak_ax = plt.subplots(figsize=(8, 6))
        self.peak_canvas = FigureCanvasTkAgg(self.peak_fig, master=plot_frame)
        self.peak_canvas.draw()
        self.peak_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        toolbar = NavigationToolbar2Tk(self.peak_canvas, plot_frame)
        toolbar.update()
        
        # 初始化显示原始数据
        self.update_peak_plot()

    def on_range_change(self, *args):
        """当上下限输入框的值改变时更新图形"""
        self.update_peak_plot()

    def update_peak_plot(self):
        """更新特征峰分析图形"""
        if not hasattr(self, 'x_data') or self.x_data is None:
            return
        
        self.peak_ax.clear()
        
        # 获取当前选择的数据
        if self.peak_data_var.get() == "smoothed" and self.smoothed_data is not None:
            y_data = self.smoothed_data
            self.peak_ax.plot(self.x_data, y_data, 'g-', label='平滑后数据')
        elif self.peak_data_var.get() == "corrected" and self.corrected_data is not None:
            y_data = self.corrected_data
            self.peak_ax.plot(self.x_data, y_data, 'r-', label='基线校正后数据')
        else:
            y_data = self.y_data
            self.peak_ax.plot(self.x_data, y_data, 'b-', label='原始数据')
        
        # 获取当前选中的峰的索引
        selected_indices = self.peaks_listbox.curselection()
        
        # 绘制所有峰值点
        for i in range(self.peaks_listbox.size()):
            peak_wavenumber = float(self.peaks_listbox.get(i).split()[0])
            peak_idx = np.argmin(np.abs(self.x_data - peak_wavenumber))
            peak_height = y_data[peak_idx]
            
            if i in selected_indices:
                # 选中的峰用绿色圆点标记
                self.peak_ax.plot(peak_wavenumber, peak_height, 'go', 
                                markersize=8, label='选中峰' if i == selected_indices[0] else "")
            else:
                # 未选中的峰用蓝色圆点标记
                self.peak_ax.plot(peak_wavenumber, peak_height, 'bo', 
                                markersize=8, label='峰值' if i == 0 and not selected_indices else "")
        
        # 绘制上下限虚线和连接线
        try:
            if self.peak_lower_var.get() and self.peak_upper_var.get():
                lower = float(self.peak_lower_var.get())
                upper = float(self.peak_upper_var.get())
                
                # 找到上下限对应的y值
                lower_idx = np.argmin(np.abs(self.x_data - lower))
                upper_idx = np.argmin(np.abs(self.x_data - upper))
                lower_y = y_data[lower_idx]
                upper_y = y_data[upper_idx]
                
                # 绘制竖向虚线（深灰色）
                self.peak_ax.axvline(x=lower, color='dimgray', linestyle='--', alpha=0.8)
                self.peak_ax.axvline(x=upper, color='dimgray', linestyle='--', alpha=0.8)
                
                # 绘制连接线（粉色）
                self.peak_ax.plot([lower, upper], [lower_y, upper_y], 
                                color='black', linestyle='--', alpha=0.8)
                
        except ValueError:
            pass  # 忽略无效的输入值
        
        self.peak_ax.set_xlabel('波数 (cm^-1)')
        self.peak_ax.set_ylabel('吸光度')
        self.peak_ax.legend()
        self.peak_ax.grid(True)
        self.peak_fig.tight_layout()
        self.peak_canvas.draw()

    def find_peaks(self):
        """自动寻峰功能"""
        try:
            # 获取当前选择的数据
            if self.peak_data_var.get() == "smoothed":
                y_data = self.smoothed_data
            elif self.peak_data_var.get() == "corrected":
                y_data = self.corrected_data
            else:
                y_data = self.y_data
            
            if y_data is None:
                messagebox.showerror("错误", "没有可用的数据")
                return
            
            # 获取参数
            threshold = float(self.peak_threshold_var.get())
            distance = int(self.peak_distance_var.get())
            
            # 使用scipy.signal.find_peaks寻找峰
            from scipy.signal import find_peaks as scipy_find_peaks
            peaks, properties = scipy_find_peaks(y_data, 
                                                  height=threshold,
                                                  distance=distance)
            
            # 清空现有峰列表
            self.peaks_listbox.delete(0, tk.END)
            
            # 添加找到的峰（波数和峰高）
            for peak in peaks:
                wavenumber = self.x_data[peak]
                height = y_data[peak]
                self.peaks_listbox.insert(tk.END, f"{wavenumber:.2f}    {height:.4f}")
            
            # 更新图形
            self.update_peak_plot()
            
        except Exception as e:
            messagebox.showerror("错误", f"寻峰出错：{str(e)}")

    def clear_peak_selection(self):
        """取消峰列表的选择和分析范围"""
        try:
            # 清除列表框中的选择
            self.peaks_listbox.selection_clear(0, tk.END)
            
            # 清除分析范围
            self.peak_lower_var.set("")
            self.peak_upper_var.set("")
            
            # 更新图形显示
            self.update_peak_plot()
        except Exception as e:
            messagebox.showerror("错误", f"取消选择出错：{str(e)}")

    def export_peak_list(self):
        """导出峰列表"""
        if not self.peaks_listbox.size():
            messagebox.showwarning("警告", "峰列表为空")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if file_path:
                peaks = []
                heights = []
                for i in range(self.peaks_listbox.size()):
                    wavenumber, height = self.peaks_listbox.get(i).split()
                    peaks.append(float(wavenumber))
                    heights.append(float(height))
                
                df = pd.DataFrame({
                    "峰位置(cm^-1)": peaks,
                    "峰高度": heights
                })
                df.to_csv(file_path, index=False)
                messagebox.showinfo("成功", "峰列表导出成功！")
        
        except Exception as e:
            messagebox.showerror("错误", f"导出出错：{str(e)}")

    def analyze_selected_peak(self):
        """分析选中的峰"""
        selection = self.peaks_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要分析的峰")
            return
        
        if not self.peak_lower_var.get() or not self.peak_upper_var.get():
            messagebox.showwarning("警告", "请先设置分析范围")
            return
        
        try:
            # 获取当前选择的数据
            if self.peak_data_var.get() == "smoothed":
                y_data = self.smoothed_data
            elif self.peak_data_var.get() == "corrected":
                y_data = self.corrected_data
            else:
                y_data = self.y_data
            
            # 获取选中峰的波数
            peak_wavenumber = float(self.peaks_listbox.get(selection[0]).split()[0])
            
            # 获取分析范围
            lower = float(self.peak_lower_var.get())
            upper = float(self.peak_upper_var.get())
            
            # 获取范围内的数据
            mask = (self.x_data >= lower) & (self.x_data <= upper)
            x_range = self.x_data[mask]
            y_range = y_data[mask]
            
            # 计算基线（使用直线连接两端点）
            y_baseline = np.interp(x_range, [x_range[0], x_range[-1]], 
                                 [y_range[0], y_range[-1]])
            
            # 计算峰高度（未校正和校正后）
            peak_idx = np.argmin(np.abs(x_range - peak_wavenumber))
            uncorrected_height = y_range[peak_idx]
            baseline_height = y_baseline[peak_idx]
            corrected_height = uncorrected_height - baseline_height
            
            # 计算峰面积（未校正和校正后）
            uncorrected_area = np.trapz(y_range, x_range)
            corrected_area = np.trapz(y_range - y_baseline, x_range)
            
            # 整理结果
            results = {
                "波数": f"{peak_wavenumber:.2f} cm^-1",
                "未校正峰高": uncorrected_height,
                "校正峰高": corrected_height,
                "未校正峰面积": uncorrected_area,
                "校正峰面积": corrected_area
            }
            
            self.display_peak_results(results)
            
        except Exception as e:
            messagebox.showerror("错误", f"峰分析出错：{str(e)}")

    def display_peak_results(self, results):
        """显示峰分析结果"""
        result_text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                for k, v in results.items()])
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", result_text)

    def copy_results(self):
        """复制分析结果到剪贴板"""
        try:
            # 获取结果文本框中的内容
            results = self.result_text.get("1.0", tk.END).strip()
            if results:
                # 复制到剪贴板
                self.root.clipboard_clear()
                self.root.clipboard_append(results)
                messagebox.showinfo("成功", "结果已复制到剪贴板")
            else:
                messagebox.showwarning("警告", "没有可复制的结果")
        except Exception as e:
            messagebox.showerror("错误", f"复制结果出错：{str(e)}")

    def on_peak_select(self, event):
        """当峰列表选择改变时更新图形"""
        self.update_peak_plot()

    # ========== 新增功能方法 ==========

    def save_config(self):
        """保存当前配置"""
        if not HAS_ENHANCED_FEATURES:
            messagebox.showwarning("警告", "增强功能模块未安装")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                title="保存配置",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if file_path:
                config = ConfigManager.create_config_from_gui(self)
                self.config_manager.save_config(config, file_path)
                messagebox.showinfo("成功", "配置保存成功！")

        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败：{str(e)}")

    def load_config(self):
        """加载配置"""
        if not HAS_ENHANCED_FEATURES:
            messagebox.showwarning("警告", "增强功能模块未安装")
            return

        try:
            file_path = filedialog.askopenfilename(
                title="加载配置",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if file_path:
                config = self.config_manager.load_config(file_path)
                ConfigManager.apply_config_to_gui(config, self)
                messagebox.showinfo("成功", "配置加载成功！")

        except Exception as e:
            messagebox.showerror("错误", f"加载配置失败：{str(e)}")

    def open_batch_dialog(self):
        """打开批量处理对话框"""
        if not HAS_ENHANCED_FEATURES:
            messagebox.showwarning("警告", "增强功能模块未安装")
            return

        BatchDialog(self.root, self)

    def export_full_report_excel(self):
        """导出完整报告到Excel"""
        if not HAS_ENHANCED_FEATURES:
            messagebox.showwarning("警告", "增强功能模块未安装")
            return

        if self.x_data is None:
            messagebox.showerror("错误", "没有可导出的数据")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                title="导出完整报告",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )

            if file_path:
                # 准备峰数据
                peaks_data = None
                if hasattr(self, 'peaks_listbox') and self.peaks_listbox.size() > 0:
                    peaks = []
                    heights = []
                    for i in range(self.peaks_listbox.size()):
                        parts = self.peaks_listbox.get(i).split()
                        peaks.append(float(parts[0]))
                        heights.append(float(parts[1]))
                    peaks_data = {
                        "峰位置(cm^-1)": peaks,
                        "峰高度": heights
                    }

                # 获取当前配置
                config = ConfigManager.create_config_from_gui(self)

                # 导出
                self.export_manager.export_to_excel(
                    file_path,
                    self.x_data,
                    self.y_data,
                    self.smoothed_data,
                    self.corrected_data,
                    peaks_data,
                    config
                )

                messagebox.showinfo("成功", f"完整报告已导出到：\n{file_path}")

        except Exception as e:
            messagebox.showerror("错误", f"导出失败：{str(e)}")

    def export_with_metadata(self):
        """导出带元数据的CSV文件"""
        if not HAS_ENHANCED_FEATURES:
            messagebox.showwarning("警告", "增强功能模块未安装")
            return

        if self.x_data is None:
            messagebox.showerror("错误", "没有可导出的数据")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                title="导出数据（带元数据）",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if file_path:
                config = ConfigManager.create_config_from_gui(self)
                self.export_manager.export_with_metadata(
                    file_path,
                    self.x_data,
                    self.y_data,
                    self.smoothed_data,
                    self.corrected_data,
                    config
                )

                messagebox.showinfo("成功", f"数据已导出到：\n{file_path}")

        except Exception as e:
            messagebox.showerror("错误", f"导出失败：{str(e)}")

    def show_about(self):
        """显示关于对话框"""
        about_text = """
FTIR 光谱处理程序
版本: 2.0 (增强版)

作者: zjnuxsl
邮箱: sl-xiao@zjnu.cn

功能特性:
• 5种平滑方法
• 5种基线校正方法
• 特征峰分析
• 批量文件处理
• 配置保存/加载
• 多格式导出

© 2024 All Rights Reserved
        """

        messagebox.showinfo("关于", about_text.strip())


class BatchDialog:
    """批量处理对话框"""

    def __init__(self, parent, main_gui):
        self.main_gui = main_gui
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("批量处理")
        self.dialog.geometry("600x500")

        # 文件列表
        list_frame = ttk.LabelFrame(self.dialog, text="待处理文件")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建列表框和滚动条
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.file_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.file_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.file_listbox.yview)

        # 文件路径存储
        self.file_paths = []

        # 按钮区域
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(btn_frame, text="添加文件", command=self.add_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="移除选中", command=self.remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="清空列表", command=self.clear_list).pack(side=tk.LEFT, padx=5)

        # 输出目录选择
        output_frame = ttk.LabelFrame(self.dialog, text="输出目录")
        output_frame.pack(fill=tk.X, padx=10, pady=5)

        self.output_dir_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        ttk.Button(output_frame, text="选择目录", command=self.select_output_dir).pack(side=tk.RIGHT, padx=5, pady=5)

        # 进度显示
        progress_frame = ttk.LabelFrame(self.dialog, text="处理进度")
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress_var = tk.StringVar(value="就绪")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(padx=5, pady=5)

        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

        # 开始处理按钮
        control_frame = ttk.Frame(self.dialog)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(control_frame, text="开始批量处理", command=self.start_batch_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="关闭", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def add_files(self):
        """添加文件"""
        files = filedialog.askopenfilenames(
            title="选择数据文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        for file in files:
            if file not in self.file_paths:
                self.file_paths.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))

    def remove_selected(self):
        """移除选中文件"""
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            self.file_listbox.delete(index)
            self.file_paths.pop(index)

    def clear_list(self):
        """清空列表"""
        self.file_listbox.delete(0, tk.END)
        self.file_paths = []

    def select_output_dir(self):
        """选择输出目录"""
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_dir_var.set(directory)

    def start_batch_processing(self):
        """开始批量处理"""
        if not self.file_paths:
            messagebox.showwarning("警告", "请先添加要处理的文件")
            return

        if not self.output_dir_var.get():
            messagebox.showwarning("警告", "请选择输出目录")
            return

        # 获取当前配置
        config = ConfigManager.create_config_from_gui(self.main_gui)

        # 准备处理函数
        processing_functions = {
            'smoothing': self._create_smoothing_function(),
            'baseline': self._create_baseline_function(),
            'peak_analysis': self._create_peak_analysis_function()
        }

        # 创建批量处理器
        processor = BatchProcessor(config, processing_functions)

        # 进度回调
        def progress_callback(current, total, filename):
            self.progress_var.set(f"处理中: {filename} ({current}/{total})")
            self.progress_bar['value'] = (current / total) * 100
            self.dialog.update()

        # 在单独线程中处理
        def process_thread():
            try:
                summary = processor.process_files(
                    self.file_paths,
                    self.output_dir_var.get(),
                    progress_callback
                )

                # 保存摘要
                processor.save_summary(self.output_dir_var.get())

                # 显示结果
                self.progress_var.set(f"完成！成功: {summary['successful']}, 失败: {summary['failed']}")
                messagebox.showinfo(
                    "批量处理完成",
                    f"处理完成！\n\n成功: {summary['successful']}\n失败: {summary['failed']}\n\n结果已保存到: {self.output_dir_var.get()}"
                )

            except Exception as e:
                messagebox.showerror("错误", f"批量处理失败：{str(e)}")
                self.progress_var.set("错误")

        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()

    def _create_smoothing_function(self):
        """创建平滑处理函数"""
        def smooth_func(x_data, y_data, config):
            method = config['method']
            params = config['params']

            if method == "savgol":
                return savgol_filter(y_data, int(params['window_length']), int(params['polyorder']))
            elif method == "moving_average":
                window_length = int(params['window_length'])
                kernel = np.ones(window_length) / window_length
                return np.convolve(y_data, kernel, mode='same')
            elif method == "gaussian":
                return gaussian_filter1d(y_data, float(params['sigma']))
            elif method == "median":
                return medfilt(y_data, int(params['window_length']))
            elif method == "lowess":
                return lowess(y_data, x_data, frac=float(params['frac']), it=int(params['it']), return_sorted=False)

            return y_data

        return smooth_func

    def _create_baseline_function(self):
        """创建基线校正函数"""
        def baseline_func(x_data, y_data, config):
            method = config['method']
            params = config['params']
            baseline_fitter = Baseline()

            if method == "rubberband":
                baseline = baseline_fitter.rubberband(y_data, num_knots=int(params['num_points']))[0]
            elif method == "modpoly":
                baseline = polynomial.modpoly(y_data, poly_order=int(params['poly_order']))[0]
            elif method == "imodpoly":
                baseline = baseline_fitter.imodpoly(y_data, poly_order=int(params['poly_order']), max_iter=int(params['num_iter']))[0]
            elif method == "asls":
                baseline = whittaker.asls(y_data, lam=float(params['lam']), p=float(params['p']))[0]
            elif method == "mixture_model":
                baseline = spline.mixture_model(y_data, num_knots=int(params['num_knots']))[0]
            else:
                baseline = np.zeros_like(y_data)

            return y_data - baseline

        return baseline_func

    def _create_peak_analysis_function(self):
        """创建峰分析函数"""
        def peak_func(x_data, y_data, config):
            threshold = float(config['threshold'])
            distance = int(config['distance'])

            peaks, properties = find_peaks(y_data, height=threshold, distance=distance)

            if len(peaks) > 0:
                return {
                    "峰位置(cm^-1)": [x_data[p] for p in peaks],
                    "峰高度": [y_data[p] for p in peaks]
                }

            return None

        return peak_func


# 在文件末尾添加以下代码
def main():
    root = tk.Tk()
    root.title("FTIR-Smoother")
    
    # 获取屏幕尺寸
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # 计算16:9比例的窗口大小
    window_height = int(screen_height * 0.9)  # 使用90%的屏幕高度
    window_width = int(window_height * 16 / 9)  # 16:9比例
    
    # 确保窗口宽度不超过屏幕宽度
    if window_width > screen_width:
        window_width = int(screen_width * 0.9)
        window_height = int(window_width * 9 / 16)
    
    # 计算窗口位置，使其居中
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    
    # 设置窗口大小和位置
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    # 设置窗口最大化
    root.state('zoomed')  # Windows系统使用'zoomed'
    # 如果是Linux或Mac系统，使用：
    # root.attributes('-zoomed', True)  # Linux
    # root.attributes('-fullscreen', True)  # Mac
    
    app = SpectralProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()