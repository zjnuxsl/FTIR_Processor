# FTIR 光谱处理器 - 脚本说明

本目录包含用于安装、运行和打包 FTIR 光谱处理器的批处理脚本。

## 📝 脚本列表

### 1. install.bat - 安装依赖

**用途**：首次使用时安装程序所需的依赖包

**功能**：
- 检查 Python 是否已安装
- 询问是否创建虚拟环境（推荐）
- 自动安装所有依赖包（numpy, pandas, matplotlib, scipy, statsmodels, pybaselines）
- 升级 pip 到最新版本

**使用方法**：
```bash
# 方法1：双击运行
双击 install.bat

# 方法2：命令行运行
cd scripts
install.bat
```

**何时使用**：
- ✅ 第一次使用程序时
- ✅ 更新依赖包时
- ✅ 重新安装环境时

---

### 2. run.bat - 启动程序

**用途**：启动 FTIR 光谱处理器

**功能**：
- 自动检测虚拟环境或系统 Python
- 检查依赖包是否已安装
- 如果缺少依赖，自动安装
- 启动主程序

**使用方法**：
```bash
# 方法1：双击运行（推荐）
双击 run.bat

# 方法2：命令行运行
cd scripts
run.bat
```

**何时使用**：
- ✅ 每次使用程序时
- ✅ 日常启动程序

---

### 3. build_exe.bat - 打包程序

**用途**：将程序打包为独立的 .exe 文件

**功能**：
- 安装 PyInstaller
- 打包所有依赖到单个 exe 文件
- 生成约 80 MB 的独立可执行文件
- 用户无需安装 Python 即可运行

**使用方法**：
```bash
# 方法1：双击运行
双击 build_exe.bat

# 方法2：命令行运行
cd scripts
build_exe.bat
```

**何时使用**：
- ✅ 需要分发给没有 Python 环境的用户
- ✅ 需要创建独立版本
- ✅ 需要简化用户安装流程

**生成文件**：
- `dist/FTIR_Processor.exe` - 独立可执行文件（约 80 MB）

---

## 🚀 快速开始

### 首次使用

1. **安装依赖**
   ```bash
   双击 install.bat
   ```
   - 选择 Y 创建虚拟环境（推荐）
   - 等待安装完成

2. **启动程序**
   ```bash
   双击 run.bat
   ```

### 日常使用

直接双击 `run.bat` 即可启动程序

---

## ⚠️ 注意事项

### 路径问题
- ✅ 所有脚本都会自动切换到项目根目录
- ✅ 可以从任何位置运行这些脚本
- ✅ 不需要手动 cd 到项目目录

### 虚拟环境
- **推荐使用虚拟环境**：依赖隔离，不影响其他项目
- **不使用虚拟环境**：依赖安装到系统 Python

### Python 版本
- 需要 Python 3.8 或更高版本
- 安装时勾选 "Add Python to PATH"

---

## 🔧 故障排除

### 问题1：提示 "Python not found"
**解决方法**：
1. 确保已安装 Python 3.8+
2. 检查 Python 是否添加到 PATH
3. 重新安装 Python，勾选 "Add Python to PATH"

### 问题2：依赖安装失败
**解决方法**：
1. 检查网络连接
2. 尝试以管理员身份运行
3. 手动安装：`pip install -r requirements.txt`

### 问题3：程序无法启动
**解决方法**：
1. 检查是否已运行 `install.bat`
2. 查看 `logs/ftir_processor.log` 日志文件
3. 尝试重新安装依赖

### 问题4：打包失败
**解决方法**：
1. 确保已安装所有依赖
2. 检查磁盘空间（需要约 500 MB）
3. 尝试手动安装 PyInstaller：`pip install pyinstaller`

---

## 📁 文件结构

```
scripts/
├── install.bat      # 安装脚本
├── run.bat          # 运行脚本
├── build_exe.bat    # 打包脚本
└── README.md        # 本说明文件
```

---

## 💡 提示

- **首次使用**：先运行 `install.bat`，再运行 `run.bat`
- **日常使用**：直接运行 `run.bat`
- **分发程序**：运行 `build_exe.bat` 生成独立版本
- **更新依赖**：重新运行 `install.bat`

---

## 📞 技术支持

如有问题，请查看：
- 项目根目录的 `README.md`
- `logs/ftir_processor.log` 日志文件

