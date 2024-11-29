
1. **查看状态和提交更改** (最基础的工作流)
```bash
# 查看当前状态
git status

# 查看具体修改
git diff

# 添加并提交修改
git add .
git commit -m "更新：优化数据处理算法"

# 推送到远程
git push origin main
```

2. **撤销修改** (当你需要取消更改时)
```bash
# 撤销未暂存的修改（还没 git add）
git checkout -- FTIR_Processor.py

# 撤销已暂存的修改（已经 git add）
git reset HEAD FTIR_Processor.py
```

3. **临时保存工作** (当需要切换任务时)
```bash
# 保存当前工作
git stash

# 恢复之前的工作
git stash pop
```

4. **查看历史** (检查修改记录)
```bash
# 查看简洁的提交历史
git log --oneline

# 查看某个文件的修改历史
git log -p FTIR_Processor.py
```

5. **同步远程代码** (与远程仓库同步)
```bash
# 获取远程更新
git pull origin main

# 推送本地更改
git push origin main
```

#实际使用场景示例：
```bash
# 场景1：日常开发提交代码
git status                                    # 查看修改
git add .                                     # 添加修改
git commit -m "新增：添加数据导出功能"         # 提交修改
git push origin main                          # 推送到远程

# 场景2：撤销错误修改
git checkout -- FTIR_Processor.py             # 撤销错误的修改

# 场景3：临时切换任务
git stash                                     # 保存当前工作
# 做其他任务
git stash pop                                 # 恢复之前的工作

# 场景4：查看修改历史
git log --oneline                             # 查看提交历史
```

# 提交信息的常用格式：
```bash
# 新功能
git commit -m "新增：添加数据导出功能"

# 修复问题
git commit -m "修复：解决大文件崩溃问题"

# 改进功能
git commit -m "优化：提升数据处理速度"

# 更新文档
git commit -m "文档：更新安装说明"
``` 

# 建议
```bash
经常使用 git status 查看状态
提交前用 git diff 检查修改
养成写清晰提交信息的习惯
定期推送到远程仓库备份
``` 