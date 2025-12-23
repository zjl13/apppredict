import os
import shutil

# 定义 30 类标签
labels = [
    '商品详情页', '商品列表页', '视频播放页', '社交信息流', '收藏/历史列表', 
    '搜索结果页', '个人主页', '登录/注册页', '设置页', '订单/账单页',
    # 继续填充剩余的 20 类标签...
]

# 创建每个标签对应的文件夹
base_dir = "rico_screenshots_classified"  # 分类后的目录
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for label in labels:
    label_dir = os.path.join(base_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
