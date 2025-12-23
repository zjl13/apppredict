import requests
from dataset.py import labels,base_dir
from PIL import Image
import os
import shutil


API_KEY = "sk-3ec2a404dfc74ecea15d449deacec3ce"  # 替换为你的 API Key
API_URL = "https://api.qwen-vl.com/v1/predict"  # 替换为实际 URL

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 你的 Prompt，向 Qwen-VL 询问场景类型
prompt = """你是一个移动应用界面理解助手。
请判断该截图最可能属于以下哪一类 UI 场景：
1. 商品详情页
2. 商品列表页
3. 视频播放页
4. 社交信息流
5. 收藏/历史列表
6. 搜索结果页
7. 个人主页
8. 登录/注册页
9. 设置页
10. 订单/账单页
只返回最可能的一类名称。"""

def get_label_from_api(image_path):
    """发送图片请求给 Qwen-VL API，获取预测标签"""
    image = Image.open(image_path)
    image.save("temp.jpg", "JPEG")  # 保存临时文件供上传
    files = {'file': open("temp.jpg", 'rb')}
    data = {
        'prompt': prompt
    }
    response = requests.post(API_URL, headers=headers, files=files, data=data)
    files['file'].close()  # 关闭文件流
    
    # 解析返回的 JSON，返回分类标签
    if response.status_code == 200:
        result = response.json()
        return result.get('predicted_label', "Unknown")  # 获取预测标签
    else:
        print("API请求失败:", response.status_code, response.text)
        return None

# 从 Rico 数据集中提取图像并根据分类标签保存
def classify_and_save_images(dataset, base_dir):
    for item in dataset['train']:  # 只处理训练集
        img_path = item['screenshot']  # 获取图片路径
        label = get_label_from_api(img_path)  # 获取 Qwen-VL 预测的标签
        
        if label and label in labels:  # 如果标签有效且属于我们预定义的类别
            label_dir = os.path.join(base_dir, label)
            img_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(label_dir, img_name))  # 将图片复制到对应的文件夹

# 调用函数
classify_and_save_images(dataset, base_dir)
