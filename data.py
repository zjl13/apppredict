import datasets as ds

# 下载 Rico 数据集中的 UI 截图和视图层次结构
dataset = ds.load_dataset(
    path="shunk031/Rico",
    name="ui-screenshots-and-view-hierarchies",
)

# 打印下载的数据集结构
print(dataset)
