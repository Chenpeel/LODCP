import os
import kagglehub
import shutil
import warnings
import openxlab
warnings.simplefilter("ignore")

data_dir = "data"

def prepare_dir():
    # 创建数据目录
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

def task1():
    kitti_dest = os.path.join(data_dir, "kitti-segmentation")
    if not os.path.exists(kitti_dest):
        print("正在下载KITTI分割数据集...")
        kitti_path = kagglehub.dataset_download("sakshaymahna/kittiroadsegmentation")
        shutil.move(kitti_path, kitti_dest)
    else:
        print("KITTI分割数据集已存在")
def task2():
    bdd_dest = os.path.join(data_dir, "bdd100k-dataset")
    if not os.path.exists(bdd_dest):
        print("正在下载BDD100K数据集...")
        bdd_path = kagglehub.dataset_download("awsaf49/bdd100k-dataset")
        shutil.move(bdd_path, bdd_dest)
    else:
        print("BDD100K数据集已存在")
def task3():
    kitti_dest = os.path.join(data_dir, "kitti-road")
    if not os.path.exists(kitti_dest):
        print("正在下载KITTI道路数据集...")
        kitti_path = kagglehub.dataset_download("sumanyughoshal/kitti-road-dataset")
        shutil.move(kitti_path, kitti_dest)
    else:
        print("KITTI道路数据集已存在")

def task4():
    ccpd_dest = os.path.join(data_dir,"ccpd-dataset")
    if not os.path.exists(ccpd_dest):
        print("正在下载CCPD数据集...")
        ccpd_path = kagglehub.dataset_download("mdfahimbinamin/car-crash-or-collision-prediction-dataset")
        shutil.move(ccpd_path, ccpd_dest)
    else:
        print("CCPD数据集已存在")


def task5():
    import os

    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "openapi.openxlab.org.cn"
    openxlab.login(ak='wyv9qm5w0k9yjrn8evgx', sk='a1m3q0kjgognbarezdknbg74zw69dpvzwy5mpbr2') # 进行登录，输入对应的AK/SK，可在个人中心添加AK/SK

    from openxlab.dataset import info
    info(dataset_repo='OpenDataLab/ApolloScape') #数据集信息查看

    from openxlab.dataset import get
    get(dataset_repo='OpenDataLab/ApolloScape', target_path='data/apolloscapelane') # 数据集下载

    from openxlab.dataset import download
    download(dataset_repo='OpenDataLab/ApolloScape',source_path='/README.md', target_path='data/apolloscapelane') #数据集文件下载

def task6():
    import os

    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "openapi.openxlab.org.cn"
    openxlab.login(ak='wyv9qm5w0k9yjrn8evgx', sk='a1m3q0kjgognbarezdknbg74zw69dpvzwy5mpbr2') # 进行登录，输入对应的AK/SK，可在个人中心添加AK/SK

    from openxlab.dataset import info
    info(dataset_repo='OpenDataLab/CalTech_Lanes')

    from openxlab.dataset import get
    get(dataset_repo='OpenDataLab/CalTech_Lanes', target_path='data/caltech_lanes') # 数据集下载

    from openxlab.dataset import download
    download(dataset_repo='OpenDataLab/CalTech_Lanes',source_path='/README.md', target_path='data/caltech_lanes') #数据集文件下载



if __name__ == "__main__":
    prepare_dir()
    task6 ()
