from pyexpat import model
import torch
from tqdm import tqdm
import time
from Edge_ANNoy import Edge_ANNoy
import numpy as np
import h5py

if __name__ == "__main__":
    h5_files = [
        r"data.hdf5",
    ]
    # 批量处理每个h5文件
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as f:
            data = f['train'][:]
            print("train 数据集形状:", data.shape[0])
            print("train 数据集形状:", f['train'].shape)
        try:
            # 生成对应pt文件路径
            pt_path = h5_path.replace('.hdf5', '.pt')
            model_name_ = h5_path.split('\\')[-1].split('.')[0]
            print(model_name_)
            # 数据转换流程
            with h5py.File(h5_path, 'r') as f:
                dataset = torch.from_numpy(f['train'][:]).to(torch.double)
                torch.save(dataset, pt_path)
                print(f"成功转换 {h5_path} → {pt_path}")
            dataset = torch.load(pt_path, weights_only=False)
            torch.cuda.empty_cache()
            tensor_num = len(dataset)
            num_push_down_values = [100]  # 参数范围
            # 批量实验
            for idx, num_push_down in enumerate(num_push_down_values):
                progress_bar = tqdm(total=100, desc='Building Edge-ANNoy', unit='vec')  # 新增进度条
                print(f"\n开始实验: num_push_down={num_push_down}")
                with open(f'./{model_name_}/node{num_push_down}.txt', 'w') as f1, \
                    open(f'./{model_name_}/id{num_push_down}.txt', 'w') as f2:
                    pass
                start_time = time.time()
                for i in range(100):
                    t = Edge_ANNoy(num_push_down, dataset, np.arange(tensor_num))
                    # 带时间统计的构建
                    t.build()
                    t.store_model(t.root, node_file=f'./{model_name_}/node{num_push_down}.txt', 
                            id_file=f'./{model_name_}/id{num_push_down}.txt')
                    progress_bar.update(1)  
                    
                elapsed = (time.time() - start_time) * 1000
                progress_bar.close()  # 构建完成后关闭进度条

                print(f"建 {num_push_down} 棵树完成 | 耗时: {elapsed:.2f}ms")

        except Exception as e:
            print(f"处理 {h5_path} 失败: {str(e)}")
            continue