import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import gc # 垃圾回收

class RealVehicleDataset(Dataset):
    def __init__(self, csv_file_path, window_size=30, sample_stride=10, is_train=True):
        """
        :param sample_stride: 采样步长。
        """
        self.window = window_size
        
        print(f"Dataset: 正在使用【分块流式读取】加载数据 (采样步长={sample_stride})...")
        
        # === 核心修改：分块读取，防止内存爆炸 ===
        chunk_size = 500000  # 每次只读 50万行到内存
        chunks_list = []
        total_rows = 0
        
        # 使用 chunksize 迭代读取
        # usecols: 只读取需要的列，进一步省内存
        required_cols = ['Current', 'Voltage', 'Temperature', 'SOC', 'SOH']
        
        try:
            reader = pd.read_csv(csv_file_path, usecols=required_cols, dtype='float32', chunksize=chunk_size)
            
            for i, chunk in enumerate(reader):
                # 1. 立即降采样：在这一小块数据中，每隔 stride 取一行
                sampled_chunk = chunk.iloc[::sample_stride]
                
                # 2. 存入列表
                chunks_list.append(sampled_chunk)
                total_rows += len(sampled_chunk)
                
                if (i+1) % 10 == 0:
                    print(f"    -> 已处理 {(i+1)*chunk_size/10000:.0f} 万行原始数据...")
                    gc.collect() # 强制清理内存
                    
        except Exception as e:
            print(f"读取出错: {e}")
            raise e

        print(f"Dataset: 分块读取完毕，正在合并... (保留数据量: {total_rows} 行)")
        
        # 合并所有小切片
        df = pd.concat(chunks_list, ignore_index=True)
        
        # 主动释放切片列表内存
        del chunks_list
        gc.collect()

        # === 归一化 (逻辑不变) ===
        # 电流: 假设 -500 ~ 500
        self.curr_min, self.curr_max = -500.0, 500.0
        # 电压: 假设 200 ~ 800
        self.volt_min, self.volt_max = 200.0, 800.0
        # 温度: 假设 -30 ~ 60
        self.temp_min, self.temp_max = -30.0, 60.0
        
        # Clip 防止越界
        df['Current'] = ((df['Current'] - self.curr_min) / (self.curr_max - self.curr_min)).clip(-1, 1)
        df['Voltage'] = ((df['Voltage'] - self.volt_min) / (self.volt_max - self.volt_min)).clip(0, 1)
        df['Temperature'] = ((df['Temperature'] - self.temp_min) / (self.temp_max - self.temp_min)).clip(0, 1)
        
        # 确保 SOC/SOH 是 0-1
        if df['SOC'].max() > 1.5: 
            df['SOC'] = df['SOC'] / 100.0
            
        print("Dataset: 数据预处理完成，转换为 Tensor...")
        
        # 转换为 Tensor
        self.x_data = torch.tensor(df[['Current', 'Voltage', 'Temperature']].values, dtype=torch.float32)
        self.soh_data = torch.tensor(df['SOH'].values, dtype=torch.float32)
        self.y_data = torch.tensor(df['SOC'].values, dtype=torch.float32)
        
        # 再次释放 dataframe
        del df
        gc.collect()

    def __len__(self):
        return len(self.x_data) - self.window

    def __getitem__(self, index):
        x_realtime = self.x_data[index : index + self.window]
        x_soh = self.soh_data[index + self.window - 1].unsqueeze(0)
        y_target = self.y_data[index + self.window - 1].unsqueeze(0)
        return x_realtime, x_soh, y_target