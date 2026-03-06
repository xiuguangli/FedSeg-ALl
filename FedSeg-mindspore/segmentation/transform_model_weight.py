import pprint
from myseg.bisenet_utils import set_model_bisenetv2
from options import args_parser
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from options import args_parser
import torch
import numpy as np
args = args_parser()
  

def transform_model():
    """
    加载PyTorch的.pth权重文件, 转换为MindSpore的权重, 并保存为.ckpt文件。
    """
    # --- 1. 创建一个空的MindSpore模型实例 ---
    try:
        from myseg.bisenetv2 import BiSeNetV2 as BiSeNetV2_MindSpore
        ms_model = BiSeNetV2_MindSpore(args=args, n_classes=args.num_classes)
        ms_model.set_train(False)
        print("Successfully created MindSpore model instance.")
    except ImportError:
        print("Error: MindSpore model class 'BiSeNetV2_MindSpore' not found.")
        return
    
    # --- 2. 加载PyTorch的.pth权重文件 ---
    torch_model_path = '/home/lxg/work/Miao/FedSeg/segmentation/myseg/backbone_v2.pth'
    torch_param_dict = torch.load(torch_model_path, map_location='cpu', weights_only=True)

    # --- 存储MindSpore加载前的所有参数值 ---
    params_before_load = {}
    for name, param in ms_model.parameters_dict().items():
        params_before_load[name] = param.asnumpy().copy()

    # --- 3. 权重转换逻辑 ---
    print("\nStarting weight transformation...")
    # 统计转换成功的层数
    count = 0
    for ms_param_name, ms_param in ms_model.parameters_dict().items():
        name_parts = ms_param_name.split('.', 1)
        if len(name_parts) < 2: continue
        top_module_name, sub_param_name = name_parts
        if top_module_name not in torch_param_dict: continue
        torch_sub_dict = torch_param_dict[top_module_name]
        torch_param_name = sub_param_name.replace('moving_mean', 'running_mean').replace('moving_variance', 'running_var')
        torch_param_name = torch_param_name.replace('.gamma', '.weight').replace('.beta', '.bias')

        if torch_param_name in torch_sub_dict:
            # 打印转换前后对比
            print(f"[{count}] {ms_param_name} <--> {torch_param_name}")
            print(f"[{count}] before {ms_param.flatten()[:3]}")
            torch_value = torch_sub_dict[torch_param_name].cpu().numpy()
            ms_value = ms.Tensor(torch_value)
            if ms_param.shape == ms_value.shape:
                ms_param.set_data(ms_value)
            print(f"[{count}] after {ms_param.flatten()[:3]}\n")
            count += 1
            
    print("Weight transformation finished.")

    # --- 4. 保存.ckpt文件 ---
    ms_ckpt_path = '/home/lxg/work/Miao/FedSeg/segmentation/myseg/backbone_v2.ckpt'
    ms.save_checkpoint(ms_model, ms_ckpt_path)
    print(f"\nMindSpore checkpoint successfully saved to: {ms_ckpt_path}")
    
    # --- 5. 验证加载并统计未变化的层 ---
    print("\n--- Verification and Statistics ---")
    # 源pth权重文件中共有266项参数权重
    unchanged_layers = []
    changed_layers_count = 0

    for name, param_after in ms_model.parameters_dict().items():
        if name in params_before_load:
            if np.array_equal(params_before_load[name], param_after.asnumpy()):
                unchanged_layers.append(name)
            else:
                changed_layers_count += 1

    print(f"\nTotal parameters checked: {len(params_before_load)}")
    print(f"Number of parameters that CHANGED: {changed_layers_count}")
    print(f"Number of parameters that remained UNCHANGED: {len(unchanged_layers)}")
    
    # --- 6. 详细分析未变化的层 ---
    if unchanged_layers:
        print("\n--- Detailed analysis of UNCHANGED parameters ---")
        for layer_name in unchanged_layers:
            print(f"\nParameter: {layer_name}")

            # 打印MindSpore的初始值
            ms_initial_value = params_before_load[layer_name].flatten()[:3]
            print(f"  - MindSpore Initial Value: {ms_initial_value}")

            # 查找并打印对应的PyTorch原始值
            try:
                name_parts = layer_name.split('.', 1)
                top_module_name, sub_param_name = name_parts
                
                torch_param_name = sub_param_name.replace('moving_mean', 'running_mean').replace('moving_variance', 'running_var')
                if 'bn' in torch_param_name or 'downsample.1' in torch_param_name:
                    torch_param_name = torch_param_name.replace('gamma', 'weight').replace('beta', 'bias')
                
                torch_original_value = torch_param_dict[top_module_name][torch_param_name].cpu().numpy().flatten()[:3]
                print(f"  - PyTorch Original Value:  {torch_original_value}")
            except KeyError:
                print("  - PyTorch Original Value:  Not found with current mapping rules.")
    
    ckpt = ms.load_checkpoint(ms_ckpt_path)
    for key in list(ckpt.keys()):
        if key.split('.')[0] not in torch_param_dict.keys():
            ckpt.pop(key)
    ms.save_checkpoint(ckpt, ms_ckpt_path)
    ckpt1 = ms.load_checkpoint(ms_ckpt_path)
    count1 = 1
    for key in ckpt1.keys():
        print(f"{count1}: {key}")
        count1 += 1

if __name__ == "__main__":
    transform_model()