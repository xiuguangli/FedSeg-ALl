
import matplotlib.pyplot as plt
import numpy as np  

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",type=str,help="file containing accuracy curve data",required=True)
    args = parser.parse_args()
    return args

def main():
    """
    给定一个文件，提取其中的曲线数据并绘制准确率曲线。
    Results after 159 global rounds of training:
    |---- Global Test Accuracy: 33.17%
    |---- Global Test IoU: 12.99%
    """
    args = get_args()
    file = args.file
    rounds = []
    accuracies = []
    ious = []
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Results after"):
                round_info = line.split(" ")[2]
                current_rounds = int(round_info)
                rounds.append(current_rounds)
            if "Global Test Accuracy" in line:
                parts = line.split(":")
                accuracy_str = parts[1].strip().replace("%","")
                accuracy = float(accuracy_str)
                accuracies.append(accuracy)
            elif "Global Test IoU" in line:
                parts = line.split(":")
                iou_str = parts[1].strip().replace("%","")
                iou = float(iou_str)
                ious.append(iou)
    
    # 绘制准确率曲线
    plt.figure(figsize=(10,5))
    plt.plot(rounds, accuracies, label='Global Test Accuracy', color='blue', marker='o')
    plt.plot(rounds, ious, label='Global Test IoU', color='orange', marker='o')
    plt.xlabel('Global Rounds')
    plt.ylabel('Percentage (%)')
    plt.title('Accuracy and IoU Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_iou_curve.png')
    plt.show()

if __name__ == "__main__":
    main()
                
    
    
    
    
    