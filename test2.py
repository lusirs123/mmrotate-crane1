import os
import glob

def analyze_prediction_files(base_dir):
    print(f"{'='*60}")
    print(f" 开始统计预测文件分布: {base_dir}")
    print(f"{'='*60}")
    
    thresholds = ['preds_005', 'preds_00001']
    
    for thr in thresholds:
        pred_dir = os.path.join(base_dir, thr, 'Task1_grab')
        if not os.path.exists(pred_dir):
            print(f" [警告] 目录不存在: {pred_dir}")
            continue
            
        files = glob.glob(os.path.join(pred_dir, '*.txt'))
        total_files = len(files)
        
        if total_files == 0:
            print(f" [{thr}] 目录下无 .txt 文件。")
            continue
            
        non_empty_files = [f for f in files if os.path.getsize(f) > 0]
        empty_files = total_files - len(non_empty_files)
        
        # 提取第一个非空文件的体积进行采样
        sample_size = f"{os.path.getsize(non_empty_files[0]) / 1024:.2f} kB" if non_empty_files else "N/A"
        
        print(f" [{thr}]")
        print(f"  -> 总帧数     : {total_files}")
        print(f"  -> 包含预测框 : {len(non_empty_files)} 帧")
        print(f"  -> 0 Bytes帧  : {empty_files} 帧 (全盘漏检)")
        print(f"  -> 样本单帧体积: {sample_size}\n")

if __name__ == '__main__':
    analyze_prediction_files('work_dirs/crane_baseline')