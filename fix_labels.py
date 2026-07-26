import os

# 标注文件存放路径
label_dirs = ['./datasets/train/labels', './datasets/val/labels']

for label_dir in label_dirs:
    if not os.path.exists(label_dir):
        continue
    
    count_corn = 0
    count_wheat = 0
    
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue
            
        filepath = os.path.join(label_dir, filename)
        
        # 根据文件名判断类别：带有 maize 或 corn 设为 0 (玉米)，否则设为 1 (小麦)
        target_cls = '0' if ('maize' in filename.lower() or 'corn' in filename.lower()) else '1'
        
        new_lines = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    parts[0] = target_cls  # 强行修正 ID
                    new_lines.append(" ".join(parts))
        
        # 覆盖写入
        with open(filepath, 'w') as f:
            f.write("\n".join(new_lines) + "\n")
            
        if target_cls == '0':
            count_corn += 1
        else:
            count_wheat += 1
            
    print(f"[{label_dir}] 处理完成：玉米 txt={count_corn} | 小麦 txt={count_wheat}")

print("\n🎉 标签类别编号全部修正成功！")