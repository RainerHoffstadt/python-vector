import matplotlib.pyplot as plt
import os
dirname = '/home/oem/Chessman-image-dataset/Chess/train'
dir_chess_folders = os.listdir(dirname)
dir_chess_paths = [os.path.join(dirname, path) for path in dir_chess_folders]
print(dir_chess_paths)

chess_dic = {}
for path in dir_chess_paths:
    head, tail = os.path.split(path)
    chess_dic[tail] = len(os.listdir(path))
label_list = ["{}: {}".format(key, chess_dic[key]) for key in chess_dic]
plt.figure(figsize=(8, 8))
plt.bar(range(len(chess_dic)), list(chess_dic.values()), color="green")
plt.xticks(range(len(chess_dic)), list(label_list))
plt.show()