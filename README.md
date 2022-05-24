# bookedge
- 比赛地址： https://aistudio.baidu.com/aistudio/competition/detail/207/0/introduction
- 取得成绩： 0.96235 third place
- ![image](https://user-images.githubusercontent.com/28752718/170086936-2d9132ba-3dd6-422c-abca-c76d92610370.png)


### data_process
1. 比赛的数据放在 preprocess_data/bookedge/ 下
2. 运行 cd preprocess_data && python3 create_train.py 构造 dataset
3. 将构造好的数据集放入路径 PaddleSeg/data/bookedge

### model train
1. 模型训练配置文件为 PaddleSeg/configs/quick_start/segformer_bookedge_512x512.yml
2. 训练环境为 8 卡 2080ti
3. 训练脚本为 python3 -m paddle.distributed.launch train.py  --config configs/quick_start/segformer_bookedge_512x512.yml   --do_eval   --use_vdl  --save_interval 1700 --save_dir B5_640

### model predict
1. 模型 best.pt 复制到 predict/model 下
2. 预测脚本为 python3 predict.py [待预测图片地址] [预测结果存放地址]
