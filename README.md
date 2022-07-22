# 车辆轨迹追踪_GUI界面
##1.环境配置
###1.1 安装PaddlePaddle
`GPU版本：
python -m pip install paddlepaddle-gpu==2.2.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
`
###1.2 安装其他依赖
`cd Vehicle tracking`  
`pip install -r requirements.txt`

##2.文件下载
###2.1推理模型下载
模型下载链接：https://pan.baidu.com/s/1QEjgGN0cG7bouPVaj-uo3w 提取码：1234  
下载之后解压， ppyoloe_crn_l_80e_visdrone_largesize是识别模型文件，deepsort_pplcnet_vehicle是车辆分类模型文件
###2.2推理视频下载
链接：https://pan.baidu.com/s/1KyCTGbtO6_Pyd-MgzajzPQ 提取码：1234

##3.程序运行
运行./python/mian.py，加载视频、识别模型、分类模型和配置文件（配置文件是./python/tracker_config）  

选择ROI区域，最后轨迹线只出现在ROI区域内，否则整个界面都有显示（最后结果在./python/output中查看）
