# 独立框架的官方源码

`ultralytics/` 来自 https://github.com/ultralytics/ultralytics ，固定提交 `d1ebfd72646d5a5b9050d1d0557d7b7fb0ab31f4`，版本8.4.202。保留运行包、安装元数据及AGPL许可证，逐字节复制，来源和SHA见 `configs/yolo26/upstream_manifest.json`。YOLO是不同框架，因此保留独立运行包。

GauCho基于MMRotate，所需三个官方模块已直接接入根目录 `mmrotate/models/`，复用现有训练入口。重复的GauCho框架副本已删除。GauCho配置、许可证和来源记录位于 `configs/gaucho/`，三个模块只做必要的导入/注册名适配，可反向还原并核对官方SHA。

正式实验配置均在根目录 `configs/`，共同评价适配位于 `crane_project/`。
