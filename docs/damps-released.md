# 官方 DAMPS 算子与适配来源

`src/mmrecsys/nn/damps_released.py` 逐字节来自
[Wmhwxl/KDD2026_DAMPS](https://github.com/Wmhwxl/KDD2026_DAMPS/blob/13dc1b8d1a9ee53fe0eca12ab2e4c8669f291cd0/src/models/damps.py)，
提交 `13dc1b8d1a9ee53fe0eca12ab2e4c8669f291cd0`。
文件SHA256：`f0183b7bf9fa238f3ffea1af6a86f5fba6d193f6c81a961fc9338e8a1062a9a9`。
原文件头部import与注释保留；主方式的 `nn/damps.py` 没有修改。

PGL/LGMRec各自文件中的 `_init_official`、`_encode_official` 保留实验适配路径，
通过 `damps_enabled=true` 与 `damps_mode=official` 启用。官方仓库的 `Damps` 导入名
与实际类 `DAMPS` 不一致，此处直接导入实际类。相位先验 `avg_R` 改为buffer，
数值不变、不可训练，以支持 `.to()` 及checkpoint。初始化采用CPU后迁移，
COMP5331训练器、采样、损失接口及评估仍保留，未复制完整作者框架或超参数搜索。

PGL外部模态特征和投影在forward中计算，但发布DAMPS只使用传入宽度，
因此其无损失梯度；DAMPS内部特征/投影/频域参数参与训练。
LGMRec只在构造时调用DAMPS，其输出用于冻结模态特征表和后续局部/全局分支；
DAMPS本身不获得推荐损失梯度。测试显式保留这些行为，不擅自修复。
