# task23：统一面板与dw4i合训

统一面板29项：白天七卡L1/3/4/6/7/8/9及原版L10（每项128局）、L7–L10各3套初始牌留出（每套64局）、夜间n9 L11/12/13/14/16/17/18/19及原版L20（每项128局），每模型2944局。四个模型均走纯学生，同一清单、同一评估种子前缀、同一程序，逐项报告与三组任务平均胜率。L10留出仍是指定初始牌的传送带测试，独立于原版空手L10。

slot v34与identity各5局跨区实机冒烟、L20空槽/首卡合法性已通过，见[证据](task23-panel-smoke.json)；两旧模型完整基线正在顺序评估。

Mac首批[导入证据](task23-mac-import.json)：15包/4241档案，HF固定revision；4b/5b各640，d9p2m仅81，其他已上传d7p包160。主数据只增不改，重复导入160局验证320文件字节与mtime不变。冻结前刷新上传内容，不宣称尚未上传的包已入训。

训练等待queue101完整3轮与GPU空闲、内存充足。将冻结同一逐档案SHA清单，按task19原identity+row_pool配方从头依次训seed1/2、12epoch，不选择最好seed/epoch；每来源实际保留行数另行计数，默认训练行为不变。最终四列逐项表、训练统计、HF模型交付与夜间/白天/留出变化在完整核验后补入。

## v34完整基线（identity与新模型仍待完整评估）

| 任务 | v34 | identity_dw3_s1 | dw4i s1 | dw4i s2 |
|---|---:|---:|---:|---:|
| day_l1_eval_d7 | 128/128 | 待测 | 待测 | 待测 |
| day_l3_eval_d7 | 128/128 | 待测 | 待测 | 待测 |
| day_l4_eval_d7 | 128/128 | 待测 | 待测 | 待测 |
| day_l6_eval_d7 | 128/128 | 待测 | 待测 | 待测 |
| day_l7_eval_d7 | 123/128 | 待测 | 待测 | 待测 |
| day_l8_eval_d7 | 125/128 | 待测 | 待测 | 待测 |
| day_l9_eval_d7 | 116/128 | 待测 | 待测 | 待测 |
| day_l10_orig_eval | 119/128 | 待测 | 待测 | 待测 |
| day_l7_eval_d7p_holdout1 | 36/64 | 待测 | 待测 | 待测 |
| day_l7_eval_d7p_holdout2 | 42/64 | 待测 | 待测 | 待测 |
| day_l7_eval_d7p_holdout3 | 0/64 | 待测 | 待测 | 待测 |
| day_l8_eval_d7p_holdout1 | 0/64 | 待测 | 待测 | 待测 |
| day_l8_eval_d7p_holdout2 | 24/64 | 待测 | 待测 | 待测 |
| day_l8_eval_d7p_holdout3 | 27/64 | 待测 | 待测 | 待测 |
| day_l9_eval_d7p_holdout1 | 2/64 | 待测 | 待测 | 待测 |
| day_l9_eval_d7p_holdout2 | 3/64 | 待测 | 待测 | 待测 |
| day_l9_eval_d7p_holdout3 | 11/64 | 待测 | 待测 | 待测 |
| day_l10_eval_d7p_holdout1 | 41/64 | 待测 | 待测 | 待测 |
| day_l10_eval_d7p_holdout2 | 50/64 | 待测 | 待测 | 待测 |
| day_l10_eval_d7p_holdout3 | 64/64 | 待测 | 待测 | 待测 |
| night_l11_eval_n9 | 0/128 | 待测 | 待测 | 待测 |
| night_l12_eval_n9 | 0/128 | 待测 | 待测 | 待测 |
| night_l13_eval_n9 | 0/128 | 待测 | 待测 | 待测 |
| night_l14_eval_n9 | 0/128 | 待测 | 待测 | 待测 |
| night_l16_eval_n9 | 0/128 | 待测 | 待测 | 待测 |
| night_l17_eval_n9 | 0/128 | 待测 | 待测 | 待测 |
| night_l18_eval_n9 | 0/128 | 待测 | 待测 | 待测 |
| night_l19_eval_n9 | 0/128 | 待测 | 待测 | 待测 |
| night_l20_eval_n9_orig | 0/128 | 待测 | 待测 | 待测 |

区域任务平均：白天97.17%（995/1024）、换卡组39.06%（300/768）、夜间0%（0/1152）。29项2944局已独立重读种子/配置/胜局/冻结模型-程序-清单身份，[基线核验](task23-baselines.json)。identity已顺序接续，尚无dw4i训练或效果结论。
