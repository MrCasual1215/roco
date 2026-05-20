#### 修改地方
- 本仓库仅对prompt内容 合作机制进行了修改
- 没有对场景设置、阈值判断以及任务的技能空间进行修改
- 另外，为了实现不同task使用不同的模式（plan or dialog），本仓库对 evaluator.py 进行了接口的修改。
- 为了方便评测, 增加了llm的模型选择


#### 评测方式
1. 跑分用
``` bash
xvfb-run   python evaluator.py --llm_source qwen3.5:27b
```
2. 最后验收使用
``` bash
xvfb-run   python evaluator.py --llm_source llama3.3:70b
```