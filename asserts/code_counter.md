代码字数统计：
[Linux shell统计代码行数及统计代码字数命令](https://www.cnblogs.com/bugutian/p/13116456.html)

```python
find . -name "*.py" | xargs awk 'END{print NR}'
find . -name "*.py" | xargs wc -m  # 字数统计
```