# 项目代码注释规范

## Docstring 的返回结构

- 使用中文说明参数、返回值和关键行为，字段名及程序标识保留原样。
- `Returns` 必须按实际返回对象的嵌套层级编写，使用缩进体现父子关系；不要把不同层级的字段平铺在同一列表中。
- 成功与失败的返回结构不同时，分别描述，明确字段所在位置及哪些字段仅在特定分支存在。
- 字典说明其字段；列表说明每个元素的结构；动态键字典说明键和值分别代表什么。
- 字段说明应简洁、准确，与代码一致，不把预期行为写成已经实现的行为。
- 添加真实示例时，从项目当前存在的运行记录中核实，并注明来源。长字段可保留开头和结尾，用省略号标记展示缩写，不混淆示例与实际返回值。
- 仅要求补充注释或 docstring 时，不改变执行逻辑。

示例：

```python
"""执行资料检索。

Returns:
    dict: 根据任务是否成功，返回以下两种结构之一。

    成功结果：
        success: True。
        task_type: "deep_research"。
        data: 本次检索收集的数据。
            agent_response: 已收集来源数量的文字说明。
            results: 来源摘要列表，每项包含：
                url: 来源地址。
                title: 来源标题。
                domain: 来源域名。
                snippet: 正文摘要。
            structured_data: 供后续分析使用的结构化数据。
                sources: 来源摘要列表。
                extracted_content: 通过相关性审查的正文列表。
                search_attempts: 搜索尝试列表，每项包含 query 和 error。
                total_sources: 相关来源数量。
                total_content_length: 已采纳正文的总字符数。

    失败结果：
        success: False。
        task_type: "deep_research"。
        error: 异常信息。
        search_attempts: 失败前已记录的搜索尝试列表。
"""
```

此示例用于说明层级写法。其他函数必须按自身实际返回结构编写，不能直接照搬字段。

<!-- postman-cli:installation:start -->
Postman CLI is installed on this device, and can be used for API Engineering work.
<!-- postman-cli:installation:end -->
