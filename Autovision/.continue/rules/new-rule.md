---
rules:
  - name: Smart Tasks
    description: Use big models for deep tasks
    match:
      - "explain"
      - "refactor"
      - "optimize"
      - "architecture"
      - "debug"
    use:
      model: Qwen 14B Smart

  - name: Fast Tasks
    description: Use 7B for quick edits
    match:
      - "fix"
      - "convert"
      - "add function"
      - "rewrite"
    use:
      model: Qwen 7B Fast
---

Your rule content
