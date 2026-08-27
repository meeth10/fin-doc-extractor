# Recovery note

The Qwen rebuild was moved off the default branch after real-report testing exposed a serious fact-model correctness issue: debt facts had null period alignment and the deterministic resolver returned a component (commercial paper) as total debt.

The pre-rebuild `main` commit is preserved at `5b18532db1cc32e96fc5621657994536b800c160`.

The post-rebuild history remains reachable from commit `d33407dbd1b498acec391463166b3a4d49ded4d1` while the rebuild is reworked separately.
