```markdown
# mempalace Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches you the core development patterns and workflows used in the `mempalace` Python codebase. You'll learn the project's coding conventions, how to extend or refactor backend logic, and how to write and organize tests. This knowledge will help you contribute effectively to mempalace, especially when adding new backends or improving backend-neutral functionality.

## Coding Conventions

### File Naming

- Use **snake_case** for all file and module names.
  - Example: `embedding_utils.py`, `test_backend_routing.py`

### Import Style

- Use **relative imports** within the package.
  - Example:
    ```python
    from .embedding import EmbeddingManager
    from .backends.chroma import ChromaBackend
    ```

### Export Style

- Use **named exports** (explicitly define what is exported from a module).
  - Example (`embedding.py`):
    ```python
    class EmbeddingManager:
        ...

    __all__ = ["EmbeddingManager"]
    ```

### Commit Messages

- Use **conventional commit** prefixes: `feat`, `fix`, `refactor`
- Keep commit messages concise (~76 characters on average).
  - Example: `feat: add Chroma backend support for embedding storage`

## Workflows

### Backend Extension Workflow

**Trigger:** When you want to add a new backend or refactor backend-neutral logic for embeddings/storage.

**Command:** `/add-backend`

**Step-by-step:**

1. **Implement or Refactor Backend-Neutral Logic**
   - Edit or extend shared modules, such as `mempalace/embedding.py`.
   - Example:
     ```python
     # mempalace/embedding.py
     class EmbeddingManager:
         def embed(self, data):
             ...
     ```

2. **Update or Create Backend-Specific Modules**
   - Add or modify files like `mempalace/backends/ruvector_postgres.py` or `mempalace/backends/chroma.py`.
   - Ensure these modules use the new shared logic or provide necessary shims.
   - Example:
     ```python
     # mempalace/backends/chroma.py
     from ..embedding import EmbeddingManager

     class ChromaBackend(EmbeddingManager):
         ...
     ```

3. **Update Server Routing or Integration Points**
   - Edit files such as `mempalace/mcp_server.py` to recognize and route to the new backend or abstraction.
   - Example:
     ```python
     # mempalace/mcp_server.py
     from .backends.chroma import ChromaBackend

     def get_backend(name):
         if name == "chroma":
             return ChromaBackend()
         ...
     ```

4. **Write or Update Tests**
   - Add or modify tests to cover the new backend and routing logic.
   - Relevant test files: `tests/test_backend_routing.py`, `tests/test_ruvector_postgres_migrate.py`
   - Example:
     ```python
     # tests/test_backend_routing.py
     def test_chroma_backend_routing():
         backend = get_backend("chroma")
         assert isinstance(backend, ChromaBackend)
     ```

**Files Involved:**
- `mempalace/embedding.py`
- `mempalace/backends/chroma.py`
- `mempalace/backends/ruvector_postgres.py`
- `mempalace/mcp_server.py`
- `tests/test_backend_routing.py`
- `tests/test_ruvector_postgres_migrate.py`

**Frequency:** ~1-2x/month

## Testing Patterns

- **Test Framework:** Not explicitly detected; use standard Python testing tools (e.g., `pytest`).
- **Test File Naming:** Files match the pattern `*.test.*` (e.g., `test_backend_routing.py`).
- **Test Placement:** Tests are located in the `tests/` directory.
- **Test Example:**
  ```python
  # tests/test_backend_routing.py
  def test_backend_selection():
      backend = get_backend("ruvector_postgres")
      assert backend is not None
  ```

## Commands

| Command        | Purpose                                                         |
|----------------|-----------------------------------------------------------------|
| /add-backend   | Start the backend extension workflow (add or refactor backend). |

```