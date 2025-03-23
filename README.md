# How to Install Poetry

Poetry is a tool for dependency management and packaging in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you.

## Installation Steps

1. **Install Poetry**:
    Open your terminal and run the following command:
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. **Configure Environment Variables**:
    Add Poetry to your PATH by adding the following line to your shell configuration file (`~/.bashrc`, `~/.zshrc`, etc.):
    ```sh
    export PATH="$HOME/.local/bin:$PATH"
    ```
    Then, reload your shell configuration:
    ```sh
    source ~/.bashrc  # or source ~/.zshrc
    ```

3. **Verify Installation**:
    To ensure Poetry is installed correctly, run:
    ```sh
    poetry --version
    ```

## Troubleshooting

If you encounter the error `The current project could not be installed: No file/folder found for package wasserstein`, you need to:

1. Make sure you have a proper Python package structure with `__init__.py` files
2. Verify that your pyproject.toml has the correct package configuration
3. If you don't need to install the current project as a package, use `poetry install --no-root`
4. If you're only using Poetry for dependency management, set `package-mode = false` in pyproject.toml

## Additional Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [GitHub Repository](https://github.com/python-poetry/poetry)

Now you are ready to use Poetry to manage your Python projects!