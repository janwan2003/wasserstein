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