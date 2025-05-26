# Contributing Guidelines

Thank you for your interest in contributing to the Martin-Habitat-Pathfinder project! We appreciate your efforts to help us identify and evaluate potential habitat locations on Mars using cutting-edge technologies like reinforcement learning and large language models.

By contributing to this project, you agree to abide by our [Code of Conduct](#code-of-conduct).

## Table of Contents

* [Code of Conduct](#code-of-conduct)
* [How to Contribute](#how-to-contribute)
    * [Reporting Bugs](#reporting-bugs)
    * [Suggesting Enhancements](#suggesting-enhancements)
    * [Your First Contribution](#your-first-contribution)
* [Development Setup](#development-setup)
* [Coding Guidelines](#coding-guidelines)
* [Pull Request Guidelines](#pull-request-guidelines)
* [License](#license)

## Code of Conduct

We are committed to fostering an open and welcoming environment. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) (please create this file if it doesn't exist yet) to ensure a positive experience for everyone.

## How to Contribute

We welcome contributions of all kinds, including bug fixes, new features, documentation improvements, and more.

### Reporting Bugs

If you find a bug, please help us by reporting it. Before opening a new issue, please check existing issues to see if the bug has already been reported.

To report a bug:
1.  Go to the [Issues](https://github.com/Estiakk/Martin-Habitat-Pathfinder/issues) page.
2.  Click on "New issue".
3.  Provide a clear and concise description of the bug.
4.  Include steps to reproduce the bug.
5.  Mention your operating system and any relevant software versions.
6.  If possible, include screenshots or error messages.

### Suggesting Enhancements

Do you have an idea for a new feature or an improvement? We'd love to hear it!

To suggest an enhancement:
1.  Go to the [Issues](https://github.com/Estiakk/Martin-Habitat-Pathfinder/issues) page.
2.  Click on "New issue".
3.  Choose a descriptive title for your suggestion.
4.  Clearly describe the proposed feature or enhancement.
5.  Explain why you think it would be beneficial to the project.
6.  If applicable, provide examples or mockups.

### Your First Contribution

If you're new to contributing to open source, here's a basic workflow:

1.  **Fork** the repository by clicking the "Fork" button on the top right of the repository page.
2.  **Clone** your forked repository to your local machine:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Martin-Habitat-Pathfinder.git](https://github.com/YOUR_USERNAME/Martin-Habitat-Pathfinder.git)
    cd Martin-Habitat-Pathfinder
    ```
3.  **Add the upstream remote** to fetch changes from the original repository:
    ```bash
    git remote add upstream [https://github.com/Estiakk/Martin-Habitat-Pathfinder.git](https://github.com/Estiakk/Martin-Habitat-Pathfinder.git)
    ```
4.  **Create a new branch** for your feature or bug fix:
    ```bash
    git checkout -b feature/your-new-feature-name # for features
    git checkout -b bugfix/your-bug-fix-name     # for bug fixes
    ```
5.  **Make your changes** and commit them with clear, concise commit messages.
    ```bash
    git add .
    git commit -m "feat: add descriptive feature name" # or "fix: fix bug description"
    ```
6.  **Push your branch** to your forked repository:
    ```bash
    git push origin your-branch-name
    ```
7.  **Open a Pull Request** from your forked repository to the `main` branch of the original `Estiakk/Martin-Habitat-Pathfinder` repository.

## Development Setup

To get the project running locally, please refer to the `README.md` file for detailed instructions on setting up the data processing pipeline, simulation environment, and LLM integration.

* Ensure you have all necessary dependencies installed (e.g., Python, specific libraries mentioned in `requirements.txt` or similar).
* Follow the instructions for data acquisition and preparation.
* Set up the simulation environment.
* Configure the LLM integration (e.g., Ollama).

## Coding Guidelines

To maintain code quality and consistency, please adhere to the following guidelines:

* **Python Specific**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style.
* **Docstrings**: Add clear and concise docstrings for all functions, classes, and modules.
* **Comments**: Use comments to explain complex logic, but prefer self-documenting code.
* **Testing**: If you're adding new features or fixing complex bugs, consider adding unit tests.
* **Modularity**: Keep functions and modules focused on a single responsibility.
* **Naming Conventions**: Use descriptive variable and function names.

## Pull Request Guidelines

When submitting a pull request, please ensure the following:

* **One feature/bug fix per PR**: Keep your PRs focused on a single change.
* **Descriptive Title**: Provide a clear and concise title for your PR.
* **Detailed Description**:
    * Explain the purpose of your changes.
    * Reference any related issues (e.g., "Closes #123").
    * Describe how you tested your changes.
    * Include screenshots or GIFs if the changes involve the UI.
* **Code Review**: Be prepared for constructive feedback and be willing to iterate on your changes.
* **Pass CI/CD**: Ensure your changes pass any automated tests or checks.

## License

By contributing to Martin-Habitat-Pathfinder, you agree that your contributions will be licensed under the MIT License, as specified in the project's `LICENSE` file.
