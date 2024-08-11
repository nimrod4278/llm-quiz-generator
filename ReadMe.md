
# LLM-Powered Quiz Question Generator

## Table of Contents

-   [Introduction](#introduction)
-   [Features](#features)
-   [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Installation](#installation)
-   [Usage](#usage)
-   [LLM Selection](#llm-selection)
-   [Prompting Techniques](#prompting-techniques)
-   [Future Enhancements](#future-enhancements)
-   [Contributing](#contributing)
-   [License](#license)

## Introduction

The LLM-Powered Quiz Question Generator is a web-based tool that leverages large language models (LLMs) and advanced prompt engineering techniques to generate quiz questions on general knowledge topics. This project draws inspiration from the STOM Web Research agent implementation in the Langgraph documentation.

Key aspects of the project:

-   Expands given topics to related subjects
-   Creates a dialogue between "Wiki Editor" and "Expert" personas
-   Generates diverse and engaging quiz questions based on the expanded context

For more information on the STOM implementation, check out:

-   [Langgraph STOM Tutorial](https://langchain-ai.github.io/langgraph/tutorials/storm/storm)
-   [YouTube Guide](https://youtu.be/1uUORSZwTz4?si=XnFLRTlsUfZJkI45)

## Features

1.  **Generate Quiz**: Create quizzes on user-specified topics with adjustable difficulty levels
2.  **Validate Quiz**: Test your knowledge by answering the generated questions (Bonus feature)
3.  **Learn More**: Access additional information about each question (Bonus feature)

## Getting Started

### Prerequisites

-   Python 3.10 or higher
-   OpenAI API key

### Installation

1.  Clone the repository: 
`git clone https://github.com/nimrod4278/llm-quiz-gen.git`

3.  Navigate to the project directory: `cd llm-quiz-gen`
4.  Create a virtual environment: `python3 -m venv venv`
5.  Activate the virtual environment:
    -   On Unix or MacOS: `source venv/bin/activate`
    -   On Windows: `venv\Scripts\activate`
6.  Install the required packages: `pip install -r requirements.txt`
7.  Create a .env file in the project root and add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

1.  Ensure your virtual environment is activated.
2.  Start the Streamlit app: `streamlit run app.py`
3.  Open your web browser and navigate to [http://localhost:8501/](http://localhost:8501/)
4.  Enter a topic, select the desired difficulty level, and generate your quiz!

## LLM Selection

### Open-source vs. Proprietary

This project uses a proprietary model for ease of use and stability. However, the architecture allows for easy integration of open-source alternatives.

### OpenAI vs. Anthropic

OpenAI's models were chosen due to prior experience and familiarity. The project structure allows for easy model switching by modifying only the `graph/llm.py` file.

### Model Choice

**GPT-3.5-turbo** was selected for its balance of performance, cost-effectiveness, and speed. No multi-modal functionality was required for this project.

## Prompting Techniques

1.  **Few-shot Learning**: Providing examples in prompts to guide the model's output.
2.  **Role-playing**: Utilizing various "personas" (e.g., Quiz Writer, Wiki Writer, Editor) to create a diverse and enriched quiz creation process.
3.  **Structured Output**: Employing Langchain Pydantic classes to ensure consistent and parseable model outputs.

## Future Enhancements

1.  Implement stronger models for improved results
2.  Further incorporate STORM paper methodologies
3.  Utilize asynchronous calls for better performance
4.  Integrate multiple models for answer and knowledge diversity
5.  Add loading animations for improved user experience
6.  Implement error handling and input validation
7.  Add unit tests and integration tests
8.  Create a more robust and scalable backend architecture
9.  Improve accessibility and responsive design

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.