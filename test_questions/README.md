# Question-answer pairs for LLM testing

This folder contains question-answer pairs creating by **gpt-4o-mini** with source documents being news articles scraped from [Yahoo Finance](https://finance.yahoo.com/).

## Folder structure

Here is the folder structure:

```md
test-questions/
├─ news_source/
├─ questions/
├─ create_test.ipynb
├─ links.json
├─ news.json
├─ README.md
```

- `news_source/`: Contains the source documents in JSON format. Each document was used to create corresponding question-answer pairs (e.g., `news_1.json` was used to create `questions_1.json`).
- `questions/`: Contains the question-answer pairs in JSON format. The `final_questions.json` file contains 200 question-answer pairs selected from the first 200 questions from the combined question-answer pairs.
- `create_test.ipynb`: Jupyter notebook used to create question-answer pairs.
- `links.json`: Contains the links to the news articles used to create the question-answer pairs.
- `news.json`: is used by the `create_test.ipynb` notebook as the input to create question-answer pairs.

## How the question-answer pairs were created

The question-answer pairs were created by using the `gpt-4o-mini` model to generate questions from the source documents. The answers were then extracted from the source documents and paired with the generated questions.

Details implementation can be found in the `create_test.ipynb` notebook.
