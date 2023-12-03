# Task Completion

## Used Technology Stack:

- Python 3.10.12

## Essential Libraries Used:

- `open-cv`
- `imutils`
- `pytesseract`

## Project Specific:

I utilized an open-source dataset called `frozen_east_text_detection` to enhance text detection accuracy.

### Important Notes:

1. The video source is hardcoded to the front camera (`src=0`).
2. Confidence is hardcoded to the default value of `0.5`. Adjusting this value impacts detection precision and speed.
3. The project for text detection already existed. Instead of starting from scratch, I incorporated essential features tailored to our project.
4. The Tesseract library requires an executable whose location must be specified. Currently, it's hardcoded to the path `/usr/bin/tesseract`. During execution, this path should be updated if a different location is used.

> Since I was applying for the **Front-end position**, I found this task slightly outside my current skill set. However, my Python knowledge and curiosity enabled me to contribute to the project.
