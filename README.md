# Eq2Tex 

Hey there! Welcome to Eq2Tex. This is a fun student project that turns pictures of math equations into LaTeX code. We know typing out complex math formulas can be a huge pain, so we built this little app to do the heavy lifting for you!

## How it Works

It's pretty simple: You spin up the app, upload an image of a math equation, and our tool gives you back the text format (LaTeX) you can paste right into your homework or documents. 

## The Brains Behind the App (Our Models)

We experimented with a couple of different AI approaches for this project:

1. **The Ready-to-Go Model (`ai.py`)**: For the main app that actually works out of the box, we hooked up a pre-trained model called `LatexOCR` (from the `pix2tex` library). It does a surprisingly good job out of the gate!
2. **Our Custom Model (`models/` folder)**: We also wanted to see if we could build our own system from scratch. So, we spent some time creating our own custom Neural Network (a CRNN using PyTorch). We fed it lots of math images and their translations from the [im2latex-100k dataset on Kaggle](https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k) to see if it could learn. While it's still an experiment, it was a huge learning experience on how these things work under the hood!

## How to Run It

If you want to try it out on your own computer:

1. Open your terminal and run the server:
   ```bash
   python main.py
   ```
2. Open your web browser and go to `http://127.0.0.1:5000/`.
3. Upload a picture of an equation and see the magic!

*(the app automatically deletes them after 10 minutes from the `upload` folder so your computer doesn't run out of space).*

Thanks for checking out our project!

